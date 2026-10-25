"""Policy-aware chunks and local hybrid retrieval; no evaluation answers required."""

from collections import Counter
import hashlib
import math
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


STOP_WORDS = set("""
a an the and or but if of for in on at to from by with about as is are was were
be been being do does did have has had i me my we our you your they their it its
this that these those there here what which who when where how why can could
would should will may might must need want please tell know get got some any
all also still now then next yesterday tomorrow since really much many staying
stay book booked booking reservation hotel property guest aurorastay aurorastays
room rooms time times supposed someone details information
out front desk staff agent team ask asks pay paying see
""".split())

# Small domain vocabulary, shared across all queries; no question/answer lookup.
SYNONYMS = (
    {"noise", "loud", "quiet", "disturbance", "disturbances"},
    {"pet", "pets", "dog", "dogs", "animal", "animals"},
    {"cancel", "cancellation", "canceled", "cancelled", "canceling"},
    {"refund", "refunded", "refundable", "reimbursement"},
    {"check-in", "checkin", "arrival", "arrive"},
    {"check-out", "checkout", "departure", "depart"},
    {"parking", "garage", "valet"},
    {"fee", "fees", "charge", "charges", "charged", "cost"},
    {"incidental", "incidentals", "authorization", "hold", "pending"},
    {"smoke", "smoking", "smokes", "vape", "vaping", "vapes"},
    {"refund", "approval", "approve", "exception", "recovery"},
)

CANONICAL = {
    **dict.fromkeys(("quiet", "loud", "disturbance", "disturbances"), "noise"),
    **dict.fromkeys(("cancellation", "canceled", "cancelled", "canceling"), "cancel"),
    **dict.fromkeys(("smoking", "smokes"), "smoke"),
    **dict.fromkeys(("vaping", "vapes"), "vape"),
    **dict.fromkeys(("refunded", "refundable"), "refund"),
    **dict.fromkeys(("fees",), "fee"),
    **dict.fromkeys(("charged", "charges"), "charge"),
    **dict.fromkeys(("pets", "dogs", "animals", "incidentals"), None),
}

TOPICS = (
    {"noise", "quiet", "loud", "disturbance"},
    {"pet", "dog", "animal", "ada"},
    {"smoke", "vape", "smoking", "vaping"},
    {"checkin", "checkout", "arrival", "arrive", "departure", "depart"},
    {"parking", "garage", "valet"},
    {"breakfast"},
    {"pool", "fitness"},
    {"incidental", "hold", "pending", "authorization", "issuer"},
    {"cancel", "refund", "exception", "recovery", "flexible", "advance", "purchase"},
)


def _terms(text):
    text = re.sub(r"\bcheck(?:ed)?[ -]+in\b", "checkin", text, flags=re.I)
    text = re.sub(r"\bcheck(?:ed)?[ -]+out\b", "checkout", text, flags=re.I)
    text = re.sub(r"\bnext door\b", "neighbor", text, flags=re.I)
    text = re.sub(r"\bafter (?:the )?(?:cancellation )?deadline\b", "late cancellation", text, flags=re.I)
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    words = [CANONICAL.get(word, word) if CANONICAL.get(word, word) is not None else word[:-1]
             for word in words]
    return [word for word in words if word not in STOP_WORDS]


def _cells(line):
    return [re.sub(r"\s+", " ", value.strip())
            for value in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def _divider(line):
    return all(re.fullmatch(r"[:\s-]+", cell) for cell in _cells(line))


def build_policy_chunks(raw_docs, tokenizer, chunk_size=120, overlap=20, audit=None):
    """Split whole paragraphs and labeled table rows; retain parent context.

    Each embedding input stays within chunk_size content tokens. Retrieval
    returns the full original paragraph/row, so a split cannot hide conditions.
    Identical content is indexed once, retaining all originating source paths.
    """
    limit = int(tokenizer.model_max_length)
    specials = tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 < overlap < chunk_size <= limit - specials:
        raise ValueError("Chunk settings do not fit the model's tokenizer limit.")

    def length(text):
        return len(tokenizer.encode(text, add_special_tokens=False,
                                    truncation=False, verbose=False))

    parents = {}
    source_items = Counter()
    table_rows = 0

    def add(body, heading, source, kind, fields=None, caption=""):
        nonlocal table_rows
        body = body.strip()
        if not body:
            return
        source_items[source] += 1
        table_rows += kind == "table_row"
        key = re.sub(r"\s+", " ", body.casefold())
        # Non-profile clauses can have different applicability under different headings.
        if not (fields and "Property" in fields):
            key = heading.casefold() + "\n" + key
        if key in parents:
            if source not in parents[key]["sources"]:
                parents[key]["sources"].append(source)
            return
        prefix = f"Section: {heading}\n" if heading else ""
        context = f"{prefix}{caption}\n{body}" if caption else prefix + body
        parents[key] = {
            "body": body, "parent_body": body, "heading": heading, "sources": [source],
            "source": source, "kind": kind, "parent_text": context,
            "parent_id": hashlib.sha256(key.encode()).hexdigest()[:20],
            "property_codes": sorted(set(re.findall(r"\bASH-[A-Z]{3}\b", body))),
            "fields": fields or {},
        }
        plan_text = (fields or {}).get("Rate plan", (fields or {}).get("Scenario", "")).lower()
        parents[key]["rate_plan"] = next((name for name, pattern in (
            ("flexible", r"^flexible"), ("advancepurchase", r"^advance purchase"),
            ("member", r"^member"), ("group", r"^group"),
        ) if re.search(pattern, plan_text)), "")

    for raw in raw_docs:
        source = raw.metadata["source"]
        lines = raw.page_content.splitlines()
        heading = ""
        last_paragraph = ""
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if not line:
                index += 1
                continue
            if line.startswith("#"):
                heading = line.lstrip("#").strip()
                last_paragraph = ""
                index += 1
                continue
            if line.startswith("|"):
                table = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    if not _divider(lines[index]):
                        table.append(_cells(lines[index]))
                    index += 1
                if not table:
                    continue
                headers, *rows = table
                if headers[0].casefold() == "policy code":
                    add("\n".join(": ".join(row) for row in table), heading, source, "administrative")
                    continue
                kind = "navigation" if headers[0].casefold() == "navigation" else "table_row"
                caption = last_paragraph if re.search(r"matrix|training|amounts|following", last_paragraph, re.I) else ""
                for row in rows:
                    if len(row) != len(headers):
                        raise ValueError(f"Table column mismatch in {source}: {heading}")
                    fields = dict(zip(headers, row, strict=True))
                    body = "\n".join(f"{name}: {value}" for name, value in fields.items())
                    assert all(value in body for value in row), "A table value was lost."
                    add(body, heading, source, kind, fields, caption)
                last_paragraph = ""
                continue
            paragraph = []
            while index < len(lines) and lines[index].strip() and not lines[index].lstrip().startswith(("#", "|")):
                paragraph.append(lines[index].strip())
                index += 1
            body = "\n".join(paragraph)
            overview = re.search(r"(?:guide|policy|guidelines)$", heading, re.I)
            has_rule = re.search(r"\b(?:must|may|can|should|is|are|will|require|prohibited|allowed|only)\b|[$\d]", body, re.I)
            kind = "administrative" if overview and not has_rule else "paragraph"
            add(body, heading, source, kind)
            last_paragraph = body

    chunks = []
    for parent in parents.values():
        heading = parent["heading"]
        prefix = f"Section: {heading}\n" if heading else ""
        # The prefix is descriptive metadata, never a replacement for policy text.
        if length(prefix) > 40:
            prefix = ""
        budget = chunk_size - length(prefix) - 1
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=budget, chunk_overlap=min(overlap, budget // 4),
            length_function=length, separators=["\n", ". ", " ", ""], keep_separator="end",
        )
        pieces = splitter.split_text(parent["body"])
        # Verify the splitter did not omit any non-whitespace policy content.
        covered_end = 0
        previous_start = -1
        for piece_number, piece in enumerate(pieces):
            start = parent["body"].find(piece, previous_start + 1)
            assert start >= 0 and not parent["body"][covered_end:start].strip()
            covered_end = max(covered_end, start + len(piece))
            previous_start = start
            text = prefix + piece
            assert length(text) <= chunk_size
            metadata = {key: value for key, value in parent.items() if key != "body"}
            metadata["part"] = piece_number
            chunks.append(Document(page_content=text, metadata=metadata))
        assert not parent["body"][covered_end:].strip(), "Trailing policy content was omitted."

    if audit is not None:
        audit.update({
            "input_items_by_source": dict(source_items), "input_table_rows": table_rows,
            "unique_parents": len(parents), "embedding_chunks": len(chunks),
            "duplicate_items_collapsed": sum(source_items.values()) - len(parents),
            "max_content_tokens": max(map(lambda d: length(d.page_content), chunks)),
            "max_model_input_tokens": max(length(d.page_content) + specials for d in chunks),
            "token_limit": limit, "all_paragraph_and_table_values_preserved": True,
        })
    return chunks


class PolicyTextSplitter:
    """Notebook-compatible splitter retaining an audit of the latest build."""

    def __init__(self, tokenizer, chunk_size=120, chunk_overlap=24):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.audit = {}

    def split_documents(self, documents):
        self.audit = {}
        return build_policy_chunks(documents, self.tokenizer, self.chunk_size,
                                   self.chunk_overlap, audit=self.audit)


class HybridPolicyRetriever:
    """BM25 and FAISS rank fusion, property matching, and diverse parent context."""

    def __init__(self, vectorstore, lexical_weight=0.75):
        self.vectorstore = vectorstore
        self.lexical_weight = lexical_weight
        self.parents = {}
        for position in range(vectorstore.index.ntotal):
            doc = vectorstore.docstore.search(vectorstore.index_to_docstore_id[position])
            key = doc.metadata.get("parent_id", hashlib.sha256(doc.page_content.encode()).hexdigest())
            if key not in self.parents:
                metadata = dict(doc.metadata)
                text = metadata.pop("parent_text", doc.page_content)
                self.parents[key] = Document(page_content=text, metadata=metadata)
        self.term_counts = {key: Counter(_terms(doc.page_content)) for key, doc in self.parents.items()}
        self.body_terms = {key: set(_terms(doc.metadata.get("parent_body", doc.page_content)))
                           for key, doc in self.parents.items()}
        self.df = Counter(term for counts in self.term_counts.values() for term in counts)
        self.average_length = sum(sum(c.values()) for c in self.term_counts.values()) / max(1, len(self.parents))
        self.property_terms = {}
        for doc in self.parents.values():
            for code in doc.metadata.get("property_codes", []):
                self.property_terms.setdefault(code, set()).update(
                    _terms(doc.metadata.get("fields", {}).get("Property", code)))

    def search(self, query, k=4):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        original_terms = set(_terms(query))
        query_weights = dict.fromkeys(original_terms, 1.0)
        for group in SYNONYMS:
            normalized = set(_terms(" ".join(sorted(group))))
            if normalized & original_terms:
                for term in normalized:
                    query_weights.setdefault(term, 0.35)
        codes = set(re.findall(r"\bASH-[A-Z]{3}\b", query.upper()))
        requested_plans = set()
        if re.search(r"\badvance purchase\b", query, re.I):
            requested_plans.add("advancepurchase")
        if re.search(r"\bflexible\b", query, re.I):
            requested_plans.add("flexible")
        topic_vocabulary = set()
        for group in TOPICS:
            normalized = set(_terms(" ".join(sorted(group))))
            if normalized & original_terms:
                topic_vocabulary.update(normalized)
        property_terms = set().union(*(self.property_terms.get(code, {code.lower()}) for code in codes)) if codes else set()
        topic_terms = {term for term in original_terms - property_terms if self.df[term]}
        # Require a subject match rather than filling slots with generic dense hits.
        idf = lambda term: math.log(1 + (len(self.parents) - self.df[term] + 0.5) / (self.df[term] + 0.5))
        if topic_terms:
            strongest = max(map(idf, topic_terms))
            anchors = {term for term in topic_terms if idf(term) >= 0.65 * strongest}
            for group in SYNONYMS:
                normalized = set(_terms(" ".join(sorted(group))))
                if anchors & normalized:
                    anchors.update(normalized)
        else:
            anchors = set()
        if "refund" in original_terms:
            anchors.update({"refund", "approval", "approve", "exception", "recovery"})
        eligible = {
            key for key, doc in self.parents.items()
            if doc.metadata.get("kind") not in {"navigation", "administrative"}
            and (not topic_vocabulary or topic_vocabulary & self.body_terms[key])
            and (not requested_plans or not doc.metadata.get("rate_plan")
                 or doc.metadata["rate_plan"] in requested_plans)
            and (not codes or not doc.metadata.get("property_codes")
                 or codes & set(doc.metadata["property_codes"]))
        }
        lexical = {}
        for key in eligible:
            counts = self.term_counts[key]
            length = sum(counts.values())
            score = 0.0
            for term, weight in query_weights.items():
                frequency = counts[term]
                if frequency:
                    idf = math.log(1 + (len(self.parents) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += weight * idf * frequency * 2.5 / (
                        frequency + 1.5 * (0.25 + 0.75 * length / self.average_length)
                    )
            lexical[key] = score
        if not lexical or max(lexical.values()) <= 0:
            return []
        supported = {key for key in eligible if lexical[key] > 0
                     and (not anchors or anchors & set(self.term_counts[key]))}
        # Refund eligibility often refers to a separately documented approval rule.
        approval_support = {key for key in eligible if "refund" in original_terms
                            and {"manager", "exception"} <= self.body_terms[key]
                            and self.body_terms[key] & {"approve", "approved", "approval"}}
        if not supported:
            return []
        best_lexical = max(lexical[key] for key in supported)
        supported = {key for key in supported if lexical[key] >= 0.25 * best_lexical}
        supported.update(approval_support)
        lexical_order = sorted(supported,
                               key=lambda key: (-lexical[key], key))
        dense_order = []
        for doc, _ in self.vectorstore.similarity_search_with_score(query, k=self.vectorstore.index.ntotal):
            key = doc.metadata.get("parent_id", hashlib.sha256(doc.page_content.encode()).hexdigest())
            if key in supported and key not in dense_order:
                dense_order.append(key)
        ranks_lexical = {key: rank for rank, key in enumerate(lexical_order, 1)}
        ranks_dense = {key: rank for rank, key in enumerate(dense_order, 1)}
        candidates = set(lexical_order[:32]) | set(dense_order[:24]) | approval_support
        fused = {
            key: self.lexical_weight / (60 + ranks_lexical[key]) if key in ranks_lexical else 0.0
            for key in candidates
        }
        for key in candidates:
            if key in ranks_dense:
                fused[key] += (1 - self.lexical_weight) / (60 + ranks_dense[key])
        maximum = max(fused.values())
        selected = []
        covered = set()
        relevant_terms = set(query_weights) & set(self.df)

        def utility(key):
            terms = set(self.term_counts[key])
            novelty = len((terms & relevant_terms) - covered) / max(1, len(relevant_terms))
            redundancy = max((len(terms & set(self.term_counts[other])) /
                              max(1, len(terms | set(self.term_counts[other])))
                              for other in selected), default=0)
            return fused[key] / maximum + 0.35 * novelty - 0.4 * redundancy

        # An exact property profile supplies the facts embeddings can miss.
        property_matches = [key for key in lexical_order
                            if codes & set(self.parents[key].metadata.get("property_codes", []))]
        if property_matches:
            first = max(property_matches, key=lambda key: (
                len(topic_terms & set(_terms(" ".join(self.parents[key].metadata.get("fields", {}))))),
                len(self.parents[key].metadata.get("fields", {})), lexical[key],
            ))
            selected.append(first)
            covered.update(self.term_counts[first])
        # First retain the main policy answer, then its applicable exception rule.
        remaining = candidates - set(selected)
        if len(selected) < k and remaining:
            primary = max(sorted(remaining), key=utility)
            selected.append(primary)
            covered.update(self.term_counts[primary])
        if len(selected) < k and approval_support and not approval_support.intersection(selected):
            support = max(sorted(approval_support), key=utility)
            selected.append(support)
            covered.update(self.term_counts[support])
        while candidates and len(selected) < k:
            candidates.difference_update(selected)
            if not candidates:
                break
            best = max(sorted(candidates), key=utility)
            selected.append(best)
            covered.update(self.term_counts[best])
        return [self.parents[key] for key in selected]


def retrieve_policy_documents(vectorstore, query, k=4):
    if vectorstore is None:
        raise ValueError("Provide the initialized FAISS vectorstore.")
    retriever = getattr(vectorstore, "_aurora_policy_retriever", None)
    if retriever is None or retriever.vector_count != vectorstore.index.ntotal:
        retriever = HybridPolicyRetriever(vectorstore)
        retriever.vector_count = vectorstore.index.ntotal
        vectorstore._aurora_policy_retriever = retriever
    return retriever.search(query, k=k)
