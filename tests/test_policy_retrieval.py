"""Regression checks for missed entities, duplicate evidence, and lost conditions."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_core.documents import Document
from transformers import AutoTokenizer
from policy_retrieval import build_policy_chunks, retrieve_policy_documents


class AdversarialDenseStore:
    """Rank the input in reverse order to expose reliance on dense entity matches."""
    def __init__(self, documents):
        self.documents = documents
        self.index = SimpleNamespace(ntotal=len(documents))
        self.index_to_docstore_id = dict(enumerate(range(len(documents))))
        self.docstore = SimpleNamespace(search=lambda key: documents[key])

    def similarity_search_with_score(self, query, k):
        return [(doc, float(i)) for i, doc in enumerate(reversed(self.documents))][:k]


class PolicyRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            "HarishMaths/Hotel-Policy-Embedding",
            cache_dir=str(ROOT / ".cache/huggingface/hub"), local_files_only=True,
        )

    def corpus(self):
        raw = [Document(page_content="""## Property profiles
| Property | Check-in / out | Parking | Pets | Breakfast |
| - | - | - | - | - |
| ASH-AAA Central | 3:00 PM / 11:00 AM | $20 per night | $80 per stay | Included |
| ASH-BBB Garden | 4:00 PM / 12:00 PM | $30 per night | $125 per stay | $15 per person |

## Conduct
Quiet hours are 10:00 PM to 7:00 AM. Staff document noise complaints and escalate repeated disturbances to the manager.
""", metadata={"source": "policy.pdf"})]
        return build_policy_chunks(raw, self.tokenizer)

    def test_exact_property_survives_wrong_dense_ranking(self):
        store = AdversarialDenseStore(self.corpus())
        hits = retrieve_policy_documents(store, "What are the pet fees at ASH-AAA Central?", 4)
        self.assertTrue(hits)
        self.assertIn("$80 per stay", hits[0].page_content)
        self.assertNotIn("$125", "\n".join(doc.page_content for doc in hits))
        self.assertIn("Pets:", hits[0].page_content)

    def test_loud_query_finds_quiet_hours_without_other_property_rows(self):
        store = AdversarialDenseStore(self.corpus())
        hits = retrieve_policy_documents(store, "The neighbor is loud at midnight. What should staff do?", 4)
        self.assertIn("10:00 PM to 7:00 AM", hits[0].page_content)
        self.assertTrue(all(not hit.metadata["property_codes"] for hit in hits))

    def test_duplicate_profile_keeps_both_sources(self):
        text = "## Profiles\n| Property | Pets |\n| - | - |\n| ASH-AAA Central | $80 per stay |"
        chunks = build_policy_chunks([
            Document(page_content=text, metadata={"source": "a.pdf"}),
            Document(page_content=text.replace("Profiles", "Reminders"), metadata={"source": "b.pdf"}),
        ], self.tokenizer)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["sources"], ["a.pdf", "b.pdf"])

    def test_full_conditions_restored_while_embedding_inputs_fit(self):
        body = " ".join(f"condition{number}" for number in range(400))
        chunks = build_policy_chunks([Document(page_content="## Conditions\n" + body,
            metadata={"source": "conditions.pdf"})], self.tokenizer)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(self.tokenizer.encode(doc.page_content, truncation=False)) <= 128
                            for doc in chunks))
        hits = retrieve_policy_documents(AdversarialDenseStore(chunks), "condition177", 4)
        self.assertEqual(len(hits), 1)
        self.assertIn("condition0", hits[0].page_content)
        self.assertIn("condition399", hits[0].page_content)

    def test_unrelated_query_has_no_evidence(self):
        hits = retrieve_policy_documents(AdversarialDenseStore(self.corpus()), "quantum interstellar spectroscopy", 4)
        self.assertEqual(hits, [])

    def test_flexible_request_excludes_other_rate_plan_rows(self):
        text = """## Rate plans
| Rate plan | Cancel or change |
| - | - |
| Flexible | Cancel by 6:00 PM the day before arrival. |
| Advance Purchase | Non-refundable except an approved service recovery. |
"""
        chunks = build_policy_chunks([Document(page_content=text, metadata={"source": "rates.pdf"})], self.tokenizer)
        hits = retrieve_policy_documents(AdversarialDenseStore(chunks),
            "When can I cancel my flexible reservation?", 4)
        self.assertTrue(hits)
        self.assertIn("6:00 PM", hits[0].page_content)
        self.assertNotIn("Advance Purchase", "\n".join(hit.page_content for hit in hits))

    def test_malformed_table_does_not_silently_drop_values(self):
        raw = Document(page_content="| Property | Pets |\n| - | - |\n| ASH-AAA | $80 | Extra |",
                       metadata={"source": "bad.pdf"})
        with self.assertRaises(ValueError):
            build_policy_chunks([raw], self.tokenizer)

    def test_refund_keeps_the_separate_manager_exception(self):
        text = """## Cancellation
Late cancellation of a flexible rate can retain one night plus tax.

## Exceptions
A manager may approve an exception for a verified reservation system error.

## Other fees
Parking has a nightly charge and breakfast may be paid separately.
"""
        chunks = build_policy_chunks([Document(page_content=text, metadata={"source": "refunds.pdf"})], self.tokenizer)
        hits = retrieve_policy_documents(AdversarialDenseStore(chunks),
            "Can I get a refund after a late cancellation of my flexible rate?", 2)
        context = "\n".join(hit.page_content for hit in hits)
        self.assertIn("one night plus tax", context)
        self.assertIn("verified reservation system error", context)

    def test_comparison_can_retrieve_both_explicit_rate_plans(self):
        text = """## Rate plans
| Rate plan | Cancel or change |
| - | - |
| Flexible | Cancel by 6:00 PM the day before arrival. |
| Advance Purchase | Non-refundable except an approved service recovery. |
"""
        chunks = build_policy_chunks([Document(page_content=text, metadata={"source": "rates.pdf"})], self.tokenizer)
        hits = retrieve_policy_documents(AdversarialDenseStore(chunks),
            "Compare cancellation terms for flexible and Advance Purchase rates.", 2)
        context = "\n".join(hit.page_content for hit in hits)
        self.assertIn("Rate plan: Flexible", context)
        self.assertIn("Rate plan: Advance Purchase", context)


if __name__ == "__main__":
    unittest.main()
