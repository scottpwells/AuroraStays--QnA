"""Build and inspect improved retrieval locally, using evaluation questions only."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
for name, value in {
    "HF_HOME": str(ROOT / ".cache/huggingface"), "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "PYTHONUTF8": "1",
}.items():
    os.environ[name] = value


def main():
    import csv
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from transformers import AutoTokenizer
    from policy_retrieval import build_policy_chunks, retrieve_policy_documents

    output = ROOT / ".cache/notebook/retrieval_improvement"
    output.mkdir(parents=True, exist_ok=True)
    baseline = output / "baseline"
    if not baseline.exists():
        baseline.mkdir()
        for name in ("chunks", "faiss_index", "evaluation"):
            shutil.copytree(ROOT / ".cache/notebook" / name, baseline / name)
        shutil.copy2(ROOT / "Full_Code_Notebook.ipynb", baseline / "Full_Code_Notebook.ipynb")
    manifest = json.loads((ROOT / ".cache/notebook/documents/manifest.json").read_text())
    raw = []
    for record in manifest:
        assert hashlib.sha256((ROOT / record["source"]).read_bytes()).hexdigest() == record["source_sha256"]
        path = ROOT / Path(record["document_json"]).with_suffix(".txt")
        raw.append(Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": record["source"]}))
    model_name = "HarishMaths/Hotel-Policy-Embedding"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    audit = {}
    chunks = build_policy_chunks(raw, tokenizer, audit=audit)
    (output / "chunks.jsonl").write_text("\n".join(json.dumps({
        "page_content": chunk.page_content, "metadata": chunk.metadata,
    }) for chunk in chunks) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"},
                                       encode_kwargs={"normalize_embeddings": True})
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(output / "faiss_index"))
    questions = list(csv.DictReader((ROOT / "Data/evaluation.csv").open(encoding="utf-8-sig", newline="")))
    probes = []
    for number, row in enumerate(questions, 1):
        hits = retrieve_policy_documents(store, row["question"], k=4)
        assert len({doc.metadata["parent_id"] for doc in hits}) == len(hits)
        probes.append({"question": row["question"], "contexts": [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in hits
        ]})
        print(f"\nQuestion {number}: {row['question']}", flush=True)
        for rank, doc in enumerate(hits, 1):
            print(f"  {rank}: {doc.page_content[:200]!r}", flush=True)
    (output / "probes.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")
    audit.update({"embedding_model": model_name, "source_hashes": {
        record["source"]: record["source_sha256"] for record in manifest},
        "index_file_hashes": {name: hashlib.sha256((output / "faiss_index" / name).read_bytes()).hexdigest()
                              for name in ("index.faiss", "index.pkl")},
        "test_set_used": False, "paid_api_calls": 0})
    (output / "summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
