"""Replay scored evaluation retrieval locally and freeze its verified configuration.

No generation/judge API calls are made. The held-out test dataset is not read.
"""

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
GENERATION = runpy.run_path(str(ROOT / "scripts/run-generation.py"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import csv
    import numpy as np
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    index_dir = ROOT / ".cache/notebook/faiss_index"
    evaluation_dir = ROOT / ".cache/notebook/evaluation"
    output = ROOT / ".cache/notebook/retrieval_improvement"
    index_summary = json.loads((index_dir / "summary.json").read_text())
    for name, expected in index_summary["index_file_hashes"].items():
        assert digest(index_dir / name) == expected
    chunk_summary = json.loads((ROOT / ".cache/notebook/chunks/summary.json").read_text())
    for source, expected in chunk_summary["source_hashes"].items():
        assert digest(ROOT / source) == expected
    embeddings = HuggingFaceEmbeddings(model_name=index_summary["embedding_model"],
        model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    store = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    namespace = {}
    notebook = json.loads((ROOT / "Full_Code_Notebook.ipynb").read_text(encoding="utf-8"))
    exec("".join(notebook["cells"][72]["source"]), namespace)
    scored_cases = json.loads((evaluation_dir / "test_cases.json").read_text(encoding="utf-8"))
    input_rows = list(csv.DictReader((ROOT / "Data/evaluation.csv").open(encoding="utf-8-sig", newline="")))
    assert len(scored_cases) == len(input_rows) == 10
    for case, row in zip(scored_cases, input_rows, strict=True):
        assert case["input"] == row["question"]
        actual = namespace["retrieve"](case["input"], k=4, vectorstore=store)
        expected = case.get("retrieval_context", case.get("retrievalContext"))
        assert actual == expected, f"Retrieval changed for an evaluated question: {case['input']}"
    vectors = store.index.reconstruct_n(0, store.index.ntotal)
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert store.index.ntotal == chunk_summary["total_chunks"]
    source_numbers = (47, 54, 56, 66, 70, 73, 85, 90, 93, 105, 106, 110, 114, 118, 122, 124)
    sources = {str(number): "".join(notebook["cells"][number - 1]["source"]) for number in source_numbers}
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    fingerprint = hashlib.sha256(json.dumps({
        "dataset": digest(ROOT / "Data/evaluation.csv"),
        "index": index_summary["index_file_hashes"],
        "retrieval_module": digest(ROOT / "policy_retrieval.py"),
        "endpoint_hash": hashlib.sha256(config["OPENAI_API_BASE"].strip().encode()).hexdigest(),
        "sources": sources,
    }, sort_keys=True).encode()).hexdigest()
    manifest_path = evaluation_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("original_scoring_fingerprint", manifest["fingerprint"])
    manifest["fingerprint"] = fingerprint
    manifest["retrieval_replay_verified"] = True
    manifest["replay_verified_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    frozen = {
        "frozen_at_utc": manifest["replay_verified_at_utc"],
        "evaluation_fingerprint": fingerprint,
        "retrieval_module_sha256": digest(ROOT / "policy_retrieval.py"),
        "index_build_module_sha256": index_summary["retrieval_module_sha256"],
        "notebook_source_hashes": {number: hashlib.sha256(source.encode()).hexdigest() for number, source in sources.items()},
        "index_hashes": index_summary["index_file_hashes"],
        "source_hashes": chunk_summary["source_hashes"],
        "embedding_model": index_summary["embedding_model"],
        "vectors": store.index.ntotal, "dimensions": store.index.d,
        "model": "gpt-4o-mini", "judge": "gpt-4o",
        "parameters": {"k": 4, "temperature": 0.0, "top_p": 1.0, "max_tokens": 512},
        "evaluation_retrieval_replay_matches": 10,
        "test_set_read_or_executed": False,
        "new_paid_api_calls": 0,
    }
    (output / "frozen_config.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print("PASS current retrieval exactly matches all 10 scored evaluation contexts.")
    print("PASS source/index integrity, unit-normalized vectors, and frozen configuration; no paid API calls or held-out data reads.")


if __name__ == "__main__":
    sys.exit(main())
