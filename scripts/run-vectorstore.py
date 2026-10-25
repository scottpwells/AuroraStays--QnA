"""Load the notebook embedding model, build FAISS, and verify local retrieval."""

import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
for name, value in {
    "JUPYTER_CONFIG_DIR": str(ROOT / ".jupyter" / "config"),
    "JUPYTER_DATA_DIR": str(ROOT / ".jupyter" / "data"),
    "JUPYTER_RUNTIME_DIR": str(ROOT / ".jupyter" / "runtime"),
    "IPYTHONDIR": str(ROOT / ".jupyter" / "ipython"),
    "HF_HOME": str(ROOT / ".cache" / "huggingface"),
    "HF_HUB_OFFLINE": "0",
    "TRANSFORMERS_OFFLINE": "0",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
    "DEEPEVAL_UPDATE_WARNING_OPT_OUT": "YES",
    "PYTHONUTF8": "1",
}.items():
    os.environ[name] = value


BOOTSTRAP = r'''
from pathlib import Path as _VectorPath
import hashlib as _vector_hashlib
_chunk_summary = json.loads(_VectorPath(".cache/notebook/chunks/summary.json").read_text(encoding="utf-8"))
assert EMBED_MODEL == _chunk_summary["embedding_model"], "Embedding model changed; recheck chunking first."
for _source, _expected_hash in _chunk_summary["source_hashes"].items():
    assert _vector_hashlib.sha256(_VectorPath(_source).read_bytes()).hexdigest() == _expected_hash, "Source PDF changed; refresh chunks first."
_chunk_rows = [json.loads(line) for line in _VectorPath(".cache/notebook/chunks/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
assert len(_chunk_rows) == _chunk_summary["total_chunks"]
lc_docs = [LCDocument(page_content=row["page_content"], metadata=row["metadata"]) for row in _chunk_rows]
assert all(c.page_content.strip() and c.metadata.get("source") for c in lc_docs)
'''

VALIDATE_MODEL = r'''
import numpy as _vector_np
_sentence_model = embeddings._client
_embedding_tokenizer = _sentence_model.tokenizer
_embedding_sequence_limit = int(_sentence_model.max_seq_length)
_embedding_tokenizer_limit = int(_embedding_tokenizer.model_max_length)
_embedding_input_limit = min(_embedding_sequence_limit, _embedding_tokenizer_limit)
_embedding_input_counts = [
    len(_embedding_tokenizer.encode(c.page_content.replace("\n", " "),
        add_special_tokens=True, truncation=False, verbose=False))
    for c in lc_docs
]
assert max(_embedding_input_counts) <= _embedding_input_limit, "Embedding model would truncate a chunk."
assert str(_sentence_model.device).startswith("cpu"), "Notebook requested CPU inference."
print(json.dumps({"model": EMBED_MODEL, "device": str(_sentence_model.device),
    "sentence_transformer_max_seq_length": _embedding_sequence_limit,
    "tokenizer_model_max_length": _embedding_tokenizer_limit,
    "largest_chunk_input_tokens": max(_embedding_input_counts),
    "embedding_dimensions": int(_sentence_model.get_embedding_dimension())}))
'''

VALIDATE_INDEX = r'''
from collections import Counter as _VectorCounter
_vectors = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)
assert _vectors.shape == (len(lc_docs), vectorstore.index.d)
assert _vector_np.isfinite(_vectors).all(), "Non-finite embedding values."
_vector_norms = _vector_np.linalg.norm(_vectors, axis=1)
assert _vector_np.allclose(_vector_norms, 1.0, atol=1e-5), "Embeddings are not normalized."
assert len(vectorstore.index_to_docstore_id) == len(lc_docs)
for _index, _expected_doc in enumerate(lc_docs):
    _stored_doc = vectorstore.docstore.search(vectorstore.index_to_docstore_id[_index])
    assert _stored_doc.page_content == _expected_doc.page_content
    assert _stored_doc.metadata == _expected_doc.metadata

_index_path = _VectorPath(VECTORSTORE_CACHE_PATH)
_saved_hashes = {name: _vector_hashlib.sha256((_index_path / name).read_bytes()).hexdigest()
    for name in ("index.faiss", "index.pkl")}
# Reload only the files generated above, verifying they have not changed.
for _name, _digest in _saved_hashes.items():
    assert _vector_hashlib.sha256((_index_path / _name).read_bytes()).hexdigest() == _digest
_reloaded_store = FAISS.load_local(
    str(_index_path), embeddings, allow_dangerous_deserialization=True
)
assert _reloaded_store.index.ntotal == vectorstore.index.ntotal
assert _reloaded_store.index.d == vectorstore.index.d
print(f"PASS {vectorstore.index.ntotal} normalized vectors; all source metadata retained; saved index reloads.")
'''

VALIDATE_RETRIEVAL = r'''
from policy_retrieval import retrieve_policy_documents
_query_vector = embeddings.embed_query(query)
assert _vector_np.isfinite(_query_vector).all()
assert _vector_np.isclose(_vector_np.linalg.norm(_query_vector), 1.0, atol=1e-5)
_original_hits = retrieve_policy_documents(vectorstore, query, k=RETRIEVAL_TEST_K)
_reloaded_hits = retrieve_policy_documents(_reloaded_store, query, k=RETRIEVAL_TEST_K)
assert [(d.page_content, d.metadata) for d in _original_hits] == [(d.page_content, d.metadata) for d in _reloaded_hits]
_expected_facts = ("ASH-CHI Loop", "$50 per night", "$150 stay cap")
assert all(text in _original_hits[0].page_content for text in _expected_facts)
assert len({d.metadata["parent_id"] for d in _original_hits}) == len(_original_hits)
_self_test_doc = next(c for c in lc_docs if all(text in c.page_content for text in _expected_facts))
_self_vector = embeddings.embed_query(_self_test_doc.page_content)
_self_hit, _self_score = vectorstore.similarity_search_with_score_by_vector(_self_vector, k=1)[0]
assert _self_hit.page_content == _self_test_doc.page_content
assert _vector_np.isclose(_self_score, 0.0, atol=1e-5)
_summary = {
    "embedding_model": EMBED_MODEL,
    "device": str(_sentence_model.device),
    "sentence_transformer_max_seq_length": _embedding_sequence_limit,
    "tokenizer_model_max_length": _embedding_tokenizer_limit,
    "largest_chunk_input_tokens": max(_embedding_input_counts),
    "vectors": int(vectorstore.index.ntotal),
    "embedding_dimensions": int(vectorstore.index.d),
    "index_type": type(vectorstore.index).__name__,
    "metric": "squared_l2_distance_on_normalized_vectors",
    "retrieval": "BM25 + FAISS weighted rank fusion; property/rate-plan/topic matching; unique parent contexts",
    "normalization_min": float(_vector_norms.min()),
    "normalization_max": float(_vector_norms.max()),
    "source_counts": dict(_VectorCounter(c.metadata["source"] for c in lc_docs)),
    "chunk_file_sha256": _vector_hashlib.sha256(_VectorPath(".cache/notebook/chunks/chunks.jsonl").read_bytes()).hexdigest(),
    "retrieval_module_sha256": _vector_hashlib.sha256(_VectorPath("policy_retrieval.py").read_bytes()).hexdigest(),
    "index_file_hashes": _saved_hashes,
    "index_reload_verified": True,
    "retrieval_roundtrip_verified": True,
    "exact_chunk_self_retrieval_verified": True,
    "test_query": query,
    "sample_retrieval_k": RETRIEVAL_TEST_K,
    "expected_property_and_pet_fees_in_top_2": True,
    "expected_property_and_pet_fees_in_sample_top_k": True,
    "expected_property_chunk_ranks": [i + 1 for i, doc in enumerate(_original_hits)
                                      if all(text in doc.page_content for text in _expected_facts)],
    "retrieved_chunks": [{"rank": i + 1, "source": doc.metadata["source"], "content": doc.page_content}
                         for i, doc in enumerate(_original_hits)],
}
(_index_path / "summary.json").write_text(json.dumps(_summary, indent=2), encoding="utf-8")
print(json.dumps(_summary, indent=2))
'''


def main():
    import nbformat
    from nbclient import NotebookClient

    path = ROOT / "Full_Code_Notebook.ipynb"
    original = json.loads(path.read_text(encoding="utf-8"))
    notebook = nbformat.reads(path.read_text(encoding="utf-8"), as_version=4)
    checkpoints = ROOT / ".cache" / "notebook"
    checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, checkpoints / f"before-vectorstore-execution-{time.time_ns()}.ipynb")
    completed = []

    def save_progress():
        current = json.loads(path.read_text(encoding="utf-8"))
        for index in completed:
            cell = current["cells"][index]
            if cell["source"] != original["cells"][index]["source"]:
                nbformat.write(notebook, checkpoints / "vectorstore-executed.ipynb")
                raise RuntimeError("A target cell changed during execution; outputs saved to checkpoint.")
            for key in ("outputs", "execution_count"):
                cell[key] = notebook.cells[index][key]
            if "execution" in notebook.cells[index].metadata:
                cell.setdefault("metadata", {})["execution"] = dict(notebook.cells[index].metadata.execution)
        nbformat.validate(nbformat.from_dict(current))
        temporary = path.with_suffix(".ipynb.tmp")
        temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    client = NotebookClient(
        notebook, kernel_name="aurorastays-312", timeout=1800, iopub_timeout=180,
        allow_errors=False, resources={"metadata": {"path": str(ROOT)}},
    )

    def run_hidden(source):
        cell = nbformat.v4.new_code_cell(source)
        notebook.cells.append(cell)
        try:
            client.execute_cell(cell, len(notebook.cells) - 1, store_history=False)
            for output in cell.outputs:
                if output.output_type == "stream":
                    print(output.text, end="", flush=True)
        finally:
            notebook.cells.pop()

    try:
        with client.setup_kernel():
            print("Restoring imports, tokenizer, and the validated policy chunks...", flush=True)
            run_hidden(notebook.cells[19].source + "\n" + notebook.cells[46].source + "\n" + BOOTSTRAP)
            for number in (66, 70, 73, 76):
                index = number - 1
                print(f"Running notebook cell {number}...", flush=True)
                started = time.monotonic()
                try:
                    client.execute_cell(notebook.cells[index], index)
                finally:
                    completed.append(index)
                    save_progress()
                print(f"Completed cell {number} in {time.monotonic() - started:.1f}s.", flush=True)
                if number == 66:
                    run_hidden(VALIDATE_MODEL)
                elif number == 70:
                    run_hidden(VALIDATE_INDEX)
                elif number == 76:
                    run_hidden(VALIDATE_RETRIEVAL)
    finally:
        save_progress()
    return 0


if __name__ == "__main__":
    sys.exit(main())
