"""Run the token-aware chunking section using the verified Docling checkpoint."""

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
    "HF_HUB_OFFLINE": "1",
    "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
    "DEEPEVAL_UPDATE_WARNING_OPT_OUT": "YES",
    "PYTHONUTF8": "1",
}.items():
    os.environ[name] = value


BOOTSTRAP = '''
from pathlib import Path as _StatePath
import hashlib as _state_hashlib
from docling_core.types.doc import DoclingDocument as _StateDocument
_state_manifest = json.loads(_StatePath(".cache/notebook/documents/manifest.json").read_text(encoding="utf-8"))
root_path = "Data"
docs = {}
for _record in _state_manifest:
    _source = _StatePath(_record["source"])
    assert _state_hashlib.sha256(_source.read_bytes()).hexdigest() == _record["source_sha256"], "Source PDF changed; reconvert it first."
    docs[_record["source"]] = _StateDocument.model_validate_json(
        _StatePath(_record["document_json"]).read_text(encoding="utf-8")
    )
pdf_paths = list(docs)
'''

CACHE_AND_VALIDATE = r'''
from pathlib import Path as _ChunkPath
from collections import Counter as _ChunkCounter
_chunk_cache = _ChunkPath(".cache/notebook/chunks")
_chunk_cache.mkdir(parents=True, exist_ok=True)
assert splitter.audit["all_paragraph_and_table_values_preserved"]
assert {source for c in lc_docs for source in c.metadata["sources"]} == set(docs)
assert len({(c.metadata["parent_id"], c.metadata["part"]) for c in lc_docs}) == len(lc_docs)
assert max(model_input_token_counts) <= MODEL_MAX_TOKENS
_chunk_rows = [{"page_content": c.page_content, "metadata": c.metadata,
    "content_tokens": content_count, "model_input_tokens": input_count}
    for c, content_count, input_count in zip(lc_docs, chunk_token_counts, model_input_token_counts, strict=True)]
(_chunk_cache / "chunks.jsonl").write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in _chunk_rows) + "\n", encoding="utf-8")
_summary = {
    **splitter.audit,
    "embedding_model": EMBED_MODEL,
    "tokenizer_model_max_length": MODEL_MAX_TOKENS,
    "special_tokens_per_chunk": SPECIAL_TOKENS,
    "chunk_size_content_tokens": CHUNK_SIZE,
    "chunk_overlap_target_tokens": CHUNK_OVERLAP,
    "total_chunks": len(lc_docs),
    "content_tokens_min": min(chunk_token_counts),
    "content_tokens_max": max(chunk_token_counts),
    "model_input_tokens_max": max(model_input_token_counts),
    "source_hashes": {r["source"]: r["source_sha256"] for r in _state_manifest},
    "canonical_source_counts": dict(_ChunkCounter(c.metadata["source"] for c in lc_docs)),
    "retrieval_module_sha256": _state_hashlib.sha256(_ChunkPath("policy_retrieval.py").read_bytes()).hexdigest(),
}
(_chunk_cache / "summary.json").write_text(json.dumps(_summary, indent=2), encoding="utf-8")
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
    shutil.copy2(path, checkpoints / f"before-chunking-execution-{time.time_ns()}.ipynb")
    completed = []

    def save_progress():
        current = json.loads(path.read_text(encoding="utf-8"))
        for index in completed:
            cell = current["cells"][index]
            if cell["source"] != original["cells"][index]["source"]:
                nbformat.write(notebook, checkpoints / "chunking-executed.ipynb")
                raise RuntimeError("A target cell changed during execution; results saved to checkpoint.")
            for key in ("outputs", "execution_count"):
                cell[key] = notebook.cells[index][key]
            if "execution" in notebook.cells[index].metadata:
                cell.setdefault("metadata", {})["execution"] = dict(notebook.cells[index].metadata.execution)
        nbformat.validate(nbformat.from_dict(current))
        temporary = path.with_suffix(".ipynb.tmp")
        temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    client = NotebookClient(
        notebook, kernel_name="aurorastays-312", timeout=600, iopub_timeout=120,
        allow_errors=False, resources={"metadata": {"path": str(ROOT)}},
    )

    def run_hidden(source):
        cell = nbformat.v4.new_code_cell(source)
        notebook.cells.append(cell)
        try:
            client.execute_cell(cell, len(notebook.cells) - 1, store_history=False)
        finally:
            notebook.cells.pop()

    try:
        with client.setup_kernel():
            print("Restoring imports and the verified converted PDF cache...", flush=True)
            run_hidden(notebook.cells[19].source + "\n" + BOOTSTRAP)
            for number in (47, 50, 54, 56, 59, 62):
                index = number - 1
                print(f"Running notebook cell {number}...", flush=True)
                try:
                    client.execute_cell(notebook.cells[index], index)
                finally:
                    completed.append(index)
                    save_progress()
                print(f"Completed cell {number}.", flush=True)
                if number == 56:
                    run_hidden(CACHE_AND_VALIDATE)
                    print("Verified token limits, complete source coverage, metadata, and overlap; cached chunks.", flush=True)
    finally:
        save_progress()
    print((checkpoints / "chunks" / "summary.json").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
