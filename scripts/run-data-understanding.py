"""Execute the notebook's import and Data Understanding cells, then checkpoint."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
for name, value in {
    "JUPYTER_CONFIG_DIR": str(ROOT / ".jupyter" / "config"),
    "JUPYTER_DATA_DIR": str(ROOT / ".jupyter" / "data"),
    "JUPYTER_RUNTIME_DIR": str(ROOT / ".jupyter" / "runtime"),
    "IPYTHONDIR": str(ROOT / ".jupyter" / "ipython"),
    "HF_HOME": str(ROOT / ".cache" / "huggingface"),
    "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
    "DEEPEVAL_UPDATE_WARNING_OPT_OUT": "YES",
    "PYTHONUTF8": "1",
}.items():
    os.environ[name] = value


def main():
    import nbformat
    from nbclient import NotebookClient

    notebook_path = ROOT / "Full_Code_Notebook.ipynb"
    raw = notebook_path.read_bytes()
    original = json.loads(raw.decode("utf-8"))
    notebook = nbformat.reads(raw.decode("utf-8"), as_version=4)
    checkpoints = ROOT / ".cache" / "notebook"
    checkpoints.mkdir(parents=True, exist_ok=True)
    backup = checkpoints / f"before-data-understanding-{time.time_ns()}.ipynb"
    shutil.copy2(notebook_path, backup)
    completed = []

    def save_progress():
        # Merge execution results into the current file, preserving other edits.
        current = json.loads(notebook_path.read_text(encoding="utf-8"))
        for index in completed:
            expected = original["cells"][index]
            cell = current["cells"][index]
            if cell["cell_type"] != expected["cell_type"] or cell["source"] != expected["source"]:
                fallback = checkpoints / "data-understanding-executed.ipynb"
                nbformat.write(notebook, fallback)
                raise RuntimeError("A target cell was edited during execution; outputs were saved to the checkpoint instead.")
            for key in ("outputs", "execution_count"):
                cell[key] = notebook.cells[index][key]
            if "execution" in notebook.cells[index].metadata:
                cell.setdefault("metadata", {})["execution"] = dict(notebook.cells[index].metadata.execution)
        current["metadata"]["kernelspec"] = {
            "display_name": "AuroraStays (Python 3.12.10)",
            "language": "python", "name": "aurorastays-312",
        }
        nbformat.validate(nbformat.from_dict(current))
        temporary = notebook_path.with_suffix(".ipynb.tmp")
        temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, notebook_path)

    client = NotebookClient(
        notebook, kernel_name="aurorastays-312", timeout=1800,
        iopub_timeout=180, resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    cache_code = '''
from pathlib import Path as _CachePath
import hashlib as _cache_hashlib
_cache_root = _CachePath(".cache/notebook/documents")
_cache_root.mkdir(parents=True, exist_ok=True)
_cache_records = []
for _source, _document in docs.items():
    _name = _CachePath(_source).stem
    _text = _document.export_to_markdown()
    _json_path = _cache_root / (_name + ".json")
    _json_path.write_text(_document.model_dump_json(indent=2), encoding="utf-8")
    (_cache_root / (_name + ".txt")).write_text(_text, encoding="utf-8")
    _cache_records.append({
        "source": _source,
        "source_sha256": _cache_hashlib.sha256(_CachePath(_source).read_bytes()).hexdigest(),
        "document_json": str(_json_path),
        "pages": len(_document.pages),
        "tables": len(_document.tables),
        "markdown_characters": len(_text),
    })
(_cache_root / "manifest.json").write_text(json.dumps(_cache_records, indent=2), encoding="utf-8")
print(json.dumps(_cache_records, indent=2))
'''

    try:
        with client.setup_kernel():
            for number in (20, 25, 27, 30, 33, 36, 37, 41):
                index = number - 1
                print(f"Running notebook cell {number}...", flush=True)
                started = time.monotonic()
                try:
                    client.execute_cell(notebook.cells[index], index)
                finally:
                    completed.append(index)
                    save_progress()
                print(f"Completed cell {number} in {time.monotonic() - started:.1f}s.", flush=True)
                if number == 33:
                    temporary_cell = nbformat.v4.new_code_cell(cache_code)
                    notebook.cells.append(temporary_cell)
                    try:
                        client.execute_cell(temporary_cell, len(notebook.cells) - 1, store_history=False)
                    finally:
                        notebook.cells.pop()
                    print("Converted documents cached for the next step.", flush=True)
    finally:
        save_progress()

    manifest = json.loads((checkpoints / "documents" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 4 and all(row["markdown_characters"] > 0 for row in manifest)
    print(json.dumps({"completed_cells": [i + 1 for i in completed], "documents": manifest}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
