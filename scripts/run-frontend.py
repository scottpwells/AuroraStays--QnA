"""Execute the frontend export cells and preserve the notebook's scored state."""

import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
runpy.run_path(str(ROOT / "scripts/run-generation.py"))


def main():
    import nbformat
    from nbclient import NotebookClient

    path = ROOT / "Full_Code_Notebook.ipynb"
    original = json.loads(path.read_text(encoding="utf-8"))
    notebook = nbformat.reads(path.read_text(encoding="utf-8"), as_version=4)
    output = ROOT / ".cache/notebook/frontend_preparation"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, output / f"before-frontend-execution-{time.time_ns()}.ipynb")
    completed = []

    def save():
        current = json.loads(path.read_text(encoding="utf-8"))
        for index in completed:
            assert current["cells"][index]["source"] == original["cells"][index]["source"]
            for key in ("outputs", "execution_count"):
                current["cells"][index][key] = notebook.cells[index][key]
            if "execution" in notebook.cells[index].metadata:
                current["cells"][index].setdefault("metadata", {})["execution"] = dict(notebook.cells[index].metadata.execution)
        nbformat.validate(nbformat.from_dict(current))
        temporary = path.with_suffix(".ipynb.tmp")
        temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    client = NotebookClient(notebook, kernel_name="aurorastays-312", timeout=180,
                            allow_errors=False, resources={"metadata": {"path": str(ROOT)}})
    try:
        with client.setup_kernel():
            for number in (159, 161, 165, 168):
                print(f"Running frontend cell {number}...", flush=True)
                index = number - 1
                client.execute_cell(notebook.cells[index], index)
                completed.append(index)
                save()
    finally:
        save()
    print("PASS frontend exports saved; notebook kernel stopped; no model/API calls.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
