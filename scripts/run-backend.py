"""Execute backend packaging cells; --live adds one local HTTP/RAG smoke check."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
GENERATION = runpy.run_path(str(ROOT / "scripts/run-generation.py"))


def main():
    import nbformat
    from nbclient import NotebookClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    path = ROOT / "Full_Code_Notebook.ipynb"
    original = json.loads(path.read_text(encoding="utf-8"))
    notebook = nbformat.reads(path.read_text(encoding="utf-8"), as_version=4)
    frozen = json.loads((ROOT / ".cache/notebook/retrieval_improvement/frozen_config.json").read_text())
    for number, expected in frozen["notebook_source_hashes"].items():
        assert hashlib.sha256(notebook.cells[int(number) - 1].source.encode()).hexdigest() == expected
    assert hashlib.sha256((ROOT / "policy_retrieval.py").read_bytes()).hexdigest() == frozen["retrieval_module_sha256"]
    output = ROOT / ".cache/notebook/backend_preparation"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, output / f"before-backend-execution-{time.time_ns()}.ipynb")
    secret = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))["OPENAI_API_KEY"]
    completed = []

    def redact(value):
        if isinstance(value, str):
            return value.replace(secret, "[REDACTED]")
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        return value

    def save():
        current = json.loads(path.read_text(encoding="utf-8"))
        for index in completed:
            cell = current["cells"][index]
            assert cell["source"] == original["cells"][index]["source"]
            for key in ("outputs", "execution_count"):
                cell[key] = redact(notebook.cells[index][key])
            if "execution" in notebook.cells[index].metadata:
                cell.setdefault("metadata", {})["execution"] = dict(notebook.cells[index].metadata.execution)
        nbformat.validate(nbformat.from_dict(current))
        temporary = path.with_suffix(".ipynb.tmp")
        temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    client = NotebookClient(notebook, kernel_name="aurorastays-312", timeout=1800,
                            iopub_timeout=180, allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}})

    def hidden(source):
        cell = nbformat.v4.new_code_cell(source)
        notebook.cells.append(cell)
        try:
            client.execute_cell(cell, len(notebook.cells) - 1, store_history=False)
            for item in cell.outputs:
                if item.output_type == "stream":
                    print(redact(item.text), end="", flush=True)
        except Exception as error:
            print(f"Backend hidden stage failed ({getattr(error, 'ename', type(error).__name__)}).", flush=True)
            raise
        finally:
            notebook.cells.pop()

    try:
        with client.setup_kernel():
            print("Restoring the unchanged local embedding model and frozen checkpoint...", flush=True)
            hidden("\n".join((GENERATION["RESTORE_IMPORTS"], notebook.cells[46].source,
                              notebook.cells[65].source, GENERATION["RESTORE_INDEX"])))
            for number in (139, 140, 144, 147, 151, 154):
                print(f"Running backend cell {number}...", flush=True)
                index = number - 1
                client.execute_cell(notebook.cells[index], index)
                completed.append(index)
                save()
            hidden("import runpy\n_backend_verify = runpy.run_path('scripts/verify-backend.py')\n"
                   f"_backend_verify['main'](live={args.live!r})")
    finally:
        save()
    print("PASS backend files/index exported and verified; notebook kernel stopped.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAIL backend preparation ({type(error).__name__}); provider details omitted.", flush=True)
        sys.exit(1)
