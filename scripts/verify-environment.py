"""Check local notebook prerequisites without model downloads or API calls."""

import ast
import csv
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
for name, value in {
    "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
    "DEEPEVAL_UPDATE_WARNING_OPT_OUT": "YES",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HOME": str(ROOT / ".cache" / "huggingface"),
    "JUPYTER_CONFIG_DIR": str(ROOT / ".jupyter" / "config"),
    "JUPYTER_DATA_DIR": str(ROOT / ".jupyter" / "data"),
    "JUPYTER_RUNTIME_DIR": str(ROOT / ".jupyter" / "runtime"),
    "IPYTHONDIR": str(ROOT / ".jupyter" / "ipython"),
}.items():
    os.environ[name] = value


def main():
    from packaging.requirements import Requirement
    import nbformat

    failures = []

    def check(label, passed, detail=""):
        print(f"{'PASS' if passed else 'FAIL'} {label}: {detail}", flush=True)
        if not passed:
            failures.append(label)

    check("Python", sys.version_info[:3] == (3, 12, 10), sys.version.split()[0])
    check("isolated environment", sys.prefix != sys.base_prefix, sys.executable)
    for line in (ROOT / "requirements-notebook.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        req = Requirement(line)
        try:
            version = metadata.version(req.name)
            check(req.name, req.specifier.contains(version), version)
        except metadata.PackageNotFoundError:
            check(req.name, False, "not installed")

    notebook = nbformat.read(ROOT / "Full_Code_Notebook.ipynb", as_version=4)
    nbformat.validate(notebook)
    check("notebook schema", True, f"v{notebook.nbformat}.{notebook.nbformat_minor}; {len(notebook.cells)} cells")
    blanks, placeholders = [], []
    for number, cell in enumerate(notebook.cells, 1):
        if cell.cell_type != "code":
            continue
        if not cell.source.strip():
            blanks.append(number)
        if "_____" in cell.source:
            placeholders.append(number)
        source = cell.source
        if source.startswith("%%writefile"):
            target = source.splitlines()[0].split()[-1]
            if not target.endswith(".py"):
                continue
            source = "\n".join(source.splitlines()[1:])
        elif any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
            # Installation magics are valid IPython, rather than Python syntax.
            continue
        try:
            ast.parse(source, filename=f"notebook_cell_{number}")
        except SyntaxError as error:
            check(f"cell {number} syntax", False, str(error))
    print(f"INFO exercise cells left blank: {blanks}")
    print(f"INFO cells containing placeholders: {placeholders}")

    # Execute only the import/setup cell in this separate process. This does
    # not execute notebook tasks or write notebook outputs.
    try:
        exec(compile(notebook.cells[19].source, "notebook_imports_cell_20", "exec"), {})
        check("all notebook cell 20 imports", True, "offline; no model or API requests")
    except Exception as error:
        check("all notebook cell 20 imports", False, f"{type(error).__name__}: {error}")

    import faiss
    import numpy as np
    import torch

    index = faiss.IndexFlatL2(2)
    index.add(np.array([[1, 0], [0, 1]], dtype="float32"))
    _, matches = index.search(np.array([[1, 0]], dtype="float32"), 1)
    check("FAISS native search", int(matches[0, 0]) == 0, f"{index.ntotal} vectors")
    check("PyTorch CPU calculation", torch.tensor([1, 2]).sum().item() == 3, torch.__version__)

    for name in (
        "aurorastay_room_and_property_information.pdf",
        "aurorastay_refunds_and_additional_charges_guide.pdf",
        "aurorastay_guest_conduct_accessibility_and_resolution_guidelines.pdf",
        "aurorastay_booking_terms_and_cancellation_policy.pdf",
    ):
        path = ROOT / "Data" / name
        present = path.is_file() and path.read_bytes().startswith(b"%PDF-")
        check(name, present, "PDF present" if present else "missing or invalid PDF signature")

    for name in ("evaluation.csv", "test.csv"):
        path = ROOT / "Data" / name
        if not path.is_file():
            check(name, False, "missing")
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns, rows = reader.fieldnames or [], list(reader)
        required = {"question", "expected_answer", "relevant_context"}
        valid = required.issubset(columns) and len(rows) == 10
        if valid:
            valid = all(all(row.get(key, "").strip() for key in required) for row in rows)
        check(name, valid, f"{len(rows)} rows; columns={columns}")

    config_path = ROOT / "config.json"
    configured = False
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))
            configured = all(
                isinstance(config.get(key), str)
                and bool(config[key].strip())
                and not config[key].startswith("REPLACE_")
                for key in ("OPENAI_API_KEY", "OPENAI_API_BASE")
            )
        except (ValueError, OSError):
            pass
    print("INFO API config: " + (
        "present; credentials and model access not tested"
        if configured else "missing/incomplete; required before cell 82; see config.example.json"
    ))
    print("INFO full model download, PDF conversion, API calls, evaluation, and deployment remain for the walkthrough.")
    print(f"RESULT {'PASS' if not failures else 'FAIL'}: {len(failures)} environment/data failures", flush=True)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
