"""Run the held-out test section once against the verified frozen configuration.

Uses the same generator, four GEval metrics, and composite routing rule as the
development evaluation. A complete fresh run makes 51 live API requests.
--resume reuses completed generation/judge work after an interruption.
--preflight-only verifies inputs without starting a kernel or making API calls.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
EVALUATION = runpy.run_path(str(ROOT / "scripts/run-evaluation.py"))
GENERATION = EVALUATION["GENERATION"]
OUTPUT = ROOT / ".cache/notebook/test_evaluation"
FROZEN_PATH = ROOT / ".cache/notebook/retrieval_improvement/frozen_config.json"


RESTORE_SCORES = r'''
import json
import pandas as pd
from IPython.display import display
from pathlib import Path as _GenerationPath
import hashlib as _generation_hashlib
from deepeval.test_case import LLMTestCase
from deepeval.evaluate.types import EvaluationResult
from deepeval.test_run import global_test_run_manager
global_test_run_manager.disable_request = True
_eval_dir = TEST_OUTPUT_DIR = _GenerationPath(".cache/notebook/test_evaluation")
_frozen = json.loads(_GenerationPath(".cache/notebook/retrieval_improvement/frozen_config.json").read_text())
MODEL, EVAL_MODEL = _frozen["model"], _frozen["judge"]
K, TEMPERATURE, TOP_P, MAX_TOKENS = (_frozen["parameters"][name] for name in ("k", "temperature", "top_p", "max_tokens"))
df = pd.read_csv("Data/test.csv")
test_cases = [LLMTestCase.model_validate(case) for case in json.loads((_eval_dir / "test_cases.json").read_text())]
results = EvaluationResult.model_validate_json((_eval_dir / "results.json").read_text())
_progress = json.loads((_eval_dir / "progress.json").read_text())
_index_summary = json.loads(_GenerationPath(".cache/notebook/faiss_index/summary.json").read_text())
# These are saved MetricData objects for validation, not executable graders.
metrics = results.test_results[0].metrics_data
assert len(test_cases) == len(results.test_results) == 10
assert all(len(result.metrics_data) == 4 for result in results.test_results)
print("Restored 10 saved answers and 40 saved scores; summary only, no model/API calls.")
'''


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen(notebook):
    frozen = read(FROZEN_PATH)
    assert digest(ROOT / "policy_retrieval.py") == frozen["retrieval_module_sha256"], "Frozen retrieval code changed."
    for number, expected in frozen["notebook_source_hashes"].items():
        source = "".join(notebook["cells"][int(number) - 1]["source"])
        assert hashlib.sha256(source.encode()).hexdigest() == expected, f"Frozen cell {number} changed."
    for name, expected in frozen["index_hashes"].items():
        assert digest(ROOT / ".cache/notebook/faiss_index" / name) == expected, "Frozen index changed."
    for source, expected in frozen["source_hashes"].items():
        assert digest(ROOT / source) == expected, "Source PDF changed since freezing."
    assert frozen["evaluation_retrieval_replay_matches"] == 10
    return frozen


def main():
    import nbformat
    from nbclient import NotebookClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    path = ROOT / "Full_Code_Notebook.ipynb"
    original = read(path)
    frozen = verify_frozen(original)
    assert len(original["cells"]) == 213
    nbformat.validate(nbformat.from_dict(original))
    # Inspect structure and split identity only; no test evidence drives tuning.
    with (ROOT / "Data/test.csv").open(encoding="utf-8-sig", newline="") as handle:
        test_rows = list(csv.DictReader(handle))
    assert len(test_rows) == 10
    assert all(all(isinstance(row.get(field), str) and row[field].strip()
                   for field in ("question", "expected_answer", "relevant_context")) for row in test_rows)
    assert len({row["question"] for row in test_rows}) == 10
    with (ROOT / "Data/evaluation.csv").open(encoding="utf-8-sig", newline="") as handle:
        development_questions = {row["question"].strip().casefold() for row in csv.DictReader(handle)}
    overlap = sum(row["question"].strip().casefold() in development_questions for row in test_rows)
    assert overlap == 0, "Held-out questions overlap with the development split."
    config = read(ROOT / "config.json")
    secret = config["OPENAI_API_KEY"]
    inputs = {
        "dataset": "Data/test.csv", "dataset_sha256": digest(ROOT / "Data/test.csv"),
        "frozen_config_sha256": digest(FROZEN_PATH),
        "endpoint_hash": hashlib.sha256(config["OPENAI_API_BASE"].strip().encode()).hexdigest(),
        "test_cell_sources": {str(number): "".join(original["cells"][number - 1]["source"])
                              for number in (129, 131, 133)},
        "shared_helper_sha256": digest(ROOT / "scripts/run-evaluation.py"),
        "generation_helper_sha256": digest(ROOT / "scripts/run-generation.py"),
        "runner_sha256": digest(Path(__file__)),
    }
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    manifest_path = OUTPUT / "run_manifest.json"
    if args.resume:
        manifest = read(manifest_path)
        assert manifest["fingerprint"] == fingerprint, "Test-run inputs changed; do not mix results across configurations."
        assert manifest.get("complete") is not True, "The held-out run is already complete; review its saved results."
    else:
        assert not manifest_path.exists(), "A held-out run already exists; use --resume only if incomplete."
        for number in (129, 131, 133):
            cell = original["cells"][number - 1]
            assert cell.get("execution_count") is None and not cell.get("outputs"), "Held-out cells already have results."
    print("PASS frozen settings, source/index hashes, notebook schema, and test structure: 10 rows, no exact development-question overlap.", flush=True)
    if args.preflight_only:
        print("Preflight only: no kernel, no API calls, no test answers generated.", flush=True)
        return 0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        shutil.copy2(path, OUTPUT / "before-test-execution.ipynb")
        protected = [FROZEN_PATH, ROOT / "policy_retrieval.py", ROOT / "Data/evaluation.csv", ROOT / "Data/test.csv",
                     *ROOT.joinpath(".cache/notebook/evaluation").glob("*.json"),
                     *ROOT.joinpath(".cache/notebook/evaluation").glob("*.csv")]
        manifest = {
            "fingerprint": fingerprint, "inputs": inputs, "complete": False,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protected_file_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in protected},
            "test_set_used_for_tuning": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.environ["AURORASTAYS_EVALUATION_RESUME"] = "1" if args.resume else "0"
    notebook = nbformat.reads(path.read_text(encoding="utf-8"), as_version=4)
    completed = []

    def redact(value):
        if isinstance(value, str):
            return value.replace(secret, "[REDACTED]")
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        return value

    def save_progress():
        current = read(path)
        verify_frozen(current)
        for index in completed:
            cell = current["cells"][index]
            assert cell["source"] == original["cells"][index]["source"], "A test cell changed during execution."
            for key in ("outputs", "execution_count"):
                cell[key] = redact(notebook.cells[index][key])
            if "execution" in notebook.cells[index].metadata:
                cell.setdefault("metadata", {})["execution"] = dict(notebook.cells[index].metadata.execution)
        nbformat.validate(nbformat.from_dict(current))
        temporary = path.with_suffix(".ipynb.tmp")
        temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    client = NotebookClient(notebook, kernel_name="aurorastays-312", timeout=3600,
                            iopub_timeout=180, allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}})

    def run_hidden(source):
        cell = nbformat.v4.new_code_cell(source)
        notebook.cells.append(cell)
        try:
            client.execute_cell(cell, len(notebook.cells) - 1, store_history=False)
            for output in cell.outputs:
                if output.output_type == "stream":
                    print(redact(output.text), end="", flush=True)
        except Exception as error:
            print(f"Hidden verification failed ({getattr(error, 'ename', type(error).__name__)}).", flush=True)
            raise
        finally:
            notebook.cells.pop()

    def run_cell(number):
        index = number - 1
        print(f"Running held-out notebook cell {number}...", flush=True)
        started = time.monotonic()
        try:
            client.execute_cell(notebook.cells[index], index)
        except Exception as error:
            notebook.cells[index].outputs = [nbformat.v4.new_output(
                "error", ename=type(error).__name__, traceback=[],
                evalue="Execution failed; provider details omitted to protect credentials.")]
            raise
        finally:
            completed.append(index)
            save_progress()
        print(f"Completed cell {number} in {time.monotonic() - started:.1f}s.", flush=True)

    try:
        with client.setup_kernel():
            saved_scores = (OUTPUT / "results.json").is_file() and (OUTPUT / "test_cases.json").is_file()
            if args.resume and saved_scores:
                run_hidden(RESTORE_SCORES)
                run_hidden(notebook.cells[121].source)
                run_cell(133)
            else:
                print("Restoring the frozen local model/index and unchanged generator/judge definitions...", flush=True)
                run_hidden("\n".join((GENERATION["RESTORE_IMPORTS"], EVALUATION["IMPORTS"],
                    notebook.cells[46].source, notebook.cells[65].source, GENERATION["RESTORE_INDEX"],
                    *(notebook.cells[n - 1].source for n in (73, 82, 85, 90, 93)),
                    GENERATION["VALIDATE_CONFIGURATION"], notebook.cells[104].source)))
                run_hidden(EVALUATION["TRACKING"].replace(".cache/notebook/evaluation", ".cache/notebook/test_evaluation"))
                # Separate judge cache from the development run, including on resume.
                run_hidden('''
from deepeval.test_run.cache import global_test_run_cache_manager
global_test_run_cache_manager.cache_file_name = str(_eval_dir / "deepeval-cache.json")
global_test_run_cache_manager.temp_cache_file_name = str(_eval_dir / "deepeval-cache.tmp.json")
''')
                progress = read(OUTPUT / "progress.json")
                if not (args.resume and progress.get("judge_preflight_passed")):
                    run_hidden(notebook.cells[105].source)
                    run_hidden('_progress["judge_preflight_passed"] = True\n_progress_path.write_text(json.dumps(_progress, indent=2), encoding="utf-8")')
                for number in (110, 114, 122):
                    run_hidden(notebook.cells[number - 1].source)
                run_hidden(EVALUATION["CHECK_PARAMETERS"])
                run_hidden(EVALUATION["CHECK_ROUTING"])
                for number in (129, 131, 133):
                    run_cell(number)
            validation = EVALUATION["VALIDATE"].replace("Data/evaluation.csv", "Data/test.csv").replace(
                '"test_set_executed": False', '"test_set_executed": True').replace(
                'len(evaluation_df)', 'len(test_evaluation_df)').replace(
                'evaluation_df[', 'test_evaluation_df[')
            run_hidden(validation)
    finally:
        save_progress()

    verify_frozen(read(path))
    for name, expected in manifest["protected_file_hashes"].items():
        assert digest(ROOT / name) == expected, f"Protected input/development result changed: {name}"
    assert read(OUTPUT / "summary.json")["test_set_evaluated"] is True
    manifest["complete"] = True
    manifest["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["frozen_configuration_and_development_artifacts_unchanged"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("PASS held-out run complete; frozen configuration/development results unchanged; kernel shut down.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAIL held-out execution ({type(error).__name__}); provider details omitted. Resume incomplete work with --resume.", flush=True)
        sys.exit(1)
