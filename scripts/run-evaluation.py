"""Execute the evaluation dataset section; --resume reuses saved work.

A fresh baseline makes 10 generation requests, 40 metric requests, and one
judge connectivity request. Results remain local; the test set is not run.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
# Reuse the preceding step's environment and verified checkpoint restoration.
GENERATION = runpy.run_path(str(ROOT / "scripts" / "run-generation.py"))
for name, value in {
    "DEEPEVAL_RETRY_MAX_ATTEMPTS": "1",
    "DEEPEVAL_NO_INSPECT_PROMPT": "1",
    "CONFIDENT_OPEN_BROWSER": "0",
}.items():
    os.environ[name] = value


IMPORTS = r'''
import pandas as pd
from IPython.display import display
from deepeval import evaluate
from deepeval.models import OpenAIModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from langchain_core.documents import Document as LCDocument
from deepeval.test_run import global_test_run_manager
from functools import partial
# Use DeepEval's local-run flag for every new test run in this kernel.
global_test_run_manager.create_test_run = partial(
    global_test_run_manager.create_test_run, disable_request=True
)
'''

TRACKING = r'''
_eval_dir = _GenerationPath(".cache/notebook/evaluation")
_resume = os.environ.get("AURORASTAYS_EVALUATION_RESUME") == "1"
_progress_path = _eval_dir / "progress.json"
_progress = json.loads(_progress_path.read_text()) if _resume and _progress_path.exists() else {
    "generation_requests": 0, "judge_requests": 0, "judge_metric_requests": 0,
    "judge_preflight_passed": False,
}
_answers_path = _eval_dir / "generated_answers.json"
_answers = json.loads(_answers_path.read_text()) if _resume and _answers_path.exists() else {}
_progress_path.write_text(json.dumps(_progress, indent=2), encoding="utf-8")
_original_rag = rag
def rag(*args, **kwargs):
    assert not args, "The checkpoint wrapper expects named RAG arguments."
    key = _generation_hashlib.sha256(json.dumps(
        {name: value for name, value in kwargs.items() if name != "vectorstore"},
        sort_keys=True,
    ).encode()).hexdigest()
    if _resume and key in _answers:
        saved = _answers[key]
        return saved["answer"], saved["retrieved_chunks"]
    _progress["generation_requests"] += 1
    _progress_path.write_text(json.dumps(_progress, indent=2), encoding="utf-8")
    answer, chunks = _original_rag(*args, **kwargs)
    _answers[key] = {"query": kwargs["query"], "answer": answer, "retrieved_chunks": chunks}
    _answers_path.write_text(json.dumps(_answers, indent=2), encoding="utf-8")
    return answer, chunks

def _track_judge(method, is_metric):
    def tracked(*args, **kwargs):
        _progress["judge_requests"] += 1
        if is_metric:
            _progress["judge_metric_requests"] += 1
        _progress_path.write_text(json.dumps(_progress, indent=2), encoding="utf-8")
        result = method(*args, **kwargs)
        if is_metric:
            completion = result[0]
            _progress["last_judge_model"] = completion.model
            _progress["last_finish_reason"] = completion.choices[0].finish_reason
        _progress_path.write_text(json.dumps(_progress, indent=2), encoding="utf-8")
        return result
    return tracked
eval_model.generate = _track_judge(eval_model.generate, False)
eval_model.generate_raw_response = _track_judge(eval_model.generate_raw_response, True)
'''

CHECK_PARAMETERS = r'''
_probe_calls = []
def _probe_rag(**kwargs):
    _probe_calls.append(kwargs)
    return "Probe answer", ["Probe retrieval"]
_saved_rag = rag
try:
    rag = _probe_rag
    _probe_cases = test_case_generator(pd.DataFrame([{
        "question": "Parameter forwarding probe?", "expected_answer": "Expected",
        "relevant_context": "First context <> Second context",
    }]), vectorstore, k=2, temperature=0.17, top_p=0.93, max_tokens=321)
finally:
    rag = _saved_rag
assert len(_probe_calls) == 1
assert {name: _probe_calls[0][name] for name in ("k", "temperature", "top_p", "max_tokens")} == {
    "k": 2, "temperature": 0.17, "top_p": 0.93, "max_tokens": 321,
}
assert _probe_cases[0].context == ["First context", "Second context"]
print("PASS parameter experiments use the supplied arguments; ground-truth contexts split correctly.")
'''

CHECK_ROUTING = r'''
from types import SimpleNamespace as _EvalNamespace
_probe_cases = [LLMTestCase(input="at boundary", actual_output="a", expected_output="e"),
                LLMTestCase(input="below boundary", actual_output="a", expected_output="e")]
_probe_results = _EvalNamespace(test_results=[
    _EvalNamespace(input=case.input, metrics_data=[
        _EvalNamespace(name=name + " [GEval]", score=score, error=None)
        for name in ("Faithfulness", "Contextual Recall", "Contextual Precision", "Answer Relevancy")
    ]) for case, score in zip(_probe_cases, (0.75, 0.749), strict=True)
])
_probe_frame = create_evaluation_df(_probe_cases, _probe_results)
assert list(_probe_frame["Type"]) == ["AUTO ANSWER", "MANUAL LOOKUP"]
print("PASS routing includes the exact 0.75 boundary and uses the two specified actions.")
'''

VALIDATE = r'''
import math as _eval_math
assert len(df) == len(test_cases) == len(results.test_results) == len(evaluation_df) == 10
assert len(metrics) == 4 and all(len(result.metrics_data) == 4 for result in results.test_results)
assert list(evaluation_df["Question"]) == list(df["question"])
assert all(not metric.error and _eval_math.isfinite(metric.score) and 0 <= metric.score <= 1
           for result in results.test_results for metric in result.metrics_data)
assert results.confident_link is None and global_test_run_manager.disable_request is True
assert set(evaluation_df["Type"]).issubset({"AUTO ANSWER", "MANUAL LOOKUP"})
assert _progress["judge_preflight_passed"]
_audit = {
    "complete": True,
    "dataset_sha256": _generation_hashlib.sha256(_GenerationPath("Data/evaluation.csv").read_bytes()).hexdigest(),
    "test_set_executed": False,
    "cases": 10, "metric_scores": 40,
    "requests": _progress,
    "local_results_only": True,
    "index_hashes": _index_summary["index_file_hashes"],
}
(_eval_dir / "validation.json").write_text(json.dumps(_audit, indent=2), encoding="utf-8")
print(json.dumps({"validation": _audit, "summary": json.loads((_eval_dir / "summary.json").read_text())}, indent=2))
'''

CHECK_RETRIEVAL = r'''
from pathlib import Path as _ProbePath
_probe_questions = list(pd.read_csv("Data/evaluation.csv")["question"])
_retrieval_probes = [retrieve(question, k=4, vectorstore=vectorstore) for question in _probe_questions]
_probe_contexts = ["\n".join(contexts) for contexts in _retrieval_probes]
_retrieval_checks = {
    "chicago_property_times": all(fact in _probe_contexts[0] for fact in ("ASH-CHI Loop", "3:00 PM / 12:00 PM")),
    "flexible_deadline": "6:00 PM" in _probe_contexts[1],
    "miami_pet_fee_and_rate": all(fact in _probe_contexts[2] for fact in ("ASH-MIA Brickell", "$100 per stay", "Non-refundable")),
    "seattle_breakfast_and_parking": all(fact in _probe_contexts[5] for fact in ("ASH-SEA Pike", "$38 per night", "Included with room")),
    "quiet_hours_and_handling": all(fact in _probe_contexts[8] for fact in ("10:00 PM to 7:00 AM", "manager")),
    "late_refund_exceptions": all(fact in _probe_contexts[9] for fact in ("documented service failure", "duplicate booking", "verified system error")),
}
_probe_output = _ProbePath(".cache/notebook/retrieval_improvement/retrieval_checks.json")
_probe_output.parent.mkdir(parents=True, exist_ok=True)
_probe_output.write_text(json.dumps({"checks": _retrieval_checks, "queries": [
    {"question": question, "contexts": contexts} for question, contexts in zip(_probe_questions, _retrieval_probes, strict=True)
]}, indent=2), encoding="utf-8")
print(json.dumps(_retrieval_checks, indent=2))
assert all(_retrieval_checks.values()), "Targeted retrieval evidence is still missing; inspect retrieval_checks.json before API evaluation."
'''


def stage_code(filename, name):
    module = ast.parse((ROOT / "scripts" / filename).read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing execution stage: {name}")


def main():
    import nbformat
    from nbclient import NotebookClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-retrieval", action="store_true",
                        help="Rebuild structured chunks and FAISS, verify retrieval, then evaluate in one kernel.")
    args = parser.parse_args()
    if args.resume and args.rebuild_retrieval:
        parser.error("Resume an already rebuilt index with --resume alone.")
    os.environ["AURORASTAYS_EVALUATION_RESUME"] = "1" if args.resume else "0"
    path = ROOT / "Full_Code_Notebook.ipynb"
    original = json.loads(path.read_text(encoding="utf-8"))
    notebook = nbformat.reads(path.read_text(encoding="utf-8"), as_version=4)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    secret = config["OPENAI_API_KEY"]
    output_dir = ROOT / ".cache" / "notebook" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    def prepare_manifest():
        fingerprint = hashlib.sha256(json.dumps({
            "dataset": hashlib.sha256((ROOT / "Data/evaluation.csv").read_bytes()).hexdigest(),
            "index": json.loads((ROOT / ".cache/notebook/faiss_index/summary.json").read_text())["index_file_hashes"],
            "retrieval_module": hashlib.sha256((ROOT / "policy_retrieval.py").read_bytes()).hexdigest(),
            "endpoint_hash": hashlib.sha256(config["OPENAI_API_BASE"].strip().encode()).hexdigest(),
            "sources": {str(n): notebook.cells[n - 1].source for n in
                        (47, 54, 56, 66, 70, 73, 85, 90, 93, 105, 106, 110, 114, 118, 122, 124)},
        }, sort_keys=True).encode()).hexdigest()
        manifest_path = output_dir / "run_manifest.json"
        if args.resume:
            assert json.loads(manifest_path.read_text())["fingerprint"] == fingerprint, "Run inputs changed; start a fresh baseline."
        else:
            if manifest_path.exists():
                archive = output_dir / f"previous-{time.time_ns()}"
                archive.mkdir()
                for item in list(output_dir.iterdir()):
                    if item.is_file():
                        shutil.copy2(item, archive / item.name)
            manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2), encoding="utf-8")

    if not args.rebuild_retrieval:
        prepare_manifest()
    checkpoints = ROOT / ".cache" / "notebook"
    shutil.copy2(path, checkpoints / f"before-evaluation-execution-{time.time_ns()}.ipynb")
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
        current = json.loads(path.read_text(encoding="utf-8"))
        for index in completed:
            cell = current["cells"][index]
            assert cell["source"] == original["cells"][index]["source"], "A target cell changed during execution."
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
            # Only the exception class is safe to print for provider failures.
            print(f"Validation cell failed ({getattr(error, 'ename', type(error).__name__)}).", flush=True)
            raise
        finally:
            notebook.cells.pop()

    def run_cell(number):
        index = number - 1
        print(f"Running notebook cell {number}...", flush=True)
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
            print("Restoring the verified model/index and generation definitions...", flush=True)
            if args.rebuild_retrieval:
                run_hidden("\n".join((GENERATION["RESTORE_IMPORTS"], IMPORTS,
                    stage_code("run-chunking.py", "BOOTSTRAP"))))
                for number in (47, 50, 54, 56, 59, 62, 66, 70, 73, 76):
                    run_cell(number)
                    if number == 62:
                        run_hidden(stage_code("run-chunking.py", "CACHE_AND_VALIDATE"))
                        run_hidden(stage_code("run-vectorstore.py", "BOOTSTRAP"))
                    elif number == 66:
                        run_hidden(stage_code("run-vectorstore.py", "VALIDATE_MODEL"))
                    elif number == 70:
                        run_hidden(stage_code("run-vectorstore.py", "VALIDATE_INDEX"))
                    elif number == 76:
                        run_hidden(stage_code("run-vectorstore.py", "VALIDATE_RETRIEVAL"))
                run_hidden(GENERATION["RESTORE_INDEX"] + "\n" + "\n".join(
                    notebook.cells[n - 1].source for n in (82, 85, 90, 93)))
                run_hidden(CHECK_RETRIEVAL)
                prepare_manifest()
            else:
                run_hidden("\n".join((GENERATION["RESTORE_IMPORTS"], IMPORTS,
                    notebook.cells[46].source, notebook.cells[65].source, GENERATION["RESTORE_INDEX"],
                    *(notebook.cells[n - 1].source for n in (73, 82, 85, 90, 93)))))
            for number in (101, 105, 106, 110, 114, 118, 122, 124):
                if number == 106 and args.resume:
                    progress = json.loads((output_dir / "progress.json").read_text())
                    if progress.get("judge_preflight_passed"):
                        print("Reusing the successful judge connectivity check.", flush=True)
                        continue
                run_cell(number)
                if number == 105:
                    run_hidden(TRACKING)
                elif number == 106:
                    run_hidden('_progress["judge_preflight_passed"] = True\n_progress_path.write_text(json.dumps(_progress, indent=2), encoding="utf-8")')
                elif number == 114:
                    run_hidden(CHECK_PARAMETERS)
                elif number == 122:
                    run_hidden(CHECK_ROUTING)
            run_hidden(VALIDATE)
    finally:
        save_progress()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAIL evaluation execution ({type(error).__name__}); provider details omitted. Saved progress can be resumed.", flush=True)
        sys.exit(1)
