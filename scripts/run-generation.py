"""Execute the notebook generation section and three small live API checks.

Uses the verified local FAISS checkpoint. Repeating this script consumes API
usage for connectivity, one grounded answer, and one missing-context answer.
Use --reuse-connectivity to retain a previously successful connectivity probe
and send only the two answer-generation checks after changing the prompt.
"""

import argparse
import asyncio
import json
import logging
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
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
    "DEEPEVAL_UPDATE_WARNING_OPT_OUT": "YES",
    "PYTHONUTF8": "1",
}.items():
    os.environ[name] = value
logging.disable(logging.CRITICAL)


RESTORE_INDEX = r'''
from pathlib import Path as _GenerationPath
import hashlib as _generation_hashlib
import logging as _generation_logging
_generation_logging.disable(_generation_logging.CRITICAL)
_index_path = _GenerationPath(".cache/notebook/faiss_index")
_index_summary = json.loads((_index_path / "summary.json").read_text(encoding="utf-8"))
_chunk_summary = json.loads(_GenerationPath(".cache/notebook/chunks/summary.json").read_text(encoding="utf-8"))
assert EMBED_MODEL == _index_summary["embedding_model"] == _chunk_summary["embedding_model"]
for _source, _digest in _chunk_summary["source_hashes"].items():
    assert _generation_hashlib.sha256(_GenerationPath(_source).read_bytes()).hexdigest() == _digest, "Source changed; refresh its chunks and index."
assert _generation_hashlib.sha256(_GenerationPath(".cache/notebook/chunks/chunks.jsonl").read_bytes()).hexdigest() == _index_summary["chunk_file_sha256"]
for _name, _digest in _index_summary["index_file_hashes"].items():
    assert _name in ("index.faiss", "index.pkl")
    assert _generation_hashlib.sha256((_index_path / _name).read_bytes()).hexdigest() == _digest, "Checkpoint changed; verify it before loading."
# These files were generated and validated locally by the preceding step.
vectorstore = FAISS.load_local(str(_index_path), embeddings, allow_dangerous_deserialization=True)
assert vectorstore.index.ntotal == _index_summary["vectors"] == _chunk_summary["total_chunks"]
assert vectorstore.index.d == _index_summary["embedding_dimensions"] == 384
print(f"Restored verified FAISS checkpoint: {vectorstore.index.ntotal} vectors, {vectorstore.index.d} dimensions.")
'''

RESTORE_IMPORTS = r'''
import json, os
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
'''

VALIDATE_CONFIGURATION = r'''
assert MODEL == llm.model_name == "gpt-4o-mini"
assert TEMPERATURE == llm.temperature == 0.0
assert TOP_P == llm.top_p == 1.0
assert MAX_TOKENS == llm.max_tokens == 512
assert K == 4
assert llm.request_timeout == 60
assert llm.max_retries == 0
assert llm.openai_api_base == config["OPENAI_API_BASE"].strip()
assert llm.openai_api_key.get_secret_value() == config["OPENAI_API_KEY"]
print("PASS LLM settings and configured endpoint; credential values omitted.")
'''

VALIDATE_CONNECTIVITY = r'''
_connectivity_text = response.content.strip()
_connectivity_usage = response.usage_metadata
assert _connectivity_text == "LLM is ready.", "Connectivity response did not match the requested marker."
print("PASS live LLM connectivity: LLM is ready.")
'''

CAPTURE_GENERATION = r'''
_generation_response_metadata = []
_notebook_generate = generate
def generate(*args, **kwargs):
    message = _notebook_generate(*args, **kwargs)
    _generation_response_metadata.append({
        "usage": message.usage_metadata,
        "model": message.response_metadata.get("model_name"),
        "finish_reason": message.response_metadata.get("finish_reason"),
    })
    return message
'''

REUSE_CONNECTIVITY = r'''
_previous_generation = json.loads(_GenerationPath(".cache/notebook/generation/summary.json").read_text(encoding="utf-8"))
assert _previous_generation["checks"]["connectivity"]
assert (_previous_generation["model"], _previous_generation["temperature"],
        _previous_generation["top_p"], _previous_generation["max_output_tokens_per_request"]) == (MODEL, TEMPERATURE, TOP_P, MAX_TOKENS)
_connectivity_text = "LLM is ready."
_connectivity_usage = _previous_generation["connectivity_usage"]
_generation_live_requests = 2
_generation_cumulative_requests = _previous_generation.get("cumulative_live_requests", _previous_generation["live_requests"]) + _generation_live_requests
_generation_connectivity_reused = True
print(f"Reusing the successful connectivity probe; {_generation_live_requests} answer check(s) will make live requests.")
'''

VALIDATE_GENERATION = r'''
import re as _generation_re
from policy_retrieval import retrieve_policy_documents
_supporting_sources = {
    _GenerationPath(source).name
    for doc in retrieve_policy_documents(vectorstore, query, k=K)
    if all(fact in doc.page_content for fact in ("ASH-CHI Loop", "$50 per night", "$150 stay cap"))
    for source in doc.metadata.get("sources", [doc.metadata["source"]])
}
_checks = {
    "connectivity": _connectivity_text == "LLM is ready.",
    "within_retrieval_context_limit": 0 < len(rel_chunks) <= K == 4,
    "retrieval_contains_property_fee_and_cap": all(
        fact in "\n".join(rel_chunks) for fact in ("ASH-CHI Loop", "$50 per night", "$150 stay cap")
    ),
    "answer_matches_property": "ASH-CHI Loop" in response,
    "answer_contains_fee": bool(_generation_re.search(r"\$\s*50(?:\.00)?\b", response)),
    "answer_contains_cap": bool(_generation_re.search(r"\$\s*150(?:\.00)?\b", response)),
    "answer_preserves_nightly_unit": any(term in response.lower() for term in ("per night", "nightly", "/night")),
    "answer_preserves_stay_cap": "stay" in response.lower() and any(term in response.lower() for term in ("cap", "maximum", "up to")),
    "answer_cites_supporting_source": any(source in response for source in _supporting_sources),
}
_negative_query = "What is the complimentary helicopter transfer schedule at ASH-CHI Loop?"
_negative_message = generate(_negative_query, [], llm)
_negative_answer = _negative_message.content
_negative_usage = _negative_message.usage_metadata
_negative_lower = _negative_answer.lower()
_checks["missing_context_abstention"] = any(phrase in _negative_lower for phrase in (
    "insufficient", "missing", "not enough", "don't have", "do not have", "not provided",
    "couldn't find", "could not find",
    "no information", "not contain", "no policy", "unable to", "cannot", "can't",
    "does not", "not specify", "not available", "no retrieved", "no excerpts",
)) and any(word in _negative_lower for word in ("property", "profile", "contact", "verify"))
_summary = {
    "tested_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "model": MODEL,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_output_tokens_per_request": MAX_TOKENS,
    "request_timeout_seconds": 60,
    "automatic_retries": 0,
    "retrieval_k": K,
    "live_requests": _generation_live_requests,
    "cumulative_live_requests": _generation_cumulative_requests,
    "connectivity_reused": _generation_connectivity_reused,
    "generation_response_metadata": _generation_response_metadata,
    "connectivity_usage": _connectivity_usage,
    "query": query,
    "answer": response,
    "retrieved_chunks": rel_chunks,
    "missing_context_query": _negative_query,
    "missing_context_answer": _negative_answer,
    "missing_context_usage": _negative_usage,
    "checks": _checks,
    "passed": all(_checks.values()),
    "scope": "Small generation smoke checks; benchmark evaluation has not run.",
}
_output = _GenerationPath(".cache/notebook/generation/summary.json")
_output.parent.mkdir(parents=True, exist_ok=True)
_output.write_text(json.dumps(_summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"checks": _checks, "answer": response, "missing_context_answer": _negative_answer, "passed": _summary["passed"]}, indent=2))
assert _summary["passed"], "Generation checks failed; inspect the sanitized generation summary."
'''


def main():
    import nbformat
    from nbclient import NotebookClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-connectivity", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    for name in ("OPENAI_API_KEY", "OPENAI_API_BASE"):
        assert isinstance(config.get(name), str) and config[name].strip(), f"Missing field: {name}"
    secret = config["OPENAI_API_KEY"]

    def redact(value):
        if isinstance(value, str):
            return value.replace(secret, "[REDACTED]")
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        return value

    path = ROOT / "Full_Code_Notebook.ipynb"
    original = json.loads(path.read_text(encoding="utf-8"))
    notebook = nbformat.reads(path.read_text(encoding="utf-8"), as_version=4)
    checkpoints = ROOT / ".cache" / "notebook"
    checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, checkpoints / f"before-generation-execution-{time.time_ns()}.ipynb")
    completed = []

    def save_progress():
        current = json.loads(path.read_text(encoding="utf-8"))
        for index in completed:
            cell = current["cells"][index]
            if cell["source"] != original["cells"][index]["source"]:
                raise RuntimeError("A target cell changed during execution; its outputs were not overwritten.")
            for key in ("outputs", "execution_count"):
                cell[key] = redact(notebook.cells[index][key])
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
            for output in cell.outputs:
                if output.output_type == "stream":
                    print(redact(output.text), end="", flush=True)
        finally:
            notebook.cells.pop()

    try:
        with client.setup_kernel():
            print("Restoring imports, cached embedding model, and verified FAISS index...", flush=True)
            run_hidden("\n".join((RESTORE_IMPORTS, notebook.cells[46].source,
                                  notebook.cells[65].source, RESTORE_INDEX,
                                  notebook.cells[72].source)))
            run_hidden("_generation_live_requests = 3\n_generation_cumulative_requests = 3\n_generation_connectivity_reused = False")
            expected_requests = 2 if args.reuse_connectivity else 3
            print(f"Executing generation cells; {expected_requests} small live API requests are expected.", flush=True)
            numbers = (82, 85, 90, 93, 95, 96) if args.reuse_connectivity else (82, 85, 86, 90, 93, 95, 96)
            for number in numbers:
                index = number - 1
                print(f"Running notebook cell {number}...", flush=True)
                started = time.monotonic()
                try:
                    client.execute_cell(notebook.cells[index], index)
                except Exception as error:
                    # SDK error strings can echo credentials; omit them from saved outputs.
                    notebook.cells[index].outputs = [nbformat.v4.new_output(
                        "error", ename=type(error).__name__,
                        evalue="Cell execution failed; provider details omitted to protect credentials.",
                        traceback=[],
                    )]
                    raise
                finally:
                    completed.append(index)
                    save_progress()
                print(f"Completed cell {number} in {time.monotonic() - started:.1f}s.", flush=True)
                if number == 85:
                    run_hidden(VALIDATE_CONFIGURATION)
                    if args.reuse_connectivity:
                        run_hidden(REUSE_CONNECTIVITY)
                elif number == 86:
                    run_hidden(VALIDATE_CONNECTIVITY)
                elif number == 90:
                    run_hidden(CAPTURE_GENERATION)
            run_hidden(VALIDATE_GENERATION)
    finally:
        save_progress()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAIL generation execution ({type(error).__name__}); exception details omitted to protect credentials.", flush=True)
        sys.exit(1)
