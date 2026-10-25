"""Verify the exported backend locally; --live adds one end-to-end API request."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
runpy.run_path(str(ROOT / "scripts/run-generation.py"))


def main(live=False):
    import faiss
    import numpy as np
    import requests
    from werkzeug.serving import make_server, WSGIRequestHandler

    sys.path.insert(0, str(ROOT / "backend_files"))
    import app as backend
    import rag_utils

    application = backend.create_app()
    store = application.config["VECTORSTORE"]
    assert store.index.ntotal == 139 and store.index.d == 384
    original = faiss.read_index(str(ROOT / ".cache/notebook/faiss_index/index.faiss"))
    vectors = store.index.reconstruct_n(0, store.index.ntotal)
    assert np.array_equal(vectors, original.reconstruct_n(0, original.ntotal))
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    replay_matches = {}
    for split in ("evaluation", "test_evaluation"):
        cases = json.loads((ROOT / ".cache/notebook" / split / "test_cases.json").read_text(encoding="utf-8"))
        for case in cases:
            assert rag_utils.retrieve(case["input"], k=4, vectorstore=store) == case["retrieval_context"]
        replay_matches[split] = len(cases)

    query = "What are the pet fees at ASH-CHI Loop?"
    response = application.test_client().post("/v1/relevant_chunks", json={"query": query})
    assert response.status_code == 200 and response.json["k"] == 4
    assert all(fact in "\n".join(response.json["relevant_chunks"]) for fact in ("ASH-CHI Loop", "$50 per night", "$150 stay cap"))
    live_result = None
    if live:
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
        for name in ("OPENAI_API_KEY", "OPENAI_API_BASE"):
            os.environ[name] = config[name].strip()
        captured = []
        original_generate = rag_utils.generate

        def capture(*args, **kwargs):
            message = original_generate(*args, **kwargs)
            captured.append({"usage": message.usage_metadata, "model": message.response_metadata.get("model_name"),
                             "finish_reason": message.response_metadata.get("finish_reason")})
            return message

        class QuietHandler(WSGIRequestHandler):
            def log_request(self, *args, **kwargs):
                pass

        server = make_server("127.0.0.1", 0, application, request_handler=QuietHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            assert requests.get(base + "/", timeout=10).status_code == 200
            invalid = requests.post(base + "/v1/answer", json={"query": ""}, timeout=10)
            assert invalid.status_code == 400
            with patch.object(rag_utils, "generate", side_effect=capture):
                result = requests.post(base + "/v1/answer_with_relevant_chunks", json={"query": query}, timeout=90)
            assert result.status_code == 200, "Backend generation did not return HTTP 200."
            payload = result.json()
            assert payload["k"] == 4 and payload["max_tokens"] == 512 and payload["model_name"] == "gpt-4o-mini"
            assert "$50" in payload["answer"] and "$150" in payload["answer"] and "stay" in payload["answer"].lower()
            assert len(captured) == 1 and captured[0]["finish_reason"] == "stop"
            live_result = {"query": query, "answer": payload["answer"], "relevant_chunks": payload["relevant_chunks"],
                           "http_status": result.status_code, "responses": captured, "live_api_requests": 1}
        finally:
            server.shutdown()
            worker.join(timeout=10)
            server.server_close()

    protected_path = ROOT / ".cache/notebook/backend_preparation/protected_file_hashes.json"
    for name, expected in json.loads(protected_path.read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    backend_dir = ROOT / "backend_files"
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    for path in [*backend_dir.glob("*"), ROOT / "Full_Code_Notebook.ipynb"]:
        if path.is_file():
            assert config["OPENAI_API_KEY"] not in path.read_text(encoding="utf-8"), "Credential leaked into an application artifact."
    summary = {"checks": {"exported_index_equal_to_frozen_index": True, "unit_normalized_vectors": True,
                          "same_retrieved_contexts_as_scored_cases": replay_matches,
                          "real_retrieval_route": True, "frozen_configuration_and_scored_results_unchanged": True,
                          "no_credentials_in_backend_package": True},
               "vectors": store.index.ntotal, "dimensions": store.index.d,
               "live_http_check": live_result, "live_api_requests": 1 if live else 0,
               "container_built_or_deployed": False, "temporary_http_server_stopped": True,
               "artifact_hashes": {str(p.relative_to(backend_dir)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in backend_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts}}
    target = ROOT / ".cache/notebook/backend_preparation/summary.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    try:
        sys.exit(main(live=args.live))
    except Exception as error:
        print(f"FAIL backend verification ({type(error).__name__}); provider details omitted.", flush=True)
        sys.exit(1)
