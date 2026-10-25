"""Check both running services and real retrieval, without calling the LLM."""

import json
import os
from urllib.request import Request, urlopen


def request(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=30) as response:
        assert response.status == 200
        return response.read()


def main():
    backend = "http://127.0.0.1:" + os.environ.get("AURORA_BACKEND_PORT", "7860")
    frontend = "http://127.0.0.1:" + os.environ.get("AURORA_FRONTEND_PORT", "8501")
    request(backend + "/")
    request(frontend + "/")
    request(frontend + "/_stcore/health")
    result = json.loads(request(backend + "/v1/relevant_chunks",
                                {"query": "What are the pet fees at ASH-CHI Loop?", "k": 4}))
    contexts = "\n".join(result["relevant_chunks"])
    assert "$50 per night" in contexts and "$150 stay cap" in contexts
    print("PASS: frontend, health endpoint, backend, and frozen policy retrieval; no paid API calls.")


if __name__ == "__main__":
    main()
