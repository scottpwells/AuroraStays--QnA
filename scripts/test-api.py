"""Send three small API probes through the notebook integrations; never print keys."""

import json
import logging
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
for name, value in {
    "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
    "DEEPEVAL_UPDATE_WARNING_OPT_OUT": "YES",
    "DEEPEVAL_RETRY_MAX_ATTEMPTS": "1",
    "HF_HOME": str(ROOT / ".cache" / "huggingface"),
    "IPYTHONDIR": str(ROOT / ".jupyter" / "ipython"),
}.items():
    os.environ[name] = value
logging.disable(logging.CRITICAL)


def main():
    try:
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        print("FAIL config.json is missing or invalid; configuration values were not printed.")
        return 1
    for key in ("OPENAI_API_KEY", "OPENAI_API_BASE"):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip() or value.startswith("REPLACE_"):
            print(f"FAIL missing configuration field: {key}")
            return 1
    endpoint = urlsplit(config["OPENAI_API_BASE"].strip())
    if (endpoint.scheme != "https" or not endpoint.hostname or endpoint.username
            or endpoint.password or endpoint.query or endpoint.fragment):
        print("FAIL API base must use HTTPS without embedded credentials or query parameters.")
        return 1

    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    os.environ["OPENAI_API_BASE"] = config["OPENAI_API_BASE"].strip()
    print("Configuration loaded; importing notebook API integrations.", flush=True)
    from langchain_openai import ChatOpenAI
    from deepeval.models import OpenAIModel
    from pydantic import BaseModel

    results = []

    def probe(label, model, max_output_tokens, action):
        started = time.monotonic()
        record = {"test": label, "model": model, "max_output_tokens": max_output_tokens}
        try:
            observed = action()
            record["passed"] = observed.strip() == "API_OK"
            # Only the expected marker is recorded, never arbitrary provider text.
            record["response"] = "API_OK" if record["passed"] else "unexpected response"
        except Exception as error:
            record["passed"] = False
            record["error_type"] = type(error).__name__
            status = getattr(error, "status_code", None)
            if isinstance(status, int):
                record["http_status"] = status
            # Provider error strings can echo credentials, so do not print them.
            code = getattr(error, "code", None)
            if code in {
                "invalid_api_key", "insufficient_quota", "model_not_found",
                "rate_limit_exceeded", "invalid_request_error", "permission_denied",
            }:
                record["error_code"] = code
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        results.append(record)
        print(json.dumps(record), flush=True)
        return record["passed"]

    def generation():
        model = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, top_p=1, max_tokens=32,
            timeout=30, max_retries=0,
        )
        return model.invoke("Reply with exactly API_OK.").content

    def judge(max_tokens):
        return OpenAIModel(
            model="gpt-4o", api_key=config["OPENAI_API_KEY"],
            base_url=config["OPENAI_API_BASE"].strip(), temperature=0,
            generation_kwargs={"max_tokens": max_tokens}, timeout=30, max_retries=0,
        )

    def evaluation():
        response, _ = judge(32).generate("Reply with exactly API_OK.")
        return response

    class ApiProbe(BaseModel):
        status: str

    def structured_evaluation():
        response, _ = judge(64).generate(
            "Return status equal to API_OK.", schema=ApiProbe,
        )
        return response.status

    probe("LangChain generation", "gpt-4o-mini", 32, generation)
    judge_passed = probe("DeepEval evaluation model", "gpt-4o", 32, evaluation)
    if judge_passed:
        probe("DeepEval structured output", "gpt-4o", 64, structured_evaluation)

    passed = len(results) == 3 and all(record["passed"] for record in results)
    report = {
        "tested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": passed,
        "tests": results,
    }
    output = ROOT / ".cache" / "audit" / "api-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"RESULT {'PASS' if passed else 'FAIL'}: {sum(r['passed'] for r in results)}/{len(results)} requests passed.", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
