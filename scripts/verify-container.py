"""Build/verify the backend Docker image without changing the evaluated pipeline.

The default checks the existing image and retrieval without paid API calls.
--build builds the backend first; --live adds exactly one answer request.
Only a uniquely named verification container is created and removed. The image
is retained. API credentials pass through process environment at runtime.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / ".cache/notebook/container_verification"

CONTAINER_CHECK = r'''
import hashlib, importlib.metadata as metadata, json, os, platform, subprocess, sys
from pathlib import Path
import faiss
import numpy as np
import torch
from huggingface_hub.constants import HF_HUB_CACHE

root = Path('/app')
manifest = json.loads((root / 'deployment_manifest.json').read_text())
for name, expected in manifest['artifact_hashes'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
index = faiss.read_index(str(root / 'faiss_index/index.faiss'))
assert (index.ntotal, index.d) == (139, 384)
vectors = index.reconstruct_n(0, index.ntotal)
assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
assert platform.python_version() == '3.13.15' and platform.system() == 'Linux'
assert torch.__version__ == '2.14.1+cpu' and torch.version.cuda is None
assert torch.tensor([1, 2]).sum().item() == 3
assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
snapshot = Path(HF_HUB_CACHE) / ('models--' + manifest['embedding_model'].replace('/', '--')) / 'snapshots' / manifest['embedding_revision']
assert snapshot.is_dir(), 'Build-time model cache missing.'
assert not (root / 'config.json').exists() and not (root / '.env').exists()
expected_versions = {
    'flask': '3.1.3', 'Werkzeug': '3.1.8', 'requests': '2.34.2', 'gunicorn': '26.2.0',
    'langchain-huggingface': '1.2.2', 'langchain-text-splitters': '1.1.2',
    'langchain-community': '0.4.2', 'langchain-openai': '1.6.1',
    'sentence-transformers': '6.1.0', 'transformers': '5.16.1',
    'faiss-cpu': '1.15.0', 'numpy': '2.5.4', 'torch': '2.14.1+cpu',
}
for name, expected in expected_versions.items():
    assert metadata.version(name) == expected, name
check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
assert check.returncode == 0, check.stdout + check.stderr
files = ['app.py', 'rag_utils.py', 'policy_retrieval.py', 'deployment_manifest.json',
         'requirements.txt', 'faiss_index/index.faiss', 'faiss_index/index.pkl']
result = {
    'python': platform.python_version(), 'platform': platform.platform(),
    'vectors': index.ntotal, 'dimensions': index.d,
    'unit_normalized_vectors': True, 'cpu_torch_calculation': True,
    'model_revision_cached': manifest['embedding_revision'], 'offline_model_runtime': True,
    'no_config_or_dotenv_in_image': True, 'required_package_versions': expected_versions,
    'pip_check': check.stdout.strip(),
    'installed_packages': {d.metadata['Name']: d.version for d in metadata.distributions()},
    'artifact_hashes': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
}
print('AURORA_CONTAINER_CHECK=' + json.dumps(result))
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_docker():
    candidates = [shutil.which("docker")]
    if os.name == "nt":
        candidates += [str(Path(os.environ.get("LOCALAPPDATA", "C:/Users/scott/AppData/Local")) /
                           "Programs/DockerDesktop/resources/bin/docker.exe"),
                       "C:/Program Files/Docker/Docker/resources/bin/docker.exe"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise RuntimeError("Docker CLI not found. Install/start Docker Desktop, then reopen the terminal.")


def main():
    import requests

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--image", default="aurorastays-backend:verified")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    docker = find_docker()
    child_env = os.environ.copy()
    child_env["PATH"] = str(Path(docker).parent) + os.pathsep + child_env.get("PATH", "")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    secrets = [config["OPENAI_API_KEY"], config["OPENAI_API_BASE"].strip()]
    for name in ("OPENAI_API_KEY", "OPENAI_API_BASE"):
        if args.live:
            child_env[name] = config[name].strip()
        else:
            child_env.pop(name, None)

    def redact(text):
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return text

    def call(*arguments, input_text=None, check=True, timeout=120):
        result = subprocess.run([docker, *arguments], input=input_text, env=child_env,
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                cwd=ROOT, timeout=timeout)
        if check and result.returncode:
            (OUTPUT / "docker-error.log").write_text(redact(result.stdout + result.stderr), encoding="utf-8")
            raise RuntimeError(f"Docker {arguments[0]} failed; sanitized details saved in docker-error.log.")
        return result

    protected = ROOT / ".cache/notebook/backend_preparation/protected_file_hashes.json"
    protected_hashes = json.loads(protected.read_text())
    for name, expected in protected_hashes.items():
        assert digest(ROOT / name) == expected, name
    version = json.loads(call("version", "--format", "{{json .}}").stdout)
    assert version["Server"]["Os"] == "linux"
    print(f"Docker access verified: client {version['Client']['Version']}, Linux engine {version['Server']['Version']}.", flush=True)
    if args.build:
        print("Building backend image; output is retained in container_verification/build.log.", flush=True)
        with (OUTPUT / "build.log").open("w", encoding="utf-8") as log:
            build = subprocess.run([docker, "build", "--progress=plain", "--tag", args.image, "backend_files"],
                                   cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
        if build.returncode:
            raise RuntimeError("Docker build failed; inspect build.log.")
    image = json.loads(call("image", "inspect", args.image).stdout)[0]
    assert image["Os"] == "linux" and image["Architecture"] == "amd64"
    assert image["Config"]["Cmd"] == ["gunicorn", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:7860", "app:create_app()"]
    assert not any(item.startswith(("OPENAI_API_KEY=", "OPENAI_API_BASE=")) for item in image["Config"].get("Env", []))
    assert secrets[0] not in json.dumps(image), "Credential present in built image metadata."

    container = None
    summary = {"complete": False, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "docker_cli": docker, "docker_client_version": version["Client"]["Version"],
               "docker_server_version": version["Server"]["Version"],
               "image": args.image, "image_id": image["Id"], "image_size_bytes": image["Size"],
               "image_created": image["Created"], "image_platform": "linux/amd64",
               "gunicorn_factory_command_verified": True, "no_api_credentials_baked_into_image": True,
               "live_api_requests_attempted": 0, "test_set_used_for_tuning_or_rescored": False}
    phase = "container startup"
    try:
        name = "aurorastays-verify-" + uuid.uuid4().hex[:12]
        arguments = ["run", "--detach", "--name", name, "--label", "aurorastays.purpose=verification",
                     "--publish", "127.0.0.1::7860"]
        if args.live:
            arguments += ["--env", "OPENAI_API_KEY", "--env", "OPENAI_API_BASE"]
        container = call(*arguments, args.image).stdout.strip()
        assert re.fullmatch(r"[a-f0-9]{64}", container)
        binding = call("port", container, "7860/tcp").stdout.strip()
        assert binding.startswith("127.0.0.1:") and "\n" not in binding
        base = "http://" + binding
        summary.update(container_id=container, temporary_local_url=base)
        print("Started temporary Gunicorn container; waiting for its model/index to load...", flush=True)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            try:
                ready = requests.get(base + "/", timeout=(2, 3))
                if ready.status_code == 200:
                    break
            except requests.RequestException:
                pass
            state = json.loads(call("inspect", "--format", "{{json .State}}", container).stdout)
            if not state["Running"]:
                raise RuntimeError("Container exited before its HTTP endpoint became ready.")
            time.sleep(2)
        else:
            raise RuntimeError("Container readiness timed out.")
        summary["root_http_status"] = ready.status_code
        print("Gunicorn HTTP endpoint ready.", flush=True)

        phase = "Linux runtime/index checks"
        probe = call("exec", "--interactive", container, "python", "-", input_text=CONTAINER_CHECK, timeout=180)
        line = next(line for line in probe.stdout.splitlines() if line.startswith("AURORA_CONTAINER_CHECK="))
        runtime = json.loads(line.split("=", 1)[1])
        for name, expected in runtime["artifact_hashes"].items():
            assert digest(ROOT / "backend_files" / name) == expected, name
        summary["runtime_checks"] = runtime
        print("PASS Linux dependencies, CPU Torch/FAISS, cached model revision, and artifact integrity.", flush=True)

        phase = "HTTP retrieval parity"
        matches = {}
        for split in ("evaluation", "test_evaluation"):
            cases = json.loads((ROOT / ".cache/notebook" / split / "test_cases.json").read_text(encoding="utf-8"))
            for index, case in enumerate(cases, 1):
                result = requests.post(base + "/v1/relevant_chunks", json={"query": case["input"], "k": 4}, timeout=30)
                assert result.status_code == 200
                assert result.json()["relevant_chunks"] == case["retrieval_context"], f"Retrieval mismatch: {split} case {index}."
            matches[split] = len(cases)
        summary["retrieval_context_replay_matches"] = matches
        invalid_statuses = {}
        for endpoint in ("relevant_chunks", "answer", "answer_with_relevant_chunks"):
            invalid = requests.post(base + "/v1/" + endpoint, json={"query": ""}, timeout=10)
            assert invalid.status_code == 400 and "error" in invalid.json()
            invalid_statuses[endpoint] = invalid.status_code
        summary["invalid_request_statuses"] = invalid_statuses
        print("PASS all 20 scored contexts reproduced through container HTTP; invalid requests rejected.", flush=True)

        if args.live:
            phase = "one live answer request"
            summary["live_api_requests_attempted"] = 1
            (OUTPUT / "progress.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            result = requests.post(base + "/v1/answer_with_relevant_chunks",
                                   json={"query": "What are the pet fees at ASH-CHI Loop?"}, timeout=90)
            assert result.status_code == 200, "Container generation did not return HTTP 200."
            payload = result.json()
            answer = payload["answer"]
            assert "$50" in answer and "$150" in answer
            assert "night" in answer.lower() and "stay" in answer.lower() and ".pdf" in answer
            assert payload["k"] == 4 and payload["model_name"] == "gpt-4o-mini" and payload["max_tokens"] == 512
            summary["live_answer"] = {"http_status": result.status_code, **payload}
            print("PASS live container answer: Chicago nightly pet fee and stay cap, with source citation.", flush=True)

        phase = "final integrity checks"
        for name, expected in protected_hashes.items():
            assert digest(ROOT / name) == expected, name
        summary["frozen_configuration_and_scored_results_unchanged"] = True
        summary["complete"] = True
    except Exception as error:
        summary["error_phase"] = phase
        summary["error_type"] = type(error).__name__
        summary["error"] = redact(str(error))
        raise
    finally:
        if container:
            logs = call("logs", container, check=False)
            (OUTPUT / "container.log").write_text(redact(logs.stdout + logs.stderr), encoding="utf-8")
            # Only remove the unique container ID returned by this script's run.
            removed = call("rm", "--force", container, check=False)
            summary["temporary_container_removed"] = removed.returncode == 0
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        serialized = json.dumps(summary, indent=2) + "\n"
        assert secrets[0] not in serialized
        (OUTPUT / "summary.json").write_text(serialized, encoding="utf-8")
    assert summary["temporary_container_removed"], "Verification container cleanup failed."
    print("PASS container verification complete; temporary container removed and image retained.", flush=True)
    print(f"Evidence: {OUTPUT / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAIL container verification ({type(error).__name__}); inspect sanitized verification artifacts.", flush=True)
        sys.exit(1)
