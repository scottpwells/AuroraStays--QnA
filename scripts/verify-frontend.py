"""Verify the frontend and frozen backend together in temporary Docker containers.

--build builds the frontend image. --live submits one question through the actual
Streamlit AppTest flow inside that image; otherwise no paid API calls are made.
The two images are retained and only this run's containers/network are removed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / ".cache/notebook/frontend_preparation"
DOCKER = runpy.run_path(str(ROOT / "scripts/verify-container.py"))

FRONTEND_CHECK = r'''
import hashlib, importlib.metadata as metadata, json, os, subprocess, sys
from pathlib import Path
from unittest.mock import patch
import requests
from streamlit.testing.v1 import AppTest

live = LIVE_PLACEHOLDER
assert metadata.version('streamlit') == '1.64.0'
assert metadata.version('requests') == '2.34.2'
assert not os.environ.get('OPENAI_API_KEY') and not os.environ.get('OPENAI_API_BASE')
assert not Path('/app/config.json').exists() and not Path('/app/.env').exists()
assert os.environ['BACKEND_URL'] == 'http://backend:7860'
assert requests.get(os.environ['BACKEND_URL'] + '/', timeout=10).status_code == 200
check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
assert check.returncode == 0, check.stdout + check.stderr
app = AppTest.from_file('/app/app.py', default_timeout=100).run()
assert not app.exception
assert [item.value for item in app.slider] == [4, 0.0, 512]
result = {'streamlit_version': metadata.version('streamlit'),
          'requests_version': metadata.version('requests'), 'python': sys.version.split()[0],
          'backend_reachable_from_frontend': True, 'frontend_has_no_api_credentials': True,
          'initial_ui_has_no_exceptions': True, 'evaluated_defaults': [4, 0.0, 512],
          'pip_check': check.stdout.strip(),
          'artifact_hashes': {name: hashlib.sha256(Path('/app', name).read_bytes()).hexdigest()
                              for name in ('app.py', 'requirements.txt')}, 'live_api_requests': 0}
if live:
    calls = []
    original_post = requests.post
    def tracked(*args, **kwargs):
        response = original_post(*args, **kwargs)
        calls.append({'url': args[0], 'payload': kwargs.get('json'),
                      'timeout': kwargs.get('timeout'), 'status': response.status_code})
        return response
    with patch('requests.post', side_effect=tracked):
        app.text_area[0].input('What are the pet fees at ASH-CHI Loop?')
        app.radio[0].set_value('Answer with sources')
        app.button[0].click().run()
        assert not app.exception and not app.error
        answer = app.session_state['policy_result']['answer']
        assert '$50' in answer and '$150' in answer and 'night' in answer.lower() and 'stay' in answer.lower()
        assert any('Sources:' in item.value and '.pdf' in item.value for item in app.caption)
        assert 'Metadata:' not in ' '.join(item.value for item in app.markdown)
        app.slider[0].set_value(3).run()
        assert not app.exception and len(calls) == 1 and calls[0]['status'] == 200
        assert app.session_state['policy_result']['answer'] == answer
    result.update(live_api_requests=1, live_answer=answer, request=calls[0],
                  source_captions=[item.value for item in app.caption if item.value.startswith('Sources:')],
                  answer_survives_settings_rerun_without_extra_request=True)
print('AURORA_FRONTEND_CHECK=' + json.dumps(result))
'''


def main():
    import requests

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    docker = DOCKER["find_docker"]()
    child_env = os.environ.copy()
    child_env["PATH"] = str(Path(docker).parent) + os.pathsep + child_env.get("PATH", "")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    secrets = [config["OPENAI_API_KEY"], config["OPENAI_API_BASE"].strip()]
    for key in ("OPENAI_API_KEY", "OPENAI_API_BASE"):
        child_env.pop(key, None)
        if args.live:
            child_env[key] = config[key].strip()

    def redact(text):
        for value in secrets:
            text = text.replace(value, "[REDACTED]") if value else text
        return text

    def call(*arguments, input_text=None, check=True, timeout=180):
        result = subprocess.run([docker, *arguments], input=input_text, cwd=ROOT, env=child_env,
                                text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout)
        if check and result.returncode:
            (OUTPUT / "docker-error.log").write_text(redact(result.stdout + result.stderr), encoding="utf-8")
            raise RuntimeError(f"Docker {arguments[0]} failed; inspect docker-error.log.")
        return result

    if args.build:
        print("Building the frontend image; log: .cache/notebook/frontend_preparation/build.log", flush=True)
        with (OUTPUT / "build.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([docker, "build", "--progress=plain", "--tag", "aurorastays-frontend:verified", "frontend_files"],
                                    cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Frontend build failed; inspect build.log.")
    previous = json.loads((ROOT / ".cache/notebook/container_verification/summary.json").read_text())
    frontend_image = json.loads(call("image", "inspect", "aurorastays-frontend:verified").stdout)[0]
    backend_image_id = previous["image_id"]
    call("image", "inspect", backend_image_id)
    assert frontend_image["Os"] == "linux"
    assert not any(value.startswith(("OPENAI_API_KEY=", "OPENAI_API_BASE=")) for value in frontend_image["Config"].get("Env", []))
    protected = json.loads((OUTPUT / "protected_file_hashes.json").read_text())
    for name, expected in protected.items():
        assert DOCKER["digest"](ROOT / name) == expected, name
    summary = {"complete": False, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "frontend_image": "aurorastays-frontend:verified", "frontend_image_id": frontend_image["Id"],
               "backend_image_id": backend_image_id, "frontend_image_size_bytes": frontend_image["Size"],
               "live_api_requests_attempted": 0, "no_credentials_in_frontend_image_config": True}
    network = None
    containers = []
    phase = "container startup"
    try:
        suffix = uuid.uuid4().hex[:12]
        network = call("network", "create", "--label", "aurorastays.purpose=frontend-verification", "aurorastays-ui-" + suffix).stdout.strip()
        backend_args = ["run", "--detach", "--name", "aurorastays-ui-backend-" + suffix,
                        "--network", network, "--network-alias", "backend", "--publish", "127.0.0.1::7860"]
        if args.live:
            backend_args += ["--env", "OPENAI_API_KEY", "--env", "OPENAI_API_BASE"]
        backend = call(*backend_args, backend_image_id).stdout.strip()
        containers.append(backend)
        frontend = call("run", "--detach", "--name", "aurorastays-ui-frontend-" + suffix,
                        "--network", network, "--publish", "127.0.0.1::8501",
                        "--env", "BACKEND_URL=http://backend:7860", "--env", "NO_PROXY=backend,localhost,127.0.0.1",
                        frontend_image["Id"]).stdout.strip()
        containers.append(frontend)
        urls = {}
        for name, identifier, port, health in (("backend", backend, "7860/tcp", "/"),
                                               ("frontend", frontend, "8501/tcp", "/_stcore/health")):
            binding = call("port", identifier, port).stdout.strip()
            assert binding.startswith("127.0.0.1:")
            url = "http://" + binding
            urls[name] = url
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                try:
                    if requests.get(url + health, timeout=(2, 3)).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                state = json.loads(call("inspect", "--format", "{{json .State}}", identifier).stdout)
                assert state["Running"], f"{name} container exited."
                time.sleep(2)
            else:
                raise RuntimeError(f"{name} readiness timed out.")
            print(f"PASS {name} HTTP readiness.", flush=True)
        assert requests.get(urls["frontend"] + "/", timeout=10).status_code == 200
        summary["http_readiness"] = {"backend": 200, "frontend": 200, "frontend_health": 200}
        phase = "container UI/runtime check"
        summary["live_api_requests_attempted"] = 1 if args.live else 0
        (OUTPUT / "progress.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        check = call("exec", "--interactive", frontend, "python", "-",
                     input_text=FRONTEND_CHECK.replace("LIVE_PLACEHOLDER", repr(args.live)), timeout=180)
        line = next(line for line in check.stdout.splitlines() if line.startswith("AURORA_FRONTEND_CHECK="))
        ui = json.loads(line.split("=", 1)[1])
        for name, expected in ui["artifact_hashes"].items():
            assert DOCKER["digest"](ROOT / "frontend_files" / name) == expected
        summary["ui_verification"] = ui
        print("PASS Streamlit UI, internal Docker backend connection, dependencies, and packaged files.", flush=True)
        if args.live:
            print("PASS one live question through the frontend: answer, sources, and no duplicate request on rerun.", flush=True)
        for name, expected in protected.items():
            assert DOCKER["digest"](ROOT / name) == expected, name
        summary["frozen_pipeline_and_all_scored_artifacts_unchanged"] = True
        summary["complete"] = True
    except Exception as error:
        summary.update(error_phase=phase, error_type=type(error).__name__, error=redact(str(error)))
        raise
    finally:
        removed = []
        for index, identifier in enumerate(reversed(containers)):
            logs = call("logs", identifier, check=False)
            (OUTPUT / f"container-{index}.log").write_text(redact(logs.stdout + logs.stderr), encoding="utf-8")
            removed.append(call("rm", "--force", identifier, check=False).returncode == 0)
        summary["temporary_containers_removed"] = len(removed) == 2 and all(removed)
        summary["temporary_network_removed"] = bool(network) and call("network", "rm", network, check=False).returncode == 0
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        content = json.dumps(summary, indent=2) + "\n"
        assert secrets[0] not in content
        (OUTPUT / "summary.json").write_text(content, encoding="utf-8")
    assert summary["temporary_containers_removed"] and summary["temporary_network_removed"]
    print("PASS frontend verification complete; temporary containers/network removed; both images retained.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAIL frontend verification ({type(error).__name__}); inspect sanitized verification artifacts.", flush=True)
        sys.exit(1)
