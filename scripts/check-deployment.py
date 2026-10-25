"""Check deployment artifacts using only the Python standard library; no API calls."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend_files"


def check():
    manifest = json.loads((BACKEND / "deployment_manifest.json").read_text())
    for name, expected in manifest["artifact_hashes"].items():
        path = BACKEND / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or changed frozen deployment artifact: {name}")
    for directory in (BACKEND, ROOT / "frontend_files"):
        for name in ("app.py", "requirements.txt", "Dockerfile", ".dockerignore"):
            if not (directory / name).is_file():
                raise ValueError(f"Missing deployment file: {directory.name}/{name}")
        for name in ("config.json", ".env"):
            if (directory / name).exists():
                raise ValueError(f"Remove credential file from build context: {directory.name}/{name}")
    config = json.loads((ROOT / ".devcontainer/devcontainer.json").read_text())
    if config["forwardPorts"] != [8501]:
        raise ValueError("Forward only the frontend port by default.")
    for name in ("scripts/start-codespaces.py", "scripts/smoke-deployment.py", "compose.yaml"):
        if not (ROOT / name).is_file():
            raise ValueError(f"Missing deployment file: {name}")
    print(f"PASS: frozen FAISS {manifest['vectors']} x {manifest['dimensions']}, retrieval module, application files, and devcontainer configuration.")
    return manifest


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, KeyError) as error:
        print(f"Deployment check failed: {error}", file=sys.stderr)
        sys.exit(1)
