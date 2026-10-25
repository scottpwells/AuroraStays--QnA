"""Start the application after Codespaces runtime secrets become available."""

import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def main():
    runpy.run_path(str(ROOT / "scripts/check-deployment.py"))["check"]()
    missing = [name for name in ("OPENAI_API_KEY", "OPENAI_API_BASE") if not os.environ.get(name, "").strip()]
    if missing:
        print("Application startup is waiting for Codespaces secrets: " + ", ".join(missing))
        print("Add them in GitHub Settings > Codespaces, grant this repository access, then stop and restart this Codespace.")
        print("See DEPLOYMENT.txt. After secrets are available, run: python scripts/start-codespaces.py")
        return 0
    for _ in range(60):
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        if result.returncode == 0:
            break
        time.sleep(1)
    else:
        print("Docker is not ready. Rebuild the dev container, then retry startup.", file=sys.stderr)
        return 1
    print("Starting the backend and frontend. The first build downloads the pinned CPU model and libraries.", flush=True)
    result = subprocess.run(["docker", "compose", "-p", "aurorastays", "-f", str(ROOT / "compose.yaml"),
                             "up", "--build", "--detach", "--wait", "--wait-timeout", "240"], cwd=ROOT)
    if result.returncode:
        print("Startup failed. Inspect: docker compose -p aurorastays logs --tail 80", file=sys.stderr)
        return result.returncode
    subprocess.run([sys.executable, str(ROOT / "scripts/smoke-deployment.py")], cwd=ROOT, check=True)
    print("Open private port 8501 in the Ports panel. The backend uses internal Docker networking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
