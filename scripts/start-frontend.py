"""Start the local Streamlit policy desk at http://127.0.0.1:8501.

Start scripts/start-backend.py in another terminal first, or set BACKEND_URL.
The frontend does not load API credentials.
"""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    os.environ.setdefault("BACKEND_URL", "http://127.0.0.1:7860")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    sys.exit(subprocess.call([
        sys.executable, "-m", "streamlit", "run", str(ROOT / "frontend_files/app.py"),
        "--server.address=127.0.0.1", "--server.port=8501",
    ], cwd=ROOT))
