"""Start the prepared backend on http://127.0.0.1:7860 using local credentials."""

import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent.parent
runpy.run_path(str(ROOT / "scripts/run-generation.py"))  # Local/offline model cache settings.
config_path = ROOT / "config.json"
if config_path.exists():
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    for name in ("OPENAI_API_KEY", "OPENAI_API_BASE"):
        os.environ.setdefault(name, config[name].strip())
if not all(os.environ.get(name, "").strip() for name in ("OPENAI_API_KEY", "OPENAI_API_BASE")):
    raise SystemExit("Set OPENAI_API_KEY and OPENAI_API_BASE, or configure local config.json.")
sys.path.insert(0, str(ROOT / "backend_files"))
from app import create_app

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=7860, debug=False)
