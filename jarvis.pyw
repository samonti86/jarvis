"""Background-app launcher.

Windows associates `.pyw` with pythonw.exe (no console window). Double-click
this file to run Jarvis as a tray-only background app — no terminal, no
flashing console window. Logs go to %LOCALAPPDATA%\\Jarvis\\jarvis.log;
right-click the tray icon → 'Open log' to view.

For development with live console output, run `python main.py` instead.

Two boot-time concerns this file handles, in order:

1. **Wrong interpreter.** Windows hands .pyw files to the Python Launcher
   (py.exe), which always runs the *system* Python — not whichever venv
   lives beside the script. That means none of our deps are visible. We
   detect that mismatch by comparing sys.executable to the venv's pythonw
   and re-launch under the right interpreter via subprocess. Only stdlib
   imports are used here so the bootstrap works on any Python.

2. **No console for prints.** Under pythonw, sys.stdout is None, so any
   print() at module import time would crash. We swap stdout/stderr to a
   logfile *before* importing main, so all import-time output is captured.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_VENV_PYTHONW = _HERE / "venv" / "Scripts" / "pythonw.exe"


def _ensure_correct_interpreter() -> None:
    """If we're not running under the venv's pythonw, re-exec there and exit.

    Triggered when the user double-clicks jarvis.pyw and Windows hands it
    to the system Python instead of the venv's. Without this, imports of
    numpy/anthropic/openwakeword etc. fail and the process exits silently.
    """
    if not _VENV_PYTHONW.exists():
        return  # no venv — let the import errors surface naturally
    if Path(sys.executable).resolve() == _VENV_PYTHONW.resolve():
        return  # already under venv pythonw — proceed
    subprocess.Popen([str(_VENV_PYTHONW), str(_HERE / "jarvis.pyw")])
    sys.exit(0)


def _redirect_to_logfile() -> Path:
    log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "jarvis.log"
    f = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
    sys.stdout = f
    sys.stderr = f
    print(f"\n--- Jarvis started {datetime.now().isoformat(timespec='seconds')} ---")
    return log_path


_ensure_correct_interpreter()
_LOG_PATH = _redirect_to_logfile()
os.environ["JARVIS_LOG_PATH"] = str(_LOG_PATH)

from main import main  # noqa: E402


if __name__ == "__main__":
    main()
