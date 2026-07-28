"""Process-wide stdout/stderr wiring for the three launch modes.

Extracted from main.py (2026-07-28) as part of breaking up a 2,800-line entry
point. This is the most self-contained piece of that file: it depends on
nothing in Jarvis except `src.logfile`, and nothing in Jarvis depends on it
except the one call at startup.

The three modes it reconciles:

- `pythonw jarvis.pyw` (production) — the launcher has already rotated and
  opened the log, replaced stdout/stderr, and exported JARVIS_LOG_PATH. We
  must not touch the streams again; just report the path.
- `python main.py` (console/dev) — tee to both the terminal and the file, so
  output is live *and* persisted. Only the file side is timestamped; the
  terminal stays clean to read.
- `pythonw main.py` (rare, no launcher) — no console exists, so redirect only.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path


class TeeStream:
    """Forwards writes to multiple underlying streams. Minimal surface — only
    the methods print() actually touches. We deliberately don't expose
    .fileno() because libraries that introspect it might try to dup our fd
    and bypass the tee."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False


def setup_logging() -> Path:
    """Always write to %LOCALAPPDATA%\\Jarvis\\jarvis.log. Behavior by mode:

    - jarvis.pyw launcher: redirected stdout/stderr at import time and set
      JARVIS_LOG_PATH. We respect that and just return the path.
    - python main.py (console): tee stdout/stderr to both terminal *and* file.
      You see live output AND the conversation is persisted.
    - pythonw main.py (rare; no launcher): no console to tee to, redirect only.
    """
    from src.logfile import rotate_if_needed, TimestampStream

    log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "jarvis.log"

    env_path = os.environ.get("JARVIS_LOG_PATH")
    if env_path:
        # jarvis.pyw already rotated + opened the file + replaced stdout/stderr.
        return Path(env_path)

    # Rotate BEFORE opening — Windows can't rename a file with an open handle.
    rotate_if_needed(log_path)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    log_file.write(f"\n--- Jarvis started {datetime.now().isoformat(timespec='seconds')} ---\n")
    log_file.flush()

    if sys.stdout is None or sys.stderr is None:
        # pythonw without the launcher — no console to tee to.
        stamped = TimestampStream(log_file)
        sys.stdout = stamped
        sys.stderr = stamped
    else:
        # Console mode — tee both. Live terminal output + persistent file. Only
        # the FILE side is timestamped (one shared wrapper so stdout/stderr share
        # the line state); the live terminal stays clean for dev readability.
        stamped_file = TimestampStream(log_file)
        sys.stdout = TeeStream(sys.stdout, stamped_file)
        sys.stderr = TeeStream(sys.stderr, stamped_file)

    return log_path
