"""Tiny persisted UI-toggle store.

Some runtime toggles must survive a restart: a tray switch the user flips ON
should still be ON next launch (the M69 voice-lock gate is the first — a user
who locks Jarvis to enrolled voices expects it to STAY locked across a restart,
not silently reopen). Backed by %LOCALAPPDATA%\\Jarvis\\ui_state.json with the
same file-as-state discipline as reminders.json: atomic write, lock-serialized,
fail-soft (any read/write error degrades to the caller's default and is logged
— a persistence hiccup must never break the UI or the listen loop).

Keyed by name so a future persisted toggle is a one-liner:
    initial = ui_state.get_flag("speaker_gate", cfg.speaker_gate_enabled)
    ...on toggle: ui_state.set_flag("speaker_gate", now_on)

The env var stays the DEFAULT for a fresh install (no file yet); once the user
toggles via the tray, the persisted value wins — last user action is sticky.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

from src.atomic_io import atomic_write_text

# All reads/writes funnel through this lock — the tray toggle (pystray worker
# thread) and startup (main thread) can both touch the file.
_LOCK = threading.Lock()


def _store_path() -> Path:
    """%LOCALAPPDATA%\\Jarvis\\ui_state.json. Computed directly (LOCALAPPDATA is
    read each call, so tests can redirect it), mirroring reminders._store_path."""
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base / "ui_state.json"


def _load() -> dict:
    """The whole toggle dict, or {} on any error (missing file, bad JSON)."""
    try:
        with open(_store_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return {}


def get_flag(name: str, default: bool = False) -> bool:
    """Persisted boolean for `name`, or `default` if unset / not a bool."""
    with _LOCK:
        val = _load().get(name)
    return val if isinstance(val, bool) else default


def set_flag(name: str, value: bool) -> None:
    """Persist `name`=value (atomic temp-then-replace). Never raises."""
    with _LOCK:
        data = _load()
        data[name] = bool(value)
        try:
            # Durable + atomic — see src/atomic_io.py.
            atomic_write_text(_store_path(), json.dumps(data))
        except OSError as exc:
            print(f"[ui_state] save failed for {name!r}: {exc}", file=sys.stderr)
