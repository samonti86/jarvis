r"""Durable store for long-horizon background tasks (M91).

WHY A STORE AT ALL:
A background task outlives the process that started it. Jarvis runs under
`jarvis_watchdog.pyw` (respawn on crash) and `update_jarvis` restarts it
deliberately, so "the process stayed up for the whole task" is not an
assumption this codebase gets to make — restarts are the normal operating
pattern, not an edge case. The session itself lives server-side; THIS FILE IS
THE LINK BACK TO IT. On startup Jarvis re-reads the store and resumes tracking
anything still running.

TWO STATE FLAGS, NOT ONE:
`status` is what the *agent* is doing; `delivered` is whether the *user* has
been told. They are separate on purpose. A task that finishes at 03:00 is
`done` immediately but must not speak — quiet hours defer it, and the morning
briefing picks up whatever is done-but-undelivered. Collapsing the two would
mean a restart between "finished" and "spoken" silently loses the report, which
is the one outcome an overnight task cannot have.

Written with the same discipline as src/reminders.py: fail-soft reads (a
corrupt file is logged and treated as empty rather than crashing a background
thread) and atomic, fsync'd writes via src/atomic_io.py — the target machine
has no UPS, so a torn file is a realistic failure mode, not a theoretical one.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

from src.atomic_io import atomic_write_text

# Terminal states — a task in one of these will never change status again.
_TERMINAL = frozenset({"done", "failed", "cancelled"})

# Serialises read-modify-write cycles. The poll thread and a voice turn can
# both touch the store; without this a concurrent cancel and a status update
# can clobber each other (last-writer-wins on the whole list).
_LOCK = threading.RLock()


def _store_path() -> Path:
    """%LOCALAPPDATA%\\Jarvis\\background_tasks.json. Computed directly (not via
    src.memory.default_base_dir) to keep this module import-light."""
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base / "background_tasks.json"


def _load() -> list[dict]:
    """Read the task list. Missing file → []. A corrupt file is logged and
    treated as empty rather than crashing the poll thread."""
    path = _store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — a bad file must not crash anything
        print(f"[bgtask] could not read {path.name}: {exc}", file=sys.stderr)
        return []
    if not isinstance(data, list):
        return []
    return [t for t in data if isinstance(t, dict)]


def _save(tasks: list[dict]) -> None:
    """Atomically + durably overwrite the store. A crash mid-write — including
    a hard power loss — leaves either the complete old file or the complete new
    one, never a torn or zero-length one. See src/atomic_io.py."""
    atomic_write_text(_store_path(), json.dumps(tasks, indent=2))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def add(prompt: str, *, session_id: str | None = None) -> dict:
    """Create a task record. `session_id` is filled in once the remote session
    exists; a record with none is a task that was stored before dispatch
    succeeded, which the reconciler can retry or fail."""
    with _LOCK:
        record = {
            "task_id": f"bgt_{uuid.uuid4().hex[:8]}",
            "session_id": session_id,
            "prompt": prompt,
            "status": "pending",
            "created_at": _now(),
            "finished_at": None,
            "result": None,
            "error": None,
            "delivered": False,
        }
        tasks = _load()
        tasks.append(record)
        _save(tasks)
        return dict(record)


def update(task_id: str, **fields) -> dict | None:
    """Patch one record. Returns the updated copy, or None if it's gone.

    Stamps `finished_at` automatically on the transition into a terminal
    state, so no caller has to remember to.
    """
    with _LOCK:
        tasks = _load()
        for t in tasks:
            if t.get("task_id") != task_id:
                continue
            was_terminal = t.get("status") in _TERMINAL
            t.update(fields)
            if not was_terminal and t.get("status") in _TERMINAL:
                t["finished_at"] = t.get("finished_at") or _now()
            _save(tasks)
            return dict(t)
        return None


def get(task_id: str) -> dict | None:
    with _LOCK:
        for t in _load():
            if t.get("task_id") == task_id:
                return dict(t)
        return None


def all_tasks() -> list[dict]:
    with _LOCK:
        return [dict(t) for t in _load()]


def active() -> list[dict]:
    """Tasks still being worked on — what the concurrency cap counts, and what
    gets re-attached after a restart."""
    with _LOCK:
        return [dict(t) for t in _load() if t.get("status") not in _TERMINAL]


def undelivered() -> list[dict]:
    """Finished but not yet reported. The morning briefing drains this, which
    is how an overnight completion reaches the user without waking them."""
    with _LOCK:
        return [
            dict(t) for t in _load()
            if t.get("status") in _TERMINAL and not t.get("delivered")
        ]


def mark_delivered(task_id: str) -> None:
    update(task_id, delivered=True)


def prune(keep_days: int = 30) -> int:
    """Drop delivered terminal records older than `keep_days`. Returns the
    number removed. Undelivered work is NEVER pruned regardless of age — an
    unreported result is the one thing this store exists to protect."""
    if keep_days <= 0:
        return 0
    cutoff = datetime.now().timestamp() - keep_days * 86400
    with _LOCK:
        tasks = _load()
        kept = []
        for t in tasks:
            if t.get("status") in _TERMINAL and t.get("delivered"):
                stamp = t.get("finished_at") or t.get("created_at") or ""
                try:
                    if datetime.fromisoformat(stamp).timestamp() < cutoff:
                        continue
                except ValueError:
                    pass  # unparseable timestamp → keep, don't guess
            kept.append(t)
        removed = len(tasks) - len(kept)
        if removed:
            _save(kept)
        return removed
