"""Reminders & timers (M53) — Jarvis's first 'act at a future time' capability.

Everything Jarvis did before this was present-tense: a question comes in, an
answer goes out. This module gives him a future tense — "remind me in 20
minutes to check the printer" schedules a spoken reminder that fires later,
on its own, with no further prompting.

Design decisions (settled with the user before the build):
  - **Tools, not a deterministic intent matcher.** `set_reminder` /
    `list_reminders` / `cancel_reminder` are M13-pattern custom tools. The
    natural-language time parsing — "in an hour and a half", "at 6:30",
    "tomorrow morning" — is exactly Claude's strength; a regex matcher would
    be brittle. (Contrast M51's `_is_dismissal`, which is deterministic
    *because* it gates loop control and must be certain. Reminder-setting is
    a capability, so the model parses.)
  - **The file is the IPC.** `set_reminder` just appends to reminders.json;
    the scheduler thread re-reads it each poll. Tool and scheduler are fully
    decoupled — no runtime object threaded through stream_response. One lock
    here guards every read/write; the JSON IS the source of truth and
    survives a restart.
  - **A poll-loop scheduler, not N threading.Timers.** One daemon thread
    wakes every ~10 s and fires what's due. A poll loop trivially handles app
    restart (reminders that came due while Jarvis was off fire on the first
    poll) and clock changes — N timers handle neither.
  - **delay_seconds XOR at.** Relative requests ("in 20 min") need no clock
    knowledge — the tool computes from now(). Absolute requests carry an ISO
    datetime (Claude knows today's date; the system prompt deliberately does
    NOT carry the time-of-day, since that would bust the prompt cache every
    minute).

Storage: %LOCALAPPDATA%\\Jarvis\\reminders.json — a JSON list, atomically
written (temp + os.replace) so a crash mid-write can never corrupt it.

Defensive contract, same as the other tool modules: every public entry point
never raises and always returns a readable string. A corrupt store, a bad
time, an empty message — all become voice-friendly strings, never an
exception into the listen loop or the scheduler thread.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

# ISO format used for every stored timestamp. Fixed-width and zero-padded, so
# lexicographic string comparison of two of these IS chronological comparison
# — pop_due relies on that to avoid parsing on the hot path.
_ISO = "%Y-%m-%dT%H:%M:%S"

_MAX_MESSAGE_LEN = 500       # a reminder is spoken aloud; cap absurd input
_MAX_HORIZON_DAYS = 366      # further out than this is almost certainly a parse error

# All reads/writes of reminders.json funnel through this lock — the tool
# executors run on the listen/text thread, the scheduler on its own thread.
_LOCK = threading.Lock()


# --- storage ---------------------------------------------------------------

def _store_path() -> Path:
    """%LOCALAPPDATA%\\Jarvis\\reminders.json. Computed directly (not via
    src.memory.default_base_dir) to keep this module import-light."""
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base / "reminders.json"


def _load() -> list[dict]:
    """Read the reminder list. Missing file → []. A corrupt file is logged
    and treated as empty rather than crashing the scheduler — the store must
    fail soft."""
    path = _store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — a bad file must not crash anything
        print(f"[reminders] could not read {path.name}: {exc}", file=sys.stderr)
        return []
    if not isinstance(data, list):
        return []
    return [r for r in data if isinstance(r, dict)]


def _save(reminders: list[dict]) -> None:
    """Atomically overwrite the store — temp file + os.replace (atomic on
    Windows). A crash mid-write leaves either the old file or the new one
    intact, never a half-written one."""
    path = _store_path()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(reminders, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def add(message: str, fire_at: datetime) -> dict:
    """Append one reminder and return the stored record."""
    rec = {
        "id": "r" + uuid.uuid4().hex[:6],
        "message": message,
        "fire_at": fire_at.strftime(_ISO),
        "created_at": datetime.now().strftime(_ISO),
    }
    with _LOCK:
        items = _load()
        items.append(rec)
        _save(items)
    return rec


def list_pending() -> list[dict]:
    """All pending reminders, soonest first."""
    with _LOCK:
        items = _load()
    items.sort(key=lambda r: r.get("fire_at", ""))
    return items


def cancel(rid: str | None = None, query: str | None = None) -> dict | list[dict] | None:
    """Cancel by exact id, or by a case-insensitive substring of the message.
    Returns the cancelled record; None if nothing matched; the list of
    candidates (nothing cancelled) if a query was ambiguous — the caller
    asks the user to disambiguate."""
    with _LOCK:
        items = _load()
        target: dict | None = None
        if rid:
            target = next((r for r in items if r.get("id") == rid), None)
        elif query:
            q = query.strip().lower()
            matches = [r for r in items if q in r.get("message", "").lower()]
            if len(matches) > 1:
                return matches
            target = matches[0] if matches else None
        if target is None:
            return None
        _save([r for r in items if r is not target])
        return target


def pop_due(now: datetime) -> list[dict]:
    """Atomically remove and return every reminder due at or before `now`,
    soonest first. The scheduler's hot path — string compare, no parsing
    (see the _ISO note above)."""
    cutoff = now.strftime(_ISO)
    with _LOCK:
        items = _load()
        due = [r for r in items if r.get("fire_at") and r["fire_at"] <= cutoff]
        if due:
            _save([r for r in items if r not in due])
    due.sort(key=lambda r: r.get("fire_at", ""))
    return due


# --- time helpers ----------------------------------------------------------

def _parse_iso(s: str) -> datetime | None:
    """Parse a stored or Claude-supplied ISO datetime to a naive local
    datetime. A tz-aware value (Claude occasionally adds one) is converted to
    local and the tzinfo dropped, so it's always comparable with now()."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def _human_dt(dt: datetime) -> str:
    """'Thu May 22, 6:00 PM' — for the tool's confirmation text. Claude
    re-voices it anyway; this just has to be unambiguous and readable."""
    return dt.strftime("%a %b %d, %I:%M %p").replace(" 0", " ")


# --- Anthropic tool definitions --------------------------------------------

SET_REMINDER_TOOL = {
    "name": "set_reminder",
    "description": (
        "Schedule a one-off spoken reminder or timer for a future time. Use "
        "this whenever the user asks to be reminded of something later, or to "
        "set a timer — 'remind me in 20 minutes to check the printer', 'set a "
        "timer for 10 minutes', 'remind me at 6 to call her back'. Give "
        "EITHER delay_seconds (a relative time like 'in 20 minutes' — you "
        "compute the seconds) OR at (an absolute time as an ISO 8601 "
        "datetime; you know today's date). Never both. `message` is what "
        "Jarvis speaks aloud when it fires, phrased as the task itself "
        "('check the printer', 'your 10-minute timer is up'). Confirm the "
        "reminder briefly once it's set."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": (
                    "What Jarvis says aloud when it fires — the task itself, "
                    "e.g. 'check the printer'. Do not include 'remind you to'."
                ),
            },
            "delay_seconds": {
                "type": "integer",
                "description": (
                    "For a RELATIVE time: whole seconds from now until it "
                    "fires. Use for 'in N minutes/hours'. Omit if using `at`."
                ),
            },
            "at": {
                "type": "string",
                "description": (
                    "For an ABSOLUTE time: ISO 8601 local datetime, e.g. "
                    "'2026-05-21T18:00:00'. Use for 'at 6pm', 'tomorrow at "
                    "9'. Omit if using `delay_seconds`."
                ),
            },
        },
        "required": ["message"],
    },
}

LIST_REMINDERS_TOOL = {
    "name": "list_reminders",
    "description": (
        "List the user's pending reminders and timers — what they are and "
        "when each fires. Use for 'what reminders do I have?', 'what am I "
        "supposed to do later?', or before cancelling when you need an id."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

CANCEL_REMINDER_TOOL = {
    "name": "cancel_reminder",
    "description": (
        "Cancel a pending reminder. Identify it by `id` (exact) or by "
        "`query` — a substring of its message, e.g. 'printer' to cancel the "
        "'check the printer' reminder. Pass `query` directly when the user "
        "names the reminder ('cancel the printer reminder'); call "
        "list_reminders first only if the query would be ambiguous."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Exact reminder id (from list_reminders).",
            },
            "query": {
                "type": "string",
                "description": (
                    "Case-insensitive substring of the reminder's message."
                ),
            },
        },
        "required": [],
    },
}


# --- tool executors --------------------------------------------------------

def execute_set_reminder(params: dict) -> str:
    """Schedule a reminder. Never raises — every failure is a readable string."""
    message = (params.get("message") or "").strip()
    if not message:
        return "I need to know what to remind you about, sir."
    if len(message) > _MAX_MESSAGE_LEN:
        message = message[:_MAX_MESSAGE_LEN]

    delay = params.get("delay_seconds")
    at = (params.get("at") or "").strip()
    if delay is not None and at:
        return "Give me either a delay or an absolute time, sir — not both."

    now = datetime.now()
    if delay is not None:
        try:
            delay = int(delay)
        except (TypeError, ValueError):
            return "I couldn't read that delay, sir."
        if delay <= 0:
            return "That time isn't in the future, sir."
        fire_at = now + timedelta(seconds=delay)
    elif at:
        parsed = _parse_iso(at)
        if parsed is None:
            return "I couldn't read that time, sir."
        if parsed <= now:
            return (
                f"{_human_dt(parsed)} has already passed, sir — "
                f"shall I set it for another time?"
            )
        fire_at = parsed
    else:
        return "I need a time for the reminder, sir."

    if fire_at > now + timedelta(days=_MAX_HORIZON_DAYS):
        return "That's further out than I can schedule, sir."

    rec = add(message, fire_at)
    return (
        f"Reminder set — \"{message}\" — will fire at {_human_dt(fire_at)}. "
        f"id: {rec['id']}."
    )


def execute_list_reminders(params: dict) -> str:  # noqa: ARG001 — no params
    """Read back pending reminders. Never raises."""
    items = list_pending()
    if not items:
        return "You have no reminders set, sir."
    lines = [f"{len(items)} reminder(s) pending:"]
    for r in items:
        dt = _parse_iso(r.get("fire_at", ""))
        when = _human_dt(dt) if dt else r.get("fire_at", "unknown time")
        lines.append(
            f"- \"{r.get('message', '?')}\" at {when} (id: {r.get('id', '?')})"
        )
    return "\n".join(lines)


def execute_cancel_reminder(params: dict) -> str:
    """Cancel a reminder by id or message substring. Never raises."""
    rid = (params.get("id") or "").strip()
    query = (params.get("query") or "").strip()
    if not rid and not query:
        return (
            "Which reminder should I cancel, sir? Name it, or I can list "
            "them first."
        )
    result = cancel(rid=rid or None, query=query or None)
    if result is None:
        return "I couldn't find a matching reminder, sir."
    if isinstance(result, list):
        opts = "; ".join(
            f"\"{r.get('message', '?')}\" (id: {r.get('id', '?')})"
            for r in result
        )
        return (
            f"Several reminders match that, sir — which one? {opts}"
        )
    return f"Cancelled, sir — \"{result.get('message', '?')}\" will no longer fire."


# --- scheduler -------------------------------------------------------------

def _fire_text(rec: dict, now: datetime) -> str:
    """The line Jarvis speaks when a reminder fires. A reminder more than ~90 s
    overdue (it came due while Jarvis was off, then fired on the first poll
    after restart) is flagged as belated so the user isn't misled about the
    time."""
    msg = (rec.get("message") or "").strip() or "your reminder"
    fire_at = _parse_iso(rec.get("fire_at", ""))
    if fire_at is not None and (now - fire_at).total_seconds() > 90:
        return f"Sir, a belated reminder — {msg}. This was due while I was away."
    return f"Sir, a reminder — {msg}."


def run_scheduler(
    announce: Callable[[str], None],
    stop_event: threading.Event,
    poll_sec: float = 10.0,
) -> None:
    """Daemon loop: every ~poll_sec, fire any due reminders via `announce`
    (main.py's WASAPI-safe Announcer path). Polls immediately on start, so
    reminders that came due while Jarvis was off fire right after launch.
    Wrapped so a single bad poll can never kill the thread — a dead scheduler
    would silently drop every future reminder."""
    print("[reminders] scheduler thread started", file=sys.stderr)
    while not stop_event.is_set():
        try:
            now = datetime.now()
            for rec in pop_due(now):
                print(f"[reminders] firing {rec.get('id')}: {rec.get('message')}",
                      file=sys.stderr)
                announce(_fire_text(rec, now))
        except Exception as exc:  # noqa: BLE001 — keep the thread alive
            print(f"[reminders] scheduler poll failed: {exc}", file=sys.stderr)
        stop_event.wait(poll_sec)
    print("[reminders] scheduler thread exited", file=sys.stderr)
