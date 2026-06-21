"""Reminders & timers (M53) — Jarvis's first 'act at a future time' capability.

Everything Jarvis did before this was present-tense: a question comes in, an
answer goes out. This module gives him a future tense — "remind me in 20
minutes to check the printer" schedules a spoken reminder that fires later,
on its own, with no further prompting. M54 added recurrence: a reminder can
repeat on an interval, weekly, or monthly schedule.

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
  - **Recurrence — three kinds, no full calendar engine (M54).** `interval`
    (a fixed period — every N minutes/hours), `weekly` (a weekday set + a
    time-of-day — every day / weekday / Monday), `monthly` (a day-of-month +
    a time, clamped to the month's last day). weekly/monthly recompute the
    wall-clock time each occurrence so they stay correct across DST; interval
    is for sub-day periods where that's moot. A recurring reminder is re-armed
    on fire (same id, fire_at advanced) rather than removed — and the next
    slot is computed from now(), so a long outage yields exactly ONE catch-up
    fire, never a backlog stampede.

Storage: %LOCALAPPDATA%\\Jarvis\\reminders.json — a JSON list, atomically
written (temp + os.replace) so a crash mid-write can never corrupt it.

Defensive contract, same as the other tool modules: every public entry point
never raises and always returns a readable string. A corrupt store, a bad
time, an empty message — all become voice-friendly strings, never an
exception into the listen loop or the scheduler thread.
"""

from __future__ import annotations

import calendar
import json
import os
import sys
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from src.atomic_io import atomic_write_text

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
    """Atomically + durably overwrite the store. A crash mid-write — including a
    hard power loss (no UPS here) — leaves either the complete old file or the
    complete new one, never a torn or zero-length one. See src/atomic_io.py."""
    atomic_write_text(_store_path(), json.dumps(reminders, indent=2))


def add(
    message: str, fire_at: datetime,
    repeat: dict | None = None, action: str | None = None,
) -> dict:
    """Append one reminder and return the stored record. `repeat` (M54), when
    given, is a validated recurrence spec — its presence makes the reminder
    recurring (re-armed on fire instead of removed). `action` (M59), when set
    to "briefing", changes what happens on fire: instead of speaking the
    `message` text, the scheduler composes + announces the M55 morning
    briefing. Other actions may follow; unknown values are stored verbatim
    and treated as a no-op by the scheduler (forward-compatible)."""
    rec = {
        "id": "r" + uuid.uuid4().hex[:6],
        "message": message,
        "fire_at": fire_at.strftime(_ISO),
        "created_at": datetime.now().strftime(_ISO),
    }
    if repeat:
        rec["repeat"] = repeat
    if action:
        rec["action"] = action
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
    """Atomically collect every reminder due at or before `now`. A one-shot
    reminder is removed; a RECURRING one (M54) is re-armed in place — same id,
    fire_at advanced to its next occurrence after `now`. Computing the next
    slot from `now` (not the stale fire_at) means a long outage yields exactly
    ONE catch-up fire, never a backlog stampede. Returns the due records as
    they fired, soonest-first, for the scheduler to announce."""
    cutoff = now.strftime(_ISO)
    with _LOCK:
        items = _load()
        due = [r for r in items if r.get("fire_at") and r["fire_at"] <= cutoff]
        if not due:
            return []
        survivors = [r for r in items if r not in due]
        for r in due:
            repeat = r.get("repeat")
            if repeat:
                nxt = _next_occurrence(repeat, now)
                if nxt is not None:  # malformed spec → drop it, fail soft
                    survivors.append({**r, "fire_at": nxt.strftime(_ISO)})
        _save(survivors)
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


def _human_time(hhmm: str) -> str:
    """'07:00' → '7:00 AM'. For describing a recurrence's time-of-day."""
    p = _parse_hhmm(hhmm)
    if p is None:
        return hhmm
    return datetime(2000, 1, 1, p[0], p[1]).strftime("%I:%M %p").lstrip("0")


# --- recurrence (M54) ------------------------------------------------------

_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"]
_MIN_INTERVAL_SEC = 60  # tighter than this is reminder-spam, almost certainly an error


def _parse_hhmm(s: str | None) -> tuple[int, int] | None:
    """'09:00' / '9:00' → (9, 0). None if unparseable or out of range."""
    try:
        parts = str(s).strip().split(":")
        hh, mm = int(parts[0]), int(parts[1])
    except (ValueError, IndexError, AttributeError):
        return None
    return (hh, mm) if 0 <= hh <= 23 and 0 <= mm <= 59 else None


def _validate_repeat(repeat: dict) -> dict | str:
    """Validate + normalize a recurrence spec from the tool input. Returns a
    clean spec dict, or a voice-friendly error string for Claude. Never raises."""
    if not isinstance(repeat, dict):
        return "That recurrence didn't make sense, sir."
    kind = str(repeat.get("kind") or "").strip().lower()
    if kind == "interval":
        try:
            secs = int(repeat.get("interval_seconds"))
        except (TypeError, ValueError):
            return "I need a valid interval for a repeating reminder, sir."
        if secs < _MIN_INTERVAL_SEC:
            return "That interval is too short, sir — a minute is the tightest I'll repeat."
        return {"kind": "interval", "interval_seconds": secs}
    if kind == "weekly":
        try:
            days = sorted({int(d) for d in (repeat.get("weekdays") or [])})
        except (TypeError, ValueError):
            return "I couldn't read those weekdays, sir."
        if not days or any(d < 0 or d > 6 for d in days):
            return "I need valid weekdays — Monday to Sunday — for that, sir."
        hhmm = _parse_hhmm(repeat.get("time"))
        if hhmm is None:
            return "I need a valid time of day for that repeating reminder, sir."
        return {"kind": "weekly", "weekdays": days, "time": f"{hhmm[0]:02d}:{hhmm[1]:02d}"}
    if kind == "monthly":
        try:
            day = int(repeat.get("day"))
        except (TypeError, ValueError):
            return "I need a day of the month for that, sir."
        if day < 1 or day > 31:
            return "That day of the month isn't valid, sir."
        hhmm = _parse_hhmm(repeat.get("time"))
        if hhmm is None:
            return "I need a valid time of day for that monthly reminder, sir."
        return {"kind": "monthly", "day": day, "time": f"{hhmm[0]:02d}:{hhmm[1]:02d}"}
    return "I can repeat reminders by interval, weekly, or monthly, sir."


def _next_occurrence(repeat: dict, after: datetime) -> datetime | None:
    """The next time this recurrence fires, strictly after `after`. Used both
    for the first occurrence (after = now) and to re-arm a fired recurring
    reminder (after = now) — so a long outage yields exactly one catch-up
    fire. None on a malformed spec (callers fail soft)."""
    kind = repeat.get("kind")
    if kind == "interval":
        try:
            return after + timedelta(seconds=int(repeat["interval_seconds"]))
        except (KeyError, TypeError, ValueError):
            return None
    if kind == "weekly":
        hhmm = _parse_hhmm(repeat.get("time"))
        days = set(repeat.get("weekdays") or [])
        if hhmm is None or not days:
            return None
        for offset in range(0, 9):  # 0..8 — guaranteed to hit a chosen weekday
            d = (after + timedelta(days=offset)).replace(
                hour=hhmm[0], minute=hhmm[1], second=0, microsecond=0)
            if d > after and d.weekday() in days:
                return d
        return None
    if kind == "monthly":
        hhmm = _parse_hhmm(repeat.get("time"))
        try:
            day = int(repeat["day"])
        except (KeyError, TypeError, ValueError):
            return None
        if hhmm is None:
            return None
        y, m = after.year, after.month
        for _ in range(13):  # this month, then forward — always finds one
            last = calendar.monthrange(y, m)[1]  # clamp: 31 → the real last day
            d = datetime(y, m, min(day, last), hhmm[0], hhmm[1])
            if d > after:
                return d
            m, y = (1, y + 1) if m == 12 else (m + 1, y)
        return None
    return None


def _describe_repeat(repeat: dict) -> str:
    """A spoken-friendly description of a recurrence, for list_reminders."""
    kind = repeat.get("kind")
    if kind == "interval":
        secs = int(repeat.get("interval_seconds", 0))
        if secs >= 3600 and secs % 3600 == 0:
            n = secs // 3600
            return f"every {n} hour{'s' if n != 1 else ''}"
        n = max(1, secs // 60)
        return f"every {n} minute{'s' if n != 1 else ''}"
    if kind == "weekly":
        days = sorted(repeat.get("weekdays") or [])
        when = _human_time(repeat.get("time", ""))
        if days == [0, 1, 2, 3, 4, 5, 6]:
            return f"every day at {when}"
        if days == [0, 1, 2, 3, 4]:
            return f"every weekday at {when}"
        if days == [5, 6]:
            return f"every weekend at {when}"
        names = ", ".join(_WEEKDAY_NAMES[d] for d in days if 0 <= d <= 6)
        return f"every {names} at {when}"
    if kind == "monthly":
        return (f"on day {repeat.get('day')} of every month "
                f"at {_human_time(repeat.get('time', ''))}")
    return "on a schedule"


# --- Anthropic tool definitions --------------------------------------------

SET_REMINDER_TOOL = {
    "name": "set_reminder",
    "description": (
        "Schedule a spoken reminder or timer — one-off OR recurring. Use this "
        "whenever the user asks to be reminded of something later, to set a "
        "timer, or to be reminded on a repeating schedule — 'remind me in 20 "
        "minutes to check the printer', 'set a timer for 10 minutes', 'remind "
        "me every weekday at 9 to stand up', 'every 30 minutes', 'on the 1st "
        "of every month'. For a ONE-OFF reminder give EITHER delay_seconds (a "
        "relative time — you compute the seconds) OR at (an absolute ISO 8601 "
        "datetime; you know today's date), never both. For a RECURRING "
        "reminder set `repeat`. By default the first occurrence is derived "
        "from the recurrence (e.g. interval starts now+interval). For an "
        "INTERVAL recurrence you MAY ALSO pass `at` or `delay_seconds` to defer "
        "the FIRST fire — use this for 'every 5 minutes STARTING at 7:50pm' "
        "(set repeat.kind=interval, interval_seconds=300, AND at='...T19:50:00'); "
        "weekly/monthly carry their own time so they ignore at/delay. `message` is "
        "what Jarvis speaks aloud when it fires, phrased as the task itself "
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
                    "9'. Omit if using `delay_seconds` or `repeat`."
                ),
            },
            "repeat": {
                "type": "object",
                "description": (
                    "OMIT for a normal one-shot reminder. Include to make the "
                    "reminder RECUR. The first occurrence is derived from the "
                    "recurrence by default; for kind=interval you may ALSO set "
                    "at/delay_seconds to defer the first fire (then it repeats "
                    "on the interval from there). weekly/monthly ignore at/delay."
                ),
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["interval", "weekly", "monthly"],
                        "description": (
                            "interval = a fixed period (every N minutes or "
                            "hours). weekly = chosen weekdays at a time of day "
                            "— use this for 'every day at 7am' too (all seven "
                            "days). monthly = a day of the month at a time."
                        ),
                    },
                    "interval_seconds": {
                        "type": "integer",
                        "description": (
                            "kind=interval only: whole seconds between fires "
                            "(minimum 60)."
                        ),
                    },
                    "weekdays": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "kind=weekly only: days as integers, 0=Monday … "
                            "6=Sunday. [0,1,2,3,4,5,6] = every day; "
                            "[0,1,2,3,4] = every weekday; [5,6] = weekends."
                        ),
                    },
                    "day": {
                        "type": "integer",
                        "description": (
                            "kind=monthly only: day of the month, 1-31. "
                            "Clamped to the month's last day — so 31 means "
                            "'the last day of every month'."
                        ),
                    },
                    "time": {
                        "type": "string",
                        "description": (
                            "kind=weekly or monthly: time of day, 'HH:MM' "
                            "24-hour (e.g. '07:00', '18:30')."
                        ),
                    },
                },
            },
            "action": {
                "type": "string",
                "enum": ["briefing", "good_night"],
                "description": (
                    "M59 + M63 — OPTIONAL. When set, the reminder TRIGGERS a "
                    "composition tool at fire time instead of just speaking "
                    "the `message` text. "
                    "'briefing' fires the M55 MORNING briefing (weather + "
                    "sports / gaming news + overnight security events + "
                    "today's reminders) — use for 'brief me every weekday at "
                    "7am' / 'every morning at 7 give me the briefing'. "
                    "'good_night' fires the M63 EVENING wrap (security state + "
                    "tomorrow's first event + tomorrow's reminders + "
                    "tomorrow's weather) — use for 'wrap me up every night at "
                    "10pm' / 'every weeknight at 10 give me the good night'. "
                    "`message` becomes a short label shown in list_reminders "
                    "(e.g. 'morning briefing' or 'good night'). Combine with "
                    "`repeat` to schedule a recurring composition — the "
                    "canonical use for both actions."
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

def _resolve_explicit_fire(params: dict) -> "datetime | str | None":
    """Parse an explicit one-off fire time from `delay_seconds` / `at`.

    Shared by the one-shot path and the interval-recurrence deferred-start
    path. Returns:
      - a datetime, if either field is given and valid;
      - a voice-friendly error string, if given but invalid (both supplied,
        unparseable, in the past, or beyond the horizon);
      - None, if NEITHER was supplied — the caller decides what that means
        (an error for a one-off; "derive from the recurrence" for interval).
    """
    delay = params.get("delay_seconds")
    at = (params.get("at") or "").strip()
    if delay is None and not at:
        return None
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
    else:
        parsed = _parse_iso(at)
        if parsed is None:
            return "I couldn't read that time, sir."
        if parsed <= now:
            return (
                f"{_human_dt(parsed)} has already passed, sir — "
                f"shall I set it for another time?"
            )
        fire_at = parsed

    if fire_at > now + timedelta(days=_MAX_HORIZON_DAYS):
        return "That's further out than I can schedule, sir."
    return fire_at


def execute_set_reminder(params: dict) -> str:
    """Schedule a reminder. Never raises — every failure is a readable string."""
    message = (params.get("message") or "").strip()
    if not message:
        return "I need to know what to remind you about, sir."
    if len(message) > _MAX_MESSAGE_LEN:
        message = message[:_MAX_MESSAGE_LEN]

    # M59 + M63 — optional action on fire. Supported: "briefing" (M55
    # morning composition), "good_night" (M63 evening wrap). Unknown values
    # are rejected here (vs. silently stored) so a typo gives the user a
    # clear error instead of a silent never-firing-as-expected.
    action = (params.get("action") or "").strip().lower() or None
    if action and action not in {"briefing", "good_night"}:
        return (f"I don't know the '{action}' action, sir — only "
                f"'briefing' and 'good_night' are supported right now.")

    # M54: a recurring reminder. Normally the first occurrence is derived from
    # the recurrence itself. EXCEPTION (2026-06-02): an `interval` reminder
    # ("every 5 minutes") otherwise starts firing now+interval, with no way to
    # express "every 5 minutes STARTING at 7:50pm" — the gap that broke a real
    # use case. So for interval, honor an explicit first-fire time (at /
    # delay_seconds) when one is given, then re-arm on the interval from there
    # (the re-arm already computes from now() at each fire, so it stays
    # correct). weekly/monthly carry their own day+time, so they ignore
    # at/delay and derive the first slot as before.
    repeat = params.get("repeat")
    if repeat:
        spec = _validate_repeat(repeat)
        if isinstance(spec, str):
            return spec  # validation failed — already a voice-friendly message
        fire_at = None
        if spec["kind"] == "interval":
            explicit = _resolve_explicit_fire(params)
            if isinstance(explicit, str):
                return explicit  # invalid at/delay — voice-friendly error
            fire_at = explicit  # datetime (deferred start) or None (start now+interval)
        if fire_at is None:
            fire_at = _next_occurrence(spec, datetime.now())
        if fire_at is None:
            return "I couldn't work out when that should recur, sir."
        rec = add(message, fire_at, repeat=spec, action=action)
        kind_label = _action_label(action, capitalised=False)
        return (
            f"Recurring {kind_label} set — \"{message}\", {_describe_repeat(spec)}. "
            f"First one at {_human_dt(fire_at)}. id: {rec['id']}."
        )

    fire_at = _resolve_explicit_fire(params)
    if isinstance(fire_at, str):
        return fire_at  # invalid at/delay — voice-friendly error
    if fire_at is None:
        return "I need a time for the reminder, sir."

    rec = add(message, fire_at, action=action)
    kind_label = _action_label(action, capitalised=True)
    return (
        f"{kind_label} set — \"{message}\" — will fire at "
        f"{_human_dt(fire_at)}. id: {rec['id']}."
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
        msg, rid = r.get("message", "?"), r.get("id", "?")
        repeat = r.get("repeat")
        # M59 + M63: an action-tagged reminder gets a "(briefing)" or
        # "(good night)" tag so the user hears that this one COMPOSES the
        # named composition at fire time, not just speaks its message.
        action_tag = _action_listing_tag(r.get("action"))
        if repeat:
            # M54: a recurring reminder — lead with the schedule, then the
            # next occurrence, so "what do I have" reads naturally.
            lines.append(
                f"- \"{msg}\"{action_tag} — {_describe_repeat(repeat)} "
                f"(next: {when}, id: {rid})"
            )
        else:
            lines.append(f"- \"{msg}\"{action_tag} at {when} (id: {rid})")
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
    # Report what's left, so Claude states "what remains" from fact rather
    # than inferring it from a stale earlier list in the conversation (which
    # gave a wrong "the trash reminder remains" when that one had fired).
    remaining = list_pending()
    if remaining:
        names = "; ".join(f"\"{x.get('message', '?')}\"" for x in remaining)
        tail = f" Still pending: {names}."
    else:
        tail = " Nothing else is pending."
    return (
        f"Cancelled, sir — \"{result.get('message', '?')}\" will no longer "
        f"fire.{tail}"
    )


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


def _greeting_for(hour: int) -> str:
    """Time-of-day greeting for a scheduled briefing fire — morning before
    noon, afternoon up to 17:00, evening after. A briefing scheduled for 6 pm
    shouldn't be greeted with 'Good morning, sir.'"""
    if hour < 12:
        return "Good morning, sir."
    if hour < 17:
        return "Good afternoon, sir."
    return "Good evening, sir."


def _compose_briefing() -> str:
    from src.briefing import execute_briefing_tool  # noqa: PLC0415 — lazy
    return execute_briefing_tool({})


def _compose_good_night() -> str:
    from src.good_night import execute_good_night_tool  # noqa: PLC0415 — lazy
    return execute_good_night_tool({})


# Dispatch table for composition actions (M59 briefing, M63 good-night). Each
# entry maps `action` → (composer, spoken-noun, listing-tag, "failed" phrase).
# Adding a new composition (e.g. a "weekly review") is a one-line entry here
# plus an entry in the set in `execute_set_reminder`. The structure scales.
_COMPOSITION_ACTIONS: dict[str, tuple[Callable[[], str], str, str, str]] = {
    "briefing":   (_compose_briefing,   "scheduled briefing", " (briefing)",
                   "Sir, the scheduled briefing failed to compile."),
    "good_night": (_compose_good_night, "evening wrap",       " (good night)",
                   "Sir, the evening wrap failed to compile."),
}


def _action_label(action: str | None, *, capitalised: bool) -> str:
    """Spoken noun for the set-reminder confirmation. Plain 'reminder' for
    a no-action reminder; action-specific noun otherwise."""
    if action in _COMPOSITION_ACTIONS:
        noun = _COMPOSITION_ACTIONS[action][1]
        return noun.capitalize() if capitalised else noun
    return "Reminder" if capitalised else "reminder"


def _action_listing_tag(action: str | None) -> str:
    """Tag suffix for list_reminders ('(briefing)' / '(good night)' / '')."""
    if action in _COMPOSITION_ACTIONS:
        return _COMPOSITION_ACTIONS[action][2]
    return ""


def _push(notify: "Callable[[str], None] | None", text: str) -> None:
    """Deliver a fired reminder to the optional remote sink (Discord push, so
    reminders reach the user when away from the PC). Defensive: a push failure
    must never break the scheduler or the spoken path — the local announce has
    already happened by the time we get here."""
    if notify is None:
        return
    try:
        notify(text)
    except Exception as exc:  # noqa: BLE001 — a push hiccup can't kill a fire
        print(f"[reminders] notify push failed: {exc}", file=sys.stderr)


def _fire_composition(rec: dict, announce: Callable[[str], None],
                      now: datetime,
                      notify: "Callable[[str], None] | None" = None) -> None:
    """M59 + M63 — a reminder with an action in _COMPOSITION_ACTIONS fires
    that composition tool instead of speaking its static message. Runs on a
    throwaway thread because composition fetches weather/news/etc. and can
    take 5–10 s — we don't want to block the scheduler poll loop. The
    `_fire_text` path stays synchronous (cheap; just a string).

    `notify` (optional): the same composed text is also pushed to the remote
    sink (Discord) so a scheduled briefing / good-night reaches the user when
    they're away from the PC."""
    rid = rec.get("id", "?")
    action = rec.get("action") or ""
    entry = _COMPOSITION_ACTIONS.get(action)
    if entry is None:
        # Shouldn't happen — caller checks the dispatch dict — but defensive:
        # an unknown action degrades to the plain-text fire path so the user
        # at least hears the reminder's message.
        text = _fire_text(rec, now)
        announce(text)
        _push(notify, text)
        return
    composer, noun, _, failed_phrase = entry

    def _worker() -> None:
        try:
            text = composer()
        except Exception as exc:  # noqa: BLE001 — never propagate to the thread top
            print(f"[reminders] {action} composition failed: {exc}",
                  file=sys.stderr)
            announce(failed_phrase)
            _push(notify, failed_phrase)
            return
        greeting = _greeting_for(now.hour)
        full = f"{greeting} Your {noun}:\n\n{text}"
        announce(full)
        _push(notify, full)

    threading.Thread(
        target=_worker, name=f"{action}-fire-{rid}", daemon=True,
    ).start()


def _fire_one(rec: dict, now: datetime, announce: Callable[[str], None],
              notify: "Callable[[str], None] | None" = None) -> None:
    """Deliver one due reminder to all sinks: the local `announce` (speak +
    console) and the optional remote `notify` (Discord). Composition reminders
    (briefing / good_night) go through `_fire_composition` (async worker);
    plain reminders speak `_fire_text` synchronously. Extracted from
    run_scheduler so the fan-out is unit-testable without the poll loop."""
    if rec.get("action") in _COMPOSITION_ACTIONS:
        _fire_composition(rec, announce, now, notify)
    else:
        text = _fire_text(rec, now)
        announce(text)
        _push(notify, text)


def run_scheduler(
    announce: Callable[[str], None],
    stop_event: threading.Event,
    poll_sec: float = 10.0,
    notify: "Callable[[str], None] | None" = None,
) -> None:
    """Daemon loop: every ~poll_sec, fire any due reminders via `announce`
    (main.py's WASAPI-safe Announcer path). Polls immediately on start, so
    reminders that came due while Jarvis was off fire right after launch.
    Wrapped so a single bad poll can never kill the thread — a dead scheduler
    would silently drop every future reminder.

    M59 + M63: a reminder with an action in `_COMPOSITION_ACTIONS`
    (briefing / good_night) is dispatched via `_fire_composition` on its
    own worker thread, since composition takes 5–10 s; the scheduler poll
    stays snappy. Plain reminders ride the synchronous _fire_text path
    unchanged.

    `notify` (optional, 2026-06-02): a second, remote sink (Discord push). When
    given, every fired reminder's text is also pushed there so reminders reach
    the user when away from the PC. Independent of `announce` and fail-soft —
    a push failure never affects the spoken path or the loop."""
    print("[reminders] scheduler thread started", file=sys.stderr)
    while not stop_event.is_set():
        try:
            now = datetime.now()
            for rec in pop_due(now):
                rid = rec.get("id")
                action = rec.get("action")
                print(f"[reminders] firing {rid}: "
                      f"{rec.get('message')} (action={action or '-'})",
                      file=sys.stderr)
                _fire_one(rec, now, announce, notify)
        except Exception as exc:  # noqa: BLE001 — keep the thread alive
            print(f"[reminders] scheduler poll failed: {exc}", file=sys.stderr)
        stop_event.wait(poll_sec)
    print("[reminders] scheduler thread exited", file=sys.stderr)
