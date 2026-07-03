"""Quiet hours — a Do-Not-Disturb policy for proactive announcements (M79).

By M78 Jarvis has many sources that can speak UNPROMPTED — the security
deterrent, the homelab monitor, calendar pre-event reminders, severe-weather
alerts, acoustic alerts, the reminder scheduler, presence greetings. None of
them respected a "don't wake me" window. This adds one: a single policy applied
at the announce chokepoint (main.py's `_announce`).

THE POLICY (deliberately a SAFE DEFAULT — deferral is opt-IN):
  During the configured quiet window, only ROUTINE status-monitoring announces
  are held back; everything else pierces. A new announce source therefore
  pierces by default and is only silenced once explicitly classified routine —
  so we can never *accidentally* mute something important (a smoke alarm, a
  power-threatening storm, a user-set reminder). The class that defers today is
  the homelab monitor (🖥) — disk-space/flap chatter that can wait for morning.

  Pierce (speak even at 3 AM): security 🚨, reminders ⏰ (user-set, explicit
  intent), severe weather ⛈ (already severity-filtered — safety), acoustic 🔔
  (armed-only — security-relevant), calendar 📅 (the user's own scheduled
  events), presence 🏠 (user-triggered by arriving home).

DEFERRED announces aren't lost — they're recorded and surfaced as a "while you
were away" line in the morning briefing (the same surfacing channel as the
prediction follow-ups), then cleared.

Config: JARVIS_QUIET_HOURS="22:00-08:00" (read at call time, hot-swappable, the
wolfram/tmdb pattern). Unset/blank/malformed ⇒ DND is OFF — every announce
speaks, exactly the pre-M79 behaviour. Overnight windows (start > end) wrap
across midnight.

Defensive contract: every entry point is fail-soft. A bad config, an unreadable
store, a clock oddity — none may break an announce. When in doubt, SPEAK (never
silently swallow an announcement because the DND logic hiccupped).
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from datetime import datetime, time, timedelta
from pathlib import Path

from src.atomic_io import atomic_write_text
from src.memory import format_relative_time


# Routine status-monitoring labels that DEFER during quiet hours. Everything
# else pierces (the safe default). Inverted on purpose: silence is opt-in.
#   🖥 — homelab status monitoring (M79)
#   🧠 — anticipatory-intelligence insights (M83): valuable but rarely so urgent
#        they can't wait for the morning "while you were away" catch-up; truly
#        urgent single-signal events (a severe storm) have their own piercing
#        monitor, so anticipation can defer cleanly.
_DEFERRABLE_LABELS: frozenset[str] = frozenset({"🖥", "🧠"})

# Deferred announces older than this when the briefing reads them are dropped as
# stale (an overnight item is a few hours old by morning; a multi-day-old one is
# noise). Bounds the store too.
_MAX_DEFERRED_AGE_HOURS = 18
_MAX_DEFERRED = 50

_WINDOW_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")


# --- Window ----------------------------------------------------------------

def _parse_window(raw: str) -> tuple[time, time] | None:
    """Parse 'HH:MM-HH:MM' → (start, end) times. None on absent/malformed
    (→ DND off). Fail-soft: a typo disables DND rather than crashing announces."""
    if not raw:
        return None
    m = _WINDOW_RE.match(raw)
    if not m:
        print(f"[quiet] ignoring malformed JARVIS_QUIET_HOURS={raw!r} "
              f"(want HH:MM-HH:MM)", file=sys.stderr)
        return None
    sh, sm, eh, em = (int(g) for g in m.groups())
    if not (0 <= sh < 24 and 0 <= eh < 24 and 0 <= sm < 60 and 0 <= em < 60):
        print(f"[quiet] ignoring out-of-range JARVIS_QUIET_HOURS={raw!r}",
              file=sys.stderr)
        return None
    return time(sh, sm), time(eh, em)


def _window() -> tuple[time, time] | None:
    return _parse_window(os.getenv("JARVIS_QUIET_HOURS", "").strip())


def is_quiet(now: datetime | None = None) -> bool:
    """True if `now` falls inside the configured quiet window. False when DND
    is unconfigured. Handles overnight windows (start > end) that wrap midnight."""
    w = _window()
    if w is None:
        return False
    start, end = w
    if start == end:
        return False  # zero-width window = effectively off
    t = (now or datetime.now()).time()
    if start < end:
        return start <= t < end            # same-day window (e.g. 13:00-14:00)
    return t >= start or t < end           # overnight wrap (e.g. 22:00-08:00)


def decide(label: str, now: datetime | None = None) -> str:
    """'speak' or 'defer' for an announce with the given source `label`. Defers
    ONLY a routine-status label inside the quiet window; everything else speaks
    (the safe default). Never raises — any oddity falls through to 'speak'."""
    try:
        if not is_quiet(now):
            return "speak"
        return "defer" if label in _DEFERRABLE_LABELS else "speak"
    except Exception as exc:  # noqa: BLE001 — never swallow an announce on a bug
        print(f"[quiet] decide failed ({exc}) — speaking", file=sys.stderr)
        return "speak"


# --- Deferred store --------------------------------------------------------
# %LOCALAPPDATA%\Jarvis\deferred_announces.json — a JSON list of
# {ts, text, label}, atomically written (temp + os.replace). Fail-soft on read.

def _store_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base / "deferred_announces.json"


def _load() -> list[dict]:
    path = _store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[quiet] could not read {path.name}: {exc}", file=sys.stderr)
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def _save(items: list[dict]) -> None:
    # Durable + atomic — see src/atomic_io.py.
    atomic_write_text(_store_path(), json.dumps(items, indent=2, ensure_ascii=False))


# 2026-07-02 QA: record_deferred (announce thread) and take_deferred (briefing
# thread) are both load-mutate-save cycles; unserialized, one interleaving
# resurrects already-surfaced items and the opposite one silently drops a fresh
# deferral. atomic_write_text prevents torn FILES, not lost UPDATES.
_STORE_LOCK = threading.Lock()


def record_deferred(text: str, label: str, now: datetime | None = None) -> None:
    """Append a held-back announce to the catch-up store. Never raises — a
    failure here just means the note won't appear in the morning catch-up; it
    must not break the announce path."""
    if not text:
        return
    try:
        with _STORE_LOCK:
            items = _load()
            items.append({
                "ts": (now or datetime.now()).isoformat(timespec="seconds"),
                "text": text,
                "label": label,
            })
            _save(items[-_MAX_DEFERRED:])  # keep the newest, bound the file
    except Exception as exc:  # noqa: BLE001
        print(f"[quiet] record_deferred failed: {exc}", file=sys.stderr)


def take_deferred(now: datetime | None = None) -> list[dict]:
    """Return the non-stale deferred announces and CLEAR the store (stale ones
    are dropped). Called by the briefing to surface 'while you were away'.
    Never raises — returns [] on any failure."""
    try:
        with _STORE_LOCK:
            items = _load()
            if not items:
                return []
            now = now or datetime.now()
            cutoff = now - timedelta(hours=_MAX_DEFERRED_AGE_HOURS)
            fresh: list[dict] = []
            for it in items:
                try:
                    if datetime.fromisoformat(it.get("ts", "")) >= cutoff:
                        fresh.append(it)
                except (ValueError, TypeError):
                    continue  # undated/garbage → drop
            _save([])  # clear all (incl. the stale ones we dropped)
        return fresh
    except Exception as exc:  # noqa: BLE001
        print(f"[quiet] take_deferred failed: {exc}", file=sys.stderr)
        return []


def format_deferred(items: list[dict]) -> str:
    """Render deferred announces as a briefing catch-up section. '' if none."""
    if not items:
        return ""
    lines = ["While you were away, sir — a few quiet-hours notes:"]
    for it in items:
        text = (it.get("text") or "").strip()
        if not text:
            continue
        when = format_relative_time(it.get("ts", "")) if it.get("ts") else ""
        lines.append(f"- {text}" + (f" ({when})" if when else ""))
    return "\n".join(lines) if len(lines) > 1 else ""
