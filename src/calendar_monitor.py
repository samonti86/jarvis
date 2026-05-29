"""Pre-event proactive calendar reminders (M62.2) — the calendar arc's
proactive layer.

M62 + M62.1 shipped the read layer: a tool + briefing section that surface
the user's Outlook calendar on demand. M62.2 is the cash-in — Jarvis quietly
watches that same feed and SPEAKS UP on his own a fixed lead time (default
15 min) before each event:

    "Sir — your 2 pm standup in 15 minutes."

The reactive→proactive jump, the same one M56 made for the homelab. Almost
entirely a recombination:

  - thread model       = homelab_monitor — a defensive daemon poll loop, a
                         stop Event, every poll wrapped so a single bad fetch
                         can never kill the thread.
  - fetch              = outlook_calendar.fetch_events_in_window — the
                         published-iCal dispatcher, called with a tight
                         `[now, now+lead+5m]` window so we don't pull a full
                         day every minute.
  - voice out          = main.py's _announce (the WASAPI-safe Announcer
                         path), tagged 📅 so reminders read distinctly from
                         🚨 security / 🖥 homelab / 🔔 acoustic / ⏰ M53.
  - phone push         = notifications.send_discord_message (same channel
                         M38/M56 use).

The interesting parts — dedupe and decision
-------------------------------------------
A calendar reminder is a one-shot edge — once we've announced an event we
must NEVER announce it again. Three layers handle that:

  1. In-memory dedupe set keyed on `f"{subject}|{start_local.isoformat()}"`.
     If the user reschedules the 2 pm to 3 pm, the start changes ⇒ a new ID
     ⇒ a fresh announce. Correct: a moved meeting is news.

  2. Persistence to disk (`%LOCALAPPDATA%\\Jarvis\\calendar_announced.json`).
     Without this a Jarvis restart between the 15-min announce and the
     meeting itself would re-announce. Pruned to entries < 24 h on every
     load/save — an old event ID can't recur (start_local is part of the
     ID), so retaining longer just bloats the file.

  3. A pure decision function `_should_announce(ev, now, lead_min,
     announced)`. The hard logic isolated from all I/O so it's trivially
     testable (see scripts/calendar_monitor_test.py). Feed it synthetic
     events; assert the answer.

Configuration / defaults
------------------------
Unlike M56 (homelab) and M58 (acoustic), this is DEFAULT ON when calendar is
configured — because the user already opted in to calendar awareness by
setting OUTLOOK_ICAL_URL, and the proactive layer is the *whole point* of
having it. Kill switch: `JARVIS_CALENDAR_REMINDERS=0`.

Tunables (all env-knobs, read once at import):
  - JARVIS_CALENDAR_LEAD_MIN       — minutes before start to announce (15)
  - JARVIS_CALENDAR_POLL_SECONDS   — seconds between polls (60, min 30)

Defensive contract — same as every monitor module: nothing here raises into
the poll thread or the listen loop. A bad fetch, a corrupt dedupe file, a
malformed event — all become log lines and a fail-soft pass, never an
exception.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

# --- Tunables (module-local, env-driven; same pattern as homelab_monitor) ---

def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, value)


# How far ahead of an event to announce. 15 min is the classic meeting-
# reminder lead; floor 1 so a typo can't disable the announce by setting 0.
_LEAD_MIN = _env_int("JARVIS_CALENDAR_LEAD_MIN", 15, 1)

# Poll cadence. 60 s is plenty — a calendar publishes server-side cache (per
# M62.1) lags by hours anyway, and `now - event.start` doesn't change minute
# to minute. Floor 30 s so a typo can't busy-loop the fetch.
_POLL_SECONDS = _env_int("JARVIS_CALENDAR_POLL_SECONDS", 60, 30)

# How long the on-disk dedupe set keeps event IDs. 24 h is plenty: an event
# ID encodes start_local, so once an event has started its ID can never
# recur. Retaining longer just bloats the file.
_DEDUPE_RETAIN_HOURS = 24


# --- Configuration discovery ------------------------------------------------

def _calendar_configured() -> bool:
    """True if the calendar is configured (published iCal URL). Reads the env
    live so the monitor's activate() reflects the current .env without
    re-importing outlook_calendar."""
    return bool(os.getenv("OUTLOOK_ICAL_URL", "").strip())


def _kill_switch_set() -> bool:
    """JARVIS_CALENDAR_REMINDERS=0/false/off/no disables the monitor even
    when calendar is configured. Default behaviour: enabled when calendar is
    configured (the user already opted in by setting up calendar)."""
    raw = os.getenv("JARVIS_CALENDAR_REMINDERS", "").strip().lower()
    return raw in ("0", "false", "no", "off")


# --- Pure decision logic (testable in isolation) ---------------------------

def _event_id(ev) -> str:
    """Stable dedupe key. `(subject, start_local)` is unique enough for
    personal calendars and naturally handles reschedules — a moved meeting
    gets a new ID and (correctly) a fresh announce."""
    return f"{ev.subject}|{ev.start_local.isoformat()}"


def _should_announce(
    ev,                        # CalendarEvent (duck-typed for tests)
    now: datetime,
    lead_min: int,
    announced: Iterable[str],
) -> tuple[bool, int]:
    """Pure: given an event, the current local time, the lead window, and
    the set of already-announced IDs, return (should_fire, mins_to_start).

    The whole interesting core of M62.2 — deliberately isolated from fetch /
    announce / disk I/O so a regression in the firing rule is caught
    structurally by the test harness, not by waiting for a real meeting.

    Rules (in order):
      1. all-day → never fire (per the design Q&A; the morning briefing
         already surfaces all-day items, and "in N minutes" is meaningless
         for them).
      2. already started (secs ≤ 0) → never fire (don't announce a meeting
         we've missed — that's just noise).
      3. outside the lead window (mins > lead_min) → not yet (we'll
         re-evaluate on the next poll).
      4. already announced → suppressed.
      5. else → fire.

    `mins_to_start` is rounded to the nearest minute and is the value spoken
    in the announce text; when fire is False it's informational only.
    """
    if ev.is_all_day:
        return (False, 0)
    secs_to_start = (ev.start_local - now).total_seconds()
    if secs_to_start <= 0:
        return (False, 0)
    mins_to_start = round(secs_to_start / 60)
    if mins_to_start > lead_min:
        return (False, mins_to_start)
    if _event_id(ev) in set(announced):
        return (False, mins_to_start)
    return (True, mins_to_start)


# --- Dedupe store (testable in isolation) ----------------------------------

def _parse_iso_safe(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return datetime.min


class DedupeStore:
    """In-memory dedupe with a JSON-on-disk backing.

    Surface:
      - has(event_id)            — has this ID been announced?
      - mark(event_id)           — record an announce (in-memory only).
      - save()                   — flush to disk atomically; prunes
                                   entries older than `retain_hours` first.
      - __len__()                — used by the self-status getter.

    Thread-safe (a single `threading.Lock` guards every mutation). All
    on-disk failures are logged + swallowed: the worst case is one
    double-announce after a crash, which is recoverable; a crash on save
    is NOT recoverable, so save NEVER raises.
    """

    def __init__(self, path: Path, retain_hours: int = _DEDUPE_RETAIN_HOURS) -> None:
        self._path = path
        self._retain_hours = retain_hours
        self._lock = threading.Lock()
        self._announced = self._read()

    def _read(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — many parse failure modes
            print(f"[calendar] dedupe read failed: {exc}", file=sys.stderr)
            return {}
        if not isinstance(raw, dict):
            return {}
        cutoff = datetime.now() - timedelta(hours=self._retain_hours)
        out: dict[str, str] = {}
        for k, v in raw.items():
            if not (isinstance(k, str) and isinstance(v, str)):
                continue
            when = _parse_iso_safe(v)
            if when >= cutoff:
                out[k] = v
        return out

    def has(self, event_id: str) -> bool:
        with self._lock:
            return event_id in self._announced

    def mark(self, event_id: str) -> None:
        with self._lock:
            self._announced[event_id] = datetime.now().isoformat()

    def snapshot(self) -> set[str]:
        """Return a copy of the announced-IDs set — for passing to
        `_should_announce` without holding the lock during decision."""
        with self._lock:
            return set(self._announced.keys())

    def save(self) -> None:
        cutoff = datetime.now() - timedelta(hours=self._retain_hours)
        with self._lock:
            pruned = {
                k: v for k, v in self._announced.items()
                if _parse_iso_safe(v) >= cutoff
            }
            self._announced = pruned
            payload = json.dumps(pruned)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as exc:
            print(f"[calendar] dedupe save failed: {exc}", file=sys.stderr)

    def __len__(self) -> int:
        with self._lock:
            return len(self._announced)


# --- The monitor -----------------------------------------------------------

def _store_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    return base / "calendar_announced.json"


class CalendarMonitor:
    """Background calendar watcher. Fires a spoken + Discord announce when
    an event's start is within `_LEAD_MIN` minutes. Dedupes across polls
    (in-memory) and across restarts (`calendar_announced.json`).

    Thread-safe surface: activate / deactivate / shutdown / is_active /
    dedupe_size may be called from any thread.

    Constructed unconditionally (so `main.py`'s status registration always
    has a target); activate() is the no-op when calendar isn't configured
    or the env kill switch is set — same shape as M56's homelab monitor.
    """

    def __init__(
        self,
        announce: Callable[[str], None],
        *,
        discord_webhook_url: str = "",
    ) -> None:
        self._announce = announce
        self._discord = (discord_webhook_url or "").strip()
        self._store = DedupeStore(_store_path())

        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def activate(self) -> None:
        """Start the poll loop. Idempotent. No-op (with a log line) when
        calendar isn't configured or the env kill switch is set — same
        graceful-degradation shape as M56's activate()."""
        if self._active.is_set():
            return
        if _kill_switch_set():
            print("[calendar] JARVIS_CALENDAR_REMINDERS disabled — not starting",
                  file=sys.stderr)
            return
        if not _calendar_configured():
            print("[calendar] not configured (set OUTLOOK_ICAL_URL) — "
                  "not starting", file=sys.stderr)
            return
        self._active.set()
        self._stop.clear()
        print(f"[calendar] monitor activated — poll {_POLL_SECONDS}s, "
              f"lead {_LEAD_MIN}min, dedupe-size {len(self._store)}",
              file=sys.stderr)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._watch_loop, name="CalendarMonitor", daemon=True,
            )
            self._thread.start()

    def deactivate(self) -> None:
        """Stop the poll loop. Idempotent."""
        if not self._active.is_set():
            return
        self._active.clear()
        self._stop.set()
        print("[calendar] monitor deactivated", file=sys.stderr)

    def shutdown(self) -> None:
        """App-quit wind-down. Daemon thread dies with the process anyway;
        this lets it exit its poll-wait cleanly without log noise."""
        self._active.clear()
        self._stop.set()

    def is_active(self) -> bool:
        return self._active.is_set()

    def dedupe_size(self) -> int:
        return len(self._store)

    # -- poll loop ---------------------------------------------------------

    def _watch_loop(self) -> None:
        """Daemon: every _POLL_SECONDS run one fetch and fire any announces
        the decision function flags. The whole body is wrapped — a dead
        scheduler would silently drop every future reminder, so a bad poll
        must never kill the thread (the M56 lesson, copied directly)."""
        print("[calendar] monitor loop started", file=sys.stderr)
        while not self._stop.is_set() and self._active.is_set():
            try:
                self._poll_once()
            except Exception as exc:  # noqa: BLE001 — keep the thread alive
                print(f"[calendar] poll failed: {exc}", file=sys.stderr)
            if self._stop.wait(_POLL_SECONDS):
                break
        print("[calendar] monitor loop exited", file=sys.stderr)

    def _poll_once(self) -> None:
        # Lazy import — keeps src/calendar_monitor.py loadable on a fresh
        # checkout before requirements.txt is installed (recurring-ical-events
        # might be missing; the fetch will report cleanly).
        from src.outlook_calendar import fetch_events_in_window  # noqa: PLC0415

        now_local = datetime.now().astimezone()
        # Tight fetch window: anything starting more than (lead + 5 min) out
        # can't possibly be due in this poll. Saves bandwidth and keeps the
        # parse cheap.
        window_end = now_local + timedelta(minutes=_LEAD_MIN + 5)
        events, err = fetch_events_in_window(now_local, window_end)
        if err or events is None:
            # The dispatcher already logs the underlying cause. Don't
            # re-announce errors as events — they aren't events.
            if err:
                print(f"[calendar] fetch err (will retry next poll): "
                      f"{err[:200]}", file=sys.stderr)
            return

        announced = self._store.snapshot()
        fired_any = False
        for ev in events:
            should_fire, mins = _should_announce(ev, now_local, _LEAD_MIN, announced)
            if not should_fire:
                continue
            # mark BEFORE emit — if the announce path raises (it shouldn't,
            # but) we still don't repeat-fire on the next poll. The save at
            # the bottom flushes the new ID to disk.
            event_id = _event_id(ev)
            self._store.mark(event_id)
            announced.add(event_id)  # affects the rest of this same poll
            self._emit(ev, mins)
            fired_any = True

        if fired_any:
            self._store.save()

    def _emit(self, ev, mins: int) -> None:
        """One announce: spoken via the WASAPI-safe Announcer path + a
        Discord push (on a throwaway thread so a slow POST never delays
        the next poll). Same shape as M56's _emit."""
        unit = "minute" if mins == 1 else "minutes"
        text = f"Sir — your {ev.subject} in {mins} {unit}."
        print(f"[calendar] ANNOUNCE: {text}", file=sys.stderr)
        self._safe_announce(text)
        if self._discord:
            push = f"📅 {text}"
            if ev.location:
                push += f" ({ev.location})"
            threading.Thread(
                target=self._push_discord, args=(push,),
                name="CalendarNotify", daemon=True,
            ).start()

    def _push_discord(self, text: str) -> None:
        from src.notifications import send_discord_message  # noqa: PLC0415 — lazy
        try:
            send_discord_message(self._discord, text)
        except Exception as exc:  # noqa: BLE001 — never break on a push
            print(f"[calendar] discord push failed: {exc}", file=sys.stderr)

    def _safe_announce(self, text: str) -> None:
        try:
            self._announce(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[calendar] announce failed: {exc}", file=sys.stderr)
