"""Outlook calendar awareness (M62.1).

Read-only calendar access via Outlook.com's published-iCal feature.

- **OUTLOOK_ICAL_URL** — Outlook.com's "Publish calendar" feature gives a
  secret `.ics` URL; Jarvis fetches it with httpx and expands recurring
  events with `recurring-ical-events`. No OAuth, no Azure, no MSAL.
  Read-only by construction.

The tool surface (`get_calendar_events`) and the briefing's
`_calendar_section()` are backend-agnostic — they call the dispatcher and
get back `CalendarEvent` instances.

> History: M62 originally shipped a second backend — Microsoft Graph via MSAL
> device-code OAuth (`OUTLOOK_CLIENT_ID`). It was never reachable on this
> machine (the user's Entra tenant is suspended, blocking the required Azure
> app registration), so M62.1 pivoted to the iCal feed. The Graph backend was
> removed in the 2026-05-29 QoL pass (dead, unreachable code + an `msal`
> dependency for a path that couldn't run); git history preserves it if the
> tenant is ever unblocked.

Defensive contract — every tool / briefing entry point returns a readable
string and never raises. An HTTP 5xx, an invalid URL, a network blip — all
become voice-friendly messages.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv

# Transient-TLS retry policy (M94). Derived from measurement, not taste — see
# the note in _fetch_events_ical. 8s is generous for a handshake that normally
# completes in well under a second; the exponential gap is what makes each
# retry sample a genuinely later moment instead of the same dead network.
_FETCH_TIMEOUT_SEC = 8.0
_FETCH_BACKOFF_SEC = 1.0


# --- Configuration ---------------------------------------------------------

# Read at IMPORT time and re-exported (briefing.py / good_night.py import the
# VALUE directly), so .env MUST already be loaded when this module is first
# imported. config.py is the canonical dotenv owner, but this module can be
# imported before config in a standalone script or a reordered import — so
# call load_dotenv() here too. It's idempotent and cheap, and it makes this
# module correct in isolation rather than dependent on import ordering.
load_dotenv()

# M62.1 — published iCal URL (Outlook.com → Settings → Calendar → Shared
# calendars → Publish a calendar → "Can view all details"). Bearer
# credential — treat like a webhook URL (gitignored .env).
ICAL_URL = os.getenv("OUTLOOK_ICAL_URL", "").strip()


@dataclass(frozen=True)
class CalendarEvent:
    """Normalised calendar event — just the fields we need for voice +
    briefing, with times pre-converted to local."""
    subject: str
    start_local: datetime
    end_local: datetime
    location: str
    is_all_day: bool


def _fetch_events(start_utc: datetime, end_utc: datetime) -> tuple[list[CalendarEvent] | None, str]:
    """Top-level dispatcher. Fetches from the published iCal feed (M62.1).
    Returns (events, status): on success (list, ""); on any failure
    (None, voice-friendly error); unconfigured returns a clear setup message."""
    if not ICAL_URL:
        return (None, "Outlook calendar isn't configured, sir — set "
                "OUTLOOK_ICAL_URL in .env (Outlook.com → Publish a calendar). "
                "See .env.example for setup instructions.")
    return _fetch_events_ical(start_utc, end_utc)


def _fetch_events_ical(start_utc: datetime, end_utc: datetime) -> tuple[list[CalendarEvent] | None, str]:
    """The M62.1 backend: fetch Outlook.com's published .ics URL, parse with
    icalendar, expand recurring events into the requested window with
    recurring-ical-events. Read-only, no auth, no Azure.

    The fetch is uncached on our side; Microsoft typically caches the
    published feed server-side for ~hours, so "today's events" can be a
    few hours stale. Acceptable for v1; the alternative (polling more
    aggressively) would burn bandwidth for no real benefit since the
    user's calendar doesn't change minute-to-minute."""
    # 2026-07-02 QA added one retry at a flat 0.5s. M93's cross-session review
    # showed that was not enough: this remained the single most frequent fault
    # in the whole log — 47 occurrences across 16 separate sessions in 30 days,
    # and 34 of them exhausted the retry rather than recovering.
    #
    # WHY THE OLD POLICY COULDN'T WORK. The failure is a TLS handshake timeout,
    # so attempt 1 burns the FULL 15s timeout before the backoff even begins.
    # Sleeping a further 0.5s puts attempt 2 about 15.5s in — still inside the
    # same blip. Measured shape of the real outages: 44 distinct bursts over 35
    # days, half of them a single event, the largest 10 events across 1.6 min.
    #
    # WHY THIS IS STRICTLY BETTER, NOT JUST MORE PATIENT. A handshake that has
    # not completed in 8s is not going to (a healthy one is well under 1s), so
    # the per-attempt timeout drops 15s -> 8s and buys a third attempt with an
    # exponential gap. Worst case actually FALLS:
    #     before   15 + 0.5 + 15          = 30.5s over 2 attempts
    #     after     8 + 1 + 8 + 2 + 8      = 27.0s over 3 attempts
    # More chances to catch a good moment, less time spent failing — which
    # matters because this path also serves the on-demand "what's on my
    # calendar?" voice turn, where a user is waiting in silence.
    resp = None
    for attempt in (1, 2, 3):
        try:
            resp = httpx.get(ICAL_URL, follow_redirects=True,
                             timeout=_FETCH_TIMEOUT_SEC)
            break
        except httpx.HTTPError as exc:
            wait = _FETCH_BACKOFF_SEC * (2 ** (attempt - 1))
            print(f"[outlook] ical fetch failed: "
                  f"{type(exc).__name__}: {exc}"
                  + (f" — retrying in {wait}s (attempt {attempt + 1}/3)"
                     if attempt < 3 else " — gave up after 3 attempts"),
                  file=sys.stderr)
            if attempt == 3:
                return (None,
                        "I couldn't reach the Outlook iCal feed just now, sir.")
            time.sleep(wait)
    if resp.status_code != 200:
        print(f"[outlook] ical HTTP {resp.status_code}: {resp.text[:200]}",
              file=sys.stderr)
        return (None, f"Outlook iCal returned HTTP {resp.status_code}, sir — "
                f"check the URL in OUTLOOK_ICAL_URL is still valid.")

    # Lazy imports — keeps the module loadable even before
    # recurring-ical-events is installed (e.g. on a fresh checkout where
    # `pip install -r requirements.txt` hasn't been run yet).
    try:
        import icalendar  # noqa: PLC0415 — lazy
        import recurring_ical_events  # noqa: PLC0415 — lazy
    except ImportError as exc:
        print(f"[outlook] ical libs missing: {exc}", file=sys.stderr)
        return (None, "I'm missing the iCal parsing library, sir — "
                "run `pip install -r requirements.txt`.")
    try:
        cal = icalendar.Calendar.from_ical(resp.content)
    except Exception as exc:  # noqa: BLE001 — many parser failure modes
        print(f"[outlook] ical parse failed: {exc}", file=sys.stderr)
        return (None, "I couldn't parse the Outlook iCal feed, sir — "
                "the URL might be pointing at something else.")
    try:
        raw_events = recurring_ical_events.of(cal).between(start_utc, end_utc)
    except Exception as exc:  # noqa: BLE001 — RRULE expansion can throw
        print(f"[outlook] ical recurrence expansion failed: {exc}",
              file=sys.stderr)
        return (None, "I couldn't expand recurring events from the feed, sir.")

    events: list[CalendarEvent] = []
    for ev in raw_events:
        normalised = _normalise_ical_event(ev)
        if normalised is not None:
            events.append(normalised)
    events.sort(key=lambda e: e.start_local)
    return (events, "")


def _normalise_ical_event(vevent) -> CalendarEvent | None:
    """Convert an icalendar VEVENT to our CalendarEvent shape."""
    try:
        from datetime import date  # noqa: PLC0415 — narrow scope
        subject = str(vevent.get("SUMMARY", "")).strip() or "(no subject)"
        location = str(vevent.get("LOCATION", "")).strip()
        dtstart_prop = vevent.get("DTSTART")
        dtend_prop = vevent.get("DTEND")
        if dtstart_prop is None:
            return None
        start = dtstart_prop.dt
        # An all-day event's DTSTART is a `date` (no time). datetime is a
        # subclass of date, so order matters: check datetime first.
        is_all_day = isinstance(start, date) and not isinstance(start, datetime)
        if is_all_day:
            # Promote DATE → DATETIME at midnight, LOCAL wall clock (an all-day
            # event is "all of that calendar day in the viewer's zone"). Leave
            # these NAIVE: the final .astimezone() below reads a naive datetime
            # as system-local, giving the intended local midnight. Stamping them
            # UTC here (as the timed branch does) would shift the calendar day
            # BACKWARD in any behind-UTC zone — a real bug masked only because
            # every consumer special-cases is_all_day.
            start = datetime.combine(start, datetime.min.time())
            if dtend_prop is not None:
                end_raw = dtend_prop.dt
                if isinstance(end_raw, date) and not isinstance(end_raw, datetime):
                    end = datetime.combine(end_raw, datetime.min.time())
                else:
                    end = end_raw
            else:
                end = start + timedelta(days=1)
        else:
            end = dtend_prop.dt if dtend_prop is not None else start + timedelta(hours=1)
            # Ensure tz-aware. icalendar returns aware datetimes when the VEVENT
            # has TZID (Microsoft always sets it); be defensive and treat a naive
            # TIMED datetime as UTC (the server's canonical zone). All-day events
            # stay naive-local (handled above) and must NOT reach this.
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
        return CalendarEvent(
            subject=subject,
            start_local=start.astimezone(),
            end_local=end.astimezone(),
            location=location,
            is_all_day=is_all_day,
        )
    except Exception as exc:  # noqa: BLE001 — defensive per-event
        print(f"[outlook] ical event normalise failed: {exc}", file=sys.stderr)
        return None


# --- Public helpers (used by briefing + tool) -----------------------------

def today_events() -> tuple[list[CalendarEvent] | None, str]:
    """Events from local-midnight today through local-midnight tomorrow."""
    now_local = datetime.now().astimezone()
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)
    return _fetch_events(start_local.astimezone(timezone.utc),
                         end_local.astimezone(timezone.utc))


def fetch_events_in_window(
    start_local: datetime, end_local: datetime,
) -> tuple[list[CalendarEvent] | None, str]:
    """Backend-agnostic event fetch in a LOCAL-time window. Wraps the
    internal dispatcher (which speaks UTC) for callers — like the M62.2
    proactive monitor — that think in wall-clock time. Returns the same
    `(events|None, err_msg)` contract as the rest of this module."""
    return _fetch_events(
        start_local.astimezone(timezone.utc),
        end_local.astimezone(timezone.utc),
    )


# --- Anthropic tool -------------------------------------------------------

GET_CALENDAR_TOOL = {
    "name": "get_calendar_events",
    "description": (
        "Read events from the user's Outlook calendar (personal Microsoft "
        "account, read-only). Use this for 'what's on my calendar', 'do I "
        "have anything later', 'what's my next meeting', 'am I free at 3pm', "
        "'what's tomorrow looking like'. Returns a list of events with "
        "time + subject (+ location when present). Read it back "
        "conversationally — don't enumerate the JSON; mention the times "
        "and subjects naturally. If the calendar isn't configured or "
        "authorisation has expired, the tool returns a clear setup "
        "instruction — relay that plainly and don't fabricate events. The "
        "user cannot create or modify events through Jarvis (read-only); "
        "if they ask to schedule something, say so."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "timeframe": {
                "type": "string",
                "enum": ["today", "tomorrow", "this_week", "next_24h"],
                "description": (
                    "Which window to fetch. "
                    "'today' = local midnight to midnight (the default); "
                    "'tomorrow' = the next calendar day; "
                    "'this_week' = today through the end of the upcoming "
                    "Sunday; "
                    "'next_24h' = the rolling next 24 hours from now "
                    "(use this for 'what's coming up' / 'anything later')."
                ),
            },
        },
    },
}


def _window_for(timeframe: str) -> tuple[datetime, datetime, str] | None:
    """Resolve a timeframe enum to (start_local, end_local, label).
    None if the timeframe is unknown."""
    now_local = datetime.now().astimezone()
    today_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    if timeframe == "today":
        return (today_start, today_start + timedelta(days=1), "today")
    if timeframe == "tomorrow":
        start = today_start + timedelta(days=1)
        return (start, start + timedelta(days=1), "tomorrow")
    if timeframe == "this_week":
        # today through the end of this Sunday (treating Monday as the start
        # of the week for compatibility with Python's weekday() = 0..6,
        # Mon..Sun). On Sunday, "this week" is just today.
        weekday = now_local.weekday()        # Mon=0 … Sun=6
        days_until_sun_end = 6 - weekday     # 0 if Sunday
        end = today_start + timedelta(days=days_until_sun_end + 1)
        return (today_start, end, "this week")
    if timeframe == "next_24h":
        return (now_local, now_local + timedelta(hours=24),
                "in the next 24 hours")
    return None


def execute_calendar_tool(params: dict) -> str:
    """Tool executor. Never raises — every failure returns a voice-friendly
    string. Time format: HH:MM 24-hour local; Claude voices these as the
    user prefers."""
    raw = (params.get("timeframe") or "today").strip().lower()
    win = _window_for(raw)
    if win is None:
        return (f"I don't recognise the timeframe '{raw}', sir — try "
                f"today, tomorrow, this_week, or next_24h.")
    start_local, end_local, label = win
    events, err = _fetch_events(
        start_local.astimezone(timezone.utc),
        end_local.astimezone(timezone.utc),
    )
    if err:
        return err
    return _format_events(events or [], label)


def _format_events(events: list[CalendarEvent], label: str) -> str:
    """Render a CalendarEvent list as a structured-but-spoken-friendly text
    block. Claude voices this conversationally; the format is just for the
    tool result."""
    if not events:
        return f"Nothing on your calendar {label}, sir."
    lines = [f"{len(events)} event(s) {label}:"]
    for ev in events:
        when = _format_event_time(ev)
        line = f"- {when}  {ev.subject}"
        if ev.location:
            line += f"  ({ev.location})"
        lines.append(line)
    return "\n".join(lines)


def _format_event_time(ev: CalendarEvent) -> str:
    if ev.is_all_day:
        return "all day"
    # Same-day window → just the times. Multi-day → include the date.
    same_day = ev.start_local.date() == ev.end_local.date()
    if same_day:
        return f"{ev.start_local.strftime('%H:%M')}–{ev.end_local.strftime('%H:%M')}"
    return (f"{ev.start_local.strftime('%a %H:%M')}–"
            f"{ev.end_local.strftime('%a %H:%M')}")
