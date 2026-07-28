"""Regression test for M62.1 Outlook calendar awareness (iCal backend).

Exercises everything that doesn't require a live HTTP round-trip: schema
shape, the unconfigured error path, timeframe → window resolution, the
event formatter (empty / single / multi-event, all-day, with-and-without
location), ICS parsing + RRULE expansion via icalendar /
recurring-ical-events, and the dispatcher's configured/unconfigured branches.

What it INTENTIONALLY does NOT test: the live iCal HTTP fetch (would need a
real published URL).

(The Microsoft Graph backend was removed in the 2026-05-29 QoL pass; the
former parse_graph_dt / _normalise / Graph-dispatcher tests went with it.)

    python tests/outlook_calendar_test.py     # exit 0 = pass
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import outlook_calendar as oc  # noqa: E402


_passed = 0
_failed = 0


def check(label: str, condition: bool) -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")


# --- Test 1: schema is well-formed ----------------------------------------
schema = oc.GET_CALENDAR_TOOL
check("GET_CALENDAR_TOOL: name + description present",
      schema.get("name") == "get_calendar_events"
      and "description" in schema)
props = schema["input_schema"]["properties"]
check("schema: `timeframe` enum has the 4 expected values",
      set(props["timeframe"]["enum"]) == {"today", "tomorrow", "this_week", "next_24h"})


# --- Test 2: not configured -> voice-friendly setup msg ------------------
# Force ICAL_URL empty so this is deterministic regardless of the developer's
# real .env (which may have a live OUTLOOK_ICAL_URL). Setting the module attr
# directly is immune to the import-time load_dotenv().
_orig_url = oc.ICAL_URL
oc.ICAL_URL = ""
out = oc.execute_calendar_tool({"timeframe": "today"})
check("not configured -> tool says 'isn't configured' + names OUTLOOK_ICAL_URL",
      "configured" in out.lower() and "OUTLOOK_ICAL_URL" in out)


# --- Test 3: invalid timeframe is rejected with a voice-friendly error ---
oc.ICAL_URL = "https://example.com/calendar.ics"  # bypass not-configured branch
out = oc.execute_calendar_tool({"timeframe": "yesterday"})
check("invalid timeframe -> voice-friendly error mentions valid options",
      "yesterday" in out and "today" in out and "tomorrow" in out)
oc.ICAL_URL = _orig_url  # restore


# --- Test 4: _window_for resolves each enum correctly ---------------------
today = oc._window_for("today")
check("_window_for('today') -> 24-hour midnight-to-midnight window",
      today is not None
      and (today[1] - today[0]) == timedelta(days=1)
      and today[2] == "today")

tomorrow = oc._window_for("tomorrow")
check("_window_for('tomorrow') -> starts 24h after today's start",
      tomorrow is not None
      and tomorrow[0] - today[0] == timedelta(days=1))

next24 = oc._window_for("next_24h")
check("_window_for('next_24h') -> exactly 24h window",
      next24 is not None
      and (next24[1] - next24[0]) == timedelta(hours=24))

week = oc._window_for("this_week")
check("_window_for('this_week') -> window of 1-7 days",
      week is not None
      and timedelta(days=1) <= (week[1] - week[0]) <= timedelta(days=7))

check("_window_for('nonsense') -> None", oc._window_for("nonsense") is None)


# --- Test 5: _format_events handles the various cases --------------------
check("format empty -> 'Nothing on your calendar today, sir.'",
      "Nothing" in oc._format_events([], "today"))

one = oc.CalendarEvent(
    subject="Standup",
    start_local=datetime(2026, 5, 26, 9, 30, tzinfo=timezone.utc).astimezone(),
    end_local=datetime(2026, 5, 26, 10, 0, tzinfo=timezone.utc).astimezone(),
    location="",
    is_all_day=False,
)
out = oc._format_events([one], "today")
check("format one event -> mentions count + subject",
      "1 event" in out and "Standup" in out)

two_with_loc = [
    one,
    oc.CalendarEvent(
        subject="Lunch with Alex",
        start_local=datetime(2026, 5, 26, 18, 0, tzinfo=timezone.utc).astimezone(),
        end_local=datetime(2026, 5, 26, 19, 0, tzinfo=timezone.utc).astimezone(),
        location="Cafe XYZ",
        is_all_day=False,
    ),
]
out = oc._format_events(two_with_loc, "today")
check("format multi-event with location -> includes both + location",
      "Standup" in out and "Lunch with Alex" in out and "Cafe XYZ" in out)

all_day = [oc.CalendarEvent(
    subject="Holiday",
    start_local=datetime(2026, 5, 26, 0, 0, tzinfo=timezone.utc).astimezone(),
    end_local=datetime(2026, 5, 27, 0, 0, tzinfo=timezone.utc).astimezone(),
    location="",
    is_all_day=True,
)]
out = oc._format_events(all_day, "tomorrow")
check("format all-day event -> says 'all day'",
      "all day" in out.lower())


# --- Test 6: _normalise_ical_event on a real icalendar VEVENT ------------
# (build the .ics in memory and let icalendar parse it — exercises the same
# code path the live fetch does, minus the httpx round-trip).
import icalendar  # noqa: E402

ICS_SINGLE = (
    "BEGIN:VCALENDAR\r\n"
    "PRODID:-//Test//EN\r\n"
    "VERSION:2.0\r\n"
    "BEGIN:VEVENT\r\n"
    "UID:test-single@example.com\r\n"
    "DTSTART:20260601T140000Z\r\n"
    "DTEND:20260601T150000Z\r\n"
    "SUMMARY:Engineering standup\r\n"
    "LOCATION:Conference Room A\r\n"
    "END:VEVENT\r\n"
    "END:VCALENDAR\r\n"
)
cal = icalendar.Calendar.from_ical(ICS_SINGLE)
vevents = [c for c in cal.walk("VEVENT")]
check("ICS single-event parse: 1 VEVENT extracted", len(vevents) == 1)

ev = oc._normalise_ical_event(vevents[0])
check("_normalise_ical_event: subject extracted",
      ev is not None and ev.subject == "Engineering standup")
check("_normalise_ical_event: location extracted",
      ev is not None and ev.location == "Conference Room A")
check("_normalise_ical_event: is_all_day=False on timed event",
      ev is not None and ev.is_all_day is False)
check("_normalise_ical_event: tz-aware datetimes returned",
      ev is not None
      and ev.start_local.tzinfo is not None
      and ev.end_local.tzinfo is not None)


# --- Test 7: an all-day event — Microsoft / icalendar use VALUE=DATE ------
ICS_ALLDAY = (
    "BEGIN:VCALENDAR\r\n"
    "PRODID:-//Test//EN\r\n"
    "VERSION:2.0\r\n"
    "BEGIN:VEVENT\r\n"
    "UID:test-allday@example.com\r\n"
    "DTSTART;VALUE=DATE:20260615\r\n"
    "DTEND;VALUE=DATE:20260616\r\n"
    "SUMMARY:Memorial Day holiday\r\n"
    "END:VEVENT\r\n"
    "END:VCALENDAR\r\n"
)
cal = icalendar.Calendar.from_ical(ICS_ALLDAY)
vevents = list(cal.walk("VEVENT"))
ev = oc._normalise_ical_event(vevents[0])
check("_normalise_ical_event: all-day flagged",
      ev is not None and ev.is_all_day is True)
check("_normalise_ical_event: all-day subject still parses",
      ev is not None and ev.subject == "Memorial Day holiday")
# Regression: an all-day DTSTART:20260615 must land on 2026-06-15 LOCALLY in any
# timezone. The old code stamped the naive midnight UTC then converted to local,
# shifting the date backward (to 06-14) in behind-UTC zones. start_local.date()
# == the source date is the timezone-independent invariant the fix guarantees.
check("_normalise_ical_event: all-day date not shifted by TZ conversion",
      ev is not None and ev.start_local.date().isoformat() == "2026-06-15")


# --- Test 8: recurring-ical-events expands RRULE into instances ----------
import recurring_ical_events  # noqa: E402

ICS_WEEKLY = (
    "BEGIN:VCALENDAR\r\n"
    "PRODID:-//Test//EN\r\n"
    "VERSION:2.0\r\n"
    "BEGIN:VEVENT\r\n"
    "UID:test-weekly@example.com\r\n"
    "DTSTART:20260601T130000Z\r\n"   # Monday 1pm UTC
    "DTEND:20260601T133000Z\r\n"
    "SUMMARY:Weekly standup\r\n"
    "RRULE:FREQ=WEEKLY;BYDAY=MO\r\n"
    "END:VEVENT\r\n"
    "END:VCALENDAR\r\n"
)
cal = icalendar.Calendar.from_ical(ICS_WEEKLY)
# Window: 4 Mondays starting June 1, 2026 → expect 4 instances
window_start = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
window_end = datetime(2026, 6, 29, 0, 0, tzinfo=timezone.utc)
expanded = recurring_ical_events.of(cal).between(window_start, window_end)
check("recurring-ical-events expands FREQ=WEEKLY into 4 instances",
      len(expanded) == 4)
normalised_count = sum(1 for ve in expanded
                       if oc._normalise_ical_event(ve) is not None)
check("all 4 expanded VEVENTs normalise cleanly",
      normalised_count == 4)


# --- Test 9: dispatcher branches (configured vs not) ---------------------
# Mock _fetch_events_ical to verify the dispatcher routes to it when ICAL_URL
# is set, and returns the setup message when it isn't.
_ical_calls: list = []


def _fake_ical(start, end):
    _ical_calls.append((start, end))
    return ([], "")


_orig_ical = oc._fetch_events_ical
_orig_url = oc.ICAL_URL
oc._fetch_events_ical = _fake_ical
try:
    # (a) ICAL_URL set ⇒ iCal backend used
    oc.ICAL_URL = "https://example.com/calendar.ics"
    _ical_calls.clear()
    oc._fetch_events(datetime.now(timezone.utc),
                     datetime.now(timezone.utc) + timedelta(hours=1))
    check("dispatcher: ICAL_URL set -> iCal backend used",
          len(_ical_calls) == 1)

    # (b) ICAL_URL empty ⇒ voice-friendly setup message, backend not called
    oc.ICAL_URL = ""
    _ical_calls.clear()
    events, err = oc._fetch_events(datetime.now(timezone.utc),
                                   datetime.now(timezone.utc) + timedelta(hours=1))
    check("dispatcher: not configured -> setup message, backend not called",
          events is None
          and "OUTLOOK_ICAL_URL" in err
          and len(_ical_calls) == 0)
finally:
    oc._fetch_events_ical = _orig_ical
    oc.ICAL_URL = _orig_url


# --- summary --------------------------------------------------------------
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
