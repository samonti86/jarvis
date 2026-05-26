"""Regression test for M62 Outlook calendar awareness.

Exercises everything that doesn't require a live Microsoft Graph round-trip:
schema shape, the no-token-cached error path, timeframe → window resolution,
Graph dict → CalendarEvent normalisation (handles fractional seconds, UTC
conversion, missing fields), and the event formatter (empty / single /
multi-event, all-day, with-and-without location).

What it INTENTIONALLY does NOT test: the live Graph HTTP call (would need a
test tenant + token) and the interactive MSAL device-code flow (manual by
nature; that's `scripts/outlook_auth.py`'s job).

    python scripts/outlook_calendar_test.py     # exit 0 = pass
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure OUTLOOK_CLIENT_ID is unset for the "not configured" test paths.
# (msal install isn't needed unless we actually instantiate the client.)
_OLD_CLIENT_ID = os.environ.pop("OUTLOOK_CLIENT_ID", None)

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


# --- Test 2: CLIENT_ID unset -> tool returns voice-friendly setup msg ----
# (We popped OUTLOOK_CLIENT_ID above; the module read it at import time and
# is now stuck with the empty value — perfect for this test. Reload would
# reset it; we want to verify the unconfigured path triggers cleanly.)
out = oc.execute_calendar_tool({"timeframe": "today"})
check("CLIENT_ID unset -> tool says 'isn't configured'",
      "configured" in out.lower() and "OUTLOOK_CLIENT_ID" in out)


# --- Test 3: invalid timeframe is rejected with a voice-friendly error ---
oc.CLIENT_ID = "dummy"  # bypass the not-configured branch for THIS test
out = oc.execute_calendar_tool({"timeframe": "yesterday"})
check("invalid timeframe -> voice-friendly error mentions valid options",
      "yesterday" in out and "today" in out and "tomorrow" in out)
oc.CLIENT_ID = ""  # restore


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


# --- Test 5: _parse_graph_dt handles Graph's quirky ISO formats ----------
# Standard ISO with no TZ -> treated as UTC.
dt = oc._parse_graph_dt("2026-05-26T13:30:00")
check("parse_graph_dt naive ISO -> UTC-tagged datetime",
      dt is not None and dt.tzinfo == timezone.utc and dt.hour == 13)

# Trailing 'Z' for UTC.
dt = oc._parse_graph_dt("2026-05-26T13:30:00Z")
check("parse_graph_dt 'Z' suffix -> UTC", dt is not None and dt.tzinfo is not None)

# Graph sometimes sends 7 fractional-second digits — Python's
# fromisoformat handles 6 (microseconds) but Graph's ".0000000" can choke.
dt = oc._parse_graph_dt("2026-05-26T13:30:00.0000000")
check("parse_graph_dt fractional-second fallback handles Graph's '.0000000'",
      dt is not None and dt.hour == 13)

# Empty string -> None.
check("parse_graph_dt empty string -> None", oc._parse_graph_dt("") is None)


# --- Test 6: _normalise handles a typical Graph event ---------------------
ev_dict = {
    "subject": "Engineering standup",
    "start": {"dateTime": "2026-05-26T13:30:00.0000000", "timeZone": "UTC"},
    "end": {"dateTime": "2026-05-26T14:00:00.0000000", "timeZone": "UTC"},
    "location": {"displayName": "Conference Room A"},
    "isAllDay": False,
}
ev = oc._normalise(ev_dict)
check("_normalise typical event -> CalendarEvent",
      ev is not None
      and ev.subject == "Engineering standup"
      and ev.location == "Conference Room A"
      and not ev.is_all_day)

# Missing optional fields shouldn't crash.
ev_min = oc._normalise({
    "subject": "",
    "start": {"dateTime": "2026-05-26T15:00:00"},
    "end": {"dateTime": "2026-05-26T16:00:00"},
})
check("_normalise missing fields -> defaults applied",
      ev_min is not None
      and ev_min.subject == "(no subject)"
      and ev_min.location == "")

# Truly malformed -> None (logged + skipped).
check("_normalise garbage -> None",
      oc._normalise({"start": {"dateTime": "not-a-date"},
                      "end": {"dateTime": "also-not"}}) is None)


# --- Test 7: _format_events handles the various cases --------------------
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


# --- M62.1 — iCal backend + dispatcher precedence ------------------------

# Test 8: _normalise_ical_event on a real icalendar VEVENT
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


# Test 9: an all-day event — Microsoft / icalendar use VALUE=DATE (no time)
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


# Test 10: recurring-ical-events expands RRULE into instances inside window
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


# Test 11: dispatcher precedence — iCal wins when both configured
# We mock _fetch_events_ical and _fetch_events_graph to verify which one
# the dispatcher actually calls under each env combination.
_ical_calls = []
_graph_calls = []

def _fake_ical(start, end):
    _ical_calls.append((start, end))
    return ([], "")

def _fake_graph(start, end):
    _graph_calls.append((start, end))
    return ([], "")

_orig_ical = oc._fetch_events_ical
_orig_graph = oc._fetch_events_graph
_orig_url = oc.ICAL_URL
_orig_client_id = oc.CLIENT_ID
oc._fetch_events_ical = _fake_ical
oc._fetch_events_graph = _fake_graph
try:
    # (a) Both set ⇒ iCal wins
    oc.ICAL_URL = "https://example.com/calendar.ics"
    oc.CLIENT_ID = "some-client-id"
    _ical_calls.clear(); _graph_calls.clear()
    oc._fetch_events(datetime.now(timezone.utc),
                     datetime.now(timezone.utc) + timedelta(hours=1))
    check("dispatcher: both configured -> iCal wins",
          len(_ical_calls) == 1 and len(_graph_calls) == 0)

    # (b) Only Graph set ⇒ Graph used
    oc.ICAL_URL = ""
    oc.CLIENT_ID = "some-client-id"
    _ical_calls.clear(); _graph_calls.clear()
    oc._fetch_events(datetime.now(timezone.utc),
                     datetime.now(timezone.utc) + timedelta(hours=1))
    check("dispatcher: only CLIENT_ID -> Graph used",
          len(_ical_calls) == 0 and len(_graph_calls) == 1)

    # (c) Only iCal set ⇒ iCal used
    oc.ICAL_URL = "https://example.com/calendar.ics"
    oc.CLIENT_ID = ""
    _ical_calls.clear(); _graph_calls.clear()
    oc._fetch_events(datetime.now(timezone.utc),
                     datetime.now(timezone.utc) + timedelta(hours=1))
    check("dispatcher: only ICAL_URL -> iCal used",
          len(_ical_calls) == 1 and len(_graph_calls) == 0)

    # (d) Neither set ⇒ voice-friendly setup message
    oc.ICAL_URL = ""
    oc.CLIENT_ID = ""
    events, err = oc._fetch_events(datetime.now(timezone.utc),
                                   datetime.now(timezone.utc) + timedelta(hours=1))
    check("dispatcher: neither configured -> setup message",
          events is None
          and "OUTLOOK_ICAL_URL" in err
          and "OUTLOOK_CLIENT_ID" in err)
finally:
    oc._fetch_events_ical = _orig_ical
    oc._fetch_events_graph = _orig_graph
    oc.ICAL_URL = _orig_url
    oc.CLIENT_ID = _orig_client_id


# --- summary --------------------------------------------------------------
# Restore env state so subsequent imports in the same session see the
# original value.
if _OLD_CLIENT_ID is not None:
    os.environ["OUTLOOK_CLIENT_ID"] = _OLD_CLIENT_ID
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
