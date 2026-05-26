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


# --- summary --------------------------------------------------------------
# Restore env state so subsequent imports in the same session see the
# original value.
if _OLD_CLIENT_ID is not None:
    os.environ["OUTLOOK_CLIENT_ID"] = _OLD_CLIENT_ID
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
