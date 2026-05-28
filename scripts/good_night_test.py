"""M63 — unit tests for the "good night" wrap composition.

The hard parts of M63 are:

  - per-section fail-soft behaviour (a dead weather API, an unconfigured
    home location, a missing security getter) — each must degrade THAT
    section, never blank the whole wrap.
  - the security-getter injection — unset, armed, standing-down, raising.
  - the reminders-tomorrow filter — only items whose fire_at.date() ==
    tomorrow's local date make it in.
  - the calendar section's "tomorrow's FIRST timed event" pick (all-day
    skip, earliest start).
  - the M59 + M63 reminders dispatch — the action enum, dispatch dict,
    label helpers.

Same instrument discipline as scripts/calendar_monitor_test.py: a
standalone asserting harness, no network needed (we stub the network-
dependent sections via env / temp-store manipulation; the live weather
call is covered by the M55 briefing's own live path).

    python scripts/good_night_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point the reminders store at a temp file BEFORE the module loads anything
# (the path is computed lazily, so this works as long as we set LOCALAPPDATA
# before any add() is called).
_tmp = tempfile.TemporaryDirectory()
os.environ["LOCALAPPDATA"] = _tmp.name

from src import good_night  # noqa: E402
from src.good_night import (  # noqa: E402
    GOOD_NIGHT_TOOL,
    execute_good_night_tool,
    register_security_getter,
    _calendar_section,
    _reminders_section,
    _security_section,
)
from src import reminders  # noqa: E402
from src.reminders import (  # noqa: E402
    SET_REMINDER_TOOL,
    _COMPOSITION_ACTIONS,
    _action_label,
    _action_listing_tag,
    add,
    cancel,
)


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


# --- GOOD_NIGHT_TOOL schema ----------------------------------------------

print("\nGOOD_NIGHT_TOOL schema:")
check("tool has name", GOOD_NIGHT_TOOL.get("name") == "get_good_night")
check("tool has description",
      isinstance(GOOD_NIGHT_TOOL.get("description"), str)
      and len(GOOD_NIGHT_TOOL["description"]) > 50)
check("tool description mentions 'good night'",
      "good night" in GOOD_NIGHT_TOOL["description"].lower())
check("tool description mentions tomorrow",
      "tomorrow" in GOOD_NIGHT_TOOL["description"].lower())


# --- _security_section: getter unset / armed / standing-down / raises ----

print("\n_security_section:")

# Reset to a clean slate (test isolation — set_reminder ran via import order
# above might have stuck a getter from a previous run on disk, though not in
# this process). The module-level singleton is process-local.
good_night._security_getter = None
out = _security_section()
check("getter unset -> silent omit (empty string)", out == "")

register_security_getter(lambda: True)
out = _security_section()
check("getter says armed -> 'armed for the night'",
      "armed" in out.lower())

register_security_getter(lambda: False)
out = _security_section()
check("getter says standing down -> 'standing down'",
      "standing down" in out.lower())


def _raising_getter() -> bool:
    raise RuntimeError("boom")


register_security_getter(_raising_getter)
out = _security_section()
check("getter raises -> degraded line, not a crash",
      "unavailable" in out.lower())

# Restore a sensible default for the remaining tests.
register_security_getter(lambda: False)


# --- _reminders_section: filters to tomorrow's date ----------------------

print("\n_reminders_section:")

# Clean any leftover reminders from a previous failed run.
for r in list(reminders.list_pending()):
    cancel(rid=r["id"])

tomorrow_dt = datetime.now() + timedelta(days=1)
yesterday_dt = datetime.now() - timedelta(days=1)
day_after_dt = datetime.now() + timedelta(days=2)

# Add one tomorrow, one yesterday (already-past = wouldn't be in pending in
# real life but defensively a corrupt store could surface it), one two days
# out. Only the tomorrow one should appear.
add("call the dentist", tomorrow_dt.replace(hour=9, minute=0,
                                            second=0, microsecond=0))
add("two days out", day_after_dt.replace(hour=10, minute=0,
                                         second=0, microsecond=0))

out = _reminders_section()
check("'call the dentist' (tomorrow) appears", "call the dentist" in out)
check("'two days out' is filtered out", "two days out" not in out)
check("section header says 'Reminders tomorrow (1)'",
      "Reminders tomorrow (1)" in out)

# Empty case
for r in list(reminders.list_pending()):
    cancel(rid=r["id"])
out = _reminders_section()
check("no tomorrow reminders -> 'none scheduled'",
      "none scheduled" in out.lower())


# --- _calendar_section: silent-omit when no backend configured -----------

print("\n_calendar_section:")

# Save + clear backend env so the test is independent of the user's setup.
saved_ical = os.environ.pop("OUTLOOK_ICAL_URL", None)
saved_cid = os.environ.pop("OUTLOOK_CLIENT_ID", None)
# The outlook_calendar module reads these at IMPORT, so the constants are
# already populated. We need to patch them on the module object.
from src import outlook_calendar  # noqa: E402
saved_mod_ical = outlook_calendar.ICAL_URL
saved_mod_cid = outlook_calendar.CLIENT_ID
outlook_calendar.ICAL_URL = ""
outlook_calendar.CLIENT_ID = ""

out = _calendar_section()
check("no calendar backend -> silent omit (empty string)", out == "")

# Restore env + module constants.
outlook_calendar.ICAL_URL = saved_mod_ical
outlook_calendar.CLIENT_ID = saved_mod_cid
if saved_ical is not None:
    os.environ["OUTLOOK_ICAL_URL"] = saved_ical
if saved_cid is not None:
    os.environ["OUTLOOK_CLIENT_ID"] = saved_cid


# --- execute_good_night_tool: composes + never raises --------------------

print("\nexecute_good_night_tool:")

# Clear reminders + ensure unconfigured calendar so the wrap is deterministic.
for r in list(reminders.list_pending()):
    cancel(rid=r["id"])
outlook_calendar.ICAL_URL = ""
outlook_calendar.CLIENT_ID = ""
# Force weather unconfigured so we don't hit the network in tests.
saved_home = os.environ.pop("JARVIS_HOME_LOCATION", None)

register_security_getter(lambda: True)
out = execute_good_night_tool({})
check("composed wrap is a non-empty string",
      isinstance(out, str) and len(out) > 10)
check("composed wrap contains the security section",
      "armed" in out.lower())
check("composed wrap contains the reminders section",
      "reminders tomorrow" in out.lower())
check("composed wrap contains weather (or its 'not configured' hint)",
      "weather" in out.lower())
check("composed wrap omits the unconfigured calendar section",
      "calendar" not in out.lower())

# Restore weather env.
if saved_home is not None:
    os.environ["JARVIS_HOME_LOCATION"] = saved_home


# --- M59 + M63 reminders dispatch hooks ----------------------------------

print("\nreminders dispatch (M63 share with M59):")

schema_props = SET_REMINDER_TOOL["input_schema"]["properties"]
check("action enum lists both briefing and good_night",
      set(schema_props["action"]["enum"]) == {"briefing", "good_night"})
check("_COMPOSITION_ACTIONS contains both",
      set(_COMPOSITION_ACTIONS.keys()) == {"briefing", "good_night"})

check("_action_label(briefing, cap=True) -> 'Scheduled briefing'",
      _action_label("briefing", capitalised=True) == "Scheduled briefing")
check("_action_label(good_night, cap=True) -> 'Evening wrap'",
      _action_label("good_night", capitalised=True) == "Evening wrap")
check("_action_label(None, cap=True) -> 'Reminder'",
      _action_label(None, capitalised=True) == "Reminder")
check("_action_label(None, cap=False) -> 'reminder'",
      _action_label(None, capitalised=False) == "reminder")

check("_action_listing_tag(briefing) -> ' (briefing)'",
      _action_listing_tag("briefing") == " (briefing)")
check("_action_listing_tag(good_night) -> ' (good night)'",
      _action_listing_tag("good_night") == " (good night)")
check("_action_listing_tag(None) -> ''",
      _action_listing_tag(None) == "")


# --- summary --------------------------------------------------------------

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
