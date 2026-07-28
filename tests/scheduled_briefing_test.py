"""M59 — regression test for the scheduled-briefing wiring.

Verifies the contract of set_reminder + add() + the scheduler dispatch for the
new `action='briefing'` path: the schema exposes the field, the record stores
it, the greeting helper returns the right time-of-day phrase, and an unknown
action is rejected with a voice-friendly error.

Same instrument discipline as tests/homelab_alert_test.py and
tests/acoustic_alert_test.py — a regression here would silently break
"brief me every weekday at 7" without py_compile noticing.

    python tests/scheduled_briefing_test.py     # exit 0 = pass
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reminders import (
    add, cancel, SET_REMINDER_TOOL, execute_set_reminder,
    _greeting_for,
)

schema_props = SET_REMINDER_TOOL["input_schema"]["properties"]
assert "action" in schema_props, "action field missing from schema"
# Asserted as a SUPERSET, not equality: this suite is about the M59 briefing
# action, and pinning the exact enum made it fail every time an unrelated
# action was added (M92's background_task). Assert what this file is
# responsible for; good_night_test.py owns the full-set assertion.
_enum = set(schema_props["action"]["enum"])
assert {"briefing", "good_night"} <= _enum, _enum
print(f"schema: action field present, enum={sorted(_enum)}")

rec = add("test briefing", datetime.now() + timedelta(seconds=3600),
          action="briefing")
assert rec.get("action") == "briefing", rec
print(f"add(action=briefing): stored OK id={rec['id']}")
cancel(rid=rec["id"])
print("cleanup: cancelled")

assert _greeting_for(7) == "Good morning, sir."
assert _greeting_for(14) == "Good afternoon, sir."
assert _greeting_for(20) == "Good evening, sir."
print("greeting: morning/afternoon/evening OK")

res = execute_set_reminder({"message": "x", "delay_seconds": 10, "action": "xxx"})
assert "xxx" in res and "briefing" in res and "good_night" in res, res
print(f"unknown action rejected: {res!r}")

print("ALL OK")
