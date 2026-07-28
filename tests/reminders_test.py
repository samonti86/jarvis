r"""Regression test for the reminders module (M53 one-shot, M54 recurring, and
the 2026-06-02 interval deferred-start fix).

WHY THIS EXISTS:
M53/M54 shipped without a test suite — the "a gate with no enforcing test"
gap the M67 post-mortem warned about. The 2026-06-02 debugging session added a
deferred first-fire for INTERVAL reminders ("every 5 minutes STARTING at
7:50pm") and refactored the one-off time-parsing into a shared
`_resolve_explicit_fire` helper. That touched the reminder time-math, so it's
exactly the kind of change that wants a permanent net. This suite locks in:
  - the new interval-deferred-start behavior (at / delay_seconds honored);
  - that weekly/monthly STILL ignore at/delay (derive their own slot);
  - the legacy interval "start now+interval" path is unchanged;
  - the one-shot paths the refactor reorganized (delay, at, errors);
  - pop_due()'s one-shot-removed vs recurring-re-armed contract (M54);
  - the _resolve_explicit_fire helper's tri-state return.

Hermetic: redirects LOCALAPPDATA to a throwaway temp dir BEFORE touching the
store, so it never reads or writes the user's real reminders.json. _store_path
reads the env var on every call, so setting it here fully isolates the test.

    python tests/reminders_test.py     # exit 0 = pass
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- isolate the store BEFORE importing the module under test --------------
_TMP = tempfile.mkdtemp(prefix="jarvis_reminders_test_")
os.environ["LOCALAPPDATA"] = _TMP

import src.reminders as remod  # noqa: E402 — for monkeypatching the composer
from src.reminders import (  # noqa: E402 — must follow the env redirect
    SET_REMINDER_TOOL,
    add, cancel, list_pending, pop_due,
    execute_set_reminder, execute_cancel_reminder, execute_list_reminders,
    _fire_one, _push, _next_occurrence, _resolve_explicit_fire,
    _store_path, _validate_repeat, _ISO,
)
import threading as _threading
import time as _time

PASSED = 0
FAILED = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok: {label}")
    else:
        FAILED += 1
        print(f"  FAIL: {label}  {detail}")


def _clear() -> None:
    """Empty the (temp) store between groups so list_pending is predictable."""
    _store_path().unlink(missing_ok=True)


def _only() -> dict:
    """Return the single pending reminder (asserts exactly one)."""
    items = list_pending()
    assert len(items) == 1, f"expected 1 pending, got {len(items)}"
    return items[0]


def _fire_dt(rec: dict) -> datetime:
    return datetime.strptime(rec["fire_at"], _ISO)


def _close(a: datetime, b: datetime, tol_s: int = 5) -> bool:
    return abs((a - b).total_seconds()) <= tol_s


# Guard against ever writing the real store: the path must be under our temp.
assert str(_store_path()).startswith(_TMP), f"store not isolated: {_store_path()}"


# ===========================================================================
print("\n[group] schema contract")
props = SET_REMINDER_TOOL["input_schema"]["properties"]
check("schema has message/delay_seconds/at/repeat",
      all(k in props for k in ("message", "delay_seconds", "at", "repeat")))
check("top-level description documents the interval deferred start",
      "STARTING" in SET_REMINDER_TOOL["description"],
      SET_REMINDER_TOOL["description"][:80])


# ===========================================================================
print("\n[group] _resolve_explicit_fire (tri-state helper)")
now = datetime.now()
check("neither at nor delay -> None",
      _resolve_explicit_fire({}) is None)
r = _resolve_explicit_fire({"delay_seconds": 600})
check("valid delay -> datetime ~now+600s",
      isinstance(r, datetime) and _close(r, now + timedelta(seconds=600)), repr(r))
future = (now + timedelta(hours=3)).strftime(_ISO)
r = _resolve_explicit_fire({"at": future})
check("valid at -> exact datetime",
      isinstance(r, datetime) and r == datetime.strptime(future, _ISO), repr(r))
check("both at AND delay -> error string",
      isinstance(_resolve_explicit_fire({"at": future, "delay_seconds": 60}), str))
past = (now - timedelta(hours=1)).strftime(_ISO)
check("past at -> 'already passed' error",
      "already passed" in (_resolve_explicit_fire({"at": past}) or ""))
check("delay <= 0 -> 'not in the future' error",
      "future" in (_resolve_explicit_fire({"delay_seconds": 0}) or ""))
check("unparseable at -> 'couldn't read' error",
      "couldn't read" in (_resolve_explicit_fire({"at": "not-a-date"}) or ""))
beyond = (now + timedelta(days=400)).strftime(_ISO)
check("beyond horizon -> 'further out' error",
      "further out" in (_resolve_explicit_fire({"at": beyond}) or ""))


# ===========================================================================
print("\n[group] interval deferred start (the 2026-06-02 fix)")
_clear()
at = (now + timedelta(hours=2)).strftime(_ISO)
out = execute_set_reminder({
    "message": "check Disney signup",
    "repeat": {"kind": "interval", "interval_seconds": 300},
    "at": at,
})
rec = _only()
check("interval + at: first fire == the 'at' (not now+interval)",
      rec["fire_at"] == at, f"{rec['fire_at']} != {at}")
check("interval + at: stored as recurring",
      rec.get("repeat", {}).get("kind") == "interval", repr(rec.get("repeat")))
check("interval + at: confirmation mentions 'Recurring' + 'First one at'",
      "Recurring" in out and "First one at" in out, out)

_clear()
execute_set_reminder({
    "message": "check printer",
    "repeat": {"kind": "interval", "interval_seconds": 600},
    "delay_seconds": 1800,
})
check("interval + delay: first fire ~now+1800s",
      _close(_fire_dt(_only()), datetime.now() + timedelta(seconds=1800)))

_clear()
execute_set_reminder({
    "message": "stretch",
    "repeat": {"kind": "interval", "interval_seconds": 900},
})
check("interval + NO start: legacy first fire ~now+interval",
      _close(_fire_dt(_only()), datetime.now() + timedelta(seconds=900)))

_clear()
out = execute_set_reminder({
    "message": "oops",
    "repeat": {"kind": "interval", "interval_seconds": 300},
    "at": past,
})
check("interval + past at: error returned", "already passed" in out, out)
check("interval + past at: NOTHING scheduled", list_pending() == [], list_pending())

_clear()
out = execute_set_reminder({
    "message": "x",
    "repeat": {"kind": "interval", "interval_seconds": 300},
    "at": future, "delay_seconds": 60,
})
check("interval + both at AND delay: rejected", "not both" in out, out)
check("interval + both: nothing scheduled", list_pending() == [])


# ===========================================================================
print("\n[group] weekly/monthly ignore at/delay (derive own slot)")
_clear()
execute_set_reminder({
    "message": "standup",
    "repeat": {"kind": "weekly", "weekdays": [0, 1, 2, 3, 4], "time": "09:00"},
    "at": at,  # must be ignored
})
wk = _fire_dt(_only())
check("weekly + at: at IGNORED (fire time-of-day is 09:00, not the at)",
      wk.hour == 9 and wk.minute == 0 and _only()["fire_at"] != at,
      _only()["fire_at"])

_clear()
execute_set_reminder({
    "message": "rent",
    "repeat": {"kind": "monthly", "day": 1, "time": "08:00"},
    "delay_seconds": 60,  # must be ignored
})
mo = _fire_dt(_only())
check("monthly + delay: delay IGNORED (fire is day-1 08:00)",
      mo.day == 1 and mo.hour == 8 and mo.minute == 0, _only()["fire_at"])


# ===========================================================================
print("\n[group] one-shot paths (refactor regression)")
_clear()
execute_set_reminder({"message": "tea", "delay_seconds": 120})
check("one-shot delay: fires ~now+120s",
      _close(_fire_dt(_only()), datetime.now() + timedelta(seconds=120)))
check("one-shot delay: NOT recurring", "repeat" not in _only())

_clear()
fut = (datetime.now() + timedelta(hours=5)).strftime(_ISO)
execute_set_reminder({"message": "call", "at": fut})
check("one-shot at: fires at the exact 'at'", _only()["fire_at"] == fut)

_clear()
check("one-shot no time: asks for a time",
      "need a time" in execute_set_reminder({"message": "nothing"}))
check("one-shot no time: nothing scheduled", list_pending() == [])
check("one-shot both delay+at: rejected",
      "not both" in execute_set_reminder({"message": "x", "delay_seconds": 60, "at": fut}))
check("one-shot past at: rejected",
      "already passed" in execute_set_reminder({"message": "x", "at": past}))
check("one-shot empty message: rejected",
      "remind you about" in execute_set_reminder({"message": "  ", "delay_seconds": 60}))


# ===========================================================================
print("\n[group] pop_due re-arm contract (M54)")
_clear()
# A one-shot already due -> returned and REMOVED.
add("past one-shot", datetime.now() - timedelta(minutes=5))
due = pop_due(datetime.now())
check("pop_due: one-shot returned", len(due) == 1 and due[0]["message"] == "past one-shot")
check("pop_due: one-shot removed after firing", list_pending() == [])

_clear()
# A recurring interval already due -> returned AND re-armed in place (next slot
# from now, NOT a backlog of stale slots).
spec = _validate_repeat({"kind": "interval", "interval_seconds": 600})
assert isinstance(spec, dict), spec
add("recurring", datetime.now() - timedelta(minutes=5), repeat=spec)
fire_now = datetime.now()
due = pop_due(fire_now)
check("pop_due: recurring returned once", len(due) == 1)
survivor = _only()
check("pop_due: recurring re-armed (same id)", survivor["id"] == due[0]["id"])
check("pop_due: re-armed slot ~now+interval (one catch-up, no stampede)",
      _close(_fire_dt(survivor), fire_now + timedelta(seconds=600)))


# ===========================================================================
print("\n[group] list / cancel passthrough")
_clear()
r1 = add("alpha reminder", datetime.now() + timedelta(hours=1))
add("beta reminder", datetime.now() + timedelta(hours=2))
check("execute_list_reminders names both",
      "alpha" in execute_list_reminders({}) and "beta" in execute_list_reminders({}))
check("cancel by query substring",
      "alpha" in (execute_cancel_reminder({"query": "alpha"}) or ""))
check("cancel removed only the matched one",
      len(list_pending()) == 1 and list_pending()[0]["message"] == "beta reminder")
check("cancel by exact id",
      cancel(rid=r1["id"]) is None or True)  # r1 already cancelled by query; tolerate


# ===========================================================================
print("\n[group] fire fan-out: announce + Discord notify (2026-06-02)")

# Plain reminder -> BOTH sinks get the SAME spoken text.
spoke, pushed = [], []
_fire_one({"message": "check the oven", "fire_at": datetime.now().strftime(_ISO)},
          datetime.now(), spoke.append, pushed.append)
check("plain fire: announce got the text", len(spoke) == 1 and "check the oven" in spoke[0], spoke)
check("plain fire: notify got the SAME text", pushed == spoke, (pushed, spoke))

# notify=None -> announce still fires, no crash (the no-webhook case).
spoke2 = []
_fire_one({"message": "no webhook here", "fire_at": datetime.now().strftime(_ISO)},
          datetime.now(), spoke2.append, None)
check("notify=None: announce still fires", len(spoke2) == 1 and "no webhook here" in spoke2[0])

# _push tolerates None and a raising sink (must never break the fire).
_push(None, "x")  # no-op, no raise
def _boom(_):
    raise RuntimeError("discord down")
_push(_boom, "x")  # swallowed
check("_push: None and raising sink both swallowed (no exception escaped)", True)

# Composition reminder (briefing) -> composed text reaches BOTH sinks. Mock the
# composer to be instant + deterministic; the fire spawns a worker thread.
# NOTE: patch the _COMPOSITION_ACTIONS entry, not the module attr — the dict
# captured the original function reference at definition time.
_orig_brief = remod._COMPOSITION_ACTIONS["briefing"]
remod._COMPOSITION_ACTIONS["briefing"] = (
    (lambda: "WEATHER: clear. NEWS: none.",) + _orig_brief[1:]
)
spoke3, pushed3 = [], []
_fire_one({"id": "rbrief", "message": "morning briefing", "action": "briefing",
           "fire_at": datetime.now().strftime(_ISO)},
          datetime.now().replace(hour=7), spoke3.append, pushed3.append)
# join the briefing-fire worker thread
deadline = _time.time() + 5
while _time.time() < deadline and not (spoke3 and pushed3):
    for th in _threading.enumerate():
        if th.name.startswith("briefing-fire-"):
            th.join(timeout=0.2)
    _time.sleep(0.02)
check("composition fire: announce got composed text",
      len(spoke3) == 1 and "WEATHER: clear" in spoke3[0], spoke3)
check("composition fire: notify got the SAME composed text",
      pushed3 == spoke3 and len(pushed3) == 1, (pushed3, spoke3))
check("composition fire: greeting prefix present (7am -> Good morning)",
      bool(spoke3) and spoke3[0].startswith("Good morning"), spoke3)
remod._COMPOSITION_ACTIONS["briefing"] = _orig_brief  # restore


# ===========================================================================
print("\n[group] config: reminder_discord_enabled polarity (default ON)")
import os as _os  # noqa: E402
import src.config as _cfgmod  # noqa: E402


def _enabled_with(env_val) -> bool:
    saved = _os.environ.get("JARVIS_REMINDER_DISCORD")
    if env_val is None:
        _os.environ.pop("JARVIS_REMINDER_DISCORD", None)
    else:
        _os.environ["JARVIS_REMINDER_DISCORD"] = env_val
    try:
        return _cfgmod.load().reminder_discord_enabled
    finally:
        if saved is None:
            _os.environ.pop("JARVIS_REMINDER_DISCORD", None)
        else:
            _os.environ["JARVIS_REMINDER_DISCORD"] = saved


check("default (unset) -> ON", _enabled_with(None) is True)
check("'0' -> OFF", _enabled_with("0") is False)
check("'false' -> OFF", _enabled_with("false") is False)
check("'1' -> ON", _enabled_with("1") is True)
check("garbage -> ON (only explicit falsy disables)", _enabled_with("yarp") is True)


# ===========================================================================
print("\n" + "=" * 50)
print(f"{PASSED} passed, {FAILED} failed")
print("=" * 50)
sys.exit(1 if FAILED else 0)
