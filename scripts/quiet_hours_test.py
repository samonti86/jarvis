"""M79 — regression test for src/quiet_hours.py (Do-Not-Disturb policy).

Hermetic: JARVIS_QUIET_HOURS set per-test via env, the deferred store pointed at
a temp LOCALAPPDATA, and a fixed `now` passed in (no clock mocking). Covers
window parsing, the overnight-wrap is_quiet logic, the decide() pierce/defer
policy, the deferred store round-trip + stale-drop + cap, formatting, and the
fail-soft contract.

    python scripts/quiet_hours_test.py     # exit 0 = pass
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import quiet_hours as qh  # noqa: E402

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


@contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- Test 1: window parsing -----------------------------------------------
check("parse valid window", qh._parse_window("22:00-08:00") is not None)
check("parse malformed -> None", qh._parse_window("10pm to 8am") is None)
check("parse out-of-range -> None", qh._parse_window("25:00-08:00") is None)
check("parse empty -> None", qh._parse_window("") is None)


# --- Test 2: is_quiet — overnight wrap ------------------------------------
with _env(JARVIS_QUIET_HOURS="22:00-08:00"):
    check("overnight: 3 AM is quiet",
          qh.is_quiet(datetime(2026, 6, 11, 3, 0)) is True)
    check("overnight: 11 PM is quiet",
          qh.is_quiet(datetime(2026, 6, 11, 23, 0)) is True)
    check("overnight: noon is NOT quiet",
          qh.is_quiet(datetime(2026, 6, 11, 12, 0)) is False)
    check("overnight: exactly 08:00 is NOT quiet (end exclusive)",
          qh.is_quiet(datetime(2026, 6, 11, 8, 0)) is False)
    check("overnight: exactly 22:00 IS quiet (start inclusive)",
          qh.is_quiet(datetime(2026, 6, 11, 22, 0)) is True)


# --- Test 3: is_quiet — same-day window + disabled ------------------------
with _env(JARVIS_QUIET_HOURS="13:00-14:00"):
    check("same-day: 13:30 is quiet",
          qh.is_quiet(datetime(2026, 6, 11, 13, 30)) is True)
    check("same-day: 09:00 is NOT quiet",
          qh.is_quiet(datetime(2026, 6, 11, 9, 0)) is False)
with _env(JARVIS_QUIET_HOURS=None):
    check("DND unset -> never quiet", qh.is_quiet(datetime(2026, 6, 11, 3, 0)) is False)
with _env(JARVIS_QUIET_HOURS="08:00-08:00"):
    check("zero-width window -> never quiet",
          qh.is_quiet(datetime(2026, 6, 11, 8, 0)) is False)


# --- Test 4: decide — pierce vs defer -------------------------------------
night = datetime(2026, 6, 11, 3, 0)
day = datetime(2026, 6, 11, 12, 0)
with _env(JARVIS_QUIET_HOURS="22:00-08:00"):
    check("quiet + homelab (routine) -> defer", qh.decide("🖥", night) == "defer")
    check("quiet + security -> speak (pierce)", qh.decide("🚨", night) == "speak")
    check("quiet + reminder -> speak (pierce)", qh.decide("⏰", night) == "speak")
    check("quiet + weather -> speak (pierce)", qh.decide("⛈", night) == "speak")
    check("quiet + acoustic -> speak (pierce)", qh.decide("🔔", night) == "speak")
    check("quiet + calendar -> speak (pierce)", qh.decide("📅", night) == "speak")
    check("quiet + presence -> speak (pierce)", qh.decide("🏠", night) == "speak")
    check("quiet + UNKNOWN label -> speak (safe default)",
          qh.decide("🆕", night) == "speak")
    check("daytime + homelab -> speak (outside window)",
          qh.decide("🖥", day) == "speak")
with _env(JARVIS_QUIET_HOURS=None):
    check("DND off + homelab -> speak", qh.decide("🖥", night) == "speak")


# --- Test 5: deferred store round-trip + take clears ----------------------
with tempfile.TemporaryDirectory() as tmp:
    with _env(LOCALAPPDATA=tmp):
        now = datetime(2026, 6, 11, 3, 0)
        qh.record_deferred("MEDIA-HOST unreachable", "🖥", now)
        qh.record_deferred("media drive low on space", "🖥", now)
        taken = qh.take_deferred(datetime(2026, 6, 11, 8, 0))
        again = qh.take_deferred(datetime(2026, 6, 11, 8, 0))
    check("two deferred announces stored + returned",
          len(taken) == 2 and taken[0]["text"] == "MEDIA-HOST unreachable")
    check("take_deferred clears the store", again == [])


# --- Test 6: stale deferred items dropped ---------------------------------
with tempfile.TemporaryDirectory() as tmp:
    with _env(LOCALAPPDATA=tmp):
        old_ts = datetime(2026, 6, 9, 3, 0)      # ~2 days before the read
        new_ts = datetime(2026, 6, 11, 3, 0)
        qh.record_deferred("ancient note", "🖥", old_ts)
        qh.record_deferred("fresh note", "🖥", new_ts)
        taken = qh.take_deferred(datetime(2026, 6, 11, 8, 0))
    check("stale (>18h) deferred item dropped, fresh kept",
          len(taken) == 1 and taken[0]["text"] == "fresh note")


# --- Test 7: store is capped ----------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    with _env(LOCALAPPDATA=tmp):
        now = datetime(2026, 6, 11, 3, 0)
        for i in range(60):
            qh.record_deferred(f"note {i}", "🖥", now)
        taken = qh.take_deferred(datetime(2026, 6, 11, 4, 0))
    check("store capped at _MAX_DEFERRED (newest kept)",
          len(taken) == qh._MAX_DEFERRED and taken[-1]["text"] == "note 59")


# --- Test 8: format_deferred ----------------------------------------------
txt = qh.format_deferred([
    {"ts": datetime(2026, 6, 11, 2, 14).isoformat(), "text": "MEDIA-HOST unreachable", "label": "🖥"},
])
check("format: header + the note present",
      "While you were away" in txt and "MEDIA-HOST unreachable" in txt)
check("format: empty list -> ''", qh.format_deferred([]) == "")


# --- Test 9: fail-soft — corrupt store, decide never raises ---------------
with tempfile.TemporaryDirectory() as tmp:
    with _env(LOCALAPPDATA=tmp):
        (Path(tmp) / "Jarvis").mkdir(parents=True, exist_ok=True)
        (Path(tmp) / "Jarvis" / "deferred_announces.json").write_text("{bad", encoding="utf-8")
        check("corrupt store -> take_deferred returns []",
              qh.take_deferred(datetime(2026, 6, 11, 8, 0)) == [])
with _env(JARVIS_QUIET_HOURS="not-a-window"):
    check("malformed window -> decide speaks (fail-soft)",
          qh.decide("🖥", datetime(2026, 6, 11, 3, 0)) == "speak")


# --- summary --------------------------------------------------------------
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
