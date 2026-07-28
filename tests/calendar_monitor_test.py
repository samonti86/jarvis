"""M62.2 — unit tests for the calendar reminder decision logic + dedupe store.

The hard, must-be-correct cores of M62.2 are:

  - `_should_announce(ev, now, lead_min, announced)` — the pure firing rule
    (all-day skip / already-started skip / lead-window / dedupe).
  - `DedupeStore`                                   — load / mark / save /
    prune across-restart behaviour.

Both are isolated from fetch / announce / Discord I/O so they're exercised
deterministically here — no waiting for a real meeting. Same instrument
discipline as tests/homelab_alert_test.py and scripts/leak_repro.py: a
standalone asserting harness so a regression in the firing rule is caught
structurally.

    python tests/calendar_monitor_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Allow `python tests/calendar_monitor_test.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.outlook_calendar import CalendarEvent  # noqa: E402
from src.calendar_monitor import (  # noqa: E402
    DedupeStore,
    _event_id,
    _should_announce,
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


# --- helpers ---------------------------------------------------------------

def make_event(
    subject: str = "Standup",
    *,
    starts_in_min: float = 10,
    duration_min: float = 30,
    all_day: bool = False,
    location: str = "",
    now: datetime | None = None,
) -> CalendarEvent:
    """Build a synthetic CalendarEvent relative to `now` (defaults to wall
    clock). Keeps tests independent of the machine's actual time."""
    base = now if now is not None else datetime.now().astimezone()
    start = base + timedelta(minutes=starts_in_min)
    end = start + timedelta(minutes=duration_min)
    return CalendarEvent(
        subject=subject,
        start_local=start,
        end_local=end,
        location=location,
        is_all_day=all_day,
    )


NOW = datetime.now().astimezone()


# --- _event_id ------------------------------------------------------------

print("\n_event_id:")

ev_a = make_event("Standup", starts_in_min=10, now=NOW)
ev_b = make_event("Standup", starts_in_min=10, now=NOW)
check("identical events -> same id", _event_id(ev_a) == _event_id(ev_b))

ev_c = make_event("1:1 with manager", starts_in_min=10, now=NOW)
check("different subject -> different id", _event_id(ev_a) != _event_id(ev_c))

ev_d = make_event("Standup", starts_in_min=70, now=NOW)
check("rescheduled (same subject, new start) -> different id",
      _event_id(ev_a) != _event_id(ev_d))


# --- _should_announce ----------------------------------------------------

print("\n_should_announce:")

# In-window event, empty dedupe -> fires.
ev = make_event(starts_in_min=10, now=NOW)
fire, mins = _should_announce(ev, NOW, lead_min=15, announced=set())
check("in-window (10m), no dedupe -> fires", fire is True)
check("in-window (10m) -> mins == 10", mins == 10)

# Same event with dedupe populated -> suppressed.
fire, _ = _should_announce(ev, NOW, lead_min=15, announced={_event_id(ev)})
check("already announced -> suppressed", fire is False)

# Outside the lead window (> lead_min minutes away).
ev_far = make_event(starts_in_min=30, now=NOW)
fire, mins = _should_announce(ev_far, NOW, lead_min=15, announced=set())
check("30m away with 15m lead -> not yet", fire is False)
check("30m away -> mins still reported (30)", mins == 30)

# At the lead-window edge — exactly 15.0 min away with 15 lead -> fires.
ev_edge = make_event(starts_in_min=15, now=NOW)
fire, mins = _should_announce(ev_edge, NOW, lead_min=15, announced=set())
check("at the lead edge (15m, lead 15) -> fires", fire is True)
check("at the lead edge -> mins == 15", mins == 15)

# Just past lead edge — 15.6 min rounds to 16, suppressed.
ev_past_edge = make_event(starts_in_min=15.6, now=NOW)
fire, mins = _should_announce(ev_past_edge, NOW, lead_min=15, announced=set())
check("15m36s away rounds to 16 -> not yet", fire is False)

# Already started (seconds <= 0).
ev_started = make_event(starts_in_min=-5, now=NOW)
fire, _ = _should_announce(ev_started, NOW, lead_min=15, announced=set())
check("already started (-5m) -> never fires (no late nag)", fire is False)

# Edge: starts in 30 seconds (still future, rounds to 0 or 1).
ev_imminent = make_event(starts_in_min=0.5, now=NOW)
fire, _ = _should_announce(ev_imminent, NOW, lead_min=15, announced=set())
check("30s in the future -> still fires (in window, not started)", fire is True)

# All-day events skip — even when 'in-window' by clock math.
ev_allday = make_event(starts_in_min=5, all_day=True, now=NOW)
fire, _ = _should_announce(ev_allday, NOW, lead_min=15, announced=set())
check("all-day event -> never fires (briefing covers these)", fire is False)


# --- DedupeStore ---------------------------------------------------------

print("\nDedupeStore:")

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "ds.json"

    # 1: fresh store on a non-existent path is empty.
    s = DedupeStore(p)
    check("fresh store / no file -> empty", len(s) == 0)
    check("has() on empty store -> False", s.has("anything") is False)

    # 2: mark + has, then save persists to disk.
    s.mark("foo")
    s.mark("bar")
    check("mark+has -> True", s.has("foo") and s.has("bar"))
    check("len reflects marks", len(s) == 2)
    s.save()
    check("save writes the file", p.exists())

    # 3: re-construct from the same path -> loads marks.
    s2 = DedupeStore(p)
    check("re-construct from disk -> marks reloaded",
          s2.has("foo") and s2.has("bar"))

    # 4: snapshot returns a separate copy.
    snap = s2.snapshot()
    snap.add("transient")
    check("snapshot is decoupled from internal set",
          not s2.has("transient"))

    # 5: prune on save — short retain window drops everything.
    s3 = DedupeStore(p, retain_hours=0)
    s3.save()                              # prunes (all timestamps fail the cutoff)
    s4 = DedupeStore(p, retain_hours=0)
    check("retain_hours=0 -> save prunes all entries", len(s4) == 0)

# 6: a corrupt JSON file -> empty store, no crash.
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "corrupt.json"
    p.write_text("{not really json", encoding="utf-8")
    s = DedupeStore(p)
    check("corrupt JSON -> empty store, no crash", len(s) == 0)

# 7: wrong shape (top-level list instead of dict) -> empty.
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "list.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    s = DedupeStore(p)
    check("list-shaped JSON -> empty store", len(s) == 0)

# 8: entries with bad ISO timestamps are skipped at load.
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "mixed.json"
    good = datetime.now().isoformat()
    payload = {"keep": good, "drop_bad_ts": "not-a-date", "drop_nondt": 42}
    p.write_text(json.dumps(payload), encoding="utf-8")
    s = DedupeStore(p)
    check("bad timestamps / non-string values are filtered",
          s.has("keep") and not s.has("drop_bad_ts") and not s.has("drop_nondt"))


# --- summary --------------------------------------------------------------

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
