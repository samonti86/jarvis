"""M83 — tests for the anticipatory-intelligence engine.

Hermetic: no network, no real Claude call, no threads. The LLM call and the
world-state snapshot are injected, so we exercise the engine's decision logic
(PASS vs announce, dedup, fail-soft, recent-context threading) plus the two pure
helpers (_format_world_state, _parse_decision).

    python tests/anticipation_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.anticipation import (  # noqa: E402
    AnticipationEngine,
    _format_world_state,
    _parse_decision,
    is_enabled,
)

_passed = 0
_failed = 0


def check(label: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")


NOW = datetime(2026, 6, 12, 14, 30, 0)


# --- _parse_decision -------------------------------------------------------
check("plain PASS -> None", _parse_decision("PASS") is None)
check("PASS. -> None", _parse_decision("PASS.") is None)
check("PASS! -> None", _parse_decision("PASS!") is None)
check("quoted PASS -> None", _parse_decision('"PASS"') is None)
check("PASS with short tail -> None", _parse_decision("PASS — nothing notable.") is None)
check("empty -> None", _parse_decision("") is None)
check("whitespace -> None", _parse_decision("   \n ") is None)
check("a real insight is returned",
      _parse_decision("Sir, your 2pm overlaps the storm window.")
      == "Sir, your 2pm overlaps the storm window.")
check("a quoted insight is unwrapped",
      _parse_decision('"Sir, the homelab is back up."') == "Sir, the homelab is back up.")
# A genuine sentence that merely contains the substring "pass" must survive.
check("sentence containing 'pass' is NOT swallowed",
      _parse_decision("Sir, your flight boarding pass is ready and the gate changed to B12.")
      is not None)


# --- _format_world_state ---------------------------------------------------
fws = _format_world_state(
    {"Calendar": "2pm standup", "Weather": "", "Reminders": None,
     "Security": "Disarmed — home"},
    NOW, [],
)
check("formatter includes the current time", "Current time:" in fws)
check("formatter includes a non-empty section", "[Calendar]" in fws and "2pm standup" in fws)
check("formatter SKIPS an empty section", "[Weather]" not in fws)
check("formatter SKIPS a None section", "[Reminders]" not in fws)
check("formatter includes another non-empty section", "[Security]" in fws)
check("no recent list -> no 'already mentioned' block", "Already mentioned" not in fws)

fws2 = _format_world_state({"Calendar": "x"}, NOW, ["You said A", "You said B"])
check("recent list -> 'already mentioned' block present", "Already mentioned" in fws2)
check("recent items are listed", "You said A" in fws2 and "You said B" in fws2)


# --- Engine harness --------------------------------------------------------
class _Eng:
    """Build an engine with injected announce + ask + snapshot."""
    def __init__(self, ask_returns, snapshot=None, raise_snapshot=False,
                 raise_ask=False):
        self.announced: list[str] = []
        self.ask_calls: list[tuple[str, str]] = []
        self._ask_returns = ask_returns
        self._raise_ask = raise_ask
        self._raise_snapshot = raise_snapshot
        snap = snapshot if snapshot is not None else {"Calendar": "2pm standup"}

        def _snapshot():
            if self._raise_snapshot:
                raise RuntimeError("snapshot boom")
            return snap

        def _ask(system, user):
            self.ask_calls.append((system, user))
            if self._raise_ask:
                raise RuntimeError("ask boom")
            # ask_returns may be a list (one per call) or a single string.
            if isinstance(self._ask_returns, list):
                return self._ask_returns.pop(0) if self._ask_returns else "PASS"
            return self._ask_returns

        self.engine = AnticipationEngine(
            announce=self.announced.append,
            snapshot_fn=_snapshot,
            api_key="test-key",
            ask_fn=_ask,
        )


# PASS → silent, no announce.
h = _Eng(ask_returns="PASS")
ret = h.engine._run_once(NOW)
check("PASS tick -> returns None", ret is None)
check("PASS tick -> nothing announced", h.announced == [])
check("PASS tick -> the LLM was still consulted", len(h.ask_calls) == 1)

# Insight → announced + recorded.
h = _Eng(ask_returns="Sir, your 2pm overlaps the storm warning.")
ret = h.engine._run_once(NOW)
check("insight tick -> returns the insight",
      ret == "Sir, your 2pm overlaps the storm warning.")
check("insight tick -> announced once",
      h.announced == ["Sir, your 2pm overlaps the storm warning."])

# Dedup: the same insight on a later tick is suppressed.
h = _Eng(ask_returns=["Sir, the homelab laptop is unreachable.",
                      "Sir, the homelab laptop is unreachable."])
first = h.engine._run_once(NOW)
second = h.engine._run_once(NOW)
check("first occurrence announces", first is not None and len(h.announced) == 1)
check("duplicate occurrence is suppressed (not re-announced)",
      second is None and len(h.announced) == 1)

# Recent insights are threaded into the next prompt (so the model won't repeat).
h = _Eng(ask_returns=["Sir, trash day is tomorrow.", "PASS"])
h.engine._run_once(NOW)
h.engine._run_once(NOW)
_, second_user = h.ask_calls[1]
check("a prior insight is fed back into the next prompt's 'already mentioned'",
      "Already mentioned" in second_user and "trash day is tomorrow" in second_user)

# Empty snapshot → no LLM call at all (nothing to synthesize).
h = _Eng(ask_returns="PASS", snapshot={"Calendar": "", "Weather": None})
ret = h.engine._run_once(NOW)
check("empty snapshot -> returns None", ret is None)
check("empty snapshot -> LLM NOT called", h.ask_calls == [])

# Snapshot raises → fail-soft (None, no crash, no announce).
h = _Eng(ask_returns="PASS", raise_snapshot=True)
check("snapshot raise -> None (no crash)", h.engine._run_once(NOW) is None)
check("snapshot raise -> nothing announced", h.announced == [])

# Ask raises → fail-soft.
h = _Eng(ask_returns="PASS", raise_ask=True)
check("ask raise -> None (no crash)", h.engine._run_once(NOW) is None)
check("ask raise -> nothing announced", h.announced == [])

# Activate without an API key is a no-op (never starts a keyless loop).
keyless = AnticipationEngine(announce=lambda _t: None,
                             snapshot_fn=lambda: {"x": "y"},
                             api_key="", ask_fn=lambda s, u: "PASS")
keyless.activate()
check("activate with no API key -> stays inactive", keyless.is_active() is False)


# --- is_enabled (env) ------------------------------------------------------
import os  # noqa: E402
_old = os.environ.get("JARVIS_ANTICIPATION")
os.environ["JARVIS_ANTICIPATION"] = "1"
check("is_enabled True when =1", is_enabled() is True)
os.environ["JARVIS_ANTICIPATION"] = "0"
check("is_enabled False when =0", is_enabled() is False)
os.environ.pop("JARVIS_ANTICIPATION", None)
check("is_enabled False when unset (default OFF)", is_enabled() is False)
if _old is not None:
    os.environ["JARVIS_ANTICIPATION"] = _old


print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
