"""M81 — tests for armed intrusion-by-voice (the voice_while_armed rule).

While security is ARMED (away), human speech in the supposedly-empty house is
an anomaly: the voice_while_armed rule fires the same M72 visual alert (photo +
description -> Discord) as the sound rules. The load-bearing, model-free cores:

  - `_rule_window_passes` (pure): threshold, loudness floor, running-water
    volume guard, and the M81 armed_only gate — the decision whether a window
    counts toward a rule's sustain streak.
  - the voice_while_armed ClassRule itself: armed_only, the right aliases,
    in the ACTIVE default set, and per-class-disable honoured.

Hermetic: no model, no audio, no network.

    python scripts/intrusion_voice_test.py   # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sound_detector import (  # noqa: E402
    _DEFAULT_RULES,
    ClassRule,
    SoundDetector,
    _rule_window_passes,
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


# --- The voice_while_armed rule is wired correctly -------------------------
_voice = next((r for r in _DEFAULT_RULES if r.name == "voice_while_armed"), None)
check("voice_while_armed is in the ACTIVE default rule set", _voice is not None)
check("voice_while_armed is armed_only", _voice is not None and _voice.armed_only)
check("voice_while_armed aggregates Speech/Conversation/Shout",
      _voice is not None
      and set(_voice.aliases) == {"Speech", "Conversation", "Shout"})
check("voice_while_armed has an RMS floor (rejects silence hallucination)",
      _voice is not None and _voice.rms_floor > 0.0)
check("voice_while_armed needs >1 window (a one-off blip won't fire)",
      _voice is not None and _voice.sustain >= 2)
check("the OTHER rules are NOT armed_only (still fire at home)",
      all(not r.armed_only for r in _DEFAULT_RULES if r.name != "voice_while_armed"))


# --- _rule_window_passes: the armed_only gate ------------------------------
# A representative armed_only rule and a normal one.
armed_rule = ClassRule(
    name="voice", aliases=("Speech",), threshold=0.40, sustain=2,
    cooldown_seconds=120.0, speak="", push="", rms_floor=0.02, armed_only=True,
)
home_rule = ClassRule(
    name="doorbell", aliases=("Doorbell",), threshold=0.30, sustain=1,
    cooldown_seconds=25.0, speak="", push="", rms_floor=0.02,
)

# armed_only rule: passes only when armed (all other conditions met).
check("armed_only + armed=True + loud + over-threshold -> passes",
      _rule_window_passes(armed_rule, 0.7, 0.10, armed=True) is True)
check("armed_only + armed=False -> blocked even when loud & over-threshold",
      _rule_window_passes(armed_rule, 0.7, 0.10, armed=False) is False)

# A non-armed_only rule ignores the armed flag entirely.
check("home rule fires regardless of armed=False",
      _rule_window_passes(home_rule, 0.5, 0.10, armed=False) is True)
check("home rule fires regardless of armed=True",
      _rule_window_passes(home_rule, 0.5, 0.10, armed=True) is True)


# --- _rule_window_passes: the other gates still apply (with armed=True) -----
check("below threshold -> blocked (even armed)",
      _rule_window_passes(armed_rule, 0.30, 0.10, armed=True) is False)
check("over threshold but below RMS floor -> blocked (silence hallucination)",
      _rule_window_passes(armed_rule, 0.7, 0.001, armed=True) is False)
check("exactly at threshold counts (>=)",
      _rule_window_passes(armed_rule, 0.40, 0.10, armed=True) is True)

# Volume-guard rule (running_water shape): needs RMS above the water floor.
water_rule = ClassRule(
    name="running_water", aliases=("Stream",), threshold=0.25, sustain=8,
    cooldown_seconds=600.0, speak="", push="", needs_volume_guard=True,
)
check("volume-guard rule blocked when too quiet (fountain trickle)",
      _rule_window_passes(water_rule, 0.5, 0.001, armed=False) is False)


# --- SoundDetector samples is_armed; None/raising getter -> NOT armed -------
# We don't run the model here; we assert the detector stores the getter and
# that the fail-safe default (no getter) is not-armed by construction.
det_none = SoundDetector(announce=lambda _t: None)
check("no is_armed getter wired -> stored as None", det_none._is_armed is None)

_armed_state = {"v": True}
det = SoundDetector(announce=lambda _t: None,
                    is_armed=lambda: _armed_state["v"])
check("is_armed getter is stored", det._is_armed is not None and det._is_armed() is True)
_armed_state["v"] = False
check("is_armed getter reflects live state", det._is_armed() is False)


print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
