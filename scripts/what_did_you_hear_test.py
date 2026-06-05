"""Unit tests for the what_did_you_hear tool (M76 — acoustic recall).

The testable core is the pure `_summarize_sounds` formatter — no threads, no
model, no audio. We feed it synthetic (ts, label, score, rms) observations and
(ts, name) fires on a fixed monotonic clock and assert the three honest states
(off / quiet / summary), age phrasing, label aggregation (peak + recency),
the item cap, and the age-window filter. Plus the executor's registration +
minutes-param handling via a tiny fake detector.

    python scripts/what_did_you_hear_test.py   # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import sound_detector as sd  # noqa: E402
from src.sound_detector import (  # noqa: E402
    WHAT_DID_YOU_HEAR_TOOL,
    _ago,
    _summarize_sounds,
    execute_what_did_you_hear,
    register_detector,
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


NOW = 1000.0  # fixed "monotonic now" for deterministic age math


def obs(age, label, score, rms=0.1):
    """A recent-window observation `age` seconds before NOW."""
    return (NOW - age, label, score, rms)


def fire(age, name):
    return (NOW - age, name)


# --- Schema --------------------------------------------------------------

print("\nschema:")
check("tool name", WHAT_DID_YOU_HEAR_TOOL.get("name") == "what_did_you_hear")
check("description mentions hearing / sound",
      "hear" in WHAT_DID_YOU_HEAR_TOOL["description"].lower())
check("minutes is optional (not required)",
      "required" not in WHAT_DID_YOU_HEAR_TOOL["input_schema"]
      or not WHAT_DID_YOU_HEAR_TOOL["input_schema"].get("required"))


# --- _ago phrasing -------------------------------------------------------

print("\n_ago phrasing:")
check("<10s -> 'just now'", _ago(3) == "just now")
check("40s -> '~40s ago'", _ago(40) == "~40s ago")
check("89s -> seconds form", _ago(89).endswith("s ago"))
check("200s -> '~3m ago'", _ago(200) == "~3m ago")


# --- State: acoustic off -------------------------------------------------

print("\nstate: off:")
out = _summarize_sounds(False, [obs(5, "Speech", 0.6)], [], NOW, 180.0)
check("not active -> 'off' message regardless of buffer",
      "off" in out.lower() and "not actively listening" in out.lower())


# --- State: active but quiet --------------------------------------------

print("\nstate: quiet:")
out = _summarize_sounds(True, [], [], NOW, 180.0)
check("active + empty -> 'been quiet'", "been quiet" in out.lower())
# Stale-only observations (older than the window) also read as quiet.
out = _summarize_sounds(True, [obs(600, "Music", 0.5)], [], NOW, 180.0)
check("only-stale observations -> 'been quiet' (age filter)",
      "been quiet" in out.lower())


# --- Summary: soundscape aggregation ------------------------------------

print("\nsoundscape aggregation:")
recent = [
    obs(5, "Speech", 0.62),
    obs(8, "Speech", 0.40),     # same label — peak should stay 0.62, recent 5s
    obs(50, "Music", 0.31),
    obs(120, "Dog", 0.22),
]
out = _summarize_sounds(True, recent, [], NOW, 180.0)
check("lists Speech with its PEAK score (0.62, not 0.40)",
      "Speech (peak 0.62" in out)
check("Speech ordered before Music (more recent first)",
      out.index("Speech") < out.index("Music"))
check("Music present", "Music (peak 0.31" in out)
check("Dog present", "Dog (peak 0.22" in out)
check("header names the window in minutes", "last 3 minutes" in out.lower())
check("Speech recency reads 'just now' (5s)", "just now" in out)


# --- Summary: fired alerts reported distinctly + first -------------------

print("\nfired alerts:")
out = _summarize_sounds(
    True,
    [obs(40, "Doorbell", 0.45), obs(5, "Speech", 0.6)],
    [fire(40, "doorbell")],
    NOW, 180.0,
)
check("fired alert line present", "alerts that fired" in out.lower())
check("fired alert names the event (doorbell)", "doorbell" in out.lower())
check("fired underscore name humanized (glass_break -> 'glass break')",
      "glass break" in _summarize_sounds(
          True, [], [fire(10, "glass_break")], NOW, 180.0).lower())
check("alerts line comes before soundscape line",
      out.lower().index("alerts that fired") < out.lower().index("sounds detected"))


# --- Age-window filter ---------------------------------------------------

print("\nage-window filter:")
recent = [obs(30, "Speech", 0.6), obs(400, "Music", 0.5)]
out3 = _summarize_sounds(True, recent, [], NOW, 180.0)  # 3 min
check("3-min window excludes the 400s-old Music",
      "Speech" in out3 and "Music" not in out3)
out15 = _summarize_sounds(True, recent, [], NOW, 900.0)  # 15 min
check("15-min window includes the 400s-old Music",
      "Speech" in out15 and "Music" in out15)


# --- Item cap ------------------------------------------------------------

print("\nitem cap:")
many = [obs(i, f"Label{i}", 0.5) for i in range(1, 11)]  # 10 distinct labels
out = _summarize_sounds(True, many, [], NOW, 900.0, max_items=6)
shown = sum(1 for i in range(1, 11) if f"Label{i} " in out)
check("caps soundscape at max_items (6)", shown == 6)
check("overflow tail '+N more' present", "more)" in out)


# --- Executor: registration + minutes param -----------------------------

print("\nexecutor:")


class _FakeDetector:
    def __init__(self):
        self.calls = []

    def recent_sounds_summary(self, max_age_seconds=180.0):
        self.calls.append(max_age_seconds)
        return f"summary@{int(max_age_seconds)}"


# No detector registered -> friendly not-set-up message.
register_detector(None)
out = execute_what_did_you_hear({})
check("no detector registered -> 'isn't set up'", "isn't set up" in out.lower())

fake = _FakeDetector()
register_detector(fake)
try:
    out = execute_what_did_you_hear({})
    check("default window is 3 minutes (180s)", fake.calls[-1] == 180.0)
    execute_what_did_you_hear({"minutes": 10})
    check("minutes=10 -> 600s", fake.calls[-1] == 600.0)
    execute_what_did_you_hear({"minutes": 99})
    check("minutes clamped to max 15 (900s)", fake.calls[-1] == 900.0)
    execute_what_did_you_hear({"minutes": 0})
    check("minutes clamped to min 1 (60s)", fake.calls[-1] == 60.0)
    execute_what_did_you_hear({"minutes": "garbage"})
    check("garbage minutes -> default 180s", fake.calls[-1] == 180.0)
finally:
    register_detector(None)  # don't leak the fake into other tests


# --- Executor: detector raising is swallowed -----------------------------

print("\nexecutor fail-soft:")


class _RaisingDetector:
    def recent_sounds_summary(self, max_age_seconds=180.0):
        raise RuntimeError("boom")


register_detector(_RaisingDetector())
try:
    out = execute_what_did_you_hear({})
    check("detector raises -> friendly fallback (no crash)",
          "couldn't read" in out.lower())
finally:
    register_detector(None)


# --- summary --------------------------------------------------------------

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
