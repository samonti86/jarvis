"""M85 — tests for tonal awareness (voice_tone).

Hermetic: synthetic int16 audio, no mic, no model. Exercises the pure feature
extraction + label thresholds, the "only notable delivery yields a cue" rule,
and the ToneAnalyzer's rolling-baseline loudness ("softer/louder than usual").

    python tests/voice_tone_test.py   # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.voice_tone import (  # noqa: E402
    SAMPLE_RATE,
    ToneAnalyzer,
    _compose,
    _features,
    _loudness_label,
    _pace_label,
    _pause_label,
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


def tone(duration_s: float, peak: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A constant 200 Hz tone of the given peak amplitude (RMS ~ peak/sqrt2),
    no pauses."""
    t = np.arange(int(duration_s * sr))
    return (peak * np.sin(2 * np.pi * 200 * t / sr)).astype(np.int16)


def words(n: int) -> str:
    return " ".join(["word"] * n)


# --- _features -------------------------------------------------------------
check("too-short clip -> None", _features(tone(0.3, 8000), SAMPLE_RATE, 5) is None)
check("too-few-words -> None", _features(tone(2.0, 8000), SAMPLE_RATE, 1) is None)
check("empty audio -> None", _features(np.array([], dtype=np.int16), SAMPLE_RATE, 5) is None)
check("None audio -> None", _features(None, SAMPLE_RATE, 5) is None)
f = _features(tone(2.8, 8000), SAMPLE_RATE, 7)
check("valid clip -> features dict", f is not None and "pace" in f and "rms" in f)
check("pace computed (7 words / 2.8s ~ 2.5)", f is not None and 2.3 < f["pace"] < 2.7)
check("constant tone -> ~no pauses", f is not None and f["silence_ratio"] < 0.1)


# --- label thresholds ------------------------------------------------------
check("slow pace flagged", _pace_label(0.7) == "slow, deliberate pace")
check("fast pace flagged", _pace_label(4.5) == "fast, rushed pace")
check("normal pace -> None", _pace_label(2.6) is None)
check("halting pauses flagged", _pause_label(0.5) == "halting, with frequent pauses")
check("fluent -> None", _pause_label(0.1) is None)
check("no baseline -> no loudness label", _loudness_label(5000, None) is None)
check("softer than baseline flagged",
      _loudness_label(1000, 5000) == "softer than usual")
check("louder than baseline flagged",
      _loudness_label(9000, 5000) == "louder, more forceful than usual")
check("near baseline -> None", _loudness_label(5200, 5000) is None)


# --- _compose --------------------------------------------------------------
check("all-None -> None (neutral delivery = no note)", _compose(None, None, None) is None)
check("one notable -> that cue", _compose(None, "slow, deliberate pace", None)
      == "slow, deliberate pace")
check("loudness leads the joined cue",
      _compose("softer than usual", "slow, deliberate pace", None)
      == "softer than usual, slow, deliberate pace")


# --- ToneAnalyzer: neutral vs notable + the rolling baseline ----------------
a = ToneAnalyzer()
# A neutral clip (normal pace, no pauses, no baseline yet) -> no cue.
check("neutral clip -> no cue", a.analyze(tone(2.8, 8000), words(7)) is None)
# A slow clip -> a cue regardless of baseline.
check("slow clip -> a cue", "slow" in (a.analyze(tone(3.0, 8000), words(2)) or ""))

# Loudness baseline: feed several normal-volume neutral clips, THEN a quiet one.
b = ToneAnalyzer()
for _ in range(6):
    b.analyze(tone(2.8, 8000), words(7))     # rms ~ 5657, neutral
quiet = b.analyze(tone(2.8, 1500), words(7))  # rms ~ 1060 -> ratio ~0.19
check("quiet clip after baseline -> 'softer than usual' cue",
      quiet is not None and "softer than usual" in quiet)

# Before the baseline is established, loudness is NOT judged.
c = ToneAnalyzer()
early = c.analyze(tone(2.8, 1500), words(7))   # first ever clip, neutral pace
check("first quiet clip (no baseline) -> no loudness cue", early is None)

# Never raises on a junk transcript.
check("junk transcript doesn't raise",
      a.analyze(tone(2.8, 8000), None) is None or True)


# --- is_enabled (default ON) -----------------------------------------------
_old = os.environ.get("JARVIS_TONE")
os.environ.pop("JARVIS_TONE", None)
check("default ON when unset", is_enabled() is True)
os.environ["JARVIS_TONE"] = "0"
check("off when =0", is_enabled() is False)
os.environ["JARVIS_TONE"] = "1"
check("on when =1", is_enabled() is True)
if _old is None:
    os.environ.pop("JARVIS_TONE", None)
else:
    os.environ["JARVIS_TONE"] = _old


print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
