"""Tonal awareness (M85) — "You sound tired, sir."

Film-JARVIS reads Tony's state, not just his words. M85 gives Jarvis a light
read of HOW the user spoke — volume, pace, pauses — from the audio clip STT
already captured (the same `Transcript.audio` M69 uses), and threads a short
DESCRIPTIVE note into the LLM call so Claude can infer the mood and adapt
(gentler + more concise when he sounds tired, fast when rushed, match an upbeat
energy). Per the design choice: we feed Claude the CUES and let it interpret —
we never hard-label an emotion here.

Deliberately conservative on two fronts:
  - **Cues, not diagnosis.** We report "softer than usual, slow pace", not
    "tired". Claude does the reading; the system prompt tells it to use the note
    SUBTLY (calibrate tone, don't robotically announce the mood).
  - **Only when NOTABLE.** Neutral delivery (normal volume, measured pace,
    fluent) produces NO note — most turns are silent here. A note appears only
    when a dimension deviates, which is exactly when "how he said it" is signal.

Robust + setup-independent where it can be:
  - Pace (words/sec) and pause ratio are time-based → independent of mic gain.
  - Loudness is judged RELATIVE to a rolling baseline of the user's own recent
    clips ("softer/louder than usual"), so it doesn't depend on absolute gain;
    omitted until a few clips establish the baseline.

numpy-only, no new dependency, no model. Never raises into the turn.
"""

from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

SAMPLE_RATE = 16_000   # STT captures int16 mono @ 16 kHz (Transcript.audio)

# Minimums below which there isn't enough signal to judge delivery.
_MIN_DURATION = 0.6
_MIN_WORDS = 2
# Baseline needs a few clips before "softer/louder than usual" means anything.
_MIN_BASELINE = 5


def is_enabled() -> bool:
    """Default ON (numpy-only, ~free) — kill with JARVIS_TONE=0."""
    return os.getenv("JARVIS_TONE", "1").strip().lower() not in ("0", "false", "no", "off")


# --- pure feature extraction + labels (no state — unit-testable) -----------

def _features(audio: "np.ndarray | None", sample_rate: int, word_count: int) -> "dict | None":
    """Cheap acoustic features from an int16 mono clip, or None when there's
    too little to judge. Pure; no I/O."""
    if audio is None or getattr(audio, "size", 0) == 0:
        return None
    f = audio.astype(np.float32) / 32768.0
    dur = len(f) / float(sample_rate or SAMPLE_RATE)
    if dur < _MIN_DURATION or word_count < _MIN_WORDS:
        return None
    rms = float(np.sqrt(np.mean(f * f))) if f.size else 0.0
    # Pause ratio: fraction of 30 ms frames whose energy is far below the clip's
    # OWN peak (so it's a within-utterance hesitation measure, gain-independent).
    frame = max(1, int(0.03 * (sample_rate or SAMPLE_RATE)))
    n = len(f) // frame
    if n >= 4:
        fr = f[: n * frame].reshape(n, frame)
        fr_rms = np.sqrt(np.mean(fr * fr, axis=1))
        peak = float(fr_rms.max()) or 1e-9
        silence_ratio = float(np.mean(fr_rms < peak * 0.15))
    else:
        silence_ratio = 0.0
    pace = word_count / max(dur, 0.1)
    return {"dur": dur, "rms": rms, "pace": pace, "silence_ratio": silence_ratio}


def _pace_label(pace_wps: float) -> "str | None":
    """Conversational English runs ~2.5-3 words/sec; flag only the extremes."""
    if pace_wps < 1.8:
        return "slow, deliberate pace"
    if pace_wps > 3.7:
        return "fast, rushed pace"
    return None  # measured/brisk — not notable


def _pause_label(silence_ratio: float) -> "str | None":
    if silence_ratio >= 0.42:
        return "halting, with frequent pauses"
    return None  # fluent / a few pauses — not notable


def _loudness_label(rms: float, baseline_median: "float | None") -> "str | None":
    if not baseline_median or baseline_median <= 1e-9:
        return None  # no baseline yet
    ratio = rms / baseline_median
    if ratio < 0.7:
        return "softer than usual"
    if ratio > 1.45:
        return "louder, more forceful than usual"
    return None


def _compose(pace_l, pause_l, loud_l) -> "str | None":
    """Join the NOTABLE descriptors into one cue string, or None when delivery
    is unremarkable (the common case → no note threaded into the turn)."""
    parts = [p for p in (loud_l, pace_l, pause_l) if p]
    if not parts:
        return None
    return ", ".join(parts)


# --- stateful analyzer (holds the rolling loudness baseline) ----------------

class ToneAnalyzer:
    """One per session. Holds a rolling baseline of recent speech loudness so
    'softer/louder than usual' is judged against the user's own voice, not an
    absolute gain. `analyze` returns a short cue string or None."""

    def __init__(self, baseline_keep: int = 24) -> None:
        self._rms: deque = deque(maxlen=max(_MIN_BASELINE, baseline_keep))

    def analyze(self, audio: "np.ndarray | None", transcript: str,
                sample_rate: int = SAMPLE_RATE) -> "str | None":
        """Compute a delivery cue for this clip (and fold its loudness into the
        baseline). Never raises — any failure returns None."""
        try:
            word_count = len((transcript or "").split())
            feats = _features(audio, sample_rate, word_count)
            if feats is None:
                return None
            baseline = self._median()
            # Update baseline AFTER reading it, so a clip is judged against the
            # PRIOR distribution, not partly against itself.
            self._rms.append(feats["rms"])
            return _compose(
                _pace_label(feats["pace"]),
                _pause_label(feats["silence_ratio"]),
                _loudness_label(feats["rms"], baseline),
            )
        except Exception as exc:  # noqa: BLE001 — tone is a nicety, never break a turn
            print(f"[tone] analyze failed: {exc}", file=sys.stderr)
            return None

    def _median(self) -> "float | None":
        if len(self._rms) < _MIN_BASELINE:
            return None
        return float(np.median(np.array(self._rms, dtype=np.float32)))
