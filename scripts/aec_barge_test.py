"""M88 Phase 2 — tests for the hands-free barge-in pure cores (src/aec_barge.py).

The real-time AEC + alignment is validated live (scripts/aec_barge_live.py); the
model-free logic is unit-tested here:
  - BargeDetector: windowed-RMS + sustain energy gate (the VAD on the cleaned
    signal) — fires once after `sustain` consecutive over-threshold windows,
    re-arms only after dropping below.
  - FarEndBuffer.frame_at: the delay-aligned far-end lookup (the alignment that
    makes the AEC actually cancel) — returns the right samples at the right
    offset, silence in gaps.
  - _resample_to_16k: far-end (24 kHz TTS) -> 16 kHz mic-rate reference.

Hermetic: no model, no audio, no pyaec, no threads.

    python scripts/aec_barge_test.py   # exit 0 = all pass
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aec_barge import (  # noqa: E402
    SAMPLE_RATE, BargeDetector, FarEndBuffer, _resample_to_16k,
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


# --- BargeDetector ---------------------------------------------------------
# win=4 frames, sustain=2 windows → fires after 2 consecutive over-threshold
# windows (8 frames), once, then re-arms after a below-threshold window.
det = BargeDetector(threshold=1000.0, win_frames=4, sustain=2)
fired = [det.push_frame(2000.0) for _ in range(7)]   # 1 full window (4) + 3
check("loud frames: no fire before sustain reached", not any(fired))
check("fires on the 2nd full over-threshold window (8th frame)",
      det.push_frame(2000.0) is True)
check("does NOT re-fire while still loud (stays armed-low)",
      not any(det.push_frame(2000.0) for _ in range(8)))

det2 = BargeDetector(threshold=1000.0, win_frames=4, sustain=2)
check("quiet frames never fire",
      not any(det2.push_frame(200.0) for _ in range(40)))

# Re-arm: loud→fire, drop quiet (a window below), then loud again → fires again.
det3 = BargeDetector(threshold=1000.0, win_frames=4, sustain=2)
[det3.push_frame(2000.0) for _ in range(7)]
check("det3 first fire", det3.push_frame(2000.0) is True)
[det3.push_frame(100.0) for _ in range(4)]   # one quiet window → re-arm
[det3.push_frame(2000.0) for _ in range(7)]
check("det3 re-fires after a quiet window re-arms it",
      det3.push_frame(2000.0) is True)

# A single loud window (sustain not met) then quiet → never fires.
det4 = BargeDetector(threshold=1000.0, win_frames=4, sustain=2)
[det4.push_frame(2000.0) for _ in range(4)]   # one loud window
check("one loud window alone does not fire (sustain=2)",
      not any(det4.push_frame(100.0) for _ in range(8)))


# --- FarEndBuffer.frame_at: delay-aligned lookup ---------------------------
fb = FarEndBuffer()
arr = np.arange(2000, dtype=np.int16)          # a ramp so indices are obvious
t0 = 100.0
fb.push(arr, SAMPLE_RATE, start_time=t0)
delay = 0.16

# Echo arriving at t0+delay was emitted at t0 → far frame = arr[0:n].
f = fb.frame_at(t0 + delay, 256, delay)
check("frame_at maps t_emit=t0 to the segment start", np.array_equal(f, arr[:256]))

# 160 samples (10 ms) later → arr[160:160+256].
f2 = fb.frame_at(t0 + delay + 160 / SAMPLE_RATE, 256, delay)
check("frame_at offsets by the elapsed time", np.array_equal(f2, arr[160:160 + 256]))

# A time before anything was played → silence.
f3 = fb.frame_at(t0 - 1.0, 256, delay)
check("frame_at returns silence before any segment", np.array_equal(f3, np.zeros(256, np.int16)))

# A time well after the segment ended (a gap) → silence.
f4 = fb.frame_at(t0 + delay + 10.0, 256, delay)
check("frame_at returns silence in a gap", np.array_equal(f4, np.zeros(256, np.int16)))

# Pruning: a segment far older than retain is dropped on the next push.
fb2 = FarEndBuffer(retain_sec=1.0)
fb2.push(np.ones(1000, np.int16), SAMPLE_RATE, start_time=10.0)
fb2.push(np.ones(1000, np.int16), SAMPLE_RATE, start_time=20.0)  # 10s later
check("old segments pruned past retain window",
      len(fb2._segments) == 1)


# --- _resample_to_16k ------------------------------------------------------
check("16k passes through unchanged length",
      _resample_to_16k(np.arange(320, dtype=np.int16), 16000).size == 320)
check("32k -> 16k halves the length (~)",
      abs(_resample_to_16k(np.arange(320, dtype=np.int16), 32000).size - 160) <= 1)
check("24k -> 16k is 2/3 length (~)",
      abs(_resample_to_16k(np.arange(300, dtype=np.int16), 24000).size - 200) <= 1)
stereo = np.stack([np.arange(100), np.arange(100)], axis=1).astype(np.int16)
check("stereo collapses to mono", _resample_to_16k(stereo, 16000).ndim == 1)
check("empty in -> empty out", _resample_to_16k(np.zeros(0, np.int16), 24000).size == 0)


print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
