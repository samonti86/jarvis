"""M88 Phase 2 — tests for the hands-free barge-in pure cores (src/aec_barge.py).

The real-time AEC + alignment is validated live (scripts/aec_barge_live.py); the
model-free logic is unit-tested here:
  - BargeDetector: windowed-RMS + sustain energy gate (the VAD on the cleaned
    signal) — fires once after `sustain` consecutive over-threshold windows,
    re-arms only after dropping below.
  - DuplexBargePlayer._next_out: the playback-buffer assembly fed to the duplex
    stream — pulls frames across array boundaries, silence on underrun, sets
    _saw_end at the feed sentinel.
  - _resample_to_16k: TTS (24 kHz) -> 16 kHz duplex playback + AEC rate.

Hermetic: no model, no audio, no pyaec, no threads.

    python tests/aec_barge_test.py   # exit 0 = all pass
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aec_barge import (  # noqa: E402
    BargeDetector, DuplexBargePlayer, _resample_to_16k,
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
# win=4 frames, sustain=2 windows -> fires after 2 consecutive over-threshold
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

# Re-arm: loud->fire, drop quiet (a window below), then loud again -> fires again.
det3 = BargeDetector(threshold=1000.0, win_frames=4, sustain=2)
[det3.push_frame(2000.0) for _ in range(7)]
check("det3 first fire", det3.push_frame(2000.0) is True)
[det3.push_frame(100.0) for _ in range(4)]   # one quiet window -> re-arm
[det3.push_frame(2000.0) for _ in range(7)]
check("det3 re-fires after a quiet window re-arms it",
      det3.push_frame(2000.0) is True)

# A single loud window (sustain not met) then quiet -> never fires.
det4 = BargeDetector(threshold=1000.0, win_frames=4, sustain=2)
[det4.push_frame(2000.0) for _ in range(4)]   # one loud window
check("one loud window alone does not fire (sustain=2)",
      not any(det4.push_frame(100.0) for _ in range(8)))


# --- DuplexBargePlayer._next_out: playback assembly + end detection --------
# Pure buffer logic — no stream, no pyaec (start() lazy-imports those).
p = DuplexBargePlayer(None, threading.Event())
p.feed(np.arange(1, 257, dtype=np.int16))            # exactly one frame
out = p._next_out(256)
check("_next_out returns the fed frame", np.array_equal(out, np.arange(1, 257, dtype=np.int16)))
check("_next_out: no end sentinel yet", p._saw_end is False)

# Spanning two fed arrays in one frame.
p2 = DuplexBargePlayer(None, threading.Event())
p2.feed(np.full(100, 7, dtype=np.int16))
p2.feed(np.full(300, 9, dtype=np.int16))
out2 = p2._next_out(256)
check("_next_out spans array boundaries",
      np.array_equal(out2[:100], np.full(100, 7, np.int16))
      and np.array_equal(out2[100:256], np.full(156, 9, np.int16)))

# Underrun: nothing fed -> silence, NOT an end.
p3 = DuplexBargePlayer(None, threading.Event())
out3 = p3._next_out(256)
check("_next_out underrun -> silence", np.array_equal(out3, np.zeros(256, np.int16)))
check("_next_out underrun is not end-of-feed", p3._saw_end is False)

# End sentinel: remaining samples, then trailing silence + _saw_end set.
p4 = DuplexBargePlayer(None, threading.Event())
p4.feed(np.full(50, 5, dtype=np.int16))
p4.done_feeding()
out4 = p4._next_out(256)
check("_next_out drains remaining before end",
      np.array_equal(out4[:50], np.full(50, 5, np.int16))
      and np.array_equal(out4[50:], np.zeros(206, np.int16)))
check("_next_out sets _saw_end at the sentinel", p4._saw_end is True)
check("_has_pending false after full drain + end", p4._has_pending() is False)

# _has_pending true while fed data remains.
p5 = DuplexBargePlayer(None, threading.Event())
p5.feed(np.full(500, 3, dtype=np.int16))
p5._next_out(256)
check("_has_pending true with remaining data", p5._has_pending() is True)


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
