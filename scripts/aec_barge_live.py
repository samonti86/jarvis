"""M88 Phase 2 — REAL-TIME barge validation, DUPLEX (single-clock) approach.

The first version aligned the far-end to the mic by Python wall-clock timestamps
across TWO independent streams (sd.play + a separate InputStream). It FAILED —
45 false fires in the silent run — because those streams' buffer latencies are
large and jittery, so pyaec's reference was smeared and the adaptive filter never
converged (the canceller cancelled nothing; the monitor just detected raw echo,
going quiet only in the inter-segment gaps).

This version uses a SINGLE DUPLEX sd.Stream: play and capture in one callback,
one clock. far and mic are sample-locked with a fixed, jitter-free offset (the
stream's own in+out latency), which is exactly what an adaptive AEC needs. We
feed pyaec the far-end delayed by that offset and detect on the cleaned signal,
all inside the callback. This is the decisive test: if it's clean here, the path
forward is to route Jarvis's TTS playback through a duplex stream too.

  silent — let it play, STAY QUIET.  Expect ~0 detections.
  talk   — talk over it from your usual spot.  Expect detections.

Tune live (no code edit): JARVIS_BARGE_DELAY_MS (try around the printed stream
latency), JARVIS_BARGE_THRESHOLD, JARVIS_BARGE_DEBUG=1.

    venv\\Scripts\\python.exe scripts\\aec_barge_live.py silent
    venv\\Scripts\\python.exe scripts\\aec_barge_live.py talk
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.aec_barge import (  # noqa: E402
    DELAY_MS, FILTER_LEN, FRAME, SAMPLE_RATE, THRESHOLD, BargeDetector,
)
from src.audio import resolve_input_device  # noqa: E402

load_dotenv()

DUR = 14.0


def _speechlike(n: int, amp: float) -> np.ndarray:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n)
    t = np.arange(n) / SAMPLE_RATE
    x = x * (0.15 + 0.85 * np.abs(np.sin(2 * np.pi * 3.0 * t)))
    x /= (np.max(np.abs(x)) + 1e-9)
    return (x * amp).astype(np.int16)


def main() -> None:
    import pyaec  # noqa: PLC0415

    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("silent", "talk"):
        print(__doc__)
        print("\nGive one of: silent | talk")
        sys.exit(2)

    device = resolve_input_device(os.getenv("JARVIS_MIC_DEVICE", "").strip())
    n = int(DUR * SAMPLE_RATE)
    clip = _speechlike(n, amp=0.5 * 32768.0)
    aec = pyaec.Aec(FRAME, FILTER_LEN, SAMPLE_RATE, False)
    detector = BargeDetector()

    delay_frames = max(0, round(DELAY_MS / 1000.0 * SAMPLE_RATE / FRAME))
    far_hist: deque[np.ndarray] = deque(maxlen=delay_frames + 1)
    pos = {"i": 0}
    detections: list[float] = []
    dbg = os.getenv("JARVIS_BARGE_DEBUG", "").strip() not in ("", "0", "false", "False")
    dbg_peak = {"v": 0.0, "t": time.monotonic()}

    def callback(indata, outdata, frames, _t, status):  # noqa: ANN001
        if status:
            print(f"[live] {status}", file=sys.stderr)
        i = pos["i"]
        far_now = clip[i:i + frames]
        if len(far_now) < frames:                      # clip done
            outdata[:len(far_now), 0] = far_now
            outdata[len(far_now):, 0] = 0
            raise sd.CallbackStop
        outdata[:, 0] = far_now
        # far reference delayed by the stream's in+out latency (jitter-free here).
        far_hist.append(far_now.copy())
        far_ref = far_hist[0] if len(far_hist) > delay_frames else np.zeros(frames, np.int16)
        if len(far_ref) != frames:
            far_ref = np.zeros(frames, np.int16)
        mic = indata[:, 0].astype(np.int16)
        cleaned = np.asarray(aec.cancel_echo(mic, far_ref), dtype=np.float64)
        rms = float(np.sqrt(np.mean(np.square(cleaned))))
        if dbg:
            dbg_peak["v"] = max(dbg_peak["v"], rms)
            if time.monotonic() - dbg_peak["t"] > 1.0:
                print(f"  cleaned-RMS peak ~1s: {dbg_peak['v']:.0f} "
                      f"(thresh {THRESHOLD:.0f})", file=sys.stderr)
                dbg_peak["v"], dbg_peak["t"] = 0.0, time.monotonic()
        if detector.push_frame(rms):
            detections.append(i / SAMPLE_RATE)
            print(f"  >>> BARGE @ {i / SAMPLE_RATE:5.1f}s (rms {rms:.0f})", file=sys.stderr)
            detector.reset()
        pos["i"] = i + frames

    if mode == "talk":
        print("\n>>> TALK over the signal NOW (from your usual spot). <<<\n")
    else:
        print("\n>>> SILENT — stay quiet; any detection here is a FALSE fire. <<<\n")

    stream = sd.Stream(samplerate=SAMPLE_RATE, blocksize=FRAME, dtype="int16",
                       channels=1, device=(device, None), callback=callback,
                       latency="low")
    with stream:
        lat = stream.latency
        print(f"  stream latency (in,out): {lat}  → delay {DELAY_MS} ms "
              f"= {delay_frames} frames", file=sys.stderr)
        while stream.active and pos["i"] < n:
            time.sleep(0.1)

    print(f"\n=== {mode.upper()}: {len(detections)} detection(s) "
          f"(duplex, delay {DELAY_MS} ms, threshold {THRESHOLD:.0f}) ===")
    if mode == "silent":
        print("  CLEAN — no false fires." if not detections else
              "  want 0. Sweep JARVIS_BARGE_DELAY_MS (start near the printed stream "
              "latency); the right delay drives the cleaned-RMS peak (DEBUG=1) lowest.")
    else:
        print("  GOOD — heard you over the playback." if detections else
              "  want >=1. Lower threshold / talk up / fix the delay.")


if __name__ == "__main__":
    main()
