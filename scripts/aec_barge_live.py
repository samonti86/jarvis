"""M88 Phase 2 — REAL-TIME validation of the hands-free barge monitor.

The offline harness (aec_barge_probe.py) tuned the threshold with PERFECT
alignment (sd.playrec). This validates the in-process path: it plays a clip in
segments via sd.play (pushing the far-end to FarEndBuffer right before each play
— EXACTLY how speak_streaming will), captures the mic on its own InputStream,
and runs the REAL AecBargeMonitor (oneshot=False so it counts every detection)
live. If the wall-clock alignment + the 160 ms delay work here, they work in
Jarvis — so I wire it in only after this passes.

  silent — let it play, STAY QUIET.  Expect ~0 detections (no false interrupts).
  talk   — talk over it from where you'd actually be.  Expect detections to fire.

Run (QUIT Jarvis first; normal volume). Tune live without editing code via
JARVIS_BARGE_DELAY_MS / JARVIS_BARGE_THRESHOLD (and JARVIS_BARGE_DEBUG=1 to watch
the cleaned-RMS peak each second):

    venv\\Scripts\\python.exe scripts\\aec_barge_live.py silent
    venv\\Scripts\\python.exe scripts\\aec_barge_live.py talk
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
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
    DELAY_MS, SAMPLE_RATE, THRESHOLD, AecBargeMonitor, FarEndBuffer,
)
from src.audio import resolve_input_device  # noqa: E402

load_dotenv()

SEG_SEC = 2.5     # play in ~2.5 s sentences ...
GAP_SEC = 0.4     # ... with gaps, like real multi-sentence TTS
SEGMENTS = 5      # ~5 * (2.5+0.4) ≈ 14 s total


def _speechlike(n: int, amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    t = np.arange(n) / SAMPLE_RATE
    x = x * (0.15 + 0.85 * np.abs(np.sin(2 * np.pi * 3.0 * t)))
    x /= (np.max(np.abs(x)) + 1e-9)
    return (x * amp).astype(np.int16)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("silent", "talk"):
        print(__doc__)
        print("\nGive one of: silent | talk")
        sys.exit(2)

    device = resolve_input_device(os.getenv("JARVIS_MIC_DEVICE", "").strip())
    far = FarEndBuffer()
    mic_q: queue.Queue[np.ndarray] = queue.Queue()
    stop = threading.Event()
    detections: list[float] = []
    t_start = time.monotonic()

    def on_audio(indata, _frames, _t, status):  # noqa: ANN001
        if status:
            print(f"[live] mic status: {status}", file=sys.stderr)
        mic_q.put(indata[:, 0].copy())

    def on_detect(t_frame: float) -> None:
        detections.append(t_frame)
        print(f"  >>> BARGE detected @ {t_frame - t_start:5.1f}s", file=sys.stderr)

    monitor = AecBargeMonitor(mic_q.get, threading.Event(), stop, far,
                              oneshot=False, on_detect=on_detect)

    seg_len = int(SEG_SEC * SAMPLE_RATE)

    def play() -> None:
        for i in range(SEGMENTS):
            if stop.is_set():
                return
            seg = _speechlike(seg_len, amp=0.5 * 32768.0, seed=i)
            far.push(seg, SAMPLE_RATE, time.monotonic())   # anchor at sd.play time
            sd.play(seg, samplerate=SAMPLE_RATE, blocking=True)
            time.sleep(GAP_SEC)
        stop.set()

    if mode == "talk":
        print("\n>>> TALK over the signal NOW (from your usual spot). <<<\n")
    else:
        print("\n>>> SILENT — stay quiet; any detection here is a FALSE fire. <<<\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=1280, device=device, callback=on_audio):
        mon_t = threading.Thread(target=monitor.run, daemon=True)
        play_t = threading.Thread(target=play, daemon=True)
        mon_t.start()
        play_t.start()
        play_t.join()
        time.sleep(0.3)
        stop.set()
        mic_q.put(np.zeros(0, dtype=np.int16))   # unblock the monitor's get()
        mon_t.join(timeout=2.0)

    print(f"\n=== {mode.upper()} result: {len(detections)} detection(s) "
          f"(delay {DELAY_MS} ms, threshold {THRESHOLD:.0f}) ===")
    if mode == "silent":
        print("  want 0. >0 = the AEC isn't aligning (tune JARVIS_BARGE_DELAY_MS) "
              "or the threshold is too low (raise JARVIS_BARGE_THRESHOLD)."
              if detections else "  CLEAN — no false fires.")
    else:
        print("  want >=1. 0 = not detecting you (lower threshold, talk louder, "
              "or fix the delay)." if not detections
              else "  GOOD — it heard you over the playback in real time.")


if __name__ == "__main__":
    main()
