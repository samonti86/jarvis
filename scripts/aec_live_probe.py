"""M88 Phase 2 de-risk, step 2 — REAL-ROOM AEC confirm + delay measurement.

The offline spike (aec_spike.py) proved pyaec cancels a SYNTHETIC linear echo
by ~19 dB while preserving the near-end. But a real room adds speaker
nonlinearity, clock drift between the playback and capture devices, and a
play->capture DELAY the real-time integration must fit inside the AEC's filter
window. This probe measures the REAL numbers on this hardware.

It uses sounddevice.playrec — plays a broadband test signal through the speakers
AND records the mic sample-aligned to playback, in one call. You stay SILENT, so
the recording is PURE ECHO of Jarvis's output path. We then:
  - estimate the play->capture delay (cross-correlation),
  - run pyaec (preprocess=False, the barge-in config) on the captured pair,
  - report the REAL-ROOM ERLE (dB of echo pyaec removes) + the delay.

Decision:
  - ERLE >= 12 dB and delay << 256 ms → build hands-free talk-over on pyaec.
  - ERLE 6-12 dB → marginal; the barge VAD will need a conservative threshold.
  - ERLE < 6 dB or delay > ~200 ms → pyaec won't carry it; reconsider (headset).

Run (QUIT the running Jarvis first — tray -> Quit — so it doesn't react to the
test signal or fight for the mic), then STAY SILENT for ~8 seconds:

    venv\\Scripts\\python.exe scripts\\aec_live_probe.py

Set your system volume to the level you'd actually run Jarvis at. Report the
ERLE + delay line back.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.audio import resolve_input_device  # noqa: E402

load_dotenv()

SR = 16_000
FRAME = 256
FILTER_LEN = 4096        # 256 ms — the AEC's delay+reverb budget
DUR = 8.0


def _speechlike(n: int, amp: float) -> np.ndarray:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n)
    t = np.arange(n) / SR
    env = 0.15 + 0.85 * np.abs(np.sin(2 * np.pi * 3.0 * t))
    x *= env
    x /= (np.max(np.abs(x)) + 1e-9)
    return x * amp


def _estimate_delay(mic: np.ndarray, far: np.ndarray, max_lag: int) -> int:
    """Bulk play->capture delay (samples) via FFT cross-correlation, searching
    lags [0, max_lag]. Uses the first 4 s for speed."""
    seg = min(len(mic), len(far), 4 * SR)
    m = mic[:seg] - mic[:seg].mean()
    f = far[:seg] - far[:seg].mean()
    nfft = 1 << int(np.ceil(np.log2(2 * seg)))
    corr = np.fft.irfft(np.fft.rfft(m, nfft) * np.conj(np.fft.rfft(f, nfft)), nfft)
    return int(np.argmax(corr[:max_lag]))


def main() -> None:
    import pyaec  # noqa: PLC0415

    device = resolve_input_device(os.getenv("JARVIS_MIC_DEVICE", "").strip())
    n = int(DUR * SR)
    far = _speechlike(n, amp=0.5 * 32768.0)          # int16-scale, half FS
    far_play = (far / 32768.0).astype(np.float32)    # playrec wants float [-1,1]

    print("\n>>> STAY SILENT for ~8 s while the test signal plays. <<<\n")
    try:
        # latency='low' to shrink the play/capture buffering (the bulk delay we
        # then compensate). Falls back to default if the backend refuses.
        rec = sd.playrec(far_play, samplerate=SR, channels=1,
                         device=(device, None), dtype="float32", latency="low")
        sd.wait()
    except Exception:
        try:
            rec = sd.playrec(far_play, samplerate=SR, channels=1,
                             device=(device, None), dtype="float32")
            sd.wait()
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] playrec failed: {exc}", file=sys.stderr)
            print("  (try a specific output device, or check the mic index)")
            sys.exit(1)

    mic = rec[:, 0].astype(np.float64) * 32768.0     # back to int16 scale

    delay = _estimate_delay(mic, far, max_lag=int(0.50 * SR))
    print(f"  estimated play->capture delay: {delay} samples "
          f"({1000 * delay / SR:.0f} ms)")

    # The 264 ms first-pass delay was almost all SYSTEM BUFFERING, not acoustics
    # — and it exceeded the filter window, so the echo fell outside the filter's
    # reach and nothing cancelled. The REAL integration compensates for this
    # known delay by feeding the far-end reference time-aligned to the mic. We
    # do the same here: drop the first `delay` mic samples so far[i] lines up
    # with the echo it produced, leaving only the residual (acoustic + reverb +
    # jitter) for the adaptive filter — which is what filter_len is actually for.
    mic_a = mic[delay:]
    far_a = far[: len(mic_a)]
    m = min(len(mic_a), len(far_a))
    far_i = np.clip(far_a[:m], -32768, 32767).astype(np.int16)
    mic_i = np.clip(mic_a[:m], -32768, 32767).astype(np.int16)

    aec = pyaec.Aec(FRAME, FILTER_LEN, SR, False)  # preprocess=False (barge config)
    out = np.zeros(m, dtype=np.float64)
    for i in range(0, m - FRAME, FRAME):
        cleaned = aec.cancel_echo(mic_i[i:i + FRAME], far_i[i:i + FRAME])
        out[i:i + FRAME] = np.asarray(cleaned, dtype=np.float64)

    # ERLE over the converged tail (skip ~2 s for adaptation), user-silent so
    # all mic energy is echo.
    seg = slice(int(2.0 * SR), m - FRAME)
    erle = 10 * np.log10(
        (np.mean(mic_a[seg] ** 2) + 1e-9) / (np.mean(out[seg] ** 2) + 1e-9))
    echo_rms = float(np.sqrt(np.mean(mic_a[seg] ** 2)))
    res_rms = float(np.sqrt(np.mean(out[seg] ** 2)))

    print(f"\n  DELAY-ALIGNED real-room ERLE (pure AEC): {erle:5.1f} dB")
    print(f"  echo RMS -> residual RMS:    {echo_rms:.0f} -> {res_rms:.0f} (int16 scale)")

    print("\n=== VERDICT ===")
    if erle >= 12:
        print(f"  STRONG. {erle:.0f} dB after delay alignment — AEC works in this "
              "room. The integration just needs to compensate the ~"
              f"{1000*delay/SR:.0f} ms play->capture delay (measure once, pre-delay"
              " the far-end reference). Build hands-free talk-over on pyaec.")
    elif erle >= 6:
        print(f"  MARGINAL. {erle:.0f} dB after alignment — buildable, but the barge"
              " VAD needs a conservative threshold; expect live tuning. The delay "
              f"(~{1000*delay/SR:.0f} ms) is compensable.")
    else:
        print(f"  WEAK. Only {erle:.0f} dB even after delay alignment — the room "
              "echo is nonlinear / drifting beyond what this AEC models. pyaec "
              "alone won't carry reliable talk-over here. The clean answer is a "
              "headset (kills echo at the source → barge-in becomes trivial); "
              "otherwise it's a heavier-AEC problem not worth the complexity.")


if __name__ == "__main__":
    main()
