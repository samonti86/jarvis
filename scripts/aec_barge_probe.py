"""M88 Phase 2 live tuning — the AEC barge-DETECTOR harness.

aec_live_probe proved pyaec cancels ~12 dB in this room (after delay alignment).
THIS harness tunes the actual barge-in DETECTOR before any of it touches Jarvis:
it captures mic + the known far-end (single clock via sd.playrec), delay-aligns,
runs pyaec (preprocess=False — preserves the user), then runs an energy-gate VAD
on the CLEANED signal and reports where it would fire.

The room is hard (mic near a loud loudspeaker): the echo stays loud everywhere while
the user's voice drops across the room. So the whole game is a threshold that
sits ABOVE the residual echo yet BELOW the user. Two runs find it:

  silent  — Jarvis talks, you stay quiet.  → residual-echo ceiling
            (the detector must fire ZERO times here — any fire is a false barge).
  talk    — talk over Jarvis from where you'd actually be (across the room is the
            hard case).                     → user-voice floor on the cleaned signal.

`compare` then prints the residual-vs-user separation and a recommended threshold,
and re-scores both runs at it (silent must stay 0 detections, talk must fire).

Run (QUIT Jarvis first; system volume = your normal Jarvis level):
    venv\\Scripts\\python.exe scripts\\aec_barge_probe.py silent
    venv\\Scripts\\python.exe scripts\\aec_barge_probe.py talk
    venv\\Scripts\\python.exe scripts\\aec_barge_probe.py compare
"""

from __future__ import annotations

import json
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
FRAME = 256                 # pyaec frame (16 ms)
FILTER_LEN = 4096           # 256 ms echo tail
DUR = 12.0                  # long enough to talk over comfortably
# Detector: windowed RMS over ~128 ms, must exceed threshold for `SUSTAIN`
# consecutive windows to count as a barge (rejects transient echo spikes).
WIN = 8                     # frames per detection window (~128 ms)
SUSTAIN = 2                 # consecutive windows over threshold → fire
DEFAULT_THRESH = 1200.0     # int16 RMS; tuned by `compare`


def _speechlike(n: int, amp: float) -> np.ndarray:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n)
    t = np.arange(n) / SR
    env = 0.15 + 0.85 * np.abs(np.sin(2 * np.pi * 3.0 * t))
    x = x * env
    x /= (np.max(np.abs(x)) + 1e-9)
    return x * amp


def _estimate_delay(mic: np.ndarray, far: np.ndarray, max_lag: int) -> int:
    seg = min(len(mic), len(far), 4 * SR)
    m = mic[:seg] - mic[:seg].mean()
    f = far[:seg] - far[:seg].mean()
    nfft = 1 << int(np.ceil(np.log2(2 * seg)))
    corr = np.fft.irfft(np.fft.rfft(m, nfft) * np.conj(np.fft.rfft(f, nfft)), nfft)
    return int(np.argmax(corr[:max_lag]))


def _clean(mic: np.ndarray, far: np.ndarray) -> tuple[np.ndarray, int]:
    """Delay-align + pyaec(preprocess=False). Returns (cleaned int16-scale, delay)."""
    import pyaec  # noqa: PLC0415
    delay = _estimate_delay(mic, far, max_lag=int(0.50 * SR))
    mic_a = mic[delay:]
    far_a = far[: len(mic_a)]
    m = min(len(mic_a), len(far_a))
    far_i = np.clip(far_a[:m], -32768, 32767).astype(np.int16)
    mic_i = np.clip(mic_a[:m], -32768, 32767).astype(np.int16)
    aec = pyaec.Aec(FRAME, FILTER_LEN, SR, False)
    out = np.zeros(m, dtype=np.float64)
    for i in range(0, m - FRAME, FRAME):
        cleaned = aec.cancel_echo(mic_i[i:i + FRAME], far_i[i:i + FRAME])
        out[i:i + FRAME] = np.asarray(cleaned, dtype=np.float64)
    return out, delay


def _window_rms(cleaned: np.ndarray) -> np.ndarray:
    """RMS per ~128 ms window, skipping the first 2 s (AEC convergence)."""
    start = int(2.0 * SR)
    wlen = WIN * FRAME
    vals = []
    for i in range(start, len(cleaned) - wlen, wlen):
        vals.append(float(np.sqrt(np.mean(cleaned[i:i + wlen] ** 2))))
    return np.array(vals)


def _detections(wrms: np.ndarray, thresh: float) -> int:
    """Count barge events: SUSTAIN consecutive windows over threshold (each event
    counted once, then must drop below before re-arming)."""
    events, run, armed = 0, 0, True
    for v in wrms:
        if v > thresh:
            run += 1
            if run >= SUSTAIN and armed:
                events += 1
                armed = False
        else:
            run = 0
            armed = True
    return events


def _out_path(mode: str) -> Path:
    base = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"barge_{mode}.json"


def _record(mode: str) -> None:
    device = resolve_input_device(os.getenv("JARVIS_MIC_DEVICE", "").strip())
    n = int(DUR * SR)
    far = _speechlike(n, amp=0.5 * 32768.0)
    far_play = (far / 32768.0).astype(np.float32)

    if mode == "talk":
        print("\n>>> TALK over the test signal NOW, from where you'd actually be. <<<\n")
    else:
        print("\n>>> SILENT — let it play, stay quiet. <<<\n")
    try:
        rec = sd.playrec(far_play, samplerate=SR, channels=1,
                         device=(device, None), dtype="float32", latency="low")
        sd.wait()
    except Exception:
        rec = sd.playrec(far_play, samplerate=SR, channels=1,
                         device=(device, None), dtype="float32")
        sd.wait()
    mic = rec[:, 0].astype(np.float64) * 32768.0

    cleaned, delay = _clean(mic, far)
    wrms = _window_rms(cleaned)
    stats = {
        "mode": mode, "delay_ms": round(1000 * delay / SR),
        "median": float(np.median(wrms)), "p90": float(np.percentile(wrms, 90)),
        "p95": float(np.percentile(wrms, 95)), "p99": float(np.percentile(wrms, 99)),
        "max": float(wrms.max()), "windows": int(wrms.size),
        "det_default": _detections(wrms, DEFAULT_THRESH),
        # Persist the raw per-window RMS so `compare` can re-score detections
        # EXACTLY at the recommended threshold (small array, ~90 floats).
        "wrms": [round(v, 1) for v in wrms.tolist()],
    }
    _out_path(mode).write_text(json.dumps(stats, indent=2))
    print(f"--- {mode.upper()}  (delay {stats['delay_ms']} ms, {stats['windows']} windows) ---")
    for k in ("median", "p90", "p95", "p99", "max"):
        print(f"  cleaned RMS {k:>6}: {stats[k]:8.0f}")
    print(f"  detections @ default thresh {DEFAULT_THRESH:.0f}: {stats['det_default']}")
    print(f"\nsaved -> {_out_path(mode)}")


def _compare() -> None:
    sp, tp = _out_path("silent"), _out_path("talk")
    if not sp.exists() or not tp.exists():
        print("[probe] need both runs first: 'silent' then 'talk'.", file=sys.stderr)
        sys.exit(1)
    s, t = json.loads(sp.read_text()), json.loads(tp.read_text())
    print(f"  SILENT cleaned RMS:  p95 {s['p95']:.0f}  p99 {s['p99']:.0f}  max {s['max']:.0f}")
    print(f"  TALK   cleaned RMS:  median {t['median']:.0f}  p90 {t['p90']:.0f}")

    # Threshold must clear the residual-echo ceiling (silent p99) and sit below
    # the user's typical level (talk median). Pick the geometric midpoint.
    echo_ceiling = s["p99"]
    user_typical = t["median"]
    print("\n=== VERDICT ===")
    if user_typical <= echo_ceiling:
        print(f"  NOT SEPARABLE: user typical ({user_typical:.0f}) <= residual-echo "
              f"ceiling ({echo_ceiling:.0f}). The AEC residual is as loud as your "
              "voice from there. Try talking closer, more loudspeaker/mic distance, "
              "or accept a headset. (A bigger filter_len / a double-talk freeze "
              "might recover some margin — worth one more iteration.)")
        return
    thresh = float(np.sqrt(echo_ceiling * user_typical))
    # Exact re-score at the recommended threshold from the persisted windows.
    sil_fp = _detections(np.array(s.get("wrms", [])), thresh)
    talk_fire = _detections(np.array(t.get("wrms", [])), thresh)
    print(f"  separable: user {user_typical:.0f} vs echo ceiling {echo_ceiling:.0f}")
    print(f"  recommended threshold: {thresh:.0f}  (geo-mean)")
    print(f"  re-scored at {thresh:.0f}: SILENT detections {sil_fp} (want 0), "
          f"TALK detections {talk_fire} (want >=1)")
    if sil_fp == 0 and talk_fire >= 1:
        print(f"\n  -> GOOD. Build AecBargeMonitor with threshold={thresh:.0f}, "
              f"sustain={SUSTAIN} windows (~{1000*SUSTAIN*WIN*FRAME/SR:.0f} ms), "
              f"filter_len={FILTER_LEN}, delay~{t['delay_ms']} ms.")
    else:
        print("\n  -> needs another iteration (adjust threshold / sustain, or "
              "re-run talk from the intended distance).")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode in ("silent", "talk"):
        _record(mode)
    elif mode == "compare":
        _compare()
    else:
        print(__doc__)
        print("\nGive one of: silent | talk | compare")
        sys.exit(2)


if __name__ == "__main__":
    main()
