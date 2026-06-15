"""M88 Phase 2 de-risk — offline AEC library validation (pyaec / SpeexDSP).

The echo probe (duplex_echo_probe.py) proved energy alone can't separate the
user from Jarvis's playback → Phase 2 needs acoustic echo cancellation. WebRTC's
APM sdist won't build on this Windows/py3.12 box, but `pyaec` ships a prebuilt
native DLL (SpeexDSP echo canceller). THIS spike answers: does it actually
cancel, and does it preserve the user's voice during double-talk?

Fully offline + synthetic (no mic, no human) — it simulates:
  - a far-end "Jarvis voice" (speech-like AM noise),
  - a room echo path (decaying multi-tap FIR + bulk delay) applied to it,
  - a near-end "user voice" present only in the second half (double-talk),
  - mic = echo + near-end, fed frame-by-frame to pyaec with the far-end ref.

Metrics:
  - ERLE (far-only first half): how many dB of Jarvis's echo are removed.
    >20 dB excellent · 12–20 good · 6–12 marginal · <6 poor.
  - Double-talk: is the user's voice preserved in the output (correlation),
    and how much echo leaks through while they talk.

Run:  venv\\Scripts\\python.exe scripts\\aec_spike.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Windows consoles are cp1252 and choke on non-ASCII; force UTF-8 so any stray
# arrow/symbol can't crash the spike.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyaec  # noqa: E402

SR = 16_000
FRAME = 256
FILTER_LEN = 4096  # 256 ms echo tail — covers a small reverberant room @16k
DUR = 8.0


def _speechlike(n: int, seed: int, amp: float) -> np.ndarray:
    """Broadband noise shaped by a slow syllable-rate envelope — a crude but
    fair stand-in for speech for an AEC test (broadband, amplitude-modulated)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    # 3 Hz syllable envelope, half-wave-ish, never fully zero.
    t = np.arange(n) / SR
    env = 0.15 + 0.85 * np.abs(np.sin(2 * np.pi * 3.0 * t + seed))
    x *= env
    x /= (np.max(np.abs(x)) + 1e-9)
    return x * amp


def _echo_path(far: np.ndarray) -> np.ndarray:
    """Apply a synthetic room impulse response: ~6 ms bulk delay + a decaying
    sparse multi-tap reflection train (~120 ms), then attenuate. Models the
    omni-mic-hears-the-speakers path."""
    rng = np.random.default_rng(7)
    h = np.zeros(int(0.12 * SR))
    h[int(0.006 * SR)] = 1.0                      # direct bleed
    for _ in range(40):                            # sparse reflections
        idx = rng.integers(int(0.008 * SR), len(h))
        h[idx] += rng.standard_normal() * np.exp(-idx / (0.04 * SR))
    echo = np.convolve(far, h)[: len(far)]
    echo /= (np.max(np.abs(echo)) + 1e-9)
    return echo


def main() -> None:
    n = int(DUR * SR)
    half = n // 2

    # Far-end (Jarvis) at a healthy level; echo attenuated to ~0.45 of it.
    far = _speechlike(n, seed=1, amp=9000.0)
    echo = _echo_path(far) * (0.45 * 9000.0)

    # Near-end (user) ONLY in the second half — and deliberately QUIETER than
    # the echo (the measured hard case: user median 0.089 < echo p95 0.12).
    near = np.zeros(n)
    near[half:] = _speechlike(half, seed=2, amp=5500.0)

    noise = np.random.default_rng(3).standard_normal(n) * 30.0
    mic = echo + near + noise

    far_i = far.astype(np.int16)
    mic_i = np.clip(mic, -32768, 32767).astype(np.int16)
    conv = int(1.0 * SR)               # skip ~1s for the filter to converge
    far_only = slice(conv, half)       # echo present, NO near-end
    dt = slice(half, n - FRAME)        # double-talk: echo + user

    def run(enable_preprocess: bool) -> tuple[float, float]:
        aec = pyaec.Aec(FRAME, FILTER_LEN, SR, enable_preprocess)
        out = np.zeros(n, dtype=np.float64)
        for i in range(0, n - FRAME, FRAME):
            cleaned = aec.cancel_echo(mic_i[i:i + FRAME], far_i[i:i + FRAME])
            out[i:i + FRAME] = np.asarray(cleaned, dtype=np.float64)
        # ERLE: echo energy removed during the far-only stretch.
        erle = 10 * np.log10(
            (np.mean(mic[far_only] ** 2) + 1e-9) / (np.mean(out[far_only] ** 2) + 1e-9))
        # Near-end preservation during double-talk: correlation of the cleaned
        # output with the clean user signal (1.0 = perfectly preserved).
        corr = float(np.corrcoef(out[dt], near[dt])[0, 1])
        return erle, corr

    # preprocess=True  → adds Speex residual-echo + noise suppression (max echo
    #                    kill, but NS distorts/eats the near-end during DT).
    # preprocess=False → pure adaptive-filter cancellation (preserves the
    #                    near-end by construction — this is the BARGE-IN config:
    #                    suppress echo enough that a VAD fires on the user).
    erle_pp, corr_pp = run(True)
    erle_lin, corr_lin = run(False)

    print("=== pyaec (SpeexDSP) offline AEC validation ===")
    print(f"  frame={FRAME}  filter_len={FILTER_LEN} ({1000*FILTER_LEN/SR:.0f} ms)  sr={SR}")
    print(f"\n  preprocess=TRUE  (full NS+RES):  ERLE {erle_pp:5.1f} dB   "
          f"corr(out,user) {corr_pp:5.2f}")
    print(f"  preprocess=FALSE (pure AEC):     ERLE {erle_lin:5.1f} dB   "
          f"corr(out,user) {corr_lin:5.2f}   <- barge-in config")

    print("\n=== VERDICT ===")
    if erle_lin >= 12 and corr_lin >= 0.6:
        print(f"  VIABLE. Pure AEC removes {erle_lin:.0f} dB of Jarvis's echo "
              f"while preserving the user (corr {corr_lin:.2f}). A VAD on the "
              "cleaned signal can detect a barge-in. Build Phase 2 on pyaec "
              "(preprocess=False for the barge path).")
    elif erle_lin >= 6 or erle_pp >= 12:
        print(f"  MARGINAL. Pure-AEC {erle_lin:.0f} dB / full {erle_pp:.0f} dB. "
              "Some real suppression, but tight enough that a live mic test is "
              "the true gate (tune filter_len; consider a louder push-to-detect).")
    else:
        print(f"  WEAK. Only {erle_lin:.0f} dB pure-AEC suppression - not enough "
              "to separate a barge-in. Reconsider a headset (kills echo at the "
              "source) or a heavier AEC than this Windows-installable option.")
    print("\n  NOTE: synthetic echo is a LINEAR, stationary path; real rooms add "
          "speaker nonlinearity, drift, and clock skew that are harder. A good "
          "number here is necessary-but-not-sufficient - a live mic test is the "
          "real gate before shipping. Also: the synthetic 'user' is shaped noise,"
          " which Speex's NS (preprocess=True) suppresses harder than real "
          "harmonic speech, so corr is pessimistic there; the FALSE row is the "
          "fair preservation read.")


if __name__ == "__main__":
    main()
