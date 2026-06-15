"""M88 Phase 2 — hands-free barge-in via acoustic echo cancellation.

The full-duplex payoff: interrupt Jarvis by just talking — no wake word. The
hard part (measured in the Phase 2 de-risk) is that the omni mic hears Jarvis's
own playback as loud as the user, so energy alone can't tell them apart. The fix:
cancel Jarvis's playback from the mic with `pyaec` (SpeexDSP), then run an
energy-gate VAD on the CLEANED signal. The de-risk found, in this room, ~12 dB
real-room cancellation leaving the user ~3.7x above the residual echo — a fixed
threshold of ~1187 (int16 RMS) separates them (silent: 0 false fires; talk: fires
every time).

Pieces:
  - FarEndBuffer  — what Jarvis is playing, wall-clock-stamped + resampled to the
                    mic rate, so the canceller can pull the far-end DELAY-ALIGNED
                    to each mic frame (the alignment is the whole ballgame —
                    feeding an unaligned reference cancels nothing).
  - BargeDetector — pure energy-gate VAD on the cleaned signal (windowed RMS +
                    sustain). Unit-tested.
  - AecBargeMonitor — the thread: read mic → pull aligned far-end → pyaec cancel
                    → detect → set interrupt_event (the SAME M52 contract; it
                    NEVER touches the audio API — speak_streaming does the cut on
                    the stream-owning thread, the WASAPI thread-affinity rule).

Drivable two ways: standalone (scripts/aec_barge_live.py, for live tuning) and
in-Jarvis (TurnRunner wires session.read + a far-end sink on speak_streaming).
Fully fail-soft — if pyaec is missing or anything raises, the monitor disables
itself and the turn completes normally (worst case: no barge that turn).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

SAMPLE_RATE = 16_000
FRAME = 256                  # pyaec frame (16 ms) — must match Aec(frame_size)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


# Tuned live on the real mic+loudspeaker (2026-06-15) via scripts/aec_barge_probe.py:
# residual echo p99 ~615, user voice median ~2292 → geo-mean threshold ~1187.
# All env-overridable so the live verification pass can tune without a code edit.
FILTER_LEN = _env_int("JARVIS_BARGE_FILTER_LEN", 4096)     # 256 ms echo tail
DELAY_MS = _env_int("JARVIS_BARGE_DELAY_MS", 160)          # play->capture delay
THRESHOLD = _env_float("JARVIS_BARGE_THRESHOLD", 1187.0)   # cleaned-RMS gate (int16)
WIN_FRAMES = _env_int("JARVIS_BARGE_WIN_FRAMES", 8)        # ~128 ms detection window
SUSTAIN = _env_int("JARVIS_BARGE_SUSTAIN", 2)              # consecutive windows
DEBUG = os.getenv("JARVIS_BARGE_DEBUG", "").strip() not in ("", "0", "false", "False")


def is_enabled() -> bool:
    """Hands-free barge-in is opt-in (default off) until live-verified — the
    M52 wake-word barge-in is the safe default. Enable with JARVIS_HANDS_FREE_BARGE=1."""
    return os.getenv("JARVIS_HANDS_FREE_BARGE", "").strip() not in ("", "0", "false", "False")


class BargeDetector:
    """Pure energy-gate VAD on the cleaned (echo-cancelled) signal. Fed one
    per-FRAME RMS at a time; fires once when WIN_FRAMES-window RMS exceeds
    `threshold` for `sustain` consecutive (non-overlapping) windows — then
    re-arms only after a window drops back below. Matches the offline tuning
    harness's window/sustain semantics. No audio, no threads — unit-testable."""

    def __init__(self, threshold: float = THRESHOLD, win_frames: int = WIN_FRAMES,
                 sustain: int = SUSTAIN) -> None:
        self._threshold = threshold
        self._win = max(1, win_frames)
        self._sustain = max(1, sustain)
        self._buf: list[float] = []
        self._consec = 0
        self._armed = True

    def push_frame(self, frame_rms: float) -> bool:
        """Add one frame's RMS; return True exactly on the window that fires."""
        self._buf.append(frame_rms)
        if len(self._buf) < self._win:
            return False
        # Window RMS from the per-frame RMS values, then reset (non-overlapping).
        wrms = float(np.sqrt(np.mean(np.square(self._buf))))
        self._buf.clear()
        if wrms > self._threshold:
            self._consec += 1
            if self._consec >= self._sustain and self._armed:
                self._armed = False
                return True
        else:
            self._consec = 0
            self._armed = True
        return False

    def reset(self) -> None:
        self._buf.clear()
        self._consec = 0
        self._armed = True


def _resample_to_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    """Mono int16 @ `sr` → mono int16 @ 16k via linear interpolation. Good
    enough for an AEC reference (the adaptive filter models whatever it's fed;
    minor resample artifacts are immaterial). Stereo → channel mean."""
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float64)
    if sr == SAMPLE_RATE or samples.size == 0:
        return samples.astype(np.int16)
    n_out = int(round(samples.size * SAMPLE_RATE / sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.int16)
    xp = np.arange(samples.size)
    xq = np.linspace(0, samples.size - 1, n_out)
    return np.interp(xq, xp, samples).astype(np.int16)


class FarEndBuffer:
    """Thread-safe record of what Jarvis is playing, resampled to 16k and
    wall-clock-stamped, so the canceller can pull the far-end frame whose echo
    is arriving at a given mic-capture time. Old segments are pruned."""

    def __init__(self, retain_sec: float = 3.0) -> None:
        self._segments: deque[tuple[float, np.ndarray]] = deque()
        self._retain = retain_sec
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray, sr: int, start_time: float) -> None:
        """Record a played segment: `samples` begins playing at `start_time`
        (monotonic). Called by speak_streaming right before sd.play."""
        far = _resample_to_16k(np.asarray(samples), sr)
        if far.size == 0:
            return
        with self._lock:
            self._segments.append((start_time, far))
            cutoff = start_time - self._retain
            while self._segments and (
                    self._segments[0][0] + len(self._segments[0][1]) / SAMPLE_RATE) < cutoff:
                self._segments.popleft()

    def clear(self) -> None:
        with self._lock:
            self._segments.clear()

    def frame_at(self, t_capture: float, n: int, delay_s: float) -> np.ndarray:
        """The far-end frame (length `n`, int16 @16k) whose echo is arriving at
        `t_capture`. The echo heard now was played `delay_s` ago, so we read the
        far timeline at t_emit = t_capture - delay_s. Gaps between sentences →
        silence (no echo expected there). Vectorized per segment."""
        t_emit = t_capture - delay_s
        out = np.zeros(n, dtype=np.int16)
        with self._lock:
            segs = list(self._segments)
        for t0, arr in segs:
            seg_end = t0 + len(arr) / SAMPLE_RATE
            jlo = max(0, int(np.ceil((t0 - t_emit) * SAMPLE_RATE)))
            jhi = min(n, int((seg_end - t_emit) * SAMPLE_RATE))
            if jhi <= jlo:
                continue
            base = round((t_emit - t0) * SAMPLE_RATE)
            aidx = base + np.arange(jlo, jhi)
            valid = (aidx >= 0) & (aidx < len(arr))
            j = np.arange(jlo, jhi)[valid]
            out[j] = arr[aidx[valid]]
        return out


class AecBargeMonitor:
    """Hands-free barge-in monitor. Runs on its own thread for a TTS reply:
    reads mic chunks, cancels Jarvis's echo (pyaec + the delay-aligned far-end),
    and on sustained user speech in the CLEANED signal sets `interrupt_event` —
    and nothing else (speak_streaming performs the sd.stop cut on the stream's
    owning thread; the M52/WASAPI thread-affinity contract).

    `mic_read()` returns the next int16 mono 16k chunk (blocking) — in Jarvis,
    session.read; standalone, an InputStream queue getter. Fail-soft: if pyaec
    can't load or anything raises, it logs and returns (the turn completes; no
    barge that round)."""

    def __init__(self, mic_read: Callable[[], np.ndarray],
                 interrupt_event: threading.Event, stop_event: threading.Event,
                 far_buffer: FarEndBuffer, *, oneshot: bool = True,
                 on_detect: "Callable[[float], None] | None" = None) -> None:
        self._mic_read = mic_read
        self._interrupt = interrupt_event
        self._stop = stop_event
        self._far = far_buffer
        self._detector = BargeDetector()
        self._delay_s = DELAY_MS / 1000.0
        # oneshot=True (Jarvis): fire interrupt + stop on the first barge.
        # oneshot=False (live validator): report each detection and keep
        # counting — exercises the SAME loop, just doesn't stop.
        self._oneshot = oneshot
        self._on_detect = on_detect

    def run(self) -> None:
        try:
            import pyaec  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            print(f"[barge-aec] pyaec unavailable ({exc}); hands-free barge "
                  "disabled this turn", file=sys.stderr)
            return
        try:
            aec = pyaec.Aec(FRAME, FILTER_LEN, SAMPLE_RATE, False)  # preprocess=False
        except Exception as exc:  # noqa: BLE001
            print(f"[barge-aec] AEC init failed ({exc}); disabled", file=sys.stderr)
            return

        print("[barge-aec] hands-free barge-in monitoring during playback",
              file=sys.stderr)
        residual = np.zeros(0, dtype=np.int16)
        dbg_last = time.monotonic()
        dbg_peak = 0.0
        try:
            while not self._stop.is_set():
                chunk = self._mic_read()
                if self._stop.is_set():
                    return
                if chunk is None or len(chunk) == 0:
                    continue
                t_read = time.monotonic()
                buf = np.concatenate([residual, np.asarray(chunk, dtype=np.int16)])
                nframes = len(buf) // FRAME
                for k in range(nframes):
                    mic_f = buf[k * FRAME:(k + 1) * FRAME]
                    # This frame's audio ended ~ t_read - (tail samples)/SR.
                    tail = len(buf) - (k + 1) * FRAME
                    t_frame = t_read - tail / SAMPLE_RATE
                    far_f = self._far.frame_at(t_frame, FRAME, self._delay_s)
                    cleaned = np.asarray(aec.cancel_echo(mic_f, far_f),
                                         dtype=np.float64)
                    rms = float(np.sqrt(np.mean(np.square(cleaned))))
                    if DEBUG:
                        dbg_peak = max(dbg_peak, rms)
                    if self._detector.push_frame(rms):
                        if self._on_detect is not None:
                            self._on_detect(t_frame)
                        if self._oneshot:
                            print(f"[barge-aec] BARGE-IN detected (cleaned RMS "
                                  f"{rms:.0f} > {THRESHOLD:.0f})", file=sys.stderr)
                            self._interrupt.set()
                            return
                        self._detector.reset()  # validator: keep counting
                residual = buf[nframes * FRAME:]
                if DEBUG and (t_read - dbg_last) > 1.0:
                    print(f"[barge-aec] cleaned-RMS peak last 1s: {dbg_peak:.0f} "
                          f"(thresh {THRESHOLD:.0f}, delay {DELAY_MS}ms)",
                          file=sys.stderr)
                    dbg_last, dbg_peak = t_read, 0.0
        except Exception as exc:  # noqa: BLE001 — never crash the turn
            print(f"[barge-aec] monitor error ({exc}); disabled this turn",
                  file=sys.stderr)
