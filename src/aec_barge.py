"""M88 Phase 2 — hands-free barge-in via acoustic echo cancellation.

The full-duplex payoff: interrupt Jarvis by just talking — no wake word. The
hard part (measured in the Phase 2 de-risk) is that the omni mic hears Jarvis's
own playback as loud as the user, so energy alone can't tell them apart. The fix:
cancel Jarvis's playback from the mic with `pyaec` (SpeexDSP), then run an
energy-gate VAD on the CLEANED signal. The de-risk found, in this room, ~12 dB
real-room cancellation leaving the user ~3.7x above the residual echo — a fixed
threshold of ~1187 (int16 RMS) separates them (silent: 0 false fires; talk: fires
every time).

The alignment lesson (live-proven): the AEC needs SAMPLE-ACCURATE far↔mic
alignment. Aligning two independent streams by Python wall-clock FAILED (jittery
buffer latencies smear the reference → the adaptive filter never converges). The
fix is a SINGLE DUPLEX stream — play + capture in one callback, one clock — so
far and mic are sample-locked with a fixed offset. Validated live: silent 0 /
talk 30.

Pieces:
  - BargeDetector     — pure energy-gate VAD on the cleaned signal (windowed RMS
                        + sustain). Unit-tested.
  - DuplexBargePlayer — plays Jarvis's TTS (resampled to 16k) through ONE duplex
                        sd.Stream whose callback cancels the echo (pyaec, the
                        far-end delayed by the stream's fixed offset) and detects
                        a barge-in → sets interrupt_event. speak_streaming drives
                        its lifecycle and closes the stream (WASAPI thread-
                        affinity: the owner thread does the cut, not the callback).

Fully fail-soft — if pyaec is missing or the duplex stream won't open (e.g. a
second mic client is refused), start() returns False and the caller plays
normally (no barge that turn). Opt-in via JARVIS_HANDS_FREE_BARGE (default off).
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

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
    enough for both playback and an AEC reference on the barge path (slight
    quality dip vs the 24k native, accepted to reuse the validated 16k AEC
    tuning). Stereo → channel mean."""
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


class DuplexBargePlayer:
    """Plays 16k PCM through ONE duplex sd.Stream while cancelling Jarvis's echo
    from the mic (pyaec) and detecting a hands-free barge-in on the cleaned
    signal — all in the stream callback, ONE clock (the sample-accurate
    alignment the AEC needs). On a barge-in it sets interrupt_event; the OWNING
    thread (speak_streaming) closes the stream (the callback only ever stops the
    stream on a NATURAL end, via CallbackStop).

    Lifecycle, driven by speak_streaming's playback consumer (the turn thread):
        p = DuplexBargePlayer(mic_device, interrupt_event, ...)
        if not p.start():       # fail-soft → caller plays normally instead
            <plain sd.play path>
        for (samples, sr) in audio:   p.feed(_resample_to_16k(samples, sr))
        p.done_feeding()
        p.wait(interrupt_event)       # blocks until drained or barge-in
        p.close()

    Fail-soft everywhere: pyaec missing / stream won't open / a second mic client
    refused → start() returns False. TTS is resampled to 16k for this path."""

    def __init__(self, mic_device: "int | None", interrupt_event: threading.Event,
                 *, on_first_audio=None, on_amplitude=None, on_barge=None) -> None:
        self._device = mic_device
        self._interrupt = interrupt_event
        self._on_first_audio = on_first_audio
        self._on_amplitude = on_amplitude
        # Fired (once) the instant a barge-in is DETECTED, so the UI can flip to
        # LISTENING immediately — before the ~0.5-1s stream teardown completes,
        # which otherwise leaves the orb lingering on SPEAKING after the cut.
        self._on_barge = on_barge
        self._barged = False
        # Playback queue (turn thread → callback): int16 16k arrays; None = end.
        self._pcm_q: "queue.Queue" = queue.Queue()
        self._cur = np.zeros(0, dtype=np.int16)
        self._cur_pos = 0
        self._saw_end = False        # the None sentinel was pulled
        self._first = True
        self._done = threading.Event()
        self._stream: "sd.Stream | None" = None
        self._aec = None
        self._detector = BargeDetector()
        self._delay_frames = max(0, round(DELAY_MS / 1000.0 * SAMPLE_RATE / FRAME))
        self._far_hist: "deque[np.ndarray]" = deque(maxlen=self._delay_frames + 1)
        self._tail_silent = 0        # silent frames since end → flush then stop

    # ----- feeding (turn thread) -----
    def feed(self, samples16k: np.ndarray) -> None:
        if samples16k is not None and len(samples16k):
            self._pcm_q.put(np.asarray(samples16k, dtype=np.int16))

    def done_feeding(self) -> None:
        self._pcm_q.put(None)

    # ----- output assembly (pure; testable without a stream) -----
    def _next_out(self, frames: int) -> np.ndarray:
        """Pull `frames` int16 from the playback queue across array boundaries;
        underrun or pre-end-of-feed gaps → trailing silence. Sets _saw_end when
        the None sentinel surfaces."""
        out = np.zeros(frames, dtype=np.int16)
        filled = 0
        while filled < frames:
            if self._cur_pos >= len(self._cur):
                try:
                    item = self._pcm_q.get_nowait()
                except queue.Empty:
                    break  # underrun → rest is silence
                if item is None:
                    self._saw_end = True
                    break
                self._cur = item
                self._cur_pos = 0
                if len(self._cur) == 0:
                    continue
            take = min(frames - filled, len(self._cur) - self._cur_pos)
            out[filled:filled + take] = self._cur[self._cur_pos:self._cur_pos + take]
            self._cur_pos += take
            filled += take
        return out

    def _has_pending(self) -> bool:
        return (self._cur_pos < len(self._cur)) or (not self._pcm_q.empty())

    # ----- the duplex callback (PortAudio thread) -----
    def _callback(self, indata, outdata, frames, _time, status) -> None:  # noqa: ANN001
        if status:
            print(f"[barge-aec] stream status: {status}", file=sys.stderr)
        out = self._next_out(frames)
        outdata[:, 0] = out
        if self._first and out.any():
            self._first = False
            if self._on_first_audio is not None:
                try:
                    self._on_first_audio()
                except Exception:  # noqa: BLE001 — UI callback must not break audio
                    pass
        # Far reference delayed by the stream's fixed in+out offset (jitter-free
        # in a single duplex stream — the whole reason this works).
        self._far_hist.append(out.copy())
        if len(self._far_hist) > self._delay_frames and len(self._far_hist[0]) == frames:
            far_ref = self._far_hist[0]
        else:
            far_ref = np.zeros(frames, dtype=np.int16)
        mic = indata[:, 0].astype(np.int16)
        try:
            cleaned = np.asarray(self._aec.cancel_echo(mic, far_ref), dtype=np.float64)
        except Exception:  # noqa: BLE001
            cleaned = mic.astype(np.float64)
        rms = float(np.sqrt(np.mean(np.square(cleaned)))) if cleaned.size else 0.0
        if self._detector.push_frame(rms):
            print(f"[barge-aec] BARGE-IN — user spoke over Jarvis (cleaned RMS "
                  f"{rms:.0f} > {THRESHOLD:.0f})", file=sys.stderr)
            self._interrupt.set()
            if not self._barged:           # flip the UI to LISTENING at once
                self._barged = True
                if self._on_barge is not None:
                    try:
                        self._on_barge()
                    except Exception:  # noqa: BLE001 — UI callback must not break audio
                        pass
        if self._on_amplitude is not None:
            amp = min(1.0, float(np.sqrt(np.mean(np.square(out.astype(np.float64))))) / 8000.0)
            try:
                self._on_amplitude(amp)
            except Exception:  # noqa: BLE001
                pass
        # Natural end: end sentinel pulled, nothing pending, output gone silent
        # and the far history has flushed → stop the stream on its own thread.
        if self._saw_end and not self._has_pending():
            if not out.any():
                self._tail_silent += 1
                if self._tail_silent >= self._delay_frames + 2:
                    self._done.set()
                    raise sd.CallbackStop
            else:
                self._tail_silent = 0

    # ----- lifecycle (turn thread) -----
    def start(self) -> bool:
        """Open the duplex stream + AEC. Returns False (fail-soft) if pyaec is
        unavailable or the stream won't open — the caller then plays normally."""
        try:
            import pyaec  # noqa: PLC0415
            self._aec = pyaec.Aec(FRAME, FILTER_LEN, SAMPLE_RATE, False)  # preprocess=False
        except Exception as exc:  # noqa: BLE001
            print(f"[barge-aec] pyaec unavailable ({exc}); plain playback",
                  file=sys.stderr)
            return False
        try:
            self._stream = sd.Stream(
                samplerate=SAMPLE_RATE, blocksize=FRAME, dtype="int16",
                channels=1, device=(self._device, None),
                callback=self._callback, latency="low",
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            print(f"[barge-aec] duplex stream open failed ({exc}); plain playback",
                  file=sys.stderr)
            self._stream = None
            return False
        print("[barge-aec] hands-free barge-in active (duplex AEC, "
              f"delay {DELAY_MS}ms, thresh {THRESHOLD:.0f})", file=sys.stderr)
        return True

    def wait(self, interrupt_event: "threading.Event | None",
             poll: float = 0.05, max_sec: float = 600.0) -> None:
        """Block (turn thread) until playback drains, a barge-in fires, or the
        stream goes inactive. A hard cap guards against a wedged stream."""
        deadline = time.monotonic() + max_sec
        while time.monotonic() < deadline:
            if self._done.wait(poll):
                return
            if interrupt_event is not None and interrupt_event.is_set():
                return
            if self._stream is not None and not self._stream.active:
                return

    def close(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
            except Exception:  # noqa: BLE001
                pass
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass
