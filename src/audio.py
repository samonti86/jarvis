"""Microphone session: a single sounddevice stream shared by wake_word + STT.

A long-lived InputStream reduces wake-word→STT latency to zero gap (vs.
opening/closing a new stream per stage, which loses 100-300ms of audio).
"""

from __future__ import annotations

import queue
import sys
import time
from typing import Optional

import numpy as np
import sounddevice as sd

CHUNK_SAMPLES = 1280  # 80 ms @ 16 kHz; matches openWakeWord's window
SAMPLE_RATE = 16_000

# 2026-08-02: rate limit for the callback's status print. See _on_audio — an
# unbounded print inside the callback is self-amplifying, so it is throttled
# to one line per interval with a count, exactly as sound_detector.py does.
_STATUS_LOG_INTERVAL_SEC = 30.0

# A healthy InputStream delivers a chunk every ~80 ms; a long gap means the
# device died under us (USB re-enumeration, KVM switch) WITHOUT raising — the
# callback just stops firing. read() surfaces that as an error instead of
# blocking forever, so the voice-loop supervisor can re-open the session and
# shutdown/quit never hangs on a dead mic.
READ_STALL_SEC = 10.0


def resolve_input_device(spec: str) -> Optional[int]:
    """Resolve a JARVIS_MIC_DEVICE spec to a sounddevice input-device index.

    - blank            → None (Windows default input; legacy behaviour)
    - an integer string → that index, passed through verbatim
    - any other string  → first INPUT-capable device whose name contains the
                          spec (case-insensitive substring)

    Returns None (default device) and logs a warning if a name spec matches
    nothing — degrade to the default rather than fail to capture at all. The
    resolved index is logged so the user can confirm which mic was bound.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.lstrip("-").isdigit():
        # 2026-07-02 QA: guarded — lstrip("-") strips ALL leading dashes, so a
        # typo like "--1" passed the check but crashed int(). A bad .env value
        # must degrade to the default mic, never fail startup (the same
        # never-crash-on-config rule as _int_env/_float_env).
        try:
            idx = int(spec)
        except ValueError:
            print(f"[audio] bad mic index {spec!r}; using default mic",
                  file=sys.stderr)
            return None
        print(f"[audio] mic pinned to device index {idx}", file=sys.stderr)
        return idx
    try:
        devices = sd.query_devices()
    except Exception as exc:  # noqa: BLE001 — never block startup on enumeration
        print(f"[audio] device enumeration failed ({exc}); using default mic",
              file=sys.stderr)
        return None
    needle = spec.lower()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0 and needle in dev["name"].lower():
            print(f"[audio] mic pinned to '{dev['name']}' (index {idx}) "
                  f"matching {spec!r}", file=sys.stderr)
            return idx
    print(f"[audio] no input device matches {spec!r}; using default mic",
          file=sys.stderr)
    return None


class AudioSession:
    """Continuous mic capture into a thread-safe queue of int16 mono chunks."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_samples: int = CHUNK_SAMPLES,
        device: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.device = device          # sounddevice input index; None = default
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        # Rate-limiter state for the callback's status print (see _on_audio).
        self._status_count = 0
        self._status_last_log = 0.0
        self._max_depth = 0

    def __enter__(self) -> "AudioSession":
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_samples,
            device=self.device,
            callback=self._on_audio,
            # 2026-08-02 — DO NOT add latency="high" here. It is a no-op, and
            # this is the second time it has been mistaken for a fix.
            #
            # Measured on this box (MOVO MC1000 / MME host API), granted input
            # latency by config:
            #     blocksize=1280  latency=default | 'low' | 'high' | 0.5
            #         -> 80.0 ms in ALL FOUR cases
            # An explicit blocksize pins the host buffer to one block, so the
            # suggested-latency hint is ignored entirely. And sounddevice's
            # own default is ALREADY sd.default.latency == ['high','high'], so
            # passing it changes nothing even where blocksize is 0.
            #
            # sound_detector.py received exactly this "fix" on 2026-07-03
            # (commit 6e8e227) for the identical symptom — and its stream kept
            # overflowing right through 2026-08-02 ("[acoustic] stream status:
            # input overflow (x14)"), which is the log telling us the change
            # never did anything. Copying it here would have been a placebo.
            #
            # The armed-mode overflow (565 in a 69-minute armed window vs 1 in
            # the 112 unarmed minutes before it) is GIL starvation: a 90 ms
            # YOLO call straddles this callback's 80 ms budget. The levers that
            # would actually work are (a) less/burstier-scheduled armed CPU
            # work, or (b) an explicit NUMERIC latency, which does deepen the
            # ring but only with blocksize=0 and at a real cost to the wake-word
            # path. Neither should be chosen without armed-mode numbers from
            # the instrumentation below.
        )
        self._stream.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # Audio callback runs on sounddevice's thread — keep it short.
    def _on_audio(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:  # noqa: ARG002
        if status:
            # 2026-08-02: rate-limited. This used to print on EVERY status, and
            # each print is a timestamped FILE write executed inside the 80 ms
            # callback budget — so an overflow lengthened the very callback
            # whose lateness caused it. 565 of them fired in one armed window,
            # each one making the next overflow marginally more likely. A
            # self-amplifying diagnostic is worse than no diagnostic; the count
            # preserves the signal without the feedback loop.
            self._status_count += 1
            now = time.monotonic()
            if now - self._status_last_log >= _STATUS_LOG_INTERVAL_SEC:
                # Queue depth rides along: it distinguishes "the device ring
                # overflowed because the callback was late" (depth ~0, the
                # armed-CPU case) from "the CONSUMER fell behind" (depth grows).
                # The queue is deliberately unbounded — the STT path reads it to
                # assemble one contiguous question clip, so dropping from it
                # would inject the exact gaps this fix exists to remove.
                print(f"[audio] status: {status} "
                      f"(x{self._status_count} since last report; "
                      f"queue depth max {self._max_depth})",
                      file=sys.stderr)
                self._status_count = 0
                self._max_depth = 0
                self._status_last_log = now
        self._queue.put(indata[:, 0].copy())
        depth = self._queue.qsize()
        if depth > self._max_depth:
            self._max_depth = depth

    def read(self, timeout: float = READ_STALL_SEC) -> np.ndarray:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(
                f"microphone stream stalled — no audio for {timeout:.0f}s"
            ) from None

    def drain(self) -> None:
        """Discard any buffered chunks. Call before re-entering wake-word loop."""
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
