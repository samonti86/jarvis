"""Microphone session: a single sounddevice stream shared by wake_word + STT.

A long-lived InputStream reduces wake-word→STT latency to zero gap (vs.
opening/closing a new stream per stage, which loses 100-300ms of audio).
"""

from __future__ import annotations

import queue
import sys
from typing import Optional

import numpy as np
import sounddevice as sd

CHUNK_SAMPLES = 1280  # 80 ms @ 16 kHz; matches openWakeWord's window
SAMPLE_RATE = 16_000


class AudioSession:
    """Continuous mic capture into a thread-safe queue of int16 mono chunks."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_samples: int = CHUNK_SAMPLES) -> None:
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None

    def __enter__(self) -> "AudioSession":
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_samples,
            callback=self._on_audio,
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
            print(f"[audio] status: {status}", file=sys.stderr)
        self._queue.put(indata[:, 0].copy())

    def read(self) -> np.ndarray:
        return self._queue.get()

    def drain(self) -> None:
        """Discard any buffered chunks. Call before re-entering wake-word loop."""
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
