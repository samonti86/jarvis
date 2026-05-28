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
        idx = int(spec)
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

    def __enter__(self) -> "AudioSession":
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_samples,
            device=self.device,
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
