"""openWakeWord integration: blocks on an AudioSession until 'Hey Jarvis' fires."""

from __future__ import annotations

import sys
import threading

import openwakeword
from openwakeword.model import Model

from src.audio import AudioSession

WAKEWORD_NAME = "hey_jarvis"


def _ensure_models_downloaded() -> None:
    try:
        openwakeword.utils.download_models([WAKEWORD_NAME])
    except Exception as exc:
        print(f"[wake_word] model download check failed: {exc}", file=sys.stderr)


def wait_for_wake_word(
    session: AudioSession,
    threshold: float = 0.5,
    shutdown_event: threading.Event | None = None,
    reset_event: threading.Event | None = None,
) -> None:
    """Block reading from `session` until the wake word scores >= threshold,
    or until shutdown_event / reset_event is set. Caller distinguishes the
    three exit paths by inspecting the events post-return.

    Audio chunks arrive every ~80ms, so the event checks at loop top mean
    quits and resets propagate within one chunk — fast enough to feel
    instant, and slow enough that we don't burn CPU spinning."""
    _ensure_models_downloaded()
    model = Model(wakeword_models=[WAKEWORD_NAME], inference_framework="onnx")

    print("[wake_word] listening for 'Hey Jarvis'...", file=sys.stderr)
    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            return
        if reset_event is not None and reset_event.is_set():
            return
        chunk = session.read()
        scores = model.predict(chunk)
        score = scores[WAKEWORD_NAME]
        if score >= threshold:
            print(f"[wake_word] detected (score={score:.2f})", file=sys.stderr)
            return
