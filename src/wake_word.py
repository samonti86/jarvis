"""openWakeWord integration: blocks on an AudioSession until 'Hey Jarvis' fires."""

from __future__ import annotations

import sys

import openwakeword
from openwakeword.model import Model

from src.audio import AudioSession

WAKEWORD_NAME = "hey_jarvis"


def _ensure_models_downloaded() -> None:
    try:
        openwakeword.utils.download_models([WAKEWORD_NAME])
    except Exception as exc:
        print(f"[wake_word] model download check failed: {exc}", file=sys.stderr)


def wait_for_wake_word(session: AudioSession, threshold: float = 0.5) -> None:
    """Block reading from `session` until the wake word scores >= threshold."""
    _ensure_models_downloaded()
    model = Model(wakeword_models=[WAKEWORD_NAME], inference_framework="onnx")

    print("[wake_word] listening for 'Hey Jarvis'...", file=sys.stderr)
    while True:
        chunk = session.read()
        scores = model.predict(chunk)
        score = scores[WAKEWORD_NAME]
        if score >= threshold:
            print(f"[wake_word] detected (score={score:.2f})", file=sys.stderr)
            return
