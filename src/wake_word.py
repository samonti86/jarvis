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


# Process-wide singleton. Building a Model spins up ONNX Runtime inference
# sessions, and ORT is notorious for not fully releasing native memory on GC
# — so the original "new Model() per wake cycle" pattern leaked tens of MB
# every wake→process→respond round-trip. We build it once and reset() its
# streaming feature buffers between cycles instead. reset() is confirmed
# present on the installed openWakeWord Model API.
#
# Threading invariant: wait_for_wake_word is called only from listen_loop
# (the single voice-path thread). The singleton therefore has exactly one
# consumer thread and needs no lock. A second caller from another thread
# would have to add one.
_model: "Model | None" = None


def _get_model() -> Model:
    global _model
    if _model is None:
        # First use: download-check (once per process, not per cycle — the
        # old code re-hit this every wake) then construct.
        _ensure_models_downloaded()
        _model = Model(wakeword_models=[WAKEWORD_NAME], inference_framework="onnx")
    else:
        # Clear accumulated streaming features so stale audio from the
        # previous cycle doesn't bias the first predictions of this one.
        # main.py already drains the AudioSession before calling us, so a
        # clean feature buffer matches a clean audio buffer.
        _model.reset()
    return _model


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
    model = _get_model()

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
