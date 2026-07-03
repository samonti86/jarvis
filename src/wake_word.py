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


# M52 — barge-in. The monitor below runs openWakeWord *concurrently* with TTS
# playback, on its own thread, so the user can cut Jarvis off mid-reply with
# "Hey Jarvis". It needs its own Model: _model (above) documents a "one
# consumer thread" invariant and the monitor is a different thread from
# listen_loop. Two ORT inference sessions cost a little extra resident memory,
# but each is built ONCE and reset() between turns — the same anti-leak
# discipline _get_model uses (ORT doesn't fully release native memory on GC).
_barge_model: "Model | None" = None


def _get_barge_model() -> Model:
    """Process-wide singleton for the barge-in monitor thread — see the note
    above for why it's separate from _model. Built lazily on first barge-in,
    reset() on subsequent turns to clear stale streaming features."""
    global _barge_model
    if _barge_model is None:
        _ensure_models_downloaded()
        _barge_model = Model(wakeword_models=[WAKEWORD_NAME], inference_framework="onnx")
    else:
        _barge_model.reset()
    return _barge_model


def monitor_for_wake_word(
    session: AudioSession,
    interrupt_event: threading.Event,
    stop_event: threading.Event,
    threshold: float = 0.5,
) -> None:
    """Barge-in monitor (M52). Runs on its own thread for the duration of a
    TTS reply: reads mic chunks, scores them with openWakeWord, and on a
    "Hey Jarvis" detection sets `interrupt_event` — and does *nothing else*.

    It deliberately touches no audio API. The actual playback cut (sd.stop)
    is performed by speak_streaming, on the thread that owns the WASAPI/COM
    output stream — those streams are COM space-threaded on Windows and
    calling sd.stop() from this monitor thread hard-crashes the process (the
    original M52 bug; cf. project_wasapi_thread_audio_owner). The single
    Event is the entire cross-thread contract: speak_streaming's poll loop
    and stream_response both watch it.

    Exits on its own detection, or when `stop_event` is set — which is how
    the caller winds it down when a reply finished normally with no barge-in.
    session.read() returns every ~80ms (the mic InputStream is continuous),
    so a stop request is honoured within one chunk.

    Echo-safety: the mic captures Jarvis's *own* TTS output while this runs,
    but the wake word cannot self-trigger — Jarvis never says "Hey Jarvis"
    in a reply. That is the whole reason the trigger is the wake word and
    not a bare "stop" (which Jarvis saying "stop" would trip on himself).

    Threading: this is the sole AudioSession reader while it runs. The
    voice-path thread that owns the session is blocked inside speak_streaming
    (playback) for exactly this window, and the text path is serialized out
    by processing_lock — so there is no concurrent reader to race.
    """
    model = _get_barge_model()
    print("[barge-in] monitoring for 'Hey Jarvis' during playback", file=sys.stderr)
    while not stop_event.is_set():
        try:
            chunk = session.read()
        except Exception as exc:  # noqa: BLE001 — a dead mic mid-reply just
            # ends the monitor (no barge-in possible); the reply keeps playing
            # and the voice loop's supervisor handles the mic on its next read.
            print(f"[barge-in] mic read failed ({exc}) — monitor exiting",
                  file=sys.stderr)
            return
        if stop_event.is_set():
            return
        scores = model.predict(chunk)
        if scores[WAKEWORD_NAME] >= threshold:
            print(
                f"[barge-in] interrupt detected (score={scores[WAKEWORD_NAME]:.2f})",
                file=sys.stderr,
            )
            interrupt_event.set()
            return
