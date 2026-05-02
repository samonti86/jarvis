"""Jarvis — top-level entry point.

Milestone 4: tray icon (main thread) + listen loop (daemon worker thread).
On Windows, pystray's icon loop hooks into the Win32 message pump and must run
on the main thread. The wake-word→STT→LLM→TTS pipeline runs in a daemon worker
that posts state transitions to the tray via tray.set_state().

Echo handling: while TTS plays through the speakers, the mic still captures it.
session.drain() at the top of each loop iteration discards that buffered echo so
the wake-word detector starts each turn on fresh, live audio. Tradeoff: a user
who interrupts during playback ("Hey Jarvis stop") will be drained too — they
need to wait for the response to finish before re-triggering.
"""

from __future__ import annotations

import sys
import threading

from src.audio import AudioSession
from src.config import Config, load
from src.llm import stream_response
from src.speech_to_text import transcribe_after_wake
from src.text_to_speech import speak
from src.tray import JarvisTray, State
from src.wake_word import wait_for_wake_word


def listen_loop(cfg: Config, tray: JarvisTray) -> None:
    """Runs in a daemon worker thread. Updates tray state at each transition."""
    with AudioSession(sample_rate=cfg.sample_rate) as session:
        while not tray.shutdown.is_set():
            tray.set_state(State.IDLE)
            session.drain()

            wait_for_wake_word(session, threshold=cfg.wake_word_threshold)
            if tray.shutdown.is_set():
                break
            tray.set_state(State.LISTENING)

            try:
                transcript = transcribe_after_wake(session, model_name=cfg.whisper_model)
            except Exception as exc:
                print(f"[main] STT failed: {exc}")
                continue

            if not transcript.text:
                print("[main] (no speech captured)\n")
                continue

            print(f"\n[user, {transcript.language}] {transcript.text}")
            print("[jarvis] ", end="", flush=True)
            tray.set_state(State.THINKING)

            response_chunks: list[str] = []
            try:
                for chunk in stream_response(
                    api_key=cfg.anthropic_api_key,
                    user_text=transcript.text,
                    model=cfg.claude_model,
                ):
                    response_chunks.append(chunk)
                    print(chunk, end="", flush=True)
            except Exception as exc:
                print(f"\n[main] LLM failed: {exc}")
                continue
            print()

            full_response = "".join(response_chunks).strip()
            if full_response:
                tray.set_state(State.SPEAKING)
                speak(full_response, language=transcript.language)
            print()


def main() -> None:
    cfg = load()
    if not cfg.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY missing. Add it to .env and try again.", file=sys.stderr)
        sys.exit(1)

    print("Jarvis ready. Tray icon active — right-click for Quit. Say 'Hey Jarvis' to begin.\n")

    tray = JarvisTray(on_quit=lambda: print("\nGoodbye."))

    worker = threading.Thread(target=listen_loop, args=(cfg, tray), daemon=True)
    worker.start()

    tray.run()  # blocks until Quit is clicked


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
