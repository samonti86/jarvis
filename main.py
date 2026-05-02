"""Jarvis — top-level entry point.

Milestone 3: wake word → STT → Claude (streaming) → TTS → speakers.
One AudioSession owns the mic for the run; wake_word and STT both consume from it.

Echo handling: while TTS plays through the speakers, the mic still captures it.
session.drain() at the top of each loop iteration discards that buffered echo so
the wake-word detector starts each turn on fresh, live audio. Tradeoff: a user
who interrupts during playback ("Hey Jarvis stop") will be drained too — they
need to wait for the response to finish before re-triggering. Acceptable for v1.
"""

from __future__ import annotations

import sys

from src.audio import AudioSession
from src.config import load
from src.llm import stream_response
from src.speech_to_text import transcribe_after_wake
from src.text_to_speech import speak
from src.wake_word import wait_for_wake_word


def main() -> None:
    cfg = load()
    if not cfg.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY missing. Add it to .env and try again.", file=sys.stderr)
        sys.exit(1)

    print("Jarvis ready. Say 'Hey Jarvis' followed by a question. Ctrl+C to quit.\n")

    with AudioSession(sample_rate=cfg.sample_rate) as session:
        while True:
            session.drain()
            wait_for_wake_word(session, threshold=cfg.wake_word_threshold)

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
                speak(full_response, language=transcript.language)
            print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
