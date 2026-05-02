"""Jarvis — top-level entry point.

Milestone 2: wake word → STT → Claude (streaming) → print. No TTS yet.
One AudioSession owns the mic for the run; wake_word and STT both consume from it.
"""

from __future__ import annotations

import sys

from src.audio import AudioSession
from src.config import load
from src.llm import stream_response
from src.speech_to_text import transcribe_after_wake
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
            try:
                for chunk in stream_response(
                    api_key=cfg.anthropic_api_key,
                    user_text=transcript.text,
                    model=cfg.claude_model,
                ):
                    print(chunk, end="", flush=True)
            except Exception as exc:
                print(f"\n[main] LLM failed: {exc}")
                continue
            print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
