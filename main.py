"""Jarvis — top-level entry point.

Milestone 1: text-only loop. One AudioSession owns the mic for the whole run;
wake_word and STT both consume from it (no stream-open gap between stages).
"""

from __future__ import annotations

from src.audio import AudioSession
from src.config import load
from src.speech_to_text import transcribe_after_wake
from src.wake_word import wait_for_wake_word


def main() -> None:
    cfg = load()
    print("Jarvis ready. Say 'Hey Jarvis' followed by a question. Ctrl+C to quit.\n")

    with AudioSession(sample_rate=cfg.sample_rate) as session:
        while True:
            # Drain any audio that piled up during the previous turn's transcription
            # so the wake-word loop starts on fresh, live audio.
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
            print(f"\n[{transcript.language}] {transcript.text}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
