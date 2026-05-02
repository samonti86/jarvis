"""Jarvis — top-level entry point.

Orchestrates the listen → transcribe → think → speak loop. Each component
lives in src/; this file is intentionally thin so the wiring stays readable.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Milestone 1: wire up wake_word + speech_to_text")


if __name__ == "__main__":
    main()
