"""Runtime configuration: loads .env and exposes tunables as a single object."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str
    claude_model: str
    whisper_model: str
    wake_word_threshold: float
    sample_rate: int


def load() -> Config:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    return Config(
        anthropic_api_key=api_key,
        claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        whisper_model=os.getenv("WHISPER_MODEL", "small"),
        wake_word_threshold=float(os.getenv("WAKE_WORD_THRESHOLD", "0.5")),
        sample_rate=16_000,  # required by both openWakeWord and Whisper
    )
