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
    summary_model: str         # Haiku by default — small task, cheap + fast
    whisper_model: str
    wake_word_threshold: float
    sample_rate: int
    memory_recall_count: int   # how many recent summaries to inject into the system prompt
    retain_raw_days: int       # transcripts older than this are pruned on app start
    plex_url: str              # M21 — empty if Plex MCP not configured
    plex_token: str            # M21 — empty if Plex MCP not configured
    plex_laptop_host: str      # M24 — IP/hostname of the Plex laptop (empty disables tools)
    plex_laptop_user: str      # M24 — SSH username on the Plex laptop
    plex_laptop_key_path: str  # M24 — path to private key (~ expanded); blank → default ed25519
    plex_laptop_log_path: str  # M24 — Plex Media Server.log path on the laptop; blank → default
    security_passphrase: str   # M35 — voice passphrase to clear a security-mode challenge; blank disables challenge step


def load() -> Config:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    return Config(
        anthropic_api_key=api_key,
        claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        summary_model=os.getenv("SUMMARY_MODEL", "claude-haiku-4-5-20251001"),
        whisper_model=os.getenv("WHISPER_MODEL", "small"),
        wake_word_threshold=float(os.getenv("WAKE_WORD_THRESHOLD", "0.5")),
        sample_rate=16_000,  # required by both openWakeWord and Whisper
        memory_recall_count=int(os.getenv("MEMORY_RECALL_COUNT", "10")),
        retain_raw_days=int(os.getenv("RETAIN_RAW_DAYS", "30")),
        plex_url=os.getenv("PLEX_URL", "").strip(),
        plex_token=os.getenv("PLEX_TOKEN", "").strip(),
        plex_laptop_host=os.getenv("PLEX_LAPTOP_HOST", "").strip(),
        plex_laptop_user=os.getenv("PLEX_LAPTOP_USER", "").strip(),
        plex_laptop_key_path=os.getenv("PLEX_LAPTOP_KEY_PATH", "").strip(),
        plex_laptop_log_path=os.getenv("PLEX_LAPTOP_LOG_PATH", "").strip(),
        security_passphrase=os.getenv("JARVIS_SECURITY_PASSPHRASE", "").strip(),
    )
