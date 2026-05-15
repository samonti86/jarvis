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
    stt_server_url: str        # M36 — HTTP URL of the GPU STT server (e.g. http://192.168.1.10:8000); blank = local CPU only
    stt_backend: str           # M36 — "auto" | "gpu" | "cpu"; auto = try GPU server first, fall back to CPU silently
    discord_webhook_url: str   # M38 — Discord webhook URL for security deterrent alerts; blank = no notification (M35 bluff only)
    smtp_host: str             # M38 — SMTP server (default: Gmail). Email fires in parallel with Discord when both are configured.
    smtp_port: int             # M38 — 587 = STARTTLS (default), 465 = implicit SSL
    smtp_username: str         # M38 — SMTP login (typically the From address). Blank disables email path.
    smtp_password: str         # M38 — for Gmail: a 16-char App Password, NOT the account password
    smtp_to: str               # M38 — recipient address (comma-separated allowed). Blank disables email path.
    face_match_threshold: float  # M39 — max Euclidean distance for a face match. 0.5 = security-grade strict; 0.6 = face_recognition's permissive default.


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
        stt_server_url=os.getenv("STT_SERVER_URL", "").strip().rstrip("/"),
        stt_backend=os.getenv("STT_BACKEND", "auto").strip().lower(),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", "").strip(),
        smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com").strip(),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        smtp_username=os.getenv("SMTP_USERNAME", "").strip(),
        # Password may contain meaningful trailing/leading whitespace? Gmail
        # App Passwords are alphanumeric 16-char strings — stripping is safe
        # and protects against an accidental newline in the .env value.
        smtp_password=os.getenv("SMTP_PASSWORD", "").strip(),
        smtp_to=os.getenv("SMTP_TO", "").strip(),
        face_match_threshold=float(os.getenv("JARVIS_FACE_MATCH_THRESHOLD", "0.5")),
    )
