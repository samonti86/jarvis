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
    remote_token: str          # M48 — shared secret the phone PWA must present on WS connect. BLANK = remote console DISABLED (safe default: the server only starts when a token is set).
    remote_port: int           # M48 — TCP port the LAN remote-console server listens on.
    remote_bind: str           # M48 — interface to bind. "0.0.0.0" = all (LAN-reachable; the token + a LAN-only firewall rule are the controls). Never port-forward this.
    tls_cert_file: str         # M48.3 prereq — absolute path to a TLS cert (e.g. from `tailscale cert <host>.ts.net`). BOTH cert+key set ⇒ HTTPS/WSS; either missing ⇒ plain HTTP/WS (LAN-mode fallback). File MUST live outside the repo (it's a credential).
    tls_key_file: str          # M48.3 prereq — absolute path to the matching TLS private key. Same on/off semantics as tls_cert_file. Keep out of git.
    homelab_monitor_enabled: bool  # M56 — start the proactive homelab monitor at launch. Default off (opt-in via this flag or the tray toggle). Poll/threshold/disk tunables live in src/homelab_monitor.py.
    acoustic_monitor_enabled: bool  # M58 — start acoustic awareness at launch. Default off (opt-in via this flag or the tray toggle). Per-class kill switch + volume floor live in src/sound_detector.py.
    mic_device: str            # Mic pin: a device-name substring (e.g. "MC1000") or an integer index. Blank = Windows default input. Resolved to an index at startup; both the main capture and acoustic awareness use it. For a dedicated Jarvis-only mic, pin by name so a USB re-enumeration / default-device change can't silently steal capture.
    speaker_threshold: float   # M69 — min cosine similarity for a recognized speaker. 0.60 default: live data showed the enrolled user 0.83-0.95 normally but 0.66 with a rough/hoarse voice, vs ~0.566 for background media — so 0.60 sits below the rough-voice floor (fail-open: never ignore the real user) while still rejecting media. Tight margin to media (~0.03); tune per setup via JARVIS_SPEAKER_THRESHOLD. Active only once a voice is enrolled.
    speaker_gate_enabled: bool  # M69 — opt-in "respond only to enrolled voices" gate. Default off. When on AND a voice is enrolled, a confidently-unrecognized turn (e.g. background TV) is dropped. Fail-open: borderline still responds.
    user_name: str             # M69 — the name the primary user is enrolled under (via "enroll my voice" / the tray). Shows in [speaker] logs. Default "you"; set JARVIS_USER_NAME to your name.


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
        # M48 remote console (LAN). Token blank → feature OFF (the server is
        # never started). This is the safe default: the remote surface — which
        # can arm/disarm security — only exists once you've deliberately set a
        # secret. Bind 0.0.0.0 so it's LAN-reachable; the token + a LAN-scoped
        # Windows Firewall rule are the security controls. Do NOT port-forward.
        remote_token=os.getenv("JARVIS_REMOTE_TOKEN", "").strip(),
        remote_port=int(os.getenv("JARVIS_REMOTE_PORT", "8765")),
        remote_bind=os.getenv("JARVIS_REMOTE_BIND", "0.0.0.0").strip(),
        # M48.3 prereq — TLS via Tailscale's MagicDNS `*.ts.net` cert
        # (`tailscale cert <host>` mints the pair via Let's Encrypt; both files
        # are credentials and MUST live outside the repo). When both are set
        # the WS server speaks HTTPS+WSS (secure context → unlocks
        # getUserMedia for push-to-talk, loosens iOS <audio> gesture rules,
        # enables PWA Add-to-Home-Screen lifecycle, AND delivers off-LAN).
        # Either missing ⇒ fall back to plain HTTP/WS like today (LAN-mode
        # dev path) — never crash on a misconfigured cert, the optional-
        # component contract.
        tls_cert_file=os.getenv("JARVIS_TLS_CERT_FILE", "").strip(),
        tls_key_file=os.getenv("JARVIS_TLS_KEY_FILE", "").strip(),
        # M56 — proactive homelab monitoring. OFF unless this is explicitly
        # truthy: the background poll loop is opt-in (the tray toggle is the
        # other surface). Same safe-default reasoning as security mode.
        homelab_monitor_enabled=(
            os.getenv("JARVIS_HOMELAB_MONITOR", "").strip().lower()
            in ("1", "true", "yes", "on")
        ),
        # M58 — acoustic awareness. OFF unless explicitly truthy: the
        # background inference loop is opt-in (the tray toggle is the other
        # surface). Same safe-default reasoning as security mode + the
        # homelab monitor.
        acoustic_monitor_enabled=(
            os.getenv("JARVIS_ACOUSTIC_MONITOR", "").strip().lower()
            in ("1", "true", "yes", "on")
        ),
        # Mic device pin. Blank ⇒ Windows default (legacy behaviour). A name
        # substring (case-insensitive, e.g. "MC1000") or an integer index pins
        # capture to a specific input — the right call for a dedicated mic so
        # it need not be the system default and a USB shuffle can't reroute
        # Jarvis. Resolved to an index at startup (see audio.resolve_input_device).
        mic_device=os.getenv("JARVIS_MIC_DEVICE", "").strip(),
        # M69 — speaker identification. The recognize-threshold is tuned
        # fail-open (below the measured you-vs-you/you-vs-media midpoint) so a
        # degraded clip of the real user still passes. The gate is opt-in and
        # OFF unless explicitly truthy — identification itself activates simply
        # by enrolling a voice (no flag), since labeling who spoke is harmless.
        speaker_threshold=float(os.getenv("JARVIS_SPEAKER_THRESHOLD", "0.60")),
        speaker_gate_enabled=(
            os.getenv("JARVIS_SPEAKER_GATE", "").strip().lower()
            in ("1", "true", "yes", "on")
        ),
        user_name=os.getenv("JARVIS_USER_NAME", "you").strip() or "you",
    )
