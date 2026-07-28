# Jarvis — Environment Variable Reference

Every environment variable Jarvis reads, in one place. Generated for audit
finding **2.2 (config sprawl)** — see [CODE_AUDIT.md](CODE_AUDIT.md).

**Two tiers, by design:**
- **Centralized** (`src/config.py` → the frozen `Config` dataclass, loaded once
  at startup via `load()`). The credentials + core knobs. `.env.example`
  documents these for setup.
- **Scattered** (read ad-hoc in the owning module, usually at import or first
  use). These are deliberately *local* tunables — they live next to the code
  they affect and rarely need touching. This file is their inventory, since
  they're otherwise invisible without grepping.

`load_dotenv()` runs in `config.py` (and, as of Tier 1.2, in
`outlook_calendar.py`) — so a `.env` at the repo root populates all of these.
Blank/unset values fall back to the listed default; an unset credential simply
disables that optional feature (the project's fail-soft contract).

---

## Core (centralized — `config.py`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | **(required)** | Claude API key. Jarvis exits if missing. |
| `CLAUDE_MODEL` | `claude-sonnet-5` | Main conversation model. |
| `SUMMARY_MODEL` | `claude-haiku-4-5-20251001` | Cheap model for session summaries. |
| `JARVIS_VOICE_EFFORT` | `medium` | Reasoning/acting depth for a spoken turn (`low`…`max`). Sonnet 5 defaults to `high` when unset, which over-thinks "what's the weather". Read in `llm.py`; an invalid value logs and falls back rather than 400-ing every turn. |
| `JARVIS_ENGINEER_EFFORT` | `high` | Same knob for engineer mode, where depth is the point. |
| `WHISPER_MODEL` | `small` | Local faster-whisper model size. |
| `WAKE_WORD_THRESHOLD` | `0.5` | openWakeWord confidence to trigger. |
| `MEMORY_RECALL_COUNT` | `10` | Recent session summaries injected into the system prompt. |
| `RETAIN_RAW_DAYS` | `30` | Transcript retention before prune-on-start. |

*(`sample_rate` is hardcoded to 16 000 Hz — required by both openWakeWord and
Whisper — and intentionally not an env var.)*

## Audio / capture

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_MIC_DEVICE` | `""` (Windows default) | config.py | Pin capture to a mic by name-substring or index (M67). |
| `CAMERA_INDEX` | `""` (auto) | cameras.py | Webcam index for vision + security. |
| `JARVIS_BARGE_IN` | `1` (on) | main.py | `0/false/no/off` disables mid-reply barge-in (M52). |

## STT GPU offload (client + `stt_server/`)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `STT_SERVER_URL` | `""` (local CPU) | config.py | GPU STT server URL (M36). |
| `STT_BACKEND` | `auto` | config.py | `auto` / `gpu` / `cpu`. |
| `STT_MODEL` | `small` | stt_server/server.py | Model on the GPU server. |
| `STT_DEVICE` | `cuda` | stt_server/server.py | Server compute device. |
| `STT_COMPUTE_TYPE` | `float16` | stt_server/server.py | Server compute precision. |

## Security (M34/M35/M39 + M44 memory watchdog)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_SECURITY_PASSPHRASE` | `""` (challenge step skipped) | config.py | Voice passphrase to clear a challenge. |
| `JARVIS_FACE_MATCH_THRESHOLD` | `0.5` | config.py | Max Euclidean distance for a face match. |
| `JARVIS_YOLO_THREADS` | `1` | security.py | Torch intra-op thread cap for YOLO (stutter-gate). |
| `JARVIS_MEMORY_WATCHDOG_MB` | `3500` | security.py | Private-bytes ceiling → auto-disarm (M44.2). |
| `JARVIS_MEMORY_MIN_AVAIL_MB` | `512` | security.py | System-available floor → auto-disarm (M44.2). |

## Acoustic awareness (M58)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_ACOUSTIC_MONITOR` | `""` (off) | config.py | Start acoustic awareness at launch (opt-in). |
| `JARVIS_ACOUSTIC_DISABLE` | `""` (none) | sound_detector.py | Comma-list of classes to suppress. |
| `JARVIS_ACOUSTIC_WATER_RMS_FLOOR` | `0.025` | sound_detector.py | RMS floor for `running_water` (pet-fountain guard). |
| `JARVIS_ACOUSTIC_THREADS` | `1` | sound_detector.py | Torch thread cap for PANNs (stutter-gate, M67). |
| `JARVIS_ACOUSTIC_DEBUG` | `""` (off) | sound_detector.py | Log real classifier scores + RMS, for calibrating the floors below. |
| `JARVIS_ACOUSTIC_RMS_FLOOR` | `0.02` | sound_detector.py | Global loudness floor an event must clear before any acoustic rule fires. |
| `JARVIS_ACOUSTIC_VOICE_THRESHOLD` | `0.60` | sound_detector.py | Confidence for the armed `voice_while_armed` rule (M81). The **weak** lever — wall-bleed *is* speech and scores high. |
| `JARVIS_ACOUSTIC_VOICE_RMS_FLOOR` | `0.06` | sound_detector.py | Loudness floor — the real through-the-wall discriminator (M87). Observed neighbour bleed: 0.020–0.044. |

## Voice, tone & speaker identity (M69 / M80 / M85)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_USER_NAME` | `you` | config.py | Name your primary voice enrolls under; appears in `[speaker]` logs. |
| `JARVIS_SPEAKER_THRESHOLD` | `0.60` | config.py | Min cosine similarity to call a voice recognized. Fail-open by design — see the note below. |
| `JARVIS_SPEAKER_GATE` | `""` (off) | config.py | Voice lock: drop confidently-unrecognized **voice** turns. Typed input is never gated. |
| `JARVIS_TONE` | `1` (on) | config.py | Tonal awareness — reads volume/pace/pauses from the STT clip into a descriptive cue. `0` disables. |

## Proactive intelligence (M78.2 / M79 / M83)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_ANTICIPATION` | `""` (off) | config.py | Cross-domain synthesis pass. Opt-in — it makes a recurring LLM call. |
| `JARVIS_ANTICIPATION_POLL_SECONDS` | `1800` | anticipation.py | Cadence of that pass. |
| `JARVIS_QUIET_HOURS` | `""` (off) | config.py | DND window, e.g. `22:00-08:00`. Only ROUTINE announces defer; important ones pierce. Wraps midnight. |
| `JARVIS_PREDICTION_FOLLOWUPS` | `1` (on) | predictions.py | Mine past sports predictions and surface resolved ones in the briefing. |
| `JARVIS_PREDICTION_MINE_MODEL` | `SUMMARY_MODEL` | predictions.py | Model for extraction (cheap). |
| `JARVIS_PREDICTION_RESOLVE_MODEL` | `CLAUDE_MODEL` | predictions.py | Model for the resolution judgment. |

## Long-horizon background agents (M91)

**This is the one setting that changes Jarvis's privacy posture.** Everything
else keeps audio and question text local (or LAN-local). A background agent
runs on Anthropic's infrastructure: it fetches pages and writes working files
inside an Anthropic-hosted sandbox. Off unless you turn it on, deliberately.

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_BACKGROUND_AGENTS` | `""` (**off**) | background_agent.py | Enable `start_background_task` and the poll/delivery loop. Blank/`0`/`false`/`no`/`off` ⇒ disabled. |

Notes:
- The agent's tool surface is **only** Anthropic's hosted toolset (bash, files,
  web search/fetch) inside a per-session container. It gets **no** Jarvis tools
  — no shell, no host filesystem, no camera, no self-update — and no route back
  to this machine. The only thing that crosses back is its report.
- All three task tools are in `_RESTRICTED_DENY`, so a phone or Discord turn
  cannot start one (dispatching is unbounded API spend, run unobserved).
- Concurrency is capped at 2. Sessions live server-side, so they survive a
  Jarvis restart and are re-attached on startup.

## Severe-weather alerts (M77 — US only, no API key)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_WEATHER_ALERTS` | on when `JARVIS_HOME_LOCATION` set | config.py | `0/false/off` disables proactive NWS alerts. |
| `JARVIS_WEATHER_ALERT_SEVERITY` | `severe` | weather_alerts.py | Lowest severity to announce: `extreme` / `severe` / `moderate` (noisier). |
| `JARVIS_WEATHER_POLL_SECONDS` | `600` (min 60) | weather_alerts.py | Poll cadence against api.weather.gov. |

## Presence / geofence auto-arm (M70)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_PRESENCE_ARM_DELAY` | `60.0` | config.py | Seconds to defer auto-arm after a "leave" ping; cancelled by an "arrive". Damps geofence boundary-flap. |
| `JARVIS_PRESENCE_GREETING` | `Welcome home, sir.` | config.py | Spoken on an arrive that disarms an armed house. |

## Interpreter mode (M87)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_INTERPRETER_LANGS` | `""` (→ `en,es`) | config.py | Language pair as `primary,secondary`. Each side is spoken in that language's `VOICE_BY_LANG` voice. |

## Ambient HUD (M84)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_HUD` | `""` (off) | config.py | Translucent click-through overlay: arc-reactor orb, waveform, clock, security badge. Windows-only, fail-soft. |

## Homelab monitor (M56)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_HOMELAB_MONITOR` | `""` (off) | config.py | Start the poll loop at launch (opt-in). |
| `JARVIS_HOMELAB_LABEL` | `the Plex laptop` | homelab_monitor.py | Spoken name of the monitored host. |
| `JARVIS_HOMELAB_POLL_SECONDS` | `60` (floor 30) | homelab_monitor.py | Poll cadence. |
| `JARVIS_HOMELAB_FAIL_THRESHOLD` | `3` (floor 1) | homelab_monitor.py | Consecutive fails before OK→DOWN (flap damping). |
| `JARVIS_HOMELAB_DISK_MIN_PCT` | `10` (floor 1) | homelab_monitor.py | Free-space % below which a drive is "low". |

## Calendar (M62.1 / M62.2)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `OUTLOOK_ICAL_URL` | `""` | outlook_calendar.py | Published iCal URL (read-only calendar access). |
| `JARVIS_CALENDAR_REMINDERS` | on when configured | calendar_monitor.py | `0/false/no/off` disables proactive pre-event reminders. |
| `JARVIS_CALENDAR_LEAD_MIN` | `15` (floor 1) | calendar_monitor.py | Minutes before an event to announce. |
| `JARVIS_CALENDAR_POLL_SECONDS` | `60` (floor 30) | calendar_monitor.py | Poll cadence. |

## Briefing / location (M55 / M63)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_HOME_LOCATION` | `""` | briefing.py, good_night.py | Default location for the briefing's weather. |
| `JARVIS_HOME_UNITS` | `imperial` | briefing.py, good_night.py | `imperial` / `metric`. |

## Plex (M21 / M24)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `PLEX_URL` | `""` | config.py | Plex Media Server URL (MCP). Blank disables. |
| `PLEX_TOKEN` | `""` | config.py, plex_actions.py | Plex auth token. |
| `PLEX_LAPTOP_HOST` | `""` | config.py | SSH host of the Plex laptop. Blank disables remote tools. |
| `PLEX_LAPTOP_USER` | `""` | config.py | SSH username. |
| `PLEX_LAPTOP_KEY_PATH` | `""` (→ `~/.ssh/id_ed25519`) | config.py | SSH private key path. |
| `PLEX_LAPTOP_LOG_PATH` | `""` (→ module default) | config.py | Plex log path on the laptop. |

## Notifications (M38)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `DISCORD_WEBHOOK_URL` | `""` | config.py | Webhook for security/monitor push. Blank disables. |
| `SMTP_HOST` | `smtp.gmail.com` | config.py | SMTP server for email alerts. |
| `SMTP_PORT` | `587` | config.py | `587` STARTTLS / `465` implicit SSL. |
| `SMTP_USERNAME` | `""` | config.py | SMTP login. Blank disables email. |
| `SMTP_PASSWORD` | `""` | config.py | App password (Gmail: 16-char). |
| `SMTP_TO` | `""` | config.py | Recipient(s), comma-separated. |
| `JARVIS_REMINDER_DISCORD` | `1` (on when webhook set) | reminders.py | Also push a reminder's text to the webhook when it fires. `0` disables. |

## Discord bot bridge (2026-06-02)

A private channel as a two-way conversational client. Distinct from the one-way
webhook above — this needs a Gateway bot. Runs on the **restricted** tool surface
(no shell, filesystem, code execution, or self-update), enforced server-side.

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `DISCORD_BOT_TOKEN` | `""` (bridge off) | discord_bot.py | Bot token. The only secret of the three. Needs the Message Content intent. |
| `DISCORD_CHANNEL_ID` | `""` | discord_bot.py | The single channel the bot listens to. |
| `DISCORD_ALLOWED_USER_IDS` | `""` (nobody) | discord_bot.py | Comma-separated user IDs permitted to talk to Jarvis. Fail-closed allowlist. |

## Hands-free barge-in (SHELVED — default off)

Interrupting mid-reply *without* the wake word. Built and shelved: once Jarvis's
own echo is cancelled, an energy gate fires on any residual sound, because it
detects the *presence* of sound, not the *intent* to interrupt. Kept behind a
default-off flag for a headset/quiet-room scenario. Wake-word barge-in
(`JARVIS_BARGE_IN`) is the robust path and is on by default.

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_HANDS_FREE_BARGE` | `""` (off) | main.py | Master switch for the shelved energy-gate path. |
| `JARVIS_BARGE_DEBUG` | `""` (off) | main.py | Log live AEC/energy numbers for calibration. |
| `JARVIS_BARGE_THRESHOLD` | `1187.0` | aec_barge.py | Cleaned-RMS gate (int16) the residual must clear to count as an interruption. |
| `JARVIS_BARGE_DELAY_MS` | see source | main.py | Delay before the gate arms after playback starts. |
| `JARVIS_BARGE_FILTER_LEN` | see source | main.py | SpeexDSP echo-canceller filter length. |
| `JARVIS_BARGE_SUSTAIN` | see source | main.py | Consecutive frames above threshold required to fire. |
| `JARVIS_BARGE_WIN_FRAMES` | see source | main.py | Sliding-window size for the energy decision. |

## Remote console (M48)

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_REMOTE_TOKEN` | `""` (server OFF) | config.py | Shared secret; **blank = remote console disabled**. |
| `JARVIS_REMOTE_PORT` | `8765` | config.py | WS server port. |
| `JARVIS_REMOTE_BIND` | `0.0.0.0` | config.py | Bind interface (LAN). Never port-forward. |
| `JARVIS_TLS_CERT_FILE` | `""` | config.py | TLS cert (Tailscale). Both cert+key set ⇒ HTTPS/WSS. |
| `JARVIS_TLS_KEY_FILE` | `""` | config.py | TLS key. Either missing ⇒ plain HTTP/WS. |

## Data-tool API keys

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `RAWG_API_KEY` | `""` | games.py | RAWG (games). Blank disables `get_game_info`. |
| `TMDB_API_KEY` | `""` | tmdb.py | TMDB (movies/TV/person). Blank disables those tools. |
| `TMDB_WATCH_REGION` | `""` (→ `US`) | tmdb.py | ISO-3166-1 country for "where can I watch X?" (`mode=providers`, M73). |
| `WOLFRAM_APP_ID` | `""` | wolfram.py | WolframAlpha. Blank disables `wolfram_query`. |

## Misc

| Variable | Default | Read in | Purpose |
|----------|---------|---------|---------|
| `JARVIS_KNOWLEDGE_DIR` | `""` (→ `~/repos/jarvis-knowledge`) | knowledge.py | Knowledge-base corpus dir. |
| `DIAGNOSTICS_COLLECTOR_PATH` | `""` (→ sibling repo) | diagnostics_collector.py | Path to the hs-windows-diagnostics script. |

## Internal (set BY Jarvis, not by the user)

| Variable | Set by | Purpose |
|----------|--------|---------|
| `JARVIS_LOG_PATH` | `jarvis.pyw` / watchdog | Tells `main.py` the log file is already opened + redirected. |
| `JARVIS_WATCHDOG` | `jarvis_watchdog.pyw` | `=1` signals main.py to exit 42 (respawn) instead of self-spawning on restart (M65). |

---

### Notes for maintainers
- **A new tunable** should follow the existing split: a credential or core knob
  goes in `config.py`'s `Config`; a module-local tuning constant stays in its
  module — but **add a row here** so the surface stays discoverable.
- `JARVIS_YOLO_THREADS` and `JARVIS_ACOUSTIC_THREADS` both cap torch's
  process-global intra-op pool; when security + acoustic run together they set
  the same value. See the stutter-gate post-mortem
  ([project_security_audio_stutter_gate](../docs/MILESTONES.md)).
- The `JARVIS_*_MONITOR` flags and `JARVIS_REMOTE_TOKEN` are all **safe-default
  OFF** — opt-in, per the least-privilege stance.
