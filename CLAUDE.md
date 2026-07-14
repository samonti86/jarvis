# Project: jarvis

Agent-facing project instructions. Read this before touching the codebase.
Full historical detail — every milestone, decision, and post-mortem — lives in
[`docs/MILESTONES.md`](docs/MILESTONES.md).

## Goal
A Windows desktop voice assistant. An always-on microphone listens for the wake
word "Jarvis" / "Hey Jarvis", transcribes the spoken question locally, sends the
text to the Claude API with a Jarvis-personality system prompt, and reads the
response back through the speakers via TTS. Inspired by Tony Stark's J.A.R.V.I.S.:
courteous, dryly witty, concise.

The interesting part is not the voice loop — it is everything hung off it: an
agentic tool layer (~40 tools), proactive background monitors, a vision/security
subsystem, a phone client, and a set of engineering conventions strict enough to
keep an always-on process honest.

## Stack / Tools
- **Language**: Python 3.12 on Windows.
  - Windows-native, not WSL: audio device access. WSL2 audio bridging is
    unreliable for always-on real-time listening.
- **Wake word**: [openWakeWord](https://github.com/dscripka/openWakeWord)
  - MIT, fully local, no API key or account. Ships a pre-trained `hey_jarvis`
    model. CPU-only, ~3-5% continuous utilization.
  - Slightly higher false-positive rate than commercial alternatives (Porcupine);
    tunable via a confidence threshold. Porcupine was the original pick but now
    requires a company email, which rules it out for an open personal project.
- **Speech-to-text**: [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
  - Local Whisper on CPU; the small/base model is plenty for short commands.
    No internet required — question audio never leaves the machine.
  - Optionally offloaded to a CUDA box on the LAN over a small HTTP server
    (cuts transcription 5-10 s → 1.5-2.5 s); falls back to local on any failure.
- **LLM**: Anthropic Python SDK (`anthropic`)
  - Default `claude-sonnet-5` (`CLAUDE_MODEL` overrides). Haiku for the cheap
    background jobs (session summarizer, prediction miner). Opus only if a
    request genuinely needs more reasoning.
  - **Thinking is explicit.** Sonnet 5 runs *adaptive* thinking when the
    `thinking` param is omitted. Voice and background paths pass
    `thinking={"type": "disabled"}` (latency, plus small `max_tokens` budgets
    would be truncated by an unplanned thinking block); engineer mode passes
    `{"type": "adaptive"}`.
  - **Prompt caching** on the system prompt — it is reused every turn. Per-turn
    volatile context (clock, speaker identity) rides a *second, uncached* system
    block so it never invalidates the cache.
  - **Streaming** so TTS can start before the reply is complete.
- **Text-to-speech**: [`edge-tts`](https://github.com/rany2/edge-tts) primary,
  [`pyttsx3`](https://pyttsx3.readthedocs.io/) fallback.
  - Edge is a free Microsoft online voice, surprisingly good. pyttsx3 is offline
    (Windows SAPI) — the graceful degradation path when Edge is unreachable.
- **Audio I/O**: [`sounddevice`](https://python-sounddevice.readthedocs.io/) —
  cleaner than PyAudio, handles streaming well.
- **UI**: `pystray` tray icon (four states) + a `customtkinter` console window.
- **Vision**: `opencv-python` + `ultralytics` (YOLOv8n person detection) +
  `face_recognition`/dlib for the enrolled-face auth path.
- **Acoustic classification**: PANNs Cnn14 (`panns_inference`) — 527 AudioSet
  classes, ~0.2 s CPU inference.
- **Speaker ID**: Resemblyzer d-vectors (256-d), cosine similarity.
- **Env**: `python-dotenv`. **Async**: `asyncio` around the listen → process →
  respond cycle; background subsystems are daemon threads.

## Environment
- Windows 10/11, Python 3.12+ in a virtualenv at `./venv`.
- Microphone + speaker. A specific capture device can be pinned by name
  substring or index via `JARVIS_MIC_DEVICE` (blank = Windows default).
- Secrets live in `.env` (gitignored). `ANTHROPIC_API_KEY` is the only hard
  requirement; every other integration (TMDB, Discord, Wolfram, Plex, calendar,
  weather) is optional and degrades to "not configured" rather than failing.
- `.env.example` is the committed template. `docs/ENV_VARS.md` is the full
  inventory of tunables.

## Key Files & Directories
The core listen → process → respond spine:
- `main.py` — top-level loop; wake word → STT → LLM → TTS. Owns `TurnRunner`
  (conversation history, per-turn orchestration) and wires every subsystem.
- `jarvis.pyw` — silent launcher (pythonw, no console); logs to
  `%LOCALAPPDATA%\Jarvis\jarvis.log`.
- `jarvis_watchdog.pyw` — supervisor; respawns `main.py` on crash.
- `src/wake_word.py` · `src/speech_to_text.py` · `src/llm.py` ·
  `src/text_to_speech.py` · `src/audio.py` — the spine's modules.
- `src/config.py` — loads `.env`, holds all runtime tunables.
- `src/ui.py` / `src/console.py` / `src/tray.py` / `src/hud.py` — the UI fan-out
  facade and its three sinks (console window, tray, ambient overlay).
- `src/security.py` · `src/sound_detector.py` · `src/speaker_id.py` ·
  `src/face_auth.py` — the sensing subsystems.
- `src/remote_console.py` / `src/remote_pwa.py` / `src/discord_bot.py` — the
  remote clients.
- `scripts/run_all_tests.py` — the unified regression gate (see Commands).
- `docs/MILESTONES.md` — the engineering log. `docs/CODE_AUDIT.md` — the
  consolidation-pass audits. `docs/ENV_VARS.md` — config inventory.

The remaining ~50 `src/` modules are individual tools and monitors (weather,
news, TMDB, reminders, knowledge base, …). They are deliberately **not**
enumerated here — a full manifest rots. Grep `src/llm.py` for the tool schemas;
that file is the authoritative index of what Jarvis can do.

## Commands
```powershell
# One-time setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then edit with real keys

# Run (console — dev/debug)
python main.py

# Run (silent — production / autostart path, no console window)
pythonw jarvis.pyw

# Run (supervised — production; respawns on crash)
pythonw jarvis_watchdog.pyw

# THE CLOSE-OUT SANITY GATE — must be green before any commit
venv\Scripts\python.exe scripts\run_all_tests.py
```

The gate folds in: `py_compile` over `main.py` + launchers + every `src/*.py`;
an `import main` module-wiring smoke test; a structural JS check on the PWA
(`scripts/js_parse_gate.py`); and every `scripts/*_test.py` suite (45 suites at
last count). It must exit 0. If the venv is unavailable, the minimum fallback is
`python -m py_compile main.py jarvis.pyw src/*.py`.

## Architecture

### The spine
`wake word → capture → STT → LLM (agentic, streaming) → TTS`, with the tray/
console/HUD reflecting each phase. `TurnRunner` in `main.py` owns a single turn:
it appends to history, streams the response, runs the tool loop, feeds sentences
to TTS as they complete, and finalizes (memory write, telemetry).

Turns arrive from **four origins** — local voice, the console text box, the
phone PWA, and a Discord channel — through one `text_queue`. The origin is a
first-class parameter: it derives `speak` (never talk to an empty room for a
phone-typed message) and `restricted` (see Least privilege below).

### The tool layer
Claude drives an agentic loop (max 8 iterations) over:
- **Server-side**: `web_search`, `web_fetch`.
- **Client-side data tools** (all fail-soft, all read-only): weather + NWS
  weather alerts, sports, news (RSS), TMDB film/TV/person/watch-providers, game
  info + playtime, WolframAlpha, a personal knowledge base (hybrid FTS5 +
  embedding search over a curated markdown corpus), conversation recall
  (full-text + semantic search over verbatim session transcripts), reminders,
  and composition tools (`get_briefing`, `get_good_night`, `status_report`,
  `what_did_you_hear`).
- **Client-side action tools** (gated — see below): `system_control`,
  `pc_shell` (read-only 18-verb allowlist), `run_code` (Python in an ephemeral
  network-less Podman container), `update_jarvis`, file/screen/camera capture.
- **MCP**: a Plex Media Server MCP subprocess bridged into the same tool loop
  behind a voice allowlist.

### Subsystems (all optional, all fail-soft)
- **Vision security** — YOLOv8n person detection on a webcam; a pet-rejection
  height heuristic; a challenge-response flow (spoken passphrase *or* enrolled
  face, first match wins) before any deterrent fires; evidence snapshots; push
  notification. Arm/disarm by voice, tray, or a geofence webhook.
- **Acoustic awareness** — PANNs classifier on its own input stream; fires on a
  small set of salient classes. While armed it also triggers a look-and-describe
  (camera frame → Claude vision → push).
- **Proactive monitors** — homelab health, calendar (pre-event announces),
  severe weather, and an "anticipation" layer that fuses the world-state and
  asks the model for at most one genuinely useful cross-domain insight per tick
  (most ticks correctly produce nothing).
- **Remote clients** — a token-gated `websockets` server + PWA (type, talk,
  hear replies, see state), and a Discord bot bridge.
- **Reliability** — a crash watchdog, a mic-session supervisor, a memory
  watchdog, atomic+fsync'd JSON stores.

## Constraints & Rules

### Privacy by design
- **Wake word runs locally.** The mic stream is always open, but only
  openWakeWord sees the raw audio, and it is only looking for one pattern.
- **Speech-to-text is local.** The audio of the actual question never leaves the
  machine (or, with GPU offload enabled, never leaves the LAN).
- **Only the transcribed text** goes to the Claude API.
- Edge TTS *is* online — Microsoft sees the response text. Switch to pyttsx3 as
  primary if that matters.
- Any subsystem that ships data off-box (push notifications, remote clients) is
  opt-in and off by default.

### Latency targets
- Wake word: <100 ms.
- STT: <2 s for a typical command.
- First audible TTS chunk within 1 s of LLM streaming start.
- End-of-question to start-of-answer: 2-3 s.

### Streaming everything
Stream Claude's response; feed TTS in sentence chunks rather than waiting for
the full reply. This is what makes the interaction feel immediate rather than
transactional.

### Engineering rules
These are the conventions that keep an always-on, always-listening process
maintainable. Follow them.

- **Never commit `.env`.** `.gitignore` enforces it.
- **Fail soft, never crash the loop.** Every optional component (TTS backend,
  model download, a network tool, an MCP subprocess) must log and degrade. The
  listening loop is the one thing that must not die. A component that cannot do
  its job returns an honest "not configured" / "unavailable" string rather than
  raising.
- **Least privilege, enforced server-side.** Remote origins (phone, Discord) get
  a *restricted tool surface*: no shell, no system control, no filesystem, no
  code execution, no self-update. This is enforced at **two gates** — the tool
  list is filtered before it is offered to the model, *and* the executor
  re-checks the deny list by name. It is never prompt-only. Re-opening a single
  tool for a single origin is a surgical per-origin allowance, not a boundary
  flip. The guiding line: *Jarvis, not Ultron.*
- **Confirmation-gate every mutating action.** A destructive or irreversible
  verb (kill a process, restart/stop a service, cycle DHCP, pull code and
  restart) takes a `confirm: bool` parameter; calling it without confirmation
  returns a *description* of what would happen. **The gate must carry the
  information needed to evaluate it** — e.g. self-update previews the actual
  pending commits rather than asking "shall I pull?". A gate the user cannot
  evaluate is procedure, not safety. Privileged verbs additionally require the
  process to be running elevated, which is an opt-in tray action, never the
  default.
- **The container is the boundary for arbitrary code.** `run_code` cannot be
  allowlisted (arbitrary code is arbitrary), so it is contained instead:
  ephemeral Podman, `--network=none`, no host mount, CPU/memory/PID caps, hard
  timeout, non-root. Zero host blast radius, therefore no confirmation gate —
  but a full audit trail in the log.
- **Measure before you tune.** Do not adjust a threshold from a symptom. Get the
  numbers first — the project has repeatedly found that the obvious knob was the
  wrong one (an inference model was 17× too slow to ship, so it was replaced
  rather than tuned; a detection threshold was the weak lever where an amplitude
  floor was the real discriminator; a memory leak was attributed to the ML model
  for two milestones before a harness proved it was the video-capture handle).
  Correlate a score with its *input*, never tune on scores alone.
- **Cooperative gates have two sides.** Heavy background CPU work (vision,
  acoustic inference) yields while audio is playing, via a shared counted event.
  A gate is only as complete as (a) every *consumer* that opts into yielding and
  (b) every *producer* — every code path that speaks — that raises it. Both
  halves have been the source of a regression; both are now tested.
- **Cost discipline.** Sonnet by default, Haiku for background jobs, prompt
  caching on the system prompt. Escalate deliberately, not reflexively.
- **Latency over cleverness.** Short replies. This is voice.
- **Rule of three before extracting.** Two call sites is a coincidence; three is
  a pattern. Shared helpers (`http_util`, `atomic_io`, `gates`) were each
  extracted on the third consumer, not speculatively.
- **New durable state uses `src/atomic_io.py`.** fsync-before-replace, unique
  temp names, Windows `os.replace` retry. The target machine has no UPS, so an
  unclean power loss is a realistic failure mode, not a theoretical one.
- **A bug that a test would have caught earns a test.** The regression gate grew
  from a handful of suites to 45 exactly this way.
- **QoL consolidation cadence.** Periodically run a no-new-features hardening
  pass: verify the regression net is green, run read-only audits across the
  tree, then fix in *risk order* (correctness → latent → cosmetic), gating each
  fix. Trigger on any of: (a) ~20 milestones since the last pass, (b) a
  regression recurs that a test would have caught, or (c) a core file crosses a
  god-file threshold (a 1,500-line `main()` is a signal regardless of milestone
  count). The count is the backstop; (b) and (c) are the real triggers — debt
  accrues with coupling events, not with the calendar. Template and findings in
  [`docs/CODE_AUDIT.md`](docs/CODE_AUDIT.md).
- **Close-out procedure.** Run the regression gate → sync this `CLAUDE.md` and
  `docs/MILESTONES.md` → commit and push. Mark deferred items done *in this
  file*, not only in a detail log: this file is the only state a context reset
  reloads.

## Personality / system prompt notes
The persona:
- British-butler tone: courteous, dryly witty, concise.
- Address the user as "sir" *sparingly* — not every sentence.
- Short replies by default. This is voice; long replies are tedious to listen
  to. The pattern is headline first, then offer the longer version — which also
  feeds naturally into the follow-up window.
- Short sentences (better TTS prosody). Never apologize unnecessarily, never
  over-explain.
- Closer to the films' understated J.A.R.V.I.S. than a parody.

Engineer mode (a tray toggle) overrides the brevity rule and enables adaptive
thinking, for when the answer genuinely needs structured depth.

**Note on model migrations:** verify *verbosity*, not just tool routing. A model
swap can leave routing perfect while default reply length triples — which, on a
voice interface, is a regression measured in minutes of unwanted speech. Probe
both.

## Multilingual support (English + Spanish)
The architecture supports this with very little added code, and it is a genuine
requirement: one intended user is not an English speaker.

- **Auto-detect per turn.** faster-whisper returns `detected_language` with each
  transcript. That threads into the LLM call (respond in the same language) and
  into TTS (pick a matching voice).
- **Voice mapping** (`VOICE_BY_LANG` in `src/text_to_speech.py`):
  ```python
  VOICE_BY_LANG = {
      'en': 'en-US-GuyNeural',      # calm, British-tinged male
      'es': 'es-MX-JorgeNeural',    # formal male, butler-like
  }
  ```
- **System prompt addendum**: respond in the language spoken; use the formal
  *usted* form in Spanish; match Latin American Spanish conventions.
- **Interpreter mode** is the recombination payoff: "be my interpreter" stops
  Jarvis answering and makes him *relay* — each utterance translated into the
  other language of the configured pair and spoken in that language's voice,
  continuously, with no wake word between turns, until "stop interpreting".
  Built from the per-turn language detection and voice map that already existed.
- **Wake-word caveat**: the `hey_jarvis` model is English-trained, so a
  Spanish-accented "Jarvis" (soft J) triggers less reliably. Mitigations, in
  order of preference: lower the confidence threshold; train a custom wake word;
  fall back to a more phonetically robust word.

## Current Status
The project is feature-complete for its intended use and running in production
as a supervised always-on process. ~88 milestones; the regression gate is at 45
suites and green.

**Working:**
- The core loop: wake word → local STT (EN/ES auto-detect) → streaming,
  prompt-cached, agentic Claude call → streaming TTS, with a tray icon, a
  console window with live transcript and waveform, and an optional ambient
  overlay.
- **Conversational**: a follow-up window after each reply (no wake word needed);
  a persistent hands-free conversation mode; barge-in (interrupt mid-reply by
  saying the wake word).
- **~40 tools** across web, data, personal knowledge, memory recall, reminders,
  diagnostics, and gated system actions. See "The tool layer" above.
- **Memory**: in-process turn history + a JSONL session store, summarized at
  session boundaries and recalled into the system prompt; a separate curated
  knowledge base (hybrid keyword + embedding retrieval); full-text/semantic
  search over verbatim past conversations.
- **Senses**: webcam (vision security, on-demand snapshots), screen capture,
  acoustic classification, per-turn speaker identification.
- **Proactive**: homelab monitoring, calendar pre-event announces, severe-weather
  alerts, scheduled briefings, an anticipation layer, and a quiet-hours policy
  that lets important announcements pierce while routine ones defer.
- **Clients**: local voice, console, a token-gated phone PWA (type / push-to-talk
  / reply audio / state), and a Discord bridge — the last two on a restricted
  tool surface.
- **Reliability**: crash watchdog, mic-session supervisor, memory watchdog,
  atomic durable writes, self-update behind a confirmation gate.

**Known unknowns / risks:**
- **Wake-word false positives.** Background speech or media saying "Jarvis" can
  trigger a capture. The confidence threshold is the knob; the opt-in speaker
  gate (answer only enrolled voices) is the stronger mitigation.
- **TTS prosody.** Edge is good, not perfect. Voice choice and pacing are the
  levers.
- **Audio device selection.** Multi-device machines need `JARVIS_MIC_DEVICE`
  pinned; note that budget microphones often enumerate under a generic OEM
  string, so identify the device by unplug-diff rather than by guessing a name.
- **Fully hands-free talk-over** (interrupting without the wake word) was built
  and shelved: once the assistant's own echo is cancelled, an energy gate fires
  on *any* residual sound, because it detects the presence of sound, not the
  intent to interrupt. The code is kept behind a default-off flag for a
  headset/quiet-room scenario. Wake-word barge-in is the robust UX.

## Possible future work
Neutral backlog; nothing here is committed. The standing discipline is
**iterate on real usage** — do not pre-build.

- Multi-camera / RTSP support (OpenCV already supports it; needs a camera
  registry and a `camera` parameter).
- Additional MCP servers through the existing bridge (home automation, 3D
  printing) — same async↔sync wrapper and voice allowlist.
- Growing the knowledge corpus. The hybrid retrieval is correct but the corpus is
  currently too small to demonstrate an aggregate win; size, not the algorithm,
  is the limiter.
- Finer-grained per-verb tuning in `pc_shell` / `system_control` as real use
  cases prove out.
- A custom multilingual wake-word model for non-English accents.
- Extracting the text/voice intent dispatch (a real hot-path refactor; wants a
  test at the right level first).
