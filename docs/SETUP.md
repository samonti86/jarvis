# Setting up Jarvis

From a fresh clone to a talking assistant. The core loop needs **one API key**
and about ten minutes; everything else is optional and can be added later
without restarting from scratch.

- **What each setting does** → [ENV_VARS.md](ENV_VARS.md)
- **Why it's built this way** → [MILESTONES.md](MILESTONES.md)
- **Is my setup correct?** → `python scripts/doctor.py`

---

## 1. Prerequisites

| | |
|---|---|
| **OS** | Windows 10 or 11. Jarvis is Windows-native by design — WASAPI capture, SAPI fallback TTS, the tray and overlay layer. WSL2 audio bridging is not reliable enough for always-on listening, so it is not supported. |
| **Python** | 3.12 or newer, from [python.org](https://www.python.org/downloads/). Tick **"Add python.exe to PATH"** in the installer. |
| **Hardware** | A microphone and a speaker. A dedicated USB mic is worth it — see [step 6](#6-pin-the-right-microphone). |
| **Disk** | ~3 GB. Most of it is PyTorch, pulled in by the vision and audio-classification features. |

Optional, only if you want the matching feature:

- **[Podman](https://podman.io/)** — sandbox for the `run_code` tool. Without it, that tool disables itself.
- **[Git](https://git-scm.com/)** — required for the self-update tool.
- **Visual Studio Build Tools (C++)** — only if you enable the enrolled-face auth path, which needs `dlib`.

---

## 2. Install

```powershell
git clone https://github.com/samonti86/jarvis.git
cd jarvis

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> If PowerShell blocks the activate script:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**One known post-install fix.** `resemblyzer` declares the legacy `typing`
backport as a dependency. On Python 3.12 that package *shadows* the standard
library and breaks imports. Remove it:

```powershell
pip uninstall -y typing
```

---

## 3. Add your API key

```powershell
Copy-Item .env.example .env
notepad .env
```

Get a key at **<https://console.anthropic.com/settings/keys>** and set:

```ini
ANTHROPIC_API_KEY=sk-ant-your-real-key-here
```

This is the only required value. Every other integration is optional and
reports "not configured" rather than failing.

> `.env` is gitignored. Never commit it.

---

## 4. Check your setup

```powershell
python scripts\doctor.py
```

The doctor is read-only — no network calls, no audio devices opened, no secret
values printed. It reports what is installed, what is configured, what hardware
it can see, and exactly what is blocking startup.

Fix any `[ FAIL ]` lines, then continue.

---

## 5. First run

```powershell
python main.py
```

First launch downloads the wake-word and Whisper models (~1 GB, once). When the
tray icon settles, say:

> **"Hey Jarvis — what's the weather?"**

You do not need the wake word for follow-ups: there's a short window after each
reply where you can just keep talking.

---

## 6. Pin the right microphone

Multi-device machines route audio unpredictably, and a USB re-enumeration can
silently move Jarvis to the wrong input. Pin it:

```ini
JARVIS_MIC_DEVICE=MC1000     # name substring (case-insensitive) or index
```

**Identify the device by unplug-diff, not by guessing the name.** Budget mics
often enumerate under a generic OEM string — a MOVO MC1000 shows up as
"PDP Audio Device". Run `python scripts\doctor.py --verbose` with the mic
plugged in, then again unplugged, and compare.

A name matching nothing logs a warning and falls back to the default mic; it
never fails to capture.

---

## 7. Optional integrations

Add only what you want. Each is independent and degrades cleanly.

| Feature | Setting | Where to get it |
|---|---|---|
| Weather, briefing, storm alerts | `JARVIS_HOME_LOCATION` | Just a location: `Austin, TX` or a ZIP. No key. NWS alerts are US-only. |
| Films & TV | `TMDB_API_KEY` | [themoviedb.org](https://www.themoviedb.org/signup) → Settings → API. Use the **v3** key (short), not the v4 bearer token. Free. |
| Video games | `RAWG_API_KEY` | [rawg.io/apidocs](https://rawg.io/apidocs). Free, 20k req/month. |
| Computation | `WOLFRAM_APP_ID` | [developer.wolframalpha.com](https://developer.wolframalpha.com/) → create an app → copy the AppID. Free tier: 2,000 calls/month. |
| News | *(none)* | No key. Feed list lives in `src/news.py` — edit it to choose your sources. |
| Calendar | `OUTLOOK_ICAL_URL` | Outlook.com → Calendar → Share → **Publish** → copy the ICS link. Read-only; Jarvis cannot modify your calendar. |
| Plex | `PLEX_URL`, `PLEX_TOKEN` | [Finding your Plex token](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/) |

### Push notifications

**Discord webhook** (one-way — security events, reminders):
Server Settings → Integrations → Webhooks → New Webhook → Copy URL →
`DISCORD_WEBHOOK_URL`.

**Discord bot** (two-way — chat with Jarvis from a channel), ~10 minutes:

1. [discord.com/developers/applications](https://discord.com/developers/applications) → **New Application** → Bot
2. Bot tab → **Reset Token** → copy → `DISCORD_BOT_TOKEN` *(shown once — guard it)*
3. Bot tab → enable **MESSAGE CONTENT INTENT** → Save
   *(without this the bot connects but sees empty message text)*
4. OAuth2 → URL Generator → scope `bot`, permissions *View Channels*,
   *Send Messages*, *Read Message History* → open the URL → add to your server
5. Discord app → Settings → Advanced → **Developer Mode** on. Right-click the
   channel → Copy Channel ID → `DISCORD_CHANNEL_ID`. Right-click each permitted
   person → Copy User ID → `DISCORD_ALLOWED_USER_IDS` (comma-separated)

The Discord bridge runs on a **restricted tool surface** — no shell, no
filesystem, no code execution, no self-update — enforced server-side at two
independent gates, not by prompt instruction. The allowlist is fail-closed:
an empty `DISCORD_ALLOWED_USER_IDS` means nobody.

**Email** — `SMTP_USERNAME` / `SMTP_PASSWORD` / `SMTP_TO`. For Gmail use a
16-character [app password](https://myaccount.google.com/apppasswords), not
your account password.

### Phone console (optional)

A token-gated PWA: type to Jarvis, push-to-talk, hear replies, see his state.

```ini
JARVIS_REMOTE_TOKEN=a-long-random-string
```

Blank token = server off. **Do not port-forward this.** Use
[Tailscale](https://tailscale.com/) for outside-the-LAN access. Browser
microphone access requires a secure context, so push-to-talk needs TLS — point
`JARVIS_TLS_CERT_FILE` / `JARVIS_TLS_KEY_FILE` at a `tailscale cert` pair.

---

## 8. Run it for real

```powershell
# Silent — no console window
pythonw jarvis.pyw

# Supervised — respawns on crash (recommended)
pythonw jarvis_watchdog.pyw
```

Logs go to `%LOCALAPPDATA%\Jarvis\jarvis.log`.

To start at login, put a shortcut to `jarvis_watchdog.pyw` in:
`shell:startup` (paste into the Run dialog).

> Use a **logon** trigger, not a startup/boot trigger, if you schedule it via
> Task Scheduler. Session 0 has no GPU access, so CUDA silently fails there.

---

## 9. Verify a change

```powershell
venv\Scripts\python.exe scripts\run_all_tests.py
```

46 gates: syntax, module wiring, a structural JS check on the PWA, a config-doc
drift gate, and 42 test suites. Must exit 0 before anything ships.

---

## Troubleshooting

**Jarvis doesn't respond to the wake word.**
Lower `WAKE_WORD_THRESHOLD` (default `0.5`; try `0.4`). Confirm the right mic is
pinned. The `hey_jarvis` model is English-trained, so a Spanish-accented
"Jarvis" (soft J) triggers less reliably — lowering the threshold is the first
lever.

**It triggers on the TV.**
Raise `WAKE_WORD_THRESHOLD`, or enrol your voice and set `JARVIS_SPEAKER_GATE=1`
so only enrolled voices are answered. The typed path is never gated, so you
cannot lock yourself out.

**No audio at all after switching inputs on a KVM.**
A KVM switch can drop the mic entirely. The mic-session supervisor recovers most
cases; if not, restart Jarvis.

**Speech-to-text is slow.**
Local CPU transcription runs 5–10 s. Either drop `WHISPER_MODEL` to `base`, or
offload to a CUDA box on the LAN with `stt_server/` and set `STT_SERVER_URL`
(1.5–2.5 s, falls back to local on any failure).

**Replies stutter while security mode is armed.**
Lower `JARVIS_YOLO_THREADS` and `JARVIS_ACOUSTIC_THREADS` to `1`. Both cap
torch's process-global thread pool, which otherwise starves the real-time audio
threads.

**`ModuleNotFoundError: typing`, or odd import errors.**
Run `pip uninstall -y typing` — see [step 2](#2-install).

**Something else.**
Run `python scripts\doctor.py` first; it names the blocker. Then check
`%LOCALAPPDATA%\Jarvis\jarvis.log` — note that non-UTF-8 bytes can trip `grep`
into binary mode, so use `grep -a`.
