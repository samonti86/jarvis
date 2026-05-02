# Jarvis

A Windows desktop voice assistant. Wake-word activated, locally transcribed,
Claude-powered, spoken back through your speakers.

Inspired by Tony Stark's J.A.R.V.I.S.: courteous, dryly witty, concise.

## Architecture

```
mic ──► openWakeWord ──► faster-whisper ──► Claude API ──► edge-tts ──► speakers
        (local)          (local)            (network)     (network)
```

- **Wake word + STT run locally** — your spoken question never leaves the machine.
- **Only the transcribed text** is sent to the Claude API.
- Auto-detects English vs. Spanish per turn; replies in the same language.

## Quick start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env  # then edit .env with your Anthropic API key
python main.py
```

Then say **"Hey Jarvis"** and ask something.

## Status

Under active development — see `CLAUDE.md` for the build plan and current milestones.
