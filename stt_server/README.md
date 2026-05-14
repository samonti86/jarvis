# Jarvis STT GPU Offload Server (M36)

FastAPI server that runs `faster-whisper` on the laptop's GPU.
The Jarvis desktop client POSTs WAV audio here for ~10-20× faster
transcription than the desktop's CPU path.

**Latency** (measured: GTX 1650, "small" model, float16, 2-3 sec audio):
- Desktop CPU local (Ryzen 5 3400G): 5-10 s per transcription
- This server: ~250-500 ms per transcription (incl. ~10 ms LAN RTT)

## Prerequisites

- Windows machine with an NVIDIA GPU (Pascal architecture or newer — GTX 10-series, GTX 16-series, RTX 20/30/40-series).
- NVIDIA driver 525+ (verify with `nvidia-smi` — current driver should be far newer).
- Python 3.10+ (Jarvis tested with 3.13 on the laptop; the desktop client runs on 3.12).

**No CUDA Toolkit (`nvcc`) install needed.** The four `nvidia-*-cu12` pip
packages in `requirements.txt` ship the CUDA runtime DLLs (cuBLAS,
cuDNN, NVRTC, cudart). `server.py` adds their `bin/` directories to
the Windows DLL search path AND `PATH` at module load — both are
required (see comments in `server.py`).

## Install

From PowerShell as the user that will run the service:

```powershell
# Clone the repo if it isn't already here (it may be — the M28 diagnostics
# bundle lived here too).
cd ~\repos
git clone https://github.com/samonti86/jarvis.git
cd jarvis\stt_server

# Separate venv from the main Jarvis venv (different deps).
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Pre-warm the model

The first request triggers a ~700 MB download of the "small" Whisper model
from Hugging Face. Pre-warm to avoid that delay on the first real call:

```powershell
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda', compute_type='float16')"
```

The model caches to `%USERPROFILE%\.cache\huggingface\hub\` — survives venv recreations.

## Run (manual)

```powershell
.\venv\Scripts\Activate.ps1
uvicorn server:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` binds all interfaces so the desktop can reach the laptop
over the LAN. `--host 127.0.0.1` would be loopback-only (fine for testing
on the same machine).

## Quick smoke test

```powershell
# Health check (no audio):
curl http://192.168.1.10:8000/health

# Transcribe a WAV file:
curl -F "audio=@test.wav" -F "language=en" http://192.168.1.10:8000/transcribe
```

## Autostart on boot (Task Scheduler, hidden via VBS wrapper)

The simplest reliable autostart on Windows that doesn't need admin or
extra tooling. The repo ships:
- `run_server.bat` — `cd`s to its own dir, launches uvicorn, redirects
  output to `server.log`.
- `run_server_hidden.vbs` — invokes the .bat with WindowStyle=Hidden so
  no cmd.exe window appears in the interactive user session. Without
  this wrapper, a visible CMD window appears whenever the task fires
  during an active login session, and accidentally closing it kills
  the server.

```powershell
# No admin needed — runs under your own user. /sc onlogon fires when
# YOU log in (interactive session, full GPU access). DO NOT use
# /sc onstart — it fires before login in Session 0 (Services), where
# python + CUDA fail to bind ports / initialize properly. Symptom of
# the Session 0 failure: zombie python.exe in tasklist with no
# LISTENING on 8000 — discovered after first M36 reboot test.
schtasks /create /tn JarvisSTTServer /tr "wscript.exe C:\Users\youruser\repos\jarvis\stt_server\run_server_hidden.vbs" /sc onlogon /ru youruser /f

# Verify:
schtasks /query /tn JarvisSTTServer

# Fire it immediately (use after a manual taskkill or anytime you need
# to restart the server without rebooting):
schtasks /run /tn JarvisSTTServer

# Stop the running instance (the server is just python.exe instances
# — taskkill them all and the schtasks /run brings it back fresh):
taskkill /F /IM python.exe
```

After a reboot, the server is up within ~15s of YOUR login (model-load
time). If the laptop auto-logs-you-in (typical for Plex hosts), it's
~15s after boot completes; if you log in manually, ~15s after that.

### Manual restart from the desktop (no RDP needed)

```powershell
# From the Jarvis desktop's PowerShell — assumes SSH key auth is set up
# (which it is, per M24):
ssh youruser@192.168.1.10 "taskkill /F /IM python.exe & schtasks /run /tn JarvisSTTServer"
```

### Alternative: NSSM service

If you want it to start before user login (true service), use NSSM.
This is what `PlexService` uses on this host. Requires admin install.

```powershell
winget install nssm

# Run as Administrator:
nssm install JarvisSTTServer C:\Users\youruser\repos\jarvis\stt_server\venv\Scripts\python.exe
nssm set    JarvisSTTServer AppParameters "-m uvicorn server:app --host 0.0.0.0 --port 8000"
nssm set    JarvisSTTServer AppDirectory  C:\Users\youruser\repos\jarvis\stt_server
nssm set    JarvisSTTServer AppStdout     C:\Users\youruser\repos\jarvis\stt_server\service.log
nssm set    JarvisSTTServer AppStderr     C:\Users\youruser\repos\jarvis\stt_server\service.log
nssm set    JarvisSTTServer AppRotateFiles 1
nssm set    JarvisSTTServer AppRotateBytes 10485760
nssm set    JarvisSTTServer Start         SERVICE_AUTO_START
nssm start JarvisSTTServer
```

## Firewall

Windows Defender Firewall blocks inbound port 8000 by default.

```powershell
# Use netsh — works without admin if you're a local administrator.
# DO NOT use New-NetFirewallRule on hosts with broken WMI/CIM
# StandardCimv2 provider (e.g. MEDIA-HOST) — it errors with
# "Invalid class". netsh is the WMI-free path.
netsh advfirewall firewall add rule name="Jarvis STT Server" dir=in action=allow protocol=TCP localport=8000 profile=any
```

`profile=any` covers Private + Public + Domain. If your LAN is correctly
classified as Private, you can use `profile=private` for tighter scoping.

## Operations

**Logs**: `service.log` next to `server.py` (rotated at 10 MB by NSSM). Contains uvicorn access logs + the model-load lines + any inference errors.

**Restart**: `nssm restart JarvisSTTServer` from any PowerShell.

**Update**: `git pull` in `~/repos/jarvis`, then `nssm restart JarvisSTTServer`. The model lives in the user cache, not the repo, so updates don't re-download.

**Concurrency**: a single `WhisperModel` instance serves all requests under
a lock — concurrent transcriptions queue, they don't crash. Single-user
system, that's fine.

**Plex contention**: the GTX 1650 has 4 GB VRAM. "small" Whisper uses ~700
MB at float16. Plex hardware transcoding (NVENC/NVDEC) uses ~200-400 MB.
Comfortable co-existence; verify with `nvidia-smi` if you're heavily
streaming during a security-mode event.

## Tuning

Set these via env vars in the NSSM service or your shell before running:

| Var | Default | Notes |
|---|---|---|
| `STT_MODEL` | `small` | Try `tiny` or `base` for less accuracy but ~3-5x faster; `medium` for better accuracy at ~2x slower + ~3 GB VRAM. |
| `STT_DEVICE` | `cuda` | Set `cpu` to force CPU on the server (useful for testing the auto-fallback path). |
| `STT_COMPUTE_TYPE` | `float16` | `int8_float16` is faster but slightly less accurate; `float32` is more accurate but uses ~2x VRAM. |
