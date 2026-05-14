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

No CUDA Toolkit install needed. `ctranslate2` ships pre-built CUDA wheels with cuDNN 9 bundled.

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

## Autostart via NSSM

[NSSM](https://nssm.cc/) (Non-Sucking Service Manager) is the standard way
to run a Python app as a Windows service. Same pattern used for
`PlexService` on this host.

```powershell
# Install NSSM if it's not already present:
winget install nssm

# Create the service (run as Administrator):
nssm install JarvisSTTServer C:\Users\youruser\repos\jarvis\stt_server\venv\Scripts\python.exe
nssm set    JarvisSTTServer AppParameters "-m uvicorn server:app --host 0.0.0.0 --port 8000"
nssm set    JarvisSTTServer AppDirectory  C:\Users\youruser\repos\jarvis\stt_server
nssm set    JarvisSTTServer DisplayName   "Jarvis STT GPU Server"
nssm set    JarvisSTTServer Description   "Runs faster-whisper on the GPU for the Jarvis desktop client. See stt_server/README.md."
nssm set    JarvisSTTServer Start         SERVICE_AUTO_START
nssm set    JarvisSTTServer AppStdout     C:\Users\youruser\repos\jarvis\stt_server\service.log
nssm set    JarvisSTTServer AppStderr     C:\Users\youruser\repos\jarvis\stt_server\service.log
nssm set    JarvisSTTServer AppRotateFiles 1
nssm set    JarvisSTTServer AppRotateBytes 10485760  # 10 MB

# Start:
nssm start JarvisSTTServer

# Check status:
nssm status JarvisSTTServer
```

## Firewall

Windows Defender Firewall blocks inbound port 8000 by default. Open it for
the LAN only (not the public profile):

```powershell
# Run as Administrator:
New-NetFirewallRule -DisplayName "Jarvis STT Server" `
    -Direction Inbound `
    -LocalPort 8000 `
    -Protocol TCP `
    -Action Allow `
    -Profile Private
```

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
