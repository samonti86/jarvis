"""Jarvis STT GPU offload server (M36).

A FastAPI app that runs faster-whisper on the laptop's GPU and exposes a
`/transcribe` endpoint. The Jarvis desktop client POSTs WAV audio here
for ~10-20x faster transcription than the local CPU path on the
Ryzen 5 3400G.

Why this lives in `stt_server/` as a separate package inside the main
jarvis repo (not a separate repo):
- Single source of truth — server + client evolve in lockstep via git.
- Independent venv on the server host so different deps (faster-whisper,
  fastapi) don't pollute the desktop's environment.
- Lightweight — ~150 LOC + requirements + README.

Why `faster-whisper` (not openai-whisper or whisper.cpp):
- Faster-whisper wraps ctranslate2 which has first-class CUDA support
  via pre-built wheels — no need to install the CUDA Toolkit on the
  server host, just the NVIDIA driver. ctranslate2's wheels bundle
  cuDNN 9 runtime.
- Same model files (transcripts match Jarvis's existing local STT
  output bit-for-bit when fed the same audio + decode params).
- 4-5x faster than openai-whisper at the same model size.

Concurrency model:
- FastAPI endpoint is `async def` so `await upload.read()` is non-blocking.
- The actual `model.transcribe()` call is CPU/GPU-bound and runs in
  `loop.run_in_executor(None, ...)` — keeps the event loop responsive
  for /health probes while inference runs.
- `_model_lock` serializes inference because WhisperModel isn't thread-
  safe for concurrent calls. Single-user system, serial is fine.

WAV-only input:
- Client sends WAV (16 kHz mono int16) — same format Jarvis's
  AudioSession captures. WAV header carries sample rate / channels /
  bit depth so we don't need separate metadata fields.
- Decoded to float32 numpy via stdlib `wave` (no tempfile, no disk).
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import threading
import time
import wave

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel


# --- Config (env-driven) ---------------------------------------------------
# All knobs default to sensible values; override via env on the systemd /
# NSSM unit if needed for testing.
MODEL_NAME = os.getenv("STT_MODEL", "small")
DEVICE = os.getenv("STT_DEVICE", "cuda")
# float16 on CUDA is the standard accuracy/speed tradeoff — same quality
# as float32 for inference, ~2x faster, half the VRAM. int8_float16 is
# faster still but loses noticeable accuracy on short voice commands.
COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "float16")


# --- Model loading (eager, at module import) -------------------------------
# Load synchronously at module load. uvicorn imports this module BEFORE
# starting the event loop, so the model is ready before the first request
# arrives — no first-request latency hit. Cold start: ~5-15s for "small"
# model loading + GPU warmup. Subsequent process restarts are faster
# (model file is already in disk cache).
print(f"[stt-server] loading whisper model '{MODEL_NAME}' on {DEVICE} ({COMPUTE_TYPE})...",
      file=sys.stderr, flush=True)
_t0 = time.monotonic()
try:
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
except Exception as exc:
    # Common causes: no GPU, missing cuDNN runtime, model download failure.
    # Log clearly + raise so the service supervisor (NSSM) restarts and
    # surfaces the error in event log instead of running blind.
    print(f"[stt-server] FATAL: model load failed: {type(exc).__name__}: {exc}",
          file=sys.stderr, flush=True)
    raise
print(f"[stt-server] model loaded in {time.monotonic() - _t0:.1f}s",
      file=sys.stderr, flush=True)

# WhisperModel is NOT thread-safe for concurrent inference. Serialize via
# a lock. The async endpoint dispatches inference to the default thread
# pool, where this lock blocks if a previous request is still running.
_model_lock = threading.Lock()


# --- FastAPI app -----------------------------------------------------------
app = FastAPI(
    title="Jarvis STT GPU Server",
    description="POST /transcribe with a WAV file → fast GPU transcription.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    """Liveness + config probe. Client can hit this to confirm the server
    is up and which model/device it's running (useful when debugging
    fallback behavior — 'is the server actually running cuda?')."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
) -> dict:
    """Transcribe a WAV upload. `language` is an optional ISO-639-1 hint
    ('en', 'es'); omit for auto-detect (slightly slower, ~100ms extra).

    Returns:
        {"text": "...", "language": "en", "duration_ms": 320}

    Errors:
        400 — malformed WAV (wrong sample rate, bad header)
        500 — inference failure (uncommon; usually a transient GPU issue)
    """
    started = time.monotonic()
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    try:
        samples = _decode_wav_bytes(audio_bytes)
    except (ValueError, wave.Error) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid WAV: {exc}")

    # Hand off CPU/GPU-bound inference to the thread pool so the event
    # loop stays free for /health pings and other concurrent requests
    # (which will queue on _model_lock but at least won't block uvicorn).
    loop = asyncio.get_running_loop()
    text, detected_lang = await loop.run_in_executor(
        None, _transcribe_sync, samples, language
    )

    return {
        "text": text,
        "language": detected_lang,
        "duration_ms": int((time.monotonic() - started) * 1000),
    }


# --- Helpers ---------------------------------------------------------------

def _decode_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    """Decode a WAV byte string into a float32 mono numpy array at 16kHz.

    Refuses anything not matching Jarvis's capture format (16kHz, int16)
    so we fail fast on mis-configured clients rather than silently
    re-sampling. Multi-channel inputs are downmixed by averaging.
    """
    with io.BytesIO(wav_bytes) as bio:
        with wave.open(bio, "rb") as wav:
            sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            n_frames = wav.getnframes()
            raw = wav.readframes(n_frames)

    if sample_rate != 16000:
        raise ValueError(f"expected 16kHz, got {sample_rate}Hz")
    if sample_width != 2:
        raise ValueError(f"expected 16-bit, got {sample_width * 8}-bit")

    audio = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    # faster-whisper wants float32 normalized to [-1, 1].
    return audio.astype(np.float32) / 32768.0


def _transcribe_sync(samples: np.ndarray, language: str | None) -> tuple[str, str]:
    """Run faster-whisper inference under the model lock. Returns
    (text, detected_language). Drops empty segments and joins with spaces."""
    with _model_lock:
        segments, info = model.transcribe(
            samples,
            language=language,  # None → auto-detect
            # beam_size=5 is the faster-whisper default; tunable if accuracy
            # matters more than speed. For short voice commands the default
            # is the right tradeoff.
            beam_size=5,
            # VAD pre-filtering reduces hallucinations on silent regions
            # at trivial cost. Keep it on.
            vad_filter=True,
        )
        # transcribe() returns a generator — fully consume before releasing
        # the lock so the next request doesn't preempt.
        text = " ".join(s.text for s in segments).strip()
    return text, info.language
