"""faster-whisper wrapper: state-machine endpointing + transcription."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from src.audio import AudioSession

# Endpointing thresholds (int16 RMS scale).
# Hysteresis: SPEECH_RMS > SILENCE_RMS prevents flapping near a single threshold.
SPEECH_RMS = 1000          # RMS above this = user is speaking
SILENCE_RMS = 700          # RMS below this = silence (raised to clear C920e AGC floor)
MAX_PRE_SPEECH_SEC = 5.0   # give up if user never starts speaking
SILENCE_HANG_SEC = 1.0     # consecutive silence after speech to stop
MAX_RECORDING_SEC = 15.0   # absolute cap


@dataclass
class Transcript:
    text: str
    language: str  # ISO-639-1, e.g. 'en', 'es'


_model: WhisperModel | None = None


def _get_model(name: str) -> WhisperModel:
    global _model
    if _model is None:
        print(f"[stt] loading whisper model '{name}'...", file=sys.stderr)
        _model = WhisperModel(name, device="cpu", compute_type="int8")
        print("[stt] model ready", file=sys.stderr)
    return _model


def _rms(chunk_i16: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk_i16.astype(np.float64) ** 2)))


def _record_until_silence(session: AudioSession) -> np.ndarray:
    """State-machine VAD: wait for speech start, then stop on sustained silence."""
    chunks: list[np.ndarray] = []
    chunks_per_sec = session.sample_rate / session.chunk_samples
    max_chunks = int(MAX_RECORDING_SEC * chunks_per_sec)
    silence_hang_chunks = int(SILENCE_HANG_SEC * chunks_per_sec)
    max_pre_speech_chunks = int(MAX_PRE_SPEECH_SEC * chunks_per_sec)

    speaking = False
    silence_chunks = 0

    print("[stt] listening for question...", file=sys.stderr)
    while len(chunks) < max_chunks:
        chunk = session.read()
        chunks.append(chunk)
        level = _rms(chunk)

        if not speaking:
            if level >= SPEECH_RMS:
                speaking = True
                print(f"[stt] speech started (rms={level:.0f})", file=sys.stderr)
            elif len(chunks) >= max_pre_speech_chunks:
                print("[stt] no speech detected within window", file=sys.stderr)
                return np.array([], dtype=np.int16)
        else:
            if level < SILENCE_RMS:
                silence_chunks += 1
                if silence_chunks >= silence_hang_chunks:
                    break
            else:
                silence_chunks = 0

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
    duration = len(audio) / session.sample_rate
    if len(chunks) >= max_chunks:
        recent_rms = _rms(np.concatenate(chunks[-int(chunks_per_sec):]))
        print(
            f"[stt] hit max duration; recent ambient rms={recent_rms:.0f} "
            f"(silence threshold={SILENCE_RMS}). "
            "Raise SILENCE_RMS in src/speech_to_text.py if this happens often.",
            file=sys.stderr,
        )
    print(f"[stt] captured {duration:.1f}s of audio", file=sys.stderr)
    return audio


def transcribe_after_wake(session: AudioSession, model_name: str = "small") -> Transcript:
    """Record from `session` until silence, then transcribe."""
    model = _get_model(model_name)
    audio_i16 = _record_until_silence(session)

    if audio_i16.size == 0:
        return Transcript(text="", language="")

    audio_f32 = audio_i16.astype(np.float32) / 32768.0
    segments, info = model.transcribe(
        audio_f32,
        beam_size=1,
        vad_filter=True,                  # Silero VAD pre-filter; kills noise hallucinations
        condition_on_previous_text=False,  # avoid Whisper repeating itself across segments
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return Transcript(text=text, language=info.language)
