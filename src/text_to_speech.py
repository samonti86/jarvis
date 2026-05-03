"""TTS: edge-tts (primary, online) with pyttsx3 (offline) fallback.

Buffered (Tier A): synthesizes the full text into a single audio clip and plays
it through the default speakers, blocking until playback completes. We tried a
sentence-streaming variant (Tier B) and reverted: the per-sentence edge-tts
round trip (~300-500ms each) made multi-sentence replies feel choppier than
the single-shot Tier A playback, and the perceived first-word win didn't
materialize in real testing.
"""

from __future__ import annotations

import asyncio
import sys

import edge_tts
import miniaudio
import numpy as np
import pyttsx3
import sounddevice as sd

VOICE_BY_LANG = {
    "en": "en-GB-RyanNeural",   # calm British male — Paul Bettany-adjacent
    "es": "es-MX-JorgeNeural",  # formal Mexican male — butler register
}
DEFAULT_VOICE = "en-GB-RyanNeural"


def speak(text: str, language: str = "en") -> None:
    """Speak `text` aloud. Tries edge-tts; falls back to pyttsx3 on failure."""
    text = text.strip()
    if not text:
        return

    voice = VOICE_BY_LANG.get(language, DEFAULT_VOICE)

    try:
        _speak_edge_tts(text, voice)
        return
    except Exception as exc:
        print(f"[tts] edge-tts failed ({exc}); falling back to pyttsx3", file=sys.stderr)

    try:
        _speak_pyttsx3(text)
    except Exception as exc:
        print(f"[tts] pyttsx3 also failed ({exc}); audio dropped", file=sys.stderr)


def _speak_edge_tts(text: str, voice: str) -> None:
    mp3_bytes = asyncio.run(_fetch_mp3(text, voice))
    decoded = miniaudio.mp3_read_s16(mp3_bytes)

    # decoded.samples is array.array of int16, interleaved if stereo.
    samples = np.frombuffer(decoded.samples.tobytes(), dtype=np.int16)
    if decoded.nchannels > 1:
        samples = samples.reshape(-1, decoded.nchannels)

    sd.play(samples, samplerate=decoded.sample_rate, blocking=True)


async def _fetch_mp3(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def _speak_pyttsx3(text: str) -> None:
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()
