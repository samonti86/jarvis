"""TTS: edge-tts (primary, online) with pyttsx3 (offline) fallback.

Two playback paths:

1. **speak(text, language)** — Tier A buffered. Synthesizes the full text into
   one audio clip and plays it. Used for short utility text or as a fallback.

2. **speak_streaming(text_iter, language, on_first_audio)** — Tier B pipelined.
   Producer-consumer: text producer extracts sentences from the LLM stream,
   synth worker calls edge-tts and decodes MP3 ahead of playback, audio queue
   (maxsize=2) buffers one sentence so playback is gapless even though each
   edge-tts call is its own HTTP round-trip (~300-500ms).

   Why this works where naive per-sentence Tier B failed: with the lookahead
   buffer, sentence N+1 is synthesized *during* sentence N's playback, so the
   inter-sentence gap collapses to sounddevice's startup latency (~50ms)
   instead of the full synth round-trip.
"""

from __future__ import annotations

import asyncio
import queue
import re
import sys
import threading
from typing import Callable, Iterable

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

# Sentence boundary: terminal punctuation followed by whitespace/EOS, or any newline.
# Note: false positives on abbreviations ("Mr. Smith") are acceptable — they just
# yield slightly shorter chunks, which still synth and play fine.
_SENTENCE_RE = re.compile(r"[.!?](?:\s+|$)|\n+")


def speak(text: str, language: str = "en") -> None:
    """Tier A: synthesize full text in one shot and play. Edge-tts → pyttsx3 fallback."""
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


def speak_streaming(
    text_iter: Iterable[str],
    language: str = "en",
    on_first_audio: Callable[[], None] | None = None,
) -> None:
    """Tier B: pipelined synth+playback as text chunks arrive from the LLM stream.

    Iterates `text_iter`, splits on sentence boundaries, synths each sentence
    on a worker thread, and plays in the caller's thread. Playback gap between
    sentences is the audio device startup time, not the edge-tts round trip,
    because the synth worker stays one sentence ahead via a maxsize=2 audio
    queue.

    `on_first_audio` (optional) fires once, when the first audio sample is
    about to play — caller uses this to flip tray state THINKING → SPEAKING.

    On any exception in the text producer (e.g., LLM stream raising), this
    function re-raises it after threads drain, so the caller can roll back
    conversation history. Per-sentence synth failures are logged and skipped.
    """
    voice = VOICE_BY_LANG.get(language, DEFAULT_VOICE)

    sentence_q: queue.Queue[str | None] = queue.Queue()
    audio_q: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue(maxsize=2)
    producer_errors: list[BaseException] = []

    def text_producer() -> None:
        """Iterate text chunks, emit complete sentences to sentence_q."""
        try:
            buffer = ""
            for chunk in text_iter:
                buffer += chunk
                while True:
                    m = _SENTENCE_RE.search(buffer)
                    if not m:
                        break
                    sentence = buffer[: m.end()].strip()
                    buffer = buffer[m.end():]
                    if sentence:
                        sentence_q.put(sentence)
            if buffer.strip():
                sentence_q.put(buffer.strip())
        except BaseException as exc:
            producer_errors.append(exc)
        finally:
            sentence_q.put(None)  # signal synth worker we're done

    def synth_worker() -> None:
        """Pull sentences, synth via edge-tts, decode, push to audio_q."""
        try:
            while True:
                sentence = sentence_q.get()
                if sentence is None:
                    return
                try:
                    mp3_bytes = asyncio.run(_fetch_mp3(sentence, voice))
                    decoded = miniaudio.mp3_read_s16(mp3_bytes)
                    samples = np.frombuffer(decoded.samples.tobytes(), dtype=np.int16)
                    if decoded.nchannels > 1:
                        samples = samples.reshape(-1, decoded.nchannels)
                    audio_q.put((samples, decoded.sample_rate))
                except Exception as exc:
                    # Drop this sentence, keep going — better than aborting playback.
                    print(f"[tts] synth failed for sentence ({len(sentence)} chars): {exc}", file=sys.stderr)
        finally:
            audio_q.put(None)  # always signal end so playback unblocks

    t_prod = threading.Thread(target=text_producer, daemon=True)
    t_synth = threading.Thread(target=synth_worker, daemon=True)
    t_prod.start()
    t_synth.start()

    first = True
    while True:
        item = audio_q.get()
        if item is None:
            break
        samples, sample_rate = item
        if first:
            first = False
            if on_first_audio is not None:
                try:
                    on_first_audio()
                except Exception:
                    pass  # never let a UI callback break playback
        sd.play(samples, samplerate=sample_rate, blocking=True)

    t_prod.join()
    t_synth.join()

    if producer_errors:
        raise producer_errors[0]


def _speak_edge_tts(text: str, voice: str) -> None:
    mp3_bytes = asyncio.run(_fetch_mp3(text, voice))
    decoded = miniaudio.mp3_read_s16(mp3_bytes)

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
