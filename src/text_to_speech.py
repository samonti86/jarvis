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
import time
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


# --- Markdown stripping (M26 follow-up) ---
# Engineer-mode responses use **bold**, bullets, and headers — great for
# reading on screen, terrible for TTS (edge-tts reads the literal asterisks
# as "asterisk asterisk"). We strip these markers at the audio-path entry
# point only; the visual transcript still shows the original markdown.
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_BOLD_UND = re.compile(r"__(.+?)__", re.DOTALL)
# Single-asterisk italic. The (?<!\w) and (?!\w) lookarounds avoid stripping
# asterisks that are mid-word (rare in English but not impossible in code).
_MD_ITALIC_AST = re.compile(r"(?<!\w)\*([^*\n]+?)\*(?!\w)")
_MD_INLINE_CODE = re.compile(r"`([^`\n]+?)`")
_MD_HEADER = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)


def _strip_markdown_for_tts(text: str) -> str:
    """Remove formatting markers so TTS doesn't read them aloud as punctuation.

    Order matters: pair-replace first (**bold** → bold), THEN the defensive
    orphan-asterisk sweep at the end. The orphan sweep handles a real edge
    case — the sentence-boundary regex can split inside a `**...**` span
    (because `1. ` looks like end-of-sentence), leaving one `**` in each
    half. Pair-replace can't fix half-spans; the orphan sweep does.

    Underscores are NOT stripped wholesale — file paths and identifiers use
    them legitimately ('MAX_TOKENS'). Only paired __bold__ underscores get
    converted. Same defensive instinct for `#` (we strip ONLY at line start
    as headers; mid-line `#` could be a hashtag, channel, or issue ref).
    """
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_BOLD_UND.sub(r"\1", text)
    text = _MD_ITALIC_AST.sub(r"\1", text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_BULLET.sub("", text)
    # Defensive sweep: orphan asterisks/backticks left after sentence-splits
    # inside a bold or code span. Voice context very rarely has legitimate
    # isolated asterisks or backticks.
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"`+", "", text)
    return text


def speak(text: str, language: str = "en") -> None:
    """Tier A: synthesize full text in one shot and play. Edge-tts → pyttsx3 fallback."""
    text = _strip_markdown_for_tts(text.strip())
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
    on_amplitude: Callable[[float], None] | None = None,
) -> None:
    """Tier B: pipelined synth+playback as text chunks arrive from the LLM stream.

    Iterates `text_iter`, splits on sentence boundaries, synths each sentence
    on a worker thread, and plays in the caller's thread. Playback gap between
    sentences is the audio device startup time, not the edge-tts round trip,
    because the synth worker stays one sentence ahead via a maxsize=2 audio
    queue.

    `on_first_audio` (optional) fires once, when the first audio sample is
    about to play — caller uses this to flip tray state THINKING → SPEAKING.

    `on_amplitude` (optional) fires ~30 times per second during playback with
    the current RMS amplitude in [0, 1]. Used by the waveform visualizer.
    Implementation: we pre-compute an envelope per sentence and walk it on a
    daemon ticker thread that runs alongside sd.play(). Audio path itself is
    untouched — the ticker only reads the envelope and emits values.

    On any exception in the text producer (e.g., LLM stream raising), this
    function re-raises it after threads drain, so the caller can roll back
    conversation history. Per-sentence synth failures are logged and skipped.
    """
    voice = VOICE_BY_LANG.get(language, DEFAULT_VOICE)

    sentence_q: queue.Queue[str | None] = queue.Queue()
    audio_q: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue(maxsize=2)
    producer_errors: list[BaseException] = []

    def text_producer() -> None:
        """Iterate text chunks, emit complete sentences to sentence_q.
        Markdown strip happens per-sentence here so the TTS pipeline never
        sees raw `**` / `*` / `` ` `` / leading bullets. Sentences with no
        alphanumeric content (horizontal rules like '---', stray punctuation
        left after stripping) are dropped — edge-tts rejects them and they
        carry no audible content anyway."""
        try:
            buffer = ""
            for chunk in text_iter:
                buffer += chunk
                while True:
                    m = _SENTENCE_RE.search(buffer)
                    if not m:
                        break
                    sentence = _strip_markdown_for_tts(buffer[: m.end()].strip())
                    buffer = buffer[m.end():]
                    if sentence and any(c.isalnum() for c in sentence):
                        sentence_q.put(sentence)
            tail = _strip_markdown_for_tts(buffer.strip())
            if tail and any(c.isalnum() for c in tail):
                sentence_q.put(tail)
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

        # Optional waveform feed: pre-compute the RMS envelope and walk it on
        # a ticker thread that runs concurrently with sd.play. They start at
        # roughly the same instant, so the bars stay in sync with the audio
        # to within one window (~33ms) — imperceptible.
        envelope = None
        ticker_stop = None
        ticker_thread = None
        if on_amplitude is not None:
            envelope = _compute_envelope(samples, sample_rate, window_sec=0.033)
            if envelope.size > 0:
                ticker_stop = threading.Event()
                ticker_thread = threading.Thread(
                    target=_run_envelope_ticker,
                    args=(envelope, 0.033, on_amplitude, ticker_stop),
                    daemon=True,
                )
                ticker_thread.start()

        sd.play(samples, samplerate=sample_rate, blocking=True)

        # Stop the ticker promptly so it doesn't run past playback end. The
        # console's own decay logic settles bars back to flat from here.
        if ticker_stop is not None:
            ticker_stop.set()
        if ticker_thread is not None:
            ticker_thread.join(timeout=0.2)

    t_prod.join()
    t_synth.join()

    if producer_errors:
        raise producer_errors[0]


def _compute_envelope(
    samples: np.ndarray, sample_rate: int, window_sec: float = 0.033
) -> np.ndarray:
    """Pre-compute RMS amplitude per fixed-duration window. Returns float32
    array of values in [0, 1] (clipped). Mono-mixed if multi-channel.

    Window choice: 33ms = ~30 fps, matches the visualizer's redraw rate.
    Faster windows look noisier; slower lose attack transients."""
    if samples.size == 0 or sample_rate <= 0:
        return np.array([], dtype=np.float32)

    if samples.ndim > 1:
        mono = samples.mean(axis=1)  # downmix without losing dynamic range much
    else:
        mono = samples

    window_size = max(1, int(window_sec * sample_rate))
    n_windows = len(mono) // window_size
    if n_windows == 0:
        return np.array([], dtype=np.float32)

    # Reshape into [n_windows, window_size] and compute RMS per row.
    truncated = mono[: n_windows * window_size].astype(np.float64)
    rms = np.sqrt((truncated.reshape(n_windows, window_size) ** 2).mean(axis=1))
    # 16-bit signed max is 32768. Clip to [0, 1] in case mono mixing pushed
    # beyond float intermediate range.
    return np.clip(rms / 32768.0, 0.0, 1.0).astype(np.float32)


def _run_envelope_ticker(
    envelope: np.ndarray,
    window_sec: float,
    on_amplitude: Callable[[float], None],
    stop_event: threading.Event,
) -> None:
    """Walk the envelope in real time, calling on_amplitude per window.

    Uses absolute time-since-start so we don't drift on slow callbacks. Stops
    early if the caller signals via stop_event (i.e., sd.play returned)."""
    start = time.monotonic()
    n = len(envelope)
    last_idx = -1
    while not stop_event.is_set():
        elapsed = time.monotonic() - start
        idx = int(elapsed / window_sec)
        if idx >= n:
            return
        if idx != last_idx:
            try:
                on_amplitude(float(envelope[idx]))
            except Exception:
                pass  # never let a UI callback kill the ticker
            last_idx = idx
        # Sleep until just past the next window boundary, so we sample once
        # per window without spin-waiting.
        next_boundary = start + (idx + 1) * window_sec
        sleep_for = max(0.0, next_boundary - time.monotonic())
        if stop_event.wait(sleep_for):
            return


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
