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
    # Drop URL noise that sounds terrible spoken — the http(s):// scheme and a
    # "www." prefix — so TTS reads a site as "espn.com", not "h-t-t-p-s colon
    # slash slash www dot espn dot com". Scheme first, then www. Only a literal
    # "www." (followed by a dot) is removed — a bare "www" with no dot is left
    # alone.
    text = re.sub(r"https?://", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwww\.", "", text, flags=re.IGNORECASE)
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


def _play_via_duplex_aec(device, audio_q, interrupt_event, interrupted,
                         on_first_audio, on_amplitude, on_barge, t_prod, t_synth,
                         producer_errors) -> bool:
    """M88 Phase 2 playback path. Feeds the synthesized reply (resampled to 16k)
    through a DuplexBargePlayer that cancels Jarvis's own echo + detects the user
    talking over him (sets interrupt_event). Returns True if it ran (incl.
    teardown); False if the duplex stream / pyaec couldn't start, so the caller
    falls back to the normal sd.play path. Teardown mirrors that path exactly."""
    from src import aec_barge  # noqa: PLC0415

    player = aec_barge.DuplexBargePlayer(
        device, interrupt_event,
        on_first_audio=on_first_audio, on_amplitude=on_amplitude, on_barge=on_barge,
    )
    if not player.start():
        return False  # fail-soft → caller plays normally
    try:
        # Feed the reply into the player's queue as sentences synth (the player
        # plays in real-time from there). Poll with a timeout so a barge-in is
        # honoured promptly EVEN while blocked waiting on a slow next sentence
        # from the LLM — otherwise the cut would lag until the in-flight sentence
        # arrives. Resample each sentence to the 16k duplex/AEC rate.
        while not interrupted():
            try:
                item = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            samples, sample_rate = item
            player.feed(aec_barge._resample_to_16k(np.asarray(samples), sample_rate))
        player.done_feeding()
        player.wait(interrupt_event)
    finally:
        player.close()

    # On a barge-in the synth worker may be parked on a full audio_q (maxsize=2);
    # drain to its None sentinel so the joins can't hang, then join. A producer
    # error is surfaced only when we weren't barged in (an interrupt is normal).
    if interrupted():
        while True:
            try:
                drained = audio_q.get(timeout=5.0)
            except queue.Empty:
                break
            if drained is None:
                break
    t_prod.join(timeout=1.0)
    t_synth.join(timeout=1.0)
    if producer_errors and not interrupted():
        raise producer_errors[0]
    return True


def speak_streaming(
    text_iter: Iterable[str],
    language: str = "en",
    on_first_audio: Callable[[], None] | None = None,
    on_amplitude: Callable[[float], None] | None = None,
    interrupt_event: "threading.Event | None" = None,
    aec_barge_device: "int | None" = None,
    on_barge: Callable[[], None] | None = None,
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

    `interrupt_event` (optional, M52 barge-in) — polled during playback;
    when set, playback stops, the synth/producer threads are wound down, and
    the function returns *normally*. An interrupt is not an error, so it
    deliberately does NOT raise (unlike the producer-error path below) — the
    caller distinguishes the two by inspecting the event afterwards.

    The mid-sentence cut is done HERE, on the caller's thread: each sentence
    plays non-blocking (sd.play(blocking=False)) and a short poll loop watches
    interrupt_event, calling sd.stop() itself when it fires. This is load-
    bearing — sd.stop() MUST run on the thread that owns the output stream.
    Windows WASAPI streams are COM space-threaded; calling sd.stop() from
    the barge-in monitor thread is a cross-space violation that hard-
    crashes the process (cf. project_wasapi_thread_audio_owner). The monitor
    therefore only sets the event — it never touches the audio API.

    `aec_barge_device` (optional, M88 Phase 2 hands-free barge-in) — when set,
    playback goes through a single DUPLEX stream that cancels Jarvis's own echo
    from the mic and detects the user talking OVER him, setting interrupt_event
    (no wake word). TTS is resampled to 16k for this path. Fully fail-soft: if
    the duplex stream / pyaec can't start, it falls back to the normal sd.play
    path below. None ⇒ unchanged behaviour.

    On any exception in the text producer (e.g., LLM stream raising), this
    function re-raises it after threads drain, so the caller can roll back
    conversation history. Per-sentence synth failures are logged and skipped.
    """
    voice = VOICE_BY_LANG.get(language, DEFAULT_VOICE)

    def _interrupted() -> bool:
        return interrupt_event is not None and interrupt_event.is_set()

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
                # M52: stop pulling the LLM stream the moment the user barges
                # in. stream_response also polls the same event and closes its
                # HTTP stream, so this is belt-and-suspenders — but it makes
                # the producer wind down promptly even mid-buffer.
                if _interrupted():
                    break
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
                    mp3_bytes = asyncio.run(_fetch_mp3_with_retry(sentence, voice))
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

    # M88 Phase 2: hands-free barge-in. Play via a duplex AEC stream that lets
    # the user talk OVER Jarvis. Returns True if it handled playback (including
    # teardown); False if it couldn't start, in which case we fall through to
    # the unchanged sd.play path below.
    if aec_barge_device is not None and _play_via_duplex_aec(
        aec_barge_device, audio_q, interrupt_event, _interrupted,
        on_first_audio, on_amplitude, on_barge, t_prod, t_synth, producer_errors,
    ):
        return

    first = True
    while True:
        # M52: poll the barge-in event before starting each sentence — covers
        # the gap between sentences. The mid-sentence cut is handled by the
        # non-blocking play + poll loop below (sd.stop on this thread).
        if _interrupted():
            break
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

        # M52: play non-blocking and poll, so the barge-in cut (sd.stop) and
        # the completion check both run on THIS thread — the one that owns
        # the WASAPI/COM output stream. sd.stop() from the monitor thread is
        # a cross-space COM call that HARD-crashes the process (the
        # original M52 bug; cf. project_wasapi_thread_audio_owner — WASAPI is
        # thread-affine). The monitor now only sets interrupt_event.
        sd.play(samples, samplerate=sample_rate, blocking=False)
        # Hard cap well past the real duration so a misbehaving stream can't
        # hang the loop; normal exit is the .active check or the interrupt.
        play_deadline = time.monotonic() + len(samples) / sample_rate + 1.0
        while time.monotonic() < play_deadline:
            if _interrupted():
                break
            try:
                if not sd.get_stream().active:
                    break  # sentence finished playing on its own
            except Exception:
                break  # no current stream / already closed — treat as done
            time.sleep(0.02)
        try:
            sd.stop()  # owner-thread cut: ends the sentence on a barge-in,
        except Exception as exc:  # or tidies the finished stream otherwise
            print(f"[tts] sd.stop failed: {exc}", file=sys.stderr)

        # Stop the ticker promptly so it doesn't run past playback end. The
        # console's own decay logic settles bars back to flat from here.
        if ticker_stop is not None:
            ticker_stop.set()
        if ticker_thread is not None:
            ticker_thread.join(timeout=0.2)

        # M52: barged in — stop here rather than play the next queued sentence.
        if _interrupted():
            break

    # M52: on a barge-in, the synth worker may be parked on a full audio_q
    # (maxsize=2) since we stopped pulling. Drain it to its None sentinel so
    # the worker unblocks and the joins below can't hang. The producer sees
    # interrupt_event in its own loop and winds down independently. Playback
    # itself was already stopped on this thread by the poll loop above.
    if _interrupted():
        while True:
            try:
                drained = audio_q.get(timeout=5.0)
            except queue.Empty:
                break
            if drained is None:
                break

    # On a clean turn the worker threads are already done (the LLM stream
    # ended), so these joins are instant. On a barge-in they wind down within
    # the timeout; both are daemon threads, so a rare straggler (e.g. producer
    # blocked inside a slow tool call) dies with the process — it must not
    # delay dropping the user into the follow-up listening window.
    t_prod.join(timeout=1.0)
    t_synth.join(timeout=1.0)

    # An interrupt is a normal outcome, not a failure — only surface a genuine
    # producer error, and only when we weren't barged in.
    if producer_errors and not _interrupted():
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
    mp3_bytes = asyncio.run(_fetch_mp3_with_retry(text, voice))
    decoded = miniaudio.mp3_read_s16(mp3_bytes)

    samples = np.frombuffer(decoded.samples.tobytes(), dtype=np.int16)
    if decoded.nchannels > 1:
        samples = samples.reshape(-1, decoded.nchannels)

    sd.play(samples, samplerate=decoded.sample_rate, blocking=True)


# edge-tts talks to Microsoft's voice endpoint over a websocket; a transient
# close-without-audio surfaces as NoAudioReceived ("No audio was received…") —
# its most common flake (rate-limit / network blip / service hiccup). It's the
# textbook retry case: the SAME text succeeds moments later (proven 2026-06-02
# — a 78-char sentence that failed live re-synthesized 5/5 OK). Without a retry
# a single flake permanently dropped a sentence from a spoken reply on the
# streaming path, which (unlike the one-shot speak() path) had no fallback.
_TTS_FETCH_ATTEMPTS = 3
_TTS_FETCH_BACKOFF_S = 0.4


async def _fetch_mp3(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


async def _fetch_mp3_with_retry(
    text: str, voice: str, attempts: int = _TTS_FETCH_ATTEMPTS
) -> bytes:
    """`_fetch_mp3` with a bounded retry on transient edge-tts failures.

    Retries on BOTH a raised exception (NoAudioReceived and friends) and an
    empty-but-no-exception result (defensive). After `attempts` tries it
    re-raises the last exception, so each caller's EXISTING terminal behaviour
    still runs on a genuine, persistent failure (synth_worker drops the
    sentence; speak() falls back to pyttsx3; the phone path logs). The
    streaming path's maxsize=2 lookahead buffer absorbs the backoff, so
    playback stays gapless in the common single-retry case.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            mp3 = await _fetch_mp3(text, voice)
            if mp3:
                return mp3
            last_exc = RuntimeError("edge-tts returned no audio bytes")
        except Exception as exc:  # noqa: BLE001 — retry ANY transient synth error
            last_exc = exc
        if i < attempts - 1:
            print(
                f"[tts] synth attempt {i + 1}/{attempts} failed "
                f"({last_exc}); retrying",
                file=sys.stderr,
            )
            await asyncio.sleep(_TTS_FETCH_BACKOFF_S * (i + 1))
    assert last_exc is not None  # loop ran >=1 time, so it's always set
    raise last_exc


def _speak_pyttsx3(text: str) -> None:
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()
