"""faster-whisper wrapper: state-machine endpointing + transcription.

M36: dual STT backend with auto-fallback. Two transcription paths:

  Local CPU (default, today's behavior):
    faster-whisper with device="cpu", int8 quant. 5-10s for a typical
    voice command on a Ryzen 5 3400G — fine for casual chat, the
    bottleneck when M35's 15s challenge window is tight.

  Remote GPU (M36):
    POST WAV to the FastAPI server running on MEDIA-HOST
    (stt_server/server.py). GTX 1650 + float16 = ~250-500ms per
    transcription, 10-20x faster than local CPU.

Backend selection driven by `backend` param ("auto" / "gpu" / "cpu") and
`server_url`. Auto-fallback: in "auto" mode, any remote failure (network,
HTTP error, parse error) silently degrades to local CPU and the turn
completes — never strand the user.
"""

from __future__ import annotations

import io
import sys
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable

import httpx
import numpy as np
from faster_whisper import WhisperModel

from src.audio import AudioSession

# Endpointing thresholds (int16 RMS scale).
# Hysteresis: SPEECH_RMS must exceed the adaptive silence threshold so the
# detector doesn't flap. The silence threshold is set per-turn from observed
# pre-speech ambient — same idea as a noise gate's auto-threshold or AGC.
SPEECH_RMS = 1000          # RMS above this = user is speaking
SILENCE_FLOOR = 700        # silence threshold never goes below this (clears C920e AGC floor)
SILENCE_MARGIN = 1.3       # silence threshold = ambient_max * this
MAX_PRE_SPEECH_SEC = 5.0   # give up if user never starts speaking
SILENCE_HANG_SEC = 1.2     # consecutive silence after speech to stop
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


def _record_until_silence(
    session: AudioSession,
    on_amplitude: "Callable[[float], None] | None" = None,
    max_pre_speech_sec: float | None = None,
    suppress_event: "threading.Event | None" = None,
) -> np.ndarray:
    """State-machine VAD: wait for speech start, then stop on sustained silence.

    `max_pre_speech_sec` (M51) overrides how long to wait for speech to START
    before giving up. None ⇒ the normal post-wake-word value (MAX_PRE_SPEECH_
    SEC); the conversational follow-up window passes a longer value so the
    user has time to gather a follow-up without re-saying "Hey Jarvis".

    `suppress_event` (2026-05-29 omni-mic echo fix) — if this Event becomes
    set DURING capture (the PC started speaking a reply or a proactive
    announce), abort immediately and return an empty array. The caller treats
    that like "no speech," so the no-wake-word follow-up window can never
    transcribe Jarvis's own voice as a question. None ⇒ no suppression.

    Silence threshold is computed adaptively from pre-speech ambient noise:
    track the loudest pre-speech chunk (excluding any chunk loud enough to be
    speech itself), and on speech-start lock the threshold for this turn at
    max(SILENCE_FLOOR, ambient_max * SILENCE_MARGIN). Auto-tunes to today's
    environment — A/C, fan noise, distant TV, mic AGC drift, etc.

    Edge case: if the user starts talking immediately after the wake word with
    no quiet preamble, ambient_max stays near 0 and we fall back to
    SILENCE_FLOOR — i.e., the same behavior as the old static threshold.

    M18: on_amplitude (optional) is called once per chunk with a normalized
    [0, 1] level — used by the waveform visualizer to react during LISTENING.
    Mic speech RMS is much smaller than TTS digital-scale audio, so we apply
    a 2x boost so the bars feel visually balanced with the speaking path.
    """
    chunks: list[np.ndarray] = []
    chunks_per_sec = session.sample_rate / session.chunk_samples
    silence_hang_chunks = int(SILENCE_HANG_SEC * chunks_per_sec)
    # M51: the follow-up window passes a longer pre-speech timeout; None ⇒
    # the normal post-wake-word value, so the wake-word path is unchanged.
    pre_speech_sec = (
        max_pre_speech_sec if max_pre_speech_sec is not None
        else MAX_PRE_SPEECH_SEC
    )
    max_pre_speech_chunks = int(pre_speech_sec * chunks_per_sec)
    # Total recording ceiling. Normal path: exactly MAX_RECORDING_SEC (no
    # change). Follow-up path: the pre-speech window PLUS the full speech
    # budget, so a user who takes a few seconds to begin a follow-up still
    # gets the normal ~15s to actually say it (the long wait must not eat
    # into the speech allowance).
    if max_pre_speech_sec is None:
        max_chunks = int(MAX_RECORDING_SEC * chunks_per_sec)
    else:
        max_chunks = int((pre_speech_sec + MAX_RECORDING_SEC) * chunks_per_sec)

    speaking = False
    silence_chunks = 0
    pre_speech_max_rms = 0.0
    adaptive_silence_rms = SILENCE_FLOOR  # gets locked in when speech starts

    print("[stt] listening for question...", file=sys.stderr)
    while len(chunks) < max_chunks:
        # Omni-mic echo fix: if the PC started speaking (a reply or announce)
        # while we're capturing, abort + discard — never transcribe our own
        # audio. Checked before each read so it also short-circuits on entry
        # if the PC is already speaking.
        if suppress_event is not None and suppress_event.is_set():
            print("[stt] capture aborted — PC is speaking (avoiding self-capture)",
                  file=sys.stderr)
            return np.array([], dtype=np.int16)
        chunk = session.read()
        chunks.append(chunk)
        level = _rms(chunk)

        # M18: feed the waveform visualizer. 2x boost lifts conversational
        # mic levels (~0.05–0.3 of full scale) into the same visual range
        # as TTS playback (~0.5–0.7). Clip so we never push beyond [0, 1].
        if on_amplitude is not None:
            try:
                on_amplitude(min(1.0, level / 32768.0 * 2.0))
            except Exception:
                pass  # never let a UI callback break the recording loop

        if not speaking:
            # Only count chunks that aren't speech themselves — otherwise a
            # zero-pause start ("Hey Jarvis tell me...") contaminates ambient.
            if level < SPEECH_RMS:
                pre_speech_max_rms = max(pre_speech_max_rms, level)
            if level >= SPEECH_RMS:
                speaking = True
                adaptive_silence_rms = max(
                    SILENCE_FLOOR, int(pre_speech_max_rms * SILENCE_MARGIN)
                )
                print(
                    f"[stt] speech started (rms={level:.0f}; "
                    f"ambient_max={pre_speech_max_rms:.0f}, "
                    f"silence_threshold={adaptive_silence_rms})",
                    file=sys.stderr,
                )
            elif len(chunks) >= max_pre_speech_chunks:
                print("[stt] no speech detected within window", file=sys.stderr)
                return np.array([], dtype=np.int16)
        else:
            if level < adaptive_silence_rms:
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
            f"(adaptive silence threshold was {adaptive_silence_rms}). "
            "Environment got louder mid-question, or user kept talking past 15s cap.",
            file=sys.stderr,
        )
    print(f"[stt] captured {duration:.1f}s of audio", file=sys.stderr)
    return audio


# M36: per-phase timeouts for remote STT.
#
# Split connect vs read so a DOWN server fails in ~2s (TCP SYN timeout)
# instead of the full 15s read budget. The auto-fallback path retries on
# every voice turn; without a fast connect timeout, every turn would
# eat 15s before the user got their answer.
#
# Connect timeout 2s: enough for LAN handshake (~5ms typical), short
# enough to be unmissable when the server is dead.
# Read timeout 15s: covers cold model load, long audio, GPU queue from
# a concurrent Plex transcode. Typical inference is 300-800ms; 15s is
# generous headroom.
_REMOTE_STT_CONNECT_TIMEOUT = 2.0
_REMOTE_STT_READ_TIMEOUT = 15.0


def transcribe_after_wake(
    session: AudioSession,
    model_name: str = "small",
    on_speech_ended: "Callable[[], None] | None" = None,
    on_amplitude: "Callable[[float], None] | None" = None,
    server_url: str = "",
    backend: str = "auto",
    max_pre_speech_sec: float | None = None,
    suppress_event: "threading.Event | None" = None,
) -> Transcript:
    """Record from `session` until silence, then transcribe.

    `suppress_event` (2026-05-29) — if set during capture, abort + return an
    empty Transcript (the PC is speaking; don't self-capture). See
    `_record_until_silence`.

    `max_pre_speech_sec` (M51) — how long to wait for the user to START
    speaking before returning an empty Transcript. None ⇒ the normal
    post-wake-word timeout; the conversational follow-up window passes a
    longer value so a follow-up needs no "Hey Jarvis".

    `on_speech_ended` fires after audio capture ends (silence detected) but
    before transcription starts — useful for flipping a UI state pill from
    LISTENING to THINKING right when the user stops talking, instead of
    letting it linger through Whisper's ~1-2s of work. Only fires if real
    speech was captured; not on the no-speech-detected timeout path.

    `on_amplitude` (M18) fires once per audio chunk during recording with the
    current normalized level in [0, 1]. Used by the waveform visualizer to
    react in real time while LISTENING.

    `server_url` + `backend` (M36) select the STT backend:
      - backend="cpu" or server_url="" → local CPU (today's behavior)
      - backend="gpu" + server_url set → remote GPU, no fallback
      - backend="auto" + server_url set → remote GPU, silent fallback to
        local CPU on any failure (the default + recommended)
    """
    # Pre-load local model only when we KNOW we'll use it. In "auto" mode
    # we defer the load to the fallback path — users whose GPU server is
    # always reachable never pay the ~5s load + ~250MB RSS cost.
    if backend == "cpu" or not server_url:
        _get_model(model_name)

    audio_i16 = _record_until_silence(
        session, on_amplitude=on_amplitude,
        max_pre_speech_sec=max_pre_speech_sec,
        suppress_event=suppress_event,
    )

    if audio_i16.size == 0:
        return Transcript(text="", language="")

    if on_speech_ended is not None:
        try:
            on_speech_ended()
        except Exception:
            pass  # never let a UI callback break STT

    return _dispatch_transcription(audio_i16, model_name, server_url, backend)


# --- M36: backend dispatch + helpers --------------------------------------

def _dispatch_transcription(
    audio_i16: np.ndarray, model_name: str, server_url: str, backend: str
) -> Transcript:
    """Choose local vs remote based on backend + URL, with auto-fallback.

    Routing rules:
      - "cpu" or no server URL → local CPU.
      - "gpu" → remote only; failures return an empty Transcript so the
        listen_loop's "no speech captured" path drops the turn instead of
        treating an error string as user input.
      - "auto" → try remote, silently fall back to local CPU on any failure.
    """
    if backend == "cpu" or not server_url:
        return _transcribe_local(audio_i16, model_name)

    if backend == "gpu":
        try:
            return _transcribe_remote(audio_i16, server_url)
        except Exception as exc:
            print(
                f"[stt] GPU backend forced but server unreachable: "
                f"{type(exc).__name__}: {exc} — dropping turn",
                file=sys.stderr,
            )
            return Transcript(text="", language="")

    # backend == "auto"
    t0 = time.monotonic()
    try:
        result = _transcribe_remote(audio_i16, server_url)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        print(f"[stt] remote ok in {elapsed_ms}ms", file=sys.stderr)
        return result
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        print(
            f"[stt] remote failed after {elapsed_ms}ms "
            f"({type(exc).__name__}: {exc}) — falling back to local CPU",
            file=sys.stderr,
        )
        return _transcribe_local(audio_i16, model_name)


def _transcribe_local(audio_i16: np.ndarray, model_name: str) -> Transcript:
    """Local CPU transcription path — the M35-and-earlier behavior."""
    model = _get_model(model_name)
    audio_f32 = audio_i16.astype(np.float32) / 32768.0
    segments, info = model.transcribe(
        audio_f32,
        beam_size=1,
        vad_filter=True,                  # Silero VAD pre-filter; kills noise hallucinations
        condition_on_previous_text=False,  # avoid Whisper repeating itself across segments
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return Transcript(text=text, language=info.language)


def _transcribe_remote(audio_i16: np.ndarray, server_url: str) -> Transcript:
    """POST WAV-encoded audio to the GPU server's /transcribe endpoint.
    Raises on any HTTP / network / parse failure — caller decides whether
    to fall back."""
    wav_bytes = _encode_wav(audio_i16)
    files = {"audio": ("audio.wav", wav_bytes, "audio/wav")}
    resp = httpx.post(
        f"{server_url}/transcribe",
        files=files,
        timeout=httpx.Timeout(
            connect=_REMOTE_STT_CONNECT_TIMEOUT,
            read=_REMOTE_STT_READ_TIMEOUT,
            write=_REMOTE_STT_READ_TIMEOUT,
            pool=_REMOTE_STT_CONNECT_TIMEOUT,
        ),
    )
    resp.raise_for_status()
    data = resp.json()
    return Transcript(
        text=(data.get("text") or "").strip(),
        language=(data.get("language") or "").strip(),
    )


def _encode_wav(audio_i16: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Encode int16 mono audio as WAV bytes for HTTP upload. WAV (not raw
    PCM) so the server doesn't need separate sample-rate/channels/bit-depth
    metadata — it's all in the header. Trivial overhead (44 bytes) vs raw."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # int16 = 2 bytes per sample
        wav.setframerate(sample_rate)
        wav.writeframes(audio_i16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# M48.3 — phone push-to-talk shim
# ---------------------------------------------------------------------------
#
# The phone's MediaRecorder produces a compressed container (Safari emits
# audio/mp4 with AAC; Chrome/Android emit audio/webm with Opus). The existing
# STT dispatch expects int16 mono 16 kHz numpy. PyAV — already a faster-
# whisper transitive dep (faster-whisper uses `av` for its own decode) — does
# the container/codec → PCM lift. Once we have the int16 array, the existing
# `_dispatch_transcription` takes over so M36 GPU offload still applies for
# phone voice (no parallel STT pipeline; one path, two sources).


def _as_list(x):
    """PyAV's `AudioResampler.resample(frame)` returns a single Frame on
    older releases and `list[Frame]` on newer ones; this normalises so the
    decode loop is version-agnostic."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _decode_audio_blob(blob: bytes) -> np.ndarray:
    """Decode an arbitrary container (mp4/webm/etc.) → int16 mono PCM @ 16 kHz.

    Container is auto-detected by PyAV from the bytes; we don't trust the
    client-supplied MIME (a phone could lie or the browser pick a codec
    we didn't expect). Empty / undecodable input ⇒ zero-length array
    (caller treats as "no speech captured" and drops the turn).
    """
    import av  # noqa: PLC0415 — keep the import local; surfaces clearly if absent

    container = av.open(io.BytesIO(blob))
    try:
        try:
            stream = next(s for s in container.streams if s.type == "audio")
        except StopIteration:
            return np.zeros(0, dtype=np.int16)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for rf in _as_list(resampler.resample(frame)):
                if rf is not None:
                    chunks.append(rf.to_ndarray().flatten())
        # Flush whatever the resampler had buffered.
        for rf in _as_list(resampler.resample(None)):
            if rf is not None:
                chunks.append(rf.to_ndarray().flatten())
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks).astype(np.int16, copy=False)
    finally:
        container.close()


def transcribe_blob(
    blob: bytes,
    mime: str,
    model_name: str,
    server_url: str,
    backend: str,
) -> Transcript:
    """Public M48.3 entry point: a phone-recorded audio blob → Transcript.

    Decodes the container to int16 mono 16 kHz then routes through the
    existing `_dispatch_transcription` so the GPU offload (M36) still
    applies for phone voice for free. Any decode failure (corrupt blob,
    unsupported codec, missing audio stream) returns an empty Transcript
    so the caller's empty-string check drops the turn — never strand the
    user with a stack trace from an inscrutable network glitch.
    """
    if not blob:
        return Transcript(text="", language="")
    try:
        audio_i16 = _decode_audio_blob(blob)
    except Exception as exc:  # noqa: BLE001 — decode failures must not crash STT
        print(
            f"[stt] phone-audio decode failed (mime={mime!r}): "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return Transcript(text="", language="")
    if audio_i16.size == 0:
        return Transcript(text="", language="")
    return _dispatch_transcription(audio_i16, model_name, server_url, backend)
