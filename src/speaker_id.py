"""Speaker identification — the voice analog of M39 face auth.

Goal: per turn, decide *who* is speaking (or "unknown") from the same audio
the STT path already captured. Two payoffs:
  1. an opt-in GATE — "respond only to enrolled voices" — which filters
     background media / strangers from the always-on omni-mic (the open-mic
     contamination seen 2026-06-01);
  2. a hook for per-speaker behaviour (language, and later an armed-mode
     "unknown voice" security signal).

This mirrors `face_auth.py` deliberately — same defensive contract, same
enroll-average-N / verify-by-distance shape — but differs in two ways:
  - MULTI-USER: a registry of named speakers (you, your family member, …), not one
    encoding. identify() returns the best match across all enrolled.
  - COSINE, not Euclidean: Resemblyzer d-vectors are L2-normalized, so
    similarity is a dot product in [-1, 1]; HIGHER is more similar (the
    opposite sense to face_recognition's distance).

Model: Resemblyzer (`VoiceEncoder`, a 256-d d-vector LSTM). Picked after a
measured de-risk (scripts/speaker_id_probe.py): on this mic/voice, you-vs-you
cosine ~0.895 vs you-vs-media ~0.566 (margin 0.33) — clean separation.
Steady-state embed ~30 ms; the ~5-15 s first-call JIT is paid once by warm().

Defensive contract (identical to face_auth): a missing/broken `resemblyzer`
import leaves AVAILABLE False and every public call returns a safe sentinel —
never raises into the listen loop.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# Resemblyzer + torch add import time; lazy-load so cold-start isn't penalized
# when speaker ID never runs. Module-level caches mirror face_auth.
_encoder = None                       # resemblyzer.VoiceEncoder instance
_preprocess = None                    # resemblyzer.preprocess_wav fn
AVAILABLE: bool = False
_IMPORT_ERROR: Exception | None = None

# d-vector dimensionality (Resemblyzer). A loaded embedding with another shape
# means version skew / corruption — safe to ignore.
_EMBED_DIM = 256

# STT hands us int16 mono @ 16 kHz; Resemblyzer works at 16 kHz too.
_SAMPLE_RATE = 16_000

# At least this many usable clips out of N before we enroll — tolerate one bad
# clip, don't enroll on a single lucky one (face_auth's 3-of-5 logic).
_MIN_USABLE_CLIPS = 2

# Minimum voiced samples that must survive VAD trimming for a clip to count
# (~0.5 s). Below this the embedding is unreliable.
_MIN_VOICED_SAMPLES = _SAMPLE_RATE // 2


def _ensure_imported() -> bool:
    """First-call import of resemblyzer. O(1) thereafter (module caches).
    Returns True if usable, False on any import error."""
    global _encoder, _preprocess, AVAILABLE, _IMPORT_ERROR
    if _encoder is not None:
        return True
    if _IMPORT_ERROR is not None:
        return False
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav  # noqa: PLC0415
        _encoder = VoiceEncoder()       # loads the bundled model (CPU)
        _preprocess = preprocess_wav
        AVAILABLE = True
        return True
    except Exception as exc:  # noqa: BLE001 — any failure disables the path
        _IMPORT_ERROR = exc
        print(f"[speaker_id] resemblyzer import failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return False


# --- Embedding -------------------------------------------------------------

def _to_float32(audio: np.ndarray) -> np.ndarray:
    """int16 → float32 in [-1, 1]; pass float32 through. STT gives int16."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32, copy=False)


def embed(audio: np.ndarray) -> np.ndarray | None:
    """Embed one mono 16 kHz utterance → a 256-d L2-normalized d-vector, or
    None if the model is unavailable or the clip has too little voiced audio.

    Runs Resemblyzer's `preprocess_wav` first (resample-noop at 16 kHz +
    normalize + webrtcvad silence-trim) so the embedding reflects speech, not
    leading/trailing silence. Never raises."""
    if not _ensure_imported():
        return None
    if audio is None or audio.size == 0:
        return None
    try:
        wav = _preprocess(_to_float32(audio), source_sr=_SAMPLE_RATE)
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"[speaker_id] preprocess failed: {exc}", file=sys.stderr)
        return None
    if wav.size < _MIN_VOICED_SAMPLES:
        return None
    try:
        return _encoder.embed_utterance(wav)
    except Exception as exc:  # noqa: BLE001
        print(f"[speaker_id] embed failed: {exc}", file=sys.stderr)
        return None


def warm() -> None:
    """Pay Resemblyzer's first-call JIT (~5-15 s: torch + numba/librosa) once,
    in silence, so it never faults in on the first real turn. Mirrors
    face_auth.warm(). Never raises."""
    if not _ensure_imported():
        return
    try:
        t0 = time.monotonic()
        # A short non-zero buffer; content is irrelevant — we only need the
        # mel/torch path to compile. Skip preprocess (VAD would trim silence
        # to nothing); embed the raw buffer directly.
        _encoder.embed_utterance(np.full(_SAMPLE_RATE, 1e-4, dtype=np.float32))
        print(f"[speaker_id] warmed in {time.monotonic() - t0:.1f}s", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 — warm-up must never raise
        print(f"[speaker_id] warm-up incomplete: {type(exc).__name__}: {exc}",
              file=sys.stderr)


# --- Registry (multi-user) -------------------------------------------------

@dataclass(frozen=True)
class Speaker:
    """One enrolled speaker: a stable name, a language hint, and the 256-d
    averaged embedding."""
    name: str
    lang: str
    embedding: np.ndarray


@dataclass(frozen=True)
class IdentifyResult:
    """Outcome of identify(). `recognized` is the gate-relevant bit; `name`/
    `lang` are populated only when recognized. `score` is the best cosine to
    any enrolled speaker (always set, for logging/tuning)."""
    recognized: bool
    name: str | None
    lang: str | None
    score: float


def _slug(name: str) -> str:
    """Filesystem-safe stem for a speaker name ('My Aunt' → 'my-aunt')."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s or "speaker"


def load_registry(registry_dir: Path) -> list[Speaker]:
    """Load all enrolled speakers from `registry_dir` (one {slug}.npy embedding
    + {slug}.json metadata per speaker). Skips corrupt/mismatched files. Pure
    numpy/json — does NOT require resemblyzer to be importable."""
    speakers: list[Speaker] = []
    if not registry_dir.exists():
        return speakers
    for npy in sorted(registry_dir.glob("*.npy")):
        try:
            emb = np.load(npy)
        except Exception as exc:  # noqa: BLE001
            print(f"[speaker_id] couldn't load {npy.name}: {exc}", file=sys.stderr)
            continue
        if emb.shape != (_EMBED_DIM,):
            print(f"[speaker_id] {npy.name} has shape {emb.shape}, expected "
                  f"({_EMBED_DIM},); ignoring.", file=sys.stderr)
            continue
        meta_path = npy.with_suffix(".json")
        name, lang = npy.stem, "en"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                name = meta.get("name", name)
                lang = meta.get("lang", lang)
            except Exception as exc:  # noqa: BLE001
                print(f"[speaker_id] bad metadata {meta_path.name}: {exc}",
                      file=sys.stderr)
        speakers.append(Speaker(name=name, lang=lang, embedding=emb))
    return speakers


def list_names(registry_dir: Path) -> list[str]:
    """Names of currently enrolled speakers (for tray/status display)."""
    return [s.name for s in load_registry(registry_dir)]


def delete_speaker(name: str, registry_dir: Path) -> bool:
    """Remove an enrolled speaker by name. True if anything was removed."""
    slug = _slug(name)
    removed = False
    for ext in (".npy", ".json"):
        p = registry_dir / f"{slug}{ext}"
        if p.exists():
            try:
                p.unlink()
                removed = True
            except OSError as exc:
                print(f"[speaker_id] couldn't delete {p.name}: {exc}", file=sys.stderr)
    if removed:
        print(f"[speaker_id] removed enrolled speaker '{name}'", file=sys.stderr)
    return removed


def enroll_from_audio(
    name: str,
    lang: str,
    clips: list[np.ndarray],
    registry_dir: Path,
) -> tuple[bool, str]:
    """Average N utterance embeddings into one enrolled vector and persist it.

    `clips` are mono 16 kHz int16 (or float32) arrays — typically a few short
    recordings of the same speaker. Per-clip embeddings are mean-averaged then
    re-normalized to unit length (the standard d-vector enrollment; mirrors
    face_auth's mean-of-encodings). Requires _MIN_USABLE_CLIPS usable clips.

    Returns (ok, user-facing message). Never raises."""
    if not _ensure_imported():
        return False, "Speaker recognition isn't available, sir."
    if not clips:
        return False, "I didn't catch any audio, sir."

    embs = [e for e in (embed(c) for c in clips) if e is not None]
    if len(embs) < _MIN_USABLE_CLIPS:
        return False, (
            f"I only got a clear recording in {len(embs)} of {len(clips)} tries, "
            "sir. Try again somewhere quieter and speak the whole time."
        )

    mean = np.mean(np.stack(embs), axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        return False, "Those recordings didn't give me anything usable, sir."
    enrolled = (mean / norm).astype(np.float32)

    slug = _slug(name)
    try:
        registry_dir.mkdir(parents=True, exist_ok=True)
        np.save(registry_dir / f"{slug}.npy", enrolled)
        (registry_dir / f"{slug}.json").write_text(
            json.dumps({"name": name, "lang": lang,
                        "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[speaker_id] couldn't save enrollment: {exc}", file=sys.stderr)
        return False, "I couldn't save the enrollment, sir."

    print(f"[speaker_id] enrolled '{name}' (lang={lang}) from "
          f"{len(embs)}/{len(clips)} clips → {slug}.npy", file=sys.stderr)
    return True, f"I'll recognize your voice now, sir. ({len(embs)} of {len(clips)} clips used.)"


def identify(
    audio: np.ndarray,
    registry: list[Speaker],
    threshold: float,
) -> IdentifyResult:
    """Who is speaking in `audio`? Embeds the clip and takes the best cosine
    match across `registry`. `recognized` is True iff the best score >=
    threshold (HIGHER cosine = more similar — note this is the opposite sense
    to face_auth's distance).

    Fail-open contract: any failure (no model, empty registry, embed failure)
    returns recognized=False with score 0.0 — the CALLER decides what an
    unrecognized turn means (the gate is opt-in; default behaviour ignores
    `recognized`). Never raises."""
    if not registry:
        return IdentifyResult(recognized=False, name=None, lang=None, score=0.0)
    probe = embed(audio)
    if probe is None:
        return IdentifyResult(recognized=False, name=None, lang=None, score=0.0)
    best_score = -1.0
    best: Speaker | None = None
    for sp in registry:
        score = float(np.dot(probe, sp.embedding))
        if score > best_score:
            best_score, best = score, sp
    recognized = best is not None and best_score >= threshold
    return IdentifyResult(
        recognized=recognized,
        name=best.name if (recognized and best) else None,
        lang=best.lang if (recognized and best) else None,
        score=best_score,
    )


# --- Enrollment voice flow -------------------------------------------------
# Both the tray "Enroll my voice" callback and the voice intent share this
# orchestration — voice-first, like face_auth.run_voice_enrollment.

# enroll/remember/register/learn/save + "voice" (optionally my/your). Anchored
# on "voice" so casual talk about memory never matches.
_ENROLL_INTENT_RE = re.compile(
    r"\b(enroll|remember|register|learn|save)\b\s+(my\s+|your\s+)?voice\b",
    re.IGNORECASE,
)


def matches_enroll_intent(transcript: str) -> bool:
    """Does the transcript express 'enroll my voice' intent? Used by the
    listen loop to consume the turn locally instead of routing to Claude."""
    return bool(_ENROLL_INTENT_RE.search(transcript or ""))


# Named enrollment of OTHER household members (M69 Phase 4) — "enroll Alice's
# voice", "enroll voice Bob in Spanish". The reliable path is the typed
# console command (Whisper mangles names); voice works too but expect retries.
# Two shapes: "voice <name>" and "<name>'s voice", plus an optional language.
_NAMED_ENROLL_RE = re.compile(
    r"\benroll\b\s+(?:the\s+)?"
    r"(?:voice\s+(?:of\s+|for\s+)?(?P<n1>[\w'-]+)"
    r"|(?P<n2>[\w'-]+)(?:'s|s')\s+voice)"
    r"(?:\s+(?:in\s+)?(?P<lang>spanish|english|español|inglés))?",
    re.IGNORECASE,
)


def parse_named_enroll_intent(transcript: str) -> "tuple[str, str] | None":
    """Parse 'enroll <name>'s voice' / 'enroll voice <name>' [in Spanish] →
    (Name, lang). Returns None when there's no named target — in particular
    'my'/'your' map to None (that's the PRIMARY user, handled by
    matches_enroll_intent). Check this BEFORE matches_enroll_intent, since
    'enroll voice Alice' also matches the primary pattern's bare 'voice'."""
    m = _NAMED_ENROLL_RE.search(transcript or "")
    if not m:
        return None
    name = (m.group("n1") or m.group("n2") or "").strip()
    if not name or name.lower() in ("my", "your", "the", "a"):
        return None
    raw = (m.group("lang") or "").lower()
    lang = "es" if raw in ("spanish", "español") else "en"
    return (name[:1].upper() + name[1:], lang)


def run_voice_enrollment(
    announce_fn: Callable[..., None],
    capture_fn: Callable[[int, float], list | None],
    registry_dir: Path,
    name: str,
    lang: str = "en",
    num_clips: int = 3,
    clip_seconds: float = 4.0,
) -> None:
    """Speak the prompt → capture N clips → enroll → speak the result. The
    capture runs on the Announcer's on_done callback so it starts only after
    the prompt has finished playing (otherwise the prompt audio leaks into the
    first clip). Mirrors face_auth.run_voice_enrollment.

    `capture_fn(n, seconds)` → list of int16 16 kHz arrays (one per clip) or
    None; decoupled for testability (production captures from the mic)."""
    if not _ensure_imported():
        announce_fn("Speaker recognition isn't installed, sir.")
        return

    def _after_prompt() -> None:
        try:
            clips = capture_fn(num_clips, clip_seconds)
        except Exception as exc:  # noqa: BLE001 — defensive against mic errors
            print(f"[speaker_id] enrollment capture raised: {exc}", file=sys.stderr)
            announce_fn("I couldn't access the microphone, sir.")
            return
        if not clips:
            announce_fn("I couldn't access the microphone, sir.")
            return
        _, msg = enroll_from_audio(name, lang, clips, registry_dir)
        announce_fn(msg)

    total = int(num_clips * clip_seconds)
    announce_fn(
        f"When I finish speaking, talk naturally for about {total} seconds so I "
        f"can learn your voice, sir — start as soon as I stop.",
        on_done=_after_prompt,
    )
