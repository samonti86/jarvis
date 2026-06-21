"""Jarvis — top-level entry point.

Threading layout:
- Main thread:    Tk's mainloop (JarvisConsole window). Tk strongly prefers
                  the main thread, so this is its home.
- Worker thread:  pystray icon loop (started by JarvisUI). Win32 lets pystray
                  run on any thread; we put it here so Tk can have main.
- Worker thread:  listen_loop (voice path). Wakes on "Hey Jarvis", transcribes,
                  then calls process_question.
- Worker thread:  text_input_loop (M15). Pops typed messages from a queue and
                  calls process_question. Both paths share state (history,
                  memory) and serialize via processing_lock so voice and text
                  can't overlap.

All UI surfaces (Tk console, pystray) are thread-safe — console via .after(),
pystray via attribute writes.

Conversation memory: history is alternating user/assistant messages. Trimmed
in pairs to MAX_PAIRS most recent exchanges. Reset paths:
(a) tray menu "Reset conversation" — fires reset_event, applied at the next
    question (voice or text)
(b) idle timeout — IDLE_RESET_SEC since last completed turn → forget
(c) app restart — history is in-process only, never persisted

Console hiding: when launched via pythonw.exe (jarvis.pyw), there's no console.
setup_logging() redirects stdout/stderr to %LOCALAPPDATA%\\Jarvis\\jarvis.log so
we don't lose debug output. Console mode (python main.py) is unchanged.

Echo handling: while TTS plays, the mic still captures it. session.drain() at
the top of each voice iteration discards that buffered echo so the wake-word
detector starts each turn on fresh, live audio.
"""

from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple

from src.audio import AudioSession, resolve_input_device
from src.config import Config, load
from src.llm import TelemetryRecord, stream_response, stream_translation
from src.memory import MemoryStore, SummaryRecord, default_base_dir, summarize_session
from src.plex_laptop import DEFAULT_LOG_PATH as DEFAULT_PLEX_LAPTOP_LOG, PlexLaptopClient
from src.plex_mcp import PlexMCPClient
from src import quiet_hours
from src import speaker_id
from src.speech_to_text import transcribe_after_wake
from src.text_to_speech import speak, speak_streaming
from src import aec_barge
from src.tray import State
from src.ui import JarvisUI
from src.wake_word import monitor_for_wake_word, wait_for_wake_word


MAX_PAIRS = 10            # cap conversation at 10 exchanges (20 messages)
IDLE_RESET_SEC = 600.0    # 10 min of silence → forget conversation


def _barge_in_enabled() -> bool:
    """M52 kill switch. JARVIS_BARGE_IN ∈ {0,false,no,off} disables barge-in
    entirely — the escape hatch if the soak ever shows the concurrent
    wake-word monitor stutters the Python-fed TTS path (the documented
    stutter-gate risk: see project_security_audio_stutter_gate). Default on.
    Read once at import; it never changes mid-session."""
    raw = os.getenv("JARVIS_BARGE_IN", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


_BARGE_IN_ENABLED = _barge_in_enabled()


class _TeeStream:
    """Forwards writes to multiple underlying streams. Minimal surface — only
    the methods print() actually touches. We deliberately don't expose
    .fileno() because libraries that introspect it might try to dup our fd
    and bypass the tee."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False


def setup_logging() -> Path:
    """Always write to %LOCALAPPDATA%\\Jarvis\\jarvis.log. Behavior by mode:

    - jarvis.pyw launcher: redirected stdout/stderr at import time and set
      JARVIS_LOG_PATH. We respect that and just return the path.
    - python main.py (console): tee stdout/stderr to both terminal *and* file.
      You see live output AND the conversation is persisted.
    - pythonw main.py (rare; no launcher): no console to tee to, redirect only.
    """
    from src.logfile import rotate_if_needed, TimestampStream

    log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "jarvis.log"

    env_path = os.environ.get("JARVIS_LOG_PATH")
    if env_path:
        # jarvis.pyw already rotated + opened the file + replaced stdout/stderr.
        return Path(env_path)

    # Rotate BEFORE opening — Windows can't rename a file with an open handle.
    rotate_if_needed(log_path)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    log_file.write(f"\n--- Jarvis started {datetime.now().isoformat(timespec='seconds')} ---\n")
    log_file.flush()

    if sys.stdout is None or sys.stderr is None:
        # pythonw without the launcher — no console to tee to.
        stamped = TimestampStream(log_file)
        sys.stdout = stamped
        sys.stderr = stamped
    else:
        # Console mode — tee both. Live terminal output + persistent file. Only
        # the FILE side is timestamped (one shared wrapper so stdout/stderr share
        # the line state); the live terminal stays clean for dev readability.
        stamped_file = TimestampStream(log_file)
        sys.stdout = _TeeStream(sys.stdout, stamped_file)
        sys.stderr = _TeeStream(sys.stderr, stamped_file)

    return log_path


def _trim_history(history: list[dict]) -> None:
    """Drop oldest pairs in place, keeping the most recent MAX_PAIRS exchanges.
    Always called when history is in clean pair state (ends on 'assistant')."""
    while len(history) > MAX_PAIRS * 2:
        del history[:2]  # drop oldest user + assistant


def _seal_session(
    memory: MemoryStore,
    history: list[dict],
    language: str,
    started_at: str,
    cfg: Config,
) -> None:
    """Best-effort: ask Haiku for a summary of the just-ended conversation
    and append it to summaries.jsonl. Idempotent — caller clears history
    after this returns. Never raises; logs and moves on if summarization fails."""
    if not history:
        return
    summary_text = summarize_session(
        history,
        language=language,
        api_key=cfg.anthropic_api_key,
        model=cfg.summary_model,
    )
    if not summary_text:
        return  # already logged by summarize_session
    memory.append_summary(SummaryRecord(
        started_at=started_at,
        ended_at=datetime.now().isoformat(timespec="seconds"),
        language=language,
        summary=summary_text,
    ))
    print(f"[memory] sealed session: {summary_text}", file=sys.stderr)


# M51 — conversational follow-up window. After Jarvis answers, the listen
# loop stays open for this many seconds and accepts a follow-up WITHOUT a
# fresh "Hey Jarvis"; silence past the window falls back to wake-word mode.
# Long enough to gather a follow-up thought, short enough to bound the
# no-wake-word false-capture window (ambient speech / TV).
_FOLLOWUP_WINDOW_SEC = 12.0

# M87 — interpreter mode. While interpreting, the listen loop waits this long
# for the next person to start speaking before re-arming (no wake word ever).
# Longer than the follow-up window: two people taking turns through an
# interpreter pause naturally, and on elapse we simply re-listen (still in
# interpreter mode), so this is just the silence cadence, not a timeout.
_INTERPRETER_WINDOW_SEC = 30.0

# M88 — conversation mode (full-duplex, Phase 1). While in conversation mode the
# loop listens hands-free with this pre-speech window; on silence it re-arms
# (stays in the mode) rather than dropping to the wake word — UNTIL this many
# consecutive empty windows elapse, at which point it auto-exits to standby (so
# Jarvis isn't left listening to an empty room indefinitely). 25s × 3 ≈ 75s of
# silence before auto-exit.
_CONVERSATION_WINDOW_SEC = 25.0
_CONVERSATION_IDLE_EXITS = 3

# M51 — conversation sign-offs. When the user's turn ENDS with one of these,
# the follow-up window is NOT opened — an explicit "that's all" should close
# the conversation cleanly, not leave the mic listening for 12s. Matched by
# suffix on a normalized transcript: a genuine sign-off is the tail of what
# the user says, so "thanks Jarvis, that is all" matches but "that's all I
# need — what about the Jets?" does not.
_DISMISSAL_PHRASES = frozenset({
    "that is all", "thats all", "that is all for now", "thats all for now",
    "that will be all", "thatll be all", "that is it", "thats it",
    "that is everything", "thats everything", "nothing else", "nothing more",
    "no thank you", "no thanks", "im done", "im good", "im all set",
    "we are done", "were done", "all done", "thank you jarvis",
    "thanks jarvis", "goodbye", "good night", "goodnight",
})

# M51 follow-on (2026-05-21): a trailing courtesy masks a sign-off. "That is
# all, thank you" ends with "thank you", not "that is all", so the bare suffix
# match missed it — the window stayed open and the user had to repeat "that is
# all". We strip ONE trailing courtesy and re-test, rather than registering
# bare "thank you" as a dismissal phrase: that would false-trip questions like
# "how do you say thank you". Longest-first so the whole phrase is removed.
_TRAILING_COURTESY = (
    "thank you very much", "thank you so much", "thanks so much",
    "thank you", "thanks", "please",
)


def _is_dismissal(text: str) -> bool:
    """True if the user's utterance is a conversation sign-off — M51 uses this
    to skip opening a follow-up window after it. Suffix match on a normalized
    transcript (lowercased, punctuation/apostrophes stripped), with one
    trailing courtesy ("...thank you") stripped before re-testing."""
    norm = " ".join(re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).split())
    if not norm:
        return False

    def _suffix_match(s: str) -> bool:
        return bool(s) and any(
            s == p or s.endswith(" " + p) for p in _DISMISSAL_PHRASES
        )

    if _suffix_match(norm):
        return True
    # Retry with a trailing courtesy removed — "that is all, thank you". A
    # bare "thank you" (norm == courtesy) strips to "" → _suffix_match False,
    # so plain politeness still does NOT close the conversation (M51's
    # deliberate choice: only "thank you JARVIS" — with the name — signs off).
    for courtesy in _TRAILING_COURTESY:
        if norm == courtesy or norm.endswith(" " + courtesy):
            return _suffix_match(norm[: -len(courtesy)].strip())
    return False


class TurnRunner:
    """Runs one conversation turn end-to-end and owns the session state.

    Owns the conversation `history`, the recalled `summaries`, the active
    session's language / start-time / last-turn-time, the `MemoryStore`, and a
    processing lock. The PC-voice path and the text/phone path share ONE
    TurnRunner; both call `process_question`, which holds the lock for the
    whole LLM-stream + TTS duration — so a typed message that lands while
    Jarvis is answering a spoken one waits its turn (no overlapping responses).

    This is the unit that used to be the `process_question` closure inside
    `listen_loop`; pulled out so it has a testable surface
    (scripts/turn_runner_test.py) and listen_loop stays a readable loop.
    """

    def __init__(
        self,
        cfg: Config,
        ui: JarvisUI,
        reset_event: threading.Event,
        plex_client: PlexMCPClient | None = None,
        plex_laptop_client: PlexLaptopClient | None = None,
        pc_speaking: "threading.Event | None" = None,
        announce_speaking: "threading.Event | None" = None,
        mic_device: "int | None" = None,
    ) -> None:
        self._cfg = cfg
        self._ui = ui
        self._reset_event = reset_event
        self._plex_client = plex_client
        self._plex_laptop_client = plex_laptop_client
        # M88 Phase 2: the resolved mic device index for the hands-free barge-in
        # duplex stream (None = default input). Only used when
        # JARVIS_HANDS_FREE_BARGE is on; otherwise inert.
        self._mic_device = mic_device
        # "PC is speaking out loud" gate — SET while this turn plays audio so
        # the always-on voice-capture loop discards self-audio (omni-mic echo
        # fix). Shared with the Announcer (announces set it too); a fresh local
        # Event if unwired (tests / standalone) so it's never None.
        self._pc_speaking = pc_speaking if pc_speaking is not None else threading.Event()
        # Cooperative SPEECH gate (the 2026-05-19 stutter post-mortem) — SET
        # while this turn plays audio so the armed CPU loops (SoundDetector's
        # PANNs inference, SecurityWatcher's YOLO/grab/gc bursts) DEFER and
        # don't starve the Python-fed TTS path. Same Event the Announcer sets
        # for proactive announces; extending it to TURN replies here closes the
        # 2026-06-01 gap where a reply spoken while armed stuttered because only
        # announces were gated (see _build_announcer's CONTRACT). Distinct from
        # `pc_speaking` (mic-capture echo) — different failure, same lifetime.
        self._announce_speaking = (
            announce_speaking if announce_speaking is not None else threading.Event()
        )

        self._history: list[dict] = []
        self._last_turn_time = 0.0
        self._memory = MemoryStore()
        self._memory.prune(retain_raw_days=cfg.retain_raw_days)
        self._summaries = self._memory.recent_summaries(cfg.memory_recall_count)
        print(f"[memory] loaded {len(self._summaries)} prior session summaries",
              file=sys.stderr)
        # Language + start-time of the *currently active* session (the one
        # being built up in `history`). Reset on every seal.
        self._session_language = "en"
        self._session_started_at = datetime.now().isoformat(timespec="seconds")
        # Serializes process_question so voice and text paths never run together.
        self._lock = threading.Lock()

    def has_active_conversation(self) -> bool:
        """True if there's an unsealed conversation in memory — the reset
        handler + shutdown seal use this to decide whether there's anything to
        seal/log."""
        return bool(self._history)

    def seal_and_refresh(self) -> None:
        """Seal the active session (if any) and reload summaries for the next one."""
        _seal_session(self._memory, self._history, self._session_language,
                      self._session_started_at, self._cfg)
        self._history.clear()
        self._summaries = self._memory.recent_summaries(
            self._cfg.memory_recall_count)
        self._session_started_at = datetime.now().isoformat(timespec="seconds")

    def seal_on_shutdown(self) -> None:
        """Final seal at app quit — persist whatever's still in memory. No
        reload (we're shutting down)."""
        _seal_session(self._memory, self._history, self._session_language,
                      self._session_started_at, self._cfg)

    def _emit_remote_reply(self, reply_text: "Callable[[str], None] | None",
                           answer: str) -> None:
        """Post the final reply to a per-turn TEXT sink (Discord), if one was
        supplied. ONLY the originating surface — deliberately NOT the
        add_jarvis_text broadcast, so a PC/voice/phone turn never leaks into a
        shared Discord channel. Covers the success, apology, and empty-reply
        paths so a remote user always gets *something* back. Fail-soft — a post
        hiccup can't break the turn."""
        if reply_text is None:
            return
        try:
            reply_text(answer)
        except Exception as exc:  # noqa: BLE001
            print(f"[main] reply_text sink failed: {exc}", file=sys.stderr)

    def _begin_turn(self, text: str, language: str,
                    attachments: list[dict] | None) -> None:
        """Open one turn: apply a pending reset/idle session boundary, capture
        the session language on the first turn, surface the user message to the
        console/UI, and append it to history (as a content-block list when there
        are attachments, else a plain string). Runs under the caller's lock —
        does NOT acquire self._lock itself."""
        # Apply any pending reset/idle boundary BEFORE this turn — so the
        # user's question starts a fresh session rather than tacking onto
        # a stale one.
        if self._reset_event.is_set():
            if self._history:
                print(f"[main] conversation reset (manual; sealing {len(self._history)} msgs)")
                self._ui.add_system_text("conversation reset.")
            self.seal_and_refresh()
            self._reset_event.clear()
        elif self._history and (time.time() - self._last_turn_time) > IDLE_RESET_SEC:
            print(f"[main] conversation reset (idle >{IDLE_RESET_SEC:.0f}s)")
            self._ui.add_system_text("conversation reset (idle).")
            self.seal_and_refresh()

        # First turn of a (possibly new) session — capture its language.
        if not self._history:
            self._session_language = language or "en"
            self._session_started_at = datetime.now().isoformat(timespec="seconds")

        print(f"\n[user, {language}] {text}")
        print("[jarvis] ", end="", flush=True)
        self._ui.add_user_text(text, language)

        # Build user-message content. With attachments: list of blocks
        # (each attachment first so Claude has the doc/image in context
        # before the question text). Without: plain string (cheaper).
        if attachments:
            content: list[dict] | str = list(attachments)
            if text:
                content.append({"type": "text", "text": text})
        else:
            content = text

        self._history.append({"role": "user", "content": content})

    def _finalize_turn(self, *, response_chunks: list[str], interrupted: bool,
                       text: str, language: str,
                       reply_text: "Callable[[str], None] | None",
                       reply_audio: "Callable[[bytes], None] | None",
                       speaker: "str | None" = None) -> bool:
        """Assemble the streamed reply, persist + relay it, and (for a phone
        turn) synthesize the reply audio. Returns `interrupted` (the value the
        voice loop uses to decide whether to open a follow-up window). Runs
        under the caller's lock — does NOT acquire self._lock itself.

        An empty reply pops the orphan user message and (for a remote turn)
        emits a short acknowledgement so the user isn't left in silence."""
        full_response = "".join(response_chunks).strip()
        if not full_response:
            self._history.pop()  # nothing came back; drop the orphan user message
            # Don't leave a REMOTE user in silence wondering if the turn
            # even landed — Claude can legitimately return an empty turn
            # (e.g. a tool-only turn that yields no prose). Emit a short
            # acknowledgement to the originating surface, mirroring the
            # apology path's fan-out. The PC/voice path stays silent (the
            # user is present and saw the state indicators); nothing is
            # appended to history (nothing was actually said).
            if not interrupted and (reply_text is not None or reply_audio is not None):
                ack = ("Disculpe, no tengo nada que añadir."
                       if language == "es"
                       else "I didn't have anything to add, sir.")
                self._ui.add_jarvis_text(ack)
                self._emit_remote_reply(reply_text, ack)
            return interrupted

        self._history.append({"role": "assistant", "content": full_response})
        _trim_history(self._history)
        self._last_turn_time = time.time()

        # Persist the completed exchange to today's transcript file. M82 —
        # tag the user side with the voice-identified speaker so
        # recall_conversation can answer "what did Alice ask me to do?".
        self._memory.record_turn(text, full_response, language, speaker=speaker)

        self._ui.add_jarvis_text(full_response)
        self._emit_remote_reply(reply_text, full_response)

        # M48.2b: speak the reply to the phone that asked (UNICAST,
        # mute-independent). Whole-reply-at-completion v1 — matches the
        # at-completion text parity; streamed audio is a documented
        # deferrable. Same edge-tts voice as the PC (VOICE_BY_LANG).
        # Defensive: the text already reached the phone via the fan-out
        # above, so a synth/transport hiccup must NEVER break the turn.
        if reply_audio is not None:
            try:
                import asyncio  # noqa: PLC0415 — lazy per main.py convention
                from src.text_to_speech import (  # noqa: PLC0415
                    DEFAULT_VOICE, VOICE_BY_LANG, _fetch_mp3_with_retry,
                )

                voice = VOICE_BY_LANG.get(language, DEFAULT_VOICE)
                mp3 = asyncio.run(_fetch_mp3_with_retry(full_response, voice))
                reply_audio(mp3)
            except Exception as exc:  # noqa: BLE001
                print(f"[main] phone reply audio failed: {exc}",
                      file=sys.stderr)
        print()
        return interrupted

    def _stream_and_speak(
        self, *, llm_stream, pc_silent: bool, barge_enabled: bool,
        session: "AudioSession | None", interrupt_event: "threading.Event | None",
        language: str, reply_text: "Callable[[str], None] | None",
        aec_barge_on: bool = False,
    ) -> bool:
        """Run the LLM stream and, when audible, speak it — holding the
        cooperative speech gates for the spoken portion and running the
        barge-in monitor for its duration. Returns True if the turn ERRORED
        (the caller then returns False after the apology surfaced here); False
        on a clean run. Runs under the caller's lock — does NOT acquire it.

        For the audible portion it SETS self._pc_speaking (omni-mic echo
        suppression) + self._announce_speaking (the armed PANNs/YOLO CPU loops
        defer so the TTS path doesn't stutter), and clears both in `finally`.
        On an exception it pops the orphan user message and surfaces a brief
        spoken/text apology."""
        speaking_aloud = not pc_silent
        monitor_stop = threading.Event()
        monitor_thread: threading.Thread | None = None

        # Mark "PC is speaking out loud" for the audible portion of this turn so
        # the always-on voice-capture loop (esp. the M51 follow-up window)
        # discards anything it picks up while we talk — the mic stops
        # self-capturing our own reply (the omni-mic echo, 2026-05-29). Silent
        # turns (mute / phone) never set it; cleared in `finally` the instant
        # our audio ends. The announce_speaking gate (set alongside) makes the
        # armed CPU loops defer for the spoken portion, exactly as for a
        # proactive announce (2026-06-01: only announces were gated → stutter).
        if speaking_aloud:
            self._pc_speaking.set()
            self._announce_speaking.set()

        try:
            if pc_silent:
                # Drain the LLM stream silently. response_chunks gets populated
                # inside llm_stream via the print + append, so finalize works
                # unchanged. State stays THINKING — caller flips to IDLE after.
                for _ in llm_stream():
                    pass
            else:
                # M52: run the wake-word barge-in monitor for the duration of
                # playback. It reads the mic on its own thread and, on a "Hey
                # Jarvis", just sets interrupt_event — speak_streaming polls it
                # and performs the sd.stop() cut on its own (stream-owning)
                # thread, the WASAPI thread-affinity fix. M88 Phase 2: when
                # `aec_barge_on` is set we do NOT start this monitor — the duplex
                # AEC stream inside speak_streaming does the detection (hands-free
                # talk-over) and sets the same interrupt_event. (Named *_on to not
                # shadow the imported `aec_barge` module — see process_question.)
                if barge_enabled:
                    monitor_thread = threading.Thread(
                        target=monitor_for_wake_word,
                        args=(session, interrupt_event, monitor_stop),
                        kwargs={"threshold": self._cfg.wake_word_threshold},
                        name="BargeInMonitor",
                        daemon=True,
                    )
                    monitor_thread.start()
                speak_streaming(
                    llm_stream(),
                    language=language,
                    on_first_audio=lambda: self._ui.set_state(State.SPEAKING),
                    on_amplitude=self._ui.set_amplitude,
                    interrupt_event=interrupt_event,
                    aec_barge_device=(self._mic_device if aec_barge_on else None),
                    # M88 Phase 2: on a hands-free cut, flip the orb to LISTENING
                    # the instant we detect — not after the stream teardown — so
                    # it doesn't linger on SPEAKING.
                    on_barge=lambda: self._ui.set_state(State.LISTENING),
                )
        except Exception as exc:
            self._history.pop()  # keep history alternating user/assistant cleanly
            print(f"\n[main] LLM/TTS failed: {exc}")
            # M20: don't leave the user in silence after a partial reply. Speak
            # a brief apology in their language via the simpler Tier A path (no
            # streaming pipeline that could fail again). Wrap in defensive
            # try/except so the apology itself can't break the loop. Add to
            # transcript too so the console reflects it. When silent (muted OR
            # phone-text origin), skip the spoken apology but still surface its
            # text — the phone/console sees the hiccup, the PC stays quiet.
            apology = (
                "Disculpe, tuve un problema técnico. ¿Podría intentarlo de nuevo?"
                if language == "es"
                else "Apologies, a technical hiccup. Could you try that again?"
            )
            if not pc_silent:
                self._ui.set_state(State.SPEAKING)
                try:
                    speak(apology, language=language)
                except Exception as apology_exc:
                    print(f"[main] apology TTS also failed: {apology_exc}")
            # Phone-audio turns: the apology TEXT still reaches the phone via the
            # fan-out; we deliberately don't synth audio on the error path
            # (extra failure surface for little gain — the text conveys it).
            self._ui.add_jarvis_text(apology)
            self._emit_remote_reply(reply_text, apology)
            return True
        finally:
            # Stop suppressing voice capture + release the speech gate the
            # instant our audio ends (so the armed loops resume promptly).
            if speaking_aloud:
                self._pc_speaking.clear()
                self._announce_speaking.clear()
            # M52: always wind the barge-in monitor down — normal end,
            # exception, OR barge-in. monitor_stop makes its next session.read()
            # (≤80ms away) the last; the join is instant on a clean turn and
            # capped short otherwise (daemon thread).
            if monitor_thread is not None:
                monitor_stop.set()
                monitor_thread.join(timeout=1.0)
        print()
        return False

    def process_question(
        self,
        text: str,
        language: str,
        attachments: list[dict] | None = None,
        origin: str = "voice",
        reply_audio: "Callable[[bytes], None] | None" = None,
        reply_text: "Callable[[str], None] | None" = None,
        reply_image: "Callable[[bytes, str], None] | None" = None,
        session: AudioSession | None = None,
        speaker_name: "str | None" = None,
        speaker_lang: "str | None" = None,
        vocal_cue: "str | None" = None,
    ) -> bool:
        """Run one full turn: reset/idle checks, LLM stream, TTS, persist.

        Returns True if the user barged in (M52) and the turn was cut short —
        the voice loop uses that to drop straight into a listening window.
        Every other path (normal completion, errors, the text/phone paths
        that ignore the return) yields False.

        Called from both voice path (after STT, no attachments) and text path
        (after typed Enter, optional attachments). Caller is responsible for
        setting THINKING before calling; SPEAKING is set automatically when
        TTS audio starts.

        `session` (M52) — the PC mic AudioSession, passed ONLY by the voice
        path. Its presence (plus an audible, non-restricted turn) is what
        enables the barge-in monitor: the monitor needs the mic, and only the
        voice path holds the session. Text/phone callers pass None, so the
        interrupt machinery threads through every layer as an inert no-op and
        those paths stay byte-identical to pre-M52.

        `origin` (M48.2/M48.2a): where the turn came from — the single
        honest signal two separate per-turn concerns derive from:
          - "voice"      : PC wake word     → speaks (subject to mute), full tools
          - "console"    : PC typed         → speaks (subject to mute), full tools
          - "phone_text" : phone typed      → text-only, RESTRICTED tools
          - "phone_voice": phone PTT (M48.3)→ audio→phone, RESTRICTED tools
        `text_only` and `restricted` are derived below — they are NOT the
        same axis (phone_voice will speak yet still be restricted), which is
        why the prior single `speak` bool was replaced by `origin`. The phone
        gets a deliberately reduced tool surface (no system_control /
        pc_shell / collector / plex_action; no file/screen/camera) —
        enforced server-side in stream_response, NOT prompt-only
        ([[feedback-jarvis-least-privilege]], [[feedback-diag-vs-action-split]]).

        `speaker_name` / `speaker_lang` (M80): the voice-identified speaker for
        this turn (from M69's identify on the captured clip), or None. Only the
        PC-voice path supplies them; they thread into stream_response as a small
        uncached system note so Claude knows WHO it's talking to (addresses them
        by name) and their usual language. Absent ⇒ no speaker block (unchanged
        behaviour for remote/unrecognized turns).

        When attachments is non-empty, the user message becomes a list of
        content blocks (attachments first, then a text block) instead of a
        plain string. History keeps that structure verbatim, so multi-turn
        Q&A about an attached document works naturally until reset/trim.
        """
        # M48.2/M48.2a — the two per-turn concerns, derived from one origin.
        # text_only: phone-text is text-only (M48.2); everything else speaks
        # (subject to mute). restricted: any phone origin gets the reduced
        # tool surface (M48.2a) — phone_voice (M48.3) will speak AND be
        # restricted, which is why these are separate axes off `origin`.
        # NB: deliberately NOT named `speak` — that shadowed the imported
        # speak() function and crashed the apology path ('bool' not callable).
        # "discord" (2026-06-02): a Discord-channel turn is text-only (the
        # reply posts to the channel, the PC never speaks to its empty room)
        # AND restricted (an internet-relayed, multi-human surface gets the
        # same reduced tool boundary as the phone — no system/shell/file/code).
        text_only = origin in ("phone_text", "discord")
        restricted = origin in ("phone_text", "phone_voice", "discord")

        with self._lock:
            # Seal any stale session, capture session language, surface + append
            # the user message. (Boundary logic lives in _begin_turn.)
            self._begin_turn(text, language, attachments)

            response_chunks: list[str] = []

            # Snapshot engineer-mode state once per turn — same discipline as
            # mute. Mid-turn toggle applies to the next turn (changing thinking
            # budget mid-stream isn't supported).
            engineer = self._ui.is_engineer_mode()

            def on_telemetry(rec: TelemetryRecord) -> None:
                """Called by stream_response at the end of each turn. Surfaces
                the per-turn signal that previously only existed as a stderr
                log line. Format chosen for SRE skim-readability — verb
                (which tools), then iteration count if >1, then how long,
                then how much it cost. The 'thinking' marker calls out
                engineer-mode turns since their token cost is meaningfully
                higher."""
                self._ui.add_session_tokens(rec.total_tokens)
                bits: list[str] = []
                if rec.tools_used:
                    bits.append(", ".join(rec.tools_used))
                if rec.iterations > 1:
                    bits.append(f"{rec.iterations} iters")
                if rec.thinking_enabled:
                    bits.append("thinking")
                bits.append(f"{rec.elapsed_sec:.1f}s")
                bits.append(f"{rec.total_tokens:,} tok")
                if rec.paused:
                    bits.append("PAUSED(10-iter cap)")
                self._ui.add_telemetry_chip(" · ".join(bits))

            def on_image_captured(image_bytes: bytes, media_type: str, tool_name: str) -> None:
                """Called by stream_response when a vision tool returns an
                image. Renders the bytes as an inline thumbnail in the
                console transcript so the user sees *what Jarvis saw*, not
                just the text description. Label maps tool name → emoji +
                source for at-a-glance recognition."""
                label = {
                    "camera_snapshot": "📷 webcam snapshot",
                    "screen_snapshot": "🖥 screen snapshot",
                }.get(tool_name, f"🖼 {tool_name}")
                self._ui.add_image_thumbnail(image_bytes, label)
                # M71 — relay the captured frame to the originating remote
                # surface (Discord) if a per-turn image sink was supplied.
                # The sink BUFFERS it; the photo is posted into the same thread
                # as the reply (see discord_bot._post). Only Discord sets this
                # sink (camera is clawed back for origin="discord" only), so a
                # PC/phone turn never relays its webcam frame anywhere.
                if reply_image is not None:
                    try:
                        reply_image(image_bytes, media_type)
                    except Exception as exc:  # noqa: BLE001 — never break the turn
                        print(f"[main] reply_image sink failed: {exc}",
                              file=sys.stderr)

            def llm_stream():
                # `interrupt_event` is a free variable bound just below (before
                # this generator is ever called) — None unless barge-in is
                # active for this turn, in which case stream_response polls it
                # and closes the HTTP stream when the user cuts in.
                for chunk in stream_response(
                    api_key=self._cfg.anthropic_api_key,
                    messages=self._history,
                    model=self._cfg.claude_model,
                    summaries=self._summaries,
                    plex_client=self._plex_client,
                    plex_laptop_client=self._plex_laptop_client,
                    on_complete=on_telemetry,
                    on_image_captured=on_image_captured,
                    engineer_mode=engineer,
                    restricted=restricted,
                    origin=origin,
                    speaker_name=speaker_name,
                    speaker_lang=speaker_lang,
                    vocal_cue=vocal_cue,
                    interrupt_event=interrupt_event,
                ):
                    response_chunks.append(chunk)
                    print(chunk, end="", flush=True)
                    yield chunk

            # Capture mute state at the start of the turn. If the user
            # toggles mid-response, the change applies to the NEXT turn —
            # mid-stream stop would be jarring and the streamed text is
            # already visible in the console regardless.
            muted = self._ui.is_muted()

            # M48.2: phone-text turns are text-only by origin (text_only),
            # independent of the global mute toggle. Folded into the same
            # silent-drain path mute already uses — one branch, no new code
            # path to keep correct.
            silent = muted or text_only

            # M48.2b: reply_audio set ⇒ this turn's reply audio goes to the
            # ORIGINATING PHONE, never the PC. The PC therefore drains
            # silently for it (like the mute path), and the synth-to-phone
            # below is mute-INDEPENDENT (the locked rule: the remote user
            # must hear the answer they asked for, regardless of PC mute).
            pc_silent = silent or reply_audio is not None

            # Barge-in eligibility: a PC-voice turn that actually speaks aloud
            # (we need the mic + audible playback). M88 Phase 2 — hands-free
            # talk-over (AEC duplex) takes precedence when enabled AND the mic is
            # pinned to a known device (the duplex stream needs an explicit input
            # index); otherwise fall back to M52 wake-word barge-in. Both set the
            # same interrupt_event; phone/console/muted turns get neither.
            _barge_eligible = (
                session is not None and origin == "voice" and not pc_silent
            )
            aec_barge_on = (
                _barge_eligible
                and aec_barge.is_enabled()
                and self._mic_device is not None
            )
            barge_enabled = (
                not aec_barge_on and _BARGE_IN_ENABLED and _barge_eligible
            )
            interrupt_event = (
                threading.Event() if (barge_enabled or aec_barge_on) else None
            )

            # Run the stream and (when audible) speak it — holds the speech
            # gates + barge-in monitor; on error it surfaces an apology, pops
            # the orphan user message, and returns True so we bail here.
            if self._stream_and_speak(
                llm_stream=llm_stream, pc_silent=pc_silent,
                barge_enabled=barge_enabled, session=session,
                interrupt_event=interrupt_event, language=language,
                reply_text=reply_text, aec_barge_on=aec_barge_on,
            ):
                return False

            # M52: did the user barge in? interrupt_event survives the monitor
            # join (Events don't auto-reset), so this read is stable. A
            # barged-in turn keeps whatever partial reply was streamed — it is
            # real context for the follow-up ("as I was saying...") — and
            # signals the voice loop to open a listening window at once.
            interrupted = interrupt_event is not None and interrupt_event.is_set()
            if interrupted:
                print("[main] turn interrupted by barge-in", file=sys.stderr)

            # Assemble + persist + relay the reply (and synth phone audio).
            return self._finalize_turn(
                response_chunks=response_chunks, interrupted=interrupted,
                text=text, language=language,
                reply_text=reply_text, reply_audio=reply_audio,
                speaker=speaker_name,
            )

    def interpret(self, text: str, language: str) -> None:
        """M87 — one interpreter-mode turn: translate `text` (spoken in
        `language`) into the OTHER language of the configured pair and speak it
        in that language's voice. A faithful relay — NO history, memory, tools,
        persona, or barge-in (it's not a conversation). Holds the speech gates
        for the spoken portion exactly like a normal turn (so the omni-mic
        doesn't self-capture the translation and the armed CPU loops defer).
        Fail-soft: a translation/TTS error logs and is swallowed so the
        interpreter loop keeps running. Runs under self._lock so a typed turn
        can't overlap its audio."""
        from src import interpreter as _interp  # noqa: PLC0415
        target = _interp.other_language(language)
        self._ui.add_user_text(text, language)
        print(f"\n[interpret {language}->{target}] {text}", file=sys.stderr)

        chunks: list[str] = []

        def translation_stream():
            for chunk in stream_translation(
                api_key=self._cfg.anthropic_api_key,
                text=text, target_lang=target,
                model=self._cfg.claude_model,
            ):
                chunks.append(chunk)
                yield chunk

        with self._lock:
            self._pc_speaking.set()
            self._announce_speaking.set()
            try:
                speak_streaming(
                    translation_stream(),
                    language=target,
                    on_first_audio=lambda: self._ui.set_state(State.SPEAKING),
                    on_amplitude=self._ui.set_amplitude,
                )
            except Exception as exc:  # noqa: BLE001 — never break the interpreter loop
                print(f"[interpret] failed: {exc}", file=sys.stderr)
            finally:
                self._pc_speaking.clear()
                self._announce_speaking.clear()

        translation = "".join(chunks).strip()
        if translation:
            self._ui.add_jarvis_text(translation)

    def speak_line(self, text: str, language: str = "en") -> None:
        """Speak a fixed line aloud on the calling (audio-owning) thread,
        holding the speech gates so the mic doesn't self-capture it and the
        armed CPU loops defer. Used for interpreter mode's start/stop
        confirmations. Runs under self._lock so it can't overlap a turn's audio.
        Fail-soft."""
        self._ui.add_jarvis_text(text)
        with self._lock:
            self._pc_speaking.set()
            self._announce_speaking.set()
            try:
                self._ui.set_state(State.SPEAKING)
                speak(text, language=language)
            except Exception as exc:  # noqa: BLE001
                print(f"[main] speak_line failed: {exc}", file=sys.stderr)
            finally:
                self._pc_speaking.clear()
                self._announce_speaking.clear()


def listen_loop(
    cfg: Config,
    ui: JarvisUI,
    reset_event: threading.Event,
    plex_client: PlexMCPClient | None = None,
    plex_laptop_client: PlexLaptopClient | None = None,
    security_watcher: "SecurityWatcher | None" = None,
    on_enroll_face: "Callable[[], None] | None" = None,
    on_enroll_voice: "Callable[[], None] | None" = None,
    on_knowledge_reindex: "Callable[[], None] | None" = None,
    on_knowledge_remember: "Callable[[str], None] | None" = None,
    remote_server: object | None = None,
    mic_device: int | None = None,
    pc_speaking: threading.Event | None = None,
    announce_speaking: threading.Event | None = None,
    speaker_gate: threading.Event | None = None,
) -> None:
    """Daemon worker. Owns the input loops; delegates each turn to a shared
    TurnRunner (which owns the conversation state, persists turns, and seals
    sessions on every memory boundary — manual reset / idle / app quit).

    Two input paths share ONE TurnRunner:
      - voice path: this function's main loop (wake-word → STT → runner.process_question)
      - text path: text_input_loop spawned below (queue → runner.process_question)

    The runner's lock serializes the two, so a typed message that lands while
    Jarvis is answering a spoken one waits its turn — no overlapping responses.
    """
    # The conversation engine: owns history/summaries/session state + the
    # MemoryStore and runs each turn end-to-end. Voice + text share this ONE
    # instance (its lock keeps them from overlapping). See the TurnRunner class.
    runner = TurnRunner(cfg, ui, reset_event, plex_client, plex_laptop_client,
                        pc_speaking, announce_speaking, mic_device=mic_device)

    # M85 — tonal awareness. One analyzer per session; it holds a rolling
    # loudness baseline so "softer/louder than usual" is judged against the
    # user's own voice. Only the voice path feeds it (it needs the audio clip).
    from src.voice_tone import ToneAnalyzer, is_enabled as _tone_enabled
    tone_analyzer = ToneAnalyzer() if _tone_enabled() else None

    # Text-submission queue. Tk's submit handler puts (text, attachments)
    # tuples here; the text_input_loop worker pops them and calls
    # process_question. attachments is list[tuple[str, dict]] (filename + block).
    # M48.2/M48.2b/M48.3: item is
    #   (text, attachments, origin, reply_audio, lang, reply_text, reply_image).
    # origin ∈ {"console","phone_text","phone_voice","discord"} — derives
    # PC-TTS gating (phone_*/discord don't speak on the PC) AND restricted tool
    # surface (phone + discord origins lose system/shell/file/etc.).
    # reply_audio: None (console — PC behaviour) or a conn-bound sink (phone)
    # — its presence routes this reply's audio to THAT phone instead of PC.
    # lang: M48.3 — whisper-detected ISO-639-1 for phone_voice (so Spanish-
    # spoken into the phone gets a Spanish reply + a Spanish voice); "en"
    # for typed inputs where we have no detection. Voice path on the PC
    # mic calls process_question directly (not via this queue), unaffected.
    # reply_text (2026-06-02): None, or a PER-TURN text sink (Discord) that
    # posts the reply back to the originating channel — NOT a broadcast, so a
    # PC/phone turn never leaks into the shared channel.
    # reply_image (M71): None, or a PER-TURN image sink (Discord only) that
    # buffers a webcam frame captured mid-turn so it's posted into the same
    # thread as the reply. Set ONLY for discord turns (camera_snapshot is
    # clawed back for origin="discord"); None everywhere else.
    text_queue: queue.Queue[
        tuple[
            str,
            list[tuple[str, dict]],
            str,
            "Callable[[bytes], None] | None",
            str,
            "Callable[[str], None] | None",
            "Callable[[bytes, str], None] | None",
        ]
    ] = queue.Queue()

    def text_input_loop() -> None:
        """Consume typed questions from the queue. Short timeout so we notice
        shutdown promptly without wasting cycles."""
        while not ui.shutdown.is_set():
            try:
                item = text_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if ui.shutdown.is_set():
                break
            text, attachments, origin, reply_audio, lang, reply_text, reply_image = item
            blocks = [block for _, block in attachments] if attachments else []
            print(
                f"[text-input] received: {text} (attachments={len(blocks)})",
                file=sys.stderr,
            )

            # Discord is a conversational-only remote surface: it must NOT fire
            # the local privileged shortcuts below (face/voice enrollment
            # captures the PC's OWN camera/mic — nonsensical and unsafe to
            # trigger from a chat message; the security + knowledge shortcuts
            # are likewise reserved for the physically-present / trusted
            # personal paths). A Discord turn goes straight to the LLM with the
            # restricted tool surface. Easy to relax later (e.g. deliberate
            # remote arming), but v1 keeps it strictly conversational.
            local_shortcuts = origin != "discord"

            # Same security-intent dispatch as the voice listen_loop:
            # "activate security", "stand down", and challenge-passphrase
            # responses are handled locally without calling Claude. Without
            # this, typed "activate security" gets routed to Claude, which
            # has no tool to arm the watcher and ends up calling
            # system_control to lock the workstation instead.
            if local_shortcuts and security_watcher is not None and not attachments:
                try:
                    if security_watcher.handle_transcript(text):
                        ui.add_user_text(text, lang)
                        ui.set_state(State.IDLE)
                        continue
                except Exception as exc:
                    print(f"[text-input] security.handle_transcript raised: {exc}",
                          file=sys.stderr)

            # M39: typed "enroll my face" — same dispatch as the voice path.
            # Uses the on_enroll_face kwarg (wired by main()) rather than a
            # bare name — _trigger_face_enrollment lives in main()'s scope,
            # not listen_loop's.
            if local_shortcuts and not attachments and on_enroll_face is not None:
                from src.face_auth import matches_enroll_intent  # noqa: PLC0415
                if matches_enroll_intent(text):
                    ui.add_user_text(text, lang)
                    on_enroll_face()
                    ui.set_state(State.IDLE)
                    continue

            # M69: typed voice-enrollment — the RELIABLE named path (typed
            # names beat Whisper's guesses). Named ("enroll Alice's voice",
            # "enroll voice Bob in Spanish") checked before the primary
            # ("enroll my voice"). This text path is never voice-gated.
            if local_shortcuts and not attachments and on_enroll_voice is not None:
                _named = speaker_id.parse_named_enroll_intent(text)
                if _named is not None:
                    ui.add_user_text(text, lang)
                    on_enroll_voice(_named[0], _named[1])
                    ui.set_state(State.IDLE)
                    continue
                if speaker_id.matches_enroll_intent(text):
                    ui.add_user_text(text, lang)
                    on_enroll_voice()
                    ui.set_state(State.IDLE)
                    continue

            # M45: typed knowledge-base intents — same dispatch as the voice
            # path (remember-permanently before reindex; both before Claude).
            if local_shortcuts and not attachments and (
                on_knowledge_remember is not None or on_knowledge_reindex is not None
            ):
                from src import knowledge as _kb  # noqa: PLC0415
                _fact = (_kb.extract_remember_fact(text)
                         if on_knowledge_remember is not None else None)
                if _fact is not None:
                    ui.add_user_text(text, lang)
                    on_knowledge_remember(_fact)
                    ui.set_state(State.IDLE)
                    continue
                if (on_knowledge_reindex is not None
                        and _kb.matches_reindex_intent(text)):
                    ui.add_user_text(text, lang)
                    on_knowledge_reindex()
                    ui.set_state(State.IDLE)
                    continue

            ui.set_state(State.THINKING)
            try:
                # M48.2/M48.2a: origin passed straight through — process_question
                # derives BOTH the text-only gate (phone_text doesn't speak on
                # the PC) AND the restricted tool surface (phone origins lose
                # system/shell/file/etc.) from it.
                # M48.3: language is "en" for typed input (no detection
                # available) but the whisper-detected ISO code for
                # phone_voice — so "¿qué hora es?" spoken into the phone
                # gets a Spanish reply with a Spanish voice. Claude still
                # follows the input's language via the system prompt; the
                # `language` arg drives TTS voice selection downstream.
                runner.process_question(
                    text, lang, attachments=blocks, origin=origin,
                    reply_audio=reply_audio, reply_text=reply_text,
                    reply_image=reply_image,
                )
            finally:
                ui.set_state(State.IDLE)

    # Wire the Tk submit handler. Putting on the queue is non-blocking, so
    # this returns immediately and Tk's mainloop stays responsive even while
    # an LLM stream + TTS playback is in progress. Lambda repackages the
    # (text, attachments) args into a single queue item.
    ui.set_on_text_submit(
        lambda text, attachments: text_queue.put(
            (text, attachments, "console", None, "en", None, None)  # console → no phone audio sink, no remote text/image sink; lang "en"
        )
    )

    # M48.1: now that text_queue exists, wire the remote console's converse
    # path to the SAME seam (the brain can't tell phone text from console
    # text — exactly the design intent). on_control was already wired in
    # main() where SecurityWatcher is in scope.
    if remote_server is not None:
        # M48.2b: the server passes a per-turn reply_audio sink bound to the
        # originating phone conn (or None if its "Speak replies" toggle is
        # off). reply_audio's presence IS the routing decision downstream:
        # non-None ⇒ this reply's audio goes to THAT phone, not the PC.
        remote_server.set_on_text(
            lambda t, reply_audio: text_queue.put(
                (t, [], "phone_text", reply_audio, "en", None, None)  # phone typed → "en"; text reply rides the broadcast sink; no image sink
            )
        )

        # M48.3 — phone push-to-talk. The phone records a blob (mp4 on
        # Safari, webm on others) and ships it base64-over-WS. The server
        # decodes the b64 and hands us (blob, mime, reply_audio); we must
        # transcribe off the WS thread (whisper can take seconds, must
        # never block the server loop) and then push the resulting text
        # onto the SAME text_queue the brain already drains — origin
        # "phone_voice" so the routing matrix flags it restricted (tool
        # boundary) AND silent on PC (reply_audio sinks the audio to the
        # phone instead). reply_audio is ALWAYS set for audio messages
        # (voice-in always returns voice-out — the locked M48.3 UX
        # decision; the "Speak replies" toggle continues to gate phone-
        # text only).
        def _on_phone_audio(blob: bytes, mime: str, reply_audio) -> None:
            def _worker() -> None:
                try:
                    from src.speech_to_text import transcribe_blob  # noqa: PLC0415
                    t = transcribe_blob(
                        blob, mime,
                        cfg.whisper_model, cfg.stt_server_url, cfg.stt_backend,
                    )
                    text = (t.text or "").strip()
                    if not text:
                        ui.add_system_text(
                            "(no speech detected in phone audio)"
                        )
                        return
                    # Whisper-detected language flows through to TTS voice
                    # selection — phone Spanish gets a Spanish voice reply.
                    lang = (t.language or "en").strip() or "en"
                    text_queue.put(
                        (text, [], "phone_voice", reply_audio, lang, None, None)
                    )
                except Exception as exc:  # noqa: BLE001 — never crash the WS loop's caller
                    print(
                        f"[phone-voice] transcription failed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    ui.add_system_text(
                        f"phone audio error: {type(exc).__name__}"
                    )
            threading.Thread(
                target=_worker, name="PhoneAudioSTT", daemon=True
            ).start()
        remote_server.set_on_audio(_on_phone_audio)

    # Discord bot bridge (2026-06-02) — a private channel as a two-way Jarvis
    # client, riding the SAME text_queue as the phone. Messages from
    # allowlisted users in the configured channel enqueue with origin="discord"
    # (→ restricted tools + text-only); the per-turn `reply` sink posts the
    # answer back to THAT channel (never a broadcast — no leak of PC/phone
    # turns into the shared channel). Fail-closed + graceful: needs ALL THREE
    # of token/channel/allowlist, else it doesn't start. Outbound gateway
    # connection, so no inbound port / Tailscale needed — works off-network.
    if (cfg.discord_bot_token and cfg.discord_channel_id
            and cfg.discord_allowed_user_ids):
        try:
            from src.discord_bot import DiscordBot  # noqa: PLC0415
            _discord_bot = DiscordBot(
                cfg.discord_bot_token, cfg.discord_channel_id,
                cfg.discord_allowed_user_ids,
            )
            # M71: the bot now hands us TWO per-turn sinks — reply_text (post
            # the answer) and reply_image (buffer a webcam frame so it's
            # attached to that same threaded reply). reply_image is Discord-only
            # because camera_snapshot is clawed back for origin="discord" alone.
            _discord_bot.set_on_text(
                lambda t, reply_text, reply_image: text_queue.put(
                    (t, [], "discord", None, "en", reply_text, reply_image)
                )
            )
            _discord_bot.start()
            print("[discord] bot bridge enabled", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — a bad bot must never stop Jarvis
            print(f"[discord] bot failed to start: {exc}", file=sys.stderr)
    elif cfg.discord_bot_token:
        print("[discord] token set but channel/allowlist incomplete — bot "
              "disabled", file=sys.stderr)

    text_thread = threading.Thread(target=text_input_loop, daemon=True)
    text_thread.start()

    try:
        with AudioSession(sample_rate=cfg.sample_rate, device=mic_device) as session:
            # M51: True ⇒ the previous turn just ended; listen for a follow-up
            # WITHOUT requiring "Hey Jarvis" (a longer pre-speech window;
            # silence past it falls back to wake-word mode). Starts False —
            # the first turn after startup always needs the wake word.
            followup = False
            # M52: True for the single iteration right after a barge-in. The
            # user typically says "Hey Jarvis" AND the new question in one
            # breath, so the buffered mic audio holds the start of that
            # question. We must NOT drain it (draining is correct after a
            # normal wake word, where the user pauses) — see below.
            barged = False
            # M87: interpreter mode. When set, the loop skips the wake word and
            # runs a continuous capture→translate→speak cycle (handled in the
            # dedicated branch at the top of the loop) until "stop interpreting".
            # A local Event — activation/exit both happen inside this loop, so
            # nothing outside needs a handle (a tray toggle is a clean follow-on).
            interpreter_mode = threading.Event()
            # M88: conversation mode — persistent hands-free Q&A (no wake word
            # between turns). Local Event like interpreter_mode; idle_empties
            # counts consecutive silent windows toward the auto-exit-to-standby.
            conversation_mode = threading.Event()
            idle_empties = 0
            while not ui.shutdown.is_set():
                # M52: skip the drain on the post-barge-in iteration so the
                # already-spoken follow-up survives into capture; otherwise
                # the user's words between "Hey Jarvis" and here are discarded
                # (the M52 "missed the start of my question" bug). The barge-in
                # monitor consumed the queue up to ~end of "Hey Jarvis", so
                # what remains is the question itself (plus a brief TTS-echo
                # tail Whisper tolerates). Every other iteration drains as
                # before — clearing the just-finished reply's echo.
                if barged:
                    barged = False
                else:
                    session.drain()

                # M51: a queued reset cancels any open follow-up window — drop
                # to the wake-word path below, which handles the reset. M88: a
                # reset also exits conversation mode (a clean-slate request means
                # back to standby).
                if (followup or conversation_mode.is_set()) and reset_event.is_set():
                    followup = False
                    if conversation_mode.is_set():
                        conversation_mode.clear()
                        print("[conversation] mode OFF (reset)", file=sys.stderr)

                # M87 — interpreter mode: a self-contained continuous loop that
                # supersedes the normal wake-word/follow-up path entirely. It
                # listens with NO wake word, translates whatever it hears, speaks
                # it in the other language's voice, and re-listens — until the
                # user says "stop interpreting". Kept as a separate branch (a
                # full capture cycle that `continue`s) so the normal hot path
                # below is untouched. Self-speech is already gated out of capture
                # by `pc_speaking` (set by interpret()), so the mic won't
                # transcribe Jarvis's own translation.
                if interpreter_mode.is_set():
                    followup = False  # interpreter supersedes the follow-up window
                    ui.set_state(State.LISTENING)
                    try:
                        transcript = transcribe_after_wake(
                            session,
                            model_name=cfg.whisper_model,
                            on_speech_ended=lambda: ui.set_state(State.THINKING),
                            on_amplitude=ui.set_amplitude,
                            server_url=cfg.stt_server_url,
                            backend=cfg.stt_backend,
                            max_pre_speech_sec=_INTERPRETER_WINDOW_SEC,
                            suppress_event=pc_speaking,
                        )
                    except Exception as exc:
                        print(f"[main] interpreter STT failed: {exc}",
                              file=sys.stderr)
                        continue
                    if ui.shutdown.is_set():
                        break
                    if not transcript.text:
                        # Window elapsed with no speech — stay in interpreter
                        # mode and keep listening (this is the silence cadence,
                        # not an exit).
                        ui.set_state(State.IDLE)
                        continue
                    from src import interpreter as _interp  # noqa: PLC0415
                    if _interp.is_stop_intent(transcript.text):
                        ui.add_user_text(transcript.text, transcript.language)
                        interpreter_mode.clear()
                        print("[interpreter] mode OFF (voice)", file=sys.stderr)
                        runner.speak_line(_interp.STOP_CONFIRM_EN, "en")
                        if "es" in _interp.LANG_PAIR:
                            runner.speak_line(_interp.STOP_CONFIRM_ES, "es")
                        ui.set_state(State.IDLE)
                        continue
                    runner.interpret(transcript.text, transcript.language)
                    ui.set_state(State.IDLE)
                    continue

                if followup or conversation_mode.is_set():
                    # Follow-up window (M51) OR conversation mode (M88): skip the
                    # wake word, capture (below) directly with a longer pre-speech
                    # timeout. The blue LISTENING pill is the visual cue; if the
                    # user says nothing, a follow-up window elapses to wake-word
                    # mode while conversation mode re-arms (until the idle cap).
                    ui.set_state(State.LISTENING)
                else:
                    ui.set_state(State.IDLE)
                    wait_for_wake_word(
                        session,
                        threshold=cfg.wake_word_threshold,
                        shutdown_event=ui.shutdown,
                        reset_event=reset_event,
                    )
                    if ui.shutdown.is_set():
                        break

                    # Reset clicked while we were idle-listening. Seal+clear
                    # now, without forcing another question first.
                    if reset_event.is_set():
                        if runner.has_active_conversation():
                            print("[main] conversation reset (manual; sealing active session)")
                            ui.add_system_text("conversation reset.")
                        else:
                            print("[main] reset clicked, but no active conversation to seal")
                        runner.seal_and_refresh()
                        reset_event.clear()
                        continue  # back to top — listen for next wake

                    ui.set_state(State.LISTENING)

                try:
                    transcript = transcribe_after_wake(
                        session,
                        model_name=cfg.whisper_model,
                        # Flip state the moment audio capture ends — don't let the
                        # LISTENING pill linger through Whisper's ~1-2s transcription.
                        on_speech_ended=lambda: ui.set_state(State.THINKING),
                        # M18: drive the waveform visualizer with mic amplitude
                        # while LISTENING. SPEAKING and LISTENING never overlap
                        # so the same callback safely serves both phases.
                        on_amplitude=ui.set_amplitude,
                        # M36: GPU offload (or local CPU if unset / forced).
                        # Auto-fallback by default.
                        server_url=cfg.stt_server_url,
                        backend=cfg.stt_backend,
                        # M51/M88: a follow-up turn waits longer for the user to
                        # start; conversation mode waits longer still (it re-arms
                        # on silence rather than exiting); the normal post-wake
                        # path uses the default.
                        max_pre_speech_sec=(
                            _CONVERSATION_WINDOW_SEC if conversation_mode.is_set()
                            else _FOLLOWUP_WINDOW_SEC if followup else None
                        ),
                        # 2026-05-29 omni-mic echo fix: abort + discard the
                        # capture if the PC starts speaking (a console-turn
                        # reply or a proactive announce) mid-window — so the
                        # no-wake-word follow-up window can't transcribe
                        # Jarvis's own voice as a question.
                        suppress_event=pc_speaking,
                    )
                except Exception as exc:
                    print(f"[main] STT failed: {exc}")
                    # M88: a PERSISTENT STT failure (GPU server down, mic device
                    # yanked, decode error) must not spin conversation mode
                    # forever — unlike the wake-word path it has no blocking
                    # fallback, so an exception that re-arms immediately is a hot
                    # loop. Count a failure like an empty window so the same idle
                    # auto-exit drops us back to standby; a transient blip just
                    # adds one (reset to 0 on the next real utterance).
                    if conversation_mode.is_set():
                        idle_empties += 1
                        if idle_empties >= _CONVERSATION_IDLE_EXITS:
                            print("[conversation] repeated STT failure — mode OFF, "
                                  "back to wake word\n", file=sys.stderr)
                            conversation_mode.clear()
                            idle_empties = 0
                            ui.set_state(State.IDLE)
                    followup = False
                    continue

                if not transcript.text:
                    # M88: in conversation mode, silence re-arms (stay hands-free)
                    # until enough consecutive empty windows accumulate, then
                    # auto-exit to standby so we're not listening to an empty
                    # room forever. Silent exit (no spoken note — the user has
                    # likely stepped away).
                    if conversation_mode.is_set():
                        idle_empties += 1
                        if idle_empties >= _CONVERSATION_IDLE_EXITS:
                            print("[conversation] idle — mode OFF, back to wake "
                                  "word\n", file=sys.stderr)
                            conversation_mode.clear()
                            followup = False
                            idle_empties = 0
                            ui.set_state(State.IDLE)
                        continue
                    # M51: no speech. In a follow-up window that just means
                    # the window elapsed — fall back to wake-word mode.
                    if followup:
                        print("[main] follow-up window elapsed — back to wake word\n",
                              file=sys.stderr)
                        followup = False
                    else:
                        print("[main] (no speech captured)\n")
                    continue

                # M88: heard something — reset the conversation-mode idle counter.
                idle_empties = 0

                # M69: identify the speaker on the SAME audio STT just captured.
                # Active only when at least one voice is enrolled. The per-turn
                # registry reload is sub-ms at personal scale, so a fresh
                # enrollment takes effect on the very next turn with no plumbing.
                # Phase 4: if the "voice lock" gate is ON and this is a
                # CONFIDENTLY unrecognized voice, drop the turn — don't route
                # background media / strangers to Claude. Fail-open by design:
                # the threshold sits below the user's rough-voice floor, so a
                # degraded real-user clip still 'recognizes' and never reaches
                # the drop. (The typed/console path is never gated — placed
                # BEFORE the security + enroll intents so a locked-out voice
                # can't drive them either.)
                # M80: the recognized speaker for THIS turn, threaded into the
                # LLM call so Jarvis addresses them by name + knows their usual
                # language. Resolved here (cfg is in scope): the primary enrolls
                # under the "you" sentinel, so map it to cfg.user_name for a
                # natural display name without re-enrolling the voice. None when
                # unrecognized or no registry → no speaker block downstream.
                speaker_name: "str | None" = None
                speaker_lang: "str | None" = None
                # M85: a short vocal-delivery cue (how he sounded — volume/pace/
                # pauses) from the SAME clip, threaded into the LLM call so Claude
                # can calibrate its tone. None on most turns (only notable
                # delivery produces a cue) and whenever tone awareness is off.
                vocal_cue: "str | None" = None
                if tone_analyzer is not None and transcript.audio is not None:
                    vocal_cue = tone_analyzer.analyze(transcript.audio, transcript.text)
                    if vocal_cue:
                        print(f"[tone] {vocal_cue}", file=sys.stderr)
                if transcript.audio is not None:
                    _speakers = speaker_id.load_registry(default_base_dir() / "speakers")
                    if _speakers:
                        _who = speaker_id.identify(
                            transcript.audio, _speakers, cfg.speaker_threshold)
                        if _who.recognized:
                            speaker_name = (cfg.user_name if _who.name == "you"
                                            else _who.name)
                            speaker_lang = _who.lang
                            print(f"[speaker] recognized {speaker_name} "
                                  f"(score={_who.score:.2f})", file=sys.stderr)
                        else:
                            print(f"[speaker] unrecognized voice "
                                  f"(best={_who.score:.2f})", file=sys.stderr)
                            if speaker_gate is not None and speaker_gate.is_set():
                                print("[speaker] voice-lock active → ignoring "
                                      "unrecognized turn", file=sys.stderr)
                                ui.set_state(State.IDLE)
                                followup = False
                                continue

                # Hand the transcript to the security subsystem first. It
                # consumes the turn (returns True) for both challenge
                # authentication AND activate/disarm intents — claude
                # isn't called for either. Falls through to process_question
                # on a non-match. M51: an intent turn is handled but is NOT a
                # conversational Q&A, so it ends any follow-up chain.
                if security_watcher is not None:
                    try:
                        if security_watcher.handle_transcript(transcript.text):
                            ui.add_user_text(transcript.text, transcript.language)
                            ui.set_state(State.IDLE)
                            followup = False
                            continue
                    except Exception as exc:
                        print(f"[main] security.handle_transcript raised: {exc}",
                              file=sys.stderr)

                # M39: "enroll my face" / "remember my face" intent. Checked
                # AFTER security so a challenge-time utterance can't be
                # hijacked into an enrollment; BEFORE Claude so the LLM
                # doesn't try to fulfill the intent with the wrong tool.
                if on_enroll_face is not None:
                    from src.face_auth import matches_enroll_intent  # noqa: PLC0415
                    if matches_enroll_intent(transcript.text):
                        ui.add_user_text(transcript.text, transcript.language)
                        on_enroll_face()
                        ui.set_state(State.IDLE)
                        followup = False
                        continue

                # M69: voice-enrollment intents — same placement as face-enroll
                # (after security, before Claude). NAMED ("enroll Alice's voice"
                # → a household member) is checked BEFORE the primary ("enroll
                # my voice"), because "enroll voice <name>" also matches the
                # primary pattern's bare "voice". Spoken names are error-prone
                # (Whisper) — the typed console path is the reliable one.
                if on_enroll_voice is not None:
                    _named = speaker_id.parse_named_enroll_intent(transcript.text)
                    if _named is not None:
                        ui.add_user_text(transcript.text, transcript.language)
                        on_enroll_voice(_named[0], _named[1])
                        ui.set_state(State.IDLE)
                        followup = False
                        continue
                    if speaker_id.matches_enroll_intent(transcript.text):
                        ui.add_user_text(transcript.text, transcript.language)
                        on_enroll_voice()
                        ui.set_state(State.IDLE)
                        followup = False
                        continue

                # M45: knowledge-base voice intents. AFTER face-enroll (a
                # challenge/enroll utterance must win first), BEFORE Claude
                # (so the LLM writes/refreshes the corpus instead of
                # answering from training). "remember ... permanently" is
                # tested before "update ... knowledge": it's the more
                # specific intent (requires an explicit permanence word).
                if on_knowledge_remember is not None or on_knowledge_reindex is not None:
                    from src import knowledge as _kb  # noqa: PLC0415
                    _fact = (_kb.extract_remember_fact(transcript.text)
                             if on_knowledge_remember is not None else None)
                    if _fact is not None:
                        ui.add_user_text(transcript.text, transcript.language)
                        on_knowledge_remember(_fact)
                        ui.set_state(State.IDLE)
                        followup = False
                        continue
                    if (on_knowledge_reindex is not None
                            and _kb.matches_reindex_intent(transcript.text)):
                        ui.add_user_text(transcript.text, transcript.language)
                        on_knowledge_reindex()
                        ui.set_state(State.IDLE)
                        followup = False
                        continue

                # M87 — interpreter-mode activation ("Jarvis, be my
                # interpreter"). Checked among the intents (after security /
                # enroll / knowledge so those win first) and BEFORE the LLM turn
                # so the request isn't answered conversationally. Engages the
                # continuous loop handled at the top of this loop; the start
                # confirmation is spoken in both languages so the other party
                # hears it too.
                from src import interpreter as _interp_start  # noqa: PLC0415
                if _interp_start.is_start_intent(transcript.text):
                    ui.add_user_text(transcript.text, transcript.language)
                    interpreter_mode.set()
                    # Modes are mutually exclusive: entering interpreter from
                    # conversation mode must drop conversation mode, else "stop
                    # interpreting" later would silently fall back INTO it
                    # instead of returning to the wake-word baseline.
                    conversation_mode.clear()
                    print("[interpreter] mode ON (voice)", file=sys.stderr)
                    runner.speak_line(_interp_start.START_CONFIRM_EN, "en")
                    if "es" in _interp_start.LANG_PAIR:
                        runner.speak_line(_interp_start.START_CONFIRM_ES, "es")
                    ui.set_state(State.IDLE)
                    followup = False
                    continue

                # M88 — conversation-mode toggle. If active, an explicit "exit
                # conversation" leaves the mode (natural sign-offs are handled by
                # the dismissal block below, which also exits it). If inactive,
                # "let's talk" enters it. Checked here among the intents (after
                # security/enroll/knowledge) and BEFORE the LLM turn so the
                # request isn't answered conversationally.
                from src import conversation_mode as _convo  # noqa: PLC0415
                if conversation_mode.is_set():
                    if _convo.is_stop_intent(transcript.text):
                        ui.add_user_text(transcript.text, transcript.language)
                        conversation_mode.clear()
                        followup = False
                        print("[conversation] mode OFF (voice)", file=sys.stderr)
                        runner.speak_line(
                            _convo.stop_confirmation(transcript.language),
                            transcript.language)
                        ui.set_state(State.IDLE)
                        continue
                elif _convo.is_start_intent(transcript.text):
                    ui.add_user_text(transcript.text, transcript.language)
                    conversation_mode.set()
                    interpreter_mode.clear()  # mutually exclusive (see above)
                    followup = True
                    idle_empties = 0
                    print("[conversation] mode ON (voice)", file=sys.stderr)
                    runner.speak_line(
                        _convo.start_confirmation(transcript.language),
                        transcript.language)
                    ui.set_state(State.IDLE)
                    continue

                # 2026-06-02: a FOLLOW-UP utterance that is a pure sign-off
                # ("thank you, that is all") must not run a full LLM turn. With
                # a freshly-set reminder still in context, Claude re-issued
                # set_reminder on exactly such a turn and created a DUPLICATE
                # (the "fired twice" report). The sign-off was already detected
                # AFTER the turn (to close the window); short-circuiting it
                # BEFORE the turn means a dismissal takes NO action at all.
                # Scoped to follow-up turns only — a wake-word-invoked "that's
                # all" still reaches Claude, since the user deliberately
                # summoned him. Closes cleanly and silently: _announce isn't on
                # this thread (so no gate-safe spoken reply here), and a quiet
                # close after "that's all" is the expected sign-off behaviour.
                # M88: in conversation mode every turn is hands-free, so a
                # natural sign-off here ("that's all", "goodbye") both closes the
                # turn AND exits the mode — with a brief spoken acknowledgement
                # (unlike the silent follow-up close, since the user explicitly
                # ended a back-and-forth).
                is_signoff = _is_dismissal(transcript.text)
                if (followup or conversation_mode.is_set()) and is_signoff:
                    ui.add_user_text(transcript.text, transcript.language)
                    print("[main] sign-off — closing without an LLM turn\n",
                          file=sys.stderr)
                    if conversation_mode.is_set():
                        conversation_mode.clear()
                        from src import conversation_mode as _convo_exit  # noqa: PLC0415
                        print("[conversation] mode OFF (sign-off)", file=sys.stderr)
                        runner.speak_line(
                            _convo_exit.stop_confirmation(transcript.language),
                            transcript.language)
                    ui.set_state(State.IDLE)
                    followup = False
                    continue

                # M52: pass the AudioSession so process_question can run the
                # barge-in monitor — only the voice path has the mic, so only
                # the voice path can be interrupted. Returns True iff the user
                # cut Jarvis off mid-reply with "Hey Jarvis".
                interrupted = runner.process_question(
                    transcript.text, transcript.language, session=session,
                    speaker_name=speaker_name, speaker_lang=speaker_lang,
                    vocal_cue=vocal_cue,
                )
                # M51/M52: open the follow-up window so the next utterance
                # needs no wake word. A barge-in ALWAYS opens it — the user
                # interrupted precisely to say something now, and the wake
                # word that triggered the barge-in carries no transcript to
                # sign off with, so the dismissal check is skipped on that
                # path. Otherwise (a normal completed turn) honour an explicit
                # sign-off ("that's all", "no thank you", ...) and close
                # cleanly instead of leaving the mic open for 12s.
                if interrupted:
                    print("[main] barge-in — straight into listening\n",
                          file=sys.stderr)
                    followup = True
                    barged = True  # M52: next iteration skips session.drain()
                elif is_signoff:
                    print("[main] user signed off — no follow-up window\n",
                          file=sys.stderr)
                    followup = False
                else:
                    followup = True
    finally:
        # Quit / shutdown: seal whatever's still in memory so we don't lose it.
        if runner.has_active_conversation():
            print("[memory] sealing in-progress session on shutdown...", file=sys.stderr)
            runner.seal_on_shutdown()


def _try_connect_plex(cfg: Config) -> PlexMCPClient | None:
    """M21: best-effort Plex MCP connection. Per the project contract,
    Plex is optional — any failure (no creds, server down, MCP server
    crashes, transitive dep import fail) logs and returns None. The rest
    of Jarvis runs unaffected."""
    if not cfg.plex_url or not cfg.plex_token:
        print("[plex-mcp] PLEX_URL/PLEX_TOKEN unset — Plex tools disabled", file=sys.stderr)
        return None
    try:
        client = PlexMCPClient(cfg.plex_url, cfg.plex_token)
    except Exception as exc:
        print(f"[plex-mcp] connect failed ({exc}); continuing without Plex tools", file=sys.stderr)
        return None
    print(f"[plex-mcp] connected — {len(client.tools)} tools available", file=sys.stderr)
    return client


def _try_connect_plex_laptop(cfg: Config) -> PlexLaptopClient | None:
    """M24: best-effort SSH client for the Plex laptop diagnostic tools.

    Lazy connect — we don't actually open the TCP+SSH handshake here, just
    construct the client (which validates params). The first tool call opens
    the connection. That keeps Jarvis startup fast even when the laptop is
    asleep or off-network; voice queries to other tools work normally, and
    only an actual plex_logs_* / plex_laptop_health call would hit the
    "Plex laptop unreachable" string.

    Any failure (missing host/user, bad key path, paramiko import problem)
    logs and returns None. The rest of Jarvis runs unaffected.
    """
    if not cfg.plex_laptop_host or not cfg.plex_laptop_user:
        print(
            "[plex-laptop] PLEX_LAPTOP_HOST/USER unset — remote tools disabled",
            file=sys.stderr,
        )
        return None
    key_path = cfg.plex_laptop_key_path or os.path.expanduser("~/.ssh/id_ed25519")
    log_path = cfg.plex_laptop_log_path or DEFAULT_PLEX_LAPTOP_LOG
    try:
        client = PlexLaptopClient(
            host=cfg.plex_laptop_host,
            user=cfg.plex_laptop_user,
            key_path=key_path,
            log_path=log_path,
        )
    except Exception as exc:
        print(
            f"[plex-laptop] init failed ({exc}); continuing without remote tools",
            file=sys.stderr,
        )
        return None
    print(
        f"[plex-laptop] ready — will connect to {cfg.plex_laptop_user}@{cfg.plex_laptop_host} on first tool call",
        file=sys.stderr,
    )
    return client


class _Announcer(NamedTuple):
    """Handle returned by _build_announcer: the public announce fn, the
    cooperative speech-gate Event, and a shutdown callable."""
    announce: "Callable[..., None]"
    speaking_event: threading.Event
    shutdown: "Callable[[], None]"


def _build_announcer(ui: JarvisUI, pc_speaking: threading.Event) -> _Announcer:
    """Construct the dedicated proactive-speech subsystem (M34).

    `pc_speaking` (2026-05-29) is the "PC is speaking out loud" gate shared
    with TurnRunner — set here too while an announce plays, so the always-on
    voice-capture loop discards self-audio (the omni-mic echo fix). Distinct
    from `speaking_event` below (that gates security/sound CPU bursts; this
    gates mic capture).

    Speech is marshaled through a single dedicated Announcer thread because
    sd.play() on Windows WASAPI silently no-ops when called from worker
    threads that haven't previously played audio (the SecurityWatcher thread
    hits this exact case). Pinning all sd.play() calls to one thread sidesteps
    it — see the project_wasapi_thread_audio_owner memory.

    Returns an _Announcer with `announce(text, on_done=None, label="🚨")` (the
    non-blocking entry every proactive caller uses), `speaking_event` (the
    cooperative speech gate the SecurityWatcher + SoundDetector defer their
    heavy CPU bursts against), and `shutdown()` for the wind-down sequence.

    ─────────────────────────────────────────────────────────────────────
    CONTRACT — `speaking_event` is the cooperative speech gate. It is SET
    while a proactive announce is on the wire and CLEARED when it finishes.

    Any background thread that does a sustained CPU/GIL burst (ML inference,
    a tight encode loop, a per-tick gc) MUST consume this Event and yield
    (defer or drop its work) while it is set — otherwise the burst starves
    the Python-fed TTS path, the audio buffer underruns, and the announce
    stutters. Treat it like a lock you must respect, not an optional hint.

    Current consumers: SecurityWatcher (defers grab/YOLO/encode — see
    `_is_announcing`/`_wait_for_quiet`) and SoundDetector (drops the PANNs
    inference window). A NEW continuous-CPU subsystem is not done until it is
    wired in here too — that omission is exactly the 2026-05-28 M67
    regression (M58's acoustic loop shipped outside the gate). See the
    stutter-gate post-mortem in docs/MILESTONES.md +
    project_security_audio_stutter_gate memory.
    ─────────────────────────────────────────────────────────────────────
    """
    # Queue items are (text, on_done, label) tuples; on_done fires after
    # playback so callers can defer state until the user has actually heard
    # the prompt (SecurityWatcher's 15s challenge timer uses this); label is
    # the console emoji tag (🚨 alert / ⏰ reminder). maxsize caps memory
    # growth if TTS wedges + announces pile up.
    announce_queue: queue.Queue[tuple[str, Callable[[], None] | None, str] | None] = (
        queue.Queue(maxsize=16)
    )
    announce_stop = threading.Event()
    # Set WHILE a proactive announce is actually playing. The SecurityWatcher
    # reads this to defer its heavy vision bursts (camera grab, YOLO + the
    # per-tick gc.collect, face encode) so they never overlap speech — the
    # cooperative speech gate that fixes the armed-only TTS stutter
    # (diagnosed 2026-05-19: any GIL/CPU burst overlapping the Python-fed
    # TTS path underruns the audio buffer). Owned by the Announcer thread.
    announce_speaking = threading.Event()

    def _announcer_loop() -> None:
        """Dedicated proactive-speech thread (see docstring above for why).

        Each queued item is (text, on_done, label). on_done is fired in
        `finally` after playback so it runs even if TTS itself failed — a
        failed announce shouldn't strand the caller's state machine."""
        print("[announcer] worker thread started", file=sys.stderr)
        while not announce_stop.is_set():
            item = announce_queue.get()
            if item is None or announce_stop.is_set():
                break
            text, on_done, label = item
            print(f"[announce] {text}")
            ui.add_system_text(f"{label} {text}")
            # Bracket actual playback so the watcher's cooperative gate
            # (security._is_announcing) defers heavy bursts for exactly the
            # window speech is on the wire. Cleared in finally so a TTS
            # failure can't leave the watcher permanently throttled.
            announce_speaking.set()
            pc_speaking.set()  # suppress voice-capture self-hearing of the announce
            try:
                speak_streaming(
                    iter([text]),
                    "en",
                    on_first_audio=lambda: ui.set_state(State.SPEAKING),
                    on_amplitude=ui.set_amplitude,
                )
            except Exception as exc:
                print(f"[announce] TTS failed: {exc}", file=sys.stderr)
            finally:
                announce_speaking.clear()
                pc_speaking.clear()
                ui.set_state(State.IDLE)
                if on_done is not None:
                    try:
                        on_done()
                    except Exception as exc:
                        print(f"[announce] on_done callback raised: {exc}", file=sys.stderr)
        print("[announcer] worker thread exited", file=sys.stderr)

    threading.Thread(
        target=_announcer_loop, name="Announcer", daemon=True
    ).start()

    def _announce(
        text: str,
        on_done: Callable[[], None] | None = None,
        label: str = "🚨",
    ) -> None:
        """Public entry: enqueue a proactive announcement. Non-blocking —
        returns immediately and the Announcer thread plays it. Bypasses
        the mute check (security alerts override quiet mode by design).
        Multiple announcements queue FIFO so they don't overlap.

        on_done (optional) fires AFTER the playback completes (or fails) —
        used by SecurityWatcher's challenge path to defer the 15s timer
        start until the user has actually heard the prompt, otherwise
        prompt-playback time eats into the response budget.

        label (optional) tags the console line — 🚨 for the default
        (security) announce, ⏰ for an M53 reminder firing. It is ALSO the
        severity signal the M79 quiet-hours policy reads: during the configured
        DND window a routine status-monitoring label (🖥) is held back and
        recorded for the morning briefing's catch-up, while everything else
        pierces (the safe default). DND off ⇒ unchanged behaviour."""
        if not text:
            return
        if quiet_hours.decide(label) == "defer":
            # Held back by quiet hours — record it for the briefing catch-up,
            # don't speak, don't fire on_done (on_done is a timing callback used
            # only by the critical security path, which always pierces).
            quiet_hours.record_deferred(text, label)
            print(f"[announce] quiet hours — deferred ({label}): {text!r}",
                  file=sys.stderr)
            return
        try:
            announce_queue.put_nowait((text, on_done, label))
        except queue.Full:
            # Queue is wedged (TTS playback stuck?). Log + drop newest so
            # callers don't block. on_done won't fire — callers using it
            # for timing must handle a None return, but currently the only
            # caller is SecurityWatcher's challenge path which has its
            # own defensive immediate-arm fallback.
            print(
                f"[announce] queue full (TTS wedged?) — dropping: {text!r}",
                file=sys.stderr,
            )

    def _shutdown() -> None:
        # M34: stop the Announcer thread. Sentinel-None wakes the .get() in
        # _announcer_loop so it can check the stop flag and exit cleanly,
        # instead of being killed mid-playback as a daemon.
        announce_stop.set()
        # Non-blocking: announce_stop already guarantees the loop exits on its
        # next iteration; the sentinel only needs to wake a *blocked* get(), and
        # a full queue (maxsize 16) means get() ISN'T blocked. A plain blocking
        # put() here could hang shutdown forever if the queue is full and TTS is
        # wedged in speak_streaming — the case _announce already guards with
        # put_nowait.
        try:
            announce_queue.put_nowait(None)
        except queue.Full:
            pass

    return _Announcer(_announce, announce_speaking, _shutdown)


def _build_remote_console(
    cfg: Config, ui: JarvisUI, security_watcher, announce
) -> object | None:
    """M48.1 LAN remote console. Returns the started RemoteConsoleServer, or
    None when disabled (no token) or on any startup failure.

    Gated on the token — blank ⇒ the server is NEVER constructed (a surface
    that can disarm security must not exist unless deliberately configured
    with a secret; [[feedback-jarvis-least-privilege]]). on_control is wired
    here (SecurityWatcher is in scope); on_text is wired later in listen_loop
    once the text_queue exists. Optional + defensive: a failure to start logs
    and returns None — never blocks the assistant."""
    if not cfg.remote_token:
        print("[remote] JARVIS_REMOTE_TOKEN unset — remote console disabled",
              file=sys.stderr)
        return None

    def _remote_control(action: str) -> dict:
        try:
            if action == "arm":
                security_watcher.activate()
            elif action == "disarm":
                security_watcher.deactivate()
            # "status" is read-only — just report below.
        except Exception as exc:  # noqa: BLE001 — never break on a remote ask
            print(f"[remote] control {action!r} failed: {exc}", file=sys.stderr)
        return {"armed": bool(security_watcher.is_armed())}

    # M70 — geofenced auto-arm. The PresenceController owns the policy (deferred
    # arm + flap damping + greet-on-real-transition); main.py just binds it to
    # the live SecurityWatcher and the WASAPI-safe announce path (tagged 🏠).
    # Reached via the token-gated /presence HTTP route an iOS Shortcuts
    # automation fires — same secret as the WS console.
    from src.presence import PresenceController  # noqa: PLC0415
    _presence = PresenceController(
        arm=security_watcher.activate,
        disarm=security_watcher.deactivate,
        is_armed=security_watcher.is_armed,
        announce=lambda t: announce(t, label="🏠"),
        arm_delay=cfg.presence_arm_delay,
        greeting=cfg.presence_greeting,
    )

    try:
        # Import is INSIDE the try: a missing `websockets` (or any
        # import error in remote_console/remote_pwa) must degrade to
        # "no remote console", never an unhandled ImportError that
        # breaks startup — the project's optional-component contract.
        import ssl as _ssl  # noqa: PLC0415 — lazy: only needed when TLS configured
        from src.remote_console import RemoteConsoleServer  # noqa: PLC0415

        # M48.3 prereq — opportunistic TLS: build an SSLContext ONLY if
        # both .env paths are set AND both files exist on disk. Either
        # missing or unreadable ⇒ log loudly and fall back to plain
        # HTTP/WS (LAN-mode dev path). A misconfigured cert MUST NOT
        # crash Jarvis — same optional-component contract as the rest of
        # the remote console. Stays defensively quiet about secret
        # values: only the *paths* (not contents) ever reach the log.
        ssl_ctx = None
        cf, kf = cfg.tls_cert_file, cfg.tls_key_file
        if cf or kf:
            if not (cf and kf):
                print(
                    "[remote] TLS half-configured (only one of "
                    "JARVIS_TLS_CERT_FILE/JARVIS_TLS_KEY_FILE set) — "
                    "falling back to plain HTTP/WS",
                    file=sys.stderr,
                )
            elif not (os.path.isfile(cf) and os.path.isfile(kf)):
                print(
                    f"[remote] TLS files not found "
                    f"(cert={cf!r}, key={kf!r}) — falling back to "
                    f"plain HTTP/WS",
                    file=sys.stderr,
                )
            else:
                try:
                    ssl_ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
                    ssl_ctx.load_cert_chain(certfile=cf, keyfile=kf)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[remote] TLS load failed ({exc!r}) — "
                        f"falling back to plain HTTP/WS",
                        file=sys.stderr,
                    )
                    ssl_ctx = None

        remote_server = RemoteConsoleServer(
            token=cfg.remote_token,
            host=cfg.remote_bind,
            port=cfg.remote_port,
            on_control=_remote_control,
            on_presence=_presence.handle_event,
            ssl_context=ssl_ctx,
        )
        ui.set_remote(remote_server)
        remote_server.start()
        return remote_server
    except Exception as exc:  # noqa: BLE001 — optional; must not block startup
        print(f"[remote] failed to start (continuing without it): {exc}",
              file=sys.stderr)
        return None


def _register_status(
    cfg: Config, security_watcher, sound_detector, homelab_monitor,
    plex_client, plex_laptop_client, remote_server, calendar_monitor,
    weather_monitor,
) -> None:
    """M60 — self-status registry. Each subsystem reports a one-line status
    via the `status_report` tool ("Jarvis, are you healthy?"). Order is
    display order in the report. Last-write-wins on a re-register. Cheap —
    the getters only run when status_report is called, not on every turn."""
    from src.self_status import (
        count_session_errors as _ss_errors,
        process_private_mb as _ss_mem,
        register as _ss_register,
    )

    def _status_security() -> str:
        if security_watcher.is_locked():
            return "Security: LOCKED (post-deterrent — awaiting passphrase)"
        if security_watcher.is_armed():
            return "Security: ARMED"
        return "Security: standing down"

    def _status_acoustic() -> str:
        return (f"Acoustic awareness: "
                f"{'active' if sound_detector.is_active() else 'off'}")

    def _status_homelab() -> str:
        return (f"Homelab monitor: "
                f"{'active' if homelab_monitor.is_active() else 'off'}")

    def _status_plex_mcp() -> str:
        if plex_client is None:
            return "Plex MCP: unavailable"
        try:
            n: object = len(plex_client.tool_names)
        except Exception:
            n = "?"
        return f"Plex MCP: connected ({n} tools)"

    def _status_plex_laptop() -> str:
        if plex_laptop_client is None:
            return "Plex laptop SSH: unconfigured"
        return f"Plex laptop SSH: configured ({cfg.plex_laptop_host})"

    def _status_remote() -> str:
        if remote_server is None:
            return "Remote console: off"
        tls = bool(cfg.tls_cert_file and cfg.tls_key_file)
        proto = "HTTPS/WSS" if tls else "HTTP/WS"
        return f"Remote console: listening on port {cfg.remote_port} ({proto})"

    def _status_stt() -> str:
        if cfg.stt_server_url:
            return (f"STT: GPU server ({cfg.stt_server_url}) — "
                    f"backend={cfg.stt_backend}")
        return f"STT: local CPU only — backend={cfg.stt_backend}"

    def _status_calendar() -> str:
        return (f"Calendar reminders: "
                f"{'active' if calendar_monitor.is_active() else 'off'}")

    def _status_weather() -> str:
        return (f"Weather alerts: "
                f"{'active' if weather_monitor.is_active() else 'off'}")

    def _status_reminders() -> str:
        from src.reminders import list_pending  # noqa: PLC0415 — lazy
        items = list_pending()
        n = len(items)
        # Any non-empty action is a scheduled composition (M59 briefing, M63
        # good_night, and whatever _COMPOSITION_ACTIONS grows to next) — count
        # them all, not just "briefing" (the old check drifted when M63 added
        # good_night as a second schedulable action).
        scheduled = sum(1 for r in items if r.get("action"))
        if scheduled:
            return f"Reminders: {n} pending ({scheduled} scheduled briefing/wrap-up(s))"
        return f"Reminders: {n} pending"

    def _status_memory() -> str:
        return f"Process memory: {_ss_mem():.0f} MB private bytes"

    def _status_errors() -> str:
        n, ctx = _ss_errors()
        return f"Log errors: {n} concerning lines ({ctx})"

    _ss_register("Security", _status_security)
    _ss_register("Acoustic awareness", _status_acoustic)
    _ss_register("Homelab monitor", _status_homelab)
    _ss_register("Plex MCP", _status_plex_mcp)
    _ss_register("Plex laptop", _status_plex_laptop)
    _ss_register("Remote console", _status_remote)
    _ss_register("STT backend", _status_stt)
    _ss_register("Reminders", _status_reminders)
    _ss_register("Calendar reminders", _status_calendar)
    _ss_register("Weather alerts", _status_weather)
    _ss_register("Process memory", _status_memory)
    _ss_register("Log errors", _status_errors)


def _shutdown_subsystems(
    *, security_watcher, homelab_monitor, sound_detector, calendar_monitor,
    weather_monitor, anticipation_engine,
    announcer: _Announcer, reminder_stop: threading.Event,
    worker: threading.Thread, plex_client, plex_laptop_client,
) -> None:
    """Wind down all background subsystems in order, join the listen loop (so
    it seals the active session to disk), then close the Plex connections.
    Does NOT handle relaunch — that's control flow (sys.exit) and stays in
    main(). Every daemon thread dies with the process regardless; the explicit
    shutdowns let in-flight work finish cleanly without log noise."""
    # M34: signal the security watcher to wind down.
    security_watcher.shutdown()
    # M56: stop the homelab monitor's poll loop.
    homelab_monitor.shutdown()
    # M58: stop acoustic awareness (closes the InputStream + inference loop).
    sound_detector.shutdown()
    # M62.2: stop the calendar reminder monitor.
    calendar_monitor.shutdown()
    # M83: stop the anticipation synthesis loop.
    anticipation_engine.shutdown()
    # M77: stop the severe-weather alerts monitor.
    weather_monitor.shutdown()
    # M34: stop the Announcer thread (sets stop + wakes its blocked get()).
    announcer.shutdown()
    # M53: stop the reminder scheduler.
    reminder_stop.set()

    # Give listen_loop time to see the shutdown event, exit its loop cleanly,
    # and run its try/finally — which seals the active session to disk. Worst
    # case: user quit mid-recording and we wait up to ~15s for max-recording
    # to time out, then a summarize call. M16 hardening: bumped from 20s → 30s
    # so the summarize round-trip (capped at 8s in memory.py, no SDK retries)
    # plus any tail TTS playback comfortably fits before we give up.
    worker.join(timeout=30.0)
    if worker.is_alive():
        print("[main] listen_loop didn't exit in time — session may not be sealed", file=sys.stderr)

    # M21: clean up MCP subprocess. Done after worker.join so any in-flight
    # tool call inside listen_loop has a chance to complete first.
    if plex_client is not None:
        try:
            plex_client.close()
        except Exception as exc:
            print(f"[plex-mcp] close failed: {exc}", file=sys.stderr)

    # M24: tear down the persistent SSH connection. Idempotent + non-blocking;
    # paramiko handles the channel cleanup internally.
    if plex_laptop_client is not None:
        try:
            plex_laptop_client.close()
        except Exception as exc:
            print(f"[plex-laptop] close failed: {exc}", file=sys.stderr)


def main() -> None:
    log_path = setup_logging()

    cfg = load()
    if not cfg.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY missing. Add it to .env and try again.", file=sys.stderr)
        sys.exit(1)

    # M21: spin up Plex MCP before the UI. Synchronous; takes a few seconds
    # the first time as plex-mcp-server initializes its plexapi connection.
    # Failure is non-fatal — we simply run without Plex tools.
    plex_client = _try_connect_plex(cfg)

    # M24: prepare the SSH client to the Plex laptop. This is lazy — no
    # network until the first tool call — so it adds zero startup latency
    # even when the laptop is off.
    plex_laptop_client = _try_connect_plex_laptop(cfg)

    # M41: log elevation status to jarvis.log on every startup. Makes it
    # trivial to grep "was this session elevated?" without digging through
    # tool-call traces. The mutating verbs in system_control (M40 —
    # flush_dns, restart_service) only work when this is True.
    from src.autostart import is_admin as _is_admin  # noqa: PLC0415
    print(f"[main] running elevated: {_is_admin()}")

    print("Jarvis ready. Tray icon active — left-click to show window. Say 'Hey Jarvis' to begin.\n")

    reset_event = threading.Event()

    def _on_reset() -> None:
        # Fires on pystray's menu thread. Just signals — actual clear happens
        # in listen_loop after the next wake word + STT, so the click applies
        # to the very next question (not the one after).
        reset_event.set()
        print("[tray] reset queued — applies to your next 'Hey Jarvis'")

    # Memory dir = %LOCALAPPDATA%\Jarvis (parent of sessions/ and summaries.jsonl)
    # — exposed on the tray so the user can browse past transcripts in one click.
    ui = JarvisUI(log_path=log_path, memory_dir=default_base_dir())
    ui.set_on_reset(_on_reset)

    # Status bar bootstrap. Model + integration dots are set once at startup;
    # token counter accumulates as turns happen. Plex/laptop status reflects
    # whether the optional integrations were configured + connected — not
    # whether the underlying network is currently reachable (lazy connect on
    # the laptop side means we don't probe until first use).
    ui.set_model_name(cfg.claude_model)
    ui.set_integration("plex", plex_client is not None)
    ui.set_integration("laptop", plex_laptop_client is not None)

    # SecurityWatcher is constructed below; import it here (lazy — heavy
    # transitive deps via ultralytics/cv2).
    from src.security import SecurityWatcher

    # "PC is speaking out loud" gate (2026-05-29 omni-mic echo fix). SET while
    # ANY PC TTS plays — proactive announces (Announcer) AND turn replies
    # (TurnRunner) — so the always-on voice-capture loop discards self-audio
    # instead of transcribing Jarvis's own reply. Distinct from
    # `announce_speaking` (which gates security/sound CPU bursts): this gates
    # mic capture, and covers turn replies too, not just announces.
    pc_speaking = threading.Event()

    # Proactive-speech subsystem: a dedicated WASAPI-safe Announcer thread.
    # _build_announcer returns the non-blocking announce() entry every
    # proactive caller uses, the cooperative speech-gate Event (the
    # SecurityWatcher + SoundDetector defer heavy CPU bursts against it), and
    # a shutdown() callable for the wind-down. See _build_announcer.
    _ann = _build_announcer(ui, pc_speaking)
    _announce = _ann.announce
    announce_speaking = _ann.speaking_event

    # M53: reminders & timers. The scheduler thread polls reminders.json and
    # fires due reminders through _announce — a reminder is a proactive
    # announce, so it rides the same WASAPI-safe Announcer path as a security
    # alert. Bound to the ⏰ label so reminders read as reminders in the
    # console, not 🚨 alerts. Daemon thread; reminder_stop ends it cleanly at
    # shutdown. The local import matches the SecurityWatcher pattern above.
    # A reminder also pushes to Discord on fire (2026-06-02) so it reaches the
    # user when away from the PC — the dual-channel pattern the security /
    # homelab alerts already use. Gated on BOTH a configured webhook AND the
    # opt-out flag; None when either is off, so the scheduler simply skips the
    # remote sink (no wasted POST per fire). No presence detection, so it
    # always pushes — redundant-but-harmless when home.
    from src.reminders import run_scheduler as _run_reminder_scheduler
    from src.notifications import send_discord_message as _send_discord_message
    _reminder_webhook = (
        cfg.discord_webhook_url if cfg.reminder_discord_enabled else ""
    )
    _reminder_notify = (
        (lambda t: _send_discord_message(_reminder_webhook, t))
        if _reminder_webhook else None
    )
    if _reminder_notify is not None:
        print("[reminders] Discord push on fire: enabled", file=sys.stderr)
    reminder_stop = threading.Event()
    threading.Thread(
        target=_run_reminder_scheduler,
        args=(lambda t: _announce(t, label="⏰"), reminder_stop),
        kwargs={"notify": _reminder_notify},
        name="ReminderScheduler",
        daemon=True,
    ).start()

    # M35: pass the configured passphrase (empty → CHALLENGE step skipped,
    # M34 announce-only behavior) and the evidence directory (where
    # deterrent-fired triggering frames get saved as JPEGs).
    # M38: Discord webhook URL + SMTP credentials. Both channels fire in
    # parallel on deterrent; either being blank simply disables that channel.
    # Reuse cameras.py's CAMERA_INDEX env reader so the watcher honors the
    # same override as camera_snapshot — otherwise users with CAMERA_INDEX
    # set get the right webcam for vision queries but the wrong one for
    # security mode.
    from src.cameras import _camera_index as _resolve_camera_index

    security_watcher = SecurityWatcher(
        announce=_announce,
        on_armed_changed=ui.set_armed_indicator,
        on_locked_changed=ui.set_locked_indicator,
        camera_index=_resolve_camera_index(),
        passphrase=cfg.security_passphrase,
        evidence_dir=default_base_dir() / "security" / "events",
        discord_webhook_url=cfg.discord_webhook_url,
        smtp_host=cfg.smtp_host,
        smtp_port=cfg.smtp_port,
        smtp_username=cfg.smtp_username,
        smtp_password=cfg.smtp_password,
        smtp_to=cfg.smtp_to,
        # M39: face-recognition auth path. Encoding (if enrolled) lives
        # alongside the deterrent evidence at %LOCALAPPDATA%/Jarvis/security/.
        # SecurityWatcher loads it lazily on each activate() so re-enrollment
        # mid-session takes effect without a restart.
        face_encoding_path=default_base_dir() / "security" / "face_encoding.npy",
        face_match_threshold=cfg.face_match_threshold,
        # Cooperative speech gate (2026-05-19): the watcher defers heavy
        # vision work for any tick this is set, so a YOLO/encode/warm burst
        # can't starve the Python-fed TTS path and stutter an announce.
        speaking_event=announce_speaking,
    )

    # M71: when armed, the watcher owns the webcam (persistent capture). Let a
    # camera_snapshot (Discord, while away) borrow a frame from it instead of
    # opening a contending handle that would only grab black. Disarmed ⇒ the
    # provider returns None ⇒ cameras.py opens its own (the camera is free).
    from src import cameras as _cameras  # noqa: PLC0415
    _cameras.set_armed_frame_provider(security_watcher.grab_frame_for_snapshot)

    # M39: face-enrollment trigger. Shared by the tray menu "Enroll my face"
    # AND the voice intent ("Jarvis, enroll my face"). Non-blocking — queues
    # the announce prompt and returns; capture + enrollment run later on the
    # Announcer thread via on_done. Both triggers reach the same flow.
    def _trigger_face_enrollment() -> None:
        from src import face_auth as _face_auth  # noqa: PLC0415 — lazy
        from src import cameras as _cameras  # noqa: PLC0415
        _face_auth.run_voice_enrollment(
            _announce,
            _cameras.capture_frames,
            default_base_dir() / "security" / "face_encoding.npy",
        )

    ui.set_on_enroll_face(_trigger_face_enrollment)

    # M69: voice-enrollment trigger (speaker ID). Mirrors face enrollment —
    # shared by the tray "Enroll my voice" and the voice intent "Jarvis,
    # enroll my voice". Records a few CONSECUTIVE clips from the pinned mic
    # (the user talks continuously) which enroll_from_audio averages into one
    # embedding. The capture opens its own short-lived recording stream — a
    # second stream on the mic device, the same coexistence M58 relies on —
    # so it doesn't race the listen loop's AudioSession.
    def _capture_voice_clips(num_clips: int, clip_seconds: float):
        import sounddevice as _sd  # noqa: PLC0415 — lazy
        clips = []
        try:
            for _ in range(num_clips):
                rec = _sd.rec(int(clip_seconds * cfg.sample_rate),
                              samplerate=cfg.sample_rate, channels=1,
                              dtype="int16", device=mic_device_index)
                _sd.wait()
                clips.append(rec[:, 0].copy())
        except Exception as exc:  # noqa: BLE001 — defensive against mic errors
            print(f"[speaker_id] enrollment capture failed: {exc}", file=sys.stderr)
            return None
        return clips

    def _trigger_voice_enrollment(name: str | None = None, lang: str = "en") -> None:
        # name=None ⇒ the primary user (the no-arg voice intent / tray item);
        # a name ⇒ a named household member (the typed "enroll Alice's voice").
        speaker_id.run_voice_enrollment(
            _announce,
            _capture_voice_clips,
            default_base_dir() / "speakers",
            name=name or cfg.user_name,
            lang=lang,
        )

    ui.set_on_enroll_voice(_trigger_voice_enrollment)

    # M69 Phase 4: the opt-in "voice lock" gate as a runtime-toggleable Event
    # (tray + JARVIS_SPEAKER_GATE). SET = only enrolled voices are answered; a
    # confidently-unrecognized VOICE turn is dropped (the text/console path is
    # never gated — physical access implies authorization). The tray toggle
    # flips it live, no restart.
    #
    # Persisted across restarts (2026-06-02): a user who locks the mic expects
    # it to STAY locked next launch. Initial state = the persisted ui_state
    # flag if present, else the env default (JARVIS_SPEAKER_GATE) for a fresh
    # install; the tray toggle writes the flag so the choice is sticky.
    from src import ui_state  # noqa: PLC0415 — local import, main.py convention
    speaker_gate = threading.Event()
    if ui_state.get_flag("speaker_gate", cfg.speaker_gate_enabled):
        speaker_gate.set()
        print("[speaker] voice-lock gate restored ON", file=sys.stderr)

    def _toggle_speaker_gate() -> None:
        if speaker_gate.is_set():
            speaker_gate.clear()
            ui_state.set_flag("speaker_gate", False)
            print("[speaker] voice-lock gate OFF", file=sys.stderr)
        else:
            speaker_gate.set()
            ui_state.set_flag("speaker_gate", True)
            print("[speaker] voice-lock gate ON", file=sys.stderr)

    ui.set_on_speaker_gate_toggle(_toggle_speaker_gate, speaker_gate.is_set)

    # M45: knowledge-base triggers. Shared by the tray ("Reindex knowledge")
    # AND the voice intents ("Jarvis, update your knowledge" / "remember this
    # permanently: ..."). Non-blocking: the reindex / file-write is pure
    # disk+sqlite I/O and runs on a throwaway daemon thread, but the spoken
    # result MUST go through _announce (the WASAPI constraint — proactive
    # speech only via the Announcer thread, never speak_streaming from a
    # fresh thread). See project_wasapi_thread_audio_owner memory.
    def _trigger_knowledge_reindex() -> None:
        from src import knowledge as _knowledge  # noqa: PLC0415 — lazy

        def _work() -> None:
            try:
                result = _knowledge.reindex()
                _announce(result.message)
            except Exception as exc:  # defensive — never strand the trigger
                print(f"[knowledge] reindex trigger failed: {exc}", file=sys.stderr)
                _announce("I couldn't update my knowledge, sir.")

        threading.Thread(target=_work, daemon=True, name="kb-reindex").start()

    def _trigger_knowledge_remember(fact: str) -> None:
        from src import knowledge as _knowledge  # noqa: PLC0415 — lazy

        def _work() -> None:
            try:
                _announce(_knowledge.remember_fact(fact))
            except Exception as exc:
                print(f"[knowledge] remember trigger failed: {exc}", file=sys.stderr)
                _announce("I couldn't save that, sir.")

        threading.Thread(target=_work, daemon=True, name="kb-remember").start()

    ui.set_on_reindex_knowledge(_trigger_knowledge_reindex)
    # Wire the tray's Security-mode toggle to the watcher. Tray was already
    # constructed inside ui.run()'s worker thread, but the toggle's `checked`
    # callback re-evaluates each menu open, so this late wiring is fine.
    ui.set_on_security_toggle(
        on_activate=security_watcher.activate,
        on_deactivate=security_watcher.deactivate,
        is_armed=security_watcher.is_armed,
    )

    # M56: proactive homelab monitoring. Constructed ALWAYS (the
    # homelab_status tool + the tray toggle need the instance) but the
    # background poll loop only runs once activate()d. Default off — opt-in
    # via JARVIS_HOMELAB_MONITOR or the tray toggle, the same safe-default,
    # least-privilege stance as security mode. Reuses the existing
    # plex_laptop_client (its run() is lock-serialised, so the monitor thread
    # safely shares the one SSH connection the tool path uses) and the M38
    # Discord webhook for phone push. Spoken alerts ride _announce (the
    # WASAPI-safe Announcer path), tagged 🖥 so they read distinctly from
    # 🚨 security alerts and ⏰ reminders in the console.
    from src.homelab_monitor import HomelabMonitor
    homelab_monitor = HomelabMonitor(
        announce=lambda t: _announce(t, label="🖥"),
        plex_host=cfg.plex_laptop_host,
        plex_laptop_client=plex_laptop_client,
        discord_webhook_url=cfg.discord_webhook_url,
    )
    if cfg.homelab_monitor_enabled:
        homelab_monitor.activate(announce=False)  # silent at boot (status pills suffice)
    ui.set_on_homelab_toggle(
        on_activate=homelab_monitor.activate,
        on_deactivate=homelab_monitor.deactivate,
        is_active=homelab_monitor.is_active,
    )

    # M58: acoustic awareness. Constructed ALWAYS (the tray toggle needs the
    # instance) but the audio capture + inference loop only runs once
    # activate()d. Default off — opt-in via JARVIS_ACOUSTIC_MONITOR or the
    # tray toggle, same safe-default + least-privilege stance as security
    # mode and the homelab monitor. Owns its own sd.InputStream at 32 kHz
    # (the AudioSession is single-reader-by-design — wake-word loop owns it
    # during listening, M52 barge-in during TTS — so M58 captures
    # independently). Reuses the M38 Discord webhook; tagged 🔔 on the
    # Announcer so alerts read distinctly from 🚨 / 🖥 / ⏰.
    # Mic device pin (JARVIS_MIC_DEVICE). Resolve ONCE here so the main
    # capture and acoustic awareness bind the same physical input and we log
    # the chosen device a single time. None ⇒ Windows default (legacy).
    mic_device_index = resolve_input_device(cfg.mic_device)

    from src.sound_detector import SoundDetector
    # M72 — multimodal acoustic alert: when Jarvis hears something WHILE ARMED,
    # he also looks through the camera and pushes a photo + a one-line Claude
    # description to Discord. Gated by the armed frame provider —
    # grab_frame_for_snapshot returns None unless armed, so this no-ops at home
    # (no camera light for every doorbell) and only fires for the away case it's
    # built for. Runs on the SoundDetector's own AcousticVisual thread; every
    # step is fail-soft so a missing key / network blip / no camera just skips
    # the visual and the text alert (already sent) stands.
    def _acoustic_visual_alert(event_name: str) -> None:
        if not cfg.discord_webhook_url:
            return
        frame = security_watcher.grab_frame_for_snapshot()
        if frame is None:
            return  # not armed → the watcher isn't holding the camera; skip
        jpeg = _cameras.encode_jpeg(frame)
        if not jpeg:
            return
        from src.vision_describe import describe_scene  # noqa: PLC0415
        from src.notifications import send_discord_photo  # noqa: PLC0415
        desc = describe_scene(
            jpeg, event_name,
            api_key=cfg.anthropic_api_key, model=cfg.claude_model,
        )
        pretty = event_name.replace("_", " ")
        caption = f"👁 {desc}" if desc else f"👁 I heard {pretty} — here's the room, sir."
        send_discord_photo(cfg.discord_webhook_url, caption, jpeg,
                           image_filename="acoustic.jpg")

    sound_detector = SoundDetector(
        announce=lambda t: _announce(t, label="🔔"),
        discord_webhook_url=cfg.discord_webhook_url,
        device=mic_device_index,
        on_visual_alert=_acoustic_visual_alert,
        # M81 — armed intrusion-by-voice: the voice_while_armed rule only counts
        # windows while away. Same getter good_night + the visual hook use.
        is_armed=security_watcher.is_armed,
        # Cooperative speech gate (same Event the SecurityWatcher uses): the
        # PANNs inference loop defers while a proactive announce plays so it
        # can't starve the TTS path. M58 coupled acoustic to armed mode but
        # left this loop ungated — the 2026-05-28 armed-stutter regression.
        speaking_event=announce_speaking,
    )
    # M76 — expose the detector's recent-sounds buffer to the what_did_you_hear
    # tool (the self_status.register decoupling pattern).
    from src.sound_detector import register_detector as _register_sound_detector
    _register_sound_detector(sound_detector)
    if cfg.acoustic_monitor_enabled:
        sound_detector.activate()
    ui.set_on_acoustic_toggle(
        on_activate=sound_detector.activate,
        on_deactivate=sound_detector.deactivate,
        is_active=sound_detector.is_active,
    )

    # M62.2 — pre-event proactive calendar reminders. Watches the Outlook
    # calendar in the background and announces an event a fixed lead time
    # (default 15 min) before it starts: "Sir — your 2 pm standup in 15
    # minutes." The proactive layer on top of the M62 read tool — same
    # reactive→proactive jump M56 made for the homelab. Constructed ALWAYS
    # (so the status registration always has a target); activate() is a
    # logged no-op when calendar isn't configured or the env kill switch is
    # set. Unlike M56/M58 (default off, opt-in) this is DEFAULT ON when
    # calendar is configured — the user already opted in by setting up
    # OUTLOOK_ICAL_URL, the proactive layer IS the
    # value-add. Kill via JARVIS_CALENDAR_REMINDERS=0. Spoken alerts ride
    # _announce (the WASAPI-safe Announcer path), tagged 📅 so they read
    # distinctly from 🚨 / 🖥 / 🔔 / ⏰ in the console.
    from src.calendar_monitor import CalendarMonitor
    calendar_monitor = CalendarMonitor(
        announce=lambda t: _announce(t, label="📅"),
        discord_webhook_url=cfg.discord_webhook_url,
    )
    calendar_monitor.activate()  # internally a no-op when not configured / disabled

    # M77 — severe-weather proactive alerts. Watches the NWS active-alerts feed
    # for the home location and warns on its own before a power-threatening
    # storm (tied to the no-UPS power-loss situation). Default ON when
    # JARVIS_HOME_LOCATION is set; no-op otherwise. Tagged ⛈.
    from src.weather_alerts import WeatherAlertMonitor
    weather_monitor = WeatherAlertMonitor(
        announce=lambda t: _announce(t, label="⛈"),
        discord_webhook_url=cfg.discord_webhook_url,
    )
    weather_monitor.activate()  # internally a no-op when not configured / disabled

    # M83 — anticipatory intelligence ("Sir, I've taken the liberty…"). The
    # SYNTHESIS layer over every single-signal monitor above: a background pass
    # fuses the live world-state (calendar + weather + alerts + reminders +
    # homelab + security) and asks Claude, as an extremely-selective chief-of-
    # staff, whether ONE cross-domain insight is worth surfacing. Tagged 🧠 so it
    # DEFERS in quiet hours (→ the morning catch-up). Default OFF — it makes a
    # recurring (small) LLM call, so it's opt-in via JARVIS_ANTICIPATION=1.
    from src.anticipation import AnticipationEngine, is_enabled as _anticip_enabled
    from src import briefing as _brief
    from src.weather_alerts import execute_weather_alerts_tool as _wx_alerts

    def _anticipation_snapshot() -> dict:
        """Compose the live world-state for the engine — each section fail-soft
        (a hiccup just drops that section). Reuses the briefing gatherers so the
        weather/reminders/calendar wording stays consistent across surfaces."""
        def _safe(fn):
            try:
                v = fn()
                return v.strip() if isinstance(v, str) else v
            except Exception as exc:  # noqa: BLE001 — a section must never break the tick
                print(f"[anticipation] snapshot section failed: {exc}", file=sys.stderr)
                return None
        snap: dict = {
            "Calendar": _safe(_brief._calendar_section),
            "Weather": _safe(_brief._weather_section),
            "Reminders": _safe(_brief._reminders_section),
        }
        alerts = _safe(lambda: _wx_alerts({}))
        if alerts and "no active" not in alerts.lower() and "isn't configured" not in alerts.lower():
            snap["Weather alerts"] = alerts
        if homelab_monitor.is_active():
            snap["Homelab"] = _safe(homelab_monitor.status_report)
        try:
            snap["Security"] = ("Armed — Saul is away from home"
                                if security_watcher.is_armed()
                                else "Disarmed — home")
        except Exception:  # noqa: BLE001
            pass
        return snap

    anticipation_engine = AnticipationEngine(
        announce=lambda t: _announce(t, label="🧠"),
        snapshot_fn=_anticipation_snapshot,
        api_key=cfg.anthropic_api_key,
        model=cfg.claude_model,
    )
    if _anticip_enabled():
        anticipation_engine.activate()
    from src.self_status import register as _ss_anticip_reg
    _ss_anticip_reg("Anticipation",
                    lambda: "active" if anticipation_engine.is_active() else "off")

    # M63 — "good night" wrap. Wire the security-state getter so the
    # composition tool can report current armed/standing-down state without
    # needing security_watcher threaded through stream_response. Same
    # decoupling pattern as homelab_monitor's _set_active_monitor — the tool
    # finds the live state through a module-level singleton.
    from src.good_night import register_security_getter as _register_gn_security
    _register_gn_security(security_watcher.is_armed)

    # M64 — self-update tool. Wire the restart trigger so a successful
    # `update_jarvis` (after the confirmation gate) can drive the standard
    # M32 restart path. `ui._handle_restart` sets relaunch_mode + signals
    # shutdown — the same thing the tray's "Restart Jarvis" click does.
    # Same decoupling pattern as the security-getter above.
    from src.self_update import register_restart_callback as _register_su_restart
    _register_su_restart(ui._handle_restart)

    # M58 follow-up — couple acoustic awareness to security mode. The user's
    # mental model is "armed = away from home, where Jarvis should also be
    # listening for non-speech events"; this chains sound_detector.activate
    # / deactivate onto the security-armed edge. The independent tray toggle
    # for acoustic still works for "I want it on while I'm home" cases — the
    # coupling is one-way (security state drives acoustic, never the
    # reverse). Activate/deactivate are idempotent, so a manual flip
    # followed by a coupled flip never double-fires.
    def _security_armed_changed(armed: bool) -> None:
        ui.set_armed_indicator(armed)
        try:
            if armed:
                sound_detector.activate()
            else:
                sound_detector.deactivate()
        except Exception as exc:
            print(f"[main] coupling acoustic to security failed: {exc}",
                  file=sys.stderr)
    security_watcher.set_on_armed_changed(_security_armed_changed)

    # M48.1: LAN remote console (token-gated, optional). Returns the started
    # server or None. on_text is wired later in listen_loop once the
    # text_queue exists (the late-injection pattern).
    remote_server = _build_remote_console(cfg, ui, security_watcher, _announce)

    # M60 — self-status registry: each subsystem reports a one-line status via
    # the `status_report` tool. Registered with the live subsystem handles.
    _register_status(
        cfg, security_watcher, sound_detector, homelab_monitor,
        plex_client, plex_laptop_client, remote_server, calendar_monitor,
        weather_monitor,
    )

    # M69 — if any voice is enrolled, warm the speaker-ID model in the
    # background now so the first identified turn doesn't eat the ~5-15s
    # first-call JIT (mirrors face_auth's cold-start warming). No enrolled
    # voice ⇒ skip entirely (no torch load, no cost).
    if speaker_id.load_registry(default_base_dir() / "speakers"):
        threading.Thread(target=speaker_id.warm, name="SpeakerWarm",
                         daemon=True).start()

    worker = threading.Thread(
        target=listen_loop,
        args=(cfg, ui, reset_event, plex_client, plex_laptop_client, security_watcher),
        kwargs={
            "on_enroll_face": _trigger_face_enrollment,
            "on_enroll_voice": _trigger_voice_enrollment,
            "on_knowledge_reindex": _trigger_knowledge_reindex,
            "on_knowledge_remember": _trigger_knowledge_remember,
            "remote_server": remote_server,
            "mic_device": mic_device_index,
            "pc_speaking": pc_speaking,
            "announce_speaking": announce_speaking,
            "speaker_gate": speaker_gate,
        },
        daemon=True,
    )
    worker.start()

    ui.run()  # blocks main thread on Tk's mainloop until Quit is clicked

    # Wind down all background subsystems, join the listen loop (so it seals
    # the active session to disk), and close the Plex connections. Relaunch is
    # handled below — it's control flow (sys.exit), not subsystem teardown.
    _shutdown_subsystems(
        security_watcher=security_watcher,
        homelab_monitor=homelab_monitor,
        sound_detector=sound_detector,
        calendar_monitor=calendar_monitor,
        weather_monitor=weather_monitor,
        anticipation_engine=anticipation_engine,
        announcer=_ann,
        reminder_stop=reminder_stop,
        worker=worker,
        plex_client=plex_client,
        plex_laptop_client=plex_laptop_client,
    )

    # Restart-Jarvis tray click sets ui.relaunch_mode. We defer the actual
    # respawn until here so the mic + Plex MCP + SSH have all released
    # their handles — the new instance starts on a clean slate.
    #
    # M41: tristate dispatch. "normal" uses subprocess.Popen (silent,
    # always works). "elevated" uses ShellExecuteW(verb="runas") which
    # fires UAC — if the user clicks No, no new instance starts and
    # this process still exits (acceptable "occasional sudo" UX).
    mode = ui.relaunch_mode
    if mode is not None:
        # M65 — watchdog cooperation. When `jarvis_watchdog.pyw` is the
        # parent process, it sets `JARVIS_WATCHDOG=1` in our env; on a
        # NORMAL restart we exit with code 42 ("please respawn me") and
        # let the watchdog handle the spawn. This keeps the watchdog as
        # the sole spawner of main.py — if we ALSO spawned, two instances
        # would race for the audio device. Elevated restart goes through
        # ShellExecuteW("runas") regardless, since the watchdog can't
        # spawn an elevated child; the elevated instance becomes
        # un-watched (documented limitation, acceptable for "occasional
        # sudo" UX). See [[project-m65-crash-watchdog]].
        under_watchdog = os.environ.get("JARVIS_WATCHDOG", "").strip() == "1"
        try:
            from src import autostart
            if mode == "elevated":
                ok = autostart.relaunch_elevated()
                if ok:
                    print("[main] relaunched elevated Jarvis instance")
                else:
                    print(
                        "[main] elevated relaunch was cancelled or failed — "
                        "Jarvis will not restart",
                        file=sys.stderr,
                    )
                # Either way the watchdog should NOT respawn — the
                # elevated child takes over (success) or the user
                # cancelled (no relaunch wanted). Exit 0.
            elif under_watchdog:
                # The watchdog parent will see exit code 42 and respawn.
                print("[main] under watchdog — exiting 42 for respawn")
                sys.exit(42)
            else:  # "normal", no watchdog
                autostart.relaunch()
                print("[main] relaunched detached Jarvis instance")
        except SystemExit:
            raise  # the sys.exit(42) path above; don't swallow it
        except Exception as exc:
            print(f"[main] relaunch failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
