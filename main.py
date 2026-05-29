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
from typing import Callable

from src.audio import AudioSession, resolve_input_device
from src.config import Config, load
from src.llm import TelemetryRecord, stream_response
from src.memory import MemoryStore, SummaryRecord, default_base_dir, summarize_session
from src.plex_laptop import DEFAULT_LOG_PATH as DEFAULT_PLEX_LAPTOP_LOG, PlexLaptopClient
from src.plex_mcp import PlexMCPClient
from src.speech_to_text import transcribe_after_wake
from src.text_to_speech import speak, speak_streaming
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
    from src.logfile import rotate_if_needed

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
        sys.stdout = log_file
        sys.stderr = log_file
    else:
        # Console mode — tee both. Live terminal output + persistent file.
        sys.stdout = _TeeStream(sys.stdout, log_file)
        sys.stderr = _TeeStream(sys.stderr, log_file)

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


def listen_loop(
    cfg: Config,
    ui: JarvisUI,
    reset_event: threading.Event,
    plex_client: PlexMCPClient | None = None,
    plex_laptop_client: PlexLaptopClient | None = None,
    security_watcher: "SecurityWatcher | None" = None,
    on_enroll_face: "Callable[[], None] | None" = None,
    on_knowledge_reindex: "Callable[[], None] | None" = None,
    on_knowledge_remember: "Callable[[str], None] | None" = None,
    remote_server: object | None = None,
    mic_device: int | None = None,
) -> None:
    """Daemon worker. Owns conversation history, persists turns + seals
    sessions on every memory boundary (manual reset / idle / app quit).

    Two input paths share state through closures:
      - voice path: this function's main loop (wake-word → STT → process_question)
      - text path: text_input_loop spawned below (queue → process_question)

    Both call process_question, which holds processing_lock for the duration of
    the LLM stream + TTS. That means a typed message that lands while Jarvis is
    answering a spoken one waits its turn — no overlapping responses.
    """
    history: list[dict] = []
    last_turn_time = 0.0

    memory = MemoryStore()
    memory.prune(retain_raw_days=cfg.retain_raw_days)
    summaries = memory.recent_summaries(cfg.memory_recall_count)
    print(f"[memory] loaded {len(summaries)} prior session summaries", file=sys.stderr)

    # Tracks the language and start-time of the *currently active* session
    # (i.e., the one being built up in `history`). Reset on every seal.
    session_language = "en"
    session_started_at = datetime.now().isoformat(timespec="seconds")

    # Serializes process_question so voice and text paths never run together.
    processing_lock = threading.Lock()

    # Text-submission queue. Tk's submit handler puts (text, attachments)
    # tuples here; the text_input_loop worker pops them and calls
    # process_question. attachments is list[tuple[str, dict]] (filename + block).
    # M48.2/M48.2b/M48.3: item is (text, attachments, origin, reply_audio, lang).
    # origin ∈ {"console","phone_text","phone_voice"} — derives PC-TTS gating
    # (phone_text/phone_voice don't speak on the PC) AND restricted tool
    # surface (phone origins lose system/shell/file/etc.).
    # reply_audio: None (console — PC behaviour) or a conn-bound sink (phone)
    # — its presence routes this reply's audio to THAT phone instead of PC.
    # lang: M48.3 — whisper-detected ISO-639-1 for phone_voice (so Spanish-
    # spoken into the phone gets a Spanish reply + a Spanish voice); "en"
    # for typed inputs where we have no detection. Voice path on the PC
    # mic calls process_question directly (not via this queue), unaffected.
    text_queue: queue.Queue[
        tuple[
            str,
            list[tuple[str, dict]],
            str,
            "Callable[[bytes], None] | None",
            str,
        ]
    ] = queue.Queue()

    def seal_and_refresh() -> None:
        """Seal the active session (if any) and reload summaries for the next one."""
        nonlocal summaries, session_started_at
        _seal_session(memory, history, session_language, session_started_at, cfg)
        history.clear()
        summaries = memory.recent_summaries(cfg.memory_recall_count)
        session_started_at = datetime.now().isoformat(timespec="seconds")

    def process_question(
        text: str,
        language: str,
        attachments: list[dict] | None = None,
        origin: str = "voice",
        reply_audio: "Callable[[bytes], None] | None" = None,
        session: AudioSession | None = None,
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

        When attachments is non-empty, the user message becomes a list of
        content blocks (attachments first, then a text block) instead of a
        plain string. History keeps that structure verbatim, so multi-turn
        Q&A about an attached document works naturally until reset/trim.
        """
        nonlocal session_language, session_started_at, last_turn_time

        # M48.2/M48.2a — the two per-turn concerns, derived from one origin.
        # text_only: phone-text is text-only (M48.2); everything else speaks
        # (subject to mute). restricted: any phone origin gets the reduced
        # tool surface (M48.2a) — phone_voice (M48.3) will speak AND be
        # restricted, which is why these are separate axes off `origin`.
        # NB: deliberately NOT named `speak` — that shadowed the imported
        # speak() function and crashed the apology path ('bool' not callable).
        text_only = origin == "phone_text"
        restricted = origin in ("phone_text", "phone_voice")

        with processing_lock:
            # Apply any pending reset/idle boundary BEFORE this turn — so the
            # user's question starts a fresh session rather than tacking onto
            # a stale one.
            if reset_event.is_set():
                if history:
                    print(f"[main] conversation reset (manual; sealing {len(history)} msgs)")
                    ui.add_system_text("conversation reset.")
                seal_and_refresh()
                reset_event.clear()
            elif history and (time.time() - last_turn_time) > IDLE_RESET_SEC:
                print(f"[main] conversation reset (idle >{IDLE_RESET_SEC:.0f}s)")
                ui.add_system_text("conversation reset (idle).")
                seal_and_refresh()

            # First turn of a (possibly new) session — capture its language.
            if not history:
                session_language = language or "en"
                session_started_at = datetime.now().isoformat(timespec="seconds")

            print(f"\n[user, {language}] {text}")
            print("[jarvis] ", end="", flush=True)
            ui.add_user_text(text, language)

            # Build user-message content. With attachments: list of blocks
            # (each attachment first so Claude has the doc/image in context
            # before the question text). Without: plain string (cheaper).
            if attachments:
                content: list[dict] | str = list(attachments)
                if text:
                    content.append({"type": "text", "text": text})
            else:
                content = text

            history.append({"role": "user", "content": content})

            response_chunks: list[str] = []

            # Snapshot engineer-mode state once per turn — same discipline as
            # mute. Mid-turn toggle applies to the next turn (changing thinking
            # budget mid-stream isn't supported).
            engineer = ui.is_engineer_mode()

            def on_telemetry(rec: TelemetryRecord) -> None:
                """Called by stream_response at the end of each turn. Surfaces
                the per-turn signal that previously only existed as a stderr
                log line. Format chosen for SRE skim-readability — verb
                (which tools), then iteration count if >1, then how long,
                then how much it cost. The 'thinking' marker calls out
                engineer-mode turns since their token cost is meaningfully
                higher."""
                ui.add_session_tokens(rec.total_tokens)
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
                ui.add_telemetry_chip(" · ".join(bits))

            def on_image_captured(image_bytes: bytes, media_type: str, tool_name: str) -> None:  # noqa: ARG001 — media_type unused for now
                """Called by stream_response when a vision tool returns an
                image. Renders the bytes as an inline thumbnail in the
                console transcript so the user sees *what Jarvis saw*, not
                just the text description. Label maps tool name → emoji +
                source for at-a-glance recognition."""
                label = {
                    "camera_snapshot": "📷 webcam snapshot",
                    "screen_snapshot": "🖥 screen snapshot",
                }.get(tool_name, f"🖼 {tool_name}")
                ui.add_image_thumbnail(image_bytes, label)

            def llm_stream():
                # `interrupt_event` is a free variable bound just below (before
                # this generator is ever called) — None unless barge-in is
                # active for this turn, in which case stream_response polls it
                # and closes the HTTP stream when the user cuts in.
                for chunk in stream_response(
                    api_key=cfg.anthropic_api_key,
                    messages=history,
                    model=cfg.claude_model,
                    summaries=summaries,
                    plex_client=plex_client,
                    plex_laptop_client=plex_laptop_client,
                    on_complete=on_telemetry,
                    on_image_captured=on_image_captured,
                    engineer_mode=engineer,
                    restricted=restricted,
                    interrupt_event=interrupt_event,
                ):
                    response_chunks.append(chunk)
                    print(chunk, end="", flush=True)
                    yield chunk

            # Capture mute state at the start of the turn. If the user
            # toggles mid-response, the change applies to the NEXT turn —
            # mid-stream stop would be jarring and the streamed text is
            # already visible in the console regardless.
            muted = ui.is_muted()

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

            # M52 — barge-in. Enabled only for a PC-voice turn that actually
            # speaks aloud: we need the mic AudioSession (only the voice path
            # passes it) and audible playback to interrupt. phone/console/
            # muted turns leave interrupt_event None, so it threads through
            # llm_stream → stream_response → speak_streaming as an inert
            # no-op and those paths are byte-identical to pre-M52.
            barge_enabled = (
                _BARGE_IN_ENABLED
                and session is not None
                and origin == "voice"
                and not pc_silent
            )
            interrupt_event = threading.Event() if barge_enabled else None
            monitor_stop = threading.Event()
            monitor_thread: threading.Thread | None = None

            try:
                if pc_silent:
                    # Drain the LLM stream silently. response_chunks gets
                    # populated inside llm_stream via the print + append, so
                    # the rest of the function (full_response assembly,
                    # history append, persist) works unchanged. State stays
                    # THINKING throughout — caller flips to IDLE after we
                    # return.
                    for _ in llm_stream():
                        pass
                else:
                    # M52: run the barge-in monitor for the duration of
                    # playback. It reads the mic on its own thread and, on a
                    # "Hey Jarvis", just sets interrupt_event — speak_streaming
                    # polls it and performs the sd.stop() cut on its own
                    # (stream-owning) thread, the WASAPI thread-affinity fix.
                    if barge_enabled:
                        monitor_thread = threading.Thread(
                            target=monitor_for_wake_word,
                            args=(session, interrupt_event, monitor_stop),
                            kwargs={"threshold": cfg.wake_word_threshold},
                            name="BargeInMonitor",
                            daemon=True,
                        )
                        monitor_thread.start()
                    speak_streaming(
                        llm_stream(),
                        language=language,
                        on_first_audio=lambda: ui.set_state(State.SPEAKING),
                        on_amplitude=ui.set_amplitude,
                        interrupt_event=interrupt_event,
                    )
            except Exception as exc:
                history.pop()  # keep history alternating user/assistant cleanly
                print(f"\n[main] LLM/TTS failed: {exc}")
                # M20: don't leave the user in silence after a partial reply.
                # Speak a brief apology in their language via the simpler Tier
                # A path (no streaming pipeline that could fail again). Wrap
                # in defensive try/except so the apology itself can't break
                # the loop. Add to transcript too so the console reflects it.
                # When silent (muted OR phone-text origin), skip the spoken
                # apology but still surface its text — the phone/console sees
                # the hiccup, the PC stays quiet to its empty room.
                apology = (
                    "Disculpe, tuve un problema técnico. ¿Podría intentarlo de nuevo?"
                    if language == "es"
                    else "Apologies, a technical hiccup. Could you try that again?"
                )
                if not pc_silent:
                    ui.set_state(State.SPEAKING)
                    try:
                        speak(apology, language=language)
                    except Exception as apology_exc:
                        print(f"[main] apology TTS also failed: {apology_exc}")
                # Phone-audio turns: the apology TEXT still reaches the phone
                # via the fan-out below; we deliberately don't synth audio on
                # the error path (extra failure surface for little gain — the
                # text is enough to convey the hiccup).
                ui.add_jarvis_text(apology)
                return False
            finally:
                # M52: always wind the barge-in monitor down — normal end,
                # exception, OR barge-in. monitor_stop makes its next
                # session.read() (≤80ms away) the last; the join is instant
                # on a clean turn and capped short otherwise (daemon thread).
                if monitor_thread is not None:
                    monitor_stop.set()
                    monitor_thread.join(timeout=1.0)
            print()

            # M52: did the user barge in? interrupt_event survives the monitor
            # join (Events don't auto-reset), so this read is stable. A
            # barged-in turn keeps whatever partial reply was streamed — it is
            # real context for the follow-up ("as I was saying...") — and
            # signals the voice loop to open a listening window at once.
            interrupted = interrupt_event is not None and interrupt_event.is_set()
            if interrupted:
                print("[main] turn interrupted by barge-in", file=sys.stderr)

            full_response = "".join(response_chunks).strip()
            if not full_response:
                history.pop()  # nothing came back; drop the orphan user message
                return interrupted

            history.append({"role": "assistant", "content": full_response})
            _trim_history(history)
            last_turn_time = time.time()

            # Persist the completed exchange to today's transcript file.
            memory.record_turn(text, full_response, language)

            ui.add_jarvis_text(full_response)

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
                        DEFAULT_VOICE, VOICE_BY_LANG, _fetch_mp3,
                    )

                    voice = VOICE_BY_LANG.get(language, DEFAULT_VOICE)
                    mp3 = asyncio.run(_fetch_mp3(full_response, voice))
                    reply_audio(mp3)
                except Exception as exc:  # noqa: BLE001
                    print(f"[main] phone reply audio failed: {exc}",
                          file=sys.stderr)
            print()
            return interrupted

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
            text, attachments, origin, reply_audio, lang = item
            blocks = [block for _, block in attachments] if attachments else []
            print(
                f"[text-input] received: {text} (attachments={len(blocks)})",
                file=sys.stderr,
            )

            # Same security-intent dispatch as the voice listen_loop:
            # "activate security", "stand down", and challenge-passphrase
            # responses are handled locally without calling Claude. Without
            # this, typed "activate security" gets routed to Claude, which
            # has no tool to arm the watcher and ends up calling
            # system_control to lock the workstation instead.
            if security_watcher is not None and not attachments:
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
            if not attachments and on_enroll_face is not None:
                from src.face_auth import matches_enroll_intent  # noqa: PLC0415
                if matches_enroll_intent(text):
                    ui.add_user_text(text, lang)
                    on_enroll_face()
                    ui.set_state(State.IDLE)
                    continue

            # M45: typed knowledge-base intents — same dispatch as the voice
            # path (remember-permanently before reindex; both before Claude).
            if not attachments and (
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
                process_question(
                    text, lang, attachments=blocks, origin=origin,
                    reply_audio=reply_audio,
                )
            finally:
                ui.set_state(State.IDLE)

    # Wire the Tk submit handler. Putting on the queue is non-blocking, so
    # this returns immediately and Tk's mainloop stays responsive even while
    # an LLM stream + TTS playback is in progress. Lambda repackages the
    # (text, attachments) args into a single queue item.
    ui.set_on_text_submit(
        lambda text, attachments: text_queue.put(
            (text, attachments, "console", None, "en")  # console → no phone audio sink; lang "en" (no detection on typed)
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
                (t, [], "phone_text", reply_audio, "en")  # phone typed → "en" (no detection on typed)
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
                        (text, [], "phone_voice", reply_audio, lang)
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
                # to the wake-word path below, which handles the reset.
                if followup and reset_event.is_set():
                    followup = False

                if followup:
                    # Follow-up window: skip the wake word, capture (below)
                    # directly with a longer pre-speech timeout. The blue
                    # LISTENING pill is the visual cue; if the user says
                    # nothing, the window simply elapses.
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
                        if history:
                            print(f"[main] conversation reset (manual; sealing {len(history)} msgs)")
                            ui.add_system_text("conversation reset.")
                        else:
                            print("[main] reset clicked, but no active conversation to seal")
                        seal_and_refresh()
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
                        # M51: a follow-up turn waits longer for the user to
                        # start; the normal post-wake path uses the default.
                        max_pre_speech_sec=(
                            _FOLLOWUP_WINDOW_SEC if followup else None
                        ),
                    )
                except Exception as exc:
                    print(f"[main] STT failed: {exc}")
                    followup = False
                    continue

                if not transcript.text:
                    # M51: no speech. In a follow-up window that just means
                    # the window elapsed — fall back to wake-word mode.
                    if followup:
                        print("[main] follow-up window elapsed — back to wake word\n",
                              file=sys.stderr)
                        followup = False
                    else:
                        print("[main] (no speech captured)\n")
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

                # M52: pass the AudioSession so process_question can run the
                # barge-in monitor — only the voice path has the mic, so only
                # the voice path can be interrupted. Returns True iff the user
                # cut Jarvis off mid-reply with "Hey Jarvis".
                interrupted = process_question(
                    transcript.text, transcript.language, session=session,
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
                elif _is_dismissal(transcript.text):
                    print("[main] user signed off — no follow-up window\n",
                          file=sys.stderr)
                    followup = False
                else:
                    followup = True
    finally:
        # Quit / shutdown: seal whatever's still in memory so we don't lose it.
        if history:
            print("[memory] sealing in-progress session on shutdown...", file=sys.stderr)
            _seal_session(memory, history, session_language, session_started_at, cfg)


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

    # SecurityWatcher with a proactive-speech callback. Speech is marshaled
    # through a single dedicated Announcer thread because sd.play() on
    # Windows WASAPI silently no-ops when called from worker threads that
    # haven't previously played audio (the SecurityWatcher thread hits this
    # exact case). Pinning all sd.play() calls to one thread sidesteps it.
    from src.security import SecurityWatcher
    from src.text_to_speech import speak_streaming

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
        """Dedicated proactive-speech thread (see comment above for why).

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
                ui.set_state(State.IDLE)
                if on_done is not None:
                    try:
                        on_done()
                    except Exception as exc:
                        print(f"[announce] on_done callback raised: {exc}", file=sys.stderr)
        print("[announcer] worker thread exited", file=sys.stderr)

    announcer_thread = threading.Thread(
        target=_announcer_loop, name="Announcer", daemon=True
    )
    announcer_thread.start()

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
        (security) announce, ⏰ for an M53 reminder firing."""
        if not text:
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

    # M53: reminders & timers. The scheduler thread polls reminders.json and
    # fires due reminders through _announce — a reminder is a proactive
    # announce, so it rides the same WASAPI-safe Announcer path as a security
    # alert. Bound to the ⏰ label so reminders read as reminders in the
    # console, not 🚨 alerts. Daemon thread; reminder_stop ends it cleanly at
    # shutdown. The local import matches the SecurityWatcher pattern above.
    from src.reminders import run_scheduler as _run_reminder_scheduler
    reminder_stop = threading.Event()
    threading.Thread(
        target=_run_reminder_scheduler,
        args=(lambda t: _announce(t, label="⏰"), reminder_stop),
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
        homelab_monitor.activate()
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
    sound_detector = SoundDetector(
        announce=lambda t: _announce(t, label="🔔"),
        discord_webhook_url=cfg.discord_webhook_url,
        device=mic_device_index,
        # Cooperative speech gate (same Event the SecurityWatcher uses): the
        # PANNs inference loop defers while a proactive announce plays so it
        # can't starve the TTS path. M58 coupled acoustic to armed mode but
        # left this loop ungated — the 2026-05-28 armed-stutter regression.
        speaking_event=announce_speaking,
    )
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
    # OUTLOOK_ICAL_URL / OUTLOOK_CLIENT_ID, the proactive layer IS the
    # value-add. Kill via JARVIS_CALENDAR_REMINDERS=0. Spoken alerts ride
    # _announce (the WASAPI-safe Announcer path), tagged 📅 so they read
    # distinctly from 🚨 / 🖥 / 🔔 / ⏰ in the console.
    from src.calendar_monitor import CalendarMonitor
    calendar_monitor = CalendarMonitor(
        announce=lambda t: _announce(t, label="📅"),
        discord_webhook_url=cfg.discord_webhook_url,
    )
    calendar_monitor.activate()  # internally a no-op when not configured / disabled

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

    # M48.1: LAN remote console. Gated on the token — blank ⇒ the server is
    # NEVER constructed (a surface that can disarm security must not exist
    # unless deliberately configured with a secret;
    # [[feedback-jarvis-least-privilege]]). on_control is wired here
    # (SecurityWatcher is in scope); on_text is wired later in listen_loop
    # once the text_queue exists (the on_enroll_face / on_knowledge_*
    # late-injection pattern). Optional + defensive: a failure to start
    # logs and continues — never blocks the assistant.
    remote_server = None
    if cfg.remote_token:

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
                ssl_context=ssl_ctx,
            )
            ui.set_remote(remote_server)
            remote_server.start()
        except Exception as exc:  # noqa: BLE001 — optional; must not block startup
            print(f"[remote] failed to start (continuing without it): {exc}",
                  file=sys.stderr)
            remote_server = None
    else:
        print("[remote] JARVIS_REMOTE_TOKEN unset — remote console disabled",
              file=sys.stderr)

    # M60 — self-status registry. Each subsystem reports a one-line status
    # via the `status_report` tool ("Jarvis, are you healthy?"). Registered
    # HERE so every closure captures the live locals; order is display order
    # in the report. Last-write-wins on a re-register (safe under any future
    # re-construction). Cheap — the getters only run when status_report is
    # called, not on every turn.
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

    def _status_reminders() -> str:
        from src.reminders import list_pending  # noqa: PLC0415 — lazy
        items = list_pending()
        n = len(items)
        briefings = sum(1 for r in items if r.get("action") == "briefing")
        if briefings:
            return f"Reminders: {n} pending ({briefings} scheduled briefing(s))"
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
    _ss_register("Process memory", _status_memory)
    _ss_register("Log errors", _status_errors)

    worker = threading.Thread(
        target=listen_loop,
        args=(cfg, ui, reset_event, plex_client, plex_laptop_client, security_watcher),
        kwargs={
            "on_enroll_face": _trigger_face_enrollment,
            "on_knowledge_reindex": _trigger_knowledge_reindex,
            "on_knowledge_remember": _trigger_knowledge_remember,
            "remote_server": remote_server,
            "mic_device": mic_device_index,
        },
        daemon=True,
    )
    worker.start()

    ui.run()  # blocks main thread on Tk's mainloop until Quit is clicked

    # M34: signal the security watcher to wind down. It's a daemon thread
    # so it dies with the process either way, but explicit shutdown lets
    # any in-flight inference complete cleanly without log noise.
    security_watcher.shutdown()

    # M56: stop the homelab monitor's poll loop. Daemon thread, so it dies
    # with the process anyway; the event lets it exit its poll-wait cleanly.
    homelab_monitor.shutdown()

    # M58: stop acoustic awareness. Closes the InputStream + signals the
    # inference loop to exit. Daemon-threaded so this is just clean wind-down.
    sound_detector.shutdown()

    # M62.2: stop the calendar reminder monitor. Daemon thread; the event
    # lets it exit its poll-wait cleanly without log noise.
    calendar_monitor.shutdown()

    # M34: stop the Announcer thread. Sentinel-None wakes the .get() in
    # _announcer_loop so it can check the stop flag and exit cleanly,
    # instead of being killed mid-playback as a daemon.
    announce_stop.set()
    # Non-blocking: announce_stop already guarantees the loop exits on its
    # next iteration; the sentinel only needs to wake a *blocked* get(), and
    # a full queue (maxsize 16) means get() ISN'T blocked. A plain blocking
    # put() here could hang shutdown forever if the queue is full and TTS is
    # wedged in speak_streaming — the same "TTS wedged" case _announce already
    # guards against with put_nowait.
    try:
        announce_queue.put_nowait(None)
    except queue.Full:
        pass

    # M53: stop the reminder scheduler. Daemon thread, so it dies with the
    # process either way; the event lets it exit its poll-wait cleanly.
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
