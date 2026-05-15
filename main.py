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
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.audio import AudioSession
from src.config import Config, load
from src.llm import TelemetryRecord, stream_response
from src.memory import MemoryStore, SummaryRecord, default_base_dir, summarize_session
from src.plex_laptop import DEFAULT_LOG_PATH as DEFAULT_PLEX_LAPTOP_LOG, PlexLaptopClient
from src.plex_mcp import PlexMCPClient
from src.speech_to_text import transcribe_after_wake
from src.text_to_speech import speak, speak_streaming
from src.tray import State
from src.ui import JarvisUI
from src.wake_word import wait_for_wake_word


MAX_PAIRS = 10            # cap conversation at 10 exchanges (20 messages)
IDLE_RESET_SEC = 600.0    # 10 min of silence → forget conversation


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


def listen_loop(
    cfg: Config,
    ui: JarvisUI,
    reset_event: threading.Event,
    plex_client: PlexMCPClient | None = None,
    plex_laptop_client: PlexLaptopClient | None = None,
    security_watcher: "SecurityWatcher | None" = None,
    on_enroll_face: "Callable[[], None] | None" = None,
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
    text_queue: queue.Queue[tuple[str, list[tuple[str, dict]]]] = queue.Queue()

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
    ) -> None:
        """Run one full turn: reset/idle checks, LLM stream, TTS, persist.

        Called from both voice path (after STT, no attachments) and text path
        (after typed Enter, optional attachments). Caller is responsible for
        setting THINKING before calling; SPEAKING is set automatically when
        TTS audio starts.

        When attachments is non-empty, the user message becomes a list of
        content blocks (attachments first, then a text block) instead of a
        plain string. History keeps that structure verbatim, so multi-turn
        Q&A about an attached document works naturally until reset/trim.
        """
        nonlocal session_language, session_started_at, last_turn_time

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
                ):
                    response_chunks.append(chunk)
                    print(chunk, end="", flush=True)
                    yield chunk

            # Capture mute state at the start of the turn. If the user
            # toggles mid-response, the change applies to the NEXT turn —
            # mid-stream stop would be jarring and the streamed text is
            # already visible in the console regardless.
            muted = ui.is_muted()

            try:
                if muted:
                    # Drain the LLM stream silently. response_chunks gets
                    # populated inside llm_stream via the print + append, so
                    # the rest of the function (full_response assembly,
                    # history append, persist) works unchanged. State stays
                    # THINKING throughout — caller flips to IDLE after we
                    # return.
                    for _ in llm_stream():
                        pass
                else:
                    speak_streaming(
                        llm_stream(),
                        language=language,
                        on_first_audio=lambda: ui.set_state(State.SPEAKING),
                        on_amplitude=ui.set_amplitude,
                    )
            except Exception as exc:
                history.pop()  # keep history alternating user/assistant cleanly
                print(f"\n[main] LLM/TTS failed: {exc}")
                # M20: don't leave the user in silence after a partial reply.
                # Speak a brief apology in their language via the simpler Tier
                # A path (no streaming pipeline that could fail again). Wrap
                # in defensive try/except so the apology itself can't break
                # the loop. Add to transcript too so the console reflects it.
                # When muted, skip TTS but still surface the apology text in
                # the console — visual feedback that something went wrong.
                apology = (
                    "Disculpe, tuve un problema técnico. ¿Podría intentarlo de nuevo?"
                    if language == "es"
                    else "Apologies, a technical hiccup. Could you try that again?"
                )
                if not muted:
                    ui.set_state(State.SPEAKING)
                    try:
                        speak(apology, language=language)
                    except Exception as apology_exc:
                        print(f"[main] apology TTS also failed: {apology_exc}")
                ui.add_jarvis_text(apology)
                return
            print()

            full_response = "".join(response_chunks).strip()
            if not full_response:
                history.pop()  # nothing came back; drop the orphan user message
                return

            history.append({"role": "assistant", "content": full_response})
            _trim_history(history)
            last_turn_time = time.time()

            # Persist the completed exchange to today's transcript file.
            memory.record_turn(text, full_response, language)

            ui.add_jarvis_text(full_response)
            print()

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
            text, attachments = item
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
                        ui.add_user_text(text, "en")
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
                    ui.add_user_text(text, "en")
                    on_enroll_face()
                    ui.set_state(State.IDLE)
                    continue

            ui.set_state(State.THINKING)
            try:
                # Hardcoded 'en' for now — Claude still replies in the input's
                # language thanks to the system prompt, but TTS picks the
                # English voice. Add language detection here if it becomes a
                # real Spanish-typed-input use case.
                process_question(text, "en", attachments=blocks)
            finally:
                ui.set_state(State.IDLE)

    # Wire the Tk submit handler. Putting on the queue is non-blocking, so
    # this returns immediately and Tk's mainloop stays responsive even while
    # an LLM stream + TTS playback is in progress. Lambda repackages the
    # (text, attachments) args into a single queue item.
    ui.set_on_text_submit(
        lambda text, attachments: text_queue.put((text, attachments))
    )

    text_thread = threading.Thread(target=text_input_loop, daemon=True)
    text_thread.start()

    try:
        with AudioSession(sample_rate=cfg.sample_rate) as session:
            while not ui.shutdown.is_set():
                ui.set_state(State.IDLE)
                session.drain()

                wait_for_wake_word(
                    session,
                    threshold=cfg.wake_word_threshold,
                    shutdown_event=ui.shutdown,
                    reset_event=reset_event,
                )
                if ui.shutdown.is_set():
                    break

                # Reset clicked while we were idle-listening. Seal+clear now,
                # without forcing the user to ask another question first.
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
                    )
                except Exception as exc:
                    print(f"[main] STT failed: {exc}")
                    continue

                if not transcript.text:
                    print("[main] (no speech captured)\n")
                    continue

                # Hand the transcript to the security subsystem first. It
                # consumes the turn (returns True) for both challenge
                # authentication AND activate/disarm intents — claude
                # isn't called for either. Falls through to process_question
                # on a non-match.
                if security_watcher is not None:
                    try:
                        if security_watcher.handle_transcript(transcript.text):
                            ui.add_user_text(transcript.text, transcript.language)
                            ui.set_state(State.IDLE)
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
                        continue

                process_question(transcript.text, transcript.language)
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

    # Queue items are (text, on_done) tuples; on_done fires after playback
    # so callers can defer state until the user has actually heard the
    # prompt (SecurityWatcher's 15s challenge timer uses this). maxsize
    # caps memory growth if TTS wedges + announces pile up.
    announce_queue: queue.Queue[tuple[str, Callable[[], None] | None] | None] = (
        queue.Queue(maxsize=16)
    )
    announce_stop = threading.Event()

    def _announcer_loop() -> None:
        """Dedicated proactive-speech thread (see comment above for why).

        Each queued item is (text, on_done). on_done is fired in `finally`
        after playback so it runs even if TTS itself failed — a failed
        announce shouldn't strand the caller's state machine."""
        print("[announcer] worker thread started", file=sys.stderr)
        while not announce_stop.is_set():
            item = announce_queue.get()
            if item is None or announce_stop.is_set():
                break
            text, on_done = item
            print(f"[announce] {text}")
            ui.add_system_text(f"🚨 {text}")
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

    def _announce(text: str, on_done: Callable[[], None] | None = None) -> None:
        """Public entry: enqueue a proactive announcement. Non-blocking —
        returns immediately and the Announcer thread plays it. Bypasses
        the mute check (security alerts override quiet mode by design).
        Multiple announcements queue FIFO so they don't overlap.

        on_done (optional) fires AFTER the playback completes (or fails) —
        used by SecurityWatcher's challenge path to defer the 15s timer
        start until the user has actually heard the prompt, otherwise
        prompt-playback time eats into the response budget."""
        if not text:
            return
        try:
            announce_queue.put_nowait((text, on_done))
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
    # Wire the tray's Security-mode toggle to the watcher. Tray was already
    # constructed inside ui.run()'s worker thread, but the toggle's `checked`
    # callback re-evaluates each menu open, so this late wiring is fine.
    ui.set_on_security_toggle(
        on_activate=security_watcher.activate,
        on_deactivate=security_watcher.deactivate,
        is_armed=security_watcher.is_armed,
    )

    worker = threading.Thread(
        target=listen_loop,
        args=(cfg, ui, reset_event, plex_client, plex_laptop_client, security_watcher),
        kwargs={"on_enroll_face": _trigger_face_enrollment},
        daemon=True,
    )
    worker.start()

    ui.run()  # blocks main thread on Tk's mainloop until Quit is clicked

    # M34: signal the security watcher to wind down. It's a daemon thread
    # so it dies with the process either way, but explicit shutdown lets
    # any in-flight inference complete cleanly without log noise.
    security_watcher.shutdown()

    # M34: stop the Announcer thread. Sentinel-None wakes the .get() in
    # _announcer_loop so it can check the stop flag and exit cleanly,
    # instead of being killed mid-playback as a daemon.
    announce_stop.set()
    announce_queue.put(None)

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
            else:  # "normal"
                autostart.relaunch()
                print("[main] relaunched detached Jarvis instance")
        except Exception as exc:
            print(f"[main] relaunch failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
