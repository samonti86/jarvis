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

from src.audio import AudioSession
from src.config import Config, load
from src.llm import stream_response
from src.memory import MemoryStore, SummaryRecord, default_base_dir, summarize_session
from src.speech_to_text import transcribe_after_wake
from src.text_to_speech import speak_streaming
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


def listen_loop(cfg: Config, ui: JarvisUI, reset_event: threading.Event) -> None:
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

            def llm_stream():
                for chunk in stream_response(
                    api_key=cfg.anthropic_api_key,
                    messages=history,
                    model=cfg.claude_model,
                    summaries=summaries,
                ):
                    response_chunks.append(chunk)
                    print(chunk, end="", flush=True)
                    yield chunk

            try:
                speak_streaming(
                    llm_stream(),
                    language=language,
                    on_first_audio=lambda: ui.set_state(State.SPEAKING),
                    on_amplitude=ui.set_amplitude,
                )
            except Exception as exc:
                history.pop()  # keep history alternating user/assistant cleanly
                print(f"\n[main] LLM/TTS failed: {exc}")
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
                    )
                except Exception as exc:
                    print(f"[main] STT failed: {exc}")
                    continue

                if not transcript.text:
                    print("[main] (no speech captured)\n")
                    continue

                process_question(transcript.text, transcript.language)
    finally:
        # Quit / shutdown: seal whatever's still in memory so we don't lose it.
        if history:
            print("[memory] sealing in-progress session on shutdown...", file=sys.stderr)
            _seal_session(memory, history, session_language, session_started_at, cfg)


def main() -> None:
    log_path = setup_logging()

    cfg = load()
    if not cfg.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY missing. Add it to .env and try again.", file=sys.stderr)
        sys.exit(1)

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

    worker = threading.Thread(target=listen_loop, args=(cfg, ui, reset_event), daemon=True)
    worker.start()

    ui.run()  # blocks main thread on Tk's mainloop until Quit is clicked

    # Give listen_loop time to see the shutdown event, exit its loop cleanly,
    # and run its try/finally — which seals the active session to disk. Worst
    # case: user quit mid-recording and we wait up to ~15s for max-recording
    # to time out, then a summarize call. M16 hardening: bumped from 20s → 30s
    # so the summarize round-trip (capped at 8s in memory.py, no SDK retries)
    # plus any tail TTS playback comfortably fits before we give up.
    worker.join(timeout=30.0)
    if worker.is_alive():
        print("[main] listen_loop didn't exit in time — session may not be sealed", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
