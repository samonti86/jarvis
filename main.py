"""Jarvis — top-level entry point.

Threading layout (M8):
- Main thread:    Tk's mainloop (JarvisConsole window). Tk strongly prefers
                  the main thread, so this is its home.
- Worker thread:  pystray icon loop (started by JarvisUI). Win32 lets pystray
                  run on any thread; we put it here so Tk can have main.
- Worker thread:  listen_loop (started here in main()). Posts state +
                  transcript updates to the UI; both UI surfaces are
                  thread-safe (Tk via .after(), pystray via attribute writes).

Conversation memory: listen_loop owns `history` (alternating user/assistant
messages). Trimmed in pairs to MAX_PAIRS most recent exchanges. Reset paths:
(a) tray menu "Reset conversation" — fires reset_event, applied at the next
    wake-word + STT
(b) idle timeout — IDLE_RESET_SEC since last completed turn → forget
(c) app restart — history is in-process only, never persisted

Console hiding: when launched via pythonw.exe (jarvis.pyw), there's no console.
setup_logging() redirects stdout/stderr to %LOCALAPPDATA%\\Jarvis\\jarvis.log so
we don't lose debug output. Console mode (python main.py) is unchanged.

Echo handling: while TTS plays, the mic still captures it. session.drain() at
the top of each iteration discards that buffered echo so the wake-word detector
starts each turn on fresh, live audio.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from src.audio import AudioSession
from src.config import Config, load
from src.llm import stream_response
from src.memory import MemoryStore, SummaryRecord, summarize_session
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
    log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "jarvis.log"

    env_path = os.environ.get("JARVIS_LOG_PATH")
    if env_path:
        # jarvis.pyw already opened the file + replaced stdout/stderr.
        return Path(env_path)

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
    sessions on every memory boundary (manual reset / idle / app quit)."""
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

    def seal_and_refresh() -> None:
        """Seal the active session (if any) and reload summaries for the next one."""
        nonlocal summaries, session_started_at
        _seal_session(memory, history, session_language, session_started_at, cfg)
        history.clear()
        summaries = memory.recent_summaries(cfg.memory_recall_count)
        session_started_at = datetime.now().isoformat(timespec="seconds")

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
                    )
                except Exception as exc:
                    print(f"[main] STT failed: {exc}")
                    continue

                if not transcript.text:
                    print("[main] (no speech captured)\n")
                    continue

                # Reset checks happen here — late enough to catch a click made any time
                # during wait_for_wake_word *or* STT, so the click always applies to the
                # turn the user is asking right now (not the next one).
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
                    session_language = transcript.language or "en"
                    session_started_at = datetime.now().isoformat(timespec="seconds")

                print(f"\n[user, {transcript.language}] {transcript.text}")
                print("[jarvis] ", end="", flush=True)
                ui.add_user_text(transcript.text, transcript.language)
                # State is already THINKING (set by on_speech_ended callback inside STT).

                history.append({"role": "user", "content": transcript.text})

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
                        language=transcript.language,
                        on_first_audio=lambda: ui.set_state(State.SPEAKING),
                    )
                except Exception as exc:
                    history.pop()  # keep history alternating user/assistant cleanly
                    print(f"\n[main] LLM/TTS failed: {exc}")
                    continue
                print()

                full_response = "".join(response_chunks).strip()
                if not full_response:
                    history.pop()  # nothing came back; drop the orphan user message
                    continue

                history.append({"role": "assistant", "content": full_response})
                _trim_history(history)
                last_turn_time = time.time()

                # Persist the completed exchange to today's transcript file.
                memory.record_turn(transcript.text, full_response, transcript.language)

                ui.add_jarvis_text(full_response)
                print()
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

    ui = JarvisUI(log_path=log_path)
    ui.set_on_reset(_on_reset)

    worker = threading.Thread(target=listen_loop, args=(cfg, ui, reset_event), daemon=True)
    worker.start()

    ui.run()  # blocks main thread on Tk's mainloop until Quit is clicked

    # Give listen_loop time to see the shutdown event, exit its loop cleanly,
    # and run its try/finally — which seals the active session to disk. Worst
    # case: user quit mid-recording and we wait up to ~15s for max-recording
    # to time out. Most quits happen during IDLE so this returns within ~80ms.
    worker.join(timeout=20.0)
    if worker.is_alive():
        print("[main] listen_loop didn't exit in time — session may not be sealed", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
