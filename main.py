"""Jarvis — top-level entry point.

Threading: pystray's icon loop hooks Win32 and must run on the main thread; the
wake-word→STT→LLM→TTS pipeline runs in a daemon worker that posts state
transitions to the tray via tray.set_state().

Conversation memory: listen_loop owns `history` (alternating user/assistant
messages). Trimmed in pairs to MAX_PAIRS most recent exchanges. Reset paths:
(a) tray menu "Reset conversation" — fires reset_event, cleared next loop top
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
from src.speech_to_text import transcribe_after_wake
from src.text_to_speech import speak
from src.tray import JarvisTray, State
from src.wake_word import wait_for_wake_word


MAX_PAIRS = 10            # cap conversation at 10 exchanges (20 messages)
IDLE_RESET_SEC = 600.0    # 10 min of silence → forget conversation


def setup_logging() -> Path | None:
    """Determine the log path for the tray's 'Open log' menu item.

    Three modes:
    - jarvis.pyw launcher: already redirected stdout/stderr at import time and
      set JARVIS_LOG_PATH. We just return the path.
    - python main.py (console): no redirect, no logfile, return None.
    - pythonw main.py (rare; no launcher): redirect now and return path. This
      is a fallback — jarvis.pyw is the supported pythonw entry point because
      it can redirect *before* import-time prints fire.
    """
    env_path = os.environ.get("JARVIS_LOG_PATH")
    if env_path:
        return Path(env_path)
    if sys.stdout is not None and sys.stderr is not None:
        return None
    log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "jarvis.log"
    f = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = f
    sys.stderr = f
    print(f"\n--- Jarvis started {datetime.now().isoformat(timespec='seconds')} ---")
    return log_path


def _trim_history(history: list[dict]) -> None:
    """Drop oldest pairs in place, keeping the most recent MAX_PAIRS exchanges.
    Always called when history is in clean pair state (ends on 'assistant')."""
    while len(history) > MAX_PAIRS * 2:
        del history[:2]  # drop oldest user + assistant


def listen_loop(cfg: Config, tray: JarvisTray, reset_event: threading.Event) -> None:
    """Daemon worker. Owns conversation history and updates tray state per phase."""
    history: list[dict] = []
    last_turn_time = 0.0

    with AudioSession(sample_rate=cfg.sample_rate) as session:
        while not tray.shutdown.is_set():
            tray.set_state(State.IDLE)
            session.drain()

            wait_for_wake_word(session, threshold=cfg.wake_word_threshold)
            if tray.shutdown.is_set():
                break

            tray.set_state(State.LISTENING)

            try:
                transcript = transcribe_after_wake(session, model_name=cfg.whisper_model)
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
                    print(f"[main] conversation reset (manual; cleared {len(history)} msgs)")
                history.clear()
                reset_event.clear()
            elif history and (time.time() - last_turn_time) > IDLE_RESET_SEC:
                print(f"[main] conversation reset (idle >{IDLE_RESET_SEC:.0f}s)")
                history.clear()

            print(f"\n[user, {transcript.language}] {transcript.text}")
            print("[jarvis] ", end="", flush=True)
            tray.set_state(State.THINKING)

            history.append({"role": "user", "content": transcript.text})

            response_chunks: list[str] = []
            try:
                for chunk in stream_response(
                    api_key=cfg.anthropic_api_key,
                    messages=history,
                    model=cfg.claude_model,
                ):
                    response_chunks.append(chunk)
                    print(chunk, end="", flush=True)
            except Exception as exc:
                history.pop()  # roll back so history stays in clean user/assistant pairs
                print(f"\n[main] LLM failed: {exc}")
                continue
            print()

            full_response = "".join(response_chunks).strip()
            if not full_response:
                history.pop()  # nothing came back; drop the orphan user message
                continue

            history.append({"role": "assistant", "content": full_response})
            _trim_history(history)
            last_turn_time = time.time()

            tray.set_state(State.SPEAKING)
            speak(full_response, language=transcript.language)
            print()


def main() -> None:
    log_path = setup_logging()

    cfg = load()
    if not cfg.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY missing. Add it to .env and try again.", file=sys.stderr)
        sys.exit(1)

    print("Jarvis ready. Tray icon active — right-click for menu. Say 'Hey Jarvis' to begin.\n")

    reset_event = threading.Event()

    def _on_reset() -> None:
        # Fires on pystray's menu thread. Just signals — actual clear happens
        # in listen_loop after the next wake word + STT, so the click applies
        # to the very next question (not the one after).
        reset_event.set()
        print("[tray] reset queued — applies to your next 'Hey Jarvis'")

    tray = JarvisTray(
        on_quit=lambda: print("\nGoodbye."),
        on_reset=_on_reset,
        log_path=log_path,
    )

    worker = threading.Thread(target=listen_loop, args=(cfg, tray, reset_event), daemon=True)
    worker.start()

    tray.run()  # blocks until Quit


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
