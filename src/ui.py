"""JarvisUI — single coordinator for the console window + system tray.

Owns:
- A JarvisConsole (Tk window — runs on main thread, must be created there too)
- A JarvisTray (pystray icon — runs on a worker thread)
- A shared shutdown threading.Event used by the listen_loop and the tray's
  animation thread

Threading layout:
- Main thread:    constructs the UI, then runs Tk's mainloop via .run()
- Worker thread:  runs pystray.Icon.run() — constructed inside the worker so
                  any thread-affined Win32 hooks live on the right thread
- Listen loop:    a separate worker (started by main.py); calls set_state /
                  add_user_text / add_jarvis_text — all are thread-safe

Lifecycle:
- Tray "Quit" click → _handle_quit → shutdown event set + console.shutdown()
- Tk mainloop returns → run() finally also calls icon.stop() to wind down tray
- listen_loop sees shutdown.is_set() and returns
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Callable

from src import autostart
from src.console import JarvisConsole
from src.tray import JarvisTray, State


class JarvisUI:
    def __init__(
        self,
        log_path: Path | None = None,
        memory_dir: Path | None = None,
    ) -> None:
        # Console must be constructed on the thread that will run mainloop
        # (Tk requirement). main.py creates JarvisUI on the main thread, so
        # the console is bound to main thread here.
        self.console = JarvisConsole()

        self.shutdown = threading.Event()
        # When set, main.py's process_question bypasses TTS and just streams
        # the text response into the console. Toggleable from the tray.
        # Always starts cleared (unmuted) — restart is a clean slate.
        self.mute_event = threading.Event()
        # When set, stream_response runs with extended thinking enabled and
        # an engineer-mode system prompt addendum (longer structured replies,
        # technical depth). Independent of mute — voice + engineer is allowed.
        # Always starts cleared (off) — engineer mode costs more tokens, so
        # off is the cheap default.
        self.engineer_event = threading.Event()
        self._log_path = log_path
        self._memory_dir = memory_dir
        self._tray: JarvisTray | None = None
        self._on_reset: Callable[[], None] | None = None
        # When True, main() should call autostart.relaunch() after the listen
        # loop has joined + the active session is sealed. The tray's "Restart
        # Jarvis" click flips this, then triggers the normal quit path — the
        # actual spawn happens later in main(), AFTER the mic is released,
        # so the new instance doesn't race the old one for the audio device.
        self._relaunch_requested = False

    def set_on_reset(self, callback: Callable[[], None]) -> None:
        """Wire a reset callback. Called when the tray menu's Reset item fires."""
        self._on_reset = callback

    def set_on_text_submit(
        self, callback: Callable[[str, list[tuple[str, dict]]], None]
    ) -> None:
        """Wire a text-submit callback. Called when the user types in the
        console's input field and presses Enter. Receives (text, attachments)
        where attachments is a list of (filename, content_block) tuples
        (possibly empty). Pass-through to console."""
        self.console.set_on_text_submit(callback)

    # ------------------------------------------------------------------
    # State + transcript API — thread-safe; called from listen_loop worker.
    # ------------------------------------------------------------------

    def set_state(self, state: State) -> None:
        self.console.set_state(state)
        t = self._tray
        if t is not None:
            t.set_state(state)

    def add_user_text(self, text: str, language: str = "en") -> None:
        self.console.add_user_text(text, language)

    def add_jarvis_text(self, text: str) -> None:
        self.console.add_jarvis_text(text)

    def add_system_text(self, text: str) -> None:
        self.console.add_system_text(text)

    def set_amplitude(self, level: float) -> None:
        """Thread-safe pass-through to the console's waveform visualizer.
        Called ~30x/s by speak_streaming during TTS playback."""
        self.console.set_amplitude(level)

    # ------------------------------------------------------------------
    # SRE status / telemetry — pass-throughs to the console's status bar.
    # ------------------------------------------------------------------

    def set_model_name(self, model: str) -> None:
        self.console.set_model_name(model)

    def set_integration(self, name: str, enabled: bool) -> None:
        self.console.set_integration(name, enabled)

    def add_session_tokens(self, n: int) -> None:
        self.console.add_session_tokens(n)

    def add_telemetry_chip(self, text: str) -> None:
        self.console.add_telemetry_chip(text)

    def add_image_thumbnail(self, image_bytes: bytes, label: str) -> None:
        """Embed a thumbnail of an image into the transcript. Called when a
        vision tool (camera_snapshot, screen_snapshot) successfully captures."""
        self.console.add_image_thumbnail(image_bytes, label)

    # ------------------------------------------------------------------
    # Mute toggle. Voice INPUT and console text both keep working when
    # muted; only TTS playback is suppressed. Toggle lives on the tray
    # menu; the console shows a passive read-only indicator.
    # ------------------------------------------------------------------

    def is_muted(self) -> bool:
        return self.mute_event.is_set()

    def set_muted(self, muted: bool) -> None:
        """Apply the mute state and propagate to the console indicator.
        Idempotent — calling with the current value is a no-op."""
        if muted == self.is_muted():
            return
        if muted:
            self.mute_event.set()
        else:
            self.mute_event.clear()
        self.console.set_muted(muted)
        self.console.add_system_text("muted — Jarvis will respond in text only." if muted
                                     else "unmuted — Jarvis will speak again.")

    def _handle_toggle_mute(self) -> None:
        """Tray menu callback. Runs on pystray's thread; UI updates inside
        set_muted are themselves thread-safe."""
        self.set_muted(not self.is_muted())

    # ------------------------------------------------------------------
    # Engineer mode toggle. Independent of mute — engineer + voice is a
    # legitimate combination (dictating a code review or a deep-dive while
    # away from the screen). When on: longer structured replies + extended
    # thinking enabled on the API call.
    # ------------------------------------------------------------------

    def is_engineer_mode(self) -> bool:
        return self.engineer_event.is_set()

    def set_engineer_mode(self, on: bool) -> None:
        if on == self.is_engineer_mode():
            return
        if on:
            self.engineer_event.set()
        else:
            self.engineer_event.clear()
        self.console.set_engineer(on)
        self.console.add_system_text(
            "engineer mode on — deeper, structured replies + extended thinking."
            if on else "engineer mode off — back to concise."
        )

    def _handle_toggle_engineer(self) -> None:
        self.set_engineer_mode(not self.is_engineer_mode())

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Blocks the main thread on Tk's mainloop. Spawns the tray on a worker."""
        threading.Thread(target=self._run_tray, daemon=True).start()
        try:
            self.console.run()
        finally:
            # Tk mainloop ended (Quit was clicked or window forced shut).
            # Make sure shutdown is set + tray is stopped so the listen_loop
            # worker exits cleanly.
            self.shutdown.set()
            t = self._tray
            if t is not None:
                try:
                    t.icon.stop()
                except Exception:
                    pass

    def _run_tray(self) -> None:
        """Construct + run the tray on this worker thread so any thread-affined
        Win32 message-pump state stays on the same thread."""
        self._tray = JarvisTray(
            on_quit=self._handle_quit,
            on_reset=self._handle_reset,
            on_show_window=self.console.show,
            log_path=self._log_path,
            memory_dir=self._memory_dir,
            autostart_enabled=autostart.is_enabled,
            on_autostart_toggle=self._toggle_autostart,
            mute_enabled=self.is_muted,
            on_mute_toggle=self._handle_toggle_mute,
            engineer_enabled=self.is_engineer_mode,
            on_engineer_toggle=self._handle_toggle_engineer,
            on_restart=self._handle_restart,
            on_create_shortcut=self._handle_create_shortcut,
            shutdown_event=self.shutdown,
        )
        self._tray.run()

    def _toggle_autostart(self) -> None:
        try:
            if autostart.is_enabled():
                autostart.disable()
                print("[autostart] disabled")
                self.console.add_system_text("autostart disabled.")
            else:
                autostart.enable()
                print(f"[autostart] enabled ({autostart.shortcut_path()})")
                self.console.add_system_text("autostart enabled.")
        except Exception as exc:
            print(f"[autostart] toggle failed: {exc}")
            self.console.add_system_text(f"autostart toggle failed: {exc}")

    def _handle_quit(self) -> None:
        # Fires on tray's thread when user clicks Quit. Coordinate teardown:
        # set shutdown (listen_loop sees it), then stop Tk so main returns.
        self.shutdown.set()
        self.console.shutdown()

    def _handle_reset(self) -> None:
        if self._on_reset is not None:
            self._on_reset()

    @property
    def relaunch_requested(self) -> bool:
        """True if the user clicked 'Restart Jarvis'. main() reads this after
        the worker has joined to decide whether to spawn a fresh instance."""
        return self._relaunch_requested

    def _handle_restart(self) -> None:
        """Tray callback: signal that this process should relaunch after it
        finishes shutting down. We do NOT spawn the new process here —
        doing so on pystray's thread would race the listen_loop for the
        microphone. Instead, mark the intent and trigger the normal quit
        path; main() picks up the flag after worker.join() and fires the
        actual relaunch from there."""
        self._relaunch_requested = True
        self.console.add_system_text("restarting Jarvis…")
        print("[ui] restart requested — will relaunch after shutdown completes")
        self._handle_quit()

    def _handle_create_shortcut(self) -> None:
        """Tray callback: drop a Jarvis.lnk on the Desktop with the custom
        icon, then open Explorer with the file selected so the user can
        right-click → Pin to taskbar. All errors surface in the console
        rather than crashing the tray thread."""
        try:
            path = autostart.create_desktop_shortcut()
        except Exception as exc:
            print(f"[shortcut] create failed: {exc}")
            self.console.add_system_text(f"desktop shortcut failed: {exc}")
            return

        self.console.add_system_text(
            f"desktop shortcut created: {path.name}. "
            "Right-click it → Pin to taskbar."
        )
        # explorer.exe /select,<path>  opens the parent folder with the file
        # pre-selected. Best-effort: if Explorer can't be invoked for some
        # reason, the shortcut still exists and the user can find it manually.
        try:
            subprocess.Popen(["explorer.exe", f"/select,{path}"])
        except Exception as exc:
            print(f"[shortcut] explorer reveal failed: {exc}")
