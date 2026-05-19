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
import sys
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
        # Security toggle wiring — set via set_on_security_toggle from main.py
        # AFTER SecurityWatcher is instantiated. Until then the tray's
        # checkbox label still renders but clicking is a no-op.
        self._security_activate: Callable[[], None] | None = None
        self._security_deactivate: Callable[[], None] | None = None
        self._security_is_armed: Callable[[], bool] | None = None
        # M39: face-enrollment trigger. Wired post-construction (after
        # face_auth / cameras / announce are available in main.py) so the
        # tray menu can render before the dependency graph is complete.
        self._on_enroll_face: Callable[[], None] | None = None
        # M45: knowledge-reindex trigger. Wired post-construction (after the
        # announce path exists in main.py), same lifecycle as _on_enroll_face.
        self._on_reindex_knowledge: Callable[[], None] | None = None
        # M48: optional LAN remote-console server, registered via set_remote
        # only when JARVIS_REMOTE_TOKEN is configured. Third fan-out sink.
        self._remote: object | None = None
        # When non-None, main() should respawn Jarvis after the listen loop
        # has joined + the active session is sealed. The tray's "Restart
        # Jarvis" click flips this to "normal"; M41's "Restart Jarvis
        # (Administrator)" flips it to "elevated". The actual spawn fires
        # from main() AFTER the mic is released, so the new instance
        # doesn't race the old one for the audio device.
        self._relaunch_mode: str | None = None  # None | "normal" | "elevated"

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

    def set_remote(self, remote: object) -> None:
        """M48: register the LAN remote-console server as a THIRD fan-out
        sink (alongside console + tray). Duck-typed — needs update_state /
        update_armed / push_line. None = no remote (feature off). Wired by
        main.py only when JARVIS_REMOTE_TOKEN is set."""
        self._remote = remote

    def _remote_call(self, method: str, *args: object) -> None:
        """Fan a facade event to the remote, if any. Defensive by
        contract: a remote/network hiccup must NEVER break the console or
        the listen loop (same posture as every optional subsystem)."""
        r = self._remote
        if r is None:
            return
        try:
            getattr(r, method)(*args)
        except Exception as exc:  # noqa: BLE001
            print(f"[remote] fan-out {method} failed: {exc}", file=sys.stderr)

    def set_state(self, state: State) -> None:
        self.console.set_state(state)
        t = self._tray
        if t is not None:
            t.set_state(state)
        self._remote_call("update_state", getattr(state, "value", str(state)))

    def add_user_text(self, text: str, language: str = "en") -> None:
        self.console.add_user_text(text, language)
        self._remote_call("push_line", "user", text)

    def add_jarvis_text(self, text: str) -> None:
        self.console.add_jarvis_text(text)
        self._remote_call("push_line", "jarvis", text)

    def add_system_text(self, text: str) -> None:
        self.console.add_system_text(text)
        self._remote_call("push_line", "system", text)

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
    # Security-armed indicator (M34). Owned by SecurityWatcher, not us —
    # we just provide a thread-safe surface for the indicator to flip and
    # a pass-through that the tray toggle can call. The watcher fires
    # on_armed_changed(bool) on every state flip, which lands here via
    # set_armed_indicator.
    # ------------------------------------------------------------------

    def set_armed_indicator(self, on: bool) -> None:
        """Thread-safe pass-through to the console's armed indicator.
        Called from SecurityWatcher's on_armed_changed callback."""
        self.console.set_armed(on)
        # Fan to the phone so its ARMED badge tracks arming from ANY source
        # (phone control, tray toggle, or voice "activate security").
        self._remote_call("update_armed", bool(on))

    def set_locked_indicator(self, on: bool) -> None:
        """Thread-safe pass-through to the console's locked indicator
        (M35 follow-on). Called from SecurityWatcher's on_locked_changed
        callback when the post-deterrent LOCKED state enters/exits."""
        self.console.set_locked(on)

    def set_on_security_toggle(
        self, on_activate: Callable[[], None], on_deactivate: Callable[[], None],
        is_armed: Callable[[], bool],
    ) -> None:
        """Wire the tray menu's Security-mode toggle to the watcher.
        main.py calls this after instantiating SecurityWatcher so the tray
        has the same activate/deactivate access voice does. is_armed is
        read each time the menu opens (pystray re-evaluates `checked`)."""
        self._security_activate = on_activate
        self._security_deactivate = on_deactivate
        self._security_is_armed = is_armed

    def set_on_enroll_face(self, callback: Callable[[], None]) -> None:
        """Wire the tray's 'Enroll my face' menu callback. main.py calls this
        after the dependency graph (announce + cameras + face_auth) is wired."""
        self._on_enroll_face = callback

    def _handle_enroll_face(self) -> None:
        """Tray callback. Runs on pystray's thread; the actual orchestration
        (announce → capture → enroll → announce) is non-blocking and runs
        on the Announcer thread via on_done."""
        if self._on_enroll_face is not None:
            self._on_enroll_face()

    def set_on_reindex_knowledge(self, callback: Callable[[], None]) -> None:
        """Wire the tray's 'Reindex knowledge' callback (M45). main.py calls
        this after the announce path is available — same as enroll-face."""
        self._on_reindex_knowledge = callback

    def _handle_reindex_knowledge(self) -> None:
        """Tray callback. Runs on pystray's thread; the reindex itself plus
        the spoken result run off-thread (the trigger in main.py is
        non-blocking), so this never stalls the menu."""
        if self._on_reindex_knowledge is not None:
            self._on_reindex_knowledge()

    def _handle_toggle_security(self) -> None:
        """Tray callback: flip armed state via the wired SecurityWatcher.
        Same code path as the voice intent parser — watcher is idempotent."""
        if self._security_is_armed is None:
            return
        if self._security_is_armed():
            if self._security_deactivate is not None:
                self._security_deactivate()
        else:
            if self._security_activate is not None:
                self._security_activate()

    def _security_is_armed_or_false(self) -> bool:
        """Tray-checkbox safe wrapper: returns False before SecurityWatcher
        is wired, so the menu still renders cleanly during early startup."""
        return bool(self._security_is_armed and self._security_is_armed())

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
            security_enabled=self._security_is_armed_or_false,
            on_security_toggle=self._handle_toggle_security,
            on_enroll_face=self._handle_enroll_face,
            on_reindex_knowledge=self._handle_reindex_knowledge,
            on_restart=self._handle_restart,
            on_restart_elevated=self._handle_restart_elevated,
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
    def relaunch_mode(self) -> str | None:
        """M41: which restart variant was requested. None = no restart,
        'normal' = standard `autostart.relaunch()`, 'elevated' = UAC-prompt
        `autostart.relaunch_elevated()`."""
        return self._relaunch_mode

    def _handle_restart(self) -> None:
        """Tray callback: signal that this process should relaunch after it
        finishes shutting down. We do NOT spawn the new process here —
        doing so on pystray's thread would race the listen_loop for the
        microphone. Instead, mark the intent and trigger the normal quit
        path; main() picks up the flag after worker.join() and fires the
        actual relaunch from there."""
        self._relaunch_mode = "normal"
        self.console.add_system_text("restarting Jarvis…")
        print("[ui] restart requested — will relaunch after shutdown completes")
        self._handle_quit()

    def _handle_restart_elevated(self) -> None:
        """M41 tray callback: identical to `_handle_restart` except the mode
        flag tells main() to use `autostart.relaunch_elevated()` (which
        triggers UAC) instead of the standard `autostart.relaunch()`.

        UAC cancel is handled in main() — if the user clicks No on the
        Windows prompt, the current Jarvis still exits (it's already
        shutting down by then) but no new instance starts. That's the
        "occasional sudo" tradeoff documented in the M41 milestone."""
        self._relaunch_mode = "elevated"
        self.console.add_system_text("restarting Jarvis (Administrator)…")
        print("[ui] elevated restart requested — UAC will fire after shutdown")
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
