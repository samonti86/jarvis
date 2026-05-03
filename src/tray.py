"""System tray icon with state indicator. SPEAKING state pulses to look alive.

Threading: pystray's Icon.run() owns the main thread (Windows Win32 requirement).
A daemon animation thread updates icon.icon and icon.title on state changes,
and pulses brightness while in SPEAKING. State transitions from the listen
loop arrive via set_state(); we wake the animation thread with an Event so it
re-renders immediately rather than waiting on its sleep timer.

Menu callbacks (Reset, Open log) run on pystray's own thread. They post
intent via callbacks the caller wires up — we don't touch listen-loop state
directly from here.
"""

from __future__ import annotations

import math
import os
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Callable

import pystray
from PIL import Image, ImageDraw


class State(Enum):
    IDLE = (128, 128, 128)        # gray
    LISTENING = (51, 153, 255)    # blue
    THINKING = (255, 204, 0)      # amber/yellow
    SPEAKING = (51, 204, 51)      # green


def _make_circle(rgb: tuple[int, int, int], brightness: float = 1.0, size: int = 64) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r, g, b = rgb
    color = (int(r * brightness), int(g * brightness), int(b * brightness), 255)
    draw.ellipse([4, 4, size - 4, size - 4], fill=color)
    return img


class JarvisTray:
    """Tray icon manager. Call set_state() from any thread; run() blocks the caller."""

    def __init__(
        self,
        on_quit: Callable[[], None],
        on_reset: Callable[[], None] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.shutdown = threading.Event()
        self._state = State.IDLE
        self._state_changed = threading.Event()
        self._on_quit = on_quit
        self._on_reset = on_reset
        self._log_path = log_path

        menu_items: list[pystray.MenuItem] = []
        if on_reset is not None:
            menu_items.append(pystray.MenuItem("Reset conversation", self._handle_reset))
        if log_path is not None:
            menu_items.append(pystray.MenuItem("Open log", self._handle_open_log))
        if menu_items:
            menu_items.append(pystray.Menu.SEPARATOR)
        menu_items.append(pystray.MenuItem("Quit", self._handle_quit))

        self.icon = pystray.Icon(
            "jarvis",
            _make_circle(State.IDLE.value),
            "Jarvis (idle)",
            menu=pystray.Menu(*menu_items),
        )

    def _handle_quit(self) -> None:
        self.shutdown.set()
        self._state_changed.set()
        try:
            self._on_quit()
        finally:
            self.icon.stop()

    def _handle_reset(self) -> None:
        if self._on_reset is not None:
            self._on_reset()

    def _handle_open_log(self) -> None:
        if self._log_path is not None and self._log_path.exists():
            os.startfile(str(self._log_path))  # type: ignore[attr-defined]

    def set_state(self, state: State) -> None:
        self._state = state
        self._state_changed.set()

    def _animation_loop(self) -> None:
        while not self.shutdown.is_set():
            state = self._state
            self.icon.title = f"Jarvis ({state.name.lower()})"

            if state == State.SPEAKING:
                # Sine-wave pulse, brightness 0.4..1.0 at 2 Hz, 8 fps update.
                t = time.time()
                brightness = 0.7 + 0.3 * math.sin(t * 2 * math.pi * 2)
                self.icon.icon = _make_circle(state.value, brightness)
                time.sleep(1 / 8)
            else:
                self.icon.icon = _make_circle(state.value, 1.0)
                # Park until next state change (or 2s safety timeout).
                self._state_changed.wait(timeout=2.0)
                self._state_changed.clear()

    def run(self) -> None:
        """Blocks. Runs the icon event loop. Animation runs in a daemon thread."""
        anim_thread = threading.Thread(target=self._animation_loop, daemon=True)
        anim_thread.start()
        self.icon.run()
