"""System tray icon with state indicator. SPEAKING state pulses to look alive.

Threading: pystray's Icon.run() owns the main thread (Windows Win32 requirement).
A daemon animation thread updates icon.icon and icon.title on state changes,
and pulses brightness while in SPEAKING. State transitions from the listen
loop arrive via set_state(); we wake the animation thread with an Event so it
re-renders immediately rather than waiting on its sleep timer.
"""

from __future__ import annotations

import math
import threading
import time
from enum import Enum
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

    def __init__(self, on_quit: Callable[[], None]) -> None:
        self.shutdown = threading.Event()
        self._state = State.IDLE
        self._state_changed = threading.Event()
        self._on_quit = on_quit

        self.icon = pystray.Icon(
            "jarvis",
            _make_circle(State.IDLE.value),
            "Jarvis (idle)",
            menu=pystray.Menu(
                pystray.MenuItem("Quit", self._handle_quit),
            ),
        )

    def _handle_quit(self) -> None:
        self.shutdown.set()
        self._state_changed.set()
        try:
            self._on_quit()
        finally:
            self.icon.stop()

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
