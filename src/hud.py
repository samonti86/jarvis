"""The ambient Iron Man HUD overlay (M84).

A translucent, click-through, always-on-top corner overlay — Tony Stark's
faceplate HUD, but on your desktop. It floats over everything (you keep working
underneath it, mouse passes straight through) and shows, at a glance:
  - the arc-reactor orb, coloured by Jarvis's state (idle/listening/thinking/
    speaking), its core throbbing with the live TTS amplitude when he speaks;
  - a live waveform that dances while he talks;
  - a digital clock;
  - a security badge (ARMED in red when away, dim SECURE at home).

Windows mechanics (the load-bearing tricks):
  - A borderless `Toplevel` with `-transparentcolor` set to a magic key colour:
    every pixel painted that colour becomes fully transparent, so only the HUD
    elements show — the rest of the window is invisible.
  - `WS_EX_LAYERED | WS_EX_TRANSPARENT` (via ctypes SetWindowLong) makes the
    whole window CLICK-THROUGH — mouse events pass to whatever's underneath, so
    it never steals a click. `WS_EX_TOOLWINDOW` keeps it out of the taskbar /
    Alt-Tab.
  - `-topmost` keeps it above normal windows.

Defensive contract: this is pure eye-candy and must NEVER affect the assistant.
Any failure to build the overlay (no display, an old Windows, a Tk quirk)
disables the HUD and logs — all setters become no-ops and Jarvis runs untouched.
Default OFF; opt in via JARVIS_HUD=1 (or the tray toggle).
"""

from __future__ import annotations

import math
import os
import sys
import time
import tkinter as tk
from typing import Callable

from src import reactor
from src.tray import State

# The transparent-colour key. A near-black almost no real element uses; any
# pixel exactly this colour is punched fully transparent + click-through.
_KEY = "#010101"

# Per-state orb colour (matches the M57 console palette).
_STATE_COLOR = {
    State.IDLE: "#3f5468",       # calm dim cyan-slate — reactor at rest
    State.LISTENING: "#22d3ee",  # electric cyan
    State.THINKING: "#f5b942",   # gold — working
    State.SPEAKING: "#34e6d0",   # bright teal-cyan
}
_ACCENT_HOT = "#7df3ff"   # near-white-hot cyan — glow cores, peaks
_RING = "#244a5e"         # dim HUD ring / ticks
_TEXT = "#7fe9f5"         # cyan readout text
_ARMED = "#ff5a5a"        # security-armed red
_RESEARCH = "#7fd4ff"     # background-research cyan


def is_enabled() -> bool:
    """Opt-in (default OFF) — set JARVIS_HUD=1 to show the overlay."""
    return os.getenv("JARVIS_HUD", "").strip().lower() in ("1", "true", "yes", "on")


# --- pure colour helpers (no Tk — unit-testable) ---------------------------

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = (max(0, min(255, int(round(c)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _dim(hexc: str, factor: float) -> str:
    """Scale a colour toward black by `factor` (0 → black, 1 → unchanged)."""
    r, g, b = _hex_to_rgb(hexc)
    return _rgb_to_hex((r * factor, g * factor, b * factor))


def _blend(a: str, b: str, t: float) -> str:
    """Linear blend from colour `a` (t=0) to `b` (t=1)."""
    t = max(0.0, min(1.0, t))
    ar, ag, ab = _hex_to_rgb(a)
    br, bg, bb = _hex_to_rgb(b)
    return _rgb_to_hex((ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t))


# --- the overlay -----------------------------------------------------------

class JarvisHUD:
    """The ambient corner overlay. Construct with the shared Tk root (the
    console's). All public setters are cheap attribute stores read by the
    animation tick, so they're safe to call from any thread."""

    W = 260
    H = 348
    MARGIN = 28
    ORB = 190          # rendered reactor size, px

    def __init__(self, root: tk.Misc, *, corner: str = "se") -> None:
        self._disabled = False
        self._state: State = State.IDLE
        self._amp = 0.0
        self._armed = False
        self._tasks_active = 0
        self._tasks_pending = 0
        self._anim_start = time.time()
        self._wave_bars: list[int] = []
        self._photos: dict[tuple, object] = {}   # (color, i) -> PhotoImage (GC anchor)
        self._img_mode = False
        try:
            self._build(root, corner)
        except Exception as exc:  # noqa: BLE001 — eye-candy must never break Jarvis
            print(f"[hud] disabled (overlay setup failed: {exc})", file=sys.stderr)
            self._disabled = True
            return
        print("[hud] ambient overlay active", file=sys.stderr)
        self._tick()

    # ---- public state setters (thread-safe: plain stores) ----

    def set_state(self, state: State) -> None:
        self._state = state

    def set_amplitude(self, level: float) -> None:
        try:
            self._amp = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            self._amp = 0.0

    def set_armed(self, on: bool) -> None:
        self._armed = bool(on)

    def set_research(self, active: int, pending: int) -> None:
        """Background research state (M97).

        This is the one thing Jarvis does that is otherwise INVISIBLE: an M91
        task runs off-box for minutes to hours with no window, no sound, and
        no tray change. Ambient status is exactly what a HUD is for, so a
        running task gets a line and a finished-but-unspoken one gets a
        different line."""
        try:
            self._tasks_active = max(0, int(active))
            self._tasks_pending = max(0, int(pending))
        except (TypeError, ValueError):
            self._tasks_active = self._tasks_pending = 0

    def is_active(self) -> bool:
        return not self._disabled

    def destroy(self) -> None:
        if self._disabled:
            return
        self._disabled = True
        try:
            self._win.destroy()
        except Exception:  # noqa: BLE001
            pass

    # ---- build ----

    def _build(self, root: tk.Misc, corner: str) -> None:
        self._win = tk.Toplevel(root)
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)
        self._win.configure(bg=_KEY)
        # The transparent-colour key (Windows). On a platform without it this
        # raises TclError; we degrade to a visible panel rather than crash.
        try:
            self._win.attributes("-transparentcolor", _KEY)
        except tk.TclError:
            print("[hud] -transparentcolor unsupported here; HUD will be opaque",
                  file=sys.stderr)

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = sw - self.W - self.MARGIN if "e" in corner else self.MARGIN
        y = sh - self.H - self.MARGIN if "s" in corner else self.MARGIN
        self._win.geometry(f"{self.W}x{self.H}+{x}+{y}")

        self._cv = tk.Canvas(self._win, width=self.W, height=self.H, bg=_KEY,
                             highlightthickness=0, bd=0)
        self._cv.pack()
        self._build_items()
        self._win.update_idletasks()
        self._make_clickthrough()

    def _make_clickthrough(self) -> None:
        """Set WS_EX_LAYERED | WS_EX_TRANSPARENT so the mouse passes through,
        and WS_EX_TOOLWINDOW so it's hidden from the taskbar / Alt-Tab."""
        try:
            from ctypes import windll  # noqa: PLC0415 — Windows-only, lazy

            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_TOOLWINDOW = 0x00000080
            hwnd = windll.user32.GetParent(self._win.winfo_id()) or self._win.winfo_id()
            cur = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE,
                cur | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW,
            )
        except Exception as exc:  # noqa: BLE001 — click-through is best-effort
            print(f"[hud] click-through unavailable ({exc}); overlay may catch clicks",
                  file=sys.stderr)

    def _build_items(self) -> None:
        """Create canvas items once; the tick mutates fills/coords/text.

        TWO renderers live here. The pre-rendered PIL reactor (M98) is the real
        look; the vector items below are the FALLBACK that draws until the
        first rotation cycle finishes warming on its background thread, and
        forever if Pillow is unavailable. They are hidden, not destroyed, the
        moment images become available — so a HUD that loses its images (it
        cannot, but) degrades rather than goes blank.
        """
        cv = self._cv
        cx = self.W // 2
        ocy = 6 + self.ORB // 2        # orb centre y — image pasted at y=6
        r_out = 64
        r_mid = int(r_out * 0.80)
        self._ocx, self._ocy, self._r_out, self._r_mid = cx, ocy, r_out, r_mid

        # The pre-rendered reactor. Created first so everything else sits above
        # it; empty until the warm completes.
        self._orb_img = cv.create_image(cx, ocy)
        reactor.warm_all(self.ORB, _STATE_COLOR.values())

        # --- vector fallback (hidden once images arrive) ---
        self._vec: list[int] = []
        self._vec.append(cv.create_oval(cx - r_out, ocy - r_out, cx + r_out, ocy + r_out,
                                        outline=_RING, width=2))
        for i in range(12):
            a = math.radians(i * 30)
            r1, r2 = r_out - 3, r_out - 10
            self._vec.append(
                cv.create_line(cx + r1 * math.cos(a), ocy + r1 * math.sin(a),
                               cx + r2 * math.cos(a), ocy + r2 * math.sin(a),
                               fill=_RING, width=2))
        # Radial glow halo — 4 discs, large → small (filled per-frame).
        self._glow: list[int] = []
        for rr in (int(r_out * 0.62), int(r_out * 0.46),
                   int(r_out * 0.32), int(r_out * 0.21)):
            self._glow.append(cv.create_oval(cx - rr, ocy - rr, cx + rr, ocy + rr,
                                             fill=_KEY, outline=""))
        self._vec.extend(self._glow)
        self._vec.append(cv.create_oval(cx - r_mid, ocy - r_mid, cx + r_mid, ocy + r_mid,
                                        outline=_RING, width=1))
        self._arc_out = cv.create_arc(cx - r_out, ocy - r_out, cx + r_out, ocy + r_out,
                                      start=0, extent=90, style="arc",
                                      outline="#22d3ee", width=3)
        self._arc_mid = cv.create_arc(cx - r_mid, ocy - r_mid, cx + r_mid, ocy + r_mid,
                                      start=180, extent=60, style="arc",
                                      outline="#22d3ee", width=2)
        self._vec.extend((self._arc_out, self._arc_mid))

        # --- live overlays (drawn in BOTH modes) ---
        # The pre-rendered frames cannot vary with amplitude, so the two things
        # that react to the voice stay cheap canvas primitives on top: a
        # throbbing core and an amplitude ring.
        self._core_r = max(8, int(self.ORB * 0.055))
        r = self._core_r
        self._core = cv.create_oval(cx - r, ocy - r, cx + r, ocy + r,
                                    fill="#22d3ee", outline="")
        self._amp_ring = cv.create_oval(cx, ocy, cx, ocy, outline="", width=2)

        # Waveform — 19 thin bars, tapered at the ends by a sine envelope so it
        # reads as a waveform rather than a barcode (the first draft did not).
        self._wave_n = 19
        self._wave_y = 230
        self._wave_step = 6
        x0 = cx - (self._wave_n - 1) * self._wave_step // 2
        for i in range(self._wave_n):
            x = x0 + i * self._wave_step
            self._wave_bars.append(
                cv.create_line(x, self._wave_y, x, self._wave_y, fill=_RING,
                               width=1, capstyle="round")
            )
        self._wave_env = [math.sin(math.pi * (i + 0.5) / self._wave_n) ** 0.8
                          for i in range(self._wave_n)]

        # --- readouts ---
        # Vertical budget is tight: the orb ends at y=196 and the waveform can
        # swing +/-16 px about y=230, so the label sits in the gap between them
        # and the clock clears the waveform's bottom.
        self._label = cv.create_text(cx, 206, text="", fill=_RING,
                                     font=("Consolas", 8))
        self._clock = cv.create_text(cx, 268, text="--:--", fill=_TEXT,
                                     font=("Consolas", 26, "bold"))
        cv.create_line(74, 292, self.W - 74, 292, fill="#142c38")
        self._badge = cv.create_text(cx, 308, text="", fill=_RING,
                                     font=("Consolas", 10, "bold"))
        self._research = cv.create_text(cx, 328, text="", fill=_RESEARCH,
                                        font=("Consolas", 9))

    def _orb_frame(self, color: str, t: float) -> bool:
        """Blit the current rotation frame. False if not warmed yet."""
        cycle = reactor.frames(self.ORB, color)
        if cycle is None:
            return False
        # Rotation speed per state — the frame index, not the render, is what
        # changes, so a "faster" state costs exactly the same as a slow one.
        rate = {State.IDLE: 1.2, State.LISTENING: 3.0,
                State.THINKING: 6.0, State.SPEAKING: 4.0}.get(self._state, 2.0)
        i = int(t * rate) % len(cycle)
        key = (color, i)
        photo = self._photos.get(key)
        if photo is None:
            from PIL import ImageTk  # noqa: PLC0415 — lazy; needs a live Tk

            # Converted once per frame per colour and kept forever: a
            # PhotoImage that loses its last Python reference is garbage
            # collected and the canvas silently draws nothing.
            photo = ImageTk.PhotoImage(cycle[i])
            self._photos[key] = photo
        self._cv.itemconfig(self._orb_img, image=photo)
        if not self._img_mode:
            self._img_mode = True
            for item in self._vec:
                self._cv.itemconfigure(item, state="hidden")
        return True

    # ---- animation ----

    def _tick(self) -> None:
        if self._disabled:
            return
        if not self._draw():
            return
        try:
            # 20 fps, not 30. Measured: the redraw costs 13% of a core at 30 fps
            # and 8% at 20, and nothing on screen needs the extra frames — the
            # reactor's own arcs only advance 1.2-6 times a second. The saving
            # lands hardest while SPEAKING, which is precisely when the
            # real-time audio threads can least afford the competition (M67/M68).
            self._win.after(50, self._tick)
        except tk.TclError:
            self._disabled = True

    def _draw(self) -> bool:
        state = self._state
        base = _STATE_COLOR.get(state, "#22d3ee")
        t = time.time() - self._anim_start

        if state == State.SPEAKING:
            pulse = 0.42 + 0.58 * min(1.0, self._amp * 1.5)
        elif state == State.IDLE:
            pulse = 0.50 + 0.16 * math.sin(t * 1.6)
        elif state == State.THINKING:
            pulse = 0.74 + 0.16 * math.sin(t * 5.0)
        else:  # LISTENING
            pulse = 0.84 + 0.12 * math.sin(t * 3.0)

        speed = {State.IDLE: 24, State.LISTENING: 64,
                 State.THINKING: 150, State.SPEAKING: 92}.get(state, 50)
        ang = (t * speed) % 360

        try:
            cv = self._cv
            cx, cy = self._ocx, self._ocy

            if not self._orb_frame(base, t):
                # Vector fallback — still warming, or no Pillow.
                cv.itemconfig(self._arc_out, start=ang,
                              outline=_blend(_dim(base, 0.35), base, pulse))
                cv.itemconfig(self._arc_mid, start=(-ang * 1.4) % 360,
                              outline=_dim(base, 0.5 + 0.3 * pulse))
                for gid, f in zip(self._glow, (0.18, 0.34, 0.58, 0.92)):
                    cv.itemconfig(gid, fill=_dim(base, f * pulse))

            cv.itemconfig(self._core,
                          fill=_blend(base, _ACCENT_HOT, 0.25 + 0.55 * pulse))
            cr = self._core_r * (0.78 + 0.5 * pulse)
            cv.coords(self._core, cx - cr, cy - cr, cx + cr, cy + cr)

            # Amplitude ring — expands with the live TTS level. Hidden when
            # silent (an empty outline is how a canvas item goes invisible).
            if self._amp > 0.02:
                rr = self.ORB * (0.25 + 0.03 * self._amp)
                cv.coords(self._amp_ring, cx - rr, cy - rr, cx + rr, cy + rr)
                cv.itemconfig(self._amp_ring,
                              outline=_blend(base, _ACCENT_HOT, self._amp))
            else:
                cv.itemconfig(self._amp_ring, outline="")

            # Waveform — reactive while speaking, a faint resting ripple
            # otherwise. `_wave_env` tapers the ends so it reads as a waveform.
            speaking = state == State.SPEAKING
            wave_fill = _dim(base, 0.5 + 0.5 * pulse) if speaking else _RING
            for i, bar in enumerate(self._wave_bars):
                phase = math.sin(t * 9 + i * 0.7)
                if speaking:
                    mag = self._amp * (0.35 + 0.65 * abs(phase)) * 16 * self._wave_env[i]
                else:
                    mag = (0.5 + 0.5 * phase) * 1.6 * self._wave_env[i]
                bx = cv.coords(bar)[0]
                cv.coords(bar, bx, self._wave_y - mag, bx, self._wave_y + mag)
                cv.itemconfig(bar, fill=wave_fill)

            cv.itemconfig(self._label, text=state.name,
                          fill=_blend(_RING, base, 0.85))
            cv.itemconfig(self._clock, text=time.strftime("%H:%M"))
            if self._armed:
                cv.itemconfig(self._badge, text="● ARMED", fill=_ARMED)
            else:
                cv.itemconfig(self._badge, text="○ SECURE", fill=_RING)

            # Research line. Empty when there is nothing to say — an always-on
            # row of "0 tasks" is clutter, and the HUD sits over the user's work.
            if self._tasks_active:
                dots = "." * (1 + int(t * 2) % 3)      # gentle "working" motion
                n = self._tasks_active
                label = f"researching{dots}" if n == 1 else f"researching x{n}{dots}"
                cv.itemconfig(self._research, text=label, fill=_RESEARCH)
            elif self._tasks_pending:
                n = self._tasks_pending
                cv.itemconfig(
                    self._research,
                    text="1 report ready" if n == 1 else f"{n} reports ready",
                    fill=_RESEARCH)
            else:
                cv.itemconfig(self._research, text="")
        except tk.TclError:
            return False
        return True
