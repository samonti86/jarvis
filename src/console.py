"""JarvisConsole — dark-themed customtkinter window showing live state + transcript.

Threading model:
- Tk's mainloop runs on the main thread (its preferred home — Tk is single-threaded).
- All public methods are thread-safe: they schedule the actual widget update via
  self.root.after(0, ...), which marshals the call onto Tk's thread.
- An .after(125, _tick_animation) recursive scheduler drives the SPEAKING-state
  pulse at 8 fps, all on Tk's thread — no separate animation thread needed.

Text input (M15): a CTkEntry at the bottom lets the user type questions instead
of speaking. Submit fires the on_text_submit callback (registered by main.py)
which enqueues the text for the same per-turn pipeline as voice.

File attachments (M16): a 📎 button next to the entry opens a file picker.
Selecting a file stages it as a content block (via src.attachments) and shows
a chip above the entry. Submit sends both the typed text and the staged
attachment in one user message.

Audio waveform (M17): 24 vertical bars between the state pill and transcript
that pulse with the TTS amplitude envelope. set_amplitude(level) is thread-safe
(called ~30x/s by speak_streaming's envelope ticker). The _wave_tick redraws
at 30 fps with smoothing + decay so bars settle to a flat resting line when
audio stops.

Closing the window still hides; control (reset, open log, quit) still lives
on the tray.
"""

from __future__ import annotations

import ctypes
import io
import math
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Callable

import customtkinter as ctk
from PIL import Image, ImageTk

from src.attachments import load_attachment
from src.tray import State


class JarvisConsole:
    # Palette — Stark HUD: near-black + electric arc-reactor cyan, with a
    # sparing Iron-Man gold. The console-local _STATE_COLOR map keeps the
    # orb's per-state palette out of tray.py's State enum.
    BG = "#0a0e14"          # near-black, faint blue
    PANEL_BG = "#10161f"    # raised panels (transcript, input, chip)
    PANEL_LINE = "#1d2b3a"  # dim HUD rings / tick marks
    BORDER = "#21465a"      # faintly glowing panel edge
    ACCENT = "#22d3ee"      # electric arc-reactor cyan — primary
    ACCENT_HOT = "#7df3ff"  # near-white-hot cyan — glow cores, peaks
    GOLD = "#f5b942"        # Iron Man gold — sparing secondary
    HEADER_FG = "#22d3ee"
    USER_FG = "#cdd9e5"
    JARVIS_FG = "#5ee0f0"
    DIM_FG = "#5b6b7f"

    ORB_SIZE = 176  # arc-reactor canvas (square; the orb is centred in it)

    # Per-state orb colour — console-local (NOT State.value, which drives the
    # tray icon). THINKING borrows Iron Man's gold; the rest are reactor cyan.
    _STATE_COLOR = {
        State.IDLE: "#3f5468",       # calm dim cyan-slate — reactor at rest
        State.LISTENING: "#22d3ee",  # electric cyan
        State.THINKING: "#f5b942",   # gold — "working"
        State.SPEAKING: "#34e6d0",   # bright teal-cyan
    }

    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Jarvis")
        self.root.geometry("560x800")
        self.root.configure(fg_color=self.BG)
        self.root.minsize(480, 660)

        self._state = State.IDLE
        self._anim_start = time.time()
        self._destroyed = False
        # Submit callback receives (text, attachments) where attachments is a
        # list of (filename, content_block) tuples — possibly empty. None means
        # not wired yet.
        self._on_text_submit: Callable[[str, list[tuple[str, dict]]], None] | None = None
        # Currently staged attachment (one at a time for first pass): (filename, content_block).
        # None when nothing is staged.
        self._staged: tuple[str, dict] | None = None
        # Thumbnails embedded in the transcript. Each entry holds:
        #   (thumb_photo, full_bytes, label)
        # - thumb_photo: PhotoImage rendered inline; we hold a strong ref
        #   because Tk only weak-references images and would silently GC them
        #   otherwise (transcript would show empty placeholders).
        # - full_bytes: original image bytes from the vision tool, kept so
        #   click-to-enlarge can re-decode at full resolution without going
        #   back to disk.
        # - label: friendly source name (e.g. "🖥 screen snapshot") — also
        #   used as the popup window title.
        # Grows unbounded across the session: each entry is ~100 KB live
        # thumb + ~200-500 KB raw bytes; a 50-turn vision-heavy session is
        # bounded at a few MB. Cleared implicitly when the window destroys.
        self._thumbnail_refs: list[tuple[ImageTk.PhotoImage, bytes, str]] = []

        # --- Header ---
        header = ctk.CTkLabel(
            self.root,
            text="J . A . R . V . I . S .",
            font=("Consolas", 28, "bold"),
            text_color=self.HEADER_FG,
        )
        header.pack(pady=(24, 3))

        subtitle = ctk.CTkLabel(
            self.root,
            text="at your service, sir",
            font=("Consolas", 10),
            text_color=self.DIM_FG,
        )
        subtitle.pack(pady=(0, 12))

        # Thin HUD divider under the header.
        divider = ctk.CTkFrame(
            self.root, height=2, fg_color=self.BORDER, corner_radius=0,
        )
        divider.pack(fill="x", padx=110, pady=(0, 10))

        # --- Arc-reactor orb (the centrepiece) ---
        # Replaces the old 22px state dot: a layered concentric reactor —
        # bezel + tick ring, two counter-rotating arcs, a 4-disc radial glow
        # halo, and a core that breathes (and, while SPEAKING, throbs with the
        # live TTS amplitude). Built once by _build_orb; _tick_orb then mutates
        # only fills/coords each frame — the same cheap-redraw pattern as the
        # waveform.
        self._orb = tk.Canvas(
            self.root, width=self.ORB_SIZE, height=self.ORB_SIZE,
            bg=self.BG, highlightthickness=0, bd=0,
        )
        self._orb.pack(pady=(2, 2))
        self._build_orb()

        self._state_label = ctk.CTkLabel(
            self.root,
            text="IDLE",
            font=("Consolas", 18, "bold"),
            text_color=self.HEADER_FG,
        )
        self._state_label.pack(pady=(0, 2))

        # Indicator row — mute / engineer / armed / locked badges sit here,
        # centred under the state label. Each label's master is this frame,
        # so the existing _apply_* pack/forget calls land in the right place
        # with no change to those methods. A plain tk.Frame (not CTkFrame):
        # it starts childless, and an empty CTkFrame reserves its 200x200
        # default size — a tk.Frame collapses to its content instead.
        self._indicator_frame = tk.Frame(self.root, bg=self.BG)
        self._indicator_frame.pack(pady=(0, 8))

        # Read-only mute indicator. Packed only while muted (set_muted). The
        # toggle lives on the tray — the console stays a passive surface.
        self._mute_label = ctk.CTkLabel(
            self._indicator_frame,
            text="🔇 muted",
            font=("Segoe UI Emoji", 12),
            text_color=self.DIM_FG,
        )
        # Read-only engineer-mode indicator.
        self._engineer_label = ctk.CTkLabel(
            self._indicator_frame,
            text="🛠 engineer",
            font=("Segoe UI Emoji", 12),
            text_color=self.HEADER_FG,
        )
        # Security-armed indicator (M34) — red: Jarvis speaks unprompted while
        # armed.
        self._armed_label = ctk.CTkLabel(
            self._indicator_frame,
            text="🛡 ARMED",
            font=("Segoe UI Emoji", 12, "bold"),
            text_color="#ef4444",
        )
        # Security-LOCKED indicator (M35) — deeper red; swapped in for ARMED
        # by _apply_locked when the deterrent has fired.
        self._locked_label = ctk.CTkLabel(
            self._indicator_frame,
            text="🔒 LOCKED",
            font=("Segoe UI Emoji", 12, "bold"),
            text_color="#dc2626",
        )
        # NB: indicator labels are not packed initially — _apply_* manage that.

        # --- Audio waveform visualizer (M17) ---
        # 24 bars that pulse with TTS amplitude. Idle state: flat resting line.
        # Pre-cache each bar's x1/x2 so the per-frame redraw only mutates y.
        # Sizes tuned in M17 review — bars feel substantial without dominating
        # the window. To resize: bar_width × num_bars + bar_gap × (num_bars-1)
        # should be ≤ canvas_width with comfortable horizontal padding.
        self._wave_num_bars = 28
        self._wave_bar_width = 7
        self._wave_bar_gap = 6
        self._wave_min_height = 3
        self._wave_max_height = 64
        self._wave_canvas_width = 480
        self._wave_canvas_height = 78
        # Resting bar colour (dim cyan); _wave_tick brightens each bar toward
        # ACCENT_HOT at its peak. Computed once — _dim 28×/frame is wasteful.
        self._wave_rest_color = self._dim(self.ACCENT, 0.42)

        self._wave_amplitude = 0.0          # latest reading from set_amplitude
        self._wave_displayed = [0.0] * self._wave_num_bars  # smoothed per-bar

        # Golden-angle phase distribution gives non-clustering, deterministic,
        # visually-irregular bar movement — no `random` import needed.
        golden = math.pi * (3 - math.sqrt(5))
        self._wave_phases = [
            (i * golden) % (2 * math.pi) for i in range(self._wave_num_bars)
        ]

        self._waveform_canvas = tk.Canvas(
            self.root,
            width=self._wave_canvas_width,
            height=self._wave_canvas_height,
            bg=self.BG,
            highlightthickness=0,
            bd=0,
        )
        self._waveform_canvas.pack(pady=(0, 12))

        # Pre-create the bars; each tick mutates only their y coords.
        self._wave_bars: list[int] = []
        self._wave_bar_x: list[tuple[int, int]] = []
        total_w = (
            self._wave_num_bars * (self._wave_bar_width + self._wave_bar_gap)
            - self._wave_bar_gap
        )
        start_x = (self._wave_canvas_width - total_w) // 2
        mid_y = self._wave_canvas_height // 2
        for i in range(self._wave_num_bars):
            x1 = start_x + i * (self._wave_bar_width + self._wave_bar_gap)
            x2 = x1 + self._wave_bar_width
            half = self._wave_min_height / 2
            bar_id = self._waveform_canvas.create_rectangle(
                x1, mid_y - half, x2, mid_y + half,
                fill=self._wave_rest_color, outline="",
            )
            self._wave_bars.append(bar_id)
            self._wave_bar_x.append((x1, x2))

        # --- Status bar (SRE-grade footer) ---
        # Packed FIRST with side="bottom" so it sits at the absolute bottom;
        # input_frame's subsequent side="bottom" pack lands directly above it.
        # Single-line, dim, monospace — non-intrusive but always-visible.
        # Format rebuilt by _update_status_text() whenever any status field
        # changes. Uptime ticks every 60s via a recursive .after().
        self._status_started_at = time.time()
        self._status_model = "(model unset)"
        self._status_tokens = 0
        # Map of integration name → enabled bool. Order preserved so dots
        # render in a stable left-to-right sequence regardless of when each
        # integration registers itself.
        self._status_integrations: dict[str, bool] = {}

        self._status_label = ctk.CTkLabel(
            self.root,
            text="",
            font=("Consolas", 10),
            text_color=self.DIM_FG,
            anchor="w",
        )
        self._status_label.pack(side="bottom", fill="x", padx=22, pady=(0, 6))
        self._update_status_text()

        # --- Text input row (packed second with side='bottom' so it sits
        # directly above the status bar; transcript above gets remaining vertical space) ---
        input_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        input_frame.pack(side="bottom", fill="x", padx=20, pady=(0, 8))

        # Chip frame for staged attachment — created here, but only packed
        # when a file is staged (pack_forget when not). Sits ABOVE the entry
        # row inside input_frame.
        self._chip_frame = ctk.CTkFrame(
            input_frame, fg_color=self.PANEL_BG, corner_radius=6, height=28
        )
        self._chip_label = ctk.CTkLabel(
            self._chip_frame,
            text="",
            font=("Consolas", 10),
            text_color=self.HEADER_FG,
            anchor="w",
        )
        self._chip_label.pack(side="left", padx=(10, 6), pady=4)
        self._chip_clear_btn = ctk.CTkButton(
            self._chip_frame,
            text="✕",
            width=24,
            height=20,
            font=("Consolas", 11),
            fg_color="transparent",
            text_color=self.DIM_FG,
            hover_color=self.BG,
            command=self._clear_attachment,
        )
        self._chip_clear_btn.pack(side="right", padx=(0, 6), pady=4)
        # NB: _chip_frame is NOT packed yet — _show_chip / _hide_chip manage that.

        # Entry row holds the 📎 attach button on the left and the entry on the right.
        self._entry_row = ctk.CTkFrame(input_frame, fg_color="transparent")
        self._entry_row.pack(fill="x")

        self._attach_button = ctk.CTkButton(
            self._entry_row,
            text="📎",
            width=38,
            height=38,
            font=("Segoe UI Emoji", 16),
            fg_color=self.PANEL_BG,
            text_color=self.HEADER_FG,
            hover_color=self.PANEL_LINE,
            border_width=1,
            border_color=self.BORDER,
            command=self._on_attach_clicked,
        )
        self._attach_button.pack(side="left", padx=(0, 6))

        self._input = ctk.CTkEntry(
            self._entry_row,
            placeholder_text="type to Jarvis…  (Enter to send)",
            font=("Consolas", 13),
            fg_color=self.PANEL_BG,
            text_color=self.USER_FG,
            border_width=1,
            border_color=self.BORDER,
            corner_radius=8,
            height=38,
        )
        self._input.pack(side="left", fill="x", expand=True)
        # Return submits; bind on the underlying widget so we can return
        # "break" to suppress Tk's default newline insertion.
        self._input.bind("<Return>", self._on_return)

        # --- Transcript pane ---
        transcript_frame = ctk.CTkFrame(
            self.root, fg_color=self.PANEL_BG, corner_radius=8,
            border_width=1, border_color=self.BORDER,
        )
        transcript_frame.pack(padx=20, pady=(0, 18), fill="both", expand=True)
        # HUD corner brackets on the four corners (placed, so they track resize).
        self._add_corner_brackets(transcript_frame)

        self._transcript = ctk.CTkTextbox(
            transcript_frame,
            font=("Consolas", 13),
            fg_color=self.PANEL_BG,
            text_color=self.USER_FG,
            wrap="word",
            corner_radius=6,
            border_width=0,
        )
        self._transcript.pack(padx=12, pady=12, fill="both", expand=True)

        # Tag-color the underlying tk.Text widget. CTkTextbox doesn't proxy
        # tag_configure, so we reach into ._textbox — that attribute has been
        # stable since customtkinter 5.x.
        tb = self._transcript._textbox
        tb.tag_configure("dim", foreground=self.DIM_FG)
        tb.tag_configure("user", foreground=self.USER_FG)
        tb.tag_configure("jarvis", foreground=self.JARVIS_FG)
        tb.tag_configure("system", foreground=self.DIM_FG)
        self._transcript.configure(state="disabled")

        # Initial empty-state hint
        self._append_raw("system", "ready. say 'hey jarvis' to begin.\n")

        # Window-close = hide (not quit). Quit comes from the tray menu.
        self.root.protocol("WM_DELETE_WINDOW", self.hide)

        # Kick off the animation ticks: state pulse (125ms = 8 fps) and the
        # waveform redraw (33ms = 30 fps). Two independent recursive .after()
        # chains so each can stop/continue without affecting the other.
        # Uptime tick is much slower (60s) — the only field that changes
        # autonomously.
        self.root.after(33, self._tick_orb)
        self.root.after(33, self._wave_tick)
        self.root.after(60_000, self._tick_uptime)
        # Dark-tint the native Windows title bar once the HWND exists (small
        # delay so winfo_id() is valid). Best-effort; cosmetic.
        self.root.after(80, self._apply_dark_titlebar)

    # ------------------------------------------------------------------
    # Public API — all thread-safe (schedule on Tk thread via .after()).
    # ------------------------------------------------------------------

    def set_state(self, state: State) -> None:
        if not self._destroyed:
            self.root.after(0, self._apply_state, state)

    def add_user_text(self, text: str, language: str = "en") -> None:
        if not self._destroyed:
            self.root.after(0, self._append_line, "you", text, language)

    def add_jarvis_text(self, text: str) -> None:
        if not self._destroyed:
            self.root.after(0, self._append_line, "jarvis", text, "")

    def add_system_text(self, text: str) -> None:
        if not self._destroyed:
            self.root.after(0, self._append_raw, "system", text + "\n")

    def set_on_text_submit(
        self, callback: Callable[[str, list[tuple[str, dict]]], None]
    ) -> None:
        """Register the function called when the user submits text. Receives
        (text, attachments) where attachments is a list of (filename, content_block)
        tuples — possibly empty. main.py wires this to a thread-safe queue so
        the callback can return immediately and Tk stays responsive."""
        self._on_text_submit = callback

    def set_amplitude(self, level: float) -> None:
        """Thread-safe: external setter for current TTS amplitude in [0, 1].
        Called ~30x/s by speak_streaming's envelope ticker thread. Latest
        reading replaces the stored value; _wave_tick handles smoothing +
        decay between updates so bars don't snap."""
        if self._destroyed:
            return
        clamped = max(0.0, min(1.0, float(level)))
        self.root.after(0, self._set_amplitude_internal, clamped)

    def set_muted(self, muted: bool) -> None:
        """Thread-safe: show/hide the '🔇 muted' indicator. Called from the
        UI coordinator when the tray's Mute checkbox is toggled."""
        if not self._destroyed:
            self.root.after(0, self._apply_muted, bool(muted))

    def set_engineer(self, on: bool) -> None:
        """Thread-safe: show/hide the '🛠 engineer' indicator. Called from
        the UI coordinator when the tray's Engineer mode checkbox is toggled."""
        if not self._destroyed:
            self.root.after(0, self._apply_engineer, bool(on))

    def set_armed(self, on: bool) -> None:
        """Thread-safe: show/hide the '🛡 ARMED' security-mode indicator (M34).
        Called from SecurityWatcher's on_armed_changed callback whenever the
        state flips via voice intent OR the tray toggle."""
        if not self._destroyed:
            self.root.after(0, self._apply_armed, bool(on))

    def set_locked(self, on: bool) -> None:
        """Thread-safe: show/hide the '🔒 LOCKED' security-mode indicator
        (M35 follow-on). Called from SecurityWatcher's on_locked_changed
        callback when the post-deterrent LOCKED state enters/exits.
        While LOCKED, the ARMED label is hidden — LOCKED implies armed and
        showing both would be visual noise."""
        if not self._destroyed:
            self.root.after(0, self._apply_locked, bool(on))

    # ------------------------------------------------------------------
    # SRE status bar API — all thread-safe via .after().
    # ------------------------------------------------------------------

    def set_model_name(self, model: str) -> None:
        """Display name for the model in the status bar. Strips the 'claude-'
        prefix for readability — 'sonnet-4-6' beats 'claude-sonnet-4-6'."""
        if self._destroyed:
            return
        clean = model.removeprefix("claude-") if model else "?"
        self.root.after(0, self._set_status_field, "_status_model", clean)

    def set_integration(self, name: str, enabled: bool) -> None:
        """Register an integration's status (e.g. 'plex', 'laptop'). The dot
        next to the name is filled when enabled, hollow when not. Multiple
        calls update in place — call once at startup, again later if the
        connection drops/recovers."""
        if self._destroyed:
            return
        self.root.after(0, self._set_integration_internal, name, bool(enabled))

    def add_session_tokens(self, n: int) -> None:
        """Increment the running session-total token counter. Called from the
        on_complete telemetry callback after each turn. Non-negative."""
        if self._destroyed or n <= 0:
            return
        self.root.after(0, self._add_session_tokens_internal, int(n))

    def add_telemetry_chip(self, text: str) -> None:
        """Append a small dim metadata line under the most recent transcript
        entry — typically right after a Jarvis response. Format is the
        caller's choice; we just render in dim style."""
        if not self._destroyed and text:
            self.root.after(0, self._append_telemetry_chip, text)

    def add_image_thumbnail(self, image_bytes: bytes, label: str) -> None:
        """Thread-safe: embed a small thumbnail of an image into the transcript,
        with a dim label above it. Called when a vision tool (camera_snapshot,
        screen_snapshot) successfully captures — the user sees *what Jarvis saw*
        inline with the conversation, not just the text description. Decoding
        and resizing happen on Tk's thread (cheap for ≤1568px source)."""
        if not self._destroyed and image_bytes:
            self.root.after(0, self._append_image_thumbnail, image_bytes, label)

    def show(self) -> None:
        if not self._destroyed:
            self.root.after(0, self._do_show)

    def hide(self) -> None:
        if not self._destroyed:
            self.root.withdraw()

    def run(self) -> None:
        """Blocks. Must be called from the thread that constructed this object
        (Tk requirement). Returns when shutdown() is called."""
        self.root.mainloop()
        self._destroyed = True

    def shutdown(self) -> None:
        """Stop mainloop. Safe to call from any thread."""
        try:
            self.root.after(0, self.root.quit)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals — only ever run on Tk's thread.
    # ------------------------------------------------------------------

    # ----- Colour helpers -----

    @staticmethod
    def _hex_rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    @staticmethod
    def _clamp8(v: float) -> int:
        return max(0, min(255, int(v)))

    def _dim(self, hexc: str, factor: float) -> str:
        """Scale a colour toward black by `factor` (0 = black, 1 = unchanged)."""
        r, g, b = self._hex_rgb(hexc)
        c = self._clamp8
        return f"#{c(r * factor):02x}{c(g * factor):02x}{c(b * factor):02x}"

    def _blend(self, a: str, b: str, t: float) -> str:
        """Linear blend a → b by t in [0, 1]."""
        t = max(0.0, min(1.0, t))
        ra, ga, ba = self._hex_rgb(a)
        rb, gb, bb = self._hex_rgb(b)
        c = self._clamp8
        return (f"#{c(ra + (rb - ra) * t):02x}{c(ga + (gb - ga) * t):02x}"
                f"{c(ba + (bb - ba) * t):02x}")

    # ----- The arc-reactor orb -----

    def _build_orb(self) -> None:
        """Create the orb's canvas items once. Layer order (back → front):
        bezel ring + 12 tick marks, the 4-disc radial glow halo (large →
        small), a mid ring, the core, then the two rotating arcs on top.
        _draw_orb mutates only fills / coords each frame."""
        cv = self._orb
        s = self.ORB_SIZE
        cx = cy = s // 2
        self._orb_cx, self._orb_cy = cx, cy
        r_out = s // 2 - 10
        r_mid = int(r_out * 0.80)
        line = self.PANEL_LINE

        # Bezel ring.
        cv.create_oval(cx - r_out, cy - r_out, cx + r_out, cy + r_out,
                       outline=line, width=2)
        # 12 clock-style tick marks just inside the bezel.
        for i in range(12):
            a = math.radians(i * 30)
            r1, r2 = r_out - 3, r_out - 11
            cv.create_line(
                cx + r1 * math.cos(a), cy + r1 * math.sin(a),
                cx + r2 * math.cos(a), cy + r2 * math.sin(a),
                fill=line, width=2,
            )
        # Radial glow halo — 4 filled discs, large → small. _draw_orb sets
        # their fills dim → bright each frame to fake a soft radial gradient.
        self._orb_glow: list[int] = []
        for rr in (int(r_out * 0.62), int(r_out * 0.46),
                   int(r_out * 0.32), int(r_out * 0.21)):
            self._orb_glow.append(
                cv.create_oval(cx - rr, cy - rr, cx + rr, cy + rr,
                               fill=self.BG, outline="")
            )
        # Mid ring — drawn after the halo so it stays visible on top of it.
        cv.create_oval(cx - r_mid, cy - r_mid, cx + r_mid, cy + r_mid,
                       outline=line, width=1)
        # Core.
        self._orb_core_r = max(9, int(r_out * 0.16))
        r = self._orb_core_r
        self._orb_core = cv.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        fill=self.ACCENT, outline="")
        # Two counter-rotating arcs (on top).
        self._orb_arc_out = cv.create_arc(
            cx - r_out, cy - r_out, cx + r_out, cy + r_out,
            start=0, extent=90, style="arc", outline=self.ACCENT, width=3,
        )
        self._orb_arc_mid = cv.create_arc(
            cx - r_mid, cy - r_mid, cx + r_mid, cy + r_mid,
            start=180, extent=60, style="arc", outline=self.ACCENT, width=2,
        )

    def _draw_orb(self) -> bool:
        """One animation frame. Colour = state; the core breathes (and, while
        SPEAKING, throbs with the live TTS amplitude); the two arcs counter-
        rotate. Returns False if the canvas is gone (stop scheduling)."""
        state = self._state
        base = self._STATE_COLOR.get(state, self.ACCENT)
        t = time.time() - self._anim_start

        if state == State.SPEAKING:
            # The reactor reacts to Jarvis's own voice.
            pulse = 0.42 + 0.58 * min(1.0, self._wave_amplitude * 1.5)
        elif state == State.IDLE:
            pulse = 0.50 + 0.16 * math.sin(t * 1.6)      # slow breathing
        elif state == State.THINKING:
            pulse = 0.74 + 0.16 * math.sin(t * 5.0)      # quick flicker
        else:  # LISTENING
            pulse = 0.84 + 0.12 * math.sin(t * 3.0)

        speed = {                                         # degrees / second
            State.IDLE: 24, State.LISTENING: 64,
            State.THINKING: 150, State.SPEAKING: 92,
        }.get(state, 50)
        ang = (t * speed) % 360

        try:
            self._orb.itemconfig(
                self._orb_arc_out, start=ang,
                outline=self._blend(self._dim(base, 0.35), base, pulse),
            )
            self._orb.itemconfig(
                self._orb_arc_mid, start=(-ang * 1.4) % 360,
                outline=self._dim(base, 0.5 + 0.3 * pulse),
            )
            for gid, f in zip(self._orb_glow, (0.18, 0.34, 0.58, 0.92)):
                self._orb.itemconfig(gid, fill=self._dim(base, f * pulse))
            self._orb.itemconfig(
                self._orb_core,
                fill=self._blend(base, self.ACCENT_HOT, 0.25 + 0.55 * pulse),
            )
            r = self._orb_core_r * (0.78 + 0.5 * pulse)
            cx, cy = self._orb_cx, self._orb_cy
            self._orb.coords(self._orb_core, cx - r, cy - r, cx + r, cy + r)
        except tk.TclError:
            return False
        return True

    def _tick_orb(self) -> None:
        """30 fps arc-reactor animation — its own recursive .after() chain,
        same pattern as _wave_tick."""
        if self._destroyed:
            return
        if not self._draw_orb():
            return  # canvas destroyed; stop scheduling
        self.root.after(33, self._tick_orb)

    def _add_corner_brackets(self, frame) -> None:
        """Place four small L-shaped HUD brackets at the corners of `frame`.
        Each is a tiny canvas .place()'d at a corner; relx/rely keeps them
        pinned through window resizes."""
        arm = 16
        # (relx, rely, anchor, x-offset, y-offset, [(x1,y1,x2,y2), ...])
        corners = [
            (0.0, 0.0, "nw", 6, 6, [(0, arm, 0, 0), (0, 0, arm, 0)]),
            (1.0, 0.0, "ne", -6, 6, [(arm, arm, arm, 0), (arm, 0, 0, 0)]),
            (0.0, 1.0, "sw", 6, -6, [(0, 0, 0, arm), (0, arm, arm, arm)]),
            (1.0, 1.0, "se", -6, -6, [(arm, 0, arm, arm), (arm, arm, 0, arm)]),
        ]
        for relx, rely, anchor, dx, dy, lines in corners:
            c = tk.Canvas(frame, width=arm + 1, height=arm + 1,
                          bg=self.PANEL_BG, highlightthickness=0, bd=0)
            for x1, y1, x2, y2 in lines:
                c.create_line(x1, y1, x2, y2, fill=self.ACCENT, width=2)
            c.place(in_=frame, relx=relx, rely=rely, anchor=anchor, x=dx, y=dy)

    def _apply_dark_titlebar(self) -> None:
        """Tint the native Windows title bar to match the app (Win10 2004+ /
        Win11). Without this the OS paints the caption with the user's accent
        colour. Best-effort — any failure (non-Windows, old build) is silently
        ignored; it's purely cosmetic."""
        if sys.platform != "win32":
            return
        try:
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            dwm = ctypes.windll.dwmapi
            # DWMWA_USE_IMMERSIVE_DARK_MODE (20) — dark caption + controls.
            dwm.DwmSetWindowAttribute(
                hwnd, 20, ctypes.byref(ctypes.c_int(1)),
                ctypes.sizeof(ctypes.c_int),
            )
            # DWMWA_CAPTION_COLOR (35, Win11) — exact tint as 0x00BBGGRR.
            r, g, b = self._hex_rgb(self.BG)
            colour = (b << 16) | (g << 8) | r
            dwm.DwmSetWindowAttribute(
                hwnd, 35, ctypes.byref(ctypes.c_int(colour)),
                ctypes.sizeof(ctypes.c_int),
            )
        except Exception:
            pass

    def _apply_state(self, state: State) -> None:
        self._state = state
        try:
            self._state_label.configure(text=state.name)
        except tk.TclError:
            pass

    def _apply_muted(self, muted: bool) -> None:
        try:
            if muted:
                # padx pushes the indicator a bit off the state label so they
                # read as related-but-distinct.
                self._mute_label.pack(side="left", padx=(8, 0))
            else:
                self._mute_label.pack_forget()
        except tk.TclError:
            pass

    def _apply_engineer(self, on: bool) -> None:
        try:
            if on:
                self._engineer_label.pack(side="left", padx=(8, 0))
            else:
                self._engineer_label.pack_forget()
        except tk.TclError:
            pass

    def _apply_armed(self, on: bool) -> None:
        try:
            if on:
                self._armed_label.pack(side="left", padx=(8, 0))
            else:
                self._armed_label.pack_forget()
        except tk.TclError:
            pass

    def _apply_locked(self, on: bool) -> None:
        """Swap between 🛡 ARMED and 🔒 LOCKED. While LOCKED is on, the
        ARMED label is hidden (LOCKED already implies armed; showing both
        would be visual noise). On LOCKED off, restore the ARMED label
        only if the watcher is still armed — which it should be, since
        deactivate() clears _locked too."""
        try:
            if on:
                # Hide ARMED first, then show LOCKED in its place so the
                # layout doesn't briefly show both.
                self._armed_label.pack_forget()
                self._locked_label.pack(side="left", padx=(8, 0))
            else:
                self._locked_label.pack_forget()
                # Show ARMED back — the watcher is still armed unless a
                # separate _apply_armed(False) is also pending. Both are
                # marshaled onto Tk's thread sequentially, so order is
                # deterministic per call site (security.py fires
                # on_locked_changed first, then on_armed_changed on
                # deactivate).
                self._armed_label.pack(side="left", padx=(8, 0))
        except tk.TclError:
            pass

    # ----- Status bar internals (Tk-thread only) -----

    def _set_status_field(self, attr: str, value) -> None:
        setattr(self, attr, value)
        self._update_status_text()

    def _set_integration_internal(self, name: str, enabled: bool) -> None:
        self._status_integrations[name] = enabled
        self._update_status_text()

    def _add_session_tokens_internal(self, n: int) -> None:
        self._status_tokens += n
        self._update_status_text()

    def _format_uptime(self) -> str:
        secs = int(time.time() - self._status_started_at)
        h, rem = divmod(secs, 3600)
        m, _ = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m}m"
        return f"{m}m"

    def _update_status_text(self) -> None:
        """Recompose the footer text from current state. Filled circle (●)
        for enabled integrations, hollow (○) for not. Single dim line, no
        wrapping — fits within a 520px window comfortably."""
        parts = [
            self._status_model,
            f"up {self._format_uptime()}",
            f"{self._status_tokens:,} tok",
        ]
        # Integrations get a fixed left-to-right ordering by registration time.
        for name, enabled in self._status_integrations.items():
            dot = "●" if enabled else "○"
            parts.append(f"{name} {dot}")
        try:
            self._status_label.configure(text="  ·  ".join(parts))
        except tk.TclError:
            pass

    def _tick_uptime(self) -> None:
        """Refresh the uptime field every 60s. Cheap; just rebuilds the
        label text. Stops itself if the window is destroyed."""
        if self._destroyed:
            return
        self._update_status_text()
        self.root.after(60_000, self._tick_uptime)

    def _append_telemetry_chip(self, text: str) -> None:
        """Add a small dim metadata line to the transcript, indented to align
        with the response text above it. Sits inline with the conversation
        rather than as a popover — keeps the SRE signal close to the answer
        it describes."""
        try:
            self._transcript.configure(state="normal")
            tb = self._transcript._textbox
            # Indent matches the timestamp prefix width so the chip lines up
            # under "you:" / "jarvis:" labels visually.
            tb.insert("end", "\n        " + text, "dim")
            tb.see("end")
            self._transcript.configure(state="disabled")
        except tk.TclError:
            pass

    # Thumbnail target width — wide enough to read UI elements in a screen
    # capture, narrow enough to leave the transcript readable around it.
    # Height auto-derived from source aspect ratio.
    _THUMBNAIL_WIDTH = 240
    # Cap on the click-to-enlarge popup. Slightly under a 1080p display so
    # the window fits with its OS chrome on most monitors. Aspect preserved.
    _ENLARGE_MAX_W = 1400
    _ENLARGE_MAX_H = 900

    def _append_image_thumbnail(self, image_bytes: bytes, label: str) -> None:
        """Decode bytes → PIL Image → resize → PhotoImage → insert into the
        transcript via tk.Text.image_create. Runs on Tk's thread (scheduled
        via .after by add_image_thumbnail). Defensive — a bad image must
        never break the transcript pane.

        Also wires per-thumbnail click + hover bindings: clicking the image
        opens a full-resolution popup; hovering changes the cursor to a hand
        so the affordance is discoverable."""
        try:
            pil = Image.open(io.BytesIO(image_bytes))
            w, h = pil.size
            if w > self._THUMBNAIL_WIDTH:
                ratio = self._THUMBNAIL_WIDTH / w
                pil = pil.resize(
                    (self._THUMBNAIL_WIDTH, int(h * ratio)),
                    Image.LANCZOS,
                )
            # ImageTk.PhotoImage is the standard tk.Text-embeddable image
            # type. CTkImage is for CTk widgets (CTkButton, CTkLabel) and
            # doesn't accept tk.Text.image_create cleanly — different path.
            photo = ImageTk.PhotoImage(pil)
        except Exception as exc:
            print(f"[console] thumbnail decode failed: {exc}")
            return

        # Index assigned BEFORE appending — used as the closure-captured key
        # so the click handler knows which thumbnail it owns even after
        # later thumbnails are added.
        idx = len(self._thumbnail_refs)
        self._thumbnail_refs.append((photo, image_bytes, label))

        try:
            self._transcript.configure(state="normal")
            tb = self._transcript._textbox
            # Match the 8-space indent of telemetry chips so labels align
            # visually under "you:" / "jarvis:" rows above.
            tb.insert("end", f"\n        {label}\n        ", "dim")

            # Capture the insertion point before image_create — we need it
            # to apply a per-thumbnail tag (one image == one character in
            # the text widget, so the tag covers exactly the image position).
            img_pos = tb.index("end-1c")  # one char before trailing implicit newline
            tb.image_create("end", image=photo)
            tag = f"thumb_{idx}"
            tb.tag_add(tag, img_pos, f"{img_pos}+1c")
            # Bindings via lambda default-arg capture so each thumbnail's
            # handler closes over its OWN idx (not the last value of the
            # loop variable).
            tb.tag_bind(tag, "<Button-1>", lambda _e, i=idx: self._open_thumbnail_popup(i))
            tb.tag_bind(tag, "<Enter>", lambda _e: tb.config(cursor="hand2"))
            tb.tag_bind(tag, "<Leave>", lambda _e: tb.config(cursor=""))

            tb.insert("end", "\n        (click to enlarge)\n", "dim")
            tb.see("end")
            self._transcript.configure(state="disabled")
        except tk.TclError:
            pass

    def _open_thumbnail_popup(self, idx: int) -> None:
        """Open a CTkToplevel showing the full-resolution captured image.
        Same dark theme as the main window. Esc / window-X closes. Multiple
        popups can coexist — useful if the user wants to compare a screen
        snapshot to a previous one. Defensive — re-decode failure logs +
        skips silently rather than crashing the UI."""
        if idx < 0 or idx >= len(self._thumbnail_refs):
            return
        _photo, image_bytes, label = self._thumbnail_refs[idx]

        try:
            pil = Image.open(io.BytesIO(image_bytes))
        except Exception as exc:
            print(f"[console] enlarge decode failed: {exc}")
            return

        w, h = pil.size
        if w > self._ENLARGE_MAX_W or h > self._ENLARGE_MAX_H:
            ratio = min(self._ENLARGE_MAX_W / w, self._ENLARGE_MAX_H / h)
            pil = pil.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        full_photo = ImageTk.PhotoImage(pil)

        try:
            win = ctk.CTkToplevel(self.root)
            win.title(label)
            win.configure(fg_color=self.BG)
            # transient ties the popup to the main window so it follows
            # min/restore and stays on top of the main window without
            # being a global always-on-top window.
            win.transient(self.root)

            lbl = tk.Label(win, image=full_photo, bg=self.BG, borderwidth=0)
            # Strong ref on the Label keeps the PhotoImage alive for the
            # popup's lifetime; closing the window GCs both together.
            lbl.image = full_photo  # type: ignore[attr-defined]
            lbl.pack(padx=10, pady=10)

            win.bind("<Escape>", lambda _e: win.destroy())
        except tk.TclError as exc:
            print(f"[console] popup open failed: {exc}")

    def _set_amplitude_internal(self, level: float) -> None:
        # Runs on Tk's thread (scheduled via .after). Just store; redraw
        # happens on the next _wave_tick frame.
        self._wave_amplitude = level

    def _wave_tick(self) -> None:
        """30 fps waveform redraw. Decays stored amplitude every frame so bars
        fall after audio stops; per-bar phase oscillation keeps the visualizer
        alive even at constant amplitude (mimics a real EQ meter); smoothing
        prevents single-frame jitter from being visually harsh."""
        if self._destroyed:
            return

        # Decay factor 0.85: ~99% gone over 0.3s. Snappy attack-and-release.
        self._wave_amplitude *= 0.85
        if self._wave_amplitude < 0.005:
            self._wave_amplitude = 0.0

        t = time.time()
        smoothing = 0.4
        span = self._wave_max_height - self._wave_min_height
        mid_y = self._wave_canvas_height // 2

        try:
            for i, bar_id in enumerate(self._wave_bars):
                # Each bar: amplitude × (0.5 + 0.5 × sin(t·ω + phase)) gives
                # a [0.5×amp, 1.0×amp] band. Bars are partially correlated
                # (all driven by amplitude) but phase-offset for visual life.
                oscillation = math.sin(t * 6 + self._wave_phases[i]) * 0.5 + 0.5
                # Faint idle shimmer (~2px) so the bar field reads as alive
                # even at rest, rather than a dead flat line.
                idle = 0.035 * (0.5 + 0.5 * math.sin(t * 2.2 + self._wave_phases[i]))
                target = self._wave_amplitude * (0.5 + 0.5 * oscillation) + idle
                self._wave_displayed[i] += (target - self._wave_displayed[i]) * smoothing

                level = min(1.0, self._wave_displayed[i])
                height = self._wave_min_height + level * span
                half = height / 2
                x1, x2 = self._wave_bar_x[i]
                self._waveform_canvas.coords(
                    bar_id, x1, mid_y - half, x2, mid_y + half
                )
                # Hotter colour toward the peaks — dim cyan → near-white-hot.
                self._waveform_canvas.itemconfig(
                    bar_id,
                    fill=self._blend(self._wave_rest_color, self.ACCENT_HOT, level),
                )
        except tk.TclError:
            return  # canvas destroyed; stop scheduling

        self.root.after(33, self._wave_tick)

    def _append_line(self, who: str, text: str, language: str) -> None:
        ts = time.strftime("%H:%M")
        try:
            self._transcript.configure(state="normal")
            tb = self._transcript._textbox
            tb.insert("end", f"\n[{ts}] ", "dim")
            if who == "you":
                label = "you" + (f" ({language})" if language and language != "en" else "")
                tb.insert("end", f"{label}: ", "dim")
                tb.insert("end", text, "user")
            else:
                tb.insert("end", "jarvis: ", "dim")
                tb.insert("end", text, "jarvis")
            tb.see("end")
            self._transcript.configure(state="disabled")
        except tk.TclError:
            pass

    def _append_raw(self, tag: str, text: str) -> None:
        try:
            self._transcript.configure(state="normal")
            self._transcript._textbox.insert("end", text, tag)
            self._transcript._textbox.see("end")
            self._transcript.configure(state="disabled")
        except tk.TclError:
            pass

    def _do_show(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.attributes("-topmost", False)

    def _on_return(self, event):  # noqa: ARG002 — Tk hands us the event
        """Enter pressed in the input field. Grab the text + any staged
        attachment, clear both, hand off to the registered callback.

        If a file is staged, prepend "[📎 filename] " to the typed text so
        the filename is visible in both the transcript and in the message
        Claude sees. Empty input with no attachment = no-op.

        Returning 'break' suppresses Tk's default newline-insert behavior.
        """
        text = self._input.get().strip()
        staged = self._staged

        if not text and staged is None:
            return "break"  # nothing to send

        # Clear entry + chip BEFORE firing the callback, so the UI feels
        # snappy and a slow downstream can't visually "stick" the staged file.
        self._input.delete(0, "end")
        self._clear_attachment()

        attachments: list[tuple[str, dict]] = []
        if staged is not None:
            filename, block = staged
            attachments.append(staged)
            text = (f"[📎 {filename}] " + text).rstrip()

        if self._on_text_submit is not None:
            try:
                self._on_text_submit(text, attachments)
            except Exception as exc:
                print(f"[console] text submit callback raised: {exc}")
        return "break"

    def _on_attach_clicked(self) -> None:
        """Open the file picker. Tk's askopenfilename blocks the GUI thread
        until the dialog closes — that's fine because the worker threads
        (listen_loop, text_input_loop) keep running independently."""
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Attach a file for Jarvis",
            filetypes=[
                ("Documents", "*.pdf"),
                ("Images", "*.png *.jpg *.jpeg *.gif *.webp"),
                ("Text / code", "*.txt *.md *.csv *.json *.yaml *.yml *.log *.py *.js *.ts *.html *.css"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return  # user cancelled
        block, error = load_attachment(path)
        if error is not None or block is None:
            self.add_system_text(f"attachment: {error or 'unknown error'}")
            return
        # Best-effort filename for display + the prefix we inject on submit.
        filename = Path(path).name
        self._staged = (filename, block)
        self._show_chip(filename)

    def _clear_attachment(self) -> None:
        """Remove the staged attachment + hide the chip. Called after submit
        OR by the chip's ✕ button."""
        self._staged = None
        self._hide_chip()

    def _show_chip(self, filename: str) -> None:
        """Reveal the chip with the staged filename. Pack BEFORE the entry_row
        so it appears above. Truncate long filenames so the chip stays compact."""
        display = filename if len(filename) <= 50 else filename[:47] + "…"
        self._chip_label.configure(text=f"📄 {display}")
        # Pack BEFORE the entry row so the chip lands above the input.
        self._chip_frame.pack(fill="x", pady=(0, 6), before=self._entry_row)

    def _hide_chip(self) -> None:
        try:
            self._chip_frame.pack_forget()
        except tk.TclError:
            pass
