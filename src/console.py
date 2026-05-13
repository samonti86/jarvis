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

import io
import math
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
    # Palette — restrained sci-fi: dark slate + light cyan accent.
    BG = "#1a1d23"
    PANEL_BG = "#22262e"
    HEADER_FG = "#7dd3fc"
    USER_FG = "#e2e8f0"
    JARVIS_FG = "#7dd3fc"
    DIM_FG = "#64748b"

    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Jarvis")
        self.root.geometry("520x640")
        self.root.configure(fg_color=self.BG)
        self.root.minsize(400, 480)

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
        # Thumbnails embedded in the transcript via image_create — Tk weakly
        # references images, so without a strong ref here they'd GC and the
        # transcript would show empty placeholders. Grows unbounded across the
        # session, but each entry is ~240x135 RGB (~100 KB live) so even a
        # very chatty session is bounded in memory. Cleared on shutdown
        # implicitly via window destruction.
        self._thumbnail_refs: list[ImageTk.PhotoImage] = []

        # --- Header ---
        header = ctk.CTkLabel(
            self.root,
            text="J . A . R . V . I . S .",
            font=("Consolas", 22, "bold"),
            text_color=self.HEADER_FG,
        )
        header.pack(pady=(22, 4))

        subtitle = ctk.CTkLabel(
            self.root,
            text="at your service, sir",
            font=("Consolas", 9),
            text_color=self.DIM_FG,
        )
        subtitle.pack(pady=(0, 18))

        # --- State pill ---
        state_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        state_frame.pack(pady=(0, 16))

        self._state_canvas = tk.Canvas(
            state_frame,
            width=22,
            height=22,
            bg=self.BG,
            highlightthickness=0,
            bd=0,
        )
        self._state_canvas.pack(side="left", padx=(0, 12))
        self._state_circle = self._state_canvas.create_oval(
            3, 3, 19, 19, fill=self._color_for(State.IDLE), outline=""
        )

        self._state_label = ctk.CTkLabel(
            state_frame,
            text="IDLE",
            font=("Consolas", 14, "bold"),
            text_color=self.HEADER_FG,
            width=110,
            anchor="w",
        )
        self._state_label.pack(side="left")

        # Read-only mute indicator. Packed only while muted (controlled by
        # set_muted). Toggle itself lives on the tray menu — keeping the
        # console as a passive surface here so we don't duplicate state.
        self._mute_label = ctk.CTkLabel(
            state_frame,
            text="🔇 muted",
            font=("Segoe UI Emoji", 11),
            text_color=self.DIM_FG,
        )
        # NB: not packed initially — _apply_muted handles pack/forget.

        # Read-only engineer-mode indicator, same shape as the mute label.
        # Sits to the right of the mute label when both are active.
        self._engineer_label = ctk.CTkLabel(
            state_frame,
            text="🛠 engineer",
            font=("Segoe UI Emoji", 11),
            text_color=self.HEADER_FG,
        )
        # NB: not packed initially — _apply_engineer handles pack/forget.

        # --- Audio waveform visualizer (M17) ---
        # 24 bars that pulse with TTS amplitude. Idle state: flat resting line.
        # Pre-cache each bar's x1/x2 so the per-frame redraw only mutates y.
        # Sizes tuned in M17 review — bars feel substantial without dominating
        # the window. To resize: bar_width × num_bars + bar_gap × (num_bars-1)
        # should be ≤ canvas_width with comfortable horizontal padding.
        self._wave_num_bars = 24
        self._wave_bar_width = 7
        self._wave_bar_gap = 5
        self._wave_min_height = 3
        self._wave_max_height = 60
        self._wave_canvas_width = 440
        self._wave_canvas_height = 72

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
                fill=self.HEADER_FG, outline="",
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
            font=("Consolas", 9),
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
            width=34,
            height=34,
            font=("Segoe UI Emoji", 14),
            fg_color=self.PANEL_BG,
            text_color=self.HEADER_FG,
            hover_color=self.BG,
            command=self._on_attach_clicked,
        )
        self._attach_button.pack(side="left", padx=(0, 6))

        self._input = ctk.CTkEntry(
            self._entry_row,
            placeholder_text="type to Jarvis…  (Enter to send)",
            font=("Consolas", 11),
            fg_color=self.PANEL_BG,
            text_color=self.USER_FG,
            border_width=0,
            corner_radius=6,
            height=34,
        )
        self._input.pack(side="left", fill="x", expand=True)
        # Return submits; bind on the underlying widget so we can return
        # "break" to suppress Tk's default newline insertion.
        self._input.bind("<Return>", self._on_return)

        # --- Transcript pane ---
        transcript_frame = ctk.CTkFrame(self.root, fg_color=self.PANEL_BG, corner_radius=8)
        transcript_frame.pack(padx=20, pady=(0, 20), fill="both", expand=True)

        self._transcript = ctk.CTkTextbox(
            transcript_frame,
            font=("Consolas", 11),
            fg_color=self.PANEL_BG,
            text_color=self.USER_FG,
            wrap="word",
            corner_radius=6,
            border_width=0,
        )
        self._transcript.pack(padx=10, pady=10, fill="both", expand=True)

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
        self.root.after(125, self._tick_animation)
        self.root.after(33, self._wave_tick)
        self.root.after(60_000, self._tick_uptime)

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

    def _color_for(self, state: State) -> str:
        r, g, b = state.value
        return f"#{r:02x}{g:02x}{b:02x}"

    def _scaled_color(self, state: State, brightness: float) -> str:
        r, g, b = state.value
        r, g, b = int(r * brightness), int(g * brightness), int(b * brightness)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _tick_animation(self) -> None:
        if self._state == State.SPEAKING:
            t = time.time() - self._anim_start
            brightness = 0.7 + 0.3 * math.sin(t * 2 * math.pi * 2)
            color = self._scaled_color(self._state, brightness)
        else:
            color = self._color_for(self._state)
        try:
            self._state_canvas.itemconfig(self._state_circle, fill=color)
        except tk.TclError:
            return  # widget destroyed; stop scheduling
        self.root.after(125, self._tick_animation)

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

    def _append_image_thumbnail(self, image_bytes: bytes, label: str) -> None:
        """Decode bytes → PIL Image → resize → PhotoImage → insert into the
        transcript via tk.Text.image_create. Runs on Tk's thread (scheduled
        via .after by add_image_thumbnail). Defensive — a bad image must
        never break the transcript pane."""
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
            self._thumbnail_refs.append(photo)  # strong ref so Tk doesn't GC
        except Exception as exc:
            print(f"[console] thumbnail decode failed: {exc}")
            return

        try:
            self._transcript.configure(state="normal")
            tb = self._transcript._textbox
            # Match the 8-space indent of telemetry chips so labels align
            # visually under "you:" / "jarvis:" rows above.
            tb.insert("end", f"\n        {label}\n        ", "dim")
            tb.image_create("end", image=photo)
            tb.insert("end", "\n", "dim")
            tb.see("end")
            self._transcript.configure(state="disabled")
        except tk.TclError:
            pass

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
                target = self._wave_amplitude * (0.5 + 0.5 * oscillation)
                self._wave_displayed[i] += (target - self._wave_displayed[i]) * smoothing

                height = self._wave_min_height + self._wave_displayed[i] * span
                half = height / 2
                x1, x2 = self._wave_bar_x[i]
                self._waveform_canvas.coords(
                    bar_id, x1, mid_y - half, x2, mid_y + half
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
