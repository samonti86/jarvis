"""JarvisConsole — dark-themed customtkinter window showing live state + transcript.

Threading model:
- Tk's mainloop runs on the main thread (its preferred home — Tk is single-threaded).
- All public methods are thread-safe: they schedule the actual widget update via
  self.root.after(0, ...), which marshals the call onto Tk's thread.
- An .after(125, _tick_animation) recursive scheduler drives the SPEAKING-state
  pulse at 8 fps, all on Tk's thread — no separate animation thread needed.

Text input (M15): a CTkEntry at the bottom lets the user type questions instead
of speaking. Submit fires the on_text_submit callback (registered by main.py)
which enqueues the text for the same per-turn pipeline as voice. Closing the
window still hides; control (reset, open log, quit) still lives on the tray.
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from typing import Callable

import customtkinter as ctk

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
        self._on_text_submit: Callable[[str], None] | None = None

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

        # --- Text input row (packed first with side='bottom' so it sticks to
        # the bottom and the transcript above gets all remaining vertical space) ---
        input_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        input_frame.pack(side="bottom", fill="x", padx=20, pady=(0, 16))

        self._input = ctk.CTkEntry(
            input_frame,
            placeholder_text="type to Jarvis…  (Enter to send)",
            font=("Consolas", 11),
            fg_color=self.PANEL_BG,
            text_color=self.USER_FG,
            border_width=0,
            corner_radius=6,
            height=34,
        )
        self._input.pack(fill="x", expand=True)
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

        # Kick off the animation tick.
        self.root.after(125, self._tick_animation)

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

    def set_on_text_submit(self, callback: Callable[[str], None]) -> None:
        """Register the function called when the user submits text. Receives
        the typed string. main.py wires this to a thread-safe queue so the
        callback can return immediately and Tk stays responsive."""
        self._on_text_submit = callback

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
        """Enter pressed in the input field. Grab the text, clear the entry,
        hand off to the registered callback. Returning 'break' tells Tk to
        suppress the default newline behavior."""
        text = self._input.get().strip()
        self._input.delete(0, "end")
        if text and self._on_text_submit is not None:
            try:
                self._on_text_submit(text)
            except Exception as exc:
                print(f"[console] text submit callback raised: {exc}")
        return "break"
