"""Anticipatory intelligence (M83) — "Sir, I've taken the liberty…".

The defining film-JARVIS trait: he connects dots across everything and
volunteers the ONE thing that matters, before you ask. Every other proactive
subsystem here is single-signal (homelab down, calendar event soon, severe
weather). This one is the SYNTHESIS layer: a background pass that reads the live
state of ALL of them at once and asks Claude, acting as an extremely-selective
chief-of-staff, whether there's a single genuinely-useful cross-domain
connection worth surfacing right now.

Design
------
- **A poll thread** (the homelab_monitor / calendar_monitor pattern), default
  every 30 min. Each tick: gather a world-state snapshot (injected getters, so
  this module is decoupled + testable), ask Claude, maybe announce.
- **The high bar is the whole game.** The system prompt makes "stay silent"
  (PASS) the default; an insight must clear a usefulness/timeliness/non-obvious
  bar AND not repeat anything recently said. Most ticks return PASS — that's
  correct behaviour, not a bug. A chatty anticipation engine is a failed one.
- **Cross-domain is the value.** "Your 8am is in Denver, NWS just warned of a 7am
  storm there, and with no UPS you'll want to shut the desktop down before you
  go" touches calendar + weather + a documented pain point in one sentence —
  something no single monitor could produce.
- **Rides `_announce`** (the WASAPI-safe path) tagged 🧠, so it's gated by the
  M79 quiet-hours policy. Anticipation labels DEFER during quiet hours (added to
  _DEFERRABLE_LABELS): an insight is rarely so urgent it can't wait for the
  morning catch-up, and genuinely urgent single-signal events (a severe storm)
  have their OWN piercing monitor. So anticipation defers cleanly to the morning
  "while you were away" digest.
- **Dedup** via a rolling list of recent insights fed back into the prompt
  ("don't repeat these") plus a local similarity guard, so a standing situation
  isn't re-announced every 30 minutes.

Cost: one modest LLM call per tick (world-state ~hundreds of tokens). Default
OFF (opt-in via JARVIS_ANTICIPATION=1) — it's a continuous, if small, recurring
spend, so we don't enable it without the user asking, per the project's cost
discipline.

Defensive contract: nothing here raises into the poll thread or the announce
path. A failed snapshot/LLM call logs and waits for the next tick.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable

ANNOUNCE_LABEL = "🧠"


# --- The selectivity prompt (the load-bearing part) ------------------------
ANTICIPATION_SYSTEM = """You are the anticipation module of Jarvis, a personal assistant for Saul (a British-butler-style AI in the spirit of Tony Stark's J.A.R.V.I.S.).

You run silently in the background. You are given a snapshot of Saul's current world — calendar, weather, active weather alerts, reminders, homelab health, security state, the time. Your ONE job: decide whether there is a single genuinely useful thing Saul would want to hear RIGHT NOW that he probably does not already know.

Be EXTREMELY selective. Staying silent is almost always the right call. Only speak if an item clears ALL of these:
- USEFUL and ACTIONABLE — it changes what Saul might do, not just trivia.
- TIMELY — it matters now or very soon, not someday.
- NON-OBVIOUS — a great chief-of-staff insight, especially a CROSS-DOMAIN connection between two signals (e.g. a calendar event + a weather alert, a reminder + the homelab state). A single fact one monitor would already report on its own is NOT enough.
- NOT ALREADY KNOWN — don't state the obvious, and never repeat anything in the "already mentioned" list.

Known context worth connecting: the primary machine has no UPS and will not auto-power-on, so a power-threatening storm may warrant a manual shutdown before leaving.

OUTPUT FORMAT — exactly one of:
- The single word: PASS   (when nothing clears the bar — this is the common case)
- ONE short spoken sentence in Jarvis's voice (concise, dryly courteous, address him as "sir" sparingly, no markdown, no preamble). It will be read aloud, so keep it to one or two sentences.

Do not explain your reasoning. Output PASS or the sentence, nothing else."""


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return max(minimum, int(raw)) if raw else default
    except ValueError:
        return default


def is_enabled() -> bool:
    """Opt-in (default OFF) — anticipation makes a recurring LLM call, so it's
    enabled only when the user asks via JARVIS_ANTICIPATION=1."""
    return os.getenv("JARVIS_ANTICIPATION", "").strip().lower() in ("1", "true", "yes", "on")


_POLL_SECONDS = _env_int("JARVIS_ANTICIPATION_POLL_SECONDS", 1800, 60)


# --- Pure helpers (no threads / no network) --------------------------------

def _format_world_state(snapshot: dict, now: datetime, recent: list[str]) -> str:
    """Compose the world-state user message from a {section: text|None} dict.
    Empty/None sections are skipped so the model isn't fed blank headers. The
    'already mentioned' list is appended so the model won't repeat itself."""
    lines = [f"Current time: {now.strftime('%A, %B %d, %I:%M %p')}"]
    for name, val in snapshot.items():
        v = (val or "").strip()
        if v:
            lines.append(f"\n[{name}]\n{v}")
    block = "\n".join(lines)
    if recent:
        block += ("\n\n[Already mentioned recently — do NOT repeat these]\n"
                  + "\n".join(f"- {r}" for r in recent))
    return block


def _parse_decision(reply: str) -> "str | None":
    """Map the model's reply to an insight or None. The model is told to output
    exactly 'PASS' when there's nothing — but be tolerant of 'PASS.', a quoted
    sentence, or a short 'PASS, nothing notable.' Returns the cleaned sentence,
    or None to stay silent. Pure + trivially testable."""
    text = (reply or "").strip().strip('"“”\'*').strip()
    if not text:
        return None
    if text.rstrip(" .!").upper() == "PASS":
        return None
    # Belt-and-suspenders: a short reply that merely starts with PASS is a
    # decline ("PASS — nothing to report"), not a real one-line insight.
    if text.upper().startswith("PASS") and len(text) <= 40:
        return None
    return text


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace for the dedup similarity check."""
    return " ".join((text or "").lower().split())


# --- The real LLM call (injectable for tests) ------------------------------

def _default_ask(api_key: str, model: str, system: str, user: str) -> str:
    """One non-streaming Claude call returning the raw text. Lazy-imports
    anthropic (the vision_describe / summarizer pattern). Raises on transport
    failure — the engine catches it and waits for the next tick."""
    import anthropic  # noqa: PLC0415 — lazy, already a project dep

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "".join(parts).strip()


# --- The engine ------------------------------------------------------------

class AnticipationEngine:
    """Background synthesis loop. Construct always (cheap); the poll loop only
    runs between activate() and deactivate(), mirroring the other monitors."""

    def __init__(
        self,
        announce: Callable[[str], None],
        snapshot_fn: Callable[[], dict],
        *,
        api_key: str = "",
        model: str = "claude-sonnet-4-6",
        poll_seconds: int = _POLL_SECONDS,
        recent_keep: int = 12,
        ask_fn: "Callable[[str, str], str] | None" = None,
    ) -> None:
        self._announce = announce
        self._snapshot_fn = snapshot_fn
        self._api_key = api_key
        self._model = model
        self._poll = max(60, int(poll_seconds))
        # ask(system, user) -> reply. Injectable so tests never hit the network;
        # the default binds the real anthropic call to this engine's key+model.
        self._ask = ask_fn or (
            lambda system, user: _default_ask(self._api_key, self._model, system, user)
        )
        self._recent: deque = deque(maxlen=max(1, recent_keep))
        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None

    # ---- lifecycle (homelab_monitor shape) ----

    def is_active(self) -> bool:
        return self._active.is_set()

    def activate(self) -> None:
        if self._active.is_set():
            return
        if not self._api_key:
            print("[anticipation] no API key — not starting", file=sys.stderr)
            return
        self._active.set()
        self._stop.clear()
        print(f"[anticipation] active — synthesizing every {self._poll}s",
              file=sys.stderr)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._loop, name="Anticipation", daemon=True
            )
            self._thread.start()

    def deactivate(self) -> None:
        if not self._active.is_set():
            return
        self._active.clear()
        self._stop.set()
        print("[anticipation] standing down", file=sys.stderr)

    def shutdown(self) -> None:
        self._active.clear()
        self._stop.set()

    # ---- the loop ----

    def _loop(self) -> None:
        print("[anticipation] loop started", file=sys.stderr)
        # First tick after one full interval — don't fire the instant Jarvis
        # boots (the morning briefing already covers "what's going on"); let the
        # day develop before synthesizing.
        while not self._stop.wait(self._poll):
            if not self._active.is_set():
                break
            try:
                self._run_once(datetime.now())
            except Exception as exc:  # noqa: BLE001 — never kill the loop
                print(f"[anticipation] tick failed: {exc}", file=sys.stderr)
        print("[anticipation] loop exited", file=sys.stderr)

    def _run_once(self, now: datetime) -> "str | None":
        """One synthesis cycle. Returns the announced insight (for tests) or
        None when it stayed silent. Never raises."""
        try:
            snapshot = self._snapshot_fn() or {}
        except Exception as exc:  # noqa: BLE001
            print(f"[anticipation] snapshot failed: {exc}", file=sys.stderr)
            return None
        if not any((v or "").strip() for v in snapshot.values()):
            return None  # nothing to synthesize this tick
        world = _format_world_state(snapshot, now, list(self._recent))
        try:
            reply = self._ask(ANTICIPATION_SYSTEM, world)
        except Exception as exc:  # noqa: BLE001
            print(f"[anticipation] LLM call failed: {exc}", file=sys.stderr)
            return None
        insight = _parse_decision(reply)
        if insight is None:
            return None
        if self._is_duplicate(insight):
            print("[anticipation] suppressed near-duplicate insight",
                  file=sys.stderr)
            return None
        self._recent.append(insight)
        print(f"[anticipation] insight: {insight}", file=sys.stderr)
        self._safe_announce(insight)
        return insight

    def _is_duplicate(self, insight: str) -> bool:
        """True if this insight closely matches one already surfaced — the
        prompt-level 'don't repeat' is the primary guard; this is the backstop
        against a reworded restatement of a standing situation."""
        norm = _normalize(insight)
        for prior in self._recent:
            p = _normalize(prior)
            if norm == p or norm in p or p in norm:
                return True
        return False

    def _safe_announce(self, text: str) -> None:
        try:
            self._announce(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[anticipation] announce failed: {exc}", file=sys.stderr)
