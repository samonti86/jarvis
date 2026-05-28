"""Self-update (M64) — "Jarvis, update yourself."

The tiniest sensible update path: `git pull --ff-only` in the project root,
then the M32 tray-restart. Hot-loaded code paths don't change in a running
Python interpreter, so a pull WITHOUT a restart would leave Jarvis running
the old code while the disk has new code — the worst possible state. The
restart is load-bearing, not a UX flourish.

Design decisions:
  - **Tool, not voice intent.** Claude routes naturally on "update yourself"
    / "pull the latest" / "self-update"; a regex matcher would be brittle
    against phrasing variants. The deterministic dismissal/intent matchers
    (M51 `_is_dismissal`, the face-enrollment intent) gate loop control;
    self-update is a capability the model should reason about.
  - **Confirmation-gated via `confirm: bool`.** Same pattern as
    `system_control`'s mutating verbs — without `confirm=true` the tool
    returns a description of what will happen so Claude can ask, then a
    second call with `confirm=true` does the work. Mutating action,
    [[feedback-jarvis-least-privilege]] + [[feedback-diag-vs-action-split]].
  - **Refuses on a dirty working tree.** `git status --porcelain` non-empty
    ⇒ abort. Don't lose user changes; the user can `git stash` or commit
    first. We could `git stash` ourselves but that's silent state mutation
    a user might not expect.
  - **Fast-forward only.** `git pull --ff-only` fails loudly rather than
    silently merging — a divergent local commit needs human attention, not
    an autobotic merge inside Jarvis.
  - **Phone DENIES this.** In `_RESTRICTED_DENY` in llm.py — the phone PWA
    must never be able to pull code into the user's machine (the M48.2a
    least-privilege boundary).
  - **Restart via callback.** main.py registers a callback at startup
    (`register_restart_callback(ui._handle_restart)`); this module fires
    it on a delayed worker thread so Claude has time to speak the
    "I'll restart in a moment" confirmation before the UI tears down.
    Same indirection pattern as good_night's `register_security_getter`.

Defensive contract: every public path returns a voice-friendly string and
never raises. A missing git binary, a network failure, a merge conflict —
all become a sentence Claude can read aloud.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# How long to wait after returning the success message before triggering the
# restart. Gives Claude time to speak the confirmation ("I'll restart in a
# moment, sir") before the UI tears down — the TTS player is single-track,
# so the spoken line completes BEFORE this delay elapses. 5 s is comfortably
# longer than a one-sentence TTS render (~2 s on this CPU) and short enough
# that the user isn't left wondering what happened.
_RESTART_DELAY_SECONDS = 5.0


# --- Anthropic tool definition --------------------------------------------

UPDATE_JARVIS_TOOL = {
    "name": "update_jarvis",
    "description": (
        "Pull the latest code from the Jarvis git repository and restart "
        "Jarvis to load it. Use this when the user says 'update yourself', "
        "'pull the latest', 'self-update', 'check for updates', 'are there "
        "any updates'. CONFIRMATION-GATED: call without `confirm` first to "
        "get a description of what will happen, relay that to the user, "
        "and only call again with `confirm=true` after they explicitly say "
        "yes / proceed / go ahead. If the working tree has uncommitted "
        "changes, the update refuses (the user must commit or stash "
        "first). If already up to date, the tool says so and does NOT "
        "restart. On a successful pull, Jarvis restarts ~5 seconds later "
        "— enough time for you to speak the confirmation. Read the tool's "
        "result back as a single concise sentence; do NOT enumerate "
        "individual changed files."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "confirm": {
                "type": "boolean",
                "description": (
                    "Set to true ONLY after the user has explicitly "
                    "confirmed they want Jarvis updated. Without this "
                    "(or with false), the tool returns a 'this will pull "
                    "and restart' description for you to relay so the "
                    "user can confirm. Pulling + restarting are not "
                    "reversible from inside this session, so a "
                    "confirmation gate is load-bearing."
                ),
            },
        },
    },
}


# --- Restart-callback wiring (the homelab_monitor decoupling pattern) ------

_restart_callback: Callable[[], None] | None = None


def register_restart_callback(callback: Callable[[], None]) -> None:
    """Wire main.py's restart trigger (`ui._handle_restart` or equivalent)
    into this module. Called once at startup. Unset ⇒ the tool warns the
    user that it can't restart and returns the pull summary text only."""
    global _restart_callback
    _restart_callback = callback


# --- Git helpers ----------------------------------------------------------

def _run_git(args: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
    """Run `git <args>` in the project root. Returns (rc, stdout, stderr).
    Never raises — every subprocess failure mode is mapped to a clear rc/stderr
    so the caller's voice-friendly error path is the ONLY exit."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return (-1, "", "git executable not found in PATH")
    except subprocess.TimeoutExpired:
        return (-1, "", f"git {' '.join(args)} timed out after {timeout:.0f}s")
    except OSError as exc:
        return (-1, "", f"git subprocess failed: {exc}")
    return (result.returncode,
            (result.stdout or "").strip(),
            (result.stderr or "").strip())


def _pending_commits() -> tuple[list[str] | None, str]:
    """Subject lines of commits pending from origin (M64.1). Returns
    (commits, err): on success (list, ""); on a git failure (None, err).
    An empty list means the upstream is already at HEAD — nothing to pull.

    Uses `HEAD..@{u}` (upstream of current branch) — robust whether the
    user is on main, a feature branch, or detached. Cap at 11 so we can
    show "10 + N more" in the describe text without pulling a thousand
    commit subjects into the prompt cache."""
    rc, stdout, stderr = _run_git(
        ["log", "HEAD..@{u}", "--oneline", "-11"],
        timeout=10.0,
    )
    if rc != 0:
        return (None, (stderr or "git log failed")[:200])
    if not stdout:
        return ([], "")
    return (stdout.splitlines(), "")


def _describe_pending(pending: list[str]) -> str:
    """Render the pending-commits list as a spoken-friendly preview. Claude
    paraphrases anyway, so this is just structured data, not the final
    speech. Caps the visible list at 5 and tags the overflow count — a long
    list read verbatim is voice noise."""
    n = len(pending)
    visible = pending[:5]
    overflow = max(0, n - len(visible))
    header = (f"{n} new commit on origin:" if n == 1
              else f"{n} new commits on origin:")
    body = "\n".join(f"  - {line}" for line in visible)
    if overflow:
        body += f"\n  - (+ {overflow} more)"
    return f"{header}\n{body}"


def _working_tree_status() -> tuple[str, str]:
    """Tri-state status of the working tree. The "unknown" case (git failed)
    must be distinguished from "clean" — pulling when we can't tell whether
    there are changes could clobber unsaved work, so the caller treats
    unknown as a hard abort.

    Returns (status, detail) where status ∈ {"clean", "dirty", "unknown"}.
    Detail is empty for "clean"; for "dirty" it's the porcelain summary
    (capped 200 chars); for "unknown" it's the git error text."""
    rc, stdout, stderr = _run_git(["status", "--porcelain"], timeout=10.0)
    if rc != 0:
        return ("unknown", (stderr or "git status failed")[:300])
    if not stdout:
        return ("clean", "")
    return ("dirty", stdout[:200])


# --- Tool executor --------------------------------------------------------

def execute_update_jarvis(params: dict) -> str:
    """Update tool. Never raises — every failure path returns a voice-
    friendly string for Claude to relay. See module docstring for design."""
    confirm = bool(params.get("confirm", False))
    if not confirm:
        # M64.1 — preview WHAT would be pulled so the confirmation is
        # informed, not procedural. Three-stage best-effort:
        #   1. `git fetch` so the local view of origin is current.
        #   2. `git log HEAD..@{u}` for the pending commit subjects.
        #   3. Branch on the result: nothing pending ⇒ "already up to
        #      date" (skip the confirmation dance); pending commits ⇒
        #      describe them; any git error along the way ⇒ fall back to
        #      the generic describe (don't block the path on an
        #      introspection blip — the confirm=true path still re-checks).
        fetch_rc, _, fetch_err = _run_git(["fetch", "--quiet"], timeout=30.0)
        if fetch_rc != 0:
            return (
                "Updating myself will run `git pull` in the project "
                "repository and restart me to load the new code, sir. "
                f"(I couldn't reach origin to preview the changes: "
                f"{fetch_err[:100]}.) Shall I proceed anyway?"
            )
        pending, err = _pending_commits()
        if err:
            return (
                "Updating myself will run `git pull` in the project "
                "repository and restart me to load the new code, sir. "
                f"(I couldn't preview the commits: {err[:100]}.) "
                f"Shall I proceed?"
            )
        if not pending:
            # Saves the user from confirming an empty pull.
            return "I'm already up to date, sir — no update needed."
        return (
            f"{_describe_pending(pending)}\n\n"
            f"Pulling these and restarting me — shall I proceed?"
        )

    status, detail = _working_tree_status()
    if status == "unknown":
        return (
            f"I couldn't read the git status, sir, so I can't safely "
            f"update: {detail}"
        )
    if status == "dirty":
        return (
            "There are uncommitted changes in the working tree, sir — "
            "commit or stash them first, then ask me again. Status:\n"
            f"{detail}"
        )

    rc, stdout, stderr = _run_git(["pull", "--ff-only"], timeout=120.0)
    if rc != 0:
        # Common failure modes the user might hear:
        #   - "Could not resolve host github.com" → no network
        #   - "Not possible to fast-forward" → divergent local commits
        #   - "Authentication failed" → credential issue
        msg = (stderr or stdout or "unknown error")[:300]
        return f"The update failed, sir: {msg}"

    # Git's "Already up to date." (with or without trailing punctuation)
    # is THE signal that nothing was pulled. Match both for safety across
    # git versions.
    first_line = stdout.splitlines()[0] if stdout else ""
    if "already up to date" in stdout.lower():
        return "I'm already up to date, sir — no update needed."

    if _restart_callback is None:
        # Update applied but we can't restart ourselves — surface honestly
        # so the user knows to restart manually. Rare in production (main.py
        # always wires the callback) but the test path hits this.
        return (
            f"Update pulled successfully, sir, but I'm not wired to "
            f"restart myself in this session. {first_line[:150]}"
        )

    # Schedule the restart on a daemon worker so this function can return
    # immediately — Claude needs the success string to speak before the
    # UI tears down.
    def _delayed_restart() -> None:
        time.sleep(_RESTART_DELAY_SECONDS)
        try:
            _restart_callback()
        except Exception as exc:  # noqa: BLE001 — never propagate
            print(f"[self_update] restart callback raised: {exc}",
                  file=sys.stderr)

    threading.Thread(
        target=_delayed_restart,
        name="SelfUpdate-Restart",
        daemon=True,
    ).start()

    return (
        f"Update successful, sir — {first_line[:100]}. "
        f"I'll restart in a moment."
    )
