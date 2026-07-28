r"""Long-horizon background tasks — orchestration, tools and delivery (M91).

Three modules, one feature:
  src/task_store.py       durable record (survives restarts)
  src/background_agent.py Managed Agents transport (dispatch/poll/cancel)
  THIS FILE               policy: when to dispatch, when to speak, what to say

DELIVERY IS THE INTERESTING PART.
A task that finishes at 03:00 must not wake the house, and a task that
finishes at 14:00 should just be said. Rather than teach the quiet-hours
module a new label, the decision is made here:

    finished during quiet hours -> leave it undelivered; the morning briefing
                                   drains it (get_briefing reads pending_reports)
    finished otherwise          -> announce immediately, mark delivered

That keeps M79's deferral behaviour untouched (lower risk), avoids the
double-delivery you would get by ALSO routing through the quiet-hours
catch-up store, and leaves the store's `delivered` flag as the single source
of truth for "has the user been told". A restart between finishing and
speaking loses nothing: the record is on disk, still undelivered.

CONCURRENCY IS CAPPED because each task is an unbounded-ish spend. Two at a
time, refused politely beyond that.
"""

from __future__ import annotations

import os
import sys
import threading

from src import background_agent, quiet_hours, task_store


def _api_key() -> str:
    """Read at call time, matching how the other credential-backed tools work
    (wolfram_query, tmdb, ...). config.load() has already run load_dotenv(), so
    the process environment is populated by the time any tool fires."""
    return os.getenv("ANTHROPIC_API_KEY", "") or ""

# Each running task is real money. Two concurrent is enough for "look into A
# while you finish B" and bounds the damage from a misunderstood request.
_MAX_CONCURRENT = 2

# How often to ask the API whether anything finished. Long-horizon work runs
# for many minutes; polling faster buys nothing and costs requests.
_POLL_SECONDS = 60.0

# Pierces quiet hours by design — but this label is only ever used OUTSIDE the
# quiet window, because the manager defers instead of announcing during it.
_LABEL = "🔬"

# Hard age cap. Generous enough for genuine overnight work, short enough that a
# wedged session is reclaimed by morning rather than occupying a slot for days.
_MAX_TASK_HOURS = 8.0


def _too_old(task: dict) -> bool:
    """True if this task has been active past the cap. Unparseable timestamps
    return False — never fail a task because its clock string looked odd."""
    from datetime import datetime  # noqa: PLC0415

    stamp = task.get("created_at") or ""
    try:
        age_h = (datetime.now() - datetime.fromisoformat(stamp)).total_seconds() / 3600
    except ValueError:
        return False
    return age_h > _MAX_TASK_HOURS


# --------------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------------
class BackgroundTaskManager:
    """Owns the poll loop. Constructed always; only started when enabled."""

    def __init__(self, api_key: str, announce=None, ui=None) -> None:
        self._api_key = api_key
        self._announce = announce
        self._ui = ui
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Begin polling. Idempotent; a no-op when the feature is off."""
        if not background_agent.enabled():
            return
        if self._thread is not None:
            return
        active = task_store.active()
        if active:
            # RE-ATTACHMENT. These sessions kept running server-side while
            # Jarvis was down — the store is the only thing that remembers
            # them, and this is why it exists.
            print(f"[bgtask] resuming {len(active)} task(s) after restart",
                  file=sys.stderr)
        self._thread = threading.Thread(
            target=self._loop, name="background-tasks", daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        """Stop polling. Running sessions are deliberately LEFT RUNNING — they
        live server-side, and a restart re-attaches to them. Cancelling work on
        shutdown would make `update_jarvis` silently destroy an overnight task."""
        self._stop.set()

    def status_summary(self) -> str:
        """Bound method so main.py registers a handle, not a module global."""
        return status_line()

    def _loop(self) -> None:
        # Poll once promptly so a restart reports anything that finished while
        # the process was down, rather than waiting a full interval.
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 — a background thread must not die
                print(f"[bgtask] poll cycle failed: {exc}", file=sys.stderr)
            self._stop.wait(_POLL_SECONDS)

    # -- the cycle ---------------------------------------------------------
    def poll_once(self) -> None:
        """Advance every active task, then deliver anything finished."""
        for task in task_store.active():
            session_id = task.get("session_id")
            task_id = task.get("task_id")
            if not session_id or not task_id:
                continue
            if _too_old(task):
                # Backstop. poll() deliberately reports a not-yet-started or
                # transiently-unreachable session as "running", which is right
                # per-poll but means nothing else would ever end a session that
                # is genuinely wedged — and it would hold a concurrency slot
                # for good. Age is the only signal that survives both cases.
                task_store.update(
                    task_id, status="failed",
                    error=f"gave up after {_MAX_TASK_HOURS} hours with no result",
                )
                continue
            state = background_agent.poll(self._api_key, session_id)
            if state["status"] == "running":
                if task.get("status") != "running":
                    task_store.update(task_id, status="running")
                continue
            task_store.update(
                task_id,
                status=state["status"],
                result=state.get("result"),
                error=state.get("error"),
            )
        self.deliver_ready()
        self._push_ui()

    def _push_ui(self) -> None:
        """Mirror task state to the ambient overlay. Fail-soft: the HUD is a
        nicety, the poll loop is not."""
        if self._ui is None:
            return
        try:
            self._ui.set_research_indicator(
                len(task_store.active()), len(task_store.undelivered()))
        except Exception as exc:  # noqa: BLE001
            print(f"[bgtask] ui update failed: {exc}", file=sys.stderr)

    def deliver_ready(self) -> None:
        """Speak finished tasks — unless it's quiet hours, in which case leave
        them for the briefing."""
        pending = task_store.undelivered()
        if not pending:
            return
        if quiet_hours.is_quiet():
            return  # the morning briefing drains these
        for task in pending:
            text = _spoken_form(task)
            if self._announce is not None:
                try:
                    self._announce(text, label=_LABEL)
                except Exception as exc:  # noqa: BLE001
                    print(f"[bgtask] announce failed: {exc}", file=sys.stderr)
                    continue  # leave undelivered; try again next cycle
            task_store.mark_delivered(task["task_id"])


def _spoken_form(task: dict) -> str:
    """One task rendered for speech."""
    prompt = (task.get("prompt") or "").strip()
    short = prompt if len(prompt) <= 60 else prompt[:57].rstrip() + "..."
    if task.get("status") == "done":
        return f"Sir, I've finished looking into {short}. {task.get('result') or ''}".strip()
    if task.get("status") == "cancelled":
        return f"I've stopped work on {short}, sir."
    return (f"Sir, I wasn't able to finish looking into {short}. "
            f"{task.get('error') or 'The task failed.'}")


def pending_reports() -> list[str]:
    """Finished-but-unspoken tasks, rendered for the morning briefing, and
    marked delivered as they are handed over. Draining here is what makes an
    overnight completion reach the user without waking them."""
    out: list[str] = []
    for task in task_store.undelivered():
        out.append(_spoken_form(task))
        task_store.mark_delivered(task["task_id"])
    return out


def status_line() -> str:
    """One line for status_report."""
    if not background_agent.enabled():
        return "off (JARVIS_BACKGROUND_AGENTS unset)"
    active = task_store.active()
    waiting = task_store.undelivered()
    if not active and not waiting:
        return "idle, nothing queued"
    bits = []
    if active:
        bits.append(f"{len(active)} running")
    if waiting:
        bits.append(f"{len(waiting)} awaiting delivery")
    return ", ".join(bits)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
START_BACKGROUND_TASK_TOOL = {
    "name": "start_background_task",
    "description": (
        "Dispatch a long-running research or analysis task that works in the "
        "background for many minutes to an hour, and report back when it is "
        "done. Use for open-ended asks the user does not expect answered "
        "immediately — 'research X overnight', 'dig into Y and tell me later', "
        "'compare these options properly and brief me in the morning'. "
        "Do NOT use it for anything answerable now: a normal web search, a "
        "weather or sports lookup, or any question the user is waiting on. "
        "The task runs in an isolated sandbox with web access; it CANNOT touch "
        "this PC, its files, or any of your other tools."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "The full task, written for someone with no other context: "
                    "what to find out, and what a good answer looks like."
                ),
            },
        },
        "required": ["task"],
    },
}

LIST_BACKGROUND_TASKS_TOOL = {
    "name": "list_background_tasks",
    "description": (
        "List background research tasks and their state — what is still "
        "running, what finished. Use for 'what are you working on?' or before "
        "cancelling, when you need a task id."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

CANCEL_BACKGROUND_TASK_TOOL = {
    "name": "cancel_background_task",
    "description": (
        "Stop a running background task. Use when the user says to drop it, "
        "stop looking into something, or never mind."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task id from list_background_tasks.",
            },
        },
        "required": ["task_id"],
    },
}


def execute_start_background_task(params: dict) -> str:
    """Never raises — an unavailable feature returns an honest sentence."""
    if not background_agent.enabled():
        return ("Background tasks are switched off, sir. They can be enabled "
                "with JARVIS_BACKGROUND_AGENTS=1 — note that unlike the rest "
                "of me, that work runs on Anthropic's infrastructure.")
    prompt = (params.get("task") or "").strip()
    if not prompt:
        return "I need to know what to look into, sir."

    running = task_store.active()
    if len(running) >= _MAX_CONCURRENT:
        return (f"I already have {len(running)} tasks running, sir, which is my "
                f"limit. Ask me to cancel one first.")

    record = task_store.add(prompt)
    session_id, error = background_agent.dispatch(_api_key(), prompt)
    if not session_id:
        task_store.update(record["task_id"], status="failed", error=error)
        return f"I couldn't start that, sir — {error or 'the service is unavailable'}."
    task_store.update(record["task_id"], session_id=session_id, status="running")
    return (f"Working on it, sir. I'll look into that in the background and "
            f"report back — reference {record['task_id']}.")


def execute_list_background_tasks(params: dict) -> str:  # noqa: ARG001 — no params
    tasks = task_store.all_tasks()
    if not tasks:
        return "No background tasks, sir."
    live = [t for t in tasks if t.get("status") in ("pending", "running")]
    recent = [t for t in tasks if t.get("status") not in ("pending", "running")][-3:]
    lines: list[str] = []
    for t in live:
        lines.append(f"{t['task_id']}: working on \"{t.get('prompt', '')[:60]}\"")
    for t in recent:
        lines.append(f"{t['task_id']}: {t.get('status')} — \"{t.get('prompt', '')[:60]}\"")
    return "\n".join(lines) if lines else "No background tasks, sir."


def execute_cancel_background_task(params: dict) -> str:
    task_id = (params.get("task_id") or "").strip()
    task = task_store.get(task_id)
    if not task:
        return f"I have no task with the id {task_id}, sir."
    if task.get("status") not in ("pending", "running"):
        return f"That task already finished, sir — it's {task.get('status')}."
    if task.get("session_id"):
        background_agent.cancel(_api_key(), task["session_id"])
    # Marked cancelled regardless: the user's intent is honoured locally even
    # if the remote archive call failed.
    task_store.update(task_id, status="cancelled", delivered=True)
    return "Stopped, sir."
