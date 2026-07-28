r"""Long-horizon background agents via Anthropic Managed Agents (M91).

WHAT THIS IS FOR:
"Research X overnight and brief me at breakfast." Jarvis's normal turn is
capped at 8 tool iterations and has to start speaking inside a second — an
hour of work cannot happen there. This dispatches that work to a Managed
Agents session: Anthropic runs the agent loop and hosts the sandbox, Jarvis
keeps the durable record (src/task_store.py) and owns delivery.

WHY NOT A LOCAL THREAD:
Restart survival. Jarvis runs under jarvis_watchdog.pyw (respawn on crash) and
update_jarvis restarts the process deliberately, so a thread holding an
overnight task dies as a matter of routine, not misfortune. A managed session
lives server-side; the store holds its id and Jarvis re-attaches after a
restart. Local execution would also put a long CPU-heavy job next to the
real-time audio threads, which is exactly the contention the M67/M68
cooperative speech gate exists to prevent.

LEAST PRIVILEGE — READ THIS BEFORE ADDING A TOOL:
A background agent runs UNOBSERVED, for a long time, with nobody watching to
interrupt it. It therefore gets strictly LESS power than a voice turn, not
more. Its entire surface is the Anthropic-hosted toolset — bash, file ops,
web search/fetch — confined to a per-session container with no route back to
this machine. It gets ZERO Jarvis tools: no pc_shell, no system_control, no
host filesystem, no camera, no self-update. The only thing that crosses back
is the text of its report.

That is a stronger boundary than the per-origin deny list used for the phone
and Discord clients, which filters a shared tool set. Here the surface is
separate by construction, so there is nothing to leak through.

PRIVACY — this is opt-in for a reason:
Jarvis's standing promise is that only transcribed text leaves the machine.
A research agent fetches pages and writes working files inside an
Anthropic-hosted container, which is genuinely more than that. Off unless
JARVIS_BACKGROUND_AGENTS=1. See the privacy section of CLAUDE.md.

Everything here is fail-soft: any API error logs and degrades. A background
feature must never take down the listening loop.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

from src.atomic_io import atomic_write_text

_AGENT_NAME = "Jarvis Researcher"
_ENV_NAME = "jarvis-research"

# Anthropic's prebuilt toolset: bash, read, write, edit, glob, grep,
# web_fetch, web_search — all executing inside the session container.
_TOOLSET = {"type": "agent_toolset_20260401"}

_SYSTEM = """You are Jarvis's research agent, working a long-running task on \
his behalf while the user is away.

You have a sandbox: use bash, files, and web search freely to do the work \
properly. Take notes as you go, verify claims against more than one source, \
and say plainly when something is uncertain or you could not confirm it.

Your final message is the whole deliverable — it is read back to the user \
aloud or placed in a morning briefing, and they will not see your working. \
Write it for someone who has been asleep:

- Open with the answer or the single most useful finding, in one sentence.
- Then the supporting detail, in short paragraphs. No headers, no bullet \
markup, no tables — this may be spoken.
- Keep it under 200 words unless the task genuinely needs more.
- If you failed, say so in the first sentence and explain what blocked you. \
Do not pad a failure into something that sounds like a result."""

_LOCK = threading.RLock()


def enabled() -> bool:
    """Opt-in. Blank/0/false/no/off ⇒ disabled (the safe default)."""
    raw = (os.getenv("JARVIS_BACKGROUND_AGENTS", "") or "").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


# --------------------------------------------------------------------------
# Persistent resource ids
# --------------------------------------------------------------------------
def _ids_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base / "background_agent_ids.json"


def _load_ids() -> dict:
    path = _ids_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        print(f"[bgagent] could not read {path.name}: {exc}", file=sys.stderr)
        return {}


def _save_ids(ids: dict) -> None:
    atomic_write_text(_ids_path(), json.dumps(ids, indent=2))


def _get_client(api_key: str):
    """Lazy import so this module stays import-light and a missing/old SDK
    degrades instead of breaking startup."""
    import anthropic  # noqa: PLC0415

    return anthropic.Anthropic(api_key=api_key)


def ensure_resources(api_key: str) -> tuple[str | None, str | None]:
    """Find-or-create the agent + environment, returning (agent_id, env_id).

    These are PERSISTENT, versioned resources — created once and reused, never
    per dispatch. Creating an agent per task would accumulate orphans, pay
    create latency on every request, and defeat the versioning model. The ids
    are cached on disk; this is the only place that calls .create().
    """
    with _LOCK:
        ids = _load_ids()
        if ids.get("agent_id") and ids.get("environment_id"):
            return ids["agent_id"], ids["environment_id"]

        try:
            client = _get_client(api_key)
            if not ids.get("environment_id"):
                env = client.beta.environments.create(
                    name=_ENV_NAME,
                    config={"type": "cloud", "networking": {"type": "unrestricted"}},
                )
                ids["environment_id"] = env.id
            if not ids.get("agent_id"):
                agent = client.beta.agents.create(
                    name=_AGENT_NAME,
                    model="claude-sonnet-5",
                    system=_SYSTEM,
                    tools=[_TOOLSET],
                )
                ids["agent_id"] = agent.id
            _save_ids(ids)
            print(f"[bgagent] resources ready agent={ids['agent_id']} "
                  f"env={ids['environment_id']}", file=sys.stderr)
            return ids["agent_id"], ids["environment_id"]
        except Exception as exc:  # noqa: BLE001 — fail soft, never crash startup
            print(f"[bgagent] could not provision resources: {exc}", file=sys.stderr)
            return None, None


# --------------------------------------------------------------------------
# Session lifecycle
# --------------------------------------------------------------------------
def dispatch(api_key: str, prompt: str) -> tuple[str | None, str | None]:
    """Start a session working on `prompt`. Returns (session_id, error).

    `initial_events` starts the agent loop in the same call as creation, so
    there is no window where a session exists but is doing nothing.
    """
    agent_id, env_id = ensure_resources(api_key)
    if not agent_id or not env_id:
        return None, "background agent resources are not available"
    try:
        client = _get_client(api_key)
        session = client.beta.sessions.create(
            agent=agent_id,
            environment_id=env_id,
            title=prompt[:80],
        )
        # Kick off in a SEPARATE call rather than via `initial_events`, which
        # this SDK (0.97.0) rejects: "Sessions.create() got an unexpected
        # keyword argument 'initial_events'". Found by the live probe — every
        # hermetic test passed against a fake that happily accepted it.
        #
        # Consequence worth knowing: the session exists IDLE for a moment
        # before this send lands, which is why poll() treats "idle with no
        # result yet" as still-running rather than finished.
        client.beta.sessions.events.send(
            session_id=session.id,
            events=[{
                "type": "user.message",
                "content": [{"type": "text", "text": prompt}],
            }],
        )
        return session.id, None
    except Exception as exc:  # noqa: BLE001
        print(f"[bgagent] dispatch failed: {exc}", file=sys.stderr)
        return None, str(exc)[:200]


def poll(api_key: str, session_id: str) -> dict:
    """Check one session. Returns {status, result, error}.

    status is normalised to the task-store vocabulary:
      running  — still working, or idle awaiting an action we don't provide
      done     — finished; `result` holds the agent's final message
      failed   — terminated without a usable result, or the API errored

    A transport error is reported as `running`, NOT `failed`: a blip must not
    orphan a session that is still working perfectly well server-side.
    """
    try:
        client = _get_client(api_key)
        session = client.beta.sessions.retrieve(session_id=session_id)
        status = getattr(session, "status", "") or ""

        if status in ("running", "rescheduling"):
            return {"status": "running", "result": None, "error": None}

        if status == "idle":
            stop = getattr(session, "stop_reason", None)
            stop_type = getattr(stop, "type", None) if stop else None
            # `requires_action` means it wants a client-side tool result. This
            # agent is configured with none, so that state is a dead end rather
            # than something to wait on.
            if stop_type == "requires_action":
                return {"status": "failed", "result": None,
                        "error": "session is waiting on a tool this agent does not provide"}
            text = _final_text(client, session_id)
            if not text:
                # Idle with nothing said yet. A session is created idle and only
                # starts once the kickoff message lands (see dispatch), so this
                # is the normal state for the first second or two — reporting
                # "done" here would deliver an EMPTY report and close the task.
                # Treat it as still working; the manager's age cap is what stops
                # a genuinely stuck session running forever.
                return {"status": "running", "result": None, "error": None}
            return {"status": "done", "result": text, "error": None}

        if status == "terminated":
            text = _final_text(client, session_id)
            if text:
                return {"status": "done", "result": text, "error": None}
            return {"status": "failed", "result": None,
                    "error": "session terminated without producing a result"}

        return {"status": "running", "result": None, "error": None}
    except Exception as exc:  # noqa: BLE001
        print(f"[bgagent] poll failed for {session_id}: {exc}", file=sys.stderr)
        return {"status": "running", "result": None, "error": None}


def _final_text(client, session_id: str) -> str:
    """The agent's last message — the deliverable.

    Events come back oldest-first, so the last agent.message is the report.
    """
    try:
        events = client.beta.sessions.events.list(session_id=session_id)
        data = getattr(events, "data", None) or list(events)
        chunks: list[str] = []
        for ev in data:
            if getattr(ev, "type", None) != "agent.message":
                continue
            parts = [
                getattr(b, "text", "") for b in (getattr(ev, "content", None) or [])
                if getattr(b, "type", None) == "text"
            ]
            joined = " ".join(p for p in parts if p).strip()
            if joined:
                chunks.append(joined)
        return chunks[-1] if chunks else ""
    except Exception as exc:  # noqa: BLE001
        print(f"[bgagent] could not read result for {session_id}: {exc}",
              file=sys.stderr)
        return ""


def cancel(api_key: str, session_id: str) -> bool:
    """Interrupt and archive. Best effort — a failure here is logged, and the
    task is still marked cancelled locally so the user's intent is honoured
    even if the remote cleanup didn't land."""
    try:
        client = _get_client(api_key)
        try:
            client.beta.sessions.events.send(
                session_id=session_id, events=[{"type": "user.interrupt"}],
            )
        except Exception:  # noqa: BLE001 — interrupt is best-effort
            pass
        client.beta.sessions.archive(session_id=session_id)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[bgagent] cancel failed for {session_id}: {exc}", file=sys.stderr)
        return False
