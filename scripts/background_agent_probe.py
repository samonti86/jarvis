r"""Live end-to-end check of the background-agent path (M91).

Does the real thing: provisions the agent + environment if absent, dispatches a
small task, polls until it finishes, prints the report, then archives the
session. Everything the hermetic suite fakes.

An instrument, not a gate — it makes live API calls and creates PERSISTENT
resources on the account (one agent, one environment, reused forever after).
It lives in scripts/, which the runner never collects.

    venv\Scripts\python.exe scripts\background_agent_probe.py
    venv\Scripts\python.exe scripts\background_agent_probe.py --task "..."
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import background_agent  # noqa: E402
from src.config import load  # noqa: E402

_DEFAULT_TASK = (
    "In two sentences: what is the single most common cause of ZFS pool "
    "performance degradation on consumer hardware? Cite one source."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=_DEFAULT_TASK)
    ap.add_argument("--timeout", type=int, default=600, help="seconds")
    args = ap.parse_args()

    cfg = load()
    if not cfg.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    print("=" * 72)
    print("background agent — live round trip")
    print("=" * 72)

    print("\n[1/4] provisioning agent + environment (created once, then reused)")
    agent_id, env_id = background_agent.ensure_resources(cfg.anthropic_api_key)
    if not agent_id or not env_id:
        print("  FAILED — could not provision", file=sys.stderr)
        return 1
    print(f"  agent       {agent_id}")
    print(f"  environment {env_id}")

    print(f"\n[2/4] dispatching:\n  {args.task}")
    session_id, err = background_agent.dispatch(cfg.anthropic_api_key, args.task)
    if not session_id:
        print(f"  FAILED — {err}", file=sys.stderr)
        return 1
    print(f"  session     {session_id}")

    print(f"\n[3/4] polling (timeout {args.timeout}s)")
    t0 = time.monotonic()
    state = {"status": "running"}
    while time.monotonic() - t0 < args.timeout:
        state = background_agent.poll(cfg.anthropic_api_key, session_id)
        elapsed = time.monotonic() - t0
        print(f"  {elapsed:6.0f}s  {state['status']}")
        if state["status"] != "running":
            break
        time.sleep(15)

    print(f"\n[4/4] outcome after {time.monotonic() - t0:.0f}s: {state['status']}")
    if state.get("error"):
        print(f"  error: {state['error']}")
    if state.get("result"):
        print("\n--- the agent's report (this is what Jarvis would speak) ---")
        print(state["result"])
        print("--- end ---")

    print("\ncleaning up this session (the agent + environment persist)")
    background_agent.cancel(cfg.anthropic_api_key, session_id)

    ok = state["status"] == "done" and bool(state.get("result"))
    print("\n" + ("PASS — full round trip" if ok else "INCOMPLETE — see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
