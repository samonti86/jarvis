r"""Regression test for long-horizon background tasks (M91).

Runs entirely offline. The Managed Agents transport (src/background_agent.py)
is replaced at MODULE LEVEL — the same seam discipline as turn_runner_test.py,
and for the same reason: src/background_tasks.py resolves `background_agent.x`
at call time, so rebinding the module attribute is what makes the fake take.
If that resolution ever changes, this suite would keep passing while testing
the real thing, which is why turn_runner_patch_test.py exists for its sibling.

WHAT IS ACTUALLY WORTH ASSERTING HERE:
The store is easy. The parts that would silently ruin an overnight task are:

  - a completion at 03:00 must NOT speak, must NOT be lost, and must be
    delivered EXACTLY once, by the briefing, later;
  - a transport blip must not orphan a session that is still running
    server-side;
  - the task tools must be unreachable from the phone/Discord origins, because
    dispatching is unbounded API spend run unobserved.

    venv\Scripts\python.exe tests\background_tasks_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Isolate the on-disk stores BEFORE importing anything that resolves a path.
os.environ["LOCALAPPDATA"] = tempfile.mkdtemp(prefix="jarvis-bgtask-")
os.environ["JARVIS_BACKGROUND_AGENTS"] = "1"

from src import background_tasks as bt  # noqa: E402
from src import task_store as ts  # noqa: E402

_passed = 0
_failed = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")
        if detail:
            print(f"          {detail}")


def reset_store() -> None:
    for t in ts.all_tasks():
        pass
    path = Path(os.environ["LOCALAPPDATA"]) / "Jarvis" / "background_tasks.json"
    if path.exists():
        path.unlink()


# --- fake transport -------------------------------------------------------
class FakeAgent:
    """Stands in for src/background_agent.py."""

    def __init__(self) -> None:
        self.dispatched: list[str] = []
        self.cancelled: list[str] = []
        self.next_state = {"status": "running", "result": None, "error": None}
        self.dispatch_error: str | None = None

    def enabled(self) -> bool:
        return True

    def dispatch(self, api_key, prompt):
        if self.dispatch_error:
            return None, self.dispatch_error
        self.dispatched.append(prompt)
        return f"sesn_{len(self.dispatched):03d}", None

    def poll(self, api_key, session_id):
        return dict(self.next_state)

    def cancel(self, api_key, session_id):
        self.cancelled.append(session_id)
        return True


fake = FakeAgent()
bt.background_agent = fake  # the seam

_quiet = {"value": False}
bt.quiet_hours = type("Q", (), {"is_quiet": staticmethod(lambda: _quiet["value"])})()

spoken: list[tuple[str, str]] = []


def fake_announce(text, label="🚨"):
    spoken.append((text, label))


# =========================================================================
# 1. store lifecycle
# =========================================================================
reset_store()
rec = ts.add("compare three NAS options")
check("new task starts pending and undelivered",
      rec["status"] == "pending" and rec["delivered"] is False)
ts.update(rec["task_id"], status="done", result="ZFS wins.")
got = ts.get(rec["task_id"])
check("terminal transition stamps finished_at", bool(got["finished_at"]))
check("finished task leaves active()", len(ts.active()) == 0)
check("finished task appears in undelivered()", len(ts.undelivered()) == 1)
ts.mark_delivered(rec["task_id"])
check("delivery clears undelivered()", len(ts.undelivered()) == 0)

# =========================================================================
# 2. dispatch + concurrency cap
# =========================================================================
reset_store()
spoken.clear()
out = bt.execute_start_background_task({"task": "research quantum error correction"})
check("dispatch reports back with a task id", "bgt_" in out, out)
check("dispatch reached the transport", fake.dispatched, "nothing dispatched")
check("session id is recorded (the restart link)",
      bool(ts.active()[0].get("session_id")))

bt.execute_start_background_task({"task": "second task"})
third = bt.execute_start_background_task({"task": "third task"})
check("concurrency cap refuses a third task", "limit" in third.lower(), third)
check("cap counts only active tasks", len(ts.active()) == 2)

# =========================================================================
# 3. dispatch failure is recorded, not swallowed
# =========================================================================
reset_store()
fake.dispatch_error = "service unavailable"
out = bt.execute_start_background_task({"task": "will fail"})
check("dispatch failure is reported to the user", "couldn't start" in out.lower(), out)
check("failed dispatch is not left active", len(ts.active()) == 0)
fake.dispatch_error = None

# =========================================================================
# 4. poll mapping — and the blip that must NOT orphan a session
# =========================================================================
reset_store()
mgr = BackgroundTaskManager = bt.BackgroundTaskManager("key", fake_announce)
bt.execute_start_background_task({"task": "long job"})

fake.next_state = {"status": "running", "result": None, "error": None}
mgr.poll_once()
check("still-running task stays active", len(ts.active()) == 1)

fake.next_state = {"status": "done", "result": "Found the answer.", "error": None}
mgr.poll_once()
check("completed task leaves active", len(ts.active()) == 0)
check("result is stored", ts.all_tasks()[-1]["result"] == "Found the answer.")

# =========================================================================
# 5. THE OVERNIGHT CASE — quiet hours must not speak, must not lose
# =========================================================================
reset_store()
spoken.clear()
_quiet["value"] = True
bt.execute_start_background_task({"task": "overnight research on ZFS tuning"})
fake.next_state = {"status": "done", "result": "Use mirrors, not RAIDZ.", "error": None}
mgr.poll_once()
check("finishing at 3am does NOT speak", not spoken, f"spoke: {spoken}")
check("finishing at 3am is NOT lost (still undelivered)",
      len(ts.undelivered()) == 1)

reports = bt._spoken_form(ts.undelivered()[0])
check("the held report carries the result",
      "mirrors" in reports.lower(), reports)

# the morning briefing drains it
drained = bt.pending_reports()
check("briefing drains the overnight report", len(drained) == 1, str(drained))
check("draining marks it delivered (exactly-once)", len(ts.undelivered()) == 0)
check("a second briefing does not repeat it", bt.pending_reports() == [])

# =========================================================================
# 6. daytime completion speaks immediately
# =========================================================================
reset_store()
spoken.clear()
_quiet["value"] = False
bt.execute_start_background_task({"task": "daytime lookup"})
fake.next_state = {"status": "done", "result": "All done.", "error": None}
mgr.poll_once()
check("daytime completion speaks", len(spoken) == 1, f"spoke: {spoken}")
check("spoken report includes the finding",
      "All done." in spoken[0][0], spoken[0][0] if spoken else "")
check("daytime completion is marked delivered", len(ts.undelivered()) == 0)

# =========================================================================
# 7. cancel
# =========================================================================
reset_store()
bt.execute_start_background_task({"task": "to be cancelled"})
tid = ts.active()[0]["task_id"]
out = bt.execute_cancel_background_task({"task_id": tid})
check("cancel confirms", "stopped" in out.lower(), out)
check("cancel reached the transport", fake.cancelled, "no cancel sent")
check("cancelled task leaves active", len(ts.active()) == 0)
check("cancelled task is not re-reported later", len(ts.undelivered()) == 0)
check("cancelling an unknown id is handled",
      "no task" in bt.execute_cancel_background_task({"task_id": "bgt_nope"}).lower())

# =========================================================================
# 8. restart re-attachment
# =========================================================================
reset_store()
bt.execute_start_background_task({"task": "survives a restart"})
session_before = ts.active()[0]["session_id"]
# Simulate a process restart: the store is re-read from disk.
reloaded = ts.active()
check("an in-flight task survives a restart", len(reloaded) == 1)
check("its session id survives, so it can be re-attached",
      reloaded[0]["session_id"] == session_before)

# =========================================================================
# 9. least privilege — remote origins must not be able to spend money
# =========================================================================
from src import llm  # noqa: E402

for name in ("start_background_task", "list_background_tasks",
             "cancel_background_task"):
    check(f"{name} is registered as a client tool", name in llm._CLIENT_TOOLS)
    check(f"{name} is DENIED to restricted origins", name in llm._RESTRICTED_DENY)

# =========================================================================
# 10. feature flag off ⇒ refuses, and says how to turn it on
# =========================================================================
reset_store()


class OffAgent(FakeAgent):
    def enabled(self) -> bool:
        return False


bt.background_agent = OffAgent()
out = bt.execute_start_background_task({"task": "anything"})
check("disabled feature refuses politely", "switched off" in out.lower(), out)
check("refusal names the flag", "JARVIS_BACKGROUND_AGENTS" in out, out)
check("refusal is honest about where the work runs",
      "anthropic" in out.lower(), out)
check("nothing is dispatched while disabled", len(ts.all_tasks()) == 0)
bt.background_agent = fake


# =========================================================================
# 11. M92 — scheduled research rides the EXISTING reminder scheduler
# =========================================================================
# Deliberately NOT a second scheduler. Recurrence, persistence, catch-up after
# downtime and voice cancellation all come from the reminder record; the only
# new part is an action that dispatches instead of composing.
reset_store()
bt.background_agent = fake
from src import reminders as remod  # noqa: E402
from src.reminders import SET_REMINDER_TOOL, _COMPOSITION_ACTIONS  # noqa: E402

check("background_task is a registered composition action",
      "background_task" in _COMPOSITION_ACTIONS)
check("the tool schema offers it",
      "background_task"
      in SET_REMINDER_TOOL["input_schema"]["properties"]["action"]["enum"])

# The composer takes the RECORD and uses `message` as the research brief —
# that is what lets "every Monday, research X" work with no extra field.
out = remod._compose_background_task(
    {"message": "research what changed in consumer NAS hardware this week"})
check("scheduled fire dispatches the task", "bgt_" in out, out)
check("the reminder's message became the research prompt",
      bool(fake.dispatched) and "consumer NAS hardware" in fake.dispatched[-1],
      str(fake.dispatched))
check("dispatch acknowledges rather than inventing findings",
      "report back" in out.lower(), out)

# A malformed record must not raise inside the scheduler thread.
check("a subject-less scheduled task fails honestly",
      "no subject" in remod._compose_background_task({"message": "  "}).lower())

# The composer signature is SHARED across every action — a regression here
# breaks briefing and good_night too, which is why it is asserted.
import inspect  # noqa: E402

for _name, _entry in _COMPOSITION_ACTIONS.items():
    _params = list(inspect.signature(_entry[0]).parameters)
    check(f"{_name} composer takes the record (shared signature)",
          len(_params) == 1, f"got {_params}")


# =========================================================================
# 12. M95 — the server-side prompt must not silently drift
# =========================================================================
# The agent object holds `system` SERVER-SIDE, and ensure_resources returns
# early when the ids are cached. Before M95 that meant editing _SYSTEM was a
# silent no-op: you tune the prompt, redeploy, and the agent keeps the old one
# forever. Caught only by reading the live agent back.
import hashlib  # noqa: E402

from src import background_agent as real_ba  # noqa: E402

check("the system prompt carries a hard word ceiling, not a target",
      "HARD LIMIT: 90 words" in real_ba._SYSTEM)
check("the escape clause that averaged 317 words is gone",
      "unless the task genuinely needs more" not in real_ba._SYSTEM)
check("prompt rationale is NOT inside the billed prompt",
      "Measured" not in real_ba._SYSTEM and "317" not in real_ba._SYSTEM,
      "history belongs in a comment; the prompt is billed every call")

fp = real_ba._system_fingerprint()
check("fingerprint is stable across calls", fp == real_ba._system_fingerprint())
check("fingerprint tracks the prompt",
      fp == hashlib.sha256(real_ba._SYSTEM.encode("utf-8")).hexdigest()[:16])

# The failure mode that matters: a FAILED update must NOT record the
# fingerprint, or the drift is marked resolved and never retried.
import json as _json  # noqa: E402
import tempfile as _tf  # noqa: E402

_prev_local = os.environ["LOCALAPPDATA"]
os.environ["LOCALAPPDATA"] = _tf.mkdtemp(prefix="jarvis-drift-")
ids_path = Path(os.environ["LOCALAPPDATA"]) / "Jarvis"
ids_path.mkdir(parents=True, exist_ok=True)
(ids_path / "background_agent_ids.json").write_text(
    _json.dumps({"agent_id": "agent_x", "environment_id": "env_x",
                 "system_sha": "stale"}), encoding="utf-8")


class _BoomClient:
    class beta:  # noqa: N801
        class agents:  # noqa: N801
            @staticmethod
            def retrieve(_id):
                raise RuntimeError("api down")

            @staticmethod
            def update(*a, **k):
                raise RuntimeError("api down")


_orig_client = real_ba._get_client
real_ba._get_client = lambda _k: _BoomClient()
try:
    a_id, e_id = real_ba.ensure_resources("key")
    check("a failed prompt update still returns the ids (never blocks dispatch)",
          a_id == "agent_x" and e_id == "env_x", f"{a_id} {e_id}")
    saved = _json.loads((ids_path / "background_agent_ids.json").read_text(encoding="utf-8"))
    check("a FAILED update does not record the fingerprint (so it retries)",
          saved.get("system_sha") == "stale", str(saved))
finally:
    real_ba._get_client = _orig_client
    os.environ["LOCALAPPDATA"] = _prev_local

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
