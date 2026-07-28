r"""Does a rubric-graded Outcome actually beat a plain research prompt? (M95 spike)

Managed Agents supports `user.define_outcome`: you supply a rubric and the
harness runs an iterate -> grade -> revise loop until the artifact satisfies it
or hits max_iterations. Verified live that SDK 0.97.0 accepts the event.

But "the API supports it" is not "it makes Jarvis better". M91's plain prompt
already asks for citations, brevity and honesty about uncertainty, and the live
round trip returned a cited answer in 17s. An Outcome re-runs the work on every
failed grade, so it costs real time and tokens. This measures whether that buys
anything.

GRADED MECHANICALLY, NOT BY A MODEL. Asking an LLM whether the LLM's answer is
good is circular and unfalsifiable. Every criterion here is checkable in code:
word count, presence of a digit, presence of a URL. Blunt, but honest.

An instrument, not a gate: live API calls, several minutes, real money.

    venv\Scripts\python.exe scripts\outcome_probe.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402

from src import background_agent as ba  # noqa: E402
from src.config import load  # noqa: E402

# (question, rubric, mechanical checks). Each check is (name, predicate).
CASES = [
    (
        "What single configuration setting most affects ZFS write performance "
        "on consumer SSDs, and at roughly what threshold does it bite?",
        "The answer must: (1) name the specific setting or property; "
        "(2) give a concrete number or threshold; (3) cite a source URL; "
        "(4) be under 90 words.",
        [
            ("gives a number", lambda t: bool(re.search(r"\d", t))),
            ("cites a URL", lambda t: "http" in t.lower()),
            ("under 90 words", lambda t: len(t.split()) < 90),
        ],
    ),
    (
        "What is the current recommended maximum fill percentage for a ZFS pool "
        "before performance degrades, and who recommends it?",
        "The answer must: (1) give a percentage; (2) name the source or vendor; "
        "(3) cite a URL; (4) be under 90 words.",
        [
            ("gives a percentage", lambda t: "%" in t or "percent" in t.lower()),
            ("cites a URL", lambda t: "http" in t.lower()),
            ("under 90 words", lambda t: len(t.split()) < 90),
        ],
    ),
    (
        "Which Python audio library is recommended for low-latency full-duplex "
        "capture on Windows, and what is the practical latency floor?",
        "The answer must: (1) name a specific library; (2) give a latency figure "
        "in milliseconds; (3) cite a URL; (4) be under 90 words.",
        [
            ("gives a latency figure", lambda t: bool(re.search(r"\d+\s*ms", t, re.I))),
            ("cites a URL", lambda t: "http" in t.lower()),
            ("under 90 words", lambda t: len(t.split()) < 90),
        ],
    ),
]


def run(client, agent_id, env_id, question, rubric, use_outcome, timeout=420):
    """One session, either plain or outcome-graded. Returns (text, secs, err)."""
    s = client.beta.sessions.create(agent=agent_id, environment_id=env_id,
                                    title=question[:60])
    t0 = time.monotonic()
    try:
        if use_outcome:
            event = {
                "type": "user.define_outcome",
                "description": question,
                "rubric": {"type": "text", "content": rubric},
                "max_iterations": 3,
            }
        else:
            event = {"type": "user.message",
                     "content": [{"type": "text", "text": question}]}
        client.beta.sessions.events.send(session_id=s.id, events=[event])

        while time.monotonic() - t0 < timeout:
            state = ba.poll(load().anthropic_api_key, s.id)
            if state["status"] != "running":
                return state.get("result") or "", time.monotonic() - t0, state.get("error")
            time.sleep(15)
        return "", time.monotonic() - t0, "timeout"
    finally:
        try:
            client.beta.sessions.archive(session_id=s.id)
        except Exception:
            pass


def main() -> int:
    cfg = load()
    agent_id, env_id = ba.ensure_resources(cfg.anthropic_api_key)
    if not agent_id:
        print("could not provision", file=sys.stderr)
        return 1
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    totals = {"plain": [0, 0, 0.0], "outcome": [0, 0, 0.0]}  # met, possible, secs

    for question, rubric, checks in CASES:
        print(f"\n=== {question[:78]}")
        for mode in ("plain", "outcome"):
            text, secs, err = run(client, agent_id, env_id, question, rubric,
                                  use_outcome=(mode == "outcome"))
            if err or not text:
                print(f"  {mode:8s} FAILED after {secs:.0f}s: {err}")
                totals[mode][1] += len(checks)
                totals[mode][2] += secs
                continue
            met = [name for name, fn in checks if fn(text)]
            totals[mode][0] += len(met)
            totals[mode][1] += len(checks)
            totals[mode][2] += secs
            missed = [n for n, _ in checks if n not in met]
            print(f"  {mode:8s} {secs:5.0f}s  {len(met)}/{len(checks)} criteria  "
                  f"{len(text.split()):3d} words"
                  + (f"  MISSED: {', '.join(missed)}" if missed else ""))

    print("\n" + "=" * 72)
    print(f"{'mode':10s} {'criteria met':>14s} {'total time':>12s}")
    print("-" * 72)
    for mode, (met, possible, secs) in totals.items():
        pct = 100 * met / possible if possible else 0
        print(f"{mode:10s} {met:6d}/{possible:<6d} ({pct:3.0f}%) {secs:10.0f}s")
    print("=" * 72)
    print("Ship the Outcome path only if criteria-met rises enough to justify\n"
          "the extra wall-clock. A tie means the M91 prompt was already doing it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
