"""Tests for src/gates.CountedEvent — the nesting speech-gate primitive
(2026-07-02 QA).

The failure this guards against: an Announcer announce overlapping a playing
turn reply. Both raise pc_speaking/announce_speaking; with a plain Event the
first `finally` to run dropped the gate while the other speaker was still
mid-audio (omni-mic echo + armed-CPU stutter re-exposed). CountedEvent must
keep the flag up until the LAST holder clears.

Run: venv\\Scripts\\python.exe scripts\\gates_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gates import CountedEvent  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def main() -> int:
    # Basic single-holder behaviour matches threading.Event.
    e = CountedEvent()
    check("starts cleared", not e.is_set())
    e.set()
    check("set -> is_set", e.is_set())
    e.clear()
    check("clear -> cleared", not e.is_set())

    # The overlap scenario: two speakers, interleaved lifetimes.
    e = CountedEvent()
    e.set()            # turn reply starts
    e.set()            # announce starts mid-reply
    e.clear()          # announce finishes FIRST
    check("still set while the other holder speaks", e.is_set())
    e.clear()          # reply finishes
    check("cleared after the LAST holder", not e.is_set())

    # Three nested holders (reply + announce + speak_line is legal).
    e = CountedEvent()
    e.set(); e.set(); e.set()
    e.clear(); e.clear()
    check("3 set / 2 clear -> still set", e.is_set())
    e.clear()
    check("3rd clear -> cleared", not e.is_set())

    # Defensive unbalanced clear clamps at zero (finally-on-error idiom).
    e = CountedEvent()
    e.clear()          # stray clear with no set
    e.set()
    check("stray clear doesn't eat a later set", e.is_set())
    e.clear()
    check("...and the pair still balances", not e.is_set())

    # wait() consumes the flag like a plain Event.
    e = CountedEvent()
    e.set()
    check("wait(0) True while set", e.wait(0))
    e.clear()
    check("wait(0.01) False when cleared", not e.wait(0.01))

    # force_clear resets regardless of nesting depth.
    e = CountedEvent()
    e.set(); e.set()
    e.force_clear()
    check("force_clear drops a nested gate", not e.is_set())
    e.set()
    check("set works after force_clear", e.is_set())

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
