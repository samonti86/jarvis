r"""Regression test for cross-session self-review (M93).

Runs against SYNTHETIC logs written to a temp LOCALAPPDATA — no dependence on
whatever happens to be in the developer's real jarvis.log, which would make
this suite pass or fail based on how the machine had been behaving that week.

WHAT IS WORTH ASSERTING:
The value of this feature is entirely in the grouping. Two failures make the
report worse than useless, and both were seen for real while building it:

  1. Reporting traceback SCAFFOLDING. The first run against the live log
     ranked "Traceback (most recent call last):" and "The above exception was
     the direct cause of..." as the top two issues. Those are the frame around
     an error, never the error, and they crowd out the real fault.
  2. Miscounting sessions. The "--- Jarvis started" banner is written directly
     by setup_logging and carries NO "[timestamp]" prefix, so a naive window
     filter inherits the previous line's state and happily counts a rotated
     log's months-old sessions as recent. That inflated 57 real sessions to
     441.

    venv\Scripts\python.exe tests\self_review_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = tempfile.mkdtemp(prefix="jarvis-selfreview-")
os.environ["LOCALAPPDATA"] = _TMP

from src import self_review as sr  # noqa: E402

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


def reset_logs() -> None:
    """Remove every log file, INCLUDING rotations.

    Needed because the rotated-log case writes jarvis.log.1, and a helper that
    only overwrites jarvis.log leaks it into every later case. That is exactly
    what happened on this suite's first run: two failures that looked like
    scanner bugs and were stale fixtures.
    """
    d = Path(_TMP) / "Jarvis"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("jarvis.log*"):
        f.unlink()


def write_log(lines: list[str], name: str = "jarvis.log") -> None:
    d = Path(_TMP) / "Jarvis"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def ts(days_ago: float) -> str:
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


def banner(days_ago: float) -> str:
    stamp = (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")
    return f"--- Jarvis started {stamp} ---"


# =========================================================================
# 1. signature normalisation collapses the volatile parts
# =========================================================================
a = sr.signature("[2026-07-20 10:00:00] [outlook] ical fetch failed: timed out after 2421ms")
b = sr.signature("[2026-07-25 18:31:02] [outlook] ical fetch failed: timed out after 998ms")
check("same fault at different times/durations shares a signature", a == b, f"{a!r} vs {b!r}")

c = sr.signature("[2026-07-20 10:00:00] [outlook] fetch failed — retrying in 0.5s")
d = sr.signature("[2026-07-20 10:00:01] [outlook] fetch failed")
check("the retry suffix does not fork the signature", c == d, f"{c!r} vs {d!r}")

e = sr.signature(r"[2026-07-20 10:00:00] load failed: C:\Users\someone\thing.pt missing")
f = sr.signature(r"[2026-07-20 10:00:00] load failed: C:\Users\other\thing.pt missing")
check("absolute paths are normalised away", e == f, f"{e!r} vs {f!r}")

# =========================================================================
# 2. THE TRACEBACK CASE — scaffolding must not outrank the real error
# =========================================================================
reset_logs()
write_log([
    banner(1),
    f"[{ts(1)}] Exception in thread Thread-1 (listen_loop):",
    "Traceback (most recent call last):",
    '  File "C:\\Users\\x\\repos\\jarvis\\src\\audio.py", line 88, in _open',
    "    raise PortAudioError(errormsg, err)",
    "    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
    "The above exception was the direct cause of the following exception:",
    "sounddevice.PortAudioError: Error opening InputStream: Device unavailable",
])
data = sr.scan(days=7)
sigs = [s["signature"] for s in data["signatures"]]
check("traceback header is NOT reported as an issue",
      not any("Traceback (most recent call last)" in s for s in sigs), str(sigs))
check("'direct cause' scaffolding is NOT reported",
      not any("direct cause" in s for s in sigs), str(sigs))
check("the real exception IS reported",
      any("PortAudioError" in s for s in sigs), str(sigs))
check("a traceback marks the session as crashed", data["crashes"] == 1, str(data["crashes"]))

# =========================================================================
# 3. recurrence is counted in SESSIONS, not just occurrences
# =========================================================================
lines: list[str] = []
for day in (1, 2, 3):
    lines.append(banner(day))
    lines.append(f"[{ts(day)}] [outlook] ical fetch failed: ConnectTimeout")
# one session that fails twice — same total count, weaker signal
lines.append(banner(4))
lines.append(f"[{ts(4)}] [plex] connect failed: refused")
lines.append(f"[{ts(4)}] [plex] connect failed: refused")
lines.append(f"[{ts(4)}] [plex] connect failed: refused")
reset_logs()
write_log(lines)

data = sr.scan(days=7)
by_sig = {s["signature"]: s for s in data["signatures"]}
outlook = next(v for k, v in by_sig.items() if "outlook" in k)
plex = next(v for k, v in by_sig.items() if "plex" in k)
check("a fault in 3 sessions records 3 sessions", outlook["sessions"] == 3, str(outlook))
check("3 hits inside ONE session records 1 session", plex["sessions"] == 1, str(plex))
check("ranking puts the multi-session fault first (standing defect beats bad day)",
      data["signatures"][0]["sessions"] == 3, str(data["signatures"][:2]))
check("session count is right", data["sessions"] == 4, str(data["sessions"]))

# =========================================================================
# 4. THE WINDOW — an old session in a rotated log must not count as recent
# =========================================================================
reset_logs()
write_log([
    banner(200),
    f"[{ts(200)}] [outlook] ancient failure: timed out",
    banner(1),
    f"[{ts(1)}] [weather] fetch err: timed out",
])
data = sr.scan(days=7)
check("a 200-day-old session is outside a 7-day window", data["sessions"] == 1,
      str(data["sessions"]))
check("its errors are excluded too",
      not any("ancient" in s["signature"] for s in data["signatures"]),
      str([s["signature"] for s in data["signatures"]]))
check("widening the window brings it back", sr.scan(days=365)["sessions"] == 2)

# =========================================================================
# 5. rotated logs are read (the trend would otherwise be cut in half)
# =========================================================================
reset_logs()
write_log([banner(1), f"[{ts(1)}] [a] failed: one"])
write_log([banner(2), f"[{ts(2)}] [b] failed: two"], name="jarvis.log.1")
data = sr.scan(days=7)
check("rotated logs are included", data["sessions"] == 2, str(data["sessions"]))
check("faults from both files are grouped together",
      len(data["signatures"]) == 2, str(data["signatures"]))

# =========================================================================
# 6. benign lines don't become issues
# =========================================================================
reset_logs()
write_log([
    banner(1),
    f"[{ts(1)}] [self] scan complete, 0 errors",
    f"[{ts(1)}] [status] no errors since session start",
    f"[{ts(1)}] [llm] ignoring JARVIS_VOICE_EFFORT='turbo' - expected one of low, medium",
])
data = sr.scan(days=7)
check("'0 errors' is not reported as an error", data["concerning"] == 0,
      str([s["signature"] for s in data["signatures"]]))

# =========================================================================
# 7. rendering — spoken, so grammar and empty states matter
# =========================================================================
reset_logs()
write_log([banner(1), f"[{ts(1)}] [x] failed: single event"])
text = sr.format_review(sr.scan(days=7))
check("a single occurrence reads as 'Once', not '1 times'",
      "Once" in text and "1 times" not in text, text)

reset_logs()
write_log([banner(1), f"[{ts(1)}] all good"])
clean = sr.format_review(sr.scan(days=7))
check("a clean week says so plainly", "nothing of concern" in clean.lower(), clean)

import shutil  # noqa: E402
shutil.rmtree(Path(_TMP) / "Jarvis", ignore_errors=True)
check("no log at all is handled without raising",
      "no log activity" in sr.execute_self_review({}).lower())

# =========================================================================
# 8. tool wiring
# =========================================================================
from src import llm  # noqa: E402

check("self_review is registered", "self_review" in llm._CLIENT_TOOLS)
check("self_review is READ-ONLY, so not denied to remote origins",
      "self_review" not in llm._RESTRICTED_DENY)
check("bad days= input is clamped, not fatal",
      isinstance(sr.execute_self_review({"days": "banana"}), str))

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
