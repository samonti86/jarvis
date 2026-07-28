r"""Regression test for per-line log timestamps (2026-06-21).

WHY THIS EXISTS:
For months only session markers ('--- Jarvis started ... ---') and the Plex MCP
child carried a timestamp; every [security]/[acoustic]/[jarvis] print() was bare
text. That made a multi-day soak un-reasonable-about temporally — two lines being
file-adjacent did NOT mean they happened close in time (across an overnight
shutdown gap, a disarm could sit right next to the next morning's boot). A live
log review tripped on exactly that. The fix stamps every emitted line at the
stdout/stderr redirect chokepoint via src/logfile.TimestampStream.

This suite locks in the stamping CONTRACT: one stamp per non-empty line, blank
lines left clean, multi-line writes (tracebacks) each stamped, continuation
writes (print's "text" then "\n") not double-stamped, and the stream wrapper's
fileno() delegation (load-bearing — the Plex MCP child inherits the real fd).

    python tests/logfile_timestamp_test.py     # exit 0 = pass
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logfile import TimestampStream, stamp_chunk  # noqa: E402

PASSED = 0
FAILED = 0

TS = "[TS] "  # fixed fake stamp so the pure core is deterministic


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok: {label}")
    else:
        FAILED += 1
        print(f"  FAIL: {label}  {detail}")


print("\n[group] stamp_chunk — single line")
out, at_start = stamp_chunk("[security] armed\n", True, TS)
check("one line gets one leading stamp", out == "[TS] [security] armed\n", repr(out))
check("ends at line start (trailing newline)", at_start is True)


print("\n[group] stamp_chunk — print()'s two-call pattern")
# print("x") => write("x") then write("\n"). The second call must NOT re-stamp.
out1, mid = stamp_chunk("hello", True, TS)
check("content-only write stamps once", out1 == "[TS] hello", repr(out1))
check("no trailing newline -> mid-line state", mid is False)
out2, end = stamp_chunk("\n", mid, TS)
check("the closing newline is not stamped", out2 == "\n", repr(out2))
check("newline returns to line-start", end is True)


print("\n[group] stamp_chunk — multi-line write (a traceback)")
out, at_start = stamp_chunk("Traceback:\n  line1\n  line2\n", True, TS)
check(
    "every line in one write is stamped",
    out == "[TS] Traceback:\n[TS]   line1\n[TS]   line2\n",
    repr(out),
)
check("multi-line ends at line start", at_start is True)


print("\n[group] stamp_chunk — blank lines stay clean")
out, at_start = stamp_chunk("\n", True, TS)
check("lone newline is NOT stamped", out == "\n", repr(out))
check("blank line keeps line-start state", at_start is True)
# The session marker print writes "\n--- Jarvis started ... ---\n": a leading
# blank separator, then the marker. The blank stays clean, the marker stamped.
out, _ = stamp_chunk("\n--- Jarvis started X ---\n", True, TS)
check(
    "leading blank then content: blank clean, content stamped",
    out == "\n[TS] --- Jarvis started X ---\n",
    repr(out),
)
check("'--- Jarvis started' still greppable after stamping",
      "--- Jarvis started" in out)


print("\n[group] stamp_chunk — empty input is a no-op")
out, at_start = stamp_chunk("", True, TS)
check("empty string -> empty out, state unchanged (True)", out == "" and at_start is True)
out, at_start = stamp_chunk("", False, TS)
check("empty string -> state unchanged (False)", out == "" and at_start is False)


print("\n[group] TimestampStream — end-to-end through a sink")
sink = io.StringIO()
ts = TimestampStream(sink)
ts.write("alpha\n")
ts.write("beta")   # no newline yet (mid-line)
ts.write("\n")     # close the line
val = sink.getvalue()
# Real clock here; assert the SHAPE rather than the exact time.
lines = val.splitlines()
check("two physical lines written", len(lines) == 2, repr(val))
check("line 1 stamped + content", lines[0].endswith("] alpha") and lines[0].startswith("["), repr(lines[0]))
check("line 2 not double-stamped across the split write",
      lines[1].endswith("] beta") and lines[1].count("[") == 1, repr(lines[1]))


print("\n[group] TimestampStream — interface contract")
check("write returns len(data) (print expects it)", TimestampStream(io.StringIO()).write("xyz") == 3)
check("isatty is False (not a terminal)", TimestampStream(io.StringIO()).isatty() is False)

# fileno() delegation — load-bearing for the Plex MCP child's stderr inheritance.
class _FakeFile:
    def fileno(self) -> int:
        return 42
    def write(self, s: str) -> int:
        return len(s)

check("fileno() delegates to the wrapped stream", TimestampStream(_FakeFile()).fileno() == 42)

# A sink without fileno() raises like a normal stream would (no silent swallow).
try:
    TimestampStream(io.StringIO()).fileno()
    check("fileno() on a filenoless sink raises", False, "expected an error")
except (io.UnsupportedOperation, AttributeError, OSError):
    check("fileno() on a filenoless sink raises", True)


print("\n" + "=" * 50)
print(f"{PASSED} passed, {FAILED} failed")
print("=" * 50)
sys.exit(1 if FAILED else 0)
