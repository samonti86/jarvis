r"""Regression test for src/atomic_io.atomic_write_text (2026-06-21).

WHY THIS EXISTS:
The small JSON stores (reminders, predictions, quiet-hours, ui_state) persisted
with temp-file + os.replace but NO fsync — so a hard power loss (this machine has
no UPS) could commit a zero-length/torn store even though os.replace is atomic
against an *interrupted* write. They also shared a single fixed temp name, so two
concurrent savers (the background prediction cycle vs. the briefing) would corrupt
the temp. atomic_io.atomic_write_text fixes both: fsync before replace, and a
unique temp per write. This suite locks that contract: content round-trips, the
target is replaced atomically, no stray temp survives (success or error), and
concurrent writers don't blow up or leave a torn file.

    python scripts/atomic_io_test.py     # exit 0 = pass
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.atomic_io import atomic_write_text  # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok: {label}")
    else:
        FAILED += 1
        print(f"  FAIL: {label}  {detail}")


_TMP = Path(tempfile.mkdtemp(prefix="jarvis_atomic_io_test_"))


def _temps_in_dir() -> list[str]:
    return [p.name for p in _TMP.iterdir() if p.name.endswith(".tmp")]


print("\n[group] basic round-trip + replace")
target = _TMP / "store.json"
atomic_write_text(target, '{"a": 1}')
check("file created with exact content", target.read_text(encoding="utf-8") == '{"a": 1}')
atomic_write_text(target, '{"a": 2}')
check("overwrite replaces content", target.read_text(encoding="utf-8") == '{"a": 2}')
check("no stray .tmp left after success", _temps_in_dir() == [], repr(_temps_in_dir()))
check("accepts a str path too", (atomic_write_text(str(target), "x") or target.read_text()) == "x")


print("\n[group] unicode + large content")
big = "ñ✓🔔" * 5000
atomic_write_text(target, big)
check("unicode/large content round-trips", target.read_text(encoding="utf-8") == big)


print("\n[group] error path leaves no stray temp, doesn't clobber the target")
atomic_write_text(target, "good")
# A non-serializable write can't happen (we pass text), so simulate a write
# failure by pointing at a directory that doesn't exist -> mkstemp raises.
raised = False
try:
    atomic_write_text(_TMP / "no_such_subdir" / "f.json", "data")
except (FileNotFoundError, OSError):
    raised = True
check("write into a missing dir raises (caller wraps if it must not)", raised)
check("the unrelated target is untouched after an error elsewhere",
      target.read_text(encoding="utf-8") == "good")
check("no stray .tmp after the failed write", _temps_in_dir() == [], repr(_temps_in_dir()))


print("\n[group] concurrent writers don't corrupt (unique temp names)")
# 20 threads hammering the same target. Each value is a complete, valid token;
# the final file must be exactly ONE of them (last writer wins), never torn,
# and no temp may survive.
errors: list[Exception] = []
barrier = threading.Barrier(20)


def _writer(i: int) -> None:
    try:
        barrier.wait()
        atomic_write_text(target, f"value-{i:03d}")
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)


threads = [threading.Thread(target=_writer, args=(i,)) for i in range(20)]
for t in threads:
    t.start()
for t in threads:
    t.join()

final = target.read_text(encoding="utf-8")
check("no writer raised under concurrency", not errors, repr(errors[:2]))
check("final content is exactly one complete write (not torn)",
      final in {f"value-{i:03d}" for i in range(20)}, repr(final))
check("no stray .tmp after concurrent writes", _temps_in_dir() == [], repr(_temps_in_dir()))


print("\n" + "=" * 50)
print(f"{PASSED} passed, {FAILED} failed")
print("=" * 50)
sys.exit(1 if FAILED else 0)
