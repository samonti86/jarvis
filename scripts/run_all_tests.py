r"""Unified test runner / close-out regression gate for Jarvis.

WHY THIS EXISTS:
Until now the close-out gate was `py_compile` (syntax only) and the dozen
`scripts/*_test.py` suites were each run by hand. That's no regression net —
a reorg that silently breaks the announce path or a tool schema compiles
fine and passes "py_compile" while being thoroughly broken. This script is
the single command that says "is the codebase still green?", so any later
refactor has something to fail against.

WHAT IT RUNS (in order, fail-fast OFF — every gate runs so you see the full
picture, exit code reflects whether ANY failed):
  1. py_compile syntax gate over main.py + the launchers + every src/*.py
  2. js_parse_gate.py — structural JS check on the inlined PWA script
  3. every scripts/*_test.py unit/regression suite, each in its own subprocess

WHAT IT DELIBERATELY EXCLUDES:
  - leak_repro.py, barge_stutter_soak.py  -> long-running soaks (minutes/hours)
  - outlook_auth.py                       -> interactive device-code OAuth
These are instruments you reach for by hand, not gates. They're listed in
_EXCLUDED so a future reader knows the omission is deliberate, not an oversight.

Each suite is run with THIS interpreter (sys.executable), so when you invoke
it as `venv\Scripts\python.exe scripts\run_all_tests.py` the children inherit
the venv and every dependency resolves.

    venv\Scripts\python.exe scripts\run_all_tests.py        # exit 0 = all green
    venv\Scripts\python.exe scripts\run_all_tests.py -v     # stream all output
"""

from __future__ import annotations

import glob
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Standalone scripts that are NOT regression gates (see module docstring).
_EXCLUDED = {"leak_repro.py", "barge_stutter_soak.py", "outlook_auth.py"}

# Per-suite hard timeout. Heavy suites import torch / sentence-transformers,
# whose first import is slow but nowhere near this; a suite hitting this is hung.
_TIMEOUT_S = 600

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv


def _run(label: str, argv: list[str]) -> tuple[bool, str]:
    """Run one gate as a subprocess; return (passed, captured_output)."""
    try:
        proc = subprocess.run(
            argv,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {_TIMEOUT_S}s"
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out


def _summary_tail(output: str) -> str:
    """Pull the '<n> passed, <m> failed' line a suite prints, if present."""
    for line in reversed(output.strip().splitlines()):
        s = line.strip()
        if "passed" in s and "failed" in s:
            return s
    return ""


def main() -> int:
    py = sys.executable
    results: list[tuple[str, bool, str]] = []
    t0 = time.monotonic()

    # --- Gate 1: py_compile syntax over the whole tree --------------------
    src_files = sorted(glob.glob(str(ROOT / "src" / "*.py")))
    compile_targets = [
        str(ROOT / "main.py"),
        str(ROOT / "jarvis.pyw"),
        str(ROOT / "jarvis_watchdog.pyw"),
        *src_files,
    ]
    ok, out = _run("py_compile", [py, "-m", "py_compile", *compile_targets])
    results.append(("py_compile (syntax gate)", ok, out))

    # --- Gate 2: structural JS check on the PWA ---------------------------
    js_gate = SCRIPTS / "js_parse_gate.py"
    if js_gate.exists():
        ok, out = _run("js_parse_gate", [py, str(js_gate)])
        results.append(("js_parse_gate.py (PWA JS structure)", ok, out))

    # --- Gate 3: every *_test.py suite ------------------------------------
    suites = sorted(
        p for p in SCRIPTS.glob("*_test.py") if p.name not in _EXCLUDED
    )
    for suite in suites:
        ok, out = _run(suite.name, [py, str(suite)])
        results.append((suite.name, ok, out))

    # --- Report -----------------------------------------------------------
    print("=" * 70)
    print("Jarvis regression gate")
    print("=" * 70)
    failed_any = False
    for label, ok, out in results:
        status = "PASS" if ok else "FAIL"
        tail = _summary_tail(out)
        suffix = f"   ({tail})" if tail else ""
        print(f"  [{status}]  {label}{suffix}")
        if not ok:
            failed_any = True
        # Show output when verbose, or always for a failure (so the cause
        # is right there, not hidden behind a re-run).
        if VERBOSE or not ok:
            for line in out.strip().splitlines():
                print(f"          | {line}")

    elapsed = time.monotonic() - t0
    n = len(results)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("-" * 70)
    print(f"  {n - n_fail}/{n} gates passed   ({elapsed:.1f}s)")
    print("=" * 70)
    return 1 if failed_any else 0


if __name__ == "__main__":
    sys.exit(main())
