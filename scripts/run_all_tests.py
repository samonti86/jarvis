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
These are instruments you reach for by hand, not gates. They're listed in
_EXCLUDED so a future reader knows the omission is deliberate, not an oversight.
(Note: _EXCLUDED only matters for files that DO end in `*_test.py`; other
standalone scripts simply aren't globbed.)

Each suite is run with THIS interpreter (sys.executable), so when you invoke
it as `venv\Scripts\python.exe scripts\run_all_tests.py` the children inherit
the venv and every dependency resolves.

    venv\Scripts\python.exe scripts\run_all_tests.py        # exit 0 = all green
    venv\Scripts\python.exe scripts\run_all_tests.py -v     # stream all output

CI MODE (--allow-missing-deps):
Jarvis's full dependency set is not installable on a stock CI runner — torch,
ctranslate2, onnxruntime, opencv and the audio/GUI bindings need native builds,
hardware, or ~2 GB of wheels. Rather than maintain a hand-curated "suites CI can
run" allowlist (which silently rots the moment a suite is added), `--allow-missing-
deps` reclassifies exactly one failure mode as SKIP: the suite died on an
ImportError for a module in _OPTIONAL_DEPS. Every other failure is still a FAIL.

That keeps ONE gate command for both local and CI — a new suite is picked up by
CI automatically if its deps are light, and skipped (loudly, by name) if not.
Skips are always printed, never silently folded into the pass count.
"""

from __future__ import annotations

import glob
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Standalone scripts that are NOT regression gates (see module docstring).
_EXCLUDED = {"leak_repro.py", "barge_stutter_soak.py"}

# Third-party modules that cannot be assumed present on a stock CI runner:
# native builds (dlib), multi-GB ML wheels (torch), hardware bindings
# (sounddevice → PortAudio), or a GUI/display (tkinter, pystray).
# A suite that dies importing one of these is SKIPPED under --allow-missing-deps,
# never silently passed. Everything else is a genuine failure.
_OPTIONAL_DEPS = {
    "torch", "torchaudio", "torchvision", "ultralytics", "cv2", "dlib",
    "face_recognition", "sounddevice", "panns_inference", "resemblyzer",
    "openwakeword", "faster_whisper", "ctranslate2", "sentence_transformers",
    "transformers", "onnxruntime", "pyttsx3", "pystray", "customtkinter",
    "tkinter", "scipy", "librosa", "numba", "miniaudio", "pyaec", "pycaw",
    "webrtcvad", "soundfile", "howlongtobeatpy", "plexapi", "mcp", "discord",
}

# `ModuleNotFoundError: No module named 'x'` / `ImportError: ... 'x'`
_MISSING_MOD_RE = re.compile(r"No module named ['\"]([A-Za-z0-9_.]+)['\"]")

# Per-suite hard timeout. Heavy suites import torch / sentence-transformers,
# whose first import is slow but nowhere near this; a suite hitting this is hung.
_TIMEOUT_S = 600

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv
ALLOW_MISSING = "--allow-missing-deps" in sys.argv


def _missing_optional_dep(output: str) -> str | None:
    """Return the optional module a suite died importing, if that's why it failed.

    Only the TOP-LEVEL package is checked against _OPTIONAL_DEPS, so
    `torch.nn.functional` matches `torch`. Returns None when the failure was
    anything other than a missing *optional* dependency — a missing REQUIRED
    dep (httpx, numpy) is a broken environment and must still fail loudly.
    """
    for mod in _MISSING_MOD_RE.findall(output):
        if mod.split(".")[0] in _OPTIONAL_DEPS:
            return mod
    return None


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

    # --- Gate 1b: import main.py -----------------------------------------
    # py_compile only checks syntax. `import main` additionally executes all
    # module-level wiring (top-level imports + any import-time side effects)
    # WITHOUT running main() (guarded by __main__). This is the cheapest net
    # for a main.py refactor: a botched extraction that leaves a NameError or
    # a broken import fails here loudly, and main() has no other test coverage.
    ok, out = _run("import_main", [py, "-c", "import main"])
    results.append(("import main (module-level wiring)", ok, out))

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
    print("Jarvis regression gate" + ("  [CI: optional deps may be absent]" if ALLOW_MISSING else ""))
    print("=" * 70)
    failed_any = False
    skipped: list[tuple[str, str]] = []
    for label, ok, out in results:
        missing = None if ok else (_missing_optional_dep(out) if ALLOW_MISSING else None)
        if missing:
            status = "SKIP"
            skipped.append((label, missing))
        else:
            status = "PASS" if ok else "FAIL"
        tail = _summary_tail(out)
        suffix = f"   ({tail})" if tail else ""
        if status == "SKIP":
            suffix = f"   (optional dep not installed: {missing})"
        print(f"  [{status}]  {label}{suffix}")
        if status == "FAIL":
            failed_any = True
        # Show output when verbose, or always for a real failure (so the cause
        # is right there, not hidden behind a re-run). A SKIP is expected in CI
        # and its traceback is noise.
        if VERBOSE or status == "FAIL":
            for line in out.strip().splitlines():
                print(f"          | {line}")

    elapsed = time.monotonic() - t0
    n = len(results)
    n_fail = sum(
        1
        for label, ok, out in results
        if not ok and not (ALLOW_MISSING and _missing_optional_dep(out))
    )
    n_skip = len(skipped)
    print("-" * 70)
    print(f"  {n - n_fail - n_skip}/{n} gates passed"
          + (f", {n_skip} skipped" if n_skip else "")
          + (f", {n_fail} FAILED" if n_fail else "")
          + f"   ({elapsed:.1f}s)")
    if skipped:
        # Never let a skip masquerade as coverage — name every one.
        print("-" * 70)
        print("  Skipped (optional dependency absent - run the full gate locally):")
        for label, mod in skipped:
            print(f"    - {label}  (needs {mod})")
    print("=" * 70)
    return 1 if failed_any else 0


if __name__ == "__main__":
    sys.exit(main())
