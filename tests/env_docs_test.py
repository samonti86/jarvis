r"""Config-documentation drift gate.

WHY THIS EXISTS (post-mortem, 2026-07-28):
`docs/ENV_VARS.md` was created for audit finding 2.2 ("config sprawl") to be the
single inventory of every tunable. Twelve milestones later it had silently
drifted: 23 settings shipped in `.env.example` that the reference never
documented (the whole M84 HUD, M85 tone, M79 quiet-hours, M83 anticipation and
Discord-bot families among them), plus 7 more read by code and documented
nowhere at all. Nothing failed — docs just quietly stopped being true, which is
worse than having no docs, because a reader trusts them.

A bug a test would have caught earns a test. This is that test.

THE INVARIANT (one direction, deliberately):
    every env var the CODE READS must appear in docs/ENV_VARS.md

Not "the docs match .env.example" — `.env.example` is a curated onboarding
template and is *supposed* to be a subset. The authoritative source is the code:
if `os.getenv("JARVIS_NEW_THING")` exists, a reader must be able to look it up.

Adding a tunable therefore costs one table row. That is the entire point.

    venv\Scripts\python.exe scripts\env_docs_test.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "ENV_VARS.md"

# Matched STRUCTURALLY rather than by an enumerated list of helper names: the
# first draft of this gate listed accessors explicitly and immediately went
# blind to `_env_float`, `_env_set` and `_int_env`. Anything named `*env*` that
# takes an UPPER_SNAKE string literal counts, so new helpers are covered for
# free — which is the whole point of a drift gate.
_READ_RE = re.compile(
    r"""(?:os\.)?(?:getenv|environ\.get|[A-Za-z_]*env[A-Za-z_]*)
        \(\s*["']([A-Z][A-Z0-9_]*)["']""",
    re.VERBOSE,
)

# Vars that belong to the OPERATING SYSTEM, not to Jarvis. Documenting these in
# a Jarvis config reference would be noise.
_OS_VARS = {
    "APPDATA", "LOCALAPPDATA", "USERPROFILE", "PATH", "TEMP", "TMP",
    "HOME", "COMPUTERNAME", "USERNAME", "SYSTEMROOT", "PROGRAMFILES",
    "PROGRAMDATA", "COMSPEC", "PATHEXT", "OS",
}

_SEARCH_ROOTS = ("src", "stt_server")


def _vars_read_by_code() -> dict[str, list[str]]:
    """Map var name -> the files that read it."""
    found: dict[str, list[str]] = {}
    targets = [ROOT / "main.py", ROOT / "jarvis.pyw", ROOT / "jarvis_watchdog.pyw"]
    for sub in _SEARCH_ROOTS:
        targets.extend(sorted((ROOT / sub).rglob("*.py")))
    for path in targets:
        if not path.exists() or "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name in _READ_RE.findall(text):
            if name in _OS_VARS:
                continue
            found.setdefault(name, []).append(
                str(path.relative_to(ROOT)).replace("\\", "/")
            )
    return found


def _vars_documented() -> set[str]:
    """Names appearing as `CODE_SPANS` in the reference doc."""
    text = DOC.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r"`([A-Z][A-Z0-9_]*)`", text))


def main() -> int:
    passed = failed = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"  PASS  {label}")
        else:
            failed += 1
            print(f"  FAIL  {label}")
            if detail:
                for line in detail.splitlines():
                    print(f"          {line}")

    check("docs/ENV_VARS.md exists", DOC.exists())
    if not DOC.exists():
        print("\n0 passed, 1 failed")
        return 1

    read = _vars_read_by_code()
    documented = _vars_documented()

    check("code reads at least 50 env vars (scanner is finding things)", len(read) >= 50,
          f"found only {len(read)} — the accessor regex may have gone stale")

    undocumented = sorted(set(read) - documented)
    check(
        "every env var read by code is documented in ENV_VARS.md",
        not undocumented,
        "\n".join(
            f"{name}  (read in {', '.join(sorted(set(read[name]))[:3])})"
            for name in undocumented
        )
        + ("\n\nAdd a row to docs/ENV_VARS.md for each." if undocumented else ""),
    )

    # A var documented but no longer read is dead config — worth flagging, but
    # not fatal: the internal vars (JARVIS_LOG_PATH) are SET by launchers and
    # read via a different path, and stt_server vars may be read server-side.
    stale = sorted(documented - set(read) - _OS_VARS)
    if stale:
        print(f"  NOTE  {len(stale)} documented but not found by the scanner "
              f"(expected for launcher-set/internal vars): {', '.join(stale[:8])}"
              + (" ..." if len(stale) > 8 else ""))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
