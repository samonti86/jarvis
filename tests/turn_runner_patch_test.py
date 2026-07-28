r"""Proves the TurnRunner test seams still bind. A meta-test.

WHY THIS EXISTS (post-mortem, 2026-07-28):
`tests/turn_runner_test.py` installs its fakes by rebinding names on a
MODULE OBJECT (`turn_runner.stream_response = fake`). That technique works for
exactly one reason: TurnRunner looks those names up in its own module globals
at CALL time. Two ordinary, well-intentioned refactors silently break it:

  1. MOVING TurnRunner to another module while the test still patches the old
     one. The patch then rebinds a name nothing reads.
  2. Changing HOW the name is resolved — `from src import llm` +
     `llm.stream_response(...)`, or snapshotting into `self` in __init__
     (`self._stream = stream_response`), which freezes the real function at
     construction, before any test can patch it.

In both cases every assertion in turn_runner_test.py still runs, and still
passes, while exercising the REAL streaming call and the REAL speak() instead
of the fakes. A green suite that tests nothing is worse than a red one,
because nobody investigates green.

Case (1) actually happened during the main.py extraction. It failed loudly
with AttributeError — but that was luck (the name was also absent), not a
guarantee. Case (2) would fail silently.

So this gate asserts the seam STRUCTURALLY, which is cheap, needs no fakes,
and cannot itself be fooled by a passing turn:

  A. each collaborator is a module-level attribute of src.turn_runner
     (i.e. there is something to patch), and
  B. TurnRunner's body reads each one as a BARE GLOBAL NAME (so rebinding the
     module attribute changes what the call resolves to), and
  C. none of them is snapshotted onto `self` in __init__ (which would defeat
     any later patch).

    venv\Scripts\python.exe scripts\turn_runner_patch_test.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import turn_runner  # noqa: E402

# The exact names tests/turn_runner_test.py rebinds. Keep the two in sync:
# adding a fake there without adding it here leaves the new seam unguarded.
PATCHED = [
    "stream_response",
    "speak",
    "speak_streaming",
    "MemoryStore",
    "_seal_session",
]

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


# --- A. the attribute exists on the module, so there is a seam to patch ----
for name in PATCHED:
    check(
        f"src.turn_runner.{name} is a module-level attribute (patchable)",
        hasattr(turn_runner, name),
        f"turn_runner_test.py rebinds {name!r}; nothing of that name exists here",
    )

# --- B/C. how TurnRunner actually resolves them ---------------------------
source = (ROOT / "src" / "turn_runner.py").read_text(encoding="utf-8")
tree = ast.parse(source)
cls = next(
    (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TurnRunner"),
    None,
)
check("TurnRunner class found in src/turn_runner.py", cls is not None)

if cls is not None:
    global_loads: set[str] = set()
    for node in ast.walk(cls):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            global_loads.add(node.id)

    for name in PATCHED:
        # MemoryStore is read in __init__; the rest in the turn path. Either
        # way it must appear as a bare Name load somewhere in the class.
        check(
            f"TurnRunner reads {name} as a bare global (late-bound)",
            name in global_loads,
            f"{name} is no longer resolved from module globals - "
            f"turn_runner_test.py's patch of it is now a no-op",
        )

    # C. nothing may be snapshotted onto self, which would freeze the REAL
    # implementation at construction time and defeat every later patch.
    init = next(
        (n for n in cls.body
         if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    snapshotted: list[str] = []
    if init is not None:
        for node in ast.walk(init):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                if node.value.id in PATCHED:
                    for t in node.targets:
                        if (isinstance(t, ast.Attribute)
                                and isinstance(t.value, ast.Name)
                                and t.value.id == "self"):
                            snapshotted.append(f"self.{t.attr} = {node.value.id}")
    check(
        "no collaborator is snapshotted onto self in __init__",
        not snapshotted,
        "; ".join(snapshotted) + "  <- freezes the real impl before patching",
    )

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
