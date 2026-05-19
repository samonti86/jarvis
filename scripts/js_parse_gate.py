"""M48.2b — structural JS gate for the inlined PWA script.

WHY THIS EXISTS (the "presence ≠ valid" lesson, institutionalized):
A bad Edit once left a DUPLICATE `function setSpk()` in src/remote_pwa.py.
`py_compile` passed (it's a valid Python raw string) and a substring smoke
test passed (the text we looked for was *present*) — but the browser threw
a JS SyntaxError and the whole console went dead. A grep for "is it there?"
cannot catch "is it well-formed?". `node --check` is the right tool but node
isn't installed on this box, so this is the next best thing: a real
comment/string/template-literal-aware brace/paren/bracket balancer plus a
few regression-specific structural assertions. It is NOT a full JS parser;
it is a cheap, deterministic structural check that would have caught the
duplicate-function (extra `}`) and any unbalanced/unterminated edit.

Run as part of the close-out sanity gate whenever remote_pwa.py changes:
    python scripts/js_parse_gate.py
Exit 0 = PASS, non-zero = FAIL (prints why).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _extract_script(html: str) -> str:
    m = re.search(r"<script>(.*?)</script>", html, re.S)
    if not m:
        raise SystemExit("FAIL: no <script> block found in PWA_HTML")
    return m.group(1)


def _balance(src: str) -> list[str]:
    """State machine: counts ()[]{} only in CODE context, skipping line
    comments, block comments, '...' "..." strings and `...` template
    literals (including nested ${ } expressions and nested templates)."""
    errs: list[str] = []
    stack: list[str] = []          # open brackets, CODE context only
    tmpl_marker: list[int] = []    # stack-len at each open ${ (its } returns to tmpl)
    pairs = {")": "(", "]": "[", "}": "{"}
    state = "code"
    i, n = 0, len(src)
    line = 1
    while i < n:
        c = src[i]
        d = src[i + 1] if i + 1 < n else ""
        if c == "\n":
            line += 1
        if state == "code":
            if c == "/" and d == "/":
                state = "line"; i += 2; continue
            if c == "/" and d == "*":
                state = "block"; i += 2; continue
            if c == "'":
                state = "sq"; i += 1; continue
            if c == '"':
                state = "dq"; i += 1; continue
            if c == "`":
                state = "tmpl"; i += 1; continue
            if c in "([{":
                stack.append(c); i += 1; continue
            if c in ")]}":
                if (c == "}" and tmpl_marker
                        and len(stack) == tmpl_marker[-1]):
                    tmpl_marker.pop(); stack.pop()  # closes a ${ } expr
                    state = "tmpl"; i += 1; continue
                if not stack:
                    errs.append(f"line {line}: unbalanced '{c}'")
                elif stack[-1] != pairs[c]:
                    errs.append(
                        f"line {line}: '{c}' closes '{stack[-1]}'")
                    stack.pop()
                else:
                    stack.pop()
                i += 1; continue
            i += 1; continue
        if state == "line":
            if c == "\n":
                state = "code"
            i += 1; continue
        if state == "block":
            if c == "*" and d == "/":
                state = "code"; i += 2; continue
            i += 1; continue
        if state == "sq":
            if c == "\\":
                i += 2; continue
            if c == "'":
                state = "code"
            i += 1; continue
        if state == "dq":
            if c == "\\":
                i += 2; continue
            if c == '"':
                state = "code"
            i += 1; continue
        if state == "tmpl":
            if c == "\\":
                i += 2; continue
            if c == "`":
                state = "code"; i += 1; continue
            if c == "$" and d == "{":
                stack.append("{")
                tmpl_marker.append(len(stack))
                state = "code"; i += 2; continue
            i += 1; continue
    if stack:
        errs.append(f"unclosed {stack!r} at EOF")
    if state != "code":
        errs.append(f"unterminated {state} at EOF")
    return errs


def main() -> int:
    pwa = (ROOT / "src" / "remote_pwa.py").read_text(encoding="utf-8")
    # The HTML is a Python raw string; pull its literal contents.
    m = re.search(r'PWA_HTML\s*=\s*r"""(.*?)"""', pwa, re.S)
    if not m:
        print("FAIL: could not locate PWA_HTML = r\"\"\"...\"\"\"")
        return 1
    html = m.group(1)
    script = _extract_script(html)

    errs = _balance(script)
    if errs:
        print("JS GATE FAIL (structure):")
        for e in errs:
            print("  -", e)
        return 1

    # Regression-specific structural assertions. Each maps to a real bug
    # this project actually hit, or a M48.2b invariant we must not lose.
    checks = [
        (script.count("function dial(") == 1, "exactly one function dial("),
        (script.count("function connect(") == 1, "exactly one function connect("),
        (script.count("function setSpk(") == 1,
         "exactly one function setSpk( (the duplicate-fn regression)"),
        (script.count("function playReply(") == 1, "one function playReply("),
        (script.count("function onPlayBlocked(") == 1, "one onPlayBlocked("),
        ("let connWD" in script and "CONN_TIMEOUT_MS" in script,
         "connect watchdog present (the stuck-CONNECTING fix)"),
        ("clearWD()" in script, "watchdog cleared on auth_ok/close"),
        ('pill.addEventListener("click"' in script,
         "status pill is tap-to-reconnect (guaranteed manual recovery)"),
        ('e.persisted' in script, "bfcache pageshow/persisted handler kept"),
        ('window.addEventListener("online"' in script, "online→dial kept"),
        ("SILENT" not in script and "ae.loop = true" not in script
         and "ae.loop=true" not in script,
         "no stale silent-WAV keepalive / no audio loop (removed)"),
        (re.search(r"dial\(\);\s*\}\)\(\);\s*$", script.strip()) is not None,
         "bootstrap ends with dial();  (single entry point)"),
    ]
    bad = [why for ok, why in checks if not ok]
    if bad:
        print("JS GATE FAIL (invariants):")
        for b in bad:
            print("  -", b)
        return 1

    print("JS GATE PASS: balanced; 1x dial/connect/setSpk/playReply; "
          "connWD watchdog + clearWD; tap-to-reconnect pill; "
          "bfcache+online kept; no audio-loop; bootstrap=dial();")
    return 0


if __name__ == "__main__":
    sys.exit(main())
