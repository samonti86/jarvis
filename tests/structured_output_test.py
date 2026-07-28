r"""Regression test for structured outputs on the background jobs (M96).

The prediction miner and resolver used to ask for JSON in prose and dig it back
out with a GREEDY regex (`\[.*\]` across the whole reply). That had never
actually failed in production — checked before changing it — but it is a latent
failure class: any bracket in surrounding prose captures the wrong span.

output_config.format removes the class rather than mitigating it: the reply IS
the JSON document. The old extractor is kept as a fallback, so this suite
asserts BOTH paths — the constrained one and the degraded one.

    venv\Scripts\python.exe tests\structured_output_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import predictions as pr  # noqa: E402

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


# --- schemas must satisfy the structured-outputs subset -------------------
for name, schema in (("miner", pr._MINER_SCHEMA), ("resolver", pr._RESOLVER_SCHEMA)):
    check(f"{name} schema is a closed object",
          schema.get("type") == "object" and schema.get("additionalProperties") is False)
    check(f"{name} schema declares required fields", bool(schema.get("required")))

item = pr._MINER_SCHEMA["properties"]["predictions"]["items"]
check("miner item is closed too", item.get("additionalProperties") is False)
check("nullable uses anyOf, not a type array (type arrays are unsupported)",
      "anyOf" in item["properties"]["resolve_after"]
      and "anyOf" in pr._RESOLVER_SCHEMA["properties"]["correct"],
      "a JSON-Schema type ARRAY is outside the supported subset")

# --- the constrained path: reply IS the document --------------------------
clean = '{"predictions": [{"claim": "c", "subject": "s", "made_at": "2026-01-01", "resolve_after": null}]}'
got = pr._parse_structured(clean, key="predictions")
check("structured reply unwraps to the list", isinstance(got, list) and len(got) == 1, str(got))
check("null survives as None", got[0]["resolve_after"] is None, str(got))

verdict = pr._parse_structured('{"resolved": true, "correct": false, "actual": "x"}')
check("resolver reply parses to a dict", isinstance(verdict, dict) and verdict["correct"] is False)

# --- the fallback path: a model that ignored the constraint ----------------
messy = 'Here you go:\n```json\n{"predictions": [{"claim": "c"}]}\n```\nHope that helps!'
got = pr._parse_structured(messy, key="predictions")
check("prose-wrapped JSON still parses via the fallback",
      isinstance(got, list) and got[0]["claim"] == "c", str(got))

check("unparseable input yields None, not a crash",
      pr._parse_structured("no json at all here") is None)
check("empty input yields None", pr._parse_structured("") is None)

# A missing key must not silently return the whole envelope as if it were the list.
check("a missing key returns the raw document, not a wrong-shaped list",
      isinstance(pr._parse_structured('{"other": 1}', key="predictions"), dict))

# --- the calls actually send the constraint -------------------------------
sent: dict = {}


class _FakeMessages:
    def create(self, **kwargs):
        sent.clear(); sent.update(kwargs)
        raise RuntimeError("stop here — we only wanted the kwargs")


class _FakeClient:
    messages = _FakeMessages()


_orig = pr.anthropic.Anthropic
pr.anthropic.Anthropic = lambda **kw: _FakeClient()
import os  # noqa: E402

# Assign, do NOT setdefault: the variable usually EXISTS as an empty string in
# a bare test process (no dotenv loaded), so setdefault is a silent no-op and
# _api_key() keeps returning "" — which makes the miner return [] before it
# ever builds a client, and the assertions below fail for the wrong reason.
os.environ["ANTHROPIC_API_KEY"] = "test-key"
try:
    pr._default_miner("transcript")
    check("miner sends output_config.format",
          sent.get("output_config", {}).get("format", {}).get("type") == "json_schema",
          str(sent.get("output_config")))
    check("miner sends the miner schema",
          sent["output_config"]["format"]["schema"] is pr._MINER_SCHEMA)

    sent.clear()
    pr._default_resolver({"claim": "c", "subject": "s", "made_at": "2026-01-01"}, "2026-07-28")
    check("resolver sends output_config.format",
          sent.get("output_config", {}).get("format", {}).get("type") == "json_schema",
          str(sent.get("output_config")))
    check("resolver KEEPS its web_search tool alongside the constraint",
          bool(sent.get("tools")),
          "structured output and a server tool are compatible — verified live")
finally:
    pr.anthropic.Anthropic = _orig

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
