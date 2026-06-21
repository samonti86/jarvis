"""M78.2 — regression test for src/predictions.py ("you called it" follow-ups).

Hermetic: temp LOCALAPPDATA for the store + a synthetic session transcript; the
two LLM steps (miner, resolver) are dependency-injected with deterministic
stubs, so the store logic, dedupe, due-window, resolution-application,
surfacing, and formatting are all tested without the network.

    python scripts/predictions_test.py     # exit 0 = pass
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import predictions as pr  # noqa: E402

_passed = 0
_failed = 0


def check(label: str, condition: bool) -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")


def _iso(days_ago: float) -> str:
    return (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")


def _write_session(base: Path, day: datetime, exchanges) -> None:
    sessions = base / "Jarvis" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    with (sessions / f"{day:%Y-%m-%d}.jsonl").open("a", encoding="utf-8") as f:
        for ts, user, asst in exchanges:
            f.write(json.dumps({"ts": ts, "role": "user", "content": user}) + "\n")
            f.write(json.dumps({"ts": ts, "role": "assistant", "content": asst}) + "\n")


@contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- Test 1: store round-trip + fail-soft on missing/corrupt --------------
with tempfile.TemporaryDirectory() as tmp:
    with _env(LOCALAPPDATA=tmp):
        check("missing store -> default shape",
              pr._load() == {"last_mined_at": None, "predictions": []})
        pr._save({"last_mined_at": "2026-06-10T00:00:00", "predictions": [{"id": "x"}]})
        back = pr._load()
        check("save/load round-trip", back["last_mined_at"] == "2026-06-10T00:00:00"
              and back["predictions"][0]["id"] == "x")
        (Path(tmp) / "Jarvis" / "predictions.json").write_text("{not json", encoding="utf-8")
        check("corrupt store -> default shape, no crash",
              pr._load()["predictions"] == [])


# --- Test 2: stable id is stable + day-granular ---------------------------
a = pr._stable_id("Jarvis predicted the Spurs win", "2026-06-03T16:42:13")
b = pr._stable_id("Jarvis predicted the Spurs win", "2026-06-03T09:00:00")  # same day
c = pr._stable_id("Jarvis predicted the Spurs win", "2026-06-04T16:42:13")  # next day
check("stable id identical for same claim+day", a == b)
check("stable id differs across days", a != c)


# --- Test 3: _normalize_mined validation ----------------------------------
ok = pr._normalize_mined(
    {"claim": "Jarvis predicted the Spurs would win the Finals",
     "subject": "NBA Finals", "resolve_after": "2026-06-18"}, _iso(1))
check("normalize: valid record built with pending status",
      ok and ok["status"] == "pending" and ok["resolve_after"] == "2026-06-18"
      and ok["surfaced"] is False)
check("normalize: too-short claim rejected",
      pr._normalize_mined({"claim": "no"}, _iso(1)) is None)
bad_date = pr._normalize_mined(
    {"claim": "Jarvis predicted a win for the Chiefs", "resolve_after": "soon"}, _iso(1))
check("normalize: bad resolve_after -> null", bad_date["resolve_after"] is None)


# --- Test 4: mining adds new + dedupes, sets watermark --------------------
def _stub_miner(_transcript):
    return [{"claim": "Jarvis predicted the Spurs would win the NBA Finals in six",
             "subject": "NBA Finals", "resolve_after": "2026-06-18"}]


with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    _write_session(base, datetime.now() - timedelta(days=2),
                   [(_iso(2), "who wins the finals?", "I'd lean Spurs in six.")])
    with _env(LOCALAPPDATA=tmp):
        n1 = pr.mine_predictions(miner=_stub_miner)
        n2 = pr.mine_predictions(miner=_stub_miner)   # same prediction again
        store = pr._load()
    check("mining adds the new prediction", n1 == 1)
    check("mining dedupes on re-run", n2 == 0 and len(store["predictions"]) == 1)
    check("mining sets the last_mined_at watermark", store["last_mined_at"] is not None)


# --- Test 4b: made_at FALLBACK uses the EARLIEST exchange, not the latest ---
# When the miner returns no per-prediction date, an old backfilled prediction
# must look OLD (earliest exchange), not fresh — else its first resolve check is
# delayed _DEFAULT_RESOLVE_DELAY_DAYS. Two exchanges 9 and 2 days back; the
# undated mined prediction should be stamped with the 9-day-old date.
def _stub_miner_nodate(_transcript):
    return [{"claim": "Jarvis predicted the Heat would take the series",
             "subject": "NBA", "resolve_after": None}]  # no made_at


with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    _write_session(base, datetime.now() - timedelta(days=9),
                   [(_iso(9), "early take?", "I like the Heat.")])
    _write_session(base, datetime.now() - timedelta(days=2),
                   [(_iso(2), "still?", "Still the Heat.")])
    with _env(LOCALAPPDATA=tmp):
        pr.mine_predictions(miner=_stub_miner_nodate)
        store = pr._load()
    made = store["predictions"][0]["made_at"] if store["predictions"] else ""
    check("undated mined prediction stamped with the EARLIEST exchange date",
          made[:10] == _iso(9)[:10])


# --- Test 5: mining with no transcripts is a clean no-op ------------------
with tempfile.TemporaryDirectory() as tmp:
    with _env(LOCALAPPDATA=tmp):
        n = pr.mine_predictions(miner=_stub_miner)
    check("mining with no sessions -> 0 added, no crash", n == 0)


# --- Test 6: _is_due window logic -----------------------------------------
now = datetime(2026, 6, 11, 9, 0, 0)
due_past = {"status": "pending", "resolve_after": "2026-06-10", "made_at": _iso(20),
            "last_checked_at": None}
not_due_future = {"status": "pending", "resolve_after": "2026-06-20", "made_at": _iso(1),
                  "last_checked_at": None}
due_no_date_old = {"status": "pending", "resolve_after": None,
                   "made_at": "2026-05-20T00:00:00", "last_checked_at": None}
not_due_no_date_recent = {"status": "pending", "resolve_after": None,
                          "made_at": "2026-06-10T00:00:00", "last_checked_at": None}
recently_checked = {"status": "pending", "resolve_after": "2026-06-10",
                    "made_at": _iso(20),
                    "last_checked_at": "2026-06-11T06:00:00"}  # 3h ago < 12h
resolved_already = {"status": "resolved", "resolve_after": "2026-06-10",
                    "made_at": _iso(20), "last_checked_at": None}
due_future_but_mature = {"status": "pending", "resolve_after": "2026-06-30",
                         "made_at": "2026-05-25T00:00:00", "last_checked_at": None}
check("due: resolve_after in the past", pr._is_due(due_past, now) is True)
check("not due: resolve_after in the future (and made recently)",
      pr._is_due(not_due_future, now) is False)
check("due: future resolve_after but pending long enough (maturity override)",
      pr._is_due(due_future_but_mature, now) is True)
check("due: no date but made long ago", pr._is_due(due_no_date_old, now) is True)
check("not due: no date, made recently", pr._is_due(not_due_no_date_recent, now) is False)
check("not due: re-checked within the recheck window", pr._is_due(recently_checked, now) is False)
check("not due: already resolved", pr._is_due(resolved_already, now) is False)


# --- Test 7: resolve_due applies a verdict, caps checks, leaves unresolved -
def _resolver_correct(rec, today):
    return {"resolved": True, "correct": False,
            "actual": "The Knicks won the series."}


def _resolver_unsure(rec, today):
    return {"resolved": False, "correct": None, "actual": ""}


with tempfile.TemporaryDirectory() as tmp:
    with _env(LOCALAPPDATA=tmp):
        pr._save({"last_mined_at": None, "predictions": [
            {"id": "p1", "claim": "Jarvis predicted the Spurs in six",
             "subject": "NBA Finals", "made_at": _iso(20),
             "resolve_after": "2026-06-01", "status": "pending",
             "correct": None, "actual": None, "resolved_at": None,
             "last_checked_at": None, "surfaced": False},
        ]})
        r = pr.resolve_due(resolver=_resolver_correct)
        rec = pr._load()["predictions"][0]
    check("resolve_due marks a confident verdict resolved",
          r == 1 and rec["status"] == "resolved" and rec["correct"] is False
          and "Knicks" in rec["actual"] and rec["surfaced"] is False)

with tempfile.TemporaryDirectory() as tmp:
    with _env(LOCALAPPDATA=tmp):
        pr._save({"last_mined_at": None, "predictions": [
            {"id": "p2", "claim": "Jarvis predicted X", "subject": "s",
             "made_at": _iso(20), "resolve_after": "2026-06-01",
             "status": "pending", "correct": None, "actual": None,
             "resolved_at": None, "last_checked_at": None, "surfaced": False},
        ]})
        r = pr.resolve_due(resolver=_resolver_unsure)
        rec = pr._load()["predictions"][0]
    check("resolve_due leaves an unresolved verdict pending + stamps last_checked",
          r == 0 and rec["status"] == "pending" and rec["last_checked_at"] is not None)


# --- Test 8: resolve_due respects max_checks ------------------------------
calls = {"n": 0}


def _counting_resolver(rec, today):
    calls["n"] += 1
    return {"resolved": False, "correct": None, "actual": ""}


with tempfile.TemporaryDirectory() as tmp:
    with _env(LOCALAPPDATA=tmp):
        pr._save({"last_mined_at": None, "predictions": [
            {"id": f"d{i}", "claim": f"Jarvis predicted thing {i}", "subject": "s",
             "made_at": _iso(20), "resolve_after": "2026-06-01", "status": "pending",
             "correct": None, "actual": None, "resolved_at": None,
             "last_checked_at": None, "surfaced": False}
            for i in range(5)
        ]})
        pr.resolve_due(resolver=_counting_resolver, max_checks=2)
    check("resolve_due honors max_checks", calls["n"] == 2)


# --- Test 9: take_unsurfaced returns once, then is empty + caps ------------
with tempfile.TemporaryDirectory() as tmp:
    with _env(LOCALAPPDATA=tmp):
        pr._save({"last_mined_at": None, "predictions": [
            {"id": "r1", "claim": "c1", "status": "resolved", "correct": True,
             "actual": "a1", "surfaced": False},
            {"id": "r2", "claim": "c2", "status": "resolved", "correct": False,
             "actual": "a2", "surfaced": False},
            {"id": "p3", "claim": "c3", "status": "pending", "surfaced": False},
        ]})
        first = pr.take_unsurfaced()
        second = pr.take_unsurfaced()
    check("take_unsurfaced returns the resolved ones (not pending)",
          {r["id"] for r in first} == {"r1", "r2"})
    check("take_unsurfaced is empty on the second call (marked surfaced)",
          second == [])


# --- Test 10: format_followups phrasing -----------------------------------
txt = pr.format_followups([
    {"claim": "Jarvis predicted the Spurs would win", "actual": "The Knicks won.",
     "correct": False},
    {"claim": "Jarvis predicted the Chiefs would make the playoffs",
     "actual": "They clinched.", "correct": True},
    {"claim": "Jarvis predicted a draw", "actual": "It ended 1-1.", "correct": None},
])
check("format: wrong prediction phrased 'got that one wrong'",
      "got that one wrong" in txt and "Knicks won" in txt)
check("format: right prediction phrased 'called that one'",
      "called that one" in txt)
check("format: null-correct prediction states the result without a verdict",
      "1-1" in txt)
check("format: empty list -> empty string", pr.format_followups([]) == "")


# --- Test 11: briefing_followups gating -----------------------------------
with tempfile.TemporaryDirectory() as tmp:
    with _env(LOCALAPPDATA=tmp, JARVIS_PREDICTION_FOLLOWUPS="0", ANTHROPIC_API_KEY="k"):
        check("briefing_followups disabled via env -> ''",
              pr.briefing_followups() == "")
    with _env(LOCALAPPDATA=tmp, JARVIS_PREDICTION_FOLLOWUPS="1", ANTHROPIC_API_KEY=None):
        check("briefing_followups with no API key -> ''",
              pr.briefing_followups() == "")


# --- Test 12: briefing_followups surfaces pre-resolved (run_cycle=False) ---
with tempfile.TemporaryDirectory() as tmp:
    with _env(LOCALAPPDATA=tmp, JARVIS_PREDICTION_FOLLOWUPS="1", ANTHROPIC_API_KEY="k"):
        pr._save({"last_mined_at": None, "predictions": [
            {"id": "z1", "claim": "Jarvis predicted the Spurs would win",
             "status": "resolved", "correct": False, "actual": "The Knicks won.",
             "surfaced": False},
        ]})
        out = pr.briefing_followups(run_cycle=False)
        out2 = pr.briefing_followups(run_cycle=False)
    check("briefing_followups surfaces a resolved prediction (no network)",
          "Knicks won" in out and "Following up" in out)
    check("briefing_followups doesn't repeat a surfaced follow-up", out2 == "")


# --- Test 13: _extract_json robustness ------------------------------------
check("_extract_json parses an array", pr._extract_json('noise [1,2,3] tail') == [1, 2, 3])
check("_extract_json parses an object",
      pr._extract_json('x {"a": 1} y') == {"a": 1})
check("_extract_json on junk -> None", pr._extract_json("no json here") is None)


# --- Test 14: run_cycle=True surfaces instantly + mines in the BACKGROUND ---
# 2026-06-21: a voice "good morning" runs the briefing inline in the agentic
# loop. mine+resolve used to run there (up to ~2 min on the network). Now the
# briefing surfaces already-resolved predictions immediately and kicks the cycle
# onto a guarded daemon thread. Lock in: instant surface, the cycle runs once,
# and a second cycle while one's in flight is a guarded no-op.
import threading as _threading  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    with _env(LOCALAPPDATA=tmp, JARVIS_PREDICTION_FOLLOWUPS="1", ANTHROPIC_API_KEY="k"):
        pr._save({"last_mined_at": None, "predictions": [
            {"id": "bg1", "claim": "Jarvis predicted the Heat would win",
             "status": "resolved", "correct": True, "actual": "The Heat won.",
             "surfaced": False},
        ]})
        calls = {"mine": 0, "resolve": 0}
        done = _threading.Event()
        release = _threading.Event()

        def _stub_mine(*a, **k):
            calls["mine"] += 1
            release.wait(3.0)   # hold the cycle so the guard can be tested
            return 0

        def _stub_resolve(*a, **k):
            calls["resolve"] += 1
            done.set()
            return 0

        _orig_mine, _orig_resolve = pr.mine_predictions, pr.resolve_due
        pr.mine_predictions, pr.resolve_due = _stub_mine, _stub_resolve
        try:
            out = pr.briefing_followups(run_cycle=True)
            check("run_cycle=True surfaces the resolved prediction immediately",
                  "Heat won" in out)
            check("a second cycle while one is in flight is a guarded no-op",
                  pr.run_prediction_cycle_async() is False)
            release.set()  # let the in-flight cycle finish
            check("background cycle ran (mine then resolve)", done.wait(3.0))
            check("mine ran exactly once", calls["mine"] == 1)
            check("resolve ran exactly once", calls["resolve"] == 1)
            # The lock must be free again once the cycle finishes.
            check("cycle lock released after completion",
                  pr._CYCLE_LOCK.acquire(blocking=False))
            pr._CYCLE_LOCK.release()
        finally:
            release.set()
            pr.mine_predictions, pr.resolve_due = _orig_mine, _orig_resolve


# --- summary --------------------------------------------------------------
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
