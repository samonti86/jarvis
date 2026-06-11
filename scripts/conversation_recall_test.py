"""C / M78 — regression test for src/conversation_recall.py.

Hermetic: each test points LOCALAPPDATA at a temp dir and writes synthetic
session transcripts there (the module reads default_base_dir() at call time,
the same pattern scripts/status_report_test.py uses for the log path). No
network, no real Jarvis state.

Covers: the schema; the headline Spurs-prediction recall scenario (the bug
this milestone exists to fix); keyword + prefix-stemming matching; the
days_back window; the limit cap; the score-then-recency ranking; the empty /
no-match / stopword-only / no-history paths; malformed-line and
attachment-content tolerance; and the never-raises contract.

    python scripts/conversation_recall_test.py     # exit 0 = pass
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

import numpy as np  # noqa: E402

from src import conversation_recall as cr  # noqa: E402
from src import embeddings as _emb  # noqa: E402

# Force embeddings OFF for the keyword tests below. Real sentence-transformers
# (if installed) would make ranking slow and nondeterministic; the semantic
# tests at the bottom swap in a controlled concept-embed via _semantic().
_emb.embed = lambda texts: None

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


def _write_session(base: Path, day: datetime, exchanges: list[tuple]) -> None:
    """exchanges = [(ts_iso, user_content, assistant_content), ...] written as
    the user-then-assistant line pairs record_turn produces."""
    sessions = base / "Jarvis" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    path = sessions / f"{day:%Y-%m-%d}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for ts, user, asst in exchanges:
            f.write(json.dumps({"ts": ts, "role": "user", "language": "en",
                                "content": user}) + "\n")
            f.write(json.dumps({"ts": ts, "role": "assistant",
                                "content": asst}) + "\n")


@contextmanager
def _localappdata(tmp: str):
    old = os.environ.get("LOCALAPPDATA")
    os.environ["LOCALAPPDATA"] = tmp
    try:
        yield
    finally:
        if old is None:
            del os.environ["LOCALAPPDATA"]
        else:
            os.environ["LOCALAPPDATA"] = old


# --- Test 1: schema is well-formed ----------------------------------------
schema = cr.RECALL_CONVERSATION_TOOL
check("RECALL_CONVERSATION_TOOL has name + description + input_schema",
      schema.get("name") == "recall_conversation"
      and "description" in schema
      and schema.get("input_schema", {}).get("type") == "object"
      and "query" in schema["input_schema"]["properties"]
      and schema["input_schema"]["required"] == ["query"])


# --- Test 2: the headline scenario — recall the Spurs prediction -----------
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    _write_session(base, datetime.now() - timedelta(days=8), [
        (_iso(8), "Who's in the NBA Finals and who do you think wins?",
         "It's Spurs vs. Knicks. If you're asking me to pick a winner — "
         "Spurs in six. Wemby gets his ring."),
    ])
    # Noise the query should NOT rank.
    _write_session(base, datetime.now() - timedelta(days=2), [
        (_iso(2), "What's the weather in Denver?", "84 and mostly clear, sir."),
    ])
    with _localappdata(tmp):
        out = cr.execute_recall_tool({"query": "what did you predict for the finals"})
    check("Spurs-prediction recall returns the actual pick",
          "Spurs in six" in out and "Finals" in out)
    check("Spurs-prediction recall excludes the unrelated weather exchange",
          "Denver" not in out)


# --- Test 3: prefix-stemming (predict<->prediction, final<->finals) -------
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    _write_session(base, datetime.now() - timedelta(days=1), [
        (_iso(1), "Make a prediction about the final game",
         "My pick is the home team."),
    ])
    with _localappdata(tmp):
        # query uses 'predict' / 'finals' — must stem-match the stored
        # 'prediction' / 'final'.
        out = cr.execute_recall_tool({"query": "predict finals"})
    check("prefix-stemming matches predict<->prediction, finals<->final",
          "home team" in out)


# --- Test 4: days_back window excludes older conversations -----------------
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    _write_session(base, datetime.now() - timedelta(days=40), [
        (_iso(40), "Tell me about the dragon project", "Old dragon details here."),
    ])
    _write_session(base, datetime.now() - timedelta(days=3), [
        (_iso(3), "Any update on the dragon project?", "Recent dragon update here."),
    ])
    with _localappdata(tmp):
        scoped = cr.execute_recall_tool({"query": "dragon", "days_back": 7})
        wide = cr.execute_recall_tool({"query": "dragon"})
    check("days_back=7 excludes the 40-day-old exchange",
          "Recent dragon" in scoped and "Old dragon" not in scoped)
    check("no days_back searches the whole retained history",
          "Recent dragon" in wide and "Old dragon" in wide)


# --- Test 5: limit caps the number of exchanges ---------------------------
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    _write_session(base, datetime.now() - timedelta(days=1), [
        (_iso(1.0), "Tell me about pizza one", "Pizza reply one."),
        (_iso(1.1), "Tell me about pizza two", "Pizza reply two."),
        (_iso(1.2), "Tell me about pizza three", "Pizza reply three."),
    ])
    with _localappdata(tmp):
        out = cr.execute_recall_tool({"query": "pizza", "limit": 2})
    check("limit=2 returns exactly two exchanges",
          out.count("You asked:") == 2)


# --- Test 6: ranking — higher score first, then recency -------------------
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    # Two single-term hits (older + newer) and one two-term hit. The two-term
    # hit must rank first (score 2); between the score-1 hits, the newer wins.
    _write_session(base, datetime.now() - timedelta(days=5), [
        (_iso(5), "Talk about apples", "Apples are great."),          # score 1, old
    ])
    _write_session(base, datetime.now() - timedelta(days=1), [
        (_iso(1.0), "Talk about apples", "More apples."),              # score 1, new
        (_iso(1.5), "Talk about apples and bananas", "Apples and bananas."),  # score 2
    ])
    with _localappdata(tmp):
        out = cr.execute_recall_tool({"query": "apples bananas"})
    lines = [ln for ln in out.splitlines() if ln.startswith("- [")]
    check("two-term hit ranks above one-term hits",
          "bananas" in lines[0].lower())
    # Among the score-1 lines, the newer ('More apples') precedes the old.
    rest = "\n".join(lines[1:])
    check("within equal score, newer exchange ranks first",
          rest.index("More apples") < rest.index("Apples are great"))


# --- Test 7: no match -> friendly message ---------------------------------
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    _write_session(base, datetime.now() - timedelta(days=1), [
        (_iso(1), "What's the weather?", "Sunny."),
    ])
    with _localappdata(tmp):
        out = cr.execute_recall_tool({"query": "quantum chromodynamics"})
    check("no match -> 'couldn't find anything' message",
          "couldn't find" in out.lower())


# --- Test 8: no sessions dir at all ---------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    with _localappdata(tmp):  # nothing written
        out = cr.execute_recall_tool({"query": "anything"})
    check("no sessions dir -> 'recorded conversation history' message",
          "recorded conversation history" in out.lower())


# --- Test 9: a query with NO usable terms -> 'couldn't make a query' -------
# A stopword-only query deliberately FALLS BACK to the bare tokens (same as
# knowledge._build_match_query) rather than rejecting — so the "no usable
# terms" path only fires when nothing alnum survives at all (punctuation).
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    _write_session(base, datetime.now() - timedelta(days=1), [
        (_iso(1), "hello", "hi"),
    ])
    with _localappdata(tmp):
        out = cr.execute_recall_tool({"query": "?!. ..."})
    check("query with no alnum tokens -> 'couldn't make a searchable query'",
          "searchable query" in out.lower())


# --- Test 10: empty query -> required-arg message -------------------------
out = cr.execute_recall_tool({"query": "   "})
check("empty query -> required message", "required" in out.lower())


# --- Test 11: malformed JSONL line is skipped, search still works ----------
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    sessions = base / "Jarvis" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    day = datetime.now() - timedelta(days=1)
    with (sessions / f"{day:%Y-%m-%d}.jsonl").open("w", encoding="utf-8") as f:
        f.write("this is not json\n")
        f.write(json.dumps({"ts": _iso(1), "role": "user",
                            "content": "tell me about robots"}) + "\n")
        f.write("{ partial json\n")
        f.write(json.dumps({"ts": _iso(1), "role": "assistant",
                            "content": "Robots are mechanical."}) + "\n")
    with _localappdata(tmp):
        out = cr.execute_recall_tool({"query": "robots"})
    check("malformed lines skipped, valid exchange still found",
          "mechanical" in out.lower())


# --- Test 12: attachment-style list content is flattened, not crashed -----
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    sessions = base / "Jarvis" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    day = datetime.now() - timedelta(days=1)
    with (sessions / f"{day:%Y-%m-%d}.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _iso(1), "role": "user", "content": [
            {"type": "text", "text": "what is this telescope"},
            {"type": "image", "source": {"data": "BASE64..."}},
        ]}) + "\n")
        f.write(json.dumps({"ts": _iso(1), "role": "assistant",
                            "content": "That's a refractor telescope."}) + "\n")
    with _localappdata(tmp):
        out = cr.execute_recall_tool({"query": "telescope"})
    check("list/attachment content flattened and matchable",
          "refractor" in out.lower() and "BASE64" not in out)


# --- Test 13: days_back <= 0 / invalid is treated as 'all' ----------------
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    _write_session(base, datetime.now() - timedelta(days=30), [
        (_iso(30), "the comet question", "A comet answer."),
    ])
    with _localappdata(tmp):
        zero = cr.execute_recall_tool({"query": "comet", "days_back": 0})
        bad = cr.execute_recall_tool({"query": "comet", "days_back": "soon"})
    check("days_back=0 searches everything (keep-forever semantics)",
          "comet answer" in zero.lower())
    check("non-numeric days_back falls back to 'all'",
          "comet answer" in bad.lower())


# --- Test 14: execute never raises on a totally broken store --------------
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    sessions = base / "Jarvis" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    # A directory named like a session file shouldn't crash the reader.
    (sessions / "2026-01-01.jsonl").mkdir()
    raised = False
    try:
        with _localappdata(tmp):
            cr.execute_recall_tool({"query": "anything at all"})
    except Exception:  # noqa: BLE001
        raised = True
    check("never raises even when a session 'file' is unreadable", not raised)


# === M78.1 semantic layer =================================================
# A deterministic concept-embed stub: words mapping to the same concept get
# identical vectors, so cosine reflects synonymy without the real model. This
# lets us prove the vector pass catches what keyword misses, and that the
# similarity floor keeps unrelated chatter out — hermetically.
_CONCEPTS = {
    "car": 0, "automobile": 0, "vehicle": 0, "sedan": 0,
    "dog": 1, "puppy": 1, "canine": 1,
    "space": 2, "cosmos": 2, "astronomy": 2,
}


def _fake_embed(texts):
    rows = []
    for t in texts:
        v = np.zeros(3, dtype=np.float32)
        for w in t.lower().split():
            w = "".join(ch for ch in w if ch.isalnum())
            if w in _CONCEPTS:
                v[_CONCEPTS[w]] += 1.0
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
        rows.append(v)
    return np.asarray(rows, dtype=np.float32)


@contextmanager
def _semantic():
    old = _emb.embed
    _emb.embed = _fake_embed
    try:
        yield
    finally:
        _emb.embed = old


# --- Test 15: semantic catch — finds a paraphrase keyword structurally misses
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    _write_session(base, datetime.now() - timedelta(days=1), [
        (_iso(1.0), "What did you think of my new automobile?",
         "A fine automobile, sir."),
        (_iso(1.1), "Tell me about the dog", "A loyal canine."),
    ])
    with _localappdata(tmp), _semantic():
        out = cr.execute_recall_tool({"query": "vehicle"})
    # 'vehicle' shares no prefix with 'automobile' (keyword miss), but the
    # concept-embed makes them identical -> the vector pass surfaces it.
    check("semantic catch: 'vehicle' finds the 'automobile' exchange",
          "automobile" in out.lower() and "canine" not in out.lower())


# --- Test 16: similarity floor — unrelated recent chatter is NOT surfaced --
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    _write_session(base, datetime.now() - timedelta(days=1), [
        (_iso(1), "Tell me about the dog", "A loyal canine."),
    ])
    with _localappdata(tmp), _semantic():
        out = cr.execute_recall_tool({"query": "automobile"})
    # query concept 0, only exchange concept 1 -> cosine 0, below the floor;
    # keyword misses too -> nothing surfaced.
    check("vector floor keeps a semantically-unrelated exchange out",
          "couldn't find" in out.lower())


# --- Test 17: embeddings unavailable -> clean keyword-only fallback --------
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    _write_session(base, datetime.now() - timedelta(days=1), [
        (_iso(1), "Tell me about robots", "Robots are mechanical."),
    ])
    # NB: NOT inside _semantic(), so embeddings.embed is the None-stub.
    with _localappdata(tmp):
        out = cr.execute_recall_tool({"query": "robots"})
    check("embeddings unavailable -> keyword-only still works",
          "mechanical" in out.lower())


# --- Test 18: RRF fusion math — an item in both lists outranks one in either
check("rrf_fuse: item present in both ranked lists wins",
      cr._rrf_fuse([5, 1], [5, 2])[0] == 5)


# --- summary --------------------------------------------------------------
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
