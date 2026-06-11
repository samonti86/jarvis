"""Prediction follow-ups — "you called it" callbacks (M78.2).

Jarvis notices when a SPORTS prediction he made earlier has been decided, and
brings it up in the morning briefing: "Following up, sir — I picked the Spurs
for the Finals; the Knicks took it. I got that one wrong."

The motivating moment: the user remembered Jarvis predicting the Spurs would win
the NBA Finals, but Jarvis had no record of it (the recall gap fixed in C/M78.1).
This closes the loop the other way — not just answering "what did you predict?"
when asked, but proactively circling back once the outcome is known.

DESIGN (settled with the user before the build):
  - CAPTURE = retrospective MINING, not a capture-time tool. A periodic pass
    re-reads recent transcripts (reusing conversation_recall._collect_exchanges)
    and an LLM extracts the falsifiable sports predictions Jarvis ASSERTED as
    his own pick. Robust (catches every prediction, not just ones Claude
    remembered to log), zero per-turn cost, and it BACKFILLS — predictions made
    before this feature existed are picked up too.
  - SURFACE = the morning briefing. Predictions resolve over days, so a daily
    follow-up line is the right cadence — proactive without the embarrassment
    risk of a real-time interruption that fires on a misread outcome.
  - SCOPE = sports only (v1). Sports outcomes are cleanly resolvable (binary,
    dated, checkable via web search). Broader/fuzzier domains are a later step
    once this is proven — conservatism keeps wrong "you called it"s rare.

RESOLUTION is the hard, conservative part: a prediction is only marked resolved
when a web-search-backed check is CONFIDENT the event is decided. Anything
ambiguous stays pending and is re-checked later (bounded). A wrong callback is
worse than a late one.

Both LLM steps (miner, resolver) are dependency-injected so the orchestration
and store logic are unit-testable without the network; the default
implementations call the Anthropic API (the summarizer pattern). Defensive
contract throughout: every public entry point never raises — a failure costs the
follow-up section, never the briefing or the listen loop.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import anthropic


# --- Tuning ----------------------------------------------------------------
# First mining run with no watermark looks back this far (enough to backfill
# recent predictions like the June 3 Spurs call); subsequent runs are
# incremental from last_mined_at.
_FIRST_RUN_DAYS = 30
# A prediction with no estimated resolve date is not checked until this long
# after it was made (avoids hammering web search on a just-made prediction).
_DEFAULT_RESOLVE_DELAY_DAYS = 7
# Don't re-check the same still-pending prediction more often than this.
_RECHECK_HOURS = 12
# Max resolution web-search checks per cycle — bounds briefing latency/cost.
_MAX_RESOLVE_PER_CYCLE = 3
# Cap follow-ups surfaced in one briefing (don't read a laundry list aloud).
_MAX_FOLLOWUPS = 3


def _mining_model() -> str:
    # Cheap extraction → the summarizer's Haiku by default.
    return os.getenv(
        "JARVIS_PREDICTION_MINE_MODEL",
        os.getenv("SUMMARY_MODEL", "claude-haiku-4-5-20251001"),
    )


def _resolve_model() -> str:
    # Judgment (did it resolve, was it right?) → the main model by default.
    return os.getenv(
        "JARVIS_PREDICTION_RESOLVE_MODEL",
        os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
    )


def _api_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "").strip()


def _enabled() -> bool:
    return os.getenv("JARVIS_PREDICTION_FOLLOWUPS", "1").strip() not in ("0", "false", "False")


# --- Store -----------------------------------------------------------------
# %LOCALAPPDATA%\Jarvis\predictions.json — a dict {last_mined_at, predictions:[]}
# atomically written (temp + os.replace), fail-soft on read. Same discipline as
# reminders.py; a dict (not a bare list) so the mining watermark lives with it.

def _store_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base / "predictions.json"


def _load() -> dict:
    path = _store_path()
    if not path.exists():
        return {"last_mined_at": None, "predictions": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — a bad file must not crash anything
        print(f"[predictions] could not read {path.name}: {exc}", file=sys.stderr)
        return {"last_mined_at": None, "predictions": []}
    if not isinstance(data, dict):
        return {"last_mined_at": None, "predictions": []}
    preds = data.get("predictions")
    if not isinstance(preds, list):
        preds = []
    return {
        "last_mined_at": data.get("last_mined_at"),
        "predictions": [p for p in preds if isinstance(p, dict)],
    }


def _save(store: dict) -> None:
    path = _store_path()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _stable_id(claim: str, made_at: str) -> str:
    """Stable id from the normalized claim + the day it was made, so the same
    prediction isn't re-added across mining runs. Day-granularity (not full ts)
    tolerates minor ts drift between the user and assistant lines of a turn."""
    import hashlib  # noqa: PLC0415 — stdlib, only needed here
    norm = re.sub(r"[^a-z0-9]+", "", (claim or "").lower())[:120]
    day = (made_at or "")[:10]
    return hashlib.sha1(f"{day}\x00{norm}".encode("utf-8")).hexdigest()[:16]


# --- Mining ----------------------------------------------------------------

_MINER_SYSTEM = """You extract FALSIFIABLE SPORTS PREDICTIONS that the assistant (named Jarvis) made about FUTURE outcomes, from a transcript of past voice conversations.

Include ONLY a prediction that JARVIS HIMSELF asserted as his own pick or lean about a future sports result — e.g. "I'd lean Spurs", "my pick is the Chiefs", "they'll win in six".

DO NOT include:
- Anything the USER said or predicted.
- Quoted odds or consensus Jarvis merely reported ("Vegas favors X", "analysts predict Y") — that is not HIS prediction.
- Non-commitments or hedges with no pick ("it could go either way", "hard to say").
- Anything that is not SPORTS.
- Past results or facts (only forward-looking predictions of an undecided outcome).

For each real prediction, output an object with:
  "claim": a self-contained sentence in the form "Jarvis predicted <the outcome>" (e.g. "Jarvis predicted the Spurs would win the NBA Finals in six games").
  "subject": a short topic label (e.g. "NBA Finals: Spurs vs. Knicks").
  "made_at": the date the prediction was made, as YYYY-MM-DD, taken from the [timestamp] on Jarvis's line where he made it.
  "resolve_after": your best estimate of the date the outcome will be decided, as YYYY-MM-DD, or null if you cannot estimate one.

Respond with ONLY a JSON array of these objects. If there are no qualifying predictions, respond with exactly []."""


def _extract_json(text: str):
    """Pull the first JSON array/object out of an LLM reply. None on failure."""
    if not text:
        return None
    # Prefer a fenced or bare array; fall back to an object.
    for pat in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except (ValueError, TypeError):
                continue
    return None


def _default_miner(transcript: str) -> list[dict]:
    """Extract predictions via Haiku. Returns [] on any failure."""
    key = _api_key()
    if not key:
        return []
    try:
        client = anthropic.Anthropic(api_key=key, max_retries=1, timeout=20.0)
        msg = client.messages.create(
            model=_mining_model(),
            max_tokens=600,
            system=_MINER_SYSTEM,
            messages=[{"role": "user", "content":
                       f"TRANSCRIPT:\n{transcript}\nEND TRANSCRIPT\n\n"
                       "Output the JSON array now."}],
        )
        text = " ".join(b.text for b in msg.content if hasattr(b, "text"))
        data = _extract_json(text)
        return data if isinstance(data, list) else []
    except Exception as exc:  # noqa: BLE001
        print(f"[predictions] mining call failed: {exc}", file=sys.stderr)
        return []


def _normalize_mined(p: dict, made_at: str) -> dict | None:
    """Validate one mined prediction into a store record. None if unusable.
    `made_at` is the batch fallback; the miner's own per-prediction date (read
    from the transcript timestamp) wins when it's a plausible YYYY-MM-DD."""
    if not isinstance(p, dict):
        return None
    claim = str(p.get("claim", "")).strip()
    if len(claim) < 8:
        return None
    subject = str(p.get("subject", "")).strip() or "a sports outcome"
    mined_made = str(p.get("made_at", "")).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", mined_made):
        made_at = mined_made
    resolve_after = p.get("resolve_after")
    if resolve_after is not None:
        ra = str(resolve_after).strip()
        # Keep only a plausible YYYY-MM-DD; anything else → null (use default delay).
        resolve_after = ra if re.fullmatch(r"\d{4}-\d{2}-\d{2}", ra) else None
    return {
        "id": _stable_id(claim, made_at),
        "claim": claim,
        "subject": subject,
        "made_at": made_at,
        "resolve_after": resolve_after,
        "status": "pending",
        "correct": None,
        "actual": None,
        "resolved_at": None,
        "last_checked_at": None,
        "surfaced": False,
    }


def _recent_exchanges(last_mined: str | None, now: datetime):
    """Exchanges newer than the mining watermark (or the first-run window).
    Reuses conversation_recall's scan + pairing — predictions live ON TOP of
    the same transcript store recall searches."""
    from src.conversation_recall import _collect_exchanges  # noqa: PLC0415
    if last_mined:
        try:
            since = datetime.fromisoformat(last_mined)
        except (ValueError, TypeError):
            since = now - timedelta(days=_FIRST_RUN_DAYS)
        days_back = max(1, (now - since).days + 1)
    else:
        since = now - timedelta(days=_FIRST_RUN_DAYS)
        days_back = _FIRST_RUN_DAYS
    out = []
    for ex in _collect_exchanges(days_back):
        ts = ex.get("ts", "")
        try:
            if datetime.fromisoformat(ts) > since:
                out.append(ex)
        except (ValueError, TypeError):
            out.append(ex)  # undated — include rather than silently drop
    return out


def _transcript(exchanges: list[dict]) -> str:
    return "\n".join(
        f"  [{ex.get('ts','')}] user: {ex.get('user','')}\n"
        f"  [{ex.get('ts','')}] Jarvis: {ex.get('assistant','')}"
        for ex in exchanges
    )


def mine_predictions(*, miner=None, now: datetime | None = None) -> int:
    """Scan transcripts since the last watermark, extract new sports predictions
    into the store. Returns the number added. Never raises."""
    now = now or datetime.now()
    store = _load()
    try:
        exchanges = _recent_exchanges(store.get("last_mined_at"), now)
    except Exception as exc:  # noqa: BLE001
        print(f"[predictions] could not read transcripts: {exc}", file=sys.stderr)
        exchanges = []
    if not exchanges:
        store["last_mined_at"] = now.isoformat(timespec="seconds")
        _save(store)
        return 0

    miner = miner or _default_miner
    try:
        found = miner(_transcript(exchanges)) or []
    except Exception as exc:  # noqa: BLE001
        print(f"[predictions] miner raised: {exc}", file=sys.stderr)
        found = []

    existing = {p.get("id") for p in store["predictions"]}
    added = 0
    # Earliest exchange ts is a reasonable made_at when the miner can't tell us;
    # but we want each prediction stamped near when it was said. The miner
    # returns no ts, so we approximate made_at with the most recent exchange in
    # the batch (predictions are usually the latest thing discussed). Good
    # enough for the day-granularity stable id and the resolve-delay clock.
    made_at = max((ex.get("ts", "") for ex in exchanges), default=now.isoformat(timespec="seconds")) \
        or now.isoformat(timespec="seconds")
    for p in found:
        rec = _normalize_mined(p, made_at)
        if rec and rec["id"] not in existing:
            store["predictions"].append(rec)
            existing.add(rec["id"])
            added += 1

    store["last_mined_at"] = now.isoformat(timespec="seconds")
    _save(store)
    if added:
        print(f"[predictions] mined {added} new prediction(s)", file=sys.stderr)
    return added


# --- Resolution ------------------------------------------------------------

_WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

_RESOLVER_SYSTEM = """You verify whether a previously-made SPORTS prediction has been DECIDED yet, using web search. Be conservative: only report it resolved if you are CONFIDENT the outcome is settled. If it is not yet decided, or you cannot confirm the result, report it unresolved.

A prediction is DECIDED not only when the event is fully over, but also when its SPECIFIC claim can no longer come true — for example a "win in N games" series prediction once the team trails by enough that winning in exactly N games is mathematically impossible, or a team that has already been eliminated. In that case it is resolved and the prediction was incorrect. (Conversely, do not call a prediction correct until it is actually clinched.)

Respond with ONLY compact JSON:
{"resolved": true|false, "correct": true|false|null, "actual": "<one short sentence on what actually happened, or empty if unresolved>"}

"correct" is whether the ORIGINAL prediction turned out right; null if unresolved."""


def _default_resolver(rec: dict, today: str) -> dict | None:
    """Resolve one prediction via a web-search-backed Anthropic call. Returns
    {resolved, correct, actual} or None on failure (caller leaves it pending)."""
    key = _api_key()
    if not key:
        return None
    prompt = (
        f"A prediction was made on {rec.get('made_at','')[:10]}: "
        f"\"{rec.get('claim','')}\" (topic: {rec.get('subject','')}). "
        f"Today is {today}. Has this been decided yet? "
        f"Output the JSON verdict only."
    )
    try:
        client = anthropic.Anthropic(api_key=key, max_retries=1, timeout=40.0)
        msg = client.messages.create(
            model=_resolve_model(),
            max_tokens=500,
            system=_RESOLVER_SYSTEM,
            tools=[_WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": prompt}],
        )
        text = " ".join(b.text for b in msg.content if hasattr(b, "text"))
        data = _extract_json(text)
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001
        print(f"[predictions] resolver call failed: {exc}", file=sys.stderr)
        return None


def _is_due(rec: dict, now: datetime) -> bool:
    """Should we attempt to resolve this pending prediction now? Due when its
    estimated resolve date has arrived (or, lacking one, enough time has passed
    since it was made), AND it hasn't been re-checked too recently."""
    if rec.get("status") != "pending":
        return False
    last = rec.get("last_checked_at")
    if last:
        try:
            if now - datetime.fromisoformat(last) < timedelta(hours=_RECHECK_HOURS):
                return False
        except (ValueError, TypeError):
            pass
    # Due once the estimated resolve date has arrived...
    ra = rec.get("resolve_after")
    if ra:
        try:
            if now.date() >= datetime.fromisoformat(ra).date():
                return True
        except (ValueError, TypeError):
            pass
    # ...OR once it's been pending long enough since it was made. The estimate
    # is just the miner's guess — outcomes resolve EARLY (a sweep, an
    # elimination, a prediction made mathematically impossible), so a mature
    # prediction gets periodic checks regardless of a far-future estimate. The
    # resolver stays conservative and returns UNRESOLVED if it truly isn't
    # decided, so an over-eager check just costs one throttled search.
    made = rec.get("made_at")
    if made:
        try:
            return now - datetime.fromisoformat(made) >= timedelta(days=_DEFAULT_RESOLVE_DELAY_DAYS)
        except (ValueError, TypeError):
            return True
    return True


def resolve_due(*, resolver=None, now: datetime | None = None,
                max_checks: int = _MAX_RESOLVE_PER_CYCLE) -> int:
    """Attempt to resolve due pending predictions (bounded). Returns the number
    newly resolved. Never raises."""
    now = now or datetime.now()
    store = _load()
    resolver = resolver or _default_resolver
    checks = 0
    resolved = 0
    dirty = False
    for rec in store["predictions"]:
        if not _is_due(rec, now):
            continue
        if checks >= max_checks:
            break
        checks += 1
        rec["last_checked_at"] = now.isoformat(timespec="seconds")
        dirty = True
        try:
            verdict = resolver(rec, now.date().isoformat())
        except Exception as exc:  # noqa: BLE001
            print(f"[predictions] resolver raised: {exc}", file=sys.stderr)
            continue
        if isinstance(verdict, dict) and verdict.get("resolved"):
            rec["status"] = "resolved"
            rec["correct"] = verdict.get("correct")
            rec["actual"] = str(verdict.get("actual", "")).strip()
            rec["resolved_at"] = now.isoformat(timespec="seconds")
            rec["surfaced"] = False
            resolved += 1
    if dirty:
        _save(store)
    if resolved:
        print(f"[predictions] resolved {resolved} prediction(s)", file=sys.stderr)
    return resolved


# --- Surfacing -------------------------------------------------------------

def take_unsurfaced() -> list[dict]:
    """Resolved-but-not-yet-surfaced predictions; marks them surfaced so a
    follow-up is read at most once. Never raises."""
    store = _load()
    out = [
        r for r in store["predictions"]
        if r.get("status") == "resolved" and not r.get("surfaced")
    ]
    if not out:
        return []
    out = out[:_MAX_FOLLOWUPS]
    surfaced_ids = {r["id"] for r in out}
    for r in store["predictions"]:
        if r.get("id") in surfaced_ids:
            r["surfaced"] = True
    _save(store)
    return out


def format_followups(recs: list[dict]) -> str:
    """Render resolved predictions as a spoken follow-up section. '' if none."""
    if not recs:
        return ""
    lines = ["Following up on earlier predictions, sir:"]
    for r in recs:
        claim = (r.get("claim") or "").rstrip(".")
        actual = (r.get("actual") or "").strip()
        correct = r.get("correct")
        if correct is True:
            verdict = "I called that one."
        elif correct is False:
            verdict = "I got that one wrong."
        else:
            verdict = ""
        piece = f"- {claim}."
        if actual:
            piece += f" {actual}"
        if verdict:
            piece += f" {verdict}"
        lines.append(piece)
    return "\n".join(lines)


def briefing_followups(*, run_cycle: bool = True) -> str:
    """The morning-briefing entry point: optionally run a mine+resolve cycle,
    then return the follow-up text for any newly-resolved predictions (marking
    them surfaced). Returns '' when disabled, unconfigured, or nothing's ready.
    Never raises — a failure costs this section, not the briefing."""
    if not _enabled() or not _api_key():
        return ""
    try:
        if run_cycle:
            mine_predictions()
            resolve_due()
        return format_followups(take_unsurfaced())
    except Exception as exc:  # noqa: BLE001
        print(f"[predictions] briefing_followups failed: {exc}", file=sys.stderr)
        return ""
