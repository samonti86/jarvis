r"""Measure what output_config.effort actually costs and buys on Jarvis's real path.

WHY THIS EXISTS:
Sonnet 5 defaults to effort=`high` when output_config is unset, so every "what's
the weather" was being answered at full reasoning depth. Lowering it is the
biggest latency/cost lever available without touching the prompt — but it is NOT
a safe blind change here, for a reason specific to this codebase:

  - the voice path runs `thinking: {"type": "disabled"}`, and Sonnet 5 with
    thinking off is already less inclined to reach for tools;
  - `effort` pushes the same direction — `high`/`xhigh` show materially more
    tool use than `low`;
  - Jarvis's usefulness IS its tool routing across ~40 tools.

A latency win that quietly costs tool recall is not a win, and routing is
probabilistic, so it cannot be reasoned about from the docs alone. Hence: measure.

This drives the REAL `stream_response` — real system prompt, real tool schemas,
real agentic loop — so what it measures is what ships. It is an instrument, not
a gate: it costs live API calls and needs network, so it is deliberately NOT
named `*_test.py` and never runs in CI.

    venv\Scripts\python.exe scripts\effort_probe.py
    venv\Scripts\python.exe scripts\effort_probe.py --levels low,medium,high --repeat 2
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import llm  # noqa: E402
from src.config import load  # noqa: E402

# (question, tool we EXPECT routing to pick, note)
# `None` = should answer directly; calling a tool would be over-triggering.
QUERIES: list[tuple[str, str | None]] = [
    ("What's the weather in Austin, Texas?",              "get_weather"),
    ("What's the top news today?",                        "get_news"),
    ("How long does it take to beat Elden Ring?",         "get_game_length"),
    ("Who directed the movie Dune?",                      "get_movie_tv_info"),
    ("Where can I watch Oppenheimer?",                    "get_movie_tv_info"),
    # NB: deliberately NOT "remind me to..." — set_reminder WRITES to the
    # real reminders.json. An early version of this probe used it and, over
    # 3 levels x 3 repeats x 2 runs, created 18 live "call the dentist"
    # reminders in production that all fired the same evening. A probe must
    # exercise routing WITHOUT mutating real state; list_reminders is the
    # read-only member of the same family.
    ("What reminders do I have set?",                     "list_reminders"),
    # NOTE: the schema name is `pc_diagnostics`, NOT `get_pc_diagnostics` — the
    # get_ prefix is not universal in this tool set. Getting an expected name
    # wrong silently scores a MISS on a tool that fired correctly, which would
    # push the conclusion the wrong way. Cross-check against the schemas.
    ("How much RAM is this machine using right now?",     "pc_diagnostics"),
    ("What did the Lakers do last night?",                None),   # sports OR search
    ("What's 17 percent of 4320?",                        None),   # may answer directly
    ("Thanks, that's all for now.",                       None),   # pure conversational
]


def run_one(api_key: str, question: str, effort: str | None) -> dict:
    """One full turn. Returns telemetry + the reply text."""
    box: dict = {}

    def on_complete(rec) -> None:
        box["rec"] = rec

    t0 = time.monotonic()
    first_token_at = None
    chunks: list[str] = []
    try:
        for piece in llm.stream_response(
            api_key=api_key,
            messages=[{"role": "user", "content": question}],
            model="claude-sonnet-5",
            on_complete=on_complete,
            effort=effort,
        ):
            if first_token_at is None:
                first_token_at = time.monotonic() - t0
            chunks.append(piece)
    except Exception as exc:  # a probe must report, not explode
        return {"error": f"{type(exc).__name__}: {exc}"[:160]}

    rec = box.get("rec")
    text = "".join(chunks)
    return {
        "ttft": first_token_at if first_token_at is not None else float("nan"),
        "elapsed": rec.elapsed_sec if rec else float("nan"),
        "in_tok": rec.input_tokens if rec else 0,
        "out_tok": rec.output_tokens if rec else 0,
        "cache_read": rec.cache_read_tokens if rec else 0,
        "iters": rec.iterations if rec else 0,
        "tools": list(rec.tools_used) if rec else [],
        "words": len(text.split()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="low,medium,high")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    cfg = load()
    if not cfg.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    n_calls = len(levels) * len(QUERIES) * args.repeat
    print(f"effort probe — {len(QUERIES)} queries x {len(levels)} levels "
          f"x {args.repeat} = {n_calls} live API calls\n")

    results: dict[str, list[dict]] = {lv: [] for lv in levels}

    # INTERLEAVED, and the level order ROTATES per query.
    #
    # The first version of this probe swept level-by-level: every `low` call,
    # then every `medium`, then every `high`. That confounds effort with
    # wall-clock time — if API latency drifts during the run, whichever level
    # occupied the calm stretch looks fastest, and re-running reproduces the
    # artefact exactly because the order is the same. It made `medium` look 32%
    # faster than `high` twice in a row, which is precisely how a measurement
    # error survives replication.
    #
    # Interleaving puts all levels in the same slice of time; rotating the order
    # per query stops any single level always going first (and paying, or
    # avoiding, a cold-cache penalty).
    for qi, (question, expected) in enumerate(QUERIES):
        print(f"=== {question}")
        for rep in range(args.repeat):
            order = levels[(qi + rep) % len(levels):] + levels[:(qi + rep) % len(levels)]
            for lv in order:
                r = run_one(cfg.anthropic_api_key, question, lv)
                r["q"] = question
                r["expected"] = expected
                results[lv].append(r)
                if "error" in r:
                    print(f"  ERR  {lv:7s} {r['error']}")
                    continue
                hit = "-" if expected is None else ("HIT " if expected in r["tools"] else "MISS")
                print(f"  {hit} {lv:7s} {r['elapsed']:5.1f}s  ttft={r['ttft']:4.1f}s  "
                      f"out={r['out_tok']:4d}  words={r['words']:3d}  "
                      f"tools={','.join(r['tools']) or '(none)'}")
        print()

    # --- comparison ------------------------------------------------------
    print("=" * 78)
    print(f"{'effort':8s} {'med lat':>8s} {'med TTFT':>9s} {'out tok':>8s} "
          f"{'words':>6s} {'iters':>6s} {'routing':>9s}")
    print("-" * 78)
    for lv in levels:
        ok = [r for r in results[lv] if "error" not in r]
        if not ok:
            print(f"{lv:8s}  all calls failed")
            continue
        graded = [r for r in ok if r["expected"] is not None]
        hits = sum(1 for r in graded if r["expected"] in r["tools"])
        print(f"{lv:8s} "
              f"{statistics.median(r['elapsed'] for r in ok):7.1f}s "
              f"{statistics.median(r['ttft'] for r in ok):8.1f}s "
              f"{statistics.mean(r['out_tok'] for r in ok):8.0f} "
              f"{statistics.mean(r['words'] for r in ok):6.0f} "
              f"{statistics.mean(r['iters'] for r in ok):6.1f} "
              f"{hits}/{len(graded):>2d}".rjust(9))
    print("=" * 78)
    print("routing = expected tool was actually called. Treat a drop there as\n"
          "disqualifying, however good the latency looks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
