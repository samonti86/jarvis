"""Shared HTTP helper for the client-side data tools.

Extracted in M43. By M22 the same ~30-LOC `_http_get_with_retry` had been
copy-pasted into src/sports.py, src/weather.py, and src/games.py, byte-
identical except for the `[tool]` log prefix. Three copies of one helper is
the classic rule-of-three signal that the cost-benefit has flipped toward
extraction; the M22 milestone note explicitly flagged "pull into
src/http_util.py with a per-tool log prefix arg on the next tool that needs
it." M43's TMDB tool is that next tool, so the refactor rides in with it.

The contract is deliberately unchanged from the original copies so the
extraction is behavior-preserving:

  - One retry, and ONLY on transport-level failures (timeout / DNS /
    connection drop) — those are transient and worth a second shot.
  - NEVER retry on an HTTP 4xx/5xx (`HTTPStatusError`). A 404 or a 500 will
    not fix itself on a retry; retrying just wastes the user's time on what
    is, for voice, a latency-critical path.
  - Returns the `httpx.Response` on success, or `None` on final failure /
    any non-2xx. Callers branch on `None` — they never see an exception.

Each caller binds its own log prefix once with `functools.partial`:

    from functools import partial
    from src.http_util import http_get_with_retry
    _http_get_with_retry = partial(http_get_with_retry, tag="games")

That keeps every existing call site (`_http_get_with_retry(url, params=...)`)
untouched — partial application pre-binds `tag`, yielding a callable with the
original signature. Smallest possible diff over working code.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import httpx


# Same defensive HTTP policy the three original copies shared: a generous
# 10s timeout (these are public APIs that occasionally lag) and a single
# 0.5s-backoff retry, transport errors only. Exposed as keyword args with
# these defaults so a caller can override per-tool if one upstream proves
# slower, without changing the shared default everyone else relies on.
_HTTP_TIMEOUT_SEC = 10.0
_RETRY_BACKOFF_SEC = 0.5
# One try plus one retry. Unchanged default: every current caller here is an
# interactive voice tool, and a user waiting in silence is the worse failure.
_DEFAULT_ATTEMPTS = 2


def http_get_with_retry(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    tag: str,
    timeout: float = _HTTP_TIMEOUT_SEC,
    backoff: float = _RETRY_BACKOFF_SEC,
    attempts: int = _DEFAULT_ATTEMPTS,
    follow_redirects: bool = True,
) -> httpx.Response | None:
    """GET with retries on transport errors. Returns None on final failure
    or any non-2xx response. Caller checks for None.

    `attempts` defaults to 2 — one try plus one retry, exactly the historical
    behaviour — because EVERY caller of this helper is an interactive voice
    tool (weather, news, sports, tmdb, games). A user is waiting in silence
    during those, so patience is the wrong trade: latency is this project's
    first constraint. Only a BACKGROUND poller, where nobody is waiting,
    should raise it.

    `tag` is the short tool name used as the stderr log prefix (`[games]`,
    `[tmdb]`, ...). Keyword-only and required so every caller is identifiable
    in the log — a shared helper with anonymous failures would be a debugging
    regression vs. the per-tool copies it replaces.

    `follow_redirects` defaults True: httpx does NOT follow redirects by
    default, and a feed/endpoint that 301/302-redirects (e.g. an http→https
    upgrade) would otherwise raise HTTPStatusError and be silently dropped —
    the exact M47 landmine the news tool already hit once. Following redirects
    is the expected behaviour for every consumer of this helper; the
    "retry transport errors only, never 4xx/5xx" contract is unaffected.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            r = httpx.get(url, params=params, timeout=timeout,
                          follow_redirects=follow_redirects)
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as exc:
            # 4xx/5xx — won't fix itself; don't retry.
            print(f"[{tag}] HTTP {exc.response.status_code} on {url}", file=sys.stderr)
            return None
        except httpx.RequestError as exc:
            # Timeout, connect error, DNS — worth retrying.
            last_exc = exc
            if attempt >= max(attempts, 1):
                break
            # EXPONENTIAL, not flat. A flat 0.5s is calibrated for a momentary
            # hiccup; the failure this actually hits is a TLS handshake timeout,
            # where the ATTEMPT ITSELF burns the full timeout before the backoff
            # even starts. Sleeping a further 0.5s re-attempts inside the same
            # blip. Growing the gap makes each retry sample a genuinely later
            # moment. (Measured: 44 distinct outage bursts over 35 days, the
            # largest 10 events across 1.6 minutes.)
            wait = backoff * (2 ** (attempt - 1))
            print(
                f"[{tag}] {type(exc).__name__} on {url} — retrying in "
                f"{wait}s (attempt {attempt + 1}/{attempts})",
                file=sys.stderr,
            )
            time.sleep(wait)
    print(f"[{tag}] gave up after {attempts} attempts: {last_exc}", file=sys.stderr)
    return None
