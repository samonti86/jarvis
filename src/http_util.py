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


def http_get_with_retry(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    tag: str,
    timeout: float = _HTTP_TIMEOUT_SEC,
    backoff: float = _RETRY_BACKOFF_SEC,
) -> httpx.Response | None:
    """GET with one retry on transport errors. Returns None on final failure
    or any non-2xx response. Caller checks for None.

    `tag` is the short tool name used as the stderr log prefix (`[games]`,
    `[tmdb]`, ...). Keyword-only and required so every caller is identifiable
    in the log — a shared helper with anonymous failures would be a debugging
    regression vs. the per-tool copies it replaces.
    """
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            r = httpx.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as exc:
            # 4xx/5xx — won't fix itself; don't retry.
            print(f"[{tag}] HTTP {exc.response.status_code} on {url}", file=sys.stderr)
            return None
        except httpx.RequestError as exc:
            # Timeout, connect error, DNS — worth one retry.
            last_exc = exc
            if attempt == 1:
                print(
                    f"[{tag}] {type(exc).__name__} on {url} — retrying in "
                    f"{backoff}s",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                continue
            break
    print(f"[{tag}] gave up after retry: {last_exc}", file=sys.stderr)
    return None
