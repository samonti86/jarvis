r"""Regression test for the transient-TLS retry policy (M94).

WHY THIS EXISTS:
The single most frequent fault in 30 days of real logs was a TLS handshake
timeout on the calendar feed — 47 occurrences across 16 separate sessions, of
which 34 exhausted the retry rather than recovering. A retry had already been
added for it once (2026-07-02) and was not enough, because it was calibrated
for the wrong failure shape: attempt 1 burns the FULL timeout before a flat
0.5s backoff, so attempt 2 lands inside the same blip.

The two things worth locking down are therefore:

  1. Backoff must GROW. A flat gap re-attempts into the same outage.
  2. The interactive default must NOT change. Every http_util caller is a voice
     tool with a user waiting in silence; making them more patient trades this
     project's first constraint (latency) for a background problem's benefit.

    venv\Scripts\python.exe tests\retry_policy_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from src import http_util  # noqa: E402
from src import outlook_calendar as oc  # noqa: E402

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


# --- instrument: record attempts and sleeps without actually waiting -------
calls: list[float] = []
sleeps: list[float] = []


def always_timeout(url, **kwargs):
    calls.append(time.monotonic())
    raise httpx.ConnectTimeout("handshake operation timed out")


_real_get, _real_sleep = httpx.get, time.sleep
httpx.get = always_timeout
time.sleep = lambda s: sleeps.append(s)
http_util.httpx.get = always_timeout
http_util.time.sleep = lambda s: sleeps.append(s)
oc.httpx.get = always_timeout
oc.time.sleep = lambda s: sleeps.append(s)

# =========================================================================
# 1. the interactive default is UNCHANGED — this is the safety property
# =========================================================================
calls.clear(); sleeps.clear()
http_util.http_get_with_retry("https://example.invalid", tag="test")
check("interactive default is still 2 attempts (one retry)", len(calls) == 2,
      f"made {len(calls)} attempts")
check("interactive default keeps the 0.5s first backoff", sleeps == [0.5], str(sleeps))

# =========================================================================
# 2. backoff GROWS — the actual defect
# =========================================================================
calls.clear(); sleeps.clear()
http_util.http_get_with_retry("https://example.invalid", tag="test", attempts=4)
check("attempts=4 makes 4 attempts", len(calls) == 4, f"made {len(calls)}")
check("backoff is exponential, not flat", sleeps == [0.5, 1.0, 2.0], str(sleeps))
check("each gap is strictly larger than the last",
      all(b > a for a, b in zip(sleeps, sleeps[1:])), str(sleeps))

# =========================================================================
# 3. a 4xx/5xx is still never retried — retrying can't fix a 404
# =========================================================================
def http_500(url, **kwargs):
    calls.append(time.monotonic())
    resp = httpx.Response(500, request=httpx.Request("GET", url))
    raise httpx.HTTPStatusError("boom", request=resp.request, response=resp)


http_util.httpx.get = http_500
calls.clear(); sleeps.clear()
http_util.http_get_with_retry("https://example.invalid", tag="test", attempts=4)
check("an HTTP 5xx is not retried even with attempts=4", len(calls) == 1,
      f"made {len(calls)} attempts")
check("and it does not sleep", sleeps == [], str(sleeps))
http_util.httpx.get = always_timeout

# =========================================================================
# 4. the calendar path — the fault this milestone was chasing
# =========================================================================
oc.ICAL_URL = "https://example.invalid/cal.ics"
calls.clear(); sleeps.clear()
events, err = oc._fetch_events_ical(None, None)
check("calendar now makes 3 attempts (was 2)", len(calls) == 3, f"made {len(calls)}")
check("calendar backoff grows: 1s then 2s", sleeps == [1.0, 2.0], str(sleeps))
check("calendar still fails soft with a spoken message",
      events is None and "couldn't reach" in err.lower(), f"{events!r} {err!r}")

# The whole justification for the change: MORE chances in LESS worst-case time.
before_worst = 15.0 + 0.5 + 15.0
after_worst = oc._FETCH_TIMEOUT_SEC * 3 + sum(sleeps)
check(f"worst case improved ({before_worst}s -> {after_worst}s) despite an extra attempt",
      after_worst < before_worst, f"{after_worst} vs {before_worst}")
check("per-attempt timeout is short enough to afford 3 tries",
      oc._FETCH_TIMEOUT_SEC <= 8.0, str(oc._FETCH_TIMEOUT_SEC))

httpx.get, time.sleep = _real_get, _real_sleep
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
