r"""Audit 3.1 — armed+acoustic resident-budget probe.

WHY THIS EXISTS
---------------
With security armed (and acoustic awareness coupled on, per M58), the Jarvis
process holds FOUR ML runtimes at once: openWakeWord (main listen) + a second
openWakeWord (the M52 barge-in singleton) + PANNs Cnn14 (acoustic) + YOLOv8n
(security). The M67 stutter-gate + the M44 watchdog keep this *managed*, but
the whole-process steady-state has never actually been profiled — the audit
(docs/CODE_AUDIT.md, finding 3.1) flagged "measure, don't guess" (the M44
lesson, where an UNmeasured cause was carried for two milestones).

This probe ATTACHES to the already-running Jarvis process (it does NOT
reconstruct the model load — that would measure a different thing) and samples
the real production process over time. Run it, then arm security from the
tray/voice and let it sit a few minutes: the idle→armed delta is the number.

It reports the SAME "private bytes" metric the M44.2 watchdog +
status_report use (psutil memory_info().private on Windows, rss fallback), so
the readings line up with what Jarvis reports about itself.

USAGE
-----
  # Auto-detect the running Jarvis brain process:
  venv\Scripts\python.exe scripts\armed_memory_probe.py

  # Or point it at a specific PID (Task Manager → Details → pythonw.exe):
  venv\Scripts\python.exe scripts\armed_memory_probe.py --pid 12345

  # Sample every 5s for 10 minutes then auto-summarise:
  venv\Scripts\python.exe scripts\armed_memory_probe.py --interval 5 --duration 600

Ctrl+C at any time prints the summary. This is a read-only observer — it never
touches Jarvis, only samples its counters. (A manual instrument, like
leak_repro.py / barge_stutter_soak.py — NOT a regression gate.)
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    import psutil
except ImportError:
    print("psutil is required: venv\\Scripts\\python.exe -m pip install psutil",
          file=sys.stderr)
    sys.exit(2)


def _private_mb(proc: psutil.Process) -> float:
    """Private (commit) bytes in MB — the M44.2 metric, matching
    src/self_status.process_private_mb so readings are comparable."""
    mem = proc.memory_info()
    raw = getattr(mem, "private", None) or mem.rss
    return raw / (1024 * 1024)


def _rss_mb(proc: psutil.Process) -> float:
    return proc.memory_info().rss / (1024 * 1024)


def _find_jarvis(exclude_pid: int) -> list[psutil.Process]:
    """Find the running Jarvis BRAIN process(es) by cmdline. The brain runs
    `main.py` (directly or via jarvis.pyw); the watchdog (jarvis_watchdog.pyw)
    is deliberately excluded — it holds no models. Returns all matches so the
    caller can disambiguate."""
    found: list[psutil.Process] = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.info["pid"] == exclude_pid:
            continue
        cmdline = " ".join(p.info.get("cmdline") or []).lower()
        if "watchdog" in cmdline:
            continue
        if "main.py" in cmdline or "jarvis.pyw" in cmdline:
            found.append(p)
    return found


def _resolve_target(pid: int | None) -> psutil.Process:
    if pid is not None:
        try:
            return psutil.Process(pid)
        except psutil.NoSuchProcess:
            print(f"No process with PID {pid}.", file=sys.stderr)
            sys.exit(1)
    candidates = _find_jarvis(exclude_pid=psutil.Process().pid)
    if not candidates:
        print("Couldn't find a running Jarvis brain process (no python "
              "process whose cmdline mentions main.py / jarvis.pyw). Is Jarvis "
              "running? Otherwise pass --pid explicitly.", file=sys.stderr)
        sys.exit(1)
    if len(candidates) > 1:
        print("Multiple candidate processes — pass --pid to pick one:",
              file=sys.stderr)
        for p in candidates:
            print(f"  PID {p.pid}: {' '.join(p.cmdline())}", file=sys.stderr)
        sys.exit(1)
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", type=int, default=None,
                    help="PID of the Jarvis process (auto-detected if omitted)")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between samples (default 2)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="total seconds to sample, 0 = until Ctrl+C (default 0)")
    args = ap.parse_args()

    proc = _resolve_target(args.pid)
    try:
        cmd = " ".join(proc.cmdline())
    except psutil.Error:
        cmd = proc.name()
    print(f"Probing PID {proc.pid}: {cmd}")
    print(f"interval={args.interval}s  duration="
          f"{'until Ctrl+C' if args.duration <= 0 else f'{args.duration}s'}")
    print("Tip: note the baseline now, then ARM security and watch the delta.\n")
    print(f"{'t(s)':>6}  {'private(MB)':>12}  {'rss(MB)':>10}  "
          f"{'threads':>8}  {'cpu%':>6}")

    proc.cpu_percent(None)  # prime the CPU counter (first call returns 0.0)
    t0 = time.monotonic()
    samples: list[float] = []
    threads_seen: list[int] = []
    try:
        while True:
            t = time.monotonic() - t0
            try:
                priv = _private_mb(proc)
                rss = _rss_mb(proc)
                nthreads = proc.num_threads()
                cpu = proc.cpu_percent(None)
            except psutil.NoSuchProcess:
                print("\nTarget process exited.", file=sys.stderr)
                break
            samples.append(priv)
            threads_seen.append(nthreads)
            print(f"{t:6.0f}  {priv:12.1f}  {rss:10.1f}  "
                  f"{nthreads:8d}  {cpu:6.1f}")
            if args.duration > 0 and t >= args.duration:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n(interrupted)")

    if samples:
        baseline = samples[0]
        peak = max(samples)
        mean = sum(samples) / len(samples)
        print("\n" + "=" * 56)
        print("Summary (private bytes, MB):")
        print(f"  first sample : {baseline:8.1f}")
        print(f"  peak         : {peak:8.1f}   (+{peak - baseline:.1f} over first)")
        print(f"  mean         : {mean:8.1f}")
        print(f"  threads      : {min(threads_seen)}–{max(threads_seen)}")
        print(f"  samples      : {len(samples)} over {time.monotonic() - t0:.0f}s")
        print("=" * 56)
        print("Compare an idle run vs an armed+acoustic run to get the cost of "
              "arming. A flat private-bytes line over a long armed run is the "
              "M44 leak-fixed signal; a steady climb is a regression.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
