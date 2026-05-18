"""M44.3 — armed-security memory-leak reproduction + bisection harness.

WHY THIS EXISTS
---------------
The armed-security watcher leaks the *process's private/commit bytes* to
~37 GB over ~90 min and locks the machine (M44 / M44.2 post-mortems in
docs/MILESTONES.md). M44's `del results` + `gc.collect()` mitigation did
NOT hold. Key insight (M44.3 thesis): `self._model = YOLO(...)` is created
once and cached forever, so ultralytics' persistent `model.predictor`
retains the last run's state. Those buffers are *reachable* from the live
model, not *garbage* — `gc.collect()` can never free them. It's a
live-retention leak, not a collectable cycle: wrong tool for the bug.

This harness reproduces that leak in MINUTES instead of 90 min by running
inference flat-out (no 2 s poll sleep) on a synthetic dark frame (the
covered-lens analog used in the real soak), while instrumenting:

  * psutil  -> process PRIVATE bytes (the M44.2 metric that actually grew)
              and RSS (shown alongside to make the M44.2 finding visible:
              expect RSS ~flat while PRIVATE climbs).
  * tracemalloc -> Python-side allocation growth, per file:line. If PRIVATE
              climbs while tracemalloc total stays flat, the leak is in
              NATIVE memory (torch/OpenCV C++), which is itself the answer.

It also serves step 3 (differential bisection): every candidate fix is a
toggle, so each hypothesis is a ~2 min experiment, not a 90 min soak.

SAFETY
------
We are deliberately reproducing an OOM leak on a memory-tight box (~1.2 GB
idle-available). The harness self-terminates — same philosophy as the
M44.2 watchdog applied to our own tool — on ANY of:
  * --max-iters reached
  * process private bytes >= --max-private-mb
  * system available memory < --min-avail-mb  (hard machine-safety abort)
Defaults are conservative. It will not take the machine down.

USAGE
-----
  # Baseline — faithfully reproduce production (M44 mitigation ON):
  venv\\Scripts\\python.exe scripts\\leak_repro.py

  # Bisection examples (step 3):
  ... scripts\\leak_repro.py --del-predictor      # candidate fix B
  ... scripts\\leak_repro.py --recreate-every 200 # candidate fix D
  ... scripts\\leak_repro.py --inference-mode     # candidate A
  ... scripts\\leak_repro.py --gc-every 0 --no-del-results  # leak, unmitigated

Run from the repo root with the project venv so the ultralytics / torch /
psutil versions match production exactly.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import tracemalloc

import numpy as np
import psutil

# Faithful to src/security.py: _CAPTURE_WIDTH=640, _CAPTURE_HEIGHT=480, the
# frame handed to self._model is a BGR uint8 ndarray of shape (H, W, 3).
_FRAME_H, _FRAME_W = 480, 640


def _make_frame(kind: str) -> np.ndarray:
    """Synthetic frame. 'dark' is the faithful covered-lens repro (what the
    real soak used). 'random' is a sanity contrast — a busy frame exercises
    more of the predictor's post-processing path; if the leak rate differs,
    that itself localizes it."""
    if kind == "random":
        return np.random.randint(0, 256, (_FRAME_H, _FRAME_W, 3), dtype=np.uint8)
    # Covered lens isn't pure zero — sensor noise floor. A few-LSB noisy
    # near-black frame is closer to reality than np.zeros and avoids any
    # degenerate all-zero fast path in the backend.
    return np.random.randint(0, 6, (_FRAME_H, _FRAME_W, 3), dtype=np.uint8)


def _backend_const(cv2, name: str) -> int:
    """Map a --camera-backend choice to the OpenCV constant. DSHOW is what
    production (src/security.py:_grab_frame) uses today and what the M44.3
    repro proved leaks. MSMF is the candidate near-one-line fix (Windows
    Media Foundation backend — commonly does NOT exhibit the DSHOW capture-
    graph leak). ANY lets OpenCV pick (usually MSMF on modern Windows)."""
    return {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": cv2.CAP_ANY,
    }[name]


def _open_capture(cv2, camera_index: int, backend: int):
    """Open + configure one VideoCapture. Returns the cap or None."""
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _FRAME_H)
    return cap


def _grab_frame_per_tick(cv2, camera_index: int, backend: int):
    """Replica of src/security.py:_grab_frame() — open-every-tick / read 2
    warmup frames / release-in-finally, now backend-parameterized. With
    backend=DSHOW this is byte-faithful to today's production code (the leak
    repro). With backend=MSMF it tests the minimal-change fix that PRESERVES
    the deliberate open/release-per-tick camera-sharing design."""
    cap = None
    try:
        cap = _open_capture(cv2, camera_index, backend)
        if cap is None:
            return None
        frame = None
        for _ in range(2):  # _WARMUP_FRAMES
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        return frame
    except Exception:  # noqa: BLE001 — production swallows + returns None too
        return None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def _mb(n: int) -> float:
    return n / (1024 * 1024)


def main() -> int:
    ap = argparse.ArgumentParser(description="M44.3 leak repro + bisection harness")
    # --- mitigation toggles (production-faithful defaults) ---
    ap.add_argument("--del-results", dest="del_results", action="store_true",
                    default=True, help="del the results each iter (M44, default ON)")
    ap.add_argument("--no-del-results", dest="del_results", action="store_false",
                    help="do NOT del results (unmitigated leak)")
    ap.add_argument("--gc-every", type=int, default=1, metavar="N",
                    help="gc.collect() every N iters (M44 = 1; 0 = never)")
    ap.add_argument("--del-predictor", action="store_true",
                    help="candidate fix B: drop model.predictor after each call")
    ap.add_argument("--recreate-every", type=int, default=0, metavar="N",
                    help="candidate fix D: rebuild YOLO every N iters (0 = never)")
    ap.add_argument("--inference-mode", action="store_true",
                    help="candidate fix A: wrap inference in torch.inference_mode()")
    # --- run shape ---
    ap.add_argument("--frame", choices=["dark", "random"], default="dark",
                    help="synthetic frame kind (default dark = covered lens)")
    ap.add_argument("--with-camera", action="store_true",
                    help="replicate src/security.py:_grab_frame() every iter "
                         "(open CAP_DSHOW -> warmup reads -> release). The "
                         "prime M44.3 suspect once bare inference was "
                         "exonerated. COVER THE LENS for soak fidelity.")
    ap.add_argument("--camera-index", type=int, default=0, metavar="N")
    ap.add_argument("--camera-backend", choices=["dshow", "msmf", "any"],
                    default="dshow",
                    help="OpenCV capture backend. dshow = production today "
                         "(leaks). msmf = candidate minimal fix.")
    ap.add_argument("--camera-persistent", action="store_true",
                    help="open the VideoCapture ONCE, read per iter, release "
                         "at end (connection-pooling fix). Without this, "
                         "open/release every iter like production.")
    ap.add_argument("--warmup", type=int, default=30, metavar="N",
                    help="iters excluded from the slope baseline (lazy init)")
    ap.add_argument("--report-every", type=int, default=50, metavar="N")
    ap.add_argument("--snapshot-every", type=int, default=200, metavar="N",
                    help="tracemalloc snapshot+diff cadence")
    ap.add_argument("--top", type=int, default=6, help="tracemalloc top growers")
    # --- safety caps (machine protection) ---
    ap.add_argument("--max-iters", type=int, default=3000, metavar="N")
    ap.add_argument("--max-private-mb", type=float, default=3000.0, metavar="MB")
    ap.add_argument("--min-avail-mb", type=float, default=700.0, metavar="MB",
                    help="hard abort if system available memory drops below this")
    args = ap.parse_args()

    print("=" * 78)
    print("M44.3 leak repro — config:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k:18s} = {v}")
    print("=" * 78)

    inference_ctx = None
    if args.inference_mode:
        import torch  # noqa: PLC0415 — lazy; ultralytics pulls torch anyway
        inference_ctx = torch.inference_mode
        print("[harness] torch.inference_mode() wrapping ENABLED")

    from ultralytics import YOLO  # noqa: PLC0415 — heavy dep, lazy by design

    def _new_model():
        # Mirrors src/security.py:_ensure_model — same weights, same call.
        return YOLO("yolov8n.pt")

    model = _new_model()
    frame = _make_frame(args.frame)
    proc = psutil.Process()

    cv2 = None
    cam_fail = 0
    backend = None
    persistent_cap = None
    if args.with_camera:
        import cv2  # noqa: PLC0415 — lazy; only the camera path needs it
        backend = _backend_const(cv2, args.camera_backend)
        mode = ("PERSISTENT (open once, read per iter, release at end)"
                if args.camera_persistent
                else "per-tick (open/read/release every iter, like prod)")
        print(f"[harness] --with-camera ON: index {args.camera_index}, "
              f"backend={args.camera_backend.upper()}, mode={mode}. An "
              f"unavailable camera returns None (counted) and falls back to "
              f"the synthetic frame — that would NOT exercise the grab cycle, "
              f"so the run is flagged invalid if >50% fail.")
        if args.camera_persistent:
            persistent_cap = _open_capture(cv2, args.camera_index, backend)
            if persistent_cap is None:
                print("[harness] FATAL: could not open camera for persistent "
                      "mode — aborting (cover lens, don't unplug).")
                return 2

    tracemalloc.start(25)  # 25 frames of traceback depth per allocation
    prev_snap = tracemalloc.take_snapshot()

    base_private = None  # private MB captured right after warmup
    base_iter = args.warmup
    t_start = time.monotonic()
    stop_reason = "max-iters reached"

    def _infer(active_frame):
        results = model(active_frame, verbose=False)
        # Faithful to _detect_person: it touches r.boxes before discarding.
        for r in results:
            _ = getattr(r, "boxes", None)
        if args.del_results:
            del results

    i = 0
    try:
        for i in range(1, args.max_iters + 1):
            if args.with_camera:
                if args.camera_persistent:
                    ok, f = persistent_cap.read()
                    grabbed = f if (ok and f is not None) else None
                else:
                    grabbed = _grab_frame_per_tick(
                        cv2, args.camera_index, backend)
                if grabbed is None:
                    cam_fail += 1
                    active_frame = frame  # fall back so the loop continues
                else:
                    active_frame = grabbed
            else:
                active_frame = frame

            if inference_ctx is not None:
                with inference_ctx():
                    _infer(active_frame)
            else:
                _infer(active_frame)

            if args.del_predictor:
                # ultralytics lazily rebuilds predictor on the next call.
                model.predictor = None

            if args.recreate_every and i % args.recreate_every == 0:
                del model
                gc.collect()
                model = _new_model()

            if args.gc_every and i % args.gc_every == 0:
                gc.collect()

            mi = proc.memory_info()
            priv_mb = _mb(getattr(mi, "private", mi.rss))
            rss_mb = _mb(mi.rss)
            avail_mb = psutil.virtual_memory().available / (1024 * 1024)

            if i == base_iter:
                base_private = priv_mb

            # ---- safety caps ----
            if avail_mb < args.min_avail_mb:
                stop_reason = (f"SAFETY ABORT: system available "
                               f"{avail_mb:.0f} MB < {args.min_avail_mb:.0f} MB")
                break
            if priv_mb >= args.max_private_mb:
                stop_reason = (f"cap: process private {priv_mb:.0f} MB >= "
                               f"{args.max_private_mb:.0f} MB")
                break

            if i % args.report_every == 0:
                d = "" if base_private is None else \
                    f"  Δpriv/baseline={priv_mb - base_private:+.0f}MB"
                el = time.monotonic() - t_start
                print(f"[{i:5d}] {el:6.1f}s  private={priv_mb:7.1f}MB  "
                      f"rss={rss_mb:7.1f}MB  sysavail={avail_mb:6.0f}MB{d}")

            if i % args.snapshot_every == 0:
                snap = tracemalloc.take_snapshot()
                diff = snap.compare_to(prev_snap, "lineno")
                traced = _mb(sum(s.size for s in snap.statistics("lineno")))
                print(f"   -- tracemalloc total traced (Python) = "
                      f"{traced:.1f} MB ; top {args.top} growers since last:")
                for st in diff[:args.top]:
                    print(f"      {st.size_diff/1024:+9.1f} KiB  "
                          f"{st.count_diff:+7d} obj  {st.traceback[0]}")
                prev_snap = snap
    except KeyboardInterrupt:
        stop_reason = "KeyboardInterrupt (manual stop)"
    finally:
        if persistent_cap is not None:
            try:
                persistent_cap.release()
            except Exception:
                pass

    # ---------------- summary / verdict ----------------
    elapsed = time.monotonic() - t_start
    mi = proc.memory_info()
    end_private = _mb(getattr(mi, "private", mi.rss))
    end_rss = _mb(mi.rss)
    end_snap = tracemalloc.take_snapshot()
    end_traced = _mb(sum(s.size for s in end_snap.statistics("lineno")))
    tracemalloc.stop()

    print("=" * 78)
    print(f"STOP: {stop_reason}")
    print(f"iters run            : {i}")
    print(f"elapsed              : {elapsed:.1f}s "
          f"({i/elapsed:.1f} infer/s)" if elapsed > 0 else "")
    print(f"private  end         : {end_private:.1f} MB")
    print(f"rss      end         : {end_rss:.1f} MB")
    if args.with_camera:
        print(f"camera grab failures : {cam_fail}/{i}")
        if i and cam_fail > i * 0.5:
            print("  !! >50% of grabs returned None — camera unavailable. "
                  "This run fell back to the synthetic frame and did NOT "
                  "exercise the grab cycle. INVALID for the camera "
                  "hypothesis (cover the lens; don't unplug).")
    if base_private is not None and i > base_iter:
        grew = end_private - base_private
        per100 = grew / max(1, (i - base_iter)) * 100
        per_min = grew / elapsed * 60 if elapsed > 0 else 0.0
        print(f"private growth       : {grew:+.1f} MB over "
              f"{i - base_iter} post-warmup iters")
        print(f"  -> slope           : {per100:+.2f} MB / 100 iters "
              f"| {per_min:+.1f} MB / min")
        print(f"tracemalloc traced   : {end_traced:.1f} MB "
              f"(Python-side heap)")
        # Heuristic verdict — the whole point of step 2.
        leaking = per100 > 5.0  # >5 MB/100 iters is unambiguous here
        py_grew = end_traced > 150.0  # arbitrary "Python heap is large/growing"
        print("-" * 78)
        if not leaking:
            print("VERDICT: slope ~flat — this configuration does NOT leak. "
                  "If a fix toggle is on, it WORKS.")
        elif py_grew:
            print("VERDICT: leaking AND Python heap grew — Python-side "
                  "retention. The tracemalloc top-growers above name the "
                  "leaking file:line. Fix there.")
        else:
            print("VERDICT: leaking BUT Python heap is flat — the leak is in "
                  "NATIVE memory (C/C++), so Python GC / tracemalloc cannot "
                  "see or fix it. M44.3 traced this to the per-tick "
                  "cv2.VideoCapture open/release (DSHOW catastrophic, MSMF "
                  "~1600x less but still nonzero); inference was exonerated "
                  "(600-iter flat run). Isolate which native op via the "
                  "toggles: --camera-persistent / --camera-backend for the "
                  "grab cycle; --del-predictor / --recreate-every only if a "
                  "no-camera run also leaks.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
