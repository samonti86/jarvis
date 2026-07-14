# Jarvis — Code Audit (consolidation pass)

**Date:** 2026-07-02
**Scope:** Full end-to-end QA pass, user-requested ("go through the code end to
end, review the logs, find any bugs… make sure the code is 100% correct").
Eleven days after pass #3 — triggered by the user's Fable-5 test drive, not the
usual ~20-milestone cadence. No new product features.
**Method:** LOG REVIEW FIRST this time — `jarvis.log` since 2026-06-28 was read
before any code, and it surfaced a live incident the tests never could (see
finding 1). Then **eight parallel read-only auditor agents**, partitioned by
subsystem cohesion, with the shared rubric tuned to this project's DELIBERATE
contracts (fail-soft; WASAPI thread-affinity; the cooperative speech gates; the
M44.3 persistent capture; fail-open speaker gate; no-gate `run_code`; the
restricted phone/Discord tool boundary; GET-only presence; Stop-Service without
`-Force`) so intentional design wasn't flagged. Full `src/` tree + `main.py` +
both launchers + the PWA JS. Every auditor finding was re-verified against the
source before a fix was written.
**Baseline:** `scripts/run_all_tests.py` = **44/44 gates green** before any
change. **After:** **45/45** (added `gates_test` [13]; `predictions` 42→47;
`dismissal` 27→28).

> Prior passes (2026-05-29 M1→M67; 2026-06-09 M68→M77; 2026-06-21 M78→M88) live
> in git history + the `[[project-qol-consolidation-pass]]` memory. This is
> pass #4 — ~37 fixes across 31 files.

Severity legend: 🔴 confirmed bug · 🟡 risk/latent · 🟢 polish.

---

## The live incident (found in the log, not by an auditor)

### 🔴 1. The listening loop was DEAD in production the morning of the audit

At 07:33:08 on 2026-07-02, `AudioSession.__enter__` raised
`PortAudioError: Device unavailable [-9985]` at startup. `listen_loop`'s `try`
had only a `finally` — the thread died unhandled while every other subsystem
(Discord, remote console, monitors, presence auto-arm) kept running. Jarvis was
deaf to voice ALL DAY with no outward sign; the M65 watchdog never fired because
it watches process exit, not thread death. The acoustic subsystem opened the
SAME device successfully 35 minutes later, so a retry would have recovered.
This violated the project's oldest engineering rule: *never crash the listening
loop*.

**Fix — a mic-session supervisor** (`main.py`): the session body is extracted
into `_voice_session_loop()` (byte-identical) and wrapped in a retry loop — any
escape (open failure, a mid-session device error out of the previously-unguarded
`wait_for_wake_word`, a stalled stream) logs the traceback once per outage,
surfaces it in the console AND aloud ("I seem to have lost the microphone,
sir"), then re-opens on a capped backoff (10 s → 60 s), announcing recovery.
Companions: `AudioSession.read()` gained a 10 s stall timeout (a silently-dead
stream — the documented KVM case — now raises into the supervisor instead of
wedging the loop AND blocking quit forever on the join), and the barge-in
monitor's `session.read()` is guarded so a mid-reply mic death ends the monitor
gracefully.

### 🔴 2. Chronic acoustic input overflow through every armed window

`[acoustic] stream status: input overflow` fired every ~3–4 s for the WHOLE of
every armed window — 1,000–2,000 log lines/day (the earlier "occasional
overflow" reading was an artifact of timestamped lines defeating `uniq -c`).
Two causes, both fixed: (a) the acoustic `InputStream` used the sounddevice
default LOW-latency host buffer (tens of ms) though nothing downstream needs
latency (2 s windows) — armed-mode PANNs+YOLO bursts overflowed it constantly,
and every overflow = dropped samples = a gap-corrupted detection window
(degrading exactly the armed rules: a knock transient can vanish;
`voice_while_armed` needs consecutive clean windows). Now `latency="high"`.
(b) The status print ran unthrottled INSIDE the 80 ms-budget PortAudio callback
— a timestamped file write lengthening the very callback whose lateness caused
the overflow. Now a counter + one deduped line per 60 s.

### 🟡 3. iCal fetch had zero retries on a flaky uplink

The log shows several transient TLS-handshake `ConnectTimeout`s per day; the
weather path retries once, the Outlook iCal path never did — one blip failed an
on-demand "what's on my calendar?" outright. Now one retry + 0.5 s backoff.
(The timeout values themselves — 10–15 s handshake budgets — were checked and
are fine.)

### 🟢 4. The clock bug ("what time is it" → "12:00 AM")

The documented follow-up from pass #3. The cached system prefix anchors the
DATE only (by design — cache invalidates once per day); the current TIME now
rides the per-turn UNcached system block (M80/M85's), so time answers come from
ground truth at a few fresh tokens per turn and zero cache-miss cost.

---

## Tier 1 — correctness / contract

### 🔴 5. Interpreter mode: STT-failure hot loop with NO exit

The exact bug pass #3 fixed for conversation mode was never applied to the M87
interpreter branch: `except: print; continue` with the mode still set re-enters
immediately — a tight wake-word-less spin, and the only exit ("stop
interpreting") requires a SUCCESSFUL transcription, so the mode was
unrecoverable short of an app restart. Now a failure counter exits the mode
after `_CONVERSATION_IDLE_EXITS` consecutive failures (reset on success).

### 🔴 6. Interpreter mode: tray "Reset conversation" silently inert

The interpreter branch never consulted `reset_event` (and bypasses
`wait_for_wake_word`, which does) — a reset clicked while interpreting did
nothing, then fired stale after a voice exit. Combined with #5, there was NO
recovery input at all when interpreter STT was failing. The branch now exits
the mode on a pending reset and falls through to the wake-word path, which
seals it normally.

### 🔴 7. `security._auto_disarm` left a permanent transcript-diversion stuck state

Unlike `deactivate()`, `_auto_disarm` (memory watchdog / model failure) never
cleared `_challenge_active`/`_locked`. Tripping mid-CHALLENGE meant the watcher
thread exited (so the timeout check never ran again) while `handle_transcript`
kept diverting EVERY subsequent utterance into `try_authenticate` forever —
disarmed but deaf to everything except the passphrase, with a stale 🔒 pinned.
Now clears challenge/locked exactly like `deactivate()` (incl.
`on_locked_changed(False)`).

### 🔴 8. The speech gates didn't nest — new `src/gates.py` `CountedEvent`

`pc_speaking`/`announce_speaking` are raised by MULTIPLE unserialized speakers
(turn replies, the Announcer, `speak_line`, interpret). With plain Events, a
reminder announce landing 2 s into a 20 s reply cleared both gates in its
`finally` while the reply was still playing — re-opening the 2026-05-29
omni-mic self-capture bug (the open follow-up window can transcribe Jarvis's
own reply tail) AND the M68 armed stutter (PANNs/YOLO resume mid-speech) for
the remainder of the reply. `CountedEvent` keeps the Event API (`set()`
increments, `clear()` decrements, flag drops on the LAST clear, stray clears
clamp at zero) so every call site works unchanged. New `scripts/gates_test.py`
(13). *The audible announce-over-reply overlap itself is a separate deferred
design item — the gates are now correct regardless.*

### 🔴 9. Long phone voice notes killed the WebSocket

The PWA sends a recording as ONE JSON frame (base64 ≈ 1.33× the blob) with a
60 s mic cap — a long Safari audio/mp4 clip tops the websockets default
`max_size` of 1 MiB, closing the connection (1009) and silently losing the
utterance. `serve(..., max_size=8 MiB)`.

### 🟡 10. Disarm race could leak an open camera for the whole disarmed period

`grab_frame_for_snapshot` checked `is_armed()` then called `_grab_frame`, which
OPENS a new persistent capture if `_cap is None` — a Discord snapshot passing
the armed check just as the M70 arrive-home disarm released the watcher's
capture would open one nobody ever releases (LED on, device blocked — the M44
leak class). It now serves ONLY from an already-open capture, under the lock.

### 🟡 11. Conversation-mode idle exits were defeatable

(a) With voice-lock ON, a gate-dropped media turn (TV/YouTube) RESET the idle
counter before the drop — background media kept hands-free mode alive (and STT
burning) forever. The reset moved below the gate; a dropped turn now COUNTS
toward the idle exit. (b) A proactive announce during a listening window aborts
the capture (suppress_event) → empty transcript → counted as user idleness; a
chatty reminder schedule could silently exit conversation mode with the user
present. A suppressed capture (pc_speaking still set) no longer counts.

---

## Tier 2 — latent

- **`knowledge.reindex` wasn't atomic** (🟡): Python's sqlite3 legacy mode runs
  DDL in AUTOCOMMIT, so the DROP/CREATE committed instantly and a crash
  mid-insert (power loss, no-UPS box) left a committed EMPTY index silently
  returning nothing — the exact state the docstring claimed impossible. Now an
  explicit `BEGIN IMMEDIATE` wraps the whole rebuild (SQLite DDL is
  transactional): a crash rolls back to the intact old index.
- **predictions: the mining watermark advanced on FAILURE** (🟡): a failed
  transcript scan or miner call (API down / unparseable output) was
  indistinguishable from "no predictions found" — the watermark advanced and
  any prediction in that window was permanently skipped (incremental scans
  never revisit). `_default_miner` now returns None on failure; the watermark
  only advances on success. +5 regression tests.
- **predictions: `surfaced` lost-update** (🟡): `take_unsurfaced` (briefing
  thread) races an in-flight background cycle whose `_save` of a stale copy
  reverted `surfaced=True` → repeated "you called it". New `_STORE_LOCK`; mine
  merges into a FRESH load; `resolve_due` collects mutations by id across its
  long resolver calls and applies them to a fresh load under the lock.
- **`DedupeStore.save` wasn't durable** (🟡): the one store missed by the
  atomic_io migration — and it backs the 168-hour WEATHER dedupe, so a torn/
  unsynced file after power loss re-announced still-active multi-day NWS
  warnings (the exact class the long retention prevents). → `atomic_write_text`.
- **quiet_hours record/take race** (🟡): both are load-mutate-save from
  different threads; one interleaving resurrects surfaced items, the other
  drops a fresh deferral. → module `_STORE_LOCK`.
- **sound_detector lifecycle** (🟡×4): `activate()`/`deactivate()` are now
  serialized by a lifecycle lock (two near-simultaneous activations — presence
  auto-arm + tray toggle — could BOTH open an InputStream and orphan one,
  capturing forever); a dying inference thread is joined before the spawn check
  (rapid toggle could leave active-with-no-loop); `_download` no longer stats
  the temp file after unlinking it (raised `FileNotFoundError` out of a
  returns-error-string contract); a stream that fails in `start()` is closed,
  not leaked.
- **Same stale-thread spawn race fixed uniformly** in calendar_monitor,
  weather_alerts, homelab_monitor, anticipation (join the mid-exit thread so
  `is_alive()` is truthful).
- **`vision_describe` had no timeout** (🟡): SDK default ≈ 10 min + retries per
  armed visual alert during an outage → thread pileup + stale photos. Now 30 s,
  1 retry (the pass-#3 anticipation fix applied to its sibling).
- **notifications: `httpx.InvalidURL` escaped the never-raises contract** (🟡):
  it's not an `HTTPError` subclass, and the security deterrent calls the
  Discord senders with no try-wrapper on exactly that documented premise. All
  three now catch `Exception`.
- **discord_bot: a non-Forbidden `create_thread` failure discarded the reply**
  (🟡): e.g. HTTP 400 "already has a thread" fell to the outer except and the
  composed answer (+ webcam images) vanished. Broadened to `HTTPException` →
  inline fallback.
- **Two lazy-load races** (🟡): `speech_to_text._get_model` (reachable from the
  listen-loop AND remote-origin threads; double ~250 MB Whisper load) and
  `embeddings.get_embedder` (tray reindex vs turn-thread recall; double ~80 MB
  model) both got load locks.
- **`TimestampStream` write race** (🟡): `print()` = two `write()` calls; an
  interleaving thread could emit its whole line UNstamped. Locked.
- **`_float_env` accepted `nan`/`inf`** (🟡): `JARVIS_SPEAKER_THRESHOLD=nan`
  makes every `score >= threshold` False — with voice-lock ON, a silent total
  lockout with no warning. Non-finite now rejected.
- **`resolve_input_device("--1")` crashed startup** (🟡): `lstrip("-")` strips
  all dashes; `int()` raised. Guarded → default mic.
- **One shared Anthropic client** (🟡/perf): `stream_response` and
  `stream_translation` built a NEW client (new httpx pool → fresh TLS
  handshake) every turn — and per relayed utterance in interpreter mode. Now a
  keyed lazy singleton (`_get_client`), thread-safe per the SDK.
- **remote_console token compare** (🟡): `hmac.compare_digest(str, str)` raises
  `TypeError` on non-ASCII — a pasted token with a Unicode dash killed the
  handler task with no `auth_fail`, so the client retried the bad token forever.
  Both auth paths now compare UTF-8 bytes inside a guard.
- **remote_console fire-and-forget task refs** (🟡): the loop holds only weak
  refs; an untracked send task could be GC'd mid-flight (dropped frame / reply
  audio). `_pending_tasks` set + done-callback discard.
- **pc_shell argv option-injection** (🟡): `_HOSTNAME_RE` allowed a leading
  `-`, which ping/tracert/nslookup would parse as an OPTION, not a host.
  Leading-hyphen targets rejected.
- **tmdb last-resort wrapper** (🟢): `execute_tmdb_tool`'s never-raises
  contract rested entirely on per-helper guards (unguarded `int(id)`); it now
  has the same defensive ceiling as system_control/pc_shell.
- **aec_barge None guard** (🟢): `self._interrupt.set()` without a None check
  inside the PortAudio callback (the event is Optional at every call site).
- **security deterrent LOCKED re-check** (🟡): a successful auth landing during
  the slow evidence/push window used to be followed by an unconditional
  `on_locked_changed(True)` + "authorities have been notified" right after
  "Welcome back, sir", pinning a stale 🔒. Re-checked under the lock; evidence/
  pushes still fire (correct — someone WAS unauthenticated the whole window).

---

## Tier 3 — polish

- **"good night" is no longer a dismissal** — as one it short-circuited before
  the LLM on every follow-up/conversation-mode turn, shadowing the M63
  `get_good_night` wrap. It now reaches Claude and routes to the wrap.
  (`dismissal_test` updated: 28.)
- **PWA transcript DOM capped** at 500 nodes (an installed PWA left connected
  for days grew without bound).
- **Watchdog no longer defeats log rotation** — its lifetime-open handle on
  `jarvis.log` made every child respawn's `rotate_if_needed` rename fail
  silently (unbounded growth past 5 MB). The watchdog now writes its own
  `jarvis_watchdog.log`. Also: spawn falls back to the running interpreter when
  the venv is absent (was a guaranteed FileNotFoundError → "giving up"), and
  the dead `_VENV_PYTHON` variable is gone.
- **weather_alerts** dead `_geocode_failed` field removed (documented a
  permanent-failure latch that never existed).
- **autostart** PowerShell single-quote escaping (`'` → `''`) on every
  interpolated path/description — an apostrophe profile path broke the shortcut
  script (and was technically an injection surface).
- **ui tray guard** — a pystray/Win32 construction failure used to kill the
  tray thread silently, leaving NO quit path (the console X only hides). Now
  logged + the console window is shown as the fallback surface.

## Deferred (real items, not rushed)

- **OneDrive-redirected Desktop**: `create_desktop_shortcut` writes to
  `%USERPROFILE%\Desktop` ignoring known-folder redirection — on an
  OneDrive-backup profile the icon lands in an invisible folder. Proper fix is
  `SHGetKnownFolderPath`; works on this box, so deferred.
- **Audible announce-over-reply overlap**: nothing serializes Announcer
  playback against a playing turn reply (both audible at once). The gate
  correctness is fixed (CountedEvent); making announces DEFER while a turn is
  speaking is a small design change for its own session.
- **Text/voice intent-dispatch rule-of-three extraction** — carried from pass
  #3; still wants its own test-harness-first pass.

---

## Verified non-issues (reassurance)

Everything pass #3 verified was re-verified clean by fresh eyes: no command
injection / allowlist bypass beyond the leading-hyphen nit (system_control +
pc_shell quote-out is sound; mutating verbs confirm + elevate server-side;
`plex_actions` mutations `retry=False`); the two-gate restricted-origin
boundary agrees at both gates incl. the Discord camera claw-back; the M44.3
persistent capture is intact (and now closed against the last escape path,
finding 10); WASAPI thread-affinity observed throughout; the async↔thread
bridges marshal correctly; per-turn reply sinks don't leak across origins; no
PWA XSS; `run_code` container isolation intact; prompt-cache discipline holds
(single cache_control breakpoint, byte-stable prefix; the new time line rides
the uncached per-turn block); the atomic_io implementation itself is correct;
`recall_conversation`'s day-window and speaker-filter fixes hold; the all-day
calendar TZ fix holds; presence's generation-token cancel race is closed.

Every fix is reversible and gated by `scripts/run_all_tests.py` (**45/45**).
Live-validation checklist for the next session: a normal voice turn (supervisor
didn't disturb the hot path), unplug/replug the mic mid-session (supervisor
recovers + speaks), an armed window (overflow lines rate-limited to ~1/min,
detection still fires), "good night" (routes to the M63 wrap), a long phone
voice note, and "what time is it" (real time).
