# Engineering log — part 5 of 5

2026-05-26 → 2026-07-02  ·  27 entries

[← part 4](part-4.md)  ·  [full index](../MILESTONES.md)  ·  [index →](../MILESTONES.md)

---

## Debugging session follow-up — TTS retry on transient edge-tts flake

*(2026-06-02)*

- **Trigger.** The 13:00 session (which live-confirmed all three fixes above) had ONE new log line: `[tts] synth failed for sentence (78 chars): No audio was received. Please verify that your parameters are correct.` — one sentence of a spoken reply was dropped.
- **Root cause (reproduced, not guessed).** The 78-char sentence re-synthesized **5/5 OK** in isolation — so the content is fine; the live failure was a **transient edge-tts flake**. `NoAudioReceived` is raised when edge-tts's websocket to Microsoft's voice endpoint closes WITHOUT delivering audio (rate-limit / network blip / service hiccup) — its most common flake, and the textbook retry case. **The latent bug it exposed:** the streaming TTS path (`speak_streaming.synth_worker`, the path used for EVERY conversational reply) caught a synth failure and just **dropped the sentence — no retry, no fallback** — whereas the one-shot `speak()` path falls back to pyttsx3. The most-used voice path was the *least* resilient to the most common failure of its online dependency, and the drop was silent (log only) — invisible on a typed turn (you read the text), audible-gap on a spoken turn.
- **Fix.** New `_fetch_mp3_with_retry` helper (bounded: 3 attempts, linear 0.4s backoff) wrapping `_fetch_mp3`; retries on BOTH a raised exception AND an empty-bytes result, then re-raises the last error so each caller's EXISTING terminal behaviour still runs on a genuine persistent failure (synth_worker drops; speak() → pyttsx3; phone path logs). Applied at all THREE raw `_fetch_mp3` call sites: `synth_worker` (streaming), `_speak_edge_tts` (one-shot, retry before the pyttsx3 voice-swap), and `main.py`'s phone end-of-turn synth (a flake there meant the phone got NO audio at all). The streaming path's maxsize=2 lookahead buffer absorbs the backoff, so playback stays gapless in the common single-retry case. **Did NOT** wire pyttsx3 into the streaming path — it plays via `runAndWait()` and produces no samples for the audio queue, so it doesn't fit the pipeline, and a mid-reply voice-swap is jarring for a rare double-failure; retry is the right-sized fix.
- **Validation.** `scripts/tts_retry_test.py` (**10 assertions**, network-free — monkeypatches the inner `_fetch_mp3` with a scripted sequence, zeroes the backoff): first-try success makes 1 call, a raised error retries-then-succeeds, persistent failure re-raises after 3 tries, empty-bytes is retried, custom `attempts` honored. Live end-to-end call through the real wrapper returned 15,840 bytes. Full gate **19/19**.
- **Files:** `src/text_to_speech.py` (`_fetch_mp3_with_retry` + 2 call sites), `main.py` (phone call site), `scripts/tts_retry_test.py` (NEW, 10/10), this entry.
- **Reusable lessons:** (1) **Reproduce before you theorize a content bug** — re-running the exact "bad" sentence 5/5 OK turned "what's wrong with this text" into "this is a transient flake," which is a completely different (and correct) fix. (2) **An online dependency needs a retry on its most common transient failure, on EVERY path that uses it** — the one-shot path had a fallback but the streaming path (the common one) had nothing; resilience belongs on the hot path most of all. (3) **A silent drop is a latent bug even when it "degrades gracefully"** — graceful-but-invisible (log-only) means it's invisible exactly when it matters (a spoken turn), so the right bar is "recover," not just "don't crash."

## Reminders → Discord push on fire

*(2026-06-02, PR #7)*

- The M53 "Discord/email push on fire" follow-on. `run_scheduler` gained an optional `notify` sink alongside `announce`; every fired reminder's text goes to BOTH (plain reminders push `_fire_text`; scheduled briefings / good-night wraps push the full composed text). Extracted `_fire_one` (testable fan-out) + a fail-soft `_push`. main.py builds the sink from the existing M38 `DISCORD_WEBHOOK_URL`. ON by default when the webhook is set (`JARVIS_REMINDER_DISCORD=0` to opt out); no presence detection, so it always pushes (harmless when local). `scripts/reminders_test.py` grew to 50. **Live-validated** (`[notify] discord message sent (HTTP 204)`).

## Voice-lock gate persists across restarts

*(2026-06-02, PR #8)*

- The M69 gate was runtime-only — a tray toggle ON didn't survive a restart (re-read the env default = off). Added `src/ui_state.py` (a tiny persisted toggle store: `%LOCALAPPDATA%\Jarvis\ui_state.json`, atomic write, fail-soft, keyed by name — same file-as-state discipline as reminders.json). main.py restores the gate at startup (persisted value wins; env default for a fresh install) and writes on toggle. `scripts/ui_state_test.py` 14/14. **Live-validated** — `[speaker] voice-lock gate restored ON` across three consecutive restarts incl. a watchdog respawn.

## Follow-up sign-off short-circuit (no duplicate actions)

*(2026-06-02, PR #9)*

- A follow-up utterance that's a pure sign-off ("Thank you that is all") was sent to Claude as a full turn (dismissal was only checked AFTER, to close the window) — with a freshly-set reminder in context, Claude re-issued `set_reminder` and created a DUPLICATE (the "fired twice" report). Now a follow-up turn short-circuits BEFORE `process_question` when `_is_dismissal` matches — registers the transcript, closes the window, takes NO action. Scoped to follow-up turns (a wake-word "that's all" still reaches Claude). `scripts/dismissal_test.py` 27/27. **Live-validated** — one reminder set, one fire, one push; `[main] follow-up sign-off — closing without an LLM turn`.

## Discord bot bridge — a two-way channel as a Jarvis client

*(2026-06-02)*

- **What:** a private Discord channel becomes a full conversational client — the primary user AND additional household users type to Jarvis in the channel and get replies. The webhook (one-way reminder push) and the bot (two-way conversation) coexist in the SAME channel.
- **Why it was mostly recombination:** the M48 phone work already built the seam — every input funnels through `text_queue` as `(text, attachments, origin, reply_audio, lang)`, and `origin` already derives PC-TTS gating + the restricted tool surface. Discord is a FOURTH input path → `origin="discord"` (text-only + restricted, alongside the phone origins). The async↔thread bridge is the SAME pattern `remote_console.py` uses (a discord.py client on its own daemon thread; replies marshalled back onto its loop via `run_coroutine_threadsafe`).
- **Bot, not webhook; gateway, not HTTP-interactions.** Receiving messages needs a bot (token + Gateway WS + the privileged Message Content intent). The gateway is an OUTBOUND connection ⇒ **no inbound port / Tailscale / TLS** — it works off-network for free (Discord is the relay). That's a strictly nicer remote path than the PWA's Tailscale requirement.
- **Security (the load-bearing part).** Discord is internet-relayed AND multi-human, so it's the most-restricted origin: `restricted=True` (server-enforced via the M48.2a two-gate filter — no system/shell/file/code/screen/camera/self-update) AND text-only. TWO access gates: the configured CHANNEL (coarse) + a user-ID ALLOWLIST (fine), both **fail-closed** (no allowlist ⇒ nobody, never everybody; garbage IDs dropped). Bot/webhook authors are ignored, so the bot can't loop on its own replies or the reminder webhook. Discord turns also SKIP the local privileged shortcuts (face/voice enrollment would capture the PC's own camera/mic from a chat message — nonsensical/unsafe; security + knowledge shortcuts reserved for the trusted local paths) — a Discord turn is purely conversational with restricted tools.
- **Privacy (the detail verified in code).** Replies fan out via `add_jarvis_text` to ALL JarvisUI sinks (console + the phone broadcast). A naive "Discord sink" would echo EVERY PC/voice/phone reply into the shared channel other people read. So Discord uses a **per-turn reply sink** (`reply_text`, sibling of `reply_audio`), set ONLY for discord-origin turns — a PC/voice/phone turn never leaks into the channel. Covers both the success and the apology paths so a remote user always gets a reply.
- **Build:** `src/discord_bot.py` (NEW — the bot on a daemon thread; pure `should_handle` filter + `chunk_message` 2000-char splitter factored out for testing); `text_queue` tuple grew a 6th `reply_text` element (all 4 enqueue sites + the unpack + `process_question`); `origin="discord"` added to the text-only + restricted derivations; `src/config.py` (`discord_bot_token` / `discord_channel_id` / `discord_allowed_user_ids` with fail-closed parsing); main.py starts the bot when all three are set; `requirements.txt` (`discord.py>=2.7.1`); `.env.example` (full setup walkthrough). `scripts/discord_bot_test.py` **24/24** (gating + chunking + config parsing). Full gate **22/22**.
- **Validated short of the live round-trip:** a read-only connection smoke proved the token valid, the bot present in the server, the channel resolving, send/read perms True, and — implicitly — the Message Content intent IS enabled (discord.py requests it on connect; a successful connect means the portal toggle is on, else `PrivilegedIntentsRequired`). **Caught a setup bug in the smoke:** the **Server ID** had been pasted into `DISCORD_CHANNEL_ID` (the right-click menu's "Copy Server ID" vs "Copy Channel ID" mix-up) — corrected to the target channel ID. Live message→reply round-trip pending a restart.
- **Decisions:** v1 shared conversation context (one TurnRunner — console + phone + Discord share history, fine at household scale; per-user split is a documented v2). Output via the bot (not the webhook) so it's "Jarvis" replying with chunking. Reply via per-turn sink, not broadcast (privacy). Daemon thread, no explicit shutdown (dies with the process — a clean `client.close()` is a minor follow-on).
- **Reusable lessons:** (1) **A good seam pays compound interest** — the M48 `text_queue`+`origin` design meant a whole new client surface was ~1 module + a tuple field, not a refactor. (2) **A new remote surface inherits the security model only if you wire its origin into the SAME derivations** — `restricted`/`text_only` are one-line edits, but forgetting them would silently hand a chat channel the full tool surface. (3) **Validate connectivity read-only before the live deploy** — the smoke test caught the server-vs-channel ID mix-up in seconds, with nothing posted to the channel. (4) **Per-turn routing vs broadcast is a privacy decision, not just a plumbing one** — verify where a reply fans out before adding a sink to a shared, multi-human surface. (5) **Outbound-gateway > inbound-endpoint for personal remote access** — no port-forward, no TLS, works anywhere; the opposite of the PWA's networking burden.
- **LIVE-VALIDATED + two refinements (PR #11, #12, 2026-06-02).** Once two people were conversing with Jarvis in the channel, real use surfaced two wants: (a) **addressed-only** — in a channel where the humans also chat with each other, the bot replies ONLY when explicitly addressed (an @-mention OR the word "jarvis", word-boundary/case-insensitive), EXCEPT inside a thread the bot itself started (there, every message is for Jarvis — natural follow-ups). New pure `is_addressed`; `should_handle` gained thread scoping (accepts the channel OR a thread whose parent is the channel); a `self._bot_threads` set tracks bot-created threads to skip the gate inside them. (b) **threaded replies** — replies post in a thread off the triggering message (or the existing thread), keeping the main channel uncluttered; `strip_bot_mention` cleans the thread TITLE (it was leaking the raw `<@id>` markup) and `auto_archive_duration=60` makes threads self-clear from the sidebar ~1h after the last message (still in Archived Threads; a reply un-archives). Falls back to an inline reply + a "grant Create Public Threads / Send Messages in Threads" log line if the bot lacks thread perms (it had them via default `@everyone`). `scripts/discord_bot_test.py` 24 → **44**. **The permissions↔token clarification:** changing a bot's permissions NEVER requires a token change or re-invite — token = identity, perms = authorization (server roles + channel overrides), and perm changes apply live over the gateway. **Lessons:** (6) **iterate-on-real-usage is real** — addressed-only + threads weren't designed up front; they came from watching two humans actually use the channel. (7) **The title is content too** — a feature that echoes user text (thread names) needs the SAME cleaning as the LLM input path; `strip_bot_mention` should have been applied to both from the start. (8) **A "declutter" feature can itself clutter** — threads kept the main channel clean but the thread LIST then cluttered the sidebar; auto-archive closed that loop. Showing the options as ASCII previews made the auto-archive-vs-immediate-archive-vs-inline choice concrete.

---

## M70 — Geofenced auto-arm (phone presence → security state) — 2026-06-03

**The last "Jarvis magic" item from the M58+ wish list, and the close on the security arc.** The M34 security flow had exactly one friction point: remembering to arm it. The phone is already a Jarvis client (M48). M70 ties them together — security auto-arms on a geofence-exit event and disarms + greets on a geofence-entry event. Pure recombination of two mature subsystems, the pattern that has worked every time in this project.

- **Geofencing is OFFLOADED to iOS, by design.** The OS does native, battery-efficient, hysteresis-buffered geofence detection far better than a web app could. The alternatives were rejected on evidence: PWA background geolocation is crippled on iOS (won't run while the page is asleep — the same secure-context/lifecycle reality the M48 spike already taught), and WiFi/Tailscale presence can't distinguish the two states (the Tailscale tunnel stays up on cellular). So an **iOS Shortcuts personal automation** ("When I Leave" / "When I Arrive") fires a token-gated HTTP request and Jarvis applies only *policy*. The teaching beat: lean on the platform's solved primitive instead of reimplementing it badly.
- **Transport — a new HTTP route on the existing remote console, no new server.** `src/remote_console.py`'s `_process_request` already short-circuits non-WS paths (PWA, manifest, `/silence.mp3`, `/healthz`); M70 adds `/presence`. Auth uses the SAME `JARVIS_REMOTE_TOKEN` as the WS console (constant-time `hmac.compare_digest`), accepted as an `Authorization: Bearer` header (preferred), an `X-Jarvis-Token` header, or a `?token=` query param. The event rides the query string (`?event=leave|arrive`). **No new enable flag** — the route exists iff the console does (iff a token is set), inheriting the least-privilege gate.
- **`src/presence.py` — `PresenceController` holds the policy, isolated from transport + I/O for unit-testing** (same discipline as `homelab_monitor._CheckTracker` and `calendar_monitor._should_announce`). Three rules:
  - **Idempotent** — a duplicate "leave" while already armed is a no-op (iOS geofences can re-fire a boundary crossing).
  - **Boundary-flap damping** — arming is DEFERRED by `JARVIS_PRESENCE_ARM_DELAY` (default 60s) and cancelled if an "arrive" lands first. This absorbs the classic geofence failure: skirting the boundary fires leave→arrive within seconds, which would otherwise arm-then-disarm and greet a user who never left. The geofence is already ~100m out, so a minute's delay costs nothing. A monotonic **generation token** guards the cancel-vs-fire race (a Timer that fired the instant an "arrive" cancelled it validates its generation and aborts — `Timer.cancel()` can't stop a callback that already started).
  - **Greet only on a real transition** — the greeting fires only when an arrive actually disarmed an *armed* system, never on a spurious arrive to an already-disarmed state. Greeting routes through the WASAPI-safe `_announce` (tagged 🏠), so it's gated like every other proactive announce (the M68 lesson — the speech gate has a producer side too).
- **Wiring** — `_build_remote_console` gained an `announce` param (the Announcer is built *before* the remote console, so `_announce` is already in scope — no late injection needed); it constructs the `PresenceController` bound to `security_watcher.activate/deactivate/is_armed` and passes `presence.handle_event` as a new `on_presence` callback. Deferred arms and the disarm-on-arrive propagate to all phones through the EXISTING `on_armed_changed → ui.set_armed_indicator → update_armed` broadcast path (the tray/voice arming path), so the handler deliberately does NOT broadcast itself.
- **The live-test war story (GET-only).** `scripts/presence_test.py` (**46/46**) covers the policy exhaustively + the route glue via a duck-typed fake `Request`. But the fake couldn't model one platform reality: a throwaway live smoke through the *real* `websockets` server revealed its HTTP parser **rejects any non-GET method at the request-line stage** — `process_request` is never even invoked for a POST (`ValueError: unsupported HTTP method; expected GET; got POST`). The first draft of the docs (and the iOS walkthrough) said POST, which would have failed on the phone. Re-ran with GET → clean 200/401 end-to-end. **Lesson (again): a unit test with a hand-built input validates your logic, not the framework's gatekeeping — a live smoke against the real dependency is what catches "the platform won't even let you in."** Same family as M48's secure-context spike and M65's launcher-chain bugs that synthetic harnesses missed.
- **Config:** `presence_arm_delay` (`JARVIS_PRESENCE_ARM_DELAY`, default 60) + `presence_greeting` (`JARVIS_PRESENCE_GREETING`). `.env.example` ships the full one-time iOS Shortcuts walkthrough (Automation → When I Leave/Arrive → Get Contents of URL, **Method=GET**, `Authorization: Bearer` header). Off-network firing requires the Tailscale/TLS path (M48.2c) since the automation must reach the PC from outside the LAN.
- **Validated:** policy 46/46; route live-verified through the real `websockets` stack (GET 200 + correct JSON, 401 on bad/missing token, both header + query auth); full regression gate **23/23**; `import main` clean. **LIVE-VALIDATED 2026-06-03** on a real iPhone (iOS 26.5): leave → `arm-scheduled` → 60s later `[presence] armed (presence: away)`; an accidental double-tap of the Leave shortcut produced `[presence] leave: arm already pending — no-op` (the idempotency/flap-damping rule firing in production — the best possible real-world test of the generation guard); arrive → `disarmed + welcomed` (`greeted: True`, the greeting heard from the desktop). **Three real-world findings from the bring-up, all client-side (no code change):** (a) **iOS autocorrect mangles the Bearer token in a header field** — the `Authorization` value is a plain text field (autocorrect ON), so a typed token was silently corrupted and 401'd, while the SAME token in the URL query worked (URL fields don't autocorrect). Fix: carry the token in `?token=` for the iOS Shortcut, or paste (never type) the header with Smart Punctuation off. (b) **arm-in-place triggers the M35 challenge** — testing at the desk, the 60s deferred arm fired while a person was on-camera, so security challenged (passed with the passphrase); a non-issue in real use, since by 60s after a genuine geofence exit nobody is in frame. (c) The iOS 26 **automation action picker is more limited than the standalone editor** (no "Get Contents of URL" inside the automation) — so build the HTTP call as a standalone shortcut and have the automation "Run Shortcut" it.
- **Reusable lessons:** (1) **Offload a solved primitive to the platform** — iOS geofencing beats anything a web app could do; we supply policy, not detection. (2) **A good seam pays compound interest, again** — a whole new presence surface was one policy module + one HTTP route + one callback, because the remote console was already the arm/disarm control plane. (3) **Damp the flap, don't chase the edge** — a deferred-and-cancellable action with a generation guard is the clean shape for any noisy external trigger (the same idea as `calendar_monitor`'s DedupeStore and `homelab_monitor`'s flap damping). (4) **Live-smoke the real dependency** — the GET-only gate was invisible to a fake-Request unit test.

---

## M71 — Webcam access from Discord (per-origin claw-back + armed-camera fix) — 2026-06-03

**"Jarvis, look through the webcam and tell me what's going on" — from Discord, while armed.** The natural companion to M70: once security auto-arms on a geofence exit, you want to *check in*. Scoped **Discord-only** (the channel already works off-network for free) and required posting the **actual photo**, not just a description.

- **The security decision — a surgical per-origin claw-back, not a boundary flip.** M48.2a put `camera_snapshot` (+ screen/file/shell/system/code/self-update) in a server-enforced `_RESTRICTED_DENY` for ALL remote origins. M71 re-allows **exactly `camera_snapshot`, for `origin="discord"` ALONE**, via a new `_RESTRICTED_ALLOW_BY_ORIGIN` map + `_effective_deny(origin) = _RESTRICTED_DENY - allow[origin]`. Both gates (the `stream_response` tools-list filter AND the `_execute_client_tool` deny-check) now compute the per-origin deny set; `origin` is threaded through `stream_response` + `_execute_client_tool`. Phone origins get **nothing** back; Discord still can't touch system/shell/file/screen/code/self-update. Least privilege held: deny-by-default, claw back one capability at a time, deliberately.
- **The prompt had to change too** (the non-obvious 4th moving part). The base `_REMOTE_RESTRICTED_ADDENDUM` explicitly tells Claude it CANNOT use the camera — so even with the tool offered, Claude would refuse. Added `_REMOTE_RESTRICTED_DISCORD_ADDENDUM` (selected when camera is clawed back for the origin) that *advertises* the webcam and notes the photo is shared into the channel. `build_system_prompt` gained `restricted_origin`.
- **Photo relay to Discord.** A per-turn `reply_image(bytes, media_type)` sink (text_queue grew 6→7-tuple, mirroring the M48.2b/Discord `reply_text` pattern). `on_image_captured` (the M31 console-thumbnail hook) relays the frame to it when set (Discord turns only). The Discord bot BUFFERS the frame in the per-message closure and `_post` attaches it (`discord.File`) to the first chunk of the SAME threaded reply — so photo + description land together (posting separately would race `_post`'s lazy thread creation and split them).
- **The privacy escalation, stated and accepted.** This uploads webcam images of an interior space to Discord (third-party, stored in channel history, visible to every allowlisted channel member). A deliberate, documented step beyond "the image never leaves the machine"; acceptable for a private channel, recorded here rather than buried.
- **The armed-camera collision (caught by the stage-2 live test — the M44.3 intersection).** The PRIMARY use case (check the webcam *while armed*) coincides exactly with the armed watcher holding the webcam open with a persistent capture (M44.3). A second handle from `camera_snapshot` got a black frame → Claude reported **"the camera looks covered."** Predicted before the test, confirmed by it. **Fix: the camera's OWNER serves the frame.** `SecurityWatcher.grab_frame_for_snapshot()` returns a frame from its persistent capture when armed (None when not), guarded by a new **RLock** so the snapshot grab and the watcher's `_watch_loop` don't race the one `cv2.VideoCapture` (RLock because `_grab_frame` calls `_release_cap` while holding it). `cameras.py` takes a dependency-injected `set_armed_frame_provider` (wired in main.py to the watcher); `execute_camera_snapshot` refactored into `_grab_own_frame` (disarmed path) + a shared black-guard/encode tail, with the provider tried first. Disarmed ⇒ provider returns None ⇒ own capture (camera free). This **resolves the M44.3-flagged "cooperative camera yield while armed" follow-on** — though not by yielding (release/reacquire would race); by *sharing* the owner's frame, which is cleaner.
- **Validated:** `scripts/remote_camera_test.py` **31/31** (claw-back logic per-origin, prompt-variant selection, denial paths — all hermetic; provider routing via a synthetic frame, no hardware). Full gate **24/24**; `import main` clean (7-tuple wiring sound). **LIVE-VALIDATED 2026-06-03** on the real Discord channel: stage 1 (disarmed) → photo + description in a thread; stage 2 (armed) → first "camera covered" (diagnosed the M44.3 collision), then after the fix `[cameras] using armed-watcher frame (camera owned by security)` + a real 1080p ~297 KB JPEG posted. Both disarmed and armed paths green.
- **Reusable lessons:** (1) **Re-opening a security boundary is a 4-part change, not 1** — the gate filter, the execution deny-check, the SYSTEM PROMPT (or the model won't use the tool), AND the result-relay plumbing. Miss the prompt and it silently no-ops. (2) **Claw back per-origin, deny-by-default** — `_RESTRICTED_DENY - allow[origin]` keeps the boundary intact and the exception auditable, vs. flipping a global flag. (3) **When two subsystems contend for one exclusive resource, let the OWNER serve it** — sharing the armed watcher's frame beat a release/reacquire yield (no race, no flicker). (4) **The stage-2 live test (the realistic condition, not the easy one) is where the real bug lives** — disarmed worked first try; the armed case (the actual use) exposed the collision. Predicted, then confirmed, then fixed in-session.

---

## M72 — Multimodal acoustic alerts: hear → look → describe → Discord — 2026-06-03

**"Jarvis, what was that?" answered before you ask.** When Jarvis hears a notable sound WHILE ARMED (M58), he now also looks through the camera, has Claude describe the scene, and pushes the **photo + a one-line description** to Discord — senses fusion: hear (M58) + see (M71's `grab_frame_for_snapshot`) + reason (Claude vision) + remote push. The session that shipped this also re-scoped acoustic awareness to three events and ran a full data-driven tuning pass off a real mic.

- **The feature.** `src/vision_describe.py` (NEW) — `describe_scene(jpeg, heard, *, api_key, model)`: a single non-streaming Claude vision call (image + what-was-heard → one butler-tone sentence). `notifications.send_discord_photo` (NEW) — generic caption+photo webhook push (the photo counterpart to `send_discord_message`). `cameras.encode_jpeg` extracted (shared). `SoundDetector` gained an `on_visual_alert` hook fired on its OWN thread (the ~2 s vision call mustn't block inference or the text push); main.py's `_acoustic_visual_alert` is GATED to armed via `grab_frame_for_snapshot` (returns None when not armed → no-op when disarmed, no camera light for every doorbell). Every step fail-soft. `scripts/acoustic_visual_test.py` 12/12.
- **Re-scoped to three (2026-06-03).** While armed, listen for exactly **knock, doorbell, glass breaking** — glass is the priority (it fires the photo alert so the event can be cross-checked against a separate camera). The other four (smoke/phone/timer/water) moved to `_OTHER_RULES` — defined, preserved, off; re-enabling any is a one-line move.
- **The debugging saga (the real story, and the lesson).** First test: the knock wasn't detected. Before guessing thresholds, added `JARVIS_ACOUSTIC_DEBUG=1` (logs the top-6 AudioSet labels + rms per window). The log was decisive: **`rms=0.000` on every window, top label "Silence"** — even while knocking. Not a classification problem; the acoustic stream was receiving *digital silence*. **Root cause: the mic's Windows Voice Focus / Voice Clarity enhancements** were gating non-speech to zero — and M58 had only ever been validated on the OLD webcam mic (2026-05-26), *before* the dedicated-mic pin (M67, 05-28). Setting the enhancement to **"Automatic"** (keeps voice clarity, passes events) let real audio through: knock 0.40–0.78, doorbell 0.69–0.95, glass 0.40–1.39.
- **The data-driven tuning, in two beats.** (a) The debug log then caught a *second* bug: the model hallucinates a class on near-silent windows (`FIRE knock score=0.43 rms=0.002`) — a false "someone's at the door" while armed. Real vs spurious separated cleanly by LOUDNESS (real ≥ 0.033, spurious ≤ 0.009), so added a per-rule `rms_floor` (the running-water volume guard, generalised). (b) Live data refined it: the **doorbell is the quietest real event (rms 0.033)** because Voice Focus attenuates non-speech even on Auto — so the floor was lowered 0.03 → **0.02** (in the gap, with margin below the real doorbell, leaning toward detection). And **"Breaking" was added to the glass aliases** (a faint break scored Glass 0.19 + Shatter 0.09 = 0.28, just missed; +Breaking clears 0.30) — but NOT "Chink, clink" (false-positives on normal glassware touching). `JARVIS_ACOUSTIC_RMS_FLOOR` env-tunable.
- **Validated:** gate **25/25**; `acoustic_visual_test` 12/12; `acoustic_alert_test` 12/12. **LIVE-VALIDATED 2026-06-03** — all three events fire on real sounds, the quiet stretch produces zero false fires (floor working), and the multimodal photo+description path runs (M71 armed-frame + Discord photo).
- **Reusable lessons:** (1) **MEASURE what the model hears before tuning thresholds** — the `rms=0.000` log killed the entire "tune the knock threshold" theory in one look; an hour would have gone into a knob that does nothing. The debug-log instrument paid for itself instantly (same discipline as M44.3's `leak_repro.py`). (2) **The OS audio stack is part of the pipeline** — a mic "enhancement" (Voice Focus) silently ate the input; a feature validated on one mic isn't validated on the next. (3) **Separate the gates by what they measure** — score = classification confidence, rms = loudness; the model hallucinates high-score on silence, so only a LOUDNESS floor separates real from spurious (score can't). (4) **Set knobs from the binding real case** — the floor is bounded below by the quietest real event (the Voice-Focus-attenuated doorbell at 0.033), not a round number. (5) **A debug flag that logs the model's raw view is the cheapest tuning tool there is** — ship it behind an env var.

---

## M73 — "Where can I watch X?" — TMDB watch-providers — 2026-06-05

**"Jarvis, where can I stream Dune?" → the streaming, rental, and purchase services for that title in your region.** A small, self-contained capability added the morning after the M70→M72 security/multimodal burst — high value-per-LOC, zero new dependencies, one session.

- **A new MODE, not a sibling tool — and that was a deliberate altitude call.** M66 (`get_person_info`) is a *separate* tool from `get_movie_tv_info` precisely because its input shape differs (a person *name* vs a *title*) — collapsing them would force Claude to disambiguate every call. "Where to watch" has the SAME input as the rest of `get_movie_tv_info`: a title in `query`. So it's a fifth `mode` (`providers`) alongside search/details/popular/similar, and routing stays unambiguous at the schema. (The M66 lesson, applied in the other direction: sibling tool when input shape differs; mode when it's the same.)
- **The endpoint + the region problem.** TMDB exposes JustWatch-sourced availability at `/{movie|tv}/{id}/watch/providers`. The payload is keyed by ISO-3166-1 **country code** — availability genuinely differs by country ("stream Dune" in the US ≠ the UK), so guessing one global answer would be wrong. `_region()` reads `TMDB_WATCH_REGION` (default `US`, uppercased — TMDB's keys are uppercase) using the same read-at-call-time pattern as `_api_key()`, so config.py stays untouched.
- **The build.** `_do_providers(query, key, media, region)` — a clean two-hop: `_search_results` → top match (resolved via `_search_results` *not* `_search_top`, so we keep the canonical title for the spoken reply) → `/watch/providers` → pull the region's lists. Categories map to spoken labels in read-out order: `flatrate`→**Stream** (the answer most people want first — included with a subscription), `free`→**Free**, `ads`→**Free with ads**, `rent`→**Rent**, `buy`→**Buy**; capped at `_MAX_PROVIDERS=6` per category. Three distinct empty-states, deliberately worded apart: region *absent* from the payload ("I don't have any streaming availability for X in DE"), region *present but no options* (just a JustWatch link → "isn't currently listed to stream, rent, or buy"), and no title match at all. Same never-raises contract as the rest of the module — missing key, no match, transport failure, malformed JSON all become voice-friendly strings.
- **Wiring.** `providers` added to the tool's `mode` enum + description; `query` is required for it (added to the `("search","details","similar","providers")` check); system-prompt item 4 in `llm.py` teaches the route ("where can I watch / stream / rent X"). `.env.example` documents `TMDB_WATCH_REGION`.
- **Validated:** `scripts/watch_providers_test.py` **22/22** — schema, missing-key, missing-query, no-match, both transport-failure hops, malformed JSON, the happy path (label mapping + read-out order + the resolved-id reaching the providers URL), region override (lowercase `gb` → proves uppercasing, and the GB list doesn't bleed US providers), region-absent, region-present-but-empty, free/ads categories, and the per-category cap. All hermetic (URL-keyed HTTP stub, no network). Full gate **26/26**. **FULLY LIVE-VALIDATED 2026-06-05** with a real `TMDB_API_KEY` — both the backend (direct tool calls: "Dune" → HBO Max + rent/buy; "The Last of Us" correctly resolved to the TV show; gibberish → clean no-match) AND voice-routing (spoken "where can I stream The Last of Us?", "Dune", and "Super Mario Galaxy Movie" all routed to `mode=providers` via the voice loop, `tmdb_tool=yes`; the no-subscription empty-state rendered correctly). **A nice fail-safe surfaced:** a clipped one-word follow-up ("What about DUN?" — STT truncated "Dune", and the prior turn had listed streaming *services*) made Claude ask for clarification rather than guess a tool call — it recovered cleanly on the next turn. Clarify-don't-hallucinate is the right behavior; left as-is per iterate-on-real-usage.
- **Reusable lessons:** (1) **Mode vs sibling-tool is an input-shape decision, not a "how related is it" feeling** — same input shape ⇒ a mode keeps routing unambiguous; different input shape ⇒ a sibling tool (M66). (2) **When data is region-keyed, make region a first-class, configurable input** — don't hard-code one country or read back a foreign list silently. (3) **Distinct empty-states deserve distinct wording** — "not available here" vs "not available anywhere right now" vs "no such title" are three different facts a voice user needs to tell apart.

---

## M74 — "How long to beat X?" — How Long To Beat — 2026-06-05

**"Jarvis, how long is Tears of the Kingdom?" → main-story / main+extras / completionist times.** The companion to the M22 RAWG games tool, finally cashed in — a fast capability picked the same session as M73.

- **A SEPARATE tool (`get_game_length`), not a 5th mode on `get_game_info` — and the rule that decides it.** M66/M73 established "mode vs sibling-tool is an input-shape decision." But that rule governs splits *within one backend* (TMDB modes, RAWG modes). Across backends the project's actual pattern is **one tool per data source**: RAWG→`get_game_info`, TMDB→`get_movie_tv_info`, Wolfram→`wolfram_query`, RSS→`get_news`. HLTB is a brand-new source (a different package, different failure modes), so it earns its own tool + module — even though its input (a game *name*) is the same shape as `get_game_info`'s. The two rules compose: **same backend + same input ⇒ mode; different backend ⇒ new tool.** (RAWG's `get_game_info` already returns a crude average `playtime`; HLTB's real main/extra/completionist breakdown is the value that justifies the new tool.)
- **De-risk FIRST (the M58 discipline).** HLTB has no public API — `howlongtobeatpy` scrapes their search endpoint, and such scrapers break when the site changes. Before building, installed it and ran live searches: data accurate (Elden Ring 60h main / 135h completionist, TotK 59/248, Hollow Knight 27/65), **but latency ~5.5 s/search** (their endpoint is slow) — noted as acceptable for an explicit "how long" question, not for a hot path. Also checked the **asyncio collision risk**: `howlongtobeatpy`'s sync `search()` spins its own event loop internally, and the agentic tool loop runs in a synchronous worker thread *with no running loop* (the same thread that calls `asyncio.run(...)` for TTS), so the sync call is safe — verified, not assumed.
- **The build.** `src/game_length.py` — `GAME_LENGTH_TOOL` + `execute_game_length_tool`, same never-raises contract as games/tmdb. The dependency is **lazy-imported behind a latching sentinel** (`_get_client` — mirrors the M69 speaker-id / M39 face-auth pattern) so a missing/broken HLTB install degrades to a clear "run pip install" message instead of breaking the import of `llm.py`. `_fmt_hours` rounds to the nearest half-hour and suppresses zero/missing playstyles; `_format` renders main-story / main+extras / completionist lines, falling back to the all-styles average when the breakdown is empty (a just-announced game), and reports "no completion-time data yet" when even that is absent. Best match chosen by HLTB's `similarity` score; the matched name is surfaced so a wrong-game match is catchable.
- **Wiring.** Imported into `llm.py`; one entry in `_CLIENT_TOOLS` (telemetry label `game_length`); added to the `tools` list; system-prompt item **3b** (paired with item 3 `get_game_info`, mirroring the 4/4b movie/person pairing) teaches the split — facts ABOUT a game → `get_game_info`; TIME to finish it → `get_game_length`. NOT in `_RESTRICTED_DENY` (read-only public data — the phone may ask). `howlongtobeatpy>=1.0.21` added to `requirements.txt`.
- **Validated:** `scripts/game_length_test.py` **29/29** — schema, `_fmt_hours` rounding (incl. the 135.67→135.5 nearest-half case + singular "1 hour" + half-kept + zero/None/garbage suppression), missing-name, package-unavailable, empty/None results, search-exception, the full-breakdown happy path + read order, zero-playstyle suppression, all-styles fallback, no-data-at-all, and best-match-by-similarity. All network-free (stubbed `_get_client`). Full gate **27/27**. **Live formatting confirmed** via the real executor (Hollow Knight / TotK rendered cleanly; gibberish → clean no-match). Live voice-routing pending a spoken query.
- **Reusable lessons:** (1) **The mode-vs-tool rule has two axes** — input shape (M66/M73) AND backend; a new data source is a new tool even when its input shape matches an existing one. (2) **De-risk a scraping dependency before building on it** — confirm it still works against the live site AND measure its latency; HLTB scrapers are a known breakage class. (3) **Check the concurrency contract of a sync library before calling it from an async app** — a library that runs its own event loop is fine *only* from a thread with no running loop; verify which thread the call lands on. (4) **Lazy-import a flaky dependency behind a latching sentinel** so its failure is a friendly message, not a dead import chain.

---

## M75 — `stop_service` + `start_service` — completing the SRE verb set — 2026-06-05

**"Jarvis, stop the Spooler service."** The two SRE verbs that round out `system_control`'s mutating set — alongside the existing `restart_service` (M40), `flush_dns` (M40), `dhcp_cycle` (M42), `kill_process` (M23). Now Jarvis can stop/start/restart a Windows service, the realistic toolkit for the 90% of single-user-desktop service troubleshooting.

- **Rule-of-three extraction (the M43 philosophy, on the next tool that needed it).** A third service verb made restart/stop/start identical except for the PowerShell cmdlet, the success/timeout wording, and the confirmation gerund. So the M40 `_do_restart_service` was generalised into a shared `_do_service_action(action, target, confirmed)` driven by a `_SERVICE_CMDLETS` map — `(cmdlet, past-tense verb, gerund, short verb)` per action. All the load-bearing parts (regex validation, confirmation gate, admin gate, subprocess plumbing, 30 s timeout, error passthrough) now live once. The refactor is **behaviour-identical for restart_service** — verified by the regression rows in the new test (same confirmation text, same "Restarted X." success, same error wording).
- **`Stop-Service` deliberately gets NO `-Force`** — the least-privilege call. A service with running dependents fails with a clear "Cannot stop … because it has dependent services" message that we surface, rather than silently stopping the dependents too (a bigger blast radius the operator should opt into explicitly — "Jarvis stays Jarvis, not Ultron"). Same reason `Start-Service` is left to surface its own "dependency failed to start" errors. A future `-Force` variant would be its own confirmation-gated decision.
- **Same gates as every mutating verb, enforced server-side:** `confirmed=true` (the tool returns a "needs confirmation" string on the first call, never acts) AND running as Administrator (`_IS_ADMIN`, cached at import) AND a service name matching `^[A-Za-z0-9._-]{1,128}$` (validation IS the boundary — rejects shell metachars before they reach PowerShell). Both verbs are in `system_control`, which is wholesale in `_RESTRICTED_DENY` — the phone can't touch any of them.
- **Closed the long-standing test gap.** The M40/M42 service/network verbs shipped with **no unit test** (the "gate with no enforcing test" pattern this project keeps flagging). M75 adds `scripts/system_control_test.py` **38/38** — schema, the confirmation gate (and that NO subprocess fires when unconfirmed), the admin gate, name validation (empty + shell-metachar rejection), the success path per verb (asserting the right cmdlet reaches the PS script + no `-Force` + the name is interpolated), stdout passthrough, the failure path (incl. the dependent-services message we deliberately don't `-Force` past), timeout, and spawn-failure. All subprocess-free (monkeypatched `subprocess.run` + `_IS_ADMIN`).
- **Wiring.** Enum + tool description + `target`/`confirmed` descriptions updated; dispatch routes `action in _SERVICE_CMDLETS` to the shared helper. System-prompt item 9 left as-is (it's a deliberately abbreviated teaser — it already omitted the M40/M42 SRE verbs; the tool's own description is the authoritative surface Claude sees). Full gate **28/28**.
- **Reusable lessons:** (1) **The third instance is when you extract** — two near-identical verbs is tolerable duplication; the third is the signal (the M43 `http_util` rule, applied to a PowerShell-cmdlet family). (2) **A refactor of a working path must be behaviour-identical — and you prove it with a regression row**, not by eyeballing. The new test asserts restart_service's exact wording survived. (3) **Least-privilege means NOT adding `-Force`** — surface the dependent-services error and let the operator opt into the bigger blast radius; don't make "it worked" mean "it quietly took down three other services." (4) **A new sibling verb is the cheap moment to backfill the missing test for the whole family.**

---

## M76 — `what_did_you_hear` — closing the acoustic recall loop — 2026-06-05

**"Jarvis, what was that?"** The acoustic arc had a gap: M58 *detects* sounds and M72 *reacts* to them (look + describe + push to Discord while armed) — but there was no way to **ask** Jarvis what he heard. M76 closes the loop: detect → react → **recall**. After an alert (or any time), "what did you just hear?" / "did you hear something?" returns the recent soundscape.

- **A thread-safe rolling buffer in `SoundDetector`.** Each salient inference window (the model already computes its top labels for the debug path) records `(monotonic_ts, dominant_label, score, rms)` into a bounded `deque`; each fired monitored alert records `(ts, rule_name)` into a second deque. Both are written by the inference thread and read by the turn-worker thread, so access is guarded by a `threading.Lock`. Silence / near-quiet windows are skipped (top score < 0.15 or label == "Silence") so the soundscape reflects real events, not the floor. Buffers clear on `activate()` so a "what did you hear?" right after re-arming never reports a prior session's sounds.
- **The pure formatter is the testable core.** `recent_sounds_summary()` snapshots the buffers under the lock, then defers to a module-level pure `_summarize_sounds(active, recent, fires, now, max_age)` — no threads, no model, no audio. It handles three honest states: **off** ("acoustic awareness is off, I'm not listening"), **quiet** ("nothing notable, it's been quiet"), and a **summary** — fired alerts first (high-confidence, with recency), then the general soundscape aggregated by label (peak score + most-recent age, capped at 6 with a "+N more" tail). Age phrasing via `_ago` ("just now" / "~40s ago" / "~3m ago").
- **Decoupled via the registration-singleton pattern** (`self_status.register` / `good_night.register_security_getter`): `register_detector(sound_detector)` in main.py; the tool reads whatever's registered. The tool is **always available** (not gated on acoustic being active) — asked while off, it honestly says so. Read-only → NOT in `_RESTRICTED_DENY` (the phone may ask what the room sounded like). Optional `minutes` param (default 3, clamped 1–15).
- **Validated:** `scripts/what_did_you_hear_test.py` **31/31** — schema, `_ago` phrasing, the three states, peak-vs-recent label aggregation, the age-window filter, the item cap + overflow tail, and fired-alert ordering + name humanisation, plus the executor's registration / minutes-clamp / fail-soft paths. All thread-free and model-free (the pure formatter + a fake detector). Full gate **29/29**. **LIVE-VALIDATED 2026-06-05**: a `glass_break` fire + "what did you hear?" returned the fired alert ("glass break about 20 seconds ago"), the soundscape ("shattering and clinking"), AND an unmonitored "animal nearby" — the dominant-label-per-window design surfacing a sound the alert rules would never flag (`acoustic_recall=yes`, clean).
- **Reusable lessons:** (1) **Cross-thread shared state wants a lock + a pure formatter** — the inference thread writes, the turn thread reads; snapshot under the lock, then format in a pure function you can unit-test without either thread. (2) **A recall tool needs an honest "off" state** — "I'm not listening" is a real and useful answer, distinct from "it's been quiet." (3) **Record the dominant label of every salient window, not just fired alerts** — it captures *unmonitored* sounds (an animal, music, speech) the alert rules would never surface, which is exactly what "what was that?" is asking. (4) **Reuse work already being computed** — the debug path already ran the argmax; recording piggybacks on it for near-zero cost.

---

## M77 — Severe-weather proactive alerts — Jarvis watches the sky — 2026-06-05

**"Sir — the National Weather Service has issued a Severe Thunderstorm Warning, until 8 PM. With no UPS on the desktop, you may want to shut it down before the power goes."** A proactive monitor personalised to a documented constraint of this deployment: the desktop has **no UPS**, the BIOS won't auto-power-on, and shutdowns for known storms are manual. Until now that depended on a human watching the forecast; M77 makes Jarvis watch it.

- **Recombination of the proven monitor pattern.** `src/weather_alerts.py`'s `WeatherAlertMonitor` is a defensive daemon poll loop (stop Event, every poll wrapped so one bad fetch can't kill the thread) over the **NWS active-alerts feed** (`api.weather.gov`, free, no key, US only). Voice out via the WASAPI-safe `_announce` (tagged ⛈, distinct from 📅/🖥/🔔/🚨/⏰); phone push via `send_discord_message`. Same shape as M62.2 calendar / M56 homelab.
- **Reused, not rebuilt.** Dedupe across polls AND restarts via `calendar_monitor.DedupeStore` — **imported, not duplicated** (it's a generic path+retain+thread-safe atomic-save set, already tested). This is its *second* consumer, so per the rule-of-three it stays put; a comment flags the extract-to-shared-util trigger if a third monitor needs it. Geocoding reuses `weather._geocode` (handles the "City, ST" comma case Open-Meteo otherwise misses). Direct `httpx` (not the shared `http_util`) — the wolfram.py precedent — so we can send the NWS-requested `User-Agent`.
- **Pure cores, isolated for tests.** `_should_announce(alert, announced, severities)` (severity-floor + dedupe), `_parse_features` (NWS GeoJSON → `WeatherAlert`), `_power_risk(event)` (the keyword set that earns the no-UPS shutdown nudge — thunderstorm/tornado/hurricane/wind/winter, NOT flood/heat/rip-current), `_fmt_local_time` (ISO+offset → "8 PM"/"8:30 PM"), `_alert_speech`. All no-I/O, all tested.
- **Defaults + the proactive/reactive pair.** DEFAULT ON when `JARVIS_HOME_LOCATION` is set (the calendar-monitor convention — the unprompted warning IS the value); kill switch `JARVIS_WEATHER_ALERTS=0`. Only **Severe + Extreme** announced proactively (advisories like a Rip Current Statement would be noise); floor tunable via `JARVIS_WEATHER_ALERT_SEVERITY`. Poll cadence `JARVIS_WEATHER_POLL_SECONDS` (default 600 — a good NWS citizen; warnings have 30-60 min lead). Companion **on-demand `get_weather_alerts` tool** (the reactive half, like `homelab_status` to the homelab monitor): "any weather alerts?" reports ALL active alerts (since it was asked for), most-severe first; reads `JARVIS_HOME_LOCATION` itself or an explicit `location`; read-only, NOT in `_RESTRICTED_DENY`.
- **Wiring.** `WeatherAlertMonitor` constructed + `activate()`d in main.py (no-op when unconfigured); threaded into `_register_status` ("Weather alerts: active/off") and `_shutdown_subsystems`. The on-demand tool imported into llm.py (dispatch + tools list + system-prompt item 2b, paired with get_weather). Geocoded lazily on first poll + cached.
- **Validated:** `scripts/weather_alerts_test.py` **44/44** — parse, severity set, decision, power-risk (incl. "wind" matching "High Wind" but not flood/heat), local-time formatting (built in local tz so it's machine-independent), speech + the no-UPS nudge, `fetch_active_alerts` (httpx monkeypatched: 200/500/bad-JSON/transport-error), and the executor (geocode + fetch monkeypatched: no-location/miss/error/empty/sorted-list). Full gate **30/30**. **Live-confirmed against the real NWS API** with two placeholder coastal/inland cities: one returned a real "Rip Current Statement [Moderate] (until 8 PM)", the other came back clean — geocode + parse + severity + end-time all correct. Live proactive fire pending an actual severe storm at the configured location.
- **Reusable lessons:** (1) **The best features are personalised to a known constraint** — a generic "weather alerts" tool is fine; "shut the desktop down, there's no UPS" is *Jarvis*. A recorded fact about the deployment turned a commodity API into a bespoke capability. (2) **Reuse a tested generic across modules before duplicating or extracting** — DedupeStore was imported (2nd consumer, rule-of-three says don't extract yet). (3) **A monitor wants a proactive half AND an on-demand half** — the poll loop warns you; the tool answers "any alerts?" — same data, two surfaces (the M56 homelab_status pairing). (4) **Build tz-dependent test inputs in the local tz** so the assertion is machine-independent.

## QoL consolidation pass (M68→M77) + process_question decomposition — 2026-06-09 (PR #22)

A no-new-features hardening/cleanup pass after the M68→M77 sprint — the second QoL pass (the first, M1→M67, was 2026-05-29). Method: **regression net first** (baseline `scripts/run_all_tests.py` 30/30 green), then **five parallel read-only auditors**, each owning a subsystem cluster (~4–5k LOC) with a rubric tuned to this project's contracts (fail-soft, WASAPI thread-affinity, the cooperative speech gate, the restricted phone/Discord boundary) so intentional design wasn't flagged as a bug; then fix in risk order. Full write-up in [docs/CODE_AUDIT.md](CODE_AUDIT.md). Gate **30/30** after.

- **Two confirmed bugs, both in the brand-new M77 weather alerts** — the kind a single live test couldn't hit: (1) **multi-day NWS warnings re-announced after 24h.** `WeatherAlertMonitor` reused `calendar_monitor.DedupeStore` with its default 24h retention — correct for calendar events (an ID encodes `start_local`, can't recur once started), wrong for weather: an NWS warning keeps a *stable id for its whole life* and warnings run multi-day, so the dedupe pruned a still-live storm warning → a repeat no-UPS shutdown nudge ~24h in (e.g. 3 AM). Fixed with `_DEDUPE_RETAIN_HOURS = 168`. (2) **proactive severity match was case-sensitive** (Title-case membership) while the on-demand path already lowercased — a casing drift would silently take the *proactive* monitor dark (the worse failure mode). Compare case-insensitively. +2 regression tests (44→46). **The reuse lesson:** a reused tested component is sound, but its *defaults are domain-specific* — DedupeStore's 24h default was right for its first domain and wrong for the second; check inherited defaults against the new domain.
- **Four hardening fixes:** an empty LLM reply left a Discord/phone user in **total silence** (now a short bilingual ack); the shared `http_util` GET helper **didn't follow redirects** (the M47 landmine — now `follow_redirects=True` for *every* consumer, killing the class, + the last bare `http://` Ars feed → https); `pc_diagnostics` `Get-WinEvent .Substring` threw on a null Message (guarded); `plex_actions` SSH auto-retry could **double-fire an irreversible `empty_trash`** on a mid-exec drop (`run(retry=False)` for mutating actions).
- **~10 polish items:** `good_night` undercounted in status_reminders; `summarize_session` dumped attachment base64 into the prompt; stale news category message; "0 MB" cap message; Discord clamp drift (1900 vs 2000); dead `_http` `_status` param; no-op try/except; false "no lock needed" + stale floor comments; docstring drift.
- **`process_question` decomposition** (separate commit `ceeaf17`) — the headline maintainability item, a ~360-line god-method on the hot path (it grew back since the M29-era TurnRunner extraction as M68/M48.2b/M71 layered on). Decomposed into four named, explicitly-parameterized `TurnRunner` methods — `_emit_remote_reply`, `_begin_turn`, `_stream_and_speak(...) -> bool` (the intertwined heart: gate set/clear + stream + TTS + barge-in monitor + error→apology, with the `pc_silent`/`barge_enabled`/`speaking_aloud` axes now explicit params), `_finalize_turn(...) -> bool` — **each extraction gated individually** by `turn_runner_test.py` (35/35) before the next. Bodies lifted verbatim (zero behavior change); all run under the caller's non-reentrant `self._lock` and never re-acquire it. `process_question` is now ~15 lines of orchestration. File +68 lines — decomposition buys named, independently-testable phases, not a smaller file.
- **Reusable lessons:** (1) **a reused component's defaults are domain-specific** — verify inherited defaults fit the new domain (the DedupeStore retention bug). (2) **Apply a robust mitigation to every instance of a hazard class, not one site** — `follow_redirects=True` on the shared helper kills the M47 landmine everywhere, like pinning every `web_search`/`web_fetch` off the dynamic-filtering version. (3) **Decompose a hot-path god-method incrementally, gating after each cut** — four small extractions each verified by the test net beats one big rewrite; the win is named/testable seams, not line count. (4) **Audit the newest code hardest** — both real bugs were in M77, shipped "live-confirmed" but only against states a single live run hits.

## Memory-recall session: NBA feed + the recall-gap diagnosis & fix (B+A) — 2026-06-11

Triggered by a real failure: Jarvis had **no recollection of a prediction he had made himself** (he picked a team to win a playoff series; days later he said he had "no record" of it). Investigation found it was **not a bug, not retention, not an unclean closeout** — the conversation was saved perfectly in BOTH the raw transcript AND a summary. Two compounding *recall* design limits:

- **Recall-window scroll-off (primary).** Only the last `MEMORY_RECALL_COUNT` *summaries* are injected into the system prompt; the prediction conversation was summary #163 of 181 (~18 sessions back), far outside the window. **Fix A:** default 10→25 ([config.py](../src/config.py)).
- **Lossy summarization (secondary, would bite even with a wider window).** The summarizer's OMIT rules (strip stale scores/values to prevent memory pollution) also stripped Jarvis's *own predictions*, treating "Jarvis predicted X in six" like a stale score. **Fix B:** taught the summarizer the dividing line — a looked-up VALUE goes stale (omit), a STANCE Jarvis took never changes (keep) ([memory.py](../src/memory.py) `_SUMMARIZER_SYSTEM`). Live-verified: the exchange now summarizes as "…Jarvis predicted [team] would win in six games" while a pure weather/score lookup still strips 84°/100-95.
- **Also shipped:** NBA added to the Sports briefing feed (ESPN NBA RSS verified 200/no-redirect per the M47 landmine; `nba`/`basketball` voice aliases; the four league-enumerating docstrings synced to NBA/NHL/NFL/WWE) ([news.py](../src/news.py)).
- **Honest caveat:** B only helps conversations summarized *from now on* — it can't rewrite the already-lossy summary #163. That's what motivated C (below): reach the verbatim transcript, where nothing is stripped.
- **Reusable lessons:** (1) **"he forgot" has three independent failure points** — saved-but-scrolled-out (window), saved-but-lossily-summarized, never-saved (unclean shutdown). Diagnose which before "fixing memory". (2) **A pollution guard can over-reach** — the OMIT rule correctly kills stale facts but wrongly killed durable stances; the fix is a sharper distinction, not removing the guard.

## M78 — `recall_conversation`: full-text search over past transcripts (C) — 2026-06-11

The structural fix the B+A caveat pointed to, and the episodic-memory analog of M45's `knowledge_search`: a `recall_conversation` tool that searches the **verbatim** text of past conversations, so "what did you predict for the Finals?" reaches the actual exchange instead of relying on a lossy 10-item summary window. New [src/conversation_recall.py](../src/conversation_recall.py).

- **Direct scan, NOT an index (the load-bearing decision).** Unlike the hand-authored knowledge corpus (rare changes → a DROP+rebuild FTS5 index is cheap), transcripts are **append-only and grow every turn**. An index would force rebuild-on-startup (today's turns unsearchable until restart) or incremental-sync (the bug-class knowledge.py deliberately avoids). A direct scan of the per-day `sessions/*.jsonl` is **always-current** ("what did we just discuss" works) and microseconds at personal scale. Keyword match + crude prefix-stemming (predict↔prediction, final↔finals) + score-then-recency ranking; returns the matching user↔Jarvis EXCHANGES with relative timestamps.
- **Retention raised — raw is now the searchable memory.** `RETAIN_RAW_DAYS` default 30→365, and `prune()` treats `0` as keep-forever. The 30-day prune had *already* deleted everything before ~May 12 (only 29 days of raw existed); extending it now is what stops the next month aging out. Text is cheap (a year ≈ a few MB).
- **Allowed remote** — read-only of the user's own data, like `knowledge_search` and the episodic summaries that already feed the phone; NOT in `_RESTRICTED_DENY`.
- **Validated:** `scripts/conversation_recall_test.py` 18/18 (hermetic, temp LOCALAPPDATA), full gate 31/31. **Live-validated against real history** — surfaces the genuine prediction exchange, and pointedly captured *that morning's* "I have no record of that" failure verbatim.
- **Reusable lessons:** (1) **match the storage strategy to the data's mutation profile** — an index fits a slowly-changing corpus; an append-only, freshness-critical store wants a direct scan. (2) **Raw transcripts were write-only; a recall tool makes retention a feature, not just cleanup** — revisit prune policy when the data gains a reader.

## M78.1 — Semantic recall (hybrid keyword + embedding) — 2026-06-11

Cashed the RRF fusion seam C left open: `recall_conversation` is now HYBRID — keyword fused with local-embedding cosine, the M46 shape. The vector pass catches paraphrases keyword structurally misses (live-confirmed: "your call on the basketball championship" → the Finals exchanges, no verbatim match).

- **Shared embedder extracted** ([src/embeddings.py](../src/embeddings.py), NEW). The SentenceTransformer (all-MiniLM-L6-v2, ~80 MB + torch) became a single lazily-loaded singleton when recall became the *second* consumer — a **shared-expensive-resource** extraction (the http_util judgement), not premature abstraction: two independent loaders would double it in RAM whenever both tools run. `knowledge.py`'s `_get_embedder`/`_embed` now thin-delegate (call sites byte-identical; M46 behaviour unchanged, gate-confirmed + a live `knowledge_search` smoke).
- **No persistent embedding index.** Transcripts are append-only, so the vector pass embeds a **bounded candidate set at query time** (keyword hits ∪ a recent-history window, capped) — recall stays always-current and per-query embed cost stays bounded. A cosine **floor (0.30)** keeps the recent-window candidates from fusing in as low-similarity noise (recall over a large history wants precision where the small curated knowledge corpus wanted recall). Embeddings absent ⇒ empty vector list ⇒ clean keyword-only fallback.
- **Validated:** test suite 18→22 (semantic-catch, the floor, keyword-only fallback, RRF math — all hermetic via a deterministic concept-embed stub; keyword tests force embeddings off so they stay fast/deterministic). Full gate 31/31. A **full-corpus embedding cache** (for paraphrases of OLD out-of-window exchanges) is the measured follow-on if real use shows that gap — the M45→M46 discipline, repeated.
- **Reusable lessons:** (1) **share a heavyweight model at the second consumer** — a resource concern justifies extraction earlier than the rule-of-three for pure logic. (2) **the append-only property that made a keyword index awkward makes embed-at-query-time clean** — same constraint, opposite implication for the two retrievers.

## M78.2 — "You called it" — proactive prediction follow-ups — 2026-06-11

The proactive complement to recall: Jarvis circles back on his own sports predictions once they're decided — "Following up, sir — I predicted [team] in six; the other side took it, I got that one wrong." New [src/predictions.py](../src/predictions.py).

- **Design:** CAPTURE = retrospective **mining** (a pass re-reads recent transcripts via `conversation_recall._collect_exchanges`; an LLM extracts the falsifiable sports predictions Jarvis asserted as *his own pick*) — robust, zero per-turn cost, and it **backfills** (picked up the real June 3 call immediately). SURFACE = the **morning briefing** (a fail-soft `_predictions_section`) — daily cadence, proactive without the embarrassment risk of a real-time interrupt on a misread outcome. SCOPE = **sports only** (cleanly resolvable, web-checkable).
- **Conservative resolution.** A web-search-backed resolver marks a prediction resolved ONLY when confident — and treats a *specific* claim as resolved-incorrect once it's mathematically impossible (e.g. "in six games" at a 3-1 series deficit), not only when the event fully concludes. Bounded: max 3 checks/cycle, 12h re-check throttle, due when the estimated date arrives OR the prediction has been pending long enough (the estimate is the miner's guess; outcomes resolve early). Both LLM steps are dependency-injected so store/dedupe/due-window/resolution/surfacing/format logic is unit-tested without the network.
- **Validated:** `scripts/predictions_test.py` 35/35 (hermetic: temp store + stub miner/resolver). Full gate 32/32. **Live-validated end-to-end against real history** — mined the June 3 prediction (correct made-date after a fix: the miner now returns the per-prediction date from the transcript `[ts]`, not the batch max), the resolver caught the mathematical elimination (opponent leading 3-1) and marked it wrong, briefing follow-up rendered. Left resolved+unsurfaced so the next "good morning" delivers the first real callback. Off via `JARVIS_PREDICTION_FOLLOWUPS=0`.
- **Reusable lessons:** (1) **mine, don't rely on capture-time logging, when you need backfill + robustness** — a tool Claude must remember to call is probabilistic; a transcript-mining pass catches everything and works retroactively. (2) **a prediction can resolve before the event ends** — its specific claim (a margin, a game number) can become impossible earlier; encode that for sports. (3) **conservatism over immediacy** — a late "you called it" beats a wrong one; the resolver defaults to UNRESOLVED.

## M79 — Quiet hours (Do-Not-Disturb for proactive announcements) — 2026-06-11

By M78 Jarvis had ~7 sources that speak unprompted and no interruption policy — nothing stopped a routine disk-space warning firing at 3 AM. M79 adds one at the single announce chokepoint ([main.py](../main.py) `_announce`). New [src/quiet_hours.py](../src/quiet_hours.py).

- **Safe-default policy — deferral is opt-IN.** During the configured window, only **routine status-monitoring** announces (homelab 🖥) are held back; everything else **pierces**. So a new announce source pierces by default and is silenced only once *explicitly* classified routine — we can never accidentally mute something important. Pierce set: security 🚨, reminders ⏰ (explicit user intent), severe weather ⛈, acoustic 🔔 (armed-only), calendar 📅 (the user's own events), presence 🏠. **The label each caller already passes IS the severity signal — zero caller changes.**
- **Deferred ≠ lost.** Held-back announces are recorded (`deferred_announces.json`, atomic) and surfaced as a "while you were away" line in the morning briefing (the prediction-followup surfacing pattern), then cleared; stale (>18h) dropped, store capped.
- **Fail-soft throughout** — any config/clock/store oddity falls through to SPEAK (never swallow an announce on a bug). Overnight windows (start > end) wrap midnight. Off by default (`JARVIS_QUIET_HOURS` unset ⇒ byte-identical pre-M79 behaviour); set e.g. `22:00-08:00`.
- **Validated:** `scripts/quiet_hours_test.py` 31/31 (hermetic: env-set window, temp store, fixed `now` — no clock mocking), briefing catch-up live-smoked end-to-end. Full gate 33/33.
- **Reusable lessons:** (1) **add the interruption policy before adding more interrupters** — N proactive sources with no DND is debt; the chokepoint is the place to pay it down. (2) **make silencing opt-in, not opt-out** — the safe default for a DND classifier is "pierce unless explicitly routine," so a forgotten classification fails loud, not silent. (3) **defer-and-digest beats suppress** — a held announce that resurfaces in the briefing loses nothing; outright suppression loses information.

## M80 — Identity-aware replies: Jarvis addresses the recognized speaker by name — 2026-06-12
Cashes in the M69 speaker identification that was computed every turn and then **thrown away** (used only by the voice-lock gate). The recognized speaker now threads into the LLM call so Jarvis knows WHO it's talking to.
- **Per-turn speaker context as a SECOND, un-cache_controlled system block.** The speaker changes turn-to-turn (Speaker A → Speaker B → Speaker A); putting identity in the cached prefix would churn the prompt cache on every handoff. `src/llm._speaker_context_block(name, lang)` (pure, tested) is appended after the cached block — ~20 fresh tokens per turn, never a full-prompt cache miss. `stream_response` gained `speaker_name`/`speaker_lang`.
- **The `"you"` sentinel → `cfg.user_name`.** The primary was enrolled under the literal name "you" (`JARVIS_USER_NAME` unset). Resolved at the main.py layer (`speaker_name = cfg.user_name if _who.name == "you" else _who.name`), so setting `JARVIS_USER_NAME` fixes the display name with NO voice re-enrollment.
- **Language stays a SOFT hint** (non-English speakers only) subordinate to Whisper's per-turn detection — zero regression to the language/TTS path. A hard override (re-transcribe a recognized speaker's clip in their language on a low-confidence mis-detect) is the documented follow-on; it needs faster-whisper's `language_probability` plumbed through `Transcript`.
- Voice path only (remote/unrecognized turns add no block). `process_question` → `stream_response` threaded.
- **LIVE-VALIDATED 2026-06-12** — with a second speaker enrolled (`enroll voice [name]`), the log confirmed `[speaker] recognized [primary] (score=0.91)`. `scripts/speaker_context_test.py` 23/23; gate 34/34.
- **THRESHOLD — settled at 0.72 after a CORRECTED diagnosis (the 0.62 step was WRONG).** A first pass dropped `JARVIS_SPEAKER_THRESHOLD` 0.70→0.62 on a fail-open argument: "the primary's own turns dip to 0.61, so 0.70 locks him out ~16% of the time — tune against the worst clip." That reasoning was built on **contaminated data**. A later log review, with ground truth supplied for each clip, showed those low "recognized" scores were **background media audio FALSE-MATCHED as an enrolled speaker at 0.63–0.70** — the transcripts were clearly a streamed video (channel-membership and sponsor patter), not any enrolled person. At 0.62 the voice-lock gate let that media through; worse, M80 would have addressed it by an enrolled speaker's name and M82 would have tagged it into memory under that name. The REAL bands: legitimate Jarvis-directed turns score **0.77–0.95** (~0.75 across the room); media false-matches top out **~0.70**. So the threshold must sit ABOVE the media ceiling and BELOW the across-room floor ⇒ **0.72** (live-confirmed rejecting the media, and chosen over 0.70 to clear the single 0.70 media hit). Note `JARVIS_SPEAKER_GATE` is the on/off switch for the gate itself (a bool), NOT a numeric — the threshold is the only tuning dial.
- **Lessons:** (1) **a feature can already be half-built** — M69 computed the identity; M80 just stopped discarding it. (2) **a second uncached system block is the right home for per-turn-varying context** — keeps the big prompt cached. (3) **NEVER tune an identification threshold on SCORES alone — correlate each score with its TRANSCRIPT.** A "recognized" impostor (background media) RAISES the floor the threshold must clear; tuning fail-open on unlabeled scores *enshrines the impostor*. The log shows a score but not whether the audio was a real enrolled person or a television — and that difference is the whole ballgame. (The first write-up's "tune against the worst clip" was right in spirit but applied to the wrong samples.)

## M81 — Armed intrusion-by-voice: human speech while armed fires a look + push — 2026-06-12
Extends the M72 multimodal acoustic alert. While security is **armed**, human speech in the monitored space fires the same look + photo + Claude-description → Discord as a glass break. Closes the acoustic-security arc: M58 hears sound *types* → M72 reacts → M81 reacts to a *person* being present by voice.
- **A new ARMED_ONLY `voice_while_armed` rule** (aliases `Speech` + `Conversation` + `Shout`, all verified against the real AudioSet label set). Rides the existing `_fire` → `on_visual_alert` pipeline — zero new alert plumbing.
- **`armed_only` is the load-bearing flag.** Acoustic awareness runs both armed AND disarmed (the independent tray toggle), so the rule MUST only count windows while armed — normal conversation in the disarmed state is never flagged. Added `armed_only` to `ClassRule`; `SoundDetector` takes an `is_armed` getter (`security_watcher.is_armed`). **Fail-safe:** a missing/raising getter ⇒ treated as NOT armed (never fire an intrusion alert unless armed is confirmed).
- **FP defenses:** `sustain=2` (~4 s sustained speech beats a one-off blip), the RMS floor (rejects faint media/ambient speech bleed from an adjacent space + the model's silence-hallucination), 120 s cooldown; Jarvis's own announces/replies are already gated out of inference by the speech gate (`_is_announcing`), so he can't self-trigger on his reply.
- **Refactor:** the per-window gate (threshold/floor/volume-guard/armed) extracted to a pure, tested `_rule_window_passes`. Threshold env-tunable (`JARVIS_ACOUSTIC_VOICE_THRESHOLD`, default 0.45); disable via `JARVIS_ACOUSTIC_DISABLE=voice_while_armed`. Marked `experimental` — wants a live tune.
- `scripts/intrusion_voice_test.py` 17/17; gate 35/35. **LIVE-VALIDATED 2026-06-12** — armed + spoken speech in the room: the full pipeline fired (`FIRE voice_while_armed score=0.85 rms=0.044` → spoken announce + Discord text [HTTP 204] + photo+description [HTTP 200, 290 KB]), zero false positives, the 120 s cooldown gave exactly one fire despite continuous talking, and the announce didn't self-capture (`[stt] capture aborted — PC is speaking`). **The `DEBUG=1` data confirmed the defaults need NO tuning:** real speech scored 0.59–0.88 (rms 0.013–0.046), quiet windows 0.04–0.08 (rms 0.000), and the single silence-hallucination (`Speech=0.38 @ rms 0.004`) was rejected by BOTH the 0.45 threshold and the 0.02 RMS floor — a clean gap. Expected real-world FP source: a television/radio/smart-speaker left on while armed reads as Speech and WOULD fire (by design — any human voice in the monitored space); a speaker-ID-filtered intrusion rule (ignore enrolled voices, needs always-on ambient speaker-ID) is the follow-on if it ever bites. *(M87 later added a dedicated RMS floor for this rule after ambient speech bleed from an adjacent space proved to be the dominant false-fire source — see M87: loudness, not classifier score, is the discriminator.)*
- **Lessons:** (1) **an armed-only rule needs the armed signal in the detector** — coupling acoustic to armed mode (M58) isn't enough; firing must check it. (2) **fail-safe = don't fire when you can't confirm the precondition.** (3) **verify model aliases against the real label set before shipping a threshold** (the M58/M72 discipline).

## M82 — Speaker-tagged conversation memory: recall by who spoke — 2026-06-12
Each conversation turn is tagged on disk with WHO spoke it (the M80/M69 identity), so `recall_conversation` can answer "what did [speaker] ask me to do?". Built on the identity M80 proved live — additive and fully backward-compatible.
- **Storage** (`src/memory.py`): `record_turn` gained an optional `speaker` arg, written onto the USER jsonl record only. `None` ⇒ no field (old/typed/remote turns unchanged); assistant records (always Jarvis) are never tagged.
- **Threading** (`main.py`): the M80 `speaker_name` flows `process_question → _finalize_turn → record_turn`, so the persisted turn carries the same name Jarvis addressed aloud.
- **Recall** (`src/conversation_recall.py`): `_collect_exchanges` carries `speaker`; a new optional `speaker` tool arg filters the search to one person BEFORE ranking (case-insensitive; an untagged turn NEVER matches a named filter, so a per-person query returns ONLY that person's turns); results are labelled by name (`[speaker] asked:` vs neutral `You asked` for untagged). Pure helpers `_speaker_matches` / `_who_label`. Tool schema + description teach Claude to scope a NAMED-person question to that speaker.
- `scripts/conversation_recall_test.py` 22→**37** (helpers, filter incl. exclusion + case-insensitivity + unknown-speaker, `record_turn` round-trip, backward-compat); `turn_runner_test` asserts the speaker reaches `record_turn`. Gate 35/35. **LIVE-PENDING:** voice-routing (does Claude pass `speaker="[name]"` on a spoken "what did [name] ask?"). The storage/filter half is fully validated.
- **Lessons:** (1) **a new field on an append-only log should be optional + omitted-when-absent** — backward compatibility for free, no migration. (2) **filter before rank** — a per-person query should only rank/embed that person's turns. (3) **an untagged record must not match a named filter** — else a per-person query leaks everyone's untagged turns.

## The "make Jarvis feel alive" arc — M83/M84/M85/M86 — 2026-06-12
Four features aimed at the film-JARVIS experience (an assistant that synthesizes, watches, reads the room, and drives the workspace) were scoped and built in this order. Each is its own milestone + commit; docs were batched at the end (this entry). All gated green; all LIVE-PENDING at the time of writing (they need real use to validate).

### M83 — Anticipatory intelligence ("Sir, I've taken the liberty…") (`12a5209`)
The SYNTHESIS layer over every single-signal monitor: a background poll fuses the live world-state (calendar + weather + active alerts + reminders + homelab + security) and asks Claude, as an extremely-selective chief-of-staff, whether ONE genuinely useful cross-domain insight is worth surfacing right now. The shape of the value is structural: an early meeting at a remote location, a severe-weather alert for that location, and no UPS on the desktop → *shut the machine down before you leave* — an insight no single monitor could produce, because each of the three signals is individually unremarkable. Most ticks correctly stay silent (PASS) — a chatty anticipation engine is a failed one.
- `src/anticipation.py`: `AnticipationEngine` (poll thread, homelab_monitor shape) + pure tested cores — `_format_world_state` (composes the snapshot, skips empty sections, threads recent insights back so it won't repeat) and `_parse_decision` (PASS/quoted/short-PASS-tail → silence; else the sentence). The LLM call AND the world-state snapshot are INJECTED → fully decoupled + network-free in tests. Dedup via a recent-list + a similarity guard.
- `main.py` composes the snapshot from the briefing gatherers + weather alerts + homelab status + security armed state (each fail-soft); engine registered with self_status; threaded into shutdown.
- Rides `_announce` tagged 🧠, added to `quiet_hours._DEFERRABLE_LABELS` so insights DEFER at night → the morning "while you were away" catch-up (urgent single signals like a storm have their own piercing monitor, so anticipation defers cleanly).
- DEFAULT OFF (recurring LLM spend) — opt in via `JARVIS_ANTICIPATION=1`; cadence `JARVIS_ANTICIPATION_POLL_SECONDS` (default 1800). `scripts/anticipation_test.py` 36/36; gate 36/36.
- **Lessons:** (1) the value is SYNTHESIS — a fused cross-domain insight no single monitor could produce; (2) make selectivity the system prompt's whole job (PASS is the default); (3) inject the snapshot + LLM call so the engine is testable without network.

### M84 — The ambient HUD (`8bb63cf`)
A translucent, click-through, always-on-top corner overlay. The arc-reactor orb (coloured by state, core throbbing with the live TTS amplitude), a live waveform, a clock, a security badge (ARMED/SECURE). It floats over everything and the mouse passes straight through.
- `src/hud.py`: `JarvisHUD` — a borderless `Toplevel` with `-transparentcolor` (a magic key colour → transparent) + `WS_EX_LAYERED|WS_EX_TRANSPARENT` via ctypes (click-through) + `WS_EX_TOOLWINDOW` (off the taskbar/Alt-Tab) + `-topmost`. Reuses the M57 orb animation technique. Pure colour helpers (`_dim`/`_blend`/`_hex<->rgb`) unit-tested; the visual surface is live-validated by eye.
- `src/console.py`: builds the HUD on the shared Tk root (rides this mainloop) when enabled, forwards state/amplitude/armed to it. Fully fail-soft — any setup failure leaves `self._hud` None and the console is untouched. Armed state mirrored to `self._armed_on` so the HUD shows it.
- DEFAULT OFF — opt in via `JARVIS_HUD=1` (restart to show). `scripts/hud_test.py` 18/18; gate 37/37.
- **Built BLIND** (no display available in the dev environment) → the feature was made robust + fail-soft + opt-in, with visual tuning (position/size/colour) expected on the first eyeball. **Lesson:** for a build-blind graphics feature, lean on known-good platform recipes and make every failure path disable cleanly.

### M85 — Tonal awareness ("You sound tired, sir.") (`ebd62be`)
Jarvis reads HOW something was spoken — volume, pace, pauses — from the clip STT already captured (the same `Transcript.audio`), and threads a short DESCRIPTIVE cue into the LLM call so Claude can infer the mood and adapt. The chosen approach: feed Claude the CUES, let it interpret — never hard-label an emotion in code.
- `src/voice_tone.py`: `ToneAnalyzer` holds a rolling loudness baseline so "softer/louder than usual" is judged against the speaker's OWN voice (gain-independent); pure tested cores (`_features`, `_pace`/`_pause`/`_loudness` labels, `_compose`). numpy-only, no model, no network. Reports a cue ONLY when delivery is NOTABLE — neutral delivery → nothing, so most turns carry no note. Never raises.
- `src/llm.py`: `stream_response` gains `vocal_cue`, appended to the M80 per-turn uncached system block (combined, so a turn adds at most one extra block). New "Tone awareness" system-prompt section tells Claude to use it SUBTLY — shape the manner, never robotically announce the mood.
- `main.py`: one `ToneAnalyzer` per session; computes the cue on the voice clip (same one speaker-ID uses) and threads it `process_question → stream_response`. DEFAULT ON (free, local); kill with `JARVIS_TONE=0`. `scripts/voice_tone_test.py` 27/27; gate 38/38.
- **Lessons:** (1) cues, not diagnosis — report the delivery, let Claude read the mood; (2) only surface when NOTABLE, so it's signal not noise; (3) judge loudness RELATIVE to the speaker's own baseline so it's gain-independent.

### M86 — Workshop control ("Jarvis, pull that up") (`62ec23c`)
Speak, and the workspace responds: open a web page, focus a window, clear the screen, hit a media key. **DESIGN PIVOT:** `system_control` was ALREADY the local-action tool (open_app, volume_*, lock, screen_off, kill, SRE verbs). A separate `desktop_control` tool would have DUPLICATED open_app/volume and split routing — so M86 EXTENDS `system_control` with only the genuinely-new verbs:
- `open_url` (build the URL; `cmd /c start "" <url>`, same launch path as open_app, normalized + space-free + separate argv → no injection seam), `focus_window` (ctypes EnumWindows + SetForegroundWindow; best-effort vs Windows foreground-lock), `show_desktop` (Win+D), `media` (transport key: play_pause/next/previous/stop via VK_MEDIA_* keybd_event).
- All four are low-impact + reversible → NO confirmation gate (like open_app); `system_control` stays in `_RESTRICTED_DENY` so a phone/Discord turn can NEVER drive the desktop. This tool still NEVER closes/kills/deletes — those stay behind the confirm + admin gates.
- `scripts/system_control_test.py` 38→**53**: new verbs covered WITHOUT firing real key-presses/launches (faked ctypes + Popen seams; error paths fire nothing). Gate 38/38.
- **Lessons:** (1) before adding a sibling tool, check whether an existing one already owns the surface — extend it, don't duplicate (routing clarity > separation); (2) test side-effecting verbs by faking the thinnest seam (keybd_event/Popen), never by firing them in the gate; (3) "benign + reversible" is the line for no-confirmation, distinct from the destructive verbs' gates.

---

## M87 — Interpreter mode: Jarvis as a live two-way interpreter — 2026-06-15
"Jarvis, be my interpreter" → he stops *answering* and starts *relaying*: he
translates each utterance into the other language of a configured pair and
speaks it in that language's voice, continuously and with no wake word between
turns, until "stop interpreting". Two people who don't share a language talk
THROUGH Jarvis. **Motivating use case:** a non-English-speaking person who does
not share a language with the primary user needs to hold a real conversation
with them.

**Why it stayed small — pure recombination.** The project was architected
multilingual from M2/M3 (Whisper `detected_language` per turn; `VOICE_BY_LANG`
maps en→en-GB-Ryan, es→es-MX-Jorge). Until now that only bought "answer in the
language you spoke." M87 flips the turn pipeline with three new pieces and zero
change to the normal hot path:
- `src/interpreter.py` (NEW, pure/testable): `other_language(spoken, pair)`
  (two-party routing — secondary→primary, else→secondary, so an ambiguous
  Whisper detect biases toward the other party's language), `is_start_intent` /
  `is_stop_intent` (NFKD accent-folded substring matchers — "intérprete" →
  "interprete" — incl. Spanish phrasings), `build_translation_prompt`
  (translate-ONLY system prompt: no persona, no "sir", no commentary, formal
  *usted* for Spanish), and the bilingual confirmation strings. `LANG_PAIR`
  from `JARVIS_INTERPRETER_LANGS` (default `en,es`).
- `src/llm.py::stream_translation` — a MINIMAL streaming call: tiny translate
  system prompt, no tools, no history, no caching. Isolated from the agentic
  `stream_response` machinery so a translation can't trip the tool loop or leak
  the Jarvis persona, and stays low-latency. Raises on transport error; caller
  swallows it.
- `main.py` — `TurnRunner.interpret()` (translate→`speak_streaming` in the
  target voice, holding the SAME `pc_speaking` + `announce_speaking` gates a
  normal turn holds, under `self._lock`; no history/memory/tools/barge-in) +
  `speak_line()` (gated fixed-line speak, for the confirmations). In
  `listen_loop`: a self-contained interpreter branch at the top of the loop
  that captures→translates→speaks→re-listens with NO wake word
  (`max_pre_speech_sec=_INTERPRETER_WINDOW_SEC=30`; silence just re-arms, it's
  the cadence not an exit), and a start-intent check among the other intents
  (after security/enroll/knowledge, before the LLM turn). The branch `continue`s
  before the normal path, so that path is byte-untouched.

**Self-echo is already solved:** `interpret()` sets `pc_speaking`, and the
M68/2026-05-29 omni-mic gate makes the capture abort while the PC speaks — so
Jarvis never transcribes his own translation. Two people take turns naturally
(the mic is deaf while he relays), exactly like working with a human interpreter.

**Activation is voice-only by design** (you're standing with someone); a tray
toggle is a clean follow-on. The start confirmation is spoken in BOTH languages
so the other party hears what's happening.

`scripts/interpreter_test.py` **50/50** (routing, both intent matchers incl.
accent folding + negative cases, the prompt contract, pair parsing); gate
**39/39**. **LIVE-PENDING** a real two-party session. **Known caveats for the
live test:** turn-taking requires both people to pause for the relay (latency =
translate round-trip + TTS); Whisper drives direction, so a mis-detect routes
an ambiguous utterance to the secondary language (it assumes the primary user is
the English-side driver) — worth watching whether the secondary speaker's
Spanish is detected reliably on this mic.
**Follow-ons (not built):** a tray toggle; a per-utterance turn-marker tone;
`JARVIS_INTERPRETER_MODEL` (Haiku for lower latency); auto-language could lean
on M69 speaker-ID instead of Whisper-only detection.

---

## M88 — Conversation mode (full-duplex, Phase 1) — 2026-06-15
"Jarvis, let's talk" → he drops the wake-word-per-turn rhythm and stays in a
continuous, hands-free conversation: you speak, your pause ends the turn (the
existing VAD endpointing), he answers, and he's listening again instantly — no
"Hey Jarvis" between turns — until "that's all" / "exit conversation", or a
silence stretch auto-exits him to standby. **This is M51's follow-up window made
PERSISTENT, applied to normal Q&A** (full tools/persona/memory — unlike M87
interpreter mode, which relays). The walkie-talkie→conversation jump.

**Phase 1 of the full-duplex arc**, sequenced deliberately: build Phase 1 now,
de-risk Phase 2 next. Phase 2 — hands-free TALK-OVER (interrupt him by just
speaking, no wake word) — is the hard part (it needs acoustic echo cancellation
so the omni mic separates the speaker's voice from Jarvis's own playback) and is
gated behind a feasibility spike. Until then the M52 wake-word barge-in still
works inside conversation mode.

**Almost pure reuse of the existing turn path** — the listen loop just keeps
`followup` effectively pinned while the mode is on and re-arms on silence
instead of dropping to the wake word:
- `src/conversation_mode.py` (NEW, pure/testable): `is_start_intent` (EXACT
  match after affix-stripping a leading wake-word + trailing politeness — so
  "let's talk ABOUT X", a real question, does NOT hijack mode entry;
  contrast interpreter.py's substring match, safe there because "be my
  interpreter" never prefixes a question), `is_stop_intent` (substring — the
  exits are specific), NFKD accent-folding ("conversación"→"conversacion"),
  language-keyed confirmations.
- `main.py` listen_loop: a `conversation_mode` Event + `idle_empties` counter;
  surgical edits to the existing follow-up machinery (listen-phase condition,
  pre-speech window `_CONVERSATION_WINDOW_SEC=25`, the empty-transcript branch
  now re-arms in mode and auto-exits after `_CONVERSATION_IDLE_EXITS=3`
  consecutive empties ≈ 75s, the reset block, and the pre-turn dismissal block
  which now also exits the mode with a spoken ack). Start/stop intent checks sit
  among the other intents (after security/enroll/knowledge, before the LLM turn).
  Reused `_is_dismissal` for natural exits. The whole turn pipeline
  (process_question, tools, memory, M52 barge-in) is unchanged.

**Mode interaction:** the M87 interpreter branch sits at the very top of the
loop and `continue`s, so saying "be my interpreter" inside conversation mode
cleanly supersedes (interpreter wins while active; conversation resumes after
"stop interpreting") — nested modes resolve without fighting.

`scripts/conversation_mode_test.py` **41/41** (exact-match activation incl. the
"let's talk about X" negative, substring exits, affix stripping, accent folding,
confirmations); gate **40/40**. **LIVE-VALIDATED 2026-06-15** — a full hands-free
multi-turn exchange (3 turns, zero wake words), the enrolled speaker recognized
every turn, a clean intentional "that is all" sign-off exit, and ZERO
echo-driven phantom turns (ambient_max stayed tiny after each reply — the
pc_speaking gate held in an open-plan room with a loud speaker near the mic).
Dismissal matcher well-calibrated.
Voice-only activation by design; a tray toggle is a clean follow-on.
**Follow-on = Phase 2 (the AEC/echo de-risk spike, next).**

---

## M88 Phase 2 — hands-free talk-over: AEC de-risk (VIABLE) — 2026-06-15
The full-duplex Phase 2 question (interrupt Jarvis by just talking, no wake word)
hinges on cancelling his own playback from the omni mic. De-risked in three
measured steps (the M44 "measure, don't assume" discipline), all via throwaway
probes (`scripts/*_probe.py`, `aec_spike.py` — not gate tests):

1. **Energy probe** (`duplex_echo_probe.py`): silent vs talk-over RMS. Verdict
   NOT SEPARABLE by energy — Jarvis's loud syllables (echo p95 0.12, peaks 0.18)
   are louder at the mic than a typical speaking voice (median 0.089). Needs AEC.
2. **AEC library spike** (`aec_spike.py`, offline synthetic): WebRTC APM sdist
   won't build on Windows/py3.12; **pyaec** (SpeexDSP, prebuilt native DLL)
   does. preprocess=False (the barge config) = **19.4 dB ERLE, corr(out,user)
   0.93** (echo cancelled, near-end voice preserved); preprocess=True kills the
   near-end.
3. **Real-room probe** (`aec_live_probe.py`, `sd.playrec`): first pass read
   0.1 dB — but the 264 ms play→capture delay (system BUFFERING, not acoustics)
   exceeded the filter window, so the echo fell outside the filter's reach. KEY
   LESSON: a real-time AEC must feed the far-end reference DELAY-ALIGNED to the
   mic. Patched to request low-latency playrec (→160 ms) and pre-align by the
   measured delay → **12.1 dB real-room ERLE** (echo RMS 1995 → 494). Post-AEC
   the near-end voice (~0.089) sits ~6× above the residual echo (~0.015) →
   separable.

**VERDICT: VIABLE — build on pyaec.** Honest limits: 12 dB is solid not
luxurious (double-talk degrades it → barge VAD needs a conservative threshold +
live tuning); the ~160 ms delay means barge-in has a beat of latency; the
integration must compensate the play→capture delay AND resample the 24 kHz TTS
far-end to the 16 kHz mic rate. **These alignment constants can only be VERIFIED
live (guessing blind silently cancels nothing — proven by the first probe run),
so the integration is a focused live-tuned build, not a blind one.** Planned
shape: an `AecBargeMonitor` mirroring the M52 barge-in (reuses interrupt_event +
the WASAPI owner-thread cut) — during TTS, feed mic + delay-aligned far-end (a
ring buffer speak_streaming publishes) → pyaec(preprocess=False) → VAD on the
cleaned signal → set interrupt_event on sustained speech. Opt-in flag,
default off. pyaec added to requirements.

**M88 Phase 2 — real-time validation (2026-06-15):** The barge monitor's first
cut aligned the far-end to the mic by WALL-CLOCK timestamps across two
independent streams (sd.play + a separate InputStream) — it FAILED live (45
false fires in the silent run; the AEC cancelled nothing because two streams'
buffer latencies are large + jittery, smearing pyaec's reference so the adaptive
filter never converged; the monitor just detected raw echo, going quiet only in
the inter-sentence gaps — a textbook signature). **The standalone validation
caught this BEFORE any wiring into Jarvis — the whole point of validating
standalone.** Fix: a SINGLE DUPLEX `sd.Stream` (play+capture in one callback,
one clock) → far and mic sample-locked with a fixed jitter-free offset (the
stream's in+out latency), which is what an adaptive AEC needs. AEC+detect run
inline in the callback. **Result (real MC1000 mic + a loud speaker close to the
mic — the hard room): SILENT 0 detections (clean), TALK 30 (fires reliably).**
Stream latency (0.016, 0.096); delay 160 ms = 10 frames covered it. **Hands-free
talk-over is VALIDATED on this hardware via the duplex architecture.**
REMAINING = the integration: route Jarvis's TTS playback (currently per-sentence
`sd.play` @24k in `speak_streaming`) through a 16 kHz duplex stream whose
callback runs the AEC+detect and sets `interrupt_event` (reusing `BargeDetector`
+ the M52 cut on the owning thread). Tradeoff to weigh: the barge path plays TTS
at 16 kHz (resampled from 24 kHz) to reuse the validated 16 kHz AEC tuning —
slight quality dip, barge-path only. The wall-clock `FarEndBuffer` in
`src/aec_barge.py` is superseded by the in-callback frame-history (the validated
design). Integration is its own focused build + live verify (streaming TTS
through a duplex ring buffer is a NEW mechanism the AEC validation didn't
exercise).

**M88 Phase 2 — INTEGRATED then SHELVED (2026-06-16).** Built the real
integration: `src/aec_barge.py::DuplexBargePlayer` (replaces the superseded
wall-clock `FarEndBuffer`/`AecBargeMonitor`) plays Jarvis's reply through ONE
duplex `sd.Stream` whose callback cancels his echo (pyaec, far-end delayed by the
stream's fixed offset) and runs the `BargeDetector` on the cleaned signal → sets
`interrupt_event`. Wired into `speak_streaming` via `_play_via_duplex_aec`
(opt-in `JARVIS_HANDS_FREE_BARGE`, normal `sd.play` path byte-untouched,
fail-soft to plain playback). `main.py` threads `mic_device` into `TurnRunner`
and picks AEC barge over the M52 wake-word monitor when enabled. An `on_barge`
callback flips the orb to LISTENING the instant a cut is detected (fixed the
SPEAKING-lingers-through-teardown UX). `scripts/aec_barge_test.py` 21/21
(`DuplexBargePlayer._next_out` buffer/end logic); gate 41/41.
**Live: it WORKS mechanically** — cut Jarvis off 4/4, clean 16k audio, snappy
orb. **But real use surfaced the fatal limit:** once Jarvis's echo is cancelled,
the energy gate fires on ANY residual sound — another person moving about the
space, ambient activity elsewhere in the room — because it detects *presence of
sound*, not *intent to interrupt*. The quiet-room de-risk (silent vs.
you-talking, one occupant) structurally could not surface this: there was no
third source of sound in it.
**VERDICT: fully-hands-free talk-over is environmentally unsuited to a busy,
open-mic room.** The robust UX that ships is conversation mode (Phase 1) +
"Hey Jarvis" to interrupt (M52) — the wake word is immune to ambient noise
because only those two words trigger. Flag left OFF (commented in `.env`); all
the code kept (opt-in, fail-soft, gated) for a headset / quiet-room future, or a
future speaker-gated rebuild (only an enrolled voice triggers — the "right" fix
for a shared space, but hard real-time on ~0.5 s echo-residual frames).
**LESSON: a barge detector must gate on INTENT (a wake word, or speaker identity),
not just sound presence, in any shared acoustic space — energy-on-cleaned-signal
only works in a single-occupant quiet room. Validate the hardest feature in the
REAL environment (with other people and background activity present), not a
controlled quiet test.**

---

**2026-06-21 — per-line log timestamps + the THIRD QoL consolidation pass
(18 fixes, gate 42→44, LIVE-VALIDATED).** Began as a debrief of a real ~17 h
continuous armed soak (the machine stayed up for the whole window — no crash, no
shutdown — including a successful mid-window Discord webcam check-in).

- **The soak validated the security arc in production:** M70 auto-arm AND
  auto-disarm both fired correctly (the transition worked in both directions),
  M81 `voice_while_armed` logged ZERO false fires across the whole window (the
  M87 RMS-floor fix held), and the M71 Discord webcam served a clean
  armed-watcher frame (`using armed-watcher frame` → 1920×1080). The M44 memory
  arc held (no watchdog trip, ~17 h armed).
- **The load-bearing post-mortem:** an initial reading of the log concluded that
  the auto-disarm had never fired and that the system sat armed all night. That
  conclusion was WRONG and was corrected on review. The log only timestamped
  session markers, so FILE-adjacency (the `welcome home` line sitting right next
  to the next morning's boot marker) was read as TIME-adjacency; the two lines
  were ~12 h apart, separated by an overnight machine shutdown. Ground truth —
  the disarm had in fact fired, and the machine was powered off afterwards —
  contradicted the reading. **Lesson: an un-timestamped log invites the adjacency
  trap — file-order ≠ wall-clock order across a process-death gap. Reconcile any
  log-only inference against independent ground truth before acting on it.**
- **(Feature) Per-line log timestamps** — the direct fix. `src/logfile.py` gained
  a pure `stamp_chunk(data, at_line_start, stamp)` core + a `TimestampStream`
  wrapper prefixing every emitted line `[YYYY-MM-DD HH:MM:SS]`; blank separators
  stay clean, print()'s two-call pattern and multi-line tracebacks each stamp
  correctly. Wired at ALL THREE stdout/stderr redirect sites (`jarvis.pyw`,
  `jarvis_watchdog.pyw`, `main.py::setup_logging` — production inherits the
  wrapper from jarvis.pyw via the early-return). **Load-bearing:** `fileno()`
  DELEGATES to the underlying file, so a subprocess that inherits stderr by fd
  (the Plex MCP child, which writes its own `[MM/DD/YY ...]` lines straight to
  the log in production) keeps working — Python `print()`s get stamped,
  subprocess output passes through raw. `scripts/logfile_timestamp_test.py` (21).
  Live-validated in both pythonw (stamped marker) and console (`python main.py`,
  unstamped marker via the direct `log_file.write`, stamped body) modes.
- **(Pass) QoL consolidation #3** — regression-net-first (gate green) → **7
  parallel read-only auditor agents** over the full `src/` tree + `main.py`
  (~28.6k LOC, 70 files), rubric tuned to this project's DELIBERATE contracts so
  intentional design (fail-open speaker gate, no-gate `run_code`, the
  `announce_speaking`/`pc_speaking` cooperative gates, WASAPI thread-affinity,
  the M44.3 persistent capture, the restricted-origin tool boundary) wasn't
  flagged. The auditors spent most effort DISPROVING suspicions (no command
  injection, no gate bypass, no XSS, no live leak, no thread-affinity violation).
  **18 fixes in risk order, each individually gated.** See
  [docs/CODE_AUDIT.md](CODE_AUDIT.md) for the full finding table; the headline
  Tier-1 items:
  - **Conversation-mode STT stuck-loop** (`main.py`): a PERSISTENT STT failure
    re-armed immediately with no wake-word fallback — a hot loop. Now an STT
    exception counts toward the M88 idle auto-exit, so repeated failures drop to
    standby. The one true fail-soft violation found.
  - **fsync on atomic writes** — new `src/atomic_io.py::atomic_write_text`:
    fsync-before-`os.replace` (durability the temp+replace dance lacked — a hard
    power loss on the no-UPS machine could commit a torn/zero-length store) +
    UNIQUE temp names + a Windows `os.replace` `PermissionError` retry the
    concurrency test surfaced (concurrent renames to one target race on Windows;
    POSIX doesn't). Wired into reminders/predictions/quiet_hours/ui_state; also
    de-dups four near-identical `_save`s. `scripts/atomic_io_test.py` (11).
  - **Briefing prediction cycle → background** (`predictions.py`): a voice "good
    morning" ran mine+resolve INLINE in the agentic loop — up to ~2min on Haiku +
    3×40s web-search. Now `briefing_followups` surfaces already-resolved
    predictions instantly and kicks mine+resolve onto a guarded daemon thread
    (`run_prediction_cycle_async`, a non-blocking lock so only one runs); fresh
    results surface NEXT briefing. **Live-proven:** "good morning" → `[predictions]
    mined 1` fired in the background while the reply composed normally (~10s), no
    stall.
  - **PWA token-wipe-on-timeout** (`remote_pwa.py`): the client wiped the saved
    token on ANY `auth_fail`, including the server's handshake-`timeout` variant
    (slow cellular / backgrounded tab) — forcing needless re-entry. Now only a
    real rejection (no `reason`) wipes; a timeout keeps the token and lets the
    backoff reconnect.
  - **Crash-proof config** (`config.py`): bare `int()`/`float()` on ~8 env
    tunables crashed startup on a typo'd `.env`. `_int_env`/`_float_env` fall
    back with a logged warning. `scripts/config_env_test.py` (16).
  - **Tier 2 (latent, 7):** `verify_person` gc cycle-break (M44 path off the loop
    collect); sound_detector audio-callback `put_nowait` guard (a `queue.Full`
    escaping a sounddevice callback aborts the stream); anticipation LLM client
    `timeout=30/max_retries=1`; **interpreter/conversation mode mutual-exclusion**
    (entering one clears the other — fixes "stop interpreting" silently dropping
    back into conversation mode); conversation_recall `days_back` made
    calendar-day-consistent with the file filter ("yesterday" stopped
    under-returning); **all-day calendar TZ shift** (naive midnight was stamped
    UTC then `.astimezone()`'d → day backward in behind-UTC zones; now kept
    naive-local — masked because every consumer special-cases all-day); predictions
    `made_at` fallback uses the EARLIEST batch exchange (latest made a backfilled
    old call look fresh, delaying its first resolve check by 7 days).
  - **Tier 3 (cosmetic, 6 + 1 deferred):** telemetry no longer labels a *denied*
    tool "fired"; websockets handshake-traceback log noise silenced
    (`websockets.server` logger → CRITICAL); `aec_barge` param-shadow renamed
    `aec_barge_on`; stale comments fixed (news "sports" is aliased now;
    system_control admin-invariant; llm image-hook surfaces only the first
    image); `_is_dismissal` computed once per turn. **DEFERRED:** the text/voice
    intent-dispatch rule-of-three extraction — a real hot-path refactor with no
    ladder-level test; deserves its own focused pass, not a tail-of-batch rush.
- **LIVE-VALIDATED 2026-06-21** across two console sessions: clean startup (whole
  batch loads), a normal turn, full conversation mode (hands-free multi-turn +
  web_search + barge-in), the sign-off exit, the briefing background cycle, and
  the interpreter mode-stack fix (conversation → interpreter → "stop interpreting"
  → STANDBY/wake-word, not a silent conversation-mode fallback — the exact failure
  signature did NOT occur). Clean shutdowns, sessions sealed, zero tracebacks.
- **Minor, out-of-scope (noted):** "what time is it" answered "12:00 AM" at
  12:30pm — the temporal-grounding prompt injects the DATE but not `%H:%M`. A tiny
  future fix; date grounding itself is fine (the briefing said "Sunday morning").

## QoL consolidation pass #4 — the full-QA sweep (2026-07-02)

**Trigger:** an explicit request for an end-to-end QA of the whole codebase, 11
days after pass #3 — not one of the usual cadence triggers. **Method evolution:
LOG REVIEW FIRST.** Reading jarvis.log before any code surfaced a LIVE INCIDENT
no test or auditor could have: **the listening loop was dead in production that
very morning** — a `PortAudioError -9985` on the mic open at 07:32 killed the
`listen_loop` thread unhandled (its `try` had only a `finally`), and the
assistant ran DEAF all day while every other subsystem (Discord, remote,
monitors, presence auto-arm→disarm) hummed along. The M65 watchdog never fired —
it watches process exit, not thread death. The acoustic stream opened the SAME
device 35 minutes later, so a retry recovers. This violated the project's oldest
rule ("never crash the listening loop") and had sat latent since day one.

Then the pass-#3 shape: baseline gate green (44/44) → **eight parallel
read-only auditors** over the whole tree (rubric tuned to the deliberate
contracts) → every finding re-verified at the source → **~37 fixes across 31
files in risk order**, each gated. Highlights (full detail in
[docs/CODE_AUDIT.md](CODE_AUDIT.md)):

- **The mic-session supervisor** (the incident fix): `listen_loop`'s session
  body extracted verbatim into `_voice_session_loop()` + a retry supervisor —
  any escape logs once per outage, alerts in-console AND aloud, re-opens on a
  capped backoff, announces recovery. `AudioSession.read()` gained a 10 s
  stall timeout (a silently-dead stream now recovers instead of wedging quit).
- **Chronic armed-mode acoustic overflow** (log finding #2): the acoustic
  stream overflowed every ~3-4 s through EVERY armed window (~1-2k lines/day,
  each overflow a gap-corrupted detection window). `latency="high"` on the
  InputStream + rate-limited status logging out of the callback hot path.
- **Interpreter mode had NO failure escape** (the pass-#3 conversation-mode
  fix was never mirrored): persistent STT failure = an unrecoverable
  wake-word-less hot loop; tray Reset was also inert while interpreting. Both
  fixed (failure counter → auto-exit; reset honoured).
- **`_auto_disarm` stuck state**: a memory-watchdog trip mid-CHALLENGE left
  `_challenge_active`/`_locked` set forever — every utterance diverted to the
  passphrase comparator, 🔒 pinned. Now clears like `deactivate()`.
- **The speech gates didn't nest** — new `src/gates.py::CountedEvent`: an
  announce overlapping a turn reply used to drop `pc_speaking`/
  `announce_speaking` when the FIRST speaker finished (re-opening the omni-mic
  echo + M68 stutter for the rest of the reply). set/clear now count.
- **A long voice note from a remote client killed the WS** (websockets 1 MiB
  default frame cap vs the 60 s mic cap) → `max_size=8 MiB`.
- **Durability/store batch**: `knowledge.reindex` made genuinely atomic (DDL
  autocommitted before — a crash left a committed EMPTY index);
  predictions watermark no longer advances on a failed scan/mine (a window
  could be permanently skipped) + a store lock closes the surfaced-flag lost
  update; `DedupeStore` migrated to atomic_io (backs the 168 h weather dedupe
  — the no-UPS re-announce class); quiet_hours store locked.
- **The clock fix** (pass #3's noted follow-up): current TIME rides the
  per-turn UNcached system block; the cached prefix keeps date-only precision.
- **Plus ~20 smaller verified fixes**: sound_detector lifecycle lock/leaks,
  camera-capture disarm-race leak, conversation-mode idle-counter defeats
  (media + announce), LOCKED-entry auth race, vision timeout, notifications/
  discord exception breadth, Whisper/embedder double-load locks,
  TimestampStream write lock, `_float_env` NaN rejection, shared Anthropic
  client (a TLS handshake per turn, gone), non-ASCII token guard, pc_shell
  leading-hyphen argv injection, watchdog log-rotation defeat + venv fallback,
  "good night" un-shadowed from the M63 wrap, PWA DOM cap, autostart quote
  escaping, tray-failure guard, monitor toggle races (×4).

**Gate 44 → 45 suites, all green** (new `gates_test` 13; predictions 42→47;
dismissal 27→28). **LIVE-PENDING:** normal turn + mic-unplug recovery + an
armed window (overflow ~1/min max) + "good night" routing + a long voice note
from a remote client + "what time is it". Deferred, on the record:
OneDrive-redirected Desktop for shortcuts, audible announce-over-reply
serialization, the intent-dispatch extraction (carried).

**Lesson (method):** the log beat the tests — a production log review belongs
at the START of every QA pass, not as an afterthought; and "a thread that must
never die" needs a supervisor, not just careful code (the same reasoning that
built the M65 process watchdog, one level down).

## M89 — effort tuning: the voice path stops over-thinking "what's the weather" — 2026-07-28

**What shipped.** `output_config.effort` is now set explicitly per path:
`medium` for a spoken turn, `high` for engineer mode, overridable via
`JARVIS_VOICE_EFFORT` / `JARVIS_ENGINEER_EFFORT`. Median turn latency drops
**6.7s → 4.6s** and time-to-first-token **6.0s → 3.8s**, with no measurable
cost to tool routing.

**Why it mattered.** Sonnet 5 defaults to `effort: high` when `output_config`
is unset. Jarvis never set it, so every "what's the weather" was answered at
the same reasoning depth as a multi-step diagnostic — on an interface whose
stated design constraint is *first audible word within one second*.

**Why it was measured rather than just set.** Lowering effort is not obviously
safe here. The voice path runs `thinking: {"type": "disabled"}`, and Sonnet 5
with thinking off is already less inclined to reach for tools; `effort` pushes
the same direction. Jarvis's usefulness *is* its routing across ~40 tools, so a
latency win that quietly costs tool recall is not a win. `scripts/effort_probe.py`
drives the real `stream_response` — real system prompt, real tool schemas, real
agentic loop — over 10 representative questions with a known expected tool.

**The measurement was wrong twice before it was right.** Both worth recording:

1. *A stale expected tool name.* The probe graded `get_pc_diagnostics`; the
   schema name is `pc_diagnostics`. That scored a MISS against a tool that had
   fired correctly. A grader bug looks exactly like a regression.
2. *Level order was confounded with wall-clock time.* The first two runs swept
   all `low`, then all `medium`, then all `high`. Drift in API latency during
   the run therefore lands on whichever level owned the calm stretch — and
   because the order was identical, **re-running reproduced the artefact
   exactly**. Replication is not control. The probe now interleaves levels
   per query and rotates their order.

The `medium` advantage survived the interleaved re-run, so it is real.

| effort | median latency | median TTFT | output tokens | reply words | routing |
|---|---|---|---|---|---|
| low | 7.1s | 6.1s | 116 | 26 | 14/14 |
| **medium** | **4.6s** | **3.8s** | 130 | 30 | 14/14 |
| high | 6.7s | 6.0s | 135 | 33 | 14/14 |

**What the data killed.** Two findings from the uncontrolled runs did *not*
survive: `high` appearing to lose 2 routing cases, and `low` appearing to skip
`wolfram_query` on a borderline arithmetic question. In the controlled run all
three levels routed 14/14 and all three answered the arithmetic directly.
Routing here is probabilistic; run-to-run variance is large enough to invent an
effect that is not there. Both were nearly written up as effort effects.

**Honest unknown.** The latency curve is **not monotonic** — `medium` is faster
than both `low` and `high`. With thinking disabled there are no thinking tokens
for effort to trim, and iteration counts are flat (1.8 across all levels), so
the usual explanation does not apply. The effect reproduced three times
including under interleaving, so it is taken; the mechanism is not understood
and is recorded as unknown rather than guessed. Worth re-measuring on any model
change.

**Guards.** Haiku 4.5 **rejects** `output_config.effort` with a 400, and the
background jobs (session summariser, prediction miner) run on Haiku — so effort
is gated by model, and `scripts/effort_config_test.py` asserts a Haiku turn
sends no `output_config` at all. An invalid value in `.env` would otherwise be
a 400 on *every* turn, taking Jarvis off the air, so it logs and falls back to
the default instead.

**Not changed.** The background Sonnet callers (anticipation, prediction
resolution, vision description) still use `messages.create` directly and keep
the API default. They were not measured, and tuning an unmeasured path on a
hunch is the thing this project keeps learning not to do.

## M90 — web_search dynamic filtering re-tested: blocker gone, pin stays — 2026-07-28

**Outcome: no code change to the tool version.** This entry exists because the
*reason* for the pin was wrong, and a wrong reason in a comment is a trap for
whoever reads it next.

**Background.** On 2026-06-02 (`02e5b91`) `WEB_SEARCH_TOOL` was rolled back from
`web_search_20260209` to the GA `web_search_20250305` after a live 400:

> `container_id is required when there are pending tool uses generated by code
> execution with tools.`

`_20260209`'s dynamic filtering runs an Anthropic-side code-execution container,
whose id must then be echoed on every subsequent request in the same exchange.
Interleaving a server-side search with a *client-side* tool call broke. The
conclusion recorded at the time was that the agentic loop "can't reliably
thread" that id, so the version was downgraded to remove the container — and
the whole error class — outright.

**Why re-test.** That conclusion was the only thing holding the pin, and it was
never re-checked against the container-capture code the loop actually has
(`stream_kwargs["container"]`, kept at the time as defense-in-depth). Dynamic
filtering trims search results *before* they enter the context window, which on
a tool-heavy assistant is a plausible token win. The 2026-06-02 note called that
win "marginal" — but that was a judgement formed while the feature was broken,
not a measurement of it working.

**Result — the blocker does not reproduce.** `scripts/web_search_version_probe.py`
drives the real agentic loop with queries deliberately shaped to interleave a
server-side search with client-side tools, versions interleaved so API drift
hits both equally (the ordering lesson from M89):

| version | ok | container 400s | median latency | input tok | output tok | cached prefix / req |
|---|---|---|---|---|---|---|
| `_20250305` (pinned) | 8/8 | **0** | 10.7s | 128 | 452 | 28,772 |
| `_20260209` (dynamic) | 8/8 | **0** | 11.8s | 128 | 453 | 31,685 |

Zero container errors on eight `_20260209` runs, including a four-tool turn
(`web_search` + `get_weather` + `get_movie_tv_info` + `pc_diagnostics`) — the
exact shape that failed. The loop threads the id correctly.

**And yet the pin stays.** The upgrade buys nothing measurable: identical input
and output tokens, ~1.1s *worse* median latency, and a tool schema that costs
~2.9k MORE cached-prefix tokens per request. Dynamic filtering pays off when
there is enough result volume to be worth filtering; Jarvis's voice queries
don't generate it. The original "marginal for voice queries" judgement was
right on the merits, even though the reason attached to it was not.

**Honest limit on that second finding.** Telemetry accumulates input/output/cache
tokens only, so server-tool result ingestion is not isolated. "No measurable
benefit" is a weaker claim than "no 400", which is unambiguous. Recorded as such
rather than rounded up.

**What actually changed.** The comment above `WEB_SEARCH_TOOL`. It now says the
blocker is gone and the pin rests on a *measurement*, so switching is a low-risk
change whenever `_20250305` retires or a search-heavy workload appears — with a
probe to re-decide on. Previously it said the newer version was broken, which
would have deterred anyone from ever revisiting it.

**Lesson.** A workaround outlives the bug it was written for. When the fix and
the workaround coexist, the workaround's justification needs a re-test date, not
just a rationale — otherwise the codebase carries a permanent detour around a
pothole that was filled months ago.

## M91 — long-horizon background agents: "research it overnight, brief me at breakfast" — 2026-07-28

**What shipped.** `start_background_task` hands an open-ended job to an
Anthropic Managed Agents session that works for minutes to an hour and reports
back on its own. Results are spoken when they land, or — if they land at 03:00
— held and folded into the morning briefing. Plus `list_background_tasks` and
`cancel_background_task`. Off by default behind `JARVIS_BACKGROUND_AGENTS=1`.

Live round trip verified end to end: dispatch → 17s → a cited, speakable
research answer.

**Why this is an architecture change, not a tool.** Every previous capability
fits the turn: wake → answer → speak, capped at 8 tool iterations, first word
inside a second. This is the first thing Jarvis does that *outlives the
conversation that started it*. Film-J.A.R.V.I.S. is not a fast question
answerer; he is a process that works while Tony sleeps and reports back. That
required a second execution mode, not a 41st tool.

**Why Managed Agents rather than a local thread — restart survival.** Jarvis
runs under `jarvis_watchdog.pyw` (respawn on crash) and `update_jarvis` restarts
the process deliberately. A thread holding an overnight task dies as a matter of
*routine*, not misfortune. A managed session lives server-side; `task_store.py`
holds its id and the manager re-attaches on the next start. A local loop would
also have parked an hour of CPU next to the real-time audio threads — exactly
the contention the M67/M68 cooperative speech gate exists to prevent.

**Least privilege, inverted from the obvious instinct.** A background agent runs
*unobserved*, for a long time, with nobody able to interrupt it. The pull is to
give it more power because it is doing real work; it gets strictly **less** than
a voice turn. Its whole surface is Anthropic's hosted toolset inside a
per-session container — no Jarvis tools at all, no host filesystem, no shell, no
camera, no self-update, no route back to this machine. Only the text of its
report crosses back. That is a stronger boundary than the per-origin deny list
used for phone and Discord, which filters a *shared* tool set: here the surface
is separate by construction, so there is nothing to leak through. All three
tools are also in `_RESTRICTED_DENY` — dispatching is unbounded API spend, and a
remote origin must not be able to start it.

**Delivery: two state flags, not one.** `status` is what the agent is doing;
`delivered` is whether the user has been told. Separating them is what makes the
overnight case work. A task finishing at 03:00 is `done` immediately but is
deliberately not announced — `deliver_ready()` checks `quiet_hours.is_quiet()`
and simply leaves it, and `get_briefing` drains it in the morning, marking each
delivered as it hands it over. Collapsing the two flags would mean a restart
between "finished" and "spoken" silently loses the report, which is the one
outcome an overnight task cannot have.

Deciding this in the manager rather than adding a label to
`_DEFERRABLE_LABELS` also avoided a double-delivery: the quiet-hours catch-up
store *and* the new briefing section would both have replayed it.

**THE LIVE PROBE CAUGHT TWO BUGS THAT 41 GREEN TESTS DID NOT.**

The hermetic suite passed completely — including a fake transport that happily
accepted whatever it was handed. The real API did not:

1. `Sessions.create() got an unexpected keyword argument 'initial_events'` —
   that field postdates the installed SDK (0.97.0). Kickoff had to become a
   separate `events.send` call.
2. Worse, and only visible *because* of fixing (1): without `initial_events` a
   session is created **idle** and only starts when the first message lands. The
   poll loop read `idle`, found no messages yet, and would have marked the task
   **done with an empty report** — a silent success that delivers nothing.
   `poll()` now treats "idle with no result yet" as still-running.

That second failure is the interesting one: it did not exist until the fix for
the first, and no fake would ever have exhibited it. A mock proves the code
agrees with your idea of the API; only the API proves the code agrees with the
API.

Treating "idle, nothing yet" as running then created a third hole — a genuinely
wedged session would poll forever and hold a concurrency slot permanently — so
the manager gained an 8-hour age cap. Age is the only signal that distinguishes
"not started yet" from "never will".

**Cost discipline.** Concurrency capped at 2; each task is real, open-ended
spend. `shutdown()` stops polling but deliberately leaves sessions running —
they live server-side and are re-attached — because cancelling on shutdown would
make `update_jarvis` silently destroy an overnight task.

**Privacy — the deliberate exception.** Jarvis's standing promise is that only
transcribed text leaves the machine. This breaks it: the agent fetches pages and
writes working files in an Anthropic-hosted sandbox. Rather than quietly weaken
the claim, the principle in CLAUDE.md now names the exception, says it is
off unless explicitly enabled, and instructs that any future off-box feature
gets its own flag and its own line. A promise is only worth making if it stays
literally true for anyone who has not opted in.

**Files:** `src/task_store.py` (NEW), `src/background_agent.py` (NEW),
`src/background_tasks.py` (NEW), `src/llm.py` (3 tools registered + denied +
system-prompt entries 27-29), `src/briefing.py` (`_background_tasks_section`),
`main.py` (construct, start, status registration, shutdown),
`tests/background_tasks_test.py` (NEW, 41 assertions),
`scripts/background_agent_probe.py` (NEW live instrument), docs.

**Gate:** 50/50 green.

## M92 — scheduled research: "every Monday, look into X" — 2026-07-28

**What shipped.** `set_reminder(action="background_task", repeat=…)` turns a
recurring reminder into a standing research job. The reminder's `message` IS
the research brief, handed verbatim to the M91 background agent at fire time.
Findings arrive on their own — spoken when they land, or folded into the next
morning briefing if they land overnight.

**This was planned as Managed Agents scheduled deployments. It isn't, and the
reason is the interesting part.**

The Phase 2 design called for CMA `deployments` — Anthropic firing the session
on a cron. Two things came out of actually looking:

1. `client.beta.deployments` does not exist in the installed SDK (0.97.0), so
   it needed either an SDK bump on a production always-on process or hand-rolled
   raw HTTP.
2. More decisively — **the project already had a scheduler.** M54 gave reminders
   interval/weekly/monthly recurrence; M59/M63 gave them an `action` that fires
   a composition instead of speaking a message. Both are persistent, both are
   voice-manageable, and `run_scheduler` already *"polls immediately on start,
   so reminders that came due while Jarvis was off fire right after launch"*.

That last detail removed the one real advantage a hosted cron had. Building
deployments would have delivered a **second scheduling system** with its own
list, its own cancel, and its own mental model — two places to answer "what is
scheduled?" for a user whose whole interface is one voice. The cheaper and more
coherent answer was one new action on the scheduler that already exists.

**One design decision worth recording.** The composition table mapped
`action → (composer(), …)` with a **zero-argument** composer, because a briefing
needs no input. A research task needs its prompt. Rather than special-case it,
the signature was generalised to `composer(rec)` and the two existing composers
now accept-and-ignore the record. The table stays one shape; `background_task`
reads `rec["message"]`, so "every Monday, research the NAS market" needs no new
field — recurrence, persistence, downtime catch-up and cancellation all come
free from the reminder record.

Unlike its siblings this composer **composes nothing**: it dispatches and
returns an acknowledgement. The findings are not available at fire time and it
must not pretend otherwise — asserted (`dispatch acknowledges rather than
inventing findings`).

**Three suites caught the signature change, which is the system working.**
`reminders_test` patched an entry with a zero-arg lambda; `good_night_test`
asserted the action set was exactly two; `scheduled_briefing_test` asserted the
schema enum equalled exactly `["briefing", "good_night"]`. The first two were
found by grepping for `_COMPOSITION_ACTIONS`; the third was not, because it
asserts on the *schema* instead — it surfaced only when the full gate ran.

The third was fixed by **loosening the assertion to a superset**, not by adding
the new value: that suite is about the M59 briefing, and pinning an exact enum
made it fail every time an unrelated action was added. `good_night_test` owns
the full-set assertion; a test should assert what its own subject is
responsible for.

**Files:** `src/reminders.py` (generalised composer signature,
`_compose_background_task`, table entry, enum + validation), `src/llm.py`
(system-prompt guidance for scheduled research),
`tests/background_tasks_test.py` (+9 assertions, 50 total),
`tests/reminders_test.py`, `tests/good_night_test.py`,
`tests/scheduled_briefing_test.py`.

**Gate:** 50/50 green.

## M93 — self-review: "how have you been, Jarvis?" — 2026-07-28

**What shipped.** `self_review` reads Jarvis's own logs **across days and
restarts**, groups failures by what actually went wrong, and reports the ones
that recur. Read-only by design.

Against the real log on the day it was written:

> In the last 30 days I've run 57 times and logged 80 concerning lines across
> 14 distinct issues. 1 of those runs ended in an unhandled exception.
> 47 times in 16 sessions: `[outlook] ical fetch failed: ConnectTimeout …`

That top line is a genuine standing defect — the calendar feed has been timing
out for weeks — and nothing in the system could see it before.

**The gap M60 left.** `status_report` counts concerning lines *since the current
session marker*. That is the right window for "are you healthy right now" and
the wrong one for "is something quietly wrong with you". A fault that appears
once per run is invisible per-run and obvious across fifty.

**Signatures, not lines.** Every line carries a timestamp, so raw counting
reports each as unique and the recurring fault drowns. Measured before writing
anything: **31,427 lines → 73 concerning → 19 distinct signatures**, one
accounting for 47 of them. Normalising away timestamps, durations, ids, paths
and retry suffixes is the whole feature.

**Recurrence is counted in SESSIONS, not occurrences.** "47 times" could be one
bad afternoon; "in 16 of 57 runs" is a standing defect. Both are reported and
the ranking sorts by sessions first, because that is the number that decides
whether to care.

**Two bugs the live data caught that a synthetic fixture never would have.**

1. *Traceback scaffolding outranked the errors.* The first run ranked
   `Traceback (most recent call last):` and `The above exception was the direct
   cause of…` as the top two "issues". Those are the frame around a failure,
   never the failure. They are now skipped and the traceback is attributed to
   its terminating exception line instead — so the report says
   `PortAudioError: Device unavailable`, which is actionable, rather than
   `Traceback`, which is not.
2. *Session counting was nonsense.* `--- Jarvis started` is written directly by
   `setup_logging`, so unlike every other line it carries **no `[timestamp]`
   prefix**. The day-window filter therefore inherited whatever state the
   previous line left, and a rotated log's months-old sessions counted as
   recent: **441 sessions in 30 days** against a true 57. The banner's own ISO
   stamp is now parsed.

Both were only visible because the module was pointed at 31,000 lines of real
history before it was pointed at a test.

**Read-only, deliberately.** The obvious next step — have it open a pull request
against its own repo — is exactly what this project keeps declining to build.
Diagnosing yourself is useful; editing yourself unattended is not the same
thing, and the fix stays a human decision. It is also therefore safe from a
remote origin, so unlike the M91 task tools it is **not** in `_RESTRICTED_DENY`.

**Test-fixture lesson.** The suite failed on its first run with two errors that
looked like scanner bugs and were stale fixtures: the rotated-log case wrote
`jarvis.log.1`, and the write helper only overwrote `jarvis.log`, so the
rotation leaked into every later case. A fixture helper that cleans *some* of
the state is worse than one that cleans none, because the leak is invisible.

**Files:** `src/self_review.py` (NEW), `src/llm.py` (tool registered +
system-prompt entry 24a drawing the line against `status_report`),
`tests/self_review_test.py` (NEW, 23 assertions).

**Gate:** 51/51 green.
## M94 — the fix M93 found: transient TLS retries that actually ride out the blip — 2026-07-28

**The loop closed.** M93 shipped a diagnostic; the first thing it said was that
the calendar feed had been failing for weeks. This is that fix — and the
investigation is more of the story than the diff.

**What the diagnostic reported.**

> 47 times in 16 sessions: `[outlook] ical fetch failed: ConnectTimeout: _ssl.c:993`

**First correction: it is not a calendar bug.** The same error class appeared in
`[weather]` (21), `[stt]` (9) and `[news]` (1). Point-fixing outlook would have
treated a symptom — the exact mistake the M67/M68 stutter-gate post-mortem
records ("stop point-fixing").

**Hypotheses tested and discarded, in order.**

1. *Network not ready at boot.* Falsified: **zero** failures in the first 60s
   of a session; 87% land more than an hour in.
2. *That 87% proves anything.* Also discarded — it is confounded. If sessions
   run for hours, most polls happen >1h in regardless, so the distribution
   reflects exposure, not cause.
3. *Silent degradation of the briefing.* Wrong, and worth recording as a
   correction: `_calendar_section` **already** surfaces the error deliberately
   ("silent failure here would mean a missed meeting"). That was designed
   right; no bug to fix there.

**The question that was not confounded** — burst or uniform? Measured: **44
distinct outage bursts over 35 days**, 22 of them a single event, the largest 10
events across 1.6 minutes. Transient network flakiness, not a config error.

**The actual defect.** 52 outlook failures, but only 18 announced a retry —
meaning 34 exhausted it. A retry had already been added for this on 2026-07-02
and *was not enough*, because it was calibrated for the wrong failure shape:

> ```
> 10:26:45  ical fetch failed ... — retrying in 0.5s
> 10:27:01  ical fetch failed ...          <- gave up
> ```

Sixteen seconds apart. The handshake burns the **full 15s timeout** before the
backoff even begins, so a flat 0.5s puts attempt 2 about 15.5s in — still inside
the same blip. The retry never had a chance.

**The fix is strictly better on both axes, not merely more patient.** A
handshake that has not completed in 8s is not going to (a healthy one is well
under a second), so the per-attempt timeout drops and buys a third attempt with
an exponential gap:

| | attempts | worst case |
|---|---|---|
| before | 2 | 15 + 0.5 + 15 = **30.5s** |
| after | 3 | 8 + 1 + 8 + 2 + 8 = **27.0s** |

More chances to catch a good moment, in less time spent failing. That matters
because this path also serves the on-demand "what's on my calendar?" turn, where
a user is waiting in silence.

**What was deliberately NOT changed.** `http_util` gained an `attempts`
parameter — defaulting to **2, exactly today's behaviour**. Every caller of that
helper (weather, news, sports, tmdb, games) is an *interactive voice tool*.
Making them more patient would trade this project's first constraint, latency,
for a background problem's benefit. Only a caller where nobody is waiting should
raise it. The backoff there is now exponential rather than flat, which is the
part that was actually wrong.

**Lesson.** A retry policy is not "did we retry" but "did we retry *later than
the thing that broke us*". The 2026-07-02 fix answered the first question and
failed the second, and nothing in the system could tell — because a fault that
appears once per run is invisible per-run. It took a cross-session view to make
a months-old, already-"fixed" defect legible.

**Files:** `src/http_util.py` (`attempts` param, exponential backoff),
`src/outlook_calendar.py` (3 attempts, 8s timeout, exponential backoff),
`tests/retry_policy_test.py` (NEW, 12 assertions).

**Gate:** 52/52 green.

## M95 — Outcomes measured and declined; the prompt fix that beat them — 2026-07-28

**Outcome: Phase 3 is NOT shipped, and that is the finding.** Managed Agents
supports rubric-graded Outcomes (`user.define_outcome` — verified live that SDK
0.97.0 accepts it). Measured against a plain prompt on three research questions
with mechanically-checkable criteria:

| approach | mean words | total time |
|---|---|---|
| plain (original prompt) | 317 | 82s |
| **tightened prompt** | **126** | **68s** |
| Outcome (rubric-graded) | 92 | 537s |

The Outcome loop *works* — it enforced brevity the prompt only requested. But a
one-line prompt change captured most of that for **free**, and Outcomes bought
the last 126→92 words for **8× the wall clock** and a re-run of the whole task
on every failed grade. Not worth it here. The probe is kept
(`scripts/outcome_probe.py`) so the trade-off can be re-decided if the pricing
or the workload changes.

**The prompt bug.** The original instruction was *"keep it under 200 words
unless the task genuinely needs more"* — and the escape clause was taken **every
single time**, averaging 317 words. Restated as a ceiling with no exemption:
126. An instruction with an opt-out is a suggestion.

**Honest flaw in the measurement.** One of the three criteria — "cites a URL" —
failed in all six runs, because the agent's own system prompt forbids markup and
tells it the answer may be *spoken*. The criterion was testing my rubric against
the agent's design, not testing Outcomes. Excluding it, the real comparison is
plain 3/6 vs Outcome 5/6. Recorded rather than quietly dropped, because the
headline number (33% → 44%) is the less honest one.

**AND THE BUG THIS UNCOVERED — a silent no-op.** The agent's `system` lives
**server-side** on the persistent agent object, and `ensure_resources` returns
early whenever the ids are cached. So editing `_SYSTEM` in this repo changed
*nothing* for an already-provisioned agent: tune the prompt, redeploy, and the
agent keeps the old one indefinitely. There was no error and no signal — the
only way to see it was to read the live agent back, which is how it was found.

Fixed by fingerprinting the prompt (sha256, stored beside the ids) and pushing
`agents.update` when it drifts. Two details earned their own assertions:

- **`version` is REQUIRED** by SDK 0.97.0's `agents.update()` — the
  optimistic-concurrency check. Omitting it raises rather than defaulting to
  last-write-wins, which is how the first attempt failed. The current version is
  read and handed straight back; this agent has one writer.
- **A failed update must not record the fingerprint.** Saving it would mark the
  drift resolved and never retry — recreating the exact silent no-op the whole
  mechanism exists to prevent.

Verified end to end: the live agent moved to **version 2** carrying the new
ceiling, and a second call is a no-op.

**Lesson.** Three SDK surprises across M91–M95 (`initial_events` unsupported,
sessions start idle, `update` needs `version`) all shared a shape: the hermetic
tests passed because the fake agreed with my *idea* of the API. Every one was
caught by pointing the code at the real service. Mocks verify wiring; only the
live call verifies the contract.

**Files:** `src/background_agent.py` (hard ceiling, `_system_fingerprint`,
drift-triggered `agents.update`), `scripts/outcome_probe.py` (NEW instrument),
`tests/background_tasks_test.py` (+8, now 57).

**Gate:** 52/52 green.

## M96 — structured outputs on the background jobs — 2026-07-28

**What shipped.** The prediction miner and resolver now constrain their replies
with `output_config.format` + a JSON schema, instead of asking for JSON in prose
and digging it back out with a regex.

**Checked before changing it: this had never failed.** Zero occurrences of
`miner output unparseable` across both log files. The old extractor searched for
a bracketed span with a **greedy** pattern — spanning the first `[` to the last
`]` anywhere in the reply — but no production reply had ever tripped it.

So why change it? Because unlike M95's Outcomes (8× cost for a marginal gain),
this is a *parameter*, not a mechanism: no extra latency, no extra tokens,
arguably fewer (the prompt no longer has to beg for clean JSON). Cheap insurance
against a latent failure class is worth buying; expensive insurance against a
marginal one is not. Both decisions came from measuring first; they differ only
in what the measurement said.

`_extract_json` is **kept as a fallback** rather than deleted — if a future
model or SDK ignores the constraint, a degraded parse beats a lost mining cycle.

**Schema notes.** Nullable fields use `anyOf`, not a JSON-Schema *type array*,
which is outside the supported subset. The miner's array is wrapped in an object
because the constraint is specified against an object root.

**Verified live that structured output composes with a SERVER-SIDE tool**: the
resolver keeps `web_search` and still returns a clean verdict — the response
carries `server_tool_use`, `web_search_tool_result` and `text` blocks, with the
text being exactly the JSON document.

**A test caught what a live test could not.** The first edit half-applied: the
resolver got its `output_config`, the miner silently did not. The live miner run
still returned perfect JSON — *via the regex fallback* — so it looked like proof
and was nothing of the sort. The unit assertion "miner sends
output_config.format" failed and exposed it. A live test proves the feature
works; only an assertion on the call proves it is the feature you think is
running.

**Two self-inflicted test bugs, both worth recording.** `_api_key()` returned an
empty string in a bare test process, so the miner returned early and never built
a client — the assertions failed for a reason unrelated to the code. And the fix
was not `os.environ.setdefault`, which is a silent no-op when the variable
already exists **as an empty string**; it needed direct assignment.

**Files:** `src/predictions.py` (schemas, `_parse_structured`, both call sites),
`tests/structured_output_test.py` (NEW, 17 assertions).

**Gate:** 53/53 green.

## M97 — the HUD learns to show background research — 2026-07-28

**What shipped.** The ambient overlay (M84) gains a research line under the
security badge: `researching...` with a gentle animated ellipsis while a task is
working, `1 report ready` when something has finished but not yet been spoken,
and **nothing at all** when there is neither.

**Why this and not a general "richer HUD".** Background research is the one
thing Jarvis does that has *no visible surface anywhere*. A voice turn lights
the orb, an announce speaks, an armed watcher flips a badge — but an M91 task
runs off-box for minutes to hours with no window, no sound and no tray change.
That is precisely the gap an ambient overlay exists to fill, so it earned the
space; a decorative addition would not have.

**The empty state is a feature.** A permanent `0 tasks` row would be clutter on
a translucent overlay that sits over the user's actual work. The line renders
only when it has something to say.

**Fan-out follows the existing chain** — `JarvisUI.set_research_indicator` →
`JarvisConsole.set_research` → `JarvisHUD.set_research` — the same path
`set_armed` already takes, so there is one pattern for "ambient state" rather
than two.

One deliberate difference: this indicator is driven by the
**BackgroundTaskManager poll thread**, not by a UI event. So the facade wraps
the call in a try/except that the other indicators do not need. A poll loop that
died on a Tk teardown would silently stop delivering finished research — the one
failure this feature cannot have. Asserted directly: *"a dead UI cannot kill the
poll loop."*

**Verified structurally, not visually.** As with M84 itself, the overlay was
built without being able to see it: the tests assert the state transitions
(1 active → 0 active/1 pending → cleared after delivery) and that the research
colour is distinct from the armed red. The rendering itself needs a human
glance.

**Files:** `src/hud.py` (research line + colour + setter), `src/console.py`
(thread-safe bridge), `src/ui.py` (facade), `src/background_tasks.py`
(`_push_ui` each cycle), `main.py` (pass the UI through),
`tests/background_tasks_test.py` (+7, now 63).

**Gate:** 53/53 green.
