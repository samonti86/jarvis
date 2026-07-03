"""M48.1 — the single-page PWA served by src/remote_console.py.

Kept in its own module so the server logic stays readable (same reason the
system prompt isn't inlined in llm.py). Fully self-contained: no external
fonts/CDNs/frameworks — it must work with nothing but the LAN PC reachable
(the whole point is "no internet needed, just the brain on the LAN").

iOS notes baked in:
  - apple-mobile-web-app-capable + status-bar meta → "Add to Home Screen"
    launches it chromeless, like an app (the client-agnostic PWA call).
  - viewport-fit=cover + env(safe-area-inset-*) → not eaten by the notch.
  - WS auto-reconnects on visibilitychange: mobile Safari suspends the
    socket when backgrounded; we re-dial when the user reopens it. (This
    is also why phone always-on wake word is impossible — push-to-talk is
    M48.3; v1 is type + arm/disarm.)
"""

PWA_MANIFEST = """{
  "name": "Jarvis",
  "short_name": "Jarvis",
  "display": "standalone",
  "background_color": "#0a0e14",
  "theme_color": "#0a0e14",
  "start_url": "/"
}"""


PWA_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Jarvis">
<meta name="theme-color" content="#0a0e14">
<link rel="manifest" href="/manifest.json">
<title>Jarvis</title>
<style>
  /* M57 — "Arc Reactor" palette: matches the desktop console (src/console.py)
     — near-black + electric arc-reactor cyan, Iron-Man gold for THINKING. */
  :root { --bg:#0a0e14; --panel:#10161f; --line:#1d2b3a; --border:#21465a;
          --tx:#cdd9e5; --dim:#5b6b7f; --accent:#22d3ee; --hot:#7df3ff;
          --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444; }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--tx);
    font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif; }
  body { display:flex; flex-direction:column;
    padding:env(safe-area-inset-top) env(safe-area-inset-right)
            env(safe-area-inset-bottom) env(safe-area-inset-left); }
  header { display:flex; align-items:center; gap:10px; padding:11px 16px;
    border-bottom:1px solid var(--border); background:var(--panel);
    box-shadow:0 1px 14px rgba(34,211,238,.10); }
  header b { font-weight:700; letter-spacing:3.5px; font-size:17px;
    color:var(--accent); text-shadow:0 0 9px rgba(34,211,238,.55); }

  /* Arc-reactor orb — the desktop console's centrepiece, header-sized for
     the phone: concentric rings, a rotating arc, a breathing glow. State
     class (.s-*) swaps the colour; idle is the default. */
  .orb { width:30px; height:30px; border-radius:50%; position:relative;
    flex:0 0 auto; --orb:#3f5468; --orbhot:#6b8299;
    background:radial-gradient(circle at 50% 50%,
      var(--orbhot) 0%, var(--orb) 40%, rgba(0,0,0,0) 72%);
    animation:orbBreathe 3.4s ease-in-out infinite; }
  .orb::before { content:""; position:absolute; inset:-3px; border-radius:50%;
    border:2px solid transparent; border-top-color:var(--orbhot);
    border-right-color:var(--orb); animation:orbSpin 3s linear infinite; }
  .orb::after { content:""; position:absolute; inset:5px; border-radius:50%;
    border:1px solid var(--orb); opacity:.55; }
  .orb.s-listening { --orb:#22d3ee; --orbhot:#7df3ff; }
  .orb.s-thinking  { --orb:#f5b942; --orbhot:#ffd98a; }
  .orb.s-speaking  { --orb:#34e6d0; --orbhot:#9af7ec;
    animation:orbBreathe 1.1s ease-in-out infinite; }
  @keyframes orbSpin { to { transform:rotate(360deg); } }
  @keyframes orbBreathe { 0%,100% { box-shadow:0 0 7px var(--orb); }
    50% { box-shadow:0 0 17px var(--orbhot); } }

  #pill { margin-left:auto; font-size:12px; padding:4px 11px; border-radius:999px;
    background:#16202c; color:var(--dim); border:1px solid var(--line); }
  #armed { font-size:11px; padding:4px 9px; border-radius:999px;
    border:1px solid var(--line); color:var(--dim); letter-spacing:.5px; }
  #armed.on { color:#fff; background:var(--bad); border-color:var(--bad); }
  #log { flex:1; overflow-y:auto; padding:14px 16px; }
  .m { margin:9px 0; max-width:88%; padding:9px 12px; border-radius:13px;
    white-space:pre-wrap; word-wrap:break-word; }
  .you { margin-left:auto; background:#13313b; border:1px solid #205265;
    color:#e2f6fa; border-bottom-right-radius:3px; }
  .jar { background:var(--panel); border:1px solid var(--border);
    color:var(--tx); border-bottom-left-radius:3px;
    box-shadow:0 0 10px rgba(34,211,238,.06); }
  .sys { background:transparent; color:var(--dim); font-size:13px;
    text-align:center; max-width:100%; }
  footer { border-top:1px solid var(--border); background:var(--panel);
    padding:10px 12px 14px; display:flex; flex-direction:column; gap:9px; }
  .row { display:flex; gap:8px; }
  input,button { font:inherit; border-radius:10px; border:1px solid var(--line);
    background:#0c141d; color:var(--tx); padding:12px 13px; }
  #txt { flex:1; min-width:0; border-color:var(--border); }
  /* M48.3: iOS long-press on a button = text-selection / context callout
     (the spike's locked finding); touch-action manipulation also kills the
     300ms double-tap-zoom delay. Applied globally — every button benefits. */
  button { background:#16212e; cursor:pointer; border-color:var(--border);
    -webkit-user-select:none; -webkit-touch-callout:none;
    touch-action:manipulation; user-select:none; }
  button:active { opacity:.7; }
  #send { background:var(--accent); border-color:var(--accent); color:#06222a;
    font-weight:700; }
  .ctl { flex:1; padding:13px; font-weight:600; }
  #arm { background:#10241a; border-color:#225133; color:#7ee2a8; }
  #disarm { background:#241313; border-color:#5a2a2a; color:#f0a3a3; }
  #setup { position:fixed; inset:0; background:var(--bg); display:none;
    flex-direction:column; justify-content:center; padding:28px; gap:14px; }
  #setup.show { display:flex; }
  #setup h2 { margin:0 0 4px; font-weight:700; color:var(--accent);
    letter-spacing:1px; }
  #setup p { margin:0; color:var(--dim); font-size:14px; }
  .hidden { display:none !important; }
  #spk { background:#16212e; }
  #spk.on { background:#10241a; border-color:#225133; color:#7ee2a8; }
  /* M48.3 — mic in two visual states. Idle = neutral; recording = the
     disarm-button red palette ("this is hot, tap to stop" — the same visual
     vocabulary used for active danger elsewhere in the UI). */
  #mic { background:#16212e; }
  #mic.on { background:#241313; border-color:#5a2a2a; color:#f0a3a3; }
</style>
</head>
<body>
<header>
  <div id="orb" class="orb"></div>
  <b>JARVIS</b>
  <span id="armed">UNARMED</span>
  <span id="pill">connecting…</span>
</header>
<div id="log"></div>
<footer>
  <div class="row">
    <button id="arm" class="ctl">Arm</button>
    <button id="disarm" class="ctl">Disarm</button>
  </div>
  <div class="row">
    <button id="spk" class="ctl" aria-pressed="false">🔇 Speak replies: off</button>
  </div>
  <div class="row">
    <button id="prb" class="ctl" style="display:none">▶︎ Tap to hear reply</button>
  </div>
  <div class="row">
    <button id="mic" class="ctl" aria-pressed="false">🎤 Tap to talk</button>
  </div>
  <div class="row">
    <input id="txt" type="text" autocomplete="off" autocapitalize="sentences"
           placeholder="Message Jarvis…" enterkeyhint="send">
    <button id="send">Send</button>
  </div>
</footer>
<audio id="ae" playsinline></audio>

<div id="setup">
  <h2>Connect to Jarvis</h2>
  <p>Enter the remote token (JARVIS_REMOTE_TOKEN from the PC's .env). It's
     saved on this device only.</p>
  <input id="tok" type="password" autocomplete="off" placeholder="Remote token">
  <button id="save" class="ctl">Connect</button>
  <p id="err" style="color:var(--bad)"></p>
</div>

<script>
(() => {
  const $ = id => document.getElementById(id);
  const log = $("log"), pill = $("pill"), armedEl = $("armed"), orb = $("orb");
  let ws = null, retry = 0, token = localStorage.getItem("jarvisToken") || "";

  // M48.2b — opt-in "Speak replies" (default OFF; phone-text stays silent
  // unless you turn this on; audio is unicast to THIS phone, never the PC).
  let speakOn = localStorage.getItem("jarvisSpeak") === "1";
  const ae = document.getElementById("ae");
  let _ctx = null, lastClip = null;
  function logerr(where, e) {
    line("sys", "audio " + where + " ✗ " +
      (e && e.name ? e.name + ": " + (e.message || "") : e));
  }
  // M48.2b iOS audio — evidence-driven (real-device test 2026-05-19): the
  // reply MP3 plays on iOS Edge via <audio> AFTER the full LLM round-trip
  // (confirmed by ear). The earlier silent-clip keepalive was net-negative
  // — the embedded WAV threw NotSupportedError (iOS <audio> rejects it) and
  // the ended→re-prime replayed the MP3 on loop (a lock-screen media
  // session that wouldn't stop). BOTH removed. Keep only a FILE-FREE
  // AudioContext resume on the gesture (cannot NotSupportedError; harmless
  // warm-up). The reply plays via <audio> in playReply(); a guaranteed
  // manual ▶︎ tap (fresh gesture) is the fallback if a given auto-play is
  // ever refused. No silent clip, no loop, no perpetual media session.
  function primeAudio() {
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) { if (!_ctx) _ctx = new AC();
        if (_ctx.state === "suspended") _ctx.resume(); }
    } catch (e) { /* file-free warm-up only; never blocks sending */ }
  }
  function playReply(b64, mime) {
    lastClip = { b64, mime: mime || "audio/mpeg" };
    $("prb").style.display = "none";
    try {
      ae.loop = false; ae.pause();  // hard stop anything prior; play ONCE
      ae.src = "data:" + lastClip.mime + ";base64," + b64;
      const p = ae.play();
      if (p) p.then(() => {}).catch(e => onPlayBlocked(e));
    } catch (e) { onPlayBlocked(e); }
  }
  function onPlayBlocked(e) {
    // NotAllowedError is EXPECTED on iOS http-LAN (deferred autoplay is
    // refused). Don't spam the scary platform string every reply — show a
    // concise, actionable line + the guaranteed manual button. Surface the
    // raw error only for genuinely unexpected failures.
    if (e && e.name === "NotAllowedError")
      line("sys", "🔊 Reply ready — tap ▶︎ below to hear it.");
    else
      logerr("play", e);
    $("prb").style.display = "block";
  }
  function setSpk() {
    const b = $("spk");
    b.textContent = speakOn ? "🔊 Speak replies: ON" : "🔇 Speak replies: off";
    b.classList.toggle("on", speakOn);
    b.setAttribute("aria-pressed", speakOn ? "true" : "false");
  }

  // ─────────────────────────────────────────────────────────────────────
  // M48.3 — push-to-talk (tap-toggle, NOT press-and-hold; the spike's
  // locked finding: iOS long-press on a button = selection/callout).
  // The MediaRecorder API is the right capture path on iOS once the page
  // is on secure context (M48.2c) — getUserMedia is now allowed and we
  // get whatever codec Safari supports (audio/mp4 in practice). The blob
  // ships base64-over-WS to reuse the existing auth boundary; voice-in
  // ALWAYS gets voice-out (the toggle continues to gate text only).
  // ─────────────────────────────────────────────────────────────────────
  let recState = "idle";          // "idle" | "recording" | "uploading"
  let mediaStream = null;
  let mediaRecorder = null;
  let recChunks = [];
  let recCapTimer = null;
  let micTranscribeTimer = null;  // safety reset if no transcript arrives
  const REC_CAP_MS = 60000;       // 60s hard cap (longer = two prompts)
  const MIC_TRANSCRIBE_TIMEOUT_MS = 30000;  // generous; mic returns to idle if no transcript by then

  // M48.3 — gesture-time `<audio>` element unlock. The server serves a
  // ~200ms silent MP3 at /silence.mp3 (generated once at startup via PyAV;
  // the bytes are cacheable + immutable, so the browser fetches once then
  // reuses forever). Playing it on the SAME <audio> element during the
  // mic-start tap user-activates the element, which lets the eventual reply
  // MP3 (which arrives OUTSIDE any gesture) play without iOS refusing it
  // with NotAllowedError. This was the gap that broke voice-in→voice-out
  // auto-play on the Safari TAB path (the home-screen PWA's lifecycle
  // is more permissive — engine ≠ app-shell, the M48.2b refinement).
  // iOS rejects synthetic WAV data URLs (NotSupportedError, M48.2b lesson)
  // but accepts MP3 — so MP3 is the right mime. Fires ONCE per mic-start;
  // no `ended`→re-prime loop (the M48.2b regression we explicitly avoided).
  // If /silence.mp3 404s (PyAV MP3 encoder absent server-side), play()
  // rejects and the existing ▶︎ fallback fires — zero regression.
  function unlockAudioElement() {
    try {
      ae.loop = false; ae.pause();
      ae.src = "/silence.mp3";
      const p = ae.play();
      if (p) p.catch(() => {});  // refusal is non-fatal; fall back to ▶︎ later
    } catch (e) { /* never block recording on an audio-unlock hiccup */ }
  }

  function setMicUI(state) {
    const b = $("mic");
    if (state === "recording") {
      b.textContent = "🔴 Recording — tap to stop";
      b.classList.add("on");
      b.setAttribute("aria-pressed", "true");
    } else if (state === "uploading") {
      b.textContent = "⏳ Transcribing…";
      b.classList.remove("on");
      b.setAttribute("aria-pressed", "false");
    } else {
      b.textContent = "🎤 Tap to talk";
      b.classList.remove("on");
      b.setAttribute("aria-pressed", "false");
    }
  }

  function blobToBase64(blob) {
    // FileReader is the only cross-iOS-version reliable path; the newer
    // blob.arrayBuffer().then(buf => btoa(...)) chokes on long blobs with
    // a stack-overflow on the spread, and TextDecoder mangles binary.
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onloadend = () => {
        const dataUrl = String(fr.result || "");
        const i = dataUrl.indexOf(",");
        resolve(i >= 0 ? dataUrl.slice(i + 1) : "");
      };
      fr.onerror = () => reject(fr.error || new Error("FileReader failed"));
      fr.readAsDataURL(blob);
    });
  }

  function stopTracks() {
    if (mediaStream) {
      try { mediaStream.getTracks().forEach(t => t.stop()); } catch (e) {}
      mediaStream = null;
    }
  }

  async function startRecording() {
    if (recState !== "idle") return;
    if (!ws || ws.readyState !== 1) {
      line("sys", "mic: not connected — tap status pill to reconnect");
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      line("sys", "mic: this browser can't capture audio (need HTTPS / secure context)");
      return;
    }
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      // NotAllowedError on first use = "user denied permission" or the iOS
      // gesture rules misfired; SecurityError happens on insecure origins
      // (but M48.2c HTTPS means we shouldn't hit that). Either way, a calm
      // line, not a stack trace.
      line("sys", "mic: " + (e && e.name ? e.name : "permission denied"));
      return;
    }
    // Pick the first mimeType the browser admits to supporting. Safari =
    // audio/mp4; others usually offer audio/webm;codecs=opus.
    let mt = "";
    if (window.MediaRecorder) {
      for (const c of ["audio/mp4", "audio/webm;codecs=opus", "audio/webm"]) {
        if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(c)) {
          mt = c; break;
        }
      }
    } else {
      stopTracks();
      line("sys", "mic: MediaRecorder not available on this browser");
      return;
    }
    try {
      mediaRecorder = mt
        ? new MediaRecorder(mediaStream, { mimeType: mt })
        : new MediaRecorder(mediaStream);
    } catch (e) {
      stopTracks();
      line("sys", "mic init failed: " + (e && e.name ? e.name : e));
      return;
    }
    recChunks = [];
    mediaRecorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) recChunks.push(ev.data);
    };
    mediaRecorder.onstop = onRecordingStop;
    mediaRecorder.onerror = (ev) => logerr("recorder", ev && ev.error);
    try {
      mediaRecorder.start();
    } catch (e) {
      stopTracks();
      line("sys", "mic start failed: " + (e && e.name ? e.name : e));
      return;
    }
    recState = "recording";
    setMicUI("recording");
    // M48.3 — INSIDE THE GESTURE: warm AudioContext AND user-activate the
    // <audio> element by playing a 200ms silent MP3. iOS requires the
    // element to have a successful gesture-time play() once before it can
    // play() from a deferred (reply-arrived-N-seconds-later) context. With
    // this, voice-in → voice-out auto-plays as designed; without it, every
    // first voice turn would fall back to the ▶︎ button (Safari tab is
    // materially stricter than the home-screen PWA — engine ≠ app-shell).
    primeAudio();
    unlockAudioElement();
    // Hard cap: stop on our terms before Safari decides the tab has been
    // recording "too long" and silently kills the stream.
    recCapTimer = setTimeout(() => {
      line("sys", "mic: 60s cap reached — stopping");
      stopRecording();
    }, REC_CAP_MS);
  }

  function stopRecording() {
    if (recState !== "recording") return;
    if (recCapTimer) { clearTimeout(recCapTimer); recCapTimer = null; }
    try { mediaRecorder.stop(); } catch (e) { /* fires onstop or onerror */ }
  }

  async function onRecordingStop() {
    stopTracks();
    recState = "uploading";
    setMicUI("uploading");
    const mime = (mediaRecorder && mediaRecorder.mimeType) || "audio/mp4";
    const blob = new Blob(recChunks, { type: mime });
    recChunks = [];
    let sent = false;
    try {
      if (!blob || blob.size === 0) {
        line("sys", "mic: empty recording");
        return;
      }
      const b64 = await blobToBase64(blob);
      if (!ws || ws.readyState !== 1) {
        line("sys", "mic: disconnected before upload");
        return;
      }
      ws.send(JSON.stringify({ type: "audio", mime: mime, b64: b64 }));
      sent = true;
    } catch (e) {
      logerr("upload", e);
    } finally {
      if (!sent) {
        // Upload never reached the server — drop straight back to idle so
        // the user can retry without staring at a stuck "Transcribing…".
        recState = "idle";
        setMicUI("idle");
      } else {
        // Stay in "uploading" / ⏳ Transcribing… until the server fans the
        // transcribed `user` line (or a `system` line on no-speech) back
        // to us — THAT's the right moment to flip to idle (Jarvis received
        // it, working on it). M36 GPU offload makes this very fast; on local
        // CPU it can be several seconds. Either way the user sees a real
        // "in flight" signal instead of an instant flip-back. Safety net:
        // if no transcript by MIC_TRANSCRIBE_TIMEOUT_MS, reset anyway.
        if (micTranscribeTimer) clearTimeout(micTranscribeTimer);
        micTranscribeTimer = setTimeout(() => {
          micTranscribeTimer = null;
          if (recState === "uploading") {
            line("sys", "mic: no transcript after " +
                 (MIC_TRANSCRIBE_TIMEOUT_MS / 1000) + "s — resetting");
            recState = "idle";
            setMicUI("idle");
          }
        }, MIC_TRANSCRIBE_TIMEOUT_MS);
      }
    }
  }

  // Called from ws.onmessage when a "user" or "system" line arrives — that's
  // our signal that the server has finished processing the just-uploaded
  // phone-voice clip (transcription happened, either yielding text or
  // "(no speech detected)"). Either way, return the mic to idle.
  function clearMicTranscribingState() {
    if (recState !== "uploading") return;
    if (micTranscribeTimer) {
      clearTimeout(micTranscribeTimer); micTranscribeTimer = null;
    }
    recState = "idle";
    setMicUI("idle");
  }

  function line(cls, text) {
    const d = document.createElement("div");
    d.className = "m " + cls; d.textContent = text;
    log.appendChild(d);
    // 2026-07-02 QA: cap the transcript DOM — an installed PWA left
    // connected for days accumulated every message + reconnect line forever.
    while (log.childNodes.length > 500) log.removeChild(log.firstChild);
    log.scrollTop = log.scrollHeight;
  }
  function setPill(t, color) {
    pill.textContent = t;
    pill.style.background = color || "#26313f";
    pill.style.color = color ? "#fff" : "var(--dim)";
  }
  // Pill colours — darker on-palette shades so white pill text stays
  // readable (the header orb carries the vivid state colour now).
  const STATE_COLOR = { idle:"", listening:"#0e7490", thinking:"#a16207",
                        speaking:"#0f766e" };
  function setArmed(on) {
    armedEl.textContent = on ? "ARMED" : "UNARMED";
    armedEl.classList.toggle("on", !!on);
  }
  // M57 — reflect Jarvis's state on the header arc-reactor orb. Whitelisted:
  // an unknown / empty / connection-only state falls back to the idle look.
  function setOrb(state) {
    const s = String(state || "").toLowerCase();
    orb.className = ["listening", "thinking", "speaking"].includes(s)
      ? "orb s-" + s : "orb";
  }

  // dial(): the single re-entrant entry point. Everything that wants a
  // connection calls dial(), never connect() directly. The guard skips a
  // redial only when a socket is OPEN, or is CONNECTING *with the watchdog
  // still arming it*. A CONNECTING socket with no live watchdog is a frozen
  // zombie — iOS leaves a post-resume handshake stuck at readyState 0
  // FOREVER, firing neither onopen nor onclose; the old guard
  // (`readyState===0` ⇒ skip) then wedged the console on "connecting…"
  // permanently (the reopen-won't-reconnect bug — connection-stuck-in-
  // CONNECTING never triggers the onclose-driven retry). The watchdog is
  // what makes a stuck handshake recoverable instead of terminal.
  let connWD = null;             // per-attempt connect watchdog
  const CONN_TIMEOUT_MS = 6000;  // iOS post-resume handshakes hang silently
  function clearWD() { if (connWD) { clearTimeout(connWD); connWD = null; } }

  function dial() {
    if (ws && ws.readyState === 1) return;            // already OPEN
    if (ws && ws.readyState === 0 && connWD) return;  // watched connect in flight
    connect();
  }

  function connect() {
    if (!token) { $("setup").classList.add("show"); return; }
    clearWD();
    // Tear down any stale socket first (bfcache restore leaves a dead one
    // whose handlers must not fire mid-redial).
    try { if (ws) { ws.onclose = null; ws.onerror = null; ws.close(); } }
    catch (e) { /* already dead */ }
    $("setup").classList.remove("show");
    setPill("connecting…");
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${location.host}/ws`);
    ws = sock;
    // Watchdog: if THIS socket hasn't reached OPEN within the timeout, the
    // handshake is wedged (the classic iOS post-resume hang). Force-close
    // it — that fires onclose → the backoff retry — and clears the
    // in-flight flag so dial() is free to try again instead of skipping
    // forever. Captures `sock` so a late timer can't kill a newer socket.
    connWD = setTimeout(() => {
      connWD = null;
      if (sock.readyState !== 1) {
        line("sys", "connection timed out — retrying");
        try { sock.close(); } catch (e) {}
      }
    }, CONN_TIMEOUT_MS);
    sock.onopen = () => sock.send(JSON.stringify({ type:"auth", token }));
    sock.onmessage = ev => {
      let m; try { m = JSON.parse(ev.data); } catch { return; }
      switch (m.type) {
        case "auth_ok": clearWD(); retry = 0;
          setPill("connected", "#15803d"); break;
        case "auth_fail":
          clearWD();
          if (m.reason === "timeout") {
            // The SERVER timed out waiting for our first auth frame (slow
            // cellular, a backgrounded tab) — our token is probably fine. Keep
            // it and let onclose's backoff retry, instead of wiping a good
            // token and forcing the user to re-enter it.
            line("sys", "auth timed out — retrying");
            sock.close();
          } else {
            localStorage.removeItem("jarvisToken"); token = "";
            $("err").textContent = "Token rejected. Try again.";
            $("setup").classList.add("show"); sock.close();
          }
          break;
        case "snapshot":
          setArmed(m.armed);
          setPill(m.state || "connected",
                  STATE_COLOR[m.state] ?? "#0f766e");
          setOrb(m.state); break;
        case "state":
          setPill(m.state, STATE_COLOR[m.state] ?? "");
          setOrb(m.state); break;
        case "armed": setArmed(m.armed); break;
        case "user":
          line("you", m.text);
          clearMicTranscribingState();  // M48.3 — transcript arrived
          break;
        case "jarvis": line("jar", m.text); break;
        case "system":
          line("sys", m.text);
          clearMicTranscribingState();  // M48.3 — e.g. "(no speech detected)"
          break;
        case "speak":  // M48.2b: unicast reply audio for THIS phone
          playReply(m.b64, m.mime);
          break;
      }
    };
    sock.onclose = () => {
      clearWD();
      setPill("reconnecting… (tap)", "#b45309");
      retry = Math.min(retry + 1, 6);
      setTimeout(dial, 400 * retry);   // guarded; capped backoff
    };
    sock.onerror = () => { try { sock.close(); } catch (e) {} };
  }

  const live = () => ws && ws.readyState === 1;
  function sendText() {
    const t = $("txt").value.trim();
    if (!t || !live()) return;
    // Inside the Send tap/Enter gesture: start the silent keepalive so the
    // <audio> element stays user-activated until the (possibly many-seconds-
    // later) reply arrives and we hot-swap its src.
    if (speakOn) primeAudio();
    ws.send(JSON.stringify({ type:"text", content:t, speak: speakOn }));
    $("txt").value = "";
  }
  function control(action) {
    if (live()) ws.send(JSON.stringify({ type:"control", action }));
  }

  $("send").onclick = sendText;
  $("txt").addEventListener("keydown", e => { if (e.key === "Enter") sendText(); });
  // Toggle is itself a user gesture — unlock audio when turning it ON so a
  // reply can play even if the first send happens fast.
  $("spk").onclick = () => {
    speakOn = !speakOn;
    localStorage.setItem("jarvisSpeak", speakOn ? "1" : "0");
    if (speakOn) primeAudio();
    setSpk();
  };
  // Manual fallback: a fresh tap is guaranteed user-activation, so this
  // always works even if iOS refused the deferred auto-play.
  $("prb").onclick = () => {
    if (lastClip) {
      ae.loop = false;
      ae.src = "data:" + lastClip.mime + ";base64," + lastClip.b64;
      ae.play().catch(e => logerr("manual", e));
    }
    $("prb").style.display = "none";
  };
  // (No ended→re-prime: that re-played the reply MP3 on loop. Each new
  //  Send is its own gesture and re-primes; a reply plays exactly once.)
  setSpk();  // reflect persisted state at load
  $("arm").onclick = () => control("arm");
  $("disarm").onclick = () => control("disarm");
  // M48.3 — push-to-talk: TAP-TOGGLE, not press-and-hold (iOS long-press
  // = selection/callout, which would fight the user). Each tap flips the
  // state machine; uploading is a transient state (no input accepted).
  $("mic").onclick = () => {
    if (recState === "idle") startRecording();
    else if (recState === "recording") stopRecording();
    // recState === "uploading" → ignore taps until the round-trip completes
  };
  // The status pill is a GUARANTEED manual reconnect: a tap is a fresh user
  // gesture (iOS is far more permissive about WS creation inside one than in
  // throttled background JS), so even if every automatic redial is refused,
  // one tap always recovers. This is the reconnect analog of the ▶︎ button —
  // ship the reliable manual path now; HTTPS/Tailscale makes the automatic
  // path robust later (same strategic call as the audio one-tap).
  pill.style.cursor = "pointer";
  pill.title = "Tap to reconnect";
  pill.addEventListener("click", () => {
    if (ws && ws.readyState === 1) return;   // already connected
    retry = 0; clearWD();
    try { if (ws) { ws.onclose = null; ws.close(); } } catch (e) {}
    ws = null; connect();
  });
  $("save").onclick = () => {
    const v = $("tok").value.trim();
    if (!v) return;
    token = v; localStorage.setItem("jarvisToken", v);
    $("err").textContent = ""; connect();
  };
  // iOS suspends the socket when backgrounded — re-dial on return.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) dial();
  });
  // bfcache: closing+reopening the browser RESTORES a frozen page with a
  // DEAD ws. visibilitychange does NOT fire for that — pageshow does, with
  // e.persisted === true. Without this, the only recovery was clearing
  // cache (forcing a fresh load). Hard-reset the stale socket + re-dial.
  window.addEventListener("pageshow", e => {
    if (e.persisted) {
      try { if (ws) { ws.onclose = null; ws.close(); } } catch (x) {}
      ws = null; retry = 0; dial();
    }
  });
  window.addEventListener("online", dial);   // network came back

  dial();
})();
</script>
</body>
</html>"""
