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
  "background_color": "#0b0f14",
  "theme_color": "#0b0f14",
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
<meta name="theme-color" content="#0b0f14">
<link rel="manifest" href="/manifest.json">
<title>Jarvis</title>
<style>
  :root { --bg:#0b0f14; --panel:#121821; --line:#1f2a37; --tx:#e6edf3;
          --dim:#8b98a5; --accent:#3b82f6; --ok:#22c55e; --warn:#f59e0b;
          --bad:#ef4444; }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--tx);
    font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif; }
  body { display:flex; flex-direction:column;
    padding:env(safe-area-inset-top) env(safe-area-inset-right)
            env(safe-area-inset-bottom) env(safe-area-inset-left); }
  header { display:flex; align-items:center; gap:10px; padding:12px 16px;
    border-bottom:1px solid var(--line); background:var(--panel); }
  header b { font-weight:600; letter-spacing:.5px; }
  #pill { margin-left:auto; font-size:13px; padding:4px 10px; border-radius:999px;
    background:#26313f; color:var(--dim); }
  #armed { font-size:12px; padding:4px 9px; border-radius:999px;
    border:1px solid var(--line); color:var(--dim); }
  #armed.on { color:#fff; background:var(--bad); border-color:var(--bad); }
  #log { flex:1; overflow-y:auto; padding:14px 16px; }
  .m { margin:9px 0; max-width:88%; padding:9px 12px; border-radius:12px;
    white-space:pre-wrap; word-wrap:break-word; }
  .you { margin-left:auto; background:var(--accent); color:#fff;
    border-bottom-right-radius:3px; }
  .jar { background:var(--panel); border:1px solid var(--line);
    border-bottom-left-radius:3px; }
  .sys { background:transparent; color:var(--dim); font-size:13px;
    text-align:center; max-width:100%; }
  footer { border-top:1px solid var(--line); background:var(--panel);
    padding:10px 12px 14px; display:flex; flex-direction:column; gap:9px; }
  .row { display:flex; gap:8px; }
  input,button { font:inherit; border-radius:10px; border:1px solid var(--line);
    background:#0f141b; color:var(--tx); padding:12px 13px; }
  #txt { flex:1; min-width:0; }
  button { background:#1c2735; cursor:pointer; }
  button:active { opacity:.7; }
  #send { background:var(--accent); border-color:var(--accent); color:#fff;
    font-weight:600; }
  .ctl { flex:1; padding:13px; font-weight:600; }
  #arm { background:#15281b; border-color:#225133; color:#7ee2a8; }
  #disarm { background:#2a1717; border-color:#5a2a2a; color:#f0a3a3; }
  #setup { position:fixed; inset:0; background:var(--bg); display:none;
    flex-direction:column; justify-content:center; padding:28px; gap:14px; }
  #setup.show { display:flex; }
  #setup h2 { margin:0 0 4px; font-weight:600; }
  #setup p { margin:0; color:var(--dim); font-size:14px; }
  .hidden { display:none !important; }
  #spk { background:#1c2735; }
  #spk.on { background:#15281b; border-color:#225133; color:#7ee2a8; }
</style>
</head>
<body>
<header>
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
  const log = $("log"), pill = $("pill"), armedEl = $("armed");
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

  function line(cls, text) {
    const d = document.createElement("div");
    d.className = "m " + cls; d.textContent = text;
    log.appendChild(d); log.scrollTop = log.scrollHeight;
  }
  function setPill(t, color) {
    pill.textContent = t;
    pill.style.background = color || "#26313f";
    pill.style.color = color ? "#fff" : "var(--dim)";
  }
  const STATE_COLOR = { idle:"", listening:"#2563eb", thinking:"#b45309",
                        speaking:"#15803d" };
  function setArmed(on) {
    armedEl.textContent = on ? "ARMED" : "UNARMED";
    armedEl.classList.toggle("on", !!on);
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
          localStorage.removeItem("jarvisToken"); token = "";
          $("err").textContent = "Token rejected. Try again.";
          $("setup").classList.add("show"); sock.close(); break;
        case "snapshot":
          setArmed(m.armed);
          setPill(m.state || "connected",
                  STATE_COLOR[m.state] ?? "#15803d"); break;
        case "state": setPill(m.state, STATE_COLOR[m.state] ?? ""); break;
        case "armed": setArmed(m.armed); break;
        case "user":   line("you", m.text); break;
        case "jarvis": line("jar", m.text); break;
        case "system": line("sys", m.text); break;
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
