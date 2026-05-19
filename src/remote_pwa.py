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
    <input id="txt" type="text" autocomplete="off" autocapitalize="sentences"
           placeholder="Message Jarvis…" enterkeyhint="send">
    <button id="send">Send</button>
  </div>
</footer>

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

  function connect() {
    if (!token) { $("setup").classList.add("show"); return; }
    $("setup").classList.remove("show");
    setPill("connecting…");
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => ws.send(JSON.stringify({ type:"auth", token }));
    ws.onmessage = ev => {
      let m; try { m = JSON.parse(ev.data); } catch { return; }
      switch (m.type) {
        case "auth_ok": retry = 0; setPill("connected", "#15803d"); break;
        case "auth_fail":
          localStorage.removeItem("jarvisToken"); token = "";
          $("err").textContent = "Token rejected. Try again.";
          $("setup").classList.add("show"); ws.close(); break;
        case "snapshot":
          setArmed(m.armed);
          setPill(m.state || "connected",
                  STATE_COLOR[m.state] ?? "#15803d"); break;
        case "state": setPill(m.state, STATE_COLOR[m.state] ?? ""); break;
        case "armed": setArmed(m.armed); break;
        case "user":   line("you", m.text); break;
        case "jarvis": line("jar", m.text); break;
        case "system": line("sys", m.text); break;
      }
    };
    ws.onclose = () => {
      setPill("reconnecting…", "#b45309");
      retry = Math.min(retry + 1, 6);
      setTimeout(connect, 400 * retry);   // capped backoff
    };
    ws.onerror = () => { try { ws.close(); } catch {} };
  }

  const live = () => ws && ws.readyState === 1;
  function sendText() {
    const t = $("txt").value.trim();
    if (!t || !live()) return;
    ws.send(JSON.stringify({ type:"text", content:t }));
    $("txt").value = "";
  }
  function control(action) {
    if (live()) ws.send(JSON.stringify({ type:"control", action }));
  }

  $("send").onclick = sendText;
  $("txt").addEventListener("keydown", e => { if (e.key === "Enter") sendText(); });
  $("arm").onclick = () => control("arm");
  $("disarm").onclick = () => control("disarm");
  $("save").onclick = () => {
    const v = $("tok").value.trim();
    if (!v) return;
    token = v; localStorage.setItem("jarvisToken", v);
    $("err").textContent = ""; connect();
  };
  // Mobile Safari suspends the socket when backgrounded — re-dial on return.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !live()) connect();
  });

  connect();
})();
</script>
</body>
</html>"""
