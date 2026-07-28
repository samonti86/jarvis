# Jarvis

[![gate](https://github.com/samonti86/jarvis/actions/workflows/ci.yml/badge.svg)](https://github.com/samonti86/jarvis/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

An always-on voice assistant for Windows. Say "Hey Jarvis," ask a question, get a
spoken answer — with a 36-tool agentic layer behind it that can search the web, run
code in a sandbox, read a calendar, watch a homelab, and see through a webcam.

Wake-word detection and speech-to-text run **locally**. Only the transcribed text ever
leaves the machine.

```text
mic ──► openWakeWord ──► faster-whisper ──► Claude ──► edge-tts ──► speakers
        (local)          (local)           (agentic    (streamed)
                                            tool loop)
```

Roughly 29k lines of Python across 66 modules. The full engineering log — the design
trade-offs, the post-mortems, and the conclusions that turned out to be wrong — is in
[docs/MILESTONES.md](docs/MILESTONES.md).

---

## The parts worth reading

**An agentic tool loop, not a chatbot.** Claude is given 36 tools and decides which to
call, chaining them across iterations. Ask *"what's the most played movie in my Plex
library?"* and it composes four separate MCP calls to get there — there is no
hand-coded path for that question.

**Latency is the design constraint.** The reply is streamed from the model, chunked
into sentences, and synthesized to speech *while the model is still generating*, so the
first audible word lands about a second in rather than after the full response. The
system prompt is prompt-cached; per-turn context (speaker identity, current time) rides
a second, deliberately **un-cached** block, so a change there costs ~20 fresh tokens
instead of invalidating the whole cached prefix.

**Least privilege, enforced in code.** Not every caller gets every tool. A request
arriving from the phone client or the Discord bridge is served a *restricted* tool
surface — no shell, no filesystem, no code execution, no self-update. That boundary is
enforced at two independent server-side gates: the tool list is filtered before the
model ever sees it, *and* the executor refuses denied tools by name if one is somehow
reached. It is never enforced by asking the model nicely in the prompt. Mutating
actions are confirmation-gated, and the gate carries the information needed to evaluate
it — a confirmation you can't assess is theater, not safety.

**Arbitrary code gets a boundary, not an allowlist.** `run_code` executes
Claude-written Python in an ephemeral Podman container: no network, no host mount,
CPU/memory/PID caps, hard-killed at 30 seconds. You cannot allowlist arbitrary code, so
it is contained instead.

**Everything fails soft.** Any single subsystem — TTS, the Plex bridge, the GPU
transcription server, the calendar feed — can fail without taking down the listening
loop. Degrade and log; never crash the thing the user is talking to.

**A regression gate, because a bug a test would have caught earns a test.**
`scripts/run_all_tests.py` runs 45 gates — syntax, module wiring, a JS structural
check, and 42 test suites totalling ~1,000 assertions — and must be green before
anything ships. CI runs the *same* command on every push; the five gates that need
a native ML toolchain or Windows SAPI are skipped **by name**, never silently
folded into the pass count.

---

## Capabilities

| Area | |
| --- | --- |
| **Voice** | Wake word, barge-in mid-reply, a follow-up window (no re-triggering), hands-free conversation mode, live two-way interpreter mode |
| **Knowledge** | Web search and fetch, a private RAG corpus (SQLite FTS5 + local embeddings, fused with Reciprocal Rank Fusion), full-text recall over past conversations |
| **Awareness** | Webcam vision, screen capture, ambient sound classification (PANNs), speaker identification from voice embeddings |
| **Proactive** | Severe-weather alerts, calendar pre-event reminders, homelab up/down monitoring, cross-domain synthesis, a morning briefing and an evening wrap |
| **System** | Read-only diagnostics shell (18-verb allowlist), confirmation-gated service control, sandboxed code execution, self-update, a crash watchdog |
| **Clients** | Desktop console, system tray, an iOS PWA with push-to-talk over WSS, a Discord bot bridge |
| **Data** | Weather, sports, news, film/TV, games, WolframAlpha, Plex (over MCP) |

Multilingual by design: the language is detected per turn, and the reply comes back in
that language, in a matching voice.

---

## Stack

Python 3.12 · [openWakeWord](https://github.com/dscripka/openWakeWord) ·
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) · Anthropic SDK (streaming,
prompt caching, tool use) · [edge-tts](https://github.com/rany2/edge-tts) with a
`pyttsx3` offline fallback · `sounddevice` · YOLOv8n · Resemblyzer · Podman ·
`websockets` · MCP

Windows-native rather than WSL, deliberately: WSL2 audio bridging is not reliable
enough for always-on real-time capture.

---

## Quick start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env    # add your ANTHROPIC_API_KEY
python main.py                 # or: pythonw jarvis.pyw  (silent, no console)
```

Then say **"Hey Jarvis."** Everything beyond the core loop is optional and off by
default — see [.env.example](.env.example), which documents every setting.

Run the gate with `python scripts/run_all_tests.py`.

---

## Layout

```text
main.py                 event loop: wake -> STT -> LLM -> TTS; owns conversation state
src/llm.py              Anthropic client, streaming, tool loop, per-origin tool boundary
src/wake_word.py        openWakeWord
src/speech_to_text.py   faster-whisper (local, or offloaded to a GPU host)
src/text_to_speech.py   edge-tts + fallback, sentence-chunked streaming
src/security.py         vision security mode (person detection, challenge/response)
src/*.py                the tool and subsystem modules (66 in total)
scripts/*_test.py       the regression suites
docs/MILESTONES.md      the engineering log
```

## License

Personal project, published as a portfolio piece.
