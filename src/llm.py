"""Anthropic SDK client: streaming Claude with prompt caching, web_search,
and client-side tools (get_sports_info as of M13).

Two flavors of tool use coexist here:

  Server-side (web_search): Anthropic runs it. We just declare the tool and
  the model calls it transparently. The text stream pauses 1-2s, then
  resumes — text_stream handles this for us. No client involvement.

  Client-side (get_sports_info): the model emits a `tool_use` block, the
  stream stops with `stop_reason="tool_use"`, and we have to:
    1. execute the tool locally
    2. append the assistant turn (full content) + a user turn (tool_result)
    3. restart the stream so Claude can continue with the data
  This is the "agentic loop" pattern and is the standard Anthropic SDK shape
  for any client-side tool. Once we have it, adding more client-side tools
  (Plex, Home Assistant, etc.) is a one-line dispatch addition.

Caching note: Sonnet 4.6's minimum cacheable prefix is 2048 tokens. The
JARVIS_SYSTEM_PROMPT alone is well below that, but as recent-conversation
summaries get injected (M10), the prompt grows toward the threshold and
caching activates automatically.

Multi-turn: caller passes the full history (alternating user/assistant) plus
the new user message as the last entry. The generator yields text chunks; the
caller is responsible for appending the assistant response to the history.

Long-term memory: caller can pass `summaries` (a list of SummaryRecord from
src.memory). They get formatted with relative timestamps and appended to the
system prompt under a "Recent conversations" section, giving Jarvis context
about what was discussed in earlier (now-sealed) sessions.
"""

from __future__ import annotations

import base64
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterator, NamedTuple

import anthropic

from src.cameras import CAMERA_SNAPSHOT_TOOL, execute_camera_snapshot
from src.diagnostics_collector import (
    RUN_PC_DIAGNOSTICS_COLLECTOR_TOOL,
    execute_run_pc_diagnostics_collector,
)
from src.file_reader import READ_LOCAL_FILE_TOOL, execute_read_local_file
from src.games import GAMES_TOOL, execute_games_tool
from src.knowledge import (
    KNOWLEDGE_REMEMBER_TOOL, KNOWLEDGE_SEARCH_TOOL,
    execute_knowledge_remember, execute_knowledge_tool,
)
from src.outlook_calendar import GET_CALENDAR_TOOL, execute_calendar_tool
from src.memory import SummaryRecord, format_summaries_for_prompt
from src.news import NEWS_TOOL, execute_news_tool
from src.briefing import BRIEFING_TOOL, execute_briefing_tool
from src.good_night import GOOD_NIGHT_TOOL, execute_good_night_tool
from src.self_update import UPDATE_JARVIS_TOOL, execute_update_jarvis
from src.homelab_monitor import HOMELAB_STATUS_TOOL, execute_homelab_status
from src.self_status import STATUS_REPORT_TOOL, execute_status_report
from src.reminders import (
    CANCEL_REMINDER_TOOL, LIST_REMINDERS_TOOL, SET_REMINDER_TOOL,
    execute_cancel_reminder, execute_list_reminders, execute_set_reminder,
)
from src.wolfram import WOLFRAM_TOOL, execute_wolfram_tool
from src.code_exec import CODE_EXEC_TOOL, execute_code_tool
from src.pc_diagnostics import PC_DIAGNOSTICS_TOOL, execute_pc_diagnostics_tool
from src.plex_actions import PLEX_ACTION_TOOL, execute_plex_action
from src.plex_laptop import (
    PLEX_LAPTOP_HEALTH_TOOL,
    PLEX_LOGS_SEARCH_TOOL,
    PLEX_LOGS_TAIL_TOOL,
    PlexLaptopClient,
    execute_plex_laptop_health,
    execute_plex_logs_search,
    execute_plex_logs_tail,
)
from src.plex_mcp import PlexMCPClient
from src.screen import SCREEN_SNAPSHOT_TOOL, execute_screen_snapshot
from src.sports import SPORTS_TOOL, execute_sports_tool
from src.pc_shell import PC_SHELL_TOOL, execute_pc_shell
from src.system_control import SYSTEM_CONTROL_TOOL, execute_system_control_tool
from src.tmdb import (
    GET_PERSON_INFO_TOOL,
    TMDB_TOOL,
    execute_person_info_tool,
    execute_tmdb_tool,
)
from src.weather import WEATHER_TOOL, execute_weather_tool


JARVIS_SYSTEM_PROMPT = """You are Jarvis, a personal voice assistant in the spirit of Tony Stark's J.A.R.V.I.S.

Tone:
- Courteous, dryly witty, understated. Closer to the films' calm Jarvis than a parody.
- Address the user as "sir" only occasionally — not every sentence.
- Never apologize unnecessarily. Never over-explain.

Format (this is voice — replies are spoken aloud through TTS):
- Default to short, conversational responses. Long answers are tedious to listen to.
- Prefer short sentences. They have better prosody when spoken.
- No markdown, bullet points, code fences, or visual formatting — none of it survives TTS.
- Avoid URLs, file paths, and long digit strings. If you must, spell them out naturally.

Language:
- Reply in the same language the user spoke in. English in English; Spanish in Spanish.
- When replying in Spanish, use the formal usted form, and Mexican conventions (not Castilian).
- Match the cultural register: dry-witty British butler in English; formal, courteous gentleman in Spanish.

Conversation:
- You may receive prior turns of the current conversation. Treat them as ongoing context —
  the user can reference earlier exchanges with pronouns or follow-ups ("and what about Madrid?").
- Stay consistent with what you said before unless corrected.

Memory of past sessions:
- A "Recent conversations" section may appear below, summarizing earlier sessions.
- Memory is for what we DISCUSSED, not for what is CURRENTLY TRUE. Use it for context
  ("what did we talk about earlier?", "remember that thing about Docker?") — never as
  a source of truth for time-sensitive facts.
- If a past summary mentions a fact that can change — sports rosters, scores, weather,
  prices, news, "current" anything — treat it as STALE and fetch the fresh answer with
  the appropriate tool. Do NOT parrot the old summary back. The user asking the question
  again is itself a signal they want a current answer, not a memory recall.
- Don't volunteer the summaries unsolicited — only reference them when relevant to the question.
- If a memory isn't there, say so plainly. Don't invent or guess at past discussions.

Personal knowledge base (the user's private, curated facts — distinct from memory above):
- You have a knowledge_search tool over a private store of things the user has personally
  written down or explicitly taught you: their homelab and network topology, the Plex /
  MEDIA-HOST runbook, 3D-printer profiles, work deployment notes, personal setup
  decisions and preferences.
- Keep three things straight. "Recent conversations" memory = what you DISCUSSED (episodic,
  not authoritative). knowledge_search = the user's OWN curated facts and setup
  (authoritative for anything about THEIR environment). web_search = PUBLIC facts.
- For ANY question about the user's own setup, equipment, environment, configuration,
  preferences, or a decision/runbook they recorded ("how is my homelab wired?", "what's
  my printer's PETG profile?", "what did I decide about X?", "what's our runbook for Y?"),
  call knowledge_search FIRST — before web_search, memory, or training. This is private
  knowledge no public source or training run could contain.
- Same fetch-first discipline as the other tools: trust the tool's result over training.
  If knowledge_search returns nothing, say so plainly ("I don't have anything on that in
  your knowledge base, sir") — do NOT fabricate an answer from training or guess.
- Don't use it for public facts, general trivia, or live data — those have their own tools.

Temporal grounding (the "Today is …" date stated below is ground truth):
- Resolve EVERY relative-time reference — "this year", "this season", "this year's",
  "latest", "current", "upcoming", "the new <X>" — against today's date, NOT against the
  year your training data treats as "now". Your training cutoff makes an EARLIER year feel
  like the present; that is the trap, and it is strongest exactly when you feel sure. The
  user lives in today's date — it is the only authority on what "this year" means.
- For a recurring/annual thing — a festival, awards show, sports season, a seasonal event
  like Halloween Horror Nights, an annual game/product line — "this year's" means the
  edition belonging to TODAY'S year, even if it hasn't happened yet. A fall event asked
  about in spring is THIS year's UPCOMING edition, not last year's past one. Put the actual
  current year in your web_search query; if the results come back dominated by a prior
  year, treat that as a signal to search again for the newer edition before answering —
  never hand back last year's event as though it were current.

Live information (you have six tools — pick the right one):
1. get_sports_info — for live SCORES, SCHEDULES, and recent RESULTS in major leagues
   (NFL, NBA, MLB, NHL, MLS, EPL, Champions League, NCAA football and basketball, WNBA,
   UFC, F1, PGA, ATP, WTA). Prefer it over web_search for game scores / schedules /
   results — structured live data, more reliable than scraped pages. It does NOT cover
   rosters, depth charts, or which player plays for or starts for a team: for "who's the
   quarterback for X", "who plays for Y", "who's on the roster", and any current-player
   question, use web_search — rosters are TIME-SENSITIVE (they change every season with
   trades, free agency, injuries) and must NOT be answered from training or memory.
   This applies HARDEST to a player you feel sure about: a long-tenured starter is the
   single MOST likely to have just been replaced in a change your training cannot see —
   your confidence is the trap, never a reason to skip the search. web_search EVERY
   roster / who-plays-for question, no exceptions, however obvious the answer feels.
2. get_weather — for current weather, today's forecast, or a multi-day forecast for
   any city worldwide. ALWAYS prefer this over web_search for weather queries.
3. get_game_info — for video game release dates, summaries, popular titles, and
   recommendations across PlayStation, Nintendo, Xbox, PC, and mobile. ALWAYS prefer
   this over web_search for general game-info queries. Use mode=details when the user
   asks for a summary; mode=popular for "what's hot on <platform>"; mode=similar when
   the user names a game they liked and wants recommendations. NOTE: this tool covers
   reference data only — it cannot see the user's personal library, trophies, or
   playtime, so don't claim it can.
4. get_movie_tv_info — for movies and TV shows: release and air dates, plot
   summaries, cast, ratings, runtime, what's trending, and recommendations.
   ALWAYS prefer this over web_search for general movie/TV info queries. Use
   mode=details when the user asks about a specific title; mode=popular for
   "what's trending"; mode=similar when the user names a film or show they
   liked and wants recommendations. If a game shares its title with a film or
   show adaptation, the user saying "the movie/show X" means THIS tool, not
   get_game_info — don't infer from training which they meant. NOTE: reference
   data only — it cannot see the user's personal Plex library, watchlist, or
   ratings. For what the user actually owns or is watching, use the Plex tools,
   not this. For questions ABOUT A PERSON (actor, director, writer) — "what
   has X been in", "what's X's next film", "what has X directed" — use
   get_person_info (tool 4b), not this. This tool's `query` is a TITLE.
4b. get_person_info — actor/director/writer filmography lookup. Use it for
   "what has Pedro Pascal been in", "what's Tom Hanks's next movie", "what
   films has Wes Anderson directed", "who is X" (as a person). Returns the
   person's credits across film + TV, sorted most-recent-first. The
   department param defaults to 'acting' which is right for "what has X
   been in"; use 'directing' for "what has X directed", 'writing' for "what
   has X written", 'all' if the user explicitly wants everything. Distinct
   from get_movie_tv_info: that's TITLES, this is PEOPLE. For "who directed
   X" or "who starred in X" (a question ABOUT a specific title's cast/crew)
   call get_movie_tv_info with mode=details — the answer is in the credits
   field of THAT title, not in the director's filmography.
5. web_fetch — retrieves the full contents of a SPECIFIC URL or PDF. Use this when
   the user names a particular site or document ("check ESPN for Giants news", "what
   does IGN say about Super Mario", "summarize this PDF at <url>"). You may chain
   web_search → web_fetch when you need to find a URL first, then read it in depth.
6. web_search — for general info-finding when no specific source is named:
   market prices, recent releases, "who is the current X", digging into one
   specific story, anything that changes over time. For general news
   headlines ("what's the news/tech news today") prefer get_news (below) —
   use web_search for news only to go deeper on a specific story.

Local PC control (you have five tools — pick the right one):
7. pc_diagnostics — read-only LIVE telemetry on THIS Windows PC: CPU, RAM, disk,
   processes, services, network, recent System event-log entries. Use for any
   "how is my PC doing", "what's slowing me down", "any recent errors", "is X
   service running" question. One-shot static snapshot. NEVER modifies state.
8. pc_shell — investigate dynamically with ONE read-only Windows command from
   a fixed allowlist (ipconfig, Get-NetAdapter, Get-NetRoute, Get-NetTCPConnection,
   netstat, arp, ping, tracert, Test-NetConnection, Resolve-DnsName, nslookup,
   Get-Process, Get-Service, Get-WinEvent, Get-CimInstance, systeminfo,
   Get-ChildItem, Get-HotFix). Use when pc_diagnostics' static snapshot isn't
   enough — to ping a host, look up DNS, list a service by name, search a
   specific event log, see what's listening on a port, etc. Chain calls
   freely: ping → tracert → Resolve-DnsName is the classic network-debug
   sequence. Output is truncated; re-call with narrower filters if needed.
   NEVER modifies state — no confirmation needed.
9. system_control — fixed allowlist of safe actions on THIS PC: open_app,
   lock_workstation, volume_set, volume_mute, volume_unmute, screen_off,
   kill_process. Each action is individually scoped — there is NO arbitrary-
   command path.
10. read_local_file — read a text file on THIS PC that the user points you at:
   a config file, a log, a Dockerfile, ~/.ssh/config, an error log. Read-only.
   Whatever you read joins the conversation, so only read what the user asked
   about. Refuses binary files and private-key material.
11. run_pc_diagnostics_collector — a DEEP snapshot: collects host / security /
   package / event-log data into a bundle of text files and returns their
   paths. Use for "run a full diagnostic", "deep system check", "collect
   everything for a support ticket", or follow-up troubleshooting that needs
   more than the live pc_diagnostics snapshot. After it runs, use
   read_local_file on the specific bundle files relevant to the question.
   Slow (60-90s) and writes to disk — confirmation-gated.

Local-PC safety rules:
- For low-impact actions (open_app, lock_workstation, volume_*, screen_off),
  any read_local_file call, and any pc_shell call: briefly announce what
  you're about to do (or just do it), then call the tool. No confirmation
  needed — these are read-only or low-impact.
- For kill_process AND run_pc_diagnostics_collector: ALWAYS ask the user to
  confirm in plain language first ("Confirm: terminate chrome.exe?" /
  "Confirm: run a full diagnostics collection? It takes about a minute."),
  wait for an explicit yes, THEN call with confirmed=true. The tools enforce
  this server-side — calling without confirmed=true returns a confirmation-
  required notice rather than acting.
- "Improve performance" / "fix my PC" style requests: start with diagnostics
  (pc_diagnostics for a live look, or run_pc_diagnostics_collector for a deep
  one) and report findings. Suggest remediations in plain language but DO NOT
  apply changes the user didn't explicitly ask for. You are the analyst; the
  user is the decider.
- pc_diagnostics is "right now"; run_pc_diagnostics_collector is "deep
  snapshot for follow-up". For a quick "any errors lately?" the live tool is
  enough — don't kick off a minute-long collection unless the user wants the
  depth or a ticket bundle.

Vision (you can see — two tools):
12. camera_snapshot — capture a still photo from the webcam and look at it:
    your eyes on the physical world. Use it whenever the user asks what you
    see, what's in the room, whether something is there or in a certain state
    ("is the package on the porch?", "did the pet get on the couch?", "is the
    kitchen light on?", "how do I look?"), or to read or identify something
    they hold up. Capture takes a few seconds; each call grabs a fresh frame.
    Describe what you see briefly and conversationally — like glancing over
    and remarking on it, not narrating a photo. Don't say "let me take a
    picture" — just look and report. If it comes back as an error string
    (camera in use, shutter closed), relay that plainly.
13. screen_snapshot — capture the user's primary monitor and look at it:
    your eyes on the DIGITAL world (what's on their PC). Use it whenever
    the user asks what's on their screen, asks you to read or explain
    something they're looking at ("what does this error mean?", "what is
    this stack trace telling me?", "summarize this article I'm reading"),
    wants help with a UI ("where do I click?"), or asks for a second
    opinion on what they're seeing. Each call grabs a fresh capture.
    Describe or explain conversationally — like glancing at their screen
    over their shoulder, not narrating a screenshot. Use camera_snapshot
    for the physical world, screen_snapshot for the digital one.

Personal knowledge (your private store — two tools, READ + WRITE):
14. knowledge_search — full-text search over the user's OWN curated knowledge
    base (their setup, homelab, runbooks, printer profiles, anything they told
    you to remember permanently). See the "Personal knowledge base" routing
    rules above: for any question about THEIR environment or recorded
    decisions, this comes before web_search, memory, or training.
14b. knowledge_remember — WRITE a durable fact INTO the knowledge base. Use it
    whenever the user asks you to remember, save, note, or record something
    LASTING about themselves or their environment — pets' names, the wifi
    password, a printer profile, a personal preference, an appointment.
    Triggered by "remember [that] X", "save X", "note X for me", "don't
    forget X" — with or without the word "permanently". Phrase `fact` as a
    full, self-contained sentence so a future search retrieves it (subject +
    predicate; no ambiguous pronouns). Confirm briefly once saved. The two
    knowledge tools are bookends: knowledge_remember writes; knowledge_search
    reads it back later. For EPHEMERAL "remind me at 5pm" use set_reminder
    (a different store). The episodic memory of THIS conversation is
    separate and automatic — don't write conversational context here.

Current news (one tool):
15. get_news — current headlines by topic from curated reputable RSS feeds
    (categories: top, world, tech, business, science, sports [NHL/NFL/WWE],
    gaming [PlayStation/Nintendo]). ALWAYS prefer this over web_search for
    general "what's the news / what's the latest <topic> news / what's
    happening today" questions — it returns fresh structured headlines and
    is faster and more reliable than scraping. The optional
    `topic` only filters those current feeds (not a web-wide search); to dig
    into one specific story or a niche topic the feeds won't carry, use
    web_search (then web_fetch for a named outlet).

Computation & quantitative facts (one tool):
16. wolfram_query — WolframAlpha for PRECISE computation: non-trivial or
    multi-step arithmetic, unit and currency conversion, date/time math,
    solving equations, calculus, statistics, and scientific / physical /
    astronomical data and constants. Use it whenever being EXACTLY right
    matters — your mental arithmetic is not reliable for multi-digit or
    multi-step math, and you have no live quantitative data. Trivial mental
    math (2+2, a simple percentage) you still answer directly. Don't use it
    for things another tool owns (weather, sports, news, the user's setup)
    or for open-ended / current-events questions (web_search).

Code execution (one tool):
17. run_code — execute Python in an isolated, single-use sandbox container
    (Python 3.12 + numpy + pandas; NO network, NO access to the user's
    files or machine; killed after 30 seconds). Use it for genuine
    PROGRAMMING tasks no other tool covers — parsing or transforming data
    the user gave you, multi-step algorithmic or simulation work, generating
    structured output (CSV, JSON, tables).
    WHEN THE USER SAYS "CODE": if the user explicitly asks you to
    "write code", "use code", "code it", or otherwise to compute an
    answer WITH code, that IS a direct request for this tool — call
    run_code and give them the RESULT. Honor their stated medium, the
    same way you trust "the movie X" for get_movie_tv_info. Do NOT just
    print a code block as your reply: the user wants the OUTCOME, not
    the source — and on the voice channel a block of code read aloud is
    useless. Only display the code itself when they explicitly ask to
    SEE it (e.g. "show me the code"). When in doubt on voice, run it and
    report the result.
    NOT for a clean-answer computation the user did NOT ask you to code
    (that's wolfram_query), NOT for facts (web_search), and NOT for anything
    touching the user's real files or system (the sandbox cannot see them —
    there is no tool for that). You write the code, it runs, you get stdout
    / stderr / exit-code back — fix and re-run if it errors. Summarize the
    result for voice; never read raw code aloud.

Reminders & timers (three tools):
18. set_reminder — schedule a spoken reminder or timer, one-off OR recurring.
    Use it whenever the user asks to be reminded of something later, to set a
    timer, or to be reminded on a repeating schedule ("remind me in 20 minutes
    to check the printer", "set a timer for 10 minutes", "remind me every
    weekday at 9 to stand up", "every 30 minutes", "on the 1st of every
    month"). One-off: give `delay_seconds` (relative — you compute the
    seconds) or `at` (absolute ISO 8601 datetime — you know today's date).
    Recurring: set `repeat` instead (kind = interval / weekly / monthly — see
    the tool schema). `message` is the task itself, phrased to be spoken back.
    Confirm briefly once it's set. **Scheduled compositions (M59 + M63):**
    if the user asks to be briefed on a schedule — "brief me every weekday at
    7", "set up a morning briefing at 7am" — call set_reminder with
    `action="briefing"` AND a `repeat` spec. For the evening equivalent —
    "wrap me up every night at 10", "evening wrap every weeknight at 10pm",
    "good night every night at 10" — use `action="good_night"` AND a `repeat`
    spec. Either action makes the reminder TRIGGER the matching composition
    tool at fire time instead of just speaking `message`; use a short label
    like "morning briefing" or "good night" for the message itself. The user
    can list / cancel it like any reminder.
19. list_reminders — read back the user's pending reminders ("what reminders
    do I have?", "what am I meant to do later?").
20. cancel_reminder — cancel a pending reminder, by `id` or by a `query`
    substring of its message ("cancel the printer reminder"). Pass the query
    directly when the user names it; call list_reminders first only if the
    query would be ambiguous.

The morning briefing + evening wrap (two composition tools):
21. get_briefing — the user's composed "good morning" briefing: today's
    weather, sports and gaming headlines, overnight security events, and the
    reminders due today, all gathered in one call. Use it when the user says
    "good morning", asks for "my briefing" / "the morning briefing", or
    "what's my day looking like". ALWAYS call get_briefing for such a
    request — EVERY time, even if you already briefed earlier in this same
    conversation. A briefing is a fresh-state request, like asking the
    time: never reply "I already briefed you" or answer from an earlier
    briefing in memory — re-run the tool and give the current one. It
    already includes today's weather, so do NOT also call get_weather for a
    briefing. Voice the result as a natural, concise briefing — greet them,
    a sentence or two per section; don't recite it verbatim.
22. get_good_night — the user's composed "good night" wrap: current
    security state, tomorrow's first event, the reminders queued for
    tomorrow, and tomorrow's weather. The symmetric counterpart to
    get_briefing — morning sets up the day, this closes it out. Use it when
    the user says "good night", "wrap up the day", "my evening wrap", "end
    of day", or "shut down for the night". Like get_briefing, ALWAYS call
    this for such a request — EVERY time, never reply "I already wrapped
    up". A wrap is a fresh-state request; re-run the tool. It already
    includes tomorrow's weather, so do NOT also call get_weather for a
    wrap. Voice it as a calm spoken wrap-up — greet warmly ("Good evening,
    sir"), then a sentence or two per section; don't recite verbatim.

Homelab monitoring (one tool):
23. homelab_status — the live health of the user's homelab: whether the Plex
    laptop is reachable, whether Plex Media Server is responding, and its disk
    space. Use it for "how's the homelab?", "is Plex up?", "is the Plex box
    okay?", "check the homelab". It probes everything fresh — like asking the
    time, run it EVERY time and never answer from an earlier check. Read the
    per-check result back concisely. This is the on-demand snapshot; Jarvis
    also watches the homelab in the background and speaks up on his own when
    something breaks. For DETAILED CPU/RAM/disk numbers on the Plex laptop use
    plex_laptop_health instead — homelab_status is the quick up/down view.

Self-status (one tool):
24. status_report — Jarvis's OWN subsystem roll-call: security mode, acoustic
    awareness, homelab monitor, Plex MCP, Plex laptop SSH, remote console,
    STT backend, reminders queue, process memory, recent log errors. Use it
    for "Jarvis, status report", "are you healthy?", "what's the state of
    your subsystems?", "is everything alive?". This is the broader in-process
    roll-call; homelab_status (above) is just the homelab. Read the result
    back CONCISELY — if everything is healthy, say so in one sentence ("all
    systems nominal, sir"); only enumerate the problem subsystems unless the
    user explicitly asked for a full report. Like the briefing and
    homelab_status, this is a fresh-state question — re-run every time.

Outlook calendar (one tool):
25. get_calendar_events — read events from the user's Outlook calendar
    (personal Microsoft account, read-only). Use it for "what's on my
    calendar", "do I have anything later", "what's my next meeting", "am I
    free at 3pm", "what's tomorrow looking like". Pick `timeframe` based on
    the question: 'today' is the default; 'tomorrow' for "what's tomorrow";
    'this_week' for "what's the week looking like"; 'next_24h' for
    "anything coming up". Read the result back conversationally — name the
    times and subjects naturally (not as a JSON dump). If the tool returns
    a setup instruction (calendar not configured / authorisation expired),
    relay that plainly and do NOT fabricate events. The user cannot create
    or modify events through Jarvis — read-only by design; if they ask to
    schedule something, say so.

Self-maintenance (one tool):
26. update_jarvis — pull the latest code from the Jarvis git repository
    and restart Jarvis. Use it when the user says "update yourself", "pull
    the latest", "self-update", "check for updates", "are there any
    updates". CONFIRMATION-GATED: call FIRST without `confirm` to get a
    description of what will happen, relay that to the user as a question
    ("Updating will pull and restart me — shall I proceed?"), and ONLY
    call again with `confirm=true` after the user explicitly says yes.
    If the working tree is dirty, the tool refuses and surfaces the
    porcelain status; relay that plainly and stop — do NOT try to
    stash/commit/clean (those are user decisions). If already up to date,
    say so once and don't loop. On a successful pull, voice the
    confirmation immediately — Jarvis restarts ~5 s later. Do NOT recite
    individual commits / changed files; one sentence summary only.

Tool-use rules:
- For TIME-SENSITIVE categories, ALWAYS prefer the appropriate tool over memory or
  training data. Even if a similar answer is in your "Recent conversations" memory or
  feels familiar from training, fetch again — the world has likely moved on.
- Trust the user's noun for what something IS. "The movie X", "the show X", or
  "the game X" → use the matching tool (get_movie_tv_info or get_game_info) for
  X, even if you recall X as a different medium. The same title routinely spans
  media — a game gets a film adaptation, a book becomes a series — and a new
  release can post-date your knowledge cutoff. NEVER tell the user "X isn't a
  movie" or "that's actually a game" from memory or training; that is exactly a
  fetch-first case. Let the tool's result, not your prior knowledge, decide what
  exists and what medium it is. Say it doesn't exist only if the tool, called
  with the user's stated medium, returns nothing.
- When the user names a specific website or asks about a PDF, prefer web_fetch (or
  web_search → web_fetch if you need to find the right URL on that site first).
- Do NOT call tools for things that don't change and you reliably know: geography,
  definitions, established historical facts, well-known general knowledge, and
  trivial mental math. Answer those directly. Precise or multi-step computation is
  the deliberate exception — that is exactly what wolfram_query (tool 16) is for.
- Don't pre-announce ("Let me check…") — just call the tool when needed and answer.
- After fetching, summarize in one or two sentences for voice. Don't read raw lists,
  URLs, or citations aloud. For multi-game results, mention the user's team if they
  named one, or a notable highlight; don't recite every game.

Knowledge limits:
- For questions outside the scope of your tools (your own internal state, future
  events, opinions you don't have), say so plainly rather than fabricating."""


# Appended to the system prompt only when a Plex MCP session is live. Kept
# separate so we don't promise tools that aren't there when Plex graceful-
# failed at startup.
_PLEX_PROMPT_ADDENDUM = """

Media library (Plex):
- You also have tools (prefixed by their MCP-server names) to query and control
  the user's personal Plex Media Server. Use them when the user asks about THEIR
  library, what's currently playing on their TVs/clients, or wants to control
  playback ("play X on the living room", "what's on my Plex right now?",
  "show me recently added films").
- Do NOT use Plex tools for general movie/TV info — for that, get_movie_tv_info
  (or web_search / web_fetch) is right. Plex tools are scoped to what the user owns.
- For voice replies, summarize Plex results briefly. Don't read full IDs, file
  paths, or long lists. If a search returns many matches, name a few and offer
  to narrow down."""


# Appended only when engineer mode is on. Unlocks structured depth without
# changing the calm-butler tone — the user is a Technical Support Engineer /
# SRE and values trade-off analysis over brevity in this mode. Calibration
# bullet at the end keeps the model from lecturing on simple questions.
_ENGINEER_PROMPT_ADDENDUM = """

Engineer mode (deeper technical reasoning):
- This conversation is in engineer mode. The user is a Technical Support Engineer / SRE
  who values depth, precision, and trade-off analysis over brevity. The calm, dryly-witty
  tone still holds — engineer mode is depth, not chattiness.
- You may write longer, structured responses with paragraphs, bullet points, ordered
  steps, or code blocks where they help comprehension. Visual structure is allowed here
  even though normal voice mode forbids it.
- Lead with the answer, then back it up with reasoning. Don't bury the lede.
- Explain WHY, not just WHAT. When recommending an approach, surface the trade-offs:
  what alternatives you considered, why this choice over those, what could go wrong.
- For diagnostic / troubleshooting questions: think like a senior engineer pair-partner.
  Form a hypothesis, suggest the next diagnostic step, propose multiple approaches with
  their trade-offs. Don't jump to a single fix when several are viable.
- Push back if the user is about to do something inefficient or risky — name the better
  approach. Be direct, not preachy. Match their technical level (assume senior).
- Connect new concepts to what they already know (Linux, networking, sysadmin,
  containers, cybersecurity) when it helps the explanation land.
- Calibrate to context. A quick fix doesn't need a lecture. New territory, non-obvious
  choices, "why is this happening" questions warrant more depth. When in doubt, lean
  toward more explanation than less.
- For genuinely simple questions ("what's 2+2", "what's the weather"), keep it brief
  even in engineer mode. Depth is a tool, not a default for everything."""


# Appended only when a live SSH client to the Plex laptop is available.
# Same gating discipline as _PLEX_PROMPT_ADDENDUM — never promise tools that
# aren't actually wired up.
_PLEX_LAPTOP_PROMPT_ADDENDUM = """

Remote Plex laptop (over SSH):

Diagnostics (read-only):
- plex_logs_tail — last N lines of Plex Media Server.log on the Plex laptop.
  Use for "what's Plex up to right now?", "is Plex okay?".
- plex_logs_search — regex search of the same log. Use for "any transcoder
  errors today?", "any 401s?", "any streaming failures recently?". The
  pattern is a .NET regex — alternation works ('error|fail|warning').
- plex_laptop_health — CPU/RAM/disk/network on the Plex laptop. Use for
  "how is the Plex box doing?", "is the Plex laptop drowning?", "what's the
  disk space on Plex?".
- For "is THIS PC vs the Plex laptop", remember pc_diagnostics is for THIS
  PC and plex_laptop_health is for the remote one.
- Voice summaries: when reading log lines aloud, paraphrase the gist
  ("a couple of transcoder warnings around 8 PM, otherwise clean") rather
  than reading raw timestamps and stack traces.

Actions (destructive, all confirmation-gated):
- plex_action — restart Plex Media Server, refresh a library, or empty
  Plex's trash for a library. ALL THREE require explicit user confirmation.
- For ANY plex_action call: ask the user to confirm in plain language
  first ("Confirm: restart Plex on the laptop?" / "Confirm: refresh the
  Movies library?"), wait for an explicit yes, THEN call plex_action with
  confirmed=true. The tool itself enforces this — if you call without
  confirmed=true it returns a confirmation-required notice rather than
  firing. Same pattern as kill_process on the local PC.
- For refresh_library and empty_trash you need a library_id. If you don't
  know it, call library_list (Plex MCP) first to enumerate sections.
- If the user asks for an action we don't support yet (rotate logs,
  restart the laptop itself, clear OS-level caches), say so plainly —
  diagnose first, propose remediations in words, but don't pretend to
  have a tool you don't."""


# M48.2a — appended only for a RESTRICTED (phone/web remote) session. The
# server-side tool filter is the real boundary (a phone turn never even
# receives the blocked tools); this addendum just keeps Claude from
# *promising* a tool it no longer has, so it degrades gracefully instead
# of "I'll run a diagnostic…" then silently failing. Defense in depth, not
# the enforcement itself ([[feedback-diag-vs-action-split]]).
_REMOTE_RESTRICTED_ADDENDUM = """

Remote session (limited capability):
- You are answering from the user's phone / web console, not the PC.
- For safety this session CANNOT run system control, the investigative
  shell, the deep diagnostics collector, Plex admin actions, local file
  reads, or screen/camera capture — those tools are simply not available
  here (this is enforced, not advisory).
- If asked for one, say plainly it's not available from the remote console
  and offer what you CAN do: answer and look things up, check live
  read-only PC/Plex health, search the user's knowledge base, and they can
  arm/disarm security with the console's own buttons. Don't pretend, don't
  apologize at length — just redirect."""


# M71 — Discord variant of the restricted addendum. Discord is the ONE remote
# origin with camera_snapshot clawed back (see _RESTRICTED_ALLOW_BY_ORIGIN), so
# its prompt must NOT tell Claude the camera is off-limits — it tells Claude the
# camera IS available and the photo is shared into the channel. Everything else
# stays denied, identical to the base addendum.
_REMOTE_RESTRICTED_DISCORD_ADDENDUM = """

Remote session (Discord — limited capability):
- You are answering from a private Discord channel, not at the PC.
- You CAN look through the webcam: call camera_snapshot when the user asks you
  to check on the home, see what's going on, or look at something. The photo
  you capture is posted into this Discord channel alongside your reply, so
  describe what you see plainly and usefully (who or what is present, lights,
  anything notable).
- For safety this session still CANNOT run system control, the investigative
  shell, the deep diagnostics collector, Plex admin actions, local file reads,
  screen capture, code execution, or self-update — those tools are simply not
  available here (enforced, not advisory).
- If asked for one of those, say plainly it's not available from Discord and
  offer what you CAN do. Don't pretend, don't apologize at length."""


# Anthropic's server-side web search tool. Server-side = Anthropic runs it,
# we just declare it.
#
# We deliberately use _20250305 (the GA version) rather than _20260209 — the
# SAME decision, for the SAME reason, as WEB_FETCH_TOOL below. The newer
# _20260209 adds "dynamic filtering", which runs Anthropic's code-execution
# sandbox to filter large result sets; that container threads a `container_id`
# through the rest of the turn, and our agentic loop's container handling in
# stream_response() proved unreliable when a search is interleaved with a
# client-side tool call (e.g. set_reminder). Live failure 2026-06-02: a
# multi-search "deep dive" query that also set reminders 400'd on the
# re-stream with "container_id is required when there are pending tool uses
# generated by code execution with tools." Downgrading removes the
# code-execution container entirely — the error class disappears. Dynamic
# filtering's token-saving benefit is marginal for voice queries; the
# container-capture code in stream_response() stays as defense-in-depth.
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
}


# Anthropic's server-side web fetch tool. Pulls the full contents of a
# specific URL (HTML or PDF) and returns the parsed text. Pairs naturally
# with web_search (search to find a URL → fetch to read it in depth).
#
# We deliberately use _20250910 (the older version) rather than _20260209.
# The newer version's "dynamic filtering" feature runs Anthropic's code
# execution sandbox to filter big pages, which threads a `container_id`
# through the conversation — and our agentic loop in stream_response() does
# not track or forward that container_id. Mismatch surfaced in M19 testing
# as: "container_id is required when there are pending tool uses generated
# by code execution with tools." For voice queries (typical pages ≤ a few
# KB), dynamic filtering's token-saving benefit is small; for the rare
# multi-MB doc, the agentic loop's MAX_LOOP_ITERATIONS still bounds cost.
# Revisit if we ever add a "summarize this 200-page document" workflow.
WEB_FETCH_TOOL = {
    "type": "web_fetch_20250910",
    "name": "web_fetch",
}


# Cap on agentic-loop iterations. Most voice queries finish in 1-2 tool calls,
# but research-style asks ("do a deep dive on X, then set a reminder") legit-
# imately chain more — search → search → web_fetch → set_reminder → cancel →
# re-set. 5 was too tight (a 2026-06-02 deep-dive query exhausted it with no
# final answer); 8 gives that headroom while still bounding a runaway turn.
# If we hit it we log AND yield a graceful fallback (see the while-else) so the
# user never gets dead silence — the real failure mode the old cap exposed.
_MAX_LOOP_ITERATIONS = 8

# Spoken when the agentic loop is cut off by the iteration cap mid-work, so a
# capped turn degrades to a sentence instead of silence.
_LOOP_CAP_FALLBACK = (
    " I'm sorry, sir — that turned into more steps than I can take in one go. "
    "Here's where I got to; ask me to continue and I'll pick it up."
)

# Token budgets. Default mode is voice-shaped — short replies, no thinking.
# Engineer mode unlocks extended thinking + a generous output budget so the
# model can both reason and write the longer structured answer it produced.
_DEFAULT_MAX_TOKENS = 1024
_ENGINEER_THINKING_BUDGET = 5000
_ENGINEER_MAX_TOKENS = 8192   # must exceed thinking_budget; ~3k left for the actual reply


@dataclass
class TelemetryRecord:
    """Per-turn structured telemetry. Same data the existing stderr log line
    carries, exposed as a callable contract so the UI can surface it.

    Latency is wall-clock LLM-and-tools time only — TTS playback is excluded.
    For an SRE skimming the console, "how long did Jarvis spend thinking" is
    the more useful number than "how long did the whole turn take".
    """
    elapsed_sec: float
    iterations: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_create_tokens: int
    tools_used: list[str] = field(default_factory=list)
    paused: bool = False
    thinking_enabled: bool = False  # M26: extended thinking was active for this turn

    @property
    def total_tokens(self) -> int:
        # cache_read_tokens are billed at a discount but still count as
        # "input" semantically; include for the SRE-grade total.
        return self.input_tokens + self.output_tokens + self.cache_read_tokens


def _format_today() -> str:
    """Cross-platform 'Sunday, May 4, 2026' (no leading zero on day)."""
    now = datetime.now()
    return now.strftime("%A, %B ") + str(now.day) + now.strftime(", %Y")


def build_system_prompt(
    summaries: list[SummaryRecord] | None = None,
    plex_available: bool = False,
    plex_laptop_available: bool = False,
    engineer_mode: bool = False,
    remote_restricted: bool = False,
    restricted_origin: str = "",
) -> str:
    """Compose the system prompt with the current date and optional memory.

    The current date gives Claude a temporal anchor for reasoning about what's
    stale vs. current. We use date precision (not time) so the cache breakpoint
    invalidates at most once per day, not per turn.

    Optional addenda are gated on actual tool availability so a graceful-fail
    startup doesn't leave Claude believing in tools it can't call:
    - `plex_available` (M21): advertise the Plex MCP media tools
    - `plex_laptop_available` (M24): advertise the remote-laptop SSH tools
    - `engineer_mode` (M26): unlock structured-depth replies and connect to
      the user's technical expertise. Toggleable per-turn from the tray.

    Note: engineer addendum is placed BEFORE the memory block so it stays in
    the cacheable prefix. Memory varies per turn; tool addenda + engineer
    addendum stay stable across turns within a session, which matters for
    prompt-cache hit rate.
    """
    base = f"{JARVIS_SYSTEM_PROMPT}\n\nToday is {_format_today()}."
    if plex_available:
        base += _PLEX_PROMPT_ADDENDUM
    if plex_laptop_available:
        base += _PLEX_LAPTOP_PROMPT_ADDENDUM
    if engineer_mode:
        base += _ENGINEER_PROMPT_ADDENDUM
    if remote_restricted:
        # Stable per session-origin (a turn is restricted or not for its
        # whole life), so it stays in the cacheable prefix like the other
        # addenda — no prompt-cache churn. M71: an origin with camera clawed
        # back (Discord) gets the variant that tells Claude the webcam IS
        # available, instead of the base "no camera" wording.
        if "camera_snapshot" in _RESTRICTED_ALLOW_BY_ORIGIN.get(
            restricted_origin, frozenset()
        ):
            base += _REMOTE_RESTRICTED_DISCORD_ADDENDUM
        else:
            base += _REMOTE_RESTRICTED_ADDENDUM
    if not summaries:
        return base
    memory_block = format_summaries_for_prompt(summaries)
    return (
        base
        + "\n\nRecent conversations (for your memory — only mention these if relevant):\n"
        + memory_block
    )


class _ClientTool(NamedTuple):
    """A built-in client-side tool Jarvis runs locally (vs. the server-side
    web_search / web_fetch that Anthropic executes)."""
    # Runs the tool. Returns text for Claude, OR a list of content blocks
    # (e.g. camera_snapshot returns an image block + caption). Never raises.
    execute: Callable[[dict], "str | list[dict]"]
    log_label: str                   # short marker for the stderr telemetry line ("sysctl", "camera", ...)


# Single source of truth for the built-in client-side tools: API name →
# (executor, telemetry label). Drives _execute_client_tool's dispatch AND the
# per-turn telemetry — the set of fired names plus this map produce both the
# stderr log markers and the console chip. Adding a tool means: import its
# executor + schema, add one entry here, and add the schema to the `tools`
# list in stream_response. Insertion order is preserved and defines telemetry
# order, so keep it sensible (the order the SRE wants to read it in).
_CLIENT_TOOLS: dict[str, _ClientTool] = {
    "get_sports_info":              _ClientTool(execute_sports_tool, "sports_tool"),
    "get_weather":                  _ClientTool(execute_weather_tool, "weather_tool"),
    "get_game_info":                _ClientTool(execute_games_tool, "games_tool"),
    "get_movie_tv_info":            _ClientTool(execute_tmdb_tool, "tmdb_tool"),
    "get_person_info":              _ClientTool(execute_person_info_tool, "tmdb_person"),
    "get_news":                     _ClientTool(execute_news_tool, "news"),
    "wolfram_query":                _ClientTool(execute_wolfram_tool, "wolfram"),
    "run_code":                     _ClientTool(execute_code_tool, "code"),
    "knowledge_search":             _ClientTool(execute_knowledge_tool, "knowledge"),
    "knowledge_remember":           _ClientTool(execute_knowledge_remember, "knowledge_write"),
    "get_calendar_events":          _ClientTool(execute_calendar_tool, "calendar"),
    "set_reminder":                 _ClientTool(execute_set_reminder, "reminder_set"),
    "list_reminders":               _ClientTool(execute_list_reminders, "reminder_list"),
    "cancel_reminder":              _ClientTool(execute_cancel_reminder, "reminder_cancel"),
    "get_briefing":                 _ClientTool(execute_briefing_tool, "briefing"),
    "get_good_night":               _ClientTool(execute_good_night_tool, "good_night"),
    "update_jarvis":                _ClientTool(execute_update_jarvis, "self_update"),
    "homelab_status":               _ClientTool(execute_homelab_status, "homelab"),
    "status_report":                _ClientTool(execute_status_report, "self_status"),
    "pc_diagnostics":               _ClientTool(execute_pc_diagnostics_tool, "diagnostics"),
    "pc_shell":                     _ClientTool(execute_pc_shell, "shell"),
    "system_control":               _ClientTool(execute_system_control_tool, "sysctl"),
    "read_local_file":              _ClientTool(execute_read_local_file, "read_file"),
    "run_pc_diagnostics_collector": _ClientTool(execute_run_pc_diagnostics_collector, "diag_collector"),
    "camera_snapshot":              _ClientTool(execute_camera_snapshot, "camera"),
    "screen_snapshot":              _ClientTool(execute_screen_snapshot, "screen"),
}

# M48.2a — the per-session capability boundary. A RESTRICTED session
# (phone/web remote, keyed off the turn `origin` in main.py) gets every
# tool EXCEPT these. Two classes, both per the locked spec:
#   BLOCK (mutating / dangerous): can change the PC or Plex.
#   EXCLUDE (read-only but sensitive on a loseable phone): can read PC
#     files or open the screen/webcam remotely.
# Single source of truth — applied at BOTH gates (the tool-list filter in
# stream_response AND the _execute_client_tool dispatch), so it is
# enforced server-side, never prompt-only ([[feedback-diag-vs-action-split]],
# [[feedback-jarvis-least-privilege]]). Match is by tool NAME, so it also
# covers plex_action (in _PLEX_LAPTOP_DISPATCH) and any future name.
_RESTRICTED_DENY: frozenset[str] = frozenset({
    # BLOCK — mutating / dangerous
    "system_control", "pc_shell", "run_pc_diagnostics_collector", "plex_action",
    # BLOCK — arbitrary code execution (M50). Sandboxed on the PC, but a
    # phone-origin turn must never be able to make Jarvis run code at all —
    # the principle holds regardless of the isolation.
    "run_code",
    # BLOCK — self-update (M64). The phone PWA must never be able to pull
    # code into the user's machine + force a restart; that's the M48.2a
    # least-privilege boundary at the most extreme (the update could be
    # anything — a phone-driven self-update IS arbitrary code execution
    # by another name).
    "update_jarvis",
    # EXCLUDE — read-only but sensitive from a remote
    "read_local_file", "screen_snapshot", "camera_snapshot",
})


# M71 — per-origin claw-back: specific otherwise-denied tools selectively
# re-allowed for ONE restricted origin, deny-by-default everywhere else.
# Discord gets camera_snapshot back so the household can ask Jarvis to look
# through the webcam while away (the captured photo is relayed into the
# channel; see main.py's reply_image sink). Least-privilege is preserved —
# this is a SURGICAL exception, not a boundary flip: every other restricted
# tool (system/shell/file/screen/code/self-update) stays denied for Discord,
# and the phone origins get nothing back. New exceptions go here, one origin
# at a time, deliberately. [[feedback-jarvis-least-privilege]]
_RESTRICTED_ALLOW_BY_ORIGIN: dict[str, frozenset[str]] = {
    "discord": frozenset({"camera_snapshot"}),
}


def _effective_deny(origin: str) -> frozenset[str]:
    """The deny-set actually applied to a restricted `origin`: the base
    _RESTRICTED_DENY minus any tools clawed back for that origin. An unknown
    or empty origin gets the full deny (safe default)."""
    return _RESTRICTED_DENY - _RESTRICTED_ALLOW_BY_ORIGIN.get(origin, frozenset())


# Server-side tools — Anthropic runs these; we only declare them (in
# stream_response's `tools` list) and record that they fired. Ordered:
# defines telemetry order. Each name doubles as its own stderr label.
_SERVER_TOOLS: tuple[str, ...] = ("web_search", "web_fetch")

# SSH-backed tools (M24 read-only + M27 actions). Their executors take the
# PlexLaptopClient as a first arg, so they're dispatched separately from
# _CLIENT_TOOLS; telemetry groups them under one `plex_laptop_tools=` marker
# since they're a family. (Plex MCP tools, M21, are dynamic — names come from
# the server at runtime — so they're only known via plex_client.tool_names.)
_PLEX_LAPTOP_DISPATCH = {
    "plex_logs_tail": execute_plex_logs_tail,
    "plex_logs_search": execute_plex_logs_search,
    "plex_laptop_health": execute_plex_laptop_health,
    "plex_action": execute_plex_action,
}


def _execute_client_tool(
    name: str,
    tool_input: dict,
    plex_client: PlexMCPClient | None = None,
    plex_laptop_client: PlexLaptopClient | None = None,
    restricted: bool = False,
    origin: str = "",
) -> str | list[dict]:
    """Dispatch a client-side tool call. Returns text (or, for camera_snapshot,
    a list of content blocks) for Claude to consume. Errors become readable
    strings — never raises.

    `restricted` (M48.2a): second gate of the capability boundary. The
    tool-list filter in stream_response already prevents a restricted
    session from being OFFERED a denied tool; this refuses to EXECUTE one
    even if a denied name reaches here anyway (model quirk, prompt
    injection from the phone, a future code path). Server-side, not
    prompt-only — the [[feedback-diag-vs-action-split]] rule. M71: the deny-set
    is per-origin (Discord has camera clawed back), so an origin's allowed
    exception executes while everything else still refuses."""
    if restricted and name in _effective_deny(origin):
        print(f"[llm] restricted session ({origin or 'remote'}) denied tool '{name}'",
              file=sys.stderr)
        return (
            f"The '{name}' tool isn't available from the remote console "
            f"(safety: this session can't run system, shell, file, or "
            f"screen/camera tools)."
        )
    try:
        client_tool = _CLIENT_TOOLS.get(name)
        if client_tool is not None:
            return client_tool.execute(tool_input)
        if plex_laptop_client is not None and name in _PLEX_LAPTOP_DISPATCH:
            return _PLEX_LAPTOP_DISPATCH[name](plex_laptop_client, tool_input)
        if plex_client is not None and name in plex_client.tool_names:
            return plex_client.call_tool(name, tool_input)
        return f"Unknown tool: {name}"
    except Exception as exc:
        # Defensive — existing tool executors swallow their own errors,
        # but a future tool might not. Don't let a tool exception kill the turn.
        print(f"[llm] tool '{name}' raised: {exc}", file=sys.stderr)
        return f"Tool error: {exc}"


def stream_response(
    api_key: str,
    messages: list[dict],
    model: str = "claude-sonnet-4-6",
    summaries: list[SummaryRecord] | None = None,
    plex_client: PlexMCPClient | None = None,
    plex_laptop_client: PlexLaptopClient | None = None,
    on_complete: Callable[[TelemetryRecord], None] | None = None,
    on_image_captured: Callable[[bytes, str, str], None] | None = None,
    engineer_mode: bool = False,
    restricted: bool = False,
    origin: str = "",
    interrupt_event: threading.Event | None = None,
) -> Iterator[str]:
    """Stream Claude's response, handling client-side tool use transparently.

    Yields text chunks suitable for direct TTS feeding. The caller sees a single
    continuous stream of text even when one or more tool calls happen in the
    middle — each tool round-trip is invisible from the caller's perspective.

    `messages` must be the full alternating history ending with a user message.
    `summaries` (optional) prepends recent-session context to the system prompt.
    `plex_client` (optional, M21) — if a live Plex MCP session is provided,
    its tools are surfaced to Claude alongside the built-in tools.
    `plex_laptop_client` (optional, M24) — if SSH-reachable, the three remote
    diagnostic tools (logs_tail, logs_search, laptop_health) get registered.
    `on_complete` (optional) — called once at the end with a TelemetryRecord
    for UI consumption. The same data is also stderr-logged in the existing
    one-line format. Wrapped in try/except so a UI bug can't poison the turn.
    `on_image_captured` (optional) — fired whenever a client-side tool returns
    a tool_result containing an image content block (currently camera_snapshot
    + screen_snapshot). Receives (image_bytes, media_type, tool_name) so the
    UI can render an inline thumbnail of what Jarvis just "saw". Wrapped in
    try/except — a UI bug must not break the agentic loop mid-turn.
    `engineer_mode` (optional, M26) — when True, append the engineer-mode
    addendum to the system prompt and enable Anthropic's extended thinking
    feature with a 5k-token reasoning budget. Captured per-turn from
    `ui.is_engineer_mode()`; mid-turn toggles apply to the next turn.
    `interrupt_event` (optional, M52 barge-in) — polled while streaming. When
    set, the generator returns immediately, which exits the `with
    client.messages.stream()` block and closes the HTTP stream — otherwise
    Claude would keep generating server-side until the generator is GC'd.
    Two poll points: mid-text-stream (the common case — the user cuts in
    while a reply is being spoken) and at the top of each agentic-loop
    iteration (so a barge-in during a tool call aborts before another LLM
    round-trip). The mid-stream return skips the final on_complete telemetry
    callback — an accepted, minor cost on an interrupted turn.
    """
    started_at = time.monotonic()
    client = anthropic.Anthropic(api_key=api_key)
    system_text = build_system_prompt(
        summaries,
        plex_available=plex_client is not None,
        plex_laptop_available=plex_laptop_client is not None,
        engineer_mode=engineer_mode,
        remote_restricted=restricted,
        restricted_origin=origin,
    )
    system_param = [
        {
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    tools = [
        WEB_SEARCH_TOOL, WEB_FETCH_TOOL,
        SPORTS_TOOL, WEATHER_TOOL, GAMES_TOOL, TMDB_TOOL, GET_PERSON_INFO_TOOL,
        NEWS_TOOL, WOLFRAM_TOOL, CODE_EXEC_TOOL,
        KNOWLEDGE_SEARCH_TOOL, KNOWLEDGE_REMEMBER_TOOL, GET_CALENDAR_TOOL,
        SET_REMINDER_TOOL, LIST_REMINDERS_TOOL, CANCEL_REMINDER_TOOL,
        BRIEFING_TOOL, GOOD_NIGHT_TOOL, HOMELAB_STATUS_TOOL, STATUS_REPORT_TOOL,
        UPDATE_JARVIS_TOOL,
        PC_DIAGNOSTICS_TOOL, PC_SHELL_TOOL, SYSTEM_CONTROL_TOOL,
        READ_LOCAL_FILE_TOOL, RUN_PC_DIAGNOSTICS_COLLECTOR_TOOL,
        CAMERA_SNAPSHOT_TOOL, SCREEN_SNAPSHOT_TOOL,
    ]
    if plex_laptop_client is not None:
        tools.extend([
            PLEX_LOGS_TAIL_TOOL,
            PLEX_LOGS_SEARCH_TOOL,
            PLEX_LAPTOP_HEALTH_TOOL,
            PLEX_ACTION_TOOL,
        ])
    if plex_client is not None:
        tools.extend(plex_client.tools)

    # M48.2a — the capability boundary's PRIMARY gate: a restricted
    # (phone/web remote) turn is never even OFFERED a denied tool. Filter
    # by schema name AFTER all extends so it covers built-ins, the SSH
    # family (plex_action), and dynamic Plex MCP names uniformly. The
    # _execute_client_tool deny-check is the defense-in-depth second gate.
    if restricted:
        denied = _effective_deny(origin)
        tools = [t for t in tools if t.get("name") not in denied]

    # Mutable working copy — we append assistant + tool_result turns as the
    # agentic loop runs. The caller's `messages` list is left untouched.
    working = list(messages)

    # Telemetry accumulators across all agentic-loop iterations. We track the
    # set of fired tool names per category; _CLIENT_TOOLS / _SERVER_TOOLS map
    # those to stderr labels + the console chip below. (The categories are
    # real — server-side and SSH-backed/MCP tools are formatted differently
    # in the log: individual `x=yes` markers vs. one grouped `x_tools=a,b`.)
    fired_server: set[str] = set()        # subset of _SERVER_TOOLS
    fired_client: set[str] = set()        # subset of _CLIENT_TOOLS keys
    fired_plex_laptop: set[str] = set()   # subset of _PLEX_LAPTOP_DISPATCH keys
    fired_plex_mcp: set[str] = set()      # dynamic names from the Plex MCP server
    paused = False
    total_input = total_output = total_cache_read = total_cache_create = 0
    iterations = 0

    # Build per-turn stream kwargs once; reuse across agentic-loop iterations.
    # max_tokens MUST exceed thinking budget when thinking is on; 8192 leaves
    # room for both the reasoning + a generous structured reply. Anthropic
    # also requires temperature=1 with extended thinking (we don't set it,
    # so default of 1 holds). Thinking blocks are preserved in the assistant
    # turn we append below — required when thinking + tool_use are combined.
    stream_kwargs: dict = {
        "model": model,
        "max_tokens": _ENGINEER_MAX_TOKENS if engineer_mode else _DEFAULT_MAX_TOKENS,
        "system": system_param,
        "messages": working,
        "tools": tools,
    }
    if engineer_mode:
        stream_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": _ENGINEER_THINKING_BUDGET,
        }

    while iterations < _MAX_LOOP_ITERATIONS:
        iterations += 1

        # M52: barged in during a tool call — abort before the next LLM
        # round-trip. `break` (vs. the mid-stream `return` below) falls
        # through to the telemetry block, so an interrupt at this boundary
        # still records a chip.
        if interrupt_event is not None and interrupt_event.is_set():
            break

        # Refresh messages each iteration since the agentic loop appends to
        # `working`. Other kwargs are stable across iterations.
        stream_kwargs["messages"] = working
        with client.messages.stream(**stream_kwargs) as stream:
            for text in stream.text_stream:
                # M52: the user barged in mid-reply. Returning here exits the
                # `with` block, which closes the HTTP stream so Claude stops
                # generating server-side.
                if interrupt_event is not None and interrupt_event.is_set():
                    return
                yield text
            final = stream.get_final_message()

        # If this turn used the server-side code-execution container (newer
        # Claude can run code as part of agentic work even though we don't
        # declare that tool), the API requires its id to be echoed on every
        # subsequent request in the same exchange. Without it, the re-stream
        # after a client tool_use 400s: "container_id is required when there
        # are pending tool uses generated by code execution". Capture it and
        # thread it into the rest of THIS call's agentic loop. Conditional —
        # untouched on the common path, so zero change when no container is
        # used. (stream_kwargs is per-call; it never leaks across turns.)
        container = getattr(final, "container", None)
        if container is not None:
            stream_kwargs["container"] = container.id

        # Accumulate token usage. cache_* fields are None on uncached turns.
        u = final.usage
        total_input += u.input_tokens
        total_output += u.output_tokens
        total_cache_read += u.cache_read_input_tokens or 0
        total_cache_create += u.cache_creation_input_tokens or 0

        # Track server-side tool use (web_search, web_fetch) for telemetry
        # only — the SDK has already executed them and the text stream above
        # included Claude's post-tool text.
        for block in final.content:
            if getattr(block, "type", None) == "server_tool_use":
                sname = getattr(block, "name", None)
                if sname in _SERVER_TOOLS:
                    fired_server.add(sname)

        # Server-side loop hit its 10-iteration cap. Rare in voice; we log
        # and bail rather than try to manually resume.
        if final.stop_reason == "pause_turn":
            paused = True
            break

        # Normal end of turn — Claude is done responding.
        if final.stop_reason != "tool_use":
            break

        # Client-side tool use. Append the assistant turn (full content,
        # including any text + tool_use blocks Claude emitted) so the SDK
        # has the canonical record. Then run each tool and feed results back.
        working.append({"role": "assistant", "content": final.content})

        tool_results = []
        for block in final.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = block.name
            if name in _CLIENT_TOOLS:
                fired_client.add(name)
            elif name in _PLEX_LAPTOP_DISPATCH:
                fired_plex_laptop.add(name)
            elif plex_client is not None and name in plex_client.tool_names:
                fired_plex_mcp.add(name)
            result_text = _execute_client_tool(
                name,
                block.input or {},
                plex_client=plex_client,
                plex_laptop_client=plex_laptop_client,
                restricted=restricted,
                origin=origin,
            )
            # Vision tools (camera_snapshot, screen_snapshot) return a list
            # containing an image block + a caption text block. If a UI hook
            # is registered, surface the raw bytes so the console can render
            # an inline thumbnail of what Jarvis just saw. Detecting on the
            # generic shape (list with an image block) keeps this hook
            # tool-agnostic — any future tool that returns an image gets
            # thumbnailed for free.
            if on_image_captured is not None and isinstance(result_text, list):
                for sub in result_text:
                    if isinstance(sub, dict) and sub.get("type") == "image":
                        src = sub.get("source", {})
                        if src.get("type") == "base64" and src.get("data"):
                            try:
                                img_bytes = base64.b64decode(src["data"])
                                on_image_captured(
                                    img_bytes,
                                    src.get("media_type", "image/png"),
                                    name,
                                )
                            except Exception as exc:
                                print(
                                    f"[llm] on_image_captured failed for {name}: {exc}",
                                    file=sys.stderr,
                                )
                        break
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })

        working.append({"role": "user", "content": tool_results})
        # Loop continues — next stream iteration sees the tool result and
        # produces Claude's final answer (or another tool call).
    else:
        # while-else: ran out of iterations without hitting break — i.e. the
        # last iteration still wanted another tool call. Whatever streamed so
        # far was interstitial ("let me check…"), with no final answer, so
        # without this the caller gets dead silence. Yield a short spoken
        # fallback so a capped turn degrades gracefully. Log it too — hitting
        # the cap is still a signal worth noticing in real use.
        print(
            f"[llm] hit MAX_LOOP_ITERATIONS={_MAX_LOOP_ITERATIONS} agentic cap",
            file=sys.stderr,
        )
        yield _LOOP_CAP_FALLBACK

    # Compact tool/run markers for the stderr line. Order: server tools, then
    # built-in client tools (in _CLIENT_TOOLS order), then the grouped MCP /
    # SSH families, then run-shape markers.
    markers: list[str] = [f"{n}=yes" for n in _SERVER_TOOLS if n in fired_server]
    markers += [f"{ct.log_label}=yes" for n, ct in _CLIENT_TOOLS.items() if n in fired_client]
    if fired_plex_mcp:
        markers.append(f"mcp_tools={','.join(sorted(fired_plex_mcp))}")
    if fired_plex_laptop:
        markers.append(f"plex_laptop_tools={','.join(sorted(fired_plex_laptop))}")
    if paused:
        markers.append("PAUSED_TURN(10-iter cap)")
    if iterations > 1:
        markers.append(f"iters={iterations}")
    if engineer_mode:
        markers.append("thinking=on")
    extra = (" " + " ".join(markers)) if markers else ""

    elapsed = time.monotonic() - started_at
    print(
        f"[llm] tokens: input={total_input} output={total_output} "
        f"cache_read={total_cache_read} cache_create={total_cache_create} "
        f"history_msgs={len(messages)} summaries={len(summaries) if summaries else 0} "
        f"elapsed={elapsed:.1f}s"
        f"{extra}",
        file=sys.stderr,
    )

    if on_complete is not None:
        # Flat ordered list of API tool names that fired this turn (the "verb"
        # the console chip leads with). Order: server tools, built-in client
        # tools (in _CLIENT_TOOLS order), then the SSH and MCP families.
        tools_used = (
            [n for n in _SERVER_TOOLS if n in fired_server]
            + [n for n in _CLIENT_TOOLS if n in fired_client]
            + sorted(fired_plex_laptop)
            + sorted(fired_plex_mcp)
        )

        record = TelemetryRecord(
            elapsed_sec=elapsed,
            iterations=iterations,
            input_tokens=total_input,
            output_tokens=total_output,
            cache_read_tokens=total_cache_read,
            cache_create_tokens=total_cache_create,
            tools_used=tools_used,
            paused=paused,
            thinking_enabled=engineer_mode,
        )
        try:
            on_complete(record)
        except Exception as exc:
            # A UI bug must never break the listen loop.
            print(f"[llm] on_complete callback raised: {exc}", file=sys.stderr)
