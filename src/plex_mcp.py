"""Plex MCP client (M21).

Spawns plex-mcp-server as a stdio subprocess and exposes its tools to the
Anthropic agentic loop.

Why this module exists, in three observations:

1. The MCP Python SDK is async-only. Our agentic loop in src/llm.py is sync
   (anthropic's `client.messages.stream(...)` is a sync context manager and
   stream_response is a generator). Wrapping the entire loop in asyncio is a
   much bigger refactor than wrapping just MCP — so we wrap MCP. This module
   is the **async↔sync bridge**: a dedicated event loop runs on a daemon
   thread, and the sync façade dispatches coroutines onto it via
   `asyncio.run_coroutine_threadsafe` and blocks on `.result()`.

2. plex-mcp-server exposes 51 tools. Many are admin (server logs, scans) or
   destructive (metadata edits, deletes). Both are bad fits for voice. We
   filter through an **allowlist** that errs permissive-include + strict-
   exclude — a destructive verb in the name (delete/remove/edit/scan/refresh)
   is auto-excluded so a misheard voice command can't nuke a library.

3. The Plex token is sensitive. Passing it via subprocess env vars (vs CLI
   args) keeps it out of `ps`/Task Manager listings. Same hygiene reason we
   pass API keys via env in scripts.

Lifecycle: PlexMCPClient(url, token) → connects synchronously (raises on
failure within _CONNECT_TIMEOUT). Caller reads .tools (Anthropic schemas) and
.tool_names (set). call_tool(name, args) blocks. close() unwinds cleanly.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Subprocess spawn + Plex handshake. First-time start of plex-mcp-server can
# be a couple seconds while plexapi negotiates with the server; 15s is
# generous safety. Lower → false-negative graceful-fail; higher → user waits.
_CONNECT_TIMEOUT = 15.0

# Plex API can be slow on big libraries (esp. recently-added on a 50k-item
# catalog). Most calls complete in <2s; 30s is the ceiling we'll wait before
# returning a "tool error" string back to Claude.
_TOOL_CALL_TIMEOUT = 30.0

# Voice budget: a tool result that's longer than this gets truncated. Claude
# summarizes for TTS anyway — there's no benefit to sending 20KB of metadata.
_RESULT_MAX_CHARS = 2000


# Voice-friendly allowlist filter. Strategy:
#   include = the tool name contains at least one INCLUDE keyword
#   exclude = the tool name contains any EXCLUDE keyword
# Exclude wins on conflict. After first run we log INCLUDED + EXCLUDED so we
# can tune empirically (see _populate_tools).
_INCLUDE_KEYWORDS = (
    "search", "list", "get", "now", "current", "recent", "playing",
    "history", "stats", "session", "info", "library", "libraries",
    "play_media", "pause", "stop", "client",
)

# Anything destructive, admin-y, write-y, or just a poor voice fit. We'd
# rather miss a feature than accept a voice command that nukes a library or
# bury Claude in artwork blobs and admin telemetry it can't usefully speak.
#
# After M21 first-run on a real Plex server, raw discovery surfaced 55 tools.
# 35 passed an early naive filter; we tightened to ~17 by adding:
#   - whole categories that don't make voice sense: server_*, playlist_*,
#     collection_* (admin info / visual organization, awkward to speak)
#   - artwork retrieval (binary blobs by URL — TTS can't render them)
#   - niche client surfaces (navigate / timelines / set_streams) that are
#     UI-shaped, not voice-shaped
#   - mutation verbs the destructive list missed (add_to, copy_to, upload)
#   - admin user-mgmt (all_users, search_users)
_EXCLUDE_KEYWORDS = (
    # destructive / write
    "delete", "remove", "edit", "scan", "refresh", "create",
    "update", "share", "merge", "ban", "kick", "log", "backup",
    "restart", "shutdown", "trash", "empty", "add_to", "copy_to",
    "upload", "set_streams",
    # whole categories — admin / visual-organization / poor voice fit
    "server_", "playlist_", "collection_",
    # artwork (binary data, can't read aloud)
    "artwork", "poster",
    # niche client controls
    "navigate", "timelines",
    # admin user mgmt
    "all_users", "search_users",
)


def _is_voice_friendly(tool_name: str) -> bool:
    name = tool_name.lower()
    if any(k in name for k in _EXCLUDE_KEYWORDS):
        return False
    return any(k in name for k in _INCLUDE_KEYWORDS)


def _format_tool_result(result: Any) -> str:
    """MCP CallToolResult → string for Claude to consume.

    MCP results are a list of content blocks (text / image / embedded). For
    voice we only want text — concat the text blocks, strip, truncate at
    _RESULT_MAX_CHARS. Errors come through with isError=True; surface them
    as readable strings so Claude can apologize gracefully rather than
    crashing the turn."""
    if result is None:
        return ""
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    full = "\n".join(parts).strip()
    if getattr(result, "isError", False):
        return f"Tool error: {full or 'unknown error'}"
    if len(full) > _RESULT_MAX_CHARS:
        full = full[:_RESULT_MAX_CHARS] + "… [truncated]"
    return full or "(empty result)"


def _translate_tool(mcp_tool: Any) -> dict:
    """MCP Tool → Anthropic tool schema. The input schema is already JSON
    Schema in both formats; only the field name differs (inputSchema vs
    input_schema). Trivial repack."""
    return {
        "name": mcp_tool.name,
        "description": (mcp_tool.description or "").strip()
        or f"Plex tool: {mcp_tool.name}",
        "input_schema": mcp_tool.inputSchema or {"type": "object", "properties": {}},
    }


class PlexMCPClient:
    """Synchronous façade over an async MCP stdio session.

    Construction blocks until the server reports ready (or _CONNECT_TIMEOUT).
    Caller is expected to wrap construction in try/except and degrade
    gracefully if Plex is unreachable — that's the contract main.py relies on.
    """

    def __init__(
        self,
        plex_url: str,
        plex_token: str,
        python_executable: str | None = None,
    ) -> None:
        if not plex_url or not plex_token:
            raise ValueError("plex_url and plex_token are required")

        self._plex_url = plex_url
        self._plex_token = plex_token
        self._python = python_executable or sys.executable

        # Cross-thread signals.
        self._connected = threading.Event()  # set when connect succeeds OR fails
        self._connect_error: Exception | None = None
        self._closed = False

        # Created on the loop side; used as the "stay alive" gate for _serve.
        self._stop_event: asyncio.Event | None = None
        self._session: ClientSession | None = None

        # Public, populated after successful connect.
        self.tools: list[dict] = []
        self.tool_names: set[str] = set()

        # Spin up dedicated event loop on a daemon thread. We don't reuse
        # the main event loop because (a) Tk owns the main thread and (b)
        # the agentic loop doesn't run inside an event loop at all.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="plex-mcp-loop", daemon=True
        )
        self._thread.start()

        # Submit the long-running connect coroutine. It will set
        # self._connected when it has either succeeded (session ready,
        # tools populated) or failed (self._connect_error set).
        self._serve_future = asyncio.run_coroutine_threadsafe(
            self._serve(), self._loop
        )

        if not self._connected.wait(timeout=_CONNECT_TIMEOUT):
            # Timed out waiting for connect signal — stop the loop and raise.
            self._serve_future.cancel()
            self._tear_down_loop()
            raise TimeoutError(
                f"Plex MCP connect timed out after {_CONNECT_TIMEOUT}s"
            )

        if self._connect_error is not None:
            self._tear_down_loop()
            raise self._connect_error

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        """Long-running coroutine: holds the stdio subprocess + ClientSession
        open until close() sets the stop event. All async lifetime is scoped
        inside this single coroutine so context-manager cleanup runs on the
        same task that opened them."""
        self._stop_event = asyncio.Event()
        try:
            # Token through env, not args — keeps it out of process listings.
            env = os.environ.copy()
            env["PLEX_URL"] = self._plex_url
            env["PLEX_TOKEN"] = self._plex_token

            params = StdioServerParameters(
                command=self._python,
                args=["-m", "plex_mcp_server", "--transport", "stdio"],
                env=env,
            )

            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_resp = await session.list_tools()
                    self._session = session
                    self._populate_tools(list(tools_resp.tools))
                    self._connected.set()
                    # Block here until close() flips the stop event. While
                    # we're awaiting, MCP continues to dispatch tool-call
                    # tasks on this same loop — see _call_tool_async.
                    await self._stop_event.wait()
        except Exception as exc:
            # Connect or runtime error. Surface to __init__ if we haven't
            # signaled connected yet; otherwise just log (we're closing).
            if not self._connected.is_set():
                self._connect_error = exc
                self._connected.set()
            else:
                print(f"[plex-mcp] session error: {exc}", file=sys.stderr)
        finally:
            self._session = None

    def _populate_tools(self, discovered: list[Any]) -> None:
        included: list[str] = []
        excluded: list[str] = []
        for t in discovered:
            if _is_voice_friendly(t.name):
                self.tools.append(_translate_tool(t))
                self.tool_names.add(t.name)
                included.append(t.name)
            else:
                excluded.append(t.name)
        print(
            f"[plex-mcp] discovered {len(discovered)} tools, "
            f"using {len(included)} after voice filter",
            file=sys.stderr,
        )
        print(f"[plex-mcp] included: {sorted(included)}", file=sys.stderr)
        print(f"[plex-mcp] excluded: {sorted(excluded)}", file=sys.stderr)

    def call_tool(self, name: str, arguments: dict) -> str:
        """Sync façade. Blocks until the tool returns (or timeout). Always
        returns a string — never raises — so the agentic loop can feed the
        result straight into a tool_result block."""
        if self._closed or self._session is None:
            return "Plex tool unavailable (session closed)."
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._call_tool_async(name, arguments), self._loop
            )
            result = future.result(timeout=_TOOL_CALL_TIMEOUT)
        except Exception as exc:
            print(f"[plex-mcp] tool '{name}' failed: {exc}", file=sys.stderr)
            return f"Plex tool error: {exc}"
        return _format_tool_result(result)

    async def _call_tool_async(self, name: str, arguments: dict) -> Any:
        # _session is the one created inside _serve; we read it from another
        # task on the same loop, which is safe.
        assert self._session is not None
        return await self._session.call_tool(name, arguments=arguments)

    def close(self) -> None:
        """Idempotent. Signals _serve to exit, waits briefly for the context
        managers to unwind, then stops the loop."""
        if self._closed:
            return
        self._closed = True
        if self._stop_event is not None:
            try:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            except RuntimeError:
                pass  # loop already stopped
        # Give _serve a chance to unwind cleanly. Bounded — on a stuck
        # subprocess this returns after the timeout and we move on.
        try:
            self._serve_future.result(timeout=5.0)
        except Exception:
            pass
        self._tear_down_loop()

    def _tear_down_loop(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass
        self._thread.join(timeout=5.0)
