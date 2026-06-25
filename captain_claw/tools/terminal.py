"""Interactive terminal tool — drive a live PTY on the Mac.

Unlike ``shell`` (one command in, output out, no tty), this tool talks to
the local PTY daemon (``captain_claw.terminal.daemon``) and holds
long-lived interactive sessions.  It can run REPLs and full-screen TUIs —
``claude``, ``python3``, ``ssh``, ``vim`` — by typing keystrokes into a
real pseudo-terminal and reading what comes back.

Because the agent is already reachable from Flight Deck web chat and
WhatsApp, this tool makes the same terminal drivable from both channels
with no extra transport.
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

from captain_claw.logging import get_logger
from captain_claw.terminal import DEFAULT_URL, TOKEN_HEADER
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# CSI / OSC / other escape sequences — stripped before returning output to
# text channels (WhatsApp, chat) where raw ANSI is noise.
_ANSI_RE = re.compile(
    r"""
    \x1b\][^\x07\x1b]*(?:\x07|\x1b\\)   # OSC ... BEL / ST
    | \x1b[@-Z\\-_]                      # two-char escapes
    | \x1b\[[0-?]*[ -/]*[@-~]            # CSI sequences
    | \r                                # bare carriage returns
    """,
    re.VERBOSE,
)


def _strip_ansi(text: str) -> str:
    cleaned = _ANSI_RE.sub("", text)
    # Collapse runs of blank lines a TUI redraw tends to leave behind.
    return re.sub(r"\n{3,}", "\n\n", cleaned)


class TerminalTool(Tool):
    """Open and drive interactive terminal sessions on the local machine."""

    name = "terminal"
    description = (
        "Run commands and drive a real terminal (PTY) on the USER'S OWN paired "
        "machine — their Mac/laptop, reached over Flight Deck. This is NOT the "
        "agent's host: `shell` runs here on the server, `terminal` runs on the "
        "user's computer. Strongly prefer `terminal` whenever the user refers "
        "to their own machine ('on my Mac', 'my laptop', 'my computer', 'at "
        "home', 'on my machine') — even for a one-shot command.\n"
        "For a single command, use action='run' (open→execute→capture→close in "
        "one call) — as simple as `shell`, but on the user's machine.\n"
        "For interactive programs that need a live tty — REPLs (python3, node), "
        "CLIs that prompt (claude, ssh, psql), or full-screen TUIs — use the "
        "session flow: `open` (returns session_id) → `send` text/keys and read "
        "what it prints → `send` again → `close`. Sessions persist across "
        "turns; reuse the session_id. Send control keys (ctrl-c, esc, up, "
        "enter, tab) via `key`. A user message prefixed with `$ ` means run it "
        "as a raw terminal command in the active session."
    )
    timeout_seconds = 150.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run", "open", "send", "read", "list", "close", "resize"],
                "description": (
                    "run: execute one command on the user's machine and return "
                    "its output (no session to manage). open: start a persistent "
                    "session. send: type text/keys and return the resulting "
                    "output. read: get new output without typing. list: show "
                    "sessions. close: end a session. resize: set size."
                ),
            },
            "session_id": {
                "type": "string",
                "description": "Session to act on (required for send/read/close/resize).",
            },
            "command": {
                "type": "string",
                "description": (
                    "run: the command to execute. open: program to run "
                    "(default: interactive login shell)."
                ),
            },
            "cwd": {
                "type": "string",
                "description": "open only: working directory (default: home).",
            },
            "data": {
                "type": "string",
                "description": "send only: text to type. Combine with enter=true to submit.",
            },
            "key": {
                "type": "string",
                "description": (
                    "send only: a named key/chord instead of (or after) text — "
                    "e.g. enter, tab, esc, up, down, ctrl-c, ctrl-d, ctrl-l."
                ),
            },
            "enter": {
                "type": "boolean",
                "description": "send only: append Enter after the text (submit the line).",
            },
            "wait": {
                "type": "number",
                "description": "send/read: max seconds to wait for output (default 2).",
            },
            "raw": {
                "type": "boolean",
                "description": "Keep ANSI escape codes in output (default false — stripped).",
            },
            "cols": {"type": "number", "description": "resize/open: terminal width."},
            "rows": {"type": "number", "description": "resize/open: terminal height."},
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._base_url = (os.environ.get("CLAW_PTY_URL") or DEFAULT_URL).rstrip("/")
        self._token = os.environ.get("CLAW_PTY_TOKEN") or None

    def _headers(self) -> dict[str, str]:
        return {TOKEN_HEADER: self._token} if self._token else {}

    async def _call(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.post(
                f"{self._base_url}{path}", json=payload, headers=self._headers()
            )
            resp.raise_for_status()
            return resp.json()

    def _fmt_output(self, data: dict[str, Any], raw: bool) -> str:
        text = data.get("output", "")
        if not raw:
            text = _strip_ansi(text)
        text = text.strip("\n")
        lines = []
        if not data.get("alive", True):
            code = data.get("exit_code")
            lines.append(f"[session ended, exit code {code}]")
        if not text:
            lines.append("[no new output]")
        else:
            lines.append(text)
        return "\n".join(lines)

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip().lower()
        raw = bool(kwargs.get("raw", False))
        wait = float(kwargs.get("wait", 2.0))
        try:
            if action == "run":
                command = kwargs.get("command")
                if not command:
                    return ToolResult(success=False, error="run requires `command`.")
                # One-shot: run under a login shell (full user env/PATH) in a
                # PTY, drain output until it exits, then clean up the session.
                data = await self._call(
                    "/open", {"cmd": ["/bin/bash", "-lc", str(command)], "cwd": kwargs.get("cwd")}
                )
                sid = data["session_id"]
                collected: list[str] = []
                exit_code: int | None = None
                for _ in range(60):
                    out = await self._call(
                        "/read", {"session_id": sid, "wait": wait, "settle": 0.3}
                    )
                    collected.append(out.get("output", ""))
                    exit_code = out.get("exit_code")
                    if not out.get("alive", False):
                        break
                try:
                    await self._call("/close", {"session_id": sid})
                except Exception:
                    pass
                text = "".join(collected)
                if not raw:
                    text = _strip_ansi(text)
                text = text.strip("\n")
                suffix = "" if exit_code in (0, None) else f"\n[exit code {exit_code}]"
                return ToolResult(
                    success=exit_code in (0, None),
                    content=(text or "[no output]") + suffix,
                )

            if action == "open":
                payload: dict[str, Any] = {}
                if kwargs.get("command"):
                    payload["cmd"] = kwargs["command"]
                if kwargs.get("cwd"):
                    payload["cwd"] = kwargs["cwd"]
                if kwargs.get("cols"):
                    payload["cols"] = int(kwargs["cols"])
                if kwargs.get("rows"):
                    payload["rows"] = int(kwargs["rows"])
                data = await self._call("/open", payload)
                sid = data["session_id"]
                # Surface the program's initial banner/prompt so the agent
                # sees the state right after opening.
                first = await self._call(
                    "/read", {"session_id": sid, "wait": max(wait, 1.5), "settle": 0.3}
                )
                banner = self._fmt_output(first, raw)
                return ToolResult(
                    success=True,
                    content=f"session_id: {sid} (pid {data.get('pid')})\n{banner}",
                )

            if action == "send":
                sid = self._require_sid(kwargs)
                send_payload = {"session_id": sid}
                if kwargs.get("data") is not None:
                    send_payload["data"] = str(kwargs["data"])
                if kwargs.get("key"):
                    send_payload["key"] = str(kwargs["key"])
                if kwargs.get("enter"):
                    send_payload["enter"] = True
                if "data" not in send_payload and "key" not in send_payload:
                    return ToolResult(
                        success=False,
                        error="send requires `data` and/or `key`.",
                    )
                await self._call("/input", send_payload)
                out = await self._call(
                    "/read", {"session_id": sid, "wait": wait, "settle": 0.3}
                )
                return ToolResult(success=True, content=self._fmt_output(out, raw))

            if action == "read":
                sid = self._require_sid(kwargs)
                out = await self._call(
                    "/read", {"session_id": sid, "wait": wait, "settle": 0.3}
                )
                return ToolResult(success=True, content=self._fmt_output(out, raw))

            if action == "list":
                data = await self._call("/list", {})
                sessions = data.get("sessions", [])
                if not sessions:
                    return ToolResult(success=True, content="[no active sessions]")
                lines = [
                    f"{s['session_id']}  {'alive' if s['alive'] else 'dead'}  "
                    f"{s['cmd']}  (idle {s['idle_seconds']}s)"
                    for s in sessions
                ]
                return ToolResult(success=True, content="\n".join(lines))

            if action == "close":
                sid = self._require_sid(kwargs)
                await self._call("/close", {"session_id": sid})
                return ToolResult(success=True, content=f"[closed {sid}]")

            if action == "resize":
                sid = self._require_sid(kwargs)
                await self._call("/resize", {
                    "session_id": sid,
                    "cols": int(kwargs.get("cols") or 120),
                    "rows": int(kwargs.get("rows") or 40),
                })
                return ToolResult(success=True, content=f"[resized {sid}]")

            return ToolResult(success=False, error=f"unknown action {action!r}")

        except httpx.ConnectError:
            return ToolResult(
                success=False,
                error=(
                    f"PTY daemon not reachable at {self._base_url}. Start it with "
                    "`python -m captain_claw.terminal.daemon` on this machine."
                ),
            )
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip() or str(exc)
            return ToolResult(success=False, error=f"terminal daemon: {detail}")
        except Exception as exc:  # pragma: no cover - defensive
            log.error("terminal tool failed", action=action, error=str(exc))
            return ToolResult(success=False, error=str(exc))

    @staticmethod
    def _require_sid(kwargs: dict[str, Any]) -> str:
        sid = str(kwargs.get("session_id", "")).strip()
        if not sid:
            raise ValueError("session_id is required for this action")
        return sid
