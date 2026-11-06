"""`custom_tools` tool — run the person's custom tools: small Python programs
Crew's coding agents wrote, which Flight Deck keeps and runs.

Flight Deck runs every call, never the agent (flight_deck/external_tools.py):
one process per call, the input as JSON, one JSON value back. Shown only while
the owner's Flight Deck has custom tools on (``custom_tools_client``). On a
shared agent's member turn the calls carry the turn's grant, so Flight Deck runs
the member's own tools with the member's secrets and files.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from captain_claw import custom_tools_client
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

MAX_SHOWN = 12_000  # characters of a tool's output put in the result


def _refused(status: int, data: Any) -> ToolResult | None:
    if status < 300:
        return None
    detail = data.get("detail") if isinstance(data, dict) else data
    return ToolResult(success=False, error=str(detail or f"Flight Deck answered {status}"))


def _schema_line(schema: dict) -> str:
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    parts = []
    for key, prop in props.items():
        if not isinstance(prop, dict):
            continue
        kind = "file (vfs: path)" if prop.get("format") == "file" else str(prop.get("type") or "any")
        if prop.get("type") == "array" and isinstance(prop.get("items"), dict) and prop["items"].get("format") == "file":
            kind = "list of files (vfs: paths)"
        what = f" — {prop['description']}" if prop.get("description") else ""
        parts.append(f"- {key}{'' if key in required else ' (optional)'}: {kind}{what}")
    return "\n".join(parts) or "- (no fields)"


class CustomToolsTool(Tool):
    name = "custom_tools"
    description = (
        "Run the user's custom tools: small programs made for jobs that code does better than a model, like "
        "reading a file format, adding up a spreadsheet the same way every time, or calling one API. Flight Deck "
        "runs each call and gives back its JSON result; you never run their code yourself. "
        "Actions: list (the tools you can use now, with a line on each); describe (one tool's input fields, what "
        "it returns, examples, and whether its secrets are set: do this before the first run of a tool); run "
        "(tool = its name, input = an object with its fields). "
        "Files: a field described as a file takes a vfs: path (vfs:<folder>/<path>), so save a file the user sent "
        "into the VFS first with write. Files the tool makes come back in vfs:custom-tools/<tool>/…; read them "
        "with read. "
        "If a run fails, tell the user what it said; don't guess the result. A missing secret is set by the user "
        "on the Custom tools page, never typed into a chat."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "describe", "run"], "description": "What to do."},
            "tool": {"type": "string", "description": "The tool's name, as list shows it (describe, run)."},
            "input": {"type": "object", "description": "run: the tool's input fields, as describe lists them."},
        },
        "required": ["action"],
    }
    timeout_seconds = 380.0

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        if not custom_tools_client.under_flight_deck():
            return ToolResult(success=False, error=custom_tools_client.NO_FLIGHT_DECK)
        action = str(action or "").strip().lower()
        try:
            if action == "list":
                return await self._list()
            if action == "describe":
                return await self._describe(str(kwargs.get("tool") or ""))
            if action == "run":
                return await self._run(str(kwargs.get("tool") or ""), kwargs.get("input"))
            return ToolResult(success=False, error=f"Unknown action '{action}'. Use list, describe or run.")
        except httpx.TimeoutException:
            return ToolResult(success=False, error=(
                "Flight Deck didn't answer in time. The tool may still have run; don't say it did or didn't."))
        except Exception as exc:  # noqa: BLE001
            why = str(exc) or type(exc).__name__
            log.warning("custom_tools error", action=action, error=why)
            return ToolResult(success=False, error=f"Custom tools request failed: {why}")

    async def _list(self) -> ToolResult:
        status, data = await custom_tools_client.call("GET", "tools")
        if (bad := _refused(status, data)) is not None:
            return bad
        tools = data.get("tools") or []
        if not tools:
            return ToolResult(content="No custom tools are ready to use yet.")
        lines = [f"Custom tools you can run ({len(tools)}):"]
        for t in tools:
            missing = t.get("secrets_missing") or []
            note = f" — needs {', '.join(missing)} set on the Custom tools page first" if missing else ""
            lines.append(f"- {t['ref']}: {t.get('title') or ''}. {t.get('description') or ''}{note}")
        lines.append("Use describe on a tool before its first run.")
        return ToolResult(content="\n".join(lines))

    async def _describe(self, tool: str) -> ToolResult:
        if not tool.strip():
            return ToolResult(success=False, error="Name the tool: custom_tools action=describe tool=<name>.")
        status, data = await custom_tools_client.call("POST", "describe", {"tool": tool})
        if (bad := _refused(status, data)) is not None:
            return bad
        lines = [f"{data.get('title')} ({data.get('ref')}, version {data.get('version')})",
                 str(data.get("description") or ""), "", "Input:", _schema_line(data.get("input") or {})]
        out_schema = data.get("output") or {}
        if out_schema.get("properties"):
            lines += ["", "Returns:", _schema_line(out_schema)]
        examples = data.get("examples") or []
        if examples:
            lines += ["", "Example input: " + json.dumps(examples[0].get("input"), ensure_ascii=False)[:600]]
        secrets = data.get("secrets") or []
        if secrets:
            lines += ["", "Secrets: " + ", ".join(f"{s['name']} ({'set' if s.get('set') else 'NOT set'})" for s in secrets)]
        lines += ["", f"Limits: {data.get('timeout_s')} s per run."
                  + (" It uses the internet." if data.get("network") else "")]
        return ToolResult(content="\n".join(lines))

    async def _run(self, tool: str, data_in: Any) -> ToolResult:
        if not tool.strip():
            return ToolResult(success=False, error="Name the tool: custom_tools action=run tool=<name> input={...}.")
        if data_in is None:
            data_in = {}
        if isinstance(data_in, str):
            try:
                data_in = json.loads(data_in)
            except ValueError:
                return ToolResult(success=False, error="input must be an object with the tool's fields.")
        if not isinstance(data_in, dict):
            return ToolResult(success=False, error="input must be an object with the tool's fields.")
        status, data = await custom_tools_client.call("POST", "run", {"tool": tool, "input": data_in})
        if (bad := _refused(status, data)) is not None:
            return bad
        if not data.get("ok"):
            err = data.get("error") or {}
            text = f"{tool} failed ({err.get('kind')}): {err.get('message')}"
            if data.get("stderr"):
                text += f"\nIts error output (last part):\n{data['stderr'][-2000:]}"
            return ToolResult(success=False, error=text)
        shown = json.dumps(data.get("output"), ensure_ascii=False, indent=2)
        if len(shown) > MAX_SHOWN:
            shown = shown[:MAX_SHOWN] + f"\n… (cut: {len(shown)} characters in all)"
        lines = [f"{tool} (version {data.get('version')}, {data.get('seconds')} s) returned:", shown]
        files = data.get("files") or []
        if files:
            lines += ["", "Files it made:"] + [f"- {f['path']} ({f.get('size', 0)} bytes)" for f in files]
        if data.get("files_skipped"):
            lines.append(f"({data['files_skipped']} more files weren't kept: too many or too big.)")
        return ToolResult(content="\n".join(lines))
