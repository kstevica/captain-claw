"""REST handler for the LLM-driven playbook wizard."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

_INSTRUCTIONS_DIR = Path(__file__).resolve().parent.parent / "instructions"
_SYSTEM_PROMPT: str | None = None


def _load_system_prompt() -> str:
    """Load and cache the wizard system prompt."""
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        path = _INSTRUCTIONS_DIR / "playbook_wizard_system_prompt.md"
        _SYSTEM_PROMPT = path.read_text(encoding="utf-8")
    return _SYSTEM_PROMPT


def _parse_llm_json(raw: str) -> dict:
    """Extract a JSON object from LLM response text."""
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in the text.
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {"type": "question", "phase": "outcome", "content": "I had trouble processing that. Could you rephrase?"}


async def wizard_step(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/playbook-wizard/step — advance the wizard conversation.

    Body: {"messages": [{"role": "assistant"|"user", "content": "..."}]}
    Returns: {"type": "question"|"playbook", ...}
    """
    from captain_claw.llm import Message

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    conversation = body.get("messages", [])
    if not isinstance(conversation, list):
        return web.json_response({"error": "messages must be a list"}, status=400)

    # Build LLM messages: system prompt + conversation history.
    system_prompt = _load_system_prompt()
    messages = [Message(role="system", content=system_prompt)]
    for msg in conversation:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        if role in ("user", "assistant") and content.strip():
            messages.append(Message(role=role, content=content))

    # Get provider.
    provider = server.agent.provider if server.agent else None
    if provider is None:
        from captain_claw.llm import get_provider
        provider = get_provider()

    # Broadcast wizard activity to monitor panel.
    turn_num = sum(1 for m in conversation if m.get("role") == "user")
    server._broadcast({
        "type": "monitor",
        "tool_name": "playbook_wizard",
        "arguments": {"phase": "llm_call", "turn": turn_num},
        "output": f"Wizard step {turn_num}: calling LLM…",
    })

    try:
        response = await asyncio.wait_for(
            provider.complete(messages=messages, tools=None, max_tokens=2000),
            timeout=60.0,
        )
        raw = str(getattr(response, "content", "") or "").strip()
        result = _parse_llm_json(raw)

        # Log wizard result to monitor.
        usage = getattr(response, "usage", None)
        tokens_info = ""
        if usage:
            prompt_t = getattr(usage, "prompt_tokens", 0) or 0
            completion_t = getattr(usage, "completion_tokens", 0) or 0
            tokens_info = f" (tokens: {prompt_t}+{completion_t})"
        result_type = result.get("type", "unknown")
        phase = result.get("phase", "")
        if result_type == "playbook":
            pb_name = result.get("playbook", {}).get("name", "?")
            output_msg = f"Wizard generated playbook: {pb_name}{tokens_info}"
        else:
            output_msg = f"Wizard asked question (phase: {phase}){tokens_info}"
        server._broadcast({
            "type": "monitor",
            "tool_name": "playbook_wizard",
            "arguments": {"phase": result_type, "turn": turn_num},
            "output": output_msg,
        })

        return web.json_response(result)
    except asyncio.TimeoutError:
        server._broadcast({
            "type": "monitor",
            "tool_name": "playbook_wizard",
            "arguments": {"phase": "error", "turn": turn_num},
            "output": "Wizard LLM call timed out",
        })
        return web.json_response(
            {"error": "LLM request timed out"}, status=504,
        )
    except Exception as exc:
        log.warning("Playbook wizard step failed", error=str(exc))
        server._broadcast({
            "type": "monitor",
            "tool_name": "playbook_wizard",
            "arguments": {"phase": "error", "turn": turn_num},
            "output": f"Wizard LLM call failed: {exc}",
        })
        return web.json_response(
            {"error": f"LLM call failed: {exc}"}, status=500,
        )
