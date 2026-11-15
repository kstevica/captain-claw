"""`request_secret` tool — ask the person for a password or a one-time code.

The person types it into a masked box in their Flight Deck. The agent gets a
reference (``sec_…``) and the length, never the value, and fills the field
with ``browser`` (``type`` or ``form_input``) passing ``secret_ref`` instead
of ``text``: the browser fetches the value as it types and says nothing of it
back. The reference works for 10 minutes, up to 3 times, for this agent and
this person only (``flight_deck/secrets_store.py``).

Same dialog as ``ask_user`` (``person_asks``, a ``secret`` question): nobody
to ask on a turn that started on its own, and outside Flight Deck (Telegram,
WhatsApp…) the box can't reach them — the tool says so, and the agent asks
them to type it in Flight Deck or do that step themselves. It never asks for a
secret in a chat message.
"""

from __future__ import annotations

from typing import Any

import httpx

from captain_claw.tools.ask_user import NO_FLIGHT_DECK, _fd_base, ask_and_follow
from captain_claw.tools.registry import Tool, ToolResult

_KINDS = {"password": "password", "code": "one-time code", "other": "secret"}


class RequestSecretTool(Tool):
    name = "request_secret"
    description = (
        "Ask the person you're working for to type a password or a one-time code (2FA, SMS, email code) into a "
        "masked box in their Flight Deck — for a login or form you're filling in the browser. You get a "
        "reference (secret_ref) and its length, never the value: pass secret_ref (instead of text) to browser "
        "type or form_input on the field, and the browser types it without showing you. A reference works for "
        "10 minutes, up to 3 times. Never ask for a password or code in a message, and never write one into a "
        "file, a note or memory. In Telegram, WhatsApp or other channels, or a turn that started on its own, "
        "the box can't reach them: the tool says so — then tell them, and let them sign in themselves."
    )
    parameters = {
        "type": "object",
        "properties": {
            "what": {"type": "string",
                     "description": "What to type, plainly: 'Your password for portal.example.com', 'The 6-digit "
                                    "code your bank just sent you'."},
            "why": {"type": "string", "description": "A line on what you'll do with it."},
            "kind": {"type": "string", "enum": list(_KINDS), "description": "password, code, or other."},
            "wait_minutes": {"type": "number", "description": "How long to wait (1-30, default 5)."},
        },
        "required": ["what"],
    }
    timeout_seconds = 1_900.0

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not _fd_base():
            return ToolResult(success=False, error=NO_FLIGHT_DECK)
        what = " ".join(str(kwargs.get("what") or "").split())[:300]
        if not what:
            return ToolResult(success=False, error="Say what they should type: what='Your password for …'.")
        kind = _KINDS.get(str(kwargs.get("kind") or "").strip().lower(), "secret")
        why = " ".join(str(kwargs.get("why") or "").split())[:300]
        help_text = (why + " " if why else "") + "It goes straight into the page; the agent never sees it."
        try:
            minutes = float(kwargs.get("wait_minutes") or 5)
        except (TypeError, ValueError):
            minutes = 5.0
        wait_s = max(1.0, min(minutes, 30.0)) * 60
        question = {"id": "secret", "question": what, "help": help_text[:500], "secret": True}
        try:
            status, data = await ask_and_follow([question], wait_s, kwargs.get("_agent"),
                                                kwargs.get("_stream_callback"),
                                                waiting=f"Waiting for them to type the {kind} in Flight Deck…\n")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=f"Flight Deck didn't answer ({type(exc).__name__}). "
                                                   "Let them do this step themselves.")
        if status != 200 or not isinstance(data, dict):
            detail = data.get("detail") if isinstance(data, dict) else data
            return ToolResult(success=False, error=f"The box couldn't be shown: {detail}")
        if data.get("elsewhere") or data.get("nobody"):
            return ToolResult(success=False, error=(
                f"{data.get('error') or 'The box can’t reach them here.'} Don't ask for the {kind} in a message: "
                "tell them what you need, and let them sign in or type it themselves."))
        if data.get("declined"):
            return ToolResult(content=f"They chose not to give the {kind}. Don't ask again; tell them where you "
                                      "stopped so they can do that step themselves.")
        if not data.get("ok"):
            return ToolResult(content=f"No {kind} within {int(wait_s // 60)} minutes. Tell them where you stopped; "
                                      "ask again only if they say so.")
        given = (data.get("answers") or {}).get("secret") or {}
        ref, length = given.get("secret_ref"), given.get("length")
        if not ref:
            return ToolResult(success=False, error="Flight Deck gave no reference for it.")
        return ToolResult(content=(
            f"They typed the {kind} ({length} characters). To fill it in, call browser with action='type' (or "
            f"'form_input') on the field and secret_ref='{ref}' instead of text. You never see the value; the "
            "reference works for 10 minutes, up to 3 times. Don't repeat the reference in messages."))
