"""Send a stored app file to a WhatsApp chat via Flight Deck.

The agent saves files into its Flight Deck *app_files* store (the same
files surfaced in the Flight Deck / glasses web app). This tool delivers
one of those files to a WhatsApp chat as a *document*, going through the
Flight Deck WhatsApp bridge (which owns the Meta Cloud API credentials).

Two actions:
  * ``list`` — list the agent's saved files (newest first) so the agent can
    discover which one the user means ("that last report").
  * ``send`` — deliver a file. Identify it by ``file_id``, by ``filename``
    (fuzzy, case-insensitive, newest match wins), or ``latest`` for the most
    recently saved file.

Recipient resolution for ``send``:
  * An explicit ``to`` (phone number, digits only, no ``+``) wins.
  * Otherwise the *current* WhatsApp chat — captured per session when the
    conversation arrived over WhatsApp (``session.metadata['whatsapp_waid']``).

Requirements: the agent must run under Flight Deck (``FD_URL`` +
``FD_AGENT_SLUG``) with WhatsApp configured on the Flight Deck side.
WhatsApp only permits free-form documents within 24h of the recipient's
last message; out-of-window sends are rejected by Meta and surfaced here
as a clear error.
"""

from __future__ import annotations

import json
from typing import Any

from captain_claw.fd_client import FDClient, flight_deck_base, flight_deck_slug
from captain_claw.tools.registry import Tool, ToolResult


class WhatsAppSendFileTool(Tool):
    """List and deliver app_files documents to a WhatsApp chat."""

    name = "whatsapp_send_file"
    timeout_seconds = 120.0
    description = (
        "Send a file the agent saved (in its Flight Deck files) to a WhatsApp "
        "chat as a document. Use this whenever the user wants a file delivered "
        "over WhatsApp — e.g. 'send me that report on WhatsApp', 'whatsapp me "
        "the invoice', 'get me the last report'. By default it sends into the "
        "current WhatsApp chat; pass 'to' (phone number, digits only, no '+') "
        "to send to a specific number. Identify the file by 'file_id', by "
        "'filename' (fuzzy match), or set 'latest' for the most recently saved "
        "file. If unsure which file the user means, first call with "
        "action='list' to see available files, then send. WhatsApp only allows "
        "sending within 24 hours of the recipient's last message."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["send", "list"],
                "description": (
                    "'send' delivers a file (default); 'list' returns the "
                    "agent's saved files so you can pick which to send."
                ),
            },
            "file_id": {
                "type": "string",
                "description": "Flight Deck file id of the file to send.",
            },
            "filename": {
                "type": "string",
                "description": (
                    "Filename (or part of it) to send. Fuzzy, case-insensitive; "
                    "the newest match wins. Used when 'file_id' is not given."
                ),
            },
            "latest": {
                "type": "boolean",
                "description": (
                    "Send the most recently saved file. Handy for 'send me the "
                    "last report' when no exact name is known."
                ),
            },
            "to": {
                "type": "string",
                "description": (
                    "Recipient WhatsApp number, digits only, no '+'. Omit to "
                    "reply into the current WhatsApp chat."
                ),
            },
            "caption": {
                "type": "string",
                "description": "Optional caption shown under the document.",
            },
        },
        "required": [],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action") or "send").strip().lower()

        # Must run under Flight Deck to reach the WhatsApp bridge / file store.
        if not flight_deck_base():
            return ToolResult(
                success=False,
                error=(
                    "Not running under Flight Deck (FD_URL unset); cannot "
                    "reach WhatsApp or the file store."
                ),
            )
        agent_id = flight_deck_slug()
        if not agent_id:
            return ToolResult(
                success=False,
                error="FD_AGENT_SLUG not set; cannot locate this agent's files.",
            )

        if action == "list":
            return await self._list(agent_id)
        return await self._send(agent_id, kwargs)

    async def _list(self, agent_id: str) -> ToolResult:
        ok, data, err = await self._post("/whatsapp/list-files", {"agent_id": agent_id})
        if not ok:
            return ToolResult(success=False, error=f"Could not list files: {err}")
        files = data.get("files") or []
        if not files:
            return ToolResult(success=True, content="No saved files found.")
        lines = [
            f"- {f.get('filename')} (id={f.get('file_id')}, {f.get('size')} bytes, "
            f"{f.get('uploaded_at')})"
            for f in files
        ]
        return ToolResult(
            success=True,
            content="Saved files (newest first):\n" + "\n".join(lines),
        )

    async def _send(self, agent_id: str, kwargs: dict[str, Any]) -> ToolResult:
        file_id = str(kwargs.get("file_id") or "").strip()
        filename = str(kwargs.get("filename") or "").strip()
        latest = bool(kwargs.get("latest"))
        to = str(kwargs.get("to") or "").lstrip("+").strip()
        caption = str(kwargs.get("caption") or "").strip()

        if not file_id and not filename and not latest:
            return ToolResult(
                success=False,
                error=(
                    "Specify which file to send: 'file_id', 'filename', or "
                    "latest=true. Use action='list' to see available files."
                ),
            )

        # Recipient: explicit 'to' wins, else the current WhatsApp chat.
        if not to:
            session = kwargs.get("_session")
            meta = getattr(session, "metadata", None) if session is not None else None
            if isinstance(meta, dict):
                to = str(meta.get("whatsapp_waid") or "").lstrip("+").strip()
        if not to:
            return ToolResult(
                success=False,
                error=(
                    "No recipient: this conversation isn't a WhatsApp chat and "
                    "no 'to' number was provided."
                ),
            )

        payload: dict[str, Any] = {"agent_id": agent_id, "to": to, "caption": caption}
        if file_id:
            payload["file_id"] = file_id
        if filename:
            payload["filename"] = filename
        if latest:
            payload["latest"] = True

        ok, data, err = await self._post("/whatsapp/send-file", payload)
        if ok and data.get("ok"):
            sent_name = data.get("filename") or filename or file_id or "file"
            return ToolResult(success=True, content=f"Sent '{sent_name}' to WhatsApp {to}.")
        detail = err or str(data.get("error") or data.get("detail") or "").strip()
        return ToolResult(success=False, error=f"WhatsApp send failed: {detail or 'unknown error'}")

    async def _post(
        self, path: str, payload: dict[str, Any]
    ) -> tuple[bool, dict[str, Any], str]:
        """POST to a Flight Deck endpoint. Returns (ok, json, error)."""
        fd = FDClient(timeout=120.0)
        try:
            resp = await fd.post(path, json=payload)
        except Exception as exc:
            return False, {}, f"Failed to reach Flight Deck: {exc}"
        finally:
            await fd.close()
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            data = {}
        if resp.status_code != 200:
            detail = str(data.get("error") or data.get("detail") or resp.text or "").strip()
            return False, data, detail or f"HTTP {resp.status_code}"
        return True, data, ""
