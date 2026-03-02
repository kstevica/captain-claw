"""BotPort tool - consult specialist agents through the BotPort network."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from captain_claw.tools.registry import Tool, ToolResult

if TYPE_CHECKING:
    from captain_claw.botport_client import BotPortClient


class BotPortTool(Tool):
    """Consult specialist agents through the BotPort agent network.

    BotPort connects multiple Captain Claw instances. Use this tool to
    delegate tasks to agents with specific expertise.
    """

    name = "botport"
    description = (
        "Consult a specialist agent through the BotPort network. "
        "Use this to delegate tasks to agents with specific expertise "
        "(e.g., legal, coding, research, creative writing). "
        "Available actions: consult, follow_up, close, status, list_agents."
    )
    timeout_seconds = 300.0  # Concerns can take several minutes.
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["consult", "follow_up", "close", "status", "list_agents"],
                "description": (
                    "Action: 'consult' to send a new task, 'follow_up' to continue, "
                    "'close' to end, 'status' to check, 'list_agents' to see available agents"
                ),
            },
            "task": {
                "type": "string",
                "description": "The task or question for the specialist agent (required for consult)",
            },
            "expertise": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Expertise tags to match the right specialist "
                    "(e.g., ['legal', 'contracts'] or ['python', 'backend'])"
                ),
            },
            "context": {
                "type": "string",
                "description": "Relevant context or background information for the task",
            },
            "concern_id": {
                "type": "string",
                "description": "Concern ID for follow_up, close, or status actions",
            },
            "message": {
                "type": "string",
                "description": "Follow-up message (required for follow_up action)",
            },
        },
        "required": ["action"],
    }

    def __init__(self, client: BotPortClient | None = None) -> None:
        self._client = client

    def set_client(self, client: BotPortClient) -> None:
        """Set the BotPort client reference."""
        self._client = client

    async def execute(
        self,
        action: str = "",
        task: str = "",
        expertise: list[str] | None = None,
        context: str = "",
        concern_id: str = "",
        message: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        if not self._client:
            return ToolResult(success=False, error="BotPort is not configured")

        if not self._client.connected:
            return ToolResult(success=False, error="Not connected to BotPort")

        action = action.strip().lower()

        if action == "consult":
            return await self._consult(task, expertise, context, kwargs)
        elif action == "follow_up":
            return await self._follow_up(concern_id, message, context)
        elif action == "close":
            return await self._close(concern_id)
        elif action == "status":
            return await self._status(concern_id)
        elif action == "list_agents":
            return await self._list_agents()
        else:
            return ToolResult(
                success=False,
                error=f"Unknown action: {action}. Use: consult, follow_up, close, status, list_agents",
            )

    async def _consult(
        self,
        task: str,
        expertise: list[str] | None,
        context: str,
        kwargs: dict[str, Any],
    ) -> ToolResult:
        if not task.strip():
            return ToolResult(success=False, error="'task' is required for consult action")

        session_id = str(kwargs.get("_session_id", "") or "")
        context_dict = {"summary": context} if context else {}

        result = await self._client.send_concern(
            task=task,
            context=context_dict,
            expertise_tags=expertise or [],
            session_id=session_id,
        )

        if result.get("ok"):
            concern_id = result.get("concern_id", "")
            from_name = result.get("from_instance_name", "specialist")
            persona_name = result.get("persona_name", "")
            response = result.get("response", "")
            attribution = from_name
            if persona_name and persona_name != from_name:
                attribution = f"{from_name} (persona: {persona_name})"
            return ToolResult(
                success=True,
                content=(
                    f"Response from {attribution} (concern_id: {concern_id}):\n\n"
                    f"{response}"
                ),
            )
        else:
            error = result.get("error", "Unknown error")
            concern_id = result.get("concern_id", "")
            return ToolResult(
                success=False,
                error=f"Consultation failed: {error} (concern_id: {concern_id})",
            )

    async def _follow_up(
        self, concern_id: str, message: str, context: str,
    ) -> ToolResult:
        if not concern_id.strip():
            return ToolResult(success=False, error="'concern_id' is required for follow_up")
        if not message.strip():
            return ToolResult(success=False, error="'message' is required for follow_up")

        context_dict = {"summary": context} if context else {}
        result = await self._client.send_follow_up(
            concern_id=concern_id,
            message=message,
            additional_context=context_dict,
        )

        if result.get("ok"):
            response = result.get("response", "")
            persona_name = result.get("persona_name", "")
            persona_info = f" [persona: {persona_name}]" if persona_name else ""
            return ToolResult(
                success=True,
                content=f"Follow-up response{persona_info} (concern_id: {concern_id}):\n\n{response}",
            )
        else:
            return ToolResult(
                success=False,
                error=f"Follow-up failed: {result.get('error', 'Unknown error')}",
            )

    async def _close(self, concern_id: str) -> ToolResult:
        if not concern_id.strip():
            return ToolResult(success=False, error="'concern_id' is required for close")

        ok = await self._client.close_concern(concern_id)
        if ok:
            return ToolResult(success=True, content=f"Concern {concern_id} closed.")
        return ToolResult(success=False, error="Failed to close concern")

    async def _status(self, concern_id: str) -> ToolResult:
        if not concern_id.strip():
            return ToolResult(
                success=True,
                content=f"Connected to BotPort: {self._client.connected}",
            )
        return ToolResult(
            success=True,
            content=f"Concern {concern_id}: status check requires BotPort API (coming in v2)",
        )

    async def _list_agents(self) -> ToolResult:
        agents = await self._client.list_agents()
        if not agents:
            return ToolResult(success=True, content="No agents info available.")

        lines: list[str] = ["Connected to BotPort. Known capabilities:"]
        for agent_info in agents:
            lines.append(json.dumps(agent_info, indent=2, default=str))
        return ToolResult(success=True, content="\n".join(lines))
