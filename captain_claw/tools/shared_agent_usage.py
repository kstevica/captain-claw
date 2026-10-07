"""shared_agent_usage — the owner's agent looks into how its members use it (PR D).

Thin wrapper around :mod:`captain_claw.shared_usage`, which holds the gates,
the Flight Deck roster, the local queries and every text. Owner only: members
never get it (it is in no speaker allowlist, and ``shared_usage.run`` refuses
any call that isn't the owner's), and the registry lists it only while Flight
Deck says this agent has members (``requires_shared_members``).
"""

from typing import Any

from captain_claw import member_privacy, shared_usage
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class SharedAgentUsageTool(Tool):
    """Who this agent is shared with, and how they use it (owner only)."""

    name = member_privacy.TOOL_NAME
    description = shared_usage.TOOL_DESCRIPTION
    parameters = shared_usage.PARAMETERS

    async def execute(
        self,
        action: str,
        member: str | None = None,
        conversation: str | None = None,
        query: str | None = None,
        table: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            return await shared_usage.run(
                action, kwargs.get("_agent"), member=member, conversation=conversation,
                query=query, table=table, offset=offset, limit=limit,
            )
        except Exception as exc:
            # Never the exception text: it could quote a member's data.
            log.warning("shared_agent_usage failed", error=type(exc).__name__)
            return ToolResult(success=False, error=shared_usage.UNAVAILABLE_MESSAGE)
