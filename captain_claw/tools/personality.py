"""Personality tool for reading and updating the agent personality profile."""

from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


_AGENT_DESCRIPTION = (
    "Read or update the agent personality profile. "
    "The personality defines the agent's name, description, background, "
    "and areas of expertise. Any name is accepted; 'of the Captain Claw family' "
    "is automatically appended in the system prompt. "
    "Use action 'get' to view the current personality, 'update' to modify fields."
)

_USER_DESCRIPTION = (
    "Read or update the current user's profile. "
    "The profile stores the user's name, description, background, and areas of "
    "expertise — this tells the agent who it is talking to. "
    "Use this when the user asks you to remember them, save their profile, "
    "update their expertise, or change how you address them. "
    "Use action 'get' to view the profile, 'update' to modify fields."
)

_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["get", "update"],
            "description": "Operation to perform.",
        },
        "name": {
            "type": "string",
            "description": "Agent name (any name accepted). For 'update' only.",
        },
        "description": {
            "type": "string",
            "description": "Short description of the agent. For 'update' only.",
        },
        "background": {
            "type": "string",
            "description": "Background / origin story. For 'update' only.",
        },
        "expertise": {
            "type": "string",
            "description": "Comma-separated list of expertise areas. For 'update' only.",
        },
        "instructions": {
            "type": "string",
            "description": "Additional freeform instructions injected into the system prompt. For 'update' only.",
        },
    },
    "required": ["action"],
}

_USER_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["get", "update"],
            "description": "Operation to perform.",
        },
        "name": {
            "type": "string",
            "description": "User's name. For 'update' only.",
        },
        "description": {
            "type": "string",
            "description": "Short description of the user (role, title, etc). For 'update' only.",
        },
        "background": {
            "type": "string",
            "description": "User's background and experience. For 'update' only.",
        },
        "expertise": {
            "type": "string",
            "description": "Comma-separated list of the user's expertise areas. For 'update' only.",
        },
        "instructions": {
            "type": "string",
            "description": "Additional freeform instructions for the agent when talking to this user. For 'update' only.",
        },
    },
    "required": ["action"],
}


class PersonalityTool(Tool):
    """Read or update personality / user profile.

    When ``_user_id`` is set (Telegram agents), the tool operates on the
    user's profile (who the agent is talking to).  Otherwise it falls back
    to the global agent personality.
    """

    name = "personality"
    description = _AGENT_DESCRIPTION
    parameters = _AGENT_PARAMETERS

    def __init__(self) -> None:
        # Set by the agent during registration when a user_id is known.
        self._user_id: str | None = None

    def set_user_mode(self, user_id: str) -> None:
        """Switch the tool to user-profile mode for the given user."""
        self._user_id = user_id
        self.description = _USER_DESCRIPTION
        self.parameters = _USER_PARAMETERS

    async def execute(
        self,
        action: str,
        name: str | None = None,
        description: str | None = None,
        background: str | None = None,
        expertise: str | None = None,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            if action == "get":
                return self._get_personality()
            if action == "update":
                return self._update_personality(
                    name=name,
                    description=description,
                    background=background,
                    expertise=expertise,
                    instructions=instructions,
                )
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("Personality tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))

    # ------------------------------------------------------------------

    def _get_personality(self) -> ToolResult:
        from captain_claw.personality import load_effective_personality

        p = load_effective_personality(self._user_id)
        scope = f"user {self._user_id}" if self._user_id else "global"
        lines = [
            f"Scope: {scope}",
            f"Name: {p.name}",
            f"Description: {p.description}",
            f"Background: {p.background}",
            f"Expertise: {', '.join(p.expertise)}",
            f"Instructions: {p.instructions or '(none)'}",
        ]
        return ToolResult(success=True, content="\n".join(lines))

    def _update_personality(
        self,
        *,
        name: str | None,
        description: str | None,
        background: str | None,
        expertise: str | None,
        instructions: str | None,
    ) -> ToolResult:
        from captain_claw.personality import (
            Personality,
            load_effective_personality,
            load_user_personality,
            save_personality,
            save_user_personality,
        )

        changed: list[str] = []

        if self._user_id:
            # Per-user personality: create from scratch if none exists.
            p = load_user_personality(self._user_id) or Personality()
        else:
            p = load_effective_personality(None)

        if name is not None:
            name = name.strip()
            if not name:
                return ToolResult(
                    success=False,
                    error="Name cannot be empty.",
                )
            p.name = name
            changed.append("name")

        if description is not None:
            p.description = description.strip()
            changed.append("description")

        if background is not None:
            p.background = background.strip()
            changed.append("background")

        if expertise is not None:
            p.expertise = [e.strip() for e in expertise.split(",") if e.strip()]
            changed.append("expertise")

        if instructions is not None:
            p.instructions = instructions.strip()
            changed.append("instructions")

        if not changed:
            return ToolResult(
                success=False, error="No fields provided to update."
            )

        if self._user_id:
            save_user_personality(self._user_id, p)
        else:
            save_personality(p)

        scope = f"user {self._user_id}" if self._user_id else "global"
        return ToolResult(
            success=True,
            content=(
                f"Personality updated ({', '.join(changed)}, scope={scope}). "
                f"Name: {p.name}"
            ),
        )
