"""Personality tool for reading and updating the agent personality profile."""

from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class PersonalityTool(Tool):
    """Read or update the agent personality profile."""

    name = "personality"
    description = (
        "Read or update the agent personality profile. "
        "The personality defines the agent's name, description, background, "
        "and areas of expertise. Any name is accepted; 'of the Captain Claw family' "
        "is automatically appended in the system prompt. "
        "Use action 'get' to view the current personality, 'update' to modify fields."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "update"],
                "description": "Operation to perform.",
            },
            "name": {
                "type": "string",
                "description": (
                    "Agent name (any name accepted). For 'update' only."
                ),
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
                "description": (
                    "Comma-separated list of expertise areas. For 'update' only."
                ),
            },
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str,
        name: str | None = None,
        description: str | None = None,
        background: str | None = None,
        expertise: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        from captain_claw.personality import (
            load_personality,
            personality_to_dict,
            save_personality,
        )

        try:
            if action == "get":
                return self._get(load_personality())
            if action == "update":
                return self._update(
                    load_personality(),
                    name=name,
                    description=description,
                    background=background,
                    expertise=expertise,
                )
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("Personality tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))

    # ------------------------------------------------------------------

    @staticmethod
    def _get(p: Any) -> ToolResult:
        lines = [
            f"Name: {p.name}",
            f"Description: {p.description}",
            f"Background: {p.background}",
            f"Expertise: {', '.join(p.expertise)}",
        ]
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    def _update(
        p: Any,
        *,
        name: str | None,
        description: str | None,
        background: str | None,
        expertise: str | None,
    ) -> ToolResult:
        from captain_claw.personality import save_personality

        changed: list[str] = []

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

        if not changed:
            return ToolResult(
                success=False, error="No fields provided to update."
            )

        save_personality(p)

        return ToolResult(
            success=True,
            content=(
                f"Personality updated ({', '.join(changed)}). "
                f"Name: {p.name}"
            ),
        )
