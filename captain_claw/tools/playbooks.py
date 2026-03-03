"""Playbooks tool for cross-session persistent orchestration pattern memory."""

from typing import Any

from captain_claw.logging import get_logger
from captain_claw.session import get_session_manager
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# Valid task type classifications for playbooks.
PLAYBOOK_TASK_TYPES = (
    "batch-processing",
    "web-research",
    "code-generation",
    "document-processing",
    "data-transformation",
    "orchestration",
    "interactive",
    "file-management",
    "other",
)


class PlaybooksTool(Tool):
    """Manage persistent cross-session orchestration playbooks.

    Playbooks capture proven patterns (do/don't pseudo-code) for recurring
    task types.  They survive across sessions and are injected into the
    planning context when a similar task is detected.

    Actions:
        add      – register a new playbook
        list     – show all playbooks (optionally filtered by task_type)
        search   – find playbooks by keyword
        info     – show full details of one playbook
        update   – modify an existing playbook
        remove   – delete a playbook
    """

    name = "playbooks"
    description = (
        "Manage persistent cross-session orchestration playbooks. "
        "Playbooks capture proven do/don't patterns for recurring task types "
        "(batch-processing, web-research, code-generation, document-processing, "
        "data-transformation, orchestration, interactive, file-management, other). "
        "Use action 'add' to register, 'list' to query, 'search' to find, "
        "'info' for details, 'update' to modify, and 'remove' to delete."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "search", "info", "update", "remove", "rate"],
                "description": "Operation to perform. Use 'rate' to rate the current session and propose a playbook.",
            },
            "name": {
                "type": "string",
                "description": "Playbook name (required for 'add').",
            },
            "playbook_id": {
                "type": "string",
                "description": "Playbook ID, #index, or name (for 'info'/'update'/'remove').",
            },
            "task_type": {
                "type": "string",
                "enum": list(PLAYBOOK_TASK_TYPES),
                "description": "Task type classification.",
            },
            "rating": {
                "type": "string",
                "enum": ["good", "bad"],
                "description": "Whether this pattern is a positive or negative example.",
            },
            "do_pattern": {
                "type": "string",
                "description": "Pseudo-code of what works well (the recommended approach).",
            },
            "dont_pattern": {
                "type": "string",
                "description": "Pseudo-code of what to avoid (the anti-pattern).",
            },
            "trigger_description": {
                "type": "string",
                "description": "Natural language description of when this playbook should activate.",
            },
            "reasoning": {
                "type": "string",
                "description": "Why this pattern matters.",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags.",
            },
            "query": {
                "type": "string",
                "description": "Search query (for 'search').",
            },
            "session_id": {
                "type": "string",
                "description": "Session ID to rate (for 'rate'). Defaults to current session.",
            },
            "note": {
                "type": "string",
                "description": "Optional user note explaining the rating (for 'rate').",
            },
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str,
        name: str | None = None,
        playbook_id: str | None = None,
        task_type: str | None = None,
        rating: str | None = None,
        do_pattern: str | None = None,
        dont_pattern: str | None = None,
        trigger_description: str | None = None,
        reasoning: str | None = None,
        tags: str | None = None,
        query: str | None = None,
        session_id: str | None = None,
        note: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        _session_id = session_id or str(kwargs.get("_session_id", "") or "").strip() or None
        sm = get_session_manager()

        try:
            if action == "add":
                return await self._add(
                    sm, name, task_type, rating, do_pattern, dont_pattern,
                    trigger_description, reasoning, tags, _session_id,
                )
            if action == "list":
                return await self._list(sm, task_type)
            if action == "search":
                return await self._search(sm, query, task_type)
            if action == "info":
                return await self._info(sm, playbook_id)
            if action == "update":
                return await self._update(
                    sm, playbook_id, name, task_type, rating, do_pattern,
                    dont_pattern, trigger_description, reasoning, tags,
                )
            if action == "remove":
                return await self._remove(sm, playbook_id)
            if action == "rate":
                return await self._rate(_session_id, rating, note)
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("Playbooks tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))

    # ------------------------------------------------------------------

    @staticmethod
    async def _add(
        sm: Any,
        name: str | None,
        task_type: str | None,
        rating: str | None,
        do_pattern: str | None,
        dont_pattern: str | None,
        trigger_description: str | None,
        reasoning: str | None,
        tags: str | None,
        session_id: str | None,
    ) -> ToolResult:
        if not name or not name.strip():
            return ToolResult(success=False, error="'name' is required for add.")
        if not task_type or not task_type.strip():
            return ToolResult(success=False, error="'task_type' is required for add.")
        if task_type not in PLAYBOOK_TASK_TYPES:
            return ToolResult(
                success=False,
                error=f"Invalid task_type '{task_type}'. Must be one of: {', '.join(PLAYBOOK_TASK_TYPES)}",
            )
        if not do_pattern and not dont_pattern:
            return ToolResult(
                success=False,
                error="At least one of 'do_pattern' or 'dont_pattern' is required.",
            )
        item = await sm.create_playbook(
            name=name.strip(),
            task_type=task_type.strip(),
            rating=rating or "good",
            do_pattern=(do_pattern or "").strip(),
            dont_pattern=(dont_pattern or "").strip(),
            trigger_description=(trigger_description or "").strip(),
            reasoning=reasoning,
            tags=tags,
            source_session=session_id,
        )
        return ToolResult(
            success=True,
            content=(
                f"Registered playbook #{item.id[:8]}: {item.name}\n"
                f"  type={item.task_type}  rating={item.rating}"
            ),
        )

    @staticmethod
    async def _list(sm: Any, task_type: str | None) -> ToolResult:
        items = await sm.list_playbooks(limit=50, task_type=task_type)
        if not items:
            label = f" for type '{task_type}'" if task_type else ""
            return ToolResult(success=True, content=f"No playbooks found{label}.")
        lines: list[str] = []
        for idx, p in enumerate(items, 1):
            lines.append(
                f"#{idx} [{p.task_type}] {p.name} ({p.rating})"
                f"  uses={p.use_count}  id={p.id[:8]}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _search(sm: Any, query: str | None, task_type: str | None) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(success=False, error="'query' is required for search.")
        items = await sm.search_playbooks(query.strip(), limit=20, task_type=task_type)
        if not items:
            return ToolResult(success=True, content=f"No playbooks matching: {query}")
        lines: list[str] = []
        for idx, p in enumerate(items, 1):
            lines.append(
                f"#{idx} [{p.task_type}] {p.name} ({p.rating})  id={p.id[:8]}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _info(sm: Any, playbook_id: str | None) -> ToolResult:
        if not playbook_id:
            return ToolResult(success=False, error="'playbook_id' is required for info.")
        item = await sm.select_playbook(playbook_id)
        if not item:
            return ToolResult(success=False, error=f"Playbook not found: {playbook_id}")
        parts = [
            f"Name: {item.name}",
            f"ID: {item.id}",
            f"Task type: {item.task_type}",
            f"Rating: {item.rating}",
        ]
        if item.trigger_description:
            parts.append(f"Trigger: {item.trigger_description}")
        if item.do_pattern:
            parts.append(f"\n--- DO (recommended) ---\n{item.do_pattern}")
        if item.dont_pattern:
            parts.append(f"\n--- DON'T (avoid) ---\n{item.dont_pattern}")
        if item.reasoning:
            parts.append(f"\nReasoning: {item.reasoning}")
        if item.tags:
            parts.append(f"Tags: {item.tags}")
        parts.append(f"Uses: {item.use_count}")
        if item.last_used_at:
            parts.append(f"Last used: {item.last_used_at}")
        parts.append(f"Created: {item.created_at}")
        if item.source_session:
            parts.append(f"Source session: {item.source_session}")
        return ToolResult(success=True, content="\n".join(parts))

    @staticmethod
    async def _update(
        sm: Any,
        playbook_id: str | None,
        name: str | None,
        task_type: str | None,
        rating: str | None,
        do_pattern: str | None,
        dont_pattern: str | None,
        trigger_description: str | None,
        reasoning: str | None,
        tags: str | None,
    ) -> ToolResult:
        if not playbook_id:
            return ToolResult(success=False, error="'playbook_id' is required for update.")
        item = await sm.select_playbook(playbook_id)
        if not item:
            return ToolResult(success=False, error=f"Playbook not found: {playbook_id}")
        if task_type and task_type not in PLAYBOOK_TASK_TYPES:
            return ToolResult(
                success=False,
                error=f"Invalid task_type '{task_type}'. Must be one of: {', '.join(PLAYBOOK_TASK_TYPES)}",
            )
        fields: dict[str, Any] = {}
        if name is not None:
            fields["name"] = name
        if task_type is not None:
            fields["task_type"] = task_type
        if rating is not None:
            fields["rating"] = rating
        if do_pattern is not None:
            fields["do_pattern"] = do_pattern
        if dont_pattern is not None:
            fields["dont_pattern"] = dont_pattern
        if trigger_description is not None:
            fields["trigger_description"] = trigger_description
        if reasoning is not None:
            fields["reasoning"] = reasoning
        if tags is not None:
            fields["tags"] = tags
        if not fields:
            return ToolResult(success=False, error="No fields to update.")
        ok = await sm.update_playbook(item.id, **fields)
        if not ok:
            return ToolResult(success=False, error="Update failed.")
        return ToolResult(success=True, content=f"Updated playbook #{item.id[:8]}: {item.name}")

    @staticmethod
    async def _remove(sm: Any, playbook_id: str | None) -> ToolResult:
        if not playbook_id:
            return ToolResult(success=False, error="'playbook_id' is required for remove.")
        item = await sm.select_playbook(playbook_id)
        if not item:
            return ToolResult(success=False, error=f"Playbook not found: {playbook_id}")
        ok = await sm.delete_playbook(item.id)
        if not ok:
            return ToolResult(success=False, error="Delete failed.")
        return ToolResult(success=True, content=f"Removed playbook #{item.id[:8]}: {item.name}")

    @staticmethod
    async def _rate(
        session_id: str | None,
        rating: str | None,
        note: str | None,
    ) -> ToolResult:
        if not rating or rating not in ("good", "bad"):
            return ToolResult(
                success=False,
                error="'rating' is required for rate and must be 'good' or 'bad'.",
            )

        sm = get_session_manager()

        # Store rating in session metadata.
        if session_id:
            session = await sm.load_session(session_id)
            if session:
                session.metadata["playbook_rating"] = rating
                if note:
                    session.metadata["playbook_rating_note"] = note
                await sm.save_session(session)

        # Run distillation via standalone LLM call.
        proposal = await _distill_session_standalone(sm, session_id, rating, note or "")

        if proposal:
            # Save the proposed playbook.
            entry = await sm.create_playbook(
                name=str(proposal.get("name", "Unnamed playbook")),
                task_type=str(proposal.get("task_type", "other")),
                rating=rating,
                do_pattern=str(proposal.get("do_pattern", "")),
                dont_pattern=str(proposal.get("dont_pattern", "")),
                trigger_description=str(proposal.get("trigger_description", "")),
                reasoning=str(proposal.get("reasoning", "")),
                source_session=session_id,
            )
            parts = [
                f"Session rated as '{rating}'. Playbook distilled and saved:",
                f"  Name: {entry.name}",
                f"  Type: {entry.task_type}",
                f"  ID: {entry.id[:8]}",
            ]
            if entry.trigger_description:
                parts.append(f"  Trigger: {entry.trigger_description}")
            if entry.do_pattern:
                parts.append(f"  DO: {entry.do_pattern[:200]}")
            if entry.dont_pattern:
                parts.append(f"  DON'T: {entry.dont_pattern[:200]}")
            parts.append(
                "\nUse playbooks(action='info', playbook_id='...') to see full details, "
                "or playbooks(action='update', ...) to refine it."
            )
            return ToolResult(success=True, content="\n".join(parts))

        return ToolResult(
            success=True,
            content=(
                f"Session rated as '{rating}'. Automatic distillation could not "
                "extract a pattern. You can manually create a playbook with action='add'."
            ),
        )


# ---------------------------------------------------------------------------
# Standalone distillation (no agent reference needed)
# ---------------------------------------------------------------------------

async def _distill_session_standalone(
    sm: Any,
    session_id: str | None,
    rating: str,
    user_note: str,
) -> dict[str, Any] | None:
    """Run a standalone LLM distillation call on a session trace."""
    import json as _json

    if not session_id:
        return None

    session = await sm.load_session(session_id)
    if not session or not session.messages:
        return None

    from captain_claw.agent_playbook_mixin import (
        _extract_session_summary,
        _extract_tool_trace,
    )
    from captain_claw.instructions import InstructionLoader

    instructions = InstructionLoader()
    note_block = f"\nUser note: {user_note}" if user_note else ""

    system_content = instructions.render(
        "playbook_distill_system_prompt.md",
        rating=rating,
        user_note=note_block,
    )
    user_content = instructions.render(
        "playbook_distill_user_prompt.md",
        session_summary=_extract_session_summary(session.messages),
        tool_trace=_extract_tool_trace(session.messages),
    )

    try:
        from captain_claw.config import get_config
        from captain_claw.llm import LLMProvider, Message as LLMMessage

        cfg = get_config()
        provider = LLMProvider(cfg)
        messages = [
            LLMMessage(role="system", content=system_content),
            LLMMessage(role="user", content=user_content),
        ]
        response = await provider.complete(
            messages=messages,
            tools=None,
            max_tokens=min(2048, max(1, int(cfg.model.max_tokens))),
        )
        raw = response.content or ""
    except Exception as e:
        log.error("Playbook distillation LLM call failed", error=str(e))
        return None

    # Parse JSON from response.
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            payload = _json.loads(raw[start:end])
        else:
            payload = _json.loads(raw)
    except (_json.JSONDecodeError, TypeError):
        log.warning("Distill: could not parse JSON from LLM response")
        return None

    if not isinstance(payload, dict):
        return None

    # Validate required fields.
    required = {"task_type", "name"}
    if not required.issubset(set(payload.keys())):
        log.warning("Distill: missing required fields", keys=list(payload.keys()))
        return None

    # Ensure at least one pattern exists.
    if not payload.get("do_pattern") and not payload.get("dont_pattern"):
        log.warning("Distill: no do_pattern or dont_pattern in proposal")
        return None

    return payload
