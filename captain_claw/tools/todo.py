"""Todo tool for cross-session persistent task management."""

from typing import Any

from captain_claw.logging import get_logger
from captain_claw.session import get_session_manager
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

_PRIORITY_ORDER = {"urgent": 0, "high": 1, "normal": 2, "low": 3}


class TodoTool(Tool):
    """Manage persistent cross-session to-do items."""

    name = "todo"
    description = (
        "Manage a persistent cross-session to-do list. "
        "Items survive across sessions and can be assigned to the bot or human. "
        "Use action 'add' to create, 'list' to query, 'update' to change status/"
        "priority/responsible, and 'remove' to delete an item."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "update", "remove"],
                "description": "Operation to perform.",
            },
            "content": {
                "type": "string",
                "description": "Task description (required for 'add').",
            },
            "todo_id": {
                "type": "string",
                "description": "Todo ID or #index (for 'update' / 'remove').",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "done", "cancelled"],
                "description": "New status (for 'update').",
            },
            "responsible": {
                "type": "string",
                "enum": ["bot", "human"],
                "description": "Who is responsible (for 'add' / 'update').",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high", "urgent"],
                "description": "Priority level (for 'add' / 'update').",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags (for 'add' / 'update').",
            },
            "filter_status": {
                "type": "string",
                "description": "Filter by status (for 'list').",
            },
            "filter_responsible": {
                "type": "string",
                "description": "Filter by responsible party (for 'list').",
            },
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str,
        content: str | None = None,
        todo_id: str | None = None,
        status: str | None = None,
        responsible: str | None = None,
        priority: str | None = None,
        tags: str | None = None,
        filter_status: str | None = None,
        filter_responsible: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        session_id = str(kwargs.get("_session_id", "") or "").strip() or None
        sm = get_session_manager()

        try:
            if action == "add":
                return await self._add(
                    sm, content, responsible, priority, tags,
                    session_id, kwargs.get("_context"),
                )
            if action == "list":
                return await self._list(
                    sm, filter_status, filter_responsible, session_id,
                )
            if action == "update":
                return await self._update(
                    sm, todo_id, status, responsible, priority, content, tags,
                )
            if action == "remove":
                return await self._remove(sm, todo_id)
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("Todo tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))

    # ------------------------------------------------------------------

    @staticmethod
    async def _add(
        sm: Any,
        content: str | None,
        responsible: str | None,
        priority: str | None,
        tags: str | None,
        session_id: str | None,
        context: str | None,
    ) -> ToolResult:
        if not content or not content.strip():
            return ToolResult(success=False, error="'content' is required for add.")
        item = await sm.create_todo(
            content=content.strip(),
            responsible=responsible or "bot",
            priority=priority or "normal",
            source_session=session_id,
            context=context,
            tags=tags,
        )
        return ToolResult(
            success=True,
            content=f"Created todo #{item.id[:8]}: {item.content}",
        )

    @staticmethod
    async def _list(
        sm: Any,
        filter_status: str | None,
        filter_responsible: str | None,
        session_id: str | None,
    ) -> ToolResult:
        items = await sm.list_todos(
            limit=50,
            status_filter=filter_status or None,
            responsible_filter=filter_responsible or None,
            session_filter=session_id,
        )
        if not items:
            return ToolResult(success=True, content="No to-do items found.")
        lines: list[str] = []
        for idx, item in enumerate(items, 1):
            tag_suffix = f" [{item.tags}]" if item.tags else ""
            lines.append(
                f"#{idx} [{item.priority}/{item.responsible}] "
                f"{item.content} ({item.status}){tag_suffix}  id={item.id[:8]}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _update(
        sm: Any,
        todo_id: str | None,
        status: str | None,
        responsible: str | None,
        priority: str | None,
        content: str | None,
        tags: str | None,
    ) -> ToolResult:
        if not todo_id:
            return ToolResult(success=False, error="'todo_id' is required for update.")
        item = await sm.select_todo(todo_id)
        if not item:
            return ToolResult(success=False, error=f"Todo not found: {todo_id}")
        ok = await sm.update_todo(
            item.id,
            status=status,
            responsible=responsible,
            priority=priority,
            content=content,
            tags=tags,
        )
        if not ok:
            return ToolResult(success=False, error="Update failed.")
        return ToolResult(success=True, content=f"Updated todo #{item.id[:8]}.")

    @staticmethod
    async def _remove(sm: Any, todo_id: str | None) -> ToolResult:
        if not todo_id:
            return ToolResult(success=False, error="'todo_id' is required for remove.")
        item = await sm.select_todo(todo_id)
        if not item:
            return ToolResult(success=False, error=f"Todo not found: {todo_id}")
        ok = await sm.delete_todo(item.id)
        if not ok:
            return ToolResult(success=False, error="Delete failed.")
        return ToolResult(success=True, content=f"Removed todo #{item.id[:8]}.")
