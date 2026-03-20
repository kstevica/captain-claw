"""Insights tool for searching and managing persistent cross-session insights."""

from typing import Any

from captain_claw.config import get_config
from captain_claw.insights import get_insights_manager, get_session_insights_manager
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


def _resolve_manager(session_id: str | None) -> Any:
    """Return session-scoped or global insights manager."""
    if session_id and get_config().web.public_run == "computer":
        return get_session_insights_manager(session_id)
    return get_insights_manager()


class InsightsTool(Tool):
    """Search and manage persistent cross-session insights."""

    name = "insights"
    description = (
        "Search and manage persistent cross-session insights — facts, contacts, "
        "decisions, preferences, and deadlines automatically extracted from "
        "conversations. Actions: search, list, add, update, delete."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "list", "add", "update", "delete"],
                "description": "Operation to perform.",
            },
            "query": {
                "type": "string",
                "description": "Search query text (for 'search').",
            },
            "category": {
                "type": "string",
                "enum": [
                    "contact", "decision", "preference", "fact",
                    "deadline", "project", "workflow",
                ],
                "description": "Category filter (for 'search'/'list') or value (for 'add').",
            },
            "content": {
                "type": "string",
                "description": "Insight text (for 'add'/'update').",
            },
            "insight_id": {
                "type": "string",
                "description": "Insight ID (for 'update'/'delete').",
            },
            "importance": {
                "type": "integer",
                "description": "Importance 1-10 (for 'add'/'update').",
            },
            "entity_key": {
                "type": "string",
                "description": "Dedup key e.g. 'contact:john@example.com' (for 'add').",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags (for 'add'/'update').",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (for 'search'/'list'). Default 10.",
            },
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str,
        query: str | None = None,
        category: str | None = None,
        content: str | None = None,
        insight_id: str | None = None,
        importance: int | None = None,
        entity_key: str | None = None,
        tags: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        session_id = str(kwargs.get("_session_id", "") or "").strip() or None
        mgr = _resolve_manager(session_id)

        try:
            if action == "search":
                return await self._search(mgr, query, category, limit)
            if action == "list":
                return await self._list(mgr, category, limit)
            if action == "add":
                return await self._add(
                    mgr, content, category, importance, entity_key, tags, session_id,
                )
            if action == "update":
                return await self._update(mgr, insight_id, content, category, importance, tags)
            if action == "delete":
                return await self._delete(mgr, insight_id)
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("Insights tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))

    # ── actions ──────────────────────────────────────────────────────

    @staticmethod
    async def _search(mgr: Any, query: str | None, category: str | None, limit: int | None) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(success=False, error="'query' is required for search.")
        results = await mgr.search(query.strip(), category=category, limit=limit or 10)
        if not results:
            return ToolResult(success=True, content="No matching insights found.")
        return ToolResult(success=True, content=_format_insights(results))

    @staticmethod
    async def _list(mgr: Any, category: str | None, limit: int | None) -> ToolResult:
        results = await mgr.list_recent(limit=limit or 10, category=category)
        if not results:
            return ToolResult(success=True, content="No insights stored yet.")
        total = await mgr.count()
        header = f"Insights ({len(results)} of {total}):\n"
        return ToolResult(success=True, content=header + _format_insights(results))

    @staticmethod
    async def _add(
        mgr: Any,
        content: str | None,
        category: str | None,
        importance: int | None,
        entity_key: str | None,
        tags: str | None,
        session_id: str | None,
    ) -> ToolResult:
        if not content or not content.strip():
            return ToolResult(success=False, error="'content' is required for add.")
        insight_id = await mgr.add(
            content=content.strip(),
            category=category or "fact",
            entity_key=entity_key,
            importance=importance or 5,
            source_tool="manual",
            source_session=session_id,
            tags=tags,
        )
        if insight_id:
            return ToolResult(success=True, content=f"Stored insight {insight_id}.")
        return ToolResult(success=True, content="Insight was deduped (similar one already exists).")

    @staticmethod
    async def _update(
        mgr: Any,
        insight_id: str | None,
        content: str | None,
        category: str | None,
        importance: int | None,
        tags: str | None,
    ) -> ToolResult:
        if not insight_id:
            return ToolResult(success=False, error="'insight_id' is required for update.")
        existing = await mgr.get(insight_id)
        if not existing:
            return ToolResult(success=False, error=f"Insight not found: {insight_id}")
        fields: dict[str, Any] = {}
        if content:
            fields["content"] = content.strip()
        if category:
            fields["category"] = category
        if importance is not None:
            fields["importance"] = importance
        if tags is not None:
            fields["tags"] = tags
        ok = await mgr.update(insight_id, **fields)
        if not ok:
            return ToolResult(success=False, error="Update failed.")
        return ToolResult(success=True, content=f"Updated insight {insight_id}.")

    @staticmethod
    async def _delete(mgr: Any, insight_id: str | None) -> ToolResult:
        if not insight_id:
            return ToolResult(success=False, error="'insight_id' is required for delete.")
        ok = await mgr.delete(insight_id)
        if not ok:
            return ToolResult(success=False, error=f"Insight not found: {insight_id}")
        return ToolResult(success=True, content=f"Deleted insight {insight_id}.")


def _format_insights(items: list[dict[str, Any]]) -> str:
    """Format a list of insight dicts for display."""
    lines: list[str] = []
    for i in items:
        imp = i.get("importance", 5)
        cat = i.get("category", "fact")
        tags = f" [{i['tags']}]" if i.get("tags") else ""
        lines.append(f"• [{cat}] (imp:{imp}) {i['content']}{tags}  id={i['id']}")
    return "\n".join(lines)
