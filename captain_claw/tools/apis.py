"""APIs tool for cross-session persistent API memory management."""

from typing import Any

from captain_claw.logging import get_logger
from captain_claw.session import get_session_manager
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class ApisTool(Tool):
    """Manage a persistent cross-session API memory."""

    name = "apis"
    description = (
        "Manage a persistent cross-session API memory. "
        "APIs survive across sessions and store base URLs, endpoints, "
        "authentication details, credentials, and accumulated context. "
        "Use action 'add' to register, 'list' to query, 'search' to find, "
        "'info' for details, 'update' to modify, and 'remove' to delete."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "search", "info", "update", "remove"],
                "description": "Operation to perform.",
            },
            "name": {
                "type": "string",
                "description": "API name (required for 'add').",
            },
            "api_id": {
                "type": "string",
                "description": "API ID, #index, or name (for 'info'/'update'/'remove').",
            },
            "base_url": {
                "type": "string",
                "description": "Base URL of the API (required for 'add').",
            },
            "endpoints": {
                "type": "string",
                "description": "JSON list of endpoint definitions [{method, path, description}].",
            },
            "auth_type": {
                "type": "string",
                "enum": ["bearer", "api_key", "basic", "none"],
                "description": "Authentication type.",
            },
            "credentials": {
                "type": "string",
                "description": "Authentication credentials (plaintext).",
            },
            "description": {
                "type": "string",
                "description": "Short description of the API.",
            },
            "purpose": {
                "type": "string",
                "description": "What this API is used for.",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags.",
            },
            "query": {
                "type": "string",
                "description": "Search query (for 'search').",
            },
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str,
        name: str | None = None,
        api_id: str | None = None,
        base_url: str | None = None,
        endpoints: str | None = None,
        auth_type: str | None = None,
        credentials: str | None = None,
        description: str | None = None,
        purpose: str | None = None,
        tags: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        session_id = str(kwargs.get("_session_id", "") or "").strip() or None
        sm = get_session_manager()

        try:
            if action == "add":
                return await self._add(
                    sm, name, base_url, endpoints, auth_type,
                    credentials, description, purpose, tags, session_id,
                )
            if action == "list":
                return await self._list(sm)
            if action == "search":
                return await self._search(sm, query)
            if action == "info":
                return await self._info(sm, api_id)
            if action == "update":
                return await self._update(
                    sm, api_id, name, base_url, endpoints, auth_type,
                    credentials, description, purpose, tags,
                )
            if action == "remove":
                return await self._remove(sm, api_id)
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("APIs tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))

    # ------------------------------------------------------------------

    @staticmethod
    async def _add(
        sm: Any,
        name: str | None,
        base_url: str | None,
        endpoints: str | None,
        auth_type: str | None,
        credentials: str | None,
        description: str | None,
        purpose: str | None,
        tags: str | None,
        session_id: str | None,
    ) -> ToolResult:
        if not name or not name.strip():
            return ToolResult(success=False, error="'name' is required for add.")
        if not base_url or not base_url.strip():
            return ToolResult(success=False, error="'base_url' is required for add.")
        item = await sm.create_api(
            name=name.strip(),
            base_url=base_url.strip(),
            endpoints=endpoints,
            auth_type=auth_type,
            credentials=credentials,
            description=description,
            purpose=purpose,
            tags=tags,
            source_session=session_id,
        )
        return ToolResult(
            success=True,
            content=f"Registered API #{item.id[:8]}: {item.name} ({item.base_url})",
        )

    @staticmethod
    async def _list(sm: Any) -> ToolResult:
        items = await sm.list_apis(limit=50)
        if not items:
            return ToolResult(success=True, content="No APIs found.")
        lines: list[str] = []
        for idx, a in enumerate(items, 1):
            auth_part = f" [{a.auth_type}]" if a.auth_type else ""
            lines.append(
                f"#{idx} {a.name}{auth_part} ({a.base_url})"
                f"  uses={a.use_count}  id={a.id[:8]}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _search(sm: Any, query: str | None) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(success=False, error="'query' is required for search.")
        items = await sm.search_apis(query.strip(), limit=20)
        if not items:
            return ToolResult(success=True, content=f"No APIs matching: {query}")
        lines: list[str] = []
        for idx, a in enumerate(items, 1):
            auth_part = f" [{a.auth_type}]" if a.auth_type else ""
            lines.append(
                f"#{idx} {a.name}{auth_part} ({a.base_url})  id={a.id[:8]}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _info(sm: Any, api_id: str | None) -> ToolResult:
        if not api_id:
            return ToolResult(success=False, error="'api_id' is required for info.")
        item = await sm.select_api(api_id)
        if not item:
            return ToolResult(success=False, error=f"API not found: {api_id}")
        parts = [f"Name: {item.name}", f"ID: {item.id}", f"Base URL: {item.base_url}"]
        if item.auth_type:
            parts.append(f"Auth type: {item.auth_type}")
        if item.credentials:
            parts.append(f"Credentials: {item.credentials}")
        if item.endpoints:
            parts.append(f"Endpoints: {item.endpoints}")
        if item.description:
            parts.append(f"Description: {item.description}")
        if item.purpose:
            parts.append(f"Purpose: {item.purpose}")
        if item.tags:
            parts.append(f"Tags: {item.tags}")
        parts.append(f"Uses: {item.use_count}")
        if item.last_used_at:
            parts.append(f"Last used: {item.last_used_at}")
        parts.append(f"Created: {item.created_at}")
        return ToolResult(success=True, content="\n".join(parts))

    @staticmethod
    async def _update(
        sm: Any,
        api_id: str | None,
        name: str | None,
        base_url: str | None,
        endpoints: str | None,
        auth_type: str | None,
        credentials: str | None,
        description: str | None,
        purpose: str | None,
        tags: str | None,
    ) -> ToolResult:
        if not api_id:
            return ToolResult(success=False, error="'api_id' is required for update.")
        item = await sm.select_api(api_id)
        if not item:
            return ToolResult(success=False, error=f"API not found: {api_id}")
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if base_url is not None:
            kwargs["base_url"] = base_url
        if endpoints is not None:
            kwargs["endpoints"] = endpoints
        if auth_type is not None:
            kwargs["auth_type"] = auth_type
        if credentials is not None:
            kwargs["credentials"] = credentials
        if description is not None:
            kwargs["description"] = description
        if purpose is not None:
            kwargs["purpose"] = purpose
        if tags is not None:
            kwargs["tags"] = tags
        if not kwargs:
            return ToolResult(success=False, error="No fields to update.")
        ok = await sm.update_api(item.id, **kwargs)
        if not ok:
            return ToolResult(success=False, error="Update failed.")
        return ToolResult(success=True, content=f"Updated API #{item.id[:8]}.")

    @staticmethod
    async def _remove(sm: Any, api_id: str | None) -> ToolResult:
        if not api_id:
            return ToolResult(success=False, error="'api_id' is required for remove.")
        item = await sm.select_api(api_id)
        if not item:
            return ToolResult(success=False, error=f"API not found: {api_id}")
        ok = await sm.delete_api(item.id)
        if not ok:
            return ToolResult(success=False, error="Delete failed.")
        return ToolResult(success=True, content=f"Removed API #{item.id[:8]}: {item.name}")
