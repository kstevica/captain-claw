"""Scripts tool for cross-session persistent script/file memory management."""

from typing import Any

from captain_claw.logging import get_logger
from captain_claw.session import get_session_manager
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class ScriptsTool(Tool):
    """Manage a persistent cross-session script/file memory."""

    name = "scripts"
    description = (
        "Manage a persistent cross-session script/file memory. "
        "Scripts survive across sessions and store file paths, languages, "
        "and accumulated context about why they were created. "
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
                "description": "Script name (required for 'add').",
            },
            "script_id": {
                "type": "string",
                "description": "Script ID, #index, or name (for 'info'/'update'/'remove').",
            },
            "file_path": {
                "type": "string",
                "description": "Relative file path to the script.",
            },
            "description": {
                "type": "string",
                "description": "Short description of the script.",
            },
            "purpose": {
                "type": "string",
                "description": "What the script does.",
            },
            "language": {
                "type": "string",
                "description": "Programming language (python, bash, javascript, etc.).",
            },
            "created_reason": {
                "type": "string",
                "description": "Why the script was created.",
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
        script_id: str | None = None,
        file_path: str | None = None,
        description: str | None = None,
        purpose: str | None = None,
        language: str | None = None,
        created_reason: str | None = None,
        tags: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        session_id = str(kwargs.get("_session_id", "") or "").strip() or None
        sm = get_session_manager()

        try:
            if action == "add":
                return await self._add(
                    sm, name, file_path, description, purpose,
                    language, created_reason, tags, session_id,
                )
            if action == "list":
                return await self._list(sm)
            if action == "search":
                return await self._search(sm, query)
            if action == "info":
                return await self._info(sm, script_id)
            if action == "update":
                return await self._update(
                    sm, script_id, name, file_path, description,
                    purpose, language, created_reason, tags,
                )
            if action == "remove":
                return await self._remove(sm, script_id)
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("Scripts tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))

    # ------------------------------------------------------------------

    @staticmethod
    async def _add(
        sm: Any,
        name: str | None,
        file_path: str | None,
        description: str | None,
        purpose: str | None,
        language: str | None,
        created_reason: str | None,
        tags: str | None,
        session_id: str | None,
    ) -> ToolResult:
        if not name or not name.strip():
            return ToolResult(success=False, error="'name' is required for add.")
        if not file_path or not file_path.strip():
            return ToolResult(success=False, error="'file_path' is required for add.")
        item = await sm.create_script(
            name=name.strip(),
            file_path=file_path.strip(),
            description=description,
            purpose=purpose,
            language=language,
            created_reason=created_reason,
            tags=tags,
            source_session=session_id,
        )
        return ToolResult(
            success=True,
            content=f"Registered script #{item.id[:8]}: {item.name} at {item.file_path}",
        )

    @staticmethod
    async def _list(sm: Any) -> ToolResult:
        items = await sm.list_scripts(limit=50)
        if not items:
            return ToolResult(success=True, content="No scripts found.")
        lines: list[str] = []
        for idx, s in enumerate(items, 1):
            lang_part = f" ({s.language})" if s.language else ""
            lines.append(
                f"#{idx} {s.name}{lang_part} [{s.file_path}]"
                f"  uses={s.use_count}  id={s.id[:8]}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _search(sm: Any, query: str | None) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(success=False, error="'query' is required for search.")
        items = await sm.search_scripts(query.strip(), limit=20)
        if not items:
            return ToolResult(success=True, content=f"No scripts matching: {query}")
        lines: list[str] = []
        for idx, s in enumerate(items, 1):
            lang_part = f" ({s.language})" if s.language else ""
            lines.append(
                f"#{idx} {s.name}{lang_part} [{s.file_path}]  id={s.id[:8]}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _info(sm: Any, script_id: str | None) -> ToolResult:
        if not script_id:
            return ToolResult(success=False, error="'script_id' is required for info.")
        item = await sm.select_script(script_id)
        if not item:
            return ToolResult(success=False, error=f"Script not found: {script_id}")
        parts = [f"Name: {item.name}", f"ID: {item.id}", f"Path: {item.file_path}"]
        if item.language:
            parts.append(f"Language: {item.language}")
        if item.description:
            parts.append(f"Description: {item.description}")
        if item.purpose:
            parts.append(f"Purpose: {item.purpose}")
        if item.created_reason:
            parts.append(f"Created reason: {item.created_reason}")
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
        script_id: str | None,
        name: str | None,
        file_path: str | None,
        description: str | None,
        purpose: str | None,
        language: str | None,
        created_reason: str | None,
        tags: str | None,
    ) -> ToolResult:
        if not script_id:
            return ToolResult(success=False, error="'script_id' is required for update.")
        item = await sm.select_script(script_id)
        if not item:
            return ToolResult(success=False, error=f"Script not found: {script_id}")
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if file_path is not None:
            kwargs["file_path"] = file_path
        if description is not None:
            kwargs["description"] = description
        if purpose is not None:
            kwargs["purpose"] = purpose
        if language is not None:
            kwargs["language"] = language
        if created_reason is not None:
            kwargs["created_reason"] = created_reason
        if tags is not None:
            kwargs["tags"] = tags
        if not kwargs:
            return ToolResult(success=False, error="No fields to update.")
        ok = await sm.update_script(item.id, **kwargs)
        if not ok:
            return ToolResult(success=False, error="Update failed.")
        return ToolResult(success=True, content=f"Updated script #{item.id[:8]}.")

    @staticmethod
    async def _remove(sm: Any, script_id: str | None) -> ToolResult:
        if not script_id:
            return ToolResult(success=False, error="'script_id' is required for remove.")
        item = await sm.select_script(script_id)
        if not item:
            return ToolResult(success=False, error=f"Script not found: {script_id}")
        ok = await sm.delete_script(item.id)
        if not ok:
            return ToolResult(success=False, error="Delete failed.")
        return ToolResult(success=True, content=f"Removed script #{item.id[:8]}: {item.name}")
