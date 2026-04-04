"""Shared workspace tools — read/write data in the orchestration workspace.

These tools are only available during orchestration runs. They allow
worker agents to explicitly read data produced by upstream tasks and
write structured data for downstream tasks to consume.
"""

from __future__ import annotations

import json
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class WorkspaceReadTool(Tool):
    """Read a value from the shared orchestration workspace."""

    name = "workspace_read"
    description = (
        "Read a value from the shared workspace. "
        "The workspace contains structured data written by upstream tasks "
        "during the current orchestration run. "
        "Use the fully-qualified key (e.g. 'task_1:api_spec') or just "
        "the short key if the namespace is unambiguous. "
        "Use action 'list' to see all available keys."
    )
    timeout_seconds = 5.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "list"],
                "description": (
                    "'read' to get a specific value, "
                    "'list' to see all available keys."
                ),
            },
            "key": {
                "type": "string",
                "description": (
                    "The workspace key to read. "
                    "Can be fully-qualified ('task_1:api_spec') or short ('api_spec'). "
                    "Required for 'read' action."
                ),
            },
            "namespace": {
                "type": "string",
                "description": "Optional namespace filter (usually a task ID).",
            },
        },
        "required": ["action"],
    }

    async def execute(  # type: ignore[override]
        self,
        action: str = "read",
        key: str = "",
        namespace: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        workspace = kwargs.get("_shared_workspace")
        if workspace is None:
            return ToolResult(
                success=False,
                error=(
                    "Shared workspace is not available. "
                    "This tool is only usable during orchestration runs."
                ),
            )

        action = (action or "read").strip().lower()

        if action == "list":
            keys = workspace.list_keys(namespace=namespace)
            if not keys:
                return ToolResult(
                    success=True,
                    content="Workspace is empty — no keys have been written yet.",
                )
            lines = [f"Workspace contains {len(keys)} key(s):\n"]
            for k in keys:
                entry = workspace.read_entry(k)
                if entry:
                    lines.append(
                        f"  - {k}  ({entry.content_type}, "
                        f"from task {entry.task_id})"
                    )
                else:
                    lines.append(f"  - {k}")
            return ToolResult(success=True, content="\n".join(lines))

        if action == "read":
            if not key:
                return ToolResult(
                    success=False,
                    error="'key' is required for the 'read' action.",
                )
            entry = workspace.read_entry(key, namespace=namespace)
            if entry is None:
                # Try fuzzy match: search for keys ending with the requested key
                all_keys = workspace.list_keys()
                matches = [k for k in all_keys if k.endswith(f":{key}") or k == key]
                if len(matches) == 1:
                    entry = workspace.read_entry(matches[0])
                elif len(matches) > 1:
                    return ToolResult(
                        success=False,
                        error=(
                            f"Key '{key}' is ambiguous. Matches: {matches}. "
                            "Use a fully-qualified key."
                        ),
                    )
                else:
                    return ToolResult(
                        success=False,
                        error=(
                            f"Key '{key}' not found in workspace. "
                            f"Available keys: {workspace.list_keys()}"
                        ),
                    )

            # Format value for display
            value = entry.value  # type: ignore[union-attr]
            if isinstance(value, (dict, list)):
                try:
                    content = json.dumps(value, indent=2, default=str)
                except Exception:
                    content = str(value)
            else:
                content = str(value)

            return ToolResult(
                success=True,
                content=content,
                system_hint=f"From task {entry.task_id}, type: {entry.content_type}",  # type: ignore[union-attr]
            )

        return ToolResult(success=False, error=f"Unknown action: {action}")


class WorkspaceWriteTool(Tool):
    """Write a value to the shared orchestration workspace."""

    name = "workspace_write"
    description = (
        "Write structured data to the shared workspace so downstream "
        "tasks can access it. Use this to pass computed results, "
        "specifications, or any structured data between tasks. "
        "Data is attributed to the current task automatically."
    )
    timeout_seconds = 5.0
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "A descriptive key for the data (e.g. 'api_spec', "
                    "'test_results', 'review_feedback'). "
                    "Will be namespaced under the current task ID."
                ),
            },
            "value": {
                "type": "string",
                "description": (
                    "The data to store. Can be plain text, JSON, "
                    "or any string representation."
                ),
            },
            "content_type": {
                "type": "string",
                "enum": ["text", "json"],
                "description": (
                    "Type of the value: 'text' for plain text, "
                    "'json' for structured data. Defaults to 'text'."
                ),
            },
        },
        "required": ["key", "value"],
    }

    async def execute(  # type: ignore[override]
        self,
        key: str = "",
        value: str = "",
        content_type: str = "text",
        **kwargs: Any,
    ) -> ToolResult:
        workspace = kwargs.get("_shared_workspace")
        if workspace is None:
            return ToolResult(
                success=False,
                error=(
                    "Shared workspace is not available. "
                    "This tool is only usable during orchestration runs."
                ),
            )

        if not key:
            return ToolResult(success=False, error="'key' is required.")
        if not value:
            return ToolResult(success=False, error="'value' is required.")

        # Resolve task context from injected kwargs
        task_id = kwargs.get("_workspace_task_id", "")
        session_id = kwargs.get("_session_id", "")

        if not task_id:
            return ToolResult(
                success=False,
                error="Cannot determine current task ID for attribution.",
            )

        # If content_type is json, try to parse for validation
        stored_value: Any = value
        if content_type == "json":
            try:
                stored_value = json.loads(value)
            except json.JSONDecodeError as exc:
                return ToolResult(
                    success=False,
                    error=f"Invalid JSON: {exc}",
                )

        fqkey = workspace.write(
            key,
            stored_value,
            task_id=task_id,
            session_id=session_id,
            content_type=content_type,
        )

        return ToolResult(
            success=True,
            content=f"Written to workspace as '{fqkey}'.",
            system_hint=f"Downstream tasks can read this with workspace_read key='{fqkey}'.",
        )
