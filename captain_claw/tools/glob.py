"""Glob tool for finding files by pattern."""

import asyncio
import os
from pathlib import Path
from typing import Any

import glob

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class GlobTool(Tool):
    """Find files by pattern."""

    name = "glob"
    description = "Find files matching a glob pattern."
    timeout_seconds = 10.0  # glob is a local FS scan — 10 s is ample
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g., '**/*.py', 'src/**/*.ts')",
            },
            "root": {
                "type": "string",
                "description": "Root directory to search from (default: current directory)",
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of results (default: 100)",
            },
        },
        "required": ["pattern"],
    }

    async def execute(
        self,
        pattern: str,
        root: str | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Find files matching pattern.

        Args:
            pattern: Glob pattern
            root: Optional root directory
            limit: Max results

        Returns:
            ToolResult with matching files
        """
        try:
            # Set root — fall back to workspace base path when no explicit
            # root is given so that relative patterns like "pdf-test/**/*.pdf"
            # resolve against the workspace directory, not the process cwd.
            if not root:
                runtime_base = kwargs.get("_runtime_base_path")
                if runtime_base is not None:
                    root = str(runtime_base)
            if root:
                pattern = str(Path(root) / pattern)

            # Find files (sync, but run in executor to not block)
            loop = asyncio.get_event_loop()
            matches = await loop.run_in_executor(
                None,
                lambda: glob.glob(pattern, recursive=True)
            )

            # When running inside an orchestrator workflow, filter to
            # files created/modified after the workflow started.  This
            # prevents downstream tasks from picking up stale files with
            # matching names from earlier workflow runs.
            workflow_started_at: float | None = kwargs.get("_workflow_started_at")
            if workflow_started_at is not None:
                # Small buffer (2 s) to account for filesystem timestamp
                # granularity and minor clock drift.
                cutoff = workflow_started_at - 2.0
                before_count = len(matches)
                matches = [
                    m for m in matches
                    if _file_modified_after(m, cutoff)
                ]
                filtered_count = before_count - len(matches)
                if filtered_count > 0:
                    log.info(
                        "Glob workflow filter applied",
                        pattern=pattern,
                        before=before_count,
                        after=len(matches),
                        filtered=filtered_count,
                    )

            # Apply limit
            matches = matches[:limit]

            if not matches:
                return ToolResult(
                    success=True,
                    content=f"No files found matching: {pattern}",
                )

            # Return workspace-relative paths when the root was auto-resolved
            # to the workspace base.  This ensures paths like
            # "pdf-test/subfolder/file.pdf" are returned instead of long
            # absolute paths, keeping output concise and consistent with how
            # other tools accept relative paths.
            if root:
                root_prefix = str(Path(root).resolve()) + "/"
                matches = [
                    m[len(root_prefix):] if m.startswith(root_prefix) else m
                    for m in matches
                ]

            # Format output
            output = f"Found {len(matches)} file(s):\n"
            output += "\n".join(f"  {m}" for m in matches)

            return ToolResult(
                success=True,
                content=output,
            )

        except Exception as e:
            log.error("Glob failed", pattern=pattern, error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            )


def _file_modified_after(path: str, cutoff: float) -> bool:
    """Check if a file's mtime is at or after the cutoff timestamp."""
    try:
        return os.path.getmtime(path) >= cutoff
    except OSError:
        # File vanished or inaccessible — include it to be safe.
        return True
