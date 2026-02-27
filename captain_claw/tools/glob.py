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
                "description": "Root directory to search from (default: workspace root)",
            },
            "scope": {
                "type": "string",
                "enum": ["workspace", "workflow"],
                "description": (
                    "Where to search. 'workspace' (default) searches pre-existing "
                    "user files in the workspace. 'workflow' searches ONLY the "
                    "workflow output directory where earlier tasks wrote files."
                ),
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
        scope: str | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Find files matching pattern.

        Args:
            pattern: Glob pattern
            root: Optional root directory
            scope: Where to search — ``"workspace"`` (default) for
                pre-existing user files, ``"workflow"`` for files created
                by earlier tasks in the current orchestration run.
            limit: Max results

        Returns:
            ToolResult with matching files
        """
        try:
            workflow_run_dir: str | None = kwargs.get("_workflow_run_dir")
            if isinstance(workflow_run_dir, Path):
                workflow_run_dir = str(workflow_run_dir)

            # Preserve the raw pattern before root is joined — used for
            # the cross-scope hint when the primary search finds nothing.
            raw_pattern = pattern

            # Resolve root based on scope.
            if scope == "workflow":
                if workflow_run_dir is not None:
                    root = workflow_run_dir
                else:
                    return ToolResult(
                        success=True,
                        content=(
                            "No workflow output directory available. "
                            "scope='workflow' is only valid during orchestrated "
                            "workflows. Use the default scope to search the workspace."
                        ),
                    )
            elif not root:
                # Default / "workspace" scope: search from workspace root.
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

            # When running inside an orchestrator workflow, apply the
            # timestamp filter ONLY to the workflow-run/ output directory
            # to avoid picking up stale outputs from earlier runs.
            # Pre-existing workspace files (user documents, config, etc.)
            # are INPUTS and must never be filtered out.
            workflow_started_at: float | None = kwargs.get("_workflow_started_at")
            if workflow_started_at is not None and workflow_run_dir is not None:
                wrd_prefix = str(Path(workflow_run_dir).resolve()) + "/"
                cutoff = workflow_started_at - 2.0
                before_count = len(matches)

                def _in_workflow_run(filepath: str) -> bool:
                    """Check if *filepath* is inside the workflow-run dir.

                    Uses resolve() to handle macOS /var → /private/var
                    and other symlink discrepancies.
                    """
                    try:
                        return str(Path(filepath).resolve()).startswith(wrd_prefix)
                    except (OSError, ValueError):
                        return False

                matches = [
                    m for m in matches
                    if (
                        # Only timestamp-filter files inside workflow-run/
                        not _in_workflow_run(m)
                        or _file_modified_after(m, cutoff)
                    )
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
                # Cross-scope hint: if we searched workspace and found
                # nothing, check whether the workflow output directory has
                # matches for the same pattern.  This helps the LLM
                # self-correct by retrying with scope='workflow'.
                hint = ""
                if scope != "workflow" and workflow_run_dir is not None:
                    wrd_pattern = str(Path(workflow_run_dir) / raw_pattern)
                    wrd_matches = await loop.run_in_executor(
                        None,
                        lambda p=wrd_pattern: glob.glob(p, recursive=True),
                    )
                    if wrd_matches:
                        hint = (
                            f"\n\nHint: {len(wrd_matches)} file(s) matching "
                            f"'{raw_pattern}' exist in the workflow output "
                            f"directory (files created by earlier tasks). "
                            f"Retry with scope='workflow' to find them."
                        )
                return ToolResult(
                    success=True,
                    content=f"No files found matching: {raw_pattern}{hint}",
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

            # Format output with scope-aware label.
            if scope == "workflow":
                output = f"Found {len(matches)} file(s) in workflow output:\n"
            else:
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
