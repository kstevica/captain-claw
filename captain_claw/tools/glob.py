"""Glob tool for finding files by pattern."""

import asyncio
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
            # Set root
            if root:
                pattern = str(Path(root) / pattern)
            
            # Find files (sync, but run in executor to not block)
            loop = asyncio.get_event_loop()
            matches = await loop.run_in_executor(
                None,
                lambda: glob.glob(pattern, recursive=True)
            )
            
            # Apply limit
            matches = matches[:limit]
            
            if not matches:
                return ToolResult(
                    success=True,
                    content=f"No files found matching: {pattern}",
                )
            
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
