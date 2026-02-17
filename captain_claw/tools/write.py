"""Write tool for writing file contents."""

import asyncio
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class WriteTool(Tool):
    """Write content to files."""

    name = "write"
    description = "Create or overwrite a file with content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
            "append": {
                "type": "boolean",
                "description": "Append to file instead of overwriting",
            },
        },
        "required": ["path", "content"],
    }

    async def execute(self, path: str, content: str, append: bool = False, **kwargs: Any) -> ToolResult:
        """Write content to a file.
        
        Args:
            path: Path to file
            content: Content to write
            append: Whether to append instead of overwrite
        
        Returns:
            ToolResult with status
        """
        try:
            file_path = Path(path).expanduser().resolve()
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                content=f"Written {len(content)} chars to {file_path}",
            )
            
        except Exception as e:
            log.error("Write failed", path=path, error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            )
