"""Read tool for reading file contents."""

import asyncio
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class ReadTool(Tool):
    """Read file contents."""

    name = "read"
    description = "Read the contents of a file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of lines to read",
            },
            "offset": {
                "type": "number",
                "description": "Line number to start reading from (1-indexed)",
            },
        },
        "required": ["path"],
    }

    async def execute(self, path: str, limit: int | None = None, offset: int | None = None, **kwargs: Any) -> ToolResult:
        """Read a file.
        
        Args:
            path: Path to file
            limit: Optional line limit
            offset: Optional line offset
        
        Returns:
            ToolResult with file contents
        """
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {path}",
                )
            
            if not file_path.is_file():
                return ToolResult(
                    success=False,
                    error=f"Not a file: {path}",
                )
            
            # Check if file is too large
            file_size = file_path.stat().st_size
            max_size = 100_000  # 100KB
            if file_size > max_size:
                return ToolResult(
                    success=False,
                    error=f"File too large: {file_size} bytes (max {max_size})",
                )
            
            # Read file
            content = file_path.read_text(encoding="utf-8")
            
            # Apply offset and limit
            lines = content.splitlines()
            if offset:
                lines = lines[offset - 1:]  # Convert to 0-indexed
            if limit:
                lines = lines[:limit]
            
            content = "\n".join(lines)
            
            # Add metadata
            info = f"[{file_path} {len(content)} chars]"
            if offset or limit:
                info += f" [lines {offset or 1}-{offset + limit if limit else len(lines)}]"
            
            return ToolResult(
                success=True,
                content=f"{info}\n{content}",
            )
            
        except Exception as e:
            log.error("Read failed", path=path, error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            )
