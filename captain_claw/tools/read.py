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
    timeout_seconds = 10.0  # local file read — 10 s is ample
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
            raw_path = Path(path).expanduser()

            # Resolve relative paths against the workspace root (not the
            # process CWD) so that paths like "pdf-test/foo.pdf" resolve
            # consistently across all tools.
            if raw_path.is_absolute():
                file_path = raw_path.resolve()
            else:
                runtime_base = kwargs.get("_runtime_base_path")
                if runtime_base is not None:
                    file_path = (Path(runtime_base) / raw_path).resolve()
                else:
                    file_path = raw_path.resolve()

            if not file_path.exists():
                # Try workflow-run directory (orchestrated workflows write
                # files here preserving relative directory structure).
                workflow_run_dir = kwargs.get("_workflow_run_dir")
                if workflow_run_dir is not None:
                    wrd = Path(workflow_run_dir)
                    # 1. Try the path as-is under workflow-run dir.
                    _rel = Path(path).expanduser()
                    if _rel.is_absolute():
                        _rel = Path(*_rel.parts[1:])
                    candidate = (wrd / _rel).resolve()
                    if not candidate.exists():
                        # 2. Try just the filename (flat lookup).
                        candidate = wrd / Path(path).name
                    if candidate.exists():
                        file_path = candidate

            if not file_path.exists():
                # Attempt file registry resolution before giving up.
                file_registry = kwargs.get("_file_registry")
                resolved_path: str | None = None
                if file_registry is not None:
                    try:
                        resolved_path = file_registry.resolve(path)
                    except Exception:
                        pass
                if resolved_path is not None:
                    candidate = Path(resolved_path).expanduser().resolve()
                    if candidate.exists():
                        file_path = candidate
                    else:
                        return ToolResult(
                            success=False,
                            error=f"File not found: {path} (registry resolved to {resolved_path}, also missing)",
                        )
                else:
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
            from captain_claw.config import get_config
            max_size = get_config().tools.read.max_file_bytes
            if file_size > max_size:
                return ToolResult(
                    success=False,
                    error=f"File too large: {file_size} bytes (max {max_size})",
                )
            
            # Read file
            content = file_path.read_text(encoding="utf-8")

            # Apply offset and limit
            all_lines = content.splitlines()
            start_line = max(1, int(offset)) if offset is not None else 1
            selected_lines = all_lines[start_line - 1 :]
            if limit is not None:
                selected_lines = selected_lines[: max(0, int(limit))]

            content = "\n".join(selected_lines)

            # Add metadata
            info = f"[{file_path} {len(content)} chars]"
            if offset is not None or limit is not None:
                if selected_lines:
                    end_line = start_line + len(selected_lines) - 1
                else:
                    end_line = start_line - 1
                info += f" [lines {start_line}-{end_line}]"
            
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
