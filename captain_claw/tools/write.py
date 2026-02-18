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

    @staticmethod
    def _normalize_session_id(raw: str | None) -> str:
        """Return filesystem-safe session identifier for path scoping."""
        value = (raw or "").strip()
        if not value:
            return "default"
        safe_parts = [c if c.isalnum() or c in "._-" else "-" for c in value]
        normalized = "".join(safe_parts).strip("-")
        return normalized or "default"

    @staticmethod
    def _resolve_saved_root(kwargs: dict[str, Any]) -> Path:
        """Resolve `<runtime_base>/saved` for tool-managed outputs."""
        saved_root_raw = kwargs.get("_saved_base_path")
        runtime_base_raw = kwargs.get("_runtime_base_path")
        if saved_root_raw is not None:
            saved_root = Path(saved_root_raw).expanduser().resolve()
        elif runtime_base_raw is not None:
            saved_root = (Path(runtime_base_raw).expanduser().resolve() / "saved").resolve()
        else:
            saved_root = (Path.cwd().resolve() / "saved").resolve()
        saved_root.mkdir(parents=True, exist_ok=True)
        return saved_root

    @staticmethod
    def _normalize_under_saved(path: str, saved_root: Path, session_id: str) -> Path:
        """Map any requested path into the saved root and enforce session scoping."""
        requested = Path(path).expanduser()
        if requested.is_absolute():
            absolute = requested.resolve()
            try:
                absolute.relative_to(saved_root)
                relative_hint = absolute.relative_to(saved_root)
            except ValueError:
                relative_hint = Path(*absolute.parts[1:])
        else:
            relative_hint = requested

        safe_parts = [part for part in relative_hint.parts if part not in ("", ".", "..")]
        if not safe_parts:
            safe_parts = ["output.txt"]
        # Accept "saved/<category>/..." inputs and normalize from category root.
        if safe_parts and safe_parts[0].lower() == "saved":
            safe_parts = safe_parts[1:] or ["output.txt"]

        categories = {"downloads", "media", "scripts", "showcase", "skills", "tmp", "tools"}
        scoped_parts: list[str]
        if safe_parts[0] in categories:
            if len(safe_parts) >= 2 and safe_parts[1] == session_id:
                scoped_parts = safe_parts
            else:
                scoped_parts = [safe_parts[0], session_id, *safe_parts[1:]]
        else:
            scoped_parts = ["tmp", session_id, *safe_parts]

        candidate = (saved_root.joinpath(*scoped_parts)).resolve()
        try:
            candidate.relative_to(saved_root)
            return candidate
        except ValueError:
            fallback = (saved_root / "tmp" / session_id / safe_parts[-1]).resolve()
            return fallback

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
            saved_root = self._resolve_saved_root(kwargs)
            session_id = self._normalize_session_id(str(kwargs.get("_session_id", "")))
            file_path = self._normalize_under_saved(path, saved_root, session_id)
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)

            redirect_note = ""
            requested = Path(path).expanduser()
            if str(requested) != str(file_path):
                redirect_note = f" (requested: {path})"
            
            return ToolResult(
                success=True,
                content=f"Written {len(content)} chars to {file_path}{redirect_note}",
            )
            
        except Exception as e:
            log.error("Write failed", path=path, error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            )
