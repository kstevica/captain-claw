"""Write tool for writing file contents."""

import asyncio
import re
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

# Late-import guard: FileRegistry is imported inside execute() to avoid
# circular imports when the module is loaded before the registry module.

log = get_logger(__name__)


class WriteTool(Tool):
    """Write content to files."""

    name = "write"
    description = "Create or overwrite a file with content."
    timeout_seconds = 10.0  # local file write — 10 s is ample
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
        """Map any requested path into the saved root and enforce session scoping.

        Paths under ``<workspace>/output/`` (the workspace root's ``output/``
        directory, parallel to ``saved/``) are passed through directly so that
        the scale micro-loop and other internal framework code can write to a
        well-known output directory outside the ``saved/`` hierarchy.
        """
        requested = Path(path).expanduser()

        # ── Passthrough for <workspace>/output/ paths ──
        # saved_root is typically <workspace>/saved.  The workspace root is
        # its parent.  If the requested absolute path falls under
        # <workspace>/output/ we allow it directly (still within the
        # workspace sandbox).
        if requested.is_absolute():
            absolute = requested.resolve()
            workspace_root = saved_root.parent
            output_root = (workspace_root / "output").resolve()
            try:
                absolute.relative_to(output_root)
                # Path is under <workspace>/output/ — allow it directly.
                return absolute
            except ValueError:
                pass

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

        categories = {"downloads", "media", "output", "scripts", "showcase", "skills", "tmp", "tools"}
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
            # Workflow-run override: bypass session scoping entirely.
            # Preserve the relative directory structure (e.g.
            # "backend/src/config/env.js") but strip absolute prefixes,
            # "../" traversals, and any "saved/<category>/<session_id>"
            # prefix the LLM may have injected.
            workflow_run_dir = kwargs.get("_workflow_run_dir")
            if workflow_run_dir is not None:
                requested = Path(path).expanduser()
                # Strip absolute root so we keep only the relative parts.
                if requested.is_absolute():
                    parts = list(requested.parts[1:])  # drop "/"
                else:
                    parts = list(requested.parts)
                # Remove ".." traversals for safety.
                parts = [p for p in parts if p not in ("", ".", "..")]
                # Strip any leading "saved/<category>/<session-id>" prefix
                # the LLM might have added from observed tool output.
                _categories = {"downloads", "media", "output", "scripts",
                               "showcase", "skills", "tmp", "tools", "saved"}
                while parts and parts[0].lower() in _categories:
                    parts = parts[1:]
                # Strip a UUID-shaped segment (session id) if it leads.
                if parts and len(parts[0]) >= 32 and parts[0].count("-") >= 4:
                    parts = parts[1:]
                if not parts:
                    parts = [Path(path).name or "output.txt"]
                file_path = Path(workflow_run_dir).joinpath(*parts)
            else:
                saved_root = self._resolve_saved_root(kwargs)
                session_id = self._normalize_session_id(str(kwargs.get("_session_id", "")))
                file_path = self._normalize_under_saved(path, saved_root, session_id)
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sanitize content: strip C0/C1 control characters that the LLM
            # may emit when it fails to reproduce Unicode (e.g. £→\x00a3,
            # €→\x01, '→\x02).  Preserve normal whitespace (\t \n \r).
            content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)

            # Write file
            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)

            redirect_note = ""
            requested = Path(path).expanduser()
            if str(requested) != str(file_path):
                redirect_note = f" (requested: {path})"

            # Register the logical -> physical mapping so other tasks /
            # later reads can find this file by its original requested path.
            file_registry = kwargs.get("_file_registry")
            if file_registry is not None:
                try:
                    file_registry.register(
                        logical_path=path,
                        physical_path=str(file_path),
                        task_id=str(kwargs.get("_task_id", "")),
                    )
                except Exception:
                    pass  # Best-effort; don't fail writes on registry errors

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
