"""Write tool for writing file contents."""

import asyncio
import html
import re
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

# Heuristic markers that indicate a "status confirmation" message rather
# than substantive deliverable content. Used by the deliverable-protection
# guard below — leading markdown decoration (#, *, >, -, ✅) is stripped,
# then we look for any of the canonical "I'm done" phrasings.
_STATUS_CONFIRMATION_PATTERNS = re.compile(
    r"^[\s#*>\-]*(?:✅|✓|☑)?\s*(?:#+\s*)?"
    r"(?:"
    r"task\s+complete"
    r"|task\s+finished"
    r"|task\s+done"
    r"|task\s+accomplished"
    r"|task\s+successful"
    r"|completed\s+successfully"
    r"|done[\s\.\!\:]"
    r"|i(?:'ve|\s+have)\s+(?:successfully\s+)?(?:saved|written|created|finished|completed|synthes)"
    r"|i(?:'ll|\s+will)\s+now\s+confirm"
    r"|here(?:'s|\s+is)\s+(?:a\s+)?(?:summary|confirmation)"
    r"|summary\s+of\s+(?:what|the\s+work|completion)"
    r")",
    re.IGNORECASE,
)

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

        categories = {"downloads", "media", "output", "scripts", "showcase", "skills", "summaries", "tmp", "tools"}
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
                               "showcase", "skills", "summaries", "tmp", "tools", "saved"}
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

            # Detect overwrite — track pre-existing file size so we can
            # warn the LLM that it's overwriting (and suggest edit instead).
            #
            # Deliverable-protection guard: when the new content is much
            # shorter than what's already on disk AND looks like a status
            # confirmation ("Task Complete ✅", "Done!", "I've successfully
            # saved...", etc.), refuse the write. This catches the failure
            # mode where an LLM correctly saves a deliverable in one tool
            # call, then makes a second write_file call to "narrate" the
            # completion — destroying the actual deliverable.
            _overwrite_info: str | None = None
            if not append and file_path.exists():
                try:
                    _prev_size = file_path.stat().st_size
                    _prev_lines = file_path.read_text(encoding="utf-8", errors="replace").count("\n") + 1
                    new_size = len(content.encode("utf-8"))
                    # Only fire on substantive shrinks (50%+ reduction) and
                    # only when the previous file was meaningful (>= 1 KB).
                    # The pattern check confines us to obvious confirmation
                    # text — refinement rewrites that happen to be shorter
                    # do NOT match.
                    if (
                        _prev_size >= 1024
                        and new_size < _prev_size // 2
                        and _STATUS_CONFIRMATION_PATTERNS.match(content[:160] or "")
                    ):
                        log.warning(
                            "WriteTool refused deliverable overwrite",
                            path=str(file_path),
                            prev_size=_prev_size,
                            new_size=new_size,
                        )
                        return ToolResult(
                            success=False,
                            content=(
                                f"❌ Refused to overwrite {path}: the existing "
                                f"file is {_prev_size} bytes (your new content "
                                f"is only {new_size} bytes and starts with a "
                                f"completion-confirmation phrase). The file you "
                                f"already wrote IS your deliverable — narrate "
                                f"completion in your text response, NOT by "
                                f"writing back to the same file. If you really "
                                f"need to revise, use the edit_file tool for "
                                f"targeted changes."
                            ),
                            error="deliverable_overwrite_refused",
                        )
                    _overwrite_info = (
                        f"⚠️ Overwrote existing file (was {_prev_lines} lines, "
                        f"{_prev_size} bytes). Consider using the edit tool "
                        f"for targeted changes instead of rewriting the entire file."
                    )
                except Exception:
                    _overwrite_info = "⚠️ Overwrote existing file."

            # Sanitize content: strip C0/C1 control characters that the LLM
            # may emit when it fails to reproduce Unicode (e.g. £→\x00a3,
            # €→\x01, '→\x02).  Preserve normal whitespace (\t \n \r).
            content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)

            # Unescape HTML entities for markup files.  LLMs sometimes emit
            # &lt; / &gt; / &amp; / &quot; instead of literal < > & " when
            # generating HTML/SVG/XML, which produces broken output.
            suffix = file_path.suffix.lower()
            if suffix in ('.html', '.htm', '.svg', '.xml', '.xhtml'):
                content = html.unescape(content)

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

            _line_count = content.count("\n") + 1
            result_msg = f"Written {len(content)} chars ({_line_count} lines) to {file_path}{redirect_note}"
            if _overwrite_info:
                result_msg = f"{result_msg}\n{_overwrite_info}"
            return ToolResult(
                success=True,
                content=result_msg,
                # Hint: prevent read-after-write waste (LLM sometimes reads
                # back a file it just wrote, wasting an iteration).
                system_hint=(
                    "Do NOT read this file back — you already know its "
                    "contents. Proceed to the next file."
                ),
            )
            
        except Exception as e:
            log.error("Write failed", path=path, error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            )
