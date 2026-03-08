"""Edit tool for surgically modifying existing text files."""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

_ACTIONS = (
    "replace_string",
    "insert_after",
    "insert_before",
    "delete_string",
    "insert_at_line",
    "delete_lines",
    "replace_lines",
    "undo",
)

# Parameters required per action (beyond the always-required path + action).
_ACTION_REQUIRED: dict[str, list[str]] = {
    "replace_string": ["old_string", "new_string"],
    "insert_after": ["old_string", "new_string"],
    "insert_before": ["old_string", "new_string"],
    "delete_string": ["old_string"],
    "insert_at_line": ["start_line", "new_string"],
    "delete_lines": ["start_line", "end_line"],
    "replace_lines": ["start_line", "end_line", "new_string"],
    "undo": [],
}


class EditTool(Tool):
    """Edit an existing file with string-match or line-based operations."""

    name = "edit"
    description = (
        "Edit an existing text file. Actions (prefer string-match for accuracy): "
        "replace_string — find exact text and replace it; "
        "insert_after / insert_before — insert text relative to an anchor string; "
        "delete_string — remove exact text; "
        "insert_at_line — insert at line number; "
        "delete_lines / replace_lines — line-range operations; "
        "undo — restore most recent backup."
    )
    timeout_seconds = 15.0
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit (relative to workspace root or absolute)",
            },
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "The editing action to perform",
            },
            "old_string": {
                "type": "string",
                "description": (
                    "The exact string to find in the file. Must match exactly "
                    "one location. Include enough surrounding context (2-3 lines) "
                    "to ensure uniqueness. Used by: replace_string, insert_after, "
                    "insert_before, delete_string."
                ),
            },
            "new_string": {
                "type": "string",
                "description": (
                    "The replacement or insertion text. "
                    "For replace_string: replaces old_string. "
                    "For insert_after: inserted after old_string. "
                    "For insert_before: inserted before old_string. "
                    "For insert_at_line / replace_lines: the content to insert."
                ),
            },
            "start_line": {
                "type": "integer",
                "description": "Starting line number (1-indexed). Used by: insert_at_line, delete_lines, replace_lines.",
            },
            "end_line": {
                "type": "integer",
                "description": "Ending line number, inclusive (1-indexed). Used by: delete_lines, replace_lines.",
            },
        },
        "required": ["path", "action"],
    }

    # ── Public API ──────────────────────────────────────────────────

    async def execute(
        self,
        path: str,
        action: str,
        old_string: str | None = None,
        new_string: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            # Validate action
            if action not in _ACTIONS:
                return ToolResult(success=False, error=f"Unknown action: {action}. Valid: {', '.join(_ACTIONS)}")

            # Validate action-specific required params
            missing = [p for p in _ACTION_REQUIRED[action] if locals().get(p) is None]
            if missing:
                return ToolResult(success=False, error=f"Action '{action}' requires: {', '.join(missing)}")

            # Resolve file path (same logic as ReadTool)
            file_path = self._resolve_path(path, kwargs)
            if file_path is None:
                return ToolResult(success=False, error=f"File not found: {path}")
            if not file_path.is_file():
                return ToolResult(success=False, error=f"Not a file: {path}")

            # Size check
            from captain_claw.config import get_config
            max_bytes = get_config().tools.edit.max_file_bytes
            file_size = file_path.stat().st_size
            if file_size > max_bytes:
                return ToolResult(success=False, error=f"File too large: {file_size} bytes (max {max_bytes})")

            # Handle undo specially (doesn't need to read the file first)
            if action == "undo":
                return self._undo(file_path, kwargs)

            # Read current contents
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    content = file_path.read_text(encoding="latin-1")
                    log.warning("File not UTF-8, fell back to latin-1", path=str(file_path))
                except Exception:
                    return ToolResult(success=False, error=f"Cannot read file (binary or unknown encoding): {path}")

            # Binary check: null bytes in first 8KB
            sample = content[:8192]
            if "\x00" in sample:
                return ToolResult(success=False, error=f"File appears to be binary: {path}")

            # Detect line ending style
            line_ending = "\r\n" if "\r\n" in content else "\n"

            # Normalize old_string / new_string line endings to match file
            if old_string is not None:
                old_string = self._normalize_line_endings(old_string, line_ending)
            if new_string is not None:
                new_string = self._normalize_line_endings(new_string, line_ending)

            # Create backup
            backup_path = self._create_backup(file_path, kwargs)

            # Dispatch to action handler
            result = self._dispatch(action, content, line_ending, old_string, new_string, start_line, end_line)

            if result.error:
                return ToolResult(success=False, error=result.error)

            # Atomic write
            self._atomic_write(file_path, result.content)

            # Build response
            summary = result.summary
            if backup_path:
                summary += f"\nBackup: {backup_path}"
            summary += f"\nFile: {file_path}"

            return ToolResult(success=True, content=summary)

        except Exception as e:
            log.error("Edit failed", path=path, action=action, error=str(e))
            return ToolResult(success=False, error=str(e))

    # ── Path resolution (mirrors ReadTool) ──────────────────────────

    @staticmethod
    def _resolve_path(path: str, kwargs: dict[str, Any]) -> Path | None:
        """Resolve file path using the same logic as ReadTool."""
        raw_path = Path(path).expanduser()

        if raw_path.is_absolute():
            file_path = raw_path.resolve()
        else:
            runtime_base = kwargs.get("_runtime_base_path")
            if runtime_base is not None:
                file_path = (Path(runtime_base) / raw_path).resolve()
            else:
                file_path = raw_path.resolve()

        if not file_path.exists():
            # Try workflow-run directory
            workflow_run_dir = kwargs.get("_workflow_run_dir")
            if workflow_run_dir is not None:
                wrd = Path(workflow_run_dir)
                _rel = Path(path).expanduser()
                if _rel.is_absolute():
                    _rel = Path(*_rel.parts[1:])
                candidate = (wrd / _rel).resolve()
                if not candidate.exists():
                    candidate = wrd / Path(path).name
                if candidate.exists():
                    file_path = candidate

        if not file_path.exists():
            # File registry resolution
            file_registry = kwargs.get("_file_registry")
            if file_registry is not None:
                try:
                    resolved = file_registry.resolve(path)
                except Exception:
                    resolved = None
                if resolved is not None:
                    candidate = Path(resolved).expanduser().resolve()
                    if candidate.exists():
                        return candidate
            return None

        return file_path

    # ── Line ending helpers ─────────────────────────────────────────

    @staticmethod
    def _normalize_line_endings(text: str, target: str) -> str:
        """Normalize line endings in text to match the target style."""
        # First normalize everything to \n, then convert to target
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if target == "\r\n":
            return normalized.replace("\n", "\r\n")
        return normalized

    # ── Backup logic ────────────────────────────────────────────────

    @staticmethod
    def _backup_dir_for(file_path: Path, kwargs: dict[str, Any]) -> Path:
        """Compute the backup directory for a file."""
        from captain_claw.config import get_config
        cfg = get_config().tools.edit
        runtime_base = kwargs.get("_runtime_base_path")
        if runtime_base:
            base = Path(runtime_base)
        else:
            base = Path.cwd()

        # Build a safe directory name from the relative path
        try:
            rel = file_path.resolve().relative_to(base.resolve())
        except ValueError:
            rel = Path(file_path.name)

        safe_dir = str(rel).replace("/", "_").replace("\\", "_")
        return (base / cfg.backup_dir / safe_dir).resolve()

    def _create_backup(self, file_path: Path, kwargs: dict[str, Any]) -> str | None:
        """Create a backup of the file. Returns backup path or None if disabled."""
        from captain_claw.config import get_config
        cfg = get_config().tools.edit
        if not cfg.backup_enabled:
            return None

        backup_dir = self._backup_dir_for(file_path, kwargs)
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}.bak"
        backup_path = backup_dir / backup_name

        shutil.copy2(str(file_path), str(backup_path))
        self._prune_backups(backup_dir, cfg.max_backups)

        return str(backup_path)

    @staticmethod
    def _prune_backups(backup_dir: Path, max_backups: int) -> None:
        """Keep only the N most recent backups in a directory."""
        backups = sorted(backup_dir.glob("*.bak"), key=lambda p: p.stat().st_mtime)
        while len(backups) > max_backups:
            oldest = backups.pop(0)
            try:
                oldest.unlink()
            except OSError:
                pass

    def _get_latest_backup(self, file_path: Path, kwargs: dict[str, Any]) -> Path | None:
        """Return the path to the most recent backup for a file."""
        backup_dir = self._backup_dir_for(file_path, kwargs)
        if not backup_dir.exists():
            return None
        backups = sorted(backup_dir.glob("*.bak"), key=lambda p: p.stat().st_mtime)
        return backups[-1] if backups else None

    def _undo(self, file_path: Path, kwargs: dict[str, Any]) -> ToolResult:
        """Restore the most recent backup of a file."""
        latest = self._get_latest_backup(file_path, kwargs)
        if latest is None:
            return ToolResult(success=False, error=f"No backup found for: {file_path}")

        # Back up the current file first (so undo is itself undoable)
        self._create_backup(file_path, kwargs)

        # Restore
        shutil.copy2(str(latest), str(file_path))
        return ToolResult(success=True, content=f"Restored from backup: {latest}\nFile: {file_path}")

    # ── Atomic write ────────────────────────────────────────────────

    @staticmethod
    def _atomic_write(file_path: Path, content: str) -> None:
        """Write content atomically via temp file + rename."""
        # Write to temp file in the same directory (ensures same filesystem)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(file_path.parent),
            prefix=f".{file_path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            # Preserve original file permissions
            shutil.copymode(str(file_path), tmp_path)
            os.replace(tmp_path, str(file_path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ── Action dispatch ─────────────────────────────────────────────

    def _dispatch(
        self,
        action: str,
        content: str,
        line_ending: str,
        old_string: str | None,
        new_string: str | None,
        start_line: int | None,
        end_line: int | None,
    ) -> _EditResult:
        """Dispatch to the appropriate action handler."""
        if action == "replace_string":
            return self._act_replace_string(content, old_string, new_string)  # type: ignore[arg-type]
        elif action == "insert_after":
            return self._act_insert_after(content, old_string, new_string, line_ending)  # type: ignore[arg-type]
        elif action == "insert_before":
            return self._act_insert_before(content, old_string, new_string, line_ending)  # type: ignore[arg-type]
        elif action == "delete_string":
            return self._act_delete_string(content, old_string)  # type: ignore[arg-type]
        elif action == "insert_at_line":
            return self._act_insert_at_line(content, line_ending, start_line, new_string)  # type: ignore[arg-type]
        elif action == "delete_lines":
            return self._act_delete_lines(content, line_ending, start_line, end_line)  # type: ignore[arg-type]
        elif action == "replace_lines":
            return self._act_replace_lines(content, line_ending, start_line, end_line, new_string)  # type: ignore[arg-type]
        else:
            return _EditResult(error=f"Unknown action: {action}")

    # ── String-match actions ────────────────────────────────────────

    def _act_replace_string(self, content: str, old_string: str, new_string: str) -> _EditResult:
        if not old_string:
            return _EditResult(error="old_string cannot be empty")
        if old_string == new_string:
            return _EditResult(error="old_string and new_string are identical — nothing to change")

        count = content.count(old_string)
        if count == 0:
            return _EditResult(error="String not found in file. Make sure old_string matches the file content exactly (including whitespace and indentation).")
        if count > 1:
            # Find line numbers of each occurrence for helpful error
            lines = content.splitlines(keepends=True)
            match_lines = []
            pos = 0
            for i, line in enumerate(lines, 1):
                idx = content.find(old_string, pos)
                if idx is not None and idx < pos + len(line):
                    # Check if this line contains a match start
                    pass
                pos += len(line)
            # Simpler approach: find all occurrences
            match_positions = []
            start = 0
            while True:
                idx = content.find(old_string, start)
                if idx == -1:
                    break
                line_num = content[:idx].count("\n") + 1
                match_positions.append(line_num)
                start = idx + 1
            lines_str = ", ".join(str(ln) for ln in match_positions)
            return _EditResult(error=f"old_string matches {count} locations (at lines: {lines_str}). Include more surrounding context to make it unique.")

        new_content = content.replace(old_string, new_string, 1)
        # Find the line where replacement happened
        idx = content.find(old_string)
        line_num = content[:idx].count("\n") + 1
        old_lines = old_string.count("\n") + 1
        new_lines = new_string.count("\n") + 1

        # Context around edit
        context = self._context_around(new_content, line_num, new_lines)

        return _EditResult(
            content=new_content,
            summary=f"replace_string: replaced {old_lines} line(s) with {new_lines} line(s) at line {line_num}\n{context}",
        )

    def _act_insert_after(self, content: str, old_string: str, new_string: str, line_ending: str) -> _EditResult:
        if not old_string:
            return _EditResult(error="old_string (anchor) cannot be empty")

        count = content.count(old_string)
        if count == 0:
            return _EditResult(error="Anchor string not found in file.")
        if count > 1:
            return _EditResult(error=f"Anchor string matches {count} locations. Include more context to make it unique.")

        idx = content.find(old_string)
        insert_pos = idx + len(old_string)

        # Ensure new_string starts on a new line if the anchor doesn't end with one
        insertion = new_string
        if not old_string.endswith(line_ending) and not new_string.startswith(line_ending):
            insertion = line_ending + new_string

        new_content = content[:insert_pos] + insertion + content[insert_pos:]
        line_num = content[:insert_pos].count("\n") + 1
        inserted_lines = new_string.count("\n") + 1

        context = self._context_around(new_content, line_num, inserted_lines + 2)

        return _EditResult(
            content=new_content,
            summary=f"insert_after: inserted {inserted_lines} line(s) after line {line_num}\n{context}",
        )

    def _act_insert_before(self, content: str, old_string: str, new_string: str, line_ending: str) -> _EditResult:
        if not old_string:
            return _EditResult(error="old_string (anchor) cannot be empty")

        count = content.count(old_string)
        if count == 0:
            return _EditResult(error="Anchor string not found in file.")
        if count > 1:
            return _EditResult(error=f"Anchor string matches {count} locations. Include more context to make it unique.")

        idx = content.find(old_string)

        # Ensure new_string ends with a line ending if it doesn't
        insertion = new_string
        if not new_string.endswith(line_ending) and not old_string.startswith(line_ending):
            insertion = new_string + line_ending

        new_content = content[:idx] + insertion + content[idx:]
        line_num = content[:idx].count("\n") + 1
        inserted_lines = new_string.count("\n") + 1

        context = self._context_around(new_content, line_num, inserted_lines + 2)

        return _EditResult(
            content=new_content,
            summary=f"insert_before: inserted {inserted_lines} line(s) before line {line_num}\n{context}",
        )

    def _act_delete_string(self, content: str, old_string: str) -> _EditResult:
        if not old_string:
            return _EditResult(error="old_string cannot be empty")

        count = content.count(old_string)
        if count == 0:
            return _EditResult(error="String not found in file.")
        if count > 1:
            return _EditResult(error=f"String matches {count} locations. Include more context to make it unique.")

        idx = content.find(old_string)
        line_num = content[:idx].count("\n") + 1
        deleted_lines = old_string.count("\n") + 1

        new_content = content.replace(old_string, "", 1)

        context = self._context_around(new_content, line_num, 3)

        return _EditResult(
            content=new_content,
            summary=f"delete_string: removed {deleted_lines} line(s) at line {line_num}\n{context}",
        )

    # ── Line-based actions ──────────────────────────────────────────

    def _act_insert_at_line(self, content: str, line_ending: str, start_line: int, new_string: str) -> _EditResult:
        lines = content.split(line_ending)
        if start_line < 1:
            return _EditResult(error="start_line must be >= 1")
        if start_line > len(lines) + 1:
            return _EditResult(error=f"start_line {start_line} is past end of file ({len(lines)} lines)")

        new_lines = new_string.split(line_ending) if new_string else [""]
        # Insert at position (0-indexed)
        insert_idx = start_line - 1
        lines[insert_idx:insert_idx] = new_lines

        new_content = line_ending.join(lines)
        context = self._context_around(new_content, start_line, len(new_lines) + 2, line_ending)

        return _EditResult(
            content=new_content,
            summary=f"insert_at_line: inserted {len(new_lines)} line(s) at line {start_line}\n{context}",
        )

    def _act_delete_lines(self, content: str, line_ending: str, start_line: int, end_line: int) -> _EditResult:
        lines = content.split(line_ending)
        if start_line < 1 or end_line < start_line:
            return _EditResult(error=f"Invalid line range: {start_line}-{end_line}")
        if start_line > len(lines):
            return _EditResult(error=f"start_line {start_line} is past end of file ({len(lines)} lines)")
        end_line = min(end_line, len(lines))

        deleted = lines[start_line - 1 : end_line]
        del lines[start_line - 1 : end_line]

        new_content = line_ending.join(lines)
        context = self._context_around(new_content, start_line, 3, line_ending)

        return _EditResult(
            content=new_content,
            summary=f"delete_lines: removed lines {start_line}-{end_line} ({len(deleted)} line(s))\n{context}",
        )

    def _act_replace_lines(self, content: str, line_ending: str, start_line: int, end_line: int, new_string: str) -> _EditResult:
        lines = content.split(line_ending)
        if start_line < 1 or end_line < start_line:
            return _EditResult(error=f"Invalid line range: {start_line}-{end_line}")
        if start_line > len(lines):
            return _EditResult(error=f"start_line {start_line} is past end of file ({len(lines)} lines)")
        end_line = min(end_line, len(lines))

        new_lines = new_string.split(line_ending) if new_string else [""]
        old_count = end_line - start_line + 1
        lines[start_line - 1 : end_line] = new_lines

        new_content = line_ending.join(lines)
        context = self._context_around(new_content, start_line, len(new_lines) + 2, line_ending)

        return _EditResult(
            content=new_content,
            summary=f"replace_lines: replaced {old_count} line(s) with {len(new_lines)} line(s) at lines {start_line}-{end_line}\n{context}",
        )

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _context_around(content: str, target_line: int, span: int = 3, line_ending: str | None = None) -> str:
        """Return a few lines of context around the edit point, with line numbers."""
        if line_ending is None:
            line_ending = "\r\n" if "\r\n" in content else "\n"
        lines = content.split(line_ending)
        start = max(0, target_line - 2)
        end = min(len(lines), target_line + span)
        context_lines = []
        for i in range(start, end):
            prefix = ">>>" if i + 1 == target_line else "   "
            context_lines.append(f"{prefix} {i + 1:4d} | {lines[i]}")
        return "\n".join(context_lines)


class _EditResult:
    """Internal result from an action handler."""

    __slots__ = ("content", "summary", "error")

    def __init__(
        self,
        content: str = "",
        summary: str = "",
        error: str = "",
    ):
        self.content = content
        self.summary = summary
        self.error = error
