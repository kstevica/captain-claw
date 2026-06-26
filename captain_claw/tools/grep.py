"""Grep tool — search for text INSIDE files (the content counterpart to glob,
which finds files by name). Sandboxed to the workspace, skips binaries, and is
tracked by the duplicate-call guard — unlike shell `grep`."""

import asyncio
import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult
from captain_claw.vfs import is_vfs_path, project_root, resolve_vfs_path, split_scheme

log = get_logger(__name__)

# Text-ish files searched by default when scanning a directory (a binary or
# undecodable file is skipped regardless of extension).
_TEXT_EXTS = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".xml", ".svg",
    ".css", ".scss", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".json",
    ".py", ".sh", ".bash", ".zsh", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".conf", ".env", ".csv", ".tsv", ".sql", ".log", ".rb", ".php", ".go",
    ".rs", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs", ".swift",
    ".lua", ".pl", ".r", ".tex", ".vue", ".svelte", ".flow",
})
_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", ".captain-claw", ".next", "dist", "build"})
_MAX_FILE_BYTES = 5_000_000   # skip files larger than ~5 MB
_MAX_FILES = 5000             # bound a worst-case directory scan
_MAX_LINE_LEN = 400           # trim very long matched lines in output


class GrepTool(Tool):
    """Search for text inside files."""

    name = "grep"
    description = (
        "Search for text INSIDE files — the content counterpart to glob (which "
        "finds files by name). Give a `pattern` and an optional `path` (a single "
        "file, or a directory scanned recursively; default workspace root). "
        "Returns matching lines as `path:line: text`. Prefer this over shell "
        "grep/rg/sed: it's sandboxed, skips binaries, and is tracked for "
        "duplicate-call detection. By default the pattern is a literal, "
        "case-insensitive substring; set regex=true for a regular expression."
    )
    timeout_seconds = 15.0
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Text to find (literal substring by default; set regex=true for a regex).",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search (default: workspace root; or vfs:<project>/<path> for the shared cross-agent filesystem). Directories are scanned recursively.",
            },
            "glob": {
                "type": "string",
                "description": "When path is a directory, only search files whose name matches this glob (e.g. '*.html', '*.py').",
            },
            "regex": {
                "type": "boolean",
                "description": "Treat pattern as a regular expression (default false = literal substring).",
            },
            "ignore_case": {
                "type": "boolean",
                "description": "Case-insensitive match (default true).",
            },
            "limit": {
                "type": "number",
                "description": "Max matching lines to return (default 100).",
            },
        },
        "required": ["pattern"],
    }

    async def execute(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        regex: bool = False,
        ignore_case: bool = True,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            if not (pattern or "").strip():
                return ToolResult(success=False, error="grep: 'pattern' is required.")
            try:
                limit = max(1, min(int(limit), 1000))
            except Exception:
                limit = 100

            flags = re.IGNORECASE if ignore_case else 0
            try:
                rx = re.compile(pattern if regex else re.escape(pattern), flags)
            except re.error as exc:
                return ToolResult(success=False, error=f"grep: invalid regex: {exc}")

            # Resolve the search root against the workspace base (like read/glob).
            base = kwargs.get("_runtime_base_path")
            _vfs_rel_base: Path | None = None
            raw = Path(path).expanduser() if path else None
            if path and is_vfs_path(path):
                # Shared VFS search — root is a project subtree.
                vfs_root = resolve_vfs_path(path)
                if vfs_root is None:
                    return ToolResult(success=False, error=f"Invalid vfs path (escapes user root): {path}")
                root = vfs_root
                _vfs_rel_base = project_root(split_scheme(path)[0])
            elif raw is None:
                root = Path(base).resolve() if base else Path.cwd()
            elif raw.is_absolute():
                root = raw.resolve()
            elif base is not None:
                root = (Path(base) / raw).resolve()
            else:
                root = raw.resolve()

            if not root.exists():
                return ToolResult(success=False, error=f"grep: path not found: {path or '.'}")

            # Make output paths relative to the workspace base when possible.
            rel_base = Path(base).resolve() if base else (root if root.is_dir() else root.parent)
            if _vfs_rel_base is not None:
                rel_base = _vfs_rel_base

            # Gather candidate files.
            files: list[Path] = []
            if root.is_file():
                files = [root]
            else:
                gl = (glob or "").strip().lower()
                for r, dirs, names in os.walk(root):
                    dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
                    for name in names:
                        if gl:
                            if not fnmatch.fnmatch(name.lower(), gl):
                                continue
                        elif Path(name).suffix.lower() not in _TEXT_EXTS:
                            continue
                        files.append(Path(r) / name)
                        if len(files) >= _MAX_FILES:
                            break
                    if len(files) >= _MAX_FILES:
                        break

            loop = asyncio.get_event_loop()
            lines, matched, scanned, truncated = await loop.run_in_executor(
                None, lambda: self._scan(files, rx, rel_base, limit)
            )

            if not lines:
                where = f" in {path}" if path else ""
                return ToolResult(
                    success=True,
                    content=f"No matches for {pattern!r}{where} ({scanned} file(s) searched).",
                )

            header = (
                f"{matched} match(es) in {scanned} file(s)"
                + (" — output truncated, narrow the search" if truncated else "")
                + ":\n"
            )
            return ToolResult(success=True, content=header + "\n".join(lines))

        except Exception as e:
            log.error("grep failed", pattern=pattern, error=str(e))
            return ToolResult(success=False, error=str(e))

    @staticmethod
    def _scan(files: list[Path], rx: "re.Pattern[str]", rel_base: Path, limit: int):
        out: list[str] = []
        matched = 0
        scanned = 0
        truncated = False
        for fp in files:
            try:
                if fp.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = fp.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue  # missing, binary, or unreadable — skip quietly
            scanned += 1
            try:
                rel = str(fp.resolve().relative_to(rel_base))
            except Exception:
                rel = str(fp)
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    matched += 1
                    disp = line.strip()
                    if len(disp) > _MAX_LINE_LEN:
                        disp = disp[:_MAX_LINE_LEN] + "…"
                    out.append(f"{rel}:{i}: {disp}")
                    if len(out) >= limit:
                        return out, matched, scanned, True
        return out, matched, scanned, truncated
