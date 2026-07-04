"""Read tool for reading file contents."""

import asyncio
import os
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult
from captain_claw.vfs import is_vfs_path, resolve_vfs_path

log = get_logger(__name__)


class ReadTool(Tool):
    """Read file contents."""

    name = "read"
    description = (
        "Read a text file. Returns the content with a `[path N chars] [lines a-b]` "
        "header. To read just part of a large file, pass offset (1-indexed start line) "
        "and limit (number of lines) — ALWAYS prefer this over shell `sed`/`head`/"
        "`tail`, which bypass the size/binary guards and duplicate-read detection. "
        "Omit offset and limit to read the whole file. Very large files are served "
        "in pages: output is capped per call and cut at a line boundary, with a "
        "header telling you the exact offset to continue from — page through only "
        "the parts you need instead of re-requesting the whole file."
    )
    timeout_seconds = 10.0  # local file read — 10 s is ample
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path to the file to read. A vfs:<project>/<path> path reads "
                    "from the shared cross-agent filesystem (see the vfs tool)."
                ),
            },
            "offset": {
                "type": "number",
                "description": (
                    "1-indexed line to start reading from. Combine with limit to read a "
                    "line range instead of using shell sed/head/tail."
                ),
            },
            "limit": {
                "type": "number",
                "description": (
                    "Number of lines to read from offset (a slice). Example: offset=630, "
                    "limit=41 reads lines 630–670."
                ),
            },
            "force": {
                "type": "boolean",
                "description": (
                    "Re-send the full content even if you already read this exact "
                    "(unchanged) file in full earlier — identical full re-reads are "
                    "otherwise short-circuited to save context."
                ),
            },
        },
        "required": ["path"],
    }

    async def execute(self, path: str, limit: int | None = None, offset: int | None = None,
                      force: bool = False, **kwargs: Any) -> ToolResult:
        """Read a file.

        Args:
            path: Path to file
            limit: Optional line limit
            offset: Optional line offset
            force: Re-send full content even if unchanged since the last full read

        Returns:
            ToolResult with file contents
        """
        try:
            # Block reads of gws side-effect artifact files.
            if Path(path).name == "download.bin":
                return ToolResult(
                    success=False,
                    error=(
                        "download.bin is a temporary gws export artifact — do not read it. "
                        "The document content was already returned inline by the docs_read "
                        "tool call. Check the previous gws docs_read result."
                    ),
                )

            # Shared VFS paths (vfs:<project>/...) resolve into the
            # cross-agent tree; skip the workspace/cwd/registry fallbacks.
            vfs_target = resolve_vfs_path(path) if is_vfs_path(path) else None
            if is_vfs_path(path) and vfs_target is None:
                return ToolResult(success=False, error=f"Invalid vfs path (escapes user root): {path}")

            raw_path = Path(path).expanduser()

            # Resolve relative paths against the workspace root (not the
            # process CWD) so that paths like "pdf-test/foo.pdf" resolve
            # consistently across all tools.
            if vfs_target is not None:
                file_path = vfs_target
            elif raw_path.is_absolute():
                file_path = raw_path.resolve()
            else:
                runtime_base = kwargs.get("_runtime_base_path")
                if runtime_base is not None:
                    file_path = (Path(runtime_base) / raw_path).resolve()
                else:
                    file_path = raw_path.resolve()

            if not file_path.exists() and vfs_target is None:
                # Some tools (gws drive_download, shell) write files
                # relative to the process CWD which may differ from the
                # workspace root.  Try CWD-based resolution.
                runtime_base = kwargs.get("_runtime_base_path")
                if not raw_path.is_absolute():
                    # Relative path: try resolving against cwd instead of
                    # workspace root.
                    cwd_candidate = raw_path.resolve()
                    if cwd_candidate.exists():
                        file_path = cwd_candidate
                elif runtime_base is not None:
                    # Absolute path under workspace root that doesn't exist:
                    # strip workspace prefix and try the remainder under cwd
                    # (e.g. workspace/saved/x.md → cwd/saved/x.md).
                    try:
                        rel = file_path.relative_to(Path(runtime_base).resolve())
                        cwd_candidate = (Path.cwd() / rel).resolve()
                        if cwd_candidate.exists():
                            file_path = cwd_candidate
                    except ValueError:
                        pass

            if not file_path.exists() and vfs_target is None:
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

            if not file_path.exists() and vfs_target is not None:
                return ToolResult(success=False, error=f"File not found: {path}")

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
            
            # Images / binaries aren't text — guide the agent to the right tool
            # instead of choking on a size limit or a utf-8 decode error.
            _img_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".heic"}
            if file_path.suffix.lower() in _img_exts:
                return ToolResult(
                    success=False,
                    error=(
                        f"'{file_path.name}' is an image, not a text file. Use the "
                        "image_vision tool (action on this path) to see/describe it — "
                        "do NOT use read. (Requires a vision-capable model configured.)"
                    ),
                )

            # Check if file is too large
            _stat = file_path.stat()
            file_size = _stat.st_size
            from captain_claw.config import get_config
            max_size = get_config().tools.read.max_file_bytes
            if file_size > max_size:
                return ToolResult(
                    success=False,
                    error=(
                        f"File too large: {file_size} bytes (max {max_size}). This is "
                        "almost certainly a log/data/generated file, not source. Use "
                        "shell `grep -n` to find the relevant region, or `sed -n 'a,bp'` "
                        "to extract a specific line range."
                    ),
                )

            # ── Duplicate full-read short-circuit (code agents) ──────
            # Code agents re-read whole files they already hold in context
            # (SW3: index.html fully read 115×) — each re-read compounds the
            # tool-loop's quadratic context growth. If THIS process already
            # served a full read of this exact file and it is unchanged on
            # disk (mtime+size), return a short pointer instead of the
            # content. offset/limit range reads and force=true always pass.
            _full_read = offset is None and limit is None
            _dedup_on = bool(os.environ.get("CLAW_CODE_AGENT"))
            _served: dict = getattr(self, "_served_full", None) or {}
            self._served_full = _served
            _key = str(file_path)
            if _full_read and _dedup_on and not force:
                prev = _served.get(_key)
                if prev and prev[0] == _stat.st_mtime_ns and prev[1] == file_size:
                    log.info("Read short-circuited (unchanged full re-read)", path=_key)
                    return ToolResult(
                        success=True,
                        content=(
                            f"[{file_path} UNCHANGED — content omitted]\n"
                            f"You already read this file in full ({prev[2]} lines, "
                            f"{file_size} bytes) and it has not changed on disk since. "
                            "Use the copy already in your context. To see a specific "
                            "part again, read a range with offset/limit; if you "
                            "genuinely need the whole file re-sent, pass force=true."
                        ),
                    )

            # Read file
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return ToolResult(
                    success=False,
                    error=(
                        f"'{file_path.name}' is a binary file and can't be read as text. "
                        "If it's an image, use the image_vision tool instead."
                    ),
                )

            # Apply offset and limit
            all_lines = content.splitlines()
            total_lines = len(all_lines)
            start_line = max(1, int(offset)) if offset is not None else 1
            selected_lines = all_lines[start_line - 1 :]
            if limit is not None:
                selected_lines = selected_lines[: max(0, int(limit))]

            # ── Context guard: one call never dumps more than the char budget ──
            # Oversized output is cut at a line boundary with an explicit
            # continuation offset, so the agent pages through big files instead
            # of blowing its context on a single read.
            max_chars = get_config().tools.read.max_full_read_chars
            truncated = False
            if sum(len(ln) + 1 for ln in selected_lines) > max_chars:
                truncated = True
                kept: list[str] = []
                used = 0
                for ln in selected_lines:
                    cost = len(ln) + 1
                    if used + cost > max_chars:
                        if not kept:
                            # A single monster line (e.g. minified JS) — serve a
                            # mid-line cut rather than nothing.
                            kept.append(ln[:max_chars] + " …[line cut at budget]")
                        break
                    kept.append(ln)
                    used += cost
                selected_lines = kept

            # Register the served full read for the duplicate short-circuit —
            # only when the file was ACTUALLY served in full: a truncated read
            # must not make a later re-read claim "you already have all of it".
            if _full_read and _dedup_on and not truncated:
                _served[_key] = (_stat.st_mtime_ns, file_size,
                                 content.count("\n") + 1)

            content = "\n".join(selected_lines)

            # Add metadata
            end_line = start_line + len(selected_lines) - 1 if selected_lines else start_line - 1
            info = f"[{file_path} {len(content)} chars]"
            if truncated:
                info += (
                    f" [lines {start_line}-{end_line} of {total_lines} — TRUNCATED at "
                    f"~{max_chars} chars. Continue with offset={end_line + 1}]"
                )
            elif offset is not None or limit is not None:
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
