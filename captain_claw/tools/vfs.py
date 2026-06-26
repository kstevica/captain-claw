"""Admin tool for the shared cross-agent virtual filesystem (VFS).

The file *contents* of the VFS are reached with the ordinary ``read`` /
``write`` / ``edit`` / ``glob`` / ``grep`` tools using ``vfs:<project>/...``
paths. This tool covers the operations those don't: discovering projects,
listing/inspecting the tree, and moving/removing/creating entries.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult
from captain_claw.vfs import (
    default_project,
    is_vfs_path,
    list_projects,
    project_root,
    resolve_vfs_path,
    to_display,
    user_root,
    vfs_base,
    vfs_user,
)

log = get_logger(__name__)

_ACTIONS = ("info", "list_projects", "ls", "tree", "stat", "mkdir", "mv", "rm")
_TREE_MAX = 500


def _as_vfs(path: str) -> str:
    """Accept either ``vfs:proj/x`` or a bare ``proj/x`` and normalise."""
    p = (path or "").strip()
    if not p:
        return f"vfs:{default_project()}"
    return p if is_vfs_path(p) else f"vfs:{p}"


class VfsTool(Tool):
    """Manage the shared virtual filesystem (discovery + directory ops)."""

    name = "vfs"
    description = (
        "Manage the SHARED cross-agent virtual filesystem — a persistent file "
        "tree that every agent (and every run) can see, addressed as "
        "vfs:<project>/<path>. Use it to share a working file context across "
        "agents collaborating on the same task (e.g. a coding project) without "
        "losing it between sessions. Read/write/edit/glob/grep file CONTENTS "
        "with those tools using vfs: paths; use THIS tool for: "
        "info (show the active user/project/root); list_projects; "
        "ls (one level); tree (recursive); stat; mkdir; mv (path -> to); "
        "rm (recursive optional). Paths may be 'vfs:proj/sub' or just 'proj/sub'."
    )
    timeout_seconds = 15.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(_ACTIONS)},
            "path": {
                "type": "string",
                "description": "Target path, 'vfs:<project>/<sub>' or '<project>/<sub>'. Omit for list_projects/info.",
            },
            "to": {
                "type": "string",
                "description": "Destination path for mv.",
            },
            "recursive": {
                "type": "boolean",
                "description": "For rm: remove a non-empty directory.",
            },
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str,
        path: str = "",
        to: str = "",
        recursive: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            if action not in _ACTIONS:
                return ToolResult(success=False, error=f"Unknown action: {action}. Valid: {', '.join(_ACTIONS)}")

            if action == "info":
                return ToolResult(success=True, content=(
                    f"Shared VFS\n"
                    f"  user:            {vfs_user()}\n"
                    f"  default project: {default_project()}\n"
                    f"  root:            {user_root()}\n"
                    f"  base:            {vfs_base()}\n"
                    f"  projects:        {', '.join(list_projects()) or '(none yet)'}\n"
                    f"Address files as vfs:<project>/<path> with read/write/edit/glob/grep."
                ))

            if action == "list_projects":
                projects = list_projects()
                if not projects:
                    return ToolResult(success=True, content="No projects yet. Writing vfs:<project>/<file> creates one.")
                lines = []
                for proj in projects:
                    n = sum(1 for p in project_root(proj).rglob("*") if p.is_file())
                    lines.append(f"  {proj}  ({n} file{'s' if n != 1 else ''})")
                return ToolResult(success=True, content="Projects:\n" + "\n".join(lines))

            # Remaining actions need a resolved path.
            target = resolve_vfs_path(_as_vfs(path))
            if target is None:
                return ToolResult(success=False, error=f"Invalid vfs path (escapes user root): {path}")

            if action == "ls":
                if not target.exists():
                    return ToolResult(success=False, error=f"Not found: {to_display(target)}")
                if target.is_file():
                    return ToolResult(success=True, content=self._fmt_entry(target))
                entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
                if not entries:
                    return ToolResult(success=True, content=f"{to_display(target)} is empty.")
                body = "\n".join(self._fmt_entry(e) for e in entries)
                return ToolResult(success=True, content=f"{to_display(target)}:\n{body}")

            if action == "tree":
                if not target.exists():
                    return ToolResult(success=False, error=f"Not found: {to_display(target)}")
                lines: list[str] = []
                self._tree(target, target, lines)
                more = "\n  … (truncated)" if len(lines) >= _TREE_MAX else ""
                return ToolResult(success=True, content=f"{to_display(target)}:\n" + "\n".join(lines[:_TREE_MAX]) + more)

            if action == "stat":
                if not target.exists():
                    return ToolResult(success=False, error=f"Not found: {to_display(target)}")
                return ToolResult(success=True, content=self._fmt_entry(target, verbose=True))

            if action == "mkdir":
                target.mkdir(parents=True, exist_ok=True)
                return ToolResult(success=True, content=f"Created directory {to_display(target)}")

            if action == "mv":
                if not target.exists():
                    return ToolResult(success=False, error=f"Source not found: {to_display(target)}")
                dest = resolve_vfs_path(_as_vfs(to))
                if dest is None:
                    return ToolResult(success=False, error=f"Invalid destination: {to}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(dest))
                return ToolResult(success=True, content=f"Moved {to_display(target)} -> {to_display(dest)}")

            if action == "rm":
                if not target.exists():
                    return ToolResult(success=False, error=f"Not found: {to_display(target)}")
                # Guard against removing a whole project/user root by accident.
                if target.resolve() == user_root().resolve():
                    return ToolResult(success=False, error="Refusing to remove the VFS user root.")
                if target.is_dir():
                    if not recursive and any(target.iterdir()):
                        return ToolResult(success=False, error=f"{to_display(target)} is a non-empty directory; pass recursive=true.")
                    shutil.rmtree(target)
                else:
                    target.unlink()
                return ToolResult(success=True, content=f"Removed {to_display(target)}")

            return ToolResult(success=False, error=f"Unhandled action: {action}")

        except Exception as e:
            log.error("vfs tool failed", action=action, path=path, error=str(e))
            return ToolResult(success=False, error=str(e))

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _fmt_entry(p: Path, *, verbose: bool = False) -> str:
        try:
            st = p.stat()
        except OSError:
            return f"  {p.name}"
        if p.is_dir():
            return f"  {p.name}/"
        size = st.st_size
        label = f"  {p.name}  ({size} bytes)"
        if verbose:
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
            label += f"  modified {mtime}\n  path: {to_display(p)}"
        return label

    def _tree(self, base: Path, node: Path, out: list[str], prefix: str = "") -> None:
        if len(out) >= _TREE_MAX:
            return
        try:
            entries = sorted(node.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        for e in entries:
            if len(out) >= _TREE_MAX:
                return
            out.append(f"  {prefix}{e.name}{'/' if e.is_dir() else ''}")
            if e.is_dir():
                self._tree(base, e, out, prefix + "  ")
