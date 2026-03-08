"""Build compact file-tree listings for context injection.

Produces Unicode tree strings for local directories and Google Drive folders
(via the ``gws`` CLI) so the LLM can see available files without calling
``glob`` or ``gws drive_list`` first.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

# ── Cache ─────────────────────────────────────────────────────────────
# key → (timestamp, tree_str, entry_count)
_tree_cache: dict[str, tuple[float, str, int]] = {}


def get_cached_tree(key: str, ttl: int) -> str | None:
    """Return cached tree string if still valid, else *None*."""
    entry = _tree_cache.get(key)
    if entry is None:
        return None
    ts, tree_str, _ = entry
    if time.time() - ts > ttl:
        del _tree_cache[key]
        return None
    return tree_str


def set_cached_tree(key: str, tree_str: str, entry_count: int) -> None:
    """Store *tree_str* in cache."""
    _tree_cache[key] = (time.time(), tree_str, entry_count)


def clear_cache() -> None:
    """Clear all cached trees."""
    _tree_cache.clear()


# ── Helpers ───────────────────────────────────────────────────────────

def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


# ── Local file tree ──────────────────────────────────────────────────

def build_local_tree(
    directory: str,
    max_entries: int = 50,
    max_depth: int = 2,
) -> tuple[str, int]:
    """Walk a local directory and return ``(tree_string, entry_count)``."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return f"[Directory not found: {directory}]", 0

    lines: list[str] = []
    total_files = 0
    total_dirs = 0
    entry_count = 0

    def _walk(path: Path, depth: int, prefix: str) -> None:
        nonlocal total_files, total_dirs, entry_count
        if entry_count >= max_entries:
            return

        try:
            entries = sorted(
                path.iterdir(),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
        except PermissionError:
            lines.append(f"{prefix}[permission denied]")
            return

        visible = [e for e in entries if not e.name.startswith(".")]

        for i, entry in enumerate(visible):
            if entry_count >= max_entries:
                remaining = len(visible) - i
                if remaining > 0:
                    lines.append(f"{prefix}... and {remaining} more")
                return

            is_last = i == len(visible) - 1
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            child_prefix = prefix + ("    " if is_last else "\u2502   ")

            if entry.is_dir():
                total_dirs += 1
                entry_count += 1
                try:
                    child_count = sum(
                        1 for c in entry.iterdir() if not c.name.startswith(".")
                    )
                except PermissionError:
                    child_count = 0
                lines.append(f"{prefix}{connector}{entry.name}/ ({child_count} items)")
                if depth < max_depth:
                    _walk(entry, depth + 1, child_prefix)
            elif entry.is_file():
                total_files += 1
                entry_count += 1
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"{prefix}{connector}{entry.name} ({_format_size(size)})")

    _walk(root, 1, "  ")

    header = f"Local: {root} ({total_files} files, {total_dirs} dirs)"
    if entry_count >= max_entries:
        header += f" [truncated at {max_entries} entries]"

    return header + "\n" + "\n".join(lines), entry_count


# ── GWS helper ────────────────────────────────────────────────────────

_GWS_TIMEOUT = 30  # seconds per call


def resolve_gws_binary() -> str | None:
    """Find the ``gws`` binary (mirrors ``GwsTool._resolve_binary``)."""
    try:
        from captain_claw.config import get_config

        cfg = get_config()
        custom = getattr(cfg.tools, "gws", None)
        if custom and hasattr(custom, "binary_path") and custom.binary_path:
            p = Path(custom.binary_path).expanduser()
            if p.exists():
                return str(p)
    except Exception:
        pass

    found = shutil.which("gws")
    return found


async def _run_gws(binary: str, args: list[str]) -> dict[str, Any] | str:
    """Run a ``gws`` command and return parsed JSON or error string."""
    cmd = [binary] + args + ["--format", "json"]
    log.debug("file_tree_builder gws", cmd=" ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_GWS_TIMEOUT,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return "gws command timed out"

    stdout_str = stdout.decode("utf-8", errors="replace").strip()
    stderr_str = stderr.decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        return stderr_str or stdout_str or f"gws exited with code {proc.returncode}"

    try:
        return json.loads(stdout_str)  # type: ignore[return-value]
    except (json.JSONDecodeError, TypeError):
        return stdout_str


# ── Google Drive file tree (via gws) ─────────────────────────────────

async def build_gdrive_tree(
    folder_id: str,
    folder_name: str,
    max_entries: int = 50,
    max_depth: int = 2,
) -> tuple[str, int]:
    """List a Google Drive folder via ``gws`` CLI and return ``(tree_string, entry_count)``."""
    binary = resolve_gws_binary()
    if binary is None:
        return "[gws CLI not available]", 0

    lines: list[str] = []
    entry_count = 0

    async def _list_folder(fid: str, depth: int, prefix: str) -> None:
        nonlocal entry_count
        if entry_count >= max_entries:
            return

        escaped = fid.replace("'", "\\'")
        params: dict[str, Any] = {
            "q": f"'{escaped}' in parents and trashed = false",
            "pageSize": min(max_entries - entry_count, 100),
            "fields": "files(id,name,mimeType,size,modifiedTime)",
            "orderBy": "folder,name",
            "corpora": "allDrives",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        result = await _run_gws(
            binary,
            ["drive", "files", "list", "--params", json.dumps(params)],
        )

        if isinstance(result, str):
            # Error.
            lines.append(f"{prefix}[error: {result[:80]}]")
            return

        files = result.get("files", [])

        for i, f in enumerate(files):
            if entry_count >= max_entries:
                remaining = len(files) - i
                if remaining > 0:
                    lines.append(f"{prefix}... and {remaining} more")
                return

            is_last = i == len(files) - 1
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            child_prefix = prefix + ("    " if is_last else "\u2502   ")
            is_folder = f.get("mimeType") == "application/vnd.google-apps.folder"

            entry_count += 1

            if is_folder:
                lines.append(f"{prefix}{connector}{f['name']}/ [id:{f['id']}]")
                if depth < max_depth:
                    await _list_folder(f["id"], depth + 1, child_prefix)
            else:
                size = f.get("size", "")
                size_str = f" ({_format_size(int(size))})" if size else ""
                lines.append(
                    f"{prefix}{connector}{f['name']}{size_str} [id:{f['id']}]"
                )

    await _list_folder(folder_id, 1, "  ")

    header = f"Google Drive: {folder_name} ({entry_count} entries)"
    if entry_count >= max_entries:
        header += f" [truncated at {max_entries} entries]"

    return header + "\n" + "\n".join(lines), entry_count


# ── Shared drives helper ──────────────────────────────────────────────

async def _list_shared_drives(binary: str) -> list[dict[str, str]]:
    """Return ``[{"id": ..., "name": ...}]`` for all accessible shared drives."""
    params = {
        "pageSize": 100,
        "fields": "drives(id,name)",
    }
    result = await _run_gws(
        binary,
        ["drive", "drives", "list", "--params", json.dumps(params)],
    )
    if isinstance(result, str):
        log.debug("shared drives listing failed", error=result[:120])
        return []
    return [
        {"id": d["id"], "name": d["name"]}
        for d in result.get("drives", [])
        if d.get("id") and d.get("name")
    ]


# ── Browse GDrive folders (for UI) ───────────────────────────────────

async def browse_gdrive_folders(folder_id: str = "root") -> dict[str, Any]:
    """List subfolders in a Google Drive folder (for the folder picker UI).

    When *folder_id* is ``"root"`` the result also includes any shared drives
    the user has access to (returned in a separate ``shared_drives`` key).

    Returns ``{"folders": [...], "shared_drives": [...], "error": ...}``.
    """
    binary = resolve_gws_binary()
    if binary is None:
        return {"folders": [], "shared_drives": [], "error": "gws CLI not available"}

    escaped = folder_id.replace("'", "\\'")
    params: dict[str, Any] = {
        "q": (
            f"'{escaped}' in parents and trashed = false "
            "and mimeType = 'application/vnd.google-apps.folder'"
        ),
        "pageSize": 100,
        "fields": "files(id,name)",
        "orderBy": "name",
        "corpora": "allDrives",
        "supportsAllDrives": "true",
        "includeItemsFromAllDrives": "true",
    }

    result = await _run_gws(
        binary,
        ["drive", "files", "list", "--params", json.dumps(params)],
    )

    if isinstance(result, str):
        return {"folders": [], "shared_drives": [], "error": result}

    folders = [
        {"id": f["id"], "name": f["name"]}
        for f in result.get("files", [])
        if f.get("id") and f.get("name")
    ]

    # When browsing root, also fetch shared drives.
    shared_drives: list[dict[str, str]] = []
    if folder_id == "root":
        shared_drives = await _list_shared_drives(binary)

    return {"folders": folders, "shared_drives": shared_drives}
