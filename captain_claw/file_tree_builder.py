"""Build compact file-tree listings for context injection.

Produces Unicode tree strings for local directories and Google Drive folders
(via the Drive API, as the agent owner) so the LLM can see available files
without calling ``glob`` or ``google_drive list`` first.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from captain_claw import drive_client
from captain_claw.drive_client import DriveClient, DriveError, DriveNotConnected
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


# ── Drive client (the agent owner's identity) ────────────────────────

# The picker shows folders only; children come back folders-first, so one
# capped page covers any realistic folder (Flight Deck's picker does the same).
_BROWSE_MAX_CHILDREN = 500


def _owner_drive_client() -> DriveClient:
    """A Drive client whose Google identity is resolved once, then reused.

    The identity is :func:`drive_client.global_token_provider` — the one every
    agent-side Google tool uses: standalone, this instance's own connection;
    under Flight Deck, the agent OWNER's token, so a listing can only ever
    show that owner's Drive. The client asks its provider on every request;
    memoising it makes a whole tree one Flight Deck token round-trip, not one
    per folder. No identity → the first request raises
    :class:`DriveNotConnected` before any HTTP call (fail closed).
    """
    resolved: tuple[str, str] | None = None

    async def _once() -> tuple[str, str]:
        nonlocal resolved
        if resolved is None:
            resolved = await drive_client.global_token_provider()
        return resolved

    return drive_client.make_client(_once)


async def _close_quietly(client: DriveClient) -> None:
    try:
        await client.close()
    except Exception:  # never let cleanup turn a listing into an exception
        pass


def _error_text(exc: Exception) -> str:
    return str(exc) or "could not resolve this agent's Google identity"


# ── Google Drive file tree ───────────────────────────────────────────

async def build_gdrive_tree(
    folder_id: str,
    folder_name: str,
    max_entries: int = 50,
    max_depth: int = 2,
) -> tuple[str, int]:
    """List a Google Drive folder via the Drive API and return ``(tree_string, entry_count)``.

    Never raises. No usable Google identity → the error text and no listing;
    a folder that fails to list renders an ``[error: ...]`` line in place.
    """
    client = _owner_drive_client()
    lines: list[str] = []
    entry_count = 0

    async def _list_folder(fid: str, depth: int, prefix: str) -> None:
        nonlocal entry_count
        if entry_count >= max_entries:
            return

        try:
            # allDrives: only the id is configured, and the folder may live
            # in a shared drive (the default corpus would read it back empty).
            files, _ = await client.list_folder(
                fid, all_drives=True, max_files=max_entries - entry_count,
            )
        except DriveNotConnected:
            raise  # no identity at all — the whole tree fails closed
        except DriveError as exc:
            lines.append(f"{prefix}[error: {str(exc)[:80]}]")
            return

        for i, f in enumerate(files):
            if entry_count >= max_entries:
                remaining = len(files) - i
                if remaining > 0:
                    lines.append(f"{prefix}... and {remaining} more")
                return

            is_last = i == len(files) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")

            entry_count += 1

            if f.is_folder:
                lines.append(f"{prefix}{connector}{f.name}/ [id:{f.id}]")
                if depth < max_depth:
                    await _list_folder(f.id, depth + 1, child_prefix)
            else:
                size_str = f" ({_format_size(f.size)})" if f.size is not None else ""
                lines.append(f"{prefix}{connector}{f.name}{size_str} [id:{f.id}]")

    try:
        await _list_folder(folder_id, 1, "  ")
    except Exception as exc:  # DriveNotConnected / FD refusal / anything else
        return f"Google Drive: {folder_name} [error: {_error_text(exc)}]", 0
    finally:
        await _close_quietly(client)

    header = f"Google Drive: {folder_name} ({entry_count} entries)"
    if entry_count >= max_entries:
        header += f" [truncated at {max_entries} entries]"

    return header + "\n" + "\n".join(lines), entry_count


# ── Browse GDrive folders (for UI) ───────────────────────────────────

async def browse_gdrive_folders(folder_id: str = "root") -> dict[str, Any]:
    """List subfolders in a Google Drive folder (for the folder picker UI).

    When *folder_id* is ``"root"`` the result also includes any shared drives
    the user has access to (returned in a separate ``shared_drives`` key).

    Returns ``{"folders": [...], "shared_drives": [...]}``, or the same with
    empty lists and an ``"error"`` string. Never raises.
    """
    client = _owner_drive_client()
    try:
        files, _ = await client.list_folder(
            folder_id, all_drives=True, max_files=_BROWSE_MAX_CHILDREN,
        )
        folders = [
            {"id": f.id, "name": f.name}
            for f in files
            if f.is_folder and f.id and f.name
        ]

        # When browsing root, also fetch shared drives.
        shared_drives: list[dict[str, str]] = []
        if folder_id == "root":
            try:
                shared_drives = [
                    {"id": d.id, "name": d.name}
                    for d in await client.list_shared_drives()
                    if d.id and d.name
                ]
            except DriveError as exc:
                log.debug("shared drives listing failed", error=str(exc)[:120])
    except Exception as exc:
        return {"folders": [], "shared_drives": [], "error": _error_text(exc)}
    finally:
        await _close_quietly(client)

    return {"folders": folders, "shared_drives": shared_drives}
