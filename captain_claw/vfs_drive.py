"""A Google Drive folder mounted as a read-only VFS folder.

The mount is a **real local directory tree of placeholder files**, registered as
an ordinary ``.vfs-links.json`` link (``kind: "gdrive"``, ``mode: "ro"``).
Because the directories and files are real, `glob`, `ls`, tree-walking, path
resolution and the FD file browser all work with no changes — only *content* is
virtual. A ``.drive-manifest.json`` at the mount root is the authority on what
each placeholder stands for and whether its bytes have been materialised.

This module is agent-importable (no Flight Deck imports), because the read-side
hooks that hydrate a placeholder (Phase 2) run inside the agent. Mount creation
and refresh are driven from the FD routes but the work here is plain filesystem.

Nothing is ever written back to Drive. Nothing is mirrored until asked for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from captain_claw import vfs
from captain_claw.drive_client import DriveClient, DriveFile, FOLDER_MIME
from captain_claw.logging import get_logger

log = get_logger(__name__)

# Mounts live under a dot-directory at the user root so they never collide with
# a real project name, and are skipped by project enumeration.
MOUNTS_DIRNAME = ".drive"
MANIFEST_NAME = ".drive-manifest.json"
CACHE_DIRNAME = ".drive-cache"  # materialised bytes (Phase 2); created lazily.

DEFAULT_MAX_FILES = 5000

# Placeholder state, recorded per file in the manifest. `placeholder` = a marker
# only; `hydrated` = real bytes fetched on demand into the cache; `cloned` =
# converted to a real local file (clonemd). grep skips `placeholder`.
STATE_PLACEHOLDER = "placeholder"
STATE_HYDRATED = "hydrated"
STATE_CLONED = "cloned"


def user_root(user_id: str) -> Path:
    """The user's VFS root, resolved the way the FD routes resolve it.

    There are two implementations of the ``<root>/vfs/<user>`` layout —
    ``vfs.vfs_base()`` (env cascade) and ``vfs_routes`` (``server.DATA_DIR``).
    A mount is created by FD and read by agents, so both must land on the same
    tree; defer to FD's answer when importable (matching the deep-memory fix),
    else the agent-side cascade.
    """
    try:
        from captain_claw.flight_deck.server import DATA_DIR

        return (DATA_DIR / "vfs" / vfs._sanitize(user_id or "", fallback="local")).resolve()
    except Exception:
        return vfs.user_root_of(user_id)


def mount_root(user_id: str, project: str) -> Path:
    """Absolute local path of a mount's placeholder tree."""
    name = vfs._sanitize(project or "", fallback="")
    return (user_root(user_id) / MOUNTS_DIRNAME / name).resolve()


def is_drive_link(entry: dict[str, Any] | None) -> bool:
    return bool(isinstance(entry, dict) and entry.get("kind") == "gdrive")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class Manifest:
    """The record of what a mount's placeholders stand for.

    ``folder_id`` is the Drive root. ``dirs`` maps a relative POSIX directory
    path (``""`` = mount root) to the Drive folder id backing it — needed to
    list a subfolder lazily, since its id is only learned when its parent is
    listed. ``files`` maps a relative POSIX file path to its Drive metadata and
    materialisation ``state``.
    """

    folder_id: str
    dirs: dict[str, str]
    files: dict[str, dict[str, Any]]
    clonemd: bool = False

    @classmethod
    def load(cls, root: Path) -> "Manifest":
        try:
            d = json.loads((root / MANIFEST_NAME).read_text())
        except (OSError, ValueError):
            return cls(folder_id="", dirs={}, files={})
        return cls(
            folder_id=str(d.get("folder_id", "")),
            dirs=dict(d.get("dirs", {}) or {}),
            files=dict(d.get("files", {}) or {}),
            clonemd=bool(d.get("clonemd", False)),
        )

    def save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "folder_id": self.folder_id,
                    "dirs": self.dirs,
                    "files": self.files,
                    "clonemd": self.clonemd,
                },
                indent=2,
                sort_keys=True,
            )
        )


# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------


def _fmt_size(n: int | None) -> str:
    if not n:
        return ""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def placeholder_text(f: DriveFile) -> str:
    """The honest one-line marker a placeholder holds.

    Not zero bytes: anything that reads a placeholder through a path this
    subsystem did *not* intercept then gets an explanation instead of an empty
    file — a legible bug rather than a confusing one.
    """
    bits = [f"⟨Google Drive · not cloned⟩ {f.name}"]
    if f.size:
        bits.append(_fmt_size(f.size))
    if f.modified_time:
        bits.append(f"modified {f.modified_time[:10]}")
    bits.append(f"id {f.id}")
    head = " · ".join(bits)
    return (
        head
        + "\n\nThis file lives in Google Drive and has not been downloaded. "
        "Read this path to fetch it on demand, or enable clonemd on the folder "
        "to convert it to a local Markdown file.\n"
    )


def _write_placeholder(path: Path, f: DriveFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(placeholder_text(f), encoding="utf-8")


def is_placeholder(mount: Path, rel: str) -> bool:
    """True when *rel* is a not-yet-materialised placeholder, per the manifest."""
    entry = Manifest.load(mount).files.get(rel)
    return bool(entry and entry.get("state") == STATE_PLACEHOLDER)


# ---------------------------------------------------------------------------
# Listing / sync
# ---------------------------------------------------------------------------


async def list_dir(
    client: DriveClient,
    mount: Path,
    rel: str,
    *,
    max_files: int = DEFAULT_MAX_FILES,
) -> dict[str, Any]:
    """List one directory of the mount, creating its children as placeholders.

    The lazy primitive: `rel` must already be a known directory (its Drive id
    recorded in the manifest by a prior listing of its parent). Returns a small
    summary; the tree itself is the filesystem.
    """
    man = Manifest.load(mount)
    drive_id = man.dirs.get(rel)
    if drive_id is None:
        raise ValueError(f"directory {rel!r} is not part of this mount")

    children, truncated = await client.list_folder(
        drive_id, max_files=max(1, max_files - len(man.files))
    )
    n_dirs = 0
    n_files = 0
    for child in children:
        child_rel = f"{rel}/{child.name}".lstrip("/") if rel else child.name
        local = mount / child_rel
        if child.mime_type == FOLDER_MIME:
            local.mkdir(parents=True, exist_ok=True)
            man.dirs.setdefault(child_rel, child.id)
            n_dirs += 1
        else:
            # Don't stomp a materialised file back to a placeholder — only
            # (re)write the marker for one that is still a placeholder or new.
            prior = man.files.get(child_rel)
            if prior is None or prior.get("state") == STATE_PLACEHOLDER:
                _write_placeholder(local, child)
                man.files[child_rel] = {
                    "file_id": child.id,
                    "mime": child.mime_type,
                    "size": child.size,
                    "modified": child.modified_time,
                    "md5": child.md5,
                    "state": STATE_PLACEHOLDER,
                }
            n_files += 1
    man.save(mount)
    return {
        "dir": rel or "/",
        "dirs": n_dirs,
        "files": n_files,
        "truncated": truncated,
    }


async def sync(
    client: DriveClient,
    mount: Path,
    *,
    max_files: int = DEFAULT_MAX_FILES,
) -> dict[str, Any]:
    """Walk the whole mount breadth-first, (re)creating the placeholder tree.

    Bounded by *max_files* across the whole tree, so a huge Drive folder can't
    blow up the mount; the cap is reported, never silently applied. This is what
    a manual refresh runs. Files that vanished upstream are removed locally so a
    mount does not accumulate ghosts.
    """
    man = Manifest.load(mount)
    if not man.folder_id:
        raise ValueError("mount has no folder_id — not initialised")

    # Start clean: rebuild dirs/files from a fresh walk, but keep clonemd flag.
    seen_files: set[str] = set()
    seen_dirs: dict[str, str] = {"": man.folder_id}
    queue: list[str] = [""]
    total_files = 0
    truncated = False

    while queue:
        rel = queue.pop(0)
        drive_id = seen_dirs[rel]
        remaining = max_files - total_files
        if remaining <= 0:
            truncated = True
            break
        children, page_truncated = await client.list_folder(drive_id, max_files=remaining)
        truncated = truncated or page_truncated
        for child in children:
            child_rel = f"{rel}/{child.name}".lstrip("/") if rel else child.name
            local = mount / child_rel
            if child.mime_type == FOLDER_MIME:
                local.mkdir(parents=True, exist_ok=True)
                seen_dirs[child_rel] = child.id
                queue.append(child_rel)
            else:
                total_files += 1
                seen_files.add(child_rel)
                # Preserve a materialised state across refresh — a cloned or
                # hydrated file whose bytes are unchanged shouldn't revert to a
                # placeholder. Content-freshness is Phase 2/3; here we keep the
                # prior state unless the modifiedTime moved.
                prior = man.files.get(child_rel)
                state = STATE_PLACEHOLDER
                if prior and prior.get("modified") == child.modified_time:
                    state = prior.get("state", STATE_PLACEHOLDER)
                if state == STATE_PLACEHOLDER:
                    _write_placeholder(local, child)
                man.files[child_rel] = {
                    "file_id": child.id,
                    "mime": child.mime_type,
                    "size": child.size,
                    "modified": child.modified_time,
                    "md5": child.md5,
                    "state": state,
                }

    _prune_vanished(mount, man, seen_files, seen_dirs)
    man.dirs = seen_dirs
    man.save(mount)
    return {
        "files": total_files,
        "dirs": len(seen_dirs) - 1,
        "truncated": truncated,
        "max_files": max_files,
    }


def _prune_vanished(
    mount: Path, man: Manifest, seen_files: set[str], seen_dirs: dict[str, str]
) -> None:
    """Remove local placeholders/dirs no longer present upstream.

    Only removes what this subsystem created and still tracks — a cloned file
    the user may treat as their own is only removed when its upstream source is
    genuinely gone, and the caller (clonemd/refresh) decides that. Here we prune
    tracked placeholders that dropped out of the fresh walk.
    """
    for rel in list(man.files):
        if rel in seen_files:
            continue
        try:
            (mount / rel).unlink(missing_ok=True)
        except OSError:
            pass
        man.files.pop(rel, None)
    # Empty directories that fell out of the tree.
    for rel in sorted(man.dirs, key=len, reverse=True):
        if rel and rel not in seen_dirs:
            d = mount / rel
            try:
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Mount lifecycle
# ---------------------------------------------------------------------------


async def create_mount(
    client: DriveClient,
    user_id: str,
    project: str,
    folder_id: str,
    *,
    clonemd: bool = False,
    max_files: int = DEFAULT_MAX_FILES,
) -> dict[str, Any]:
    """Create the placeholder tree and the ``.vfs-links.json`` entry.

    Returns the sync summary. Idempotent on the link entry — re-mounting the
    same project refreshes it. Writing the link is the caller's (FD route's)
    job via :func:`link_entry`, kept separate so this stays FD-free.
    """
    root = mount_root(user_id, project)
    root.mkdir(parents=True, exist_ok=True)
    man = Manifest.load(root)
    man.folder_id = folder_id
    man.clonemd = clonemd
    man.dirs.setdefault("", folder_id)
    man.save(root)
    summary = await sync(client, root, max_files=max_files)
    return summary


def link_entry(user_id: str, project: str, folder_id: str, *, clonemd: bool = False) -> dict[str, Any]:
    """The ``.vfs-links.json`` value for a Drive mount.

    ``path`` points at the local placeholder tree, so every existing resolver
    (`link_target_at`, `project_root`, `resolve_under`) works unchanged;
    ``mode: "ro"`` makes the write tools refuse it; the ``drive`` block is read
    only by this subsystem.
    """
    return {
        "path": str(mount_root(user_id, project)),
        "mode": "ro",
        "kind": "gdrive",
        "drive": {"folder_id": folder_id, "clonemd": bool(clonemd), "synced_at": 0},
    }


def remove_mount(user_id: str, project: str, *, keep_cloned: bool = True) -> dict[str, Any]:
    """Delete a mount's local tree.

    ``keep_cloned`` protects clonemd output the user may now treat as their own
    files — with it set, only placeholders/cache are removed and cloned Markdown
    stays. The caller removes the link entry separately.
    """
    import shutil

    root = mount_root(user_id, project)
    if not root.exists():
        return {"removed": 0, "kept": 0}
    man = Manifest.load(root)
    removed = kept = 0
    if not keep_cloned:
        shutil.rmtree(root, ignore_errors=True)
        return {"removed": len(man.files), "kept": 0}
    for rel, entry in man.files.items():
        if entry.get("state") == STATE_CLONED:
            kept += 1
            continue
        try:
            (root / rel).unlink(missing_ok=True)
            removed += 1
        except OSError:
            pass
    cache = root / CACHE_DIRNAME
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
    (root / MANIFEST_NAME).unlink(missing_ok=True)
    return {"removed": removed, "kept": kept}
