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
from captain_claw.drive_client import DriveClient, DriveError, DriveFile, FOLDER_MIME
from captain_claw.logging import get_logger

log = get_logger(__name__)

# Mounts live under a dot-directory at the user root so they never collide with
# a real project name, and are skipped by project enumeration.
MOUNTS_DIRNAME = ".drive"
MANIFEST_NAME = ".drive-manifest.json"
CACHE_DIRNAME = ".drive-cache"  # materialised bytes (Phase 2); created lazily.

DEFAULT_MAX_FILES = 5000

# Extensions read verbatim as text; everything else convertible goes through the
# document extractors, and the rest can't be read as text at all.
TEXT_EXTS = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".html", ".htm", ".log",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".sql", ".go", ".rs", ".java",
    ".c", ".h", ".cpp", ".rb", ".php", ".swift",
})
_CONVERT_EXTS = frozenset({".pdf", ".docx", ".xlsx", ".pptx"})

_MAX_FETCH_BYTES = 50 * 1024 * 1024

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
# Finding a mount from an absolute path (the interception primitive)
# ---------------------------------------------------------------------------


def find_mount(abs_path: Path) -> tuple[Path, str] | None:
    """If *abs_path* lives inside a Drive mount, return ``(mount_root, rel)``.

    Walks up looking for a ``.drive-manifest.json``. This is how the read/grep
    hooks recognise a Drive placeholder without threading mount state through
    every tool — an absolute path is all they have.
    """
    try:
        p = abs_path.resolve()
    except OSError:
        return None
    for anc in [p, *p.parents]:
        if (anc / MANIFEST_NAME).is_file():
            try:
                rel = p.relative_to(anc).as_posix()
            except ValueError:
                return None
            return anc, ("" if rel == "." else rel)
    return None


# ---------------------------------------------------------------------------
# Conversion (shared by read-hydration and, later, clonemd)
# ---------------------------------------------------------------------------


def bytes_to_text(data: bytes, ext: str, name: str) -> str | None:
    """Render fetched bytes as readable text, or ``None`` if it isn't text.

    Text extensions decode directly; Office/PDF go through the same
    ``document_extract`` helpers the extract tools use. Anything else (images,
    archives, unknown binaries) returns ``None`` — the caller explains rather
    than dumping bytes.
    """
    ext = ext.lower()
    if ext in TEXT_EXTS:
        return data.decode("utf-8", errors="replace")
    if ext not in _CONVERT_EXTS:
        return None

    import tempfile

    from captain_claw.tools.document_extract import (
        _extract_docx_markdown,
        _extract_pdf_markdown,
        _extract_pptx_markdown,
        _extract_xlsx_markdown,
    )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        if ext == ".pdf":
            md, err = _extract_pdf_markdown(tmp_path, 200)
            return md if md is not None else f"_(could not extract PDF: {err})_"
        if ext == ".docx":
            return _extract_docx_markdown(tmp_path)
        if ext == ".xlsx":
            return _extract_xlsx_markdown(tmp_path, 10_000)
        if ext == ".pptx":
            return _extract_pptx_markdown(tmp_path, 500)
    finally:
        tmp_path.unlink(missing_ok=True)
    return None


def _entry_as_file(rel: str, entry: dict[str, Any]) -> DriveFile:
    return DriveFile(
        id=str(entry.get("file_id", "")),
        name=Path(rel).name,
        mime_type=str(entry.get("mime", "")),
        size=entry.get("size"),
        modified_time=str(entry.get("modified", "")),
        md5=str(entry.get("md5", "")),
    )


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


def _cache_path(mount: Path, file_id: str) -> Path:
    return mount / CACHE_DIRNAME / file_id


async def hydrate(client: DriveClient, mount: Path, rel: str, *, sleep=None) -> str:
    """Fetch a placeholder's content on demand, cache it, return readable text.

    Idempotent and cheap on repeat: the cached text is reused while the file's
    ``modifiedTime`` is unchanged. Marks the manifest entry ``hydrated`` — the
    on-disk file stays a marker (so the tree is untouched), which is why grep,
    which reads the disk, does not count a hydrated file as searchable.
    """
    import asyncio

    sleep = sleep or asyncio.sleep
    man = Manifest.load(mount)
    entry = man.files.get(rel)
    if entry is None:
        raise ValueError(f"{rel!r} is not part of this mount")

    cache = _cache_path(mount, str(entry["file_id"]))
    if (
        entry.get("state") in (STATE_HYDRATED, STATE_CLONED)
        and entry.get("cached_modified") == entry.get("modified")
        and cache.is_file()
    ):
        return cache.read_text(encoding="utf-8")

    f = _entry_as_file(rel, entry)
    if f.size and f.size > _MAX_FETCH_BYTES:
        raise DriveError(f"{f.name} is too large to fetch ({f.size} bytes)")
    data, ext = await client.fetch(f, sleep=sleep)
    text = bytes_to_text(data, ext, f.name)
    if text is None:
        raise DriveError(
            f"{f.name} is a binary file ({f.mime_type or ext}) and can't be read as text."
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    entry["state"] = STATE_HYDRATED
    entry["cached_modified"] = entry.get("modified")
    man.save(mount)
    return text


async def read_through(abs_path: Path) -> str | None:
    """Content for a Drive placeholder at *abs_path*, or ``None`` to read normally.

    The read tool calls this for any ``vfs:`` path. ``None`` means "not a Drive
    placeholder — read the file as usual" (covers non-mount paths and cloned
    files whose real bytes are already on disk). Drive errors surface as a
    readable note rather than an exception, so a read never hard-fails on a
    transient outage — it falls back to the marker with the reason.
    """
    found = find_mount(abs_path)
    if not found:
        return None
    mount, rel = found
    entry = Manifest.load(mount).files.get(rel)
    if entry is None or entry.get("state") == STATE_CLONED:
        return None  # untracked, or real content already on disk

    from captain_claw.drive_client import make_client

    client = make_client()
    try:
        return await hydrate(client, mount, rel)
    except DriveError as exc:
        marker = ""
        try:
            marker = abs_path.read_text(encoding="utf-8")
        except OSError:
            pass
        return f"[Google Drive — could not fetch this file: {exc}]\n\n{marker}"
    finally:
        await client.close()


def filter_searchable(paths: list[Path]) -> tuple[list[Path], int]:
    """Split *paths* into (searchable, n_skipped_placeholders) for grep.

    A Drive placeholder (or a hydrated-to-cache file) has only a marker on disk,
    so searching it would silently miss content that is really there — grep
    skips it and reports the count instead. Cloned files and non-mount files are
    searchable. Mount-internal files (the manifest, the cache dir) are dropped
    quietly, neither searched nor counted.
    """
    manifests: dict[Path, Manifest] = {}
    searchable: list[Path] = []
    skipped = 0
    for p in paths:
        found = find_mount(p)
        if not found:
            searchable.append(p)
            continue
        mount, rel = found
        if rel == MANIFEST_NAME or rel.split("/", 1)[0] == CACHE_DIRNAME:
            continue
        man = manifests.get(mount)
        if man is None:
            man = manifests[mount] = Manifest.load(mount)
        entry = man.files.get(rel)
        if entry is None or entry.get("state") == STATE_CLONED:
            searchable.append(p)
        else:
            skipped += 1
    return searchable, skipped


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
