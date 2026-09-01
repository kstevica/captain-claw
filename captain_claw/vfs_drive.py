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
from pathlib import PurePosixPath as PurePosix
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
    materialisation ``state``. ``shared_drive_id`` is set when the mount root
    lives in a Shared (Team) Drive — every listing then needs that drive as its
    corpus, or shared-drive folders read back empty.
    """

    folder_id: str
    dirs: dict[str, str]
    files: dict[str, dict[str, Any]]
    clonemd: bool = False
    shared_drive_id: str = ""

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
            shared_drive_id=str(d.get("shared_drive_id", "")),
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
                    "shared_drive_id": self.shared_drive_id,
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
        + "\n\nThis file lives in Google Drive and has not been downloaded yet. "
        "It IS reachable — the file tools fetch its real content on demand: use "
        "`read` for text/documents, the extract tools (pdf_extract/docx_extract/"
        "xlsx_extract/pptx_extract) for Office files, or the vision tools for "
        "images/video, all on THIS path. Do NOT use the google_drive tool to "
        "re-download it and do NOT treat this marker as the file's content. To "
        "make it a permanent local file, enable clonemd on the folder.\n"
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
        # A corrupt or unexpected file must not crash the read/clone — a bad
        # download degrades to "can't convert" (→ stays a placeholder), never an
        # exception out of the tool.
        if ext == ".pdf":
            md, err = _extract_pdf_markdown(tmp_path, 200)
            return md if md is not None else None
        if ext == ".docx":
            return _extract_docx_markdown(tmp_path)
        if ext == ".xlsx":
            return _extract_xlsx_markdown(tmp_path, 10_000)
        if ext == ".pptx":
            return _extract_pptx_markdown(tmp_path, 500)
    except Exception as exc:
        log.debug("Drive content conversion failed", name=name, ext=ext, error=str(exc))
        return None
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
    folder_id = man.dirs.get(rel)
    if folder_id is None:
        raise ValueError(f"directory {rel!r} is not part of this mount")

    children, truncated = await client.list_folder(
        folder_id, drive_id=man.shared_drive_id,
        max_files=max(1, max_files - len(man.files)),
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


# ---------------------------------------------------------------------------
# clonemd — convert a file to a real local Markdown file
# ---------------------------------------------------------------------------


def _clone_target(rel: str, source_name: str, is_native: bool, claimed: set[str]) -> str:
    """Local path for a cloned file.

    A source that is *itself* plain text keeps its name (copied verbatim). Every
    converted source becomes ``<stem>.md`` — ``report.docx`` → ``report.md``, a
    Google Doc "Report" → ``Report.md``. The verbatim decision is on the SOURCE
    extension, not the export format: a Google Doc exports to markdown but has no
    text extension of its own, so it renames, not passes through.

    Collisions (two sources wanting one ``.md`` name in a directory) fall back to
    keeping the original extension (``report.docx.md``), then to a numeric
    suffix (``Report (2).md``) when even that repeats — e.g. two identically
    named Google Docs.
    """
    p = PurePosix(rel)
    ext = PurePosix(source_name).suffix.lower()
    if not is_native and ext in TEXT_EXTS:
        return rel
    parent = str(p.parent) if str(p.parent) != "." else ""

    def _join(n: str) -> str:
        return f"{parent}/{n}".lstrip("/")

    cand = _join(f"{p.stem}.md")
    if cand not in claimed:
        return cand
    with_ext = _join(f"{p.name}.md")
    if with_ext != cand and with_ext not in claimed:
        return with_ext
    i = 2
    while True:
        numbered = _join(f"{p.stem} ({i}).md")
        if numbered not in claimed:
            return numbered
        i += 1


async def _clone_child(
    client: DriveClient,
    mount: Path,
    rel: str,
    child: DriveFile,
    claimed: set[str],
    sleep,
) -> tuple[dict[str, Any], str] | None:
    """Convert one Drive file to a real local file. Returns ``(entry, disk_rel)``
    or ``None`` if it can't be cloned (binary with no extractor, or a fetch
    error) — the caller then falls back to a placeholder.
    """
    entry = {
        "file_id": child.id,
        "mime": child.mime_type,
        "size": child.size,
        "modified": child.modified_time,
        "md5": child.md5,
    }
    try:
        text = await _fetch_text(client, rel, entry, sleep)
    except DriveError as exc:
        log.debug("clonemd skipped (not text)", rel=rel, error=str(exc))
        return None

    disk_rel = _clone_target(rel, child.name, child.is_google_native, claimed)
    claimed.add(disk_rel)

    target = mount / disk_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    entry["state"] = STATE_CLONED
    entry["cloned_path"] = disk_rel
    entry["cached_modified"] = child.modified_time
    return entry, disk_rel


async def sync(
    client: DriveClient,
    mount: Path,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    progress=None,
) -> dict[str, Any]:
    """Walk the whole mount breadth-first, (re)creating the placeholder tree.

    Bounded by *max_files* across the whole tree, so a huge Drive folder can't
    blow up the mount; the cap is reported, never silently applied. This is what
    a manual refresh runs. Files that vanished upstream are removed locally so a
    mount does not accumulate ghosts.

    *progress*, if given, is called with small dict events as work proceeds
    (``{"phase": "reading", "folders": …, "files": …}`` per folder listed,
    ``{"phase": "cloning", "done": …, "name": …}`` per file converted) so a
    caller can stream "what's happening" to the user while a large folder mounts.
    """
    man = Manifest.load(mount)
    if not man.folder_id:
        raise ValueError("mount has no folder_id — not initialised")

    import asyncio

    sleep = asyncio.sleep
    clonemd = man.clonemd

    # Start clean: rebuild dirs/files from a fresh walk, but keep clonemd flag.
    seen_files: set[str] = set()
    seen_dirs: dict[str, str] = {"": man.folder_id}
    queue: list[str] = [""]
    # Names already taken on disk this pass, so two sources can't collide on one
    # cloned .md name.
    claimed: set[str] = set()
    total_files = 0
    cloned = 0
    truncated = False

    while queue:
        rel = queue.pop(0)
        folder_id = seen_dirs[rel]
        remaining = max_files - total_files
        if remaining <= 0:
            truncated = True
            break
        children, page_truncated = await client.list_folder(
            folder_id, drive_id=man.shared_drive_id, max_files=remaining
        )
        truncated = truncated or page_truncated
        for child in children:
            child_rel = f"{rel}/{child.name}".lstrip("/") if rel else child.name
            local = mount / child_rel
            if child.mime_type == FOLDER_MIME:
                local.mkdir(parents=True, exist_ok=True)
                seen_dirs[child_rel] = child.id
                queue.append(child_rel)
                continue

            total_files += 1
            seen_files.add(child_rel)
            prior = man.files.get(child_rel)
            unchanged = bool(prior and prior.get("modified") == child.modified_time)

            # Unchanged + already cloned: keep the .md, no refetch. Just re-claim
            # its name so nothing else grabs it this pass.
            if clonemd and unchanged and prior and prior.get("state") == STATE_CLONED:
                cp = prior.get("cloned_path", child_rel)
                if (mount / cp).is_file():
                    claimed.add(cp)
                    man.files[child_rel] = prior
                    cloned += 1
                    continue

            # Unchanged + already known unconvertible: keep the placeholder
            # without re-downloading it every refresh just to fail again.
            known_unclonable = bool(prior and prior.get("clonable") is False)
            if clonemd and not (unchanged and known_unclonable):
                result = await _clone_child(client, mount, child_rel, child, claimed, sleep)
                if result is not None:
                    entry, disk_rel = result
                    # A converted file lands under a new name (.md); drop the
                    # original placeholder path so the tree shows one file.
                    if disk_rel != child_rel and local.exists():
                        local.unlink(missing_ok=True)
                    man.files[child_rel] = entry
                    cloned += 1
                    if progress:
                        progress({"phase": "cloning", "done": cloned, "name": child.name})
                    continue
                # Unconvertible (image, archive): placeholder, and remember so a
                # later refresh doesn't re-fetch it.
                _write_placeholder(local, child)
                man.files[child_rel] = {
                    "file_id": child.id, "mime": child.mime_type, "size": child.size,
                    "modified": child.modified_time, "md5": child.md5,
                    "state": STATE_PLACEHOLDER, "clonable": False,
                }
                continue

            # Not cloning (clonemd off, or an unchanged known-unclonable file):
            # keep a materialised prior state when unchanged, else (re)write the
            # marker. Carry the clonable flag forward.
            state = STATE_PLACEHOLDER
            if unchanged and prior:
                state = prior.get("state", STATE_PLACEHOLDER)
            if state == STATE_PLACEHOLDER:
                _write_placeholder(local, child)
            new_entry = {
                "file_id": child.id,
                "mime": child.mime_type,
                "size": child.size,
                "modified": child.modified_time,
                "md5": child.md5,
                "state": state,
            }
            if prior and prior.get("clonable") is False:
                new_entry["clonable"] = False
            if state == STATE_CLONED and prior:
                # keep cloned_path/cached_modified when carrying a cloned entry
                new_entry = prior
            man.files[child_rel] = new_entry

        if progress:
            progress({"phase": "reading", "folders": len(seen_dirs) - 1,
                      "files": total_files})

    _prune_vanished(mount, man, seen_files, seen_dirs)
    man.dirs = seen_dirs
    man.save(mount)
    return {
        "files": total_files,
        "dirs": len(seen_dirs) - 1,
        "cloned": cloned,
        "truncated": truncated,
        "max_files": max_files,
    }


def _prune_vanished(
    mount: Path, man: Manifest, seen_files: set[str], seen_dirs: dict[str, str]
) -> None:
    """Remove local files/dirs no longer present upstream.

    Prunes the placeholder path *and* any cloned output for a source that
    dropped out of the fresh walk — a mount should never keep a Markdown file
    whose Drive original is gone. (A whole unmount is a separate, guarded path.)
    """
    for rel in list(man.files):
        if rel in seen_files:
            continue
        entry = man.files.get(rel) or {}
        for victim in {rel, entry.get("cloned_path", "")}:
            if victim:
                try:
                    (mount / victim).unlink(missing_ok=True)
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
    shared_drive_id: str = "",
    max_files: int = DEFAULT_MAX_FILES,
    progress=None,
) -> dict[str, Any]:
    """Create the placeholder tree and the ``.vfs-links.json`` entry.

    Returns the sync summary. Idempotent on the link entry — re-mounting the
    same project refreshes it. Writing the link is the caller's (FD route's)
    job via :func:`link_entry`, kept separate so this stays FD-free.
    Pass *shared_drive_id* when the root lives in a Shared (Team) Drive.
    *progress* is forwarded to :func:`sync` for live "what's happening" events.
    """
    root = mount_root(user_id, project)
    root.mkdir(parents=True, exist_ok=True)
    man = Manifest.load(root)
    man.folder_id = folder_id
    man.clonemd = clonemd
    man.shared_drive_id = shared_drive_id
    man.dirs.setdefault("", folder_id)
    man.save(root)
    summary = await sync(client, root, max_files=max_files, progress=progress)
    return summary


def link_entry(
    user_id: str, project: str, folder_id: str, *,
    clonemd: bool = False, shared_drive_id: str = "", source_path: str = "",
) -> dict[str, Any]:
    """The ``.vfs-links.json`` value for a Drive mount.

    ``path`` points at the local placeholder tree, so every existing resolver
    (`link_target_at`, `project_root`, `resolve_under`) works unchanged;
    ``mode: "ro"`` makes the write tools refuse it; the ``drive`` block is read
    only by this subsystem. ``source_path`` is the human breadcrumb of the
    mounted Drive folder (e.g. ``FRC3/Reporting/…/VC``), shown as a subtitle so a
    short mount name like ``VC`` isn't ambiguous across many similar folders.
    """
    return {
        "path": str(mount_root(user_id, project)),
        "mode": "ro",
        "kind": "gdrive",
        "drive": {"folder_id": folder_id, "clonemd": bool(clonemd),
                  "shared_drive_id": shared_drive_id, "source_path": source_path,
                  "synced_at": 0},
    }


def _cache_path(mount: Path, file_id: str) -> Path:
    return mount / CACHE_DIRNAME / file_id


async def _fetch_text(client: DriveClient, rel: str, entry: dict[str, Any], sleep) -> str:
    """Fetch and convert one file to readable text. Raises DriveError if it
    can't be read as text (a binary with no extractor) or is too large.

    The single fetch+convert path shared by read-hydration and clonemd, so a
    file is never downloaded twice for the two features and both agree on how a
    given type renders.
    """
    f = _entry_as_file(rel, entry)
    if f.size and f.size > _MAX_FETCH_BYTES:
        raise DriveError(f"{f.name} is too large to fetch ({f.size} bytes)")
    data, ext = await client.fetch(f, sleep=sleep)
    text = bytes_to_text(data, ext, f.name)
    if text is None:
        raise DriveError(
            f"{f.name} is a binary file ({f.mime_type or ext}) and can't be read as text."
        )
    return text


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

    text = await _fetch_text(client, rel, entry, sleep)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    entry["state"] = STATE_HYDRATED
    entry["cached_modified"] = entry.get("modified")
    man.save(mount)
    return text


async def read_through(abs_path: Path, *, client_factory=None) -> str | None:
    """Content for a Drive placeholder at *abs_path*, or ``None`` to read normally.

    The read tool calls this for any ``vfs:`` path. ``None`` means "not a Drive
    placeholder — read the file as usual" (covers non-mount paths and cloned
    files whose real bytes are already on disk). Drive errors surface as a
    readable note rather than an exception, so a read never hard-fails on a
    transient outage — it falls back to the marker with the reason.

    ``client_factory`` (async, returns a DriveClient) supplies a per-user client
    — used by Flight Deck's file routes to hydrate with the MOUNT OWNER's token.
    Agent-side callers pass nothing and get the global connection, which on the
    agent already resolves that agent's own Google account. Built lazily, only
    once the path is confirmed to be a Drive placeholder, so an ordinary read by
    a user with no Google connection never triggers it.
    """
    found = find_mount(abs_path)
    if not found:
        return None
    mount, rel = found
    entry = Manifest.load(mount).files.get(rel)
    if entry is None or entry.get("state") == STATE_CLONED:
        return None  # untracked, or real content already on disk

    from captain_claw.drive_client import make_client

    client = await client_factory() if client_factory else make_client()
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


# ---------------------------------------------------------------------------
# Byte materialisation — the seam for binary readers
# ---------------------------------------------------------------------------


def _blob_cache_path(mount: Path, file_id: str, ext: str) -> Path:
    """On-disk slot for a placeholder's *raw* bytes.

    Distinct from :func:`_cache_path`, which holds converted *text* for the read
    tool: a binary reader (docx/pdf/xlsx/pptx, image_ocr, video_vision, cv) needs
    the original bytes, and the real extension is kept in the name so a tool that
    validates by suffix accepts the returned path.
    """
    safe = ext if ext.startswith(".") else (f".{ext}" if ext else "")
    return mount / CACHE_DIRNAME / f"{file_id}.blob{safe}"


async def materialize(abs_path: Path, *, sleep=None, client_factory=None) -> Path | None:
    """Ensure a Drive placeholder's real bytes are on disk; return a readable path.

    The byte-level sibling of :func:`read_through` (which returns converted
    *text* for the read tool). Any tool that opens a file with a binary reader —
    the document extractors, image_ocr/image_vision, video_vision, cv — calls
    this so a mounted Drive file behaves like a local one: the returned path is a
    real file, holding the actual bytes, carrying the source extension.

    Returns ``None`` when *abs_path* is not a materialisable Drive placeholder —
    an ordinary file, a mount-internal file, or a clonemd file whose real bytes
    are already on disk — so the caller uses the path unchanged (and pays only a
    couple of stat calls for the common non-Drive case). Raises
    :class:`DriveError` when the file is known but its bytes can't be fetched
    (outage, too large, no export path), so the caller surfaces a clear reason
    rather than a downstream "not a valid ZIP"/decode error.

    Cached by Drive ``modifiedTime``: a repeat call while the file is unchanged
    reuses the cached bytes and never touches the network.
    """
    import asyncio

    found = find_mount(abs_path)
    if not found:
        return None
    mount, rel = found
    if rel == MANIFEST_NAME or rel.split("/", 1)[0] == CACHE_DIRNAME:
        return None  # the manifest and the cache dir are ordinary files
    man = Manifest.load(mount)
    entry = man.files.get(rel)
    if entry is None or entry.get("state") == STATE_CLONED:
        return None  # untracked, or real content already on disk (clonemd)

    ext = PurePosix(rel).suffix.lower()
    blob = _blob_cache_path(mount, str(entry["file_id"]), ext)
    if blob.is_file() and entry.get("blob_modified") == entry.get("modified"):
        return blob

    f = _entry_as_file(rel, entry)
    if f.size and f.size > _MAX_FETCH_BYTES:
        raise DriveError(f"{f.name} is too large to fetch ({f.size} bytes).")

    from captain_claw.drive_client import make_client

    client = await client_factory() if client_factory else make_client()
    try:
        data, got_ext = await client.fetch(f, sleep=sleep or asyncio.sleep)
    finally:
        await client.close()

    if not ext and got_ext:
        # An extension-less source (rare for a binary reader) — name the cache by
        # the fetched/export extension so a suffix check still sees the type.
        blob = _blob_cache_path(mount, str(entry["file_id"]), got_ext)
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)
    entry["blob_modified"] = entry.get("modified")
    man.save(mount)
    return blob


def materialize_sync(abs_path: Path) -> Path | None:
    """Blocking :func:`materialize`, for callers running OFF the event loop.

    cv's ops run in a worker thread (``asyncio.to_thread``), where there is no
    running loop and ``await`` is impossible; they use this. It must never be
    called from within a running loop — an async caller uses :func:`materialize`
    directly.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(materialize(abs_path))
    raise RuntimeError(
        "materialize_sync() called from within an event loop — use materialize()"
    )


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
