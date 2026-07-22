"""Flight Deck HTTP API for browsing the shared cross-agent virtual filesystem.

Serves the same on-disk tree the agents use (``<fd-data>/vfs/<user>/<project>/``)
so the Flight Deck UI can list projects, walk directories, preview/edit text
files, download, and tidy up. All paths are sandboxed to the authenticated
user's VFS root; ``..`` traversal is rejected.
"""

from __future__ import annotations

import mimetypes
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger
import json

from captain_claw.vfs import (
    META_FILENAME,
    link_target_at,
    read_authors,
    read_links_at,
    safe_join,
    safe_name,
)

_LINKS_FILE = ".vfs-links.json"

log = get_logger(__name__)

router = APIRouter(prefix="/fd/vfs", tags=["vfs"])

# Files larger than this aren't inlined for preview — download instead.
_PREVIEW_MAX_BYTES = 1_000_000

# Per-file ceiling for uploads into the VFS — generous for docs/images, small
# enough to avoid an accidental denial-of-disk.
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


# ── path resolution ──────────────────────────────────────────────────

def _user_root(user_id: str) -> Path:
    from captain_claw.flight_deck.server import DATA_DIR
    # fallback must match captain_claw.vfs._DEFAULT_USER ("local")
    return (DATA_DIR / "vfs" / safe_name(user_id, fallback="local")).resolve()


def _project_root(user_id: str, project: str) -> Path:
    """On-disk root for a project — the external path if it's a linked folder."""
    name = safe_name(project, fallback="shared")
    tgt = link_target_at(_user_root(user_id), name)
    if tgt is not None:
        return tgt
    return (_user_root(user_id) / name).resolve()


def _link_entry(user_id: str, project: str) -> dict | None:
    """The registry entry for *project* if it's a linked folder, else None."""
    ent = read_links_at(_user_root(user_id)).get(safe_name(project, fallback=""))
    return ent if isinstance(ent, dict) else None


def _assert_writable(user_id: str, project: str) -> None:
    """Reject mutations on a read-only linked folder."""
    ent = _link_entry(user_id, project)
    if ent and str(ent.get("mode", "rw")).lower() == "ro":
        raise HTTPException(403, "this linked folder is read-only")


def _dm():
    """The deep memory service, imported lazily.

    Every ``on_*`` hook below is fire-and-forget: they no-op unless the project
    opted into indexing, and they swallow their own failures, because a
    Typesense outage must never turn a successful file write into a failed
    request. Imported inside the call to keep this module free of a startup
    dependency on the FD server's import graph.
    """
    from captain_claw.flight_deck import deep_memory_service

    return deep_memory_service


_STATS_SKIP = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build"}


def _stats(root: Path, cap: int = 20000) -> tuple[int, int, float]:
    """(files, bytes, latest_mtime) for a folder — bounded, skipping heavy dirs
    (a linked repo can be huge). Stops counting after *cap* files."""
    files = total = 0
    latest = 0.0
    stack = [root]
    while stack and files < cap:
        try:
            for c in stack.pop().iterdir():
                if c.is_dir():
                    if c.name not in _STATS_SKIP:
                        stack.append(c)
                elif c.is_file():
                    files += 1
                    try:
                        st = c.stat()
                        total += st.st_size
                        latest = max(latest, st.st_mtime)
                    except OSError:
                        pass
                    if files >= cap:
                        break
        except OSError:
            pass
    return files, total, latest


def _resolve(user_id: str, project: str, path: str) -> Path:
    """Resolve a (project, path) to an absolute path, or 400 on escape."""
    if not project.strip():
        raise HTTPException(400, "project is required")
    target = safe_join(_project_root(user_id, project), path)
    if target is None:
        raise HTTPException(400, "invalid path (escapes project root)")
    return target


async def _eff_owner(caller_id: str, project: str, owner: str, *, write: bool) -> str:
    """Resolve which user's VFS root a (project, owner) request targets.

    Own project (owner empty or == caller) → caller. A shared project (owner set
    and differing) → the owner, after verifying a folder share to the caller;
    404 if no share, 403 if a write is attempted with only 'view' permission.
    Folder shares are stored under type 'vfs' (a Code project is the same folder).
    """
    if not owner or owner == caller_id:
        return caller_id
    share = await get_db().get_share_for_grantee("vfs", project, caller_id, owner)
    if not share:
        raise HTTPException(404, "not found")
    if write and (share.get("permission") or "view") != "edit":
        raise HTTPException(403, "read-only shared folder")
    return owner


def _project_origin(name: str) -> tuple[str, str]:
    """Parse a VFS project name into ``(kind, run_id)``.

    Multi-agent runs auto-bind a project named ``<mode>-<session-id[:8]>`` (e.g.
    ``vatra-a925b98c``), so the folder name already carries which run wrote it.
    """
    for kind in ("basna", "vatra", "council"):
        if name.startswith(f"{kind}-"):
            return kind, name[len(kind) + 1:]
    return "", ""


async def _run_titles(user_id: str) -> dict[str, str]:
    """Map ``session-id[:8] -> title`` so a ``<mode>-<sid8>`` project can show the
    human name of the Basna/Vatra run that created it (best-effort)."""
    try:
        rows = await get_db().list_basna_sessions(user_id)
    except Exception:  # noqa: BLE001 — titles are a nicety, never block listing
        return {}
    out: dict[str, str] = {}
    for r in rows:
        sid = str(r.get("id") or "")
        if sid:
            out[sid[:8]] = str(r.get("title") or "")
    return out


def _entry(p: Path, project: str, root: Path) -> dict:
    try:
        st = p.stat()
        is_dir = p.is_dir()
        rel = str(p.relative_to(root))
        return {
            "name": p.name,
            "type": "dir" if is_dir else "file",
            "path": rel,
            "project": project,
            "size": 0 if is_dir else st.st_size,
            "mtime": st.st_mtime,
        }
    except OSError:
        return {"name": p.name, "type": "file", "path": p.name, "project": project, "size": 0, "mtime": 0}


# ── read endpoints ───────────────────────────────────────────────────

@router.get("/projects")
async def list_projects(user: dict = Depends(get_current_user)):
    """List the user's VFS projects with file counts and total bytes."""
    root = _user_root(user["id"])
    out: list[dict] = []
    if root.is_dir():
        for proj in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not proj.is_dir():
                continue
            # Skip FD-managed dot-directories (e.g. .drive mount trees) — they
            # are not projects and back linked mounts listed further down.
            if proj.name.startswith("."):
                continue
            files = 0
            total = 0
            latest = 0.0
            for f in proj.rglob("*"):
                if f.is_file():
                    files += 1
                    try:
                        stt = f.stat()
                        total += stt.st_size
                        latest = max(latest, stt.st_mtime)
                    except OSError:
                        pass
            kind, run_id = _project_origin(proj.name)
            out.append({"name": proj.name, "files": files, "bytes": total, "mtime": latest,
                        "kind": kind, "run_id": run_id})
    titles = await _run_titles(user["id"]) if any(p.get("run_id") for p in out) else {}
    for p in out:
        p["title"] = titles.get(p.get("run_id") or "", "")
    # Linked folders (external dirs) — not physical children of the user root.
    for name, ent in sorted(read_links_at(root).items()):
        is_drive = ent.get("kind") == "gdrive"
        kind = "gdrive" if is_drive else "link"
        drive_meta = dict(ent.get("drive", {})) if is_drive else None
        if is_drive and drive_meta is not None:
            # Enrich with materialisation counts from the manifest so the panel
            # can show "N of M cloned" and warn about un-cloned files.
            try:
                from captain_claw import vfs_drive

                man = vfs_drive.Manifest.load(vfs_drive.mount_root(user["id"], name))
                total_f = len(man.files)
                cloned_f = sum(
                    1 for e in man.files.values() if e.get("state") == "cloned"
                )
                drive_meta["total"] = total_f
                drive_meta["cloned"] = cloned_f
                drive_meta["uncloned"] = total_f - cloned_f
            except Exception:  # noqa: BLE001 — counts are a nicety
                pass
        tgt = link_target_at(root, name)
        if tgt is None or not tgt.is_dir():
            out.append({"name": name, "files": 0, "bytes": 0, "mtime": 0.0,
                        "kind": kind, "run_id": "", "title": "",
                        "link_path": str(ent.get("path", "")), "mode": ent.get("mode", "rw"),
                        "missing": True, "drive": drive_meta})
            continue
        files, total, latest = _stats(tgt)
        out.append({"name": name, "files": files, "bytes": total, "mtime": latest,
                    "kind": kind, "run_id": "", "title": "",
                    "link_path": str(tgt), "mode": ent.get("mode", "rw"),
                    "drive": drive_meta})
    # Folders shared TO this user by other owners. Resolve the run's human title
    # from the *owner's* sessions (a shared vatra-<sid8> folder should show its
    # name, not the raw hash) — cached per owner to avoid repeat queries.
    _title_cache: dict[str, dict[str, str]] = {}
    for s in await get_db().list_shares_for_grantee(user["id"], "vfs"):
        oroot = _user_root(s["owner_id"])
        proj = safe_name(s["resource_id"], fallback="")
        pdir = oroot / proj
        if not proj or not pdir.is_dir():
            continue
        files, total, latest = _stats(pdir)
        _kind, run_id = _project_origin(proj)
        title = ""
        if run_id:
            if s["owner_id"] not in _title_cache:
                _title_cache[s["owner_id"]] = await _run_titles(s["owner_id"])
            title = _title_cache[s["owner_id"]].get(run_id, "")
        out.append({"name": s["resource_id"], "files": files, "bytes": total, "mtime": latest,
                    "kind": "shared", "run_id": run_id, "title": title,
                    "shared": True, "owner_id": s["owner_id"],
                    "owner_email": s.get("owner_email", ""), "owner_name": s.get("owner_name", ""),
                    "permission": s["permission"]})
    return {"projects": out}


class LinkBody(BaseModel):
    name: str
    path: str
    mode: str = "rw"          # "rw" | "ro"


def _write_links(root: Path, links: dict) -> None:
    (root / _LINKS_FILE).write_text(json.dumps(links, indent=2))


@router.get("/links")
async def list_links(user: dict = Depends(get_current_user)):
    """List the user's linked folders."""
    root = _user_root(user["id"])
    links = read_links_at(root)
    out = []
    for name, ent in sorted(links.items()):
        p = Path(str(ent.get("path", ""))).expanduser()
        out.append({"name": name, "path": str(ent.get("path", "")),
                    "mode": ent.get("mode", "rw"), "exists": p.is_dir()})
    return {"links": out}


@router.post("/links")
async def add_link(body: LinkBody, user: dict = Depends(get_current_user)):
    """Link an external absolute directory as a VFS project (no fs symlink)."""
    from captain_claw.flight_deck.server import DATA_DIR
    name = safe_name(body.name, fallback="")
    if not name:
        raise HTTPException(400, "invalid link name")
    p = Path(body.path).expanduser()
    if not p.is_absolute():
        raise HTTPException(400, "path must be absolute")
    if not p.is_dir():
        raise HTTPException(400, "path must be an existing directory")
    p = p.resolve()
    data = Path(DATA_DIR).resolve()
    # Never link the FD data tree itself or an ancestor of it (recursion / self-mount).
    if p == data or data in p.parents or p in data.parents:
        raise HTTPException(400, "cannot link the Flight Deck data directory or its ancestors")
    root = _user_root(user["id"])
    if (root / name).exists():
        raise HTTPException(409, f"a physical project named '{name}' already exists")
    mode = "ro" if str(body.mode).lower() == "ro" else "rw"
    links = read_links_at(root)
    root.mkdir(parents=True, exist_ok=True)
    links[name] = {"path": str(p), "mode": mode}
    _write_links(root, links)
    return {"ok": True, "name": name, "path": str(p), "mode": mode}


@router.get("/browse-fs")
async def browse_fs(path: str = "", user: dict = Depends(get_current_user)):
    """List sub-directories of a local path (for the 'Link folder' picker).

    FD runs on the user's machine, so this browses the same filesystem the user
    would link. Directories only; never returns file contents. Starts at $HOME.
    """
    base = Path(path).expanduser() if path.strip() else Path.home()
    try:
        base = base.resolve()
    except OSError:
        raise HTTPException(400, "invalid path")
    if not base.is_dir():
        raise HTTPException(404, "not a directory")
    dirs: list[dict] = []
    try:
        for c in base.iterdir():
            try:
                if c.is_dir():
                    dirs.append({"name": c.name, "hidden": c.name.startswith("."),
                                 "is_git": (c / ".git").exists()})
            except OSError:
                pass
    except PermissionError:
        raise HTTPException(403, "permission denied")
    dirs.sort(key=lambda d: (d["hidden"], d["name"].lower()))
    parent = str(base.parent) if base.parent != base else ""
    return {"path": str(base), "parent": parent, "dirs": dirs}


@router.delete("/links/{name}")
async def remove_link(name: str, user: dict = Depends(get_current_user)):
    """Unlink a folder — removes only the mapping; never touches the real files."""
    root = _user_root(user["id"])
    key = safe_name(name, fallback="")
    links = read_links_at(root)
    if key in links:
        del links[key]
        _write_links(root, links)
    return {"ok": True}


# ── Google Drive mounts ───────────────────────────────────────────────
# A Drive folder mounted as a read-only VFS folder: a real local tree of
# placeholder files (see captain_claw/vfs_drive). The link entry uses the same
# .vfs-links.json registry, so every resolver treats it as an ordinary folder.


class DriveMountBody(BaseModel):
    name: str
    folder_id: str
    clonemd: bool = False
    drive_id: str = ""  # set when the folder lives in a Shared (Team) Drive


class DriveToggleBody(BaseModel):
    clonemd: bool


def _drive_client():
    """A DriveClient over the deployment's Google connection.

    Per-user tokens are a later swap (the client is provider-agnostic); today
    every mount uses the shared connection, which is correct single-operator.
    """
    from captain_claw.drive_client import make_client

    return make_client()


def _stamp_synced(user_id: str, name: str) -> None:
    """Record the last-synced time on a mount's link entry (best-effort)."""
    import time

    root = _user_root(user_id)
    links = read_links_at(root)
    ent = links.get(name)
    if isinstance(ent, dict) and ent.get("kind") == "gdrive":
        ent.setdefault("drive", {})["synced_at"] = int(time.time())
        links[name] = ent
        _write_links(root, links)


@router.get("/drive/browse")
async def drive_browse(folder_id: str = "root", drive_id: str = "",
                       user: dict = Depends(get_current_user)):
    """List Drive folders under *folder_id*, for the mount picker.

    Folders only — this is a place to choose a mount root, not a file browser.
    At My Drive root (``folder_id=root`` and no *drive_id*) the response also
    carries the account's Shared (Team) Drives so the picker can offer them;
    descending into one passes its id as *drive_id* so the listing uses that
    drive's corpus (the default corpus excludes shared-drive items).
    """
    from captain_claw.drive_client import FOLDER_MIME, DriveError

    client = _drive_client()
    try:
        files, truncated = await client.list_folder(
            folder_id, drive_id=drive_id, max_files=500
        )
        shared = []
        if folder_id == "root" and not drive_id:
            shared = [{"id": d.id, "name": d.name} for d in await client.list_shared_drives()]
    except DriveError as exc:
        raise HTTPException(400, str(exc))
    finally:
        await client.close()
    folders = [
        {"id": f.id, "name": f.name}
        for f in files
        if f.mime_type == FOLDER_MIME
    ]
    return {"folder_id": folder_id, "folders": folders,
            "shared_drives": shared, "truncated": truncated}


@router.post("/links/gdrive")
async def mount_drive(body: DriveMountBody, user: dict = Depends(get_current_user)):
    """Mount a Drive folder as a read-only VFS folder and populate its tree."""
    from captain_claw import vfs_drive
    from captain_claw.drive_client import DriveError

    name = safe_name(body.name, fallback="")
    if not name:
        raise HTTPException(400, "invalid mount name")
    if not body.folder_id.strip():
        raise HTTPException(400, "folder_id is required")
    root = _user_root(user["id"])
    if (root / name).exists():
        raise HTTPException(409, f"a physical project named '{name}' already exists")
    existing = read_links_at(root).get(name)
    if existing and existing.get("kind") != "gdrive":
        raise HTTPException(409, f"a link named '{name}' already exists")

    drive_id = body.drive_id.strip()
    client = _drive_client()
    try:
        summary = await vfs_drive.create_mount(
            client, user["id"], name, body.folder_id.strip(),
            clonemd=body.clonemd, shared_drive_id=drive_id,
        )
    except DriveError as exc:
        raise HTTPException(400, str(exc))
    finally:
        await client.close()

    import time

    links = read_links_at(root)
    root.mkdir(parents=True, exist_ok=True)
    entry = vfs_drive.link_entry(user["id"], name, body.folder_id.strip(),
                                 clonemd=body.clonemd, shared_drive_id=drive_id)
    entry["drive"]["synced_at"] = int(time.time())
    links[name] = entry
    _write_links(root, links)
    return {"ok": True, "name": name, **summary}


@router.post("/links/gdrive/{name}/refresh")
async def refresh_drive(name: str, user: dict = Depends(get_current_user)):
    """Re-walk a mount: new files appear, vanished ones are pruned locally."""
    from captain_claw import vfs_drive
    from captain_claw.drive_client import DriveError

    key = safe_name(name, fallback="")
    ent = read_links_at(_user_root(user["id"])).get(key)
    if not vfs_drive.is_drive_link(ent):
        raise HTTPException(404, "not a Drive mount")
    client = _drive_client()
    try:
        summary = await vfs_drive.sync(client, vfs_drive.mount_root(user["id"], key))
    except (DriveError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    finally:
        await client.close()
    _stamp_synced(user["id"], key)
    return {"ok": True, "name": key, **summary}


@router.post("/links/gdrive/{name}/clonemd")
async def toggle_clonemd(name: str, body: DriveToggleBody,
                         user: dict = Depends(get_current_user)):
    """Turn clonemd on/off for a mount.

    Enabling converts the tree to real Markdown now (a full sync). Disabling
    only sets the flag — existing cloned files are left in place, since the
    user may have come to treat them as their own; unmount removes them.
    """
    from captain_claw import vfs_drive
    from captain_claw.drive_client import DriveError

    root = _user_root(user["id"])
    key = safe_name(name, fallback="")
    links = read_links_at(root)
    ent = links.get(key)
    if not vfs_drive.is_drive_link(ent):
        raise HTTPException(404, "not a Drive mount")
    enabled = bool(body.clonemd)
    ent.setdefault("drive", {})["clonemd"] = enabled
    links[key] = ent
    _write_links(root, links)
    mroot = vfs_drive.mount_root(user["id"], key)
    man = vfs_drive.Manifest.load(mroot)
    man.clonemd = enabled
    man.save(mroot)

    summary: dict = {}
    if enabled:
        client = _drive_client()
        try:
            summary = await vfs_drive.sync(client, mroot)
        except (DriveError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        finally:
            await client.close()
        _stamp_synced(user["id"], key)
    return {"ok": True, "name": key, "clonemd": enabled, **summary}


@router.delete("/links/gdrive/{name}")
async def unmount_drive(name: str, keep_cloned: bool = True,
                        user: dict = Depends(get_current_user)):
    """Remove a Drive mount. Cloned Markdown is kept unless keep_cloned=false."""
    from captain_claw import vfs_drive

    root = _user_root(user["id"])
    key = safe_name(name, fallback="")
    links = read_links_at(root)
    if not vfs_drive.is_drive_link(links.get(key)):
        raise HTTPException(404, "not a Drive mount")
    result = vfs_drive.remove_mount(user["id"], key, keep_cloned=keep_cloned)
    del links[key]
    _write_links(root, links)
    return {"ok": True, "name": key, **result}


@router.get("/list")
async def list_dir(project: str, path: str = "", owner: str = "",
                   user: dict = Depends(get_current_user)):
    """List one directory level within a project."""
    oid = await _eff_owner(user["id"], project, owner, write=False)
    proj_root = _project_root(oid, project)
    target = _resolve(oid, project, path)
    if not target.exists():
        raise HTTPException(404, "not found")
    if not target.is_dir():
        raise HTTPException(400, "not a directory")
    authors = read_authors(proj_root)
    entries = []
    for c in target.iterdir():
        if c.name == META_FILENAME:  # hide the authorship sidecar itself
            continue
        e = _entry(c, project, proj_root)
        a = authors.get(e["path"])
        if a:
            e["author"] = a.get("agent", "")
            e["author_ts"] = a.get("ts", 0)
        entries.append(e)
    entries.sort(key=lambda e: (e["type"] == "file", e["name"].lower()))
    return {"project": project, "path": path, "entries": entries}


@router.get("/read")
async def read_file(project: str, path: str, owner: str = "",
                    user: dict = Depends(get_current_user)):
    """Return a text file's contents for preview/edit (size-guarded)."""
    oid = await _eff_owner(user["id"], project, owner, write=False)
    target = _resolve(oid, project, path)
    if not target.is_file():
        raise HTTPException(404, "file not found")
    # Drive mount: preview the fetched content, not the placeholder marker, so
    # the panel shows the real file just like the read tool does.
    try:
        from captain_claw import vfs_drive

        hydrated = await vfs_drive.read_through(target)
        if hydrated is not None:
            return {"project": project, "path": path, "name": target.name,
                    "size": len(hydrated.encode("utf-8")), "binary": False,
                    "truncated": False, "text": hydrated}
    except Exception as exc:
        log.debug("Drive preview hydration skipped: %s", exc)
    size = target.stat().st_size
    if size > _PREVIEW_MAX_BYTES:
        return {"project": project, "path": path, "name": target.name, "size": size,
                "binary": False, "truncated": True, "text": ""}
    try:
        text = target.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError):
        return {"project": project, "path": path, "name": target.name, "size": size,
                "binary": True, "truncated": False, "text": ""}
    return {"project": project, "path": path, "name": target.name, "size": size,
            "binary": False, "truncated": False, "text": text}


@router.get("/download")
async def download_file(project: str, path: str, owner: str = "",
                        user: dict = Depends(get_current_user)):
    """Stream a file as a download."""
    oid = await _eff_owner(user["id"], project, owner, write=False)
    target = _resolve(oid, project, path)
    if not target.is_file():
        raise HTTPException(404, "file not found")
    media = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(target, filename=target.name, media_type=media)


@router.get("/download-zip")
async def download_zip(project: str, owner: str = "",
                       user: dict = Depends(get_current_user)):
    """Zip an entire project folder and stream it (zip name = project name).

    Skips ``.git`` (internal object store — large and not useful in a zip); keeps
    everything else, including ``.reports`` and ``.code``.
    """
    import io
    import zipfile
    from fastapi.responses import StreamingResponse

    oid = await _eff_owner(user["id"], project, owner, write=False)
    root = _project_root(oid, project)
    if not root.is_dir():
        raise HTTPException(404, "project not found")
    name = safe_name(project, fallback="project")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(root.rglob("*")):
            if not f.is_file() or ".git" in f.relative_to(root).parts:
                continue
            zf.write(f, arcname=f"{name}/{f.relative_to(root).as_posix()}")
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
    )


# ── write endpoints ──────────────────────────────────────────────────

class WriteBody(BaseModel):
    project: str
    path: str
    content: str = ""
    owner: str = ""  # owner id when writing into a shared folder


class MkdirBody(BaseModel):
    project: str
    path: str
    owner: str = ""


class RenameBody(BaseModel):
    project: str
    path: str
    to: str  # destination relative path within the same project
    owner: str = ""


@router.post("/write")
async def write_file(body: WriteBody, user: dict = Depends(get_current_user)):
    """Create or overwrite a text file (used by the in-panel editor)."""
    oid = await _eff_owner(user["id"], body.project, body.owner, write=True)
    _assert_writable(oid, body.project)
    target = _resolve(oid, body.project, body.path)
    if target.exists() and target.is_dir():
        raise HTTPException(400, "path is a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body.content, encoding="utf-8")
    _dm().on_write(oid, body.project, body.path)
    return {"ok": True, "size": target.stat().st_size}


@router.post("/mkdir")
async def make_dir(body: MkdirBody, user: dict = Depends(get_current_user)):
    oid = await _eff_owner(user["id"], body.project, body.owner, write=True)
    _assert_writable(oid, body.project)
    target = _resolve(oid, body.project, body.path)
    target.mkdir(parents=True, exist_ok=True)
    return {"ok": True}


@router.post("/upload")
async def upload_files(
    project: str = Form(...),
    path: str = Form(""),
    owner: str = Form(""),
    files: list[UploadFile] = File(...),
    user: dict = Depends(get_current_user),
):
    """Upload one or more files into ``project/path`` (a directory).

    Each file is stored under its basename — any directory component in the
    client-supplied filename is stripped, so a malicious name can't escape the
    target folder. The destination directory is created if missing.
    """
    oid = await _eff_owner(user["id"], project, owner, write=True)
    _assert_writable(oid, project)
    dest_dir = _resolve(oid, project, path)
    if dest_dir.exists() and not dest_dir.is_dir():
        raise HTTPException(400, "target path is not a directory")
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved: list[dict] = []
    for f in files:
        name = safe_name(Path(f.filename or "").name, fallback="upload")
        content = await f.read()
        if len(content) > _MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"'{name}' exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")
        rel = f"{path}/{name}" if path else name
        target = _resolve(oid, project, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        _dm().on_write(oid, project, rel)
        saved.append({"name": name, "size": len(content)})
    return {"ok": True, "files": saved}


@router.post("/rename")
async def rename_entry(body: RenameBody, user: dict = Depends(get_current_user)):
    oid = await _eff_owner(user["id"], body.project, body.owner, write=True)
    _assert_writable(oid, body.project)
    src = _resolve(oid, body.project, body.path)
    dst = _resolve(oid, body.project, body.to)
    if not src.exists():
        raise HTTPException(404, "source not found")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    _dm().on_rename(oid, body.project, body.path, body.to)
    return {"ok": True}


@router.delete("/entry")
async def delete_entry(project: str, path: str, recursive: bool = False, owner: str = "",
                       user: dict = Depends(get_current_user)):
    oid = await _eff_owner(user["id"], project, owner, write=True)
    _assert_writable(oid, project)
    target = _resolve(oid, project, path)
    # Deleting the root of a linked folder would wipe the user's real directory.
    if _link_entry(oid, project) and target.resolve() == _project_root(oid, project).resolve():
        raise HTTPException(400, "refusing to delete a linked folder's root — unlink it instead")
    if target.resolve() == _project_root(oid, project).resolve() and not recursive:
        raise HTTPException(400, "refusing to delete a project root without recursive=true")
    if not target.exists():
        raise HTTPException(404, "not found")
    is_dir = target.is_dir()
    if is_dir:
        if not recursive and any(target.iterdir()):
            raise HTTPException(400, "directory not empty; pass recursive=true")
        shutil.rmtree(target)
    else:
        target.unlink()
    _dm().on_delete(oid, project, path, is_dir=is_dir)
    return {"ok": True}


@router.delete("/project")
async def delete_project(project: str, user: dict = Depends(get_current_user)):
    """Remove an entire project namespace. A linked folder is only unlinked —
    its real files on disk are never deleted."""
    if _link_entry(user["id"], project):
        return await remove_link(project, user)
    root = _project_root(user["id"], project)
    if root.resolve() == _user_root(user["id"]).resolve():
        raise HTTPException(400, "invalid project")
    if not root.is_dir():
        raise HTTPException(404, "project not found")
    shutil.rmtree(root)
    _dm().on_delete(user["id"], project, "", is_dir=True)
    return {"ok": True}


# ── Shared datastore viewer ──────────────────────────────────────────
#
# A Basna/Vatra run with the shared-datastore option on keeps ONE SQLite store
# at vfs:<project>/.datastore/store.db. These endpoints let the FD UI browse its
# tables/rows/export — reusing the same DatastoreManager the agents write to.
# Resolved under THIS user's VFS root (not the ambient env workers use).

def _vfs_datastore_path(user_id: str, project: str) -> Path:
    return _project_root(user_id, project) / ".datastore" / "store.db"


@router.get("/datastore/{project}/tables")
async def vfs_datastore_tables(project: str, owner: str = "",
                               user: dict = Depends(get_current_user)):
    """List the tables of a folder-bound shared datastore (empty if none yet)."""
    from captain_claw.datastore import get_datastore_manager_at
    oid = await _eff_owner(user["id"], project, owner, write=False)
    db_path = _vfs_datastore_path(oid, project)
    if not db_path.is_file():
        return []
    mgr = get_datastore_manager_at(db_path)
    tables = await mgr.list_tables()
    return [
        {
            "name": t.name,
            "columns": [{"name": c.name, "type": c.col_type, "position": c.position} for c in t.columns],
            "row_count": t.row_count,
            "created_at": t.created_at,
            "updated_at": t.updated_at,
        }
        for t in tables
    ]


@router.get("/datastore/{project}/tables/{table_name}/rows")
async def vfs_datastore_rows(
    project: str, table_name: str,
    limit: int = 100, offset: int = 0,
    order_by: str = "_id", order_dir: str = "ASC",
    owner: str = "",
    user: dict = Depends(get_current_user),
):
    """Paginated rows from a folder-bound datastore table (rows as dicts)."""
    from captain_claw.datastore import get_datastore_manager_at
    oid = await _eff_owner(user["id"], project, owner, write=False)
    db_path = _vfs_datastore_path(oid, project)
    if not db_path.is_file():
        raise HTTPException(404, "no datastore for this folder")
    mgr = get_datastore_manager_at(db_path)
    order_param = ["-" + order_by] if order_dir.upper() == "DESC" else [order_by]
    try:
        result = await mgr.query(
            table_name=table_name, columns=None, where=None,
            order_by=order_param, limit=min(int(limit), 500), offset=int(offset),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc))
    cols = result.get("columns", [])
    result["rows"] = [
        {cols[i]: v for i, v in enumerate(row)}
        for row in result.get("rows", [])
        if isinstance(row, (list, tuple))
    ]
    return result


@router.get("/datastore/{project}/tables/{table_name}/export")
async def vfs_datastore_export(
    project: str, table_name: str, format: str = "csv", owner: str = "",
    user: dict = Depends(get_current_user),
):
    """Export a folder-bound datastore table as csv/json/xlsx."""
    import csv
    import io
    import os
    import tempfile
    from fastapi.responses import Response as FastAPIResponse

    from captain_claw.config import get_config
    from captain_claw.datastore import get_datastore_manager_at
    if format not in ("csv", "json", "xlsx"):
        raise HTTPException(400, f"Unsupported format: {format}")
    oid = await _eff_owner(user["id"], project, owner, write=False)
    db_path = _vfs_datastore_path(oid, project)
    if not db_path.is_file():
        raise HTTPException(404, "no datastore for this folder")
    mgr = get_datastore_manager_at(db_path)
    try:
        result = await mgr.query(table_name, limit=get_config().datastore.max_export_rows, bypass_max=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc))
    columns = result["columns"]
    rows = result["rows"]
    filename = f"{table_name}.{format}"
    disp = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if format == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(columns)
        w.writerows(rows)
        return FastAPIResponse(content=buf.getvalue().encode("utf-8"), media_type="text/csv", headers=disp)
    if format == "json":
        data = [dict(zip(columns, row)) for row in rows]
        body = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        return FastAPIResponse(content=body.encode("utf-8"), media_type="application/json", headers=disp)
    # xlsx
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp.close()
    try:
        mgr._write_xlsx(Path(tmp.name), columns, rows)
        data_bytes = Path(tmp.name).read_bytes()
        return FastAPIResponse(
            content=data_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=disp,
        )
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ── Snapshot-on-start (protect existing files from new runs) ──────────

# Don't duplicate a huge folder into .history on every fresh run — the resumable /
# continue pattern (shared datastore) accumulates on purpose, and the datastore (the
# real ledger) is excluded from the snapshot anyway.
_SNAPSHOT_MAX_BYTES = 100 * 1024 * 1024
_SNAPSHOT_MAX_FILES = 2000


def snapshot_existing_project(user_id: str, project: str, tag: str) -> int:
    """Before a FRESH run writes into an existing, non-empty VFS folder, copy its
    current contents into ``<folder>/.history/<tag>/`` so prior work is never
    overrun. Only physical (non-linked) projects are snapshotted; the datastore
    and prior history are skipped, and a folder over the size cap is left alone
    (logged). Returns the number of entries backed up (0 = nothing / skipped)."""
    root = (_user_root(user_id) / safe_name(project, fallback="")).resolve()
    if not root.is_dir():
        return 0  # brand-new or linked folder → nothing to protect
    entries = [p for p in root.iterdir() if p.name not in (".history", ".datastore")]
    if not entries:
        return 0
    # Size guard — walk once; skip the backup if the folder is large.
    total_bytes = 0
    total_files = 0
    for p in entries:
        walk = p.rglob("*") if p.is_dir() else [p]
        for f in walk:
            if f.is_file():
                total_files += 1
                try:
                    total_bytes += f.stat().st_size
                except OSError:
                    pass
    if total_bytes > _SNAPSHOT_MAX_BYTES or total_files > _SNAPSHOT_MAX_FILES:
        log.info("VFS snapshot skipped — folder too large",
                 project=project, files=total_files, bytes=total_bytes)
        return 0
    dest = root / ".history" / safe_name(tag, fallback="run")
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in entries:
        try:
            if p.is_dir():
                shutil.copytree(p, dest / p.name, dirs_exist_ok=True)
            else:
                shutil.copy2(p, dest / p.name)
            n += 1
        except OSError as e:
            log.warning("VFS snapshot copy failed", path=str(p), error=str(e))
    return n


async def folder_state_manifest(user_id: str, project: str) -> str:
    """A short summary of what a VFS folder ALREADY holds — datastore tables
    (name · rows · columns) and top-level files — to seed the Lead so it plans to
    CONTINUE existing work instead of restarting. Empty string for a new/empty
    folder (nothing to continue)."""
    if not project:
        return ""
    lines: list[str] = []
    db_path = _vfs_datastore_path(user_id, project)
    if db_path.is_file():
        try:
            from captain_claw.datastore import get_datastore_manager_at
            mgr = get_datastore_manager_at(db_path)
            for t in await mgr.list_tables():
                cols = ", ".join(c.name for c in t.columns)
                lines.append(f"  - datastore table `{t.name}` — {t.row_count} row(s): {cols}")
        except Exception as e:  # noqa: BLE001 — manifest is best-effort
            log.debug("folder_state_manifest datastore read failed", error=str(e))
    root = _project_root(user_id, project)
    if root.is_dir():
        try:
            names = sorted(p.name for p in root.iterdir()
                           if p.is_file() and not p.name.startswith("."))
            for name in names[:30]:
                lines.append(f"  - file `{name}`")
            if len(names) > 30:
                lines.append(f"  - …and {len(names) - 30} more file(s)")
        except OSError:
            pass
    if not lines:
        return ""
    return (
        "\n\n## This folder already contains prior work — CONTINUE it, do NOT restart\n"
        "The shared folder/datastore already holds:\n" + "\n".join(lines) +
        "\nPlan only the OUTSTANDING work. Owners must read/query these first and "
        "upsert/append rather than recreate."
    )
