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

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger
from captain_claw.vfs import META_FILENAME, read_authors, safe_join, safe_name

log = get_logger(__name__)

router = APIRouter(prefix="/fd/vfs", tags=["vfs"])

# Files larger than this aren't inlined for preview — download instead.
_PREVIEW_MAX_BYTES = 1_000_000


# ── path resolution ──────────────────────────────────────────────────

def _user_root(user_id: str) -> Path:
    from captain_claw.flight_deck.server import DATA_DIR
    # fallback must match captain_claw.vfs._DEFAULT_USER ("local")
    return (DATA_DIR / "vfs" / safe_name(user_id, fallback="local")).resolve()


def _project_root(user_id: str, project: str) -> Path:
    return (_user_root(user_id) / safe_name(project, fallback="shared")).resolve()


def _resolve(user_id: str, project: str, path: str) -> Path:
    """Resolve a (project, path) to an absolute path, or 400 on escape."""
    if not project.strip():
        raise HTTPException(400, "project is required")
    target = safe_join(_project_root(user_id, project), path)
    if target is None:
        raise HTTPException(400, "invalid path (escapes project root)")
    return target


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
    return {"projects": out}


@router.get("/list")
async def list_dir(project: str, path: str = "", user: dict = Depends(get_current_user)):
    """List one directory level within a project."""
    proj_root = _project_root(user["id"], project)
    target = _resolve(user["id"], project, path)
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
async def read_file(project: str, path: str, user: dict = Depends(get_current_user)):
    """Return a text file's contents for preview/edit (size-guarded)."""
    target = _resolve(user["id"], project, path)
    if not target.is_file():
        raise HTTPException(404, "file not found")
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
async def download_file(project: str, path: str, user: dict = Depends(get_current_user)):
    """Stream a file as a download."""
    target = _resolve(user["id"], project, path)
    if not target.is_file():
        raise HTTPException(404, "file not found")
    media = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(target, filename=target.name, media_type=media)


# ── write endpoints ──────────────────────────────────────────────────

class WriteBody(BaseModel):
    project: str
    path: str
    content: str = ""


class MkdirBody(BaseModel):
    project: str
    path: str


class RenameBody(BaseModel):
    project: str
    path: str
    to: str  # destination relative path within the same project


@router.post("/write")
async def write_file(body: WriteBody, user: dict = Depends(get_current_user)):
    """Create or overwrite a text file (used by the in-panel editor)."""
    target = _resolve(user["id"], body.project, body.path)
    if target.exists() and target.is_dir():
        raise HTTPException(400, "path is a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body.content, encoding="utf-8")
    return {"ok": True, "size": target.stat().st_size}


@router.post("/mkdir")
async def make_dir(body: MkdirBody, user: dict = Depends(get_current_user)):
    target = _resolve(user["id"], body.project, body.path)
    target.mkdir(parents=True, exist_ok=True)
    return {"ok": True}


@router.post("/rename")
async def rename_entry(body: RenameBody, user: dict = Depends(get_current_user)):
    src = _resolve(user["id"], body.project, body.path)
    dst = _resolve(user["id"], body.project, body.to)
    if not src.exists():
        raise HTTPException(404, "source not found")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return {"ok": True}


@router.delete("/entry")
async def delete_entry(project: str, path: str, recursive: bool = False,
                       user: dict = Depends(get_current_user)):
    target = _resolve(user["id"], project, path)
    if target.resolve() == _project_root(user["id"], project).resolve() and not recursive:
        raise HTTPException(400, "refusing to delete a project root without recursive=true")
    if not target.exists():
        raise HTTPException(404, "not found")
    if target.is_dir():
        if not recursive and any(target.iterdir()):
            raise HTTPException(400, "directory not empty; pass recursive=true")
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"ok": True}


@router.delete("/project")
async def delete_project(project: str, user: dict = Depends(get_current_user)):
    """Remove an entire project namespace."""
    root = _project_root(user["id"], project)
    if root.resolve() == _user_root(user["id"]).resolve():
        raise HTTPException(400, "invalid project")
    if not root.is_dir():
        raise HTTPException(404, "project not found")
    shutil.rmtree(root)
    return {"ok": True}
