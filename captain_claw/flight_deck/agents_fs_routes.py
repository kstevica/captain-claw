"""Flight Deck HTTP API for inspecting and cleaning up agent subfolders.

Each managed/process agent (and every Basna/Vatra/Dubina sub-agent) gets its
own directory under ``<fd-data>/<slug>/``. Over time the directory tree fills
with *orphaned* folders — agents that were spawned for a one-off run and whose
slug is no longer in the process registry (``.processes.json``). This router
lets the Flight Deck UI enumerate those folders, see their on-disk size and
workspace files, tell whether the agent still exists in the Agent Desktop, and
delete the dead ones.

Admin-only: deleting folders is destructive and is not scoped per-user (the
whole point is to surface orphans that belong to nobody).
"""

from __future__ import annotations

import mimetypes
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.logging import get_logger
from captain_claw.vfs import safe_join

log = get_logger(__name__)

router = APIRouter(prefix="/fd/agentfs", tags=["agentfs"])

# Folders under fd-data that are NOT agent directories — never list or delete.
_RESERVED = {"vfs", "basna_files"}

# Inline text preview cap; larger files must be downloaded.
_PREVIEW_MAX_BYTES = 1_000_000

# Extensions we offer an in-browser text preview for.
_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".json", ".jsonl",
    ".yaml", ".yml", ".csv", ".tsv", ".toml", ".xml",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".htm",
    ".css", ".scss", ".sh", ".bash", ".sql", ".log",
    ".env", ".ini", ".cfg", ".conf", ".rb", ".go", ".rs",
    ".java", ".c", ".cpp", ".h",
}


# ── path resolution ──────────────────────────────────────────────────

def _data_dir() -> Path:
    from captain_claw.flight_deck.server import DATA_DIR
    return DATA_DIR


def _resolve_folder(folder: str) -> Path:
    """Resolve a folder name to a *direct* child of fd-data, or 400/404."""
    if not folder or "/" in folder or "\\" in folder or folder.startswith(".") or folder in _RESERVED:
        raise HTTPException(400, "invalid folder")
    root = _data_dir()
    target = (root / folder).resolve()
    if target.parent != root.resolve():
        raise HTTPException(400, "invalid folder (escapes data dir)")
    if not target.is_dir():
        raise HTTPException(404, "folder not found")
    return target


def _resolve_file(folder: str, path: str) -> Path:
    """Resolve a workspace-relative file path under an agent folder."""
    base = _resolve_folder(folder) / "data" / "workspace"
    target = safe_join(base, path)
    if target is None:
        raise HTTPException(400, "invalid path (escapes workspace)")
    if not target.is_file():
        raise HTTPException(404, "file not found")
    return target


def _folder_stats(p: Path) -> tuple[int, int, float]:
    """Return (total_bytes, file_count, latest_mtime) for a directory tree."""
    total = 0
    files = 0
    latest = 0.0
    for f in p.rglob("*"):
        if f.is_file():
            files += 1
            try:
                st = f.stat()
                total += st.st_size
                latest = max(latest, st.st_mtime)
            except OSError:
                pass
    return total, files, latest


def _scan_workspace(folder_dir: Path) -> list[dict]:
    """List every file under ``<folder>/data/workspace`` (recursive)."""
    workspace = folder_dir / "data" / "workspace"
    out: list[dict] = []
    if not workspace.is_dir():
        return out
    for f in workspace.rglob("*"):
        if not f.is_file():
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        ext = f.suffix.lower()
        out.append({
            "path": str(f.relative_to(workspace)),
            "name": f.name,
            "size": st.st_size,
            "mtime": st.st_mtime,
            "ext": ext,
            "is_text": ext in _TEXT_EXTS,
        })
    out.sort(key=lambda e: e["path"].lower())
    return out


# ── endpoints ────────────────────────────────────────────────────────

@router.get("/folders")
async def list_folders(user: dict = Depends(require_admin)):
    """List every agent subfolder in fd-data with size + desktop presence.

    ``registered`` means the slug is still in the process registry (i.e. the
    agent exists in the Agent Desktop). ``orphaned`` is the inverse — a folder
    left behind by an agent that no longer exists.
    """
    from captain_claw.flight_deck.server import _load_process_registry, _process_is_alive

    root = _data_dir()
    registry = _load_process_registry()
    out: list[dict] = []
    if root.is_dir():
        for d in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not d.is_dir() or d.name.startswith(".") or d.name in _RESERVED:
                continue
            slug = d.name
            entry = registry.get(slug)
            total, files, latest = _folder_stats(d)
            ws_files = sum(1 for f in (d / "data" / "workspace").rglob("*")
                           if f.is_file()) if (d / "data" / "workspace").is_dir() else 0
            out.append({
                "name": slug,
                "bytes": total,
                "files": files,
                "workspace_files": ws_files,
                "mtime": latest,
                "registered": entry is not None,
                "running": _process_is_alive(slug) if entry is not None else False,
                "display_name": entry.get("name", slug) if entry else "",
                "owner": entry.get("owner", "") if entry else "",
            })
    return {"folders": out}


@router.get("/files")
async def list_files(folder: str, user: dict = Depends(require_admin)):
    """List the workspace files for one agent folder."""
    folder_dir = _resolve_folder(folder)
    return {"folder": folder, "files": _scan_workspace(folder_dir)}


@router.get("/view")
async def view_file(folder: str, path: str, user: dict = Depends(require_admin)):
    """Return a workspace text file's contents for inline preview."""
    target = _resolve_file(folder, path)
    size = target.stat().st_size
    if size > _PREVIEW_MAX_BYTES:
        return {"folder": folder, "path": path, "name": target.name, "size": size,
                "binary": False, "truncated": True, "text": ""}
    try:
        text = target.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError):
        return {"folder": folder, "path": path, "name": target.name, "size": size,
                "binary": True, "truncated": False, "text": ""}
    return {"folder": folder, "path": path, "name": target.name, "size": size,
            "binary": False, "truncated": False, "text": text}


@router.get("/download")
async def download_file(folder: str, path: str, user: dict = Depends(require_admin)):
    """Stream a workspace file as a download."""
    target = _resolve_file(folder, path)
    media = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(target, filename=target.name, media_type=media)


@router.delete("/folder")
async def delete_folder(folder: str, user: dict = Depends(require_admin)):
    """Delete an agent's entire subfolder.

    Refuses to delete a folder whose agent is still running — stop it first.
    """
    from captain_claw.flight_deck.server import _load_process_registry, _process_is_alive

    target = _resolve_folder(folder)
    registry = _load_process_registry()
    if folder in registry and _process_is_alive(folder):
        raise HTTPException(409, "agent is still running; stop it before deleting its folder")
    shutil.rmtree(target)
    log.info("agent folder deleted", folder=folder, by=user.get("id", ""))
    return {"ok": True}
