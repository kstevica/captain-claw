"""HTTP routes for agent-app file uploads / downloads.

Apps that declare ``type: "file"`` inputs (or an ``upload`` surface)
need a place to put bytes. The renderer uploads here, then passes the
returned ``file_id`` into the MCP tool call.

Authentication is via the standard Flight-Deck JWT. The download
endpoint also accepts ``?fd_token=`` so an ``<img src>`` works without
custom headers.
"""

from __future__ import annotations

import mimetypes
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from captain_claw.flight_deck import app_files, app_manifests
from captain_claw.flight_deck.auth import get_current_user


router = APIRouter(prefix="/fd/apps", tags=["apps"])


# Phase 1 ceiling — generous enough for images/PDFs, small enough to
# avoid accidental denial-of-disk. Override via env.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _require_app(agent_id: str) -> None:
    if app_manifests.get(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"No app '{agent_id}'")


@router.post("/{agent_id}/files")
async def upload_file(
    agent_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Upload a single file. Returns the file's metadata."""
    _require_app(agent_id)
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 25MB limit")
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    mime = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"
    meta = app_files.get_store().save(
        agent_id,
        filename=file.filename or "upload",
        mime=mime,
        content=content,
        uploaded_by=str(user.get("id") or user.get("email") or ""),
    )
    return {"file": meta.model_dump()}


@router.get("/{agent_id}/files")
async def list_files(
    agent_id: str,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """List metadata for every file uploaded to this app."""
    _require_app(agent_id)
    metas = app_files.get_store().list(agent_id)
    return {"files": [m.model_dump() for m in metas]}


@router.get("/{agent_id}/files/{file_id}/meta")
async def get_file_meta(
    agent_id: str,
    file_id: str,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    _require_app(agent_id)
    meta = app_files.get_store().get_meta(agent_id, file_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="No such file")
    return {"file": meta.model_dump()}


@router.get("/{agent_id}/files/{file_id}")
async def download_file(
    agent_id: str,
    file_id: str,
    request: Request,
    _user: dict = Depends(get_current_user),
) -> FileResponse:
    """Stream the file bytes. Accepts ``?fd_token=`` for direct browser use."""
    _require_app(agent_id)
    store = app_files.get_store()
    meta = store.get_meta(agent_id, file_id)
    path = store.get_path(agent_id, file_id)
    if meta is None or path is None:
        raise HTTPException(status_code=404, detail="No such file")
    return FileResponse(path, media_type=meta.mime, filename=meta.filename)


@router.delete("/{agent_id}/files/{file_id}")
async def delete_file(
    agent_id: str,
    file_id: str,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    _require_app(agent_id)
    ok = app_files.get_store().delete(agent_id, file_id)
    if not ok:
        raise HTTPException(status_code=404, detail="No such file")
    return {"ok": True}
