"""Member routes for a shared agent's files and data (PR C) — see ``shared_workspace``.

Browser (JWT — Authorization or ``fd_token``; plain ``get_current_user``, an
admin's act-as header doesn't reach these). The agent is named ONLY by
``?ref=``; any host, port or token the browser sends is ignored.

* ``GET  /fd/shared-agents/files?ref=`` — every file in the agent's ``saved/``
  commons with its creator, plus the upload limits;
* ``GET  /fd/shared-agents/files/view?ref=&id=`` / ``…/download`` — one file,
  inline (sandboxed, only media keep their type) or as an attachment;
* ``POST /fd/shared-agents/files/upload?ref=&lane=`` (multipart ``file``) —
  into the member's own ``saved/downloads/<their session>/``;
* ``POST /fd/shared-agents/files/delete?ref=`` ``{"id"}`` — one of their own;
* ``GET  /fd/shared-agents/datastore/tables?ref=``,
  ``…/tables/{name}/rows``, ``…/tables/{name}/export`` — the datastore,
  read-only (members change data through chat).

Each re-checks membership (uploads and deletes straight from the DB) and calls
the agent's ``/api/speaker/*`` route with a fresh signed assertion
(``shared_workspace.call_agent``); the agent decides who may change what.
Responses carry ids relative to ``saved/`` — never a host, port, token, other
members' ids or emails. Uploads and deletes leave a ``usage`` row.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from starlette.datastructures import UploadFile
from starlette.responses import Response

from captain_claw.flight_deck import agent_sharing as sharing
from captain_claw.flight_deck import shared_workspace as sw
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["shared-workspace"])

_SYSTEM_KEYS = ("_created_by", "_created_by_name")
_ROWS_LIMIT_MAX = 500
_ROWS_OFFSET_MAX = 10_000_000
_READ_CHUNK = 1 << 20


class FileIdBody(BaseModel):
    id: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────


def _json(resp) -> object:
    try:
        return resp.json()
    except Exception:
        raise HTTPException(502, sw.AGENT_ERROR_DETAIL) from None


async def _member_file(db, uid: str, rec, item: object, cache: dict) -> dict | None:
    """``MemberFile`` of one agent ``AgentFile``; None for anything malformed."""
    if not isinstance(item, dict) or not sw.valid_file_id(item.get("id")):
        return None
    fid = item["id"]
    created_by = await sw.member_creator(db, uid, rec, item.get("created_by"), cache)
    return {
        "id": fid,
        "path": "saved/" + fid,
        "filename": item.get("filename"),
        "extension": item.get("extension"),
        "size": item.get("size"),
        "modified": item.get("modified"),
        "mime_type": item.get("mime_type"),
        "is_text": item.get("is_text"),
        "created_by": created_by,
        "can_delete": created_by["kind"] == "me",
    }


async def _log_usage(db, uid: str, event: str, detail: dict) -> None:
    try:
        await db.log_usage(uid, event, json.dumps(detail))
    except Exception as exc:
        log.warning("Could not record a shared-workspace usage row", error=type(exc).__name__)


def _check_table(name: str) -> None:
    if not sw.valid_table(name):
        raise HTTPException(400, sw.BAD_TABLE_DETAIL)


# ── Files ─────────────────────────────────────────────────────────────────


@router.get("/fd/shared-agents/files")
async def list_files(ref: str = "", user: dict = Depends(get_current_user)) -> dict:
    db = get_db()
    uid = str(user["id"])
    rec = await sw.member_target(db, user, ref, fresh=False)
    resp = await sw.call_agent(db, user, rec, "GET", "/api/speaker/files")
    body = _json(resp)
    if not isinstance(body, dict) or not isinstance(body.get("files"), list):
        raise HTTPException(502, sw.AGENT_ERROR_DETAIL)
    cache: dict = {}
    files = []
    for item in body["files"]:
        entry = await _member_file(db, uid, rec, item, cache)
        if entry is not None:
            files.append(entry)
    return {"files": files, "truncated": bool(body.get("truncated")),
            "upload": {"max_bytes": sw.MEMBER_UPLOAD_MAX_BYTES,
                       "extensions": sorted(sw.MEMBER_UPLOAD_EXTENSIONS)}}


async def _raw(user: dict, ref: str, file_id: str):
    if not sw.valid_file_id(file_id):
        raise HTTPException(400, sw.BAD_ID_DETAIL)
    db = get_db()
    rec = await sw.member_target(db, user, ref, fresh=False)
    resp = await sw.call_agent(db, user, rec, "GET", "/api/speaker/files/raw",
                               params={"id": file_id}, timeout=sw.AGENT_TRANSFER_TIMEOUT_S)
    if len(resp.content) > sw.MEMBER_DOWNLOAD_MAX_BYTES:
        raise HTTPException(413, sw.DOWNLOAD_TOO_LARGE_DETAIL)
    return resp.content, file_id.rsplit("/", 1)[-1]


@router.get("/fd/shared-agents/files/view")
async def view_file(ref: str = "", file_id: str = Query("", alias="id"),
                    user: dict = Depends(get_current_user)) -> Response:
    content, filename = await _raw(user, ref, file_id)
    media, headers = sw.view_headers(filename)
    return Response(content=content, media_type=media, headers=headers)


@router.get("/fd/shared-agents/files/download")
async def download_file(ref: str = "", file_id: str = Query("", alias="id"),
                        user: dict = Depends(get_current_user)) -> Response:
    content, filename = await _raw(user, ref, file_id)
    return Response(content=content, headers=sw.download_headers(filename))


@router.post("/fd/shared-agents/files/upload")
async def upload_file(request: Request, ref: str = "", lane: str = "A",
                      user: dict = Depends(get_current_user)) -> dict:
    """Into the member's own folder. The size and the membership are checked
    before the body is touched (no ``UploadFile`` parameter: that would make
    Starlette read and spool the whole body before this runs)."""
    raw_length = request.headers.get("content-length")
    try:
        length = int(raw_length) if raw_length is not None else -1
    except ValueError:
        length = -1
    if length < 0:
        raise HTTPException(411, sw.LENGTH_DETAIL)
    if length > sw.MEMBER_UPLOAD_MAX_BYTES + sw.UPLOAD_BODY_SLACK:
        raise HTTPException(413, sw.TOO_LARGE_DETAIL)
    lane = str(lane or "").strip().upper()
    if lane not in sharing.MEMBER_LANES:
        raise HTTPException(400, sw.BAD_LANE_DETAIL)
    db = get_db()
    uid = str(user["id"])
    rec = await sw.member_target(db, user, ref, fresh=True)

    form = await request.form(max_files=1, max_fields=2)
    try:
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(400, sw.EMPTY_DETAIL)
        name = PurePosixPath((file.filename or "").replace("\\", "/")).name[:200]
        if PurePosixPath(name).suffix.lower() not in sw.MEMBER_UPLOAD_EXTENSIONS:
            raise HTTPException(400, sw.BAD_TYPE_DETAIL)
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = await file.read(_READ_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > sw.MEMBER_UPLOAD_MAX_BYTES:
                raise HTTPException(413, sw.TOO_LARGE_DETAIL)
            chunks.append(chunk)
        if size == 0:
            raise HTTPException(400, sw.EMPTY_DETAIL)
        data = b"".join(chunks)
    finally:
        await form.close()

    resp = await sw.call_agent(
        db, user, rec, "POST", "/api/speaker/files/upload", lane=lane,
        files={"file": (name, data, "application/octet-stream")},
        timeout=sw.AGENT_TRANSFER_TIMEOUT_S)
    entry = await _member_file(db, uid, rec, _json(resp), {})
    if entry is None:
        raise HTTPException(502, sw.AGENT_ERROR_DETAIL)
    await _log_usage(db, uid, "shared_agent_file_upload",
                     {"agent_ref": ref, "filename": name, "size": len(data)})
    return entry


@router.post("/fd/shared-agents/files/delete")
async def delete_file(body: FileIdBody, ref: str = "",
                      user: dict = Depends(get_current_user)) -> dict:
    """One of the member's own files (the agent refuses anyone else's)."""
    if not sw.valid_file_id(body.id):
        raise HTTPException(400, sw.BAD_ID_DETAIL)
    db = get_db()
    uid = str(user["id"])
    rec = await sw.member_target(db, user, ref, fresh=True)
    await sw.call_agent(db, user, rec, "POST", "/api/speaker/files/delete",
                        json_body={"id": body.id})
    await _log_usage(db, uid, "shared_agent_file_delete", {"agent_ref": ref, "id": body.id})
    return {"ok": True}


# ── Datastore (read-only) ─────────────────────────────────────────────────


@router.get("/fd/shared-agents/datastore/tables")
async def list_tables(ref: str = "", user: dict = Depends(get_current_user)) -> list:
    db = get_db()
    uid = str(user["id"])
    rec = await sw.member_target(db, user, ref, fresh=False)
    resp = await sw.call_agent(db, user, rec, "GET", "/api/speaker/datastore/tables")
    body = _json(resp)
    if not isinstance(body, dict) or not isinstance(body.get("tables"), list):
        raise HTTPException(502, sw.AGENT_ERROR_DETAIL)
    cache: dict = {}
    tables = []
    for t in body["tables"]:
        if not isinstance(t, dict) or not sw.valid_table(t.get("name")):
            continue
        tables.append({
            "name": t["name"],
            "columns": t.get("columns"),
            "row_count": t.get("row_count"),
            "created_at": t.get("created_at"),
            "updated_at": t.get("updated_at"),
            "created_by": await sw.member_creator(db, uid, rec, t.get("created_by"), cache),
        })
    return tables


@router.get("/fd/shared-agents/datastore/tables/{name}/rows")
async def table_rows(name: str, ref: str = "", limit: int = 100, offset: int = 0,
                     order_by: str = "_id", order_dir: str = "asc",
                     user: dict = Depends(get_current_user)) -> dict:
    _check_table(name)
    # A bad sort column or direction reads as a bad table request (part 1 §2).
    if not sw.valid_order_by(order_by):
        raise HTTPException(400, sw.BAD_TABLE_DETAIL)
    order_dir = str(order_dir or "").lower()
    if order_dir not in ("asc", "desc"):
        raise HTTPException(400, sw.BAD_TABLE_DETAIL)
    limit = max(1, min(int(limit), _ROWS_LIMIT_MAX))
    offset = max(0, min(int(offset), _ROWS_OFFSET_MAX))
    db = get_db()
    uid = str(user["id"])
    rec = await sw.member_target(db, user, ref, fresh=False)
    resp = await sw.call_agent(
        db, user, rec, "GET", f"/api/speaker/datastore/tables/{name}/rows",
        params={"limit": limit, "offset": offset, "order_by": order_by,
                "order_dir": order_dir})
    body = _json(resp)
    if not isinstance(body, dict) or not isinstance(body.get("rows"), list):
        raise HTTPException(502, sw.AGENT_ERROR_DETAIL)
    columns = body.get("columns")
    if isinstance(columns, list):
        columns = [c for c in columns if c not in _SYSTEM_KEYS]
    cache: dict = {}
    rows = []
    for row in body["rows"]:
        if not isinstance(row, dict):
            continue
        out = {k: v for k, v in row.items() if k not in _SYSTEM_KEYS and k != "_creator"}
        out["_creator"] = await sw.member_creator(db, uid, rec, row.get("_creator"), cache)
        rows.append(out)
    return {"columns": columns, "rows": rows, "total": body.get("total"),
            "offset": body.get("offset"), "limit": body.get("limit")}


@router.get("/fd/shared-agents/datastore/tables/{name}/export")
async def table_export(name: str, ref: str = "", fmt: str = Query("csv", alias="format"),
                       user: dict = Depends(get_current_user)) -> Response:
    _check_table(name)
    if fmt not in sw.EXPORT_FORMATS:
        raise HTTPException(400, sw.BAD_FORMAT_DETAIL)
    db = get_db()
    rec = await sw.member_target(db, user, ref, fresh=False)
    resp = await sw.call_agent(
        db, user, rec, "GET", f"/api/speaker/datastore/tables/{name}/export",
        params={"format": fmt}, timeout=sw.AGENT_TRANSFER_TIMEOUT_S)
    media, headers = sw.export_headers(name, fmt)
    return Response(content=resp.content, media_type=media, headers=headers)
