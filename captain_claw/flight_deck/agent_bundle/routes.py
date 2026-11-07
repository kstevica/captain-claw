"""Agent bundles over HTTP: export (and back up) an agent, import one.

    POST   /fd/agent-bundle/export/{slug}           make a bundle of one of your agents
    GET    /fd/agent-bundle/exports/{id}/download   fetch it (yours only)
    DELETE /fd/agent-bundle/exports/{id}            throw it away
    POST   /fd/agent-bundle/upload                  upload a .captain file → what is in it
    POST   /fd/agent-bundle/uploads/{id}/import     make a new agent of it
    DELETE /fd/agent-bundle/uploads/{id}            throw the upload away

An admin acting for a user (``X-FD-Act-As``) works on that user's agents, as
on the other agent routes. Made bundles and uploads are swept after a day.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from captain_claw.flight_deck.agent_bundle import archive, catalog, export, restore
from captain_claw.flight_deck.auth import get_managed_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/agent-bundle", tags=["agent-bundle"])

_ID = re.compile(r"[0-9a-f]{24}")
_KEEP_S = 24 * 3600


def _owner(user: dict) -> str:
    from captain_claw.flight_deck.auth import _fd_auth_enabled
    return str(user.get("id") or "") if _fd_auth_enabled() else ""


def _sweep() -> None:
    for sub in ("exports", "uploads", "tmp"):
        archive.sweep(export.bundles_dir() / sub, _KEEP_S)


def _record(sub: str, bundle_id: str, owner_id: str) -> tuple[Path, dict[str, Any]]:
    """A made bundle or an upload, with its record — 404 unless it is *owner_id*'s."""
    if not _ID.fullmatch(bundle_id or ""):
        raise HTTPException(404, "Not found")
    path = export.bundles_dir() / sub / f"{bundle_id}{archive.EXTENSION}"
    try:
        record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(404, "Not found — it may have been swept (they are kept for a day).")
    if not path.is_file() or str(record.get("owner") or "") != owner_id:
        raise HTTPException(404, "Not found")
    return path, record


class ExportBody(BaseModel):
    preset: str = "full"
    categories: list[str] = Field(default_factory=list)


@router.post("/export/{slug}")
async def export_agent(slug: str, body: ExportBody, request: Request,
                       user: dict = Depends(get_managed_user)) -> dict:
    """Bundle one of the caller's process agents (Docker agents aren't supported yet)."""
    from captain_claw.flight_deck import server as srv

    owner_id = _owner(user)
    entry = srv._verify_process_owner(slug, owner_id)
    preset = body.preset if body.preset in export.PRESETS else "full"
    categories = set(body.categories) if body.categories else set(export.PRESETS[preset])
    _sweep()
    try:
        made = await export.build(slug, owner_id, entry, categories=categories, preset=preset,
                                  made_by=str(user.get("display_name") or user.get("email") or ""))
    except archive.BundleError as exc:
        raise HTTPException(400, str(exc))
    log.info("agent exported", agent=slug, owner=owner_id or "-", size=made["size"], preset=preset)
    return made


@router.get("/exports/{bundle_id}/download")
async def download_export(bundle_id: str, user: dict = Depends(get_managed_user)) -> FileResponse:
    path, record = _record("exports", bundle_id, _owner(user))
    return FileResponse(path, media_type="application/zip",
                        filename=str(record.get("file_name") or f"agent{archive.EXTENSION}"))


@router.delete("/exports/{bundle_id}")
async def delete_export(bundle_id: str, user: dict = Depends(get_managed_user)) -> dict:
    path, _ = _record("exports", bundle_id, _owner(user))
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)
    return {"ok": True}


@router.post("/upload")
async def upload_bundle(file: UploadFile = File(...), user: dict = Depends(get_managed_user)) -> dict:
    """Take a .captain file and say what is in it; nothing is made yet."""
    import secrets as _secrets
    import time

    owner_id = _owner(user)
    _sweep()
    upload_id = _secrets.token_hex(12)
    path = export.bundles_dir() / "uploads" / f"{upload_id}{archive.EXTENSION}"
    try:
        await asyncio.to_thread(archive.save_stream, file.file, path)
        summary = await asyncio.to_thread(restore.inspect, path)
    except archive.BundleError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, str(exc))
    path.with_suffix(".json").write_text(json.dumps({
        "id": upload_id, "owner": owner_id, "file_name": str(file.filename or "")[:200], "made": time.time()}),
        encoding="utf-8")
    return {"upload_id": upload_id, "suggested_name": restore.free_name(summary["agent"]["name"]), **summary}


class ImportBody(BaseModel):
    name: str = ""


@router.post("/uploads/{upload_id}/import")
async def import_bundle(upload_id: str, body: ImportBody, request: Request,
                        user: dict = Depends(get_managed_user)) -> dict:
    """Make a new agent of the upload: its files and memory as they were, its
    scheduled tasks switched off. The upload is removed once the agent is made."""
    from captain_claw.flight_deck import server as srv

    owner_id = _owner(user)
    path, _ = _record("uploads", upload_id, owner_id)
    try:
        bundle = await asyncio.to_thread(archive.open_bundle, path)
        agent = bundle.read_json("fd/agent.json") or {}
        if not isinstance(agent, dict):
            raise archive.BundleError("The bundle doesn't say which agent it holds.")
        name = restore.free_name(body.name.strip() or str(agent.get("name") or "Agent"))
        config = restore.spawn_config(agent, name)
    except archive.BundleError as exc:
        raise HTTPException(400, str(exc))
    ref: dict[str, Any] = {"path": str(path), "eco_mode": str(agent.get("eco_mode") or "")}
    config._bundle = ref
    async with srv._get_spawn_lock():
        result = await srv._spawn_process_locked(config, request, user)
    slug = result.slug
    try:
        extra = await restore.after_spawn(path, slug, name, owner_id, agent)
    except archive.BundleError as exc:
        extra = {"chat_messages": 0, "schedules": 0, "notes": [str(exc)]}
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)
    notes = [*ref.get("notes", []), *extra["notes"]]
    if restore.needs_key(config):
        notes.append(f"It has no key for {config.provider} yet — give it one in its Settings.")
    if extra["schedules"]:
        notes.append(f"{extra['schedules']} scheduled task(s) came with it, switched off — turn them on "
                     "in Scheduled tasks.")
    log.info("agent imported", agent=slug, owner=owner_id or "-", notes=len(notes))
    return {"ok": True, "slug": slug, "name": name, "web_auth": result.web_auth,
            "fleet_instructions": result.fleet_instructions, "chat_messages": extra["chat_messages"],
            "schedules": extra["schedules"], "notes": notes}


@router.delete("/uploads/{upload_id}")
async def delete_upload(upload_id: str, user: dict = Depends(get_managed_user)) -> dict:
    path, _ = _record("uploads", upload_id, _owner(user))
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)
    return {"ok": True}


@router.get("/catalog")
async def bundle_catalog(user: dict = Depends(get_managed_user)) -> dict:
    """The categories an export picks from, and the presets."""
    return {"categories": [{"id": c, "label": catalog.CATEGORY_LABELS[c]} for c in catalog.CATEGORIES],
            "presets": {k: list(v) for k, v in export.PRESETS.items()}}
