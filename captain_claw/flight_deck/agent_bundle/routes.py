"""Agent bundles over HTTP: export (and back up) an agent, import one.

    GET    /fd/agent-bundle/review/{slug}           what it learned, per category, to look over first
    POST   /fd/agent-bundle/export/{slug}           make a bundle of one of your agents
    GET    /fd/agent-bundle/exports/{id}/download   fetch it (yours only)
    DELETE /fd/agent-bundle/exports/{id}            throw it away
    POST   /fd/agent-bundle/upload                  upload a .captain file → what is in it
    POST   /fd/agent-bundle/uploads/{id}/import     make a new agent of it — or put it into one of yours
                                                    (target.kind "existing": reset or append)
    GET    /fd/agent-bundle/safety/{id}/download    the backup taken before an import into an agent
    POST   /fd/agent-bundle/safety/{id}/restore     undo that import: the agent as it was
    DELETE /fd/agent-bundle/uploads/{id}            throw the upload away
    GET    /fd/agent-bundle/tiers                   your model tiers (no keys), for an import
    GET    /fd/agent-bundle/packages/{slug}         the packages an agent was exported as (their last version)

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

from captain_claw.flight_deck.agent_bundle import (
    archive,
    catalog,
    export,
    merge,
    provenance,
    restore,
    review,
)
from captain_claw.flight_deck.auth import get_managed_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/agent-bundle", tags=["agent-bundle"])

_ID = re.compile(r"[0-9a-f]{24}")
_KEEP_S = 24 * 3600
_KEEP_SAFETY_S = 7 * 24 * 3600      # the backup taken before an import into an agent


def _owner(user: dict) -> str:
    from captain_claw.flight_deck.auth import _fd_auth_enabled
    return str(user.get("id") or "") if _fd_auth_enabled() else ""


def _sweep() -> None:
    for sub in ("exports", "uploads", "tmp"):
        archive.sweep(export.bundles_dir() / sub, _KEEP_S)
    archive.sweep(export.bundles_dir() / "safety", _KEEP_SAFETY_S)


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
    # The review: insight categories that go (None: all), single insights left
    # out, its contacts (None: only in a full backup).
    insight_categories: list[str] | None = None
    exclude_insights: list[str] = Field(default_factory=list, max_length=20000)
    contacts: bool | None = None
    # A full backup with a passphrase carries the agent's keys, sealed.
    passphrase: str = Field(default="", max_length=500)
    # A named package: this export is its next version (an agent holding an
    # earlier one can be updated from it).
    package: str = Field(default="", max_length=80)


@router.get("/packages/{slug}")
async def agent_packages(slug: str, user: dict = Depends(get_managed_user)) -> dict:
    """The packages this agent was exported as, with their last version."""
    from captain_claw.flight_deck import server as srv

    srv._verify_process_owner(slug, _owner(user))
    return {"packages": provenance.packages(srv.DATA_DIR / slug)}


@router.get("/review/{slug}")
async def review_agent(slug: str, user: dict = Depends(get_managed_user)) -> dict:
    """What the agent learned, per insight category, and its contacts — for the
    person exporting it to decide what goes."""
    from captain_claw.flight_deck import server as srv

    srv._verify_process_owner(slug, _owner(user))
    return await asyncio.to_thread(review.review, srv.DATA_DIR / slug)


@router.post("/export/{slug}")
async def export_agent(slug: str, body: ExportBody, request: Request,
                       user: dict = Depends(get_managed_user)) -> dict:
    """Bundle one of the caller's process agents (Docker agents aren't supported yet)."""
    from captain_claw.flight_deck import server as srv

    owner_id = _owner(user)
    entry = srv._verify_process_owner(slug, owner_id)
    preset = body.preset if body.preset in export.PRESETS else "full"
    categories = set(body.categories) if body.categories else set(export.PRESETS[preset])
    options = export.Options(
        insight_categories=set(body.insight_categories) if body.insight_categories is not None else None,
        exclude_insights=set(body.exclude_insights),
        contacts=(preset == "full") if body.contacts is None else body.contacts,
        passphrase=body.passphrase, package=body.package)
    _sweep()
    try:
        made = await export.build(slug, owner_id, entry, categories=categories, preset=preset,
                                  made_by=str(user.get("display_name") or user.get("email") or ""), options=options)
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
    updates = provenance.updates_for(_own_agents(owner_id), summary["package"]) if summary.get("package") else []
    return {"upload_id": upload_id, "suggested_name": restore.free_name(summary["agent"]["name"]),
            "updates": updates, **summary}


def _own_agents(owner_id: str) -> list[tuple[str, str, Path]]:
    """The caller's process agents: (slug, name, folder)."""
    from captain_claw.flight_deck import server as srv

    return [(slug, str(e.get("name") or slug), srv.DATA_DIR / slug)
            for slug, e in srv._load_process_registry().items()
            if not owner_id or str(e.get("owner") or "") == owner_id]


class ModelChoice(BaseModel):
    mode: str = "keep"          # "keep": its own model · "tier": one of the importer's tiers
    tier: str = ""


class Target(BaseModel):
    kind: str = "new"           # "new": a new agent · "existing": one of the importer's own
    slug: str = ""
    mode: str = "append"        # into an existing one: "reset" or "append"


class ImportBody(BaseModel):
    name: str = ""
    model: ModelChoice = Field(default_factory=ModelChoice)
    target: Target = Field(default_factory=Target)
    # Opens the keys a full backup was sealed with; empty: it comes without them.
    passphrase: str = Field(default="", max_length=500)


@router.get("/tiers")
async def import_tiers(user: dict = Depends(get_managed_user)) -> dict:
    """The caller's model tiers (no keys) — an import may run on one of them."""
    return {"tiers": await restore.tiers(_owner(user))}


@router.post("/uploads/{upload_id}/import")
async def import_bundle(upload_id: str, body: ImportBody, request: Request,
                        user: dict = Depends(get_managed_user)) -> dict:
    """Make a new agent of the upload — its files and memory as they were, its
    scheduled tasks switched off — or, with ``target.kind`` "existing", put it
    into one of the caller's agents (reset or append, see merge.py). The
    upload is removed once it is done."""
    from captain_claw.flight_deck import server as srv

    owner_id = _owner(user)
    path, _ = _record("uploads", upload_id, owner_id)
    try:
        bundle = await asyncio.to_thread(archive.open_bundle, path)
        agent = bundle.read_json("fd/agent.json") or {}
        if not isinstance(agent, dict):
            raise archive.BundleError("The bundle doesn't say which agent it holds.")
        if body.target.kind == "existing":
            keys = await asyncio.to_thread(restore.open_secrets, bundle, body.passphrase)
    except archive.BundleError as exc:
        raise HTTPException(400, str(exc))
    if body.target.kind == "existing":
        out = await _into_existing(path, agent, body.target.slug, body.target.mode, owner_id, keys)
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
        return out
    try:
        name = restore.free_name(body.name.strip() or str(agent.get("name") or "Agent"))
        keep = body.model.mode != "tier"
        config = restore.spawn_config(agent, name, keep_model=keep)
        if not keep:
            await restore.apply_tier(config, owner_id, body.model.tier.strip())
        keys = await asyncio.to_thread(restore.open_secrets, bundle, body.passphrase)
        if keys:
            restore.apply_secrets(config, keys)
    except archive.BundleError as exc:
        raise HTTPException(400, str(exc))
    ref: dict[str, Any] = {"path": str(path), "eco_mode": str(agent.get("eco_mode") or ""),
                           "columns": keys.get("columns") or {} if keys else {},
                           "same_person": str(bundle.manifest.get("owner_ref") or "") == export.person_ref(owner_id)}
    config._bundle = ref
    async with srv._get_spawn_lock():
        result = await srv._spawn_process_locked(config, request, user)
    slug = result.slug
    try:
        extra = await restore.after_spawn(path, slug, name, owner_id, agent)
    except archive.BundleError as exc:
        extra = {"chat_messages": 0, "schedules": 0, "connections": [], "notes": [str(exc)]}
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)
    notes = [*ref.get("notes", []), *extra["notes"]]
    if restore.needs_key(config):
        notes.append(f"It has no key for {config.provider} yet — give it one in its Settings.")
    if extra["schedules"]:
        notes.append(f"{extra['schedules']} scheduled task(s) came with it, switched off — turn them on "
                     "in Scheduled tasks.")
    if bundle.has("secrets.enc") and not keys:
        notes.append("Its keys stayed in the file (no passphrase was given).")
    log.info("agent imported", agent=slug, owner=owner_id or "-", notes=len(notes))
    return {"ok": True, "slug": slug, "name": name, "web_auth": result.web_auth,
            "fleet_instructions": result.fleet_instructions, "chat_messages": extra["chat_messages"],
            "schedules": extra["schedules"], "connections": extra.get("connections", []),
            "keys_restored": bool(keys), "needs_key": restore.needs_key(config), "notes": notes}


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
            "presets": {k: list(v) for k, v in export.PRESETS.items()},
            "insight_categories": [{"id": c, "label": label, "default": on}
                                   for c, label, on in review.INSIGHT_CATEGORIES]}


async def _into_existing(path: Path, agent: dict[str, Any], slug: str, mode: str, owner_id: str,
                         keys: dict[str, Any], *, safety: bool = True) -> dict[str, Any]:
    """The bundle into one of the caller's agents: it is stopped, backed up
    (kept a week, to undo), the bundle reset or appended in, and started again
    if it was running. Its own model and keys stay; the APIs' credentials of a
    sealed backup are put back."""
    import secrets as _secrets

    from captain_claw.flight_deck import server as srv

    entry = srv._verify_process_owner(slug, owner_id)
    if mode not in merge.MODES:
        raise HTTPException(400, "Pick reset, append or update.")
    try:
        manifest = (await asyncio.to_thread(archive.open_bundle, path)).manifest
    except archive.BundleError as exc:
        raise HTTPException(400, str(exc))
    same = str(manifest.get("owner_ref") or "") == export.person_ref(owner_id)
    package = provenance.bundle_package(manifest)
    earlier = provenance.imported(srv.DATA_DIR / slug).get(package["id"]) if package else None
    if mode == "update" and not earlier:
        raise HTTPException(400, "This agent holds no earlier version of that package — add it or replace instead.")
    was_running = srv._process_is_alive(slug)
    if was_running:
        await asyncio.to_thread(srv._do_stop_process, slug)
    backup: dict[str, Any] | None = None
    try:
        if safety:
            try:
                backup = await export.build(slug, owner_id, entry, categories=set(catalog.CATEGORIES), preset="full",
                                            made_by="Taken before an import", keep_in="safety")
            except archive.BundleError as exc:
                raise HTTPException(400, f"Its backup couldn't be taken, so nothing was changed: {exc}")
        try:
            done = await asyncio.to_thread(restore.apply_to_existing, path, srv.DATA_DIR / slug, mode,
                                           (keys or {}).get("columns"), same_person=same, owner_id=owner_id)
            fd = await restore.fd_side_into_existing(path, slug, owner_id, agent, mode, _secrets.token_hex(3),
                                                     earlier=earlier)
        except archive.BundleError as exc:
            raise HTTPException(400, f"{exc} — undo it from its backup.")
    finally:
        if was_running:
            await srv._heal_agent_model_key(slug)
            await srv._prime_deck_routes()
            await asyncio.to_thread(srv._do_start_process, slug)
    log.info("agent bundle imported into an agent", agent=slug, owner=owner_id or "-", mode=mode)
    notes = [*done["notes"], *fd["notes"]]
    if fd["schedules"]:
        notes.append(f"{fd['schedules']} scheduled task(s) came with it, switched off.")
    return {"ok": True, "slug": slug, "name": str(entry.get("name") or slug), "mode": mode, "into": True,
            "counts": restore.headline(done["stores"]),
            "files": {k: done["files"].get(k, 0) for k in ("written", "kept")},
            "chat_messages": fd["chat_messages"], "schedules": fd["schedules"], "connections": fd["connections"],
            "restarted": was_running, "notes": notes, "package": done.get("package"),
            "taken_out": done.get("taken_out") or {}, "from_version": int((earlier or {}).get("version") or 0),
            "safety_backup": {k: backup[k] for k in ("id", "file_name", "size")} if backup else None}


@router.get("/safety/{bundle_id}/download")
async def download_safety(bundle_id: str, user: dict = Depends(get_managed_user)) -> FileResponse:
    path, record = _record("safety", bundle_id, _owner(user))
    return FileResponse(path, media_type="application/zip",
                        filename=str(record.get("file_name") or f"agent{archive.EXTENSION}"))


@router.post("/safety/{bundle_id}/restore")
async def restore_safety(bundle_id: str, user: dict = Depends(get_managed_user)) -> dict:
    """Undo an import into an agent: everything the backup holds back as it was (a reset)."""
    owner_id = _owner(user)
    path, record = _record("safety", bundle_id, owner_id)
    try:
        bundle = await asyncio.to_thread(archive.open_bundle, path)
        agent = bundle.read_json("fd/agent.json") or {}
    except archive.BundleError as exc:
        raise HTTPException(400, str(exc))
    out = await _into_existing(path, agent if isinstance(agent, dict) else {}, str(record.get("slug") or ""),
                               "reset", owner_id, {}, safety=False)
    return {**out, "undone": True}
