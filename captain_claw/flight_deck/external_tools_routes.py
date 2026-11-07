"""Custom tools over HTTP: ``/fd/external-tools``. See ``external_tools.py`` and
docs/external-tools-plan.md.

Phase 0: a person works with their own tools only. Making or publishing one
needs Crew (a tool is code a coding agent writes); running, going back to a kept
version, secrets and deleting don't. The whole feature is off until an admin
turns it on (Admin → Settings → Custom tools).
"""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck import external_tools as xt
from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/external-tools", tags=["external-tools"])

OFF = "Custom tools are off on this deck. An admin turns them on under Admin → Settings."
NEEDS_CREW = ("Making and publishing custom tools needs Crew (coding agents), which your account "
              "isn't approved for. An admin approves it on the Users page.")
DELIVERY_PROJECT = "custom-tools"


async def deck_settings() -> dict[str, bool]:
    """The admin's switches (system config): on/off, and whether sharing a tool
    with everybody waits for an admin."""
    from captain_claw.flight_deck.admin_routes import _load_system_config

    cfg = _load_system_config(await get_db().get_system_setting("fd:system-config"))
    return {
        "enabled": bool(cfg.get("external_tools_enabled")),
        "everybody_needs_approval": bool(cfg.get("external_tools_everybody_approval", True)),
    }


async def _enabled(user: dict = Depends(get_current_user)) -> dict:
    if not (await deck_settings())["enabled"]:
        raise HTTPException(403, OFF)
    return user


async def _can_build(user: dict) -> bool:
    from captain_claw.flight_deck import crew_access, crew_runner

    row = await crew_runner._user_of(user.get("id") or "")
    return bool(row) and crew_access.grant_of(row).allowed


async def _require_builder(user: dict) -> None:
    if not await _can_build(user):
        raise HTTPException(403, NEEDS_CREW)


def _http(exc: xt.ToolError) -> HTTPException:
    return HTTPException(exc.status, exc.message)


def _uid(user: dict) -> str:
    uid = str(user.get("id") or "")
    if not uid:
        raise HTTPException(401, "Sign in first.")
    return uid


def _own(user: dict, slug: str) -> tuple[str, str]:
    """Phase 0: the caller's own tool, or 404 (sharing comes in Phase 3)."""
    uid = _uid(user)
    try:
        slug = xt.check_slug(slug)
        if not xt.exists(uid, slug):
            raise HTTPException(404, f"There's no tool called {slug}.")
    except xt.ToolError as exc:
        raise _http(exc) from None
    return uid, slug


# ── Secrets: each person's own values, sealed in their settings ──────────


async def secret_values(caller: str, owner: str, slug: str) -> dict[str, str]:
    from captain_claw.flight_deck import fd_crypto

    db = get_db()
    out: dict[str, str] = {}
    for s in xt.secret_names(owner, slug):
        token = await db.get_setting(caller, xt.secret_key(owner, slug, s["name"]))
        if not token:
            continue
        try:
            value = fd_crypto.unseal(token)
        except fd_crypto.CryptoUnavailable:
            value = None
        if isinstance(value, str) and value:
            out[s["name"]] = value
    return out


# ── Files in and out ─────────────────────────────────────────────────────


def _file_resolver(uid: str):
    """``vfs:<folder>/<path>`` in the caller's own VFS (Phase 0)."""
    from captain_claw.flight_deck import vfs_routes

    def resolve(ref: str) -> Path:
        ref = str(ref or "").strip()
        if not ref.startswith("vfs:"):
            raise xt.ToolError("files", f"{ref!r}: name a file as vfs:<folder>/<path>.")
        project, _, rel = ref[4:].lstrip("/").partition("/")
        if not project or not rel:
            raise xt.ToolError("files", f"{ref!r}: name a file as vfs:<folder>/<path>.")
        try:
            vfs_routes._refuse_reserved(rel)
            return vfs_routes._resolve(uid, project, rel)
        except HTTPException as exc:
            raise xt.ToolError("files", f"{ref!r}: {exc.detail}") from None

    return resolve


def _deliverer(uid: str, slug: str):
    """A call's files go to the caller's VFS: vfs:custom-tools/<slug>/<day>-<call>/."""
    from captain_claw.flight_deck import vfs_routes

    async def deliver(call_id: str, files: list[tuple[str, Path]]) -> list[dict[str, Any]]:
        folder_rel = f"{slug}/{time.strftime('%Y-%m-%d')}-{call_id}"
        dest = vfs_routes._user_root(uid) / DELIVERY_PROJECT / folder_rel
        out = []

        def copy() -> None:
            for rel, src in files:
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
                out.append({"name": rel, "path": f"vfs:{DELIVERY_PROJECT}/{folder_rel}/{rel}",
                            "size": target.stat().st_size})

        await asyncio.to_thread(copy)
        return out

    return deliver


# ── Routes ───────────────────────────────────────────────────────────────


class CreateBody(BaseModel):
    slug: str = Field(..., max_length=48)
    title: str = Field("", max_length=200)
    description: str = Field("", max_length=2000)


class PublishBody(BaseModel):
    sha: str = Field("", max_length=64)


class LiveBody(BaseModel):
    version: str = Field(..., max_length=12)


class RunBody(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)


class SecretsBody(BaseModel):
    values: dict[str, str]


@router.get("/status")
async def status(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Whether custom tools are on here, and whether this person can make them."""
    settings = await deck_settings()
    return {**settings, "can_build": settings["enabled"] and await _can_build(user)}


@router.get("")
async def list_mine(user: dict = Depends(_enabled)) -> dict[str, Any]:
    try:
        return {"tools": xt.list_tools(_uid(user)), "can_build": await _can_build(user)}
    except xt.ToolError as exc:
        raise _http(exc) from None


@router.post("")
async def create(body: CreateBody, user: dict = Depends(_enabled)) -> dict[str, Any]:
    await _require_builder(user)
    try:
        return await xt.create_tool(_uid(user), body.slug, title=body.title, description=body.description)
    except xt.ToolError as exc:
        raise _http(exc) from None


@router.get("/{slug}")
async def get_one(slug: str, user: dict = Depends(_enabled)) -> dict[str, Any]:
    owner, slug = _own(user, slug)
    out = xt.detail(owner, slug)
    have = await secret_values(_uid(user), owner, slug)
    out["secrets"] = [{**s, "set": s["name"] in have} for s in xt.secret_names(owner, slug)]
    out["can_build"] = await _can_build(user)
    return out


@router.delete("/{slug}")
async def delete(slug: str, user: dict = Depends(_enabled)) -> dict[str, Any]:
    owner, slug = _own(user, slug)
    try:
        await xt.delete_tool(owner, slug)
    except xt.ToolError as exc:
        raise _http(exc) from None
    # The owner's values for it go with it (Phase 3: everyone's who added it).
    db = get_db()
    prefix = xt.secret_key(owner, slug, "")
    for key in await db.get_all_settings(owner):
        if key.startswith(prefix):
            await db.delete_setting(owner, key)
    return {"ok": True}


@router.post("/{slug}/publish", status_code=202)
async def publish(slug: str, body: PublishBody | None = None, user: dict = Depends(_enabled)) -> dict[str, Any]:
    owner, slug = _own(user, slug)
    await _require_builder(user)
    sha = (body.sha if body else "").strip()
    if sha and not re.fullmatch(r"[0-9a-f]{7,64}", sha):
        raise HTTPException(400, "That isn't a commit id.")
    try:
        return xt.start_publish(owner, slug, sha=sha, by=_uid(user),
                                secret_values=await secret_values(_uid(user), owner, slug))
    except xt.ToolError as exc:
        raise _http(exc) from None


@router.get("/{slug}/publish")
async def publish_state(slug: str, user: dict = Depends(_enabled)) -> dict[str, Any]:
    owner, slug = _own(user, slug)
    return {"publish": xt.read_publish(owner, slug)}


@router.post("/{slug}/live")
async def make_live(slug: str, body: LiveBody, user: dict = Depends(_enabled)) -> dict[str, Any]:
    owner, slug = _own(user, slug)
    try:
        return {"live": await xt.make_live(owner, slug, body.version)}
    except xt.ToolError as exc:
        raise _http(exc) from None


@router.post("/{slug}/run")
async def run(slug: str, body: RunBody, user: dict = Depends(_enabled)) -> dict[str, Any]:
    """Run the live version as the signed-in person, with their secrets."""
    owner, slug = _own(user, slug)
    caller = _uid(user)
    try:
        return await xt.run_tool(owner, slug, caller, body.input,
                                 secret_values=await secret_values(caller, owner, slug),
                                 resolve_file=_file_resolver(caller), deliver=_deliverer(caller, slug))
    except xt.ToolError as exc:
        raise _http(exc) from None


@router.get("/{slug}/secrets")
async def get_secrets(slug: str, user: dict = Depends(_enabled)) -> dict[str, Any]:
    owner, slug = _own(user, slug)
    have = await secret_values(_uid(user), owner, slug)
    return {"secrets": [{**s, "set": s["name"] in have} for s in xt.secret_names(owner, slug)]}


@router.put("/{slug}/secrets")
async def put_secrets(slug: str, body: SecretsBody, user: dict = Depends(_enabled)) -> dict[str, Any]:
    """Set (or, with "", clear) this person's values for the tool's secrets. Values
    are sealed and never sent back."""
    from captain_claw.flight_deck import fd_crypto

    owner, slug = _own(user, slug)
    caller = _uid(user)
    names = {s["name"] for s in xt.secret_names(owner, slug)}
    unknown = sorted(set(body.values) - names)
    if unknown:
        raise HTTPException(400, f"This tool doesn't ask for {', '.join(unknown)}.")
    db = get_db()
    to_set: dict[str, str] = {}
    for name, value in body.values.items():
        if len(value) > 8_000:
            raise HTTPException(400, f"{name} is longer than 8,000 characters.")
        key = xt.secret_key(owner, slug, name)
        if value == "":
            await db.delete_setting(caller, key)
            continue
        try:
            to_set[key] = fd_crypto.seal(value)
        except fd_crypto.CryptoUnavailable as exc:
            raise HTTPException(503, f"Secrets can't be kept safely here: {exc}") from None
    if to_set:
        await db.set_settings(caller, to_set)
    have = await secret_values(caller, owner, slug)
    return {"secrets": [{**s, "set": s["name"] in have} for s in xt.secret_names(owner, slug)]}
