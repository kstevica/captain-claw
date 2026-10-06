"""Context-pack routes (PR B) — see ``context_packs``.

UI (JWT, plain ``get_current_user`` — an admin's act-as header doesn't reach
these):

* ``GET /fd/context-packs?agent_ref=`` — what is shared on one agent I own or
  am a member of, what I share there, and what I could share;
* ``POST /fd/context-packs`` — publish one of my own resources there;
* ``DELETE /fd/context-packs/{pack_id}`` — stop sharing (the publisher, or
  the agent's owner for anyone's pack); works with sharing off;
* ``GET /fd/context-packs/mine`` — everything I share, on every agent; works
  with sharing off.

Agent (no JWT): ``POST /fd/context-packs/agent/vfs`` — the roots of the
shared folders a tool call names, computed live (grant-aware: a member's turn
sends its grant). Never returns or logs a token; roots go only to the agent.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from captain_claw.flight_deck import agent_sharing as sharing
from captain_claw.flight_deck import context_packs as cp
from captain_claw.flight_deck import speaker_grants, tenant_profile
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["context-packs"])

_ALIAS_UNIQUE = "UNIQUE constraint failed: context_packs.agent_ref, context_packs.alias"


class PackCreate(BaseModel):
    agent_ref: str
    kind: str
    project: str = ""
    alias: str = ""          # full alias "<prefix>-<name>" or "" to derive
    tags: list[str] = []


class VfsResolveBody(BaseModel):
    aliases: list[str] = []


# ── Helpers ───────────────────────────────────────────────────────────────


async def _role(db, uid: str, ref: str) -> tuple[str, sharing.AgentRecord]:
    """("owner" | "member", the agent) for a caller who may see its packs;
    400 for a malformed ref, 404 for anything else."""
    try:
        sharing.parse_ref(ref)
    except ValueError:
        raise HTTPException(400, "Invalid agent reference") from None
    rec = await asyncio.to_thread(sharing.resolve_agent_record, ref)
    if rec is None or sharing.check_shareable(rec, rec.owner):
        raise HTTPException(404, cp.AGENT_NOT_FOUND)
    if uid == rec.owner:
        return "owner", rec
    if await sharing.member_check(db, ref, rec.owner, uid, max_age=0):
        return "member", rec
    raise HTTPException(404, cp.AGENT_NOT_FOUND)


def _check_aliases(body: VfsResolveBody) -> list[str]:
    aliases = list(body.aliases or [])
    if len(aliases) > cp.VFS_RESOLVE_MAX_ALIASES or not all(cp.valid_alias(a) for a in aliases):
        raise HTTPException(400, "Invalid aliases")
    return aliases


def _row_tags(row: dict) -> list[str]:
    try:
        data = json.loads(str(row.get("slice") or "{}"))
        tags = data.get("tags") if isinstance(data, dict) else None
        return [t for t in (tags or []) if isinstance(t, str)]
    except (ValueError, TypeError):
        return []


def _pack_row(*, pack_id: str, kind: str, pack_owner: str, owner_name: str, project: str,
              alias: str, tags, created_at: str, uid: str, agent_owner: str,
              owner_tag: str = "") -> dict:
    """The UI's PackRow. ``owner_id`` only on the caller's own rows and for the
    agent's owner (members never learn other members' ids); never a root,
    path, key, token or email. ``owner_tag`` is the collision tag the agent's
    prompt gives this publisher (``ActivePack.tag``: set only when two
    publishers on the agent share a name, else "")."""
    mine = pack_owner == uid
    is_agent_owner = bool(agent_owner) and uid == agent_owner
    return {
        "id": pack_id,
        "kind": kind,
        "owner_id": pack_owner if (mine or is_agent_owner) else "",
        "owner_name": owner_name or "",
        "role": "owner" if pack_owner == agent_owner else "member",
        "owner_tag": owner_tag or "",
        "mine": mine,
        "project": project if kind == "vfs" else "",
        "alias": alias if kind == "vfs" else "",
        "tags": list(tags) if kind == "deep_memory" else [],
        "created_at": created_at,
        "can_remove": mine or is_agent_owner,
    }


def _active_row(p: cp.ActivePack, uid: str, agent_owner: str) -> dict:
    return _pack_row(pack_id=p.id, kind=p.kind, pack_owner=p.pack_owner,
                     owner_name=p.owner_name, project=p.project, alias=p.alias, tags=p.tags,
                     created_at=p.created_at, uid=uid, agent_owner=agent_owner, owner_tag=p.tag)


def _db_row(row: dict, uid: str, owner_name: str, agent_owner: str,
            owner_tag: str = "") -> dict:
    return _pack_row(pack_id=str(row.get("id") or ""), kind=str(row.get("kind") or ""),
                     pack_owner=str(row.get("pack_owner") or ""), owner_name=owner_name,
                     project=str(row.get("resource_id") or ""),
                     alias=str(row.get("alias") or ""), tags=_row_tags(row),
                     created_at=str(row.get("created_at") or ""), uid=uid,
                     agent_owner=agent_owner, owner_tag=owner_tag)


def _tag_of(packs: list[cp.ActivePack], pack_owner: str) -> str:
    """The collision tag the agent gives ``pack_owner`` ("" when none / not in effect)."""
    return next((p.tag for p in packs if p.pack_owner == pack_owner), "")


async def _active_state(db, ref: str, rec, uid: str) -> tuple[set[str], str]:
    """(ids of the packs in effect on ``ref`` whose folder, for vfs, still
    resolves as the same folder; ``uid``'s collision tag there)."""
    if rec is None or not sharing.sharing_active():
        return set(), ""
    ids: set[str] = set()
    packs = await cp.active_packs(db, ref, rec=rec)
    for p in packs:
        if p.kind == "vfs" and await asyncio.to_thread(
                cp.pack_project_root, p.pack_owner, p.project, p.resource_key) is None:
            continue
        ids.add(p.id)
    return ids, _tag_of(packs, uid)


# ── UI routes ─────────────────────────────────────────────────────────────


@router.get("/fd/context-packs/mine")
async def list_my_packs(user: dict = Depends(get_current_user)) -> dict:
    """Everything I share, on every agent (also with sharing off: then nothing
    is active)."""
    db = get_db()
    uid = str(user["id"])
    rows = await db.list_context_packs_for_owner(uid)
    name = await tenant_profile.owner_name(db, uid)
    by_ref: dict[str, list[dict]] = {}
    for row in rows:
        by_ref.setdefault(str(row.get("agent_ref") or ""), []).append(row)
    out: list[dict] = []
    for ref, ref_rows in by_ref.items():
        rec = await asyncio.to_thread(sharing.resolve_agent_record, ref)
        active, my_tag = await _active_state(db, ref, rec, uid)
        if rec is not None:
            agent_name, agent_owner = rec.name, rec.owner
        else:
            try:
                agent_name = sharing.parse_ref(ref)[1]
            except ValueError:
                agent_name = ""
            agent_owner = str(ref_rows[0].get("agent_owner") or "")
        for row in ref_rows:
            item = _db_row(row, uid, name, agent_owner, my_tag)
            item.update(agent_ref=ref, agent_name=agent_name, active=item["id"] in active)
            out.append(item)
    return {"packs": out}


@router.get("/fd/context-packs")
async def list_packs(agent_ref: str = "", user: dict = Depends(get_current_user)) -> dict:
    """What is shared on one agent, what I share there, and what I could share."""
    if not sharing.sharing_active():
        raise HTTPException(400, cp.SHARING_OFF_DETAIL)
    db = get_db()
    uid = str(user["id"])
    role, rec = await _role(db, uid, agent_ref)
    process = rec.runtime == "process"
    kinds = list(cp.PACK_KINDS) if process else ["profile"]
    active = await cp.active_packs(db, agent_ref, rec=rec)
    packs: list[dict] = []
    active_ids: set[str] = set()
    for p in active:
        if p.kind == "vfs" and await asyncio.to_thread(
                cp.pack_project_root, p.pack_owner, p.project, p.resource_key) is None:
            continue
        active_ids.add(p.id)
        packs.append(_active_row(p, uid, rec.owner))
    caller_name = await tenant_profile.owner_name(db, uid)
    mine: list[dict] = []
    for row in await db.list_context_packs_for_agent(agent_ref):
        if str(row.get("pack_owner") or "") != uid:
            continue
        item = _db_row(row, uid, caller_name, rec.owner, _tag_of(active, uid))
        item["active"] = item["id"] in active_ids
        mine.append(item)
    return {
        "agent_ref": agent_ref,
        "agent_name": rec.name,
        "runtime": rec.runtime,
        "role": role,
        "owner_name": await tenant_profile.owner_name(db, rec.owner),
        "kinds": kinds,
        "agent_supports_packs": await cp.agent_supports_packs(rec),
        # So a null probe answer reads as "couldn't check" on a running agent,
        # not "isn't running".
        "agent_running": bool(rec.running),
        "alias_prefix": cp.alias_prefix(caller_name),
        "deep_memory_tags": (await cp.deep_memory_tags(uid)) if "deep_memory" in kinds else [],
        "packs": packs,
        "mine": mine,
        "eligible_projects": (await asyncio.to_thread(cp.eligible_projects, uid)) if process else [],
        "limits": {"max_packs": cp.MAX_PACKS_PER_AGENT,
                   "max_vfs_per_owner": cp.MAX_VFS_PACKS_PER_OWNER,
                   "max_tags": cp.MAX_SLICE_TAGS},
    }


@router.post("/fd/context-packs")
async def create_pack(body: PackCreate, user: dict = Depends(get_current_user)) -> dict:
    """Publish one of my own resources to an agent I own or am a member of."""
    if not sharing.sharing_active():
        raise HTTPException(400, cp.SHARING_OFF_DETAIL)
    db = get_db()
    uid = str(user["id"])
    ref = body.agent_ref
    role, rec = await _role(db, uid, ref)
    kind = body.kind
    if kind not in cp.PACK_KINDS:
        raise HTTPException(400, cp.BAD_KIND_DETAIL)
    if kind in cp.PROCESS_ONLY_KINDS and rec.runtime != "process":
        raise HTTPException(400, cp.DOCKER_KIND_DETAIL)

    resource_id = resource_key = alias = project = ""
    slice_json = "{}"
    tags: list[str] = []
    caller_name = await tenant_profile.owner_name(db, uid)
    if kind == "vfs":
        project = body.project
        root = await asyncio.to_thread(cp.pack_project_root, uid, project)
        if root is None:
            raise HTTPException(400, cp.PROJECT_DETAIL)
        try:
            resource_key = await asyncio.to_thread(cp.project_key, root)
        except OSError:
            raise HTTPException(400, cp.PROJECT_DETAIL) from None
        rows = await db.list_context_packs_for_agent(ref)
        folded = project.casefold()
        if any(str(r.get("pack_owner") or "") == uid and r.get("kind") == "vfs"
               and str(r.get("resource_id") or "").casefold() == folded for r in rows):
            raise HTTPException(409, cp.DUPLICATE_DETAIL)
        prefix = cp.alias_prefix(caller_name)
        tombstones = await db.list_pack_aliases(ref)
        alias = (body.alias or "").strip()
        if alias:
            if not cp.alias_ok_for(alias, prefix):
                raise HTTPException(400, cp.ALIAS_DETAIL.format(prefix=prefix))
            if tombstones.get(alias, uid) != uid:
                raise HTTPException(409, cp.ALIAS_TAKEN_DETAIL)
        else:
            taken = {str(r.get("alias") or "") for r in rows if r.get("alias")}
            taken |= {a for a, owner in tombstones.items() if owner != uid}
            alias = cp.derive_alias(prefix, project, taken)
            if not alias:
                raise HTTPException(409, cp.ALIAS_TAKEN_DETAIL)
        resource_id = project
    elif kind == "deep_memory":
        try:
            tags = cp.clean_tags(body.tags)
        except ValueError:
            raise HTTPException(400, cp.TAGS_DETAIL) from None
        slice_json = json.dumps({"tags": tags})

    try:
        res = await db.create_context_pack(
            agent_ref=ref, agent_owner=rec.owner, pack_owner=uid, kind=kind,
            resource_id=resource_id, resource_key=resource_key, alias=alias,
            slice_json=slice_json, max_per_agent=cp.MAX_PACKS_PER_AGENT,
            max_vfs_per_owner=cp.MAX_VFS_PACKS_PER_OWNER)
    except sqlite3.IntegrityError as exc:
        msg = str(exc)
        if _ALIAS_UNIQUE in msg:
            raise HTTPException(409, cp.ALIAS_TAKEN_DETAIL) from None
        if "UNIQUE constraint failed" in msg:
            raise HTTPException(409, cp.DUPLICATE_DETAIL) from None
        if "FOREIGN KEY constraint failed" in msg:
            raise HTTPException(404, cp.AGENT_NOT_FOUND) from None
        raise
    if res == "limit":
        raise HTTPException(400, cp.LIMIT_DETAIL)
    if res == "vfs_limit":
        raise HTTPException(400, cp.VFS_LIMIT_DETAIL)
    if res == "alias_taken":
        raise HTTPException(409, cp.ALIAS_TAKEN_DETAIL)
    if not isinstance(res, dict):
        raise HTTPException(500, "Could not save the shared context")
    # A revoke that landed between the membership check and the insert ran its
    # delete before this row existed: undo it (the alias reservation is theirs).
    if role == "member" and not await sharing.member_check(db, ref, rec.owner, uid, max_age=0):
        await db.delete_context_pack(res["id"])
        raise HTTPException(404, cp.AGENT_NOT_FOUND)
    await cp.refresh_agent(db, ref)
    await cp.notify_owner_of_publish(db, rec, uid, kind, project)
    log.info("Context pack published", agent=rec.slug, publisher=uid, kind=kind)
    owner_tag = _tag_of(await cp.active_packs(db, ref, rec=rec), uid)
    return {"pack": _pack_row(pack_id=res["id"], kind=kind, pack_owner=uid,
                              owner_name=caller_name, project=resource_id, alias=alias,
                              tags=tags, created_at=res["created_at"], uid=uid,
                              agent_owner=rec.owner, owner_tag=owner_tag)}


@router.delete("/fd/context-packs/{pack_id}")
async def delete_pack(pack_id: str, user: dict = Depends(get_current_user)) -> dict:
    """Stop sharing: the publisher, or the agent's (current) owner for anyone's
    pack. Works with sharing off. Alias reservations stay."""
    if not cp.PACK_ID_RE.fullmatch(pack_id or ""):
        raise HTTPException(404, "Not found")
    db = get_db()
    row = await db.get_context_pack(pack_id)
    if row is None:
        raise HTTPException(404, "Not found")
    uid = str(user["id"])
    ref = str(row.get("agent_ref") or "")
    allowed = uid == str(row.get("pack_owner") or "")
    if not allowed:
        rec = await asyncio.to_thread(sharing.resolve_agent_record, ref)
        if rec is not None:
            allowed = bool(rec.owner) and uid == rec.owner
        else:
            allowed = uid == str(row.get("agent_owner") or "")
    if not allowed:
        raise HTTPException(404, "Not found")
    await db.delete_context_pack(pack_id)
    try:
        await cp.refresh_agent(db, ref)
    except Exception as exc:
        log.warning("Could not rewrite an agent's shared context", error=type(exc).__name__)
    log.info("Context pack removed", publisher=str(row.get("pack_owner") or ""),
             kind=str(row.get("kind") or ""), by_publisher=uid == row.get("pack_owner"))
    return {"ok": True}


# ── Agent route ───────────────────────────────────────────────────────────


@router.post(cp.VFS_PACK_ROUTE)
async def agent_vfs_packs(body: VfsResolveBody, request: Request) -> dict:
    """The roots of the shared folders a tool call names (``[]`` = all), for the
    calling agent, computed live. An alias that isn't in effect is absent."""
    if not sharing.sharing_active():
        raise HTTPException(403, cp.SHARING_OFF_DETAIL)
    db = get_db()
    if not speaker_grants.member_request(request) and not await db.has_context_packs("vfs"):
        # An owner's call while no folder is shared on any agent of this deck
        # (the usual case, and owners ask on every `vfs list_projects`): answer
        # without identifying the caller, which lists this deck's containers.
        # The transport gates and the body check still apply; a member's call
        # keeps its full grant check.
        cp.agent_transport_gate(request)
        _check_aliases(body)
        log.debug("Shared folders resolved", route=cp.VFS_PACK_ROUTE, count=0)
        return {"packs": []}
    acting, rec = await cp.caller_agent(request)
    if rec.runtime != "process":
        raise HTTPException(403, cp.PROCESS_ONLY_DETAIL)
    aliases = _check_aliases(body)
    packs = await cp.active_packs(db, rec.ref, rec=rec, kinds={"vfs"})
    if aliases:
        wanted = set(aliases)
        packs = [p for p in packs if p.alias in wanted]
    out: list[dict] = []
    for p in packs:
        root = await asyncio.to_thread(cp.pack_project_root, p.pack_owner, p.project,
                                       p.resource_key)
        if root is None:
            continue
        out.append({"alias": p.alias, "owner_name": p.label, "project": p.project,
                    "root": str(root)})
    # An owner's agent asks on every `vfs list_projects`: a miss is debug noise.
    emit = log.info if out else log.debug
    emit("Shared folders resolved", route=cp.VFS_PACK_ROUTE, agent=rec.slug,
         member=acting.user_id if acting is not None else "owner", count=len(out))
    return {"packs": out}
