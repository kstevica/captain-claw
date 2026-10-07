"""Cross-user resource sharing — grant/revoke access to your own resources.

Lets an owner give other Flight Deck users access to selected archetypes, Code
projects, Basnas, Councils and VFS folders. One generic table
(``resource_shares``) plus a per-type ownership check. Enforcement of *reading*
shared resources lives in each resource's own routes (they consult the share
table); this module manages the grants and the user roster for the picker.

``resource_type`` ∈ {archetype, code, basna, council, vfs} — plus ``agent``
(chat-only shared agents, see ``agent_sharing``) when ``FD_AGENT_SHARING`` is on.
``permission`` is 'view' (read-only) or 'edit' (collaborate); archetypes are
always use-only and agents always "can chat" ('view').
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from captain_claw.flight_deck import agent_sharing, speaker_grants
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/shares", tags=["shares"])

VALID_TYPES = {"archetype", "code", "basna", "council", "vfs"}
VALID_PERMS = {"view", "edit"}


def _valid_types() -> set[str]:
    """The shareable types; ``agent`` only while agent sharing is active."""
    if agent_sharing.sharing_active():
        return VALID_TYPES | {agent_sharing.AGENT_RESOURCE}
    return VALID_TYPES


async def owns_resource(db, user_id: str, resource_type: str, resource_id: str) -> bool:
    """Does ``user_id`` own this resource? (per-type dispatch)."""
    if resource_type == agent_sharing.AGENT_RESOURCE:
        rec = await asyncio.to_thread(agent_sharing.resolve_agent_record, resource_id)
        return agent_sharing.check_shareable(rec, user_id) is None
    if resource_type == "archetype":
        return await db.get_user_archetype(user_id, resource_id) is not None
    if resource_type == "basna":
        return await db.get_basna_session(resource_id, user_id) is not None
    if resource_type == "council":
        return await db.get_council_session(resource_id, user_id) is not None
    if resource_type in ("code", "vfs"):
        # A Code project and a VFS folder are the same on-disk resource.
        from captain_claw.flight_deck.vfs_routes import _user_root
        try:
            return (_user_root(user_id) / resource_id).is_dir()
        except Exception:
            return False
    return False


class ShareCreate(BaseModel):
    resource_type: str
    resource_id: str
    grantee_id: str
    permission: str = "view"


@router.get("/users")
async def list_share_users(user: dict = Depends(get_current_user)):
    """Roster of other FD users to share with (id, email, display_name)."""
    db = get_db()
    users = await db.list_users(limit=1000, offset=0)
    me = user["id"]
    return {"users": [
        {"id": u["id"], "email": u.get("email", ""), "display_name": u.get("display_name", "")}
        for u in users if u["id"] != me
    ]}


@router.get("")
async def list_resource_shares(
    resource_type: str = Query(...),
    resource_id: str = Query(...),
    user: dict = Depends(get_current_user),
):
    """Who a resource I own is currently shared with."""
    if resource_type not in _valid_types():
        raise HTTPException(400, "Invalid resource_type")
    db = get_db()
    if not await owns_resource(db, user["id"], resource_type, resource_id):
        raise HTTPException(404, "Resource not found")
    shares = await db.list_shares_for_resource(resource_type, resource_id, user["id"])
    rows = []
    for s in shares:
        row = {
            "grantee_id": s["grantee_id"],
            "grantee_email": s.get("grantee_email", ""),
            "grantee_name": s.get("grantee_name", ""),
            "permission": s["permission"],
        }
        if resource_type == agent_sharing.AGENT_RESOURCE:
            # Has this member let the agent use their Google during their chats?
            row["google_enabled"] = await speaker_grants.google_opted_in(
                db, s["grantee_id"], resource_id, user["id"])
        rows.append(row)
    return {"shares": rows}


@router.get("/mine")
async def list_shared_with_me(
    resource_type: str | None = Query(None),
    user: dict = Depends(get_current_user),
):
    """Resources shared TO me (all types, or one), with owner info + permission."""
    db = get_db()
    if resource_type is not None and resource_type not in _valid_types():
        raise HTTPException(400, "Invalid resource_type")
    shares = await db.list_shares_for_grantee(user["id"], resource_type)
    return {"shares": [
        {
            "resource_type": s["resource_type"],
            "resource_id": s["resource_id"],
            "owner_id": s["owner_id"],
            "owner_email": s.get("owner_email", ""),
            "owner_name": s.get("owner_name", ""),
            "permission": s["permission"],
        }
        for s in shares
    ]}


@router.post("")
async def create_resource_share(body: ShareCreate, user: dict = Depends(get_current_user)):
    """Grant (or update the permission of) access to a resource I own."""
    if body.resource_type not in _valid_types():
        raise HTTPException(400, "Invalid resource_type")
    is_agent = body.resource_type == agent_sharing.AGENT_RESOURCE
    perm = body.permission if body.permission in VALID_PERMS else "view"
    if body.resource_type == "archetype" or is_agent:
        perm = "view"  # archetypes are use-only; a shared agent is chat-only
    if body.grantee_id == user["id"]:
        raise HTTPException(400, "Cannot share with yourself")
    db = get_db()
    if not await db.get_user_by_id(body.grantee_id):
        raise HTTPException(404, "User not found")
    rec = None
    if is_agent:
        rec = await asyncio.to_thread(agent_sharing.resolve_agent_record, body.resource_id)
        if rec is not None and not rec.owner:
            raise HTTPException(400, "Unowned agents can't be shared")
        if rec is None or rec.owner != user["id"]:
            raise HTTPException(404, "Resource not found or not yours")
        reason = agent_sharing.check_shareable(rec, user["id"])
        if reason:
            raise HTTPException(400, reason)
    if not await owns_resource(db, user["id"], body.resource_type, body.resource_id):
        raise HTTPException(404, "Resource not found or not yours")
    if is_agent and rec is not None and rec.runtime == "process":
        # Pin the effective instance id, so the ref survives a later token change.
        # On the event loop, not a thread: every other .processes.json writer
        # does its read-modify-write synchronously here, which is what keeps
        # them from losing each other's updates (a spawn's pid, a port announce).
        agent_sharing.ensure_process_instance_persisted(rec.slug)
    share = await db.create_share(
        body.resource_type, body.resource_id, user["id"], body.grantee_id, perm
    )
    if is_agent:
        # PR D: the agent's members file names its current members.
        try:
            from captain_claw.flight_deck import context_packs

            await context_packs.refresh_agent(db, body.resource_id)
        except Exception as exc:
            log.warning("Could not update a shared agent's member list",
                        error=type(exc).__name__)
    # Tell the grantee (persistent bell notification).
    try:
        sharer = user.get("display_name") or user.get("email") or "A teammate"
        if is_agent and rec is not None:
            title = f"{sharer} shared the agent “{rec.name}” with you"
            body_text = rec.name
        else:
            title = f"{sharer} shared a {body.resource_type} with you"
            body_text = body.resource_id
        await db.add_notification(
            body.grantee_id, "share", title,
            body=body_text,
            ref_type=body.resource_type, ref_id=body.resource_id,
        )
    except Exception:
        pass
    return {"ok": True, "share": share}


async def _agent_name(ref: str) -> str:
    try:
        rec = await asyncio.to_thread(agent_sharing.resolve_agent_record, ref)
    except Exception:
        rec = None
    if rec is not None:
        return rec.name
    try:
        return agent_sharing.parse_ref(ref)[1]
    except ValueError:
        return "an agent"


async def _revoke_agent_member(ref: str, member_id: str, reason: str) -> None:
    """A member lost access: forget their cached membership, close their open
    turn grants, drop their Google opt-in for the agent (a re-share starts with
    it off) and close their live sockets now (the socket watchdog would only
    notice on its next tick). In that order: the generation bump comes first,
    so no membership check in flight can re-cache them."""
    agent_sharing.invalidate_member_cache(ref, member_id)
    try:
        speaker_grants.revoke(ref, member_id)
    except Exception as exc:
        log.warning("Could not close a member's shared-agent grants", error=type(exc).__name__)
    try:
        await speaker_grants.clear_google_optins(get_db(), ref, member_id)
    except Exception as exc:
        log.warning("Could not clear a member's shared-agent Google opt-in",
                    error=type(exc).__name__)
    # PR B: their context packs on the agent go too (a re-share starts with
    # none; their alias reservations stay), and the agent's shared-context
    # files are rewritten before this returns.
    try:
        from captain_claw.flight_deck import context_packs

        await get_db().delete_context_packs_for_member(ref, member_id)
        await context_packs.refresh_agent(get_db(), ref)
    except Exception as exc:
        log.warning("Could not drop a member's context packs", error=type(exc).__name__)
    try:
        await agent_sharing.close_member_sockets(ref, member_id, code=4403, reason=reason)
    except Exception as exc:
        log.warning("Could not close a member's shared-agent sockets", error=type(exc).__name__)


@router.delete("")
async def delete_resource_share(
    resource_type: str = Query(...),
    resource_id: str = Query(...),
    grantee_id: str = Query(...),
    user: dict = Depends(get_current_user),
):
    """Revoke a grant I made (owner side)."""
    db = get_db()
    ok = await db.delete_share(resource_type, resource_id, user["id"], grantee_id)
    if ok and resource_type == agent_sharing.AGENT_RESOURCE:
        await _revoke_agent_member(resource_id, grantee_id, "Access removed")
        try:
            owner = user.get("display_name") or user.get("email") or "A teammate"
            name = await _agent_name(resource_id)
            await db.add_notification(
                grantee_id, "share",
                f"{owner} stopped sharing “{name}” with you",
                body=name, ref_type=agent_sharing.AGENT_RESOURCE, ref_id=resource_id,
            )
        except Exception:
            pass
    return {"ok": ok}


@router.delete("/leave")
async def leave_resource_share(
    resource_type: str = Query(...),
    resource_id: str = Query(...),
    owner_id: str = Query(...),
    user: dict = Depends(get_current_user),
):
    """Remove a resource that was shared with me from my view (grantee side)."""
    db = get_db()
    ok = await db.delete_share(resource_type, resource_id, owner_id, user["id"])
    if ok and resource_type == agent_sharing.AGENT_RESOURCE:
        await _revoke_agent_member(resource_id, user["id"], "You left this shared agent")
    return {"ok": ok}
