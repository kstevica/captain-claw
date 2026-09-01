"""Cross-user resource sharing — grant/revoke access to your own resources.

Lets an owner give other Flight Deck users access to selected archetypes, Code
projects, Basnas, Councils and VFS folders. One generic table
(``resource_shares``) plus a per-type ownership check. Enforcement of *reading*
shared resources lives in each resource's own routes (they consult the share
table); this module manages the grants and the user roster for the picker.

``resource_type`` ∈ {archetype, code, basna, council, vfs}. ``permission`` is
'view' (read-only) or 'edit' (collaborate); archetypes are always use-only.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/shares", tags=["shares"])

VALID_TYPES = {"archetype", "code", "basna", "council", "vfs"}
VALID_PERMS = {"view", "edit"}


async def owns_resource(db, user_id: str, resource_type: str, resource_id: str) -> bool:
    """Does ``user_id`` own this resource? (per-type dispatch)."""
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
    if resource_type not in VALID_TYPES:
        raise HTTPException(400, "Invalid resource_type")
    db = get_db()
    if not await owns_resource(db, user["id"], resource_type, resource_id):
        raise HTTPException(404, "Resource not found")
    shares = await db.list_shares_for_resource(resource_type, resource_id, user["id"])
    return {"shares": [
        {
            "grantee_id": s["grantee_id"],
            "grantee_email": s.get("grantee_email", ""),
            "grantee_name": s.get("grantee_name", ""),
            "permission": s["permission"],
        }
        for s in shares
    ]}


@router.get("/mine")
async def list_shared_with_me(
    resource_type: str | None = Query(None),
    user: dict = Depends(get_current_user),
):
    """Resources shared TO me (all types, or one), with owner info + permission."""
    db = get_db()
    if resource_type is not None and resource_type not in VALID_TYPES:
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
    if body.resource_type not in VALID_TYPES:
        raise HTTPException(400, "Invalid resource_type")
    perm = body.permission if body.permission in VALID_PERMS else "view"
    if body.resource_type == "archetype":
        perm = "view"  # archetypes are use-only
    if body.grantee_id == user["id"]:
        raise HTTPException(400, "Cannot share with yourself")
    db = get_db()
    if not await db.get_user_by_id(body.grantee_id):
        raise HTTPException(404, "User not found")
    if not await owns_resource(db, user["id"], body.resource_type, body.resource_id):
        raise HTTPException(404, "Resource not found or not yours")
    share = await db.create_share(
        body.resource_type, body.resource_id, user["id"], body.grantee_id, perm
    )
    # Tell the grantee (persistent bell notification).
    try:
        sharer = user.get("display_name") or user.get("email") or "A teammate"
        await db.add_notification(
            body.grantee_id, "share",
            f"{sharer} shared a {body.resource_type} with you",
            body=body.resource_id,
            ref_type=body.resource_type, ref_id=body.resource_id,
        )
    except Exception:
        pass
    return {"ok": True, "share": share}


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
    return {"ok": ok}
