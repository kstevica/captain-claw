"""Admin REST endpoints for Flight Deck — user & usage management."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck.rate_limiter import (
    PLAN_LIMITS, PLAN_FIELDS, update_plan_limits, get_plan_limits_json,
)

router = APIRouter(prefix="/fd/admin", tags=["admin"])


# ── Auth guard: admin only ──

async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    return user


# ── Models ──

class UpdateUserRequest(BaseModel):
    display_name: str | None = None
    role: str | None = None
    plan: str | None = None
    max_agents: int | None = None
    max_storage_mb: int | None = None
    requests_per_minute: int | None = None
    spawns_per_hour: int | None = None


# ── Endpoints ──

@router.get("/users")
async def list_users(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    admin: dict = Depends(require_admin),
):
    """List all users (admin only)."""
    db = get_db()
    users = await db.list_users(limit=limit, offset=offset)
    total = await db.count_users()
    return {"users": users, "total": total}


@router.get("/users/{user_id}")
async def get_user(user_id: str, admin: dict = Depends(require_admin)):
    """Get a single user's details (admin only)."""
    db = get_db()
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user


@router.put("/users/{user_id}")
async def update_user(
    user_id: str, body: UpdateUserRequest,
    admin: dict = Depends(require_admin),
):
    """Update a user's profile, role, or plan (admin only)."""
    db = get_db()
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")

    updates: dict = {}
    if body.display_name is not None:
        updates["display_name"] = body.display_name
    if body.role is not None:
        if body.role not in ("user", "admin"):
            raise HTTPException(400, "Role must be 'user' or 'admin'")
        updates["role"] = body.role

    # Plan & limit overrides go into metadata
    meta = {}
    try:
        meta = json.loads(user.get("metadata", "{}"))
    except (json.JSONDecodeError, TypeError):
        pass

    changed_meta = False
    if body.plan is not None:
        if body.plan not in PLAN_LIMITS:
            raise HTTPException(400, f"Plan must be one of: {', '.join(PLAN_LIMITS.keys())}")
        meta["plan"] = body.plan
        changed_meta = True
    for field in ("max_agents", "max_storage_mb", "requests_per_minute", "spawns_per_hour"):
        val = getattr(body, field, None)
        if val is not None:
            meta[field] = val
            changed_meta = True
    if changed_meta:
        updates["metadata"] = json.dumps(meta)

    if not updates:
        return {"ok": True, "message": "No changes"}

    await db.update_user(user_id, **updates)
    return {"ok": True, "user_id": user_id}


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin: dict = Depends(require_admin)):
    """Delete a user (admin only). Cannot delete yourself."""
    if user_id == admin["id"]:
        raise HTTPException(400, "Cannot delete your own account")
    db = get_db()
    deleted = await db.delete_user(user_id)
    if not deleted:
        raise HTTPException(404, "User not found")
    return {"ok": True}


@router.get("/usage")
async def get_usage(
    user_id: str | None = None,
    event_type: str | None = None,
    since: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
    admin: dict = Depends(require_admin),
):
    """Get usage logs with optional filters (admin only)."""
    db = get_db()
    logs = await db.get_usage_logs(
        user_id=user_id, event_type=event_type,
        since=since, limit=limit,
    )
    return {"logs": logs, "count": len(logs)}


@router.get("/usage/summary")
async def get_usage_summary(
    user_id: str | None = None,
    since: str | None = None,
    admin: dict = Depends(require_admin),
):
    """Get usage summary (event counts by type) (admin only)."""
    db = get_db()
    summary = await db.get_usage_summary(user_id=user_id, since=since)
    return {"summary": summary}


class UpdatePlanRequest(BaseModel):
    max_agents: int | None = None
    max_storage_mb: int | None = None
    requests_per_minute: int | None = None
    spawns_per_hour: int | None = None


@router.get("/plans")
async def list_plans(admin: dict = Depends(require_admin)):
    """List available plan tiers and their limits (admin only)."""
    return {"plans": PLAN_LIMITS}


@router.put("/plans/{plan}")
async def update_plan(plan: str, body: UpdatePlanRequest, admin: dict = Depends(require_admin)):
    """Update limits for a plan tier (admin only)."""
    if plan not in PLAN_LIMITS:
        raise HTTPException(400, f"Unknown plan '{plan}'. Available: {', '.join(PLAN_LIMITS.keys())}")
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    if not changes:
        return {"ok": True, "message": "No changes"}
    update_plan_limits(plan, changes)
    # Persist to DB as a system setting
    db = get_db()
    await db.set_settings("__system__", {"fd:plan-limits": get_plan_limits_json()})
    return {"ok": True, "plan": plan, "limits": PLAN_LIMITS[plan]}
