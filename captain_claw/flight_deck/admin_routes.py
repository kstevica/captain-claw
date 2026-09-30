"""Admin REST endpoints for Flight Deck — user & usage management."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db, hash_password
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

class CreateUserRequest(BaseModel):
    email: str
    password: str  # min 6 chars
    display_name: str | None = None
    role: str = "user"


class UpdateUserRequest(BaseModel):
    display_name: str | None = None
    role: str | None = None
    plan: str | None = None
    max_agents: int | None = None
    max_storage_mb: int | None = None
    requests_per_minute: int | None = None
    spawns_per_hour: int | None = None
    password: str | None = None  # admin password reset (min 6 chars)


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


@router.post("/users", status_code=201)
async def create_user(body: CreateUserRequest, admin: dict = Depends(require_admin)):
    """Create a new user account (admin only).

    This is the account-provisioning path for team deployments where public
    self-registration is closed (``FD_REGISTRATION_OPEN`` unset). It bypasses
    that gate deliberately — only an existing admin can reach it. The new user
    can change their own password afterward under their profile.
    """
    db = get_db()
    email = (body.email or "").strip().lower()
    if not email or not body.password:
        raise HTTPException(400, "Email and password required")
    if "@" not in email:
        raise HTTPException(400, "Invalid email address")
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if body.role not in ("user", "admin"):
        raise HTTPException(400, "Role must be 'user' or 'admin'")

    existing = await db.get_user_by_email(email)
    if existing:
        raise HTTPException(409, "Email already registered")

    display = (body.display_name or "").strip() or email.split("@")[0]
    user = await db.create_user(
        email=email, password_hash=hash_password(body.password),
        display_name=display, role=body.role,
    )
    return {
        "ok": True,
        "user": {"id": user["id"], "email": user["email"],
                 "display_name": display, "role": body.role},
    }


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
    if body.password:
        if len(body.password) < 6:
            raise HTTPException(400, "Password must be at least 6 characters")
        updates["password_hash"] = hash_password(body.password)

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


@router.get("/costs")
async def get_cost_rollup(
    since: str | None = None,
    until: str | None = None,
    admin: dict = Depends(require_admin),
):
    """Team cost rollup: $ spend by user and by run kind (admin only).

    `since`/`until` are ISO timestamps; both optional. Aggregates the fully
    attributed cost_ledger rows (each stamped with owner_user_id + run_kind)."""
    db = get_db()
    roll = await db.aggregate_run_costs(since=since, until=until)
    return {**roll, "since": since, "until": until}


class UpdatePlanRequest(BaseModel):
    max_agents: int | None = None
    max_storage_mb: int | None = None
    requests_per_minute: int | None = None
    spawns_per_hour: int | None = None


class UpdateConfigRequest(BaseModel):
    docker_spawn_enabled: bool | None = None


class ProviderKeysRequest(BaseModel):
    keys: dict[str, str]


# System config defaults
SYSTEM_CONFIG_DEFAULTS = {
    "docker_spawn_enabled": True,
}


def _load_system_config(raw: str | None) -> dict:
    """Parse system config from DB, merge with defaults."""
    cfg = {**SYSTEM_CONFIG_DEFAULTS}
    if raw:
        try:
            stored = json.loads(raw)
            cfg.update(stored)
        except (json.JSONDecodeError, TypeError):
            pass
    return cfg


@router.get("/plans")
async def list_plans(admin: dict = Depends(require_admin)):
    """List available plan tiers and their limits (admin only)."""
    return {"plans": PLAN_LIMITS}


@router.get("/config")
async def get_config(admin: dict = Depends(require_admin)):
    """Get system configuration (admin only)."""
    db = get_db()
    raw = await db.get_system_setting("fd:system-config")
    return _load_system_config(raw)


@router.put("/config")
async def update_config(body: UpdateConfigRequest, admin: dict = Depends(require_admin)):
    """Update system configuration (admin only)."""
    db = get_db()
    raw = await db.get_system_setting("fd:system-config")
    cfg = _load_system_config(raw)
    for key, val in body.model_dump(exclude_none=True).items():
        cfg[key] = val
    await db.set_system_setting("fd:system-config", json.dumps(cfg))
    return cfg


PROVIDER_KEYS_SETTING = "fd:provider-keys"


@router.get("/provider-keys")
async def get_provider_keys(admin: dict = Depends(require_admin)):
    """Get system-level provider API keys (admin only)."""
    db = get_db()
    raw = await db.get_system_setting(PROVIDER_KEYS_SETTING)
    if not raw:
        return {"keys": {}}
    try:
        return {"keys": json.loads(raw)}
    except (json.JSONDecodeError, TypeError):
        return {"keys": {}}


@router.put("/provider-keys")
async def update_provider_keys(body: ProviderKeysRequest, admin: dict = Depends(require_admin)):
    """Save system-level provider API keys (admin only)."""
    db = get_db()
    # Merge with existing: allows partial updates, empty string removes key
    raw = await db.get_system_setting(PROVIDER_KEYS_SETTING)
    existing: dict[str, str] = {}
    if raw:
        try:
            existing = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    for k, v in body.keys.items():
        if v:
            existing[k] = v
        else:
            existing.pop(k, None)
    await db.set_system_setting(PROVIDER_KEYS_SETTING, json.dumps(existing))
    _drop_org_key_cache()
    return {"ok": True, "keys": existing}


def _drop_org_key_cache() -> None:
    """Basna/Vatra (and the archetype spawn) resolve ``@system`` from a
    short-lived cache of the org keys — make the next read a fresh one."""
    from captain_claw.flight_deck import basna_routes

    basna_routes._SYSTEM_KEYS_TS = 0.0


# ── Shared (team-default) tier sets ──
# An admin publishes one or more of their own tier sets as team defaults. Stored
# as system_settings 'fd:shared-tier-sets' = {"sets": [...], "defaultSetId": id}.
# The stored copy never holds a raw tier key: a tier says "@system" — resolved at
# run time from fd:provider-keys by provider (see basna_routes._effective_key) —
# or nothing. Teammates who configured no tier set of their own fall back to the
# default set, so publishing also has to leave them a key to run on: see
# `_plan_team_keys`.
SHARED_TIER_SETS_SETTING = "fd:shared-tier-sets"


class SharedTierSetsRequest(BaseModel):
    sets: list[dict]
    defaultSetId: str | None = None


def _tier_key(t: dict) -> str:
    return str(t.get("api_key") or "").strip()


def _on_custom_endpoint(t: dict) -> bool:
    return bool(str(t.get("base_url") or "").strip())


def _plan_team_keys(sets: list[dict], org_keys: dict) -> dict:
    """What publishing ``sets`` means for the org provider keys.

    An org key is THE key for a provider's own endpoint, so only a tier on that
    endpoint can supply one — never a tier behind a custom ``base_url`` (its
    key belongs to that gateway or local server, often a placeholder), and only
    for providers an agent reads a key for. An existing org key is never
    replaced. Returns provider ids / tier names only, never key values:

    * ``add``      — {provider: key} to store: own-endpoint keys with no org key yet
    * ``differs``  — providers whose own-endpoint key in the set is not the
                     existing org key (teammates keep running on the org key)
    * ``unshared`` — tiers on a custom endpoint whose key can't be shared
                     (published keyless — see `_mask_shared_sets`)
    * ``missing``  — providers teammates would have no key for at all: a tier
                     needs the provider's key, there is no org key, and neither
                     the set's env vars nor Flight Deck's environment supply one
    """
    import os

    from captain_claw.flight_deck.server import _needs_provider_key, _provider_key_env_names

    have = {str(p): str(k) for p, k in (org_keys or {}).items() if k}
    add: dict[str, str] = {}
    differs: set[str] = set()
    unshared: list[str] = []
    for s in sets or []:
        if not isinstance(s, dict):
            continue
        for name, t in (s.get("tiers") or {}).items():
            if not isinstance(t, dict) or not t.get("model"):
                continue
            provider, key = str(t.get("provider") or ""), _tier_key(t)
            if not key or key == "@system":
                continue
            if _on_custom_endpoint(t):
                if have.get(provider) != key and str(name) not in unshared:
                    unshared.append(str(name))
            elif _provider_key_env_names(provider):
                if provider in have:
                    if have[provider] != key:
                        differs.add(provider)
                else:
                    add.setdefault(provider, key)
    after = {**have, **add}
    missing: set[str] = set()
    for s in sets or []:
        if not isinstance(s, dict):
            continue
        env_names = {str(ev.get("key") or "") for ev in (s.get("envVars") or [])
                     if isinstance(ev, dict) and str(ev.get("value") or "")}
        for t in (s.get("tiers") or {}).values():
            if not isinstance(t, dict) or not t.get("model"):
                continue
            provider = str(t.get("provider") or "")
            if provider in after or not _needs_provider_key(
                    provider, str(t.get("model") or ""), str(t.get("base_url") or "")):
                continue
            names = _provider_key_env_names(provider)
            if not any(n in env_names or os.environ.get(n) for n in names):
                missing.add(provider)
    return {"add": add, "differs": sorted(differs), "unshared": unshared, "missing": sorted(missing)}


def _mask_shared_sets(sets: list[dict], org_keys: dict | None = None) -> list[dict]:
    """The publishable copy of ``sets``: no raw tier key survives.

    A key becomes the ``@system`` sentinel where that resolves to the right key
    for the tier's endpoint: on the provider's own endpoint, or on a custom one
    whose key IS the org key (a deck that runs everything through one gateway).
    Any other custom-endpoint key is dropped — ``@system`` there would send the
    provider's real key to that other host.
    """
    org = {str(p): str(k) for p, k in (org_keys or {}).items() if k}
    out: list[dict] = []
    for s in sets or []:
        if not isinstance(s, dict):
            continue
        tiers: dict = {}
        for tname, t in (s.get("tiers") or {}).items():
            if not isinstance(t, dict):
                continue
            tt = dict(t)
            key = _tier_key(tt)
            if not key:
                tt["api_key"] = ""
            elif key != "@system":
                shareable = not _on_custom_endpoint(tt) or org.get(str(tt.get("provider") or "")) == key
                tt["api_key"] = "@system" if shareable else ""
            tiers[tname] = tt
        out.append({**s, "tiers": tiers})
    return out


@router.get("/shared-tier-sets")
async def get_shared_tier_sets(admin: dict = Depends(require_admin)):
    """Get the team-default tier sets (admin only)."""
    db = get_db()
    raw = await db.get_system_setting(SHARED_TIER_SETS_SETTING)
    if not raw:
        return {"sets": [], "defaultSetId": None}
    try:
        blob = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"sets": [], "defaultSetId": None}
    return blob if isinstance(blob, dict) else {"sets": [], "defaultSetId": None}


@router.put("/shared-tier-sets")
async def update_shared_tier_sets(body: SharedTierSetsRequest, admin: dict = Depends(require_admin)):
    """Publish tier sets as team defaults (admin only).

    Teammates' agents run on the ORG key of each provider, so a key in a
    published tier becomes the org key for its provider when none is set
    (`_plan_team_keys`) — otherwise they would get the team's models and no
    credentials. That shares the key with the team: it is written into each
    teammate's agent, whose owner can read it. The response names providers and
    tiers, never key values.
    """
    db = get_db()
    raw = await db.get_system_setting(PROVIDER_KEYS_SETTING)
    try:
        org_keys = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        org_keys = {}
    if not isinstance(org_keys, dict):
        org_keys = {}
    plan = _plan_team_keys(body.sets, org_keys)
    if plan["add"]:
        org_keys = {**org_keys, **plan["add"]}
        await db.set_system_setting(PROVIDER_KEYS_SETTING, json.dumps(org_keys))
        _drop_org_key_cache()
    payload = {"sets": _mask_shared_sets(body.sets, org_keys), "defaultSetId": body.defaultSetId}
    await db.set_system_setting(SHARED_TIER_SETS_SETTING, json.dumps(payload))
    return {
        "ok": True, **payload,
        "team_keys_added": sorted(plan["add"]),
        "team_keys_differ": plan["differs"],
        "team_keys_unshared": plan["unshared"],
        "team_keys_missing": plan["missing"],
    }


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
    await db.set_system_setting("fd:plan-limits", get_plan_limits_json())
    return {"ok": True, "plan": plan, "limits": PLAN_LIMITS[plan]}
