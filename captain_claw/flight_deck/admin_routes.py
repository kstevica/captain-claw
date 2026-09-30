"""Admin REST endpoints for Flight Deck — user & usage management."""

from __future__ import annotations

import asyncio
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
    async with _team_keys_lock():
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
    return {"ok": True, "keys": existing, "agents_given_key": await _give_agents_their_key()}


def _drop_org_key_cache() -> None:
    """Basna/Vatra (and the archetype spawn) resolve ``@system`` from a
    short-lived cache of the org keys — make the next read a fresh one."""
    from captain_claw.flight_deck import basna_routes

    basna_routes._SYSTEM_KEYS_TS = 0.0


# Publishing and recovery both rewrite the org key stores and the published sets.
_TEAM_KEYS_LOCK: asyncio.Lock | None = None


def _team_keys_lock() -> asyncio.Lock:
    global _TEAM_KEYS_LOCK
    if _TEAM_KEYS_LOCK is None:  # built in the running loop, not at import
        _TEAM_KEYS_LOCK = asyncio.Lock()
    return _TEAM_KEYS_LOCK


async def _system_dict(db, setting: str) -> dict:
    raw = await db.get_system_setting(setting)
    try:
        data = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        data = {}
    return data if isinstance(data, dict) else {}


# `server.heal_keyless_agents` of the server module that is actually SERVING,
# registered by its lifespan. Importing it by name here would, on a deck started
# as `python -m captain_claw.flight_deck.server` (it runs as __main__), load a
# second copy of the module with its own process table — agents restarted
# through that copy can no longer be stopped by the routes.
_agent_key_healer = None


async def _give_agents_their_key() -> int:
    """Agents created while the team had no key for their model get it now
    (`server.heal_keyless_agents`). Returns how many."""
    try:
        healer = _agent_key_healer
        if healer is None:  # no lifespan ran (tests, tooling): the module by name
            from captain_claw.flight_deck.server import heal_keyless_agents as healer

        return len(await healer())
    except Exception:
        return 0


# ── Shared (team-default) tier sets ──
# An admin publishes one or more of their own tier sets as team defaults. Stored
# as system_settings 'fd:shared-tier-sets' = {"sets": [...], "defaultSetId": id}.
# The stored copy never holds a raw tier key: a keyed tier says "@system",
# resolved at run time from the org keys (see basna_routes._effective_key).
# Teammates who configured no tier set of their own fall back to the default
# set — so publishing a set also publishes its keys to those org stores, or the
# teammates would get the team's models and no credentials: see `_plan_team_keys`.
SHARED_TIER_SETS_SETTING = "fd:shared-tier-sets"


class SharedTierSetsRequest(BaseModel):
    sets: list[dict]
    defaultSetId: str | None = None


def _tier_key(t: dict) -> str:
    return str(t.get("api_key") or "").strip()


def _endpoint_host(base_url: str) -> str:
    """A custom endpoint's host[:port] for messages — never the URL's userinfo,
    path or query, which can carry a credential."""
    from urllib.parse import urlsplit

    url = base_url.strip()
    try:
        parts = urlsplit(url if "://" in url else "//" + url)
        host, port = parts.hostname or "", parts.port
    except ValueError:
        host, port = "", None
    if not host:
        return "custom endpoint"
    return f"{host}:{port}" if port else host


def _keyed_tiers(sets: list[dict] | None):
    """(provider, model, base_url, key) of every tier with a model."""
    for s in sets or []:
        if not isinstance(s, dict):
            continue
        for t in (s.get("tiers") or {}).values():
            if isinstance(t, dict) and t.get("model"):
                yield (str(t.get("provider") or ""), str(t.get("model") or ""),
                       str(t.get("base_url") or "").strip(), _tier_key(t))


def _plan_team_keys(sets: list[dict], org_keys: dict, endpoint_keys: dict,
                    published: list[dict] | None = None) -> dict:
    """What publishing ``sets`` means for the org keys — a published tier must
    work for a teammate the way it works in its owner's own space, where the
    tier's key goes straight into the agent.

    * A tier on the provider's OWN endpoint: its key is the org key for that
      provider (``fd:provider-keys``) — set, or replaced when it differs.
    * A tier behind a custom ``base_url`` (a gateway, a local server): its key
      belongs to THAT endpoint, so it is stored per endpoint
      (``fd:endpoint-keys``) and only ever resolved for it. It never becomes
      the provider's key. Where it IS the provider's org key (a deck that runs
      everything through one gateway) nothing is stored: the tier keeps
      following Admin → Provider keys.
    * Replacing a provider's org key must not change what an already-published
      ``@system`` custom-endpoint tier runs on (``published`` = the sets stored
      now): with no key of its own, such an endpoint is pinned to the old one.

    An org key that one of the set's tiers carries is left alone; otherwise the
    first tier decides. One store per provider / endpoint can't hold two keys,
    so those are reported as ``conflicts``.

    Returns ``provider_keys`` / ``endpoint_keys`` to store, plus what to tell
    the admin — provider ids and hostnames only, never key values:
    ``added`` / ``updated`` providers, shared ``endpoints``, ``conflicts``, and
    ``missing`` providers no source supplies a key for.
    """
    import os

    from captain_claw.flight_deck.basna_routes import endpoint_key_id
    from captain_claw.flight_deck.server import _needs_provider_key, _provider_key_env_names

    have = {str(p): str(k) for p, k in (org_keys or {}).items() if k}
    have_ep = {str(p): str(k) for p, k in (endpoint_keys or {}).items() if k}
    own: dict[str, list[str]] = {}                       # provider -> distinct keys, in order
    custom: dict[str, tuple[str, str, list[str]]] = {}   # endpoint id -> (provider, host, keys)
    for provider, _model, base_url, key in _keyed_tiers(sets):
        if not key or key == "@system":
            continue
        if base_url:
            entry = custom.setdefault(
                endpoint_key_id(provider, base_url), (provider, _endpoint_host(base_url), []))
            if key not in entry[2]:
                entry[2].append(key)
        elif _provider_key_env_names(provider) and key not in own.setdefault(provider, []):
            own[provider].append(key)

    provider_keys: dict[str, str] = {}
    added: list[str] = []
    updated: list[str] = []
    for provider, keys in own.items():
        if have.get(provider) in keys:
            continue
        provider_keys[provider] = keys[0]
        (updated if provider in have else added).append(provider)
    after = {**have, **provider_keys}

    ep_keys: dict[str, str] = {}
    endpoints: list[str] = []
    for eid, (provider, host, keys) in custom.items():
        if host not in endpoints:
            endpoints.append(host)
        if have_ep.get(eid) in keys:
            continue
        if eid not in have_ep and keys[0] == after.get(provider):
            continue  # it is the provider's org key: resolved through that
        ep_keys[eid] = keys[0]
    for provider, _model, base_url, key in [*_keyed_tiers(sets), *_keyed_tiers(published)]:
        if provider in updated and base_url and key == "@system":
            eid = endpoint_key_id(provider, base_url)
            if eid not in have_ep and eid not in custom:
                ep_keys[eid] = have[provider]

    conflicts = [p for p, keys in own.items() if len(keys) > 1]
    conflicts += [host for _p, host, keys in custom.values() if len(keys) > 1 and host not in conflicts]

    missing: set[str] = set()
    for s in sets or []:
        if not isinstance(s, dict):
            continue
        env_names = {str(ev.get("key") or "") for ev in (s.get("envVars") or [])
                     if isinstance(ev, dict) and str(ev.get("value") or "")}
        for provider, model, base_url, _key in _keyed_tiers([s]):
            if provider in after or not _needs_provider_key(provider, model, base_url):
                continue
            names = _provider_key_env_names(provider)
            if not any(n in env_names or os.environ.get(n) for n in names):
                missing.add(provider)
    return {"provider_keys": provider_keys, "endpoint_keys": ep_keys,
            "added": sorted(added), "updated": sorted(updated),
            "endpoints": endpoints, "conflicts": conflicts, "missing": sorted(missing)}


def _mask_shared_sets(sets: list[dict]) -> list[dict]:
    """The publishable copy of ``sets``: every tier key becomes the ``@system``
    sentinel (a blank one stays blank). The real keys live in the org stores,
    per provider or per endpoint — see `_plan_team_keys`."""
    out: list[dict] = []
    for s in sets or []:
        if not isinstance(s, dict):
            continue
        tiers: dict = {}
        for tname, t in (s.get("tiers") or {}).items():
            if not isinstance(t, dict):
                continue
            tt = dict(t)
            tt["api_key"] = "@system" if _tier_key(tt) else ""
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

    Teammates' agents run on the org key of each tier's endpoint, so the keys
    in a published set are stored as those org keys (`_plan_team_keys`). That
    shares them with the team: a key is written into each teammate's agent,
    whose owner can read it. The response names providers and endpoint hosts,
    never key values.
    """
    from captain_claw.flight_deck.basna_routes import ENDPOINT_KEYS_SETTING, NO_PROVIDER_FALLBACK

    db = get_db()
    async with _team_keys_lock():
        org_keys = await _system_dict(db, PROVIDER_KEYS_SETTING)
        endpoint_keys = await _system_dict(db, ENDPOINT_KEYS_SETTING)
        published = (await _system_dict(db, SHARED_TIER_SETS_SETTING)).get("sets")
        plan = _plan_team_keys(body.sets, org_keys, endpoint_keys,
                               published if isinstance(published, list) else None)
        if plan["provider_keys"]:
            await db.set_system_setting(
                PROVIDER_KEYS_SETTING, json.dumps({**org_keys, **plan["provider_keys"]}))
        if plan["endpoint_keys"]:
            await db.set_system_setting(
                ENDPOINT_KEYS_SETTING, json.dumps({**endpoint_keys, **plan["endpoint_keys"]}))
        if plan["provider_keys"] or plan["endpoint_keys"]:
            _drop_org_key_cache()
        payload = {"sets": _mask_shared_sets(body.sets), "defaultSetId": body.defaultSetId}
        # These sets' keys are dealt with from here on: `recover_team_keys`
        # leaves them alone (a team key an admin later removes stays removed).
        now_keyed = {**endpoint_keys, **plan["endpoint_keys"]}
        marks = (await _system_dict(db, SHARED_TIER_SETS_SETTING)).get(NO_PROVIDER_FALLBACK)
        stored = {**payload, KEYS_SHARED: [str(s.get("id") or "") for s in payload["sets"]],
                  NO_PROVIDER_FALLBACK: sorted(str(e) for e in (marks if isinstance(marks, list) else [])
                                               if not now_keyed.get(str(e)))}
        await db.set_system_setting(SHARED_TIER_SETS_SETTING, json.dumps(stored))
    return {
        "ok": True, **payload,
        "team_keys_added": plan["added"],
        "team_keys_updated": plan["updated"],
        "team_endpoints_shared": plan["endpoints"],
        "team_keys_conflict": plan["conflicts"],
        "team_keys_missing": plan["missing"],
        "agents_given_key": await _give_agents_their_key(),
    }


# In the published blob: ids of the sets whose keys a publish (or the recovery
# below) has already put in the org stores.
KEYS_SHARED = "keys_shared"
# …and, for a set that is still open, the providers / endpoints already settled.
KEYS_SETTLED = "keys_settled"


async def recover_team_keys(db) -> dict:
    """Give the sets that are ALREADY published the team keys a publish gives
    them now — without anyone having to publish again. A one-time migration
    per set.

    A set published before publishing shared its keys (or before custom
    endpoints had keys of their own) sits in ``fd:shared-tier-sets`` with
    "@system" or blank tier keys and nothing behind them: teammates' agents get
    the model and no credentials. The keys are still in the publishing admin's
    own copy of that set (same set id). They are taken from there — only an
    admin's copy, only for a tier that still names the same provider and
    endpoint — and stored exactly as a publish stores them. Gaps only: a team
    key that exists is never replaced, and one that reaches agents through
    Flight Deck's own environment is left to it.

    A set is done (``KEYS_SHARED``) once none of its tiers is left without a
    key — as is every set published from now on — and is never looked at
    again; in a set that stays open, what has a key is settled
    (``KEYS_SETTLED``) and only the keyless tiers are tried again. So removing
    a team key in Admin → Provider keys sticks.

    A provider key recovered here came from a tier on the provider's OWN
    endpoint: it is not the key of a gateway tier whose own key could not be
    found, and must not be sent there (``NO_PROVIDER_FALLBACK``).

    Returns what it shared (provider ids / endpoint hosts, never key values)
    and, under ``"unresolved"``, the tiers it found no key for; empty when
    there was nothing to do.
    """
    import os

    from captain_claw.flight_deck.basna_routes import (
        ENDPOINT_KEYS_SETTING, NO_PROVIDER_FALLBACK, endpoint_key_id,
    )
    from captain_claw.flight_deck.server import _provider_key_env_names, _signs_in_through_chatgpt

    async with _team_keys_lock():
        blob = await _system_dict(db, SHARED_TIER_SETS_SETTING)
        sets = [s for s in (blob.get("sets") if isinstance(blob.get("sets"), list) else [])
                if isinstance(s, dict)]

        def _listed(field: str) -> set[str]:
            return {str(i) for i in (blob.get(field) if isinstance(blob.get(field), list) else [])}

        done, settled, no_fallback = _listed(KEYS_SHARED), _listed(KEYS_SETTLED), _listed(NO_PROVIDER_FALLBACK)
        before = (frozenset(settled), frozenset(no_fallback))
        org = {str(p): str(k) for p, k in (await _system_dict(db, PROVIDER_KEYS_SETTING)).items() if k}
        eps = {str(p): str(k) for p, k in (await _system_dict(db, ENDPOINT_KEYS_SETTING)).items() if k}

        def _facts(t: dict) -> tuple[str, str, str, str]:
            provider = str(t.get("provider") or "")
            base_url = str(t.get("base_url") or "").strip()
            return provider, base_url, endpoint_key_id(provider, base_url), _tier_key(t)

        def _in_set_env(s: dict, provider: str) -> bool:
            names = _provider_key_env_names(provider)
            return any(isinstance(ev, dict) and ev.get("key") in names and ev.get("value")
                       for ev in (s.get("envVars") or []))

        def _candidate(s: dict, t: dict) -> bool:
            """A published tier the admin's copy may hold a team key for."""
            provider, _base_url, eid, key = _facts(t)
            if not t.get("model") or key not in ("", "@system") or (eid or provider) in settled:
                return False
            if eid:  # a custom endpoint: until it has a key of its own
                return not eps.get(eid)
            names = _provider_key_env_names(provider)
            if not names or org.get(provider) or _signs_in_through_chatgpt(provider, str(t.get("model") or "")):
                return False
            # …which agents don't already get with the set's extra keys, or
            # inherit from Flight Deck's environment.
            return not (_in_set_env(s, provider) or any(os.environ.get(n) for n in names))

        def _keyless(s: dict, t: dict) -> bool:
            """…and has nothing at all behind its "@system": an agent on it gets
            no key. (A BLANK tier was published without one — nothing is owed.)"""
            provider, _base_url, eid, key = _facts(t)
            if key != "@system" or not _candidate(s, t) or not _provider_key_env_names(provider):
                return False
            # A gateway tier otherwise runs on the provider's key.
            return not (eid and org.get(provider) and eid not in no_fallback)

        pending = [str(s.get("id") or "") for s in sets if str(s.get("id") or "") not in done]
        if not pending:
            return {}
        wanted = [(s, name, t) for s in sets if str(s.get("id") or "") in pending
                  for name, t in (s.get("tiers") or {}).items()
                  if isinstance(t, dict) and _candidate(s, t)]

        # The admins' own copies of the published sets, by set id (oldest admin first).
        copies: dict[str, list[dict]] = {}
        ids = {str(s.get("id") or "") for s, _n, _t in wanted}
        for user in reversed(await db.list_users(limit=1_000_000)) if wanted else []:
            if user.get("role") != "admin":
                continue
            try:
                own = json.loads(await db.get_setting(user["id"], "fd:forge-tiers") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            for own_set in (own.get("sets") if isinstance(own, dict) else None) or []:
                if isinstance(own_set, dict) and str(own_set.get("id") or "") in ids:
                    copies.setdefault(str(own_set["id"]), []).append(own_set)

        def _publisher_key(set_id: str, name: str, provider: str, eid: str) -> str:
            for own_set in copies.get(set_id, []):
                tiers = own_set.get("tiers") if isinstance(own_set.get("tiers"), dict) else {}
                on_endpoint = False
                for t in [tiers.get(name), *tiers.values()]:  # the same tier first
                    if not isinstance(t, dict):
                        continue
                    t_provider, _b, t_eid, key = _facts(t)
                    if t_provider == provider and t_eid == eid:
                        on_endpoint = True
                        if key not in ("", "@system"):
                            return key
                if on_endpoint and not eid:  # …or kept with the set's extra keys
                    names = _provider_key_env_names(provider)
                    for ev in own_set.get("envVars") or []:
                        if isinstance(ev, dict) and ev.get("key") in names and str(ev.get("value") or "").strip():
                            return str(ev["value"]).strip()
            return ""

        providers: list[str] = []
        hosts: list[str] = []
        for s, name, t in wanted:
            provider, base_url, eid, key = _facts(t)
            found = (eps.get(eid) if eid else org.get(provider)) or _publisher_key(
                str(s.get("id") or ""), name, provider, eid)
            if not found:
                continue
            if eid:
                stored = eid not in eps and found != org.get(provider)
                if stored:
                    eps[eid] = found
                host = _endpoint_host(base_url)
                if (stored or key == "") and host not in hosts:
                    hosts.append(host)
            elif provider not in org:
                org[provider] = found
                providers.append(provider)
            if key == "":  # published blank: now there is a key behind the sentinel
                t["api_key"] = "@system"
        for _s, _name, t in wanted:
            provider, _base_url, eid, key = _facts(t)
            if eid and key == "@system" and not eps.get(eid) and provider in providers:
                no_fallback.add(eid)

        # What is still without a key keeps its set open for a later try.
        unresolved: list[str] = []
        open_sets: set[str] = set()
        for s, t in ((s, t) for s in sets if str(s.get("id") or "") in pending
                     for t in (s.get("tiers") or {}).values() if isinstance(t, dict) and _keyless(s, t)):
            provider, base_url, _eid, _key = _facts(t)
            label = f"{provider} at {_endpoint_host(base_url)}" if base_url else provider
            if label not in unresolved:
                unresolved.append(label)
            open_sets.add(str(s.get("id") or ""))
        newly_done = [i for i in pending if i not in open_sets]
        # In a set that stays open, everything but its keyless tiers is settled.
        for s in sets:
            if str(s.get("id") or "") in open_sets:
                for t in (s.get("tiers") or {}).values():
                    if isinstance(t, dict) and t.get("model") and not _keyless(s, t):
                        provider, _base_url, eid, _key = _facts(t)
                        settled.add(eid or provider)
        if not (providers or hosts or newly_done) and before == (frozenset(settled), frozenset(no_fallback)):
            return {"unresolved": unresolved} if unresolved else {}
        if providers or hosts:
            await db.set_system_setting(PROVIDER_KEYS_SETTING, json.dumps(org))
            await db.set_system_setting(ENDPOINT_KEYS_SETTING, json.dumps(eps))
        await db.set_system_setting(SHARED_TIER_SETS_SETTING, json.dumps({
            **blob, "sets": sets, KEYS_SHARED: sorted(done | set(newly_done)),
            KEYS_SETTLED: sorted(settled), NO_PROVIDER_FALLBACK: sorted(no_fallback)}))
        _drop_org_key_cache()
    if not (providers or hosts):
        return {"unresolved": unresolved} if unresolved else {}
    print(f"Flight Deck: recovered team keys for the published tier sets — "
          f"{', '.join([*providers, *hosts])}", flush=True)
    return {"providers": providers, "endpoints": hosts, **({"unresolved": unresolved} if unresolved else {})}


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
