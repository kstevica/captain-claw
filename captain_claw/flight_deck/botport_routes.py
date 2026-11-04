"""BotPort link REST endpoints: status for everyone, configuration for admins,
and sharing one's own agents on the network for every user.

The link itself lives in :mod:`botport_link`; these routes read its state and
persist its settings (system setting ``fd:botport-link``, secret sealed) and
the owners' own shares (``fd:botport-shares``).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from captain_claw.flight_deck import botport_link as bl
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import _fd_auth_enabled, get_current_user, get_db

router = APIRouter(prefix="/fd/botport", tags=["botport"])

_LOOPBACK = ("127.0.0.1", "::1", "localhost")
_shares_lock = asyncio.Lock()


def _require_local_when_open(request: Request) -> None:
    """With sign-in off everyone is the local admin: only this machine may
    change what other decks can drive."""
    if not _fd_auth_enabled():
        host = request.client.host if request.client else ""
        if host not in _LOOPBACK:
            raise HTTPException(403, "BotPort settings can only be changed from this machine while "
                                     "Flight Deck sign-in is off (or set FD_BOTPORT_* environment variables)")


async def link_admin(request: Request, admin: dict = Depends(require_admin)) -> dict:
    """An admin — and on a deck with sign-in off (where everyone is the local
    admin) only from this machine: the link's settings decide which agents
    other decks may drive."""
    _require_local_when_open(request)
    return admin


class LinkConfigUpdate(BaseModel):
    """Partial update; omitted fields keep their value. ``secret`` replaces the
    stored one only when non-empty (``clear_secret`` empties it)."""

    enabled: bool | None = None
    url: str | None = None
    deck_name: str | None = None
    key: str | None = None
    secret: str | None = None
    clear_secret: bool = False
    expose: list[str] | None = None
    accept_from: list[str] | None = None
    allow_outbound: bool | None = None
    outbound_users: bool | None = None
    max_concurrent: int | None = None
    max_inbound_timeout: float | None = None
    deny_tools: list[str] | None = None
    user_sharing: bool | None = None


class ShareUpdate(BaseModel):
    shared: bool


async def _stored() -> dict[str, Any]:
    return bl.unseal_settings(await get_db().get_system_setting(bl.SETTING_KEY))


def _public_config(stored: dict[str, Any]) -> dict[str, Any]:
    link = bl.get_link()
    eff = link.effective(stored).to_dict()
    secret_set = bool(eff.pop("secret", ""))
    env = bl.env_overrides()
    return {
        **eff,
        "secret_set": secret_set,
        "env_locked": sorted(k for k in env if k in eff or k == "secret"),
        "deck_id": link.deck_id,
        "problems": link.effective(stored).problems(),
        "normalized_url": _safe_normalize(eff.get("url", "")),
    }


def _safe_normalize(url: str) -> str:
    try:
        return bl.normalize_url(url) if url else ""
    except ValueError:
        return ""


@router.get("/status")
async def botport_status(user: dict = Depends(get_current_user)):
    """Link state. Admins get everything; others see what they can consult."""
    link = bl.get_link()
    full = link.status()
    if user.get("role") == "admin":
        return full
    # The other decks are the admins' unless they let every user's agents reach them.
    reach = bool(link.settings.outbound_users)
    return {
        "enabled": full["enabled"],
        "state": full["state"],
        "deck_name": full["deck_name"],
        "remote_agents": [r["name"] for r in link.remote_agents()] if reach else [],
        # The decks those agents run on, by what the hub's operator calls them.
        "decks": [{"name": d.get("name", ""), "label": str(d.get("label") or "")}
                  for d in full["network"].get("decks", []) if d.get("status") == "connected"]
        if reach else [],
    }


@router.get("/config")
async def botport_get_config(_admin: dict = Depends(link_admin)):
    return _public_config(await _stored())


@router.put("/config")
async def botport_put_config(body: LinkConfigUpdate, _admin: dict = Depends(link_admin)):
    stored = await _stored()
    updates = body.model_dump(exclude_none=True)
    clear_secret = bool(updates.pop("clear_secret", False))
    secret = updates.pop("secret", None)
    if "url" in updates:
        updates["url"] = updates["url"].strip()
        if updates["url"]:
            try:
                bl.normalize_url(updates["url"])
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from None
    if "deck_name" in updates:
        updates["deck_name"] = updates["deck_name"].strip()
        if updates["deck_name"] and not bl.valid_deck_name(updates["deck_name"]):
            raise HTTPException(400, "deck name must be letters, digits, '-', '_' or '.' "
                                     "(max 63, no spaces or '@')")
    stored.update(updates)
    if clear_secret:
        stored["secret"] = ""
    elif secret:
        stored["secret"] = secret.strip()
    # Normalise through the dataclass so only known, well-typed fields persist.
    clean = bl.LinkSettings.from_dict(stored).to_dict()
    await get_db().set_system_setting(bl.SETTING_KEY, bl.seal_settings(clean))
    await bl.get_link().apply(clean)
    return _public_config(clean)


@router.post("/reconnect")
async def botport_reconnect(_admin: dict = Depends(link_admin)):
    await bl.get_link().reconnect()
    return {"ok": True}


@router.get("/agents")
async def botport_local_agents(_admin: dict = Depends(link_admin)):
    """This deck's agents (all owners) with whether each is exposed."""
    return {"agents": await bl.get_link().local_agents()}


# ── Sharing one's own agents ─────────────────────────────────────


def _mine(agent: dict[str, Any], user: dict) -> bool:
    """With sign-in off every agent is the (only) user's."""
    return not _fd_auth_enabled() or (bool(agent["owner"]) and agent["owner"] == str(user.get("id") or ""))


async def _shares_view(user: dict) -> dict[str, Any]:
    link = bl.get_link()
    s = link.settings
    admin = user.get("role") == "admin"
    # Admins see every agent here (they may stop anyone's share); others their own.
    agents = await link.local_agents(owner=None if admin else str(user.get("id") or ""))
    return {
        "enabled": bool(s.enabled),
        "state": link.state,
        "deck_name": s.deck_name,
        "user_sharing": bool(s.user_sharing),
        # Sharing one's own agents is on here (an admin offers others' agents
        # through the Exposed agents list instead).
        "can_share": bool(s.enabled) and bool(s.user_sharing),
        "admin": admin,
        "agents": [
            {**{k: a[k] for k in ("name", "slug", "kind", "status", "exposed", "by_admin", "by_owner", "shareable")},
             "mine": _mine(a, user)}
            for a in agents
        ],
    }


@router.get("/shares")
async def botport_my_shares(user: dict = Depends(get_current_user)):
    """The caller's agents and whether each is on the network (and why)."""
    return await _shares_view(user)


_EXPOSE_HINT = " — an admin offers an agent to the network under Admin console → BotPort → Exposed agents"


@router.put("/shares/{slug}")
async def botport_set_share(slug: str, body: ShareUpdate, request: Request, kind: str = "",
                            user: dict = Depends(get_current_user)):
    """Share one of your agents on the network, or stop sharing it. Only its
    owner shares it; its owner or an admin stops a share. ``slug`` is the
    agent's slug (a name is accepted when no slug matches); ``kind``
    (docker / process) tells two agents of one name apart."""
    _require_local_when_open(request)
    link = bl.get_link()
    admin = user.get("role") == "admin"
    uid = str(user.get("id") or "")
    want = slug.strip().lower()
    kind = kind.strip().lower()
    agents = [a for a in await link.local_agents() if not kind or str(a["kind"]).lower() == kind]
    found = ([a for a in agents if bl.agent_key(a) == want]
             or [a for a in agents if str(a["name"]).strip().lower() == want])
    if not found:
        raise HTTPException(404, f"No agent '{slug}' on this Flight Deck")
    agent = next((a for a in found if a["owner"] and a["owner"] == uid), found[0])
    mine = _mine(agent, user)
    if body.shared:
        if not mine:
            raise HTTPException(403, "Only the agent's owner can share it on the network"
                                + (_EXPOSE_HINT if admin else ""))
        if not agent["shareable"]:
            raise HTTPException(400, f"'{agent['name']}' is one of Flight Deck's own workers; it cannot be shared")
        if not link.settings.enabled:
            raise HTTPException(409, "This Flight Deck is not connected to a BotPort network")
        if not link.settings.user_sharing:
            raise HTTPException(403, "Sharing by users is turned off on this Flight Deck"
                                + (_EXPOSE_HINT if admin else ""))
    elif not (mine or admin):
        raise HTTPException(403, "Only the agent's owner (or an admin) can stop sharing it")

    key = bl.share_key(agent)
    async with _shares_lock:
        db = get_db()
        shares = bl.parse_shares(await db.get_system_setting(bl.SHARES_KEY))
        if body.shared:
            shares[key] = {
                "owner": agent["owner"], "by": uid,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        else:
            shares.pop(key, None)
        await db.set_system_setting(bl.SHARES_KEY, bl.dump_shares(shares))
        link.set_user_shares(shares)
    return await _shares_view(user)


async def forget_share(kind: str, slug: str, owner: str) -> None:
    """An agent was removed or replaced: drop its owner's share, so a later
    agent of the same name is not on the network until someone shares it.
    Only that agent's share (kind, slug, owner) goes. Best-effort."""
    if not (slug or "").strip():
        return
    key = bl.share_key({"kind": kind, "slug": slug, "owner": owner})
    try:
        async with _shares_lock:
            db = get_db()
            shares = bl.parse_shares(await db.get_system_setting(bl.SHARES_KEY))
            if shares.pop(key, None) is None:
                return
            await db.set_system_setting(bl.SHARES_KEY, bl.dump_shares(shares))
            bl.get_link().set_user_shares(shares)
    except Exception:
        pass


@router.get("/probe")
async def botport_probe(url: str = "", _admin: dict = Depends(link_admin)):
    """Is the hub reachable? GETs its /healthz (does not register, so it never
    disturbs the live link)."""
    import httpx

    target = url or bl.get_link().settings.url
    try:
        ws_url = bl.normalize_url(target)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    health = bl.health_url(ws_url)
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(health)
        if resp.status_code == 404:
            return {"ok": False, "url": health, "error": "no /healthz there — not BotPort, or a BotPort "
                                                         "older than protocol 2 (upgrade it)"}
        if resp.status_code != 200:
            return {"ok": False, "url": health, "error": f"HTTP {resp.status_code} from {health}"}
        data = resp.json()
        if data.get("service") != "botport":
            return {"ok": False, "url": health, "error": "that address answered, but it is not BotPort"}
        proto = int(data.get("protocol") or 1)
        return {
            "ok": proto >= bl.PROTOCOL,
            "url": health,
            "version": data.get("version", ""),
            "protocol": proto,
            "error": "" if proto >= bl.PROTOCOL else
                     f"hub speaks protocol {proto}; Flight Deck links need protocol {bl.PROTOCOL} (upgrade BotPort)",
        }
    except Exception as exc:
        return {"ok": False, "url": health, "error": bl._describe(exc)}
