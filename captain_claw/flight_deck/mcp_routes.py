"""HTTP routes for Flight Deck's centrally-managed MCP servers.

Two audiences hit these endpoints:

1. **Connections UI** (``/fd/mcp/servers``, ``/fd/mcp/servers/<name>``,
   ``/fd/mcp/servers/<name>/test``) — gated by the normal Flight Deck
   user auth. Returns secrets in masked form only. A server is private
   to the user who added it (``mcp_storage``): others never see it, and
   a name they can't see answers 404 (or 409 when they try to reuse it).
   Deck-wide servers are an admin's opt-in and only admins change them.
   Anything touching a stdio server (a command FD spawns on its host) is
   admin-only.

2. **Captain-claw agents** (``/fd/mcp/<server>/tools``,
   ``/fd/mcp/<server>/call``) — gated like the Codex token endpoint:
   loopback / shared secret AND the agent's own ``X-Agent-Auth`` token.
   Agents proxy here instead of dialing the upstream MCP server
   themselves; that gives FD a single chokepoint for OAuth,
   rate-limiting and observability. An agent reaches only its owner's
   servers and the deck-wide ones, then the server's allowlist.
   Agents now call tools through ``/fd/mcp/agent/calls`` (start, then
   ``/follow`` in short requests, ``/cancel`` when the turn stops:
   :mod:`mcp_calls`), saying whom the call is for, so a server that asks
   the person first gets its question onto their card
   (:mod:`mcp_elicitation`). The old one-request ``/call`` stays for
   agents from before; its server's questions are cancelled.

3. **The sign-in popup** (``/fd/mcp/oauth/*``) — for a server with
   ``auth: "oauth"``, the browser leg of the MCP authorization flow:
   discovery + Dynamic Client Registration + PKCE (see
   :mod:`mcp_client_oauth`).

Phase 1 keeps the schema flat: name + URL + optional OAuth client
credentials + optional headers. Anything more (per-tenant scoping,
ACLs, audit log) is intentionally deferred.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from captain_claw.flight_deck import google_oauth_routes as _g
from captain_claw.flight_deck import mcp_calls, mcp_client_oauth, mcp_events, mcp_storage
from captain_claw.flight_deck.agent_identity import AgentCaller, require_agent_caller
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.mcp_elicitation import Asker
from captain_claw.flight_deck.mcp_manager import MCPServerError, get_manager
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/mcp", tags=["mcp"])


# ── agent auth (same gate as codex_oauth_routes) ────────────────────


def _authorize_agent_call(request: Request) -> AgentCaller:
    """Gate the proxy endpoints captain-claw agents call: one of THIS deck's
    agents, identified by its ``X-Agent-Auth`` token on top of the loopback /
    ``X-Agent-Secret`` transport gate (see :mod:`agent_identity`) — a web page
    in the user's browser is on loopback too, and must not drive the fleet's
    MCP connections.

    Returns the calling agent's slug and owner as FD recorded them at spawn —
    not the self-declared ``X-Agent-Slug`` — for :func:`_agent_may_use`. An
    empty slug (an agent FD knows but has no slug for) is only allowed servers
    with no allowlist configured; an empty owner only deck-wide servers.
    """
    return require_agent_caller(request, what="the MCP proxy")


def _agent_may_use(record: dict[str, Any] | None, caller: AgentCaller) -> bool:
    return record is not None and mcp_storage.agent_may_use(record, caller.slug, caller.owner)


# ── ownership gates (see mcp_storage: a server is private to its owner) ──


def _no_such_server(name: str) -> HTTPException:
    # Same answer for "doesn't exist" and "someone else's", so a private
    # server's existence never leaks.
    return HTTPException(status_code=404, detail=f"No MCP server named '{name}'")


def _visible(record: dict[str, Any], user: dict) -> bool:
    return mcp_storage.user_may_use(record, user) or mcp_storage.user_may_manage(record, user)


def _usable_server(name: str, user: dict) -> dict[str, Any]:
    """The record, when ``user`` may call its tools."""
    record = mcp_storage.get_server(name)
    if record is None or not _visible(record, user):
        raise _no_such_server(name)
    if not mcp_storage.user_may_use(record, user):
        raise HTTPException(
            status_code=403,
            detail=(
                f"Nobody owns MCP server '{name}' any more, so it can't be used — "
                "remove it and add it again"
            ),
        )
    return record


def _managed_server(name: str, user: dict) -> dict[str, Any]:
    """The record, when ``user`` may change it."""
    record = mcp_storage.get_server(name)
    if record is None or not _visible(record, user):
        raise _no_such_server(name)
    if not mcp_storage.user_may_manage(record, user):
        raise HTTPException(
            status_code=403,
            detail=f"Only an admin can change the deck-wide MCP server '{name}'",
        )
    return record


async def migrate_legacy_ownership(db: Any) -> None:
    """FD startup: give pre-ownership servers an owner (``mcp_storage``).
    An OAuth server goes to whoever signed it in — by the user id newer
    grants carry, else by the email older ones recorded — and stays private
    and unusable when that user can't be found. Never raises."""

    async def _signer(record: dict[str, Any]) -> str:
        user_id, label = mcp_client_oauth.signer(record)
        if user_id or db is None or "@" not in label:
            return user_id
        user = await db.get_user_by_email(label)
        return str((user or {}).get("id") or "")

    try:
        changed = await mcp_storage.migrate_legacy_ownership(_signer)
    except Exception as exc:
        log.warning("MCP ownership migration failed", error=str(exc))
        return
    for name in changed:
        rec = mcp_storage.get_server(name) or {}
        log.info(
            "MCP server ownership set",
            server=name,
            deck_wide=bool(rec.get("deck_wide")),
            owner=rec.get("owner_id") or "(none — unusable until re-added)",
        )


# ── stdio gate ──────────────────────────────────────────────────────


def _require_stdio_admin(user: dict) -> None:
    """A stdio server is a command FD spawns on its own host, so only an admin
    may add one, switch a server to it, edit / remove / test one, or probe an
    ad-hoc stdio config. Any signed-in user may add HTTP servers of their own
    (the kiosk Connections dialog). An auth-off deck has one trusted user and
    ``get_current_user`` hands it a synthetic admin, so it passes unchanged."""
    if (user or {}).get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail=(
                "Only an admin can manage stdio MCP servers — they run a "
                "command on the Flight Deck host."
            ),
        )


def _is_stdio(record: dict | None) -> bool:
    return bool(record) and str(record.get("transport") or "").strip().lower() == "stdio"


# ── admin: list / add / remove servers ──────────────────────────────


@router.get("/servers")
async def list_servers(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """The servers this user may see — their own, the deck-wide ones (and,
    for an admin, ownerless ones to clean up) — with secrets masked, runtime
    status, and what the user may do with each."""
    out = []
    for entry in get_manager().status_snapshot():
        if not _visible(entry, user):
            continue
        out.append({
            **entry,
            "can_use": mcp_storage.user_may_use(entry, user),
            "can_manage": mcp_storage.user_may_manage(entry, user),
        })
    return {"servers": out}


@router.post("/servers")
async def add_or_update_server(
    payload: dict = Body(...),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Insert or update a server by ``name``. Existing client_secret is
    preserved when the new payload has an empty / masked client_secret."""
    name = str(payload.get("name") or "").strip()
    transport = str(payload.get("transport") or "http").strip().lower() or "http"
    url = str(payload.get("url") or "").strip()
    command = str(payload.get("command") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    existing = mcp_storage.get_server(name)
    if existing is not None and not mcp_storage.user_may_manage(existing, user):
        if _visible(existing, user):
            raise HTTPException(
                status_code=403,
                detail=f"Only an admin can change the deck-wide MCP server '{name}'",
            )
        # Names are deck-wide keys; say it's taken, never whose it is.
        raise HTTPException(
            status_code=409,
            detail=f"An MCP server named '{name}' already exists — pick another name",
        )
    # Creating a stdio server, switching one to stdio, or touching an existing
    # stdio server in any way (command/args/env, enabled, or overwriting it
    # with an http config) is admin-only.
    if transport == "stdio" or _is_stdio(existing):
        _require_stdio_admin(user)
    owner_id, deck_wide = _ownership(payload, existing, user)
    if transport == "http" and not url:
        raise HTTPException(status_code=400, detail="url is required for http transport")
    if transport == "stdio" and not command:
        raise HTTPException(
            status_code=400, detail="command is required for stdio transport"
        )

    incoming_secret = str(payload.get("client_secret") or "")
    # The list view returns a placeholder of "•••..." — preserve the
    # stored secret if the user re-submits without changing that field.
    looks_masked = incoming_secret and set(incoming_secret) <= {"•", "*"}
    if looks_masked or not incoming_secret:
        if existing is not None:
            incoming_secret = existing.get("client_secret", "")

    record = {
        "name": name,
        "transport": transport,
        "url": url,
        "command": command,
        "args": payload.get("args") or [],
        "env": payload.get("env") or {},
        "auth": str(payload.get("auth") or ""),
        "scope": str(payload.get("scope") or ""),
        "client_id": str(payload.get("client_id") or ""),
        "client_secret": incoming_secret,
        "token_endpoint": str(payload.get("token_endpoint") or ""),
        "headers": payload.get("headers") or {},
        "enabled": bool(payload.get("enabled", True)),
        "allowed_agents": payload.get("allowed_agents") or [],
        "owner_id": owner_id,
        "deck_wide": deck_wide,
    }
    existed_before = existing is not None
    saved = await mcp_storage.upsert_server(record)
    if existing is not None and (
        not mcp_client_oauth.is_oauth(saved)
        or existing.get("url") != saved.get("url")
        or existing.get("client_id") != saved.get("client_id")
    ):
        # The sign-in was for another server, or another client — drop it.
        await mcp_client_oauth.forget(name)
    # Drop any cached upstream session so the new config takes effect.
    # Awaited so a running stdio child gets terminated before we return —
    # otherwise the next call would race a brand-new state against the
    # still-shutting-down old one.
    await get_manager().forget_server(name)
    # Tell subscribed agents to refresh.  We publish *after* both
    # storage and manager state have settled so the subscriber's
    # follow-up GET observes the new world.
    if existed_before:
        mcp_events.publish_server_updated(name)
    else:
        mcp_events.publish_server_added(name)
    return {"ok": True, "server": mcp_storage._public_view(saved)}


def _ownership(payload: dict, existing: dict[str, Any] | None, user: dict) -> tuple[str, bool]:
    """``(owner_id, deck_wide)`` for a save: a new server belongs to its
    creator; an existing one keeps its owner. Only an admin may make a server
    deck-wide or take that back."""
    is_admin = user.get("role") == "admin"
    requested = payload.get("deck_wide")
    if existing is None:
        owner_id = str(user.get("id") or "")
        deck_wide = bool(requested)
    else:
        owner_id = str(existing.get("owner_id") or "")
        deck_wide = bool(existing.get("deck_wide")) if requested is None else bool(requested)
    if deck_wide != bool((existing or {}).get("deck_wide")) and not is_admin:
        raise HTTPException(
            status_code=403,
            detail="Only an admin can share an MCP server with everyone on this deck",
        )
    if not deck_wide and not owner_id:
        # An admin taking a legacy deck-wide server private keeps it.
        owner_id = str(user.get("id") or "")
    return owner_id, deck_wide


@router.post("/servers/{name}/sharing")
async def set_server_sharing(
    name: str,
    payload: dict = Body(...),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Admin → Integrations: share a server with everyone on the deck, or
    take that back (``{"deck_wide": bool}``). Changes nothing else — the
    full save above needs the whole record."""
    record = _managed_server(name, user)
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only an admin can share an MCP server with everyone on this deck",
        )
    owner_id, deck_wide = _ownership({"deck_wide": bool(payload.get("deck_wide"))}, record, user)
    saved = await mcp_storage.upsert_server({**record, "owner_id": owner_id, "deck_wide": deck_wide})
    # Agents re-read their tool lists; access itself is checked per call.
    mcp_events.publish_server_updated(name)
    return {"ok": True, "server": mcp_storage._public_view(saved)}


@router.delete("/servers/{name}")
async def remove_server(
    name: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    if _is_stdio(_managed_server(name, user)):
        _require_stdio_admin(user)
    deleted = await mcp_storage.delete_server(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    await get_manager().forget_server(name)
    await mcp_client_oauth.forget(name)
    mcp_events.publish_server_removed(name)
    return {"ok": True}


@router.post("/servers/{name}/test")
async def test_server(
    name: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Probe a configured server: re-init, list tools, surface any error."""
    record = _usable_server(name, user)
    if _is_stdio(record):
        _require_stdio_admin(user)
    return await get_manager().test_server(name)


@router.post("/probe")
async def probe_transient(
    payload: dict = Body(...),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Probe an ad-hoc server config without persisting it.

    Used by the "Test connection" button before the user actually saves
    a server. The payload accepts the same shape as ``POST /servers``
    (``name``, ``url``, ``client_id``, ``client_secret``, ``token_endpoint``,
    ``headers``). When the submitted ``client_secret`` is empty/masked
    and a server with the same ``name`` already exists, the saved
    secret is used so users can re-test without re-entering it.
    """
    transport = str(payload.get("transport") or "http").strip().lower() or "http"
    url = str(payload.get("url") or "").strip()
    command = str(payload.get("command") or "").strip()
    if transport == "stdio":
        # A stdio probe spawns ``command`` right now.
        _require_stdio_admin(user)
    if transport == "http" and not url:
        raise HTTPException(status_code=400, detail="url is required for http transport")
    if transport == "stdio" and not command:
        raise HTTPException(
            status_code=400, detail="command is required for stdio transport"
        )
    name = str(payload.get("name") or "").strip()
    existing = mcp_storage.get_server(name) if name else None
    if existing is not None and not mcp_storage.user_may_manage(existing, user):
        raise HTTPException(
            status_code=409,
            detail=f"An MCP server named '{name}' already exists — pick another name",
        )
    # Only the server's own manager re-tests with its stored secret / sign-in;
    # anyone else probes under a throwaway name that matches no grant.
    reuse = existing is not None and mcp_storage.user_may_use(existing, user)

    incoming_secret = str(payload.get("client_secret") or "")
    looks_masked = incoming_secret and set(incoming_secret) <= {"•", "*"}
    if looks_masked:
        incoming_secret = ""
    if not incoming_secret and reuse:
        incoming_secret = existing.get("client_secret", "")

    record = {
        "name": name if reuse else f"_probe-{secrets.token_hex(6)}",
        "transport": transport,
        "url": url,
        "command": command,
        "args": payload.get("args") or [],
        "env": payload.get("env") or {},
        "auth": str(payload.get("auth") or ""),
        "scope": str(payload.get("scope") or ""),
        "client_id": str(payload.get("client_id") or ""),
        "client_secret": incoming_secret,
        "token_endpoint": str(payload.get("token_endpoint") or ""),
        "headers": payload.get("headers") or {},
        "enabled": True,
        "allowed_agents": payload.get("allowed_agents") or [],
    }
    return await get_manager().probe_record(record)


# ── user-facing: OAuth sign-in to a remote server (DCR) ─────────────
#
# The protocol lives in mcp_client_oauth; these routes run its browser leg
# exactly like the Google / Microsoft popups: an authenticated POST mints a
# single-use ticket (also an HttpOnly SameSite=Strict cookie), /login takes it
# only from that browser and sends the popup to the server's authorization
# endpoint, and /callback completes only a state whose cookie this browser
# holds. Every cookie lives under /fd/mcp/oauth.

_OAUTH_MESSAGE_TYPE = "captain-claw-mcp-oauth"
_OAUTH_COOKIE_PATH = "/fd/mcp/oauth"
_OAUTH_STATE_COOKIE = "fd_mcp_oauth"
_OAUTH_TICKET_COOKIE = "fd_mcp_ticket"
_OAUTH_PENDING_TTL = 600
_OAUTH_TICKET_TTL = 120

_pending_oauth: dict[str, dict[str, Any]] = {}
_oauth_tickets: dict[str, dict[str, Any]] = {}


def _purge_stale_oauth() -> None:
    now = time.time()
    for store, ttl in ((_pending_oauth, _OAUTH_PENDING_TTL), (_oauth_tickets, _OAUTH_TICKET_TTL)):
        for k in [k for k, v in store.items() if v.get("ts", 0) < now - ttl]:
            store.pop(k, None)


def _oauth_redirect_uri(request: Request) -> str:
    """Registered with the server's authorization server at sign-in time, so
    no admin has to copy it anywhere (unless the server lacks DCR)."""
    public = _g._public_url()
    if public:
        return f"{public}/fd/mcp/oauth/callback"
    host = request.headers.get("host") or request.url.netloc
    return f"{request.url.scheme}://{host}/fd/mcp/oauth/callback"


def _oauth_page(*, ok: bool, title: str, detail: str = "", status_code: int = 200) -> HTMLResponse:
    return _g._callback_html(
        ok=ok, title=title, detail=detail, status_code=status_code,
        message_type=_OAUTH_MESSAGE_TYPE,
    )


def _oauth_server(name: str, user: dict) -> dict[str, Any]:
    record = _managed_server(name, user)
    if not mcp_client_oauth.is_oauth(record):
        raise HTTPException(
            status_code=400,
            detail=f"MCP server '{name}' doesn't use OAuth sign-in — set its Auth to 'OAuth sign-in' first",
        )
    return record


@router.post("/oauth/connect-ticket")
async def oauth_connect_ticket(
    request: Request,
    payload: dict = Body(...),
    user: dict = Depends(get_current_user),
) -> JSONResponse:
    """A single-use ticket that lets THIS browser start a sign-in to ``server``."""
    name = str(payload.get("server") or "").strip()
    _oauth_server(name, user)
    _purge_stale_oauth()
    label = str(user.get("email") or user.get("display_name") or user.get("id") or "")
    ticket = secrets.token_urlsafe(32)
    _oauth_tickets[ticket] = {
        "server": name, "by": label, "ts": time.time(),
        "by_user": {"id": str(user.get("id") or ""), "role": user.get("role") or ""},
    }
    resp = JSONResponse({"ticket": ticket}, headers={"Cache-Control": "no-store"})
    resp.set_cookie(
        key=_OAUTH_TICKET_COOKIE, value=ticket, max_age=_OAUTH_TICKET_TTL, httponly=True,
        samesite="strict", path=_OAUTH_COOKIE_PATH, secure=_g._state_cookie_secure(request),
    )
    return resp


@router.get("/oauth/login")
async def oauth_login(request: Request) -> Response:
    """Discover + register (DCR) as needed, then send the popup to the
    server's authorization endpoint."""
    ticket = request.query_params.get("ticket", "")
    entry = _oauth_tickets.pop(ticket, None) if ticket else None
    if (
        not entry
        or time.time() - float(entry.get("ts") or 0) > _OAUTH_TICKET_TTL
        or not _g._same_secret(ticket, request.cookies.get(_OAUTH_TICKET_COOKIE, ""))
    ):
        return _oauth_page(
            ok=False,
            title="Sign-in link expired",
            detail=(
                "This link is expired, already used, or was opened outside the "
                "Flight Deck tab that created it. Click Sign in again."
            ),
            status_code=401,
        )
    record = mcp_storage.get_server(str(entry["server"]))
    signer = entry.get("by_user") or {}
    if (
        record is None
        or not mcp_client_oauth.is_oauth(record)
        or not mcp_storage.user_may_manage(record, signer)
    ):
        return _oauth_page(ok=False, title="MCP server not found", status_code=404)
    _purge_stale_oauth()
    state = secrets.token_urlsafe(32)
    try:
        url, pending = await mcp_client_oauth.begin_sign_in(
            record, _oauth_redirect_uri(request), state,
        )
    except mcp_client_oauth.MCPOAuthError as exc:
        return _oauth_page(ok=False, title=f"Can't sign in to {record['name']}", detail=str(exc))
    browser = secrets.token_urlsafe(32)
    _pending_oauth[state] = {
        **pending, "by": entry.get("by", ""), "by_user": signer, "browser": browser, "ts": time.time(),
    }
    resp = RedirectResponse(url, status_code=302)
    secure = _g._state_cookie_secure(request)
    resp.set_cookie(
        key=_OAUTH_STATE_COOKIE, value=browser, max_age=_OAUTH_PENDING_TTL, httponly=True,
        samesite="lax", path=_OAUTH_COOKIE_PATH, secure=secure,
    )
    resp.delete_cookie(
        _OAUTH_TICKET_COOKIE, path=_OAUTH_COOKIE_PATH, secure=secure, httponly=True, samesite="strict",
    )
    return resp


@router.get("/oauth/callback")
async def oauth_callback(request: Request) -> HTMLResponse:
    """The authorization server's redirect back: swap the code for tokens."""
    resp = await _complete_oauth_callback(request)
    resp.delete_cookie(
        _OAUTH_STATE_COOKIE, path=_OAUTH_COOKIE_PATH, secure=_g._state_cookie_secure(request),
        httponly=True, samesite="lax",
    )
    return resp


async def _complete_oauth_callback(request: Request) -> HTMLResponse:
    state = request.query_params.get("state", "")
    pending = _pending_oauth.pop(state, None) if state else None
    if pending and time.time() - float(pending.get("ts") or 0) > _OAUTH_PENDING_TTL:
        pending = None

    error = request.query_params.get("error")
    if error:
        desc = request.query_params.get("error_description", error)
        return _oauth_page(ok=False, title="Sign-in error", detail=desc.splitlines()[0][:400])
    code = request.query_params.get("code", "")
    if not code or not state:
        return _oauth_page(ok=False, title="Missing code or state")
    if not pending:
        return _oauth_page(ok=False, title="Invalid or expired state, please try again.")
    if not _g._same_secret(str(pending.get("browser") or ""), request.cookies.get(_OAUTH_STATE_COOKIE, "")):
        log.warning("MCP OAuth callback refused: state not started in this browser")
        return _oauth_page(
            ok=False,
            title="Sign-in not started in this browser",
            detail="Nothing was saved. Open Flight Deck in this browser and click Sign in again.",
            status_code=400,
        )
    name = str(pending["server"])
    record = mcp_storage.get_server(name)
    signer = pending.get("by_user") or {}
    if (
        record is None
        or record.get("url") != pending.get("server_url")
        or not mcp_client_oauth.is_oauth(record)
        or not mcp_storage.user_may_manage(record, signer)
    ):
        return _oauth_page(ok=False, title="The MCP server changed during sign-in — nothing was saved.")
    try:
        await mcp_client_oauth.complete_sign_in(
            pending, code,
            connected_by=str(pending.get("by") or ""),
            connected_by_id=str(signer.get("id") or ""),
            issuer=request.query_params.get("iss", ""),
        )
    except mcp_client_oauth.MCPOAuthError as exc:
        log.warning("MCP OAuth token exchange failed", server=name, error=str(exc))
        return _oauth_page(ok=False, title="Token exchange failed", detail=str(exc))
    # New credentials: drop the cached session so the next call uses them,
    # and tell agents to fetch the (now reachable) tool list.
    await get_manager().forget_server(name)
    mcp_events.publish_server_updated(name)
    return _oauth_page(ok=True, title="Connected", detail=f"Flight Deck is signed in to {name}.")


@router.post("/oauth/disconnect")
async def oauth_disconnect(
    payload: dict = Body(...),
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Sign Flight Deck out of ``server`` (revoking the token when the server
    supports it)."""
    name = str(payload.get("server") or "").strip()
    record = _oauth_server(name, _user)
    await mcp_client_oauth.sign_out(record)
    await get_manager().forget_server(name)
    mcp_events.publish_server_updated(name)
    return {"ok": True}


# ── shared SSE helpers (used by both streaming routes below) ────────


# How often to send a comment-only "ping" through an idle SSE
# connection. Many proxies (and our own dev nginx) drop a TCP socket
# after ~60s of no bytes, so we send at half that.
_SSE_PING_INTERVAL_SECONDS = 25.0


def _sse_format(event: dict[str, Any]) -> bytes:
    """Encode a payload dict as a single SSE ``data:`` frame."""
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode("utf-8")


# ── agent-facing: discovery + tools/list + tools/call passthrough ───


@router.get("/agent/servers")
async def agent_list_servers(request: Request) -> dict[str, Any]:
    """Return enabled server names for an agent to enumerate.

    Filtered by the calling agent's slug against each server's
    ``allowed_agents`` list — servers with an empty allowlist are
    visible to every agent (Phase 1 behaviour preserved for
    backwards-compat).

    Secrets are *not* included — agents never need them, since every
    upstream call routes through this proxy. Use ``/fd/mcp/<name>/tools``
    for per-server discovery.
    """
    caller = _authorize_agent_call(request)
    records = mcp_storage.load_servers()
    out: list[dict[str, Any]] = []
    for rec in records:
        if not rec.get("enabled", True):
            continue
        if not _agent_may_use(rec, caller):
            continue
        out.append({"name": rec["name"]})
    return {"servers": out}


@router.get("/{name}/tools")
async def proxy_tools_list(
    name: str,
    request: Request,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return the cached ``tools/list`` for ``name``. Agents hit this
    instead of running their own MCP discovery, so OAuth + session is
    single-flighted across the fleet."""
    caller = _authorize_agent_call(request)
    record = mcp_storage.get_server(name)
    if not _agent_may_use(record, caller):
        # Match the 404 shape of "not configured" so disallowed servers
        # are indistinguishable from non-existent ones — don't leak the
        # existence of restricted servers to agents that can't use them.
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    try:
        tools = await get_manager().list_tools(name, force_refresh=refresh)
    except MCPServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"server": name, "tools": tools}


@router.post("/{name}/call")
async def proxy_tool_call(
    name: str,
    request: Request,
    payload: dict = Body(...),
) -> dict[str, Any]:
    caller = _authorize_agent_call(request)
    record = mcp_storage.get_server(name)
    if not _agent_may_use(record, caller):
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
    if not tool_name:
        raise HTTPException(status_code=400, detail="payload.tool is required")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="arguments must be an object")
    try:
        result = await get_manager().call_tool(name, tool_name, arguments)
    except MCPServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"server": name, "tool": tool_name, "result": result}


# ── agent-facing: a followed call, for whoever the turn is for ──────

# The route a member's turn reaches with its grant (speaker_grants.GRANT_AWARE_PATHS):
# the server's questions go to the member whose turn it is.
GRANT_AWARE = frozenset({"/fd/mcp/agent/calls"})


class AgentCallBody(BaseModel):
    server: str = Field(..., min_length=1, max_length=200)
    tool: str = Field(..., min_length=1, max_length=256)
    arguments: dict = Field(default_factory=dict)
    # The turn's source (turn_source.py): a run nobody attends can't be asked.
    origin: str = Field("", max_length=60)
    session_id: str = Field("", max_length=200)
    # A member's turn without a grant: the call runs, but nobody can be asked.
    nobody: bool = False
    wait: float = Field(20, ge=0, le=30)
    call_id: str = Field(..., pattern=r"^xm_[0-9a-f]{16}$")


class AgentFollowBody(BaseModel):
    id: str = Field(..., max_length=40)
    wait: float = Field(25, ge=0, le=30)


async def _asker_for(request: Request, body: AgentCallBody) -> Asker | None:
    """The person this call is for: the member whose turn it is (by its
    grant), else the agent's owner; None when there's nobody to ask."""
    if body.nobody:
        return None
    from captain_claw.flight_deck import speaker_grants
    from captain_claw.flight_deck.agent_names import display_name
    from captain_claw.flight_deck.ask_routes import _runner

    try:
        runner = await _runner(request)
    except HTTPException:
        if speaker_grants.member_request(request):
            raise  # a member's turn never falls back to the owner
        return None  # an agent with no owner on record: nobody to ask
    try:
        name = await display_name(runner.agent, runner.user_id)
    except Exception:  # noqa: BLE001 — the card still names the server
        name = ""
    return Asker(user_id=runner.user_id, agent=runner.agent, agent_name=name or runner.agent,
                 origin=body.origin, session_id=body.session_id)


async def _followed(run: mcp_calls.Run, wait: float) -> dict[str, Any]:
    try:
        out = await mcp_calls.follow(run, wait)
    except (MCPServerError, mcp_calls.CallError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 — the upstream call's own failure
        raise HTTPException(status_code=502, detail=str(exc) or type(exc).__name__)
    if "result" in out:
        return {"server": run.server, "tool": run.tool, "result": out["result"]}
    return out


@router.post("/agent/calls")
async def agent_call(body: AgentCallBody, request: Request) -> dict[str, Any]:
    """Start a tool call for the person this turn is for: its result, or
    ``{"pending": id, "asking": bool}`` to follow."""
    caller = _authorize_agent_call(request)
    record = mcp_storage.get_server(body.server)
    if not _agent_may_use(record, caller):
        raise HTTPException(status_code=404, detail=f"No MCP server named '{body.server}'")
    asker = await _asker_for(request, body)
    try:
        run = mcp_calls.start(call_id=body.call_id, caller=(caller.slug, caller.owner), server=body.server,
                              tool=body.tool, arguments=body.arguments, asker=asker,
                              call=get_manager().call_tool)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return await _followed(run, body.wait)


@router.post("/agent/calls/follow")
async def agent_call_follow(body: AgentFollowBody, request: Request) -> dict[str, Any]:
    caller = _authorize_agent_call(request)
    try:
        run = mcp_calls.get(body.id, (caller.slug, caller.owner))
    except KeyError:
        raise HTTPException(status_code=404, detail="There's no such call (it ended a while ago).") from None
    return await _followed(run, body.wait)


@router.post("/agent/calls/cancel")
async def agent_call_cancel(body: AgentFollowBody, request: Request) -> dict[str, Any]:
    caller = _authorize_agent_call(request)
    try:
        run = mcp_calls.get(body.id, (caller.slug, caller.owner))
    except KeyError:
        return {"cancelled": False}
    return {"cancelled": mcp_calls.cancel(run)}


# ── user-facing: tool list + call for the agent-app runtime ─────────


@router.get("/{name}/user_tools")
async def user_tools_list(
    name: str,
    refresh: bool = False,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """User-authed counterpart of ``/{name}/tools``. Used by the
    Flight-Deck app runtime so the browser can introspect a server
    without holding an agent shared secret — the user's own servers and
    the deck-wide ones only."""
    _usable_server(name, user)
    try:
        tools = await get_manager().list_tools(name, force_refresh=refresh)
    except MCPServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"server": name, "tools": tools}


@router.post("/{name}/user_call")
async def user_tool_call(
    name: str,
    payload: dict = Body(...),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """User-authed counterpart of ``/{name}/call``. Lets the app
    runtime in the browser invoke MCP tools as the logged-in user
    (no agent slug, no shared secret) — on their own servers and the
    deck-wide ones only. A question the server asks meanwhile goes to
    that user's card."""
    _usable_server(name, user)
    tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
    if not tool_name:
        raise HTTPException(status_code=400, detail="payload.tool is required")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="arguments must be an object")
    uid = str(user.get("id") or "")
    asker = Asker(user_id=uid, origin="web") if uid else None
    try:
        result = await get_manager().call_tool(name, tool_name, arguments, asker=asker)
    except MCPServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"server": name, "tool": tool_name, "result": result}


# ── agent-facing: streaming tool call (Phase 2.4) ───────────────────


@router.post("/{name}/call_stream")
async def proxy_tool_call_stream(
    name: str,
    request: Request,
    payload: dict = Body(...),
) -> StreamingResponse:
    """Streaming variant of ``/{name}/call``.

    Returns an SSE stream of two event flavours:

    * ``{"type": "progress", "params": {...}}`` — one frame per
      ``notifications/progress`` from the upstream MCP server.
    * ``{"type": "result",   "result": {...}}`` — exactly one final
      frame with the same ``result`` shape ``/{name}/call`` returns.
    * ``{"type": "error",    "error": "..."}`` — sent in place of
      ``result`` when the upstream call fails.

    Callers that don't care about progress should keep using
    ``/{name}/call``; this endpoint exists for UIs that want to
    surface "tool is running, 30%…" indicators while the model
    waits.
    """
    caller = _authorize_agent_call(request)
    record = mcp_storage.get_server(name)
    if not _agent_may_use(record, caller):
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
    if not tool_name:
        raise HTTPException(status_code=400, detail="payload.tool is required")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="arguments must be an object")

    # Bridge the synchronous ``on_progress`` callback (called from the
    # transport reader) into an asyncio.Queue we can drain from the
    # streaming response. Using a queue rather than wiring directly
    # into the StreamingResponse generator keeps the transport's
    # callback invariants simple — it never has to await anything.
    progress_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1024)
    sentinel: dict[str, Any] = {"_done": True}

    def _on_progress(params: dict[str, Any]) -> None:
        try:
            progress_queue.put_nowait(params)
        except asyncio.QueueFull:
            log.warning(
                "MCP call_stream: progress queue full, dropping frame",
                server=name,
                tool=tool_name,
            )

    async def event_iterator():
        # Run the upstream call as a background task so we can pull
        # progress frames off the queue while it's in flight.
        manager = get_manager()
        call_task = asyncio.create_task(
            manager.call_tool(
                name, tool_name, arguments, on_progress=_on_progress
            ),
            name=f"mcp-call[{name}/{tool_name}]",
        )
        # Wake the queue when the call finishes so we can break.
        call_task.add_done_callback(
            lambda _t: progress_queue.put_nowait(sentinel)
        )
        try:
            while True:
                if await request.is_disconnected():
                    call_task.cancel()
                    break
                try:
                    item = await asyncio.wait_for(
                        progress_queue.get(), timeout=_SSE_PING_INTERVAL_SECONDS
                    )
                except asyncio.TimeoutError:
                    yield _sse_format({"type": "ping"})
                    continue
                if item is sentinel:
                    break
                yield _sse_format({"type": "progress", "params": item})
            # Drain whatever the call returned (or its error).
            try:
                result = await call_task
                yield _sse_format(
                    {"type": "result", "server": name, "tool": tool_name, "result": result}
                )
            except asyncio.CancelledError:
                # Caller disconnected; nothing more to emit.
                return
            except MCPServerError as exc:
                yield _sse_format({"type": "error", "error": str(exc)})
            except Exception as exc:
                yield _sse_format({"type": "error", "error": str(exc)})
        finally:
            if not call_task.done():
                call_task.cancel()

    return StreamingResponse(
        event_iterator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── agent-facing: hot-push event stream (Phase 2.3) ─────────────────


@router.get("/agent/events")
async def agent_event_stream(request: Request) -> StreamingResponse:
    """SSE stream of MCP state changes filtered by the caller's allowlist.

    Captain-claw agents subscribe on boot and on each event they
    receive they re-run :func:`register_mcp_tools` so the model sees
    fresh tool catalogues without a restart.

    Events are only delivered if the calling agent is allowed to use
    the affected server (see
    :func:`mcp_storage.agent_may_use`). That keeps restricted-server
    existence opaque to disallowed agents — same rule as the rest of
    the agent-facing surface.

    The connection is kept alive with a periodic ``ping`` event so it
    survives proxies / NATs that drop idle TCP. The client should
    treat ``ping`` as a no-op.
    """
    caller = _authorize_agent_call(request)
    bus = mcp_events.get_event_bus()
    queue = await bus.subscribe()

    async def event_iterator():
        try:
            # Send an initial hello so the client knows the connection
            # is live before any real event arrives.
            yield _sse_format({"type": "hello", "agent": caller.slug})
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=_SSE_PING_INTERVAL_SECONDS
                    )
                except asyncio.TimeoutError:
                    # No real event in the window; emit a ping to keep
                    # intermediaries from closing the socket.
                    yield _sse_format({"type": "ping"})
                    continue
                # Filter per-server events by the caller's allowlist.
                # Pings / hellos never have a ``server`` field and are
                # always forwarded.
                server_name = event.get("server")
                if server_name:
                    record = mcp_storage.get_server(str(server_name))
                    if record is not None and not _agent_may_use(record, caller):
                        continue
                    # ``server_removed`` fires *after* deletion, so
                    # ``record is None`` is normal there — forward
                    # anyway so the agent can drop its proxies.
                yield _sse_format(event)
        finally:
            await bus.unsubscribe(queue)

    return StreamingResponse(
        event_iterator(),
        media_type="text/event-stream",
        headers={
            # Prevent buffering by intermediaries (nginx in particular).
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
