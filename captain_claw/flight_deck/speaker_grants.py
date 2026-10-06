"""Per-turn grants that let a shared agent act as the member it is answering (A2).

A1 let a member chat with another user's agent; A2 lets that member's OWN
Google account, deep memory and files be used during their own turns. The agent
process belongs to the owner, so it never holds the member's credentials: while
one of the member's messages is being answered it holds a short-lived **grant**
and asks Flight Deck, which acts as the member or refuses.

* **Minting.** ``agent_sharing_routes`` mints one grant per member chat turn on
  a *process* agent (docker agents stay chat-only for members) and puts the
  token in that chat frame only (``_fd_grant``). FD keeps ``sha256(token)``, in
  memory; the token itself is never stored, logged or sent to a browser.
* **Closing.** A grant closes on its turn's ``turn_end``, on revocation (share
  deleted, member left, agent removed, owner changed, membership lost), 20
  minutes after mint, 120 s after the member socket that sent the message
  closes, and with the FD process. It is never re-opened.
* **Using.** The agent sends ``X-FD-Speaker-Grant: <token>`` together with the
  query marker ``fd_member=1`` to the grant-aware routes (``GRANT_AWARE_PATHS``).
  :func:`acting_member` validates it (contract part 0 §6) and never falls back
  to the owner once either marker is present. :class:`GrantGuardMiddleware`
  refuses the markers on every other route, so a member request can't reach an
  owner-keyed route by accident (or by a proxy stripping the header).
* **Google opt-in.** A member's Google is used only after they turned it on for
  that agent — a ``user_settings`` row in the MEMBER's settings whose value is
  the agent's owner at opt-in time, so consent never carries over to a new owner.

FD is one process (A1 assumption): the store is a plain dict and every grant
function is synchronous (one event loop).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import secrets
import time
from dataclasses import dataclass
from urllib.parse import parse_qs

from fastapi import HTTPException, Request
from starlette.types import ASGIApp, Receive, Scope, Send

from captain_claw.logging import get_logger

log = get_logger(__name__)

# ── Wire + constants (contract part 0 §8) ─────────────────────────────────

GRANT_HEADER = "X-FD-Speaker-Grant"
MEMBER_MARKER_PARAM = "fd_member"            # query parameter, value "1"
CHAT_GRANT_FIELD = "_fd_grant"
GRANT_TOKEN_RE = r"^[A-Za-z0-9_-]{43}$"

GRANT_TTL_S = 1200
ORPHAN_GRACE_S = 120
MAX_OPEN_GRANTS = 4096
MAX_OPEN_GRANTS_PER_AGENT = 256
MAX_OPEN_GRANTS_PER_MEMBER = 12
# 12 = A1 MAX_MEMBER_SOCKETS_PER_AGENT (6) × MAX_OUTSTANDING_TURNS_PER_CONN (2)

GRANT_AWARE_PATHS = frozenset({
    "/fd/google/access_token", "/fd/google/agent_status", "/fd/google/gmail/send",
    "/fd/deep-memory/agent/search", "/fd/deep-memory/agent/index",
    "/fd/deep-memory/agent/delete",
    "/fd/context-packs/agent/vfs",
})  # exact match on the path with root_path removed, rstrip("/")

NO_TURN_DETAIL = "No active shared-agent turn for this request"
NOT_MEMBER_DETAIL = "This member no longer has access to this agent"
GOOGLE_OFF_DETAIL = ("Not enabled for this agent: the member hasn't turned on "
                     "“Let this agent use my Google during my chats”")
OFF_PATH_DETAIL = "Not available during a shared-agent member's turn"
MEMBER_DELETE_DETAIL = "In a shared chat, deep-memory deletes need a reference"
GOOGLE_OPTIN_PREFIX = "fd:shared-agent-google:"
# key = prefix + agent_ref, in the MEMBER's user_settings; value = the agent's
# owner id at opt-in time (an opt-in counts only while that is still the owner)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{43}")
_HEADER_BYTES = b"x-fd-speaker-grant"
_HEADER_LOWER = _HEADER_BYTES.decode("ascii")

_BROWSER_DETAIL = "This endpoint is for Flight Deck agents, not browsers"


# ── Store ─────────────────────────────────────────────────────────────────


@dataclass
class Grant:
    key: str            # sha256 hex of the token — the token itself is never stored
    agent_ref: str
    owner: str
    speaker: str
    lane: str
    turn: str
    conn_id: str
    issued_at: float    # time.monotonic()
    expires_at: float   # issued_at + GRANT_TTL_S, may be cut by conn_dropped
    closed: bool = False


_GRANTS: dict[str, Grant] = {}


def _now(now: float | None) -> float:
    return time.monotonic() if now is None else float(now)


def _is_open(g: Grant, now: float) -> bool:
    return not g.closed and now < g.expires_at


def _purge(now: float) -> None:
    for key in [k for k, g in _GRANTS.items() if not _is_open(g, now)]:
        _GRANTS.pop(key, None)


def _close(g: Grant) -> None:
    g.closed = True
    if _GRANTS.get(g.key) is g:
        _GRANTS.pop(g.key, None)


def grant_key(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def valid_token_format(token: object) -> bool:
    return isinstance(token, str) and _TOKEN_RE.fullmatch(token) is not None


def mint(*, agent_ref: str, owner: str, speaker: str, lane: str, turn: str, conn_id: str,
         now: float | None = None) -> str:
    """A fresh grant token for one member turn, or "" when a cap is reached (or
    an identifier is missing). Never evicts: an evicted grant could belong to a
    running turn."""
    t = _now(now)
    _purge(t)
    if not (agent_ref and owner and speaker and turn):
        return ""
    per_member = per_agent = 0
    for g in _GRANTS.values():
        if g.agent_ref == agent_ref:
            per_agent += 1
            if g.speaker == speaker:
                per_member += 1
    cap = ""
    if per_member >= MAX_OPEN_GRANTS_PER_MEMBER:
        cap = "per-member"
    elif per_agent >= MAX_OPEN_GRANTS_PER_AGENT:
        cap = "per-agent"
    elif len(_GRANTS) >= MAX_OPEN_GRANTS:
        cap = "deck"
    if cap:
        log.warning("Shared-agent grant not minted: open-grant cap reached",
                    cap=cap, owner=owner, member=speaker)
        return ""
    token = secrets.token_urlsafe(32)
    key = grant_key(token)
    while key in _GRANTS:  # astronomically unlikely; never overwrite a live grant
        token = secrets.token_urlsafe(32)
        key = grant_key(token)
    _GRANTS[key] = Grant(key=key, agent_ref=agent_ref, owner=owner, speaker=speaker,
                         lane=lane, turn=turn, conn_id=conn_id, issued_at=t,
                         expires_at=t + GRANT_TTL_S)
    return token


def end_turn(agent_ref: str, speaker: str, lane: str, turn: str) -> bool:
    """Close every open grant of exactly this turn. True when one was closed."""
    hit = [g for g in _GRANTS.values() if not g.closed and g.agent_ref == agent_ref
           and g.speaker == speaker and g.lane == lane and g.turn == turn]
    for g in hit:
        _close(g)
    return bool(hit)


def conn_dropped(conn_id: str, *, now: float | None = None) -> int:
    """The member socket that minted these grants closed: they now expire within
    ``ORPHAN_GRACE_S`` (a turn already answering may still finish), even when
    sibling sockets of the same member and lane stay open."""
    if not conn_id:
        return 0
    t = _now(now)
    count = 0
    for g in _GRANTS.values():
        if not g.closed and g.conn_id == conn_id:
            g.expires_at = min(g.expires_at, t + ORPHAN_GRACE_S)
            count += 1
    return count


def revoke(agent_ref: str, speaker: str | None = None) -> int:
    """Close the open grants on ``agent_ref`` (one member's, or everyone's)."""
    hit = [g for g in _GRANTS.values() if not g.closed and g.agent_ref == agent_ref
           and (speaker is None or g.speaker == speaker)]
    for g in hit:
        _close(g)
    return len(hit)


def lookup(token: str, *, now: float | None = None) -> Grant | None:
    """The open, unexpired grant behind ``token``, else None."""
    if not valid_token_format(token):
        return None
    t = _now(now)
    _purge(t)
    g = _GRANTS.get(grant_key(token))
    return g if g is not None and _is_open(g, t) else None


def open_grants() -> list[Grant]:
    """Open grants (tests / diagnostics). Grants carry no token, only its hash."""
    t = time.monotonic()
    return [g for g in _GRANTS.values() if _is_open(g, t)]


def _reset_for_tests() -> None:
    _GRANTS.clear()


# ── Google opt-in (per member, per agent, per owner) ──────────────────────


def google_optin_key(agent_ref: str) -> str:
    return GOOGLE_OPTIN_PREFIX + agent_ref


async def google_opted_in(db, user_id: str, agent_ref: str, owner: str) -> bool:
    """Has ``user_id`` let ``agent_ref`` use their Google while ``owner`` owns it?"""
    if not (owner and user_id and agent_ref):
        return False
    try:
        value = await db.get_setting(user_id, google_optin_key(agent_ref))
    except Exception as exc:
        log.warning("Could not read a shared-agent Google opt-in", error=type(exc).__name__)
        return False
    return isinstance(value, str) and hmac.compare_digest(
        value.encode("utf-8"), owner.encode("utf-8"))


async def set_google_optin(db, user_id: str, agent_ref: str, owner: str, enabled: bool) -> None:
    key = google_optin_key(agent_ref)
    if enabled:
        await db.set_settings(user_id, {key: owner})
    else:
        await db.delete_setting(user_id, key)


async def clear_google_optins(db, agent_ref: str, user_id: str | None = None) -> int:
    """Drop the opt-in for ``agent_ref`` — one member's, or every member's."""
    key = google_optin_key(agent_ref)
    if user_id is not None:
        return 1 if await db.delete_setting(user_id, key) else 0
    return int(await db.delete_setting_for_all_users(key) or 0)


# ── Validation (contract part 0 §6) ───────────────────────────────────────


@dataclass(frozen=True)
class ActingMember:
    user_id: str
    owner: str
    agent_ref: str
    lane: str
    turn: str
    slug: str
    name: str


def _header_names(request) -> list[str]:
    headers = getattr(request, "headers", None)
    if headers is None:
        return []
    try:
        return [str(k).lower() for k in headers.keys()]
    except Exception:
        return []


def _grant_header_value(request) -> str:
    headers = getattr(request, "headers", None)
    if headers is None:
        return ""
    try:
        for k in headers.keys():
            if str(k).lower() == _HEADER_LOWER:
                value = headers.get(k)
                return value if isinstance(value, str) else ""
    except Exception:
        return ""
    return ""


def member_request(request: Request) -> bool:
    """Does this request carry either member marker (the grant header, or the
    ``fd_member`` query parameter)? Either one puts it on the member path."""
    if _HEADER_LOWER in _header_names(request):
        return True
    params = getattr(request, "query_params", None)
    if params is None:
        return False
    try:
        return MEMBER_MARKER_PARAM in params
    except Exception:
        return True  # unreadable query string: treat as marked (fail closed)


def _no_turn() -> HTTPException:
    return HTTPException(status_code=403, detail=NO_TURN_DETAIL)


def _route(request) -> str:
    try:
        return str(request.url.path)
    except Exception:
        return ""


async def acting_member(request: Request, *, google: bool = False) -> ActingMember | None:
    """The member a grant-aware request acts for; None only when the request
    carries neither marker (the route keeps its owner logic). With either
    marker this returns an ActingMember or raises 401/403 — never the owner."""
    if not member_request(request):
        return None
    value = _grant_header_value(request)

    from captain_claw.flight_deck import agent_sharing
    from captain_claw.flight_deck import google_oauth_routes as _google
    from captain_claw.flight_deck.auth import get_db

    # 2. agents only, through the usual agent transport gate
    if _google._is_browser_request(request):
        raise HTTPException(status_code=403, detail=_BROWSER_DETAIL)
    _google._authorize_agent_call(request)  # 401, same text as the owner routes

    # 3–4. a well-formed token of an open grant
    if not valid_token_format(value):
        raise _no_turn()
    g = lookup(value)
    if g is None:
        raise _no_turn()
    ref = g.agent_ref

    # 5. the agent is still the granting owner's process agent
    try:
        rec = await asyncio.to_thread(agent_sharing.resolve_agent_record, ref, strict=True)
    except agent_sharing.RecordUnavailable:
        # Can't tell right now: refuse this call, keep the grant and the opt-ins.
        raise _no_turn() from None
    if rec is None or rec.owner != g.owner or rec.runtime != "process":
        revoke(ref)
        if rec is None or rec.owner != g.owner:
            try:
                await clear_google_optins(get_db(), ref)
            except Exception as exc:
                log.warning("Could not clear a shared agent's Google opt-ins",
                            error=type(exc).__name__)
        raise _no_turn()

    # 6. the caller is that agent (a wrong caller must not end the member's turn)
    auth = request.headers.get("X-Agent-Auth", "") or ""
    if not auth or not rec.web_auth or not hmac.compare_digest(
            rec.web_auth.encode("utf-8"), auth.encode("utf-8")):
        raise _no_turn()

    # 7. the member still has access
    db = get_db()
    if not await agent_sharing.member_check(db, ref, g.owner, g.speaker):
        revoke(ref, g.speaker)
        agent_sharing.invalidate_member_cache(ref, g.speaker)
        raise HTTPException(status_code=403, detail=NOT_MEMBER_DETAIL)

    # 8. Google needs the member's opt-in for this agent under this owner
    if google and not await google_opted_in(db, g.speaker, ref, g.owner):
        raise HTTPException(status_code=403, detail=GOOGLE_OFF_DETAIL)

    # The grant may have closed while this request awaited (turn_end, a
    # revocation): it acts for nobody then.
    if lookup(value) is not g:
        raise _no_turn()

    log.info("Shared-agent member request", route=_route(request), agent=rec.slug,
             owner=g.owner, member=g.speaker)
    return ActingMember(user_id=g.speaker, owner=g.owner, agent_ref=ref, lane=g.lane,
                        turn=g.turn, slug=rec.slug, name=rec.name)


# ── Middleware ────────────────────────────────────────────────────────────


def _scope_has_marker(scope: Scope) -> bool:
    for item in scope.get("headers") or ():
        try:
            name = item[0]
        except Exception:
            continue
        if isinstance(name, (bytes, bytearray)) and bytes(name).lower() == _HEADER_BYTES:
            return True
    raw = scope.get("query_string") or b""
    try:
        text = raw.decode("latin-1") if isinstance(raw, (bytes, bytearray)) else str(raw)
        return MEMBER_MARKER_PARAM in parse_qs(text, keep_blank_values=True)
    except Exception:
        return True  # unparseable query string: treat as marked (fail closed)


def _scope_path(scope: Scope) -> str:
    path = str(scope.get("path") or "")
    root = str(scope.get("root_path") or "")
    if root and path.startswith(root):
        path = path[len(root):]
    return path.rstrip("/") or "/"


class GrantGuardMiddleware:
    """Refuse the member markers everywhere but the grant-aware routes (HTTP
    403, WebSocket close 4403 before accept). Pure ASGI; never reads a body."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope.get("type")
        if kind not in ("http", "websocket") or not _scope_has_marker(scope):
            await self.app(scope, receive, send)
            return
        path = _scope_path(scope)
        if kind == "websocket":
            log.warning("Refused a shared-agent member marker on a WebSocket", path=path)
            await send({"type": "websocket.close", "code": 4403})
            return
        if path in GRANT_AWARE_PATHS:
            await self.app(scope, receive, send)
            return
        log.warning("Refused a shared-agent member request off the grant-aware routes",
                    method=str(scope.get("method") or ""), path=path)
        body = json.dumps({"detail": OFF_PATH_DETAIL}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})
