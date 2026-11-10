"""``/fd/external-memory/*`` — the user's external memories, for the dashboard
and for their agents.

User routes (Connections → External memory)::

    GET    /providers         the providers this deck knows (and what each needs to connect)
    GET    /status            this user's memories — never a credential
    PUT    /connection        {provider, credentials: {...}, memory?, name?} — a new memory, or new
                              credentials for one (memory = its id); checked with the provider, then saved
    DELETE /connection        ?memory=<id> — remove one (and whatever was still queued for it)
    PUT    /memories/{id}     {name?, priority?, mirror?, enabled?} — how this person uses one memory
    POST   /test              {memory?} — a live check; refreshes what it may do
    POST   /run               {action, args, memory?} — the playground (search, remember, ...)
    GET    /outbox            ?memory= — recent mirrored items and their state
    POST   /outbox/retry      {memory?} — failed and held items go again now

Admin routes (the deck's connection — a scoped provider such as Hindsight, Qdrant or ArcadeDB, one
scope per person; one more memory in each person's list, which they can switch off)::

    GET    /deck              the deck's connection — never a credential
    PUT    /deck-connection   {provider, credentials} — checked, then saved
    DELETE /deck-connection   stop sharing it (users' own connections are untouched)
    POST   /deck-test         a live check

Agent routes — owner resolved from this deck's own records (``X-Agent-Auth``),
the same gate as Jev; owner-only (not grant-aware, so the speaker-grant
middleware refuses a shared-agent member's request)::

    GET  /agent/status        {connected, memories, actions, rules, mirror, priority, recall_limit, ...}
    POST /agent/run           {action, args, memory?} (args.memory works too)
    POST /agent/recall        {query} → {groups: [{name, priority, hits}], hits} — before a reply; never fails
    POST /agent/mirror        {kind, source_id, content, title, tags, created_at, meta}

The frontend reads a 401 as "the Flight Deck session expired", so a user route
never answers 401 for a provider-side problem — refused credentials are a 400.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from captain_claw.flight_deck import google_oauth_routes as _g
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.agent_identity import require_agent_caller
from captain_claw.flight_deck.auth import _LOCAL_USER, _fd_auth_enabled, get_current_user
from captain_claw.flight_deck.external_memory import providers
from captain_claw.flight_deck.external_memory import service as svc
from captain_claw.flight_deck.external_memory.base import (
    MemoryItem,
    ProviderAuthError,
    ProviderError,
    ProviderInputError,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/external-memory", tags=["external-memory"])


def http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, svc.NotConnected):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ProviderInputError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, ProviderAuthError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ProviderError):
        if exc.status == 429:
            return HTTPException(status_code=429, detail=str(exc))
        if exc.status in (400, 403, 404, 422, 428):
            # The provider's answer to THIS call — the agent reads and relays it.
            return HTTPException(status_code=409 if exc.status == 428 else 400, detail=str(exc))
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=502, detail=f"External memory request failed: {type(exc).__name__}")


async def _json_body(request: Request) -> dict[str, Any]:
    """The JSON object body, read by hand: a malformed body must never reach
    FD's validation handler, which logs (and echoes) the raw input — a key."""
    try:
        data = await request.json()
    except Exception:
        data = None
    return data if isinstance(data, dict) else {}


# ── user routes ─────────────────────────────────────────────────────


@router.get("/providers")
async def em_providers(_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return {"providers": providers.catalog()}


@router.get("/status")
async def em_status(_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await svc.status_for_owner(_g._effective_owner(_user))


@router.put("/connection")
async def em_connect(request: Request, _user: dict = Depends(get_current_user)) -> dict[str, Any]:
    data = await _json_body(request)
    provider = str(data.get("provider") or "").strip().lower()
    creds = data.get("credentials")
    # Only an admin (or the one person of a deck with sign-in off) may point
    # Flight Deck at this machine or its network.
    private_ok = not _fd_auth_enabled() or str((_user or {}).get("role") or "") == "admin"
    try:
        out, warning = await svc.connect(
            _g._effective_owner(_user), provider, creds if isinstance(creds, dict) else {},
            private_ok=private_ok, memory_id=str(data.get("memory") or "").strip(),
            name=str(data.get("name") or "").strip(),
        )
    except ProviderAuthError as exc:
        raise HTTPException(status_code=400, detail=f"{exc} Nothing was saved.") from None
    except ProviderInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if warning:
        out["warning"] = warning
    return out


@router.delete("/connection")
async def em_disconnect(memory: str = "", _user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _g._effective_owner(_user)
    if not memory:
        own = [m for m in await svc.memories(owner, enabled_only=False) if m.source == "own"]
        if len(own) != 1:
            raise HTTPException(status_code=422, detail="Say which memory to remove (?memory=<id>).")
        memory = own[0].id
    try:
        return await svc.disconnect(owner, memory)
    except (svc.NotConnected, ProviderInputError) as exc:
        raise http_error(exc) from None


@router.put("/memories/{memory_id}")
async def em_memory_settings(memory_id: str, request: Request,
                             _user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """How this person uses one memory: its name, reading priority (first |
    alongside | on_demand), what is copied in on its own, on or off."""
    data = await _json_body(request)
    if isinstance(data.get("kinds"), dict) and "mirror" not in data:
        data["mirror"] = data["kinds"]
    try:
        return await svc.update_memory(_g._effective_owner(_user), memory_id, data)
    except (svc.NotConnected, ProviderInputError) as exc:
        raise http_error(exc) from None


@router.post("/test")
async def em_test(request: Request, _user: dict = Depends(get_current_user)) -> dict[str, Any]:
    data = await _json_body(request)
    try:
        return await svc.test(_g._effective_owner(_user), str(data.get("memory") or "").strip())
    except (svc.NotConnected, ProviderInputError, ProviderError) as exc:
        raise http_error(exc) from None


@router.post("/run")
async def em_run(request: Request, _user: dict = Depends(get_current_user)) -> dict[str, Any]:
    data = await _json_body(request)
    args = data.get("args")
    try:
        result = await svc.run_for_owner(
            _g._effective_owner(_user), str(data.get("action") or ""), args if isinstance(args, dict) else {},
            source="ui", memory=str(data.get("memory") or ""),
        )
    except (svc.NotConnected, ProviderInputError, ProviderError) as exc:
        raise http_error(exc) from None
    return {"text": result.text, "data": result.data}


@router.get("/outbox")
async def em_outbox(state: str = "", limit: int = 30, memory: str = "",
                    _user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = svc.owner_key(_g._effective_owner(_user))
    state = state if state in ("pending", "sent", "failed") else ""
    box = svc.outbox()
    target = memory or None
    return {
        "items": await box.recent(owner, max(1, min(int(limit or 30), 200)), state, target=target),
        "counts": await box.counts(owner, target=target),
    }


@router.post("/outbox/retry")
async def em_outbox_retry(request: Request, _user: dict = Depends(get_current_user)) -> dict[str, Any]:
    data = await _json_body(request)
    n = await svc.outbox().retry_owner(svc.owner_key(_g._effective_owner(_user)),
                                       target=str(data.get("memory") or "") or None)
    svc.wake()
    return {"requeued": n}


# ── admin routes (the deck's connection) ────────────────────────────


@router.get("/deck")
async def em_deck_status(_admin: dict = Depends(require_admin)) -> dict[str, Any]:
    return await svc.deck_status()


@router.put("/deck-connection")
async def em_deck_connect(request: Request, _admin: dict = Depends(require_admin)) -> dict[str, Any]:
    data = await _json_body(request)
    creds = data.get("credentials")
    try:
        out, warning = await svc.deck_connect(
            str(data.get("provider") or "").strip().lower(), creds if isinstance(creds, dict) else {},
        )
    except ProviderAuthError as exc:
        raise HTTPException(status_code=400, detail=f"{exc} Nothing was saved.") from None
    except ProviderInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if warning:
        out["warning"] = warning
    return out


@router.delete("/deck-connection")
async def em_deck_disconnect(_admin: dict = Depends(require_admin)) -> dict[str, Any]:
    return await svc.deck_disconnect()


@router.post("/deck-test")
async def em_deck_test(_admin: dict = Depends(require_admin)) -> dict[str, Any]:
    try:
        return await svc.deck_test()
    except (svc.NotConnected, ProviderInputError, ProviderError) as exc:
        raise http_error(exc) from None


# ── agent routes ────────────────────────────────────────────────────


def _agent_identity(request: Request) -> tuple[str, str]:
    """``(slug, name)`` of the calling agent, from this deck's records (for
    attribution and the usage log); ``("", "")`` when unknown."""
    try:
        from captain_claw.flight_deck import server as srv

        matched, _owner, slug = srv._find_agent_by_auth(request.headers.get("X-Agent-Auth", ""))
        if not (matched and slug):
            return "", ""
        name = str((srv._load_process_registry().get(slug) or {}).get("name") or slug)
        return str(slug), name
    except Exception:
        return "", ""


async def _agent_caller(request: Request) -> tuple[str, str, str]:
    """``(owner, slug, name)`` of the calling agent. Auth on: the Google /
    Microsoft gate (transport, then ``X-Agent-Auth`` → owner). Auth off: the
    agent gate alone, and the deck's connection."""
    if _fd_auth_enabled():
        _g._authorize_agent_call(request)
        owner = await _g._agent_owner(request, service="External memory")
        slug, name = _agent_identity(request)
        return owner, slug, name
    caller = require_agent_caller(request, what="External memory")
    slug, name = _agent_identity(request)
    return _LOCAL_USER["id"], slug or caller.slug, name or caller.slug


@router.get("/agent/status")
async def em_agent_status(request: Request) -> dict[str, Any]:
    owner, _slug, _name = await _agent_caller(request)
    return await svc.manifest_for_owner(owner)


@router.post("/agent/run")
async def em_agent_run(request: Request) -> dict[str, Any]:
    owner, slug, _name = await _agent_caller(request)
    data = await _json_body(request)
    args = data.get("args")
    try:
        result = await svc.run_for_owner(
            owner, str(data.get("action") or ""), args if isinstance(args, dict) else {},
            source="agent", agent=slug, memory=str(data.get("memory") or ""),
        )
    except (svc.NotConnected, ProviderInputError, ProviderError) as exc:
        raise http_error(exc) from None
    return {"text": result.text, "data": result.data}


@router.post("/agent/recall")
async def em_agent_recall(request: Request) -> dict[str, Any]:
    """What the owner's memory holds on the message the agent is about to
    answer (automatic recall; empty when their reading priority is on_demand)."""
    owner, slug, _name = await _agent_caller(request)
    data = await _json_body(request)
    return await svc.recall_for_owner(owner, str(data.get("query") or ""), agent=slug)


@router.post("/agent/mirror")
async def em_agent_mirror(request: Request) -> dict[str, Any]:
    owner, _slug, name = await _agent_caller(request)
    data = await _json_body(request)
    if str(data.get("kind") or "") == "lesson":
        # Lessons are taught in Flight Deck and mirrored from there.
        raise HTTPException(status_code=422, detail="Lessons are mirrored by Flight Deck itself")
    item = MemoryItem.from_dict(data)
    item.agent = name or item.agent  # the deck's record, not the agent's claim
    try:
        return await svc.mirror_for_owner(owner, item)
    except ProviderInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
