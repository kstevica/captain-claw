"""HTTP side of WhatsApp Web (service: :mod:`.whatsapp_web`, Flight Deck's
end of it: :mod:`.whatsapp_web_supervisor`).

* ``/fd/whatsapp-web`` — the user's own link (Connections → WhatsApp Web):
  link with consent (a QR follows), a pairing code instead of the QR,
  reconnect, cancel, disconnect (the user says whether the history stays).
  Also the retention choice, a peek at what's kept, and deleting what's kept
  from earlier links.
* ``/fd/whatsapp-web/agent/*`` — what the user's agents read: owner only,
  identified like the Google / Telegram routes (``X-Agent-Auth``); a shared
  agent's member never gets here (GrantGuardMiddleware refuses member markers).
* ``/fd/admin/whatsapp-web`` — the deck's switch, off by default.

Agents send only through ``/fd/whatsapp-web/agent/send``: automatic once the
user turned sending on, and only to chats on their whitelist (the gate is the
service's, :meth:`Engine.send`); ``FD_WHATSAPP_WEB_SEND=off`` refuses every
send. Auto-replies (:mod:`.whatsapp_web_autoreply`) go through the same gate;
the user picks their chats, the agent and its trigger words with the sending
rules. Nothing here reacts, marks read or changes presence.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from captain_claw.flight_deck import fd_crypto, whatsapp_web_autoreply as autoreply
from captain_claw.flight_deck import gmail_autoreply_writer as writer
from captain_claw.flight_deck import whatsapp_web_supervisor as sup
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck.whatsapp_web.link_state import now_iso
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["whatsapp-web"])


async def _unavailable_reason(db: Any) -> str:
    """Why this deck can't link WhatsApp Web right now ("" when it can)."""
    from captain_claw.flight_deck import google_oauth_routes as _g

    if sup.kill_switch():
        return "WhatsApp Web is switched off on this deck (FD_WHATSAPP_WEB=off)."
    if _g._fd_desktop_deck():
        return "WhatsApp Web isn't available in the desktop app yet."
    if not _g._fd_auth_enabled():
        # Agents read through their owner; a deck that can't tell its users
        # apart has no owner to read for.
        return "WhatsApp Web needs Flight Deck sign-in, which is off on this deck (FD_AUTH_ENABLED)."
    if not fd_crypto.available():
        return ("This Flight Deck can't encrypt kept messages: it needs the 'cryptography' "
                "package (pip install cryptography) and a restart.")
    if not sup.installed():
        return ("WhatsApp Web isn't installed on this deck: pip install "
                "'captain-claw[whatsapp-web]' and restart Flight Deck.")
    if not (await load_cfg(db))["allowed"]:
        return "An admin hasn't allowed WhatsApp Web on this deck."
    if sup.get() is None:
        return "WhatsApp Web hasn't started on this deck: restart Flight Deck."
    return ""


async def load_cfg(db: Any) -> dict[str, Any]:
    return await sup.load_config(db)


def _service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, sup.ServiceError):
        return HTTPException(status_code=exc.status, detail=exc.detail)
    return HTTPException(status_code=503, detail=str(exc) or "The WhatsApp Web service isn't running")


async def _view(user: dict, *, link: dict[str, Any] | None = None) -> dict[str, Any]:
    db = get_db()
    uid = user["id"]
    reason = await _unavailable_reason(db)
    s = sup.get()
    out: dict[str, Any] = {
        "available": not reason,
        "reason": reason,
        "allowed": (await load_cfg(db))["allowed"],
        "can_admin": user.get("role") == "admin",
        "link": {"state": "off", "running": False},
        # History from earlier links: a banned number's, one the user switched from.
        "kept": s.kept(uid) if s is not None else [],
    }
    if link is not None:
        out["link"] = link
    elif s is not None and s.user_has_link(uid):
        if reason:
            out["link"] = s.offline_view(uid)
        else:
            try:
                out["link"] = await s.request("GET", f"/links/{uid}")
            except (sup.ServiceDown, sup.ServiceError) as exc:
                out["link"] = {**s.offline_view(uid), "detail": str(exc)}
    return out


def _supervisor() -> sup.Supervisor:
    s = sup.get()
    if s is None:
        raise HTTPException(status_code=503, detail="WhatsApp Web hasn't started on this deck: restart Flight Deck.")
    return s


async def _call(user: dict, method: str, suffix: str = "", body: dict[str, Any] | None = None) -> dict[str, Any]:
    reason = await _unavailable_reason(get_db())
    if reason:
        raise HTTPException(status_code=409, detail=reason)
    try:
        link = await _supervisor().request(method, f"/links/{user['id']}{suffix}", body)
    except (sup.ServiceDown, sup.ServiceError) as exc:
        raise _service_error(exc) from None
    return await _view(user, link=link)


# ── the user's own link ─────────────────────────────────────────────


@router.get("/fd/whatsapp-web")
async def my_link(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await _view(user)


class LinkBody(BaseModel):
    # Every new link needs it: the user read that this is an unofficial client
    # (against WhatsApp's terms, bans happen) and that agents will read it.
    consent: bool = False


@router.post("/fd/whatsapp-web/link")
async def start_link(body: LinkBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    if not body.consent:
        raise HTTPException(status_code=422, detail="Linking WhatsApp needs your consent")
    return await _call(user, "POST", "/start", {"consented_at": now_iso()})


class PairCodeBody(BaseModel):
    phone: str


@router.post("/fd/whatsapp-web/pair-code")
async def pair_code(body: PairCodeBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await _call(user, "POST", "/pair-code", {"phone": body.phone})


@router.post("/fd/whatsapp-web/reconnect")
async def reconnect(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await _call(user, "POST", "/reconnect")


@router.post("/fd/whatsapp-web/cancel")
async def cancel(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await _call(user, "POST", "/cancel")


@router.delete("/fd/whatsapp-web")
async def unlink(keep_history: bool = True, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Disconnect: log the device out (it leaves the phone's Linked devices) and
    forget the link. The messages stay readable for the user's agents, unless
    the user says ``keep_history=false``. With the service off, this happens to
    the files and the phone still lists the device until it's removed there."""
    s = _supervisor()
    uid = user["id"]
    if not s.user_has_link(uid):
        out = await _view(user)
        out["removed_from_phone"] = False
        return out
    reason = await _unavailable_reason(get_db())
    if reason:
        s.forget_local(uid, keep_history=keep_history)
        out = await _view(user)
        out["removed_from_phone"] = False
        return out
    try:
        link = await s.request("POST", f"/links/{uid}/disconnect", {"keep_history": keep_history})
    except sup.ServiceError as exc:
        if exc.status in (404, 405):
            # Flight Deck adopts a running service, and an older one can't keep history.
            raise HTTPException(status_code=409, detail=(
                "The WhatsApp Web service is older than this Flight Deck, so nothing was changed. "
                "An admin can restart it by turning Allow WhatsApp Web off and on.")) from None
        raise _service_error(exc) from None
    except sup.ServiceDown as exc:
        log.warning("WhatsApp Web unlink for %s via the service failed (%s); doing it locally", uid, exc)
        s.forget_local(uid, keep_history=keep_history)
        link = {"state": "off", "running": False, "removed_from_phone": False}
    out = await _view(user, link={k: v for k, v in link.items() if k != "removed_from_phone"})
    out["removed_from_phone"] = bool(link.get("removed_from_phone"))
    return out


@router.delete("/fd/whatsapp-web/kept/{entry}")
async def delete_kept(entry: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Delete what's kept from one earlier link: only ever the user's own call."""
    s = _supervisor()
    uid = user["id"]
    if not any(k["id"] == entry for k in s.kept(uid)):
        raise HTTPException(status_code=404, detail="Nothing like that is kept")
    if await _unavailable_reason(get_db()):
        s.forget_kept_local(uid, entry)
        return await _view(user)
    try:
        await s.request("DELETE", f"/links/{uid}/kept/{entry}")
    except sup.ServiceDown:
        s.forget_kept_local(uid, entry)
    except sup.ServiceError as exc:
        raise _service_error(exc) from None
    return await _view(user)


# ── the deck's switch ───────────────────────────────────────────────


async def _admin_view() -> dict[str, Any]:
    db = get_db()
    s = sup.get()
    service = None
    if s is not None and s.running():
        try:
            service = await s.request("GET", "/health")
        except (sup.ServiceDown, sup.ServiceError):
            service = None
    return {
        "allowed": (await load_cfg(db))["allowed"],
        "installed": sup.installed(),
        "kill_switch": sup.kill_switch(),
        "service": service,
    }


@router.get("/fd/admin/whatsapp-web")
async def admin_get(_admin: dict = Depends(require_admin)) -> dict[str, Any]:
    return await _admin_view()


class AdminBody(BaseModel):
    allowed: bool


@router.put("/fd/admin/whatsapp-web")
async def admin_put(body: AdminBody, _admin: dict = Depends(require_admin)) -> dict[str, Any]:
    db = get_db()
    await sup.save_config(db, allowed=body.allowed)
    s = sup.get()
    if s is not None:
        if not body.allowed:
            await s.stop()
        elif s.any_links() and not sup.kill_switch() and sup.installed():
            try:
                await s.ensure_running()
            except sup.ServiceDown as exc:
                log.warning("WhatsApp Web service not started: %s", exc)
    return await _admin_view()


# ── the user's peek and retention ───────────────────────────────────


@router.get("/fd/whatsapp-web/recent")
async def recent(limit: int = 20, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """The newest kept messages — what the user's agents see."""
    s = sup.get()
    if s is None or not s.user_has_link(user["id"]) or await _unavailable_reason(get_db()):
        return {"messages": []}
    try:
        data = await s.request("POST", f"/links/{user['id']}/messages",
                               {"limit": max(1, min(int(limit or 20), 100)), "kept": True})
    except (sup.ServiceDown, sup.ServiceError) as exc:
        raise _service_error(exc) from None
    return {"messages": data.get("messages") or []}


class RetentionBody(BaseModel):
    # None = keep everything.
    days: int | None = None


@router.put("/fd/whatsapp-web/retention")
async def set_retention(body: RetentionBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await _call(user, "PUT", "/retention", {"days": body.days})


# ── the user's sending rules ────────────────────────────────────────


@router.get("/fd/whatsapp-web/sending")
async def get_sending(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Sending on/off, the whitelist, the daily limit, the signature, the
    latest sends — and the chats the whitelist can be picked from."""
    s = sup.get()
    if s is None or not s.user_has_link(user["id"]) or await _unavailable_reason(get_db()):
        return {"available": False}
    try:
        policy = await s.request("GET", f"/links/{user['id']}/send-policy")
        chats = (await s.request("GET", f"/links/{user['id']}/chats")).get("chats") or []
        auto = await s.request("GET", f"/links/{user['id']}/autoreply")
    except (sup.ServiceDown, sup.ServiceError) as exc:
        raise _service_error(exc) from None
    agents = await writer.list_eligible_agents(get_db(), user["id"])
    return {"available": True, "kill_switch": sup.send_kill_switch(), **policy,
            "chats": [{k: c.get(k) for k in ("jid", "title", "kind", "number", "messages", "last_ts")}
                      for c in chats],
            "autoreply": {
                # A chat taken off the whitelist keeps no auto-reply.
                "chats": [j for j in auto.get("chats") or [] if j in (policy.get("allowed") or [])],
                "agent": auto.get("agent") or "",
                "triggers": auto.get("triggers") or [],
                "kill_switch": autoreply.kill_switch(),
                "agents": [{k: a.get(k) for k in ("ref", "name", "running", "eligible", "reason")}
                           for a in agents],
            }}


class AutoreplyBody(BaseModel):
    chats: list[str] | None = None
    agent: str | None = None
    triggers: list[str] | None = None


class SendingBody(BaseModel):
    enabled: bool | None = None
    allowed: list[str] | None = None
    daily_limit: int | None = None
    signature: str | None = None
    signature_on: bool | None = None
    autoreply: AutoreplyBody | None = None


@router.put("/fd/whatsapp-web/sending")
async def set_sending(body: SendingBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    reason = await _unavailable_reason(get_db())
    if reason:
        raise HTTPException(status_code=409, detail=reason)
    s = _supervisor()
    if not s.user_has_link(user["id"]):
        raise HTTPException(status_code=409, detail="Link WhatsApp first")
    auto = body.autoreply.model_dump() if body.autoreply is not None else None
    if auto and auto.get("agent"):
        agents = await writer.list_eligible_agents(get_db(), user["id"])
        if not any(a["ref"] == auto["agent"] for a in agents):
            raise HTTPException(status_code=422, detail="Pick one of your own agents to answer")
    try:
        await s.request("PUT", f"/links/{user['id']}/send-policy", body.model_dump(exclude={"autoreply"}))
        if auto is not None:
            await s.request("PUT", f"/links/{user['id']}/autoreply", auto)
    except (sup.ServiceDown, sup.ServiceError) as exc:
        raise _service_error(exc) from None
    return await get_sending(user)


@router.get("/fd/whatsapp-web/autoreply-activity")
async def autoreply_activity(limit: int = 50, ignored: bool = False,
                             user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """The user's latest auto-reply decisions, newest first: the messages,
    what the model thought and said, what was sent. Without *ignored*, not
    the messages that didn't call the agent."""
    s = sup.get()
    if s is None or not s.user_has_link(user["id"]) or await _unavailable_reason(get_db()):
        return {"available": False, "entries": []}
    try:
        data = await s.request("GET", f"/links/{user['id']}/autoreply-log?limit={max(1, min(limit, 200))}"
                                      f"&ignored={'true' if ignored else 'false'}")
    except (sup.ServiceDown, sup.ServiceError) as exc:
        if isinstance(exc, sup.ServiceError) and exc.status == 404:
            # A service started before this Flight Deck was updated.
            return {"available": False, "entries": [], "stale_service": True}
        raise _service_error(exc) from None
    return {"available": True, "entries": data.get("entries") or []}


# ── agents (owner only) ─────────────────────────────────────────────

_NOT_LINKED = ("Your user hasn't linked WhatsApp — they can in Flight Deck → Connections → "
               "WhatsApp Web")
# Link states whose kept messages an agent may read.
_READABLE = ("connected", "disconnected", "connecting", "stream_replaced", "temporary_ban", "outdated")


async def _agent_owner(request: Request) -> str:
    from captain_claw.flight_deck import google_oauth_routes as _g

    _g._authorize_agent_call(request)
    return await _g._agent_owner(request, service="WhatsApp Web")


def _agent_slug(request: Request) -> str:
    """The calling agent's slug on this deck (its news cursor's key)."""
    try:
        from captain_claw.flight_deck.server import _find_agent_by_auth

        matched, _owner, slug = _find_agent_by_auth(request.headers.get("X-Agent-Auth", ""))
    except Exception:
        return ""
    return str(slug) if matched and slug else ""


def _readable(s: sup.Supervisor, owner: str) -> bool:
    link = s.offline_view(owner) if s.user_has_link(owner) else {"state": "off"}
    # With no link at all, agents still read what's kept from an earlier one.
    return (link.get("state") in _READABLE and bool(link.get("linked_at"))) or s.has_kept(owner)


async def _agent_ready(request: Request) -> tuple[str, sup.Supervisor]:
    """(owner, supervisor) for an agent call, or the HTTP refusal."""
    reason = await _unavailable_reason(get_db())
    if reason:
        raise HTTPException(status_code=409, detail=reason)
    owner = await _agent_owner(request)
    s = _supervisor()
    if not _readable(s, owner):
        raise HTTPException(status_code=409, detail=_NOT_LINKED)
    return owner, s


async def _agent_call(request: Request, path: str, body: dict[str, Any] | None = None) -> tuple[str, dict]:
    owner, s = await _agent_ready(request)
    try:
        return owner, await s.request("GET" if body is None else "POST", f"/links/{owner}{path}", body)
    except (sup.ServiceDown, sup.ServiceError) as exc:
        raise _service_error(exc) from None


@router.get("/fd/whatsapp-web/agent/status")
async def agent_status(request: Request) -> dict[str, Any]:
    """Whether the calling agent's owner has WhatsApp linked (no network)."""
    s = sup.get()
    if s is None or await _unavailable_reason(get_db()):
        return {"connected": False}  # before asking whose agent: off decks have no owners
    return {"connected": _readable(s, await _agent_owner(request))}


@router.post("/fd/whatsapp-web/agent/chats")
async def agent_chats(request: Request) -> dict[str, Any]:
    _owner, data = await _agent_call(request, "/chats?kept=true")
    return data


class AgentMessagesBody(BaseModel):
    chat: str = ""
    query: str = ""
    since: int | None = None
    until: int | None = None
    limit: int = 50


@router.post("/fd/whatsapp-web/agent/messages")
async def agent_messages(body: AgentMessagesBody, request: Request) -> dict[str, Any]:
    """Messages, newest first — of one chat (id, number or name) or all of them;
    ``query`` keeps those containing every word. A look at the newest ones
    (no ``until``, no ``query``) counts as seen for this agent's news line."""
    _owner, data = await _agent_call(request, "/messages", {
        **body.model_dump(), "agent": _agent_slug(request), "mark_seen": True, "kept": True,
    })
    return data


class AgentContactsBody(BaseModel):
    query: str = ""
    limit: int = 20


@router.post("/fd/whatsapp-web/agent/contacts")
async def agent_contacts(body: AgentContactsBody, request: Request) -> dict[str, Any]:
    _owner, data = await _agent_call(request, "/contacts", body.model_dump())
    return data


class AgentAttachmentBody(BaseModel):
    chat: str
    id: str


@router.post("/fd/whatsapp-web/agent/attachment")
async def agent_attachment(body: AgentAttachmentBody, request: Request) -> Response:
    """One message's file, downloaded now from WhatsApp's servers: the bytes,
    with its details (JSON) in ``X-WAW-Meta``."""
    owner, s = await _agent_ready(request)
    try:
        content, headers = await s.request_raw("POST", f"/links/{owner}/attachment", body.model_dump())
    except (sup.ServiceDown, sup.ServiceError) as exc:
        raise _service_error(exc) from None
    return Response(content=content, media_type=headers.get("content-type") or "application/octet-stream",
                    headers={"X-WAW-Meta": headers.get("x-waw-meta") or "{}"})


class AgentSendBody(BaseModel):
    chat: str
    text: str
    allow_repeat: bool = False


@router.post("/fd/whatsapp-web/agent/send")
async def agent_send(body: AgentSendBody, request: Request) -> dict[str, Any]:
    """One text message, sent now — automatic once the owner turned sending on,
    only to a chat on their whitelist; every refusal says why."""
    if sup.send_kill_switch():
        raise HTTPException(status_code=403, detail=(
            "Sending WhatsApp messages is switched off on this deck (FD_WHATSAPP_WEB_SEND=off)"))
    owner, s = await _agent_ready(request)
    try:
        return await s.request("POST", f"/links/{owner}/send", {
            **body.model_dump(), "agent": _agent_slug(request)}, timeout=60.0)
    except (sup.ServiceDown, sup.ServiceError) as exc:
        raise _service_error(exc) from None


@router.post("/fd/whatsapp-web/agent/news")
async def agent_news(request: Request) -> dict[str, Any]:
    """What others wrote since this agent last looked — counts per chat, no
    text — and the look itself moves the agent's cursor (an owner's own turn
    calls it once)."""
    s = sup.get()
    if s is None or await _unavailable_reason(get_db()):
        return {"connected": False}
    owner = await _agent_owner(request)
    if not _readable(s, owner):
        return {"connected": False}
    try:
        return await s.request("POST", f"/links/{owner}/news", {"agent": _agent_slug(request) or "_"})
    except (sup.ServiceDown, sup.ServiceError):
        return {"connected": False}
