"""The service's loopback HTTP API — Flight Deck is its only caller.

Bound to 127.0.0.1 on a port the OS picks (``service.json`` says which), and
every route needs ``Authorization: Bearer <service.secret>``. The user id in
the path is Flight Deck's to vouch for: it has already authenticated the
person and only ever passes their own id.

* ``GET    /health``
* ``GET    /links/{user_id}``
* ``POST   /links/{user_id}/start``      ``{"consented_at": "…"}`` → pairing (QR)
* ``POST   /links/{user_id}/pair-code``  ``{"phone": "…"}`` → an 8-character code
* ``POST   /links/{user_id}/reconnect``  start again on the kept session
* ``POST   /links/{user_id}/cancel``     forget a link that never paired
* ``POST   /links/{user_id}/disconnect`` ``{keep_history}`` log out, stop, forget the link
* ``DELETE /links/{user_id}/kept/{entry}`` delete what's kept from one earlier link
* ``GET    /links/{user_id}/chats``      the linked account's chats, latest first (``?kept=true``
  with what's kept from earlier links)
* ``POST   /links/{user_id}/messages``   ``{chat, query, since, until, limit, agent, mark_seen, kept}``
* ``POST   /links/{user_id}/contacts``   ``{query, limit}``
* ``POST   /links/{user_id}/news``       ``{agent}`` → what's new for that agent (moves its cursor)
* ``PUT    /links/{user_id}/retention``  ``{days}`` (null = keep everything)
* ``POST   /links/{user_id}/attachment`` ``{chat, id}`` → the file's bytes, details in ``X-WAW-Meta``
* ``GET/PUT /links/{user_id}/send-policy`` sending on/off, whitelist, daily limit, signature
* ``POST   /links/{user_id}/send``       ``{chat, text, agent, allow_repeat}`` → through the gate
* ``GET/PUT /links/{user_id}/autoreply`` the chats the agent answers by itself, the agent, trigger words
* ``GET/POST /links/{user_id}/autoreply-log`` Flight Deck's auto-reply decisions (sealed), for the card
* ``GET    /events?after=<seq>``         bells for Flight Deck to ring, and (``kind: "incoming"``)
  messages in auto-reply chats for Flight Deck to answer
"""

from __future__ import annotations

import hmac
import json
import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from captain_claw.flight_deck.whatsapp_web.engine import Engine, LinkError


class StartBody(BaseModel):
    consented_at: str


class PairCodeBody(BaseModel):
    phone: str


class MessagesBody(BaseModel):
    chat: str = ""
    query: str = ""
    since: int | None = None
    until: int | None = None
    limit: int = 50
    # The agent reading (its news cursor), and whether this look counts as seen.
    agent: str = ""
    mark_seen: bool = False
    # Also what's kept from earlier links (agents reading, not the auto-replies).
    kept: bool = False


class DisconnectBody(BaseModel):
    keep_history: bool = True


class ContactsBody(BaseModel):
    query: str = ""
    limit: int = 20


class NewsBody(BaseModel):
    agent: str


class AttachmentBody(BaseModel):
    chat: str
    id: str


class SendPolicyBody(BaseModel):
    enabled: bool | None = None
    allowed: list[str] | None = None
    daily_limit: int | None = None
    signature: str | None = None
    signature_on: bool | None = None


class SendBody(BaseModel):
    chat: str
    text: str
    agent: str = ""
    allow_repeat: bool = False


class AutoreplyBody(BaseModel):
    chats: list[str] | None = None
    agent: str | None = None
    triggers: list[str] | None = None


class AutoreplyLogBody(BaseModel):
    chat: str
    outcome: str
    details: dict[str, Any] = {}


class RetentionBody(BaseModel):
    # None = keep everything.
    days: int | None = None


def build_app(engine: Engine, secret: str, *, version: str = "", lifespan: Any = None) -> FastAPI:
    app = FastAPI(title="WhatsApp Web service", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)

    def authorized(request: Request) -> None:
        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not secret or not hmac.compare_digest(given.encode(), secret.encode()):
            raise HTTPException(status_code=401, detail="unauthorized")

    def answer(fn):
        async def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
            try:
                return await fn(*args, **kwargs)
            except LinkError as exc:
                raise HTTPException(status_code=exc.status, detail=exc.detail) from None
        return run

    auth = [Depends(authorized)]

    @app.get("/health", dependencies=auth)
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "pid": os.getpid(),
            "available": not engine.unavailable,
            "reason": engine.unavailable,
            "neonize": version,
            "links": engine.counts(),
        }

    @app.get("/links/{user_id}", dependencies=auth)
    async def get_link(user_id: str) -> dict[str, Any]:
        try:
            return engine.view(user_id)
        except LinkError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from None

    @app.post("/links/{user_id}/start", dependencies=auth)
    async def start(user_id: str, body: StartBody) -> dict[str, Any]:
        return await answer(engine.link)(user_id, consented_at=body.consented_at)

    @app.post("/links/{user_id}/pair-code", dependencies=auth)
    async def pair_code(user_id: str, body: PairCodeBody) -> dict[str, Any]:
        return await answer(engine.pair_code)(user_id, body.phone)

    @app.post("/links/{user_id}/reconnect", dependencies=auth)
    async def reconnect(user_id: str) -> dict[str, Any]:
        return await answer(engine.reconnect)(user_id)

    @app.post("/links/{user_id}/cancel", dependencies=auth)
    async def cancel(user_id: str) -> dict[str, Any]:
        return await answer(engine.cancel)(user_id)

    # A new path rather than DELETE /links/{user_id}: a service from before kept
    # history answers 404 instead of deleting what the user asked to keep.
    @app.post("/links/{user_id}/disconnect", dependencies=auth)
    async def disconnect(user_id: str, body: DisconnectBody) -> dict[str, Any]:
        return await answer(engine.unlink)(user_id, keep_history=body.keep_history)

    @app.delete("/links/{user_id}/kept/{entry}", dependencies=auth)
    async def delete_kept(user_id: str, entry: str) -> dict[str, Any]:
        return await answer(engine.delete_kept)(user_id, entry)

    @app.get("/links/{user_id}/chats", dependencies=auth)
    async def chats(user_id: str, kept: bool = False) -> dict[str, Any]:
        return await answer(engine.chats)(user_id, with_kept=kept)

    @app.post("/links/{user_id}/messages", dependencies=auth)
    async def messages(user_id: str, body: MessagesBody) -> dict[str, Any]:
        return await answer(engine.messages)(
            user_id, chat=body.chat, query=body.query, since=body.since, until=body.until,
            limit=body.limit, agent=body.agent, mark_seen=body.mark_seen, with_kept=body.kept)

    @app.post("/links/{user_id}/contacts", dependencies=auth)
    async def contacts(user_id: str, body: ContactsBody) -> dict[str, Any]:
        return await answer(engine.contacts)(user_id, query=body.query, limit=body.limit)

    @app.post("/links/{user_id}/news", dependencies=auth)
    async def news(user_id: str, body: NewsBody) -> dict[str, Any]:
        return await answer(engine.news)(user_id, agent=body.agent)

    @app.post("/links/{user_id}/attachment", dependencies=auth)
    async def attachment(user_id: str, body: AttachmentBody) -> Response:
        """The file's bytes; its details (JSON, ASCII) in ``X-WAW-Meta``."""
        try:
            data, meta = await engine.attachment(user_id, chat=body.chat, message_id=body.id)
        except LinkError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from None
        return Response(content=data, media_type=meta.get("mimetype") or "application/octet-stream",
                        headers={"X-WAW-Meta": json.dumps(meta, ensure_ascii=True)})

    @app.get("/links/{user_id}/send-policy", dependencies=auth)
    async def get_send_policy(user_id: str) -> dict[str, Any]:
        return await answer(engine.send_policy)(user_id)

    @app.put("/links/{user_id}/send-policy", dependencies=auth)
    async def put_send_policy(user_id: str, body: SendPolicyBody) -> dict[str, Any]:
        return await answer(engine.set_send_policy)(user_id, body.model_dump())

    @app.post("/links/{user_id}/send", dependencies=auth)
    async def send(user_id: str, body: SendBody) -> dict[str, Any]:
        return await answer(engine.send)(user_id, chat=body.chat, text=body.text, agent=body.agent,
                                         allow_repeat=body.allow_repeat)

    @app.get("/links/{user_id}/autoreply", dependencies=auth)
    async def get_autoreply(user_id: str) -> dict[str, Any]:
        return await answer(engine.autoreply_config)(user_id)

    @app.put("/links/{user_id}/autoreply", dependencies=auth)
    async def put_autoreply(user_id: str, body: AutoreplyBody) -> dict[str, Any]:
        return await answer(engine.set_autoreply_config)(user_id, body.model_dump())

    @app.post("/links/{user_id}/autoreply-log", dependencies=auth)
    async def add_autoreply_log(user_id: str, body: AutoreplyLogBody) -> dict[str, Any]:
        return await answer(engine.add_autoreply_log)(user_id, chat=body.chat, outcome=body.outcome,
                                                       details=body.details)

    @app.get("/links/{user_id}/autoreply-log", dependencies=auth)
    async def get_autoreply_log(user_id: str, limit: int = 50, chat: str = "", ignored: bool = True) -> dict[str, Any]:
        return await answer(engine.autoreply_log)(user_id, limit=limit, chat=chat, ignored=ignored)

    @app.put("/links/{user_id}/retention", dependencies=auth)
    async def retention(user_id: str, body: RetentionBody) -> dict[str, Any]:
        return await answer(engine.set_retention)(user_id, body.days)

    @app.get("/events", dependencies=auth)
    async def events(after: int = 0) -> dict[str, Any]:
        return {"events": engine.events_after(after)}

    return app
