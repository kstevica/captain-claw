"""HTTP side of the deck's shared Telegram bot (engine: telegram_shared.py).

* ``/fd/telegram-shared``        — a user's own link: request access with one of
  their agents, change the agent, disconnect, and the Telegram groups they
  brought that agent into (Connections → Telegram).
* ``/fd/admin/telegram-shared``  — the bot token, the poller's health, and the
  access requests to approve / reject / remove (Admin → Integrations).
* ``/fd/telegram-relay/…``       — what the user's agent talks to instead of
  api.telegram.org: Bot-API-shaped, identified by ``X-Agent-Auth``, limited to
  the agent's own chats (see `telegram_shared.check_call`).

The shared bot needs sign-in (it routes Telegram accounts to Flight Deck users);
on a deck with auth off every route here answers "not available".
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from captain_claw import telegram_groups
from captain_claw.flight_deck import telegram_agent_groups, telegram_inbox
from captain_claw.flight_deck import telegram_shared as ts
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.agent_identity import require_agent_caller
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["telegram-shared"])


def _auth_enabled() -> bool:
    return os.environ.get("FD_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")


def _require_auth_deck() -> None:
    if not _auth_enabled():
        raise HTTPException(
            status_code=409,
            detail="The shared Telegram bot needs Flight Deck sign-in, which is off on this deck",
        )


# ── the user's agents ───────────────────────────────────────────────


def own_agents(user_id: str) -> list[dict[str, str]]:
    """The agents ``user_id`` owns on this deck (process registry + this deck's
    containers): ``[{slug, name, kind, status}]``."""
    from captain_claw.flight_deck import server as _srv

    out: list[dict[str, str]] = []
    try:
        for c in _srv._deck_containers(all=True):
            labels = c.labels or {}
            if labels.get(_srv.OWNER_LABEL, "") != user_id:
                continue
            name = labels.get("flight-deck.agent-name", "") or c.name
            out.append({"slug": _srv._slug(name), "name": name, "kind": "docker",
                        "status": "running" if c.status == "running" else "stopped"})
    except Exception:
        pass  # no Docker on this host
    for slug, entry in _srv._load_process_registry().items():
        if str(entry.get("owner") or "") != user_id:
            continue
        out.append({"slug": slug, "name": str(entry.get("name") or slug), "kind": "process",
                    "status": "running" if _srv._process_is_alive(slug) else "stopped"})
    out.sort(key=lambda a: (a["status"] != "running", a["name"].lower()))
    return out


# ── user routes ─────────────────────────────────────────────────────


def _deep_link(code: str) -> str:
    bot = ts.bot_username()
    return f"https://t.me/{bot}?start={code}" if bot and code else ""


def _link_view(link: dict | None) -> dict[str, Any] | None:
    if not link:
        return None
    slug = str(link.get("agent_slug") or "")
    return {
        "status": link.get("status"),
        "agent_slug": slug,
        "agent_name": link.get("agent_name") or slug,
        "linked": bool(link.get("tg_user_id")),
        "tg_username": link.get("tg_username") or "",
        "tg_first_name": link.get("tg_first_name") or "",
        "requested_at": link.get("requested_at"),
        "decided_at": link.get("decided_at"),
        "linked_at": link.get("linked_at"),
        "deep_link": _deep_link(str(link.get("link_code") or "")),
        "agent_online": ts.agent_online(slug) if slug else None,
    }


async def _me(user: dict) -> dict[str, Any]:
    auth = _auth_enabled()
    cfg = await ts.load_config(get_db())
    available = bool(auth and cfg["token"] and cfg["enabled"])
    link = await get_db().get_telegram_link(user["id"]) if auth else None
    connected = bool(available and link and link.get("status") == "approved" and link.get("tg_user_id"))
    return {
        "auth_enabled": auth,
        "available": available,
        "auto_approve": bool(available and cfg["auto_approve"]),
        "bot_username": cfg["bot_username"] if available else "",
        "link": _link_view(link),
        "agents": own_agents(user["id"]) if auth else [],
        # Groups: the add-to-group link once connected, and the groups the
        # user's agent is in (telegram_agent_groups).
        "group_link": telegram_groups.add_to_group_url(cfg["bot_username"]) if connected else "",
        "groups": await telegram_agent_groups.groups_of(get_db(), user["id"]) if auth else [],
    }


@router.get("/fd/telegram-shared")
async def my_telegram(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """This user's view of the shared bot: is there one, their link, their agents."""
    return await _me(user)


class RequestBody(BaseModel):
    agent_slug: str


async def _notify_admins(title: str, body: str, ref_id: str) -> None:
    db = get_db()
    try:
        for admin_id in await db.list_admin_ids():
            await db.add_notification(admin_id, "telegram", title, body=body,
                                      ref_type="telegram_link", ref_id=ref_id)
    except Exception as exc:
        log.warning("Telegram request notification failed: %s", type(exc).__name__)


@router.post("/fd/telegram-shared/request")
async def request_access(body: RequestBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Ask for the shared bot with one of your agents — or, once asked, move
    your Telegram to another of your agents (no new approval needed: the admin
    approves the person, which of their own agents answers is theirs to pick)."""
    _require_auth_deck()
    db = get_db()
    cfg = await ts.load_config(db)
    if not (cfg["token"] and cfg["enabled"]):
        raise HTTPException(status_code=409, detail="This deck has no shared Telegram bot yet")
    agent = next((a for a in own_agents(user["id"]) if a["slug"] == body.agent_slug), None)
    if agent is None:
        raise HTTPException(status_code=404, detail="Pick one of your own agents")
    link = await db.get_telegram_link(user["id"])
    if link and link.get("status") in ("pending", "approved"):
        await db.update_telegram_link(user["id"], agent_slug=agent["slug"], agent_name=agent["name"])
        await ts.reload_links(db)
        if link.get("tg_user_id") and link.get("status") == "approved" \
                and link.get("agent_slug") != agent["slug"]:
            await ts.send_text(link["tg_user_id"], f"↪️ From now on {agent['name']} answers you here.")
    else:
        # Turned down before: that stays an admin's call, auto-approve or not.
        auto = cfg["auto_approve"] and not (link and link.get("status") == "rejected")
        await db.create_telegram_link(user["id"], agent["slug"], agent["name"], ts.new_link_code())
        who = user.get("display_name") or user.get("email") or user["id"]
        if auto:
            await _set_decision(db, user["id"], "approved", "auto")
            await _notify_admins(
                "Telegram access approved automatically",
                f"{who} connected the team Telegram bot to their agent {agent['name']}. "
                "Requests are approved automatically (Admin → Integrations).",
                user["id"],
            )
        else:
            await ts.reload_links(db)
            await _notify_admins(
                "Telegram access request",
                f"{who} asked to use the team Telegram bot with their agent {agent['name']}. "
                "Approve it in Admin → Integrations.",
                user["id"],
            )
    return await _me(user)


@router.delete("/fd/telegram-shared/groups/{chat_id}")
async def remove_group(chat_id: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Take your agent out of a Telegram group."""
    _require_auth_deck()
    db = get_db()
    if telegram_agent_groups.bound_to(chat_id) != user["id"]:
        raise HTTPException(status_code=404, detail="Your agent isn't in that group")
    _slug, name = telegram_agent_groups._agent_of(user["id"])
    await telegram_agent_groups.release(
        db, chat_id, say=telegram_agent_groups._REMOVED.format(agent=name or "This agent"))
    return await _me(user)


@router.delete("/fd/telegram-shared")
async def disconnect(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Withdraw the request / disconnect Telegram from your agent."""
    _require_auth_deck()
    db = get_db()
    link = await db.get_telegram_link(user["id"])
    if link:
        # The agent leaves the user's groups; then the link goes.
        await telegram_agent_groups.forget_user(db, user["id"])
        await db.delete_telegram_link(user["id"])
        await ts.reload_links(db)
        # The Telegram inbox rides on this link: it goes with it.
        await _forget_inbox(db, user["id"])
        if link.get("tg_user_id"):
            await ts.send_text(link["tg_user_id"], (
                "Disconnected — this bot no longer forwards your messages. "
                "Reconnect any time in Flight Deck → Connections → Telegram."
            ))
    return await _me(user)


async def _forget_inbox(db, user_id: str) -> None:
    """Delete the user's Telegram inbox (nothing to do if they never had one)."""
    try:
        if await db.get_tg_inbox_settings(user_id):
            await telegram_inbox.forget(db, user_id)
    except Exception as exc:
        log.warning("Telegram inbox not cleared: %s", type(exc).__name__)


# ── admin routes ────────────────────────────────────────────────────


async def _admin_view() -> dict[str, Any]:
    db = get_db()
    cfg = await ts.load_config(db)
    links = []
    for row in await db.list_telegram_links():
        view = _link_view(row) or {}
        view.pop("deep_link", None)
        links.append({
            **view,
            "user_id": row["user_id"],
            "user_email": row.get("user_email") or "",
            "user_name": row.get("user_name") or "",
        })
    return {
        "auth_enabled": _auth_enabled(),
        "config": {
            "configured": bool(cfg["token"]),
            "enabled": cfg["enabled"],
            "auto_approve": cfg["auto_approve"],
            "inbox": cfg["inbox"],
            "bot_username": cfg["bot_username"],
            "bot_name": cfg["bot_name"],
            "token_hint": ts.token_hint(cfg["token"]),
        },
        "status": ts.status(),
        "links": links,
    }


@router.get("/fd/admin/telegram-shared")
async def admin_get(_admin: dict = Depends(require_admin)) -> dict[str, Any]:
    await ts.refresh_flags()
    return await _admin_view()


@router.put("/fd/admin/telegram-shared")
async def admin_put(request: Request, _admin: dict = Depends(require_admin)) -> dict[str, Any]:
    """Set the bot token (checked with Telegram first) and/or switch the bot on/off.
    Body read by hand so a malformed request never echoes the token into logs."""
    _require_auth_deck()
    try:
        data = await request.json()
    except Exception:
        data = None
    data = data if isinstance(data, dict) else {}
    db = get_db()
    cfg = await ts.load_config(db)
    raw_token = data.get("token")
    if isinstance(raw_token, str) and raw_token.strip():
        token = raw_token.strip()
        if not (token.isascii() and token.isprintable()) or any(c.isspace() for c in token):
            raise HTTPException(status_code=422, detail="That doesn't look like a bot token")
        try:
            me = await ts.verify_token(token)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        cfg.update(token=token, bot_username=str(me.get("username") or ""),
                   bot_name=str(me.get("first_name") or ""), bot_id=str(me.get("id") or ""))
    if isinstance(data.get("enabled"), bool):
        cfg["enabled"] = data["enabled"]
    newly_auto = data.get("auto_approve") is True and not cfg["auto_approve"]
    if isinstance(data.get("auto_approve"), bool):
        cfg["auto_approve"] = data["auto_approve"]
    if isinstance(data.get("inbox"), bool):
        cfg["inbox"] = data["inbox"]
    await ts.save_config(db, cfg)
    await ts.start(db)
    if newly_auto:
        # Switching it on lets in the people waiting now, too.
        for row in await db.list_telegram_links():
            if row.get("status") == "pending":
                await _approve(str(row["user_id"]), _admin)
    return await _admin_view()


@router.delete("/fd/admin/telegram-shared")
async def admin_clear(_admin: dict = Depends(require_admin)) -> dict[str, Any]:
    """Forget the bot token and stop polling. Requests and links stay, so a new
    token picks them up again."""
    db = get_db()
    cfg = await ts.load_config(db)
    await ts.save_config(db, {"enabled": True, "auto_approve": cfg["auto_approve"],
                              "inbox": cfg["inbox"]})
    await ts.start(db)
    return await _admin_view()


async def _set_decision(db, user_id: str, status: str, decided_by: str) -> None:
    from datetime import UTC, datetime

    await db.update_telegram_link(
        user_id, status=status, decided_at=datetime.now(UTC).isoformat(),
        decided_by=decided_by,
    )
    await ts.reload_links(db)


async def _decide(user_id: str, status: str, admin: dict) -> dict:
    db = get_db()
    link = await db.get_telegram_link(user_id)
    if not link:
        raise HTTPException(status_code=404, detail="No such request")
    await _set_decision(db, user_id, status, str(admin.get("id") or ""))
    return link


async def _approve(user_id: str, admin: dict) -> None:
    """Approve *user_id*'s request and tell them (Telegram once linked, else a
    Flight Deck notification with the way to finish connecting)."""
    link = await _decide(user_id, "approved", admin)
    name = link.get("agent_name") or link.get("agent_slug")
    if link.get("tg_user_id"):
        await ts.send_text(link["tg_user_id"], (
            f"✅ Approved. What you send here now goes to your agent {name}."))
    await get_db().add_notification(
        user_id, "success", "Telegram access approved",
        body=(f"Your agent {name} now answers you on Telegram." if link.get("tg_user_id")
              else "Open Connections → Telegram and tap the link to finish connecting."),
        ref_type="telegram_access", ref_id=user_id,  # a result, not the request's needs-you ref
    )


@router.post("/fd/admin/telegram-shared/links/{user_id}/approve")
async def admin_approve(user_id: str, admin: dict = Depends(require_admin)) -> dict[str, Any]:
    await _approve(user_id, admin)
    return await _admin_view()


@router.post("/fd/admin/telegram-shared/links/{user_id}/reject")
async def admin_reject(user_id: str, admin: dict = Depends(require_admin)) -> dict[str, Any]:
    await telegram_agent_groups.forget_user(get_db(), user_id)
    link = await _decide(user_id, "rejected", admin)
    await _forget_inbox(get_db(), user_id)
    if link.get("tg_user_id"):
        await ts.send_text(link["tg_user_id"], "Your access to this bot was turned off by an admin.")
    await get_db().add_notification(
        user_id, "info", "Telegram access not approved",
        body="An admin didn't approve your request for the team Telegram bot.",
        ref_type="telegram_access", ref_id=user_id,  # a result, not the request's needs-you ref
    )
    return await _admin_view()


@router.delete("/fd/admin/telegram-shared/links/{user_id}")
async def admin_remove(user_id: str, _admin: dict = Depends(require_admin)) -> dict[str, Any]:
    db = get_db()
    link = await db.get_telegram_link(user_id)
    if link:
        await telegram_agent_groups.forget_user(db, user_id)
        await db.delete_telegram_link(user_id)
        await ts.reload_links(db)
        await _forget_inbox(db, user_id)
        if link.get("tg_user_id"):
            await ts.send_text(link["tg_user_id"], "Your access to this bot was removed by an admin.")
    return await _admin_view()


# ── relay (the agent's api.telegram.org) ────────────────────────────


def _tg_error(status: int, description: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={
        "ok": False, "error_code": status, "description": description})


async def _call_params(request: Request) -> tuple[dict[str, Any], dict[str, tuple]]:
    """A Bot API call's parameters, however the agent sent them (query string,
    JSON, form or multipart), plus any uploaded files."""
    params: dict[str, Any] = dict(request.query_params)
    files: dict[str, tuple] = {}
    if request.method != "POST":
        return params, files
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            body = await request.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            params.update(body)
    elif "multipart/form-data" in ctype or "application/x-www-form-urlencoded" in ctype:
        form = await request.form()
        for key, value in form.multi_items():
            if hasattr(value, "read"):
                files[key] = (value.filename or key, await value.read(),
                              value.content_type or "application/octet-stream")
            else:
                params[key] = value
    return params, files


def _commands(params: dict[str, Any]) -> list:
    cmds = params.get("commands")
    if isinstance(cmds, str):
        try:
            cmds = json.loads(cmds)
        except ValueError:
            cmds = []
    return cmds if isinstance(cmds, list) else []


async def _relay_response(r) -> JSONResponse:
    try:
        data = r.json()
    except Exception:
        return _tg_error(502, f"Bad response from Telegram (HTTP {r.status_code})")
    return JSONResponse(status_code=r.status_code, content=data)


@router.api_route("/fd/telegram-relay/bot{token}/{method}", methods=["GET", "POST"])
async def relay(token: str, method: str, request: Request):
    """One Bot API call from an agent on the shared bot. The path's token is a
    placeholder: the agent is known by ``X-Agent-Auth``, never by what it sends."""
    caller = require_agent_caller(request, what="the shared Telegram bot")
    params, files = await _call_params(request)

    if method == "getUpdates":
        raw_offset = params.get("offset")
        try:
            offset = int(raw_offset) if raw_offset not in (None, "") else None
            timeout = float(params.get("timeout") or 0)
        except (TypeError, ValueError):
            return _tg_error(400, "Bad Request: offset / timeout")
        items = await ts.relay_get_updates(caller.slug, caller.owner, offset, timeout)
        return {"ok": True, "result": items}

    bot_token = ts.token()
    if not bot_token:
        return _tg_error(503, "The deck's shared Telegram bot is off")
    try:
        ts.check_call(caller.slug, caller.owner, method, params)
    except ts.RelayRefused as exc:
        return _tg_error(403, f"Forbidden: {exc}")
    if method == "leaveChat":
        # The agent leaves one of its groups; the bot stays while someone keeps
        # the group in their Telegram inbox.
        await telegram_agent_groups.release(get_db(), str(params.get("chat_id")).strip())
        return {"ok": True, "result": True}

    chats = sorted(ts.chats_for_slug(caller.slug, caller.owner))
    # Bot-wide settings are never the agent's to change: commands and the menu
    # button land on the agent's own chats instead.
    if method in ("setMyCommands", "deleteMyCommands"):
        payload = {k: v for k, v in params.items() if k not in ("scope", "commands")}
        if method == "setMyCommands":
            payload["commands"] = _commands(params)
        for chat in chats:
            await ts.call(bot_token, method, {**payload, "scope": {"type": "chat", "chat_id": int(chat)}})
        return {"ok": True, "result": True}
    if method == "setChatMenuButton":
        if "chat_id" in params:
            if str(params["chat_id"]).strip() not in chats:
                return _tg_error(403, "Forbidden: not one of this agent's chats")
            targets = [str(params["chat_id"]).strip()]
        else:
            targets = chats
        button = params.get("menu_button")
        if isinstance(button, str):
            try:
                button = json.loads(button)
            except ValueError:
                button = None
        for chat in targets:
            body: dict[str, Any] = {"chat_id": int(chat)}
            if isinstance(button, dict):
                body["menu_button"] = button
            await ts.call(bot_token, "setChatMenuButton", body)
        return {"ok": True, "result": True}

    url = f"{ts.api_base()}/bot{bot_token}/{method}"
    try:
        if files:
            r = await ts._http().post(url, data={k: str(v) for k, v in params.items()}, files=files)
        else:
            r = await ts._http().post(url, json=params)
    except Exception as exc:
        return _tg_error(502, f"Can't reach Telegram ({type(exc).__name__})")
    if method == "getFile":
        try:
            path = str(((r.json() or {}).get("result") or {}).get("file_path") or "")
        except Exception:
            path = ""
        if path:
            ts.note_file_path(caller.slug, path)
    return await _relay_response(r)


@router.get("/fd/telegram-relay/file/bot{token}/{file_path:path}")
async def relay_file(token: str, file_path: str, request: Request):
    """A file download for a file this agent was sent (``getFile`` first)."""
    caller = require_agent_caller(request, what="the shared Telegram bot")
    bot_token = ts.token()
    if not bot_token:
        return _tg_error(503, "The deck's shared Telegram bot is off")
    if not ts.file_path_allowed(caller.slug, file_path):
        return _tg_error(403, "Forbidden: unknown file")
    try:
        r = await ts._http().get(f"{ts.api_base()}/file/bot{bot_token}/{file_path}")
    except Exception as exc:
        return _tg_error(502, f"Can't reach Telegram ({type(exc).__name__})")
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type") or "application/octet-stream")
