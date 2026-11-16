"""The deck's shared Telegram bot: one bot for the whole Flight Deck, many agents.

An agent talks Telegram one of two ways:

* **Its own bot** (unchanged): ``telegram.enabled`` + ``telegram.bot_token`` in
  the agent's config, and the agent long-polls Telegram itself.
* **The deck's shared bot** (this module): an admin sets one bot token in
  Admin → Integrations. A user asks for it in Connections → Telegram, picking
  one of their own agents; an admin approves; the user opens the bot through a
  one-time link (``t.me/<bot>?start=<code>``) that binds their Telegram account.

Flight Deck is the only process that polls the shared bot. It routes each update
by the sender's Telegram id to that user's agent, and the agent reaches Telegram
through a relay shaped like the Bot API (``/fd/telegram-relay/bot…/<method>``,
see telegram_shared_routes). The agent runs the same Telegram code it runs
against its own bot; only its ``api_base_url`` points here. The relay:

* serves ``getUpdates`` from the agent's queue (a long poll), and
* forwards every other call with the shared token, after checking that it only
  touches the agent's own chats, files and button presses.

So the token never leaves Flight Deck, no agent can read or write another
user's chat, and conversations, sessions and files stay in the user's agent.
Flight Deck relays the bytes and stores only the link (Telegram id ↔ user ↔
agent).

Groups: a user can bring their agent into a Telegram group
(telegram_agent_groups); that group's messages then go to their agent too, and
the agent may post there and leave it.

Messages for an agent that isn't polling wait in memory (bounded, a few hours),
so a restart of Flight Deck drops them; the sender is told the agent is offline.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

SETTING = "fd:telegram-shared"

_DEFAULT_API = "https://api.telegram.org"
_POLL_TIMEOUT = 25
_QUEUE_MAX = 500
_QUEUE_TTL = 6 * 3600.0
_SEEN_MAX = 2000
_NOTICE_EVERY = 600.0
# An agent counts as offline once it hasn't polled for this long (a poll is
# held up to ~25 s, so a running agent is back well within it).
_OFFLINE_AFTER = 75.0

# Methods that act on one chat: ``chat_id`` (and ``from_chat_id`` for copies /
# forwards) must be a chat of the calling agent.
CHAT_METHODS = frozenset({
    "sendMessage", "sendPhoto", "sendDocument", "sendAudio", "sendVoice",
    "sendVideo", "sendAnimation", "sendVideoNote", "sendSticker", "sendContact",
    "sendLocation", "sendVenue", "sendChatAction", "sendMediaGroup", "sendPoll",
    "sendDice", "editMessageText", "editMessageCaption", "editMessageMedia",
    "editMessageReplyMarkup", "deleteMessage", "deleteMessages",
    "setMessageReaction", "copyMessage", "forwardMessage",
})


def api_base() -> str:
    """Telegram's Bot API (``FD_TELEGRAM_API_BASE`` points a deck at a stub)."""
    return (os.environ.get("FD_TELEGRAM_API_BASE") or "").strip().rstrip("/") or _DEFAULT_API


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_link_code() -> str:
    # Telegram's /start payload allows [A-Za-z0-9_-], up to 64 characters.
    return secrets.token_urlsafe(18)


# ── config (system setting) ─────────────────────────────────────────


async def load_config(db) -> dict[str, Any]:
    raw = await db.get_system_setting(SETTING)
    try:
        cfg = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "token": str(cfg.get("token") or ""),
        "enabled": bool(cfg.get("enabled", True)),
        # New requests are approved at once (Admin → Integrations); a person
        # an admin turned down still waits for an admin.
        "auto_approve": bool(cfg.get("auto_approve", False)),
        # Users may keep a Telegram inbox through the bot (telegram_inbox.py).
        "inbox": bool(cfg.get("inbox", True)),
        "bot_username": str(cfg.get("bot_username") or ""),
        "bot_name": str(cfg.get("bot_name") or ""),
        "bot_id": str(cfg.get("bot_id") or ""),
    }


async def save_config(db, cfg: dict[str, Any]) -> None:
    await db.set_system_setting(SETTING, json.dumps(cfg, ensure_ascii=True))


def token_hint(token: str) -> str:
    return f"····{token[-4:]}" if len(token) >= 8 else ("····" if token else "")


# ── Telegram calls ──────────────────────────────────────────────────

_client: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0))
    return _client


def _method_url(token: str, method: str) -> str:
    return f"{api_base()}/bot{token}/{method}"


async def call(token: str, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """One JSON Bot API call; Telegram's answer (``{"ok": false, …}`` on errors)."""
    try:
        r = await _http().post(_method_url(token, method), json=payload or {})
        data = r.json()
    except Exception as exc:  # network / not JSON
        return {"ok": False, "description": f"{type(exc).__name__}: {exc}"}
    return data if isinstance(data, dict) else {"ok": False, "description": "bad response"}


async def verify_token(token: str) -> dict[str, Any]:
    """``getMe`` for ``token``; raises ``ValueError`` with a readable reason."""
    data = await call(token, "getMe")
    if not data.get("ok") or not isinstance(data.get("result"), dict):
        raise ValueError(
            "Telegram didn't accept this bot token"
            + (f" ({data.get('description')})" if data.get("description") else "")
        )
    return data["result"]


async def send_text(chat_id: str | int, text: str) -> bool:
    """A plain message from the shared bot (Flight Deck's own notices)."""
    token = _state.token
    if not token:
        return False
    data = await call(token, "sendMessage", {
        "chat_id": int(chat_id), "text": text, "disable_web_page_preview": True,
    })
    return bool(data.get("ok"))


# ── per-agent mailbox ───────────────────────────────────────────────


@dataclass
class _Box:
    queue: deque = field(default_factory=lambda: deque(maxlen=_QUEUE_MAX))  # (t, update)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    last_poll: float = 0.0
    waiting: int = 0
    files: OrderedDict = field(default_factory=OrderedDict)
    file_paths: OrderedDict = field(default_factory=OrderedDict)
    callbacks: OrderedDict = field(default_factory=OrderedDict)

    def online(self) -> bool:
        return self.waiting > 0 or (time.monotonic() - self.last_poll) < _OFFLINE_AFTER

    def ack(self, offset: int) -> None:
        while self.queue and int(self.queue[0][1].get("update_id", 0)) < offset:
            self.queue.popleft()

    def expire(self) -> None:
        cutoff = time.monotonic() - _QUEUE_TTL
        while self.queue and self.queue[0][0] < cutoff:
            self.queue.popleft()


def _remember(seen: OrderedDict, key: str) -> None:
    if not key:
        return
    seen[key] = None
    seen.move_to_end(key)
    while len(seen) > _SEEN_MAX:
        seen.popitem(last=False)


def _file_ids(obj: Any, out: list[str]) -> list[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "file_id" and isinstance(v, str):
                out.append(v)
            else:
                _file_ids(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _file_ids(v, out)
    return out


def update_sender(update: dict[str, Any]) -> str:
    """The Telegram user id an update came from ("" if none)."""
    for key in ("message", "callback_query"):
        part = update.get(key)
        if isinstance(part, dict) and isinstance(part.get("from"), dict):
            return str(part["from"].get("id") or "")
    return ""


def update_group(update: dict[str, Any]) -> str:
    """The group (chat id) an update belongs to; "" for a private chat. A
    group's last message before it became a supergroup belongs to the new id."""
    for key in ("message", "my_chat_member"):
        part = update.get(key)
        chat = part.get("chat") if isinstance(part, dict) else None
        if isinstance(chat, dict):
            if chat.get("type") not in ("group", "supergroup"):
                return ""
            return str(part.get("migrate_to_chat_id") or chat.get("id") or "")
    return ""


def _belongs(update: dict[str, Any], chats: set[str], groups: set[str]) -> bool:
    """An update for the agent whose private chats and groups these are."""
    group = update_group(update)
    if not group:
        return update_sender(update) in chats
    member = update.get("my_chat_member")
    if isinstance(member, dict) and str((member.get("new_chat_member") or {}).get("status") or "") \
            in ("left", "kicked"):
        return True  # the bot left a group it was in for this agent (no longer bound)
    return group in groups


# ── state ───────────────────────────────────────────────────────────


@dataclass
class _State:
    token: str = ""
    bot_username: str = ""
    task: asyncio.Task | None = None
    running: bool = False
    last_ok_at: str | None = None
    error: str = ""
    links_by_tg: dict[str, dict] = field(default_factory=dict)
    links_by_user: dict[str, dict] = field(default_factory=dict)
    # Groups a user brought their agent into: chat id → Flight Deck user id
    # (telegram_agent_groups; the user's link names the agent).
    groups: dict[str, str] = field(default_factory=dict)
    bot_id: str = ""
    boxes: dict[str, _Box] = field(default_factory=dict)
    notices: dict[tuple[str, str], float] = field(default_factory=dict)
    inbox: bool = True
    # What BotFather allows the bot (getMe): connecting to accounts through
    # Chat Automation, and reading every group message (privacy mode off).
    can_business: bool | None = None
    reads_groups: bool | None = None
    flags_at: float = 0.0


_state = _State()


def _box(slug: str) -> _Box:
    box = _state.boxes.get(slug)
    if box is None:
        box = _state.boxes[slug] = _Box()
    return box


def status() -> dict[str, Any]:
    return {
        "running": _state.running,
        "last_ok_at": _state.last_ok_at,
        "error": _state.error,
        "bot_flags": bot_flags(),
    }


def bot_username() -> str:
    return _state.bot_username


def inbox_allowed() -> bool:
    return _state.inbox


def bot_flags() -> dict[str, bool | None]:
    return {"business": _state.can_business, "groups": _state.reads_groups}


_FLAGS_EVERY = 600.0


async def refresh_flags(force: bool = False) -> None:
    """Re-read the bot's BotFather switches (getMe), at most every few minutes."""
    token = _state.token
    if not token or (not force and time.monotonic() - _state.flags_at < _FLAGS_EVERY):
        return
    _state.flags_at = time.monotonic()
    data = await call(token, "getMe")
    me = data.get("result") if data.get("ok") else None
    if isinstance(me, dict):
        _state.can_business = bool(me.get("can_connect_to_business"))
        _state.reads_groups = bool(me.get("can_read_all_group_messages"))
        _state.bot_id = str(me.get("id") or _state.bot_id)
        _state.bot_username = str(me.get("username") or _state.bot_username)


def bot_id() -> str:
    return _state.bot_id


def agent_online(slug: str) -> bool | None:
    """Whether the agent is polling the relay (``None``: never seen)."""
    box = _state.boxes.get(slug)
    if box is None or (box.last_poll == 0.0 and box.waiting == 0):
        return None
    return box.online()


async def reload_links(db) -> None:
    """Re-read the links from the database (after every change)."""
    rows = await db.list_telegram_links()
    _state.links_by_user = {str(r["user_id"]): r for r in rows}
    _state.links_by_tg = {str(r["tg_user_id"]): r for r in rows if r.get("tg_user_id")}
    # Drop queued messages that no longer belong to the agent they wait for
    # (access removed, or the user moved to another agent), and wake every
    # held poll so a new approval is picked up at once.
    for slug, box in _state.boxes.items():
        keep, groups = chats_for_slug(slug), groups_for_slug(slug)
        if box.queue:
            box.queue = deque(
                ((t, u) for t, u in box.queue if _belongs(u, keep, groups)), maxlen=_QUEUE_MAX)
        box.event.set()


async def reload_groups(db) -> None:
    """Re-read which groups belong to whom (start, and every change)."""
    _state.groups = {str(r["chat_id"]): str(r["user_id"]) for r in await db.list_tg_agent_groups()}


def chats_for_slug(slug: str, owner: str | None = None) -> set[str]:
    """Telegram chats (= user ids; private chats only) routed to ``slug`` —
    approved, linked, and (when ``owner`` is given) requested by that owner."""
    return {
        tg for tg, link in _state.links_by_tg.items()
        if link.get("status") == "approved" and link.get("agent_slug") == slug
        and (owner is None or link.get("user_id") == owner)
    }


def groups_for_slug(slug: str, owner: str | None = None) -> set[str]:
    """Groups whose messages go to ``slug``: brought in by a user whose
    approved link names that agent (and who is ``owner``, when given)."""
    out: set[str] = set()
    for chat_id, user_id in _state.groups.items():
        link = _state.links_by_user.get(user_id)
        if link and link.get("status") == "approved" and link.get("agent_slug") == slug \
                and (owner is None or user_id == owner):
            out.add(chat_id)
    return out


def owner_chat(user_id: str) -> str:
    """The user's own approved, linked chat with the shared bot ("" when
    none) — where Flight Deck sends them things itself (a scheduled task's
    result)."""
    link = _state.links_by_user.get(str(user_id or ""))
    if not link or link.get("status") != "approved":
        return ""
    return str(link.get("tg_user_id") or "")


# ── poller ──────────────────────────────────────────────────────────


async def start(db) -> None:
    """(Re)start polling with the configured token — no-op without one."""
    await stop()
    cfg = await load_config(db)
    _state.token = cfg["token"] if cfg["enabled"] else ""
    _state.bot_username = cfg["bot_username"]
    _state.bot_id = cfg["bot_id"]
    _state.inbox = cfg["inbox"]
    _state.can_business = _state.reads_groups = None
    _state.flags_at = 0.0
    _state.error = ""
    _state.last_ok_at = None
    await reload_groups(db)
    await reload_links(db)
    if not _state.token:
        return
    _state.task = asyncio.create_task(_poll_loop(db, _state.token))


async def stop() -> None:
    task, _state.task = _state.task, None
    _state.running = False
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# Chats, and the bot joining / leaving groups (agents' groups and the inbox's).
_CHAT_UPDATES = ["message", "callback_query", "my_chat_member"]
# The Telegram inbox's: Chat Automation (business) connections and their
# messages, and group edits.
_INBOX_UPDATES = [
    "edited_message", "business_connection", "business_message",
    "edited_business_message", "deleted_business_messages",
]


async def _poll_loop(db, token: str) -> None:
    offset: int | None = None
    backoff = 2.0
    _state.running = True
    log.info("Shared Telegram bot polling started")
    updates = _CHAT_UPDATES + (_INBOX_UPDATES if _state.inbox else [])
    try:
        while True:
            try:
                await refresh_flags()
            except Exception:
                pass
            if _state.inbox:
                from captain_claw.flight_deck import telegram_inbox

                await telegram_inbox.maybe_purge(db)
            try:
                params: dict[str, Any] = {
                    "timeout": _POLL_TIMEOUT,
                    "allowed_updates": json.dumps(updates),
                }
                if offset is not None:
                    params["offset"] = offset
                r = await _http().get(_method_url(token, "getUpdates"), params=params,
                                      timeout=_POLL_TIMEOUT + 15)
                if r.status_code == 409:
                    _state.error = (
                        "Another program is reading this bot's updates (or a webhook is "
                        "set on it). Give the shared bot a token nothing else uses."
                    )
                    await asyncio.sleep(30)
                    continue
                if r.status_code in (401, 404):
                    _state.error = "Telegram rejected the bot token — set a new one."
                    await asyncio.sleep(120)
                    continue
                r.raise_for_status()
                data = r.json()
                for update in data.get("result") or []:
                    if not isinstance(update, dict):
                        continue
                    offset = max(offset or 0, int(update.get("update_id", 0)) + 1)
                    try:
                        await route_update(db, update)
                    except Exception as exc:
                        log.warning("Shared Telegram update not routed: %s", type(exc).__name__)
                _state.error = ""
                _state.last_ok_at = _now_iso()
                backoff = 2.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _state.error = f"Can't reach Telegram ({type(exc).__name__})"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
    finally:
        _state.running = False


# ── routing ─────────────────────────────────────────────────────────

_NOTICE_UNKNOWN = (
    "👋 This is your team's Flight Deck bot. To use it, open Flight Deck → "
    "Connections → Telegram, pick the agent that should answer you and request "
    "access. Once you're approved, tap the link shown there."
)
_NOTICE_PENDING = (
    "⏳ Your access is waiting for an admin's approval. I'll let you know here "
    "as soon as it's approved."
)
_NOTICE_REJECTED = (
    "Your request to use this bot wasn't approved. Ask your Flight Deck admin, "
    "or request it again in Connections → Telegram."
)


async def _notice(chat_id: Any, tg_user: str, kind: str, text: str) -> None:
    """A Flight Deck notice to a sender, at most every few minutes per kind —
    the bot never becomes a chatty reflector for strangers."""
    key = (tg_user, kind)
    now = time.monotonic()
    if now - _state.notices.get(key, -_NOTICE_EVERY) < _NOTICE_EVERY:
        return
    _state.notices[key] = now
    if len(_state.notices) > 5000:
        _state.notices.clear()
    try:
        await send_text(chat_id, text)
    except Exception:
        pass


async def route_update(db, update: dict[str, Any]) -> str:
    """Deliver one update; returns what happened (for logs and tests)."""
    # Groups first: the bot joining one, and the messages of a group a user
    # brought their agent into (which the inbox may keep a copy of too).
    from captain_claw.flight_deck import telegram_agent_groups, telegram_inbox

    grouped = await telegram_agent_groups.handle(db, update)
    if grouped is not None:
        return grouped
    # Then the Telegram inbox: Chat Automation traffic, other groups' messages
    # and the bot's membership in them are its, and never reach an agent's chat.
    kept = await telegram_inbox.handle(db, update)
    if kept is not None:
        return kept
    msg = update.get("message")
    cbq = update.get("callback_query")
    if isinstance(msg, dict):
        chat = msg.get("chat") if isinstance(msg.get("chat"), dict) else {}
        sender = msg.get("from") if isinstance(msg.get("from"), dict) else {}
    elif isinstance(cbq, dict):
        cb_msg = cbq.get("message") if isinstance(cbq.get("message"), dict) else {}
        chat = cb_msg.get("chat") if isinstance(cb_msg.get("chat"), dict) else {}
        sender = cbq.get("from") if isinstance(cbq.get("from"), dict) else {}
    else:
        return "ignored"
    # Private chats only: a group would mix several people into one agent.
    if chat and chat.get("type") not in (None, "private"):
        return "ignored:group"
    tg_user = str(sender.get("id") or "")
    if not tg_user or sender.get("is_bot"):
        return "ignored"
    chat_id = chat.get("id") or sender.get("id")

    text = str((msg or {}).get("text") or "").strip()
    if isinstance(msg, dict) and (text == "/start" or text.startswith("/start ")):
        code = text[len("/start"):].strip()
        if code and await _claim_code(db, code, sender, chat_id):
            return "linked"

    link = _state.links_by_tg.get(tg_user)
    if link is None:
        await _notice(chat_id, tg_user, "unknown", _NOTICE_UNKNOWN)
        return "unknown"
    if link.get("status") == "pending":
        await _notice(chat_id, tg_user, "pending", _NOTICE_PENDING)
        return "pending"
    if link.get("status") != "approved" or not link.get("agent_slug"):
        await _notice(chat_id, tg_user, "rejected", _NOTICE_REJECTED)
        return "rejected"

    if isinstance(msg, dict):
        # A message counts against the linked user's plan allowance for today.
        from captain_claw.flight_deck import usage_quota

        allowed, quota = await usage_quota.admit_message(str(link.get("user_id") or ""), text)
        if not allowed:
            try:
                await send_text(chat_id, usage_quota.limit_text(quota))
            except Exception:
                pass
            return "over_limit"

    if isinstance(msg, dict):
        # The user is talking here now: Autonomous Work's phone gate holds its
        # nudges meanwhile and reads this conversation before ringing the phone.
        from captain_claw.flight_deck import phone_gate
        phone_gate.note_inbound(phone_gate.tg_key(tg_user), text or str(msg.get("caption") or ""))

    slug = str(link["agent_slug"])
    from captain_claw.flight_deck import agent_idle

    # Asleep for being idle: enqueue wakes it, and it answers in a moment.
    asleep = agent_idle.is_asleep(slug)
    box = enqueue(slug, update)
    if not box.online() and not asleep:
        name = link.get("agent_name") or slug
        await _notice(chat_id, tg_user, "offline", (
            f"⏸ {name} isn't running right now — start it in Flight Deck. Your "
            "messages wait here for a few hours and reach it once it's back."
        ))
    return "queued"


def enqueue(slug: str, update: dict[str, Any]) -> _Box:
    """Hand ``update`` to the agent ``slug`` (its next ``getUpdates``)."""
    box = _box(slug)
    box.expire()
    box.queue.append((time.monotonic(), update))
    # Asleep for being idle (agent_idle): it polls again once woken.
    from captain_claw.flight_deck import agent_idle

    agent_idle.touch(slug)
    agent_idle.wake_soon(slug)
    for fid in _file_ids(update, []):
        _remember(box.files, fid)
    cbq = update.get("callback_query")
    if isinstance(cbq, dict):
        _remember(box.callbacks, str(cbq.get("id") or ""))
    box.event.set()
    return box


async def _claim_code(db, code: str, sender: dict, chat_id: Any) -> bool:
    """``/start <code>`` from the link shown in Connections: binds this Telegram
    account to the user who requested access. False when the code isn't ours."""
    row = await db.get_telegram_link_by_code(code)
    if row is None:
        return False
    tg_user = str(sender.get("id") or "")
    other = _state.links_by_tg.get(tg_user)
    if other and other.get("user_id") != row["user_id"]:
        await send_text(chat_id, (
            "This Telegram account is already connected to another Flight Deck user. "
            "Disconnect it there first."
        ))
        return True
    await db.update_telegram_link(
        row["user_id"], tg_user_id=tg_user,
        tg_username=str(sender.get("username") or "")[:64],
        tg_first_name=str(sender.get("first_name") or "")[:64],
        linked_at=_now_iso(), link_code="",
    )
    await reload_links(db)
    name = row.get("agent_name") or row.get("agent_slug") or "your agent"
    if row.get("status") == "approved":
        await send_text(chat_id, f"✅ Connected. What you send here now goes to your agent {name}.")
    else:
        await send_text(chat_id, (
            "🔗 Your Telegram account is linked. An admin still has to approve access — "
            "I'll let you know here when they do."
        ))
    return True


# ── relay (agent side of the shared bot) ────────────────────────────


async def relay_get_updates(
    slug: str, owner: str, offset: int | None, timeout: float,
) -> list[dict[str, Any]]:
    """The agent's ``getUpdates``: its queued updates, held up to ``timeout``.

    Like Telegram, ``offset`` confirms everything below it. An agent with no
    approved link just waits out the poll — it notices an approval at once
    because `reload_links` wakes it."""
    box = _box(slug)
    box.waiting += 1
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, min(float(timeout), 50.0))
    try:
        if offset is not None:
            box.ack(int(offset))
        while True:
            box.expire()
            chats, groups = chats_for_slug(slug, owner), groups_for_slug(slug, owner)
            items = [u for _t, u in box.queue if _belongs(u, chats, groups)] if chats or groups else []
            if items:
                return items[:100]
            remaining = deadline - loop.time()
            if remaining <= 0:
                return []
            box.event.clear()
            try:
                await asyncio.wait_for(box.event.wait(), timeout=remaining)
            except TimeoutError:
                return []
    finally:
        box.waiting -= 1
        box.last_poll = time.monotonic()


class RelayRefused(Exception):
    """A relayed call the shared bot won't make for this agent."""


def _int_or_str(v: Any) -> str:
    return str(v).strip()


def check_call(slug: str, owner: str, method: str, params: dict[str, Any]) -> None:
    """Raise `RelayRefused` unless ``method`` with ``params`` stays inside the
    calling agent's own chats, files and button presses."""
    box = _box(slug)
    groups = groups_for_slug(slug, owner)
    chats = chats_for_slug(slug, owner) | groups
    # Inline messages and business connections aren't tied to a chat the
    # relay can check — the shared bot uses neither.
    for key in ("inline_message_id", "business_connection_id"):
        if params.get(key):
            raise RelayRefused(f"{key} isn't available through the shared bot")
    if method == "leaveChat":
        # Only a group this agent was brought into (the relay route unbinds it).
        if _int_or_str(params.get("chat_id", "")) not in groups:
            raise RelayRefused("the shared bot only leaves this agent's own groups")
        return
    if method in CHAT_METHODS:
        for key in ("chat_id", "from_chat_id"):
            if key in params and _int_or_str(params[key]) not in chats:
                raise RelayRefused(f"chat {params[key]} isn't one of this agent's chats")
        if "chat_id" not in params:
            raise RelayRefused("the shared bot only acts in this agent's own chats")
        return
    if method == "answerCallbackQuery":
        if str(params.get("callback_query_id") or "") not in box.callbacks:
            raise RelayRefused("unknown callback query")
        return
    if method == "getFile":
        if str(params.get("file_id") or "") not in box.files:
            raise RelayRefused("unknown file")
        return
    if method in ("getMe", "setMyCommands", "deleteMyCommands", "getMyCommands",
                  "setChatMenuButton"):
        return
    raise RelayRefused(f"{method} isn't available through the shared bot")


def note_file_path(slug: str, file_path: str) -> None:
    _remember(_box(slug).file_paths, file_path)


def file_path_allowed(slug: str, file_path: str) -> bool:
    return file_path in _box(slug).file_paths


def token() -> str:
    return _state.token
