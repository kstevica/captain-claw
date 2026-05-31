"""Send-only Telegram delivery for the FD scheduler.

Deliberately **send-only** (``sendMessage`` via the Bot API). It never calls
``getUpdates`` — that would steal updates from captain-claw's inbound
Telegram bridge (``captain_claw/telegram_bridge.py``), which long-polls. Two
consumers of ``getUpdates`` on one bot conflict; one outbound-only consumer
plus the inbound poller do not.

Configuration (env, picked up from ``.env`` via FD's load_dotenv)::

    TELEGRAM_BOT_TOKEN=123456:ABC...        # same token the inbound bridge uses
    TELEGRAM_RECIPIENTS=Me:123456789,Family:-1009876543210

``TELEGRAM_RECIPIENTS`` is a comma-separated list of ``Name:chat_id`` pairs.
The names populate the scheduler UI's recipient picker; the chat_ids are
what actually get stored on a job's ``delivery_target`` (comma-separated for
multi-recipient).

Finding a chat_id: message ``@userinfobot`` on Telegram, or for a group/
channel add the bot and read the chat id from the inbound bridge logs.
"""

from __future__ import annotations

import logging
import os

import httpx

_log = logging.getLogger(__name__)


def _bot_token() -> str:
    return os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()


def _api_base() -> str:
    # Mirror the inbound bridge's configurable base if present.
    base = os.environ.get("TELEGRAM_API_BASE_URL", "").strip()
    return base or "https://api.telegram.org"


def list_recipients() -> list[dict[str, str]]:
    """Parse ``TELEGRAM_RECIPIENTS`` into ``[{name, chat_id}]``.

    Tolerant: skips malformed entries, trims whitespace. A chat_id may be
    negative (groups/channels), so we only split on the FIRST colon to
    allow names containing colons is unnecessary — names are simple — but
    splitting on the last colon keeps negative ids intact regardless.
    """
    raw = os.environ.get("TELEGRAM_RECIPIENTS", "").strip()
    if not raw:
        return []
    out: list[dict[str, str]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        name, _, chat_id = chunk.rpartition(":")
        name = name.strip()
        chat_id = chat_id.strip()
        if name and chat_id:
            out.append({"name": name, "chat_id": chat_id})
    return out


async def send_telegram(chat_id: str, text: str) -> bool:
    """Send one plain-text message. Returns True on HTTP 200.

    Plain text (no parse_mode) — the agent's markdown is stripped upstream
    before delivery, and unbalanced markdown would otherwise make Telegram
    reject the whole message.
    """
    token = _bot_token()
    chat_id = (chat_id or "").strip()
    text = (text or "").strip()
    if not token or not chat_id or not text:
        return False
    url = f"{_api_base()}/bot{token}/sendMessage"
    # Telegram caps messages at 4096 chars.
    if len(text) > 4096:
        text = text[:4096]
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, json={
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            })
        if r.status_code != 200:
            _log.warning("telegram send to %s failed: %s %s",
                         chat_id, r.status_code, r.text[:200])
            return False
        return True
    except Exception as exc:
        _log.warning("telegram send to %s errored: %s", chat_id, exc)
        return False


async def send_telegram_multi(chat_ids: list[str], text: str) -> tuple[int, int]:
    """Send the same text to several chats. Returns ``(sent, total)``."""
    total = 0
    sent = 0
    for cid in chat_ids:
        cid = (cid or "").strip()
        if not cid:
            continue
        total += 1
        if await send_telegram(cid, text):
            sent += 1
    return sent, total
