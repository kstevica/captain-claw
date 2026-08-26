"""Flight Deck delivery router — the single origin-aware sink that routes an
asynchronously-produced agent result back to wherever the request came from.

The agent hands FD ``{session_id, origin, text}`` (see captain_claw.delivery);
FD switches on ``origin.kind`` and uses the channel's own delivery machinery —
all of which already lives here:

  - whatsapp → push_to_waid           (Cloud API, allowlist + mute enforced)
  - telegram → send_telegram_multi    (configured recipients)
  - glasses / channel → channel bus   (_broadcast to bound subscribers)
  - web → no-op                       (already in the session; nothing to push)

This is the durable counterpart to the old per-turn channel binding: it works
long after the live turn ended, because the origin came from the session's
persisted metadata rather than an in-memory map.
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from captain_claw.flight_deck.auth import get_optional_user
from captain_claw.origin import (
    KIND_CHANNEL,
    KIND_GLASSES,
    KIND_TELEGRAM,
    KIND_WEB,
    KIND_WHATSAPP,
    normalize_origin,
)

_log = logging.getLogger(__name__)

router = APIRouter()


async def deliver_to_origin(origin: dict, text: str) -> tuple[bool, str]:
    """Send ``text`` to ``origin``. Returns ``(delivered, note)``."""
    o = normalize_origin(origin)
    if not o:
        return False, "no-origin"
    kind, address = o["kind"], o["address"]

    if kind == KIND_WHATSAPP:
        from captain_claw.flight_deck.whatsapp_bridge import push_to_waid
        sent = await push_to_waid(address, text)
        return sent, "ok" if sent else "skipped:muted-or-not-allowed"

    if kind == KIND_TELEGRAM:
        from captain_claw.flight_deck.meta_webhook_bridge import strip_markdown
        from captain_claw.flight_deck.telegram_out import send_telegram_multi
        sent, total = await send_telegram_multi([address], strip_markdown(text))
        return sent > 0, f"ok ({sent}/{total})" if sent else "error:telegram-0-delivered"

    if kind in (KIND_GLASSES, KIND_CHANNEL):
        from captain_claw.flight_deck.glasses_bridge import (
            _broadcast,
            _get_or_create_channel,
        )
        ch = await _get_or_create_channel(address)
        await _broadcast(ch, {
            "type": "agent",
            "text": text,
            "source": "async",
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        return True, "ok"

    if kind == KIND_WEB:
        # The result is already saved in the session; the web UI shows it when
        # the user returns. Nothing to push.
        return True, "noop:web"

    return False, f"error:unknown-kind:{kind}"


def _deliver_caller_ok(request: Request, user: dict | None) -> bool:
    """Only a trusted caller may push to the owner's channels.

    Legit callers are agents on localhost finishing a deferred/cron task, so
    accept a loopback request or the shared agent secret (X-Agent-Secret) — or
    an authenticated Flight Deck user. Previously this route ignored auth, so
    any caller could push text to the owner's WhatsApp/Telegram.
    """
    if user:
        return True
    provided = request.headers.get("X-Agent-Secret", "")
    if provided:
        try:
            from captain_claw.flight_deck.agent_secret import get_or_create_agent_secret
            if secrets.compare_digest(provided, get_or_create_agent_secret()):
                return True
        except Exception:
            pass
    if os.environ.get("FD_LOCKDOWN", "").lower() in ("true", "1", "yes"):
        return False
    client_host = request.client.host if request.client else ""
    return client_host in ("127.0.0.1", "::1", "localhost")


@router.post("/fd/deliver")
async def fd_deliver(request: Request, _user: dict | None = Depends(get_optional_user)):
    """Route an async agent result back to its origin. Called by agents (over
    localhost) when a deferred/cron task finishes. Destination safety is
    enforced downstream (WhatsApp allowlist + mute, configured TG recipients)."""
    if not _deliver_caller_ok(request, _user):
        raise HTTPException(status_code=403, detail="delivery requires loopback or X-Agent-Secret")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON")
    text = str((body or {}).get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    origin = (body or {}).get("origin") or {}
    delivered, note = await deliver_to_origin(origin, text)
    return {"ok": delivered, "note": note}
