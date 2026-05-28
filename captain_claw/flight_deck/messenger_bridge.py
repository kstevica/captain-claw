"""Facebook Messenger → channel-bus bridge.

Stop-gap for the Ray-Ban Display SDK gap: until Meta's wearables DAT exposes
mic/camera to third parties, this lets a whitelisted Messenger user act as
a remote input surface for the glasses HUD. Texts and photos flow into the
same per-channel bus that ``glasses_bridge.py`` already runs, so the agent
reply renders on the glasses *and* echoes back to the Messenger thread —
the user can read on whichever screen they happen to be looking at.

Architecture
------------
::

    Meta Cloud ── webhook POST ──▶ /messenger/webhook (this file)
                                    │
                                    ├─ HMAC verify (MESSENGER_APP_SECRET)
                                    ├─ PSID allow-list
                                    ├─ slash "/c <channel>" → rebind PSID
                                    ├─ photos: download → face_index.recognize()
                                    │   (broadcasts person-card to channel)
                                    └─ push "user" + image_path → agent WS
                                       (reuses glasses_bridge channel state)

                              Agent replies arrive on the channel bus
                              ─────────────────────────────────────────
                              callback_subscriber forwards each "agent"
                              event back to the PSID via Graph API
                              ▲
    glasses_view.html ──── WS ┘  (also receives the same payload)

Setup checklist (single-user, dev-mode app — no Meta review needed)
-------------------------------------------------------------------
1. Create a Facebook App at developers.facebook.com (Business → Messenger).
2. Generate a **Page Access Token** for your Page.
3. Note your **App Secret** (settings → basic) and pick a **Verify Token**
   string of your choosing.
4. Add yourself as a tester under "Roles" so dev-mode delivery works.
5. Subscribe the webhook callback URL: ``https://<your-tunnel>/messenger/webhook``
   with fields ``messages`` and ``messaging_postbacks``.
6. Get your own PSID (sent in any webhook event you trigger).
7. Set these env vars before starting Flight Deck::

      MESSENGER_PAGE_ACCESS_TOKEN=EAAG...
      MESSENGER_APP_SECRET=abc123...
      MESSENGER_VERIFY_TOKEN=any-string-you-pick
      MESSENGER_ALLOWED_PSIDS=1234567890,9876543210     # comma-sep PSIDs
      MESSENGER_DEFAULT_CHANNEL=lounge                  # initial channel
      MESSENGER_DEFAULT_AGENT_HOST=localhost            # optional, default localhost
      MESSENGER_DEFAULT_AGENT_PORT=8765                 # required: pick your agent's WS port
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from captain_claw.flight_deck import face_index
from captain_claw.flight_deck.glasses_bridge import (
    _GLASSES_SYSTEM_CONTEXT,
    _broadcast,
    _ensure_agent_binding,
    _get_or_create_channel,
)
from captain_claw.flight_deck.meta_webhook_bridge import (
    now_iso as _now_iso,
    register_channel_callback,
    verify_hub_challenge,
    verify_signature,
)

router = APIRouter()

# ── Config (env-driven, read on every request so a reload picks up changes) ──


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _allowed_psids() -> set[str]:
    raw = _env("MESSENGER_ALLOWED_PSIDS")
    if not raw:
        return set()
    return {p.strip() for p in raw.split(",") if p.strip()}


def _default_channel() -> str:
    return _env("MESSENGER_DEFAULT_CHANNEL") or "messenger"


def _default_agent() -> tuple[str, int]:
    host = _env("MESSENGER_DEFAULT_AGENT_HOST") or "localhost"
    try:
        port = int(_env("MESSENGER_DEFAULT_AGENT_PORT") or "0")
    except ValueError:
        port = 0
    return host, port


# ── Per-PSID state ────────────────────────────────────────────────────


# Channel each PSID is currently bound to. Defaults to MESSENGER_DEFAULT_CHANNEL;
# changeable via the ``/c <name>`` slash command. In-memory: rebinds reset on
# restart, which is fine — this is a hack, not a system of record.
_PSID_CHANNEL: dict[str, str] = {}

# Channels we've already wired a Messenger-forwarding callback into. Without
# this we'd register duplicate callbacks every time a Messenger user sends a
# second message to the same channel.
_WIRED_CHANNELS: set[str] = set()

# PSIDs we currently fan agent replies out to, per channel. The callback
# subscriber consults this so a /c rebind doesn't keep spraying old channels
# to the user's Messenger thread.
_CHANNEL_PSIDS: dict[str, set[str]] = {}


# ── Webhook verification (GET) ────────────────────────────────────────


@router.get("/messenger/webhook")
async def messenger_verify(request: Request) -> PlainTextResponse:
    """Meta's webhook handshake.

    On webhook subscribe, Meta sends ``GET /messenger/webhook?hub.mode=subscribe
    &hub.verify_token=<your token>&hub.challenge=<random>``. We must echo
    back ``hub.challenge`` verbatim *only* if the verify_token matches.
    """
    mode = request.query_params.get("hub.mode", "")
    token = request.query_params.get("hub.verify_token", "")
    challenge = request.query_params.get("hub.challenge", "")
    if verify_hub_challenge(mode, token, _env("MESSENGER_VERIFY_TOKEN")):
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="verify token mismatch")


# ── Webhook handler (POST) ────────────────────────────────────────────


@router.post("/messenger/webhook")
async def messenger_webhook(request: Request) -> JSONResponse:
    """Receive a webhook event from Meta and dispatch each message.

    Meta strongly recommends returning 200 within a couple of seconds — they
    retry otherwise, which would double-send. We acknowledge fast and spawn
    background tasks for the actual work.
    """
    body = await request.body()
    if not verify_signature(
        body,
        request.headers.get("x-hub-signature-256", ""),
        _env("MESSENGER_APP_SECRET"),
    ):
        raise HTTPException(status_code=401, detail="bad signature")

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"bad json: {exc}") from exc

    # Page-event payloads only — anything else is misconfiguration.
    if payload.get("object") != "page":
        return JSONResponse({"ok": True, "ignored": True})

    allowed = _allowed_psids()
    if not allowed:
        # An empty allowlist is treated as "deny all" — refusing to act on
        # any message is much safer than the default-allow alternative for
        # an internet-facing webhook.
        return JSONResponse({"ok": True, "ignored": "no allowlist"})

    for entry in payload.get("entry", []) or []:
        for evt in entry.get("messaging", []) or []:
            sender = (evt.get("sender") or {}).get("id", "")
            msg = evt.get("message") or {}
            if not sender or not msg or sender not in allowed:
                continue
            # Spawn — webhook ack must not wait for face inference / agent reply.
            asyncio.create_task(_handle_message(sender, msg))

    return JSONResponse({"ok": True})


# ── Inbound dispatch ──────────────────────────────────────────────────


async def _handle_message(psid: str, message: dict[str, Any]) -> None:
    """Process one Messenger message: route text/photos onto the channel bus
    and wake the bound agent."""
    text = str(message.get("text") or "").strip()
    attachments = message.get("attachments") or []

    # 1. Slash command first — never falls through to the agent.
    if text.startswith("/c "):
        new_ch = text[3:].strip()
        if new_ch:
            _rebind_psid(psid, new_ch)
            await _send_messenger_text(psid, f"Channel → {new_ch}")
        return
    if text == "/c":
        await _send_messenger_text(
            psid, f"Channel: {_PSID_CHANNEL.get(psid, _default_channel())}"
        )
        return

    channel = _PSID_CHANNEL.setdefault(psid, _default_channel())
    ch = await _get_or_create_channel(channel)
    _ensure_messenger_forwarding(ch.channel_id)
    _CHANNEL_PSIDS.setdefault(channel, set()).add(psid)

    # 2. Bind agent (default for this PSID — Messenger user doesn't pick).
    agent_host, agent_port = _default_agent()
    if not agent_port:
        await _send_messenger_text(
            psid, "Bridge offline: MESSENGER_DEFAULT_AGENT_PORT not configured."
        )
        return
    await _ensure_agent_binding(ch, agent_host, agent_port)

    # 3. Handle photo attachments: face recognition first (so the card lands
    #    on the channel before the agent reply), then forward to the agent.
    image_path: str | None = None
    for att in attachments:
        if att.get("type") != "image":
            continue
        url = ((att.get("payload") or {}).get("url") or "").strip()
        if not url:
            continue
        try:
            blob = await _download(url)
        except Exception as exc:
            await _send_messenger_text(psid, f"Couldn't fetch photo: {exc}")
            continue
        # Face recognition broadcasts a person card to the channel; the
        # messenger callback forwards it to the user's thread, and the
        # glasses view picks it up over WS. Identical content on both.
        try:
            await face_index.get_index().recognize(image_blob=blob, channel=channel)
        except RuntimeError:
            # ``[faces]`` extra not installed — skip silently.
            pass
        # Hand the blob to the agent for further analysis. We reuse the
        # same agent upload endpoint glasses_bridge already calls.
        try:
            image_path = await _forward_image_to_agent(agent_host, agent_port, blob)
        except Exception as exc:
            await _send_messenger_text(psid, f"Image forward failed: {exc}")
            image_path = None

    # 4. If neither text nor a successfully-forwarded image, nothing to do.
    if not text and not image_path:
        return

    # 5. Echo the user message to the channel so glasses see it instantly
    #    (mirrors what /glasses/send does). The Messenger thread already has
    #    a copy on the user's screen — we suppress that surface in the
    #    callback below so we don't echo it back to Messenger.
    user_event: dict[str, Any] = {
        "type": "user",
        "text": text,
        "ts": _now_iso(),
        "via": "messenger",
    }
    if image_path:
        user_event["image_path"] = image_path
    await _broadcast(ch, user_event)

    # 6. Send to the agent. Same shape as glasses_bridge's /glasses/send.
    for _ in range(50):  # up to ~5s for first-time bind
        if ch.agent_ws is not None:
            break
        await asyncio.sleep(0.1)
    if ch.agent_ws is None:
        await _send_messenger_text(psid, "Agent not ready, try again.")
        return

    async with ch.send_lock:
        effective_text = text or ("Please analyze this image." if image_path else text)
        if not ch.context_sent:
            agent_content = _GLASSES_SYSTEM_CONTEXT + effective_text
            ch.context_sent = True
        else:
            agent_content = effective_text
        payload_obj: dict[str, Any] = {"type": "chat", "content": agent_content}
        if image_path:
            payload_obj["image_path"] = image_path
        try:
            await ch.agent_ws.send(json.dumps(payload_obj))
        except Exception as exc:
            ch.context_sent = False
            await _send_messenger_text(psid, f"Send failed: {exc}")


def _rebind_psid(psid: str, new_channel: str) -> None:
    """Move a PSID's binding to a new channel. The old channel's set of
    forwarded PSIDs is cleaned up so its callback stops fanning to this user."""
    old = _PSID_CHANNEL.get(psid)
    if old and old in _CHANNEL_PSIDS:
        _CHANNEL_PSIDS[old].discard(psid)
    _PSID_CHANNEL[psid] = new_channel
    _CHANNEL_PSIDS.setdefault(new_channel, set()).add(psid)


# ── Channel-bus callback: fan agent replies out to Messenger ──────────


def _ensure_messenger_forwarding(channel_id: str) -> None:
    """Wire a Messenger Send-API forwarder onto the channel bus.

    Thin wrapper around the shared registrar — passes Messenger-specific
    state (``_WIRED_CHANNELS`` + ``_CHANNEL_PSIDS``) and the platform's
    send function. Cross-bridge fan-out (Messenger + WhatsApp on the same
    channel) works automatically because each bridge registers its own
    callback on the same channel's ``callback_subscribers`` list.
    """
    register_channel_callback(
        channel_id=channel_id,
        wired_set=_WIRED_CHANNELS,
        recipients_for_channel=lambda ch: _CHANNEL_PSIDS.get(ch, ()),
        send_one=_send_messenger_text,
    )


# ── Send API helpers ──────────────────────────────────────────────────


_MESSENGER_API = "https://graph.facebook.com/v18.0/me/messages"
# Messenger's per-message text limit is 2000 chars; we leave headroom for
# agent suffixes/punctuation and chunk at 1900 to be safe.
_MAX_CHUNK = 1900


async def _send_messenger_text(psid: str, text: str) -> None:
    token = _env("MESSENGER_PAGE_ACCESS_TOKEN")
    if not token:
        return  # No token → silently no-op (still appears on glasses HUD).
    text = text.strip()
    if not text:
        return

    chunks: list[str] = []
    while text:
        chunks.append(text[:_MAX_CHUNK])
        text = text[_MAX_CHUNK:]

    async with httpx.AsyncClient(timeout=20.0) as client:
        for chunk in chunks:
            await client.post(
                _MESSENGER_API,
                params={"access_token": token},
                json={
                    "recipient": {"id": psid},
                    "message": {"text": chunk},
                    "messaging_type": "RESPONSE",
                },
            )


async def _download(url: str) -> bytes:
    """Pull a Messenger CDN URL into memory.

    Messenger attachment URLs are temporary (signed, expire in ~minutes),
    so we fetch synchronously while we have the request. Capped at 10 MB
    via the content-length check — phone photos are ~2–8 MB.
    """
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        if len(r.content) > 10 * 1024 * 1024:
            raise ValueError("attachment > 10 MB, ignored")
        return r.content


async def _forward_image_to_agent(host: str, port: int, blob: bytes) -> str:
    """POST the image bytes to the agent's ``/api/image/upload``. Returns
    the absolute path on the agent's filesystem that ``/glasses/send``'s
    ``image_path`` field expects."""
    try:
        from captain_claw.flight_deck.server import _resolve_agent_auth
        auth = _resolve_agent_auth(port)
    except Exception:
        auth = ""
    params = {"token": auth} if auth else {}
    target = f"http://{host}:{port}/api/image/upload"

    # Use a random-ish filename so the agent's storage doesn't collide on
    # identical-name uploads from multiple chats.
    fname = f"messenger-{secrets.token_hex(6)}.jpg"
    files = {"file": (fname, blob, "image/jpeg")}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(target, files=files, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"agent upload returned {resp.status_code}: {resp.text}")
    data = resp.json()
    return str(data.get("path") or "")
