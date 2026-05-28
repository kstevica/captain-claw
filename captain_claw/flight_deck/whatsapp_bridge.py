"""WhatsApp Business Cloud API → channel-bus bridge.

Companion to ``messenger_bridge.py`` — same architectural shape (webhook in,
channel-bus echo, Send-API forwarding of agent replies), different transport.

Cross-bridge note
-----------------
WhatsApp and Messenger users with ``/c lounge`` (same channel name) land on
the **same channel** in ``glasses_bridge``. Each bridge attaches its own
callback subscriber to that channel, so a single agent reply fans out to
*every* platform recipient + the glasses HUD. This is intentional: lets
you keep one conversation across surfaces.

Architecture
------------
::

    WhatsApp user ── webhook POST ──▶ /whatsapp/webhook
                                       │
                                       ├─ HMAC verify (WHATSAPP_APP_SECRET)
                                       ├─ WAID (phone-number) allow-list
                                       ├─ /c <channel> rebind
                                       ├─ photos: media_id → 2-step fetch
                                       │   → face_index.recognize()
                                       │   → forward to agent /api/image/upload
                                       └─ user event → channel bus
                                                            │
                            agent reply on channel bus ─────┤
                              callback fan-out:             │
                                → WhatsApp Send API ────────┘
                                → glasses_view (over WS)
                                → Messenger (if any PSIDs on this channel)

Setup checklist (Cloud API "test number" tier — free, no business verification)
-------------------------------------------------------------------------------
1. In your existing Meta App (the one you set up for Messenger), enable the
   **WhatsApp** product. Meta will provision a free test sender number.
2. Under WhatsApp → API setup, add your personal phone to the **recipient
   allowlist** (up to 5 numbers in test mode). Verify each via SMS.
3. Copy the **temporary access token** and **Phone number ID** Meta shows
   you. Token rotates every 24 h on the test tier — generate a **System
   User permanent token** when you want stable.
4. Settings → Basic → reuse the **App Secret** (same as Messenger).
5. Subscribe the webhook callback URL: ``https://<your-tunnel>/whatsapp/webhook``
   with field ``messages``. Use the same verify-token string convention
   as Messenger or pick a separate one.
6. Set env vars before starting Flight Deck::

      WHATSAPP_PHONE_NUMBER_ID=1234567890     # the sender's numeric ID
      WHATSAPP_ACCESS_TOKEN=EAAG...            # temporary or System User token
      WHATSAPP_APP_SECRET=abc123...            # usually same as Messenger
      WHATSAPP_VERIFY_TOKEN=any-string         # any string, also configured
                                                # on Meta's side
      WHATSAPP_ALLOWED_WAIDS=31612345678,1234567890  # phone numbers, no '+'
      WHATSAPP_DEFAULT_CHANNEL=lounge          # initial channel
      WHATSAPP_DEFAULT_AGENT_SLUG=personal     # FD process slug (preferred —
                                                # survives port reassignment)
      WHATSAPP_DEFAULT_AGENT_PORT=8765         # legacy / fixed-port fallback
      # If neither set, the bridge auto-binds to the first alive FD agent.
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
    resolve_agent_target,
    verify_hub_challenge,
    verify_signature,
)

router = APIRouter()


# ── Config (env-driven, re-read on every request) ────────────────────


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _allowed_waids() -> set[str]:
    raw = _env("WHATSAPP_ALLOWED_WAIDS")
    if not raw:
        return set()
    # WAIDs are phone numbers without the leading "+". Strip any the user
    # might have typed anyway so config is forgiving.
    return {p.strip().lstrip("+") for p in raw.split(",") if p.strip()}


def _default_channel() -> str:
    return _env("WHATSAPP_DEFAULT_CHANNEL") or "whatsapp"


def _default_agent() -> tuple[str, int, str]:
    """Resolve the target agent fresh on every call.

    Prefers ``WHATSAPP_DEFAULT_AGENT_SLUG`` (looked up in Flight Deck's
    process registry — survives FD restarts that reassign web ports).
    Falls back to ``WHATSAPP_DEFAULT_AGENT_PORT`` for legacy / out-of-FD
    setups, then to "first alive agent" for single-agent boxes.

    Returns ``(host, port, auth)``. ``auth`` is the env-supplied override
    (``WHATSAPP_DEFAULT_AGENT_AUTH``); empty string when not set, in which
    case the bridge falls back to FD's registry-based lookup.
    """
    return resolve_agent_target(
        slug_env="WHATSAPP_DEFAULT_AGENT_SLUG",
        port_env="WHATSAPP_DEFAULT_AGENT_PORT",
        auth_env="WHATSAPP_DEFAULT_AGENT_AUTH",
        host_env="WHATSAPP_DEFAULT_AGENT_HOST",
    )


# ── Per-WAID state (mirrors messenger_bridge for cross-bridge symmetry) ──


# WAID → channel. Defaults to WHATSAPP_DEFAULT_CHANNEL; changeable via the
# ``/c <name>`` slash command. In-memory, resets on restart.
_WAID_CHANNEL: dict[str, str] = {}

# Channels we've already wired a WhatsApp-forwarding callback into. Disjoint
# from the Messenger bridge's ``_WIRED_CHANNELS`` — both can co-exist on the
# same channel id because each bridge installs its own callback.
_WIRED_CHANNELS: set[str] = set()

# Channel → set of WAIDs currently bound to it. Consulted at delivery time
# by the per-channel callback, so ``/c`` rebinds take effect immediately.
_CHANNEL_WAIDS: dict[str, set[str]] = {}


# ── Webhook verification (GET) ────────────────────────────────────────


@router.get("/whatsapp/webhook")
async def whatsapp_verify(request: Request) -> PlainTextResponse:
    """Meta's webhook handshake. Same shape as Messenger's."""
    mode = request.query_params.get("hub.mode", "")
    token = request.query_params.get("hub.verify_token", "")
    challenge = request.query_params.get("hub.challenge", "")
    if verify_hub_challenge(mode, token, _env("WHATSAPP_VERIFY_TOKEN")):
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="verify token mismatch")


# ── Webhook handler (POST) ────────────────────────────────────────────


@router.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request) -> JSONResponse:
    """Receive a webhook event from Meta and dispatch each message.

    Meta retries if we don't ack within ~5 s, so we verify-and-spawn:
    HMAC check + payload parse synchronously, then ``asyncio.create_task``
    each message handler. Long work (face inference, agent round-trip,
    media download) happens in the background.
    """
    body = await request.body()
    if not verify_signature(
        body,
        request.headers.get("x-hub-signature-256", ""),
        _env("WHATSAPP_APP_SECRET"),
    ):
        raise HTTPException(status_code=401, detail="bad signature")

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"bad json: {exc}") from exc

    # WhatsApp uses ``object: "whatsapp_business_account"`` — anything else
    # arriving here is misconfiguration on Meta's side.
    if payload.get("object") != "whatsapp_business_account":
        return JSONResponse({"ok": True, "ignored": True})

    allowed = _allowed_waids()
    if not allowed:
        # Fail-safe: empty allowlist = deny all. Refusing to process is
        # always safer than the default-allow alternative on an open
        # webhook endpoint.
        return JSONResponse({"ok": True, "ignored": "no allowlist"})

    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value") or {}
            # Status events (delivered / read / failed) arrive here too —
            # not actionable for us, ignore.
            messages = value.get("messages") or []
            for msg in messages:
                waid = str(msg.get("from") or "").lstrip("+")
                if not waid or waid not in allowed:
                    continue
                asyncio.create_task(_handle_message(waid, msg))

    return JSONResponse({"ok": True})


# ── Inbound dispatch ──────────────────────────────────────────────────


async def _handle_message(waid: str, message: dict[str, Any]) -> None:
    """Process one inbound WhatsApp message.

    Routes text/photos onto the channel bus, then wakes the bound agent.
    Voice notes and other unsupported types are dropped silently for v1
    (the user already has a perfect transcription path via the WhatsApp
    Display app for voice notes they *receive*).
    """
    mtype = str(message.get("type") or "")
    text = ""
    if mtype == "text":
        text = str((message.get("text") or {}).get("body") or "").strip()

    # Acknowledge receipt visually as soon as possible. The "typing…" stays
    # until the agent's reply lands (or ~25 s). Background task so it can't
    # delay the rest of the handler.
    inbound_message_id = str(message.get("id") or "").strip()
    if inbound_message_id:
        asyncio.create_task(_mark_read_and_typing(inbound_message_id))

    # 1. Slash command first — never falls through to the agent.
    if text.startswith("/c "):
        new_ch = text[3:].strip()
        if new_ch:
            _rebind_waid(waid, new_ch)
            await _send_whatsapp_text(waid, f"Channel → {new_ch}")
        return
    if text == "/c":
        await _send_whatsapp_text(
            waid, f"Channel: {_WAID_CHANNEL.get(waid, _default_channel())}"
        )
        return

    channel = _WAID_CHANNEL.setdefault(waid, _default_channel())
    ch = await _get_or_create_channel(channel)
    _ensure_whatsapp_forwarding(ch.channel_id)
    _CHANNEL_WAIDS.setdefault(channel, set()).add(waid)

    # 2. Bind agent. Same fixed-target rule as messenger_bridge: env-var
    #    picks a single agent per platform; WhatsApp users don't pick.
    agent_host, agent_port, agent_auth = _default_agent()
    if not agent_port:
        await _send_whatsapp_text(
            waid,
            "Bridge offline: no agent available. Set WHATSAPP_DEFAULT_AGENT_SLUG "
            "(preferred) or WHATSAPP_DEFAULT_AGENT_PORT, or make sure at least "
            "one Flight Deck agent is running.",
        )
        return
    await _ensure_agent_binding(ch, agent_host, agent_port, agent_auth)

    # 3. Photo? Two-step media fetch, then face recognition + agent upload.
    image_path: str | None = None
    if mtype == "image":
        media_id = str((message.get("image") or {}).get("id") or "")
        caption = str((message.get("image") or {}).get("caption") or "").strip()
        if caption and not text:
            # WhatsApp delivers image captions separately from text; treat
            # them as the user's actual prompt for this message.
            text = caption
        if media_id:
            try:
                blob = await _download_media(media_id)
            except Exception as exc:
                await _send_whatsapp_text(waid, f"Couldn't fetch photo: {exc}")
                blob = None
            if blob is not None:
                # Face card auto-broadcasts to the channel; both the glasses
                # HUD and the user's WhatsApp thread receive it.
                try:
                    await face_index.get_index().recognize(
                        image_blob=blob, channel=channel
                    )
                except RuntimeError:
                    # ``[faces]`` extra not installed — skip silently.
                    pass
                try:
                    image_path = await _forward_image_to_agent(
                        agent_host, agent_port, blob
                    )
                except Exception as exc:
                    await _send_whatsapp_text(
                        waid, f"Image forward failed: {exc}"
                    )
                    image_path = None

    # 4. If neither text nor a successfully-forwarded image, nothing to do.
    if not text and not image_path:
        return

    # 5. Echo user message to the channel — gives the glasses an instant
    #    "user said X" line. The WhatsApp thread already shows the original
    #    on the user's screen; our callback below skips ``user`` events.
    user_event: dict[str, Any] = {
        "type": "user",
        "text": text,
        "ts": _now_iso(),
        "via": "whatsapp",
    }
    if image_path:
        user_event["image_path"] = image_path
    await _broadcast(ch, user_event)

    # 6. Send to the agent. Identical shape to /glasses/send / messenger_bridge.
    for _ in range(50):  # up to ~5s
        if ch.agent_ws is not None:
            break
        await asyncio.sleep(0.1)
    if ch.agent_ws is None:
        await _send_whatsapp_text(waid, "Agent not ready, try again.")
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
            await _send_whatsapp_text(waid, f"Send failed: {exc}")


def _rebind_waid(waid: str, new_channel: str) -> None:
    """Move a WAID's binding to a new channel. Cleans up the old channel's
    recipient set so its callback stops fanning to this number."""
    old = _WAID_CHANNEL.get(waid)
    if old and old in _CHANNEL_WAIDS:
        _CHANNEL_WAIDS[old].discard(waid)
    _WAID_CHANNEL[waid] = new_channel
    _CHANNEL_WAIDS.setdefault(new_channel, set()).add(waid)


# ── Channel-bus callback ──────────────────────────────────────────────


def _ensure_whatsapp_forwarding(channel_id: str) -> None:
    """Wire a WhatsApp Send-API forwarder onto the channel bus.

    Independent of any Messenger callback registered on the same channel —
    both can co-exist (cross-bridge fan-out is intentional).
    """
    register_channel_callback(
        channel_id=channel_id,
        wired_set=_WIRED_CHANNELS,
        recipients_for_channel=lambda ch: _CHANNEL_WAIDS.get(ch, ()),
        send_one=_send_whatsapp_text,
    )


# ── Cloud API: send text ──────────────────────────────────────────────


# Cloud API text limit is 4096 chars; we chunk at 3500 to leave headroom
# for agent-side punctuation surprises.
_MAX_CHUNK = 3500


def _send_url() -> str:
    pid = _env("WHATSAPP_PHONE_NUMBER_ID")
    return f"https://graph.facebook.com/v18.0/{pid}/messages" if pid else ""


async def _mark_read_and_typing(message_id: str) -> None:
    """Mark an inbound WhatsApp message as read AND show the typing indicator.

    The Cloud API exposes both via the same POST to ``/<phone-id>/messages``
    when ``status: "read"`` is paired with ``typing_indicator: {type: "text"}``.
    Effect on the user's chat:

      * Blue double-tick on the message they just sent (read receipt)
      * "typing…" appears under the business name in the header

    The typing indicator auto-clears the moment we send the agent's reply
    (or after ~25 s of inactivity). Fire-and-forget — best-effort UX, never
    blocks the main message flow.
    """
    token = _env("WHATSAPP_ACCESS_TOKEN")
    url = _send_url()
    if not token or not url or not message_id:
        return
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {"type": "text"},
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(url, headers=headers, json=payload)
    except Exception:
        # Pure UX nicety; if Meta returns an error or the network is flaky,
        # the conversation still works — just no read tick / typing dots.
        pass


async def _send_whatsapp_text(waid: str, text: str) -> None:
    """POST a text message to the Cloud API. No-op if config is missing —
    the glasses HUD will still show the agent reply via the channel bus."""
    token = _env("WHATSAPP_ACCESS_TOKEN")
    url = _send_url()
    if not token or not url:
        return
    text = text.strip()
    if not text:
        return

    chunks: list[str] = []
    s = text
    while s:
        chunks.append(s[:_MAX_CHUNK])
        s = s[_MAX_CHUNK:]

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        for chunk in chunks:
            await client.post(
                url,
                headers=headers,
                json={
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": waid,
                    "type": "text",
                    "text": {"body": chunk, "preview_url": False},
                },
            )


# ── Cloud API: media download (2-step) ────────────────────────────────


async def _download_media(media_id: str) -> bytes:
    """Resolve a Cloud API media_id to bytes.

    Two-step dance:
      1. ``GET /v18.0/<media_id>`` with Bearer auth → JSON containing the
         actual CDN ``url``.
      2. ``GET <url>`` *also* with Bearer auth → bytes.

    Note the second GET — unlike Messenger, the Cloud API's media URLs
    require authentication even after the first hop. The token must be
    sent on both requests. Cap at 10 MB.
    """
    token = _env("WHATSAPP_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("WHATSAPP_ACCESS_TOKEN not configured")

    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        meta_resp = await client.get(
            f"https://graph.facebook.com/v18.0/{media_id}",
            headers=headers,
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        url = str(meta.get("url") or "")
        if not url:
            raise RuntimeError("media metadata missing 'url'")

        media_resp = await client.get(url, headers=headers)
        media_resp.raise_for_status()
        blob = media_resp.content
        if len(blob) > 10 * 1024 * 1024:
            raise ValueError("media > 10 MB, ignored")
        return blob


# ── Agent image upload (same shape as messenger_bridge) ──────────────


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

    fname = f"whatsapp-{secrets.token_hex(6)}.jpg"
    files = {"file": (fname, blob, "image/jpeg")}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(target, files=files, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"agent upload returned {resp.status_code}: {resp.text}")
    data = resp.json()
    return str(data.get("path") or "")
