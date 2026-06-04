"""WhatsApp Business Cloud API → channel-bus bridge.

By default, WhatsApp is **standalone**: each WAID gets its own private
channel (``whatsapp:<waid>``) so user messages don't echo to the glasses
HUD. Photos handled here bypass the agent entirely — face recognition
runs and the card lands directly back in the user's WhatsApp thread.

Cross-bridge fan-out (the same agent reply landing on WhatsApp + glasses
HUD + Messenger simultaneously) is opt-in:

* Set ``WHATSAPP_DEFAULT_CHANNEL=lounge`` to share by default, or
* Have the WhatsApp user send ``/c lounge`` to rebind at runtime.

When sharing, each bridge attaches its own callback subscriber to the
channel; a single agent reply fans out to every platform recipient.

Optional voice reply
--------------------
When ``WHATSAPP_AUDIO_REPLY=on``, every substantive bridge reply (agent
answer, face card) is also synthesized to MP3 via Soniox TTS and sent
as a WhatsApp audio message after the text. Slash-command replies and
error messages stay text-only.

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

      # Channel binding (private per-user by default — leave unset to
      # keep WhatsApp standalone; set to share with glasses/Messenger).
      # WHATSAPP_DEFAULT_CHANNEL=lounge

      # Agent target (slug strongly preferred — survives FD restarts that
      # reassign web ports).
      WHATSAPP_DEFAULT_AGENT_SLUG=personal
      WHATSAPP_DEFAULT_AGENT_AUTH=tAz6q…       # from agent's config.yaml
                                                # web.auth_token (only needed
                                                # when agent isn't in FD's
                                                # process/Docker registry)

      # Optional MP3 voice reply via Soniox (needs SONIOX_API_KEY).
      # WHATSAPP_AUDIO_REPLY=on
      # WHATSAPP_AUDIO_VOICE=Adrian            # default falls back to
      # WHATSAPP_AUDIO_LANGUAGE=en             # SONIOX_TTS_VOICE / *_LANGUAGE
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from captain_claw.flight_deck import face_index
from captain_claw.flight_deck.glasses_bridge import (
    _GLASSES_SYSTEM_CONTEXT,
    _NO_CACHE,
    _broadcast,
    _check_token,
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


def _channel_for_waid(waid: str) -> str:
    """Resolve the channel a fresh WAID lands on.

    * ``WHATSAPP_DEFAULT_CHANNEL`` set → that value (shared mode; same
      channel id can be opened by the glasses HUD or used by another
      bridge, e.g. Messenger, for cross-surface fan-out).
    * Empty/unset → ``whatsapp:<waid>`` — a **per-user private channel**.
      Nothing else subscribes to this by default, so the conversation
      stays inside WhatsApp.

    The slash command ``/c <name>`` lets a user override at runtime.
    """
    env_default = _env("WHATSAPP_DEFAULT_CHANNEL")
    if env_default:
        return env_default
    return f"whatsapp:{waid}"


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

# Inbound image awaiting an instruction. When a photo arrives with no
# caption we ask the user what to do and stash the bytes here; their next
# message routes it (identify / describe / enroll). In-memory, TTL-gated.
_PENDING_IMAGE: dict[str, dict[str, Any]] = {}
_PENDING_IMAGE_TTL = 300.0  # seconds

# Caption intent matchers for inbound images. Face recognition stays a
# Flight Deck capability (face_index) — these just decide which FD path to
# run; anything else is forwarded to the agent for vision.
# Stems intentionally lack a trailing boundary so "identif" matches
# "identify", "recogn" matches "recognise"/"recognize", etc.
_IDENTIFY_RE = re.compile(
    r"(?i)\b(who(?:'s| is| are)?|whose|identif|recogn|tko\s+(?:je|su)|prepoznaj)"
)
_ENROLL_LEAD_RE = re.compile(
    r"(?i)^(?:please\s+)?(?:remember|save|enroll|zapamti|upamti)"
    r"(?:\s+(?:this|that|this\s+person|the\s+face|ovu\s+osobu|ovo|to))?"
    r"\s*(?:is|as|je|kao|[:\-])?\s*(?P<rest>.+)$"
)
_ENROLL_OVO_RE = re.compile(r"(?i)^(?:ovo|to)\s+je\s+(?P<rest>.+)$")


def _parse_enroll(caption: str) -> tuple[str, str] | None:
    """If *caption* is an enroll request, return (name, notes); else None."""
    c = (caption or "").strip()
    m = _ENROLL_LEAD_RE.match(c) or _ENROLL_OVO_RE.match(c)
    if not m:
        return None
    rest = (m.group("rest") or "").strip(" :,-")
    if not rest:
        return None
    name, _, notes = rest.partition(",")
    return name.strip(), notes.strip()


# Last-seen inbound message id per WAID. WhatsApp's typing-indicator API
# requires the wamid of a *real* user message — there's no "show typing"
# call without one — so we cache the most recent one to re-fire the
# indicator after sending intermediate status text (e.g. "Generating
# audio…"). Cleared implicitly on FD restart; no persistence needed.
_WAID_LAST_MESSAGE_ID: dict[str, str] = {}

# Per-WAID proactive-push mute. Maps WAID → epoch seconds until which
# pushes are suppressed (math.inf = muted indefinitely). Set via the
# ``/mute [duration]`` slash command, cleared via ``/unmute``. Mute ONLY
# affects proactive pushes (the FD scheduler and the /whatsapp/push
# endpoint) — direct replies to a message the user just sent always go
# through, so muting never makes the bot feel broken in active use.
_MUTED_UNTIL: dict[str, float] = {}


def is_push_muted(waid: str) -> bool:
    """Whether proactive pushes to this WAID are currently suppressed."""
    import time as _time
    until = _MUTED_UNTIL.get(waid)
    if until is None:
        return False
    if until == float("inf"):
        return True
    if _time.time() < until:
        return True
    # Expired — clean up so the dict doesn't grow unbounded.
    _MUTED_UNTIL.pop(waid, None)
    return False


def _parse_duration_seconds(text: str) -> float | None:
    """Parse ``30m`` / ``2h`` / ``1d`` → seconds. None if unparseable/empty."""
    import re as _re
    m = _re.fullmatch(r"\s*(\d+)\s*([mhd])\s*", text or "", _re.I)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    return n * {"m": 60, "h": 3600, "d": 86400}[unit]


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
        # Cache for later — the audio-reply path re-fires the typing
        # indicator after sending intermediate status text, and the API
        # needs a real wamid to reference.
        _WAID_LAST_MESSAGE_ID[waid] = inbound_message_id
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
            waid, f"Channel: {_WAID_CHANNEL.get(waid, _channel_for_waid(waid))}"
        )
        return
    if text.startswith("/mute"):
        # "/mute" → forever; "/mute 2h" → until now+2h.
        arg = text[len("/mute"):].strip()
        if arg:
            secs = _parse_duration_seconds(arg)
            if secs is None:
                await _send_whatsapp_text(
                    waid, "Usage: /mute  or  /mute 30m | 2h | 1d"
                )
                return
            import time as _t
            _MUTED_UNTIL[waid] = _t.time() + secs
            await _send_whatsapp_text(waid, f"🔕 Proactive pushes muted for {arg}.")
        else:
            _MUTED_UNTIL[waid] = float("inf")
            await _send_whatsapp_text(
                waid, "🔕 Proactive pushes muted. Send /unmute to resume."
            )
        return
    if text == "/unmute":
        _MUTED_UNTIL.pop(waid, None)
        await _send_whatsapp_text(waid, "🔔 Proactive pushes resumed.")
        return

    channel = _WAID_CHANNEL.setdefault(waid, _channel_for_waid(waid))
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

    # 3. Multi-type dispatch. Images terminate here (face-only); the other
    #    non-text types translate to text (formatted FYI for location /
    #    contacts, transcription for voice notes) and fall through to the
    #    shared agent send at the bottom.
    if mtype == "image":
        img = message.get("image") or {}
        media_id = str(img.get("id") or "")
        if not media_id:
            return
        caption = str(img.get("caption") or "").strip()
        try:
            blob = await _download_media(media_id)
        except Exception as exc:
            await _send_whatsapp_text(waid, f"Couldn't fetch photo: {exc}")
            return
        # Flow override: an enabled image Flow takes precedence over the built-in
        # identify/describe/enroll automation. Match FIRST (cheap, no upload); only
        # if a Flow matches do we upload the photo and run it. No match → the
        # built-in below runs unchanged.
        try:
            from captain_claw.flight_deck import flow_router
            if flow_router.engine_ready():
                _fp = flow_router.classify_payload(
                    channel="whatsapp", text=caption, waid=waid,
                    origin_host=agent_host, origin_port=int(agent_port or 0),
                    extra={"has_image": True},
                )
                _flow = await flow_router.match_flow(_fp)
                if _flow is not None:
                    _fp["image_path"] = await _upload_image_to_agent(blob, agent_host, agent_port, agent_auth)
                    await flow_router.run_flow(_flow, _fp)
                    return
        except Exception as _exc:
            log.warning("image flow override check failed: %s", _exc)
        if caption:
            # Caption present → route immediately (identify / enroll / vision).
            await _route_image(
                waid, blob, caption, ch, agent_host, agent_port, agent_auth
            )
        else:
            # Bare photo → ask what to do, stash bytes for the follow-up reply.
            _PENDING_IMAGE[waid] = {"blob": blob, "ts": time.time()}
            await _send_whatsapp_text(
                waid,
                "📷 Got the photo. What should I do with it?\n"
                "• \"who is this?\" — identify the people\n"
                "• \"describe it\" / \"read this\" — analyse the image\n"
                "• \"remember this is <name>\" — save the face",
            )
        return

    if mtype == "video":
        vid = message.get("video") or {}
        media_id = str(vid.get("id") or "")
        if not media_id:
            return
        caption = str(vid.get("caption") or "").strip()
        mime = str(vid.get("mime_type") or "video/mp4")
        await _send_whatsapp_text(
            waid, "🎬 Got the video — analyzing it (frames + audio, ~a couple of minutes)…",
            mirror=True,
        )
        if inbound_message_id:
            asyncio.create_task(_mark_read_and_typing(inbound_message_id))
        try:
            blob = await _download_media(media_id)
        except Exception as exc:
            await _send_whatsapp_text(waid, f"Couldn't fetch the video: {exc}")
            return
        await _forward_video_to_agent(
            waid, blob, caption, mime, ch, agent_host, agent_port, agent_auth
        )
        return

    if mtype == "location":
        # Build the FYI text and let the standard flow forward it to the agent.
        loc = message.get("location") or {}
        text = _format_location_as_text(loc)

    elif mtype == "contacts":
        text = _format_contacts_as_text(message.get("contacts") or [])

    elif mtype == "audio":
        audio = message.get("audio") or {}
        media_id = str(audio.get("id") or "")
        if not media_id:
            return
        mime = str(audio.get("mime_type") or "audio/ogg")
        # Tell the user we're working on it — voice notes can take a few
        # seconds end-to-end (download + Soniox upload + transcribe + poll).
        await _send_whatsapp_text(waid, "🎙 Transcribing voice note…", mirror=True)
        if inbound_message_id:
            # Re-fire the typing indicator; the status text we just sent
            # cleared the initial one.
            asyncio.create_task(_mark_read_and_typing(inbound_message_id))
        try:
            blob = await _download_media(media_id)
        except Exception as exc:
            await _send_whatsapp_text(waid, f"Couldn't fetch voice note: {exc}", mirror=True)
            return
        transcript = await _transcribe_soniox(blob, mime)
        if not transcript:
            await _send_whatsapp_text(
                waid,
                "Couldn't transcribe that — Soniox returned no text. "
                "Try sending the message again, or speak more clearly.",
                mirror=True,
            )
            return
        # Send the transcript back to the user clearly marked so they know
        # this is what the agent is seeing.
        await _send_whatsapp_text(waid, f"🎙 Transcription:\n\n\"{transcript}\"", mirror=True)
        # Forward to the agent as if the user had typed it.
        text = transcript

    # 4. Text only. If empty, nothing to do.
    if not text:
        return

    # 4b. Pending-image follow-up: if we asked what to do with a bare photo,
    #     this message is the instruction — route the stashed image with it.
    pending = _PENDING_IMAGE.pop(waid, None)
    if pending is not None:
        if (time.time() - pending.get("ts", 0.0)) <= _PENDING_IMAGE_TTL:
            await _route_image(
                waid, pending["blob"], text, ch, agent_host, agent_port, agent_auth
            )
            return
        # else: stale — fall through and treat as a normal message.

    # 4c. Flow engine: if an enabled text-triggered flow matches, run it and
    #     stop here. No-op (falls through to the normal agent forward) when no
    #     flow matches — so this is inert until the user enables a text flow.
    try:
        from captain_claw.flight_deck import flow_router
        if flow_router.engine_ready():
            _fp = flow_router.classify_payload(
                channel="whatsapp", text=text, waid=waid,
                origin_host=agent_host, origin_port=int(agent_port or 0),
            )
            if await flow_router.try_match_and_run(_fp):
                return
    except Exception as _exc:
        log.warning("flow trigger check failed: %s", _exc)

    # 5. Mirror the user's message onto the channel bus so the glasses HUD
    #    shows what arrived over WhatsApp (matches mobile + messenger). The
    #    ``via`` tag lets the view badge the source. This does NOT echo back
    #    to WhatsApp: the forwarding callback only relays ``agent``/``error``
    #    events, never ``user`` ones (see meta_webhook_bridge._forward).
    await _broadcast(ch, {
        "type": "user",
        "text": text,
        "ts": _now_iso(),
        "via": "whatsapp",
    })

    # 6. Send to the agent. The agent's reply flows back through the channel
    #    (that's how _agent_pump delivers it), and the bridge's callback
    #    forwards it to the WhatsApp thread.
    for _ in range(50):  # up to ~5s
        if ch.agent_ws is not None:
            break
        await asyncio.sleep(0.1)
    if ch.agent_ws is None:
        await _send_whatsapp_text(waid, "Agent not ready, try again.")
        return

    async with ch.send_lock:
        if not ch.context_sent:
            agent_content = _GLASSES_SYSTEM_CONTEXT + text
            ch.context_sent = True
        else:
            agent_content = text
        # Tag the message with the originating WAID so the agent can target
        # "the current WhatsApp chat" (e.g. whatsapp_send_file with no 'to').
        payload_obj: dict[str, Any] = {
            "type": "chat",
            "content": agent_content,
            "whatsapp_waid": waid,
        }
        try:
            await ch.agent_ws.send(json.dumps(payload_obj))
        except Exception as exc:
            ch.context_sent = False
            await _send_whatsapp_text(waid, f"Send failed: {exc}")


# ── Non-text inbound formatters ──────────────────────────────────────


def _format_location_as_text(loc: dict[str, Any]) -> str:
    """Render a WhatsApp location payload as agent-friendly text.

    Schema (from Cloud API webhook):
      ``{latitude, longitude, name?, address?}``

    Prefixes the result with ``[FYI: …]`` so the agent treats it as
    context, not as a question requiring an answer (the system prompt
    sets the tone; the agent picks the angle).
    """
    lat = loc.get("latitude")
    lng = loc.get("longitude")
    name = str(loc.get("name") or "").strip()
    address = str(loc.get("address") or "").strip()
    lines: list[str] = ["[FYI: user shared a location]"]
    if name:
        lines.append(f"📍 {name}")
    if address:
        lines.append(f"Address: {address}")
    if lat is not None and lng is not None:
        lines.append(f"Coordinates: {lat}, {lng}")
        lines.append(f"Map: https://www.google.com/maps?q={lat},{lng}")
    return "\n".join(lines)


def _format_contacts_as_text(contacts: list[dict[str, Any]]) -> str:
    """Render a WhatsApp contacts payload as agent-friendly text.

    The Cloud API delivers contacts as a list (a single share can
    contain several vCards) with ``name``, ``phones``, ``emails`` and
    optionally ``addresses``, ``urls``. We surface the fields a casual
    "FYI" most likely needs and skip the rest.
    """
    lines: list[str] = ["[FYI: user shared a contact]"]
    for c in contacts or []:
        name = str((c.get("name") or {}).get("formatted_name") or "").strip()
        if name:
            lines.append(f"👤 {name}")
        for phone in c.get("phones") or []:
            num = str(phone.get("phone") or "").strip()
            ptype = str(phone.get("type") or "").strip().lower()
            if num:
                lines.append(f"📞 {num}" + (f" ({ptype})" if ptype else ""))
        for email in c.get("emails") or []:
            addr = str(email.get("email") or "").strip()
            etype = str(email.get("type") or "").strip().lower()
            if addr:
                lines.append(f"✉️ {addr}" + (f" ({etype})" if etype else ""))
        for org in [c.get("org") or {}]:
            company = str(org.get("company") or "").strip()
            title = str(org.get("title") or "").strip()
            if company or title:
                lines.append("🏢 " + (f"{title}, {company}" if title and company else (title or company)))
    return "\n".join(lines)


# ── Soniox STT (async REST) ──────────────────────────────────────────


# Soniox async transcription is a 3-step REST dance:
#   1. POST /v1/files               — upload audio, get file_id
#   2. POST /v1/transcriptions      — create job referencing file_id
#   3. GET  /v1/transcriptions/{id} — poll until status=completed
#   4. GET  /v1/transcriptions/{id}/transcript — fetch the text
# Plus DELETEs on both file and transcription to keep the user's Soniox
# storage clean. Source of the schema:
#   https://github.com/soniox/soniox_examples/blob/master/speech_to_text/python/soniox_async.py
_SONIOX_API_BASE = "https://api.soniox.com"
_SONIOX_STT_MODEL = "stt-async-v4"
_SONIOX_STT_POLL_MAX = 60  # 60 × 1 s = up to 60 s per WhatsApp voice note


async def _transcribe_soniox(audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
    """Transcribe audio bytes via Soniox async REST. Empty string on any
    failure — caller decides how to communicate that to the user.

    Language hints default to ``WHATSAPP_AUDIO_LANGUAGE`` / ``SONIOX_TTS_LANGUAGE``
    (single language). For multi-language users, set
    ``WHATSAPP_STT_LANGUAGES=en,es,hr`` to bias the model.
    """
    api_key = os.environ.get("SONIOX_API_KEY", "").strip()
    if not api_key or not audio_bytes:
        return ""

    headers = {"Authorization": f"Bearer {api_key}"}

    # Language hints: comma-sep override, else fall back to TTS language env.
    raw_hints = _env("WHATSAPP_STT_LANGUAGES")
    if raw_hints:
        language_hints = [h.strip() for h in raw_hints.split(",") if h.strip()]
    else:
        lang = (
            _env("WHATSAPP_AUDIO_LANGUAGE")
            or os.environ.get("SONIOX_TTS_LANGUAGE", "").strip()
            or "en"
        )
        language_hints = [lang]

    file_id = ""
    transcription_id = ""
    transcript_text = ""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # 1. Upload file
            try:
                up = await client.post(
                    f"{_SONIOX_API_BASE}/v1/files",
                    headers=headers,
                    files={"file": ("audio", audio_bytes, mime_type or "audio/ogg")},
                )
                up.raise_for_status()
                file_id = str(up.json().get("id") or "")
                if not file_id:
                    return ""
            except Exception:
                return ""

            # 2. Create transcription
            try:
                cr = await client.post(
                    f"{_SONIOX_API_BASE}/v1/transcriptions",
                    headers={**headers, "Content-Type": "application/json"},
                    json={
                        "model": _SONIOX_STT_MODEL,
                        "file_id": file_id,
                        "language_hints": language_hints,
                        "enable_language_identification": True,
                    },
                )
                cr.raise_for_status()
                transcription_id = str(cr.json().get("id") or "")
                if not transcription_id:
                    return ""
            except Exception:
                return ""

            # 3. Poll for completion (Soniox example uses 1 s interval)
            for _ in range(_SONIOX_STT_POLL_MAX):
                try:
                    p = await client.get(
                        f"{_SONIOX_API_BASE}/v1/transcriptions/{transcription_id}",
                        headers=headers,
                    )
                    p.raise_for_status()
                    status = str(p.json().get("status") or "")
                except Exception:
                    break
                if status == "completed":
                    break
                if status == "error":
                    break
                await asyncio.sleep(1)
            else:
                # Loop exited without break → polled out without completion.
                pass

            # 4. Fetch the actual text
            try:
                tr = await client.get(
                    f"{_SONIOX_API_BASE}/v1/transcriptions/{transcription_id}/transcript",
                    headers=headers,
                )
                if tr.status_code == 200:
                    transcript_text = str((tr.json() or {}).get("text") or "").strip()
            except Exception:
                pass

            # 5. Cleanup — fire-and-forget; failures don't matter to the caller.
            for path in (
                f"/v1/transcriptions/{transcription_id}" if transcription_id else "",
                f"/v1/files/{file_id}" if file_id else "",
            ):
                if not path:
                    continue
                try:
                    await client.delete(f"{_SONIOX_API_BASE}{path}", headers=headers)
                except Exception:
                    pass
    except Exception:
        pass

    return transcript_text


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

    Uses ``_send_whatsapp_reply`` (not ``_send_whatsapp_text``) so the
    optional Soniox audio reply attaches to every agent answer when
    ``WHATSAPP_AUDIO_REPLY=on``.

    Independent of any Messenger callback registered on the same channel —
    both can co-exist (cross-bridge fan-out is intentional).
    """
    register_channel_callback(
        channel_id=channel_id,
        wired_set=_WIRED_CHANNELS,
        recipients_for_channel=lambda ch: _CHANNEL_WAIDS.get(ch, ()),
        send_one=_send_whatsapp_reply,
    )


# ── Cloud API: send text ──────────────────────────────────────────────


# Cloud API text limit is 4096 chars; we chunk at 3500 to leave headroom
# for agent-side punctuation surprises.
_MAX_CHUNK = 3500

# Soniox TTS endpoint used to synthesize the optional audio reply.
_SONIOX_TTS_URL = "https://tts-rt.soniox.com/tts"

# WhatsApp Cloud API caps audio messages at 16 MB. Real synthesized MP3s
# are far smaller (~10 KB/s), so a couple of minutes of speech fits — but
# we cap the text length up-front to keep latency sane and bills bounded.
_TTS_MAX_TEXT = 4000

# Hard cap on synthesized audio bytes before upload. If Soniox ever
# returns more than this (it shouldn't for sane text lengths), we bail.
_MAX_AUDIO_BYTES = 12 * 1024 * 1024


def _send_url() -> str:
    pid = _env("WHATSAPP_PHONE_NUMBER_ID")
    return f"https://graph.facebook.com/v18.0/{pid}/messages" if pid else ""


def _media_url() -> str:
    pid = _env("WHATSAPP_PHONE_NUMBER_ID")
    return f"https://graph.facebook.com/v18.0/{pid}/media" if pid else ""


def _audio_reply_enabled() -> bool:
    """Whether to attach a synthesized MP3 to every agent/face reply."""
    return _env("WHATSAPP_AUDIO_REPLY").lower() in ("on", "true", "yes", "1")


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


async def _send_whatsapp_text(waid: str, text: str, *, mirror: bool = False) -> None:
    """POST a text message to the Cloud API. No-op if config is missing —
    the glasses HUD will still show the agent reply via the channel bus.

    When ``mirror`` is set, the same text is also broadcast onto this WAID's
    channel bus as a ``system`` breadcrumb so the glasses view shows the
    bot's own status replies (e.g. "transcribing…", "transcription: …").
    Mirroring is opt-in precisely because the agent-reply path routes through
    here too (via ``_send_whatsapp_reply``) and is already on the bus — only
    bridge-originated status lines pass ``mirror=True`` to avoid duplicates.
    The ``system`` type is never re-forwarded to WhatsApp/Messenger, so this
    can't echo back to the user (see meta_webhook_bridge._forward)."""
    text = text.strip()
    if not text:
        return
    # WhatsApp bold is *single* asterisks; Markdown **double** shows literal '**'.
    # Convert paired **bold** → *bold* so flow/agent output renders cleanly.
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"*\1*", text)

    if mirror:
        channel = _WAID_CHANNEL.get(waid)
        if channel:
            try:
                ch = await _get_or_create_channel(channel)
                await _broadcast(ch, {
                    "type": "system",
                    "text": text,
                    "ts": _now_iso(),
                    "via": "whatsapp",
                })
            except Exception:
                pass  # best-effort mirror; never block the WhatsApp send

    token = _env("WHATSAPP_ACCESS_TOKEN")
    url = _send_url()
    if not token or not url:
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


# ── Optional audio reply (Soniox TTS → Meta media upload → audio msg) ─


async def _synth_audio_mp3(text: str) -> bytes | None:
    """Synthesize ``text`` to MP3 via Soniox TTS.

    Returns the audio bytes, or ``None`` if Soniox isn't configured or the
    request fails. Errors don't surface to the user — audio reply is a
    nicety; if it can't happen, the text reply still goes through.
    """
    api_key = os.environ.get("SONIOX_API_KEY", "").strip()
    if not api_key:
        return None
    text = (text or "").strip()
    if not text:
        return None
    if len(text) > _TTS_MAX_TEXT:
        text = text[:_TTS_MAX_TEXT]

    # Bridge-specific voice/language overrides; fall back to whatever the
    # glasses TTS already uses so the user gets a consistent voice across
    # surfaces by default.
    voice = (
        _env("WHATSAPP_AUDIO_VOICE")
        or os.environ.get("SONIOX_TTS_VOICE", "").strip()
        or "Adrian"
    )
    language = (
        _env("WHATSAPP_AUDIO_LANGUAGE")
        or os.environ.get("SONIOX_TTS_LANGUAGE", "").strip()
        or "en"
    )

    payload = {
        "model": os.environ.get("SONIOX_TTS_MODEL", "tts-rt-v1"),
        "language": language,
        "voice": voice,
        "audio_format": "mp3",
        "text": text,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                _SONIOX_TTS_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    blob = resp.content
    if not blob or len(blob) > _MAX_AUDIO_BYTES:
        return None
    return blob


async def _upload_whatsapp_audio(blob: bytes) -> str:
    """Upload MP3 bytes to ``/<phone-id>/media`` and return the media id.

    Cloud API requires the messaging_product field in the multipart form
    body alongside the file. Empty return on failure.
    """
    token = _env("WHATSAPP_ACCESS_TOKEN")
    url = _media_url()
    if not token or not url:
        return ""
    headers = {"Authorization": f"Bearer {token}"}
    files = {
        "file": ("reply.mp3", blob, "audio/mpeg"),
        "messaging_product": (None, "whatsapp"),
        "type": (None, "audio/mpeg"),
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, files=files)
    except Exception:
        return ""
    if resp.status_code != 200:
        return ""
    try:
        return str((resp.json() or {}).get("id") or "")
    except Exception:
        return ""


async def _send_whatsapp_audio(waid: str, text: str) -> None:
    """Generate + upload + send an audio message containing ``text``.

    Three steps, any of which can fail silently — the text reply path is
    the source of truth, audio is a UX add-on:

      1. Soniox TTS turns text into MP3
      2. Meta media upload returns a media_id
      3. Send API delivers an ``audio`` message referencing that id

    Failure at any step logs nothing and the user just sees the text-only
    reply they would have received without ``WHATSAPP_AUDIO_REPLY=on``.
    """
    blob = await _synth_audio_mp3(text)
    if not blob:
        return
    media_id = await _upload_whatsapp_audio(blob)
    if not media_id:
        return

    token = _env("WHATSAPP_ACCESS_TOKEN")
    url = _send_url()
    if not token or not url:
        return
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": waid,
        "type": "audio",
        "audio": {"id": media_id},
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(url, headers=headers, json=payload)
    except Exception:
        pass


async def _send_whatsapp_reply(waid: str, text: str) -> None:
    """Send a reply: text always, optional MP3 audio if env opts in.

    Used for substantive replies (agent answers, face cards). Slash
    command and error responses use ``_send_whatsapp_text`` directly so
    they stay text-only regardless of the audio-reply flag.

    Audio path order of operations
    ------------------------------
    1. Send the text reply (user can read immediately).
    2. Send "🎙 Generating audio…" as a status breadcrumb so the user
       knows audio is coming and isn't just stuck waiting.
    3. Re-fire the typing indicator — sending step-2's text cleared the
       one we triggered on inbound, so the dots disappeared. The Cloud
       API requires a real user wamid to attach the indicator to; we
       pull the cached last message id for this WAID.
    4. Synthesize, upload, send audio (background — 1-3 s typically).

    Failures inside any step are silent; the user just sees the text-only
    reply in the worst case.
    """
    await _send_whatsapp_text(waid, text)
    if not _audio_reply_enabled():
        return

    # Status breadcrumb + typing re-trigger, then audio in the background.
    await _send_whatsapp_text(waid, "🎙 Generating audio…")
    last_msg_id = _WAID_LAST_MESSAGE_ID.get(waid, "")
    if last_msg_id:
        asyncio.create_task(_mark_read_and_typing(last_msg_id))
    asyncio.create_task(_send_whatsapp_audio(waid, text))


async def push_to_waid(waid: str, text: str) -> bool:
    """Proactive push entrypoint (FD scheduler + /whatsapp/push endpoint).

    Differs from ``_send_whatsapp_reply`` in two ways:
      * Honours the per-WAID mute set by ``/mute`` — returns ``False``
        without sending if muted.
      * Enforces the allowlist (a proactive push to a non-allowed number
        would be unsolicited messaging — never do it).

    Returns ``True`` if the message was sent, ``False`` if suppressed
    (muted / not allowed / empty). Uses ``_send_whatsapp_reply`` under the
    hood, so the optional audio reply still applies.
    """
    waid = (waid or "").lstrip("+").strip()
    text = (text or "").strip()
    if not waid or not text:
        return False
    if waid not in _allowed_waids():
        return False
    if is_push_muted(waid):
        return False
    await _send_whatsapp_reply(waid, text)
    return True


@router.post("/whatsapp/push")
async def whatsapp_push(request: Request) -> JSONResponse:
    """Proactive push delivery primitive. Body: ``{to, text}``.

    Token-gated (``FD_GLASSES_BRIDGE_TOKEN`` when set) AND allowlist-gated.
    Respects ``/mute``. This is what external triggers / the FD scheduler
    call when they have final text ready for a specific WhatsApp number.
    """
    _check_token(request)
    body = await request.json()
    to = str(body.get("to", "")).strip()
    text = str(body.get("text", "")).strip()
    if not to or not text:
        raise HTTPException(status_code=400, detail="to and text required")
    sent = await push_to_waid(to, text)
    return JSONResponse(
        {"ok": sent, "suppressed": (not sent)}, headers=_NO_CACHE
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


# ── Inbound image routing ─────────────────────────────────────────────
# Photos are no longer "kidnapped" into face recognition. The caption (or,
# for a bare photo, the user's follow-up reply) routes each image:
#   • enroll intent  → face_index.enroll()      (Flight Deck)
#   • identify intent → face_index.recognize()  (Flight Deck)
#   • anything else  → forwarded to the agent for vision
# Face recognition stays entirely on Flight Deck — never an agent tool.


async def _upload_image_to_agent(
    blob: bytes, host: str, port: int, auth: str, filename: str = "whatsapp.jpg"
) -> str:
    """POST image bytes to the agent's /api/image/upload; return saved path."""
    params = {"token": auth} if auth else {}
    files = {"file": (filename, blob, "image/jpeg")}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"http://{host}:{port}/api/image/upload", params=params, files=files
            )
    except Exception as exc:
        log.warning("Image upload to agent failed: %s", exc)
        return ""
    if resp.status_code != 200:
        log.warning("Image upload rejected (%s): %s", resp.status_code, resp.text[:200])
        return ""
    try:
        return str((resp.json() or {}).get("path") or "")
    except Exception:
        return ""


async def _forward_image_to_agent(
    waid: str, blob: bytes, prompt: str, ch: Any,
    agent_host: str, agent_port: int, agent_auth: str,
) -> None:
    """Hand a photo to the agent for vision analysis (describe/read/etc.)."""
    path = await _upload_image_to_agent(blob, agent_host, agent_port, agent_auth)
    if not path:
        await _send_whatsapp_text(
            waid, "Couldn't hand the image to the agent — try again."
        )
        return
    prompt = (prompt or "").strip() or "Look at this image and tell me what it shows."
    # Mirror onto the channel bus so the glasses HUD shows the request.
    await _broadcast(ch, {
        "type": "user", "text": f"🖼 {prompt}", "ts": _now_iso(), "via": "whatsapp",
    })
    for _ in range(50):  # up to ~5s for the agent WS to be ready
        if ch.agent_ws is not None:
            break
        await asyncio.sleep(0.1)
    if ch.agent_ws is None:
        await _send_whatsapp_text(waid, "Agent not ready, try again.")
        return
    async with ch.send_lock:
        content = (_GLASSES_SYSTEM_CONTEXT + prompt) if not ch.context_sent else prompt
        ch.context_sent = True
        payload = {
            "type": "chat",
            "content": content,
            "whatsapp_waid": waid,
            "image_paths": [path],
        }
        try:
            await ch.agent_ws.send(json.dumps(payload))
        except Exception as exc:
            ch.context_sent = False
            await _send_whatsapp_text(waid, f"Send failed: {exc}")


_VIDEO_MIME_EXT = {
    "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
    "video/x-matroska": ".mkv", "video/3gpp": ".3gp", "video/x-msvideo": ".avi",
}


async def _upload_video_to_agent(
    blob: bytes, host: str, port: int, auth: str, filename: str = "whatsapp.mp4"
) -> str:
    """POST video bytes to the agent's /api/file/upload; return saved path."""
    params = {"token": auth} if auth else {}
    files = {"file": (filename, blob, "application/octet-stream")}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"http://{host}:{port}/api/file/upload", params=params, files=files
            )
    except Exception as exc:
        log.warning("Video upload to agent failed: %s", exc)
        return ""
    if resp.status_code != 200:
        log.warning("Video upload rejected (%s): %s", resp.status_code, resp.text[:200])
        return ""
    try:
        return str((resp.json() or {}).get("path") or "")
    except Exception:
        return ""


async def _forward_video_to_agent(
    waid: str, blob: bytes, prompt: str, mime: str, ch: Any,
    agent_host: str, agent_port: int, agent_auth: str,
) -> None:
    """Hand a video to the agent; chat_handler auto-runs video_vision on it."""
    ext = _VIDEO_MIME_EXT.get((mime or "").split(";")[0].strip(), ".mp4")
    path = await _upload_video_to_agent(blob, agent_host, agent_port, agent_auth, f"whatsapp{ext}")
    if not path:
        await _send_whatsapp_text(waid, "Couldn't hand the video to the agent — try again.")
        return
    prompt = (prompt or "").strip() or "Describe this video."
    await _broadcast(ch, {
        "type": "user", "text": f"🎬 {prompt}", "ts": _now_iso(), "via": "whatsapp",
    })
    for _ in range(50):  # up to ~5s for the agent WS to be ready
        if ch.agent_ws is not None:
            break
        await asyncio.sleep(0.1)
    if ch.agent_ws is None:
        await _send_whatsapp_text(waid, "Agent not ready, try again.")
        return
    async with ch.send_lock:
        content = (_GLASSES_SYSTEM_CONTEXT + prompt) if not ch.context_sent else prompt
        ch.context_sent = True
        payload = {
            "type": "chat",
            "content": content,
            "whatsapp_waid": waid,
            "file_paths": [path],
        }
        try:
            await ch.agent_ws.send(json.dumps(payload))
        except Exception as exc:
            ch.context_sent = False
            await _send_whatsapp_text(waid, f"Send failed: {exc}")


async def _route_image(
    waid: str, blob: bytes, caption: str, ch: Any,
    agent_host: str, agent_port: int, agent_auth: str,
) -> None:
    """Route an inbound image by caption intent. Face paths stay on FD."""
    from captain_claw.flight_deck.meta_webhook_bridge import strip_markdown

    # 1. Enroll: "remember this is Alice, colleague from X" / "ovo je Alice".
    enroll = _parse_enroll(caption)
    if enroll:
        name, notes = enroll
        try:
            res = await face_index.get_index().enroll(
                name=name, notes=notes, image_blobs=[blob]
            )
        except RuntimeError:
            await _send_whatsapp_text(
                waid,
                "Face recognition isn't available on this Flight Deck "
                "(install with: pip install captain-claw[faces]).",
            )
            return
        except Exception as exc:
            await _send_whatsapp_text(waid, f"Couldn't save that face: {exc}")
            return
        if res.embeddings_added:
            await _send_whatsapp_reply(
                waid, f"✅ Saved {res.name}'s face. I'll recognise them next time."
            )
        else:
            await _send_whatsapp_text(
                waid, "Couldn't find a clear face in that photo — try another shot."
            )
        return

    # 2. Identify: "who is this", "tko je ovo", "recognise", ...
    if _IDENTIFY_RE.search(caption or ""):
        try:
            result = await face_index.get_index().recognize(image_blob=blob, channel="")
        except RuntimeError:
            await _send_whatsapp_text(
                waid,
                "Face recognition isn't available on this Flight Deck "
                "(install with: pip install captain-claw[faces]).",
            )
            return
        plain = strip_markdown(result.card_markdown) or "Unknown face."
        await _send_whatsapp_reply(waid, plain)
        return

    # 3. Anything else → the agent's vision pipeline, caption as the prompt.
    await _forward_image_to_agent(
        waid, blob, caption, ch, agent_host, agent_port, agent_auth
    )
