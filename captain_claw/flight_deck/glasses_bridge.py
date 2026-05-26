"""Glasses bridge (v1) — mobile-web text input → agent → glasses-web output.

Flow:
  1. User opens ``/glasses/mobile?c=<channel>`` on their phone.
  2. User opens ``/glasses/view?c=<channel>`` on the Meta Ray-Ban Display.
  3. Mobile picks a Flight Deck container (running captain-claw agent) and
     submits text.
  4. Flight Deck routes that text into the agent's WebSocket and fans the
     agent's ``chat_message`` replies out to every subscriber on the channel
     (the glasses, the mobile, plus any other tab on the same channel).

Why a per-channel in-process bus:
  - Glasses and mobile aren't paired by login — they share a URL.
  - Flight Deck has at most one persistent WS open per channel, regardless
    of how many tabs are listening. The agent doesn't know it's being
    "broadcast"; from its side it sees a normal client.

Auth: skipped for v1 (as requested). Optional ``FD_GLASSES_BRIDGE_TOKEN``
gives a single shared secret check for ``/glasses/send`` and the mobile
page if you want to expose the tunnel publicly.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

if TYPE_CHECKING:  # for type hints only — avoid runtime import cycles
    import websockets as _ws_lib  # noqa: F401

UTC = timezone.utc

router = APIRouter()

_STATIC_DIR = Path(__file__).resolve().parent / "static"

_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

# Hidden context injected once per channel→agent binding. Sent to the agent
# only — NEVER broadcast to the channel bus, so it never appears in the
# mobile log or the glasses view. Wrapped in a clear sentinel so the agent
# recognises it as instructions about the rendering surface, not as user
# content to reply to.
_GLASSES_SYSTEM_CONTEXT = (
    "[SYSTEM CONTEXT — do not echo, quote, or acknowledge this block in your reply.]\n"
    "Your reply will be rendered on Meta Ray-Ban Display smart glasses with a "
    "very small viewport in the user's peripheral vision. The user cannot scroll "
    "comfortably and reads at a glance.\n"
    "Output rules for every reply on this session:\n"
    "  - Lead with the answer. No preamble, no 'Sure', no 'Here is', no recap.\n"
    "  - Keep it short: 1–3 short sentences for prose answers, or a compact list.\n"
    "  - Markdown renders (GFM): use **bold** sparingly for the key term; tables "
    "are fine for small structured data (≤4 columns, ≤5 rows); avoid code blocks "
    "unless essential and keep them ≤6 lines.\n"
    "  - Aim for ~30–40 characters per line where possible; avoid wide paragraphs.\n"
    "  - No follow-up questions unless the request is genuinely ambiguous.\n"
    "  - Do not mention the glasses, this context, or these rules.\n"
    "Apply these rules to this and every subsequent message until told otherwise.\n"
    "---\n"
    "USER MESSAGE:\n"
)

# ── In-memory channel bus ─────────────────────────────────────────────


@dataclass
class _ChannelState:
    """Per-channel runtime state. Lives only in-process."""

    channel_id: str
    subscribers: set[WebSocket] = field(default_factory=set)
    # Bound agent — set by the first /glasses/send call. Subsequent sends to
    # the same channel reuse the same outbound link unless the target changes.
    bound_host: str | None = None
    bound_port: int | None = None
    agent_ws_task: asyncio.Task | None = None
    agent_ws: Any | None = None  # websockets.WebSocketClientProtocol
    # Last N messages for backfill when a new subscriber joins (e.g. glasses
    # reload). Keep it small — glasses screen shows ~3 messages anyway.
    recent: deque[dict] = field(default_factory=lambda: deque(maxlen=10))
    # Serializes outbound agent-WS writes so two near-simultaneous mobile
    # sends don't interleave on the wire.
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Whether we've already prepended the hidden glasses-context block to a
    # message sent to the currently bound agent. Reset on every rebind so a
    # freshly-picked agent gets the context too.
    context_sent: bool = False


_channels: dict[str, _ChannelState] = {}
_channels_lock = asyncio.Lock()


async def _get_or_create_channel(channel_id: str) -> _ChannelState:
    async with _channels_lock:
        ch = _channels.get(channel_id)
        if ch is None:
            ch = _ChannelState(channel_id=channel_id)
            _channels[channel_id] = ch
        return ch


def _check_token(request: Request) -> None:
    """Optional shared-secret gate (v1 simple)."""
    required = os.environ.get("FD_GLASSES_BRIDGE_TOKEN", "").strip()
    if not required:
        return
    got = request.query_params.get("t", "") or request.headers.get("x-glasses-token", "")
    if got != required:
        raise HTTPException(status_code=401, detail="invalid bridge token")


def _check_token_ws(ws: WebSocket) -> bool:
    required = os.environ.get("FD_GLASSES_BRIDGE_TOKEN", "").strip()
    if not required:
        return True
    got = ws.query_params.get("t", "")
    return got == required


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def _broadcast(ch: _ChannelState, payload: dict) -> None:
    """Send a JSON payload to every subscriber on this channel. Drops dead
    sockets; never raises."""
    msg = json.dumps(payload, default=str)
    ch.recent.append(payload)
    stale: list[WebSocket] = []
    for ws in list(ch.subscribers):
        try:
            await ws.send_text(msg)
        except Exception:
            stale.append(ws)
    for ws in stale:
        ch.subscribers.discard(ws)


# ── Outbound agent-WS pump ────────────────────────────────────────────


async def _agent_pump(ch: _ChannelState, host: str, port: int) -> None:
    """Persistent connection to a captain-claw agent's WebSocket.

    Forwards every ``chat_message`` (and useful status updates) onto the
    channel bus. Exits cleanly when the channel rebinds to a different
    target or the task is cancelled.
    """
    import websockets

    # Resolve auth token via the same helper Flight Deck uses everywhere.
    try:
        from captain_claw.flight_deck.server import _resolve_agent_auth
        auth = _resolve_agent_auth(port)
    except Exception:
        auth = ""
    params = f"?token={auth}" if auth else ""
    agent_url = f"ws://{host}:{port}/ws{params}"

    try:
        async with websockets.connect(
            agent_url,
            max_size=4 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=10,
        ) as agent_ws:
            ch.agent_ws = agent_ws
            await _broadcast(ch, {
                "type": "status", "status": "agent_connected",
                "host": host, "port": port, "ts": _now_iso(),
            })
            async for raw in agent_ws:
                text = raw if isinstance(raw, str) else raw.decode("utf-8", "ignore")
                try:
                    data = json.loads(text)
                except Exception:
                    continue
                mtype = data.get("type")
                # Only surface what's useful for a tiny glasses display.
                if mtype == "chat_message":
                    role = data.get("role", "")
                    if role == "assistant":
                        await _broadcast(ch, {
                            "type": "agent",
                            "text": str(data.get("content", "")),
                            "ts": data.get("timestamp") or _now_iso(),
                        })
                    # We don't echo "user" chat_messages back — mobile already
                    # injected its own "user" event for instant local display.
                elif mtype == "status":
                    await _broadcast(ch, {
                        "type": "status",
                        "status": str(data.get("status", "")),
                        "ts": _now_iso(),
                    })
                elif mtype == "error":
                    await _broadcast(ch, {
                        "type": "error",
                        "text": str(data.get("message", "")),
                        "ts": _now_iso(),
                    })
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await _broadcast(ch, {
            "type": "error",
            "text": f"agent connection lost: {exc}",
            "ts": _now_iso(),
        })
    finally:
        ch.agent_ws = None


async def _ensure_agent_binding(ch: _ChannelState, host: str, port: int) -> None:
    """Bind the channel to (host, port). Spawns the pump task if needed.

    If the channel was already bound to a different target, cancel the old
    pump first.
    """
    if ch.bound_host == host and ch.bound_port == port and ch.agent_ws_task and not ch.agent_ws_task.done():
        return  # already wired up

    # Tear down any prior binding.
    if ch.agent_ws_task and not ch.agent_ws_task.done():
        ch.agent_ws_task.cancel()
        try:
            await ch.agent_ws_task
        except (asyncio.CancelledError, Exception):
            pass

    ch.bound_host = host
    ch.bound_port = port
    # Fresh agent → it hasn't seen our glasses context yet.
    ch.context_sent = False
    ch.agent_ws_task = asyncio.create_task(_agent_pump(ch, host, port))


# ── HTTP routes ───────────────────────────────────────────────────────


@router.get("/glasses")
async def glasses_root() -> RedirectResponse:
    """Helper: open with no params → bounce to a fresh mobile page on a new channel."""
    channel = secrets.token_urlsafe(6)
    return RedirectResponse(url=f"/glasses/mobile?c={channel}", status_code=303)


@router.get("/glasses/mobile", response_class=HTMLResponse)
async def glasses_mobile_page(request: Request, c: str = "") -> HTMLResponse:
    _check_token(request)
    if not c:
        c = secrets.token_urlsafe(6)
        return RedirectResponse(url=f"/glasses/mobile?c={c}", status_code=303)  # type: ignore[return-value]
    path = _STATIC_DIR / "glasses_mobile.html"
    html = path.read_text(encoding="utf-8")
    return HTMLResponse(content=html, headers=_NO_CACHE)


@router.get("/glasses/view", response_class=HTMLResponse)
async def glasses_view_page(request: Request, c: str = "") -> HTMLResponse:
    # No auth on the glasses page (v1 — as requested).
    if not c:
        raise HTTPException(status_code=400, detail="missing channel ?c=")
    path = _STATIC_DIR / "glasses_view.html"
    html = path.read_text(encoding="utf-8")
    # Bake the SSR freshness token into the page so a glasses reload that
    # returns the same token means the HTML was cached on device.
    token = secrets.token_hex(3).upper()
    html = html.replace("{{SSR_TOKEN}}", token).replace("{{SSR_TS}}", _now_iso())
    return HTMLResponse(content=html, headers=_NO_CACHE)


@router.get("/glasses/agents")
async def glasses_list_agents(request: Request) -> JSONResponse:
    """List Flight Deck **process** agents the mobile page can target.

    Returns ``[{id, name, host, port, status}]`` where ``id`` is the
    process slug and ``host`` is always localhost (FD spawns processes
    on the same machine). Only running processes with a web port are
    surfaced — anything else can't take chat over WS.
    """
    _check_token(request)
    try:
        from captain_claw.flight_deck.server import (
            _load_process_registry,
            _process_is_alive,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"flight deck unavailable: {exc}") from exc

    registry = _load_process_registry()
    out: list[dict] = []
    for slug, entry in registry.items():
        if not _process_is_alive(slug):
            continue
        try:
            port = int(entry.get("web_port", 0) or 0)
        except Exception:
            port = 0
        if not port:
            continue
        out.append({
            "id": slug,
            "name": entry.get("name", slug),
            "host": "localhost",
            "port": port,
            "status": "running",
            "model": entry.get("model", ""),
            "provider": entry.get("provider", ""),
        })
    return JSONResponse(out, headers=_NO_CACHE)


@router.post("/glasses/send")
async def glasses_send(request: Request) -> JSONResponse:
    """Mobile → agent. Body: ``{channel, host, port, text, image_path?}``.

    ``image_path`` is the absolute path returned by ``/glasses/upload-image``;
    when present, the agent receives it as an attachment (same shape as the
    captain-claw web UI uses).
    """
    _check_token(request)
    body = await request.json()
    channel = str(body.get("channel", "")).strip()
    host = str(body.get("host", "")).strip() or "localhost"
    try:
        port = int(body.get("port", 0))
    except Exception:
        port = 0
    text = str(body.get("text", "")).strip()
    image_path = str(body.get("image_path", "")).strip()
    if not channel or not port:
        raise HTTPException(status_code=400, detail="channel, port required")
    if not text and not image_path:
        raise HTTPException(status_code=400, detail="text or image_path required")

    ch = await _get_or_create_channel(channel)
    await _ensure_agent_binding(ch, host, port)

    # Echo the user's message onto the bus immediately — mobile and glasses
    # both want to render it without waiting for the agent's first reply.
    # ``image_path`` is forwarded so the glasses view can show a 📷 marker.
    user_event: dict = {"type": "user", "text": text, "ts": _now_iso()}
    if image_path:
        user_event["image_path"] = image_path
    await _broadcast(ch, user_event)

    # Wait briefly for the agent WS to come up (first send after binding).
    for _ in range(50):  # up to ~5s
        if ch.agent_ws is not None:
            break
        await asyncio.sleep(0.1)

    if ch.agent_ws is None:
        raise HTTPException(status_code=502, detail="agent WS not ready")

    # Inject the hidden glasses-rendering context on the FIRST message sent
    # to this agent binding. Sent to the agent only — the broadcast above
    # used the user's plain text, so the context never reaches the channel
    # bus and is invisible to both the glasses view and the mobile log.
    async with ch.send_lock:
        # Default caption if the user attached an image without text.
        effective_text = text or ("Please analyze this image." if image_path else text)
        if not ch.context_sent:
            agent_content = _GLASSES_SYSTEM_CONTEXT + effective_text
            ch.context_sent = True
        else:
            agent_content = effective_text
        payload_obj: dict = {"type": "chat", "content": agent_content}
        if image_path:
            # Matches the contract in captain_claw/web/ws_handler.py: the
            # agent reads ``image_path`` and prefixes the prompt with
            # ``[Attached image: <abs path>]`` before running the chat.
            payload_obj["image_path"] = image_path
        payload = json.dumps(payload_obj)
        try:
            await ch.agent_ws.send(payload)
        except Exception as exc:
            # If the send failed the agent didn't actually receive the
            # context — let the next attempt resend it.
            ch.context_sent = False
            raise HTTPException(status_code=502, detail=f"agent send failed: {exc}") from exc
    return JSONResponse({"ok": True}, headers=_NO_CACHE)


# ── Image upload proxy ────────────────────────────────────────────────


@router.post("/glasses/upload-image")
async def glasses_upload_image(request: Request) -> JSONResponse:
    """Receive a photo from the mobile bridge and forward it to the bound
    agent's ``/api/image/upload``.

    Form fields (multipart):
      - ``file``: the image bytes
      - ``host``, ``port``: target agent (same values the mobile uses for /glasses/send)

    Returns the agent's JSON response (``{path, filename, size}``). The path
    lives on the agent's filesystem and is what the mobile then passes back
    in ``/glasses/send`` as ``image_path``.
    """
    _check_token(request)
    import httpx
    from starlette.datastructures import UploadFile as _Upload

    form = await request.form()
    host = str(form.get("host", "")).strip() or "localhost"
    try:
        port = int(str(form.get("port", "0")).strip())
    except Exception:
        port = 0
    upload = form.get("file")
    if not isinstance(upload, _Upload):
        raise HTTPException(status_code=400, detail="file field required (multipart)")
    if not port:
        raise HTTPException(status_code=400, detail="port required")

    # Read into memory. Phones produce 2–10 MB photos — fine in-process.
    blob = await upload.read()
    if not blob:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        from captain_claw.flight_deck.server import _resolve_agent_auth
        auth = _resolve_agent_auth(port)
    except Exception:
        auth = ""
    params = f"?token={auth}" if auth else ""
    target = f"http://{host}:{port}/api/image/upload{params}"

    files = {"file": (upload.filename or "photo.jpg", blob, upload.content_type or "image/jpeg")}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(target, files=files)
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=502, detail=f"cannot reach agent: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"forward failed: {exc}") from exc

    if resp.status_code != 200:
        # Surface the agent's error verbatim — easier to debug than a generic 502.
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    try:
        data = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="agent returned non-JSON")
    return JSONResponse(data, headers=_NO_CACHE)


# ── WebSocket bus ─────────────────────────────────────────────────────


@router.websocket("/glasses/ws")
async def glasses_ws(ws: WebSocket, c: str = Query(""), role: str = Query("glasses")) -> None:
    """Bidirectional but mostly one-way bus. Clients send nothing important;
    the server pushes ``user``/``agent``/``status``/``error`` events."""
    if not _check_token_ws(ws):
        await ws.close(code=4401)
        return
    if not c:
        await ws.close(code=4400)
        return
    await ws.accept()
    ch = await _get_or_create_channel(c)
    ch.subscribers.add(ws)
    try:
        # Backfill the most recent messages so a freshly-loaded glasses page
        # sees the last thing the agent said.
        for msg in list(ch.recent):
            try:
                await ws.send_text(json.dumps(msg, default=str))
            except Exception:
                break
        await ws.send_text(json.dumps({
            "type": "status",
            "status": "subscribed",
            "role": role,
            "channel": c,
            "bound": ch.bound_host is not None,
            "ts": _now_iso(),
        }))
        # Idle loop — we don't act on client messages in v1, but we need to
        # await so the socket stays open.
        while True:
            try:
                await ws.receive_text()
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    finally:
        ch.subscribers.discard(ws)


# ── TTS proxy (Soniox fallback) ───────────────────────────────────────


# Default to MP3 — broadest browser support, smallest payload. Soniox docs:
# https://soniox.com/docs/tts/rest-api/generate-speech
_SONIOX_TTS_URL = "https://tts-rt.soniox.com/tts"
_SONIOX_FORMAT_TO_MIME = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "aac": "audio/aac",
    "opus": "audio/ogg",   # opus in ogg container; most browsers accept ogg
    "flac": "audio/flac",
}


@router.post("/glasses/tts")
async def glasses_tts(request: Request) -> Response:
    """Server-side fallback for browsers that don't have a working
    ``speechSynthesis``. Proxies to Soniox TTS and returns the raw audio
    bytes so the glasses view can play it via ``<audio>``.

    Body: ``{text: str}``.  Configuration via env:
      - ``SONIOX_API_KEY`` (required)
      - ``SONIOX_TTS_MODEL``    default ``tts-rt-v1``
      - ``SONIOX_TTS_VOICE``    default ``Adrian``
      - ``SONIOX_TTS_LANGUAGE`` default ``en``
      - ``SONIOX_TTS_FORMAT``   default ``mp3``  (one of: mp3, wav, aac, opus, flac)
    """
    _check_token(request)

    api_key = os.environ.get("SONIOX_API_KEY", "").strip()
    if not api_key:
        # 503 lets the client distinguish "TTS not configured" from real errors.
        raise HTTPException(status_code=503, detail="SONIOX_API_KEY not set")

    body = await request.json()
    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    # Cap to a sensible size — protects against runaway agent replies that
    # would blow up the TTS bill and latency.
    if len(text) > 4000:
        text = text[:4000]

    # Optional per-request overrides from the client; fall back to env.
    language = str(body.get("language", "")).strip() or os.environ.get("SONIOX_TTS_LANGUAGE", "en")
    voice = str(body.get("voice", "")).strip() or os.environ.get("SONIOX_TTS_VOICE", "Adrian")

    audio_format = os.environ.get("SONIOX_TTS_FORMAT", "mp3").strip().lower()
    mime = _SONIOX_FORMAT_TO_MIME.get(audio_format, "audio/mpeg")
    payload = {
        "model": os.environ.get("SONIOX_TTS_MODEL", "tts-rt-v1"),
        "language": language,
        "voice": voice,
        "audio_format": audio_format,
        "text": text,
    }

    import httpx
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                _SONIOX_TTS_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=502, detail=f"soniox unreachable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"tts request failed: {exc}") from exc

    if resp.status_code != 200:
        # Surface Soniox's error verbatim — easier to debug a 401 / quota issue.
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return Response(
        content=resp.content,
        media_type=mime,
        headers=_NO_CACHE,
    )


# ── TTS streaming proxy (Soniox WebSocket) ────────────────────────────


# Defaults tuned for browser Web Audio scheduling: 16-bit signed little-endian
# PCM at 24 kHz, single channel. Lowest decode overhead on the client and
# avoids the "first chunk arrives but the decoder is still buffering"
# problem MP3/Opus can have with chunked input.
_SONIOX_TTS_WS_URL = "wss://tts-rt.soniox.com/tts-websocket"
_STREAM_FORMAT = "pcm_s16le"
_STREAM_SAMPLE_RATE = 24000


@router.websocket("/glasses/tts-stream")
async def glasses_tts_stream(ws: WebSocket) -> None:
    """Stream TTS audio from Soniox to the glasses with minimal latency.

    Client protocol:
      - On connect, send a single JSON: ``{"text": "..."}``.
      - Server replies with a JSON ``info`` message describing the audio:
        ``{"type":"info","format":"pcm_s16le","sample_rate":24000}``.
      - Then a stream of **binary** frames: raw little-endian int16 PCM
        chunks, mono, at the announced sample rate.
      - Server sends ``{"type":"end"}`` and closes when the stream finishes,
        or ``{"type":"error","code":...,"message":...}`` on failure.
    """
    if not _check_token_ws(ws):
        await ws.close(code=4401)
        return

    api_key = os.environ.get("SONIOX_API_KEY", "").strip()
    await ws.accept()
    if not api_key:
        await ws.send_text(json.dumps({
            "type": "error", "code": 503, "message": "SONIOX_API_KEY not set",
        }))
        await ws.close(code=1011)
        return

    # First client message picks the text to synthesize.
    try:
        first = await ws.receive_text()
        client_msg = json.loads(first)
    except WebSocketDisconnect:
        return
    except Exception:
        await ws.send_text(json.dumps({"type": "error", "code": 400, "message": "first message must be JSON"}))
        await ws.close(code=1003)
        return

    text = str(client_msg.get("text", "")).strip()
    if not text:
        await ws.send_text(json.dumps({"type": "error", "code": 400, "message": "text required"}))
        await ws.close(code=1003)
        return
    if len(text) > 4000:
        text = text[:4000]

    # Optional per-request overrides from the client.
    language = str(client_msg.get("language", "")).strip() or os.environ.get("SONIOX_TTS_LANGUAGE", "en")
    voice = str(client_msg.get("voice", "")).strip() or os.environ.get("SONIOX_TTS_VOICE", "Adrian")

    import base64
    import secrets as _secrets
    import websockets

    stream_id = "g_" + _secrets.token_hex(4)
    config = {
        "api_key": api_key,
        "stream_id": stream_id,
        "model": os.environ.get("SONIOX_TTS_MODEL", "tts-rt-v1"),
        "language": language,
        "voice": voice,
        "audio_format": _STREAM_FORMAT,
        "sample_rate": _STREAM_SAMPLE_RATE,
    }

    try:
        async with websockets.connect(
            _SONIOX_TTS_WS_URL,
            max_size=4 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=10,
        ) as upstream:
            await upstream.send(json.dumps(config))
            await upstream.send(json.dumps({
                "stream_id": stream_id,
                "text": text,
                "text_end": True,
            }))

            # Announce format to the browser before binary frames start.
            await ws.send_text(json.dumps({
                "type": "info",
                "format": _STREAM_FORMAT,
                "sample_rate": _STREAM_SAMPLE_RATE,
                "channels": 1,
            }))

            async for raw in upstream:
                # Soniox sends JSON envelopes; audio is base64 inside.
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", "ignore")
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                err = msg.get("error_code") or msg.get("error")
                if err:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": str(msg.get("error_message") or err),
                    }))
                    break

                audio_b64 = msg.get("audio")
                if audio_b64:
                    try:
                        await ws.send_bytes(base64.b64decode(audio_b64))
                    except WebSocketDisconnect:
                        return

                if msg.get("audio_end") or msg.get("terminated") or msg.get("finished"):
                    break

            try:
                await ws.send_text(json.dumps({"type": "end"}))
            except Exception:
                pass
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": f"upstream: {exc}"}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
