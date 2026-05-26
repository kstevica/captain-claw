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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

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
    """Mobile → agent. Body: ``{channel, host, port, text}``."""
    _check_token(request)
    body = await request.json()
    channel = str(body.get("channel", "")).strip()
    host = str(body.get("host", "")).strip() or "localhost"
    try:
        port = int(body.get("port", 0))
    except Exception:
        port = 0
    text = str(body.get("text", "")).strip()
    if not channel or not port or not text:
        raise HTTPException(status_code=400, detail="channel, port, text required")

    ch = await _get_or_create_channel(channel)
    await _ensure_agent_binding(ch, host, port)

    # Echo the user's message onto the bus immediately — mobile and glasses
    # both want to render it without waiting for the agent's first reply.
    await _broadcast(ch, {"type": "user", "text": text, "ts": _now_iso()})

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
        if not ch.context_sent:
            agent_content = _GLASSES_SYSTEM_CONTEXT + text
            ch.context_sent = True
        else:
            agent_content = text
        payload = json.dumps({"type": "chat", "content": agent_content})
        try:
            await ch.agent_ws.send(payload)
        except Exception as exc:
            # If the send failed the agent didn't actually receive the
            # context — let the next attempt resend it.
            ch.context_sent = False
            raise HTTPException(status_code=502, detail=f"agent send failed: {exc}") from exc
    return JSONResponse({"ok": True}, headers=_NO_CACHE)


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
