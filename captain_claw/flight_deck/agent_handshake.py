"""Telling an agent who it is when no chat is open.

An agent's standing instructions live in its owner's settings and reach it in a
``peer_agents`` frame, which until now only an open chat in a browser sent. So an
agent nobody had opened, or one restarted since (idle sleep does that), ran its
scheduled turns knowing neither its job nor its teammates. Flight Deck sends the
same frame itself: before a scheduled turn, and when Forge agents makes a team.

Nothing is sent when the owner has no stored instructions for it, as on a deck with
sign-in off, where they're kept in the browser: an empty frame would wipe what a
chat had set.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

_READY_S = 60.0
_WELCOME_S = 10.0


async def identity_frame(owner_id: str, slug: str, *, instructions: str = "") -> dict[str, Any] | None:
    """The frame a chat would send, from what Flight Deck has: the agent's name and
    instructions, and its owner's other running agents as peers."""
    from captain_claw.flight_deck.server import (
        _load_process_registry,
        _owner_agent_instructions,
        _process_is_alive,
    )

    text = instructions or (await _owner_agent_instructions(owner_id, "process")).get(slug, "")
    if not text:
        return None
    registry = _load_process_registry()
    me = registry.get(slug) or {}
    peers = []
    for other, entry in registry.items():
        port = int(entry.get("web_port") or 0)
        if other == slug or not port or str(entry.get("owner") or "") != str(me.get("owner") or ""):
            continue
        if not _process_is_alive(other):
            continue
        peers.append({"name": entry.get("name") or other, "description": entry.get("description") or "",
                      "forwardingTask": "", "host": "localhost", "port": port, "auth": "", "requireApproval": False})
    return {"type": "peer_agents", "agents": peers,
            "self": {"name": me.get("name") or slug, "description": me.get("description") or "",
                     "port": int(me.get("web_port") or 0), "fleet_instructions": text}}


async def hand_identity(owner_id: str, slug: str, *, instructions: str = "") -> bool:
    """Connect to a just-started agent, once it answers, and tell it. Best-effort:
    its first scheduled turn tells it again."""
    import websockets

    from captain_claw.flight_deck.server import _load_process_registry

    frame = await identity_frame(owner_id, slug, instructions=instructions)
    if frame is None:
        return False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _READY_S
    while loop.time() < deadline:
        entry = _load_process_registry().get(slug) or {}
        port, token = int(entry.get("web_port") or 0), str(entry.get("web_auth") or "")
        if port:
            uri = f"ws://localhost:{port}/ws" + (f"?token={token}" if token else "")
            try:
                async with websockets.connect(uri, open_timeout=5, max_size=None) as ws:
                    # The agent reads the frame once it has welcomed this connection.
                    try:
                        while json.loads(await asyncio.wait_for(ws.recv(), _WELCOME_S)).get("type") != "replay_done":
                            pass
                    except TimeoutError:
                        pass
                    await ws.send(json.dumps(frame))
                    await asyncio.sleep(0.2)
                    return True
            except (OSError, websockets.WebSocketException, ValueError) as exc:
                log.debug("agent handshake: not ready", agent=slug, error=str(exc))
        await asyncio.sleep(1.0)
    log.info("agent handshake: never answered", agent=slug)
    return False
