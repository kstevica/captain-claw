"""WebSocket side of a shared-agent member ("speaker") — A1.

Flight Deck opens ``/ws?token=<web_auth>`` for a member and adds a signed
``X-FD-Speaker`` header. :func:`speaker_ws_session` verifies it, binds the
socket to that member's own Agent instance and private session, sends the
welcome (with ``speaker_ack``) and the replay, and only THEN registers the
socket for live frames — Flight Deck drops everything that precedes the
welcome, so nothing may overtake it.

:func:`speaker_gate` runs first for every frame on such a socket: frames
outside :data:`SPEAKER_FRAME_ALLOWLIST` are refused, chat and slash commands
are handled here, and every ``chat`` frame ends in exactly ONE
``{"type":"status","status":"ready","turn_end":<_fd_turn>}`` — from the
turn's own ``finally`` when a turn was launched, from here otherwise.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.logging import get_logger
from captain_claw.speaker import (
    MAX_BTW_INSTRUCTIONS,
    MAX_CHAT_CONTENT,
    NOT_ALLOWED_MESSAGE,
    SPEAKER_FRAME_ALLOWLIST,
    SpeakerAuthError,
    slash_allowed,
    speaker_ack_for,
    speaker_commands,
    speaker_error,
    speaker_key_of,
    verify_assertion,
)

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

_PROFILE_FULL_MAX = 20_000
_PROFILE_COMPACT_MAX = 2_000


def _not_allowed() -> dict[str, Any]:
    return speaker_error("not_allowed", NOT_ALLOWED_MESSAGE)


async def speaker_ws_session(
    server: WebServer,
    ws: web.WebSocketResponse,
    request: web.Request,
    header_value: str,
    public_mode: bool,
) -> web.WebSocketResponse:
    """Serve one member socket from handshake to disconnect."""
    from captain_claw.config import get_config
    from captain_claw.web_server import SpeakerCapacityError

    # 1. A public_run agent is a different product; it never takes speakers.
    if public_mode:
        await ws.close(code=4400, message=b"Speakers are not accepted by a public agent")
        return ws

    # 2. Verify the signed assertion (never logged).
    try:
        principal = verify_assertion(header_value, str(get_config().web.auth_token or ""))
    except SpeakerAuthError as exc:
        log.warning("Speaker handshake refused", reason=str(exc))
        await ws.close(code=4401, message=b"Invalid speaker assertion")
        return ws

    # 3. The member's own instance (capacity-capped).
    try:
        agent = await server._get_speaker_agent(principal)
    except SpeakerCapacityError:
        await ws.close(code=4429, message=b"This agent is at member capacity")
        return ws
    except Exception as exc:
        log.error("Speaker instance failed", error=str(exc))
        await ws.close(code=1011, message=b"Speaker instance failed")
        return ws

    # 4. Bind the socket. NOT added to server.clients or server._lane_sockets:
    #    that keeps it out of _broadcast, the inbound consumer and hotkeys.
    key = (principal.speaker_id, principal.lane)
    ws._speaker_key = key  # type: ignore[attr-defined]
    ws._speaker_principal = principal  # type: ignore[attr-defined]
    ws._lane = principal.lane  # type: ignore[attr-defined]
    ws._is_admin = False  # type: ignore[attr-defined]

    try:
        # 5. Welcome — the member's own session only.
        from captain_claw.session import get_session_manager

        sm = get_session_manager()
        try:
            pb_entries = await sm.list_playbooks(limit=100)
        except Exception:
            pb_entries = []
        playbook_list = [
            {"id": p.id, "name": p.name, "task_type": p.task_type,
             "trigger_description": p.trigger_description or ""}
            for p in pb_entries
        ]
        session_info = dict(server._session_info(agent) or {})
        session_info["tools"] = server._session_tools(agent)
        session = getattr(agent, "session", None)
        sess_meta = (getattr(session, "metadata", None) or {}) if session else {}

        await server._send(ws, {
            "type": "welcome",
            "session": session_info,
            "models": [],
            "commands": speaker_commands(),
            "personalities": [],
            "playbooks": playbook_list,
            "is_public": False,
            "public_code": "",
            "session_settings": {
                "session_name": sess_meta.get("session_display_name", ""),
                "session_description": sess_meta.get("session_description", ""),
                "session_instructions": sess_meta.get("session_instructions", ""),
                "locked": bool(sess_meta.get("session_settings_locked", False)),
            },
            "speaker_ack": speaker_ack_for(header_value),
            "speaker": {
                "id": principal.speaker_id,
                "name": principal.display_name,
                "owner_name": principal.owner_name,
                "lane": principal.lane,
            },
        })

        # 6. Replay this member's session; always close with replay_done.
        from captain_claw.web.ws_handler import _build_replay_batch

        batch = _build_replay_batch(session) if session is not None else []
        if batch:
            await server._send(ws, {"type": "replay_batch", "messages": batch})
        await server._send(ws, {"type": "replay_done"})
    except Exception as exc:
        log.error("Speaker welcome failed", error=str(exc))
        await ws.close(code=1011, message=b"Speaker welcome failed")
        server._speaker_last_used[key] = time.monotonic()
        return ws

    # 7. Only now may live frames (another tab's turn, instance chatter) reach it.
    server._speaker_sockets.setdefault(key, set()).add(ws)

    # 8. Receive loop — same shape as the owner's (ws_handler.ws_handler).
    from aiohttp.client_exceptions import ClientConnectionResetError

    from captain_claw.web.ws_handler import handle_ws_message

    try:
        async for raw_msg in ws:
            if raw_msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(raw_msg.data)
                except json.JSONDecodeError:
                    await server._send(ws, speaker_error("invalid", "Invalid JSON"))
                    continue
                await handle_ws_message(server, ws, data)
            elif raw_msg.type == web.WSMsgType.ERROR:
                log.error("WebSocket error", error=str(ws.exception()))
    except (ClientConnectionResetError, ConnectionResetError, ConnectionError):
        pass
    finally:
        server._speaker_sockets.get(key, set()).discard(ws)
        server._speaker_last_used[key] = time.monotonic()

    return ws


async def speaker_gate(server: WebServer, ws: Any, data: Any) -> bool:
    """First stop for every frame on a member socket.

    Returns False when the frame was handled (or refused) here, True when the
    existing ``handle_ws_message`` branch should run (``cancel``,
    ``message_feedback``, ``session_settings`` — they resolve the agent through
    ``server.resolve_agent(ws)``, which is the member's own instance,
    ``ws._is_admin`` is False so the settings lock holds, and the settings
    fields are capped for a member there).
    """
    if not isinstance(data, dict):
        await server._send(ws, speaker_error("invalid", "Invalid frame"))
        return False
    msg_type = data.get("type", "")
    if not isinstance(msg_type, str) or msg_type not in SPEAKER_FRAME_ALLOWLIST:
        await server._send(ws, _not_allowed())
        return False

    if msg_type == "fd_speaker_context":
        await _speaker_context(server, ws, data)
        return False
    if msg_type == "chat":
        await _speaker_chat(server, ws, data)
        return False
    if msg_type == "set_playbook":
        await _speaker_set_playbook(server, ws, data)
        return False
    if msg_type == "btw":
        await _speaker_btw(server, ws, data)
        return False
    if msg_type == "approval_response":
        request_id = str(data.get("id", ""))
        key = speaker_key_of(ws)
        if request_id and key and server._speaker_approval_ids.get(request_id) == key:
            server.resolve_playbook_approval(request_id, bool(data.get("approved", False)))
        return False
    return True


async def _speaker_context(server: WebServer, ws: Any, data: dict) -> None:
    """The member's profile, sent once by Flight Deck before any client frame."""
    if getattr(ws, "_speaker_ctx_done", False) or getattr(ws, "_speaker_chatted", False):
        await server._send(ws, _not_allowed())
        return
    ws._speaker_ctx_done = True
    agent = await server.resolve_agent(ws)
    full = data.get("profile_full", "")
    compact = data.get("profile_compact", "")
    agent._speaker_profile = (
        (full if isinstance(full, str) else "")[:_PROFILE_FULL_MAX],
        (compact if isinstance(compact, str) else "")[:_PROFILE_COMPACT_MAX],
    )
    instructions = getattr(agent, "instructions", None)
    cache = getattr(instructions, "_cache", None)
    if isinstance(cache, dict):
        cache.pop("system_prompt.md", None)
        cache.pop("micro_system_prompt.md", None)


async def _speaker_chat(server: WebServer, ws: Any, data: dict) -> None:
    """A member chat frame: exactly one turn_end, whatever happens."""
    from captain_claw.web_server import SpeakerCapacityError

    tid = str(data.get("_fd_turn", ""))[:32]
    owned = False
    try:
        ws._speaker_chatted = True
        content = str(data.get("content", "")).strip()
        if not content:
            return
        if len(content) > MAX_CHAT_CONTENT:
            await server._send(ws, speaker_error("invalid", "Message is too long."))
            return
        if content.startswith("/"):
            if not slash_allowed(content):
                await server._send(ws, {
                    "type": "command_result", "command": content, "content": NOT_ALLOWED_MESSAGE,
                })
            else:
                from captain_claw.web.slash_commands import handle_command

                agent = await server.resolve_agent(ws)
                await handle_command(server.lane_view(ws, agent), ws, content)
            return
        # Attachments, origin, whatsapp_waid, deny_tools, no_tools and
        # no_broadcast are never read for a member; flows never run.
        from captain_claw.speaker import CHAT_GRANT_FIELD, sanitize_grant
        from captain_claw.web.chat_handler import handle_chat

        # The per-turn grant Flight Deck minted for THIS message (A2). Read
        # only here, on a verified speaker socket; never logged.
        grant = sanitize_grant(data.get(CHAT_GRANT_FIELD))
        owned = bool(await handle_chat(
            server, ws, content,
            rewind_to=str(data.get("rewind_to", "")).strip() or None,
            no_flow=True,
            no_next_steps=bool(data.get("no_next_steps", False)),
            no_rephrase=bool(data.get("no_rephrase", False)),
            speaker_turn=tid,
            speaker_grant=grant,
        ))
    except SpeakerCapacityError:
        await server._send(ws, speaker_error("capacity", "This agent is at member capacity."))
    except Exception as exc:
        log.error("Speaker chat failed", error=str(exc))
        await server._send(ws, speaker_error("invalid", "Your message could not be handled."))
    finally:
        if not owned:
            await server._send(ws, {"type": "status", "status": "ready", "turn_end": tid})


async def _speaker_btw(server: WebServer, ws: Any, data: dict) -> None:
    """A note for the member's RUNNING turn. Kept only while that turn runs
    (``_run_agent`` clears the list when it ends) and bounded, so an idle
    instance never accumulates notes in the owner's agent process."""
    raw = data.get("content", "")
    content = (raw if isinstance(raw, str) else "").strip()
    if not content:
        return
    if len(content) > MAX_CHAT_CONTENT:
        await server._send(ws, speaker_error("invalid", "Message is too long."))
        return
    agent = await server.resolve_agent(ws)
    if not getattr(agent, "_lane_busy", False):
        await server._send(ws, {
            "type": "command_result", "command": "/btw",
            "content": "Nothing is running right now — send it as a message instead.",
        })
        return
    notes = getattr(agent, "_btw_instructions", None)
    if not isinstance(notes, list):
        notes = []
        agent._btw_instructions = notes
    if len(notes) >= MAX_BTW_INSTRUCTIONS:
        await server._send(ws, speaker_error(
            "invalid", "That's as many notes as one message can take — send a new message instead.",
        ))
        return
    notes.append(content)
    log.info("BTW instruction added", count=len(notes), speaker=True)
    await server._send(ws, {
        "type": "command_result", "command": "/btw",
        "content": f"Got it — noted for the remaining steps: *{content[:500]}*",
    })


async def _speaker_set_playbook(server: WebServer, ws: Any, data: dict) -> None:
    """Playbook override for THIS member's instance; reported to them only."""
    playbook_id = str(data.get("playbook_id", "")).strip()
    agent = await server.resolve_agent(ws)
    agent._playbook_override = playbook_id or None
    if not playbook_id:
        agent._playbook_override_name = "Auto"
        msg_text = "Playbook mode set to **Auto**. The system will automatically select relevant playbooks."
    elif playbook_id == "__none__":
        agent._playbook_override_name = "None"
        msg_text = "Playbook mode set to **None**. No playbook guidance will be injected."
    else:
        from captain_claw.session import get_session_manager

        try:
            pb = await get_session_manager().load_playbook(playbook_id)
        except Exception:
            pb = None
        label = pb.name if pb else playbook_id
        agent._playbook_override_name = label
        msg_text = f"Playbook override set to **{label}**. This playbook will be used for all tasks."
    key = speaker_key_of(ws)
    if key:
        server._speaker_send(key)({"type": "session_info", **server._session_info(agent)})
    await server._send(ws, {"type": "command_result", "command": "/playbook", "content": msg_text})
