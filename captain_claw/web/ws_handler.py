"""WebSocket protocol handler for the web UI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.agent import Agent
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def ws_handler(server: WebServer, request: web.Request) -> web.WebSocketResponse:
    """Handle WebSocket connections."""
    ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
    await ws.prepare(request)
    server.clients.add(ws)

    # Send welcome payload
    from captain_claw.web_server import COMMANDS

    models = server.agent.get_allowed_models() if server.agent else []

    # Available user profiles for the persona selector.
    # These describe who the agent is talking to (not the agent's identity).
    from captain_claw.personality import list_user_personalities
    user_personalities = list_user_personalities()
    approved = getattr(server, "_approved_telegram_users", {})
    for up in user_personalities:
        uid = str(up.get("user_id", "")).strip()
        up["id"] = uid
        up["is_telegram"] = uid in approved
    personalities = user_personalities

    # Fetch available playbooks for the playbook override selector.
    from captain_claw.session import get_session_manager as _get_sm
    _sm = _get_sm()
    _pb_entries = await _sm.list_playbooks(limit=100)
    _playbook_list = [
        {"id": p.id, "name": p.name, "task_type": p.task_type,
         "trigger_description": p.trigger_description or ""}
        for p in _pb_entries
    ]

    await server._send(ws, {
        "type": "welcome",
        "session": server._session_info(),
        "models": models,
        "commands": COMMANDS,
        "personalities": personalities,
        "playbooks": _playbook_list,
    })

    # Replay existing session messages for the connecting client
    if server.agent and server.agent.session:
        for msg in server.agent.session.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_name = msg.get("tool_name", "")
            timestamp = msg.get("timestamp", "")
            model = msg.get("model", "")
            if role in ("user", "assistant"):
                payload = {
                    "type": "chat_message",
                    "role": role,
                    "content": content,
                    "replay": True,
                    "timestamp": timestamp,
                    "model": model,
                }
                if msg.get("feedback"):
                    payload["feedback"] = msg["feedback"]
                await server._send(ws, payload)
            elif role == "tool" and tool_name == "task_rephrase":
                # Replay task rephrase as a visible chat panel.
                await server._send(ws, {
                    "type": "chat_message",
                    "role": "rephrase",
                    "content": content,
                    "replay": True,
                })
            elif role == "tool" and tool_name and not Agent._is_monitor_only_tool_name(tool_name):
                await server._send(ws, {
                    "type": "monitor",
                    "tool_name": tool_name,
                    "arguments": msg.get("tool_arguments", {}),
                    "output": content,
                    "replay": True,
                })
        await server._send(ws, {"type": "replay_done"})

    try:
        async for raw_msg in ws:
            if raw_msg.type in (
                web.WSMsgType.TEXT,
            ):
                try:
                    data = json.loads(raw_msg.data)
                except json.JSONDecodeError:
                    await server._send(ws, {"type": "error", "message": "Invalid JSON"})
                    continue
                await handle_ws_message(server, ws, data)
            elif raw_msg.type == web.WSMsgType.ERROR:
                log.error("WebSocket error", error=str(ws.exception()))
    finally:
        server.clients.discard(ws)

    return ws


async def handle_ws_message(
    server: WebServer, ws: web.WebSocketResponse, data: dict
) -> None:
    """Dispatch incoming WebSocket messages."""
    msg_type = data.get("type", "")

    if msg_type == "chat":
        content = str(data.get("content", "")).strip()
        image_path = str(data.get("image_path", "")).strip() or None
        file_path = str(data.get("file_path", "")).strip() or None
        if not content and not image_path and not file_path:
            return
        if content.startswith("/"):
            from captain_claw.web.slash_commands import handle_command
            await handle_command(server, ws, content)
        else:
            from captain_claw.web.chat_handler import handle_chat
            await handle_chat(server, ws, content, image_path=image_path, file_path=file_path)

    elif msg_type == "command":
        command = str(data.get("command", "")).strip()
        if command:
            from captain_claw.web.slash_commands import handle_command
            await handle_command(server, ws, command)

    elif msg_type == "btw":
        # Inject additional instructions while a task is running.
        btw_content = str(data.get("content", "")).strip()
        if btw_content and server.agent:
            if not hasattr(server.agent, "_btw_instructions"):
                server.agent._btw_instructions = []
            server.agent._btw_instructions.append(btw_content)
            log.info("BTW instruction added", count=len(server.agent._btw_instructions))
            await server._send(ws, {
                "type": "command_result",
                "command": "/btw",
                "content": f"Got it — noted for the remaining steps: *{btw_content}*",
            })

    elif msg_type == "set_model":
        # Switch model for the current session via direct WebSocket message.
        selector = str(data.get("selector", "")).strip()
        if selector and server.agent:
            ok, msg = await server.agent.set_session_model(selector, persist=True)
            if ok:
                server._broadcast({"type": "session_info", **server._session_info()})
            await server._send(ws, {
                "type": "command_result",
                "command": "/session model",
                "content": msg,
            })

    elif msg_type == "set_personality":
        # Switch the active user profile for the web chat session.
        # This does NOT change the agent's identity — it sets context
        # about who the agent is talking to (user's expertise/background).
        personality_id = str(data.get("personality_id", "")).strip() or None
        if server.agent:
            server.agent._active_personality_id = personality_id
            # Clear instruction caches so the prompt rebuilds with new user context.
            if hasattr(server.agent, "instructions") and hasattr(server.agent.instructions, "_cache"):
                server.agent.instructions._cache.pop("system_prompt.md", None)
                server.agent.instructions._cache.pop("micro_system_prompt.md", None)
            server._broadcast({"type": "session_info", **server._session_info()})
            # Report the change.
            if personality_id:
                from captain_claw.personality import load_user_personality
                up = load_user_personality(personality_id)
                label = up.name if up else personality_id
                msg_text = f"User profile set to **{label}**. Responses will be tailored to this user's perspective."
            else:
                msg_text = "User profile cleared. Using default context."
            await server._send(ws, {
                "type": "command_result",
                "command": "/user-profile",
                "content": msg_text,
            })

    elif msg_type == "set_playbook":
        # Override which playbook the agent uses for retrieval.
        # '' = auto (default), '__none__' = disabled, or a specific playbook ID.
        playbook_id = str(data.get("playbook_id", "")).strip()
        if server.agent:
            server.agent._playbook_override = playbook_id or None
            # Report the change.
            if not playbook_id:
                server.agent._playbook_override_name = "Auto"
                msg_text = "Playbook mode set to **Auto**. The system will automatically select relevant playbooks."
            elif playbook_id == "__none__":
                server.agent._playbook_override_name = "None"
                msg_text = "Playbook mode set to **None**. No playbook guidance will be injected."
            else:
                from captain_claw.session import get_session_manager as _get_sm
                _sm = _get_sm()
                pb = await _sm.load_playbook(playbook_id)
                label = pb.name if pb else playbook_id
                server.agent._playbook_override_name = label
                msg_text = f"Playbook override set to **{label}**. This playbook will be used for all tasks."
            server._broadcast({"type": "session_info", **server._session_info()})
            await server._send(ws, {
                "type": "command_result",
                "command": "/playbook",
                "content": msg_text,
            })

    elif msg_type == "set_force_script":
        enabled = bool(data.get("enabled", False))
        if server.agent:
            server.agent._force_script_mode = enabled
            server._broadcast({"type": "session_info", **server._session_info()})
            log.info("Force script mode toggled", enabled=enabled)

    elif msg_type == "message_feedback":
        # Store like/dislike feedback on a session message.
        ts = str(data.get("timestamp", "")).strip()
        fb = data.get("feedback")  # "good", "bad", or null to clear
        if ts and server.agent and server.agent.session:
            from captain_claw.session import get_session_manager
            session = server.agent.session
            for msg in session.messages:
                if msg.get("timestamp") == ts and msg.get("role") == "assistant":
                    if fb:
                        msg["feedback"] = fb
                    else:
                        msg.pop("feedback", None)
                    break
            sm = get_session_manager()
            await sm.save_session(session)

    elif msg_type == "cancel":
        if server.agent and hasattr(server.agent, "cancel_event"):
            server.agent.cancel_event.set()
            log.info("Cancel signal received via WebSocket")

    elif msg_type == "approval_response":
        pass
