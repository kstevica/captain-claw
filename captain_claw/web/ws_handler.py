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

    from captain_claw.config import get_config
    cfg = get_config()
    public_mode = bool(cfg.web.public_run)

    # ── Public-mode authentication & session binding ──────────────
    public_session_id: str | None = None
    if public_mode:
        from captain_claw.web.public_auth import _is_admin
        if _is_admin(request, cfg.web):
            ws._is_admin = True  # type: ignore[attr-defined]
        else:
            from captain_claw.web.public_session import read_public_cookie
            identity = read_public_cookie(request, cfg.web.auth_token)
            if identity is None:
                await ws.close(code=4001, message=b"No valid public session")
                return ws
            public_session_id = identity[0]
            ws._is_admin = False  # type: ignore[attr-defined]
            ws._public_session_id = public_session_id  # type: ignore[attr-defined]
    else:
        ws._is_admin = True  # type: ignore[attr-defined]

    server.clients.add(ws)

    # Send welcome payload
    from captain_claw.web_server import COMMANDS

    models = server.agent.get_allowed_models() if server.agent else []

    # Available user profiles for the persona selector.
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

    # For public users, build session info from their specific session.
    if public_session_id:
        pub_session = await _sm.load_session(public_session_id)
        session_info = {
            "id": public_session_id,
            "name": pub_session.name if pub_session else "Public",
        }
    else:
        session_info = server._session_info()

    await server._send(ws, {
        "type": "welcome",
        "session": session_info,
        "models": models,
        "commands": COMMANDS if not public_session_id else [],
        "personalities": personalities,
        "playbooks": _playbook_list if not public_session_id else [],
        "is_public": bool(public_session_id),
    })

    # Replay existing session messages for the connecting client.
    replay_session = None
    if public_session_id:
        replay_session = await _sm.load_session(public_session_id)
    elif server.agent and server.agent.session:
        replay_session = server.agent.session

    if replay_session:
        for msg in replay_session.messages:
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
        rewind_to = str(data.get("rewind_to", "")).strip() or None

        # Multi-file support: collect all image/file paths into lists.
        image_paths: list[str] = []
        file_paths: list[str] = []
        # Single-file (backward compat)
        _ip = str(data.get("image_path", "")).strip()
        if _ip:
            image_paths.append(_ip)
        _fp = str(data.get("file_path", "")).strip()
        if _fp:
            file_paths.append(_fp)
        # Multi-file arrays
        for p in (data.get("image_paths") or []):
            v = str(p).strip()
            if v and v not in image_paths:
                image_paths.append(v)
        for p in (data.get("file_paths") or []):
            v = str(p).strip()
            if v and v not in file_paths:
                file_paths.append(v)

        if not content and not image_paths and not file_paths:
            return
        if content.startswith("/"):
            from captain_claw.web.slash_commands import handle_command
            await handle_command(server, ws, content)
        else:
            from captain_claw.web.chat_handler import handle_chat
            await handle_chat(
                server, ws, content,
                image_path=image_paths[0] if len(image_paths) == 1 else None,
                file_path=file_paths[0] if len(file_paths) == 1 else None,
                image_paths=image_paths if len(image_paths) > 1 else None,
                file_paths=file_paths if len(file_paths) > 1 else None,
                rewind_to=rewind_to,
            )

    elif msg_type == "command":
        command = str(data.get("command", "")).strip()
        if command:
            from captain_claw.web.slash_commands import handle_command
            await handle_command(server, ws, command)

    elif msg_type == "btw":
        # Inject additional instructions while a task is running.
        btw_content = str(data.get("content", "")).strip()
        _pub_sid = getattr(ws, "_public_session_id", None)
        _target_agent = server._public_agents.get(_pub_sid) if _pub_sid else server.agent
        if btw_content and _target_agent:
            if not hasattr(_target_agent, "_btw_instructions"):
                _target_agent._btw_instructions = []
            _target_agent._btw_instructions.append(btw_content)
            log.info("BTW instruction added", count=len(_target_agent._btw_instructions))
            await server._send(ws, {
                "type": "command_result",
                "command": "/btw",
                "content": f"Got it — noted for the remaining steps: *{btw_content}*",
            })

    elif msg_type == "set_model":
        # Switch model for the current session via direct WebSocket message.
        # Public users cannot change the model — it affects the shared agent.
        _is_pub = getattr(ws, "_public_session_id", None)
        if _is_pub:
            await server._send(ws, {
                "type": "command_result",
                "command": "/session model",
                "content": "Model selection is not available in public mode.",
            })
        else:
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

    elif msg_type == "set_byok":
        # BYOK: public user supplies their own LLM provider/model/API key.
        _pub_sid = getattr(ws, "_public_session_id", None)
        if not _pub_sid:
            await server._send(ws, {
                "type": "byok_status",
                "active": False,
                "provider": "",
                "model": "",
                "error": "BYOK is only available in public mode.",
            })
        else:
            _byok_provider = str(data.get("provider", "")).strip()
            _byok_model = str(data.get("model", "")).strip()
            _byok_key = str(data.get("api_key", "")).strip()
            try:
                _pub_agent = await server._get_public_agent(_pub_sid)
                ok, err = _pub_agent.set_byok_provider(_byok_provider, _byok_model, _byok_key)
                if ok:
                    await server._send(ws, {
                        "type": "byok_status",
                        "active": True,
                        "provider": _byok_provider,
                        "model": _byok_model,
                        "error": None,
                    })
                else:
                    await server._send(ws, {
                        "type": "byok_status",
                        "active": False,
                        "provider": "",
                        "model": "",
                        "error": err,
                    })
            except Exception as _byok_exc:
                await server._send(ws, {
                    "type": "byok_status",
                    "active": False,
                    "provider": "",
                    "model": "",
                    "error": str(_byok_exc),
                })

    elif msg_type == "clear_byok":
        # Revert public user to server's default LLM provider.
        _pub_sid = getattr(ws, "_public_session_id", None)
        if _pub_sid:
            try:
                _pub_agent = await server._get_public_agent(_pub_sid)
                _server_provider = server.agent.provider if server.agent else None
                if _server_provider:
                    _pub_agent.clear_byok_provider(_server_provider)
            except Exception:
                pass
        await server._send(ws, {
            "type": "byok_status",
            "active": False,
            "provider": "",
            "model": "",
            "error": None,
        })

    elif msg_type == "set_personality":
        # Switch the active user profile for the web chat session.
        # Public users cannot change personality — it affects the shared agent.
        _is_pub = getattr(ws, "_public_session_id", None)
        if _is_pub:
            await server._send(ws, {
                "type": "command_result",
                "command": "/user-profile",
                "content": "User profile selection is not available in public mode.",
            })
        else:
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
        # Public users cannot change playbook — it affects the shared agent.
        _is_pub = getattr(ws, "_public_session_id", None)
        if _is_pub:
            await server._send(ws, {
                "type": "command_result",
                "command": "/playbook",
                "content": "Playbook selection is not available in public mode.",
            })
        else:
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
        _is_pub = getattr(ws, "_public_session_id", None)
        if _is_pub:
            pass  # Public users cannot toggle force-script mode.
        elif server.agent:
            server.agent._force_script_mode = enabled
            server._broadcast({"type": "session_info", **server._session_info()})
            log.info("Force script mode toggled", enabled=enabled)

    elif msg_type == "message_feedback":
        # Store like/dislike feedback on a session message.
        ts = str(data.get("timestamp", "")).strip()
        fb = data.get("feedback")  # "good", "bad", or null to clear
        _pub_sid = getattr(ws, "_public_session_id", None)
        _fb_agent = server._public_agents.get(_pub_sid) if _pub_sid else server.agent
        if ts and _fb_agent and _fb_agent.session:
            from captain_claw.session import get_session_manager
            session = _fb_agent.session
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
        _pub_sid = getattr(ws, "_public_session_id", None)
        _cancel_agent = server._public_agents.get(_pub_sid) if _pub_sid else server.agent
        if _cancel_agent and hasattr(_cancel_agent, "cancel_event"):
            _cancel_agent.cancel_event.set()
            log.info("Cancel signal received via WebSocket", public=bool(_pub_sid))

    elif msg_type == "approval_response":
        request_id = str(data.get("id", ""))
        approved = bool(data.get("approved", False))
        if request_id:
            server.resolve_playbook_approval(request_id, approved)
