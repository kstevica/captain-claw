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
        _sess_meta = (pub_session.metadata if pub_session else {}) or {}
    else:
        session_info = server._session_info()
        _sess_meta = {}
        if server.agent and server.agent.session:
            _sess_meta = server.agent.session.metadata or {}

    await server._send(ws, {
        "type": "welcome",
        "session": session_info,
        "models": models,
        "commands": COMMANDS if not public_session_id else [],
        "personalities": personalities,
        "playbooks": _playbook_list,
        "is_public": bool(public_session_id),
        "public_code": _sess_meta.get("public_code", ""),
        "session_settings": {
            "session_name": _sess_meta.get("session_display_name", ""),
            "session_description": _sess_meta.get("session_description", ""),
            "session_instructions": _sess_meta.get("session_instructions", ""),
            "locked": bool(_sess_meta.get("session_settings_locked", False)),
        },
    })

    # Replay existing session messages for the connecting client.
    replay_session = None
    if public_session_id:
        replay_session = await _sm.load_session(public_session_id)
    elif server.agent and server.agent.session:
        replay_session = server.agent.session

    if replay_session:
        batch: list[dict] = []
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
                batch.append(payload)
            elif role == "tool" and tool_name == "task_rephrase":
                batch.append({
                    "type": "chat_message",
                    "role": "rephrase",
                    "content": content,
                    "replay": True,
                })
            elif role == "tool" and tool_name and not Agent._is_monitor_only_tool_name(tool_name):
                batch.append({
                    "type": "monitor",
                    "tool_name": tool_name,
                    "arguments": msg.get("tool_arguments", {}),
                    "output": content,
                    "replay": True,
                })
        if batch:
            await server._send(ws, {"type": "replay_batch", "messages": batch})
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


async def _handle_telegram_delegate_result(
    server: WebServer,
    tg_agent: Agent,
    user_id: str,
    chat_id: int,
    content: str,
) -> None:
    """Process a delegate result that originated from a Telegram session.

    Runs the content through the telegram user's agent and sends the
    response back to the Telegram chat (not the web UI).
    """
    import asyncio

    from captain_claw.web.telegram import _tg_send, _tg_get_user_lock

    lock = _tg_get_user_lock(server, user_id)

    async def _run() -> None:
        async with lock:
            try:
                response = await tg_agent.complete(content)
                if response and chat_id:
                    await _tg_send(server, chat_id, response)
                    log.info("Telegram delegate result sent to chat",
                             user_id=user_id, chat_id=chat_id, response_len=len(response))
                # Broadcast the assistant response to FD UI for visibility
                if response:
                    from datetime import datetime, timezone
                    server._broadcast({
                        "type": "chat_message",
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "notification": True,
                    })
            except Exception as exc:
                log.error("Failed to process telegram delegate result",
                          user_id=user_id, error=str(exc))

    asyncio.ensure_future(_run())


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

    elif msg_type == "notification":
        # System notification — inject into session history without triggering LLM.
        # Used for fleet events, delegated results, etc. Does NOT set _busy.
        # If trigger_response=true AND the agent is free, process as a chat message
        # so the agent can act on it (e.g. relay delegated results to the user).
        notif_content = str(data.get("content", "")).strip()
        if not notif_content:
            return
        trigger = data.get("trigger_response", False)
        _pub_sid = getattr(ws, "_public_session_id", None)
        origin_platform = data.get("origin_platform", "web")
        origin_user_id = str(data.get("origin_user_id", ""))
        origin_chat_id = int(data.get("origin_chat_id", 0))

        # ── Telegram-origin delegate results → route to the telegram session ──
        if trigger and origin_platform == "telegram" and origin_user_id:
            log.info("Telegram-origin delegate result, routing to telegram session",
                     user_id=origin_user_id, chat_id=origin_chat_id,
                     content_len=len(notif_content))
            tg_agent = server._telegram_agents.get(origin_user_id)
            if tg_agent:
                # Inject into the telegram user's session
                if tg_agent.session:
                    tg_agent.session.add_message("user", notif_content)
                # Process with the telegram agent and send result to telegram
                await _handle_telegram_delegate_result(
                    server, tg_agent, origin_user_id, origin_chat_id, notif_content,
                )
            else:
                log.warning("Telegram agent not found for delegate result, falling through to web",
                            user_id=origin_user_id)
            # Also broadcast to FD UI for visibility
            from datetime import datetime, timezone
            server._broadcast({
                "type": "chat_message",
                "role": "user",
                "content": notif_content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "notification": True,
            })
            return

        if trigger and not server._busy and not _pub_sid:
            # Agent is free — process as a regular chat so it triggers an LLM response
            log.info("Notification with trigger_response, agent is free — routing to chat handler",
                     content_len=len(notif_content))
            from captain_claw.web.chat_handler import handle_chat
            await handle_chat(server, ws, notif_content)
            return

        # Agent is busy or no trigger requested — inject silently into session
        _target_agent = server._public_agents.get(_pub_sid) if _pub_sid else server.agent
        if _target_agent and _target_agent.session:
            _target_agent.session.add_message("user", notif_content)
            log.info("Notification injected into session", content_len=len(notif_content),
                     agent_busy=server._busy, trigger=trigger)
        # Broadcast to UI so it appears in the chat
        from datetime import datetime, timezone
        server._broadcast({
            "type": "chat_message",
            "role": "user",
            "content": notif_content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notification": True,
        })
        # If trigger requested but agent is busy, queue for processing when agent finishes
        if trigger and server._busy and not _pub_sid:
            if not hasattr(server, "_pending_triggered_notifications"):
                server._pending_triggered_notifications = []
            server._pending_triggered_notifications.append(notif_content)
            log.info("Triggered notification queued for when agent is free",
                     queue_size=len(server._pending_triggered_notifications))

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
        playbook_id = str(data.get("playbook_id", "")).strip()
        # Resolve the target agent — public users have their own agent.
        _pub_sid = getattr(ws, "_public_session_id", None)
        _target_agent = server._public_agents.get(_pub_sid) if _pub_sid else server.agent
        if not _target_agent:
            _target_agent = server.agent  # fallback
        if _target_agent:
                _target_agent._playbook_override = playbook_id or None
                # Report the change.
                if not playbook_id:
                    _target_agent._playbook_override_name = "Auto"
                    msg_text = "Playbook mode set to **Auto**. The system will automatically select relevant playbooks."
                elif playbook_id == "__none__":
                    _target_agent._playbook_override_name = "None"
                    msg_text = "Playbook mode set to **None**. No playbook guidance will be injected."
                else:
                    from captain_claw.session import get_session_manager as _get_sm
                    _sm = _get_sm()
                    pb = await _sm.load_playbook(playbook_id)
                    label = pb.name if pb else playbook_id
                    _target_agent._playbook_override_name = label
                    msg_text = f"Playbook override set to **{label}**. This playbook will be used for all tasks."
                if not _pub_sid:
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

    elif msg_type == "session_settings":
        # Update session name, description, and/or instructions.
        # Works for both public and admin sessions.
        _pub_sid = getattr(ws, "_public_session_id", None)
        _is_ws_admin = getattr(ws, "_is_admin", False)
        _target_agent = server._public_agents.get(_pub_sid) if _pub_sid else server.agent
        if not _target_agent:
            _target_agent = server.agent
        if _target_agent and _target_agent.session:
            from captain_claw.session import get_session_manager
            session = _target_agent.session
            # Enforce lock: public users cannot edit locked sessions.
            _locked = (session.metadata or {}).get("session_settings_locked", False)
            if _locked and not _is_ws_admin:
                await server._send(ws, {
                    "type": "error",
                    "message": "Session settings are locked by the administrator.",
                })
                return
            changed = False
            if "session_name" in data and isinstance(data["session_name"], str):
                val = data["session_name"].strip()
                if val:
                    session.metadata["session_display_name"] = val
                    changed = True
            if "session_description" in data and isinstance(data["session_description"], str):
                session.metadata["session_description"] = data["session_description"].strip()
                changed = True
            if "session_instructions" in data and isinstance(data["session_instructions"], str):
                session.metadata["session_instructions"] = data["session_instructions"].strip()
                changed = True
            if changed:
                sm = get_session_manager()
                await sm.save_session(session)
                # Clear instruction cache so the system prompt rebuilds with new settings.
                if hasattr(_target_agent, "instructions") and hasattr(_target_agent.instructions, "_cache"):
                    _target_agent.instructions._cache.pop("system_prompt.md", None)
                    _target_agent.instructions._cache.pop("micro_system_prompt.md", None)
                log.info(
                    "Session settings updated",
                    session_id=session.id,
                    public=bool(_pub_sid),
                )
            await server._send(ws, {
                "type": "session_settings_saved",
                "session_name": session.metadata.get("session_display_name", ""),
                "session_description": session.metadata.get("session_description", ""),
                "session_instructions": session.metadata.get("session_instructions", ""),
            })

    elif msg_type == "peer_agents":
        # Flight Deck sends info about other available agents so this
        # agent can be aware of its peers and recommend handoffs.
        _pub_sid = getattr(ws, "_public_session_id", None)
        _target_agent = server._public_agents.get(_pub_sid) if _pub_sid else server.agent
        if not _target_agent:
            _target_agent = server.agent
        if _target_agent:
            agents_list = data.get("agents", [])
            fd_url = data.get("fd_url", "")
            if isinstance(agents_list, list):
                if fd_url:
                    # Rewrite localhost to host.docker.internal only when
                    # the agent is running inside a Docker container.
                    import os as _os
                    _in_docker = _os.path.exists("/.dockerenv") or _os.environ.get("CAPTAIN_CLAW_DOCKER")
                    if _in_docker:
                        import re as _re
                        fd_url = _re.sub(
                            r"(https?://)localhost(:\d+)?",
                            r"\1host.docker.internal\2",
                            fd_url,
                        )
                # Store self identity so agent knows its fleet name
                self_identity = data.get("self", None)
                if isinstance(self_identity, dict):
                    if _target_agent.session:
                        _target_agent.session.metadata["fleet_identity"] = self_identity
                    _target_agent._fleet_identity = self_identity

                    # Extract and store fleet-level instructions
                    _fleet_inst = self_identity.get("fleet_instructions", "")
                    if _target_agent.session:
                        _target_agent.session.metadata["fleet_instructions"] = _fleet_inst
                    _target_agent._fleet_instructions = _fleet_inst

                # Store on session metadata if session exists
                if _target_agent.session:
                    _target_agent.session.metadata["peer_agents"] = agents_list
                    if fd_url:
                        _target_agent.session.metadata["fd_url"] = fd_url
                # Also store directly on agent as fallback (session may
                # not exist yet at welcome time or may be swapped later)
                _target_agent._peer_agents = agents_list
                _target_agent._fd_url = fd_url
                # Clear instruction cache so system prompt rebuilds
                if hasattr(_target_agent, "instructions") and hasattr(_target_agent.instructions, "_cache"):
                    _target_agent.instructions._cache.pop("system_prompt.md", None)
                    _target_agent.instructions._cache.pop("micro_system_prompt.md", None)
                log.info("Peer agents updated", count=len(agents_list), fd_url=fd_url, has_session=bool(_target_agent.session), public=bool(_pub_sid))

    elif msg_type == "approval_response":
        request_id = str(data.get("id", ""))
        approved = bool(data.get("approved", False))
        if request_id:
            server.resolve_playbook_approval(request_id, approved)
