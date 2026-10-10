"""WebSocket protocol handler for the web UI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw import mail_authority
from captain_claw.agent import Agent
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def ws_handler(server: WebServer, request: web.Request) -> web.WebSocketResponse:
    """Handle WebSocket connections.

    ``heartbeat=20`` makes aiohttp send a ping every 20s when idle so
    intermediaries (FD proxy, reverse proxies) don't drop the connection.
    """
    ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024, heartbeat=20)
    await ws.prepare(request)

    from captain_claw.config import get_config
    cfg = get_config()
    public_mode = bool(cfg.web.public_run)

    # ── Shared-agent member (Flight Deck speaker handshake) ───────
    # A verified X-FD-Speaker socket is a different principal: its own
    # instance, its own session, never in `clients` / `_lane_sockets`.
    # Without the header nothing below changes.
    _speaker_header = request.headers.get("X-FD-Speaker")
    if _speaker_header:
        from captain_claw.web.speaker_ws import speaker_ws_session
        return await speaker_ws_session(server, ws, request, _speaker_header, public_mode)

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

    # ── Lane binding ──────────────────────────────────────────────
    # `?lane=B` puts this socket on a parallel context with its own agent,
    # session and busy flag. Absent or unrecognised → lane A, which IS the
    # shared main agent, so every existing client is unaffected.
    lane = server.normalize_lane(request.query.get("lane", ""))
    # A public visitor has one session of their own; they never watch a lane
    # (the automation lane carries the owner's scheduled and autonomous work).
    if getattr(ws, "_public_session_id", None):
        lane = server.LANE_MAIN
    ws._lane = lane  # type: ignore[attr-defined]
    server._lane_sockets.setdefault(lane, set()).add(ws)

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
        # A lane socket must see ITS lane's session in the welcome payload and
        # in the replay below — otherwise lane B opens showing lane A's name,
        # model and entire history.
        _welcome_agent = await server.resolve_agent(ws)
        session_info = server._session_info(_welcome_agent)
        _sess_meta = {}
        if _welcome_agent and _welcome_agent.session:
            _sess_meta = _welcome_agent.session.metadata or {}

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
    elif _welcome_agent and _welcome_agent.session:
        replay_session = _welcome_agent.session

    if replay_session:
        batch = _build_replay_batch(replay_session)
        if batch:
            await server._send(ws, {"type": "replay_batch", "messages": batch})
        await server._send(ws, {"type": "replay_done"})

    # ConnectionResetError comes from aiohttp's internal PONG-on-PING write
    # when the peer transport is already closing (FD page refresh, network
    # blip). It surfaces through `async for raw_msg in ws` and would
    # otherwise dump a 500-style traceback for what is just a clean
    # disconnect — treat it as end-of-iteration.
    from aiohttp.client_exceptions import ClientConnectionResetError
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
    except (ClientConnectionResetError, ConnectionResetError, ConnectionError):
        # Peer hung up; nothing to do but exit the receive loop quietly.
        pass
    finally:
        server.clients.discard(ws)
        server._lane_sockets.get(getattr(ws, "_lane", ""), set()).discard(ws)

    return ws


_FLEET_EVENTS_KEPT = 40

# A chat frame's ``attachment_notes``: at most this many, each cut to this length.
_MAX_ATTACHMENT_NOTES = 20
_MAX_ATTACHMENT_NOTE_CHARS = 8000  # chat_handler trims to 4000, marked


def _attachment_notes(raw: object) -> list[str]:
    """The frame's ``attachment_notes`` as a capped list of strings
    (non-strings and blank entries are dropped; chat_handler sanitises)."""
    if not isinstance(raw, list):
        return []
    notes: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            continue
        notes.append(item[:_MAX_ATTACHMENT_NOTE_CHARS])
        if len(notes) >= _MAX_ATTACHMENT_NOTES:
            break
    return notes


async def _fleet_notice_to_automation_lane(server, text: str) -> bool:
    """Keep a fleet notice out of the main chat when the automation lane is
    on: the notice is stored in that lane's session (where the user can read
    it) and the main session keeps only the event, for the one-line "fleet
    changes" note. False when there is no automation lane."""
    import os
    from datetime import datetime, timezone

    from captain_claw.agent_reasoning_mixin import _is_fd_spawned_worker
    from captain_claw.config import get_config

    lane = server.normalize_lane(get_config().session.automation_lane or "")
    if lane == server.LANE_MAIN:
        return False
    # FD workers and beings keep notices where they were (no lane of their own).
    if _is_fd_spawned_worker() or str(os.environ.get("CLAW_BEING_WORKER", "")).strip().lower() in (
            "1", "true", "yes"):
        return False
    try:
        auto = await server._get_lane_agent(lane)
    except Exception as exc:
        log.warning("Automation lane unavailable for a fleet notice", error=str(exc))
        return False
    if getattr(auto, "session", None) is None:
        return False
    auto.session.add_message("user", text, origin="fleet_notice")
    events = server.agent.session.metadata.setdefault("fleet_events", [])
    events.append({"text": text, "at": datetime.now(timezone.utc).isoformat()})
    del events[: max(0, len(events) - _FLEET_EVENTS_KEPT)]
    try:      # the main session saves with its next turn, as notices always did
        await auto.session_manager.save_session(auto.session)
    except Exception as exc:
        log.debug("Could not save the automation lane after a fleet notice", error=str(exc))
    return True


def _build_replay_batch(session) -> list[dict]:
    """The ``replay_batch`` messages that rebuild *session*'s transcript in a
    freshly connected client (chat, rephrase panels, monitor cards)."""
    batch: list[dict] = []
    for msg in session.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_name = msg.get("tool_name", "")
        timestamp = msg.get("timestamp", "")
        model = msg.get("model", "")
        if role in ("user", "assistant"):
            if msg.get("origin_detail") in ("cut_off", "continue_cut_off"):
                continue        # plumbing of a continued answer; the joined reply follows
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
            # Where the message came from (human, corrective, fleet notice…),
            # so a client can label synthetic rows instead of showing them as
            # something the user typed.
            from captain_claw import msg_origin

            payload["origin"] = msg_origin.origin_of(msg)
            if msg.get("channel"):
                payload["channel"] = msg["channel"]
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
    return batch


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
                # Another agent's result, not the Telegram user typing.
                with mail_authority.bound(mail_authority.automated("peer_relay", "", "deny")):
                    from captain_claw import msg_origin as _msg_origin

                    _msg_origin.hint_turn_provenance(
                        tg_agent, turn_origin="delegated_result", channel="telegram",
                    )
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
    # Shared-agent member: frame allowlist + member-only handling first.
    from captain_claw.speaker import speaker_key_of
    if speaker_key_of(ws) is not None:
        from captain_claw.web.speaker_ws import speaker_gate
        if not await speaker_gate(server, ws, data):
            return
    msg_type = data.get("type", "")
    # Every frame resets it: a turn this frame starts (even via /code or the
    # plan-auto route) echoes the frame's own id, never a previous frame's.
    from captain_claw.web.chat_handler import set_frame_client_msg_id
    set_frame_client_msg_id(str(data.get("client_msg_id") or "").strip()[:64])

    if msg_type == "chat":
        content = str(data.get("content", "")).strip()
        rewind_to = str(data.get("rewind_to", "")).strip() or None
        # When the message arrived over WhatsApp, the bridge tags it with the
        # originating WAID so tools can target "the current WhatsApp chat".
        whatsapp_waid = str(data.get("whatsapp_waid", "")).strip() or None
        # Durable origin descriptor ({kind, address}) so an async/cron result
        # can be routed back to this source later. Bridges may send it
        # explicitly; otherwise handle_chat synthesizes one from whatsapp_waid.
        origin = data.get("origin") if isinstance(data.get("origin"), dict) else None
        # The chat surface this frame came from (glasses / whatsapp /
        # messenger), sent by the bridges on every frame: the turn renders
        # that surface's rules, and only on that surface's turns.
        surface = str(data.get("surface", "") or "").strip().lower() or None
        # A bridge talks over its own socket: remember its surface for the
        # frames that don't carry one (slash-routed turns, older bridges,
        # whose first message carries the surface rules block instead).
        if not surface:
            from captain_claw import msg_origin as _msg_origin

            if _msg_origin.SURFACE_BLOCK_RE.search(content):
                _wa = whatsapp_waid or str((origin or {}).get("kind", "")).lower() == "whatsapp"
                surface = "whatsapp" if _wa else "glasses"
            else:
                surface = getattr(ws, "_claw_surface", None)
        if surface:
            try:
                ws._claw_surface = surface
            except Exception:
                pass

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
        # Lines a bridge wrote about what it attached (a voice note's length,
        # a file it couldn't fetch, …). They reach the model beside the
        # attachments — never as the user's words.
        attachment_notes = _attachment_notes(data.get("attachment_notes"))
        # A bridge's id for this frame, echoed in a busy refusal (retry exactly it).
        client_msg_id = str(data.get("client_msg_id") or "").strip()[:64]
        # A frame carrying attachments is always a chat turn: a caption that
        # starts with "/" is not a command, and neither route takes files.
        has_attachments = bool(image_paths or file_paths or attachment_notes)

        # Automated-turn marker (Flight Deck: autonomy, scheduler, flows,
        # peers …). Absent = a human turn. An automated frame never runs a
        # slash command or the plan-auto route — it goes to handle_chat,
        # which binds it for the mail-write guard.
        automation = (
            mail_authority.from_wire(data.get("automation"), default_kind="unknown")
            if "automation" in data else None
        )
        # Flight Deck delivers this automated turn's result itself (a nudge's
        # push, a scheduled job's delivery): its mirror into the main chat is
        # shown there but not relayed again by the chat bridges.
        fd_delivers = bool(isinstance(data.get("automation"), dict)
                           and data["automation"].get("fd_delivers") is True)

        if not content and not has_attachments:
            return
        # "Nova tema: …" / "new topic" opening a typed message: a new session
        # first (as /new), then the rest of the message is the turn. Never for
        # a public visitor (their /new would reach the owner's session).
        if (automation is None and not content.startswith("/")
                and not getattr(ws, "_public_session_id", None)):
            from captain_claw import msg_origin as _cue_origin
            from captain_claw.web.slash_commands import handle_command, rotation_cue

            # A bridge's first message carries the surface rules block ahead
            # of what the person wrote: the cue is in the person's words.
            _block, _said = _cue_origin.split_surface_block(content)
            rest = rotation_cue(_said)
            if rest is not None:
                _cue_agent = await server.resolve_agent(ws)
                _cue_busy = (getattr(_cue_agent, "_lane_busy", False)
                             if _cue_agent is not server.agent else server._busy)
                if _cue_busy:
                    from captain_claw.web.chat_handler import busy_refusal_fields
                    await server._send(ws, {
                        "type": "error",
                        "message": "Still answering the previous message — start the new "
                                   "session once it is done (or stop it first).",
                        **busy_refusal_fields(client_msg_id),
                    })
                    return
                # On this socket's own agent: a lane's or lane A.
                await handle_command(server.lane_view(ws, _cue_agent), ws, "/new")
                content = (_block + rest) if (_block and rest) else rest
                if rest:
                    # The client cleared its transcript for the new session;
                    # what the person asked goes back in.
                    await server._send(ws, {"type": "chat_message", "role": "user", "content": rest,
                                            "rotation_cue": True})
                if not rest and not has_attachments:
                    return
        if automation is None and not has_attachments and content.startswith("/"):
            from captain_claw.web.slash_commands import handle_command
            # On this socket's own agent, as a "command" frame is: a lane's
            # /new must not switch lane A's session.
            _cmd_agent = await server.resolve_agent(ws)
            await handle_command(server.lane_view(ws, _cmd_agent), ws, content)
        elif (automation is None and not has_attachments
              and getattr(server.agent, "plan_mode_auto", False)):
            from captain_claw.web.plan_auto_route import handle_plan_auto_route
            with mail_authority.bound(mail_authority.interactive(content)):
                await handle_plan_auto_route(server, ws, content)
        else:
            from captain_claw.web.chat_handler import handle_chat
            await handle_chat(
                server, ws, content,
                image_path=image_paths[0] if len(image_paths) == 1 else None,
                file_path=file_paths[0] if len(file_paths) == 1 else None,
                image_paths=image_paths if len(image_paths) > 1 else None,
                file_paths=file_paths if len(file_paths) > 1 else None,
                attachment_notes=attachment_notes or None,
                client_msg_id=client_msg_id,
                rewind_to=rewind_to,
                whatsapp_waid=whatsapp_waid,
                origin=origin,
                surface=surface,
                no_flow=bool(data.get("no_flow", False)),
                deny_tools=[str(t) for t in (data.get("deny_tools") or [])],
                no_tools=bool(data.get("no_tools", False)),
                no_broadcast=bool(data.get("no_broadcast", False)),
                # Queue-dispatched turns skip the post-turn "what next?" call.
                no_next_steps=bool(data.get("no_next_steps", False)),
                no_rephrase=bool(data.get("no_rephrase", False)),
                automation=automation,
                fd_delivers=fd_delivers,
                # A scheduled job delivering to WhatsApp: where its files go.
                whatsapp_media_to=str(data.get("whatsapp_media_to", "") or "").strip() or None,
            )

    elif msg_type == "run_tool":
        # Deterministic, structured tool invocation (the autonomous action rail).
        # Run ONE named tool with structured args through the guard and return the
        # ToolResult — no LLM. Used by the action catalog / autonomous loop.
        req_id = str(data.get("req_id", ""))
        tool = str(data.get("tool", "")).strip()
        args = data.get("args") if isinstance(data.get("args"), dict) else {}
        if not tool or not server.agent:
            await server._send(ws, {"type": "tool_result", "req_id": req_id,
                                    "ok": False, "error": "no tool or agent"})
            return
        try:
            # Refresh the Google-connected cache. The registry auto-hides Google
            # tools when that flag is stale/False, and it's normally refreshed only
            # during an agent turn — which run_tool skips — so a connected agent
            # would otherwise look disconnected and get policy-blocked.
            try:
                from captain_claw.google_oauth_manager import GoogleOAuthManager
                await GoogleOAuthManager(server.agent.session_manager).is_connected()
            except Exception:
                pass
            # The action catalog is the authority for a deterministic invocation,
            # so explicitly allow the named tool past the per-session tool policy
            # (which exists to constrain the LLM mid-turn). The script_tool guard
            # still runs; if the tool lacks creds it errors normally.
            # No marker = deny mail writes (fail closed); FD sends
            # mail_write="allow" only for a human-approved action.
            _auth = (
                mail_authority.from_wire(data.get("automation"), default_kind="autonomy_tool")
                or mail_authority.automated("autonomy_tool", "", "deny")
            )
            with mail_authority.bound(_auth):
                res = await server.agent._execute_tool_with_guard(
                    tool, args, "autonomous-action",
                    task_policy={"also_allow": [tool]},
                )
            await server._send(ws, {
                "type": "tool_result", "req_id": req_id,
                "ok": bool(getattr(res, "success", False)),
                "content": getattr(res, "content", "") or "",
                "error": getattr(res, "error", None),
            })
        except Exception as exc:
            log.warning("run_tool failed", tool=tool, error=str(exc))
            await server._send(ws, {"type": "tool_result", "req_id": req_id,
                                    "ok": False, "error": str(exc)})

    elif msg_type == "command":
        command = str(data.get("command", "")).strip()
        if command:
            from captain_claw.web.slash_commands import handle_command
            # On a lane, `/new`, `/model`, `/planning` … must act on THAT
            # lane's agent and reply into that lane, not lane A's.
            _cmd_agent = await server.resolve_agent(ws)
            await handle_command(server.lane_view(ws, _cmd_agent), ws, command)

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
                # Process with the telegram agent and send result to telegram;
                # the turn records the result as its opening message (writing
                # it here as well stored every relayed result twice).
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

        # Broadcast to UI so the inbound message appears in the chat.
        from datetime import datetime, timezone
        server._broadcast({
            "type": "chat_message",
            "role": "user",
            "content": notif_content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notification": True,
        })

        if trigger and not _pub_sid:
            # Route ALL triggered peer notifications through the serialized
            # inbound queue — the single consumer drains them one-at-a-time only
            # when the agent is free. This replaces the old "route directly when
            # free / append to a list when busy" split, which raced and caused
            # duplicate "waiting" replies + re-planning.
            server._inbound_queue.put_nowait(notif_content)
            log.info("Triggered notification enqueued", content_len=len(notif_content),
                     queue_size=server._inbound_queue.qsize(), agent_busy=server._busy)
            return

        # No trigger requested — inject silently into session history.
        _target_agent = await server.resolve_agent(ws)
        if _target_agent and _target_agent.session:
            from captain_claw import msg_origin

            _found = msg_origin.detect_literal(notif_content)
            _fleet = bool(_found and _found[0] == "fleet_notice")
            if _fleet and _target_agent is server.agent and await _fleet_notice_to_automation_lane(
                    server, notif_content):
                return
            _target_agent.session.add_message(
                "user", notif_content,
                origin="fleet_notice" if _fleet else "notification",
            )
            log.info("Notification injected into session", content_len=len(notif_content),
                     agent_busy=server._busy, trigger=trigger)

    elif msg_type == "btw":
        # Inject additional instructions while a task is running.
        btw_content = str(data.get("content", "")).strip()
        _pub_sid = getattr(ws, "_public_session_id", None)
        _target_agent = await server.resolve_agent(ws)
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
        _target_agent = await server.resolve_agent(ws)
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
        _fb_agent = await server.resolve_agent(ws)
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
        # (A socket whose automated turn moved to the automation lane is on
        # that lane for the turn, so this cancels that turn.)
        _cancel_agent = await server.resolve_agent(ws)
        if _cancel_agent and hasattr(_cancel_agent, "cancel_event"):
            _cancel_agent.cancel_event.set()
            log.info("Cancel signal received via WebSocket", public=bool(_pub_sid))

    elif msg_type == "session_settings":
        # Update session name, description, and/or instructions.
        # Works for both public and admin sessions.
        _pub_sid = getattr(ws, "_public_session_id", None)
        _is_ws_admin = getattr(ws, "_is_admin", False)
        _target_agent = await server.resolve_agent(ws)
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
                    "code": "not_allowed",
                    "message": "Session settings are locked by the administrator.",
                })
                return
            # A shared-agent member's values are persisted and go into every
            # system prompt of their session: bounded.
            from captain_claw.speaker import SESSION_SETTINGS_LIMITS
            _limits = SESSION_SETTINGS_LIMITS if speaker_key_of(ws) is not None else {}

            def _setting(key: str) -> str:
                val = data[key].strip()
                cap = _limits.get(key)
                return val[:cap].strip() if cap else val

            changed = False
            if "session_name" in data and isinstance(data["session_name"], str):
                val = _setting("session_name")
                if val:
                    session.metadata["session_display_name"] = val
                    changed = True
            if "session_description" in data and isinstance(data["session_description"], str):
                session.metadata["session_description"] = _setting("session_description")
                changed = True
            if "session_instructions" in data and isinstance(data["session_instructions"], str):
                session.metadata["session_instructions"] = _setting("session_instructions")
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
        _target_agent = await server.resolve_agent(ws)
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
