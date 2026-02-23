"""Telegram bridge integration for the web server."""

from __future__ import annotations

import asyncio
import json
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from captain_claw.logging import get_logger
from captain_claw.telegram_bridge import TelegramBridge, TelegramMessage

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def start_telegram(server: WebServer) -> None:
    """Initialize and start Telegram polling + worker if configured."""
    if not server._telegram_enabled:
        return
    token = server.config.telegram.bot_token.strip()
    if not token:
        log.info("Telegram enabled but bot_token is empty; skipping in web mode.")
        return
    server._telegram_bridge = TelegramBridge(
        token=token, api_base_url=server.config.telegram.api_base_url,
    )
    if server.agent:
        server._approved_telegram_users = await _tg_load_state(
            server, "telegram_approved_users"
        )
        server._pending_telegram_pairings = await _tg_load_state(
            server, "telegram_pending_pairings"
        )
        _tg_cleanup_expired(server)
        await _tg_save_state(server)
    server._telegram_poll_task = asyncio.create_task(_telegram_poll_loop(server))
    server._telegram_worker_task = asyncio.create_task(_telegram_worker(server))
    log.info("Telegram bridge started in web mode (long polling).")
    print("  Telegram integration active (long polling).")


async def stop_telegram(server: WebServer) -> None:
    """Gracefully stop Telegram tasks."""
    for task in (server._telegram_poll_task, server._telegram_worker_task):
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    if server._telegram_bridge:
        await server._telegram_bridge.close()


async def _telegram_poll_loop(server: WebServer) -> None:
    """Background Telegram long-polling loop."""
    assert server._telegram_bridge is not None
    poll_timeout = max(1, int(server.config.telegram.poll_timeout_seconds))
    while True:
        try:
            updates = await server._telegram_bridge.get_updates(
                offset=server._telegram_offset, timeout=poll_timeout,
            )
            for update in updates:
                next_offset = int(update.update_id) + 1
                server._telegram_offset = (
                    next_offset
                    if server._telegram_offset is None
                    else max(server._telegram_offset, next_offset)
                )
                await server._telegram_queue.put(update)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.error("Telegram poll error", error=str(exc))
            await asyncio.sleep(2.0)


async def _telegram_worker(server: WebServer) -> None:
    """Dispatch queued Telegram messages as concurrent tasks."""
    while True:
        try:
            message = await server._telegram_queue.get()
            asyncio.create_task(_handle_telegram_message(server, message))
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.error("Telegram worker error", error=str(exc))


async def _handle_telegram_message(server: WebServer, message: TelegramMessage) -> None:
    """Process a single Telegram message."""
    bridge = server._telegram_bridge
    if not bridge or not server.agent:
        return
    try:
        if message.business_connection_id:
            try:
                await bridge.read_business_message(
                    message.business_connection_id, message.chat_id, message.message_id,
                )
            except Exception:
                pass

        user_id_key = str(message.user_id)
        if user_id_key not in server._approved_telegram_users:
            await _tg_pair_unknown_user(server, message)
            return

        text = message.text.strip()
        if not text:
            return

        if text.startswith("/") and "@" in text.split()[0]:
            parts = text.split(None, 1)
            command_word = parts[0].split("@")[0]
            text = command_word if len(parts) == 1 else f"{command_word} {parts[1]}"

        lowered = text.lower()
        if lowered == "/start" or lowered.startswith("/start "):
            await _tg_send(
                server, message.chat_id,
                "Captain Claw connected (web mode).\nSend plain text to chat.",
                reply_to_message_id=message.message_id,
            )
            return
        if lowered == "/help" or lowered.startswith("/help "):
            await _tg_send(
                server, message.chat_id,
                "Send any text to chat with the current session.\n"
                "Full command support is available in the Web UI.",
                reply_to_message_id=message.message_id,
            )
            return

        if text.startswith("/"):
            result = await execute_telegram_command(server, text)
            if result is not None:
                await _tg_send(
                    server, message.chat_id, result,
                    reply_to_message_id=message.message_id,
                )
                return

        user_label = message.username or message.first_name or str(message.user_id)
        server._broadcast({
            "type": "chat_message",
            "role": "user",
            "content": f"[TG {user_label}] {text}",
        })

        await _tg_process_with_typing(server, message.chat_id, text, message.message_id)

    except Exception as exc:
        log.error("Telegram message handler failed", error=str(exc))
        try:
            await _tg_send(
                server, message.chat_id,
                f"Error while processing your request: {str(exc)}",
                reply_to_message_id=message.message_id,
            )
        except Exception:
            pass


async def execute_telegram_command(server: WebServer, raw: str) -> str | None:
    """Execute a slash command from Telegram, returning the response text.

    Returns ``None`` only when the command should fall through to the
    agent (e.g. ``/orchestrate``).
    """
    if not server.agent:
        return "Agent not ready."

    parts = raw.strip().split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    try:
        if cmd in ("/clear",):
            if server.agent.session:
                if server.agent.is_session_memory_protected():
                    return "Session memory is protected. Disable with /session protect off."
                server.agent.session.messages.clear()
                await server.agent.session_manager.save_session(server.agent.session)
                server.agent.last_usage = server.agent._empty_usage()
                server.agent.last_context_window = {}
                return "Session messages cleared."
            return "No active session."

        if cmd in ("/config",):
            details = server.agent.get_runtime_model_details()
            return (
                f"Model: {details.get('provider', '')}:{details.get('model', '')}\n"
                f"Session: {server.agent.session.name if server.agent.session else 'none'}\n"
                f"Pipeline: {server.agent.pipeline_mode}"
            )

        if cmd in ("/history",):
            if server.agent.session:
                msgs = server.agent.session.messages[-20:]
                if not msgs:
                    return "Session history is empty."
                lines = []
                for i, m in enumerate(msgs, 1):
                    role = m.get("role", "?")
                    content = str(m.get("content", ""))[:150].replace("\n", " ")
                    lines.append(f"{i}. {role}: {content}")
                return "\n".join(lines)
            return "No active session."

        if cmd in ("/compact",):
            if server.agent.session:
                compacted, stats = await server.agent.compact_session(force=True, trigger="telegram")
                if compacted:
                    return (
                        f"Session compacted ({int(stats.get('before_tokens', 0))} "
                        f"-> {int(stats.get('after_tokens', 0))} tokens)"
                    )
                return f"Compaction skipped: {stats.get('reason', 'not_needed')}"
            return "No active session."

        if cmd in ("/new",):
            name = args.strip() or "default"
            session = await server.agent.session_manager.create_session(name=name)
            server.agent.session = session
            server.agent.refresh_session_runtime_flags()
            await server.agent.session_manager.set_last_active_session(session.id)
            server.agent.last_usage = server.agent._empty_usage()
            server.agent.last_context_window = {}
            server._broadcast({"type": "session_info", **server._session_info()})
            return f"New session: {session.name} ({session.id[:12]})"

        if cmd in ("/session",):
            if not args.strip():
                if not server.agent.session:
                    return "No active session."
                details = server.agent.get_runtime_model_details()
                return (
                    f"Session: {server.agent.session.name}\n"
                    f"ID: {server.agent.session.id}\n"
                    f"Messages: {len(server.agent.session.messages)}\n"
                    f"Model: {details.get('provider')}/{details.get('model')}"
                )
            from captain_claw.web.slash_commands import handle_session_subcommand
            return await handle_session_subcommand(server, args.strip())

        if cmd in ("/sessions",):
            sessions = await server.agent.session_manager.list_sessions(limit=20)
            if not sessions:
                return "No sessions found."
            lines = ["Sessions:"]
            for i, s in enumerate(sessions, 1):
                marker = "*" if (server.agent.session and s.id == server.agent.session.id) else " "
                lines.append(f"{marker} [{i}] {s.name} ({s.id}) messages={len(s.messages)}")
            return "\n".join(lines)

        if cmd in ("/models",):
            models = server.agent.get_allowed_models()
            details = server.agent.get_runtime_model_details()
            lines = ["Allowed models:"]
            for i, m in enumerate(models, 1):
                marker = ""
                if (
                    str(m.get("provider", "")).strip() == str(details.get("provider", "")).strip()
                    and str(m.get("model", "")).strip() == str(details.get("model", "")).strip()
                ):
                    marker = " *"
                lines.append(f"[{i}] {m.get('id')} -> {m.get('provider')}/{m.get('model')}{marker}")
            return "\n".join(lines)

        if cmd in ("/pipeline",):
            if args.strip():
                mode = args.strip().lower()
                if mode in ("loop", "contracts"):
                    await server.agent.set_pipeline_mode(mode)
                    return f"Pipeline mode set to {mode}."
                return "Invalid mode. Use /pipeline loop|contracts"
            return f"Pipeline mode: {server.agent.pipeline_mode}"

        if cmd in ("/planning",):
            if args.strip().lower() == "on":
                await server.agent.set_pipeline_mode("contracts")
                return "Pipeline mode set to contracts."
            if args.strip().lower() == "off":
                await server.agent.set_pipeline_mode("loop")
                return "Pipeline mode set to loop."
            return f"Planning: {'on' if server.agent.pipeline_mode == 'contracts' else 'off'}"

        if cmd in ("/skills",):
            skills = server.agent.list_user_invocable_skills()
            if not skills:
                return "No user-invocable skills available."
            lines = ["Available skills:"]
            for s in skills:
                lines.append(f"- /skill {s.name}")
            return "\n".join(lines)

        if cmd in ("/monitor",):
            return "Monitor is available in the Web UI."

        if cmd in ("/exit", "/quit"):
            return "/exit is only available on the server terminal."

        if cmd in ("/approve",):
            return await handle_approve_command(server, args.strip())

        if cmd in ("/orchestrate",):
            return None

        return f"Unknown command: {cmd}\nUse /help for available commands."

    except Exception as e:
        return f"Command error: {e}"


async def _tg_process_with_typing(
    server: WebServer, chat_id: int, text: str, reply_to_message_id: int | None = None,
) -> None:
    """Run agent.complete() with Telegram typing indicator, respecting busy state."""
    bridge = server._telegram_bridge
    if not bridge or not server.agent:
        return

    if server._busy:
        await _tg_send(
            server, chat_id,
            "Agent is busy processing another request. Your message is queued\u2026",
            reply_to_message_id=reply_to_message_id,
        )

    async with server._busy_lock:
        server._busy = True
        server._broadcast({"type": "status", "status": "thinking"})

        stop_typing = asyncio.Event()

        async def _typing_heartbeat() -> None:
            while not stop_typing.is_set():
                try:
                    await bridge.send_chat_action(chat_id=chat_id, action="typing")
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(stop_typing.wait(), timeout=4.0)
                except TimeoutError:
                    continue

        heartbeat = asyncio.create_task(_typing_heartbeat())
        try:
            stripped = text.strip()
            if stripped.lower().startswith("/orchestrate ") and server._orchestrator:
                orchestrate_input = stripped[len("/orchestrate "):].strip()
                if not orchestrate_input:
                    await _tg_send(
                        server, chat_id, "Usage: /orchestrate <request>",
                        reply_to_message_id=reply_to_message_id,
                    )
                else:
                    response = await server._orchestrator.orchestrate(orchestrate_input)
                    await _tg_send(server, chat_id, response, reply_to_message_id=reply_to_message_id)
                    server._broadcast({
                        "type": "chat_message",
                        "role": "assistant",
                        "content": response,
                    })
            else:
                response = await server.agent.complete(text)

                await _tg_send(server, chat_id, response, reply_to_message_id=reply_to_message_id)

                server._broadcast({
                    "type": "chat_message",
                    "role": "assistant",
                    "content": response,
                })
            server._broadcast({
                "type": "usage",
                "last": server.agent.last_usage,
                "total": server.agent.total_usage,
            })
            server._broadcast({
                "type": "session_info",
                **server._session_info(),
            })
        except Exception as exc:
            log.error("Telegram agent error", error=str(exc))
            await _tg_send(
                server, chat_id, f"Error: {str(exc)}", reply_to_message_id=reply_to_message_id,
            )
            server._broadcast({"type": "error", "message": f"Telegram error: {str(exc)}"})
        finally:
            stop_typing.set()
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            server._busy = False
            server._broadcast({"type": "status", "status": "ready"})


# ── Telegram helpers ──────────────────────────────────────────────


async def _tg_send(
    server: WebServer, chat_id: int, text: str, *, reply_to_message_id: int | None = None,
) -> None:
    if server._telegram_bridge:
        try:
            await server._telegram_bridge.send_message(
                chat_id=chat_id, text=text, reply_to_message_id=reply_to_message_id,
            )
        except Exception as exc:
            log.error("Telegram send failed", error=str(exc))


async def _tg_load_state(server: WebServer, key: str) -> dict[str, dict[str, object]]:
    if not server.agent:
        return {}
    raw = await server.agent.session_manager.get_app_state(key)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _tg_save_state(server: WebServer) -> None:
    if not server.agent:
        return
    await server.agent.session_manager.set_app_state(
        "telegram_approved_users",
        json.dumps(server._approved_telegram_users, ensure_ascii=True, sort_keys=True),
    )
    await server.agent.session_manager.set_app_state(
        "telegram_pending_pairings",
        json.dumps(server._pending_telegram_pairings, ensure_ascii=True, sort_keys=True),
    )


def _tg_cleanup_expired(server: WebServer) -> None:
    now_ts = datetime.now(UTC).timestamp()
    expired = []
    for token, payload in server._pending_telegram_pairings.items():
        expires_at_raw = str(payload.get("expires_at", "")).strip()
        if not expires_at_raw:
            continue
        try:
            expires_at = datetime.fromisoformat(
                expires_at_raw.replace("Z", "+00:00")
            )
            if expires_at.timestamp() <= now_ts:
                expired.append(token)
        except Exception:
            expired.append(token)
    for token in expired:
        server._pending_telegram_pairings.pop(token, None)


def _tg_generate_pairing_token(server: WebServer) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        token = "".join(secrets.choice(alphabet) for _ in range(8))
        if token not in server._pending_telegram_pairings:
            return token


async def _tg_pair_unknown_user(server: WebServer, message: TelegramMessage) -> None:
    _tg_cleanup_expired(server)
    user_id_key = str(message.user_id)
    if user_id_key in server._approved_telegram_users:
        return
    existing_token = ""
    for token, payload in server._pending_telegram_pairings.items():
        if str(payload.get("user_id", "")).strip() == str(message.user_id):
            existing_token = token
            break
    if not existing_token:
        existing_token = _tg_generate_pairing_token(server)
        ttl_minutes = max(1, int(server.config.telegram.pairing_ttl_minutes))
        expires = datetime.now(UTC).timestamp() + ttl_minutes * 60
        expires_dt = datetime.fromtimestamp(expires, tz=UTC)
        server._pending_telegram_pairings[existing_token] = {
            "user_id": message.user_id,
            "chat_id": message.chat_id,
            "username": message.username,
            "first_name": message.first_name,
            "created_at": datetime.now(UTC).isoformat(),
            "expires_at": expires_dt.isoformat(),
        }
        await _tg_save_state(server)

    await _tg_send(
        server, message.chat_id,
        (
            "Pairing required.\n"
            f"Your pairing token: `{existing_token}`\n\n"
            "Ask the Captain Claw operator to approve you with:\n"
            f"/approve user telegram {existing_token}"
        ),
        reply_to_message_id=message.message_id,
    )
    server._broadcast({
        "type": "chat_message",
        "role": "system",
        "content": (
            f"Telegram pairing request from "
            f"{message.username or message.first_name or message.user_id}. "
            f"Token: {existing_token}"
        ),
    })


async def handle_approve_command(server: WebServer, args: str) -> str:
    """Handle /approve user telegram <token>."""
    parts = args.split()
    if len(parts) < 3 or parts[0].lower() != "user" or parts[1].lower() != "telegram":
        return "Usage: `/approve user telegram <token>`"
    token = parts[2].strip().upper()
    if not token:
        return "Usage: `/approve user telegram <token>`"
    _tg_cleanup_expired(server)
    record = server._pending_telegram_pairings.get(token)
    if not isinstance(record, dict):
        return f"Telegram pairing token not found or expired: `{token}`"
    user_id = str(record.get("user_id", "")).strip()
    if not user_id:
        server._pending_telegram_pairings.pop(token, None)
        await _tg_save_state(server)
        return f"Telegram pairing token invalid: `{token}`"

    server._approved_telegram_users[user_id] = {
        "user_id": int(record.get("user_id", 0) or 0),
        "chat_id": int(record.get("chat_id", 0) or 0),
        "username": str(record.get("username", "")).strip(),
        "first_name": str(record.get("first_name", "")).strip(),
        "approved_at": datetime.now(UTC).isoformat(),
        "token": token,
    }
    server._pending_telegram_pairings.pop(token, None)
    await _tg_save_state(server)

    chat_id = int(server._approved_telegram_users[user_id].get("chat_id", 0) or 0)
    if chat_id and server._telegram_bridge:
        await _tg_send(
            server, chat_id,
            "Pairing approved. You can now use Captain Claw.\nSend any text to chat.",
        )

    username = str(record.get("username", "")).strip()
    first_name = str(record.get("first_name", "")).strip()
    label = username or first_name or user_id
    return f"Approved Telegram user: **{label}**"
