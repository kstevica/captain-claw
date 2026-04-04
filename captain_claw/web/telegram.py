"""Telegram bridge integration for the web server.

Each approved Telegram user gets a dedicated Agent instance bound to
their own session.  Agents are created lazily on first message and
cached for the lifetime of the server process.  Per-user asyncio locks
serialise requests from the same user while allowing different users to
run concurrently.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from captain_claw.instructions import InstructionLoader
from captain_claw.logging import get_logger
from captain_claw.session import Session, get_session_manager
from captain_claw.next_steps import extract_next_steps, next_steps_to_dicts
from captain_claw.telegram_bridge import TelegramBridge, TelegramCallbackQuery, TelegramMessage

if TYPE_CHECKING:
    from captain_claw.agent import Agent
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

# Commands disabled for Telegram users (per-user sessions, no switching).
_TG_DISABLED_COMMANDS = frozenset({"/new", "/sessions"})
_TG_DISABLED_SESSION_SUBCMDS = frozenset({"list", "switch", "load", "new"})


# ── Lifecycle ────────────────────────────────────────────────────


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
        server._telegram_user_sessions = await _tg_load_user_sessions(server)
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


# ── Polling / dispatch ───────────────────────────────────────────


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
    """Dispatch queued Telegram messages and callback queries as concurrent tasks."""
    while True:
        try:
            update = await server._telegram_queue.get()
            if isinstance(update, TelegramCallbackQuery):
                asyncio.create_task(_handle_telegram_callback_query(server, update))
            else:
                asyncio.create_task(_handle_telegram_message(server, update))
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.error("Telegram worker error", error=str(exc))


# ── Per-user agent management ────────────────────────────────────


async def _tg_get_or_create_agent(server: WebServer, message: TelegramMessage) -> Agent:
    """Return a cached Agent for this Telegram user, creating one if needed.

    Each approved Telegram user gets their own lightweight Agent that
    shares the main agent's provider, session manager and tool registry
    but has its own session, usage counters and runtime flags.
    """
    from captain_claw.agent import Agent as AgentCls

    user_id_key = str(message.user_id)

    # Fast path: already cached.
    if user_id_key in server._telegram_agents:
        return server._telegram_agents[user_id_key]

    # ── Resolve or create the user's session ──
    sm = get_session_manager()
    session: Session | None = None
    session_id = server._telegram_user_sessions.get(user_id_key)
    if session_id:
        session = await sm.load_session(session_id)

    if session is None:
        user_info = server._approved_telegram_users.get(user_id_key, {})
        username = str(user_info.get("username", "")).strip()
        first_name = str(user_info.get("first_name", "")).strip()
        label = username or first_name or user_id_key
        session = await sm.create_session(name=f"tg-{label}")
        server._telegram_user_sessions[user_id_key] = session.id
        await _tg_save_user_sessions(server)

    # ── Build a lightweight agent (mirrors AgentPool pattern) ──
    agent = AgentCls(
        provider=server.agent.provider,  # shared LLM provider
        status_callback=None,            # TG users don't drive the web status bar
        tool_output_callback=server._tool_output_callback,
        thinking_callback=server._thinking_callback,
    )
    agent.session = session
    agent.session_manager = sm
    agent._sync_runtime_flags_from_session()

    # Share deep-memory index if available.
    main_dm = getattr(server.agent, "_deep_memory", None)
    if main_dm is not None:
        agent._deep_memory = main_dm

    agent._user_id = user_id_key  # Per-user personality lookup
    agent._register_default_tools()
    agent.instructions = InstructionLoader()
    agent._initialized = True
    # NOT a worker — full interactive agent (keeps default max_iterations).

    server._telegram_agents[user_id_key] = agent

    # Ensure a per-user lock exists.
    if user_id_key not in server._telegram_user_locks:
        server._telegram_user_locks[user_id_key] = asyncio.Lock()

    log.info(
        "Created Telegram user agent",
        user_id=user_id_key, session_id=session.id, session_name=session.name,
    )
    return agent


def _tg_get_user_lock(server: WebServer, user_id: str) -> asyncio.Lock:
    """Return the per-user asyncio lock, creating it if necessary."""
    lock = server._telegram_user_locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        server._telegram_user_locks[user_id] = lock
    return lock


# ── Message handling ─────────────────────────────────────────────


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

        # Resolve per-user agent (creates on first use).
        user_agent = await _tg_get_or_create_agent(server, message)

        text = message.text.strip()

        # Download attached photo into the user's session media folder.
        image_path: str | None = None
        if message.photo_file_id:
            try:
                from datetime import UTC, datetime

                from captain_claw.config import get_config

                cfg = get_config()
                workspace = cfg.resolved_workspace_path()
                session_id = user_agent.session.id if user_agent.session else "uploads"
                stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
                dest_dir = workspace / "saved" / "media" / session_id
                dest = dest_dir / f"tg-photo-{stamp}.jpg"
                await bridge.download_file(message.photo_file_id, dest)
                image_path = str(dest)
                log.info("Telegram photo downloaded", path=image_path)
            except Exception as dl_exc:
                log.warning("Telegram photo download failed", error=str(dl_exc))

        # Download attached document (Word, PDF, Excel, PowerPoint, etc.).
        file_path: str | None = None
        if message.document_file_id:
            try:
                from datetime import UTC, datetime
                from pathlib import Path

                from captain_claw.config import get_config

                cfg = get_config()
                workspace = cfg.resolved_workspace_path()
                session_id = user_agent.session.id if user_agent.session else "uploads"
                stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
                original_name = message.document_file_name or "document"
                ext = Path(original_name).suffix.lower() if Path(original_name).suffix else ""
                safe_stem = "".join(
                    c if c.isalnum() or c in "-_." else "_"
                    for c in Path(original_name).stem
                )[:60]
                filename = f"tg-{safe_stem}-{stamp}{ext}"
                dest_dir = workspace / "saved" / "downloads" / session_id
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / filename
                await bridge.download_file(message.document_file_id, dest)
                file_path = str(dest)
                log.info("Telegram document downloaded", path=file_path, original=original_name)
            except Exception as dl_exc:
                log.warning("Telegram document download failed", error=str(dl_exc))

        # Extract contact information.
        contact_text: str | None = None
        if message.contact_phone:
            parts = [message.contact_first_name, message.contact_last_name]
            name = " ".join(p for p in parts if p).strip() or "Unknown"
            contact_text = f"[Contact: {name}, phone: {message.contact_phone}]"
            if message.contact_vcard:
                # Save vCard file for the agent to process.
                try:
                    from datetime import UTC, datetime
                    from pathlib import Path

                    from captain_claw.config import get_config

                    cfg = get_config()
                    workspace = cfg.resolved_workspace_path()
                    session_id = user_agent.session.id if user_agent.session else "uploads"
                    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
                    safe_name = "".join(
                        c if c.isalnum() or c in "-_." else "_" for c in name
                    )[:40]
                    dest_dir = workspace / "saved" / "downloads" / session_id
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    vcf_path = dest_dir / f"tg-contact-{safe_name}-{stamp}.vcf"
                    vcf_path.write_text(message.contact_vcard, encoding="utf-8")
                    contact_text += f"\n[Attached file: {vcf_path}]"
                    log.info("Telegram contact vCard saved", path=str(vcf_path))
                except Exception as vcf_exc:
                    log.warning("Telegram vCard save failed", error=str(vcf_exc))

        if not text and not image_path and not file_path and not contact_text:
            return

        # Strip @botname suffix from commands.
        if text.startswith("/") and "@" in text.split()[0]:
            parts = text.split(None, 1)
            command_word = parts[0].split("@")[0]
            text = command_word if len(parts) == 1 else f"{command_word} {parts[1]}"

        lowered = text.lower()

        # ── /btw: inject live instructions without waiting for the lock ──
        import re as _re
        _btw_m = _re.match(r"^/btw\s+([\s\S]+)", text, _re.IGNORECASE) or _re.match(r"^btw\s+([\s\S]+)", text, _re.IGNORECASE)
        if _btw_m:
            _btw_text = _btw_m.group(1).strip()
            if _btw_text:
                if not hasattr(user_agent, "_btw_instructions"):
                    user_agent._btw_instructions = []
                user_agent._btw_instructions.append(_btw_text)
                log.info("BTW instruction added (Telegram)", user_id=user_id_key, count=len(user_agent._btw_instructions))
                await _tg_send(
                    server, message.chat_id,
                    f"Noted for remaining steps: _{_btw_text}_",
                    reply_to_message_id=message.message_id,
                )
            return

        if lowered == "/start" or lowered.startswith("/start "):
            session_name = user_agent.session.name if user_agent.session else "none"
            await _tg_send(
                server, message.chat_id,
                f"Captain Claw connected.\nYour session: {session_name}\nSend plain text to chat.",
                reply_to_message_id=message.message_id,
            )
            return
        if lowered == "/help" or lowered.startswith("/help "):
            await _tg_send(
                server, message.chat_id,
                "Send any text to chat with your session.\n"
                "Commands: /clear, /history, /compact, /config, /session, /cron\n"
                "Each Telegram user has a dedicated session.",
                reply_to_message_id=message.message_id,
            )
            return

        if text.startswith("/"):
            result = await execute_telegram_command(server, text, user_agent)
            if result is not None:
                await _tg_send(
                    server, message.chat_id, result,
                    reply_to_message_id=message.message_id,
                )
                return

        # Build effective text with attachment context if present.
        effective_text = text
        attachment_lines: list[str] = []
        if image_path:
            attachment_lines.append(f"[Attached image: {image_path}]")
        if file_path:
            attachment_lines.append(f"[Attached file: {file_path}]")
        if contact_text:
            attachment_lines.append(contact_text)
        if attachment_lines:
            prefix = "\n".join(attachment_lines) + "\n"
            default_msg = "Please analyze this image." if image_path else "I've attached a file."
            if contact_text and not image_path and not file_path:
                default_msg = "I'm sharing this contact."
            effective_text = prefix + (text or default_msg)

        user_label = message.username or message.first_name or str(message.user_id)
        server._broadcast({
            "type": "chat_message",
            "role": "user",
            "content": f"[TG {user_label}] {effective_text}",
        })

        # Store the chat_id on the agent so delegate callbacks can route back here
        user_agent._telegram_chat_id = message.chat_id  # type: ignore[attr-defined]

        await _tg_process_with_typing(
            server, message.chat_id, effective_text, message.message_id,
            user_agent=user_agent,
            user_id=str(message.user_id),
        )

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


async def _handle_telegram_callback_query(server: WebServer, cbq: TelegramCallbackQuery) -> None:
    """Handle an inline keyboard button press from Telegram."""
    bridge = server._telegram_bridge
    if not bridge or not server.agent:
        return

    try:
        # Acknowledge the button press immediately.
        await bridge.answer_callback_query(cbq.callback_query_id)
    except Exception:
        pass

    try:
        user_id_key = str(cbq.user_id)
        if user_id_key not in server._approved_telegram_users:
            return

        # Decode the action from callback_data.
        # We store a key like "ns:0", "ns:1", etc. and look up the full
        # action from a per-user cache.
        action_text = ""
        if cbq.data.startswith("ns:"):
            idx_str = cbq.data[3:]
            cache_key = f"_tg_next_steps_{user_id_key}"
            cached_steps = getattr(server, cache_key, None)
            if isinstance(cached_steps, list):
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(cached_steps):
                        action_text = cached_steps[idx].get("action", "")
                except (ValueError, IndexError):
                    pass
            # Clean up the cache.
            if hasattr(server, cache_key):
                delattr(server, cache_key)

        if not action_text:
            return

        # Echo the selected action back so the user sees what prompt
        # is being executed (Telegram hides the button text after click).
        try:
            await bridge.send_message(cbq.chat_id, f"*Action:* {action_text}")
        except Exception:
            pass

        # Get or create the user agent and process the action.
        user_agent = await _tg_get_or_create_agent(server, TelegramMessage(
            update_id=cbq.update_id,
            message_id=cbq.message_id,
            chat_id=cbq.chat_id,
            user_id=cbq.user_id,
            username=cbq.username,
            first_name=cbq.first_name,
            text=action_text,
        ))

        user_label = cbq.username or cbq.first_name or str(cbq.user_id)
        server._broadcast({
            "type": "chat_message",
            "role": "user",
            "content": f"[TG {user_label}] {action_text}",
        })

        await _tg_process_with_typing(
            server, cbq.chat_id, action_text, None,
            user_agent=user_agent,
            user_id=user_id_key,
        )

    except Exception as exc:
        log.error("Telegram callback query handler failed", error=str(exc))


# ── Commands ─────────────────────────────────────────────────────


async def execute_telegram_command(
    server: WebServer, raw: str, user_agent: Agent,
) -> str | None:
    """Execute a slash command from Telegram.

    Returns ``None`` only when the command should fall through to the
    agent (e.g. ``/orchestrate``).

    Each Telegram user operates on their own *user_agent*. Commands that
    create or switch sessions are disabled.
    """
    parts = raw.strip().split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    session = user_agent.session

    try:
        # ── Disabled commands ────────────────────────────────────
        if cmd in _TG_DISABLED_COMMANDS:
            return "Not available on Telegram. Each user has a dedicated session."

        # ── /clear ───────────────────────────────────────────────
        if cmd in ("/clear",):
            if session:
                if session.metadata and session.metadata.get("memory_protection"):
                    return "Session memory is protected. Disable with /session protect off."
                session.messages.clear()
                session.metadata = {}
                await user_agent.session_manager.save_session(session)
                user_agent.refresh_session_runtime_flags()
                user_agent.last_usage = user_agent._empty_usage()
                user_agent.last_context_window = {}
                return "Session messages cleared."
            return "No active session."

        # ── /config ──────────────────────────────────────────────
        if cmd in ("/config",):
            details = user_agent.get_runtime_model_details()
            return (
                f"Model: {details.get('provider', '')}:{details.get('model', '')}\n"
                f"Session: {session.name if session else 'none'}\n"
                f"Pipeline: {user_agent.pipeline_mode}"
            )

        # ── /history ─────────────────────────────────────────────
        if cmd in ("/history",):
            if session:
                msgs = session.messages[-20:]
                if not msgs:
                    return "Session history is empty."
                lines = []
                for i, m in enumerate(msgs, 1):
                    role = m.get("role", "?")
                    content = str(m.get("content", ""))[:150].replace("\n", " ")
                    lines.append(f"{i}. {role}: {content}")
                return "\n".join(lines)
            return "No active session."

        # ── /compact ─────────────────────────────────────────────
        if cmd in ("/compact",):
            if session:
                compacted, stats = await user_agent.compact_session(
                    force=True, trigger="telegram",
                )
                if compacted:
                    return (
                        f"Session compacted ({int(stats.get('before_tokens', 0))} "
                        f"-> {int(stats.get('after_tokens', 0))} tokens)"
                    )
                return f"Compaction skipped: {stats.get('reason', 'not_needed')}"
            return "No active session."

        # ── /session ─────────────────────────────────────────────
        if cmd in ("/session",):
            if not args.strip():
                if not session:
                    return "No active session."
                details = user_agent.get_runtime_model_details()
                return (
                    f"Session: {session.name}\n"
                    f"ID: {session.id}\n"
                    f"Messages: {len(session.messages)}\n"
                    f"Model: {details.get('provider')}/{details.get('model')}"
                )
            return await _tg_handle_session_subcommand(server, args.strip(), user_agent)

        # ── /models ──────────────────────────────────────────────
        if cmd in ("/models",):
            models = user_agent.get_allowed_models()
            details = user_agent.get_runtime_model_details()
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
                    await user_agent.set_pipeline_mode(mode)
                    return f"Pipeline mode set to {mode}."
                return "Invalid mode. Use /pipeline loop|contracts"
            return f"Pipeline mode: {user_agent.pipeline_mode}"

        if cmd in ("/planning",):
            if args.strip().lower() == "on":
                await user_agent.set_pipeline_mode("contracts")
                return "Pipeline mode set to contracts."
            if args.strip().lower() == "off":
                await user_agent.set_pipeline_mode("loop")
                return "Pipeline mode set to loop."
            return f"Planning: {'on' if user_agent.pipeline_mode == 'contracts' else 'off'}"

        if cmd in ("/skills",):
            skills = user_agent.list_user_invocable_skills()
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

        if cmd in ("/todo",):
            from captain_claw.web.slash_commands import handle_todo_command
            return await handle_todo_command(server, args.strip())

        if cmd in ("/contacts",):
            from captain_claw.web.slash_commands import handle_contacts_command
            return await handle_contacts_command(server, args.strip())

        if cmd in ("/scripts",):
            from captain_claw.web.slash_commands import handle_scripts_command
            return await handle_scripts_command(server, args.strip())

        if cmd in ("/apis",):
            from captain_claw.web.slash_commands import handle_apis_command
            return await handle_apis_command(server, args.strip())

        if cmd in ("/cron",):
            return await _tg_handle_cron_command(server, args.strip(), user_agent)

        if cmd in ("/orchestrate",):
            return None

        return f"Unknown command: {cmd}\nUse /help for available commands."

    except Exception as e:
        return f"Command error: {e}"


async def _tg_handle_session_subcommand(
    server: WebServer, args: str, user_agent: Agent,
) -> str:
    """Handle /session subcommands for Telegram (restricted subset)."""
    parts = args.split(None, 1)
    subcmd = parts[0].lower()
    subargs = parts[1].strip() if len(parts) > 1 else ""
    session = user_agent.session

    if subcmd in _TG_DISABLED_SESSION_SUBCMDS:
        return "Not available on Telegram. Each user has a dedicated session."

    if subcmd in ("rename",):
        if not subargs:
            return "Usage: /session rename <new-name>"
        session.name = subargs
        await user_agent.session_manager.save_session(session)
        return f"Session renamed to {subargs}."

    if subcmd in ("description",):
        if not subargs:
            return "Usage: /session description <text>"
        if subargs.lower() == "auto":
            desc = await user_agent._auto_generate_session_description()
            return f"Auto-generated description: {desc}"
        session.metadata = session.metadata or {}
        session.metadata["description"] = subargs
        await user_agent.session_manager.save_session(session)
        return f"Description set to: {subargs}"

    if subcmd in ("model",):
        if not subargs:
            details = user_agent.get_runtime_model_details()
            return f"Active model: {details.get('provider', '')}:{details.get('model', '')}"
        await user_agent.set_session_model(subargs, persist=True)
        details = user_agent.get_runtime_model_details()
        return f"Model set to {details.get('provider', '')}:{details.get('model', '')}"

    if subcmd in ("protect",):
        if session:
            if subargs.lower() == "on":
                session.metadata = session.metadata or {}
                session.metadata["memory_protection"] = True
                await user_agent.session_manager.save_session(session)
                return "Memory protection enabled."
            elif subargs.lower() == "off":
                session.metadata = session.metadata or {}
                session.metadata["memory_protection"] = False
                await user_agent.session_manager.save_session(session)
                return "Memory protection disabled."
            return "Usage: /session protect on|off"
        return "No active session."

    if subcmd in ("export",):
        if not session:
            return "No active session."
        mode = subargs.strip().lower() or "all"
        from captain_claw.session_export import export_session_history

        try:
            written = export_session_history(
                mode=mode,
                session_id=session.id,
                session_name=session.name,
                messages=session.messages,
                saved_base_path=user_agent.tools.get_saved_base_path(create=True),
            )
        except Exception as e:
            return f"Export failed: {e}"
        if not written:
            return "No files exported."
        lines = [f"Exported {len(written)} file(s):"]
        for p in written:
            lines.append(f"- {p}")
        return "\n".join(lines)

    return f"Unknown session subcommand: {subcmd}"


async def _tg_handle_cron_command(
    server: WebServer, args: str, user_agent: Agent,
) -> str:
    """Handle /cron subcommands for Telegram."""
    from captain_claw.cron import compute_next_run, schedule_to_text, to_utc_iso
    from captain_claw.cron_dispatch import (
        cron_monitor_event,
        execute_cron_job,
        parse_cron_add_args,
    )

    sm = user_agent.session_manager
    session = user_agent.session

    if not args:
        return (
            "Usage:\n"
            "/cron list\n"
            "/cron add every <Nm|Nh> <task>\n"
            "/cron pause <#index>\n"
            "/cron resume <#index>\n"
            "/cron remove <#index>\n"
            "/cron run <#index>"
        )

    parts = args.split(None, 1)
    subcmd = parts[0].lower()
    subargs = parts[1].strip() if len(parts) > 1 else ""

    # ── list ──
    if subcmd in ("list", "ls"):
        jobs = await sm.list_cron_jobs(limit=200, active_only=False)
        # Filter to jobs belonging to this user's session.
        if session:
            jobs = [j for j in jobs if j.session_id == session.id]
        if not jobs:
            return "No cron jobs."
        lines: list[str] = []
        for idx, job in enumerate(jobs, 1):
            sched_text = schedule_to_text(job.schedule) if isinstance(job.schedule, dict) else "?"
            status = job.last_status or "pending"
            enabled_tag = "" if job.enabled else " [paused]"
            kind = job.kind
            payload_summary = ""
            if kind == "prompt":
                payload_summary = str(job.payload.get("text", ""))[:60]
            elif kind in ("script", "tool"):
                payload_summary = str(job.payload.get("path", ""))[:60]
            elif kind == "orchestrate":
                payload_summary = str(job.payload.get("workflow", ""))[:60]
            lines.append(
                f"#{idx} [{kind}] {sched_text}{enabled_tag} ({status})\n  {payload_summary}"
            )
        return "\n".join(lines)

    # ── add ──
    if subcmd == "add":
        if not session:
            return "No active session."
        if not subargs:
            return "Usage: /cron add every <Nm|Nh> <task>"
        try:
            schedule, kind, payload = parse_cron_add_args(subargs)
        except ValueError as e:
            return str(e)
        next_run_at_iso = to_utc_iso(compute_next_run(schedule))
        job = await sm.create_cron_job(
            kind=kind, payload=payload, schedule=schedule,
            session_id=session.id, next_run_at=next_run_at_iso, enabled=True,
        )
        if server._runtime_ctx:
            await cron_monitor_event(
                server._runtime_ctx, "job_added", history_job_id=job.id,
                job_id=job.id, session_id=job.session_id, kind=job.kind,
                schedule=schedule_to_text(schedule), next_run_at=next_run_at_iso,
            )
        return (
            f"Cron job added: {job.id}\n"
            f"Kind: {kind}, Schedule: {schedule_to_text(schedule)}\n"
            f"Next run: {next_run_at_iso}"
        )

    # ── remove ──
    if subcmd in ("remove", "rm", "delete", "del"):
        if not subargs:
            return "Usage: /cron remove <#index>"
        job = await _tg_select_user_cron_job(sm, session, subargs)
        if not job:
            return f"Cron job not found: {subargs}"
        await sm.delete_cron_job(job.id)
        return f"Removed cron job: {job.id}"

    # ── pause ──
    if subcmd in ("pause", "disable"):
        if not subargs:
            return "Usage: /cron pause <#index>"
        job = await _tg_select_user_cron_job(sm, session, subargs)
        if not job:
            return f"Cron job not found: {subargs}"
        await sm.update_cron_job(job.id, enabled=False, last_status="paused")
        return f"Paused cron job: {job.id}"

    # ── resume ──
    if subcmd in ("resume", "enable"):
        if not subargs:
            return "Usage: /cron resume <#index>"
        job = await _tg_select_user_cron_job(sm, session, subargs)
        if not job:
            return f"Cron job not found: {subargs}"
        next_run_at_iso = to_utc_iso(compute_next_run(job.schedule))
        await sm.update_cron_job(
            job.id, enabled=True, last_status="scheduled",
            next_run_at=next_run_at_iso,
        )
        return f"Resumed cron job: {job.id}\nNext run: {next_run_at_iso}"

    # ── run ──
    if subcmd == "run":
        if not subargs:
            return "Usage: /cron run <#index>"
        job = await _tg_select_user_cron_job(sm, session, subargs)
        if not job:
            return f"Cron job not found: {subargs}"
        if not server._runtime_ctx:
            return "Server runtime not available."
        try:
            await execute_cron_job(server._runtime_ctx, job, trigger="manual")
        except Exception as e:
            return f"Cron run failed: {e}"
        return f"Manual cron run finished: {job.id}"

    # ── one-off prompt — just send the text as a regular message ──
    return "Unknown cron subcommand. Use /cron list|add|remove|pause|resume|run"


async def _tg_select_user_cron_job(sm: Any, session: Session | None, selector: str) -> Any:
    """Select a cron job by #index, scoped to the user's session."""
    if not session:
        return None
    # Support #N index selection.
    if selector.startswith("#"):
        try:
            idx = int(selector[1:]) - 1
        except ValueError:
            return None
        jobs = await sm.list_cron_jobs(limit=200, active_only=False)
        user_jobs = [j for j in jobs if j.session_id == session.id]
        if 0 <= idx < len(user_jobs):
            return user_jobs[idx]
        return None
    # Fall back to global selector.
    return await sm.select_cron_job(selector, active_only=False)


# ── Agent processing ─────────────────────────────────────────────


async def _tg_process_with_typing(
    server: WebServer,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None = None,
    *,
    user_agent: Agent,
    user_id: str,
) -> None:
    """Run user_agent.complete() with typing indicator and per-user lock.

    Different Telegram users run concurrently.  Requests from the *same*
    user are serialised by a per-user asyncio lock.
    """
    bridge = server._telegram_bridge
    if not bridge:
        return

    lock = _tg_get_user_lock(server, user_id)

    if lock.locked():
        await _tg_send(
            server, chat_id,
            "Your previous request is still running. This message is queued\u2026",
            reply_to_message_id=reply_to_message_id,
        )

    async with lock:
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
                turn_start_idx = len(user_agent.session.messages) if user_agent.session else 0
                log.info("Telegram agent.complete() start", user_id=user_id, text_len=len(text))
                response = await user_agent.complete(text)
                log.info("Telegram agent.complete() done", user_id=user_id, response_len=len(response or ""))

                await _tg_send(server, chat_id, response, reply_to_message_id=reply_to_message_id)

                # Send any generated images back to Telegram.
                from captain_claw.platform_adapter import (
                    collect_turn_generated_document_paths,
                    collect_turn_generated_image_paths,
                )
                if user_agent.session and server._telegram_bridge:
                    for img_path in collect_turn_generated_image_paths(user_agent.session, turn_start_idx):
                        try:
                            await server._telegram_bridge.send_photo(
                                chat_id, img_path, reply_to_message_id=reply_to_message_id,
                            )
                        except Exception as img_exc:
                            log.warning("Telegram send_photo failed", path=str(img_path), error=str(img_exc))

                # Send any generated documents back to Telegram.
                if user_agent.session and server._telegram_bridge:
                    for doc_path in collect_turn_generated_document_paths(user_agent.session, turn_start_idx):
                        try:
                            await server._telegram_bridge.send_document(
                                chat_id, doc_path,
                                caption=doc_path.name,
                                reply_to_message_id=reply_to_message_id,
                            )
                        except Exception as doc_exc:
                            log.warning("Telegram send_document failed", path=str(doc_path), error=str(doc_exc))

                server._broadcast({
                    "type": "chat_message",
                    "role": "assistant",
                    "content": response,
                })

                # Extract and send suggested next steps as inline keyboard.
                from captain_claw.config import get_config
                if get_config().ui.next_steps:
                    try:
                        steps = await extract_next_steps(user_agent.provider, response)
                        if steps and server._telegram_bridge:
                            step_dicts = next_steps_to_dicts(steps)
                            # Cache for callback handler lookup.
                            cache_key = f"_tg_next_steps_{user_id}"
                            setattr(server, cache_key, step_dicts)
                            # Build inline keyboard rows (max 2 buttons per row).
                            keyboard_rows: list[list[dict[str, str]]] = []
                            row: list[dict[str, str]] = []
                            for i, s in enumerate(step_dicts):
                                row.append({"text": s["label"], "callback_data": f"ns:{i}"})
                                if len(row) >= 2:
                                    keyboard_rows.append(row)
                                    row = []
                            if row:
                                keyboard_rows.append(row)
                            await server._telegram_bridge.send_message_with_inline_keyboard(
                                chat_id,
                                "What would you like to do next?",
                                keyboard_rows,
                                reply_to_message_id=reply_to_message_id,
                            )
                    except Exception as ns_err:
                        log.debug("Telegram next steps extraction error", error=str(ns_err))

            server._broadcast({
                "type": "usage",
                "last": user_agent.last_usage,
                "total": user_agent.total_usage,
            })
        except Exception as exc:
            log.error("Telegram agent error", user_id=user_id, error=str(exc))
            await _tg_send(
                server, chat_id, f"Error: {str(exc)}", reply_to_message_id=reply_to_message_id,
            )
            server._broadcast({"type": "error", "message": f"Telegram error: {str(exc)}"})
        finally:
            stop_typing.set()
            heartbeat.cancel()
            # Clear any /btw instructions accumulated during this task.
            if hasattr(user_agent, "_btw_instructions"):
                user_agent._btw_instructions = []
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass


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


async def _tg_load_user_sessions(server: WebServer) -> dict[str, str]:
    """Load the user_id -> session_id mapping from app_state."""
    if not server.agent:
        return {}
    raw = await server.agent.session_manager.get_app_state("telegram_user_sessions")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return {str(k): str(v) for k, v in parsed.items()}
    return {}


async def _tg_save_user_sessions(server: WebServer) -> None:
    """Persist the user_id -> session_id mapping."""
    if not server.agent:
        return
    await server.agent.session_manager.set_app_state(
        "telegram_user_sessions",
        json.dumps(server._telegram_user_sessions, ensure_ascii=True, sort_keys=True),
    )


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
    await _tg_save_user_sessions(server)


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
