"""Telegram bridge integration for the web server.

Each approved Telegram user gets a dedicated Agent instance bound to
their own session.  Agents are created lazily on first message and
cached for the lifetime of the server process.  Per-user asyncio locks
serialise requests from the same user while allowing different users to
run concurrently.

The agent's owner is the exception: their own private chat joins the main
conversation (lane A) — the one the web chat, WhatsApp, the app and the
devices share — so Telegram is not a silo of its own
(``telegram.owner_in_main``). The owner is whoever reaches the agent through
Flight Deck's shared bot (it relays only the chat the owner linked), or the
approved user marked ``owner`` on a bot of the agent's own.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from captain_claw import mail_authority
from captain_claw.instructions import InstructionLoader
from captain_claw.logging import get_logger
from captain_claw.session import Session, get_session_manager
from captain_claw.next_steps import extract_next_steps, next_steps_to_dicts
from captain_claw import telegram_groups
from captain_claw.telegram_bridge import (
    TelegramBridge,
    TelegramCallbackQuery,
    TelegramGroupEvent,
    TelegramMessage,
)
from captain_claw.web import telegram_group_chat

if TYPE_CHECKING:
    from captain_claw.agent import Agent
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

# Commands disabled for Telegram users: each has ONE session at a time —
# /new starts a fresh one, but there is no listing or switching back.
_TG_DISABLED_COMMANDS = frozenset({"/sessions"})
_TG_DISABLED_SESSION_SUBCMDS = frozenset({"list", "switch", "load"})


# ── Voice notes, audio, video ───────────────────────────────────────

# The Bot API lets a bot download files up to this size (getFile).
_TG_DOWNLOAD_LIMIT = 20 * 1024 * 1024
_MEDIA_LABEL = {
    "voice": "voice note", "audio": "audio file", "video": "video",
    "video_note": "video message", "animation": "GIF",
}
_MEDIA_DEFAULT_EXT = {
    "voice": ".ogg", "audio": ".mp3", "video": ".mp4", "video_note": ".mp4", "animation": ".mp4",
}
# Extensions the attachment hints and the video analysis know.
_MEDIA_MIME_EXT = {
    "audio/ogg": ".ogg", "audio/opus": ".opus", "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/m4a": ".m4a", "audio/aac": ".aac",
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/flac": ".flac", "audio/x-flac": ".flac",
    "audio/amr": ".amr", "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
    "video/x-matroska": ".mkv", "video/x-msvideo": ".avi", "video/x-m4v": ".m4v",
}
_MEDIA_KNOWN_EXT = frozenset(_MEDIA_MIME_EXT.values())
_VIDEO_KINDS = frozenset({"video", "video_note", "animation"})


def _tg_media_filename(message: TelegramMessage, stamp: str) -> str:
    """``tg-<kind>[-<name>]-<stamp><ext>`` with an extension the attachment
    hints and the video analysis recognise."""
    from pathlib import Path

    name = Path(message.media_file_name or "")
    ext = name.suffix.lower()
    if ext not in _MEDIA_KNOWN_EXT:
        mime = (message.media_mime or "").split(";", 1)[0].strip().lower()
        ext = _MEDIA_MIME_EXT.get(mime) or _MEDIA_DEFAULT_EXT.get(message.media_kind, ".bin")
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in name.stem)[:50]
    kind = message.media_kind.replace("_", "")
    return f"tg-{kind}{'-' + stem if stem else ''}-{stamp}{ext}"


async def _tg_download_media(
    bridge: TelegramBridge, user_agent: Agent, message: TelegramMessage,
) -> tuple[str | None, str]:
    """Save the message's voice note / audio / video under the session's
    media folder: ``(path, "")``, or ``(None, what to tell the user)``."""
    label = _MEDIA_LABEL.get(message.media_kind, "file")
    too_big = (
        f"That {label} is too big for me to fetch — Telegram only lets bots download files "
        "up to 20 MB. Send a shorter one, or share a link to it."
    )
    if message.media_size > _TG_DOWNLOAD_LIMIT:
        return None, too_big
    try:
        from captain_claw.config import get_config

        workspace = get_config().resolved_workspace_path()
        session_id = user_agent.session.id if user_agent.session else "uploads"
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        dest = workspace / "saved" / "media" / session_id / _tg_media_filename(message, stamp)
        await bridge.download_file(message.media_file_id, dest)
    except Exception as exc:
        if "too big" in str(exc).lower():
            return None, too_big
        log.warning("Telegram media download failed", kind=message.media_kind, error=str(exc))
        return None, f"Couldn't fetch that {label} ({type(exc).__name__})."
    log.info("Telegram media downloaded", kind=message.media_kind, path=str(dest))
    return str(dest), ""


async def _tg_transcribe(path: str, mime: str) -> tuple[str, str]:
    """A voice note's transcript (Soniox, the owner's key through Flight Deck):
    ``(text, "")`` or ``("", why not)``."""
    from pathlib import Path

    from captain_claw import soniox_client
    from captain_claw.soniox_api import SonioxError

    if not await soniox_client.available():
        return "", ("speech-to-text isn't set up — connect Soniox in Flight Deck → "
                    "Connections → Soniox")
    try:
        blob = Path(path).read_bytes()
        result = await soniox_client.transcribe(
            blob, filename=Path(path).name, mime=(mime or "audio/ogg").split(";", 1)[0])
    except SonioxError as exc:
        return "", str(exc)
    except Exception as exc:
        return "", type(exc).__name__
    text = (result.text or "").strip()
    return (text, "") if text else ("", "no speech was recognised")


# ── Lifecycle ────────────────────────────────────────────────────


def _shared_bot_relay(server: WebServer) -> tuple[str, dict[str, str]] | None:
    """Flight Deck's shared-bot relay for this agent — ``(api_base_url,
    headers)`` — or None outside Flight Deck (or with ``telegram.shared_bot``
    off). The relay speaks the Bot API, so the bridge and everything below run
    unchanged; it only hands this agent its own owner's chats, and only once an
    admin approved them (Connections → Telegram). Until then its polls stay empty."""
    if not getattr(server.config.telegram, "shared_bot", True):
        return None
    from captain_claw import fd_client

    base = fd_client.pinned_flight_deck_url()
    if not base:
        return None
    headers = fd_client.agent_identity_headers(base)
    if not headers.get("X-Agent-Auth"):
        return None
    return f"{base}/fd/telegram-relay", headers


async def start_telegram(server: WebServer) -> None:
    """Initialize and start Telegram polling + worker if configured.

    The agent's own bot when it has one (``telegram.enabled`` + a token);
    otherwise, under Flight Deck, the deck's shared bot through FD's relay."""
    token = server.config.telegram.bot_token.strip()
    if server._telegram_enabled and token:
        server._telegram_bridge = TelegramBridge(
            token=token, api_base_url=server.config.telegram.api_base_url,
        )
    else:
        relay = _shared_bot_relay(server)
        if relay is None:
            if server.config.telegram.enabled:
                log.info("Telegram enabled but bot_token is empty; skipping in web mode.")
            return
        base, headers = relay
        # The path's token is a placeholder: FD knows the agent by its headers.
        server._telegram_bridge = TelegramBridge(token="shared", api_base_url=base, headers=headers)
        server._telegram_relay = True
    if server.agent:
        server._approved_telegram_users = await _tg_load_state(
            server, "telegram_approved_users"
        )
        server._pending_telegram_pairings = await _tg_load_state(
            server, "telegram_pending_pairings"
        )
        server._telegram_user_sessions = await _tg_load_user_sessions(server)
        await telegram_group_chat.load(server)
        _tg_cleanup_expired(server)
        await _tg_save_state(server)
    server._telegram_poll_task = asyncio.create_task(_telegram_poll_loop(server))
    server._telegram_worker_task = asyncio.create_task(_telegram_worker(server))
    if getattr(server, "_telegram_relay", False):
        log.info("Telegram: listening on Flight Deck's shared bot (relay).")
        return
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
    failures = 0
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
            if getattr(server, "_telegram_relay", False):
                # Flight Deck restarting, or this agent not (yet) known to it:
                # back off quietly instead of logging every two seconds.
                failures += 1
                if failures in (1, 10) or failures % 100 == 0:
                    log.warning("Shared Telegram relay unreachable", error=str(exc), attempts=failures)
                await asyncio.sleep(min(60.0, 2.0 * failures))
                continue
            log.error("Telegram poll error", error=str(exc))
            await asyncio.sleep(2.0)
        else:
            failures = 0


async def _telegram_worker(server: WebServer) -> None:
    """Dispatch queued Telegram messages and callback queries as concurrent tasks."""
    while True:
        try:
            update = await server._telegram_queue.get()
            if isinstance(update, TelegramCallbackQuery):
                asyncio.create_task(_handle_telegram_callback_query(server, update))
            elif isinstance(update, TelegramGroupEvent):
                # In order, before the group's next message (the add-to-group
                # link's "/start@bot agent" follows the join at once).
                await telegram_group_chat.handle_event(server, update)
            elif getattr(update, "chat_type", "private") in telegram_groups.GROUP_KINDS:
                asyncio.create_task(telegram_group_chat.handle_message(server, update))
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
        session = await sm.create_session(name=_tg_session_name(server, user_id_key))
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
    # A Telegram user's agent: the `sessions` tool refuses to move between
    # topics here (the list would be the owner's; /new is the way).
    agent._channel = "telegram"

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


def _tg_session_name(server: WebServer, user_id_key: str) -> str:
    """``tg-<username | first name | id>`` — the name of a Telegram user's session."""
    user_info = server._approved_telegram_users.get(user_id_key, {})
    username = str(user_info.get("username", "")).strip()
    first_name = str(user_info.get("first_name", "")).strip()
    return f"tg-{username or first_name or user_id_key}"


async def _tg_new_session(server: WebServer, user_agent: Agent, name: str = "") -> str:
    """``/new`` on Telegram: the user's agent moves to a fresh session, which
    becomes their session from now on (also after a restart). The old one is
    frozen into history first, as on the web."""
    user_id_key = str(getattr(user_agent, "_user_id", "") or "")
    if not user_id_key:
        return "No active session."
    # The command runs outside the per-user lock: never swap the session
    # under a turn that is still running.
    if _tg_get_user_lock(server, user_id_key).locked():
        return "Still answering your previous message — send /new once the reply is in."
    user_agent.archive_current_session_to_history()
    session = await user_agent.session_manager.create_session(
        name=name.strip() or _tg_session_name(server, user_id_key),
    )
    user_agent.session = session
    user_agent.refresh_session_runtime_flags()
    user_agent.last_usage = user_agent._empty_usage()
    user_agent.last_context_window = {}
    server._telegram_user_sessions[user_id_key] = session.id
    await _tg_save_user_sessions(server)
    log.info("Telegram user started a new session",
             user_id=user_id_key, session_id=session.id, session_name=session.name)
    return f"New session started: {session.name}. The previous conversation is kept in history."


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
        if not _tg_is_approved(server, user_id_key):
            if not getattr(server, "_telegram_relay", False):
                await _tg_pair_unknown_user(server, message)
                return
            await _tg_admit_relay_user(
                server, message.user_id, message.chat_id, message.username, message.first_name,
            )

        # The owner's private chat: the main conversation; anyone else: a
        # per-user agent (created on first use).
        user_agent, owner = await _tg_agent_for(server, message)

        # Card numbers / CVVs never go further (card_scrub).
        from captain_claw.card_scrub import scrub_user_text
        text = scrub_user_text(message.text.strip())

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

        # Voice notes, audio files, videos, round video messages, GIFs.
        media_kind = str(getattr(message, "media_kind", "") or "")
        forwarded = bool(getattr(message, "forwarded", False))
        media_path: str | None = None
        attachment_notes: list[str] = []
        if getattr(message, "media_file_id", ""):
            media_path, problem = await _tg_download_media(bridge, user_agent, message)
            if problem:
                await _tg_send(server, message.chat_id, problem,
                               reply_to_message_id=message.message_id)
        if media_path and media_kind == "voice" and not forwarded:
            # The user speaking to the agent: the transcript is the message (as on
            # WhatsApp). The recording is named in a note, not attached — an
            # attached .ogg only invites a weak model to transcribe it again.
            voice_path, media_path = media_path, None
            await _tg_send(server, message.chat_id, "🎙 Transcribing voice note…",
                           reply_to_message_id=message.message_id)
            transcript, why = await _tg_transcribe(voice_path, message.media_mime)
            if not transcript:
                await _tg_send(
                    server, message.chat_id,
                    f"Couldn't transcribe that — {why}. Please type it or send it again.",
                    reply_to_message_id=message.message_id,
                )
                return
            # A spoken card number is scrubbed like a typed one (card_scrub).
            from captain_claw.card_scrub import scrub, scrub_user_text

            await _tg_send(server, message.chat_id,
                           f"🎙 Transcription:\n\n\"{scrub(transcript)[0]}\"",
                           reply_to_message_id=message.message_id)
            transcript = scrub_user_text(transcript)
            text = f"{text}\n\n{transcript}" if text else transcript
            attachment_notes.append(
                "This message is the transcript of the user's Telegram voice note; "
                f"the recording is saved at {voice_path}."
            )
        elif media_path and media_kind == "voice":
            attachment_notes.append(
                "A forwarded voice note — someone else's words, not the user's own.")

        if not text and not image_path and not file_path and not contact_text and not media_path:
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
                "Commands: /new, /clear, /history, /compact, /config, /session, /cron, /basna, /group\n"
                "/new starts a fresh session (the old one is kept in history).\n"
                "/basna <task> runs a parallel multi-agent research ensemble.\n"
                "/group adds me to a Telegram group.\n"
                + ("This chat shares my main conversation — the web chat, WhatsApp, the app and devices."
                   if owner else "Each Telegram user has a dedicated session."),
                reply_to_message_id=message.message_id,
            )
            return
        # Bots can't use a group's invite link: offer the add-to-group button.
        if telegram_group_chat.wants_group_link(text):
            await telegram_group_chat.offer_add_link(server, message)
            return

        if owner and lowered.split()[0] == "/new":
            await _tg_send(server, message.chat_id, await _tg_owner_new_session(server, text),
                           reply_to_message_id=message.message_id)
            return
        if text.startswith("/"):
            result = await execute_telegram_command(server, text, user_agent)
            if result is not None:
                await _tg_send(
                    server, message.chat_id, result,
                    reply_to_message_id=message.message_id,
                )
                return

        # Build effective text with attachment context if present — the same
        # lines the web chat and WhatsApp get (reader hints, video guidance,
        # attachment notes).
        from captain_claw.web.chat_handler import (
            agent_sees_attached_images,
            build_attachment_prefix,
        )

        sees_inline = agent_sees_attached_images(user_agent)
        attached = build_attachment_prefix(
            [image_path] if image_path else [],
            [p for p in (file_path, media_path) if p],
            attachment_notes,
            sees_inline=sees_inline,
        )
        effective_text = text
        attachment_lines = list(attached.lines)
        if contact_text:
            attachment_lines.append(contact_text)
        if attachment_lines:
            prefix = "\n".join(attachment_lines) + "\n"
            if image_path:
                default_msg = "Please analyze this image."
            elif media_path and media_kind in _VIDEO_KINDS:
                default_msg = "I've sent a video."
            elif media_path:
                default_msg = "I've sent an audio file."
            elif file_path:
                default_msg = "I've attached a file."
            else:
                default_msg = "I'm sharing this contact."
            effective_text = prefix + (text or default_msg)

        if not owner:
            # The owner's turn shows on the main conversation once it has it.
            user_label = message.username or message.first_name or str(message.user_id)
            server._broadcast({
                "type": "chat_message",
                "role": "user",
                "content": f"[TG {user_label}] {effective_text}",
            })
            # Store the chat_id on the agent so delegate callbacks can route back here
            user_agent._telegram_chat_id = message.chat_id  # type: ignore[attr-defined]

        # A model that sees images gets the photo inline (its marker); any other
        # has it described first — a vision model or a multimodal peer — as the
        # chat path does for WhatsApp and Flight Deck.
        describe = [image_path] if image_path and not sees_inline else None
        await _tg_process_with_typing(
            server, message.chat_id, effective_text, message.message_id,
            user_agent=user_agent,
            user_id=str(message.user_id),
            describe_images=describe,
            analyze_videos=attached.videos or None,
            owner=owner,
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
        if cbq.chat_id < 0:
            return  # a group: its answers carry no buttons
        user_id_key = str(cbq.user_id)
        if not _tg_is_approved(server, user_id_key):
            if not getattr(server, "_telegram_relay", False):
                return
            await _tg_admit_relay_user(
                server, cbq.user_id, cbq.chat_id, cbq.username, cbq.first_name,
            )

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

        # The user's agent (the owner: the main conversation) takes the action.
        user_agent, owner = await _tg_agent_for(server, TelegramMessage(
            update_id=cbq.update_id,
            message_id=cbq.message_id,
            chat_id=cbq.chat_id,
            user_id=cbq.user_id,
            username=cbq.username,
            first_name=cbq.first_name,
            text=action_text,
        ))

        if not owner:
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
            owner=owner,
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

    Each Telegram user operates on their own *user_agent*. ``/new`` starts
    a fresh session; listing and switching sessions are disabled.
    """
    parts = raw.strip().split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    session = user_agent.session

    try:
        # ── Disabled commands ────────────────────────────────────
        if cmd in _TG_DISABLED_COMMANDS:
            return "Not available on Telegram. Each user has a dedicated session."

        # ── /new ─────────────────────────────────────────────────
        if cmd in ("/new",):
            return await _tg_new_session(server, user_agent, args)

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

        if cmd in ("/basna",):
            return await user_agent.run_basna_command(args.strip())

        if cmd in ("/code", "/publish"):
            # Fall through to the normal agent turn — complete() rewrites /code
            # (code tool) and /publish (hosting tool) into the right directive;
            # both may need conversation context to resolve the target.
            return None

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

    if subcmd in ("new",):
        return await _tg_new_session(server, user_agent, subargs)

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
                metadata=session.metadata,
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
    describe_images: list[str] | None = None,
    analyze_videos: list[str] | None = None,
    owner: bool = False,
) -> None:
    """Run user_agent.complete() with typing indicator and per-user lock.

    *owner*: the owner's private chat on the main agent — the turn also holds
    the main conversation (:func:`_tg_owner_lane`).

    Different Telegram users run concurrently.  Requests from the *same*
    user are serialised by a per-user asyncio lock. *describe_images* are
    described server-side first (for a model that doesn't see images), and
    *analyze_videos* run through video_vision first (frames + transcript), as
    the web chat does; both go to the agent, never into the user's words the
    mail guard reads.
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

    async with lock, contextlib.AsyncExitStack() as owner_lane:
        if owner:
            await owner_lane.enter_async_context(
                _tg_owner_lane(server, chat_id, user_id, text, reply_to_message_id))
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
                turn_started_at = time.time()  # what this turn writes counts as a deliverable
                log.info("Telegram agent.complete() start", user_id=user_id, text_len=len(text))
                # Narration goes out live, one message per blurb; the agent
                # then answers with just its last response instead of folding
                # every blurb into one long message.
                narration = _TgNarration(server, chat_id) if _stream_narration() else None
                # The main agent's own hook (the web chat's) comes back after.
                prev_narration = getattr(user_agent, "narration_callback", None)
                user_agent.narration_callback = narration
                agent_text = text
                if analyze_videos:
                    from captain_claw.web.chat_handler import _prefix_video_analysis

                    await _tg_send(server, chat_id, "🎬 Watching the video…",
                                   reply_to_message_id=reply_to_message_id)
                    agent_text = await _prefix_video_analysis(
                        user_agent, agent_text, analyze_videos, lambda _event: None)
                if describe_images:
                    from captain_claw.web.chat_handler import _prefix_image_analysis

                    agent_text = await _prefix_image_analysis(
                        user_agent, agent_text, describe_images, lambda _event: None)
                try:
                    with mail_authority.bound(mail_authority.interactive(text)):
                        from captain_claw import msg_origin as _msg_origin

                        _msg_origin.hint_turn_provenance(
                            user_agent, channel="telegram", source="telegram", source_label="Telegram")
                        response = await user_agent.complete(agent_text)
                finally:
                    user_agent.narration_callback = prev_narration
                    if narration is not None:
                        await narration.drain()  # the answer always lands last
                log.info("Telegram agent.complete() done", user_id=user_id, response_len=len(response or ""))

                await _tg_send(server, chat_id, response, reply_to_message_id=reply_to_message_id)
                # The whole account — narration and answer, as the reply read
                # before narration went live: the web mirror shows it, and a
                # document a blurb named still goes out.
                full_reply = (
                    "\n\n---\n\n".join([*narration.texts, response or ""])
                    if narration is not None and narration.texts else response
                )

                # A reply that came out garbled gets a /new hint (Jev judges,
                # beside the deliverables and next steps below).
                from captain_claw import garbled_reply

                garble_task = asyncio.create_task(garbled_reply.is_garbled(response or "", text))

                # Send what the turn made for the user back to Telegram — only
                # deliverables: files in the agent's saved/ area written this
                # turn (pictures a tool made, documents the reply names), never
                # a path some tool output, email body or web page printed.
                from captain_claw.platform_adapter import (
                    collect_turn_generated_document_paths,
                    collect_turn_generated_image_paths,
                    effective_turn_start_idx,
                )
                from captain_claw.turn_deliverables import saved_root_of
                turn_start_idx = effective_turn_start_idx(user_agent, turn_start_idx)
                if user_agent.session and server._telegram_bridge:
                    from captain_claw.platform_adapter import with_reply_images

                    for img_path in await with_reply_images(user_agent, collect_turn_generated_image_paths(
                        user_agent.session, turn_start_idx, saved_root=saved_root_of(user_agent),
                        since=turn_started_at, agent=user_agent, reply=full_reply or "",
                    ), full_reply or "", turn_start_idx, turn_started_at):
                        try:
                            await server._telegram_bridge.send_photo(
                                chat_id, img_path, reply_to_message_id=reply_to_message_id,
                            )
                        except Exception as img_exc:
                            log.warning("Telegram send_photo failed", path=str(img_path), error=str(img_exc))

                # Send any generated documents back to Telegram.
                if user_agent.session and server._telegram_bridge:
                    for doc_path in collect_turn_generated_document_paths(
                        user_agent.session, turn_start_idx, agent=user_agent,
                        reply=full_reply or "", since=turn_started_at,
                    ):
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
                    "content": full_reply,
                })

                # Extract and send suggested next steps as inline keyboard —
                # skip for FD-spawned orchestrated workers (Basna/Vatra/Council/Code).
                from captain_claw.config import get_config
                from captain_claw.agent_reasoning_mixin import _is_fd_spawned_worker
                if get_config().ui.next_steps and not _is_fd_spawned_worker():
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

                if await garble_task:
                    await _tg_send(server, chat_id, garbled_reply.HINT)

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


def _stream_narration() -> bool:
    """Live narration is on unless ``STREAM_NARRATION`` is off — the same
    switch the WhatsApp lane reads. Off, the blurbs fold into the answer."""
    return os.environ.get("STREAM_NARRATION", "on").strip().lower() not in ("off", "0", "false", "no")


class _TgNarration:
    """A turn's narration callback: each blurb the model writes beside its
    tool calls goes to the chat as its own message the moment it is written,
    like the WhatsApp lane. Only to this chat — the agent's web socket would
    carry it to Flight Deck's WhatsApp channels too.

    Sends are chained so blurbs arrive in order; ``drain`` waits for them."""

    def __init__(self, server: WebServer, chat_id: int) -> None:
        self._server = server
        self._chat_id = chat_id
        self._tail: asyncio.Task[None] | None = None
        self.texts: list[str] = []

    def __call__(self, text: str, iteration: int = 0) -> None:
        text = str(text or "").strip()
        if not text:
            return
        self.texts.append(text)
        self._tail = asyncio.get_running_loop().create_task(self._send(text, self._tail))

    async def _send(self, text: str, prev: asyncio.Task[None] | None) -> None:
        if prev is not None:
            await asyncio.wait({prev})  # done, however it ended
        await _tg_send(self._server, self._chat_id, text)
        # A message clears "typing…" until the next heartbeat; put it back.
        bridge = self._server._telegram_bridge
        if bridge:
            with contextlib.suppress(Exception):
                await bridge.send_chat_action(chat_id=self._chat_id, action="typing")

    async def drain(self) -> None:
        if self._tail is not None:
            with contextlib.suppress(Exception):
                await self._tail


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


def _tg_is_approved(server: WebServer, user_id_key: str) -> bool:
    """Approved to talk to this agent. Someone admitted through Flight Deck's
    shared bot counts only there — never on a bot of the agent's own."""
    info = server._approved_telegram_users.get(user_id_key)
    if info is None:
        return False
    return getattr(server, "_telegram_relay", False) or info.get("via") != "shared_bot"


def _tg_is_owner(server: WebServer, user_id_key: str, chat_type: str = "private") -> bool:
    """The agent's owner writing in their own private chat: it joins the main
    conversation. Through Flight Deck's shared bot that is anyone relayed (it
    only relays the chat the owner linked); on a bot of the agent's own, the
    approved user marked ``owner``. Groups never."""
    if chat_type and chat_type != "private":
        return False
    info = (getattr(server, "_approved_telegram_users", None) or {}).get(str(user_id_key))
    if not isinstance(info, dict) or not (info.get("via") == "shared_bot" or info.get("owner") is True):
        return False
    if not getattr(getattr(getattr(server, "config", None), "telegram", None), "owner_in_main", True):
        return False
    return _tg_is_approved(server, str(user_id_key))


# How long the owner's Telegram message waits for the main conversation to be
# free (a web, WhatsApp or device turn running there) before giving up.
_OWNER_LANE_WAIT_SECONDS = 15 * 60
# What the main agent carries while it answers the owner's Telegram, so a
# delegation made in that turn reports back to the chat (tools/flight_deck.py,
# basna, bat, code_session).
_OWNER_TURN_ATTRS = ("_telegram_chat_id", "_telegram_user_id")


@contextlib.asynccontextmanager
async def _tg_owner_lane(
    server: WebServer, chat_id: int, user_id: str, content: str = "",
    reply_to_message_id: int | None = None,
):
    """The owner's Telegram turn on the main conversation (lane A): wait while
    another turn runs there, hold it as busy, show the turn on every client
    ("via Telegram"), and on the way out do what a lane-A turn's end does."""
    agent = server.agent
    deadline = time.monotonic() + _OWNER_LANE_WAIT_SECONDS
    told = False
    while True:
        while server._busy:
            if not told:
                told = True
                await _tg_send(server, chat_id, "Still answering another message — yours is next…",
                               reply_to_message_id=reply_to_message_id)
            if time.monotonic() > deadline:
                raise RuntimeError("the main conversation stayed busy — send it again in a moment")
            await asyncio.sleep(0.5)
        # Automation results still waiting go in ahead of this turn, as on
        # every lane-A turn.
        from captain_claw.web.chat_handler import PENDING_RESULTS_KEY, flush_automation_results

        meta = getattr(getattr(agent, "session", None), "metadata", None)
        if isinstance(meta, dict) and meta.get(PENDING_RESULTS_KEY):
            await flush_automation_results(server)
        if not server._busy:
            break
    server._busy = True
    saved = {k: getattr(agent, k) for k in _OWNER_TURN_ATTRS if hasattr(agent, k)}
    agent._telegram_chat_id = int(chat_id)
    agent._telegram_user_id = str(user_id)
    server._broadcast({"type": "status", "status": "thinking"})
    try:
        if content:
            from captain_claw.web.chat_handler import user_turn_frame

            server._broadcast(user_turn_frame(content, "telegram", "", "telegram", "Telegram"))
        yield
    finally:
        for k in _OWNER_TURN_ATTRS:
            if k in saved:
                setattr(agent, k, saved[k])
            elif hasattr(agent, k):
                delattr(agent, k)
        server._busy = False
        server._broadcast({"type": "status", "status": "ready"})
        # A `sessions` tool call in the turn moves the conversation now.
        try:
            from captain_claw.web.slash_commands import apply_pending_session_change

            await apply_pending_session_change(server, agent)
        except Exception as exc:
            log.warning("Pending session change failed", error=str(exc))


async def _tg_agent_for(server: WebServer, message: TelegramMessage) -> tuple[Agent, bool]:
    """The agent a Telegram sender talks to, and whether it is the owner's
    main conversation (the owner's private chat) — anyone else has a session
    of their own."""
    if _tg_is_owner(server, str(message.user_id), getattr(message, "chat_type", "private")):
        return server.agent, True
    return await _tg_get_or_create_agent(server, message), False


async def _tg_owner_new_session(server: WebServer, text: str) -> str:
    """``/new`` from the owner's Telegram: a new main conversation, as the web
    chat's /new — every client on it is told."""
    if server._busy:
        return "Still answering a message — send /new once the reply is in."
    from captain_claw.web.slash_commands import apply_pending_session_change

    parts = text.split(None, 1)
    server.agent._pending_session_change = {"action": "new", "name": parts[1].strip() if len(parts) > 1 else ""}
    await apply_pending_session_change(server, server.agent)
    name = getattr(server.agent.session, "name", "") if server.agent.session else ""
    return f"New session started: {name}. The previous conversation is kept in history."


async def _tg_admit_relay_user(
    server: WebServer, user_id: int, chat_id: int, username: str, first_name: str,
) -> None:
    """Shared bot: Flight Deck relays only the chats an admin approved for this
    agent's owner, so whoever reaches us through it is admitted without the
    agent's own pairing step."""
    server._approved_telegram_users[str(user_id)] = {
        "user_id": int(user_id),
        "chat_id": int(chat_id),
        "username": str(username or "").strip(),
        "first_name": str(first_name or "").strip(),
        "approved_at": datetime.now(UTC).isoformat(),
        "token": "",
        "via": "shared_bot",
    }
    await _tg_save_state(server)
    log.info("Telegram user admitted via Flight Deck's shared bot", user_id=user_id)


async def _tg_pair_unknown_user(server: WebServer, message: TelegramMessage) -> None:
    _tg_cleanup_expired(server)
    user_id_key = str(message.user_id)
    if _tg_is_approved(server, user_id_key):
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
    """Handle /approve user telegram <token> [owner].

    ``owner`` marks the person as the agent's owner: their private chat joins
    the main conversation. For someone already approved, their @username or
    Telegram id stands in for the token."""
    usage = "Usage: `/approve user telegram <token> [owner]`"
    parts = args.split()
    if len(parts) < 3 or parts[0].lower() != "user" or parts[1].lower() != "telegram":
        return usage
    token = parts[2].strip().upper()
    if not token:
        return usage
    owner = len(parts) > 3 and parts[3].lower() == "owner"
    _tg_cleanup_expired(server)
    record = server._pending_telegram_pairings.get(token)
    if not isinstance(record, dict) and owner:
        return await _tg_mark_owner(server, parts[2].strip())
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
        **({"owner": True} if owner else {}),
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
    if owner:
        return f"Approved Telegram user: **{label}** — the owner: their private chat shares the main conversation."
    return f"Approved Telegram user: **{label}**"


async def _tg_mark_owner(server: WebServer, who: str) -> str:
    """Mark an approved Telegram user (by @username or id) as the owner."""
    key = who.strip().lstrip("@").lower()
    for uid, info in server._approved_telegram_users.items():
        if not isinstance(info, dict):
            continue
        if key in (str(uid), str(info.get("username", "")).strip().lstrip("@").lower()):
            info["owner"] = True
            await _tg_save_state(server)
            label = info.get("username") or info.get("first_name") or uid
            return f"**{label}** is the owner: their private chat now shares the main conversation."
    return f"No approved Telegram user `{who}` — approve them first (their pairing token)."
