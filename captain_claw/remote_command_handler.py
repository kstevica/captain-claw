"""Remote (Telegram/Slack/Discord) command and message handling.

Provides the unified ``_handle_remote_command`` logic and the
per-platform ``handle_platform_message`` dispatcher.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from captain_claw.config import get_config
from captain_claw.logging import log
from captain_claw.platform_adapter import PlatformAdapter, truncate_chat_text

if TYPE_CHECKING:
    from captain_claw.runtime_context import RuntimeContext


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def format_recent_history(ctx: RuntimeContext, limit: int = 30) -> str:
    agent = ctx.agent
    if not agent.session:
        return "No active session."
    messages = agent.session.messages[-max(1, int(limit)):]
    if not messages:
        return "Session history is empty."
    lines: list[str] = [f"Session: {agent.session.name} ({agent.session.id})", ""]
    for idx, msg in enumerate(messages, start=1):
        role = str(msg.get("role", "")).strip().lower() or "unknown"
        content = str(msg.get("content", "")).strip().replace("\n", " ")
        if len(content) > 220:
            content = content[:220].rstrip() + "..."
        lines.append(f"{idx}. {role}: {content}")
    return "\n".join(lines)


def format_active_configuration_text(ctx: RuntimeContext) -> str:
    cfg = get_config()
    agent = ctx.agent
    active_model = agent.get_runtime_model_details()
    active_provider = str(active_model.get("provider", "")).strip() or cfg.model.provider
    active_model_name = str(active_model.get("model", "")).strip() or cfg.model.model
    active_model_id = str(active_model.get("id", "")).strip()
    active_model_source = str(active_model.get("source", "")).strip() or "default"
    model_id_part = f" [id={active_model_id}]" if active_model_id else ""
    workspace_path = str(cfg.resolved_workspace_path(Path.cwd()))
    return "\n".join([
        "Configuration (active):",
        f"- model: {active_provider}/{active_model_name}{model_id_part} (source={active_model_source})",
        f"- workspace: {workspace_path}",
        f"- pipeline: {agent.pipeline_mode}",
        f"- context size: {int(cfg.context.max_tokens)} tokens",
        (
            "- guards: "
            f"input(enabled={cfg.guards.input.enabled}, level={cfg.guards.input.level}), "
            f"output(enabled={cfg.guards.output.enabled}, level={cfg.guards.output.level}), "
            f"script/tool(enabled={cfg.guards.script_tool.enabled}, level={cfg.guards.script_tool.level})"
        ),
    ])


def remote_help_text(ctx: RuntimeContext, platform_label: str) -> str:
    lines = [f"Captain Claw {platform_label} commands:", ""]
    for command, description in ctx.telegram_command_specs:
        lines.append(f"/{command} - {description}")
    lines.extend([
        "",
        "Tip: send normal text (without `/`) to chat with the active session.",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unified remote command handler
# ---------------------------------------------------------------------------

async def handle_remote_command(
    ctx: RuntimeContext,
    *,
    platform: str,
    raw_text: str,
    help_label: str,
    sender_label: str,
    send_text: Callable[[str], Awaitable[None]],
    execute_prompt: Callable[[str, str], Awaitable[None]],
) -> bool:
    from captain_claw.prompt_execution import enqueue_agent_task

    agent = ctx.agent
    ui = ctx.ui

    lowered_text = raw_text.strip().lower()
    if lowered_text == "/start" or lowered_text.startswith("/start "):
        await send_text(
            "Captain Claw connected.\n"
            "Use /help to see available commands.\n"
            "Send plain text to chat with the current session."
        )
        return True
    if lowered_text == "/help" or lowered_text.startswith("/help "):
        await send_text(remote_help_text(ctx, help_label))
        return True

    result = ui.handle_special_command(raw_text)
    if result is None:
        await send_text("Command processed.")
        return True
    if result == "EXIT":
        await send_text("`/exit` is only available in local console.")
        return True
    if result.startswith("APPROVE_CHAT_USER:") or result.startswith("APPROVE_TELEGRAM_USER:"):
        await send_text("This command is operator-only in local console.")
        return True
    if result.startswith("ORCHESTRATE:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            await send_text("Invalid /orchestrate payload.")
            return True
        request_text = str(payload.get("request", "")).strip()
        if not request_text:
            await send_text("Usage: /orchestrate <request>")
            return True
        try:
            from captain_claw.session_orchestrator import SessionOrchestrator
            orchestrator = SessionOrchestrator(
                main_agent=agent,
                provider=agent.provider,
                status_callback=agent.status_callback,
                tool_output_callback=agent.tool_output_callback,
            )
            orch_response = await enqueue_agent_task(
                ctx,
                agent.session.id if agent.session else None,
                lambda: orchestrator.orchestrate(request_text),
            )
            await send_text(str(orch_response or "Orchestration returned no result."))
            await orchestrator.shutdown()
        except Exception as e:
            await send_text(f"Orchestration failed: {e}")
        return True
    if result == "PIPELINE_INFO":
        await send_text(
            f"Pipeline mode: {agent.pipeline_mode} "
            "(loop=fast/simple, contracts=planner+completion gate)"
        )
        return True
    if result == "PLANNING_ON":
        await agent.set_pipeline_mode("contracts")
        await send_text("Pipeline mode set to contracts.")
        return True
    if result == "PLANNING_OFF":
        await agent.set_pipeline_mode("loop")
        await send_text("Pipeline mode set to loop.")
        return True
    if result.startswith("PIPELINE_MODE:"):
        mode = result.split(":", 1)[1].strip().lower()
        try:
            await agent.set_pipeline_mode(mode)
        except Exception:
            await send_text("Invalid pipeline mode. Use /pipeline loop|contracts")
            return True
        await send_text(f"Pipeline mode set to {agent.pipeline_mode}.")
        return True
    if result == "SKILLS_LIST":
        skills = agent.list_user_invocable_skills()
        if not skills:
            await send_text("No user-invocable skills available.")
            return True
        lines = ["Available skills:", "Use `/skill <name> [args]` to run one:"]
        for command in skills:
            lines.append(f"- /skill {command.name}")
        lines.append("Search catalog: `/skill search <criteria>`")
        lines.append("Install from GitHub: `/skill install <github-url>`")
        lines.append("Install skill deps: `/skill install <skill-name> [install-id]`")
        await send_text("\n".join(lines))
        return True
    if result.startswith("SKILL_SEARCH:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            await send_text("Invalid /skill search payload.")
            return True
        query = str(payload.get("query", "")).strip()
        if not query:
            await send_text("Usage: /skill search <criteria>")
            return True
        search_result = await agent.search_skill_catalog(query)
        if not bool(search_result.get("ok", False)):
            await send_text(str(search_result.get("error", "Skill search failed.")))
            return True
        source = str(search_result.get("source", "")).strip()
        items = list(search_result.get("results", []))
        lines = [f'Top skills for "{query}":']
        if source:
            lines.append(f"Source: {source}")
        if not items:
            lines.append("No matching skills found.")
        for idx, item in enumerate(items, start=1):
            name = str(item.get("name", "")).strip() or "Unnamed"
            desc = str(item.get("description", "")).strip()
            url = str(item.get("url", "")).strip()
            line = f"{idx}. {name}"
            if desc:
                line += f" - {desc}"
            lines.append(line)
            if url:
                lines.append(f"   {url}")
        await send_text("\n".join(lines))
        return True
    if result.startswith("SKILL_INSTALL:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            await send_text("Invalid /skill install payload.")
            return True
        skill_url = str(payload.get("url", "")).strip()
        skill_name = str(payload.get("name", "")).strip()
        install_id = str(payload.get("install_id", "")).strip()
        if skill_url:
            install_result = await agent.install_skill_from_github(skill_url)
            if not bool(install_result.get("ok", False)):
                await send_text(str(install_result.get("error", "Skill install failed.")))
                return True
            skill_name = str(install_result.get("skill_name", "")).strip() or "unknown"
            destination = str(install_result.get("destination", "")).strip()
            alias_list = list(install_result.get("aliases", []))
            lines = [f'Installed skill "{skill_name}".']
            if destination:
                lines.append(f"Path: {destination}")
            if alias_list:
                lines.append(f"Invoke with: /skill {alias_list[0]}")
            await send_text("\n".join(lines))
            return True
        if not skill_name:
            await send_text("Usage: /skill install <github-url> | /skill install <skill-name> [install-id]")
            return True
        install_result = await agent.install_skill_dependencies(
            skill_name=skill_name, install_id=install_id or None,
        )
        if not bool(install_result.get("ok", False)):
            await send_text(str(install_result.get("error", "Skill dependency install failed.")))
            return True
        lines = [str(install_result.get("message", "Dependencies installed.")).strip()]
        command = str(install_result.get("command", "")).strip()
        if command:
            lines.append(f"Command: {command}")
        await send_text("\n".join(lines))
        return True
    if result.startswith("SKILL_INVOKE:") or result.startswith("SKILL_ALIAS_INVOKE:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            await send_text("Invalid /skill payload.")
            return True
        skill_name = str(payload.get("name", "")).strip()
        skill_args = str(payload.get("args", "")).strip()
        if not skill_name:
            await send_text(
                "Usage: /skill <name> [args] | /skill search <criteria> | "
                "/skill install <github-url> | /skill install <skill-name> [install-id]"
            )
            return True
        invocation = await agent.invoke_skill_command(skill_name, args=skill_args)
        if not bool(invocation.get("ok", False)):
            if result.startswith("SKILL_ALIAS_INVOKE:"):
                await send_text(f"Unknown command: /{skill_name}")
            else:
                await send_text(str(invocation.get("error", "Skill invocation failed.")))
            return True
        mode = str(invocation.get("mode", "")).strip().lower()
        if mode == "dispatch":
            await send_text(str(invocation.get("text", "")).strip() or "Done.")
            return True
        prompt = str(invocation.get("prompt", "")).strip()
        if not prompt:
            await send_text("Skill invocation returned empty prompt.")
            return True
        await execute_prompt(prompt, f"[{sender_label} skill:{skill_name}] {skill_args}".strip())
        return True
    if result == "SESSION_INFO":
        if not agent.session:
            await send_text("No active session.")
        else:
            details = agent.get_runtime_model_details()
            await send_text(
                f"Session: {agent.session.name}\n"
                f"ID: {agent.session.id}\n"
                f"Messages: {len(agent.session.messages)}\n"
                f"Model: {details.get('provider')}/{details.get('model')}"
            )
        return True
    if result == "SESSIONS":
        sessions = await agent.session_manager.list_sessions(limit=20)
        if not sessions:
            await send_text("No sessions found.")
            return True
        lines = ["Sessions:"]
        for idx, session in enumerate(sessions, start=1):
            marker = "*" if agent.session and session.id == agent.session.id else " "
            lines.append(
                f"{marker} [{idx}] {session.name} ({session.id}) messages={len(session.messages)}"
            )
        await send_text("\n".join(lines))
        return True
    if result == "MODELS":
        models = agent.get_allowed_models()
        details = agent.get_runtime_model_details()
        lines = ["Allowed models:"]
        for idx, model in enumerate(models, start=1):
            marker = ""
            if (
                str(model.get("provider", "")).strip() == str(details.get("provider", "")).strip()
                and str(model.get("model", "")).strip() == str(details.get("model", "")).strip()
            ):
                marker = " *"
            lines.append(
                f"[{idx}] {model.get('id')} -> {model.get('provider')}/{model.get('model')}{marker}"
            )
        await send_text("\n".join(lines))
        return True
    if result == "CLEAR":
        if agent.session:
            if agent.is_session_memory_protected():
                await send_text("Session memory is protected. Disable with /session protect off.")
                return True
            agent.session.messages = []
            await agent.session_manager.save_session(agent.session)
            await send_text("Session cleared.")
        else:
            await send_text("No active session.")
        return True
    if result == "COMPACT":
        compacted, stats = await agent.compact_session(force=True, trigger="manual")
        if compacted:
            await send_text(
                f"Session compacted ({int(stats.get('before_tokens', 0))} -> {int(stats.get('after_tokens', 0))} tokens)"
            )
        else:
            await send_text(f"Compaction skipped: {stats.get('reason', 'not_needed')}")
        return True
    if result == "CONFIG":
        await send_text(format_active_configuration_text(ctx))
        return True
    if result == "HISTORY":
        await send_text(format_recent_history(ctx, limit=30))
        return True
    if result == "SESSION_MODEL_INFO":
        details = agent.get_runtime_model_details()
        await send_text(
            f"Active model: {details.get('provider')}/{details.get('model')} "
            f"(source={details.get('source') or 'unknown'}, id={details.get('id') or '-'})"
        )
        return True
    if result.startswith("SESSION_MODEL_SET:"):
        selector = result.split(":", 1)[1].strip()
        ok, message = await agent.set_session_model_by_selector(selector, persist=True)
        await send_text(message)
        return True
    if result == "NEW" or result.startswith("NEW:"):
        session_name = "default"
        if result.startswith("NEW:"):
            session_name = result.split(":", 1)[1].strip() or "default"
        agent.session = await agent.session_manager.create_session(name=session_name)
        agent.refresh_session_runtime_flags()
        if agent.session:
            await agent.session_manager.set_last_active_session(agent.session.id)
            await send_text(f"Started new session: {agent.session.name} ({agent.session.id})")
        return True
    if result.startswith("SESSION_SELECT:"):
        selector = result.split(":", 1)[1].strip()
        selected = await agent.session_manager.select_session(selector)
        if not selected:
            await send_text(f"Session not found: {selector}")
            return True
        agent.session = selected
        agent.refresh_session_runtime_flags()
        await agent.session_manager.set_last_active_session(selected.id)
        await send_text(f"Switched session: {selected.name} ({selected.id})")
        return True
    if result.startswith("SESSION_RENAME:"):
        new_name = result.split(":", 1)[1].strip()
        if not agent.session:
            await send_text("No active session.")
            return True
        ok = await agent.session_manager.rename_session(agent.session.id, new_name)
        if not ok:
            await send_text("Failed to rename session.")
            return True
        updated = await agent.session_manager.load_session(agent.session.id)
        if updated:
            agent.session = updated
        await send_text(f"Session renamed to: {new_name}")
        return True
    if result.startswith("CRON_ONEOFF:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            await send_text("Invalid /cron payload.")
            return True
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            await send_text('Usage: /cron "<task>"')
            return True
        await execute_prompt(prompt, f"[{sender_label} cron oneoff] {prompt}")
        return True
    # Unhandled command -> local-only message
    _ = platform
    await send_text("Command requires local console in this version.")
    return True


# ---------------------------------------------------------------------------
# Per-platform message handlers (unified)
# ---------------------------------------------------------------------------

async def handle_platform_message(
    ctx: RuntimeContext, platform: str, message: Any,
) -> None:
    """Dispatch an incoming message from any chat platform."""
    from captain_claw.prompt_execution import run_prompt_in_active_session

    adapter = PlatformAdapter(ctx, platform)
    state = ctx.get_platform_state(platform)

    try:
        # Discord: skip guild messages unless bot is mentioned
        if platform == "discord":
            is_guild = bool(str(getattr(message, "guild_id", "") or "").strip())
            discord_cfg = state.config
            requires_mention = bool(getattr(discord_cfg, "require_mention_in_guild", True))
            if is_guild and requires_mention and not bool(getattr(message, "mentioned_bot", False)):
                return

        await adapter.mark_read(message)

        # Build monitor kwargs
        monitor_kwargs: dict[str, Any] = {
            "user_id": getattr(message, "user_id", ""),
            "username": getattr(message, "username", "") or "",
            "is_command": bool(getattr(message, "text", "").strip().startswith("/")),
            "text_preview": truncate_chat_text(getattr(message, "text", "")),
        }
        if platform == "telegram":
            monitor_kwargs["chat_id"] = message.chat_id
            monitor_kwargs["message_id"] = message.message_id
        elif platform == "slack":
            monitor_kwargs["channel_id"] = message.channel_id
            monitor_kwargs["message_ts"] = message.message_ts
        elif platform == "discord":
            monitor_kwargs["channel_id"] = message.channel_id
            monitor_kwargs["guild_id"] = getattr(message, "guild_id", "") or ""
            monitor_kwargs["message_id"] = message.id
            monitor_kwargs["mentioned_bot"] = bool(getattr(message, "mentioned_bot", False))

        await adapter.monitor_event("incoming_message", **monitor_kwargs)

        # Pairing check
        user_id_key = str(getattr(message, "user_id", "")).strip()
        if user_id_key not in state.approved_users:
            await adapter.pair_unknown_user(message)
            return

        text = getattr(message, "text", "").strip()
        if not text:
            return

        # Strip Telegram-style @BotName suffix from commands (e.g. /help@MyBot -> /help)
        if text.startswith("/") and "@" in text.split()[0]:
            parts = text.split(None, 1)
            command_word = parts[0].split("@")[0]
            text = command_word if len(parts) == 1 else f"{command_word} {parts[1]}"

        channel_id = adapter._message_channel_id(message)
        reply_to = adapter._message_reply_id(message)

        # Slash command
        if text.startswith("/"):
            async def _execute_prompt(prompt: str, display_prompt: str) -> None:
                await run_prompt_in_active_session(
                    ctx, prompt,
                    display_prompt=display_prompt,
                    on_assistant_text=lambda out: adapter.send(channel_id, out, reply_to=reply_to),
                    after_turn=lambda ti, up, at: adapter.maybe_send_audio_for_turn(
                        channel_id, reply_to, up, at, ti,
                    ),
                )

            platform_labels = {"telegram": "Telegram", "slack": "Slack", "discord": "Discord"}
            sender_labels = {"telegram": "TG", "slack": "SLACK", "discord": "DISCORD"}
            await adapter.run_with_typing(
                channel_id,
                handle_remote_command(
                    ctx,
                    platform=platform,
                    raw_text=text,
                    help_label=platform_labels.get(platform, platform.title()),
                    sender_label=sender_labels.get(platform, platform.upper()),
                    send_text=lambda t: adapter.send(channel_id, t, reply_to=reply_to),
                    execute_prompt=_execute_prompt,
                ),
            )
            return

        # Normal chat message
        user_label = getattr(message, "username", "") or str(getattr(message, "user_id", ""))
        sender_labels = {"telegram": "TG", "slack": "SLACK", "discord": "DISCORD"}
        label = sender_labels.get(platform, platform.upper())
        await adapter.run_with_typing(
            channel_id,
            run_prompt_in_active_session(
                ctx, text,
                display_prompt=f"[{label} {user_label}] {text}",
                on_assistant_text=lambda out: adapter.send(channel_id, out, reply_to=reply_to),
                after_turn=lambda ti, up, at: adapter.maybe_send_audio_for_turn(
                    channel_id, reply_to, up, at, ti,
                ),
            ),
        )
    except Exception as e:
        log.error(f"{platform.title()} message handler failed", error=str(e))
        try:
            await adapter.monitor_event(
                "handler_error",
                **adapter._channel_kwargs(adapter._message_channel_id(message)),
                user_id=getattr(message, "user_id", ""),
                error=str(e),
            )
        except Exception:
            pass
        try:
            await adapter.send(
                adapter._message_channel_id(message),
                f"Error while processing your request: {e}",
                reply_to=adapter._message_reply_id(message),
            )
        except Exception:
            pass
