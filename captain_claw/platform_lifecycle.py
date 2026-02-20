"""Platform bridge initialization and teardown helpers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from captain_claw.config import get_config
from captain_claw.discord_bridge import DiscordBridge
from captain_claw.platform_adapter import PlatformAdapter, cleanup_expired_pairings
from captain_claw.slack_bridge import SlackBridge
from captain_claw.telegram_bridge import TelegramBridge

if TYPE_CHECKING:
    from captain_claw.runtime_context import RuntimeContext


async def init_platforms(ctx: RuntimeContext) -> None:
    """Initialize enabled chat platform bridges and start poll loops."""
    cfg = get_config()

    # Telegram
    telegram_cfg = cfg.telegram
    ctx.telegram.config = telegram_cfg
    ctx.telegram.enabled = bool(telegram_cfg.enabled or telegram_cfg.bot_token.strip())
    if ctx.telegram.enabled:
        token = telegram_cfg.bot_token.strip()
        if token:
            ctx.telegram.bridge = TelegramBridge(token=token, api_base_url=telegram_cfg.api_base_url)
            adapter = PlatformAdapter(ctx, "telegram")
            await adapter.load_state()
            cleanup_expired_pairings(ctx.telegram.pending_pairings)
            await adapter._save_state()
            await _register_telegram_commands(ctx)
            ctx.telegram.poll_task = asyncio.create_task(
                _telegram_poll_loop(ctx)
            )
            ctx.ui.append_system_line("Telegram UI enabled (long polling started).")
        else:
            ctx.ui.append_system_line("Telegram enabled but bot_token is empty; skipping Telegram startup.")

    # Slack
    slack_cfg = cfg.slack
    ctx.slack.config = slack_cfg
    ctx.slack.enabled = bool(slack_cfg.enabled or slack_cfg.bot_token.strip())
    if ctx.slack.enabled:
        token = slack_cfg.bot_token.strip()
        if token:
            ctx.slack.bridge = SlackBridge(token=token, api_base_url=slack_cfg.api_base_url)
            ctx.slack.offsets = {}
            adapter = PlatformAdapter(ctx, "slack")
            await adapter.load_state()
            cleanup_expired_pairings(ctx.slack.pending_pairings)
            await adapter._save_state()
            ctx.slack.poll_task = asyncio.create_task(
                _slack_poll_loop(ctx)
            )
            ctx.ui.append_system_line("Slack UI enabled (polling started).")
        else:
            ctx.ui.append_system_line("Slack enabled but bot_token is empty; skipping Slack startup.")

    # Discord
    discord_cfg = cfg.discord
    ctx.discord.config = discord_cfg
    ctx.discord.enabled = bool(discord_cfg.enabled or discord_cfg.bot_token.strip())
    if ctx.discord.enabled:
        token = discord_cfg.bot_token.strip()
        if token:
            ctx.discord.bridge = DiscordBridge(token=token, api_base_url=discord_cfg.api_base_url)
            ctx.discord.offsets = {}
            adapter = PlatformAdapter(ctx, "discord")
            await adapter.load_state()
            cleanup_expired_pairings(ctx.discord.pending_pairings)
            await adapter._save_state()
            ctx.discord.poll_task = asyncio.create_task(
                _discord_poll_loop(ctx)
            )
            ctx.ui.append_system_line("Discord UI enabled (polling started).")
        else:
            ctx.ui.append_system_line("Discord enabled but bot_token is empty; skipping Discord startup.")


async def teardown_platforms(ctx: RuntimeContext) -> None:
    """Cancel poll tasks and close bridge connections."""
    for platform_name in ctx.platform_names():
        state = ctx.get_platform_state(platform_name)
        if state.poll_task is not None:
            state.poll_task.cancel()
            try:
                await state.poll_task
            except asyncio.CancelledError:
                pass
        if state.bridge is not None:
            try:
                await state.bridge.close()
            except Exception:
                pass


async def _register_telegram_commands(ctx: RuntimeContext) -> None:
    bridge = ctx.telegram.bridge
    if not bridge:
        return
    try:
        await bridge.set_my_commands(ctx.telegram_command_specs)
        await bridge.set_chat_menu_button_commands()
        ctx.ui.append_system_line(
            f"Telegram commands registered ({len(ctx.telegram_command_specs)} entries)."
        )
    except Exception as e:
        ctx.ui.append_system_line(f"Telegram command registration failed: {e}")


async def _telegram_poll_loop(ctx: RuntimeContext) -> None:
    """Background Telegram polling and message dispatch."""
    from captain_claw.remote_command_handler import handle_platform_message

    bridge = ctx.telegram.bridge
    assert bridge is not None
    cfg = ctx.telegram.config
    while True:
        try:
            updates = await bridge.get_updates(
                offset=ctx.telegram.offsets,
                timeout=max(1, int(cfg.poll_timeout_seconds)),
            )
            for update in updates:
                next_offset = int(update.update_id) + 1
                ctx.telegram.offsets = (
                    next_offset
                    if ctx.telegram.offsets is None
                    else max(ctx.telegram.offsets, next_offset)
                )
                asyncio.create_task(handle_platform_message(ctx, "telegram", update))
        except asyncio.CancelledError:
            break
        except Exception as e:
            ctx.ui.append_system_line(f"Telegram poll error: {e}")
            await asyncio.sleep(2.0)


async def _slack_poll_loop(ctx: RuntimeContext) -> None:
    """Background Slack polling and message dispatch."""
    from captain_claw.remote_command_handler import handle_platform_message

    bridge = ctx.slack.bridge
    assert bridge is not None
    cfg = ctx.slack.config
    while True:
        try:
            updates, next_offsets = await bridge.get_updates(ctx.slack.offsets or {})
            ctx.slack.offsets = dict(next_offsets)
            for update in updates:
                asyncio.create_task(handle_platform_message(ctx, "slack", update))
            await asyncio.sleep(max(1, int(cfg.poll_timeout_seconds)))
        except asyncio.CancelledError:
            break
        except Exception as e:
            ctx.ui.append_system_line(f"Slack poll error: {e}")
            await asyncio.sleep(2.0)


async def _discord_poll_loop(ctx: RuntimeContext) -> None:
    """Background Discord polling and message dispatch."""
    from captain_claw.remote_command_handler import handle_platform_message

    bridge = ctx.discord.bridge
    assert bridge is not None
    cfg = ctx.discord.config
    while True:
        try:
            updates, next_offsets = await bridge.get_updates(ctx.discord.offsets or {})
            ctx.discord.offsets = dict(next_offsets)
            for update in updates:
                asyncio.create_task(handle_platform_message(ctx, "discord", update))
            await asyncio.sleep(max(1, int(cfg.poll_timeout_seconds)))
        except asyncio.CancelledError:
            break
        except Exception as e:
            ctx.ui.append_system_line(f"Discord poll error: {e}")
            await asyncio.sleep(2.0)
