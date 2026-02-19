"""Main entry point for Captain Claw."""

import argparse
import asyncio
import json
import os
import secrets
import shlex
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TypeVar

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from captain_claw.agent import Agent
from captain_claw.cli import TerminalUI, get_ui
from captain_claw.config import Config, get_config, set_config
from captain_claw.cron import (
    compute_next_run,
    now_utc,
    parse_schedule_tokens,
    schedule_to_text,
    to_utc_iso,
)
from captain_claw.discord_bridge import DiscordBridge, DiscordMessage
from captain_claw.execution_queue import (
    CommandLane,
    CommandQueueManager,
    FollowupQueueManager,
    FollowupRun,
    QueueSettings,
    normalize_queue_drop_policy,
    normalize_queue_mode,
    resolve_global_lane,
    resolve_session_lane,
)
from captain_claw.logging import configure_logging, log, set_system_log_sink
from captain_claw.onboarding import run_onboarding_wizard, should_run_onboarding
from captain_claw.slack_bridge import SlackBridge, SlackMessage
from captain_claw.telegram_bridge import TelegramBridge, TelegramMessage

T = TypeVar("T")


async def _run_cancellable(ui: TerminalUI, work: Awaitable[T]) -> tuple[T | None, bool]:
    """Run work and cancel on ESC."""
    work_task = asyncio.create_task(work)
    esc_task = asyncio.create_task(ui.wait_for_escape()) if ui.can_capture_escape() else None
    try:
        if esc_task is None:
            return await work_task, False

        done, _ = await asyncio.wait(
            {work_task, esc_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if esc_task in done:
            ui.append_system_line("ESC pressed, cancelling current action")
            ui.set_runtime_status("waiting")
            work_task.cancel()
            try:
                await work_task
            except asyncio.CancelledError:
                pass
            return None, True

        esc_task.cancel()
        try:
            await esc_task
        except asyncio.CancelledError:
            pass
        return await work_task, False
    finally:
        if esc_task and not esc_task.done():
            esc_task.cancel()


def _build_runtime_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captain-claw",
        add_help=False,
        description="Captain Claw - A powerful console-based AI agent",
    )
    parser.add_argument("-c", "--config", default="", help="Path to config file")
    parser.add_argument("-m", "--model", default="", help="Override model")
    parser.add_argument("-p", "--provider", default="", help="Override provider")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument(
        "--onboarding",
        action="store_true",
        help="Run interactive onboarding wizard before starting",
    )
    parser.add_argument("--version", action="store_true", help="Show version information and exit")
    parser.add_argument("--web", action="store_true", help="Start the web UI instead of the terminal UI")
    parser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")
    return parser


def _should_parse_runtime_cli_from_argv(
    config: str,
    model: str,
    provider: str,
    no_stream: bool,
    verbose: bool,
    onboarding: bool,
    web: bool = False,
) -> bool:
    if config or model or provider or no_stream or verbose or onboarding or web:
        return False
    if len(sys.argv) <= 1:
        return False
    program = Path(sys.argv[0]).name.lower()
    return (
        "captain-claw" in program
        or program in {"captain_claw", "captain_claw.py", "main.py"}
    )


def main(
    config: str = "",
    model: str = "",
    provider: str = "",
    no_stream: bool = False,
    verbose: bool = False,
    onboarding: bool = False,
    web: bool = False,
) -> None:
    """Start Captain Claw interactive session."""
    if _should_parse_runtime_cli_from_argv(
        config=config,
        model=model,
        provider=provider,
        no_stream=no_stream,
        verbose=verbose,
        onboarding=onboarding,
        web=web,
    ):
        runtime_args = list(sys.argv[1:])
        if runtime_args and runtime_args[0].strip().lower() == "run":
            runtime_args = runtime_args[1:]
        if runtime_args and runtime_args[0].strip().lower() in {"ver", "version"}:
            version()
            return
        parser = _build_runtime_arg_parser()
        parsed, unknown = parser.parse_known_args(runtime_args)
        if parsed.help:
            parser.print_help()
            return
        if parsed.version:
            version()
            return
        config = str(parsed.config or "")
        model = str(parsed.model or "")
        provider = str(parsed.provider or "")
        no_stream = bool(parsed.no_stream)
        verbose = bool(parsed.verbose)
        onboarding = bool(parsed.onboarding)
        web = bool(parsed.web)
        if unknown:
            print(f"Warning: ignoring unsupported arguments: {' '.join(unknown)}")

    set_system_log_sink(None)

    # Configure logging first
    if verbose:
        os.environ["CLAW_LOGGING__LEVEL"] = "DEBUG"
    configure_logging()

    if should_run_onboarding(
        force=onboarding,
        config_path=(config or None),
    ):
        try:
            selected_config_path = run_onboarding_wizard(
                config_path=(config or None),
                require_interactive=onboarding,
            )
            if selected_config_path is not None:
                config = str(selected_config_path)
        except KeyboardInterrupt:
            print("\nOnboarding cancelled.")
            if onboarding:
                sys.exit(1)
        except RuntimeError as e:
            log.error("Onboarding failed", error=str(e))
            print(f"Error: {e}")
            sys.exit(1)

    # Load configuration
    if config:
        try:
            cfg = Config.from_yaml(Path(config))
        except Exception as e:
            log.error("Failed to load config", error=str(e))
            cfg = Config.load()
    else:
        cfg = Config.load()

    # Apply CLI overrides
    if model:
        cfg.model.model = model
    if provider:
        cfg.model.provider = provider
    if no_stream:
        cfg.ui.streaming = False

    # Set global config
    set_config(cfg)

    # Ensure session directory exists
    session_path = Path(cfg.session.path).expanduser()
    session_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path = cfg.resolved_workspace_path(Path.cwd())
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Web UI mode
    if web or cfg.web.enabled:
        from captain_claw.web_server import run_web_server

        try:
            run_web_server(cfg)
        except KeyboardInterrupt:
            log.info("Web server shutting down...")
            sys.exit(0)
        except Exception as e:
            log.error("Web server fatal error", error=str(e))
            sys.exit(1)
        return

    ui = get_ui()
    set_system_log_sink(ui.append_system_line if ui.has_sticky_layout() else None)

    # Run the interactive loop
    try:
        asyncio.run(run_interactive())
    except KeyboardInterrupt:
        log.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        log.error("Fatal error", error=str(e))
        sys.exit(1)


async def run_interactive() -> None:
    """Run the interactive agent loop."""
    ui = get_ui()
    ui.set_monitor_full_output(bool(get_config().ui.monitor_full_output))
    agent = Agent(
        status_callback=ui.set_runtime_status,
        tool_output_callback=ui.append_tool_output,
        approval_callback=ui.confirm,
    )

    # Show welcome
    ui.print_welcome()

    # Initialize agent
    await agent.initialize()
    ui.set_monitor_mode(True)
    if agent.session:
        ui.load_monitor_tool_output_from_session(agent.session.messages)
    ui.set_runtime_status("user input")
    last_exec_seconds: float | None = None
    last_completed_at: datetime | None = None
    command_queue = CommandQueueManager()
    followup_queue = FollowupQueueManager()
    cron_running_job_ids: set[str] = set()
    cron_poll_seconds = 2.0
    telegram_cfg = get_config().telegram
    slack_cfg = get_config().slack
    discord_cfg = get_config().discord
    telegram_enabled = bool(telegram_cfg.enabled or telegram_cfg.bot_token.strip())
    slack_enabled = bool(slack_cfg.enabled or slack_cfg.bot_token.strip())
    discord_enabled = bool(discord_cfg.enabled or discord_cfg.bot_token.strip())
    telegram_bridge: TelegramBridge | None = None
    slack_bridge: SlackBridge | None = None
    discord_bridge: DiscordBridge | None = None
    telegram_poll_task: asyncio.Task[None] | None = None
    slack_poll_task: asyncio.Task[None] | None = None
    discord_poll_task: asyncio.Task[None] | None = None
    telegram_offset: int | None = None
    slack_offsets: dict[str, str] = {}
    discord_offsets: dict[str, str] = {}
    approved_telegram_users: dict[str, dict[str, object]] = {}
    pending_telegram_pairings: dict[str, dict[str, object]] = {}
    approved_slack_users: dict[str, dict[str, object]] = {}
    pending_slack_pairings: dict[str, dict[str, object]] = {}
    approved_discord_users: dict[str, dict[str, object]] = {}
    pending_discord_pairings: dict[str, dict[str, object]] = {}
    telegram_state_key_approved = "telegram_approved_users"
    telegram_state_key_pending = "telegram_pending_pairings"
    slack_state_key_approved = "slack_approved_users"
    slack_state_key_pending = "slack_pending_pairings"
    discord_state_key_approved = "discord_approved_users"
    discord_state_key_pending = "discord_pending_pairings"
    telegram_command_specs: list[tuple[str, str]] = [
        ("start", "Start Captain Claw in Telegram"),
        ("help", "Show Telegram command guide"),
        ("new", "Create a new session"),
        ("sessions", "List recent sessions"),
        ("session", "Inspect/switch session"),
        ("models", "List allowed models"),
        ("config", "Show active configuration"),
        ("history", "Show recent message history"),
        ("clear", "Clear active session messages"),
        ("compact", "Compact active session memory"),
        ("pipeline", "Set pipeline mode (loop/contracts)"),
        ("planning", "Legacy alias for pipeline"),
        ("skills", "List available skills"),
        (
            "skill",
            "Run/search/install skill: /skill <name> [args] | /skill search <criteria> | /skill install <github-url> | /skill install <skill-name> [install-id]",
        ),
        ("cron", "Run one-off cron prompt"),
    ]

    def _normalize_session_id(raw: str) -> str:
        safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in (raw or "").strip())
        safe = safe.strip("-")
        return safe or "default"

    def _utc_now_iso() -> str:
        return to_utc_iso(now_utc())

    async def _load_json_state(key: str) -> dict[str, dict[str, object]]:
        raw = await agent.session_manager.get_app_state(key)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    async def _save_telegram_state() -> None:
        await agent.session_manager.set_app_state(
            telegram_state_key_approved,
            json.dumps(approved_telegram_users, ensure_ascii=True, sort_keys=True),
        )
        await agent.session_manager.set_app_state(
            telegram_state_key_pending,
            json.dumps(pending_telegram_pairings, ensure_ascii=True, sort_keys=True),
        )

    async def _save_slack_state() -> None:
        await agent.session_manager.set_app_state(
            slack_state_key_approved,
            json.dumps(approved_slack_users, ensure_ascii=True, sort_keys=True),
        )
        await agent.session_manager.set_app_state(
            slack_state_key_pending,
            json.dumps(pending_slack_pairings, ensure_ascii=True, sort_keys=True),
        )

    async def _save_discord_state() -> None:
        await agent.session_manager.set_app_state(
            discord_state_key_approved,
            json.dumps(approved_discord_users, ensure_ascii=True, sort_keys=True),
        )
        await agent.session_manager.set_app_state(
            discord_state_key_pending,
            json.dumps(pending_discord_pairings, ensure_ascii=True, sort_keys=True),
        )

    def _generate_pairing_token(pending_map: dict[str, dict[str, object]]) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        while True:
            token = "".join(secrets.choice(alphabet) for _ in range(8))
            if token not in pending_map:
                return token

    async def _telegram_send(chat_id: int, text: str, *, reply_to_message_id: int | None = None) -> None:
        if not telegram_bridge:
            return
        try:
            await telegram_bridge.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
            )
            await _telegram_monitor_event(
                "outgoing_message",
                chat_id=chat_id,
                reply_to_message_id=reply_to_message_id or 0,
                chars=len(str(text or "")),
                text_preview=_truncate_telegram_text(text),
            )
        except Exception as e:
            ui.append_system_line(f"Telegram send failed: {str(e)}")

    async def _telegram_send_chat_action(chat_id: int, action: str = "typing") -> None:
        if not telegram_bridge:
            return
        try:
            await telegram_bridge.send_chat_action(chat_id=chat_id, action=action)
        except Exception as e:
            ui.append_system_line(f"Telegram chat action failed: {str(e)}")

    async def _telegram_mark_read(message: TelegramMessage) -> None:
        """Best-effort read mark for Business chats (Bot API limitation)."""
        if not telegram_bridge:
            return
        connection_id = str(message.business_connection_id or "").strip()
        if not connection_id:
            return
        if int(message.message_id) <= 0:
            return
        try:
            await telegram_bridge.read_business_message(
                business_connection_id=connection_id,
                chat_id=message.chat_id,
                message_id=message.message_id,
            )
        except Exception as e:
            ui.append_system_line(f"Telegram mark-read failed: {str(e)}")

    async def _slack_send(
        channel_id: str,
        text: str,
        *,
        reply_to_message_ts: str = "",
    ) -> None:
        if not slack_bridge:
            return
        try:
            await slack_bridge.send_message(
                channel_id=channel_id,
                text=text,
                reply_to_message_ts=reply_to_message_ts,
            )
            await _slack_monitor_event(
                "outgoing_message",
                channel_id=channel_id,
                reply_to_message_ts=reply_to_message_ts,
                chars=len(str(text or "")),
                text_preview=_truncate_chat_text(text),
            )
        except Exception as e:
            ui.append_system_line(f"Slack send failed: {str(e)}")

    async def _slack_send_chat_action(channel_id: str, action: str = "typing") -> None:
        if not slack_bridge:
            return
        try:
            await slack_bridge.send_chat_action(channel_id=channel_id, action=action)
        except Exception as e:
            ui.append_system_line(f"Slack chat action failed: {str(e)}")

    async def _slack_mark_read(message: SlackMessage) -> None:
        if not slack_bridge:
            return
        try:
            await slack_bridge.mark_read(channel_id=message.channel_id, message_ts=message.message_ts)
        except Exception as e:
            ui.append_system_line(f"Slack mark-read failed: {str(e)}")

    async def _discord_send(
        channel_id: str,
        text: str,
        *,
        reply_to_message_id: str = "",
    ) -> None:
        if not discord_bridge:
            return
        try:
            await discord_bridge.send_message(
                channel_id=channel_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
            )
            await _discord_monitor_event(
                "outgoing_message",
                channel_id=channel_id,
                reply_to_message_id=reply_to_message_id,
                chars=len(str(text or "")),
                text_preview=_truncate_chat_text(text),
            )
        except Exception as e:
            ui.append_system_line(f"Discord send failed: {str(e)}")

    async def _discord_send_chat_action(channel_id: str, action: str = "typing") -> None:
        if not discord_bridge:
            return
        try:
            await discord_bridge.send_chat_action(channel_id=channel_id, action=action)
        except Exception as e:
            ui.append_system_line(f"Discord chat action failed: {str(e)}")

    async def _discord_mark_read(message: DiscordMessage) -> None:
        if not discord_bridge:
            return
        try:
            await discord_bridge.mark_read(channel_id=message.channel_id, message_id=message.id)
        except Exception as e:
            ui.append_system_line(f"Discord mark-read failed: {str(e)}")

    def _truncate_chat_text(text: str, max_chars: int = 220) -> str:
        compact = str(text or "").strip().replace("\n", " ")
        if len(compact) <= max_chars:
            return compact
        return compact[:max_chars].rstrip() + "..."

    def _truncate_telegram_text(text: str, max_chars: int = 220) -> str:
        return _truncate_chat_text(text, max_chars=max_chars)

    async def _telegram_monitor_event(step: str, **args: object) -> None:
        payload: dict[str, object] = {"step": step}
        payload.update(args)
        body_lines = [f"step={step}"]
        for key, value in args.items():
            body_lines.append(f"{key}={value}")
        body_text = "\n".join(body_lines)
        ui.append_tool_output("telegram", payload, body_text)
        if agent.session:
            agent.session.add_message(
                role="tool",
                content=body_text,
                tool_name="telegram",
                tool_arguments=payload,
                token_count=agent._count_tokens(body_text),
            )
            await agent.session_manager.save_session(agent.session)

    async def _slack_monitor_event(step: str, **args: object) -> None:
        payload: dict[str, object] = {"step": step}
        payload.update(args)
        body_lines = [f"step={step}"]
        for key, value in args.items():
            body_lines.append(f"{key}={value}")
        body_text = "\n".join(body_lines)
        ui.append_tool_output("slack", payload, body_text)
        if agent.session:
            agent.session.add_message(
                role="tool",
                content=body_text,
                tool_name="slack",
                tool_arguments=payload,
                token_count=agent._count_tokens(body_text),
            )
            await agent.session_manager.save_session(agent.session)

    async def _discord_monitor_event(step: str, **args: object) -> None:
        payload: dict[str, object] = {"step": step}
        payload.update(args)
        body_lines = [f"step={step}"]
        for key, value in args.items():
            body_lines.append(f"{key}={value}")
        body_text = "\n".join(body_lines)
        ui.append_tool_output("discord", payload, body_text)
        if agent.session:
            agent.session.add_message(
                role="tool",
                content=body_text,
                tool_name="discord",
                tool_arguments=payload,
                token_count=agent._count_tokens(body_text),
            )
            await agent.session_manager.save_session(agent.session)

    def _extract_audio_paths_from_tool_output(content: str) -> list[Path]:
        """Extract resolved MP3 paths from pocket_tts tool output text."""
        text = str(content or "")
        paths: list[Path] = []
        seen: set[str] = set()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.lower().startswith("path:"):
                continue
            raw_path = stripped.split(":", 1)[1].strip()
            if " (requested:" in raw_path:
                raw_path = raw_path.split(" (requested:", 1)[0].strip()
            if not raw_path:
                continue
            try:
                resolved = Path(raw_path).expanduser().resolve()
            except Exception:
                continue
            if not resolved.exists() or not resolved.is_file():
                continue
            if resolved.suffix.lower() != ".mp3":
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            paths.append(resolved)
        return paths

    def _collect_turn_generated_audio_paths(turn_start_idx: int) -> list[Path]:
        """Collect pocket_tts generated MP3 files from current turn tool outputs."""
        if not agent.session:
            return []
        paths: list[Path] = []
        seen: set[str] = set()
        for msg in agent.session.messages[max(0, int(turn_start_idx)) :]:
            if str(msg.get("role", "")).strip().lower() != "tool":
                continue
            if str(msg.get("tool_name", "")).strip().lower() != "pocket_tts":
                continue
            if str(msg.get("content", "")).strip().lower().startswith("error:"):
                continue
            for path in _extract_audio_paths_from_tool_output(str(msg.get("content", ""))):
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                paths.append(path)
        return paths

    def _remote_user_requested_audio(text: str) -> bool:
        """Heuristic check for remote audio/TTS request intent."""
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        audio_hints = (
            "tts",
            "text to speech",
            "text-to-speech",
            "voice",
            "audio",
            "voice note",
            "read aloud",
            "mp3",
            "audio overview",
            "spoken overview",
        )
        return any(hint in lowered for hint in audio_hints)

    def _telegram_user_requested_audio(text: str) -> bool:
        return _remote_user_requested_audio(text)

    async def _telegram_send_audio_file(
        chat_id: int,
        path: Path,
        *,
        caption: str = "",
        reply_to_message_id: int | None = None,
    ) -> bool:
        """Send audio file to Telegram chat and mirror monitor event."""
        if not telegram_bridge:
            return False
        try:
            await _telegram_send_chat_action(chat_id, action="upload_document")
            await telegram_bridge.send_audio_file(
                chat_id=chat_id,
                file_path=path,
                caption=caption,
                reply_to_message_id=reply_to_message_id,
            )
            size_bytes = 0
            try:
                size_bytes = int(path.stat().st_size)
            except Exception:
                size_bytes = 0
            await _telegram_monitor_event(
                "outgoing_audio",
                chat_id=chat_id,
                reply_to_message_id=reply_to_message_id or 0,
                path=str(path),
                size_bytes=size_bytes,
            )
            return True
        except Exception as e:
            await _telegram_monitor_event(
                "outgoing_audio_error",
                chat_id=chat_id,
                reply_to_message_id=reply_to_message_id or 0,
                path=str(path),
                error=str(e),
            )
            ui.append_system_line(f"Telegram audio send failed: {str(e)}")
            return False

    async def _telegram_maybe_send_audio_for_turn(
        chat_id: int,
        reply_to_message_id: int | None,
        user_prompt: str,
        assistant_text: str,
        turn_start_idx: int,
    ) -> None:
        """Auto-deliver generated/synthesized audio for Telegram TTS-like prompts."""
        generated_paths = _collect_turn_generated_audio_paths(turn_start_idx)
        if generated_paths:
            for path in generated_paths:
                await _telegram_send_audio_file(
                    chat_id=chat_id,
                    path=path,
                    caption="Requested audio output",
                    reply_to_message_id=reply_to_message_id,
                )
            return

        if not _telegram_user_requested_audio(user_prompt):
            return
        if "pocket_tts" not in agent.tools.list_tools():
            return

        source_text = str(assistant_text or "").strip()
        if not source_text and agent.session:
            for msg in reversed(agent.session.messages):
                if str(msg.get("role", "")).strip().lower() != "assistant":
                    continue
                candidate = str(msg.get("content", "")).strip()
                if candidate:
                    source_text = candidate
                    break
        if not source_text:
            return

        cfg = get_config()
        max_chars = max(1, int(getattr(cfg.tools.pocket_tts, "max_chars", 4000)))
        tts_text = source_text[:max_chars]
        tool_args: dict[str, object] = {"text": tts_text}
        try:
            result = await agent._execute_tool_with_guard(
                name="pocket_tts",
                arguments=tool_args,
                interaction_label="telegram_auto_tts",
            )
            tool_output = result.content if result.success else f"Error: {result.error}"
            agent._add_session_message(
                role="tool",
                content=tool_output,
                tool_name="pocket_tts",
                tool_arguments=tool_args,
            )
            agent._emit_tool_output("pocket_tts", tool_args, tool_output)
            if agent.session:
                await agent.session_manager.save_session(agent.session)

            if not result.success:
                await _telegram_send(
                    chat_id,
                    f"Audio generation failed: {str(result.error or 'unknown error')}",
                    reply_to_message_id=reply_to_message_id,
                )
                return

            paths = _extract_audio_paths_from_tool_output(tool_output)
            for path in paths:
                await _telegram_send_audio_file(
                    chat_id=chat_id,
                    path=path,
                    caption="Audio summary",
                    reply_to_message_id=reply_to_message_id,
                )
        except Exception as e:
            await _telegram_monitor_event(
                "auto_tts_error",
                chat_id=chat_id,
                reply_to_message_id=reply_to_message_id or 0,
                error=str(e),
            )
            ui.append_system_line(f"Telegram auto TTS failed: {str(e)}")

    async def _slack_send_audio_file(
        channel_id: str,
        path: Path,
        *,
        caption: str = "",
    ) -> bool:
        if not slack_bridge:
            return False
        try:
            await slack_bridge.send_audio_file(
                channel_id=channel_id,
                file_path=path,
                caption=caption,
            )
            size_bytes = 0
            try:
                size_bytes = int(path.stat().st_size)
            except Exception:
                size_bytes = 0
            await _slack_monitor_event(
                "outgoing_audio",
                channel_id=channel_id,
                path=str(path),
                size_bytes=size_bytes,
            )
            return True
        except Exception as e:
            await _slack_monitor_event(
                "outgoing_audio_error",
                channel_id=channel_id,
                path=str(path),
                error=str(e),
            )
            ui.append_system_line(f"Slack audio send failed: {str(e)}")
            return False

    async def _discord_send_audio_file(
        channel_id: str,
        path: Path,
        *,
        caption: str = "",
        reply_to_message_id: str = "",
    ) -> bool:
        if not discord_bridge:
            return False
        try:
            await discord_bridge.send_audio_file(
                channel_id=channel_id,
                file_path=path,
                caption=caption,
                reply_to_message_id=reply_to_message_id,
            )
            size_bytes = 0
            try:
                size_bytes = int(path.stat().st_size)
            except Exception:
                size_bytes = 0
            await _discord_monitor_event(
                "outgoing_audio",
                channel_id=channel_id,
                reply_to_message_id=reply_to_message_id,
                path=str(path),
                size_bytes=size_bytes,
            )
            return True
        except Exception as e:
            await _discord_monitor_event(
                "outgoing_audio_error",
                channel_id=channel_id,
                reply_to_message_id=reply_to_message_id,
                path=str(path),
                error=str(e),
            )
            ui.append_system_line(f"Discord audio send failed: {str(e)}")
            return False

    async def _maybe_send_audio_for_turn(
        *,
        user_prompt: str,
        assistant_text: str,
        turn_start_idx: int,
        interaction_label: str,
        send_audio: Callable[[Path, str], Awaitable[bool]],
        send_error_text: Callable[[str], Awaitable[None]],
        monitor_error: Callable[[str], Awaitable[None]],
    ) -> None:
        generated_paths = _collect_turn_generated_audio_paths(turn_start_idx)
        if generated_paths:
            for path in generated_paths:
                await send_audio(path, "Requested audio output")
            return
        if not _remote_user_requested_audio(user_prompt):
            return
        if "pocket_tts" not in agent.tools.list_tools():
            return

        source_text = str(assistant_text or "").strip()
        if not source_text and agent.session:
            for msg in reversed(agent.session.messages):
                if str(msg.get("role", "")).strip().lower() != "assistant":
                    continue
                candidate = str(msg.get("content", "")).strip()
                if candidate:
                    source_text = candidate
                    break
        if not source_text:
            return

        cfg = get_config()
        max_chars = max(1, int(getattr(cfg.tools.pocket_tts, "max_chars", 4000)))
        tts_text = source_text[:max_chars]
        tool_args: dict[str, object] = {"text": tts_text}
        try:
            result = await agent._execute_tool_with_guard(
                name="pocket_tts",
                arguments=tool_args,
                interaction_label=interaction_label,
            )
            tool_output = result.content if result.success else f"Error: {result.error}"
            agent._add_session_message(
                role="tool",
                content=tool_output,
                tool_name="pocket_tts",
                tool_arguments=tool_args,
            )
            agent._emit_tool_output("pocket_tts", tool_args, tool_output)
            if agent.session:
                await agent.session_manager.save_session(agent.session)

            if not result.success:
                await send_error_text(f"Audio generation failed: {str(result.error or 'unknown error')}")
                return

            for path in _extract_audio_paths_from_tool_output(tool_output):
                await send_audio(path, "Audio summary")
        except Exception as e:
            await monitor_error(str(e))

    async def _slack_maybe_send_audio_for_turn(
        channel_id: str,
        user_prompt: str,
        assistant_text: str,
        turn_start_idx: int,
    ) -> None:
        await _maybe_send_audio_for_turn(
            user_prompt=user_prompt,
            assistant_text=assistant_text,
            turn_start_idx=turn_start_idx,
            interaction_label="slack_auto_tts",
            send_audio=lambda path, caption: _slack_send_audio_file(
                channel_id=channel_id,
                path=path,
                caption=caption,
            ),
            send_error_text=lambda text: _slack_send(channel_id=channel_id, text=text),
            monitor_error=lambda error: _slack_monitor_event(
                "auto_tts_error",
                channel_id=channel_id,
                error=error,
            ),
        )

    async def _discord_maybe_send_audio_for_turn(
        channel_id: str,
        reply_to_message_id: str,
        user_prompt: str,
        assistant_text: str,
        turn_start_idx: int,
    ) -> None:
        await _maybe_send_audio_for_turn(
            user_prompt=user_prompt,
            assistant_text=assistant_text,
            turn_start_idx=turn_start_idx,
            interaction_label="discord_auto_tts",
            send_audio=lambda path, caption: _discord_send_audio_file(
                channel_id=channel_id,
                path=path,
                caption=caption,
                reply_to_message_id=reply_to_message_id,
            ),
            send_error_text=lambda text: _discord_send(
                channel_id=channel_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
            ),
            monitor_error=lambda error: _discord_monitor_event(
                "auto_tts_error",
                channel_id=channel_id,
                reply_to_message_id=reply_to_message_id,
                error=error,
            ),
        )

    async def _run_with_telegram_typing(
        chat_id: int,
        work: Awaitable[T],
        *,
        heartbeat_seconds: float = 4.0,
    ) -> T:
        """Emit Telegram `typing` indicator while async work is running."""
        if not telegram_bridge:
            return await work

        stop = asyncio.Event()

        async def _typing_heartbeat() -> None:
            interval = max(1.0, float(heartbeat_seconds))
            while not stop.is_set():
                await _telegram_send_chat_action(chat_id, action="typing")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                except TimeoutError:
                    continue

        heartbeat = asyncio.create_task(_typing_heartbeat())
        try:
            return await work
        finally:
            stop.set()
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

    async def _run_with_slack_typing(
        channel_id: str,
        work: Awaitable[T],
        *,
        heartbeat_seconds: float = 4.0,
    ) -> T:
        if not slack_bridge:
            return await work
        stop = asyncio.Event()

        async def _typing_heartbeat() -> None:
            interval = max(1.0, float(heartbeat_seconds))
            while not stop.is_set():
                await _slack_send_chat_action(channel_id, action="typing")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                except TimeoutError:
                    continue

        heartbeat = asyncio.create_task(_typing_heartbeat())
        try:
            return await work
        finally:
            stop.set()
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

    async def _run_with_discord_typing(
        channel_id: str,
        work: Awaitable[T],
        *,
        heartbeat_seconds: float = 4.0,
    ) -> T:
        if not discord_bridge:
            return await work
        stop = asyncio.Event()

        async def _typing_heartbeat() -> None:
            interval = max(1.0, float(heartbeat_seconds))
            while not stop.is_set():
                await _discord_send_chat_action(channel_id, action="typing")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                except TimeoutError:
                    continue

        heartbeat = asyncio.create_task(_typing_heartbeat())
        try:
            return await work
        finally:
            stop.set()
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

    def _chat_user_display(record: dict[str, object]) -> str:
        user_id = str(record.get("user_id", "")).strip()
        username = str(record.get("username", "")).strip()
        first_name = str(record.get("first_name", "")).strip()
        if username:
            return f"@{username} ({user_id})"
        if first_name:
            return f"{first_name} ({user_id})"
        return user_id or "unknown"

    def _telegram_user_display(record: dict[str, object]) -> str:
        return _chat_user_display(record)

    def _cleanup_expired_pairings(pending_map: dict[str, dict[str, object]]) -> None:
        if not pending_map:
            return
        now_ts = now_utc().timestamp()
        expired = []
        for token, payload in pending_map.items():
            expires_at_raw = str(payload.get("expires_at", "")).strip()
            if not expires_at_raw:
                continue
            try:
                expires_at = datetime.fromisoformat(expires_at_raw.replace("Z", "+00:00"))
                if expires_at.timestamp() <= now_ts:
                    expired.append(token)
            except Exception:
                expired.append(token)
        for token in expired:
            pending_map.pop(token, None)

    async def _approve_telegram_pairing_token(raw_token: str) -> tuple[bool, str]:
        token = str(raw_token or "").strip().upper()
        if not token:
            return False, "Usage: /approve user telegram <token>"
        _cleanup_expired_pairings(pending_telegram_pairings)
        record = pending_telegram_pairings.get(token)
        if not isinstance(record, dict):
            return False, f"Telegram pairing token not found or expired: {token}"
        user_id = str(record.get("user_id", "")).strip()
        if not user_id:
            pending_telegram_pairings.pop(token, None)
            await _save_telegram_state()
            return False, f"Telegram pairing token invalid: {token}"

        approved_telegram_users[user_id] = {
            "user_id": int(record.get("user_id", 0) or 0),
            "chat_id": int(record.get("chat_id", 0) or 0),
            "username": str(record.get("username", "")).strip(),
            "first_name": str(record.get("first_name", "")).strip(),
            "approved_at": _utc_now_iso(),
            "token": token,
        }
        pending_telegram_pairings.pop(token, None)
        await _save_telegram_state()

        chat_id = int(approved_telegram_users[user_id].get("chat_id", 0) or 0)
        if chat_id and telegram_bridge:
            await _telegram_send(
                chat_id,
                (
                    "Pairing approved. You can now use Captain Claw.\n"
                    "All chat and supported slash commands are available."
                ),
            )
        return True, f"Approved Telegram user: {_chat_user_display(approved_telegram_users[user_id])}"

    async def _approve_slack_pairing_token(raw_token: str) -> tuple[bool, str]:
        token = str(raw_token or "").strip().upper()
        if not token:
            return False, "Usage: /approve user slack <token>"
        _cleanup_expired_pairings(pending_slack_pairings)
        record = pending_slack_pairings.get(token)
        if not isinstance(record, dict):
            return False, f"Slack pairing token not found or expired: {token}"
        user_id = str(record.get("user_id", "")).strip()
        if not user_id:
            pending_slack_pairings.pop(token, None)
            await _save_slack_state()
            return False, f"Slack pairing token invalid: {token}"

        approved_slack_users[user_id] = {
            "user_id": user_id,
            "channel_id": str(record.get("channel_id", "")).strip(),
            "username": str(record.get("username", "")).strip(),
            "first_name": "",
            "approved_at": _utc_now_iso(),
            "token": token,
        }
        pending_slack_pairings.pop(token, None)
        await _save_slack_state()

        channel_id = str(approved_slack_users[user_id].get("channel_id", "")).strip()
        if channel_id and slack_bridge:
            await _slack_send(
                channel_id,
                (
                    "Pairing approved. You can now use Captain Claw.\n"
                    "All chat and supported slash-style commands are available."
                ),
            )
        return True, f"Approved Slack user: {_chat_user_display(approved_slack_users[user_id])}"

    async def _approve_discord_pairing_token(raw_token: str) -> tuple[bool, str]:
        token = str(raw_token or "").strip().upper()
        if not token:
            return False, "Usage: /approve user discord <token>"
        _cleanup_expired_pairings(pending_discord_pairings)
        record = pending_discord_pairings.get(token)
        if not isinstance(record, dict):
            return False, f"Discord pairing token not found or expired: {token}"
        user_id = str(record.get("user_id", "")).strip()
        if not user_id:
            pending_discord_pairings.pop(token, None)
            await _save_discord_state()
            return False, f"Discord pairing token invalid: {token}"

        approved_discord_users[user_id] = {
            "user_id": user_id,
            "channel_id": str(record.get("channel_id", "")).strip(),
            "username": str(record.get("username", "")).strip(),
            "first_name": "",
            "approved_at": _utc_now_iso(),
            "token": token,
        }
        pending_discord_pairings.pop(token, None)
        await _save_discord_state()

        channel_id = str(approved_discord_users[user_id].get("channel_id", "")).strip()
        if channel_id and discord_bridge:
            await _discord_send(
                channel_id,
                (
                    "Pairing approved. You can now use Captain Claw.\n"
                    "All chat and supported slash-style commands are available."
                ),
            )
        return True, f"Approved Discord user: {_chat_user_display(approved_discord_users[user_id])}"

    async def _approve_chat_pairing_token(platform: str, raw_token: str) -> tuple[bool, str]:
        target = str(platform or "").strip().lower()
        if target == "telegram":
            return await _approve_telegram_pairing_token(raw_token)
        if target == "slack":
            return await _approve_slack_pairing_token(raw_token)
        if target == "discord":
            return await _approve_discord_pairing_token(raw_token)
        return False, "Usage: /approve user <telegram|slack|discord> <token>"

    async def _pair_unknown_telegram_user(message: TelegramMessage) -> None:
        _cleanup_expired_pairings(pending_telegram_pairings)
        user_id_key = str(message.user_id)
        if user_id_key in approved_telegram_users:
            return
        existing_token = ""
        for token, payload in pending_telegram_pairings.items():
            if str(payload.get("user_id", "")).strip() == str(message.user_id):
                existing_token = token
                break
        if not existing_token:
            existing_token = _generate_pairing_token(pending_telegram_pairings)
            ttl_minutes = max(1, int(telegram_cfg.pairing_ttl_minutes))
            expires = datetime.fromtimestamp(now_utc().timestamp() + ttl_minutes * 60, tz=now_utc().tzinfo)
            pending_telegram_pairings[existing_token] = {
                "user_id": message.user_id,
                "chat_id": message.chat_id,
                "username": message.username,
                "first_name": message.first_name,
                "created_at": _utc_now_iso(),
                "expires_at": expires.isoformat(),
            }
            await _save_telegram_state()

        await _telegram_send(
            message.chat_id,
            (
                "Pairing required.\n"
                f"Your pairing token: `{existing_token}`\n\n"
                "Ask the Captain Claw operator to approve you with:\n"
                f"/approve user telegram {existing_token}"
            ),
            reply_to_message_id=message.message_id,
        )

    async def _pair_unknown_slack_user(message: SlackMessage) -> None:
        _cleanup_expired_pairings(pending_slack_pairings)
        user_id_key = str(message.user_id).strip()
        if user_id_key in approved_slack_users:
            return
        existing_token = ""
        for token, payload in pending_slack_pairings.items():
            if str(payload.get("user_id", "")).strip() == user_id_key:
                existing_token = token
                break
        if not existing_token:
            existing_token = _generate_pairing_token(pending_slack_pairings)
            ttl_minutes = max(1, int(slack_cfg.pairing_ttl_minutes))
            expires = datetime.fromtimestamp(now_utc().timestamp() + ttl_minutes * 60, tz=now_utc().tzinfo)
            pending_slack_pairings[existing_token] = {
                "user_id": user_id_key,
                "channel_id": message.channel_id,
                "username": message.username,
                "first_name": "",
                "created_at": _utc_now_iso(),
                "expires_at": expires.isoformat(),
            }
            await _save_slack_state()

        await _slack_send(
            message.channel_id,
            (
                "Pairing required.\n"
                f"Your pairing token: `{existing_token}`\n\n"
                "Ask the Captain Claw operator to approve you with:\n"
                f"/approve user slack {existing_token}"
            ),
            reply_to_message_ts=message.message_ts,
        )

    async def _pair_unknown_discord_user(message: DiscordMessage) -> None:
        _cleanup_expired_pairings(pending_discord_pairings)
        user_id_key = str(message.user_id).strip()
        if user_id_key in approved_discord_users:
            return
        existing_token = ""
        for token, payload in pending_discord_pairings.items():
            if str(payload.get("user_id", "")).strip() == user_id_key:
                existing_token = token
                break
        if not existing_token:
            existing_token = _generate_pairing_token(pending_discord_pairings)
            ttl_minutes = max(1, int(discord_cfg.pairing_ttl_minutes))
            expires = datetime.fromtimestamp(now_utc().timestamp() + ttl_minutes * 60, tz=now_utc().tzinfo)
            pending_discord_pairings[existing_token] = {
                "user_id": user_id_key,
                "channel_id": message.channel_id,
                "username": message.username,
                "first_name": "",
                "created_at": _utc_now_iso(),
                "expires_at": expires.isoformat(),
            }
            await _save_discord_state()

        await _discord_send(
            message.channel_id,
            (
                "Pairing required.\n"
                f"Your pairing token: `{existing_token}`\n\n"
                "Ask the Captain Claw operator to approve you with:\n"
                f"/approve user discord {existing_token}"
            ),
            reply_to_message_id=message.id,
        )

    async def _enqueue_agent_task(
        session_id: str | None,
        task: Callable[[], Awaitable[T]],
        *,
        lane: str = CommandLane.MAIN,
        warn_after_ms: int = 2_000,
    ) -> T:
        resolved_session_lane = resolve_session_lane((session_id or "").strip() or "default")
        resolved_global_lane = resolve_global_lane(lane)
        return await command_queue.enqueue_in_lane(
            resolved_session_lane,
            lambda: command_queue.enqueue_in_lane(
                resolved_global_lane,
                lambda: command_queue.enqueue_in_lane(
                    CommandLane.AGENT_RUNTIME,
                    task,
                    warn_after_ms=warn_after_ms,
                ),
                warn_after_ms=warn_after_ms,
            ),
            warn_after_ms=warn_after_ms,
        )

    def _safe_int(value: object, default: int, minimum: int) -> int:
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except Exception:
            return default
        return max(minimum, parsed)

    async def _resolve_queue_settings_for_session(session_id: str) -> QueueSettings:
        cfg_queue = get_config().execution_queue
        session = agent.session if agent.session and agent.session.id == session_id else await agent.session_manager.load_session(session_id)
        queue_meta: dict[str, object] = {}
        if session and isinstance(session.metadata.get("queue"), dict):
            queue_meta = dict(session.metadata.get("queue") or {})
        mode = (
            normalize_queue_mode(str(queue_meta.get("mode", "")).strip())
            or normalize_queue_mode(str(getattr(cfg_queue, "mode", "")).strip())
            or "collect"
        )
        drop_policy = (
            normalize_queue_drop_policy(str(queue_meta.get("drop_policy", "")).strip())
            or normalize_queue_drop_policy(str(getattr(cfg_queue, "drop", "")).strip())
            or "summarize"
        )
        debounce_ms = _safe_int(
            queue_meta.get("debounce_ms", getattr(cfg_queue, "debounce_ms", 1000)),
            default=1000,
            minimum=0,
        )
        cap = _safe_int(
            queue_meta.get("cap", getattr(cfg_queue, "cap", 20)),
            default=20,
            minimum=1,
        )
        return QueueSettings(
            mode=mode,
            debounce_ms=debounce_ms,
            cap=cap,
            drop_policy=drop_policy,
        )

    async def _wait_until_session_idle(session_id: str) -> None:
        session_lane = resolve_session_lane(session_id)
        while command_queue.get_queue_size(session_lane) > 0:
            await asyncio.sleep(0.05)

    async def _run_queued_followup_prompt(run: FollowupRun) -> None:
        payload = dict(run.metadata)
        session_id = str(payload.get("session_id", "")).strip()
        prompt_text = str(run.prompt or "").strip()
        if not session_id or not prompt_text:
            return
        await _run_prompt_in_session(
            session_id=session_id,
            prompt_text=prompt_text,
            source=str(payload.get("source", "followup")).strip() or "followup",
            cron_job_id=str(payload.get("cron_job_id", "")).strip() or None,
            trigger=str(payload.get("trigger", "scheduled")).strip() or "scheduled",
        )

    async def _dispatch_prompt_in_session(
        session_id: str,
        prompt_text: str,
        source: str,
        *,
        cron_job_id: str | None = None,
        trigger: str = "scheduled",
        dedupe_mode: str = "prompt",
    ) -> str:
        session_lane = resolve_session_lane(session_id)
        is_busy = command_queue.get_queue_size(session_lane) > 0
        has_followup_backlog = followup_queue.get_queue_depth(session_id) > 0
        if not is_busy and not has_followup_backlog:
            await _run_prompt_in_session(
                session_id=session_id,
                prompt_text=prompt_text,
                source=source,
                cron_job_id=cron_job_id,
                trigger=trigger,
            )
            return "executed"

        settings = await _resolve_queue_settings_for_session(session_id)
        if settings.mode == "interrupt":
            lane_cleared = command_queue.clear_lane(session_lane)
            followup_cleared = followup_queue.clear_queue(session_id)
            await _cron_monitor_event(
                "followup_interrupt",
                history_job_id=cron_job_id,
                session_id=session_id,
                lane_cleared=lane_cleared,
                followup_cleared=followup_cleared,
                source=source,
            )
            await _cron_chat_event(
                cron_job_id,
                "system",
                (
                    f"[CRON] followup interrupt requested for session={session_id} "
                    f"(cleared lane={lane_cleared}, followup={followup_cleared})"
                ),
                trigger=trigger,
                source=source,
            )

        queued = followup_queue.enqueue_followup(
            session_id,
            FollowupRun(
                prompt=prompt_text,
                enqueued_at_ms=int(asyncio.get_running_loop().time() * 1000),
                message_id=(cron_job_id or ""),
                summary_line=prompt_text[:180],
                metadata={
                    "session_id": session_id,
                    "source": source,
                    "trigger": trigger,
                    "cron_job_id": cron_job_id or "",
                },
            ),
            settings,
            dedupe_mode=dedupe_mode if dedupe_mode in {"message-id", "prompt", "none"} else "prompt",
        )
        if not queued:
            await _cron_monitor_event(
                "followup_skipped",
                history_job_id=cron_job_id,
                session_id=session_id,
                source=source,
                reason="deduplicated_or_drop_policy",
            )
            return "skipped"

        followup_queue.schedule_drain(
            session_id,
            _run_queued_followup_prompt,
            wait_until_ready=lambda: _wait_until_session_idle(session_id),
        )
        await _cron_monitor_event(
            "followup_queued",
            history_job_id=cron_job_id,
            session_id=session_id,
            source=source,
            mode=settings.mode,
            queued_depth=followup_queue.get_queue_depth(session_id),
        )
        await _cron_chat_event(
            cron_job_id,
            "system",
            f"[CRON] queued followup for busy session={session_id} mode={settings.mode}",
            trigger=trigger,
            source=source,
        )
        return "queued"

    def _cron_monitor(step: str, **args: object) -> None:
        payload: dict[str, object] = {"step": step}
        payload.update(args)
        lines = [f"step={step}"]
        for key, value in args.items():
            lines.append(f"{key}={value}")
        ui.append_tool_output("cron", payload, "\n".join(lines))

    async def _append_cron_history(
        job_id: str | None = None,
        *,
        chat_event: dict[str, object] | None = None,
        monitor_event: dict[str, object] | None = None,
    ) -> None:
        if not job_id:
            return
        await agent.session_manager.append_cron_job_history(
            job_id,
            chat_event=chat_event,
            monitor_event=monitor_event,
        )

    async def _cron_monitor_event(step: str, history_job_id: str | None = None, **args: object) -> None:
        _cron_monitor(step, **args)
        monitor_event: dict[str, object] = {"timestamp": to_utc_iso(now_utc()), "step": step}
        monitor_event.update(args)
        await _append_cron_history(history_job_id, monitor_event=monitor_event)

    async def _cron_chat_event(
        job_id: str | None,
        role: str,
        content: str,
        **meta: object,
    ) -> None:
        event: dict[str, object] = {
            "timestamp": to_utc_iso(now_utc()),
            "role": role,
            "content": content,
        }
        event.update(meta)
        await _append_cron_history(job_id, chat_event=event)

    def _queue_meta(session_obj: object) -> dict[str, object]:
        if not hasattr(session_obj, "metadata"):
            return {}
        metadata = getattr(session_obj, "metadata", {})
        if not isinstance(metadata, dict):
            return {}
        raw = metadata.get("queue")
        if isinstance(raw, dict):
            return raw
        return {}

    async def _update_active_session_queue_settings(
        *,
        mode: str | None = None,
        debounce_ms: int | None = None,
        cap: int | None = None,
        drop_policy: str | None = None,
    ) -> tuple[bool, str]:
        if not agent.session:
            return False, "No active session"
        queue_meta = dict(_queue_meta(agent.session))
        if mode is not None:
            normalized_mode = normalize_queue_mode(mode)
            if not normalized_mode:
                return False, "Invalid queue mode. Use steer|followup|collect|steer-backlog|interrupt|queue."
            queue_meta["mode"] = normalized_mode
        if debounce_ms is not None:
            queue_meta["debounce_ms"] = max(0, int(debounce_ms))
        if cap is not None:
            queue_meta["cap"] = max(1, int(cap))
        if drop_policy is not None:
            normalized_drop = normalize_queue_drop_policy(drop_policy)
            if not normalized_drop:
                return False, "Invalid queue drop policy. Use old|new|summarize."
            queue_meta["drop_policy"] = normalized_drop
        queue_meta["updated_at"] = datetime.now().isoformat()
        agent.session.metadata["queue"] = queue_meta
        await agent.session_manager.save_session(agent.session)
        settings = await _resolve_queue_settings_for_session(agent.session.id)
        return (
            True,
            (
                "Session queue settings updated: "
                f"mode={settings.mode} debounce_ms={settings.debounce_ms} "
                f"cap={settings.cap} drop={settings.drop_policy}"
            ),
        )

    def _truncate_history_text(text: str, max_chars: int = 8000) -> str:
        cleaned = str(text or "")
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "... [truncated]"

    def _render_chat_export_markdown(
        session_id: str,
        session_name: str,
        messages: list[dict[str, object]],
    ) -> str:
        chat_messages = [msg for msg in messages if str(msg.get("role", "")).lower() in {"user", "assistant", "system"}]
        lines = [
            "# Session Chat Export",
            f"- Exported at (UTC): {to_utc_iso(now_utc())}",
            f"- Session ID: {session_id}",
            f"- Session name: {session_name}",
            f"- Messages: {len(chat_messages)}",
            "",
        ]
        if not chat_messages:
            lines.append("(no chat messages found)")
            lines.append("")
            return "\n".join(lines)

        for idx, msg in enumerate(chat_messages, start=1):
            role = str(msg.get("role", "unknown")).strip() or "unknown"
            timestamp = str(msg.get("timestamp", "")).strip()
            content = str(msg.get("content", ""))
            lines.append(f"## {idx}. role={role} timestamp={timestamp or '-'}")
            lines.append(content if content else "(empty)")
            lines.append("")
        return "\n".join(lines)

    def _render_monitor_export_markdown(
        session_id: str,
        session_name: str,
        messages: list[dict[str, object]],
    ) -> str:
        monitor_messages = [msg for msg in messages if str(msg.get("role", "")).lower() == "tool"]
        lines = [
            "# Session Monitor Export",
            f"- Exported at (UTC): {to_utc_iso(now_utc())}",
            f"- Session ID: {session_id}",
            f"- Session name: {session_name}",
            f"- Monitor entries: {len(monitor_messages)}",
            "",
        ]
        if not monitor_messages:
            lines.append("(no monitor/tool messages found)")
            lines.append("")
            return "\n".join(lines)

        for idx, msg in enumerate(monitor_messages, start=1):
            tool_name = str(msg.get("tool_name") or "tool")
            timestamp = str(msg.get("timestamp", "")).strip()
            args = msg.get("tool_arguments")
            if isinstance(args, dict):
                try:
                    args_text = json.dumps(args, ensure_ascii=True, sort_keys=True)
                except Exception:
                    args_text = str(args)
            else:
                args_text = "{}"
            content = str(msg.get("content", ""))
            lines.append(f"## {idx}. tool={tool_name} timestamp={timestamp or '-'}")
            lines.append(f"args={args_text}")
            lines.append(content if content else "(empty)")
            lines.append("")
        return "\n".join(lines)

    def _render_pipeline_export_jsonl(
        session_id: str,
        session_name: str,
        messages: list[dict[str, object]],
    ) -> str:
        entries = _collect_pipeline_trace_entries(session_id, session_name, messages)
        return "\n".join(json.dumps(item, ensure_ascii=True, sort_keys=True) for item in entries)

    def _collect_pipeline_trace_entries(
        session_id: str,
        session_name: str,
        messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        pipeline_messages = [
            msg
            for msg in messages
            if str(msg.get("role", "")).lower() == "tool"
            and str(msg.get("tool_name", "")).strip().lower() == "pipeline_trace"
        ]

        entries: list[dict[str, object]] = []
        for idx, msg in enumerate(pipeline_messages, start=1):
            args = msg.get("tool_arguments")
            payload = dict(args) if isinstance(args, dict) else {}
            payload["seq"] = idx
            payload["timestamp"] = str(msg.get("timestamp", "")).strip()
            payload["session_id"] = session_id
            payload["session_name"] = session_name
            entries.append(payload)

        if not entries:
            fallback_sources = {"planning", "task_contract", "completion_gate"}
            for idx, msg in enumerate(messages, start=1):
                if str(msg.get("role", "")).lower() != "tool":
                    continue
                source = str(msg.get("tool_name", "")).strip().lower()
                if source not in fallback_sources:
                    continue
                args = msg.get("tool_arguments")
                payload = dict(args) if isinstance(args, dict) else {}
                payload["source"] = source
                payload["seq"] = idx
                payload["timestamp"] = str(msg.get("timestamp", "")).strip()
                payload["session_id"] = session_id
                payload["session_name"] = session_name
                entries.append(payload)

        return entries

    def _render_pipeline_summary_markdown(
        session_id: str,
        session_name: str,
        messages: list[dict[str, object]],
    ) -> str:
        entries = _collect_pipeline_trace_entries(session_id, session_name, messages)
        lines = [
            "# Session Pipeline Trace Summary",
            f"- Exported at (UTC): {to_utc_iso(now_utc())}",
            f"- Session ID: {session_id}",
            f"- Session name: {session_name}",
            f"- Trace entries: {len(entries)}",
            "",
        ]
        if not entries:
            lines.append("(no pipeline trace entries found)")
            lines.append("")
            return "\n".join(lines)

        by_source: dict[str, int] = {}
        by_step: dict[str, int] = {}
        by_event: dict[str, int] = {}
        first_ts = str(entries[0].get("timestamp", "")).strip()
        last_ts = str(entries[-1].get("timestamp", "")).strip()
        for entry in entries:
            source = str(entry.get("source", "")).strip() or "unknown"
            by_source[source] = by_source.get(source, 0) + 1
            step = str(entry.get("step", "")).strip()
            if step:
                by_step[step] = by_step.get(step, 0) + 1
            event = str(entry.get("event", "")).strip()
            if event:
                by_event[event] = by_event.get(event, 0) + 1

        lines.append(f"- First entry timestamp: {first_ts or '-'}")
        lines.append(f"- Last entry timestamp: {last_ts or '-'}")
        lines.append("")
        lines.append("## Sources")
        for source, count in sorted(by_source.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- {source}: {count}")
        lines.append("")
        if by_event:
            lines.append("## Planning Events")
            for event, count in sorted(by_event.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- {event}: {count}")
            lines.append("")
        if by_step:
            lines.append("## Completion/Contract Steps")
            for step, count in sorted(by_step.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- {step}: {count}")
            lines.append("")

        lines.append("## Timeline")
        for entry in entries:
            seq = int(entry.get("seq", 0))
            timestamp = str(entry.get("timestamp", "")).strip() or "-"
            source = str(entry.get("source", "")).strip() or "unknown"
            item = f"{seq}. [{timestamp}] source={source}"
            event = str(entry.get("event", "")).strip()
            step = str(entry.get("step", "")).strip()
            if event:
                item += f" event={event}"
            if step:
                item += f" step={step}"
            leaf_index = entry.get("leaf_index")
            leaf_tasks = entry.get("leaf_tasks")
            leaf_remaining = entry.get("leaf_remaining")
            if isinstance(leaf_index, int) and isinstance(leaf_tasks, int):
                item += f" progress={leaf_index}/{leaf_tasks}"
            if isinstance(leaf_remaining, int):
                item += f" remaining={leaf_remaining}"
            current_path = str(entry.get("current_path", "")).strip()
            if current_path:
                item += f" path={current_path}"
            eta_text = str(entry.get("eta_text", "")).strip()
            if eta_text:
                item += f" eta={eta_text}"
            lines.append(item)
        lines.append("")
        return "\n".join(lines)

    def _export_active_session_history(mode: str) -> list[Path]:
        if not agent.session:
            return []
        mode_key = (mode or "all").strip().lower()
        if mode_key not in {"chat", "monitor", "pipeline", "pipeline-summary", "all"}:
            mode_key = "all"

        session_id = str(agent.session.id)
        session_name = str(agent.session.name)
        safe_session = _normalize_session_id(session_id)
        snapshot: list[dict[str, object]] = [dict(msg) for msg in agent.session.messages]

        saved_root = agent.tools.get_saved_base_path(create=True)
        export_root = (saved_root / "showcase" / safe_session / "exports").resolve()
        export_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        written: list[Path] = []

        if mode_key in {"chat", "all"}:
            chat_path = export_root / f"chat-{stamp}.md"
            chat_path.write_text(
                _render_chat_export_markdown(session_id, session_name, snapshot) + "\n",
                encoding="utf-8",
            )
            written.append(chat_path)

        if mode_key in {"monitor", "all"}:
            monitor_path = export_root / f"monitor-{stamp}.md"
            monitor_path.write_text(
                _render_monitor_export_markdown(session_id, session_name, snapshot) + "\n",
                encoding="utf-8",
            )
            written.append(monitor_path)

        if mode_key in {"pipeline", "all"}:
            pipeline_path = export_root / f"pipeline-{stamp}.jsonl"
            pipeline_path.write_text(
                _render_pipeline_export_jsonl(session_id, session_name, snapshot) + "\n",
                encoding="utf-8",
            )
            written.append(pipeline_path)
        if mode_key in {"pipeline-summary", "all"}:
            pipeline_summary_path = export_root / f"pipeline-summary-{stamp}.md"
            pipeline_summary_path.write_text(
                _render_pipeline_summary_markdown(session_id, session_name, snapshot) + "\n",
                encoding="utf-8",
            )
            written.append(pipeline_summary_path)

        return written

    def _resolve_saved_file_for_kind(kind: str, session_id: str, path_text: str) -> Path:
        saved_root = agent.tools.get_saved_base_path(create=True)
        requested = Path(path_text).expanduser()
        safe_session = _normalize_session_id(session_id)
        categories = {"downloads", "media", "scripts", "showcase", "skills", "tmp", "tools"}

        if requested.is_absolute():
            candidate = requested.resolve()
        else:
            if requested.parts and requested.parts[0] in categories:
                requested_parts = [part for part in requested.parts if part not in ("", ".", "..")]
                if len(requested_parts) >= 2 and requested_parts[1] == safe_session:
                    scoped_rel = Path(*requested_parts)
                else:
                    scoped_rel = Path(requested_parts[0]) / safe_session
                    if len(requested_parts) > 1:
                        scoped_rel = scoped_rel.joinpath(*requested_parts[1:])
                candidate = (saved_root / scoped_rel).resolve()
            else:
                default_category = "scripts" if kind == "script" else "tools"
                candidate = (saved_root / default_category / safe_session / requested).resolve()

        try:
            relative_candidate = candidate.relative_to(saved_root)
        except ValueError as e:
            raise ValueError(f"Path must be inside saved root: {saved_root}") from e

        relative_parts = [part for part in relative_candidate.parts if part not in ("", ".", "..")]
        if relative_parts and relative_parts[0] in categories:
            if len(relative_parts) < 2 or relative_parts[1] != safe_session:
                expected_prefix = f"{relative_parts[0]}/{safe_session}"
                raise ValueError(f"{kind} path must be inside saved/{expected_prefix}/...")

        if not candidate.exists() or not candidate.is_file():
            raise ValueError(f"{kind} file not found: {candidate}")
        return candidate

    async def _run_script_or_tool_in_session(
        target_session_id: str,
        kind: str,
        path_text: str,
        trigger: str,
        cron_job_id: str | None = None,
    ) -> None:
        target_session = await agent.session_manager.load_session(target_session_id)
        if not target_session:
            raise ValueError(f"Session not found: {target_session_id}")

        file_path = _resolve_saved_file_for_kind(kind=kind, session_id=target_session_id, path_text=path_text)
        command = agent._build_script_runner_command(file_path)
        if not command:
            command = f"cd {shlex.quote(str(file_path.parent))} && ./{shlex.quote(file_path.name)}"

        async def _execute() -> None:
            previous_session = agent.session
            previous_session_id = previous_session.id if previous_session else None
            switched = previous_session_id != target_session.id
            if switched:
                agent.session = target_session
                agent.refresh_session_runtime_flags()

            try:
                ui.print_message(
                    "system",
                    f"[CRON] {trigger} {kind} run in session={target_session.id} path={file_path}",
                )
                if agent.session:
                    start_note = f"[CRON] {trigger} {kind} start: {file_path}"
                    agent.session.add_message(
                        role="tool",
                        content=start_note,
                        tool_name="cron",
                        tool_arguments={
                            "step": "run_script_tool_start",
                            "trigger": trigger,
                            "kind": kind,
                            "path": str(file_path),
                            "job_id": cron_job_id or "",
                        },
                        token_count=agent._count_tokens(start_note),
                    )
                    await agent.session_manager.save_session(agent.session)
                await _cron_chat_event(
                    cron_job_id,
                    "system",
                    f"[CRON] {trigger} {kind} run start: {file_path}",
                    trigger=trigger,
                    kind=kind,
                    session_id=target_session.id,
                )
                await _cron_monitor_event(
                    "run_script_tool_start",
                    history_job_id=cron_job_id,
                    trigger=trigger,
                    kind=kind,
                    path=str(file_path),
                    session_id=target_session.id,
                )
                result = await agent._execute_tool_with_guard(
                    name="shell",
                    arguments={"command": command},
                    interaction_label=f"cron_{kind}_{trigger}",
                )
                shell_output = result.content if result.success else f"Error: {result.error}"
                if agent.session:
                    agent.session.add_message(
                        role="tool",
                        content=shell_output,
                        tool_name="shell",
                        tool_arguments={"command": command, "cron": True, "job_id": cron_job_id or ""},
                        token_count=agent._count_tokens(shell_output),
                    )
                    await agent.session_manager.save_session(agent.session)
                ui.append_tool_output("shell", {"command": command, "cron": True}, shell_output)
                await _cron_monitor_event(
                    "run_script_tool_output",
                    history_job_id=cron_job_id,
                    trigger=trigger,
                    kind=kind,
                    path=str(file_path),
                    session_id=target_session.id,
                    output=_truncate_history_text(shell_output),
                )
                await _cron_chat_event(
                    cron_job_id,
                    "tool",
                    shell_output,
                    trigger=trigger,
                    kind=kind,
                    session_id=target_session.id,
                    path=str(file_path),
                )
                if not result.success:
                    raise RuntimeError(shell_output)
                await _cron_monitor_event(
                    "run_script_tool_done",
                    history_job_id=cron_job_id,
                    trigger=trigger,
                    kind=kind,
                    path=str(file_path),
                    session_id=target_session.id,
                )
                await _cron_chat_event(
                    cron_job_id,
                    "system",
                    f"[CRON] {trigger} {kind} run complete: {file_path}",
                    trigger=trigger,
                    kind=kind,
                    session_id=target_session.id,
                )
                if agent.session:
                    done_note = f"[CRON] {trigger} {kind} complete: {file_path}"
                    agent.session.add_message(
                        role="tool",
                        content=done_note,
                        tool_name="cron",
                        tool_arguments={
                            "step": "run_script_tool_done",
                            "trigger": trigger,
                            "kind": kind,
                            "path": str(file_path),
                            "job_id": cron_job_id or "",
                        },
                        token_count=agent._count_tokens(done_note),
                    )
                    await agent.session_manager.save_session(agent.session)
            except Exception as e:
                if agent.session:
                    failed_note = f"[CRON] {trigger} {kind} failed: {str(e)}"
                    agent.session.add_message(
                        role="tool",
                        content=failed_note,
                        tool_name="cron",
                        tool_arguments={
                            "step": "run_script_tool_failed",
                            "trigger": trigger,
                            "kind": kind,
                            "path": str(file_path),
                            "job_id": cron_job_id or "",
                        },
                        token_count=agent._count_tokens(failed_note),
                    )
                    await agent.session_manager.save_session(agent.session)
                raise
            finally:
                if switched and previous_session is not None:
                    restored = await agent.session_manager.load_session(previous_session_id)
                    agent.session = restored or previous_session
                    agent.refresh_session_runtime_flags()

        await _enqueue_agent_task(target_session.id, _execute, lane=CommandLane.CRON)

    def _parse_cron_add_args(raw_add: str) -> tuple[dict[str, object], str, dict[str, str]]:
        tokens = shlex.split(raw_add)
        schedule, consumed = parse_schedule_tokens(tokens)
        schedule["_text"] = schedule_to_text(schedule)
        remaining = tokens[consumed:]
        if not remaining:
            raise ValueError("Usage: /cron add every <Nm|Nh> <task|script|tool ...>")

        kind_head = remaining[0].strip().lower()
        if kind_head in {"script", "tool"}:
            if len(remaining) < 2:
                raise ValueError(f"Usage: /cron add ... {kind_head} <path>")
            path_text = " ".join(remaining[1:]).strip()
            if not path_text:
                raise ValueError(f"Usage: /cron add ... {kind_head} <path>")
            return schedule, kind_head, {"path": path_text}

        prompt_text = " ".join(remaining).strip()
        if not prompt_text:
            raise ValueError("Usage: /cron add ... \"<task>\"")
        return schedule, "prompt", {"text": prompt_text}

    async def _run_prompt_in_active_session(
        prompt_text: str,
        *,
        display_prompt: str | None = None,
        cron_job_id: str | None = None,
        cron_trigger: str | None = None,
        cron_source: str | None = None,
        raise_on_error: bool = False,
        lane: str = CommandLane.MAIN,
        queue: bool = True,
        on_assistant_text: Callable[[str], Awaitable[None]] | None = None,
        after_turn: Callable[[int, str, str], Awaitable[None]] | None = None,
    ) -> None:
        """Execute one user prompt using the currently selected session."""
        nonlocal last_exec_seconds
        nonlocal last_completed_at

        if not prompt_text.strip():
            return

        async def _execute() -> None:
            nonlocal last_exec_seconds
            nonlocal last_completed_at
            shown_prompt = display_prompt if display_prompt is not None else prompt_text
            ui.print_message("user", shown_prompt)
            ui.print_blank_line()
            turn_start_idx = len(agent.session.messages) if agent.session else 0
            if cron_job_id:
                await _cron_chat_event(
                    cron_job_id,
                    "user",
                    prompt_text,
                    trigger=cron_trigger or "",
                    source=cron_source or "",
                )

            started = time.perf_counter()
            assistant_text = ""
            try:
                ui.set_runtime_status("thinking")
                if get_config().ui.streaming:
                    ui.begin_assistant_stream()
                    ui.set_runtime_status("streaming")
                    chunks: list[str] = []

                    async def _consume_stream() -> None:
                        async for chunk in agent.stream(prompt_text):
                            chunks.append(chunk)
                            ui.print_streaming(chunk)
                        ui.complete_stream_line()

                    try:
                        _, cancelled = await _run_cancellable(ui, _consume_stream())
                    finally:
                        ui.end_assistant_stream()
                    if cancelled:
                        if cron_job_id:
                            await _cron_chat_event(
                                cron_job_id,
                                "system",
                                "[CRON] prompt cancelled",
                                trigger=cron_trigger or "",
                                source=cron_source or "",
                            )
                        ui.print_blank_line()
                        return
                    assistant_text = "".join(chunks)
                else:
                    response, cancelled = await _run_cancellable(ui, agent.complete(prompt_text))
                    if cancelled:
                        if cron_job_id:
                            await _cron_chat_event(
                                cron_job_id,
                                "system",
                                "[CRON] prompt cancelled",
                                trigger=cron_trigger or "",
                                source=cron_source or "",
                            )
                        ui.print_blank_line()
                        return
                    assistant_text = response or ""
                    ui.print_message("assistant", response)
                    ui.print_blank_line()

                if cron_job_id:
                    await _cron_chat_event(
                        cron_job_id,
                        "assistant",
                        assistant_text,
                        trigger=cron_trigger or "",
                        source=cron_source or "",
                    )
                    await _cron_monitor_event(
                        "prompt_assistant_output",
                        history_job_id=cron_job_id,
                        trigger=cron_trigger or "",
                        source=cron_source or "",
                        output=_truncate_history_text(assistant_text),
                    )
                if on_assistant_text:
                    outbound_text = assistant_text.strip()
                    if not outbound_text and agent.session:
                        for msg in reversed(agent.session.messages):
                            if str(msg.get("role", "")).strip().lower() != "assistant":
                                continue
                            candidate = str(msg.get("content", "")).strip()
                            if candidate:
                                outbound_text = candidate
                                break
                    if not outbound_text:
                        outbound_text = "Task completed. Check monitor output for details."
                    await on_assistant_text(outbound_text)
                if after_turn:
                    await after_turn(turn_start_idx, prompt_text, assistant_text)
                last_exec_seconds = time.perf_counter() - started
                last_completed_at = datetime.now()
                ui.set_runtime_status("waiting")
            except Exception as e:
                last_exec_seconds = time.perf_counter() - started
                last_completed_at = datetime.now()
                ui.set_runtime_status("waiting")
                if cron_job_id:
                    await _cron_chat_event(
                        cron_job_id,
                        "system",
                        f"[CRON] prompt failed: {str(e)}",
                        trigger=cron_trigger or "",
                        source=cron_source or "",
                    )
                ui.print_error(str(e))
                log.error("Error in agent", error=str(e))
                if on_assistant_text:
                    try:
                        await on_assistant_text(f"Error: {str(e)}")
                    except Exception:
                        pass
                if raise_on_error:
                    raise

        if not queue:
            await _execute()
            return

        session_id = agent.session.id if agent.session else "default"
        await _enqueue_agent_task(session_id, _execute, lane=lane)

    async def _run_prompt_in_session(
        session_id: str,
        prompt_text: str,
        source: str,
        *,
        cron_job_id: str | None = None,
        trigger: str = "scheduled",
    ) -> None:
        selected = await agent.session_manager.load_session(session_id)
        if not selected:
            raise ValueError(f"Session not found: {session_id}")

        async def _execute() -> None:
            previous_session = agent.session
            previous_session_id = previous_session.id if previous_session else None
            switched = previous_session_id != selected.id
            if switched:
                agent.session = selected
                agent.refresh_session_runtime_flags()
            try:
                await _cron_monitor_event(
                    "prompt_start",
                    history_job_id=cron_job_id,
                    source=source,
                    session_id=selected.id,
                )
                ui.print_message(
                    "system",
                    f"[CRON] {trigger} prompt run in session={selected.id} job={cron_job_id or 'oneoff'}",
                )
                if agent.session:
                    start_note = f"[CRON] {trigger} prompt start: {source}"
                    agent.session.add_message(
                        role="tool",
                        content=start_note,
                        tool_name="cron",
                        tool_arguments={
                            "step": "prompt_start",
                            "trigger": trigger,
                            "source": source,
                            "job_id": cron_job_id or "",
                        },
                        token_count=agent._count_tokens(start_note),
                    )
                    await agent.session_manager.save_session(agent.session)
                await _run_prompt_in_active_session(
                    prompt_text,
                    display_prompt=f"[CRON job={cron_job_id or 'oneoff'}] {prompt_text}",
                    cron_job_id=cron_job_id,
                    cron_trigger=trigger,
                    cron_source=source,
                    raise_on_error=bool(cron_job_id),
                    lane=CommandLane.CRON,
                    queue=False,
                )
                await _cron_monitor_event(
                    "prompt_done",
                    history_job_id=cron_job_id,
                    source=source,
                    session_id=selected.id,
                )
                if agent.session:
                    done_note = f"[CRON] {trigger} prompt done: {source}"
                    agent.session.add_message(
                        role="tool",
                        content=done_note,
                        tool_name="cron",
                        tool_arguments={
                            "step": "prompt_done",
                            "trigger": trigger,
                            "source": source,
                            "job_id": cron_job_id or "",
                        },
                        token_count=agent._count_tokens(done_note),
                    )
                    await agent.session_manager.save_session(agent.session)
            except Exception as e:
                if agent.session:
                    fail_note = f"[CRON] {trigger} prompt failed: {str(e)}"
                    agent.session.add_message(
                        role="tool",
                        content=fail_note,
                        tool_name="cron",
                        tool_arguments={
                            "step": "prompt_failed",
                            "trigger": trigger,
                            "source": source,
                            "job_id": cron_job_id or "",
                        },
                        token_count=agent._count_tokens(fail_note),
                    )
                    await agent.session_manager.save_session(agent.session)
                raise
            finally:
                if switched and previous_session is not None:
                    restored = await agent.session_manager.load_session(previous_session_id)
                    agent.session = restored or previous_session
                    agent.refresh_session_runtime_flags()

        await _enqueue_agent_task(selected.id, _execute, lane=CommandLane.CRON)

    async def _execute_cron_job(job: object, trigger: str = "scheduled") -> None:
        job_id = str(getattr(job, "id", "")).strip()
        if not job_id or job_id in cron_running_job_ids:
            return

        cron_running_job_ids.add(job_id)
        started_at_iso = to_utc_iso(now_utc())
        success = False
        queued_for_followup = False
        error_text = ""
        try:
            kind = str(getattr(job, "kind", "prompt")).strip().lower()
            payload = getattr(job, "payload", {})
            session_id = str(getattr(job, "session_id", "")).strip()
            schedule = getattr(job, "schedule", {})

            await _cron_monitor_event("job_start", history_job_id=job_id, trigger=trigger, job_id=job_id, kind=kind, session_id=session_id)
            await _cron_chat_event(
                job_id,
                "system",
                f"[CRON] job start trigger={trigger} kind={kind} session={session_id}",
                trigger=trigger,
                kind=kind,
                session_id=session_id,
            )
            if not session_id:
                raise ValueError(f"Cron job {job_id} has no session_id")

            if kind == "prompt":
                prompt_text = str(payload.get("text", "")).strip() if isinstance(payload, dict) else ""
                if not prompt_text:
                    raise ValueError(f"Cron job {job_id} has empty prompt payload")
                dispatch_status = await _dispatch_prompt_in_session(
                    session_id=session_id,
                    prompt_text=prompt_text,
                    source=f"cron:{trigger}:{job_id}",
                    cron_job_id=job_id,
                    trigger=trigger,
                )
                queued_for_followup = dispatch_status == "queued"
            elif kind in {"script", "tool"}:
                path_text = str(payload.get("path", "")).strip() if isinstance(payload, dict) else ""
                if not path_text:
                    raise ValueError(f"Cron job {job_id} has empty {kind} path payload")
                await _run_script_or_tool_in_session(
                    target_session_id=session_id,
                    kind=kind,
                    path_text=path_text,
                    trigger=trigger,
                    cron_job_id=job_id,
                )
            else:
                raise ValueError(f"Unsupported cron job kind: {kind}")

            success = True
            await _cron_monitor_event(
                "job_queued" if queued_for_followup else "job_done",
                history_job_id=job_id,
                trigger=trigger,
                job_id=job_id,
                kind=kind,
                session_id=session_id,
            )
            await _cron_chat_event(
                job_id,
                "system",
                (
                    f"[CRON] job queued trigger={trigger} kind={kind} session={session_id}"
                    if queued_for_followup
                    else f"[CRON] job done trigger={trigger} kind={kind} session={session_id}"
                ),
                trigger=trigger,
                kind=kind,
                session_id=session_id,
            )

            next_run_at_iso = to_utc_iso(compute_next_run(schedule))
            await agent.session_manager.update_cron_job(
                job_id,
                last_run_at=started_at_iso,
                next_run_at=next_run_at_iso,
                last_status="queued" if queued_for_followup else "ok",
                last_error="",
            )
        except Exception as e:
            error_text = str(e)
            await _cron_monitor_event("job_failed", history_job_id=job_id, trigger=trigger, job_id=job_id, error=error_text)
            await _cron_chat_event(
                job_id,
                "system",
                f"[CRON] job failed: {error_text}",
                trigger=trigger,
            )
            try:
                next_run_at_iso = to_utc_iso(compute_next_run(getattr(job, "schedule", {})))
            except Exception:
                next_run_at_iso = to_utc_iso(now_utc())
            await agent.session_manager.update_cron_job(
                job_id,
                last_run_at=started_at_iso,
                next_run_at=next_run_at_iso,
                last_status="failed",
                last_error=error_text,
            )
        finally:
            cron_running_job_ids.discard(job_id)
            if trigger == "manual" and not success:
                ui.print_error(error_text or f"Cron job failed: {job_id}")

    async def _cron_scheduler_loop() -> None:
        """Background pseudo-cron runner (Captain Claw cron, not system cron)."""
        while True:
            await asyncio.sleep(cron_poll_seconds)
            due_jobs = await agent.session_manager.get_due_cron_jobs(
                now_iso=to_utc_iso(now_utc()),
                limit=10,
            )
            for job in due_jobs:
                await _execute_cron_job(job, trigger="scheduled")

    def _format_recent_history(limit: int = 30) -> str:
        if not agent.session:
            return "No active session."
        messages = agent.session.messages[-max(1, int(limit)) :]
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

    def _format_active_configuration_text() -> str:
        cfg = get_config()
        active_model = agent.get_runtime_model_details()
        active_provider = str(active_model.get("provider", "")).strip() or cfg.model.provider
        active_model_name = str(active_model.get("model", "")).strip() or cfg.model.model
        active_model_id = str(active_model.get("id", "")).strip()
        active_model_source = str(active_model.get("source", "")).strip() or "default"
        model_id_part = f" [id={active_model_id}]" if active_model_id else ""
        workspace_path = str(cfg.resolved_workspace_path(Path.cwd()))
        return "\n".join(
            [
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
            ]
        )

    def _remote_help_text(platform_label: str) -> str:
        lines = [f"Captain Claw {platform_label} commands:", ""]
        for command, description in telegram_command_specs:
            lines.append(f"/{command} - {description}")
        lines.extend(
            [
                "",
                "Tip: send normal text (without `/`) to chat with the active session.",
            ]
        )
        return "\n".join(lines)

    def _telegram_help_text() -> str:
        return _remote_help_text("Telegram")

    async def _register_telegram_commands() -> None:
        if not telegram_bridge:
            return
        try:
            await telegram_bridge.set_my_commands(telegram_command_specs)
            await telegram_bridge.set_chat_menu_button_commands()
            ui.append_system_line(
                f"Telegram commands registered ({len(telegram_command_specs)} entries)."
            )
        except Exception as e:
            ui.append_system_line(f"Telegram command registration failed: {str(e)}")

    async def _handle_remote_command(
        *,
        platform: str,
        raw_text: str,
        help_label: str,
        sender_label: str,
        send_text: Callable[[str], Awaitable[None]],
        execute_prompt: Callable[[str, str], Awaitable[None]],
    ) -> bool:
        lowered_text = raw_text.strip().lower()
        if lowered_text == "/start" or lowered_text.startswith("/start "):
            await send_text(
                (
                    "Captain Claw connected.\n"
                    "Use /help to see available commands.\n"
                    "Send plain text to chat with the current session."
                ),
            )
            return True
        if lowered_text == "/help" or lowered_text.startswith("/help "):
            await send_text(_remote_help_text(help_label))
            return True

        result = ui.handle_special_command(raw_text)
        if result is None:
            # Includes /help output and parser errors already shown in console.
            await send_text("Command processed.")
            return True
        if result == "EXIT":
            await send_text("`/exit` is only available in local console.")
            return True
        if result.startswith("APPROVE_CHAT_USER:") or result.startswith("APPROVE_TELEGRAM_USER:"):
            await send_text("This command is operator-only in local console.")
            return True
        if result == "PIPELINE_INFO":
            await send_text(
                (
                    "Pipeline mode: "
                    f"{agent.pipeline_mode} "
                    "(loop=fast/simple, contracts=planner+completion gate)"
                ),
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
                skill_name=skill_name,
                install_id=install_id or None,
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
                    "Usage: /skill <name> [args] | /skill search <criteria> | /skill install <github-url> | /skill install <skill-name> [install-id]"
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
                    (
                        f"Session: {agent.session.name}\n"
                        f"ID: {agent.session.id}\n"
                        f"Messages: {len(agent.session.messages)}\n"
                        f"Model: {details.get('provider')}/{details.get('model')}"
                    ),
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
                    (
                        "Session compacted "
                        f"({int(stats.get('before_tokens', 0))} -> {int(stats.get('after_tokens', 0))} tokens)"
                    ),
                )
            else:
                await send_text(f"Compaction skipped: {str(stats.get('reason', 'not_needed'))}")
            return True
        if result == "CONFIG":
            await send_text(_format_active_configuration_text())
            return True
        if result == "HISTORY":
            await send_text(_format_recent_history(limit=30))
            return True
        if result == "SESSION_MODEL_INFO":
            details = agent.get_runtime_model_details()
            await send_text(
                (
                    "Active model: "
                    f"{details.get('provider')}/{details.get('model')} "
                    f"(source={details.get('source') or 'unknown'}, id={details.get('id') or '-'})"
                ),
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
                await send_text("Usage: /cron \"<task>\"")
                return True
            await execute_prompt(prompt, f"[{sender_label} cron oneoff] {prompt}")
            return True
        _ = platform
        await send_text("Command requires local console in this version.")
        return True

    async def _handle_telegram_command(message: TelegramMessage) -> bool:
        chat_id = message.chat_id

        async def _execute_prompt(prompt: str, display_prompt: str) -> None:
            await _run_prompt_in_active_session(
                prompt,
                display_prompt=display_prompt,
                on_assistant_text=lambda text: _telegram_send(chat_id, text),
                after_turn=lambda turn_start_idx, user_prompt, assistant_text: _telegram_maybe_send_audio_for_turn(
                    chat_id=chat_id,
                    reply_to_message_id=message.message_id,
                    user_prompt=user_prompt,
                    assistant_text=assistant_text,
                    turn_start_idx=turn_start_idx,
                ),
            )

        return await _handle_remote_command(
            platform="telegram",
            raw_text=message.text,
            help_label="Telegram",
            sender_label="TG",
            send_text=lambda text: _telegram_send(chat_id, text),
            execute_prompt=_execute_prompt,
        )

    async def _handle_slack_command(message: SlackMessage) -> bool:
        channel_id = message.channel_id

        async def _execute_prompt(prompt: str, display_prompt: str) -> None:
            await _run_prompt_in_active_session(
                prompt,
                display_prompt=display_prompt,
                on_assistant_text=lambda text: _slack_send(channel_id, text, reply_to_message_ts=message.message_ts),
                after_turn=lambda turn_start_idx, user_prompt, assistant_text: _slack_maybe_send_audio_for_turn(
                    channel_id=channel_id,
                    user_prompt=user_prompt,
                    assistant_text=assistant_text,
                    turn_start_idx=turn_start_idx,
                ),
            )

        return await _handle_remote_command(
            platform="slack",
            raw_text=message.text,
            help_label="Slack",
            sender_label="SLACK",
            send_text=lambda text: _slack_send(channel_id, text, reply_to_message_ts=message.message_ts),
            execute_prompt=_execute_prompt,
        )

    async def _handle_discord_command(message: DiscordMessage) -> bool:
        channel_id = message.channel_id

        async def _execute_prompt(prompt: str, display_prompt: str) -> None:
            await _run_prompt_in_active_session(
                prompt,
                display_prompt=display_prompt,
                on_assistant_text=lambda text: _discord_send(
                    channel_id,
                    text,
                    reply_to_message_id=message.id,
                ),
                after_turn=lambda turn_start_idx, user_prompt, assistant_text: _discord_maybe_send_audio_for_turn(
                    channel_id=channel_id,
                    reply_to_message_id=message.id,
                    user_prompt=user_prompt,
                    assistant_text=assistant_text,
                    turn_start_idx=turn_start_idx,
                ),
            )

        return await _handle_remote_command(
            platform="discord",
            raw_text=message.text,
            help_label="Discord",
            sender_label="DISCORD",
            send_text=lambda text: _discord_send(channel_id, text, reply_to_message_id=message.id),
            execute_prompt=_execute_prompt,
        )

    async def _handle_telegram_message(message: TelegramMessage) -> None:
        try:
            await _telegram_mark_read(message)
            await _telegram_monitor_event(
                "incoming_message",
                chat_id=message.chat_id,
                user_id=message.user_id,
                username=message.username or "",
                message_id=message.message_id,
                is_command=bool(message.text.strip().startswith("/")),
                text_preview=_truncate_telegram_text(message.text),
            )
            user_id_key = str(message.user_id)
            if user_id_key not in approved_telegram_users:
                await _pair_unknown_telegram_user(message)
                return
            text = message.text.strip()
            if not text:
                return
            if text.startswith("/"):
                await _run_with_telegram_typing(
                    message.chat_id,
                    _handle_telegram_command(message),
                )
                return
            user_label = message.username or message.first_name or str(message.user_id)
            await _run_with_telegram_typing(
                message.chat_id,
                _run_prompt_in_active_session(
                    text,
                    display_prompt=f"[TG {user_label}] {text}",
                    on_assistant_text=lambda out: _telegram_send(
                        message.chat_id,
                        out,
                        reply_to_message_id=message.message_id,
                    ),
                    after_turn=lambda turn_start_idx, user_prompt, assistant_text: _telegram_maybe_send_audio_for_turn(
                        chat_id=message.chat_id,
                        reply_to_message_id=message.message_id,
                        user_prompt=user_prompt,
                        assistant_text=assistant_text,
                        turn_start_idx=turn_start_idx,
                    ),
                ),
            )
        except Exception as e:
            log.error("Telegram message handler failed", error=str(e))
            try:
                await _telegram_monitor_event(
                    "handler_error",
                    chat_id=message.chat_id,
                    user_id=message.user_id,
                    message_id=message.message_id,
                    error=str(e),
                )
            except Exception:
                pass
            try:
                await _telegram_send(
                    message.chat_id,
                    f"Error while processing your request: {str(e)}",
                    reply_to_message_id=message.message_id,
                )
            except Exception:
                pass

    async def _handle_slack_message(message: SlackMessage) -> None:
        try:
            await _slack_mark_read(message)
            await _slack_monitor_event(
                "incoming_message",
                channel_id=message.channel_id,
                user_id=message.user_id,
                username=message.username or "",
                message_ts=message.message_ts,
                is_command=bool(message.text.strip().startswith("/")),
                text_preview=_truncate_chat_text(message.text),
            )
            user_id_key = str(message.user_id).strip()
            if user_id_key not in approved_slack_users:
                await _pair_unknown_slack_user(message)
                return
            text = message.text.strip()
            if not text:
                return
            if text.startswith("/"):
                await _run_with_slack_typing(
                    message.channel_id,
                    _handle_slack_command(message),
                )
                return
            user_label = message.username or str(message.user_id)
            await _run_with_slack_typing(
                message.channel_id,
                _run_prompt_in_active_session(
                    text,
                    display_prompt=f"[SLACK {user_label}] {text}",
                    on_assistant_text=lambda out: _slack_send(
                        message.channel_id,
                        out,
                        reply_to_message_ts=message.message_ts,
                    ),
                    after_turn=lambda turn_start_idx, user_prompt, assistant_text: _slack_maybe_send_audio_for_turn(
                        channel_id=message.channel_id,
                        user_prompt=user_prompt,
                        assistant_text=assistant_text,
                        turn_start_idx=turn_start_idx,
                    ),
                ),
            )
        except Exception as e:
            log.error("Slack message handler failed", error=str(e))
            try:
                await _slack_monitor_event(
                    "handler_error",
                    channel_id=message.channel_id,
                    user_id=message.user_id,
                    message_ts=message.message_ts,
                    error=str(e),
                )
            except Exception:
                pass
            try:
                await _slack_send(
                    message.channel_id,
                    f"Error while processing your request: {str(e)}",
                    reply_to_message_ts=message.message_ts,
                )
            except Exception:
                pass

    async def _handle_discord_message(message: DiscordMessage) -> None:
        try:
            is_guild_message = bool(str(message.guild_id or "").strip())
            requires_mention = bool(getattr(discord_cfg, "require_mention_in_guild", True))
            if is_guild_message and requires_mention and not bool(message.mentioned_bot):
                return
            await _discord_mark_read(message)
            await _discord_monitor_event(
                "incoming_message",
                channel_id=message.channel_id,
                guild_id=message.guild_id or "",
                user_id=message.user_id,
                username=message.username or "",
                message_id=message.id,
                mentioned_bot=bool(message.mentioned_bot),
                is_command=bool(message.text.strip().startswith("/")),
                text_preview=_truncate_chat_text(message.text),
            )
            user_id_key = str(message.user_id).strip()
            if user_id_key not in approved_discord_users:
                await _pair_unknown_discord_user(message)
                return
            text = message.text.strip()
            if not text:
                return
            if text.startswith("/"):
                await _run_with_discord_typing(
                    message.channel_id,
                    _handle_discord_command(message),
                )
                return
            user_label = message.username or str(message.user_id)
            await _run_with_discord_typing(
                message.channel_id,
                _run_prompt_in_active_session(
                    text,
                    display_prompt=f"[DISCORD {user_label}] {text}",
                    on_assistant_text=lambda out: _discord_send(
                        message.channel_id,
                        out,
                        reply_to_message_id=message.id,
                    ),
                    after_turn=lambda turn_start_idx, user_prompt, assistant_text: _discord_maybe_send_audio_for_turn(
                        channel_id=message.channel_id,
                        reply_to_message_id=message.id,
                        user_prompt=user_prompt,
                        assistant_text=assistant_text,
                        turn_start_idx=turn_start_idx,
                    ),
                ),
            )
        except Exception as e:
            log.error("Discord message handler failed", error=str(e))
            try:
                await _discord_monitor_event(
                    "handler_error",
                    channel_id=message.channel_id,
                    user_id=message.user_id,
                    message_id=message.id,
                    error=str(e),
                )
            except Exception:
                pass
            try:
                await _discord_send(
                    message.channel_id,
                    f"Error while processing your request: {str(e)}",
                    reply_to_message_id=message.id,
                )
            except Exception:
                pass

    async def _telegram_poll_loop() -> None:
        """Background Telegram polling and message dispatch."""
        nonlocal telegram_offset
        assert telegram_bridge is not None
        while True:
            try:
                updates = await telegram_bridge.get_updates(
                    offset=telegram_offset,
                    timeout=max(1, int(telegram_cfg.poll_timeout_seconds)),
                )
                for update in updates:
                    next_offset = int(update.update_id) + 1
                    telegram_offset = next_offset if telegram_offset is None else max(telegram_offset, next_offset)
                    asyncio.create_task(_handle_telegram_message(update))
            except asyncio.CancelledError:
                break
            except Exception as e:
                ui.append_system_line(f"Telegram poll error: {str(e)}")
                await asyncio.sleep(2.0)

    async def _slack_poll_loop() -> None:
        """Background Slack polling and message dispatch."""
        nonlocal slack_offsets
        assert slack_bridge is not None
        while True:
            try:
                updates, next_offsets = await slack_bridge.get_updates(slack_offsets)
                slack_offsets = dict(next_offsets)
                for update in updates:
                    asyncio.create_task(_handle_slack_message(update))
                await asyncio.sleep(max(1, int(slack_cfg.poll_timeout_seconds)))
            except asyncio.CancelledError:
                break
            except Exception as e:
                ui.append_system_line(f"Slack poll error: {str(e)}")
                await asyncio.sleep(2.0)

    async def _discord_poll_loop() -> None:
        """Background Discord polling and message dispatch."""
        nonlocal discord_offsets
        assert discord_bridge is not None
        while True:
            try:
                updates, next_offsets = await discord_bridge.get_updates(discord_offsets)
                discord_offsets = dict(next_offsets)
                for update in updates:
                    asyncio.create_task(_handle_discord_message(update))
                await asyncio.sleep(max(1, int(discord_cfg.poll_timeout_seconds)))
            except asyncio.CancelledError:
                break
            except Exception as e:
                ui.append_system_line(f"Discord poll error: {str(e)}")
                await asyncio.sleep(2.0)

    if telegram_enabled:
        token = telegram_cfg.bot_token.strip()
        if token:
            telegram_bridge = TelegramBridge(token=token, api_base_url=telegram_cfg.api_base_url)
            approved_telegram_users = await _load_json_state(telegram_state_key_approved)
            pending_telegram_pairings = await _load_json_state(telegram_state_key_pending)
            _cleanup_expired_pairings(pending_telegram_pairings)
            await _save_telegram_state()
            await _register_telegram_commands()
            telegram_poll_task = asyncio.create_task(_telegram_poll_loop())
            ui.append_system_line("Telegram UI enabled (long polling started).")
        else:
            ui.append_system_line("Telegram enabled but bot_token is empty; skipping Telegram startup.")

    if slack_enabled:
        token = slack_cfg.bot_token.strip()
        if token:
            slack_bridge = SlackBridge(token=token, api_base_url=slack_cfg.api_base_url)
            approved_slack_users = await _load_json_state(slack_state_key_approved)
            pending_slack_pairings = await _load_json_state(slack_state_key_pending)
            _cleanup_expired_pairings(pending_slack_pairings)
            await _save_slack_state()
            slack_poll_task = asyncio.create_task(_slack_poll_loop())
            ui.append_system_line("Slack UI enabled (polling started).")
        else:
            ui.append_system_line("Slack enabled but bot_token is empty; skipping Slack startup.")

    if discord_enabled:
        token = discord_cfg.bot_token.strip()
        if token:
            discord_bridge = DiscordBridge(token=token, api_base_url=discord_cfg.api_base_url)
            approved_discord_users = await _load_json_state(discord_state_key_approved)
            pending_discord_pairings = await _load_json_state(discord_state_key_pending)
            _cleanup_expired_pairings(pending_discord_pairings)
            await _save_discord_state()
            discord_poll_task = asyncio.create_task(_discord_poll_loop())
            ui.append_system_line("Discord UI enabled (polling started).")
        else:
            ui.append_system_line("Discord enabled but bot_token is empty; skipping Discord startup.")

    cron_worker = asyncio.create_task(_cron_scheduler_loop())
    try:
        # Main loop
        while True:
            try:
                ui.print_status_line(
                    last_usage=agent.last_usage,
                    total_usage=agent.total_usage,
                    last_exec_seconds=last_exec_seconds,
                    last_completed_at=last_completed_at,
                    session_id=agent.session.id if agent.session else None,
                    context_window=agent.last_context_window,
                    model_details=agent.get_runtime_model_details(),
                )
                ui.set_runtime_status("user input")
                # Get user input (threaded so event loop keeps servicing Captain Claw cron).
                user_input = await asyncio.to_thread(ui.prompt)

                # Handle special commands
                result = ui.handle_special_command(user_input)

                if result is None:
                    continue
                elif result == "EXIT":
                    log.info("User requested exit")
                    break
                elif result.startswith("APPROVE_CHAT_USER:"):
                    parts = result.split(":", 2)
                    platform = parts[1].strip().lower() if len(parts) > 1 else ""
                    token = parts[2].strip() if len(parts) > 2 else ""
                    enabled_map = {
                        "telegram": telegram_enabled,
                        "slack": slack_enabled,
                        "discord": discord_enabled,
                    }
                    if not enabled_map.get(platform, False):
                        ui.print_error(f"{platform.title() if platform else 'Target'} integration is not enabled.")
                        continue
                    ok, message = await _approve_chat_pairing_token(platform, token)
                    if ok:
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result.startswith("APPROVE_TELEGRAM_USER:"):
                    token = result.split(":", 1)[1].strip()
                    ok, message = await _approve_chat_pairing_token("telegram", token)
                    if ok:
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result == "CLEAR":
                    if agent.session:
                        if agent.is_session_memory_protected():
                            ui.print_error(
                                "Session memory is protected. Disable it with '/session protect off' first."
                            )
                            continue
                        agent.session.messages = []
                        await agent.session_manager.save_session(agent.session)
                        ui.clear_monitor_tool_output()
                        ui.print_success("Session cleared")
                    continue
                elif result == "NEW" or result.startswith("NEW:"):
                    session_name = "default"
                    if result.startswith("NEW:"):
                        session_name = result.split(":", 1)[1].strip() or "default"
                    agent.session = await agent.session_manager.create_session(name=session_name)
                    agent.refresh_session_runtime_flags()
                    if agent.session:
                        await agent.session_manager.set_last_active_session(agent.session.id)
                    if agent.session:
                        ui.load_monitor_tool_output_from_session(agent.session.messages)
                        ui.print_session_info(agent.session)
                    ui.print_success("Started new session")
                    continue
                elif result == "SESSIONS":
                    sessions = await agent.session_manager.list_sessions(limit=20)
                    ui.print_session_list(
                        sessions,
                        current_session_id=agent.session.id if agent.session else None,
                    )
                    continue
                elif result == "MODELS":
                    ui.print_model_list(
                        agent.get_allowed_models(),
                        active_model=agent.get_runtime_model_details(),
                    )
                    continue
                elif result == "SESSION_INFO":
                    if agent.session:
                        ui.print_session_info(agent.session)
                    else:
                        ui.print_error("No active session")
                    continue
                elif result == "SESSION_MODEL_INFO":
                    details = agent.get_runtime_model_details()
                    ui.print_success(
                        "Active model: "
                        f"{details.get('provider')}/{details.get('model')} "
                        f"(source={details.get('source') or 'unknown'}, id={details.get('id') or '-'})"
                    )
                    continue
                elif result in {"SESSION_PROTECT_ON", "SESSION_PROTECT_OFF"}:
                    enabled = result.endswith("_ON")
                    ok, message = await agent.set_session_memory_protection(enabled, persist=True)
                    if ok:
                        if agent.session:
                            ui.print_session_info(agent.session)
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result.startswith("SESSION_MODEL_SET:"):
                    selector = result.split(":", 1)[1].strip()
                    ok, message = await agent.set_session_model(selector, persist=True)
                    if ok:
                        if agent.session:
                            ui.print_session_info(agent.session)
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result == "SESSION_QUEUE_INFO":
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    settings = await _resolve_queue_settings_for_session(agent.session.id)
                    ui.print_success(
                        "Session queue settings: "
                        f"mode={settings.mode} debounce_ms={settings.debounce_ms} "
                        f"cap={settings.cap} drop={settings.drop_policy} "
                        f"pending={followup_queue.get_queue_depth(agent.session.id)}"
                    )
                    continue
                elif result.startswith("SESSION_QUEUE_MODE:"):
                    mode_value = result.split(":", 1)[1].strip()
                    ok, message = await _update_active_session_queue_settings(mode=mode_value)
                    if ok:
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result.startswith("SESSION_QUEUE_DEBOUNCE:"):
                    raw_value = result.split(":", 1)[1].strip()
                    try:
                        parsed = int(raw_value)
                    except Exception:
                        ui.print_error("Usage: /session queue debounce <ms>")
                        continue
                    ok, message = await _update_active_session_queue_settings(debounce_ms=parsed)
                    if ok:
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result.startswith("SESSION_QUEUE_CAP:"):
                    raw_value = result.split(":", 1)[1].strip()
                    try:
                        parsed = int(raw_value)
                    except Exception:
                        ui.print_error("Usage: /session queue cap <n>")
                        continue
                    ok, message = await _update_active_session_queue_settings(cap=parsed)
                    if ok:
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result.startswith("SESSION_QUEUE_DROP:"):
                    drop_value = result.split(":", 1)[1].strip()
                    ok, message = await _update_active_session_queue_settings(drop_policy=drop_value)
                    if ok:
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result == "SESSION_QUEUE_CLEAR":
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    session_id = agent.session.id
                    followup_cleared = followup_queue.clear_queue(session_id)
                    lane_cleared = command_queue.clear_lane(resolve_session_lane(session_id))
                    ui.print_success(
                        f"Cleared session queue: followup={followup_cleared} lane={lane_cleared}"
                    )
                    continue
                elif result.startswith("SESSION_SELECT:"):
                    selector = result.split(":", 1)[1].strip()
                    selected = await agent.session_manager.select_session(selector)
                    if not selected:
                        ui.print_error(f"Session not found: {selector}")
                        continue
                    agent.session = selected
                    agent.refresh_session_runtime_flags()
                    await agent.session_manager.set_last_active_session(agent.session.id)
                    ui.load_monitor_tool_output_from_session(agent.session.messages)
                    ui.print_session_info(agent.session)
                    ui.print_success("Loaded session")
                    continue
                elif result.startswith("SESSION_RENAME:"):
                    new_name = result.split(":", 1)[1].strip()
                    if not new_name:
                        ui.print_error("Usage: /session rename <new-name>")
                        continue
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    old_name = agent.session.name
                    agent.session.name = new_name
                    await agent.session_manager.save_session(agent.session)
                    ui.print_session_info(agent.session)
                    ui.print_success(f'Session renamed: "{old_name}" -> "{new_name}"')
                    continue
                elif result == "SESSION_DESCRIPTION_INFO":
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    description = str(agent.session.metadata.get("description", "")).strip()
                    if description:
                        ui.print_success(f"Session description: {description}")
                    else:
                        ui.print_warning("Session has no description yet")
                    continue
                elif result == "SESSION_DESCRIPTION_AUTO":
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    generated = await agent.generate_session_description(agent.session, max_sentences=5)
                    description = agent.sanitize_session_description(generated, max_sentences=5)
                    if not description:
                        ui.print_error("Could not generate a session description")
                        continue
                    agent.session.metadata["description"] = description
                    agent.session.metadata["description_source"] = "auto"
                    agent.session.metadata["description_updated_at"] = datetime.now().isoformat()
                    await agent.session_manager.save_session(agent.session)
                    ui.print_session_info(agent.session)
                    ui.print_success("Session description auto-generated")
                    continue
                elif result.startswith("SESSION_DESCRIPTION_SET:"):
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /session description payload")
                        continue
                    raw_description = str(payload.get("description", "")).strip()
                    description = agent.sanitize_session_description(raw_description, max_sentences=5)
                    if not description:
                        ui.print_error("Usage: /session description <text> | /session description auto")
                        continue
                    agent.session.metadata["description"] = description
                    agent.session.metadata["description_source"] = "manual"
                    agent.session.metadata["description_updated_at"] = datetime.now().isoformat()
                    await agent.session_manager.save_session(agent.session)
                    ui.print_session_info(agent.session)
                    ui.print_success("Session description updated")
                    continue
                elif result.startswith("SESSION_EXPORT:"):
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    mode = result.split(":", 1)[1].strip().lower() or "all"
                    session_id = agent.session.id

                    async def _export_task() -> list[Path]:
                        return _export_active_session_history(mode)

                    written_paths = await _enqueue_agent_task(
                        session_id,
                        _export_task,
                        lane=CommandLane.NESTED,
                    )
                    if not written_paths:
                        ui.print_error("Failed to export session history")
                        continue
                    ui.append_tool_output(
                        "session_export",
                        {
                            "session_id": agent.session.id,
                            "mode": mode,
                            "count": len(written_paths),
                        },
                        "\n".join(f"path={path}" for path in written_paths),
                    )
                    for path in written_paths:
                        ui.print_success(f"Exported: {path}")
                    continue
                elif result.startswith("SESSION_RUN:"):
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /session run payload")
                        continue

                    selector = str(payload.get("selector", "")).strip()
                    prompt = str(payload.get("prompt", "")).strip()
                    if not selector or not prompt:
                        ui.print_error("Usage: /session run <id|name|#index> <prompt>")
                        continue

                    selected = await agent.session_manager.select_session(selector)
                    if not selected:
                        ui.print_error(f"Session not found: {selector}")
                        continue

                    async def _run_selected_session_prompt() -> None:
                        previous_session = agent.session
                        previous_session_id = previous_session.id if previous_session else None
                        switched_temporarily = previous_session_id != selected.id

                        if switched_temporarily:
                            agent.session = selected
                            agent.refresh_session_runtime_flags()
                            ui.load_monitor_tool_output_from_session(agent.session.messages)
                            ui.print_success(f'Running in session "{agent.session.name}"')

                        try:
                            await _run_prompt_in_active_session(
                                prompt,
                                lane=CommandLane.NESTED,
                                queue=False,
                            )
                        finally:
                            if switched_temporarily and previous_session is not None:
                                restored = await agent.session_manager.load_session(previous_session_id)
                                agent.session = restored or previous_session
                                agent.refresh_session_runtime_flags()
                                ui.load_monitor_tool_output_from_session(agent.session.messages)
                                ui.print_success(f'Restored session "{agent.session.name}"')

                    await _enqueue_agent_task(
                        selected.id,
                        _run_selected_session_prompt,
                        lane=CommandLane.NESTED,
                    )
                    continue
                elif result.startswith("SESSION_PROCREATE:"):
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /session procreate payload")
                        continue

                    parent_one_selector = str(payload.get("parent_one", "")).strip()
                    parent_two_selector = str(payload.get("parent_two", "")).strip()
                    new_name = str(payload.get("new_name", "")).strip()
                    if not parent_one_selector or not parent_two_selector or not new_name:
                        ui.print_error(
                            "Usage: /session procreate <id|name|#index> <id|name|#index> <new-name>"
                        )
                        continue

                    ui.append_tool_output(
                        "session_procreate",
                        {"step": "resolve_parents", "parent_one_selector": parent_one_selector, "parent_two_selector": parent_two_selector},
                        "step=resolve_parents\nstatus=locating_parent_sessions",
                    )
                    parent_one = await agent.session_manager.select_session(parent_one_selector)
                    if not parent_one:
                        ui.print_error(f"Session not found: {parent_one_selector}")
                        continue
                    parent_two = await agent.session_manager.select_session(parent_two_selector)
                    if not parent_two:
                        ui.print_error(f"Session not found: {parent_two_selector}")
                        continue

                    if parent_one.id == parent_two.id:
                        ui.print_error("Choose two different sessions for /session procreate")
                        continue

                    try:
                        child_session, stats = await agent.procreate_sessions(
                            parent_one=parent_one,
                            parent_two=parent_two,
                            new_name=new_name,
                            persist=True,
                        )
                    except ValueError as e:
                        ui.print_error(str(e))
                        continue

                    ui.append_tool_output(
                        "session_procreate",
                        {"step": "switch_to_child", "session_id": child_session.id},
                        (
                            "step=switch_to_child\n"
                            f'session_id="{child_session.id}"\n'
                            f'session_name="{child_session.name}"'
                        ),
                    )
                    agent.session = child_session
                    agent.refresh_session_runtime_flags()
                    await agent.session_manager.set_last_active_session(agent.session.id)
                    ui.load_monitor_tool_output_from_session(agent.session.messages)
                    ui.append_tool_output(
                        "session_procreate",
                        {"step": "complete", "session_id": child_session.id},
                        (
                            "step=complete\n"
                            f'session_id="{child_session.id}"\n'
                            f"merged_messages={stats.get('merged_messages', 0)}"
                        ),
                    )
                    ui.print_session_info(agent.session)
                    ui.print_success(
                        f'Procreated session "{child_session.name}" '
                        f"(merged_messages={stats.get('merged_messages', 0)}, "
                        f"compacted={stats.get('parent_one_compacted', 0)}+{stats.get('parent_two_compacted', 0)})"
                    )
                    continue
                elif result == "CRON_LIST":
                    jobs = await agent.session_manager.list_cron_jobs(limit=200, active_only=True)
                    for job in jobs:
                        if isinstance(job.schedule, dict):
                            job.schedule["_text"] = schedule_to_text(job.schedule)
                    ui.print_cron_jobs(jobs)
                    continue
                elif result.startswith("CRON_HISTORY:"):
                    selector = result.split(":", 1)[1].strip()
                    job = await agent.session_manager.select_cron_job(selector, active_only=False)
                    if not job:
                        ui.print_error(f"Cron job not found: {selector}")
                        continue
                    ui.print_cron_job_history(job)
                    continue
                elif result.startswith("CRON_ONEOFF:"):
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /cron payload")
                        continue
                    prompt = str(payload.get("prompt", "")).strip()
                    if not prompt:
                        ui.print_error("Usage: /cron \"<task>\"")
                        continue
                    if not agent.session:
                        ui.print_error("No active session for /cron")
                        continue
                    _cron_monitor("oneoff_prompt", session_id=agent.session.id if agent.session else "", chars=len(prompt))
                    status = await _dispatch_prompt_in_session(
                        session_id=agent.session.id,
                        prompt_text=prompt,
                        source="cron:oneoff",
                        cron_job_id=None,
                        trigger="oneoff",
                    )
                    if status == "queued":
                        ui.print_success("Cron one-off queued as follow-up (session busy)")
                    continue
                elif result.startswith("CRON_ADD:"):
                    if not agent.session:
                        ui.print_error("No active session for /cron add")
                        continue
                    raw_add = result.split(":", 1)[1].strip()
                    try:
                        schedule, kind, payload = _parse_cron_add_args(raw_add)
                    except ValueError as e:
                        ui.print_error(str(e))
                        continue

                    if kind in {"script", "tool"}:
                        try:
                            _ = _resolve_saved_file_for_kind(
                                kind=kind,
                                session_id=agent.session.id,
                                path_text=str(payload.get("path", "")),
                            )
                        except ValueError as e:
                            ui.print_error(str(e))
                            continue

                    next_run_at_iso = to_utc_iso(compute_next_run(schedule))
                    job = await agent.session_manager.create_cron_job(
                        kind=kind,
                        payload=payload,
                        schedule=schedule,
                        session_id=agent.session.id,
                        next_run_at=next_run_at_iso,
                        enabled=True,
                    )
                    await _cron_monitor_event(
                        "job_added",
                        history_job_id=job.id,
                        job_id=job.id,
                        session_id=job.session_id,
                        kind=job.kind,
                        schedule=schedule_to_text(schedule),
                        next_run_at=next_run_at_iso,
                    )
                    ui.print_success(
                        f"Cron job added: id={job.id} kind={job.kind} "
                        f"schedule={schedule_to_text(schedule)} next={next_run_at_iso}"
                    )
                    continue
                elif result.startswith("CRON_REMOVE:"):
                    selector = result.split(":", 1)[1].strip()
                    job = await agent.session_manager.select_cron_job(selector, active_only=False)
                    if not job:
                        ui.print_error(f"Cron job not found: {selector}")
                        continue
                    job_id = job.id
                    deleted = await agent.session_manager.delete_cron_job(job_id)
                    if not deleted:
                        ui.print_error(f"Cron job not found: {job_id}")
                        continue
                    _cron_monitor("job_removed", job_id=job_id)
                    ui.print_success(f"Removed cron job: {job_id}")
                    continue
                elif result.startswith("CRON_PAUSE:"):
                    selector = result.split(":", 1)[1].strip()
                    job = await agent.session_manager.select_cron_job(selector, active_only=False)
                    if not job:
                        ui.print_error(f"Cron job not found: {selector}")
                        continue
                    job_id = job.id
                    updated = await agent.session_manager.update_cron_job(
                        job_id,
                        enabled=False,
                        last_status="paused",
                    )
                    if not updated:
                        ui.print_error(f"Cron job not found: {job_id}")
                        continue
                    await _cron_monitor_event("job_paused", history_job_id=job_id, job_id=job_id)
                    ui.print_success(f"Paused cron job: {job_id}")
                    continue
                elif result.startswith("CRON_RESUME:"):
                    selector = result.split(":", 1)[1].strip()
                    job = await agent.session_manager.select_cron_job(selector, active_only=False)
                    if not job:
                        ui.print_error(f"Cron job not found: {selector}")
                        continue
                    job_id = job.id
                    updated = await agent.session_manager.update_cron_job(
                        job_id,
                        enabled=True,
                        last_status="scheduled",
                    )
                    if not updated:
                        ui.print_error(f"Cron job not found: {job_id}")
                        continue
                    await _cron_monitor_event("job_resumed", history_job_id=job_id, job_id=job_id)
                    ui.print_success(f"Resumed cron job: {job_id}")
                    continue
                elif result.startswith("CRON_RUN:"):
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /cron run payload")
                        continue
                    raw_args = str(payload.get("args", "")).strip()
                    if not raw_args:
                        ui.print_error("Usage: /cron run <job-id|#index> | /cron run script <path> | /cron run tool <path>")
                        continue
                    try:
                        run_tokens = shlex.split(raw_args)
                    except ValueError:
                        ui.print_error("Invalid /cron run arguments")
                        continue
                    if not run_tokens:
                        ui.print_error("Usage: /cron run <job-id|#index> | /cron run script <path> | /cron run tool <path>")
                        continue

                    head = run_tokens[0].strip().lower()
                    if head in {"script", "tool"}:
                        if not agent.session:
                            ui.print_error("No active session for /cron run script|tool")
                            continue
                        path_text = " ".join(run_tokens[1:]).strip()
                        if not path_text:
                            ui.print_error(f"Usage: /cron run {head} <path>")
                            continue
                        try:
                            await _run_script_or_tool_in_session(
                                target_session_id=agent.session.id,
                                kind=head,
                                path_text=path_text,
                                trigger="manual",
                            )
                            ui.print_success(f"Cron manual {head} run completed")
                        except Exception as e:
                            ui.print_error(str(e))
                        continue

                    selector = run_tokens[0].strip()
                    job = await agent.session_manager.select_cron_job(selector, active_only=False)
                    if not job:
                        ui.print_error(f"Cron job not found: {selector}")
                        continue
                    job_id = job.id
                    await _execute_cron_job(job, trigger="manual")
                    ui.print_success(f"Manual cron run finished: {job_id}")
                    continue
                elif result == "CONFIG":
                    ui.print_message("system", _format_active_configuration_text())
                    continue
                elif result == "HISTORY":
                    if agent.session:
                        ui.print_history(agent.session.messages)
                    continue
                elif result == "COMPACT":
                    compacted, stats = await agent.compact_session(force=True, trigger="manual")
                    if compacted:
                        if agent.session:
                            ui.load_monitor_tool_output_from_session(agent.session.messages)
                        ui.print_success(
                            "Session compacted "
                            f"({int(stats.get('before_tokens', 0))} -> {int(stats.get('after_tokens', 0))} tokens)"
                        )
                    else:
                        reason = str(stats.get("reason", "not_needed"))
                        ui.print_warning(f"Compaction skipped: {reason}")
                    continue
                elif result == "PLANNING_ON":
                    await agent.set_pipeline_mode("contracts")
                    ui.print_success("Pipeline mode set to contracts")
                    continue
                elif result == "PLANNING_OFF":
                    await agent.set_pipeline_mode("loop")
                    ui.print_success("Pipeline mode set to loop")
                    continue
                elif result == "PIPELINE_INFO":
                    ui.print_success(
                        "Pipeline mode: "
                        f"{agent.pipeline_mode} "
                        "(loop=fast/simple, contracts=planner+completion gate)"
                    )
                    continue
                elif result.startswith("PIPELINE_MODE:"):
                    mode = result.split(":", 1)[1].strip().lower()
                    try:
                        await agent.set_pipeline_mode(mode)
                    except ValueError:
                        ui.print_error("Invalid pipeline mode. Use /pipeline loop|contracts")
                        continue
                    ui.print_success(
                        "Pipeline mode set to "
                        f"{agent.pipeline_mode} "
                        "(loop=fast/simple, contracts=planner+completion gate)"
                    )
                    continue
                elif result == "SKILLS_LIST":
                    skills = agent.list_user_invocable_skills()
                    if not skills:
                        ui.print_warning("No user-invocable skills available.")
                        continue
                    lines = ["Available skills:", "Use `/skill <name> [args]` to run one:"]
                    for command in skills:
                        lines.append(f"- /skill {command.name} - {command.description}")
                    lines.append("Search catalog: `/skill search <criteria>`")
                    lines.append("Install from GitHub: `/skill install <github-url>`")
                    lines.append("Install skill deps: `/skill install <skill-name> [install-id]`")
                    ui.print_message("system", "\n".join(lines))
                    continue
                elif result.startswith("SKILL_SEARCH:"):
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /skill search payload")
                        continue
                    query = str(payload.get("query", "")).strip()
                    if not query:
                        ui.print_error("Usage: /skill search <criteria>")
                        continue
                    ui.print_message("system", f'Searching skills catalog for: "{query}"')
                    search_result = await agent.search_skill_catalog(query)
                    if not bool(search_result.get("ok", False)):
                        ui.print_error(str(search_result.get("error", "Skill search failed.")))
                        continue
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
                    ui.print_message("system", "\n".join(lines))
                    continue
                elif result.startswith("SKILL_INSTALL:"):
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /skill install payload")
                        continue
                    skill_url = str(payload.get("url", "")).strip()
                    skill_name = str(payload.get("name", "")).strip()
                    install_id = str(payload.get("install_id", "")).strip()
                    if skill_url:
                        ui.print_message("system", f"Installing skill from GitHub: {skill_url}")
                        install_result = await agent.install_skill_from_github(skill_url)
                        if not bool(install_result.get("ok", False)):
                            ui.print_error(str(install_result.get("error", "Skill install failed.")))
                            continue
                        skill_name = str(install_result.get("skill_name", "")).strip() or "unknown"
                        destination = str(install_result.get("destination", "")).strip()
                        aliases = list(install_result.get("aliases", []))
                        ui.print_success(f'Installed skill "{skill_name}"')
                        if destination:
                            ui.print_message("system", f"Path: {destination}")
                        if aliases:
                            ui.print_message("system", f"Invoke with: /skill {aliases[0]}")
                        continue
                    if not skill_name:
                        ui.print_error("Usage: /skill install <github-url> | /skill install <skill-name> [install-id]")
                        continue
                    ui.print_message("system", f'Installing dependencies for skill "{skill_name}"')
                    install_result = await agent.install_skill_dependencies(
                        skill_name=skill_name,
                        install_id=install_id or None,
                    )
                    if not bool(install_result.get("ok", False)):
                        ui.print_error(str(install_result.get("error", "Skill dependency install failed.")))
                        continue
                    ui.print_success(str(install_result.get("message", "Dependencies installed.")))
                    command = str(install_result.get("command", "")).strip()
                    if command:
                        ui.print_message("system", f"Command: {command}")
                    continue
                elif result.startswith("SKILL_INVOKE:") or result.startswith("SKILL_ALIAS_INVOKE:"):
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /skill payload")
                        continue
                    skill_name = str(payload.get("name", "")).strip()
                    skill_args = str(payload.get("args", "")).strip()
                    if not skill_name:
                        ui.print_error(
                            "Usage: /skill <name> [args] | /skill search <criteria> | /skill install <github-url> | /skill install <skill-name> [install-id]"
                        )
                        continue
                    invocation = await agent.invoke_skill_command(skill_name, args=skill_args)
                    if not bool(invocation.get("ok", False)):
                        if result.startswith("SKILL_ALIAS_INVOKE:"):
                            ui.print_error(f"Unknown command: /{skill_name}")
                        else:
                            ui.print_error(str(invocation.get("error", "Skill invocation failed.")))
                        continue
                    mode = str(invocation.get("mode", "")).strip().lower()
                    if mode == "dispatch":
                        text = str(invocation.get("text", "")).strip() or "Done."
                        ui.print_message("assistant", text)
                        continue
                    rewritten_prompt = str(invocation.get("prompt", "")).strip()
                    if not rewritten_prompt:
                        ui.print_error("Skill invocation did not return a prompt.")
                        continue
                    display_prompt = f"/skill {skill_name}"
                    if skill_args:
                        display_prompt += f" {skill_args}"
                    await _run_prompt_in_active_session(
                        rewritten_prompt,
                        display_prompt=display_prompt,
                    )
                    continue
                elif result == "MONITOR_ON":
                    ui.set_monitor_mode(True)
                    if agent.session:
                        ui.load_monitor_tool_output_from_session(agent.session.messages)
                    ui.print_success("Monitor enabled")
                    continue
                elif result == "MONITOR_OFF":
                    ui.set_monitor_mode(False)
                    ui.print_success("Monitor disabled")
                    continue
                elif result == "MONITOR_TRACE_ON":
                    await agent.set_monitor_trace_llm(True)
                    ui.print_success("Monitor trace enabled (full intermediate LLM responses will be logged)")
                    continue
                elif result == "MONITOR_TRACE_OFF":
                    await agent.set_monitor_trace_llm(False)
                    ui.print_success("Monitor trace disabled")
                    continue
                elif result == "MONITOR_PIPELINE_ON":
                    await agent.set_monitor_trace_pipeline(True)
                    ui.print_success("Pipeline trace enabled (compact pipeline-only events will be logged)")
                    continue
                elif result == "MONITOR_PIPELINE_OFF":
                    await agent.set_monitor_trace_pipeline(False)
                    ui.print_success("Pipeline trace disabled")
                    continue
                elif result == "MONITOR_FULL_ON":
                    ui.set_monitor_full_output(True)
                    ui.print_success("Monitor full output rendering enabled")
                    continue
                elif result == "MONITOR_FULL_OFF":
                    ui.set_monitor_full_output(False)
                    ui.print_success("Monitor compact output rendering enabled")
                    continue
                elif result == "MONITOR_SCROLL_STATUS":
                    ui.print_success(f"Monitor scroll: {ui.describe_monitor_scroll()}")
                    continue
                elif result.startswith("MONITOR_SCROLL:"):
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /monitor scroll payload")
                        continue
                    pane = str(payload.get("pane", "")).strip().lower()
                    action = str(payload.get("action", "")).strip().lower()
                    amount_raw = payload.get("amount", 1)
                    try:
                        amount = int(amount_raw)
                    except Exception:
                        ui.print_error("Invalid scroll amount")
                        continue
                    ok, message = ui.scroll_monitor_pane(pane=pane, action=action, amount=amount)
                    if ok:
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue

                # Skip empty input
                if not user_input.strip():
                    continue

                await _run_prompt_in_active_session(user_input)

            except KeyboardInterrupt:
                log.info("Interrupted by user")
                break
            except EOFError:
                log.info("EOF received")
                break
            except Exception as e:
                ui.print_error(str(e))
                log.error("Error in interactive loop", error=str(e))
    finally:
        if telegram_poll_task is not None:
            telegram_poll_task.cancel()
            try:
                await telegram_poll_task
            except asyncio.CancelledError:
                pass
        if slack_poll_task is not None:
            slack_poll_task.cancel()
            try:
                await slack_poll_task
            except asyncio.CancelledError:
                pass
        if discord_poll_task is not None:
            discord_poll_task.cancel()
            try:
                await discord_poll_task
            except asyncio.CancelledError:
                pass
        if telegram_bridge is not None:
            try:
                await telegram_bridge.close()
            except Exception:
                pass
        if slack_bridge is not None:
            try:
                await slack_bridge.close()
            except Exception:
                pass
        if discord_bridge is not None:
            try:
                await discord_bridge.close()
            except Exception:
                pass
        cron_worker.cancel()
        try:
            await cron_worker
        except asyncio.CancelledError:
            pass


def version() -> None:
    """Show version information."""
    from captain_claw import __version__
    print(f"Captain Claw v{__version__}")


if __name__ == "__main__":
    import typer

    cli = typer.Typer(help="Captain Claw - A powerful console-based AI agent")

    @cli.command()
    def run(
        config: str = typer.Option("", "-c", "--config", help="Path to config file"),
        model: str = typer.Option("", "-m", "--model", help="Override model"),
        provider: str = typer.Option("", "-p", "--provider", help="Override provider"),
        no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming"),
        verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug logging"),
        onboarding: bool = typer.Option(
            False,
            "--onboarding",
            help="Run interactive onboarding wizard before starting",
        ),
    ) -> None:
        main(config, model, provider, no_stream, verbose, onboarding)

    @cli.command()
    def ver() -> None:
        version()

    cli()
