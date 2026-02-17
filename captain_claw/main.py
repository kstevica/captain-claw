"""Main entry point for Captain Claw."""

import asyncio
from datetime import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Awaitable, TypeVar

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from captain_claw.config import Config, get_config, set_config
from captain_claw.logging import configure_logging, get_logger, log, set_system_log_sink
from captain_claw.cli import TerminalUI, get_ui
from captain_claw.agent import Agent

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


def main(
    config: str = "",
    model: str = "",
    provider: str = "",
    no_stream: bool = False,
    verbose: bool = False,
) -> None:
    """Start Captain Claw interactive session."""
    ui = get_ui()
    set_system_log_sink(ui.append_system_line if ui.has_sticky_layout() else None)

    # Configure logging first
    if verbose:
        os.environ["CLAW_LOGGING__LEVEL"] = "DEBUG"
    configure_logging()
    
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
    agent = Agent(
        status_callback=ui.set_runtime_status,
        tool_output_callback=ui.append_tool_output,
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

    async def _run_prompt_in_active_session(prompt_text: str) -> None:
        """Execute one user prompt using the currently selected session."""
        nonlocal last_exec_seconds
        nonlocal last_completed_at

        if not prompt_text.strip():
            return

        ui.print_message("user", prompt_text)
        ui.print_blank_line()

        try:
            started = time.perf_counter()
            ui.set_runtime_status("thinking")
            if get_config().ui.streaming:
                ui.begin_assistant_stream()
                ui.set_runtime_status("streaming")

                async def _consume_stream() -> None:
                    async for chunk in agent.stream(prompt_text):
                        ui.print_streaming(chunk)
                    ui.complete_stream_line()

                try:
                    _, cancelled = await _run_cancellable(ui, _consume_stream())
                finally:
                    ui.end_assistant_stream()
                if cancelled:
                    ui.print_blank_line()
                    return
            else:
                response, cancelled = await _run_cancellable(ui, agent.complete(prompt_text))
                if cancelled:
                    ui.print_blank_line()
                    return
                ui.print_message("assistant", response)
                ui.print_blank_line()

            last_exec_seconds = time.perf_counter() - started
            last_completed_at = datetime.now()
            ui.set_runtime_status("waiting")
        except Exception as e:
            last_exec_seconds = time.perf_counter() - started
            last_completed_at = datetime.now()
            ui.set_runtime_status("waiting")
            ui.print_error(str(e))
            log.error("Error in agent", error=str(e))
    
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
            # Get user input
            user_input = ui.prompt()
            
            # Handle special commands
            result = ui.handle_special_command(user_input)
            
            if result is None:
                continue
            elif result == "EXIT":
                log.info("User requested exit")
                break
            elif result == "CLEAR":
                if agent.session:
                    agent.session.messages = []
                    await agent.session_manager.save_session(agent.session)
                    ui.clear_monitor_tool_output()
                    ui.print_success("Session cleared")
                continue
            elif result == "NEW" or result.startswith("NEW:"):
                session_name = "default"
                if result.startswith("NEW:"):
                    session_name = result.split(":", 1)[1].strip() or "default"
                planning_enabled_before = agent.planning_enabled
                agent.session = await agent.session_manager.create_session(name=session_name)
                agent.refresh_session_runtime_flags()
                if planning_enabled_before and not agent.planning_enabled:
                    await agent.set_planning_mode(True)
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
            elif result.startswith("SESSION_SELECT:"):
                selector = result.split(":", 1)[1].strip()
                selected = await agent.session_manager.select_session(selector)
                if not selected:
                    ui.print_error(f"Session not found: {selector}")
                    continue
                agent.session = selected
                agent.refresh_session_runtime_flags()
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

                previous_session = agent.session
                previous_session_id = previous_session.id if previous_session else None
                switched_temporarily = previous_session_id != selected.id

                if switched_temporarily:
                    agent.session = selected
                    agent.refresh_session_runtime_flags()
                    ui.load_monitor_tool_output_from_session(agent.session.messages)
                    ui.print_success(f'Running in session "{agent.session.name}"')

                try:
                    await _run_prompt_in_active_session(prompt)
                finally:
                    if switched_temporarily and previous_session is not None:
                        restored = await agent.session_manager.load_session(previous_session_id)
                        agent.session = restored or previous_session
                        agent.refresh_session_runtime_flags()
                        ui.load_monitor_tool_output_from_session(agent.session.messages)
                        ui.print_success(f'Restored session "{agent.session.name}"')
                continue
            elif result == "CONFIG":
                ui.print_config(get_config())
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
                await agent.set_planning_mode(True)
                ui.print_success("Planning mode enabled")
                continue
            elif result == "PLANNING_OFF":
                await agent.set_planning_mode(False)
                ui.print_success("Planning mode disabled")
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
    ) -> None:
        main(config, model, provider, no_stream, verbose)
    
    @cli.command()
    def ver() -> None:
        version()
    
    cli()
