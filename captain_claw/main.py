"""Main entry point for Captain Claw."""

import asyncio
from datetime import datetime
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
    
    # Main loop
    while True:
        try:
            ui.print_status_line(
                last_usage=agent.last_usage,
                total_usage=agent.total_usage,
                last_exec_seconds=last_exec_seconds,
                last_completed_at=last_completed_at,
                session_id=agent.session.id if agent.session else None,
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
                agent.session = await agent.session_manager.create_session(name=session_name)
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
            elif result == "SESSION_INFO":
                if agent.session:
                    ui.print_session_info(agent.session)
                else:
                    ui.print_error("No active session")
                continue
            elif result.startswith("SESSION_SELECT:"):
                selector = result.split(":", 1)[1].strip()
                selected = await agent.session_manager.select_session(selector)
                if not selected:
                    ui.print_error(f"Session not found: {selector}")
                    continue
                agent.session = selected
                ui.load_monitor_tool_output_from_session(agent.session.messages)
                ui.print_session_info(agent.session)
                ui.print_success("Loaded session")
                continue
            elif result == "CONFIG":
                ui.print_config(get_config())
                continue
            elif result == "HISTORY":
                if agent.session:
                    ui.print_history(agent.session.messages)
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
            
            # Process through agent
            ui.print_message("user", user_input)
            ui.print_blank_line()
            
            # Get response
            try:
                started = time.perf_counter()
                ui.set_runtime_status("thinking")
                if get_config().ui.streaming:
                    ui.begin_assistant_stream()
                    ui.set_runtime_status("streaming")

                    async def _consume_stream() -> None:
                        async for chunk in agent.stream(user_input):
                            ui.print_streaming(chunk)
                        ui.complete_stream_line()

                    try:
                        _, cancelled = await _run_cancellable(ui, _consume_stream())
                    finally:
                        ui.end_assistant_stream()
                    if cancelled:
                        ui.print_blank_line()
                        continue
                else:
                    response, cancelled = await _run_cancellable(ui, agent.complete(user_input))
                    if cancelled:
                        ui.print_blank_line()
                        continue
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
