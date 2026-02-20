"""Main entry point for Captain Claw."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from captain_claw.agent import Agent
from captain_claw.cli import get_ui
from captain_claw.config import Config, get_config, set_config
from captain_claw.logging import configure_logging, log, set_system_log_sink
from captain_claw.onboarding import run_onboarding_wizard, should_run_onboarding


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
    from captain_claw.cron_dispatch import cron_scheduler_loop
    from captain_claw.local_command_dispatch import dispatch_local_command
    from captain_claw.platform_lifecycle import init_platforms, teardown_platforms
    from captain_claw.prompt_execution import run_prompt_in_active_session
    from captain_claw.runtime_context import RuntimeContext

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

    ctx = RuntimeContext(agent=agent, ui=ui)

    # Start platform bridges
    await init_platforms(ctx)

    # Background cron scheduler
    cron_worker = asyncio.create_task(cron_scheduler_loop(ctx))
    try:
        # Main loop
        while True:
            try:
                ui.print_status_line(
                    last_usage=agent.last_usage,
                    total_usage=agent.total_usage,
                    last_exec_seconds=ctx.last_exec_seconds,
                    last_completed_at=ctx.last_completed_at,
                    session_id=agent.session.id if agent.session else None,
                    context_window=agent.last_context_window,
                    model_details=agent.get_runtime_model_details(),
                )
                ui.set_runtime_status("user input")
                # Get user input (threaded so event loop keeps servicing cron).
                user_input = await asyncio.to_thread(ui.prompt)

                # Handle special commands
                result = ui.handle_special_command(user_input)

                if result is None:
                    continue

                # Dispatch local commands
                action = await dispatch_local_command(ctx, result, user_input)
                if action == "break":
                    break
                if action == "continue":
                    continue

                # Skip empty input
                if not user_input.strip():
                    continue

                await run_prompt_in_active_session(ctx, user_input)

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
        await teardown_platforms(ctx)
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
