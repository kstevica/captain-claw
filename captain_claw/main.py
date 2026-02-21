"""Main entry point for Captain Claw."""

import argparse
import asyncio
import os
import signal
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
    parser.add_argument("--tui", action="store_true", help="Start the terminal UI instead of the web UI")
    parser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")

    # Orchestrate subcommand (headless workflow execution for cron/scripts).
    subparsers = parser.add_subparsers(dest="subcommand")
    orch_parser = subparsers.add_parser("orchestrate", help="Run orchestrated workflows headlessly")
    orch_group = orch_parser.add_mutually_exclusive_group()
    orch_group.add_argument("-w", "--workflow", default="", help="Saved workflow name to execute")
    orch_group.add_argument("prompt", nargs="?", default="", help="Ad-hoc prompt to orchestrate")
    orch_parser.add_argument("--max-parallel", type=int, default=0, help="Max parallel workers")
    orch_parser.add_argument("-q", "--quiet", action="store_true", help="Suppress status output")
    orch_parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    orch_parser.add_argument("--list", action="store_true", dest="list_workflows", help="List saved workflows")

    return parser


def _should_parse_runtime_cli_from_argv(
    config: str,
    model: str,
    provider: str,
    no_stream: bool,
    verbose: bool,
    onboarding: bool,
    tui: bool = False,
) -> bool:
    if config or model or provider or no_stream or verbose or onboarding or tui:
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
    tui: bool = False,
) -> None:
    """Start Captain Claw interactive session."""
    if _should_parse_runtime_cli_from_argv(
        config=config,
        model=model,
        provider=provider,
        no_stream=no_stream,
        verbose=verbose,
        onboarding=onboarding,
        tui=tui,
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
        # Handle orchestrate subcommand before regular startup.
        if getattr(parsed, "subcommand", None) == "orchestrate":
            if verbose:
                os.environ["CLAW_LOGGING__LEVEL"] = "DEBUG"
            from captain_claw.orchestrator_cli import run_orchestrator_headless, _list_workflows_cli
            configure_logging()

            cfg_path = str(parsed.config or "")
            mdl = str(parsed.model or "")
            prov = str(parsed.provider or "")

            if getattr(parsed, "list_workflows", False):
                asyncio.run(_list_workflows_cli(cfg_path))
                return

            wf = getattr(parsed, "workflow", "") or ""
            prompt_text = getattr(parsed, "prompt", "") or ""
            if not wf and not prompt_text:
                print("Error: Either --workflow or a prompt is required.")
                sys.exit(1)

            result = asyncio.run(run_orchestrator_headless(
                workflow_name=wf,
                prompt=prompt_text,
                config_path=cfg_path,
                model=mdl,
                provider=prov,
                max_parallel=getattr(parsed, "max_parallel", 0),
                quiet=getattr(parsed, "quiet", False),
                json_output=getattr(parsed, "json_output", False),
            ))

            if getattr(parsed, "json_output", False):
                import json as _json
                print(_json.dumps(result, ensure_ascii=False, indent=2))
            elif result["ok"]:
                print(result["result"])
            else:
                sys.stderr.write(f"Error: {result['error']}\n")
                sys.exit(1)
            return

        config = str(parsed.config or "")
        model = str(parsed.model or "")
        provider = str(parsed.provider or "")
        no_stream = bool(parsed.no_stream)
        verbose = bool(parsed.verbose)
        onboarding = bool(parsed.onboarding)
        tui = bool(parsed.tui)
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

    # TUI mode (opt-in via --tui or web.enabled=false in config)
    if tui or not cfg.web.enabled:
        ui = get_ui()
        set_system_log_sink(ui.append_system_line if ui.has_sticky_layout() else None)

        try:
            asyncio.run(run_interactive())
        except KeyboardInterrupt:
            log.info("Shutting down...")
            sys.exit(0)
        except Exception as e:
            log.error("Fatal error", error=str(e))
            sys.exit(1)
        return

    # Web UI mode (default)
    from captain_claw.web_server import run_web_server

    try:
        run_web_server(cfg)
    except KeyboardInterrupt:
        log.info("Web server shutting down...")
        sys.exit(0)
    except Exception as e:
        log.error("Web server fatal error", error=str(e))
        sys.exit(1)


async def run_interactive() -> None:
    """Run the interactive agent loop."""
    from captain_claw.cron_dispatch import cron_scheduler_loop
    from captain_claw.local_command_dispatch import dispatch_local_command
    from captain_claw.platform_lifecycle import init_platforms, teardown_platforms
    from captain_claw.prompt_execution import run_prompt_in_active_session
    from captain_claw.runtime_context import RuntimeContext

    loop = asyncio.get_event_loop()

    # Stop event: set by signal handler to trigger graceful shutdown,
    # mirroring the web server pattern.  The blocking ``input()`` call
    # lives in a thread-pool worker that cannot be interrupted by Python
    # signals, so we need this event to break out of the main loop and
    # then force-exit after cleanup (just like web mode does).
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        if not stop_event.is_set():
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except (NotImplementedError, OSError):
            pass  # Windows doesn't support add_signal_handler for SIGTERM.

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
        while not stop_event.is_set():
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

                # Get user input (threaded so event loop keeps servicing
                # cron / platform bridges).  We race the blocking prompt
                # against the stop event so Ctrl+C breaks out immediately
                # even though input() is stuck in a thread.
                prompt_task = asyncio.ensure_future(
                    asyncio.to_thread(ui.prompt)
                )
                stop_task = asyncio.ensure_future(stop_event.wait())

                done, pending = await asyncio.wait(
                    [prompt_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # If stop was signalled, exit the loop.
                if stop_task in done:
                    prompt_task.cancel()
                    break

                # Otherwise stop_task is still pending; cancel it so it
                # doesn't linger.
                stop_task.cancel()
                user_input: str = prompt_task.result()

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
        log.info("Shutting down...")
        await teardown_platforms(ctx)
        cron_worker.cancel()
        try:
            await cron_worker
        except asyncio.CancelledError:
            pass
        # Restore terminal scroll region & cursor position before
        # force-exiting — os._exit() skips atexit handlers, so the
        # registered _reset_scroll_region would never run.
        ui._reset_scroll_region()
        # Move cursor below the status/prompt area so the shell
        # prompt appears on a clean line.
        sys.stdout.write("\033[999;1H\n")
        sys.stdout.flush()
        # Force exit to avoid hanging on asyncio.run()'s
        # shutdown_default_executor / thread-pool joins — the
        # blocking input() thread cannot be interrupted cleanly.
        os._exit(0)


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
        tui: bool = typer.Option(
            False,
            "--tui",
            help="Start the terminal UI instead of the web UI",
        ),
    ) -> None:
        main(config, model, provider, no_stream, verbose, onboarding, tui=tui)

    @cli.command()
    def ver() -> None:
        version()

    cli()
