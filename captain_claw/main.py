"""Main entry point for Captain Claw."""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from captain_claw.config import Config, get_config, set_config
from captain_claw.logging import configure_logging, get_logger, log
from captain_claw.cli import TerminalUI, get_ui
from captain_claw.agent import Agent


def main(
    config: str = "",
    model: str = "",
    provider: str = "",
    no_stream: bool = False,
    verbose: bool = False,
) -> None:
    """Start Captain Claw interactive session."""
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
    agent = Agent()
    
    # Show welcome
    ui.print_welcome()
    
    # Initialize agent
    await agent.initialize()
    
    # Main loop
    while True:
        try:
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
                    ui.print_success("Session cleared")
                continue
            elif result == "NEW":
                agent.session = await agent.session_manager.get_or_create_session()
                ui.print_success("Started new session")
                continue
            elif result == "CONFIG":
                ui.print_config(get_config())
                continue
            elif result == "HISTORY":
                if agent.session:
                    ui.print_history(agent.session.messages)
                continue
            
            # Skip empty input
            if not user_input.strip():
                continue
            
            # Process through agent
            ui.print_message("user", user_input)
            print()
            
            # Get response
            try:
                response = await agent.complete(user_input)
                ui.print_message("assistant", response)
                print()
            except Exception as e:
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
