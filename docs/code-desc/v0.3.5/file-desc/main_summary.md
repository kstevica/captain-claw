# Summary: main.py

# Captain Claw Main Entry Point Analysis

## Summary

Main.py serves as the primary entry point for Captain Claw, a console-based AI agent platform. It orchestrates application initialization, CLI argument parsing, configuration loading, and routing between interactive TUI mode, web server mode, and headless orchestration workflows. The file handles complex signal management, graceful shutdown, and multi-modal UI support with sophisticated async event loop coordination.

## Purpose

This module solves several critical problems:
1. **Multi-modal execution**: Routes users to appropriate interface (TUI, web server, or headless orchestration)
2. **Configuration management**: Loads, validates, and applies CLI overrides to system configuration
3. **Graceful lifecycle management**: Implements proper signal handling and cleanup for both interactive and server modes
4. **Onboarding workflow**: Guides first-time users through setup before launching the agent
5. **Headless automation**: Enables cron/script integration via orchestrator subcommand for workflow execution

## Most Important Functions/Classes

### 1. **main()**
Primary entry point that orchestrates the entire startup sequence. Handles CLI argument parsing, configuration loading with overrides, onboarding wizard execution, and routing to either TUI interactive mode or web server mode. Manages logging configuration and ensures required directories exist.

### 2. **run_interactive()**
Implements the core TUI event loop for interactive agent sessions. Manages complex async coordination between user input (blocking thread), signal handlers, cron scheduler, platform lifecycle, and agent execution. Implements two-stage Ctrl+C handling: first signal cancels current turn, second signal forces exit. Handles next-step shortcuts and special command dispatch.

### 3. **_build_runtime_arg_parser()**
Constructs comprehensive argument parser supporting main mode options (config, model, provider, streaming, verbose, onboarding, TUI, port) and orchestrate subcommand for headless workflow execution. Supports mutually exclusive workflow vs. ad-hoc prompt modes with JSON output and parallel worker configuration.

### 4. **_signal_handler()**
Implements graceful shutdown mechanism with two-stage interrupt handling. First SIGINT/SIGTERM sets agent cancel event to stop current turn; second signal sets stop event to break main loop. Enables clean cancellation without force-killing threads, critical for proper resource cleanup and terminal state restoration.

### 5. **_should_parse_runtime_cli_from_argv()**
Determines whether to parse CLI arguments from sys.argv based on program name and argument presence. Prevents double-parsing when main() is called programmatically with explicit parameters. Checks for captain-claw variants and validates that meaningful arguments exist before parsing.

## Architecture & Dependencies

**Key Dependencies:**
- `captain_claw.agent`: Core Agent class for LLM interaction
- `captain_claw.cli`: UI abstraction (get_ui())
- `captain_claw.config`: Configuration loading and management
- `captain_claw.web_server`: Web UI server implementation
- `captain_claw.orchestrator_cli`: Headless workflow execution
- `captain_claw.platform_lifecycle`: Platform bridge initialization/teardown
- `captain_claw.prompt_execution`: Session-based prompt execution
- `captain_claw.cron_dispatch`: Background scheduler for automated tasks
- `typer`: CLI framework for command definition

**System Flow:**
1. CLI parsing → Configuration loading → Onboarding (if needed)
2. Config overrides applied from CLI arguments
3. Session/workspace directories created
4. Route selection: TUI mode (--tui or web.enabled=false) → run_interactive() OR Web mode (default) → run_web_server()
5. For orchestrate subcommand: Skip UI entirely, execute headless workflow, output JSON/text result

**Critical Design Patterns:**
- **Dual-mode CLI**: Supports both typer-based programmatic calls and sys.argv parsing for shell invocation
- **Async event loop coordination**: Races blocking input() against stop_event using asyncio.wait() to enable responsive signal handling
- **Signal handler state machine**: Two-stage Ctrl+C (cancel turn → force exit) prevents abrupt termination
- **Configuration cascade**: Default config → file-based config → CLI overrides
- **Resource cleanup**: Finally block ensures platform teardown, cron cancellation, and terminal state restoration even on force exit

**Notable Technical Decisions:**
- Uses `os._exit(0)` instead of normal exit to bypass thread-pool cleanup (blocking input() thread cannot be interrupted cleanly)
- Terminal scroll region reset before exit prevents shell prompt misalignment
- Cron scheduler runs as background task during interactive session
- Platform bridges initialized after agent setup to ensure proper context availability
- Onboarding wizard runs before configuration finalization to allow user selection of config paths