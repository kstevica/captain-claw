# Summary: orchestrator_cli.py

# orchestrator_cli.py Summary

**Summary:**
Headless orchestrator runner enabling CLI and cron execution of saved workflows or ad-hoc orchestrations without web server or TUI. Provides both programmatic async interface and command-line entry point for integration into automation pipelines, CI/CD systems, and scheduled jobs.

**Purpose:**
Solves the need to execute Captain Claw orchestrations outside interactive environments (web UI, TUI). Enables:
- Cron job scheduling of saved workflows
- One-shot CLI invocation with ad-hoc prompts
- Programmatic integration into external systems
- CI/CD pipeline integration
- Clean stdout/stderr separation (results on stdout, status on stderr)

**Most Important Functions/Classes/Procedures:**

1. **`run_orchestrator_headless()`** — Core async function orchestrating the entire headless execution flow. Handles config loading, agent/orchestrator initialization, workflow loading or ad-hoc prompt decomposition, execution, error handling, and returns structured result dict. Accepts workflow name or prompt, optional config/model/provider overrides, and output formatting flags.

2. **`cli_orchestrate()`** — CLI entry point for `captain-claw orchestrate` command. Parses arguments (workflow name, ad-hoc prompt, config path, model/provider overrides, parallelism settings, quiet/JSON flags), delegates to `run_orchestrator_headless()`, and formats output (JSON or plain text). Handles `--list` flag for workflow enumeration.

3. **`_print_status()` & `_print_event()`** — Callback functions for real-time monitoring. Route status updates and broadcast events (task status, synthesis completion, output saves) to stderr to keep stdout clean for result data. Event filtering by type enables selective logging of orchestration lifecycle events.

4. **`_list_workflows_cli()`** — Utility function supporting `--list` flag. Loads config, initializes orchestrator, retrieves saved workflows, and displays formatted table of workflow names and task counts.

5. **Config & Agent Initialization Block** — Handles configuration loading (from YAML or defaults), parameter overrides (model, provider, parallelism), workspace directory creation, and Agent/SessionOrchestrator instantiation with appropriate callbacks.

**Architecture & Dependencies:**
- Imports from `captain_claw.agent` (LLM provider abstraction), `captain_claw.session_orchestrator` (workflow execution engine), `captain_claw.config` (configuration management), `captain_claw.logging` (structured logging)
- Async-first design using `asyncio` for concurrent task execution
- Callback-based event system for non-blocking status/event reporting
- Dual execution paths: saved workflow loading vs. ad-hoc prompt decomposition
- Clean separation of concerns: CLI parsing, orchestration logic, output formatting