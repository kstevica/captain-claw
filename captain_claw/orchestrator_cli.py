"""Headless orchestrator runner for CLI and cron execution.

Provides a standalone async function to run saved workflows or ad-hoc
orchestrations without the web server or TUI.  Designed for:

- Cron jobs: ``captain-claw orchestrate --workflow my-workflow``
- One-shot CLI: ``captain-claw orchestrate "Fetch X, Y, Z and compare"``
- Programmatic use: ``await run_orchestrator_headless(...)``
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from captain_claw.config import Config, get_config, set_config
from captain_claw.logging import configure_logging, get_logger

log = get_logger(__name__)


def _print_status(status: str) -> None:
    """Print status updates to stderr (keeps stdout clean for result)."""
    sys.stderr.write(f"[orchestrator] {status}\n")
    sys.stderr.flush()


def _print_event(event: dict[str, Any]) -> None:
    """Print broadcast events to stderr for monitoring."""
    etype = event.get("type", "")
    if etype == "task_status":
        tid = event.get("task_id", "")
        status = event.get("status", "")
        title = event.get("title", "")
        sys.stderr.write(f"  [{status}] {title} ({tid})\n")
        sys.stderr.flush()
    elif etype in ("completed", "synthesizing", "executing"):
        sys.stderr.write(f"[orchestrator] {etype}\n")
        sys.stderr.flush()
    elif etype == "output_saved":
        path = event.get("path", "")
        sys.stderr.write(f"[orchestrator] Output saved: {path}\n")
        sys.stderr.flush()


async def run_orchestrator_headless(
    *,
    workflow_name: str = "",
    prompt: str = "",
    config_path: str = "",
    model: str = "",
    provider: str = "",
    max_parallel: int = 0,
    quiet: bool = False,
    json_output: bool = False,
) -> dict[str, Any]:
    """Run an orchestration headlessly (no web server, no TUI).

    Either ``workflow_name`` or ``prompt`` must be provided.

    Args:
        workflow_name: Name of a saved workflow to load and execute.
        prompt: Ad-hoc prompt to decompose and execute.
        config_path: Optional path to config YAML.
        model: Override model name.
        provider: Override LLM provider.
        max_parallel: Override max parallel workers (0 = use config).
        quiet: Suppress stderr status output.
        json_output: Return structured JSON result.

    Returns:
        Dict with ``ok``, ``result`` (synthesis text), ``workflow_name``,
        ``output_path``, and ``error`` (if failed).
    """
    from captain_claw.agent import Agent
    from captain_claw.session_orchestrator import SessionOrchestrator

    if not workflow_name and not prompt:
        return {"ok": False, "error": "Either --workflow or a prompt is required."}

    # --- Setup config ---
    if config_path:
        cfg = Config.from_yaml(Path(config_path))
    else:
        cfg = Config.load()

    if model:
        cfg.model.model = model
    if provider:
        cfg.model.provider = provider
    set_config(cfg)

    # Ensure workspace exists.
    workspace = cfg.resolved_workspace_path(Path.cwd())
    workspace.mkdir(parents=True, exist_ok=True)

    session_path = Path(cfg.session.path).expanduser()
    session_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Callbacks ---
    status_cb = (lambda s: None) if quiet else _print_status
    broadcast_cb = (lambda e: None) if quiet else _print_event

    # --- Create agent (for LLM provider + decompose/synthesize) ---
    agent = Agent(
        status_callback=status_cb,
    )
    await agent.initialize()

    # --- Create orchestrator ---
    orch_cfg = cfg.orchestrator
    orchestrator = SessionOrchestrator(
        main_agent=agent,
        max_parallel=max_parallel or orch_cfg.max_parallel,
        max_agents=orch_cfg.max_agents,
        provider=agent.provider,
        status_callback=status_cb,
        broadcast_callback=broadcast_cb,
    )

    result_data: dict[str, Any] = {
        "ok": False,
        "result": "",
        "workflow_name": "",
        "output_path": "",
        "error": "",
    }

    try:
        if workflow_name:
            # Load saved workflow and execute.
            status_cb(f"Loading workflow: {workflow_name}")
            load_result = await orchestrator.load_workflow(workflow_name)
            if not load_result.get("ok"):
                result_data["error"] = load_result.get("error", "Failed to load workflow.")
                return result_data

            result_data["workflow_name"] = load_result.get("workflow_name", workflow_name)
            status_cb(f"Executing workflow: {result_data['workflow_name']} "
                       f"({len(load_result.get('tasks', []))} tasks)")

            synthesis = await orchestrator.execute()
            result_data["ok"] = True
            result_data["result"] = synthesis

        else:
            # Ad-hoc prompt: decompose + execute.
            status_cb("Preparing orchestration...")
            synthesis = await orchestrator.orchestrate(prompt)
            result_data["ok"] = True
            result_data["result"] = synthesis
            result_data["workflow_name"] = orchestrator._workflow_name

    except Exception as e:
        log.error("Orchestration failed", error=str(e))
        result_data["error"] = str(e)
    finally:
        await orchestrator.shutdown()

    return result_data


def cli_orchestrate() -> None:
    """CLI entry point for ``captain-claw orchestrate``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="captain-claw orchestrate",
        description="Run orchestrated workflows headlessly (for cron, scripts, CI).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-w", "--workflow",
        default="",
        help="Name of a saved workflow to load and execute.",
    )
    group.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="Ad-hoc prompt to decompose and execute.",
    )

    parser.add_argument("-c", "--config", default="", help="Path to config YAML.")
    parser.add_argument("-m", "--model", default="", help="Override model.")
    parser.add_argument("-p", "--provider", default="", help="Override LLM provider.")
    parser.add_argument("--max-parallel", type=int, default=0, help="Max parallel workers.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress status output.")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output result as JSON.")
    parser.add_argument("--list", action="store_true", dest="list_workflows", help="List saved workflows and exit.")

    args = parser.parse_args()

    configure_logging()

    if args.list_workflows:
        asyncio.run(_list_workflows_cli(args.config))
        return

    result = asyncio.run(run_orchestrator_headless(
        workflow_name=args.workflow,
        prompt=args.prompt or "",
        config_path=args.config,
        model=args.model,
        provider=args.provider,
        max_parallel=args.max_parallel,
        quiet=args.quiet,
        json_output=args.json_output,
    ))

    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["ok"]:
            print(result["result"])
        else:
            sys.stderr.write(f"Error: {result['error']}\n")
            sys.exit(1)


async def _list_workflows_cli(config_path: str = "") -> None:
    """List saved workflows (for --list flag)."""
    from captain_claw.session_orchestrator import SessionOrchestrator

    if config_path:
        cfg = Config.from_yaml(Path(config_path))
    else:
        cfg = Config.load()
    set_config(cfg)

    workspace = cfg.resolved_workspace_path(Path.cwd())
    workspace.mkdir(parents=True, exist_ok=True)

    orchestrator = SessionOrchestrator()
    workflows = await orchestrator.list_workflows()

    if not workflows:
        print("No saved workflows found.")
        return

    print(f"{'Name':<40} {'Tasks':>5}")
    print("-" * 47)
    for wf in workflows:
        print(f"{wf['name']:<40} {wf['task_count']:>5}")


if __name__ == "__main__":
    cli_orchestrate()
