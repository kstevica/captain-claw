"""Web UI server for Captain Claw."""

import asyncio
import json
import os
import secrets
import signal
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from captain_claw.agent import Agent
from captain_claw.agent_pool import AgentPool
from captain_claw.config import Config, get_config, set_config
from captain_claw.google_oauth import (
    build_authorization_url,
    exchange_code_for_tokens,
    fetch_user_info,
    generate_pkce_pair,
)
from captain_claw.google_oauth_manager import GoogleOAuthManager
from captain_claw.instructions import InstructionLoader
from captain_claw.logging import configure_logging, get_logger
from captain_claw.session import get_session_manager
from captain_claw.session_orchestrator import SessionOrchestrator
from captain_claw.telegram_bridge import TelegramBridge, TelegramMessage

log = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"

# Available commands for the help/suggestion system
COMMANDS: list[dict[str, str]] = [
    {"command": "/help", "description": "Show command reference", "category": "General"},
    {"command": "/clear", "description": "Clear active session messages", "category": "General"},
    {"command": "/config", "description": "Show active configuration", "category": "General"},
    {"command": "/history", "description": "Show recent conversation history", "category": "General"},
    {"command": "/compact", "description": "Manually compact session memory", "category": "General"},
    {"command": "/exit", "description": "Exit Captain Claw", "category": "General"},
    {"command": "/new [name]", "description": "Create a new session", "category": "Sessions"},
    {"command": "/session", "description": "Show active session info", "category": "Sessions"},
    {"command": "/sessions", "description": "List recent sessions", "category": "Sessions"},
    {"command": "/session switch <id|name|#N>", "description": "Switch to another session", "category": "Sessions"},
    {"command": "/session rename <name>", "description": "Rename the active session", "category": "Sessions"},
    {"command": "/session description <text>", "description": "Set session description", "category": "Sessions"},
    {"command": "/session description auto", "description": "Auto-generate description from context", "category": "Sessions"},
    {"command": "/session export [chat|monitor|all]", "description": "Export session history", "category": "Sessions"},
    {"command": "/session protect on|off", "description": "Toggle memory reset protection", "category": "Sessions"},
    {"command": "/session model", "description": "Show active model for session", "category": "Models"},
    {"command": "/session model <id|#N|default>", "description": "Set model for session", "category": "Models"},
    {"command": "/models", "description": "List allowed models", "category": "Models"},
    {"command": "/pipeline", "description": "Show pipeline mode", "category": "Pipeline"},
    {"command": "/pipeline loop|contracts", "description": "Set pipeline execution mode", "category": "Pipeline"},
    {"command": "/planning on|off", "description": "Legacy alias for pipeline", "category": "Pipeline"},
    {"command": "/skills", "description": "List available skills", "category": "Skills"},
    {"command": "/skill <name> [args]", "description": "Invoke a skill", "category": "Skills"},
    {"command": "/skill search <criteria>", "description": "Search skill catalog", "category": "Skills"},
    {"command": "/skill install <url|name>", "description": "Install a skill", "category": "Skills"},
    {"command": "/cron \"<task>\"", "description": "Run one-off background task", "category": "Cron"},
    {"command": "/cron add every <interval> <task>", "description": "Schedule recurring task", "category": "Cron"},
    {"command": "/cron add daily <HH:MM> <task>", "description": "Schedule daily task", "category": "Cron"},
    {"command": "/cron list", "description": "List scheduled jobs", "category": "Cron"},
    {"command": "/cron run <job-id>", "description": "Execute a scheduled job now", "category": "Cron"},
    {"command": "/cron pause|resume|remove <job-id>", "description": "Manage scheduled jobs", "category": "Cron"},
    {"command": "/monitor on|off", "description": "Toggle monitor split view", "category": "Monitor"},
    {"command": "/monitor trace on|off", "description": "Toggle LLM trace logging", "category": "Monitor"},
    {"command": "/approve user telegram <token>", "description": "Approve a Telegram user pairing", "category": "Telegram"},
    {"command": "/orchestrate <request>", "description": "Run parallel multi-session orchestration", "category": "Orchestrator"},
]


class WebServer:
    """Captain Claw web UI server."""

    def __init__(self, config: Config):
        self.config = config
        self.agent: Agent | None = None
        self.clients: set[web.WebSocketResponse] = set()
        self._busy = False
        self._busy_lock = asyncio.Lock()
        self._telegram_queue: asyncio.Queue[TelegramMessage] = asyncio.Queue()
        self._instr_loader = InstructionLoader()
        self._instructions_dir = self._instr_loader.base_dir
        self._instructions_personal_dir = self._instr_loader.personal_dir
        self._loop: asyncio.AbstractEventLoop | None = None

        # Telegram bridge state
        telegram_cfg = config.telegram
        self._telegram_enabled = bool(
            telegram_cfg.enabled and telegram_cfg.bot_token.strip()
        )
        self._telegram_bridge: TelegramBridge | None = None
        self._telegram_offset: int | None = None
        self._approved_telegram_users: dict[str, dict[str, object]] = {}
        self._pending_telegram_pairings: dict[str, dict[str, object]] = {}
        self._telegram_poll_task: asyncio.Task[None] | None = None
        self._telegram_worker_task: asyncio.Task[None] | None = None
        # Orchestrator (lazy init in _init_agent)
        self._orchestrator: SessionOrchestrator | None = None
        # OpenAI-compatible API agent pool (lazy init in _init_agent)
        self._api_pool: AgentPool | None = None
        # Google OAuth state
        self._oauth_manager: GoogleOAuthManager | None = None
        self._pending_oauth: dict[str, dict[str, Any]] = {}  # state → {verifier, ts}

    async def _init_agent(self) -> None:
        """Initialize the agent with web callbacks."""
        self.agent = Agent(
            status_callback=self._status_callback,
            tool_output_callback=self._tool_output_callback,
            approval_callback=self._approval_callback,
        )
        await self.agent.initialize()

        # Initialize orchestrator (shares provider and callbacks with main agent).
        cfg = get_config()
        self._orchestrator = SessionOrchestrator(
            main_agent=self.agent,
            max_parallel=cfg.orchestrator.max_parallel,
            max_agents=cfg.orchestrator.max_agents,
            provider=self.agent.provider,
            status_callback=self._status_callback,
            tool_output_callback=self._tool_output_callback,
            broadcast_callback=self._broadcast,
        )

        # Initialize OpenAI-compatible API agent pool if enabled.
        if cfg.web.api_enabled:
            self._api_pool = AgentPool(
                max_agents=cfg.web.api_pool_max_agents,
                idle_evict_seconds=cfg.web.api_pool_idle_seconds,
                provider=self.agent.provider,
                session_name_prefix="api",
            )

        # Initialize Google OAuth manager and inject stored tokens.
        if cfg.google_oauth.enabled and cfg.google_oauth.client_id:
            self._oauth_manager = GoogleOAuthManager(self.agent.session_manager)
            await self._inject_oauth_into_provider()

    def _status_callback(self, status: str) -> None:
        """Broadcast status updates to all connected clients."""
        self._broadcast({"type": "status", "status": status})

    def _tool_output_callback(
        self, tool_name: str, arguments: dict[str, Any], output: str
    ) -> None:
        """Broadcast tool output to monitor pane."""
        self._broadcast({
            "type": "monitor",
            "tool_name": tool_name,
            "arguments": arguments,
            "output": output,
        })

    def _approval_callback(self, message: str) -> bool:
        """Handle tool approval requests from the agent.

        The agent calls this synchronously from within its async tool execution
        path.  Since this blocks the event loop thread, we cannot do a
        WebSocket round-trip to ask the user.  Instead we auto-approve and
        notify all connected clients so they can see what was executed in the
        monitor pane.  Users can restrict tool execution via config policies.
        """
        self._broadcast({
            "type": "approval_notice",
            "message": message,
        })
        return True

    def _broadcast(self, msg: dict[str, Any]) -> None:
        """Send a message to all connected WebSocket clients."""
        data = json.dumps(msg, default=str)
        stale: list[web.WebSocketResponse] = []
        for ws in self.clients:
            if ws.closed:
                stale.append(ws)
                continue
            try:
                asyncio.ensure_future(ws.send_str(data))
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.clients.discard(ws)

    async def _send(self, ws: web.WebSocketResponse, msg: dict[str, Any]) -> None:
        """Send a message to a single WebSocket client."""
        if not ws.closed:
            await ws.send_str(json.dumps(msg, default=str))

    # ── Session helpers ──────────────────────────────────────────────

    def _session_info(self) -> dict[str, Any]:
        """Current session info payload."""
        if not self.agent or not self.agent.session:
            return {}
        s = self.agent.session
        model_details = self.agent.get_runtime_model_details()
        return {
            "id": s.id,
            "name": s.name,
            "model": model_details.get("model", ""),
            "provider": model_details.get("provider", ""),
            "description": (s.metadata or {}).get("description", ""),
            "message_count": len(s.messages),
        }

    # ── WebSocket handler ────────────────────────────────────────────

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)
        self.clients.add(ws)

        # Send welcome payload
        models = self.agent.get_allowed_models() if self.agent else []
        await self._send(ws, {
            "type": "welcome",
            "session": self._session_info(),
            "models": models,
            "commands": COMMANDS,
        })

        # Replay existing session messages for the connecting client
        if self.agent and self.agent.session:
            for msg in self.agent.session.messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                tool_name = msg.get("tool_name", "")
                if role in ("user", "assistant"):
                    await self._send(ws, {
                        "type": "chat_message",
                        "role": role,
                        "content": content,
                        "replay": True,
                    })
                elif role == "tool" and tool_name and not Agent._is_monitor_only_tool_name(tool_name):
                    await self._send(ws, {
                        "type": "monitor",
                        "tool_name": tool_name,
                        "arguments": msg.get("tool_arguments", {}),
                        "output": content,
                        "replay": True,
                    })
            await self._send(ws, {"type": "replay_done"})

        try:
            async for raw_msg in ws:
                if raw_msg.type in (
                    web.WSMsgType.TEXT,
                ):
                    try:
                        data = json.loads(raw_msg.data)
                    except json.JSONDecodeError:
                        await self._send(ws, {"type": "error", "message": "Invalid JSON"})
                        continue
                    await self._handle_ws_message(ws, data)
                elif raw_msg.type == web.WSMsgType.ERROR:
                    log.error("WebSocket error", error=str(ws.exception()))
        finally:
            self.clients.discard(ws)

        return ws

    async def _handle_ws_message(
        self, ws: web.WebSocketResponse, data: dict[str, Any]
    ) -> None:
        """Dispatch incoming WebSocket messages."""
        msg_type = data.get("type", "")

        if msg_type == "chat":
            content = str(data.get("content", "")).strip()
            if not content:
                return
            # Check if it's a command
            if content.startswith("/"):
                await self._handle_command(ws, content)
            else:
                await self._handle_chat(ws, content)

        elif msg_type == "command":
            command = str(data.get("command", "")).strip()
            if command:
                await self._handle_command(ws, command)

        elif msg_type == "cancel":
            # Future: implement cancellation
            pass

        elif msg_type == "approval_response":
            # Reserved for future interactive approval support
            pass

    # ── Chat handler ─────────────────────────────────────────────────

    async def _handle_chat(self, ws: web.WebSocketResponse, content: str) -> None:
        """Process a chat message through the agent."""
        if not self.agent:
            await self._send(ws, {"type": "error", "message": "Agent not initialized"})
            return

        if self._busy:
            await self._send(ws, {
                "type": "error",
                "message": "Agent is busy processing another request. Please wait.",
            })
            return

        async with self._busy_lock:
            self._busy = True
            self._broadcast({"type": "status", "status": "thinking"})

            # Echo user message to all clients
            self._broadcast({
                "type": "chat_message",
                "role": "user",
                "content": content,
            })

            try:
                # Route /orchestrate requests to the orchestrator.
                stripped = content.strip()
                if stripped.lower().startswith("/orchestrate ") and self._orchestrator:
                    orchestrate_input = stripped[len("/orchestrate "):].strip()
                    if not orchestrate_input:
                        self._broadcast({
                            "type": "error",
                            "message": "Usage: /orchestrate <request>",
                        })
                    else:
                        response = await self._orchestrator.orchestrate(orchestrate_input)
                        self._broadcast({
                            "type": "chat_message",
                            "role": "assistant",
                            "content": response,
                        })
                else:
                    # Use complete() which handles tool calls and guards
                    response = await self.agent.complete(content)

                    self._broadcast({
                        "type": "chat_message",
                        "role": "assistant",
                        "content": response,
                    })

                # Send updated usage/session info
                self._broadcast({
                    "type": "usage",
                    "last": self.agent.last_usage,
                    "total": self.agent.total_usage,
                })
                self._broadcast({
                    "type": "session_info",
                    **self._session_info(),
                })

            except Exception as e:
                log.error("Chat error", error=str(e))
                self._broadcast({
                    "type": "error",
                    "message": f"Error: {str(e)}",
                })
            finally:
                self._busy = False
                self._broadcast({"type": "status", "status": "ready"})

    # ── Command handler ──────────────────────────────────────────────

    async def _handle_command(self, ws: web.WebSocketResponse, raw: str) -> None:
        """Handle slash commands."""
        if not self.agent:
            return

        parts = raw.strip().split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        result = ""

        try:
            if cmd in ("/help", "/h"):
                result = self._format_help()

            elif cmd in ("/clear",):
                if self.agent.session:
                    self.agent.session.messages.clear()
                    await self.agent.session_manager.save_session(self.agent.session)
                    self.agent.last_usage = self.agent._empty_usage()
                    self.agent.last_context_window = {}
                result = "Session messages cleared."

            elif cmd in ("/config",):
                cfg = get_config()
                details = self.agent.get_runtime_model_details()
                result = (
                    f"**Model:** {details.get('provider', '')}:{details.get('model', '')}\n"
                    f"**Temperature:** {details.get('temperature', '')}\n"
                    f"**Max tokens:** {details.get('max_tokens', '')}\n"
                    f"**Session:** {self.agent.session.name if self.agent.session else 'none'}\n"
                    f"**Pipeline:** {self.agent.pipeline_mode}\n"
                    f"**Planning:** {'on' if self.agent.planning_enabled else 'off'}\n"
                )

            elif cmd in ("/history",):
                if self.agent.session:
                    msgs = self.agent.session.messages[-20:]
                    lines = []
                    for m in msgs:
                        role = m.get("role", "?")
                        content = str(m.get("content", ""))[:120]
                        lines.append(f"**{role}**: {content}")
                    result = "\n".join(lines) if lines else "No messages in session."
                else:
                    result = "No active session."

            elif cmd in ("/compact",):
                if self.agent.session:
                    await self.agent.compact_session(force=True, trigger="web_manual")
                    result = "Session compacted."
                else:
                    result = "No active session."

            elif cmd in ("/new",):
                name = args.strip() or None
                session = await self.agent.session_manager.create_session(
                    name=name or "web-session"
                )
                self.agent.session = session
                self.agent.refresh_session_runtime_flags()
                await self.agent.session_manager.set_last_active_session(session.id)
                self.agent.last_usage = self.agent._empty_usage()
                self.agent.last_context_window = {}
                result = f"New session created: **{session.name}** (`{session.id[:8]}`)"
                self._broadcast({"type": "session_info", **self._session_info()})

            elif cmd in ("/session",):
                if not args.strip():
                    info = self._session_info()
                    result = (
                        f"**Session:** {info.get('name', '?')}\n"
                        f"**ID:** {info.get('id', '?')}\n"
                        f"**Model:** {info.get('provider', '')}:{info.get('model', '')}\n"
                        f"**Messages:** {info.get('message_count', 0)}\n"
                        f"**Description:** {info.get('description', 'none')}"
                    )
                else:
                    result = await self._handle_session_subcommand(args.strip())

            elif cmd in ("/sessions",):
                sessions = await self.agent.session_manager.list_sessions(limit=20)
                if sessions:
                    lines = []
                    for i, s in enumerate(sessions, 1):
                        active = " (active)" if (self.agent.session and s.id == self.agent.session.id) else ""
                        desc = (s.metadata or {}).get("description", "")
                        desc_str = f" - {desc}" if desc else ""
                        lines.append(f"{i}. **{s.name}**{active}{desc_str}")
                    result = "\n".join(lines)
                else:
                    result = "No sessions found."

            elif cmd in ("/models",):
                models = self.agent.get_allowed_models()
                current = self.agent.get_runtime_model_details()
                lines = []
                for m in models:
                    active = " (active)" if m.get("model") == current.get("model") else ""
                    lines.append(f"- **{m.get('id', '?')}**: {m.get('provider', '')}:{m.get('model', '')}{active}")
                result = "\n".join(lines) if lines else "No models configured."

            elif cmd in ("/pipeline",):
                if args.strip():
                    mode = args.strip().lower()
                    if mode in ("loop", "contracts"):
                        self.agent.pipeline_mode = mode
                        if mode == "contracts":
                            self.agent.planning_enabled = True
                        result = f"Pipeline mode set to **{mode}**."
                    else:
                        result = "Invalid mode. Use `loop` or `contracts`."
                else:
                    result = f"Pipeline mode: **{self.agent.pipeline_mode}**"

            elif cmd in ("/planning",):
                if args.strip().lower() == "on":
                    self.agent.planning_enabled = True
                    self.agent.pipeline_mode = "contracts"
                    result = "Planning enabled (contracts mode)."
                elif args.strip().lower() == "off":
                    self.agent.planning_enabled = False
                    self.agent.pipeline_mode = "loop"
                    result = "Planning disabled (loop mode)."
                else:
                    result = f"Planning: **{'on' if self.agent.planning_enabled else 'off'}** (mode: {self.agent.pipeline_mode})"

            elif cmd in ("/skills",):
                skills = self.agent.discover_available_skills()
                if skills:
                    lines = []
                    for sk in skills:
                        name = getattr(sk, "name", "?")
                        desc = getattr(sk, "description", "")[:80]
                        lines.append(f"- **{name}**: {desc}")
                    result = "\n".join(lines)
                else:
                    result = "No skills available."

            elif cmd in ("/monitor",):
                result = "Monitor is always visible in the web UI. Use the monitor panel on the right."

            elif cmd in ("/exit", "/quit"):
                result = "Use Ctrl+C on the server terminal or close this browser tab."

            elif cmd in ("/approve",):
                result = await self._handle_approve_command(args.strip())

            elif cmd in ("/orchestrate",):
                if not args.strip():
                    result = "Usage: `/orchestrate <request>`"
                else:
                    # Delegate to _handle_chat which contains the orchestration logic.
                    await self._handle_chat(ws, raw)
                    return

            elif cmd in ("/orchestrate-execute",):
                # Execute a previously prepared orchestration graph.
                if not self._orchestrator:
                    result = "Orchestrator not available."
                else:
                    task_overrides = None
                    variable_values = None
                    if args.strip():
                        try:
                            parsed = json.loads(args.strip())
                            # New format: {task_overrides: ..., variable_values: ...}
                            if isinstance(parsed, dict) and (
                                "variable_values" in parsed or "task_overrides" in parsed
                            ):
                                task_overrides = parsed.get("task_overrides")
                                variable_values = parsed.get("variable_values")
                            else:
                                # Backward compat: bare dict is task overrides
                                task_overrides = parsed
                        except json.JSONDecodeError:
                            task_overrides = None
                    try:
                        response = await self._orchestrator.execute(
                            task_overrides, variable_values=variable_values,
                        )
                        self._broadcast({
                            "type": "chat_message",
                            "role": "assistant",
                            "content": response,
                        })
                    except Exception as e:
                        self._broadcast({
                            "type": "error",
                            "message": f"Orchestrator execute failed: {e}",
                        })
                    return

            else:
                # Try to process it as a chat message (the agent might understand it)
                result = f"Unknown command: `{cmd}`. Type `/help` for available commands."

        except Exception as e:
            result = f"Command error: {str(e)}"

        await self._send(ws, {
            "type": "command_result",
            "command": raw,
            "content": result,
        })

    async def _handle_session_subcommand(self, args: str) -> str:
        """Handle /session subcommands."""
        parts = args.split(None, 1)
        subcmd = parts[0].lower()
        subargs = parts[1].strip() if len(parts) > 1 else ""

        if subcmd in ("list",):
            sessions = await self.agent.session_manager.list_sessions(limit=20)
            lines = []
            for i, s in enumerate(sessions, 1):
                active = " (active)" if (self.agent.session and s.id == self.agent.session.id) else ""
                lines.append(f"{i}. **{s.name}**{active}")
            return "\n".join(lines) if lines else "No sessions."

        elif subcmd in ("switch", "load"):
            if not subargs:
                return "Usage: `/session switch <id|name|#N>`"
            session = await self.agent.session_manager.select_session(subargs)
            if session:
                self.agent.session = session
                await self.agent.session_manager.set_last_active_session(session.id)
                self.agent._sync_runtime_flags_from_session()
                self._broadcast({"type": "session_info", **self._session_info()})
                self._broadcast({"type": "session_switched"})
                return f"Switched to session **{session.name}**."
            return f"Session not found: `{subargs}`"

        elif subcmd in ("new",):
            name = subargs or "web-session"
            session = await self.agent.session_manager.create_session(name=name)
            self.agent.session = session
            self.agent.refresh_session_runtime_flags()
            await self.agent.session_manager.set_last_active_session(session.id)
            self.agent.last_usage = self.agent._empty_usage()
            self.agent.last_context_window = {}
            self._broadcast({"type": "session_info", **self._session_info()})
            self._broadcast({"type": "session_switched"})
            return f"Created and switched to session **{session.name}**."

        elif subcmd in ("rename",):
            if not subargs:
                return "Usage: `/session rename <new-name>`"
            if self.agent.session:
                self.agent.session.name = subargs
                await self.agent.session_manager.save_session(self.agent.session)
                self._broadcast({"type": "session_info", **self._session_info()})
                return f"Session renamed to **{subargs}**."
            return "No active session."

        elif subcmd in ("description",):
            if not subargs:
                return "Usage: `/session description <text>` or `/session description auto`"
            if self.agent.session:
                if subargs.lower() == "auto":
                    desc = await self.agent._auto_generate_session_description()
                    return f"Auto-generated description: {desc}"
                self.agent.session.metadata = self.agent.session.metadata or {}
                self.agent.session.metadata["description"] = subargs
                await self.agent.session_manager.save_session(self.agent.session)
                return f"Description set to: {subargs}"
            return "No active session."

        elif subcmd in ("model",):
            if not subargs:
                details = self.agent.get_runtime_model_details()
                return f"Active model: **{details.get('provider', '')}:{details.get('model', '')}**"
            await self.agent.set_session_model_by_selector(subargs, persist=True)
            details = self.agent.get_runtime_model_details()
            self._broadcast({"type": "session_info", **self._session_info()})
            return f"Model set to **{details.get('provider', '')}:{details.get('model', '')}**"

        elif subcmd in ("protect",):
            if self.agent.session:
                if subargs.lower() == "on":
                    self.agent.session.metadata = self.agent.session.metadata or {}
                    self.agent.session.metadata["memory_protection"] = True
                    await self.agent.session_manager.save_session(self.agent.session)
                    return "Memory protection enabled."
                elif subargs.lower() == "off":
                    self.agent.session.metadata = self.agent.session.metadata or {}
                    self.agent.session.metadata["memory_protection"] = False
                    await self.agent.session_manager.save_session(self.agent.session)
                    return "Memory protection disabled."
                return "Usage: `/session protect on|off`"
            return "No active session."

        return f"Unknown session subcommand: `{subcmd}`"

    def _format_help(self) -> str:
        """Format help text for the /help command."""
        categories: dict[str, list[dict[str, str]]] = {}
        for cmd in COMMANDS:
            cat = cmd["category"]
            categories.setdefault(cat, []).append(cmd)
        lines = ["## Captain Claw Commands\n"]
        for cat, cmds in categories.items():
            lines.append(f"### {cat}")
            for c in cmds:
                lines.append(f"- `{c['command']}` - {c['description']}")
            lines.append("")
        lines.append("**Tip:** Type `/` to see command suggestions. Press `Ctrl+K` for the command palette.")
        return "\n".join(lines)

    async def _handle_approve_command(self, args: str) -> str:
        """Handle /approve user telegram <token>."""
        parts = args.split()
        if len(parts) < 3 or parts[0].lower() != "user" or parts[1].lower() != "telegram":
            return "Usage: `/approve user telegram <token>`"
        token = parts[2].strip().upper()
        if not token:
            return "Usage: `/approve user telegram <token>`"
        self._tg_cleanup_expired()
        record = self._pending_telegram_pairings.get(token)
        if not isinstance(record, dict):
            return f"Telegram pairing token not found or expired: `{token}`"
        user_id = str(record.get("user_id", "")).strip()
        if not user_id:
            self._pending_telegram_pairings.pop(token, None)
            await self._tg_save_state()
            return f"Telegram pairing token invalid: `{token}`"

        self._approved_telegram_users[user_id] = {
            "user_id": int(record.get("user_id", 0) or 0),
            "chat_id": int(record.get("chat_id", 0) or 0),
            "username": str(record.get("username", "")).strip(),
            "first_name": str(record.get("first_name", "")).strip(),
            "approved_at": datetime.now(UTC).isoformat(),
            "token": token,
        }
        self._pending_telegram_pairings.pop(token, None)
        await self._tg_save_state()

        chat_id = int(self._approved_telegram_users[user_id].get("chat_id", 0) or 0)
        if chat_id and self._telegram_bridge:
            await self._tg_send(
                chat_id,
                "Pairing approved. You can now use Captain Claw.\nSend any text to chat.",
            )

        username = str(record.get("username", "")).strip()
        first_name = str(record.get("first_name", "")).strip()
        label = username or first_name or user_id
        return f"Approved Telegram user: **{label}**"

    # ── Telegram integration ────────────────────────────────────────

    async def _start_telegram(self) -> None:
        """Initialize and start Telegram polling + worker if configured."""
        if not self._telegram_enabled:
            return
        token = self.config.telegram.bot_token.strip()
        if not token:
            log.info("Telegram enabled but bot_token is empty; skipping in web mode.")
            return
        self._telegram_bridge = TelegramBridge(
            token=token, api_base_url=self.config.telegram.api_base_url,
        )
        # Load persisted pairing state
        if self.agent:
            self._approved_telegram_users = await self._tg_load_state(
                "telegram_approved_users"
            )
            self._pending_telegram_pairings = await self._tg_load_state(
                "telegram_pending_pairings"
            )
            self._tg_cleanup_expired()
            await self._tg_save_state()
        self._telegram_poll_task = asyncio.create_task(self._telegram_poll_loop())
        self._telegram_worker_task = asyncio.create_task(self._telegram_worker())
        log.info("Telegram bridge started in web mode (long polling).")
        print("  Telegram integration active (long polling).")

    async def _stop_telegram(self) -> None:
        """Gracefully stop Telegram tasks."""
        for task in (self._telegram_poll_task, self._telegram_worker_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._telegram_bridge:
            await self._telegram_bridge.close()

    async def _telegram_poll_loop(self) -> None:
        """Background Telegram long-polling loop."""
        assert self._telegram_bridge is not None
        poll_timeout = max(1, int(self.config.telegram.poll_timeout_seconds))
        while True:
            try:
                updates = await self._telegram_bridge.get_updates(
                    offset=self._telegram_offset, timeout=poll_timeout,
                )
                for update in updates:
                    next_offset = int(update.update_id) + 1
                    self._telegram_offset = (
                        next_offset
                        if self._telegram_offset is None
                        else max(self._telegram_offset, next_offset)
                    )
                    await self._telegram_queue.put(update)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Telegram poll error", error=str(exc))
                await asyncio.sleep(2.0)

    async def _telegram_worker(self) -> None:
        """Dispatch queued Telegram messages as concurrent tasks.

        Each message is spawned as its own task so that the busy-lock
        wait in ``_tg_process_with_typing`` does not block the queue
        for subsequent messages (matching the CLI/TUI behaviour).
        """
        while True:
            try:
                message = await self._telegram_queue.get()
                asyncio.create_task(self._handle_telegram_message(message))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Telegram worker error", error=str(exc))

    async def _handle_telegram_message(self, message: TelegramMessage) -> None:
        """Process a single Telegram message."""
        bridge = self._telegram_bridge
        if not bridge or not self.agent:
            return
        try:
            # Mark as read (business messages)
            if message.business_connection_id:
                try:
                    await bridge.read_business_message(
                        message.business_connection_id, message.chat_id, message.message_id,
                    )
                except Exception:
                    pass

            # User approval check
            user_id_key = str(message.user_id)
            if user_id_key not in self._approved_telegram_users:
                await self._tg_pair_unknown_user(message)
                return

            text = message.text.strip()
            if not text:
                return

            # Strip Telegram-style @BotName suffix from commands
            if text.startswith("/") and "@" in text.split()[0]:
                parts = text.split(None, 1)
                command_word = parts[0].split("@")[0]
                text = command_word if len(parts) == 1 else f"{command_word} {parts[1]}"

            # Handle /start and /help directly
            lowered = text.lower()
            if lowered == "/start" or lowered.startswith("/start "):
                await self._tg_send(
                    message.chat_id,
                    "Captain Claw connected (web mode).\nSend plain text to chat.",
                    reply_to_message_id=message.message_id,
                )
                return
            if lowered == "/help" or lowered.startswith("/help "):
                await self._tg_send(
                    message.chat_id,
                    "Send any text to chat with the current session.\n"
                    "Full command support is available in the Web UI.",
                    reply_to_message_id=message.message_id,
                )
                return

            # Handle all other slash commands
            if text.startswith("/"):
                result = await self._execute_telegram_command(text)
                if result is not None:
                    await self._tg_send(
                        message.chat_id,
                        result,
                        reply_to_message_id=message.message_id,
                    )
                    return

            # Notify web UI clients about the incoming Telegram message
            user_label = message.username or message.first_name or str(message.user_id)
            self._broadcast({
                "type": "chat_message",
                "role": "user",
                "content": f"[TG {user_label}] {text}",
            })

            # Process through agent with typing indicator
            await self._tg_process_with_typing(message.chat_id, text, message.message_id)

        except Exception as exc:
            log.error("Telegram message handler failed", error=str(exc))
            try:
                await self._tg_send(
                    message.chat_id,
                    f"Error while processing your request: {str(exc)}",
                    reply_to_message_id=message.message_id,
                )
            except Exception:
                pass

    async def _execute_telegram_command(self, raw: str) -> str | None:
        """Execute a slash command from Telegram, returning the response text.

        Returns ``None`` only when the command should fall through to the
        agent (e.g. ``/orchestrate``).
        """
        if not self.agent:
            return "Agent not ready."

        parts = raw.strip().split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        try:
            if cmd in ("/clear",):
                if self.agent.session:
                    if self.agent.is_session_memory_protected():
                        return "Session memory is protected. Disable with /session protect off."
                    self.agent.session.messages.clear()
                    await self.agent.session_manager.save_session(self.agent.session)
                    self.agent.last_usage = self.agent._empty_usage()
                    self.agent.last_context_window = {}
                    return "Session messages cleared."
                return "No active session."

            if cmd in ("/config",):
                details = self.agent.get_runtime_model_details()
                return (
                    f"Model: {details.get('provider', '')}:{details.get('model', '')}\n"
                    f"Session: {self.agent.session.name if self.agent.session else 'none'}\n"
                    f"Pipeline: {self.agent.pipeline_mode}"
                )

            if cmd in ("/history",):
                if self.agent.session:
                    msgs = self.agent.session.messages[-20:]
                    if not msgs:
                        return "Session history is empty."
                    lines = []
                    for i, m in enumerate(msgs, 1):
                        role = m.get("role", "?")
                        content = str(m.get("content", ""))[:150].replace("\n", " ")
                        lines.append(f"{i}. {role}: {content}")
                    return "\n".join(lines)
                return "No active session."

            if cmd in ("/compact",):
                if self.agent.session:
                    compacted, stats = await self.agent.compact_session(force=True, trigger="telegram")
                    if compacted:
                        return (
                            f"Session compacted ({int(stats.get('before_tokens', 0))} "
                            f"-> {int(stats.get('after_tokens', 0))} tokens)"
                        )
                    return f"Compaction skipped: {stats.get('reason', 'not_needed')}"
                return "No active session."

            if cmd in ("/new",):
                name = args.strip() or "default"
                session = await self.agent.session_manager.create_session(name=name)
                self.agent.session = session
                self.agent.refresh_session_runtime_flags()
                await self.agent.session_manager.set_last_active_session(session.id)
                self.agent.last_usage = self.agent._empty_usage()
                self.agent.last_context_window = {}
                self._broadcast({"type": "session_info", **self._session_info()})
                return f"New session: {session.name} ({session.id[:12]})"

            if cmd in ("/session",):
                if not args.strip():
                    if not self.agent.session:
                        return "No active session."
                    details = self.agent.get_runtime_model_details()
                    return (
                        f"Session: {self.agent.session.name}\n"
                        f"ID: {self.agent.session.id}\n"
                        f"Messages: {len(self.agent.session.messages)}\n"
                        f"Model: {details.get('provider')}/{details.get('model')}"
                    )
                return await self._handle_session_subcommand(args.strip())

            if cmd in ("/sessions",):
                sessions = await self.agent.session_manager.list_sessions(limit=20)
                if not sessions:
                    return "No sessions found."
                lines = ["Sessions:"]
                for i, s in enumerate(sessions, 1):
                    marker = "*" if (self.agent.session and s.id == self.agent.session.id) else " "
                    lines.append(f"{marker} [{i}] {s.name} ({s.id}) messages={len(s.messages)}")
                return "\n".join(lines)

            if cmd in ("/models",):
                models = self.agent.get_allowed_models()
                details = self.agent.get_runtime_model_details()
                lines = ["Allowed models:"]
                for i, m in enumerate(models, 1):
                    marker = ""
                    if (
                        str(m.get("provider", "")).strip() == str(details.get("provider", "")).strip()
                        and str(m.get("model", "")).strip() == str(details.get("model", "")).strip()
                    ):
                        marker = " *"
                    lines.append(f"[{i}] {m.get('id')} -> {m.get('provider')}/{m.get('model')}{marker}")
                return "\n".join(lines)

            if cmd in ("/pipeline",):
                if args.strip():
                    mode = args.strip().lower()
                    if mode in ("loop", "contracts"):
                        await self.agent.set_pipeline_mode(mode)
                        return f"Pipeline mode set to {mode}."
                    return "Invalid mode. Use /pipeline loop|contracts"
                return f"Pipeline mode: {self.agent.pipeline_mode}"

            if cmd in ("/planning",):
                if args.strip().lower() == "on":
                    await self.agent.set_pipeline_mode("contracts")
                    return "Pipeline mode set to contracts."
                if args.strip().lower() == "off":
                    await self.agent.set_pipeline_mode("loop")
                    return "Pipeline mode set to loop."
                return f"Planning: {'on' if self.agent.pipeline_mode == 'contracts' else 'off'}"

            if cmd in ("/skills",):
                skills = self.agent.list_user_invocable_skills()
                if not skills:
                    return "No user-invocable skills available."
                lines = ["Available skills:"]
                for s in skills:
                    lines.append(f"- /skill {s.name}")
                return "\n".join(lines)

            if cmd in ("/monitor",):
                return "Monitor is available in the Web UI."

            if cmd in ("/exit", "/quit"):
                return "/exit is only available on the server terminal."

            if cmd in ("/approve",):
                return await self._handle_approve_command(args.strip())

            if cmd in ("/orchestrate",):
                # Fall through to agent processing
                return None

            return f"Unknown command: {cmd}\nUse /help for available commands."

        except Exception as e:
            return f"Command error: {e}"

    async def _tg_process_with_typing(
        self, chat_id: int, text: str, reply_to_message_id: int | None = None,
    ) -> None:
        """Run agent.complete() with Telegram typing indicator, respecting busy state."""
        bridge = self._telegram_bridge
        if not bridge or not self.agent:
            return

        # If agent is already busy (web UI or another message), inform the user
        if self._busy:
            await self._tg_send(
                chat_id,
                "Agent is busy processing another request. Your message is queued…",
                reply_to_message_id=reply_to_message_id,
            )

        # Wait for exclusive agent access
        async with self._busy_lock:
            self._busy = True
            self._broadcast({"type": "status", "status": "thinking"})

            stop_typing = asyncio.Event()

            async def _typing_heartbeat() -> None:
                while not stop_typing.is_set():
                    try:
                        await bridge.send_chat_action(chat_id=chat_id, action="typing")
                    except Exception:
                        pass
                    try:
                        await asyncio.wait_for(stop_typing.wait(), timeout=4.0)
                    except TimeoutError:
                        continue

            heartbeat = asyncio.create_task(_typing_heartbeat())
            try:
                # Route /orchestrate requests to the orchestrator.
                stripped = text.strip()
                if stripped.lower().startswith("/orchestrate ") and self._orchestrator:
                    orchestrate_input = stripped[len("/orchestrate "):].strip()
                    if not orchestrate_input:
                        await self._tg_send(
                            chat_id, "Usage: /orchestrate <request>",
                            reply_to_message_id=reply_to_message_id,
                        )
                    else:
                        response = await self._orchestrator.orchestrate(orchestrate_input)
                        await self._tg_send(chat_id, response, reply_to_message_id=reply_to_message_id)
                        self._broadcast({
                            "type": "chat_message",
                            "role": "assistant",
                            "content": response,
                        })
                else:
                    response = await self.agent.complete(text)

                    # Send response to Telegram
                    await self._tg_send(chat_id, response, reply_to_message_id=reply_to_message_id)

                    # Broadcast to web UI
                    self._broadcast({
                        "type": "chat_message",
                        "role": "assistant",
                        "content": response,
                    })
                self._broadcast({
                    "type": "usage",
                    "last": self.agent.last_usage,
                    "total": self.agent.total_usage,
                })
                self._broadcast({
                    "type": "session_info",
                    **self._session_info(),
                })
            except Exception as exc:
                log.error("Telegram agent error", error=str(exc))
                await self._tg_send(
                    chat_id, f"Error: {str(exc)}", reply_to_message_id=reply_to_message_id,
                )
                self._broadcast({"type": "error", "message": f"Telegram error: {str(exc)}"})
            finally:
                stop_typing.set()
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass
                self._busy = False
                self._broadcast({"type": "status", "status": "ready"})

    # ── Telegram helpers ──────────────────────────────────────────────

    async def _tg_send(
        self, chat_id: int, text: str, *, reply_to_message_id: int | None = None,
    ) -> None:
        if self._telegram_bridge:
            try:
                await self._telegram_bridge.send_message(
                    chat_id=chat_id, text=text, reply_to_message_id=reply_to_message_id,
                )
            except Exception as exc:
                log.error("Telegram send failed", error=str(exc))

    async def _tg_load_state(self, key: str) -> dict[str, dict[str, object]]:
        if not self.agent:
            return {}
        raw = await self.agent.session_manager.get_app_state(key)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    async def _tg_save_state(self) -> None:
        if not self.agent:
            return
        await self.agent.session_manager.set_app_state(
            "telegram_approved_users",
            json.dumps(self._approved_telegram_users, ensure_ascii=True, sort_keys=True),
        )
        await self.agent.session_manager.set_app_state(
            "telegram_pending_pairings",
            json.dumps(self._pending_telegram_pairings, ensure_ascii=True, sort_keys=True),
        )

    def _tg_cleanup_expired(self) -> None:
        now_ts = datetime.now(UTC).timestamp()
        expired = []
        for token, payload in self._pending_telegram_pairings.items():
            expires_at_raw = str(payload.get("expires_at", "")).strip()
            if not expires_at_raw:
                continue
            try:
                expires_at = datetime.fromisoformat(
                    expires_at_raw.replace("Z", "+00:00")
                )
                if expires_at.timestamp() <= now_ts:
                    expired.append(token)
            except Exception:
                expired.append(token)
        for token in expired:
            self._pending_telegram_pairings.pop(token, None)

    def _tg_generate_pairing_token(self) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        while True:
            token = "".join(secrets.choice(alphabet) for _ in range(8))
            if token not in self._pending_telegram_pairings:
                return token

    async def _tg_pair_unknown_user(self, message: TelegramMessage) -> None:
        self._tg_cleanup_expired()
        user_id_key = str(message.user_id)
        if user_id_key in self._approved_telegram_users:
            return
        existing_token = ""
        for token, payload in self._pending_telegram_pairings.items():
            if str(payload.get("user_id", "")).strip() == str(message.user_id):
                existing_token = token
                break
        if not existing_token:
            existing_token = self._tg_generate_pairing_token()
            ttl_minutes = max(1, int(self.config.telegram.pairing_ttl_minutes))
            expires = datetime.now(UTC).timestamp() + ttl_minutes * 60
            expires_dt = datetime.fromtimestamp(expires, tz=UTC)
            self._pending_telegram_pairings[existing_token] = {
                "user_id": message.user_id,
                "chat_id": message.chat_id,
                "username": message.username,
                "first_name": message.first_name,
                "created_at": datetime.now(UTC).isoformat(),
                "expires_at": expires_dt.isoformat(),
            }
            await self._tg_save_state()

        await self._tg_send(
            message.chat_id,
            (
                "Pairing required.\n"
                f"Your pairing token: `{existing_token}`\n\n"
                "Ask the Captain Claw operator to approve you with:\n"
                f"/approve user telegram {existing_token}"
            ),
            reply_to_message_id=message.message_id,
        )
        # Notify web UI about the pairing request
        self._broadcast({
            "type": "chat_message",
            "role": "system",
            "content": (
                f"Telegram pairing request from "
                f"{message.username or message.first_name or message.user_id}. "
                f"Token: {existing_token}"
            ),
        })

    # ── REST handlers ────────────────────────────────────────────────

    async def list_instructions(self, request: web.Request) -> web.Response:
        """List instruction .md files, merging system and personal dirs."""
        seen: dict[str, dict] = {}

        # System (base) directory first
        if self._instructions_dir.is_dir():
            for f in sorted(self._instructions_dir.iterdir()):
                if f.suffix == ".md" and f.is_file():
                    seen[f.name] = {
                        "name": f.name,
                        "size": f.stat().st_size,
                        "overridden": False,
                    }

        # Personal overrides — update size and mark as overridden
        if self._instructions_personal_dir.is_dir():
            for f in sorted(self._instructions_personal_dir.iterdir()):
                if f.suffix == ".md" and f.is_file():
                    seen[f.name] = {
                        "name": f.name,
                        "size": f.stat().st_size,
                        "overridden": True,
                    }

        files = sorted(seen.values(), key=lambda x: x["name"])
        return web.json_response(files)

    async def get_instruction(self, request: web.Request) -> web.Response:
        """Read an instruction file (personal override wins over system)."""
        name = request.match_info["name"]
        # Sanitize: only allow .md files, no path traversal
        if ".." in name or "/" in name or not name.endswith(".md"):
            return web.json_response({"error": "Invalid file name"}, status=400)

        # Personal override takes precedence
        personal_path = self._instructions_personal_dir / name
        system_path = self._instructions_dir / name
        overridden = personal_path.is_file()
        path = personal_path if overridden else system_path

        if not path.is_file():
            return web.json_response({"error": "File not found"}, status=404)

        content = path.read_text(encoding="utf-8")
        return web.json_response({
            "name": name,
            "content": content,
            "overridden": overridden,
        })

    async def put_instruction(self, request: web.Request) -> web.Response:
        """Save instruction to the personal override directory."""
        name = request.match_info["name"]
        # Security: no path traversal, no subdirectories, .md only
        if ".." in name or "/" in name or "\\" in name or not name.endswith(".md"):
            return web.json_response({"error": "Invalid file name"}, status=400)
        path = self._instructions_personal_dir / name
        # Ensure the resolved path stays inside the personal directory
        try:
            path.resolve().relative_to(self._instructions_personal_dir.resolve())
        except ValueError:
            return web.json_response({"error": "Invalid path"}, status=400)
        body = await request.json()
        content = body.get("content", "")
        is_new = not path.exists()
        self._instructions_personal_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        # Clear instruction cache so changes take effect on next agent request
        if self.agent:
            self.agent.instructions._cache.pop(name, None)
        status = "created" if is_new else "saved"
        return web.json_response({
            "status": status,
            "name": name,
            "size": path.stat().st_size,
            "overridden": True,
        })

    async def revert_instruction(self, request: web.Request) -> web.Response:
        """Delete the personal override, reverting to the system default."""
        name = request.match_info["name"]
        if ".." in name or "/" in name or "\\" in name or not name.endswith(".md"):
            return web.json_response({"error": "Invalid file name"}, status=400)
        personal_path = self._instructions_personal_dir / name
        system_path = self._instructions_dir / name
        if not personal_path.is_file():
            return web.json_response({"error": "No personal override to revert"}, status=404)
        if not system_path.is_file():
            return web.json_response(
                {"error": "No system default exists — cannot revert"},
                status=400,
            )
        personal_path.unlink()
        # Clear instruction cache so the system default is loaded next
        if self.agent:
            self.agent.instructions._cache.pop(name, None)
        content = system_path.read_text(encoding="utf-8")
        return web.json_response({
            "status": "reverted",
            "name": name,
            "content": content,
            "size": system_path.stat().st_size,
            "overridden": False,
        })

    async def get_config_summary(self, request: web.Request) -> web.Response:
        """Return a safe config summary (no secrets)."""
        cfg = get_config()
        details = {}
        if self.agent:
            details = self.agent.get_runtime_model_details()
        return web.json_response({
            "model": {
                "provider": details.get("provider", cfg.model.provider),
                "model": details.get("model", cfg.model.model),
                "temperature": details.get("temperature", cfg.model.temperature),
                "max_tokens": details.get("max_tokens", cfg.model.max_tokens),
            },
            "context": {
                "max_tokens": cfg.context.max_tokens,
                "compaction_threshold": cfg.context.compaction_threshold,
            },
            "tools": cfg.tools.enabled,
            "guards": {
                "input": cfg.guards.input.enabled,
                "output": cfg.guards.output.enabled,
                "script_tool": cfg.guards.script_tool.enabled,
            },
        })

    async def list_sessions_api(self, request: web.Request) -> web.Response:
        """List sessions via REST."""
        sm = get_session_manager()
        sessions = await sm.list_sessions(limit=20)
        result = []
        for s in sessions:
            result.append({
                "id": s.id,
                "name": s.name,
                "message_count": len(s.messages),
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "description": (s.metadata or {}).get("description", ""),
            })
        return web.json_response(result)

    async def get_commands_api(self, request: web.Request) -> web.Response:
        """Return the available commands list."""
        return web.json_response(COMMANDS)

    # ── Google OAuth ────────────────────────────────────────────────

    async def _auth_google_login(self, request: web.Request) -> web.Response:
        """Start the Google OAuth2 authorization flow."""
        cfg = get_config()
        oauth = cfg.google_oauth
        if not oauth.enabled or not oauth.client_id:
            return web.json_response(
                {"error": "Google OAuth not configured"}, status=400
            )

        # Purge stale PKCE states (older than 10 minutes).
        cutoff = time.time() - 600
        self._pending_oauth = {
            k: v for k, v in self._pending_oauth.items()
            if v.get("ts", 0) > cutoff
        }

        state = secrets.token_urlsafe(32)
        verifier, challenge = generate_pkce_pair()
        self._pending_oauth[state] = {
            "verifier": verifier,
            "ts": time.time(),
        }

        redirect_uri = (
            f"http://localhost:{cfg.web.port}/auth/google/callback"
        )
        auth_url = build_authorization_url(
            client_id=oauth.client_id,
            redirect_uri=redirect_uri,
            scopes=oauth.scopes,
            state=state,
            code_challenge=challenge,
        )
        raise web.HTTPFound(auth_url)

    async def _auth_google_callback(self, request: web.Request) -> web.Response:
        """Handle the OAuth2 callback from Google."""
        error = request.query.get("error")
        if error:
            desc = request.query.get("error_description", error)
            return web.Response(
                text=f"<html><body><h2>OAuth Error</h2><p>{desc}</p>"
                     f"<p><a href='/'>Back to home</a></p></body></html>",
                content_type="text/html",
            )

        code = request.query.get("code", "")
        state = request.query.get("state", "")
        if not code or not state:
            return web.Response(text="Missing code or state", status=400)

        # Validate state and retrieve PKCE verifier.
        pending = self._pending_oauth.pop(state, None)
        if not pending:
            return web.Response(
                text="<html><body><h2>Invalid or expired state</h2>"
                     "<p>Please try again.</p>"
                     "<p><a href='/'>Back to home</a></p></body></html>",
                content_type="text/html",
                status=400,
            )

        cfg = get_config()
        oauth = cfg.google_oauth
        redirect_uri = f"http://localhost:{cfg.web.port}/auth/google/callback"

        try:
            tokens = await exchange_code_for_tokens(
                code=code,
                client_id=oauth.client_id,
                client_secret=oauth.client_secret,
                redirect_uri=redirect_uri,
                code_verifier=pending["verifier"],
            )
        except Exception as exc:
            log.error("Google OAuth token exchange failed: %s", exc)
            return web.Response(
                text=f"<html><body><h2>Token Exchange Failed</h2>"
                     f"<p>{exc}</p>"
                     f"<p><a href='/'>Back to home</a></p></body></html>",
                content_type="text/html",
                status=500,
            )

        # Fetch user info.
        try:
            user = await fetch_user_info(tokens.access_token)
        except Exception as exc:
            log.warning("Failed to fetch Google user info: %s", exc)
            user = {}

        # Store tokens and user info.
        if self._oauth_manager:
            await self._oauth_manager.store_tokens(tokens)
            if user:
                await self._oauth_manager.store_user_info(user)
            await self._inject_oauth_into_provider()

        email = user.get("email", "your Google account")
        return web.Response(
            text=(
                "<!DOCTYPE html><html><head>"
                "<meta charset='utf-8'>"
                "<meta http-equiv='refresh' content='2;url=/'>"
                "<title>Connected</title>"
                "<style>"
                "body{background:#0d1117;color:#e6edf3;font-family:sans-serif;"
                "display:flex;align-items:center;justify-content:center;"
                "min-height:100vh;margin:0;}"
                ".box{text-align:center;}"
                ".box h2{margin-bottom:8px;}"
                ".box p{color:#8b949e;}"
                "</style>"
                "</head><body>"
                f"<div class='box'><h2>Connected as {email}</h2>"
                "<p>Redirecting to home page...</p></div>"
                "</body></html>"
            ),
            content_type="text/html",
        )

    async def _auth_google_status(self, request: web.Request) -> web.Response:
        """Return Google OAuth connection status as JSON."""
        cfg = get_config()
        if not cfg.google_oauth.enabled or not cfg.google_oauth.client_id:
            return web.json_response({"connected": False, "enabled": False})

        if not self._oauth_manager:
            return web.json_response({"connected": False, "enabled": True})

        connected = await self._oauth_manager.is_connected()
        user = None
        if connected:
            user = await self._oauth_manager.get_user_info()

        return web.json_response({
            "connected": connected,
            "enabled": True,
            "user": user,
        })

    async def _auth_google_logout(self, request: web.Request) -> web.Response:
        """Revoke Google OAuth tokens and disconnect."""
        if self._oauth_manager:
            await self._oauth_manager.disconnect()
            # Clear vertex credentials from active provider.
            self._clear_oauth_from_provider()
        return web.json_response({"disconnected": True})

    async def _inject_oauth_into_provider(self) -> None:
        """Inject stored Google OAuth credentials into the Gemini provider."""
        if not self._oauth_manager:
            return
        creds = await self._oauth_manager.get_vertex_credentials()
        if not creds:
            return

        cfg = get_config()
        oauth = cfg.google_oauth

        from captain_claw.llm import LiteLLMProvider

        # Inject into the main agent's provider.
        if self.agent and isinstance(self.agent.provider, LiteLLMProvider):
            if self.agent.provider.provider == "gemini":
                self.agent.provider.set_vertex_credentials(
                    credentials=creds,
                    project=oauth.project_id,
                    location=oauth.location,
                )
                log.info("Google OAuth credentials injected into Gemini provider.")

    def _clear_oauth_from_provider(self) -> None:
        """Remove vertex credentials from the active provider."""
        from captain_claw.llm import LiteLLMProvider

        if self.agent and isinstance(self.agent.provider, LiteLLMProvider):
            self.agent.provider.clear_vertex_credentials()

    # ── Cron management API ──────────────────────────────────────────

    def _get_web_runtime_context(self) -> "RuntimeContext":
        """Create a lightweight RuntimeContext for web-mode cron execution."""
        from captain_claw.runtime_context import RuntimeContext

        class _WebCronUI:
            """Minimal TerminalUI stand-in for web-mode cron execution."""

            def print_message(self, role: str, content: str) -> None:
                pass

            def print_error(self, error: str) -> None:
                log.error("WebCronUI error", error=error)

            def set_runtime_status(self, status: str) -> None:
                pass

            def append_tool_output(
                self, tool_name: str, arguments: object, output: str
            ) -> None:
                pass

            def confirm(self, message: str) -> bool:
                return True

        if not self.agent:
            raise RuntimeError("Agent not initialized")
        return RuntimeContext(agent=self.agent, ui=_WebCronUI())  # type: ignore[arg-type]

    async def _list_cron_jobs(self, request: web.Request) -> web.Response:
        """GET /api/cron/jobs — list all cron jobs."""
        if not self.agent:
            return web.json_response({"error": "Agent not initialized"}, status=503)
        sm = self.agent.session_manager
        jobs = await sm.list_cron_jobs(limit=200, active_only=False)
        result = []
        for j in jobs:
            result.append({
                "id": j.id,
                "kind": j.kind,
                "payload": j.payload,
                "schedule": j.schedule,
                "session_id": j.session_id,
                "enabled": j.enabled,
                "created_at": j.created_at,
                "updated_at": j.updated_at,
                "last_run_at": j.last_run_at,
                "next_run_at": j.next_run_at,
                "last_status": j.last_status,
                "last_error": j.last_error,
            })
        return web.json_response(result)

    async def _create_cron_job(self, request: web.Request) -> web.Response:
        """POST /api/cron/jobs — create a new cron job."""
        if not self.agent:
            return web.json_response({"error": "Agent not initialized"}, status=503)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        kind = str(body.get("kind", "")).strip().lower()
        if kind not in ("prompt", "script", "tool", "orchestrate"):
            return web.json_response({"error": "Invalid kind"}, status=400)

        schedule = body.get("schedule")
        if not isinstance(schedule, dict) or "type" not in schedule:
            return web.json_response({"error": "Invalid schedule"}, status=400)

        payload = body.get("payload")
        if not isinstance(payload, dict):
            return web.json_response({"error": "Invalid payload"}, status=400)

        session_id = str(body.get("session_id", "")).strip()
        if not session_id:
            return web.json_response({"error": "session_id is required"}, status=400)

        from captain_claw.cron import compute_next_run, schedule_to_text, to_utc_iso

        # Add human-readable _text field to schedule.
        schedule["_text"] = schedule_to_text(schedule)

        try:
            next_run = to_utc_iso(compute_next_run(schedule))
        except Exception as e:
            return web.json_response({"error": f"Bad schedule: {e}"}, status=400)

        sm = self.agent.session_manager
        job = await sm.create_cron_job(
            kind=kind,
            payload=payload,
            schedule=schedule,
            session_id=session_id,
            next_run_at=next_run,
        )
        return web.json_response({
            "ok": True,
            "id": job.id,
            "next_run_at": job.next_run_at,
        })

    async def _run_cron_job(self, request: web.Request) -> web.Response:
        """POST /api/cron/jobs/{id}/run — execute a job immediately."""
        if not self.agent:
            return web.json_response({"error": "Agent not initialized"}, status=503)

        job_id = request.match_info.get("id", "")
        sm = self.agent.session_manager
        job = await sm.load_cron_job(job_id)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)

        from captain_claw.cron_dispatch import execute_cron_job

        try:
            ctx = self._get_web_runtime_context()
            asyncio.create_task(execute_cron_job(ctx, job, trigger="manual"))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

        return web.json_response({"ok": True, "status": "started"})

    async def _pause_cron_job(self, request: web.Request) -> web.Response:
        """POST /api/cron/jobs/{id}/pause — disable a job."""
        if not self.agent:
            return web.json_response({"error": "Agent not initialized"}, status=503)

        job_id = request.match_info.get("id", "")
        sm = self.agent.session_manager
        ok = await sm.update_cron_job(job_id, enabled=False, last_status="paused")
        if not ok:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response({"ok": True})

    async def _resume_cron_job(self, request: web.Request) -> web.Response:
        """POST /api/cron/jobs/{id}/resume — re-enable a paused job."""
        if not self.agent:
            return web.json_response({"error": "Agent not initialized"}, status=503)

        job_id = request.match_info.get("id", "")
        sm = self.agent.session_manager
        job = await sm.load_cron_job(job_id)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)

        from captain_claw.cron import compute_next_run, to_utc_iso

        next_run = to_utc_iso(compute_next_run(job.schedule))
        await sm.update_cron_job(
            job_id, enabled=True, last_status="pending", next_run_at=next_run,
        )
        return web.json_response({"ok": True, "next_run_at": next_run})

    async def _update_cron_job_payload(self, request: web.Request) -> web.Response:
        """PATCH /api/cron/jobs/{id} — update job payload (e.g. variable values)."""
        if not self.agent:
            return web.json_response({"error": "Agent not initialized"}, status=503)

        job_id = request.match_info.get("id", "")
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        sm = self.agent.session_manager
        new_payload = body.get("payload")
        if new_payload is None:
            return web.json_response({"error": "Missing payload"}, status=400)

        ok = await sm.update_cron_job(job_id, payload=new_payload)
        if not ok:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response({"ok": True})

    async def _delete_cron_job(self, request: web.Request) -> web.Response:
        """DELETE /api/cron/jobs/{id} — remove a job."""
        if not self.agent:
            return web.json_response({"error": "Agent not initialized"}, status=503)

        job_id = request.match_info.get("id", "")
        sm = self.agent.session_manager
        ok = await sm.delete_cron_job(job_id)
        if not ok:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response({"ok": True})

    async def _get_cron_job_history(self, request: web.Request) -> web.Response:
        """GET /api/cron/jobs/{id}/history — get job execution history."""
        if not self.agent:
            return web.json_response({"error": "Agent not initialized"}, status=503)

        job_id = request.match_info.get("id", "")
        sm = self.agent.session_manager
        job = await sm.load_cron_job(job_id)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response(
            {"chat_history": job.chat_history, "monitor_history": job.monitor_history},
            dumps=lambda obj: json.dumps(obj, default=str),
        )

    async def _serve_cron(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "cron.html")

    # ── Workflow browser API ──────────────────────────────────────────

    def _workflows_dir(self) -> Path:
        """Return the workflows directory (same as SessionOrchestrator)."""
        cfg = get_config()
        ws = cfg.resolved_workspace_path()
        d = ws / "workflows"
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def _list_workflow_outputs(self, request: web.Request) -> web.Response:
        """GET /api/workflow-browser — list workflows with their outputs.

        Returns a list of workflow objects, each with:
        - name: workflow name (stem of the .json file)
        - outputs: list of {filename, timestamp, size} for each -output-*.md
        """
        d = self._workflows_dir()
        # Collect JSON workflow files.
        workflows: dict[str, dict[str, Any]] = {}
        for p in sorted(d.glob("*.json")):
            name = p.stem
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                wf_name = data.get("workflow_name", name)
                user_input = data.get("user_input", "")
                task_count = len(data.get("tasks", []))
            except Exception:
                wf_name = name
                user_input = ""
                task_count = 0
            workflows[name] = {
                "name": wf_name,
                "filename": name,
                "user_input": user_input,
                "task_count": task_count,
                "outputs": [],
            }

        # Collect output .md files and match to workflows.
        for p in sorted(d.glob("*-output-*.md")):
            fname = p.name
            stem = p.stem  # e.g. "fetch-news-output-20260221-083216"
            # Find the matching workflow by checking if the stem starts
            # with a known workflow filename.
            matched = False
            for wf_key in workflows:
                if stem.startswith(wf_key + "-output-"):
                    # Extract timestamp portion.
                    ts_part = stem[len(wf_key) + len("-output-"):]
                    try:
                        stat = p.stat()
                        size = stat.st_size
                    except Exception:
                        size = 0
                    workflows[wf_key]["outputs"].append({
                        "filename": fname,
                        "timestamp": ts_part,
                        "size": size,
                    })
                    matched = True
                    break

            if not matched:
                # Orphan output — create a virtual entry.
                # Try to infer the workflow name from the filename.
                idx = stem.rfind("-output-")
                if idx > 0:
                    inferred_key = stem[:idx]
                    ts_part = stem[idx + len("-output-"):]
                else:
                    inferred_key = stem
                    ts_part = ""
                if inferred_key not in workflows:
                    workflows[inferred_key] = {
                        "name": inferred_key,
                        "filename": inferred_key,
                        "user_input": "",
                        "task_count": 0,
                        "outputs": [],
                    }
                try:
                    size = p.stat().st_size
                except Exception:
                    size = 0
                workflows[inferred_key]["outputs"].append({
                    "filename": fname,
                    "timestamp": ts_part,
                    "size": size,
                })

        # Sort outputs newest-first within each workflow.
        for wf in workflows.values():
            wf["outputs"].sort(key=lambda o: o.get("timestamp", ""), reverse=True)

        result = sorted(workflows.values(), key=lambda w: w["name"])
        return web.json_response(result)

    async def _get_workflow_output(self, request: web.Request) -> web.Response:
        """GET /api/workflow-browser/output/{filename} — read an output .md file."""
        filename = request.match_info.get("filename", "")
        if not filename or ".." in filename or "/" in filename:
            return web.json_response({"error": "Invalid filename"}, status=400)

        d = self._workflows_dir()
        path = d / filename
        if not path.is_file():
            return web.json_response({"error": "File not found"}, status=404)

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

        return web.json_response({"filename": filename, "content": content})

    async def _serve_workflows(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "workflows.html")

    # ── App setup ────────────────────────────────────────────────────

    def create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_get("/api/instructions", self.list_instructions)
        app.router.add_get("/api/instructions/{name}", self.get_instruction)
        app.router.add_put("/api/instructions/{name}", self.put_instruction)
        app.router.add_delete("/api/instructions/{name}", self.revert_instruction)
        app.router.add_get("/api/config", self.get_config_summary)
        app.router.add_get("/api/sessions", self.list_sessions_api)
        app.router.add_get("/api/commands", self.get_commands_api)
        app.router.add_get("/api/orchestrator/status", self._get_orchestrator_status)
        app.router.add_post("/api/orchestrator/reset", self._reset_orchestrator)
        app.router.add_get("/api/orchestrator/skills", self._get_orchestrator_skills)
        app.router.add_post("/api/orchestrator/rephrase", self._rephrase_orchestrator_input)
        app.router.add_post("/api/orchestrator/task/edit", self._edit_orchestrator_task)
        app.router.add_post("/api/orchestrator/task/update", self._update_orchestrator_task)
        app.router.add_post("/api/orchestrator/task/restart", self._restart_orchestrator_task)
        app.router.add_post("/api/orchestrator/task/pause", self._pause_orchestrator_task)
        app.router.add_post("/api/orchestrator/task/resume", self._resume_orchestrator_task)
        app.router.add_post("/api/orchestrator/task/postpone", self._postpone_orchestrator_task)
        # Orchestrator prepare/execute/workflow endpoints
        app.router.add_post("/api/orchestrator/prepare", self._prepare_orchestrator)
        app.router.add_get("/api/orchestrator/sessions", self._get_orchestrator_sessions)
        app.router.add_get("/api/orchestrator/models", self._get_orchestrator_models)
        app.router.add_get("/api/orchestrator/workflows", self._list_workflows)
        app.router.add_post("/api/orchestrator/workflows/save", self._save_workflow)
        app.router.add_post("/api/orchestrator/workflows/load", self._load_workflow)
        app.router.add_delete("/api/orchestrator/workflows/{name}", self._delete_workflow)
        # Cron management API routes
        app.router.add_get("/api/cron/jobs", self._list_cron_jobs)
        app.router.add_post("/api/cron/jobs", self._create_cron_job)
        app.router.add_post("/api/cron/jobs/{id}/run", self._run_cron_job)
        app.router.add_post("/api/cron/jobs/{id}/pause", self._pause_cron_job)
        app.router.add_post("/api/cron/jobs/{id}/resume", self._resume_cron_job)
        app.router.add_patch("/api/cron/jobs/{id}", self._update_cron_job_payload)
        app.router.add_delete("/api/cron/jobs/{id}", self._delete_cron_job)
        app.router.add_get("/api/cron/jobs/{id}/history", self._get_cron_job_history)
        # Workflow browser API routes
        app.router.add_get("/api/workflow-browser", self._list_workflow_outputs)
        app.router.add_get("/api/workflow-browser/output/{filename}", self._get_workflow_output)
        # OpenAI-compatible API proxy routes
        if self.config.web.api_enabled and self._api_pool:
            app.router.add_post("/v1/chat/completions", self._api_chat_completions)
            app.router.add_get("/v1/models", self._api_list_models)
        # Google OAuth routes
        app.router.add_get("/auth/google/login", self._auth_google_login)
        app.router.add_get("/auth/google/callback", self._auth_google_callback)
        app.router.add_get("/auth/google/status", self._auth_google_status)
        app.router.add_post("/auth/google/logout", self._auth_google_logout)
        # Static files (serve index.html at /)
        if STATIC_DIR.is_dir():
            app.router.add_static("/static/", STATIC_DIR, show_index=False)
            app.router.add_get("/", self._serve_home)
            app.router.add_get("/chat", self._serve_chat)
            app.router.add_get("/orchestrator", self._serve_orchestrator)
            app.router.add_get("/instructions", self._serve_instructions)
            app.router.add_get("/cron", self._serve_cron)
            app.router.add_get("/workflows", self._serve_workflows)
            app.router.add_get("/favicon.ico", self._serve_favicon)
        return app

    async def _serve_home(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "home.html")

    async def _serve_chat(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "index.html")

    async def _serve_favicon(self, request: web.Request) -> web.Response:
        # Return a simple empty response if no favicon exists
        favicon = STATIC_DIR / "favicon.svg"
        if favicon.is_file():
            return web.FileResponse(favicon)
        return web.Response(status=204)

    async def _serve_orchestrator(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "orchestrator.html")

    async def _serve_instructions(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "instructions.html")

    async def _get_orchestrator_status(self, request: web.Request) -> web.Response:
        """REST endpoint: current orchestrator graph state."""
        if not self._orchestrator:
            return web.json_response({"status": None})
        status = self._orchestrator.get_status()
        return web.json_response({"status": status}, dumps=lambda obj: json.dumps(obj, default=str))

    async def _reset_orchestrator(self, request: web.Request) -> web.Response:
        """POST /api/orchestrator/reset — cancel work and reset to idle."""
        if not self._orchestrator:
            return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
        try:
            await self._orchestrator.reset()
        except Exception as e:
            log.error("Orchestrator reset error", error=str(e))
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        return web.json_response({"ok": True})

    async def _get_orchestrator_skills(self, request: web.Request) -> web.Response:
        """REST endpoint: list available skills for orchestrator workers."""
        if not self.agent:
            return web.json_response({"skills": []})
        try:
            commands = self.agent.list_user_invocable_skills()
            skills = [
                {"name": cmd.name, "skill_name": cmd.skill_name, "description": cmd.description}
                for cmd in commands
            ]
        except Exception:
            skills = []

        # Also expose available tools as a separate list.
        try:
            tool_names = self.agent.tools.list_tools()
        except Exception:
            tool_names = []

        return web.json_response({"skills": skills, "tools": tool_names})

    async def _rephrase_orchestrator_input(self, request: web.Request) -> web.Response:
        """Rephrase a casual user request into a structured orchestrator prompt."""
        from captain_claw.llm import Message

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        user_input = str(body.get("input", "")).strip()
        if not user_input:
            return web.json_response({"error": "Empty input"}, status=400)

        # Use shared provider (same model as main agent).
        provider = self.agent.provider if self.agent else None
        if provider is None:
            from captain_claw.llm import get_provider
            provider = get_provider()

        loader = InstructionLoader()
        prompt = loader.render(
            "orchestrator_rephrase_prompt.md",
            user_input=user_input,
        )

        try:
            import asyncio as _asyncio

            response = await _asyncio.wait_for(
                provider.complete(
                    messages=[Message(role="user", content=prompt)],
                    tools=None,
                    max_tokens=2000,
                ),
                timeout=60.0,
            )
            rephrased = str(getattr(response, "content", "") or "").strip()
            if not rephrased:
                rephrased = user_input  # fallback to original
        except Exception as e:
            log.error("Rephrase failed", error=str(e))
            rephrased = user_input  # fallback to original

        return web.json_response({"rephrased": rephrased, "original": user_input})

    # ------------------------------------------------------------------
    # Orchestrator task control endpoints
    # ------------------------------------------------------------------

    async def _edit_orchestrator_task(self, request: web.Request) -> web.Response:
        """Put a task into edit mode."""
        if not self._orchestrator:
            return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
        task_id = str(body.get("task_id", "")).strip()
        if not task_id:
            return web.json_response({"ok": False, "error": "Missing task_id"}, status=400)
        result = await self._orchestrator.edit_task(task_id)
        return web.json_response(result)

    async def _update_orchestrator_task(self, request: web.Request) -> web.Response:
        """Update task instructions (description)."""
        if not self._orchestrator:
            return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
        task_id = str(body.get("task_id", "")).strip()
        description = str(body.get("description", ""))
        if not task_id:
            return web.json_response({"ok": False, "error": "Missing task_id"}, status=400)
        result = await self._orchestrator.update_task(task_id, description)
        return web.json_response(result)

    async def _restart_orchestrator_task(self, request: web.Request) -> web.Response:
        """Restart a failed/completed/paused task."""
        if not self._orchestrator:
            return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
        task_id = str(body.get("task_id", "")).strip()
        if not task_id:
            return web.json_response({"ok": False, "error": "Missing task_id"}, status=400)
        result = await self._orchestrator.restart_task(task_id)
        return web.json_response(result)

    async def _pause_orchestrator_task(self, request: web.Request) -> web.Response:
        """Pause a running task."""
        if not self._orchestrator:
            return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
        task_id = str(body.get("task_id", "")).strip()
        if not task_id:
            return web.json_response({"ok": False, "error": "Missing task_id"}, status=400)
        result = await self._orchestrator.pause_task(task_id)
        return web.json_response(result)

    async def _resume_orchestrator_task(self, request: web.Request) -> web.Response:
        """Resume a paused/editing task."""
        if not self._orchestrator:
            return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
        task_id = str(body.get("task_id", "")).strip()
        if not task_id:
            return web.json_response({"ok": False, "error": "Missing task_id"}, status=400)
        result = await self._orchestrator.resume_task(task_id)
        return web.json_response(result)

    async def _postpone_orchestrator_task(self, request: web.Request) -> web.Response:
        """Postpone a task's timeout warning, granting another timeout period."""
        if not self._orchestrator:
            return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
        task_id = str(body.get("task_id", "")).strip()
        if not task_id:
            return web.json_response({"ok": False, "error": "Missing task_id"}, status=400)
        result = await self._orchestrator.postpone_task(task_id)
        return web.json_response(result)

    # ── Orchestrator: prepare / workflows / sessions / models ───────

    async def _prepare_orchestrator(self, request: web.Request) -> web.Response:
        """Decompose a request into tasks without executing (preview)."""
        if not self._orchestrator:
            return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
        user_input = str(body.get("input", "")).strip()
        if not user_input:
            return web.json_response({"ok": False, "error": "Missing input"}, status=400)
        result = await self._orchestrator.prepare(user_input)
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, default=str))

    async def _get_orchestrator_sessions(self, request: web.Request) -> web.Response:
        """List sessions for per-task session selection."""
        if not self.agent:
            return web.json_response({"sessions": []})
        try:
            sessions = await self.agent.session_manager.list_sessions(limit=30)
            result = [
                {"id": s.id, "name": s.name}
                for s in sessions
            ]
        except Exception:
            result = []
        return web.json_response({"sessions": result})

    async def _get_orchestrator_models(self, request: web.Request) -> web.Response:
        """List allowed models for per-task model selection."""
        if not self.agent:
            return web.json_response({"models": []})
        try:
            models = self.agent.get_allowed_models()
        except Exception:
            models = []
        return web.json_response({"models": models})

    async def _list_workflows(self, request: web.Request) -> web.Response:
        """List saved workflows."""
        if not self._orchestrator:
            return web.json_response({"workflows": []})
        workflows = await self._orchestrator.list_workflows()
        return web.json_response({"workflows": workflows})

    async def _save_workflow(self, request: web.Request) -> web.Response:
        """Save the current graph as a workflow."""
        if not self._orchestrator:
            return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
        name = str(body.get("name", "")).strip() or None
        task_overrides = body.get("task_overrides") or None
        result = await self._orchestrator.save_workflow(name, task_overrides=task_overrides)
        return web.json_response(result)

    async def _load_workflow(self, request: web.Request) -> web.Response:
        """Load a saved workflow for preview."""
        if not self._orchestrator:
            return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
        name = str(body.get("name", "")).strip()
        if not name:
            return web.json_response({"ok": False, "error": "Missing name"}, status=400)
        result = await self._orchestrator.load_workflow(name)
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, default=str))

    async def _delete_workflow(self, request: web.Request) -> web.Response:
        """Delete a saved workflow."""
        if not self._orchestrator:
            return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
        name = request.match_info.get("name", "").strip()
        if not name:
            return web.json_response({"ok": False, "error": "Missing name"}, status=400)
        result = await self._orchestrator.delete_workflow(name)
        return web.json_response(result)

    # ── OpenAI-compatible API proxy ──────────────────────────────────

    @staticmethod
    def _extract_api_session_id(request: web.Request) -> str | None:
        """Extract session ID from ``Authorization: Bearer <session_id>``."""
        auth = request.headers.get("Authorization", "").strip()
        if not auth.lower().startswith("bearer "):
            return None
        token = auth[7:].strip()
        return token or None

    @staticmethod
    def _build_chat_completion_response(
        content: str,
        model: str,
        usage: dict[str, int],
        completion_id: str | None = None,
    ) -> dict[str, Any]:
        """Build an OpenAI-compatible chat completion response."""
        return {
            "id": completion_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

    async def _write_sse_streaming_response(
        self,
        request: web.Request,
        content: str,
        model: str,
        usage: dict[str, int],
        completion_id: str,
    ) -> web.StreamResponse:
        """Stream a completed response as Server-Sent Events.

        Chunks the response at word boundaries for a natural streaming feel.
        """
        created = int(time.time())
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await resp.prepare(request)

        async def _sse(data: str) -> None:
            await resp.write(f"data: {data}\n\n".encode("utf-8"))

        # First chunk: role delta.
        await _sse(json.dumps({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }))

        # Content chunks (~5 words each).
        words = content.split(" ")
        chunk_size = 5
        for i in range(0, len(words), chunk_size):
            text_chunk = " ".join(words[i : i + chunk_size])
            if i > 0:
                text_chunk = " " + text_chunk
            await _sse(json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}],
            }))

        # Final chunk: finish_reason + usage.
        await _sse(json.dumps({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }))

        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def _api_chat_completions(self, request: web.Request) -> web.Response | web.StreamResponse:
        """``POST /v1/chat/completions`` — OpenAI-compatible endpoint.

        The Bearer token in the Authorization header is used as the
        Captain Claw session ID.  Only the last user message is extracted
        from the ``messages`` array — the agent manages its own history.
        """
        if not self._api_pool:
            return web.json_response(
                {"error": {"message": "API proxy is disabled", "type": "server_error", "code": "api_disabled"}},
                status=503,
            )

        # Auth — Bearer token = session ID.
        session_id = self._extract_api_session_id(request)
        if not session_id:
            return web.json_response(
                {"error": {"message": "Missing or invalid Authorization header. Expected: Bearer <session_id>", "type": "invalid_request_error", "code": "missing_api_key"}},
                status=401,
            )

        # Parse body.
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error", "code": "invalid_json"}},
                status=400,
            )

        messages = body.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return web.json_response(
                {"error": {"message": "messages array is required and must not be empty", "type": "invalid_request_error", "code": "invalid_messages"}},
                status=400,
            )

        # Extract the last user message.
        user_message: str | None = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_message = content.strip()
                elif isinstance(content, list):
                    # Multi-part content (e.g. text + image).
                    parts = [
                        str(p.get("text", ""))
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    user_message = " ".join(parts).strip()
                break

        if not user_message:
            return web.json_response(
                {"error": {"message": "No user message found in messages array", "type": "invalid_request_error", "code": "no_user_message"}},
                status=400,
            )

        stream = bool(body.get("stream", False))

        # Get or create agent for this session.
        try:
            agent = await self._api_pool.get_or_create(session_id)
        except Exception as exc:
            log.error("API agent creation failed", session_id=session_id, error=str(exc))
            return web.json_response(
                {"error": {"message": f"Failed to initialize session: {exc}", "type": "server_error", "code": "agent_init_failed"}},
                status=500,
            )

        # Run agent.
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        try:
            response_text = await agent.complete(user_message)
        except Exception as exc:
            log.error("API completion failed", session_id=session_id, error=str(exc))
            return web.json_response(
                {"error": {"message": f"Completion failed: {exc}", "type": "server_error", "code": "completion_failed"}},
                status=500,
            )
        finally:
            await self._api_pool.release(session_id)

        # Resolve model name.
        model_name = "captain-claw"
        try:
            details = agent.get_runtime_model_details()
            model_name = details.get("model", "captain-claw")
        except Exception:
            pass

        usage = getattr(agent, "last_usage", None) or {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }

        if not stream:
            return web.json_response(
                self._build_chat_completion_response(
                    content=response_text,
                    model=model_name,
                    usage=usage,
                    completion_id=completion_id,
                )
            )

        return await self._write_sse_streaming_response(
            request=request,
            content=response_text,
            model=model_name,
            usage=usage,
            completion_id=completion_id,
        )

    async def _api_list_models(self, request: web.Request) -> web.Response:
        """``GET /v1/models`` — list available models."""
        models_data: list[dict[str, Any]] = []
        created = int(time.time())

        if self.agent:
            for entry in self.agent.get_allowed_models():
                model_id = entry.get("id", "unknown")
                models_data.append({
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "captain-claw",
                    "permission": [],
                    "root": model_id,
                    "parent": None,
                })

        # Always include a generic captain-claw entry.
        if not any(m["id"] == "captain-claw" for m in models_data):
            models_data.insert(0, {
                "id": "captain-claw",
                "object": "model",
                "created": created,
                "owned_by": "captain-claw",
                "permission": [],
                "root": "captain-claw",
                "parent": None,
            })

        return web.json_response({"object": "list", "data": models_data})


async def _run_server(config: Config) -> None:
    """Start the web server."""
    server = WebServer(config)
    loop = asyncio.get_event_loop()
    server._loop = loop

    # Stop event: set by signal handler to trigger graceful shutdown.
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        if not stop_event.is_set():
            stop_event.set()

    # Install signal handlers so the first Ctrl+C triggers graceful shutdown
    # without raising KeyboardInterrupt inside asyncio.run().
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except (NotImplementedError, OSError):
            # Windows doesn't support add_signal_handler for SIGTERM.
            pass

    print("Initializing Captain Claw agent...")
    await server._init_agent()

    # Start Telegram bridge if configured
    await server._start_telegram()

    app = server.create_app()
    runner = web.AppRunner(app)
    await runner.setup()

    host = config.web.host
    port = config.web.port
    site = web.TCPSite(runner, host, port)
    await site.start()

    print(f"\n  Captain Claw Web UI running at http://{host}:{port}")
    if config.web.api_enabled and server._api_pool:
        print(f"  OpenAI-compatible API at http://{host}:{port}/v1")
    if server._oauth_manager:
        connected = await server._oauth_manager.is_connected()
        status = "connected" if connected else "ready (not connected)"
        print(f"  Google OAuth: {status}")
    print(f"  Press Ctrl+C to stop.\n")

    # Start background cron scheduler loop so cron jobs fire automatically.
    cron_worker: asyncio.Task[None] | None = None
    try:
        from captain_claw.cron_dispatch import cron_scheduler_loop

        ctx = server._get_web_runtime_context()
        cron_worker = asyncio.create_task(cron_scheduler_loop(ctx))
        log.info("cron_scheduler_started")
        print("  Cron scheduler started.")
    except Exception as exc:
        log.warning("cron_scheduler_failed_to_start", error=str(exc))

    # Keep running until stop signal.
    await stop_event.wait()

    print("\nShutting down...")

    # Cancel background cron scheduler.
    if cron_worker and not cron_worker.done():
        cron_worker.cancel()
        try:
            await cron_worker
        except (asyncio.CancelledError, Exception):
            pass

    # Orchestrator must shut down first — cancels workers before pool.
    if server._orchestrator:
        try:
            await server._orchestrator.shutdown()
        except Exception:
            pass
    # Shut down API agent pool.
    if server._api_pool:
        try:
            await server._api_pool.shutdown()
        except Exception:
            pass
    try:
        await server._stop_telegram()
    except Exception:
        pass
    # Close all active WebSocket connections so the server can shut down cleanly.
    for ws in list(server.clients):
        try:
            await ws.close()
        except Exception:
            pass
    server.clients.clear()
    try:
        await runner.cleanup()
    except Exception:
        pass
    # All resources cleaned up — force exit to avoid hanging on
    # asyncio.run()'s shutdown_default_executor / thread joins.
    os._exit(0)


def run_web_server(config: Config) -> None:
    """Entry point for running the web server."""
    try:
        asyncio.run(_run_server(config))
    except KeyboardInterrupt:
        pass  # Signal handler handles graceful shutdown.


def main() -> None:
    """Standalone entry point for captain-claw-web."""
    configure_logging()

    # Load config same way as main.py
    cfg = Config.load()
    set_config(cfg)

    # Ensure directories
    session_path = Path(cfg.session.path).expanduser()
    session_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path = cfg.resolved_workspace_path(Path.cwd())
    workspace_path.mkdir(parents=True, exist_ok=True)

    try:
        run_web_server(cfg)
    except KeyboardInterrupt:
        print("\nWeb server stopped.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
