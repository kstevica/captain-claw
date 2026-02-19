"""Web UI server for Captain Claw."""

import asyncio
import json
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from captain_claw.agent import Agent
from captain_claw.config import Config, get_config, set_config
from captain_claw.instructions import InstructionLoader
from captain_claw.logging import configure_logging, get_logger
from captain_claw.session import get_session_manager
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
        self._instructions_dir = InstructionLoader().base_dir
        self._loop: asyncio.AbstractEventLoop | None = None

        # Telegram bridge state
        telegram_cfg = config.telegram
        self._telegram_enabled = bool(
            telegram_cfg.enabled or telegram_cfg.bot_token.strip()
        )
        self._telegram_bridge: TelegramBridge | None = None
        self._telegram_offset: int | None = None
        self._approved_telegram_users: dict[str, dict[str, object]] = {}
        self._pending_telegram_pairings: dict[str, dict[str, object]] = {}
        self._telegram_poll_task: asyncio.Task[None] | None = None
        self._telegram_worker_task: asyncio.Task[None] | None = None

    async def _init_agent(self) -> None:
        """Initialize the agent with web callbacks."""
        self.agent = Agent(
            status_callback=self._status_callback,
            tool_output_callback=self._tool_output_callback,
            approval_callback=self._approval_callback,
        )
        await self.agent.initialize()

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
                await self.agent.session_manager.set_last_active_session(session.id)
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
            await self.agent.session_manager.set_last_active_session(session.id)
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
        """Process queued Telegram messages one at a time."""
        while True:
            try:
                message = await self._telegram_queue.get()
                await self._handle_telegram_message(message)
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
        """List instruction .md files."""
        files = []
        if self._instructions_dir.is_dir():
            for f in sorted(self._instructions_dir.iterdir()):
                if f.suffix == ".md" and f.is_file():
                    files.append({
                        "name": f.name,
                        "size": f.stat().st_size,
                    })
        return web.json_response(files)

    async def get_instruction(self, request: web.Request) -> web.Response:
        """Read an instruction file."""
        name = request.match_info["name"]
        # Sanitize: only allow .md files, no path traversal
        if ".." in name or "/" in name or not name.endswith(".md"):
            return web.json_response({"error": "Invalid file name"}, status=400)
        path = self._instructions_dir / name
        if not path.is_file():
            return web.json_response({"error": "File not found"}, status=404)
        content = path.read_text(encoding="utf-8")
        return web.json_response({"name": name, "content": content})

    async def put_instruction(self, request: web.Request) -> web.Response:
        """Save or create an instruction file."""
        name = request.match_info["name"]
        # Security: no path traversal, no subdirectories, .md only
        if ".." in name or "/" in name or "\\" in name or not name.endswith(".md"):
            return web.json_response({"error": "Invalid file name"}, status=400)
        path = self._instructions_dir / name
        # Ensure the resolved path stays inside the instructions directory
        try:
            path.resolve().relative_to(self._instructions_dir.resolve())
        except ValueError:
            return web.json_response({"error": "Invalid path"}, status=400)
        body = await request.json()
        content = body.get("content", "")
        is_new = not path.exists()
        self._instructions_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        # Clear instruction cache so changes take effect on next agent request
        if self.agent:
            self.agent.instructions._cache.pop(name, None)
        status = "created" if is_new else "saved"
        return web.json_response({"status": status, "name": name, "size": path.stat().st_size})

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

    # ── App setup ────────────────────────────────────────────────────

    def create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_get("/api/instructions", self.list_instructions)
        app.router.add_get("/api/instructions/{name}", self.get_instruction)
        app.router.add_put("/api/instructions/{name}", self.put_instruction)
        app.router.add_get("/api/config", self.get_config_summary)
        app.router.add_get("/api/sessions", self.list_sessions_api)
        app.router.add_get("/api/commands", self.get_commands_api)
        # Static files (serve index.html at /)
        if STATIC_DIR.is_dir():
            app.router.add_static("/static/", STATIC_DIR, show_index=False)
            app.router.add_get("/", self._serve_index)
            app.router.add_get("/favicon.ico", self._serve_favicon)
        return app

    async def _serve_index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "index.html")

    async def _serve_favicon(self, request: web.Request) -> web.Response:
        # Return a simple empty response if no favicon exists
        favicon = STATIC_DIR / "favicon.svg"
        if favicon.is_file():
            return web.FileResponse(favicon)
        return web.Response(status=204)


async def _run_server(config: Config) -> None:
    """Start the web server."""
    server = WebServer(config)
    server._loop = asyncio.get_event_loop()

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
    print(f"  Press Ctrl+C to stop.\n")

    # Keep running until interrupted
    try:
        await asyncio.Event().wait()
    finally:
        await server._stop_telegram()
        await runner.cleanup()


def run_web_server(config: Config) -> None:
    """Entry point for running the web server."""
    asyncio.run(_run_server(config))


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
