"""Web UI server for Captain Claw."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from aiohttp import web

from captain_claw.agent import Agent
from captain_claw.agent_pool import AgentPool
from captain_claw.config import Config, get_config, set_config
from captain_claw.google_oauth_manager import GoogleOAuthManager
from captain_claw.instructions import InstructionLoader
from captain_claw.logging import configure_logging, get_logger
from captain_claw.session_orchestrator import SessionOrchestrator
from captain_claw.telegram_bridge import TelegramBridge, TelegramMessage

log = get_logger(__name__)

STATIC_DIR = Path(
    os.environ.get("CAPTAIN_CLAW_STATIC_DIR", "")
    or str(Path(__file__).resolve().parent / "web" / "static")
)

# Available commands for the help/suggestion system
COMMANDS: list[dict[str, str]] = [
    {"command": "/help", "description": "Show command reference", "category": "General"},
    {"command": "/stop", "description": "Stop current processing (Esc)", "category": "General"},
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
    {"command": "/todo", "description": "List to-do items", "category": "Todo"},
    {"command": "/todo add <text>", "description": "Add a to-do item", "category": "Todo"},
    {"command": "/todo done <id|#index>", "description": "Mark item done", "category": "Todo"},
    {"command": "/todo remove <id|#index>", "description": "Remove a to-do item", "category": "Todo"},
    {"command": "/todo assign bot|human <id|#index>", "description": "Reassign responsible", "category": "Todo"},
    {"command": "/contacts", "description": "List contacts", "category": "Contacts"},
    {"command": "/contacts add <name>", "description": "Add a contact", "category": "Contacts"},
    {"command": "/contacts info <id|#index|name>", "description": "Show contact details", "category": "Contacts"},
    {"command": "/contacts search <query>", "description": "Search contacts", "category": "Contacts"},
    {"command": "/contacts update <id> <field=value>", "description": "Update contact fields", "category": "Contacts"},
    {"command": "/contacts importance <id> <1-10>", "description": "Set contact importance", "category": "Contacts"},
    {"command": "/contacts remove <id|#index|name>", "description": "Remove a contact", "category": "Contacts"},
    {"command": "/scripts", "description": "List scripts", "category": "Scripts"},
    {"command": "/scripts add <name> <path>", "description": "Register a script", "category": "Scripts"},
    {"command": "/scripts info <id|#index|name>", "description": "Show script details", "category": "Scripts"},
    {"command": "/scripts search <query>", "description": "Search scripts", "category": "Scripts"},
    {"command": "/scripts update <id> <field=value>", "description": "Update script fields", "category": "Scripts"},
    {"command": "/scripts remove <id|#index|name>", "description": "Remove a script", "category": "Scripts"},
    {"command": "/apis", "description": "List APIs", "category": "APIs"},
    {"command": "/apis add <name> <base_url>", "description": "Register an API", "category": "APIs"},
    {"command": "/apis info <id|#index|name>", "description": "Show API details", "category": "APIs"},
    {"command": "/apis search <query>", "description": "Search APIs", "category": "APIs"},
    {"command": "/apis update <id> <field=value>", "description": "Update API fields", "category": "APIs"},
    {"command": "/apis remove <id|#index|name>", "description": "Remove an API", "category": "APIs"},
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
        self._active_task: asyncio.Task | None = None
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
        self._telegram_user_sessions: dict[str, str] = {}  # user_id -> session_id
        self._telegram_agents: dict[str, Any] = {}  # user_id -> Agent
        self._telegram_user_locks: dict[str, asyncio.Lock] = {}  # user_id -> Lock
        self._telegram_poll_task: asyncio.Task[None] | None = None
        self._telegram_worker_task: asyncio.Task[None] | None = None
        # Orchestrator (lazy init in _init_agent)
        self._orchestrator: SessionOrchestrator | None = None
        # Loop runner state
        self._loop_runner_task: asyncio.Task[None] | None = None
        self._loop_runner_state: dict[str, Any] = {}
        self._loop_runner_stop: bool = False
        # OpenAI-compatible API agent pool (lazy init in _init_agent)
        self._api_pool: AgentPool | None = None
        # Google OAuth state
        self._oauth_manager: GoogleOAuthManager | None = None
        self._pending_oauth: dict[str, dict[str, Any]] = {}  # state → {verifier, ts}
        # BotPort client
        self._botport_client: Any = None

    async def _init_agent(self) -> None:
        """Initialize the agent with web callbacks."""
        self.agent = Agent(
            status_callback=self._status_callback,
            tool_output_callback=self._tool_output_callback,
            approval_callback=self._approval_callback,
            thinking_callback=self._thinking_callback,
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
            thinking_callback=self._thinking_callback,
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
            from captain_claw.web.google_oauth import inject_oauth_into_provider
            await inject_oauth_into_provider(self)

    # ── Callbacks ─────────────────────────────────────────────────────

    def _status_callback(self, status: str) -> None:
        """Broadcast status updates to all connected clients."""
        self._broadcast({"type": "status", "status": status})

    def _thinking_callback(self, text: str, tool: str = "", phase: str = "tool") -> None:
        """Broadcast inline thinking/reasoning updates to all connected clients."""
        self._broadcast({"type": "thinking", "text": text, "tool": tool, "phase": phase})

    _THINKING_SILENT_TOOLS: set[str] = {
        "llm_trace", "pipeline_trace", "memory_select", "memory_semantic_select",
        "compaction", "guard_input", "guard_output", "guard_web", "guard_exec",
        "guard_file", "approval", "scale_micro_loop", "task_rephrase",
    }

    def _tool_output_callback(
        self, tool_name: str, arguments: dict[str, Any], output: str
    ) -> None:
        """Broadcast tool output to monitor pane and update thinking indicator."""
        self._broadcast({
            "type": "monitor",
            "tool_name": tool_name,
            "arguments": arguments,
            "output": output,
        })
        normalized = str(tool_name or "").strip().lower()
        # When a task rephrase completes, broadcast the rephrased content
        # as a chat message so the UI can display it in a visible panel.
        if normalized == "task_rephrase":
            self._broadcast({
                "type": "chat_message",
                "role": "rephrase",
                "content": str(output or ""),
            })
        if normalized in ("image_gen", "termux", "browser") and output and not str(output).strip().lower().startswith("error"):
            from captain_claw.platform_adapter import extract_image_paths_from_tool_output
            for img_path in extract_image_paths_from_tool_output(str(output)):
                self._broadcast({
                    "type": "chat_message",
                    "role": "image",
                    "content": str(img_path),
                })
        if normalized not in self._THINKING_SILENT_TOOLS:
            from captain_claw.agent_tool_loop_mixin import AgentToolLoopMixin
            summary = AgentToolLoopMixin._tool_thinking_summary(tool_name, arguments or {})
            self._thinking_callback(summary, tool=tool_name, phase="tool")

    def _approval_callback(self, message: str) -> bool:
        """Handle tool approval requests from the agent."""
        self._broadcast({
            "type": "approval_notice",
            "message": message,
        })
        return True

    # ── Broadcast / Send ──────────────────────────────────────────────

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

        # Active user profile for the web UI (describes who the agent talks to).
        active_pid = getattr(self.agent, "_active_personality_id", None)
        personality_name = ""
        if active_pid:
            from captain_claw.personality import load_user_personality
            up = load_user_personality(active_pid)
            personality_name = up.name if up else ""

        # Active playbook override for the web UI.
        active_pbid = getattr(self.agent, "_playbook_override", None) or ""
        playbook_name = getattr(self.agent, "_playbook_override_name", "") or ""
        if not playbook_name:
            if active_pbid == "__none__":
                playbook_name = "None"
            elif not active_pbid:
                playbook_name = "Auto"

        return {
            "id": s.id,
            "name": s.name,
            "model": model_details.get("model", ""),
            "provider": model_details.get("provider", ""),
            "description": (s.metadata or {}).get("description", ""),
            "message_count": len(s.messages),
            "tools": self.agent.tools.list_tools() if self.agent else [],
            "skills": [
                {"name": cmd.name, "skill": cmd.skill_name, "description": cmd.description}
                for cmd in (self.agent.list_user_invocable_skills() if self.agent else [])
            ],
            "personality_id": active_pid or "",
            "personality_name": personality_name,
            "playbook_id": active_pbid,
            "playbook_name": playbook_name,
        }

    # ── Cron runtime context ─────────────────────────────────────────

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

    # ── Delegated handlers ───────────────────────────────────────────
    # Each handler delegates to a function in the captain_claw.web package.

    # WebSocket
    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        from captain_claw.web.ws_handler import ws_handler
        return await ws_handler(self, request)

    # Instructions
    async def list_instructions(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_instructions import list_instructions
        return await list_instructions(self, request)

    async def get_instruction(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_instructions import get_instruction
        return await get_instruction(self, request)

    async def put_instruction(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_instructions import put_instruction
        return await put_instruction(self, request)

    async def revert_instruction(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_instructions import revert_instruction
        return await revert_instruction(self, request)

    # Config / Sessions / Commands
    async def get_config_summary(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_config import get_config_summary
        return await get_config_summary(self, request)

    async def list_sessions_api(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sessions import list_sessions
        return await list_sessions(self, request)

    async def get_commands_api(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_config import get_commands_api
        return await get_commands_api(self, request)

    # Orchestrator
    async def _get_orchestrator_status(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import get_orchestrator_status
        return await get_orchestrator_status(self, request)

    async def _reset_orchestrator(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import reset_orchestrator
        return await reset_orchestrator(self, request)

    async def _get_orchestrator_skills(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import get_orchestrator_skills
        return await get_orchestrator_skills(self, request)

    async def _rephrase_orchestrator_input(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import rephrase_orchestrator_input
        return await rephrase_orchestrator_input(self, request)

    async def _edit_orchestrator_task(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import edit_orchestrator_task
        return await edit_orchestrator_task(self, request)

    async def _update_orchestrator_task(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import update_orchestrator_task
        return await update_orchestrator_task(self, request)

    async def _restart_orchestrator_task(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import restart_orchestrator_task
        return await restart_orchestrator_task(self, request)

    async def _pause_orchestrator_task(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import pause_orchestrator_task
        return await pause_orchestrator_task(self, request)

    async def _resume_orchestrator_task(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import resume_orchestrator_task
        return await resume_orchestrator_task(self, request)

    async def _postpone_orchestrator_task(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import postpone_orchestrator_task
        return await postpone_orchestrator_task(self, request)

    async def _prepare_orchestrator(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import prepare_orchestrator
        return await prepare_orchestrator(self, request)

    async def _get_orchestrator_sessions(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import get_orchestrator_sessions
        return await get_orchestrator_sessions(self, request)

    async def _get_orchestrator_models(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import get_orchestrator_models
        return await get_orchestrator_models(self, request)

    async def _list_workflows(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import list_workflows
        return await list_workflows(self, request)

    async def _save_workflow(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import save_workflow
        return await save_workflow(self, request)

    async def _load_workflow(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import load_workflow
        return await load_workflow(self, request)

    async def _delete_workflow(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import delete_workflow
        return await delete_workflow(self, request)

    # Cron
    async def _list_cron_jobs(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_cron import list_cron_jobs
        return await list_cron_jobs(self, request)

    async def _create_cron_job(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_cron import create_cron_job
        return await create_cron_job(self, request)

    async def _run_cron_job(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_cron import run_cron_job
        return await run_cron_job(self, request)

    async def _pause_cron_job(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_cron import pause_cron_job
        return await pause_cron_job(self, request)

    async def _resume_cron_job(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_cron import resume_cron_job
        return await resume_cron_job(self, request)

    async def _update_cron_job_payload(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_cron import update_cron_job_payload
        return await update_cron_job_payload(self, request)

    async def _delete_cron_job(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_cron import delete_cron_job
        return await delete_cron_job(self, request)

    async def _get_cron_job_history(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_cron import get_cron_job_history
        return await get_cron_job_history(self, request)

    # Todos
    async def _list_todos(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import list_todos
        return await list_todos(self, request)

    async def _create_todo(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import create_todo
        return await create_todo(self, request)

    async def _update_todo(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import update_todo
        return await update_todo(self, request)

    async def _delete_todo(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import delete_todo
        return await delete_todo(self, request)

    # Contacts
    async def _list_contacts(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import list_contacts
        return await list_contacts(self, request)

    async def _search_contacts(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import search_contacts
        return await search_contacts(self, request)

    async def _create_contact(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import create_contact
        return await create_contact(self, request)

    async def _get_contact(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import get_contact
        return await get_contact(self, request)

    async def _update_contact(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import update_contact
        return await update_contact(self, request)

    async def _delete_contact(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import delete_contact
        return await delete_contact(self, request)

    # Personality
    async def _get_personality(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_personality import get_personality
        return await get_personality(self, request)

    async def _put_personality(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_personality import put_personality
        return await put_personality(self, request)

    # User personalities
    async def _list_user_personalities(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_personality import list_user_personalities
        return await list_user_personalities(self, request)

    async def _get_user_personality(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_personality import get_user_personality
        return await get_user_personality(self, request)

    async def _put_user_personality(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_personality import put_user_personality
        return await put_user_personality(self, request)

    async def _delete_user_personality(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_personality import delete_user_personality
        return await delete_user_personality(self, request)

    async def _list_telegram_users(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_personality import list_telegram_users
        return await list_telegram_users(self, request)

    # Scripts
    async def _list_scripts(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import list_scripts
        return await list_scripts(self, request)

    async def _search_scripts(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import search_scripts
        return await search_scripts(self, request)

    async def _create_script(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import create_script
        return await create_script(self, request)

    async def _get_script(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import get_script
        return await get_script(self, request)

    async def _update_script_api(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import update_script
        return await update_script(self, request)

    async def _delete_script_api(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import delete_script
        return await delete_script(self, request)

    # APIs
    async def _list_apis_api(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import list_apis
        return await list_apis(self, request)

    async def _search_apis_api(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import search_apis
        return await search_apis(self, request)

    async def _create_api_api(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import create_api
        return await create_api(self, request)

    async def _get_api_api(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import get_api
        return await get_api(self, request)

    async def _update_api_api(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import update_api
        return await update_api(self, request)

    async def _delete_api_api(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import delete_api
        return await delete_api(self, request)

    # Workflow browser
    async def _list_workflow_outputs(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_workflows import list_workflow_outputs
        return await list_workflow_outputs(self, request)

    async def _get_workflow_output(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_workflows import get_workflow_output
        return await get_workflow_output(self, request)

    # File browser
    async def _list_files(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_files import list_files
        return await list_files(self, request)

    async def _list_session_files(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_files import list_session_files
        return await list_session_files(self, request)

    async def _get_file_content(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_files import get_file_content
        return await get_file_content(self, request)

    async def _download_file(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_files import download_file
        return await download_file(self, request)

    async def _serve_media(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_files import serve_media
        return await serve_media(self, request)

    async def _image_upload(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_image_upload import upload_image
        return await upload_image(self, request)

    async def _file_upload(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_file_upload import upload_file
        return await upload_file(self, request)

    # Loop runner
    async def _start_loop(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_loops import start_loop
        return await start_loop(self, request)

    async def _get_loop_status(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_loops import get_loop_status
        return await get_loop_status(self, request)

    async def _stop_loop(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_loops import stop_loop
        return await stop_loop(self, request)

    # OpenAI-compatible API proxy
    async def _api_chat_completions(self, request: web.Request) -> web.Response | web.StreamResponse:
        from captain_claw.web.openai_proxy import api_chat_completions
        return await api_chat_completions(self, request)

    async def _api_list_models(self, request: web.Request) -> web.Response:
        from captain_claw.web.openai_proxy import api_list_models
        return await api_list_models(self, request)

    # Google OAuth
    async def _auth_google_login(self, request: web.Request) -> web.Response:
        from captain_claw.web.google_oauth import auth_google_login
        return await auth_google_login(self, request)

    async def _auth_google_callback(self, request: web.Request) -> web.Response:
        from captain_claw.web.google_oauth import auth_google_callback
        return await auth_google_callback(self, request)

    async def _auth_google_status(self, request: web.Request) -> web.Response:
        from captain_claw.web.google_oauth import auth_google_status
        return await auth_google_status(self, request)

    async def _auth_google_logout(self, request: web.Request) -> web.Response:
        from captain_claw.web.google_oauth import auth_google_logout
        return await auth_google_logout(self, request)

    # Static pages
    async def _serve_home(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_home
        return await serve_home(self, request)

    async def _serve_chat(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_chat
        return await serve_chat(self, request)

    async def _serve_favicon(self, request: web.Request) -> web.Response:
        from captain_claw.web.static_pages import serve_favicon
        return await serve_favicon(self, request)

    async def _serve_orchestrator(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_orchestrator
        return await serve_orchestrator(self, request)

    async def _serve_instructions(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_instructions
        return await serve_instructions(self, request)

    async def _serve_cron(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_cron
        return await serve_cron(self, request)

    async def _serve_workflows(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_workflows
        return await serve_workflows(self, request)

    async def _serve_loop_runner(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_loop_runner
        return await serve_loop_runner(self, request)

    async def _serve_memory(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_memory
        return await serve_memory(self, request)

    async def _serve_deep_memory(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_deep_memory
        return await serve_deep_memory(self, request)

    async def _dm_status(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_deep_memory import get_status
        return await get_status(self, request)

    async def _dm_documents(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_deep_memory import list_documents
        return await list_documents(self, request)

    async def _dm_document_detail(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_deep_memory import get_document
        return await get_document(self, request)

    async def _dm_document_delete(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_deep_memory import delete_document
        return await delete_document(self, request)

    async def _dm_facets(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_deep_memory import get_facets
        return await get_facets(self, request)

    async def _dm_index(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_deep_memory import index_document
        return await index_document(self, request)

    async def _serve_settings(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_settings
        return await serve_settings(self, request)

    async def _serve_sessions(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_sessions
        return await serve_sessions(self, request)

    async def _serve_files(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_files
        return await serve_files(self, request)

    async def _serve_onboarding(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_onboarding
        return await serve_onboarding(self, request)

    async def _serve_datastore(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_datastore
        return await serve_datastore(self, request)

    async def _serve_playbooks(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_playbooks
        return await serve_playbooks(self, request)

    async def _serve_skills(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_skills
        return await serve_skills(self, request)

    # Playbooks REST
    async def _pb_list(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_playbooks import list_playbooks
        return await list_playbooks(self, request)

    async def _pb_search(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_playbooks import search_playbooks
        return await search_playbooks(self, request)

    async def _pb_get(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_playbooks import get_playbook
        return await get_playbook(self, request)

    async def _pb_create(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_playbooks import create_playbook
        return await create_playbook(self, request)

    async def _pb_update(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_playbooks import update_playbook
        return await update_playbook(self, request)

    async def _pb_delete(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_playbooks import delete_playbook
        return await delete_playbook(self, request)

    async def _serve_browser_workflows(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_browser_workflows
        return await serve_browser_workflows(self, request)

    # Browser Workflows REST
    async def _bw_list(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_browser_workflows import list_workflows
        return await list_workflows(self, request)

    async def _bw_get(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_browser_workflows import get_workflow
        return await get_workflow(self, request)

    async def _bw_create(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_browser_workflows import create_workflow
        return await create_workflow(self, request)

    async def _bw_update(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_browser_workflows import update_workflow
        return await update_workflow(self, request)

    async def _bw_delete(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_browser_workflows import delete_workflow
        return await delete_workflow(self, request)

    # Direct API Calls page
    async def _serve_direct_api_calls(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_direct_api_calls
        return await serve_direct_api_calls(self, request)

    # Direct API Calls REST
    async def _dac_list(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_direct_api import list_calls
        return await list_calls(self, request)

    async def _dac_get(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_direct_api import get_call
        return await get_call(self, request)

    async def _dac_create(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_direct_api import create_call
        return await create_call(self, request)

    async def _dac_update(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_direct_api import update_call
        return await update_call(self, request)

    async def _dac_delete(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_direct_api import delete_call
        return await delete_call(self, request)

    async def _dac_execute(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_direct_api import execute_call
        return await execute_call(self, request)

    # Datastore REST
    async def _ds_list_tables(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import list_tables
        return await list_tables(self, request)

    async def _ds_describe_table(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import describe_table
        return await describe_table(self, request)

    async def _ds_create_table(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import create_table
        return await create_table(self, request)

    async def _ds_drop_table(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import drop_table
        return await drop_table(self, request)

    async def _ds_rename_table(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import rename_table
        return await rename_table(self, request)

    async def _ds_query_rows(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import query_rows
        return await query_rows(self, request)

    async def _ds_insert_rows(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import insert_rows
        return await insert_rows(self, request)

    async def _ds_update_rows(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import update_rows
        return await update_rows(self, request)

    async def _ds_delete_rows(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import delete_rows
        return await delete_rows(self, request)

    async def _ds_add_column(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import add_column
        return await add_column(self, request)

    async def _ds_drop_column(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import drop_column
        return await drop_column(self, request)

    async def _ds_run_sql(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import run_sql
        return await run_sql(self, request)

    async def _ds_list_protections(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import list_protections
        return await list_protections(self, request)

    async def _ds_add_protection(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import add_protection
        return await add_protection(self, request)

    async def _ds_remove_protection(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import remove_protection
        return await remove_protection(self, request)

    async def _ds_upload_and_import(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import upload_and_import
        return await upload_and_import(self, request)

    async def _ds_export_table(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_datastore import export_table
        return await export_table(self, request)

    # Onboarding REST
    async def _get_onboarding_status(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_onboarding import get_onboarding_status
        return await get_onboarding_status(self, request)

    async def _post_onboarding_validate(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_onboarding import post_onboarding_validate
        return await post_onboarding_validate(self, request)

    async def _post_onboarding_save(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_onboarding import post_onboarding_save
        return await post_onboarding_save(self, request)

    # Sessions REST
    async def _get_session_detail(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sessions import get_session_detail
        return await get_session_detail(self, request)

    async def _update_session(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sessions import update_session
        return await update_session(self, request)

    async def _delete_session_api(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sessions import delete_session
        return await delete_session(self, request)

    async def _auto_describe_session(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sessions import auto_describe_session
        return await auto_describe_session(self, request)

    async def _export_session(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sessions import export_session
        return await export_session(self, request)

    async def _bulk_delete_sessions(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sessions import bulk_delete_sessions
        return await bulk_delete_sessions(self, request)

    async def _get_settings_schema(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_settings import get_settings_schema
        return await get_settings_schema(self, request)

    async def _get_settings_values(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_settings import get_settings_values
        return await get_settings_values(self, request)

    async def _put_settings(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_settings import put_settings
        return await put_settings(self, request)

    # ── Skills REST ─────────────────────────────────────────────────

    async def _list_skills(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import list_skills
        return await list_skills(self, request)

    async def _install_skill(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import install_skill
        return await install_skill(self, request)

    async def _toggle_skill(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import toggle_skill
        return await toggle_skill(self, request)

    async def _browse_directory(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import browse_directory
        return await browse_directory(self, request)

    async def _list_read_folders(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import list_read_folders
        return await list_read_folders(self, request)

    async def _add_read_folder(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import add_read_folder
        return await add_read_folder(self, request)

    async def _remove_read_folder(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import remove_read_folder
        return await remove_read_folder(self, request)

    # ── App setup ────────────────────────────────────────────────────

    def create_app(self) -> web.Application:
        app = web.Application(client_max_size=50 * 1024 * 1024)  # 50 MB for file uploads
        if self.config.web.auth_token:
            from captain_claw.web.auth import create_auth_middleware
            app.middlewares.append(create_auth_middleware(self.config.web))
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_get("/api/instructions", self.list_instructions)
        app.router.add_get("/api/instructions/{name}", self.get_instruction)
        app.router.add_put("/api/instructions/{name}", self.put_instruction)
        app.router.add_delete("/api/instructions/{name}", self.revert_instruction)
        app.router.add_get("/api/config", self.get_config_summary)
        app.router.add_get("/api/settings/schema", self._get_settings_schema)
        app.router.add_get("/api/settings", self._get_settings_values)
        app.router.add_put("/api/settings", self._put_settings)
        app.router.add_get("/api/skills", self._list_skills)
        app.router.add_post("/api/skills/install", self._install_skill)
        app.router.add_post("/api/skills/toggle", self._toggle_skill)
        app.router.add_get("/api/read-folders", self._list_read_folders)
        app.router.add_post("/api/read-folders", self._add_read_folder)
        app.router.add_delete("/api/read-folders", self._remove_read_folder)
        app.router.add_get("/api/browse", self._browse_directory)
        app.router.add_get("/api/sessions", self.list_sessions_api)
        app.router.add_post("/api/sessions/bulk-delete", self._bulk_delete_sessions)
        app.router.add_get("/api/sessions/{id}", self._get_session_detail)
        app.router.add_patch("/api/sessions/{id}", self._update_session)
        app.router.add_delete("/api/sessions/{id}", self._delete_session_api)
        app.router.add_post("/api/sessions/{id}/auto-describe", self._auto_describe_session)
        app.router.add_get("/api/sessions/{id}/export", self._export_session)
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
        app.router.add_post("/api/orchestrator/prepare", self._prepare_orchestrator)
        app.router.add_get("/api/orchestrator/sessions", self._get_orchestrator_sessions)
        app.router.add_get("/api/orchestrator/models", self._get_orchestrator_models)
        app.router.add_get("/api/orchestrator/workflows", self._list_workflows)
        app.router.add_post("/api/orchestrator/workflows/save", self._save_workflow)
        app.router.add_post("/api/orchestrator/workflows/load", self._load_workflow)
        app.router.add_delete("/api/orchestrator/workflows/{name}", self._delete_workflow)
        app.router.add_get("/api/cron/jobs", self._list_cron_jobs)
        app.router.add_post("/api/cron/jobs", self._create_cron_job)
        app.router.add_post("/api/cron/jobs/{id}/run", self._run_cron_job)
        app.router.add_post("/api/cron/jobs/{id}/pause", self._pause_cron_job)
        app.router.add_post("/api/cron/jobs/{id}/resume", self._resume_cron_job)
        app.router.add_patch("/api/cron/jobs/{id}", self._update_cron_job_payload)
        app.router.add_delete("/api/cron/jobs/{id}", self._delete_cron_job)
        app.router.add_get("/api/cron/jobs/{id}/history", self._get_cron_job_history)
        app.router.add_get("/api/todos", self._list_todos)
        app.router.add_post("/api/todos", self._create_todo)
        app.router.add_patch("/api/todos/{id}", self._update_todo)
        app.router.add_delete("/api/todos/{id}", self._delete_todo)
        app.router.add_get("/api/contacts", self._list_contacts)
        app.router.add_get("/api/contacts/search", self._search_contacts)
        app.router.add_post("/api/contacts", self._create_contact)
        app.router.add_get("/api/contacts/{id}", self._get_contact)
        app.router.add_patch("/api/contacts/{id}", self._update_contact)
        app.router.add_delete("/api/contacts/{id}", self._delete_contact)
        app.router.add_get("/api/scripts", self._list_scripts)
        app.router.add_get("/api/scripts/search", self._search_scripts)
        app.router.add_post("/api/scripts", self._create_script)
        app.router.add_get("/api/scripts/{id}", self._get_script)
        app.router.add_patch("/api/scripts/{id}", self._update_script_api)
        app.router.add_delete("/api/scripts/{id}", self._delete_script_api)
        app.router.add_get("/api/apis", self._list_apis_api)
        app.router.add_get("/api/apis/search", self._search_apis_api)
        app.router.add_post("/api/apis", self._create_api_api)
        app.router.add_get("/api/apis/{id}", self._get_api_api)
        app.router.add_patch("/api/apis/{id}", self._update_api_api)
        app.router.add_delete("/api/apis/{id}", self._delete_api_api)
        # Personality
        app.router.add_get("/api/personality", self._get_personality)
        app.router.add_put("/api/personality", self._put_personality)
        # User personalities
        app.router.add_get("/api/user-personalities", self._list_user_personalities)
        app.router.add_get("/api/user-personalities/{user_id}", self._get_user_personality)
        app.router.add_put("/api/user-personalities/{user_id}", self._put_user_personality)
        app.router.add_delete("/api/user-personalities/{user_id}", self._delete_user_personality)
        app.router.add_get("/api/telegram-users", self._list_telegram_users)
        app.router.add_get("/api/workflow-browser", self._list_workflow_outputs)
        app.router.add_get("/api/workflow-browser/output/{filename}", self._get_workflow_output)
        app.router.add_get("/api/files", self._list_files)
        app.router.add_get("/api/files/session/{session_id}", self._list_session_files)
        app.router.add_get("/api/files/content", self._get_file_content)
        app.router.add_get("/api/files/download", self._download_file)
        app.router.add_get("/api/media", self._serve_media)
        app.router.add_post("/api/image/upload", self._image_upload)
        app.router.add_post("/api/file/upload", self._file_upload)
        app.router.add_post("/api/loops/start", self._start_loop)
        app.router.add_get("/api/loops/status", self._get_loop_status)
        app.router.add_post("/api/loops/stop", self._stop_loop)
        # Datastore
        app.router.add_get("/api/datastore/tables", self._ds_list_tables)
        app.router.add_post("/api/datastore/tables", self._ds_create_table)
        app.router.add_get("/api/datastore/tables/{name}/export", self._ds_export_table)
        app.router.add_get("/api/datastore/tables/{name}", self._ds_describe_table)
        app.router.add_delete("/api/datastore/tables/{name}", self._ds_drop_table)
        app.router.add_patch("/api/datastore/tables/{name}", self._ds_rename_table)
        app.router.add_get("/api/datastore/tables/{name}/rows", self._ds_query_rows)
        app.router.add_post("/api/datastore/tables/{name}/rows", self._ds_insert_rows)
        app.router.add_patch("/api/datastore/tables/{name}/rows", self._ds_update_rows)
        app.router.add_delete("/api/datastore/tables/{name}/rows", self._ds_delete_rows)
        app.router.add_post("/api/datastore/tables/{name}/columns", self._ds_add_column)
        app.router.add_delete("/api/datastore/tables/{name}/columns/{col}", self._ds_drop_column)
        app.router.add_post("/api/datastore/sql", self._ds_run_sql)
        app.router.add_get("/api/datastore/tables/{name}/protections", self._ds_list_protections)
        app.router.add_post("/api/datastore/tables/{name}/protections", self._ds_add_protection)
        app.router.add_delete("/api/datastore/tables/{name}/protections", self._ds_remove_protection)
        app.router.add_post("/api/datastore/upload", self._ds_upload_and_import)
        # Playbooks API
        app.router.add_get("/api/playbooks", self._pb_list)
        app.router.add_get("/api/playbooks/search", self._pb_search)
        app.router.add_post("/api/playbooks", self._pb_create)
        app.router.add_get("/api/playbooks/{id}", self._pb_get)
        app.router.add_patch("/api/playbooks/{id}", self._pb_update)
        app.router.add_delete("/api/playbooks/{id}", self._pb_delete)
        # Browser Workflows API
        app.router.add_get("/api/browser-workflows", self._bw_list)
        app.router.add_post("/api/browser-workflows", self._bw_create)
        app.router.add_get("/api/browser-workflows/{id}", self._bw_get)
        app.router.add_patch("/api/browser-workflows/{id}", self._bw_update)
        app.router.add_delete("/api/browser-workflows/{id}", self._bw_delete)
        # Direct API Calls API
        app.router.add_get("/api/direct-api-calls", self._dac_list)
        app.router.add_post("/api/direct-api-calls", self._dac_create)
        app.router.add_get("/api/direct-api-calls/{id}", self._dac_get)
        app.router.add_patch("/api/direct-api-calls/{id}", self._dac_update)
        app.router.add_delete("/api/direct-api-calls/{id}", self._dac_delete)
        app.router.add_post("/api/direct-api-calls/{id}/execute", self._dac_execute)
        # Deep memory (Typesense) API
        app.router.add_get("/api/deep-memory/status", self._dm_status)
        app.router.add_get("/api/deep-memory/facets", self._dm_facets)
        app.router.add_get("/api/deep-memory/documents", self._dm_documents)
        app.router.add_get("/api/deep-memory/documents/{doc_id}", self._dm_document_detail)
        app.router.add_delete("/api/deep-memory/documents/{doc_id}", self._dm_document_delete)
        app.router.add_post("/api/deep-memory/index", self._dm_index)
        if self.config.web.api_enabled and self._api_pool:
            app.router.add_post("/v1/chat/completions", self._api_chat_completions)
            app.router.add_get("/v1/models", self._api_list_models)
        app.router.add_get("/api/onboarding/status", self._get_onboarding_status)
        app.router.add_post("/api/onboarding/validate", self._post_onboarding_validate)
        app.router.add_post("/api/onboarding/save", self._post_onboarding_save)
        app.router.add_get("/auth/google/login", self._auth_google_login)
        app.router.add_get("/auth/google/callback", self._auth_google_callback)
        app.router.add_get("/auth/google/status", self._auth_google_status)
        app.router.add_post("/auth/google/logout", self._auth_google_logout)
        if STATIC_DIR.is_dir():
            app.router.add_static("/static/", STATIC_DIR, show_index=False)
            app.router.add_get("/", self._serve_home)
            app.router.add_get("/chat", self._serve_chat)
            app.router.add_get("/orchestrator", self._serve_orchestrator)
            app.router.add_get("/instructions", self._serve_instructions)
            app.router.add_get("/cron", self._serve_cron)
            app.router.add_get("/workflows", self._serve_workflows)
            app.router.add_get("/loop-runner", self._serve_loop_runner)
            app.router.add_get("/memory", self._serve_memory)
            app.router.add_get("/deep-memory", self._serve_deep_memory)
            app.router.add_get("/settings", self._serve_settings)
            app.router.add_get("/sessions", self._serve_sessions)
            app.router.add_get("/files", self._serve_files)
            app.router.add_get("/onboarding", self._serve_onboarding)
            app.router.add_get("/datastore", self._serve_datastore)
            app.router.add_get("/playbooks", self._serve_playbooks)
            app.router.add_get("/browser-workflows", self._serve_browser_workflows)
            app.router.add_get("/direct-api-calls", self._serve_direct_api_calls)
            app.router.add_get("/skills", self._serve_skills)
            app.router.add_get("/favicon.ico", self._serve_favicon)
        return app


async def _run_server(config: Config) -> None:
    """Start the web server."""
    server = WebServer(config)
    loop = asyncio.get_event_loop()
    server._loop = loop

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        if not stop_event.is_set():
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except (NotImplementedError, OSError):
            pass

    print("Initializing Captain Claw agent...")
    await server._init_agent()

    from captain_claw.web.telegram import start_telegram, stop_telegram
    await start_telegram(server)

    # Start BotPort client if configured.
    if config.botport.enabled and config.botport.url.strip():
        try:
            from captain_claw.botport_client import BotPortClient
            bp_client = BotPortClient(
                config=config.botport,
                provider=server.agent.provider if server.agent else None,
                status_callback=server._status_callback,
                tool_output_callback=server._tool_output_callback,
                thinking_callback=server._thinking_callback,
            )
            await bp_client.start()
            server._botport_client = bp_client
            # Set client on agent's botport tool if registered.
            if server.agent:
                server.agent._botport_client = bp_client
                bt = server.agent.tools.get("botport")
                if bt is not None:
                    bt.set_client(bp_client)
            print(f"  BotPort client connected to {config.botport.url}")
        except Exception as exc:
            log.warning("BotPort client failed to start", error=str(exc))
            print(f"  BotPort client failed: {exc}")

    app = server.create_app()
    runner = web.AppRunner(app)
    await runner.setup()

    host = config.web.host
    port = config.web.port
    max_retries = 10
    for attempt in range(max_retries):
        try:
            site = web.TCPSite(runner, host, port)
            await site.start()
            break
        except OSError as exc:
            if attempt < max_retries - 1:
                log.warning("Port unavailable, trying next", port=port, next_port=port + 1, error=str(exc))
                port += 1
            else:
                raise

    print(f"\n  Captain Claw Web UI running at http://{host}:{port}")
    if config.web.auth_token:
        print(f"  Authentication enabled. Access via: http://{host}:{port}/?token=<your-token>")
    if config.web.api_enabled and server._api_pool:
        print(f"  OpenAI-compatible API at http://{host}:{port}/v1")
    if server._oauth_manager:
        connected = await server._oauth_manager.is_connected()
        status = "connected" if connected else "ready (not connected)"
        print(f"  Google OAuth: {status}")
    print(f"  Press Ctrl+C to stop.\n")

    cron_worker: asyncio.Task[None] | None = None
    try:
        from captain_claw.cron_dispatch import cron_scheduler_loop

        ctx = server._get_web_runtime_context()
        cron_worker = asyncio.create_task(cron_scheduler_loop(ctx))
        log.info("cron_scheduler_started")
        print("  Cron scheduler started.")
    except Exception as exc:
        log.warning("cron_scheduler_failed_to_start", error=str(exc))

    await stop_event.wait()

    print("\nShutting down...")

    if cron_worker and not cron_worker.done():
        cron_worker.cancel()
        try:
            await cron_worker
        except (asyncio.CancelledError, Exception):
            pass

    if server._orchestrator:
        try:
            await server._orchestrator.shutdown()
        except Exception:
            pass
    if server._api_pool:
        try:
            await server._api_pool.shutdown()
        except Exception:
            pass
    try:
        await stop_telegram(server)
    except Exception:
        pass
    if server._botport_client:
        try:
            await server._botport_client.stop()
        except Exception:
            pass
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
    os._exit(0)


def run_web_server(config: Config) -> None:
    """Entry point for running the web server."""
    try:
        asyncio.run(_run_server(config))
    except KeyboardInterrupt:
        pass


def main() -> None:
    """Standalone entry point for captain-claw-web."""
    configure_logging()

    cfg = Config.load()
    set_config(cfg)

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


if __name__ == "__main__":
    main()
