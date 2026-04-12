"""Web UI server for Captain Claw."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

# Ensure HOME is set — some environments (Docker, systemd, PyInstaller)
# may not have it, causing Path.expanduser() / Path.home() to fail.
if "HOME" not in os.environ:
    try:
        os.environ["HOME"] = str(Path.home())
    except RuntimeError:
        try:
            import pwd
            os.environ["HOME"] = pwd.getpwuid(os.getuid()).pw_dir
        except (KeyError, ImportError):
            os.environ["HOME"] = "/tmp"
from typing import Any

from aiohttp import web

from captain_claw.agent import Agent
from captain_claw.agent_pool import AgentPool
from captain_claw.config import Config, get_config, set_config
from captain_claw.google_oauth_manager import GoogleOAuthManager
from captain_claw.instructions import InstructionLoader
from captain_claw.logging import configure_logging, get_logger
from captain_claw.ws_utils import fire_and_forget_send
from captain_claw.session_orchestrator import SessionOrchestrator
from captain_claw.telegram_bridge import TelegramBridge, TelegramMessage

log = get_logger(__name__)


# ── Server-side audio playback ─────────────────────────────
_audio_player_proc: "subprocess.Popen[bytes] | None" = None


def _play_audio_local(file_path: str) -> None:
    """Play an audio file on the server machine using the system player.

    Uses afplay (macOS), ffplay, or aplay (Linux) — whichever is found first.
    Runs as a background subprocess so it doesn't block the event loop.
    Any previous playback is stopped before starting a new one.
    """
    import shutil
    import subprocess

    global _audio_player_proc

    resolved = Path(file_path)
    if not resolved.is_file():
        log.debug("play_audio_local: file not found", path=file_path)
        return

    # Stop any currently playing audio.
    if _audio_player_proc is not None:
        try:
            _audio_player_proc.terminate()
        except Exception:
            pass
        _audio_player_proc = None

    # Find a suitable player.
    player_cmds: list[list[str]] = []
    if sys.platform == "darwin":
        if shutil.which("afplay"):
            player_cmds.append(["afplay", str(resolved)])
    if shutil.which("ffplay"):
        player_cmds.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(resolved)])
    if shutil.which("mpv"):
        player_cmds.append(["mpv", "--no-video", str(resolved)])
    if shutil.which("aplay") and resolved.suffix.lower() == ".wav":
        player_cmds.append(["aplay", str(resolved)])

    if not player_cmds:
        log.warning("play_audio_local: no audio player found (install afplay/ffplay/mpv)")
        return

    cmd = player_cmds[0]
    try:
        _audio_player_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.debug("play_audio_local: started", cmd=cmd[0], pid=_audio_player_proc.pid)
    except Exception as e:
        log.warning("play_audio_local: failed to start", cmd=cmd, error=str(e))


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
    {"command": "/nuke", "description": "Delete all workspace files, memory, and datastore", "category": "General"},
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
    {"command": "/contacts import <path>", "description": "Import from Google CSV or vCard", "category": "Contacts"},
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
    {"command": "/insights", "description": "List recent insights", "category": "Insights"},
    {"command": "/insights search <query>", "description": "Search insights", "category": "Insights"},
    {"command": "/insights add <text>", "description": "Add an insight manually", "category": "Insights"},
    {"command": "/insights delete <id>", "description": "Delete an insight", "category": "Insights"},
    {"command": "/monitor on|off", "description": "Toggle monitor split view", "category": "Monitor"},
    {"command": "/monitor trace on|off", "description": "Toggle LLM trace logging", "category": "Monitor"},
    {"command": "/approve user telegram <token>", "description": "Approve a Telegram user pairing", "category": "Telegram"},
    {"command": "/orchestrate <request>", "description": "Run parallel multi-session orchestration", "category": "Orchestrator"},
    {"command": "/screenshot [prompt]", "description": "Capture screen and analyze with vision", "category": "Screen"},
    {"command": "/reflection", "description": "Show latest self-reflection", "category": "Reflections"},
    {"command": "/reflection generate", "description": "Trigger a new self-reflection", "category": "Reflections"},
    {"command": "/reflection list", "description": "List recent reflections", "category": "Reflections"},
    {"command": "/intuition", "description": "List recent intuitions", "category": "Nervous System"},
    {"command": "/intuition search <query>", "description": "Search intuitions", "category": "Nervous System"},
    {"command": "/intuition dream", "description": "Trigger a dream cycle", "category": "Nervous System"},
    {"command": "/intuition add <text>", "description": "Add an intuition manually", "category": "Nervous System"},
    {"command": "/intuition delete <id>", "description": "Delete an intuition", "category": "Nervous System"},
    {"command": "/intuition stats", "description": "Show nervous system statistics", "category": "Nervous System"},
    {"command": "/intuition validate <id>", "description": "Validate an intuition (protect from decay)", "category": "Nervous System"},
    {"command": "/briefing", "description": "List unread briefings", "category": "Sister Session"},
    {"command": "/briefing all", "description": "List all briefings", "category": "Sister Session"},
    {"command": "/briefing dismiss <id>", "description": "Dismiss a briefing", "category": "Sister Session"},
    {"command": "/briefing dismiss all", "description": "Dismiss all unread briefings", "category": "Sister Session"},
    {"command": "/briefing stats", "description": "Show sister session statistics", "category": "Sister Session"},
    {"command": "/sister investigate <query>", "description": "Create an investigation task", "category": "Sister Session"},
    {"command": "/sister status", "description": "Show sister session status", "category": "Sister Session"},
    {"command": "/sister tasks", "description": "List queued tasks", "category": "Sister Session"},
    {"command": "/sister delete <id>", "description": "Cancel a queued task", "category": "Sister Session"},
    {"command": "/watch", "description": "List active watches", "category": "Sister Session"},
    {"command": "/watch every <interval> <query>", "description": "Create a recurring watch", "category": "Sister Session"},
    {"command": "/watch delete <id>", "description": "Delete a watch", "category": "Sister Session"},
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
        # Runtime context for cron execution (set when cron scheduler starts).
        self._runtime_ctx: Any = None
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
        # Hotkey daemon state
        self._hotkey_listener: Any = None
        self._hotkey_state: Any = None
        # Public-run mode: per-session agent isolation.
        self._public_agents: dict[str, Agent] = {}
        self._public_agent_locks: dict[str, asyncio.Lock] = {}
        # Tracks which WS each public agent is currently sending to.
        self._public_active_ws: dict[str, web.WebSocketResponse] = {}
        # Pending playbook approval requests: request_id → (event, [result_bool])
        self._pending_playbook_approvals: dict[str, tuple[asyncio.Event, list[bool]]] = {}

    async def _init_agent(self) -> None:
        """Initialize the agent with web callbacks."""
        self.agent = Agent(
            status_callback=self._status_callback,
            tool_output_callback=self._tool_output_callback,
            approval_callback=self._approval_callback,
            thinking_callback=self._thinking_callback,
            tool_stream_callback=self._tool_stream_callback,
        )
        self.agent.response_stream_callback = self._response_stream_callback
        self.agent.playbook_approval_callback = self._playbook_approval_callback
        self.agent.peer_consult_approval_callback = self._peer_consult_approval_callback
        self.agent.ws_broadcast = self._broadcast
        await self.agent.initialize()

        # Give the game registry access to the LLM provider for AgentSeat.
        from captain_claw.games.registry import get_registry
        get_registry().set_provider(self.agent.provider)

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

        # Prime GDrive tree cache for context injection (non-blocking).
        if hasattr(self.agent, "_refresh_gdrive_trees"):
            asyncio.ensure_future(self.agent._refresh_gdrive_trees())

        # Wire sister session broadcast callback for WebSocket notifications.
        try:
            from captain_claw.sister_session import set_broadcast_callback
            set_broadcast_callback(self._broadcast)
        except Exception:
            pass

    # ── Public-mode per-session agents ──────────────────────────────

    async def _get_public_agent(self, session_id: str) -> Agent:
        """Return (or lazily create) an Agent for a public session.

        Each public session gets its own Agent instance so that multiple
        users can chat concurrently without interference.  The agents
        share the same LLM provider as the main agent but maintain
        independent sessions, tool registries, and instruction caches.
        """
        agent = self._public_agents.get(session_id)
        if agent is not None:
            return agent

        # Serialise creation per session_id to avoid double-init.
        lock = self._public_agent_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            # Re-check after acquiring the lock.
            agent = self._public_agents.get(session_id)
            if agent is not None:
                return agent

            from captain_claw.session import get_session_manager

            sm = get_session_manager()
            session = await sm.load_session(session_id)
            if session is None:
                raise ValueError(f"Public session {session_id} not found")

            # Build WS-scoped callbacks that send only to the active WS
            # for this session (stored in _public_active_ws[session_id]).
            def _make_send(sid: str):
                """Create a sender closure bound to a session id."""
                def _send_msg(msg: dict) -> None:
                    ws = self._public_active_ws.get(sid)
                    if ws is not None:
                        fire_and_forget_send(ws, json.dumps(msg, default=str))
                return _send_msg

            send = _make_send(session_id)

            def status_cb(status: str) -> None:
                send({"type": "status", "status": status})

            def thinking_cb(text: str, tool: str = "", phase: str = "tool") -> None:
                send({"type": "thinking", "text": text, "tool": tool, "phase": phase})

            def tool_stream_cb(chunk: str) -> None:
                send({"type": "tool_stream", "chunk": chunk})

            def response_stream_cb(text: str) -> None:
                send({"type": "response_stream", "text": text})

            def tool_output_cb(tool_name: str, arguments: dict, output: str) -> None:
                send({
                    "type": "monitor",
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "output": output,
                })
                normalized = str(tool_name or "").strip().lower()
                if normalized == "task_rephrase":
                    send({
                        "type": "chat_message",
                        "role": "rephrase",
                        "content": str(output or ""),
                    })
                if normalized in ("image_gen", "termux", "browser") and output and not str(output).strip().lower().startswith("error"):
                    from captain_claw.platform_adapter import extract_image_paths_from_tool_output
                    for img_path in extract_image_paths_from_tool_output(str(output)):
                        send({"type": "chat_message", "role": "image", "content": str(img_path)})
                if normalized == "write" and output and not str(output).strip().lower().startswith("error"):
                    import re as _re
                    html_match = _re.search(r"to\s+(\S+\.(?:html|htm|svg))", str(output))
                    if html_match:
                        send({"type": "chat_message", "role": "html_file", "content": html_match.group(1).strip()})
                if normalized not in self._THINKING_SILENT_TOOLS:
                    from captain_claw.agent_tool_loop_mixin import AgentToolLoopMixin
                    summary = AgentToolLoopMixin._tool_thinking_summary(tool_name, arguments or {})
                    raw = str(output or "")
                    truncated = raw[:3000]
                    if len(raw) > 3000:
                        truncated += f"\n... [{len(raw)} total chars]"
                    send({"type": "tool_output_inline", "tool": tool_name, "summary": summary, "output": truncated})

            def approval_cb(message: str) -> bool:
                send({"type": "approval_notice", "message": message})
                return True

            # Playbook approval callback scoped to this public session's WS.
            async def _public_playbook_approval(message: str, _send=send) -> bool:
                import uuid as _uuid
                request_id = str(_uuid.uuid4())
                event = asyncio.Event()
                result_holder: list[bool] = [True]  # default: approve on timeout
                self._pending_playbook_approvals[request_id] = (event, result_holder)
                _send({
                    "type": "approval_request",
                    "id": request_id,
                    "message": message,
                    "category": "playbook",
                })
                try:
                    await asyncio.wait_for(event.wait(), timeout=60.0)
                except asyncio.TimeoutError:
                    pass
                finally:
                    self._pending_playbook_approvals.pop(request_id, None)
                return result_holder[0]

            agent = Agent(
                provider=self.agent.provider if self.agent else None,
                status_callback=status_cb,
                tool_output_callback=tool_output_cb,
                approval_callback=approval_cb,
                thinking_callback=thinking_cb,
                tool_stream_callback=tool_stream_cb,
            )
            agent.response_stream_callback = response_stream_cb
            agent.playbook_approval_callback = _public_playbook_approval

            # Bind to the public session (skip default session loading).
            agent.session = session
            agent.session_manager = sm
            agent._sync_runtime_flags_from_session()
            agent._register_default_tools()
            agent.instructions = self.agent.instructions if self.agent else agent.instructions
            agent._initialized = True
            agent._byok_active = False

            # Warm up context caches that initialize() would normally populate.
            try:
                await agent._refresh_insights_context_cache()
            except Exception:
                pass
            try:
                await agent._refresh_nervous_system_cache()
            except Exception:
                pass
            try:
                await agent._refresh_briefing_context_cache()
            except Exception:
                pass

            self._public_agents[session_id] = agent
            log.info("Created public agent", session_id=session_id, session_name=session.name)
            return agent

    # ── Callbacks ─────────────────────────────────────────────────────

    def _status_callback(self, status: str) -> None:
        """Broadcast status updates to all connected clients."""
        self._broadcast({"type": "status", "status": status})

    def _thinking_callback(self, text: str, tool: str = "", phase: str = "tool") -> None:
        """Broadcast inline thinking/reasoning updates to all connected clients."""
        self._broadcast({"type": "thinking", "text": text, "tool": tool, "phase": phase})

    def _tool_stream_callback(self, chunk: str) -> None:
        """Broadcast a live tool output chunk to the thinking console."""
        self._broadcast({"type": "tool_stream", "chunk": chunk})

    def _response_stream_callback(self, text: str) -> None:
        """Broadcast LLM response content to the stream panel."""
        self._broadcast({"type": "response_stream", "text": text})

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
        # Auto-broadcast audio player when pocket_tts (or similar) produces a file.
        if normalized in ("pocket_tts", "tts") and output and not str(output).strip().lower().startswith("error"):
            import re
            audio_match = re.search(r"Path:\s*(\S+\.(?:mp3|wav|ogg|flac|m4a|aac))", str(output))
            if audio_match:
                audio_path = audio_match.group(1).strip()
                self._broadcast({
                    "type": "chat_message",
                    "role": "audio",
                    "content": audio_path,
                })
                # Play audio on the server machine (works even when browser tab is unfocused).
                _play_audio_local(audio_path)
        # Auto-broadcast HTML view card when write tool produces an HTML/SVG file.
        if normalized == "write" and output and not str(output).strip().lower().startswith("error"):
            import re as _re
            html_match = _re.search(r"to\s+(\S+\.(?:html|htm|svg))", str(output))
            if html_match:
                self._broadcast({
                    "type": "chat_message",
                    "role": "html_file",
                    "content": html_match.group(1).strip(),
                })
        if normalized not in self._THINKING_SILENT_TOOLS:
            from captain_claw.agent_tool_loop_mixin import AgentToolLoopMixin
            summary = AgentToolLoopMixin._tool_thinking_summary(tool_name, arguments or {})
            # Send tool output inline to the thinking indicator in the chat pane.
            raw = str(output or "")
            truncated = raw[:3000]
            if len(raw) > 3000:
                truncated += f"\n... [{len(raw)} total chars]"
            self._broadcast({
                "type": "tool_output_inline",
                "tool": tool_name,
                "summary": summary,
                "output": truncated,
            })

    def _approval_callback(self, message: str) -> bool:
        """Handle tool approval requests from the agent."""
        self._broadcast({
            "type": "approval_notice",
            "message": message,
        })
        return True

    # ── Playbook approval (blocking async) ────────────────────────────

    async def _playbook_approval_callback(self, message: str) -> bool:
        """Async blocking approval for playbook usage.

        Broadcasts an ``approval_request`` message to all clients and waits
        for a response.  Falls back to auto-approve after 15 s timeout.
        """
        import uuid
        request_id = str(uuid.uuid4())
        event = asyncio.Event()
        result_holder: list[bool] = [True]  # default: approve on timeout
        self._pending_playbook_approvals[request_id] = (event, result_holder)

        self._broadcast({
            "type": "approval_request",
            "id": request_id,
            "message": message,
            "category": "playbook",
        })

        try:
            await asyncio.wait_for(event.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            # Auto-approve on timeout so automated tasks don't hang forever.
            result_holder[0] = True
        finally:
            self._pending_playbook_approvals.pop(request_id, None)

        return result_holder[0]

    async def _peer_consult_approval_callback(self, message: str) -> bool:
        """Async blocking approval for peer agent consultation.

        Broadcasts an ``approval_request`` message with category ``peer_consult``
        to all clients and waits for a response.  Falls back to auto-approve
        after 60 s timeout.
        """
        import uuid
        request_id = str(uuid.uuid4())
        event = asyncio.Event()
        result_holder: list[bool] = [True]  # default: approve on timeout
        self._pending_playbook_approvals[request_id] = (event, result_holder)

        self._broadcast({
            "type": "approval_request",
            "id": request_id,
            "message": message,
            "category": "peer_consult",
        })

        try:
            await asyncio.wait_for(event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            result_holder[0] = True
        finally:
            self._pending_playbook_approvals.pop(request_id, None)

        return result_holder[0]

    def resolve_playbook_approval(self, request_id: str, approved: bool) -> None:
        """Called by ws_handler when the client responds to an approval request."""
        pending = self._pending_playbook_approvals.get(request_id)
        if pending:
            event, result_holder = pending
            result_holder[0] = approved
            event.set()

    # ── Broadcast / Send ──────────────────────────────────────────────

    def _broadcast(self, msg: dict[str, Any]) -> None:
        """Send a message to all connected *admin* WebSocket clients.

        In public-run mode, public agents use their own per-session
        callbacks that send directly to the user's WebSocket — so this
        method only needs to reach admin connections.
        """
        data = json.dumps(msg, default=str)
        stale: list[web.WebSocketResponse] = []

        from captain_claw.config import get_config
        public_mode = bool(get_config().web.public_run)

        for ws in self.clients:
            if ws.closed:
                stale.append(ws)
                continue
            # In public mode, only broadcast to admin connections.
            if public_mode and not getattr(ws, "_is_admin", False):
                continue
            try:
                fire_and_forget_send(ws, data)
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
            "force_script": bool(getattr(self.agent, "_force_script_mode", False)),
        }

    # ── Cron runtime context ─────────────────────────────────────────

    def _get_web_runtime_context(self) -> "RuntimeContext":
        """Create a lightweight RuntimeContext for web-mode cron execution."""
        from captain_claw.runtime_context import RuntimeContext

        class _WebCronUI:
            """Minimal TerminalUI stand-in for web-mode cron execution."""

            def print_message(self, role: str, content: str) -> None:
                pass

            def print_blank_line(self) -> None:
                pass

            def print_error(self, error: str) -> None:
                log.error("WebCronUI error", error=error)

            def set_runtime_status(self, status: str) -> None:
                pass

            def set_thinking(self, text: str, phase: str = "") -> None:
                pass

            def append_tool_output(
                self, tool_name: str, arguments: object, output: str
            ) -> None:
                pass

            def append_system_line(self, text: str) -> None:
                pass

            def begin_assistant_stream(self) -> None:
                pass

            def end_assistant_stream(self) -> None:
                pass

            def print_streaming(self, chunk: str) -> None:
                pass

            def complete_stream_line(self) -> None:
                pass

            def print_next_steps(self, steps: list[dict[str, str]]) -> None:
                pass

            def can_capture_escape(self) -> bool:
                return False

            def confirm(self, message: str) -> bool:
                return True

        if not self.agent:
            raise RuntimeError("Agent not initialized")

        server = self

        async def _cron_output_to_telegram(session_id: str, text: str) -> None:
            """Deliver cron output to Telegram if the session belongs to a TG user."""
            bridge = server._telegram_bridge
            if not bridge:
                return
            # Reverse lookup: find user_id whose session matches.
            for user_id, sid in server._telegram_user_sessions.items():
                if sid == session_id:
                    user_info = server._approved_telegram_users.get(user_id, {})
                    chat_id = int(user_info.get("chat_id", 0) or 0)
                    if chat_id:
                        await bridge.send_message(chat_id, f"*Cron result:*\n{text}")
                    break

        return RuntimeContext(
            agent=self.agent,
            ui=_WebCronUI(),  # type: ignore[arg-type]
            on_cron_output=_cron_output_to_telegram,
        )

    # ── Delegated handlers ───────────────────────────────────────────
    # Each handler delegates to a function in the captain_claw.web package.

    # WebSocket
    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        from captain_claw.web.ws_handler import ws_handler
        return await ws_handler(self, request)

    async def ws_stt_handler(self, request: web.Request) -> web.WebSocketResponse:
        from captain_claw.web.ws_stt import handle_stt_ws
        return await handle_stt_ws(self, request)

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

    async def _prepare_tasks(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import prepare_tasks
        return await prepare_tasks(self, request)

    async def _run_tasks(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import run_tasks
        return await run_tasks(self, request)

    async def _get_workspace_snapshot(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import get_workspace_snapshot
        return await get_workspace_snapshot(self, request)

    async def _get_traces(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_orchestrator import get_traces
        return await get_traces(self, request)

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

    async def _import_contacts(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import import_contacts
        return await import_contacts(self, request)

    async def _preview_contacts_import(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_entities import preview_contacts_import
        return await preview_contacts_import(self, request)

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

    async def _rephrase_personality_field(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_personality import rephrase_personality_field
        return await rephrase_personality_field(self, request)

    # Visualization style
    async def _get_visualization_style(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_visualization_style import get_visualization_style
        return await get_visualization_style(self, request)

    async def _put_visualization_style(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_visualization_style import put_visualization_style
        return await put_visualization_style(self, request)

    async def _analyze_visualization_style(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_visualization_style import analyze_visualization_style
        return await analyze_visualization_style(self, request)

    async def _rephrase_visualization_style(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_visualization_style import rephrase_visualization_style
        return await rephrase_visualization_style(self, request)

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

    async def _view_file(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_files import view_file
        return await view_file(self, request)

    async def _delete_files(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_files import delete_files
        return await delete_files(self, request)

    async def _export_md(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_files import export_md
        return await export_md(self, request)

    async def _serve_media(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_files import serve_media
        return await serve_media(self, request)

    async def _image_upload(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_image_upload import upload_image
        return await upload_image(self, request)

    async def _file_upload(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_file_upload import upload_file
        return await upload_file(self, request)

    async def _audio_transcribe(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_audio_transcribe import transcribe_audio_handler
        return await transcribe_audio_handler(self, request)

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

    async def _auth_google_config_get(self, request: web.Request) -> web.Response:
        from captain_claw.web.google_oauth import auth_google_config_get
        return await auth_google_config_get(self, request)

    async def _auth_google_config_post(self, request: web.Request) -> web.Response:
        from captain_claw.web.google_oauth import auth_google_config_post
        return await auth_google_config_post(self, request)

    async def _auth_google_cors_preflight(self, request: web.Request) -> web.Response:
        from captain_claw.web.google_oauth import auth_google_cors_preflight
        return await auth_google_cors_preflight(self, request)

    # Static pages
    async def _serve_home(self, request: web.Request) -> web.Response:
        if self.config.web.public_run:
            section = self.config.web.public_run.strip().lower()
            raise web.HTTPFound(location=f"/{section}")
        from captain_claw.web.static_pages import serve_home
        return await serve_home(self, request)

    async def _serve_chat(self, request: web.Request) -> web.Response:
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

    # ── Semantic memory browser ────────────────────────────

    async def _serve_semantic_memory(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_semantic_memory
        return await serve_semantic_memory(self, request)

    async def _sm_status(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_semantic_memory import get_status
        return await get_status(self, request)

    async def _sm_documents(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_semantic_memory import list_documents
        return await list_documents(self, request)

    async def _sm_document_detail(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_semantic_memory import get_document_chunks
        return await get_document_chunks(self, request)

    async def _sm_search(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_semantic_memory import search
        return await search(self, request)

    async def _sm_promote(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_semantic_memory import promote
        return await promote(self, request)

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

    async def _serve_insights(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_insights
        return await serve_insights(self, request)

    async def _serve_intuitions(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_intuitions
        return await serve_intuitions(self, request)

    async def _serve_playbooks(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_playbooks
        return await serve_playbooks(self, request)

    async def _serve_playbook_wizard(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_playbook_wizard
        return await serve_playbook_wizard(self, request)

    async def _serve_skills(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_skills
        return await serve_skills(self, request)

    async def _serve_usage(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_usage
        return await serve_usage(self, request)

    async def _serve_reflections(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_reflections
        return await serve_reflections(self, request)

    async def _serve_computer(self, request: web.Request) -> web.StreamResponse:
        from captain_claw.web.static_pages import serve_computer
        return await serve_computer(self, request)

    async def _serve_personality(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_personality
        return await serve_personality(self, request)

    async def _serve_briefings(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_briefings
        return await serve_briefings(self, request)

    async def _serve_brain_graph(self, request: web.Request) -> web.Response:
        from captain_claw.web.static_pages import serve_brain_graph
        return await serve_brain_graph(self, request)

    async def _bg_get_data(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_brain_graph import get_graph_data
        return await get_graph_data(self, request)

    async def _bg_get_message(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_brain_graph import get_message_content
        return await get_message_content(self, request)

    async def _get_version(self, request: web.Request) -> web.Response:
        from captain_claw import __version__, __build_date__
        return web.json_response({
            "version": __version__,
            "build_date": __build_date__,
            "name": "Captain Claw",
        })

    # ── Public session API ──────────────────────────────────────────

    async def _public_session_new(self, request: web.Request) -> web.Response:
        from captain_claw.web.public_session import create_public_session, set_public_cookie
        from captain_claw.web.auth import _is_behind_tls
        session, code = await create_public_session(self)
        meta = session.metadata or {}
        resp = web.json_response({
            "ok": True, "session_id": session.id, "code": code,
            "session_settings": {
                "session_name": meta.get("session_display_name", ""),
                "session_description": meta.get("session_description", ""),
                "session_instructions": meta.get("session_instructions", ""),
            },
        })
        set_public_cookie(resp, session.id, code, self.config.web.auth_token, secure=_is_behind_tls(request))
        return resp

    async def _public_session_resume(self, request: web.Request) -> web.Response:
        from captain_claw.web.public_session import load_session_by_code, set_public_cookie
        from captain_claw.web.auth import _is_behind_tls
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad_request", "message": "Invalid JSON"}, status=400)
        code = str(body.get("code", "")).strip().upper()
        if len(code) != 6:
            return web.json_response({"error": "invalid_code", "message": "Code must be exactly 6 characters."}, status=400)
        session = await load_session_by_code(code)
        if session is None:
            return web.json_response({"error": "not_found", "message": "No session found for this code."}, status=404)
        meta = session.metadata or {}
        resp = web.json_response({
            "ok": True, "session_id": session.id,
            "session_settings": {
                "session_name": meta.get("session_display_name", ""),
                "session_description": meta.get("session_description", ""),
                "session_instructions": meta.get("session_instructions", ""),
            },
        })
        set_public_cookie(resp, session.id, code, self.config.web.auth_token, secure=_is_behind_tls(request))
        return resp

    async def _public_session_enter(self, request: web.Request) -> web.Response:
        """GET /api/public/session/enter?code=XXXXXX — validate code, set cookie, redirect."""
        from captain_claw.web.public_session import load_session_by_code, set_public_cookie
        from captain_claw.web.auth import _is_behind_tls
        code = request.query.get("code", "").strip().upper()
        if len(code) != 6:
            raise web.HTTPFound(location="/computer")
        session = await load_session_by_code(code)
        if session is None:
            raise web.HTTPFound(location="/computer")
        resp = web.HTTPFound(location="/computer")
        set_public_cookie(resp, session.id, code, self.config.web.auth_token, secure=_is_behind_tls(request))
        return resp

    async def _public_session_settings(self, request: web.Request) -> web.Response:
        """PATCH /api/public/session/settings — save session name/description/instructions."""
        from captain_claw.web.public_session import read_public_cookie
        from captain_claw.web.public_auth import _is_admin
        from captain_claw.session import get_session_manager
        is_admin = _is_admin(request, self.config.web)
        identity = read_public_cookie(request, self.config.web.auth_token)
        if identity is None and not is_admin:
            return web.json_response({"error": "unauthorized"}, status=401)
        session_id = identity[0] if identity else None
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad_request"}, status=400)
        # Admin can target any session by id.
        if is_admin and "session_id" in body:
            session_id = body["session_id"]
        if not session_id:
            return web.json_response({"error": "no_session"}, status=400)
        sm = get_session_manager()
        session = await sm.load_session(session_id)
        if session is None:
            return web.json_response({"error": "not_found"}, status=404)
        # Enforce lock for non-admin users.
        if not is_admin and (session.metadata or {}).get("session_settings_locked", False):
            return web.json_response({"error": "locked", "message": "Session settings are locked."}, status=403)
        if session.metadata is None:
            session.metadata = {}
        changed = False
        if "session_name" in body and isinstance(body["session_name"], str):
            session.metadata["session_display_name"] = body["session_name"].strip()
            changed = True
        if "session_description" in body and isinstance(body["session_description"], str):
            session.metadata["session_description"] = body["session_description"].strip()
            changed = True
        if "session_instructions" in body and isinstance(body["session_instructions"], str):
            session.metadata["session_instructions"] = body["session_instructions"].strip()
            changed = True
        # Admin-only: lock/unlock settings.
        if is_admin and "locked" in body:
            session.metadata["session_settings_locked"] = bool(body["locked"])
            changed = True
        if changed:
            await sm.save_session(session)
        return web.json_response({"ok": True})

    async def _public_session_logout(self, request: web.Request) -> web.Response:
        from captain_claw.web.public_session import COOKIE_NAME
        resp = web.json_response({"ok": True})
        resp.del_cookie(COOKIE_NAME, path="/")
        return resp

    # ── Admin session management ──────────────────────────────────

    async def _serve_public_sessions(self, request: web.Request) -> web.StreamResponse:
        static_dir = Path(__file__).parent / "web" / "static"
        return web.FileResponse(static_dir / "public_sessions.html")

    async def _admin_list_sessions(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_public_admin import list_public_sessions
        return await list_public_sessions(self, request)

    async def _admin_create_session(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_public_admin import create_public_session
        return await create_public_session(self, request)

    async def _admin_bulk_create(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_public_admin import bulk_create_sessions
        return await bulk_create_sessions(self, request)

    async def _admin_update_session(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_public_admin import update_public_session
        return await update_public_session(self, request)

    async def _admin_delete_session(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_public_admin import delete_public_session
        return await delete_public_session(self, request)

    async def _computer_visualize(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_computer import computer_visualize
        return await computer_visualize(self, request)

    async def _computer_visualize_stream(self, request: web.Request) -> web.StreamResponse:
        from captain_claw.web.rest_computer import computer_visualize_stream
        return await computer_visualize_stream(self, request)

    async def _exploration_save(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_computer import exploration_save
        return await exploration_save(self, request)

    async def _exploration_list(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_computer import exploration_list
        return await exploration_list(self, request)

    async def _exploration_get(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_computer import exploration_get
        return await exploration_get(self, request)

    async def _exploration_update_visual(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_computer import exploration_update_visual
        return await exploration_update_visual(self, request)

    async def _exploration_delete(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_computer import exploration_delete
        return await exploration_delete(self, request)

    async def _export_visual_pdf(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_computer import export_visual_pdf
        return await export_visual_pdf(self, request)

    async def _list_reflections(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_reflections import list_reflections_api
        return await list_reflections_api(self, request)

    async def _get_latest_reflection(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_reflections import get_latest_reflection
        return await get_latest_reflection(self, request)

    async def _generate_reflection(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_reflections import trigger_reflection
        return await trigger_reflection(self, request)

    async def _update_reflection(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_reflections import update_reflection_api
        return await update_reflection_api(self, request)

    async def _delete_reflection(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_reflections import delete_reflection_api
        return await delete_reflection_api(self, request)

    async def _export_memory(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_memory_transfer import export_memory
        return await export_memory(self, request)

    async def _import_memory(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_memory_transfer import import_memory
        return await import_memory(self, request)

    async def _list_imported_reflections(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_memory_transfer import list_imported
        return await list_imported(self, request)

    async def _merge_reflection(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_memory_transfer import merge_reflection
        return await merge_reflection(self, request)

    async def _list_semantic_imports(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_memory_transfer import list_semantic_imports
        return await list_semantic_imports(self, request)

    async def _delete_semantic_import(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_memory_transfer import delete_semantic_import
        return await delete_semantic_import(self, request)

    async def _list_pending_insights(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_insights_pending import list_pending
        return await list_pending(self, request)

    async def _count_pending_insights(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_insights_pending import count_pending
        return await count_pending(self, request)

    async def _approve_pending_insight(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_insights_pending import approve_pending
        return await approve_pending(self, request)

    async def _reject_pending_insight(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_insights_pending import reject_pending
        return await reject_pending(self, request)

    async def _clear_pending_insights(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_insights_pending import clear_pending
        return await clear_pending(self, request)

    async def _get_usage(self, request: web.Request) -> web.Response:
        """GET /api/usage — query LLM usage records with date filtering."""
        import json as _json
        from datetime import datetime, timedelta, timezone

        from captain_claw.session import get_session_manager
        sm = get_session_manager()
        period = request.query.get("period", "today")
        session_id = request.query.get("session_id")
        provider = request.query.get("provider")
        model = request.query.get("model")

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        period_map = {
            "last_hour": (now - timedelta(hours=1), now),
            "today": (today_start, now),
            "yesterday": (today_start - timedelta(days=1), today_start),
            "this_week": (today_start - timedelta(days=now.weekday()), now),
            "last_week": (
                today_start - timedelta(days=now.weekday() + 7),
                today_start - timedelta(days=now.weekday()),
            ),
            "this_month": (today_start.replace(day=1), now),
            "last_month": (
                (today_start.replace(day=1) - timedelta(days=1)).replace(day=1),
                today_start.replace(day=1),
            ),
            "all": (None, None),
        }

        since_dt, until_dt = period_map.get(period, period_map["today"])
        since = since_dt.isoformat() if since_dt else None
        until = until_dt.isoformat() if until_dt else None

        # Explicit overrides
        if "since" in request.query:
            since = request.query["since"]
        if "until" in request.query:
            until = request.query["until"]

        rows = await sm.query_llm_usage(
            since=since, until=until,
            session_id=session_id, provider=provider, model=model,
        )

        # Optional BYOK filter: "1" = BYOK only, "0" = server only.
        byok_filter = request.query.get("byok", "").strip()
        if byok_filter == "1":
            rows = [r for r in rows if r.get("byok")]
        elif byok_filter == "0":
            rows = [r for r in rows if not r.get("byok")]

        totals = {
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "total_tokens": sum(r["total_tokens"] for r in rows),
            "cache_creation_input_tokens": sum(r["cache_creation_input_tokens"] for r in rows),
            "cache_read_input_tokens": sum(r["cache_read_input_tokens"] for r in rows),
            "input_bytes": sum(r["input_bytes"] for r in rows),
            "output_bytes": sum(r["output_bytes"] for r in rows),
            "total_calls": len(rows),
            "avg_latency_ms": (
                sum(r["latency_ms"] for r in rows) // len(rows) if rows else 0
            ),
            "error_count": sum(1 for r in rows if r["error"]),
            "byok_calls": sum(1 for r in rows if r.get("byok")),
        }

        return web.json_response(
            {"totals": totals, "records": rows, "period": period,
             "since": since, "until": until},
            dumps=lambda obj: _json.dumps(obj, default=str),
        )

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

    async def _pb_export_one(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_playbooks import export_playbook
        return await export_playbook(self, request)

    async def _pb_export_all(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_playbooks import export_all_playbooks
        return await export_all_playbooks(self, request)

    async def _pb_import(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_playbooks import import_playbooks
        return await import_playbooks(self, request)

    async def _pb_wizard_step(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_playbook_wizard import wizard_step
        return await wizard_step(self, request)

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

    # MCP Connectors page
    async def _serve_mcp_connectors(self, request: web.Request) -> web.FileResponse:
        from captain_claw.web.static_pages import serve_mcp_connectors
        return await serve_mcp_connectors(self, request)

    async def _mcp_test(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_mcp import test_connection
        return await test_connection(self, request)

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

    # Insights REST
    async def _ins_list(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_insights import list_insights
        return await list_insights(self, request)

    async def _ins_get(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_insights import get_insight
        return await get_insight(self, request)

    async def _ins_create(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_insights import create_insight
        return await create_insight(self, request)

    async def _ins_update(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_insights import update_insight
        return await update_insight(self, request)

    async def _ins_delete(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_insights import delete_insight
        return await delete_insight(self, request)

    # Nervous System REST
    async def _ns_list(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_nervous_system import list_intuitions
        return await list_intuitions(self, request)

    async def _ns_get(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_nervous_system import get_intuition
        return await get_intuition(self, request)

    async def _ns_create(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_nervous_system import create_intuition
        return await create_intuition(self, request)

    async def _ns_update(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_nervous_system import update_intuition
        return await update_intuition(self, request)

    async def _ns_delete(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_nervous_system import delete_intuition
        return await delete_intuition(self, request)

    async def _ns_dream(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_nervous_system import trigger_dream
        return await trigger_dream(self, request)

    async def _ns_stats(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_nervous_system import get_stats
        return await get_stats(self, request)

    # Sister Session REST

    async def _ss_list_tasks(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sister_session import list_tasks
        return await list_tasks(self, request)

    async def _ss_get_task(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sister_session import get_task
        return await get_task(self, request)

    async def _ss_create_task(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sister_session import create_task
        return await create_task(self, request)

    async def _ss_delete_task(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sister_session import delete_task
        return await delete_task(self, request)

    async def _ss_list_briefings(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sister_session import list_briefings
        return await list_briefings(self, request)

    async def _ss_get_briefing(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sister_session import get_briefing
        return await get_briefing(self, request)

    async def _ss_update_briefing(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sister_session import update_briefing
        return await update_briefing(self, request)

    async def _ss_delete_briefing(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sister_session import delete_briefing
        return await delete_briefing(self, request)

    async def _ss_stats(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sister_session import get_stats
        return await get_stats(self, request)

    async def _ss_list_watches(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sister_session import list_watches
        return await list_watches(self, request)

    async def _ss_create_watch(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sister_session import create_watch
        return await create_watch(self, request)

    async def _ss_delete_watch(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_sister_session import delete_watch
        return await delete_watch(self, request)

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

    async def _get_codex_auth(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_onboarding import get_codex_auth
        return await get_codex_auth(self, request)

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

    async def _install_skill_upload(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import install_skill_upload
        return await install_skill_upload(self, request)

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

    async def _list_drives(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import list_drives
        return await list_drives(self, request)

    async def _gws_status(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import gws_status
        return await gws_status(self, request)

    async def _list_gdrive_folders(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import list_gdrive_folders
        return await list_gdrive_folders(self, request)

    async def _add_gdrive_folder(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import add_gdrive_folder
        return await add_gdrive_folder(self, request)

    async def _remove_gdrive_folder(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import remove_gdrive_folder
        return await remove_gdrive_folder(self, request)

    async def _browse_gdrive(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import browse_gdrive
        return await browse_gdrive(self, request)

    async def _get_folder_trees(self, request: web.Request) -> web.Response:
        from captain_claw.web.rest_skills import get_folder_trees
        return await get_folder_trees(self, request)

    async def _llm_complete(self, request: web.Request) -> web.Response:
        """Expose the local LLM provider for remote agent seats."""
        provider = getattr(getattr(self, "agent", None), "provider", None)
        if provider is None:
            return web.json_response({"ok": False, "error": "no LLM provider"}, status=503)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)

        from captain_claw.llm import Message, LLMResponse
        raw_messages = body.get("messages", [])
        messages = [Message(role=m["role"], content=m["content"]) for m in raw_messages]
        temperature = body.get("temperature")
        max_tokens = body.get("max_tokens")

        try:
            resp: LLMResponse = await provider.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return web.json_response({
                "ok": True,
                "content": resp.content,
                "model": resp.model,
                "usage": resp.usage,
                "finish_reason": resp.finish_reason,
            })
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    # ── App setup ────────────────────────────────────────────────────

    def create_app(self) -> web.Application:
        app = web.Application(client_max_size=50 * 1024 * 1024)  # 50 MB for file uploads
        if self.config.web.public_run:
            # Public mode: lock down routes to the allowed section.
            # Admin bypass via auth_token is handled inside the middleware.
            from captain_claw.web.public_auth import create_public_middleware
            app.middlewares.append(create_public_middleware(self.config.web))
        elif self.config.web.auth_token:
            from captain_claw.web.auth import create_auth_middleware
            app.middlewares.append(create_auth_middleware(self.config.web))
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_get("/ws/stt", self.ws_stt_handler)
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
        app.router.add_post("/api/skills/install-upload", self._install_skill_upload)
        app.router.add_post("/api/skills/toggle", self._toggle_skill)
        app.router.add_get("/api/read-folders", self._list_read_folders)
        app.router.add_post("/api/read-folders", self._add_read_folder)
        app.router.add_delete("/api/read-folders", self._remove_read_folder)
        app.router.add_get("/api/browse", self._browse_directory)
        app.router.add_get("/api/drives", self._list_drives)
        app.router.add_get("/api/gws-status", self._gws_status)
        app.router.add_get("/api/read-folders/gdrive", self._list_gdrive_folders)
        app.router.add_post("/api/read-folders/gdrive", self._add_gdrive_folder)
        app.router.add_delete("/api/read-folders/gdrive", self._remove_gdrive_folder)
        app.router.add_get("/api/read-folders/gdrive/browse", self._browse_gdrive)
        app.router.add_get("/api/folder-trees", self._get_folder_trees)
        # ── LLM proxy (for remote agent seats) ──
        app.router.add_post("/api/llm/complete", self._llm_complete)
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
        app.router.add_post("/api/orchestrator/prepare-tasks", self._prepare_tasks)
        app.router.add_post("/api/orchestrator/run-tasks", self._run_tasks)
        app.router.add_get("/api/orchestrator/workspace", self._get_workspace_snapshot)
        app.router.add_get("/api/orchestrator/traces", self._get_traces)
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
        app.router.add_post("/api/contacts/import", self._import_contacts)
        app.router.add_post("/api/contacts/import/preview", self._preview_contacts_import)
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
        app.router.add_post("/api/personality/rephrase", self._rephrase_personality_field)
        # Visualization style
        app.router.add_get("/api/visualization-style", self._get_visualization_style)
        app.router.add_put("/api/visualization-style", self._put_visualization_style)
        app.router.add_post("/api/visualization-style/analyze", self._analyze_visualization_style)
        app.router.add_post("/api/visualization-style/rephrase", self._rephrase_visualization_style)
        # Reflections
        app.router.add_get("/api/reflections", self._list_reflections)
        app.router.add_get("/api/reflections/latest", self._get_latest_reflection)
        app.router.add_post("/api/reflections/generate", self._generate_reflection)
        app.router.add_put("/api/reflections/{timestamp}", self._update_reflection)
        app.router.add_delete("/api/reflections/{timestamp}", self._delete_reflection)
        # Cross-machine memory transfer (curated layers + optional
        # text-only semantic chunks; the target re-embeds on import so
        # different embedding models across agents still work).
        app.router.add_get("/api/memory/export", self._export_memory)
        app.router.add_post("/api/memory/import", self._import_memory)
        app.router.add_get("/api/memory/reflections/imported", self._list_imported_reflections)
        app.router.add_post("/api/memory/reflections/merge", self._merge_reflection)
        app.router.add_get("/api/memory/semantic/labels", self._list_semantic_imports)
        app.router.add_delete("/api/memory/semantic/labels/{label}", self._delete_semantic_import)
        app.router.add_get("/api/insights/pending", self._list_pending_insights)
        app.router.add_get("/api/insights/pending/count", self._count_pending_insights)
        app.router.add_post("/api/insights/pending/{pending_id}/approve", self._approve_pending_insight)
        app.router.add_post("/api/insights/pending/{pending_id}/reject", self._reject_pending_insight)
        app.router.add_delete("/api/insights/pending", self._clear_pending_insights)
        app.router.add_get("/api/workflow-browser", self._list_workflow_outputs)
        app.router.add_get("/api/workflow-browser/output/{filename}", self._get_workflow_output)
        app.router.add_get("/api/files", self._list_files)
        app.router.add_get("/api/files/session/{session_id}", self._list_session_files)
        app.router.add_get("/api/files/content", self._get_file_content)
        app.router.add_get("/api/files/download", self._download_file)
        app.router.add_get("/api/files/view", self._view_file)
        app.router.add_post("/api/files/delete", self._delete_files)
        app.router.add_post("/api/files/export", self._export_md)
        app.router.add_get("/api/media", self._serve_media)
        app.router.add_post("/api/image/upload", self._image_upload)
        app.router.add_post("/api/file/upload", self._file_upload)
        app.router.add_post("/api/audio/transcribe", self._audio_transcribe)
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
        # Insights
        app.router.add_get("/api/insights", self._ins_list)
        app.router.add_post("/api/insights", self._ins_create)
        app.router.add_get("/api/insights/{id}", self._ins_get)
        app.router.add_patch("/api/insights/{id}", self._ins_update)
        app.router.add_delete("/api/insights/{id}", self._ins_delete)
        # Nervous System
        app.router.add_get("/api/nervous-system", self._ns_list)
        app.router.add_post("/api/nervous-system", self._ns_create)
        app.router.add_get("/api/nervous-system/stats", self._ns_stats)
        app.router.add_post("/api/nervous-system/dream", self._ns_dream)
        app.router.add_get("/api/nervous-system/{id}", self._ns_get)
        app.router.add_patch("/api/nervous-system/{id}", self._ns_update)
        app.router.add_delete("/api/nervous-system/{id}", self._ns_delete)
        # Sister Session
        app.router.add_get("/api/sister/tasks", self._ss_list_tasks)
        app.router.add_post("/api/sister/tasks", self._ss_create_task)
        app.router.add_get("/api/sister/stats", self._ss_stats)
        app.router.add_get("/api/sister/tasks/{id}", self._ss_get_task)
        app.router.add_delete("/api/sister/tasks/{id}", self._ss_delete_task)
        app.router.add_get("/api/briefings", self._ss_list_briefings)
        app.router.add_get("/api/briefings/{id}", self._ss_get_briefing)
        app.router.add_patch("/api/briefings/{id}", self._ss_update_briefing)
        app.router.add_delete("/api/briefings/{id}", self._ss_delete_briefing)
        app.router.add_get("/api/sister/watches", self._ss_list_watches)
        app.router.add_post("/api/sister/watches", self._ss_create_watch)
        app.router.add_delete("/api/sister/watches/{id}", self._ss_delete_watch)
        # Playbooks API
        app.router.add_get("/api/playbooks", self._pb_list)
        app.router.add_get("/api/playbooks/search", self._pb_search)
        app.router.add_get("/api/playbooks-export", self._pb_export_all)
        app.router.add_post("/api/playbooks-import", self._pb_import)
        app.router.add_post("/api/playbooks", self._pb_create)
        app.router.add_get("/api/playbooks/{id}", self._pb_get)
        app.router.add_get("/api/playbooks/{id}/export", self._pb_export_one)
        app.router.add_patch("/api/playbooks/{id}", self._pb_update)
        app.router.add_delete("/api/playbooks/{id}", self._pb_delete)
        # Playbook Wizard API
        app.router.add_post("/api/playbook-wizard/step", self._pb_wizard_step)
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
        # Semantic memory (SQLite) browser API
        app.router.add_get("/api/semantic-memory/status", self._sm_status)
        app.router.add_get("/api/semantic-memory/documents", self._sm_documents)
        app.router.add_get("/api/semantic-memory/documents/{doc_id}", self._sm_document_detail)
        app.router.add_get("/api/semantic-memory/search", self._sm_search)
        app.router.add_get("/api/semantic-memory/promote", self._sm_promote)
        if self.config.web.api_enabled and self._api_pool:
            app.router.add_post("/v1/chat/completions", self._api_chat_completions)
            app.router.add_get("/v1/models", self._api_list_models)
        app.router.add_get("/api/onboarding/status", self._get_onboarding_status)
        app.router.add_post("/api/onboarding/validate", self._post_onboarding_validate)
        app.router.add_post("/api/onboarding/save", self._post_onboarding_save)
        app.router.add_get("/api/onboarding/codex-auth", self._get_codex_auth)
        app.router.add_get("/auth/google/login", self._auth_google_login)
        app.router.add_get("/auth/google/callback", self._auth_google_callback)
        app.router.add_get("/auth/google/status", self._auth_google_status)
        app.router.add_post("/auth/google/logout", self._auth_google_logout)
        app.router.add_get("/auth/google/config", self._auth_google_config_get)
        app.router.add_post("/auth/google/config", self._auth_google_config_post)
        app.router.add_route("OPTIONS", "/auth/google/{tail:.*}", self._auth_google_cors_preflight)
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
            app.router.add_get("/insights", self._serve_insights)
            app.router.add_get("/intuitions", self._serve_intuitions)
            app.router.add_get("/playbooks", self._serve_playbooks)
            app.router.add_get("/playbook-wizard", self._serve_playbook_wizard)
            app.router.add_get("/browser-workflows", self._serve_browser_workflows)
            app.router.add_get("/direct-api-calls", self._serve_direct_api_calls)
            app.router.add_get("/mcp-connectors", self._serve_mcp_connectors)
            app.router.add_post("/api/mcp/test", self._mcp_test)
            app.router.add_get("/skills", self._serve_skills)
            app.router.add_get("/usage", self._serve_usage)
            app.router.add_get("/reflections", self._serve_reflections)
            app.router.add_get("/computer", self._serve_computer)
            app.router.add_get("/public-sessions", self._serve_public_sessions)
            app.router.add_get("/personality", self._serve_personality)
            app.router.add_get("/semantic-memory", self._serve_semantic_memory)
            app.router.add_get("/briefings", self._serve_briefings)
            app.router.add_get("/brain-graph", self._serve_brain_graph)
            app.router.add_get("/api/brain-graph", self._bg_get_data)
            app.router.add_get("/api/brain-graph/message/{msg_id}", self._bg_get_message)
            app.router.add_get("/api/version", self._get_version)
            app.router.add_post("/api/computer/visualize", self._computer_visualize)
            app.router.add_post("/api/computer/visualize/stream", self._computer_visualize_stream)
            app.router.add_post("/api/computer/exploration", self._exploration_save)
            app.router.add_get("/api/computer/exploration", self._exploration_list)
            app.router.add_get("/api/computer/exploration/{id}", self._exploration_get)
            app.router.add_put("/api/computer/exploration/{id}/visual", self._exploration_update_visual)
            app.router.add_delete("/api/computer/exploration/{id}", self._exploration_delete)
            app.router.add_post("/api/computer/export-pdf", self._export_visual_pdf)
            app.router.add_get("/api/usage", self._get_usage)
            app.router.add_get("/favicon.ico", self._serve_favicon)
        # Public session endpoints (always registered; middleware controls access).
        app.router.add_post("/api/public/session/new", self._public_session_new)
        app.router.add_post("/api/public/session/resume", self._public_session_resume)
        app.router.add_get("/api/public/session/enter", self._public_session_enter)
        app.router.add_patch("/api/public/session/settings", self._public_session_settings)
        app.router.add_post("/api/public/session/logout", self._public_session_logout)
        # Admin session management endpoints (admin auth enforced inside handlers).
        app.router.add_get("/api/public/admin/sessions", self._admin_list_sessions)
        app.router.add_post("/api/public/admin/sessions", self._admin_create_session)
        app.router.add_post("/api/public/admin/sessions/bulk", self._admin_bulk_create)
        app.router.add_patch("/api/public/admin/sessions/{id}", self._admin_update_session)
        app.router.add_delete("/api/public/admin/sessions/{id}", self._admin_delete_session)
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

    # Apply Old Man config overrides before anything else initializes.
    from captain_claw.old_man import (
        apply_old_man_config_overrides,
        is_old_man_enabled,
        print_old_man_banner,
        setup_old_man_session,
    )
    apply_old_man_config_overrides(config)

    print("Initializing Captain Claw agent...")
    await server._init_agent()

    # Set up Old Man supervisor session (tags metadata + injects instructions).
    if is_old_man_enabled(config):
        await setup_old_man_session(server.agent)
        print_old_man_banner(config)

    from captain_claw.web.telegram import start_telegram, stop_telegram
    await start_telegram(server)

    from captain_claw.web.hotkey_daemon import start_hotkey_daemon, stop_hotkey_daemon
    await start_hotkey_daemon(server)

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
                # Wrap main agent callbacks to also stream activity to BotPort.
                s_cb, t_cb, to_cb = bp_client.wrap_callbacks(
                    server.agent.status_callback,
                    server.agent.thinking_callback,
                    server.agent.tool_output_callback,
                )
                server.agent.status_callback = s_cb
                server.agent.thinking_callback = t_cb
                server.agent.tool_output_callback = to_cb
            print(f"  BotPort client connected to {config.botport.url}")
        except Exception as exc:
            log.warning("BotPort client failed to start", error=str(exc))
            print(f"  BotPort client failed: {exc}")

    app = server.create_app()
    runner = web.AppRunner(app)
    await runner.setup()

    import socket as _socket

    def _probe_port_free(_host: str, _port: int) -> bool:
        """Return True if (host, port) accepts a fresh bind right now."""
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                # No SO_REUSEADDR — we want a strict "is this free" answer.
                _s.bind((_host, _port))
        except OSError:
            return False
        return True

    host = config.web.host
    requested_port = config.web.port
    port = requested_port
    max_search = 200
    bound = False
    scanned: list[int] = []
    for _ in range(max_search):
        # Probe first so we skip ports that are clearly in use (another agent
        # listening, a stale docker binding, etc.) instead of feeding them to
        # aiohttp and parsing the OSError after the fact.
        if not _probe_port_free(host, port):
            scanned.append(port)
            port += 1
            continue
        try:
            site = web.TCPSite(runner, host, port)
            await site.start()
            bound = True
            break
        except OSError as exc:
            # TOCTOU: something grabbed the port between our probe and the
            # aiohttp bind. Move on to the next candidate.
            log.warning(
                "Port grabbed between probe and bind, trying next",
                port=port,
                error=str(exc),
            )
            scanned.append(port)
            port += 1

    if not bound:
        raise RuntimeError(
            f"No free TCP port found in range {requested_port}..{requested_port + max_search - 1} "
            f"(scanned {len(scanned)} ports)"
        )

    if port != requested_port:
        log.warning(
            "Port drifted from requested value",
            requested=requested_port,
            actual=port,
            skipped=scanned[:10],
        )

    # Reflect the actual bound port everywhere downstream code reads it.
    config.web.port = port

    # If we fell through to a different port, tell Flight Deck so it can update
    # its registry and re-broadcast the fleet membership to peer agents.
    if port != requested_port:
        _slug = os.environ.get("FD_AGENT_SLUG", "").strip()
        _fd_url = (os.environ.get("FD_URL", "") or os.environ.get("FD_INTERNAL_URL", "")).strip().rstrip("/")
        if _slug and _fd_url:
            async def _announce_port_with_retry():
                # Flight Deck may not be listening yet — it kicks off child
                # agents during its own startup lifespan, before uvicorn has
                # bound the socket. Retry with backoff so the registry still
                # converges to the correct port once FD is up.
                import httpx as _httpx
                _auth_token = config.web.auth_token or ""
                _delays = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 10.0, 10.0]
                for _attempt, _delay in enumerate(_delays, start=1):
                    try:
                        async with _httpx.AsyncClient(timeout=5.0) as _client:
                            _resp = await _client.post(
                                f"{_fd_url}/fd/processes/{_slug}/announce-port",
                                json={"port": port, "auth": _auth_token},
                            )
                        if _resp.status_code == 200:
                            log.info(
                                "Announced port to Flight Deck",
                                slug=_slug,
                                port=port,
                                requested=requested_port,
                                attempts=_attempt,
                            )
                            return
                        # 4xx/5xx that is NOT a connection error — log and
                        # keep retrying a few times in case FD is still
                        # warming up its routes.
                        log.warning(
                            "Flight Deck port announce rejected, will retry",
                            slug=_slug,
                            port=port,
                            status=_resp.status_code,
                            body=_resp.text[:200],
                            attempt=_attempt,
                        )
                    except Exception as _exc:
                        log.info(
                            "Flight Deck not reachable yet, retrying port announce",
                            slug=_slug,
                            error=str(_exc),
                            attempt=_attempt,
                        )
                    await asyncio.sleep(_delay)
                log.error(
                    "Gave up announcing drifted port to Flight Deck — registry is stale",
                    slug=_slug,
                    requested=requested_port,
                    actual=port,
                )

            # Fire-and-forget so startup isn't blocked by the retry loop.
            asyncio.create_task(_announce_port_with_retry())
        else:
            # No FD to announce to (manual launch). Still shout about the
            # drift so whoever started this agent notices their registry /
            # config is now out of sync with the bound port.
            log.warning(
                "Port fallback occurred but FD_AGENT_SLUG / FD_URL not set — "
                "any external registry pointing at this agent is now stale",
                requested=requested_port,
                actual=port,
            )

    print(f"\n  Captain Claw Web UI running at http://{host}:{port}")
    if config.web.public_run:
        section = config.web.public_run.strip().lower()
        print(f"  PUBLIC MODE: only /{section} is accessible to anonymous visitors")
        if config.web.auth_token:
            print(f"  Admin access via: http://{host}:{port}/?token=<your-token>")
    elif config.web.auth_token:
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
        server._runtime_ctx = ctx
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
    try:
        await stop_hotkey_daemon(server)
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
    import argparse as _ap
    parser = _ap.ArgumentParser(prog="captain-claw-web", description="Captain Claw Web Server")
    parser.add_argument("--public-run", default="", metavar="SECTION",
                        help="Run in public mode exposing only SECTION (e.g. 'computer')")
    parser.add_argument("--port", type=int, default=0, help="Override web server port")
    args, _ = parser.parse_known_args()

    configure_logging()

    cfg = Config.load()
    if args.public_run:
        cfg.web.public_run = args.public_run
    if args.port:
        cfg.web.port = args.port
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
