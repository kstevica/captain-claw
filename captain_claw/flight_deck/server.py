"""Flight Deck backend — Docker container & process management for Captain Claw agents."""

from __future__ import annotations

import os
import sys
import json
import secrets
import signal
import asyncio
import subprocess
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

import docker
import yaml
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import logging

from captain_claw.flight_deck.auth import get_current_user, get_optional_user, get_ws_user, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB


# ── Console logging: timestamps + ANSI colors ──────────────────────────
_ANSI = {
    "reset":   "\033[0m",
    "dim":     "\033[2m",
    "bold":    "\033[1m",
    "grey":    "\033[90m",
    "red":     "\033[31m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "blue":    "\033[34m",
    "magenta": "\033[35m",
    "cyan":    "\033[36m",
    "white":   "\033[37m",
    "br_red":  "\033[91m",
    "br_green":"\033[92m",
    "br_yellow":"\033[93m",
    "br_blue": "\033[94m",
    "br_cyan": "\033[96m",
}

_LEVEL_COLORS = {
    "DEBUG":    _ANSI["grey"],
    "INFO":     _ANSI["br_green"],
    "WARNING":  _ANSI["br_yellow"],
    "ERROR":    _ANSI["br_red"],
    "CRITICAL": _ANSI["bold"] + _ANSI["br_red"],
}


class _FDColorFormatter(logging.Formatter):
    """Date/time + colored level + dim logger name + message."""

    def __init__(self, use_color: bool):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level = record.levelname
        name = record.name
        msg = record.getMessage()
        if record.exc_info:
            msg = msg + "\n" + self.formatException(record.exc_info)
        if self.use_color:
            lvl_col = _LEVEL_COLORS.get(level, "")
            R = _ANSI["reset"]
            return (
                f"{_ANSI['grey']}[{ts}]{R} "
                f"{lvl_col}{level:<8}{R} "
                f"{_ANSI['cyan']}{name}{R}  "
                f"{msg}"
            )
        return f"[{ts}] {level:<8} {name}  {msg}"


def _configure_fd_logging() -> None:
    """Install our colored handler on the root logger and silence dupes."""
    use_color = sys.stderr.isatty() and os.environ.get("NO_COLOR", "") == ""
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_FDColorFormatter(use_color=use_color))
    root = logging.getLogger()
    # Replace any pre-existing handlers so we don't double-print.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # Make sure these loggers don't add their own handlers on top of ours.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "flight_deck"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    # Quiet down access log a touch — INFO level keeps it visible.
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


_configure_fd_logging()

class _KwargLoggerAdapter:
    """Thin wrapper around stdlib Logger that accepts structured kwargs.

    Multiple call sites in this file use a structlog-like API
    (`log.info("event", key=val, ...)`). The stdlib logger raises
    ``TypeError: Logger._log() got an unexpected keyword argument ...`` for
    those, which historically masked bugs (e.g. /announce-port returning 500
    on every call, leaving the registry stale). This adapter converts the
    kwargs into ``key=value key2=value2`` and appends them to the message
    so the existing call sites Just Work without churn.

    Reserved kwargs that the stdlib logger understands (``exc_info``,
    ``stack_info``, ``stacklevel``, ``extra``) are passed through unchanged.
    """

    _RESERVED = {"exc_info", "stack_info", "stacklevel", "extra"}

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _emit(self, level: int, msg, args, kwargs):
        passthrough = {k: kwargs.pop(k) for k in list(kwargs) if k in self._RESERVED}
        if kwargs:
            extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
            if isinstance(msg, str) and args:
                # Preserve old %-style formatting if used
                try:
                    msg = msg % args
                    args = ()
                except Exception:
                    pass
            msg = f"{msg}  {extras}" if msg else extras
        self._logger.log(level, msg, *args, **passthrough)

    def debug(self, msg=None, *args, **kwargs): self._emit(logging.DEBUG, msg, args, kwargs)
    def info(self, msg=None, *args, **kwargs): self._emit(logging.INFO, msg, args, kwargs)
    def warning(self, msg=None, *args, **kwargs): self._emit(logging.WARNING, msg, args, kwargs)
    def error(self, msg=None, *args, **kwargs): self._emit(logging.ERROR, msg, args, kwargs)
    def critical(self, msg=None, *args, **kwargs): self._emit(logging.CRITICAL, msg, args, kwargs)
    # Alias used by some libs
    warn = warning

    def __getattr__(self, name):
        # Fall through to the underlying logger for anything we don't override
        return getattr(self._logger, name)


log = _KwargLoggerAdapter(logging.getLogger("flight_deck"))
from captain_claw.flight_deck.rate_limiter import (
    check_api_rate_limit, check_spawn_rate_limit, check_agent_count_limit,
    load_plan_limits_from_db_sync,
)

def _resolve_static_dir() -> Path:
    """Resolve static dir, handling PyInstaller bundles where __file__ is at _internal/ root."""
    normal = Path(__file__).parent / "static"
    if normal.is_dir():
        return normal
    # PyInstaller: entry script lands in _internal/, data is in _internal/captain_claw/flight_deck/static/
    if getattr(sys, "_MEIPASS", None):
        bundled = Path(sys._MEIPASS) / "captain_claw" / "flight_deck" / "static"
        if bundled.is_dir():
            return bundled
    return normal

STATIC_DIR = _resolve_static_dir()

# ── Config ──

def _default_data_dir() -> str:
    """Use ~/.captain-claw/fd-data for standalone builds, ./fd-data otherwise."""
    if getattr(sys, "_MEIPASS", None):
        return str(Path.home() / ".captain-claw" / "fd-data")
    return "./fd-data"

DATA_DIR = Path(os.environ.get("FD_DATA_DIR", _default_data_dir())).resolve()
CONTAINER_LABEL = "flight-deck.managed"
OWNER_LABEL = "flight-deck.owner"
CC_IMAGE_DEFAULT = "kstevica/captain-claw:latest"
AUTH_ENABLED = os.environ.get("FD_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")


# ── Docker client ──

_client: docker.DockerClient | None = None


def get_docker() -> docker.DockerClient:
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


# ── Process registry ──

PROCESS_REGISTRY_FILE = DATA_DIR / ".processes.json"
_processes: dict[str, subprocess.Popen] = {}  # slug -> Popen

# Serialise process spawns so concurrent requests can't both pick the same
# port between the availability check and Popen. Lazily constructed in the
# running event loop (asyncio.Lock() at import time would bind to whichever
# loop happens to exist then).
_spawn_lock: asyncio.Lock | None = None

def _get_spawn_lock() -> asyncio.Lock:
    global _spawn_lock
    if _spawn_lock is None:
        _spawn_lock = asyncio.Lock()
    return _spawn_lock


class ProcessEntry(BaseModel):
    """Persisted metadata for a managed process agent."""
    slug: str
    name: str
    description: str = ""
    web_port: int
    web_auth: str = ""
    pid: int | None = None
    provider: str = ""
    model: str = ""


def _load_process_registry() -> dict[str, dict]:
    """Load process registry from disk.

    Uses an advisory shared flock so we don't observe a partially-written file
    while another coroutine / process is in the middle of `_save_process_registry`.
    Falls back to {} only if the file is genuinely missing or unparseable —
    NEVER on transient lock contention.
    """
    if not PROCESS_REGISTRY_FILE.is_file():
        return {}
    try:
        import fcntl as _fcntl
        with PROCESS_REGISTRY_FILE.open("r") as _f:
            try:
                _fcntl.flock(_f.fileno(), _fcntl.LOCK_SH)
            except OSError:
                pass  # flock unsupported on some FSes (NFS, etc.) — best effort
            data = _f.read()
        return json.loads(data) if data else {}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("process registry read failed", error=str(exc))
        return {}


def _save_process_registry(registry: dict[str, dict]):
    """Persist process registry to disk atomically.

    Writes to a sibling .tmp file then os.replace()s — a partial write can
    never leave the canonical file in a half-written state, and concurrent
    readers will always see either the previous or the new full snapshot.
    Also takes an exclusive flock around the rename for cross-process safety.
    """
    PROCESS_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(registry, indent=2)
    tmp_path = PROCESS_REGISTRY_FILE.with_suffix(PROCESS_REGISTRY_FILE.suffix + ".tmp")
    try:
        import fcntl as _fcntl
        with tmp_path.open("w") as _f:
            try:
                _fcntl.flock(_f.fileno(), _fcntl.LOCK_EX)
            except OSError:
                pass
            _f.write(payload)
            _f.flush()
            try:
                os.fsync(_f.fileno())
            except OSError:
                pass
        os.replace(str(tmp_path), str(PROCESS_REGISTRY_FILE))
    except OSError as exc:
        log.error("process registry write failed", error=str(exc))
        raise


def _process_is_alive(slug: str) -> bool:
    """Check if a managed process is still running."""
    proc = _processes.get(slug)
    if proc and proc.poll() is None:
        return True
    # Also check by PID from registry
    registry = _load_process_registry()
    entry = registry.get(slug)
    if entry and entry.get("pid"):
        try:
            os.kill(entry["pid"], 0)
            return True
        except (OSError, ProcessLookupError):
            pass
    return False


def _kill_pid(pid: int, timeout: float = 5.0):
    """Send SIGTERM to a PID and wait for it to die; SIGKILL if needed."""
    import time
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.3)
        except (OSError, ProcessLookupError):
            return
    # Still alive — force kill
    try:
        os.kill(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def _stop_all_processes():
    """Stop all managed process agents in parallel (called on FD shutdown)."""
    import threading
    registry = _load_process_registry()
    pids_to_kill: list[int] = []
    for slug, entry in registry.items():
        pid = entry.get("pid")
        if pid and _process_is_alive(slug):
            pids_to_kill.append(pid)
            entry["pid"] = None

    if pids_to_kill:
        # Send SIGTERM to all at once
        for pid in pids_to_kill:
            try:
                os.kill(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass

        # Wait for all in parallel threads
        def _wait_and_kill(pid: int, timeout: float = 5.0):
            import time
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                    time.sleep(0.2)
                except (OSError, ProcessLookupError):
                    return
                try:
                    os.kill(pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass

        threads = [threading.Thread(target=_wait_and_kill, args=(pid,)) for pid in pids_to_kill]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=6)

    _save_process_registry(registry)
    _processes.clear()


def _start_registered_process(slug: str, entry: dict) -> bool:
    """Start a single process agent from its registry entry. Returns True on success."""
    agent_dir = DATA_DIR / slug
    if not agent_dir.is_dir():
        return False

    web_port = entry.get("web_port", 24080)

    # Rebuild environment from .env file
    environment = dict(os.environ)
    env_file = agent_dir / ".env"
    if env_file.is_file():
        content = env_file.read_text().strip()
        if content:
            for line in content.split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    environment[k] = v

    environment["HOME"] = str(agent_dir / "data" / "home-config-parent")
    # Slug + URL for port-fallback callbacks. Without FD_URL the agent can't
    # announce a drifted port back to Flight Deck and the registry goes stale
    # (chat panel then 401s because FD proxies to the old port).
    environment["FD_AGENT_SLUG"] = slug
    if "FD_URL" not in environment:
        fd_port = os.environ.get("FD_PORT", "25080")
        environment["FD_URL"] = f"http://localhost:{fd_port}"

    log_file = agent_dir / "process.log"
    try:
        log_fh = open(log_file, "a")
        proc = subprocess.Popen(
            ["captain-claw-web", "--port", str(web_port)],
            cwd=str(agent_dir),
            env=environment,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _processes[slug] = proc
        entry["pid"] = proc.pid
        return True
    except Exception:
        return False


def _reattach_processes():
    """On startup, check registered processes and restart any that were running."""
    import time as _time

    registry = _load_process_registry()
    restarted = []
    skipped = []
    stagger_s = float(os.environ.get("FD_REATTACH_STAGGER_S", "0.3"))
    first_launch = True
    for slug, entry in registry.items():
        # Skip agents that were intentionally stopped by the user
        if entry.get("stopped"):
            skipped.append(slug)
            continue
        pid = entry.get("pid")
        if pid:
            try:
                os.kill(pid, 0)  # Still alive — just track it
                continue
            except (OSError, ProcessLookupError):
                pass
        # Process was registered but is dead — restart it. Stagger the
        # launches so concurrent port probes don't race and pick the same
        # fallback port.
        if entry.get("web_port"):
            if not first_launch and stagger_s > 0:
                _time.sleep(stagger_s)
            first_launch = False
            if _start_registered_process(slug, entry):
                restarted.append(slug)
            else:
                entry["pid"] = None
    _save_process_registry(registry)
    if restarted:
        print(f"Flight Deck: restarted {len(restarted)} process agent(s): {', '.join(restarted)}")
    if skipped:
        print(f"Flight Deck: skipped {len(skipped)} stopped process agent(s): {', '.join(skipped)}")


# ── App ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Defer reattach so uvicorn has time to actually bind its listening
    # socket. Otherwise the children we launch here race to call
    # /fd/processes/{slug}/announce-port before FD is accepting connections,
    # and their drift announcements get "connection refused".
    async def _deferred_reattach():
        await asyncio.sleep(1.0)
        await asyncio.to_thread(_reattach_processes)
    asyncio.create_task(_deferred_reattach())
    # Initialize database for auth & settings
    if AUTH_ENABLED:
        _fd_db = FlightDeckDB(DATA_DIR / "flight-deck.db")
        await _fd_db.init()
        set_auth_db(_fd_db)
        app.state.fd_db = _fd_db
        # Load admin-configured plan limits from DB
        plan_limits_raw = await _fd_db.get_system_setting("fd:plan-limits")
        load_plan_limits_from_db_sync(plan_limits_raw)
    yield
    # Shutdown: stop all managed process agents
    print("Flight Deck: stopping managed process agents...")
    _stop_all_processes()
    if AUTH_ENABLED and hasattr(app.state, "fd_db"):
        await app.state.fd_db.close()
    if _client:
        _client.close()


app = FastAPI(title="Flight Deck", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Log validation errors with full detail for debugging
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    log.error("Validation error on %s %s: %s", request.method, request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# ── Auth dependency that always uses Depends ──
# When AUTH_ENABLED is False, using `= None` instead of `= Depends(...)` causes
# FastAPI to treat the parameter as a body field, breaking request parsing.
# This wrapper ensures we always use Depends regardless of auth state.

async def _no_user() -> None:
    return None

_optional_user_dep = Depends(get_optional_user) if AUTH_ENABLED else Depends(_no_user)
_required_user_dep = Depends(get_current_user) if AUTH_ENABLED else Depends(_no_user)

# ── Auth & user routes ──

from captain_claw.flight_deck.auth_routes import router as auth_router
from captain_claw.flight_deck.settings_routes import router as settings_router
from captain_claw.flight_deck.chat_routes import router as chat_router
from captain_claw.flight_deck.admin_routes import router as admin_router
from captain_claw.flight_deck.council_routes import router as council_router
from captain_claw.flight_deck.google_oauth_routes import router as google_oauth_router
from captain_claw.flight_deck.codex_oauth_routes import router as codex_oauth_router

app.include_router(auth_router)
app.include_router(settings_router)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(council_router)
app.include_router(google_oauth_router)
app.include_router(codex_oauth_router)


# ── Auth dependency helper ──

def _optional_user():
    """Return a dependency that requires auth when enabled, skips when disabled."""
    if AUTH_ENABLED:
        return Depends(get_current_user)
    return None


async def _get_user_id(request: Request) -> str:
    """Extract user_id from request state (set by auth middleware). Returns '' when auth disabled."""
    return getattr(request.state, "user_id", "")


def _require_auth():
    """FastAPI dependency that enforces auth when FD_AUTH_ENABLED=true."""
    async def _dep(user: dict = Depends(get_current_user)):
        return user
    if AUTH_ENABLED:
        return Depends(_dep)
    return None


# ── Models ──

class AgentConfig(BaseModel):
    """Agent spawn configuration — matches the frontend form."""
    # Identity
    name: str = ""
    description: str = ""
    hostname: str = "captain-claw"
    image: str = CC_IMAGE_DEFAULT

    # LLM
    provider: str = "ollama"
    model: str = "minimax-m2.7:cloud"
    temperature: float = 0.7
    max_tokens: int = 32768
    provider_api_key: str = ""
    base_url: str = ""

    # BotPort
    botport_enabled: bool = True
    botport_url: str = ""
    botport_instance_name: str = ""
    botport_key: str = ""
    botport_secret: str = ""
    botport_max_concurrent: int = 5

    # Tools
    tools: list[str] = Field(default_factory=lambda: [
        "shell", "read", "write", "glob", "edit",
        "web_fetch", "web_search", "browser", "botport",
    ])

    # Web
    web_enabled: bool = True
    web_port: int = 24080
    web_auth_token: str = ""

    # Platforms
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    discord_enabled: bool = False
    discord_bot_token: str = ""
    slack_enabled: bool = False
    slack_bot_token: str = ""

    # Cognitive mode
    cognitive_mode: str = "neutra"

    # Docker
    network_mode: str = "host"
    restart_policy: str = "unless-stopped"
    extra_volumes: list[dict] = Field(default_factory=list)
    env_vars: list[dict] = Field(default_factory=list)

    # Ownership hint — used by internal callers (e.g. Old Man) that cannot
    # authenticate via JWT but need the spawned agent to inherit the owner.
    owner_hint: str = ""


class ContainerInfo(BaseModel):
    id: str
    name: str
    status: str
    image: str
    created: str
    agent_name: str = ""
    description: str = ""
    ports: dict = Field(default_factory=dict)
    web_port: int | None = None
    web_auth: str = ""


class ContainerActionResult(BaseModel):
    ok: bool
    container_id: str
    message: str = ""
    old_container_id: str = ""


# ── Helpers ──

def _slug(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9-]", "-", (name or "cc-agent").lower()).strip("-") or "cc-agent"


def _docker_host() -> str:
    """Return the hostname containers should use to reach the Docker host."""
    import platform
    # On macOS/Windows, host.docker.internal resolves to the host.
    # On Linux with host networking, localhost works; with bridge, use the gateway.
    if platform.system() in ("Darwin", "Windows"):
        return "host.docker.internal"
    return "127.0.0.1"


def _build_config_yaml(c: AgentConfig) -> str:
    """Generate config.yaml content from agent config."""
    dhost = _docker_host()
    cfg: dict = {
        "model": {
            "provider": c.provider,
            "model": c.model,
            "temperature": c.temperature,
            "max_tokens": c.max_tokens,
            "api_key": "",  # Key goes in .env
            "base_url": (
                c.base_url
                if c.base_url
                else (f"http://{dhost}:11434" if c.provider == "ollama" else "")
            ),
        },
        "context": {
            "max_tokens": 160000,
            "compaction_threshold": 0.8,
            "compaction_ratio": 0.4,
        },
        "memory": {
            "enabled": True,
            "path": "/home/claw/.captain-claw/memory.db",
            "index_workspace": True,
            "index_sessions": True,
            "embeddings": {
                "provider": "auto",
                "ollama_model": "nomic-embed-text",
                "ollama_base_url": f"http://{dhost}:11434",
                "fallback_to_local_hash": True,
            },
        },
        "tools": {
            "enabled": c.tools,
            "shell": {"timeout": 120, "default_policy": "ask"},
            "browser": {"headless": True, "viewport_width": 1280, "viewport_height": 720},
            "web_search": {"provider": "brave", "max_results": 5},
            "require_confirmation": ["shell", "write", "edit"],
        },
        "session": {"storage": "sqlite", "path": "/data/sessions/sessions.db", "auto_save": True},
        "workspace": {"path": "/data/workspace"},
        "web": {
            "enabled": c.web_enabled,
            "host": "0.0.0.0",
            "port": c.web_port,
            "api_enabled": True,
            "auth_token": c.web_auth_token,
        },
        "botport": {
            "enabled": c.botport_enabled,
            "url": c.botport_url,
            "instance_name": c.botport_instance_name or c.name or "default",
            "key": c.botport_key,
            "secret": c.botport_secret,
            "advertise_personas": True,
            "advertise_tools": True,
            "advertise_models": True,
            "max_concurrent": c.botport_max_concurrent,
            "reconnect_delay_seconds": 5.0,
            "heartbeat_interval_seconds": 30.0,
        },
        "telegram": {"enabled": c.telegram_enabled, "bot_token": c.telegram_bot_token},
        "discord": {"enabled": c.discord_enabled, "bot_token": c.discord_bot_token},
        "slack": {"enabled": c.slack_enabled, "bot_token": c.slack_bot_token},
        "logging": {"level": "INFO", "format": "console"},
        "cognitive_mode": {
            "enabled": True,
            "default_mode": c.cognitive_mode,
        },
    }
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _build_env(c: AgentConfig) -> str:
    lines: list[str] = []
    if c.provider_api_key:
        # Map provider to env var name
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "xai": "XAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_name = key_map.get(c.provider, "API_KEY")
        lines.append(f"{env_name}={c.provider_api_key}")
    # Ollama inside Docker needs to reach the host
    if c.provider == "ollama":
        dhost = _docker_host()
        lines.append(f"OLLAMA_BASE_URL=http://{dhost}:11434")
    for ev in c.env_vars:
        if ev.get("key"):
            lines.append(f"{ev['key']}={ev.get('value', '')}")
    return "\n".join(lines) + "\n" if lines else ""


def _localize_url(url: str) -> str:
    """Rewrite Docker-internal hostnames to localhost for process agents."""
    return url.replace("host.docker.internal", "localhost").replace("host.docker.internal", "127.0.0.1") if url else url


def _build_process_config_yaml(c: AgentConfig, agent_dir: Path) -> str:
    """Generate config.yaml for a pip-installed process agent (local paths)."""
    home_config = agent_dir / "data" / "home-config"
    # For process agents, rewrite Docker-internal URLs to localhost
    botport_url = _localize_url(c.botport_url)
    cfg: dict = {
        "model": {
            "provider": c.provider,
            "model": c.model,
            "temperature": c.temperature,
            "max_tokens": c.max_tokens,
            "api_key": "",  # Key goes in .env
            "base_url": (
                c.base_url
                if c.base_url
                else ("http://127.0.0.1:11434" if c.provider == "ollama" else "")
            ),
        },
        "context": {
            "max_tokens": 160000,
            "compaction_threshold": 0.8,
            "compaction_ratio": 0.4,
        },
        "memory": {
            "enabled": True,
            "path": str(home_config / "memory.db"),
            "index_workspace": True,
            "index_sessions": True,
            "embeddings": {
                "provider": "auto",
                "ollama_model": "nomic-embed-text",
                "ollama_base_url": "http://127.0.0.1:11434",
                "fallback_to_local_hash": True,
            },
        },
        "tools": {
            "enabled": c.tools,
            "shell": {"timeout": 120, "default_policy": "ask"},
            "browser": {"headless": True, "viewport_width": 1280, "viewport_height": 720},
            "web_search": {"provider": "brave", "max_results": 5},
            "require_confirmation": ["shell", "write", "edit"],
        },
        "session": {
            "storage": "sqlite",
            "path": str(agent_dir / "data" / "sessions" / "sessions.db"),
            "auto_save": True,
        },
        "workspace": {"path": str(agent_dir / "data" / "workspace")},
        "web": {
            "enabled": c.web_enabled,
            "host": "127.0.0.1",
            "port": c.web_port,
            "api_enabled": True,
            "auth_token": c.web_auth_token,
        },
        "botport": {
            "enabled": c.botport_enabled,
            "url": botport_url,
            "instance_name": c.botport_instance_name or c.name or "default",
            "key": c.botport_key,
            "secret": c.botport_secret,
            "advertise_personas": True,
            "advertise_tools": True,
            "advertise_models": True,
            "max_concurrent": c.botport_max_concurrent,
            "reconnect_delay_seconds": 5.0,
            "heartbeat_interval_seconds": 30.0,
        },
        "telegram": {"enabled": c.telegram_enabled, "bot_token": c.telegram_bot_token},
        "discord": {"enabled": c.discord_enabled, "bot_token": c.discord_bot_token},
        "slack": {"enabled": c.slack_enabled, "bot_token": c.slack_bot_token},
        "logging": {"level": "INFO", "format": "console"},
    }
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _container_info(c: docker.models.containers.Container) -> ContainerInfo:
    labels = c.labels or {}
    web_port_str = labels.get("flight-deck.web-port", "")
    # Try label first, then fall back to Docker port bindings
    web_port: int | None = int(web_port_str) if web_port_str else None
    if web_port is None:
        # Extract from Docker port mappings (e.g. {"24080/tcp": [{"HostPort": "24080"}]})
        docker_ports = c.attrs.get("NetworkSettings", {}).get("Ports", {}) or {}
        for container_port, bindings in docker_ports.items():
            if bindings:
                try:
                    web_port = int(bindings[0].get("HostPort", 0))
                    if web_port:
                        break
                except (ValueError, IndexError, TypeError):
                    pass
    return ContainerInfo(
        id=c.short_id,
        name=c.name,
        status=c.status,
        image=str(c.image.tags[0]) if c.image.tags else str(c.image.short_id),
        created=str(c.attrs.get("Created", "")),
        agent_name=labels.get("flight-deck.agent-name", ""),
        description=labels.get("flight-deck.description", ""),
        ports=c.attrs.get("NetworkSettings", {}).get("Ports", {}),
        web_port=web_port,
        web_auth=labels.get("flight-deck.web-auth", ""),
    )


def _find_container(container_id: str, owner_id: str = "") -> docker.models.containers.Container:
    client = get_docker()
    # Try by short ID, full ID, or name
    for c in client.containers.list(all=True, filters={"label": CONTAINER_LABEL}):
        if c.short_id == container_id or c.id == container_id or c.name == container_id:
            if AUTH_ENABLED and owner_id:
                if (c.labels or {}).get(OWNER_LABEL, "") != owner_id:
                    raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
            return c
    raise HTTPException(status_code=404, detail=f"Container {container_id} not found")


# ── Endpoints ──

@app.get("/fd/containers", response_model=list[ContainerInfo])
async def list_containers(request: Request, user: dict | None = _required_user_dep):
    """List all Flight Deck managed containers (filtered by owner when auth enabled)."""
    try:
        client = get_docker()
        containers = client.containers.list(all=True, filters={"label": CONTAINER_LABEL})
    except Exception:
        return []  # Docker not available (e.g. running inside a container)
    user_id = getattr(request.state, "user_id", "")
    if AUTH_ENABLED and user_id:
        containers = [c for c in containers if (c.labels or {}).get(OWNER_LABEL, "") == user_id]
    return [_container_info(c) for c in containers]


def _is_port_available(port: int) -> bool:
    """Check if a TCP port is available on localhost."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _schedule_fleet_notify(name: str, port: int, event: str = "joined", owner_id: str = ""):
    """Schedule a fleet notification as a background async task."""
    async def _run():
        # Give the new agent a moment to start up before notifying peers
        await asyncio.sleep(5)
        await _notify_fleet_change(name, port, event, owner_id=owner_id)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_run())
        else:
            asyncio.run(_run())
    except RuntimeError:
        pass  # Best-effort


@app.post("/fd/spawn", response_model=ContainerActionResult)
async def spawn_agent(config: AgentConfig, request: Request, user: dict | None = _optional_user_dep):
    """Spawn a new Captain Claw container."""
    # Check if docker spawn is allowed
    sys_cfg = await _get_system_config()
    docker_default = not os.environ.get("CAPTAIN_CLAW_DOCKER")
    if not sys_cfg.get("docker_spawn_enabled", docker_default):
        raise HTTPException(403, "Docker container spawning is disabled by the administrator.")
    # Rate limiting & agent count check
    if AUTH_ENABLED and user:
        check_api_rate_limit(user)
        check_spawn_rate_limit(user)
        # Count existing containers for this user
        client_tmp = get_docker()
        user_id = user["id"]
        owned = [c for c in client_tmp.containers.list(all=True, filters={"label": CONTAINER_LABEL})
                 if (c.labels or {}).get(OWNER_LABEL, "") == user_id]
        await check_agent_count_limit(user, len(owned))

    client = get_docker()
    slug = _slug(config.name)

    # Ensure port is available; find a free one if not
    if config.web_enabled and (config.web_port <= 0 or not _is_port_available(config.web_port)):
        config.web_port = _find_available_port(config.web_port if config.web_port > 0 else 24080)

    # Auto-generate auth token if none provided — prevents unauthenticated
    # direct access to agent ports bypassing Flight Deck.
    if config.web_enabled and not config.web_auth_token:
        config.web_auth_token = secrets.token_urlsafe(32)

    # Check for name collision
    try:
        existing = client.containers.get(slug)
        if existing.status == "running":
            raise HTTPException(400, f"Container '{slug}' already running. Stop it first or use a different name.")
        # Remove stopped container with same name
        existing.remove()
    except docker.errors.NotFound:
        pass

    # Prepare data directory
    agent_dir = DATA_DIR / slug
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "workspace").mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "skills").mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "home-config").mkdir(parents=True, exist_ok=True)

    # Write config files
    config_yaml = _build_config_yaml(config)
    (agent_dir / "config.yaml").write_text(config_yaml)
    # Also write into home-config so it takes precedence over any stale config
    # that CC's settings page may have written into ~/.captain-claw/config.yaml
    (agent_dir / "data" / "home-config" / "config.yaml").write_text(config_yaml)

    env_content = _build_env(config)
    (agent_dir / ".env").write_text(env_content)

    # Write cognitive mode file for the agent (Docker spawn).
    if config.cognitive_mode and config.cognitive_mode != "neutra":
        mode_file = agent_dir / "data" / "home-config" / "cognitive_mode.txt"
        mode_file.write_text(config.cognitive_mode, encoding="utf-8")

    # Build volume mounts
    # CC WORKDIR is /app — it loads ./config.yaml from CWD (/app/config.yaml)
    # and ~/.captain-claw/config.yaml (home dir overlay, highest priority).
    # Mount config to /app/config.yaml (CWD) so CC finds it on startup.
    # Mount home-config dir for memory.db and other persistent state.
    config_file = str(agent_dir / "config.yaml")
    volumes = {
        config_file: {"bind": "/app/config.yaml", "mode": "ro"},
        str(agent_dir / ".env"): {"bind": "/app/.env", "mode": "ro"},
        str(agent_dir / "data" / "home-config"): {"bind": "/home/claw/.captain-claw", "mode": "rw"},
        str(agent_dir / "data" / "workspace"): {"bind": "/data/workspace", "mode": "rw"},
        str(agent_dir / "data" / "sessions"): {"bind": "/data/sessions", "mode": "rw"},
        str(agent_dir / "data" / "skills"): {"bind": "/data/skills", "mode": "rw"},
    }
    for ev in config.extra_volumes:
        host = ev.get("host", "")
        container = ev.get("container", "")
        if host and container:
            volumes[host] = {"bind": container, "mode": "rw"}

    # Build environment
    environment: dict[str, str] = {}
    if env_content:
        for line in env_content.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                environment[k] = v
    for ev in config.env_vars:
        if ev.get("key"):
            environment[ev["key"]] = ev.get("value", "")

    # Resolve owner: authenticated user > owner_hint > infer from existing agents
    owner_id = getattr(request.state, "user_id", "") or config.owner_hint
    if not owner_id:
        # Fallback: inherit owner from an existing agent in the registry.
        registry = _load_process_registry()
        for _entry in registry.values():
            if _entry.get("owner"):
                owner_id = _entry["owner"]
                break
        if not owner_id:
            # Try Docker containers
            try:
                for c in client.containers.list(all=True, filters={"label": CONTAINER_LABEL}):
                    o = (c.labels or {}).get(OWNER_LABEL, "")
                    if o:
                        owner_id = o
                        break
            except Exception:
                pass

    # Pass owner ID so child agents can propagate ownership when spawning
    if owner_id:
        environment["FD_OWNER_ID"] = owner_id

    # Slug for port-fallback callbacks (Docker path)
    environment["FD_AGENT_SLUG"] = slug

    # Labels for tracking
    labels = {
        CONTAINER_LABEL: "true",
        OWNER_LABEL: owner_id,
        "flight-deck.agent-name": config.name or slug,
        "flight-deck.description": config.description or "",
        "flight-deck.image": config.image,
        "flight-deck.web-port": str(config.web_port) if config.web_enabled else "",
        "flight-deck.web-auth": config.web_auth_token or "",
    }

    # Security options
    security_opt = ["no-new-privileges:true", "seccomp:unconfined"]

    # Restart policy
    restart_map = {
        "unless-stopped": {"Name": "unless-stopped"},
        "always": {"Name": "always"},
        "on-failure": {"Name": "on-failure", "MaximumRetryCount": 5},
        "no": {"Name": ""},
    }
    restart = restart_map.get(config.restart_policy, {"Name": "unless-stopped"})

    # Port publishing — needed on macOS where host networking doesn't work.
    import platform
    ports: dict[str, int] = {}
    use_network: str | None = config.network_mode
    if platform.system() == "Darwin" and config.network_mode == "host":
        # macOS: host networking is a no-op, switch to default bridge + port mapping
        use_network = None
        if config.web_enabled:
            ports[f"{config.web_port}/tcp"] = config.web_port
    elif config.network_mode != "host":
        # Explicit bridge/custom network: publish web port
        if config.web_enabled:
            ports[f"{config.web_port}/tcp"] = config.web_port

    try:
        container = client.containers.run(
            image=config.image,
            name=slug,
            hostname=config.hostname or slug,
            detach=True,
            network_mode=use_network,
            ports=ports or None,
            volumes=volumes,
            environment=environment,
            labels=labels,
            security_opt=security_opt,
            cap_drop=["ALL"],
            cap_add=["CHOWN", "SETUID", "SETGID", "SYS_CHROOT"],
            tmpfs={"/tmp": "", "/run": ""},
            restart_policy=restart,
            stop_signal="SIGTERM",
        )
        port_info = f" on port {config.web_port}" if config.web_enabled else ""
        # Log usage
        if AUTH_ENABLED and user:
            db = app.state.fd_db
            await db.log_usage(user["id"], "agent_spawn", json.dumps({"agent": slug, "type": "container", "image": config.image}))
        # Notify other agents about the new peer (scoped to same owner)
        if config.web_enabled:
            _schedule_fleet_notify(config.name or slug, config.web_port, owner_id=owner_id)
        return ContainerActionResult(ok=True, container_id=container.short_id, message=f"Agent '{slug}' spawned{port_info}")
    except docker.errors.ImageNotFound:
        raise HTTPException(404, f"Docker image '{config.image}' not found. Pull it first.")
    except docker.errors.APIError as exc:
        raise HTTPException(500, f"Docker error: {exc.explanation or str(exc)}")


@app.post("/fd/containers/{container_id}/stop", response_model=ContainerActionResult)
async def stop_container(container_id: str, request: Request, user: dict | None = _required_user_dep):
    import asyncio
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    if c.status != "running":
        return ContainerActionResult(ok=True, container_id=c.short_id, message="Already stopped")
    await asyncio.get_event_loop().run_in_executor(None, lambda: c.stop(timeout=5))
    return ContainerActionResult(ok=True, container_id=c.short_id, message="Stopped")


@app.post("/fd/containers/{container_id}/start", response_model=ContainerActionResult)
async def start_container(container_id: str, request: Request, user: dict | None = _required_user_dep):
    import asyncio
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    if c.status == "running":
        return ContainerActionResult(ok=True, container_id=c.short_id, message="Already running")
    try:
        await asyncio.get_event_loop().run_in_executor(None, c.start)
    except docker.errors.APIError as exc:
        explanation = exc.explanation or str(exc)
        raise HTTPException(500, f"Docker start failed: {explanation}")
    return ContainerActionResult(ok=True, container_id=c.short_id, message="Started")


@app.post("/fd/containers/{container_id}/restart", response_model=ContainerActionResult)
async def restart_container(container_id: str, request: Request, user: dict | None = _required_user_dep):
    import asyncio
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    await asyncio.get_event_loop().run_in_executor(None, lambda: c.restart(timeout=5))
    return ContainerActionResult(ok=True, container_id=c.short_id, message="Restarted")


@app.delete("/fd/containers/{container_id}", response_model=ContainerActionResult)
async def remove_container(container_id: str, force: bool = False, request: Request = None, user: dict | None = _required_user_dep):
    import asyncio
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    name = c.name
    await asyncio.get_event_loop().run_in_executor(None, lambda: c.remove(force=force))
    return ContainerActionResult(ok=True, container_id=container_id, message=f"Removed '{name}'")


class RebuildRequest(BaseModel):
    description: str = ""  # Frontend sends current description override


class CloneRequest(BaseModel):
    new_name: str  # Name for the cloned agent


@app.post("/fd/containers/{container_id}/rebuild", response_model=ContainerActionResult)
async def rebuild_container(container_id: str, request: Request, req: RebuildRequest | None = None, user: dict | None = _required_user_dep):
    """Rebuild a container: stop, remove, pull latest image, re-spawn with same config."""
    import platform

    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    old_short_id = c.short_id
    labels = c.labels or {}

    # If frontend sent a description override, update the label
    if req and req.description:
        labels["flight-deck.description"] = req.description

    # Read the original spawn config from labels + container inspection
    agent_name = labels.get("flight-deck.agent-name", c.name)
    slug = _slug(agent_name)
    image = labels.get("flight-deck.image", CC_IMAGE_DEFAULT)
    web_port_str = labels.get("flight-deck.web-port", "")
    web_port = int(web_port_str) if web_port_str else None
    web_auth = labels.get("flight-deck.web-auth", "")
    description = labels.get("flight-deck.description", "")

    # Read environment from the running container
    env_list = c.attrs.get("Config", {}).get("Env", [])
    environment: dict[str, str] = {}
    for e in env_list:
        if "=" in e:
            k, v = e.split("=", 1)
            environment[k] = v

    # Read mounts from container
    mounts = c.attrs.get("Mounts", [])
    volumes: dict[str, dict] = {}
    for m in mounts:
        src = m.get("Source", "")
        dst = m.get("Destination", "")
        mode = m.get("Mode", "rw")
        if src and dst:
            volumes[src] = {"bind": dst, "mode": mode}

    # Read restart policy
    host_config = c.attrs.get("HostConfig", {})
    restart_policy = host_config.get("RestartPolicy", {"Name": "unless-stopped"})

    # Read security opts, cap_drop, cap_add
    security_opt = host_config.get("SecurityOpt", ["no-new-privileges:true", "seccomp:unconfined"])
    cap_drop = host_config.get("CapDrop", ["ALL"])
    cap_add = host_config.get("CapAdd", ["CHOWN", "SETUID", "SETGID", "SYS_CHROOT"])

    # Read tmpfs
    tmpfs = host_config.get("Tmpfs", {"/tmp": "", "/run": ""})

    # Read network mode
    network_mode = host_config.get("NetworkMode", "host")

    # Read hostname
    hostname = c.attrs.get("Config", {}).get("Hostname", slug)

    # Port publishing
    ports: dict[str, int] = {}
    network_mode_use: str | None = network_mode
    if platform.system() == "Darwin" and network_mode == "host":
        network_mode_use = None
        if web_port:
            ports[f"{web_port}/tcp"] = web_port
    elif network_mode != "host":
        if web_port:
            ports[f"{web_port}/tcp"] = web_port

    # Stop and remove old container
    if c.status == "running":
        c.stop(timeout=5)
    c.remove(force=True)

    # Pull latest image
    client = get_docker()
    try:
        client.images.pull(image)
    except docker.errors.APIError:
        pass  # If pull fails, use whatever's cached locally

    # Re-create with same config
    try:
        new_container = client.containers.run(
            image=image,
            name=slug,
            hostname=hostname,
            detach=True,
            network_mode=network_mode_use,
            ports=ports or None,
            volumes=volumes,
            environment=environment,
            labels=labels,
            security_opt=security_opt,
            cap_drop=cap_drop,
            cap_add=cap_add,
            tmpfs=tmpfs,
            restart_policy=restart_policy,
            stop_signal="SIGTERM",
        )
        return ContainerActionResult(
            ok=True,
            container_id=new_container.short_id,
            old_container_id=old_short_id,
            message=f"Agent '{agent_name}' rebuilt with latest image",
        )
    except docker.errors.ImageNotFound:
        raise HTTPException(404, f"Docker image '{image}' not found.")
    except docker.errors.APIError as exc:
        raise HTTPException(500, f"Docker error: {exc.explanation or str(exc)}")


def _find_available_port(start: int) -> int:
    """Find first available TCP port starting from `start`, checking
    running containers, managed processes, and the host's listening sockets."""
    import socket

    used_ports: set[int] = set()

    # Collect ports from Docker containers
    try:
        client = get_docker()
        for c in client.containers.list(all=True, filters={"label": CONTAINER_LABEL}):
            lbl_port = (c.labels or {}).get("flight-deck.web-port", "")
            if lbl_port:
                try:
                    used_ports.add(int(lbl_port))
                except ValueError:
                    pass
            docker_ports = c.attrs.get("NetworkSettings", {}).get("Ports", {}) or {}
            for _cp, bindings in docker_ports.items():
                if bindings:
                    for b in bindings:
                        try:
                            used_ports.add(int(b.get("HostPort", 0)))
                        except (ValueError, TypeError):
                            pass
    except Exception:
        pass  # Docker may not be available

    # Collect ports from managed processes
    registry = _load_process_registry()
    for entry in registry.values():
        wp = entry.get("web_port")
        if wp:
            used_ports.add(wp)

    max_search = int(os.environ.get("FD_PORT_RANGE", "500"))
    port = start
    while port < start + max_search:
        if port not in used_ports:
            # Also check if host port is free
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    pass
        port += 1
    raise HTTPException(500, f"No available port found in range {start}-{start + max_search}")


@app.post("/fd/containers/{container_id}/clone", response_model=ContainerActionResult)
async def clone_container(container_id: str, req: CloneRequest, request: Request, user: dict | None = _required_user_dep):
    """Clone a container: create a new agent with same config but its own data folder."""
    import platform
    import shutil

    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    labels = dict(c.labels or {})

    # Read the original config
    image = labels.get("flight-deck.image", CC_IMAGE_DEFAULT)
    web_port_str = labels.get("flight-deck.web-port", "")
    old_web_port = int(web_port_str) if web_port_str else None
    web_auth = labels.get("flight-deck.web-auth", "")
    old_agent_name = labels.get("flight-deck.agent-name", c.name)

    new_name = req.new_name.strip()
    if not new_name:
        raise HTTPException(400, "Name is required")
    new_slug = _slug(new_name)

    # Check name collision
    client = get_docker()
    try:
        existing = client.containers.get(new_slug)
        if existing.status == "running":
            raise HTTPException(400, f"Container '{new_slug}' already running.")
        existing.remove()
    except docker.errors.NotFound:
        pass

    # Determine old agent directory from the actual mount sources
    old_slug = _slug(old_agent_name)
    old_agent_dir = DATA_DIR / old_slug
    new_agent_dir = DATA_DIR / new_slug

    # Create new data directory structure
    new_agent_dir.mkdir(parents=True, exist_ok=True)
    (new_agent_dir / "data" / "workspace").mkdir(parents=True, exist_ok=True)
    (new_agent_dir / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (new_agent_dir / "data" / "skills").mkdir(parents=True, exist_ok=True)
    (new_agent_dir / "data" / "home-config").mkdir(parents=True, exist_ok=True)

    # Copy config.yaml and .env from original if they exist
    for fname in ("config.yaml", ".env"):
        src_file = old_agent_dir / fname
        if src_file.is_file():
            shutil.copy2(str(src_file), str(new_agent_dir / fname))
    # Copy home-config/config.yaml too
    src_hc = old_agent_dir / "data" / "home-config" / "config.yaml"
    if src_hc.is_file():
        shutil.copy2(str(src_hc), str(new_agent_dir / "data" / "home-config" / "config.yaml"))

    # Read environment from the source container
    env_list = c.attrs.get("Config", {}).get("Env", [])
    environment: dict[str, str] = {}
    for e in env_list:
        if "=" in e:
            k, v = e.split("=", 1)
            environment[k] = v

    # Build volume mounts for the clone — use the known structure
    # instead of trying to remap arbitrary paths from the old container.
    old_agent_dir_str = str(old_agent_dir)
    new_agent_dir_str = str(new_agent_dir)
    config_file = str(new_agent_dir / "config.yaml")
    env_file = str(new_agent_dir / ".env")

    # Start with the standard CC mounts pointing to new data dir
    volumes: dict[str, dict] = {
        config_file: {"bind": "/app/config.yaml", "mode": "ro"},
        env_file: {"bind": "/app/.env", "mode": "ro"},
        str(new_agent_dir / "data" / "home-config"): {"bind": "/home/claw/.captain-claw", "mode": "rw"},
        str(new_agent_dir / "data" / "workspace"): {"bind": "/data/workspace", "mode": "rw"},
        str(new_agent_dir / "data" / "sessions"): {"bind": "/data/sessions", "mode": "rw"},
        str(new_agent_dir / "data" / "skills"): {"bind": "/data/skills", "mode": "rw"},
    }
    # Carry over any extra volumes that weren't part of the agent data dir
    mounts = c.attrs.get("Mounts", [])
    known_dests = {"/app/config.yaml", "/app/.env", "/home/claw/.captain-claw",
                   "/data/workspace", "/data/sessions", "/data/skills"}
    for m in mounts:
        src = m.get("Source", "")
        dst = m.get("Destination", "")
        mode = m.get("Mode", "rw")
        if src and dst and dst not in known_dests:
            volumes[src] = {"bind": dst, "mode": mode}

    # Read host config
    host_config = c.attrs.get("HostConfig", {})
    restart_policy = host_config.get("RestartPolicy", {"Name": "unless-stopped"})
    security_opt = host_config.get("SecurityOpt", ["no-new-privileges:true", "seccomp:unconfined"])
    cap_drop = host_config.get("CapDrop", ["ALL"])
    cap_add = host_config.get("CapAdd", ["CHOWN", "SETUID", "SETGID", "SYS_CHROOT"])
    tmpfs = host_config.get("Tmpfs", {"/tmp": "", "/run": ""})
    network_mode = host_config.get("NetworkMode", "host")

    # Find first available port starting from original
    new_web_port: int | None = None
    if old_web_port:
        new_web_port = _find_available_port(old_web_port + 1)

    # Update labels for the clone
    labels["flight-deck.agent-name"] = new_name
    labels["flight-deck.description"] = ""
    if new_web_port:
        labels["flight-deck.web-port"] = str(new_web_port)

    # Update config.yaml with new web port and botport instance name
    cfg_path = new_agent_dir / "config.yaml"
    if cfg_path.is_file() and new_web_port and old_web_port:
        cfg_text = cfg_path.read_text()
        cfg_text = cfg_text.replace(f"port: {old_web_port}", f"port: {new_web_port}")
        # Update botport instance name
        if old_agent_name:
            cfg_text = cfg_text.replace(f"instance_name: {old_agent_name}", f"instance_name: {new_name}")
            cfg_text = cfg_text.replace(f"instance_name: '{old_agent_name}'", f"instance_name: '{new_name}'")
        cfg_path.write_text(cfg_text)
        # Also update home-config copy
        hc_path = new_agent_dir / "data" / "home-config" / "config.yaml"
        if hc_path.is_file():
            hc_path.write_text(cfg_text)

    # Make sure .env file exists (even if empty) so Docker doesn't create a directory
    env_path = new_agent_dir / ".env"
    if not env_path.is_file():
        env_path.write_text("")

    hostname = new_slug

    # Port publishing
    ports: dict[str, int] = {}
    network_mode_use: str | None = network_mode
    if platform.system() == "Darwin" and network_mode == "host":
        network_mode_use = None
        if new_web_port:
            ports[f"{new_web_port}/tcp"] = new_web_port
    elif network_mode != "host":
        if new_web_port:
            ports[f"{new_web_port}/tcp"] = new_web_port

    try:
        new_container = client.containers.run(
            image=image,
            name=new_slug,
            hostname=hostname,
            detach=True,
            network_mode=network_mode_use,
            ports=ports or None,
            volumes=volumes,
            environment=environment,
            labels=labels,
            security_opt=security_opt,
            cap_drop=cap_drop,
            cap_add=cap_add,
            tmpfs=tmpfs,
            restart_policy=restart_policy,
            stop_signal="SIGTERM",
        )
        return ContainerActionResult(
            ok=True,
            container_id=new_container.short_id,
            message=f"Agent '{new_name}' cloned from '{old_agent_name}' (port {new_web_port})",
        )
    except docker.errors.ImageNotFound:
        raise HTTPException(404, f"Docker image '{image}' not found.")
    except docker.errors.APIError as exc:
        raise HTTPException(500, f"Docker error: {exc.explanation or str(exc)}")


@app.get("/fd/containers/{container_id}/logs")
async def container_logs(container_id: str, tail: int = 200, follow: bool = False, request: Request = None, user: dict | None = _required_user_dep):
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    if follow:
        def stream():
            for chunk in c.logs(stream=True, follow=True, tail=tail):
                yield chunk
        return StreamingResponse(stream(), media_type="text/plain")
    else:
        logs = c.logs(tail=tail).decode("utf-8", errors="replace")
        return {"logs": logs}


# ── Agent config editing ──

class AgentConfigUpdate(BaseModel):
    config_yaml: str | None = None
    env: str | None = None


def _resolve_agent_dir(identifier: str, kind: str, user_id: str) -> Path:
    """Resolve the on-disk data directory for a container or process agent."""
    if kind == "docker":
        c = _find_container(identifier, user_id)
        slug = c.name
    else:
        entry = _verify_process_owner(identifier, user_id)
        slug = identifier
    agent_dir = DATA_DIR / slug
    if not agent_dir.is_dir():
        raise HTTPException(404, f"Agent data directory not found for '{slug}'")
    return agent_dir


@app.get("/fd/agent-config/{kind}/{identifier}")
async def get_agent_config(
    kind: str, identifier: str, request: Request,
    user: dict | None = _required_user_dep,
):
    """Read an agent's config.yaml and .env files."""
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")
    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)

    config_yaml = ""
    env = ""
    config_file = agent_dir / "config.yaml"
    env_file = agent_dir / ".env"
    if config_file.is_file():
        config_yaml = config_file.read_text()
    if env_file.is_file():
        env = env_file.read_text()
    return {"config_yaml": config_yaml, "env": env}


@app.put("/fd/agent-config/{kind}/{identifier}")
async def update_agent_config(
    kind: str, identifier: str, body: AgentConfigUpdate, request: Request,
    user: dict | None = _required_user_dep,
):
    """Update an agent's config.yaml and/or .env files. Agent must be restarted for changes to take effect."""
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")
    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)

    updated = []
    if body.config_yaml is not None:
        config_file = agent_dir / "config.yaml"
        config_file.write_text(body.config_yaml)
        # Also update home-config copy so it takes precedence on restart
        home_config = agent_dir / "data" / "home-config" / "config.yaml"
        if home_config.parent.is_dir():
            home_config.write_text(body.config_yaml)
        # Also update home-config-parent/.captain-claw copy
        parent_config = agent_dir / "data" / "home-config-parent" / ".captain-claw" / "config.yaml"
        if parent_config.parent.is_dir():
            parent_config.write_text(body.config_yaml)
        updated.append("config.yaml")
    if body.env is not None:
        env_file = agent_dir / ".env"
        env_file.write_text(body.env)
        updated.append(".env")

    return {"ok": True, "updated": updated, "message": "Restart the agent for changes to take effect."}


class AgentModelUpdate(BaseModel):
    provider: str
    model: str
    api_key: str | None = None


@app.put("/fd/agent-model/{kind}/{identifier}")
async def update_agent_model(
    kind: str, identifier: str, body: AgentModelUpdate, request: Request,
    user: dict | None = _required_user_dep,
):
    """Quick-update an agent's provider, model, and optionally api_key in all config locations."""
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")
    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)

    # Gather all config file paths that may exist
    config_paths = [
        agent_dir / "config.yaml",
        agent_dir / "data" / "home-config" / "config.yaml",
        agent_dir / "data" / "home-config-parent" / ".captain-claw" / "config.yaml",
    ]

    updated_count = 0
    for cfg_path in config_paths:
        if not cfg_path.is_file():
            continue
        try:
            data = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        if "model" not in data or not isinstance(data.get("model"), dict):
            data["model"] = {}
        data["model"]["provider"] = body.provider
        data["model"]["model"] = body.model
        if body.api_key is not None:
            data["model"]["api_key"] = body.api_key
        cfg_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True))
        updated_count += 1

    return {"ok": True, "updated": updated_count, "message": "Restart the agent for changes to take effect."}


class AgentModeUpdate(BaseModel):
    mode: str = "neutra"


@app.put("/fd/agent-mode/{kind}/{identifier}")
async def update_agent_mode(
    kind: str, identifier: str, body: AgentModeUpdate, request: Request,
    user: dict | None = _required_user_dep,
):
    """Update an agent's cognitive mode at runtime (no restart needed).

    Writes the mode to cognitive_mode.txt — the agent picks it up
    on the next system prompt build via mtime-based cache invalidation.
    """
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")

    # Validate mode name.
    from captain_claw.cognitive_mode import MODES
    mode_name = body.mode.lower().strip()
    if mode_name not in MODES:
        raise HTTPException(400, f"Unknown cognitive mode: {mode_name!r}. Valid: {', '.join(MODES)}")

    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)

    # Write to all potential home config locations.
    for subdir in ("home-config", "home-config-parent"):
        cc_dir = agent_dir / "data" / subdir / ".captain-claw"
        if cc_dir.is_dir():
            mode_file = cc_dir / "cognitive_mode.txt"
            if mode_name == "neutra":
                # Remove file for neutra (default no-op).
                mode_file.unlink(missing_ok=True)
            else:
                mode_file.write_text(mode_name, encoding="utf-8")

    return {"ok": True, "mode": mode_name, "message": "Mode updated. Takes effect on the agent's next response."}


class AgentEcoModeUpdate(BaseModel):
    enabled: bool = False


@app.put("/fd/agent-eco-mode/{kind}/{identifier}")
async def update_agent_eco_mode(
    kind: str, identifier: str, body: AgentEcoModeUpdate, request: Request,
    user: dict | None = _required_user_dep,
):
    """Toggle eco mode (micro instructions + lazy tools) at runtime.

    Writes ``eco_mode.txt`` — the agent picks it up on the next system
    prompt build, just like cognitive mode.
    """
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")

    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)

    for subdir in ("home-config", "home-config-parent"):
        cc_dir = agent_dir / "data" / subdir / ".captain-claw"
        if cc_dir.is_dir():
            eco_file = cc_dir / "eco_mode.txt"
            if body.enabled:
                eco_file.write_text("on", encoding="utf-8")
            else:
                eco_file.unlink(missing_ok=True)

    return {"ok": True, "enabled": body.enabled, "message": "Eco mode updated. Takes effect on the agent's next response."}


@app.get("/fd/agent-eco-mode/{kind}/{identifier}")
async def get_agent_eco_mode(
    kind: str, identifier: str, request: Request,
    user: dict | None = _required_user_dep,
):
    """Read current eco mode state for an agent."""
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")

    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)

    for subdir in ("home-config", "home-config-parent"):
        eco_file = agent_dir / "data" / subdir / ".captain-claw" / "eco_mode.txt"
        if eco_file.is_file():
            return {"enabled": True}

    return {"enabled": False}


@app.get("/fd/cognitive-modes")
async def list_cognitive_modes():
    """Return all available cognitive modes for UI dropdowns."""
    from captain_claw.cognitive_mode import list_modes, mode_to_dict
    return {"modes": [mode_to_dict(m) for m in list_modes()]}


@app.get("/fd/containers/{container_id}")
async def get_container(container_id: str, request: Request, user: dict | None = _required_user_dep):
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    info = _container_info(c)
    # Add extra details
    return {
        **info.model_dump(),
        "labels": c.labels,
        "env": c.attrs.get("Config", {}).get("Env", []),
        "mounts": [
            {"source": m.get("Source", ""), "destination": m.get("Destination", ""), "mode": m.get("Mode", "")}
            for m in c.attrs.get("Mounts", [])
        ],
    }


@app.websocket("/fd/agent-ws/{host}/{port}")
async def agent_ws_proxy(ws: WebSocket, host: str, port: int, token: str = ""):
    """Proxy WebSocket to a CC agent — avoids browser CORS restrictions."""
    import websockets

    await ws.accept()
    # Auto-resolve auth token if the caller didn't provide one
    auth = token or _resolve_agent_auth(port)
    params = f"?token={auth}" if auth else ""
    agent_url = f"ws://{host}:{port}/ws{params}"

    try:
        async with websockets.connect(agent_url, max_size=4 * 1024 * 1024) as agent_ws:
            async def client_to_agent():
                try:
                    while True:
                        data = await ws.receive_text()
                        await agent_ws.send(data)
                except WebSocketDisconnect:
                    pass

            async def agent_to_client():
                try:
                    async for msg in agent_ws:
                        await ws.send_text(msg if isinstance(msg, str) else msg.decode())
                except Exception:
                    pass

            done, pending = await asyncio.wait(
                [asyncio.create_task(client_to_agent()), asyncio.create_task(agent_to_client())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
    except Exception as exc:
        try:
            await ws.send_text(f'{{"type":"error","message":"Connection failed: {exc}"}}')
            await ws.close()
        except Exception:
            pass


@app.get("/fd/probe")
async def probe_agent(host: str = "localhost", port: int = 23080):
    """Probe a CC agent's web server (server-side, avoids CORS)."""
    import httpx
    url = f"http://{host}:{port}/"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url)
            return {"ok": resp.status_code < 500, "status": resp.status_code}
    except Exception:
        return {"ok": False, "status": 0}


class FleetAgent(BaseModel):
    name: str
    kind: str  # docker | process | local
    host: str = "localhost"
    port: int
    status: str
    description: str = ""


@app.get("/fd/fleet", response_model=list[FleetAgent])
async def get_fleet(request: Request, user: dict | None = _optional_user_dep):
    """Return all running/known agents across docker, process, and local stores."""
    fleet: list[FleetAgent] = []
    user_id = getattr(request.state, "user_id", "")

    # Docker containers
    try:
        client = get_docker()
        for c in client.containers.list(all=True, filters={"label": CONTAINER_LABEL}):
            labels = c.labels or {}
            if AUTH_ENABLED and user_id and labels.get(OWNER_LABEL, "") != user_id:
                continue
            wp = labels.get("flight-deck.web-port", "")
            fleet.append(FleetAgent(
                name=labels.get("flight-deck.agent-name", c.name),
                kind="docker",
                host="localhost",
                port=int(wp) if wp else 0,
                status=c.status,
                description=labels.get("flight-deck.description", ""),
            ))
    except Exception:
        pass

    # Process agents
    registry = _load_process_registry()
    for slug, entry in registry.items():
        if AUTH_ENABLED and user_id and entry.get("owner", "") != user_id:
            continue
        alive = _process_is_alive(slug)
        fleet.append(FleetAgent(
            name=entry.get("name", slug),
            kind="process",
            host="localhost",
            port=entry.get("web_port", 0),
            status="running" if alive else "stopped",
            description=entry.get("description", ""),
        ))

    return fleet


def _resolve_agent_auth(port: int) -> str:
    """Look up the auth token for an agent by its web port from Docker labels or process registry."""
    # Check Docker containers
    try:
        client = get_docker()
        for c in client.containers.list(filters={"label": CONTAINER_LABEL}):
            labels = c.labels or {}
            wp = labels.get("flight-deck.web-port", "")
            if wp and int(wp) == port:
                return labels.get("flight-deck.web-auth", "")
    except Exception:
        pass

    # Check process registry
    registry = _load_process_registry()
    for entry in registry.values():
        if entry.get("web_port") == port:
            return entry.get("web_auth", "")

    return ""


async def _notify_fleet_change(new_agent_name: str, new_agent_port: int, event: str = "joined", owner_id: str = ""):
    """Notify running agents about a fleet change. When auth is enabled, only notify agents owned by the same user."""
    import websockets as _ws
    import json as _json

    # Build list of running agent WebSocket endpoints (excluding the new one),
    # scoped to the same owner when auth is enabled.
    targets: list[tuple[str, int, str]] = []  # (host, port, auth)

    try:
        client = get_docker()
        for c in client.containers.list(filters={"label": CONTAINER_LABEL}):
            labels = c.labels or {}
            if AUTH_ENABLED and owner_id and labels.get(OWNER_LABEL, "") != owner_id:
                continue
            wp = labels.get("flight-deck.web-port", "")
            if wp and int(wp) != new_agent_port:
                targets.append(("localhost", int(wp), labels.get("flight-deck.web-auth", "")))
    except Exception:
        pass

    registry = _load_process_registry()
    for slug, entry in registry.items():
        if AUTH_ENABLED and owner_id and entry.get("owner", "") != owner_id:
            continue
        wp = entry.get("web_port", 0)
        if wp and wp != new_agent_port and _process_is_alive(slug):
            targets.append(("localhost", wp, entry.get("web_auth", "")))

    if not targets:
        return

    # Build fleet list scoped to the same owner
    fleet: list[dict] = []
    try:
        client = get_docker()
        for c in client.containers.list(all=True, filters={"label": CONTAINER_LABEL}):
            labels = c.labels or {}
            if AUTH_ENABLED and owner_id and labels.get(OWNER_LABEL, "") != owner_id:
                continue
            wp = labels.get("flight-deck.web-port", "")
            fleet.append({"name": labels.get("flight-deck.agent-name", c.name), "status": c.status, "port": int(wp) if wp else 0})
    except Exception:
        pass
    reg = _load_process_registry()
    for slug, entry in reg.items():
        if AUTH_ENABLED and owner_id and entry.get("owner", "") != owner_id:
            continue
        fleet.append({"name": entry.get("name", slug), "status": "running" if _process_is_alive(slug) else "stopped", "port": entry.get("web_port", 0)})

    notification = (
        f"[Flight Deck] Agent '{new_agent_name}' has {event} the fleet on port {new_agent_port}. "
        f"Current fleet: {', '.join(a['name'] + ' (' + a['status'] + ', :' + str(a['port']) + ')' for a in fleet)}"
    )

    async def _send_to(host: str, port: int, auth: str):
        params = f"?token={auth}" if auth else ""
        url = f"ws://{host}:{port}/ws{params}"
        try:
            async with _ws.connect(url, open_timeout=5, close_timeout=3) as ws:
                # Wait for welcome
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                welcome = _json.loads(raw)
                if welcome.get("type") != "welcome":
                    return
                # Skip replay
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    msg = _json.loads(raw)
                    if msg.get("type") == "replay_done":
                        break
                # Send as notification (injected into session, no LLM response triggered)
                await ws.send(_json.dumps({"type": "notification", "content": notification}))
                # Wait briefly then disconnect
                try:
                    await asyncio.wait_for(ws.recv(), timeout=5)
                except asyncio.TimeoutError:
                    pass
        except Exception:
            pass  # Best-effort; don't fail spawn if notification fails

    # Fire all notifications concurrently
    await asyncio.gather(*[_send_to(h, p, a) for h, p, a in targets], return_exceptions=True)


class ConsultPeerRequest(BaseModel):
    host: str = "localhost"
    port: int
    auth: str = ""
    message: str
    source_name: str = "another agent"
    timeout: float = Field(default=480.0, le=600.0)


# Track active consultations to prevent duplicate requests to the same target
_active_consults: dict[int, str] = {}  # target_port -> source_name


@app.post("/fd/consult-peer")
async def consult_peer(req: ConsultPeerRequest, request: Request, user: dict | None = _optional_user_dep):
    """Send a message to a peer agent and stream back intermediate events + final response as ndjson."""
    import websockets
    import json

    # Prevent duplicate consultations to the same agent
    existing = _active_consults.get(req.port)
    if existing:
        async def _busy_stream():
            yield json.dumps({
                "ok": False,
                "error": f"Agent on port {req.port} is already being consulted by '{existing}'. Wait for that consultation to finish, or use 'delegate' for fire-and-forget tasks.",
            }) + "\n"
        return StreamingResponse(_busy_stream(), media_type="application/x-ndjson")

    # Event types we forward as peer activity so the caller can show progress
    _FORWARD_TYPES = {"status", "thinking", "monitor", "tool_stream"}

    # Resolve auth token from Fleet Deck records if not provided by caller.
    auth = req.auth
    if not auth:
        auth = _resolve_agent_auth(req.port)

    params = f"?token={auth}" if auth else ""
    agent_url = f"ws://{req.host}:{req.port}/ws{params}"

    async def _event_stream():
        _active_consults[req.port] = req.source_name
        try:
            async with websockets.connect(agent_url, max_size=4 * 1024 * 1024) as ws:
                # Wait for welcome
                welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if welcome.get("type") != "welcome":
                    yield json.dumps({"ok": False, "error": "Unexpected handshake"}) + "\n"
                    return

                # Skip replay messages
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    msg = json.loads(raw)
                    if msg.get("type") == "replay_done":
                        break
                    if msg.get("type") not in ("chat_message",) or not msg.get("replay"):
                        break

                # Send the chat message
                await ws.send(json.dumps({"type": "chat", "content": req.message}))

                # Stream events until we get the final assistant response
                response_parts: list[str] = []
                deadline = asyncio.get_event_loop().time() + req.timeout
                recv_interval = 15.0  # heartbeat every 15s of silence
                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        if not response_parts:
                            yield json.dumps({"ok": False, "error": "Timed out waiting for response"}) + "\n"
                            return
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, recv_interval))
                    except asyncio.TimeoutError:
                        # No message in recv_interval — send heartbeat and keep waiting
                        elapsed = int(req.timeout - remaining)
                        yield json.dumps({"event": "heartbeat", "data": {"elapsed": elapsed, "timeout": int(req.timeout)}}) + "\n"
                        continue
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    # Forward interesting intermediate events
                    if msg_type in _FORWARD_TYPES:
                        yield json.dumps({"event": msg_type, "data": msg}) + "\n"

                    if msg_type == "chat_message" and msg.get("role") == "assistant" and not msg.get("replay"):
                        content = msg.get("content", "")
                        if content:
                            response_parts.append(content)
                        break
                    elif msg_type == "error":
                        yield json.dumps({"ok": False, "error": msg.get("message", "Agent error")}) + "\n"
                        return

            yield json.dumps({
                "ok": True,
                "done": True,
                "response": "\n".join(response_parts) if response_parts else "(no response)",
            }) + "\n"
        except Exception as exc:
            yield json.dumps({"ok": False, "error": f"Connection failed: {exc}"}) + "\n"
        finally:
            _active_consults.pop(req.port, None)

    return StreamingResponse(_event_stream(), media_type="application/x-ndjson")


class DelegatePeerRequest(BaseModel):
    target_host: str = "localhost"
    target_port: int
    target_name: str = ""
    source_host: str = "localhost"
    source_port: int
    source_name: str = "another agent"
    message: str
    timeout: float = Field(default=600.0, le=1800.0)
    # Origin platform tracking — so results are delivered to the correct session
    origin_platform: str = "web"       # "web" or "telegram"
    origin_user_id: str = ""           # telegram user id
    origin_chat_id: int = 0            # telegram chat id


@app.post("/fd/delegate-peer")
async def delegate_peer(req: DelegatePeerRequest, request: Request, user: dict | None = _optional_user_dep):
    """Fire-and-forget: send a task to a peer agent. When the peer finishes, deliver the result back to the source agent as a chat message."""
    import websockets
    import json

    _FORWARD_TYPES = {"status", "thinking", "monitor", "tool_stream"}

    target_auth = _resolve_agent_auth(req.target_port)
    source_auth = _resolve_agent_auth(req.source_port)

    peer_display = req.target_name or f"agent@{req.target_port}"

    async def _background():
        log.info("delegate_background: started", target=peer_display, source=req.source_name,
                 target_port=req.target_port, source_port=req.source_port)

        # Phase 1: send task to target agent and wait for response
        t_params = f"?token={target_auth}" if target_auth else ""
        target_url = f"ws://{req.target_host}:{req.target_port}/ws{t_params}"
        response_text = ""
        try:
            async with websockets.connect(target_url, max_size=4 * 1024 * 1024) as ws:
                welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if welcome.get("type") != "welcome":
                    response_text = f"[Error] Unexpected handshake from {peer_display}"
                    log.error("delegate_background: bad handshake from target", target=peer_display)
                else:
                    # Skip replay
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=10)
                        msg = json.loads(raw)
                        if msg.get("type") == "replay_done":
                            break
                        if msg.get("type") not in ("chat_message",) or not msg.get("replay"):
                            break

                    await ws.send(json.dumps({"type": "chat", "content": req.message}))
                    log.info("delegate_background: task sent to target", target=peer_display)

                    # Wait for the final response
                    deadline = asyncio.get_event_loop().time() + req.timeout
                    recv_interval = 30.0
                    while True:
                        remaining = deadline - asyncio.get_event_loop().time()
                        if remaining <= 0:
                            response_text = f"[Timeout] {peer_display} did not finish within {int(req.timeout)}s"
                            log.warning("delegate_background: target timed out", target=peer_display, timeout=req.timeout)
                            break
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, recv_interval))
                        except asyncio.TimeoutError:
                            continue
                        msg = json.loads(raw)
                        msg_type = msg.get("type", "")
                        if msg_type == "chat_message" and msg.get("role") == "assistant" and not msg.get("replay"):
                            response_text = msg.get("content", "(no response)")
                            log.info("delegate_background: got response from target",
                                     target=peer_display, response_len=len(response_text))
                            break
                        elif msg_type == "error":
                            response_text = f"[Error from {peer_display}] {msg.get('message', 'Unknown error')}"
                            log.error("delegate_background: target returned error", target=peer_display, error=msg.get("message"))
                            break
        except Exception as exc:
            response_text = f"[Error] Could not reach {peer_display}: {exc}"
            log.error("delegate_background: phase 1 failed", target=peer_display, error=str(exc))

        if not response_text:
            response_text = "(no response received)"

        # Phase 2: deliver the result back to the source agent
        s_params = f"?token={source_auth}" if source_auth else ""
        source_url = f"ws://{req.source_host}:{req.source_port}/ws{s_params}"
        callback_msg = f"[Delegated result from {peer_display}]\n\n{response_text}"
        log.info("delegate_background: delivering result to source",
                 source=req.source_name, source_port=req.source_port, result_len=len(response_text))
        try:
            async with websockets.connect(source_url, open_timeout=10, close_timeout=5) as ws:
                welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if welcome.get("type") != "welcome":
                    log.error("delegate_callback: unexpected handshake from source", source=req.source_name)
                    return
                # Skip replay
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    msg = json.loads(raw)
                    if msg.get("type") == "replay_done":
                        break
                notification_payload = {
                    "type": "notification",
                    "content": callback_msg,
                    "trigger_response": True,
                }
                # Include origin platform info so the agent routes the result correctly
                if req.origin_platform and req.origin_platform != "web":
                    notification_payload["origin_platform"] = req.origin_platform
                    notification_payload["origin_user_id"] = req.origin_user_id
                    notification_payload["origin_chat_id"] = req.origin_chat_id
                await ws.send(json.dumps(notification_payload))
                log.info("delegate_callback: result delivered to source", source=req.source_name,
                         origin_platform=req.origin_platform)
                # Wait briefly for acknowledgment
                try:
                    await asyncio.wait_for(ws.recv(), timeout=30)
                except asyncio.TimeoutError:
                    pass
        except Exception as exc:
            log.error("delegate_callback: failed to deliver result to source", source=req.source_name, error=str(exc))

    # Keep a strong reference so the task doesn't get garbage-collected
    task = asyncio.create_task(_background())
    if not hasattr(app.state, "_delegate_tasks"):
        app.state._delegate_tasks = set()
    app.state._delegate_tasks.add(task)
    task.add_done_callback(app.state._delegate_tasks.discard)

    return {"ok": True, "message": f"Task delegated to {peer_display}. Results will be delivered to {req.source_name} when ready."}


_WORKSPACE_ALLOWED_EXTS: set[str] = {
    # Documents
    ".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".pdf",
    ".doc", ".docx", ".odt", ".rtf",
    # Presentations & spreadsheets
    ".ppt", ".pptx", ".xls", ".xlsx", ".ods", ".odp",
    # Data
    ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".xml",
    # Scripts & code
    ".py", ".js", ".ts", ".jsx", ".tsx", ".sh", ".bash", ".sql",
    ".rb", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".css", ".scss",
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico",
    # Audio / video
    ".mp3", ".wav", ".ogg", ".mp4", ".webm",
    # Archives
    ".zip", ".tar", ".gz",
    # Config / misc text
    ".env", ".ini", ".cfg", ".conf", ".log",
}

_TEXT_EXTS: set[str] = {
    ".txt", ".md", ".markdown", ".rst", ".json", ".jsonl",
    ".yaml", ".yml", ".csv", ".tsv", ".toml", ".xml",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".htm",
    ".css", ".scss", ".sh", ".bash", ".sql", ".log",
    ".env", ".ini", ".cfg", ".conf", ".rb", ".go", ".rs",
    ".java", ".c", ".cpp", ".h",
}


def _scan_workspace_files(agent_dir: Path) -> list[dict]:
    """Scan an agent's workspace directory for user-facing files."""
    import mimetypes
    results: list[dict] = []
    workspace = agent_dir / "data" / "workspace"
    if not workspace.is_dir():
        return results
    for f in workspace.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext not in _WORKSPACE_ALLOWED_EXTS:
            continue
        try:
            stat = f.stat()
            mt, _ = mimetypes.guess_type(f.name)
            results.append({
                "logical": "workspace/" + str(f.relative_to(workspace)),
                "physical": str(f),
                "filename": f.name,
                "extension": ext,
                "exists": True,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "mime_type": mt or "application/octet-stream",
                "is_text": ext in _TEXT_EXTS,
                "source": "workspace",
            })
        except OSError:
            continue
    return results


# ── Agent Datastore proxy ─────────────────────────────────────────────

@app.get("/fd/agent-datastore/{host}/{port}/tables")
async def agent_datastore_tables(
    host: str, port: int, token: str = "",
    user: dict | None = _required_user_dep,
):
    """List datastore tables from a CC agent (proxied)."""
    import httpx
    auth = token or _resolve_agent_auth(port)
    params = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/datastore/tables{params}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/agent-datastore/{host}/{port}/tables/{table_name}/rows")
async def agent_datastore_rows(
    host: str, port: int, table_name: str,
    limit: int = 100, offset: int = 0,
    order_by: str = "", order_dir: str = "asc",
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Query rows from a datastore table on a CC agent (proxied)."""
    import httpx
    auth = token or _resolve_agent_auth(port)
    qs = f"token={auth}&" if auth else ""
    qs += f"limit={limit}&offset={offset}"
    if order_by:
        qs += f"&order_by={order_by}&order_dir={order_dir}"
    url = f"http://{host}:{port}/api/datastore/tables/{table_name}/rows?{qs}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/agent-files/{host}/{port}")
async def agent_files(host: str, port: int, token: str = "", request: Request = None, user: dict | None = _required_user_dep):
    """List files from a CC agent (proxied to avoid CORS), merged with workspace scan."""
    import httpx
    registered: list[dict] = []
    # Auto-resolve auth token if the caller didn't provide one
    auth = token or _resolve_agent_auth(port)
    params = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/files{params}"
    agent_reachable = False
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                registered = resp.json()
                agent_reachable = True
    except (httpx.ConnectError, Exception):
        pass

    # Also scan the workspace directory for any unregistered files
    workspace_files: list[dict] = []
    user_id = getattr(request.state, "user_id", "") if request else ""
    # Try to find the agent directory by matching port to a process or container
    registry = _load_process_registry()
    for slug, entry in registry.items():
        if entry.get("web_port") == port:
            if AUTH_ENABLED and user_id and entry.get("owner", "") != user_id:
                continue
            agent_dir = DATA_DIR / slug
            if agent_dir.is_dir():
                workspace_files = _scan_workspace_files(agent_dir)
            break

    if not workspace_files:
        if not registered and not agent_reachable:
            raise HTTPException(502, "Cannot connect to agent")
        return registered

    # Merge: use registered files as base, add workspace files not already listed
    registered_physicals = {f.get("physical", "") for f in registered}
    registered_filenames = {f.get("filename", "") for f in registered}
    for wf in workspace_files:
        if wf["physical"] not in registered_physicals and wf["filename"] not in registered_filenames:
            registered.append(wf)
    return registered


class TransferRequest(BaseModel):
    src_host: str
    src_port: int
    src_auth: str = ""
    src_path: str
    dst_host: str
    dst_port: int
    dst_auth: str = ""


@app.post("/fd/transfer")
async def transfer_file(req: TransferRequest, request: Request, user: dict | None = _required_user_dep):
    """Download a file from one agent and upload it to another."""
    import httpx

    # Build query params properly: a path with `?` collisions plus an
    # unencoded source path used to silently corrupt both URLs and the source
    # agent ended up parsing `path=/foo/bar?token=xyz` as a single value.
    src_params: dict[str, str] = {"path": req.src_path}
    if req.src_auth:
        src_params["token"] = req.src_auth
    dst_params: dict[str, str] = {}
    if req.dst_auth:
        dst_params["token"] = req.dst_auth

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        # Download from source
        dl_url = f"http://{req.src_host}:{req.src_port}/api/files/download"
        dl_resp = await client.get(dl_url, params=src_params)
        if dl_resp.status_code != 200:
            raise HTTPException(
                502,
                f"Source agent download failed: {dl_resp.status_code} "
                f"({dl_resp.text[:200]})",
            )

        # Get filename from content-disposition or path
        cd = dl_resp.headers.get("content-disposition", "")
        if "filename=" in cd:
            filename = cd.split("filename=")[-1].strip('" ')
        else:
            filename = req.src_path.rsplit("/", 1)[-1]

        # Upload to destination
        up_url = f"http://{req.dst_host}:{req.dst_port}/api/file/upload"
        files = {"file": (filename, dl_resp.content, dl_resp.headers.get("content-type", "application/octet-stream"))}
        up_resp = await client.post(up_url, params=dst_params, files=files)
        if up_resp.status_code != 200:
            raise HTTPException(
                502,
                f"Destination agent upload failed: {up_resp.status_code} "
                f"({up_resp.text[:200]})",
            )

        result = up_resp.json()
        return {"ok": True, "filename": filename, "dest_path": result.get("path", ""), "size": result.get("size", 0)}


@app.get("/fd/agent-file-download/{host}/{port}")
async def agent_file_download(host: str, port: int, path: str, token: str = "", request: Request = None, user: dict | None = _required_user_dep):
    """Proxy file download from a CC agent."""
    import httpx
    import urllib.parse
    auth = token or _resolve_agent_auth(port)
    params = f"path={urllib.parse.quote(path)}"
    if auth:
        params += f"&token={urllib.parse.quote(auth)}"
    url = f"http://{host}:{port}/api/files/download?{params}"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, f"Agent returned {resp.status_code}")
            cd = resp.headers.get("content-disposition", "")
            ct = resp.headers.get("content-type", "application/octet-stream")
            headers = {"Content-Type": ct}
            if cd:
                headers["Content-Disposition"] = cd
            else:
                filename = path.rsplit("/", 1)[-1]
                headers["Content-Disposition"] = f'attachment; filename="{filename}"'
            from starlette.responses import Response
            return Response(content=resp.content, headers=headers)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/agent-file-view/{host}/{port}")
async def agent_file_view(host: str, port: int, path: str, token: str = "", request: Request = None, user: dict | None = _required_user_dep):
    """Proxy file view from a CC agent (inline, no download header)."""
    import httpx
    import urllib.parse
    auth = token or _resolve_agent_auth(port)
    params = f"path={urllib.parse.quote(path)}"
    if auth:
        params += f"&token={urllib.parse.quote(auth)}"
    # Try the /api/files/view endpoint first, fall back to /download
    url = f"http://{host}:{port}/api/files/view?{params}"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                # Fallback to download endpoint
                url2 = f"http://{host}:{port}/api/files/download?{params}"
                resp = await client.get(url2)
                if resp.status_code != 200:
                    raise HTTPException(resp.status_code, f"Agent returned {resp.status_code}")
            ct = resp.headers.get("content-type", "text/plain")
            from starlette.responses import Response
            return Response(content=resp.content, headers={
                "Content-Type": ct,
                "Content-Disposition": "inline",
            })
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.post("/fd/agent-file-upload/{host}/{port}")
async def agent_file_upload(host: str, port: int, token: str = "", file: UploadFile = File(...), request: Request = None, user: dict | None = _required_user_dep):
    """Proxy file upload to a CC agent."""
    import httpx

    auth = token or _resolve_agent_auth(port)
    params = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/file/upload{params}"
    content = await file.read()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {"file": (file.filename or "upload", content, file.content_type or "application/octet-stream")}
            resp = await client.post(url, files=files)
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, f"Agent upload failed: {resp.status_code}")
            # Log usage
            if AUTH_ENABLED and user:
                db = app.state.fd_db
                await db.log_usage(user["id"], "file_upload", json.dumps({"filename": file.filename, "size": len(content), "agent_port": port}))
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/agent-usage/{host}/{port}")
async def agent_usage(host: str, port: int, token: str = "", period: str = "today", request: Request = None, user: dict | None = _required_user_dep):
    """Proxy /api/usage from a CC agent."""
    import httpx
    auth = token or _resolve_agent_auth(port)
    params = f"?period={period}"
    if auth:
        params += f"&token={auth}"
    url = f"http://{host}:{port}/api/usage{params}"
    try:
        # First request with token may return a 302 redirect that sets a cookie.
        # Use follow_redirects=False to capture the cookie, then retry.
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, follow_redirects=False)

            if resp.status_code in (301, 302, 303, 307, 308):
                # Token was accepted and a cookie was set — retry with the cookie
                cookies = resp.cookies
                retry_url = f"http://{host}:{port}/api/usage?period={period}"
                resp = await client.get(retry_url, cookies=cookies)

            if resp.status_code != 200:
                raise HTTPException(resp.status_code, f"Agent returned {resp.status_code}")
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


# ── Orchestrator proxy endpoints ──


@app.post("/fd/orchestrator/{host}/{port}/prepare-tasks")
async def proxy_prepare_tasks(
    host: str, port: int, request: Request,
    user: dict | None = _required_user_dep,
):
    """Proxy prepare-tasks to a CC agent's orchestrator."""
    import httpx

    auth = _resolve_agent_auth(port)
    params = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/orchestrator/prepare-tasks{params}"
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, content=body, headers={"Content-Type": "application/json"})
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.post("/fd/orchestrator/{host}/{port}/run-tasks")
async def proxy_run_tasks(
    host: str, port: int, request: Request,
    user: dict | None = _required_user_dep,
):
    """Proxy run-tasks to a CC agent's orchestrator.

    Note: This is a long-running request. Real-time progress comes
    via the agent WebSocket (orchestrator_event messages).
    """
    import httpx

    auth = _resolve_agent_auth(port)
    params = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/orchestrator/run-tasks{params}"
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(url, content=body, headers={"Content-Type": "application/json"})
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/orchestrator/{host}/{port}/workspace")
async def proxy_workspace_snapshot(
    host: str, port: int, token: str = "",
    user: dict | None = _required_user_dep,
):
    """Proxy workspace snapshot from a CC agent's orchestrator."""
    import httpx

    auth = token or _resolve_agent_auth(port)
    params = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/orchestrator/workspace{params}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/orchestrator/{host}/{port}/traces")
async def proxy_traces(
    host: str, port: int, token: str = "",
    user: dict | None = _required_user_dep,
):
    """Proxy trace spans from a CC agent's orchestrator."""
    import httpx

    auth = token or _resolve_agent_auth(port)
    params = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/orchestrator/traces{params}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


# ── Memory transfer (curated insights + reflections) ──


@app.get("/fd/agent-memory/{host}/{port}/export")
async def proxy_memory_export(
    host: str, port: int,
    token: str = "",
    min_importance: int = 0,
    include_expired: bool = False,
    reflection_limit: int = 50,
    include_semantic: bool = False,
    semantic_limit: int = 1000,
    semantic_min_chars: int = 100,
    semantic_sources: str = "",
    semantic_include_imported: bool = False,
    user: dict | None = _required_user_dep,
):
    """Proxy a curated-memory bundle download from a CC agent.

    Streams the agent's ``/api/memory/export`` response back to the
    Flight Deck client, preserving the file download headers so the
    browser saves it as a JSON file.

    When ``include_semantic`` is true, the bundle additionally contains
    a text-only dump of the agent's semantic memory chunks (no vectors;
    the importing agent re-embeds with its own provider).
    """
    import httpx
    from fastapi.responses import Response

    auth = token or _resolve_agent_auth(port)
    qs_parts = []
    if auth:
        qs_parts.append(f"token={auth}")
    qs_parts.append(f"min_importance={int(min_importance)}")
    if include_expired:
        qs_parts.append("include_expired=1")
    qs_parts.append(f"reflection_limit={int(reflection_limit)}")
    if include_semantic:
        qs_parts.append("include_semantic=1")
        qs_parts.append(f"semantic_limit={int(semantic_limit)}")
        qs_parts.append(f"semantic_min_chars={int(semantic_min_chars)}")
        if semantic_sources:
            qs_parts.append(f"semantic_sources={semantic_sources}")
        if semantic_include_imported:
            qs_parts.append("semantic_include_imported=1")
    url = f"http://{host}:{port}/api/memory/export?{'&'.join(qs_parts)}"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, resp.text)
            disposition = resp.headers.get(
                "content-disposition",
                f'attachment; filename="captain-claw-memory-{host}-{port}.json"',
            )
            return Response(
                content=resp.content,
                media_type="application/json",
                headers={"Content-Disposition": disposition},
            )
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.post("/fd/agent-memory/{host}/{port}/import")
async def proxy_memory_import(
    host: str, port: int,
    request: Request,
    token: str = "",
    min_importance: int = 0,
    source_label: str = "",
    stage_conflicts: bool = False,
    skip_semantic: bool = False,
    user: dict | None = _required_user_dep,
):
    """Proxy a curated-memory bundle into a CC agent.

    Body: the JSON bundle previously downloaded via ``proxy_memory_export``
    (or any compatible bundle from another agent). When ``stage_conflicts``
    is true, conflicting decision/preference/workflow insights are routed
    to the agent's pending-review queue instead of being silently deduped.
    When ``skip_semantic`` is true, any ``semantic_chunks`` present in the
    bundle are dropped on the server side (curated-only import).
    """
    import httpx

    auth = token or _resolve_agent_auth(port)
    qs_parts = []
    if auth:
        qs_parts.append(f"token={auth}")
    qs_parts.append(f"min_importance={int(min_importance)}")
    if source_label:
        qs_parts.append(f"source_label={source_label}")
    if stage_conflicts:
        qs_parts.append("stage_conflicts=1")
    if skip_semantic:
        qs_parts.append("skip_semantic=1")
    url = f"http://{host}:{port}/api/memory/import?{'&'.join(qs_parts)}"
    body = await request.body()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                url,
                content=body,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/agent-memory/{host}/{port}/reflections/imported")
async def proxy_list_imported_reflections(
    host: str, port: int,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """List imported (staged) reflections on a CC agent for the merge picker."""
    import httpx

    auth = token or _resolve_agent_auth(port)
    qs = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/memory/reflections/imported{qs}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.post("/fd/agent-memory/{host}/{port}/reflections/merge")
async def proxy_merge_reflection(
    host: str, port: int,
    request: Request,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Trigger a personality-preserving reflection merge on a CC agent.

    Body: ``{"label": "<imported-subdir>", "filename": "<optional>"}``.
    The agent runs its reflection-merge LLM flow and writes the result as
    the new active reflection.
    """
    import httpx

    auth = token or _resolve_agent_auth(port)
    qs = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/memory/reflections/merge{qs}"
    body = await request.body()

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                url,
                content=body,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


# ── Semantic memory import management ──


@app.get("/fd/agent-memory/{host}/{port}/semantic/labels")
async def proxy_list_semantic_imports(
    host: str, port: int,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """List imported semantic sources on a CC agent."""
    import httpx

    auth = token or _resolve_agent_auth(port)
    qs = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/memory/semantic/labels{qs}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.delete("/fd/agent-memory/{host}/{port}/semantic/labels/{label}")
async def proxy_delete_semantic_import(
    host: str, port: int, label: str,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Purge one imported semantic source on a CC agent."""
    import httpx

    auth = token or _resolve_agent_auth(port)
    qs = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/memory/semantic/labels/{label}{qs}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(url)
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


# ── Pending-review insights (stage_conflicts queue) ──


@app.get("/fd/agent-insights/{host}/{port}/pending")
async def proxy_list_pending_insights(
    host: str, port: int,
    token: str = "",
    category: str = "",
    limit: int = 100,
    user: dict | None = _required_user_dep,
):
    """List the agent's staged insight conflicts awaiting review."""
    import httpx

    auth = token or _resolve_agent_auth(port)
    qs_parts = []
    if auth:
        qs_parts.append(f"token={auth}")
    if category:
        qs_parts.append(f"category={category}")
    qs_parts.append(f"limit={int(limit)}")
    url = f"http://{host}:{port}/api/insights/pending?{'&'.join(qs_parts)}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/agent-insights/{host}/{port}/pending/count")
async def proxy_count_pending_insights(
    host: str, port: int,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Cheap poll for the UI badge."""
    import httpx

    auth = token or _resolve_agent_auth(port)
    qs = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/insights/pending/count{qs}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.post("/fd/agent-insights/{host}/{port}/pending/{pending_id}/approve")
async def proxy_approve_pending_insight(
    host: str, port: int, pending_id: str,
    token: str = "",
    supersede: bool = True,
    user: dict | None = _required_user_dep,
):
    """Promote a staged insight into the live table."""
    import httpx

    auth = token or _resolve_agent_auth(port)
    qs_parts = []
    if auth:
        qs_parts.append(f"token={auth}")
    qs_parts.append(f"supersede={'1' if supersede else '0'}")
    url = (
        f"http://{host}:{port}/api/insights/pending/{pending_id}/approve"
        f"?{'&'.join(qs_parts)}"
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url)
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.post("/fd/agent-insights/{host}/{port}/pending/{pending_id}/reject")
async def proxy_reject_pending_insight(
    host: str, port: int, pending_id: str,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Drop a staged insight without promoting it."""
    import httpx

    auth = token or _resolve_agent_auth(port)
    qs = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/insights/pending/{pending_id}/reject{qs}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url)
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


# ── Today aggregator proxy endpoints (reflections / cron / intuitions / skills) ──


async def _proxy_get_json(host: str, port: int, path: str, token: str = "", timeout: float = 15.0):
    """Helper: GET an agent endpoint and return parsed JSON or raise HTTPException."""
    import httpx
    auth = token or _resolve_agent_auth(port)
    sep = "&" if "?" in path else "?"
    url = f"http://{host}:{port}{path}"
    if auth:
        url = f"{url}{sep}token={auth}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/agent-reflections/{host}/{port}")
async def proxy_list_reflections(
    host: str, port: int,
    token: str = "",
    limit: int = 20,
    user: dict | None = _required_user_dep,
):
    """List recent reflections on a CC agent."""
    return await _proxy_get_json(host, port, f"/api/reflections?limit={int(limit)}", token=token)


@app.get("/fd/agent-reflections/{host}/{port}/latest")
async def proxy_latest_reflection(
    host: str, port: int,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Get the latest active reflection from a CC agent."""
    return await _proxy_get_json(host, port, "/api/reflections/latest", token=token)


@app.get("/fd/agent-cron/{host}/{port}")
async def proxy_list_cron(
    host: str, port: int,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """List cron jobs on a CC agent."""
    return await _proxy_get_json(host, port, "/api/cron/jobs", token=token)


@app.get("/fd/agent-intuitions/{host}/{port}")
async def proxy_list_intuitions(
    host: str, port: int,
    token: str = "",
    limit: int = 20,
    user: dict | None = _required_user_dep,
):
    """List recent intuitions (nervous system) on a CC agent."""
    return await _proxy_get_json(
        host, port, f"/api/nervous-system?limit={int(limit)}", token=token
    )


# ── Skills proxy endpoints ──


@app.get("/fd/agent-skills/{host}/{port}")
async def proxy_list_skills(
    host: str, port: int,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """List installed skills on a CC agent."""
    return await _proxy_get_json(host, port, "/api/skills", token=token, timeout=30.0)


@app.post("/fd/agent-skills/{host}/{port}/install")
async def proxy_install_skill(
    host: str, port: int,
    request: Request,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Install a skill on a CC agent from a GitHub URL."""
    import httpx

    auth = token or _resolve_agent_auth(port)
    qs = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/skills/install{qs}"
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                url, content=body, headers={"Content-Type": "application/json"}
            )
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.post("/fd/agent-skills/{host}/{port}/install-upload")
async def proxy_install_skill_upload(
    host: str, port: int,
    request: Request,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Install a skill on a CC agent from an uploaded .md or .zip file."""
    import httpx

    auth = token or _resolve_agent_auth(port)
    qs = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/skills/install-upload{qs}"
    body = await request.body()
    content_type = request.headers.get("content-type", "application/octet-stream")
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                url, content=body, headers={"Content-Type": content_type}
            )
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.post("/fd/agent-skills/{host}/{port}/toggle")
async def proxy_toggle_skill(
    host: str, port: int,
    request: Request,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Enable or disable a skill on a CC agent."""
    import httpx

    auth = token or _resolve_agent_auth(port)
    qs = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/skills/toggle{qs}"
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url, content=body, headers={"Content-Type": "application/json"}
            )
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


# ── Captain Claw Game proxy endpoints ──


async def _proxy_post_json(
    host: str, port: int, path: str, token: str, body: bytes, timeout: float = 30.0,
):
    """Helper: POST a JSON body to an agent endpoint and return parsed JSON."""
    import httpx
    auth = token or _resolve_agent_auth(port)
    sep = "&" if "?" in path else "?"
    url = f"http://{host}:{port}{path}"
    if auth:
        url = f"{url}{sep}token={auth}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url, content=body or b"{}",
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


async def _proxy_delete_json(
    host: str, port: int, path: str, token: str, timeout: float = 15.0,
):
    """Helper: DELETE an agent endpoint and return parsed JSON."""
    import httpx
    auth = token or _resolve_agent_auth(port)
    sep = "&" if "?" in path else "?"
    url = f"http://{host}:{port}{path}"
    if auth:
        url = f"{url}{sep}token={auth}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.delete(url)
            if resp.status_code == 200:
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/agent-games/{host}/{port}/worlds")
async def proxy_games_worlds(
    host: str, port: int,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """List available demo worlds on a CC agent."""
    return await _proxy_get_json(host, port, "/api/games/worlds", token=token)


@app.get("/fd/agent-games/{host}/{port}")
async def proxy_games_list(
    host: str, port: int,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """List active games on a CC agent."""
    return await _proxy_get_json(host, port, "/api/games", token=token)


@app.post("/fd/agent-games/{host}/{port}")
async def proxy_games_create(
    host: str, port: int,
    request: Request,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Create a new game on a CC agent."""
    return await _proxy_post_json(host, port, "/api/games", token, await request.body())


@app.post("/fd/agent-games/{host}/{port}/generate")
async def proxy_games_generate(
    host: str, port: int,
    request: Request,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Generate a new game from a WorldSpec."""
    return await _proxy_post_json(
        host, port, "/api/games/generate", token, await request.body(),
        timeout=180.0,  # pipeline mode makes multiple LLM calls
    )


@app.get("/fd/agent-games/{host}/{port}/{game_id}")
async def proxy_games_get(
    host: str, port: int, game_id: str,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Get full state of a game."""
    return await _proxy_get_json(host, port, f"/api/games/{game_id}", token=token)


@app.post("/fd/agent-games/{host}/{port}/{game_id}/restart")
async def proxy_games_restart(
    host: str, port: int, game_id: str,
    request: Request,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Reset a game back to tick 0."""
    return await _proxy_post_json(
        host, port, f"/api/games/{game_id}/restart", token, await request.body(),
    )


@app.post("/fd/agent-games/{host}/{port}/{game_id}/seats")
async def proxy_games_reassign_seats(
    host: str, port: int, game_id: str,
    request: Request,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Reassign seat types for a game (only at tick 0)."""
    return await _proxy_post_json(
        host, port, f"/api/games/{game_id}/seats", token, await request.body(),
    )


@app.post("/fd/agent-games/{host}/{port}/{game_id}/tick")
async def proxy_games_tick(
    host: str, port: int, game_id: str,
    request: Request,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Advance one tick."""
    return await _proxy_post_json(
        host, port, f"/api/games/{game_id}/tick", token, await request.body(),
    )


@app.post("/fd/agent-games/{host}/{port}/{game_id}/intent")
async def proxy_games_intent(
    host: str, port: int, game_id: str,
    request: Request,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Queue a human intent for the next tick."""
    return await _proxy_post_json(
        host, port, f"/api/games/{game_id}/intent", token, await request.body(),
    )


@app.post("/fd/agent-games/{host}/{port}/{game_id}/natural")
async def proxy_games_natural(
    host: str, port: int, game_id: str,
    request: Request,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Parse natural language input and queue as intent."""
    return await _proxy_post_json(
        host, port, f"/api/games/{game_id}/natural", token, await request.body(),
    )


@app.post("/fd/agent-games/{host}/{port}/{game_id}/replay")
async def proxy_games_replay(
    host: str, port: int, game_id: str,
    request: Request,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Replay a game from its intent log."""
    return await _proxy_post_json(
        host, port, f"/api/games/{game_id}/replay", token, await request.body(),
    )


@app.delete("/fd/agent-games/{host}/{port}/{game_id}")
async def proxy_games_delete(
    host: str, port: int, game_id: str,
    token: str = "",
    user: dict | None = _required_user_dep,
):
    """Drop a game session from the agent."""
    return await _proxy_delete_json(host, port, f"/api/games/{game_id}", token=token)


# ── Process agent endpoints ──


class ProcessInfo(BaseModel):
    slug: str
    name: str
    description: str = ""
    status: str  # running | stopped
    web_port: int
    web_auth: str = ""
    pid: int | None = None
    provider: str = ""
    model: str = ""


class ProcessActionResult(BaseModel):
    ok: bool
    slug: str
    message: str = ""


@app.get("/fd/processes", response_model=list[ProcessInfo])
async def list_processes(request: Request, user: dict | None = _required_user_dep):
    """List all Flight Deck managed process agents."""
    registry = _load_process_registry()
    user_id = getattr(request.state, "user_id", "")
    result = []
    for slug, entry in registry.items():
        if AUTH_ENABLED and user_id and entry.get("owner", "") != user_id:
            continue
        alive = _process_is_alive(slug)
        result.append(ProcessInfo(
            slug=slug,
            name=entry.get("name", slug),
            description=entry.get("description", ""),
            status="running" if alive else "stopped",
            web_port=entry.get("web_port", 0),
            web_auth=entry.get("web_auth", ""),
            pid=entry.get("pid") if alive else None,
            provider=entry.get("provider", ""),
            model=entry.get("model", ""),
        ))
    return result


@app.post("/fd/spawn-process", response_model=ProcessActionResult)
async def spawn_process(config: AgentConfig, request: Request, user: dict | None = _optional_user_dep):
    """Spawn a new Captain Claw process agent (pip-installed, no Docker)."""
    # Serialise spawns so two concurrent requests can't both land on the same
    # port between the port-pick and the Popen. The lock also covers a small
    # settle delay after launch to let the child's TCPSite.bind succeed (or
    # drift + announce) before the next spawn's port probe runs.
    async with _get_spawn_lock():
        return await _spawn_process_locked(config, request, user)


async def _spawn_process_locked(config: AgentConfig, request: Request, user: dict | None):
    # Rate limiting & agent count check
    if AUTH_ENABLED and user:
        check_api_rate_limit(user)
        check_spawn_rate_limit(user)
        # Count existing processes for this user
        registry = _load_process_registry()
        user_id = user["id"]
        owned = [e for e in registry.values() if e.get("owner") == user_id]
        await check_agent_count_limit(user, len(owned))

    slug = _slug(config.name)

    # Check if already running
    if _process_is_alive(slug):
        raise HTTPException(400, f"Process '{slug}' is already running. Stop it first or use a different name.")

    # Ensure port is available; find a free one if not
    if config.web_enabled and (config.web_port <= 0 or not _is_port_available(config.web_port)):
        config.web_port = _find_available_port(config.web_port if config.web_port > 0 else 24080)

    # Auto-generate auth token if none provided — prevents unauthenticated
    # direct access to agent ports bypassing Flight Deck.
    if config.web_enabled and not config.web_auth_token:
        config.web_auth_token = secrets.token_urlsafe(32)

    # Prepare data directory
    agent_dir = DATA_DIR / slug
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "workspace").mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "skills").mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "home-config").mkdir(parents=True, exist_ok=True)

    # Write config files with local paths
    config_yaml = _build_process_config_yaml(config, agent_dir)
    (agent_dir / "config.yaml").write_text(config_yaml)
    (agent_dir / "data" / "home-config" / "config.yaml").write_text(config_yaml)

    env_content = _build_env(config)
    (agent_dir / ".env").write_text(env_content)

    # Build environment variables
    environment = dict(os.environ)
    if env_content:
        for line in env_content.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                environment[k] = v
    for ev in config.env_vars:
        if ev.get("key"):
            environment[ev["key"]] = ev.get("value", "")

    # Resolve owner: authenticated user > owner_hint > infer from existing registry
    owner_id = getattr(request.state, "user_id", "") or config.owner_hint
    if not owner_id:
        # Fallback: inherit owner from an existing agent in the registry.
        # Covers the case where an internal caller (e.g. Old Man spawned before
        # the FD_OWNER_ID env var was added) doesn't have owner info.
        _reg = _load_process_registry()
        for _entry in _reg.values():
            if _entry.get("owner"):
                owner_id = _entry["owner"]
                break

    # Tell agents how to reach Flight Deck internally (for Telegram, Discord, etc.)
    if "FD_URL" not in environment:
        fd_port = os.environ.get("FD_PORT", "25080")
        environment["FD_URL"] = f"http://localhost:{fd_port}"

    # Pass owner ID so child agents can propagate ownership when spawning
    if owner_id:
        environment["FD_OWNER_ID"] = owner_id

    # Slug for port-fallback callbacks: lets the agent tell FD which actual
    # port it bound to if the requested one was already in use.
    environment["FD_AGENT_SLUG"] = slug

    # Set HOME to the agent's home-config directory so captain-claw
    # picks up ~/.captain-claw/config.yaml from there
    environment["HOME"] = str(agent_dir / "data" / "home-config-parent")
    home_cc_dir = agent_dir / "data" / "home-config-parent" / ".captain-claw"
    home_cc_dir.mkdir(parents=True, exist_ok=True)
    # Symlink or copy home-config -> ~/.captain-claw
    hc_config = agent_dir / "data" / "home-config" / "config.yaml"
    hc_target = home_cc_dir / "config.yaml"
    if hc_target.exists() or hc_target.is_symlink():
        hc_target.unlink()
    shutil.copy2(str(hc_config), str(hc_target))

    # Write cognitive mode file for the agent.
    if config.cognitive_mode and config.cognitive_mode != "neutra":
        mode_file = home_cc_dir / "cognitive_mode.txt"
        mode_file.write_text(config.cognitive_mode, encoding="utf-8")

    # Open log file
    log_file = agent_dir / "process.log"
    log_fh = open(log_file, "a")

    # Resolve captain-claw-web binary: bundled (PyInstaller) or PATH
    cc_web_bin = "captain-claw-web"
    if getattr(sys, "_MEIPASS", None):
        # In standalone build, the binary is next to this executable
        bundled = Path(sys._MEIPASS).parent / "captain-claw-web"
        if bundled.exists():
            cc_web_bin = str(bundled)

    # IMPORTANT: write the registry entry BEFORE we Popen the child. The
    # child can bind, drift to a fallback port, and POST to /announce-port
    # in the milliseconds between Popen and the post-Popen registry write —
    # if we wrote the registry after Popen we'd race the announce and
    # silently overwrite the drifted port back to the original (stale)
    # value. Pre-writing the entry also guarantees the announce-port
    # endpoint can find the slug.
    registry = _load_process_registry()
    registry[slug] = {
        "slug": slug,
        "name": config.name or slug,
        "description": config.description,
        "web_port": config.web_port,
        "web_auth": config.web_auth_token,
        "pid": None,  # filled in below once Popen returns
        "provider": config.provider,
        "model": config.model,
        "owner": owner_id,
    }
    _save_process_registry(registry)

    try:
        proc = subprocess.Popen(
            [cc_web_bin, "--port", str(config.web_port)],
            cwd=str(agent_dir),
            env=environment,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Detach from parent
        )
    except FileNotFoundError:
        log_fh.close()
        raise HTTPException(500, f"captain-claw-web not found at '{cc_web_bin}'. Install captain-claw via pip first.")
    except Exception as exc:
        log_fh.close()
        raise HTTPException(500, f"Failed to start process: {exc}")

    _processes[slug] = proc

    # Stamp the live PID. Read-modify-write so we don't clobber a port the
    # child may have already announced between Popen and now.
    registry = _load_process_registry()
    if slug in registry:
        registry[slug]["pid"] = proc.pid
        _save_process_registry(registry)

    # Log usage
    if AUTH_ENABLED and user:
        db = app.state.fd_db
        await db.log_usage(user["id"], "agent_spawn", json.dumps({"agent": slug, "type": "process", "provider": config.provider, "model": config.model}))

    # Let the child actually bind its TCP port (and potentially announce a
    # drift back to us) before the spawn lock is released and the next queued
    # spawn runs its port probe. Without this window two back-to-back spawns
    # can both see the same port as "free".
    settle_s = float(os.environ.get("FD_SPAWN_SETTLE_S", "0.3"))
    if settle_s > 0:
        await asyncio.sleep(settle_s)

    # After settle, re-read the registry: if the child drifted and announced
    # a new port, we want to surface the actual bound port in our response
    # (and to peers we notify) instead of the originally requested one.
    final_registry = _load_process_registry()
    actual_port = final_registry.get(slug, {}).get("web_port", config.web_port)
    log.info(
        "spawn-process: post-settle registry read",
        slug=slug,
        requested=config.web_port,
        actual=actual_port,
        drifted=actual_port != config.web_port,
    )
    if actual_port != config.web_port:
        log.info(
            "spawn-process: child drifted to a fallback port",
            slug=slug,
            requested=config.web_port,
            actual=actual_port,
        )
        config.web_port = actual_port
    else:
        # No drift detected yet — but the child's announce might still be in
        # flight (network/scheduler latency, slow startup). Schedule a deferred
        # re-check that won't block the spawn response but will still surface
        # the corrected port to the FD UI on its next poll.
        async def _late_drift_check():
            for _ in range(10):  # up to ~5 s
                await asyncio.sleep(0.5)
                _later = _load_process_registry().get(slug, {}).get("web_port", 0)
                if _later and _later != config.web_port:
                    log.info(
                        "spawn-process: late drift detected after spawn returned",
                        slug=slug,
                        was=config.web_port,
                        now=_later,
                    )
                    return
        try:
            asyncio.get_event_loop().create_task(_late_drift_check())
        except RuntimeError:
            pass

    # Notify other agents about the new peer (scoped to same owner). Done
    # AFTER the settle so the announced port is used, not the stale one.
    if config.web_enabled:
        _schedule_fleet_notify(config.name or slug, config.web_port, owner_id=owner_id)

    return ProcessActionResult(ok=True, slug=slug, message=f"Process agent '{slug}' spawned (PID {proc.pid}, port {config.web_port})")


def _verify_process_owner(slug: str, user_id: str) -> dict:
    """Check that a process exists and belongs to the user. Returns the registry entry."""
    registry = _load_process_registry()
    entry = registry.get(slug)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Process '{slug}' not found")
    if AUTH_ENABLED and user_id and entry.get("owner", "") != user_id:
        raise HTTPException(status_code=404, detail=f"Process '{slug}' not found")
    return entry


def _do_stop_process(slug: str) -> ProcessActionResult:
    """Internal helper to stop a process agent (no auth check)."""
    if not _process_is_alive(slug):
        return ProcessActionResult(ok=True, slug=slug, message="Already stopped")

    proc = _processes.get(slug)
    registry = _load_process_registry()
    pid = proc.pid if proc else registry.get(slug, {}).get("pid")

    if pid:
        _kill_pid(pid)

    # Update registry — mark as intentionally stopped
    if slug in registry:
        registry[slug]["pid"] = None
        registry[slug]["stopped"] = True
        _save_process_registry(registry)

    _processes.pop(slug, None)
    return ProcessActionResult(ok=True, slug=slug, message="Stopped")


def _do_start_process(slug: str) -> ProcessActionResult:
    """Internal helper to start a process agent (no auth check)."""
    registry = _load_process_registry()
    entry = registry.get(slug)
    if not entry:
        raise HTTPException(404, f"Process '{slug}' not found in registry")

    if _process_is_alive(slug):
        return ProcessActionResult(ok=True, slug=slug, message="Already running")

    if not (DATA_DIR / slug).is_dir():
        raise HTTPException(404, f"Agent directory not found: {DATA_DIR / slug}")

    # Clear stopped flag — user is intentionally starting this agent
    entry.pop("stopped", None)

    if not _start_registered_process(slug, entry):
        raise HTTPException(500, "Failed to start process (captain-claw-web not found?)")

    _save_process_registry(registry)

    return ProcessActionResult(ok=True, slug=slug, message=f"Started (PID {entry.get('pid', '?')})")


class AnnouncePortRequest(BaseModel):
    """Agent → FD callback: actual port the agent successfully bound to."""
    port: int
    auth: str = ""  # web_auth_token used as a shared secret to authenticate the callback


@app.post("/fd/processes/{slug}/announce-port", response_model=ProcessActionResult)
async def announce_process_port(slug: str, body: AnnouncePortRequest):
    """Update a process's actual web port after the agent fell back to a free
    port (because the requested one was already in use). Authenticated via the
    process's existing web_auth token, so no user session is required.

    Side-effects: persists the new port in the registry and re-broadcasts the
    fleet membership so peer agents and Flight Deck UI clients learn the new
    address.
    """
    log.info("announce-port: received", slug=slug, new_port=body.port)
    registry = _load_process_registry()
    entry = registry.get(slug)
    if not entry:
        log.warning("announce-port: slug not in registry", slug=slug, registry_slugs=list(registry.keys()))
        raise HTTPException(404, f"Process '{slug}' not found")

    expected_auth = entry.get("web_auth", "")
    # Compare with constant-time helper to avoid trivial timing leaks.
    import hmac
    if expected_auth and not hmac.compare_digest(expected_auth, body.auth or ""):
        log.warning("announce-port: auth mismatch", slug=slug)
        raise HTTPException(401, "Invalid auth token for port announce")

    if body.port <= 0 or body.port > 65535:
        raise HTTPException(400, f"Invalid port: {body.port}")

    old_port = entry.get("web_port", 0)
    if old_port == body.port:
        log.info("announce-port: unchanged", slug=slug, port=body.port)
        return ProcessActionResult(ok=True, slug=slug, message=f"Port unchanged ({body.port})")

    entry["web_port"] = body.port
    _save_process_registry(registry)
    log.info("announce-port: registry updated", slug=slug, old_port=old_port, new_port=body.port)

    # Verify the write actually landed by re-reading. Catches the (theoretical)
    # case where another writer raced us between save and return.
    _verify = _load_process_registry().get(slug, {}).get("web_port", 0)
    if _verify != body.port:
        log.error(
            "announce-port: registry write was clobbered immediately after save!",
            slug=slug,
            wrote=body.port,
            now_reads=_verify,
        )

    # Re-notify the fleet so peers update their connection details.
    owner_id = entry.get("owner", "")
    name = entry.get("name", slug)
    _schedule_fleet_notify(name, body.port, event="rebound", owner_id=owner_id)

    return ProcessActionResult(
        ok=True,
        slug=slug,
        message=f"Updated port: {old_port} → {body.port}",
    )


@app.post("/fd/processes/{slug}/stop", response_model=ProcessActionResult)
async def stop_process(slug: str, request: Request, user: dict | None = _required_user_dep):
    """Stop a running process agent."""
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, _do_stop_process, slug)


@app.post("/fd/processes/{slug}/start", response_model=ProcessActionResult)
async def start_process(slug: str, request: Request, user: dict | None = _required_user_dep):
    """Start a stopped process agent."""
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, _do_start_process, slug)


@app.post("/fd/processes/{slug}/restart", response_model=ProcessActionResult)
async def restart_process(slug: str, request: Request, user: dict | None = _required_user_dep):
    """Restart a process agent."""
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    _do_stop_process(slug)
    import time
    time.sleep(1)
    return _do_start_process(slug)


@app.delete("/fd/processes/{slug}", response_model=ProcessActionResult)
async def remove_process(slug: str, force: bool = False, request: Request = None, user: dict | None = _required_user_dep):
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    """Remove a process agent from the registry. Stops it first if running."""
    if _process_is_alive(slug):
        _do_stop_process(slug)

    registry = _load_process_registry()
    registry.pop(slug, None)
    _save_process_registry(registry)
    _processes.pop(slug, None)

    return ProcessActionResult(ok=True, slug=slug, message=f"Removed '{slug}' from registry")


@app.get("/fd/processes/{slug}/logs")
async def process_logs(slug: str, tail: int = 200, request: Request = None, user: dict | None = _required_user_dep):
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    """Read logs from a process agent's log file."""
    log_file = DATA_DIR / slug / "process.log"
    if not log_file.is_file():
        return {"logs": "(no logs yet)"}
    try:
        lines = log_file.read_text().splitlines()
        tail_lines = lines[-tail:] if len(lines) > tail else lines
        return {"logs": "\n".join(tail_lines)}
    except Exception as exc:
        return {"logs": f"(error reading logs: {exc})"}


@app.post("/fd/processes/{slug}/clone", response_model=ProcessActionResult)
async def clone_process(slug: str, req: CloneRequest, request: Request, user: dict | None = _required_user_dep):
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    """Clone a process agent with a new name and port."""
    registry = _load_process_registry()
    entry = registry.get(slug)
    if not entry:
        raise HTTPException(404, f"Process '{slug}' not found")

    new_name = req.new_name.strip()
    if not new_name:
        raise HTTPException(400, "Name is required")
    new_slug = _slug(new_name)

    if new_slug in registry:
        if _process_is_alive(new_slug):
            raise HTTPException(400, f"Process '{new_slug}' already running.")

    old_agent_dir = DATA_DIR / slug
    new_agent_dir = DATA_DIR / new_slug

    # Create new data directory
    new_agent_dir.mkdir(parents=True, exist_ok=True)
    (new_agent_dir / "data" / "workspace").mkdir(parents=True, exist_ok=True)
    (new_agent_dir / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (new_agent_dir / "data" / "skills").mkdir(parents=True, exist_ok=True)
    (new_agent_dir / "data" / "home-config").mkdir(parents=True, exist_ok=True)

    # Copy config files
    for fname in ("config.yaml", ".env"):
        src = old_agent_dir / fname
        if src.is_file():
            shutil.copy2(str(src), str(new_agent_dir / fname))
    src_hc = old_agent_dir / "data" / "home-config" / "config.yaml"
    if src_hc.is_file():
        shutil.copy2(str(src_hc), str(new_agent_dir / "data" / "home-config" / "config.yaml"))

    # Find available port
    old_port = entry.get("web_port", 24080)
    new_port = _find_available_port(old_port + 1)

    # Update config.yaml with new port and name
    cfg_path = new_agent_dir / "config.yaml"
    if cfg_path.is_file():
        cfg_text = cfg_path.read_text()
        cfg_text = cfg_text.replace(f"port: {old_port}", f"port: {new_port}")
        old_name = entry.get("name", slug)
        if old_name:
            cfg_text = cfg_text.replace(f"instance_name: {old_name}", f"instance_name: {new_name}")
        # Update local paths to point to new agent dir
        cfg_text = cfg_text.replace(str(old_agent_dir), str(new_agent_dir))
        cfg_path.write_text(cfg_text)
        hc_path = new_agent_dir / "data" / "home-config" / "config.yaml"
        if hc_path.is_file():
            hc_path.write_text(cfg_text)

    # Register clone (but don't start it)
    registry[new_slug] = {
        "slug": new_slug,
        "name": new_name,
        "description": "",
        "web_port": new_port,
        "web_auth": entry.get("web_auth", ""),
        "pid": None,
        "provider": entry.get("provider", ""),
        "model": entry.get("model", ""),
    }
    _save_process_registry(registry)

    return ProcessActionResult(ok=True, slug=new_slug, message=f"Cloned '{slug}' → '{new_slug}' (port {new_port})")


async def _get_system_config() -> dict:
    """Load system config from DB. Returns defaults when auth disabled."""
    from captain_claw.flight_deck.admin_routes import _load_system_config, SYSTEM_CONFIG_DEFAULTS
    if not AUTH_ENABLED or not hasattr(app.state, "fd_db"):
        return {**SYSTEM_CONFIG_DEFAULTS}
    raw = await app.state.fd_db.get_system_setting("fd:system-config")
    return _load_system_config(raw)


@app.get("/fd/auth/status")
async def auth_status():
    """Check if auth is enabled and return system config flags (public endpoint)."""
    cfg = await _get_system_config()
    # When running inside a container, default docker spawn to False (no Docker socket)
    docker_default = not os.environ.get("CAPTAIN_CLAW_DOCKER")
    # Provide internal FD URL for agent-to-FD calls (inside container, use localhost)
    internal_fd_url = os.environ.get("FD_INTERNAL_URL", "")
    if not internal_fd_url and os.environ.get("CAPTAIN_CLAW_DOCKER"):
        internal_fd_url = "http://localhost:25080"
    return {
        "auth_enabled": AUTH_ENABLED,
        "docker_spawn_enabled": cfg.get("docker_spawn_enabled", docker_default),
        "internal_fd_url": internal_fd_url,
    }


# ── Agent Forge: LLM-powered team decomposition ──────────────────────────

class ForgeRequest(BaseModel):
    prompt: str
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""


@app.post("/fd/forge")
async def forge_decompose(
    body: ForgeRequest, request: Request,
    user: dict | None = _required_user_dep,
):
    """Use an LLM to decompose a user objective into a team of specialized agents."""
    if not body.prompt.strip():
        raise HTTPException(400, "prompt is required")

    # Load the forge system prompt from instructions
    instructions_dir = Path(__file__).parent.parent / "instructions"
    system_prompt_file = instructions_dir / "forge_decompose_system_prompt.md"
    if not system_prompt_file.is_file():
        raise HTTPException(500, "Forge system prompt not found")
    system_prompt = system_prompt_file.read_text()

    # Create an LLM provider and make the decomposition call
    try:
        from captain_claw.llm import create_provider, Message
        provider = create_provider(
            provider=body.provider,
            model=body.model,
            api_key=body.api_key or None,
            temperature=0.7,
            max_tokens=16384,
        )
        response = await provider.complete(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=body.prompt.strip()),
            ],
            temperature=0.7,
            max_tokens=16384,
        )
    except Exception as e:
        log.error("Forge LLM call failed", exc_info=True)
        raise HTTPException(502, f"LLM call failed: {e}")

    # Parse the JSON response
    content = response.content.strip()
    # Strip markdown fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(502, f"LLM returned invalid JSON: {content[:500]}")

    return result


@app.get("/fd/health")
def health():
    try:
        client = get_docker()
        client.ping()
        docker_ok = True
    except Exception:
        docker_ok = False
    # Always report ok if at least the server is running
    # (processes don't need Docker)
    return {"ok": True, "docker": docker_ok, "processes": True}


# ── Old Man preset ──

OLD_MAN_TOOLS = [
    "shell", "read", "write", "glob", "edit",
    "web_fetch", "web_search", "browser",
    "pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract",
    "pocket_tts", "send_mail", "clipboard",
    "screen_capture", "desktop_action",
    "scripts", "playbooks", "personality", "datastore", "insights",
    "cron", "summarize_files", "gws", "direct_api",
    "flight_deck",
]


def _build_old_man_config(
    *,
    provider: str = "ollama",
    model: str = "minimax-m2.7:cloud",
    api_key: str = "",
    web_port: int = 24080,
) -> AgentConfig:
    """Return an AgentConfig pre-filled for Old Man supervisor mode."""
    return AgentConfig(
        name="Old Man",
        description="Desktop supervisor — triages requests, delegates to fleet agents",
        provider=provider,
        model=model,
        provider_api_key=api_key,
        tools=OLD_MAN_TOOLS,
        web_port=web_port,
        # Old Man adds old_man.enabled + hotkey overrides via env var
        env_vars=[
            {"key": "CLAW_OLD_MAN__ENABLED", "value": "true"},
            {"key": "CLAW_TOOLS__SCREEN_CAPTURE__HOTKEY_ENABLED", "value": "true"},
            {"key": "FD_URL", "value": f"http://localhost:{os.environ.get('FD_PORT', '25080')}"},
        ],
    )


class OldManSpawnRequest(BaseModel):
    """Minimal request body for the Old Man quick-spawn endpoint."""
    provider: str = "ollama"
    model: str = "minimax-m2.7:cloud"
    api_key: str = ""
    web_port: int = 24080
    mode: str = "auto"  # "docker", "process", or "auto" (try docker first)


@app.post("/fd/spawn-old-man")
async def spawn_old_man(
    body: OldManSpawnRequest,
    request: Request,
    user: dict | None = _optional_user_dep,
):
    """One-click Old Man spawn — creates a supervisor agent with sane defaults.

    Tries Docker first (if available), falls back to process spawn.
    """
    config = _build_old_man_config(
        provider=body.provider,
        model=body.model,
        api_key=body.api_key,
        web_port=body.web_port,
    )

    use_docker = body.mode == "docker"
    use_process = body.mode == "process"

    if body.mode == "auto":
        # Prefer docker if available
        try:
            get_docker()
            sys_cfg = await _get_system_config()
            docker_default = not os.environ.get("CAPTAIN_CLAW_DOCKER")
            use_docker = sys_cfg.get("docker_spawn_enabled", docker_default)
            use_process = not use_docker
        except Exception:
            use_process = True

    if use_docker:
        return await spawn_agent(config, request, user)
    else:
        return await spawn_process(config, request, user)


# ── Static frontend serving ──

if STATIC_DIR.is_dir():
    # Serve built React assets
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}")
    async def spa_catch_all(path: str):
        """Serve the SPA — any non-API path returns index.html."""
        file = STATIC_DIR / path
        if file.is_file():
            return FileResponse(file)
        return FileResponse(STATIC_DIR / "index.html")


def main():
    """CLI entry point for Flight Deck."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Flight Deck — Captain Claw agent management UI")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=25080, help="Bind port (default: 25080)")
    parser.add_argument("--dev", action="store_true", help="Development mode (no static serving)")
    args = parser.parse_args()

    if not args.dev and not STATIC_DIR.is_dir():
        print(f"Warning: Static files not found at {STATIC_DIR}")
        print("Run 'cd flight-deck && npm run build' to build the frontend first.")
        print("Starting API-only mode (use --dev with Vite dev server).\n")

    log.info("Flight Deck starting on http://%s:%s", args.host, args.port)
    # log_config=None: keep our colored formatter from _configure_fd_logging()
    # instead of letting uvicorn slap its default handlers back on.
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
