"""Flight Deck backend — Docker container & process management for Captain Claw agents."""

from __future__ import annotations

import os
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

log = logging.getLogger("flight_deck")
from captain_claw.flight_deck.rate_limiter import (
    check_api_rate_limit, check_spawn_rate_limit, check_agent_count_limit,
    load_plan_limits_from_db_sync,
)

STATIC_DIR = Path(__file__).parent / "static"

# ── Config ──

DATA_DIR = Path(os.environ.get("FD_DATA_DIR", "./fd-data")).resolve()
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
    """Load process registry from disk."""
    if PROCESS_REGISTRY_FILE.is_file():
        try:
            return json.loads(PROCESS_REGISTRY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_process_registry(registry: dict[str, dict]):
    """Persist process registry to disk."""
    PROCESS_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROCESS_REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


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
    registry = _load_process_registry()
    restarted = []
    for slug, entry in registry.items():
        pid = entry.get("pid")
        if pid:
            try:
                os.kill(pid, 0)  # Still alive — just track it
                continue
            except (OSError, ProcessLookupError):
                pass
        # Process was registered but is dead — restart it
        if entry.get("web_port"):
            if _start_registered_process(slug, entry):
                restarted.append(slug)
            else:
                entry["pid"] = None
    _save_process_registry(registry)
    if restarted:
        print(f"Flight Deck: restarted {len(restarted)} process agent(s): {', '.join(restarted)}")


# ── App ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _reattach_processes()
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

# ── Auth & user routes ──

from captain_claw.flight_deck.auth_routes import router as auth_router
from captain_claw.flight_deck.settings_routes import router as settings_router
from captain_claw.flight_deck.chat_routes import router as chat_router
from captain_claw.flight_deck.admin_routes import router as admin_router

app.include_router(auth_router)
app.include_router(settings_router)
app.include_router(chat_router)
app.include_router(admin_router)


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

    # Docker
    network_mode: str = "host"
    restart_policy: str = "unless-stopped"
    extra_volumes: list[dict] = Field(default_factory=list)
    env_vars: list[dict] = Field(default_factory=list)


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
            "base_url": f"http://{dhost}:11434" if c.provider == "ollama" else "",
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
            "base_url": "http://127.0.0.1:11434" if c.provider == "ollama" else "",
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
async def list_containers(request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
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
async def spawn_agent(config: AgentConfig, request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
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
    if config.web_enabled and not _is_port_available(config.web_port):
        config.web_port = _find_available_port(config.web_port)

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

    # Labels for tracking
    user_id = getattr(request.state, "user_id", "")
    labels = {
        CONTAINER_LABEL: "true",
        OWNER_LABEL: user_id,
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
            _schedule_fleet_notify(config.name or slug, config.web_port, owner_id=user_id)
        return ContainerActionResult(ok=True, container_id=container.short_id, message=f"Agent '{slug}' spawned{port_info}")
    except docker.errors.ImageNotFound:
        raise HTTPException(404, f"Docker image '{config.image}' not found. Pull it first.")
    except docker.errors.APIError as exc:
        raise HTTPException(500, f"Docker error: {exc.explanation or str(exc)}")


@app.post("/fd/containers/{container_id}/stop", response_model=ContainerActionResult)
async def stop_container(container_id: str, request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    if c.status != "running":
        return ContainerActionResult(ok=True, container_id=c.short_id, message="Already stopped")
    c.stop(timeout=5)
    return ContainerActionResult(ok=True, container_id=c.short_id, message="Stopped")


@app.post("/fd/containers/{container_id}/start", response_model=ContainerActionResult)
async def start_container(container_id: str, request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    if c.status == "running":
        return ContainerActionResult(ok=True, container_id=c.short_id, message="Already running")
    try:
        c.start()
    except docker.errors.APIError as exc:
        explanation = exc.explanation or str(exc)
        raise HTTPException(500, f"Docker start failed: {explanation}")
    return ContainerActionResult(ok=True, container_id=c.short_id, message="Started")


@app.post("/fd/containers/{container_id}/restart", response_model=ContainerActionResult)
async def restart_container(container_id: str, request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    c.restart(timeout=5)
    return ContainerActionResult(ok=True, container_id=c.short_id, message="Restarted")


@app.delete("/fd/containers/{container_id}", response_model=ContainerActionResult)
async def remove_container(container_id: str, force: bool = False, request: Request = None, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    name = c.name
    c.remove(force=force)
    return ContainerActionResult(ok=True, container_id=container_id, message=f"Removed '{name}'")


class RebuildRequest(BaseModel):
    description: str = ""  # Frontend sends current description override


class CloneRequest(BaseModel):
    new_name: str  # Name for the cloned agent


@app.post("/fd/containers/{container_id}/rebuild", response_model=ContainerActionResult)
async def rebuild_container(container_id: str, request: Request, req: RebuildRequest | None = None, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
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
async def clone_container(container_id: str, req: CloneRequest, request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
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
async def container_logs(container_id: str, tail: int = 200, follow: bool = False, request: Request = None, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
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
    user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None,
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
    user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None,
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
        updated.append("config.yaml")
    if body.env is not None:
        env_file = agent_dir / ".env"
        env_file.write_text(body.env)
        updated.append(".env")

    return {"ok": True, "updated": updated, "message": "Restart the agent for changes to take effect."}


@app.get("/fd/containers/{container_id}")
async def get_container(container_id: str, request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
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
async def get_fleet(request: Request, user: dict | None = Depends(get_optional_user) if AUTH_ENABLED else None):
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
async def consult_peer(req: ConsultPeerRequest, request: Request, user: dict | None = Depends(get_optional_user) if AUTH_ENABLED else None):
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


@app.post("/fd/delegate-peer")
async def delegate_peer(req: DelegatePeerRequest, request: Request, user: dict | None = Depends(get_optional_user) if AUTH_ENABLED else None):
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
                await ws.send(json.dumps({"type": "notification", "content": callback_msg, "trigger_response": True}))
                log.info("delegate_callback: result delivered to source", source=req.source_name)
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


@app.get("/fd/agent-files/{host}/{port}")
async def agent_files(host: str, port: int, token: str = "", request: Request = None, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    """List files from a CC agent (proxied to avoid CORS), merged with workspace scan."""
    import httpx
    registered: list[dict] = []
    params = f"?token={token}" if token else ""
    url = f"http://{host}:{port}/api/files{params}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                registered = resp.json()
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
        if not registered:
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
async def transfer_file(req: TransferRequest, request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    """Download a file from one agent and upload it to another."""
    import httpx

    src_params = f"?token={req.src_auth}" if req.src_auth else ""
    dst_params = f"?token={req.dst_auth}" if req.dst_auth else ""

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        # Download from source
        dl_url = f"http://{req.src_host}:{req.src_port}/api/files/download?path={req.src_path}{src_params}"
        dl_resp = await client.get(dl_url)
        if dl_resp.status_code != 200:
            raise HTTPException(502, f"Source agent download failed: {dl_resp.status_code}")

        # Get filename from content-disposition or path
        cd = dl_resp.headers.get("content-disposition", "")
        if "filename=" in cd:
            filename = cd.split("filename=")[-1].strip('" ')
        else:
            filename = req.src_path.rsplit("/", 1)[-1]

        # Upload to destination
        up_url = f"http://{req.dst_host}:{req.dst_port}/api/file/upload{dst_params}"
        files = {"file": (filename, dl_resp.content, dl_resp.headers.get("content-type", "application/octet-stream"))}
        up_resp = await client.post(up_url, files=files)
        if up_resp.status_code != 200:
            raise HTTPException(502, f"Destination agent upload failed: {up_resp.status_code}")

        result = up_resp.json()
        return {"ok": True, "filename": filename, "dest_path": result.get("path", ""), "size": result.get("size", 0)}


@app.get("/fd/agent-file-download/{host}/{port}")
async def agent_file_download(host: str, port: int, path: str, token: str = "", request: Request = None, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    """Proxy file download from a CC agent."""
    import httpx
    import urllib.parse
    params = f"path={urllib.parse.quote(path)}"
    if token:
        params += f"&token={urllib.parse.quote(token)}"
    url = f"http://{host}:{port}/api/files/download?{params}"
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
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
async def agent_file_view(host: str, port: int, path: str, token: str = "", request: Request = None, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    """Proxy file view from a CC agent (inline, no download header)."""
    import httpx
    import urllib.parse
    params = f"path={urllib.parse.quote(path)}"
    if token:
        params += f"&token={urllib.parse.quote(token)}"
    # Try the /api/files/view endpoint first, fall back to /download
    url = f"http://{host}:{port}/api/files/view?{params}"
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
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
async def agent_file_upload(host: str, port: int, token: str = "", file: UploadFile = File(...), request: Request = None, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    """Proxy file upload to a CC agent."""
    import httpx

    params = f"?token={token}" if token else ""
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
async def agent_usage(host: str, port: int, token: str = "", period: str = "today", request: Request = None, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    """Proxy /api/usage from a CC agent."""
    import httpx
    params = f"?period={period}"
    if token:
        params += f"&token={token}"
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
async def list_processes(request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
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
async def spawn_process(config: AgentConfig, request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    """Spawn a new Captain Claw process agent (pip-installed, no Docker)."""
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
    if config.web_enabled and not _is_port_available(config.web_port):
        config.web_port = _find_available_port(config.web_port)

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

    # Tell agents how to reach Flight Deck internally (for Telegram, Discord, etc.)
    if "FD_URL" not in environment:
        fd_port = os.environ.get("FD_PORT", "25080")
        environment["FD_URL"] = f"http://localhost:{fd_port}"

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

    # Open log file
    log_file = agent_dir / "process.log"
    log_fh = open(log_file, "a")

    try:
        proc = subprocess.Popen(
            ["captain-claw-web", "--port", str(config.web_port)],
            cwd=str(agent_dir),
            env=environment,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Detach from parent
        )
    except FileNotFoundError:
        log_fh.close()
        raise HTTPException(500, "captain-claw-web not found. Install captain-claw via pip first.")
    except Exception as exc:
        log_fh.close()
        raise HTTPException(500, f"Failed to start process: {exc}")

    _processes[slug] = proc

    # Save to registry
    registry = _load_process_registry()
    registry[slug] = {
        "slug": slug,
        "name": config.name or slug,
        "description": config.description,
        "web_port": config.web_port,
        "web_auth": config.web_auth_token,
        "pid": proc.pid,
        "provider": config.provider,
        "model": config.model,
        "owner": getattr(request.state, "user_id", ""),
    }
    _save_process_registry(registry)

    # Log usage
    if AUTH_ENABLED and user:
        db = app.state.fd_db
        await db.log_usage(user["id"], "agent_spawn", json.dumps({"agent": slug, "type": "process", "provider": config.provider, "model": config.model}))

    # Notify other agents about the new peer (scoped to same owner)
    if config.web_enabled:
        _schedule_fleet_notify(config.name or slug, config.web_port, owner_id=getattr(request.state, "user_id", ""))

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

    # Update registry
    if slug in registry:
        registry[slug]["pid"] = None
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

    if not _start_registered_process(slug, entry):
        raise HTTPException(500, "Failed to start process (captain-claw-web not found?)")

    _save_process_registry(registry)

    return ProcessActionResult(ok=True, slug=slug, message=f"Started (PID {entry.get('pid', '?')})")


@app.post("/fd/processes/{slug}/stop", response_model=ProcessActionResult)
async def stop_process(slug: str, request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    """Stop a running process agent."""
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    return _do_stop_process(slug)


@app.post("/fd/processes/{slug}/start", response_model=ProcessActionResult)
async def start_process(slug: str, request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    """Start a stopped process agent."""
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    return _do_start_process(slug)


@app.post("/fd/processes/{slug}/restart", response_model=ProcessActionResult)
async def restart_process(slug: str, request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
    """Restart a process agent."""
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    _do_stop_process(slug)
    import time
    time.sleep(1)
    return _do_start_process(slug)


@app.delete("/fd/processes/{slug}", response_model=ProcessActionResult)
async def remove_process(slug: str, force: bool = False, request: Request = None, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
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
async def process_logs(slug: str, tail: int = 200, request: Request = None, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
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
async def clone_process(slug: str, req: CloneRequest, request: Request, user: dict | None = Depends(get_current_user) if AUTH_ENABLED else None):
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

    print(f"Flight Deck starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
