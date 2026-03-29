"""Flight Deck backend — Docker container management for Captain Claw agents."""

from __future__ import annotations

import os
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

import docker
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).parent / "static"

# ── Config ──

DATA_DIR = Path(os.environ.get("FD_DATA_DIR", "./fd-data")).resolve()
CONTAINER_LABEL = "flight-deck.managed"
CC_IMAGE_DEFAULT = "kstevica/captain-claw:latest"


# ── Docker client ──

_client: docker.DockerClient | None = None


def get_docker() -> docker.DockerClient:
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


# ── App ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if _client:
        _client.close()


app = FastAPI(title="Flight Deck", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _find_container(container_id: str) -> docker.models.containers.Container:
    client = get_docker()
    # Try by short ID, full ID, or name
    for c in client.containers.list(all=True, filters={"label": CONTAINER_LABEL}):
        if c.short_id == container_id or c.id == container_id or c.name == container_id:
            return c
    raise HTTPException(status_code=404, detail=f"Container {container_id} not found")


# ── Endpoints ──

@app.get("/fd/containers", response_model=list[ContainerInfo])
def list_containers():
    """List all Flight Deck managed containers."""
    client = get_docker()
    containers = client.containers.list(all=True, filters={"label": CONTAINER_LABEL})
    return [_container_info(c) for c in containers]


@app.post("/fd/spawn", response_model=ContainerActionResult)
def spawn_agent(config: AgentConfig):
    """Spawn a new Captain Claw container."""
    client = get_docker()
    slug = _slug(config.name)

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
    labels = {
        CONTAINER_LABEL: "true",
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
        return ContainerActionResult(ok=True, container_id=container.short_id, message=f"Agent '{slug}' spawned")
    except docker.errors.ImageNotFound:
        raise HTTPException(404, f"Docker image '{config.image}' not found. Pull it first.")
    except docker.errors.APIError as exc:
        raise HTTPException(500, f"Docker error: {exc.explanation or str(exc)}")


@app.post("/fd/containers/{container_id}/stop", response_model=ContainerActionResult)
def stop_container(container_id: str):
    c = _find_container(container_id)
    if c.status != "running":
        return ContainerActionResult(ok=True, container_id=c.short_id, message="Already stopped")
    c.stop(timeout=5)
    return ContainerActionResult(ok=True, container_id=c.short_id, message="Stopped")


@app.post("/fd/containers/{container_id}/start", response_model=ContainerActionResult)
def start_container(container_id: str):
    c = _find_container(container_id)
    if c.status == "running":
        return ContainerActionResult(ok=True, container_id=c.short_id, message="Already running")
    try:
        c.start()
    except docker.errors.APIError as exc:
        explanation = exc.explanation or str(exc)
        raise HTTPException(500, f"Docker start failed: {explanation}")
    return ContainerActionResult(ok=True, container_id=c.short_id, message="Started")


@app.post("/fd/containers/{container_id}/restart", response_model=ContainerActionResult)
def restart_container(container_id: str):
    c = _find_container(container_id)
    c.restart(timeout=5)
    return ContainerActionResult(ok=True, container_id=c.short_id, message="Restarted")


@app.delete("/fd/containers/{container_id}", response_model=ContainerActionResult)
def remove_container(container_id: str, force: bool = False):
    c = _find_container(container_id)
    name = c.name
    c.remove(force=force)
    return ContainerActionResult(ok=True, container_id=container_id, message=f"Removed '{name}'")


class RebuildRequest(BaseModel):
    description: str = ""  # Frontend sends current description override


@app.post("/fd/containers/{container_id}/rebuild", response_model=ContainerActionResult)
def rebuild_container(container_id: str, req: RebuildRequest | None = None):
    """Rebuild a container: stop, remove, pull latest image, re-spawn with same config."""
    import platform

    c = _find_container(container_id)
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


@app.get("/fd/containers/{container_id}/logs")
def container_logs(container_id: str, tail: int = 200, follow: bool = False):
    c = _find_container(container_id)
    if follow:
        def stream():
            for chunk in c.logs(stream=True, follow=True, tail=tail):
                yield chunk
        return StreamingResponse(stream(), media_type="text/plain")
    else:
        logs = c.logs(tail=tail).decode("utf-8", errors="replace")
        return {"logs": logs}


@app.get("/fd/containers/{container_id}")
def get_container(container_id: str):
    c = _find_container(container_id)
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
    params = f"?token={token}" if token else ""
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


@app.get("/fd/agent-files/{host}/{port}")
async def agent_files(host: str, port: int, token: str = ""):
    """List files from a CC agent (proxied to avoid CORS)."""
    import httpx
    params = f"?token={token}" if token else ""
    url = f"http://{host}:{port}/api/files{params}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, f"Agent returned {resp.status_code}")
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


class TransferRequest(BaseModel):
    src_host: str
    src_port: int
    src_auth: str = ""
    src_path: str
    dst_host: str
    dst_port: int
    dst_auth: str = ""


@app.post("/fd/transfer")
async def transfer_file(req: TransferRequest):
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
async def agent_file_download(host: str, port: int, path: str, token: str = ""):
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
async def agent_file_view(host: str, port: int, path: str, token: str = ""):
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
async def agent_file_upload(host: str, port: int, token: str = "", file: UploadFile = File(...)):
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
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/agent-usage/{host}/{port}")
async def agent_usage(host: str, port: int, token: str = "", period: str = "today"):
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


@app.get("/fd/health")
def health():
    try:
        client = get_docker()
        client.ping()
        return {"ok": True, "docker": True}
    except Exception as exc:
        return {"ok": False, "docker": False, "error": str(exc)}


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
