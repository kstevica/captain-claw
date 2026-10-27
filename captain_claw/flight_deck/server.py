"""Flight Deck backend — Docker container & process management for Captain Claw agents."""

from __future__ import annotations

import os
import sys
import json
import hashlib
import re
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
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, PrivateAttr

# Load a project-local ``.env`` into ``os.environ`` early — before any
# captain_claw submodule imports — so env-driven bridges (Messenger,
# WhatsApp) and threshold-tunable subsystems (face_index) see values
# placed in a CWD-relative ``.env`` without needing the user to source
# the file from the shell or wire systemd ``EnvironmentFile=`` entries.
#
# python-dotenv is already a core dep. ``load_dotenv()`` with no args
# walks up from the CWD looking for ``.env`` and is a no-op if not found.
# ``override=False`` (the default) preserves any value already set in
# the real environment — shell exports win over the file, which is the
# behaviour everyone expects.
from dotenv import load_dotenv
load_dotenv()

import logging

from captain_claw.config import without_retired_tools
from captain_claw.flight_deck.auth import (
    _fd_desktop_deck, decode_access_token, get_current_user, get_managed_user, get_optional_managed_user,
    get_optional_user, get_ws_user, set_auth_db,
)
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.endpoints import same_endpoint as _same_endpoint
from captain_claw.flight_deck import origin_guard
from captain_claw.flight_deck import agent_ws_quiet
from captain_claw.flight_deck import tenant_profile
from captain_claw.flight_deck import agent_sharing
from captain_claw.flight_deck import context_packs
from captain_claw.flight_deck import speaker_grants
from captain_claw.flight_deck import shared_workspace
from captain_claw.flight_deck import usage_quota
from captain_claw.flight_deck import agent_idle


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


class _SuppressShutdownCancelFilter(logging.Filter):
    """Drop uvicorn's noisy "Exception in ASGI application" tracebacks
    whose root cause is a ``CancelledError`` raised because uvicorn was
    shutting down. Each open browser tab (SSE / long-poll / streaming
    response) produces one such traceback at Ctrl+C — they are expected
    and harmless, and they crowd out the real exit messages.

    Genuine ASGI exceptions (real bugs) are kept: they have a different
    root cause and pass through untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.ERROR:
            return True
        if "Exception in ASGI application" not in record.getMessage():
            return True
        exc = record.exc_info[1] if record.exc_info else None
        # Walk the exception chain to the root cause.
        seen: set[int] = set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            if isinstance(exc, asyncio.CancelledError):
                return False  # drop the record
            exc = exc.__cause__ or exc.__context__
        return True


class _RedactQueryTokenFilter(logging.Filter):
    """Blank the values of ``fd_token=``, ``token=`` and ``t=`` query params in
    what uvicorn logs. Its access line and its WebSocket "[accepted]" line carry
    the request path WITH the query string, and those params hold a user's FD
    access JWT (owner and member agent sockets), an agent's web token or a
    download token — replayable by anyone who reads FD's log."""

    _TOKEN_PARAM = re.compile(r"([?&](?:fd_token|token|t)=)[^&#\s\"']*", re.IGNORECASE)

    @classmethod
    def redact(cls, value):
        return cls._TOKEN_PARAM.sub("\\1…", value) if isinstance(value, str) else value

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redact(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(self.redact(a) for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: self.redact(v) for k, v in record.args.items()}
        return True


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
    # Suppress the per-stream CancelledError tracebacks during shutdown.
    logging.getLogger("uvicorn.error").addFilter(_SuppressShutdownCancelFilter())
    # Never log a token from a request's query string (once per logger, even
    # when this module is loaded twice — `python -m …server` plus an import).
    for name in ("uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        if not any(type(f).__name__ == "_RedactQueryTokenFilter" for f in lg.filters):
            lg.addFilter(_RedactQueryTokenFilter())
    # A chat socket retrying an agent that is down (FD just restarted it) logs
    # once when it goes down and once when it is back — not on every attempt.
    _uv_err = logging.getLogger("uvicorn.error")
    if not any(isinstance(f, agent_ws_quiet.AgentWsLogFilter) for f in _uv_err.filters):
        _uv_err.addFilter(agent_ws_quiet.AgentWsLogFilter())


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
# Which deck spawned a container. Docker (and so every label) is host-global,
# but several decks can share a host, each with its own FD_DATA_DIR — see
# `_deck_containers`.
DECK_LABEL = "flight-deck.deck"
CC_IMAGE_DEFAULT = "kstevica/captain-claw:latest"
AUTH_ENABLED = os.environ.get("FD_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")


# ── Docker client ──

_client: docker.DockerClient | None = None


def get_docker() -> docker.DockerClient:
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def _deck_id() -> str:
    """Stable short id of THIS deck: a hash of its resolved DATA_DIR (decks on
    one host are told apart by their data dirs)."""
    return hashlib.sha256(str(DATA_DIR).encode("utf-8")).hexdigest()[:16]


def _is_this_decks_container(c) -> bool:
    """A container this deck may treat as its own: stamped with this deck's
    label, or with no deck label at all (spawned before the label existed —
    legacy, accepted as before). Another deck's container never is: its labels
    — owner, web-auth — are that deck's business, and on an auth-disabled deck
    whoever spawns it chooses them."""
    deck = (c.labels or {}).get(DECK_LABEL, "")
    return not deck or deck == _deck_id()


def _deck_containers(*, all: bool = False, client=None) -> list:
    """This deck's managed containers (see `_is_this_decks_container`). Every
    label-based ownership / identity / authorization lookup goes through this
    rather than listing CONTAINER_LABEL host-wide. Raises when Docker is
    unavailable, like the list call it wraps."""
    client = client or get_docker()
    return [c for c in client.containers.list(all=all, filters={"label": CONTAINER_LABEL})
            if _is_this_decks_container(c)]


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


def _load_process_registry_strict() -> dict[str, dict]:
    """Like :func:`_load_process_registry`, but raises (OSError / ValueError)
    when the file exists and can't be read or parsed, so a caller that acts
    destructively on "no such agent" can tell a missing agent from a failed read."""
    if not PROCESS_REGISTRY_FILE.is_file():
        return {}
    import fcntl as _fcntl
    with PROCESS_REGISTRY_FILE.open("r") as _f:
        try:
            _fcntl.flock(_f.fileno(), _fcntl.LOCK_SH)
        except OSError:
            pass
        data = _f.read()
    return json.loads(data) if data else {}


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


def _resolve_cc_web_bin() -> str:
    """Resolve the ``captain-claw-web`` executable.

    In a PyInstaller standalone build the binary lives next to this process's
    executable (``sys._MEIPASS``'s parent) and is NOT on PATH, so a bare
    ``"captain-claw-web"`` lookup fails with FileNotFoundError. Resolve the
    bundled path when frozen; fall back to PATH for pip/dev installs.
    """
    if getattr(sys, "_MEIPASS", None):
        bundled = Path(sys._MEIPASS).parent / "captain-claw-web"
        if bundled.exists():
            return str(bundled)
    return "captain-claw-web"


def _fd_self_url() -> str:
    """Base URL THIS deck's process agents use to call back into Flight Deck.

    ``FD_INTERNAL_URL`` is the explicit operator override (the same variable
    app subprocesses use); otherwise loopback on the port ``main()`` records
    as ``FD_PORT`` — the port this deck actually bound.
    """
    explicit = os.environ.get("FD_INTERNAL_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    return f"http://localhost:{os.environ.get('FD_PORT', '25080')}"


def _pin_fd_url(environment: dict[str, str]) -> None:
    """Point a process agent's FD callbacks at THIS deck, overriding anything
    inherited. Several decks can share a host (each with its own FD_DATA_DIR);
    an FD_URL leaking in from the shell, a CWD ``.env``, the agent's own
    ``.env`` or a spawn's env_vars would send this deck's agents to ANOTHER
    deck, which then answers their Google / VFS / memory calls for a tenant
    that isn't theirs. The agent-side ``google_oauth.flight_deck_url`` config
    key outranks FD_URL, so an inherited env override of it is dropped too.
    """
    environment["FD_URL"] = _fd_self_url()
    environment.pop("CLAW_GOOGLE_OAUTH__FLIGHT_DECK_URL", None)


# FD's own secrets, never handed to a process agent. Agents inherit FD's
# environment and every one of them can run shell, so anything left here is
# readable by every tenant on the deck:
# * FD_JWT_SECRET — mints a session JWT for ANY user of this deck (agents
#   never verify FD JWTs; only FD and the Lupa BFF, which FD doesn't spawn).
# * FD_EVENTS_WEBHOOK_TOKEN — posts events into any user's event spine.
# * GOOGLE_WORKSPACE_CLI_TOKEN / _CREDENTIALS_FILE — an operator's ambient gws
#   identity. The gws tool is retired (agents use google_drive /
#   google_calendar / google_mail with the OWNER's token), but a stray gws
#   binary on PATH is still one shell call away, and with these it would act
#   as that one Google account for every tenant — and send Gmail past the
#   FD send gate.
# FD_AGENT_SHARED_SECRET stays: agents send it as X-Agent-Secret. So does
# GOOGLE_APPLICATION_CREDENTIALS (operators point non-Workspace Google client
# libraries at it).
_FD_ONLY_ENV_VARS = (
    "FD_JWT_SECRET",
    "FD_EVENTS_WEBHOOK_TOKEN",
    "GOOGLE_WORKSPACE_CLI_TOKEN",
    "GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE",
    # The deck's Soniox key is a fallback FD itself uses (soniox_routes); an
    # agent transcribes through FD with its owner's key instead of holding one.
    "SONIOX_API_KEY",
)


def _agent_base_env() -> dict[str, str]:
    """FD's environment as a process agent should inherit it: minus the FD-only
    secrets above. The agent's own .env / env_vars are layered on top by the
    caller, so a user can still hand THEIR agent a credential deliberately.

    Per-deck path vars are made absolute: the agent runs with cwd = its own dir,
    so a relative FD_DATA_DIR would resolve somewhere else there — a different
    agent_secret file (X-Agent-Secret then never matches, which FD_LOCKDOWN
    turns into a refused call) and a different VFS root.
    """
    env = {k: v for k, v in os.environ.items() if k not in _FD_ONLY_ENV_VARS}
    if env.get("FD_DATA_DIR", "").strip():
        env["FD_DATA_DIR"] = str(DATA_DIR)
    home = env.get("CAPTAIN_CLAW_FD_HOME", "").strip()
    if home:
        env["CAPTAIN_CLAW_FD_HOME"] = str(Path(home).expanduser().resolve())
    return env


# Keys Flight Deck shares with an agent that didn't set its own: the WhatsApp
# Cloud API creds for the agent-side send path (files and pictures back to a
# WhatsApp chat). Soniox is not one: agents ask FD (/fd/soniox/agent/*).
_SHARED_FD_SECRETS = (
    "WHATSAPP_ACCESS_TOKEN", "WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_ALLOWED_WAIDS",
)


def _share_fd_secrets(environment: dict[str, str]) -> None:
    """Fill the shared keys from Flight Deck's own env when the agent's env
    leaves them unset or blank — one place holds them, not every .env."""
    for key in _SHARED_FD_SECRETS:
        if not str(environment.get(key, "")).strip():
            value = str(os.environ.get(key, "")).strip()
            if value:
                environment[key] = value


# Tenant confinement (captain_claw/deck_confine.py is the agent side). Every
# process agent runs as this deck's OS user, so on a sign-in deck FD pins, after
# the agent's .env and env_vars, which files its tools may name and whether it
# may run code at all. FD_AGENT_CONFINE=off turns it off for a deck whose users
# all trust each other.
_CONFINE_ENV_KEYS = ("CLAW_CONFINE", "CLAW_CONFINE_HOME", "CLAW_CONFINE_WORKSPACE")
# What a non-admin's .env / env_vars may not change from FD's own: each makes
# the agent's Python, git, node, Java, OpenSSL, Playwright or shell load code
# or config from a path (and the agent writes its own workspace) or run a
# command it names, moves the roots its tools are held to, routes its calls
# through another host (a proxy, a resolver alias) or has it trust another
# certificate authority, or turns its audit off.
# (CLAW_VFS_SCOPE stays: it only ever narrows the agent's VFS — Iskra's wall.)
_PINNED_ENV_PREFIXES = (
    "PYTHON", "LD_", "DYLD_", "GIT_", "NODE_", "NPM_CONFIG_", "PERL", "RUBY",
    "JAVA_", "_JAVA_", "JDK_JAVA_", "OPENSSL_", "PLAYWRIGHT_", "XDG_", "PIP_", "UV_",
    "SSL_CERT_", "CLAW_PTY_", "CLAW_TOOL_AUDIT",
)
_PINNED_ENV_KEYS = frozenset({
    "PATH", "BASH_ENV", "ENV", "PROMPT_COMMAND", "SHELL", "ZDOTDIR",
    "BROWSER", "EDITOR", "VISUAL", "PAGER", "LESSOPEN", "SSH_ASKPASS",
    "CURL_HOME", "WGETRC", "NETRC", "GCONV_PATH", "LOCPATH",
    "CLAW_VFS_ROOT", "FD_DATA_DIR", "CAPTAIN_CLAW_FD_HOME", "FD_DATA_KEY_FILE",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "HOSTALIASES", "RES_OPTIONS", "LOCALDOMAIN",
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSLKEYLOGFILE",
})


def _pinned_env_name(name: str) -> bool:
    """Whether ``name`` is one a non-admin's agent takes from Flight Deck only."""
    upper = name.upper()
    return upper in _PINNED_ENV_KEYS or upper.startswith(_PINNED_ENV_PREFIXES)


def _confine_agents() -> bool:
    """Whether this deck confines its process agents: sign-in on (it has users
    to keep apart) and not turned off with FD_AGENT_CONFINE=off."""
    flag = os.environ.get("FD_AGENT_CONFINE", "").strip().lower()
    return AUTH_ENABLED and flag not in ("off", "0", "false", "no")


def _owner_is_admin(owner_id: str) -> bool:
    """The owner's role, read now (a role change applies at the agent's next
    start). Sync so the restart path can use it; False when unknown."""
    if not owner_id:
        return False
    import sqlite3

    try:
        con = sqlite3.connect(f"file:{DATA_DIR / 'flight-deck.db'}?mode=ro", uri=True, timeout=5)
        try:
            row = con.execute("SELECT role FROM users WHERE id = ?", (owner_id,)).fetchone()
        finally:
            con.close()
    except sqlite3.Error as exc:
        log.warning("Could not read an agent owner's role", error=type(exc).__name__)
        return False
    return bool(row) and row[0] == "admin"


def _pin_confinement(environment: dict[str, str], base: dict[str, str], *,
                     agent_dir: Path, owner_id: str, workspace: str = "") -> None:
    """Pin the agent's confinement (deck_confine) LAST, over whatever its .env
    and env_vars said. A non-admin owner's env also gets FD's own values back
    for :data:`_PINNED_ENV_KEYS`, and a workspace FD didn't place in the
    agent's folder or the owner's VFS root is not one of its roots."""
    for key in _CONFINE_ENV_KEYS:
        environment.pop(key, None)
    if not _confine_agents():
        return
    tier = "admin" if _owner_is_admin(owner_id) else "user"
    home = agent_dir.resolve()
    default_ws = home / "data" / "workspace"
    ws = Path(workspace).expanduser().resolve() if workspace else default_ws
    if tier == "user":
        for key in list(environment):
            if _pinned_env_name(key):
                if key in base:
                    environment[key] = base[key]
                else:
                    environment.pop(key, None)
        from captain_claw.vfs import path_within

        owner_vfs = (DATA_DIR / "vfs" / owner_id).resolve() if owner_id else None
        if not (path_within(ws, home) or (owner_vfs is not None and path_within(ws, owner_vfs))):
            log.warning("Agent workspace outside its own roots — not one of them",
                        agent=home.name)
            ws = default_ws
    environment["CLAW_CONFINE"] = tier
    environment["CLAW_CONFINE_HOME"] = str(home)
    environment["CLAW_CONFINE_WORKSPACE"] = str(ws)


def _start_registered_process(slug: str, entry: dict) -> bool:
    """Start a single process agent from its registry entry. Returns True on success."""
    agent_dir = DATA_DIR / slug
    if not agent_dir.is_dir():
        return False

    web_port = entry.get("web_port", 24080)

    # Rebuild environment from .env file (on FD's env minus its own secrets)
    base_env = _agent_base_env()
    environment = dict(base_env)
    env_file = agent_dir / ".env"
    if env_file.is_file():
        content = env_file.read_text().strip()
        if content:
            for line in content.split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    environment[k] = v

    _share_fd_secrets(environment)

    environment["HOME"] = str(agent_dir / "data" / "home-config-parent")
    # Slug + URL for port-fallback callbacks. Without FD_URL the agent can't
    # announce a drifted port back to Flight Deck and the registry goes stale
    # (chat panel then 401s because FD proxies to the old port). Always THIS
    # deck — a stale FD_URL frozen into the agent's .env must not win.
    environment["FD_AGENT_SLUG"] = slug
    _pin_fd_url(environment)
    # Re-pin the recorded owner exactly as spawn does. A restart otherwise
    # drops FD_OWNER_ID (its own spawns lose their owner) and inherits FD's own
    # CLAW_VFS_USER (the primary owner, written by lifespan), so every
    # restarted agent would silently work in the primary owner's VFS instead
    # of its own tenant's.
    _owner = str(entry.get("owner") or "")
    if _owner:
        environment["FD_OWNER_ID"] = _owner
        environment["CLAW_VFS_USER"] = _owner
    _pin_confinement(environment, base_env, agent_dir=agent_dir, owner_id=_owner,
                     workspace=str(entry.get("workspace") or ""))

    log_file = agent_dir / "process.log"
    try:
        log_fh = open(log_file, "a")
        proc = subprocess.Popen(
            [_resolve_cc_web_bin(), "--port", str(web_port)],
            cwd=str(agent_dir),
            env=environment,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _processes[slug] = proc
        entry["pid"] = proc.pid
        return True
    except Exception as exc:
        log.error("Failed to start process agent", slug=slug, error=str(exc))
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
        # Skip agents that were intentionally stopped by the user — and the
        # ones the idle controller put to sleep (agent_idle): they wake when
        # something needs them.
        if entry.get("stopped") or entry.get("asleep"):
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
    # Persist our pid updates WITHOUT clobbering a port a child announced while
    # we were spawning: a freshly-restarted agent whose port was taken drifts to
    # a fallback and POSTs /announce-port, which rewrites the on-disk registry
    # independently. Re-read and carry the on-disk web_port/web_auth forward (the
    # announce handler is the authority for those) — only pid is ours to set.
    fresh = _load_process_registry()
    for slug, entry in registry.items():
        f = fresh.get(slug)
        if f is None:
            fresh[slug] = entry
        else:
            f["pid"] = entry.get("pid")
    _save_process_registry(fresh)
    if restarted:
        print(f"Flight Deck: restarted {len(restarted)} process agent(s): {', '.join(restarted)}")
    if skipped:
        print(f"Flight Deck: skipped {len(skipped)} stopped process agent(s): {', '.join(skipped)}")


# ── App ──

def _upsert_dotenv_var(env_path: Path, key: str, value: str) -> bool:
    """Idempotently set ``KEY=value`` in a .env file, preserving other lines.

    Returns True when the file was changed. Creates the file if absent.
    """
    try:
        lines = env_path.read_text().splitlines() if env_path.is_file() else []
    except OSError:
        return False
    new_line = f"{key}={value}"
    prefix = f"{key}="
    out: list[str] = []
    found = changed = False
    for ln in lines:
        if ln.strip().startswith(prefix):
            found = True
            if ln != new_line:
                changed = True
                out.append(new_line)
            else:
                out.append(ln)
        else:
            out.append(ln)
    if not found:
        out.append(new_line)
        changed = True
    if changed:
        try:
            env_path.write_text("\n".join(out) + "\n")
        except OSError:
            return False
    return changed


async def _resolve_primary_owner(db) -> str:
    """The owning user id to bind the standalone main agent's VFS to.

    Single-user deployments → that user. Multi-user → the oldest admin (the
    oldest user when there is no admin at all). Google's legacy-token and
    fallback identity hang off this too, so it must be the genuinely oldest
    admin however many users exist — not the oldest of whichever page of
    users happened to be fetched.
    """
    from captain_claw.flight_deck.auth import _LOCAL_USER

    try:
        total = await db.count_users()
        if not total:
            return ""
        # list_users is ordered created_at DESC, so the oldest users sit at the
        # END. Walk pages from the oldest end backwards: the first page holding
        # an admin holds the oldest admin (its last admin, pages being DESC).
        page = 200
        offset = max(0, total - page)
        oldest_user = ""
        while True:
            # Never the row the no-sign-in local user stands for (it owns
            # nothing once sign-in is on; callers fall back to it without).
            users = [u for u in await db.list_users(limit=page, offset=offset)
                     if u.get("id") != _LOCAL_USER["id"]]
            if users and not oldest_user:
                oldest_user = str(users[-1].get("id", ""))
            admins = [u for u in users if u.get("role") == "admin"]
            if admins:
                return str(admins[-1].get("id", ""))
            if offset == 0:
                return oldest_user
            offset = max(0, offset - page)
    except Exception:
        return ""


async def _init_fd_db() -> FlightDeckDB:
    """Open this deck's settings/auth DB and make it the auth module's DB.

    Always, not only with auth on: the connector routers (Codex / MCP /
    Typesense) deliberately run under the synthetic local user when auth is
    disabled (desktop build) and keep their client + tokens in this DB —
    without it every call trips get_db()'s "not initialized" assert and
    Connections is dead on that deck type. (Google via FD still refuses on
    such decks — see google_oauth_routes._require_auth_deck.) Safe on an auth-disabled deck only
    because origin_guard keeps other websites and DNS-rebinding pages off the
    API, and /fd/auth/register refuses there (no web page can plant an admin
    that would become real if auth is switched on later).
    """
    db = FlightDeckDB(DATA_DIR / "flight-deck.db")
    await db.init()
    set_auth_db(db)
    # Without sign-in every request runs as the synthetic local user; give it
    # the users row the per-user tables reference, or saving its settings
    # (tier sets the archetype spawn reads), archetypes and shares fails.
    from captain_claw.flight_deck import auth as _auth

    if not _auth._fd_auth_enabled():
        import secrets

        await db.ensure_local_user(
            _auth._LOCAL_USER["id"], _auth.hash_password(secrets.token_urlsafe(32)),
        )
    # Prime the Typesense connection (Connections → Typesense) before any
    # request lands — an agent's proxied search may well be the first one.
    try:
        from captain_claw.flight_deck.deep_memory_routes import load_connection

        await load_connection()
    except Exception as _exc:
        log.debug("Deep memory connection not primed", error=str(_exc))
    return db


async def _start_botport_link(db: Any) -> None:
    """Wire the BotPort link to this deck's fleet and agent proxy, then start it
    with the stored (and FD_BOTPORT_* environment) settings."""
    from captain_claw import __version__ as _cc_version
    from captain_claw.flight_deck import botport_link as _bl

    link = _bl.get_link()
    deck_id = await db.get_system_setting(_bl.DECK_ID_KEY)
    if not deck_id:
        deck_id = secrets.token_hex(16)
        await db.set_system_setting(_bl.DECK_ID_KEY, deck_id)
    stored = _bl.unseal_settings(await db.get_system_setting(_bl.SETTING_KEY))
    link.user_shares = _bl.parse_shares(await db.get_system_setting(_bl.SHARES_KEY))
    link.configure(
        fleet=_fleet_snapshot,
        consult_events=_consult_peer_events,
        resolve_auth=_resolve_agent_auth,
        version=_cc_version,
    )
    await link.start(stored, deck_id)
    s = link.settings
    if s.enabled:
        print(f"Flight Deck: BotPort link enabled — deck '{s.deck_name}' → {s.url}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Own API keys move agents through THIS module's process table (it runs
    # as __main__; an import of `server` elsewhere is a second copy).
    from captain_claw.flight_deck import byok as _byok

    _byok.agent_switcher = byok_switch_agents
    _byok.owner_agents = byok_owner_agents
    # Initialize game registry with FD data dir
    from captain_claw.games.registry import set_games_dir
    games_dir = DATA_DIR / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    set_games_dir(games_dir)
    # Set up image generation provider (default from IMAGE_PROVIDERS order)
    from captain_claw.games.image_service import switch_provider, list_providers, get_provider_id
    providers = list_providers()
    default_id = get_provider_id()
    default_prov = next((p for p in providers if p["id"] == default_id and p["available"]), None)
    if not default_prov:
        default_prov = next((p for p in providers if p["available"]), None)
    if default_prov:
        switch_provider(default_prov["id"])
        print(f"Flight Deck: image provider '{default_prov['id']}' enabled")
    else:
        print("Flight Deck: no image provider available")
    # Defer reattach so uvicorn has time to actually bind its listening
    # socket. Otherwise the children we launch here race to call
    # /fd/processes/{slug}/announce-port before FD is accepting connections,
    # and their drift announcements get "connection refused".
    async def _deferred_reattach():
        await asyncio.sleep(1.0)
        await _startup_team_keys_then_reattach()
    asyncio.create_task(_deferred_reattach())
    print(f"Flight Deck: {origin_guard.describe_policy()}")
    # Initialize database for auth & settings — always, see _init_fd_db.
    _fd_db = await _init_fd_db()
    # MCP servers saved before they had owners: settle that before serving,
    # so an OAuth sign-in never answers another user's agent in between.
    from captain_claw.flight_deck.mcp_routes import migrate_legacy_ownership

    await migrate_legacy_ownership(_fd_db)
    # Bring every agent's owner-profile files in line with the DB (an auth-mode
    # switch, edits made while FD was down) — in the background, best-effort.
    app.state.tenant_profile_reconcile = asyncio.create_task(tenant_profile.reconcile(_fd_db))
    # Shared-agent context packs: the same for the shared-context files (with
    # sharing off it removes leftovers), then — sharing on — a reconcile loop
    # for what FD only learns by polling (direct DB edits, removed agents).
    app.state.context_packs_reconcile = asyncio.create_task(context_packs.reconcile(_fd_db))
    if agent_sharing.sharing_active():
        app.state.context_packs_stop = asyncio.Event()
        app.state.context_packs_task = asyncio.create_task(
            context_packs.reconcile_loop(_fd_db, app.state.context_packs_stop))
        # PR C: once per deck, tell owners of shared process agents that
        # members can now open those agents' saved/ folders and datastores.
        app.state.shared_workspace_notice = asyncio.create_task(
            shared_workspace.notify_owners_of_commons_once(_fd_db))
        # PR D: once per deck, tell members of shared agents that the owner's
        # agent can now look into their use of it.
        from captain_claw.flight_deck import shared_usage as _shared_usage

        app.state.shared_usage_notice = asyncio.create_task(
            _shared_usage.notify_members_once(_fd_db))
    if AUTH_ENABLED:
        app.state.fd_db = _fd_db
        # Persist the owning user id into the project-local .env so the
        # standalone main agent resolves the SAME VFS user as the dashboard
        # (otherwise it falls back to "local" and its vfs:<project>/ files
        # never show up in the panel, which reads under the logged-in UUID).
        try:
            _owner = await _resolve_primary_owner(_fd_db)
            if _owner:
                if _upsert_dotenv_var(Path(".env"), "CLAW_VFS_USER", _owner):
                    print(f"Flight Deck: bound VFS to owner {_owner} (wrote CLAW_VFS_USER to .env)")
                os.environ.setdefault("CLAW_VFS_USER", _owner)
        except Exception as _vfs_exc:
            print(f"Flight Deck: could not bind VFS owner: {_vfs_exc}")
        # Load admin-configured plan limits from DB
        plan_limits_raw = await _fd_db.get_system_setting("fd:plan-limits")
        load_plan_limits_from_db_sync(plan_limits_raw)
        # Initialize vast.ai GPU cloud integration.
        from captain_claw.vastai.manager import VastAIManager
        from captain_claw.flight_deck.vastai_routes import set_vastai_manager
        from captain_claw.vastai.wake import register_manager as _register_vastai_wake
        _vastai_mgr = VastAIManager(_fd_db)
        await _vastai_mgr.initialize()
        set_vastai_manager(_vastai_mgr)
        _register_vastai_wake(_vastai_mgr)
        app.state.vastai_manager = _vastai_mgr
        # The deck's shared Telegram bot (Admin → Integrations): FD polls it and
        # relays each user's chat to their agent. Inert until a token is set.
        try:
            from captain_claw.flight_deck import telegram_shared as _tg_shared

            await _tg_shared.start(_fd_db)
        except Exception as _tg_exc:
            print(f"Flight Deck: shared Telegram bot not started: {_tg_exc}")
    # WhatsApp Web (Connections → WhatsApp Web): the service holding people's
    # linked accounts. Adopted when one survived the last restart, else started
    # in the background when links exist and the admin allows it.
    try:
        from captain_claw.flight_deck import whatsapp_web_supervisor as _waw

        await _waw.init(DATA_DIR, _fd_db)
    except Exception as _waw_exc:
        print(f"Flight Deck: WhatsApp Web not set up: {_waw_exc}")
    # ── Code-app subprocess runtime ──
    # Lazy-spawn per app on first request; reaper kicks in here.
    from captain_claw.flight_deck.app_runtime import get_runtime as _get_app_runtime
    _app_rt = _get_app_runtime()
    await _app_rt.start()
    app.state.app_runtime = _app_rt
    # ── Outbound agent links: Flight Deck connects out to the systems that
    # won't take connections from agents (agent_link_out.py).
    try:
        from captain_claw.flight_deck import agent_link_out as _link_out

        _n_out = await _link_out.start_all()
        if _n_out:
            print(f"Flight Deck: connecting out for {_n_out} agent link(s)")
    except Exception as _out_exc:
        print(f"Flight Deck: outbound agent links not started: {_out_exc}")
    # ── FD scheduler (proactive push: agent prompts on a timer → WhatsApp /
    # channel). Disable with FD_SCHEDULER_DISABLED=true. ──
    if os.environ.get("FD_SCHEDULER_DISABLED", "").lower() not in ("true", "1", "yes"):
        from captain_claw.flight_deck.fd_scheduler import scheduler_loop as _sched_loop
        _sched_stop = asyncio.Event()
        app.state.scheduler_stop = _sched_stop
        app.state.scheduler_task = asyncio.create_task(_sched_loop(_sched_stop))
        print("Flight Deck: scheduler started")
    # ── Agent controller: idle process agents go to sleep, and wake 5 min
    # before a scheduled run or when something needs them (agent_idle.py).
    # Inert until an admin sets a time (Settings → Agents). ──
    _idle_stop = asyncio.Event()
    app.state.agent_idle_stop = _idle_stop
    app.state.agent_idle_task = asyncio.create_task(agent_idle.controller_loop(_idle_stop))
    # ── Tool-call register: rows past the retention are pruned (6-hourly). ──
    _audit_stop = asyncio.Event()
    app.state.tool_audit_stop = _audit_stop
    app.state.tool_audit_task = asyncio.create_task(tool_audit_routes.retention_loop(_audit_stop))
    # ── Payments loop (FD_PAYMENTS): expiry, bank statement syncs (Phase 4). ──
    if payments_routes.enabled():
        from captain_claw.flight_deck.payments_jobs import payments_loop as _pay_loop
        _pay_stop = asyncio.Event()
        app.state.payments_stop = _pay_stop
        app.state.payments_task = asyncio.create_task(_pay_loop(_pay_stop))
    # ── Consciousness heartbeat (free-running, per-user inner life that quietly
    # observes each user's agents). Disable with FD_CONSCIOUSNESS_DISABLED=true. ──
    if os.environ.get("FD_CONSCIOUSNESS_DISABLED", "").lower() not in ("true", "1", "yes"):
        from captain_claw.flight_deck.consciousness import heartbeat_loop as _hb_loop
        _hb_stop = asyncio.Event()
        app.state.consciousness_stop = _hb_stop
        app.state.consciousness_task = asyncio.create_task(_hb_loop(_hb_stop))
        print("Flight Deck: consciousness heartbeat started")
    # ── Event spine poll loop (#2): source adapters → external_events → arbiter.
    # Disable with FD_EVENTS_DISABLED=true. Adapters self-gate, so it's cheap. ──
    if os.environ.get("FD_EVENTS_DISABLED", "").lower() not in ("true", "1", "yes"):
        from captain_claw.flight_deck.event_sources import events_loop as _ev_loop
        _ev_stop = asyncio.Event()
        app.state.events_stop = _ev_stop
        app.state.events_task = asyncio.create_task(_ev_loop(_ev_stop))
        print("Flight Deck: event spine poll loop started")
    # ── Gmail topic auto-replies: FD's own History-API poller → ledger →
    # classifier / writer / guard → send (gmail_autoreply.py). Inert until a
    # user opts in; FD_GMAIL_AUTOREPLY=off keeps it from starting at all. ──
    from captain_claw.flight_deck import gmail_autoreply as _gmail_autoreply
    if not _gmail_autoreply.autoreply_deck_disabled():
        _ar_stop = asyncio.Event()
        app.state.autoreply_stop = _ar_stop
        app.state.autoreply_task = asyncio.create_task(
            _gmail_autoreply.autoreply_loop(_ar_stop, _fd_db))
        print("Flight Deck: Gmail topic auto-reply loop started")
    # ── Iskra beings heartbeat (living beings tick/dream on their own clock).
    # Disable with FD_BEINGS_DISABLED=true. Cheap when no being is due. ──
    if os.environ.get("FD_BEINGS_DISABLED", "").lower() not in ("true", "1", "yes"):
        from captain_claw.flight_deck.beings_loop import beings_loop as _beings_loop
        _beings_stop = asyncio.Event()
        app.state.beings_stop = _beings_stop
        app.state.beings_task = asyncio.create_task(
            _beings_loop(getattr(app.state, "fd_db", None), _beings_stop))
        print("Flight Deck: beings heartbeat started")
    # ── Bat supervisor (durable stubborn-finisher runs: adopt a leased run,
    # drive it to done, re-adopt a crashed run after restart). Disable with
    # FD_BAT_DISABLED=true. Idle until a Bat run exists and the runner is wired. ──
    if os.environ.get("FD_BAT_DISABLED", "").lower() not in ("true", "1", "yes"):
        from captain_claw.flight_deck import bat_loop as _bat_loop_mod
        from captain_claw.flight_deck.bat_store import BatStore as _BatStore
        _bat_store = _BatStore(DATA_DIR / "bat.db")
        await _bat_store.init()
        app.state.bat_store = _bat_store
        _bat_loop_mod.set_store(_bat_store)
        _bat_stop = asyncio.Event()
        app.state.bat_stop = _bat_stop
        app.state.bat_task = asyncio.create_task(_bat_loop_mod.bat_loop(_bat_store, _bat_stop))
        print("Flight Deck: Bat supervisor started")
    # ── Flow engine (process automations: trigger → steps on the agent pool) ──
    try:
        from captain_claw.flight_deck.flows_store import FlowStore
        from captain_claw.flight_deck.flow_runner import FlowRunner
        from captain_claw.flight_deck import flow_router
        _flow_store = FlowStore(DATA_DIR / "flows.db")

        # ── `agent on archetype:<id>` seams ──
        # A flow step can run on a freshly spawned ephemeral archetype agent. The
        # runner owns the lifecycle (spawn-once-per-run, dispose at end); these
        # closures give it the registry + the proven background-spawn path used by
        # Basna/Dubina (a stub Request whose .state.user_id stamps the owner).
        def _flow_owner_id(payload: dict) -> str:
            """Resolve the user who owns this flow run. Flows carry no owner column
            yet, so: explicit payload.user_id → the origin agent's registry owner
            (the agent that received the triggering message) → FD_OWNER_ID env. This
            is what lets us load the RIGHT user's Library tier keys for the spawn."""
            uid = str(payload.get("user_id") or "")
            if uid:
                return uid
            port = int(payload.get("origin_port") or 0)
            name = str(payload.get("origin_name") or "")
            if port or name:
                for _slug, e in _load_process_registry().items():
                    if (port and int(e.get("web_port") or 0) == port) or (name and e.get("name") == name):
                        owner = str(e.get("owner") or "")
                        if owner:
                            return owner
            return os.environ.get("FD_OWNER_ID", "")

        async def _flow_load_archetype(payload: dict, aid: str):
            from captain_claw.flight_deck.archetypes import merged_archetypes
            from captain_claw.flight_deck.archetype_compose import resolve_pair
            from captain_claw.flight_deck.auth import get_db
            # Flag-gated function×domain grid: `on archetype:reviewer.legal` composes
            # a leaf whose `fleet_instructions` the spawned dubina agent then uses.
            composed = resolve_pair(aid)
            if composed is not None:
                return composed
            uid = _flow_owner_id(payload) or None
            for a in await merged_archetypes(get_db(), uid):
                if a.get("id") == aid:
                    return a
            return None

        async def _flow_owner(payload: dict):
            """(user_dict | None, stub_request) for a background archetype spawn.
            `spawn_process` reads request.state.user_id for ownership and `user`
            for plan/quota; with auth off both degrade to the local/None defaults."""
            import types as _types
            from captain_claw.flight_deck.auth import get_db
            uid = _flow_owner_id(payload)
            user = None
            if AUTH_ENABLED and uid:
                try:
                    user = await get_db().get_user_by_id(uid)
                except Exception:
                    user = None
            stub = _types.SimpleNamespace(state=_types.SimpleNamespace(user_id=uid))
            return user, stub

        async def _flow_spawn_archetype(arch: dict, tier: str, tcfg: dict, payload: dict):
            from captain_claw.flight_deck import dubina_agents
            from captain_claw.flight_deck.basna_routes import _load_owner_tiers
            from captain_claw.flight_deck.auth import get_db
            user, stub = await _flow_owner(payload)
            # No explicit `@tier` → fall back to the archetype's own default tier.
            eff_tier = tier or str(arch.get("tier") or "")
            # Resolve the OWNER's Library config: the tier's provider/model/api_key/
            # base_url AND the env vars (BRAVE_API_KEY, TAVILY_API_KEY, …). The tier
            # is only a NAME until resolved against the user — without the LLM key
            # the agent can't call the model, and without the env vars its tools
            # (web_search/browser/…) have no credentials. Same source Basna uses.
            resolved = dict(tcfg or {})
            owner_env: list = []
            try:
                tiers_map, owner_env = await _load_owner_tiers(get_db(), _flow_owner_id(payload))
                if not resolved.get("api_key") and eff_tier:
                    resolved = (tiers_map or {}).get(eff_tier) or resolved
            except Exception as exc:
                log.warning("flow archetype owner-config resolve failed", tier=eff_tier, error=str(exc))
            # Env precedence: the triggering agent's own .env (base) overlaid by the
            # owner's Library env (authoritative where set). So a spawned specialist
            # inherits the main agent's tool keys by default, while explicit Library
            # config still wins.
            merged_env: dict = {}
            for ev in _flow_origin_env(payload):
                merged_env[ev["key"]] = ev.get("value", "")
            for ev in (owner_env or []):
                if ev.get("key"):
                    merged_env[ev["key"]] = ev.get("value", "")
            # Don't let an inherited LLM-provider key override the resolved tier key
            # (_build_env writes env_vars AFTER provider_api_key, so a same-named env
            # var would otherwise clobber it). Only strip when we actually resolved one.
            from captain_claw.flight_deck.basna_routes import _effective_key
            _tier_key = str(resolved.get("api_key") or "").strip()
            if _tier_key == "@system":  # no org key behind it → the inherited one stays
                _tier_key = _effective_key(
                    str(resolved.get("provider") or ""), _tier_key, resolved.get("base_url")) or ""
            if _tier_key:
                for _prov_env in _provider_key_env_names(str(resolved.get("provider") or "")):
                    merged_env.pop(_prov_env, None)
            env_list = [{"key": k, "value": v} for k, v in merged_env.items()]
            return await dubina_agents.spawn_archetype_agent(
                arch, eff_tier, resolved, stub, user, env_vars=env_list)

        async def _flow_stop_archetype(slug: str):
            from captain_claw.flight_deck import dubina_agents
            await dubina_agents.stop_archetype_agent(slug)

        _flow_runner = FlowRunner(
            _flow_store,
            get_agents=_running_agents,
            resolve_auth=_resolve_agent_auth,
            fd_self_base=_fd_self_url(),
            fd_tools=_fd_internal_tools(),
            whatsapp_send=_flow_whatsapp_send,
            transfer_file=_transfer_file_to_agent,
            consult_peer=_consult_peer_events,
            load_archetype=_flow_load_archetype,
            spawn_archetype=_flow_spawn_archetype,
            stop_archetype=_flow_stop_archetype,
            # Tenancy: a run reaches only its owner's agents (admins: any).
            resolve_owner=_resolve_agent_owner,
            user_is_admin=_flow_user_is_admin,
            enforce_owner=AUTH_ENABLED,
        )
        app.state.flow_store = _flow_store
        app.state.flow_runner = _flow_runner
        flow_router.set_engine(_flow_store, _flow_runner)
        print("Flight Deck: flow engine ready")
    except Exception as _exc:
        print(f"Flight Deck: flow engine init failed: {_exc}")

    # ── Dubina (Frontier Horizon: simulate a frontier model on a cheaper tier) ──
    try:
        from captain_claw.flight_deck import dubina_routes
        from captain_claw.flight_deck.dubina_store import DubinaStore
        _dubina_store = DubinaStore(DATA_DIR / "dubina.db")
        await _dubina_store.init()
        app.state.dubina_store = _dubina_store
        dubina_routes.set_store(_dubina_store)
        print("Flight Deck: dubina engine ready")
    except Exception as _exc:
        print(f"Flight Deck: dubina engine init failed: {_exc}")

    # ── BotPort link: this deck <-> other Flight Decks (off until configured) ──
    try:
        await _start_botport_link(_fd_db)
    except Exception as _exc:
        print(f"Flight Deck: BotPort link not started: {_exc}")
    yield
    try:
        from captain_claw.flight_deck import agent_link_out as _link_out

        await _link_out.stop_all()
    except Exception:
        pass
    try:
        from captain_claw.flight_deck.botport_link import get_link as _get_botport_link

        await _get_botport_link().stop()
    except Exception:
        pass
    try:
        from captain_claw.flight_deck import telegram_shared as _tg_shared

        await _tg_shared.stop()
    except Exception:
        pass
    try:
        # Stop watching only: the service and its links outlive this process.
        from captain_claw.flight_deck import whatsapp_web_supervisor as _waw

        await _waw.shutdown()
    except Exception:
        pass
    # Stop the scheduler loop first so it doesn't fire mid-shutdown.
    if hasattr(app.state, "scheduler_stop"):
        app.state.scheduler_stop.set()
        if hasattr(app.state, "scheduler_task"):
            try:
                await asyncio.wait_for(app.state.scheduler_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
    if hasattr(app.state, "agent_idle_stop"):
        app.state.agent_idle_stop.set()
        try:
            await asyncio.wait_for(app.state.agent_idle_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
    if hasattr(app.state, "payments_stop"):
        app.state.payments_stop.set()
        try:
            await asyncio.wait_for(app.state.payments_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
    # Stop the consciousness heartbeat.
    if hasattr(app.state, "consciousness_stop"):
        app.state.consciousness_stop.set()
        if hasattr(app.state, "consciousness_task"):
            try:
                await asyncio.wait_for(app.state.consciousness_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
    # Stop the Gmail topic auto-reply loop. A send it interrupts is never
    # resent: the next boot records it as "may have auto-replied".
    if hasattr(app.state, "autoreply_stop"):
        app.state.autoreply_stop.set()
        if hasattr(app.state, "autoreply_task"):
            try:
                await asyncio.wait_for(app.state.autoreply_task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                pass
    # Stop the context-pack reconcile loop.
    if hasattr(app.state, "context_packs_stop"):
        app.state.context_packs_stop.set()
        if hasattr(app.state, "context_packs_task"):
            try:
                await asyncio.wait_for(app.state.context_packs_task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                pass
    # Stop the Bat supervisor (before killing worker processes). Running Bat
    # runs go stale and are re-adopted on the next boot from their checkpoints.
    if hasattr(app.state, "bat_stop"):
        app.state.bat_stop.set()
        if hasattr(app.state, "bat_task"):
            try:
                await asyncio.wait_for(app.state.bat_task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                pass
        if hasattr(app.state, "bat_store"):
            try:
                await app.state.bat_store.close()
            except Exception:
                pass
    try:
        await payments_routes.close()
    except Exception:
        pass
    if hasattr(app.state, "tool_audit_stop"):
        app.state.tool_audit_stop.set()
        try:
            await asyncio.wait_for(app.state.tool_audit_task, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError, Exception):
            pass
    try:
        await tool_audit_routes.close()
    except Exception:
        pass
    # Shutdown: stop all managed process agents
    print("Flight Deck: stopping managed process agents...")
    _stop_all_processes()
    # Stop every running code-app subprocess + cancel the idle reaper.
    if hasattr(app.state, "app_runtime"):
        try:
            await app.state.app_runtime.shutdown()
        except Exception as _exc:
            print(f"Flight Deck: app_runtime shutdown error: {_exc}")
    if hasattr(app.state, "vastai_manager"):
        await app.state.vastai_manager.shutdown()
    await _fd_db.close()
    if _client:
        _client.close()


app = FastAPI(title="Flight Deck", lifespan=lifespan)

# gzip for the smart-glasses HUD page and the built /assets/* (plus long-lived
# caching for the content-hashed assets); every other path, /fd/* streams
# included, is untouched. Added first so it sits innermost, next to the routes.
from captain_claw.flight_deck.hud_routes import HudDeliveryMiddleware  # noqa: E402

app.add_middleware(HudDeliveryMiddleware)

# CORS: this deck's own origins only (FD_PUBLIC_URL, FD_ALLOWED_HOSTS names,
# the Vite dev server) plus FD_CORS_ORIGINS — a comma-separated allowlist for
# a frontend served from another origin (e.g. "https://app.example.com").
# The old unset default was '*' with credentials, which Starlette answers by
# mirroring ANY origin: every website the user visited could read (and, on an
# auth-disabled deck, drive) the whole API. FD_CORS_ORIGINS=* restores that,
# explicitly. The browser guard below enforces the same origin set.
app.add_middleware(
    origin_guard.FDCORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Deployment hardening (default-off) ──
# Two independent switches for deployments where FD is reachable beyond the
# owner's own machine:
#
# * Agent-facing routes (/fd/basna/agent/*, /fd/vatra/agent/*) identify their
#   caller by web_auth/source_port with an owner_id fallback — fine when only
#   locally spawned agents can reach the port, spoofable otherwise. They now
#   require loopback or the shared agent secret (X-Agent-Secret).
# * The code- and hosting-studio agent routes (/fd/code/agent/*,
#   /fd/hosting/agent/*) use the same web_auth/source_port/owner_id fallback but
#   ALSO make a verified bearer authoritative in-handler (the caller acts only
#   as itself). They require a valid bearer OR loopback OR the shared secret —
#   so a bearer client (e.g. Captain Spark) and locally-spawned agents pass,
#   while an off-machine caller with none of the three cannot impersonate an
#   owner via a forged owner_id / source_port.
# * The peer routes (/fd/consult-peer, /fd/delegate-peer) sit behind the same
#   bearer-or-agent guard; in-handler they additionally require the caller's
#   identity (bearer, or the agent's own X-Agent-Auth) to own the target — see
#   `_resolve_peer_caller`.
# * FD_LOCKDOWN=1 additionally (a) makes the agent secret mandatory even from
#   loopback (a TLS reverse proxy on the same host would otherwise launder
#   remote callers into "loopback"), and (b) disables the host-filesystem
#   surfaces that make no sense off-machine: /fd/vfs/browse-fs, POST
#   /fd/vfs/links (mounts arbitrary host dirs), and the auth-less
#   /fd/projects/* router.


def _lockdown_enabled() -> bool:
    return os.environ.get("FD_LOCKDOWN", "").lower() in ("true", "1", "yes")


_AGENT_GUARD_PREFIXES = ("/fd/basna/agent/", "/fd/vatra/agent/", "/fd/bat/agent/")

# Agent routes whose handlers treat a verified bearer as authoritative (caller
# may act only as itself). A valid bearer is an accepted caller here in
# addition to loopback / the shared agent secret; without any of the three the
# owner_id/source_port fallback would be spoofable off-machine, so the same
# transport guard applies. (Basna/Vatra do not consult a bearer, so they stay
# loopback-or-secret only, above.)
_BEARER_OR_AGENT_GUARD_PREFIXES = ("/fd/code/agent/", "/fd/hosting/agent/",
                                   "/fd/consult-peer", "/fd/delegate-peer")


def _agent_caller_ok(request: Request) -> bool:
    provided = request.headers.get("X-Agent-Secret", "")
    if provided:
        from captain_claw.flight_deck.agent_secret import get_or_create_agent_secret
        if secrets.compare_digest(provided, get_or_create_agent_secret()):
            return True
    if _lockdown_enabled():
        return False
    client_host = request.client.host if request.client else ""
    return client_host in ("127.0.0.1", "::1", "localhost")


def _agent_proxy_port(path: str) -> int | None:
    """Return the target agent web-port for a port-addressed proxy path.

    All cross-user agent proxies are shaped ``/fd/agent-<x>/{host}/{port}/...``
    or ``/fd/orchestrator/{host}/{port}/...`` — host at segment 3, port at
    segment 4. The ``/fd/agent-config|model|mode/{kind}/{identifier}`` family
    uses a slug (non-numeric) at segment 4, so keying on a numeric 4th segment
    cleanly selects only the port-addressed routes.
    """
    parts = path.split("/")
    if len(parts) >= 5 and parts[1] == "fd":
        seg = parts[2]
        if (seg.startswith("agent-") or seg == "orchestrator") and parts[4].isdigit():
            return int(parts[4])
    return None


def _request_jwt_payload(request: Request) -> dict | None:
    """Decode the caller's access token from Authorization or fd_token/token.

    Runs inside HTTP middleware, before route auth dependencies populate
    ``request.state``. Returns None on any missing/invalid token.
    """
    auth_hdr = request.headers.get("Authorization", "")
    tok = ""
    if auth_hdr.lower().startswith("bearer "):
        tok = auth_hdr[7:].strip()
    if not tok:
        tok = request.query_params.get("fd_token") or request.query_params.get("token") or ""
    if not tok:
        return None
    try:
        return decode_access_token(tok)
    except HTTPException:
        return None


@app.middleware("http")
async def _hardening_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith(_AGENT_GUARD_PREFIXES):
        if not _agent_caller_ok(request):
            return JSONResponse(
                status_code=403,
                content={"detail": "agent routes require loopback or X-Agent-Secret"})
    elif path.startswith(_BEARER_OR_AGENT_GUARD_PREFIXES):
        # A valid bearer (e.g. Captain Spark's service account) is authoritative
        # in the handler, so accept it here; otherwise require loopback or the
        # shared secret, exactly as for the guarded prefixes above. This closes
        # the unauthenticated owner_id/source_port impersonation vector while
        # leaving bearer callers and locally-spawned agents unaffected.
        if _request_jwt_payload(request) is None and not _agent_caller_ok(request):
            return JSONResponse(
                status_code=403,
                content={"detail": "agent routes require a bearer token, loopback, or X-Agent-Secret"})
    else:
        # Ownership guard for port-addressed agent proxies: an authenticated
        # user may only reach agents they own (admins reach any). Without this,
        # any logged-in teammate could iterate ports and read another user's
        # agent data — the FD proxy auto-injects the target agent's own token.
        _pport = _agent_proxy_port(path)
        if _pport is not None and AUTH_ENABLED:
            payload = _request_jwt_payload(request)
            if not payload:
                return JSONResponse(status_code=401, content={"detail": "Not authenticated"})
            if payload.get("role", "user") != "admin":
                owner = _resolve_agent_owner(_pport)
                if owner and owner != payload.get("sub", ""):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "This agent belongs to another user"})
    if _lockdown_enabled() and not path.startswith(_AGENT_GUARD_PREFIXES):
        if (path == "/fd/projects" or path.startswith("/fd/projects/")
                or path == "/fd/vfs/browse-fs"
                or (path == "/fd/vfs/links" and request.method == "POST")):
            return JSONResponse(
                status_code=403, content={"detail": "disabled by FD_LOCKDOWN"})
    return await call_next(request)


# A2: a shared-agent member's request (X-FD-Speaker-Grant header or the
# fd_member marker) reaches only the grant-aware routes; anywhere else it is
# refused (403 / WebSocket 4403). Just inside the browser guard.
app.add_middleware(speaker_grants.GrantGuardMiddleware)
# Host allowlist + cross-site browser guard for every HTTP and WebSocket route
# (see origin_guard). Added last so it runs outermost — before CORS and the
# hardening middleware — and WebSockets, which neither of those covers, pass
# through it too.
app.add_middleware(origin_guard.BrowserGuardMiddleware)


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
# Agent management (list / create / start / stop / config / remove): the same
# two, but an admin may act for another user with X-FD-Act-As — see
# auth.act_as_target. Only the routes below that take these honour the header.
_agent_manager_dep = Depends(get_managed_user) if AUTH_ENABLED else Depends(_no_user)
_optional_agent_manager_dep = Depends(get_optional_managed_user) if AUTH_ENABLED else Depends(_no_user)

# ── Auth & user routes ──

from captain_claw.flight_deck.auth_routes import router as auth_router
from captain_claw.flight_deck.settings_routes import router as settings_router
from captain_claw.flight_deck.botport_routes import router as botport_router
from captain_claw.flight_deck.botport_routes import forget_share as botport_forget_share
from captain_claw.flight_deck.chat_routes import router as chat_router
from captain_claw.flight_deck.admin_routes import router as admin_router
from captain_claw.flight_deck.profile_routes import router as profile_router
from captain_claw.flight_deck.council_routes import router as council_router
from captain_claw.flight_deck.basna_routes import router as basna_router
from captain_claw.flight_deck.vatra_routes import router as vatra_router
from captain_claw.flight_deck.bat_routes import router as bat_router
from captain_claw.flight_deck.dubina_routes import router as dubina_router
from captain_claw.flight_deck.vfs_routes import router as vfs_router
from captain_claw.flight_deck.deep_memory_routes import router as deep_memory_router
from captain_claw.flight_deck.hosting_routes import router as hosting_router
from captain_claw.flight_deck.code_routes import router as code_router
from captain_claw.flight_deck.queue_planner import router as queue_router
from captain_claw.flight_deck.google_oauth_routes import router as google_oauth_router
from captain_claw.flight_deck.gmail_send_routes import router as gmail_send_router
from captain_claw.flight_deck.gmail_autoreply_routes import router as gmail_autoreply_router
from captain_claw.flight_deck.microsoft_oauth_routes import router as microsoft_oauth_router
from captain_claw.flight_deck.jev_routes import router as jev_router
from captain_claw.flight_deck.soniox_routes import router as soniox_router
from captain_claw.flight_deck.telegram_shared_routes import router as telegram_shared_router
from captain_claw.flight_deck.telegram_inbox_routes import router as telegram_inbox_router
from captain_claw.flight_deck.whatsapp_web_routes import router as whatsapp_web_router
from captain_claw.flight_deck.branding_routes import router as branding_router
from captain_claw.flight_deck.invite_routes import admin_router as invite_admin_router
from captain_claw.flight_deck.invite_routes import public_router as invite_public_router
from captain_claw.flight_deck.info_routes import router as info_router
from captain_claw.flight_deck.codex_oauth_routes import router as codex_oauth_router
from captain_claw.flight_deck.antigravity_routes import router as antigravity_router
from captain_claw.flight_deck.games_routes import router as games_router
from captain_claw.flight_deck.vastai_routes import router as vastai_router
from captain_claw.flight_deck.prompt_routes import router as prompt_router
from captain_claw.flight_deck.archetype_routes import router as archetype_router
from captain_claw.flight_deck.project_routes import router as project_router
from captain_claw.flight_deck.mcp_routes import router as mcp_router
from captain_claw.flight_deck.app_routes import router as apps_router
from captain_claw.flight_deck.app_files_routes import router as app_files_router
from captain_claw.flight_deck.app_builtin_routes import router as app_builtin_router
from captain_claw.flight_deck.app_code_routes import router as app_code_router
from captain_claw.flight_deck.glasses_bridge import router as glasses_router
from captain_claw.flight_deck.agent_link import router as agent_link_router
from captain_claw.flight_deck.agent_link_http import router as agent_link_http_router
from captain_claw.flight_deck.device_bridge import router as device_router
from captain_claw.flight_deck.device_firmware import router as device_firmware_router
from captain_claw.flight_deck.device_backgrounds import router as device_backgrounds_router
from captain_claw.flight_deck.face_routes import router as face_router
from captain_claw.flight_deck.messenger_bridge import router as messenger_router
from captain_claw.flight_deck.whatsapp_bridge import router as whatsapp_router
from captain_claw.flight_deck.fd_scheduler import router as scheduler_router
from captain_claw.flight_deck.consciousness_routes import router as consciousness_router
from captain_claw.flight_deck.autonomy_routes import router as autonomy_router
from captain_claw.flight_deck.event_routes import router as event_router
from captain_claw.flight_deck.delivery_routes import router as delivery_router
from captain_claw.flight_deck.agents_fs_routes import router as agents_fs_router
from captain_claw.flight_deck.file_library import router as file_library_router
from captain_claw.flight_deck.data_library import router as data_library_router
from captain_claw.flight_deck.agent_skills import router as agent_skills_router
from captain_claw.flight_deck import agent_preset
from captain_claw.flight_deck.system_routes import router as system_router
from captain_claw.flight_deck.share_routes import router as share_router
from captain_claw.flight_deck import public_file_shares
from captain_claw.flight_deck import payments_routes
from captain_claw.flight_deck import tool_audit_routes
from captain_claw.flight_deck.agent_sharing_routes import router as agent_sharing_router
from captain_claw.flight_deck.context_pack_routes import router as context_pack_router
from captain_claw.flight_deck.shared_workspace_routes import router as shared_workspace_router
from captain_claw.flight_deck.shared_usage_routes import router as shared_usage_router
from captain_claw.flight_deck.notification_routes import router as notification_router
from captain_claw.flight_deck.mcp_server_routes import router as mcp_inbound_router
from captain_claw.flight_deck.mcp_oauth_routes import router as mcp_oauth_router
from captain_claw.flight_deck.costs_routes import router as costs_router
from captain_claw.flight_deck.usage_quota import router as usage_quota_router
from captain_claw.flight_deck.byok import router as byok_router
from captain_claw.flight_deck.llm_routes import router as llm_router
from captain_claw.flight_deck.being_routes import router as being_router
from captain_claw.flight_deck.being_public_routes import router as being_public_router
from captain_claw.flight_deck.being_public_routes import village_router as being_village_router
from captain_claw.flight_deck.hud_routes import router as hud_router
from captain_claw.flight_deck.hud_routes import (
    ROOT_HEADERS as HUD_ROOT_HEADERS,
    root_override as hud_root_override,
)
from captain_claw.terminal.relay import router as pty_router

app.include_router(auth_router)
app.include_router(settings_router)
app.include_router(botport_router)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(profile_router)
app.include_router(council_router)
app.include_router(basna_router)
app.include_router(vatra_router)
app.include_router(bat_router)
app.include_router(dubina_router)
app.include_router(vfs_router)
app.include_router(deep_memory_router)
app.include_router(hosting_router)
app.include_router(code_router)
app.include_router(queue_router)
app.include_router(google_oauth_router)
app.include_router(gmail_send_router)
app.include_router(gmail_autoreply_router)
app.include_router(microsoft_oauth_router)
app.include_router(jev_router)
app.include_router(soniox_router)
app.include_router(telegram_shared_router)
app.include_router(telegram_inbox_router)
app.include_router(whatsapp_web_router)
app.include_router(branding_router)
app.include_router(invite_admin_router)
app.include_router(invite_public_router)
app.include_router(info_router)
app.include_router(codex_oauth_router)
app.include_router(antigravity_router)
app.include_router(games_router)
app.include_router(vastai_router)
app.include_router(prompt_router)
app.include_router(archetype_router)
app.include_router(project_router)
app.include_router(mcp_router)
# Legacy manifest-based app routes. The agent-coded app runtime
# (``app_code_router``) is the going-forward path; the manifest
# routes are kept for migration but disabled by default. Re-enable
# by setting ``FD_LEGACY_MANIFEST_APPS=true``.
if os.environ.get("FD_LEGACY_MANIFEST_APPS", "false").lower() in ("true", "1", "yes"):
    app.include_router(apps_router)
    app.include_router(app_files_router)
    app.include_router(app_builtin_router)
app.include_router(app_code_router)
app.include_router(glasses_router)
# Handheld voice device (M5Stack StickS3): /device/ws.
app.include_router(device_router)
# Agent links: a connected system talks to one agent (/link/ws, /fd/links),
# or over HTTP and a webhook, and the agent sends it messages (/fd/agent-links).
app.include_router(agent_link_router)
app.include_router(agent_link_http_router)
# Over-the-air firmware for the devices (docs/device-ota-plan.md).
app.include_router(device_firmware_router)
# A picture behind the StopWatch's screens, sent by the owner's agent.
app.include_router(device_backgrounds_router)
# Smart-glasses HUD (/hud/*, /fd/hud/config). Must precede spa_catch_all.
app.include_router(hud_router)
app.include_router(face_router)
app.include_router(messenger_router)
app.include_router(whatsapp_router)
app.include_router(scheduler_router)
app.include_router(consciousness_router)
app.include_router(autonomy_router)
app.include_router(event_router)
app.include_router(delivery_router)
app.include_router(agents_fs_router)
app.include_router(file_library_router)
app.include_router(data_library_router)
app.include_router(agent_skills_router)
app.include_router(agent_preset.router)
app.include_router(system_router)
app.include_router(share_router)
# Public file links: owner API (port-addressed → the ownership guard above
# covers it) + the account-free /s/{token} viewer. Before spa_catch_all.
public_file_shares.configure(
    data_dir=DATA_DIR, resolve_auth=lambda port, host=None: _resolve_agent_auth(port, host=host),
    resolve_agent_endpoint=lambda token: _find_agent_endpoint_by_auth(token))
app.include_router(public_file_shares.owner_router)
app.include_router(public_file_shares.public_router)
# Payments (bill pay by barcode, FD_PAYMENTS): agent + owner API and the
# account-free /p/{token} phone page. Before spa_catch_all.
payments_routes.configure(data_dir=DATA_DIR)
# Purchases first: its /fd/payments/purchases… paths must win over
# /fd/payments/{payment_id}.
from captain_claw.flight_deck import purchases_routes  # noqa: E402 — Phase 3 purchases
from captain_claw.flight_deck import statements_routes  # noqa: E402 — Phase 4 bank statements
from captain_claw.flight_deck import payment_reminders  # noqa: E402 — Phase 4c due-date reminders
from captain_claw.flight_deck import recurring_bills  # noqa: E402 — Phase 4d recurring bills
app.include_router(purchases_routes.router)
app.include_router(recurring_bills.router)
app.include_router(statements_routes.router)
app.include_router(statements_routes.public_router)
app.include_router(payment_reminders.router)
app.include_router(payment_reminders.public_router)
app.include_router(payments_routes.router)
app.include_router(payments_routes.public_router)
# Tool-call register: every tool call the agents make (tool_audit_routes.py).
tool_audit_routes.configure(data_dir=DATA_DIR)
app.include_router(tool_audit_routes.router)
app.include_router(agent_sharing_router)
app.include_router(context_pack_router)
app.include_router(shared_workspace_router)
app.include_router(shared_usage_router)
app.include_router(notification_router)
app.include_router(mcp_inbound_router)
app.include_router(mcp_oauth_router)
app.include_router(costs_router)
app.include_router(usage_quota_router)
app.include_router(byok_router)
app.include_router(llm_router)
app.include_router(being_router)
app.include_router(being_public_router)
app.include_router(being_village_router)
app.include_router(pty_router)


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
    # Set by the archetype resolve, never by the caller: the tier the model
    # came from, and whether it is on the owner's own API key (byok.py).
    _model_tier: str = PrivateAttr(default="")
    _byok: bool = PrivateAttr(default=False)
    # Identity
    name: str = ""
    description: str = ""
    hostname: str = "captain-claw"
    image: str = CC_IMAGE_DEFAULT

    # LLM
    provider: str = "ollama"
    model: str = "minimax-m2.7:cloud"
    temperature: float = 0.7
    max_tokens: int = 32768  # output token limit (max completion tokens)
    max_context: int = 0  # input context window; 0 → default (160000)
    provider_api_key: str = ""
    base_url: str = ""
    # Does the model see images itself (auto | on | off)? Written as the
    # agent's `model.vision`; "auto" leaves it to the agent (known multimodal
    # families on, local builds off — Qwen quants ship with and without vision).
    vision: str = "auto"
    # Model recommendation tier (reason | balanced | fast | longctx). When set,
    # the spawn endpoints resolve it to a concrete provider/model via the central
    # tier table in instructions/archetypes.json, so model choices live in ONE
    # place and survive model releases/reprices. To pin a specific model instead,
    # leave `tier` empty and set provider/model explicitly (the Forge review step
    # clears `tier` when the user overrides the model).
    tier: str = ""
    # Optional archetype selector — `id` or `id@tier` (e.g. "fact-checker@reason").
    # When set, the spawn endpoints resolve it via `merged_archetypes` and fold the
    # archetype's cognitive_mode / tools / role / tier→model into this config
    # (mirroring dubina's `_build_agent_config`). Explicit fields the caller already
    # set win over the archetype; an unknown id is a non-fatal no-op.
    archetype: str = ""
    # The agent's standing instructions (its role brief — what a chat sends it
    # as `fleet_instructions` on connect). Filled from the archetype when empty,
    # and stored for the OWNER at spawn: the chat reads them from the owner's
    # settings, so they must be there whoever created the agent (the owner's
    # own browser, or an admin acting for them).
    fleet_instructions: str = Field(default="", max_length=64_000)
    # The kiosk dialog's "Advanced" setup, as the person reviewed it: the id
    # of a prepared preset (its documents wait server-side) and the items they
    # kept — rules, intents, facts… (see agent_preset). Process agents only:
    # seeded into the new agent's folder before it starts.
    preset: dict | None = None
    # Deep-memory grid config, populated when `archetype` resolves to a composed
    # function×domain leaf (see archetype_compose). `grid_memory_tags` are stamped
    # onto the agent's deep-memory writes; `grid_recall_mode` (pool | domain | self)
    # narrows its deep-memory reads. Empty for a normal archetype — the agent then
    # pools by owner with no tag filter, exactly as before. Carried into the
    # process registry / Docker labels at spawn so the FD deep-memory proxy can
    # resolve them without the agent asserting anything.
    grid_memory_tags: list[str] = Field(default_factory=list)
    grid_recall_mode: str = ""
    # Optional per-session selectable model list (config.model.allowed). Used by
    # the free-OpenRouter "Freebie" spawn so all free models are available to
    # switch between at runtime, with `model` as the default.
    allowed_models: list[dict] = Field(default_factory=list)

    # BotPort — the agent's own legacy direct link. Off: agents reach other
    # Flight Decks through this deck's BotPort link (flight_deck tool).
    botport_enabled: bool = False
    botport_url: str = ""
    botport_instance_name: str = ""
    botport_key: str = ""
    botport_secret: str = ""
    botport_max_concurrent: int = 5

    # Tools
    tools: list[str] = Field(default_factory=lambda: [
        "shell", "read", "write", "glob", "edit",
        "web_fetch", "web_search", "browser",
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

    # Runtime — "" / "classic" = the full 16-mixin agent loop; "mrav" = the
    # micro small-model runtime (hard 8k input cap per LLM call, see
    # docs/mrav-micro-agent-plan.md). Spawn writes `mrav.enabled: true` into
    # the agent's config.yaml; the agent process swaps loops behind the same
    # web server and chat WS.
    runtime: str = ""

    # Docker
    network_mode: str = "host"
    restart_policy: str = "unless-stopped"
    extra_volumes: list[dict] = Field(default_factory=list)
    env_vars: list[dict] = Field(default_factory=list)

    # Ownership hint — used by internal callers (e.g. Old Man) that cannot
    # authenticate via JWT but need the spawned agent to inherit the owner.
    # Never authoritative from an HTTP body: see _resolve_spawn_owner.
    owner_hint: str = ""

    # Workspace override — absolute path the agent's tools (read/write/edit/glob/
    # grep/shell) anchor to. Empty → the default per-agent ``data/workspace``.
    # Code mode sets this to a VFS project dir so the agent works directly in a
    # real repo (npm/pytest/git "just work") instead of via the ``vfs:`` scheme.
    workspace_path: str = ""


def _registry_tier(tier: str) -> dict | None:
    """A tier's definition in the central table (instructions/archetypes.json)."""
    registry_file = Path(__file__).parent.parent / "instructions" / "archetypes.json"
    tier_def = (json.loads(registry_file.read_text()).get("tiers") or {}).get(tier)
    return tier_def if isinstance(tier_def, dict) else None


def _resolve_tier(config: AgentConfig) -> None:
    """Resolve a model-recommendation `tier` to a concrete provider/model.

    Tier definitions live in the central table in instructions/archetypes.json
    (the same registry the Forge gallery reads). Keeping the tier→model mapping
    in one place means a model release or reprice is a single-file edit rather
    than touching every archetype.

    Mutates `config` in place. No-op when `tier` is empty (the common case for
    callers that pin provider/model directly). On a missing/invalid registry or
    an unknown tier, logs and leaves provider/model untouched rather than failing
    the spawn.
    """
    if not config.tier:
        return
    try:
        tier_def = _registry_tier(config.tier)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Tier resolution failed; using provider/model as-is",
                    tier=config.tier, error=str(exc))
        return
    if not tier_def:
        log.warning("Unknown model tier; using provider/model as-is", tier=config.tier)
        return
    config.provider = tier_def.get("provider", config.provider)
    config.model = tier_def.get("model", config.model)
    if tier_def.get("base_url"):
        config.base_url = tier_def["base_url"]
    log.info("Resolved model tier",
             tier=config.tier, provider=config.provider, model=config.model)


def _registry_tier_fits_the_caller(config: AgentConfig, tier: str) -> bool:
    """May an archetype spawn fall back on the registry's model for ``tier``?

    The registry names a provider and a model, no key and no endpoint — the
    child would run it on whatever the caller sent. A caller that is itself an
    agent sends its own working model, key and endpoint: fine when the
    registry's tier is on that same provider (a different model, same place),
    but another provider would be given the caller's key and gateway URL. Then
    the caller's own working model is the better child.
    """
    if "provider" not in config.model_fields_set:
        return True  # the caller brought no model of its own
    try:
        tier_def = _registry_tier(tier)
    except (OSError, json.JSONDecodeError):
        return True
    if not tier_def or tier_def.get("provider", config.provider) == config.provider:
        return True
    log.info("Archetype tier not in the owner's tier set and the registry's is another provider — "
             "keeping the caller's model", tier=tier, provider=config.provider, model=config.model)
    return False


async def _resolve_archetype(config: AgentConfig, request: Request, user: dict | None) -> None:
    """Resolve `config.archetype` (`id` or `id@tier`) into a concrete spawn config.

    Mirrors dubina's ``_build_agent_config``: the archetype supplies
    ``cognitive_mode``, ``tools``, a role-based ``description``, and — via its tier
    — the model. Fields the caller already set explicitly win over the archetype
    (so overrides still work), and the model is only overridden when the requested
    tier actually resolves against the owner's Library config, so a caller-inherited
    working model is never replaced by a keyless one. An unknown id is a non-fatal
    no-op — a bad selector must never fail the spawn. Runs before ``_resolve_tier``.
    """
    if not config.archetype:
        return
    aid, _, tier_suffix = config.archetype.strip().partition("@")
    aid, explicit_tier = aid.strip(), tier_suffix.strip()
    if not aid:
        return

    # Owner: authenticated user → request state → owner_hint → env. Determines
    # whose Library archetypes + tier keys we resolve against.
    uid = str((user or {}).get("id") or "")
    if not uid:
        uid = str(getattr(getattr(request, "state", None), "user_id", "") or "")
    if not uid:
        uid = config.owner_hint or os.environ.get("FD_OWNER_ID", "")
    if not uid:
        # Without sign-in (the desktop app) the deck's one user is the local
        # user: its archetypes and tier sets are the ones to resolve against.
        from captain_claw.flight_deck import auth as _auth

        if not _auth._fd_auth_enabled():
            uid = _auth._LOCAL_USER["id"]

    from captain_claw.flight_deck.archetypes import merged_archetypes
    from captain_claw.flight_deck.archetype_compose import resolve_pair
    from captain_claw.flight_deck.auth import get_db
    # Function×domain grid (flag-gated): a `function.domain` selector composes a
    # leaf from the two axis registries. None when the flag is off, the id isn't a
    # pair, or an axis is unknown — in which case we fall through to the normal
    # single-id lookup below, so base/user archetypes behave exactly as before.
    arch = resolve_pair(aid)
    if arch is None:
        try:
            arch = next(
                (a for a in await merged_archetypes(get_db(), uid or None) if a.get("id") == aid),
                None,
            )
        except Exception as exc:
            log.warning("Archetype resolution failed; spawning config as-is",
                        archetype=aid, error=str(exc))
            return
    if not arch:
        log.warning("Unknown archetype; spawning config as-is", archetype=aid)
        return

    # Fill from the archetype only where the caller left the AgentConfig defaults,
    # so explicit spawn overrides still take precedence.
    if config.cognitive_mode in ("", "neutra") and arch.get("cognitive_mode"):
        config.cognitive_mode = str(arch["cognitive_mode"])
    if config.tools == AgentConfig().tools and arch.get("tools"):
        # A stored archetype may predate a tool's retirement (gws).
        config.tools = without_retired_tools(list(arch["tools"]))
    if not config.description:
        config.description = str(arch.get("role") or arch.get("description") or f"archetype:{aid}")
    if not config.runtime and arch.get("runtime") in ("classic", "mrav"):
        config.runtime = str(arch["runtime"])
    if not config.fleet_instructions and arch.get("fleet_instructions"):
        config.fleet_instructions = str(arch["fleet_instructions"])
    # Composed function×domain leaves carry deep-memory grid config; base/user
    # archetypes don't, so these stay empty (today's behaviour) for them.
    if arch.get("memory_tags"):
        config.grid_memory_tags = [str(t) for t in arch["memory_tags"]]
    if arch.get("recall_mode"):
        config.grid_recall_mode = str(arch["recall_mode"])

    # Model: the requested tier (`@tier` wins, else the archetype's own default
    # tier) resolved against the OWNER's Library tier config — the same source
    # Basna/flows use, so the child gets a real provider/model/key. Only override
    # the caller-inherited model when we actually resolve one; otherwise leave it
    # and let `_resolve_tier` try the central registry table as a last resort.
    eff_tier = explicit_tier or str(arch.get("tier") or "")
    from captain_claw.flight_deck.basna_routes import _effective_key, _load_owner_tiers
    tiers_map: dict = {}
    owner_env: list = []
    try:
        tiers_map, owner_env = await _load_owner_tiers(get_db(), uid)
    except Exception as exc:
        log.warning("Archetype owner-config resolve failed", tier=eff_tier, error=str(exc))
    if eff_tier:
        tcfg: dict = (tiers_map or {}).get(eff_tier) or {}
        if tcfg.get("model"):
            new_provider = tcfg.get("provider", config.provider)
            provider_changed = new_provider != config.provider
            tier_key = str(tcfg.get("api_key") or "").strip()
            tier_base_url = str(tcfg.get("base_url") or "").strip()
            # Where the child runs. A caller that is itself an agent sends its
            # own key and endpoint as a baseline (flight_deck tool), and a key
            # belongs to its endpoint — so the two travel together:
            #   * the tier names an endpoint            → that one;
            #   * the tier carries a key but no endpoint → the provider's own
            #     (the key — or the team key behind "@system" — is that
            #     endpoint's, never the caller's gateway's);
            #   * the tier is just a model on the caller's provider (no key, no
            #     endpoint) → the caller's endpoint AND key, as one — unless one
            #     of the two signs in through ChatGPT and the other with a key:
            #     those are different places too.
            if tier_base_url:
                child_base_url = tier_base_url
            elif (provider_changed or tier_key
                  or _signs_in_through_chatgpt(new_provider, tcfg["model"])
                  != _signs_in_through_chatgpt(config.provider, config.model)):
                child_base_url = ""
            else:
                child_base_url = config.base_url
            # The tier's key verbatim, as the Library spawn sends it: "@system"
            # (team-default sets) is swapped for the org key by
            # _resolve_spawn_provider_key; a blank one stays blank, so the set's
            # own env var (below) or the caller's key applies — never the
            # provider's org key by default (a custom endpoint's own team key
            # is the one exception, see there). A caller key for another
            # provider, or another endpoint, is no use here.
            if tier_key:
                config.provider_api_key = tcfg["api_key"]
            elif not _same_endpoint(new_provider, child_base_url, config.provider, config.base_url):
                config.provider_api_key = ""
            config.provider = new_provider
            config.model = tcfg["model"]
            config.base_url = child_base_url
            # The tier's context sizes, where the caller didn't set its own (a
            # model whose output cap is below the 32768 default would 400).
            if "max_tokens" not in config.model_fields_set and int(tcfg.get("output_ctx") or 0) > 0:
                config.max_tokens = int(tcfg["output_ctx"])
            if "max_context" not in config.model_fields_set and int(tcfg.get("input_ctx") or 0) > 0:
                config.max_context = int(tcfg["input_ctx"])
            # The tier's model sees images (Library "Sees images"), unless the
            # caller said otherwise.
            if "vision" not in config.model_fields_set and _vision_switch(tcfg.get("vision")) is not None:
                config.vision = "on" if _vision_switch(tcfg.get("vision")) else "off"
            config.tier = ""  # pinned now — don't let _resolve_tier re-map it
            config._model_tier = eff_tier
            config._byok = bool(tcfg.get("byok"))
        elif not config.tier and _registry_tier_fits_the_caller(config, eff_tier):
            config.tier = eff_tier  # last resort: central registry tier table
    # The tier set's own keys (BRAVE_API_KEY, TAVILY_API_KEY, …) — what the
    # Library spawn sends as env_vars; without them the agent's tools have no
    # credentials. Names the caller set win, and a set var never clobbers the
    # resolved LLM key (_build_env writes env_vars after provider_api_key) —
    # unless that key is an "@system" with no org key behind it, where the
    # set's own var is the only key there is.
    have = {str(ev.get("key") or "") for ev in config.env_vars}
    llm_key = (config.provider_api_key or "").strip()
    if llm_key == "@system":
        llm_key = _effective_key(config.provider, llm_key, config.base_url) or ""
    llm_envs = _provider_key_env_names(config.provider) if llm_key else ()
    for ev in owner_env or []:
        if not isinstance(ev, dict):  # settings are free-form; never fail the spawn
            continue
        name = str(ev.get("key") or "").strip()
        if name and name not in have and name not in llm_envs and str(ev.get("value") or ""):
            config.env_vars.append({"key": name, "value": str(ev["value"])})
            have.add(name)
    log.info("Resolved archetype spawn", archetype=aid, tier=eff_tier or "(default)",
             cognitive_mode=config.cognitive_mode, provider=config.provider, model=config.model)


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
    # Stable id of this agent for sharing (agent_sharing): "docker:<slug>:<instance>".
    agent_ref: str = ""


class ContainerActionResult(BaseModel):
    ok: bool
    container_id: str
    message: str = ""
    old_container_id: str = ""
    # Set by a spawn only when FD replaced the requested web_auth_token (it was
    # another agent's): the token the new agent actually got.
    web_auth: str = ""


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


def _vision_switch(value: Any) -> bool | None:
    """An agent's `model.vision` from a spawn/tier value: True (on), False
    (off) or None (auto — the key is left out, the agent decides)."""
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"on", "true", "yes", "1"}:
        return True
    if text in {"off", "false", "no", "0"}:
        return False
    return None


def _vision_yaml(value: Any) -> dict:
    switch = _vision_switch(value)
    return {} if switch is None else {"vision": switch}


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
            **_vision_yaml(c.vision),
            **({"allowed": c.allowed_models} if c.allowed_models else {}),
        },
        "context": {
            "max_tokens": c.max_context if c.max_context > 0 else 160000,
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
            # Retired tools (gws) never reach an agent's config, whatever an
            # old archetype or spawn request still lists.
            "enabled": without_retired_tools(c.tools),
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
    if (c.runtime or "").strip().lower() == "mrav":
        cfg["mrav"] = {"enabled": True, "persona": (c.description or c.name or "")[:200]}
        # The tier's context sizes ARE the mrav caps: input_ctx → input_cap
        # (hard per-call prompt budget), output_ctx → output_cap. Unset (0)
        # keeps the runtime defaults (8192 / 1024).
        if c.max_context > 0:
            cfg["mrav"]["input_cap"] = c.max_context
        if c.max_tokens > 0:
            cfg["mrav"]["output_cap"] = c.max_tokens
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)


# The env var an agent reads its LLM provider key from.
_PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


# Other env names an agent also accepts for a provider's key.
_PROVIDER_KEY_ENV_ALSO = {"gemini": ("GOOGLE_API_KEY",)}


def _provider_key_env_names(provider: str) -> tuple[str, ...]:
    """Every env var an agent reads ``provider``'s API key from (() if none)."""
    name = _PROVIDER_KEY_ENV.get(provider or "", "")
    return (name, *_PROVIDER_KEY_ENV_ALSO.get(provider, ())) if name else ()


# Providers whose client refuses to make a call without a key, whatever the
# endpoint (the others send a keyless request to a custom ``base_url``).
_KEY_REQUIRED_ON_ANY_ENDPOINT = ("openai", "anthropic", "gemini")


def _signs_in_through_chatgpt(provider: str, model: str) -> bool:
    """OpenAI's GPT-5 / Codex family runs on the ChatGPT connection, not an API key."""
    if provider != "openai":
        return False
    try:
        from captain_claw.llm import _is_codex_family_model

        return bool(_is_codex_family_model(model or ""))
    except Exception:
        return False


def _needs_provider_key(provider: str, model: str = "", base_url: str = "") -> bool:
    """Does an agent on this provider/model/endpoint need the PROVIDER's API key?

    Not on a custom endpoint (a local server or gateway has its own key, or
    none), and not a model that signs in through the ChatGPT connection.
    """
    if not _provider_key_env_names(provider) or str(base_url or "").strip():
        return False
    return not _signs_in_through_chatgpt(provider, model)


def _build_env(c: AgentConfig) -> str:
    lines: list[str] = []
    if c.provider_api_key:
        env_name = _PROVIDER_KEY_ENV.get(c.provider, "API_KEY")
        lines.append(f"{env_name}={c.provider_api_key}")
    # Ollama inside Docker needs to reach the host
    if c.provider == "ollama":
        dhost = _docker_host()
        lines.append(f"OLLAMA_BASE_URL=http://{dhost}:11434")
    for ev in c.env_vars:
        if ev.get("key"):
            lines.append(f"{ev['key']}={ev.get('value', '')}")
    return "\n".join(lines) + "\n" if lines else ""


async def _system_json(setting: str) -> dict:
    from captain_claw.flight_deck.auth import get_db

    raw = await get_db().get_system_setting(setting)
    try:
        data = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


async def _team_key(provider: str, base_url: str) -> str:
    """The team's API key for a model endpoint ("" when there is none).

    A custom endpoint (``base_url``) has its own key (``fd:endpoint-keys``);
    the provider's own endpoint runs on the provider's (``fd:provider-keys``).
    A custom endpoint with no key of its own runs on the provider's only where
    the published team sets send the provider's "@system" tiers (a deck that
    routes a provider through one gateway — `routed_endpoint_ids`). One the
    caller names is never handed the provider's key, whether the caller's key
    is blank or the "@system" sentinel: the endpoint is theirs, the key the
    team's.
    """
    from captain_claw.flight_deck.basna_routes import (
        ENDPOINT_KEYS_SETTING, NO_PROVIDER_FALLBACK, endpoint_key_id, routed_endpoint_ids,
    )

    eid = endpoint_key_id(provider, base_url)
    if eid:
        key = str((await _system_json(ENDPOINT_KEYS_SETTING)).get(eid) or "")
        if key:
            return key
        shared = await _system_json("fd:shared-tier-sets")
        no_fallback = shared.get(NO_PROVIDER_FALLBACK)
        if eid not in routed_endpoint_ids(shared) or eid in (no_fallback if isinstance(no_fallback, list) else []):
            return ""  # not the team's endpoint, or the provider's key is known not to be its
    return str((await _system_json("fd:provider-keys")).get(provider) or "")


def _endpoint_label(provider: str, base_url: str) -> str:
    from captain_claw.flight_deck.admin_routes import _endpoint_host

    return f"{provider} at {_endpoint_host(base_url)}" if (base_url or "").strip() else provider


# Where an agent's environment sends its model calls when its config names no
# endpoint: the variables LiteLLM reads a provider's endpoint from then.
_PROVIDER_BASE_ENV = {
    "anthropic": ("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"),
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
    "gemini": ("GEMINI_API_BASE",),
    "xai": ("XAI_API_BASE",),
    "openrouter": ("OPENROUTER_API_BASE",),
}
# …and the ones that carry every request through another host, or have the
# agent trust another certificate authority — a proxy that can read the key.
_TRAFFIC_ENV = re.compile(
    r"^(?:HTTPS?_PROXY|ALL_PROXY|SSL_CERT_(?:FILE|DIR)|REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE)$", re.IGNORECASE)


async def _org_keys() -> set[str]:
    """The team's model keys as Flight Deck holds them (provider and endpoint)."""
    from captain_claw.flight_deck import auth as fd_auth
    from captain_claw.flight_deck.basna_routes import ENDPOINT_KEYS_SETTING

    if fd_auth._db is None:
        return set()  # no accounts database, so no team keys in it either
    keys: set[str] = set()
    for setting in ("fd:provider-keys", ENDPOINT_KEYS_SETTING):
        keys.update(v.strip() for v in (await _system_json(setting)).values() if isinstance(v, str))
    return keys - {"", "@system"}


def _agent_key_routes(model: dict, env: dict[str, str]) -> list[tuple[str, str]]:
    """(provider, endpoint) of every model call an agent makes with a key from
    its environment, as it picks them: its model unless the config gives it a
    key, and each ``model.allowed`` entry — on the model's provider with the
    model's key and (naming none) its endpoint, on another with that
    provider's key. A blank endpoint is the one the environment names for the
    provider (`_PROVIDER_BASE_ENV`), else the provider's own ("")."""
    main = str(model.get("provider") or "").strip().lower()
    main_key = str(model.get("api_key") or "").strip()
    main_base = str(model.get("base_url") or "").strip()
    allowed = model.get("allowed")
    entries = [(main, main_base)] + [
        (str(e.get("provider") or "").strip().lower(), str(e.get("base_url") or "").strip())
        for e in (allowed if isinstance(allowed, list) else []) if isinstance(e, dict)]
    routes: list[tuple[str, str]] = []
    for provider, base in entries:
        if provider == main:
            if main_key:
                continue
            base = base or main_base
        named = [str(env.get(n) or "").strip() for n in _PROVIDER_BASE_ENV.get(provider, ())]
        for endpoint in ([base] if base else [u for u in named if u] or [""]):
            if (provider, endpoint) not in routes:
                routes.append((provider, endpoint))
    return routes


async def _misrouted_team_keys(model: dict, env: dict[str, str]) -> list[tuple[str, str, str]]:
    """(env var, provider, endpoint) for each team key in ``env`` that the
    agent would send to an endpoint it isn't the team's key for (`_team_key`)."""
    org = await _org_keys()
    if not any(str(v).strip() in org for v in env.values()):
        return []
    out = []
    for provider, endpoint in _agent_key_routes(model, env):
        for name in _provider_key_env_names(provider):
            key = str(env.get(name) or "").strip()
            if key in org and await _team_key(provider, endpoint) != key:
                out.append((name, provider, endpoint))
    return out


async def _team_key_blockers(model: dict, env: dict[str, str], *, kept: dict[str, str] | None = None) -> list[str]:
    """Why a team key in an agent's ``env`` would leave the team's hands, as
    sentences ([] when it wouldn't, or ``env`` holds none): each endpoint it
    would go to that isn't the team's for it, and each proxy / certificate
    variable that would carry it — unless Flight Deck's own environment, a
    published team set, or ``kept`` (the .env as it was) sets it so already."""
    org = await _org_keys()
    if not any(str(v).strip() in org for v in env.values()):
        return []
    out = [f"the team's key in {name} would go to {_endpoint_label(provider, endpoint)}"
           for name, provider, endpoint in await _misrouted_team_keys(model, env)]
    sets = (await _system_json("fd:shared-tier-sets")).get("sets")
    team_env = {(str(ev.get("key") or ""), str(ev.get("value") or ""))
                for st in (sets if isinstance(sets, list) else []) if isinstance(st, dict)
                for ev in (st.get("envVars") if isinstance(st.get("envVars"), list) else [])
                if isinstance(ev, dict)}
    for name, value in env.items():
        value = str(value or "").strip()
        if (_TRAFFIC_ENV.match(name) and value and value != (kept or {}).get(name, "").strip()
                and value != os.environ.get(name, "").strip() and (name, value) not in team_env):
            out.append(f"{name} would carry the team's key through a host of its own")
    return out


def _team_key_refusal(blockers: list[str]) -> HTTPException:
    return HTTPException(400, (
        "Not saved: " + "; ".join(blockers) + ". The team's keys only go to the endpoints the "
        "team set up — give a key of your own for that endpoint, or ask an admin."))


async def _resolve_spawn_provider_key(config: AgentConfig, *, inherits_fd_env: bool = True) -> None:
    """`_resolve_spawn_key`, then: a team key the agent would hold — resolved
    there, or one its env vars carry (a flow spawn copies its origin agent's
    .env) — never goes on to an endpoint the spawn names, nor through a proxy
    of its own (`_team_key_blockers`): the spawn is refused (400)."""
    await _resolve_spawn_key(config, inherits_fd_env=inherits_fd_env)
    blockers = await _team_key_blockers(
        {"provider": config.provider, "base_url": config.base_url, "allowed": config.allowed_models},
        _dotenv_values(_build_env(config)))
    if blockers:
        raise HTTPException(400, (
            "This agent can't run on the team's key: " + "; ".join(blockers) + ". The team's "
            "keys only go to the endpoints the team set up — give a key of your own."))


async def _resolve_spawn_key(config: AgentConfig, *, inherits_fd_env: bool) -> None:
    """Give a spawn that brings no model key of its own the team's (`_team_key`).

    The browser never receives raw org keys (GET /fd/settings/provider-keys is
    masked). A spawn on a team tier carries the sentinel ``@system``; here —
    server-side, before anything is written for the agent — it is swapped for
    the real key. Both spawn endpoints call this right after the archetype /
    tier resolve.

    A BLANK key gets the team's key too (a user's own set that holds no key for
    a model the team has one for — typically a copy of a team set), but never
    over a key the spawn already has: a real one, one in its env vars, or the
    one a process agent inherits from Flight Deck's environment.

    Where the team has no key for the endpoint, a set published before
    publishing shared its keys may still be waiting for them: they are
    recovered from the publishing admin's own copy (`recover_team_keys`).

    A spawn that ends with no key at all for a model that needs one is refused
    (409) rather than started as an agent whose first call fails with "missing
    credentials" — when it asked for the team key, or is an archetype spawn
    (Flight Deck picked the model; the caller had no say in the key). Not when
    the model needs none (no key env for the provider, ChatGPT sign-in).
    """
    requested = (config.provider_api_key or "").strip()
    if requested and requested != "@system":
        return
    asked = requested == "@system"
    env_names = _provider_key_env_names(config.provider)
    custom = bool((config.base_url or "").strip())
    chatgpt = _signs_in_through_chatgpt(config.provider, config.model)
    # A key that reaches the agent without us: its own env vars, or (process
    # agents only — containers don't inherit) Flight Deck's environment. That
    # one is the PROVIDER's key: on a custom endpoint it doesn't stand in for
    # the endpoint's team key (it would be sent to that host instead).
    in_env_vars = any(isinstance(ev, dict) and ev.get("key") in env_names and ev.get("value")
                      for ev in config.env_vars)
    fd_env = inherits_fd_env and any(os.environ.get(n) for n in env_names)
    supplied = in_env_vars or (fd_env and not custom)
    if not asked and (chatgpt or not (env_names or custom) or supplied):
        return
    config.provider_api_key = ""
    try:
        key = await _team_key(config.provider, config.base_url)
        if not key and not supplied:
            from captain_claw.flight_deck.admin_routes import recover_team_keys
            from captain_claw.flight_deck.auth import get_db

            if await recover_team_keys(get_db()):
                key = await _team_key(config.provider, config.base_url)
    except Exception as exc:
        log.warning("team key resolve failed", provider=config.provider, error=str(exc))
        return
    config.provider_api_key = key
    if key or supplied or fd_env or chatgpt or not env_names:
        return
    if not (asked or config.archetype):
        return
    if not asked and custom and config.provider not in _KEY_REQUIRED_ON_ANY_ENDPOINT:
        return  # a local / no-auth server behind this provider runs keyless
    raise HTTPException(409, (
        f"No API key for {_endpoint_label(config.provider, config.base_url)}: the tier set this "
        "agent runs on has none, and neither has the team. An admin can publish the tier set "
        "again in Library (that shares its keys), or add the key in Admin → Provider keys."
        + (" A local server that needs no key still needs a placeholder one in the tier."
           if custom else "")
    ))


def _agent_config_model(slug: str) -> tuple[dict, dict] | None:
    """(model, provider_keys) sections of a process agent's config — the copy in
    its home overlaid on the one in its directory, as the agent loads them."""
    agent_dir = DATA_DIR / slug
    model: dict = {}
    provider_keys: dict = {}
    found = False
    for path in (agent_dir / "config.yaml",
                 agent_dir / "data" / "home-config-parent" / ".captain-claw" / "config.yaml"):
        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        found = True
        if isinstance(data.get("model"), dict):
            # A blank base_url in the home copy is a choice — back to the
            # provider's own endpoint — and wins, as it does in the agent.
            model.update({k: v for k, v in data["model"].items() if v not in (None, "") or k == "base_url"})
        if isinstance(data.get("provider_keys"), dict):
            provider_keys.update({k: v for k, v in data["provider_keys"].items() if v})
    return (model, provider_keys) if found else None


def _agent_model_key_gap(slug: str) -> tuple[str, str, str] | None:
    """(provider, base_url, env name) of a process agent that has no key for its
    model anywhere it looks — .env, its config, or the environment it inherits
    from Flight Deck — and needs one. None when it is fine, or not ours to judge."""
    loaded = _agent_config_model(slug)
    if loaded is None:
        return None
    model, provider_keys = loaded
    provider = str(model.get("provider") or "")
    base_url = str(model.get("base_url") or "").strip()
    env_names = _provider_key_env_names(provider)
    if not env_names or _signs_in_through_chatgpt(provider, str(model.get("model") or "")):
        return None
    if str(model.get("api_key") or "").strip() or str(provider_keys.get(provider) or "").strip():
        return None
    if not base_url and any(os.environ.get(n) for n in env_names):
        return None
    try:
        from captain_claw.config import Config

        dotenv = Config._read_dotenv_file(DATA_DIR / slug / ".env")
    except Exception:
        return None
    if any(str(dotenv.get(n) or "").strip() for n in env_names):
        return None
    return provider, base_url, env_names[0]


async def _heal_agent_model_key(slug: str) -> bool:
    """Write the team's key into a process agent that was created without one
    for its model (the team had none then). True when the .env changed — a
    running agent has to be restarted to pick it up."""
    try:
        gap = _agent_model_key_gap(slug)
        if not gap:
            return False
        provider, base_url, env_name = gap
        key = await _team_key(provider, base_url)
        if not key:
            log.warning("Agent has no model key and the team has none for it",
                        slug=slug, provider=provider, endpoint=_endpoint_label(provider, base_url))
            return False
        from captain_claw.config import Config

        dotenv = Config._read_dotenv_file(DATA_DIR / slug / ".env")
        blockers = await _team_key_blockers(
            (_agent_config_model(slug) or ({}, {}))[0], {**dotenv, env_name: key})
        if blockers:
            log.warning("Agent not given the team's key: " + "; ".join(blockers), slug=slug)
            return False
        if not _upsert_dotenv_var(DATA_DIR / slug / ".env", env_name, key):
            return False
    except Exception as exc:
        log.warning("agent key heal failed", slug=slug, error=str(exc))
        return False
    log.info("Agent had no model key — team key written", slug=slug, provider=provider)
    return True


def _restart_processes(slugs: list[str]) -> None:
    import time

    for slug in slugs:
        try:
            _do_stop_process(slug)
            time.sleep(1)
            _do_start_process(slug)
        except Exception as exc:
            log.warning("restart after key heal failed", slug=slug, error=str(exc))


async def heal_keyless_agents(*, restart: bool = True) -> list[str]:
    """`_heal_agent_model_key` for every process agent. Returns the slugs whose
    .env changed; with ``restart`` the running ones among them are restarted in
    the background so the key takes effect."""
    healed = [slug for slug in list(_load_process_registry()) if await _heal_agent_model_key(slug)]
    running = [slug for slug in healed if _process_is_alive(slug)] if restart else []
    if running:
        asyncio.get_running_loop().run_in_executor(None, _restart_processes, running)
    return healed


# ── Own API keys (byok.py): moving a user's agents onto them and back ──


def _drop_dotenv_values(env_path: Path, names: set[str], values: set[str] | None = None) -> bool:
    """Remove the ``NAME=value`` lines of a .env whose name is in *names* and
    (given *values*) whose value is one of them. True when the file changed."""
    try:
        lines = env_path.read_text().splitlines() if env_path.is_file() else []
    except OSError:
        return False
    keep = []
    for ln in lines:
        name, sep, value = ln.strip().partition("=")
        if sep and name in names and (values is None or value.strip().strip("'\"") in values):
            continue
        keep.append(ln)
    if len(keep) == len(lines):
        return False
    try:
        env_path.write_text("\n".join(keep) + "\n" if keep else "")
    except OSError:
        return False
    return True


def _set_agent_config_model(agent_dir: Path, provider: str, model: str, base_url: str,
                            max_tokens: int = 0) -> dict:
    """Point every copy of an agent's config at *provider*/*model* with no key
    in the config (the .env's wins). Returns what the first copy had."""
    before: dict = {}
    for cfg_path in _agent_config_paths(agent_dir):
        if not cfg_path.is_file():
            continue
        try:
            data = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        m = data.get("model") if isinstance(data.get("model"), dict) else {}
        if not before:
            before = dict(m)
        m.update(provider=provider, model=model, api_key="",
                 base_url=base_url or _default_base_url(provider))
        if max_tokens > 0:
            m["max_tokens"] = max_tokens
        data["model"] = m
        cfg_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True))
    return before


def _byok_agent_tier(entry: dict, tiers: dict) -> str:
    """The tier an agent's model came from: recorded at spawn, else its
    archetype's, else the tier whose model it runs, else "balanced"."""
    recorded = str(entry.get("model_tier") or entry.get("tier") or "")
    if recorded:
        return recorded
    aid = str(entry.get("archetype") or "")
    if aid:
        from captain_claw.flight_deck.basna_routes import _load_registry

        try:
            for arch in _load_registry().get("archetypes") or []:
                if isinstance(arch, dict) and arch.get("id") == aid and arch.get("tier"):
                    return str(arch["tier"])
        except Exception:
            pass
    for name, t in (tiers or {}).items():
        if isinstance(t, dict) and (t.get("provider"), t.get("model")) == (entry.get("provider"), entry.get("model")):
            return name
    return "balanced"


def byok_owner_agents(owner_id: str) -> list[dict]:
    """*owner_id*'s process agents: whether each runs, and on own keys."""
    return [{"slug": slug, "running": _process_is_alive(slug), "byok": entry.get("key_source") == "byok"}
            for slug, entry in _load_process_registry().items() if entry.get("owner") == owner_id]


async def byok_switch_agents(owner_id: str, tiers: dict | None, keys: dict) -> dict:
    """Move *owner_id*'s process agents onto their own API keys — *tiers* is
    their tier set with the keys over it (``byok.overlay_tiers``) — or, with
    None, back to the models they ran before. An agent on a local model stays
    there. Running ones are restarted. ``{"switched", "restarted"}``."""
    from captain_claw.flight_deck import byok
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.flight_deck.basna_routes import _owner_base_tier_set

    base: dict = {}
    if tiers is None:
        try:
            base = (await _owner_base_tier_set(get_db(), owner_id)).get("tiers") or {}
        except Exception:
            base = {}
    key_values = {str(k) for k in (keys or {}).values() if k}
    key_names = {n for p in byok.PROVIDERS for n in _provider_key_env_names(p)}
    registry = _load_process_registry()
    changed: list[str] = []
    for slug, entry in registry.items():
        if entry.get("owner") != owner_id:
            continue
        agent_dir = DATA_DIR / slug
        if not agent_dir.is_dir():
            continue
        env_path = agent_dir / ".env"
        on_own = entry.get("key_source") == "byok"
        if tiers is not None:
            if not on_own and str(entry.get("provider") or "") in byok.LOCAL_PROVIDERS:
                continue
            name = _byok_agent_tier(entry, tiers)
            tier = tiers.get(name) if isinstance(tiers.get(name), dict) else None
            if not (tier and tier.get("byok")):
                tier = tiers.get("balanced") if isinstance(tiers.get("balanced"), dict) else None
            if not (tier and tier.get("byok")):
                continue
            provider, model = str(tier["provider"]), str(tier["model"])
            key = str(keys.get(provider) or tier.get("api_key") or "")
            env_name = _PROVIDER_KEY_ENV.get(provider, "")
            if not (key and env_name):
                continue
            try:
                from captain_claw.config import Config

                held = str(Config._read_dotenv_file(env_path).get(env_name) or "")
            except Exception:
                held = ""
            if on_own and (entry.get("provider"), entry.get("model"), held) == (provider, model, key):
                continue
            if on_own and entry.get("provider") != provider:
                # Off the provider it ran on with an own key: that key (put
                # there by this switch, maybe one since removed) goes too.
                _drop_dotenv_values(env_path, set(_provider_key_env_names(str(entry.get("provider") or ""))))
            before = _set_agent_config_model(agent_dir, provider, model, "",
                                             int(tier.get("output_ctx") or 0))
            if not on_own:
                entry["byok_prev"] = {
                    "provider": str(entry.get("provider") or before.get("provider") or ""),
                    "model": str(entry.get("model") or before.get("model") or ""),
                    "base_url": str(before.get("base_url") or ""),
                    "max_tokens": int(before.get("max_tokens") or 0),
                }
            _upsert_dotenv_var(env_path, env_name, key)
            entry.update(provider=provider, model=model, key_source="byok", model_tier=name)
        else:
            if not on_own:
                continue
            prev = entry.get("byok_prev") if isinstance(entry.get("byok_prev"), dict) else {}
            if not prev.get("model"):
                # Made on own keys: back to its tier on the deck's set.
                name = _byok_agent_tier(entry, base)
                team = base.get(name) if isinstance(base.get(name), dict) else {}
                prev = {"provider": team.get("provider"), "model": team.get("model"),
                        "base_url": team.get("base_url") or "", "max_tokens": int(team.get("output_ctx") or 0)}
            if prev.get("model"):
                _set_agent_config_model(agent_dir, str(prev.get("provider") or ""), str(prev["model"]),
                                        str(prev.get("base_url") or ""), int(prev.get("max_tokens") or 0))
                entry.update(provider=str(prev.get("provider") or ""), model=str(prev["model"]))
            _drop_dotenv_values(env_path, key_names, key_values)
            entry.pop("key_source", None)
            entry.pop("byok_prev", None)
        changed.append(slug)
    if changed:
        _save_process_registry(registry)
    if tiers is None:
        for slug in changed:
            await _heal_agent_model_key(slug)  # the deck's key, where it has one
    running = [slug for slug in changed if _process_is_alive(slug)]
    if running:
        asyncio.get_running_loop().run_in_executor(None, _restart_processes, running)
    log.info("Agents moved %s own API keys", "onto" if tiers is not None else "off",
             owner=owner_id, switched=len(changed), restarted=len(running))
    return {"switched": len(changed), "restarted": len(running)}


async def _startup_team_keys_then_reattach() -> None:
    """Startup, once the server is listening: team keys first — a set published
    before publishing shared its keys gets them now, and agents created without
    a model key get theirs — then the dead agents are started again (with the
    key), and the ones that were running without it are restarted."""
    stale: list[str] = []
    try:
        from captain_claw.flight_deck import admin_routes
        from captain_claw.flight_deck.auth import get_db

        admin_routes._agent_key_healer = heal_keyless_agents  # THIS module's — see there
        left = (await admin_routes.recover_team_keys(get_db())).get("unresolved")
        if left:
            log.warning("No team API key for %s — the published tier set names it, but no admin's "
                        "copy of the set holds a key. Publish the set again in Library.", ", ".join(left))
        healed = await heal_keyless_agents(restart=False)
        stale = [slug for slug in healed if _process_is_alive(slug)]
    except Exception as exc:
        log.warning("team key recovery at startup failed", error=str(exc))
    await asyncio.to_thread(_reattach_processes)
    if stale:  # after the reattach: both rewrite the process registry
        await asyncio.to_thread(_restart_processes, stale)


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
            **_vision_yaml(c.vision),
            **({"allowed": c.allowed_models} if c.allowed_models else {}),
        },
        "context": {
            "max_tokens": c.max_context if c.max_context > 0 else 160000,
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
            # Retired tools (gws) never reach an agent's config, whatever an
            # old archetype or spawn request still lists.
            "enabled": without_retired_tools(c.tools),
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
        "workspace": {"path": c.workspace_path or str(agent_dir / "data" / "workspace")},
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
    if (c.runtime or "").strip().lower() == "mrav":
        cfg["mrav"] = {"enabled": True, "persona": (c.description or c.name or "")[:200]}
        # The tier's context sizes ARE the mrav caps: input_ctx → input_cap
        # (hard per-call prompt budget), output_ctx → output_cap. Unset (0)
        # keeps the runtime defaults (8192 / 1024).
        if c.max_context > 0:
            cfg["mrav"]["input_cap"] = c.max_context
        if c.max_tokens > 0:
            cfg["mrav"]["output_cap"] = c.max_tokens
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
        agent_ref=_container_agent_ref(c),
    )


def _container_agent_ref(c) -> str:
    """``docker:<slug>:<instance>`` (see agent_sharing), "" when it has none."""
    try:
        return agent_sharing.docker_ref(c)
    except Exception:
        return ""


def _find_container(container_id: str, owner_id: str = "") -> docker.models.containers.Container:
    # Try by short ID, full ID, or name — among THIS deck's containers only: a
    # container another deck on this host spawned is not ours to stop, rebuild,
    # clone or read (with auth off there is no owner check to stop it).
    for c in _deck_containers(all=True):
        if c.short_id == container_id or c.id == container_id or c.name == container_id:
            if AUTH_ENABLED and owner_id:
                if (c.labels or {}).get(OWNER_LABEL, "") != owner_id:
                    raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
            return c
    raise HTTPException(status_code=404, detail=f"Container {container_id} not found")


# ── Endpoints ──

@app.get("/fd/containers", response_model=list[ContainerInfo])
async def list_containers(request: Request, user: dict | None = _agent_manager_dep):
    """List this deck's managed containers (filtered by owner when auth enabled).

    Never another deck's: each entry carries the container's web_auth — its
    identity token to the deck that spawned it (X-Agent-Auth → that deck's
    user's Google etc.).
    """
    try:
        containers = _deck_containers(all=True)
    except Exception:
        return []  # Docker not available (e.g. running inside a container)
    user_id = getattr(request.state, "user_id", "")
    if AUTH_ENABLED and user_id:
        containers = [c for c in containers if (c.labels or {}).get(OWNER_LABEL, "") == user_id]
    infos = [_container_info(c) for c in containers]
    if (not origin_guard.may_expose_agent_secrets(request.headers)
            or getattr(request.state, "acting_admin_id", "")):  # see list_processes
        for info in infos:
            info.web_auth = ""
    return infos


def _port_reserved(port: int, slug: str) -> bool:
    """``port`` is another registered process agent's (running, stopped or asleep)."""
    for other, entry in _load_process_registry().items():
        if other != slug and int(entry.get("web_port") or 0) == port:
            return True
    return False


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


# ── Spawn ownership ──
#
# The owner recorded at spawn IS the agent's tenant: `_resolve_agent_owner_by_auth`
# maps the agent's web_auth back to it, and that decides whose Google account,
# VFS root, deep-memory pool and Library keys the agent acts with. So it comes
# from a verified identity or FD's own records — never from a request body, and
# never borrowed from some unrelated agent.


def _legacy_inherited_owner(include_docker: bool = False) -> str:
    """Auth-disabled only: the first owner FD has on record (process registry,
    then this deck's Docker labels). With auth off there is a single tenant, so any recorded
    owner is it; with auth on this would hand the agent an arbitrary tenant."""
    for entry in _load_process_registry().values():
        if entry.get("owner"):
            return str(entry["owner"])
    if include_docker:
        try:
            for c in _deck_containers(all=True):
                o = (c.labels or {}).get(OWNER_LABEL, "")
                if o:
                    return o
        except Exception:
            pass
    return ""


async def _is_deck_user(uid: str) -> bool:
    """True when ``uid`` is a user of THIS deck's DB."""
    if not uid:
        return False
    try:
        from captain_claw.flight_deck.auth import get_db
        return await get_db().get_user_by_id(uid) is not None
    except Exception:
        return False


async def _sole_user_id() -> str:
    """The deck's only user when exactly one exists (single-user mode), else ""."""
    try:
        from captain_claw.flight_deck.auth import get_db
        db = get_db()
        if await db.count_users() != 1:
            return ""
        users = await db.list_users(limit=1)
    except Exception:
        return ""
    return str(users[0].get("id", "")) if users else ""


async def _resolve_unverified_spawn_owner(request: Request, hint: str) -> str:
    """Owner for an HTTP spawn that carried no valid JWT while auth is on — in
    practice an FD-spawned agent's ``flight_deck`` spawn tool, which sends its
    own ``X-Agent-Auth`` + ``X-Agent-Secret``.

    The owner comes ONLY from the calling agent's identity, never from the
    body: an ``owner_hint`` names a tenant but proves nothing (user ids are
    listed to every user by /fd/shares/users), and there is no "sole user"
    fallback either — any web page open on the FD host can POST here.

    1. Transport guard (loopback or X-Agent-Secret; FD_LOCKDOWN makes the
       secret mandatory), so an off-machine caller can't spawn anything.
    2. A browser (it sends ``Origin``) must be signed in: 401, which also
       sends a SPA whose access token expired off to refresh.
    3. ``X-Agent-Auth`` must be a web_auth THIS deck issued
       (`_resolve_agent_identity_by_auth`), and its recorded owner a user of
       this deck. An agent this deck recorded without an owner (spawned while
       auth was off, pre-owner entries) belongs to the deck's sole user in
       single-user mode — the token being this deck's own makes that safe —
       and is refused on a multi-user deck. The synthetic ``local`` owner an
       auth-disabled deck records counts as no owner (as for Google), and so
       does an ``owner_hint`` of ``local`` (such an agent's FD_OWNER_ID);
       any other ``owner_hint`` must agree.
    """
    from captain_claw.flight_deck.auth import _LOCAL_USER

    local = str(_LOCAL_USER["id"])
    if not _agent_caller_ok(request):
        raise HTTPException(401, "Not authenticated")
    if request.headers.get("Origin"):
        raise HTTPException(401, "Not authenticated")
    token = request.headers.get("X-Agent-Auth", "").strip()
    if not token:
        raise HTTPException(401, "Not authenticated")
    matched, owner = _resolve_agent_identity_by_auth(token)
    if not matched:
        raise HTTPException(403, "could not resolve calling agent's owner")
    if owner and owner != local:
        if not await _is_deck_user(owner):
            raise HTTPException(403, "could not resolve calling agent's owner")
    else:
        owner = await _sole_user_id()
        if not owner:
            raise HTTPException(403, "calling agent has no recorded owner on this deck")
    if hint and hint != local and hint != owner:
        raise HTTPException(403, "owner_hint does not match the calling agent")
    return owner


async def _resolve_spawn_owner(config: AgentConfig, request, *, docker: bool = False) -> str:
    """Authoritative owner for a new agent, or an HTTPException.

    * ``request.state.user_id`` — set only by the auth dependency from a
      verified JWT, or by an in-process caller's stub Request (Basna / Vatra /
      Dubina / flows / beings) carrying its run owner. Authoritative.
    * Auth disabled (desktop / local single-user): no tenant boundary, so the
      body's owner_hint is ignored — the calling agent's recorded owner when
      it proves one (X-Agent-Auth), else the first recorded owner.
    * In-process stub without a uid: its owner_hint was set by FD code; else
      the deck's sole user; else refuse rather than borrow another tenant.
    * Real HTTP request without a JWT: `_resolve_unverified_spawn_owner`. Such
      a caller also may not choose the child's web_auth (its identity token).

    Normalises ``config.owner_hint`` to the result so `_resolve_archetype`
    loads the SAME owner's Library archetypes and tier keys.
    """
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    hint = (config.owner_hint or "").strip()
    owner = str(getattr(getattr(request, "state", None), "user_id", "") or "")
    if not owner:
        is_http = isinstance(request, Request)
        if not _fd_auth_enabled():
            if is_http:
                token = request.headers.get("X-Agent-Auth", "").strip()
                owner = _resolve_agent_identity_by_auth(token)[1]
            else:
                owner = hint
            owner = owner or _legacy_inherited_owner(include_docker=docker)
        elif is_http:
            owner = await _resolve_unverified_spawn_owner(request, hint)
            config.web_auth_token = ""
            # …nor write its own text into the owner's settings: only an
            # archetype's instructions (filled in later) are stored for it.
            config.fleet_instructions = ""
        else:
            owner = hint or await _sole_user_id()
            if not owner:
                raise HTTPException(403, "could not resolve an owner for this spawn")
    config.owner_hint = owner
    return owner


def _web_auth_in_use(token: str, slug: str) -> bool:
    """True when an agent other than ``slug`` (process or container) already
    holds ``token``. web_auth IS an agent's identity to FD (X-Agent-Auth →
    recorded owner → that tenant's Google account etc.): two agents sharing one
    would be indistinguishable, and a spawn naming a victim's token would get
    the victim's identity."""
    if not token:
        return False
    for s, e in _load_process_registry().items():
        if s != slug and e.get("web_auth") == token:
            return True
    try:
        for c in _deck_containers(all=True):
            if c.name != slug and (c.labels or {}).get("flight-deck.web-auth", "") == token:
                return True
    except Exception:
        pass
    return False


# Appended to a spawn's message when the requested web_auth_token was replaced.
_WEB_AUTH_REPLACED_NOTE = (
    " — the requested web auth token is already used by another agent, so this"
    " agent got a new one"
)


def _claim_web_auth(config: AgentConfig, slug: str) -> bool:
    """Give the new agent a web_auth no other agent holds.

    Minted when none was requested — with web disabled too: it is also the
    agent's X-Agent-Auth identity (its owner's Google, spawning children), and
    captain-claw-web serves its port either way, without a token
    unauthenticated. A requested one (only a signed-in user or FD code can
    still request one: `_resolve_spawn_owner` clears an unverified caller's) is
    kept unless another agent already holds it — e.g. a Spawner preset that
    fixes one password for every agent. Then a fresh one is minted instead:
    two agents sharing an identity would be indistinguishable to FD.

    Returns True when a requested token was replaced.
    """
    requested = config.web_auth_token
    if requested and not _web_auth_in_use(requested, slug):
        return False
    config.web_auth_token = secrets.token_urlsafe(32)
    return bool(requested)


@app.post("/fd/spawn", response_model=ContainerActionResult)
async def spawn_agent(config: AgentConfig, request: Request, user: dict | None = _optional_agent_manager_dep):
    """Spawn a new Captain Claw container."""
    # Owner first: it gates the whole spawn (nothing is written or removed for
    # a caller we can't attribute) and feeds the archetype's Library lookup.
    owner_id = await _resolve_spawn_owner(config, request, docker=True)
    # Resolve an archetype selector (if any) into cognitive_mode/tools/tier/model,
    # then a bare model-recommendation tier to a concrete provider/model.
    await _resolve_archetype(config, request, user)
    _resolve_tier(config)
    await _resolve_spawn_provider_key(config, inherits_fd_env=False)  # a container doesn't get FD's env
    # Check if docker spawn is allowed
    sys_cfg = await _get_system_config()
    docker_default = not os.environ.get("CAPTAIN_CLAW_DOCKER")
    if not sys_cfg.get("docker_spawn_enabled", docker_default):
        raise HTTPException(403, "Docker container spawning is disabled by the administrator.")
    # Rate limiting & agent count check
    if AUTH_ENABLED and user:
        # An admin acting for this user spends their own request / spawn budget;
        # the agent-count cap below stays the owner's.
        _limited = getattr(request.state, "acting_admin", None) or user
        check_api_rate_limit(_limited)
        check_spawn_rate_limit(_limited)
        # Count existing containers for this user
        client_tmp = get_docker()
        user_id = user["id"]
        owned = [c for c in _deck_containers(all=True, client=client_tmp)
                 if (c.labels or {}).get(OWNER_LABEL, "") == user_id]
        await check_agent_count_limit(user, len(owned))

    client = get_docker()
    slug = _slug(config.name)
    # Before a stopped container of that name goes: the folder may be a
    # removed agent's (process or container) that isn't the caller's.
    _refuse_other_owners_folder(slug, owner_id)

    # Ensure port is available; find a free one if not. A stopped or sleeping
    # agent's port is free on the host but still its own: it comes back on it.
    if config.web_enabled and (config.web_port <= 0 or not _is_port_available(config.web_port)
                               or _port_reserved(config.web_port, slug)):
        config.web_port = _find_available_port(config.web_port if config.web_port > 0 else 24080)

    # Auto-generate auth token if none provided — prevents unauthenticated
    # direct access to agent ports bypassing Flight Deck. A caller-chosen one
    # that duplicates another agent's (its FD identity) is replaced. Minted
    # with web disabled too (a token in config doesn't turn a web UI on).
    web_auth_replaced = _claim_web_auth(config, slug)

    # Check for name collision
    try:
        existing = client.containers.get(slug)
        if not _is_this_decks_container(existing):
            # Container names are host-global; never remove another deck's.
            raise HTTPException(409, f"Container name '{slug}' is used by another Flight Deck on this host.")
        # A name is a directory (DATA_DIR/<slug>): a container that isn't the
        # caller's stays — refuse rather than hand them its data (the process
        # spawn refuses another user's name the same way).
        _refuse_other_owners_container(existing, owner_id, slug)
        if existing.status == "running":
            raise HTTPException(400, f"Container '{slug}' already running. Stop it first or use a different name.")
        # Remove stopped container with same name
        await _remove_replaced_container(existing)
    except docker.errors.NotFound:
        pass

    # Prepare data directory
    agent_dir = DATA_DIR / slug
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "workspace").mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "skills").mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "home-config").mkdir(parents=True, exist_ok=True)
    _record_folder_owner(slug, owner_id)

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

    # New agents deploy in eco mode by default.
    _write_eco_flag_on_spawn(agent_dir)
    # …and know who they work for (the owner profile — see tenant_profile).
    await tenant_profile.write_on_spawn(agent_dir, owner_id, "docker")
    # A new container is a new agent: no other agent's shared context (packs).
    await context_packs.remove_files_locked(agent_dir, "docker")

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

    # Pass owner ID (resolved up front by _resolve_spawn_owner) so child agents
    # can propagate ownership when spawning
    if owner_id:
        environment["FD_OWNER_ID"] = owner_id
        # A spawned worker belongs to THIS run's owner, so it must write its VFS
        # files under the owner's root. `vfs_user()` ranks CLAW_VFS_USER above
        # FD_OWNER_ID, and a child inherits the FD server's whole environment — so
        # a global CLAW_VFS_USER (a single-user leftover in the server's .env)
        # would silently funnel EVERY user's run into that one account. Pin it to
        # the run owner so the inherited global can never misdirect it.
        environment["CLAW_VFS_USER"] = owner_id

    # Slug for port-fallback callbacks (Docker path)
    environment["FD_AGENT_SLUG"] = slug

    # Labels for tracking
    labels = {
        CONTAINER_LABEL: "true",
        DECK_LABEL: _deck_id(),
        OWNER_LABEL: owner_id,
        "flight-deck.agent-name": config.name or slug,
        "flight-deck.description": config.description or "",
        "flight-deck.image": config.image,
        "flight-deck.web-port": str(config.web_port) if config.web_enabled else "",
        "flight-deck.web-auth": config.web_auth_token or "",
        # Deep-memory grid config, mirrored from the process registry so a
        # containerised agent resolves the same tags/recall via its Docker labels.
        "flight-deck.grid-tags": json.dumps(list(config.grid_memory_tags or [])),
        "flight-deck.grid-recall": config.grid_recall_mode or "",
        # A fresh agent: members of an earlier agent with this slug don't carry over.
        agent_sharing.INSTANCE_LABEL: agent_sharing.new_instance_id(),
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
            await db.log_usage(user["id"], "agent_spawn", json.dumps({
                "agent": slug, "type": "container", "image": config.image, **_acting_admin_detail(request)}))
        if config.fleet_instructions:
            await _set_owner_agent_instructions(
                owner_id, "docker", container.short_id, config.fleet_instructions)
        # Notify other agents about the new peer (scoped to same owner)
        if config.web_enabled:
            _schedule_fleet_notify(config.name or slug, config.web_port, owner_id=owner_id)
        if web_auth_replaced:
            return ContainerActionResult(
                ok=True, container_id=container.short_id,
                message=f"Agent '{slug}' spawned{port_info}{_WEB_AUTH_REPLACED_NOTE}",
                web_auth=config.web_auth_token)
        return ContainerActionResult(ok=True, container_id=container.short_id, message=f"Agent '{slug}' spawned{port_info}")
    except docker.errors.ImageNotFound:
        raise HTTPException(404, f"Docker image '{config.image}' not found. Pull it first.")
    except docker.errors.APIError as exc:
        raise HTTPException(500, f"Docker error: {exc.explanation or str(exc)}")


@app.post("/fd/containers/{container_id}/stop", response_model=ContainerActionResult)
async def stop_container(container_id: str, request: Request, user: dict | None = _agent_manager_dep):
    import asyncio
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    if c.status != "running":
        return ContainerActionResult(ok=True, container_id=c.short_id, message="Already stopped")
    await asyncio.get_event_loop().run_in_executor(None, lambda: c.stop(timeout=5))
    return ContainerActionResult(ok=True, container_id=c.short_id, message="Stopped")


@app.post("/fd/containers/{container_id}/start", response_model=ContainerActionResult)
async def start_container(container_id: str, request: Request, user: dict | None = _agent_manager_dep):
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
async def restart_container(container_id: str, request: Request, user: dict | None = _agent_manager_dep):
    import asyncio
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    await asyncio.get_event_loop().run_in_executor(None, lambda: c.restart(timeout=5))
    return ContainerActionResult(ok=True, container_id=c.short_id, message="Restarted")


@app.delete("/fd/containers/{container_id}", response_model=ContainerActionResult)
async def remove_container(container_id: str, force: bool = False, purge: bool = False, request: Request = None, user: dict | None = _agent_manager_dep):
    import asyncio
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    name = c.name
    owner_id, short_id = str((c.labels or {}).get(OWNER_LABEL, "") or ""), c.short_id
    agent_ref = _container_agent_ref(c)
    if not purge:
        # The folder stays; once the container is gone only this says whose it is.
        _record_folder_owner(name, owner_id)
    await asyncio.get_event_loop().run_in_executor(None, lambda: c.remove(force=force))
    await _set_owner_agent_instructions(owner_id, "docker", short_id, "")
    await _forget_shared_agent(agent_ref, owner_id)
    await botport_forget_share("docker", _slug((c.labels or {}).get("flight-deck.agent-name", name)), owner_id)
    if purge:
        problem = await _purge_agent(name, short_id)
        if problem:
            return ContainerActionResult(ok=False, container_id=container_id, message=f"Removed '{name}', but {problem}")
    return ContainerActionResult(ok=True, container_id=container_id, message=f"Removed '{name}'")


def _remove_agent_dir(slug: str) -> str:
    """Delete ``DATA_DIR/<slug>``, a removed agent's whole folder (config,
    workspace, sessions, memory, datastore, logs). Only ever one folder
    directly under DATA_DIR. Returns "" or why some of it stayed."""
    if not slug or slug in (".", "..") or "/" in slug or "\\" in slug:
        return f"its folder name {slug!r} is not one Flight Deck deletes"
    target = DATA_DIR / slug
    if target.is_symlink():
        target.unlink()  # the link, never what it points at
        return ""
    if not target.exists():
        return ""
    if target.resolve().parent != DATA_DIR.resolve():
        return f"its folder {target} is outside the data directory"

    failed: list[str] = []

    def _retry(func, path, _exc_info):
        # Read-only entries (git objects, a copied package): make them and
        # their folder writable, then try once more.
        try:
            os.chmod(os.path.dirname(path), 0o700)
            os.chmod(path, 0o700)
            func(path)
        except OSError:
            failed.append(path)

    shutil.rmtree(target, onerror=_retry)
    if target.exists():
        return f"{len(failed) or 'some'} item(s) in {target} could not be deleted"
    return ""


async def _purge_agent(slug: str, chat_agent_id: str) -> str:
    """Delete what a removed agent leaves on this deck: its folder, Flight
    Deck's log of the chats with it, its scheduled tasks and what the File and
    Data libraries keep of it — so a later agent of the same name, whoever's it
    is, starts clean. Returns "" or what could not go."""
    import asyncio
    problem = await asyncio.get_event_loop().run_in_executor(None, _remove_agent_dir, slug)
    if problem:
        log.warning("Could not delete a removed agent's folder", agent=slug, problem=problem)
    try:
        if AUTH_ENABLED:
            from captain_claw.flight_deck.auth import get_db

            await get_db().delete_chat_sessions_for_agent(chat_agent_id)
    except Exception as exc:
        log.warning("Could not drop a removed agent's chat log", agent=slug, error=type(exc).__name__)
    try:
        from captain_claw.flight_deck.fd_scheduler import get_store as _scheduler_store

        _scheduler_store().delete_for_agent(slug)
    except Exception as exc:
        log.warning("Could not drop a removed agent's scheduled tasks", agent=slug, error=type(exc).__name__)
    # The libraries key their rows by name and drop a gone agent's only on the
    # next scan; meanwhile a new agent of that name would list them.
    from captain_claw.flight_deck import data_library, file_library

    for library in (file_library, data_library):
        try:
            library.get_store().drop_agent(slug)
        except Exception as exc:
            log.warning("Could not drop a removed agent's library entries", agent=slug,
                        library=library.__name__.rsplit(".", 1)[-1], error=type(exc).__name__)
    return problem


def _refuse_other_owners_container(existing, owner_id: str, slug: str) -> None:
    """409 unless ``existing`` — the container a spawn or clone named ``slug``
    would replace — is ``owner_id``'s: its data dir (sessions, members' private
    chats, memory) would go to the new agent. With auth off there is no tenant
    boundary."""
    if AUTH_ENABLED and owner_id and str((existing.labels or {}).get(OWNER_LABEL, "") or "") != owner_id:
        raise HTTPException(409, f"An agent named '{slug}' already exists. Choose a different name.")


# Who an agent's folder (DATA_DIR/<slug>) belongs to, written into it when
# Flight Deck makes the agent and when it removes one but keeps the folder.
# The folder outlives the agent's registry entry or container, and a new agent
# of the same name — anyone's — would start on its sessions, memory and files.
FOLDER_OWNER_FILE = ".fd-owner"


def _folder_owner(slug: str) -> str:
    """The owner recorded in ``DATA_DIR/<slug>``, or "" when none is."""
    try:
        return (DATA_DIR / slug / FOLDER_OWNER_FILE).read_text().strip()
    except (OSError, UnicodeDecodeError):
        return ""


def _owners_on_record(slug: str) -> set[str]:
    """Every owner Flight Deck has on record for the name: its process
    registry entry's and its container's ("" for one recorded without)."""
    owners: set[str] = set()
    entry = _load_process_registry().get(slug)
    if entry is not None:
        owners.add(str(entry.get("owner") or ""))
    try:
        for c in _deck_containers(all=True):
            if c.name == slug:
                owners.add(str((c.labels or {}).get(OWNER_LABEL, "") or ""))
    except Exception:
        pass  # no Docker: no containers
    return owners


def _holds_files(folder: Path) -> bool:
    """Is there any file in *folder* (one it can't look through counts)?"""
    try:
        return any(p.is_file() for p in folder.rglob("*"))
    except OSError:
        return True


def _refuse_other_owners_folder(slug: str, owner_id: str) -> None:
    """409 unless ``DATA_DIR/<slug>`` — the folder a new agent named ``slug``
    takes over — is free or ``owner_id``'s: the owner recorded in it, else (a
    folder from before owners were recorded) the only owner Flight Deck has on
    record for the name. A folder with no files holds no one's data; one no
    owner can be named for stays as it is. With auth off there is no tenant
    boundary."""
    if not AUTH_ENABLED or not owner_id:
        return
    folder = DATA_DIR / slug
    if not folder.is_dir():
        return
    recorded = _folder_owner(slug)
    if recorded == owner_id:
        return
    if not recorded and (not _holds_files(folder) or _owners_on_record(slug) == {owner_id}):
        return
    raise HTTPException(409, f"An agent named '{slug}' already exists. Choose a different name.")


def _record_folder_owner(slug: str, owner_id: str) -> None:
    """Write ``owner_id`` into ``DATA_DIR/<slug>`` as its owner (best-effort)."""
    folder = DATA_DIR / slug
    if not owner_id or not folder.is_dir():
        return
    try:
        (folder / FOLDER_OWNER_FILE).write_text(owner_id + "\n")
    except OSError as exc:
        log.warning("Could not record an agent folder's owner", agent=slug, error=type(exc).__name__)


async def _remove_replaced_container(existing) -> None:
    """Remove the stopped container a spawn or clone replaces. Its successor is
    a new agent (a new instance id), so the old one's shares go with it."""
    agent_ref = _container_agent_ref(existing)
    owner_id = str((existing.labels or {}).get(OWNER_LABEL, "") or "")
    existing.remove()
    await _forget_shared_agent(agent_ref, owner_id)
    await botport_forget_share(
        "docker", _slug((existing.labels or {}).get("flight-deck.agent-name", existing.name)), owner_id)


async def _forget_shared_agent(agent_ref: str, owner_id: str) -> None:
    """An agent was removed: drop its member grants, close its members' turn
    grants, drop their Google opt-ins for it and close their sockets (4404).
    Best-effort — a removal never fails over it."""
    if not agent_ref:
        return
    try:
        if AUTH_ENABLED and owner_id:
            from captain_claw.flight_deck.auth import get_db

            await get_db().delete_shares_for_resource(
                agent_sharing.AGENT_RESOURCE, agent_ref, owner_id)
    except Exception as exc:
        log.warning("Could not drop a removed agent's shares", error=str(exc))
    try:
        speaker_grants.revoke(agent_ref)
    except Exception as exc:
        log.warning("Could not close a removed agent's member grants", error=type(exc).__name__)
    try:
        if AUTH_ENABLED:
            from captain_claw.flight_deck.auth import get_db

            await speaker_grants.clear_google_optins(get_db(), agent_ref)
    except Exception as exc:
        log.warning("Could not clear a removed agent's Google opt-ins", error=type(exc).__name__)
    try:
        agent_sharing.invalidate_member_cache(agent_ref)
        await agent_sharing.close_member_sockets(agent_ref, code=4404, reason="Agent removed")
    except Exception as exc:
        log.warning("Could not close a removed agent's member sockets", error=str(exc))
    # Its context packs (and alias reservations) and its shared-context files.
    if AUTH_ENABLED:
        try:
            from captain_claw.flight_deck.auth import get_db

            await get_db().delete_context_packs_for_agent(agent_ref)
        except Exception as exc:
            log.warning("Could not drop a removed agent's context packs",
                        error=type(exc).__name__)
        try:
            context_packs.remove_files_for_ref(agent_ref)
        except Exception as exc:
            log.warning("Could not remove a removed agent's shared-context files",
                        error=type(exc).__name__)


class RebuildRequest(BaseModel):
    description: str = ""  # Frontend sends current description override


class CloneRequest(BaseModel):
    new_name: str  # Name for the cloned agent


@app.post("/fd/containers/{container_id}/rebuild", response_model=ContainerActionResult)
async def rebuild_container(container_id: str, request: Request, req: RebuildRequest | None = None, user: dict | None = _agent_manager_dep):
    """Rebuild a container: stop, remove, pull latest image, re-spawn with same config."""
    import platform

    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    old_short_id = c.short_id
    labels = dict(c.labels or {})
    # The rebuilt container is this deck's (a legacy unlabelled one included:
    # _find_container only hands out this deck's or legacy containers).
    labels[DECK_LABEL] = _deck_id()
    # Same agent, same members: keep (or pin) its instance id across the rebuild.
    _instance = agent_sharing.docker_instance_id(c)
    if _instance:
        labels.setdefault(agent_sharing.INSTANCE_LABEL, _instance)

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

    # Collect ports from Docker containers — every deck's (not _deck_containers):
    # host ports are shared by all decks on this host.
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
    clone_owner = str(labels.get(OWNER_LABEL, "") or "")
    _refuse_other_owners_folder(new_slug, clone_owner)

    # Check name collision
    client = get_docker()
    try:
        existing = client.containers.get(new_slug)
        if not _is_this_decks_container(existing):
            # Container names are host-global; never remove another deck's.
            raise HTTPException(409, f"Container name '{new_slug}' is used by another Flight Deck on this host.")
        # The clone is the source owner's: only their own stopped agent may go.
        _refuse_other_owners_container(existing, clone_owner, new_slug)
        if existing.status == "running":
            raise HTTPException(400, f"Container '{new_slug}' already running.")
        await _remove_replaced_container(existing)
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
    _record_folder_owner(new_slug, clone_owner)

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

    # The clone gets its OWN web_auth: that token is an agent's identity to FD
    # (X-Agent-Auth → recorded owner), so two containers sharing one would be
    # indistinguishable. The container reads it at start from the label FD
    # checks, its config files (the clone's own host-side copies, bind-mounted
    # in — nothing inside the image) and possibly its env.
    new_web_auth = secrets.token_urlsafe(32) if web_auth else ""
    if new_web_auth:
        for k, v in environment.items():
            if web_auth in v:
                environment[k] = v.replace(web_auth, new_web_auth)
        for copied in (new_agent_dir / "config.yaml", new_agent_dir / ".env",
                       new_agent_dir / "data" / "home-config" / "config.yaml"):
            if copied.is_file():
                text = copied.read_text()
                if web_auth in text:
                    copied.write_text(text.replace(web_auth, new_web_auth))

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

    # Update labels for the clone (spawned by — so belonging to — this deck)
    labels[DECK_LABEL] = _deck_id()
    labels[agent_sharing.INSTANCE_LABEL] = agent_sharing.new_instance_id()
    labels["flight-deck.agent-name"] = new_name
    labels["flight-deck.description"] = ""
    if new_web_auth:
        labels["flight-deck.web-auth"] = new_web_auth
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

    # The clone works for the source's owner: their profile, not whatever an
    # earlier agent of this slug left in the folder.
    await tenant_profile.write_on_spawn(new_agent_dir, str(labels.get(OWNER_LABEL, "") or ""), "docker")
    # …and no shared context (packs) left by an earlier agent of this slug.
    await context_packs.remove_files_locked(new_agent_dir, "docker")

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
async def container_logs(container_id: str, tail: int = 200, since_ts: float = 0, follow: bool = False, request: Request = None, user: dict | None = _agent_manager_dep):
    """Fetch container logs.

    When *since_ts* > 0 the Docker ``since`` parameter is used so only
    new log lines written after that Unix timestamp are returned
    (incremental fetch).  The response includes ``timestamp`` – the
    current time – for the frontend to pass back on the next poll.
    """
    import time
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    now = time.time()
    if follow:
        def stream():
            for chunk in c.logs(stream=True, follow=True, tail=tail):
                yield chunk
        return StreamingResponse(stream(), media_type="text/plain")
    elif since_ts > 0:
        logs = c.logs(tail=0, since=since_ts, timestamps=False).decode("utf-8", errors="replace")
        return {"logs": logs, "timestamp": now}
    else:
        logs = c.logs(tail=tail).decode("utf-8", errors="replace")
        return {"logs": logs, "timestamp": now}


def _acting_admin_detail(request) -> dict:
    """``{"acting_admin": id}`` when an admin made this request for the user
    (X-FD-Act-As), so the usage log says who really did it."""
    admin_id = getattr(getattr(request, "state", None), "acting_admin_id", "") or ""
    return {"acting_admin": admin_id} if admin_id else {}


# ── Agent instructions ──
#
# An agent's standing instructions live in its OWNER's settings — a map per
# agent kind — and a chat sends them to the agent as `fleet_instructions` on
# connect. Flight Deck is the writer (one entry at a time, under a per-owner
# lock) and the SPA syncs from the agent list, so an entry survives whoever
# wrote it: the owner, an archetype spawn, or an admin acting for the owner.

_AGENT_INSTRUCTIONS_SETTING = {
    "process": "fd:process-fleet-instructions",
    "docker": "fd:container-fleet-instructions",
}
# The owner's browser-side labels for an agent, laid over the registry's.
_PROCESS_LABEL_SETTINGS = ("fd:process-names", "fd:process-descriptions")
_MAX_AGENT_INSTRUCTIONS = 64_000  # characters
_owner_setting_locks: dict[str, asyncio.Lock] = {}


def _owner_setting_lock(owner_id: str) -> asyncio.Lock:
    lock = _owner_setting_locks.get(owner_id)
    if lock is None:
        lock = _owner_setting_locks[owner_id] = asyncio.Lock()
    return lock


async def _read_owner_map(owner_id: str, key: str) -> dict[str, str]:
    """A ``{agent id: text}`` user setting ({} when unset or not a map).
    Read errors propagate: a caller about to write the map back must not
    mistake "couldn't read" for "empty" and wipe every other entry."""
    from captain_claw.flight_deck.auth import get_db

    raw = await get_db().get_setting(owner_id, key)
    try:
        data = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        data = {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


async def _update_owner_map(owner_id: str, key: str, identifier: str, text: str | None) -> None:
    """Set (or, with empty / None, drop) one entry of an owner's map setting."""
    from captain_claw.flight_deck.auth import get_db

    async with _owner_setting_lock(owner_id):
        current = await _read_owner_map(owner_id, key)
        if text:
            if current.get(identifier) == text:
                return
            current[identifier] = text
        elif identifier in current:
            del current[identifier]
        else:
            return
        await get_db().set_settings(owner_id, {key: json.dumps(current)})


async def _owner_agent_instructions(owner_id: str, kind: str) -> dict[str, str]:
    """``{agent id: instructions}`` from ``owner_id``'s settings — {} when there
    are none, no accounts, or the read fails (for display; never written back)."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    if not owner_id or not _fd_auth_enabled():
        return {}
    try:
        return await _read_owner_map(owner_id, _AGENT_INSTRUCTIONS_SETTING[kind])
    except Exception as exc:
        log.warning("Could not read agent instructions", owner=owner_id, error=str(exc))
        return {}


async def _set_owner_agent_instructions(
    owner_id: str, kind: str, identifier: str, text: str, *, strict: bool = False,
) -> None:
    """Record (or, with empty text, drop) ``identifier``'s instructions in
    ``owner_id``'s settings. No-op without accounts (auth off keeps these in the
    browser). Best-effort for spawn / remove, which must not fail over it;
    ``strict`` for an explicit save, which must not report a write that didn't
    happen."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    if not owner_id or not _fd_auth_enabled():
        return
    try:
        await _update_owner_map(
            owner_id, _AGENT_INSTRUCTIONS_SETTING[kind], identifier,
            (text or "")[:_MAX_AGENT_INSTRUCTIONS])
    except Exception as exc:
        if strict:
            raise
        log.warning("Could not store agent instructions", agent=identifier, error=str(exc))


class AgentInstructionsUpdate(BaseModel):
    instructions: str = Field(default="", max_length=_MAX_AGENT_INSTRUCTIONS)


def _owned_agent_key(kind: str, identifier: str, user_id: str) -> str:
    """The id an owner's settings key this agent by, after the ownership check."""
    if kind not in _AGENT_INSTRUCTIONS_SETTING:
        raise HTTPException(400, "kind must be 'docker' or 'process'")
    if kind == "docker":
        return _find_container(identifier, user_id).short_id
    _verify_process_owner(identifier, user_id)
    return identifier


@app.get("/fd/agent-instructions/{kind}/{identifier}")
async def get_agent_instructions(
    kind: str, identifier: str, request: Request,
    user: dict | None = _agent_manager_dep,
):
    """An agent's standing instructions, from its owner's settings."""
    user_id = getattr(request.state, "user_id", "")
    key = _owned_agent_key(kind, identifier, user_id)
    return {"instructions": (await _owner_agent_instructions(user_id, kind)).get(key, "")}


@app.put("/fd/agent-instructions/{kind}/{identifier}")
async def update_agent_instructions(
    kind: str, identifier: str, body: AgentInstructionsUpdate, request: Request,
    user: dict | None = _agent_manager_dep,
):
    """Set an agent's standing instructions in its owner's settings. The owner's
    Flight Deck picks them up from the agent list; they reach the agent the next
    time a chat connects to it."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    if not _fd_auth_enabled():
        raise HTTPException(400, "Without accounts, instructions are kept in the browser")
    user_id = getattr(request.state, "user_id", "")
    key = _owned_agent_key(kind, identifier, user_id)
    try:
        await _set_owner_agent_instructions(
            user_id, kind, key, body.instructions.strip(), strict=True)
    except Exception as exc:
        log.warning("Agent instructions save failed", agent=key, error=str(exc))
        raise HTTPException(500, "Could not save the instructions — nothing was changed")
    return {"ok": True}


# ── Agent config editing ──

class AgentConfigUpdate(BaseModel):
    config_yaml: str | None = None
    env: str | None = None


# An agent on a team tier holds the deck's own model key in its .env
# (`_build_env`, `heal_keyless_agents`), and org keys never reach a browser: a
# non-admin owner reads its secrets as dots and, for a long one, the last four
# characters — as Settings hints the team keys. A save that still carries such
# a placeholder keeps the value behind it.
_SECRET_MASK = "····"
_SECRET_ENV_NAME = re.compile(r"(?:^|_)(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD)$", re.IGNORECASE)
_ENV_ASSIGNMENT = re.compile(r"^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z0-9_.]*)(\s*=\s*)(.*?)(\s*)$")
# A team key shorter than this (a local server's placeholder "x") is masked
# where it is a whole value, never wherever it happens to occur.
_TEAM_KEY_MIN_MATCH = 12


def _mask_secret(value: str) -> str:
    return _SECRET_MASK + value[-4:] if len(value) >= 12 else _SECRET_MASK


def _sees_agent_secrets(request: Request, user: dict | None) -> bool:
    """Whether the caller reads an agent's .env as it is: an admin — also one
    acting for the owner (X-FD-Act-As) — or anyone on a deck without accounts
    (one trusted operator). Any other owner reads its secrets masked."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    if not _fd_auth_enabled() or getattr(request.state, "acting_admin", None):
        return True
    return bool(user) and user.get("role") == "admin"


async def _deck_secrets() -> set[str]:
    """Every model key the deck holds for its team: the org provider and
    endpoint keys behind "@system", and the provider keys in Flight Deck's own
    environment. Best-effort — without them the name rule still masks."""
    keys = {os.environ.get(n) or "" for p in _PROVIDER_KEY_ENV for n in _provider_key_env_names(p)}
    try:
        keys |= await _org_keys()
    except Exception as exc:
        log.warning("Could not read the team keys to mask", error=str(exc))
    return {k.strip() for k in keys if k.strip() and k.strip() != "@system"}


def _holds_team_key(value: str, team_keys: set[str]) -> bool:
    return any(value == k or (len(k) >= _TEAM_KEY_MIN_MATCH and k in value) for k in team_keys)


def _mask_team_keys(text: str, team_keys: set[str]) -> str:
    """``text`` with every (long enough) team key in it masked where it stands."""
    for k in team_keys:
        if len(k) >= _TEAM_KEY_MIN_MATCH and k in text:
            text = text.replace(k, _mask_secret(k))
    return text


def _unmask_team_keys(text: str, current: str, team_keys: set[str]) -> str:
    """`_mask_team_keys` undone for the team keys ``current`` (the file on disk) holds."""
    for k in team_keys:
        if len(k) >= _TEAM_KEY_MIN_MATCH and k in current:
            text = text.replace(_mask_secret(k), k)
    return text


def _env_value(raw: str) -> tuple[str, str]:
    """(quote, value) of a .env value as written: ``"x"`` → ('"', 'x')."""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[0], raw[1:-1]
    return "", raw


def _mask_env_line(line: str, team_keys: set[str]) -> str:
    """One .env line (no line ending) as a non-admin reads it: the value of an
    assignment named like a secret (``*_API_KEY``, ``*_TOKEN``, ``*_SECRET`` …)
    or holding a team key is masked; a team key anywhere else — a comment — is
    masked where it stands."""
    m = _ENV_ASSIGNMENT.match(line)
    if not m:
        return _mask_team_keys(line, team_keys)
    lead, name, eq, raw, trail = m.groups()
    quote, value = _env_value(raw)
    if value and (_SECRET_ENV_NAME.search(name) or _holds_team_key(value, team_keys)):
        return f"{lead}{name}{eq}{quote}{_mask_secret(value)}{quote}{trail}"
    return line


def _dotenv_values(text: str) -> dict[str, str]:
    """``{name: value}`` of a .env text, read as the agent reads its file."""
    values = {}
    for line in text.splitlines():
        m = _ENV_ASSIGNMENT.match(line)
        if m:
            values[m.group(2)] = _env_value(m.group(4))[1]
    return values


def _mask_agent_env(text: str, team_keys: set[str]) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        out.append(_mask_env_line(body, team_keys) + line[len(body):])
    return "".join(out)


def _unmask_agent_env(text: str, current: str, team_keys: set[str]) -> str:
    """``text`` (a .env being saved) with every placeholder `_mask_agent_env`
    wrote swapped back for the value it stands for in ``current`` (the file on
    disk), so a save from a masked read never writes the mask over the key. A
    placeholder with nothing behind it — a name that holds no such value — is
    a ValueError naming it, rather than a mask written as the value."""
    restore: dict[str, str] = {}
    held: dict[str, list[str]] = {}
    for line in current.splitlines():
        masked = _mask_env_line(line, team_keys)
        if masked != line:
            restore.setdefault(masked, line)
        m = _ENV_ASSIGNMENT.match(line)
        if m:
            held.setdefault(m.group(2), []).append(_env_value(m.group(4))[1])
    out = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        end = line[len(body):]
        m = _ENV_ASSIGNMENT.match(body)
        if _SECRET_MASK not in body:
            out.append(line)
        elif body in restore:
            out.append(restore[body] + end)
        elif m and _env_value(m.group(4))[1].startswith(_SECRET_MASK):
            lead, name, eq, raw, trail = m.groups()
            quote, value = _env_value(raw)
            real = next((v for v in held.get(name, []) if _mask_secret(v) == value), None)
            if real is None:
                raise ValueError(name)
            out.append(f"{lead}{name}{eq}{quote}{real}{quote}{trail}{end}")
        else:
            out.append(line)  # a comment that shows a mask — nothing behind it to lose
    return "".join(out)


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
    user: dict | None = _agent_manager_dep,
):
    """Read an agent's config.yaml and .env files — secrets masked unless the
    caller is an admin (`_sees_agent_secrets`)."""
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
    if not _sees_agent_secrets(request, user):
        team_keys = await _deck_secrets()
        config_yaml = _mask_team_keys(config_yaml, team_keys)
        env = _mask_agent_env(env, team_keys)
    return {"config_yaml": config_yaml, "env": env}


@app.put("/fd/agent-config/{kind}/{identifier}")
async def update_agent_config(
    kind: str, identifier: str, body: AgentConfigUpdate, request: Request,
    user: dict | None = _agent_manager_dep,
):
    """Update an agent's config.yaml and/or .env files. Agent must be restarted for changes to take effect.

    A value the read masked (`get_agent_config`) and that comes back as its
    placeholder keeps what is on disk."""
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")
    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)

    config_file = agent_dir / "config.yaml"
    env_file = agent_dir / ".env"
    config_yaml, env = body.config_yaml, body.env
    if _SECRET_MASK in (config_yaml or "") + (env or ""):
        team_keys = await _deck_secrets()
        if config_yaml is not None:
            config_yaml = _unmask_team_keys(
                config_yaml, config_file.read_text() if config_file.is_file() else "", team_keys)
        if env is not None:
            try:
                env = _unmask_agent_env(env, env_file.read_text() if env_file.is_file() else "", team_keys)
            except ValueError as exc:
                raise HTTPException(400, (
                    f"{exc} in .env shows a hidden value (····) that it doesn't hold — type the "
                    "value, or remove the line. Nothing was saved."))
    if not _sees_agent_secrets(request, user) and (config_yaml is not None or env is not None):
        # The endpoint is the owner's to choose; the team's key goes only to the team's.
        from captain_claw.config import Config

        kept = Config._read_dotenv_file(env_file)
        model = (_agent_config_model(agent_dir.name) or ({}, {}))[0]
        if config_yaml is not None:
            try:
                data = yaml.safe_load(config_yaml)
            except yaml.YAMLError:
                data = None
            model = data.get("model") if isinstance(data, dict) and isinstance(data.get("model"), dict) else {}
        blockers = await _team_key_blockers(model, kept if env is None else _dotenv_values(env), kept=kept)
        if blockers:
            raise _team_key_refusal(blockers)
    if env is not None and _confine_agents() and not _owner_is_admin(user_id):
        # A non-admin's agent takes these from Flight Deck at every start
        # (`_pin_confinement`): one set here would never apply — say so.
        from captain_claw.config import Config

        was = Config._read_dotenv_file(env_file)
        pinned = sorted(n for n, v in _dotenv_values(env).items() if _pinned_env_name(n) and v != was.get(n))
        if pinned:
            raise HTTPException(400, (
                f"Not saved: Flight Deck sets {', '.join(pinned)} for this agent — it can load code "
                "or reroute the agent's calls, so it isn't taken from .env. Remove the line."))

    updated = []
    if config_yaml is not None:
        config_file.write_text(config_yaml)
        # Also update home-config copy so it takes precedence on restart
        home_config = agent_dir / "data" / "home-config" / "config.yaml"
        if home_config.parent.is_dir():
            home_config.write_text(config_yaml)
        # Also update home-config-parent/.captain-claw copy
        parent_config = agent_dir / "data" / "home-config-parent" / ".captain-claw" / "config.yaml"
        if parent_config.parent.is_dir():
            parent_config.write_text(config_yaml)
        updated.append("config.yaml")
    if env is not None:
        env_file.write_text(env)
        updated.append(".env")

    return {"ok": True, "updated": updated, "message": "Restart the agent for changes to take effect."}


class AgentModelUpdate(BaseModel):
    # Both empty leaves the model as it is (a vision-only change).
    provider: str = ""
    model: str = ""
    # None keeps the key / endpoint — unless the provider changes: then the old
    # provider's key is dropped and the endpoint is the new provider's default.
    api_key: str | None = None
    base_url: str | None = None
    vision: str | None = None  # auto | on | off; None leaves it as it is


def _agent_config_paths(agent_dir: Path) -> list[Path]:
    """Every copy of an agent's config.yaml: its directory's, the home-config
    one (a container's ~/.captain-claw) and the one a process agent loads from
    its HOME. A change must reach all of them, or the one that wins at start-up
    still holds the old value."""
    return [
        agent_dir / "config.yaml",
        agent_dir / "data" / "home-config" / "config.yaml",
        agent_dir / "data" / "home-config-parent" / ".captain-claw" / "config.yaml",
    ]


def _default_base_url(provider: str) -> str:
    """A provider's endpoint when nobody names one — as spawn writes it."""
    return "http://127.0.0.1:11434" if provider == "ollama" else ""


@app.put("/fd/agent-model/{kind}/{identifier}")
async def update_agent_model(
    kind: str, identifier: str, body: AgentModelUpdate, request: Request,
    user: dict | None = _agent_manager_dep,
):
    """Quick-update an agent's provider, model, and optionally api_key and vision
    in all config locations."""
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")
    changes_model = bool(body.provider.strip() or body.model.strip())
    if not changes_model and body.vision is None:
        raise HTTPException(400, "Nothing to update: give a provider and model, or vision")
    if changes_model and not (body.provider.strip() and body.model.strip()):
        raise HTTPException(400, "Give both a provider and a model")
    if changes_model:
        from captain_claw.llm import SUPPORTED_PROVIDERS, is_supported_provider

        if not is_supported_provider(body.provider):
            raise HTTPException(400, f"Unknown provider '{body.provider.strip()}'. "
                                f"Use one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}")
    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)
    provider, model_name = body.provider.strip(), body.model.strip()

    def change(section: dict) -> None:
        switched = str(section.get("provider") or "") != provider
        section["provider"] = provider
        section["model"] = model_name
        if provider in {"antigravity-cli", "antigravity", "google-subscription"}:
            section.pop("api_key", None)
            section.pop("base_url", None)
            return
        if body.api_key is not None:
            section["api_key"] = body.api_key.strip()
        elif switched:
            section["api_key"] = ""  # never the old provider's key
        if body.base_url is not None:
            section["base_url"] = body.base_url.strip()
        elif switched:
            section["base_url"] = _default_base_url(provider)

    # An owner on their own API keys (byok.py) has no daily limits, so their
    # agents stay on those keys: a provider they brought one for, or a local
    # model — never one that would run on the deck's key.
    own_keys: dict = {}
    if changes_model and kind == "process":
        from captain_claw.flight_deck import byok
        from captain_claw.flight_deck.auth import get_db

        owner = str((_load_process_registry().get(identifier) or {}).get("owner") or "")
        if owner and await byok.active(get_db(), owner):
            own_keys = await byok.read_keys(get_db(), owner)
            if provider not in own_keys and provider not in byok.LOCAL_PROVIDERS:
                raise HTTPException(409, (
                    "You're running on your own API keys, so this agent can use "
                    f"{', '.join(byok.PROVIDERS[p] for p in own_keys) or 'them'} or a local model. "
                    "Add a key for this provider in Settings → Your own AI keys, or turn them off."))
            if byok.chatgpt_only(provider, model_name):
                raise HTTPException(409, f"{model_name} signs in through ChatGPT, not your API key — "
                                         "pick another OpenAI model")

    # A non-admin picks the endpoint, not where the team's key goes: a team
    # key in the agent's .env that the change would send elsewhere becomes the
    # new endpoint's own team key — or, with none, the change is refused.
    rekey: dict[str, str] = {}
    owner_only = changes_model and not _sees_agent_secrets(request, user)
    if owner_only:
        from captain_claw.config import Config

        dotenv = Config._read_dotenv_file(agent_dir / ".env")
        if provider in own_keys:  # …which the owner's own key replaces below
            dotenv[_PROVIDER_KEY_ENV[provider]] = own_keys[provider]
        after = dict((_agent_config_model(agent_dir.name) or ({}, {}))[0])
        change(after)
        where: dict[str, str] = {}
        for name, p, e in await _misrouted_team_keys(after, dotenv):
            where.setdefault(name, _endpoint_label(p, e))
        routes = _agent_key_routes(after, dotenv)
        for name in where:
            fits = {await _team_key(p, e) for p, e in routes if name in _provider_key_env_names(p)}
            rekey[name] = fits.pop() if len(fits) == 1 else ""
        blockers = [f"the team's key in {n} would go to {where[n]}" for n, k in rekey.items() if not k]
        if blockers:
            raise _team_key_refusal(blockers)

    updated_count = 0
    for cfg_path in _agent_config_paths(agent_dir):
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
        if changes_model:
            change(data["model"])
        if body.vision is not None:
            switch = _vision_switch(body.vision)
            if switch is None:
                data["model"].pop("vision", None)
            else:
                data["model"]["vision"] = switch
        cfg_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True))
        updated_count += 1

    # …and the registry, which the agent list reads its model from.
    if changes_model and kind == "process":
        registry = _load_process_registry()
        if identifier in registry:
            registry[identifier]["provider"] = provider
            registry[identifier]["model"] = model_name
            if own_keys:
                if provider in own_keys:
                    _upsert_dotenv_var(agent_dir / ".env", _PROVIDER_KEY_ENV[provider], own_keys[provider])
                    registry[identifier].setdefault("byok_prev", {})
                    registry[identifier]["key_source"] = "byok"
                else:  # a local model: nothing to pay for, nothing to switch back
                    registry[identifier].pop("key_source", None)
                    registry[identifier].pop("byok_prev", None)
            _save_process_registry(registry)
    for name, key in rekey.items():
        _upsert_dotenv_var(agent_dir / ".env", name, key)
    if owner_only and kind == "process" and not own_keys:
        # A model left with no key gets the team's for its endpoint, if it has one.
        await _heal_agent_model_key(identifier)

    return {"ok": True, "updated": updated_count, "message": "Restart the agent for changes to take effect."}


class AgentModeUpdate(BaseModel):
    mode: str = "neutra"


@app.put("/fd/agent-mode/{kind}/{identifier}")
async def update_agent_mode(
    kind: str, identifier: str, body: AgentModeUpdate, request: Request,
    user: dict | None = _agent_manager_dep,
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

    # Write to all potential home config locations — the agent's own
    # ~/.captain-claw among them. Neutra is written too, not left out: with no
    # file the agent falls back to its config's default mode, which an
    # archetype may have set to another one.
    cc_dirs = [agent_dir / "data" / subdir / ".captain-claw" for subdir in ("home-config", "home-config-parent")]
    if kind == "docker":
        cc_dirs.append(_agent_home_cc(agent_dir, kind))
    for cc_dir in cc_dirs:
        if cc_dir.is_dir():
            (cc_dir / "cognitive_mode.txt").write_text(mode_name, encoding="utf-8")
    # The configs' default mode too, in every copy — the one the agent falls
    # back on without a file, which spawn wrote from the archetype.
    for cfg_path in _agent_config_paths(agent_dir):
        if not cfg_path.is_file():
            continue
        try:
            data = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        section = data.get("cognitive_mode")
        data["cognitive_mode"] = {**(section if isinstance(section, dict) else {}), "default_mode": mode_name}
        cfg_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True))

    return {"ok": True, "mode": mode_name, "message": "Mode updated. Takes effect on the agent's next response."}


class AgentReasoningUpdate(BaseModel):
    # A level, "auto" (the model's own), or None to remove the setting. Only
    # the keys sent change.
    default: str | None = None
    channels: dict[str, str | None] | None = None
    clear_default: bool = False


def _agent_reasoning_dirs(agent_dir: Path, kind: str) -> list[Path]:
    """Every ~/.captain-claw the agent may run with (as for the mode file)."""
    dirs = [agent_dir / "data" / subdir / ".captain-claw" for subdir in ("home-config", "home-config-parent")]
    own = _agent_home_cc(agent_dir, kind)
    return dirs + ([own] if own not in dirs else [])


def _read_agent_reasoning(agent_dir: Path, kind: str) -> dict:
    from captain_claw.llm import reasoning

    return reasoning.load_settings(_agent_home_cc(agent_dir, kind) / "reasoning_effort.json")


def _agent_model_triplet(agent_dir: Path, kind: str) -> tuple[str, str, str]:
    """(provider, model, base_url) from the config the agent starts with."""
    for cfg in (_agent_home_cc(agent_dir, kind) / "config.yaml", agent_dir / "config.yaml"):
        try:
            data = yaml.safe_load(cfg.read_text()) or {}
        except Exception:
            continue
        section = data.get("model") if isinstance(data, dict) else None
        if isinstance(section, dict):
            return (str(section.get("provider") or ""), str(section.get("model") or ""),
                    str(section.get("base_url") or ""))
    return "", "", ""


@app.get("/fd/agent-reasoning/{kind}/{identifier}")
async def get_agent_reasoning(
    kind: str, identifier: str, request: Request,
    user: dict | None = _agent_manager_dep,
):
    """An agent's reasoning effort for every source but the web chat: its
    settings, the levels its configured model offers, and the sources."""
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")
    from captain_claw import turn_source
    from captain_claw.llm import reasoning

    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)
    provider, model, base_url = _agent_model_triplet(agent_dir, kind)
    profile = reasoning.profile_for_config(provider, model, base_url or None)
    return {
        "settings": _read_agent_reasoning(agent_dir, kind),
        "profile": profile.to_wire() if profile else None,
        "model": {"provider": provider, "model": model},
        "catalog": turn_source.settings_catalog(),
    }


@app.put("/fd/agent-reasoning/{kind}/{identifier}")
async def update_agent_reasoning(
    kind: str, identifier: str, body: AgentReasoningUpdate, request: Request,
    user: dict | None = _agent_manager_dep,
):
    """Change an agent's default reasoning effort and/or per-source levels.
    The agent reads the file afresh: it applies from its next message."""
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")
    from captain_claw import turn_source
    from captain_claw.llm import reasoning

    def _value(raw: str | None, what: str) -> str:
        text = str(raw or "").strip().lower()
        if text == reasoning.AUTO or reasoning.normalize_level(text):
            return reasoning.AUTO if text == reasoning.AUTO else str(reasoning.normalize_level(text))
        raise HTTPException(400, f"Unknown reasoning level for {what}: {raw!r}")

    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)
    current = _read_agent_reasoning(agent_dir, kind)
    channels = dict(current.get("channels") or {})
    if body.clear_default:
        current.pop("default", None)
    elif body.default is not None:
        current["default"] = _value(body.default, "the default")
    for key, raw in (body.channels or {}).items():
        source = turn_source.normalize(key)
        if not source or source == "web":
            raise HTTPException(400, f"Unknown source: {key!r}")
        if raw is None:
            channels.pop(source, None)
        else:
            channels[source] = _value(raw, source)
    current["channels"] = channels
    cleaned = reasoning.clean_settings(current)
    text = json.dumps(cleaned, indent=2, sort_keys=True)
    written = 0
    for cc_dir in _agent_reasoning_dirs(agent_dir, kind):
        if cc_dir.is_dir():
            (cc_dir / "reasoning_effort.json").write_text(text, encoding="utf-8")
            written += 1
    if not written:
        raise HTTPException(409, "The agent has no home folder yet — start it once first")
    return {"ok": True, "settings": cleaned}


def _agent_home_cc(agent_dir: Path, kind: str) -> Path:
    """The agent's own ~/.captain-claw: a process agent runs with
    HOME=data/home-config-parent; a container bind-mounts data/home-config."""
    if kind == "docker":
        return agent_dir / "data" / "home-config"
    return agent_dir / "data" / "home-config-parent" / ".captain-claw"


@app.get("/fd/agent-mode/{kind}/{identifier}")
async def get_agent_mode(
    kind: str, identifier: str, request: Request,
    user: dict | None = _agent_manager_dep,
):
    """An agent's cognitive mode as the agent resolves it: its
    cognitive_mode.txt, else its config's default mode, else neutra."""
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")
    from captain_claw.cognitive_mode import MODES

    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)
    cc_dir = _agent_home_cc(agent_dir, kind)
    try:
        text = (cc_dir / "cognitive_mode.txt").read_text(encoding="utf-8").strip().lower()
        if text in MODES:
            return {"mode": text}
    except OSError:
        pass
    for cfg in (cc_dir / "config.yaml", agent_dir / "config.yaml"):
        try:
            data = yaml.safe_load(cfg.read_text()) or {}
        except Exception:
            continue
        section = data.get("cognitive_mode") if isinstance(data, dict) else None
        mode = str((section or {}).get("default_mode") or "").strip().lower() if isinstance(section, dict) else ""
        if mode in MODES:
            return {"mode": mode}
        break
    return {"mode": "neutra"}


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


def _write_eco_flag_on_spawn(agent_dir: Path, enabled: bool = True) -> None:
    """Write the ``eco_mode.txt`` flag at spawn time so a new agent deploys
    in eco mode (micro instructions + lazy tools) from its very first response.

    Writes to every location an agent might read it from, covering both runtimes:

      * Process agents run with ``HOME=data/home-config-parent`` and read
        ``~/.captain-claw/eco_mode.txt`` → ``home-config-parent/.captain-claw/``.
      * Docker agents bind-mount ``data/home-config`` as ``/home/claw/.captain-claw``,
        so the agent reads ``home-config/eco_mode.txt`` directly.

    The nested ``.captain-claw/eco_mode.txt`` files also match what the
    ``/fd/agent-eco-mode`` GET endpoint reads, so the Flight Deck UI shows
    the toggle as ON immediately after spawn.
    """
    if not enabled:
        return
    # Agent-read locations + FD UI state locations.
    targets = [
        agent_dir / "data" / "home-config-parent" / ".captain-claw" / "eco_mode.txt",
        agent_dir / "data" / "home-config" / ".captain-claw" / "eco_mode.txt",
        # Docker: home-config IS the container's ~/.captain-claw.
        agent_dir / "data" / "home-config" / "eco_mode.txt",
    ]
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("on", encoding="utf-8")
        except Exception as exc:  # best-effort; never fail a spawn over this
            log.warning("Failed to write eco flag on spawn", path=str(target), error=str(exc))


class AgentNanoModeUpdate(BaseModel):
    enabled: bool = False


@app.put("/fd/agent-nano-mode/{kind}/{identifier}")
async def update_agent_nano_mode(
    kind: str, identifier: str, body: AgentNanoModeUpdate, request: Request,
    user: dict | None = _required_user_dep,
):
    """Toggle nano (barebone) mode at runtime.

    Writes ``nano_mode.txt`` — the agent picks it up on the next system
    prompt build.  Nano implies eco/micro: tool definitions are stripped
    to a barebone allowlist and prompts use nano_<name> templates.
    """
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")

    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)

    for subdir in ("home-config", "home-config-parent"):
        cc_dir = agent_dir / "data" / subdir / ".captain-claw"
        if cc_dir.is_dir():
            nano_file = cc_dir / "nano_mode.txt"
            if body.enabled:
                nano_file.write_text("on", encoding="utf-8")
            else:
                nano_file.unlink(missing_ok=True)

    return {"ok": True, "enabled": body.enabled, "message": "Nano mode updated. Takes effect on the agent's next response."}


@app.get("/fd/agent-nano-mode/{kind}/{identifier}")
async def get_agent_nano_mode(
    kind: str, identifier: str, request: Request,
    user: dict | None = _required_user_dep,
):
    """Read current nano mode state for an agent."""
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")

    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)

    for subdir in ("home-config", "home-config-parent"):
        nano_file = agent_dir / "data" / subdir / ".captain-claw" / "nano_mode.txt"
        if nano_file.is_file():
            return {"enabled": True}

    return {"enabled": False}


class AgentMravModeUpdate(BaseModel):
    enabled: bool = False


@app.put("/fd/agent-mrav-mode/{kind}/{identifier}")
async def update_agent_mrav_mode(
    kind: str, identifier: str, body: AgentMravModeUpdate, request: Request,
    user: dict | None = _required_user_dep,
):
    """Toggle the Mrav micro runtime (8k-capped loop) at runtime.

    Writes ``mrav_mode.txt`` with an explicit "on"/"off" — unlike eco/nano
    (present=on), because a mrav-spawned agent has ``mrav.enabled: true`` in
    its config.yaml and "off" must be able to override it. The agent checks
    the flag on every message, so no restart is needed.
    """
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")

    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)

    for subdir in ("home-config", "home-config-parent"):
        cc_dir = agent_dir / "data" / subdir / ".captain-claw"
        if cc_dir.is_dir():
            (cc_dir / "mrav_mode.txt").write_text(
                "on" if body.enabled else "off", encoding="utf-8"
            )

    return {"ok": True, "enabled": body.enabled,
            "message": "Mrav runtime updated. Takes effect on the agent's next message."}


@app.get("/fd/agent-mrav-mode/{kind}/{identifier}")
async def get_agent_mrav_mode(
    kind: str, identifier: str, request: Request,
    user: dict | None = _required_user_dep,
):
    """Effective Mrav state: the runtime flag if set, else the spawn config."""
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")

    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)

    for subdir in ("home-config", "home-config-parent"):
        flag = agent_dir / "data" / subdir / ".captain-claw" / "mrav_mode.txt"
        if flag.is_file():
            try:
                text = flag.read_text(encoding="utf-8").strip().lower()
            except Exception:
                continue
            if text in ("on", "true", "1", "yes"):
                return {"enabled": True, "source": "flag"}
            if text in ("off", "false", "0", "no"):
                return {"enabled": False, "source": "flag"}

    # No flag → the spawn-time config decides (runtime: "mrav" wrote this).
    try:
        cfg_file = agent_dir / "config.yaml"
        if cfg_file.is_file():
            data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
            enabled = bool(((data.get("mrav") or {}).get("enabled")) is True)
            return {"enabled": enabled, "source": "config"}
    except Exception:
        pass
    return {"enabled": False, "source": "config"}


# ── Browser inference workers (mrav Phase 2) ─────────────────────────
# A browser tab registers over /fd/infer-ws (WebLLM in a WebWorker) and
# serves completion jobs; agents call POST /fd/infer/complete through
# BrowserProvider. The tab never executes tools — it only turns
# (messages, schema) into tokens. docs/mrav-micro-agent-plan.md.


def _authorize_infer_call(request: Request) -> None:
    """Gate the broker for captain-claw agents: loopback OR X-Agent-Secret.

    Same model as /fd/codex/access_token — agents hold FD_AGENT_SHARED_SECRET
    from their spawn env; local processes come in over loopback anyway.
    """
    secret = os.environ.get("FD_AGENT_SHARED_SECRET", "").strip()
    if secret:
        provided = request.headers.get("X-Agent-Secret", "")
        if provided and secrets.compare_digest(provided, secret):
            return
    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "::1", "localhost"):
        return
    raise HTTPException(status_code=401, detail="Unauthorized agent call")


@app.websocket("/fd/infer-ws")
async def infer_worker_ws(ws: WebSocket, token: str = ""):
    """A browser tab registers as an inference worker for its signed-in user."""
    from captain_claw.flight_deck import auth as _auth_mod
    from captain_claw.flight_deck.infer_broker import get_infer_broker

    owner_id = ""
    if AUTH_ENABLED:
        try:
            from captain_claw.flight_deck.auth import decode_access_token
            owner_id = str(decode_access_token(token).get("sub") or "")
        except Exception:
            owner_id = ""
        if not owner_id:
            await ws.close(code=4401)
            return
    else:
        owner_id = str(_auth_mod._LOCAL_USER["id"])

    await ws.accept()
    broker = get_infer_broker()
    worker_id = ""
    try:
        registration = await ws.receive_json()
        if str(registration.get("type") or "") != "register":
            await ws.close(code=4400)
            return
        worker_id = broker.register(owner_id, registration, ws.send_json)
        await ws.send_json({"type": "registered", "worker_id": worker_id})
        while True:
            message = await ws.receive_json()
            broker.handle_message(worker_id, message)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"infer-ws error: {exc}")
    finally:
        if worker_id:
            broker.unregister(worker_id)


class InferCompleteBody(BaseModel):
    messages: list[dict] = Field(default_factory=list)
    response_schema: dict | None = None
    max_tokens: int = 1024
    temperature: float = 0.2
    session_key: str = ""
    owner_id: str = ""  # optional explicit owner (in-process callers)
    # Optional job timeout (seconds) for a long single call; None = broker default.
    timeout: float | None = Field(default=None, ge=1.0, le=3600.0)


@app.post("/fd/infer/complete")
async def infer_complete(body: InferCompleteBody, request: Request):
    """Run one completion on the calling owner's browser inference worker."""
    from captain_claw.flight_deck.infer_broker import (
        InferJobError, NoWorkerError, get_infer_broker,
    )

    _authorize_infer_call(request)
    broker = get_infer_broker()

    # Owner resolution: explicit body → agent slug → the only owner with
    # workers → the synthetic local user (auth-disabled desktop).
    owner = (body.owner_id or "").strip()
    if not owner:
        slug = request.headers.get("X-Agent-Slug", "").strip()
        if slug:
            entry = _load_process_registry().get(slug) or {}
            owner = str(entry.get("owner") or "")
    if not owner:
        owners = broker.owner_ids()
        if len(owners) == 1:
            owner = owners[0]
    if not owner and not AUTH_ENABLED:
        from captain_claw.flight_deck import auth as _auth_mod
        owner = str(_auth_mod._LOCAL_USER["id"])
    if not owner:
        raise HTTPException(503, "no inference worker online (owner could not be resolved)")

    if not body.messages:
        raise HTTPException(400, "messages required")
    try:
        return await broker.submit(
            owner,
            body.messages,
            response_schema=body.response_schema,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            session_key=body.session_key,
            timeout=body.timeout,
        )
    except NoWorkerError:
        raise HTTPException(
            503,
            "no browser inference worker online for this owner — open Flight "
            "Deck and enable Local inference, or switch the agent to ollama",
        )
    except InferJobError as exc:
        raise HTTPException(502, f"inference worker failed: {exc}")
    except TimeoutError:
        raise HTTPException(504, "inference worker timed out")


@app.get("/fd/infer/status")
async def infer_status(request: Request, user: dict | None = _required_user_dep):
    """The signed-in user's live inference workers (Local inference panel)."""
    from captain_claw.flight_deck import auth as _auth_mod
    from captain_claw.flight_deck.infer_broker import get_infer_broker

    owner = str(getattr(request.state, "user_id", "") or "")
    if not owner and not AUTH_ENABLED:
        owner = str(_auth_mod._LOCAL_USER["id"])
    return {"workers": get_infer_broker().status(owner)}


@app.get("/fd/cognitive-modes")
async def list_cognitive_modes():
    """Return all available cognitive modes for UI dropdowns."""
    from captain_claw.cognitive_mode import list_modes, mode_to_dict
    return {"modes": [mode_to_dict(m) for m in list_modes()]}


@app.get("/fd/containers/{container_id}")
async def get_container(container_id: str, request: Request, user: dict | None = _agent_manager_dep):
    c = _find_container(container_id, getattr(request.state, "user_id", ""))
    info = _container_info(c)
    labels = dict(c.labels or {})
    env = c.attrs.get("Config", {}).get("Env", [])
    if not origin_guard.may_expose_agent_secrets(request.headers):
        info.web_auth = ""
        labels.pop("flight-deck.web-auth", None)
        env = []  # provider keys live here
    elif getattr(request.state, "acting_admin_id", ""):  # see list_processes
        info.web_auth = ""
        labels.pop("flight-deck.web-auth", None)
    # Add extra details
    return {
        **info.model_dump(),
        "labels": labels,
        "env": env,
        "mounts": [
            {"source": m.get("Source", ""), "destination": m.get("Destination", ""), "mode": m.get("Mode", "")}
            for m in c.attrs.get("Mounts", [])
        ],
    }


def _agent_ws_url(host: str, port: int, auth: str = "", lane: str = "") -> str:
    """The agent /ws URL for a proxied chat connection.

    `lane` rides along untouched — the agent owns lane normalization, and an
    omitted lane means A, the agent's main context.
    """
    from urllib.parse import quote

    query = []
    if auth:
        query.append(f"token={quote(str(auth), safe='')}")
    if lane:
        query.append(f"lane={quote(str(lane), safe='')}")
    return f"ws://{host}:{port}/ws" + (("?" + "&".join(query)) if query else "")


# How long the chat proxy holds a client's socket open for an agent that isn't
# answering yet (Flight Deck just restarted it) before reporting it unreachable.
_AGENT_WS_WAIT_S = float(os.environ.get("FD_AGENT_WS_WAIT_S", "30"))


def _process_chat_target(agent: str) -> dict | None:
    """Registry entry of the process agent the SPA calls ``proc-<slug>``."""
    if not agent.startswith("proc-"):
        return None
    return _load_process_registry().get(agent[len("proc-"):])


def _is_wake_frame(text: str) -> bool:
    """``{"type": "fd_wake"}``: the chat asks Flight Deck to wake a sleeping agent."""
    if '"fd_wake"' not in text:
        return False
    try:
        return json.loads(text).get("type") == "fd_wake"
    except (ValueError, AttributeError):
        return False


def _agent_ws_retryable(exc: BaseException) -> bool:
    """A connect failure that waiting can fix: nothing listening yet, a
    handshake cut short, a 5xx while the agent boots. A 4xx is a real answer."""
    from websockets.exceptions import InvalidHandshake, InvalidStatus

    if isinstance(exc, InvalidStatus):
        return exc.response.status_code >= 500
    return isinstance(exc, (OSError, TimeoutError, InvalidHandshake))


@app.websocket("/fd/agent-ws/{host}/{port}")
async def agent_ws_proxy(ws: WebSocket, host: str, port: int, token: str = "", lane: str = "",
                         fd_token: str = "", agent: str = ""):
    """Proxy WebSocket to a CC agent — avoids browser CORS restrictions.

    `lane` selects a parallel context on the agent (docs/queue-lanes-plan.md).
    It is forwarded verbatim; the agent normalizes it, and an absent or
    unknown lane resolves to A — which IS the agent's main context, so every
    caller that never heard of lanes is unaffected.

    `fd_token` carries the caller's Flight Deck JWT. When auth is enabled the
    socket is refused unless the JWT is valid AND the caller owns the target
    agent (admins may reach any). HTTP middleware can't guard WebSockets, so
    the ownership check lives here.

    `agent` is the SPA's chat id for the agent (``proc-<slug>``). Flight Deck
    restarts its process agents when it restarts, and one whose port is still
    taken comes back on another one and announces it. The proxy follows the
    registry to the agent's current port; the port in the path is only the
    fallback, for agents the registry doesn't know. While the agent isn't
    answering, the proxy holds the socket for up to FD_AGENT_WS_WAIT_S seconds
    and tells the client once (``fd_agent_waiting``). If the agent comes back
    on a new port, the client hears it (``fd_agent_endpoint``).

    An agent put to sleep for being idle (agent_idle) is woken by the client's
    first frame — a message, or ``fd_wake`` — not by the connect itself, so a
    chat left open doesn't keep it awake: the client hears ``fd_agent_asleep``.
    """
    import websockets

    path_port = port
    entry = _process_chat_target(agent)
    if entry and entry.get("web_port"):
        port = int(entry["web_port"])
    else:
        entry = None

    # Ownership guard (HTTP middleware does not run for WebSockets).
    speaker_id = ""  # whose daily allowance this socket's messages count against
    if AUTH_ENABLED:
        payload = None
        if fd_token:
            try:
                payload = decode_access_token(fd_token)
            except HTTPException:
                payload = None
        if not payload:
            # Accept first, so the browser sees 4001 rather than a bare failed
            # handshake: its token is expired, or was signed before Flight Deck
            # restarted (without FD_JWT_SECRET every start re-keys). It then
            # refreshes the token and reopens instead of retrying it forever.
            await ws.accept()
            await ws.close(code=4001, reason="Missing or invalid token")
            return
        if payload.get("role", "user") != "admin":
            owner = str(entry.get("owner") or "") if entry else _resolve_agent_owner(port)
            if owner and owner != payload.get("sub", ""):
                await ws.close(code=4403, reason="This agent belongs to another user")
                return
        speaker_id = str(payload.get("sub") or "")

    await ws.accept()
    # Auto-resolve auth token if the caller didn't provide one — only for the
    # agent's own address (a process agent's is loopback), never a host the
    # caller named.
    own_host = _norm_agent_host(host) in _AGENT_LOOPBACK_HOSTS
    auth = (token or (str(entry.get("web_auth") or "") if entry and own_host else "")
            or _resolve_agent_auth(port, host=host))

    key = agent_ws_quiet.target_key(host, path_port, agent)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _AGENT_WS_WAIT_S
    # Read the client while waiting for the agent: to notice it leave, and to
    # keep whatever it sends meanwhile for the agent.
    from_client = asyncio.create_task(ws.receive())
    held: list[str] = []
    told_waiting = False
    agent_ws = None
    slug = agent[len("proc-"):] if entry is not None else ""

    async def hold_while_asleep() -> bool:
        """The agent is asleep (agent_idle): keep the socket without waking it
        until the client says something — a message typed, or ``fd_wake``
        (sent when the chat is opened). A chat that only reconnected leaves it
        asleep. False when the socket is done (the client left, or the agent
        couldn't be woken)."""
        nonlocal from_client, entry, port, deadline
        # A frame the client sent while the proxy waited asks for it already.
        if not held:
            try:
                await ws.send_text(json.dumps({"type": "fd_agent_asleep"}))
            except Exception:
                from_client.cancel()
                return False
        while not held:
            try:
                msg = await from_client
            except Exception:
                return False
            if msg.get("type") == "websocket.disconnect":
                return False
            from_client = asyncio.create_task(ws.receive())
            text = msg.get("text")
            if text is None:
                continue
            if not _is_wake_frame(text):
                held.append(text)
            break
        # The word that woke it waits in `held` for the agent's link.
        if not await agent_idle.wake(slug) and agent_idle.is_asleep(slug):
            # It didn't start: said once, as for an agent that stays away.
            from_client.cancel()
            try:
                await ws.send_text(json.dumps({"type": "error", "code": "agent_unreachable",
                                               "message": "Couldn't wake the agent"}))
                await ws.close()
            except Exception:
                pass
            return False
        fresh = _process_chat_target(agent)
        if fresh and fresh.get("web_port"):
            entry, port = fresh, int(fresh["web_port"])
        deadline = loop.time() + _AGENT_WS_WAIT_S
        return True

    while agent_ws is None:
        if slug and agent_idle.is_asleep(slug):
            if not await hold_while_asleep():
                return
        try:
            # ping_interval/ping_timeout keep the upstream link alive through any
            # intermediaries and surface dead peers fast — without these the proxy
            # silently drops after a few minutes of idle, forcing the user to
            # re-click the chat button in FD.
            agent_ws = await websockets.connect(
                _agent_ws_url(host, port, auth, lane),
                max_size=4 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=10,
                open_timeout=5,
            )
            break
        except Exception as exc:
            if agent_ws_quiet.mark_down(key):
                log.info("agent-ws: can't reach agent — further attempts aren't logged until it is back",
                         agent=agent or f"{host}:{path_port}", port=port, error=str(exc)[:200])
            stopped = bool(entry and entry.get("stopped"))
            if stopped or not _agent_ws_retryable(exc) or loop.time() >= deadline:
                from_client.cancel()
                try:
                    await ws.send_text(json.dumps({"type": "error", "code": "agent_unreachable",
                                                   "message": f"Connection failed: {exc}"}))
                    await ws.close()
                except Exception:
                    pass
                return
            if not told_waiting:
                told_waiting = True
                try:
                    await ws.send_text(json.dumps({"type": "fd_agent_waiting", "port": port}))
                except Exception:
                    from_client.cancel()
                    return
        done, _ = await asyncio.wait({from_client}, timeout=1.0)
        if from_client in done:
            try:
                msg = from_client.result()
            except Exception:
                return
            if msg.get("type") == "websocket.disconnect":
                return
            if msg.get("text") is not None and not _is_wake_frame(msg["text"]):
                held.append(msg["text"])
            from_client = asyncio.create_task(ws.receive())
        # The agent may have announced a new port meanwhile.
        if entry is not None:
            fresh = _process_chat_target(agent)
            if fresh and fresh.get("web_port"):
                entry, port = fresh, int(fresh["web_port"])

    down_for = agent_ws_quiet.mark_up(key)
    if down_for is not None:
        log.info("agent-ws: agent is back", agent=agent or f"{host}:{path_port}", port=port,
                 down_s=f"{down_for:.1f}")

    async with agent_ws:
        if port != path_port:
            try:
                await ws.send_text(json.dumps({"type": "fd_agent_endpoint", "port": port}))
            except Exception:
                pass

        async def forward(data: str) -> None:
            if _is_wake_frame(data):
                return  # Flight Deck's own (hold_while_asleep): not the agent's
            agent_idle.touch(slug)
            # A chat message counts against the speaker's plan allowance for
            # today; over it, the turn is refused and the client told why.
            ok, reply = await usage_quota.screen_chat_frame(speaker_id, data)
            if reply is not None:
                await ws.send_text(json.dumps(reply))
            if ok:
                await agent_ws.send(data)

        async def client_to_agent():
            nonlocal from_client
            try:
                for data in held:
                    await forward(data)
                while True:
                    msg = await from_client
                    if msg.get("type") == "websocket.disconnect":
                        return
                    if msg.get("text") is not None:
                        await forward(msg["text"])
                    from_client = asyncio.create_task(ws.receive())
            except Exception:
                pass

        async def agent_to_client():
            try:
                async for msg in agent_ws:
                    agent_idle.touch(slug)
                    await ws.send_text(msg if isinstance(msg, str) else msg.decode())
            except Exception:
                pass

        done, pending = await asyncio.wait(
            [asyncio.create_task(client_to_agent()), asyncio.create_task(agent_to_client())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        from_client.cancel()


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
    slug: str = ""  # FD_AGENT_SLUG — what the agent identifies itself as in X-Agent-Slug
    kind: str  # docker | process | local | remote (another deck's, via BotPort)
    host: str = "localhost"
    port: int
    status: str
    description: str = ""
    deck: str = ""  # remote agents: the Flight Deck that runs it
    deck_label: str = ""  # ...and what the hub's operator calls that deck


def _fleet_snapshot() -> list[dict[str, Any]]:
    """Every agent this deck runs (docker + process), with its owner — this
    deck's only (another deck's containers on this host are not peers).
    Blocking (Docker API): call from a thread in async code."""
    fleet: list[dict[str, Any]] = []

    # Docker containers
    try:
        client = get_docker()
        for c in _deck_containers(all=True, client=client):
            labels = c.labels or {}
            wp = labels.get("flight-deck.web-port", "")
            _agent_name = labels.get("flight-deck.agent-name", c.name)
            fleet.append({
                "name": _agent_name,
                "slug": _slug(_agent_name),
                "kind": "docker",
                "host": "localhost",
                "port": int(wp) if wp else 0,
                "status": c.status,
                "description": labels.get("flight-deck.description", ""),
                "owner": labels.get(OWNER_LABEL, ""),
            })
    except Exception:
        pass

    # Process agents
    registry = _load_process_registry()
    for slug, entry in registry.items():
        alive = _process_is_alive(slug)
        fleet.append({
            "name": entry.get("name", slug),
            "slug": slug,
            "kind": "process",
            "host": "localhost",
            "port": entry.get("web_port", 0),
            # asleep: stopped for being idle (agent_idle) — a consult wakes it.
            "status": "running" if alive else "asleep" if agent_idle.is_asleep(slug, entry) else "stopped",
            "description": entry.get("description", ""),
            "owner": entry.get("owner", ""),
        })
    return fleet


@app.get("/fd/fleet", response_model=list[FleetAgent])
async def get_fleet(request: Request, include_remote: bool = False, user: dict | None = _optional_user_dep):
    """Return all running/known agents across docker, process, and local stores
    — this deck's only (another deck's containers on this host are not peers).

    ``include_remote=1`` adds the agents other Flight Decks expose over the
    BotPort link (kind ``remote``, named ``Agent@deck``, port 0)."""
    user_id = getattr(request.state, "user_id", "")
    fleet: list[FleetAgent] = []
    for a in _fleet_snapshot():
        if AUTH_ENABLED and user_id and a.get("owner", "") != user_id:
            continue
        fleet.append(FleetAgent(**{k: v for k, v in a.items() if k != "owner"}))

    # The federation roster only for a caller FD can name: a signed-in user,
    # or one of its own agents (transport guard + the agent's X-Agent-Auth).
    if include_remote is True and (
        (isinstance(user, dict) and user.get("id"))
        or (_agent_caller_ok(request)
            and _agent_record_by_auth(request.headers.get("X-Agent-Auth", "").strip()) is not None)
    ):
        from captain_claw.flight_deck.botport_link import get_link

        for r in get_link().remote_agents():
            fleet.append(FleetAgent(
                name=r["name"], slug=r.get("slug", ""), kind="remote", host="", port=0,
                status="running", description=r.get("description", ""), deck=r["deck"],
                deck_label=r.get("deck_label", ""),
            ))
    return fleet


# Where this deck reaches every agent it runs: process agents listen on
# loopback, containers use host networking or publish their web port (what
# /fd/fleet reports as the host).
_AGENT_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _norm_agent_host(host: str) -> str:
    # No trailing-dot folding: "localhost." is a DNS name the connection would
    # resolve, not loopback.
    return str(host or "").strip().strip("[]").lower()


def _container_hosts(c) -> set[str]:
    """The names a container answers to besides loopback — its name, hostname,
    network aliases and IPs — for a Flight Deck that shares its Docker network."""
    attrs = c.attrs or {}
    names = {c.name, (attrs.get("Config") or {}).get("Hostname", "")}
    for net in ((attrs.get("NetworkSettings") or {}).get("Networks") or {}).values():
        net = net or {}
        names.update(net.get("Aliases") or [])
        names.add(net.get("IPAddress") or "")
    return {_norm_agent_host(n) for n in names if n}


def _resolve_agent_auth(port: int, host: str | None = None) -> str:
    """Look up the auth token for an agent by its web port from Docker labels or process registry.

    ``host`` is where the caller will send the token (a proxy route's
    ``{host}``). The lookup is by port only, so with a host the token comes
    back only when that host is this deck's own address for the agent —
    loopback, or one of the container's names (`_container_hosts`) — else "".
    Internal callers that always connect to loopback pass no host.
    """
    want = None if host is None else _norm_agent_host(host)
    loopback = want is None or want in _AGENT_LOOPBACK_HOSTS
    # Check Docker containers
    try:
        client = get_docker()
        for c in _deck_containers(client=client):
            labels = c.labels or {}
            wp = labels.get("flight-deck.web-port", "")
            if wp and int(wp) == port:
                if not loopback and want not in _container_hosts(c):
                    return ""
                return labels.get("flight-deck.web-auth", "")
    except Exception:
        pass

    if not loopback:
        return ""
    # Check process registry. Multiple (stale) entries can share a web_port, so
    # prefer the one whose process is actually alive before falling back to any.
    registry = _load_process_registry()
    matches = [(slug, e) for slug, e in registry.items() if e.get("web_port") == port]
    for slug, entry in matches:
        if _process_is_alive(slug):
            return entry.get("web_auth", "")
    if matches:
        return matches[0][1].get("web_auth", "")

    return ""


def _resolve_agent_owner(port: int) -> str:
    """Resolve the owning user_id of the agent at `web_port`, or "" if unknown.

    The authority is the same per-agent identity used by `_resolve_agent_auth`:
    a Docker label or the process-registry entry (which stores `owner` at spawn
    time). Used by the internal /fd/basna/agent/* endpoints to scope an agent's
    Basna access to its owner without a user JWT.
    """
    # Check Docker containers (owner stamped as a label at spawn time).
    try:
        client = get_docker()
        for c in _deck_containers(client=client):
            labels = c.labels or {}
            wp = labels.get("flight-deck.web-port", "")
            if wp and int(wp) == port:
                owner = labels.get(OWNER_LABEL, "")
                if owner:
                    return owner
    except Exception:
        pass

    # Process registry — prefer a live process over stale entries on the same port.
    registry = _load_process_registry()
    matches = [(slug, e) for slug, e in registry.items() if e.get("web_port") == port]
    for slug, entry in matches:
        if _process_is_alive(slug):
            return entry.get("owner", "") or ""
    if matches:
        return matches[0][1].get("owner", "") or ""

    return ""


def _token_eq(a: str, b: str) -> bool:
    """Constant-time token compare that can't raise on non-ASCII header junk."""
    return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _find_agent_by_auth(token: str) -> tuple[bool, str, str]:
    """``(matched, owner, slug)`` of the agent THIS deck issued web_auth ``token``
    to — a process-registry entry or a running managed container — else
    ``(False, "", "")``, including for an empty token.

    ``owner`` is the owner recorded at spawn ("" when none was); ``slug`` is the
    FD_AGENT_SLUG the agent was spawned with (registry key / container slug).
    Containers come from `_deck_containers`: one carrying ANOTHER deck's label
    is never consulted — its labels were chosen by whoever spawned it there
    (with auth off, anyone), so honouring them would let that deck mint an
    identity here. Unlabelled (pre-label) containers are accepted as before.
    When legacy duplicates exist (old clones copied their source's token), a
    match with a recorded owner wins over an ownerless one. Uses a constant-time
    compare: callers hand this whatever arrived in an X-Agent-Auth header.
    """
    if not token:
        return False, "", ""
    found: tuple[bool, str, str] = (False, "", "")
    for slug, entry in _load_process_registry().items():
        wa = str(entry.get("web_auth") or "")
        if wa and _token_eq(wa, token):
            if entry.get("owner"):
                return True, str(entry["owner"]), slug
            if not found[0]:
                found = (True, "", slug)
    try:
        for c in _deck_containers():
            labels = c.labels or {}
            wa = str(labels.get("flight-deck.web-auth", "") or "")
            if wa and _token_eq(wa, token):
                slug = _slug(labels.get("flight-deck.agent-name", "") or c.name)
                owner = labels.get(OWNER_LABEL, "") or ""
                if owner:
                    return True, owner, slug
                if not found[0]:
                    found = (True, "", slug)
    except Exception:
        pass
    return found


def _find_agent_endpoint_by_auth(token: str) -> dict | None:
    """Where this deck's agent holding web_auth ``token`` serves its files:
    ``{"port", "agent_ref", "name"}``, with ``agent_ref`` the id the SPA keys
    that agent by (``proc-<slug>`` / the container's short id). None when the
    token isn't one this deck issued or the agent has no web port."""
    if not token:
        return None
    for slug, entry in _load_process_registry().items():
        wa = str(entry.get("web_auth") or "")
        if wa and _token_eq(wa, token) and entry.get("web_port"):
            return {"port": int(entry["web_port"]), "agent_ref": f"proc-{slug}",
                    "name": str(entry.get("name") or slug)}
    try:
        for c in _deck_containers():
            labels = c.labels or {}
            wa = str(labels.get("flight-deck.web-auth", "") or "")
            if wa and _token_eq(wa, token):
                info = _container_info(c)
                if info.web_port:
                    return {"port": int(info.web_port), "agent_ref": info.id,
                            "name": info.agent_name or info.name}
    except Exception:
        pass
    return None


def _resolve_agent_identity_by_auth(token: str) -> tuple[bool, str]:
    """Is ``token`` a web_auth THIS deck issued — and to which owner?

    ``(True, owner)`` for one of this deck's agents (``owner`` "" when none was
    recorded), ``(False, "")`` otherwise. Agent-facing connector endpoints use
    this to tell "an FD-spawned agent" (the transport gate: loopback / shared
    secret) apart from "which one": a browser page, or another deck's agent on
    this host, presents no token this deck issued.

    More reliable than port-based lookup (a spawn-time port reassignment can make
    an agent's configured port differ from its registry entry): the auth token is
    unique per agent and stored in both its config and the registry/Docker label.
    When legacy duplicates exist (old clones copied their source's token), a
    match with a recorded owner wins over an ownerless one.
    """
    matched, owner, _slug_ = _find_agent_by_auth(token)
    return matched, owner


def _resolve_agent_owner_by_auth(token: str) -> str:
    """The owning user_id of this deck's agent holding web_auth ``token``, or ""
    (unknown token, or an agent recorded without an owner). See
    `_resolve_agent_identity_by_auth`, which also tells those two apart."""
    return _resolve_agent_identity_by_auth(token)[1]


def _parse_grid_labels(labels: dict) -> tuple[list[str], str]:
    """Grid config from Docker labels: `flight-deck.grid-tags` (JSON list) and
    `flight-deck.grid-recall`. Malformed → empty (never breaks resolution)."""
    try:
        tags = json.loads(labels.get("flight-deck.grid-tags") or "[]")
    except Exception:
        tags = []
    if not isinstance(tags, list):
        tags = []
    return [str(t) for t in tags], str(labels.get("flight-deck.grid-recall") or "")


def _resolve_agent_grid_by_auth(token: str) -> tuple[list[str], str]:
    """Resolve an agent's deep-memory grid config (memory tags, recall mode) by
    its unique web_auth token — the same authority as `_resolve_agent_owner_by_auth`.
    Returns ([], "") for a non-grid agent, so the proxy pools by owner unchanged."""
    if not token:
        return [], ""
    try:
        client = get_docker()
        for c in _deck_containers(client=client):
            labels = c.labels or {}
            if labels.get("flight-deck.web-auth", "") == token:
                return _parse_grid_labels(labels)
    except Exception:
        pass
    for slug, entry in _load_process_registry().items():
        if entry.get("web_auth") == token:
            return list(entry.get("grid_tags") or []), str(entry.get("grid_recall") or "")
    return [], ""


def _resolve_agent_grid(port: int) -> tuple[list[str], str]:
    """Port-keyed fallback for `_resolve_agent_grid_by_auth`, mirroring
    `_resolve_agent_owner`. Prefers a live process over stale same-port entries."""
    try:
        client = get_docker()
        for c in _deck_containers(client=client):
            labels = c.labels or {}
            wp = labels.get("flight-deck.web-port", "")
            if wp and int(wp) == port:
                return _parse_grid_labels(labels)
    except Exception:
        pass
    registry = _load_process_registry()
    matches = [(slug, e) for slug, e in registry.items() if e.get("web_port") == port]
    for slug, entry in matches:
        if _process_is_alive(slug):
            return list(entry.get("grid_tags") or []), str(entry.get("grid_recall") or "")
    if matches:
        e = matches[0][1]
        return list(e.get("grid_tags") or []), str(e.get("grid_recall") or "")
    return [], ""


# ── Flow engine helpers ────────────────────────────────────────────────

def _running_agents() -> list[dict[str, Any]]:
    """Lightweight pool snapshot for the FlowRunner's agent selector."""
    out: list[dict[str, Any]] = []
    for slug, entry in _load_process_registry().items():
        out.append({
            "name": entry.get("name", slug),
            "host": "localhost",
            "port": entry.get("web_port", 0),
            "status": ("running" if _process_is_alive(slug)
                       else "asleep" if agent_idle.is_asleep(slug, entry) else "stopped"),
            "description": entry.get("description", ""),
            # Carry the auth from the SAME entry so the consult uses a token that
            # matches THIS agent — port-keyed re-resolution can collide when stale
            # entries share a port.
            "auth": entry.get("web_auth", ""),
            # The owner recorded at spawn, from the same entry: a flow run may
            # only address its owner's agents.
            "owner": entry.get("owner", "") or "",
        })
    return out


def _flow_origin_env(payload: dict) -> list:
    """The triggering agent's own .env (KEY=VALUE pairs) so a spawned
    specialist inherits ITS tool/provider keys (BRAVE_API_KEY, OPENAI_API_KEY,
    …). This is the 'use the same keys as the agent I'm talking to' fallback
    — tool keys live in the agent's .env, not its config.yaml. Best-effort."""
    port = int(payload.get("origin_port") or 0)
    name = str(payload.get("origin_name") or "")
    # Tenancy: an owned run (the runner stamps its owner as user_id)
    # inherits only from that user's own agent — never another user's
    # .env, whatever origin the payload or a gosub arg names.
    uid = str(payload.get("user_id") or "") if AUTH_ENABLED else ""
    slug = ""
    for s, e in _load_process_registry().items():
        if uid and str(e.get("owner") or "") != uid:
            continue
        if (port and int(e.get("web_port") or 0) == port) or (name and e.get("name") == name):
            slug = s
            break
    if not slug:
        return []
    out: list = []
    try:
        for line in (DATA_DIR / slug / ".env").read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip():
                out.append({"key": k.strip(), "value": v})
    except Exception:
        pass
    return out


def _fd_internal_tools() -> dict[str, Any]:
    """FD-side tools the FlowRunner can call directly (no agent), e.g.
    face_identify. Each is ``async fn(args, *, owner_id="") -> str``: the
    runner passes the run's owner (who the run acts as)."""
    async def _face_identify(args: dict[str, Any], **_kw: Any) -> str:
        """Recognize faces in an image, in-process in Flight Deck.

        ``args.image`` is a filesystem path the FD process can read — use the
        ``{{trigger.fd_image_path}}`` Flow variable (FD-local copy of the
        inbound photo), NOT ``{{trigger.image_path}}`` (which points at the
        agent host). Returns JSON the rest of the Flow can branch on:
          {name, person_id, confident, confidence, count, card}
        On any failure returns {error: "..."} with confident=false so a
        ``branch`` step still has a defined shape to test.
        """
        image = str(args.get("image") or "").strip()
        if not image:
            return json.dumps({"confident": False, "name": None, "error": "no image path provided"})
        try:
            from pathlib import Path as _Path

            p = _Path(image).expanduser()
            if not p.is_file():
                return json.dumps({
                    "confident": False, "name": None,
                    "error": f"image not readable by Flight Deck: {image} "
                             "(use {{trigger.fd_image_path}}, not {{trigger.image_path}})",
                })
            blob = p.read_bytes()

            from captain_claw.flight_deck import face_index  # type: ignore
            result = await face_index.get_index().recognize(image_blob=blob, channel="flow")
            return json.dumps({
                "confident": bool(result.name is not None),
                "name": result.name,
                "person_id": result.person_id,
                "confidence": round(float(result.confidence), 4),
                "count": len(result.faces),
                "card": result.card_markdown,
            })
        except RuntimeError as exc:
            # Missing 'faces' extra (insightface/sqlite-vec) surfaces here.
            return json.dumps({"confident": False, "name": None, "error": str(exc)})
        except Exception as exc:
            return json.dumps({"confident": False, "name": None, "error": f"face_identify failed: {exc}"})

    async def _jev(args: dict[str, Any], *, owner_id: str = "") -> str:
        """Ask Jev (TypeSafe AI) with the run owner's key, in Flight Deck.

        ``args.state`` (text; JSON object / array text is parsed),
        ``args.questions`` (JSON: a list of questions with a ``name``, or a
        map), optional ``args.model``. Returns JSON, never raises:
          {ok: true, model, answers, values, usage, summary} |
          {ok: false, error}
        ``{{steps.<id>.values.<question>}}`` is the headline answer (p(yes) /
        the chosen option / the score).
        """
        from captain_claw.flight_deck import jev_routes

        return await jev_routes.flow_tool(args, owner_id=owner_id)

    return {"face_identify": _face_identify, "jev": _jev}


async def _flow_whatsapp_send(waid: str, text: str) -> None:
    try:
        from captain_claw.flight_deck.whatsapp_bridge import _send_whatsapp_text
        await _send_whatsapp_text(waid, text, mirror=True)
    except Exception as exc:
        log.warning("flow whatsapp send failed: %s", exc)


# ── Flow engine API (/fd/flows) ────────────────────────────────────────

def _flow_store():
    store = getattr(app.state, "flow_store", None)
    if store is None:
        raise HTTPException(503, "Flow engine not ready")
    return store


# ── Flow tenancy: a flow belongs to the user who created it. Non-admins see,
# edit and run only their own; admins see all; legacy rows with no owner are
# admin-only. Auth off → one trusted user, nothing is scoped. ──

def _flow_caller(user: dict | None) -> tuple[str, bool]:
    """(user_id, is_admin) of a flow-route caller; ("", True) with auth off."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled
    if not _fd_auth_enabled() or not user:
        return "", True
    return str(user.get("id") or ""), user.get("role") == "admin"


def _flow_visible(flow: dict | None, user: dict | None) -> bool:
    uid, admin = _flow_caller(user)
    return bool(flow) and (admin or (bool(uid) and flow.get("owner_id") == uid))


async def _owned_flow(flow_id: str, user: dict | None) -> dict:
    """The flow, if the caller may use it — else 404 (no existence oracle)."""
    flow = await _flow_store().get_flow(flow_id)
    if not _flow_visible(flow, user):
        raise HTTPException(404, "flow not found")
    return flow


async def _check_flow_run(run_id: str, user: dict | None) -> None:
    """404 unless the caller may see run *run_id* (it follows its flow)."""
    if _flow_caller(user)[1]:
        return
    detail = await _flow_store().get_run(run_id)
    flow_id = str(((detail or {}).get("run") or {}).get("flow_id") or "")
    flow = await _flow_store().get_flow(flow_id) if flow_id else None
    if not _flow_visible(flow, user):
        raise HTTPException(404, "run not found")


async def _flow_user_is_admin(uid: str) -> bool:
    """Whether user *uid* is an admin (the flow runner's tenancy seam)."""
    if not uid:
        return False
    try:
        from captain_claw.flight_deck.auth import get_db
        u = await get_db().get_user_by_id(uid)
    except Exception:
        return False
    return bool(u) and u.get("role") == "admin"


@app.get("/fd/flows")
async def fd_flows_list(request: Request, user: dict | None = _required_user_dep):
    uid, admin = _flow_caller(user)
    return {"flows": await _flow_store().list_flows(owner_id=None if admin else uid)}


@app.post("/fd/flows")
async def fd_flows_create(request: Request, user: dict | None = _required_user_dep):
    spec = await request.json()
    fid = await _flow_store().create_flow(spec, owner_id=_flow_caller(user)[0])
    return {"id": fid}


@app.get("/fd/flows/runs/{run_id}")
async def fd_flows_run_detail(run_id: str, request: Request, user: dict | None = _required_user_dep):
    await _check_flow_run(run_id, user)
    detail = await _flow_store().get_run(run_id)
    if not detail:
        raise HTTPException(404, "run not found")
    return detail


@app.post("/fd/flows/evaluate")
async def fd_flows_evaluate(request: Request, user: dict | None = _optional_user_dep):
    """Agent-handled channels (web/glasses) call this before a turn: classify the
    inbound, and if a Flow matches, run it and return its output to relay."""
    from captain_claw.flight_deck import flow_router
    if not flow_router.engine_ready():
        return {"matched": False}
    raw = await request.json()
    # An automated agent turn (autonomy nudge, scheduler prompt, peer relay,
    # MCP task…) started this: its text is not a person's message, so it never
    # counts as a request to write email (flow_runner._human_trigger_text). Its
    # own mail authority comes along too: a ``deny`` turn starts a run whose
    # every step denies email (flow_runner._root_mail_ctx) — no escalation.
    _auto: dict[str, str] = {}
    if isinstance(raw.get("automated"), str) and raw.get("automated"):
        _auto["automated"] = str(raw["automated"])[:32]
    if raw.get("automated_mail_write") is not None:
        _amw = str(raw["automated_mail_write"]).strip().lower()
        # Present but malformed fails closed, like the agent's own wire parser.
        _auto["automated_mail_write"] = _amw if _amw in ("deny", "intent", "allow") else "deny"
    payload = flow_router.classify_payload(
        channel=str(raw.get("channel") or "web"),
        text=str(raw.get("text") or ""),
        mime=str(raw.get("mime") or ""),
        image_path=str(raw.get("image_path") or ""),
        video_path=str(raw.get("video_path") or ""),
        audio_path=str(raw.get("audio_path") or ""),
        waid=str(raw.get("waid") or ""),
        origin_host=str(raw.get("origin_host") or "localhost"),
        origin_port=int(raw.get("origin_port") or 0),
        origin_name=str(raw.get("origin_name") or ""),
        extra=_auto or None,
    )
    # 0. Flow control command ('/flow stop|pause|resume', slash optional) —
    #    intercept before treating the message as input or a new trigger.
    if await flow_router.maybe_handle_flow_command(payload):
        return {"matched": True, "deferred": True}

    # 0b. Explicit start: '/flow run|start <name|id>' — launch a specific flow
    #     BOUND to this channel/origin (same path as a trigger match) so that
    #     /flow status|stop and input-resume all target it. Works for disabled
    #     flows too (enable/disable only gates automatic trigger firing).
    import re as _re
    _run_m = _re.match(r"^\s*/?flow\s+(?:run|start|launch)\s+(.+)$", str(payload.get("text") or ""), _re.I)
    if _run_m:
        _target = _run_m.group(1).strip()
        _store = flow_router._STORE
        _flow = None
        # Only the origin agent's owner's flows (an admin's agent: any flow).
        _owner = _resolve_agent_owner(int(payload.get("origin_port") or 0)) if (
            AUTH_ENABLED and payload.get("origin_port")) else ""
        _scope = _owner if (_owner and not await _flow_user_is_admin(_owner)) else None
        if _store is not None:
            _flow = await _store.get_flow(_target)
            if _flow is not None and _scope is not None and _flow.get("owner_id") != _scope:
                _flow = None
            _flow = _flow or await _store.get_flow_by_name(_target, owner_id=_scope)
        if _flow is None:
            return {"matched": True, "output": f"No flow named “{_target}”."}
        if flow_router._flow_needs_async(_flow):
            asyncio.create_task(flow_router._bg_run(_flow, payload))
            return {"matched": True, "flow": _flow.get("name"), "deferred": True}
        _result = await app.state.flow_runner.run(_flow, payload)
        return {
            "matched": True, "flow": _flow.get("name"),
            "run_id": _result.get("run_id"), "output": _result.get("output") or "",
            **_flow_private_field(_result),
        }

    # 1. Resume a paused flow first: if one is waiting on an `input` step for
    #    this channel+agent, this message is the reply — feed it and stop (the
    #    flow continues in the background and delivers via the channel).
    #    Only a person's message is a reply: an automated turn's text (autonomy
    #    nudge, peer relay, scheduler prompt…) never answers an `input` step or
    #    satisfies a `wait until` gate a flow put in front of the user.
    if not _auto and flow_router.deliver_pending_input(
        waid=payload.get("waid", ""), channel=payload.get("channel", ""),
        origin_port=int(payload.get("origin_port") or 0), text=payload.get("text", ""),
    ):
        return {"matched": True, "deferred": True}

    flow = await flow_router.match_flow(payload)
    if not flow:
        return {"matched": False}

    # 2. Flows that can pause for input or consult the ORIGIN agent must run
    #    detached: the origin agent is the one blocked on THIS evaluate call, so
    #    a synchronous origin-consult would deadlock it. Background it and let it
    #    deliver via the channel (agent chat-push); the agent ends its turn now.
    if flow_router._flow_needs_async(flow):
        asyncio.create_task(flow_router._bg_run(flow, payload))
        return {"matched": True, "flow": flow.get("name"), "deferred": True}

    # 3. Simple flow → run inline and relay its output (unchanged behaviour).
    result = await app.state.flow_runner.run(flow, payload)
    return {
        "matched": True, "flow": flow.get("name"),
        "run_id": result.get("run_id"), "output": result.get("output") or "",
        **_flow_private_field(result),
    }


def _flow_private_field(result: dict[str, Any]) -> dict[str, str]:
    """PR D: ``{"member_private": level}`` when an inline flow run returned
    members' private data, else ``{}``. The output stays plain (a person reads
    it); the agent relaying it marks its reply frame with the level, so a
    consult / delegate that triggered the flow still gets the header."""
    _lvl = (result or {}).get("member_private")
    return {"member_private": _lvl} if _lvl in ("data", "content") else {}


# ── Flow DSL: text <-> flow, and agent-assisted NL -> flow ──────────────
# Registered BEFORE /fd/flows/{flow_id} so 'dsl'/'compile' aren't read as ids.

@app.get("/fd/flows/docs")
async def fd_flows_docs(request: Request, user: dict | None = _optional_user_dep):
    """Serve the Flow language reference (FLOWS.md) for the in-app docs viewer."""
    from pathlib import Path as _Path
    here = _Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "FLOWS.md",   # repo root (../../ from this file)
        here.parent.parent / "FLOWS.md",
        _Path.cwd() / "FLOWS.md",
    ]
    for p in candidates:
        try:
            if p.is_file():
                return {"ok": True, "markdown": p.read_text(encoding="utf-8")}
        except Exception:
            continue
    return {"ok": False, "markdown": "# Flow docs\n\nFLOWS.md was not found on the server."}


@app.post("/fd/flows/dsl/compile")
async def fd_flows_dsl_compile(request: Request, user: dict | None = _optional_user_dep):
    """Deterministic: DSL text → flow dict (with structured errors)."""
    from captain_claw.flight_deck import flow_dsl
    body = await request.json()
    try:
        flow = flow_dsl.compile_dsl(str(body.get("dsl") or ""),
                                    strict_refs=bool(body.get("strict_refs", False)))
        return {"ok": True, "flow": flow}
    except flow_dsl.DSLError as exc:
        return {"ok": False, "error": exc.msg, "line": exc.line}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "line": 0}


@app.post("/fd/flows/dsl/decompile")
async def fd_flows_dsl_decompile(request: Request, user: dict | None = _optional_user_dep):
    """Deterministic: flow dict → DSL text (for the code view / round-trip)."""
    from captain_claw.flight_deck import flow_dsl
    body = await request.json()
    flow = body.get("flow") or {}
    try:
        return {"ok": True, "dsl": flow_dsl.decompile(flow)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _ai_compile_flow(text: str, agent: str = "", current: str = "") -> dict:
    """Agent-assisted: free text / loose code → canonical DSL (via a pooled
    agent's model) → deterministic compile + validate. The model's output is
    always run through the real parser, so invalid output is rejected. Shared by
    /fd/flows/compile and /fd/flows/synthesize."""
    import re
    from captain_claw.flight_deck import flow_dsl
    text = str(text or "").strip()
    if not text:
        return {"ok": False, "error": "no text to compile"}
    current = str(current or "").strip()
    body = {"agent": agent}

    # Pick a RUNNING agent with a live web port to host the model call. Try
    # several (a registry "running" flag can lag a just-died web server).
    running = [a for a in _running_agents()
               if str(a.get("status")) == "running" and int(a.get("port") or 0) > 0]
    if not running:
        return {"ok": False, "error": "no running agent available to compile with"}
    # The caller may pick which agent compiles; prefer it, fall back to others.
    chosen = str(body.get("agent") or "").strip()
    if chosen:
        picked = [a for a in running if str(a.get("name")) == chosen]
        agents = picked + [a for a in running if str(a.get("name")) != chosen]
    else:
        agents = running

    system = (
        "You translate a user's request into a Captain Claw Flow DSL program. "
        "Output ONLY the DSL — no prose, no code fences.\n"
        "Grammar:" + flow_dsl.__doc__.split("Grammar", 1)[1] + "\n"
        "STEP TYPE RULES (important):\n"
        "• agent — open-ended work the model decides how to do: search the web, "
        "research, look something up, summarize, answer. Use `agent on origin` "
        "with a `prompt:`. THIS is what you use for 'search for X', 'find', "
        "'research', 'look up'.\n"
        "• tool — a SINGLE named, deterministic tool. ONLY use it when you know "
        "the exact tool name; it needs `tool: <name>` and `arg <k>: <v>`. The "
        "`on fd` tools are `face_identify` and `jev` (Jev: calibrated yes/no, "
        "choice and score judgments about text with the user's Jev key — `arg "
        "state:` the text, `arg questions:` a one-line JSON list like "
        "[{\"name\":\"urgent\",\"type\":\"noul\",\"instructions\":\"Is this urgent?\"}] "
        "(choice: \"options\":[...], score: \"levels\":[lowest … highest]); "
        "branch on {{steps.<id>.values.<name>}}). NEVER emit a `tool` step without "
        "a `tool:` name — use an `agent` step instead.\n"
        "• vision — describe/read an image (`on capability:vision`, `image:`).\n"
        "• input — ask the user something and wait (`prompt:`). The reply is "
        "{{steps.<id>.output}}.\n"
        "• emit — send a message (`emit \"...\"`).\n"
        "• gosub — call ANOTHER flow as a subroutine and wait for it: "
        "`gosub \"Other Flow\"` with optional `with <k>: <v>` argument lines. The "
        "child's return value is {{calls.<step_id>.output}} and its status is "
        "{{calls.<step_id>.status}}. Use it to reuse an existing flow.\n"
        "• return — end the flow now and hand a value back to the caller: "
        "`return {{steps.x.output}}` (works inside a branch path too). A flow that "
        "is meant to be gosub'd should end with `output -> return`.\n"
        "• spawn — start another flow in the BACKGROUND (don't wait): "
        "`spawn \"Other Flow\"` with optional `with <k>: <v>`. Later `join <step_id>` "
        "to collect it. Use it to run things in parallel.\n"
        "• join — wait for a spawned flow: `join <spawn_step_id>` with optional "
        "`timeout: <seconds>`. Result is {{joins.<spawn_step_id>.output}} and "
        "status {{joins.<spawn_step_id>.status}} (done/error/timeout).\n"
        "• error — an error-handler step that reports a problem: "
        "`error \"It failed: {{error.message}}\"`. Reach it from a failing call's "
        "`on error -> <step>` line, or by branching on {{calls.<id>.status}}.\n"
        "On a gosub/join/spawn step, add `on error -> <step>` to jump to a handler "
        "if that call fails. Add `retry: <N>` to a gosub/join/spawn to re-try on "
        "failure before on-error.\n"
        "• set — compute a value into {{vars.<name>}}: `set total = {{vars.total}} "
        "+ 1`. The expression supports + - * / (+ also concatenates strings/lists), "
        "list literals [a, b], {{path}} operands, and functions split, join, len, "
        "upper, lower, trim, first, last, append, int, str, contains. e.g. "
        "`set cities = split({{steps.ask.output}}, \",\")` turns text into a list.\n"
        "• foreach — run a flow once per list item: a `foreach <var> in <list>` "
        "header, then a `gosub \"Flow\"` (sequential) OR `spawn \"Flow\"` (parallel) "
        "line, then `with <k>: <v>` args that use {{<var>}}. {{steps.<id>.output}} "
        "is the list of each result. Use it instead of repeating near-identical "
        "steps.\n"
        "• while — loop: `while <condition> -> <target>` jumps to <target> (whose "
        "path loops back) while the condition holds; else falls through. Pair with "
        "`set` for a counter.\n"
        "• sleep — pause the run: `sleep 30s` / `5m` / `2h` / `1d`.\n"
        "• wait — pause until an inbound message matches: `wait until contains "
        "\"approved\"` (the flow parks; other messages go to the agent). The "
        "matching text is {{steps.<id>.output}}. Good for approvals.\n"
        "Lists: any value can be a list (from set/split/foreach); when shown in a "
        "string it joins by newlines.\n"
        "WHEN TO USE WHAT: open-ended thinking → agent; iterate over many items → "
        "foreach; parallel work → spawn+join or foreach+spawn; loop with a counter "
        "→ set+while; reuse an existing flow → gosub; wait for the user → input "
        "(asks now) or wait (gates on a condition).\n"
        "TRIGGER: default to `trigger any` unless the user explicitly names a "
        "channel (whatsapp/web/glasses). Add `when <rules>` only if they gave a "
        "condition. Rules are joined with `and` (all must match) OR `or` (any "
        "matches) — pick ONE, don't mix. For 'any of these words' use `or`, e.g. "
        "`trigger any when contains \"hungry\" or contains \"gladan\" or contains "
        "\"gladni\"`. Write quotes plainly — never backslash-escape them.\n"
        "Selectors: origin, fd, any, capability:vision, name:<agent>, and "
        "archetype:<id>[@tier]. Use `agent on archetype:<id>` to run a step on a "
        "freshly spawned, role-specialised agent that is disposed when the flow "
        "ends — ideal for multi-stage pipelines (research → fact-check → write). "
        "Archetype ids include: deep-researcher, market-scanner, fact-checker, "
        "editor-writer, comms-outbound, social-repurposer, data-analyst, "
        "report-builder, monitor-watchdog, triage-router. Example: "
        "`agent on archetype:fact-checker`. Add `@tier` (reason/balanced/fast/"
        "longctx/coding/vision) only to override the archetype's default model. "
        "Template with {{trigger.text}}, {{trigger.image_path}}, "
        "{{trigger.fd_image_path}}, {{steps.<id>.output}}. Always include a "
        "trigger, at least one step, and an `output -> same` line."
    )
    if current:
        system += (
            "\n\nEDIT MODE: The user already has a flow (given below) and wants "
            "to MODIFY it. Apply their requested change to that existing program "
            "and output the COMPLETE updated DSL — keep every other step, the "
            "trigger, and the output line unchanged unless the request says "
            "otherwise. Preserve existing step ids; give any new step a fresh, "
            "descriptive snake_case id. Do not drop or reorder unrelated steps."
        )
    import httpx

    async def _complete(messages: list[dict[str, Any]]) -> tuple[str, str]:
        """Return (dsl, error). Tries running agents until one responds."""
        last = "no reachable agent"
        for agent in agents:
            host, port, token = agent["host"], int(agent["port"]), agent.get("auth", "")
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    r = await client.post(
                        f"http://{host}:{port}/api/llm/complete",
                        params={"token": token} if token else {},
                        json={"messages": messages, "temperature": 0.1},
                    )
                d = r.json() or {}
                if not d.get("ok"):
                    last = f"model error on {agent.get('name')}: {d.get('error') or r.status_code}"
                    continue
                content = str(d.get("content") or "").strip()
                content = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", content).strip()
                return content, ""
            except Exception as exc:
                last = f"{agent.get('name')}: {exc}"
        return "", last

    if current:
        user_msg = (
            "Here is the current flow DSL:\n\n"
            f"{current}\n\n"
            f"Requested change: {text}\n\n"
            "Output ONLY the complete updated DSL."
        )
    else:
        user_msg = text
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    dsl, err = await _complete(messages)
    if not dsl:
        return {"ok": False, "error": f"compile call failed ({err})"}

    # Up to two attempts: if the generated DSL fails to compile, feed the error
    # back to the model once and let it repair its own output.
    for attempt in range(2):
        try:
            flow = flow_dsl.compile_dsl(dsl)
            return {"ok": True, "flow": flow, "dsl": dsl}
        except flow_dsl.DSLError as exc:
            problem = f"line {exc.line}: {exc.msg}"
        except Exception as exc:
            problem = str(exc)
        if attempt == 0:
            messages += [
                {"role": "assistant", "content": dsl},
                {"role": "user", "content": (
                    f"That DSL failed to compile ({problem}). Fix it and output "
                    "ONLY the corrected DSL. Remember: open-ended work like "
                    "searching is an `agent on origin` step with a prompt, never "
                    "a bare `tool` step."
                )},
            ]
            fixed, ferr = await _complete(messages)
            if fixed:
                dsl = fixed
                continue
        return {"ok": False, "error": f"generated DSL invalid ({problem})", "dsl": dsl}
    return {"ok": False, "error": "generated DSL invalid", "dsl": dsl}


@app.post("/fd/flows/compile")
async def fd_flows_compile(request: Request, user: dict | None = _required_user_dep):
    """Agent-assisted NL/loose-code → validated flow (for the Code view)."""
    body = await request.json()
    return await _ai_compile_flow(
        str(body.get("text") or ""), str(body.get("agent") or ""), str(body.get("current") or ""),
    )


@app.post("/fd/flows/synthesize")
async def fd_flows_synthesize(request: Request, user: dict | None = _required_user_dep):
    """Agent synthesis: a natural-language goal → a validated, **call-only**
    flow stored in the SCRATCH space (origin=agent). Dedups by canonical hash
    (retrieve-before-generate), optionally runs it, and returns its handle/name.

    Body: {goal, agent?, author?, run?(bool), payload?(dict)}."""
    from captain_claw.flight_deck import flow_dsl
    body = await request.json()
    goal = str(body.get("goal") or body.get("text") or "").strip()
    if not goal:
        return {"ok": False, "error": "no goal to synthesize"}
    author = str(body.get("author") or body.get("agent") or "").strip()
    store = _flow_store()
    uid, admin = _flow_caller(user)

    # 1. Compile the goal to a validated flow via a pooled model.
    res = await _ai_compile_flow(goal, str(body.get("agent") or ""))
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error") or "synthesis failed", "dsl": res.get("dsl")}
    flow = res["flow"]
    dsl = res.get("dsl") or flow_dsl.decompile(flow)
    flow["origin"] = "agent"

    # 2. Dedup: a structurally-identical scratch flow already exists → reuse it.
    #    A quarantined one is negative memory — don't re-create the same bad flow.
    h = flow_dsl.canonical_hash(flow)
    existing = await store.find_scratch_by_hash(h, owner_id=uid or None)  # the caller's own
    if existing and existing.get("state") == "quarantined":
        return {"ok": False, "quarantined": True,
                "error": "This flow pattern failed repeatedly before (quarantined) — refine the goal."}
    if existing:
        fid, name, reused = existing["id"], existing["name"], True
    else:
        fid = await store.create_scratch_flow(flow, author=author, dsl_hash=h, owner_id=uid)
        name, reused = flow.get("name") or "Synthesized flow", False

    run = bool(body.get("run"))
    # A selection without a run counts as a use; a run records its own outcome.
    if reused and not run:
        await store.bump_use(fid)

    out: dict = {"ok": True, "flow_id": fid, "name": name, "reused": reused, "dsl": dsl}

    # 3. Optionally run it now (this records the outcome → drives promotion/quarantine).
    if run:
        target = await store.get_flow(fid)
        result = await app.state.flow_runner.run(
            target, body.get("payload") or {}, owner_id=uid, is_admin=admin)
        out["run_id"] = result.get("run_id")
        out["status"] = result.get("status")
        out["output"] = result.get("output") or ""
        _lvl = result.get("member_private")
        if _lvl in ("data", "content"):
            out["member_private"] = _lvl
            if "author" in body:
                # PR D: the synthesize_flow tool (it always sends `author`; the
                # UI never does) hands the output to its agent — header first.
                out["output"] = _private_relay_header(_lvl) + "\n" + out["output"]
    return out


@app.get("/fd/flows/scratch")
async def fd_flows_scratch(request: Request, user: dict | None = _required_user_dep):
    """List the scratch space (synthesized flows) with provenance + lifecycle.
    Self-maintains: reclassifies states and GCs expired flows on view."""
    store = _flow_store()
    try:
        await store.maintain_scratch()
    except Exception as exc:
        log.warning("scratch maintain failed: %s", exc)
    uid, admin = _flow_caller(user)
    return {"flows": await store.list_scratch_flows(owner_id=None if admin else uid)}


@app.post("/fd/flows/scratch/maintain")
async def fd_flows_scratch_maintain(request: Request, user: dict | None = _required_user_dep):
    """Janitor: reclassify every scratch flow (candidate/quarantined) and GC the
    expired ones. Returns a summary. Safe to call on a schedule."""
    return {"ok": True, **(await _flow_store().maintain_scratch())}


@app.post("/fd/flows/{flow_id}/promote")
async def fd_flows_promote(flow_id: str, request: Request, user: dict | None = _required_user_dep):
    """Promote a scratch flow into the permanent space (optionally rename)."""
    await _owned_flow(flow_id, user)
    try:
        body = await request.json()
    except Exception:
        body = {}
    ok = await _flow_store().promote_flow(flow_id, name=str(body.get("name") or "") or None)
    if not ok:
        raise HTTPException(404, "scratch flow not found")
    return {"ok": True}


@app.get("/fd/flows/{flow_id}")
async def fd_flows_get(flow_id: str, request: Request, user: dict | None = _required_user_dep):
    return await _owned_flow(flow_id, user)


@app.put("/fd/flows/{flow_id}")
async def fd_flows_update(flow_id: str, request: Request, user: dict | None = _required_user_dep):
    await _owned_flow(flow_id, user)
    spec = await request.json()
    ok = await _flow_store().update_flow(flow_id, spec)
    if not ok:
        raise HTTPException(404, "flow not found")
    return {"ok": True}


@app.delete("/fd/flows/{flow_id}")
async def fd_flows_delete(flow_id: str, request: Request, user: dict | None = _required_user_dep):
    await _owned_flow(flow_id, user)
    ok = await _flow_store().delete_flow(flow_id)
    return {"ok": ok}


@app.post("/fd/flows/{flow_id}/enable")
async def fd_flows_enable(flow_id: str, request: Request, user: dict | None = _required_user_dep):
    await _owned_flow(flow_id, user)
    body = await request.json()
    ok = await _flow_store().set_enabled(flow_id, bool(body.get("enabled", True)))
    if not ok:
        raise HTTPException(404, "flow not found")
    return {"ok": True}


@app.post("/fd/flows/{flow_id}/run")
async def fd_flows_run(flow_id: str, request: Request, user: dict | None = _required_user_dep):
    store = _flow_store()
    flow = await _owned_flow(flow_id, user)
    try:
        body = await request.json()
    except Exception:
        body = {}
    payload = body.get("payload") or {}
    uid, admin = _flow_caller(user)
    run_id = await store.start_run(flow_id, flow.get("name", ""), payload)
    # Run in the background; the UI polls /fd/flows/runs/{run_id} for the log.
    # The run acts as the caller: its steps reach only the caller's agents.
    asyncio.create_task(app.state.flow_runner.run(
        flow, payload, run_id=run_id, owner_id=uid, is_admin=admin))
    return {"run_id": run_id}


@app.post("/fd/flows/runs/{run_id}/pause")
async def fd_flows_run_pause(run_id: str, request: Request, user: dict | None = _required_user_dep):
    await _check_flow_run(run_id, user)
    from captain_claw.flight_deck import flow_runner
    ok = flow_runner.request_pause(run_id)
    if ok:
        try:
            await _flow_store().set_run_status(run_id, "paused")
        except Exception:
            pass
    return {"ok": ok, "status": "paused" if ok else "not_running"}


@app.post("/fd/flows/runs/{run_id}/resume")
async def fd_flows_run_resume(run_id: str, request: Request, user: dict | None = _required_user_dep):
    await _check_flow_run(run_id, user)
    from captain_claw.flight_deck import flow_runner
    ok = flow_runner.request_resume(run_id)
    if ok:
        try:
            await _flow_store().set_run_status(run_id, "running")
        except Exception:
            pass
    return {"ok": ok, "status": "running" if ok else "not_running"}


@app.post("/fd/flows/runs/{run_id}/stop")
async def fd_flows_run_stop(run_id: str, request: Request, user: dict | None = _required_user_dep):
    await _check_flow_run(run_id, user)
    try:
        body = await request.json()
    except Exception:
        body = {}
    message = str(body.get("message") or "")
    from captain_claw.flight_deck import flow_runner
    ok = flow_runner.request_stop(run_id, message)
    return {"ok": ok, "status": "stopping" if ok else "not_running"}


@app.post("/fd/flows/{flow_id}/test")
async def fd_flows_test(flow_id: str, request: Request, user: dict | None = _required_user_dep):
    flow = await _owned_flow(flow_id, user)
    try:
        body = await request.json()
    except Exception:
        body = {}
    uid, admin = _flow_caller(user)
    # A dry run still dispatches its steps — same owner context as /run.
    result = await app.state.flow_runner.run(
        flow, body.get("payload") or {}, dry=True, owner_id=uid, is_admin=admin)
    return {"steps": result.get("steps", []), "status": result.get("status")}


@app.get("/fd/flows/{flow_id}/runs")
async def fd_flows_runs(flow_id: str, request: Request, user: dict | None = _required_user_dep):
    await _owned_flow(flow_id, user)
    return {"runs": await _flow_store().list_runs(flow_id)}


async def _notify_fleet_change(new_agent_name: str, new_agent_port: int, event: str = "joined", owner_id: str = ""):
    """Notify running agents about a fleet change. When auth is enabled, only notify agents owned by the same user."""
    import websockets as _ws
    import json as _json

    # Other Flight Decks see this deck's exposed agents: re-advertise them.
    try:
        from captain_claw.flight_deck.botport_link import get_link as _get_botport_link

        _get_botport_link().notify_fleet_changed()
    except Exception:
        pass

    # Build list of running agent WebSocket endpoints (excluding the new one),
    # scoped to the same owner when auth is enabled.
    targets: list[tuple[str, int, str]] = []  # (host, port, auth)

    try:
        client = get_docker()
        for c in _deck_containers(client=client):
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
        for c in _deck_containers(all=True, client=client):
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


_IMAGE_EXTS_TRANSFER = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


async def _upload_to_agent(host: str, port: int, auth: str, filename: str,
                           blob: bytes) -> tuple[list[str], list[str]]:
    """Upload ``blob`` to the TARGET agent's /api/image|file/upload,
    authenticated with ``auth`` (its web_auth — FD holds every agent's, which
    is why it uploads on the sender's behalf). Returns (image_paths,
    file_paths) ON THE TARGET; ([], []) when the upload fails."""
    import httpx

    is_img = Path(filename).suffix.lower() in _IMAGE_EXTS_TRANSFER
    endpoint = "/api/image/upload" if is_img else "/api/file/upload"
    params = {"token": auth} if auth else {}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"http://{host}:{port}{endpoint}", params=params,
                files={"file": (filename, blob)},
            )
        if resp.status_code == 200:
            tgt = str((resp.json() or {}).get("path") or "")
            if tgt:
                return ([tgt], []) if is_img else ([], [tgt])
        else:
            log.warning("agent file transfer rejected", status=resp.status_code, body=resp.text[:200])
    except Exception as exc:
        log.warning("agent file transfer failed: %s", exc)
    return [], []


async def _transfer_file_to_agent(host: str, port: int, abs_path: str) -> tuple[list[str], list[str]]:
    """Upload a file FD can read to the TARGET agent — the flow engine's seam.

    Reads ``abs_path`` from FD's own disk UNCONFINED, so it is for in-process
    callers only. The HTTP peer routes never use it: they read through
    `_read_agent_attachment`, which confines the read to the calling agent's
    own workspace. Returns (image_paths, file_paths) on the TARGET.
    """
    p = Path(abs_path)
    if not abs_path or not p.is_file():
        return [], []
    try:
        blob = p.read_bytes()
    except Exception as exc:
        log.warning("agent file transfer failed: %s", exc)
        return [], []
    return await _upload_to_agent(host, port, _resolve_agent_auth(port, host=host), p.name, blob)


# ── Peer consult / delegate: who may drive which agent ──
#
# /fd/consult-peer and /fd/delegate-peer make FD open a WebSocket to an agent
# with that agent's own web_auth, instruct it (it then acts with its owner's
# accounts), and optionally upload a file on the caller's behalf. So:
#
# * The target's host, port and token come ONLY from FD's own records — a
#   token FD resolved is never sent to a host the caller named.
# * The caller proves who it is and may drive only its owner's agents: a
#   verified bearer (the target's owner, or an admin), or an FD-spawned agent —
#   loopback or X-Agent-Secret (the transport guard in _hardening_middleware)
#   PLUS its own web_auth in X-Agent-Auth, whose recorded owner must be the
#   target's. With auth disabled (single-user local mode) there is no tenant
#   boundary, so any agent this deck spawned may drive any other.
# * An attachment is read only from the calling agent's own workspace.

# Every agent FD records is reachable from FD at localhost (what /fd/fleet reports).
_PEER_AGENT_HOST = "localhost"
# Where a Docker agent sees its workspace (bind-mounted from DATA_DIR/<name>/data/workspace).
_AGENT_WORKSPACE_IN_CONTAINER = "/data/workspace"


def _agent_record_by_auth(token: str) -> dict | None:
    """This deck's agent holding web_auth ``token`` — ``{kind, slug, port,
    auth, owner}`` — or None (unknown or empty token). Same records as
    `_resolve_agent_owner_by_auth` (process registry, labelled containers);
    when legacy clones share a token, a match with a recorded owner wins.
    Constant-time compare (`_token_eq`): ``token`` is whatever arrived in an
    X-Agent-Auth header."""
    if not token or not isinstance(token, str):
        return None
    found: dict | None = None
    for slug, entry in _load_process_registry().items():
        wa = str(entry.get("web_auth") or "")
        if wa and _token_eq(wa, token):
            rec = {"kind": "process", "slug": slug, "port": int(entry.get("web_port") or 0),
                   "auth": token, "owner": str(entry.get("owner") or "")}
            if rec["owner"]:
                return rec
            found = found or rec
    try:
        for c in _deck_containers():  # never another deck's (forgeable) labels
            labels = c.labels or {}
            wa = str(labels.get("flight-deck.web-auth", "") or "")
            if wa and _token_eq(wa, token):
                wp = str(labels.get("flight-deck.web-port", ""))
                rec = {"kind": "docker", "slug": c.name, "port": int(wp) if wp.isdigit() else 0,
                       "auth": token, "owner": str(labels.get(OWNER_LABEL, "") or "")}
                if rec["owner"]:
                    return rec
                found = found or rec
    except Exception:
        pass
    return found


def _resolve_peer_caller(request: Request, user: dict | None) -> tuple[str, bool, dict | None]:
    """``(owner, is_admin, calling_agent)`` for a peer-route request, else 403.

    A verified bearer is authoritative: the caller acts only as that user, and
    an X-Agent-Auth it also sends must name one of that user's agents (it is
    what an attachment is read from). Otherwise the caller must pass the
    transport guard (loopback or X-Agent-Secret) AND name itself with its own
    web_auth in X-Agent-Auth; its owner is the one FD recorded at spawn. A body
    field never identifies anyone.
    """
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    agent = _agent_record_by_auth(request.headers.get("X-Agent-Auth", "").strip())
    if user and user.get("id"):
        uid = str(user["id"])
        is_admin = user.get("role") == "admin"
        if agent is not None and agent["owner"] != uid and not is_admin:
            raise HTTPException(403, "X-Agent-Auth names another user's agent")
        return uid, is_admin, agent
    if not _agent_caller_ok(request):
        raise HTTPException(403, "peer routes require a bearer token, or loopback / "
                                 "X-Agent-Secret plus the calling agent's X-Agent-Auth")
    if agent is None:
        raise HTTPException(403, "could not identify the calling agent (X-Agent-Auth)")
    if _fd_auth_enabled() and not agent["owner"]:
        raise HTTPException(403, "calling agent has no recorded owner on this deck")
    return agent["owner"], False, agent


def _resolve_peer_target(port: int, owner: str, is_admin: bool) -> str:
    """The web_auth FD recorded for its agent on web port ``port`` — once the
    caller (``owner``) is allowed to drive it: 404 when FD has no agent there,
    403 when it belongs to someone else (auth enabled; admins reach any)."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    auth = _resolve_agent_auth(int(port)) if port else ""
    if not auth:
        raise HTTPException(404, f"no Flight Deck agent on port {port}")
    if _fd_auth_enabled() and not is_admin:
        target_owner = _resolve_agent_owner(int(port))
        if not target_owner or target_owner != owner:
            raise HTTPException(403, "This agent belongs to another user")
    return auth


def _workspace_relative_parts(agent: dict, attach_path: str) -> tuple[str, ...]:
    """``attach_path``'s components below the calling agent's workspace, or 403.

    The path is taken as the agent names it: a process agent's host path under
    DATA_DIR/<slug>/data/workspace, a Docker agent's /data/workspace/... . It is
    normalised lexically ('..' collapsed) before the prefix check; symlinks are
    dealt with by the read itself (`_read_regular_file_under`)."""
    import posixpath

    host_root = DATA_DIR / agent["slug"] / "data" / "workspace"
    if agent["kind"] == "docker":
        prefixes = [_AGENT_WORKSPACE_IN_CONTAINER]
    else:
        prefixes = list(dict.fromkeys([str(host_root), str(host_root.resolve())]))
    path = posixpath.normpath(str(attach_path or "").strip()) if attach_path else ""
    if path.startswith("/"):
        for prefix in prefixes:
            prefix = prefix.rstrip("/")
            if path.startswith(prefix + "/"):
                parts = tuple(p for p in path[len(prefix):].split("/") if p)
                if parts and ".." not in parts:
                    return parts
    raise HTTPException(403, "attach_path must be an absolute path to a file inside the "
                             "calling agent's own workspace")


def _read_regular_file_under(root: Path, parts: tuple[str, ...]) -> bytes:
    """Read ``root/parts...`` without following a symlink at ANY component and
    only if it is a regular file. The agent owns its workspace (a Docker agent
    through the bind mount), so it could plant a symlink — or swap one in
    between a check and the read — to point FD at, say, its database; and a
    FIFO would hang the read. Raises OSError otherwise."""
    import stat

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    o_dir = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or os.open not in os.supports_dir_fd:
        # No openat()/O_NOFOLLOW (Windows): resolve, then re-check containment.
        target = root.joinpath(*parts).resolve(strict=True)
        try:
            target.relative_to(root.resolve(strict=True))
        except ValueError:
            raise PermissionError("attachment resolves outside the workspace") from None
        if not target.is_file():
            raise IsADirectoryError("attachment is not a regular file")
        return target.read_bytes()
    fd = os.open(root, os.O_RDONLY | o_dir | nofollow)
    try:
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | o_dir | nofollow, dir_fd=fd)
            os.close(fd)
            fd = nxt
        ffd = os.open(parts[-1], os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0), dir_fd=fd)
    finally:
        os.close(fd)
    with os.fdopen(ffd, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            raise IsADirectoryError("attachment is not a regular file")
        return fh.read()


async def _read_agent_attachment(agent: dict | None, attach_path: str) -> tuple[str, bytes]:
    """``(filename, bytes)`` of a peer-route ``attach_path`` — read ONLY from the
    calling agent's own workspace (never anywhere else FD's OS user can read)."""
    if agent is None:
        raise HTTPException(400, "attach_path needs the calling agent's identity (X-Agent-Auth); "
                                 "pass image_paths/file_paths already on the target instead")
    parts = _workspace_relative_parts(agent, attach_path)
    root = DATA_DIR / agent["slug"] / "data" / "workspace"
    try:
        blob = await asyncio.to_thread(_read_regular_file_under, root, parts)
    except OSError:
        raise HTTPException(404, f"attach_path is not a readable file in the calling agent's "
                                 f"workspace: {attach_path}") from None
    return parts[-1], blob


class ConsultPeerRequest(BaseModel):
    # ``host`` and ``auth`` are accepted from older callers but IGNORED: FD
    # talks to its own agent on ``port`` at the host and token IT recorded.
    host: str = "localhost"
    port: int = 0
    auth: str = ""
    # A remote agent (another Flight Deck's, over the BotPort link): set both
    # and leave ``port`` 0.
    target_deck: str = ""
    target_agent: str = ""
    message: str
    source_name: str = "another agent"
    timeout: float = Field(default=480.0, le=600.0)
    # Agent-to-agent file transfer. The sender passes an absolute path inside
    # its OWN workspace in ``attach_path``; Flight Deck uploads it to the
    # target (with the target's recorded auth token) and forwards the resulting
    # target-local path into the chat payload. image_paths/file_paths may also
    # be passed directly when already uploaded.
    attach_path: str = ""
    image_paths: list[str] = Field(default_factory=list)
    file_paths: list[str] = Field(default_factory=list)
    # When set, the target must NOT evaluate Flow triggers for this message
    # (loop guard: a Flow's agent-step consult shouldn't re-trigger a Flow).
    no_flow: bool = False
    # Tools the target must NOT use for this consult (deterministic guardrail —
    # e.g. an image-describe step denies shell/scripts/read so the model uses
    # the attached image instead of operating on the path).
    deny_tools: list[str] = Field(default_factory=list)
    # When set, the target replies ONLY to this consult (does NOT broadcast its
    # reply to its own channels/UI). Prevents double-delivery when a flow step
    # runs on a channel-connected agent (e.g. the WhatsApp origin agent).
    no_broadcast: bool = False
    # The human-written text that may authorize the target to write email this
    # turn (the caller's narrower of its question and its originating human
    # message / job text); "" = it may not. Stamped as the ``peer`` marker.
    mail_intent_text: str = ""


def _peer_automation(text: str) -> dict[str, Any]:
    """The ``automation`` marker on a peer consult/delegate chat frame: an
    automated turn, judged on the caller-forwarded ``mail_intent_text``."""
    return {"kind": "peer", "job_text": str(text or "")[:4000], "mail_write": "intent"}


# Track active consultations to prevent duplicate requests to the same target
_active_consults: dict[int, str] = {}  # target_port -> source_name
_active_delegates: set[tuple[int, int]] = set()  # (source_port, target_port) in-flight


def _private_relay_level(current: str, marked: Any) -> str:
    """PR D (part 0 §5.4): the privacy level of a peer's reply so far — a final
    assistant frame marked ``"member_private": "data" | "content"`` raises it
    (content beats data); anything else leaves it as it was."""
    if current == "content" or (isinstance(marked, str) and marked == "content"):
        return "content"
    if current == "data" or (isinstance(marked, str) and marked == "data"):
        return "data"
    return ""


def _private_relay_header(level: str) -> str:
    """The header a member-private peer reply carries when FD relays it to
    another agent (``member_privacy.header_for``; imported here so FD loads
    without it — only a peer that marks its replies needs it)."""
    from captain_claw import member_privacy

    return member_privacy.header_for(level)


async def _consult_peer_events(
    host: str, port: int, auth: str, message: str, *,
    source_name: str = "another agent", timeout: float = 480.0,
    image_paths: list[str] | None = None, file_paths: list[str] | None = None,
    no_flow: bool = False, deny_tools: list[str] | None = None, no_broadcast: bool = False,
    automation: dict[str, Any] | None = None, lane: str = "",
):
    """Consult the agent at ``host:port`` (authenticated with ``auth``) and yield
    its intermediate events, then a final ``{"ok": True, "done": True, ...}`` or
    ``{"ok": False, "error": ...}`` dict. A reply the peer marked member-private
    adds ``"member_private": "data" | "content"`` to the final dict; the
    response itself stays plain (callers that relay it to an agent add
    ``_private_relay_header``; FlowRunner's outputs can reach a person).

    No authorization here: the caller vouches for the target. /fd/consult-peer
    resolves host/port/auth from FD's records after `_resolve_peer_caller`; the
    flow engine calls it in-process with agents from its own pool."""
    import websockets

    # NB: a peer serves one consult at a time. Rather than reject when it's
    # busy (which made flows/agents fail or loop), we QUEUE: wait for the
    # in-flight consult to finish, then acquire the lock.

    # Event types we forward as peer activity so the caller can show progress
    _FORWARD_TYPES = {"status", "thinking", "monitor", "tool_stream"}

    from urllib.parse import urlencode

    query = {k: v for k, v in (("token", auth), ("lane", lane)) if v}
    params = f"?{urlencode(query)}" if query else ""
    agent_url = f"ws://{host}:{port}/ws{params}"

    # Queue behind any in-flight consult to this agent (it serves one at a
    # time). Bounded by the request timeout. The check-then-set has no await
    # between, so only one waiter acquires per loop tick (no race).
    _waited = 0.0
    _cap = min(float(timeout), 180.0)
    while _active_consults.get(port) and _waited < _cap:
        if _waited == 0.0:
            yield {"event": "status", "data": {"status": f"Agent on port {port} busy — queuing…"}}
        await asyncio.sleep(1.0)
        _waited += 1.0
    if _active_consults.get(port):
        yield {"ok": False, "error": f"Agent on port {port} still busy after {int(_waited)}s — try again."}
        return
    _active_consults[port] = source_name
    try:
        if await agent_idle.wake_port(port, host):
            yield {"event": "status", "data": {"status": f"Agent on port {port} was asleep — woke it"}}
        # Loopback to FD's own agent: never through an HTTP(S)_PROXY, and room
        # for a long session's welcome replay (it arrives as one frame).
        from captain_claw.flight_deck.botport_link import ws_proxy_kwargs

        async with websockets.connect(agent_url, max_size=64 * 1024 * 1024, **ws_proxy_kwargs(None)) as ws:
            # Wait for welcome
            welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if welcome.get("type") != "welcome":
                yield {"ok": False, "error": "Unexpected handshake"}
                return

            # Skip replay messages
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                msg = json.loads(raw)
                if msg.get("type") == "replay_done":
                    break
                if msg.get("type") not in ("chat_message",) or not msg.get("replay"):
                    break

            _chat_payload: dict[str, Any] = {"type": "chat", "content": message}
            if image_paths:
                _chat_payload["image_paths"] = list(image_paths)
            if file_paths:
                _chat_payload["file_paths"] = list(file_paths)
            if no_flow:
                _chat_payload["no_flow"] = True
            if deny_tools:
                _chat_payload["deny_tools"] = list(deny_tools)
            if no_broadcast:
                _chat_payload["no_broadcast"] = True
            if automation:
                # Automated-turn marker: no user typed this (flow step, peer).
                _chat_payload["automation"] = automation
            await ws.send(json.dumps(_chat_payload))

            # Stream events until we get the final assistant response
            response_parts: list[str] = []
            relay_level = ""  # PR D: the peer marked its reply member-private
            final_usage: dict | None = None  # trailing LLM-usage summary
            deadline = asyncio.get_event_loop().time() + timeout
            recv_interval = 15.0  # heartbeat every 15s of silence
            _busy_retries = 0     # peer is single-threaded; wait it out
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    if not response_parts:
                        yield {"ok": False, "error": "Timed out waiting for response"}
                        return
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, recv_interval))
                except asyncio.TimeoutError:
                    # No message in recv_interval — send heartbeat and keep waiting
                    elapsed = int(timeout - remaining)
                    yield {"event": "heartbeat", "data": {"elapsed": elapsed, "timeout": int(timeout)}}
                    continue
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                # Forward interesting intermediate events
                if msg_type in _FORWARD_TYPES:
                    yield {"event": msg_type, "data": msg}

                if msg_type == "chat_message" and msg.get("role") == "assistant" and not msg.get("replay"):
                    content = msg.get("content", "")
                    if content:
                        response_parts.append(content)
                    relay_level = _private_relay_level(relay_level, msg.get("member_private"))
                    # The agent emits a `usage` summary (model + token counts)
                    # right AFTER the final reply. Drain briefly to capture it
                    # so the done payload can carry per-turn LLM usage; bail the
                    # moment it arrives (usually within ms) so we add no latency.
                    try:
                        for _ in range(4):
                            raw2 = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            m2 = json.loads(raw2)
                            if m2.get("type") == "usage":
                                final_usage = m2
                                break
                            if m2.get("type") in _FORWARD_TYPES:
                                yield {"event": m2.get("type"), "data": m2}
                    except Exception:
                        pass
                    break
                elif msg_type == "error":
                    _err = str(msg.get("message", "Agent error"))
                    # Transient "busy" → wait and re-send on the same socket.
                    if _busy_retries < 8 and ("busy processing" in _err.lower() or "session is busy" in _err.lower()):
                        _busy_retries += 1
                        _wait = min(2 + _busy_retries * 2, 12)
                        yield {"event": "status", "data": {"status": f"Peer busy, retrying ({_busy_retries})…"}}
                        await asyncio.sleep(_wait)
                        try:
                            await ws.send(json.dumps(_chat_payload))
                        except Exception:
                            yield {"ok": False, "error": _err}
                            return
                        continue
                    yield {"ok": False, "error": _err}
                    return

        _done: dict[str, Any] = {
            "ok": True,
            "done": True,
            "response": "\n".join(response_parts) if response_parts else "(no response)",
        }
        if relay_level:
            # PR D: the level only — the agent-facing header is added by the
            # agent-to-agent relays (/fd/consult-peer, FlowRunner's next agent
            # step), never to a flow output a human receives.
            _done["member_private"] = relay_level
        if final_usage is not None:
            _done["usage"] = final_usage  # per-turn LLM token usage for the caller
        yield _done
    except Exception as exc:
        yield {"ok": False, "error": f"Connection failed: {exc}"}
    finally:
        _active_consults.pop(port, None)


@app.post("/fd/consult-peer")
async def consult_peer(req: ConsultPeerRequest, request: Request, user: dict | None = _optional_user_dep):
    """Send a message to a peer agent and stream back intermediate events + final response as ndjson.

    The caller must be allowed to drive the target (`_resolve_peer_caller`);
    FD reaches it at the host and token it recorded, never a caller-named one."""
    import contextlib

    owner, is_admin, caller_agent = _resolve_peer_caller(request, user)
    if req.target_deck or req.target_agent:
        return _remote_consult_response(req.target_deck, req.target_agent, req.message,
                                        await _remote_caller_name(req.source_name, caller_agent, user),
                                        req.timeout, attach_path=req.attach_path)
    auth = _resolve_peer_target(req.port, owner, is_admin)
    attachment = await _read_agent_attachment(caller_agent, req.attach_path) if req.attach_path else None

    async def _event_stream():
        # Transfer the attached file to the target (FD holds its auth).
        _img, _fil = list(req.image_paths), list(req.file_paths)
        if attachment is not None:
            await agent_idle.wake_port(req.port, _PEER_AGENT_HOST)  # asleep: the upload needs it up
            _ti, _tf = await _upload_to_agent(_PEER_AGENT_HOST, req.port, auth, *attachment)
            _img += _ti
            _fil += _tf
        async with contextlib.aclosing(_consult_peer_events(
            _PEER_AGENT_HOST, req.port, auth, req.message,
            source_name=req.source_name, timeout=req.timeout,
            image_paths=_img, file_paths=_fil, no_flow=req.no_flow,
            deny_tools=req.deny_tools, no_broadcast=req.no_broadcast,
            automation=_peer_automation(req.mail_intent_text),
        )) as events:
            async for evt in events:
                _lvl = evt.get("member_private") if evt.get("done") else ""
                if _lvl in ("data", "content"):
                    # Header first, so the receiving agent flags the relayed text too.
                    evt = {**evt, "response": _private_relay_header(_lvl) + "\n" + str(evt.get("response") or "")}
                yield json.dumps(evt) + "\n"

    return StreamingResponse(_event_stream(), media_type="application/x-ndjson")


class DelegatePeerRequest(BaseModel):
    # ``target_host`` / ``source_host`` are accepted from older callers but
    # IGNORED (FD reaches its agents at the host it recorded). The result goes
    # back to the CALLING agent (X-Agent-Auth); ``source_port`` only names the
    # recipient for a bearer caller, which has no agent of its own.
    target_host: str = "localhost"
    target_port: int = 0
    target_name: str = ""
    # A remote agent (another Flight Deck's, over the BotPort link).
    target_deck: str = ""
    target_agent: str = ""
    source_host: str = "localhost"
    source_port: int = 0
    source_name: str = "another agent"
    message: str
    timeout: float = Field(default=600.0, le=1800.0)
    # Origin platform tracking — so results are delivered to the correct session
    origin_platform: str = "web"       # "web" or "telegram"
    origin_user_id: str = ""           # telegram user id
    origin_chat_id: int = 0            # telegram chat id
    # Agent-to-agent file transfer (FD uploads attach_path — a file in the
    # calling agent's own workspace — to the target).
    attach_path: str = ""
    image_paths: list[str] = Field(default_factory=list)
    file_paths: list[str] = Field(default_factory=list)
    # See ConsultPeerRequest.mail_intent_text — stamped as the ``peer`` marker.
    mail_intent_text: str = ""


@app.post("/fd/delegate-peer")
async def delegate_peer(req: DelegatePeerRequest, request: Request, user: dict | None = _optional_user_dep):
    """Fire-and-forget: send a task to a peer agent. When the peer finishes, deliver the result back to the source agent as a chat message."""
    import websockets

    owner, is_admin, caller_agent = _resolve_peer_caller(request, user)
    remote = bool(req.target_deck or req.target_agent)
    target_port = int(req.target_port)
    target_auth = "" if remote else _resolve_peer_target(target_port, owner, is_admin)
    # The result goes back to the calling agent, at the port FD recorded for it.
    if caller_agent is not None and caller_agent["port"]:
        source_port, source_auth = caller_agent["port"], caller_agent["auth"]
    else:
        source_port = int(req.source_port or 0)
        source_auth = _resolve_peer_target(source_port, owner, is_admin)

    if remote:
        return await _remote_delegate(req, source_port, source_auth,
                                      await _remote_caller_name(req.source_name, caller_agent, user))

    peer_display = req.target_name or f"agent@{target_port}"

    # Guard: an agent must not delegate to itself (e.g. a vision agent that
    # tried image_vision, failed, then "delegated" the image to its own name).
    if source_port == target_port:
        return {
            "ok": False,
            "message": f"{peer_display} is the requesting agent itself — handle the task directly, do not delegate to yourself.",
        }

    # Guard: if this source already has a delegation in flight to this target,
    # don't pile on another (the model sometimes re-delegates the same task with
    # a reworded message, which dodges arg-based dedup and storms the peer).
    _deleg_key = (source_port, target_port)
    if _deleg_key in _active_delegates:
        return {
            "ok": True,
            "message": (
                f"A task is already being processed by {peer_display} for you — "
                f"waiting on that result. Do NOT delegate again; tell the user you're waiting."
            ),
        }

    attachment = await _read_agent_attachment(caller_agent, req.attach_path) if req.attach_path else None

    async def _background():
        log.info("delegate_background: started", target=peer_display, source=req.source_name,
                 target_port=target_port, source_port=source_port)

        # Phase 1: send task to target agent and wait for response
        t_params = f"?token={target_auth}" if target_auth else ""
        target_url = f"ws://{_PEER_AGENT_HOST}:{target_port}/ws{t_params}"
        response_text = ""
        relay_level = ""  # PR D: the target marked its reply member-private
        await agent_idle.wake_port(target_port, _PEER_AGENT_HOST)
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

                    _img, _fil = list(req.image_paths), list(req.file_paths)
                    if attachment is not None:
                        _ti, _tf = await _upload_to_agent(_PEER_AGENT_HOST, target_port, target_auth, *attachment)
                        _img += _ti
                        _fil += _tf
                    _payload: dict[str, Any] = {"type": "chat", "content": req.message}
                    if _img:
                        _payload["image_paths"] = _img
                    if _fil:
                        _payload["file_paths"] = _fil
                    _payload["automation"] = _peer_automation(req.mail_intent_text)
                    await ws.send(json.dumps(_payload))
                    log.info("delegate_background: task sent to target", target=peer_display)

                    # Wait for the final response
                    deadline = asyncio.get_event_loop().time() + req.timeout
                    recv_interval = 30.0
                    _busy_retries = 0  # peer is single-threaded; wait it out instead of erroring back
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
                            relay_level = _private_relay_level(relay_level, msg.get("member_private"))
                            log.info("delegate_background: got response from target",
                                     target=peer_display, response_len=len(response_text))
                            break
                        elif msg_type == "error":
                            _err = str(msg.get("message", "Unknown error"))
                            # "Agent is busy" is transient (peer serves one request
                            # at a time). Wait and re-send on the SAME connection
                            # instead of bouncing a fake "result" back to the caller,
                            # which made the caller re-delegate in a loop.
                            if _busy_retries < 8 and ("busy processing" in _err.lower() or "session is busy" in _err.lower()):
                                _busy_retries += 1
                                _wait = min(2 + _busy_retries * 2, 12)
                                log.info("delegate_background: target busy, waiting then retrying",
                                         target=peer_display, attempt=_busy_retries, wait=_wait)
                                await asyncio.sleep(_wait)
                                try:
                                    await ws.send(json.dumps(_payload))
                                except Exception:
                                    response_text = f"[Error from {peer_display}] {_err}"
                                    break
                                continue
                            response_text = f"[Error from {peer_display}] {_err}"
                            log.error("delegate_background: target returned error", target=peer_display, error=_err)
                            break
        except Exception as exc:
            response_text = f"[Error] Could not reach {peer_display}: {exc}"
            log.error("delegate_background: phase 1 failed", target=peer_display, error=str(exc))

        if not response_text:
            response_text = "(no response received)"

        # Phase 2: deliver the result back to the source agent
        await _deliver_delegate_result(req, source_port, source_auth, peer_display,
                                       response_text, relay_level)

    _start_delegate_task(_deleg_key, _background())
    return {"ok": True, "message": f"Task delegated to {peer_display}. Results will be delivered to {req.source_name} when ready."}


def _start_delegate_task(deleg_key: tuple, coro: Any) -> None:
    """Run a delegation in the background, holding a strong reference to it
    and marking (source, target) in flight until it ends."""
    _active_delegates.add(deleg_key)
    task = asyncio.create_task(coro)
    if not hasattr(app.state, "_delegate_tasks"):
        app.state._delegate_tasks = set()
    app.state._delegate_tasks.add(task)

    def _on_delegate_done(t: Any) -> None:
        app.state._delegate_tasks.discard(t)
        _active_delegates.discard(deleg_key)

    task.add_done_callback(_on_delegate_done)


async def _deliver_delegate_result(req: "DelegatePeerRequest", source_port: int, source_auth: str,
                                   peer_display: str, response_text: str, relay_level: str = "") -> None:
    """Hand a delegated task's result back to the agent that delegated it, as
    a notification that triggers its reply (on its originating channel)."""
    import websockets

    s_params = f"?token={source_auth}" if source_auth else ""
    source_url = f"ws://{_PEER_AGENT_HOST}:{source_port}/ws{s_params}"
    callback_msg = (
        f"[Delegated result from {peer_display}] This is the ANSWER you were "
        f"waiting for. Relay it to the user now, concisely, in their language. "
        f"Do NOT delegate again, do NOT call any tool, and do NOT say you are "
        f"still waiting — you already have the result below:\n\n{response_text}"
    )
    if relay_level:
        # Header first (within the receiving agent's header scan), so the
        # source agent flags the delegated result as member-private too.
        callback_msg = _private_relay_header(relay_level) + "\n" + callback_msg
    log.info("delegate_background: delivering result to source",
             source=req.source_name, source_port=source_port, result_len=len(response_text))
    # A long delegation can outlast the source's idle time (agent_idle).
    await agent_idle.wake_port(source_port, _PEER_AGENT_HOST)
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


# ── Remote agents (other Flight Decks, over the BotPort link) ──


async def _remote_caller_name(source_name: str, caller_agent: dict | None, user: dict | None) -> str:
    """How a remote deck sees who is asking: the calling agent's name (from
    FD's own records when it is an FD agent), else the bearer user's name."""
    if caller_agent is not None:
        for a in await asyncio.to_thread(_fleet_snapshot):
            if a.get("port") == caller_agent.get("port") and a.get("kind") == caller_agent.get("kind"):
                return str(a.get("name") or source_name or "agent")
    if user and user.get("display_name"):
        return str(user["display_name"])
    return (source_name or "agent").strip()[:128] or "agent"


def _remote_split(target_deck: str, target_agent: str) -> tuple[str, str]:
    deck, agent = (target_deck or "").strip(), (target_agent or "").strip()
    if not deck and "@" in agent:
        agent, _, deck = agent.rpartition("@")
    if not deck or not agent:
        raise HTTPException(400, "a remote agent needs both target_deck and target_agent "
                                 "(or target_agent written as Agent@deck)")
    return deck, agent


def _remote_consult_response(target_deck: str, target_agent: str, message: str, from_agent: str,
                             timeout: float, *, attach_path: str = "") -> StreamingResponse:
    """/fd/consult-peer for a remote agent: same ndjson contract, carried by the BotPort link."""
    import contextlib

    from captain_claw.flight_deck.botport_link import get_link

    deck, agent = _remote_split(target_deck, target_agent)
    if attach_path:
        raise HTTPException(400, "files cannot be attached to a remote (other Flight Deck) agent yet — "
                                 "put the relevant content in the message instead")

    async def _event_stream():
        async with contextlib.aclosing(get_link().consult(
            target_deck=deck, target_agent=agent, message=message,
            from_agent=from_agent, timeout=timeout, mode="consult",
        )) as events:
            # A member-private reply already carries its header (the remote
            # deck put it first, as /fd/consult-peer does locally).
            async for evt in events:
                yield json.dumps(evt) + "\n"

    return StreamingResponse(_event_stream(), media_type="application/x-ndjson")


async def _remote_delegate(req: "DelegatePeerRequest", source_port: int, source_auth: str,
                           from_agent: str) -> dict[str, Any]:
    """/fd/delegate-peer for a remote agent: consult it over the link in the
    background, then deliver the answer to the calling agent like a local delegation."""
    import contextlib

    from captain_claw.flight_deck.botport_link import get_link

    deck, agent = _remote_split(req.target_deck, req.target_agent)
    if req.attach_path:
        raise HTTPException(400, "files cannot be attached to a remote (other Flight Deck) agent yet — "
                                 "put the relevant content in the message instead")
    link = get_link()
    remote = link.find_remote(deck, agent)
    if remote is None:
        available = ", ".join(r["name"] for r in link.remote_agents()) or "none"
        state = "" if link.connected else f" (BotPort link: {link.state}{': ' + link.last_error if link.last_error else ''})"
        return {"ok": False, "message": f"{agent}@{deck} is not reachable{state}. Available remote agents: {available}"}
    peer_display = remote["name"]  # the agent's own name, e.g. "Researcher@pi"

    deleg_key = (source_port, peer_display.lower())
    if deleg_key in _active_delegates:
        return {
            "ok": True,
            "message": (
                f"A task is already being processed by {peer_display} for you — "
                f"waiting on that result. Do NOT delegate again; tell the user you're waiting."
            ),
        }

    async def _background() -> None:
        response_text, relay_level = "", ""
        async with contextlib.aclosing(link.consult(
            target_deck=deck, target_agent=agent, message=req.message,
            from_agent=from_agent, timeout=req.timeout, mode="delegate",
        )) as events:
            async for evt in events:
                if evt.get("done") and evt.get("ok"):
                    response_text = str(evt.get("response") or "(no response)")
                    relay_level = str(evt.get("member_private") or "")
                elif evt.get("ok") is False:
                    response_text = f"[Error from {peer_display}] {evt.get('error', 'failed')}"
        if not response_text:
            response_text = "(no response received)"
        # The remote deck already put the member-private header first.
        await _deliver_delegate_result(req, source_port, source_auth, peer_display, response_text,
                                       "")
        if relay_level:
            log.info("remote delegate result was member-private", level=relay_level, target=peer_display)

    _start_delegate_task(deleg_key, _background())
    return {"ok": True, "message": f"Task delegated to {peer_display} (via BotPort). "
                                   f"Results will be delivered to {req.source_name} when ready."}


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
    ".mp3", ".wav", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".flac", ".amr",
    ".mp4", ".webm", ".mov", ".m4v", ".mkv", ".avi", ".3gp",
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
    auth = token or _resolve_agent_auth(port, host=host)
    params = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/datastore/tables{params}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                payload = resp.json()
                if AUTH_ENABLED:  # PR C: member creators by their current name
                    payload = await shared_workspace.decorate_owner_payload(
                        getattr(app.state, "fd_db", None), payload)
                return payload
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
    auth = token or _resolve_agent_auth(port, host=host)
    qs = f"token={auth}&" if auth else ""
    qs += f"limit={limit}&offset={offset}"
    if order_by:
        qs += f"&order_by={order_by}&order_dir={order_dir}"
    url = f"http://{host}:{port}/api/datastore/tables/{table_name}/rows?{qs}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                payload = resp.json()
                if AUTH_ENABLED:  # PR C: member creators by their current name
                    payload = await shared_workspace.decorate_owner_payload(
                        getattr(app.state, "fd_db", None), payload)
                return payload
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/agent-datastore/{host}/{port}/tables/{table_name}/export")
async def agent_datastore_export(
    host: str, port: int, table_name: str,
    format: str = "csv", token: str = "",
    user: dict | None = _required_user_dep,
):
    """Proxy datastore export (csv/json/xlsx) from a CC agent."""
    import httpx
    from fastapi.responses import Response as FastAPIResponse
    if format not in ("csv", "json", "xlsx"):
        raise HTTPException(400, f"Unsupported format: {format}")
    auth = token or _resolve_agent_auth(port, host=host)
    qs = f"token={auth}&" if auth else ""
    qs += f"format={format}"
    url = f"http://{host}:{port}/api/datastore/tables/{table_name}/export?{qs}"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return FastAPIResponse(
                    content=resp.content,
                    media_type=resp.headers.get("content-type", "application/octet-stream"),
                    headers={"Content-Disposition": resp.headers.get("content-disposition", f'attachment; filename="{table_name}.{format}"')},
                )
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.post("/fd/agent-rephrase/{host}/{port}")
async def agent_rephrase(
    host: str, port: int, request: Request,
    user: dict | None = _required_user_dep,
):
    """Proxy a rephrase request to a CC agent's LLM."""
    import httpx
    auth = _resolve_agent_auth(port, host=host)
    params = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/llm/complete{params}"
    body = await request.json()
    prompt_text = body.get("content", "")
    payload = {
        "messages": [
            {"role": "system", "content": "You are a prompt engineer. Rephrase the following user prompt to be clearer, more specific, and more effective for an AI agent. Keep the same intent but improve clarity, structure, and specificity. Return only the improved prompt text, nothing else."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url, json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                return {"content": data.get("content", prompt_text)}
            raise HTTPException(resp.status_code, resp.text)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/agent-files/{host}/{port}")
async def agent_files(host: str, port: int, token: str = "", since: str = "", request: Request = None, user: dict | None = _required_user_dep):
    """List files from a CC agent (proxied to avoid CORS), merged with workspace scan.

    The optional ``since=<unix_seconds>`` query param is forwarded to the
    agent's ``/api/files`` endpoint, which filters by mtime. Used by the
    plan-monitor "Files" tab to show files touched during a plan run; the
    workspace-scan merge is skipped when ``since`` is set so unrelated old
    files don't leak through.
    """
    import httpx
    registered: list[dict] = []
    # Auto-resolve auth token if the caller didn't provide one
    auth = token or _resolve_agent_auth(port, host=host)
    qparts = []
    if auth:
        qparts.append(f"token={auth}")
    if since:
        qparts.append(f"since={since}")
    params = ("?" + "&".join(qparts)) if qparts else ""
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
    # PR C: member creators by their current name (before the workspace-scan
    # merge below; scan entries carry no created_by).
    if AUTH_ENABLED and registered:
        registered = await shared_workspace.decorate_owner_payload(
            getattr(app.state, "fd_db", None), registered)

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

    # When the caller filtered by `since`, the agent already applied the
    # mtime filter to its own registry. Merging in unfiltered workspace
    # files would defeat that filter, so we return the agent's view as-is.
    if since:
        return registered

    # Merge: use registered files as base, add workspace files not already listed
    registered_physicals = {f.get("physical", "") for f in registered}
    registered_filenames = {f.get("filename", "") for f in registered}
    for wf in workspace_files:
        if wf["physical"] not in registered_physicals and wf["filename"] not in registered_filenames:
            registered.append(wf)
    return registered


# ── Intentions proxy (FD panel → agent /api/intentions*) ──────────────


async def _proxy_agent_intentions(
    method: str, host: str, port: int, token: str, path: str,
    *, query: dict | None = None, body: dict | None = None, timeout: float = 15.0,
):
    """Forward an intentions request to an agent, preserving its status code."""
    import httpx
    from fastapi.responses import JSONResponse

    params = dict(query or {})
    auth = token or _resolve_agent_auth(port, host=host)
    if auth:
        params["token"] = auth
    url = f"http://{host}:{port}{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, params=params, json=body)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")
    except Exception as exc:
        raise HTTPException(502, f"Agent error: {exc}")
    try:
        payload = resp.json()
    except Exception:
        payload = {"error": resp.text[:500]}
    return JSONResponse(payload, status_code=resp.status_code)


@app.get("/fd/agent-intentions/{host}/{port}")
async def agent_intentions(
    host: str, port: int, token: str = "", status: str = "", origin: str = "",
    user: dict | None = _required_user_dep,
):
    q: dict = {}
    if status:
        q["status"] = status
    if origin:
        q["origin"] = origin
    return await _proxy_agent_intentions("GET", host, port, token, "/api/intentions", query=q)


@app.get("/fd/agent-intentions-decisions/{host}/{port}")
async def agent_intention_decisions(
    host: str, port: int, token: str = "", user: dict | None = _required_user_dep,
):
    return await _proxy_agent_intentions(
        "GET", host, port, token, "/api/intentions/decisions"
    )


@app.post("/fd/agent-intentions-decision/{host}/{port}/{decision_id}/resolve")
async def agent_resolve_intention_decision(
    host: str, port: int, decision_id: str, request: Request,
    token: str = "", user: dict | None = _required_user_dep,
):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await _proxy_agent_intentions(
        "POST", host, port, token,
        f"/api/intentions/decisions/{decision_id}/resolve", body=body,
    )


@app.post("/fd/agent-intention/{host}/{port}/{intention_id}/status")
async def agent_set_intention_status(
    host: str, port: int, intention_id: str, request: Request,
    token: str = "", user: dict | None = _required_user_dep,
):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await _proxy_agent_intentions(
        "POST", host, port, token,
        f"/api/intentions/{intention_id}/status", body=body,
    )


# ── Cron proxy (FD panel → agent /api/cron/jobs*) ─────────────────────


@app.get("/fd/agent-cron/{host}/{port}")
async def agent_cron_list(
    host: str, port: int, token: str = "", user: dict | None = _required_user_dep,
):
    return await _proxy_agent_intentions("GET", host, port, token, "/api/cron/jobs")


@app.post("/fd/agent-cron/{host}/{port}")
async def agent_cron_create(
    host: str, port: int, request: Request, token: str = "",
    user: dict | None = _required_user_dep,
):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await _proxy_agent_intentions(
        "POST", host, port, token, "/api/cron/jobs", body=body
    )


@app.post("/fd/agent-cron/{host}/{port}/{job_id}/run")
async def agent_cron_run(
    host: str, port: int, job_id: str, token: str = "",
    user: dict | None = _required_user_dep,
):
    return await _proxy_agent_intentions(
        "POST", host, port, token, f"/api/cron/jobs/{job_id}/run"
    )


@app.post("/fd/agent-cron/{host}/{port}/{job_id}/pause")
async def agent_cron_pause(
    host: str, port: int, job_id: str, token: str = "",
    user: dict | None = _required_user_dep,
):
    return await _proxy_agent_intentions(
        "POST", host, port, token, f"/api/cron/jobs/{job_id}/pause"
    )


@app.post("/fd/agent-cron/{host}/{port}/{job_id}/resume")
async def agent_cron_resume(
    host: str, port: int, job_id: str, token: str = "",
    user: dict | None = _required_user_dep,
):
    return await _proxy_agent_intentions(
        "POST", host, port, token, f"/api/cron/jobs/{job_id}/resume"
    )


@app.delete("/fd/agent-cron/{host}/{port}/{job_id}")
async def agent_cron_delete(
    host: str, port: int, job_id: str, token: str = "",
    user: dict | None = _required_user_dep,
):
    return await _proxy_agent_intentions(
        "DELETE", host, port, token, f"/api/cron/jobs/{job_id}"
    )


# ── Conversation topics (agent owns the store; FD is a thin client) ──────

@app.get("/fd/agent-topics/{host}/{port}")
async def agent_topics(
    host: str, port: int, token: str = "", q: str = "", limit: int = 300, order: str = "recent",
    group: str = "", tags: str = "", user: dict | None = _required_user_dep,
):
    query: dict = {"limit": limit, "order": order}
    if q:
        query["q"] = q
    if group:
        query["group"] = group
    if tags:
        query["tags"] = tags
    return await _proxy_agent_intentions("GET", host, port, token, "/api/topics", query=query)


@app.get("/fd/agent-topic-groups/{host}/{port}")
async def agent_topic_groups(host: str, port: int, token: str = "", user: dict | None = _required_user_dep):
    return await _proxy_agent_intentions("GET", host, port, token, "/api/topics/groups")


@app.post("/fd/agent-topic-groups/{host}/{port}")
async def agent_topic_group_create(
    host: str, port: int, request: Request, token: str = "", user: dict | None = _required_user_dep,
):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await _proxy_agent_intentions("POST", host, port, token, "/api/topics/groups", body=body)


@app.post("/fd/agent-topic-group-delete/{host}/{port}/{group_id}")
async def agent_topic_group_delete(
    host: str, port: int, group_id: str, token: str = "", user: dict | None = _required_user_dep,
):
    return await _proxy_agent_intentions(
        "POST", host, port, token, f"/api/topics/groups/{group_id}/delete"
    )


@app.post("/fd/agent-topic-set-groups/{host}/{port}/{topic_id}")
async def agent_topic_set_groups(
    host: str, port: int, topic_id: str, request: Request, token: str = "",
    user: dict | None = _required_user_dep,
):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await _proxy_agent_intentions(
        "POST", host, port, token, f"/api/topics/{topic_id}/groups", body=body
    )


@app.post("/fd/agent-topic-append/{host}/{port}/{topic_id}")
async def agent_topic_append(
    host: str, port: int, topic_id: str, request: Request, token: str = "",
    user: dict | None = _required_user_dep,
):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await _proxy_agent_intentions(
        "POST", host, port, token, f"/api/topics/{topic_id}/append", body=body
    )


@app.post("/fd/agent-topic-unclassify/{host}/{port}/{topic_id}")
async def agent_topic_unclassify(
    host: str, port: int, topic_id: str, token: str = "",
    user: dict | None = _required_user_dep,
):
    return await _proxy_agent_intentions(
        "POST", host, port, token, f"/api/topics/{topic_id}/unclassify"
    )


@app.post("/fd/agent-topic-message-move/{host}/{port}/{message_id}")
async def agent_topic_message_move(
    host: str, port: int, message_id: str, request: Request, token: str = "",
    user: dict | None = _required_user_dep,
):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await _proxy_agent_intentions(
        "POST", host, port, token, f"/api/topic-message/{message_id}/move", body=body
    )


@app.post("/fd/agent-topic-star/{host}/{port}/{topic_id}")
async def agent_topic_star(
    host: str, port: int, topic_id: str, request: Request, token: str = "",
    user: dict | None = _required_user_dep,
):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await _proxy_agent_intentions(
        "POST", host, port, token, f"/api/topics/{topic_id}/star", body=body
    )


@app.get("/fd/agent-topic/{host}/{port}/{topic_id}")
async def agent_topic_get(
    host: str, port: int, topic_id: str, token: str = "", limit: int = 60,
    user: dict | None = _required_user_dep,
):
    return await _proxy_agent_intentions(
        "GET", host, port, token, f"/api/topics/{topic_id}", query={"limit": limit}
    )


@app.post("/fd/agent-topics-backfill/{host}/{port}")
async def agent_topics_backfill(
    host: str, port: int, token: str = "", hours: int = 0,
    user: dict | None = _required_user_dep,
):
    return await _proxy_agent_intentions(
        "POST", host, port, token, "/api/topics/backfill", body={"hours": hours}, timeout=120.0
    )


@app.post("/fd/agent-topics-backfill-history/{host}/{port}")
async def agent_topics_backfill_history(
    host: str, port: int, token: str = "", limit: int = 200,
    user: dict | None = _required_user_dep,
):
    return await _proxy_agent_intentions(
        "POST", host, port, token, "/api/topics/backfill-history",
        body={"limit": limit}, timeout=120.0,
    )


@app.post("/fd/agent-topic-refresh/{host}/{port}/{topic_id}")
async def agent_topic_refresh(
    host: str, port: int, topic_id: str, token: str = "",
    user: dict | None = _required_user_dep,
):
    return await _proxy_agent_intentions(
        "POST", host, port, token, f"/api/topics/{topic_id}/refresh", timeout=60.0
    )


@app.post("/fd/agent-topics-reset/{host}/{port}")
async def agent_topics_reset(
    host: str, port: int, request: Request, token: str = "",
    user: dict | None = _required_user_dep,
):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await _proxy_agent_intentions(
        "POST", host, port, token, "/api/topics/reset", body=body
    )


@app.post("/fd/agent-topics-combine/{host}/{port}")
async def agent_topics_combine(
    host: str, port: int, request: Request, token: str = "",
    user: dict | None = _required_user_dep,
):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await _proxy_agent_intentions(
        "POST", host, port, token, "/api/topics/combine", body=body
    )


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
    auth = token or _resolve_agent_auth(port, host=host)
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


# Shared with the agent's own file routes (captain_claw/web/rest_files.py).
from captain_claw.web import active_content as _active_content  # noqa: E402

_ACTIVE_VIEW_TYPES = _active_content.ACTIVE_VIEW_TYPES
_active_view_type = _active_content.active_view_type
_ACTIVE_VIEW_CSP = _active_content.ACTIVE_VIEW_CSP


@app.get("/fd/agent-file-view/{host}/{port}")
async def agent_file_view(host: str, port: int, path: str, token: str = "", request: Request = None, user: dict | None = _required_user_dep):
    """Proxy file view from a CC agent (inline, no download header)."""
    import httpx
    import urllib.parse
    auth = token or _resolve_agent_auth(port, host=host)
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
            headers = {
                "Content-Type": ct,
                "Content-Disposition": "inline",
                "X-Content-Type-Options": "nosniff",
            }
            # PR C (J13): a file a member wrote may be active content — opened
            # directly, it runs sandboxed (opaque origin), never as Flight Deck.
            if _active_view_type(ct):
                headers["Content-Security-Policy"] = _ACTIVE_VIEW_CSP
            return Response(content=resp.content, headers=headers)
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.post("/fd/agent-file-upload/{host}/{port}")
async def agent_file_upload(host: str, port: int, token: str = "", file: UploadFile = File(...), request: Request = None, user: dict | None = _required_user_dep):
    """Proxy file upload to a CC agent."""
    import httpx

    auth = token or _resolve_agent_auth(port, host=host)
    params = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/file/upload{params}"
    content = await file.read()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {"file": (file.filename or "upload", content, file.content_type or "application/octet-stream")}
            resp = await client.post(url, files=files)
            if resp.status_code != 200:
                # Surface the agent's actual error (e.g. "Unsupported file type
                # '.mov'") instead of a bare status code — invaluable for debugging.
                detail = resp.text[:500] if resp.text else ""
                raise HTTPException(resp.status_code, f"Agent upload failed ({resp.status_code}): {detail}")
            # Log usage
            if AUTH_ENABLED and user:
                db = app.state.fd_db
                await db.log_usage(user["id"], "file_upload", json.dumps({"filename": file.filename, "size": len(content), "agent_port": port}))
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.post("/fd/agent-file-save/{host}/{port}")
async def agent_file_save(host: str, port: int, token: str = "", request: Request = None, user: dict | None = _required_user_dep):
    """Proxy a text-file save to a CC agent (POST /api/files/content).

    Body: ``{"path": "<physical>", "content": "<text>"}``. The agent applies
    the same path-sandbox + token auth as its read endpoints."""
    import httpx

    auth = token or _resolve_agent_auth(port, host=host)
    params = f"?token={auth}" if auth else ""
    url = f"http://{host}:{port}/api/files/content{params}"
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=body)
            if resp.status_code != 200:
                detail = resp.text[:500] if resp.text else ""
                raise HTTPException(resp.status_code, f"Agent save failed ({resp.status_code}): {detail}")
            if AUTH_ENABLED and user:
                db = app.state.fd_db
                await db.log_usage(user["id"], "file_save", json.dumps({"path": body.get("path", ""), "agent_port": port}))
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to agent")


@app.get("/fd/agent-usage/{host}/{port}")
async def agent_usage(host: str, port: int, token: str = "", period: str = "today", request: Request = None, user: dict | None = _required_user_dep):
    """Proxy /api/usage from a CC agent."""
    import httpx
    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = _resolve_agent_auth(port, host=host)
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

    auth = _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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
    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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

    auth = token or _resolve_agent_auth(port, host=host)
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
    auth = token or _resolve_agent_auth(port, host=host)
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
    auth = token or _resolve_agent_auth(port, host=host)
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



# (Game proxy routes removed — games hosted directly by FD via games_routes.py)


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
    freebie: bool = False
    # The owner's standing instructions for this agent (their settings) — the
    # SPA syncs its copy from here, so a change made elsewhere (an archetype
    # spawn, an admin) reaches an already-open Flight Deck.
    fleet_instructions: str = ""
    # Stable id of this agent for sharing (agent_sharing): "process:<slug>:<instance>".
    agent_ref: str = ""
    # Stopped for being idle (agent_idle), not by its owner: still on, and it
    # wakes when its chat is opened or something needs it. status is "stopped".
    asleep: bool = False


class ProcessActionResult(BaseModel):
    ok: bool
    slug: str
    message: str = ""
    # See ContainerActionResult.web_auth.
    web_auth: str = ""
    # A spawn's standing instructions as stored for the owner (the archetype's,
    # plus an Advanced setup's rules): what the browser mirrors, not its own copy.
    fleet_instructions: str = ""


@app.get("/fd/processes", response_model=list[ProcessInfo])
async def list_processes(request: Request, user: dict | None = _agent_manager_dep):
    """List all Flight Deck managed process agents.

    ``web_auth`` (the token that drives an agent directly) is only returned to
    FD's own pages and non-browser callers — see
    ``origin_guard.may_expose_agent_secrets``; FD's /fd/agent-* proxies inject
    it server-side for everyone else.
    """
    registry = _load_process_registry()
    user_id = getattr(request.state, "user_id", "")
    # An admin managing someone's agents gets the list, not the tokens that
    # drive them (their dialog never opens a chat).
    expose_auth = (origin_guard.may_expose_agent_secrets(request.headers)
                   and not getattr(request.state, "acting_admin_id", ""))
    instructions = await _owner_agent_instructions(user_id, "process")
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
            web_auth=entry.get("web_auth", "") if expose_auth else "",
            fleet_instructions=instructions.get(slug, ""),
            pid=entry.get("pid") if alive else None,
            provider=entry.get("provider", ""),
            model=entry.get("model", ""),
            freebie=bool(entry.get("freebie", False)),
            agent_ref=agent_sharing.process_ref(slug, entry),
            asleep=not alive and bool(entry.get("asleep")) and not entry.get("stopped"),
        ))
    return result


@app.post("/fd/spawn-process", response_model=ProcessActionResult)
async def spawn_process(config: AgentConfig, request: Request, user: dict | None = _optional_agent_manager_dep):
    """Spawn a new Captain Claw process agent (pip-installed, no Docker)."""
    # Serialise spawns so two concurrent requests can't both land on the same
    # port between the port-pick and the Popen. The lock also covers a small
    # settle delay after launch to let the child's TCPSite.bind succeed (or
    # drift + announce) before the next spawn's port probe runs.
    async with _get_spawn_lock():
        return await _spawn_process_locked(config, request, user)


async def _spawn_process_locked(config: AgentConfig, request: Request, user: dict | None):
    # Owner first: it gates the whole spawn (nothing is written for a caller we
    # can't attribute) and feeds the archetype's Library lookup. `request` may
    # be a lightweight stub (headless/background spawns) — the resolver copes.
    owner_id = await _resolve_spawn_owner(config, request)
    # Resolve an archetype selector (if any) into cognitive_mode/tools/tier/model,
    # then a bare model-recommendation tier to a concrete provider/model.
    await _resolve_archetype(config, request, user)
    _resolve_tier(config)
    await _resolve_spawn_provider_key(config)
    # Rate limiting & agent count check
    if AUTH_ENABLED and user:
        # An admin acting for this user spends their own request / spawn budget;
        # the agent-count cap below stays the owner's.
        _limited = getattr(request.state, "acting_admin", None) or user
        check_api_rate_limit(_limited)
        check_spawn_rate_limit(_limited)
        # Count existing processes for this user
        registry = _load_process_registry()
        user_id = user["id"]
        owned = [e for e in registry.values() if e.get("owner") == user_id]
        await check_agent_count_limit(user, len(owned))

    slug = _slug(config.name)

    # A name is a directory: re-spawning a stopped agent reuses its data dir
    # (workspace, sessions, memory). Another user's name is theirs — refuse it
    # rather than hand this caller that agent's data and registry entry.
    prior_owner = str((_load_process_registry().get(slug) or {}).get("owner") or "")
    if AUTH_ENABLED and prior_owner and owner_id and prior_owner != owner_id:
        raise HTTPException(409, f"An agent named '{slug}' already exists. Choose a different name.")
    # …and a removed agent's folder outlives its entry.
    _refuse_other_owners_folder(slug, owner_id)

    # Check if already running
    if _process_is_alive(slug):
        raise HTTPException(400, f"Process '{slug}' is already running. Stop it first or use a different name.")

    # Ensure port is available; find a free one if not. A stopped or sleeping
    # agent's port is free on the host but still its own: it comes back on it.
    if config.web_enabled and (config.web_port <= 0 or not _is_port_available(config.web_port)
                               or _port_reserved(config.web_port, slug)):
        config.web_port = _find_available_port(config.web_port if config.web_port > 0 else 24080)

    # Auto-generate auth token if none provided — prevents unauthenticated
    # direct access to agent ports bypassing Flight Deck. A caller-chosen one
    # that duplicates another agent's (its FD identity) is replaced. Minted
    # with web disabled too — see _claim_web_auth.
    web_auth_replaced = _claim_web_auth(config, slug)

    # Prepare data directory
    agent_dir = DATA_DIR / slug
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "workspace").mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "skills").mkdir(parents=True, exist_ok=True)
    (agent_dir / "data" / "home-config").mkdir(parents=True, exist_ok=True)
    _record_folder_owner(slug, owner_id)

    # Write config files with local paths
    config_yaml = _build_process_config_yaml(config, agent_dir)
    (agent_dir / "config.yaml").write_text(config_yaml)
    (agent_dir / "data" / "home-config" / "config.yaml").write_text(config_yaml)

    env_content = _build_env(config)
    (agent_dir / ".env").write_text(env_content)

    # Build environment variables (FD's env minus its own secrets)
    base_env = _agent_base_env()
    environment = dict(base_env)
    if env_content:
        for line in env_content.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                environment[k] = v
    for ev in config.env_vars:
        if ev.get("key"):
            environment[ev["key"]] = ev.get("value", "")

    # Tell agents how to reach Flight Deck internally (for Telegram, Discord,
    # Google, etc.) — always THIS deck, whatever the env or env_vars carried.
    _pin_fd_url(environment)

    # Pass owner ID so child agents can propagate ownership when spawning
    if owner_id:
        environment["FD_OWNER_ID"] = owner_id
        # A spawned worker belongs to THIS run's owner, so it must write its VFS
        # files under the owner's root. `vfs_user()` ranks CLAW_VFS_USER above
        # FD_OWNER_ID, and a child inherits the FD server's whole environment — so
        # a global CLAW_VFS_USER (a single-user leftover in the server's .env)
        # would silently funnel EVERY user's run into that one account. Pin it to
        # the run owner so the inherited global can never misdirect it.
        environment["CLAW_VFS_USER"] = owner_id

    # Slug for port-fallback callbacks: lets the agent tell FD which actual
    # port it bound to if the requested one was already in use.
    environment["FD_AGENT_SLUG"] = slug
    _pin_confinement(environment, base_env, agent_dir=agent_dir, owner_id=owner_id or "",
                     workspace=config.workspace_path or "")

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

    # New agents deploy in eco mode by default.
    _write_eco_flag_on_spawn(agent_dir)
    # …and know who they work for (the owner profile — see tenant_profile).
    await tenant_profile.write_on_spawn(agent_dir, owner_id, "process")
    # Never another agent's shared context (packs); a re-spawn of the same
    # agent gets its own back once the registry entry is saved (below).
    await context_packs.remove_files_locked(agent_dir, "process")
    # The New agent dialog's Advanced setup: skills, knowledge and the owner's
    # rules go in before the agent starts (its facts once it is up, below). A
    # failure here costs the setup, never the agent.
    preset_facts: list[dict] = []
    if config.preset:
        try:
            block, preset_facts = await asyncio.to_thread(
                agent_preset.seed, agent_dir,
                Path(config.workspace_path) if config.workspace_path else agent_dir / "data" / "workspace",
                owner_id or "", config.preset)
            config.fleet_instructions = agent_preset.with_block(config.fleet_instructions, block)
        except Exception:
            log.warning("spawn-process: the Advanced setup could not be written", slug=slug, exc_info=True)

    # Open log file
    log_file = agent_dir / "process.log"
    log_fh = open(log_file, "a")

    # Resolve captain-claw-web binary: bundled (PyInstaller) or PATH
    cc_web_bin = _resolve_cc_web_bin()

    # IMPORTANT: write the registry entry BEFORE we Popen the child. The
    # child can bind, drift to a fallback port, and POST to /announce-port
    # in the milliseconds between Popen and the post-Popen registry write —
    # if we wrote the registry after Popen we'd race the announce and
    # silently overwrite the drifted port back to the original (stale)
    # value. Pre-writing the entry also guarantees the announce-port
    # endpoint can find the slug.
    registry = _load_process_registry()
    prior = registry.get(slug) or {}
    registry[slug] = {
        "slug": slug,
        "name": config.name or slug,
        "description": config.description,
        "web_port": config.web_port,
        "web_auth": config.web_auth_token,
        # Re-spawning one's own stopped agent keeps its id (and its members);
        # anything else is a new agent to share.
        "instance_id": agent_sharing.process_instance_for_spawn(prior, owner_id, slug),
        "pid": None,  # filled in below once Popen returns
        "provider": config.provider,
        "model": config.model,
        "tier": config.tier,
        "owner": owner_id,
        # The archetype it was made from (id, no `@tier`): the kiosk's skill
        # cards start from what the agent is for.
        "archetype": config.archetype.strip().partition("@")[0],
        # The tier its model came from, and — on its owner's own API key — that
        # it is (byok.py): its tokens never count against the plan.
        **({"model_tier": config._model_tier} if config._model_tier else {}),
        **({"key_source": "byok"} if config._byok else {}),
        # Deep-memory grid config (empty for non-grid agents) — the FD deep-memory
        # proxy reads these to stamp write-tags and narrow reads without the agent
        # asserting anything (mirrors how `owner` is server-resolved).
        "grid_tags": list(config.grid_memory_tags or []),
        "grid_recall": config.grid_recall_mode or "",
        # A workspace FD placed outside the agent's folder (Code mode's repo):
        # a restart pins the same confinement root (_pin_confinement).
        **({"workspace": config.workspace_path} if config.workspace_path else {}),
    }
    _save_process_registry(registry)
    # A re-spawn keeps its ref, so its context packs: rewrite their files.
    if agent_sharing.sharing_active():
        try:
            from captain_claw.flight_deck.auth import get_db as _get_db

            context_packs.schedule_refresh(
                _get_db(), agent_sharing.process_ref(slug, registry[slug]))
        except Exception as _cp_exc:
            log.warning("Could not schedule a shared-context refresh",
                        error=type(_cp_exc).__name__)

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
    agent_idle.touch(slug)

    # Stamp the live PID. Read-modify-write so we don't clobber a port the
    # child may have already announced between Popen and now.
    registry = _load_process_registry()
    if slug in registry:
        registry[slug]["pid"] = proc.pid
        _save_process_registry(registry)

    # Log usage. Use the auth module's DB (a single instance set at startup),
    # NOT app.state.fd_db: when the server is launched as `python -m ...server`
    # it runs as __main__ and the lifespan sets fd_db on __main__.app, but this
    # function may be imported under the canonical module name (a second app
    # instance whose lifespan never ran) — its app.state.fd_db is unset. The
    # auth DB is the same connection and is always resolvable.
    if AUTH_ENABLED and user:
        db = getattr(app.state, "fd_db", None)
        if db is None:
            from captain_claw.flight_deck.auth import get_db as _get_auth_db
            db = _get_auth_db()
        await db.log_usage(user["id"], "agent_spawn", json.dumps({
            "agent": slug, "type": "process", "provider": config.provider, "model": config.model,
            **_acting_admin_detail(request)}))

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

    if config.fleet_instructions:
        await _set_owner_agent_instructions(owner_id, "process", slug, config.fleet_instructions)
    if preset_facts and config.web_enabled:
        agent_preset.schedule_facts(
            slug, lambda: (_load_process_registry().get(slug) or {}).get("web_port") or config.web_port,
            config.web_auth_token, preset_facts)

    message = f"Process agent '{slug}' spawned (PID {proc.pid}, port {config.web_port})"
    if web_auth_replaced:
        return ProcessActionResult(ok=True, slug=slug, message=message + _WEB_AUTH_REPLACED_NOTE,
                                   web_auth=config.web_auth_token, fleet_instructions=config.fleet_instructions)
    return ProcessActionResult(ok=True, slug=slug, message=message, fleet_instructions=config.fleet_instructions)


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
        # Asleep (agent_idle): switched off for good now.
        registry = _load_process_registry()
        if registry.get(slug, {}).get("asleep"):
            registry[slug].pop("asleep", None)
            registry[slug].pop("asleep_at", None)
            registry[slug]["stopped"] = True
            _save_process_registry(registry)
        return ProcessActionResult(ok=True, slug=slug, message="Already stopped")

    # Our handle only while it is the live process: an agent restarted through
    # another copy of this module (`python -m …server` runs it as __main__, and
    # importers get a second one) leaves a dead handle here — the registry has
    # the pid that is actually running.
    proc = _processes.get(slug)
    pid = proc.pid if proc and proc.poll() is None else _load_process_registry().get(slug, {}).get("pid")

    if pid:
        _kill_pid(pid)

    # Update registry — mark as intentionally stopped. Re-read: the kill can
    # take seconds, and whatever was written meanwhile (a spawn, a port
    # announce) must not be overwritten with the snapshot from before it.
    registry = _load_process_registry()
    if slug in registry:
        registry[slug]["pid"] = None
        registry[slug]["stopped"] = True
        # Off is off: no longer asleep (nothing wakes it).
        registry[slug].pop("asleep", None)
        registry[slug].pop("asleep_at", None)
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

    # Clear stopped flag — user is intentionally starting this agent (or the
    # idle controller is waking it).
    entry.pop("stopped", None)
    entry.pop("asleep", None)
    entry.pop("asleep_at", None)

    if not _start_registered_process(slug, entry):
        raise HTTPException(500, "Failed to start process (captain-claw-web not found?)")

    # Re-read before saving so a drifted-port announce from the child we just
    # started isn't clobbered by our stale in-memory web_port (see announce-port).
    fresh = _load_process_registry()
    f = fresh.get(slug)
    if f is None:
        fresh[slug] = entry
    else:
        f["pid"] = entry.get("pid")
        f.pop("stopped", None)
        f.pop("asleep", None)
        f.pop("asleep_at", None)
    _save_process_registry(fresh)
    agent_idle.touch(slug)

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
async def stop_process(slug: str, request: Request, user: dict | None = _agent_manager_dep):
    """Stop a running process agent."""
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, _do_stop_process, slug)


@app.post("/fd/processes/{slug}/start", response_model=ProcessActionResult)
async def start_process(slug: str, request: Request, user: dict | None = _agent_manager_dep):
    """Start a stopped process agent."""
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    import asyncio
    await _heal_agent_model_key(slug)
    return await asyncio.get_event_loop().run_in_executor(None, _do_start_process, slug)


@app.post("/fd/processes/{slug}/restart", response_model=ProcessActionResult)
async def restart_process(slug: str, request: Request, user: dict | None = _agent_manager_dep):
    """Restart a process agent."""
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    await _heal_agent_model_key(slug)
    _do_stop_process(slug)
    import time
    time.sleep(1)
    return _do_start_process(slug)


@app.delete("/fd/processes/{slug}", response_model=ProcessActionResult)
async def remove_process(slug: str, force: bool = False, purge: bool = False, request: Request = None, user: dict | None = _agent_manager_dep):
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    """Remove a process agent from the registry. Stops it first if running.
    With ``purge`` its folder, chat log and scheduled tasks are deleted too."""
    if _process_is_alive(slug):
        _do_stop_process(slug)

    registry = _load_process_registry()
    owner_id = str((registry.get(slug) or {}).get("owner") or "")
    agent_ref = agent_sharing.process_ref(slug, registry.get(slug) or {})
    if not purge:
        # The folder stays; once the entry is gone only this says whose it is.
        _record_folder_owner(slug, owner_id)
    registry.pop(slug, None)
    _save_process_registry(registry)
    _processes.pop(slug, None)
    # …and its instructions, so a later agent of the same name doesn't inherit them.
    await _set_owner_agent_instructions(owner_id, "process", slug, "")
    await _forget_shared_agent(agent_ref, owner_id)
    await botport_forget_share("process", slug, owner_id)
    if purge:
        problem = await _purge_agent(slug, f"proc-{slug}")
        if problem:
            return ProcessActionResult(ok=False, slug=slug, message=f"Removed '{slug}', but {problem}")
        return ProcessActionResult(ok=True, slug=slug, message=f"Removed '{slug}' and deleted its folder")

    return ProcessActionResult(ok=True, slug=slug, message=f"Removed '{slug}' from registry")


class ProcessIdentityUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


@app.post("/fd/processes/{slug}/identity", response_model=ProcessActionResult)
async def update_process_identity(
    slug: str, body: ProcessIdentityUpdate, request: Request,
    user: dict | None = _agent_manager_dep,
):
    """Persist a process agent's display name and/or description into the
    registry. Unlike the FD-side per-user override, this is canonical: it
    survives a browser wipe, is returned by /fd/processes for every client,
    and needs no restart."""
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    registry = _load_process_registry()
    entry = registry.get(slug)
    if entry is None:
        raise HTTPException(404, f"Process '{slug}' not found")
    if body.name is not None:
        entry["name"] = body.name.strip() or entry.get("name", slug)
    if body.description is not None:
        entry["description"] = body.description
    registry[slug] = entry
    _save_process_registry(registry)
    # The owner's Flight Deck lays its own saved labels over the registry's, so
    # an admin's rename would stay hidden behind them: drop those for this agent.
    if getattr(request.state, "acting_admin_id", ""):
        owner_id = getattr(request.state, "user_id", "")
        for key, changed in zip(_PROCESS_LABEL_SETTINGS, (body.name, body.description)):
            if changed is None:
                continue
            try:
                await _update_owner_map(owner_id, key, slug, None)
            except Exception as exc:
                log.warning("Could not clear the owner's label override", agent=slug, error=str(exc))
    return ProcessActionResult(ok=True, slug=slug, message="Saved.")


@app.get("/fd/processes/{slug}/logs")
async def process_logs(slug: str, tail: int = 200, since_byte: int = 0, request: Request = None, user: dict | None = _agent_manager_dep):
    _verify_process_owner(slug, getattr(request.state, "user_id", ""))
    """Read logs from a process agent's log file.

    When *since_byte* > 0 the response only contains bytes written after
    that offset (incremental fetch).  The response always includes
    ``byte_offset`` – the file size at the time of reading – so the
    frontend can pass it back on the next poll.
    """
    log_file = DATA_DIR / slug / "process.log"
    if not log_file.is_file():
        return {"logs": "(no logs yet)", "byte_offset": 0}
    try:
        file_size = log_file.stat().st_size
        if since_byte > 0 and since_byte <= file_size:
            # Incremental: read only new bytes since last poll
            with open(log_file, "rb") as f:
                f.seek(since_byte)
                new_bytes = f.read()
            text = new_bytes.decode("utf-8", errors="replace")
            # Strip leading partial line if we didn't land on a boundary
            if since_byte > 0 and text and text[0] != "\n" and since_byte < file_size:
                nl = text.find("\n")
                if nl >= 0:
                    text = text[nl + 1:]
            return {"logs": text, "byte_offset": file_size}
        # Initial fetch: efficiently read last N lines from end of file
        chunk_size = max(4096, tail * 200)  # reasonable estimate
        with open(log_file, "rb") as f:
            f.seek(0, 2)  # seek to end
            end_pos = f.tell()
            start = max(0, end_pos - chunk_size)
            f.seek(start)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        # If we didn't read from start, first line may be partial – drop it
        if start > 0 and len(lines) > tail:
            lines = lines[1:]
        tail_lines = lines[-tail:] if len(lines) > tail else lines
        return {"logs": "\n".join(tail_lines), "byte_offset": end_pos}
    except Exception as exc:
        return {"logs": f"(error reading logs: {exc})", "byte_offset": 0}


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

    # A name is a directory (see the spawn guard): only the caller's own stopped
    # agent may be replaced — never another user's (or an unowned) entry and the
    # data dir that goes with it.
    replaced = registry.get(new_slug)
    if replaced is not None:
        caller = str(getattr(request.state, "user_id", "") or "")
        if AUTH_ENABLED and (not caller or str(replaced.get("owner") or "") != caller):
            raise HTTPException(409, f"An agent named '{new_slug}' already exists. Choose a different name.")
        if _process_is_alive(new_slug):
            raise HTTPException(400, f"Process '{new_slug}' already running.")
    clone_owner = str(entry.get("owner") or "")
    _refuse_other_owners_folder(new_slug, clone_owner)

    old_agent_dir = DATA_DIR / slug
    new_agent_dir = DATA_DIR / new_slug

    # Create new data directory
    new_agent_dir.mkdir(parents=True, exist_ok=True)
    (new_agent_dir / "data" / "workspace").mkdir(parents=True, exist_ok=True)
    (new_agent_dir / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (new_agent_dir / "data" / "skills").mkdir(parents=True, exist_ok=True)
    (new_agent_dir / "data" / "home-config").mkdir(parents=True, exist_ok=True)
    _record_folder_owner(new_slug, clone_owner)

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

    # The clone gets its OWN web_auth: that token is an agent's identity to FD
    # (X-Agent-Auth → recorded owner), so two agents sharing one would be
    # indistinguishable — and the clone would keep resolving through the
    # source's entry, or through nothing at all once the source is removed.
    old_auth = str(entry.get("web_auth", "") or "")
    new_auth = secrets.token_urlsafe(32) if old_auth else ""

    # Update config.yaml with new port, name and web auth token
    cfg_path = new_agent_dir / "config.yaml"
    if cfg_path.is_file():
        cfg_text = cfg_path.read_text()
        cfg_text = cfg_text.replace(f"port: {old_port}", f"port: {new_port}")
        if old_auth:
            cfg_text = cfg_text.replace(old_auth, new_auth)
        old_name = entry.get("name", slug)
        if old_name:
            cfg_text = cfg_text.replace(f"instance_name: {old_name}", f"instance_name: {new_name}")
        # Update local paths to point to new agent dir
        cfg_text = cfg_text.replace(str(old_agent_dir), str(new_agent_dir))
        cfg_path.write_text(cfg_text)
        hc_path = new_agent_dir / "data" / "home-config" / "config.yaml"
        if hc_path.is_file():
            hc_path.write_text(cfg_text)

    # Register clone (but don't start it). It belongs to the source's owner —
    # without `owner` it would resolve to no tenant (Google then falls back to
    # the primary owner) and the proxy's ownership guard would let anyone in.
    # The grid fields travel with it too: they scope its deep-memory like owner.
    registry[new_slug] = {
        "slug": new_slug,
        "name": new_name,
        "description": "",
        "web_port": new_port,
        "web_auth": new_auth,
        "instance_id": agent_sharing.new_instance_id(),
        "pid": None,
        "provider": entry.get("provider", ""),
        "model": entry.get("model", ""),
        "tier": entry.get("tier", ""),
        "owner": entry.get("owner", ""),
        "grid_tags": list(entry.get("grid_tags") or []),
        "grid_recall": entry.get("grid_recall", ""),
    }
    _save_process_registry(registry)
    if replaced is not None:  # the clone is a new agent: the replaced one's shares go
        await _forget_shared_agent(agent_sharing.process_ref(new_slug, replaced),
                                   str(replaced.get("owner") or ""))
        await botport_forget_share("process", new_slug, str(replaced.get("owner") or ""))
    # Its owner's profile, not whatever an earlier agent of this slug left.
    await tenant_profile.write_on_spawn(new_agent_dir, str(entry.get("owner") or ""), "process")
    # …and no shared context (packs): a clone is a new agent.
    await context_packs.remove_files_locked(new_agent_dir, "process")

    return ProcessActionResult(ok=True, slug=new_slug, message=f"Cloned '{slug}' → '{new_slug}' (port {new_port})")


async def _get_system_config() -> dict:
    """Load system config from DB. Returns defaults when auth disabled."""
    from captain_claw.flight_deck.admin_routes import _load_system_config
    if not AUTH_ENABLED or not hasattr(app.state, "fd_db"):
        return _load_system_config(None)
    raw = await app.state.fd_db.get_system_setting("fd:system-config")
    return _load_system_config(raw)


@app.get("/fd/auth/status")
async def auth_status():
    """Check if auth is enabled and return system config flags (public endpoint)."""
    from captain_claw.flight_deck.admin_routes import waitlist_cohort

    cfg = await _get_system_config()
    # When running inside a container, default docker spawn to False (no Docker socket)
    docker_default = not os.environ.get("CAPTAIN_CLAW_DOCKER")
    # Provide internal FD URL for agent-to-FD calls (inside container, use localhost)
    internal_fd_url = os.environ.get("FD_INTERNAL_URL", "")
    if not internal_fd_url and os.environ.get("CAPTAIN_CLAW_DOCKER"):
        internal_fd_url = "http://localhost:25080"
    # Kiosk lock for EVERY non-admin (Admin → Settings → Kiosk mode "Everyone";
    # until an admin picks a mode, CLI `--simple-chat` / env FD_SIMPLE_CHAT).
    # Whether a signed-in user is locked comes with the user (`kiosk`); this
    # deck-wide flag is what an auth-off deck goes by.
    # Sign-up: "Create account" while registration is open, else the waitlist
    # (an admin approves applications) when the admin left it on.
    registration_open = False
    logo = ""
    if AUTH_ENABLED and hasattr(app.state, "fd_db"):
        from captain_claw.flight_deck.auth_routes import registration_open as _registration_open
        from captain_claw.flight_deck.branding_routes import logo_url
        registration_open = await _registration_open(app.state.fd_db)
        logo = await logo_url(app.state.fd_db)
    return {
        "auth_enabled": AUTH_ENABLED,
        "docker_spawn_enabled": cfg.get("docker_spawn_enabled", docker_default),
        "internal_fd_url": internal_fd_url,
        "simple_chat_only": cfg.get("kiosk_mode") == "all",
        "registration_open": registration_open,
        "waitlist_open": AUTH_ENABLED and not registration_open and bool(cfg.get("waitlist_enabled")),
        "waitlist_cohort": waitlist_cohort(cfg),
        # Branding (Admin → Settings): the sign-in page and the panels' title.
        "instance_name": cfg.get("instance_name") or "Captain",
        "instance_description": cfg.get("instance_description") or "",
        "instance_logo": logo,
        "show_powered_by": cfg.get("show_powered_by") is not False,
        # The desktop app's own deck (auth._fd_desktop_deck): sign-in off, yet
        # Google and Microsoft connect — through the system browser.
        "desktop_deck": _fd_desktop_deck(),
    }


# ── Agent Forge: LLM-powered team decomposition ──────────────────────────

class ForgeRequest(BaseModel):
    prompt: str
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    base_url: str = ""  # optional custom endpoint (matches the chosen tier)
    # Output token budget for the decomposition. The frontend sends the forge
    # tier's configured output_ctx so a big team's JSON isn't truncated.
    max_tokens: int = 32768
    project_id: str = ""  # optional: forge agents for an existing project


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

    # Inject the curated archetype catalog so the generator bases each agent on a
    # proven shape (inheriting its cognitive_mode, tier, tools, and SOP) instead
    # of inventing every agent from scratch. Built from the same merged registry
    # the gallery reads (base + this user's own archetypes), so it stays in sync;
    # each line leads with the archetype `id` so the model can reference it, and
    # model ids are intentionally omitted (tier resolves them). The id set is
    # captured for post-parse validation (drop hallucinated references).
    valid_archetype_ids: set[str] = set()
    try:
        from captain_claw.flight_deck.archetypes import merged_registry
        from captain_claw.flight_deck.auth import get_db
        reg = await merged_registry(get_db(), user["id"] if user else None)
        if reg:
            tier_names = ", ".join((reg.get("tiers") or {}).keys())
            cat_lines = [
                "\n\n## Archetype Catalog",
                "Base each agent on one of these archetypes when its purpose matches — "
                "set the agent's `archetype` to the entry's `id` and write only a "
                "task-specific `additional_instructions` delta. Define a `new_archetype` "
                "only when nothing here fits.",
                "",
            ]
            for a in reg.get("archetypes", []):
                aid = a.get("id", "")
                if aid:
                    valid_archetype_ids.add(aid)
                # A stored archetype may predate a tool's retirement (gws);
                # never show the model one as a tool a proven archetype uses.
                cat_lines.append(
                    f"- id: `{aid}` — {a['role']} [{a.get('family', '')}] — {a['description']} "
                    f"(mode: {a['cognitive_mode']}, tier: {a['tier']}, "
                    f"tools: {', '.join(without_retired_tools(a.get('tools') or []))})"
                )
            if tier_names:
                cat_lines.append(
                    f"\nValid tiers: {tier_names}. When you set a per-agent `tier` "
                    "override or a `new_archetype.tier`, use one of these. "
                    "Do NOT output model ids — the platform resolves tier→model."
                )
            system_prompt += "\n".join(cat_lines)
    except Exception:
        # Catalog injection is a best-effort bias; never fail a forge over it
        # (e.g. a not-yet-initialized DB in a standalone deployment).
        pass

    # If forging for a project, augment the prompt with project context.
    user_prompt = body.prompt.strip()
    if body.project_id:
        try:
            from captain_claw.projects import get_project_manager
            pm = get_project_manager()
            if pm:
                ctx = await pm.get_project_context(body.project_id)
                if ctx:
                    project = ctx["project"]
                    project_context_parts = [
                        "\n\n## Project Context",
                        f"Project: {project['name']}",
                        f"Status: {project['status']}",
                        f"Description: {project['description']}",
                    ]
                    goals = project.get("goals", [])
                    if goals:
                        project_context_parts.append("\nGoals:")
                        for g in goals:
                            project_context_parts.append(f"- [{g.get('status', 'pending')}] {g['goal']}")
                    members = ctx.get("members", [])
                    if members:
                        project_context_parts.append("\nExisting team members:")
                        for m in members:
                            tags = ", ".join(m.get("expertise_tags", []))
                            project_context_parts.append(
                                f"- {m.get('agent_name') or m['agent_id']} ({m['role']})"
                                + (f" — {tags}" if tags else "")
                            )
                    decisions = ctx.get("decisions", [])
                    if decisions:
                        project_context_parts.append("\nPrior decisions:")
                        for d in decisions[:8]:
                            project_context_parts.append(f"- {d['title']}: {d.get('content', '')[:200]}")
                    project_context_parts.append(
                        "\nDesign the team considering existing members and prior decisions. "
                        "Do not duplicate roles already filled."
                    )
                    user_prompt += "\n".join(project_context_parts)
                    # Log activity.
                    await pm.log_activity(
                        body.project_id, "forge_run",
                        detail={"prompt": body.prompt[:200]},
                    )
        except Exception as exc:
            log.warning("Failed to load project context for forge", error=str(exc))

    # Create an LLM provider and make the decomposition call. Long objectives and
    # big teams (10–15 agents) can outgrow a small output budget, so we floor the
    # cap, detect truncation (finish_reason == "length" across providers), and
    # retry once at a bumped budget before surfacing a clear, actionable error.
    from captain_claw.llm import create_provider, Message

    def _extract_json(raw: str) -> str:
        s = (raw or "").strip()
        if s.startswith("```"):
            s = "\n".join(l for l in s.split("\n") if not l.strip().startswith("```"))
        return s.strip()

    def _truncated(resp) -> bool:
        return str(getattr(resp, "finish_reason", "") or "").lower() in {"length", "max_tokens"}

    async def _run_forge(max_toks: int):
        provider = create_provider(
            provider=body.provider,
            model=body.model,
            api_key=body.api_key or None,
            base_url=body.base_url or None,
            temperature=0.7,
            max_tokens=max_toks,
        )
        return await provider.complete(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            temperature=0.7,
            max_tokens=max_toks,
        )

    # Floor the output budget: 8k+ comfortably fits a 15-agent team of short
    # archetype deltas even when the caller's forge tier is configured low.
    FORGE_TOKEN_FLOOR = 8192
    FORGE_TOKEN_CEIL = 64000
    forge_max_tokens = max(body.max_tokens if body.max_tokens > 0 else 32768, FORGE_TOKEN_FLOOR)

    try:
        response = await _run_forge(forge_max_tokens)
        # Retry once at a larger budget if the model ran out of output room.
        if _truncated(response) and forge_max_tokens < FORGE_TOKEN_CEIL:
            retry_tokens = min(forge_max_tokens * 2, FORGE_TOKEN_CEIL)
            log.warning(
                "Forge output truncated; retrying with larger budget",
                first=forge_max_tokens, retry=retry_tokens,
            )
            try:
                response = await _run_forge(retry_tokens)
                forge_max_tokens = retry_tokens
            except Exception:
                log.warning("Forge retry failed; keeping first response", exc_info=True)
    except Exception as e:
        log.error("Forge LLM call failed", exc_info=True)
        raise HTTPException(502, f"LLM call failed: {e}")

    # Parse the JSON response.
    content = _extract_json(response.content)
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        if _truncated(response):
            raise HTTPException(
                502,
                "The team design was cut off before it finished (hit the output token "
                f"limit at {forge_max_tokens} tokens). Raise the forge tier's output "
                "budget in the Library, or split the objective into a smaller team.",
            )
        raise HTTPException(502, f"LLM returned invalid JSON: {content[:500]}")

    # Drop hallucinated archetype references so the frontend always gets an id it
    # can resolve — or null, meaning bespoke / build from new_archetype. Lenient:
    # an unknown id with no new_archetype simply falls back to bespoke.
    if valid_archetype_ids:
        for agent in (result.get("agents") or []):
            if not isinstance(agent, dict):
                continue
            aid = agent.get("archetype")
            if aid and aid not in valid_archetype_ids:
                log.info(
                    "Forge dropped unknown archetype id",
                    archetype=aid, agent=agent.get("name"),
                )
                agent["archetype"] = None
            # archetype and new_archetype are mutually exclusive; prefer the
            # resolved catalog archetype when the model emitted both.
            if agent.get("archetype") and agent.get("new_archetype"):
                agent["new_archetype"] = None

    # Retired tools (gws) never come back in a per-agent override or a forged
    # new_archetype, so the review screen offers none.
    forged = result.get("agents") if isinstance(result, dict) else None
    for agent in (forged or []):
        if not isinstance(agent, dict):
            continue
        for holder in (agent, agent.get("new_archetype")):
            if isinstance(holder, dict) and isinstance(holder.get("tools"), list):
                holder["tools"] = without_retired_tools(holder["tools"])

    # Tag result with project_id so spawned agents can be auto-joined.
    if body.project_id:
        result["project_id"] = body.project_id

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
    "cron", "summarize_files", "direct_api",
    "google_drive", "google_calendar", "google_mail",
    "microsoft_onedrive", "microsoft_calendar", "microsoft_mail",
    "jev",
    "flight_deck",
]


def _build_old_man_config(
    *,
    name: str = "Old Man",
    description: str = "",
    provider: str = "ollama",
    model: str = "minimax-m2.7:cloud",
    api_key: str = "",
    base_url: str = "",
    web_port: int = 24080,
) -> AgentConfig:
    """Return an AgentConfig pre-filled for supervisor mode.

    The agent keeps the Old Man supervisor tooling (hotkey listener, fleet
    delegation), but its public identity (name/description) is taken from the
    first-run onboarding wizard so the fleet's seed agent can be named by the
    user instead of always being "Old Man".
    """
    return AgentConfig(
        name=(name or "Old Man").strip(),
        description=(description.strip() or "Desktop supervisor — triages requests, delegates to fleet agents"),
        provider=provider,
        model=model,
        provider_api_key=api_key,
        base_url=base_url.strip(),
        tools=OLD_MAN_TOOLS,
        web_port=web_port,
        # Old Man adds old_man.enabled + hotkey overrides via env var
        env_vars=[
            {"key": "CLAW_OLD_MAN__ENABLED", "value": "true"},
            {"key": "CLAW_TOOLS__SCREEN_CAPTURE__HOTKEY_ENABLED", "value": "true"},
            {"key": "FD_URL", "value": _fd_self_url()},
        ],
    )


class OldManSpawnRequest(BaseModel):
    """Request body for the supervisor quick-spawn endpoint.

    ``name``/``description`` let the first-run onboarding wizard name the seed
    supervisor agent; both default to the classic "Old Man" identity.
    """
    name: str = "Old Man"
    description: str = ""
    provider: str = "ollama"
    model: str = "minimax-m2.7:cloud"
    api_key: str = ""
    base_url: str = ""
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
        name=body.name,
        description=body.description,
        provider=body.provider,
        model=body.model,
        api_key=body.api_key,
        base_url=body.base_url,
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


# ── Free OpenRouter ("Freebie") agents ──

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def _is_free_openrouter_model(m: dict) -> bool:
    mid = str(m.get("id", ""))
    if mid.endswith(":free"):
        return True
    pricing = m.get("pricing", {}) or {}

    def _zero(v) -> bool:
        try:
            return float(v) == 0.0
        except Exception:
            return False

    return _zero(pricing.get("prompt")) and _zero(pricing.get("completion")) and _zero(pricing.get("request", "0"))


async def _fetch_openrouter_free_models() -> list[dict]:
    """Fetch OpenRouter's public model list (no key needed) and return the free
    models that support tool calling — Captain Claw agents are tool-heavy, so a
    model without ``tools`` in its supported parameters is useless here."""
    import httpx
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(f"{OPENROUTER_API_BASE}/models")
        r.raise_for_status()
        data = r.json().get("data", []) or []
    out: list[dict] = []
    for m in data:
        if not _is_free_openrouter_model(m):
            continue
        if "tools" not in (m.get("supported_parameters") or []):
            continue
        mid = str(m.get("id", ""))
        ctx = m.get("context_length") or (m.get("top_provider", {}) or {}).get("context_length") or 0
        try:
            ctx = int(ctx or 0)
        except Exception:
            ctx = 0
        out.append({"id": mid, "name": str(m.get("name", mid)), "context_length": ctx})
    out.sort(key=lambda x: x["id"])
    return out


def _free_allowed_entries(model_ids: list[str]) -> list[dict]:
    """Build config.model.allowed entries for a set of free OpenRouter model ids."""
    return [
        {"id": mid, "provider": "openrouter", "model": mid,
         "base_url": OPENROUTER_API_BASE, "model_type": "llm"}
        for mid in model_ids if mid
    ]


def _apply_free_models_to_configs(
    agent_dir: Path, *, default_model: str, model_ids: list[str], set_api_key: str | None = None,
) -> int:
    """Patch model.{provider,model,base_url,allowed} (+ api_key if given) across all
    3 config files for a freebie agent. Returns the number of files updated."""
    allowed = _free_allowed_entries(model_ids)
    config_paths = [
        agent_dir / "config.yaml",
        agent_dir / "data" / "home-config" / "config.yaml",
        agent_dir / "data" / "home-config-parent" / ".captain-claw" / "config.yaml",
    ]
    count = 0
    for cfg_path in config_paths:
        if not cfg_path.is_file():
            continue
        try:
            data = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        if not isinstance(data.get("model"), dict):
            data["model"] = {}
        data["model"]["provider"] = "openrouter"
        data["model"]["model"] = default_model
        data["model"]["base_url"] = OPENROUTER_API_BASE
        if set_api_key is not None:
            data["model"]["api_key"] = set_api_key
        data["model"]["allowed"] = allowed
        cfg_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True))
        count += 1
    return count


@app.get("/fd/openrouter/free-models")
async def openrouter_free_models(user: dict | None = _optional_user_dep):
    """List currently-free OpenRouter models (public list, no key required)."""
    try:
        return {"models": await _fetch_openrouter_free_models()}
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch OpenRouter models: {exc}")


class FreeAgentSpawnRequest(BaseModel):
    name: str = "Freebie"
    description: str = ""
    api_key: str = ""
    default_model: str = ""
    models: list[str] = []  # free model ids to allow (empty = fetch all free)
    web_port: int = 24080


@app.post("/fd/spawn-free")
async def spawn_free_agent(body: FreeAgentSpawnRequest, request: Request, user: dict | None = _optional_user_dep):
    """Spawn a free OpenRouter ("Freebie") process agent: default model + all free
    models as the per-session allowed list, marked freebie for later refresh."""
    if not body.api_key.strip():
        raise HTTPException(400, "An OpenRouter API key is required.")

    model_ids = [m for m in (body.models or []) if m]
    if not model_ids:
        model_ids = [m["id"] for m in await _fetch_openrouter_free_models()]
    if not model_ids:
        raise HTTPException(502, "No free OpenRouter models are available right now.")

    default_model = (body.default_model or model_ids[0]).strip()
    if default_model not in model_ids:
        model_ids = [default_model, *model_ids]

    config = AgentConfig(
        name=body.name or "Freebie",
        description=body.description or "Free OpenRouter agent",
        provider="openrouter",
        model=default_model,
        provider_api_key=body.api_key.strip(),
        base_url=OPENROUTER_API_BASE,
        allowed_models=_free_allowed_entries(model_ids),
        web_port=body.web_port,
    )

    result = await spawn_process(config, request, user)

    # Mark the agent as a freebie so the card can offer "Refresh free models".
    registry = _load_process_registry()
    if result.slug in registry:
        registry[result.slug]["freebie"] = True
        _save_process_registry(registry)

    return result


@app.post("/fd/agent-refresh-free-models/{kind}/{identifier}")
async def refresh_free_models(
    kind: str, identifier: str, request: Request, user: dict | None = _required_user_dep,
):
    """Re-fetch the current free OpenRouter models and rewrite the agent's
    model.allowed list (and default if it dropped off) across all 3 config files.
    The agent must be stopped — config is only read at startup."""
    if kind not in ("docker", "process"):
        raise HTTPException(400, "kind must be 'docker' or 'process'")

    # Guard: agent must be stopped.
    if kind == "process":
        if _process_is_alive(identifier):
            raise HTTPException(409, "Stop the agent before refreshing free models.")
    else:
        try:
            c = get_docker().containers.get(identifier)
            if c.status == "running":
                raise HTTPException(409, "Stop the agent before refreshing free models.")
        except HTTPException:
            raise
        except Exception:
            pass

    user_id = getattr(request.state, "user_id", "")
    agent_dir = _resolve_agent_dir(identifier, kind, user_id)

    free = await _fetch_openrouter_free_models()
    model_ids = [m["id"] for m in free]
    if not model_ids:
        raise HTTPException(502, "No free OpenRouter models are available right now.")

    # Keep the current default if it's still free, otherwise pick the first.
    current_default = ""
    cfg_file = agent_dir / "config.yaml"
    if cfg_file.is_file():
        try:
            current_default = ((yaml.safe_load(cfg_file.read_text()) or {}).get("model", {}) or {}).get("model", "")
        except Exception:
            current_default = ""
    default_model = current_default if current_default in model_ids else model_ids[0]

    updated = _apply_free_models_to_configs(agent_dir, default_model=default_model, model_ids=model_ids)
    return {
        "ok": True,
        "updated": updated,
        "count": len(model_ids),
        "default_model": default_model,
        "message": f"Refreshed {len(model_ids)} free models. Start the agent to use them.",
    }


# ── Static frontend serving ──

if STATIC_DIR.is_dir():
    # Serve built React assets
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    # Prefixes that belong to the backend API surface. If a request
    # targets any of these and we got here, it means no real route
    # matched — return a proper 404 instead of falling through to
    # ``index.html``. Serving HTML with a 200 status for unknown API
    # paths silently breaks clients that ``resp.json()`` the body
    # (e.g. the agent's app_runner tool), producing opaque
    # "Expecting value: line 1 column 1 (char 0)" errors that look
    # like the API returned something nonsensical when really the
    # route just wasn't registered. Fail loudly here so missing /
    # stale routes are obvious.
    _API_PREFIXES = ("fd/", "api/", "ws/", ".well-known/", "authorize")

    @app.get("/{path:path}")
    async def spa_catch_all(path: str, request: Request):
        """Serve the SPA — any non-API path returns index.html."""
        # Don't shadow unknown API routes with HTML.
        if path.startswith(_API_PREFIXES):
            raise HTTPException(status_code=404, detail=f"No route for /{path}")
        if path == "":
            # Meta Ray-Ban Display's WebView → /hud/ (opt out: /?ui=full).
            override = hud_root_override(request, STATIC_DIR / "index.html")
            if override is not None:
                return override
            # Depends on the User-Agent and fd_ui: a cached copy is always revalidated.
            return FileResponse(STATIC_DIR / "index.html", headers=HUD_ROOT_HEADERS)
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
    parser.add_argument(
        "--simple-chat",
        action="store_true",
        help=(
            "Kiosk mode for every non-admin: lock the UI to the chat-first "
            "Simple layout — agents on the left, conversation in the middle, "
            "the active agent's files and data on the right. Strips the "
            "full-view switch, the Spawner, agent config and start/stop "
            "controls so users can only chat, create an agent from a Library "
            "archetype, manage their connections, or sign out. Admins are "
            "exempt. Only the default: once an admin picks a kiosk mode in "
            "Admin → Settings (off / everyone / selected users) that wins. "
            "Also settable via the FD_SIMPLE_CHAT env var."
        ),
    )
    args = parser.parse_args()

    # Kiosk lock default. The deck's kiosk mode (admin_routes._load_system_config)
    # falls back to FD_SIMPLE_CHAT until an admin picks one; the CLI flag simply
    # seeds the env var so the HTTP handlers (and any reload/worker) see it.
    if args.simple_chat:
        os.environ["FD_SIMPLE_CHAT"] = "1"

    # Record the actual bound port so spawned agents get a correct FD_URL
    # callback (_fd_self_url reads FD_PORT; without this it would fall back to
    # a hardcoded 25080 guess). Overwrite, never setdefault: an FD_PORT
    # inherited from the shell or a CWD .env belongs to whatever deck set it,
    # and with several decks on one host it would point this deck's agents at
    # another deck. A different agent-facing URL (e.g. through a local proxy)
    # is what FD_INTERNAL_URL is for.
    os.environ["FD_PORT"] = str(args.port)
    # Likewise the bound host: the desktop deck (auth._fd_desktop_deck) only
    # counts as one while it listens on loopback.
    os.environ["FD_BIND_HOST"] = args.host

    if not args.dev and not STATIC_DIR.is_dir():
        print(f"Warning: Static files not found at {STATIC_DIR}")
        print("Run 'cd flight-deck && npm run build' to build the frontend first.")
        print("Starting API-only mode (use --dev with Vite dev server).\n")

    log.info("Flight Deck starting on http://%s:%s", args.host, args.port)
    # log_config=None: keep our colored formatter from _configure_fd_logging()
    # instead of letting uvicorn slap its default handlers back on.
    # timeout_graceful_shutdown: cap the "waiting for connections to close"
    # phase on Ctrl+C. Without this, in-flight StreamingResponse handlers
    # (consult SSE, ndjson event streams) keep uvicorn waiting indefinitely
    # — the user then has to hit Ctrl+C several times to force-exit.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=None,
        timeout_graceful_shutdown=3,
    )


if __name__ == "__main__":
    # Frozen (PyInstaller) builds: a multiprocessing helper re-runs this
    # binary — without this it starts another full server instead.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
