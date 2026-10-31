"""Self-assessment: what this agent can reach right now, and how its server is.

``run_self_check`` tests, live and in parallel, everything the agent is
connected to, allowed to use and expected to work — its models, Flight Deck,
the owner's Google and Microsoft accounts (each service the tools use), Jev,
Soniox, deep memory, MCP servers, mail, web search, the Telegram and WhatsApp
inboxes — and reports the host it runs on (OS, CPU load, memory, disk, the
process itself, the clock) and which tools the model can see, with the reason
for every hidden one.

Every probe has its own timeout and reads only: a model gets a one-word
prompt, an API a profile or "list one" request, SMTP a login without a
message. Nothing is sent, created or changed, and no secret is ever printed
(keys are shown as their last four characters at most).

Statuses: ``ok`` (tested, works), ``warn`` (works, but degraded or about to
break), ``fail`` (should work and doesn't), ``off`` (not set up — not a
failure), ``info`` (a fact).
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import socket
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from captain_claw.config import get_config
from captain_claw.logging import get_logger

log = get_logger(__name__)

SECTIONS = ("server", "agent", "models", "connections", "tools")
ICONS = {"ok": "✅", "warn": "⚠️", "fail": "❌", "off": "⏸", "info": "•"}
PROBE_TIMEOUT = 20.0
MODEL_TIMEOUT = 45.0
_STARTED = time.time()

# Why the registry hides a tool (its requires_* flag) and where to fix it.
HIDDEN_BECAUSE = {
    "requires_google": "Google not connected (Flight Deck → Connections → Google)",
    "requires_microsoft": "Microsoft 365 not connected (Flight Deck → Connections → Microsoft)",
    "requires_jev": "no Jev key (Flight Deck → Connections → Jev)",
    "requires_soniox": "no Soniox key (Flight Deck → Connections → Soniox)",
    "requires_payments": "payments are off on this Flight Deck",
    "requires_crew": "Crew isn't approved for this account (an admin approves it on the Users page)",
    "requires_telegram_inbox": "no Telegram inbox kept in Flight Deck",
    "requires_whatsapp_web": "WhatsApp not linked in Flight Deck",
    "requires_shared_members": "this agent isn't shared with anyone",
}


@dataclass
class Check:
    section: str
    name: str
    status: str
    detail: str = ""
    ms: int | None = None
    fix: str = ""


def _hint(secret: str) -> str:
    secret = str(secret or "")
    return f"…{secret[-4:]}" if len(secret) >= 8 else "set"


def _bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _span(seconds: float) -> str:
    seconds = int(max(0, seconds))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {seconds % 60}s"


def _connect(what: str, standalone: str = "") -> str:
    """Where the owner sets *what* up: Flight Deck, or the standalone way."""
    from captain_claw.fd_client import is_under_flight_deck

    if is_under_flight_deck():
        return f"connect {what} in Flight Deck → Connections"
    return standalone or f"connect {what} in this agent's settings"


def _http_problem(resp: httpx.Response) -> str:
    text = (resp.text or "").strip().replace("\n", " ")
    return f"HTTP {resp.status_code}" + (f": {text[:160]}" if text else "")


class _Probe:
    """Runs one check with a timeout; whatever it raises becomes a fail."""

    def __init__(self, section: str, name: str, fn: Callable[[], Awaitable[Check | list[Check]]],
                 timeout: float = PROBE_TIMEOUT) -> None:
        self.section, self.name, self.fn, self.timeout = section, name, fn, timeout

    async def run(self) -> list[Check]:
        start = time.monotonic()
        try:
            out = await asyncio.wait_for(self.fn(), timeout=self.timeout)
        except asyncio.TimeoutError:
            return [Check(self.section, self.name, "fail", f"no answer within {self.timeout:.0f}s",
                          int((time.monotonic() - start) * 1000))]
        except Exception as exc:
            return [Check(self.section, self.name, "fail", f"{type(exc).__name__}: {str(exc)[:200]}",
                          int((time.monotonic() - start) * 1000))]
        checks = out if isinstance(out, list) else [out]
        elapsed = int((time.monotonic() - start) * 1000)
        for c in checks:
            if c.ms is None and c.status in ("ok", "warn", "fail"):
                c.ms = elapsed
        return checks


# ── Server ───────────────────────────────────────────────────────────


def _version() -> str:
    try:
        from importlib.metadata import version

        v = version("captain-claw")
    except Exception:
        v = "?"
    commit = ""
    try:
        import subprocess

        root = Path(__file__).resolve().parent.parent
        if (root / ".git").exists():
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        commit = ""
    return f"{v}" + (f" (commit {commit})" if commit else "")


def server_checks(workspace: Path | None) -> list[Check]:
    out: list[Check] = []
    try:
        import psutil  # optional: richer process and memory figures
    except Exception:
        psutil = None  # type: ignore[assignment]
    cpus = os.cpu_count() or 1
    os_name = platform.system()
    release = platform.mac_ver()[0] if os_name == "Darwin" else platform.release()
    host = f"{socket.gethostname()} · {os_name} {release} ({platform.machine()}) · {cpus} CPUs"
    if psutil is not None:
        try:
            host += f" · up {_span(time.time() - psutil.boot_time())}"
        except Exception:
            pass
    out.append(Check("server", "Host", "info", host))
    out.append(Check("server", "Software", "info",
                     f"Captain {_version()} · Python {platform.python_version()}"
                     + (" · in Docker" if Path("/.dockerenv").exists() else "")))
    try:
        load1, load5, load15 = os.getloadavg()
        status = "warn" if load5 > cpus * 1.5 else "ok"
        out.append(Check("server", "CPU load", status,
                         f"{load1:.2f} / {load5:.2f} / {load15:.2f} (1/5/15 min) on {cpus} CPUs",
                         fix="the host is overloaded — slow replies and timeouts are likely"
                         if status == "warn" else ""))
    except (OSError, AttributeError):
        pass
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            free_pct = vm.available / vm.total * 100 if vm.total else 100
            status = "fail" if free_pct < 5 else "warn" if free_pct < 12 else "ok"
            out.append(Check("server", "Memory", status,
                             f"{_bytes(vm.available)} free of {_bytes(vm.total)} ({free_pct:.0f}%)",
                             fix="free memory or add RAM; processes may be killed" if status != "ok" else ""))
        except Exception:
            pass
    disk_root = workspace if workspace and workspace.exists() else Path.home()
    try:
        du = shutil.disk_usage(disk_root)
        free_pct = du.free / du.total * 100 if du.total else 100
        status = ("fail" if du.free < 512 * 1024 ** 2 else
                  "warn" if du.free < 2 * 1024 ** 3 or free_pct < 8 else "ok")
        out.append(Check("server", "Disk", status,
                         f"{_bytes(du.free)} free of {_bytes(du.total)} ({free_pct:.0f}%) "
                         "on the workspace volume",
                         fix="free disk space — saving files and sessions will start failing"
                         if status != "ok" else ""))
    except OSError as exc:
        out.append(Check("server", "Disk", "fail", str(exc)))
    proc = f"PID {os.getpid()} · running {_span(time.time() - _STARTED)}"
    if psutil is not None:
        try:
            me = psutil.Process()
            proc += (f" · {_bytes(me.memory_info().rss)} memory · {me.num_threads()} threads"
                     f" · CPU {me.cpu_percent(interval=0.1):.0f}%")
        except Exception:
            pass
    out.append(Check("server", "This process", "info", proc))
    return out


# ── Agent ────────────────────────────────────────────────────────────


def agent_checks(agent: Any, workspace: Path | None, saved_root: Path | None) -> list[Check]:
    from captain_claw.fd_client import flight_deck_slug, is_under_flight_deck

    cfg = get_config()
    out: list[Check] = []
    try:
        from captain_claw.personality import load_personality

        name = load_personality().name
    except Exception:
        name = "?"
    mode = (f"run by Flight Deck as '{flight_deck_slug() or '?'}'" if is_under_flight_deck()
            else "standalone")
    session = getattr(getattr(agent, "session", None), "id", "") or ""
    out.append(Check("agent", "Identity", "info",
                     f"{name} · {mode}" + (f" · session {session[:8]}" if session else "")))
    if saved_root is not None:
        probe = saved_root / "tmp" / f".self-check-{os.getpid()}"
        try:
            probe.parent.mkdir(parents=True, exist_ok=True)
            probe.write_text("ok", encoding="utf-8")
            ok = probe.read_text(encoding="utf-8") == "ok"
            probe.unlink()
            out.append(Check("agent", "Workspace", "ok" if ok else "fail",
                             f"{workspace or saved_root.parent} is writable"))
        except OSError as exc:
            out.append(Check("agent", "Workspace", "fail", f"can't write files: {exc}",
                             fix="check the folder's permissions and free space"))
    for label, raw in (("Session store", cfg.session.path), ("Datastore", cfg.datastore.path)):
        path = Path(str(raw)).expanduser()
        if not path.exists():
            out.append(Check("agent", label, "off" if label == "Datastore" else "warn",
                             "not created yet"))
            continue
        try:
            import sqlite3

            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            try:
                tables = conn.execute(
                    "select count(*) from sqlite_master where type='table'").fetchone()[0]
            finally:
                conn.close()
            out.append(Check("agent", label, "ok", f"{_bytes(path.stat().st_size)}, {tables} tables"))
        except Exception as exc:
            out.append(Check("agent", label, "fail", f"can't open: {exc}"))
    return out


# ── Models ───────────────────────────────────────────────────────────


async def _ping(provider: Any) -> str:
    from captain_claw.llm import Message

    resp = await provider.complete([Message(role="user", content="Reply with the single word OK.")],
                                   max_tokens=16)
    return (getattr(resp, "content", "") or "").strip()


def model_probes(agent: Any) -> list[_Probe]:
    cfg = get_config()
    probes: list[_Probe] = []
    live = getattr(agent, "provider", None)

    if live is not None:
        label = f"{getattr(live, 'provider', '?')}/{getattr(live, 'model', '?')}"

        async def main() -> Check:
            start = time.monotonic()
            reply = await _ping(live)
            ms = int((time.monotonic() - start) * 1000)
            status = "ok" if reply else "warn"
            return Check("models", f"In use: {label}", status,
                         f"answered in {ms} ms" if reply else "answered, but with no text",
                         ms)

        probes.append(_Probe("models", f"In use: {label}", main, MODEL_TIMEOUT))

    running = (str(getattr(live, "provider", "")), str(getattr(live, "model", "")))
    for entry in cfg.model.allowed:
        kind = str(getattr(entry, "model_type", "llm") or "llm").lower()
        name = f"{entry.id}: {entry.provider}/{entry.model}"
        if (entry.provider, entry.model) == running:
            continue
        if kind not in ("llm", "multimodal", "vision"):
            probes.append(_Probe("models", name, _const(Check(
                "models", name, "info", f"{kind} model — configured, not tested"))))
            continue
        if str(entry.provider).lower().startswith("litert"):
            probes.append(_Probe("models", name, _const(Check(
                "models", name, "info", "local model — not loaded just for a test"))))
            continue
        probes.append(_Probe("models", name, _allowed_ping(agent, entry, name), MODEL_TIMEOUT))
    return probes


def _const(check: Check) -> Callable[[], Awaitable[Check]]:
    async def fn() -> Check:
        return check
    return fn


def _allowed_ping(agent: Any, entry: Any, name: str) -> Callable[[], Awaitable[Check]]:
    async def fn() -> Check:
        from captain_claw.llm import create_provider

        cfg = get_config()
        norm = agent._normalize_provider_key(entry.provider) if hasattr(
            agent, "_normalize_provider_key") else str(entry.provider).lower()
        cfg_norm = agent._normalize_provider_key(str(cfg.model.provider)) if hasattr(
            agent, "_normalize_provider_key") else str(cfg.model.provider).lower()
        headers = cfg.provider_keys.headers_for(norm) or None
        if headers or norm in ("ollama", "litert"):
            key = None
        elif norm == cfg_norm:
            key = cfg.model.api_key or None
        else:
            resolver = getattr(agent, "_resolve_provider_api_key", None)
            key = resolver(norm) if callable(resolver) else None
            if not key:
                return Check("models", name, "fail", f"no API key for {entry.provider}",
                             fix=f"add the {entry.provider} key (provider keys) or remove the model")
        base = (entry.base_url or (cfg.model.base_url if norm == cfg_norm else "")) or None
        provider = create_provider(provider=entry.provider, model=entry.model, api_key=key,
                                   base_url=base, max_tokens=16, extra_headers=headers)
        start = time.monotonic()
        reply = await _ping(provider)
        ms = int((time.monotonic() - start) * 1000)
        return Check("models", name, "ok" if reply else "warn",
                     f"answered in {ms} ms" if reply else "answered, but with no text", ms)
    return fn


# ── Connections ──────────────────────────────────────────────────────


async def _internet() -> Check:
    async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
        resp = await client.get("https://www.google.com/generate_204")
    if resp.status_code in (200, 204):
        return Check("connections", "Internet", "ok", "reachable")
    return Check("connections", "Internet", "warn", f"unexpected answer: {_http_problem(resp)}")


async def _flight_deck(clock: dict[str, float]) -> Check:
    from captain_claw.fd_client import flight_deck_base, flight_deck_headers, is_under_flight_deck

    if not is_under_flight_deck():
        return Check("connections", "Flight Deck", "off", "standalone agent (no Flight Deck)")
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{flight_deck_base()}/fd/health", headers=flight_deck_headers())
    if resp.status_code != 200:
        return Check("connections", "Flight Deck", "fail", _http_problem(resp),
                     fix="Flight Deck isn't answering — most tools that need your accounts will fail")
    date = resp.headers.get("date")
    if date:
        try:
            clock["skew"] = time.time() - parsedate_to_datetime(date).timestamp()
        except Exception:
            pass
    return Check("connections", "Flight Deck", "ok", "reachable")


def _clock_check(clock: dict[str, float]) -> Check:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    skew = clock.get("skew")
    if skew is None:
        return Check("server", "Clock", "info", now)
    status = "warn" if abs(skew) > 60 else "ok"
    return Check("server", "Clock", status, f"{now} · {skew:+.0f}s against Flight Deck",
                 fix="sync the clock (NTP) — sign-ins and schedules depend on it"
                 if status == "warn" else "")


async def _google() -> list[Check]:
    from captain_claw.google_oauth_manager import GoogleOAuthManager
    from captain_claw.session import get_session_manager

    mgr = GoogleOAuthManager(get_session_manager())
    if not await mgr.is_connected():
        return [Check("connections", "Google", "off", "not connected",
                      fix=_connect("Google") + " to use Gmail, Calendar and Drive")]
    tokens = await mgr.get_tokens()
    if tokens is None or not tokens.access_token:
        return [Check("connections", "Google", "fail", "connected, but no access token came back",
                      fix="re" + _connect("Google"))]
    headers = {"Authorization": f"Bearer {tokens.access_token}"}
    services = (
        ("Gmail", "https://gmail.googleapis.com/gmail/v1/users/me/profile", "emailAddress"),
        ("Calendar", "https://www.googleapis.com/calendar/v3/users/me/calendarList?maxResults=1", ""),
        ("Drive", "https://www.googleapis.com/drive/v3/about?fields=user", ""),
    )
    out: list[Check] = []
    async with httpx.AsyncClient(timeout=12) as client:
        results = await asyncio.gather(*(client.get(url, headers=headers) for _, url, _ in services),
                                       return_exceptions=True)
    for (label, _, field), resp in zip(services, results):
        name = f"Google {label}"
        if isinstance(resp, Exception):
            out.append(Check("connections", name, "fail", f"{type(resp).__name__}: {resp}"))
            continue
        if resp.status_code == 200:
            who = ""
            try:
                body = resp.json()
                who = body.get(field, "") if field else (body.get("user", {}) or {}).get("emailAddress", "")
            except Exception:
                pass
            out.append(Check("connections", name, "ok", f"works{f' ({who})' if who else ''}"))
        elif resp.status_code == 403 and "scope" in resp.text.lower():
            out.append(Check("connections", name, "warn", "connected, but this permission wasn't granted",
                             fix=f"reconnect Google and allow {label}"))
        elif resp.status_code == 403 and "has not been used" in resp.text.lower():
            out.append(Check("connections", name, "fail", "the API is not enabled for this Google project",
                             fix=f"enable the {label} API in the Google Cloud console"))
        else:
            out.append(Check("connections", name, "fail", _http_problem(resp),
                             fix="re" + _connect("Google")))
    return out


async def _microsoft() -> list[Check]:
    from captain_claw.microsoft_oauth_manager import MicrosoftOAuthManager

    mgr = MicrosoftOAuthManager()
    if not await mgr.is_connected():
        return [Check("connections", "Microsoft 365", "off", "not connected",
                      fix=_connect("Microsoft 365", "Microsoft 365 needs Flight Deck")
                      + " for Outlook, Calendar and OneDrive")]
    tokens = await mgr.get_tokens()
    if tokens is None or not tokens.access_token:
        return [Check("connections", "Microsoft 365", "fail", "connected, but no access token came back",
                      fix="re" + _connect("Microsoft 365"))]
    headers = {"Authorization": f"Bearer {tokens.access_token}"}
    graph = "https://graph.microsoft.com/v1.0"
    services = (
        ("Account", f"{graph}/me?$select=userPrincipalName"),
        ("Outlook mail", f"{graph}/me/mailFolders/inbox?$select=id"),
        ("Calendar", f"{graph}/me/calendars?$top=1&$select=id"),
        ("OneDrive", f"{graph}/me/drive?$select=id"),
    )
    out: list[Check] = []
    async with httpx.AsyncClient(timeout=12) as client:
        results = await asyncio.gather(*(client.get(url, headers=headers) for _, url in services),
                                       return_exceptions=True)
    for (label, _), resp in zip(services, results):
        name = f"Microsoft {label}"
        if isinstance(resp, Exception):
            out.append(Check("connections", name, "fail", f"{type(resp).__name__}: {resp}"))
        elif resp.status_code == 200:
            who = ""
            if label == "Account":
                try:
                    who = resp.json().get("userPrincipalName", "")
                except Exception:
                    pass
            out.append(Check("connections", name, "ok", f"works{f' ({who})' if who else ''}"))
        elif resp.status_code == 403:
            out.append(Check("connections", name, "warn", "connected, but this permission wasn't granted",
                             fix=f"reconnect Microsoft and allow {label}"))
        else:
            out.append(Check("connections", name, "fail", _http_problem(resp),
                             fix="re" + _connect("Microsoft 365")))
    return out


async def _jev() -> Check:
    from captain_claw import jev_client

    await jev_client.refresh_connected(force=True)
    if not jev_client.jev_usable():
        return Check("connections", "Jev", "off", "no key",
                     fix=_connect("a Jev key", "set tools.jev.api_key or JEV_API_KEY"))
    if not jev_client.under_flight_deck():
        models = await jev_client.standalone_api().models()
        return Check("connections", "Jev", "ok", f"key works ({len(models)} models)")
    return Check("connections", "Jev", "ok", "key available through Flight Deck")


async def _soniox() -> Check:
    from captain_claw import soniox_client

    if await soniox_client.available():
        return Check("connections", "Soniox (speech)", "ok", "key available")
    return Check("connections", "Soniox (speech)", "off", "no key",
                 fix=_connect("a Soniox key", "set SONIOX_API_KEY") + " for transcription and voice")


async def _flag(name: str, module: str, refresh: str, usable: str, off_text: str) -> Check:
    import importlib

    from captain_claw.fd_client import is_under_flight_deck

    if not is_under_flight_deck():
        return Check("connections", name, "off", "comes through Flight Deck")

    mod = importlib.import_module(f"captain_claw.{module}")
    fn = getattr(mod, refresh)
    try:
        await fn(force=True)
    except TypeError:
        await fn()
    if getattr(mod, usable)():
        return Check("connections", name, "ok", "available")
    return Check("connections", name, "off", off_text)


async def _deep_memory(agent: Any) -> Check:
    from captain_claw.fd_client import FDClient, is_under_flight_deck

    name = "Deep memory"
    if is_under_flight_deck():
        token = str(getattr(get_config().web, "auth_token", "") or "")
        fd = FDClient(timeout=15)
        try:
            resp = await fd.post("/fd/deep-memory/agent/search",
                                 json={"query": "self check", "max_results": 1, "filter_by": ""},
                                 headers={"X-Agent-Auth": token} if token else None)
        finally:
            await fd.close()
        if resp.status_code == 200:
            return Check("connections", name, "ok", "search works (through Flight Deck)")
        if resp.status_code in (404, 409, 503):
            return Check("connections", name, "off", f"not set up on Flight Deck ({resp.status_code})",
                         fix="configure Typesense in Flight Deck to give agents long-term memory")
        return Check("connections", name, "fail", _http_problem(resp))
    dm = getattr(agent, "_deep_memory", None)
    cfg = get_config().deep_memory
    if dm is None or not cfg.enabled:
        return Check("connections", name, "off", "not enabled")
    url = f"{cfg.protocol}://{cfg.host}:{cfg.port}/health"
    async with httpx.AsyncClient(timeout=8) as client:
        resp = await client.get(url, headers={"X-TYPESENSE-API-KEY": cfg.api_key})
    if resp.status_code == 200:
        return Check("connections", name, "ok", f"Typesense healthy at {cfg.host}:{cfg.port}")
    return Check("connections", name, "fail", _http_problem(resp))


async def _mcp() -> list[Check]:
    from captain_claw.fd_client import FDClient, is_under_flight_deck
    from captain_claw.tools.mcp_connector import MCPProxyConnector

    if not is_under_flight_deck():
        return [Check("connections", "MCP servers", "off", "MCP servers come through Flight Deck")]
    fd = FDClient(timeout=15)
    try:
        resp = await fd.get("/fd/mcp/agent/servers")
        if resp.status_code != 200:
            return [Check("connections", "MCP servers", "fail", _http_problem(resp))]
        servers = [str(s.get("name") or "") for s in (resp.json() or {}).get("servers", [])
                   if isinstance(s, dict) and s.get("name")]
        if not servers:
            return [Check("connections", "MCP servers", "off", "none allowed for this agent")]

        async def one(name: str) -> Check:
            try:
                tools = await asyncio.wait_for(MCPProxyConnector(name, fd).discover_tools(), 15)
                return Check("connections", f"MCP {name}", "ok", f"{len(tools)} tools")
            except Exception as exc:
                return Check("connections", f"MCP {name}", "fail", str(exc)[:200],
                             fix="reconnect this server in Flight Deck → Connections → MCP")

        return list(await asyncio.gather(*(one(s) for s in servers)))
    finally:
        await fd.close()


async def _mail() -> Check:
    cfg = get_config().tools.send_mail
    provider = str(cfg.provider or "smtp").lower()
    name = "Outgoing mail (send_mail)"
    if provider == "mailgun":
        if not (cfg.mailgun_api_key and cfg.mailgun_domain):
            return Check("connections", name, "off", "Mailgun not configured")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{cfg.mailgun_base_url.rstrip('/')}/domains/{cfg.mailgun_domain}",
                                    auth=("api", cfg.mailgun_api_key))
        return Check("connections", name, "ok" if resp.status_code == 200 else "fail",
                     f"Mailgun {cfg.mailgun_domain}" if resp.status_code == 200 else _http_problem(resp))
    if provider == "sendgrid":
        if not cfg.sendgrid_api_key:
            return Check("connections", name, "off", "SendGrid not configured")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.sendgrid.com/v3/scopes",
                                    headers={"Authorization": f"Bearer {cfg.sendgrid_api_key}"})
        return Check("connections", name, "ok" if resp.status_code == 200 else "fail",
                     "SendGrid key works" if resp.status_code == 200 else _http_problem(resp))
    if not cfg.smtp_username or not cfg.from_address:
        return Check("connections", name, "off", "SMTP not configured")

    def smtp_login() -> str:
        import smtplib

        with smtplib.SMTP(cfg.smtp_host, int(cfg.smtp_port), timeout=10) as server:
            server.ehlo()
            if cfg.smtp_use_tls:
                server.starttls()
                server.ehlo()
            server.login(cfg.smtp_username, cfg.smtp_password)
        return f"logged in to {cfg.smtp_host}:{cfg.smtp_port} as {cfg.smtp_username} (nothing sent)"

    detail = await asyncio.to_thread(smtp_login)
    return Check("connections", name, "ok", detail)


async def _web_search(deep: bool) -> Check:
    cfg = get_config().tools.web_search
    provider = str(cfg.provider or "brave").lower()
    key = {"tavily": cfg.tavily_api_key, "youcom": cfg.youcom_api_key}.get(provider, cfg.api_key)
    key = key or os.environ.get({"tavily": "TAVILY_API_KEY", "youcom": "YDC_API_KEY"}.get(
        provider, "BRAVE_API_KEY"), "")
    name = "Web search"
    if not key:
        return Check("connections", name, "fail", f"{provider} has no API key",
                     fix=f"set the {provider} search key — web_search can't work without it")
    if not deep:
        return Check("connections", name, "ok", f"{provider} key {_hint(key)} (run deep=true to query it)")
    from captain_claw.tools.web_search import WebSearchTool

    result = await WebSearchTool().execute(query="Captain Claw", count=1)
    if result.success:
        return Check("connections", name, "ok", f"{provider} answered a test query")
    return Check("connections", name, "fail", str(result.error or result.content)[:200])


async def _botport(agent: Any) -> Check:
    cfg = get_config().botport
    if not cfg.enabled:
        return Check("connections", "BotPort", "off", "not enabled")
    client = getattr(agent, "_botport_client", None)
    if client is not None and getattr(client, "connected", False):
        return Check("connections", "BotPort", "ok", "connected")
    return Check("connections", "BotPort", "fail", "enabled but not connected",
                 fix="check the BotPort URL and key")


async def _telegram_bot() -> Check:
    cfg = get_config().telegram
    if not cfg.enabled or not cfg.bot_token:
        return Check("connections", "Telegram bot", "off", "no bot of its own")
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{cfg.api_base_url.rstrip('/')}/bot{cfg.bot_token}/getMe")
    if resp.status_code == 200 and (resp.json() or {}).get("ok"):
        return Check("connections", "Telegram bot", "ok",
                     f"@{(resp.json().get('result') or {}).get('username', '?')}")
    return Check("connections", "Telegram bot", "fail", f"HTTP {resp.status_code}",
                 fix="the bot token was rejected — check it with @BotFather")


async def _browser(deep: bool) -> Check:
    try:
        from playwright.async_api import async_playwright
    except Exception:
        return Check("connections", "Browser", "off", "Playwright isn't installed")
    if not deep:
        return Check("connections", "Browser", "ok", "Playwright installed (run deep=true to launch it)")
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        version = browser.version
        await browser.close()
    return Check("connections", "Browser", "ok", f"Chromium {version} starts")


def connection_probes(agent: Any, deep: bool, clock: dict[str, float]) -> list[_Probe]:
    probes = [
        _Probe("connections", "Internet", _internet, 12),
        _Probe("connections", "Flight Deck", lambda: _flight_deck(clock), 12),
        _Probe("connections", "Google", _google, 25),
        _Probe("connections", "Microsoft 365", _microsoft, 25),
        _Probe("connections", "Jev", _jev),
        _Probe("connections", "Soniox (speech)", _soniox),
        _Probe("connections", "Telegram inbox", lambda: _flag(
            "Telegram inbox", "telegram_inbox_client", "refresh_connected", "inbox_usable",
            "none kept (Flight Deck → Connections → Telegram)")),
        _Probe("connections", "WhatsApp", lambda: _flag(
            "WhatsApp", "whatsapp_web_client", "refresh_connected", "usable",
            "not linked (Flight Deck → Connections → WhatsApp)")),
        _Probe("connections", "Payments", lambda: _flag(
            "Payments", "payments_client", "refresh", "payments_usable", "off on this Flight Deck")),
        _Probe("connections", "Crew (coding CLIs)", lambda: _flag(
            "Crew", "crew_client", "refresh", "crew_usable", "not approved for this account")),
        _Probe("connections", "Deep memory", lambda: _deep_memory(agent)),
        _Probe("connections", "MCP servers", _mcp, 40),
        _Probe("connections", "Outgoing mail (send_mail)", _mail, 20),
        _Probe("connections", "Web search", lambda: _web_search(deep), 25),
        _Probe("connections", "BotPort", lambda: _botport(agent), 5),
        _Probe("connections", "Telegram bot", _telegram_bot, 12),
        _Probe("connections", "Browser", lambda: _browser(deep), 40),
    ]
    return probes


# ── Tools ────────────────────────────────────────────────────────────


def tool_checks(agent: Any, session_id: str) -> list[Check]:
    registry = getattr(agent, "tools", None)
    if registry is None:
        return [Check("tools", "Tools", "info", "no tool registry to inspect")]
    cfg = get_config()
    registered = set(getattr(registry, "_tools", {}) or {})
    try:
        visible = set(registry.list_tools(session_id=session_id or None))
    except Exception:
        visible = set(registered)
    meta = getattr(registry, "_tool_metadata", {}) or {}
    hidden: dict[str, list[str]] = {}
    for name in sorted(registered - visible):
        flags = [HIDDEN_BECAUSE[k] for k, v in (meta.get(name) or {}).items() if v and k in HIDDEN_BECAUSE]
        if not flags and name == "typesense":
            flags = ["deep memory isn't set up (no Typesense key or local index)"]
        hidden.setdefault(flags[0] if flags else "hidden by a tool policy", []).append(name)
    mcp = sorted(n for n in visible if n.startswith("mcp_"))
    out = [Check("tools", "Available", "ok",
                 f"{len(visible)} tools the model can use"
                 + (f" ({len(mcp)} from MCP servers)" if mcp else ""))]
    for reason, names in hidden.items():
        out.append(Check("tools", "Hidden", "off", f"{', '.join(names)} — {reason}"))
    from captain_claw.config import RETIRED_TOOLS

    known_aliases = {"web_fetch", "facts", "terminal"}
    missing = [n for n in cfg.tools.enabled
               if n not in registered and n not in RETIRED_TOOLS and n not in known_aliases]
    if missing:
        out.append(Check("tools", "Not loaded", "warn",
                         f"enabled in config but not loaded: {', '.join(missing)}",
                         fix="misspelled or unknown tool names in the agent's tool list"))
    return out


# ── Running ──────────────────────────────────────────────────────────


async def run_self_check(
    agent: Any,
    *,
    sections: list[str] | None = None,
    deep: bool = False,
    session_id: str = "",
    workspace: Path | None = None,
    saved_root: Path | None = None,
) -> list[Check]:
    wanted = [s for s in (sections or SECTIONS) if s in SECTIONS] or list(SECTIONS)
    clock: dict[str, float] = {}
    probes: list[_Probe] = []
    if "models" in wanted:
        probes += model_probes(agent)
    if "connections" in wanted or "server" in wanted:
        probes += [p for p in connection_probes(agent, deep, clock)
                   if "connections" in wanted or p.name == "Flight Deck"]
    # Models in parallel with everything else, but at most four at a time.
    limiter = asyncio.Semaphore(4)

    async def guarded(p: _Probe) -> list[Check]:
        if p.section == "models":
            async with limiter:
                return await p.run()
        return await p.run()

    results = await asyncio.gather(*(guarded(p) for p in probes))
    checks: list[Check] = []
    if "server" in wanted:
        checks += await asyncio.to_thread(server_checks, workspace)
        checks.append(_clock_check(clock))
    if "agent" in wanted:
        checks += await asyncio.to_thread(agent_checks, agent, workspace, saved_root)
    for group in results:
        checks += [c for c in group if c.section in wanted]
    if "tools" in wanted:
        checks += tool_checks(agent, session_id)
    order = {s: i for i, s in enumerate(SECTIONS)}
    checks.sort(key=lambda c: order.get(c.section, 99))
    return checks


_TITLES = {"server": "Server", "agent": "Agent", "models": "Models",
           "connections": "Connections", "tools": "Tools"}


def format_report(checks: list[Check], *, took: float) -> str:
    counts = {k: sum(1 for c in checks if c.status == k) for k in ("ok", "warn", "fail", "off")}
    lines = [
        f"Self-check · {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} · took {took:.1f}s",
        f"{counts['ok']} ok · {counts['warn']} warning(s) · {counts['fail']} failing · "
        f"{counts['off']} not set up",
    ]
    attention = [c for c in checks if c.status in ("fail", "warn")]
    if attention:
        lines += ["", "Needs attention:"]
        for c in sorted(attention, key=lambda c: c.status != "fail"):
            lines.append(f"{ICONS[c.status]} {c.name} — {c.detail}" + (f" → {c.fix}" if c.fix else ""))
    section = ""
    for c in checks:
        if c.section != section:
            section = c.section
            lines += ["", _TITLES.get(section, section)]
        timing = f" ({c.ms} ms)" if c.ms is not None and c.status in ("ok", "warn") else ""
        fix = f" → {c.fix}" if c.fix and c.status == "off" else ""
        lines.append(f"{ICONS.get(c.status, '•')} {c.name}: {c.detail}{timing}{fix}")
    return "\n".join(lines)
