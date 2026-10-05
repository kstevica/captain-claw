"""Shared-agent speakers (A1: chat-only shared agent).

An owner can share an agent with other Flight Deck users ("members"). A
member never talks to this process directly: Flight Deck opens the agent's
``/ws`` socket on the member's behalf and adds ONE header,
``X-FD-Speaker: v1.<payload_b64>.<sig_b64>`` — an HMAC-signed assertion of
who is speaking. Everything speaker-specific in the agent hangs off a socket
that passed :func:`verify_assertion` (``ws._speaker_key``) or an Agent built
for such a socket (``agent._speaker_scoped is True``). Without the header
nothing here is reachable, so a deck with sharing off is unaffected.

The signing key is derived from the agent's own web auth token
(``config.web.auth_token``), which Flight Deck already holds for the agent:
no new secret store, nothing to respawn, and the key never travels.

A member turn may only use :data:`SPEAKER_TOOL_ALLOWLIST`. That is enforced
in ``ToolRegistry.execute`` from three independent signals (the
:data:`CURRENT` contextvar, a registered speaker session key, and
``arguments["_agent"]._speaker_scoped``), so a lost contextvar or a tool path
that passes only a session id still fails closed.
"""

from __future__ import annotations

import base64
import contextvars
import hashlib
import hmac
import ipaddress
import json
import os
import socket
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpcore
import httpx

# ── Shared constants (contract part 0 §8) ────────────────────────────

MEMBER_LANES = ("A", "B", "C")
SPEAKER_HEADER = "X-FD-Speaker"
SPEAKER_TOOL_ALLOWLIST = frozenset(
    {"insights", "playbooks", "topics", "web_search", "web_fetch"}
)
SPEAKER_FRAME_ALLOWLIST = frozenset({
    "chat", "cancel", "btw", "message_feedback", "session_settings",
    "set_playbook", "approval_response", "fd_speaker_context",
})
SPEAKER_SLASH_ALLOWLIST = frozenset({
    "/help", "/h", "/clear", "/new", "/compact", "/history",
    "/stop", "/cancel", "/session",
})
# Bare `/session` (info) plus the two that only touch the member's own session.
SPEAKER_SESSION_SUBCOMMANDS = frozenset({"", "rename", "description"})


def _env_int(name: str, default: int) -> int:
    try:
        value = int(str(os.environ.get(name, "") or "").strip() or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


SPEAKER_MAX_INSTANCES = _env_int("CLAW_SPEAKER_MAX_INSTANCES", 32)
SPEAKER_MAX_PER_USER = 3
SPEAKER_IDLE_EVICT_S = 1800
MAX_CHAT_CONTENT = 100_000
# A speaker turn refreshes the commons caches (insights, intuitions) when
# they are older than this.
SPEAKER_CACHE_REFRESH_S = 600

NOT_ALLOWED_MESSAGE = "Not available on a shared agent"
# What a member sees when their turn fails: never the exception text, which
# can carry the owner's provider errors, LLM base URLs or key fragments.
TURN_FAILED_MESSAGE = "The agent couldn't answer that — try again."

# Contract part 0 §6: every `error` frame on the member wire carries one.
ERROR_CODES = frozenset({"not_allowed", "busy", "capacity", "invalid"})

# Bounds on what a member can park in the agent's memory between turns.
MAX_BTW_INSTRUCTIONS = 20
SESSION_SETTINGS_LIMITS = {
    "session_name": 200,
    "session_description": 2000,
    "session_instructions": 8000,
}
# Sessions one member may have on this agent (all lanes together).
MAX_SESSIONS_PER_SPEAKER = 20

SPEAKER_MODE_NOTE = (
    "This is a shared agent and you are talking with a member, not your owner. "
    "In this conversation you can only search the web, read public web pages, and "
    "read or add the agent's shared insights, playbooks and topics. You cannot run "
    "commands, use files, Google, deep memory, MCP servers or other agents, or "
    "schedule anything — say so if asked. Never reveal other people's conversations."
)

# Assertion limits (contract part 0 §5).
_ASSERTION_VERSION = "v1"
_KEY_LABEL = b"captain-claw/fd-speaker/v1"
_IAT_SKEW_S = 30
_MAX_LIFETIME_S = 120
_NONCE_TTL_S = 180
_MAX_HEADER_LEN = 8192
_MAX_FIELD_LEN = 512


# ── Principal + contextvar ──────────────────────────────────────────


@dataclass(frozen=True)
class Principal:
    """The verified member a socket (and its turns) speaks for."""

    speaker_id: str
    display_name: str
    owner_name: str
    lane: str
    agent_ref: str


CURRENT: ContextVar[Principal | None] = ContextVar("claw_speaker", default=None)


def current() -> Principal | None:
    """The principal bound to the running task, if any."""
    return CURRENT.get()


def bind(p: Principal | None) -> contextvars.Token:
    """Bind *p* for the current task; pair with :func:`reset`."""
    return CURRENT.set(p)


def reset(tok: contextvars.Token) -> None:
    try:
        CURRENT.reset(tok)
    except (ValueError, RuntimeError):
        # Token from another context (shouldn't happen) — never let cleanup
        # raise out of a turn's `finally`.
        CURRENT.set(None)


def principal_for(agent: Any = None) -> Principal | None:
    """The bound principal, else the one recorded on a speaker agent."""
    p = current()
    if p is not None:
        return p
    if agent is not None and getattr(agent, "_speaker_scoped", False) is True:
        rec = getattr(agent, "_speaker_principal", None)
        if isinstance(rec, Principal):
            return rec
    return None


def speaker_key_of(ws: Any) -> tuple[str, str] | None:
    """``(speaker_id, lane)`` for a verified speaker socket, else None.

    Type-checked so a mock or an unrelated attribute never reads as a
    speaker socket.
    """
    key = getattr(ws, "_speaker_key", None)
    if isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, str) for k in key):
        return key
    return None


def is_speaker_agent(agent: Any) -> bool:
    return agent is not None and getattr(agent, "_speaker_scoped", False) is True


def speaker_error(code: str, message: str) -> dict[str, Any]:
    """An ``error`` frame for a member socket, always with a contract code
    (part 0 §6); an unknown *code* becomes ``invalid``."""
    return {
        "type": "error",
        "code": code if code in ERROR_CODES else "invalid",
        "message": message,
    }


# Bound for a tool call that the registry recognised as a member's by its
# session key or `_agent` alone (no principal in context), so the tool's own
# member rules (public-only web_fetch, hidden sources/excerpts) still apply.
UNKNOWN_PRINCIPAL = Principal(
    speaker_id="", display_name="Member", owner_name="", lane="", agent_ref="",
)


def session_name_reserved(name: Any) -> bool:
    """Session names the agent itself looks sessions up by (the owner's
    ``default`` session, lane ``lane-<X>`` sessions). A member may not take
    them, or the owner's lane/default agent could adopt the member's session."""
    n = str(name or "").strip().lower()
    return n == "default" or n.startswith("lane-")


# ── Member sessions seen from the owner's side ──────────────────────

MEMBER_SESSION_REFUSAL = "That's a member's private conversation on this shared agent."


def is_member_session(session: Any) -> bool:
    """A shared-agent member's private session (tagged at creation)."""
    meta = getattr(session, "metadata", None)
    return isinstance(meta, dict) and bool(meta.get("speaker_id"))


def member_session_refusal(session: Any) -> str | None:
    """Why the owner's agent may not switch into / load *session*, or None.

    Adopting it would make two instances save the same session row, and the
    owner's turns would land in the member's transcript.
    """
    return MEMBER_SESSION_REFUSAL if is_member_session(session) else None


_OWNER_SCAN_MAX = 2000


async def list_owner_sessions(sm: Any, limit: int = 20) -> list[Any]:
    """``sm.list_sessions(limit)`` without members' sessions, so the owner's
    session lists and their ``#N`` indices never count them."""
    limit = max(1, int(limit))
    fetch = limit
    while True:
        rows = await sm.list_sessions(limit=fetch)
        owned = [s for s in rows if not is_member_session(s)]
        if len(owned) >= limit or len(rows) < fetch or fetch >= _OWNER_SCAN_MAX:
            return owned[:limit]
        fetch = min(fetch * 2 + 20, _OWNER_SCAN_MAX)


async def select_owner_session(sm: Any, selector: str) -> Any:
    """``sm.select_session`` as the owner sees it: an exact id still resolves
    (callers refuse a member's with :func:`member_session_refusal`), a name
    prefers the owner's own session of that name, and ``#N`` / ``N`` index
    :func:`list_owner_sessions`."""
    key = str(selector or "").strip()
    if not key:
        return None
    by_id = await sm.load_session(key)
    if by_id is not None:
        return by_id
    by_name = await sm.load_session_by_name(key)
    if by_name is not None:
        if not is_member_session(by_name):
            return by_name
        # The newest of that name is a member's: the owner's own, if any.
        for s in await list_owner_sessions(sm, limit=_OWNER_SCAN_MAX):
            if s.name == key:
                return s
        return by_name
    index_text = key[1:] if key.startswith("#") else key
    if not index_text.isdigit() or int(index_text) <= 0:
        return None
    index = int(index_text)
    sessions = await list_owner_sessions(sm, limit=max(20, index))
    return sessions[index - 1] if index <= len(sessions) else None


# ── Assertion verification ──────────────────────────────────────────


class SpeakerAuthError(Exception):
    """The X-FD-Speaker assertion is missing, malformed, forged, stale or replayed."""


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def speaker_signing_key(web_auth: str) -> bytes:
    """HMAC-SHA256(key=web_auth, msg="captain-claw/fd-speaker/v1")."""
    return hmac.new(web_auth.encode("utf-8"), _KEY_LABEL, hashlib.sha256).digest()


def sign_assertion(payload: dict[str, Any], web_auth: str) -> str:
    """Build the header value for *payload* (FD's side; used by tests)."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload_b64 = _b64url(body.encode("ascii"))
    signed = f"{_ASSERTION_VERSION}.{payload_b64}"
    sig = hmac.new(speaker_signing_key(web_auth), signed.encode("ascii"), hashlib.sha256).digest()
    return f"{signed}.{_b64url(sig)}"


def speaker_ack_for(header_value: str) -> str:
    """The ``welcome.speaker_ack`` that proves this agent understood the header."""
    return hashlib.sha256(header_value.encode("ascii")).hexdigest()[:16]


# nonce → time after which the entry may be forgotten. Process-wide: a
# replayed header is refused on ANY socket of this agent.
_NONCES: dict[str, float] = {}
_NONCE_LOCK = threading.Lock()


def _str_field(payload: dict[str, Any], name: str, *, required: bool) -> str:
    value = payload.get(name, "")
    if not isinstance(value, str) or len(value) > _MAX_FIELD_LEN:
        raise SpeakerAuthError(f"bad field: {name}")
    if required and not value.strip():
        raise SpeakerAuthError(f"missing field: {name}")
    return value


def _int_field(payload: dict[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpeakerAuthError(f"bad field: {name}")
    return value


def verify_assertion(header_value: str, web_auth: str, *, now: float | None = None) -> Principal:
    """Verify an ``X-FD-Speaker`` header and return its principal.

    Contract part 0 §5: version ``v1``; constant-time signature check;
    ``iat - 30 <= now <= exp``; ``exp - iat <= 120``; lane in A/B/C; nonce
    never seen before (kept 180 s). Raises :class:`SpeakerAuthError` on any
    failure. The header value never appears in an error or a log line.
    """
    if not isinstance(web_auth, str) or not web_auth:
        raise SpeakerAuthError("agent has no web auth token")
    if not isinstance(header_value, str) or not header_value or len(header_value) > _MAX_HEADER_LEN:
        raise SpeakerAuthError("malformed assertion")
    try:
        header_value.encode("ascii")
    except UnicodeEncodeError:
        raise SpeakerAuthError("malformed assertion") from None
    parts = header_value.split(".")
    if len(parts) != 3 or parts[0] != _ASSERTION_VERSION or not parts[1] or not parts[2]:
        raise SpeakerAuthError("malformed assertion")
    _, payload_b64, sig_b64 = parts

    expected = hmac.new(
        speaker_signing_key(web_auth),
        f"{_ASSERTION_VERSION}.{payload_b64}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(sig_b64.encode("ascii"), _b64url(expected).encode("ascii")):
        raise SpeakerAuthError("bad signature")

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise SpeakerAuthError("malformed payload") from None
    if not isinstance(payload, dict):
        raise SpeakerAuthError("malformed payload")
    if payload.get("v") != 1 or isinstance(payload.get("v"), bool):
        raise SpeakerAuthError("unsupported version")

    sub = _str_field(payload, "sub", required=True)
    name = _str_field(payload, "name", required=False)
    _str_field(payload, "owner", required=False)
    owner_name = _str_field(payload, "owner_name", required=False)
    ref = _str_field(payload, "ref", required=False)
    _str_field(payload, "conn", required=False)
    nonce = _str_field(payload, "nonce", required=True)
    lane = payload.get("lane")
    if lane not in MEMBER_LANES:
        raise SpeakerAuthError("bad lane")
    iat = _int_field(payload, "iat")
    exp = _int_field(payload, "exp")

    t = time.time() if now is None else float(now)
    if exp < iat or exp - iat > _MAX_LIFETIME_S:
        raise SpeakerAuthError("bad lifetime")
    if t < iat - _IAT_SKEW_S:
        raise SpeakerAuthError("assertion from the future")
    if t > exp:
        raise SpeakerAuthError("assertion expired")

    with _NONCE_LOCK:
        for seen, until in list(_NONCES.items()):
            if until < t:
                _NONCES.pop(seen, None)
        if nonce in _NONCES:
            raise SpeakerAuthError("replayed assertion")
        _NONCES[nonce] = t + _NONCE_TTL_S

    return Principal(
        speaker_id=sub,
        display_name=name.strip() or "Member",
        owner_name=owner_name.strip(),
        lane=lane,
        agent_ref=ref,
    )


# ── Slash commands ──────────────────────────────────────────────────


def slash_allowed(text: str) -> bool:
    """Whether a member may run this slash command (base + /session subcommand)."""
    parts = str(text or "").strip().split()
    if not parts:
        return False
    base = parts[0].lower()
    if base not in SPEAKER_SLASH_ALLOWLIST:
        return False
    if base == "/session":
        sub = parts[1].lower() if len(parts) > 1 else ""
        return sub in SPEAKER_SESSION_SUBCOMMANDS
    return True


def speaker_commands() -> list[dict]:
    """``web_server.COMMANDS`` narrowed to what a member may run."""
    from captain_claw.web_server import COMMANDS

    return [dict(c) for c in COMMANDS if slash_allowed(c.get("command", ""))]


# ── Per-tool argument rules (contract part 2b §7) ───────────────────

_INSIGHTS_ACTIONS = frozenset({"search", "list", "add"})
_PLAYBOOKS_ACTIONS = frozenset({"add", "list", "search", "info", "rate"})
_TOPICS_ACTIONS = frozenset({"list", "search", "get"})


def _action_of(arguments: dict) -> str:
    raw = arguments.get("action", "")
    return raw if isinstance(raw, str) else ""


def check_public_url(url: Any) -> str | None:
    """An error when *url* isn't a plain public-looking http(s) URL, else None.

    The address itself is checked at connect time (every hop) by
    :class:`_PublicOnlyBackend`; this only rejects what can never be fetched.
    """
    if not isinstance(url, str) or not url.strip() or len(url) > 8192:
        return "A URL is required."
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return "That URL could not be parsed."
    if parts.scheme.lower() not in ("http", "https"):
        return "Only http(s) URLs can be fetched on a shared agent."
    if not parts.hostname:
        return "The URL has no host."
    if parts.username is not None or parts.password is not None or "@" in (parts.netloc or ""):
        return "URLs with credentials can't be fetched on a shared agent."
    return None


def apply_tool_rules(name: str, arguments: dict, agent: Any) -> tuple[dict, str | None]:
    """Narrow one allowlisted tool call for a member; ``(args, error)``.

    One rule per allowlisted tool. An allowlisted name without a rule is
    refused, so growing the allowlist without a rule fails closed.
    """
    args = dict(arguments or {})
    action = _action_of(args)

    if name == "insights":
        if action.strip().lower() in ("update", "delete"):
            return args, "Only the agent's owner can change or delete insights."
        if action not in _INSIGHTS_ACTIONS:
            return args, NOT_ALLOWED_MESSAGE
        return args, None

    if name == "playbooks":
        if action.strip().lower() in ("update", "remove"):
            return args, "Only the agent's owner can change or remove playbooks."
        if action not in _PLAYBOOKS_ACTIONS:
            return args, NOT_ALLOWED_MESSAGE
        session = getattr(agent, "session", None) if is_speaker_agent(agent) else None
        sid = getattr(session, "id", None)
        own_sid = sid if isinstance(sid, str) and sid else None
        if action == "rate":
            # A member rates THEIR conversation only — never another session.
            if own_sid is None:
                return args, "Rating is not available here."
            args["session_id"] = own_sid
        elif action == "add":
            # The recorded source of a member's playbook is their own session,
            # never one they name.
            if own_sid is None:
                args.pop("session_id", None)
            else:
                args["session_id"] = own_sid
            # Nor may it link the owner's scripts (they would be injected
            # alongside the playbook into everyone's prompt).
            args.pop("script_ids", None)
        return args, None

    if name == "topics":
        if action not in _TOPICS_ACTIONS:
            return args, NOT_ALLOWED_MESSAGE
        return args, None

    if name == "web_search":
        return args, None

    if name == "web_fetch":
        args["deep_fetch"] = False
        err = check_public_url(args.get("url"))
        if err:
            return args, err
        return args, None

    return args, NOT_ALLOWED_MESSAGE


# ── SSRF-safe HTTP client for member web_fetch ──────────────────────

_NAT64 = ipaddress.ip_network("64:ff9b::/96")
# IPv6 ranges Python 3.11's `is_global` still calls global but that are never
# a public web server: IPv4-compatible (::/96, deprecated — wraps loopback and
# private v4), site-local (fec0::/10, deprecated) and local-use NAT64
# (64:ff9b:1::/48, RFC 8215 — translates to private v4 on the local network).
_NON_PUBLIC_V6 = (
    ipaddress.ip_network("::/96"),
    ipaddress.ip_network("fec0::/10"),
    ipaddress.ip_network("64:ff9b:1::/48"),
)


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Globally routable unicast only. Embedded IPv4 (mapped, 6to4, Teredo,
    NAT64) must itself be public, so a wrapped loopback can't slip through."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return _is_public_ip(ip.ipv4_mapped)
        if any(ip in net for net in _NON_PUBLIC_V6):
            return False
        if ip.sixtofour is not None and not _is_public_ip(ip.sixtofour):
            return False
        if ip.teredo is not None:
            server, client = ip.teredo
            if not (_is_public_ip(server) and _is_public_ip(client)):
                return False
        if ip in _NAT64 and not _is_public_ip(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)):
            return False
    return bool(ip.is_global) and not ip.is_multicast


def _parse_ip(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    raw = str(text or "").strip().strip("[]").split("%", 1)[0]
    try:
        return ipaddress.ip_address(raw)
    except ValueError:
        return None


async def _getaddrinfo(host: str, port: int) -> list[str]:
    """Resolve *host* to address strings (patched in tests)."""
    import asyncio

    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


async def resolve_public(host: str, port: int) -> list[str]:
    """Resolve *host* and return its addresses, or raise ``httpcore.ConnectError``
    unless EVERY address is public."""
    literal = _parse_ip(host)
    if literal is not None:
        if not _is_public_ip(literal):
            raise httpcore.ConnectError(f"Refused: {host} is not a public address")
        return [str(literal)]
    name = str(host or "").strip().rstrip(".").lower()
    if not name or name == "localhost" or name.endswith(".localhost"):
        raise httpcore.ConnectError(f"Refused: {host} is not a public address")
    try:
        found = await _getaddrinfo(name, port)
    except OSError as exc:
        raise httpcore.ConnectError(f"Could not resolve {host}: {exc}") from exc
    addresses: list[str] = []
    for raw in found:
        ip = _parse_ip(raw)
        if ip is None or not _is_public_ip(ip):
            raise httpcore.ConnectError(f"Refused: {host} resolves to a non-public address")
        if str(ip) not in addresses:
            addresses.append(str(ip))
    if not addresses:
        raise httpcore.ConnectError(f"Could not resolve {host}")
    return addresses


class _PublicOnlyBackend(httpcore.AsyncNetworkBackend):
    """Network backend that only connects to public addresses.

    The check and the connect happen here, on the resolved IP, so there is
    no DNS-rebinding gap between them; every redirect hop opens its
    connection through this method. TLS still uses the original hostname
    (httpcore passes it as ``server_hostname`` to ``start_tls``).
    """

    def __init__(self, inner: httpcore.AsyncNetworkBackend | None = None) -> None:
        self._inner = inner or httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        addresses = await resolve_public(host, port)
        last: Exception | None = None
        for ip in addresses:
            try:
                return await self._inner.connect_tcp(
                    ip, port, timeout=timeout,
                    local_address=local_address, socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as exc:
                last = exc
        if isinstance(last, (httpcore.ConnectError, httpcore.ConnectTimeout)):
            raise last
        raise httpcore.ConnectError(f"Could not connect to {host}: {last}")

    async def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        raise httpcore.ConnectError("Unix sockets are not allowed")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def make_public_http_client(
    *, network_backend: httpcore.AsyncNetworkBackend | None = None,
) -> httpx.AsyncClient:
    """An httpx client for member ``web_fetch``: public addresses only, no
    proxies from the environment, at most 5 redirects, 30 s timeout.

    *network_backend* is the inner backend the public-only check wraps
    (tests pass a mock); production uses httpcore's AnyIO backend.
    """
    transport = httpx.AsyncHTTPTransport()
    transport._pool = httpcore.AsyncConnectionPool(
        ssl_context=httpx.create_ssl_context(),
        network_backend=_PublicOnlyBackend(network_backend),
    )
    return httpx.AsyncClient(
        transport=transport,
        trust_env=False,
        follow_redirects=True,
        max_redirects=5,
        timeout=30.0,
        headers={"User-Agent": "Captain Claw/0.1.0 (Web Fetch Tool)"},
    )
