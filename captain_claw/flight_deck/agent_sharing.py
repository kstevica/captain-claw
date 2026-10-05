"""Chat-only shared agents (A1) — identifiers, the speaker handshake, membership.

An owner shares one of their agents with other deck users ("members"). A member
never reaches the agent directly: Flight Deck connects to it on their behalf
(``agent_sharing_routes``) with the token FD recorded at spawn, plus a signed
``X-FD-Speaker`` assertion naming who is speaking, and relays an allowlisted set
of chat frames. This module holds the pure logic behind that route and the
share routes:

* **Identifiers.** ``agent_ref = "{runtime}:{slug}:{instance}"``. ``instance``
  (16 hex) tells apart agents that reuse a slug — another owner's, the other
  runtime's, or a removed-and-recreated one — so a member never silently
  inherits access to a different agent. New agents get a random id; existing
  ones fall back to a deterministic hash of their recorded token.
* **Resolution.** ``resolve_agent_record`` looks an agent up by ref in FD's own
  records (process registry / this deck's container labels) — never by port and
  never from anything the browser sent.
* **Handshake.** ``sign_speaker_assertion`` / ``speaker_ack_for``: HMAC with a
  key derived from the agent's ``web_auth``, so no new secret store and no
  respawn are needed. The agent verifies with its own ``web.auth_token``.
* **Live sockets.** An in-memory registry (FD is one process) so revocation —
  a share deleted, a member leaving, an agent removed — closes the member's
  open sockets immediately.
* **Membership.** ``member_check`` with a short positive-only cache.

Off unless ``FD_AGENT_SHARING`` is set and FD auth is on (``sharing_active``):
with it off nothing here is reachable — the share routes refuse the ``agent``
type and the member route closes 4503 — and FD never sends ``X-FD-Speaker``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from captain_claw.logging import get_logger

log = get_logger(__name__)

# ── Flag + constants (contract part 0 §2, §8) ─────────────────────────────


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


SHARING_ENABLED: bool = _env_flag("FD_AGENT_SHARING")

AGENT_RESOURCE = "agent"
INSTANCE_LABEL = "flight-deck.instance-id"
HOST_TRUST_WARNING = (
    "Anyone on this deck who runs their own shell-capable process agent can act as "
    "any agent on this host, including this one."
)
MEMBER_LANES = ("A", "B", "C")
MAX_MEMBER_SOCKETS_PER_AGENT = 6        # per member, per agent
MAX_OUTSTANDING_TURNS_PER_CONN = 2
ACK_TIMEOUT_S = 10
RECHECK_INTERVAL_S = 10
MEMBERSHIP_CACHE_TTL_S = 10             # positive results only
JWT_GRACE_S = 30
SPEAKER_MAX_INSTANCES = _env_int("CLAW_SPEAKER_MAX_INSTANCES", 32)
SPEAKER_MAX_PER_USER = 3
SPEAKER_IDLE_EVICT_S = 1800
MAX_CHAT_CONTENT = 100_000              # chars; also caps a `btw` note
BTW_MIN_INTERVAL_S = 1.0                # per member socket: at most one `btw` a second
# The `session_settings` strings a member may send, with their caps (chars):
# the agent persists them and puts them in every system prompt of that session.
SESSION_SETTING_MAX: dict[str, int] = {
    "session_name": 200, "session_description": 2000, "session_instructions": 8000}
SPEAKER_TOOL_ALLOWLIST = frozenset({"insights", "playbooks", "topics", "web_search", "web_fetch"})

AGENT_REF_RE = re.compile(r"^(process|docker):([a-z0-9][a-z0-9-]{0,99}):([0-9a-f]{16})$")
_INSTANCE_RE = re.compile(r"[0-9a-f]{16}")
RUNTIMES = ("process", "docker")

# Client → FD frames a member may send, each re-serialised with only these keys
# (FD adds ``_fd_turn`` to ``chat``). Contract part 0 §6.
MEMBER_FRAME_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "chat": ("type", "content", "rewind_to", "no_next_steps", "no_rephrase"),
    "cancel": ("type",),
    "btw": ("type", "content"),
    "message_feedback": ("type", "timestamp", "feedback"),
    "session_settings": ("type", "session_name", "session_description", "session_instructions"),
    "set_playbook": ("type", "playbook_id"),
    "approval_response": ("type", "id", "approved"),
}

SPEAKER_HEADER = "X-FD-Speaker"
_SPEAKER_KEY_INFO = b"captain-claw/fd-speaker/v1"
ASSERTION_TTL_S = 60


def sharing_active() -> bool:
    """Sharing is on: the flag is set AND this deck has accounts (with auth off
    there is nobody to share with, so it is treated as off)."""
    if not SHARING_ENABLED:
        return False
    from captain_claw.flight_deck import server as _srv

    return bool(_srv.AUTH_ENABLED)


# ── Identifiers ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgentRecord:
    """One agent as FD recorded it. ``web_auth`` and ``port`` never leave FD."""

    runtime: str
    slug: str
    instance: str
    owner: str
    name: str
    description: str
    port: int
    web_auth: str
    running: bool

    @property
    def ref(self) -> str:
        return format_ref(self.runtime, self.slug, self.instance)


def format_ref(runtime: str, slug: str, instance: str) -> str:
    return f"{runtime}:{slug}:{instance}"


def parse_ref(ref: str) -> tuple[str, str, str]:
    """``(runtime, slug, instance)`` of a well-formed ref; ValueError otherwise."""
    m = AGENT_REF_RE.fullmatch(ref) if isinstance(ref, str) else None
    if not m:
        raise ValueError("invalid agent_ref")
    return m.group(1), m.group(2), m.group(3)


def valid_ref(runtime: str, slug: str, instance: str) -> str:
    """The ref for these parts when it is well-formed, else "" (a slug FD can't
    express in a ref — too long, or a legacy key — just isn't shareable)."""
    if not instance:
        return ""
    ref = format_ref(runtime, slug, instance)
    try:
        parse_ref(ref)
    except ValueError:
        return ""
    return ref


def new_instance_id() -> str:
    return uuid.uuid4().hex[:16]


def derived_instance_id(runtime: str, slug: str, web_auth: str) -> str:
    """Deterministic id for an agent recorded before instance ids existed; ""
    without a token (such an agent can't be shared)."""
    if not web_auth:
        return ""
    raw = f"fd-instance-v1|{runtime}|{slug}|{web_auth}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _stored_instance(value) -> str:
    v = value if isinstance(value, str) else ""
    return v if _INSTANCE_RE.fullmatch(v) else ""


def process_instance_id(slug: str, entry: dict) -> str:
    """Effective instance id of a process-registry entry: the stored one, else
    the derived one ("" when it has neither)."""
    entry = entry if isinstance(entry, dict) else {}
    return _stored_instance(entry.get("instance_id")) or derived_instance_id(
        "process", slug, str(entry.get("web_auth") or ""))


def _docker_slug(container) -> str:
    from captain_claw.flight_deck import server as _srv

    labels = getattr(container, "labels", None) or {}
    return _srv._slug(labels.get("flight-deck.agent-name") or getattr(container, "name", ""))


def docker_instance_id(container) -> str:
    """Effective instance id of a managed container: its INSTANCE_LABEL, else
    derived from its slug and web-auth label."""
    labels = getattr(container, "labels", None) or {}
    return _stored_instance(labels.get(INSTANCE_LABEL)) or derived_instance_id(
        "docker", _docker_slug(container), str(labels.get("flight-deck.web-auth") or ""))


def process_ref(slug: str, entry: dict) -> str:
    """``agent_ref`` of a process-registry entry ("" when it has no id)."""
    return valid_ref("process", slug, process_instance_id(slug, entry))


def docker_ref(container) -> str:
    """``agent_ref`` of a managed container ("" when it has no id)."""
    return valid_ref("docker", _docker_slug(container), docker_instance_id(container))


def process_instance_for_spawn(prior: dict, owner_id: str, slug: str = "") -> str:
    """Instance id for a process spawn. Re-spawning a stopped agent of the same
    owner (it reuses the data dir) keeps the prior entry's effective id; any
    other spawn gets a fresh one."""
    prior = prior if isinstance(prior, dict) else {}
    if prior and str(prior.get("owner") or "") == str(owner_id or ""):
        kept = process_instance_id(slug or str(prior.get("slug") or ""), prior)
        if kept:
            return kept
    return new_instance_id()


def ensure_process_instance_persisted(slug: str) -> None:
    """Write the effective (derived) id into ``.processes.json`` when the entry
    has none stored, so a later token change can't change the agent's ref."""
    from captain_claw.flight_deck import server as _srv

    registry = _srv._load_process_registry()
    entry = registry.get(slug)
    if not isinstance(entry, dict) or _stored_instance(entry.get("instance_id")):
        return
    inst = process_instance_id(slug, entry)
    if not inst:
        return
    entry["instance_id"] = inst
    registry[slug] = entry
    _srv._save_process_registry(registry)


def resolve_agent_record(ref: str) -> AgentRecord | None:
    """The agent ``ref`` names, from FD's own records; None when there is none
    (or the instance doesn't match — a recreated agent is a different agent)."""
    try:
        runtime, slug, instance = parse_ref(ref)
    except ValueError:
        return None
    from captain_claw.flight_deck import server as _srv

    if runtime == "process":
        try:
            entry = _srv._load_process_registry().get(slug)
        except Exception:
            return None
        if not isinstance(entry, dict):
            return None
        inst = process_instance_id(slug, entry)
        if not inst or not hmac.compare_digest(inst, instance):
            return None
        try:
            port = int(entry.get("web_port") or 0)
        except (TypeError, ValueError):
            port = 0
        return AgentRecord(
            runtime="process", slug=slug, instance=inst,
            owner=str(entry.get("owner") or ""),
            name=str(entry.get("name") or slug),
            description=str(entry.get("description") or ""),
            port=port, web_auth=str(entry.get("web_auth") or ""),
            running=bool(_srv._process_is_alive(slug)),
        )

    try:
        containers = _srv._deck_containers(all=True)
    except Exception:  # Docker unavailable
        return None
    for c in containers:
        if _docker_slug(c) != slug:
            continue
        inst = docker_instance_id(c)
        if not inst or not hmac.compare_digest(inst, instance):
            continue
        labels = c.labels or {}
        try:
            port = int(labels.get("flight-deck.web-port") or 0)
        except (TypeError, ValueError):
            port = 0
        return AgentRecord(
            runtime="docker", slug=slug, instance=inst,
            owner=str(labels.get(_srv.OWNER_LABEL) or ""),
            name=str(labels.get("flight-deck.agent-name") or c.name or slug),
            description=str(labels.get("flight-deck.description") or ""),
            port=port, web_auth=str(labels.get("flight-deck.web-auth") or ""),
            running=getattr(c, "status", "") == "running",
        )
    return None


# Exact port of flight-deck/src/components/layout/SimpleLayout.tsx MANAGED_AGENT
# (JS `$` = end of input → `\Z`).
_MANAGED_AGENT_RE = re.compile(
    r"^(?:(?:basna|vatra)-[0-9a-f]{8}-|council-[0-9a-f]{6}-|iskra-.+-[0-9a-f]{4}\Z)")


def is_managed_agent(slug: str, description: str) -> bool:
    """FD spawns and stops these itself (a run's worker, a being's body)."""
    slug = slug or ""
    return bool(_MANAGED_AGENT_RE.match(slug)) or (
        slug.startswith("dubina-") and (description or "").startswith("Dubina ephemeral"))


def check_shareable(rec: AgentRecord | None, owner_id: str) -> str | None:
    """Why ``owner_id`` can't share (or members can't use) ``rec``; None if fine."""
    if rec is None or rec.owner != owner_id:
        return "Agent not found"
    if not rec.owner:
        return "Unowned agents can't be shared"
    if is_managed_agent(rec.slug, rec.description):
        return "Flight Deck–managed workers can't be shared"
    if not rec.web_auth:
        return "This agent has no access token; respawn it to share it"
    return None


# ── Speaker handshake (contract part 0 §5) ────────────────────────────────


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _speaker_key(web_auth: str) -> bytes:
    return hmac.new(web_auth.encode("utf-8"), _SPEAKER_KEY_INFO, hashlib.sha256).digest()


def sign_speaker_assertion(web_auth: str, payload: dict) -> str:
    """``v1.<payload_b64>.<sig_b64>`` for the ``X-FD-Speaker`` header. The key
    is derived from the agent's recorded ``web_auth`` and never sent or logged."""
    if not web_auth:
        raise ValueError("agent has no web_auth")
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload_b64 = _b64url(body.encode())
    sig = hmac.new(_speaker_key(web_auth), ("v1." + payload_b64).encode("ascii"),
                   hashlib.sha256).digest()
    return f"v1.{payload_b64}.{_b64url(sig)}"


def speaker_ack_for(header_value: str) -> str:
    """What an agent that understood the header echoes in its ``welcome``."""
    return hashlib.sha256(header_value.encode("ascii")).hexdigest()[:16]


def speaker_payload(*, speaker_id: str, name: str, owner: str, owner_name: str, ref: str,
                    lane: str, conn: str, now: int | None = None) -> dict:
    iat = int(time.time()) if now is None else int(now)
    return {
        "v": 1, "sub": speaker_id, "name": name, "owner": owner, "owner_name": owner_name,
        "ref": ref, "lane": lane, "conn": conn, "iat": iat, "exp": iat + ASSERTION_TTL_S,
        "nonce": secrets.token_hex(8),
    }


# ── Live member sockets ───────────────────────────────────────────────────

CloseFn = Callable[[int, str], Awaitable[None]]


@dataclass
class _MemberSocket:
    ref: str
    user_id: str
    lane: str
    close: CloseFn


_SOCKETS: dict[str, _MemberSocket] = {}
_CLOSE_TIMEOUT_S = 5.0


def register_member_socket(ref: str, user_id: str, lane: str, close: CloseFn) -> str:
    conn_id = secrets.token_hex(8)
    while conn_id in _SOCKETS:
        conn_id = secrets.token_hex(8)
    _SOCKETS[conn_id] = _MemberSocket(ref=ref, user_id=user_id, lane=lane, close=close)
    return conn_id


def unregister_member_socket(conn_id: str) -> None:
    _SOCKETS.pop(conn_id, None)


def live_conn_count(ref: str, user_id: str) -> int:
    return sum(1 for s in _SOCKETS.values() if s.ref == ref and s.user_id == user_id)


async def close_member_sockets(ref: str, user_id: str | None = None, *, code: int = 4403,
                               reason: str = "Access removed") -> int:
    """Close the live member sockets on ``ref`` (one member's, or everyone's).
    Returns how many were closed. Never raises."""
    targets = [(cid, s) for cid, s in list(_SOCKETS.items())
               if s.ref == ref and (user_id is None or s.user_id == user_id)]
    closed = 0
    for cid, sock in targets:
        _SOCKETS.pop(cid, None)
        try:
            await asyncio.wait_for(sock.close(code, reason), timeout=_CLOSE_TIMEOUT_S)
        except Exception as exc:
            log.warning("Could not close a shared-agent member socket",
                        error=type(exc).__name__)
        closed += 1
    return closed


# ── Membership ────────────────────────────────────────────────────────────

# (ref, owner, user) → monotonic time a True result was cached. False is never cached.
_MEMBER_CACHE: dict[tuple[str, str, str], float] = {}
_MEMBER_CACHE_MAX = 4096


async def member_check(db, ref: str, owner_id: str, user_id: str, *,
                       max_age: float = MEMBERSHIP_CACHE_TTL_S) -> bool:
    """Is ``user_id`` (still a user of this deck) a member of ``owner_id``'s
    agent ``ref``? A True result is reused for at most ``max_age`` seconds
    (``max_age=0`` always asks the DB); False is never cached. Fails closed."""
    if not (ref and owner_id and user_id):
        return False
    key = (ref, owner_id, user_id)
    now = time.monotonic()
    cached_at = _MEMBER_CACHE.get(key)
    if cached_at is not None and max_age > 0 and now - cached_at < max_age:
        return True
    _MEMBER_CACHE.pop(key, None)
    try:
        ok = bool(await db.get_user_by_id(user_id)) and bool(
            await db.is_agent_member(ref, owner_id, user_id))
    except Exception as exc:
        log.warning("Shared-agent membership check failed", error=type(exc).__name__)
        ok = False
    if ok:
        if len(_MEMBER_CACHE) >= _MEMBER_CACHE_MAX:
            horizon = time.monotonic() - MEMBERSHIP_CACHE_TTL_S
            for k in [k for k, t in _MEMBER_CACHE.items() if t < horizon]:
                _MEMBER_CACHE.pop(k, None)
        _MEMBER_CACHE[key] = time.monotonic()
    return ok


def invalidate_member_cache(ref: str, user_id: str | None = None) -> None:
    for key in [k for k in _MEMBER_CACHE if k[0] == ref and (user_id is None or k[2] == user_id)]:
        _MEMBER_CACHE.pop(key, None)
