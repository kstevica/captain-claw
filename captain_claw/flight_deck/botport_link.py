"""Flight Deck ⇄ BotPort link — deck federation (BotPort protocol 2).

The deck dials OUT to a BotPort hub over one WebSocket (NAT-friendly: only the
hub needs to be reachable) and keeps it up. Over it the deck:

* advertises the agents an admin chose to expose (``expose``) and those
  their owners shared themselves (``user_sharing``, see ``SHARES_KEY``),
* receives the ``network`` roster — the other decks and their agents,
* serves ``dispatch`` from other decks by proxying them to its own agent with
  the same in-process primitive ``/fd/consult-peer`` uses
  (``_consult_peer_events``), streaming progress back,
* sends its own agents' consults/delegations to remote agents (``consult``),
  yielding the same ndjson event contract as ``/fd/consult-peer``.

Robustness: reconnect with jittered backoff; results produced while the link
is down wait in an outbox until the hub acknowledges them; on reconnect the
deck tells the hub what it is still working on (``in_flight``) and waiting for
(``awaiting``) and resubmits requests the hub never acknowledged — submission
is idempotent by ``concern_id``. A remote turn can never write email
(``mail_write: deny``) and never triggers the target's Flows.

Settings live in the system setting ``fd:botport-link`` (admin-editable from
the UI; the secret sealed with fd_crypto when available) and every field can
be pinned by environment (``FD_BOTPORT_*``) for headless deployments.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from captain_claw.logging import get_logger

log = get_logger(__name__)

SETTING_KEY = "fd:botport-link"
DECK_ID_KEY = "fd:botport-deck-id"
# Agents their owners shared on the network themselves:
# {"agents": {share_key: {"owner": user_id, "by": user_id, "at": iso}}} (see share_key).
SHARES_KEY = "fd:botport-shares"
PROTOCOL = 2
_MAX_FRAME = 16 * 1024 * 1024
_HEARTBEAT_SECONDS = 25.0
_AGENT_REFRESH_SECONDS = 30.0
_BACKOFF_MAX = 60.0
_STABLE_SECONDS = 10.0  # a session this long counts as healthy: the next retry is quick again
_PROGRESS_MIN_INTERVAL = 0.5
_DONE_MEMORY = 512
_RECENT = 40

# A remote deck's request is an automated turn that may never write email.
REMOTE_AUTOMATION: dict[str, str] = {"kind": "botport", "job_text": "", "mail_write": "deny"}
# Tools a remote turn never gets: no reaching back into the network from a
# turn another deck started (no A -> B -> A loops, no fan-out on its behalf).
LOOP_GUARD_TOOLS = ("flight_deck", "consult_peer", "botport")
# Everything a remote turn never gets: the loop guard plus the tools bound to
# the owner's own session and memory. Agents enforce the same set themselves
# for any "BP" lane (web_server.REMOTE_LANE_DENY); this is defence in depth.
REMOTE_DENY_TOOLS = LOOP_GUARD_TOOLS + (
    "cron", "sessions", "history", "topics", "personality", "terminal",
    "insights", "intentions", "shared_agent_usage",
)
# Agents run each remote deck's requests in a lane of their own ("BP" + hash):
# never the owner's main chat, never another deck's history.
REMOTE_LANE_PREFIX = "BP"


def remote_lane(sender: str) -> str:
    """The agent lane a sender's requests run in: "BP" + 80 bits of a hash.

    ``sender`` is a hub-verified deck name, or "agent:<name>" for a legacy
    agent (deck names cannot contain ':', so the two never share a lane)."""
    import hashlib

    digest = hashlib.sha256((sender or "?").strip().lower().encode("utf-8")).hexdigest().upper()
    return REMOTE_LANE_PREFIX + digest[:20]

_ENV = {
    "enabled": "FD_BOTPORT_ENABLED",
    "url": "FD_BOTPORT_URL",
    "deck_name": "FD_BOTPORT_DECK_NAME",
    "key": "FD_BOTPORT_KEY",
    "secret": "FD_BOTPORT_SECRET",
    "expose": "FD_BOTPORT_EXPOSE",
    "accept_from": "FD_BOTPORT_ACCEPT_FROM",
    "allow_outbound": "FD_BOTPORT_ALLOW_OUTBOUND",
    "max_concurrent": "FD_BOTPORT_MAX_CONCURRENT",
    "deny_tools": "FD_BOTPORT_DENY_TOOLS",
    "user_sharing": "FD_BOTPORT_USER_SHARING",
    "outbound_users": "FD_BOTPORT_OUTBOUND_USERS",
}


class LinkError(Exception):
    """A request the link cannot carry (message is user-facing)."""


def ws_proxy_kwargs(proxy: Any) -> dict[str, Any]:
    """``{"proxy": proxy}`` for ``websockets.connect`` where it exists (15.0+,
    which reads HTTP(S)_PROXY by default); ``{}`` on older releases, which
    neither take the argument nor use a proxy."""
    try:
        import websockets

        major = int(str(getattr(websockets, "__version__", "0")).split(".")[0])
    except (ImportError, ValueError):
        return {}
    return {"proxy": proxy} if major >= 15 else {}


# ── Settings ─────────────────────────────────────────────────────


@dataclass
class LinkSettings:
    enabled: bool = False
    url: str = ""                 # ws(s)://host:port/ws — http(s):// and host:port accepted
    deck_name: str = ""
    key: str = ""
    secret: str = ""
    expose: list[str] = field(default_factory=list)       # agent names/slugs, or ["*"]
    accept_from: list[str] = field(default_factory=lambda: ["*"])  # deck names, or ["*"]
    allow_outbound: bool = True   # this deck's agents may consult remote agents
    # ...every user's agents, not only the admins' (on a deck with accounts):
    # the other decks on the hub are usually the operator's own.
    outbound_users: bool = False
    max_concurrent: int = 4       # inbound tasks run at once
    max_inbound_timeout: float = 900.0
    # Extra tools remote turns may not use (e.g. shell, scripts, write) — on
    # top of LOOP_GUARD_TOOLS, which are always denied.
    deny_tools: list[str] = field(default_factory=list)
    # Users may share the agents they own themselves (next to the admin's list).
    user_sharing: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LinkSettings:
        d = dict(data or {})
        s = cls()
        s.enabled = _as_bool(d.get("enabled", s.enabled))
        s.url = str(d.get("url") or "").strip()
        s.deck_name = str(d.get("deck_name") or "").strip()
        s.key = str(d.get("key") or "").strip()
        s.secret = str(d.get("secret") or "").strip()
        s.expose = _as_list(d.get("expose", []))
        s.accept_from = _as_list(d.get("accept_from", ["*"])) or []
        s.allow_outbound = _as_bool(d.get("allow_outbound", True))
        s.outbound_users = _as_bool(d.get("outbound_users", False))
        try:
            s.max_concurrent = max(1, min(int(d.get("max_concurrent", 4)), 64))
        except (TypeError, ValueError):
            s.max_concurrent = 4
        try:
            s.max_inbound_timeout = max(30.0, min(float(d.get("max_inbound_timeout", 900.0)), 3600.0))
        except (TypeError, ValueError):
            s.max_inbound_timeout = 900.0
        s.deny_tools = _as_list(d.get("deny_tools", []))
        s.user_sharing = _as_bool(d.get("user_sharing", True))
        return s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def problems(self) -> list[str]:
        out: list[str] = []
        if not self.enabled:
            return out
        if not self.url:
            out.append("hub URL is empty")
        else:
            try:
                normalize_url(self.url)
            except ValueError as exc:
                out.append(str(exc))
        if not valid_deck_name(self.deck_name):
            out.append("deck name must be letters, digits, '-', '_' or '.' (max 63, no spaces or '@')")
        return out


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _as_list(v: Any) -> list[str]:
    if isinstance(v, str):
        items = v.split(",")
    elif isinstance(v, (list, tuple)):
        items = [str(x) for x in v]
    else:
        items = []
    out: list[str] = []
    for item in items:
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


def env_overrides() -> dict[str, Any]:
    """Settings pinned by environment (they win over the stored ones)."""
    out: dict[str, Any] = {}
    for fld, var in _ENV.items():
        raw = os.environ.get(var)
        if raw is None or raw == "":
            continue
        out[fld] = raw
    # Setting the URL by env implies the link is wanted.
    if "url" in out and "enabled" not in out:
        out["enabled"] = "true"
    return out


def agent_key(agent: dict[str, Any]) -> str:
    """How exposure lists name an agent: its slug, else its name (lower case)."""
    return str(agent.get("slug") or agent.get("name") or "").strip().lower()


def share_key(agent: dict[str, Any]) -> str:
    """An owner's share names the agent by kind, slug and owner: a docker and
    a process agent of one name stay apart, and a share lapses by itself when
    the agent changes hands (its key no longer matches)."""
    return f"{str(agent.get('kind') or '').lower()}:{agent_key(agent)}|{agent.get('owner') or ''}"


def parse_shares(raw: str | None) -> dict[str, dict[str, str]]:
    """The stored user shares (``SHARES_KEY``); anything malformed is dropped."""
    try:
        data = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {}
    agents = data.get("agents") if isinstance(data, dict) else None
    out: dict[str, dict[str, str]] = {}
    for key, rec in (agents or {}).items() if isinstance(agents, dict) else ():
        if isinstance(key, str) and key.strip() and isinstance(rec, dict):
            out[key.strip()[:300]] = {k: str(rec.get(k) or "")[:128] for k in ("owner", "by", "at")}
    return out


def dump_shares(shares: dict[str, dict[str, str]]) -> str:
    return json.dumps({"agents": shares}, sort_keys=True)


def valid_deck_name(name: str) -> bool:
    import re

    return bool(re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$", name or ""))


def normalize_url(url: str) -> str:
    """The hub's WebSocket URL from what an admin typed.

    ``host:port`` → ``ws://host:port/ws``; ``http(s)://`` → ``ws(s)://``; a
    bare origin gets ``/ws``. Anything else is refused.
    """
    raw = (url or "").strip()
    if not raw:
        raise ValueError("hub URL is empty")
    if "://" not in raw:
        raw = "ws://" + raw
    parts = urlsplit(raw)
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(parts.scheme.lower())
    if scheme is None:
        raise ValueError(f"unsupported hub URL scheme {parts.scheme!r} (use ws://, wss://, http:// or https://)")
    if not parts.hostname:
        raise ValueError("hub URL has no host")
    path = parts.path.rstrip("/")
    if not path:
        path = "/ws"
    return urlunsplit((scheme, parts.netloc, path, parts.query, ""))


def health_url(ws_url: str) -> str:
    parts = urlsplit(ws_url)
    scheme = "https" if parts.scheme == "wss" else "http"
    path = parts.path[: -len("/ws")] if parts.path.endswith("/ws") else parts.path
    return urlunsplit((scheme, parts.netloc, (path.rstrip("/") + "/healthz"), "", ""))


# ── Requests in flight ───────────────────────────────────────────


@dataclass
class _Outbound:
    concern_id: str
    submit: dict[str, Any]
    queue: asyncio.Queue
    deadline: float
    target: str
    started: float = field(default_factory=time.time)
    acked: bool = False
    state: str = "submitted"


@dataclass
class _Inbound:
    concern_id: str
    task: asyncio.Task
    from_label: str
    agent: str
    started: float = field(default_factory=time.time)
    last_event: str = "accepted"


# Callbacks wired by server.py at startup (kept as plain callables so this
# module never imports the FD server).
FleetProvider = Callable[[], list[dict[str, Any]]]
ConsultEvents = Callable[..., AsyncIterator[dict[str, Any]]]
ResolveAuth = Callable[[int], str]
SettingsLoader = Callable[[], Awaitable[dict[str, Any] | None]]
SettingsSaver = Callable[[dict[str, Any]], Awaitable[None]]


class BotPortLink:
    """One outbound link from this Flight Deck to a BotPort hub."""

    def __init__(self) -> None:
        self.settings = LinkSettings()
        self.deck_id = ""
        self.version = ""
        self._fleet: FleetProvider = lambda: []
        self._consult_events: ConsultEvents | None = None
        self._resolve_auth: ResolveAuth = lambda port: ""
        self._save: SettingsSaver | None = None

        self.state = "disabled"   # disabled | connecting | connected | backoff | error | stopped
        self.last_error = ""
        self.last_error_at = 0.0
        self.connected_since = 0.0
        self.next_retry_at = 0.0
        self.hub_version = ""
        self.hub_deck_name = ""
        self.network: dict[str, Any] = {"decks": [], "agents": []}

        self._ws: Any = None
        self._conn: Any = None          # the socket being connected (before "registered")
        self._settings_gen = 0          # bumped by apply(): a session on older settings ends
        self._send_lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._stopping = False
        self._outbound: dict[str, _Outbound] = {}
        self._inbound: dict[str, _Inbound] = {}
        self._outbox: OrderedDict[str, dict[str, Any]] = OrderedDict()  # result frames awaiting hub ack
        self._done: OrderedDict[str, None] = OrderedDict()   # finished outbound ids (dedupe)
        self._advertised: list[dict[str, Any]] = []
        self.user_shares: dict[str, dict[str, str]] = {}   # see SHARES_KEY
        self._recent: deque[dict[str, Any]] = deque(maxlen=_RECENT)
        self.counters: dict[str, int] = {
            "connects": 0, "disconnects": 0, "outbound": 0, "outbound_ok": 0, "outbound_failed": 0,
            "inbound": 0, "inbound_ok": 0, "inbound_failed": 0, "inbound_refused": 0,
        }

    # ── wiring / lifecycle ────────────────────────────────────────

    def configure(self, *, fleet: FleetProvider, consult_events: ConsultEvents,
                  resolve_auth: ResolveAuth, save: SettingsSaver | None = None,
                  version: str = "") -> None:
        self._fleet = fleet
        self._consult_events = consult_events
        self._resolve_auth = resolve_auth
        self._save = save
        self.version = version

    async def start(self, stored: dict[str, Any] | None, deck_id: str) -> None:
        self.deck_id = deck_id or uuid.uuid4().hex
        self.settings = self.effective(stored)
        self._stopping = False
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="botport-link")

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        ws = self._ws
        if ws is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws.close(code=1001, reason="Flight Deck shutting down"), timeout=3)
        for inb in list(self._inbound.values()):
            inb.task.cancel()
        self._fail_outbound("Flight Deck is shutting down")
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._task, timeout=5)
        self.state = "stopped"

    def _fail_outbound(self, error: str) -> None:
        """End every request this deck's agents are waiting on."""
        for cid, o in list(self._outbound.items()):
            self._outbound.pop(cid, None)
            self._remember_done(cid)
            o.queue.put_nowait({"ok": False, "error": error})

    def effective(self, stored: dict[str, Any] | None) -> LinkSettings:
        merged = dict(stored or {})
        merged.update(env_overrides())
        return LinkSettings.from_dict(merged)

    async def apply(self, stored: dict[str, Any] | None) -> None:
        """New settings: reconnect with them (a connection still being set up
        on the old ones is dropped too)."""
        self.settings = self.effective(stored)
        self._settings_gen += 1
        self.last_error = ""
        if not self.settings.enabled:
            self._fail_outbound("The BotPort link was switched off")
        await self.reconnect()

    async def reconnect(self) -> None:
        self.next_retry_at = 0.0
        self._wake.set()
        for ws in {id(w): w for w in (self._ws, self._conn) if w is not None}.values():
            with contextlib.suppress(Exception):
                await ws.close(code=1000, reason="reconnecting")

    # ── status for the UI / fleet ────────────────────────────────

    @property
    def connected(self) -> bool:
        return self.settings.enabled and self.state == "connected" and self._ws is not None

    def status(self) -> dict[str, Any]:
        s = self.settings
        now = time.time()
        return {
            "enabled": s.enabled,
            "state": self.state,
            "url": s.url,
            "deck_name": self.hub_deck_name or s.deck_name,
            "hub_version": self.hub_version,
            "connected_since": self.connected_since or None,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at or None,
            "retry_in": max(0, int(self.next_retry_at - time.monotonic())) if self.state == "backoff" else None,
            "problems": s.problems(),
            "exposed_agents": [a.get("name") for a in self._advertised],
            "network": self.network,
            "inbound": [
                {"concern_id": i.concern_id, "from": i.from_label, "agent": i.agent,
                 "seconds": int(now - i.started), "last_event": i.last_event}
                for i in self._inbound.values()
            ],
            "outbound": [
                {"concern_id": o.concern_id, "target": o.target, "seconds": int(now - o.started),
                 "state": o.state}
                for o in self._outbound.values()
            ],
            "queued_results": len(self._outbox),
            "counters": dict(self.counters),
            "recent": list(self._recent)[::-1],
        }

    def remote_agents(self) -> list[dict[str, Any]]:
        """Agents on other decks this deck's agents may consult (connected decks only)."""
        if not self.connected or not self.settings.allow_outbound:
            return []
        out: list[dict[str, Any]] = []
        for deck in self.network.get("decks", []):
            if deck.get("status") != "connected":
                continue
            for a in deck.get("agents", []):
                if a.get("status", "running") != "running":
                    continue
                out.append({
                    "name": f"{a.get('name', '?')}@{deck.get('name', '?')}",
                    "agent": a.get("name", ""),
                    "slug": a.get("slug", ""),
                    "deck": deck.get("name", ""),
                    # What the hub's operator calls that deck (may be empty).
                    "deck_label": str(deck.get("label") or ""),
                    "description": a.get("description", ""),
                    "expertise_tags": a.get("expertise_tags", []),
                })
        return out

    async def local_agents(self, owner: str | None = None) -> list[dict[str, Any]]:
        """This deck's agents (only ``owner``'s when given) with how each is
        shared: ``exposed`` (on the network), ``by_admin`` (the admin's list
        covers it), ``by_owner`` (its owner shared it), ``shareable`` (its
        owner may share it: not a Flight Deck worker)."""
        try:
            fleet = await asyncio.to_thread(self._fleet) or []
        except Exception:
            fleet = []
        exposed = {agent_key(a) for a in await self._exposed()}
        multi_user, is_managed = _deck_rules()
        out = []
        for a in fleet:
            if a.get("kind") == "remote":
                continue
            if owner is not None and multi_user and str(a.get("owner") or "") != owner:
                continue
            key = agent_key(a)
            managed = is_managed(str(a.get("slug") or ""), str(a.get("description") or ""))
            out.append({
                "name": a.get("name", ""), "slug": a.get("slug", ""), "kind": a.get("kind", ""),
                "status": a.get("status", ""), "description": a.get("description", ""),
                "owner": str(a.get("owner") or ""),
                "exposed": key in exposed,
                "by_admin": not managed and self._admin_lists(a, multi_user),
                "by_owner": self._owner_shared(a, multi_user),
                "shareable": not managed,
            })
        return out

    def set_user_shares(self, shares: dict[str, dict[str, str]]) -> None:
        """Owners' own shares changed: keep them and re-advertise."""
        self.user_shares = dict(shares)
        self.notify_fleet_changed()

    def _admin_lists(self, a: dict[str, Any], multi_user: bool) -> bool:
        """Does the admin's ``expose`` list cover this agent?"""
        want = {x.strip().lower() for x in self.settings.expose if x.strip()}
        name = str(a.get("name") or "").strip().lower()
        slug = str(a.get("slug") or "").strip().lower()
        if name in want or (slug and slug in want):
            return True
        # "*" on a multi-user deck: only agents someone owns (an orphan
        # from before sign-in has nobody who agreed to its exposure).
        return "*" in want and not (multi_user and not a.get("owner"))

    def _owner_shared(self, a: dict[str, Any], multi_user: bool) -> bool:
        """Did this agent's current owner share it? (The key carries the owner,
        so a share made by an earlier owner no longer matches.)"""
        if share_key(a) not in self.user_shares:
            return False
        return not multi_user or bool(a.get("owner"))

    def _note(self, direction: str, peer: str, agent: str, ok: bool | None, detail: str = "") -> None:
        self._recent.append({
            "at": time.time(), "dir": direction, "peer": peer, "agent": agent,
            "ok": ok, "detail": detail[:300],
        })

    # ── agents this deck exposes ─────────────────────────────────

    async def _exposed(self) -> list[dict[str, Any]]:
        """The fleet agents on the network (running or not): those the admin
        chose to expose and, unless the admin turned it off, those their
        owners shared themselves."""
        owners = self.settings.user_sharing and bool(self.user_shares)
        if not any(x.strip() for x in self.settings.expose) and not owners:
            return []
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        try:
            # The fleet comes from Docker / the process registry (blocking).
            fleet = await asyncio.to_thread(self._fleet) or []
        except Exception as exc:
            log.warning("botport link: fleet lookup failed", error=str(exc))
            fleet = []
        multi_user, is_managed_agent = _deck_rules()
        for a in fleet:
            name = str(a.get("name") or "").strip()
            slug = str(a.get("slug") or "").strip()
            if not name or (a.get("kind") == "remote"):
                continue
            # Flight Deck's own managed workers are internal: never exposed.
            if is_managed_agent(slug, str(a.get("description") or "")):
                continue
            if not (self._admin_lists(a, multi_user) or (owners and self._owner_shared(a, multi_user))):
                continue
            key = agent_key(a)
            if key in seen:
                continue
            seen.add(key)
            out.append(a)
        return out

    async def _ads(self) -> list[dict[str, Any]]:
        ads = []
        for a in await self._exposed():
            ads.append({
                "name": a.get("name", ""),
                "slug": a.get("slug", ""),
                "description": str(a.get("description") or "")[:1000],
                "expertise_tags": list(a.get("expertise_tags") or [])[:30],
                # A sleeping agent (agent_idle) takes requests: one wakes it.
                "status": "running" if a.get("status") in ("running", "asleep") else str(a.get("status") or "stopped"),
            })
        return ads

    def notify_fleet_changed(self) -> None:
        """The local fleet changed: re-advertise soon (no-op when offline)."""
        if self.connected:
            asyncio.ensure_future(self._refresh_agents())

    async def _refresh_agents(self) -> None:
        ads = await self._ads()
        if ads != self._advertised:
            self._advertised = ads
            await self._send({"type": "deck_agents", "agents": ads})

    # ── connection loop ──────────────────────────────────────────

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stopping:
            s = self.settings
            if not s.enabled or s.problems():
                self.state = "disabled" if not s.enabled else "error"
                if s.enabled:
                    self.last_error = "; ".join(s.problems())
                await self._sleep(None)
                continue
            try:
                url = normalize_url(s.url)
                outcome = await self._session(url)
                if outcome == "fatal":
                    # Bad credentials / name / old hub: wait for new settings (or 10 min).
                    self.state = "error"
                    await self._sleep(600.0)
                    continue
                if outcome == "superseded":
                    # Another process with this deck's identity took the link:
                    # two of them would take it from each other forever.
                    self.state = "error"
                    self._error("another Flight Deck process with this deck's identity took over the link "
                                "(code 4000) — run one process per deck data directory; retrying in 5 min")
                    await self._sleep(300.0)
                    continue
                if outcome == "restart":
                    continue  # settings changed while connecting: go again at once
                if outcome == "connected" and time.time() - self.connected_since >= _STABLE_SECONDS:
                    backoff = 1.0  # it worked: come back quickly after a drop
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._error(_describe(exc))
            finally:
                self._ws = None
                self._conn = None
            if self._stopping:
                break
            delay = min(backoff, _BACKOFF_MAX) * (0.75 + random.random() * 0.5)
            backoff = min(backoff * 2, _BACKOFF_MAX)
            self.state = "backoff"
            self.next_retry_at = time.monotonic() + delay
            await self._sleep(delay)

    async def _sleep(self, seconds: float | None) -> None:
        """Sleep until ``seconds`` pass or something asks for a reconnect
        (a wake-up that arrived earlier counts — it is not lost)."""
        try:
            if seconds is None:
                await self._wake.wait()
            else:
                await asyncio.wait_for(self._wake.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass
        self._wake.clear()

    def _error(self, text: str) -> None:
        if text != self.last_error:
            log.warning("botport link", error=text)
        self.last_error = text
        self.last_error_at = time.time()

    async def _session(self, url: str) -> str:
        """One connection's life: "connected" (it was up, then dropped),
        "refused" (retry later), "fatal" (retrying with these settings is
        pointless), "superseded" (code 4000) or "restart" (settings changed)."""
        import websockets
        from websockets.exceptions import ConnectionClosed

        s = self.settings
        gen = self._settings_gen
        self.state = "connecting"
        proxy: Any = None if os.environ.get("FD_BOTPORT_NO_PROXY", "").lower() in ("1", "true", "yes") else True
        async with websockets.connect(
            url, open_timeout=15, ping_interval=20, ping_timeout=20, close_timeout=5,
            max_size=_MAX_FRAME, **ws_proxy_kwargs(proxy),
            user_agent_header=f"FlightDeck/{self.version or '?'} botport-link/{PROTOCOL}",
        ) as ws:
            self._conn = ws
            self._advertised = await self._ads()
            await ws.send(json.dumps({
                "type": "register",
                "kind": "deck",
                "protocol": PROTOCOL,
                "instance_name": s.deck_name,
                "deck_id": self.deck_id,
                "key": s.key,
                "secret": s.secret,
                "capabilities": {
                    "agents": self._advertised,
                    "max_concurrent": s.max_concurrent,
                    "version": self.version,
                    "features": ["progress", "cancel", "resume"],
                },
                # Still working on, or finished with the result not yet
                # acknowledged (it is resent right after this) — either way
                # the hub must keep the concern alive.
                "in_flight": list(dict.fromkeys([*self._inbound, *self._outbox])),
                # Acked requests we still wait for, and ones whose ack never
                # came (resubmitted right after this — submission is idempotent).
                "awaiting": [cid for cid, o in self._outbound.items() if o.acked],
                "submitted": [cid for cid, o in self._outbound.items() if not o.acked],
            }))
            reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            if reply.get("type") != "registered" or not reply.get("ok"):
                err = str(reply.get("error") or "registration refused")
                self._error(f"hub refused registration: {err}")
                return "fatal" if reply.get("retry") is False else "refused"
            hub_protocol = int(reply.get("protocol") or 1)
            if hub_protocol < PROTOCOL:
                self._error(f"the hub speaks BotPort protocol {hub_protocol}; Flight Deck links need "
                            f"protocol {PROTOCOL} — upgrade BotPort on the hub")
                return "fatal"
            if gen != self._settings_gen:
                return "restart"

            self._ws = ws
            self.state = "connected"
            self.connected_since = time.time()
            self.hub_version = str(reply.get("botport_version") or "")
            self.hub_deck_name = str(reply.get("deck_name") or s.deck_name)
            self.counters["connects"] += 1
            self.last_error = ""
            log.info("botport link connected", hub=url, deck=self.hub_deck_name,
                     resumed=bool(reply.get("resumed")), agents=len(self._advertised))

            # Requests the hub never acknowledged: submit again (idempotent).
            for o in list(self._outbound.values()):
                if not o.acked:
                    await self._send(o.submit)
            # Results the hub never acknowledged.
            for frame in list(self._outbox.values()):
                await self._send(frame)

            beat = asyncio.create_task(self._beat())
            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(msg, dict):
                        try:
                            await self._on_message(msg)
                        except Exception as exc:
                            log.warning("botport link: handler failed", type=msg.get("type"), error=str(exc))
            except ConnectionClosed:
                pass  # a drop after a good session: reported below, not as an exception
            finally:
                beat.cancel()
                self._ws = None
                self.counters["disconnects"] += 1
                code = getattr(ws, "close_code", None)
                reason = getattr(ws, "close_reason", "") or ""
                if not self._stopping:
                    self._error(f"link closed ({code}{': ' + reason if reason else ''})")
            if code == 4000:
                return "superseded"
        return "connected"

    async def _beat(self) -> None:
        last_refresh = time.monotonic()
        while True:
            await asyncio.sleep(_HEARTBEAT_SECONDS)
            await self._send({"type": "heartbeat", "active_concerns": len(self._inbound)})
            if time.monotonic() - last_refresh >= _AGENT_REFRESH_SECONDS:
                last_refresh = time.monotonic()
                await self._refresh_agents()

    async def _send(self, data: dict[str, Any]) -> bool:
        ws = self._ws
        if ws is None:
            return False
        try:
            async with self._send_lock:
                await ws.send(json.dumps(data, default=str))
            return True
        except Exception as exc:
            log.debug("botport link: send failed", error=str(exc))
            return False

    # ── inbound frames ───────────────────────────────────────────

    async def _on_message(self, msg: dict[str, Any]) -> None:
        t = msg.get("type")
        if t == "network":
            self.network = {"decks": list(msg.get("decks") or []), "agents": list(msg.get("agents") or [])}
        elif t == "dispatch":
            self._start_inbound(msg)
        elif t == "concern_closed":
            inb = self._inbound.get(str(msg.get("concern_id") or ""))
            if inb is not None:
                inb.last_event = f"cancelled ({msg.get('reason', '')})"
                inb.task.cancel()
        elif t == "result_ack":
            self._outbox.pop(str(msg.get("concern_id") or ""), None)
        elif t == "concern_ack":
            self._on_ack(msg)
        elif t == "progress":
            self._on_progress(msg)
        elif t == "concern_result":
            await self._on_result(msg)
        elif t in ("heartbeat_ack", "registered"):
            pass

    def _on_ack(self, msg: dict[str, Any]) -> None:
        cid = str(msg.get("concern_id") or "")
        o = self._outbound.get(cid)
        if o is None:
            return
        if msg.get("ok"):
            o.acked = True
            o.state = "routed"
            o.queue.put_nowait({"event": "status", "data": {
                "status": f"Routed via BotPort to {msg.get('assigned_to_name') or o.target}"}})
        else:
            self._outbound.pop(cid, None)
            o.queue.put_nowait({"ok": False, "error": f"BotPort: {msg.get('error') or 'request refused'}"})

    def _on_progress(self, msg: dict[str, Any]) -> None:
        o = self._outbound.get(str(msg.get("concern_id") or ""))
        if o is None:
            return
        event = str(msg.get("event") or "")
        data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
        o.state = "working"
        if event == "accepted":
            o.queue.put_nowait({"event": "status", "data": {"status": f"{o.target} accepted the task"}})
        elif event == "heartbeat":
            o.queue.put_nowait({"event": "status", "data": {
                "status": f"{o.target} still working ({data.get('elapsed', '?')}s)"}})
        elif event == "thinking":
            o.queue.put_nowait({"event": "thinking", "data": {
                "tool": data.get("tool", ""), "text": str(data.get("text", ""))[:200]}})
        elif event == "tool":
            o.queue.put_nowait({"event": "monitor", "data": {"tool_name": data.get("tool_name", "")}})
        else:
            o.queue.put_nowait({"event": "status", "data": {"status": str(data.get("status") or data.get("text") or "")[:200]}})

    async def _on_result(self, msg: dict[str, Any]) -> None:
        cid = str(msg.get("concern_id") or "")
        await self._send({"type": "result_ack", "concern_id": cid})
        o = self._outbound.pop(cid, None)
        if o is None:
            if cid not in self._done:
                log.info("botport link: result for a request nobody waits for", concern_id=cid[:8])
            return
        self._remember_done(cid)
        if msg.get("ok"):
            self.counters["outbound_ok"] += 1
            self._note("out", o.target, "", True)
            final: dict[str, Any] = {"ok": True, "done": True, "response": str(msg.get("response") or "(no response)")}
            meta = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else {}
            if meta.get("member_private") in ("data", "content"):
                final["member_private"] = meta["member_private"]
            o.queue.put_nowait(final)
        else:
            self.counters["outbound_failed"] += 1
            err = str(msg.get("error") or "remote agent failed")
            self._note("out", o.target, "", False, err)
            o.queue.put_nowait({"ok": False, "error": err})

    def _remember_done(self, cid: str) -> None:
        self._done[cid] = None
        while len(self._done) > _DONE_MEMORY:
            self._done.popitem(last=False)

    # ── serving other decks (inbound) ────────────────────────────

    def _start_inbound(self, d: dict[str, Any]) -> None:
        cid = str(d.get("concern_id") or "")
        if not cid or cid in self._inbound:
            return  # a re-dispatch of something already running
        if cid in self._outbox:
            # Finished while the link was down; the result is on its way.
            asyncio.ensure_future(self._send(self._outbox[cid]))
            return
        task = asyncio.create_task(self._serve(d), name=f"botport-inbound-{cid[:8]}")
        from_label = _from_label(d)
        self._inbound[cid] = _Inbound(cid, task, from_label, str(d.get("target_agent") or ""))
        task.add_done_callback(lambda _t, c=cid: self._inbound.pop(c, None))

    async def _resolve_local(self, ref: str) -> dict[str, Any] | None:
        r = (ref or "").strip().lower()
        for a in await self._exposed():
            if r in (str(a.get("name") or "").lower(), str(a.get("slug") or "").lower()):
                return a
        return None

    async def _serve(self, d: dict[str, Any]) -> None:
        cid = str(d.get("concern_id"))
        from_deck = str(d.get("from_deck") or "")
        from_label = _from_label(d)
        s = self.settings
        self.counters["inbound"] += 1

        # Only a deck's name is hub-verified (bound to its key); anything else
        # (a legacy agent, the hub's swarm) names itself, so it is let in only
        # when this deck accepts everyone.
        allowed = {x.strip().lower() for x in s.accept_from}
        who = from_deck.lower()
        if "*" not in allowed and (not from_deck or who not in allowed):
            self.counters["inbound_refused"] += 1
            asker = f"'{from_deck}'" if from_deck else f"agent '{d.get('from_instance_name') or '?'}' (not a Flight Deck)"
            await self._finish(cid, ok=False, error=f"Flight Deck '{self.hub_deck_name or s.deck_name}' "
                                                    f"does not accept requests from {asker}")
            self._note("in", from_label, "", False, "refused (accept_from)")
            return

        target = str(d.get("target_agent") or "")
        agent = await self._resolve_local(target)
        if agent is None:
            self.counters["inbound_refused"] += 1
            exposed = ", ".join(a.get("name", "") for a in await self._exposed()) or "none"
            await self._finish(cid, ok=False, error=f"No exposed agent '{target}' on this deck (exposed: {exposed})")
            self._note("in", from_label, target, False, "unknown agent")
            return
        name = str(agent.get("name") or target)
        if agent.get("status") not in ("running", "asleep") or not agent.get("port"):
            await self._finish(cid, ok=False, error=f"'{name}' is not running right now")
            self._note("in", from_label, name, False, "not running")
            return
        busy = [i for i in self._inbound.values() if i.concern_id != cid]
        if len(busy) >= s.max_concurrent:
            await self._finish(cid, ok=False, error=f"this deck is busy ({len(busy)} remote tasks running) — retry later")
            self._note("in", from_label, name, False, "busy")
            return
        if self._consult_events is None:
            await self._finish(cid, ok=False, error="this deck cannot run remote tasks (not wired)")
            return

        inb = self._inbound.get(cid)
        if inb is not None:
            inb.agent = name
        await self._send({"type": "progress", "concern_id": cid, "event": "accepted", "data": {"agent": name}})

        port = int(agent["port"])
        auth = await asyncio.to_thread(self._resolve_auth, port)  # Docker lookup: off the loop
        timeout = float(d.get("timeout_seconds") or 600.0)
        timeout = max(30.0, min(timeout, s.max_inbound_timeout))
        sender = f"{d.get('from_agent')} on " if d.get("from_agent") else ""
        origin = (f"Flight Deck '{from_deck}'" if from_deck
                  else f"agent '{d.get('from_instance_name') or '?'}' (not a Flight Deck)")
        message = (
            f"[Request from {sender}{origin} via BotPort. "
            f"Answer it fully in your reply; you cannot message them back.]\n\n{d.get('task') or ''}"
        )
        last_sent = 0.0
        pending: dict[str, Any] | None = None   # newest frame held back by the throttle
        flusher: asyncio.Task | None = None

        async def _flush_later(delay: float) -> None:
            nonlocal pending, last_sent
            await asyncio.sleep(delay)
            frame, pending = pending, None
            if frame is not None:
                last_sent = time.monotonic()
                await self._send(frame)

        final: dict[str, Any] | None = None
        try:
            async with contextlib.aclosing(self._consult_events(
                "localhost", port, auth, message,
                source_name=f"{d.get('from_agent') or 'agent'}@{from_deck or 'botport'}",
                timeout=timeout, no_flow=True, no_broadcast=True, automation=dict(REMOTE_AUTOMATION),
                deny_tools=sorted({*REMOTE_DENY_TOOLS, *s.deny_tools}),
                lane=remote_lane(from_deck if from_deck else "agent:" + str(d.get("from_instance_name") or "")),
            )) as events:
                async for evt in events:
                    if "ok" in evt and (evt.get("done") or evt.get("ok") is False):
                        final = evt
                        break
                    kind = str(evt.get("event") or "")
                    data = evt.get("data") if isinstance(evt.get("data"), dict) else {}
                    if inb is not None:
                        inb.last_event = kind
                    frame = {"type": "progress", "concern_id": cid,
                             "event": _progress_kind(kind), "data": _progress_data(kind, data)}
                    now = time.monotonic()
                    wait = _PROGRESS_MIN_INTERVAL - (now - last_sent)
                    if kind != "heartbeat" and wait > 0:
                        # Throttled: keep the newest, send it when the interval is up.
                        pending = frame
                        if flusher is None or flusher.done():
                            flusher = asyncio.create_task(_flush_later(wait))
                        continue
                    pending = None
                    last_sent = now
                    await self._send(frame)
        except asyncio.CancelledError:
            self._note("in", from_label, name, None, "cancelled")
            return
        except Exception as exc:
            final = {"ok": False, "error": f"{name} failed: {exc}"}
        finally:
            if flusher is not None:
                flusher.cancel()

        if final is None:
            final = {"ok": False, "error": f"{name} produced no answer"}
        if final.get("ok") and final.get("member_private") in ("data", "content"):
            # Marked for the agent's owner only (members' private data): it
            # does not leave this deck, whoever the other deck belongs to.
            final = {"ok": False, "error": f"{name}'s answer drew on private data that cannot leave this "
                                           f"Flight Deck"}
        if final.get("ok"):
            response = str(final.get("response") or "(no response)")
            meta: dict[str, Any] = {"agent": name}
            if isinstance(final.get("usage"), dict):
                meta["usage"] = {k: final["usage"].get(k) for k in ("model", "input_tokens", "output_tokens",
                                                                    "prompt_tokens", "completion_tokens")
                                 if k in final["usage"]}
            self.counters["inbound_ok"] += 1
            self._note("in", from_label, name, True)
            await self._finish(cid, ok=True, response=response, metadata=meta)
        else:
            self.counters["inbound_failed"] += 1
            self._note("in", from_label, name, False, str(final.get("error") or ""))
            await self._finish(cid, ok=False, error=str(final.get("error") or "failed"))

    async def _finish(self, cid: str, *, ok: bool, response: str = "", error: str = "",
                      metadata: dict[str, Any] | None = None) -> None:
        frame = {"type": "result", "concern_id": cid, "ok": ok, "response": response,
                 "error": error, "metadata": metadata or {}}
        self._outbox[cid] = frame
        while len(self._outbox) > 1000:
            self._outbox.popitem(last=False)
        await self._send(frame)

    # ── asking other decks (outbound) ────────────────────────────

    def find_remote(self, deck: str, agent: str) -> dict[str, Any] | None:
        d, a = (deck or "").strip().lower(), (agent or "").strip().lower()
        for r in self.remote_agents():
            if r["deck"].lower() == d and a in (r["agent"].lower(), str(r.get("slug") or "").lower()):
                return r
        return None

    async def consult(self, *, target_deck: str, target_agent: str, message: str,
                      from_agent: str = "", timeout: float = 480.0, mode: str = "consult",
                      context: str = "") -> AsyncIterator[dict[str, Any]]:
        """Ask ``target_agent`` on ``target_deck``; yields /fd/consult-peer-style events
        and ends with ``{"ok": True, "done": True, "response": ...}`` or ``{"ok": False, "error": ...}``."""
        if not self.settings.enabled:
            yield {"ok": False, "error": "This Flight Deck's BotPort link is switched off"}
            return
        if not self.settings.allow_outbound:
            yield {"ok": False, "error": "This Flight Deck does not allow its agents to consult remote agents"}
            return
        if not self.connected:
            why = f" ({self.last_error})" if self.last_error else ""
            yield {"ok": False, "error": f"BotPort link is not connected{why}"}
            return
        remote = self.find_remote(target_deck, target_agent)
        if remote is None:
            available = ", ".join(r["name"] for r in self.remote_agents()) or "none"
            yield {"ok": False, "error": f"No remote agent '{target_agent}@{target_deck}' on the BotPort "
                                         f"network (available: {available})"}
            return

        cid = uuid.uuid4().hex
        timeout = max(10.0, min(float(timeout or 480.0), 3600.0))
        submit = {
            "type": "concern", "concern_id": cid, "task": message,
            "context": {"summary": context} if context else {},
            "target_deck": remote["deck"], "target_agent": remote.get("slug") or remote["agent"],
            "from_agent": from_agent[:128], "mode": mode, "timeout_seconds": timeout,
        }
        loop = asyncio.get_running_loop()
        req = _Outbound(cid, submit, asyncio.Queue(), loop.time() + timeout + 15.0, remote["name"])
        self._outbound[cid] = req
        self.counters["outbound"] += 1
        started = loop.time()
        await self._send(submit)
        yield {"event": "status", "data": {"status": f"Sending to {remote['name']} via BotPort…"}}
        try:
            while True:
                remaining = req.deadline - loop.time()
                if remaining <= 0:
                    self.counters["outbound_failed"] += 1
                    self._note("out", req.target, "", False, "timed out")
                    yield {"ok": False, "error": f"Timed out waiting for {remote['name']} after {int(timeout)}s"}
                    return
                try:
                    evt = await asyncio.wait_for(req.queue.get(), timeout=min(remaining, 15.0))
                except asyncio.TimeoutError:
                    yield {"event": "heartbeat", "data": {"elapsed": int(loop.time() - started),
                                                         "timeout": int(timeout)}}
                    continue
                yield evt
                if "ok" in evt:
                    return
        finally:
            if self._outbound.pop(cid, None) is not None:
                # The caller gave up (timeout, client gone): cancel remotely.
                self._remember_done(cid)
                asyncio.ensure_future(self._send({"type": "close_concern", "concern_id": cid}))


def _from_label(d: dict[str, Any]) -> str:
    deck = str(d.get("from_deck") or "")
    if not deck:
        return f"agent {d.get('from_instance_name') or '?'}"
    agent = str(d.get("from_agent") or "")
    return f"{agent}@{deck}" if agent else deck


def _progress_kind(kind: str) -> str:
    return {"status": "status", "thinking": "thinking", "monitor": "tool",
            "tool_stream": "tool", "heartbeat": "heartbeat"}.get(kind, "status")


def _progress_data(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    if kind == "heartbeat":
        return {"elapsed": data.get("elapsed"), "timeout": data.get("timeout")}
    if kind == "thinking":
        return {"tool": str(data.get("tool") or "")[:80], "text": str(data.get("text") or "")[:200]}
    if kind in ("monitor", "tool_stream"):
        return {"tool_name": str(data.get("tool_name") or data.get("tool") or "")[:80]}
    return {"status": str(data.get("status") or data.get("text") or "")[:200]}


def _deck_rules() -> tuple[bool, Callable[[str, str], bool]]:
    """(sign-in on, "is this a Flight Deck–managed worker?") for this deck."""
    try:
        from captain_claw.flight_deck.agent_sharing import is_managed_agent
        from captain_claw.flight_deck.auth import _fd_auth_enabled

        return _fd_auth_enabled(), is_managed_agent
    except Exception:  # pragma: no cover - import guard
        return False, lambda slug, description: False


def _describe(exc: Exception) -> str:
    try:
        from websockets.exceptions import InvalidStatus, InvalidURI

        if isinstance(exc, InvalidStatus):
            code = getattr(getattr(exc, "response", None), "status_code", "?")
            if code == 429:
                return "hub refused: too many failed registrations from this address — wait and check the key/secret"
            if code == 404:
                return "hub answered 404 — is the URL path right (…/ws)?"
            return f"hub answered HTTP {code} instead of a WebSocket upgrade"
        if isinstance(exc, InvalidURI):
            return f"invalid hub URL: {exc}"
    except Exception:
        pass
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timed out connecting to the hub"
    if isinstance(exc, ConnectionRefusedError):
        return "connection refused — is BotPort running and the port open?"
    if isinstance(exc, OSError):
        return f"network error: {exc.strerror or exc}"
    return f"{type(exc).__name__}: {exc}"


# ── Persistence helpers + the process-wide link ──────────────────


_LINK: BotPortLink | None = None


def get_link() -> BotPortLink:
    global _LINK
    if _LINK is None:
        _LINK = BotPortLink()
    return _LINK


def seal_settings(data: dict[str, Any]) -> str:
    """Stored form: JSON, with the secret sealed when encryption is available."""
    out = dict(data)
    secret = out.pop("secret", "")
    if secret:
        try:
            from captain_claw.flight_deck import fd_crypto

            if fd_crypto.available():
                out["secret_sealed"] = fd_crypto.seal(secret)
            else:
                out["secret"] = secret
        except Exception:
            out["secret"] = secret
    return json.dumps(out)


def unseal_settings(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    sealed = data.pop("secret_sealed", "")
    if sealed:
        try:
            from captain_claw.flight_deck import fd_crypto

            data["secret"] = fd_crypto.unseal(sealed) or ""
        except Exception:
            data["secret"] = ""
    return data
