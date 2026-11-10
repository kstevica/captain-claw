"""External memory — each user's memories, and everything done with them.

A person can have several external memories (up to :data:`MAX_MEMORIES` of
their own), all of them shared by every agent of theirs. They live ONLY in
Flight Deck, like Jev's key:

* Their own, per user in ``user_settings`` — ``external_memory:memories``, a
  JSON list of ``{id, name, provider, credentials, info, verified,
  verify_error, verified_at, updated_at, checked_at, private_ok, priority,
  mirror, enabled}`` (the credentials sealed with ``fd_crypto`` when it is
  usable). ``external_memory:`` is server-owned: ``/fd/settings`` never
  returns it and refuses to write it. An auth-disabled deck keeps its
  memories in ``system_settings``. A person's single connection from before
  (``external_memory:connection`` + ``:mirror`` + ``:priority``) is moved into
  the list the first time it is read.
* The deck's: an admin may connect one *scoped* provider (Hindsight: one
  server, a bank per person; Qdrant / ArcadeDB: one collection / database, a
  space per person) for everyone — ``external_memory:deck_connection`` in
  ``system_settings``. It is one more memory in each person's list (id
  ``deck``), in a scope of their own (:func:`base.scope_id`), with their own
  priority and choices (``external_memory:deck_prefs``); they can switch it
  off. Someone who had a connection of their own before (which used to
  replace the deck's) starts with it off. A *key* provider (Membase: one
  account per person) can't be the deck's — one key would pour every user's
  facts into one memory.
* Agents never hold the credentials; they reach the memories through
  ``/fd/external-memory/agent/*``, the owner resolved from this deck's
  records, never from the agent's word, and never choose a scope.
* A user who isn't an admin can't point their own memory at this machine or
  its network (:func:`base.private_address`) — Flight Deck would make those
  requests for them. The deck's, set by an admin, can.

Each memory has its own *reading priority* (:data:`PRIORITIES`): ``first`` —
the person's main memory (more items, kept when the prompt is crowded,
trusted over the agent's own notes); ``alongside`` (default) — next to
Captain's own memory; ``on_demand`` — nothing recalled on its own. And its own
*mirror* choices: which of what agents learn is copied into it.

Three ways things get in:

* The agent tool's actions (:func:`run_for_owner`). ``remember`` without a
  memory named saves to every memory that can save; reading actions (search,
  list, …) without one look in all of them; anything else names one.
* Mirroring (:func:`mirror_for_owner`): insights, intentions, reflections,
  intuitions (from the agents) and lessons (taught here) are queued in the
  outbox once per memory that takes that kind, and written by
  :func:`drain_loop`, one at a time, retried with backoff. While a memory is
  connected, agents pick facts out of their chats every
  :data:`CAPTURE_EVERY` messages (Captain's own pace is 20).

And one way out besides the tool: automatic recall (:func:`recall_for_owner`).
Before a reply the agent sends the person's message; every memory not
``on_demand`` is asked at once and returns only what bears on it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from captain_claw.flight_deck.auth import _LOCAL_USER, _fd_auth_enabled, get_db
from captain_claw.flight_deck.external_memory import providers
from captain_claw.flight_deck.external_memory.base import (
    COMMON_RULES,
    DEFAULT_MIRROR,
    MIRROR_KINDS,
    ActionResult,
    ConnectionInfo,
    ExternalMemoryProvider,
    MemoryItem,
    ProviderAuthError,
    ProviderError,
    ProviderInputError,
    UnknownAction,
    private_address,
    scope_id,
)
from captain_claw.flight_deck.external_memory.outbox import Outbox
from captain_claw.logging import get_logger

log = get_logger(__name__)

_K_MEMORIES = "external_memory:memories"
_K_DECK_PREFS = "external_memory:deck_prefs"
_K_DECK = "external_memory:deck_connection"
# From before a person could have several (read once, to move them over).
_K_CONN = "external_memory:connection"
_K_MIRROR = "external_memory:mirror"
_K_PRIORITY = "external_memory:priority"

MAX_MEMORIES = 5
DECK_ID = "deck"
MAX_NAME = 40

PRIORITIES: tuple[str, ...] = ("first", "alongside", "on_demand")
DEFAULT_PRIORITY = "alongside"
# Automatic recall: (items at most, seconds Flight Deck gives the provider).
RECALL: dict[str, tuple[int, float]] = {"first": (8, 6.0), "alongside": (5, 4.0)}
# While a memory is connected, agents pick facts out of their chats every this
# many of the person's messages (Captain's own pace is 20).
CAPTURE_EVERY = 5

NOT_CONNECTED = (
    "No external memory is connected — the person connects one in Flight Deck → Connections → "
    "External memory."
)
PRIVATE_REFUSED = (
    "That address is on this machine or its network, which only an admin may connect (for the whole "
    "deck, under Connections → External memory). Use a public address, or ask your admin."
)
# How old a connection's stored capabilities may get before an agent's status
# call re-checks them in the background (reach and access change in the
# service itself).
INFO_MAX_AGE_S = 30 * 60
MAX_MIRROR_CONTENT = 20_000
DRAIN_BATCH = 20
DRAIN_IDLE_S = 30.0
SEND_GAP_S = 0.25
RATE_LIMIT_HOLD_S = 60.0
HOLD_S = 3600.0
PRUNE_EVERY_S = 6 * 3600
_HOST_CHECK_TTL_S = 60.0


class NotConnected(LookupError):  # noqa: N818 — reads as the condition
    def __init__(self, message: str = NOT_CONNECTED) -> None:
        super().__init__(message)


@dataclass
class Memory:
    """One of a person's memories: their own (``source`` "own") or the deck's
    ("deck"), with their scope on it ("" for a key provider) and their
    choices for it."""

    id: str
    name: str
    source: str
    conn: dict[str, Any]
    scope: str
    priority: str = DEFAULT_PRIORITY
    mirror: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_MIRROR))
    enabled: bool = True

    @property
    def provider_name(self) -> str:
        return str(self.conn["provider"])

    @property
    def label(self) -> str:
        cls = providers.provider_class(self.provider_name)
        return cls.label if cls else self.provider_name


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ── storage ─────────────────────────────────────────────────────────


def _deck_scope() -> bool:
    """Auth off: one set of memories for the whole deck, in ``system_settings``."""
    return not _fd_auth_enabled()


def owner_key(owner: str) -> str:
    """The id the outbox, scopes and usage log file things under."""
    return _LOCAL_USER["id"] if _deck_scope() else owner


async def _read(owner: str, key: str) -> str:
    db = get_db()
    if _deck_scope():
        return (await db.get_system_setting(key)) or ""
    if not owner:
        return ""
    return (await db.get_setting(owner, key)) or ""


async def _write(owner: str, key: str, value: str) -> None:
    db = get_db()
    if _deck_scope():
        await db.set_system_setting(key, value)
        return
    await db.set_settings(owner, {key: value})


async def _erase(owner: str, key: str) -> None:
    db = get_db()
    if _deck_scope():
        # system_settings has no delete; an empty value reads as "none".
        await db.set_system_setting(key, "")
        return
    await db.delete_setting(owner, key)


def _seal(credentials: dict[str, str]) -> dict[str, Any]:
    from captain_claw.flight_deck import fd_crypto

    if fd_crypto.available():
        try:
            return {"sealed": fd_crypto.seal(credentials)}
        except fd_crypto.CryptoUnavailable:
            pass
    return {"plain": credentials}


def _unseal(stored: Any) -> dict[str, str]:
    if not isinstance(stored, dict):
        return {}
    if "sealed" in stored:
        from captain_claw.flight_deck import fd_crypto

        try:
            value = fd_crypto.unseal(str(stored.get("sealed") or ""))
        except fd_crypto.CryptoUnavailable:
            value = None
        return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}
    plain = stored.get("plain")
    return {str(k): str(v) for k, v in plain.items()} if isinstance(plain, dict) else {}


def _open(raw: Any) -> dict[str, Any] | None:
    """A stored connection (a dict as kept) with its credentials opened, or
    None. One to a provider this deck no longer knows, or whose credentials
    can't be opened (the data key changed), reads as none."""
    if not isinstance(raw, dict) or not providers.provider_class(str(raw.get("provider") or "")):
        return None
    creds = _unseal(raw.get("credentials"))
    if not creds:
        return None
    conn = dict(raw)
    conn["credentials"] = creds
    conn["info"] = ConnectionInfo.from_dict(raw.get("info"))
    return conn


def _parse(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        return _open(json.loads(raw))
    except (TypeError, ValueError):
        return None


def _closed(conn: dict[str, Any]) -> dict[str, Any]:
    """A connection as it is kept: credentials sealed, info as a dict."""
    out = dict(conn)
    out["credentials"] = _seal(conn.get("credentials") or {})
    info = conn.get("info")
    out["info"] = info.to_dict() if isinstance(info, ConnectionInfo) else (info or {})
    return out


def _clean_priority(value: Any) -> str:
    value = str(value or "").strip().lower()
    return value if value in PRIORITIES else DEFAULT_PRIORITY


def _clean_mirror(value: Any, base: dict[str, bool] | None = None) -> dict[str, bool]:
    out = dict(base or DEFAULT_MIRROR)
    if isinstance(value, dict):
        for kind in MIRROR_KINDS:
            if isinstance(value.get(kind), bool):
                out[kind] = value[kind]
    return out


def _json(raw: str, default: Any) -> Any:
    try:
        value = json.loads(raw) if raw else default
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


async def _raw_memories(owner: str) -> list[dict[str, Any]]:
    """*owner*'s own memories as kept (credentials sealed). A connection
    from before there could be several is moved into the list here, once."""
    raw = await _read(owner, _K_MEMORIES)
    if raw:
        return [m for m in _json(raw, []) if isinstance(m, dict) and m.get("id")]
    legacy = _json(await _read(owner, _K_CONN), {})
    mirror = _clean_mirror(_json(await _read(owner, _K_MIRROR), {}))
    priority = _clean_priority(await _read(owner, _K_PRIORITY))
    memories: list[dict[str, Any]] = []
    if legacy.get("provider"):
        cls = providers.provider_class(str(legacy["provider"]))
        memories.append({**legacy, "id": "m_" + secrets.token_hex(4),
                         "name": cls.label if cls else str(legacy["provider"]),
                         "priority": priority, "mirror": mirror, "enabled": True})
        # Their own used to replace the deck's: keep it that way for them.
        if not await _read(owner, _K_DECK_PREFS):
            await _write(owner, _K_DECK_PREFS, json.dumps({"enabled": False, "priority": priority,
                                                           "mirror": mirror}))
    await _write(owner, _K_MEMORIES, json.dumps(memories))
    if legacy:
        await _erase(owner, _K_CONN)
    return memories


async def _save_memories(owner: str, memories: list[dict[str, Any]]) -> None:
    await _write(owner, _K_MEMORIES, json.dumps(memories, ensure_ascii=True))


async def _edit_memory(owner: str, memory_id: str, change: Callable[[dict[str, Any]], None]) -> bool:
    """Apply *change* to one of *owner*'s own memories as kept, and save."""
    memories = await _raw_memories(owner)
    for m in memories:
        if m.get("id") == memory_id:
            change(m)
            await _save_memories(owner, memories)
            return True
    return False


async def _deck_prefs(owner: str) -> dict[str, Any]:
    """*owner*'s use of the deck's memory: on/off, priority, mirror choices."""
    await _raw_memories(owner)  # moves an old connection over first (it sets these)
    prefs = _json(await _read(owner, _K_DECK_PREFS), {})
    if not prefs:  # someone who was on the deck's memory: their old choices
        prefs = {"enabled": True, "priority": await _read(owner, _K_PRIORITY),
                 "mirror": _json(await _read(owner, _K_MIRROR), {})}
    return {"enabled": prefs.get("enabled") is not False, "priority": _clean_priority(prefs.get("priority")),
            "mirror": _clean_mirror(prefs.get("mirror"))}


async def load_deck_connection() -> dict[str, Any] | None:
    """The admin's connection for everyone, or None (always None with sign-in
    off — that deck's memories are already everyone's)."""
    if _deck_scope():
        return None
    return _parse((await get_db().get_system_setting(_K_DECK)) or "")


def _deck_name(deck: dict[str, Any]) -> str:
    cls = providers.provider_class(str(deck.get("provider") or ""))
    return f"{cls.label if cls else deck.get('provider')} (shared)"


async def memories(owner: str, *, enabled_only: bool = True) -> list[Memory]:
    """*owner*'s memories: their own (in the order they added them), then the
    deck's. Ones whose credentials can't be opened are left out."""
    out: list[Memory] = []
    for raw in await _raw_memories(owner):
        conn = _open(raw)
        if not conn:
            continue
        cls = providers.provider_class(conn["provider"])
        scope = ""
        if cls is not None and cls.tenancy == "scoped":
            named = str(conn["credentials"].get(cls.scope_field) or "") if cls.scope_field else ""
            scope = named or scope_id(owner_key(owner))
        out.append(Memory(
            id=str(raw["id"]), name=str(raw.get("name") or (cls.label if cls else conn["provider"])),
            source="own", conn=conn, scope=scope, priority=_clean_priority(raw.get("priority")),
            mirror=_clean_mirror(raw.get("mirror")), enabled=raw.get("enabled") is not False,
        ))
    deck = await load_deck_connection()
    if deck:
        prefs = await _deck_prefs(owner)
        out.append(Memory(id=DECK_ID, name=_deck_name(deck), source="deck", conn=deck,
                          scope=scope_id(owner_key(owner)), priority=prefs["priority"],
                          mirror=prefs["mirror"], enabled=prefs["enabled"]))
    return [m for m in out if m.enabled or not enabled_only]


async def find_memory(owner: str, ref: str, *, enabled_only: bool = False) -> Memory | None:
    """One of *owner*'s memories by id or name (any case)."""
    ref = str(ref or "").strip().lower()
    for m in await memories(owner, enabled_only=enabled_only):
        if ref in (m.id.lower(), m.name.lower()):
            return m
    return None


async def load_connection(owner: str) -> dict[str, Any] | None:
    """*owner*'s first own memory's connection, or None."""
    for m in await memories(owner, enabled_only=False):
        if m.source == "own":
            return m.conn
    return None


async def _update(owner: str, mem: Memory, **fields: Any) -> None:
    """Store what was learned about a memory's connection (verified, info…)."""
    if mem.source == "deck":
        deck = await load_deck_connection()
        if deck:
            deck.update(fields)
            await get_db().set_system_setting(_K_DECK, json.dumps(_closed(deck), ensure_ascii=True))
        return
    closed = _closed({"credentials": {}, **fields})
    closed.pop("credentials", None)

    def change(m: dict[str, Any]) -> None:
        m.update({k: v for k, v in closed.items() if k in fields})

    await _edit_memory(owner, mem.id, change)


# ── names, priorities, rules ────────────────────────────────────────


def _unique_name(wanted: str, taken: set[str]) -> str:
    base = " ".join(str(wanted or "").split())[:MAX_NAME] or "Memory"
    name, n = base, 2
    while name.lower() in taken:
        name = f"{base[:MAX_NAME - 3]} {n}"
        n += 1
    return name


def _check_name(name: Any) -> str:
    name = " ".join(str(name or "").split())
    if not name:
        raise ProviderInputError("Give the memory a name.")
    if len(name) > MAX_NAME or not re.fullmatch(r"[\w .()&+-]+", name):
        raise ProviderInputError(f"A memory's name is at most {MAX_NAME} letters, digits, spaces and . ( ) & + -")
    return name


_PRIORITY_RULES = {
    "first": (
        "This is the person's MAIN memory. Before each reply what it holds on the message is put in front "
        "of you; search it FIRST — before your own notes, files or the web — whenever the person asks about "
        "themselves, people, their work, plans or past decisions. When it and your own notes disagree, go "
        "by it (and by the newer of two facts)."
    ),
    "alongside": (
        "Before each reply what it holds on the message is put in front of you, next to your own notes; "
        "search it for more whenever the person asks about themselves, people, their work or past decisions."
    ),
    "on_demand": (
        "Nothing from it is put in front of you on its own — search it whenever the person asks about "
        "themselves, people, their work or past decisions."
    ),
}


def priority_rule(priority: str) -> str:
    """The rule telling an agent how much to lean on a memory."""
    return _PRIORITY_RULES.get(priority, _PRIORITY_RULES[DEFAULT_PRIORITY])


_KIND_PLURAL = {
    "insight": "insights", "intention": "intentions (once active)", "reflection": "reflections",
    "lesson": "lessons the person taught", "intuition": "intuitions",
}


def mirror_rule(mirror: dict[str, bool]) -> str:
    """The rule telling an agent what is copied into a memory on its own."""
    kinds = [_KIND_PLURAL[k] for k in MIRROR_KINDS if mirror.get(k)]
    if not kinds:
        return "Nothing is copied here automatically — save what matters with remember."
    return "Copied here automatically — never save these again by hand: " + ", ".join(kinds) + "."


# ── building a provider ─────────────────────────────────────────────

_host_checks: dict[str, tuple[bool, float]] = {}


async def _is_private(url: str) -> bool:
    hit = _host_checks.get(url)
    if hit and time.time() - hit[1] < _HOST_CHECK_TTL_S:
        return hit[0]
    private = await asyncio.to_thread(private_address, url)
    _host_checks[url] = (private, time.time())
    return private


async def provider_for(mem: Memory, **kwargs: Any) -> ExternalMemoryProvider:
    """The provider for one call. A user's own memory that may not reach
    private addresses is checked again here (names can change what they
    point at)."""
    cls = providers.provider_class(mem.provider_name)
    if mem.source == "own" and cls is not None and not mem.conn.get("private_ok"):
        for name in cls.address_fields():
            url = str(mem.conn["credentials"].get(name) or "")
            if url and await _is_private(url):
                raise ProviderError(PRIVATE_REFUSED, status=403, code="private_address")
    return providers.build(mem.provider_name, mem.conn["credentials"], mem.conn.get("info"),
                           scope=mem.scope, **kwargs)


def _plain_provider(mem: Memory) -> ExternalMemoryProvider:
    """For views only (no network): actions, rules, hints."""
    return providers.build(mem.provider_name, mem.conn["credentials"], mem.conn.get("info"), scope=mem.scope)


# ── views ───────────────────────────────────────────────────────────


def _conn_view(conn: dict[str, Any], provider: ExternalMemoryProvider) -> dict[str, Any]:
    verify_error = conn.get("verify_error")
    return {
        "provider": provider.name,
        "label": provider.label,
        "credential_hints": provider.credential_hints(),
        "verified": conn.get("verified") if isinstance(conn.get("verified"), bool) else None,
        "verify_error": verify_error if verify_error in ("refused", "unreachable") else None,
        "verified_at": conn.get("verified_at") or None,
        "updated_at": conn.get("updated_at") or None,
    }


async def _deck_offer() -> dict[str, Any] | None:
    deck = await load_deck_connection()
    if not deck:
        return None
    cls = providers.provider_class(str(deck["provider"]))
    return {"provider": str(deck["provider"]), "label": cls.label if cls else str(deck["provider"]),
            "scope_label": cls.scope_label if cls else "", "name": _deck_name(deck)}


async def _memory_view(owner: str, mem: Memory) -> dict[str, Any]:
    provider = _plain_provider(mem)
    view = _conn_view(mem.conn, provider)
    if mem.source == "deck":
        view["credential_hints"] = {}  # the admin's, not this user's to see
    try:
        counts = await outbox().counts(owner_key(owner), target=mem.id)
    except Exception as exc:  # the dashboard still shows the memory
        log.debug("external memory: outbox counts failed", error=type(exc).__name__)
        counts = {}
    return {
        **view,
        "id": mem.id,
        "name": mem.name,
        "source": mem.source,
        "enabled": mem.enabled,
        "priority": mem.priority,
        "mirror": mem.mirror,
        "tenancy": provider.tenancy,
        "scope_label": provider.scope_label,
        "scope_id": mem.scope,
        "reaches": provider.reaches(),
        "actions": [a.to_dict() for a in provider.actions()],
        "rules": [*provider.rules(), priority_rule(mem.priority), mirror_rule(mem.mirror)],
        "outbox": counts,
    }


async def status_for_owner(owner: str) -> dict[str, Any]:
    """The person's memories as the dashboard shows them — never a credential."""
    mems = await memories(owner, enabled_only=False)
    try:
        counts = await outbox().counts(owner_key(owner))
    except Exception as exc:
        log.debug("external memory: outbox counts failed", error=type(exc).__name__)
        counts = {}
    return {
        "connected": any(m.enabled for m in mems),
        "memories": [await _memory_view(owner, m) for m in mems],
        "max_memories": MAX_MEMORIES,
        "providers": providers.catalog(),
        "mirror_kinds": list(MIRROR_KINDS),
        "default_mirror": dict(DEFAULT_MIRROR),
        "priorities": list(PRIORITIES),
        "capture_every": CAPTURE_EVERY,
        "scope": "deck" if _deck_scope() else "user",
        "auth_enabled": _fd_auth_enabled(),
        "deck_connection": await _deck_offer(),
        "outbox": counts,
    }


async def deck_status() -> dict[str, Any]:
    """The deck's connection, as the admin sees it (never a credential)."""
    deck = await load_deck_connection()
    out: dict[str, Any] = {
        "available": not _deck_scope(),
        "connected": False,
        "providers": [p for p in providers.catalog() if p.get("tenancy") == "scoped"],
    }
    if not deck:
        return out
    provider = providers.build(str(deck["provider"]), deck["credentials"], deck.get("info"))
    return {**out, **_conn_view(deck, provider), "connected": True,
            "scope_label": provider.scope_label, "scope_example": scope_id("<user id>")}


_refreshing: set[str] = set()


async def _refresh_info(owner: str, mem: Memory) -> None:
    """Re-learn what a memory may do (background, at most one per memory)."""
    key = DECK_ID if mem.source == "deck" else f"{owner}:{mem.id}"
    if key in _refreshing:
        return
    _refreshing.add(key)
    try:
        try:
            info = await (await provider_for(mem, max_retries=0)).verify()
        except ProviderAuthError:
            await _update(owner, mem, verified=False, verify_error="refused", checked_at=time.time())
            return
        except (ProviderError, ProviderInputError):
            await _update(owner, mem, checked_at=time.time())
            return
        await _update(owner, mem, info=info, verified=True, verify_error=None,
                      verified_at=_now(), checked_at=time.time())
    except Exception as exc:
        log.debug("external memory: background re-check failed", error=type(exc).__name__)
    finally:
        _refreshing.discard(key)


def _union_actions(mems: list[Memory]) -> list[dict[str, Any]]:
    """Every action any of the memories offers, its parameters merged; with
    several memories, a ``memory`` parameter on each and who offers what."""
    merged: dict[str, dict[str, Any]] = {}
    offered: dict[str, list[str]] = {}
    for m in mems:
        for a in _plain_provider(m).actions():
            d = a.to_dict()
            offered.setdefault(a.name, []).append(m.name)
            if a.name not in merged:
                merged[a.name] = d
                continue
            have = {p["name"] for p in merged[a.name]["params"]}
            merged[a.name]["params"] += [p for p in d["params"] if p["name"] not in have]
            merged[a.name]["writes"] = merged[a.name]["writes"] or d["writes"]
            merged[a.name]["destructive"] = merged[a.name]["destructive"] or d["destructive"]
    if len(mems) < 2:
        return list(merged.values())
    names = tuple(m.name for m in mems)
    for name, d in merged.items():
        who = offered[name]
        if len(who) < len(mems):
            d["description"] += f" (Only in: {', '.join(who)}.)"
        if name == "remember":
            how = "Which memory to save to; without it, every one that can save gets it."
        elif d["writes"] or d["destructive"]:
            how = "Which memory (needed when more than one offers this)."
        else:
            how = "Which memory to look in; without it, all of them."
        d["params"] = [*d["params"], {"name": "memory", "type": "string", "description": how,
                                      "required": False, "enum": list(names)}]
    return list(merged.values())


def _aggregate_priority(mems: list[Memory]) -> str:
    for p in ("first", "alongside"):
        if any(m.priority == p for m in mems):
            return p
    return "on_demand"


async def manifest_for_owner(owner: str) -> dict[str, Any]:
    """What an agent needs (no network): the person's memories, the actions
    and rules across them, which kinds to mirror, and how to recall."""
    mems = await memories(owner)
    if not mems:
        return {"connected": False}
    for m in mems:
        if time.time() - float(m.conn.get("checked_at") or 0) > INFO_MAX_AGE_S:
            asyncio.create_task(_refresh_info(owner, m))
    rules: list[str] = list(COMMON_RULES)
    if len(mems) > 1:
        rules.append(
            f"The person has {len(mems)} external memories: "
            + ", ".join(f"{m.name} ({m.label}, {m.priority.replace('_', ' ')})" for m in mems)
            + ". search and list look in all of them unless you name one with memory; remember saves to "
              "every one that can save unless you name one; forget and other changes name the memory "
              "the item came from."
        )
    for m in mems:
        provider = _plain_provider(m)
        own = [*provider.provider_rules(), *provider.info.rules, priority_rule(m.priority), mirror_rule(m.mirror)]
        rules += [f"{m.name}: {r}" for r in own] if len(mems) > 1 else own
    reaches = [r for m in mems for r in _plain_provider(m).reaches()]
    recall = [RECALL[m.priority] for m in mems if m.priority in RECALL]
    single = mems[0] if len(mems) == 1 else None
    return {
        "connected": True,
        "provider": single.provider_name if single else "several",
        "label": single.label if single else ", ".join(m.name for m in mems),
        "memories": [{"id": m.id, "name": m.name, "label": m.label, "provider": m.provider_name,
                      "priority": m.priority} for m in mems],
        "actions": _union_actions(mems),
        "rules": rules,
        "reaches": reaches,
        # The ones an AGENT sends (lessons are mirrored here, by Flight Deck);
        # Flight Deck routes each to the memories that take it.
        "mirror": [k for k in MIRROR_KINDS if k != "lesson" and any(m.mirror.get(k) for m in mems)],
        "refused": any(m.conn.get("verify_error") == "refused" for m in mems),
        # Reading priority: whether the agent asks for recall before a reply,
        # how long it waits, and where it puts what comes back (per memory,
        # in the recall answer).
        "priority": _aggregate_priority(mems),
        "recall_limit": max((n for n, _w in recall), default=0),
        "recall_timeout": max((w for _n, w in recall), default=0.0),
        "capture_every": CAPTURE_EVERY,
    }


# ── connecting ──────────────────────────────────────────────────────


async def _checked_conn(provider_name: str, raw_credentials: dict[str, Any], *, private_ok: bool,
                        scope: str = "") -> tuple[dict[str, Any], str, ExternalMemoryProvider]:
    """Clean and check credentials: ``(conn, warning, provider)``. Refused
    credentials raise :class:`ProviderAuthError` (nothing to save); a provider
    that couldn't be asked gives an unverified connection and a warning."""
    cls = providers.provider_class(provider_name)
    if cls is None:
        raise ProviderInputError(f"Unknown external memory '{provider_name}'")
    credentials = cls.clean_credentials(raw_credentials)
    if not private_ok:
        for name in cls.address_fields():
            url = str(credentials.get(name) or "")
            if url and await asyncio.to_thread(private_address, url):
                raise ProviderInputError(PRIVATE_REFUSED + " Nothing was saved.")
    provider = providers.build(cls.name, credentials, scope=scope)
    now = _now()
    conn: dict[str, Any] = {
        "provider": cls.name, "credentials": credentials, "info": ConnectionInfo(),
        "verified": None, "verify_error": None, "verified_at": None, "updated_at": now, "checked_at": 0,
        "private_ok": bool(private_ok),
    }
    warning = ""
    try:
        conn["info"] = await provider.verify()
        conn.update(verified=True, verified_at=now, checked_at=time.time())
    except ProviderAuthError:
        raise
    except ProviderError as exc:
        conn.update(verified=False, verify_error="unreachable")
        warning = f"Saved, but couldn't check it with {cls.label}: {exc}"
    return conn, warning, provider


async def connect(owner: str, provider_name: str, raw_credentials: dict[str, Any],
                  *, private_ok: bool = False, memory_id: str = "", name: str = "") -> tuple[dict[str, Any], str]:
    """Check credentials with the provider and save them: a new memory of
    *owner*'s, or — with *memory_id* — new credentials for one of their own.
    ``(status, warning)``. *private_ok*: may point at this machine or its
    network (admins, and a deck with sign-in off)."""
    if memory_id == DECK_ID:
        raise ProviderInputError("The shared memory is the admin's to change — connect one of your own instead.")
    kept = await _raw_memories(owner)
    current = next((m for m in kept if m.get("id") == memory_id), None) if memory_id else None
    if memory_id and current is None:
        raise ProviderInputError("That memory isn't there any more — add it again.")
    if not memory_id and len(kept) >= MAX_MEMORIES:
        raise ProviderInputError(f"You have {MAX_MEMORIES} memories already — remove one to add another.")
    taken = {str(m.name).lower() for m in await memories(owner, enabled_only=False) if m.id != memory_id}
    if name:
        name = _check_name(name)
        if name.lower() in taken:
            raise ProviderInputError(f"You already have a memory called “{name}”.")
    conn, warning, _provider = await _checked_conn(provider_name, raw_credentials, private_ok=private_ok)
    closed = _closed(conn)
    if current is not None:
        if current.get("provider") != conn["provider"]:
            # A different service: what was queued for the old one won't go there.
            await outbox().drop_owner(owner_key(owner), target=memory_id)

        def change(m: dict[str, Any]) -> None:
            m.update(closed)
            if name:
                m["name"] = name

        await _edit_memory(owner, memory_id, change)
        target = memory_id
    else:
        cls = providers.provider_class(conn["provider"])
        target = "m_" + secrets.token_hex(4)
        kept.append({**closed, "id": target, "name": name or _unique_name(cls.label if cls else "", taken),
                     "priority": DEFAULT_PRIORITY, "mirror": dict(DEFAULT_MIRROR), "enabled": True})
        await _save_memories(owner, kept)
    await outbox().retry_owner(owner_key(owner), target=target)
    wake()
    status = await status_for_owner(owner)
    status["memory_id"] = target
    return status, warning


async def disconnect(owner: str, memory_id: str) -> dict[str, Any]:
    """Remove one of *owner*'s own memories (and what was still queued for
    it). The deck's can only be switched off."""
    if memory_id == DECK_ID:
        raise ProviderInputError("The shared memory is the admin's — switch it off instead of removing it.")
    kept = await _raw_memories(owner)
    left = [m for m in kept if m.get("id") != memory_id]
    if len(left) == len(kept):
        raise NotConnected("That memory isn't there.")
    await _save_memories(owner, left)
    try:
        await outbox().drop_owner(owner_key(owner), target=memory_id)
    except Exception as exc:
        log.debug("external memory: outbox drop failed", error=type(exc).__name__)
    return await status_for_owner(owner)


async def update_memory(owner: str, memory_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Change how *owner* uses one memory: ``name`` (own only), ``priority``,
    ``mirror`` (kinds → bool), ``enabled``."""
    mem = await find_memory(owner, memory_id)
    if mem is None or mem.id != memory_id:
        raise NotConnected("That memory isn't there.")
    if "priority" in data and str(data.get("priority") or "").strip().lower() not in PRIORITIES:
        raise ProviderInputError(f"Reading priority is one of: {', '.join(PRIORITIES)}")
    if "enabled" in data and not isinstance(data.get("enabled"), bool):
        raise ProviderInputError("enabled is true or false")
    if "name" in data:
        if mem.source == "deck":
            raise ProviderInputError("The shared memory keeps its name.")
        name = _check_name(data["name"])
        taken = {m.name.lower() for m in await memories(owner, enabled_only=False) if m.id != memory_id}
        if name.lower() in taken:
            raise ProviderInputError(f"You already have a memory called “{name}”.")
        data = {**data, "name": name}

    def apply(m: dict[str, Any]) -> None:
        if "priority" in data:
            m["priority"] = _clean_priority(data["priority"])
        if isinstance(data.get("mirror"), dict):
            m["mirror"] = _clean_mirror(data["mirror"], m.get("mirror") or DEFAULT_MIRROR)
        if "enabled" in data:
            m["enabled"] = bool(data["enabled"])
        if "name" in data and mem.source == "own":
            m["name"] = data["name"]

    if mem.source == "deck":
        prefs = await _deck_prefs(owner)
        apply(prefs)
        await _write(owner, _K_DECK_PREFS, json.dumps(prefs))
    else:
        await _edit_memory(owner, memory_id, apply)
    if data.get("enabled") is True:
        await outbox().retry_owner(owner_key(owner), target=memory_id)
        wake()
    return await status_for_owner(owner)


async def deck_connect(provider_name: str, raw_credentials: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """The admin connects one scoped provider for everyone."""
    if _deck_scope():
        raise ProviderInputError("Sign-in is off on this deck, so its memories are already everyone's.")
    cls = providers.provider_class(provider_name)
    if cls is None:
        raise ProviderInputError(f"Unknown external memory '{provider_name}'")
    if not cls.shareable():
        raise ProviderInputError(
            f"A {cls.label} key is one person's memory — each person connects their own. "
            "Only a memory that keeps people apart itself can be the deck's."
        )
    conn, warning, _provider = await _checked_conn(provider_name, raw_credentials, private_ok=True,
                                                   scope=scope_id("check"))
    if cls.scope_field:
        conn["credentials"].pop(cls.scope_field, None)  # every person gets their own
    await get_db().set_system_setting(_K_DECK, json.dumps(_closed(conn), ensure_ascii=True))
    wake()
    return await deck_status(), warning


async def deck_disconnect() -> dict[str, Any]:
    if not _deck_scope():
        await get_db().set_system_setting(_K_DECK, "")
    return await deck_status()


async def _verify(owner: str, mem: Memory) -> dict[str, Any]:
    t0 = time.monotonic()
    try:
        info = await (await provider_for(mem)).verify()
    except ProviderAuthError:
        await _update(owner, mem, verified=False, verify_error="refused", checked_at=time.time())
        raise
    await _update(owner, mem, info=info, verified=True, verify_error=None,
                  verified_at=_now(), checked_at=time.time())
    return {"ok": True, "latency_ms": int((time.monotonic() - t0) * 1000)}


async def test(owner: str, memory_id: str = "") -> dict[str, Any]:
    """A live check of one memory (the only one, when none is named);
    updates what it may do."""
    mems = await memories(owner, enabled_only=False)
    mem = next((m for m in mems if m.id == memory_id), None) if memory_id else (mems[0] if len(mems) == 1 else None)
    if mem is None:
        raise NotConnected("Say which memory to check." if mems and not memory_id else NOT_CONNECTED)
    out = await _verify(owner, mem)
    await outbox().retry_owner(owner_key(owner), target=mem.id)
    wake()
    return {**out, "memory_id": mem.id, **await status_for_owner(owner)}


async def deck_test() -> dict[str, Any]:
    deck = await load_deck_connection()
    if not deck:
        raise NotConnected("No memory is connected for the deck.")
    out = await _verify("", Memory(id=DECK_ID, name="", source="deck", conn=deck, scope=scope_id("check")))
    return {**out, **await deck_status()}


# ── actions ─────────────────────────────────────────────────────────


async def _log_call(owner: str, *, source: str, agent: str, provider: str, action: str,
                    ok: bool, status: int, latency_ms: int, memory: str = "") -> None:
    """One ``usage_logs`` row per action. Never fails the call."""
    try:
        await get_db().log_usage(owner_key(owner), "external_memory_call", json.dumps({
            "source": source, "agent": agent or "", "provider": provider, "action": action,
            "ok": ok, "status": status, "latency_ms": latency_ms, "memory": memory,
        }))
    except Exception as exc:
        log.warning("external memory usage log failed: %s", type(exc).__name__)


async def _run_one(owner: str, mem: Memory, action: str, args: dict[str, Any], *, source: str,
                   agent: str) -> ActionResult:
    provider = await provider_for(mem)
    t0 = time.monotonic()
    ok, status = False, 0
    try:
        result = await provider.run(action, dict(args))
        ok, status = True, 200
    except ProviderAuthError as exc:
        status = exc.status
        await _update(owner, mem, verified=False, verify_error="refused")
        raise
    except ProviderError as exc:
        status = exc.status
        raise
    finally:
        await _log_call(owner, source=source, agent=agent, provider=provider.name, action=action,
                        ok=ok, status=status, latency_ms=int((time.monotonic() - t0) * 1000), memory=mem.name)
    if action == "capabilities" and provider.info.allowed:
        await _update(owner, mem, info=provider.info, verified=True, verify_error=None,
                      verified_at=_now(), checked_at=time.time())
    elif mem.conn.get("verified") is not True:
        await _update(owner, mem, verified=True, verify_error=None, verified_at=_now())
    return result


async def run_for_owner(owner: str, action: str, args: dict[str, Any], *, source: str,
                        agent: str = "", memory: str = "") -> ActionResult:
    """Run one action on *owner*'s memories, each in their scope: the one
    named by *memory* (or ``args["memory"]``); without one, ``remember`` and
    reading actions go to every memory that offers them, and anything else
    to the only one that does. Raises :class:`NotConnected`,
    :class:`ProviderInputError` (incl. an action no memory offers) or
    :class:`ProviderError`."""
    args = dict(args) if isinstance(args, dict) else {}
    ref = str(memory or args.pop("memory", "") or "").strip()
    args.pop("memory", None)
    mems = await memories(owner)
    if not mems:
        raise NotConnected()
    action = (action or "").strip().lower()
    if ref:
        mem = next((m for m in mems if ref.lower() in (m.id.lower(), m.name.lower())), None)
        if mem is None:
            raise ProviderInputError(f"No memory called “{ref}” — the person's memories: "
                                     + ", ".join(m.name for m in mems) + ".")
        _plain_provider(mem).action(action)  # UnknownAction before anything is sent or logged
        return await _run_one(owner, mem, action, args, source=source, agent=agent)
    offering = [(m, a) for m in mems for a in _plain_provider(m).actions() if a.name == action]
    if not offering:
        names = sorted({a.name for m in mems for a in _plain_provider(m).actions()})
        raise UnknownAction(f"No memory offers '{action}' — they offer: {', '.join(names) or 'none'}.")
    if len(offering) == 1:
        return await _run_one(owner, offering[0][0], action, args, source=source, agent=agent)
    act = offering[0][1]
    if action != "remember" and (act.writes or act.destructive):
        raise ProviderInputError(f"{action} works on one memory — say which with memory: "
                                 + ", ".join(m.name for m, _a in offering) + ".")
    results = await asyncio.gather(
        *(_run_one(owner, m, action, args, source=source, agent=agent) for m, _a in offering),
        return_exceptions=True,
    )
    parts: list[str] = []
    data: dict[str, Any] = {}
    failures: list[BaseException] = []
    for (m, _a), res in zip(offering, results, strict=True):
        title = m.name if m.label.lower() in m.name.lower() else f"{m.name} ({m.label})"
        if isinstance(res, BaseException):
            if not isinstance(res, (ProviderError, ProviderInputError)):
                raise res
            failures.append(res)
            parts.append(f"── {title} — failed: {res}")
            continue
        parts.append(f"── {title} ──\n{res.text}")
        data[m.id] = res.data
    if len(failures) == len(offering):
        raise failures[0]
    return ActionResult("\n\n".join(parts), {"memories": data})


# ── automatic recall ────────────────────────────────────────────────


async def _recall_one(owner: str, mem: Memory, query: str, *, agent: str) -> dict[str, Any]:
    limit, wait = RECALL[mem.priority]
    group: dict[str, Any] = {"id": mem.id, "name": mem.name, "label": mem.label, "priority": mem.priority,
                             "hits": []}
    t0 = time.monotonic()
    ok, status = False, 0
    try:
        provider = await provider_for(mem, max_retries=0)
        hits = await asyncio.wait_for(provider.recall(query, limit), timeout=wait)
        group["hits"] = [h.to_dict() for h in hits[:limit] if h.text.strip()]
        ok, status = True, 200
    except TimeoutError:
        group["error"], status = "the memory didn't answer in time", 504
    except ProviderAuthError as exc:
        status = exc.status
        group["error"] = "the memory refused the key"
        await _update(owner, mem, verified=False, verify_error="refused")
    except (ProviderError, ProviderInputError) as exc:
        status = getattr(exc, "status", 0) or 0
        group["error"] = str(exc)[:200]
    except Exception as exc:  # never the agent's problem
        log.warning("external memory: recall failed", error=type(exc).__name__)
        group["error"] = type(exc).__name__
    finally:
        await _log_call(owner, source="recall", agent=agent, provider=mem.provider_name, action="recall",
                        ok=ok, status=status, latency_ms=int((time.monotonic() - t0) * 1000), memory=mem.name)
    return group


async def recall_for_owner(owner: str, query: str, *, agent: str = "") -> dict[str, Any]:
    """What *owner*'s memories hold on *query*, for an agent about to reply:
    ``{connected, priority, label, groups: [{id, name, label, priority, hits,
    error?}], hits}`` — every memory not ``on_demand`` asked at once, each in
    its own time limit. Never raises: a slow or failing memory must not hold
    up or break the turn, it comes back with no hits and the reason."""
    mems = await memories(owner)
    if not mems:
        return {"connected": False, "hits": []}
    out: dict[str, Any] = {"connected": True, "priority": _aggregate_priority(mems),
                           "label": mems[0].label if len(mems) == 1 else ", ".join(m.name for m in mems),
                           "groups": [], "hits": []}
    query = str(query or "").strip()[:4000]
    asked = [m for m in mems if m.priority in RECALL]
    if not asked or not query:
        return out
    groups = await asyncio.gather(*(_recall_one(owner, m, query, agent=agent) for m in asked))
    out["groups"] = list(groups)
    out["hits"] = [h for g in groups for h in g["hits"]]
    errors = [f"{g['name']}: {g['error']}" for g in groups if g.get("error")]
    if errors:
        out["error"] = "; ".join(errors)
    return out


# ── mirroring ───────────────────────────────────────────────────────


async def mirror_for_owner(owner: str, item: MemoryItem) -> dict[str, Any]:
    """Queue one item for every one of *owner*'s memories that takes its kind:
    ``{queued, reason, memories}``. Never raises for "not wanted" — no
    memory, that kind turned off everywhere, or an item already queued are
    ordinary answers."""
    if item.kind not in MIRROR_KINDS:
        raise ProviderInputError(f"Unknown kind '{item.kind}' — one of {', '.join(MIRROR_KINDS)}")
    item.content = (item.content or "").strip()
    item.source_id = (item.source_id or "").strip()[:200]
    if not item.content or not item.source_id:
        raise ProviderInputError("A mirrored item needs content and a source_id")
    item.content = item.content[:MAX_MIRROR_CONTENT]
    mems = await memories(owner)
    if not mems:
        return {"queued": False, "reason": "not_connected", "memories": []}
    wanted = [m for m in mems if m.mirror.get(item.kind)]
    if not wanted:
        return {"queued": False, "reason": "kind_off", "memories": []}
    queued: list[str] = []
    for m in wanted:
        if await outbox().enqueue(
            owner_key(owner), m.provider_name, item.kind, item.source_id, item.to_dict(),
            agent=item.agent, title=item.title or item.content[:120], target=m.id,
        ):
            queued.append(m.name)
    if queued:
        wake()
    return {"queued": bool(queued), "reason": "" if queued else "already_queued", "memories": queued}


def _backoff(attempts: int) -> float:
    return min(30.0 * (2 ** max(0, attempts)), HOLD_S)


async def drain_once(limit: int = DRAIN_BATCH) -> int:
    """Write what is due; returns how many rows were looked at. Each row goes
    to its memory as the owner has it NOW, in their scope (a row from before
    several memories: to their memory of that provider)."""
    box = outbox()
    rows = await box.due(limit)
    owned: dict[str, list[Memory]] = {}
    held: set[tuple[str, str]] = set()
    for row in rows:
        owner_id, target = str(row["owner_id"]), str(row.get("target") or "")
        if (owner_id, target) in held:
            continue
        if owner_id not in owned:
            owned[owner_id] = await memories(owner_id)
        mem = next((m for m in owned[owner_id] if (m.id == target if target else m.provider_name == row["provider"])),
                   None)
        if mem is None or mem.provider_name != row["provider"]:
            await box.mark_failed(int(row["id"]), "No longer connected to this external memory.")
            continue
        try:
            item = MemoryItem.from_dict(json.loads(row["payload"]))
        except (TypeError, ValueError):
            await box.mark_failed(int(row["id"]), "Unreadable queued item.")
            continue
        try:
            remote_id = await (await provider_for(mem, max_retries=0)).mirror(item)
        except ProviderAuthError as exc:
            held.add((owner_id, target))
            await box.defer_owner(owner_id, HOLD_S, str(exc), target=target)
            await _update(owner_id, mem, verified=False, verify_error="refused")
            continue
        except ProviderError as exc:
            if exc.code in ("rate_limited",) or (exc.status == 429 and exc.retryable):
                held.add((owner_id, target))
                await box.defer_owner(owner_id, RATE_LIMIT_HOLD_S, str(exc), target=target)
            elif exc.retryable:
                await box.mark_retry(int(row["id"]), str(exc), _backoff(int(row["attempts"])))
            elif exc.code in ("quota_exceeded", "write_target_required", "private_address") or exc.status == 403:
                # Fixable by the person (or the admin): hold, don't fail.
                held.add((owner_id, target))
                await box.defer_owner(owner_id, HOLD_S, str(exc), target=target)
            else:
                await box.mark_failed(int(row["id"]), str(exc))
            continue
        except Exception as exc:  # noqa: BLE001 — one bad row never stops the drain
            log.warning("external memory: mirror failed", error=type(exc).__name__)
            await box.mark_retry(int(row["id"]), f"{type(exc).__name__}", _backoff(int(row["attempts"])))
            continue
        await box.mark_sent(int(row["id"]), remote_id)
        await asyncio.sleep(SEND_GAP_S)
    return len(rows)


_wake: asyncio.Event | None = None


def wake() -> None:
    """Have the drainer look now (something was queued)."""
    if _wake is not None:
        _wake.set()


async def drain_loop(stop: asyncio.Event) -> None:
    """Write the outbox until *stop*: right after something is queued, else
    every ~30 s; sent rows past the retention are pruned every 6 hours."""
    global _wake
    _wake = asyncio.Event()
    last_prune = 0.0
    try:
        await asyncio.wait_for(stop.wait(), timeout=5.0)
        return
    except TimeoutError:
        pass
    while not stop.is_set():
        looked = 0
        try:
            looked = await drain_once()
        except Exception as exc:  # noqa: BLE001
            log.warning("external memory: drain failed", error=type(exc).__name__)
        if time.time() - last_prune > PRUNE_EVERY_S:
            last_prune = time.time()
            try:
                await outbox().prune()
            except Exception:
                log.debug("external memory: prune failed", exc_info=True)
        if looked >= DRAIN_BATCH:
            continue
        _wake.clear()
        stopper = asyncio.ensure_future(stop.wait())
        waker = asyncio.ensure_future(_wake.wait())
        try:
            await asyncio.wait({stopper, waker}, timeout=DRAIN_IDLE_S, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (stopper, waker):
                if not t.done():
                    t.cancel()


# ── the outbox store ────────────────────────────────────────────────

_data_dir: Path | None = None
_outbox: Outbox | None = None


def configure(*, data_dir: Path) -> None:
    global _data_dir
    _data_dir = Path(data_dir)


def outbox() -> Outbox:
    global _outbox
    if _outbox is None:
        base = _data_dir
        if base is None:
            env = (os.environ.get("FD_DATA_DIR") or "").strip()
            if env:
                base = Path(env).expanduser().resolve()
            else:
                from captain_claw.flight_deck.server import DATA_DIR

                base = DATA_DIR
        _outbox = Outbox(base / "external_memory.db")
    return _outbox


def set_outbox(value: Outbox | None) -> None:
    """Tests: point the service at another outbox."""
    global _outbox
    _outbox = value


async def close() -> None:
    if _outbox is not None:
        await _outbox.close()
