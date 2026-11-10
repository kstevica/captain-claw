"""External memory — each user's connection, and everything done with it.

The connection lives ONLY in Flight Deck, like Jev's key:

* Per user in ``user_settings`` — ``external_memory:connection`` (JSON
  ``{provider, credentials, info, verified, verify_error, verified_at,
  updated_at, checked_at, private_ok}``; the credentials sealed with
  ``fd_crypto`` when it is usable) and ``external_memory:mirror`` (which kinds
  are mirrored). ``external_memory:`` is server-owned: ``/fd/settings`` never
  returns it and refuses to write it. An auth-disabled deck keeps one
  connection for the deck in ``system_settings``.
* The deck's connection: an admin may connect one *scoped* provider
  (Hindsight: one server, a bank per person; Qdrant / ArcadeDB: one
  collection / database, a space per person) for everyone —
  ``external_memory:deck_connection`` in ``system_settings``. A user with no
  connection of their own uses it, in a scope of their own
  (:func:`base.scope_id`); their own connection, once saved, wins. A *key*
  provider (Membase: one account per person) can't be the deck's — one key
  would pour every user's facts into one memory.
* Every agent of the user shares it — that is the point. Agents never hold the
  credentials; they reach it through ``/fd/external-memory/agent/*``, the owner
  resolved from this deck's records, never from the agent's word.
* A user who isn't an admin can't point their own connection at this machine
  or its network (:func:`base.private_address`) — Flight Deck would make those
  requests for them. The deck's connection, set by an admin, can.

Two ways things get in:

* The agent tool's actions (:func:`run_for_owner`) — ``remember``, ``search``
  and whatever else the provider offers.
* Mirroring (:func:`mirror_for_owner`): insights, intentions, reflections,
  intuitions (from the agents) and lessons (taught here) are queued in the
  outbox and written by :func:`drain_loop`, one at a time, retried with backoff.
  While a memory is connected, agents pick facts out of their chats every
  :data:`CAPTURE_EVERY` messages (Captain's own pace is 20), so insights — and
  with them this memory — fill as the person talks.

And one way out besides the tool: automatic recall (:func:`recall_for_owner`).
Before a reply the agent sends the person's message; the provider returns
only what bears on it, and the agent puts that in front of the model. How much
the person's agents lean on the memory is their *reading priority*
(``external_memory:priority``, :data:`PRIORITIES`): ``first`` — their main
memory (more items, kept when the prompt is crowded, trusted over the agent's
own notes); ``alongside`` (default) — next to Captain's own memory;
``on_demand`` — nothing recalled on its own, the agent searches when it
decides to.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from captain_claw.flight_deck.auth import _LOCAL_USER, _fd_auth_enabled, get_db
from captain_claw.flight_deck.external_memory import providers
from captain_claw.flight_deck.external_memory.base import (
    DEFAULT_MIRROR,
    MIRROR_KINDS,
    ActionResult,
    ConnectionInfo,
    ExternalMemoryProvider,
    MemoryItem,
    ProviderAuthError,
    ProviderError,
    ProviderInputError,
    private_address,
    scope_id,
)
from captain_claw.flight_deck.external_memory.outbox import Outbox
from captain_claw.logging import get_logger

log = get_logger(__name__)

_K_CONN = "external_memory:connection"
_K_MIRROR = "external_memory:mirror"
_K_DECK = "external_memory:deck_connection"
_K_PRIORITY = "external_memory:priority"

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
class Resolved:
    """The connection a user's memory goes through: their own, or the deck's
    (``source``), and their scope on it ("" for a key provider)."""

    conn: dict[str, Any]
    source: str
    scope: str

    @property
    def provider_name(self) -> str:
        return str(self.conn["provider"])


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ── storage ─────────────────────────────────────────────────────────


def _deck_scope() -> bool:
    """Auth off: one connection for the whole deck, in ``system_settings``."""
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


def _parse(raw: str) -> dict[str, Any] | None:
    """A stored connection with its credentials opened, or None. One to a
    provider this deck no longer knows, or whose credentials can't be opened
    (the data key changed), reads as none."""
    if not raw:
        return None
    try:
        conn = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(conn, dict) or not providers.provider_class(str(conn.get("provider") or "")):
        return None
    creds = _unseal(conn.get("credentials"))
    if not creds:
        return None
    conn["credentials"] = creds
    conn["info"] = ConnectionInfo.from_dict(conn.get("info"))
    return conn


def _dump(conn: dict[str, Any]) -> str:
    out = dict(conn)
    out["credentials"] = _seal(conn.get("credentials") or {})
    info = conn.get("info")
    out["info"] = info.to_dict() if isinstance(info, ConnectionInfo) else (info or {})
    return json.dumps(out, ensure_ascii=True)


async def load_connection(owner: str) -> dict[str, Any] | None:
    """*owner*'s OWN connection (the deck's one with sign-in off), or None."""
    return _parse(await _read(owner, _K_CONN))


async def load_deck_connection() -> dict[str, Any] | None:
    """The admin's connection for everyone, or None (always None with sign-in
    off — that deck's one connection is already everyone's)."""
    if _deck_scope():
        return None
    return _parse((await get_db().get_system_setting(_K_DECK)) or "")


async def _store(owner: str, source: str, conn: dict[str, Any]) -> None:
    if source == "deck":
        await get_db().set_system_setting(_K_DECK, _dump(conn))
    else:
        await _write(owner, _K_CONN, _dump(conn))


async def _update(owner: str, source: str, **fields: Any) -> None:
    conn = await (load_deck_connection() if source == "deck" else load_connection(owner))
    if not conn:
        return
    conn.update(fields)
    await _store(owner, source, conn)


async def resolve(owner: str) -> Resolved | None:
    """The connection *owner*'s memory goes through: their own, else the
    deck's (scoped providers only), else None."""
    own = await load_connection(owner)
    if own:
        cls = providers.provider_class(str(own["provider"]))
        scope = ""
        if cls is not None and cls.tenancy == "scoped":
            named = str(own["credentials"].get(cls.scope_field) or "") if cls.scope_field else ""
            scope = named or scope_id(owner_key(owner))
        return Resolved(own, "own", scope)
    deck = await load_deck_connection()
    if deck:
        return Resolved(deck, "deck", scope_id(owner_key(owner)))
    return None


async def load_mirror(owner: str) -> dict[str, bool]:
    raw = await _read(owner, _K_MIRROR)
    out = dict(DEFAULT_MIRROR)
    if raw:
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            data = {}
        if isinstance(data, dict):
            for kind in MIRROR_KINDS:
                if isinstance(data.get(kind), bool):
                    out[kind] = data[kind]
    return out


async def save_mirror(owner: str, kinds: dict[str, Any]) -> dict[str, bool]:
    current = await load_mirror(owner)
    for kind in MIRROR_KINDS:
        if isinstance(kinds.get(kind), bool):
            current[kind] = kinds[kind]
    await _write(owner, _K_MIRROR, json.dumps(current))
    return current


async def load_priority(owner: str) -> str:
    raw = (await _read(owner, _K_PRIORITY) or "").strip()
    return raw if raw in PRIORITIES else DEFAULT_PRIORITY


async def save_priority(owner: str, value: Any) -> str:
    priority = str(value or "").strip().lower()
    if priority not in PRIORITIES:
        raise ProviderInputError(f"Reading priority is one of: {', '.join(PRIORITIES)}")
    await _write(owner, _K_PRIORITY, priority)
    return priority


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
    """The rule telling an agent how much to lean on the memory."""
    return _PRIORITY_RULES.get(priority, _PRIORITY_RULES[DEFAULT_PRIORITY])


_KIND_PLURAL = {
    "insight": "insights", "intention": "intentions (once active)", "reflection": "reflections",
    "lesson": "lessons the person taught", "intuition": "intuitions",
}


def mirror_rule(mirror: dict[str, bool]) -> str:
    """The rule telling an agent what is copied in on its own."""
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


async def provider_for(res: Resolved, **kwargs: Any) -> ExternalMemoryProvider:
    """The provider for one call. A user's own connection that may not reach
    private addresses is checked again here (names can change what they
    point at)."""
    cls = providers.provider_class(res.provider_name)
    if res.source == "own" and cls is not None and not res.conn.get("private_ok"):
        for name in cls.address_fields():
            url = str(res.conn["credentials"].get(name) or "")
            if url and await _is_private(url):
                raise ProviderError(PRIVATE_REFUSED, status=403, code="private_address")
    return providers.build(res.provider_name, res.conn["credentials"], res.conn.get("info"),
                           scope=res.scope, **kwargs)


def _plain_provider(res: Resolved) -> ExternalMemoryProvider:
    """For views only (no network): actions, rules, hints."""
    return providers.build(res.provider_name, res.conn["credentials"], res.conn.get("info"), scope=res.scope)


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
            "scope_label": cls.scope_label if cls else ""}


async def status_for_owner(owner: str) -> dict[str, Any]:
    """The connection as the dashboard shows it — never a credential."""
    res = await resolve(owner)
    mirror = await load_mirror(owner)
    priority = await load_priority(owner)
    base: dict[str, Any] = {
        "connected": False,
        "providers": providers.catalog(),
        "mirror": mirror,
        "mirror_kinds": list(MIRROR_KINDS),
        "priority": priority,
        "priorities": list(PRIORITIES),
        "capture_every": CAPTURE_EVERY,
        "scope": "deck" if _deck_scope() else "user",
        "auth_enabled": _fd_auth_enabled(),
        "deck_connection": await _deck_offer(),
    }
    if not res:
        return base
    provider = _plain_provider(res)
    try:
        counts = await outbox().counts(owner_key(owner))
    except Exception as exc:  # the dashboard still shows the connection
        log.debug("external memory: outbox counts failed", error=type(exc).__name__)
        counts = {}
    view = _conn_view(res.conn, provider)
    if res.source == "deck":
        view["credential_hints"] = {}  # the admin's, not this user's to see
    return {
        **base,
        **view,
        "connected": True,
        "source": res.source,
        "tenancy": provider.tenancy,
        "scope_label": provider.scope_label,
        "scope_id": res.scope,
        "reaches": provider.reaches(),
        "actions": [a.to_dict() for a in provider.actions()],
        "rules": [*provider.rules(), priority_rule(priority), mirror_rule(mirror)],
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


async def _refresh_info(owner: str) -> None:
    """Re-learn what the connection may do (background, at most one per
    connection)."""
    res = await resolve(owner)
    if not res:
        return
    key = "deck" if res.source == "deck" else owner
    if key in _refreshing:
        return
    _refreshing.add(key)
    try:
        try:
            info = await (await provider_for(res, max_retries=0)).verify()
        except ProviderAuthError:
            await _update(owner, res.source, verified=False, verify_error="refused", checked_at=time.time())
            return
        except ProviderError:
            await _update(owner, res.source, checked_at=time.time())
            return
        await _update(owner, res.source, info=info, verified=True, verify_error=None,
                      verified_at=_now(), checked_at=time.time())
    except Exception as exc:
        log.debug("external memory: background re-check failed", error=type(exc).__name__)
    finally:
        _refreshing.discard(key)


async def manifest_for_owner(owner: str) -> dict[str, Any]:
    """What an agent needs (no network): whether a memory is connected, its
    actions and rules, and which kinds to mirror."""
    res = await resolve(owner)
    if not res:
        return {"connected": False}
    if time.time() - float(res.conn.get("checked_at") or 0) > INFO_MAX_AGE_S:
        asyncio.create_task(_refresh_info(owner))
    provider = _plain_provider(res)
    mirror = await load_mirror(owner)
    priority = await load_priority(owner)
    manifest = provider.manifest()
    manifest["rules"] = [*manifest["rules"], priority_rule(priority), mirror_rule(mirror)]
    limit, wait = RECALL.get(priority, (0, 0.0))
    return {
        "connected": True,
        **manifest,
        # The ones an AGENT sends (lessons are mirrored here, by Flight Deck).
        "mirror": [k for k in MIRROR_KINDS if mirror.get(k) and k != "lesson"],
        "refused": res.conn.get("verify_error") == "refused",
        # Reading priority: whether the agent asks for recall before a reply,
        # how long it waits, and where it puts what comes back.
        "priority": priority,
        "recall_limit": limit,
        "recall_timeout": wait,
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
                  *, private_ok: bool = False) -> tuple[dict[str, Any], str]:
    """Check *owner*'s own credentials with the provider and save them:
    ``(status, warning)``. *private_ok*: may point at this machine or its
    network (admins, and a deck with sign-in off)."""
    conn, warning, _provider = await _checked_conn(provider_name, raw_credentials, private_ok=private_ok)
    previous = await resolve(owner)
    if previous and previous.provider_name != conn["provider"]:
        # A different service: what was queued for the old one won't go there.
        await outbox().drop_owner(owner_key(owner))
    await _store(owner, "own", conn)
    await outbox().retry_owner(owner_key(owner))
    wake()
    return await status_for_owner(owner), warning


async def disconnect(owner: str) -> dict[str, Any]:
    """Forget *owner*'s own connection (and what was still queued for it).
    With a deck connection, they fall back to it."""
    await _erase(owner, _K_CONN)
    try:
        await outbox().drop_owner(owner_key(owner))
    except Exception as exc:
        log.debug("external memory: outbox drop failed", error=type(exc).__name__)
    return await status_for_owner(owner)


async def deck_connect(provider_name: str, raw_credentials: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """The admin connects one scoped provider for everyone."""
    if _deck_scope():
        raise ProviderInputError("Sign-in is off on this deck, so its one connection is already everyone's.")
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
    await _store("", "deck", conn)
    wake()
    return await deck_status(), warning


async def deck_disconnect() -> dict[str, Any]:
    if not _deck_scope():
        await get_db().set_system_setting(_K_DECK, "")
    return await deck_status()


async def _verify(owner: str, res: Resolved) -> dict[str, Any]:
    t0 = time.monotonic()
    try:
        info = await (await provider_for(res)).verify()
    except ProviderAuthError:
        await _update(owner, res.source, verified=False, verify_error="refused", checked_at=time.time())
        raise
    await _update(owner, res.source, info=info, verified=True, verify_error=None,
                  verified_at=_now(), checked_at=time.time())
    return {"ok": True, "latency_ms": int((time.monotonic() - t0) * 1000)}


async def test(owner: str) -> dict[str, Any]:
    """A live check with the connection *owner* uses; updates what it may do."""
    res = await resolve(owner)
    if not res:
        raise NotConnected()
    out = await _verify(owner, res)
    await outbox().retry_owner(owner_key(owner))
    wake()
    return {**out, **await status_for_owner(owner)}


async def deck_test() -> dict[str, Any]:
    deck = await load_deck_connection()
    if not deck:
        raise NotConnected("No memory is connected for the deck.")
    out = await _verify("", Resolved(deck, "deck", scope_id("check")))
    return {**out, **await deck_status()}


# ── actions ─────────────────────────────────────────────────────────


async def _log_call(owner: str, *, source: str, agent: str, provider: str, action: str,
                    ok: bool, status: int, latency_ms: int) -> None:
    """One ``usage_logs`` row per action. Never fails the call."""
    try:
        await get_db().log_usage(owner_key(owner), "external_memory_call", json.dumps({
            "source": source, "agent": agent or "", "provider": provider, "action": action,
            "ok": ok, "status": status, "latency_ms": latency_ms,
        }))
    except Exception as exc:
        log.warning("external memory usage log failed: %s", type(exc).__name__)


async def run_for_owner(owner: str, action: str, args: dict[str, Any], *, source: str,
                        agent: str = "") -> ActionResult:
    """Run one provider action with the connection *owner* uses, in their
    scope. Raises :class:`NotConnected`, :class:`ProviderInputError` (incl. an
    action the connection doesn't offer) or :class:`ProviderError`."""
    res = await resolve(owner)
    if not res:
        raise NotConnected()
    action = (action or "").strip().lower()
    _plain_provider(res).action(action)  # UnknownAction before anything is sent or logged
    provider = await provider_for(res)
    t0 = time.monotonic()
    ok, status = False, 0
    try:
        result = await provider.run(action, args if isinstance(args, dict) else {})
        ok, status = True, 200
    except ProviderAuthError as exc:
        status = exc.status
        await _update(owner, res.source, verified=False, verify_error="refused")
        raise
    except ProviderError as exc:
        status = exc.status
        raise
    finally:
        await _log_call(owner, source=source, agent=agent, provider=provider.name, action=action,
                        ok=ok, status=status, latency_ms=int((time.monotonic() - t0) * 1000))
    if action == "capabilities" and provider.info.allowed:
        await _update(owner, res.source, info=provider.info, verified=True, verify_error=None,
                      verified_at=_now(), checked_at=time.time())
    elif res.conn.get("verified") is not True:
        await _update(owner, res.source, verified=True, verify_error=None, verified_at=_now())
    return result


# ── automatic recall ────────────────────────────────────────────────


async def recall_for_owner(owner: str, query: str, *, agent: str = "") -> dict[str, Any]:
    """What *owner*'s memory holds on *query*, for an agent about to reply:
    ``{connected, priority, label, hits: [{text, id, when, kind}], error?}``.
    Never raises — a slow or failing memory must not hold up or break the
    turn: it comes back with no hits and the reason. Nothing is asked when the
    person's reading priority is ``on_demand``."""
    res = await resolve(owner)
    if not res:
        return {"connected": False, "hits": []}
    priority = await load_priority(owner)
    cls = providers.provider_class(res.provider_name)
    out: dict[str, Any] = {"connected": True, "priority": priority, "provider": res.provider_name,
                           "label": cls.label if cls else res.provider_name, "hits": []}
    query = str(query or "").strip()[:4000]
    if priority not in RECALL or not query:
        return out
    limit, wait = RECALL[priority]
    t0 = time.monotonic()
    ok, status = False, 0
    try:
        provider = await provider_for(res, max_retries=0)
        hits = await asyncio.wait_for(provider.recall(query, limit), timeout=wait)
        out["hits"] = [h.to_dict() for h in hits[:limit] if h.text.strip()]
        ok, status = True, 200
    except TimeoutError:
        out["error"], status = "the memory didn't answer in time", 504
    except ProviderAuthError as exc:
        status = exc.status
        out["error"] = "the memory refused the key"
        await _update(owner, res.source, verified=False, verify_error="refused")
    except (ProviderError, ProviderInputError) as exc:
        status = getattr(exc, "status", 0) or 0
        out["error"] = str(exc)[:200]
    except Exception as exc:  # never the agent's problem
        log.warning("external memory: recall failed", error=type(exc).__name__)
        out["error"] = type(exc).__name__
    finally:
        await _log_call(owner, source="recall", agent=agent, provider=res.provider_name, action="recall",
                        ok=ok, status=status, latency_ms=int((time.monotonic() - t0) * 1000))
    return out


# ── mirroring ───────────────────────────────────────────────────────


async def mirror_for_owner(owner: str, item: MemoryItem) -> dict[str, Any]:
    """Queue one item for *owner*'s external memory: ``{queued, reason}``.
    Never raises for "not wanted" — no connection, that kind turned off, or an
    item already queued are ordinary answers."""
    if item.kind not in MIRROR_KINDS:
        raise ProviderInputError(f"Unknown kind '{item.kind}' — one of {', '.join(MIRROR_KINDS)}")
    item.content = (item.content or "").strip()
    item.source_id = (item.source_id or "").strip()[:200]
    if not item.content or not item.source_id:
        raise ProviderInputError("A mirrored item needs content and a source_id")
    item.content = item.content[:MAX_MIRROR_CONTENT]
    res = await resolve(owner)
    if not res:
        return {"queued": False, "reason": "not_connected"}
    if not (await load_mirror(owner)).get(item.kind):
        return {"queued": False, "reason": "kind_off"}
    queued = await outbox().enqueue(
        owner_key(owner), res.provider_name, item.kind, item.source_id, item.to_dict(),
        agent=item.agent, title=item.title or item.content[:120],
    )
    if queued:
        wake()
    return {"queued": queued, "reason": "" if queued else "already_queued"}


def _backoff(attempts: int) -> float:
    return min(30.0 * (2 ** max(0, attempts)), HOLD_S)


async def drain_once(limit: int = DRAIN_BATCH) -> int:
    """Write what is due; returns how many rows were looked at. Each row goes
    to the connection its owner uses NOW, in their scope."""
    box = outbox()
    rows = await box.due(limit)
    resolved: dict[str, Resolved | None] = {}
    held: set[str] = set()
    for row in rows:
        owner_id = str(row["owner_id"])
        if owner_id in held:
            continue
        if owner_id not in resolved:
            resolved[owner_id] = await resolve(owner_id)
        res = resolved[owner_id]
        if not res or res.provider_name != row["provider"]:
            await box.mark_failed(int(row["id"]), "No longer connected to this external memory.")
            continue
        try:
            item = MemoryItem.from_dict(json.loads(row["payload"]))
        except (TypeError, ValueError):
            await box.mark_failed(int(row["id"]), "Unreadable queued item.")
            continue
        try:
            remote_id = await (await provider_for(res, max_retries=0)).mirror(item)
        except ProviderAuthError as exc:
            held.add(owner_id)
            await box.defer_owner(owner_id, HOLD_S, str(exc))
            await _update(owner_id, res.source, verified=False, verify_error="refused")
            continue
        except ProviderError as exc:
            if exc.code in ("rate_limited",) or (exc.status == 429 and exc.retryable):
                held.add(owner_id)
                await box.defer_owner(owner_id, RATE_LIMIT_HOLD_S, str(exc))
            elif exc.retryable:
                await box.mark_retry(int(row["id"]), str(exc), _backoff(int(row["attempts"])))
            elif exc.code in ("quota_exceeded", "write_target_required", "private_address") or exc.status == 403:
                # Fixable by the person (or the admin): hold, don't fail.
                held.add(owner_id)
                await box.defer_owner(owner_id, HOLD_S, str(exc))
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
