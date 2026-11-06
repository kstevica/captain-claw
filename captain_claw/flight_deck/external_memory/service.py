"""External memory — each user's connection, and everything done with it.

The connection lives ONLY in Flight Deck, like Jev's key:

* Per user in ``user_settings`` — ``external_memory:connection`` (JSON
  ``{provider, credentials, info, verified, verify_error, verified_at,
  updated_at, checked_at}``; the credentials sealed with ``fd_crypto`` when it
  is usable) and ``external_memory:mirror`` (which kinds are mirrored).
  ``external_memory:`` is server-owned: ``/fd/settings`` never returns it and
  refuses to write it. An auth-disabled deck keeps one connection for the deck
  in ``system_settings``.
* There is no team connection on purpose: an external memory is a person's
  own (Membase: "each end user gets their own account and key"); one shared
  key would pour every user's facts into one memory.
* Every agent of the user shares it — that is the point. Agents never hold the
  credentials; they reach it through ``/fd/external-memory/agent/*``, the owner
  resolved from this deck's records, never from the agent's word.

Two ways things get in:

* The agent tool's actions (:func:`run_for_owner`) — ``remember``, ``search``
  and whatever else the provider offers.
* Mirroring (:func:`mirror_for_owner`): insights, intentions, reflections,
  intuitions (from the agents) and lessons (taught here) are queued in the
  outbox and written by :func:`drain_loop`, one at a time, retried with backoff.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
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
)
from captain_claw.flight_deck.external_memory.outbox import Outbox
from captain_claw.logging import get_logger

log = get_logger(__name__)

_K_CONN = "external_memory:connection"
_K_MIRROR = "external_memory:mirror"

NOT_CONNECTED = (
    "No external memory is connected — the person connects one in Flight Deck → Connections → "
    "External memory."
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


class NotConnected(LookupError):  # noqa: N818 — reads as the condition
    def __init__(self, message: str = NOT_CONNECTED) -> None:
        super().__init__(message)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ── storage ─────────────────────────────────────────────────────────


def _deck_scope() -> bool:
    """Auth off: one connection for the whole deck, in ``system_settings``."""
    return not _fd_auth_enabled()


def owner_key(owner: str) -> str:
    """The id the outbox and usage log file things under."""
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


async def load_connection(owner: str) -> dict[str, Any] | None:
    """The stored connection with its credentials opened, or None. A
    connection to a provider this deck no longer knows, or whose credentials
    can't be opened (the data key changed), reads as none."""
    raw = await _read(owner, _K_CONN)
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


async def _store_connection(owner: str, conn: dict[str, Any]) -> None:
    out = dict(conn)
    out["credentials"] = _seal(conn.get("credentials") or {})
    info = conn.get("info")
    out["info"] = info.to_dict() if isinstance(info, ConnectionInfo) else (info or {})
    await _write(owner, _K_CONN, json.dumps(out, ensure_ascii=True))


async def _update_connection(owner: str, **fields: Any) -> None:
    conn = await load_connection(owner)
    if not conn:
        return
    conn.update(fields)
    await _store_connection(owner, conn)


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


def provider_for(conn: dict[str, Any], **kwargs: Any) -> ExternalMemoryProvider:
    return providers.build(str(conn["provider"]), conn["credentials"], conn.get("info"), **kwargs)


# ── views ───────────────────────────────────────────────────────────


async def status_for_owner(owner: str) -> dict[str, Any]:
    """The connection as the dashboard shows it — never a credential."""
    conn = await load_connection(owner)
    mirror = await load_mirror(owner)
    base: dict[str, Any] = {
        "connected": False,
        "providers": providers.catalog(),
        "mirror": mirror,
        "mirror_kinds": list(MIRROR_KINDS),
        "scope": "deck" if _deck_scope() else "user",
        "auth_enabled": _fd_auth_enabled(),
    }
    if not conn:
        return base
    provider = provider_for(conn)
    try:
        counts = await outbox().counts(owner_key(owner))
    except Exception as exc:  # the dashboard still shows the connection
        log.debug("external memory: outbox counts failed", error=type(exc).__name__)
        counts = {}
    verify_error = conn.get("verify_error")
    return {
        **base,
        "connected": True,
        "provider": provider.name,
        "label": provider.label,
        "credential_hints": provider.credential_hints(),
        "verified": conn.get("verified") if isinstance(conn.get("verified"), bool) else None,
        "verify_error": verify_error if verify_error in ("refused", "unreachable") else None,
        "verified_at": conn.get("verified_at") or None,
        "updated_at": conn.get("updated_at") or None,
        "reaches": list(provider.info.reaches),
        "actions": [a.to_dict() for a in provider.actions()],
        "rules": [*provider.rules(), mirror_rule(mirror)],
        "outbox": counts,
    }


_refreshing: set[str] = set()


async def _refresh_info(owner: str) -> None:
    """Re-learn what the connection may do (background, at most one per owner)."""
    if owner in _refreshing:
        return
    _refreshing.add(owner)
    try:
        conn = await load_connection(owner)
        if not conn:
            return
        try:
            info = await provider_for(conn, max_retries=0).verify()
        except ProviderAuthError:
            await _update_connection(owner, verified=False, verify_error="refused", checked_at=time.time())
            return
        except ProviderError:
            await _update_connection(owner, checked_at=time.time())
            return
        await _update_connection(owner, info=info, verified=True, verify_error=None,
                                 verified_at=_now(), checked_at=time.time())
    except Exception as exc:
        log.debug("external memory: background re-check failed", error=type(exc).__name__)
    finally:
        _refreshing.discard(owner)


async def manifest_for_owner(owner: str) -> dict[str, Any]:
    """What an agent needs (no network): whether a memory is connected, its
    actions and rules, and which kinds to mirror."""
    conn = await load_connection(owner)
    if not conn:
        return {"connected": False}
    if time.time() - float(conn.get("checked_at") or 0) > INFO_MAX_AGE_S:
        asyncio.create_task(_refresh_info(owner))
    provider = provider_for(conn)
    mirror = await load_mirror(owner)
    manifest = provider.manifest()
    manifest["rules"] = [*manifest["rules"], mirror_rule(mirror)]
    return {
        "connected": True,
        **manifest,
        # The ones an AGENT sends (lessons are mirrored here, by Flight Deck).
        "mirror": [k for k in MIRROR_KINDS if mirror.get(k) and k != "lesson"],
        "refused": conn.get("verify_error") == "refused",
    }


# ── connecting ──────────────────────────────────────────────────────


async def connect(owner: str, provider_name: str, raw_credentials: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Check the credentials with the provider and save them: ``(status,
    warning)``. Credentials the provider refuses raise
    :class:`ProviderAuthError` (nothing saved); a provider that couldn't be
    asked leaves them saved unverified, with a warning."""
    cls = providers.provider_class(provider_name)
    if cls is None:
        raise ProviderInputError(f"Unknown external memory '{provider_name}'")
    credentials = cls.clean_credentials(raw_credentials)
    provider = providers.build(cls.name, credentials)
    now = _now()
    conn: dict[str, Any] = {
        "provider": cls.name, "credentials": credentials, "info": ConnectionInfo(),
        "verified": None, "verify_error": None, "verified_at": None, "updated_at": now, "checked_at": 0,
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
    previous = await load_connection(owner)
    if previous and previous.get("provider") != cls.name:
        # A different service: what was queued for the old one won't go there.
        await outbox().drop_owner(owner_key(owner))
    await _store_connection(owner, conn)
    await outbox().retry_owner(owner_key(owner))
    wake()
    return await status_for_owner(owner), warning


async def disconnect(owner: str) -> dict[str, Any]:
    await _erase(owner, _K_CONN)
    try:
        await outbox().drop_owner(owner_key(owner))
    except Exception as exc:
        log.debug("external memory: outbox drop failed", error=type(exc).__name__)
    return await status_for_owner(owner)


async def test(owner: str) -> dict[str, Any]:
    """A live check with the saved connection; updates what it may do."""
    conn = await load_connection(owner)
    if not conn:
        raise NotConnected()
    t0 = time.monotonic()
    try:
        info = await provider_for(conn).verify()
    except ProviderAuthError:
        await _update_connection(owner, verified=False, verify_error="refused", checked_at=time.time())
        raise
    await _update_connection(owner, info=info, verified=True, verify_error=None,
                             verified_at=_now(), checked_at=time.time())
    await outbox().retry_owner(owner_key(owner))
    wake()
    return {"ok": True, "latency_ms": int((time.monotonic() - t0) * 1000), **await status_for_owner(owner)}


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
    """Run one provider action with *owner*'s connection. Raises
    :class:`NotConnected`, :class:`ProviderInputError` (incl. an action the
    connection doesn't offer) or :class:`ProviderError`."""
    conn = await load_connection(owner)
    if not conn:
        raise NotConnected()
    provider = provider_for(conn)
    action = (action or "").strip().lower()
    provider.action(action)  # UnknownAction before anything is sent or logged
    t0 = time.monotonic()
    ok, status = False, 0
    try:
        result = await provider.run(action, args if isinstance(args, dict) else {})
        ok, status = True, 200
    except ProviderAuthError as exc:
        status = exc.status
        await _update_connection(owner, verified=False, verify_error="refused")
        raise
    except ProviderError as exc:
        status = exc.status
        raise
    finally:
        await _log_call(owner, source=source, agent=agent, provider=provider.name, action=action,
                        ok=ok, status=status, latency_ms=int((time.monotonic() - t0) * 1000))
    if action == "capabilities":
        await _update_connection(owner, info=provider.info, verified=True, verify_error=None,
                                 verified_at=_now(), checked_at=time.time())
    elif conn.get("verified") is not True:
        await _update_connection(owner, verified=True, verify_error=None, verified_at=_now())
    return result


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
    conn = await load_connection(owner)
    if not conn:
        return {"queued": False, "reason": "not_connected"}
    if not (await load_mirror(owner)).get(item.kind):
        return {"queued": False, "reason": "kind_off"}
    queued = await outbox().enqueue(
        owner_key(owner), str(conn["provider"]), item.kind, item.source_id, item.to_dict(),
        agent=item.agent, title=item.title or item.content[:120],
    )
    if queued:
        wake()
    return {"queued": queued, "reason": "" if queued else "already_queued"}


def _backoff(attempts: int) -> float:
    return min(30.0 * (2 ** max(0, attempts)), HOLD_S)


async def drain_once(limit: int = DRAIN_BATCH) -> int:
    """Write what is due; returns how many rows were looked at."""
    box = outbox()
    rows = await box.due(limit)
    conns: dict[str, dict[str, Any] | None] = {}
    held: set[str] = set()
    for row in rows:
        owner_id = str(row["owner_id"])
        if owner_id in held:
            continue
        if owner_id not in conns:
            conns[owner_id] = await load_connection(owner_id)
        conn = conns[owner_id]
        if not conn or conn.get("provider") != row["provider"]:
            await box.mark_failed(int(row["id"]), "No longer connected to this external memory.")
            continue
        try:
            item = MemoryItem.from_dict(json.loads(row["payload"]))
        except (TypeError, ValueError):
            await box.mark_failed(int(row["id"]), "Unreadable queued item.")
            continue
        try:
            remote_id = await provider_for(conn, max_retries=0).mirror(item)
        except ProviderAuthError as exc:
            held.add(owner_id)
            await box.defer_owner(owner_id, HOLD_S, str(exc))
            await _update_connection(owner_id, verified=False, verify_error="refused")
            continue
        except ProviderError as exc:
            if exc.code in ("rate_limited",) or (exc.status == 429 and exc.retryable):
                held.add(owner_id)
                await box.defer_owner(owner_id, RATE_LIMIT_HOLD_S, str(exc))
            elif exc.retryable:
                await box.mark_retry(int(row["id"]), str(exc), _backoff(int(row["attempts"])))
            elif exc.code in ("quota_exceeded", "write_target_required") or exc.status == 403:
                # Fixable by the person in the service itself: hold, don't fail.
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
