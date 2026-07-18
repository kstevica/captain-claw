"""The Iskra heartbeat — the FD service that keeps beings alive (plan §5).

Mirrors the canonical consciousness.heartbeat_loop shape: an asyncio task
started from the FD lifespan, an interruptible stop-event sleep, an env
disable flag (``FD_BEINGS_DISABLED``), and cheap skips — a poll that finds no
due being costs nothing.

Per due being each pass: quiet hours decide wake-vs-dream (dreams happen at
night, once per night); everything else — allowance, torpor physics, thinking,
metering, digesting — lives in being_life.tick(). One being's failure never
touches another's.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

from captain_claw.flight_deck import being_instinct
from captain_claw.flight_deck import being_life
from captain_claw.flight_deck import being_world
from captain_claw.flight_deck.beings import get_store
from captain_claw.logging import get_logger

log = get_logger(__name__)

POLL_SECONDS = float(os.environ.get("BEINGS_POLL_SECONDS", "60"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _quiet_window(owner_id: str) -> tuple[int, int]:
    try:
        from captain_claw.flight_deck.autonomy import resolve_config
        cfg = resolve_config(owner_id)
        return int(cfg.get("quiet_hours_start", 22)), int(cfg.get("quiet_hours_end", 8))
    except Exception:  # noqa: BLE001
        return 22, 8


def _in_quiet(owner_id: str, now: datetime) -> tuple[bool, int]:
    start, end = _quiet_window(owner_id)
    h = now.hour
    if start == end:
        return False, end
    if start < end:
        return (start <= h < end), end
    return (h >= start or h < end), end


def _dreamed_tonight(store, being: dict, now: datetime) -> bool:
    for e in store.events(being["owner_id"], being["slug"], limit=20):
        if e["kind"] == "tick" and e["data"].get("kind") == "dream":
            try:
                at = datetime.fromisoformat(e["at"])
            except ValueError:
                return False
            return (now - at) < timedelta(hours=10)
    return False


async def _pass(db, now: datetime | None = None) -> int:
    """One poll pass; returns how many beings ticked. Pure enough to test."""
    store = get_store()
    now = now or _utcnow()
    ticked = 0
    for being in store.due_beings(now=now):
        try:
            quiet, quiet_end = _in_quiet(being["owner_id"], now)
            if quiet:
                if being["state"] == "alive" and not _dreamed_tonight(store, being, now):
                    await being_life.tick(db, store, being, kind="dream", now=now)
                    ticked += 1
                else:
                    wake = now.replace(minute=5, second=0, microsecond=0)
                    wake = wake.replace(hour=quiet_end)
                    if wake <= now:
                        wake += timedelta(days=1)
                    store.tick_bookkeeping(
                        being["id"], drives=being.get("drives") or {},
                        next_wake_at=wake, now=now)
                continue
            await being_life.tick(db, store, being, kind="wake", now=now)
            ticked += 1
        except Exception as e:  # noqa: BLE001 — one being never sinks another
            log.warning("being tick pass failed", slug=being.get("slug"),
                        error=str(e))
            try:
                store.tick_bookkeeping(
                    being["id"], drives=being.get("drives") or {},
                    next_wake_at=now + timedelta(hours=1), now=now)
            except Exception:  # noqa: BLE001
                pass
    return ticked


async def _instinct_pass(db, now: datetime | None = None) -> int:
    """The body brain's pass (docs/being-body-brain-plan.md): for every
    alive being with instincts on — first the reflexes (settle finished
    walks, feel co-presence, turn a fevered body home, fulfill plan
    steps: pure Python, $0), then at most ONE tiny decision call if a
    trigger says the ground moved (being_instinct.decide — hard-capped
    context, metered to the allowance). Quiet hours mean the feet sleep
    too; one being's failure never touches another's."""
    store = get_store()
    now = now or _utcnow()
    acted = 0
    try:
        beings = store.instinct_beings()
    except Exception as e:  # noqa: BLE001
        log.warning("instinct pass listing failed", error=str(e))
        return 0
    for being in beings:
        try:
            quiet, _ = _in_quiet(being["owner_id"], now)
            if quiet:
                continue
            acted += being_world.reflex_pass(store, being, now)
            fresh = store.get(being["owner_id"], being["slug"])
            if await being_instinct.decide(db, store, fresh, now=now):
                acted += 1
        except Exception as e:  # noqa: BLE001
            log.warning("instinct pass failed", slug=being.get("slug"),
                        error=str(e))
    return acted


async def beings_loop(db, stop_event: asyncio.Event) -> None:
    log.info("beings loop started", poll_seconds=POLL_SECONDS)
    while not stop_event.is_set():
        try:
            n = await _pass(db)
            if n:
                log.info("beings loop pass", ticked=n)
        except Exception as e:  # noqa: BLE001
            log.warning("beings loop pass error", error=str(e))
        # The body brain rides the same poll (body-brain plan): reflexes
        # are milliseconds of Python; a decision call fires only when a
        # trigger says the ground moved, hard-capped and metered.
        try:
            a = await _instinct_pass(db)
            if a:
                log.info("instinct pass", acted=a)
        except Exception as e:  # noqa: BLE001
            log.warning("instinct pass error", error=str(e))
        # Federation (§9.1): keep an outbound WS link alive for each visiting
        # being (NAT-friendly), and drop visitors of ours that stopped checking
        # in. reconcile() only starts/stops maintainers — the links themselves
        # persist and reconnect on their own.
        try:
            from captain_claw.flight_deck import being_federation
            await being_federation.village_client.reconcile(get_store())
            get_store().expire_visitors(
                ttl_minutes=being_life.VISITOR_TTL_MINUTES)
        except Exception as e:  # noqa: BLE001
            log.warning("federation reconcile error", error=str(e))
        # Visiting beings walk this village (§1): settle arrivals, then let an
        # idle guest stroll between civic places — host-side, $0, no LLM.
        try:
            st = get_store()
            now2 = _utcnow()
            for oid in st.owners_with_live_visitors(now=now2):
                being_world.settle_visitors(st, oid, now=now2)
                being_world.wander_visitors(st, oid, now=now2)
        except Exception as e:  # noqa: BLE001
            log.warning("visitor wander error", error=str(e))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=POLL_SECONDS)
        except asyncio.TimeoutError:
            continue
    try:
        from captain_claw.flight_deck import being_federation
        await being_federation.village_client.stop_all()
    except Exception:  # noqa: BLE001
        pass
    log.info("beings loop stopped")
