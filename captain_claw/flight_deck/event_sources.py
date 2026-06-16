"""Source adapters + the poll loop for the event spine (#2).

An *adapter* turns a real-world signal into ``external_events`` rows. The
``events_loop`` ticks on a timer, and for each user runs every due, enabled
adapter, ingesting whatever it returns (the store dedups via ``dedup_key``) and
advancing that source's cursor.

Adapters self-gate (``enabled()`` + ``requires_google``) so the loop is cheap
when nothing is turned on. A synthetic adapter (env ``CLAW_EVENTS_SYNTHETIC=1``)
emits a dummy event each tick to validate the loop without any external creds —
the real Calendar/Gmail adapters fetch FD-side once Google is connected.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from captain_claw.config import get_config
from captain_claw.flight_deck.events import get_store

_log = logging.getLogger(__name__)

# poll(user_id, cursor) -> (events_to_ingest, new_cursor)
PollFn = Callable[[str, str], Awaitable[tuple[list[dict[str, Any]], str]]]


@dataclass
class Adapter:
    name: str                       # also the event `source`
    interval_seconds: float
    poll: PollFn
    enabled: Callable[[], bool]     # source-level on/off (config/env)
    requires_google: bool = False


_ADAPTERS: list[Adapter] = []


def register(adapter: Adapter) -> None:
    _ADAPTERS.append(adapter)


def _google_connected() -> bool:
    try:
        from captain_claw.google_oauth_manager import is_google_connected_cached
        return is_google_connected_cached()
    except Exception:
        return False


async def poll_user(user_id: str) -> int:
    """Run every due, enabled adapter for one user; ingest results. Returns the
    number of new events ingested."""
    store = get_store()
    now = time.monotonic()
    ingested = 0
    for ad in _ADAPTERS:
        try:
            if not ad.enabled():
                continue
            if ad.requires_google and not _google_connected():
                continue
            st = store.get_poll_state(user_id, ad.name)
            if now - float(st.get("last_poll_at") or 0.0) < ad.interval_seconds:
                continue
            events, new_cursor = await ad.poll(user_id, st.get("cursor") or "")
            for ev in events or []:
                row = store.add_event(
                    user_id,
                    source=ev.get("source") or ad.name,
                    event_type=ev.get("event_type", ""),
                    summary=ev.get("summary", ""),
                    body=ev.get("body", ""),
                    metadata=ev.get("metadata") if isinstance(ev.get("metadata"), dict) else {},
                    dedup_key=ev.get("dedup_key", ""),
                )
                if row is not None:
                    ingested += 1
            store.set_poll_state(user_id, ad.name, last_poll_at=now, cursor=new_cursor)
        except Exception as exc:
            _log.warning("event adapter %s failed for %s: %s", ad.name, user_id, exc)
    return ingested


async def events_loop(stop_event: asyncio.Event) -> None:
    """Background poll loop. Ticks every ``events.poll_seconds``; per user runs due
    adapters. New events nudge the arbiter via a debounced force-pulse."""
    _log.info("events loop started")
    while not stop_event.is_set():
        try:
            cfg = get_config().events
            synthetic = os.environ.get("CLAW_EVENTS_SYNTHETIC", "").lower() in ("1", "true", "yes")
            if cfg.poll_enabled or synthetic:
                from captain_claw.flight_deck.consciousness import distinct_owners_with_agents
                users = distinct_owners_with_agents() or (["local"] if synthetic else [])
                for uid in users:
                    if stop_event.is_set():
                        break
                    n = await poll_user(uid)
                    if n > 0:
                        _log.info("events: ingested %d for %s", n, uid)
                        _nudge(uid)
        except Exception as exc:
            _log.warning("events loop iteration error: %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=max(15, get_config().events.poll_seconds))
        except asyncio.TimeoutError:
            pass
    _log.info("events loop stopped")


_pulse_tasks: set = set()


def _nudge(user_id: str) -> None:
    """Force a pulse so freshly-ingested events reach the arbiter promptly."""
    try:
        from captain_claw.flight_deck.autonomy import resolve_config
        if not resolve_config(user_id).get("enabled"):
            return
        from captain_claw.flight_deck.consciousness import pulse
        t = asyncio.create_task(pulse(user_id, force=True))
        _pulse_tasks.add(t)
        t.add_done_callback(_pulse_tasks.discard)
    except Exception:
        pass


# ── Adapters ─────────────────────────────────────────────────────────

_synth_counter: dict[str, int] = {}


async def _synthetic_poll(user_id: str, cursor: str) -> tuple[list[dict[str, Any]], str]:
    n = _synth_counter.get(user_id, 0) + 1
    _synth_counter[user_id] = n
    return ([{
        "source": "synthetic", "event_type": "tick",
        "summary": f"Synthetic event #{n} (loop heartbeat)",
        "dedup_key": f"synth-{user_id}-{n}",
    }], str(n))


register(Adapter(
    name="synthetic",
    interval_seconds=float(os.environ.get("CLAW_EVENTS_SYNTHETIC_INTERVAL", "60") or 60),
    poll=_synthetic_poll,
    enabled=lambda: os.environ.get("CLAW_EVENTS_SYNTHETIC", "").lower() in ("1", "true", "yes"),
))

# Real adapters (Calendar, Gmail) are registered in event_sources_google.py once
# the FD-side Google fetch is wired — kept out of this module so the framework
# stays validate-able without creds.
try:
    from captain_claw.flight_deck import event_sources_google  # noqa: F401
except Exception:  # not yet present / import guarded
    pass
