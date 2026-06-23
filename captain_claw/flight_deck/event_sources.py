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
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from string import Formatter
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
    enabled: Callable[[str], bool]  # per-user on/off (env / per-user config)
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
    # Wall-clock, not time.monotonic(): last_poll_at is persisted across process
    # (and machine) restarts. A monotonic value stored before a reboot would read
    # as ~90 days in the future after the clock resets to ~0, so `now - last` goes
    # hugely negative and polling silently stalls until monotonic catches up.
    now = time.time()
    ingested = 0
    for ad in _ADAPTERS:
        try:
            if not ad.enabled(user_id):
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
    ingested += await _poll_custom_sources(user_id, now, store)
    return ingested


# ── Generic tool-poller sources (Theme A) ───────────────────────────────


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:  # so a template can reference a missing field
        return ""


def _extract_rows(content: str, items_path: str) -> list[dict[str, Any]] | None:
    """Parse a tool's text output into row dicts. Returns None when it isn't
    structured JSON (caller then treats the whole output as one digest event)."""
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return None
    if items_path:
        for part in items_path.split("."):
            data = data.get(part) if isinstance(data, dict) else None
            if data is None:
                break
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for v in data.values():  # auto: first list-of-objects field
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return [data]
    return None


def _row_summary(label: str, row: dict[str, Any], template: str) -> str:
    if template:
        try:
            return Formatter().vformat(template, (), _SafeDict(row))[:300]
        except Exception:
            pass
    bits = ", ".join(f"{k}={row[k]}" for k in list(row)[:4] if not str(k).startswith("_"))
    return f"{label}: {bits}"[:300]


async def _poll_custom_sources(user_id: str, now: float, store: Any) -> int:
    """Poll each enabled user CustomSource by calling its read tool on the agent
    and mapping rows → events. Stamps the fetch contract (_fetch_tool/_handle_id)
    onto metadata so dispatch can ground the agent on any source."""
    try:
        from captain_claw.flight_deck.autonomy import resolve_config
        srcs = resolve_config(user_id).get("custom_sources") or []
    except Exception:
        return 0
    if not srcs:
        return 0
    from captain_claw.flight_deck.actions import run_tool_on_agent
    from captain_claw.flight_deck.fd_dispatch import _strongest_agent
    agent = None
    ingested = 0
    for src in srcs:
        try:
            if not src.get("enabled"):
                continue
            name, tool = str(src.get("name") or "").strip(), str(src.get("tool") or "").strip()
            if not name or not tool:
                continue
            if src.get("requires_google") and not _google_connected():
                continue
            st = store.get_poll_state(user_id, name)
            if now - float(st.get("last_poll_at") or 0.0) < float(src.get("interval_seconds") or 600):
                continue
            if agent is None:
                agent = _strongest_agent(user_id)
            if not agent:
                break
            res = await run_tool_on_agent(agent, tool, dict(src.get("args") or {}))
            store.set_poll_state(user_id, name, last_poll_at=now)
            if not res.get("ok"):
                _log.debug("custom source %s tool not ok: %s", name, res.get("error"))
                continue
            content = str(res.get("content") or "").strip()
            label = src.get("label") or name
            fetch_tool = str(src.get("fetch_tool") or "")
            id_field = str(src.get("id_field") or "id")
            template = str(src.get("summary_template") or "")
            rows = _extract_rows(content, str(src.get("items_path") or ""))
            if rows is None:  # unstructured → one digest event, deduped by content
                if content and store.add_event(
                    user_id, source=name, event_type="custom",
                    summary=f"{label}: {content}"[:300], body=content[:8000],
                    metadata={"_fetch_tool": fetch_tool},
                    dedup_key=f"{name}:{hash(content)}",
                ) is not None:
                    ingested += 1
                continue
            for row in rows[:10]:
                rid = str(row.get(id_field) or "")
                md = dict(row)
                md.update({"_fetch_tool": fetch_tool, "_handle_id": rid, "_id_field": id_field})
                if store.add_event(
                    user_id, source=name, event_type="custom",
                    summary=_row_summary(label, row, template), body="",
                    metadata=md, dedup_key=(f"{name}:{rid}" if rid else ""),
                ) is not None:
                    ingested += 1
        except Exception as exc:
            _log.warning("custom source %s failed for %s: %s", src.get("name"), user_id, exc)
    return ingested


async def events_loop(stop_event: asyncio.Event) -> None:
    """Background poll loop. Ticks every ``events.poll_seconds``; per user runs due
    adapters. New events nudge the arbiter via a debounced force-pulse."""
    _log.info("events loop started")
    while not stop_event.is_set():
        try:
            # The loop always ticks (cheap — adapters self-gate per user). The
            # synthetic source falls back to the local bucket when no agents run.
            synthetic = os.environ.get("CLAW_EVENTS_SYNTHETIC", "").lower() in ("1", "true", "yes")
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
    enabled=lambda _uid: os.environ.get("CLAW_EVENTS_SYNTHETIC", "").lower() in ("1", "true", "yes"),
))

# Real adapters (Calendar, Gmail) are registered in event_sources_google.py once
# the FD-side Google fetch is wired — kept out of this module so the framework
# stays validate-able without creds.
try:
    from captain_claw.flight_deck import event_sources_google  # noqa: F401
except Exception:  # not yet present / import guarded
    pass
