"""Pub/sub bus for MCP-state changes inside Flight Deck.

Phase 2.3 introduced a "hot push" channel so captain-claw agents can
re-register MCP proxy tools the moment the FD admin changes a server
(adds, removes, edits, or upstream signals
``notifications/tools/list_changed``). Without this, an agent has to
restart to pick up new tools — fine for development, painful in
production.

Architecture
------------

* :class:`MCPEventBus` — process-wide singleton with a fan-out
  publish/subscribe API. Each subscriber owns one
  :class:`asyncio.Queue`; publishers drop the same payload into every
  queue. Slow subscribers don't block other subscribers (we use
  ``Queue.put_nowait`` and silently drop if the queue is full — events
  are advisory and idempotent, missing one only delays a refresh by
  the next event).

* Event payloads are tiny dicts::

      {"type": "server_added",   "server": "examplemcp"}
      {"type": "server_removed", "server": "examplemcp"}
      {"type": "server_updated", "server": "examplemcp"}
      {"type": "tools_changed",  "server": "examplemcp"}
      {"type": "ping"}

  ``ping`` events are emitted periodically by the SSE route to keep
  the connection alive through proxies that drop idle TCP.

* The agent-side connector subscribes via the ``GET /fd/mcp/agent/events``
  SSE endpoint (which filters by the caller's allowlist) and re-runs
  registration when it sees an event for a server it's allowed to use.

Why a bus, not a callback list?
-------------------------------

Subscribers are async iterators that may live across many tasks; a
callback list would force every publisher to know every subscriber's
event loop. The queue per subscriber decouples the two — publishers
``publish()`` synchronously, subscribers ``async for`` at their own
pace.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from captain_claw.logging import get_logger

log = get_logger(__name__)


# Per-subscriber queue size. Generous because events are tiny dicts and
# a slow subscriber gracefully drops, so there's no memory risk — but
# small enough that a wedged subscriber doesn't accumulate forever.
_QUEUE_MAXSIZE = 256


class MCPEventBus:
    """Fan-out pub/sub for MCP state changes."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()

    def publish(self, event: dict[str, Any]) -> None:
        """Deliver ``event`` to every active subscriber.

        Best-effort: if a subscriber's queue is full we drop and log.
        Synchronous so storage / route handlers can call it without
        awaiting — the queues themselves are bounded so backpressure is
        bounded too.
        """
        if not self._subscribers:
            return
        # Snapshot so a subscriber unsubscribing mid-publish doesn't
        # mutate the set we're iterating.
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                log.warning(
                    "MCP event bus: subscriber queue full, dropping event",
                    event_type=event.get("type"),
                )

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a new subscriber and return its queue.

        Caller must call :meth:`unsubscribe` when done — typically in a
        ``finally`` block at the end of the SSE handler.
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Convenience async iterator for callers that want
        ``async for event in bus.stream(): ...`` semantics.

        The iterator stays open until the consumer breaks out; the
        underlying subscription is cleaned up automatically.
        """
        queue = await self.subscribe()
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            await self.unsubscribe(queue)


# ── module-level singleton ──────────────────────────────────────────


_bus: MCPEventBus | None = None


def get_event_bus() -> MCPEventBus:
    global _bus
    if _bus is None:
        _bus = MCPEventBus()
    return _bus


# ── convenience publishers ──────────────────────────────────────────


def publish_server_added(name: str) -> None:
    get_event_bus().publish({"type": "server_added", "server": name})


def publish_server_removed(name: str) -> None:
    get_event_bus().publish({"type": "server_removed", "server": name})


def publish_server_updated(name: str) -> None:
    get_event_bus().publish({"type": "server_updated", "server": name})


def publish_tools_changed(name: str) -> None:
    get_event_bus().publish({"type": "tools_changed", "server": name})
