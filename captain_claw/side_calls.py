"""Side calls on a self-hosted model wait until the agent's run is over.

A self-hosted server (llama.cpp, Ollama, vLLM, LM Studio …) keeps the cache of
the prompt it ran last. A run sends the same long prompt call after call, and
each call reuses almost all of it — until another request lands in between: a
Work board triage, a "consciousness" reflection, a task name. The run's next
call then reads its whole context again (90k tokens, ~30 s on a home server,
where a cached call took 2–6 s). A cloud API keeps prompt caches per prompt, so
there a side call costs the run nothing and is never held.

Calls that can wait — borrowed through ``/api/llm/complete`` with ``defer``,
and the task name — wait while the agent is at work, for up to a cap, then go.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

# Longest a side call waits for a run to end before it goes anyway.
DEFAULT_MAX_WAIT_S = 900.0
MAX_WAIT_CAP_S = 1800.0
_POLL_S = 0.5


def evicts_cache(provider: Any) -> bool:
    """Whether a request from outside the run would push the run's prompt
    out of this provider's server cache: a self-hosted server, not a cloud
    API (Ollama's own cloud included)."""
    from captain_claw.llm import reasoning

    name = str(getattr(provider, "provider", "") or "").strip().lower()
    base_url = str(getattr(provider, "base_url", "") or "").strip()
    if not name:
        return False
    transport = reasoning.transport_for(name, base_url or None)
    if transport == reasoning.T_OLLAMA:
        try:
            host = (urlparse(base_url if "://" in base_url else f"http://{base_url}").hostname or "").lower()
        except ValueError:
            host = ""
        return not host.endswith("ollama.com")
    return transport == reasoning.T_LOCAL and bool(base_url)


def max_wait(value: Any) -> float:
    """A caller's ``defer_max_s``, clamped (the default when missing)."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_WAIT_S
    return max(0.0, min(seconds, MAX_WAIT_CAP_S))


async def wait_until_quiet(at_work: Callable[[], bool], max_wait_s: float = DEFAULT_MAX_WAIT_S) -> float:
    """Wait while ``at_work()`` is true, at most *max_wait_s*; the seconds waited."""
    start = time.monotonic()
    while at_work() and time.monotonic() - start < max_wait_s:
        await asyncio.sleep(_POLL_S)
    return time.monotonic() - start
