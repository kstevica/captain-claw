"""Auto-wake hook for vast.ai GPU instances.

When an agent uses an Ollama provider whose ``base_url`` points to a
vast.ai instance, this module ensures the instance is running before the
request is made.  It also records activity to drive the auto-stop timer.

Two modes of operation:

1. **Flight Deck process** — ``register_manager()`` is called at startup,
   so ``_manager`` is available.  Wake uses the in-process manager directly.

2. **Agent process** — ``_manager`` is ``None``.  Wake falls back to HTTP
   calls against the Flight Deck API (``POST /fd/vastai/wake``) using the
   ``FD_URL`` environment variable that Flight Deck injects when spawning
   agents.

Usage (integrated into OllamaProvider)::

    from captain_claw.vastai.wake import maybe_wake_instance
    await maybe_wake_instance(base_url)   # no-op if not a vast.ai URL
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, Awaitable

import httpx

if TYPE_CHECKING:
    from captain_claw.vastai.manager import VastAIManager

log = logging.getLogger(__name__)

# Module-level reference to the manager (set by Flight Deck on startup).
_manager: VastAIManager | None = None

# Callback to inform the user about wake-up (set by the UI layer).
_wake_callback: Callable[[str], Awaitable[None]] | None = None

# Timeout for the remote wake HTTP call (seconds).
_REMOTE_WAKE_TIMEOUT = 150  # Must exceed manager.ensure_running's 120s.


def register_manager(manager: VastAIManager) -> None:
    """Called once during Flight Deck startup to enable auto-wake."""
    global _manager
    _manager = manager


def register_wake_callback(cb: Callable[[str], Awaitable[None]]) -> None:
    """Register an optional callback that is awaited with a status message
    when an instance is being woken up."""
    global _wake_callback
    _wake_callback = cb


def _get_fd_url() -> str:
    """Resolve the Flight Deck base URL from environment."""
    return (
        os.environ.get("FD_URL", "")
        or os.environ.get("FD_INTERNAL_URL", "")
    ).strip().rstrip("/")


async def maybe_wake_instance(base_url: str) -> None:
    """If *base_url* belongs to a tracked vast.ai instance that is not
    running, start it and wait for Ollama to become ready.

    This is a no-op if:
    - The URL does not match any tracked instance
    - The instance is already running and healthy
    - No manager AND no Flight Deck URL available
    """
    if not base_url:
        return

    # ── Path 1: in-process manager (Flight Deck server) ──
    if _manager is not None:
        await _wake_via_manager(base_url)
        return

    # ── Path 2: remote via Flight Deck API (agent process) ──
    fd_url = _get_fd_url()
    if not fd_url:
        return  # Not connected to Flight Deck — nothing we can do.

    await _wake_via_api(base_url, fd_url)


# ------------------------------------------------------------------
# Path 1: in-process (Flight Deck)
# ------------------------------------------------------------------

async def _wake_via_manager(base_url: str) -> None:
    """Wake using the local VastAIManager instance."""
    assert _manager is not None

    inst = _manager.find_instance_by_url(base_url)
    if inst is None:
        return

    # Fast path: already running and ready.
    if inst.state.value == "running" and inst.ollama_ready:
        _manager.touch_activity(inst.id)
        return

    # Need to wake — notify user if callback is registered.
    label = inst.label or f"Instance {inst.id}"
    msg = f"GPU Cloud instance \"{label}\" ({inst.gpu_name}) is waking up..."
    log.info(msg)

    if _wake_callback:
        try:
            await _wake_callback(msg)
        except Exception:
            pass

    # This blocks until the instance is running + Ollama ready (up to 120s).
    await _manager.ensure_running(inst.id, timeout=120)

    if _wake_callback:
        try:
            await _wake_callback(f"GPU Cloud instance \"{label}\" is ready.")
        except Exception:
            pass


# ------------------------------------------------------------------
# Path 2: remote HTTP call (agent process)
# ------------------------------------------------------------------

async def _wake_via_api(base_url: str, fd_url: str) -> None:
    """Wake a vast.ai instance by calling the Flight Deck REST API.

    ``POST /fd/vastai/wake`` with ``{"base_url": "..."}`` triggers
    ``ensure_running()`` on the Flight Deck side and blocks until the
    instance is ready (or returns 404 if the URL doesn't match).
    """
    secret = os.environ.get("FD_AGENT_SHARED_SECRET", "").strip()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if secret:
        headers["X-Agent-Secret"] = secret

    log.info("Requesting wake for %s via Flight Deck API at %s", base_url, fd_url)

    try:
        async with httpx.AsyncClient(timeout=_REMOTE_WAKE_TIMEOUT) as client:
            resp = await client.post(
                f"{fd_url}/fd/vastai/wake",
                json={"base_url": base_url},
                headers=headers,
            )
            if resp.status_code == 404:
                # URL doesn't match any tracked instance — not a vast.ai URL.
                return
            if resp.status_code == 200:
                data = resp.json()
                label = data.get("label") or data.get("id", "")
                log.info("GPU Cloud instance %s is awake and ready.", label)
            else:
                log.warning(
                    "Wake request failed: %d %s",
                    resp.status_code,
                    resp.text[:200],
                )
    except httpx.TimeoutException:
        log.warning("Wake request timed out after %ds for %s", _REMOTE_WAKE_TIMEOUT, base_url)
    except Exception:
        log.warning("Wake request failed for %s", base_url, exc_info=True)
