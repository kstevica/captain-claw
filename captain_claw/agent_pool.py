"""Agent pool for parallel session orchestration.

Manages Agent lifecycle: create, get-or-create by session, evict idle.
Each worker agent is bound to a specific session and bypasses the normal
initialize() flow (which loads the last-active session).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable

from captain_claw.config import get_config
from captain_claw.instructions import InstructionLoader
from captain_claw.logging import get_logger
from captain_claw.llm import LLMProvider, get_provider
from captain_claw.session import get_session_manager
from captain_claw.tools import get_tool_registry

log = get_logger(__name__)


class AgentPool:
    """Pool of worker Agent instances for parallel session execution.

    Each agent is keyed by session_id and automatically binds to its
    session on creation. Workers auto-approve tool calls (the user
    already approved orchestration at the top level).
    """

    def __init__(
        self,
        max_agents: int = 50,
        idle_evict_seconds: float = 300.0,
        provider: LLMProvider | None = None,
        status_callback: Callable[[str], None] | None = None,
        tool_output_callback: Callable[[str, dict[str, Any], str], None] | None = None,
    ):
        self.max_agents = max(1, int(max_agents))
        self.idle_evict_seconds = max(30.0, float(idle_evict_seconds))
        self._provider = provider
        self._status_callback = status_callback
        self._tool_output_callback = tool_output_callback
        self._agents: dict[str, Any] = {}  # session_id → Agent
        self._last_used: dict[str, float] = {}
        self._lock = asyncio.Lock()
        # Per-session creation locks — prevents duplicate concurrent creation
        # for the same session while allowing different sessions to create
        # agents in parallel.
        self._creating: dict[str, asyncio.Lock] = {}
        # Shared instruction loader so all workers reuse the same template cache.
        self._shared_instructions = InstructionLoader()

    @property
    def size(self) -> int:
        """Current number of agents in the pool."""
        return len(self._agents)

    async def get_or_create(self, session_id: str) -> Any:
        """Get a cached agent for session_id or create a fresh one.

        Worker agents:
        - Are bound to a specific session (not last-active)
        - Auto-approve tool calls
        - Share instruction templates cache
        - Share the LLM provider instance
        - Skip full initialize() (no last-active session loading)

        The global lock is held only for dict lookups/inserts so that
        multiple sessions can create their agents concurrently (the heavy
        part — LLM provider init, session loading, tool registration).

        Returns:
            Agent instance bound to session_id.
        """
        # --- Fast path: already exists. ---
        async with self._lock:
            if session_id in self._agents:
                self._last_used[session_id] = time.monotonic()
                return self._agents[session_id]

            # Get or create a per-session lock so only one coroutine
            # creates the agent for this session; others wait on *it*
            # (not the global lock).
            if session_id not in self._creating:
                self._creating[session_id] = asyncio.Lock()
            session_lock = self._creating[session_id]

        # --- Per-session lock: create agent outside global lock. ---
        async with session_lock:
            # Re-check: another coroutine may have finished creating it.
            async with self._lock:
                if session_id in self._agents:
                    self._last_used[session_id] = time.monotonic()
                    self._creating.pop(session_id, None)
                    return self._agents[session_id]

                # Evict idle agents if at capacity.
                if len(self._agents) >= self.max_agents:
                    evicted = self._evict_idle_locked()
                    if evicted == 0 and len(self._agents) >= self.max_agents:
                        # Force evict the oldest.
                        oldest_id = min(self._last_used, key=self._last_used.get)
                        self._agents.pop(oldest_id, None)
                        self._last_used.pop(oldest_id, None)
                        log.info("Force-evicted oldest agent from pool", session_id=oldest_id)

            # Heavy work runs WITHOUT the global lock — concurrent creation
            # of agents for *different* sessions proceeds in parallel.
            agent = await self._create_worker_agent(session_id)

            # --- Register under global lock. ---
            async with self._lock:
                self._agents[session_id] = agent
                self._last_used[session_id] = time.monotonic()
                self._creating.pop(session_id, None)
                log.info("Created worker agent", session_id=session_id, pool_size=len(self._agents))
            return agent

    async def _create_worker_agent(self, session_id: str) -> Any:
        """Create a worker agent bound to a specific session.

        Bypasses Agent.initialize() and manually configures the agent
        for worker use.
        """
        # Import here to avoid circular dependency.
        from captain_claw.agent import Agent

        # Workers auto-approve everything — user already approved at
        # orchestration level.
        def _auto_approve(question: str) -> bool:
            return True

        agent = Agent(
            provider=self._provider or get_provider(),
            status_callback=self._status_callback,
            tool_output_callback=self._tool_output_callback,
            approval_callback=_auto_approve,
        )

        # Override provider if not already set.
        if agent.provider is None:
            agent.provider = get_provider()

        # Bind to specific session instead of last-active.
        session_manager = get_session_manager()
        session = await session_manager.load_session(session_id)
        if session is None:
            # Session doesn't exist yet — create it.
            session = await session_manager.create_session(
                name=f"orchestrator::{session_id}",
            )
        agent.session = session
        agent.session_manager = session_manager

        # Sync runtime flags from session metadata.
        agent._sync_runtime_flags_from_session()

        # Register default tools (shared registry, per-call overrides
        # handle multi-agent safety via Phase 1 changes).
        agent._register_default_tools()

        # Share instruction loader cache.
        agent.instructions = self._shared_instructions

        # Mark as initialized to allow complete() calls.
        agent._initialized = True

        return agent

    async def evict(self, session_id: str) -> bool:
        """Remove a specific agent from the pool (e.g. after task edit).

        Returns True if an agent was evicted.
        """
        async with self._lock:
            removed = self._agents.pop(session_id, None)
            self._last_used.pop(session_id, None)
            self._creating.pop(session_id, None)
            if removed is not None:
                log.info("Evicted agent for edited task", session_id=session_id)
                return True
            return False

    async def release(self, session_id: str) -> None:
        """Mark agent as idle (eligible for eviction)."""
        async with self._lock:
            self._last_used[session_id] = time.monotonic()

    async def evict_idle(self) -> int:
        """Free agents that have been unused beyond idle_evict_seconds."""
        async with self._lock:
            return self._evict_idle_locked()

    def _evict_idle_locked(self) -> int:
        """Internal eviction (must be called while holding self._lock)."""
        now = time.monotonic()
        threshold = self.idle_evict_seconds
        to_evict: list[str] = []
        for sid, last_used in self._last_used.items():
            if (now - last_used) > threshold:
                to_evict.append(sid)

        for sid in to_evict:
            self._agents.pop(sid, None)
            self._last_used.pop(sid, None)

        if to_evict:
            log.info("Evicted idle agents", count=len(to_evict), remaining=len(self._agents))
        return len(to_evict)

    async def shutdown(self) -> None:
        """Release all agents from the pool."""
        async with self._lock:
            count = len(self._agents)
            self._agents.clear()
            self._last_used.clear()
            log.info("Agent pool shut down", released=count)
