# Summary: agent_pool.py

# agent_pool.py Summary

## Summary
Manages a pool of worker Agent instances for parallel session-based orchestration, enabling concurrent execution of multiple independent tasks. Each agent is bound to a specific session, auto-approves tool calls (since user approval happens at orchestration level), and shares cached resources (instruction templates, LLM provider, file registry) to minimize overhead. The pool implements intelligent lifecycle management with idle eviction, capacity-based culling, and per-session creation locks to prevent race conditions while maximizing concurrent agent initialization.

## Purpose
Solves the problem of efficiently managing multiple autonomous agents operating in parallel across different sessions within a single orchestration run. Prevents resource exhaustion through capacity limits and idle timeouts, eliminates duplicate agent creation via per-session locking, and enables cross-task file resolution through shared registries—critical for complex multi-step workflows where subtasks need to reference outputs from sibling tasks.

## Most Important Functions/Classes/Procedures

1. **`get_or_create(session_id, file_registry)`**
   - Core method implementing double-checked locking pattern: fast path checks global dict under lock, then releases lock for expensive agent creation, re-validates before registration. Handles capacity management by evicting idle agents or force-evicting oldest when at max_agents limit. Updates file_registry on every call to ensure agents see latest cross-task outputs.

2. **`_create_worker_agent(session_id, file_registry)`**
   - Constructs a worker agent bound to a specific session (bypassing normal initialize() flow). Configures auto-approval callback, loads/creates session, syncs runtime flags, registers default tools, attaches shared instruction loader and file registry, sets worker-specific tuning (max_iterations=5, _is_worker=True). Avoids circular imports via local Agent import.

3. **`_evict_idle_locked()`**
   - Internal eviction logic (requires holding global lock) that identifies agents unused beyond idle_evict_seconds threshold and removes them. Called both reactively (when pool reaches capacity) and proactively (via evict_idle() public method). Logs eviction count and remaining pool size.

4. **`evict(session_id)` and `release(session_id)`**
   - Explicit lifecycle management: evict() removes specific agent (e.g., after task edit), release() updates last_used timestamp to mark agent as idle. Both acquire global lock for dict operations.

5. **`get_scale_progress_age(session_id)`**
   - Monitoring utility returning seconds since agent's last micro-loop progress, enabling timeout detection for stuck workers. Returns None if agent doesn't exist or has no recorded progress timestamp.

## Architecture & Dependencies

**Concurrency Model:**
- Global `asyncio.Lock` protects agent dict/metadata (minimal hold time)
- Per-session `asyncio.Lock` in `_creating` dict prevents duplicate concurrent creation for same session while allowing different sessions to create agents in parallel
- Double-checked locking pattern minimizes contention

**Resource Sharing:**
- Single `InstructionLoader` instance cached across all agents (template reuse)
- Single LLM provider instance (or factory for runtime model switches)
- Shared `FileRegistry` passed to all agents for cross-task file resolution
- Shared `_deep_memory` (Typesense index) to avoid re-initialization

**Key Dependencies:**
- `captain_claw.agent.Agent` (imported locally to avoid circular deps)
- `captain_claw.session.get_session_manager()` (session lifecycle)
- `captain_claw.llm.get_provider()` (LLM provider)
- `captain_claw.instructions.InstructionLoader` (template caching)
- `captain_claw.tools.get_tool_registry()` (tool registration)

**Configuration Parameters:**
- `max_agents` (default 50): pool capacity
- `idle_evict_seconds` (default 300): eviction threshold
- `session_name_prefix` (default "orchestrator"): naming convention for auto-created sessions
- Callbacks: status_callback, tool_output_callback, thinking_callback (propagated to agents)

**Worker-Specific Tuning:**
- Auto-approval enabled (user already approved at orchestration level)
- max_iterations=5 (prevents endless loops on simple tasks)
- _is_worker=True flag (disables heavyweight pipeline features like list-task extraction, contract validation)
- Skips full initialize() flow (no last-active session loading)