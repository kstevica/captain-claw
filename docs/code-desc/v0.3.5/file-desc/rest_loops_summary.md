# Summary: rest_loops.py

# rest_loops.py Summary

**Summary:** This module implements REST API handlers and execution logic for running workflows repeatedly with different variable sets. It manages the complete lifecycle of loop execution—from initiation through sequential iteration processing to completion—with real-time status broadcasting and graceful cancellation support.

**Purpose:** Solves the problem of batch workflow execution where users need to run the same workflow multiple times with varying input parameters. Provides HTTP endpoints for starting loops, monitoring progress, and stopping execution, while maintaining detailed state tracking and broadcasting events to connected clients.

**Most Important Functions/Classes:**

1. **`start_loop(server, request)`** — REST handler (POST /api/loops/start) that validates incoming workflow and iteration data, generates a unique loop ID, initializes iteration state objects, and spawns an async task to execute the loop. Performs input validation (workflow name, iterations list) and prevents concurrent loop execution.

2. **`run_loop(server, loop_id, workflow_name, iterations)`** — Core execution engine that sequentially processes each iteration. For each iteration: creates a SessionOrchestrator instance, loads the workflow, merges default and user-provided variables, executes the workflow, captures results/errors, and broadcasts real-time events. Implements cancellation checking and comprehensive error handling with cleanup via orchestrator shutdown.

3. **`get_loop_status(server, request)`** — REST handler (GET /api/loops/status) that returns the current loop runner state including loop ID, workflow name, execution state, and detailed per-iteration status (pending/running/completed/failed/cancelled with duration and results).

4. **`stop_loop(server, request)`** — REST handler (POST /api/loops/stop) that sets a stop flag checked during iteration processing, enabling graceful cancellation that marks remaining iterations as "cancelled" and broadcasts a loop_stopped event.

5. **`_make_loop_broadcast(idx)`** — Higher-order function that wraps the broadcast callback to inject loop context (loop_id, iteration_index) into orchestrator events, enabling clients to correlate events with specific iterations within a loop.

**Architecture & Dependencies:**
- **Async/Concurrent:** Uses asyncio for non-blocking execution; loop runs sequentially but doesn't block the server
- **State Management:** Maintains `_loop_runner_state` dict on WebServer instance tracking loop metadata and per-iteration progress
- **Broadcasting:** Integrates with server's `_broadcast()` method to push real-time events (loop_started, iteration_started/completed/failed, loop_stopped/completed) to connected WebSocket clients
- **Orchestration:** Delegates workflow execution to SessionOrchestrator, which manages agent parallelization and variable resolution
- **Error Resilience:** Captures exceptions per-iteration without halting the loop; ensures orchestrator cleanup in finally blocks
- **Logging:** Structured logging with loop_id and iteration context for observability