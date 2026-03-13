# Summary: rest_orchestrator.py

# rest_orchestrator.py Summary

**Summary:**
REST API handler module providing HTTP endpoints for orchestrator control, task lifecycle management, and workflow operations. Implements async aiohttp handlers that delegate to an underlying orchestrator instance, with comprehensive error handling, logging, and support for LLM-based request rephrasing.

**Purpose:**
Exposes orchestrator functionality (task execution, state management, workflow persistence) as a REST API, enabling client applications to decompose user requests into executable task graphs, manage task execution states, and save/load workflow templates. Bridges the gap between user-facing HTTP requests and the orchestrator's internal graph execution engine.

**Most Important Functions/Classes:**

1. **`prepare_orchestrator()`** — Decomposes a user input string into a task graph without executing it (preview mode). Accepts optional model override and auto-selection flags. Returns structured task list with dependencies and execution plan. Critical for workflow planning and validation before execution.

2. **`rephrase_orchestrator_input()`** — Transforms casual user requests into structured orchestrator prompts using an LLM. Handles dynamic model selection with provider resolution, API key management, and session logging. Includes timeout logic for local models (Ollama). Fallback to original input on failure.

3. **`reset_orchestrator()`** — Cancels all in-flight work and resets orchestrator to idle state. Simple but critical for emergency stops and state recovery.

4. **Task Control Functions** (`edit_orchestrator_task()`, `update_orchestrator_task()`, `restart_orchestrator_task()`, `pause_orchestrator_task()`, `resume_orchestrator_task()`, `postpone_orchestrator_task()`) — Provide granular task lifecycle management. Enable mid-execution editing, pausing, resuming, and timeout postponement. Each validates task_id and delegates to orchestrator methods.

5. **Workflow Persistence Functions** (`list_workflows()`, `save_workflow()`, `load_workflow()`, `delete_workflow()`) — CRUD operations for workflow templates. Support task overrides and per-workflow model selection. Enable workflow reuse and template management.

6. **Metadata Endpoints** (`get_orchestrator_status()`, `get_orchestrator_skills()`, `get_orchestrator_sessions()`, `get_orchestrator_models()`) — Provide introspection into available resources, current state, and configuration options for client UI rendering.

**Architecture & Dependencies:**
- **Framework:** aiohttp (async HTTP server)
- **Key Dependencies:** `captain_claw.instructions` (InstructionLoader for prompt templates), `captain_claw.llm` (LLM provider abstraction), `captain_claw.config` (configuration management), `captain_claw.logging` (structured logging)
- **Integration Points:** Requires `WebServer` instance with `_orchestrator` and `agent` attributes. Orchestrator must implement async methods for task/workflow operations.
- **Error Handling:** Consistent pattern—check orchestrator/agent availability, parse JSON, validate inputs, catch exceptions, return JSON responses with status codes (400 for client errors, 500 for server errors)
- **Logging:** Structured logging with context (input length, model names, error types) for debugging and monitoring