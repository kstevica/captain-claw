# Captain Claw v0.4.14 Release Notes

## Orchestrator Overhaul: Parallel Execution, Structured Data Flow & Real-Time Observability

v0.4.14 is a major upgrade to the orchestrator's parallel execution engine. Four new subsystems make multi-task orchestration more powerful, more reliable, and more observable.

---

### Shared Workspace — Structured Data Flow Between Tasks

Tasks within an orchestration run now share a **namespaced key-value store** for passing structured data without intermediate files.

- **Automatic output sharing** — Every task's text output (and validated JSON output) is written to the workspace automatically, so downstream tasks can access upstream results as structured data rather than raw text blobs
- **Manual read/write** — Workers can use the new `workspace_read` and `workspace_write` tools for fine-grained data sharing during execution
- **Namespace isolation** — Keys are prefixed by task ID to prevent collisions between concurrent tasks
- **Prompt injection** — Relevant workspace data from upstream dependencies is automatically injected into downstream worker prompts
- **Real-time sync** — Workspace changes are broadcast to Flight Deck via WebSocket

Tasks can declare `workspace_inputs` and `workspace_outputs` for explicit data flow documentation, visible as cyan/amber arrow indicators on task nodes in the workflow editor.

### Structured Output Validation — JSON Schema Enforcement

Tasks can now declare a **JSON Schema** via `output_schema` to guarantee structured output:

- After a worker completes, the output is extracted (supports raw JSON, ```json fences, and embedded JSON detection) and validated against the schema
- If validation fails, the agent gets **one automatic retry** with the validation error and schema fed back into the prompt
- On success, `task.validated_output` stores the parsed data and it's written to the shared workspace as `json` content type
- Validation uses `jsonschema` when available, with a built-in recursive fallback validator for environments without the library

This enables reliable data pipelines — e.g., "extract products as `{name, price, url}`" with guaranteed schema conformance. Tasks with schemas show a violet `{ }` badge in the workflow editor.

### Explicit Task Pipelines — `run_tasks()` API

New REST endpoints let you define a task DAG **without LLM decomposition**:

| Endpoint | Purpose |
|---|---|
| `POST /api/orchestrator/prepare-tasks` | Build a graph from an explicit task list (preview) |
| `POST /api/orchestrator/run-tasks` | Build + execute in one call |
| `GET /api/orchestrator/workspace` | Retrieve the shared workspace snapshot |

Each task in the payload supports: `id`, `title`, `description`, `depends_on`, `model_id`, `session_name`, `session_id`, `output_schema`, `output_schema_name`, `workspace_outputs`, `workspace_inputs`.

This powers Flight Deck's Swarm Workflows and enables programmatic pipeline construction from any client. Flight Deck proxies these endpoints to agents via `POST /fd/orchestrator/{slug}/run-tasks`.

### Trace Timeline — Real-Time Orchestrator Observability

Every orchestration run now emits **structured trace spans** for full lifecycle visibility:

| Span Type | What It Tracks |
|---|---|
| `decompose` | LLM request decomposition (timing) |
| `execution` | Overall DAG execution with completed/failed task counts |
| `task` | Individual worker tasks with token usage and error details |
| `synthesize` | Final synthesis step |

**Flight Deck integration:**
- A new **Activity icon** (with live badge count) appears in the chat panel tab bar
- Click to toggle a full-height **Gantt-style trace timeline** showing all spans
- Color-coded bars per span type (purple=decompose, blue=execution, green=task, amber=synthesize)
- Status indicators: checkmark (completed), X (failed), pulsing dot (running)
- Duration per span and summary header with total span count, completion stats, and token accounting
- Spans stream in real-time via WebSocket as the orchestration progresses

Trace data is also available programmatically via `GET /api/orchestrator/traces` (returns spans + summary JSON).

---

### New Tools

| Tool | Description |
|---|---|
| `workspace_read` | Read values or list keys from the shared workspace (supports fuzzy key matching) |
| `workspace_write` | Write structured data to the shared workspace (auto-namespaced by task) |

Both tools gracefully return errors when used outside of orchestration mode (single-agent chat).

### New Files

| File | Purpose |
|---|---|
| `captain_claw/shared_workspace.py` | Core SharedWorkspace class — thread-safe namespaced KV store |
| `captain_claw/tools/shared_workspace.py` | WorkspaceReadTool and WorkspaceWriteTool implementations |
| `captain_claw/output_validation.py` | JSON extraction, schema validation, retry prompt builder |
| `captain_claw/tracing.py` | TraceSpan, TraceContext, TraceCallback for structured observability |
| `flight-deck/src/stores/traceStore.ts` | Zustand store for trace spans with real-time WS updates |
| `flight-deck/src/components/observability/TraceTimeline.tsx` | Gantt-style trace timeline component |

### Modified Files

- **`session_orchestrator.py`** — Shared workspace creation, output validation with retry, trace span instrumentation across all phases (decompose/execute/task/synthesize), `prepare_tasks()` and `run_tasks()` methods
- **`task_graph.py`** — New fields on OrchestratorTask: `workspace_outputs`, `workspace_inputs`, `output_schema`, `output_schema_name`, `validated_output`
- **`agent_guard_mixin.py`** — Context injection of `_shared_workspace` and `_workspace_task_id` into tool arguments
- **`agent_context_mixin.py`** — Registered workspace tools as always-on tools
- **`web/rest_orchestrator.py`** — New endpoints: `prepare_tasks`, `run_tasks`, `get_workspace_snapshot`, `get_traces`
- **`web_server.py`** — Route registration for all new orchestrator endpoints
- **`flight_deck/server.py`** — Proxy routes for `prepare-tasks`, `run-tasks`, `workspace`, `traces`
- **`flight-deck/src/types/index.ts`** — New types: `TraceSpan`, `TraceSummary`, `TraceData`, `WorkspaceEntry`, `WorkspaceSnapshot`; extended `SwarmTask` with workspace/schema fields
- **`flight-deck/src/services/api.ts`** — New API functions: `prepareTasksOnAgent`, `runTasksOnAgent`, `getTracesFromAgent`
- **`flight-deck/src/stores/chatStore.ts`** — Wired `orchestrator_event` / `trace_span` events to the trace store
- **`flight-deck/src/components/agents/ChatPanel.tsx`** — Activity icon button with badge count in tab bar, toggleable trace panel
- **`flight-deck/src/components/workflow/TaskNode.tsx`** — Visual indicators for workspace inputs (cyan), outputs (amber), and schema (violet)

### Upgrade Notes

- **No breaking changes** — All new features are additive. Existing orchestrations work identically.
- **No config changes required** — Shared workspace, validation, and tracing activate automatically when orchestration runs.
- **New tools are always-on** — `workspace_read` and `workspace_write` are available in all sessions but return graceful errors outside orchestration.
- **Flight Deck rebuild required** — The frontend bundle must be rebuilt (`npm run build` in `flight-deck/`) for the trace timeline to appear.
