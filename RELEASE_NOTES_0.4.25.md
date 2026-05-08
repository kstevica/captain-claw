# Captain Claw v0.4.25 Release Notes

**Release title:** Plan Mode

**Release date:** 2026-05-08

## Highlights

Captain Claw 0.4.25 ships **plan mode** end-to-end: a reviewable, verifiable, self-revising planner that turns a single user request into an ordered DAG of executable steps, runs them under the existing orchestrator, judges each step against per-step acceptance criteria, and revises failing steps automatically inside a bounded loop. Plan mode is wired through the Flight Deck chat UI as a first-class peer to free-form chat — flip the **planning toggle** on once and every subsequent message is auto-routed through `/plan` + `/plan-execute` with a live, persistent plan card rendered inline.

The release is the convergence of eight progressive milestones (`plan-mode step 2` through `step 8` plus a final round of integration polish), all under one umbrella. If you were running 0.4.24 nothing breaks — plan mode is purely additive and the slash commands stay opt-in even when the toggle is off.

## What changed

### 1. `OrchestratorTask` extended with plan fields

`OrchestratorTask` (the unit the orchestrator already executes) gained the metadata plan mode needs without forking the schema:

- `step_kind: "atomic" | "orchestrate" | "verify" | "revise"` — defaults to `atomic`. `orchestrate` steps are fanned out by the planner via the SessionOrchestrator's decomposer.
- `acceptance_criteria: str` — a one-sentence measurable check the verifier evaluates against the step's output.
- `output_schema: dict | None` — optional JSON-schema fragment for fast deterministic validation before the LLM judge runs.
- `verification_status: "unverified" | "passed" | "failed"`, `verification_notes: str` — populated by `PlanVerifier`.
- `revision_count: int`, `revision_of: str`, `original_description: str` — populated by `PlanReviser`.

`from_dict` / `to_dict` round-trip every field, so a plan written by the planner persists via `SessionOrchestrator.save_workflow` and reloads byte-identical.

### 2. `/plan <request>` — `PlanGenerator`

A new `captain_claw/plan_mode.py` module with `PlanGenerator` calls the same provider as the main agent (no separate planner-model config) and parses the JSON response into a `Plan` of validated `OrchestratorTask`s. The slash command:

1. Asks the LLM (using `plan_mode_system_prompt.md` + `plan_mode_user_prompt.md`) for a 3-8 step plan with concrete `description` / `acceptance_criteria`.
2. Validates IDs, drops dangling `depends_on` edges, coerces unknown `step_kind` values to `atomic`.
3. Persists the plan as a workflow JSON in `workspace/workflows/` and renders a Markdown preview in the chat panel.
4. Broadcasts a `plan_generated` event so the UI can paint the plan card immediately.

The planner prompt enforces a hard rule new in 0.4.25: **deliverable steps must name their output file**. Any step that drafts, writes, generates, summarizes, reports, or otherwise produces a textual artifact has to declare an explicit filename (preferably `saved/tmp/<short-slug>.md`) in both the `description` and the `acceptance_criteria`. This closes the "the plan succeeded but the brief is buried inside the run transcript" trap that earlier prototype runs hit when a step said only "Write a research brief…" — the executor returned inline text and the deliverable was lost in the 50 KB workflow output transcript.

### 3. `/plan-execute` — `PlanExecutor`

`PlanExecutor` runs a loaded plan through the existing DAG runner and adds plan-mode-specific orchestration on top:

- **Re-uses the orchestrator's worker pool.** Sequential plans serialize naturally because the planner declares each step as `depends_on: [previous_id]`; the orchestrator's existing topological scheduler does the rest.
- **`skip_synthesize=True`.** Plan mode owns its own per-step verification + result rendering, so the orchestrator's final synthesis pass is redundant work that used to leave the chat sitting at "Orchestrator: synthesizing results…" for 5 sequential LLM calls after the plan card already said `VERIFIED`. `SessionOrchestrator.execute(skip_synthesize=True)` short-circuits with a compact result line ("Plan execution finished — 6 step(s) completed.") — the plan card itself is the deliverable surface.
- **Streams events** (`plan_execution_started`, `plan_step_verified`, `plan_step_revised`, `plan_execution_verified`, `plan_execution_completed`, `plan_execution_failed`) so the chat-panel plan card updates live, step by step.
- **Cycle budget = `max_revisions + 1`.** One initial run plus up to `max_revisions` retry cycles after revision proposals.

### 4. `PlanVerifier` — two-stage acceptance gate

Step 4 lands `PlanVerifier`. Each completed step is walked in topological order and gated through:

1. **Schema validation** (fast, deterministic). When `output_schema` is set, output is parsed and validated; failures stop execution before the LLM judge runs.
2. **LLM judge** against `acceptance_criteria`. Uses `plan_mode_verifier_system_prompt.md` + `plan_mode_verifier_user_prompt.md`. Returns `{passed: bool, notes: str}`. The first failure stops the walk and surfaces a `plan_step_verified` failure event to the UI.

Steps with no `acceptance_criteria` and no `output_schema` auto-pass with a recorded note. Verifier timeouts default to 60 seconds with truncated output context (8 KB) so the judge call stays cheap.

### 5. Orchestrate-kind fan-out via `SessionOrchestrator`

`step_kind: "orchestrate"` steps are no longer coerced to atomic. `PlanExecutor._expand_orchestrate_steps` calls a pluggable `expander` (default: `orchestrate_expander_from_orchestrator(orch)`, which delegates to `SessionOrchestrator._decompose`) and rewires the graph in place:

- Each sub-task `Si` is added with `depends_on=[D]` (the original parent's deps), so the fan-out runs in parallel.
- The parent step `P` is converted to an atomic **join**: `step_kind="atomic"`, `depends_on=[S1, S2, …]`, with its description rewritten as a synthesis instruction over the sub-task outputs.
- Tasks that depended on `P` keep doing so — the join preserves the original edge, so no downstream rewiring is needed.

Sub-task IDs are namespaced (`<parent_id>__<sub_id>`) and de-duplicated against the existing graph so two orchestrate steps with overlapping sub-task IDs can never collide. Expansions emit a `plan_orchestrate_expanded` event with the rewrite map for the UI.

### 6. `PlanReviser` + bounded auto-revision loop

When a step fails verification, `PlanReviser` is asked for a sharper description. The reviser sees the failed step's title/description, its acceptance criteria, the truncated output (8 KB) and the verifier's notes, and returns:

```json
{
  "revised_description": "…",
  "revised_acceptance_criteria": "…",   // optional
  "rationale": "why this should now pass"
}
```

`PlanExecutor._apply_revision` swaps the description in place, optionally tightens the acceptance criteria, increments `revision_count`, stamps `revision_of`, and resets the failed step **plus all transitive dependents** to `PENDING` so the next `execute()` cycle re-runs only what's actually affected. Revisions are emitted as `plan_step_revised` events; the loop is bounded by `DEFAULT_MAX_REVISIONS = 2` (one initial run + two retries). Run-time failures (worker errors / timeouts) still skip the revision loop — those need orchestrator-level recovery, not a description rewrite.

### 7. Inline plan-execution card in chat

Flight Deck's `ChatPanel.tsx` renders a live plan card directly inline with the conversation:

- **Per-step list** with status chips (`pending` / `running` / `completed` / `failed`), verification chip (`verified` / `revised` / `verification failed`), and a one-line description preview that expands on click.
- **Live progress** driven by the `plan_*` event stream: steps light up as they start, transition through `running` → `completed`, and pick up a `verified` chip the moment the verifier responds.
- **Revision badge** when a step has `revision_count > 0`, with a hover popover showing the rationale and previous failure notes — the auto-revision loop is fully transparent to the user.
- **Persistent across refresh.** Plan state, the planning-mode toggle, and the card-collapsed flag are persisted to `localStorage` keyed by `containerId` (`fd.plan.${id}`). Refreshing Flight Deck mid-plan repaints the card with the same step states; nothing is lost.
- **Collapsible.** Long plans collapse to a one-line summary chip ("6 steps · 4 verified · running"); click expands.

### 8. `/planning on` — auto-route chat through `/plan` + `/plan-execute`

The new **planning toggle** is the headline UX surface for plan mode. A bold violet `ON` / muted `OFF` pill sits in the chat composer next to the message input.

- **Toggle on:** every plain-text message you send is auto-routed through `/plan <message>` followed by `/plan-execute` — no slash commands, no copy-paste between two prompts.
- **Toggle off:** chat behaves exactly like 0.4.24. The slash commands `/plan`, `/plan-execute`, and `/planning` are still available for explicit invocation.
- **State mirror.** The toggle reflects the agent's authoritative plan-mode flag — typing `/planning on` in the input flips the pill on; the agent's confirmation message is parsed and the pill stays in sync. The reflection regex covers all four agent-message formats (`auto-routing enabled`, `auto-routing disabled`, `auto-routing: on`, `auto-routing: off`) and is robust against the help-text mention "Use `/planning off` to disable" inside an enabled response.
- **Live indicator.** While a plan is running, a small emerald pulse dot appears on the pill so it's obvious at a glance that the toggle is doing something.

### 9. Worker iteration budget — research-aware estimator

`SessionOrchestrator._estimate_task_iterations` learned the shape of plan-mode research steps. Several previous runs exhausted the `_WORKER_MAX_ITERATIONS` budget mid-`gather_sources` because the estimator only counted explicit fetch calls. New `_COMPLEXITY_SIGNALS` add weight for:

- "reputable / credible / authoritative / reliable / primary sources" → +12
- "research / gather / collect … sources" → +8
- "citation / cite / references" → +2
- "at least N sources" → +6
- `web_search` mentioned as a tool → +2
- `each <thing>` patterns now also match "source"

The hard ceiling moved from **25 → 40** iterations to give multi-source research steps headroom without uncapping the loop entirely. Iteration accounting is unchanged for non-research workloads.

### 10. Connection-resilience pass

Plan runs are long — minutes of executor + verifier + reviser calls — so the WebSocket plumbing was hardened for the first time since the original Flight Deck handshake:

- **Agent ⇄ FD proxy**: the upstream `websockets.connect()` now sets `ping_interval=20` / `ping_timeout=10` so idle keepalives match aiohttp's `WebSocketResponse` heartbeat. Long verifier calls no longer race against the kernel idle-timeout.
- **FD ⇄ browser**: the React client (`agentChat.ts`) now auto-reconnects with exponential backoff (500 ms → 15 s cap), tracks `_shouldReconnect` so an explicit `disconnect()` doesn't bounce, and emits a `_reconnecting` event the chat panel surfaces as a transient banner.
- **Server-side `_send`** wraps every outbound frame in `try / except (ConnectionResetError, ConnectionError, ClientConnectionResetError)` so a browser refresh mid-step no longer fills the agent log with traceback noise.
- **`ws_handler.py` receive loop** wraps the `async for raw_msg in ws` in the same exception envelope and discards the client cleanly in `finally`. The two `ClientConnectionResetError` tracebacks reproducible in 0.4.24 are gone.

### 11. Workflow output files now viewable in the file panel

`SessionOrchestrator._save_run_output` writes the per-run Markdown transcript to `workspace/workflows/`, but `_is_allowed_path` in `web/rest_files.py` previously only allowed `workspace/saved/` and `workspace/output/`. The transcript was never registered as a logical file, so the FD file-preview proxy returned **HTTP 403 "File not in registry"** when you tried to open it from the chat — even though the file existed.

`_is_allowed_path` now permits `workspace/workflows/` alongside `saved/` and `output/`. The plan card's "open run transcript" link works without a 403.

### 12. UX polish

- **Planning toggle visibility.** The previous subtle `bg-violet-600/20` pill was easy to miss in the dim chat header; the new `bg-violet-500` saturated pill with `ring-violet-300/60` and bold `ON` text makes the active state unmissable.
- **Spinner clears on slash-command results.** The first `command_result` handler in `chatStore.ts` now resets `busy: false` / `statusText: ''` so the "Thinking…" indicator no longer lingers after `/planning on`, `/plan-execute`, etc.

## REST surface

No new endpoints — plan mode runs through the existing `/api/orchestrator/...` routes plus the slash-command WebSocket plumbing. The workflow JSON files written by `/plan` are loadable via the standard `/api/orchestrator/workflows/load` route, so any tool that already consumed Captain Claw workflows keeps working.

## File / module additions

| File | Purpose |
|---|---|
| `captain_claw/plan_mode.py` | `PlanGenerator`, `PlanExecutor`, `PlanVerifier`, `PlanReviser`, `Plan`, `PlanExecutionResult`, `RevisionProposal`, `orchestrate_expander_from_orchestrator`. |
| `captain_claw/instructions/plan_mode_system_prompt.md` | Planner system prompt. New: deliverable-filename rule. |
| `captain_claw/instructions/plan_mode_user_prompt.md` | Planner user prompt template. |
| `captain_claw/instructions/plan_mode_verifier_system_prompt.md` | Verifier judge system prompt. |
| `captain_claw/instructions/plan_mode_verifier_user_prompt.md` | Verifier judge user prompt template. |
| `captain_claw/instructions/plan_mode_reviser_system_prompt.md` | Reviser system prompt. |
| `captain_claw/instructions/plan_mode_reviser_user_prompt.md` | Reviser user prompt template. |
| `captain_claw/web/plan_commands.py` | `/plan`, `/plan-execute`, `/planning` slash-command handlers. |

Existing files materially changed: `session_orchestrator.py` (skip-synthesize, complexity signals, allow-list update), `task_graph.py` / `OrchestratorTask` (plan fields), `web/ws_handler.py` (receive-loop hardening), `web/rest_files.py` (workflows path allow-list), `web_server.py` (`_send` hardening), `flight-deck/src/components/agents/ChatPanel.tsx` (plan card + toggle), `flight-deck/src/stores/chatStore.ts` (plan slice + LS persistence + reflection regex), `flight-deck/src/services/agentChat.ts` (auto-reconnect), `flight_deck/server.py` (proxy ping params).

## Migrating from 0.4.24

Nothing required. Plan mode is opt-in:

1. **Try a one-shot plan.** Type `/plan summarize the README and write the summary to saved/tmp/readme-summary.md` then `/plan-execute`.
2. **Or flip the toggle.** Click the **OFF** pill in the chat composer to switch to **ON**, then ask the agent anything — it will plan, execute, verify, and revise without you typing a slash command.
3. **Adjust the revision budget.** `DEFAULT_MAX_REVISIONS = 2` is a module constant on `captain_claw.plan_mode`; bump it if you want longer self-correction loops.

## Known limitations / next steps

- **Per-step retry for run-time failures.** Today, if a worker crashes (timeout, transport error) the plan stops without invoking the reviser — the reviser is for verification failures only. A `step_kind: "atomic_with_retry"` variant that loops the worker is a natural follow-up.
- **No partial-resume yet.** Re-running a plan after a failure starts from the failed step, but if you edit the plan JSON manually between runs the executor doesn't diff — it trusts whatever's loaded.
- **Plan-card timeline export.** The card has all the data for an executive summary (steps, verifier notes, revisions, durations) but there's no "export run report" button yet. Easy follow-up.
- **Verifier model is the same as the agent model.** A cheaper model just for the judge (e.g. Haiku for the LLM judge while the executor stays on Sonnet) is a future cost optimization.
