# Flow Engine — Design Draft (v2)

Status: draft for review. A declarative **Flow engine** that runs deterministic
multi-step automations with **agent judgment as steps**. Runs **inside Flight Deck**
and executes steps on the **existing agent pool**. Not a visual n8n.

> Naming: FD already uses "process registry" for the *agent subprocess* table
> (`_load_process_registry`). To avoid collision the feature is **"Flows"**
> (tables `flows`, `flow_runs`). User-facing we can still call it the process engine.

## The one principle

> **If it must be reliable or repeatable, it's a step.
> If it needs judgment, it's an agent node inside the flow.**

FD's runner owns the *seams* where agents drift — triggering, routing, sequencing,
guardrails, unattended execution. Pooled agents do the *work*. A flow that doesn't
match falls through to the normal agentic turn (the flexible 20%). Generalizes the
video auto-analysis pattern we hand-coded.

## Runtime: FD-resident, pool-dispatched

The **FlowRunner lives in Flight Deck** (FastAPI). It holds flow state + context and
is the deterministic loop. Each step executes on a **pooled agent**, selected from
`/fd/fleet`:

- **`tool` step** → direct RPC to a pooled agent's **new `/api/tool` endpoint**
  (runs `registry.execute(name, args)`, **no LLM, no agent turn**). The reliable
  spine, cheap.
- **`agent` step** → `/fd/consult-peer` to a pooled agent with a **scoped prompt +
  step-local tool policy** (the `deny:[scripts,shell]` pattern). Judgment only.
- Both reuse the consult/transfer plumbing (incl. the busy-retry + in-flight guards
  we just added).

### Agent selection & affinity

A step declares `on:` an agent selector:
- `origin` (default) — the agent the inbound message arrived at. **File-bound tool
  steps run here** (file locality: `video_vision` needs the file in that workspace).
- `capability: vision|...` — FD picks a running, idle agent whose fleet
  `description`/tags match (e.g. MiniMax = vision). Files are shipped with
  `_transfer_file_to_agent` when crossing agents.
- `name: <agent>` — a specific agent.

Selection = read `/fd/fleet` → filter by selector + `status: running` → prefer idle
→ dispatch (busy-retry covers transient contention).

### Why this is mostly assembly

| Need | Reuse |
|---|---|
| Dispatch to pool | `/fd/consult-peer` (+ busy-retry, `_active_consults/_delegates`) |
| Deterministic tool step | **new** `/api/tool` on agent web server → `registry.execute` |
| Pool + capabilities | `/fd/fleet` (name, kind, status, description) |
| Guardrails | `ToolPolicy(allow/deny)` passed into the scoped consult |
| Cross-agent files | `_transfer_file_to_agent` |
| Time triggers | cron worker |
| Decisions (approve/pick) | intentions decisions bus + `follow_through()` |
| Channel I/O | glasses `_WAID_CHANNEL`/`_broadcast`, `send_whatsapp_text` |
| Inbound trigger point | WhatsApp bridge forward fns + glasses channel bus (FD-side) |

New code: sqlite schema + FlowRunner + trigger router + `/api/tool` endpoint + the
FD CRUD API + the React UI.

## The spec (stored as rows; authored via UI)

```yaml
id: video-describe
name: Describe attached video
enabled: true
priority: 50
trigger:
  on: message            # message | schedule | decision
  channel: any           # any | whatsapp | glasses | web
  match: { kind: rule, rules: [has_video] }   # rule | classifier | always
steps:
  - id: analyze
    type: tool
    on: origin           # file lives on the originating agent
    tool: video_vision
    args: { path: "{{trigger.video_path}}" }
  - id: reply
    type: agent
    on: origin
    guardrails: { deny: [scripts, shell] }
    prompt: "Describe the video using ONLY this analysis:\n{{steps.analyze.output}}"
output: { channel: same }
guardrails: { max_steps: 12, timeout_s: 600 }
```

Node types: **tool** (deterministic RPC), **agent** (scoped judgment), **decision**
(emit to decisions bus, resume on answer), **branch** (`when` → `goto`), **emit**
(send to a channel). Context flows via `{{trigger.*}}` and `{{steps.<id>.output}}`
substitution (no code execution in templates).

## Persistence — sqlite (`flows.db`, beside intentions.db)

```sql
CREATE TABLE flows (
  id TEXT PRIMARY KEY, name TEXT, description TEXT,
  enabled INTEGER DEFAULT 1, priority INTEGER DEFAULT 50,
  trigger_json TEXT, steps_json TEXT, guardrails_json TEXT, output_json TEXT,
  created_at TEXT, updated_at TEXT
);
CREATE TABLE flow_runs (
  id TEXT PRIMARY KEY, flow_id TEXT, status TEXT,         -- running|done|error|parked
  trigger_payload_json TEXT, started_at TEXT, ended_at TEXT,
  error TEXT
);
CREATE TABLE flow_run_steps (
  run_id TEXT, step_id TEXT, seq INTEGER, status TEXT,
  agent TEXT, input_json TEXT, output_text TEXT, ms INTEGER, started_at TEXT
);
```

Runs are first-class so the UI can show a live, step-by-step execution log.

## FD API (FastAPI, under `/fd/flows`)

```
GET    /fd/flows                 list (with enabled, last run)
POST   /fd/flows                 create
GET    /fd/flows/{id}            full spec
PUT    /fd/flows/{id}            update
DELETE /fd/flows/{id}
POST   /fd/flows/{id}/enable     {enabled: bool}
POST   /fd/flows/{id}/run        run now (manual trigger, optional sample payload)
POST   /fd/flows/{id}/test       dry-run with sample input, return per-step trace
GET    /fd/flows/{id}/runs       run history
GET    /fd/flows/runs/{run_id}   one run with step-by-step results (for live log)
```

## UI (flight-deck React) — day 0, nice UX

A **Flows** section (sidebar). Three views:
1. **List** — cards: name, trigger summary ("WhatsApp · has video"), enable toggle,
   last-run status, run-now. Search/filter.
2. **Builder** (form-based, not node canvas for v1):
   - Trigger: channel dropdown + match builder (rule chips: has image/video/audio,
     mime, from number, face label; or "classify with labels").
   - Steps: ordered list, add/reorder/remove; each step → type (tool/agent/decision/
     branch/emit) + a typed sub-form (tool picker + args, or prompt + allowed-tools,
     etc.) with `{{...}}` autocomplete from prior steps.
   - Guardrails + output selectors.
   - **"Test with sample"** button → calls `/test`, shows the per-step trace inline.
3. **Run log** — timeline of a run: each step with agent used, input, output,
   duration, status; live-updates while running (via the existing ws/event stream).

Form-first keeps it usable immediately; a visual canvas is a later enhancement once
the spec is proven.

## Trigger router (FD-side)

Hooks where FD already forwards inbound to agents (WhatsApp bridge forward fns,
glasses channel bus):
```
on inbound:
  payload = classify_payload(msg)        # cheap rules: has_*, mime, from_waid, face_label
  flow = router.match(payload)            # enabled flows by priority; rules-first,
                                          # opt-in LLM/embedding classifier only if asked
  if flow:  FlowRunner.run(flow, payload) # else → normal forward to agent (unchanged)
```
Cron triggers register with the cron worker; decision triggers resume from the bus.

## Worked examples

**A. Video auto-analysis** — the spec above (replaces the hand-coded `_has_video` +
auto-run + deny-policy in `chat_handler`).

**B. Social-memory whisper** (glasses photo of a person):
```yaml
id: social-memory
trigger: { on: message, channel: glasses, match: { kind: rule, rules: [has_image] } }
steps:
  - { id: who,  type: tool,  on: name:FlightDeck, tool: face_identify, args: { image: "{{trigger.image_path}}" } }
  - { id: gate, type: branch, when: "{{steps.who.label}} == none", goto: generic }
  - { id: hist, type: tool,  on: origin, tool: insights,   args: { query: "person:{{steps.who.label}}" } }
  - { id: open, type: tool,  on: origin, tool: intentions, args: { action: search, tags: ["{{steps.who.label}}"] } }
  - id: whisper
    type: agent
    on: capability:vision
    guardrails: { allow: [] }
    prompt: "One-line whisper: who, last decision, open intention.\nPerson:{{steps.who.label}}\nHistory:{{steps.hist.output}}\nOpen:{{steps.open.output}}"
  - { id: generic, type: agent, on: capability:vision, prompt: "Describe this image: {{trigger.image_path}}" }
output: { channel: glasses, format: whisper }
```
(`face_identify` runs FD-side — see open questions.)

## Build plan

1. **sqlite layer** — `flows.py` (store: flows + runs + steps), migrations.
2. **`/api/tool` endpoint** on the agent web server — `registry.execute(name, args)`
   directly, no LLM. The deterministic step primitive.
3. **FlowRunner** (FD) — step loop (`tool`/`agent`/`branch`/`emit`), templating,
   pool selection + affinity, guardrails, run logging. (`decision` in phase 2.)
4. **FD `/fd/flows` API** — CRUD + run + test + runs.
5. **React Flows UI** — list + builder + run log.
6. **Trigger router** — `classify_payload` + match, hook into bridge/glasses inbound.
7. **Port two flows** (video, social-memory) → delete hand-coded versions. Proven if
   both run green from the UI.

## Open questions

- **`/api/tool` auth/safety** — it executes tools without a turn; lock to FD-origin
  (loopback/token) and respect the same tool policy.
- **`face_identify` as a callable** — lives bridge-side today; either expose it as an
  FD-internal tool the runner calls directly, or wrap as a pseudo-agent in the pool.
- **File locality** — file-bound tool steps default to `on: origin`; FD ships files
  with `_transfer_file_to_agent` only when a step targets another agent.
- **Flow-calls-flow** composition — allow one level (like the workflow tool)?
