# Recursive Flows & Synthesis — Design

**Status:** Draft / pre-implementation
**Date:** 2026-06-06
**Scope:** Turns Flows from flat, message-triggered automations into a composition
language: flows call flows (`gosub`), spawn background workers (`spawn`/futures),
return values, handle errors, and can be **synthesized by agents** on the fly into
a separate scratch space that earns its way to permanence.

This document is the agreed spec from the design discussion. It captures *what* and
*why*, plus a sketch of *how it maps onto the current engine*. No code yet.

---

## 1. Motivation

The thesis: flows are the next layer of applications. Web/mobile apps are **nouns** —
places you navigate, where the UI *is* the product. Flows are **verb-first**: an
intent arrives on a channel (text/voice/image from phone, earbuds, glasses), agent(s)
do the crunching, and the result returns — by default on the same channel, but really
on *the right channel at the right time*. The interface collapses; the artifact is the
flow, not the screen.

Two properties make this more than "an agent does stuff":

1. **Deterministic spine, judgment at the leaves.** A flow is a deterministic program
   (trigger, branch, tool, emit, gosub) with agent judgment grafted only where needed
   (`agent`/`vision` steps). Not a black box; not rigid code. This duality is the
   differentiator and must be preserved as flows grow up.
2. **Flows are legible artifacts.** The DSL makes a flow readable, diffable, shareable,
   and forkable — an app you can distribute without an app store. Recursion turns that
   artifact into a *function*, and synthesis turns the agent into an author.

Design principle throughout: **the substrate guarantees correctness, not the author.**
Every flow — hand-written or model-synthesized — round-trips through the deterministic
validator. The agent's competence (or model tier) never determines a flow's safety.

---

## 2. Two spaces + the promotion gradient

Flows live in two spaces:

| Space | Origin | Lifetime | Trust |
|---|---|---|---|
| **Permanent** | human-authored (`origin: user`) | durable | vetted once by a human |
| **Scratch** | agent-synthesized (`origin: agent`) | TTL + GC | unvetted, tighter leash |

Synthesized flows are **call-only by default** (no message trigger — they exist to be
`gosub`'d), live in the scratch space, and never silently mix into the user's permanent
list. A flow moves between spaces along a one-way gradient:

```
ephemeral  →  candidate  →  proposed  →  promoted (permanent)
 (just ran)   (earned a    (intention   (human said yes;
              promotion     surfaced)     leaves scratch)
              score)
```

The promotion loop — "the agent crystallizes a repeated habit into a permanent app" —
is the actual product. Mechanics in §9.

---

## 3. Flows as functions

A flow becomes a callable unit. Four verbs:

### `gosub <flow> [with <args>]` — synchronous subroutine
- The caller **blocks**; the child runs to completion; its return value lands in the
  caller as `{{calls.<id>.output}}` (and `{{calls.<id>.status}}`).
- The child joins the caller's **stack** as a new frame (see §6).
- Killed when the parent's stack is stopped.
- Resolves a flow name across both spaces (see §8 for shadowing rules).

### `spawn <flow> [with <args>]` — asynchronous worker, returns a future
- Starts the child as its **own root run** (own stack, own control handle).
- Returns a **handle/future** immediately; the parent continues.
- **Not** killed by the parent's `flow stop` (it's independent), but is reachable by
  its own handle and by `flow stop all`.
- The parent may later **`join`** the future to collect the result.

### `synthesize <goal>` — author + run a throwaway flow (agent tool, §7)
- Natural-language goal → FD compiler → validator → run, stored in the scratch space.

### `save <flow>` — promote a scratch flow to permanent (§9)

### Arguments & the generalized trigger
A flow's "input" is just a payload — whether it came from a **message** (the trigger)
or from a **parent's `gosub`/`spawn` args**. Same machinery, two sources. Inside the
child, args are addressable the same way trigger fields are (`{{trigger.*}}` /
`{{args.*}}`). This unifies "triggered flow" and "called flow" — they differ only in
where the payload originates.

### `return [<expr>]` — the single exit verb
- Ends the current flow **now**, from anywhere (including inside a branch path), handing
  `<expr>` up to the caller as the flow's output.
- `return` with no expression = end with empty/last output. The legacy bare `stop`
  flag becomes **sugar for `return` with no value**.
- For a **root** flow, the return value is what reaches the output channel; for a
  `gosub`'d child it goes to the parent; for a `spawn`'d child it resolves the future.

### Output default flips for sub-flows
- A **root** flow: `output -> same` delivers to the user (today's behavior).
- A **gosub/spawn child**: defaults to `output -> return` — the value goes to the
  caller, **not** the user's channel. Only the root reaches the user unless a child
  *explicitly* emits. (Without this, every nested call spams the user — this is the
  single most important sub-flow semantic.)

---

## 4. Error handling

A stopped or failed sub-flow is an **exception** that propagates up the stack and
unwinds it — **unless a caller catches it.** Catching is explicit via a dedicated
handler:

- An **`error` / `handler` step** (or a `try`-scoped block) that runs when a `gosub`'d
  child returns `status in (error, stopped, timeout)`.
- Caught → the parent treats it as a clean return into the handler path and continues.
- Uncaught → bubbles to the next frame; at the root it ends the run with that status.

The caller can also branch on the child's status without a dedicated handler:
`if {{calls.search.status}} == "stopped" -> ...`. The `error` step is the structured
form; the branch is the inline form.

Joins (§5) participate: a stopped/failed **spawn** resolves its future with that status,
so the awaiting parent's handler can catch it. **Stop must propagate into outstanding
joins** — a parent must never hang forever on a future that will never resolve.

---

## 5. Futures & `join` (spawn)

`spawn` returns a **future handle**. The parent may:

- ignore it (fire-and-forget worker),
- `join <handle>` later to block until it completes and read `{{joins.<handle>.output}}`
  / `.status`,
- `join` with a timeout (the join itself can error → caught by an `error` step).

Futures are first-class from day one (decided), because background workers that can't
feed results back lose half their value. A future's lifecycle:

```
pending → fulfilled(value) | rejected(error|stopped|timeout)
```

A `spawn`'d run is an independent root: it has its own control handle, shows in
`flow status`, and is independently stoppable. Stopping it (directly, or via
`flow stop all`) **rejects** any pending join on it.

---

## 6. Runtime model: frames, stacks, roots

Today a "run" = one flow execution with a `run_id`. With recursion a **logical run is a
tree of frames**:

- A **root run** is started by a message trigger or by `spawn`. It owns a `run_id`, a
  human handle (§7.x), and a **frame stack**.
- A **frame** = one flow invocation: `{frame_id, depth, flow_id, flow_name, active_step,
  status}`. `gosub` pushes a frame; `return`/end pops it.
- A `spawn`'d flow is **not** a frame of the spawner — it's a new root with its own stack.

### Guards (all on the **root**, not per-flow)
- **Depth cap** — the real infinite-recursion guard (`A→B→A→B`). Bound depth; allow
  recursion under the cap.
- **Shared step/token budget** on the root, decremented across *all* frames — so
  recursion can't multiply per-flow `max_steps`.
- **Per-frame timeout** and a **whole-stack timeout**.
- Cycle detection is optional; a depth cap alone is sufficient for v1.

### Control attaches to the root
The existing pause/resume/stop control registry moves up one level: it keys on the
**root run** and carries the frame stack. Pause pauses the **active (top) frame** →
the whole stack is effectively paused. Stop semantics in §7.

---

## 7. Control & addressability

### Handles
Every **root run** gets a short, stable, human-memorable **handle** at start (e.g. a
2-char tag or a slug of the flow name), and **stamps it on every message it sends**:

> ⏳ *Hungry Search* `[hs]` needs your input:

The handle is stable for the run's life. Benefit beyond control: the user always knows
*which* flow is talking when several run at once.

### Stop scopes
- **`flow stop`** → the **most-recently-touched** stack; stops its **top frame**, which
  **bubbles** up the stack per the exception model (§4) unless caught.
- **`flow stop <handle>`** / **`flow stop <name>`** → target a specific root (fuzzy name
  match → if ambiguous, list and ask).
- **`flow stop all`** → every stack the user has running.
- **One stop message**, delivered from the **root**, naming what was cancelled — never N
  messages from N frames, never silent.

### Pause / resume / status grammar
Same targeting: `flow pause [handle]`, `flow resume [handle]`. `flow status` enumerates
running stacks **with their handles** and shows each as a breadcrumb:

```
📊 Flow status
• Hungry Search [hs]  —  hs › Geocode › Web lookup  ⏳ waiting for your input
• Morning Brief [mb]  —  ▶️ running
```

### What's addressable
Only **roots** (interactive-triggered + spawned). A `gosub`'d child is a *frame*, not a
stack — never independently stoppable except as "top of stack" via the bubble. This is
the `gosub`/`spawn` distinction viewed from the control side.

### Pause + input (already shipped, preserved)
Pausing a flow that's waiting on `input` must **not** let the next message be consumed as
the answer — it goes to the agent as normal chat; the flow stays on the step; `resume`
re-shows the question. This rule extends naturally to the active frame of a stack.

---

## 8. Synthesis as a tool (all modes)

Agents get flow synthesis as a tool, available in **standard / eco / nano**.

### When to synthesize (gate it hard)
Only when the work is **repeatable**, **durable** (spans time / waits / survives
restarts), **auditable**, or **handoff-able**. For one-shot reasoning the tool must
**refuse** and just answer — otherwise weak agents wrap single responses in flows and
burn cost.

### Mode strategy — decouple *what* from *how*
The agent's mode must **not** determine flow quality.
- **Retrieve before generate.** In nano/eco the tool first searches the scratch +
  permanent spaces for an existing flow to **parameterize**; only standard authors new
  DSL. Retrieval is something small models do well; freeform DSL authoring isn't. This
  is also what makes GC dedup work (§9).
- **The compiler is a separate, escalatable step.** The agent emits *intent* in natural
  language; `/fd/flows/compile` round-trips it through a capable model **plus the
  deterministic validator with auto-repair**. Correctness is guaranteed by the
  substrate, independent of who asked.

### The flow is the cost/escalation boundary
A synthesized flow's steps carry their own selectors (`capability:vision`, `name:Big`).
So a nano agent can orchestrate a flow whose heavy steps run on capable agents — cheap
orchestrator, expensive workers, the flow as the contract between them. This is a prime
reason to want synthesis in nano at all.

### Synthesis triggers (both)
- **Reactive** — the agent decides mid-task (tool call), grounded in its memory /
  experience / current task.
- **Proactive** — a reflection / janitor pass notices a repeated pattern and *proposes*
  synthesizing or promoting (an intentions feature, see §9).

---

## 9. GC & promotion mechanics

### Dedup at synthesis time (the keystone)
Before storing a synthesized flow: normalize to canonical DSL, hash it, **match against
the scratch space**. If an equivalent exists, increment **its** counter instead of
creating a near-twin. Without this, slight re-wordings each get use-count 1 and nothing
ever promotes.

### What counts as a "use"
A **successful, distinct-context** run:
- failed runs do **not** count toward promotion (they count toward quarantine),
- the same flow fired 50× inside one recursive loop is **one** use (dedupe by an
  intent/time window), not 50.

### Promotion is a score, not a raw count
`score = f(distinct uses, success rate, recency, non-reversal)`. The cheapest strong
signal in an ambient world is **non-reversal** — the user didn't immediately undo,
correct, or contradict the output. Frequency is only the proxy.

### Tiered TTL + caps
- *ephemeral* → short TTL since last use; GC'd quietly.
- *candidate* (crossed the score) → long TTL; survives quiet periods.
- *proposed* → pinned until the user decides.
- *promoted* → leaves scratch entirely.
- Hard **LRU cap per owner** regardless of TTL.

### Failure → quarantine (with negative memory)
A flow that errors repeatedly is archived **and remembered as a negative**, so the agent
doesn't re-synthesize the same broken pattern. A memory write, not just a GC.

### Promotion = reviewable diff + one decision
Synthesized flows are usually **call-only**. Promotion isn't a blind copy: the user sees
the (legible) DSL, renames it, and chooses what it becomes —
- a **library function** (stays call-only), or
- a **user-facing automation** (gets a real message trigger).

Surfaced via the existing intentions approve/announce/undo path. Both reactive (post-run
threshold check) and proactive (janitor sweep) raise the proposal.

---

## 10. Trust & security

- **Provenance** on every flow (`origin: user|agent`), like intentions.
- **Namespaced resolution**, no silent shadowing: a scratch flow may **not** shadow a
  permanent one. Permanent wins, or require explicit `user:` / `scratch:` qualifiers.
- **Authority = transitive closure.** A synthesized flow's real capability is everything
  it can reach via `gosub`/`spawn`. Tool-palette restrictions must be **transitive**.
- **No borrowing vetted authority.** A synthesized flow may **not** `gosub` a permanent
  *world-acting* flow without approval — **no-by-default, yes-after-promotion**.
  Read-only synthesized flows run freely; world-acting ones route through the intentions
  approve path and/or dry-run first.

---

## 11. DSL surface (sketch)

New keywords layered onto the existing line-oriented DSL:

```text
flow "Plan dinner"
trigger any when contains "dinner"

step where:
  input
  prompt: "Where are you?"

step geo:
  gosub Geocode with place: {{steps.where.output}}      # sync subroutine

step worker:
  spawn DeepSearch with q: "ćevapi near {{geo.output}}" # async → future 'worker'

step wait:
  join worker timeout: 30                                # collect the future

step check:
  branch
  if {{calls.geo.status}} == "error" -> oops
  else -> done

step oops:
  error                                                  # handler frame
  emit "Couldn't locate you — try again later."
  return                                                 # exit, no value

step done:
  return {{joins.worker.output}}                         # exit with value

output -> same                                           # root → user
```

- `gosub <Flow> [with k: v, ...]` → `{{calls.<step_id>.output|status}}`
- `spawn <Flow> [with ...]` → future named by step id → `{{joins.<id>.output|status}}`
- `join <future> [timeout: n]`
- `return [<expr>]` — exit-from-anywhere; supersedes bare `stop`
- `error` / handler step — catch a failed/stopped child
- `output -> return` — default for sub-flows; `-> same/whatsapp/web/glasses/log` for roots

Round-trip (compile/decompile/validate) and the AI compiler prompt all extend to cover
these. Validation adds: depth/recursion sanity, future-defined-before-join, handler
placement, and cross-space resolution checks.

---

## 12. Mapping to the current engine (what changes)

| Area | Today | Change |
|---|---|---|
| `flow_runner.py` | flat loop, one run | frame stack; `gosub`/`spawn`/`join`/`return`/`error`; root-level budget/depth |
| control registry (`_RUN_CONTROL`) | keyed by `run_id` | keyed by **root**; carries frame stack; handles; stop-scope logic |
| `flow_router.py` | `/flow stop|pause|resume|status` | add handle targeting, `stop all`, stack-aware status, one-message rule |
| `flows_store.py` | one flow table + runs | **scratch space** + provenance + use/score/TTL columns; frame/stack run records |
| `flow_dsl.py` | tool/agent/vision/input/emit/branch | add `gosub`/`spawn`/`join`/`return`/`error`; args + return; `output -> return` |
| `server.py` (FD) | flow CRUD + compile | synthesize endpoint hardening, scratch-space CRUD, promotion endpoint, GC/janitor |
| compiler `/fd/flows/compile` | NL → flow | retrieve-before-generate; emits call-only scratch flows; transitive palette guard |
| UI | builder + run log | scratch list (provenance/TTL), promotion review (call-only vs trigger), stack breadcrumb in run log, handles |
| intentions | approve/announce/undo | reused as the promotion + world-acting-approval surface |

---

## 13. Phasing

1. **Composition core** ✅ *shipped* — `gosub` (sync), args, `return` (+ branch-exit),
   `output -> return`, frame stack, root budget/depth guards, stack-aware `flow status`
   + stop bubbling.
2. **Async** ✅ *shipped* — `spawn`, futures, `join` (timeout + stop-aborts-join), the
   `error`/handler step + `on error -> <step>` routing, branchable call/join status.
3. **Addressability** ✅ *shipped* — per-run handles (stamped on input prompts +
   `flow status`), `flow stop/pause/resume <handle|name|all>`, bare = most-recent,
   multi-stack status.
4. **Synthesis** — the tool (gated), retrieve-before-generate, scratch space, compiler
   hardening, transitive palette guard, world-acting approval.
5. **Lifecycle** — dedup/hash, use-scoring, tiered TTL/GC, quarantine, promotion review,
   reactive + proactive (janitor) triggers.

Each phase is independently shippable and useful.

---

## 14. Open questions (carry into implementation)

- **Handle format** — derived slug vs random 2-char tag vs user-renamable. Stability vs
  memorability.
- **`with` arg syntax** — inline `k: v` list vs a body block; how templated args render.
- **Frame persistence across restart** — stacks are in-memory today; do paused stacks
  survive an FD restart (needs serialized frames) or drop (current input behavior)?
- **Non-reversal signal** — how to detect "user didn't correct it" cheaply per channel.
- **Scratch dedup canonicalization** — exact-hash (simple, misses near-variants) vs
  fuzzy clustering (promotes a representative; more complex). Start exact.
- **Cross-owner sharing** — can a promoted flow be shared/forked to another user? (Future;
  the DSL already makes it portable.)
- **Budget accounting across `spawn`** — does a spawned root draw from the spawner's
  budget or get its own? (Leaning: its own, since it's an independent root.)

---

## 15. Phase 1 — Implementation Plan (Composition Core)

**Goal:** human-authored flows can call other flows synchronously, pass args, return
values, and exit early — with a real frame stack the existing control layer
(pause/resume/stop/status) understands. **No** `spawn`/futures, synthesis, scratch
space, GC, or handles beyond what status needs. Everything additive; existing flat
flows behave identically.

### 15.0 Architectural decision — recursion *is* the stack

Synchronous `gosub` maps exactly onto Python's call stack. So:

- Execution uses **recursion**: a new internal `_run_frame()` runs one flow's step loop;
  a `gosub` step resolves the target and **recursively calls `_run_frame()`** for the
  child, capturing its return into the parent's ctx.
- The **explicit frame stack** lives on the root's `_RunControl` purely as a *mirror*
  for introspection/control: push a frame descriptor before recursing, pop after. Status
  reads it; stop sets a flag every frame checks.
- A stop unwinds via a sentinel exception (`_FlowStopped`) caught at the root — natural
  Python-stack unwinding, no manual frame popping for the abort path.

`run()` splits into two:

- **`run(flow, payload, *, dry, run_id)`** — *root concerns*: build root ctx, register
  root control, allocate run_id, call `_run_frame(depth=0)`, deliver final output to the
  user channel, `finish_run`, cleanup control. (Unchanged signature.)
- **`_run_frame(flow, payload, *, root, depth, call_id) -> FrameResult`** — the current
  `while` step loop, returning `{value, status}`. Builds its own ctx scope; shares
  root-level state (budget, control, trace) via the `root` handle.

`FrameResult = {value: str, status: "done"|"stopped"|"error"|"returned"}`.

### 15.1 Data / ctx shapes

- **Per-frame ctx**: `{trigger, args, steps, calls, system}` where
  - `args` = the `with` arguments passed by the caller (root frame: empty; child frame:
    the gosub args). Addressable as `{{args.<k>}}`.
  - `trigger` = the original message payload for the root; for a child it mirrors `args`
    so `{{trigger.*}}` keeps working inside reused flows.
  - `calls.<step_id>` = `{output, status}` of a `gosub`'d child (already-flat-field
    access via the existing `_maybe_attach_fields` convention).
- **Root handle** (passed down, not in templated ctx):
  `{run_id, control, budget, trace, store, depth_cap}` where `budget` is a mutable
  counter `{steps_left}` decremented across **all** frames.

### 15.2 `flow_dsl.py` — grammar additions

Add to `VALID_TYPES`: `gosub`, `return`.

- **`gosub` step**
  - Header: `gosub <FlowName>` (FlowName may be quoted if it has spaces).
  - Args via repeated `with` lines or inline: `with <k>: <value>` (templated values
    allowed). Parse into `step["args"]`; target into `step["flow"]`.
  - Example:
    ```text
    step geo:
      gosub "Geocode"
      with place: {{steps.where.output}}
    ```
  - Decompile mirrors it; `_step_to_dsl` emits `gosub "<flow>"` + `with k: v` lines.
- **`return` — dual form**
  1. **Step type:** `step done:` / `  return {{expr}}` → `{type: "return", value: "<expr>"}`.
  2. **Trailing directive** on any step (supersedes `stop`): a `return` or
     `return <expr>` body line sets `step["return"] = "<expr>"` (empty string = no value).
     Keep parsing the legacy bare `stop` line as `step["return"] = ""` (alias).
- **Branch exit:** keep `-> stop` (`__stop__`, no value). Add `-> return` as an alias for
  the same no-value exit. Value-returns from a branch path are done by jumping to a
  `return`-type step (already supported — any step id).
- **`output -> return`**: add `return` to `OUTPUT_CHANNELS`. Meaning: hand the flow's
  result to the caller instead of a channel. For a **root** flow, `-> return` is treated
  as `-> log` (nowhere to return to) — validator warns, doesn't error.
- **`validate_flow` additions:**
  - `gosub` needs a non-empty `flow` name. (Existence is resolved at **runtime** — the
    validator has no cross-flow view; a missing target becomes a frame error.)
  - `return` step: `value` optional; fine empty.
  - No new branch-target checks beyond existing (`__stop__`/ids).

### 15.3 `flow_runner.py` — execution

- **Split `run()` / `_run_frame()`** per §15.0.
- **Step dispatch** in `_run_frame`: add
  - `gosub`: render args → build child payload (`args` + carry `waid/channel/origin_*`
    so delivery/identity still resolve) → resolve target flow by name (§15.4) →
    `depth+1 > depth_cap` ⇒ raise frame error → recurse `_run_frame(child, ..., depth+1,
    call_id=sid)` → store `{output, status}` in `ctx["calls"][sid]`. Output of the gosub
    *step itself* = child's return value (so `{{steps.geo.output}}` also works).
  - `return` (type): render `value`, set frame result `{value, status:"returned"}`, end
    the frame loop.
- **`return` directive** (any step): after executing a step, if `"return" in step`,
  render it, set frame result value, end the frame.
- **Output flip:** in `run()` (root only) deliver final value to the user channel when
  `output.channel in (whatsapp/same/web/glasses)`. In `_run_frame` for `depth>0`, never
  deliver to a user channel — the value just returns. (`emit` steps still deliver
  explicitly at any depth — that's the "child explicitly emits" escape.)
- **Root budget:** decrement `root.budget.steps_left` per executed step across all
  frames; exhausted ⇒ raise (mapped to `error`). Replaces per-flow `max_steps` as the
  global guard (keep per-flow `max_steps` as a local sanity cap).
- **Depth cap:** `root.depth_cap` (default e.g. 8); exceeded on `gosub` ⇒ frame error
  `"max flow recursion depth (N) exceeded"`.
- **Frame stack mirror & stop bubbling:**
  - On entering `_run_frame`: `control.push_frame({depth, flow_name, active_step})`;
    on exit: `control.pop_frame()`. Update `active_step` as the loop advances.
  - The existing per-loop control check stays; when `control.stopped`, raise
    `_FlowStopped` → propagates through the recursion to `run()`, which sets status
    `stopped`, delivers the **one** stop message from the root, and cleans up.
  - Pause check unchanged (active/top frame parks on the resume event).

### 15.4 Name resolution

- Add `FlowStore.get_flow_by_name(name) -> flow | None` (case-insensitive exact match;
  Phase 1: permanent space only). Cache per-run to avoid repeated lookups.
- Resolution failure ⇒ frame error `"gosub: no flow named '<x>'"` (catchable later by the
  §4 handler in Phase 2; for now it fails the run cleanly).

### 15.5 `flows_store.py` — run records

- Extend `add_step_result(...)` with `depth: int = 0` and `frame: str = ""`
  (the call path, e.g. `root/geo`). Add the two columns to `flow_run_steps`
  (nullable/defaulted — backward compatible).
- No new tables in Phase 1. The run log nests by `depth`.

### 15.6 `flow_router.py` — status & stop (stack-aware)

- `owner_run_states` / `flow status`: read the control's frame stack and render a
  breadcrumb per root (`hs › geo › web ⏳`). Single root per owner still the common case.
- Stop: `request_stop` already sets `control.stopped`; ensure every recursive frame's
  loop observes it (it does, via the shared `control`). The `_FlowStopped` unwind +
  single root message already covered in §15.3.
- No handles/`stop all`/multi-root targeting in Phase 1 (that's Phase 3).

### 15.7 UI (flight-deck) — additive

- `flowsApi.ts`: `StepType |= 'gosub' | 'return'`; `FlowStep += { flow?, value? }`
  (reuse `args` for `with`). `OutputChannel` add `'return'`.
- `FlowBuilder.tsx`: `STEP_TYPE_META` + `newStep` for `gosub` (target = dropdown of
  existing flow names + `with` arg rows) and `return` (value template field). DSL code
  view round-trips automatically once `flow_dsl` supports them.
- `FlowRunLog.tsx`: indent steps by `depth`; show the frame/flow name on nested steps.
- UI can **lag** the engine — engine + DSL + tests first, builder editors second.

### 15.8 Test matrix (before any UI)

Deterministic, runner-level (mirrors the existing pytest-style smoke tests):

1. **Plain gosub + return value** — A `gosub`s B; B `return {{args.x}}`; A reads
   `{{calls.b.output}}`. Assert value threads through.
2. **`return` exits early from a branch** — branch jumps to a `return` step; later steps
   don't run; root delivers the returned value.
3. **`return` directive supersedes `stop`** — a step with a `return "bye"` line ends the
   flow with "bye"; legacy bare `stop` still ends with last output.
4. **Output flip** — a `gosub`'d child with `output -> same` does **not** message the
   user; only the root's output is delivered. A child `emit` *does* deliver.
5. **Depth cap** — A `gosub`s A (self-recursion) → fails at the cap with a clear error,
   no hang, run status `error`.
6. **Root budget** — a deep/wide call tree exhausts the shared budget → clean `error`,
   not a per-flow reset.
7. **Stop bubbles** — stop while 3 frames deep → whole stack unwinds, status `stopped`,
   exactly one stop message from the root.
8. **Status breadcrumb** — mid-run `flow status` shows the nested frame path.
9. **Pause/resume across a gosub** — pause while the child is on an `input` step; the
   "don't consume next message as input while paused" rule still holds; resume continues
   the child; parent then completes.
10. **DSL round-trip** — compile → decompile → compile is stable for a flow using
    `gosub`/`with`/`return`/`output -> return`.

### 15.9 Risks / watch-items

- **Identity/delivery in child frames** — children must carry `waid/channel/origin_*`
  from the root payload so `_deliver` and the owner key still resolve (esp. for `input`
  steps and `emit` inside a child). Thread the root payload's identity into child
  payloads.
- **`input` keying under recursion** — input parking is keyed by owner (waid/channel),
  not by frame. Two concurrent `input` waits in one stack can't happen in Phase 1 (sync,
  single active frame), so the existing key is safe — but assert it in test 9.
- **ctx bleed** — each frame must get a *fresh* `steps`/`calls` namespace; only
  root-level state (budget, control, trace) is shared. Easy to get wrong; cover in tests
  1 & 4.
- **Backward compatibility** — flat flows (no gosub/return) must produce byte-identical
  behavior. The new code paths only trigger on the new step types; `add_step_result`
  gains defaulted params. Keep a regression test on an existing flow.

### 15.10 Definition of done (Phase 1)

- `gosub` (sync, args, return value), `return` (step + directive + branch alias),
  `output -> return`, frame stack, depth cap, root budget, stack-aware `flow status`,
  and stop-bubbling all work and are covered by the §15.8 matrix.
- DSL compile/decompile/validate + the AI-compiler prompt updated for the new keywords.
- Existing flat flows unchanged (regression test green).
- UI: at minimum the DSL code view authors/edits these (the builder editors may follow).

