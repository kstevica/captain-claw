# Vatra — Collaborative Flight Deck Mode — Implementation Plan

> Status: **All 4 phases built** (P1: Lead decompose → parallel subtasks → reporter assemble.
> P2: blackboard ask protocol — non-blocking delegation via a concurrent coordinator. P3:
> learning loop — score owners/answerers/lead/reporter into reliability. P4: Flight Deck UI —
> Vatra badge, team-plan header, and the delegation blackboard view). Prompt-tuned post-test so
> delegation actually fires: the Lead now KEEPS interdependent pieces separate (records
> `depends_on`) instead of folding them, never emits an integration/polish subtask (that's the
> reporter), and owners get the full team roster + a proactive nudge to ask teammates for
> cross-slice needs. A **Lead-run review round** follows the first pass: the Lead gathers an
> exec-summary digest of every piece and each owner (kept alive) revises its own piece against
> the whole team's work before the reporter assembles (config `review_round`, default on) — this
> is where real collaboration happens, since round-1 owners are blind to each other. The Lead is
> an LLM **coordinator** (decompose → gather digest → orchestrate review), not a spawned worker
> agent. Name **Vatra** (a hearth — agents gather round one fire and build together, not solo sorties).
>
> Phase 1 deviations from the plan below (deliberate, for a low-risk first cut):
> - **`mode` rides in the session `config` JSON** (`config.mode="vatra"`), not a new column —
>   no DB migration for v1.
> - **Spawn/teardown are mirrored locally** in `vatra_routes.py` (stamped `CLAW_VATRA_WORKER`)
>   rather than sharing Basna's `_spawn` — keeps Basna's working path byte-for-byte untouched.
>   Unifying into one shared helper is a fast-follow once Phase 1 validates.
> - **Learning loop deferred to Phase 3** as planned — Phase 1 persists runs with `success=NULL`
>   and does not score reliability.
> - Files: `captain_claw/flight_deck/vatra_routes.py`, `instructions/vatra/{lead,reporter}.md`;
>   wired via `server.py` (router) and `tools/basna.py` (`start` gains a `mode` param).
>
> Decisions locked: **hub-and-spoke / blackboard** topology · **a dedicated reporter
> archetype** writes the final artifact (Lead decides + routes, reporter assembles) · the
> **user/flag picks** Basna-vs-Vatra in v1 (router may auto-pick in Phase 4).

## Thesis

Basna is an **independent ensemble**: spawn fresh agents that never see each other, then
**merge** their uncorrelated outputs weighted by learned reliability. That independence is
the whole product — `agreement`/`blind_spots` analysis and per-domain weights are only
meaningful because errors are uncorrelated.

Vatra is the opposite contract: **a team that collaborates.** A Lead decomposes the task,
specialists work in parallel on a shared blackboard, and they **delegate sub-asks to each
other without blocking** — post the ask, keep working, reintegrate the answer when it lands.
There is no merge; a dedicated **reporter assembles** one coherent artifact from the blackboard.

This is *not* a flag on Basna. Collaboration deletes the two properties Basna is built on
(independence → uncorrelated error; merge → reliability weighting). So Vatra is a **sibling
mode that reuses the spine and forks only the coordinator.**

```
                    ┌──────────── ROUTER (shared, +mode axis) ────────────┐
                    │ domain · difficulty · independent│collaborative      │
                    └───────────────────────┬──────────────────────────────┘
                                            │ collaborative → Vatra
                                            ▼
                              ┌───────────────────────────┐
                              │   LEAD (new)               │  decompose task →
                              │   decompose · assign ·     │  N subtasks, name owners
                              │   route asks · call done   │
                              └─────────────┬──────────────┘
                                            │ spawn (shared _spawn, fresh)
                 ┌──────────────┬───────────┴───────┬──────────────┐
                 ▼              ▼                    ▼              ▼
            specialist A   specialist B         specialist C   (idle pool)
                 │              │                    │
                 │   ask:"need X" │                  │
                 ▼              ▼                    ▼
        ┌──────────────────── BLACKBOARD (session) ────────────────────┐
        │ posts · asks (open/claimed/answered) · artifacts · budget    │
        └───────────────────────────┬──────────────────────────────────┘
                                     │ Lead routes each ask (reliability hint)
                                     │ → idle specialist OR spawn helper
                                     ▼
                       answer posted → asker reintegrates (non-blocking)
                                     │
                                     ▼  when budget spent / no open asks / Lead says done
                              ┌───────────────┐
                              │ REPORTER (new)│ → one artifact + contribution log
                              │ assembles bb  │   (spawned fresh; reads blackboard)
                              └───────────────┘
                                     │
                                     ▼ (shared) judge → archetype_reliability
```

## Why hub-and-spoke, not peer-to-peer

True peer mailboxes (A messages B directly) is the seductive design and it's where this
goes wrong: cycles (A→B→A), deadlock even when "non-blocking" (A's task can't finish
without B's answer and vice versa), and unbounded cost. **Hub-and-spoke via a blackboard**
keeps every cross-agent interaction observable and routable:

- Specialists never call each other. They **post an ask to the blackboard**; the **Lead**
  decides who answers (an idle specialist, or a freshly spawned helper) using the same
  reliability hints the router uses.
- "Non-blocking" = the asker writes the ask, continues the rest of its subtask, and polls
  the blackboard for the answer at natural checkpoints. No agent ever hard-blocks on another.
- The blackboard *is* the Basna session row, extended — so persistence, files, and the
  agent-side `basna` read tool come for free.

## Seams (what this hangs off)

- **Router** — `route_intent` + `instructions/basna/router.md`. Add one classification axis
  (`coordination: independent|collaborative`) and a `lead` field to the route. Catalog,
  reliability hints, difficulty scaling all reused verbatim.
- **Spawn** — `_spawn` in `basna_routes.py:1641`. Reused as-is (fresh, tool-stripped,
  `CLAW_BASNA_WORKER`-marked). Vatra adds a sibling marker `CLAW_VATRA_WORKER` so a Vatra
  worker can post asks but still cannot start a *new* run (recursion stays banned).
- **Dispatch** — `_dispatch_one` / `_send_chat_and_collect` (`basna_routes.py:1376,1549`).
  Reused for the initial subtask dispatch *and* for answering an ask. The difference is the
  prompt and that dispatch results can re-enter the loop instead of going straight to merge.
- **Persistence / files / read-tool** — `db.update_basna_session`, `_session_files_dir`,
  the `basna` tool (`tools/basna.py`). Reused; Vatra rows are the same table with `mode='vatra'`.
- **Judge + learning** — `_score_runs` / `_llm_judge` / `adjust_archetype_reliability`
  (`basna_routes.py:1303,1332`, `db.adjust_archetype_reliability`). Reused, but scored per
  **subtask owner** + the Lead, not per independent answerer (see Learning below).
- **Notify / callback** — `_notify_source_agent` (fire-and-forget + WS callback). Reused
  unchanged; Vatra reports back to origin exactly like Basna's `agent_start`.

## What is genuinely new (the fork)

Only the **coordinator** is new. Everything above is shared.

### 1. Lead loop (`vatra_routes.py:run_vatra`)
Replaces `_aggregate`. Pseudocode:

```
plan = lead_decompose(intent)          # → [{subtask, owner_archetype, depends_on}]
spawn(owners)                          # shared _spawn
dispatch_all(subtasks)                 # shared _dispatch, non-blocking, results stream to blackboard
while not done():
    for ask in blackboard.open_asks():
        if budget.exhausted(): break
        answerer = lead_route_ask(ask)     # idle owner OR spawn helper (reliability hint)
        dispatch_ask(answerer, ask)        # answer → blackboard; asker reintegrates
    done = (no open asks AND all subtasks reported) OR budget.exhausted() OR lead_calls_done()
truth = spawn_reporter_and_assemble(blackboard)   # dedicated reporter, NOT a merge, NOT the Lead
```

The Lead **decides and routes**; it never writes the deliverable. When the loop ends it spawns
a fresh **reporter** archetype whose whole job is to assemble the blackboard (artifacts +
answered asks) into one coherent document. Separating "decide" from "write" keeps the Lead's
context lean during the loop and lets the reporter be a writing-specialist tier.

### 2. Ask protocol (blackboard schema)
An `ask` is a row: `{id, from_owner, subtask_id, text, status(open|claimed|answered|dropped),
answer, depth, created_at}`. The asker's prompt gets: "If you need something outside your
slice, **post an ask** (tool: `vatra_ask`) and keep working — do not wait. Check
`vatra_inbox` for answers at each checkpoint." A new lightweight agent tool `vatra` (sibling
to `basna`) exposes `ask` / `inbox` / `post_artifact` against the blackboard.

### 3. Termination budget — the real risk
Basna ends at the merge barrier; a delegation graph has no natural end. Hard bounds, all
config-driven, all enforced in the Lead loop:
- `max_asks` total across the run (global counter, like the agent cap).
- `max_ask_depth` — an answer that itself spawns an ask increments depth; cap kills cascades.
- cycle guard — `(from_owner, normalized_ask_text)` seen-set; a repeat ask is auto-dropped.
- `wall_clock_s` / `max_tokens` — whole-run ceiling; on hit, Lead synthesizes with what exists.
- no-progress guard — N consecutive loop turns with no new artifact AND no answered ask → done.

### 4. Reporter assembly, not merge
When the Lead calls done, spawn a fresh **reporter** archetype (shared `_spawn`) that reads
the blackboard (artifacts + answered asks) and writes one document. Reuse `_llm_synthesize`'s
plumbing/creds but a different prompt and a real agent: "assemble these interdependent parts
into one coherent deliverable," not "reconcile independent answers." The reporter is a normal
catalog archetype (a writer/editor), so the router can pick a domain-appropriate one; if none
fits, fall back to a generic `vatra-reporter`.

## Parametrisation — `VatraConfig` (captain_claw/config.py)
Pydantic block beside the Basna config; read fresh per run; env `CLAW_VATRA__*`.

Fields: `enabled`, `max_agents` (reuse 1–10), `max_helpers` (spawned mid-run, default 3),
`max_asks` (default 12), `max_ask_depth` (default 2), `wall_clock_s` (default 600),
`max_tokens`, `no_progress_turns` (default 3), `lead_tier` (default `reason`),
`reporter_tier` (default `reason`), `reporter_archetype` (override; default = router pick or
`vatra-reporter`), `auto_mode` (let router pick independent vs collaborative, default false →
user/flag chooses).

## Data (reuse the Basna tables)
- **`basna_sessions`** — add `mode TEXT DEFAULT 'basna'` (`'basna'|'vatra'`) and a nullable
  `lead_archetype_id`. Vatra reuses `truth`, `confidence`, `files`, `analysis`.
- **`basna_asks`** (new) — `id, session_id, from_owner, subtask_id, text, status, answer,
  depth, created_at, answered_by`. The blackboard's ask ledger; also what the UI renders.
- **`archetype_reliability`** — unchanged table. Vatra writes outcomes keyed by the same
  `(user_id, archetype_id, domain)`; the Lead gets its own pseudo-archetype id `vatra-lead`
  so its decomposition/synthesis quality is learned separately.

## Learning — what "good" means here
Basna scores each agent against the merged truth (`_llm_judge`). Vatra can't — contributions
are interdependent, so "did A agree with the truth" is meaningless. Instead score three things:
- **subtask owners** — did the final artifact use this owner's slice, and was it sound? (judge
  per-slice against the synthesized whole).
- **ask answerers** — was the answer used by the asker? (the asker's reintegration is the signal).
- **the Lead** (`vatra-lead`) — did the decomposition hold up: were slices well-scoped, asks
  well-routed, and the run terminated cleanly (not budget-killed)?
- **the reporter** (`vatra-reporter` or the chosen writer archetype) — coherence/completeness
  of the assembled artifact, judged holistically.

All flow through `adjust_archetype_reliability`, so the next route's `prior_weight` improves —
same machinery, different scoring inputs. Lead and reporter are scored as separate
pseudo-archetypes so "good at planning" and "good at writing it up" learn independently.

## When the router picks Vatra vs Basna
The router already classifies `merge_kind: converge|diverge`. Add the orthogonal axis:
- **independent (Basna)** → *truth-finding*: "what's true," "what are the options," verify a
  claim. Collaboration would only inject groupthink.
- **collaborative (Vatra)** → *build a multi-part artifact whose pieces depend on each other*:
  one agent's output is another's input and you want a coherent whole, not N takes to merge.
Heuristic for the prompt: choose collaborative when subtasks have **dependencies**
(A's output feeds B); independent when subtasks are **parallel samples** of the same question.

## Frontend (flight-deck) — ✅ Built (Phase 4)
- Vatra sessions share the Basna list/detail (same `basna_sessions` table). The session row
  shows a **`vatra` chip** (`isVatra(config)`); the detail header reads **"Team plan ·
  collaborative · N owner(s)"** instead of difficulty/merge_kind.
- New self-contained `components/VatraDelegation.tsx`: the **decomposition** (owner · piece
  chips) + the **delegation blackboard** — each ask as `from_owner → answered_by` with status
  icon, expandable to show the ask text and the answer; live-polls `/fd/vatra/sessions/{id}/asks`
  (new user-scoped read endpoint) while the run is active. Summary counts (answered/pending/dropped).
- `basnaStore` extended: `VatraSubtask`/`VatraAsk` types, `RoutePlan.mode`/`.subtasks`,
  `apiListVatraAsks`. Built bundle committed (vite → `flight_deck/static`).
- `basna` tool `start` gained a `mode` param (Phase 1); `vatra` is `basna` with `mode='vatra'`.

## Shared vs forked — at a glance
| Concern | Basna | Vatra | Status |
|---|---|---|---|
| Router / catalog / reliability hints | ✓ | ✓ | **shared** (+1 axis, +lead field) |
| Spawn (fresh, tool-strip, no-recursion) | ✓ | ✓ | **shared** (+`CLAW_VATRA_WORKER`) |
| Initial dispatch | ✓ | ✓ | **shared** |
| Cross-agent interaction | none | blackboard asks | **new** |
| Coordinator | `_aggregate` (merge) | Lead loop | **forked** |
| Termination | merge barrier | budget/depth/cycle guards | **new** |
| Final output | weighted merge | reporter assembles | **forked** |
| Judge + reliability write | ✓ | ✓ (owners + lead + reporter) | **shared substrate** |
| Persistence / files / read-tool / callback | ✓ | ✓ | **shared** |

## Phases
- **Phase 1 — Lead loop, no delegation. ✅ Built.** Lead decompose (LLM) → spawn one owner per
  subtask → parallel dispatch with self-contained briefs → dedicated reporter assembles. No asks
  yet. Entry: `basna` tool `start` with `mode='vatra'` → `/fd/vatra/agent/start` (fire-and-forget
  + callback, reusing Basna's per-owner concurrency/run-rate guards). Budget = dispatch wall-clock.
  **Next: prod-test end-to-end**, then unify spawn into a shared helper.
- **Phase 2 — Ask protocol. ✅ Built.** `basna_asks` blackboard table (+6 db methods); `vatra`
  agent tool (`ask`/`inbox`), registered unconditionally, context injected via
  `CLAW_VATRA_SESSION/SUBTASK/OWNER/DEPTH` env. `/fd/vatra/agent/{ask,inbox}` endpoints
  (port-identified, owner-scoped; inbox long-polls). A **concurrent coordinator** (`_coordinate_asks`)
  runs alongside the owners: claims open asks, spawns a keyword-routed helper per ask
  (`_fulfill_ask`), writes the answer back; the reporter folds answered asks into the deliverable.
  Termination guards enforced at ask-creation: `_MAX_ASKS=12`, `_MAX_ASK_DEPTH=2`, dedup/cycle
  guard (identical ask → reuse), `_MAX_HELPERS=3` concurrency. Recursion guard moved worker-side
  into the `basna` tool (refuses `start` when CLAW_BASNA_WORKER **or** CLAW_VATRA_WORKER is set).
  **Next: prod-test end-to-end** (esp. that owners actually use `ask` for genuine cross-slice needs).
- **Phase 3 — Helper spawn + learning. ✅ Built.** (Helper spawn landed in Phase 2.) `_learn`
  scores four things into `archetype_reliability` via `record_archetype_outcome`: **owners**
  (usable slices judged against the deliverable with Basna's `_llm_judge`; empty owners auto-fail
  — also backfills `score_basna_run`), **ask-answerers** (each answer judged for soundness/use),
  and **lead + reporter** holistically (`_llm_judge_holistic` → one reason-tier verdict on
  decomposition quality and artifact coherence). Owners learn under their **real archetype id**
  (key shared with Basna, so the loop closes — Vatra outcomes feed the next route's catalog
  hints); **lead/reporter learn as separate pseudo-archetypes** (`vatra-lead`, `vatra-reporter`).
  Resilient: any judge failure leaves the affected contributions unscored rather than guessed.
- **Phase 4 — Frontend + auto-mode.** Ask ledger + delegation graph UI; optional router
  `auto_mode` so it picks independent vs collaborative itself, learned over time.

## Risks / things to watch
- **Cost blow-up** is the headline risk — bound it in Phase 1 (wall-clock) before asks exist.
- **Groupthink** — if Vatra is mis-routed onto truth-finding tasks it'll underperform Basna;
  keep the mode-choice heuristic conservative and measurable.
- **Lead as bottleneck/SPOF** — every ask routes through it, and it also gates the reporter.
  Fine at this scale; revisit only if a single Lead turn becomes the latency wall.
- **Reporter context limit** — on a big run the blackboard may not fit one reporter context.
  Mitigation: reporter reads slice summaries first, pulls full artifacts on demand via the
  `vatra`/`basna` read-tool; or `longctx` tier. Watch from Phase 1.
- **Don't re-enable recursion by accident** — `CLAW_VATRA_WORKER` must still block `basna`
  `start`/`deepen`, exactly as `CLAW_BASNA_WORKER` does today.

## Decisions (locked 2026-06-25)
1. **Topology** — hub-and-spoke / blackboard. Specialists post asks; the Lead routes. No
   peer-to-peer mailboxes.
2. **Final writer** — a **dedicated reporter archetype** assembles the deliverable. The Lead
   decides + routes and never writes the output. Lead and reporter are scored separately.
3. **Mode selection** — **user/flag picks** Basna-vs-Vatra in v1 (`auto_mode=false`). The
   router may learn to auto-pick in Phase 4.
4. **Name** — **Vatra** (hearth). Sits alongside Basna (ensemble) and Council (deliberation).
