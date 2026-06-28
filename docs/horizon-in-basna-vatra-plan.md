# Long-Horizon for Basna & Vatra — Design Plan (v1)

Status: draft for review. Goal: let **Basna** and **Vatra** *optionally* "think way ahead
and way long" — produce frontier-grade depth — by borrowing the **Frontier Horizon /
Dubina** engine. Today Basna/Vatra parallelize **breadth** (a fleet, one shot each);
Dubina drives **depth** (one thread, verifier-gated, escalating). This plan makes that
depth an opt-in lever inside the breadth modes.

See `FRONTIER_HORIZON_DESIGN.md` (the engine), `project_basna`, `project_vatra_collaborative_mode`.

---

## 1. How Dubina actually buys a long horizon

Dubina's one principle: *a weak model fails long tasks because per-step errors compound
(`p^n`). Buy back the horizon with test-time compute — but only where a **verifier** can
prove a step is good. Never let the model be its own ground truth.* It reconstructs the
three things frontier models get "for free" — low per-step error, error recovery, goal
coherence — externally in the harness. Five concrete mechanisms (all already built):

| # | Mechanism | Where | What it buys |
|---|---|---|---|
| 1 | **Sequential decompose → step chain** | `engine.py` `Decompose` hook + `run()` loop | the *horizon* — drive each step to a **verified** result before advancing; never compound an unverified step. Currently `_identity_decompose` (single step) — the seam is **unfilled**. |
| 2 | **Escalation ladder per step** | `engine.py` `_run_step` | depth on one step: single pass → **N-sample vote** (parallel) → **fix loop w/ feedback** (sequential) → **climb tier** (up to `max_tier`). Most steps stop at rung 1. |
| 3 | **Verifier-gated advancement** | `Verifier` protocol; `CoderVerifier` (tests = ground truth) / `ReasonVerifier` + `ReasoningJudge` (self-consistency + diverse-lens critics) | the gate. Statistical track: agreement (free, always-on) then **phrygian/aeolian/locrian critics** only when agreement is low or stakes high. Agreement catches *unsure*; critics catch *confidently wrong*. |
| 4 | **Budget + ceiling, no silent truncation** | `Budget`, `EngineConfig(compute_budget, ladder, max_step_samples, max_fix_attempts)` | cost discipline + honest "stopped at rung N". |
| 5 | **Run-target adapter** (`DispatchProvider`) | `dubina_agents.py` `make_agent_factory` / `ArchetypeRunner` | **the bridge.** A "provider" is anything with `complete(messages)->text`, so the engine runs **unchanged over agents** — a live fleet agent (tier ignored, sampling+fix axes) or a freshly-spawned archetype (re-spawn per tier = real tier climb). |

**The key realization:** Dubina already inverted the relationship we want. Its Intent track
drives *an agent* through the engine. We want the *orchestrator* to drive *its workers*
through the engine. Same adapter (`make_agent_factory` / `ArchetypeRunner`), opposite caller.
Basna spawns each worker at `(port, token)`; wrapping that in `make_agent_factory(port,
token, on_action, on_usage)` turns one worker into a horizon-driven generator with ~zero
new plumbing.

---

## 2. Why Basna/Vatra have no horizon today

- **Basna** = strictly one-shot per agent. Each worker gets one prompt → one reply. No
  sampling, no verify, no fix, no escalate. Breadth only. (Deepen spawns a *sibling*
  session on blind-spots — forward-only, not a loop.)
- **Vatra** = breadth + collaboration (blackboard asks) + a shallow fixed horizon:
  intro → main → review (~3 passes), then a reporter glues slices. The Lead's
  `depends_on` is a *breadth* DAG, not a verify-gated *temporal* pipeline. No re-plan on
  failure, no per-piece verification gate.

So "make them think way longer" = inject Dubina's three axes (sample, verify/critique, fix/
escalate) and optionally its sequential decompose loop.

---

## 3. Three horizon levers (composable; ship in this order)

Each lever is independently switchable via a `horizon` block in the existing session
`config` JSON — **no DB migration** (same pattern as `intro_round`/`review_round`/
`reporter_archetype`). Default OFF → both modes behave exactly as today.

```jsonc
// rides in basna_sessions.config / vatra config, all optional
"horizon": {
  "mode": "off",            // off | worker | closer | plan
  "samples": 3,             // N-sample self-consistency vote per step
  "fix_attempts": 1,        // feedback-driven retries
  "critics": ["phrygian","aeolian","locrian"],
  "stakes": "normal",       // "high" forces critics even on agreement
  "agreement_threshold": 0.6,
  "escalate": false,        // re-spawn archetype at next tier on fail
  "max_tier": "reason",     // ceiling for escalation
  "compute_budget": 0       // <=0 unbounded; else Tier.cost units
}
```

### Lever A — **Horizon Worker** ("each agent thinks way longer") — Basna-first

Replace each worker's single `_dispatch_one(...)` with an engine-driven horizon over that
worker. Per worker, instead of one shot:

1. **N-sample vote** (self-consistency) — *requires independent rollouts*, see §5.1.
2. **Diverse-lens critics** gate (on a **different** Library-tier model — never self-judge).
3. **Fix loop** — re-dispatch one rollout with the critic feedback folded in.
4. *(optional)* **Escalate** — re-spawn the archetype at the next Library tier.

The merge / blackboard / learning stages are **untouched** — they just receive a far-more-
verified per-worker output. This is the highest-value, lowest-friction lever, and it maps
perfectly onto Basna's "one-shot per agent" weakness.

- **Best for:** reasoning/decision archetypes (answer clusters cleanly on the `Answer:`
  line). For artifact producers, set `samples: 2, stakes: "high"` — skip the agreement
  signal (weak on long prose) and lean on the fix-loop + critics.
- **Build:** a thin `horizon_worker.py` helper:
  `async def run_worker_with_horizon(spawn_or_port, subtask_prompt, cfg, critic_provider, on_action, on_usage) -> {output, actions, confidence, rungs}`.
  Internally builds `HorizonEngine(EngineConfig(ladder=[Tier(tier)], …),
  make_reasoning_generator(provider_factory), ReasonVerifier(), aggregator=ReasoningJudge(
  load_critic_modes(critic_provider, modes), …))` where `provider_factory` is
  `make_agent_factory(port, token, …)` (no escalation) **or** an `ArchetypeRunner`
  (escalation). Returns the engine's best verified candidate as the worker's `output`.
- **Wire:** Basna `execute_route` dispatch phase (~`basna_routes.py:1742`) — when
  `horizon.mode in {worker}`, route through the helper instead of `_dispatch_one`. Vatra
  `_dispatch_owner` (~`vatra_routes.py:603`) — same swap.
- **Stream:** pass Basna/Vatra's existing `on_action`/`on_usage` straight through (the
  adapters already accept them), so every sample/critic/fix shows in the live log with a
  rung label.

### Lever B — **Horizon Closer** ("verify the team's answer, push deeper if weak")

After Basna merges (`_aggregate`) or Vatra's reporter assembles, run the **final artifact**
through the statistical verifier once. If it fails the agreement+critic gate, take **one**
recovery action instead of returning a weak answer:

- Basna: re-dispatch the lowest-weight / disagreeing contributors with the critic feedback
  (a real *back-edge* — today Basna only goes forward, or spawns a sibling deepen session).
- Vatra: hand the critic feedback to the reporter (or the owning specialist) for one revise
  pass.

This closes the loop Basna/Vatra structurally lack, cheaply (one verify + at most one
recovery round). Reuses `ReasoningJudge` + the merge/reporter seams already in place.

### Lever C — **Horizon Plan** ("think way ahead — a verify-gated multi-step pipeline")

The genuinely *long* version, and the one that fills Dubina's unfilled `Decompose` seam.
A planner turns the task into an **ordered** chain of steps; each step is itself a **Basna
ensemble** (or Vatra team) whose verified output feeds the next; a failed verify triggers
**re-plan** (the `plan_mode.py` verify→replan pattern the design doc already cites).

- Basna/Vatra become "the generator for **one step**" of `HorizonEngine.run()`.
- For Vatra this is natural: the Lead already decomposes — upgrade `depends_on` from a
  parallel DAG into a **topological, verify-gated** pipeline, with re-decomposition on a
  failed step (reuse `_llm_decompose` for the re-plan).
- Reuse candidates (all present in repo): `plan_mode.py` (verify→replan),
  `agent_pipeline_mixin.py` (planner+critic contracts), `session_orchestrator.py`,
  `agent_scale_loop_mixin.py` / `agent_scale_detection_mixin.py` (difficulty → how deep
  to plan).
- Heaviest lever; gate behind budget/ceiling hard. This is "way ahead and way long."

---

## 4. What's reused vs. new

**Reused as-is** (no changes): `HorizonEngine`, `EngineConfig`, `Budget`, `Tier`,
`make_reasoning_generator`, `ReasoningJudge`, `ReasonVerifier`, `load_critic_modes`,
`extract_answer`/`agreement_score`; the run-target adapters `make_agent_factory` /
`ArchetypeRunner` / `DispatchProvider`; Basna's `_send_chat_and_collect` / `_dispatch_one`,
spawn machinery, `_PROGRESS` live log, `record_archetype_outcome` learning; Vatra's Lead
`_llm_decompose`, reporter, blackboard.

**New** (small): `horizon_worker.py` (Lever A helper, ~120 lines), the `horizon` config
block + UI toggle, the closer hook (Lever B, ~60 lines in each mode), and — Lever C only —
a `BasnaStepGenerator` / `VatraStepGenerator` that adapts "run one Basna/Vatra round" to the
engine's `Generator`, plus a real `Decompose` + re-plan binding.

---

## 5. Tensions to design around (call these out, don't paper over)

1. **Self-consistency needs *independent* rollouts.** Dubina's `make_agent_factory`
   dispatches N samples to **one** agent over one websocket — that accumulates
   conversation state, not independent samples. For a sound vote the worker step must
   **spawn N fresh instances** of the archetype (Basna's spawn machinery already does this —
   spawn the same archetype ×N for the vote rung) rather than re-prompting one agent. This
   is the one real semantic fix; budget for it. (Live-agent targets keep
   `samples: 1` + fix-loop only.)
2. **Cost multiplies.** N spawns × tiers × fix attempts per worker. That *is* the
   "spend compute to simulate a frontier model" trade — but make it opt-in, budget-bounded,
   and surfaced ("stopped at rung N, spent Xu"). Default OFF.
3. **Never self-judge.** Critics must run on a **different** model than the worker — reuse
   Dubina's Intent-track pattern exactly: `critic_provider =
   _library_provider_factory(tiers_map)(base_tier)` (agreement-only if none configured).
4. **Agreement clustering is weak on long artifacts.** It keys on a final `Answer:` line.
   For artifact-producing archetypes prefer `samples: 2, stakes: "high"` (critics + fix,
   skip agreement) or wire the **coder verifier** when a subtask emits testable code.
5. **Termination (Lever C).** A plan-horizon can run away. Bound it: `max_steps`,
   `compute_budget`, re-plan cap, no-progress guard — mirror Vatra's existing
   `_MAX_ASKS`/`_MAX_ASK_DEPTH` discipline.
6. **The hard ceiling (from the design doc).** Scaffolding elevates *capable-but-incoherent*,
   not *fundamentally-weaker-per-atom*. Bias to ground-truth verifiers where possible;
   accept that statistical verification has a real model-judges-itself floor.

---

## 6. Phased build

- **Phase 1 — Lever A on Basna** ✅ **built (2026-06-28, not yet prod-tested).**
  `captain_claw/flight_deck/horizon_worker.py` — `run_worker_horizon()` spawns a pool of
  N fresh archetype instances (independent rollouts), round-robins the engine's vote
  samples across them, runs the reasoning track (self-consistency + diverse-lens critics +
  fix loop) on a **single-tier** ladder (no escalation yet), disposes the pool. Reuses the
  Dubina engine/judge/verifier verbatim; `spawn`/`send`/`stop` injected → stub-tested.
  Wired into `basna_routes.execute_route`: `ExecuteRequest.horizon` (or `route.horizon`)
  → `HorizonConfig`; `_dispatch_horizon_workers` replaces the one-shot spawn+dispatch when
  on, leaving merge/learning untouched; critics run on a **separate** Library-tier provider
  (`_resolve_merge_creds(critic_tier)`, default `reason`) — never self-judge; pool slugs
  registered in `_run_workers` for Stop; per-sample/rung lines stream into `_PROGRESS`.
  Frontend: a "Deep" toggle + samples knob (`basnaStore.deep/deepSamples`, Basna-mode only)
  that rides the execute body as `horizon:{samples}`. 9 unit tests
  (`tests/test_flight_deck/test_horizon_worker.py`) + 62 dubina/horizon tests green, ruff
  clean, `npm run build` clean.
  - Known P1 limitations (by design): single tier only (no model escalation); the engine's
    "single" rung always defers in reasoning mode (judge needs ≥2 samples), so it costs one
    extra dispatch per worker before the vote; agreement clustering keys on a final
    `Answer:` line so it's best for reasoning/decision archetypes, not long artifacts.
- **Phase 2 — Lever B closer on Basna + Vatra** ✅ **built (2026-06-28, not yet prod-tested).**
  `run_horizon_closer(question, answer, critic_provider, revise_provider, critics)` in
  `horizon_worker.py`: a diverse-lens critic panel (run on a separate Library-tier model)
  reviews the final answer; on **strict-majority refutation** it produces one feedback-driven
  revision (`_revise`), else passes through. `HorizonConfig` gained `worker`/`close` flags
  (Lever A and Lever B toggle independently). Basna `execute_route` runs the closer after
  `_aggregate`/analysis when `cfg.close` (updates `agg.truth`, method `…+revised`); worker
  dispatch now gated on `cfg.worker`. Vatra `execute_vatra` runs it after `_run_reporter`
  assembles `truth` (when `cfg.close`); `horizon` threaded through `VatraExecuteRequest`/
  `VatraStartRequest` + persisted in session config. Frontend: the "Deep" toggle now shows in
  **both** modes — Basna sends `{samples, close:true}` (worker+closer), Vatra sends
  `{close:true}` (closer only); samples knob is Basna-only. 5 closer unit tests (14 total in
  the horizon suite) + 67 dubina/horizon green, ruff clean, `npm run build` clean.
  - **Lever A on Vatra deferred** (was the original P2 scope): per-owner self-consistency
    pools conflict with Vatra's shared blackboard — N independent copies of one owner would
    each post/ask, polluting the board and multiplying the coordinator's work. Needs a
    blackboard-aware design (e.g. a per-owner critics+fix closer that keeps the single
    collaborating owner) → fold into a later phase.
- **Phase 3 — Lever C plan-horizon** ✅ **built (2026-06-28, not yet prod-tested).**
  `captain_claw/flight_deck/horizon_plan.py` — `run_plan_horizon(task, planner, step_runner,
  verifier, synthesizer, cfg)`: decompose → for each step run+verify (with a fix loop) →
  **re-plan the remainder on a hard step failure** → advance → synthesize. Pure orchestration,
  all four seams injected → stub-tested (13 tests). Bounded by `max_steps`/`max_fix_per_step`/
  `max_replans` + a hard iteration cap; explicit `stopped_reason` (`""`/`max_steps`/
  `step_unverified`/`empty_plan`); never silently drops an unverified step (kept + flagged).
  Concrete LLM seams `make_llm_planner`/`make_llm_step_runner`/`make_llm_verifier`/
  `make_llm_synthesizer` (provider injected; verifier soft-passes low-conf on a flaky judge).
  Wired into Basna as **`POST /fd/basna/plan`** (`PlanRequest` → background `_run_plan`):
  creates a `mode:"plan"` session, runs the engine on one Library tier (`plan_tier`, default
  `reason`), streams plan/step/verify/re-plan/synthesize lines into the shared `_PROGRESS`
  log, persists the deliverable as the session `truth` + a `kind:"plan"` analysis (per-step
  verified/confidence). Frontend: a green **"Plan"** toggle + steps knob (Basna-mode,
  mutually exclusive with Deep); the Run button becomes **"Run plan"** → `runPlan` →
  `/fd/basna/plan`; reuses the existing live log + truth view. 13 plan tests (80 total) green,
  ruff clean, `npm run build` clean.
  - Two step runners: **lean** (one `reason`-tier generation per step, default — fast) and
    **ensemble** (each step = a full Basna run). The sequential verify-gated re-planning loop
    is the new long-horizon capability either way.
- **Phase 3.1 — ensemble-per-step** ✅ **built (2026-06-28, not yet prod-tested).**
  `make_basna_ensemble_step_runner(parent_sid, body, user, route_fn, execute_fn, on_step)` in
  `basna_routes.py`: per step it routes+executes a **child Basna session** on the step goal
  (Library `fast` creds threaded into the child router; prior verified results as context),
  tags the child `{source:"plan-step", parent}`, and returns the ensemble's merged truth.
  Child sessions are real Basna runs — their archetype-reliability learning still closes — and
  are hidden from the session list (filtered client-side by `source=="plan-step"`).
  `route_fn`/`execute_fn` injected → 3 unit tests without spawning agents.
  `PlanRequest` gained `step_mode` (`llm`|`ensemble`) + `step_max_agents` + `dispatch_timeout`;
  `_run_plan` picks the runner and streams per-step `↳ ensemble · N agent(s) · conf …` lines.
  Frontend: an **"ensemble steps"** checkbox under Plan → `step_mode:"ensemble"`. 3 tests
  (83 total) green, ruff clean, `npm run build` clean.
- **Phase 3.2 — Vatra-team-per-step** ✅ **built (2026-06-28, not yet prod-tested).**
  `make_vatra_team_step_runner(parent_sid, body, user, create_session_fn, execute_fn, on_step)`:
  per step it creates a child **Vatra** session on the step goal (prior verified results as
  context) and runs `execute_vatra` (Lead decomposes → specialists collaborate on the
  blackboard → reporter assembles), returning the deliverable. `execute_vatra` is imported
  **lazily** (vatra_routes imports basna_routes — avoids the cycle); seams injected → 2 tests.
  `step_mode` is now `llm | ensemble | vatra`; `_run_plan` picks the runner and streams
  `↳ team · N subtask(s) · conf …`. Frontend: the Plan control's "ensemble steps" checkbox
  became a **"Steps" select** (single / Basna ensemble / Vatra team) → `step_mode`. So one
  plan-horizon now drives three step engines; Vatra "thinks way ahead" by running a full team
  per verified step. 5 plan-step tests (85 total) green, ruff clean, `npm run build` clean.
- **Phase 3.3 — `depends_on` topological gate (DAG)** ✅ **built (2026-06-28, not yet prod-tested).**
  `run_dag_horizon` in `horizon_plan.py`: the planner emits a DAG (`make_llm_dag_planner` →
  `[{id, goal, depends_on}]`); `_normalize_dag` guarantees a runnable graph (unique ids, drop
  dangling/self/back-edges → cycles broken, cap to `max_steps`); steps run in **dependency
  waves** (`asyncio.gather` within a wave — independent steps concurrent); each step sees only
  its **direct dependencies'** verified outputs (`_dag_context`); the shared `_execute_step`
  fix loop runs per step; an unverified step still feeds its dependents (flagged), and an
  unschedulable graph ends `blocked` rather than hanging. Wired into `/fd/basna/plan` via
  `PlanRequest.dag` (`_run_plan` branches to `run_dag_horizon`; `_plan_on_event` made
  id-aware). Frontend: a **"parallel"** checkbox under Plan → `dag:true`. 9 DAG tests (94
  total) green, ruff clean, `npm run build` clean.
- **Phase 3.4 — Vatra per-owner depth (Lever A, blackboard-safe)** ✅ **built (2026-06-28,
  not yet prod-tested).** Instead of spawn-×N pools (which would each post/ask and pollute the
  blackboard), `execute_vatra` runs the **closer per owner slice** before the reporter
  assembles: each specialist's piece is adversarially verified by a diverse-lens critic panel
  (separate Library-tier model) and revised once if refuted, concurrently across owners
  (`run_horizon_closer` reused). Gated by `horizon.worker`; the parse is shared with the final
  closer. Frontend: Vatra Deep now sends `{worker:true, close:true}` (per-slice **and** final
  verify+revise). Reuses the tested `run_horizon_closer`; 94 tests green, ruff clean, build clean.
  - **Only remaining refinement:** DAG **re-plan** on hard step failure (the DAG recovers via
    the per-step fix loop + best-so-far; whole-graph re-plan is linear-only). Everything else
    in the plan is built.
- **Phase 4 — Tuning & (optional) coder verifier.** Empirically set agreement threshold /
  critic bar; wire `CoderVerifier` for code-emitting subtasks (ground truth beats
  statistical). Optionally let the router auto-enable horizon by detected difficulty
  (`agent_scale_detection_mixin`).

**First cut to de-risk:** Phase 1 only — Lever A on Basna, reasoning archetypes, no tier
escalation (`make_agent_factory` over already-spawned workers won't give independent
samples, so use spawn-×N), default OFF. It proves the depth win on the simplest mode before
touching Vatra's collaboration or the plan loop.
