# Loops & Graphs — Methodology Review

Status: review for kstevica. Compares an external agent-design methodology (Hanako /
@hanakoxbt, "Prompts → Agents → Loops → Graphs"; companion course *agent-layers*)
against what Captain Claw and Flight Deck actually implement, and recommends
concrete improvements. Findings are backed by a five-track code audit; file:line
refs are inline.

---

## TL;DR

Captain Claw is **already one of the most methodology-aligned agent systems you
could point this framework at** — and ahead of the thread in two places (blast-radius
gating; a verifier-gated depth engine). The framework is still a sharp lens: it
exposes **three real gaps** and a handful of smaller ones, all fixable by wiring
together machinery that *already exists in the repo*.

The three that matter:

1. **The check isn't the decision** (Code studio). Deterministic gates exist (test
   exit code, predicate contract, AST import check) but are opt-in and *advisory* —
   the ship/no-ship verdict is always an LLM triage, and a triage exception
   **defaults to PASS**. This is the thread's "no errors raised ≠ correct" trap.
2. **The learning edge learns the wrong thing.** The automatic accept/fail loop
   distills a *routing weight* ("which archetype to trust") — not a *task constraint*
   ("adapters preserve keyword args exactly"). The one path that carries a real
   derived rule into a planner (playbooks) is human-triggered.
3. **Planners default to false sequential edges.** The DAG planner prompt says
   "`depends_on` the previous step by default"; `"after"` is accepted as a synonym
   for a dependency. That is exactly the thread's "treating *and then* as an edge,"
   and it serializes independent work.

Everything below expands these, and Part 4 is the prioritized "what & how."

---

## Part 1 — The methodology in one page

- **Loop** = `produce → CHECK → correct → repeat-until-green`. The **check is the
  whole thing**: it must be program-evaluable and able to **fail while nobody is
  watching** ("test suite exits 0", "every claim carries a source line", "diff
  touches only planned files"). *"Output looks good", "model is confident", and "no
  errors were raised" are NOT checks.* A loop makes **one unit** correct.
- **Graph** = the layer above: what runs, what runs in parallel, what never runs,
  where results flow back. Only two primitives: a **node** (one bounded unit, one
  input, one output) and an **edge** (a real data dependency — this node's output
  *feeds* that node's input). **The core mistake is treating "and then" as an edge.**
  Test every arrow: *can you name the variable that crosses?* If not, there is no
  edge and the wait is waste.
- **Four node types:** **splitter** (cuts work into units; the *dimension* it cuts by
  matters most), **worker** (one unit, one lens, **its own context** — a shared window
  makes workers converge into an echo chamber), **code node** (deterministic
  merge/rank/dedupe/vote — *"if you can describe it without judge/decide/assess/
  summarize, it's code,"* not a model), **gate** (accept/reject/route on a check).
- **The loop lives inside a node; the graph lives between nodes.**
- **Two return paths.** The **correction edge** is short: a gate returns **one failing
  unit** (with VERDICT + REASON + EVIDENCE + **SCOPE** = "fix only this file"), capped
  at ~3 tries — *return the unit, not the batch.* The **learning edge** is long: an
  accepted result becomes a **derived constraint** fed back into the **splitter's
  brief** — it carries the *rule*, not the output, and "lands in the brief that shapes
  how work gets cut." Its absence = "a system that is fast and never gets smarter."
- **Gate on blast radius, not confidence.** Confidence is the one input the model can
  influence. Sort by cost-to-undo: reversible+contained → can auto-open; reversible
  but wide → gate on deterministic checks; **hard-to-reverse (migrations, deletions,
  prod, money) → a lane that does not open, regardless of score** (a closed lane, not
  a high threshold). Inside an open lane read evidence in order: deterministic
  results → run trajectory → this node's rollback history → **model's opinion last**.
- **Human on one step:** highest consequence, lowest reversibility. Approve the
  merge / which fixes ship — *not* intermediate output. "A human in the middle of a
  graph becomes the slowest node in it."
- **Three maxims:** measure the *path*, not just the answer · a verdict that doesn't
  change what runs next is a report · any failure you don't turn into a permanent
  constraint, you'll meet again.

---

## Part 2 — Scorecard

| # | Principle | Captain/FD status | Where |
|---|---|---|---|
| 1 | Loop = produce→**check**→correct→repeat | **Strong** (Dubina) / **Partial** (Code) | `dubina/engine.py`, `code_routes.py` |
| 2 | The check is deterministic & can fail unattended | **Partial** — exists but **advisory**, opt-in | `code_verify.py`, `quality_profile.py` |
| 3 | Node = unit, Edge = real data dependency | **Split** — DAG ✓, Vatra data-edge ✗, Flows ✗ | `horizon_plan.py`, `vatra_groups.py`, `flow_runner.py` |
| 4 | "and then" ≠ edge (auto-parallelize independents) | **Weak** — planners default to sequential | `plan_mode_complete_system_prompt.md:20` |
| 5 | Splitter, and the cut *dimension* is chosen | **Good** cut, **no** dimension-choice node | `instructions/vatra/lead.md:8,12` |
| 6 | Worker in its own context (no echo) | **Strong** (Basna/Code) / **mixed** (Vatra board, Council) | `basna_routes.py:1814`, `tools/vatra.py:31` |
| 7 | Code node for deterministic merge/vote | **Strong** (Basna) / **gaps** (Council vote, fan-in) | `basna_routes.py:1963-2038`, `council_routes.py` |
| 8 | Correction edge: return the **unit**, scoped, capped | **Good** cap+honest-stop / **batch-per-round**, soft scope | `code_routes.py:1641-1816` |
| 9 | Learning edge: accept/fail → **constraint** → splitter | **Partial** — learns *who to trust*, not *what rule* | `db.py:1687`, `agent_playbook_mixin.py:496` |
| 10 | Gate on **blast radius**, not confidence | **Strong** — but siloed in the autonomy loop | `action_catalog.py`, `fd_dispatch.py:125-183` |
| 11 | Human on one high-consequence gate | **Strong** (autonomy, Vatra plan gate) | `arbiter.py`, `autonomy_routes.py` |
| 12 | Measure the path, not the answer | **Partial** — captured; LLM judge sees output only | `tracing.py`, `fd_dispatch.py:205` |

Legend: Strong = matches or exceeds; Partial = present but incomplete; Weak = mostly missing.

---

## Part 3 — Findings by principle

### 3.1 The loop and the check

**Frontier Horizon / Dubina is a textbook loop.** `decompose → step → verify →
recover/escalate → advance`, budget-bounded (`dubina/engine.py`). The coder verifier's
pass/fail *is `pytest`'s exit code* — zero LLM tokens — and when no tests exist it
**generates them first** (spec→tests→code) so a ground-truth check always exists.
Design value, verbatim: *"Never let the model be its own ground truth."* Reasoning
track uses self-consistency + **diverse-lens** critics (not N identical ones), and
caps attempts with *"no silent truncation."* This is arguably a cleaner statement of
the thread's loop than the thread itself.

**The Code studio has the machinery but not the discipline.** It runs a real loop
(`_big_job`, `code_routes.py:1641-1816`) with independent reviewers, a test gate
(`code_verify.py`), a rubric-from-plan contract (`code_contract.py`), an
anti-fabrication honesty guard (`code_honesty.py`), a `_acted()` gate (*"Text is NOT
work"*), a cap of 3 fix rounds, and an **honest resumable stop** (writes a backlog,
never ships-as-clean). But:

- Deterministic checks are **opt-in** (`QualityProfile` default `"off"` →
  `quality_profile.py:99-136`). Out of the box, pass/fail is 100% LLM (3 reviewers +
  triage).
- Even when armed, the deterministic signal is **advisory**: a failing test becomes a
  synthetic *"GROUND TRUTH"* review entry handed **into the LLM triage**, and
  `final_clean` is set **solely** from `triage.needs_fix` (`code_routes.py:1763-1766`).
  A green suite is never a hard precondition to ship. The one deterministic veto
  (`block_on_critical`) is opt-in, in no preset, and fires only on **regression**.
- **Fails open:** a triage exception returns `needs_fix:False` → declared clean
  (`code_routes.py:1060-1064`). The small-edit path ships on "dispatch didn't error +
  a commit landed," with no check at all (`code_routes.py:2454-2467`).

That last cluster is precisely the thread's *"a system that confidently repeats a
mistake until the budget runs out, with a clean log the whole way."*

### 3.2 The graph: nodes, edges, "and then"

There are **two** step systems and they answer the graph question differently:

- **Flows DSL** (`flow_dsl.py`, `flow_runner.py`) is **not a graph** — a linear script
  executed by index. Independent steps written normally run **in series**; the only
  parallelism is hand-written `spawn`/`join`. Dependencies are positional, not data
  edges, and a typo'd `{{steps.x.output}}` **silently becomes `""`** (no data-edge
  validation). Its strengths are real determinism (`tool` = no-LLM call; `set` = a
  safe non-`eval` expression evaluator) and true parser validation — but the parser
  checks syntax + control-flow targets, never data flow.
- **task_graph** (`task_graph.py`) is a **real DAG** — explicit `depends_on`, Kahn sort,
  automatic parallel fan-out (`activate_next`, up to `max_parallel`). It has a
  splitter (`orchestrate`), workers (`atomic`, own session), a gate (`verify`), a
  fan-in join.

The DAG *supports* minimal edges, but **the planner manufactures false ones**:

- `instructions/plan_mode_complete_system_prompt.md:20`: *"Steps `depends_on` the
  **previous step by default** (sequential plan)."* — the thread's exact error.
- `agent_reasoning_mixin.py:571` accepts `"after"` as a synonym for `depends_on` —
  encoding "and then" as a dependency.
- Nothing checks that a variable actually crosses an edge: `workspace_inputs` /
  `workspace_outputs` exist on a task (`task_graph.py:68-70`) but are **advisory
  manifest keys**, never validated against the `depends_on` set the scheduler uses.

**The best-kept secret:** `horizon_plan.py` is a DAG planner that already does it
right — *"Decompose the task into a DAG … so independent steps can run in parallel and
**each step only consumes what it actually needs**"* (`:364-370`), and `_dag_context`
(`:213-221`) **pushes the verified upstream outputs into the downstream step's
context** (a real output→input edge), runs in dependency waves, and re-plans. It is
wired into Basna as an **opt-in** (`basna_routes.py:4356`, `body.dag=True`) and the
headline modes don't use it by default.

### 3.3 Node types

- **Splitter** — Vatra's Lead is a genuine anti-overlap splitter: *"smallest set of
  **complementary, non-overlapping** pieces"*, *"Pieces, not perspectives … Do NOT
  assign two agents the same thing"* (`instructions/vatra/lead.md:8,12`). Good cut.
  **Gap:** the *dimension* is hard-coded to "parts of a deliverable"; there is no node
  that *chooses* the cutting dimension (the thread's "cut by folder vs by blast
  radius" decision). The cut is LLM-made with no deterministic fallback.
- **Worker in own context** — **Basna** is genuinely blind/independent
  (`basna_routes.py:1811-1839`, spawned fresh, torn down); **Code** reviewers each run
  in a disposed ephemeral archetype with its own port/token/workspace — no
  self-grading. Both match well. **Mixed:** Vatra uses a **fully shared blackboard**
  (`tools/vatra.py:31-34`) — the shared-window shape the thread warns about, mitigated
  because slices are non-overlapping but with a deliberate convergence step in the
  review round. **Council** is a shared deliberation thread — convergence is the point.
- **Code node** — **Basna's merge is deterministic Python** (`_aggregate`,
  `_merge_diverge`, `_mean_weight`, `basna_routes.py:1963-2038`): weighted dedup +
  ranking + pick; an LLM synthesizer runs **only on genuine disagreement**. Clean
  match. **Gaps:** task_graph's fan-in is an LLM *"Synthesize the outputs"* step;
  **Council's decision is an LLM synthesis over a shared thread while the votes are
  stored but never tallied** (no `tally`/`majority` logic in the backend) — a decision
  point that should have a code node.
- **Gate** — see 3.6 (autonomy) and 3.1 (Dubina/Code); the strongest gates in the repo.

### 3.4 Correction edge (short return path)

Well-built, partial on "return the unit." Triage produces failure-scoped
`fix_instructions` (blocking/major only; minors ship) handed to a chosen fixer with
*"Fix ONLY the issues listed"*; rounds 2+ review **only the diff**; cap 3; honest
non-clean stop with a resumable backlog. **Gaps vs the thread:** all failing findings
go to **one fixer per round** (a batch of units, not one unit at a time); SCOPE is a
prose instruction, not a hard per-file lock enforced on the diff; there's no per-unit
`{UNIT, VERDICT, REASON, EVIDENCE, SCOPE}` envelope. So the thread's "return the unit,
not the batch — a returned unit must not grow past its file" is only half-honored.

### 3.5 Learning edge (long return path) — the deepest gap

The graph *does* bend back on itself, in two places, each missing a different half of
the definition:

- **Automatic, but a routing weight.** At each Vatra/Code run's end an unconditional
  `_learn` phase auto-judges contributions and folds success/fail into a
  **Bayesian-shrunk archetype-reliability weight** (fails count 2×; `db.py:1648-1717`),
  consumed by the Vatra/Basna/Code/Council splitters+routers (`basna_routes.py:209-227`).
  Cross-run, cross-session, per-domain. But it encodes **who is reliable**, not a task
  rule — it cannot express *"adapters preserve keyword args exactly."*
- **A real derived rule, but human-triggered.** Playbooks distill a rated session into
  structured `do_pattern`/`dont_pattern`/`trigger` about **orchestration decisions**
  and inject them into the task-contract planner (`agent_playbook_mixin.py:496`,
  `agent_reasoning_mixin.py:706-725`) — the one place a derived rule lands in "the
  brief that shapes how work gets cut." But it fires only on a human `rate` call, is
  human-approved, and is **main-agent only** (no flight-deck planner reads it).

Supporting facts: **contracts** are hard constraints but derived from the *current*
task, feed the *short* correction edge, and persist **per-folder** — turning a prior
*failure* into a standing constraint needs a manual `.contract.json` edit. The one
path that feeds a prior outcome into the splitter, `prior_knowledge`, carries the
**report text + gaps** — the output itself, precisely the anti-pattern the thread
names — and is opt-in. Insights/reflections reach only the system prompt, never a
planner.

Net: **the automatic learning edge learns *who to trust*; the constraint-shaped edge
depends on a human pulling the trigger.** "Any failure you don't turn into a permanent
constraint, you'll meet again" is realized for agent-reliability (auto) and
orchestration playbooks (manual), not as an automatic, task-level constraint injected
into the next decomposition.

### 3.6 Gate on blast radius, not confidence — Captain is *ahead* here

The autonomy loop is a near-verbatim implementation:

- Every action is tagged in a **curated catalog** with `risk`, `reversibility`
  (`read_only`/`reversible`/`irreversible`), `human_only` (`action_catalog.py:6-13`).
  Auto-dispatch fires **only** for reversible+low-risk+not-human-only
  (`fd_dispatch.py:125-183`), and *"risk/reversibility come from the catalog, never the
  LLM"* (`arbiter.py:507-508`).
- A **true closed lane**: `AUTONOMY_HARD_EXCLUDE` (payments, transfers, shell, browser,
  social; `config.py:854-863`) can never be reached by the loop — a wall, not a
  threshold. Self-mod keeps the genome/Constitution permanently out of scope.
- **Confidence is deliberately excluded** from the trust decision: auto-fire ignores
  the arbiter's score and uses **earned reliability** (Bayesian success-rate over runs)
  = rollback history (`fd_dispatch.py:164-168`).
- **Evidence order matches:** deterministic completion gates (file produced, script
  executed, DB re-queried; `agent_completion_mixin.py:180-359`) + reverse-read
  verification precede any LLM opinion; the LLM judge is last and narrow.
- Same reasoning recurs at small scale: datastore never infers a table for
  destructive ops (test-enforced); the terminal watcher never auto-answers anything
  destructive.

**The catch — it's siloed.** This discipline lives *only* in the Flight Deck autonomy
loop. The interactive agent guard (`agent_guard_mixin.py`), which sits in front of
*every* tool call in normal use, is an LLM *"is this suspicious?"* classifier with **no
notion of reversibility**; the Code studio **auto-approves plans**; shell falls back to
a flat allow/deny/ask list. So the reversibility taxonomy does **not** reach the place
where big irreversible *code* changes happen. Also, reverse handles are best-effort
(a single `ID:\s*(\S+)` regex) — a "reversible" action whose id can't be parsed
silently has no working Undo.

### 3.7 Human placement & "measure the path"

Human placement is strong: the autonomy loop is **one** approval gate with async
dispatch + one-tap **Undo**; Vatra pauses at the **Group 0 plan gate** before anything
spawns; Code gates at plan approval. Residual risk of a "human in the middle": the
interactive shell `ask` policy and guard `ask_for_approval` are **per-action**
confirmations (mitigated by worker auto-approve and a 15s auto-approve timeout — which
itself weakens the gate). **Trajectory** is captured (`tracing.py`,
`cognitive_metrics.py`) and used by the deterministic completion gate and reliability
weight — but the **LLM outcome judge reads only the final output** (`output[:4000]`,
`fd_dispatch.py:205`), so "measure the path" holds for the deterministic layer, not the
model verdict.

---

## Part 4 — Recommendations (what & how), by leverage

Effort/impact are rough (S/M/L). Each reuses machinery already in the repo.

### R1 — Make the deterministic check the *decision*, not advice · impact **High** · effort **S–M**
*Principle 2 — "the check is the whole thing."*

- When a contract/test gate is armed, make a failing deterministic check a **hard veto**
  on `final_clean`, independent of triage: `final_clean = triage_clean AND
  deterministic_gates_pass`. A green suite becomes a precondition to ship
  (`code_routes.py:1763-1766`).
- **Fail closed:** a triage exception or unparseable JSON should yield `needs_fix:True`,
  not `False` (`code_routes.py:1060-1064`). Reuse the repo's own
  `output_validation.validate_task_output` + `build_retry_prompt` instead of
  hand-coercing triage JSON.
- Give the small-edit path a minimal check — run the test command if one exists, or a
  one-line contract predicate — before recording success (`code_routes.py:2454-2467`).
- Promote `constraints_contract` + the test gate into the **Balanced** preset (or make
  Code refuse to declare "clean" without at least one program-evaluable check),
  so the discipline is on by default (`quality_profile.py`).

### R2 — Turn the learning edge into a *constraint* edge · impact **High** · effort **L**
*Principle 9 — "carry the rule, not the output; land it in the splitter's brief."*

- Add an automatic **constraint-distillation** step at the existing `_learn` seam
  (`vatra_routes.py:2282`) and Code post-run hook (`code_routes.py:1887`): on a failure
  that was fixed (or an accepted result), derive a **short structured rule**
  `{trigger, constraint_text, severity, domain}` — reuse the playbook distiller
  (`playbook_distill_system_prompt.md`) but fire it *automatically*, not on a human
  `rate`.
- Persist these as **domain-scoped standing constraints** (a sibling table to
  `archetype_reliability`), *not* per-folder.
- **Inject them into the splitters/planners that today get nothing learned:** the Code
  slice planner (`code_plan.py:decompose_prompt`, `code_routes.py:_plan_prompt`), the
  Vatra Lead decompose prompt, and the task-contract planner. This is the literal "lands
  in the brief that shapes how work gets cut."
- Fix `build_prior_knowledge` (`basna_routes.py:1843`) to forward **derived constraints
  first**, not the raw report text.
- Auto-promote a **repeated** contract failure into a standing constraint (close the
  manual `.contract.json` gap).

### R3 — Kill false sequential edges; validate that a variable crosses each edge · impact **High** · effort **M**
*Principle 4 — "and then ≠ edge" (the thread's largest single speedup).*

- Rewrite the two planner prompts to be **parallel-first**:
  `plan_mode_complete_system_prompt.md:20` and `orchestrator_decompose_system_prompt.md`
  → *"A step `depends_on` another ONLY if it consumes a named output of that step.
  Default to no dependency. For each edge, name the artifact that crosses."* Drop
  "previous step by default." Deprecate the `"after"` synonym
  (`agent_reasoning_mixin.py:571`) or require it to name a produced artifact.
- Add a **data-edge lint** in `plan_mode`/`task_graph`: validate each `depends_on`
  against `workspace_inputs`/`workspace_outputs` — drop an edge whose consumer reads no
  output the producer emits; error on an input with no producer. This operationalizes
  "name the variable that crosses" and makes the advisory manifest load-bearing.
- For **Flows**: add a validator/authoring hint that flags adjacent sequential steps
  where the later step does **not** reference `{{steps.<earlier>.output}}` (candidates to
  parallelize), and validate `{{steps.x.*}}` references against declared step ids so a
  typo stops being a silent `""` (`flow_dsl.py:validate_flow`).

### R4 — Promote the DAG engine; make Vatra edges carry data · impact **Med–High** · effort **M**
*Principles 3 & 4 — enforced output→input edges.*

- Vatra (grouped mode) already knows the producer→consumer edges (`consumes_from`,
  enforced for *ordering* in `vatra_groups.py:144-148`). Make it **push** the producer's
  verified output into the consumer's context — reuse `horizon_plan._dag_context`
  (`:213-221`) — instead of relying on the consumer to `vatra(action=search/wait)` and
  possibly proceed without it. Keep pull as fallback.
- Surface the **DAG Plan-Horizon** lever (`horizon_plan.py`) as a first-class default
  for multi-step work rather than a hidden `body.dag=True` — it *is* the graph
  methodology already implemented (real edges, blind steps, verify-gates, re-plan).

### R5 — Extend the blast-radius gate beyond the autonomy loop · impact **Med–High** · effort **M**
*Principle 10 — the closed lane belongs everywhere irreversible work happens.*

- Reuse the `action_catalog` reversibility/risk tags (or a light classifier) in the
  **Code studio ship step** and the **interactive agent guard**: a slice/action that
  touches migrations, deletions, schema, prod, force-push, or money routes to the human
  gate (or a closed lane) even outside the autonomy loop — not just "is this
  suspicious?".
- Make the Code fix loop's implicit auto-approve honor a **hard-to-reverse lane**: such
  a slice never auto-ships without the one human gate.
- Firm up reverse handles: if Undo can't be captured, **downgrade the action out of the
  reversible lane** rather than claiming a lane you can't honor (`action_catalog.py`).

### R6 — Return the unit, not the batch, in the Code fix loop · impact **Med** · effort **M**
*Principle 8.*

- Send each failing finding/file back as its **own** scoped correction with a hard
  `{UNIT, VERDICT, REASON, EVIDENCE, SCOPE="only touch file X"}` envelope, and fix
  independent ones in **parallel** (mirror Basna's per-worker isolation). Don't rewrite
  passing slices.
- **Enforce scope on the diff:** after a fix, assert it touched only the in-scope
  file(s) (you already have `block_on_critical`'s git plumbing) and reject/revert
  out-of-scope edits — the thread's "diff touches only planned files."

### R7 — Put a code node at deterministic decision points · impact **Med** · effort **S–M**
*Principle 7.*

- **Council:** add a deterministic vote **tally** (majority, or reliability-weighted)
  as the recorded decision; keep the LLM synthesis as the *rationale*, not the verdict
  (`council_routes.py` already stores votes — just count them).
- **task_graph fan-in:** where the merge is concatenation/dedup/rank, use a code node
  (extend Flows' `set` with `dedupe`/`sort`/`rank`/`unique`, or a shared merge helper)
  instead of an LLM "Synthesize."
- Generalize Basna's deterministic-merge seam (`_aggregate`) into a reusable code-node
  primitive other modes can call.

### R8 — Feed trajectory into the judge; make "measure the path" real · impact **Low–Med** · effort **S**
*Maxim 1.*

- Include a compact trajectory summary (tool calls, retries, test results, cost) from
  `tracing.py`/`cognitive_metrics.py` in `_judge_outcome` (`fd_dispatch.py:186-233`), so
  the model verdict weighs the path, not just `output[:4000]`.

---

## Part 5 — Where Captain already *beats* the thread

- **Blast-radius gating** (R-principle 10): a curated reversibility/risk catalog, a real
  closed lane, confidence excluded from the trust decision, evidence ordered
  deterministic-first, Undo handles. Most systems (and the thread) describe this; Captain
  ships it.
- **A verifier-gated depth engine** (Dubina): "never let the model be its own ground
  truth," generate-tests-first, diverse-lens critics, cost-laddered escalation, no silent
  truncation — a more complete loop than the thread's sketch.
- **Independent reviewers in disposed contexts** with an explicit anti-fabrication guard
  and a `_acted()` "text is not work" gate — a strong answer to "two optimists agreeing."
- **Deterministic ensemble merge** (Basna): reliability-weighted dedup/pick in Python,
  LLM only on genuine conflict — the code-node discipline, done right.

---

## Appendix — Key files

- **Loop / check:** `captain_claw/dubina/engine.py`, `captain_claw/flight_deck/code_routes.py`
  (`_big_job`), `code_verify.py`, `code_contract.py`, `code_consistency.py`,
  `code_honesty.py`, `quality_profile.py`, `captain_claw/output_validation.py`
- **Graph / edges:** `captain_claw/task_graph.py`, `captain_claw/plan_mode.py`,
  `captain_claw/flight_deck/horizon_plan.py`, `captain_claw/flight_deck/flow_dsl.py`,
  `flow_runner.py`, `instructions/plan_mode_complete_system_prompt.md`,
  `instructions/orchestrator_decompose_system_prompt.md`, `agent_reasoning_mixin.py`
- **Modes:** `captain_claw/flight_deck/vatra_routes.py`, `vatra_groups.py`,
  `basna_routes.py`, `council_routes.py`, `instructions/vatra/lead.md`
- **Learning edge:** `captain_claw/agent_playbook_mixin.py`, `tools/playbooks.py`,
  `instructions/playbook_distill_system_prompt.md`,
  `instructions/task_contract_planner_user_prompt.md`, `agent_reasoning_mixin.py`,
  `flight_deck/db.py`, `flight_deck/code_contract.py`, `research_contract.py`
- **Blast-radius gate / human:** `captain_claw/flight_deck/action_catalog.py`,
  `fd_dispatch.py`, `arbiter.py`, `plans.py`, `autonomy.py`, `autonomy_routes.py`,
  `config.py`, `agent_guard_mixin.py`, `agent_completion_mixin.py`, `tools/registry.py`,
  `tools/datastore.py`, `terminal_watcher.py`, `tracing.py`

---

*Methodology source: Hanako (@hanakoxbt) "Prompts → Agents → Loops → Graphs" thread +
the agent-layers course. This review maps it onto Captain Claw as of the current tree;
line numbers may drift as the code evolves.*
