# Code ↔ Basna/Vatra Cross-Pollination Plan

Date: 2026-07-05. Branch: `feat/code-basna-vatra-cross-pollination`.

## Build status (updated 2026-07-05)

Everything ships behind ONE opt-in safety envelope: `quality_profile.py`. With an
absent/empty `quality` config the profile is all-off → **byte-for-byte current
behaviour** in Code, Basna, and Vatra. Presets: `off` (default), `balanced` (the
free/saving levers), `thorough` (adds the paid-but-cheap ones). A `token_budget`
caps the extra spend the opt-in levers add. 145 flight-deck tests green; MCP test
failures are pre-existing and unrelated.

**Shipped (opt-in, tested):**
- **Foundation** — `quality_profile.py` (`QualityProfile` presets + `TokenBudget`
  + `worker_produced_nothing`/`escalate_reason` helpers). Settings endpoint:
  `GET/PUT /fd/code/projects/{project}/quality`; Basna/Vatra read `quality` from
  the session config or `ExecuteRequest.quality`.
- **C1 test gate** — `code_verify.py`; runs the repo's tests after build/fix and
  feeds a failure to triage as a ground-truth finding. Zero LLM tokens.
- **C2 reliability learning** — Code records builder/fixer/planner/small-path
  outcomes into `archetype_reliability`; the router surfaces learned weights.
  Success derived from triage + test gate → no extra LLM call.
- **R2 acted-gate** — one corrective retry for a Basna/Vatra worker that produced
  nothing (no text + no file).
- **R1 Research Map** — `research_map.py` + `researchmap` tool; FTS index over the
  shared VFS folder, preamble for workers, reporter searches it past its cap.
  Reindex before/after each Basna/Vatra round.
- **C5 coverage check** — judge the approved plan vs the built repo; gaps → backlog.
- **R5 worker escalate** — `ESCALATE:` flag → one focused-retry (budget-gated).
- **R6 git snapshots** — per-round `git` commits of the research folder.
- **R7 budget parity** — `TokenBudget` caps the opt-in retries in Basna/Vatra.

**Reserved (flag defined, not yet wired — a dedicated pass; the two starred are
where "don't break current behaviour" is at real risk, so they belong in their
own carefully-tested change):**
- **C3 deep-build** (`deep_build`) — Dubina ladder over the Code build with the C1
  verifier as ground truth. Integration point: wrap the build dispatch in
  `_run_build_loop` the way Basna wraps workers in `run_worker_horizon`.
- **R4 delta rounds** (`delta_rounds`) — continuation critics/closer see only the
  new `r{N}-` files. (R1 already delivers most of the continuation token savings.)
- **R3 critic-triage** — turn horizon-closer critic findings into per-owner ordered
  fixes (Code's triage shape) instead of one blanket revision.
- **C6 continuation lineage** — root/parent/round bookkeeping + one-click
  harden/cover/simplify follow-ups (needs endpoints + frontend).
- **★ C4 parallel Vatra build** — planner emits dependency-scoped steps; independent
  steps build in parallel waves via Vatra ask/wait/board. Major Code build-path
  change; gate behind a flag so the default path is untouched.
- **★ Spawn-pipeline unification** — fold Basna `_spawn`, Vatra's mirror, and Code's
  spawn usage into one helper. Pure refactor of shared hot code; do it isolated
  with the full suite green before/after.

---

## Original proposal

## Ground truth (what the research found)

Code mode already leans on Basna's plumbing — it imports `_dispatch_one`, progress
tracking, registry/tier loading from `basna_routes.py` (code_routes.py:35-47) and
spawns archetype agents through `dubina_agents.spawn_archetype_agent`. But the
deeper machinery of each side has never crossed over:

**Code has, Basna/Vatra lack:**
- **Code Map** (code_map.py): persistent SQLite+FTS5 symbol index, incremental
  reindex by git blob hash, LLM semantic layers (overview.md / models.json /
  ui.json), a cartographer archetype that maintains it, a `codemap` query tool,
  and a preamble that forces read discipline.
- **Delta review** (code_routes.py:1143-1148): rounds 1+ only re-review the fix
  commit's diff, and the security reviewer drops out unless findings mention
  security. Massive token savings vs full re-review.
- **Triage** (code_routes.py:922-974, instructions/code/triage.md): LLM turns raw
  reviewer findings into `{needs_fix, fixer, ordered fix_instructions, findings}`
  — actionable routing, not just a verdict.
- **Acted-gates + corrective retry** (code_routes.py:977-1014): `_acted()` checks
  a dispatch made real write/edit tool calls; if not, retry with a blunt
  corrective ("files = work"). Weak-model insurance.
- **Escalation** (code_routes.py:611-621): a small-path agent can flag
  `ESCALATE: <reason>` → auto-promotion to the full plan→build pipeline.
- **Git-per-phase audit trail** (code_git.py): every phase is a commit
  (`[plan]`, `[build]`, `[review rN]`, `[fix rN]`), lock-serialized per project.
- **Backlog continuation** (code_routes.py:1019-1029): unfixed findings persist
  to `.reports/backlog.md`; "continue fixing" resumes the loop from there.
- **Per-turn token accounting** (`_TURN_USAGE`, code_routes.py:121-152) and an
  agent-run circuit breaker.

**Basna/Vatra have, Code lacks:**
- **Reliability learning** (`archetype_reliability`, db.py:1171-1283): Bayesian
  weight per (user, archetype, domain), fails count double, router reads learned
  weights as catalog hints. Code records nothing — its router picks archetypes
  blind, forever.
- **Horizon levers** (horizon_worker.py, horizon_plan.py): Lever A per-worker
  N-sample + critics + fix, Lever B final-artifact closer, Lever C verify-gated
  multi-step plan with DAG waves and re-plan. All built, none touch Code.
- **Dubina CoderVerifier** (dubina/coder.py:179): ground-truth test-command
  verification — built for exactly this, used by nobody (Horizon Phase 4
  deferred it; Code never adopted it).
- **True parallel build**: Vatra's decompose → owners → blackboard → reporter.
  Code's build phase is a single code-implementer even for "big" tasks; the
  comment "big drives the Vatra build" (code_routes.py:15) is aspirational —
  only the review phase is parallel.
- **Continuation lineage** (`_continue_run`, basna_routes.py:846-991): chained
  rounds pinned to the root VFS folder with root/parent/round bookkeeping,
  same-cast option, VFS manifest + round-filename rule.
- **Cross-agent analysis** (basna_routes.py:1460-1516): agreement / differences /
  unique insights / blind spots — blind spots seed deepen runs. Code has no
  "what did the plan promise that never got built?" check.
- **Vatra coordination primitives**: asks with depth/budget caps, `agent_wait`
  rendezvous (file or board post, ≤90s), board with note/output/narration kinds.

---

## Direction 1: Basna/Vatra → Code

### C1. Wire Dubina's CoderVerifier into the build/fix loop  ★ do first
Code's quality gate today is LLM opinion (3 reviewers + triage). Add ground
truth: after each `[build]` / `[fix rN]` commit, run the repo's test command via
`CoderVerifier` (dubina/coder.py). Failing output becomes a finding injected
into triage alongside reviewer reports — objective evidence outranks opinion.
- Test command discovery: detect (package.json scripts, pytest, Makefile) or ask
  once, persist in `.code/project.json` per folder.
- No tests found → skip silently (today's behavior), but the cartographer can
  note "untested repo" in overview.md.
- This simultaneously closes Horizon Phase 4's deferred "coder verifier for
  code-emitting subtasks" — one implementation, both systems use it.

### C2. Feed Code runs into `archetype_reliability`  ★ do first
After each run, judge outcomes and call `record_archetype_outcome`:
- Builder: did review round 0 come back with zero blocking findings? (Later:
  did tests pass — free once C1 lands.)
- Fixer: did the round's findings actually close?
- Planner: holistic judge on plan quality vs final state (mirror Vatra's
  `vatra-lead` pseudo-archetype pattern).
- Small-path archetype: success = no escalation, no error.
Then surface learned weights in the Code router prompt (router.md) the same way
Basna's catalog hints work — the router starts learning which planner/builder
combos work per domain (frontend vs API vs infra).

### C3. Horizon "deep build" (Lever A for the build phase)
Opt-in toggle per session: wrap the build dispatch in the Dubina ladder —
single pass → N-sample vote → verifier-fed fix loop → tier climb. With C1 the
verifier is ground truth (tests), which is exactly where the ladder shines.
Replaces the current blunt retry-2x-with-corrective for hard tasks. Budget-
bounded, surfaced as "stopped at rung N" like Basna's Horizon config.

### C4. Make "Vatra build" real: parallel multi-owner build for big tasks
Today's big path: one planner → one builder → parallel review. Upgrade the
build phase for multi-module plans:
- Planner emits plan steps with file/module scopes and `depends_on` (Vatra's
  `_normalize_plan` shape).
- Independent steps dispatch in parallel waves (reuse `run_dag_horizon`'s wave
  logic or Vatra's owner dispatch), each owner scoped to its files; the
  existing per-project git lock serializes commits.
- Cross-step needs go through Vatra's ask/board machinery instead of guessing;
  `agent_wait` lets a step block on a dependency's artifact.
- Commit per owner (`[build <owner>]`) keeps the audit trail.
Biggest-effort item; do after C1–C3 prove out. Start with "parallel only when
the planner marks steps independent" to dodge merge conflicts entirely.

### C5. Plan-coverage analysis → backlog
Port Basna's blind-spots idea: when triage says done, run a cheap "coverage"
judge comparing the final state against the approved plan (`.plans/*.md`) —
which plan items are unimplemented or untested? Gaps append to
`.reports/backlog.md` so "continue fixing" also picks up scope gaps, not just
review findings. Near-free: it's one LLM call and the backlog plumbing exists.

### C6. Continuation lineage for Code sessions
Adopt Basna's root/parent/round bookkeeping for chained work in the same folder
(sessions already share a folder, but chains aren't first-class). Enables
Basna-style follow-up actions as one-clicks: "harden" (security pass), "cover"
(qa-engineer test pass), "simplify" (simplifier pass) — each a `_continue_run`
analog seeding from the last session's reports.

---

## Direction 2: Code → Basna/Vatra

### R1. Research Map — generalize Code Map to VFS research projects  ★ do first
The single biggest transplant. Continuation rounds today seed workers with a
file manifest and "READ what's relevant" — every round re-reads accumulated
files, and the Vatra reporter inlines a 12k-char cap of slices. Build `vfsmap`:
- Reuse code_map.py's skeleton: SQLite + FTS5 over the VFS project's files
  (md/txt/json instead of symbols — index headings, claims, sources), file
  purposes, `overview.md` maintained by a "research-cartographer" pass at the
  end of each round (mirror Code's cartographer phase).
- A `vfsmap` tool (clone of codemap: overview/search/file) + preamble injected
  into worker prompts, same read-discipline wording.
- Reporter queries the map instead of relying on the inline cap — directly
  attacks the documented reporter-context-limit problem
  (vatra-collaborative-mode-plan.md:272).
- Payoff grows with chain length: round 5 workers find prior claims by search
  instead of re-reading 40 files.

### R2. Acted-gate + corrective retry in worker dispatch  ★ do first (tiny)
Basna already backfills empty outputs from generated files, but a worker that
narrates without writing anything still burns its slot and gets judged as a
fail. Port `_acted()` + `_no_change_corrective` into `_dispatch_one` for
artifact-producing runs: no write tool calls AND empty output → one corrective
retry. Cheap, direct hit on the weak/fast-tier failure mode.

### R3. Triage between critics and revision
Horizon closer today: strict-majority refute → one blanket revision. Code's
triage is strictly more actionable: convert critic findings into
`{needs_fix, ordered fix_instructions, which owner}` and hand each owner its
own instructions. Best fit: Vatra's per-owner closer (Phase 3.4) — instead of
"revise your slice", the owner gets concrete numbered fixes. Also gives
Basna/Vatra a findings ledger analogous to `.reports/`, which R4 can diff.

### R4. Delta-aware continuation rounds
Port Code's delta-review economics: round N+1 critics/closers see the prior
conclusion summary + only the NEW `r{N}-` files (the round-filename rule makes
the delta trivially identifiable), not the whole corpus. Same idea as
reviewing only the fix commit. Cuts continuation-chain token cost roughly in
proportion to chain length.

### R5. ESCALATE flag for workers
Code's small→big promotion, generalized: a Basna worker or Vatra owner that
recognizes the task exceeds it flags `ESCALATE: <reason>` in output. The
orchestrator re-dispatches that slice at a higher tier (or, in Vatra, converts
it into an ask/subteam). Today a struggling worker just times out or emits junk
the merge has to absorb. Pairs naturally with reliability learning: escalations
count as a soft fail for the archetype at that tier.

### R6. Git snapshots for research folders (opportunistic)
`code_git.py` is reusable as-is. Optionally `git init` VFS research projects
and commit per round (`[r2] deepen: ...`). Gives round diffs, rollback, and
provenance beyond `.vfs-meta.jsonl`. Low priority; do it when R1's cartographer
pass lands (natural commit point).

### R7. Token/budget parity
Vatra has no token ceiling (wall-clock only in design); Code has `_TURN_USAGE`.
Extract one accounting helper; enforce Horizon's `compute_budget` semantics in
Vatra runs and surface spend in the board UI like Code's usage events.

---

## Shared plumbing (both sides win)

- **Unify the spawn pipeline**: Basna `_spawn`, Vatra's mirrored copy, and
  Code's usage of `spawn_archetype_agent` → one helper with mode markers as
  parameters. Already flagged as deferred fast-follow in the Vatra plan; every
  item above touches spawn, so do this early to avoid 3-way drift.
- **Strategic (not now)**: Code's build loop IS a fixed-shape Horizon Plan
  (plan → build → review → fix). Once C1–C4 land, expressing it via
  `run_dag_horizon` with step runners would give Code re-plan-on-failure,
  budgets, and `stopped_reason` for free — but it's a big refactor; only worth
  it after the smaller grafts prove the fit.

## Suggested order

| # | Item | Effort | Why this order |
|---|------|--------|----------------|
| 1 | C1 CoderVerifier test gate | M | Ground truth beats opinion; unblocks C2/C3 quality signals; closes Horizon Phase 4 debt |
| 2 | C2 Code → reliability learning | S | One table+judge call away; router improves from day one |
| 3 | R2 acted-gate in dispatch | S | Tiny port, immediate reliability win for all ensemble runs |
| 4 | R1 Research Map (vfsmap) | L | Biggest payoff for continuation chains + reporter limits |
| 5 | R3+R4 triage + delta rounds | M | Compose with R1; big token savings on chains |
| 6 | C3 deep build (Horizon Lever A) | M | Needs C1's verifier to be worth it |
| 7 | C5+C6 coverage→backlog, lineage | S/M | Cheap once patterns exist |
| 8 | R5 ESCALATE, R7 budgets | S | Nice-to-haves, slot in anywhere |
| 9 | C4 true Vatra build | XL | Do last, informed by everything above |
