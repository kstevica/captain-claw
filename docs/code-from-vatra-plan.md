# Captain Claw Code — Borrowing Vatra's Coordination Gains

> Status: **PROPOSED** (2026-07-13). Not started. Three phases, cheapest/safest first.
> Motivation: Basna/Vatra on a DeepSeek v4 Pro now beat Opus 4.8 on a light scaffold —
> the gap was closed by the **coordination + verification layer** built in the quality-
> tightening initiative (`docs/vatra-quality-tightening-plan.md`) and the group-execution
> work (`docs/vatra-execution-groups-plan.md`). **Code never received that layer.** This
> plan ports it into Code.

Related: `docs/code-basna-vatra-cross-pollination-plan.md` (the earlier, already-merged
round), `docs/vatra-quality-tightening-plan.md`, `docs/vatra-execution-groups-plan.md`,
`docs/horizon-in-basna-vatra-plan.md`.

## The gap (grounded 2026-07-13)

Code's earlier cross-pollination is **on main** — `code_routes.py` reads a `QualityProfile`
(`_load_quality`, code_routes.py:265) and honors `test_gate`, `coverage_check`, `deep_build`,
`token_budget`, delta reviews, reliability learning, and the fix-backlog. Good.

But Code missed everything Vatra grew *after* that branch. Two hard facts:

1. **Code's build is still a single `code-implementer`** — code_routes.py:1346
   (`_phase(pkey, "Building")` → one implementer). `deep_build` is best-of-N *single*
   implementers, not a team. Only **review** fans out (code_routes.py:1408, three reviewers
   via `asyncio.gather`). This is the **C4 lever explicitly reserved** in the cross-
   pollination plan and never built.
2. `grep -E 'vatra_groups|group0|facts_ledger|research_contract|consistency|honesty|
   execution_group' code_routes.py` returns **nothing**. Code consumes none of the
   coordination or calibration machinery.

What Vatra has that Code lacks (all merged to main):

| Vatra capability | Where | Code has it? |
|---|---|---|
| Group 0 Long Horizon Planner + coordination-plan gate | vatra_routes.py:794 (`_GROUP0_PLANNER_ID`) | No — Code has a plan gate but no per-agent coordination plan |
| Execution groups A→B→C→D + dependency repair | vatra_groups.py:138 (`resolve_groups`) | No |
| Group scheduling: TEAM SCHEDULE, auto-extend, honest timeouts, write-as-you-go, pull-forward | vatra_groups.py:206 (`schedule_block`), `pull_decision` | No — Code has a flat 900s per-agent bound |
| `facts_ledger` (canonical shared values + `facts` tool) | facts_ledger.py | No |
| `constraints_contract` (checkable predicates, `.contract.json`) | research_contract.py | No |
| `consistency_check` (deterministic cross-section verify) | research_consistency.py | No |
| `honesty_guard` / `output_mode` (anti-fabrication) | quality_profile.py:121 | No — Code has `_acted`/`_no_change_corrective` only |
| `block_on_critical` (bounded revise-until-clean, never discard) | quality_findings.py | Partial — Code has a 3-round fix cap + backlog |
| `quality_metrics` (unified per-run record) | quality_profile.py `build_quality_metrics()` | No — Code tracks only tokens (`_TURN_USAGE`) |

Framing: **Vatra learned to coordinate a team on interdependent work; Code is still a
one-person shop with a review committee.**

## Envelope discipline (load-bearing, unchanged from prior rounds)

Every new lever is an **opt-in `QualityProfile` flag, default-off; empty `quality` config
== byte-for-byte today's Code behavior.** New flags ride the existing profile
(`quality_profile.py`); no DB migration (Code state lives in `<project>/.code/`, VFS folder
files, and `project.json`'s `quality` key). Paid work is `TokenBudget`-gated. Blocking loops
are capped. The one exception mirrors research: `honesty_guard` defaults ON, and an explicit
`false` restores the pre-guard prompts exactly.

`honesty_guard`, `output_mode`, `block_on_critical`, and `constraints_contract` **already
exist** in `QualityProfile` (research side). Phase A is mostly *wiring Code's paths to flags
that are already defined*, not adding new schema.

---

## Phase A — Prompt & gate borrows (low risk, no topology change)

Make each existing Code agent more honest and each gate sharper. No change to the single-
implementer structure, so near-zero regression risk. Highest payoff on the weak local models
Code targets (the SW10 post-mortem: glm-5.2 / kimi2.7 claiming work it never did).

### A1 — Acceptance contract for Code (`constraints_contract`, reused)
- At plan-approval time, derive a **checkable acceptance contract** from the approved
  `.plans/*.md` — once, reason tier — using `research_contract.derive_prompt`. Predicates
  are code-flavored: `test "test_login" exists`, `command "npm run build" exits 0`,
  `endpoint POST /x returns 201`, `no literal "TODO(claude)" remains`.
- Persist `.contract.json` in the repo (like `.plans/`, committed — not gitignored).
- **Validate deterministically** after each build/fix round: `range|equals|expr` predicates
  against test output / `code_verify` results (code_verify.py already runs the suite and
  returns structured ground truth); `judge` predicates fold one reason call only for the
  ones code can't check.
- Feed CRITICAL contract violations into triage as *ground-truth targets* instead of
  reviewer prose. Reuses `research_contract.py`'s safe recursive-descent evaluator verbatim
  (no `eval`).
- Flag: reuse existing `constraints_contract`. Gate the derivation behind the plan approval
  Code already has.

### A2 — Completion-honesty guard (`honesty_guard` + `output_mode`, reused)
- Append `UNVERIFIED_GUARD_DIRECTIVE` (+ a code-specific overlay) to the implementer/fixer/
  triage prompts: never claim "tests pass", "build succeeds", or "done" without a real test
  run or a real file edit on disk. Name the class ("a completed acceptance criterion"),
  domain-agnostic.
- `output_mode`: `conservative` → an agent that can't finish a slice says so and writes a
  precise blocker note, instead of emitting plausible-but-broken code.
- Default ON (kill-switch `honesty_guard: false` restores today's prompts byte-for-byte).
- This is the prevention half of the SW10 "I fixed it" hallucination that
  `_acted`/`_no_change_corrective` (code_routes.py) only catch *after the fact*.

### A3 — Blocking gate semantics for the fix loop (`block_on_critical` + `quality_findings`)
- Wrap Code's fix loop in `quality_findings.run_gate`'s discipline: **only text/code-re-
  verifiable criticals drive rounds** (failing test, contract predicate that code can
  recheck); reviewer-opinion findings ride the checklist but never loop.
- Adopt the two safety rules verbatim: **a worsening fix round is reverted** (restore prior
  commit); at cap/budget, **keep the best commit + persist a `quality_verdict`** instead of
  only dumping a backlog. Work is never lost.
- Flag: reuse `block_on_critical` + `block_max_rounds`; when off, today's 3-round cap +
  backlog behavior is unchanged.

### A4 — Unified quality metrics (`build_quality_metrics()`, reused)
- Emit one per-turn record via the *same* `build_quality_metrics()` both engines use
  (schemas can't drift): fix rounds, escalations, tests pass/fail, contract violations,
  tokens, verdict. Persist into the session's `.code/state`/trace; surface in the Code
  header (Code already shows a run-total token chip).
- No flag (metrics are free); absent levers → absent keys, per the existing contract.

**Phase A deliverables:** new `code_contract.py` (thin adapter over `research_contract.py`
for code predicates + `code_verify` validation) and `code_honesty.py` overlay strings;
wiring in `code_routes._run_build_loop` / `_triage_reviews`; tests. No new topology.

---

## Phase B — Parallel, phased team build (finally build C4)

Turn Code's build from *one implementer* into *a scheduled team*, reusing Vatra's group
machinery almost verbatim. This is where the DeepSeek>Opus result actually lives. Gate the
whole thing behind one new flag `parallel_build`; when off, fall through to today's single
`code-implementer` path unchanged.

### B1 — Group 0 planner for Code
- Adapt `long-horizon-planner` (vatra_routes.py:794) into a **coding coordination plan**:
  decompose the approved plan into **file/module-owned slices with `depends_on` edges** —
  e.g. `schema/types` → `API handlers` → `UI` → `tests`. Each slice names its owner
  archetype (`code-implementer` / specialist), its target files, and its contract predicates
  (from A1).
- Reuse the existing **edit / Execute / Cancel gate**: Code already pauses at plan approval,
  so grafting the coordination plan into that gate is natural (mirror `plan_vatra_group0`'s
  `awaiting_plan` + `_emit_awaiting_plan` flow).

### B2 — Execution groups with barriers
- Run slices in ordered phases via `vatra_groups.resolve_groups(subtasks, arch_by_id)` —
  Group 0 = shared foundations (schema/types), then parallel coders in later groups, a
  **barrier between groups**. Dependency ordering *is* the write-conflict-avoidance mechanism
  that made the original single-implementer choice necessary — with groups, parallel writers
  never touch the same foundations concurrently.
- Per-phase git commits (Code already commits per phase: `[build]`, `[review rN]`).

### B3 — Interface ledger (`facts_ledger`, reused) — load-bearing under parallelism
- Once coders run in parallel, the ledger stops being optional: Group 0 writes the canonical
  **interface decisions** (function signatures, API shapes, DB fields, shared type names)
  into `.facts.db` in the repo; later-group coders **read them via the `facts` tool** instead
  of guessing. Conflict-preserving upsert (facts_ledger.py) means two coders can't silently
  invent incompatible signatures — a conflict is returned as content, forcing reconciliation.
- Flag: reuse `facts_ledger`; auto-enabled whenever `parallel_build` is on (a parallel build
  without a shared interface contract is the failure mode).

### B4 — Group scheduling
- Adopt the TEAM SCHEDULE prompt block (`vatra_groups.schedule_block`), **dispatch auto-
  extend** (3× cap), **honest ⏱ timeouts**, **write-as-you-go** to an `extracts/`-style
  staging dir, and **bounded pull-forward** (`pull_decision`, PULL_CAP=2). Code's flat 900s
  per-agent bound (`_dispatch_one`) is exactly the hard limit Vatra replaced with honest
  extension.
- Keep the `CLAW_CODE_AGENT` scale-machinery kill-switch (code_routes.py:849) — the group
  layer must not re-enable list/coverage reply-scanning that the SENKO2/SW10 fixes turned off.

**Phase B risk:** this rewrites the build hot path. Prototype on a branch, A/B against the
single-implementer path on 2–3 real projects (a fresh scaffold, a mid-size feature, a bugfix)
before trusting it. Success metric: fewer fix rounds + fewer interface-drift bugs at equal-
or-lower total tokens vs single implementer.

---

## Phase C — Cross-file interface consistency (code analog of `consistency_check`)

Vatra's `consistency_check` verifies numbers agree across report sections. The code analog:
**does every call site match the signature the callee actually exports?**

- Code already has `code_map.py` (AST symbol skeleton + FTS5 at `<repo>/.codemap/map.db`), so
  extraction is nearly free: pull exported symbols + their signatures and all call sites.
- Deterministic checker (pattern from `research_consistency.py`: LLM extracts, pure code
  verifies): arity/keyword mismatches, calls to symbols that don't exist, imports of missing
  names → CRITICAL findings.
- Route those criticals into the **same `block_on_critical` gate** from A3, so an interface
  mismatch blocks "done" and drives a fix round with a precise target.
- Flag: new `interface_consistency` (thorough preset once proven).
- Lower priority: build A + B first; this pays off most once parallel coders (B) can
  introduce cross-slice interface drift.

---

## Rollout & flags summary

| Flag | Phase | Preset | New or reused |
|---|---|---|---|
| `constraints_contract` (code predicates) | A1 | thorough | reused + `code_contract.py` adapter |
| `honesty_guard` / `output_mode` | A2 | on by default / — | reused + code overlay |
| `block_on_critical` / `block_max_rounds` | A3 | explicit opt-in | reused + fix-loop wiring |
| quality_metrics | A4 | always (free) | reused `build_quality_metrics()` |
| `parallel_build` | B | explicit opt-in (paid) | **new** |
| `facts_ledger` (interface ledger) | B3 | auto-on with `parallel_build` | reused |
| `interface_consistency` | C | thorough (once proven) | **new** + `code_consistency.py` over `code_map` |

**Recommended order:** Phase A now (days, near-zero risk, mostly wiring existing flags) →
Phase B on a branch with an A/B gate before merge → Phase C after B proves out.

**Test discipline:** mirror the tightening initiative — pure helpers in new modules
(`code_contract.py`, `code_consistency.py`) fully unit-tested; route files stay thin call
sites; every OFF path asserted byte-identical to today. Backend restart + FD bundle rebuild
(`cd flight-deck && npm run build`, emptyOutDir:false, commit bundle) required on deploy for
any UI knob added to `QualityControls` (scope=code).
