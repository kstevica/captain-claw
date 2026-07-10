# Vatra/Basna Quality Tightening — Phase 1 Findings

Date: 2026-07-10. Source: gap analysis of the Opus "DIGIT SPARK Agent System Upgrade
Brief" (4-run grant-application evaluation) against Captain Claw's Vatra/Basna
machinery on `main`. Goal: better results with weaker models — surpassing a strong
single model on a lighter scaffold (the report's "Opus + Cowork" baseline).

> Phase 2 (implementation plan with flags) follows from this doc. Nothing here is
> code yet.

## The report's thesis, restated for Captain Claw

The report's decisive finding (§2.2): the same model produced a confidently-wrong
application in a multi-agent pipeline and a calibrated, honest one in a light
scaffold. The pipeline didn't lack knowledge — **it suppressed the model's own
calibration** by rewarding completeness. Its fact checker rubber-stamped arithmetic
contradictions, regulatory violations, and fabrications because it checked narrative
plausibility only.

Captain Claw is generic (not grant-specific), so the mapping targets the report's
*root causes* (RC1–RC7), not its DIGIT SPARK specifics.

## Scorecard — report root causes vs Captain Claw today

| # | Report root cause | Captain Claw status | Evidence |
|---|---|---|---|
| RC1 | No single source of truth (facts ledger) | **MISSING** | Blackboard = prose posts capped at 30k chars (`vatra_board`, db.py:257); asks = prose; `shared_context` = plan-time prose conventions, never updated at runtime; opt-in datastore is persistence-oriented, not canonical values; R11 "judgment ledger" is a prose prompt directive (quality_profile.py:293), not a data structure |
| RC2a | Fact checker: no arithmetic/internal-consistency pass | **MISSING** | No cross-section numeric-consistency gate exists anywhere; the reporter is told to "resolve [contradictions] sensibly … don't narrate the disagreement" (reporter.md:23) — a single prose pass, unchecked |
| RC2b | Fact checker: no regulatory/rulebook pass | **MISSING** | R9 rubric (research_rubric.py) is a *completeness* checklist, not constraints with checkable predicates; nothing validates outputs against hard rules |
| RC2c | Fact checker: no provenance pass | **PARTIAL** | R8 claim check (research_verify.py) is exactly this — verdicts confirmed/refuted/unverifiable, forced hedges, audit ledger `*.fact-check.md` — but it's in **no preset**, advisory (one verify + one correction), capped at 8 claims |
| RC2d | No blocking on critical failures | **MISSING** | Only "no usable slice" hard-fails a Vatra run (vatra_routes.py:1453). Every quality gate is advisory: one-shot retry or in-place revision; nothing loops back to the owning stage until severity clears |
| RC3 | No encoded domain rulebook | **MISSING** | R9 derives its rubric per run and discards it; no persisted, reusable rulebook artifact per project/domain; `shared_context` pins conventions but is prose and plan-time-frozen |
| RC4 | Completeness pressure overrides calibration | **CONFIRMED — the defaults are the risk case** | Reporter: "Finish it — completely" (reporter.md:26), "state the resolved position — don't narrate the disagreement" (reporter.md:23); Basna synthesis: "do not narrate the disagreement" (basna_routes.py:2047); Vatra worker: "never stop to say you're missing teammate input — produce your best version" (vatra_routes.py:1897-1899); review round: "ADD anything important that's missing — do real extra work" (1941-42). Counterweights (honesty guard + judgment ledger) are opt-in and OFF by default |
| RC5 | Research not provenance-tagged | **PARTIAL** | R10 source corpus saves full page + URL + epoch timestamp (web_fetch.py:52) — page-level, opt-in (`thorough` only). No claim-level provenance in any data model; archetype SOPs (deep-researcher: "inline citations (URL per claim)") demand citations but nothing enforces or checks them |
| RC6 | No fabrication policy | **PARTIAL** | `UNVERIFIED_GUARD_DIRECTIVE` (quality_profile.py:314-324) is nearly verbatim the report's §7 guardrail — but gated behind `judgment_ledger` (balanced+; off by default). Base Basna worker prompt has **zero** anti-fabrication text (basna_routes.py:1675-1708). No placeholder convention (`[TO BE COMPLETED]`), no estimate-labeling policy, no estimate-vs-placeholder-vs-must-source distinction |
| RC7 | Locale not enforced | **MISSING (minor)** | Nothing ties output language/number/date format to the task's jurisdiction; the Lead *may* pin it in shared_context but isn't prompted to |

Also from the report:
- **§5.2 deterministic computation** ("compute money as code, never in prose") — no equivalent lever; workers do arithmetic in prose. Workers have shell access, so a compute-as-code directive + ledger is feasible.
- **§6 output modes** (draft_to_finish vs review_copy) — no conservatism toggle exists. `off|balanced|thorough` bundles quality levers, not the completeness-vs-correctness trade.
- **§8 metrics** — cost accounting is rich (pricing.py:134-178) but there is **no persisted fabrication rate, consistency count, or critical-findings count** per run. The fact-check tally lives only in the per-deliverable audit file.
- **Adversarial reviewer** — EXISTS (Horizon closer critic panel, horizon_worker.py:332; fact-checker archetype with refute-first stance, archetypes.json:99) but the router treats the adversarial role as optional ("Add it only when the task's correctness actually matters", router.md:28), and the closer is gated by HorizonConfig, separate from the quality envelope.

## What Captain Claw already has that the report asks for

Credit where due — much of §5/§7 exists, shipped opt-in in the quality envelope:

- **R8 claim check** ≈ Fact Checker Pass C: web-verifying fact-checker, structured
  verdicts, forced hedging of unconfirmable specifics, non-destructive audit ledger.
- **R9 rubric contract** ≈ the completeness half of a rulebook: derived once,
  injected into every worker + reporter, coverage-scored field-by-field with
  major/minor severities into `analysis.gaps`.
- **R11 judgment ledger + honesty guard** ≈ §7 prompt guardrails: explicit hard-call
  enumeration + "do not assert the unconfirmable."
- **R10 source corpus** ≈ provenance raw material (page-level).
- **R12 intent brief** ≈ scope fidelity (original intent stays authoritative).
- **Horizon closer + R3 critic triage** ≈ adversarial review with actionable,
  deduped fix checklists; anti-collapse guard on revisions.
- **Acted-gate/escalate (R2/R5), reliability learning, TokenBudget** — worker-level
  resilience the report doesn't even ask for.

The gap is therefore NOT "build a fact checker" — it's **four structural upgrades
plus a defaults/rebalancing problem**.

## The seven findings, ranked by leverage

### F1 — No shared facts ledger (RC1) — the biggest architectural gap
Everything crossing between workers is prose. When worker B builds on worker A's
numbers, B re-derives or re-states them from prose; nothing guarantees a value is
identical everywhere it appears. This is the root cause of the report's most
damning failure class (self-contradictory budgets) and it is structural: no prompt
fixes it.

Direction: a per-run, machine-readable facts store (key → {value, status ∈
{verified, estimated, assumed, derived, to_be_completed}, provenance, confidence,
computed_from}) that workers write via a tool and MUST read numbers from; the
reporter renders from it; validators check against it. The opt-in shared datastore
(vfs `.datastore`) is the natural substrate — it already exists, is folder-scoped,
and the Lead already knows how to pin schemas.

### F2 — No deterministic consistency pass (RC2a)
The one thing every run of the report's pipeline failed and no LLM opinion catches
reliably: Σ(parts) == total, the same figure identical across sections, derived
values recomputing correctly. With F1's ledger this becomes near-free: extract
figure occurrences (cheap LLM extract to schema) → verify identity/arithmetic in
code (deterministic). Catches the €549k-vs-€157k class of failure regardless of
domain.

### F3 — Nothing blocks (RC2d)
All quality gates are advisory. A refuted claim, a major coverage gap, and an
arithmetic contradiction all reach the final deliverable; they're recorded, not
enforced. Direction: a `block_on_critical` mode — findings get severities
(CRITICAL/MAJOR/MINOR); CRITICAL loops back to the owning stage (owner revision or
reporter re-pass) with a bounded cap (the CLARIFY_CAP=2 pattern already exists for
exactly this shape). Budget-gated like everything else.

### F4 — Completeness pressure with the counterweight off by default (RC4, RC6)
The prompts actively reproduce the report's failure mechanism: reporters must
"finish completely" and hide disagreements; workers must never stop for missing
input; the honesty guard that counters this is off by default and bundled behind
an unrelated flag (`judgment_ledger`). Directions:
- Decouple `UNVERIFIED_GUARD_DIRECTIVE` from judgment_ledger; make it always-on
  (it is free, prompt-only, domain-agnostic) or on in every preset including a new
  default.
- Rebalance the reporter prompt: keep "complete", but replace "don't narrate the
  disagreement" with "resolve what the evidence resolves; surface what it doesn't
  in a labeled **Unresolved/Assumptions** section."
- Add the placeholder/estimate policy from the report §5.3: applicant-only unknowns
  → labeled placeholder; estimates → labeled with basis; identifiers/entities →
  never invented. Generic wording, one directive.

### F5 — No constraints contract / reusable rulebook (RC2b, RC3)
R9 proves the shape works (derive once, inject everywhere, score against it) but
only for completeness. Direction: extend to a **constraints contract** — hard rules
with checkable predicates (ranges, caps, relationships between ledger fields,
mandatory items) derived at run start, persisted per project for reuse across runs,
validated in the F2/F3 pass. Where a constraint references ledger fields, the check
is deterministic.

### F6 — Provenance is page-level, claims are unbound (RC5)
R10 stores pages; nothing binds a claim to its source. Full claim-level provenance
is heavy; the pragmatic middle: (a) ledger entries carry provenance (F1 gives the
slot), (b) worker directive "any load-bearing specific gets its source inline
(URL or file)", (c) R8's audit ledger already records evidence per checked claim —
persist its tally as run metrics.

### F7 — Output modes and metrics (§6, §8)
- One `output_mode: complete | conservative` knob (report: draft_to_finish vs
  review_copy) — mostly prompt assembly, cheap to add, directly user-facing.
- Persist per-run quality metrics into the session row: claims
  checked/refuted/hedged, consistency failures found/fixed, CRITICAL/MAJOR counts,
  % placeholders. Everything already computed somewhere; nothing is stored where
  trends can be seen. This also feeds reliability learning with better signals.

## Constraints for Phase 2 (locked by the existing architecture)

1. **Everything ships behind the `quality_profile` envelope** — absent/empty config
   must remain byte-for-byte current behavior (the module's founding guarantee).
   Exception to discuss: making the honesty guard default-on is a deliberate,
   flagged behavior change.
2. Paid levers stay budget-gated via `TokenBudget`; blocking loops need caps
   (CLARIFY_CAP pattern) so runs still terminate.
3. Both engines share wiring where possible (Basna `_dispatch_one` vs Vatra's
   mirror — the known spawn-unification debt makes double-wiring likely for now).
4. Advisory-first rollout: each new gate lands advisory, promoted to blocking by
   flag once observed on real runs.

## Suggested Phase 2 scope (to be planned, not yet designed)

| Lever | Closes | Rough shape | Flag (proposed) |
|---|---|---|---|
| Facts ledger | F1, F6 | datastore-backed facts table + `facts` tool + directives | `facts_ledger` |
| Consistency check | F2 | extract-to-schema + deterministic verify after reporter | `consistency_check` |
| Blocking severities | F3 | severity model + bounded loop-back to owner/reporter | `block_on_critical` |
| Constraints contract | F5 | R9 extension: rules + predicates, persisted per project | `constraints_contract` |
| Honesty rebalance | F4 | decouple guard, reporter prompt fix, placeholder policy | `honesty_guard` (default-on candidate) |
| Output modes | F7 | prompt assembly switch | `output_mode` |
| Quality metrics | F7 | persist tallies to session analysis | (rides other flags) |

Priority mirrors the report's: F1+F2+F3 are the correctness-critical core
(ledger → deterministic checks → blocking); F4 is near-free and immediate; F5–F7
follow.
