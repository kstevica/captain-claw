# Captain Claw — Story Integrity & Finalization Protocol

> Status: **P0 + P1 + P2 IMPLEMENTED** (2026-09-16) on branch `feat/story-integrity-p0`.
> Extends `docs/vatra-run-hardening-*.md`. From the reviewer's "Story Integrity and
> Finalization Protocol" + the Captain Spark test-run feedback. Opt in with
> `quality.validation == "story_integrity"`.
>
> **P0 — deterministic passes (`story_state.py`).** The story-state store + the six
> DETERMINISTIC passes (A chronology, B knowledge, D provenance, E hypothesis scope,
> G quantities/identity, H clue payoff) over a store EXTRACTED from the merged draft,
> a bounded patch-revise loop, and a severity→done-gate. Kills age drift,
> paid>attempted, future-dated data, two-places-at-once, "the pool is one",
> cross-role elimination, and unpaid clues — cheaply, no trust in the model.
>
> **P1 — model passes (`story_passes.py`).** C physical mechanism, F research
> claims, I narrative economy, J emotional-fairness/seeding — reason-tier judgment
> over the merged draft + store, run in and across the revise loop, feeding the same
> severity gate (I is clamped to soft; C/F/J may block). Plus the pre-draft
> `STORY_INTEGRITY_DIRECTIVE` folded into the team's shared context (build the causal
> outline + scene plan, simulate the mechanism, one continuous artifact — riding the
> shipped strict_deps/require_inputs/single-writer machinery).
>
> **P2 — CA claim-attacker + PR professional-procedure passes** (`story_passes.py`),
> plus a plot-simplification escalation in the revise loop (round 2+ is told to
> simplify: a time range, a simpler mechanism, a changed action, a seeded-or-cut
> twist). The shipped CA pass is a **knowledge-only** adversarial judgment pass (no
> tool research, no ledger write); a tool-using research gate that writes a
> constraint ledger (§9's fuller form) remains future work.
>
> **Fail-safe design (verified):** every check must fire only on a contradiction
> clearly present in the state; sparse/absent extraction must not false-block a good
> story. Two adversarial-verification workflows (one per phase) found and fixed 5
> P0 + 4 P1/P2 fail-safe/correctness bugs before merge.

## How P0 maps onto the hardening build

P0 reuses everything the run-hardening work already shipped and **deepens
`canon_pass` into a full validator**:

| Protocol piece | Implementation |
|---|---|
| Story-state store (§2) | `captain_claw/flight_deck/story_state.py` — `parse_store`/`merge`, the five+one collections (characters, timeline, evidence, claims, hypotheses, clues). |
| Deterministic passes A/B/D/E/G/H (§4) | `story_state.check_*` pure functions over the store; `verify()` runs all six. |
| Validator run with rewrite authority (§1, §6) | `story_state.run_validator` — extract → verify → bounded patch-revise (anchored find/replace; may simplify the plot) → re-verify; `validation_max_rounds`. |
| Severity → done-gate (§5) | `bucket()` splits hard/major/soft; hard blocks `done` under `quality.gate_blocks_done` → `_finish_blocked` with verdict `story_integrity_failed`. |
| Result contract (§7) | `analysis.integrity` = `{rounds, hard_fixed, remaining, …}`; `analysis.blocking.hard[]`/`.major[]` = `{pass, scene, reason}`; a `.integrity.md` audit as a non-deliverable file. |
| Clean output / certification (§8) | Already enforced by the write guard + delivery contract + `deliverable_kind:"fiction"`; the gate refuses `done` while hard errors remain. |
| Umbrella opt-in (§7) | `quality.validation == "story_integrity"` supersedes `canon_pass`; `canon_pass` alone stays the lighter check. |
| Weak-model rules (§10) | Deterministic checks are pure code (no tokens, no miscount); extraction + revision run on the `qa` role tier / `qa_tier`. |

## Request / result contract (client boundary)

Opt in on `/fd/vatra/start` (inherited at approve):

```json
"quality": { "profile": "long_form", "deliverable_kind": "fiction",
             "gate_blocks_done": true, "block_on_critical": true,
             "token_budget": 4000000, "qa_tier": "reason",
             "validation": "story_integrity", "validation_max_rounds": 2 },
"deliverable": { "path": "<name>.md", "kind": "fiction",
                 "section_regex": "^##\\s+Chapter\\s+\\S" },
"role_tiers": { "lead":"reason","planner":"reason","clarify":"reason","reporter":"reason","qa":"reason" },
"dispatch_timeout": 1200
```

Result (`analysis`): `quality_verdict` may be `"story_integrity_failed"` (only when a
**deterministic** hard survives); `analysis.integrity` = `{rounds, hard_fixed, remaining,
hard, blocking, major, soft, scores, passes}` where `blocking` is the count of gating
(deterministic) hard errors — `hard` can exceed `blocking` when a model pass raised a
hard that doesn't gate; `analysis.scores` = `{integrity, grade, continuity, plausibility,
grounding, craft}`; `analysis.blocking.hard[]`/`.major[]`/`.advisory[]` = `{pass, scene,
reason}` (`.advisory[]` carries surviving model-pass hards that did not gate; client shows
these with the kept draft). A `deliverable.integrity.md` audit file is written but never
merged into the story. **Client rule:** treat `status:"error"` + `quality_verdict:
"story_integrity_failed"` as do-not-export; a `done` run with a low `scores.grade` or
`analysis.blocking.advisory[]` is exportable but worth a human look.

---

_The full reviewer protocol (verbatim) and the phased plan follow below for P1/P2._

## 1. Core architecture — three pieces

1. **A story-state store** (the §3 ledgers): structured session state that agents WRITE as they plan and draft, and that deterministic validators READ. Not prose. It is the spine — everything validates against it.
2. **A staged pipeline** (research → state → outline → simulation → draft → validate), enforced by the existing grouped-execution machinery. Each stage declares its inputs; a stage cannot start until its inputs exist (`require_inputs`), and an agent missing an input must **block, never fabricate a substitute**.
3. **A mandatory validator run** after the draft, with **authority to rewrite scenes**, that loops until zero hard errors (or a bounded cap) and then certifies. Not a report — a *fixing* pass.

Reuse, don't rebuild. The hardening build already has `canon_pass`, `gate_blocks_done`, `block_on_critical`, `require_deliverable_file`, `require_inputs`, `strict_deps`, `role_tiers`, the deterministic assembly, the deliverable contract, the write guard, and the `analysis` result block. This protocol **deepens `canon_pass` into a full validator** and **adds the state store + the pre-draft stages**.

## 2. The story-state store (§3)

Five collections (P0 also carries `clues`): characters, timeline, evidence, claims, hypotheses, clues. Agents emit patches; the store is authoritative; prose must not contradict it. Ages are DERIVED from dates. Elimination is per-role; never collapse across roles. (Field-level schema in `story_state.py`.)

## 3. The five stages → execution groups (P1)

Group A — Constraints & state (researcher + story-architect build the ledgers, causal outline, scene plan; no prose). Group B — Physical simulation (replay the crime minute-by-minute; PASS or redesign; draft cannot start while the sim fails). Group C — Draft (one writer, one continuous artifact, binding every exact figure to the store). Group D — Validate & revise (the mandatory validator; loops, then certifies). `strict_deps` sequences the groups; `require_inputs` blocks any stage whose declared inputs are absent.

## 4. The ten passes — DETERMINISTIC vs MODEL

Deterministic (P0, code over the store): A chronology, B knowledge, D provenance, E hypothesis scope, G quantities/identity, H clue payoff. Model (P1, reason-tier agent): C physical replay on final prose, F research-claims, I narrative economy, J emotional fairness. Each finding: pass, severity, earliest scene, proposed fix.

## 5. Severity gate

Hard → blocks finalization (`status:"error"`, draft kept, verdict `story_integrity_failed`, `analysis.blocking.hard[]`). Major → revise in the loop; if unresolved, `analysis.blocking.major[]` (allows `done`). Soft → fix if cheap; never block.

> **Gate refinement (2026-09-16, "Story run 7"):** only **deterministic** hard findings
> (A/B/D/E/G/H — fail-safe by construction) gate `done`. **Model-pass** hard findings
> (C/F/I/J/CA/PR) are weak-model judgments: they surface, drive revision, and lower the
> quality scores, but **never block on their own** — a single hallucinated model objection
> must not false-fail a good draft (a real run's excellent story was blocked by one PR-pass
> hard whose premise the text contradicted). Findings carry `origin: "deterministic" | "model"`
> (`story_state.is_deterministic()`, pass-id fallback); `story_state.blocking_findings()` is the
> gating set. `run_validator` returns `result["blocking"]` (deterministic hard, drives the
> gate) alongside `result["hard"]` (every hard, for reporting). Surviving model hards appear
> as `analysis.blocking.advisory[]`. The revise-loop keep-guard protects the **blocking**
> count (never grow the deterministic-hard set), not all-hard.

### Quality scores (advisory; never gate)

`story_state.score()` emits 0–100 scores from the surviving findings (`analysis.scores`,
also in `analysis.integrity.scores`):

| Score | From | Meaning |
|---|---|---|
| `continuity` | A,B,D,E,G,H (deterministic) | internal consistency — the trustworthy backbone |
| `plausibility` | C, PR (model) | physical + procedural realism |
| `grounding` | F, CA (model) | research / claim discipline (no overclaiming) |
| `craft` | I, J (model) | narrative economy + fair-play seeding |
| `integrity` | composite | `0.7·continuity + 0.1·(each model dim)` → headline |

Each dimension = 100 − severity-weighted penalties (hard 25 / major 10 / soft 3), floored at
0; a dimension whose passes never ran reports `null` (not a hollow 100). `grade` = clean ≥85 /
sound ≥70 / caution ≥50 / weak. The continuity weight (0.7) keeps a deterministically-airtight
story's headline high even when the weak model quibbles. Future scores (not yet built):
`coverage` (manifest parts/sections satisfied), `hedge_discipline` (deterministic scan for
overclaim phrases), `source_diversity` / `recency` (research runs), `readability`.

## 6. Revision loop

Locate the earliest scene, repair the CAUSE (redesign mechanism / fix state), update ledgers, revalidate downstream, rewrite affected prose, re-run passes. Bounded by `validation_max_rounds`. The loop may **simplify the plot** (range instead of exact time; simpler mechanism; change an action; seed or cut a twist) — never paper over an impossible sequence with one sentence.

## 9. Research-reality gate (P2)

> Shipped as a **knowledge-only** adversarial judgment pass (`story_passes.py`, pass
> CA): a reason-tier agent attacks each material claim from its own knowledge and
> emits findings. The fuller form below — a claim-attacker that RESEARCHES with tools
> and writes a persisted constraint ledger downstream agents read — remains future
> work (the existing `claim_check` R8 web-fact-checker is the nearest tool-using lever).

An internal "claim-attacker" agent (reason tier) adversarially researches each material story claim with Vatra's existing tools and writes the result into the constraint ledger. Never turn "possible" into "established", "not found" into "proved absent", or a current finding into a historical one without a bridge. If research can't support the plot timing, change the plot — don't compress a multi-day process into minutes or cherry-pick a weaker source.

## 11. Phased plan

- **P0 (done)** — story-state store + deterministic passes A/B/D/E/G/H as a gated post-draft validator with rewrite authority + severity→done-gate.
- **P1 (done)** — model passes C/F/I/J on the reason tier (post-draft, in the revise loop) + the pre-draft causal-outline/simulation staging as a shared-context directive riding strict_deps/require_inputs + the seeding check (Pass J, downgraded to non-blocking when the draft is windowed).
- **P2 (done)** — the claim-attacker pass (CA, knowledge-only) + the professional-procedure pass (PR) + plot-simplification escalation in the revise loop.
- **Remaining future work** — a true pre-draft physical-simulation GATE as a distinct dispatch-loop stage (P1 delivers it as a post-draft blocking pass + pre-draft guidance) and the tool-using claim-attacker research gate that writes a persisted constraint ledger (§9's fuller form; the shipped CA pass is knowledge-only).

## 12. Acceptance test

Re-run the review briefs on the weak tier, no hand-repair. `done` only if the store is populated and the validator reports zero hard errors: no future-dated fact cited earlier; no character in two places at once; no age/quantity/name mismatch; no cross-role elimination; every high-salience clue paid off. Otherwise `status:"error"` + `analysis.blocking.hard[]` with the draft kept.
