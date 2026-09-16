# Captain Claw — Story Integrity & Finalization Protocol

> Status: **P0 IMPLEMENTED** (2026-09-16) on branch `feat/story-integrity-p0`.
> Extends `docs/vatra-run-hardening-*.md`. From the reviewer's "Story Integrity and
> Finalization Protocol" + the Captain Spark test-run feedback.
>
> **What P0 ships (this build):** the story-state store, the six DETERMINISTIC
> passes (A chronology, B knowledge, D provenance, E hypothesis scope, G
> quantities/identity, H clue payoff) as a gated post-draft validator that reads a
> store EXTRACTED from the merged draft, with a bounded patch-revise loop and a
> severity→done-gate. Opt in with `quality.validation == "story_integrity"`.
> This is the deterministic 80/20: it kills age drift, paid>attempted, future-dated
> data, two-places-at-once, "the pool is one", cross-role elimination, and unpaid
> clues — cheaply, on a weak model, with no trust in the model to self-audit.
>
> **Deferred to P1/P2** (see §11): the pre-draft Group-A/B pipeline (constraints,
> ledgers, causal outline, physical-simulation gate), the model passes (C physical
> replay on final prose, F research-claims, I narrative economy, J emotional
> fairness), and the §9 claim-attacker research gate + professional-procedure pass +
> full plot-simplification authority. P0 runs the validator POST-draft over the
> assembled file; P1 moves state-building before the draft.

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

Result (`analysis`): `quality_verdict` may be `"story_integrity_failed"`;
`analysis.integrity` = `{rounds, hard_fixed, remaining, hard, major, soft, passes}`;
`analysis.blocking.hard[]`/`.major[]` = `{pass, scene, reason}` (client shows these
with the kept draft on a `status:"error"` run). A `deliverable.integrity.md` audit
file is written but never merged into the story.

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

## 6. Revision loop

Locate the earliest scene, repair the CAUSE (redesign mechanism / fix state), update ledgers, revalidate downstream, rewrite affected prose, re-run passes. Bounded by `validation_max_rounds`. The loop may **simplify the plot** (range instead of exact time; simpler mechanism; change an action; seed or cut a twist) — never paper over an impossible sequence with one sentence.

## 9. Research-reality gate (P2)

An internal "claim-attacker" agent (reason tier) adversarially researches each material story claim with Vatra's existing tools and writes the result into the constraint ledger. Never turn "possible" into "established", "not found" into "proved absent", or a current finding into a historical one without a bridge. If research can't support the plot timing, change the plot — don't compress a multi-day process into minutes or cherry-pick a weaker source.

## 11. Phased plan

- **P0 (done)** — story-state store + deterministic passes A/B/D/E/G/H as a gated post-draft validator with rewrite authority + severity→done-gate.
- **P1** — Group-B physical-simulation gate + model passes C/F/I/J on the reason tier + the causal outline as a Group-A artifact + the seeding check (Pass J).
- **P2** — the §9 claim-attacker research gate + the professional-procedure pass + full plot-simplification authority in the revise loop.

## 12. Acceptance test

Re-run the review briefs on the weak tier, no hand-repair. `done` only if the store is populated and the validator reports zero hard errors: no future-dated fact cited earlier; no character in two places at once; no age/quantity/name mismatch; no cross-role elimination; every high-salience clue paid off. Otherwise `status:"error"` + `analysis.blocking.hard[]` with the draft kept.
