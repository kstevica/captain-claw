# Vatra/Basna Quality Tightening — Implementation Plan

> Status: **ALL 7 INCREMENTS SHIPPED** (2026-07-10, commits 9d22918 → 57a406f).
> 1 honesty guard + output modes · 2 consistency check · 3 quality metrics ·
> 4 facts ledger · 5 constraints contract · 6 blocking gate · 7 FD surface.
> 84 new tests; suite green (8 pre-existing unrelated failures). Backend restart
> required on deploy. One scope refinement vs the text below: the blocking gate
> loops ONLY on text-re-verifiable consistency criticals; contract/ledger
> criticals ride the checklist + verdict but never drive rounds (a prose
> revision can't fix ledger-level values). Consistency-knob name shipped as
> `consistency_max_values` (not max_figures).

Date: 2026-07-10. Findings: `docs/vatra-quality-tightening-findings.md`.
Goal: weak-model ensemble runs that beat a strong single model on a light scaffold,
by adding the four structural levers the DIGIT SPARK analysis showed matter most:
a shared facts ledger, deterministic consistency checking, severity-based blocking,
and a calibration-honest default posture.

## Decisions locked (2026-07-10)

1. **Honesty guard is ON by default** (`honesty_guard: true`). The user can turn it
   off (`{"quality": {"honesty_guard": false}}` → byte-for-byte today's prompts).
   This is the ONE deliberate exception to the envelope's "absent config = current
   behaviour" guarantee; everything else in this plan stays strictly opt-in.
2. All other new levers ride the existing `quality_profile` envelope: absent/empty
   `quality` config keeps them off; paid work is `TokenBudget`-gated; blocking
   loops are capped (the CLARIFY_CAP pattern).
3. Advisory-first rollout: every new check lands advisory (findings recorded, run
   completes); blocking is a separate explicit flag (`block_on_critical`).
4. No DB migrations. New state rides `analysis` JSON on `basna_sessions`, VFS-folder
   files (`.facts.db`, `.contract.json`, audit `.md` files), and env-context tools —
   same pattern as datastore/research-map.
5. Code mode untouched. All new directives/wiring are research-side (Basna/Vatra
   workers, reporter, merge). Shared modules stay pure so Code can adopt later.

## New flags (quality_profile.py)

| Flag | Default | Preset | Cost | What |
|---|---|---|---|---|
| `honesty_guard` | **True** | n/a (independent of presets) | free, prompt-only | Decoupled unverified-guard + placeholder/estimate policy + reporter honesty overlay |
| `output_mode` | `""` | none | free, prompt-only | `""` (today) \| `"complete"` \| `"conservative"` |
| `consistency_check` | False | `thorough` | ~1–3 fast-tier calls | Extract figures → deterministic identity/arithmetic verify → one correction |
| `facts_ledger` | False | `thorough` | free-ish (tool + directives) | Shared machine-readable facts store; workers read/write via `facts` tool |
| `constraints_contract` | False | `thorough` | ~1 reason-tier call | R9 extension: hard rules with checkable predicates, persisted per folder |
| `block_on_critical` | False | none (explicit, like `claim_check`) | paid loop, capped | CRITICAL findings loop back (reporter revision) until clean, capped, budget-gated |

New knobs: `consistency_max_figures: int = 40`, `block_max_rounds: int = 2`.

`honesty_guard` stays OUT of `_BOOL_FLAGS`/`any_enabled` — `any_enabled` means "any
opt-in lever on" and a default-true flag would break skip-fast semantics (currently
unused beyond the property, but keep the meaning). `from_dict`: absent key → True;
explicit `false` → False; presets never touch it.

---

## Increment 1 — Honesty rebalance + output modes (prompt-only, ship first)

The near-free increment that directly attacks the report's §2.2 mechanism
(completeness pressure suppressing calibration).

**1a. `honesty_guard` flag.**
- `quality_profile.py`: add the field + special-case in `from_dict` (absent → True).
- Basna: `basna_routes.py:3231-3232` — split the bundle. `judgment_ledger` appends
  only `JUDGMENT_LEDGER_DIRECTIVE`; a new independent
  `if quality.honesty_guard: base_prompt += UNVERIFIED_GUARD_DIRECTIVE`.
- Vatra: `vatra_routes.py:918-919` — gate the shared_context fold-in on
  `quality.honesty_guard` instead of `quality.judgment_ledger` (shared_context
  already reaches every owner AND the reporter).

**1b. Placeholder/estimate policy** (report §5.3, generic wording). Extend
`UNVERIFIED_GUARD_DIRECTIVE` in place with a compact second paragraph:
- unknown facts only the requester can supply → `[TO BE PROVIDED: <what>]`, never a
  plausible stand-in;
- estimates are allowed but labeled `(estimate — basis: <one line>)`;
- identifiers, names, financials, third-party entities: never invented — a correct
  placeholder beats a plausible fabrication.

**1c. Reporter/synthesis honesty overlay** — new `REPORTER_HONESTY_DIRECTIVE` in
`quality_profile.py`, appended when `honesty_guard` is on:
- Vatra reporter: in `_run_reporter` prompt assembly (`vatra_routes.py:~2039`,
  where shared_context enforcement already lands).
- Basna synthesizer: appended to the `_llm_synthesize` prompt
  (`basna_routes.py:2032-2050`).
- Text (sketch): *"Exception to 'resolve it and move on': when the evidence does
  not actually resolve a contradiction between pieces, or a section rests on an
  assumption or estimate, resolve it provisionally AND record it in a final
  '**Unresolved & assumptions**' section — item, the call you made, what would
  confirm it. Do not silently absorb disagreements between specialists. Never pad
  a section with invented specifics to look finished."*
- `instructions/vatra/reporter.md` is NOT edited — the overlay supersedes at
  runtime, so `honesty_guard: false` restores today's prompts byte-for-byte.

**1d. `output_mode`.** String knob, prompt assembly only, both engines (worker
prompt builders + reporter/synthesis):
- `"conservative"` → review-copy posture: only ledger/source/input-backed specifics
  stated as fact; everything else placeholder or labeled estimate; the
  Unresolved/assumptions register is mandatory; completeness reported, not forced.
- `"complete"` → today's maximal-completion push, but estimates must be labeled
  (composes with 1b).
- `""` → no directive (today).

**Tests:** `from_dict` default/override/preset-independence; prompt-assembly units
asserting directive presence/absence per flag for both engines. Audit existing
prompt-content tests for breakage (the split in 1a changes what `judgment_ledger`
alone injects).

**Escape hatch from day one:** all of this is settable via the session/request
`quality` JSON — no frontend needed to turn the guard off (FD surface is
Increment 7).

---

## Increment 2 — `consistency_check` (Pass A: deterministic internal consistency)

New shared module `captain_claw/flight_deck/research_consistency.py` (mirrors
`research_verify.py`'s shape: prompt builder + tolerant parser + pure logic +
audit markdown + summary line):

- `extract_prompt(deliverable, max_figures)` — fast-tier LLM extracts every
  load-bearing figure/date/identifier occurrence to JSON:
  `[{id, kind: figure|date|identifier, label, value_raw, normalized, unit,
  quote(≤120 chars), relation?: {type: sum|percent_of|identity|difference,
  operands: [labels], stated_result}}]`. The LLM only *extracts and labels* — it
  does no arithmetic.
- `parse_figures()` — fence/bracket tolerant parse (the `parse_findings` pattern),
  coerce/drop malformed rows.
- `verify(figures, ledger_rows=None)` — **pure code, no LLM**:
  - identity: same normalized label, different normalized values → CRITICAL;
  - relations: evaluate sum/percent/difference with rounding tolerance
    (max(0.5%, 1 unit)) → mismatch = CRITICAL;
  - ledger cross-check (when Increment 4 is on): text value ≠ ledger value for the
    same key → CRITICAL; text asserts a `to_be_completed`/`assumed` ledger fact as
    plain fact → MAJOR;
  - date sanity (end before start etc. when relations declare it) → MAJOR.
- `audit_markdown()` → `<deliverable>.consistency.md`; `summary_line()` tally.

**Wiring** (both engines, run BEFORE claim check so internal fixes land before
external verification):
- Vatra: in the post-reporter block near `vatra_routes.py:1542`;
- Basna: post-merge near `basna_routes.py:3462`.
- If findings: ONE correction dispatch with an R3-style numbered checklist
  (`_triage_feedback` shape), collapsed-revision guard reused
  (`horizon_worker.py:405` pattern), then ONE re-extract+verify to confirm. All
  budget-gated (`_budget.can_afford(_retry_est)`).
- Findings + fixed-count → `analysis.consistency`; audit file written even when
  clean (like the fact-check ledger).

**Tests:** `verify()` is the star — fixture tests straight from the report's
regression suite (§10): Σ(lines) ≠ total, same-label figure drift (549k vs 157k
class), percent-of-base recompute, tolerance edges, ledger mismatch. Parser tests.
This IS the report's regression suite, generalized.

---

## Increment 3 — Quality metrics persistence (no flag, rides everything)

Assemble `analysis.quality_metrics` at run end in both engines:
`{claims_checked, confirmed, refuted, unverifiable, hedged}` (from R8 when it ran),
`{consistency_critical, consistency_major, consistency_fixed}` (Increment 2),
`{gaps_major, gaps_minor}` (coverage/rubric), `{escalations, acted_retries,
block_rounds, quality_verdict, budget_stopped_reason}`. Absent lever → absent keys.
Persisted with the session row (`vatra_routes.py:1574/1651` and the Basna
equivalent). Pure additive JSON; surfaced in FD in Increment 7. This gives the
report's §8 metrics and, later, better reliability-learning signals.

---

## Increment 4 — `facts_ledger` (RC1: single source of truth)

**Store** — new `captain_claw/flight_deck/facts_ledger.py`: small SQLite at
`vfs:<project>/.facts.db` (research_map's folder-SQLite pattern; NOT inside the
datastore, which stays LLM-schema'd and persistence-oriented):
- `facts(key PK, value, unit, status CHECK(verified|derived|estimated|assumed|
  to_be_completed), provenance, confidence REAL, computed_from, updated_by,
  updated_at)`;
- `facts_conflicts(key, offered_value, offered_by, existing_value, created_at)`.
- API: `upsert()` (conflict-aware: same key, different normalized value → keep the
  original, record the conflict, RETURN the conflict to the writer so it can
  reconcile or `ask`), `get()`, `list()`, `conflicts()`, `dump_markdown()` (compact
  table for the reporter), row cap ~200 (directive says load-bearing only).

**Tool** — `captain_claw/tools/facts.py` (actions `set`/`get`/`list`), context via
env `CLAW_FACTS_VFS=<project>` mirroring `CLAW_DATASTORE_VFS`
(`vatra_routes.py:276`; Basna spawn env equivalently). Registered unconditionally,
inert without the env — the `vatra` tool pattern.

**Directives** (in `quality_profile.py`, injected when flag on):
- `FACTS_LEDGER_DIRECTIVE` (workers): every load-bearing number/date/identifier you
  establish → `facts set` with status + provenance (URL, file, or "derived from
  <keys>"); any number another piece owns → `facts get`, never restated from
  memory; a conflict return means reconcile or post an `ask` — don't overwrite.
- Reporter: `dump_markdown()` inlined into the reporter/synthesis prompt (it's
  small and structured) + *"every figure in the deliverable must match the ledger;
  ledger conflicts and `to_be_completed`/`assumed` facts go to Unresolved &
  assumptions, not stated as plain fact."*
- Lead (`lead.md` assembly, Vatra): one line telling it to name the canonical
  ledger keys pieces must share in `shared_context` when the task is quantitative.

**Integration:** Increment 2's `verify(ledger_rows=...)` lights up ledger
cross-checking. R8's fact-checker gets `facts list` output as claimed-provenance
context.

**Tests:** ledger module units (upsert/conflict/dump/cap), tool action tests,
prompt assembly, consistency-vs-ledger fixtures.

---

## Increment 5 — `constraints_contract` (Pass B: rules, not just completeness)

Extend `research_rubric.py` (same derive pass; one reason-tier call emits both when
both flags on):
- `derive_contract_prompt` → `{rubric: [...], constraints: [{id, text, severity,
  check: {type: range|max|min|equals|requires|sum_eq|date_before|enum,
  key?|keys?, value?, expr?}}]}` — `key`s reference facts-ledger keys.
- Tiny safe evaluator for `expr` (numeric + − × ÷ and comparisons over ledger
  values only — a ~40-line recursive parser, **no `eval()`**).
- **Persistence/reuse:** write `vfs:<project>/.contract.json` at derive time;
  `_continue_run` chains load it instead of re-deriving (the folder-chain already
  persists per project) — repeat tasks against the same rulebook stop re-deriving
  their rules per run (RC3). User-editable file by design.
- **Validation:** post-deliverable — deterministic where `check` resolves against
  the ledger; non-resolvable constraints fall into the existing coverage-judge
  prompt. Violations → severity findings (`analysis.gaps` shape + Increment 6
  model). Constraint items injected into worker/reporter prompts like
  `rubric_directive` today.
- Degrades gracefully without `facts_ledger` (LLM-judged only); the plan pairs
  them in `thorough`.

**Tests:** evaluator units (expressions, div-by-zero, missing keys), derive-parse,
deterministic validation fixtures (range breach, sum_eq vs ledger, date_before).

---

## Increment 6 — `block_on_critical` (the blocking gate)

- Small `quality_findings.py`: `Finding{source: claim_check|consistency|contract|
  coverage, severity: critical|major|minor, detail, location}` + mappers from each
  lever's native output (refuted claim → CRITICAL; asserted-unverifiable → MAJOR;
  consistency identity/arithmetic → CRITICAL; contract per its own severity;
  rubric missing → MAJOR).
- Loop, after all enabled checks ran and their single corrections applied:
  while CRITICALs remain AND rounds < `block_max_rounds` (default 2) AND
  `_budget.can_afford`: build one triaged checklist across all criticals
  (`_triage_feedback`), ONE reporter/synthesis revision (Vatra re-drives the
  reporter agent; Basna a closer-style `_revise`), then re-run ONLY the checks
  (consistency re-extract+verify; claim check re-verifies only previously-refuted
  claims; contract re-validates deterministically — cheap).
- Exit clean → done. Cap/budget exhausted with CRITICALs → run still completes
  (never discard work — the dispatch-resilience philosophy), but
  `analysis.quality_verdict = "critical_findings_remain"` with the surviving list;
  FD badge in Increment 7. Collapsed-revision guard on every revision.
- Flag stays out of all presets (it multiplies paid passes) — explicit opt-in,
  ideally with `token_budget`, exactly like `claim_check`/`deep_build`.

**Tests:** loop logic with stubbed checkers — converges in 1, caps at
`block_max_rounds`, budget-stops, verdict recorded, work never lost.

---

## Increment 7 — FD surface

- Run config (Basna/Vatra panels): honesty-guard toggle (default on), output-mode
  select, advanced quality flags + `token_budget` field.
- Session detail: `quality_metrics` panel, links to `.consistency.md` /
  `.fact-check.md` audits, `quality_verdict` badge, unresolved/assumptions count.
- flight-deck: `npm run build` + commit the bundle (per repo build discipline);
  backend restart required after each increment lands on prod.

---

## Build order & effort

| # | Increment | Effort | Depends on | Why this order |
|---|---|---|---|---|
| 1 | Honesty rebalance + output_mode | S | — | Near-free, immediate calibration win; the default-on decision ships alone so its diff is auditable |
| 2 | consistency_check | M | — | Highest-leverage new check; works standalone on deliverable text |
| 3 | metrics persistence | S | 2 (partly) | Trivial once tallies exist; establishes the measurement baseline |
| 4 | facts_ledger | L | — | Biggest structural piece; lights up 2's ledger mode |
| 5 | constraints_contract | M | 4 (soft) | Reuses R9 pattern; deterministic half needs the ledger |
| 6 | block_on_critical | M | 2, (4, 5) | Needs findings worth blocking on |
| 7 | FD surface | S/M | 1–6 | One frontend pass, one bundle rebuild |

Each increment: tests green before/after with `quality` absent (regression
guarantee), plus the increment's own units. Commit per increment.

## Risks

- **Prompt bloat on small models** — directives stack (vfs + datastore + map +
  rubric + honesty + ledger + mode). Mitigation: directives stay ≤10 lines each,
  only stack with flags; watch worker token counts in cost events.
- **Weak-model extraction quality (Increment 2)** — deterministic side is safe
  against false positives (quotes anchor every finding); false negatives just mean
  fewer catches. Tolerant parser drops malformed rows rather than failing.
- **Revision degradation** — every correction/blocking revision reuses the
  collapsed-revision guard; blocking is capped and never discards prior text.
- **Ledger misuse** (workers dumping trivia) — row cap + "load-bearing only"
  directive; conflicts are surfaced, never silently overwritten.
- **Basna/Vatra double-wiring drift** — known spawn-unify debt. All new logic
  lives in shared pure modules (`research_consistency`, `facts_ledger`,
  `quality_findings`); route files get thin call sites only.
- **Honesty default-on changes existing outputs** (hedged language, Unresolved
  sections appear) — intended; kill-switch documented; called out in the commit
  message and release note.
