# Vatra execution groups (ordered phases + bounded clarification loop)

Opt-in Vatra orchestration where agents run in ordered phases (A→B→C→D) with a
barrier between groups, instead of all-at-once. A later group already sees
everything earlier groups posted, so it replaces most of the runtime `vatra wait`
guessing with a deterministic pipeline. A bounded clarification loop lets a later
agent ask an earlier one for more, gated by the Lead.

## Locked decisions (2026-07)
1. **Rollout:** opt-in mode, default off. Today's intro→main→review all-at-once
   stays the default — nothing existing changes unless `execution_groups` is on.
2. **Group tags:** archetype **preset** is the floor; the **Lead may push a subtask
   to a LATER group** but never earlier; untagged archetypes → middle (B).
3. **Loop-back:** on an approved clarification, re-run **only the named earlier
   owner(s)**, then re-run the requester. Not the whole group/sequence.
4. **Cap:** **2 total** approved loop-backs per run (global ceiling).

## Group presets (overridable per archetype via a `group` field in archetypes.json)
- **A (first)** — Research & Intelligence (deep-researcher, market-scanner,
  fact-checker), architects, planners, cartographers, visual extractors.
- **B (middle, default)** — build / consolidate / write: code-implementer,
  quick-dirty, data-analyst, editor-writer, comms, most others.
- **C (last)** — reviewers, debuggers, QA, security, report-builder, git-operator,
  simplifier, UI/brand reviewers.
- **D** — reserved; used only if an archetype/Lead assigns it.

The pipeline runs the DISTINCT groups present, ascending (so a team of {A,C} runs
two phases, A then C). Within a group, owners run in parallel, subject to the
`Max parallel` gate.

## Architecture
- `vatra_groups.py` (pure, tested): `archetype_group(arch)->int` (explicit
  `group` field → role/family heuristics → middle), `GROUP_LETTERS`,
  `effective_group(subtask, arch)` = `max(archetype_floor, lead_override)`,
  `group_label(ord)`.
- `_normalize_plan`: parse an optional per-subtask `group` (A–D), clamped to the
  archetype floor.
- `execute_vatra`: when `execution_groups` is on, replace the flat main round with
  a group loop — for each group ascending: dispatch its owners (parallel, gated),
  barrier, then resolve clarification requests (Increment 2) before the next
  group. Intro/review rounds are skipped in grouped mode (group order supersedes
  them); the reporter runs at the end unchanged.
- Opt-in flag: `execution_groups: bool` on VatraExecuteRequest/ExecuteRequest,
  default False. UI toggle next to Max parallel.

### Clarification loop (Increment 2)
- A group-B+ owner that is blocked on missing data ends with a marker
  `REQUEST: <earlier-role> — <what it needs>` (prompt-taught, like ESCALATE).
- After a group's barrier, parse requests; the **Lead** approves/denies each (one
  reason-tier call). For each approved request while `loop_backs < 2`:
  re-dispatch the named earlier owner(s) with the ask, then re-dispatch the
  requester; `loop_backs += 1`. Denials + cap exhaustion are logged.

## Increments
1. **Foundation (this):** group resolver + presets, opt-in flag, grouped pipeline
   (ordered phases, no loop yet), UI toggle. Delivers "researchers first, report
   builders last." Fully back-compatible (off by default).
2. **Clarification loop:** REQUEST marker + Lead approve/deny + targeted re-run +
   global cap.
3. **Polish (SHIPPED):** per-phase display in the live panel — grouped runs
   section the "Agents working" cards by phase A→B→C→D (each owner's progress
   events carry a `group` letter; flat mode is unchanged). The Lead is taught the
   optional push-later-only `group` field in `lead.md`. Docs + tests.

## Safety
Off by default; when off, `execute_vatra` takes exactly today's path. Grouped mode
reuses the existing `_dispatch_owner`, board, cost, and max-parallel plumbing.
