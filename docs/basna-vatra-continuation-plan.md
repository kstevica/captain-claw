# Basna / Vatra continuation rounds

Add real **continuation** to Basna and Vatra: a finished run can be carried forward
into additional rounds that build on its conclusion, **in the same VFS folder, on
the same accumulated data**. The next round respawns fresh agents (no warm reuse —
the architecture tears agents down at run end), reusing the **same cast** of
archetypes by default, with continuity carried by the shared VFS folder + the prior
conclusion + (Vatra) the blackboard.

Status: Phase 1 implemented (backend + agent tool + Flight Deck UI). Phases 2–3 pending.

## Why

Both modes already have a follow-up primitive, but neither continues properly:

- Basna `_deepen_run()` — `basna_routes.py:820`. Seeds a follow-up from the prior
  `truth` + `analysis.blind_spots`.
- Vatra `_fill_gaps_run()` — `vatra_routes.py:1801`. Seeds from the prior report +
  `analysis.gaps`.

Three gaps:

1. **New folder every round (core bug).** The VFS project is derived from the run's
   *own* session id — hardcoded `f"basna-{sid8}"` (`basna_routes.py:1946`) and
   `_vfs_project(sid)` (`vatra_routes.py:113`, used in `_vatra_env` at `:146`). A
   follow-up creates a fresh session → fresh folder, so round 2's agents **cannot
   see round 1's VFS files**. Only the `truth` text (± a `prior-synthesis.md` copy)
   carries over; all accumulated working data is orphaned in the old folder.
2. **Gated on gaps only.** Both raise `400` when there are no blind spots / coverage
   gaps. No path for a free-form "continue the research / next chapter / act on the
   conclusion".
3. **No chaining.** Single fire-and-forget, one `parent_session_id`, no round
   counter, no stop condition, no lineage.

## Decisions

- **Respawn fresh, never warm-reuse.** Agents are fully torn down at run end
  (process killed + workspace `rmtree`). Continuity = VFS folder + conclusion +
  blackboard, not live processes.
- **Same cast by default.** A continuation reuses the parent's `route.selected`
  archetype set/roles; re-routing stays available per call. (Locked with user.)
- **Explicit per-round.** Each next round is triggered by the user/agent with an
  instruction. Auto-loop with stop conditions is deferred to Phase 2. (Locked.)
- **One chain → one VFS folder**, pinned to the *root* session's project name.
- **No schema changes** — lineage rides in the existing JSON `config`.

## Design

### 1. Folder inheritance (the crux)

Carry a `vfs_project` override through the run so a continuation binds to the root
folder instead of computing from its own sid.

- Add `vfs_project: str | None` to `ExecuteRequest` (and persist in `config`).
- Basna `_spawn` (`basna_routes.py:1921`): replace hardcoded
  `f"basna-{sid8}"` with `body.vfs_project or f"basna-{sid8}"`.
- Vatra: thread an override into `_vfs_project` / `_vatra_env`
  (`vatra_routes.py:107`, `:137`) and `_vfs_directive` so all prompts pin the
  inherited folder. Default unchanged when absent → backward compatible.

A continuation sets `vfs_project = "<mode>-<root_session_id[:8]>"`, so rounds 2..N
all read/write the round-1 folder.

### 2. Unified `_continue_run`

Generalize `_deepen_run` / `_fill_gaps_run` into one helper:

```
_continue_run(parent_session_id, instruction, kind, same_cast=True)
```

- Inherits the root `vfs_project`.
- Seeds the new intent with: prior `truth` (preview + full-text file as today),
  `analysis`, and a **VFS manifest** — a listing of files already in the shared
  folder with the directive "read and build on these; do not recreate".
- Writes lineage into `config`: `root_session_id`, `parent_session_id`,
  `round` (parent.round + 1), `kind`, `vfs_project`.
- `same_cast=True`: reuse parent `route.selected` (skip the router). `False`:
  re-route on the continuation instruction.

`kind` ∈:

- `continue` — extend forward with a free-form instruction (new).
- `deepen` / `fill-gaps` — blind spots / coverage gaps (existing behavior; now
  folder-continuous). Still default the instruction from `analysis` when none given.
- `revise` — improve the existing deliverable per instruction (new).

### 3. Collision guard

Same folder across rounds risks round 2 clobbering round 1's `output.md`. The
continuation prompt must mandate **round-scoped filenames** (`r<round>-<name>`) and
"read prior files, write new ones — don't overwrite". `.vfs-meta.jsonl` authorship
is append-only (last-writer-wins), so only file *content* is at risk.

### 4. Surface

- Basna tool (`tools/basna.py`): add `continue` action (instruction, kind, same_cast).
  Keep `deepen` as a thin alias.
- Vatra tool (`tools/vatra.py`): add `continue` action; keep fill-gaps alias.
- Endpoints: `POST /sessions/{id}/continue` for both routers, mirroring the existing
  `/deepen` endpoint at `basna_routes.py:905`.

## Phases

- **Phase 1 (DONE).** Folder inheritance + `_continue_run` + `continue`/`revise`
  tool actions + endpoints + collision-guard prompt. Same cast, explicit per-round.
  Shipped:
  - `ExecuteRequest.vfs_project` threaded through both Basna spawn sites + Vatra via
    a per-run `_run_vfs_project` override map (`_vfs_project` consults it).
  - Basna `_continue_run(parent, instruction, kind, same_cast)` + `_deepen_run` alias;
    Vatra `_continue_run` + `_fill_gaps_run` alias. Lineage in `config`
    (`root_session_id`, `parent_session_id`, `round`, `vfs_project`).
  - `_vfs_manifest` + `_round_filename_rule` seed every continuation prompt.
  - Endpoints: `POST /fd/basna/sessions/{id}/continue`, `/fd/basna/agent/continue`
    (mode-aware → Vatra), `/fd/vatra/sessions/{id}/continue`.
  - `basna` tool gains a `continue` action (works for Basna + Vatra parents).
  - Existing `deepen` / `fill-gaps` now also inherit the root folder + same cast.
  - Flight Deck UI: `continueSession` store action (`basnaStore.ts`) +
    a "Continue" panel under the truth in `BasnaPage.tsx` (instruction textarea,
    mode-aware kind select, "Same team" toggle). Bundle rebuilt + committed.
- **Phase 2.** Round loop with stop conditions (N rounds / confidence threshold /
  coverage-dry / budget); chain-listing helper grouping sessions by
  `root_session_id` for the UI; optional "closer judges done".
- **Phase 3.** Vatra blackboard carryover (seed the new run's `vatra_board` with
  prior outputs so agents see history, not just files); UI lineage / round view.

## Touch list (Phase 1)

- `captain_claw/flight_deck/basna_routes.py` — `ExecuteRequest`, `_spawn` env,
  `_deepen_run` → `_continue_run`, new `/continue` endpoint.
- `captain_claw/flight_deck/vatra_routes.py` — `_vfs_project`/`_vatra_env`/
  `_vfs_directive` override, `_fill_gaps_run` → `_continue_run`, `/continue`.
- `captain_claw/tools/basna.py`, `captain_claw/tools/vatra.py` — `continue` action.
- Prompt templates (`instructions/vatra/…`, Basna dispatch prompt builder) — VFS
  manifest + round-scoped-filename directive.
