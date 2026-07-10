# Basna / Vatra stalled-run resume

Let a **stalled or cancelled** Basna/Vatra run be resumed from where it froze —
restore every owner/agent that already finished (no re-run, no re-spend),
re-dispatch only the missing ones, then synthesize. "Continue as if nothing
happened."

Motivation: a real Vatra run ("DIGIT Spark Grant", 8 agents) *looked* stuck in the
Synthesizing phase after the user had already spent ~$20; losing it and re-running
would have cost another ~$20. (It turned out the reporter was legitimately busy for
~28 min and finished — see the timeout note below.)

Status: **Phase 1 (backend) implemented.** Phase 2 (UI + tool actions) pending.

## Locked decisions (with the user)

- **Manual trigger, no watchdog.** No auto stall-detection. Resume is an explicit
  action (a button in Phase 2; the `/resume` endpoints now). The user judges when a
  run is stuck.
- **Any-stage recovery.** Recover a mid-run *worker* stall, not just the final
  merge/report. This is what required adding a durable per-owner checkpoint to Vatra
  (Basna already had one).

## Design

Resume = *tear down any live workers for the session → reconstruct finished work from
durable checkpoints → re-enter the normal run coroutine with `resume=True`, skipping
completed owners → synthesize.* It reuses the shipped continuation plumbing (same
`vfs_project` folder + same cast via the persisted `route`), so a resume binds the
SAME shared folder, blackboard, and datastore as the original run.

### Durable checkpoints (the enabler for "any stage")

- **Basna** already persists every agent to `basna_runs` (that's what `/recompile`
  recovers from). Reused as-is.
- **Vatra** had no per-owner table — a slice lived in the in-memory coroutine until
  the reporter wrote `vatra-slices.md`. Added a **`vatra_runs`** checkpoint table
  (`db.py`), UPSERT-keyed on `(session_id, subtask_id)`. Each owner is checkpointed
  the moment it finishes in `_dispatch_owner` (survives process death), and again in
  the finalize pass with any captured-file text merged in. Idempotent, so a
  re-dispatched owner overwrites its own row.

### Resume re-entry

Both `execute_vatra` and `execute_route` take a new `ExecuteRequest.resume` flag:

- **Vatra** — on resume, load `vatra_runs` into a map. `_dispatch_owner` short-circuits
  any owner with a `done` checkpoint (restores its output, emits a "restored from
  checkpoint" event, spends nothing). Missing/failed owners re-dispatch and
  re-checkpoint. The folder backup and the intro/review refinement rounds are skipped
  (resume just fills the gaps + reports).
- **Basna** — on resume, load prior `basna_runs` into a `(archetype_id, role)` map.
  `_dispatch_tracked` restores a matching agent (skipping dispatch) and carries the
  prior run id so scoring updates that row instead of inserting a duplicate; the save
  step inserts only freshly-dispatched agents and rebuilds `run_ids` aligned with
  `results`. If every agent already finished and only the MERGE stalled, all restore
  and it just re-merges (equivalent to `/recompile`, via the normal path).

### Endpoints

- `POST /fd/vatra/sessions/{id}/resume` — backgrounds a resumed `execute_vatra`;
  reconstructs run knobs (grouped, parallelism, datastore, folder) from the session
  config so the resumed flow matches the original.
- `POST /fd/basna/sessions/{id}/resume` — runs a resumed `execute_route` inline (like
  `/execute`) and returns the final result.

Both refuse a completed run (`status == done` and `truth` set) and tear down any
still-alive workers via `_cancel_basna_run` before re-entering.

### Per-worker timeout — deliberately NOT tightened

The plan floated a tighter per-worker LLM timeout to convert a hang into a
checkpointed failure. Investigation showed worker + reporter calls are **already**
bounded by the configurable `dispatch_timeout` (default 600s, up to 3600s), and the
observed "stall" was a *legitimately long, active* synthesis (~28 min, reading 8
files, 250k+ tokens) that **succeeded**. A tighter timeout would have **killed that
successful run** — a regression. Decision: keep the existing configurable ceiling;
resume is the recovery mechanism. Recorded so it isn't "re-fixed" later.

## Cost behaviour

Resume re-runs only the *missing* owners (or, for a merge-only stall, just the
synthesis). Restored owners spend zero tokens. The "$20 again" scenario doesn't recur.

## Phases

- **Phase 1 (DONE — backend).** `vatra_runs` table + `save_vatra_run`/`list_vatra_runs`;
  `ExecuteRequest.resume`; Vatra checkpoint writes + resume skip + gated
  backup/intro/review; Basna resume skip + aligned run-id save + score guard; both
  `/resume` endpoints. Verified: DB upsert idempotency + import/route registration.
- **Phase 2 (surface).** Resume button on any non-`done` Basna/Vatra session
  (VatraPage/BasnaPage) + store action; `resume` action on the `vatra`/`basna` tools;
  FD bundle rebuild.

## Follow-ups / known limits (Phase 2+)

- On resume, restored owners are still *spawned* (cheap) though not dispatched
  (expensive). A skip-spawn optimization would save the process-launch cost.
- Concurrency: a resumed coroutine and a still-alive stalled UI coroutine can briefly
  both touch `_run_workers[sid]`. `_cancel_basna_run` kills the old workers first so
  the old one unwinds; acceptable for manual use (user Stops, then Resumes).
- Horizon per-owner verification (opt-in) re-verifies restored slices on resume.

## Touch list (Phase 1)

- `captain_claw/flight_deck/db.py` — `vatra_runs` table + `save_vatra_run` /
  `list_vatra_runs`.
- `captain_claw/flight_deck/basna_routes.py` — `ExecuteRequest.resume`; resume restore
  in `execute_route` / `_dispatch_tracked`; aligned run-id save; score guard;
  `resume_basna` endpoint.
- `captain_claw/flight_deck/vatra_routes.py` — resume flag + checkpoint load; checkpoint
  writes in `_dispatch_owner` + finalize; gated backup/intro/review; `resume_vatra`
  endpoint.
