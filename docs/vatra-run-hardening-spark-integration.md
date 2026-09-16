# Captain Spark — integrating the Vatra run-hardening (front-end instructions)

> Audience: the Captain Spark session (the research/fiction front-end that drives
> Vatra over Captain Claw's HTTP API). This tells you exactly what to send and how
> to read the result now that Captain Claw has been hardened for long-form
> deliverables on weak models. You do not need the Captain Claw source to follow
> this — every field and behaviour below is stable API surface.
>
> Companion docs (Captain Claw side): `docs/vatra-run-hardening-incident.md` (the
> "story run 3" post-mortem) and `docs/vatra-run-hardening-plan.md` (the design).

---

## 0. TL;DR

Captain Claw now makes a Vatra run correct on a fast/weak model **structurally**,
not just by prompting. You opt in per run with three new request fields —
`deliverable`, `role_tiers`, `dispatch_timeout` — and a `quality` profile. In
return you get: no placeholder/pointer deliverables, deterministic assembly into
one file, waits that don't expire on a live producer, a continuity gate, and a
run that ends `status: "error"` (with the work kept) instead of shipping a note
as the deliverable.

Two changes are **on by default and need nothing from you** — the write-boundary
guard and the honest action log. Everything else is opt-in.

---

## 1. Prerequisite — is the hardening deployed?

None of this applies until the Captain Claw instance you talk to is running the
`feat/vatra-run-hardening` build (merged + Flight Deck restarted; no DB
migrations). If it is not deployed, the new request fields are simply ignored and
you get today's behaviour.

Quick check: send a run with `quality: {"profile": "long_form"}`. On a hardened
build the finished session's `analysis` will contain a `deliverable` block (see
§5). On an old build it won't.

Deployment-level kill switch (operator, not per-request): `CLAW_WRITE_GUARD=0`
turns the default-on write guard off. Leave it on.

---

## 2. Which entry path — you are already on the right one

Captain Claw exposes two ways to start a Vatra run. The new fields are honoured
on only one of them today, and **you already use that one.**

`backend/app/fd_client.py` drives the two-step JWT flow (its own header comment
spells it out): `POST /fd/vatra/start` → upload files → poll to `awaiting_plan` →
`POST /fd/vatra/plan/approve {plan: null}` → poll to `done`/`error`. That path
honours `deliverable`, `role_tiers`, `quality`, and `dispatch_timeout`. Good — no
migration off `/fd/vatra/agent/start` is needed (that headless path drops the new
fields, but you don't use it).

**One consequence that makes this easy:** your `approve()` sends only
`{session_id, plan, execution_groups}` — it does NOT resend `tiers`/`quality`.
That's fine, because **everything you send on `/fd/vatra/start` is now persisted
to the session config and inherited at approve.** So you add the new fields to
your `start()` call and leave `approve()` almost untouched.

---

## 3. The request recipe (fiction / long-form)

> **Where to change your code.** Your `FlightDeckClient.start(...)` in
> `backend/app/fd_client.py` already forwards `quality`, `tiers`, and `env_vars`
> into the `/fd/vatra/start` body. Add three optional params —
> `deliverable`, `role_tiers`, `dispatch_timeout` — and drop them into `body` the
> same way (`if deliverable: body["deliverable"] = deliverable`, etc.). Thread
> them from `runs.py` where you already build `quality`/`tiers`. Your `approve()`
> needs no change for these (they're inherited from config), but see §3c for the
> fiction owner-mapping option, and make sure `execution_groups` is on (§3d).

### 3a. `POST /fd/vatra/start` — the body to build

```json
{
  "intent": "Write a fair-play mystery novella: <brief>",
  "title": "Fair Measure",
  "max_agents": 6,
  "tiers": {
    "reason":   {"provider": "openai", "model": "<stronger-model>", "api_key": "…", "base_url": "…", "input_ctx": 200000, "output_ctx": 32768},
    "balanced": {"provider": "deepseek", "model": "deepseek-v4.1-flash", "api_key": "…", "base_url": "…", "input_ctx": 128000, "output_ctx": 32768, "weak": true},
    "fast":     {"provider": "deepseek", "model": "deepseek-v4.1-flash", "api_key": "…", "base_url": "…", "output_ctx": 32768, "weak": true}
  },
  "execution_groups": true,
  "dispatch_timeout": 1200,
  "role_tiers": {"lead": "reason", "planner": "reason", "clarify": "reason", "reporter": "reason", "qa": "reason"},
  "deliverable": {
    "path": "fair-measure.md",
    "kind": "fiction",
    "min_bytes": 60000,
    "min_sections": 10,
    "parts": [
      {"path": "part-one.md", "order": 1, "range": "ch1-5",  "min_bytes": 24000},
      {"path": "part-two.md", "order": 2, "range": "ch6-10", "min_bytes": 24000}
    ],
    "seam_owner": 1
  },
  "quality": {
    "profile": "long_form",
    "deliverable_kind": "fiction",
    "canon_pass": true,
    "gate_blocks_done": true,
    "block_on_critical": true,
    "token_budget": 4000000,
    "deliverable_min_chars": 60000,
    "deliverable_min_sections": 10,
    "qa_tier": "reason"
  }
}
```

Response: `{"session_id": "...", "status": "planning"}`.

### 3b. `parts` and the concat model — read this before choosing fiction

Captain Claw assembles the deliverable by **concatenating the declared parts in
order**. A part's `owner` is a **subtask id**, which does not exist until the Lead
decomposes (after `/start`). Two ways to fill parts:

- **Derived (`quality.derive_manifest`, already on in `long_form`).** Captain Claw
  makes **one part per subtask** and concatenates them. This is exactly right for a
  **research report or multi-section document**, where every subtask produces a
  section of the whole. For those, send only `deliverable: {path, kind, min_bytes,
  min_sections}` (no `parts`) and change nothing in `approve()`. Done.
- **Explicit parts.** You declare the parts and map each `owner` to a subtask id.
  Needed for **fiction**, because a fiction run's subtasks are NOT all prose — a
  story-bible / research / fact-check subtask must not be concatenated into the
  novella. Derive would over-include them. See §3c.

### 3c. Fiction: map the writer subtasks (an `approve()` change)

Two options, cheapest first:

- **Single writer (simplest).** Have the Lead produce ONE writer subtask for the
  whole manuscript (Captain Claw's `lead.md` now nudges "one continuous artifact,
  one writer"). Then you don't need parts at all: send `deliverable: {path,
  kind: "fiction", min_bytes, min_sections}` with `derive_manifest` on — but to
  avoid concatenating the non-prose subtasks, prefer the explicit form below even
  here, declaring the single writer's file as the only part.
- **Explicit parts, mapped at approve (robust).** Your `_poll_loop` already fetches
  the plan (`get_plan(session_id)` / `route.subtasks`) at `awaiting_plan`. Before
  approving, pick the writer subtasks from `route.subtasks` (match
  `owner_archetype_id == "editor-writer"`, or your writer archetype), build
  `deliverable.parts` mapping each to its `range`/`order`, set `seam_owner`, and
  **resend `deliverable` on `approve()`**. Your `approve()` currently sends
  `{session_id, plan, execution_groups}`; add `deliverable` (and, if you want to
  change them, `role_tiers`/`quality`) to that body:

  ```python
  # in FlightDeckClient.approve(...), when you have a fiction deliverable:
  body = {"session_id": session_id, "plan": plan, "execution_groups": eg,
          "deliverable": deliverable_with_owners}   # owners = writer subtask ids
  ```

Response of approve: `{"session_id": "...", "status": "running"}`, then your poll
loop continues as today.

### 3d. `execution_groups` must be on

`strict_deps` (writer-before-writer sequencing) and the deterministic
gap-closing clarify grant only apply in **grouped** mode. You already pass
`execution_groups` on `approve()` from `self._execution_groups`
(`settings.vatra_execution_groups`) — make sure that setting is `true` for
long-form runs.

> **Tiers are request-scoped (secrets), never persisted.** Your `approve()` omits
> `tiers`, so the RUN uses the **service account's saved Library tier set**, not
> the per-run `tiers` you sent on `start()` (those were only used by the Group-0
> planner). If you want a specific per-run tier map (and the `weak` flag) on the
> actual owners, either configure the service account's active tier set to match,
> or start resending `tiers` on `approve()`. `quality`/`deliverable`/`role_tiers`/
> `dispatch_timeout` do NOT have this problem — they persist at `start()`.

---

## 4. Field reference

### The three new top-level fields

| Field | Type | Where | What it does |
|---|---|---|---|
| `deliverable` | object | start + approve | Declares the target file, ordered parts, owners, ranges, seam owner. Drives deterministic assembly, the strict write tier, and the done-gate. See schema below. |
| `role_tiers` | object | start + approve | `{lead, planner, clarify, reporter, qa}` → a tier NAME from your `tiers` map. Put the load-bearing roles on a stronger model while specialists stay on Flash. Unset → each role uses its default tier. |
| `dispatch_timeout` | number (s) | start + approve | Per-dispatch wall. Long-form needs minutes: use ~1200. Default 600. |

### `deliverable` schema

| Key | Meaning |
|---|---|
| `path` | The single assembled deliverable filename (extension required; `.md` added if missing). |
| `kind` | `""` \| `"document"` \| `"fiction"`. `fiction` marks parts sequential. |
| `min_bytes` / `min_sections` | Size / section-count floor the done-gate enforces. |
| `section_regex` | Optional custom heading regex if your manuscripts don't use `## Chapter N`. |
| `parts[]` | `{path, order, owner (subtask id), range ("ch6-7" / "chapters six to seven"), min_bytes, sequential}`. |
| `seam_owner` | The subtask id (or the integer index into `parts`) that owns the seam and writes last. |

### `quality` — the `long_form` preset vs the paid flags

`quality.profile: "long_form"` turns on the FREE structural levers:
`write_guard`, `derive_manifest`, `resolve_pointer_truth`,
`require_deliverable_file`, `require_inputs`, `wait_heartbeat`, `strict_deps`,
`clarify_dep_grant`, `push_deps`, `synthesis_emit: concat_then_smooth`, plus the
`balanced` research levers. It deliberately does **not** enable the paid QA
levers, so a preset never surprise-spends.

Set these **explicitly** (they cost tokens — pair with `token_budget`):

| Flag | Effect |
|---|---|
| `canon_pass: true` | Continuity check over the assembled draft (ages, whereabouts per scene, alibis, sums, tallies, chronology, duplicate scenes). Writes a `.canon.md` audit, fixes by anchored patches. |
| `gate_blocks_done: true` | A surviving canon **critical** blocks completion → `status: "error"` (work kept). |
| `block_on_critical: true` | The existing bounded revise-until-clean loop over consistency findings. |
| `token_budget: <n>` | Hard output-token ceiling shared by all paid levers. Always set it with the paid flags. |
| `deliverable_kind: "fiction"` | Suppresses the in-prose "Unresolved & assumptions" ledger so QA stays out of the narrative. Set this AND `deliverable.kind`. |
| `qa_tier: "reason"` | Tier for the canon extractor/reviser (or set it via `role_tiers.qa`). |

Other useful knobs: `deliverable_min_chars`, `deliverable_min_sections`,
`wait_max_total_s` (alive-wait ceiling; 0 → the run's `dispatch_timeout`),
`clarify_cap` (max gap-grants per run).

### `tiers[<name>].weak`

Add `"weak": true` to the tier entry your Flash model uses. It surfaces
`CLAW_MODEL_WEAK` to the worker so its runtime leans on the deterministic
backstops. It is an explicit signal — Captain Claw never guesses "weak" from a
model name. Also give the Flash tier `output_ctx >= 32768` so a chapter-sized
append write isn't cut off.

---

## 5. Reading the result

Poll `GET /fd/basna/sessions/{session_id}`. Terminal states:

- **`status == "done"`** — the deliverable is good. Your existing `_poll_loop`
  already stores `session.get("truth")` on `done`; with the hardening, `truth` on
  a `done` run is now the REAL assembled deliverable bytes (never a pointer note),
  so the incident's core failure is fixed with **no change to your done path**.
  Read it from:
  - `truth` (the assembled deliverable text), OR
  - `GET /fd/vfs/read?project=<config.vfs_project>&path=<config.deliverable_resolved.path>`
    (use this for a very large deliverable; `truth` may be capped). `config` is the
    session row's `config` JSON; it now carries `vfs_project` and
    `deliverable_resolved`.
  - `files[]` lists the deliverable and part files; a VFS-written file carries a
    `"vfs"` key with its `vfs:` path.
- **`status == "error"` with an `analysis.quality_verdict`** — treat this as **"do
  not export"**, NOT a crash. The run assembled work but a gate failed. `truth`
  and `files` are still populated (best assembly), so you can show/continue it,
  but you must not present it as the finished deliverable. Verdicts:
  - `"deliverable_missing"` — the file was missing, undersized, a placeholder, or
    parts collided/gapped. `analysis.deliverable.reasons` says which.
  - `"critical_findings_remain"` — the canon gate found unresolved contradictions.
    `analysis.blocking.canon` lists them.

> **Behaviour change to plan for in `runs.py`.** Your `_poll_loop` today does, on
> `status == "error"`: `db.finish_run(run_id, "error", None, "Flight Deck
> reported an error")` and emails failure — it DISCARDS `truth`. With the gates on
> (`require_deliverable_file`, `gate_blocks_done`), `error` becomes the signal for
> "assembled but a gate blocked it", and those errors carry a usable `truth` +
> `analysis.quality_verdict`. So a nearly-complete manuscript that failed the
> canon gate would currently be reported to your user as a bare failure with no
> content. Decide which you want:
> - **Safe (no change):** keep discarding — you never ship a gate-failed
>   deliverable, but you lose the partial.
> - **Better (small change):** on `error`, if `session["truth"]` is non-empty and
>   `analysis.quality_verdict` is set, store `truth` as a **draft** and surface the
>   verdict/reasons, or auto-fire a `continue`/`fill-gaps` round (which inherits
>   your `deliverable`/`quality`) to fix it. This turns a hard failure into a
>   "needs one more pass".

`analysis` JSON keys you can rely on: `deliverable` (verdict, reasons, bytes,
sections, parts[]), `canon` (chunks, initial/remaining findings), `consistency`,
`quality_verdict`, `blocking`, `tiers_used`, `gaps`.

To improve an errored (or thin) run, call the continuation endpoints
(`/fd/vatra/sessions/{id}/continue` or `/fill-gaps`) — they inherit the
`deliverable`, `role_tiers`, `dispatch_timeout`, and `quality` you set.

---

## 6. What you can stop doing (and what to keep)

Captain Claw now enforces in code several things you had been patching at the
prompt level:

- **Stop** relying on the prompt alone to prevent a pointer/placeholder
  deliverable — the write guard, the pointer-note rejection, and the done-gate
  handle it.
- **Stop** hand-tuning board-wait timing hopes — waits now extend while a
  producer is alive.

**Keep** your Spark-side delivery contract and the silent continuity-pass /
fair-play prompts. They compose with the structural backstops (prompt + structure
together), and the continuity prompts still guide the writers even before the
canon gate checks the merged draft. Just guarantee your fiction templates emit
`## Chapter N` headings (or pass `deliverable.section_regex`) — chunking,
`min_sections`, and the range checks depend on it.

---

## 7. Acceptance test (re-run "story run 3" on Flash)

Run the same fiction brief with the §3 request on your Flash tiers and assert,
with no hand-repair:

1. **No placeholder / no truncated path** — every part in
   `analysis.deliverable.parts[*].landed` is true; no file matches a
   `[written to disk: …]` marker.
2. **One assembled file; `truth` is its content** — `analysis.deliverable.verdict
   == "ok"`, `files[]` has the `{kind: "vfs", path: "vfs:…/fair-measure.md"}`
   entry, and `GET /fd/vfs/read` returns it.
3. **No duplicated scene / no missing chapter** — no `duplicate_chapter`,
   `missing_chapter`, or `duplicate_scene` in `analysis.blocking`.
4. **Canon ran on the merged draft, gate held** — `analysis.canon.remaining == 0`
   with `status == "done"`, else `status == "error"` + `analysis.blocking.canon`.
5. **No wait expired on a live producer** — the run completes without an owner
   auditing a "reconstruction"; a blocked owner replies `BLOCKED: <artifact>`
   rather than fabricating.

If all five hold on Flash, the pipeline is robust to the weakest model you run.

---

## 8. Minimal path (if you want less than the full recipe)

You don't have to adopt everything at once. Cheapest useful step:

```json
{ "quality": {"profile": "long_form"}, "deliverable": {"path": "<name>.md", "kind": "<document|fiction>"} }
```

with `execution_groups: true`, `derive_manifest` on (it's in the preset), and no
paid flags. That already gives you deterministic assembly, the done-gate, honest
waits, and seam ordering — for free. Add `canon_pass` + `gate_blocks_done` +
`token_budget` when you want the continuity gate too.
