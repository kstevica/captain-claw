# The work board — the mind assigns, the feet take up or refuse

Status: SHIPPED 2026-07-19, live-verified (end-to-end loop + UI in both
themes). 17 new tests; full FD suite green minus the 8 pre-existing
mcp/vfs failures.

Sibling to `docs/being-instinct-build-plan.md` (the impulsive feet that
break ground) and `docs/being-body-brain-plan.md` (the two-brain split).
This arc closes the loop **between** the two brains: the analytical mind
writes a board of tasks; the impulsive feet actively pick the one that
suits them now — or refuse it, with a reason — and the mind reviews the
board each tick and edits it.

Today a `being_plans` table already exists, but it is a **one-way nudge
list**: the mind suggests `go`/`meet`, the *world* silently fulfills a
step on arrival / co-presence, and the feet only read it as ambient
context. There is no active selection, no in-progress / refused feedback,
no build tasks, and no mind-review loop. This arc grows that seed into a
real, two-way **work board**.

## The law it keeps

- **Physics decides.** A task is `done` only on a real effect — an
  arrival actually recorded, or a real `staked` row created. Never a
  theatrical "done".
- **The two brains stay divided.** The feet break ground (stake); the
  mind still names + pays to make the thing real (finish). So a **build
  task is `done` the moment the feet stake it** — the gesture is the
  feet's to complete; the object still needs the mind, via the existing
  crumble/finish loop.
- **The mind owns the words.** The mind writes tasks and reads the board;
  the feet's only utterance is a short **refusal reason** — a body-status
  note on a task, not a public voice (the inscription stays the mind's).

## Locked decisions (user, 2026-07-19)

1. **Build = done when the feet break ground** (stake). The mind finishes
   the staked thing separately via the existing loop.
2. **Refusal reason = a short free-text tag** (≤40 chars) the feet write —
   expressive, at the cost of bending "the feet never write words" a
   little (it is a status note, never a spoken/inscribed voice).
3. **Anyone can interrupt.** Any being may stop a current walk to seize an
   actionable task, regardless of impulse (rate-limited so it never
   thrashes).
4. **Yes, a Work panel** — a small read-only board on the being page:
   open / active / done / refused.

## The board (as built)

`being_plans` grows from `{id, being_id, kind, target, state, created_at,
done_at}` with four columns and a richer state machine:

- `detail TEXT` — for a **build** task, the object KIND (bench, cairn,
  planter, …); empty for go/meet.
- `note TEXT` — the feet's refusal reason (a short tag), or a mind note.
- `claimed_at TEXT` — when the feet took the task up (`active`).
- `object_id TEXT` — the stake a completed build task produced (links the
  board to the real thing).

Task `kind` gains **`build`** alongside `go` and `meet`. The user's verbs
map: *go there / move here* → `go`; *build that* → `build`; *plant this* →
`build` with `detail=planter`.

State machine: `open` (mind wrote it, unclaimed) → `active` (feet claimed,
walking / working) → `done` (grounded) | `refused` (+reason); plus
`dropped` (mind removed it) and the existing `lapsed` (world outran it).

## The big brain (each mind tick)

- **Sees the board.** A new `board_percept` in the umwelt sweep every wake
  lists the open tasks, what the feet *finished* since the last wake, what
  they *refused and why*, and what is *active*.
- **Edits the board** via the digest:
  - `plan` (extended) adds go / meet / **build** tasks. A build task is
    `{"build": "bench", "at": "plaza"}` (KIND + where). go stays
    `{"go": "library"}`, meet `{"meet": "ada"}`.
  - `plan_drop` (new) removes open / refused tasks by target or id:
    `"plan_drop": ["the mill", "<id>"]`.
- Still **finishes** the stake a build task produced, via the unchanged
  `stake_confirm` percept → `finish` loop.

## The small brain (each feet decision)

- **Trigger.** `wants_decision` gains a `task` trigger: an actionable
  (open or active) task the feet could serve now. It may fire **mid-walk**
  (the interrupt), rate-limited by `FEET_TASK_MINUTES` so a walk is not
  re-decided every reflex pass.
- **Prompt.** `feet_prompt` lists the actionable tasks with short handles
  and walk-times — `[t1] build bench @ plaza (3 min)`, `[t2] go to mill
  (12 min)` — and teaches the two new acts.
- **Acts** (new, whitelisted):
  - `{"act": "do", "task": "t1"}` — take it up. The engine resolves by
    task kind: a **go** task departs toward the target (→ active; the
    world's arrival-settle marks it done); a **build** task at/near the
    spot **stakes** that KIND (→ done, `object_id` set); a build task far
    off departs toward the spot (→ active) and a later arrival-triggered
    feet call finds the active task underfoot and stakes it.
  - `{"act": "refuse", "task": "t2", "why": "too far"}` — decline; task →
    refused + note; the mind hears it next wake.
  - Ignoring the board (linger / free go / free impulsive build) is
    "select nothing" — untouched.
- **Selection.** Tasks are handled in a stable order (oldest-first,
  matching the prompt) so `tN` means the same task in prompt and apply.
  The feet LLM makes the final pick from the presented, walk-timed list.

## Tests, bundle, verify, docs, memory, commit — as every arc.

## Deferred

- Meet tasks staying world-fulfilled (co-presence) — the feet don't
  actively pursue a `meet` beyond the existing nudge.
- A being working *another's* board (only its own mind assigns to it).
- Task priorities / ordering hints from the mind (v1 is oldest-first +
  the feet's own judgement).
