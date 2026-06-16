# Jarvis — #1 Trusted Action Catalog + #2 Event-Ingestion Spine

> The pair that turns the closed loop from "proposes" into "acts on your world."
> #1 = hands (do real things, safely, reversibly). #2 = senses (react to your
> world, not just its own reflections). Build #1 and #2, then #3 trust ladder,
> #4 long-horizon planning, #5 grounded verification.

Grounding: builds on the Autonomous Work loop (`arbiter.py`, `fd_dispatch.py`,
the autonomy store/ledger), the tool layer (`tools/registry.py`,
`_execute_tool_with_guard`), and the sister-session/cron scaffolding.

---

## Part #1 — Trusted Action Catalog

### The gap
- Side-effecting tools exist and run in the **agent** via
  `ToolRegistry.execute(name, args, …)` wrapped by `agent_guard_mixin._execute_tool_with_guard`.
- The arbiter dispatches only **free-form text** (`fd_dispatch._instruction_for` →
  `_dispatch_one` → the agent's LLM decides which tool). No deterministic
  "run tool X with structured args Y."
- Existing risk notion: `GuardConfig.script_tool` + `config.tools.require_confirmation`
  (`["shell","write","edit"]`). No reversibility model.

### The four pieces

**A. Deterministic tool-invocation primitive (the missing rail).**
Add a way to run ONE named tool with structured args, bypassing the LLM, and
return the `ToolResult`. Two execution homes:
- **Agent tools** (have creds/workspace): new WS message `run_tool`
  (`web/ws_handler.py`) → `agent._execute_tool_with_guard(name, args, "autonomous-action")`
  → returns `ToolResult`. Reuses the guard. (Alternatively a small HTTP endpoint
  on the agent web server; WS keeps parity with the existing notification path.)
- **FD-native actions** (deliver, scheduler, basna, stop_run): keep their current
  in-process handlers in `fd_dispatch`.

**B. The Action Catalog** — a *curated*, named subset of capabilities the
autonomous loop may use (NOT every tool). New module
`captain_claw/flight_deck/action_catalog.py`. Each entry:
```
{
  id: "calendar.hold",            # stable action id
  label: "Create a calendar hold",
  home: "agent" | "fd",
  tool: "google_calendar", subtype: "create_event",   # for agent home
  arg_schema: {…JSON schema…},     # validated before dispatch
  risk: "low" | "normal" | "high",
  reversibility: "read_only" | "reversible" | "irreversible",
  reverse: {tool, subtype, args_from_result},          # reversible only
  grant: "calendar",               # which user grant must be enabled
}
```
Ship a small, safe v1 set (bias to **reversible / draft-don't-send**):
- `mail.draft` (create_draft — reversible: delete draft) — NOT mail.send (irreversible → stays propose-only)
- `calendar.hold` (create_event — reversible: delete_event)
- `note.write` (write to a workspace notes file — reversible: prior content kept)
- `reminder.schedule` (cron prompt — reversible: delete job)
- `task.investigate` (basna/sister — read-only-ish)
Irreversible actions (`mail.send`, `calendar.delete`, `drive.delete`) are in the
catalog but flagged `irreversible` so the trust gate (#3) keeps them human-only.

**C. Reversibility model + Undo.**
- Each dispatched action records its reverse handle into the ledger row's
  `payload` (e.g. the created `event_id`/`draft_id` from the `ToolResult`), so a
  concrete reverse call can be built later.
- New `POST /fd/autonomy/actions/{id}/undo` → runs the reverse via the same
  deterministic primitive; marks the action `undone`. "Undo" button on done
  reversible actions in the Autonomous Work page.
- Extends the existing intentions "announce + undo" notion to real actions.

**D. Arbiter + dispatch integration.**
- New action kind `tool_action`; payload `{action_id, args}`.
- Arbiter prompt gains the **catalog** (ids + labels + arg fields + risk) filtered
  to the user's enabled grants; it proposes a concrete `tool_action`.
- Validation in `arbiter.py` (mirror the `stop_run` target check): `action_id`
  must be in the catalog, args validated against `arg_schema`, **risk/reversibility
  come from the catalog, never the LLM**.
- `fd_dispatch.dispatch_action`: `tool_action` branch → resolve catalog entry →
  run via the primitive (agent `run_tool` or FD handler) → capture reverse handle
  → judge/learn as today.
- `should_auto_dispatch`: extend so only `reversibility ∈ {read_only, reversible}`
  AND `risk == low` may auto-fire at `act_low_risk` (irreversible always gated).
  Full trust policy is #3; this is the safe floor.

**E. Grants (per-user allow-list).** Which catalog actions are enabled. Store in
the autonomy config (reuse `autonomy_config` / `AutonomousWorkConfig`) — a
`granted_actions: list[str]`. Surface as toggles on the Autonomous Work page
("Allowed actions"). Default: empty (nothing actable until the user opts in).

### #1 phases
1. ✅ **Primitive + catalog:** `run_tool` WS rail + `action_catalog.py` + manual
   `/run-action`. Proven on dev (note.write writes a file). Fixes: also_allow past
   the per-session policy; refresh the Google-connected cache before exec.
2. ✅ **Reversibility + undo:** `build_reverse` (id from ToolResult), capture at
   dispatch, `/undo` + UI Undo button.
3. ✅ **Arbiter `tool_action`:** catalog in the prompt (grant-filtered), propose +
   validate (risk/reversibility from catalog) + dispatch via the rail with a
   grounded outcome. Propose-only for now.
4. ⬜ **Grants UI + auto-fire:** per-user allowed-actions toggles; let granted
   reversible/low-risk actions auto-fire (this is also the #3 trust hook).

### #1 files
- New: `captain_claw/flight_deck/action_catalog.py`
- Edit: `web/ws_handler.py` (run_tool), `agent_guard_mixin.py` (reuse), `flight_deck/arbiter.py`
  (catalog in prompt + `tool_action` validate), `flight_deck/fd_dispatch.py`
  (`tool_action` dispatch + `should_auto_dispatch` reversibility), `flight_deck/autonomy.py`
  (grants + reverse-handle in ledger), `flight_deck/autonomy_routes.py` (`/undo`, `/run-action`),
  `config.py` (`granted_actions`), frontend `AutonomousWorkPage.tsx` + `autonomyStore.ts`.

---

## Part #2 — Event-Ingestion Spine

### The gap
- No external watchers (no gmail/calendar/rss/file). Consciousness observes only
  **agent session deltas**. Arbiter candidates come from reflections/intuitions.
- Reusable scaffolding: sister-session `watches` + `proactive_tasks`, the
  `cron_scheduler_loop` poll cadence, the channel bus, FD scheduler.

### The design — a normalized event spine feeding the arbiter

**A. `external_events` store** (new table in the autonomy DB or a dedicated
`events.db`): `id, user_id, source, event_type, summary, body, metadata(json),
dedup_key, ingested_at, status(new|surfaced|acted|ignored), processed_at`.
`dedup_key` (e.g. gmail message id) prevents re-ingesting the same event.

**B. Source adapters** — each normalizes a real-world signal into `external_events`.
Two intake modes:
- **Pollers** (start here — no push setup, reuse existing google creds/tools):
  `poll_gmail` (new/important unread since cursor), `poll_calendar` (upcoming /
  changed events in a window). Run on a new FD tick (reuse the heartbeat loop or a
  dedicated `events_loop`, mirroring `cron_scheduler_loop` cadence).
- **Webhook receiver** (later, for true push): `POST /fd/events/ingest` with
  per-source verification → normalize → insert. Lets Gmail push / calendar push /
  third parties feed in without polling.

**C. Event → arbiter bridge.** The arbiter already builds candidates. Add new-event
intake to `arbiter._gather_candidates` (or a sibling): read `status="new"` events
for the user, surface as candidates (`(event) gmail: flight delayed …`), and mark
`surfaced`. The arbiter then proposes a `tool_action`/`nudge`/`basna` referencing
the event — the same propose→approve→dispatch path from #1.
- Run the arbiter **on event arrival**, not just on the 180s pulse: when a poller
  inserts new events, trigger `pulse(uid, force=True)` (debounced) so reaction is
  prompt. This makes the heartbeat react to the world, not only to agent chatter.

**D. Reuse, don't duplicate.** High-signal events (deadlines, "needs investigation")
can also spawn a sister-session task (existing `maybe_create_proactive_task`) for a
read-only investigation whose briefing becomes an arbiter candidate. Keep
`external_events` as the raw spine; sister-session is one consumer, the arbiter the
primary one.

**E. Controls + safety.** Per-source enable/disable + poll interval + quiet hours
(reuse the autonomy config). Dedup via `dedup_key`. Rate-cap events surfaced per
pulse so a busy inbox can't flood the arbiter. Retention/cleanup of old events.

### #2 phases
1. **Spine:** `external_events` table + store methods + a manual `POST /fd/events/ingest`
   to insert test events. Arbiter reads `new` events as candidates (grant/flag-gated).
2. **First poller — Calendar:** `poll_calendar` adapter (upcoming/changed) on the
   events loop; debounced `pulse(force)` on new events. (Calendar first: lower
   volume, naturally maps to reversible `calendar.hold` actions from #1.)
3. **Gmail poller:** `poll_gmail` (important unread) — higher volume, needs good
   dedup + surfacing caps.
4. **Webhook receiver + per-source controls UI.**

### #2 files
- New: `captain_claw/flight_deck/events.py` (store + adapters + loop),
  `captain_claw/flight_deck/event_routes.py` (`/fd/events/*`).
- Edit: `flight_deck/server.py` (start events loop, mount router),
  `flight_deck/arbiter.py` (`_gather_candidates` reads events; mark surfaced),
  `flight_deck/consciousness.py` (debounced force-pulse on new events),
  `config.py` (event sources config), frontend (sources panel on the page).

---

## How #1 + #2 click together (the Jarvis moment)
`poll_gmail` ingests "flight delayed" → arbiter sees the event → proposes
`tool_action: calendar.hold` (reversible) + `mail.draft` reply → at propose it
waits for one tap; once #3 trust is earned for those reversible low-risk actions,
they fire and you just get told. #5 later confirms the hold actually landed.

## Sets up #3–#5
- #3 (trust ladder): `should_auto_dispatch` already keys on risk + reversibility +
  per-kind reliability — extend to per-`action_id` reliability + a policy table.
- #4 (planning): a `tool_action` sequence becomes a plan; the ledger + reverse
  handles give rollback on partial failure.
- #5 (verification): the deterministic primitive returns real `ToolResult`s to
  verify against (did the draft/hold actually get created?) instead of LLM self-grading.

## Decisions (LOCKED 2026-06-16)
- **Execution home for agent actions: WS `run_tool`** (parity with the notification path).
- **Catalog v1:** 4 reversible auto-eligible — `note.write`, `calendar.hold`,
  `mail.draft` (draft, NOT send), `reminder.schedule`; + 4 in-catalog-but-human-only —
  `mail.send`, `calendar.invite`/`calendar.delete`, `message.send`, `drive.delete`.
  **Hard-excluded from the autonomous catalog entirely:** raw `shell`, `browser`
  form-submit, social posting, anything payment-like (available in normal chat, never autonomous).
- **Events store: dedicated `events.db`** under FD_DATA_DIR (matches per-subsystem DB
  convention), WAL, status-flag idempotency (`new→surfaced→acted/ignored`), retention sweep.
- **Intake: poll-first** (Calendar then Gmail); webhook receiver later.
