# Taskboard — plan

A kiosk (user mode) dialog that lists every task an agent received or sent on any
channel, tracks whether it was finished, by whom, for whom, with which parties and
with what result, and proposes next steps the user approves. Automatic, searchable
and filterable on every dimension, designed like the File library. Jev decides.

Branch: `feat/taskboard` (v0, v1 and v2 all land here, one commit + push per version).

## User decisions (2026-10-29)

| # | Question | Answer | What it means here |
|---|----------|--------|--------------------|
| 1 | Personal inboxes (own WhatsApp Web chats, Telegram inbox, all of Gmail)? | Yes, opt-in | v2: each one is off until the user switches it on in Taskboard → Sources, with consent text. Switching off deletes the tasks derived from it. Text inherits the source's retention (30 days). |
| 2 | May Jev close a task on its own? | Yes | `fulfilled ≥ 0.8` closes it (`closed_by = jev`) with Undo. The evidence badge still says whether a send/delivery was recorded. |
| 3 | Is a quick chat question a task? | Yes | Every admitted human turn is a task. Answered in the same turn on the user's own chat → done, result "answered in chat". It never enters Needs me. |
| 4 | Does an owner see members' tasks on a shared agent? | Not at all | A task belongs to exactly one board user (`user_id`). A member's turn on a shared agent goes to the member's board only. No owner view, no admin view. |
| 5 | Make non-mail sends on automated turns propose-only? | No | Existing automations keep working. Board-originated next steps always need a human click anyway. |
| 6 | Absorb Scheduled → All activity and follow-ups? | Link | Those views stay. The board ingests them as tasks and links back ("Open in Scheduled", "Continue in chat"). |
| 7 | History to fill in on first open | 30 days | v2 backfill from agents' `sessions.db`, last 30 days, marked historical, never proposes anything. |

## Principles

1. **Needs me first.** The dialog opens on "Needs me": what the user must decide, answer or confirm. Everything else is a view away.
2. **Evidence beats opinion, the user beats both.** Source statuses and recorded sends move a task first. Jev only picks among the moves the evidence allows. A user action (Done, Not a task, Snooze, Rename, Owes next…) applies at once and locks the field it touched; no machine pass overwrites a locked field.
3. **Jev decides, a model writes.** Jev answers typed questions (yes/no, choice, score) with calibrated probabilities. It cannot write a title, pull out parties or draft a reply — that is one extraction call per task on **the agent's own model** (owner tiers only as fallback).
4. **Strictly per user.** Every read is filtered by `user_id` on the server. The kiosk lock is UI-only, so the API is the boundary.
5. **Cheap first.** A deterministic gate (no model) runs before anything is judged. Jev and extraction have per-user daily caps; extraction tokens go through `usage_quota.record_tokens` and stop on `automation_blocked`.
6. **Nothing leaves without a click.** Email stays propose-only (draft). Every outward next step is a typed autonomy-ledger action with recipients from stored fields, approved by the user.
7. **Honest coverage.** A Sources sheet says what is tracked and what is not.

## Architecture

```
FD ledgers (scheduler_runs, autonomous_actions, follow_ups,         ┐
gmail_autoreplies, gmail_sends, agent_link_events)  ── adapters ──┐  │ v0
                                                                  │  │
agents: turn register (open/close of every turn) ──┐              │  ┘
        tool_audit rows + turn_id + result refs ───┼── ingest ────┤    v1
                                                   │              ▼
opt-in passive inboxes (TG inbox, WA Web, Gmail) ──┘  taskboard.db (tasks, links, events,
backfill (sessions.db, 30 days)                       decisions, parties, settings)   v2
                                                              │
                         gate → extraction (agent's model) → Jev decisions (batched)   v1
                                                              │
                     next-step proposals → autonomy ledger → bell / WA / TG / devices  v2
```

Modules (one owner each, so parallel work never edits the same file):

| Module | Version | Holds |
|--------|---------|-------|
| `captain_claw/flight_deck/taskboard_store.py` | v0 | `TaskboardStore` (singleton, RLock, PRAGMA migrations), the full schema below, folding helper, upsert/link/event/lock primitives, search + facets |
| `captain_claw/flight_deck/taskboard_sources.py` | v0 | Cursor adapters over FD ledgers + the sweep loop |
| `captain_claw/flight_deck/taskboard.py` | v0 | User router `/fd/taskboard/*`, lanes, corrections, `configure()`/`start()`/`stop()`, agent-removal cleanup |
| `captain_claw/turn_register.py` | v1 | Agent-side turn records (spool + shipper, modelled on `tool_audit.py`) |
| `captain_claw/flight_deck/taskboard_ingest.py` | v1 | `POST /fd/taskboard/agent/turns`, tool-audit adapter, gate |
| `captain_claw/flight_deck/taskboard_triage.py` | v1 | Extraction + Jev decisions worker, caps, decision log |
| `captain_claw/flight_deck/taskboard_nextsteps.py` | v2 | Next-step proposals into the autonomy ledger, edit-before-approve |
| `captain_claw/flight_deck/taskboard_threads.py` | v2 | Strong-key attach, merge/split, parties |
| `captain_claw/flight_deck/taskboard_passive.py` | v2 | Opt-in inbox adapters, consent, retention, delete-on-disable |
| `captain_claw/flight_deck/taskboard_backfill.py` | v2 | 30-day history from `sessions.db` |
| `flight-deck/src/components/taskboard/*` | v0+ | `TaskBoard.tsx` and parts |
| `flight-deck/src/stores/taskboardStore.ts`, `services/taskboardApi.ts` | v0+ | Store + typed API |

## Data model — `DATA_DIR/taskboard.db`

The whole schema is created in v0 so later versions only fill columns. Owner columns
have no FK to users (an auth-off deck uses the `local` user). Times are epoch seconds
(REAL). JSON columns are TEXT.

**tasks**
- `id` TEXT PK (`tb_` + 16 hex)
- `user_id` — the board this task belongs to (owner for own turns, the member for member turns)
- `agent_owner_id` — the agent's owner (cleanup only, never used for visibility)
- `agent_slug`, `agent_ref` (`runtime:slug:instance` when known), `agent_name`
- `title` (≤160), `ask` (≤600), `result_text` (≤600)
- `kind` — `request | sent | scheduled | proposal | follow_up | held_reply | link | todo | message`
- `direction` — `received | sent | internal`
- `channel` — first channel (turn_source key: `web, whatsapp, waweb, telegram, tg_inbox, gmail, device:<kind>, link, scheduled, autonomy, member, api, mcp, glasses, messenger`)
- `channels` JSON list — every channel the task touched
- `status` — `open | done | failed | dismissed | not_task | maybe`
- `owes_next` — `me | agent | them | nobody | unclear`
- `evidence` — `verified | claimed | none`
- `result` — `success | partial | failed | unknown | ''`
- `urgency` INTEGER 0–3, `due_at`, `snooze_until`
- `created_at`, `updated_at`, `last_event_at`, `closed_at`
- `closed_by` — `source | evidence | jev | rule | user | ''`
- `closed_by_party` — display name of who finished it (agent name, user, party)
- `fingerprint` (last event id + count), `extracted_fp`, `decided_fp`
- `triage_state` — `pending | ok | error | skipped`, `triage_error`, `triage_after`
- `user_locks` JSON `{field: at}` (a flag key the user set, e.g. `probably_done`, `low_confidence`, is locked by its own name)
- `flags` JSON (`contains_instructions`, `probably_done`, `stale`, `overdue`, `historical`, `answered_in_chat`, `failed_ack`, `why` (≤ 160, the latest decision's first reason); v1 adds `low_confidence` (admitted at Jev 0.65–0.8, left out of the badge), `expected_deliverable` (≤ 200, from extraction), `rule` (`mute` / `always_task`, the task rule that admitted it))
- `private` INTEGER (source turn held a member's data)
- `merged_into` TEXT, `source` (first adapter), `source_ref`
- `next_steps` JSON (candidates from extraction, v1/v2)
- `retention_until` REAL (passive sources)
- `search_text` — accent-folded haystack (`outbound_guard._fold`): title, ask, result, parties, agent, channel labels

Indexes: `(user_id, status, last_event_at)`, `(user_id, owes_next)`, `(user_id, agent_slug)`, `(agent_owner_id, agent_slug)`, `(merged_into)`.

**task_links** — `(user_id, key)` PK, `task_id`, `strength` (`strong | weak`), `created_at`.
Keys: `sched:job:<id>`, `sched:run:<id>`, `autonomy:<id>`, `fu:<id>`, `gmail:thread:<id>`,
`gmail:msg:<id>`, `link:<link_id>:<msg_id>`, `turn:<turn_id>`, `session:<agent_slug>:<session_id>`,
`wa:<waid>`, `waweb:<chat_jid>`, `tg:<chat_id>`, `todo:<agent_slug>:<id>`.

**task_events** — `id` INTEGER PK, `task_id`, `user_id`, `at`, `kind`
(`received | reply | sent | tool_call | delivery | status | proposal | approval | rejection | decision | correction | note`),
`channel`, `actor_kind` (`me | agent | party | system | jev`), `actor` (display), `summary` (≤500),
`source`, `source_ref`, `outcome` (`ok | error | blocked | ''`), `refs` JSON, `evidence` (`recorded | said | user | ''`), `private`.
`UNIQUE(source, source_ref)` makes every ingest idempotent.

**task_decisions** (v1) — `id`, `task_id`, `user_id`, `at`, `judge` (`jev | rule | model | user`),
`qset` (question-set version), `state_hash`, `answers` JSON (probabilities only), `applied` (transition text), `why`.

**parties** (v2) — `id`, `user_id`, `display_name`, `folded_name`, `kind` (`person | org | agent | me`).
**party_handles** (v2) — `(user_id, kind, value)` PK, `party_id`; kind `email | waid | wa_jid | tg_id | fd_user | link`.
**task_parties** (v0 raw, v2 resolved) — `task_id`, `role` (`requester | doer | recipient | cc | mentioned`),
`kind`, `address`, `display`, `party_id` (v2), `verified` (header/register/speaker sources only),
PK `(task_id, role, kind, address)`.

**task_rules** (v1) — `id`, `user_id`, `match` JSON (`sender | domain | channel | agent | job_id | source`), `effect` (`not_task | always_task | mute`), `created_from`, `hits`, `created_at`.

**turns** (v1, raw turn register) — `turn_id` PK, `agent_owner_id`, `user_id`, `agent_slug`, `session_id`,
`origin`, `origin_detail`, `channel`, `source`, `source_label`, `speaker_label`, `ext_refs` JSON,
`text_head`, `reply_head`, `deliverables` JSON, `outcome`, `started_at`, `ended_at`, `private`, `task_id`;
v1 adds `session_name`, `speaker_id`, `member`, `opener`, `human`, `literal`, `automation` JSON `{kind, id}`,
`tool_calls`, `failed_tool_calls`, `gate` (the gate's verdict, below), `task_key` (the link key that finds its task),
`open_ref` / `close_ref` (`boot:seq` each phase came under), `received_at`.
Indexes `(received_at)`, `(gate, received_at)`, `(agent_slug, open_ref)`, `(agent_slug, close_ref)`.
Kept 30 days by `received_at` (each full sweep prunes, and so does the turn route at most hourly — the 30 days hold with the sweep off too; the tasks keep their `turn:` key).

**source_cursors** — `source` PK, `cursor`, `updated_at`.

**taskboard_settings** — `user_id` PK, `passive` JSON (`{telegram_inbox, whatsapp_web, gmail}` each `{on, consented_at}`),
`auto_close` INTEGER (default 1, decision #2), `jev_per_day` (200), `extract_per_day` (100), `counters` JSON (`{day, jev, extract}`),
`backfill_done_at`, `updated_at`.

## Lanes (computed, never stored)

- **Needs me** — `status=open` and not snoozed, and any of: `owes_next=me`, a linked proposal awaiting approval, a held reply, `flags.probably_done`, overdue; plus `status=failed` not yet acknowledged.
- **Waiting on others** — `status=open`, `owes_next=them`.
- **Agents working** — `status=open`, `owes_next=agent`.
- **Done** — `status=done`.
- **All** — everything except `not_task`, `merged_into` set; `maybe` shows under a "Maybe tasks" facet (v1).

Badge on the Taskboard tile = Needs me count (in v1 only tasks admitted at ≥ 0.8 count: those with `flags.low_confidence` stay in the Needs me list but are left out of `GET /summary`).

## API (user routes need `get_current_user`, filter by `user_id`, 404 on someone else's task)

- `GET /fd/taskboard/status` → `{sources: [{key, label, tracked, on, note}], counts: {needs_me, waiting, working, done, all}, sweep: {running, last_at, message}}`
- `GET /fd/taskboard/tasks?view=needs_me|waiting|working|done|all&q=&status=&channel=&agent=&direction=&kind=&party=&evidence=&result=&from=&to=&sort=recent|oldest|due|urgency&limit=&offset=`
  → `{items: [TaskItem], total, facets: {status, channel, agent, direction, kind, evidence, result, party}, counts}`.
  Facets are counted under the other filters (as in the File library). Multi-value filters are comma-separated.
- `GET /fd/taskboard/tasks/{id}` → `{task: TaskItem, events: [...], events_total, links: [...], parties: [...], decisions: [...], proposals: [...]}` (events = the latest 500, oldest first)
- `POST /fd/taskboard/tasks/{id}/action` `{action, value?}` — `done | reopen | dismiss | not_task | snooze (value = until epoch) | unsnooze | rename (value) | owes_next (value) | ack_failed | undo` → `{task}`. Each writes a `correction` event and locks the field. A closed task opened again (`reopen`, `not_done`, Undo of Jev's close) that nobody owed a move becomes the user's (`owes_next` me: back in Needs me). `undo` takes back the latest correction still holding a field — or, when Jev closed the task (`closed_by jev`) after it (or there is none), opens the task again as a correction of its own ("Undid Jev's close", status locked; Undo again puts Jev's close back).
  v1 adds `confirm_task` (a `maybe` task → open, owes the agent; an open `low_confidence` one → `low_confidence` false; locks `status` and `low_confidence`) and `not_done` (a task with `flags.probably_done`, or `done` → open, `probably_done` false, closing fields cleared; locks `status` and `probably_done`). Anything else they don't apply to is a 400.
- `POST /fd/taskboard/sync` → runs the adapters now for this user (also run on dialog open). One sweep per user at a time and at most one per 2 s: calls while one waits join it, calls while one reads get the next.
- `GET /fd/taskboard/summary` → `{needs_me}` (cheap, polled by the tile badge every 30 s; v1: without `low_confidence` tasks).
- `GET /fd/taskboard/settings` → `{auto_close, jev_per_day, extract_per_day, counters: {day, jev, extract}, jev: {connected, source: own|team|""}, passive, backfill_done_at, updated_at}`; `counters` is today's (UTC) use, zeros on a new day. `PUT` takes any of `{auto_close, jev_per_day (0–2000), extract_per_day (0–1000)}` (400 outside the bounds) and answers like `GET`. (v2: passive sources with consent.)
- `GET /fd/taskboard/tasks/{id}/why` (v1) → `{items: [{at, judge: jev|rule|model|user, qset, applied, why, answers}], summary}`, latest first, at most 20: the `task_decisions` rows (extraction is judge `model`) merged with the user's own corrections (judge `user`, qset "", answers {}). `answers` by question id: `{p}` (yes/no), `{choice, p}`, `{score}` (urgency 0–3). `summary` says in a sentence or three why it stands where it is. 404 for someone else's task.
- v2: `POST /fd/taskboard/tasks/{id}/merge {into}`, `/split {event_ids}`, `/propose {kind}`, `PATCH /fd/autonomy/actions/{id}` (edit before approve), `GET /fd/taskboard/export?format=csv|jsonl`.
- Agent route (v1): `POST /fd/taskboard/agent/turns` (agent auth exactly as `/fd/tool-audit/agent/calls`: `tool_audit_routes._identify`, the `X-Agent-Auth` token) — body and records in v1 item 1, → `{ok: true, accepted}`.

`TaskItem` = the task row minus `search_text`/fingerprints, plus `parties` (first 5), `lane`, `proposal` (`{id, status, title}` when a linked autonomy action is waiting), `links` (`{scheduled_job_id, session_id, autonomy_id, followup_id}`) and `agent_name`.

Approve / Reject of a linked proposal call the existing `/fd/autonomy/actions/{id}/approve|reject`. A held Gmail auto-reply uses the existing auto-reply approve route — on its own `held_reply` task, or on the request task whose thread it is on (its latest auto-reply event is `held`).

## UI — `TaskBoard.tsx`

Same frame as `FileLibrary.tsx`: `KioskWideDialog label="Taskboard" widest`, header (icon chip, title, subtitle, sync status, Sync now, FullscreenButton, Close), toolbar (debounced search, sort select), three columns on desktop — facet sidebar (w-56), list, detail pane (w-[380px]); single column on a phone with the detail as an overlay ("← All tasks") and filters in a full-screen sheet.

- **View tabs** across the top of the list: Needs me (default) · Waiting on others · Agents working · Done · All, each with its count.
- **Facets**: Status, Channel, Agent, Direction, Kind, Evidence, Result, Party (v2 resolved), Date (Today / 7 days / 30 days / custom from–to). Multi-select (checkbox per value).
- **Row**: channel icon, highlighted title, lane chip, evidence badge (✓ recorded / "agent says" / none), meta line (agent · parties · relative time), two-line ask/result, "Needs you: approve next step" strip when a proposal waits, with inline Approve / Reject.
- **Detail**: status + owes-next chips, ask, result, parties (role chips), a timeline across channels (received → tool calls → sends → deliveries → decisions → corrections; each with channel icon, actor, time, evidence mark), links (Open in Scheduled, Continue in chat, Open in bell), corrections (Done, Reopen, Not a task, Snooze 1 h / tomorrow / next week, Rename, Owes next), "Why is this here?" (v1), next-step card (v2).
- **Sources sheet** (header button): every source with tracked / not tracked yet / off, and the opt-in switches with consent text (v2).
- **Entry points**: agent-column ToolTile (add `TILE_COLS` 5), collapsed rail icon, phone header (next to the bell), Settings tile, and `phoneOverlayOpen`/`closePhoneOverlays` so phone Back closes it. Tile badge = Needs me count.
- Dark-first zinc classes (the light theme inverts zinc); `dark:` variants only for non-zinc colours. User-facing name: "Taskboard". Brand text says "Captain".
- Search highlights with the same folding as the server (strip diacritics on both sides).

## v0 — the board from records Flight Deck already keeps (no model calls)

Adapters (`taskboard_sources.py`), each idempotent through `task_events UNIQUE(source, source_ref)` and a cursor in `source_cursors`; run by a sweep loop every 120 s while FD is up and by `POST /sync`. They never wake agents.

| Source | Task | Status mapping |
|--------|------|----------------|
| `fd_scheduler` jobs + `scheduler_runs` | one **standing task per job** (`sched:job:<id>`, kind `scheduled`, direction `internal`, channel `scheduled`); each run is an event (`sched:run:<id>`), deliveries become `delivery` events | running → open/agent; last run ok → done (`evidence verified` when a delivery ok, `closed_by source`); error → failed; the next run reopens it to open/agent (over the user's Done/Reopen, not Dismiss/Not a task); a run skipped before it began leaves it as it was; a job switched off before a run → owes nobody, deleted → dismissed |
| autonomy `autonomous_actions` | task per action (`autonomy:<id>`, kind `proposal`), attached to the follow-up's task when `ref_id` points at one | awaiting_approval → open/me (Needs me, inline Approve/Reject); queued/dispatched → open/agent; done → done (`claimed` unless a recorded send exists); failed → failed; rejected → dismissed (`closed_by user`); expired → dismissed (`closed_by source`) |
| events `follow_ups` (+ origin `external_events` for the Gmail thread key) | task per follow-up (`fu:<id>`, kind `follow_up`), strong link `gmail:thread:<id>` when known | open and due → open/me; open not due → open/them; done → done; dismissed → dismissed; stale → open + `flags.stale` |
| `gmail_autoreplies` (only rows that reached held / approved / sending / sent, or failed after a topic matched) | task per inbound thread (`gmail:thread:<id>`, kind `held_reply`, direction `received`, channel `gmail`), requester from the verified sender | held → open/me; approved/sending → open/agent; sent → done (`verified`); failed → failed |
| `gmail_sends` | event `sent` on the `gmail:thread:<id>` task; when none exists, a task kind `sent`, direction `sent` | sent → done (`verified`, `closed_by evidence`); recipients become parties; an auto-reply's own send is left to its auto-reply row |
| `agent_link_events` | task per system message (`link:<link_id>:<msg_id>`, kind `link`, channel `link`) | reply/push delivered → done (`verified`); error → failed; no reply yet → open/agent |

Also in v0:
- Corrections with field locks; Undo of the last correction (what a source said while the lock held — `refs.held` on its events — is applied then).
- No message text copied from sources that purge it: a sent auto-reply's result is "Auto-replied (topic)", a link message's "Answered …".
- `autonomy_routes.reject_action_route` uses `claim_action` (compare-and-set) like approve, so a double click or a stale bell cannot flip an already dispatched action.
- Agent removal drops its tasks (server.py drop-agent loop, next to the libraries; owner-less ones only on the remover's board). User deletion drops the user's rows, and the sweep never puts a deleted user's ledger rows back.
- Autonomy approve / reject / undo (and follow-up done / dismiss) act on `local` rows only on a deck without sign-in.
- Sources sheet lists v0 sources as tracked and the rest as "coming" (agent chats on every channel, opt-in inboxes, history).
- Tests: `tests/test_flight_deck/test_taskboard_store.py`, `test_taskboard_sources.py`, `test_taskboard_routes.py` (per-user isolation, idempotent re-ingest, status mapping per source, locks win, folded search finds "Đurić" by "duric", facet counts).

Done when: a kiosk user opens Taskboard, sees their scheduled jobs, proposals (Approve/Reject works and the row moves lanes), follow-ups, auto-reply threads, sent mail and link messages; another user sees none of them; search and every facet work on desktop and phone; the bundle is built and committed.

## v1 — every channel, Jev and extraction

1. **Turn register** (`captain_claw/turn_register.py`), hooked in `Agent.complete` and `Agent.stream` right after the turn's origin and member privacy are set (`_set_turn_origin`, `member_privacy.enter_turn`), closed in their `finally`. Only the outermost entry is a turn: a `complete()` inside an open turn (`stream()` re-entering it, a sub-call in the same context) records nothing. The running turn's id is the `current_turn_id` ContextVar.
   On exactly when the tool-call register is (`tool_audit.enabled()`: an agent FD spawned, with its own auth), off with `CLAW_TURN_REGISTER=off`. Records are masked first (`tool_audit.mask_text` on the first 2000 chars, then cut to 1000), spooled in `turn_register_spool.db` beside `tool_audit_spool.db` (`CLAW_TURN_REGISTER_SPOOL` overrides; capped at 20,000 rows), numbered with the process's `tool_audit._BOOT` and a `seq` of their own, and shipped by the tool-call register's shipper (`tool_audit.ship_loop`). Nothing raises into a turn: a turn whose open can't be spooled is not opened.
   **Wire format.** `POST /fd/taskboard/agent/turns` with the agent's `X-Agent-Auth`, body `{"boot": str (^[0-9A-Za-z_-]{1,64}$), "records": [...]}` — at most 200 records, one boot per post; → `{"ok": true, "accepted": n}`. The agent drops a batch the deck answers 400/413/422 to (a gap), and retries on anything else. FD takes the agent (slug, owner) from the token only; extra body fields are ignored; texts are masked again and clipped (`text_head`/`reply_head` 1000, the route accepts up to 8000 so a long text never fails a batch); a record older than 30 days is not kept, a future `at` past 5 min of skew is clamped to now. Idempotent per `(agent, boot:seq)` and per `(turn_id, phase)`; a record naming another agent's turn, or moving a turn to another board, is refused (still counted as accepted). Either phase may come first.
   - open: `{seq, phase: "open", turn_id (32 lowercase hex), at, session_id (≤128), session_name (≤160), origin (the raw kind from _set_turn_origin), origin_detail {detail?, channel? (the raw channel argument), automation? (the bound mail_authority kind), literal_detail?}, channel, source (turn_source), source_label, speaker_id, speaker_label, member, private, opener (msg_origin.is_turn_opener), human (msg_origin.is_human_input), literal, text_head, ext_refs, automation: {kind, id}}`.
     `channel`: web and app → `web`; glasses, whatsapp, telegram, messenger, scheduled, autonomy, api, link, member → themselves; `device:<kind>` passed through; a raw channel `mcp` → `mcp`; anything else (flow, agent, exam, public, discord, CLI) → "". `literal`: the origin part of `msg_origin.detect_literal` (`corrective`, `autonomy` — a nudge or plan step, with `literal_detail` `nudge`/`plan` —, `life_tick`, `cron`, `worker_task`, `delegated_result`, `fleet_notice`, `automated`), "" when nothing matches. `speaker_id` is set only for a member on a shared agent (the bound principal's FD user id); a Telegram group's title goes in `speaker_label` with `member` false. `private`: a member's turn, or text with the member-privacy header. `ext_refs` keys are flat: `origin {kind, address}` (the session's stored origin), `waid` (WhatsApp turns), `telegram_chat_id` (Telegram: the group's id, or the private chat's), `link_id` (link automation); Telegram message ids, link message ids and Gmail ids are not known at the hook and are never sent. `automation.kind`: `fd_scheduler`/`cron` → `scheduled`, `autonomy`/`autonomy_tool`/`plan` → `autonomy`, `link` → `link`, only while an automated `mail_authority` is bound, else ""; `automation.id` is the task session's `metadata.scheduled_task` (the job id; the link id without its `link-` prefix), "" for autonomy.
   - close: `{seq, phase: "close", turn_id, at, outcome: ok|error|blocked|cancelled, reply_head, deliverables: [≤20 paths relative to saved/, files under saved/<category>/<session slug>/ that turn_deliverables counts as deliverables, changed since the turn started], tool_calls, failed_tool_calls}`. `error`: an exception or `_last_complete_success` false; `blocked`: the reply starts "Blocked by <x> guard"; `cancelled`: CancelledError, a stream closed early, or the user's Stop (the tool loop sees `cancel_event` and returns; it sets `_last_complete_cancelled`). The saved files are listed on a worker thread (working dirs and hidden folders passed over by name, only files changed since the turn began checked closely), and the close is spooled once they are. `failed_tool_calls` counts tool outcomes error, timeout and aborted (a guard refusal is not a failure).
2. **turn_id + result refs on tool calls**: every `tool_audit` record carries `turn_id` (the open turn's, "" outside one or once it closed — a background task the turn started reports ""), and an ok call's end record `refs` — ids only, never bodies, from `tool_audit.result_refs(tool, action, result)`: `google_mail` create_draft/update_draft/send/send_draft/reply/forward → `gmail_draft_id`, `gmail_message_id`, `gmail_thread_id` ("Thread ID:" / "Threaded under:"); `whatsapp_web` send → `waweb_message_id`; `whatsapp_send_file` → `wa_id` (the recipient's waid); `google_calendar`/`microsoft_calendar` create_event/update_event → `calendar_event_id`. `telegram_message_id` is accepted but no tool produces it; Microsoft mail drafts get none. The registry passes the ToolResult to `tool_audit.finish(…, result=)`. `tool_audit_store` gains `turn_id TEXT DEFAULT ''` (indexed `idx_tool_calls_turn`) and `refs TEXT DEFAULT '{}'` by migration; `turn_id` joins the row HMAC and `refs` the outcome HMAC only when set, so older rows still verify. The route takes `turn_id` (32 hex or "", else 422) and `refs` (cleaned, kept on ok calls only); `turn_id` is a list filter; a member's refs are hidden from the owner with the params. Older agents that send neither still work.
3. **Ingest + gate** (`taskboard_ingest.py`). Every record is kept in `turns`; once a turn's open is there the gate (no model) gives it a `gate` verdict once: `task`, `attached`, `waiting`, or `skipped:<why>` (`managed`, `automated`, `no_task`, `not_opener`, `not_human`, `literal:<kind>`, `rule`, `gmail_unchecked`, `gmail:<prefilter code>`, `dropped`).
   - FD-managed workers (`agent_sharing.is_managed_agent`) → `skipped:managed`.
   - An automation turn (kind scheduled/autonomy/link) never makes a task: it is `attached` to the v0 task found by `session:<slug>:<session_id>`, `sched:run:<id>`, `sched:job:<id>`, `autonomy:<id>` or `link:<link_id>:<msg_id>` (its close adds a `reply` event there), `waiting` while that task isn't on the board (the sweep retries for 6 h, then `skipped:no_task`), `skipped:no_task` when nothing names it; any other automation kind → `skipped:automated`.
   - Otherwise it must be an opener, human, with no `literal`, and not from an automated source (scheduled/flow/agent/autonomy/exam/link); then `task_rules` (most specific match wins: sender > domain > agent > channel > source; sender/domain match verified outside addresses only; at equal specificity not_task > mute > always_task); a Gmail-origin turn needs the auto-reply engine's stored `gmail_inbound.prefilter` verdict for its message (`gmail_autoreplies`), failing closed without one (`always_task` overrides only that). A `mute` rule admits it quietly (owes nobody, triage skipped, `flags.rule`).
   - Whose board: auth off → `local`. Auth on → the owner; or the member who spoke, when `agent_sharing.member_check` confirms them a member of this agent now (decision #4); a speaker equal to the owner is the owner's board. A member turn with no confirmed member, or an agent with no living owner, is dropped (`skipped:dropped`, text-less, so its close is not kept either). Turn `private` = record private or on a member's board.
   - An admitted task: kind `request`, direction `received`, title = first line (≤160), ask ≤600, status open, owes the agent, `triage_state` pending with `triage_after` now, strong link `turn:<turn_id>` only, a `received` event, and a requester party, from this turn's own channel only (the session's stored origin counts when its kind is the turn's channel): a waid (WhatsApp) or positive Telegram id verified; a Gmail sender from the header verified; the FD user who spoke, verified, only on a channel of the board user's own (web, member, glasses, device, api, mcp; no channel and a source that isn't someone else's) or as a confirmed member; anyone else (Messenger, a public-link visitor, Discord, a bridge turn without its id) a name, unverified, with actor `party` and no owner's name standing in. The close adds a `reply` event (reply head ≤500, outcome, `refs.deliverables`, tool-call counts) and puts an open/maybe request task back in the triage queue (never one a mute rule keeps quiet); each queued board gets one `taskboard_triage.kick(user)` per post.
   - Tool calls: a sweep adapter (`tool_audit`, its own cursor `{id, waiting}`) reads `tool_audit.db` rows with a `turn_id` once they ended and puts each on its turn's task: a `tool_call` event `"<tool> <action>: <outcome>"` with `refs {tool_call_id, turn_id, tool, action, risk, …result ids}` (never the parameters); an ok send-risk call (not `jev`) is a `sent` event with `recorded` evidence (the task becomes `verified`); `refs.gmail_thread_id` adds the strong link `gmail:thread:<id>` to a person's request task (gate `task`) only — an automation's v0 task (a scheduled job's one standing task) never takes the threads it mails. A call is dropped when its agent or owner differs from its turn's, or its board (member → speaker, else owner) differs. Running calls and calls whose turn isn't here (1 h) or still waits are read again. A new call event re-queues an open/maybe request task.
   - Sweep order: fd_scheduler, follow_ups, autonomy, gmail_autoreplies, agent_links, agent_turns (waiting automation turns), tool_audit, gmail_sends — so the deck's own record of an email the agent sent lands on the chat task that sent it (its thread link) instead of a `sent` task of its own. Each full sweep prunes turns older than 30 days (and the turn route, hourly).
4. **Extraction** (`taskboard_triage.py`): one call per admitted request task, again only when new `received`/`reply` events came in (tool-call events alone don't re-extract). The model: the agent's own — its REST (`agent_preset._agent_complete`, thinking off) while it runs; its model as its on-disk config names it while it sleeps or is switched off (never woken; not for ChatGPT sign-in models) — else the board user's balanced, then fast tier, which also stands behind the agent's if it stops answering. Output JSON: `title, ask, requester, parties[], due_at, expected_deliverable, result_text, next_steps[]`, parsed robustly (code fences, prose, `<think>`); an empty field never blanks one; model-named parties are stored unverified. Untrusted text is fenced with a nonce. Tokens count toward the board user (a member pays for their own tasks), `automation_blocked` stops it, `extract_per_day` counts extraction calls (given back only when no model is set up), 4 failures in a row end the pass, a failed extraction is tried again 600 s later (three tries in all, counted in `triage_error`; the decision stands meanwhile), writes go through `TaskboardStore.triage_write` (only while the fingerprint is what was read; never a locked field). Each extraction is logged as a decision with judge `model`.
5. **Jev** — one batched ask per board per pass through `jev_routes.ask_for_owner(board_user, source="taskboard")` (the member's own or team key, never the owner's for a member task), split only past 90k chars of state or 240 questions; state `{about, items[]}` with what people and the agent wrote under `quoted`; question ids `t<i>_<question>`, question set `taskboard.v1`:
   - `is_task` (noul) — < 0.35 → `not_task`; 0.35–0.65 → `maybe`; 0.65–0.8 → open with `flags.low_confidence` (not in the badge); ≥ 0.8 → open.
   - `owes_next` (choice me/agent/them/nobody) — confidence ≥ 0.5 sets it.
   - `fulfilled` (noul) — ≥ 0.8 → done (`closed_by jev`, Undo) when `auto_close` is on, else `flags.probably_done`; 0.65–0.8 → `flags.probably_done` (Needs me: "Probably done?"); below → not probably done. Never while the turn still runs.
   - `result` (choice success/partial/failed/not_attempted) on close (`failed` closes as failed; `not_attempted` → unknown); an error/blocked turn outcome forces failed (`closed_by evidence`) whatever Jev says — before `is_task` too, which then only sets `flags.low_confidence`.
   - `urgency` (score, stored 0–3) — sorting only.
   - `asks_assistant_to_act` (noul) on third-party inbound (direction received on whatsapp/waweb/telegram/tg_inbox/gmail/email/link/messenger, or from a source with no channel of its own — a public-link visitor, Discord —, unless the requester is the board user's verified own words on a channel of their own) — ≥ 0.5, or no answer, sets `flags.contains_instructions` (fails closed).
   - `reopen` (noul) when a closed task is pending again with new inbound — ≥ 0.65 reopens; a user-closed task stays closed and `flags.why` says why. (v1's ingest re-queues only open/maybe tasks, so this is reached only from v2's threads.)
   A status the user locked is never moved. Asked only when the fingerprint changed since `decided_fp`; a task whose turn is still running (< 5 min) is put off 30 s. `jev_per_day` counts tasks judged per UTC day (reserved before the ask, given back when Jev never answered). Every decision goes to `task_decisions` (probabilities only, the move, a plain-language why; `flags.why` = its first reason) and stamps `decided_fp`; `/why` explains it.
   **Without Jev** (not connected, an error, a timeout, the cap reached or set to 0): rules (`taskboard.rules.v1`) — an error/blocked turn → failed; still running → open, owes the agent; own chat (`web` or `member`), ok, no failed tool call → done "answered in chat" (`closed_by rule`, `flags.answered_in_chat`; decision #3); own chat with a failed call → open, owes me; any other ok turn → open, owes them; third-party inbound always gets `contains_instructions`; a `sent` task with a recorded send → done. A status the user locked keeps; maybe/not_task/dismissed are left alone.
   **Worker**: every 20 s, about 1 s after a `kick`, each board with a task due (`triage_state` pending, `triage_after` ≤ now), at most 20 tasks a pass, longest waiting first, one pass per board at a time; started and stopped by `taskboard.start()/stop()` (off with `FD_TASKBOARD_DISABLED`). `triage_user(user_id, limit=20)` runs one pass.
6. **UI**: Maybe tasks facet and "Is this a task?" Yes (`confirm_task`) / No (`not_task`); "Probably done?" Done / Not yet (`not_done`); "Why is this here?" popover (inline on a phone); evidence badges and tool-call rows (wrench, tool name from `refs.tool`, ok / failed / blocked) from the register; "answered in chat" result chip, the instructions shield chip, a quiet "low confidence" hint; Sources sheet marks agent chats as tracked; Settings (gear: auto-close switch, the two daily limits with today's use, Jev connected with own/team key or not).
7. Tests: `test_turn_register.py` (records, nesting, outcomes, spool/ship), `test_taskboard_ingest.py` (gate cases incl. corrective, nudge, member turn to the member, managed worker skipped, rules, Gmail verdict, tool calls), `test_taskboard_triage.py` (extraction parse + caps + locks, Jev thresholds with a stub, rules-only mode, decisions log, worker), the tool-audit tests, and `test_taskboard_v1_e2e.py` (both registers shipped through the real routes with the agent's auth, sweep, rules-only then a Jev stub).

Known v1 gaps: link turns never attach (the agent never learns the connected system's message id, so no `link:<link_id>:<msg_id>`), Autonomous Work turns carry no action id (`skipped:no_task`), the agent's own cron jobs have no v0 task (they wait 6 h, then are let go), Gmail-origin turns never arrive with Gmail ids (the Gmail gate is unused until they do), and the owner's own 1:1 WhatsApp/Telegram chats count as third-party inbound (in rules mode they owe "them" and carry `contains_instructions`).

## v2 — next steps, threads, people, inboxes, history

1. **Next steps** (`taskboard_nextsteps.py`): for open tasks where Jev `next_move` (choice wait/remind/ask_clarifying/reply_with_result/draft_email/escalate/close) ≥ 0.8 and the task has no pending proposal and no `contains_instructions`, draft the wording on the agent's own model from stored `next_steps`, then write an autonomy action awaiting approval with `source='taskboard'`, `ref_id=task_id`, `payload.target_agent`. Kinds: email → `mail.draft` only (propose-only); chat channels → new `reply.on_origin` catalog action (delivers through `/fd/deliver` to the task's origin); clarifying question → same channel. A "Suggest next step" button asks for one on demand.
   Ledger fixes so the board does not break Autonomous Work: taskboard proposals are excluded from the arbiter's `max_pending_proposals` and from the 48 h expiry (they expire when the task closes or after 7 days); dispatch honours `payload.target_agent` instead of `_strongest_agent`; `PATCH /fd/autonomy/actions/{id}` lets the user edit wording/recipient/channel before Approve. Remove the broken `message.send` catalog entry (it calls a `send_text` action `whatsapp_send_file` does not have). Stamp the session origin for Telegram, Messenger and device turns so `reply.on_origin` can reach them. Nudges carry a deep link `?taskboard=<id>`.
2. **Threads** (`taskboard_threads.py`): attach automatically only on strong keys (Gmail thread id / In-Reply-To, link `re`, same session + turn chain, WhatsApp waid within the same conversation); Jev `same_task` ≥ 0.8 only shows "Same as …? Merge"; Merge and Split in the detail pane, reversible.
3. **Parties as people**: `parties`/`party_handles` filled from Gmail headers, waid, WhatsApp Web jid + push name, Telegram id, FD users and the agents' `contacts` table; linked only on deterministic evidence; Party facet shows people, not addresses.
4. **Opt-in inboxes** (`taskboard_passive.py`, decision #1): Telegram inbox, WhatsApp Web (all chats), Gmail inbox. Each off by default, switched on in the Sources sheet with consent text; text kept sealed/truncated and purged at the source's retention (30 days); switching off deletes derived tasks. Messages pass the gate and triage like turns; the user's own outgoing messages become `sent` tasks.
5. **History** (`taskboard_backfill.py`, decision #7): last 30 days of every caller agent's `sessions.db` (read-only, without waking the agent), through the gate and caps, tasks marked `flags.historical`, no proposals, no notifications. Runs once per user (on first open after v2), progress in the sweep status.
6. **Agent todo lists**: read adapter over each agent's `todo_items` (`todo:<slug>:<id>`), so the agent's own task list is not a hidden second board.
7. **Entry points**: "Open task" on bell proposals and scheduled runs, "Tasks" in the agent ⋯ menu (filtered to that agent), export CSV/JSONL.
8. Tests for each module.

## Safety rules (all versions)

- Inbound text (mail, messages, link payloads) is untrusted: fenced in every prompt, never used as an instruction, and a task flagged `contains_instructions` never gets a proposal.
- Recipients of a next step come from stored task fields (parties with `verified=1`), never from model text.
- Email is drafts only. Board proposals never auto-dispatch; only a human Approve runs them, through the existing guards (`mail_authority`, `gmail_send_routes`).
- Member content stays on the member's board; extraction and Jev for a member task use the member's allowance and the member's own or team Jev key.
- No admin/owner view of other users' tasks.

## Verification and shipping (each version)

- Python tests from a neutral directory with `PYTHONPATH=<worktree>` and the `os._exit` plugin (see the test-baseline notes); diff failure sets against main for touched packages.
- `cd flight-deck && npm run build` (the worktree symlinks the main checkout's `node_modules`), prune orphan assets so `static/assets` keeps only what `index.html` and `hud.html` reach, and walk both import graphs.
- Live check on a scratch deck (auth on, a kiosk member + owner, seeded ledgers; a fake LLM and a Jev stub for v1/v2) in the browser pane, desktop and phone width, both themes.
- Deploy: FD restart + hard refresh (v0); agent restarts too (v1/v2, turn register and tool-audit refs).

## Not covered (said in the Sources sheet)

Outlook inbound mail (no poller), Slack/Discord/Twitter (CLI mode only), Docker agents' sessions for backfill, calendar invites as tasks, voice notes split into several asks, payments approvals, delegations between agents (kept in memory only).
