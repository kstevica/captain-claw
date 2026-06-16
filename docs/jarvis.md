# Captain Claw — The Jarvis Layer

> A complete reference for the autonomous layer that lets Captain Claw act on the
> user's behalf: how it works, why it's built this way, and what makes it safe.
>
> Status: all five capability gaps implemented and live. Branch history merged to
> `main`. Companion planning docs: `docs/autonomous-work-plan.md` (the closed loop)
> and `docs/jarvis-actions-events-plan.md` (#1 + #2 plan with locked decisions).

---

## 0. The thesis

> *"You're not supposed to prompt the assistant. You're supposed to build a system
> that prompts itself."*

A normal assistant is **request→response**: it waits for you, does one thing, stops.
Jarvis is different — it **notices** things in your world, **decides** what's worth
doing, **acts**, **checks** that the action landed, and **learns** what to do
unprompted next time. The whole point is to remove the human from the *initiation*
of routine work while keeping the human in control of *consequence*.

Captain Claw reaches this in two layers:

1. **The closed loop** (`Autonomous Work`) — the "mind": a self-running cycle of
   **sense → decide → act → judge → learn**. This is what makes the system
   *prompt itself*.
2. **The five Jarvis capabilities** — what turn that mind into an assistant that
   acts on the *real world*, safely:

   | # | Gap | One line |
   |---|-----|----------|
   | 1 | **Hands** | A curated, reversible catalog of real actions it can perform |
   | 2 | **Senses** | It watches your world (calendar, mail) — not just its own thoughts |
   | 3 | **Trust** | It *earns* the right to act without asking, per action, and loses it on failure |
   | 4 | **Planning** | It decomposes a goal into steps and drives them across days |
   | 5 | **Verification** | It confirms a side effect actually happened — trust rests on ground truth |

Everything below explains each layer: the mechanism, the reasoning, the safety, and
where it lives in the code.

---

## 1. Mental model & where things run

Captain Claw is multi-process. Understanding the topology is the key to everything
else, because the Jarvis layer is deliberately split across it.

### 1.1 Flight Deck (FD) — the tenant boundary / orchestrator
- One process (default `:25080`). It is the **"self" per user**: it can see across
  *all* of a user's agents, which is why the autonomous mind lives here and not
  inside any single agent.
- Owns: the **consciousness heartbeat**, the **arbiter**, the **action dispatch**,
  the **event spine**, the **plans engine**, the **Google OAuth tokens**, and the
  per-subsystem databases.
- FD has no model/keys of its own. When it needs to *think*, it **borrows one of
  the user's running agents as a brain** — calling that agent's
  `/api/llm/complete` via `RemoteLLMProvider`. This keeps the whole thing
  multi-tenant by construction: it only ever uses the logged-in user's own agents
  and credentials.

### 1.2 Agents — the workers
- Each agent is a separate `captain-claw-web` process with its own tools, session,
  memory, model, and channel bindings (web/WhatsApp/Telegram).
- FD reaches *into* an agent two ways:
  - **To think**: borrow its LLM (`RemoteLLMProvider` → `/api/llm/complete`).
  - **To act**: send a WebSocket message — a `notification` (proactive nudge, agent
    replies on the user's channel) or a `run_tool` (deterministic, structured tool
    call, no LLM). See §3.

### 1.3 Why this split matters
- A single agent only knows its own sessions. The **consciousness** is *one self
  per user* that watches across all of them, so it must sit at the tenant boundary
  (FD). The arbiter, trust ladder, plans, and event spine all sit beside it.
- Agents hold the *capabilities* (tools + credentials). So FD *decides*, agents
  *do*. The `run_tool` rail (§3) is the bridge.

### 1.4 The databases (each subsystem owns its own SQLite, under `FD_DATA_DIR`)
| DB | Owner | Holds |
|----|-------|-------|
| `consciousness.db` | `flight_deck/consciousness.py` | pulse cursor/state, journal (thoughts/dreams), standing intentions |
| `intuitions.db` | `nervous_system.py` | dream-derived intuitions |
| `sister_sessions.db` | `sister_session.py` | proactive tasks, watches, briefings |
| `autonomy.db` | `flight_deck/autonomy.py` + `plans.py` | **the action ledger, reliability weights, per-user config, live log, plans** |
| `events.db` | `flight_deck/events.py` | **external events + per-source poll cursors** |
| Google tokens | FD `system_settings` | OAuth client + refresh/access tokens (FD-owned) |

This per-subsystem-DB convention means any layer can be inspected, wiped, or
retained independently without touching the loop's state.

---

## 2. The closed loop (Autonomous Work) — the mind

This is the foundation the five capabilities plug into. It runs as a free,
self-driving cycle.

### 2.1 The heartbeat
`flight_deck/consciousness.py` → `heartbeat_loop(stop_event)` started in FD's
lifespan (`server.py`). Every `CONSCIOUSNESS_PULSE_SECONDS` (default **180s**) it
calls `pulse(user_id)` for each user who has running agents
(`distinct_owners_with_agents()`).

`pulse()`:
1. **Sense** — `_gather_delta()` diffs each agent's session list against a stored
   cursor: how many new messages / sessions since the last beat.
2. **Cheap when quiet** — if the delta is below `MIN_NEW_MESSAGES` (default 3) and
   not `force`d, it advances the cursor and **spends no tokens**. Most beats are free.
3. **Reflect** — otherwise it borrows the strongest (or user-pinned "narrator")
   running agent via `_reflect()` and produces a JSON reflection: `thought`,
   optional `dream`, `mood`, `salience`, and a list of standing `intentions`.
   Journaled to `consciousness.db`.
4. **Decide** — calls the **arbiter** (§2.2).
5. **Plan** — auto-advances any active **plans** one step (§7).

A manual **nudge** (`POST /fd/autonomy/nudge` or the Observatory) forces a pulse
regardless of how little changed — the way to test the loop on demand.

### 2.2 The arbiter — the decider
`flight_deck/arbiter.py` → `maybe_run_arbiter(user_id, reflection, author, …)`.
This is the part that turns *noticing* into *one concrete next action*.

**Gather candidates** — the raw material to choose from:
- the reflection's standing `intentions` + current `thought`;
- the latest **agent self-reflection** bullets (`reflections.py`), when
  `reflection_to_intention` is on (this is "reflections → proposed work");
- up to 5 **new external events** (§5) — marked `surfaced` once read;
- a list of **active runs** (for the `stop_run` safety action);
- the **action catalog** (§3), non-human-only entries.

**Rank** — one LLM call through the same agent that authored the reflection. The
prompt is framed as *"turn the best candidate into one concrete action; abstain
only if nothing is actionable."* (Originally it abstained too readily; the prompt
was reframed to propose, with quality enforced downstream.) It returns a JSON
array of 0–1 actions, each `{kind, title, rationale, risk, domain, score, …}`.

**Filter** — keep the single best action that survives:
- `score ≥ arbiter_min_score` (default 0.6);
- not **suppressed** — its learned reliability ≥ `suppress_below_weight` (0.25);
- not a **duplicate** — title not in the open ledger, the lookback window, *or
  recently completed* (this last one stops it re-proposing reworded variants of
  work it already did);
- kind-specific validation (catalog action exists + args valid; `stop_run` target
  is a live run).

**Act** — write the chosen action to the ledger as `awaiting_approval`, then ask
the trust gate (§6) whether it may **auto-fire** now or must wait for approval.

The arbiter is deliberately **single-pick per pulse** — it does the most valuable
*one* thing, not everything. Combined with caps and quiet hours, this is the
primary defense against noise.

### 2.3 Action kinds
| kind | meaning | dispatch |
|------|---------|----------|
| `nudge` | proactively message the user | WS notification → agent replies on the user's channel |
| `run_prompt` | run a task with the agent's tools | agent turn, output captured |
| `basna` | spawn a parallel multi-agent research run | Basna pipeline |
| `materialize_schedule` | create a recurring/scheduled job | cron |
| `stop_run` | halt a stuck/looping run (safety) | in-process cancel; **always propose→approve** |
| `tool_action` | **a concrete catalog action** (#1) | the `run_tool` rail |

### 2.4 Judge & learn
Every dispatched action produces an **outcome** that updates a learned reliability
weight (`autonomy_reliability`, Bayesian-shrunk toward a seed, fails counting
double). Sources of truth, by kind:
- **`tool_action`**: the *grounded* `ToolResult.success`, then **verified** (§8) —
  not an LLM's opinion.
- **`nudge`**: **delivery** is success (a nudge's job is to reach you; LLM-grading
  "did it accomplish reaching out" wrongly failed good nudges).
- **`run_prompt`/`basna`**: an LLM judge of the result.
- **human Approve/Reject** always feeds in too (`judge_mode = both`).

The arbiter reads these weights both to **suppress** losing actions and to power
the **trust ladder** (§6). A live trace of every decision — passes, skips, drops,
dispatches, judge verdicts, errors — is written to `autonomy_log` and shown on the
page, so nothing is swallowed silently.

---

## 3. #1 — Hands: the trusted action catalog

The gap: the loop could *propose* but had no safe, reliable way to *do* real things.

### 3.1 The `run_tool` rail (the missing primitive)
`web/ws_handler.py` handles a new WS message type `run_tool`: it runs **one named
tool with structured args** through the agent's guard
(`_execute_tool_with_guard`) and returns the real `ToolResult` — **no LLM in the
loop**. FD drives it via `actions.run_tool_on_agent` (open WS → send `run_tool` →
await the matching `tool_result`).

Two subtleties that make it work in practice:
- It passes `task_policy={"also_allow": [tool]}` — the **catalog is the authority**
  for a deterministic invocation, so the agent's LLM-oriented per-session tool
  policy doesn't block a pre-vetted action. (The `script_tool` guard still runs.)
- It refreshes the Google-connected cache first — that flag is normally refreshed
  only during a live agent turn, which `run_tool` skips, so without this a
  connected agent would wrongly look disconnected.

### 3.2 The catalog (`flight_deck/action_catalog.py`)
Not every tool — a **vetted subset**, each entry carrying the metadata the loop
needs to act safely: `risk`, `reversibility` (`read_only|reversible|irreversible`),
a `reverse` spec (how to undo), a `verify` spec (how to confirm), and a `grant`.

**v1 catalog:**
- *Reversible · auto-eligible:* `note.write`, `calendar.hold`, `mail.draft`
  (**draft, never send**), `reminder.schedule`.
- *In catalog but human-only (irreversible / outward-facing):* `mail.send`,
  `calendar.invite`, `calendar.delete`, `message.send`, `drive.delete`.
- **Hard-excluded entirely** (never autonomously reachable, by design): raw
  `shell`, `browser` form-submit, social posting, anything payment-like. These stay
  available in normal chat; the autonomous loop simply cannot get to them.

**Why "draft, don't send" / "hold, don't invite":** the auto-eligible set is
chosen so the worst case of an over-eager loop is a draft you delete or a tentative
hold you remove — never an email sent in your name or a meeting that pings others.

### 3.3 Execution + reversibility
`actions.run_action(user_id, action_id, args)`: resolve the catalog entry →
validate args against its schema → pick the user's strongest agent → run via the
rail → **verify** (§8) → return. On success it captures a **reverse handle** from
the `ToolResult` (`build_reverse` pulls the created id — both cron and calendar
print `ID: <value>`), stored on the ledger row. `POST /fd/autonomy/actions/{id}/undo`
(and the **Undo** button) runs that reverse and marks the action `undone`.

`note.write` has no auto-reverse (it's a note; you edit it). `calendar.hold` →
`delete_event(id)`; `reminder.schedule` → `cron remove(id)`.

---

## 4. The arbiter ↔ catalog integration (how a proposal becomes an action)

- The grant-filtered, non-human-only catalog goes into the arbiter prompt; it
  proposes `kind: "tool_action"` with `{action_id, args}`.
- Validation pulls the spec — **risk/reversibility come from the catalog, never the
  LLM** — and drops unknown / human-only / bad-arg proposals.
- Dispatch (`fd_dispatch._dispatch_tool_action`) runs it via `run_action`, records
  the **grounded** outcome (no LLM judge), and captures the reverse handle.

So the LLM chooses *what* and *with which args*, but every safety-relevant
property (risk, reversibility, whether it can auto-fire, whether it worked) is
decided by code and ground truth, not by the model's say-so.

---

## 5. #2 — Senses: the event-ingestion spine

The gap: the loop reacted only to its own reflections about *agent chatter* — it
had no awareness of the user's actual world.

### 5.1 The spine (`flight_deck/events.py`, `event_routes.py`)
A dedicated `events.db` with `external_events`: `{source, event_type, summary,
body, metadata, dedup_key, status, …}`. Status flows
`new → surfaced → acted | ignored` (idempotent; no transactions needed). `dedup_key`
+ a partial unique index prevent re-ingesting the same signal.

`POST /fd/events/ingest` normalizes a signal in; `GET /fd/events` lists them. A
genuinely new event triggers a **debounced force-pulse** so the arbiter reacts
promptly instead of waiting for the 180s tick. The arbiter reads up to 5 `new`
events as candidates and marks them `surfaced` (seen once — no flood).

### 5.2 The poller framework (`flight_deck/event_sources.py`)
An `Adapter` registry + an `events_loop` (started in FD's lifespan). Each adapter
self-gates (`enabled(user_id)` + `requires_google`), has an interval and an opaque
cursor (`poll_state`), and returns events to ingest. A **synthetic** adapter
(`CLAW_EVENTS_SYNTHETIC=1`) validates the whole loop with no credentials.

### 5.3 The Google adapters (`flight_deck/event_sources_google.py`)
FD-side fetch using FD's own OAuth token
(`google_oauth_routes.get_valid_google_access_token()`, refresh-if-needed) — so the
poller works even with no agent up, and parses structured JSON rather than tool text:
- **Calendar**: new/changed events in the next 7 days (incremental via
  `updatedMin`) **and** events starting in the next 24h. The first poll only
  *establishes the cursor* (+ upcoming) so it doesn't dump every existing event as
  "changed."
- **Gmail**: `is:important is:unread in:inbox`, capped, deduped by message id
  (unread messages persist, so re-listing is a no-op once ingested).

Per-user toggles live in the autonomy config (`event_calendar_enabled`,
`event_gmail_enabled`).

### 5.4 The result, in practice
Gmail poller sees new deal emails → arbiter scores them worth surfacing → auto-fires
a `nudge` → an agent **reads the threads, summarizes them, and surfaces a decision**
("Ales wants the green light — approve?"), in the user's language. Senses → hands →
useful proactive output, unprompted.

---

## 6. #3 — Trust: the ladder

The gap: auto-fire was a binary manual grant. Real trust should be *earned* and
*revocable*, per action.

### 6.1 The three rungs (per action)
Driven by the learned reliability weight for that action (`autonomy_reliability`,
keyed **per `action_id`** via `reliability_key()` — so `calendar.hold` earns trust
independently of `mail.draft`):

```
weight < suppress_below_weight (0.25)              → SUPPRESSED  (not even proposed)
suppress_below ≤ weight < trust_threshold          → PROPOSE     (awaits your approval)
weight ≥ trust_threshold (0.85) over ≥ trust_min_runs (3)
                                                   → TRUSTED     (auto-fires)
```
Plus an explicit **grant** (`granted_actions`) = a manual floor that trusts an
action outright, skipping the earning.

### 6.2 How it's wired
- `should_auto_dispatch` (in `fd_dispatch.py`) gates `tool_action` auto-fire on:
  reversible + low-risk + not-human-only + `autonomy_level ≥ act_low_risk` +
  `allow_auto_dispatch`, **and** (granted **or** earned). Irreversible/human-only
  *never* auto-fire, regardless of trust.
- **Earning**: every Approve and every verified success raises the weight; the
  action climbs from *propose* to *trusted*.
- **Demotion**: failures (and Rejects) drop the weight (fails count double in the
  Bayesian formula); a trusted action that starts failing falls back to *propose*,
  then *suppressed* — automatically.
- The arbiter's suppression is also per-action-id, so one bad action can't mute the
  rest.

The "Allowed actions" panel shows each action's live rung:
`propose` → `learning 0.72 · 4✓✗` → `auto · trusted 0.87` (or `auto · granted`,
or `suppressed`).

### 6.3 Why this is the safety keystone
Trust is the dial between "helpful" and "dangerous." Tying it to *demonstrated,
per-action reliability that you control via approvals* means the system can only
earn autonomy on the specific things it has repeatedly gotten right — and loses it
the moment it doesn't. The ceiling (`max_autonomy_level`, shipped at
`act_low_risk`) caps how far any of this can go regardless of earned trust.

---

## 7. #4 — Planning: the goal executor

The gap: the arbiter picks one next action; it couldn't own a multi-step goal to
completion.

`flight_deck/plans.py`:
- **Decompose** (`decompose_goal`) — one LLM pass turns a goal into an ordered list
  of catalog-validated steps (`tool_action` / `nudge` / `run_prompt`). Human-only
  and invalid steps are dropped.
- **Advance** (`advance_one`) — runs the next pending step through the **same rail**
  (`run_action` → trust ladder → verification → reverse capture). `auto=True` (from
  the heartbeat) runs only **trust-eligible** steps and *pauses* at untrusted ones;
  a manual "Run next step" *is* the approval. The heartbeat advances active plans
  one step per beat, so a plan progresses across pulses/days.
- **Rollback** (`abandon_plan`) — undoes completed reversible steps newest-first via
  their captured reverse handles.

Routes: `GET/POST /fd/autonomy/plans`, `/plans/{id}/advance`, `/plans/{id}/abandon`.
The Plans UI panel shows the goal, the step ladder with live status, and
Run-next-step / Abandon.

*Deferred:* **replan-on-failure** — currently a failed step fails the plan; the
intended enhancement is to re-decompose the remainder.

---

## 8. #5 — Verification: trust on ground truth

The gap: even a grounded `ToolResult.success` only means *the tool reported*
success. To raise trust safely, the system must confirm the side effect *exists*.

In `actions.run_action`, after a catalog action with a `verify` spec succeeds, it
**reads the side effect back** (`_verify_side_effect`):
- `note.write` → `read(path)`; `calendar.hold` → `get_event(event_id)`.
- **confirmed** (read ok) → outcome stays success.
- **absent** (read says not-found/404) → outcome **downgraded to fail** — so trust
  never builds on phantom successes, and reliability drops.
- **unknown** (the read itself errored — transient) → keep success, flag
  `[unverified]` (no false-negative penalty).

This applies to both arbiter-dispatched actions and plan steps (both go through
`run_action`). It's the floor under the trust ladder: a calendar hold that *claims*
to exist but doesn't is recorded as a failure, so the ladder won't promote it.
Gated by `verify_enabled` (default on).

---

## 9. The complete lifecycle (worked example)

> *Your flight gets rebooked; a deal email lands.*

1. **Sense** — the Gmail poller (#2) sees the airline + deal emails →
   `external_events` rows (`new`).
2. **Wake** — the new events debounce-trigger a forced pulse.
3. **Reflect** — the heartbeat borrows your strongest agent; the reflection + the
   events become candidates.
4. **Decide** — the arbiter ranks them and picks the single best action: propose
   `tool_action: calendar.hold` for the new flight time, or a `nudge` summarizing
   the deal thread.
5. **Trust gate** (#3) — is `calendar.hold` trusted/granted? If yes →
   auto-fire; if not → it lands in *Awaiting approval* for one tap.
6. **Act** (#1) — `run_action` invokes `google_calendar.create_event` via the rail.
7. **Verify** (#5) — read the event back by id → confirmed.
8. **Learn** — verified success raises `calendar.hold`'s reliability; a few more and
   it graduates to auto.
9. **Reversible** — the ledger keeps the reverse handle; an **Undo** removes the
   hold if you didn't want it.
10. **Plan** (#4) — if you'd said "prep for the trip," this would be one step of a
    multi-step plan the loop drives to completion.

Every step of this is in the **live log**, and every consequential one is gated by
approval or earned trust.

---

## 10. Safety model (why you can leave it on)

Defense in depth — no single mechanism is load-bearing:

1. **Off by default.** `enabled=false`; nothing acts until you opt in per user.
2. **Graduated ceiling.** `autonomy_level` (`off | propose | act_low_risk | act`)
   is clamped to `max_autonomy_level` (shipped at `act_low_risk`). "act" is not
   reachable without a code change.
3. **Catalog curation.** Only vetted actions are reachable; raw shell / browser /
   social / payments are hard-excluded, not merely disabled.
4. **Reversible-first.** Only reversible, low-risk actions can auto-fire;
   irreversible/outward-facing ones are human-only.
5. **Earned, revocable trust** (#3) — per action, demotes on failure.
6. **Grounded verification** (#5) — phantom successes become failures.
7. **One-tap undo** — reverse handles captured for reversible actions.
8. **Caps & quiet hours.** `max_actions_per_day`, `max_concurrent_actions`,
   `quiet_hours_*`; the arbiter does one thing per pulse.
9. **Stop actions.** A `cancel` primitive + UI Stop buttons for Basna/Council; the
   arbiter can propose `stop_run` for a runaway (always approval).
10. **Recursion guard.** Basna workers are spawned without the `basna` tool +
    a `CLAW_BASNA_WORKER` env guard — a worker can't start another run.
11. **Run-rate breaker.** A deterministic burst limiter on agent-started Basna runs.
12. **Full audit.** Every action and decision is in the ledger + live log.

The throughline: **the LLM decides *what*; code and ground truth decide *whether it
may* and *whether it worked*.**

---

## 11. Configuration reference

Per-user, stored in `autonomy.db` (`autonomy_config`), overlaid on the global
defaults from `AutonomousWorkConfig` in `config.py`; edited from the **Autonomous
Work → Control** page or `PUT /fd/autonomy/config`. Read fresh each tick (live, no
restart).

**Master:** `enabled`, `autonomy_level`, `max_autonomy_level` (server-owned ceiling).
**Arbiter:** `arbiter_on_pulse`, `arbiter_min_score`, `max_actions_per_day`,
`max_concurrent_actions`, `candidate_lookback_hours`, `quiet_hours_start/end`.
**Dispatch:** `allow_auto_dispatch`, `low_risk_kinds`, `high_risk_requires_approval`.
**Learn:** `learning_enabled`, `judge_mode` (`auto|human|both`), `reliability_seed`,
`suppress_below_weight`.
**Trust (#3):** `trust_threshold` (0.85), `trust_min_runs` (3).
**Verify (#5):** `verify_enabled`.
**Reflections→intentions:** `reflection_to_intention`, `max_intentions_per_reflection`,
`reflection_intention_max_risk`.
**Grants (#1/#3):** `granted_actions`.
**Event sources (#2):** `event_calendar_enabled`, `event_gmail_enabled`.

**Global events (`EventsConfig`):** `poll_seconds`, `calendar_interval_seconds`,
`gmail_interval_seconds`.

**Env:** `FD_EVENTS_DISABLED`, `FD_CONSCIOUSNESS_DISABLED`, `CLAW_EVENTS_SYNTHETIC`,
`CONSCIOUSNESS_PULSE_SECONDS`, `FD_DATA_DIR`.

---

## 12. File map

| Area | File |
|------|------|
| Heartbeat / consciousness | `captain_claw/flight_deck/consciousness.py` |
| Arbiter (decider) | `captain_claw/flight_deck/arbiter.py` |
| Dispatch + trust gate | `captain_claw/flight_deck/fd_dispatch.py` |
| Autonomy store (ledger/reliability/config/log) | `captain_claw/flight_deck/autonomy.py` |
| Action catalog | `captain_claw/flight_deck/action_catalog.py` |
| Action execution + verify + undo | `captain_claw/flight_deck/actions.py` |
| `run_tool` rail | `captain_claw/web/ws_handler.py` |
| Event spine store | `captain_claw/flight_deck/events.py` |
| Event routes (ingest/list) | `captain_claw/flight_deck/event_routes.py` |
| Poller framework | `captain_claw/flight_deck/event_sources.py` |
| Google pollers | `captain_claw/flight_deck/event_sources_google.py` |
| FD Google token helper | `captain_claw/flight_deck/google_oauth_routes.py` |
| Plans engine | `captain_claw/flight_deck/plans.py` |
| Autonomy + plans + events API | `captain_claw/flight_deck/autonomy_routes.py`, `event_routes.py` |
| Config | `captain_claw/config.py` (`AutonomousWorkConfig`, `EventsConfig`) |
| UI | `flight-deck/src/pages/AutonomousWorkPage.tsx`, `stores/autonomyStore.ts` |

---

## 13. Operating it

1. **Enable** (per user, Autonomous Work → Control): `enabled` on; `autonomy_level`
   `propose` (watch first) or `act_low_risk` (let trusted things fire);
   `allow_auto_dispatch` on; quiet hours `0/0` while testing.
2. **Grant or earn**: tick an action under *Allowed actions* to trust it outright,
   or just approve its proposals a few times and watch it climb to *trusted*.
3. **Senses**: toggle Calendar/Gmail under *Event sources* (needs FD Google
   connected). For a no-creds smoke test, run FD with `CLAW_EVENTS_SYNTHETIC=1`.
4. **Test on demand**: *Run arbiter now* forces a pulse; the **Live log** shows
   exactly what it considered and why. *Recent events* shows the feed; the
   *Action ledger* shows what it did (with Undo on reversible successes).
5. **Plan**: give it a goal under *Plans*, then Run-next-step or let the heartbeat
   advance trusted steps.

A new event auto-nudges the arbiter (debounced); a significant pulse reflects and
decides; trusted reversible low-risk actions fire and get verified; everything else
waits for your tap.

---

## 14. What's intentionally not done (and why)

- **`act` autonomy level** — gated behind `max_autonomy_level` until there's reason
  to let normal-risk actions auto-fire. A deliberate ceiling.
- **Irreversible auto-fire** — `mail.send`, deletes, messaging others stay
  human-only by design, even when "trusted."
- **#4 replan-on-failure** — a failed plan step fails the plan; re-decomposition of
  the remainder is the planned enhancement.
- **Gmail filter breadth** — `is:important is:unread in:inbox` is conservative; it
  can pull Google Docs comment notifications. Tunable.
- **Webhook push** — Calendar/Gmail are polled (5-min default); a real-time push
  receiver is a future optimization.

---

## 15. Design principles (why it works)

1. **The mind is one thing; capabilities plug in.** The closed loop is the
   substrate; #1–#5 are adapters on it. Everything reuses the ledger, reliability,
   dispatch, and verification — no parallel systems.
2. **The LLM proposes; code disposes.** Models pick intent and arguments; risk,
   reversibility, permission, and success are decided deterministically and by
   ground truth.
3. **Trust is earned, scoped, and revocable.** Per action, via demonstrated
   reliability you control — never a blanket switch.
4. **Reversible by default; verified always.** The worst auto case is something you
   can undo, and the system checks its own work.
5. **Nothing is swallowed.** Every decision and action is logged and visible; the
   human can always see, approve, undo, and stop.
6. **Cheap when idle.** Quiet beats cost nothing; one action per pulse; conservative
   defaults. Autonomy that's affordable to leave running.

The result is a system that genuinely *prompts itself* — it notices your world,
decides, acts through trust it has earned, plans across time, verifies, and
learns — while every consequential lever stays in your hands.
