# Autonomous Work — Implementation Plan

> Status: Phase 1 in progress. Decisions locked: per-user config · propose-only ceiling
> (ship max = "propose", no auto-act until later release) · judge mode = both (LLM + human).

## Thesis

Today the autonomy loop is **open**: the system senses richly (heartbeat, dreams,
reflections, insights) and executes richly (cron, intentions→scheduler, Basna), but
nothing **decides what to do next** and nothing **learns from how it went**. We close
the arc with four connected pieces, all driven by one config object and one new page.

```
        ┌─────────────────── SENSE (exists) ───────────────────┐
        │ heartbeat · dreams/intuitions · reflections · insights │
        └───────────────────────────┬───────────────────────────┘
                                     │  candidate actions
                   Topic 4 ──────────┤  (reflections → intentions)
                                     ▼
                          ┌──────────────────────┐
              Topic 1 ───▶│      ARBITER          │ rank → pick best
                          │  (in heartbeat pulse)  │ reads reliability ◀── Topic 3
                          └───────────┬────────────┘
                                      │ chosen action + risk
                   Topic 2 ───────────┤  (efferent consciousness)
                   ┌──────────────────┴──────────────────┐
          low-risk ▼                            high-risk ▼
    auto-dispatch (cron / WS nudge)       awaiting_approval → page/WhatsApp
                   │                                      │
                   └──────────────┬───────────────────────┘
                                  ▼  EXECUTE (exists)
                            ┌─────────────┐
                Topic 3 ───▶│ JUDGE+LEARN │ score → autonomy_reliability → Arbiter
                            └─────────────┘
```

## Seams
- **Consciousness heartbeat** — FD level, per-user, calls LLM each significant pulse
  (`heartbeat_loop`/`pulse` in `captain_claw/flight_deck/consciousness.py`). Arbiter lives here.
- **Intentions / reflections / nervous system** — agent-runtime (core). Topic 4 lives here.
- **Cron + scheduler** — `captain_claw/cron_dispatch.py` + FD `/scheduler/jobs`. Execution substrate.
- FD→agent dispatch reuses Basna's `_notify_source_agent` (WS notification, `trigger_response=True`).
- agent→FD reuses `_materialize_scheduler` → `/scheduler/jobs`.

## Parametrisation — `AutonomousWorkConfig` (captain_claw/config.py)
Pydantic block alongside `IntentionsConfig`/`NervousSystemConfig`; read fresh via
`get_config()` each tick (live, no restart). Env overrides via `CLAW_AUTONOMOUS_WORK__*`.
Per-user overrides stored in `user_settings` (db.py `get_all_settings`/`set_settings`);
effective config = `user_settings ?? get_config().autonomous_work`.

Fields: `enabled`, `autonomy_level` (off|propose|act_low_risk|act — **shipped ceiling = propose**),
`arbiter_on_pulse`, `arbiter_min_score`, `max_actions_per_day`, `max_concurrent_actions`,
`candidate_lookback_hours`, `quiet_hours_start/end`, `allow_auto_dispatch`, `low_risk_kinds`,
`high_risk_requires_approval`, `learning_enabled`, `judge_mode` (**both**), `reliability_seed`,
`suppress_below_weight`, `reflection_to_intention`, `max_intentions_per_reflection`,
`reflection_intention_max_risk`.

## Data (consciousness DB / new autonomy.db)
- **`autonomous_actions`** — the backlog/ledger: id, user_id, source(reflection|intuition|
  intention|heartbeat|manual), kind(nudge|run_prompt|basna|materialize_schedule), title,
  rationale, risk, score, status(candidate|queued|awaiting_approval|dispatched|done|rejected|
  expired), target, ref_id, created_at, dispatched_at, completed_at, outcome, outcome_note.
- **`autonomy_reliability`** — mirror of `archetype_reliability` keyed by (user_id, kind, domain);
  reuse `_reliability_weight` (Bayesian shrink).

## The page (flight-deck)
Cockpit, two tabs:
- **Control** — renders `AutonomousWorkConfig` (toggles/sliders/selects). Add an `autonomous_work`
  section to `captain_claw/web/rest_settings.py` schema for the field renderer.
- **Activity** — `autonomous_actions` feed; pending high-risk with Approve/Reject; reliability weights.

Files: new `flight-deck/src/pages/AutonomousWorkPage.tsx`, new `flight-deck/src/stores/autonomyStore.ts`
(copy basnaStore `_authedFetch`/401 pattern), add `'autonomous-work'` to `ViewMode`
(`src/types/index.ts`), nav entry in `Sidebar.tsx`, render branch in `App.tsx`. Build:
`npm run build` → committed `../captain_claw/flight_deck/static/` (keep `emptyOutDir:false`).

New router `captain_claw/flight_deck/autonomy_routes.py` mounted in `server.py` near
`consciousness_router`:
- `GET/PUT /fd/autonomy/config` (per-user)
- `GET /fd/autonomy/actions?status=&limit=`
- `POST /fd/autonomy/actions/{id}/approve|reject` (→ `follow_through`)
- `POST /fd/autonomy/nudge` · `GET /fd/autonomy/reliability`

## Topic 1 — Arbiter (`captain_claw/flight_deck/arbiter.py`)
Invoked from `pulse()` after `_reflect()` succeeds, before `save_state`. Gather candidates
(reflection intentions, open intentions, mature intuitions, reflection bullets) → one LLM rank
pass (sees reliability weights) → filter by `arbiter_min_score` + `suppress_below_weight` + caps +
quiet hours → write chosen action to `autonomous_actions`, hand to dispatcher.
`autonomy_level` gates how far: off|propose|act_low_risk|act.

## Topic 2 — Efferent dispatch (`dispatch_action`)
Low-risk + allowed → WS nudge (extract shared `_notify_source_agent` into `fd_dispatch.py`),
or scheduler job, or `/fd/basna/agent/start`. High-risk / propose → `create_proposal(...)` in
`awaiting_approval`, push to page + WhatsApp; Approve/Reject → `follow_through`.

## Topic 3 — Judge & learn (`autonomy_learn.py`)
Extract Basna's score→learn (`score_basna_run` + `record_archetype_outcome` + `_reliability_weight`)
into shared `judge_outcome` / `record_outcome` writing `autonomy_reliability`. judge_mode=both:
LLM auto-judge every outcome + human Approve/Reject overrides. Arbiter reads weights and suppresses
kinds below `suppress_below_weight`.

## Topic 4 — Reflections → intentions
`reflection_to_intentions(reflection)` in `intentions_generator.py`, called from
`maybe_auto_reflect` (`reflections.py`). Gated by `reflection_to_intention`; parse summary bullets,
keep `risk <= reflection_intention_max_risk`, cap `max_intentions_per_reflection`, `create_proposal`.

## Safety
`enabled=false` hard kill switch; defaults ship off/propose (no behavior change until opt-in).
Concurrency cap (default 2), daily cap, quiet hours. Every action logged with rationale (page = audit).
High-risk requires approval.

## Phasing
1. **Phase 1 — Spine + cockpit (off):** config, tables, router, page (Control + read-only Activity). ✅ done
2. **Phase 2 — Arbiter in propose mode + Topic 4:** ✅ done. `arbiter.py` runs inside `pulse()`
   after `_reflect()`; gathers candidates from the consciousness reflection (intentions + thought)
   plus — when `reflection_to_intention` is on (Topic 4) — the latest agent self-reflection bullets
   via `load_latest_reflection()`; one LLM rank pass through the reflection's author agent; filters
   by `arbiter_min_score`, learned-loser suppression (`suppress_below_weight`), dedup vs open ledger,
   daily/concurrent caps, quiet hours; writes the single best as `awaiting_approval` (never dispatches
   — ceiling is propose). `/fd/autonomy/nudge` now forces a pulse; page has "Run arbiter now".
   NOTE: Topic 4 delivered *through* the Arbiter into the unified ledger (one backlog, one approval
   surface) rather than as separate agent intentions.
3. **Phase 3 — Topic 2 dispatch at act_low_risk:** auto-fire low-risk (future release; ceiling is propose for now).
4. **Phase 4 — Topic 3 learning:** outcomes → reliability → Arbiter suppression (read-back already wired).
