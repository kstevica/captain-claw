# First Mate — plan

First Mate runs **factories** for the user's real-life tasks. A factory is a
job that keeps going: a charter that says what it's for and what done looks
like, checks its workers can't see, a runner that does the work (usually Bat),
a budget, triggers and a scorecard. First Mate starts the factories' runs,
gives them their share of the budget, watches how well each one does, and asks
the user only when it has to.

It is the "software factory" idea (specs in, hidden scenarios, a satisfaction
score instead of green tests, nobody reading the code) brought to work that
isn't code: inbox drafts, follow-ups, reports, price watches, meeting prep.

Almost everything it needs already exists. This plan adds the parts that hold
it together: a factory that lasts between runs, one handle over every kind of
run, a shared budget, a scorecard over time, and First Mate's own loop.

## User decisions (2026-11-17)

| # | Question | Answer |
|---|----------|--------|
| 1 | Where does First Mate live? | **Under the user's own account, for now.** Its factories work on the user's mail, calendar and files, as the user's own runs. The Superagent ([superagent-plan.md](superagent-plan.md)), an artificial user with its own account and mailbox, stays a separate plan. |

## 1. Real life is not code

A code factory can run its software a thousand times against hidden
scenarios, because nothing outside the test changes. The real world can't be
replayed: an email sent is sent. So a real-life factory works like this:

1. **Practise** with nothing reaching the world (a rehearsal: outward actions
   become intents).
2. **Do it once**: the intents are carried out in one run, within the grants
   the user approved.
3. **Prove it happened**: from records the model can't write (the tool-call
   register) and from the source itself (Gmail has the message, Calendar has
   the event).
4. **Score it later** from what the user did with the result: sent it as
   written, edited it, ignored it, undid it.

Captain Swarm already works this way for one task at a time (rehearsal, commit
point, carry out once). First Mate makes it the normal way a standing job
runs.

## 2. Names

- User-facing: **First Mate**, **factory**, **charter**, **scorecard**.
- Internal id `first_mate`; modules `flight_deck/first_mate_*.py`; store
  `FD_DATA_DIR/first_mate.db`; routes `/fd/first-mate/*`; agent tool
  `first_mate`; table prefix `fm_`.
- The thing that does a factory's work is its **runner** (Bat, Vatra, Basna,
  Code, Crew, a flow, a plan). Never call it the factory's "line": **the
  Line** is the user's timeline. Never call it "crew" (Crew is the CLI
  workers) or "swarm" (BotPort owns that id).

## 3. What it stands on

| Need | What exists | Module |
|------|-------------|--------|
| One run that finishes honestly | Bat: fixed contract, steps in parallel, fix rounds, a judge on a different model, survives deck restarts, asks the owner, spend ledger, token cap | `bat_loop.py`, `bat_judge.py`, `bat_store.py`, `bat_routes.py` |
| Child runs counted and stopped | Bat adopts Basna, Vatra, Code and Crew runs started under it, adds their spend, stops them with the run | `bat_nested.py` |
| Other runners | Vatra (multi-part deliverables), Basna (truth-finding), Code, Crew, flows, plans (multi-day catalog actions with rollback) | `vatra_routes.py`, `basna_routes.py`, `crew_runner.py`, `flow_runner.py`, `plans.py` |
| Practise without acting | Rehearsal: outward actions recorded as intents, browser stopped at the commit point; carry out once through Bat | `rehearsal_turn.py`, `captain_swarm_rehearsal.py`, `captain_swarm_outcome.py` |
| Checks the workers can't see | Captain Swarm's sealed folder, held-out cases, "a proposer never sees held-out numbers" | `captain_swarm_loop.sealed_dir`, `captain_swarm_document.py`, `captain_swarm_learning.py` |
| Proof | Tool-call register (agent side), turn register, Gmail/Calendar read-back | `tool_audit.py`, `turn_register.py`, `taskboard_proof.py` |
| Why the work exists | Course goals (deck, user, agent) with horizons, links, weekly progress | `course.py`, `course_progress.py` |
| Senses | Event store with Gmail and Calendar pollers | `events.py`, `event_sources_google.py` |
| The user's window | Line (timeline, Needs you, phone channels, threads), durable asks with secrets kept out of storage | `line.py`, `human_ask.py` |
| Hands | Action catalog with risk, undo and a trust ladder; Bat's mail, spend and account grants | `action_catalog.py`, `bat_routes.py` |
| Time | Deck scheduler, agent sleep and pre-wake | `fd_scheduler.py`, `agent_idle.py` |
| Calibration | Judge-vs-human agreement (Cohen's kappa) | `label_eval.py` |
| Getting better | Captain Swarm runs on a way of working, lessons per family, playbooks | `captain_swarm_*.py` |

## 4. A factory

One row in `fm_factories`, with its versions in `fm_versions`. A change to the
charter, checks, runner or grants is a new version; runs record which version
they ran.

| Field | What it holds |
|-------|---------------|
| `title` | "Inbox drafts", "Follow-up chaser", "Weekly report" |
| `charter` | What it's for, what done looks like for one run, what it must never do. Plain text, quoted into every run's brief. |
| `goal_id` | The Course goal it serves (`course_links` kind `factory`), optional |
| `runner` | `bat` (default), `vatra`, `basna`, `code`, `crew`, `flow`, `plan` |
| `runner_config` | Worker mode, step outline, tiers, a flow id |
| `triggers` | Schedule, events filter, on demand (§9) |
| `grants` | What its runs may do in the world: `email`, `organize`, `spend`, `accounts`, with limits (replies only to existing threads, at most N sends per run, `per_item_usd`, weekly `real_usd_cap`) |
| `trust` | `propose` · `commit` · `direct` (§6) |
| `budget` | Token share per week, token cap per run (§10) |
| `checks` | Run checks for Bat's contract, the sealed case set, the satisfaction target (§7, §8) |
| `state` | `draft` · `awaiting_charter` · `running` · `paused` · `paused_budget` · `stopped` · `retired` |

Other tables: `fm_runs` (one per run: factory, version, trigger, runner,
child id, status, tokens, about-$, verdict, proof, score), `fm_intents`
(what a rehearsal would have done), `fm_signals` (what the user did with a
result, §8), `fm_events` (history for the factory's drawer).

Starter templates: inbox drafts, follow-up chaser, weekly report from
connected sources, price or availability watch, meeting prep from the
calendar, research digest on a topic.

## 5. First Mate's loop

First Mate is a **deck-side supervisor** with a **voice agent**.

**The supervisor** (`first_mate_loop.py`) ticks every 60 seconds, started in
the deck's lifespan like Bat's supervisor. Most ticks call no model. Each tick,
per user with First Mate on:

1. Collect due triggers: schedules due, event batches ready, Run now presses.
2. Drop what can't run: paused factories, factories over their share,
   factories with a run already going (one run per factory at a time).
3. Order the rest: Course goal horizon and the factory's own priority, then
   how overdue it is, then health (a red factory never starts).
4. Start as many as there are free slots (§5.1) and budget.
5. Poll running runs through their runner (§5.2); add spend.
6. Score finished runs: verdict, proof, intents (§6–§8). Post to the Line.
7. Pause factories that hit a brake (§13).

The supervisor holds a lock per scan, as Bat's and Captain Swarm's ticks do
(overlapping scans once gave a Bat run three drivers).

**The voice agent.** Turning First Mate on creates an agent called First Mate
in the user's account from an archetype. The user may instead point First
Mate at an agent they already have. That agent:

- is where the user talks to First Mate (its chat and the Line's ask box);
- does First Mate's thinking on its own model and key: drafting charters,
  routing events that rules can't place, the weekly review (deck calls for
  one agent go through that agent's model);
- gets the `first_mate` tool (§11);
- sleeps when idle like any agent (`agent_idle.py`). Bat workers are spawned
  fresh per run, so factory runs never wait for it.

**Next to the Arbiter.** The Arbiter stays as it is for nudges and single
actions. Events a factory claims aren't Arbiter candidates (§9). In P5 the
Arbiter gets one more proposal kind, `factory`: "start a factory for this
stalled goal".

**Next to Vatra.** First Mate is Vatra's Lead one level up: factories instead
of specialists, the charter approval instead of Group 0's plan gate, a
scorecard instead of the done-gate. The difference is that a Vatra run ends
when its deliverable is done, while factories keep going.

### 5.1 Slots

Bat allows `_BAT_MAX_RUNS_PER_OWNER` (3) running runs per owner. First Mate
creates Bat runs in the deck process, the way Captain Swarm's `BatLane` does,
so the route's check doesn't apply and First Mate counts for itself:

- free slots = Bat's limit − the owner's busy Bat runs − 1;
- the −1 keeps one slot for the user's own runs;
- Crew and Code slots follow their own grants.

### 5.2 One handle for every runner

`first_mate_runs.py` gives every runner the same three calls:

- `start(factory, trigger, brief) → child_id`
- `poll(child) → {status, tokens, usd_about, result, done}`
- `stop(child)`

| Runner | How |
|--------|-----|
| Bat | `bat_store.create_run(...)` + `bat_loop.kick()`, as `captain_swarm_outcome.BatLane.start` does. Config: `source: "first_mate"`, `factory_id`, `token_cap`, and for a carry-out `plan_approved` plus the charter's grants (§6). |
| Basna, Vatra, Code, Crew | The readers and stoppers in `bat_nested.py` (`_read_team`, `_read_code`, `_read_crew`, `_stop_child`), moved into a shared module that Bat and First Mate both import, with no change to Bat's behaviour. |
| Flow | `flow_runner` (its run states, `request_stop`) |
| Plan | `plans.advance_one` |

Bat is the default runner because it brings the contract, the independent
judge, fix rounds, asks, the spend ledger and restart recovery.

## 6. Outward actions: practise, do once, prove

Each factory has a trust level:

| Level | What a run does |
|-------|-----------------|
| `propose` | The run is a rehearsal. Its intents go to the Line as one item ("Inbox drafts would: reply to 3 threads, archive 5"). Approve → carried out once. Reject → recorded as a signal. |
| `commit` | The run is a rehearsal, then the intents are carried out without asking, but only the ones inside the charter's grants and limits. Anything outside them goes to the Line as in `propose`. |
| `direct` | No rehearsal: the run acts itself. For factories whose actions are reads and their own files, or reversible low-risk catalog actions. |

**Moving up.** Every factory starts at `propose`. It moves up one level when
its last 10 carried-out runs have satisfaction ≥ 0.9 and no proof doubt. One
proof doubt or one undo drops it back one level. The user can pin a level:
a pin stops the record from moving it up, but a doubt or an undo still drops
it and removes the pin. A version that adds a grant sets trust back to
`propose`, because trust was earned for the old grants. This is the action
catalog's trust ladder (0.85 over 3 runs), made stricter because one run does
more than one action.

**Carrying out.** A carry-out is one normal Bat run, created with
`plan_approved: true` and exactly the grant flags the charter lists
(`email_allowed`, `organize_allowed`, `spend_allowed`, `account_allowed`).
Bat's `_gate_check` already lets such a run through for those grants and asks
again for anything beyond them. The grants live on that run's config and are
never copied to a child run (Bat already does this). Sending still goes by the
user's own switches (Connections → Google → Email sending, inbox send gates).

**Rehearsal tags.** Today a tag (`captain_swarm_rehearsal.make_tag`) names a
Captain Swarm run and experiment, and intents land in `cs_intents`. First Mate
needs its own tag namespace (`fm:<factory>:<run>`), with intents written to
`fm_intents`. The guard in `rehearsal_turn.py` stays as it is; only the place
it records to depends on the tag.

**Proof.** After a carry-out, `taskboard_proof.recorded` / `check` /
`confirm` read the run's turns against the tool-call register and read Gmail
and Calendar back. A doubtful proof is a signal, puts a Needs-you item on the
Line and drops the trust level.

**Real money** stays off for factories until P3 and until Bat's spend path
(`FD_BAT_SPEND`, the checkout interceptor) has been tested live. Then it runs
through Bat's ledger with a per-factory weekly `real_usd_cap` and a
`per_item_usd` threshold for asking first.

## 7. Checks the workers can't see

Three layers:

1. **Run checks.** Bat's contract and judge, as today.
2. **Sealed cases.** Each factory keeps a set of cases: inputs and what a good
   result looks like.
   - **Open cases** may appear in the brief as examples.
   - **Sealed cases** live in `FD_DATA_DIR/first_mate/sealed/<factory>/`,
     out of any worker's reach, and are never shown to whoever writes or
     improves the factory's brief (Captain Swarm's `shown_verdict` rule).
   - They run in rehearsal when a new version is proposed (a version that
     scores lower than the current one doesn't go live) and once a week (a
     drop turns the factory amber).
   - As built (P4): up to 30 cases a factory, each an input and one to ten
     lines saying what a good result does; graded full, partial or none per
     line. A check runs one case at a time (`kind` `check`, a rehearsal that
     proposes nothing), and its runs stay out of the run list and the day's
     run count but count against the factory's share. Approving a draft of a
     live factory that has cases starts a check of the draft; it goes live
     when it scores within 5 points of the live version's last score, through
     the same hash-pinned approval, so a draft edited during the check never
     goes live. Otherwise it stays waiting, the owner is told on the Line, and
     may approve it anyway. A weekly score 10 points or more under the one
     before turns the factory amber. Open cases go into the brief as examples
     (three at most, never the case being checked).
   - A finished run that started on new mail can become a case: its emails
     are the input, its contract the first guess at the lines.
3. **Owner signals** (§8): the real satisfaction.

Building the cases: reuse Captain Swarm's practice sets where a kind exists
(inbox triage, calendar invitations, the document checklist). For a new kind,
take the factory's first real runs once the user's signals have labelled them.

Judged checks use a reviewer from a different model family than the workers
(Bat's `_panel_creds`). No answer from a reviewer counts as failed, never as
passed.

## 8. Satisfaction and the scorecard

**Signals** (`fm_signals`), one per result:

| Source | Signal |
|--------|--------|
| Line | Approve, reject, 👍, 👎 on a factory's item |
| Gmail | Draft sent as written; sent after edits (scored by how much changed); draft deleted |
| Taskboard | Done, reopen, not done on a task the factory handled |
| Actions | An undo of something the factory did |
| Nothing for 7 days | Unknown, left out of the score |

**Satisfaction** = good ÷ (good + bad) over the last 20 known signals. Sent as
written, approved and 👍 count 1. A light edit counts 0.7. Rejected, undone,
reopened and 👎 count 0.

**Scorecard** per factory: runs this week, done %, judge passes, proof
doubts, satisfaction, latest sealed score, tokens per run, about-$ per run,
median time, asks per run.

**Health:**
- green: on target;
- amber: satisfaction under target, or the sealed score dropped;
- red: 3 failed runs in a row, a proof doubt at `commit` or `direct`, or the
  budget hit twice in a row. A red factory pauses.

**Calibration before trust.** Before satisfaction may move a trust level, it
is checked against the user's own labels with `label_eval`: kappa ≥ 0.6 on at
least 30 labels, the same bar Captain Swarm uses. Until then trust moves only
by hand.

As built (P4): every finished run gets the deck's own verdict with no word
from the user (failed, a leak or a doubtful proof is bad; finished is good).
The user's label for the same run is their 👍/👎, else approving or rejecting
what it would do, else what became of its drafts (sent with 70% or more kept
is good). Kappa runs over the user's latest 200 labelled runs across their
factories. When the user and the deck called every run good, kappa can't be
worked out, and that doesn't count as calibrated. The record behind a move is
the last 10 carry-outs and outright rejections since trust last moved: a
carry-out counts what the user did with its result, else the deck's verdict,
times the share of its proposal they approved; a rejection counts 0. The
supervisor raises a running, unpinned factory one level when they average
≥ 0.9 with no doubt and the user is calibrated, and says so on the Line.

## 9. Triggers and events

- **Schedule**: the same schedule strings scheduled tasks use, with the next
  time from `fd_scheduler.compute_next_run`. First Mate's tick fires it, not a
  scheduler job, because a scheduler job prompts one agent while a factory
  starts a run.
- **Events**: a filter on `external_events` (source, event type, sender,
  label, words). A router runs before the Arbiter:
  - rules first;
  - when two or more factories match, or a filter asks for it, one model pass
    on the voice agent picks;
  - the claimed event gets status `claimed` (new) and
    `metadata.factory_id`; the Arbiter reads only `new`;
  - each factory has a batch window (default 15 minutes), so 20 new emails
    make one run, not 20.
- **On demand**: Run now on the page, the `first_mate` tool, the Line.

Event content reaches a run quoted as data, labelled with where it came from,
the way the Taskboard's hand-over fences what others wrote. The charter and
the user are the only sources of orders.

## 10. Budget

- **LLM caps are in tokens, never dollars** (the owner's rule from Captain
  Swarm). Dollars appear only as "about $x". Yardstick: ~5M tokens per dollar
  on Haiku-class workers.
- **First Mate's weekly budget** is a user setting (default 25M tokens). It
  sits inside the user's `usage_quota` allowance, never beside it.
- **Each factory** gets a share of that budget (a weight; default equal) and
  a cap per run (default 3M, passed to Bat as `config.token_cap`).
- A factory that spends its share pauses (`paused_budget`) until Monday, with
  a Line item. Unused shares don't roll over.
- Rehearsals, sealed-case runs and carry-outs all count against the
  factory's share.
- Spend is read per runner: Bat's cumulative total, including the children
  `bat_nested` adds; the others from their own cost records.

## 11. Talking to First Mate

**The `first_mate` tool** (owner's live turns only). It is refused inside Bat,
Vatra and Basna workers and in rehearsals (the `CLAW_BAT_WORKER`,
`CLAW_VATRA_WORKER`, `CLAW_BASNA_WORKER` and `CLAW_REHEARSAL` markers).
Actions:
- `list` and `status`;
- `run`, `pause`, `resume`;
- `draft`: writes a charter draft and puts an approval card on the Line;
- `report`.

**A new factory** starts from a template, from a Course goal or from the
user's words. First Mate drafts the charter, the run checks and the first
sealed cases. The user sees an approval card with fields they can edit, like
Vatra's Group 0 plan editor, and approves or cancels. Adding a grant or
raising a limit is a new version and needs approval again. The approval is
checked against the version's content hash, so a draft changed after the user
looked at it can't be approved without another look. Moving trust up by hand
is the owner's own act: only the owner can do it, and it's recorded in the
factory's history.

**The weekly report** is one Line item, "First Mate's week": a scorecard line
per factory, spend against budget, and what it proposes (pause a red factory,
change a share, a new factory for a stalled goal, an improvement run, §12).
Each proposal can be accepted from the item.

## 12. Factories get better

A factory below its satisfaction target for two weeks leads to a proposal for
an **improvement run**: a Captain Swarm run measured on the factory's sealed
cases. The run varies the factory's brief, rules, step outline and runner
settings (Swarm's Kind A, a better way of working). A kept gain becomes a new
version: the user approves it and it passes the regression check (§7).
Lessons stay in Captain Swarm under the family `factory:<id>`.

Captain Swarm is admin-only today (#402). While First Mate is also admin-only
this changes nothing. Opening First Mate to members means deciding the same
for improvement runs.

## 13. Safety and containment

- **Never more than the user.** Factory runs are the user's own runs, within
  the user's switches and gates.
- **Grants only from an approved charter.** They are pinned on the carry-out
  run and never copied to its children.
- **Bat's floors apply.** The system-harm denylist (`bat_floor.py`), deck
  confinement and the tool-call register cover every Bat worker.
- **Kill switch.** Turning First Mate off pauses every factory and stops
  running runs (Bat's cancel, and `bat_nested.stop` for their children).
- **Brakes:**
  - at most 8 factories per user;
  - running runs limited to Bat's limit − 1 (§5.1);
  - the weekly token budget;
  - pause after 3 failed runs in a row;
  - event batch windows;
  - at most 24 runs per factory per day.
- **Prompt injection.** Mail, pages and messages are data with their source;
  only the charter and the user give orders.
- **Who has it.** A deck switch `first_mate`, off by default. Admins only
  through P3, as Captain Swarm was for testing.
- **Privacy.** Sealed cases are built from the user's own mail and files.
  They stay in the deck's data folder, are visible only to that user and are
  deleted with the factory. An admin never sees a user's factories (the rule
  Course follows for goal conflicts).

## 14. UI

**The First Mate page.** It has a full-view nav entry and a kiosk tile that
opens it in `KioskWideDialog`. It uses the Course and Inbox look: `fd-card`,
pills, soft headings, little text.

- **Header:** a Running / Paused switch and the week's budget bar.
- **Needs you:** proposals, asks, red factories.
- **Factory cards:** name, runner chip, trust pill, health ring,
  satisfaction, last and next run, tokens this week, and a ⋯ menu (Run now,
  Pause, Edit charter, Retire).
- **The drawer** has five tabs:
  - Charter (versions);
  - Runs (each links to its Bat page);
  - Scorecard (a chart over weeks);
  - Checks (open cases; sealed cases with a count, which the owner may open);
  - Signals.
- **New factory:** a template, a Course goal, or "describe it".

**On the Line.** First Mate's items carry source "First Mate" and one thread
per factory (`thread_id` `fm:<factory>`), so a factory's week reads as one
conversation.

## 15. Phases

Each phase is its own PR. The deck owner deploys.

**P0: Deploy and prove the foundation (no new code).** Bat's fixes (#384), the
Line, Captain Swarm and the Course redesign are merged but not deployed. Bat's
outward actions (mail, spend, accounts) haven't been tested live. Deploy them
to the owner's deck, then run, for real:
- a research Bat run;
- a Bat run that drafts and sends one email under an approved plan;
- a Captain Swarm rehearsal with an intent and a carry-out.

Fix what breaks. **Exit:** all three finish honestly, and their Line items
arrive.

**P1: Factories by hand.**
- `first_mate.db`, factories and versions, the charter approval card;
- `first_mate_runs.py` with Bat first, then the shared child readers for
  Basna, Vatra, Code and Crew;
- Run now;
- a minimal page: cards, drawer, runs;
- scorecard basics: verdict, tokens, time.

No triggers, no autonomy. **Exit:** a factory made from a template, approved
and run by hand 5 times, with history and spend shown.

**P2: The loop.**
- the supervisor tick;
- schedule triggers and the event router (status `claimed`, batch windows);
- slots, budget shares, `paused_budget`;
- failure-streak pause and the kill switch;
- Line posts.

**Exit:** two factories run unattended for a week inside their budget.

**P3: Outward actions.**
- rehearsal tags for First Mate and `fm_intents`;
- `propose` with intents on the Line;
- carry-out runs with the charter's grants;
- proof read-back;
- trust levels, moved by hand only until P4's calibration;
- spend, if Bat's spend path has passed its live test.

**Exit:** an inbox-drafts factory carries out approved replies, and the proof
confirms them.

**P4: Satisfaction.**
- signals from the Line, Gmail, the Taskboard and undo;
- sealed cases and the regression check;
- the scorecard and health;
- calibration with `label_eval`;
- the weekly report.

**Exit:** kappa ≥ 0.6 against the user's labels on 30+ results; a factory
moves from `propose` to `commit` on its own record.

**P5: First Mate talks and proposes.**
- the voice agent and the `first_mate` tool;
- charters drafted from words or a Course goal;
- weekly proposals;
- the Arbiter's `factory` proposal kind.

**Exit:** the user creates a factory by describing it, and accepts a
proposal from the Line.

**P6: Factories get better.** Improvement runs through Captain Swarm on sealed
cases, and a kept gain as a new version. **Exit:** one measured kept gain on a
real factory.

**Later, not in this plan:**
- asks between factories (Vatra's blackboard, one level up);
- fake services for mail and calendar factories at volume;
- deck-wide factories (instance scope);
- moving First Mate to a Superagent account;
- computer use, which is parked in Captain Swarm too.

## 16. Containment checklist (keep in sync)

- [ ] Factory runs are owned by the user and pass the user's own switches and
      gates.
- [ ] Grants come only from an approved charter version; they are pinned on
      the carry-out run and never copied to children.
- [ ] Every outward action before trust `commit` is an intent, not an act.
- [ ] A doubtful proof drops trust and asks the user.
- [ ] Sealed cases are never in a worker's reach or in an improver's brief.
- [ ] A reviewer with no answer means failed.
- [ ] LLM caps are tokens; real money goes through Bat's ledger with a cap.
- [ ] The `first_mate` tool is refused in workers and rehearsals.
- [ ] Event content is quoted data with its source.
- [ ] The kill switch stops running runs and their children.
- [ ] The supervisor's scan holds a lock.
- [ ] Admins never see a user's factories.

## 17. Decisions made in this plan (the owner can override any)

1. First Mate is a deck-side supervisor plus a voice agent created for it
   (or an existing agent the user picks).
2. Bat is the default runner; the others come through one handle.
3. One run per factory at a time; First Mate leaves one Bat slot free for the
   user.
4. Trust starts at `propose`; moving up takes 10 carried-out runs at ≥ 0.9
   satisfaction with no proof doubt; one doubt or undo drops a level and
   removes a pin; a version that adds a grant sets it back to `propose`.
5. Satisfaction is good ÷ (good + bad) over the last 20 known signals;
   silence for 7 days isn't counted.
6. Trust moves on its own only after kappa ≥ 0.6 on 30+ of the user's labels.
7. Defaults: 25M tokens per week, 3M per run, 8 factories, 24 runs per
   factory per day, a 15-minute event batch window.
8. The deck switch is off by default, admins only through P3.
9. Improvement runs go through Captain Swarm; First Mate has no learning
   machinery of its own.
10. Events a factory claims get a new status `claimed` and never reach the
    Arbiter.

## 18. Open questions

1. **Desktop app.** Captain Swarm is hidden there. Should First Mate be too,
   or is a one-person desktop deck a good home for it?
2. **Members.** When First Mate opens beyond admins, do members get
   improvement runs (Captain Swarm is admin-only)?
3. **Shared agents.** Should a factory ever run on a shared agent, or only on
   the user's own?
4. **Sealed cases from mail.** How long do they live, and may the user mark
   threads that must never become cases?
5. **Spend in factories.** Is real-money spend wanted at all at first, or
   only after a factory has earned `commit`?
