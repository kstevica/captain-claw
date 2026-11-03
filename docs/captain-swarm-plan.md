# Captain Swarm — agents that get measurably better at real work

**Status:** Phases 1–2 merged (measurement; the loop on code, proven live except a keep) · Phase 3a built (an agent's triage rules) · Phases 3b–7 not started · **Date:** 2026-11-03 · **Prerequisite done:** Bat finishes for real (#384)

Captain Swarm is a new orchestration mode, a sibling of Bat, Basna, Vatra and Council. Agents reach
it through the `captain_swarm` tool, the way they reach Bat through `bat`.

Bat drives **one task** until its checks and reviewers pass. Captain Swarm keeps **improving a
number**:
- Each round it proposes several different attempts and runs each in isolation through a mode
  Captain already has.
- It measures every result itself and keeps what beats the current best by more than noise.
- It writes everything to a ledger the next round reads.
- It stops at the target, at the budget, or when progress stalls, and hands the best result to the
  owner.

It runs no agents of its own. Bat, Crew, Vatra, Basna and agents' practice lanes do the work.
Captain Swarm owns four things: **the objective, the measurement, the experiment ledger and the
budget.**

**The goal is real work, not only code.** That means research, mail, documents, bookings, purchases
and forms, and, once Captain has it, **computer use**: operating websites and desktop apps the way a
person does. Code is the first proof only because its number is fast, cheap and hard to fake, which
makes it the quickest place to debug the loop itself. Every part of this design is built for the
real world from the start: the objective, the lanes, the measurement, the delivery and the safety
rules.

Why now: on 2026-11-02 the owner asked how far Captain is from Alexandr Wang's claim that a swarm of
agents, working toward a defined objective and improving through measurable feedback, can beat a
hundred engineers. Captain had the workers but not the loop, and Bat's loop had to be fixed first
(#384).

---

## 1. Three kinds of objective

| Kind | What varies | Example | Where attempts run | Measured by | Delivered as |
|---|---|---|---|---|---|
| **A. A better way of working** (recurring work) | code, a playbook, an agent's setup, a recorded browser workflow | "triage my inbox the way I would"; "Mrav passes more tasks"; "fill supplier invoices into the books" | rehearsal on practice cases from history | agreement with what the owner actually did; checks; calibrated rubric | a PR, or a proposal the owner approves (playbook, lessons, workflow) |
| **B. The best outcome for one task** (one-off) | the attempt itself: strategy, sources, route | "the three best offers for X under Y"; "a slot for Z next week"; "this form, ready to submit" | parallel rehearsals, up to the commit point | Captain re-checks every claim itself; predicates; price, fit, time | the winner, carried out **once** through Bat's gate, or by the owner |
| **C. A number in the world** (live) | a way of working, run for real | reply rate of outreach; time to answer customers | real work, under Bat's gate, with a capped audience | numbers read through connectors, re-checked later | the better way, adopted by the owner |

Phases 1–6 build A and B. C comes last (Phase 7), because it is the only kind where the experiment
itself acts on the world.

"Beats 100 engineers" only holds where the number is fast, cheap and hard to fake. For real work
the honest version is narrower but still big:
- many attempts on one task, each re-checked by Captain;
- ways of working measured against what the owner actually did;
- no attempt allowed to touch the world until one of them has won.

For small knowledge firms that is the "Captain checks" story.

---

## 2. Names

The user-facing name is **Captain Swarm** and the tool is **`captain_swarm`**.

"swarm" is already taken inside the repo by BotPort:
- the engine in `botport/botport/swarm/`
- the FD routes `/swarm/*`
- the TS types `Swarm` and `SwarmProject`, and `startSwarm` in `flight-deck/src/services/api.ts`
- the "Swarm Workflows" tab in `WorkflowPage.tsx`

Every internal id of Captain Swarm carries a `captain_swarm`, `captain-swarm` or `CaptainSwarm`
prefix and never the bare word. All of these are verified unused today:

| Thing | Name |
|---|---|
| Python modules | `captain_claw/flight_deck/captain_swarm_store.py`, `captain_swarm_loop.py`, `captain_swarm_score.py`, `captain_swarm_routes.py`; the lane guard `captain_claw/rehearsal_turn.py` |
| Store | `captain_swarm.db` in `FD_DATA_DIR`; sealed folders in `FD_DATA_DIR/captain_swarm/<run>/sealed/` |
| Routes | `/fd/captain-swarm/*` (page), `/fd/captain-swarm/agent/*` (tool) |
| Tool | `captain_claw/tools/captain_swarm.py`, class `CaptainSwarmTool`, name `captain_swarm` |
| Frontend | `CaptainSwarmPage.tsx`, `captainSwarmStore.ts`, types `CaptainSwarmRun` and `CaptainSwarmExperiment` |
| Env | `FD_CAPTAIN_SWARM_*`; lanes carry `CLAW_CAPTAIN_SWARM_RUN` |
| Branches | lanes `crew/cs-<run8>-<n>`; best so far `captain-swarm/<run8>/champion` |

The BotPort tab label changes from "Swarm Workflows" to "BotPort Workflows", so the deck never shows
two things called swarm. Only the label changes; BotPort's ids stay.

---

## 3. The loop

```
objective + scorer + practice cases (with held-out) + world limits + budget
                                                    (pinned at start, never changed by an attempt)
  → measure the current way / a first attempt several times      → baseline + noise
  → round:
      propose K different attempts                  (one model call, reads the ledger)
      run each in its own lane, in REHEARSAL         (nothing in the world changes)
      screen each one                               (protected paths, gaming scan, register check)
      measure in stages                             (small slice → full set → held-out for the winner)
      keep the best if it beats the champion by more than noise and breaks no guardrail
      write every experiment to the ledger          (hypothesis, what changed, scores, cost, verdict, lesson)
  → stop: target confirmed | budget | stall | world changed | owner
  → deliver: a PR | a proposal to approve | the winner carried out once through Bat's gate
```

### 3.1 The objective

The objective is pinned in the run at start. Attempts never see the parts marked sealed. Two
examples follow.

**Kind A, real work (the first real-life target, §15 Phase 3):**

```json
{
  "kind": "way_of_working",
  "title": "Triage my inbox the way I would",
  "subject": {"type": "playbook", "family": "mail-triage", "agent": "assistant"},
  "practice": {"source": "gmail", "window_days": 90,
               "label": "owner_replied_within_days:3", "holdout": "last_21_days", "sealed": true},
  "metric": {"kind": "agreement", "read": "f1_needs_reply", "direction": "max"},
  "guardrails": [
    {"name": "intents per email", "read": "intents_per_case", "max": 0},
    {"name": "seconds per email", "read": "seconds_per_case", "max_increase": 0.3}
  ],
  "lanes": {"kind": "practice", "parallel": 1},
  "budget": {"usd": 50, "hours": 8, "max_experiments": 40},
  "stop": {"target": 0.85, "stall_rounds": 3},
  "deliver": "proposal"
}
```

The labels in this example cost the owner no time. They come from what the owner actually did:
whether they replied within three days, read from their own sent mail.

**Kind A, code (the first proof, §15 Phase 2):**

```json
{
  "kind": "way_of_working",
  "title": "Mrav on gemma4:e2b passes more tasks",
  "subject": {"type": "code", "repo_path": "/Users/…/captain-claw",
              "editable": ["captain_claw/mrav/**"],
              "protected": ["scripts/mrav_eval.py", "evals/mrav/**", "tests/**"]},
  "metric": {"kind": "command",
             "run": "python scripts/mrav_eval.py --model gemma4:e2b --tasks {dev} --repeat 3 --isolate --json",
             "read": "pass_rate", "direction": "max"},
  "holdout": {"run": "… --tasks {holdout} …", "sealed": true},
  "guardrails": [{"name": "tokens per task", "read": "tokens_per_task", "max_increase": 0.15},
                 {"name": "Mrav tests", "run": "pytest tests/test_mrav*.py -q", "must_pass": true}],
  "lanes": {"kind": "crew", "harness": "claude", "parallel": 1},
  "budget": {"usd": 50, "hours": 8, "max_experiments": 40},
  "deliver": "pr"
}
```

**Kind B** objectives need fewer fields:
- the task;
- the predicates a good result must meet (constraints Captain can re-check itself);
- the number to rank by (price, time, fit);
- the world limits (§4);
- a freshness window, because offers and slots change.

The page offers templates for all three kinds.

### 3.2 What varies, and how it's delivered

| Subject | One attempt is | Delivered as |
|---|---|---|
| `code` | a branch cut from the champion (Crew worktree) | a PR; Captain never merges |
| `playbook` | a version of a playbook for one task family (Captain's playbooks are human-reviewed do/don't patterns, `agent_playbook_mixin.py`) | a playbook update the owner approves |
| `agent_setup` | an agent's instructions, lessons or tool choices | a proposal on the agent's Onboarding page (lessons stay checked by the owner, as `agent_onboarding` does today) |
| `workflow` | a recorded browser routine (`browser_workflow.py`, "teach once, run many") or, later, a computer-use routine | a proposal; it runs for real only through Bat |
| `attempt` | one try at a one-off task, stopped at the commit point | carried out once (§4.3) |

---

## 4. Real life: the world is shared

Code attempts each get a worktree of their own. Real attempts share one world: one inbox, one bank
account, one calendar, the same websites. That changes the rules.

### 4.1 Rehearsal by default

**In Captain Swarm, no lane ever acts on the world.** Every lane runs as a **rehearsal turn**: a new
guard, `rehearsal_turn.py`, modelled on the onboarding exam's `exam_turn.py`.
- `exam_turn` already runs a turn that keeps nothing and can't send or change anything. It works
  from an allowlist and fails closed, MCP and plugin tools included.
- **Reads run.** Rehearsal adds what the exam leaves out: the web, the browser's reading actions, and
  reading mail, calendar and files the objective names.
- **Writes go only to the attempt's own folder** (`vfs:captain-swarm/<run8>/<exp>/`).
- **Every outward action becomes an intent:** sending or drafting mail, changing the mailbox or
  calendar, settling a spend, creating an account, publishing, messaging, and accepting terms. The
  tool returns "recorded as an intent: it runs only if this attempt wins and the owner approves",
  and the intent goes into the ledger with everything needed to carry it out.
- **Bat's spend ledger** is used authorize-only in rehearsal: an amount can be held for comparison,
  never settled.
- **The agent keeps nothing,** as in the exam: no insights, reflections, topics or memory writes from
  a rehearsal.

### 4.2 The commit point (browser and computer use)

A browser or computer-use attempt may navigate, read, search, compare and fill in fields in its own
browser profile or desktop. Before an action that **submits, buys, pays, sends, deletes, confirms,
signs, publishes, books or accepts terms**, it stops.
- It records a screenshot, the accessibility snapshot, what it filled in and the action it would
  take, then ends as "ready to commit".
- Detection looks at the control's label and role, the page text and the URL (checkout, payment,
  confirm).
- **When unsure, it's a commit point.** Detection is a heuristic, so lanes use test or sandbox
  accounts wherever a service offers them.

### 4.3 Carry out the winner once

When the run ends, the winning attempt's intents and commit point go to the owner. Either the owner
does the last step themselves, or **one Bat run** carries it out:
- that run first re-checks the world: the same price, the slot still free, the page as recorded;
- it asks the owner if anything changed;
- it acts through Bat's existing gate: the owner's Go for mail, spending and accounts, and the
  spend-cap approval.

A swarm never multiplies an outward action. Four attempts produce one email, one booking, one
purchase.

### 4.4 One lane per site and per account

Parallel lanes on one website look like a bot farm. They trip rate limits, lockouts and CAPTCHAs,
and several sign-ins to one account set off security alerts.
- **By default, one lane at a time per site and per account** (`FD_CAPTAIN_SWARM_PER_SITE`, 1).
- Parallelism comes from **different** sources, sites and strategies, never from hammering one.
- Captain never evades detection and never solves CAPTCHAs (Bat's rule). A lane that meets a CAPTCHA
  asks the owner, or ends.

### 4.5 Computer use: a desktop per lane

A screen has one mouse.
- Today `desktop_action` (pyautogui) and `screen_capture` drive the deck's one real screen, so they
  can't run in parallel lanes.
- Each computer-use lane needs **its own desktop**: a container or VM with a virtual display, its own
  browser profile and only the accounts the objective allows.
- The owner's real desktop takes **at most one lane**, and only with the owner's consent. It is
  usually where the winner is carried out (§4.3).
- Capacity becomes how many desktops the deck can host. Each is likely 1–2 GB (an estimate, to be
  measured).
- Computer use is expensive in tokens (screenshots), so each experiment has its own spend cap.

### 4.6 Proof after the fact

The tool-call register (`tool_audit.db`) records every tool call an agent makes, with how it ended
and the ids of any send or draft. Its rows are HMAC-chained, so they can't be changed unseen. After
each experiment Captain reads that lane's rows. **Any outward outcome means the experiment is
disqualified and the owner is told.** A broken guard is a bug to fix, never a result.

### 4.7 Privacy

- **Practice cases stay on the deck.** They are built from the owner's mail, calendar and files and
  kept in the run's sealed folder, as frozen copies.
- **A lane sees only the case it is working on,** never the mailbox.
- **Models are the same as for that agent's normal work.** Lanes call the agent's own model under
  the same rules; no case goes to a model the agent wouldn't use anyway.

---

## 5. Measurement

A swarm that can raise its score by cheating will. These rules hold for every kind:

1. **Captain measures, and an attempt never does.** An attempt learns its own score on the open set
   and nothing else.
2. **The scorer is pinned.** It is copied into the sealed folder at start and every measurement runs
   that copy. For code, a diff that touches a protected path is disqualified before it's measured
   (`git diff --name-only base..branch`).
3. **Held-out cases are sealed.**
   - They are kept only in the sealed folder, never in a repo or a lane.
   - For practice sets from history, the held-out set is the most recent weeks, a split by time,
     because that is how the work will meet the future.
   - Proposers learn only whether held-out confirmed a gain.
4. **Noise is measured.**
   - The baseline is measured several times.
   - A win needs at least `max(min_gain, 2 × stdev)` on the open set, then confirmation on held-out.
   - The champion is re-measured every few rounds, so drift such as a model update isn't credited
     to an attempt.
5. **Captain checks claims itself.** A Kind B attempt that reports "€412, two stops, arrives 14:05"
   is scored on what Captain's own read-only fetch of that page says, never on what the attempt
   says.
6. **The gaming scan:**
   - Deterministic first: does what the attempt changed contain case contents, case ids or expected
     answers? Does it read eval paths or the run's environment?
   - Then a model reviewer, on a model the lanes don't run on (Bat's `_panel_creds`), reads anything
     that would become champion.

### 5.1 Scorers

| Scorer | For | Notes |
|---|---|---|
| `command` | code | runs on the deck through `bat_floor.screen`, so it needs a shell (`_owner_has_shell`: auth off, or an admin) |
| `predicate` | files, extracted fields, intents | `code_contract`'s shapes (file exists, file contains, no pattern) plus checks on intents and extracted values |
| `agreement` | ways of working | compared with what the owner actually did (replied or not, accepted or declined, filed where); precision, recall, F1, accuracy |
| `state` | Kind B, Kind C | read through connectors (Gmail, Calendar, Drive, a datastore) or Captain's own fetch; never the attempt's word |
| `rubric` | documents, replies, research | a model judge. It may be the **main** metric only once `label_eval` shows Cohen's kappa ≥ 0.6 against at least 30 owner labels; until then it is only a guardrail or a tie-break |
| `owner` | anything | the owner's thumbs on the page; slow, but the truest, and the source of labels for the rest |
| `screen` | computer use | the final screenshot and accessibility tree against the expected state; the weakest signal, never used alone |
| `delayed` | Kind C | a number read again after hours or days (reply rate after 48 h); the experiment stays "measuring" under a Bat-style lease |

### 5.2 Practice sets from history

Recurring real work already left a trail: mail, calendar, files, the tool-call register, the
Practice tab's owner-judged tasks (`agent_practice.py`) and the human labels on Basna and Vatra runs
(`label_eval.py`).
- A practice set is a **frozen** sample of past cases.
- Each case carries the owner's real decision as its label wherever one exists, so it costs the
  owner no time.
- Where none exists, the owner labels a few dozen cases on the page.
- Captain can propose a set ("the last 90 days of mail, labelled by whether you replied"). The owner
  approves its source and size at the start gate.

### 5.3 Guardrails for real work

Besides cost and time:
- **Asks per case:** human interventions are the reliability number for browser and computer-use
  work.
- **Intents per case:** a triage playbook that wants to reply to everything is worse, even when it is
  accurate.
- **Steps per case.**
- **Safety refusals hit.**

### 5.4 A known limit for code

When the thing being improved is code the scorer runs, that code can see the scorer.
`scripts/mrav_eval.py` runs the agent under test in the same process as its checks. Phase 1 changes
that:
- With `--isolate`, the agent runs in a child process that gets only the prompt and its workdir.
- On macOS that child has no OS-level confinement. `CLAW_CONFINE` is the tools' own rule, and the
  code under test is the very code that would have to obey it.
- So the gaming scan, the model review and the owner's read of the champion are the protection
  there. A Linux deck can add Landlock (`docs/agent-os-sandbox-plan.md`).

---

## 6. Lanes

An experiment is one hypothesis, one lane, one isolated workspace and one result.

| Lane | For | From |
|---|---|---|
| **Crew** | code | `crew_runner.dispatch(kind="ship", deliver="branch")`. Crew always cuts a worktree from HEAD or `origin/HEAD` (`_make_worktree`), so it gets a `base_ref` field for cutting from the champion. Git gives the evidence (`_evidence`) |
| **Practice** | playbooks and agent setups, tried on practice cases | the agent answers each case on its exam lane, extended into a rehearsal turn (§4.1), with the variant playbook or setup applied for that turn only |
| **Bat, in rehearsal** | real tasks: research, documents, mail work, browser | a Bat run with `rehearsal: true`. Bat brings its checks, fix rounds, asks, harm floor and budget. Its workers run rehearsal turns, so mail becomes intents and spending is authorize-only |
| **Computer use** | operating websites and desktop apps | once computer use lands: a Bat worker with desktop tools, in a desktop of its own (§4.5) |
| **Vatra / Basna** | research and writing ensembles | as Bat's children, or as direct lanes for documents scored by a rubric |

- **Children are tracked as Bat tracks them.** The readers, the stop and the spend counting move out
  of `bat_nested.py` into a parent-agnostic `child_runs.py`. Captain Swarm records each child's id at
  dispatch, so it needs none of Bat's port-and-slug matching.
- **No recursion.** The tool refuses inside any run: a Captain Swarm lane, a Bat worker, or a Basna
  or Vatra worker.

---

## 7. The proposer

The proposer is one model call per round.

- **Input:** the objective (without the sealed parts), the champion (a diff summary, playbook text,
  or the best attempt's record), the last ledger rows (kept and discarded, with reasons and lessons)
  and the stall state.
- **Output:** K hypotheses as JSON: `{hypothesis, strategy_tag, changes, why}`.
  - For Kind B, each is a strategy: "airline sites directly", "a meta-search", "nearby airports",
    "different dates".
- **Diversity:**
  - Two hypotheses in one round with the same strategy tag and overlapping changes: the second is
    dropped.
  - Anything already discarded is dropped unless its `why` says what is different now.
  - In Kind B, hypotheses that would put two lanes on one site are merged or queued (§4.4).
- **Model:** when an agent started the run, the proposer runs on that agent's model through its
  `/api/llm/complete`. The owner's tiers are only the fallback.

---

## 8. Champion

- **Code:** the branch `captain-swarm/<run8>/champion`. A kept experiment was cut from it, so keeping
  is a fast-forward.
- **Text subjects** (playbook, setup, workflow): a version in `captain_swarm.db`. The next round's
  attempts start from it.
- **Kind B:** the best attempt so far, with its commit-point record, re-checked against its
  freshness window.
- **One keeper per round.** The runner-up comes back next round, built on the new champion.

The report at the end shows:
- the baseline → champion on the open set and on held-out;
- every guardrail;
- the cost;
- the ledger;
- for Kind B, the attempts side by side.

---

## 9. Stopping

| Stop | When |
|---|---|
| Target | reached on the open set **and** confirmed on held-out (Kind B: met and re-checked) |
| Budget | the $ cap, hours cap or maximum number of experiments is reached |
| Stall | `stall_rounds` rounds keep nothing → one strategy round (§12) → two more rounds → still nothing: stop as "stalled" |
| World changed | Kind B: the best result's freshness window ran out and a re-check failed → one more round, or stop and say so |
| Owner | cancel stops every lane, desktop and measurement within about 10 s, as Bat does |
| Broken measurement | the baseline can't be measured, or a champion re-measure drifts by more than 3× noise → pause and ask the owner |

---

## 10. Budget and capacity

**Caps.** Every run has a $ cap, an hours cap and a maximum number of experiments. The defaults are
$50, 8 hours and 40 experiments, shown on the start screen. Unlike Bat, 0 is not allowed: a swarm
exists to spend in parallel. Computer-use experiments also get a cap per experiment.

**What counts:**
- **The proposer and the reviewers:** Bat's `_metered`.
- **Lanes:** `child_runs`.
  - Lanes on models through the deck proxy are counted.
  - A Crew worker on the owner's own CLI login has no price Captain can see. The run's log says so,
    and the maximum number of experiments is the backstop.
  - **Lanes run through the deck proxy by default, so the cap is real.**
- **Measurement:** free on local Ollama. A scorer on a priced model is counted under a per-run key.

**Capacity today** is far from 100 agents:
- Crew allows 8 running tasks per deck (`_DECK_MAX_RUNNING`), and each user gets 2 parallel by
  default, up to 8 (`crew_access.py`).
- Bat allows 3 runs per owner.
- Basna and Vatra allow 2 runs per agent, with up to 10 agents each.
- A full agent import takes about 200 MB.
- Computer-use lanes will be bounded by desktops (§4.5).

So the first version runs 1 lane, then up to 4 (`FD_CAPTAIN_SWARM_MAX_LANES`, counted against the
owner's grants). For real work, the per-site limit (§4.4) usually binds before any of these.

**Measurement is often the bottleneck.**
- A local-model scorer uses one GPU, so measurements queue (`FD_CAPTAIN_SWARM_MEASURE_PARALLEL`,
  default 1).
- Staged measurement keeps the queue short:
  1. A small slice, for every experiment.
  2. The full open set, for the better half.
  3. Held-out, only for the round's winner.
- A rough guess, to be measured in Phase 1: one full Mrav dev measurement (~20 tasks × 3 repeats on a
  local 4B model) takes half an hour to an hour.

---

## 11. Safety and the start gate

Captain Swarm reuses Bat's floor and adds nothing that loosens it.

- **Rehearsal (§4.1)** is the main rule. Lanes can't act on the world. The winner acts once, through
  Bat (§4.3).
- **Lanes** run under their own mode's rules as well: Crew's worktree contract, Bat's harm floor.
- **Command scorers** run through `bat_floor.screen`.
- **Start gate.** A run an agent starts always waits for the owner's Go. The owner sees:
  - the objective and the scorer;
  - the practice set's sources and size;
  - the held-out handling;
  - the lanes, sites and accounts it may use;
  - the budget;
  - which kinds of intents the winner could carry out at the end.

  A run the owner starts from the page starts on their confirm.
- **Kind C (live)** additionally needs the owner's consent per run, an audience cap, and every
  outward action through Bat's gate.
- **Asks:** `human_ask`, with the Captain Swarm store providing `create_ask` and the other ask
  methods Bat's store has.
- **Leases and restarts:** the same pattern as `BatSupervisor`. A measurement interrupted by a
  restart is re-run, never trusted.

---

## 12. Learning

- **Within a run:** the ledger feeds the proposer. Every experiment gets a one-line lesson.
  - The deterministic part comes from measurement, for example "+0.07 F1, held-out confirmed, intents
    0, seconds +4%".
  - The proposer adds its own reading next round.
- **On a stall: a strategy round.**
  - A small server-side panel (three lenses, reusing Bat's panel and vote code) reads the ledger and
    proposes a different direction.
  - This is not Council. Council's rounds run in the browser (`councilStore.ts`, `_runRoundRobin`),
    and moving them to the server is a project of its own.
- **Across runs:**
  - Lessons are kept per task family, for example `mail-triage`, `travel-search` or `mrav-prompts`,
    and given to the next run's first proposer call.
  - Strategy tags get a reliability weight using the same Bayesian shrink as archetypes
    (`db._reliability_weight`: fails count double). For example, "a meta-search beats airline sites
    for this owner's routes" earns more lanes.
- **Where learning lands:** a champion for recurring work becomes a **playbook**, lessons or a
  workflow proposal. That is how a swarm's result reaches everyday work. Nothing is written into an
  agent's memory or insights without the owner's approval.

---

## 13. UI

Captain Swarm gets its own page, next to Bat:

- **Objective card:** the kind, the metric, the baseline ± noise, the champion, the target and the
  held-out status.
- **Score chart:** every measured experiment as a dot by round, with the champion as a step line.
- **Ledger table:** hypothesis, lane, strategy, open and held-out scores, guardrails, cost, and
  verdict with its reason.
- **Each experiment's evidence:** a diff, a playbook diff, or the commit-point screenshot with its
  intents.
- **Kind B compare:** the attempts side by side, re-checked, with "carry out this one" and "I'll do
  it myself".
- **Owner labels:** thumbs on cases and attempts, feeding the `owner` scorer and rubric calibration.
- **Live state:** lanes and desktops in flight, the measurement queue, a budget bar, and asks (the
  same cards as Bat).
- **Start dialog:** the three kinds as templates: inbox triage, a one-off search, and the Mrav eval.

The frontend bundle is rebuilt and committed with the change (the FD build rule).

---

## 14. The `captain_swarm` tool

- **Actions:** `start` (an objective, or a template plus overrides), `status`, `get`, `list`,
  `ledger`, `cancel`.
- **Registration:** always on, like `BatTool`. It gives a clear error when `FD_URL` isn't
  available.
- **Refusals:** it refuses inside any run (§6), and every run it starts waits for the owner's Go
  (§11).

---

## 15. Phasing

| Phase | What | Size | Done when |
|---|---|---|---|
| 0 | Bat finishes for real | — | **done, #384** |
| 1 | Measurement first | S–M | baselines and noise known for Mrav and for one real-life practice set |
| 2 | The loop, proved on code | M | a real run lifts Mrav's score and the gain holds up |
| 3 | Real work, in rehearsal | M–L | a real run lifts inbox triage on held-out weeks, and one one-off task is carried out once |
| 4 | Parallel lanes | M | 4 lanes reach the same results faster at no more $ per kept gain |
| 5 | Computer use | L | a computer-use attempt reaches its commit point in its own desktop, and the winner is carried out once |
| 6 | Learning | M | a second run on the same family needs fewer experiments; a champion lands as a playbook |
| 7 | Live in the world (Kind C) | L | one live objective with a capped audience moves its number |

**Phase 1: measurement first.**
- **Scorers:** one interface for every kind in §5.1. This phase builds `command`, `predicate`,
  `agreement` and `owner`.
- **The Mrav eval** gets:
  - **task files:** `--tasks FILE`, with setup files, predicate checks in `code_contract`'s shapes,
    and reply checks;
  - **repeats:** `--repeat N`;
  - **child-process runs:** `--isolate`;
  - **machine-readable output:** `--json`.
- **Mrav task sets:**
  - a dev set of about 20 tasks in `evals/mrav/dev.json`, including today's six;
  - a held-out set of about 10 tasks in the deck's data dir only.
- **The first real-life practice set:** inbox triage. About 90 days of the owner's mail, frozen into
  the sealed folder, labelled by whether the owner replied within three days. The last three weeks
  are held out.
  - Start it now: it costs no owner time, but needs the owner's OK on the source.
- **Exit:**
  - Mrav: baseline and noise for qwen3.5:4b and gemma4:e2b.
  - Triage: baseline and noise for the assistant agent's current setup, run on the exam lane.

**Phase 1 as built (2026-11-03).**

*Code.*
- **Scorers and the noise rule:** `captain_claw/flight_deck/captain_swarm_score.py`.
- **The Mrav harness:** `scripts/mrav_eval.py`, with `--tasks`, `--repeat`, `--isolate` and `--json`.
  The `--isolate` child refuses to report if it imported `captain_claw` from anywhere but
  `--code-root`.
- **Mrav task sets:**
  - `evals/mrav/dev.json` (20 tasks);
  - a held-out set of 10 in `fd-data/captain_swarm/sealed/mrav/holdout.json` (0600), outside the
    repo.
- **Inbox triage:**
  - **The practice set:** `captain_swarm_practice.py`. It reads Gmail read-only through the owner's
    own token. A case's label is "the owner wrote in its thread within N days". Mail too new to know
    is left out, and the last weeks are held out.
  - **The measurement:** `captain_swarm_measure.py`, run on the agent's exam lane. It keeps case ids
    and answers, never mail text.
- **Routes:** `captain_swarm_routes.py`, under `/fd/captain-swarm/*`.
- **The page:** "Captain Swarm" in the sidebar, next to Bat.
- **Tests:** `tests/test_captain_swarm_phase1.py`.

*Measured.* Only `ornith:9b` is installed locally. qwen3.5:4b and gemma4:e2b aren't installed, so
their baselines are still owed.

| Set | Model | Pass rate | ± (across 3 runs) | Tokens / task | Seconds / task | Failures |
|---|---|---|---|---|---|---|
| Mrav dev (20 × 3) | ornith:9b | 0.967 | 0.029 | 6,499 | 28.5 | `append-line` 2/3 |
| Mrav held-out (10 × 3) | ornith:9b | 0.933 | 0.058 | 6,808 | 23.3 | `h-edit-in-place` 1/3, `h-honesty` 1 timeout |

- **Edit in place is the one weak skill.** The write tool puts a relative path under `saved/`, so
  "add a line to todo.txt" writes a new copy instead of changing the file. Only `shell` edits in
  place.
- **Little headroom.** A 9B model leaves almost none. The plan's choice of a smaller model for Phase
  2 stands.

*Triage, wiring only.* An `ornith:9b` agent was measured on a test deck. The set was 20 synthetic
emails with no real mail, and every email was asked twice on the agent's exam lane.

| F1 | ± | Precision | Recall | Unanswered | Seconds / case | Tokens / case |
|---|---|---|---|---|---|---|
| 0.895 | 0.054 | 1.00 | 0.81 | 0 | 23.1 | 15,672 |

- **What it missed:** "can we move tomorrow's call to 3pm?" both times, and one of two runs of
  "which brochure version should we print?".
- **What it cost:** about 15.7k tokens per email, because the exam lane carries the agent's whole
  context. A swarm round over 50 cases × 3 runs costs about 2.4M tokens on a priced model.
- **Still owed:** the owner's real baseline. The owner builds the set from their own Gmail on their
  deck, then measures their assistant agent.

**Phase 2: the loop, proved on code.**
- **What gets built:**
  - **Core:** the store, the supervisor with leases, the objective, and the sealed scorer.
  - **Screening:** the protected-path check and the deterministic gaming scan.
  - **The loop:** the proposer, and the Crew lane with `base_ref`.
  - **Results:** the champion branch, the ledger, the report and PR, the tool and a minimal page.
- **One lane**, so experiments run one after another.
- **Target:** gemma4:e2b. qwen3.5:4b already passes 5 of 6 tasks and has little headroom.
- **Exit:** the dev score rises beyond noise and held-out confirms it. No guardrail breaks, and the
  owner reading the diff finds no special-casing.
- **If it can't,** that is the finding: stop and report before Phase 3.

**Phase 2 as built (2026-11-03).**

*Code.*
- **The store:** `captain_swarm_store.py` (`captain_swarm.db`), holding runs, experiments and events,
  with leases.
- **The objective:** `captain_swarm_objective.py`, with the Mrav template, checks and placeholders.
  A command metric needs a shell owner.
- **Screening:** `captain_swarm_screen.py`.
  - A git snapshot of only what was committed.
  - The scope check: editable, protected, nothing changed.
  - The gaming scan: file names, expected answers and distinctive tokens from both case sets, plus
    eval paths. Plain words, short phrases, task ids and code shapes are left out, because they
    misfire on honest changes.
- **The loop:** `captain_swarm_loop.py`.
  - **Sealing:** the base commit's scorer and the held-out cases are copied once per run.
  - **Measuring:** a fresh snapshot per measurement, run by the sealed harness with `--code-root`.
    The environment is scrubbed, so code under test never sees the deck's secrets. `bat_floor`
    screens the command.
  - **The judge:** the open cases beyond noise, guardrails, then held-out.
  - **The proposer:** the owner's tiers, metered, with a diversity filter.
  - **The Crew lane:** a worktree cut from the champion, through a new `base_ref`
    (`crew_runner._make_worktree(start_at=…)`).
  - **The driver:** resumes after a restart, re-measures the champion on a schedule, and stops on a
    budget, time or experiment limit, a stall, drift or a cancel.
  - **The supervisor:** started with Flight Deck; `FD_CAPTAIN_SWARM_DISABLED` turns it off.
- **Routes:** `/fd/captain-swarm/{templates,runs}`, with Go and decline for an agent's run, and a PR
  from the champion through Crew's `open_pr`.
- **The tool:** `captain_swarm` for agents. It is admin-only (`deck_confine`), has exec risk, and
  every run it starts waits for the owner's Go, with a bell.
- **The page:** the Runs tab, with a start form, the run, a chart, the ledger and the log.

*What turned out different from the plan.*
- **Crew's proxy doesn't make the dollar cap real.** It counts a lane's tokens toward the owner's
  daily quota, not into the task's dollars. A lane's dollars are what its CLI reports: Claude Code
  reports `total_cost_usd`, Codex reports none. The run counts both dollars and tokens, and the
  experiment limit remains the backstop.
- **One lane can keep more than one gain per round.** Each experiment starts from the newest best,
  so the plan's "one keeper per round" applies only to parallel lanes (Phase 4).

*Live run (2026-11-03, test deck, no cost).*
- **Setup:**
  - A throwaway clone of the repo.
  - The Mrav objective cut down to 6 open and 3 held-out tasks, each run twice.
  - Measured on `ornith:9b`.
  - The coding worker was OpenCode on the deck's own `ornith:9b` tier, through Crew's proxy.
- **The start:** 0.917 ± 0.118 on the open cases, 1.000 ± 0 on the held-out ones. Sealing,
  snapshotting, scrubbed-environment measurement and the champion branch all worked.
- **Ideas:** the proposer made two sensible ones. One was spelling out how to append a line; the
  other was re-checking an edit before claiming success.
- **Both attempts committed nothing.** The 9B worker worked for 16 and 21 minutes, used about 1.19M
  tokens in all (counted toward the run), left one uncommitted change, and stopped mid-thought.
  The loop marked each "it committed nothing", closed the Crew task and removed its worktree and
  branch. It stopped as "stalled" after the one round this run allowed.
- **Shown live:** the whole path except keep. The baseline, sealing, ideas, the Crew lane from the
  champion, failure handling, token accounting, cleanup, the stop and the page all worked. Keeping a
  gain, held-out confirmation and moving the champion branch are covered by the tests (48), not yet
  by a live run.

*Two findings for the real Phase 2 run.*
1. **A local 9B model can't be the coding worker.** The real run needs Claude Code or Codex,
   which spends the owner's quota and needs their OK.
2. **The noise must be smaller than the headroom.** Here a gain had to be 0.236 or more, against a
   ceiling only 0.083 away. A run can only keep something when the measured model has room to
   improve (`gemma4:e2b`, not installed here) and the sets are large enough: at least 20 open tasks
   with 3 repeats.

**Phase 3: real work, in rehearsal.**
- **Rehearsal:** `rehearsal_turn.py`, intents, authorize-only spending, and the register check
  (§4.6).
- **Lanes:** the practice lane, and Bat with `rehearsal: true`.
- **Subjects:** `playbook` and `agent_setup`.
- **Scorers:** `state`, plus rubric calibration through `label_eval`.
- **Delivery:** proposals, and the carry-out-once Bat run (§4.3).
- **Two targets:**
  - **Kind A:** inbox triage. F1 on "needs your reply" rises on the held-out weeks, with zero
    intents.
  - **Kind B:** one one-off search, such as offers or slots. Several attempts end at their commit
    points, Captain re-checks each claim, and the owner's choice is carried out once through Bat's
    gate.

Phase 3 is built in two halves. **3a**, Kind A on the owner's own mail, needs no new safety
machinery, because the exam lane already can't send or change anything. **3b** is the rehearsal
guard, Bat in rehearsal, Kind B with commit points, and carrying out once. It is the groundwork for
computer use and gets its own review.

**Phase 3a as built (2026-11-03): an agent's triage rules.**
- **What varies:** a short list of rules in the owner's voice. Versions live in a new `cs_versions`
  table, and an attempt is an edit: `add` rules, `remove` rules by number.
  - The "lane" applies the edit at once (`RulesLane`), with nothing to wait for or clean up.
  - The model proposes the edits. It sees the rules now, the score with its precision and recall,
    and the open emails the best version still gets wrong, never held-out ones.
- **How a version is measured:** its rules go in front of each Phase 1 question, asked on the
  agent's exam lane, and Phase 1's agreement scorer gives F1 across runs. The agent itself isn't
  changed during a run, so its real conversations never see an experiment.
- **The screen:** a version must change something and stay within 12 rules of 300 characters. A rule
  carrying something only the held-out mail contains (a sender's address, a subject, a message id)
  disqualifies it. Senders who also appear in the open part are fair to learn.
- **The guardrail:** tokens per case, since rules make every question longer.
- **Delivery:** **Teach these rules** adds the best version's new rules to the agent as lessons
  (`agent_onboarding.teach_rules`, `taught_by: swarm`), only when the owner clicks.
  - Like rules taught by hand, they reach the agent with the messages they bear on.
  - In measurement they were always in front of the question, so the agent's real use is close to
    what was measured, but not identical.
- **What it needs:** no shell (nothing runs on the deck but the agent's own exam). Any owner with a
  ready practice set and their own agent can start one, and the `captain_swarm` tool can propose one
  (`template: "triage"`).
- **The loop itself is shared with code:** `parts_for(kind)` picks the lane, the proposer, the
  measurement, the seal, the screen, the champion marker and the starting point.

*Live run (2026-11-03): the first kept gain.*
- **Setup:**
  - A throwaway deck and a made-up inbox, with no real mail: 23 open and 12 held-out emails of five
    kinds. Direct asks and quiet asks need a reply; a question to someone else with the owner only in
    Cc, marketing that says "reply YES", and FYI mail don't.
  - The agent measured was "Inbox Helper" on **Claude Haiku 5.5**. The proposer was **Claude Sonnet
    5.5**.
  - A first try on the local `ornith:9b` was too slow (about 25 seconds a question) and was stopped.
- **The start:** F1 0.900 ± 0 on the open emails and 0.899 ± 0.014 on the held-out ones.
- **Round 1, idea 1, kept:** "If I'm only in Cc and the request is addressed by name to someone else,
  I don't need to reply." It scored **1.000 ± 0** on the open emails and **1.000 on the held-out
  ones**, so it became the best version.
- **Three more ideas, all discarded.** Two scored 1.000, which is no gain, and one scored 0.974 ±
  0.037. The run stopped at its 4-attempt limit, **8 minutes** after it started.
- **Taught live:** **Teach** added the rule to the agent as a lesson (`taught_by: swarm`).
- **Cost:** 6.18M tokens in all, about $0.62 at most at Haiku's price, plus about $0.01 for Sonnet.
- **One gap found and fixed:** measurement answers were counted as tokens only, so the $ cap couldn't
  see them.
  - The exam lane now passes each answer's usage on (`agent_exam.ask` returns `usage`).
  - Every answer is priced on the agent's own model as it comes, cache reads at their own rate, and
    counted against the run at once.
  - A measurement stops itself if the run's budget runs out partway.
- **What it proves:** the whole loop on real work, keep included. A held-out-confirmed gain landed in
  the agent through the owner's click. The made-up inbox is easy, and the owner's real mail is the
  real test.

**Phase 3b (next): the rehearsal guard.** `rehearsal_turn.py`, intents, authorize-only spending,
the register check, Bat in rehearsal, Kind B with commit points, and carrying out once. As planned
above.

**Phase 4: parallel lanes.**
- **Lanes and measurement:** K lanes, the per-site and per-account limit, staged measurement, and
  the measurement queue.
- **Shared plumbing:** `child_runs.py` extracted from `bat_nested.py`, with the budget counted
  across all lanes.
- **Review:** the model reviewer on every champion candidate.

**Phase 5: computer use**, as soon as Captain's computer use lands. With one lane it doesn't need
Phase 4, so it can come before Phase 4.
- **Where it runs:** desktops per lane (§4.5).
- **Rehearsal:** commit-point detection (§4.2).
- **Scoring:** `screen` scorers paired with a second signal, and asks per case as a guardrail.
- **Subjects:** `workflow`.

**Phase 6: learning.**
- **From the ledger:** lessons, and the strategy round on a stall.
- **Across runs:** lessons carried to the next run, and reliability weights per strategy.
- **Delivery:** champions delivered as playbooks, lessons and workflows.

**Phase 7: live in the world (Kind C).**
- **Measurement:** `delayed` scorers through connectors.
- **Safety:** audience caps, the owner's consent per run, and every outward action through Bat's
  gate.
- **First target:** something with a clear number and a forgiving audience, chosen with the owner.

---

## 16. Containment checklist (keep in sync)

- [ ] A lane never acts on the world. Outward actions become intents; spending is authorize-only;
      browser and computer-use attempts stop at the commit point.
- [ ] A lane's rows in the tool-call register are checked after every experiment. Any outward
      outcome disqualifies it, and the owner is told.
- [ ] The winner is carried out once, after a re-check, through Bat's gate or by the owner.
- [ ] One lane per site and per account by default. No detection evasion, no CAPTCHA solving.
- [ ] Each computer-use lane has its own desktop. The owner's desktop takes at most one lane, with
      consent.
- [ ] Attempts never run the scorer, never see held-out cases and never see held-out scores.
- [ ] Every measurement runs the sealed scorer. A code diff that touches a protected path is
      disqualified before it is measured.
- [ ] Claims are scored on Captain's own re-check, never on the attempt's word.
- [ ] A rubric is the main metric only after calibration (kappa ≥ 0.6, ≥ 30 owner labels).
- [ ] Practice cases stay on the deck. A lane sees only its own case.
- [ ] Champions arrive as a PR or a proposal. Captain never merges, installs or carries out a
      champion without the owner.
- [ ] Every run has $, hours and experiment caps, and computer-use experiments have their own cap.
      Spend Captain can't price is logged.
- [ ] Cancel stops every lane, desktop and measurement within about 10 s.
- [ ] No Captain Swarm starts from inside any run, and a run an agent starts waits for the owner's
      Go.
- [ ] Command scorers run only for owners with a shell, and through `bat_floor`.
- [ ] Kind C needs consent per run and an audience cap.
- [ ] A measurement interrupted by a restart is re-run, never trusted.

---

## 17. Decisions made in this plan (the owner can override any)

1. **Real work is the goal,** including computer use. Code comes first only because it debugs the
   loop fastest.
2. **Rehearsal by default.** No lane acts on the world, and the winner acts once, through Bat's gate
   or by the owner.
3. **One lane per site and per account** by default. A swarm must never look like a bot farm.
4. **Computer-use lanes get their own desktop.** The owner's real desktop takes at most one lane, with
   consent.
5. **The first real-life target is inbox triage,** measured against the owner's actual replies, so
   the labels cost the owner no time.
6. **The first code target is the Mrav eval on gemma4:e2b.**
7. **Budget defaults** are $50, 8 hours and 40 experiments. No run is uncapped.
8. **Command scorers need a shell.** Every other scorer is open to any owner who can run Bat.
9. **Lanes run through the deck proxy,** so the $ cap counts them.
10. **The strategy round is a light panel,** not Council.
11. **Names** as in §2, including renaming BotPort's tab label to "BotPort Workflows".
