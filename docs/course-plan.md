# Course: Implementation Plan

> Status: Phases 1–3 built. Phase 1: store, routes, Course page, admin card, kiosk dialog,
> prompt files, review sweep. Phase 2: the Arbiter, plans, the Taskboard, projects,
> scheduled tasks, Swarm runs, Forge agents and the agent's `course` tool. Phase 3:
> suggestions from agents (the tool, self-reflection) and from the deck (the heartbeat),
> each waiting for a person. Phase 4 not started.
> Decisions locked: the feature is called **Course** · each item is a **goal** · each goal
> has a **horizon** (strategic · long · medium · short) · three scopes (instance · user ·
> agent) · goals form a tree · proposals always wait for a person · goals guide priorities
> and never grant permission · **the admin chooses which horizons each scope uses, and can
> change it at any time** · **members of a shared agent see the deck-wide goals** (and
> nothing of the owner's).

## Thesis

The deck has plenty of machinery for *how* work gets done: the Arbiter, intentions,
plans, the Taskboard, Captain Swarm, schedules. Nothing says what the work is *for*.
The Arbiter's prompt (`flight_deck/arbiter.py`) asks for "the ONE most useful thing to
do now" and never defines useful. Consciousness, Taskboard triage and plans all guess
at importance the same way.

Course is that missing *why*. People write it down at three levels, and agents and the
deck suggest additions. It reaches every agent's prompt and every place where the deck
decides what comes next.

```
  INSTANCE (admin)        strategic · long
      │  serves
  USER SPACE (user)       strategic · long · medium · short
      │  serves                       ▲ proposals from the deck (consciousness pulse)
  AGENT (owner)           medium · short
                                      ▲ proposals from the agent (self-reflection, course tool)

  every proposal ──▶ Line "Needs you" ──▶ the scope's owner accepts / declines
  every active goal ──▶ course_context.md in each agent's prompt (owner turns only)
                    ──▶ Arbiter · Taskboard · plans · projects · Forge (goal_id links)
  evidence (links) ──▶ progress · "last moved" · off-course view
```

## Vocabulary

| Word | Meaning |
|---|---|
| **Course** | The feature and the page: all the goals that apply to a user and their agents. |
| **goal** | One item: an outcome to pursue, or (strategic) a direction to keep. |
| **horizon** | `strategic` (a direction, never "done", only revised) · `long` (about a quarter or more) · `medium` (weeks) · `short` (this week). |
| **scope** | `instance` (whole deck, admin) · `user` (one user space) · `agent` (one agent). |
| **serves** | A goal's parent. An agent goal serves a user goal, which serves an instance goal. |
| **on course / off course** | Progress words in the UI. "Off course": an active goal with no work toward it. |

"Goal" already appears in code with other meanings: the Arbiter's "candidate goals",
the goal a plan breaks into steps (`plans.py`), each step of a horizon plan
(`horizon_plan.py`), projects ("goal-oriented containers"), and Forge's "describe a
goal". Course does not rename them. To avoid confusion in new code:
- the new store, routes and tool are always named `course*`;
- the Arbiter's prompt wording "Candidate goals" becomes "Candidates" (one string, Phase 2);
- user-facing text says "your course" or "a goal on your course", never just "goals"
  next to Forge or plans.

## What is not a goal

Course stays useful only if it holds outcomes and nothing else. The page and the
propose prompts both say so:

- **Rules** ("never email customers without approval") go in the owner profile's
  instructions (`tenant_profile.py`) or the agent's instructions.
- **Tasks** ("send the Q3 report") go on the Taskboard, in intentions or in todo.
- **Roles** ("you are our support agent") go in the agent's personality and description.
  That is why agents have no strategic goals of their own.

## Scopes, horizons and who does what

| Scope | Default horizons | Who sets it | Who proposes | Who decides | Who sees it |
|---|---|---|---|---|---|
| instance | strategic, long | admin | nobody in v1 | admin | every user; every agent, on owner **and member** turns |
| user | all four | the user | the deck (pulse), the user's agents (course tool) | the user | the user, and their agents on owner turns |
| agent | medium, short | the agent's owner | the agent itself (reflection, course tool) | the owner | the owner, and that agent on owner turns |

**Horizons per scope are the admin's choice.** The table shows the defaults. The admin
changes them in the deck-wide Course card; the change applies at once. An empty list
switches a scope off: nobody can add goals there. When a horizon is taken away from a
scope, goals already there are kept and still reach the agents (an admin setting never
silently erases what people wrote). The page flags them, "this level no longer uses this
horizon", and their owner can move them to another horizon, finish them or drop them. No
new goal can be added at that horizon.

On an auth-off deck the `local` user is both the admin and the only user. The Course page
shows both the deck band and the user's own band, and the deck band is editable there.

**Caps** (active goals per scope, per horizon): strategic 3 · long 5 · medium 7 · short 3,
also set by the admin. Waiting proposals: at most 5 per scope. The caps keep the prompt
block small and force a choice.

**Where the admin's settings live:** a `course_settings` table in `course.db` (key
`horizons` = `{instance: [...], user: [...], agent: [...]}`, key `caps` =
`{strategic, long, medium, short}`). Keeping them in the Course store keeps it
self-contained and synchronous.

## Data: `course.db`

A sibling store like `autonomy.db` and `plans`: `CourseStore` in
`captain_claw/flight_deck/course.py`, sync `sqlite3` behind a lock (mirrors
`AutonomyStore`), at `FD_DATA_DIR/course.db`.

```
course_goals
  id                 TEXT PK          "goal_" + 16 hex
  scope              TEXT             instance | user | agent
  owner_id           TEXT             '' for instance; user id ('local' auth-off) otherwise
  agent_ref          TEXT             '' unless scope=agent; "{runtime}:{slug}:{instance}"
  horizon            TEXT             strategic | long | medium | short
  parent_id          TEXT             '' = top of its scope
  title              TEXT  ≤120
  detail             TEXT  ≤1000      what it means and why it matters
  success            TEXT  ≤500       how we'll know; required unless horizon=strategic
  metric             TEXT  json       optional {kind, spec, target, unit} (Phase 4)
  status             TEXT             proposed | active | paused | achieved | dropped |
                                      superseded | declined
  origin             TEXT             admin | user | agent | deck
  proposed_by        TEXT             user id, agent_ref, or 'deck'
  rationale          TEXT  ≤500       why it was proposed (proposals only)
  supersedes_id      TEXT             a proposed change points at the goal it replaces
  review_every_days  INTEGER          default by horizon: short 7 · medium 30 · long 90 · strategic 365
  review_at          TEXT             next review (ISO)
  due_at             TEXT             optional deadline
  progress           REAL             0..1 estimate, NULL = unknown (Phase 4)
  progress_note      TEXT
  sort               INTEGER
  created_at, updated_at, decided_at TEXT
  decided_by         TEXT

course_links                          the evidence: what moved a goal
  goal_id, kind, ref_id               PK; kind = autonomy_action | plan | project | task |
                                      schedule | swarm_run | session | line_item
  note TEXT, added_by TEXT, at TEXT

course_events                         audit trail, shown in the goal's History
  id, goal_id, at, actor, event, detail json
  event = created | proposed | accepted | declined | edited | status | reviewed |
          linked | progress
```

**The agent key.** Agent goals use `agent_ref` from `agent_sharing.process_ref` /
`docker_ref`, not the slug. A removed-and-recreated agent with the same slug gets a new
`instance` part, so it never inherits the old agent's goals.

**Tree rules** (enforced in the store, tested):
- A parent must be in the same scope or a wider one: agent → agent/user/instance,
  user → user/instance, instance → instance.
- The parent must be visible to the child's owner: the same user, or instance.
- The parent's horizon must be the same as the child's or longer. A short goal can
  serve a long one, not the other way round.
- No cycles. The parent must be `active`, or `proposed` in the same batch.
- Orphans are allowed. Non-strategic orphans are flagged "serves nothing above it" on the page.

**Lifecycle.**
- `proposed` → `active` (accept) or `declined`.
- `active` ↔ `paused`.
- `active` → `achieved` or `dropped`.
- An edit proposed by an agent or the deck is a new `proposed` row with `supersedes_id`.
  Accepting it marks the old goal `superseded` and moves the old goal's children under
  the new one.
- A person editing their own goal edits it in place (logged in `course_events`).

**Review.** When `review_at` passes, a goal is due. The deck posts one Line item, "Still on
course? <title>", with Keep / Edit / Pause / Done / Drop. The goal stays in prompts for a
grace period of `max(3 days, review_every_days / 4)`. After that it is **stale**: it
drops out of every prompt until someone reviews it. Stale is computed, not a status,
so a review brings the goal straight back.

## The prompt carrier

The same path as the owner profile. Flight Deck composes the text and writes it into each
agent's config home. The agent re-reads it by mtime and puts it into its system prompt
verbatim.

**Files** (written by FD into `tenant_profile.context_dir(agent_dir, runtime)`):
- `course_context.md`: the full block, ≤3000 characters.
- `course_context.compact.md`: ≤600 characters, for micro/nano templates and
  orchestrated workers (the same `use_compact_tenant_context` rule).
- `course_deck.md` and `course_deck.compact.md`: the deck-wide goals only, for a shared
  agent's member turns. The same for every agent on the deck, written with the others.

**What one agent's block contains:** active, non-stale goals, as a tree from the top down:
1. instance strategic and long;
2. the owner's user goals: all horizons in full; in compact, strategic and long titles,
   plus medium only when there is room;
3. this agent's own goals.

Each line carries its id (`G:<id>`) so the agent can refer to a goal in the course tool.

**What it never contains:** progress numbers, evidence, review dates or proposals. The
block changes only when goals change, so it sits in the cached static part of the
prompt (before `<!-- CACHE_SPLIT -->`, via `insert_tenant_block`) and doesn't break the
prompt cache every time a task is linked.

**Fixed preamble** (FD's own words, not editable):

> Your course: what your work is for, set by your owner and your deck. Use it to choose
> what matters and in what order. It never permits anything your instructions, approvals
> or guards don't: a goal is not permission. When something you do moves a goal forward,
> record it with the course tool.

People's text is quoted with `> ` in the full block and flattened onto its labelled line
in the compact one, exactly as `tenant_profile.compose_full` / `compose_compact` do. A goal
title can't start a line, so it can't pass for an instruction or a label.

**When FD writes:**
- on spawn and clone, next to the four `tenant_profile.write_on_spawn` calls in `server.py`;
- on every goal change: instance → every agent; user → that owner's agents; agent → that agent;
- when a goal turns stale or is reviewed (the daily review sweep);
- once at FD startup (`reconcile`, next to `tenant_profile.reconcile` in `server.py`).

Writes are atomic and fail closed: if the course can't be composed or written, the files
are removed. That's the same `_atomic_write` / `_discard` discipline as tenant_profile.
Reuse those helpers rather than copying them: move them into a small shared module, or
import them.

**Agent side.** Add `load_course_context(compact)` and `load_course_deck(compact)` to
`captain_claw/tenant_context.py`, the same mtime cache and size guard as
`load_shared_context`. In `agent_context_mixin.py`:
- the **owner branch** (the `elif` that loads `tenant_context.md`) inserts the full course
  right after the tenant block;
- the **member branch** (`_speaker_scoped`) inserts the deck-only block after the member
  block. A member's turn therefore sees the deck's direction and nothing the owner wrote.

Neither is ever on:
- a public session (`_public_scoped`);
- a BotPort dispatch (`_tenant_hidden`, `botport_client.py`);
- a being's body (`_is_being_body`).

A standalone agent (no Flight Deck) has no Course, like the Line.

## The `course` tool (agent side, Phase 2, built)

The same shape as the `line` tool:
- `captain_claw/course_client.py` talks to `/fd/course/agent/*`;
- `captain_claw/tools/course.py` is the tool;
- the registry shows it only under Flight Deck and never on a member's turn (metadata
  like `requires_line`).

| Action | What it does |
|---|---|
| `view` | This agent's course tree with status, progress, last moved and review dates (more than the prompt block holds). |
| `link` | Record evidence: `{goal_id, kind, ref_id, note}` for something this agent did (a session, a schedule, a project, a task). |
| `report` | A progress estimate `{goal_id, progress 0..1, note}`, marked as the agent's. `looks_achieved: true` posts "Looks achieved?" to the Line; it never sets `achieved`. |
| `propose` | A new goal `{scope: agent\|user, horizon, title, detail, success, parent_id, rationale}`, stored as `proposed`. |
| `propose_change` | A changed version of an existing goal `{goal_id, fields…, rationale}`, stored as `proposed` with `supersedes_id`. |

What an agent can never do: activate, decline, edit, pause or delete a goal; touch an
instance goal; see another agent's goals; or see anything on a member's turn.

**FD side.** `course_routes.py` has `/fd/course/agent/*`. It identifies the caller the way
`line_routes._agent` does: `_authorize_agent_call`, then this deck's own record of the
`X-Agent-Auth` token (`server._find_agent_by_auth`), then the owner. The `agent_ref` is
derived from FD's registry, never from anything the agent sent. It is owner only, so a
member never reaches it.

## Routes (people)

New router `captain_claw/flight_deck/course_routes.py`, mounted in `server.py`.

- `GET /fd/course` returns the caller's whole picture:
  - instance goals (read-only unless admin);
  - their user goals;
  - each of their agents' goals;
  - waiting proposals;
  - flags (orphan, stale, due for review; off course in Phase 4).
- `POST /fd/course/goals` creates a goal (scope `user`, or `agent` for an agent the caller owns).
- `PATCH /fd/course/goals/{id}` edits it (owner of the scope only).
- `POST /fd/course/goals/{id}/accept|decline|pause|resume|achieve|drop|review`.
- `POST /fd/course/goals/{id}/links`, `DELETE /fd/course/goals/{id}/links/{kind}/{ref_id}`.
- Instance goals are created and edited through the same goal routes with
  `scope = "instance"`; only an admin may (`role == "admin"`, 403 otherwise).
- `GET/PUT /fd/admin/course/settings`: the horizons per scope and the caps
  (`require_admin`). A save changes no agent's files: existing goals stay where they are.

Plain `get_current_user`, as on the profile routes: an admin's `X-FD-Act-As` does **not**
reach another user's Course. Someone else's goal is a 404, as on the Taskboard. There is
no admin view of user goals.

## Proposals and approval

A proposal is a goal row with `status = proposed`, plus one Line item:
- `bell_type = "course_proposal"`: kind `proposal`, needs the user;
- `ref_type = "course_goal"`, `ref_id = <goal id>`;
- `trust = "agent"` when an agent proposed it, `"deck"` for the pulse.

Accept or decline is answered on the Course page, or from the Line item's buttons. Either
resolves the item: add `"course_goal"` to `line.ANSWERED_REFS` and call
`db.resolve_line_items(uid, "course_goal", [id])`.

**Why not the autonomy ledger.** The chat discussion suggested routing proposals through
`autonomous_actions`. That is a poor fit: approving a ledger row dispatches it to an
agent (`fd_dispatch.dispatch_action`), and approve/reject trains action reliability
(`record_human_feedback`). A goal is neither. The Line is already the one place that
"waits for you", so proposals go there and their state stays on the goal row.

**Line types to add** to `line._TYPES`:

| Type | Kind | Needs the user |
|---|---|---|
| `course_proposal` | proposal | yes |
| `course_review` | ask | yes |
| `course_looks_achieved` | ask | yes |
| `course_conflict` | deck | yes (Phase 4) |
| `course_achieved` | result | no |

**Hygiene** (server-side in `CourseStore._proposal_rules`, so a chatty model can't get
around it; built in Phase 3):
- The admin's switch: with suggestions off, every suggestion is refused.
- Each agent may suggest at most `agent_per_week` (default 2) per week, across both
  scopes; the deck at most `deck_per_week` (default 2) per user per week. Every
  suggestion counts, whatever became of it.
- At most 5 suggestions wait per scope (one user's, one agent's).
- A title (compared lower case, letters and digits only) is refused when the same one is
  already waiting, already on the course, or was declined in that scope in the last 90
  days.
- A suggestion may change a goal a person set (`propose_change`): a new `proposed` row
  with `supersedes_id`, one waiting change per goal. The page shows what it changes, field
  by field. Accepting it supersedes the old goal and moves its children (that still fit),
  evidence and progress to the new one; a goal that was achieved or dropped meanwhile
  can't be replaced.
- A proposal becomes active only through `accept` (the status rules don't allow
  `proposed → active` any other way).
- Optional, off by default: `auto_accept_agent_short` lets an agent's own short-term goal
  go straight onto its course; the Line says so (`course_auto_added`, nothing to answer).
  Each user switches it on for themselves on the Course page.
- Agents may suggest goals for their owner too (`scope = "user"`), not only for
  themselves.

## Where Course plugs in

### Phase 1: the prompt (above)

### Arbiter (`flight_deck/arbiter.py`, Phase 2)
- Add a "Course" section to `user_prompt`: the compact active user and instance goals with
  their `G:` ids.
- Add to `_SYSTEM_PROMPT`: rank candidates that serve a goal higher, and set `"goal_id"` on
  the chosen action. A timely candidate that serves no goal (a reply someone is waiting
  for) is still allowed. Course changes ranking, not eligibility.
- Add a `goal_id` column to `autonomous_actions` (an `ALTER TABLE` in `AutonomyStore`'s
  init, the existing migration style) and to `add_action(..., goal_id="")`.
- When an action ends `done`, link it to its goal automatically (`course_links` kind
  `autonomy_action`). `line.action_done` is the natural hook.
- `goal_id` never changes `risk`, `should_auto_dispatch`, grants or caps.
- Phase 4, off by default (`goal_driven_candidates`): `_gather_candidates` may add an
  active short or medium goal with no evidence for a while ("No progress on G:… in 9 days")
  as a candidate. That lets the Arbiter propose work that comes from goals, still behind
  every existing dial.

### The review sweep (`flight_deck/course_carrier.py`, Phase 1)
- A small loop of its own, started at FD startup like `context_packs.reconcile_loop`, not
  inside the consciousness heartbeat (which a user can switch off, and reviews must still
  happen). It's cheap and needs no model: about every 30 minutes it finds goals that came
  due or went stale, posts `course_review` items, and rewrites the files of agents whose
  block changed.

### Consciousness pulse (`flight_deck/consciousness.py`, Phase 3, built)
- The heartbeat's prompt lists the user's live goals; its JSON gains an optional
  `course` object (`horizon`, `title`, `success`, `why`), "almost always null". One
  becomes a user-scope proposal with `origin = deck` (`_suggest_goal`), under every rule
  above. A heartbeat never fails over it.

### Agent self-reflection (`captain_claw/reflections.py`, Phase 3, built)
- No extra model call: under Flight Deck on the owner's turns (`course_client.usable`),
  the reflection prompt gains a short Course addendum asking for at most one final line,
  `GOAL_PROPOSAL: {…}` (a medium or short goal for the agent itself). The line is taken
  off before the reflection is saved, so it never ends up in the agent's own prompt; the
  suggestion goes to Flight Deck through `course_client.propose`.
- The transcript it reads is already filtered by `member_privacy.learnable`, so a member's
  conversation never turns into a goal proposal for the owner.

### Plans (`flight_deck/plans.py`, Phase 2, built)
- A `goal_id` column on `plans` (migrated in on open).
- The goal drawer's "Plan a route" calls `POST /fd/autonomy/plans` with `{goal_id}`: the
  plan's `goal` is the goal's title plus its success text. Only the caller's own or a
  deck-wide active goal (past its review too); anything else is a 404.
- A step that finishes links the goal, with the step's title and result as the note.
- Autonomous Work shows "Route to: <goal>" on such a plan.

### Taskboard (`flight_deck/taskboard_triage.py`, `taskboard_store.py`, Phase 2, built)
- The goal is read by the **extraction** call (the agent's own model, one task at a time),
  not by Jev. Extraction runs whether or not Jev is set up, and "what is this task about"
  is the extraction's job. The owner's live goals go in the trusted part of the prompt,
  above the fenced data; the answer gains `serves_goal`, kept only when it names one of the
  goals the call was shown.
- Stored in a new `goal_id` column on tasks (it reaches the UI with the other fields).
- A sort `course` ("Serves your course"): tasks tied to a goal first, then by `urgency`.
  A goal chip on the task row and in its detail.
- A task closed as done links its goal, whether a source closed it or the person did.

### Evidence belongs to someone (Phase 2, built)
- Every link records whose work it is (`course_links.owner_id`). On a deck-wide goal each
  user sees only their own links (in the drawer, in its history and in "last moved");
  someone else's work is never shown to them, the admin included.

### Projects, scheduled tasks, Captain Swarm runs (Phase 2, built)
- Linked through `course_links` only; their own stores are unchanged.
- Each one's page has a "Serves a goal" picker (`GET/PUT /fd/course/served`): one goal per
  item per user, set or cleared. The item must be the caller's own (a scheduled task, a
  Swarm run, a plan, a task, an action); projects belong to the whole deck.
- **Captain Swarm:** a run's measured objective (`captain_swarm_objective.py`) can become a
  goal's metric in Phase 4 (`metric.kind = swarm_score`).

### Forge (Phase 2, built for "Forge agents")
- The "Forge agents" review step (`ForgeReview.tsx`, `agent_forge.py`) has an optional,
  off-by-default "Add this team's goal to your Course": the goal, how you'll know it's done,
  its horizon, and "a goal for each agent too". Once the agents exist, the user goal is
  created, then one goal per agent made (its mandate as the title, what it produces as
  the success), serving the team's goal. They're active, because the person confirmed
  them right there. Horizons and limits the admin set are respected; what can't be added
  is listed on the ready screen, never fatal.
- **Agent Forge** (`ForgePage.tsx`, `POST /fd/forge`) is not wired yet: it spawns from the
  browser and has no server step after the agents exist. Its review step could create
  the goals through `POST /fd/course/goals` once each spawn returns its slug.

## Progress and off course (Phase 4)

- **Last moved:** the newest `course_links.at`. No model needed. Shown on every goal.
- **Metrics in v1:**
  - `manual`: a person enters the value;
  - `taskboard_done`: tasks with this `goal_id` done since the goal became active;
  - `link_count`.
- **Later metrics:** `datastore_query`, `swarm_score`.
- **Weekly estimate:** one pass per active goal that has new evidence, on the user's
  economy model through the pulse's author agent. It writes `progress` and
  `progress_note`. It never sets `achieved`: at most it posts `course_looks_achieved`.
- **Off course:** active user goals with no agent goal under them and no evidence within
  their review period, plus agent goals that serve nothing above them. Shown as a band at
  the top of the Course page.
- **Conflicts:** when a goal is created or accepted, one LLM pass compares it with active
  goals in its own and wider scopes. A likely conflict is flagged on both goals and posted
  as `course_conflict` to the owner of the narrower goal. The deck flags conflicts; it
  never settles them. For hard limits, instance outranks user, and user outranks agent.
  An admin is never told about a user's goals.

## The UI

**Course page** (`flight-deck/src/pages/CoursePage.tsx`, `src/stores/courseStore.ts`):
- add `'course'` to `ViewMode` in `src/types/index.ts`;
- a nav entry in `Sidebar.tsx`;
- a render branch in `App.tsx`;
- copy the `_authedFetch`/401 pattern from the autonomy store.

What the page shows:
- a **waiting proposals** strip at the top: Accept · Decline · Edit and accept;
- three bands, each a tree with horizon chips:
  - **Deck** (instance; read-only for non-admins);
  - **You** (user goals);
  - **Agents** (one group per agent);
- each goal shows status, last moved, review due and (Phase 4) a progress bar;
- the **goal drawer**: success, metric, links (evidence), History (`course_events`), and
  Plan a route (Phase 2).

**Other places:**
- **Admin:** a "Deck-wide course" card next to `DeckProfileDefaults` (in `AdminPage.tsx`
  and `SettingsConsole.tsx`): which horizons each scope uses, the caps, and a link to the
  deck band on the Course page where the deck's goals are edited.
- **Kiosk:** `KioskCourseDialog` in `SimpleLayout.tsx`, beside `KioskProfileDialog`, with
  the user's goals and their agents' goals. Proposals reach kiosk users through the bell
  like everything else.
- **Agent config** (`AgentConfigEditor.tsx`): the agent's own goals, read-only, with a link
  to the Course page.

Build with `npm run build`, which writes into `captain_claw/flight_deck/static/`.

## Settings (as built)

Course's settings live in its own store (`course_settings` in `course.db`), not in
`config.py`: the admin changes them from the UI at any time and they apply at once.

| Key | Who | Default | |
|---|---|---|---|
| `horizons` | admin | deck: strategic, long · user: all · agent: medium, short | Which horizons each level uses. |
| `caps` | admin | 3 / 5 / 7 / 3 | Active goals per horizon per level. |
| `proposals.recommendations` | admin | true | Agents and the deck may suggest goals. |
| `proposals.agent_per_week` | admin | 2 | Per agent, per week (0–20). |
| `proposals.deck_per_week` | admin | 2 | Per user, per week (0–20). |
| `prefs:<user>.auto_accept_agent_short` | each user | false | Their agents may add their own short-term goals without asking. |

Fixed in code for now: the prompt block sizes (3000 / 600), the review grace
(`max(3 days, a quarter of the review period)`), 5 waiting suggestions per scope and the
90-day decline memory. Phase 4's switches (goal-driven Arbiter candidates, weekly
progress estimates) will join the admin's settings.

## Privacy and safety

- **A goal is not permission.** The fixed preamble says so. No code path reads `goal_id`
  to decide risk, approval, auto-dispatch, grants or guards. A test pins that the Arbiter's
  dispatch decision is identical with and without a goal.
- **User goals are private to the user.** No admin view and no act-as, the same rule as the
  Taskboard. Instance goals are visible to every user.
- **Shared agents:** a member's turn gets the deck-wide goals and nothing else. The
  owner's user goals and the agent's own goals never reach it.
- **BotPort, public sessions, being bodies:** no course block.
- **Agent words are labelled.** Proposed text is shown to the owner as the agent's
  (`trust = "agent"`) and quoted in the prompt like profile text.
- **Deletion:**
  - a deleted user: add `course.forget_user` to `admin_routes.delete_user`, next to
    `taskboard.forget_user`;
  - a removed agent: its agent-scope goals are dropped and its files removed;
  - slug reuse is safe because goals are keyed by `agent_ref`.
- **Limits:** text caps, count caps and proposal rate limits are all enforced in the store,
  not only in the UI.

## Phasing

1. **Spine, no models.**
   - Built: `CourseStore`, people's routes, admin routes, the Course page, the admin card,
     the kiosk dialog, the carrier (compose, write on spawn and change, reconcile), the
     agent loader and prompt insertion, the review sweep with `course_review` Line items,
     `forget_user`, and agent removal.
   - What users get: people write their course, and every agent works with it.
2. **Wiring.** (built)
   - `goal_id` on Arbiter proposals (prompt, column, auto-link, "Serves:" in the bell),
     `plans.goal_id` and Plan a route, Taskboard `serves_goal` and the `course` sort, the
     "Serves a goal" picker for projects, scheduled tasks and Swarm runs, the goal drawer
     (evidence, history, progress, Plan a route), the Forge agents review step, and the
     `course` tool's `view`, `link` and `report`.
   - The Arbiter's "Candidate goals" string became "Candidates".
3. **Recommendations.** (built)
   - `course` tool `propose` / `propose_change`, agent self-reflection proposals, pulse
     proposals, the `course_proposal` / `course_auto_added` Line types and their
     resolution, weekly limits, the waiting cap, dedup, the 90-day decline memory,
     `auto_accept_agent_short`, the "Waiting for you" strip (Accept, Edit and accept,
     Decline, a change's differences), Accept / Decline on the Line, and the admin's
     suggestion settings.
4. **Progress and off course.**
   - Last moved, metrics (`manual`, `taskboard_done`, `link_count`), the weekly estimate,
     the off-course band, the conflict check, and goal-driven Arbiter candidates.

## Tests

New `tests/test_course_*.py`:
- **Store:** tree rules (scope, visibility, horizon order, cycles), caps, lifecycle,
  supersede re-parenting, stale computation, `forget_user`.
- **Compose:** size caps, quoting (a goal title can't pose as a label), staleness, the
  order of the tree, compact selection.
- **Carrier:** atomic write, fail closed on error, rewrite on change per scope, reconcile.
- **Agent loader:** mtime cache, insertion before `CACHE_SPLIT`, absent on member, public,
  BotPort and being turns.
- **Routes:** ownership (404 for others), admin only for instance, act-as refused, agent
  auth through `X-Agent-Auth` only, member refused.
- **Proposals:** weekly caps, waiting dedup, the 90-day decline block, Line item posted and
  resolved.
- **Arbiter:** `goal_id` parsed and stored; dispatch decision unchanged by `goal_id`.

## Open questions

Decided: members of a shared agent see the deck-wide goals; the admin chooses the
horizons per scope.

Decided in Phase 3: agents may suggest goals for their owner (user scope) as well as for
themselves.

1. Should the deck ever propose instance goals to the admin? v1: no. It would have to read
   across users' private goals.
2. Should run workers (Basna, Vatra, Council teammates) get the compact block? v1: no. The
   run's task already carries the `goal_id` of the action that started it.
3. Should waiting suggestions expire? v1: no; at most 5 wait per scope, so old ones have
   to be answered before new ones arrive.
