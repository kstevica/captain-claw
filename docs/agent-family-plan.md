# Agent family — plan

When an agent makes another agent, the new one is its **child**, not a
sibling. A parent, its children and its **siblings** (children of the same
parent) know about each other and can talk. You see who made whom in the
sidebar and on the Agents page, and you can move an existing agent under
another agent by dragging it there. First Mate is a parent too: its factories'
work hangs under it.

Status: decisions taken 2026-11-17; Phases 1–3 (data, talk, sidebar) built, see §11–§13.

## User decisions (2026-11-17)

| # | Question | Answer | Not chosen |
|---|----------|--------|------------|
| 1 | Run workers (Bat, Basna, Vatra, Council, flows) | **A live "Working · N workers" line** under whoever started the run; gone when the run ends | Keep them hidden entirely |
| 2 | Removing a parent | **Its children move up one level**; the dialog offers "Remove them too" | Always remove the children with it |
| 3 | Who may talk to whom | **As today, any of the owner's agents**; family only adds context | Only family |
| 4 | Groups | **A child follows its parent's group**; only top-level agents are placed in groups | A child may sit in another group |
| 5 | First Mate worker groups (uncommitted, branch `feat/first-mate-worker-groups`) | **Fold it into Phase 1**: record `parent` instead of `group` | Let it land first, replace it later |
| 6 | Moving existing agents | **Drag an agent onto another agent to make it a child** (§6), plus a ⋯ picker | — |

## 1. What exists today (origin/main, 2026-11-17)

- **No parent link.** The registry entry in `.processes.json`, `ProcessInfo`
  and the Docker labels don't say who made an agent. FD can name the calling
  agent (`_find_agent_by_auth` gives its slug), but the spawn path keeps only
  the owner (`_resolve_agent_identity_by_auth`). The only trace is the default
  description "Agent spawned by Old Man" in `tools/flight_deck.py`.
- **Nobody is told who's talking.** A spawned child isn't told who made it.
  `consult` and `delegate` targets get `{"type":"chat", ...}` with no sender.
  The "Other Available Agents" prompt block lists every running agent,
  short-lived workers included.
- **What already works:** colleagues hand-over (`colleagues.py`, CHAIN_MAX 4,
  DAILY_MAX 60, result back in the asking task's session); `consult`
  (synchronous); `delegate` (result pushed back later). Crew's `agent_slug` is
  the only stored "which agent asked". Forge teams live in
  `basna_sessions.route.forge.created`.
- **Workers look like your agents.** `bat-…` slugs aren't in the
  managed-agent pattern (`agent_sharing.py`, `utils/managedAgents.ts`), so
  Bat workers, First Mate's included, show with power switches.
- **Groups** are a per-user settings blob (`fd:agent-groups`, `groupStore.ts`),
  one level deep. "Other agents" is computed.

## 2. The model

Three kinds of agent:

1. **Your agents.** You made them. They are the top level.
2. **Standing children.** An agent made them (`spawn_agent`, Forge, First
   Mate's "Make one"), or you moved them under another agent (§6). They stay,
   have their own chat and power, and are nested under their parent.
3. **Run workers.** Bat, Basna, Vatra, Council and flow workers. They live
   for minutes and are torn down. They never get their own row: only a live
   line under whoever started the run.

Siblings aren't stored: they are the agents with the same `parent`.

Two separate facts:
- `made_by`: who created the agent. Set once, never changes.
- `parent`: where it sits now. Starts equal to `made_by`; you can change it
  by dragging (§6).

## 3. Data

Registry entry gains (process agents; Docker agents stay top-level for now, §11):

| Field | Meaning |
|-------|---------|
| `parent` | Slug of the current parent; `""` = one of yours |
| `made_by` | Slug of the agent that created it, `"first_mate"`, or `""` (you) |
| `made_for` | One line: why it exists. Editable |
| `role` | `standing` or `worker` |
| `made_at` | When it was created |
| `parent_set_by` | `spawn` or `person` (after a move) |
| `run` | Workers only: `bat:<id>`, `vatra:<id>` |
| `factory` | A First Mate factory worker's factory id |

`ProcessInfo` exposes `parent`, `made_by`, `made_for`, `role`, `run` and
`factory`. A child always has its parent's owner.

Set at:
- `/fd/spawn-process` and `/fd/spawn`: when the caller is an agent, keep its
  slug as `parent`/`made_by`. The `spawn_agent` tool sends `made_for`.
- `agent_forge._spawn_agent`: the asking agent, or `""` when the person
  started Forge.
- Bat, Basna and Vatra workers: the agent that started the run (pass its slug,
  not only `source_host`/`source_port`), First Mate for factory runs, `""`
  otherwise. `role=worker`.
- Old entries: the deck never recorded who made an agent, so there is no
  parent to backfill. They read as top-level; their `role` comes from the
  worker names (`agent_sharing.is_managed_agent`). Moving them is by hand (§6).

Limits, checked on spawn and on every move:
- Depth: at most 3 levels below a top-level agent.
- Children: at most 12 per parent (Forge's MAX_TEAM).
- Never a cycle (an agent can't sit under its own descendant).

## 4. Knowing and talking

- **"Your family" prompt block** instead of the full "Other Available Agents"
  list: your parent and what it made you for, your children (what each is for,
  awake/asleep), your siblings. About 12 lines at most; the rest of the owner's
  agents are one line ("and 7 more of your owner's agents — `colleagues`
  roster"). Sent through the identity frame (`agent_handshake`) on spawn and
  whenever the family changes (spawn, move, removal, rename).
- **Every message names its sender**, with the relation: "Glasses, who made
  you, asks…", "Your sibling Deal digest asks…", "Your child Follow-up writer
  reports…". Covers consult, delegate and colleagues hand-over.
- **Report to the parent.** A child's `report` lands in the session it was made
  in (or the parent's latest), and wakes the parent if it's asleep (the
  pre-wake used for cron).
- **Reach stays as today** (decision 3): family adds context, not walls.
- **Permission:** at spawn and hand-over a child never gets more than its
  parent (the colleagues rule). Moving an agent by hand (§6) doesn't change
  its permissions: the person did it on purpose.

## 5. How it shows

**Sidebar (`SimpleLayout.tsx` AgentsColumn, `KioskAgentGroups.tsx`):**
- Children sit right under their parent, on the thin rail line groups already
  use. A chevron on the parent folds them; a folded parent shows "+N" and passes
  up busy/unread dots, like a folded group.
- Rendered as sibling `<li>`s with `data-parent` and `aria-level`, never
  nested inside the parent's `<li>`, so `e2e/groups/check.py` selectors
  (`aside li`, `li span.font-medium`) still hold.
- Workers: one "Working · N workers" line under the starter, no power switch.

**Agents page (`AgentCards.tsx`):**
- A parent's card gets a "Made N agents" tray of small child cards, each with
  its `made_for` line and state.
- Hovering a card lights up its parent and siblings and dims the rest.
- Workers never count towards "N more agents are off".

**Groups** (decision 4): a child follows its parent's group; only top-level
agents are members of groups.

## 6. Moving an agent under another agent (drag to adopt)

You can take any of your standing agents and make it another agent's child,
or take a child out and make it one of yours again.

**Sidebar:**
- Drag a row **onto another agent's row** → that agent becomes its parent.
  The target row highlights with "Put Deal digest under Glasses".
- Dropping **between rows, on a section, or on the group header** keeps
  today's meaning (move to a group, "Drop here for a new group"). The row body
  is the adopt target; the top and bottom quarter of a row and the gaps fall
  through to the section, so both kinds of drop stay possible.
- Drag a child **out onto a section or "Other agents"** → it becomes
  top-level again and joins that group.
- The whole subtree moves with the agent.

**Agents page:** cards become draggable the same way: drop a card on another
card to put it under that agent; drop a child card from a tray onto the page
background or a section heading to make it top-level.

**Without a mouse** (phone, keyboard, screen reader): agent ⋯ →
"Put under another agent…" opens a picker of allowed parents, with "None —
one of yours" at the top. The phone has no drag, so this is its only way.

**After a drop:**
- A small optional prompt: "What is it for?" fills `made_for` (skip keeps
  the old one).
- A toast: "Deal digest is now under Glasses · Undo".
- Both agents (and the old parent) get a fresh "Your family" block.
- `parent_set_by = person`. `made_by` stays as it was.

**Refused, with the reason on the drop target (no drop highlight):**
- the target is the agent itself or one of its descendants (cycle);
- it would go past 3 levels, counting the subtree being moved;
- the target already has 12 children;
- the agent or the target is a run worker, a local agent ("On this
  computer"), or a shared agent someone else owns;
- the target is a First Mate factory line (only First Mate's agent can be a
  parent).

**Backend:** `PUT /fd/processes/{slug}/parent {parent: "<slug>" | "", made_for?}`.
Owner only (act-as follows the usual admin rule), under the spawn lock,
checks the limits above, writes `parent` + `parent_set_by`, re-sends the
identity frame to the affected agents, and logs a `family_moved` event (for
the agent's info panel history). Docker agents are left out for now (decided
in Phase 1): a running container's labels can't change, and agents spawn as
processes first; they stay top-level and can't be moved.

**e2e:** new cases in `flight-deck/e2e/groups/`: drop on a row adopts, drop
between rows still moves to a group, drop on own child is refused, ⋯ picker
on the phone, Undo restores.

## 7. First Mate

- First Mate's agent is the parent of its factories' work: factory Bat
  workers get `made_by = "first_mate"`, `factory = <id>`, `parent =` the
  picked First Mate agent (or a virtual "First Mate" row when none is picked).
- Its sidebar row and card show **factories**, not raw workers ("Inbox triage ·
  Working · 2 workers", "Weekly report · Next Mon 07:00"); a factory opens
  its drawer on the First Mate page.
- Workers report to First Mate's agent, so it can tell you in chat what its
  factories did (it already has the `first_mate` tool).
- You can drag your own agents under First Mate (helpers it can ask), but not
  onto a factory line.
- The `feat/first-mate-worker-groups` work (`label_agent`, `label_worker`,
  `ProcessInfo.group`, `groupStore.autoPlace`) merged as #621 while Phase 1
  was being built. Phase 1 keeps it: factory workers are still named for their
  factory and shown in the "First Mate" group, and now also carry
  `parent`/`made_by`/`factory` (bat_routes `_label_worker`). P3 replaces the
  group with the tree (decision 5): the workers show under First Mate, and
  `group`/`autoPlace` go.

## 8. Lifecycle

- **Removing a parent:** "Glasses has 2 agents under it. They move up to
  <grandparent / your agents>" with a "Remove them too" option. Never a silent
  cascade.
- **Power off a parent:** children keep running; the dialog offers "and the 2
  under it". A child's report wakes a sleeping parent.
- **Rename:** the family blocks of parent, children and siblings refresh.
- **Clone:** the clone gets the same `parent`, a fresh `made_at`, and
  `made_by` = the clone's maker (you).
- **Export/import (.captain bundles):** `made_for` travels; `parent` doesn't
  (the bundle lands top-level).

## 9. Phases (one PR each)

| Phase | What |
|-------|------|
| P1 Data | Registry fields, spawn paths record the parent, `bat-` treated as managed, `PUT /fd/processes/{slug}/parent` with the limits, `ProcessInfo` fields, First Mate labels → `parent`/`factory` |
| P2 Talk | "Your family" prompt block (a file the deck keeps), sender named in consult/delegate/hand-over, `report` to the parent, refresh on change, workers out of the peer lists |
| P3 Sidebar | Tree rows, fold, worker line, drag onto a row to adopt, drag out to make top-level, ⋯ "Put under another agent…" picker, Undo toast, e2e |
| P4 Agents page | Trays, hover family highlight, draggable cards, First Mate's factory tray, footer count without workers |
| P5 Lifecycle | Remove and power dialogs, clone/export rules, info-panel family history, Basna/Vatra/Council workers under whoever started their run |

## 10. Not in this plan

- Agents moving each other (only the person drags; an agent's only way to
  get a child is to make one).
- Family across owners (a shared agent's children stay its owner's).
- A separate org-chart view; the trays and the sidebar tree come first.

## 11. As built (P1, 2026-11-17)

- **`agent_family.py`**: the fields, `role_of` (old entries by worker names),
  `parent_of` (a parent that is gone or another owner's counts as none),
  `children` / `ancestors` / `depth` / `descendants` / `height`, the checks
  `check_child` (an agent spawning) and `check_move` (a move), the changes
  `move`, `on_removed` (children move up), `label` (a worker right after its
  spawn), and `for_spawn` (what a (re)spawn records, in order: FD code's
  `config._family` → a new agent an agent made → a run's worker by its env →
  a re-spawn keeps its own → the owner's own).
- **Spawn** (`server._spawn_process_locked`): the calling agent comes from its
  `X-Agent-Auth` (`_spawn_caller_slug`, same owner only); a new child past the
  limits is refused with 409 before anything is written. `AgentConfig.made_for`
  (body) and `_family` (FD code only).
- **Move**: `PUT /fd/processes/{slug}/parent {parent, made_for?}` → 200
  `{ok, slug, parent, made_for, message}`; 409 with the reason (loop, past 3
  levels, 12 under one parent, a worker); 404 for an agent that isn't the
  caller's. Under the spawn lock. No identity frame or `family_moved` event yet
  (P2 / P5).
- **Removal** (`_remove_process_agent`): what sat under the agent moves up.
- **Workers**: every runner's worker is `role=worker` by its env
  (`CLAW_BAT_WORKER`, `CLAW_VATRA_WORKER`, `CLAW_BASNA_WORKER`,
  `CLAW_COUNCIL_WORKER`, `CLAW_CODE_AGENT`), with `run` for Bat and Vatra.
  Bat's (`bat_routes._label_worker`, both worker modes): a factory's run →
  `parent` = the picked First Mate agent, `made_by=first_mate`, `factory`,
  `made_for` = "<factory> · [improving ·] <step>"; a run an agent started →
  that agent (found by its port, same owner only). Basna/Vatra/Council workers
  get no parent yet: who started their run isn't passed to the spawn — P3,
  with the worker line.
- **Managed**: `bat-bat-<hex4>-…` joins the worker names (backend and
  frontend), and the SPA also treats `role === "worker"` as one
  (`isManagedProcess`): Bat workers no longer show as your agents with power
  switches. Until P3, a First Mate factory's workers are the exception: they
  keep showing in the First Mate group (#621).
- **Agent tool**: `spawn_agent` takes `made_for`; a 409 from
  `/fd/spawn-process` is said back, never retried through Docker.
- **Not done in P1**: Forge-made agents get their mandate as `made_for` but stay
  top-level (Forge is started by the person); the First Mate agent itself is
  top-level; clone lands top-level (P5).
- Deploy: FD restart (server + routes), agent restarts (the `spawn_agent`
  tool), hard refresh (bundle).

## 12. As built (P2, 2026-11-17)

- **The block** (`family_carrier.py`): `family_context.md` in a process agent's
  config home, next to the Course files; the agent inserts it on its owner's
  turns (`tenant_context.load_family_context`, `agent_context_mixin`), in the
  cached part of the prompt. It says who the agent's parent is (and whether it
  made the agent or the owner put it there, with `made_for`), its siblings (12
  by name, then "and N more") and its children. No live state (awake/asleep):
  the block changes only when the family does. None for workers or for agents
  without family. Names and `made_for` lines are flattened, quoted and capped.
- **Kept in step**: rewritten for the owner's agents after a spawn, a move, a
  removal and a rename (an unchanged file is left alone, so the prompt cache
  holds), and once at FD startup. Not through the identity frame: a file reaches
  a sleeping agent too.
- **"Other Available Agents" stays** (decision 3), minus run workers: the
  browser's peer frame (`chatStore`) and FD's (`agent_handshake.identity_frame`)
  leave out `role=worker` agents.
- **Who's talking**: consult and delegate messages from one of the deck's process
  agents start with `[From <name>, <relation>]` (your parent / your child /
  your sibling, also under X / another of your owner's agents). The colleagues
  hand-over says the relation in place of "another of your owner's agents".
- **Report**: `POST /fd/family/agent/report {message}` (the calling agent's
  X-Agent-Auth, as the peer routes) gives the parent a turn on it — woken if
  asleep, in its latest session (a notification with `trigger_response`, as a
  delegated result), headed `[Report from <name>, your child … made for: …]`
  and marked as not the owner's words. 409 without a parent, 429 past 20 an
  hour. The agent's `flight_deck` tool has a `report` action.
- **Not done**: the session the child was made in isn't recorded, so a report
  goes to the parent's latest session; a report doesn't follow the parent's
  channel (Telegram etc.); Docker agents have no block.
- Deploy: FD restart, agent restarts (prompt loader, `flight_deck` tool), hard
  refresh (bundle).

## 13. As built (P3, 2026-11-17)

- **Tree rows** (`KioskAgentGroups.AgentSections`): an agent with a `parent`
  in the list shows right under it, indented 14px a level with a thin line,
  in its parent's group (decision 4: a child's own group membership is
  ignored; the header count is the whole tree). Rows stay sibling `<li>`s
  with `data-parent` and `aria-level`, so `e2e/groups` selectors hold.
- **Fold**: a chevron in the left gutter of a parent row (the plain list gets
  a 14px gutter when it holds a tree); a folded parent shows `+N` with a
  busy/unread dot; the open chat's agent stays in view. Per device in
  localStorage `fd:agent-tree-folded`.
- **Drag onto a row** (desktop): held over the middle half of a row for a
  moment (450 ms), an agent goes under it ("Put X under Y", violet); a
  refusal shows on the row in red with the deck's words (loop, past 3 levels,
  12 under one) and drops nothing. A drag that only passes over rows, or is
  let go at once, the row's edges and the gaps still move it between groups
  as before. A stopped agent can't be dragged from the middle of its row (a
  disabled button), as before. A child dropped on a
  section, on the plain list, or on "Drop here for a new group" is one of the
  person's own again (and goes into that group).
- **"⋯" → Put under** (desktop and phone): a sheet of the person's agents,
  "None — one of your own" first; ones the rules refuse are listed, disabled,
  with the reason.
- **Undo**: after a move, a bar under the list says what happened ("Deal
  digest is now under Glasses") with Undo, for 8 seconds.
- **Run workers**: a worker under an agent in the list has no row; that agent
  gets a "Working · N workers" line while they run (First Mate's: one line per
  factory, "Inbox triage · Working · 2 workers", which opens First Mate).
  Workers without such a parent (a run the person started) keep their row,
  without a power switch, as before.
- **First Mate (decision 5)**: its factory workers no longer go into the
  "First Mate" group (`processStore` skips `role=worker` in `autoPlace`;
  `isManagedProcess` has no exception for them); First Mate's own agent still
  lands there once (#621).
- **e2e**: `flight-deck/e2e/groups` has a `family` scenario (13 checks:
  through the deck, tree rows, fold, hold-drag, Undo, refused loop, picker).
  Four group checks fail on the unchanged sidebar too (P2 branch baseline):
  drag to Other agents, picker's "No group" first, drop above the pin, unpin.
- Client rules in `utils/agentFamily.ts` mirror `agent_family.py` (the deck
  still decides); components in `components/layout/AgentFamily.tsx`; API
  `services/docker.setProcessParent`.
- **Not done**: Basna/Vatra/Council workers still don't know who started
  their run (moved to P5); no virtual First Mate row when no First Mate agent
  is picked; the Today home still shows running workers as awake agents (P4).
