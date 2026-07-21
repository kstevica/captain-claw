# Captain Claw v0.7.8 Release Notes

**Release title:** Lanes, Queues That Finish · Iskre Shape Their World
**Release date:** 2026-07-21

0.7.8 is about **long work that runs unattended**. One agent gains three
parallel lanes; the task queue stops stranding itself and learns to plan its own
batches; the datastore tool stops turning one fumbled argument into thirty tool
calls of hunting. And Iskra's beings get hands: they craft and place objects,
break ground on impulse, hand work between their two brains, and the keeper can
paint roads and grow the plot beneath them.

The house rule holds: **everything new is opt-in, and defaults are unchanged.**
Lane A *is* the agent you already have. A queue with no plan behaves as before.
An agent that never opens the planner never makes an extra LLM call.

## Highlights

### Lanes — three parallel contexts on one agent

An agent could hold one conversation at a time: a single busy flag, a single
active session, and every token broadcast to every client. Three tasks meant
three turns, one after another.

Lanes give the same process three rooms — **A, B, C** — each with its own
session, transcript, queue and busy flag, running **at the same time**. Three id
ranges enrich in parallel against the **same** `fund_portfolio`, instead of
serially.

- **Lane A is the agent that exists today.** An omitted lane resolves to A, so
  WhatsApp, the glasses bridge, botport, cron turns and every REST caller keep
  the agent and session they always had. B and C are additions, not a migration.
- A tab strip over the chat with a state dot per lane — never opened, running,
  waiting on an answer, idle — plus a pending count and an **unread mark** for a
  lane that produced output while you were looking elsewhere.
- A lane's socket opens the first time you look at it. Files and the datastore
  are shared (all three lanes are one agent); session state is not.
- The concurrency machinery already existed for public/multi-tenant chat — this
  generalizes it rather than adding a second one.

Two bugs found on the way, both fixed here: scoped agents (lanes **and** public
sessions, long-standing) were **dropping narration entirely**, and every lane
shared one provider object, so a stall in lane A could force a tool call in
lane B and lose its own forcing in the process.

### A queue that finishes what it starts

Four ways a queue could stall, all closed:

- **A slash command** answered with a `command_result`, which the queue never
  watched for — `/new` spun forever and blocked everything behind it.
- **A question at the end of a finished task.** Any reply ending in `?` was read
  as "waiting for the user", with no timer and no signal. An agent that upserted
  both batches, verified them, printed a ten-row table and then offered *"did you
  mean a different range?"* stopped the queue. A trailing question now blocks only
  on a short reply, and a genuine question holds for **three minutes, visibly**,
  then moves on.
- **A give-up reply** (`I got stuck…`, budget/retries exhausted) was filed as
  completion, so real work was ticked off and skipped. The queue now **re-runs**
  it: verbatim first, then with a nudge shaped to how it failed — a
  budget-exhausted agent is told to keep what it produced and work in smaller
  steps; a stuck one is told to work through the obstacle. Three attempts, then
  it **stops rather than skipping**, and says so.
- **Lanes B and C opened in MANUAL** while A said AUTO, so their items sat
  dispatched forever. Auto-progress now inherits from the agent's existing lane.

Queued turns also skip the post-turn "suggested next steps" call and the
task-rephrase — the next message is already written, and rephrasing rewrites
instructions you chose word by word.

Each card now reports **what its run cost**: start and end times to the second,
elapsed (live while running), tool calls, and tokens in/cached/out — accumulated
across re-runs, so a task that gave up twice reports what it actually spent.
Plus **Clear finished**, a taller input, and full task text instead of 240
characters.

### The Task Planner — one description becomes a reviewed queue

Twenty-five near-identical queue messages differing only in an id range is a
morning's work to write, and how batches get skipped or overlapped. **Plan** in
the queue header takes one description (attach files if it needs them), and
returns a plan you read and edit. **Nothing is queued until you press Send.**

The model is never asked for the messages. It returns **one template plus the
overall range**, and Flight Deck expands it. That message is ~90% standing rules,
and a model asked to reproduce them twenty-five times paraphrases, compresses,
and eventually drops the clause that looked redundant — here `never do +1 on the
id!`, which corrupts a table. Expansion in Python means every task is
byte-identical except its range, one LLM call whether there are three batches or
two hundred, and slicing that cannot produce an overlap, a gap or an off-by-one.

- **Ranges are facts.** The agent's real tables, columns and the MIN/MAX of the
  batching key are read first; a plan for 1..500 against a table ending at 318
  would be eighteen tasks that can only fail.
- **Editing the template re-renders every task**, free — a task you hand-edit is
  pinned so template edits stop overwriting it.
- **Attachments** upload to the agent (tasks get the path) and are previewed for
  the planner (so it can see the ids inside), reusing the document extractors.
- **Continue where it stopped**: reopening says *"last plan covered `_id` up to
  490, 25 tasks, 3h ago"* and offers the next stretch — same approved template,
  no model call.
- Table and batch key are **dropdowns of the agent's real schema**, and the
  planner uses **the agent's own model and key**, not Flight Deck's.

### The datastore tool answers a fumbled call

From one agent's log: a dropped `table` named no tables and threw the payload
away; `data`, `table_name`, `select` and `order` were silently ignored or
rejected; a write sent to `action="sql"` was refused with no alternative.

- A dropped `table` **recovers** to the table this session was already writing
  to for reads and appends, and for `update`/`delete`/`drop` refuses **with the
  table list and a filled-in retry**.
- **Synonyms are renamed** — `data`, `table_name`, `select`, `order`, `sql`,
  `column_type`. `select` and `order` used to be dropped in silence, so a query
  succeeded and quietly returned every column unsorted.
- A **SQL write comes back translated** into the `update`/`insert`/`delete` call
  that would have worked. SELECT-only stays: protection rules live in the
  structured paths.
- **Every way a model asks for a range now works** — `BETWEEN`, paired
  operators, a list of conditions — and a repeated `where` key is combined
  rather than dropped. That last one was expensive: `{"id": …, "id": …}` silently
  became `id <= 379` and returned **1.38M characters**, 71% of a 200k context, in
  one tool message.
- **A re-read after a write is not a duplicate.** `query → upsert → verify` is
  the prescribed workflow, and the verify query was blocked with *"the content
  has not changed"* about a table just written to — sending the model hunting for
  "a fresh query pattern to avoid the duplicate guard" for 31 tool calls.

Also: **`web_get` hands over readable text first**, markup only if you ask twice,
and **a weak model can no longer fail in silence**.

### Iskra — the Iskre shape their world

Beings gain hands, and the keeper gains a trowel.

- **World shaping** — Iskre craft and place objects, name and re-look their
  homes, read each other's inscriptions, and a steward can commission public
  works. Objects are discovered by sight, sense and hearsay.
- **Restless Hands** — an eighth genome stat, *Impulse*: the body brain breaks
  ground on its own, wordlessly and free; the analytical mind later finishes it
  with a name and an inscription, or abandons it, and unclaimed ground crumbles
  in a day.
- **The work board** — the mind assigns tasks, the feet take up or refuse the
  ones that suit them, mid-walk, and the board sits in plain view on the being
  card.
- **The parent shapes the ground** — paint roads tile by tile, grow the plot,
  and place any of eight object kinds from the map or in first person.
- **Visiting beings have bodies** in the host village, write the host's world
  rather than their home siblings', and beings now sit **out front** of
  buildings and take distinct apron tiles in a crowd.
- Plus: a beings-loop owner lock (one Flight Deck ticks a village), letters
  grounded in where the being stands, and the feet's budget spent on answering
  rather than thinking.

## Upgrading

Nothing to migrate. Restart agents and Flight Deck, hard-refresh the browser.

Lane A is your existing agent and session; queues, plans and lane state are
per-browser. If lanes B/C were opened during testing, their auto-progress toggle
persists from that session — click it once if it reads MANUAL.
