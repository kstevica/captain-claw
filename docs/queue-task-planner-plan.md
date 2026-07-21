# Queue Task Planner — one description → a reviewed queue

**Goal.** Describe a complex job once, attach files if it needs them, and get
back a *plan*: a list of self-contained queue messages, batched, optionally
separated by `/new`. You read and edit the plan, then send it to a lane.

The shape to hit is the message the user writes by hand today:

> always start fresh, data you need to enrich is not in your memory. enrich
> data in table fund_portfolio, _id from 241 to 250. Use only these _ids, do
> not proceed automatically to the next batch… upsert data in batches of 5…
> _id and id are identical, never do +1 on the id!…

Twenty-five of those, differing **only in the range**, is a morning's work to
write by hand and the reason batches get skipped or overlap.

---

## 1. The central decision: template + parameters, not N generated messages

The obvious build is "ask an LLM for a list of messages". It is the wrong one.

That message is ~90% standing rules and ~10% batch specifics. Asking a model
to reproduce the rules twenty-five times invites exactly what models do to
repeated text: paraphrase, compress, drop the clause that looked redundant.
The one clause it drops (`never do +1 on the id!`) is the one that corrupts a
table. Cost scales with batch count, too.

So the model is asked for **one** thing:

```
{
  "template": "…the standing rules, with {range} placeholders…",
  "batches":  [{"from": 241, "to": 250}, {"from": 251, "to": 260}, …],
  "rationale": "…",
  "warnings": ["fund_portfolio has no rows above 1818 — capped there"]
}
```

Flight Deck expands `template × batches` **deterministically**. Every message
is then byte-identical except its range: no drift, no summarization, no lost
clause, and the cost is one call whether there are 3 batches or 200.

It also makes the review UI honest — editing the template edits all messages
at once, which is what a user means by "actually, always use English".

## 2. Grounding: the planner must not invent ranges

`_id from 241 to 250` is only correct if the table really has those rows. The
planner gets the target agent's actual datastore facts, fetched before the
call via the existing proxy (`/fd/agent-datastore/{host}/{port}/tables`):

- table names, row counts, column names and types
- for the batching key (usually `_id`): min, max, and the count of rows whose
  target columns are still empty

That last one matters: "enrich what's missing" should produce batches over the
*unfilled* rows, not blindly over 1..1818. Ranges become facts rather than
guesses, and the model gets a `warnings` slot to say when the request exceeds
what exists.

Attached files are uploaded to the agent first (existing `uploadFileToAgent`),
so generated messages can reference real paths. Text-ish files also get a
short extract passed into the planning call so the planner can see, say, the
ID list in a spreadsheet — capped hard, since this is one small call.

## 3. Flow

```
Queue panel  ─ "Plan tasks…" ─▶  Modal
                                  │  describe the job + attach files
                                  │  batch size (5–10), key column, lane
                                  ▼
                          POST /fd/queue/plan
                          (datastore facts + file extracts + intent)
                                  │  ONE LLM call
                                  ▼
                          template + batches + warnings
                                  │  expanded deterministically in FD
                                  ▼
                        Review: N message cards
                          edit template (all) · edit one · drop · reorder
                          ☑ /new between tasks
                                  ▼
                          "Send 25 tasks to lane B"
```

Nothing reaches the queue until that last click. The plan is a proposal.

## 4. Backend

`POST /fd/queue/plan`, modelled on `/fd/basna/recommend` — one call, no
session, spawns nothing, uses the registry's `reason` tier
([basna_routes.py:1404](../captain_claw/flight_deck/basna_routes.py)).

```
{ intent, host, port, batch_size, key_column, max_tasks, file_refs[] }
   → { template, batches[], messages[], rationale, warnings[], facts }
```

`messages[]` is the expansion, returned so the client renders exactly what
will be sent rather than re-implementing the expansion in TypeScript.

Guards worth having from the start: `max_tasks` (default 50) so a fumbled
range can't enqueue 1,800 items; batch size clamped 1–50; and the whole
response is a proposal with no side effects, so a bad plan costs one call.

## 5. Frontend

- **Trigger**: a "Plan tasks…" button in the queue panel header, next to
  AUTO/MANUAL.
- **Modal, two panes**: left is the request (description, attachments, batch
  size, key column, target lane, `/new` toggle); right is the returned plan.
- **Review affordances**: the template in one editable box at the top —
  editing it re-expands every message live — plus per-message edit and remove.
  A message the user hand-edits is pinned so template edits stop overwriting
  it (and says so).
- **Transfer**: `enqueueQueueMessage` per task against the selected lane's
  key, interleaving `/new` when the toggle is on. The lane's existing queue
  is appended to, never replaced.

## 6. Phases

| phase | scope |
|---|---|
| 1 | `POST /fd/queue/plan` + datastore-facts gathering + the expansion, with tests on the expansion and the guards |
| 2 | The modal: request pane, plan pane, editing, transfer to lane |
| 3 | Attachments: upload to agent, extract for the planner, reference in messages |
| 4 | Re-plan / continue: "another 25 from where this left off", seeded by the last batch's end |

Phases 1–2 are the feature. 3 and 4 are worth having but nothing depends on
them.

## 7. Deliberately not in v1

- **Running the planner on the target agent.** It has the datastore and the
  files right there, but it would occupy the lane, pollute a session, and
  couple planning to whatever model that agent runs. The facts it has are
  cheap to fetch over the existing proxy instead.
- **Auto-send.** A plan that enqueues itself is a plan nobody reads, and the
  failure mode is 25 wrong tasks instead of one.
- **Cross-task dependencies.** Queue items are independent by construction —
  each is a fresh session. Anything needing real handoff is Vatra's job.
