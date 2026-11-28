# Team up (mini Vatra): an agent splits a task among copies of itself

Status: **P0 built** on branch `feat/team-up-p0` (PR #818), with the chat UI
and the per-agent setting pulled in from P2. **P1 (the board)** on
`feat/team-up-p1` (PR #819), stacked on P0. **P2 (the rest of the chat UI)**
on `feat/team-up-p2` (PR #820), stacked on P1. **P3 (learning and recall)**
on `feat/team-up-p3` (PR #821), stacked on P2. **P4 (Telegram and WhatsApp)**
on `feat/team-up-p4`, stacked on P3. Owner decisions in. See the "as built"
sections at the end.

## The idea

In a web chat with an agent, the agent sees that a task has 2–4 independent
pieces and offers to **team up**. It starts 2–4 **helpers**: forks of itself
with the same memory, tools, connections and model. They share the work and
talk on a Vatra-style board. Meanwhile the person can see every helper's
progress and keep talking to the agent, and the agent can check on the team.
When the team is done, the agent writes the reply.

Vatra stays what it is: a separate run in Flight Deck, with specialist
archetypes as separate processes, a Group 0 planner and an approval gate.
Team up lives inside one agent's chat, starts in about a second, and has no
gate beyond one tap.

## Owner decisions (2026-11-27)

| # | Question | Decision |
|---|---|---|
| 1 | Name people see | **Team up** (the button and setting); the panel is "Team"; the copies are "helpers" |
| 2 | Default per agent | **Ask me** |
| 3 | Shared agents (one agent, many members) | **Off** in v1 |
| 4 | Budget per team | **A multiple of a normal turn** |
| 5 | Seeing the team | The person sees **each helper's progress** live |
| 6 | Talking during the run | The person can **keep talking to the agent** while the team works, and the agent can **check the team's status** |

Decisions 5 and 6 mean the team runs **in the background**, not inside one
blocking tool call. See "The run".

## What already exists

| Piece | Where | Use |
|---|---|---|
| `AgentPool` | `captain_claw/agent_pool.py` | In-process worker `Agent`s that share the provider, deep memory, instruction cache and the tool registry (per-call overrides). The base for running helpers. |
| `SessionOrchestrator` (`/orchestrate`) | `captain_claw/session_orchestrator.py`, `chat_handler.py:2029` | Admin-only. Splits into a DAG, independent workers, separate synthesis call. Workers are fresh sessions with `max_iterations=5`, not forks, and they never talk to each other. Team up replaces it later. |
| `SharedWorkspace` | `captain_claw/shared_workspace.py` | Run-scoped, in-memory namespaced key-value store. A base for the board's storage. |
| Child sessions | `agent_session_mixin.ensure_pipeline_subagent_contexts` | `metadata.subagent.spawned_by / spawn_depth`. The convention for helper sessions. |
| `/btw` notes | `ws_handler.py:806` → `agent._btw_instructions` → `agent_context_mixin.py:5605` | A per-agent list added to every LLM call of a running turn. The way to push messages into a helper mid-run, and the lead's live "Team now" block. |
| `vatra` tool | `captain_claw/tools/vatra.py` | The protocol (search, map, read, post, ask, inbox, wait). Today it goes to Flight Deck over HTTP. Team up keeps the verbs and runs them in memory. |
| Notification turn | FD `_notify_agent_turn` (agent family `report`) | The pattern for starting a lead turn when the team finishes. In-process, the runner calls the same entry the agent's notify endpoint uses. |
| Rehearsal rule | Captain Swarm, `rehearsal_turn.py` | "Outward actions become intents". Helpers follow it, and the lead carries the action out once. |
| Run cost | `flight_deck/pricing.py`, the `turn_usage` broadcast (`agent_guard_mixin.py:442`) | Tokens and $ per helper and per team. |

## Design

1. **In-process helpers, not Flight Deck processes.** One agent process runs
   the lead plus N helper `Agent` instances (AgentPool style). They share
   memory stores, connections (Gmail, calendar, IMAP), files and config,
   because they really are the same agent. It also works for an agent that
   isn't under Flight Deck.

2. **A true fork, built for a cache hit.** A helper starts from the lead's exact
   prefix: the same tools, the same system prompt, the same history up to the
   moment the team starts. Only the last message differs ("You are Prices, one
   of 3 helpers. Your part: …"). On hosted providers with prompt caching, most
   of each helper's input is then cached, and the helper knows everything the
   person said. The tool list must stay identical, so limits are enforced when
   a tool is called (see 5). Fallback for small context windows: a short brief
   that the lead writes.

3. **Trigger: the agent proposes, the person taps.** The model calls
   `team propose` with the parts when the task really splits. With the
   per-agent setting **Ask me** (default), the chat shows a card:
   "Team up: 3 helpers — Prices · Reviews · Delivery" with **Team up** and
   **Do it alone**. With **On**, it starts without asking. With **Off**, the
   tool isn't offered. The person can force a team with `/team <task>`.
   Refused on shared agents, public sessions and member turns (decision 3).

4. **The lead is the agent itself.** It doesn't take a part itself. It stays
   free to talk with the person, answers helpers' questions addressed to it,
   steers the team, and writes the final reply in its own voice.

5. **Helpers don't act outward. The lead does, once.** Sending mail, WhatsApp or
   Telegram, publishing, payments, calendar writes, cron changes, model or
   effort switches and agent spawns are refused inside a helper, with "post a
   draft to the board; the lead sends it". Reading is fine.

6. **Memory: shared reads, writes controlled.** Helpers read every memory
   layer. Automatic post-turn learning runs **once, on the lead's finishing
   turn**, over the team record, never per helper. A helper's deliberate
   durable write (facts, playbooks, personality) becomes a board note
   `remember: …`, and the lead decides.

7. **One browser holder at a time.** A lease for the agent's browser (and the
   person's Chrome): the first helper that needs it holds it, and the others,
   the lead included, get "browser in use by Prices, ask on the board or
   wait". The lead can give all browser work to one part from the start.

8. **Budget: a multiple of a normal turn** (decision 4).
   - A normal turn = the median total tokens of this agent's last 50 chat turns,
     from the `turn_usage` records, with a 30k floor and a default for new agents.
   - Team budget = normal turn × the setting (default **6×**). It's one pool for
     the whole team, and no single helper may take more than half of it.
   - At 80% the panel warns and the lead is told. At 100% each helper gets one
     last step to post what it has and call `done`, and is then stopped.
   - No wall-clock limit; Stop ends it.

9. **Other limits.** 2–4 helpers (setting, default up to 3); depth 1 (a
   helper can't team up, enforced by a helper marker); at most 12 asks per team;
   one team per agent at a time.

10. **Local models: cap or switch off.** On a self-hosted server with one KV
    slot, parallel helpers evict each other's cache and run more slowly than one
    agent. Use at most `slots` helpers, and offer no Team up when the provider
    is local and has one slot.

## The run

```
person ──▶ lead turn: "compare these 3 suppliers"
            └─ team propose [Prices, Reviews, Delivery] ──▶ card: [Team up] [Do it alone]
person taps Team up
            ├─ runner forks 3 helpers from the lead's prefix (background task)
            └─ chat line: "3 helpers are on it. You can keep talking to me."
                       ┌──────────────────────────────────────────────┐
person ◀─▶ lead turns  │ helpers work, post, ask each other, wait      │ ──▶ panel (live)
 (normal chat; "Team   │ the lead's `team tell` arrives at a helper's  │
  now" block on every  │ next step; questions for the person show as   │
  lead call)           │ a dialog labelled with the helper's name      │
                       └──────────────────────────────────────────────┘
all helpers done/stopped ──▶ runner starts a lead turn with the results ──▶ final reply in chat
```

1. **Start.** On Team up, the runner forks the helpers, each in its own child
   session (`metadata.subagent.spawned_by = <lead session>`, kept out of the
   session list), and runs them as one background task owned by the agent
   process, not by the lead's turn. The lead's turn ends at once with a short
   line.
2. **The person and the lead talk normally.** While a team runs, every lead LLM
   call gets a small **"Team now"** block at the end of its context (one line per
   helper: state, last step, tokens; open asks to the lead; budget used). It is
   built the same way as `/btw` notes, never part of the cached prefix, and capped
   at ~600 tokens. The lead can answer "how's it going?" without a tool call.
3. **The lead's verbs** (the `team` tool, outside helpers):

   | Verb | What it does |
   |---|---|
   | `propose` | Offer a team with these parts (card under Ask me, start under On). |
   | `status` | Details: each helper's part, state, recent steps, tokens, files, open asks. |
   | `read` | The board, optionally filtered by helper or kind. |
   | `tell` | A message to one helper or all, delivered at their next step ("Delivery: only EU suppliers"). |
   | `stop` | Stop one helper or the whole team. |

4. **Questions.**
   - Helpers ask each other on the board.
   - For the person, a helper uses `ask_user` as usual, and the dialog says "Prices asks".
   - For the lead, a helper uses `ask lead`. If the lead is in a turn with the
     person, the question shows in "Team now" and the lead answers with
     `tell`. If the lead is idle, the runner starts a short quiet lead turn
     whose answer goes to the helper, not to the chat. The panel shows
     "Lead answered Prices".
5. **Finish.** When every helper has called `done`, failed or been stopped,
   the runner starts a lead turn with the results (each part's summary, files,
   open issues, board digest, cost). The lead writes the reply into the chat.
   It can redo a failed part itself, and it carries out any outward actions the
   helpers drafted, once, with the usual confirmations. If the person is mid-turn
   with the lead at that moment, the finish waits until that turn ends.
6. **Record.** The team record (parts, who did what, board digest, cost, helper
   session ids) is saved on the lead's session, so later turns can answer "what
   did Prices find?".

Files: each helper works in `team/<run>/<helper>/`, and deliverables go to
`saved/`. The rule "chat sends only saved/ files made this turn" counts helpers'
files as files of the finishing turn.

Stop: the chat's Stop button stops the lead's current turn only. The team
has its own **Stop all** in the panel, and "stop the team" said to the lead
works too.

## How helpers talk (the Vatra protocol, in memory)

One `TeamBoard` object per team (built on `SharedWorkspace`), plus asyncio
events. The `team` tool inside a helper:

| Verb | What it does |
|---|---|
| `post` | Share a finding, decision or draft (`text`, `title`). |
| `read` / `search` | Read the board, or search it by keyword. |
| `map` | Who covers what, which sources were read, which files exist. |
| `ask` | Ask a named helper, `anyone` or `lead` for new work. Doesn't block. |
| `inbox` | Answers to my asks. |
| `wait` | Block until a helper's file or post exists (timeout, then a board digest). |
| `done` | Finish my part: a summary, files, open issues. Required. |

**Push, not only pull.** Before each LLM call a helper makes, a short "New for
you" block (built like `/btw` notes) adds asks addressed to it, answers to its
asks, the lead's `tell` messages, and posts that match its part. It is capped at
~1.5k tokens. A helper reacts without having to poll.

## What the person sees

- **The proposal card** in the transcript. After Team up it becomes the team's
  status card (helpers, states, tokens), and it opens the panel.
- **A Team pill** in the conversation bar, next to Queue: "Team · 2 working · 1
  done". It stays while the person chats with the lead.
- **The Team panel** (drawer, can stay open while chatting):
  - **Header:** elapsed time, tokens used against the budget ("41k of 300k"),
    $, and **Stop all**.
  - **One card per helper:** the agent's picture with the helper's name (from
    its part), the part in one line, a state chip, a live step list (narration,
    tool lines, the "Calling LLM (model)" line, as on Vatra's live cards),
    tokens and $, and **Stop**. States: working · waiting on Reviews · asking
    you · done · failed · stopped.
  - **A helper's card opens its full transcript** (its child session, read-only).
  - **Board tab:** posts, asks and answers in order, plus the lead's `tell`
    messages.
- **Coming back later:** the team keeps running if the tab closes. The panel
  rebuilds from the team record, the helper sessions and a per-helper event log
  (last 200 steps).
- **Events:** the agent's websocket broadcasts `team_*` (`team_started`,
  `helper_step`, `helper_state`, `board_post`, `team_budget`, `team_done`). Flight
  Deck's chat already proxies that socket.

## Settings (per agent)

- **Team up:** Off / **Ask me** / On.
- **Most helpers:** 2 / **3** / 4.
- **Team budget:** **6×** a normal turn (3×–12×).
- Not offered on shared agents (decision 3), or when the provider is a local
  server with one slot.

## Other lanes (later)

- **Telegram** (P4): it has live narration. One line ("Teaming up with 3
  helpers: …"), the person can keep chatting with the lead, then the reply.
  Questions go through the lead. No per-helper narration.
- **WhatsApp** (P4b): the same, with fewer lines.
- **Scheduled and autonomous turns:** off in v1. The cost could run away, and
  nobody is there to press Stop.

## Phases

- **P0, background runner and lead control.** `team propose`/`start` (setting
  through config), in-process forks from the lead's prefix as a background task,
  helper marker, no recursion, outward-action refusal, `status`/`tell`/`stop`,
  the "Team now" block, the finishing lead turn, the team record, the budget
  pool. Only plain chat lines, no panel. **Prove:**
  - concurrency is safe while the lead and 3 helpers all run in one process;
  - the cache-hit rate on Anthropic/OpenAI, measured;
  - the lead answers mid-run and steers a helper with `tell`.
- **P1, board.** `TeamBoard`, the helper verbs, the push block, `ask lead` with
  quiet lead turns, browser lease, memory write rule.
- **P2, chat UI and settings.** The Ask me card, Team pill, the panel with live
  helper cards, helper transcripts, the board tab, Stop per helper / all, the
  budget line, `/team`, the three settings.
- **P3, learning and record.** Post-turn learning once over the team record,
  `remember:` notes, answering "what did Prices find?" later.
- **P4, lanes.** Telegram, then WhatsApp.
- Then: retire `/orchestrate`, or point it at Team up.

## Risks to check in P0

- **Shared mutable state in one process.** The tool registry is a global
  singleton (per-call overrides exist). Check what else is global: current
  session in singletons, `get_config()` mutations, `model_switch` /
  `reasoning_effort`, cwd, the file registry, approvals, and the chat
  handler's `server._busy` / `agent._lane_busy` flags (helpers must not set
  them, and the lead's turns must not wait on the team).
- **Event loop blocking.** CPU-heavy tools (OCR, cv, document extraction)
  that run on the loop would stall every helper and the lead's chat. Check
  which ones already run in a thread.
- **Where turn usage lives.** The normal-turn median needs the last 50 chat
  turns' tokens. Find where `turn_usage` is stored, or start recording it.
- **The finish turn and the session.** If the person starts `/new` or switches
  topic during a run, the reply still goes to the session the team started in,
  and the chat says so.
- **When not to split.** Tool description and prompt: propose only when there
  are 2+ independent, substantial pieces (research N companies, compare
  options, go through a big folder in parts, gather data while drafting).
  Never for short answers, sequential steps, or work on a single account
  (one inbox).


## P0 as built (2026-11-27)

**Agent side**
- `captain_claw/team_up.py`: settings (`team_up.json` in the agent's home), the
  normal-turn baseline (`team_up_turns.json`, recorded after each owner chat
  turn), team and helper state, propose / start / dismiss / stop / tell, the
  background runner, the finishing turn (`handle_chat` with a
  `[Team '<title>' finished]` opener, origin `delegated_result`), the record on
  the lead's session (`team_runs`), the helper guard and the live notes.
- `captain_claw/tools/team.py`: the `team` tool (propose, start, status, tell,
  stop; helpers only `status`). Listed only while helpers can run here and the
  setting isn't Off (`requires_team_up`), and in tool search's always-shown set
  (the model has to think of it by itself).
- Helpers are `_build_scoped_agent(..., register_tools=False)` agents, like
  lanes, in child sessions with `subagent.spawned_by` and
  `files_session_id = <lead session>`.
- Guard in `ToolRegistry._execute_call` (`team_up.tool_block`): helpers can't
  send, pay, change accounts, memory or settings, or start other runs or agents;
  the browser has one holder (the lead included); a helper's `ask_user` question
  starts with its name.
- Live notes in `_build_messages`: the lead's "Team now" rides on the last
  message as a live note (`team_status`), and a helper's messages from its lead
  go in as an instruction block, like `/btw`.
- Post-turn learning (reflection, insights, topics, lessons) runs only in the
  chat handler, so helpers never trigger it; it runs once, on the finishing turn.
- Websocket: `{"type": "team", "event": proposed|started|team|helper|step|draft|
  sync|error}` to the lead's lane; `team_action` (start / dismiss / stop) and
  `team_sync` from the chat; a sync after every connect.

**Flight Deck**
- `GET/PUT /fd/agent-team-up/{kind}/{id}`. A Team up control (Off / Ask me / On,
  helpers 2–4, budget 3×–12×) in Agent settings and in the kiosk agent dialog.
- `stores/teamStore.ts`, `components/agents/team/TeamUp.tsx`: the card in the
  transcript (proposed → working → done), the Team pill in the conversation bar
  (compact in the full layout), and the Team panel docked beside the chat (a view
  of its own on a phone) with a live card per helper, a merged timeline, the
  budget meter, and Stop per helper and for all. The finishing turn's opener
  shows as a "Team results" note.

**Checked**
- 53 unit tests (`tests/test_team_up.py`).
- A run on the real stack with a scripted model: a real WebServer, a real lead
  Agent and three helpers forked in one process.
  - Helpers ran at the same time.
  - The guard refused `send_mail` and `team stop`.
  - The person talked to the lead mid-run, and the lead saw "Team now".
  - The lead's `tell` reached the helper's next call.
  - The finishing turn saw every result and wrote the reply.
- Prompt cache: the first try shared only 37% of the prompt between lead and
  helper, because the system prompt names the session's folder. With
  `files_session_id` it shares 97% (identical system prompt and tool list).
- UI: the real ChatPanel in a harness with a fake agent socket, dark and light,
  desktop and phone.

**Not done yet**
- A run with a real model, and a measured cache-hit rate on a provider.
- Helper sessions show in the session list (no filter yet).
- The `remember:` notes, and answering "what did Prices find?" from the record
  (P3).
- Telegram and WhatsApp (P4).


## P1 as built (2026-11-27): the board

**Agent side**
- `captain_claw/team_board.py`: one board per team, in memory. Entries are post,
  ask, answer and remember. A board event wakes waiters. Limits: 12 asks per
  team, 40 posts per author, 300 entries, 4,000 characters per text.
- `team_up.py`:
  - `board_post` turns "remember: …" into a remember note.
  - `board_ask` takes a helper's name, `anyone` or `lead`, refuses duplicates
    and self-asks, and doesn't block.
  - `board_answer`, `board_inbox` (wait up to 30 s), and `board_wait` (on a
    helper's result, or a post matching a query; up to 120 s, then a board
    digest).
  - `board_read`, `board_search` and `board_map`.
- **Asks to the lead** are answered at once and quietly: one call on a copy
  of the lead's provider, no tools, nothing in the chat. Its context is the
  conversation before the team, everything the user and the lead said since,
  the helpers' parts and results, the board, and the question. The tokens
  count toward the team budget (`lead_tokens`).
- **Routing to the lead.** The lead also answers asks to a helper that has
  finished, and asks to `anyone` when nobody else is working. When a helper
  finishes, its open asks go to the lead (`_route_open_asks`).
- **Push.** A helper's "New for you" block now carries the lead's messages,
  asks addressed to it, its own asks with their answers, and one-time
  pointers to new posts. It is capped at 6,000 characters.
- **The lead's "Team now"** shows the asks addressed to it with its answers,
  plus the board's size.
- **Memory.** Memory writes still refuse; the refusal now points to
  `team post "remember: …"`. The finishing turn lists the remember notes
  ("keep the ones that will matter") and up to 4 unanswered asks. The record
  keeps the last 20 entries.
- **The `team` tool** has the board verbs (post, ask, answer, inbox, wait,
  read, search, map) and a 150 s timeout. The lead can read, search and map.
  Helpers can't start, tell or stop.

**Flight Deck**
- `teamStore`: `board` entries on each team, `{"event": "board"}`, and an
  answer marks its ask as answered.
- Team panel: a **Board** tab, with a count badge that turns blue while asks
  are open.
  - Posts show as cards. Asks show who asks whom and Open / Answered, with
    the answers threaded under them; the lead's answers show under the
    agent's name.
  - "Worth remembering" notes are amber.
  - The tab follows new entries while the team works.
- Board steps (post, ask, answer, remember) show in the helper cards and the
  timeline.

**Checked**
- 65 unit tests.
- The regression subset fails exactly as main does: the same 24 tests.
- The real-stack run, with a scripted model:
  - Prices posted, was refused `send_mail`, answered Reviews' ask, and posted
    a remember note.
  - Reviews asked Prices and got the answer through `inbox`.
  - Delivery asked the lead. The quiet answer arrived, and its context held
    the question.
  - The lead's `tell` still reached Delivery.
  - The finishing turn saw every result and the remember note.
- Board UI: the harness in dark and light, desktop and phone.


## P2 as built (2026-11-28): the rest of the chat UI

Most of P2 (the card, the pill, the panel, Stop, the budget line, the
settings) came with P0 and P1. This phase adds the rest.

**`/team <task>`**
- `complete()` rewrites it into a directive, as `/code` and `/publish` are
  rewritten: split the task, call `team propose`. It also sets
  `agent._team_forced`, and `team_up.propose` then starts the team at once,
  as "started by you", with no card to tap.
- `/team` with no task gets a short help. Where Team up can't run, the person
  gets the reason in their own terms (`refusal_for_person`, e.g. "Team up is
  off for this agent. Turn it on in … Agent settings → Team up.").
- The web chat routes `/team` like `/code`, and `/help` lists it.

**Helper transcripts**
- A helper card has **Transcript**. It sends `team_transcript`, and the agent
  answers with `{"event": "transcript"}`.
- The agent sends only the helper's own messages, after the conversation it
  inherited (`fork_len`), in small form:
  - the brief;
  - the helper's text, with its tool calls as name + arguments;
  - tool results, cut to 1,500 characters.
  At most the last 150 items.
- The panel shows it in place of the tabs' body. The brief and the tool results
  fold. While the helper works, the panel re-reads it every 4 s. Esc goes back.

**Cards after a restart**
- The team record on the lead's session now holds everything a card needs:
  - each helper's part, state, tokens, usage, model, result (2,000 characters),
    files and session id;
  - the board (the last 40 entries);
  - `fork_len`.
- `sync_frames` adds the records of teams that are no longer in memory
  (`record_snapshot`). A team that was working when the agent stopped shows as
  stopped ("the agent restarted while the team worked"), and its working helpers
  as "Stopped by a restart". Their transcripts still open from their sessions.
- A record that changes while no lead turn is running (a team stopped from the
  panel) is saved right away.

**Cost**
- Each helper's and the team's dollars come from `flight_deck/pricing.py`,
  including the lead's quiet answers. They show in the budget line and on each
  helper card. They are left out when the model has no known price.

**Session lists**
- Helpers' sessions (`metadata.team_helper`) are left out of the owner's lists:
  `list_owner_sessions` (used by `/sessions`, `/session list`, the sessions
  tool and the remote commands), `GET /api/sessions` and the MCP session list.
  Loading one by id still works.

**Checked**
- 73 unit tests (8 new).
- The real-stack run with a scripted model, now with P2:
  - The transcript of a real helper session lists the brief, the `team` and
    `send_mail` calls with their results, and the result.
  - After the in-memory teams were cleared, as a restart would, the card came
    back from the record, and the transcript opened from the record.
  - `/team Compare suppliers D and E` went through the slash command,
    `complete()` and `team propose`. The team started "started by you", ran
    and delivered.
  - The owner's session list showed none of the 5 helper sessions.
  - The cost came out at $0.0022 for the scripted usage.
- UI: the harness in dark and light, desktop and phone: the cost, the
  transcript view, and a card rebuilt from a record.


## P3 as built (2026-11-28): learning once over the team, recalling it later

**Learning, once**
- After learning on the finishing turn has ended, the runner runs insight
  extraction **once** over a digest of the team. It runs only when the
  finishing turn reached the chat and the team is done.
  - The digest (`team_digest`, at most 3,000 characters) holds each helper's
    part and result and the `remember:` notes.
  - Extraction uses `trigger="team"`.
- Why: the chat's own learning sees only the last 15 messages, each cut to
  500 characters, and runs on a cooldown. On its own it would learn almost
  nothing from the helpers' results.
- Helpers still never learn: none of their turns go through the chat
  handler's learning jobs.
- It honours the insights setting (enabled + auto-extract). It waits while the
  lead is busy or another extraction runs, at most 10 minutes.
- The record gets `learned_at`.
- `remember:` notes: as in P1, the finishing turn lists them and the agent
  keeps the ones that matter with its memory tools. They are also in the
  digest above.

**"What did Prices find?" later**
- New lead action **`team recall`**, which works from the team records on the
  conversation's session, so it also works after a restart:
  - With no arguments, it lists the conversation's teams, newest first, with a
    line per helper.
  - With `helper`, it gives that helper's **full** result, read from its own
    session (its last plain answer after the conversation it inherited). It
    falls back to the record's 2,000 characters. Its files and its posts on
    the board come with it. `query` narrows which team.
  - With `query`, it searches the helpers' results and the boards of every
    recorded team.
- With no team in memory, `status`, `read`, `search` and `map` answer from the
  records too, instead of "You have no team".
- The tool description tells the model to use `recall` for questions about
  what an earlier team found.

**Checked**
- 78 unit tests (5 new).
- The real-stack run, with a scripted model:
  - After the team finished, the real insights path ran once, with the team
    digest in its context, and the record got `learned_at`.
  - The in-memory teams were then cleared, as a restart would. The person
    asked "what did Prices find …?". The lead called `team recall
    helper=Prices`. The tool returned Prices' full result from its session,
    and its board post, and the reply was built from it.


## P4 as built (2026-11-28): Telegram and WhatsApp

Team up now works in the owner's own Telegram and WhatsApp chats with the
agent, as well as in the web chat. Both are turns on the main conversation
(lane A).

**Where the person is (`turn_place`)**
- A turn's channel decides:
  - **web**;
  - **telegram**: the owner's private chat. It answers with `{chat_id, user_id}`,
    captured during the turn, since the agent holds them only then;
  - **whatsapp**: it answers with `{waid}`, the number of the turn being
    answered, else the session's `whatsapp_waid`.
- Refused elsewhere, in the agent's words and the person's:
  - other surfaces (glasses, Messenger, devices);
  - Telegram users who aren't the owner (their agents aren't lanes);
  - Telegram groups and shared agents.
- The team keeps `channel` and `reply_to`. They are in the snapshot and the
  record.
  - A start from a turn (the agent's `team start`, mode On, or `/team`) follows
    that turn's channel.
  - A **tap in the web chat** moves the results to the web chat.

**The run on Telegram / WhatsApp** (as planned: one line at the start, the
person keeps chatting, then one reply)
- **No card to tap.** Under "Ask me", `team propose` tells the lead to ask in
  one short line, naming the helpers; "yes" → `team start`. The web chat still
  shows the card, so a tap there works too. Under "On" and with `/team`, the
  team starts at once, and the lead ends with one line.
- **The person keeps talking to the lead on the same chat.** Every lead call
  gets "Team now". The web chat shows the live panel, since the frames go to
  lane A. There is no per-helper narration on the phone.
- **Questions go through the lead.** A helper's `ask_user` is refused with
  "ask the lead with team ask to='lead'". Its brief says the user is on
  Telegram/WhatsApp. The lead answers at once from what the user said (P1).
- **The reply**
  - **Telegram** (`_deliver_telegram`):
    - It waits quietly until the main conversation is free. The owner lane
      would otherwise post "yours is next".
    - It takes the user's lock and the owner lane, and runs the lead's turn on
      the results message, as `peer_relay`/`delegated_result`.
    - The reply is sent with `_tg_send`. The web chat gets the "Team results"
      note and the reply.
    - Files are not sent to Telegram in v1; the reply names them.
  - **WhatsApp**: the finishing turn is the usual lane-A turn, which Flight
    Deck's channel bus already relays to the person's number. It runs with
    `surface="whatsapp"`, so the reply is written for WhatsApp, and with
    `whatsapp_media_to=<waid>`, so files the turn produces go to that number.
- **`/team <task>`** passes through to the agent's turn on both: Telegram's
  command handler lets it fall through, as `/code` does, and Flight Deck's
  WhatsApp bridge queues it as a chat command.
- **Flight Deck:** a small "Telegram" / "WhatsApp" chip on the card and the
  panel. The footer says where the results go.

**Checked**
- Unit tests for each part: where the person is, the offer as a question,
  `ask_user` refused, a web tap bringing the results to the web, the reply
  going to Telegram (owner lane, send, web mirror), and WhatsApp's surface and
  number.
  84 unit tests in all.
- Real-stack runs with a scripted model:
  - **Telegram.** The run went through the real owner turn path
    (`_tg_process_with_typing`), with a fake bridge:
    - The proposal recorded `{chat_id, user_id}`, and the lead asked on
      Telegram.
    - "Go ahead with the team" made the lead call `team start`. The team
      followed the Telegram turn.
    - The finishing turn saw every result. Its reply went to the Telegram chat,
      and the web chat got the "Team results" note and the reply.
    - `/team …` on Telegram fell through the command handler to the agent and
      started a team at once, which was delivered to Telegram.
    - Both records say `telegram`.
  - **WhatsApp.** The turns went through `handle_chat` with the person's number,
    as Flight Deck sends them:
    - The team recorded `{waid}`.
    - The finishing turn ran with `surface="whatsapp"` and
      `whatsapp_media_to=<waid>`, and its reply went out on lane A, which
      Flight Deck relays to the number.
- UI: the chip and the footer, in the harness.
