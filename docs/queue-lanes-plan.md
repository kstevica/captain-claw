# Queue Lanes — A, B, C in one chat

**Goal.** One agent, three independent lanes. Each lane has its own chat
transcript, its own queue, and its own CC session, and the three run *at the
same time*. Selecting a tab — `A - Agent`, `B - Agent`, `C - Agent` — swaps
the chat pane to that lane. Nothing about lane B is visible in lane A.

The motivating workload: enriching `fund_portfolio` in ID ranges. Three lanes
working 370-379, 380-389, 390-399 concurrently against the **same** datastore
table, instead of one lane doing thirty rows serially.

---

## 1. What stands in the way today

Three separate serializations, all deliberate, all in different places.

| Where | What it does | File |
|---|---|---|
| `server._busy` | One boolean for the whole process. A second message while set is rejected with *"Agent is busy processing another request"* | [chat_handler.py:194](../captain_claw/web/chat_handler.py) |
| `server.agent.session` | One active session per process. `/new` *switches* it — it does not add a lane | [slash_commands.py:126](../captain_claw/web/slash_commands.py) |
| `server._broadcast` | Streams tokens to **every** admin client. Two lanes on one agent would cross-talk into each other's transcript | [web_server.py:650](../captain_claw/web_server.py) |
| `tryDispatchNext` | Client-side: refuses to dispatch while an item is in flight | [chatStore.ts:1551](../flight-deck/src/stores/chatStore.ts) |

The client guard is a *consequence* of the server ones. Fix the server and the
client guard becomes per-lane bookkeeping rather than a global gate.

## 2. What we already have

The hard part is built. Public/multi-tenant mode already runs **one Agent
instance per session inside one process**:

- `_get_public_agent(session_id)` — lazily creates an Agent per session with
  its own session, instruction cache, and lock ([web_server.py:320](../captain_claw/web_server.py))
- `_public_active_ws[sid]` + the `_make_send(sid)` closure — callbacks that
  stream to *one* socket instead of broadcasting ([web_server.py:349](../captain_claw/web_server.py))
- `agent._public_busy` — a per-Agent busy flag, so two public agents run
  concurrently ([chat_handler.py:184](../captain_claw/web/chat_handler.py))
- `ws._public_session_id`, stamped at connect ([ws_handler.py:46](../captain_claw/web/ws_handler.py))

**This plan generalizes that machinery from "public session" to "lane" and
makes it available in admin mode.** It is not new concurrency infrastructure;
it is removing the `if public_mode` around infrastructure that already works.

## 3. Design

### 3.1 A lane is a labelled Agent + session

```
ws://…/ws?lane=B
   └─ ws._lane = "B"
      └─ server._lane_agents["B"]  → Agent(session="lane-B")
         ├─ own busy flag           (parallel with lanes A and C)
         ├─ own transcript          (streamed only to lane B's sockets)
         └─ own /new, /model, /planning
```

### 3.2 Lane A **is** the agent that exists today

An omitted `lane` resolves to **A**, and A is not a new context — it is
`server.agent`, with its existing session, history, and channel routing:

```
?lane=A ─┐
(absent) ─┴─→ server.agent          today's agent, unchanged
?lane=B ────→ _lane_agents["B"]     new
?lane=C ────→ _lane_agents["C"]     new
```

This is what makes lanes additive rather than a migration. Every existing
client — WhatsApp, the glasses bridge, botport, cron turns, the inbound queue
consumer, REST callers — keeps hitting exactly the agent and session it always
has, because that agent is now merely *named* A. Opening the FD chat lands on
A and shows the transcript you already have, instead of an empty fourth room.

Three consequences, all deliberate:

- **A is the broadcast lane.** `_broadcast` reaches no-lane and lane-A sockets;
  B and C are served only by their own scoped callbacks.
- **Anything arriving from outside lands in A.** A WhatsApp message or a cron
  turn carries no lane. A is therefore the noisy lane — put long unattended
  runs in B and C.
- **A keeps `server._busy`.** The existing flag keeps its exact meaning; new
  lanes bring their own per-agent flag. A WhatsApp message still collides with
  A's queue exactly as it does today. Lanes don't fix that; they give you two
  rooms where it can't happen.

The rejected alternative — A as a fresh context with the legacy agent hidden
behind it — orphans the current session and streams every external channel
into a transcript nobody is watching.

### 3.3 Shared vs per-lane

| Per lane | Shared across lanes |
|---|---|
| CC session + transcript | The datastore file (**the point** — one `fund_portfolio`) |
| Queue + auto-mode | Memory DB, insights, workspace files |
| Busy flag, model override, planning state | The tool registry (see risk R1) |
| WS sockets and streaming callbacks | LLM provider + credentials |

### 3.4 Naming

Lane id is a bare letter (`A`/`B`/`C`), the session is `lane-<id>` (A excepted
— it keeps the agent's existing session), and the tab label is
`<id> - <agent name>` — `A - Deep Researcher`. Lane ids are fixed at three to
start; the code should not assume three.

---

## 4. Phases

Each phase ends somewhere shippable.

### Phase 1 — Backend: lanes exist (no UI)

1. `WebServer.__init__`: add `_lane_agents: dict[str, Agent]`,
   `_lane_locks`, `_lane_sockets: dict[str, set[WebSocketResponse]]`.
2. `_get_lane_agent(lane)` — generalize `_get_public_agent`. Same lazy
   creation, same lock, same per-lane sender closure; the only difference is
   the session it loads (`lane-<id>`, created on first use) and that it does
   not require public auth.
3. `ws_handler`: read `?lane=` from the query string, stamp `ws._lane`,
   register the socket in `_lane_sockets[lane]`, unregister on close.
4. `chat_handler`: where it picks `agent` and checks busy, add the lane branch
   *before* the admin branch — lane agents check `agent._busy`, not
   `server._busy`.
5. `_broadcast`: skip sockets that carry a `_lane` (they are served by their
   lane's own callbacks), exactly as it already skips public sockets.

**Verify:** two `websocat` connections with `?lane=A` and `?lane=B`, one long
prompt each, sent within a second of each other. Both stream. Neither sees the
other's tokens. A third connection with no lane still talks to the main agent.

### Phase 2 — FD proxy passes the lane through

`/fd/agent-ws/{host}/{port}` forwards a `lane` query param to the agent URL
([server.py:2696](../captain_claw/flight_deck/server.py)). Four lines. The
proxy is already a dumb passthrough.

### Phase 3 — Frontend: the store learns lanes

The store keys everything by `containerId` today. Introduce a **session key**
`` `${containerId}::${lane}` `` and key `sessions`, `_queueLSKey`, and
`savePlanSlice` by it. Lane `''` keeps today's key format so existing
localStorage queues survive the upgrade.

- `openChat(id, name, host, port, auth, lane?)` — one more argument, one WS
  per lane, connected lazily when the lane is first selected (do not open
  three sockets for a user who only uses A).
- `activeLane: string` in the store + `setActiveLane(containerId, lane)`.
- Every action already takes `containerId` — they take the session key
  instead. Mechanical, but it is the bulk of the diff.

### Phase 4 — Frontend: the tabs

A tab strip above the chat pane: `A - Deep Researcher` · `B - …` · `C - …`,
each with a dot showing lane state (idle / running / awaiting answer) and its
pending count. Clicking swaps the chat *and* the queue panel to that lane.

Unread marker: a lane that produced output while you were looking elsewhere
gets a dot. Without it, parallel lanes are worse than serial ones — work
finishes unseen.

### Phase 5 — Lane-aware queue dispatch

`tryDispatchNext` becomes per-lane (it already reads one session's state, so
this mostly falls out of Phase 3). Add: a lane never dispatches while *its*
agent is busy, and lanes do not wait on each other.

---

## 5. Risks

**R1 — The tool registry is a process-wide singleton.** `agent.tools =
get_tool_registry()` returns one global instance ([registry.py:743](../captain_claw/tools/registry.py)),
so every lane shares it. Stateless tools are fine. Stateful ones are not: the
browser tool's page, shell cwd, any tool holding a handle. Two lanes driving
the browser at once will interleave and produce nonsense.

*Note this is already true of public mode today* — lanes do not introduce it,
they make it reachable. Mitigation, in order of cost: (a) document it and keep
lanes to research/datastore work; (b) an asyncio lock per stateful tool, so
lane B's browser call waits for lane A's; (c) per-lane tool instances for the
stateful few. Recommend (b) — a lock in the browser tool's entry point, no
registry surgery.

**R2 — Concurrent SQLite writes.** Three lanes upserting one datastore file.
SQLite in WAL handles concurrent writers with retries, but the datastore
manager should be checked for a busy timeout. Practically: give each lane a
disjoint ID range and they never touch the same rows.

**R3 — Memory footprint.** Each Agent carries an instruction cache and
per-session state. Three lanes ≈ three times the per-agent overhead in one
process — measure before promising more than three.

**R4 — Rate limits.** Three concurrent streams against one provider key. The
enrichment workload is web-search heavy, so this will show up as provider
429s, not as a Captain Claw error. Surface it per lane.

**R5 — `_inbound_queue_consumer` assumes one lane.** It picks the first
non-public client and waits on `server._busy`
([web_server.py:1294](../captain_claw/web_server.py)). WhatsApp/glasses
messages should keep going to the main agent — verify lanes do not steal them.

---

## 6. The alternative, and why not

**One process per lane** (FD spawns three agents, the UI hides them behind
three tabs) gives perfect isolation and needs almost no backend work — the
whole plan collapses into Phase 4.

It is rejected for *this* use case because each spawned agent gets its own
datastore under `fd-data/<agent>/…/datastore.db`, and the entire point is
three lanes enriching **one** `fund_portfolio`. Pointing three processes at
one SQLite file across process boundaries is a worse concurrency story than
three Agents in one event loop, not a better one. Memory cost is also ~3×
higher.

Worth revisiting if lanes ever need different models or different tool
policies — that is where separate processes win.

---

## 7. Estimate

| Phase | Size |
|---|---|
| 1 — backend lanes | M — generalizing existing code, but touches the busy/broadcast paths |
| 2 — proxy param | XS |
| 3 — store re-keying | M — wide but mechanical |
| 4 — tabs | S |
| 5 — per-lane dispatch | S |
| R1 tool lock | S |

Phases 1+2 are independently valuable: they make the agent genuinely
multi-session over the wire, which the API and any other client can use.
