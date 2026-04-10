# Captain Claw Game — Design Document

**Status:** Draft v0.1
**Date:** 2026-04-09
**Author:** design conversation between user + Claude

---

## 1. Vision

Captain Claw Game (CCG) is a **multiplayer text-adventure framework** that runs inside Captain Claw. Players are Captain Claw agents and/or humans, sharing a turn-based world rendered in ASCII. A user describes the game in natural language ("a haunted lighthouse, three rangers must find the keeper before dawn") and the framework instantiates a complete, playable world from that description.

The framework itself is the Game Master. There is no separate GM agent. **The engine is deterministic; the content is probabilistic.** Same seed + same actions = same outcome, byte for byte. But world generation, narration, and NPC behavior are LLM-driven and creative.

---

## 2. Design Principles

1. **Engine is deterministic, content is creative.** All state mutations go through a pure rules engine. LLM calls produce *content* (descriptions, names, NPC dialogue) which is then frozen into deterministic state.
2. **No GM agent.** The framework arbitrates. Players cannot lie to the rules; only to each other.
3. **Tick-based, never real-time.** A turn closes when all required intents are submitted (or a wall-clock budget expires). Spectators can pause/replay.
4. **Fog of war by construction.** Each player gets a *projected view* of the world from their character's perspective. The full world state is never sent to a player agent.
5. **Replayable & forkable.** Every game is `(seed, world_spec, intent_log)`. Replay by re-running. Fork by copying log up to tick N.
6. **Reuse Captain Claw primitives.** Insights = character memory. Reflections = character personality. Council = party deliberation. Agent messaging = transport. Datastore = world tables. Don't reinvent.

---

## 3. Core Concepts

### 3.1 World Spec
A user-authored (or LLM-elaborated) declarative description of the game:
```yaml
title: "The Lighthouse Keeper"
genre: cozy-mystery
seats: 3              # number of player slots
human_seats: 0..3     # how many may be human
goal: "Find the keeper before dawn (tick 30)."
tone: "Gothic, melancholic, slow."
constraints:
  - "No combat."
  - "One real culprit, hidden among NPCs."
  - "Players have asymmetric clues."
size: small           # small | medium | large (controls room count, NPC count, item count)
```
The spec is the **only** thing the user must provide. Everything else is generated.

### 3.2 World
The instantiated world after generation. Frozen, deterministic, addressable:
```
World
├── rooms[]            # graph of locations
├── entities[]         # NPCs, items, props
├── facts[]            # truth table — who did what, what's behind the door
├── clues[]            # discoverable subset of facts, with conditions
├── characters[]       # player-controllable bodies
└── flags{}            # world-level toggles (dawn_broken, door_open, etc.)
```

### 3.3 Tick
The atomic unit of game time. Each tick:
1. Engine emits *per-player views* (fog of war).
2. Each player submits an **intent** (structured action).
3. Engine resolves all intents in deterministic order.
4. Engine writes the new state, narration, and log entry.
5. Goto 1.

### 3.4 Intent
A structured action proposed by a player. The engine validates and resolves:
```json
{ "actor": "char_2", "verb": "examine", "target": "ent_lantern", "args": {} }
```
Free-text from human players is parsed into intents by an *input adapter*, not the engine.

### 3.5 View
What a single player sees this tick. Includes:
- ASCII map (visible rooms only)
- Character sheet (own HP/inventory/objectives)
- Local narration (what *I* perceived)
- Visible entities & exits
- Available verbs (closed verb set — see §6)

The view is the **only** input the player agent receives. No world state leaks.

---

## 4. Architecture

```
captain_claw/
└── games/
    ├── __init__.py
    ├── spec.py              # WorldSpec dataclass + YAML loader
    ├── generator.py         # LLM-driven world generation from spec
    ├── world.py             # World, Room, Entity, Character dataclasses (frozen)
    ├── engine.py            # Pure rules engine: (state, [intent]) -> state'
    ├── tick.py              # Tick loop, intent collection, ordering
    ├── view.py              # Per-player projection (fog of war)
    ├── render.py            # ASCII map + sidebar rendering
    ├── intent.py            # Intent schema, validation, verb registry
    ├── resolver.py          # Verb handlers (move, take, examine, talk, ...)
    ├── narrator.py          # LLM call: state diff -> prose narration
    ├── log.py               # Append-only intent + resolution log
    ├── replay.py            # Re-run from (seed, spec, log)
    ├── seats.py             # Player seat abstraction (agent | human)
    ├── adapters/
    │   ├── agent_seat.py    # Captain Claw agent as player
    │   └── human_seat.py    # CLI / web input as player
    └── games_cli.py         # `cc game new|run|replay|spectate`
```

### 4.1 Determinism boundary
```
┌────────────────── DETERMINISTIC ──────────────────┐
│  engine.py  resolver.py  view.py  log.py  tick.py │
└────────────────────────┬──────────────────────────┘
                         │ produces inputs only
┌────────────────────────┴──────────────────────────┐
│         LLM-DRIVEN (generator, narrator)          │
│  - Output is captured & frozen into state         │
│  - Re-runs replay frozen output, never re-call    │
└───────────────────────────────────────────────────┘
```

LLM output is **frozen on first generation** and stored in the log. Replays read from the log and never re-call the LLM. This is what makes the framework deterministic *with* probabilistic content.

---

## 5. World Generation Pipeline

Input: `WorldSpec`. Output: a fully-instantiated, frozen `World` plus a generation transcript.

Stages (each is an LLM call with strict JSON schema output):

1. **Premise expansion** — turn the one-line goal into a 3-act outline + secret truth.
2. **Room graph** — N rooms, exits, tags. Validated as a connected graph.
3. **Cast** — NPCs with names, descriptions, secret motives. One culprit if mystery.
4. **Inventory** — items distributed across rooms; some are clues.
5. **Clue web** — for each fact in the secret truth, pick discoverability conditions (which item, in which room, examined by which character class).
6. **Character sheets** — N player characters (matching `seats`), each with a *private objective* and *asymmetric starting knowledge*.
7. **ASCII art** — per-room ASCII tiles, composed into a global map. Generator outputs character grids; renderer composes views.
8. **Validation** — engine runs the world through a *solvability checker*: from start state, is there at least one intent sequence that reaches the goal? If not, regenerate failed stage with feedback.

The transcript of every LLM call is stored alongside the world so generation is fully auditable and re-runnable.

---

## 6. Verb Set (initial)

Closed, small, grows by version. Engine knows all of these; anything else is rejected.

| Verb     | Args                   | Notes                                  |
|----------|------------------------|----------------------------------------|
| `move`   | direction or room_id   | Validates exit                         |
| `look`   | —                      | Re-emits current view                  |
| `examine`| entity_id              | Reveals clues if conditions met        |
| `take`   | entity_id              | Inventory check                        |
| `drop`   | entity_id              | —                                      |
| `give`   | entity_id, char_id     | Same room                              |
| `talk`   | char_id, topic         | LLM narration, frozen into log         |
| `use`    | entity_id, target_id   | Resolver consults entity rules         |
| `wait`   | —                      | Skip                                   |
| `say`    | text, audience         | In-fiction speech (visible to others)  |
| `note`   | text                   | Private to the actor                   |

Free verbs (unknown text) from humans go through the input adapter, which **must** map to one of the above or return an error to the player. The engine never sees free text.

---

## 7. Player Seats

A seat is *who* controls a character. Seats are interchangeable mid-game.

- **Agent seat** — a Captain Claw agent receives the view as a structured prompt and returns a single intent JSON. Uses existing agent messaging. The agent's per-game `insights` DB *is* the character's memory. The agent's reflections file *is* the character's personality.
- **Human seat** — view rendered to terminal/web; input parsed via the adapter.
- **Open seat** — engine substitutes a "default behavior" intent (`wait` or LLM-suggested) until a seat is filled.

Hot-swap rules: any seat can be reassigned between ticks without state changes. A human can take over an agent character; an agent can take over an absent human.

### 7.1 Party mode vs Solo mode

A single agent can occupy seats in two distinct ways. **Both are first-class.**

| Mode       | One agent, one seat                           | One agent, many seats                                |
|------------|-----------------------------------------------|------------------------------------------------------|
| **Party**  | Agent A → char_1, Agent B → char_2 (etc.)     | Each character is a separate Captain Claw agent. Agents only know what their character knows. They may form a *council* between ticks to deliberate, but each emits an independent intent. This is where hidden info, asymmetric objectives, and possible betrayal live. |
| **Solo**   | Agent A controls char_1, char_2, char_3 alone | One agent runs the whole party. It receives **one merged view per tick** (union of its characters' projections, clearly labeled per character) and submits **one intent per character**. Useful for single-agent puzzle/exploration play and for testing. |

The mode is set in `WorldSpec`:
```yaml
seats: 3
seat_mode: party     # party | solo | mixed
```
- `party` — N agents, N characters, 1:1.
- `solo`  — 1 agent, N characters, 1:N.
- `mixed` — explicit per-seat assignment (`seats.json`); some agents run multiple characters, others run one. Humans always count as `party` (one human, one character).

The engine doesn't care which mode is in effect — it always asks "give me an intent for character X" through the seat abstraction. The difference is purely in `seats.py`: a solo seat batches its character's views into a single prompt to its single agent, then unpacks the response into per-character intents. Cheat detection runs **per character**, not per agent — a solo agent legitimately knows what all its characters know, but each character's projection is still computed independently so the cheat audit (§14.5) remains meaningful for party play.

---

## 8. Tick Loop (pseudocode)

```python
def run_game(world, seats, log, seed):
    rng = Rng(seed)
    state = world.initial_state
    while not state.terminal:
        views = {c.id: project_view(state, c) for c in state.characters}
        intents = collect_intents(seats, views)         # async, with timeout
        intents = order_intents(intents, rng)           # stable: by char_id then submit time
        new_state, events = engine.resolve(state, intents, rng)
        narrations = narrator.render(state, new_state, events)  # frozen into log
        log.append(state.tick, intents, events, narrations)
        state = new_state
    return state, log
```

`engine.resolve` is **pure**: same `(state, intents, rng)` always yields the same `(new_state, events)`. The narrator is impure but its output is captured into the log on first run; replay reads from the log.

---

## 9. Fog of War (`project_view`)

Given `state` and `character c`, return only:
- Rooms `c` has *ever* visited (with last-known contents, not live).
- Current room: live contents.
- Entities `c` is currently observing.
- `c`'s own inventory, sheet, private notes.
- Public say-events `c` heard (same room or adjacent depending on rules).
- `c`'s objective (private; differs from other players').

The projection function is the security boundary. It must never return a reference to the full state. Unit tests assert that no field of `World` leaks through.

---

## 10. Narration

Two channels per tick:

- **Machine channel** (for agent players): structured JSON view. Compact, stable schema.
- **Human channel** (for spectators and human players): ASCII map + prose narration generated by `narrator.py`. The narrator is given the *state diff*, not the full state, and is told *who* it's narrating for. Output is frozen into the log.

The narrator never invents facts. It can only describe events the engine emitted. This is enforced by a post-generation check: any noun in the narration that is not present in the events list flags the narration for regeneration.

---

## 11. Persistence & Replay

Each game lives at `~/.captain-claw/games/<game_id>/`:
```
spec.yaml             # original user spec
world.json            # frozen generated world
generation.log        # all LLM calls during generation
intents.log           # append-only tick log
narrations.log        # frozen LLM narration per tick
seats.json            # current seat assignments
seed                  # the RNG seed
```

To **replay**: load spec + world + intents.log, re-run engine; narrations come from narrations.log (no LLM calls).
To **fork**: copy the directory, truncate intents.log to tick N, resume.
To **spectate**: tail the logs, render to terminal/web.

---

## 12. Access Surface — Web Only

**There is no CLI for CCG.** Game state lives inside each Captain Claw
agent's web server, and Flight Deck is the only client. This keeps the
"games as observable agent activity" framing first-class and avoids a
parallel CLI codepath that would drift.

### 12.1 Agent REST API (`captain_claw/web/rest_games.py`)

Mounted on the agent's existing aiohttp web server:

```
GET    /api/games/worlds                 → list available worlds
GET    /api/games                        → list active game sessions
POST   /api/games                        → create a game from a world spec
                                           body: { world_id, seats, seed? }
GET    /api/games/{game_id}              → full state + per-character views + ASCII
POST   /api/games/{game_id}/tick         → advance one tick
POST   /api/games/{game_id}/intent       → queue a human intent
                                           body: { actor, verb, args }
POST   /api/games/{game_id}/replay       → re-run intent log, return determinism check
DELETE /api/games/{game_id}              → drop the in-memory session
```

### 12.2 Flight Deck proxy (`/fd/agent-games/{host}/{port}/...`)

Flight Deck never talks to the engine directly. It uses the same
per-agent proxy pattern as `agent-skills` and `agent-reflections`, so
auth, multi-agent selection, and tenant isolation come for free.

### 12.3 Flight Deck UI (`pages/GamesPage.tsx`)

A new top-level "Games" tab in the sidebar. The page lets the user:
- Pick which connected agent to host the game on.
- Pick a world from the agent's world catalog.
- Assign each character's seat (`scripted` | `human`).
- Create a game.
- Step the tick manually with a Tick button (M0). Auto-tick comes in M3.
- Read each character's per-view ASCII pane (fog of war).
- For human seats, type intents into a small input box (`move north`,
  `take brass_key`, `say hello`) which queue for the next tick.
- Run a Replay button that re-runs the intent log and reports whether
  the replayed final state is byte-identical to the live state — this
  is the determinism receipt.

Future M2/M4 milestones add a spectator mode (tail of the log over WS),
fork-at-tick, and a richer prose / narration channel.

---

## 13. Integration with Captain Claw

| CC primitive          | Game role                                              |
|-----------------------|--------------------------------------------------------|
| Agent messaging       | Seat ↔ engine transport                                |
| Insights DB           | Per-character long-term memory                         |
| Reflections           | Per-character personality / playstyle                  |
| Datastore             | Optional: world tables for large worlds                |
| Council framework     | Party deliberation between ticks (advisory, not binding) |
| Skills                | Verb resolvers can call skills (e.g. `pdf` to read an in-game book) |
| Sister session        | Spectator UI                                            |
| Cron                  | Slow-burn games that tick once per hour                |

Per the project memory: in `public_run == "computer"` mode, every game's insights DB is per-session — characters from different tenants must not share memory.

---

## 14. Decisions & Open Questions

**Decided:**

1. **Council vs intent**: agents always submit **independent** intents. A pre-tick council is *advisory only* — it can shape what each agent decides, but each agent still emits its own intent. This preserves asymmetric information and lets traitors / hidden objectives actually function.
2. **Combat**: **not in v1.** Cozy / mystery / exploration first. Reconsider after M4.
3. **Inter-game memory**: **default no**, but **opt-in supported.** A `WorldSpec` field `carry_reflections: [agent_id, ...]` (or `all`) lets a returning agent bring its prior reflections file into the new game. Off by default so most games start clean.
4. **NPC agency**: **v1 scripted** reactions with constrained LLM dialogue inside the `talk` verb. **v2 mini-agents** — NPCs become lightweight Captain Claw agents with their own intent slots, scheduled by the engine. The verb set and projection model already accommodate this; it's a `seats.py` extension.
5. **Cheat detection**: **yes, audit & log.** When an agent's intent references an entity, room, or fact that its most recent projected view did not contain, the engine flags the intent (`flag: leaked_knowledge`) and writes it to a `cheats.log`. The intent still resolves (we don't want the engine fighting the agent at runtime), but flags are surfaced in the spectator UI and the post-game report.

**Still open:**

6. **World size limits**: how many rooms before view projection is too slow? Benchmark in M0.

---

## 15. Milestones

**M0 — Spine (no LLM) — DELIVERED**
Hardcoded 3-room "Lighthouse" world (`captain_claw/games/demo_world.py`),
two characters, scripted + human seat kinds, verb set: `wait`, `look`,
`move`, `take`, `drop`, `say`. Tick loop, fog-of-war view projection,
ASCII render, append-only intent log, deterministic replay. End-to-end
proven via `lighthouse_demo` running to win-state and a replay step
that produces a byte-identical final state. Reachable from Flight Deck
via the new Games tab.

**M1 — Generator**
WorldSpec → generated world. One genre (cozy mystery). Solvability check. Generation transcript stored.

**M2 — Narrator + Human seat**
Prose channel. CLI human seat. Fog-of-war hardening (tests).

**M3 — Council & richer verbs**
`talk`, `use`, `give`. Party deliberation hooks. Insights = character memory wired up.

**M4 — Web spectator + replay UI**
Fork/replay via web. Multi-game dashboard.

**M5 — Slow-burn cron games**
Tick-per-hour mode for ambient async games across multiple humans + agents.

---

## 16. Appendix: Why this design is "deterministic framework, probabilistic content"

The engine is a pure function. The LLM is used **only** at:
- World generation (output frozen into `world.json`)
- Narration (output frozen into `narrations.log`)
- NPC dialogue inside `talk` verb (output frozen into `intents.log` events)
- Optional: human input parsing (output is a validated intent — if validation fails, retry)

Once frozen, every artifact is deterministic. Re-running a game with the same `(seed, spec, intents.log)` produces byte-identical state transitions, because all the creative output is read from disk, not regenerated. This gives us the best of both worlds: stories feel alive on first play, and games are perfectly debuggable, replayable, and forkable forever after.
