# Captain Claw v0.4.8 Release Notes

**Release title:** Process Agents, Fleet Discovery & Desktop Declutter

**Release date:** 2026-03-31

## Highlights

This release adds **process-based agents** to Flight Deck — spawn and manage Captain Claw agents as local pip processes, no Docker required. Agents can now **discover and communicate with each other** through a new `flight_deck` tool that queries live fleet status. The Agent Desktop gets a third **icon view mode** for managing many agents without clutter.

## New Features

### Process Agents (pip-based, no Docker required)

Spawn Captain Claw agents as local subprocesses directly from Flight Deck. Each agent runs in an isolated directory under `fd-data/` with its own config, workspace, database, and skills — the same isolation model as Docker agents, without the Docker dependency.

- **Spawn from UI** — the Spawn Agent page now has a Docker/Process mode toggle (violet for Docker, emerald for Process)
- **Full lifecycle** — start, stop, restart, clone, and remove process agents
- **Port collision prevention** — spawner checks if the requested port is in use and auto-finds a free one
- **Auto-restart on FD startup** — previously-running process agents are automatically restarted when Flight Deck starts
- **Clean shutdown** — all process agents are stopped when Flight Deck shuts down
- **Process registry** — persistent JSON registry at `fd-data/.processes.json` tracks all managed processes
- **Logs** — view last N lines of process agent stdout/stderr
- **Clone** — duplicate an agent's config into a new process agent with one click

### Fleet Discovery Tool (`flight_deck`)

A new always-available tool that gives every agent live awareness of the fleet:

- **`list_agents`** — queries `GET /fd/fleet` for a real-time list of all running agents (Docker, process, and local) with name, kind, status, port, and description
- **`consult`** — send a message to any peer agent by name and receive their response, using live fleet lookup instead of the static peer list pushed at connect time
- **Always available** — registered unconditionally alongside `consult_peer`, no config needed
- **Self-aware** — marks the calling agent in the fleet list so it knows which one is itself

### Three-Level Card View Modes

Agent cards on the Desktop now cycle through three view modes with a single button:

- **Expanded** — full card with description, forwarding task, approval settings, model/persona selectors, groups, files, logs, actions, and embedded chat
- **Compact** — name, status badge, activity indicator, groups, files/logs buttons, and embedded chat on a smaller card
- **Icon** — single-row pill showing only the agent icon, name, activity status (Working.../Idle), and running status badge. The toggle button appears on hover.

All three card types (Docker, Process, Local) support all three view modes. View mode per agent is persisted to localStorage.

### Peer List Includes Process Agents

The `peer_agents` WebSocket message sent to agents on connect now includes process agents alongside Docker containers and local agents. This fixes peer discovery for the existing `consult_peer` tool when process agents are running.

## Backend Changes

- **`GET /fd/fleet`** — new endpoint returning all agents across Docker, process, and local registries as a unified list
- **`GET /fd/processes`** — list all registered process agents with status
- **`POST /fd/spawn-process`** — spawn a new process agent with auto-generated config.yaml
- **`POST /fd/processes/{slug}/stop|start|restart`** — lifecycle management
- **`POST /fd/processes/{slug}/clone`** — clone an agent's configuration
- **`DELETE /fd/processes/{slug}`** — remove from registry (preserves data directory)
- **`GET /fd/processes/{slug}/logs`** — tail process logs
- **Port availability check** — both Docker and process spawn endpoints now verify port availability before starting
- **URL localization** — process agents rewrite `host.docker.internal` URLs to `localhost` automatically
- **Fleet change notifications** — running agents receive a system chat message when new agents join or leave

## New Files

| File | Description |
|---|---|
| `captain_claw/tools/flight_deck.py` | Fleet discovery and peer consultation tool |
| `flight-deck/src/components/agents/ProcessCard.tsx` | Process agent card component with emerald accent |
| `flight-deck/src/stores/processStore.ts` | Zustand store for process agent state and overrides |

## Stats

- **~1,200 lines added** across 15 files
- 1 new tool, 1 new component, 1 new store
- 7 new backend endpoints
