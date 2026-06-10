# Captain Claw v0.4.10 Release Notes

## Agent Forge — AI-Powered Team Decomposition

The headline feature of v0.4.10 is **Agent Forge**, a new Flight Deck page that turns a text description of your goal into a working team of AI agents.

### How it works

1. Describe your business objective in plain language
2. Agent Forge calls an LLM (user-configured provider/model/API key) to decompose it into a team of specialized agents
3. Review the proposed team — edit names, roles, instructions, tools, and models
4. Click Spawn — agents are created with fleet-level instructions, group tags, and a lead coordinator

### What makes it powerful

- **Full tool awareness** — The decomposition prompt includes all 44 Captain Claw tools with descriptions, so the LLM makes informed tool selections per role
- **Standard Operating Procedures** — Each agent's fleet instructions include a pseudo-code SOP describing how to approach its work
- **Lead agent** — One agent is designated as the team coordinator with routing guidelines for the entire team
- **Per-agent LLM config** — Each agent can use a different provider and model
- **Additional API keys** — Set environment variables (e.g., BRAVE_API_KEY) that are passed to all spawned agents
- **Group organization** — Agents are auto-tagged with team and role groups

## Fleet-Level Instructions

A new system for pushing operating instructions to agents from Flight Deck without restarting them.

- New **Instructions** tab in the agent config editor (alongside config.yaml and .env)
- Instructions are injected into the agent's system prompt under "Fleet-Level Instructions"
- Take effect immediately when the agent's chat session is opened
- Works in all card view modes (expanded, compact, icon)

## Datastore Browser

View agent datastore tables and rows directly from Flight Deck.

- New **Data** button on agent cards (next to Files and Logs)
- Lists all tables with schemas, column types, and row counts
- Browse rows with pagination in an 80% viewport modal
- Available on both ProcessCard and ContainerCard

## UI Improvements

- **Agent card layout** — Chat, Open, and status pill moved to the PID/port line in expanded view for a cleaner layout
- **Configure modal fix** — Config modal now renders correctly in compact and icon card modes (was only visible in expanded mode)
- **Auto port assignment** — Spawning agents with port 0 now auto-assigns an available port starting from 24080
- **Desktop layout fix** — All pages (including Forge) now render correctly in desktop layout mode

## Technical Details

### New files
- `captain_claw/instructions/forge_decompose_system_prompt.md` — System prompt for team decomposition
- `flight-deck/src/pages/ForgePage.tsx` — Agent Forge page component
- `flight-deck/src/components/agents/DatastoreBrowser.tsx` — Datastore browser modal

### New endpoints
- `POST /fd/forge` — LLM-powered team decomposition
- `GET /fd/agent-datastore/{host}/{port}/tables` — List agent datastore tables
- `GET /fd/agent-datastore/{host}/{port}/tables/{name}/rows` — Query table rows

### Modified files
- `captain_claw/flight_deck/server.py` — Forge endpoint, datastore proxy, auto port assignment
- `captain_claw/web/ws_handler.py` — Fleet instructions delivery via peer_agents message
- `captain_claw/agent_context_mixin.py` — Fleet instructions injection into system prompt
- `captain_claw/instructions/system_prompt.md` — Fleet instructions placeholder
- `flight-deck/src/stores/containerStore.ts` — Fleet instructions storage
- `flight-deck/src/stores/processStore.ts` — Fleet instructions storage
- `flight-deck/src/stores/chatStore.ts` — Fleet instructions in peer_agents message
- `flight-deck/src/components/agents/AgentConfigEditor.tsx` — Instructions tab
- `flight-deck/src/components/agents/ProcessCard.tsx` — Data button, layout changes, config modal fix
- `flight-deck/src/components/agents/ContainerCard.tsx` — Data button, config modal fix
- `flight-deck/src/components/layout/Sidebar.tsx` — Agent Forge nav item
- `flight-deck/src/types/index.ts` — Forge view mode
- `flight-deck/src/App.tsx` — Forge page routing
