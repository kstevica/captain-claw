# Captain Claw v0.4.15 Release Notes

**Release title:** Agent Council — Multi-Agent Deliberation

**Release date:** 2026-04-05

## Highlights

Flight Deck gains **Agent Council**, a new page for structured multi-agent deliberation. Assemble a panel of agents to debate, brainstorm, review, or plan together in turn-based rounds. Agents self-assess their suitability, choose actions (answer, challenge, refine, broaden), and build on each other's contributions across rounds. Sessions include synthesis with voting, per-agent TL;DR summaries, and Markdown minutes export — all re-generable even after conclusion.

## New Features

### Agent Council

A full multi-agent discussion system accessible from the Flight Deck sidebar:

**Session setup:**
- Four session types: Debate, Brainstorm, Review, Planning — each with tailored instructions
- Four verbosity levels: Message (5 sentences), Short (3 paragraphs), Medium (5 paragraphs), Long (10 paragraphs)
- Configurable max rounds (1–20), first speaker selection (or random)
- Multi-agent picker with automatic moderator detection (Old Man)

**Turn-based discussion:**
- Each agent receives full prior discussion context and responds with a suitability score (0–1) and an action: answer, respond, challenge, refine, broaden, or pass
- Action badges and suitability bars displayed on every message
- Full Markdown rendering with GFM support (tables, code blocks, lists)
- Auto-scroll follows new messages when the user is at the bottom

**Two moderation modes:**
- **Round-Robin** — all agents speak each round, sorted by suitability score (highest first)
- **Moderator** — Old Man selects speakers based on suitability scores and discussion flow

**Council memory:**
- Agents receive context from all prior rounds, grouped by round number
- Configurable memory window: 5, 10, 20, 30 rounds, or indefinite — adjustable mid-session
- 30k token safety cap prevents context overflow regardless of setting

**Synthesis & voting:**
- Request synthesis at any point — the moderator (or first agent) produces a summary
- All other agents vote (agree / disagree / abstain) with reasoning
- Results displayed as a proportional summary bar with individual vote cards

**TL;DR panel:**
- Each agent generates a 2-3 sentence personal takeaway from the discussion
- Collapsible panel with regenerate button
- Works during active sessions and after conclusion (agents are temporarily reconnected)

**Minutes export:**
- Export full session as Markdown (.md) with all rounds, action badges, suitability scores, synthesis, votes, and TL;DRs
- Available during active sessions and after conclusion

**Real-time activity log:**
- Timestamped entries for tool usage, speaking status, connections, and system events
- Color-coded by type (tool=amber, speaking=violet, done=emerald, system=cyan, error=red)
- Auto-scrolls during active sessions
- Shows last 3 tool names per agent while speaking

**Agent awareness:**
- Agents know the council type, current round, max rounds, and moderation mode
- Fleet instructions from Flight Deck are relayed to all council agents
- Agents receive the peer list via `peer_agents` on connection

**Cognitive mode & eco mode display:**
- Sidebar shows each agent's current cognitive mode (colored dot) and eco mode (leaf icon)
- Read-only in council — change settings from the Agent Desktop

**Resizable layout:**
- 50/50 default split between discussion panel and sidebar
- Drag handle to resize (min 280px sidebar, max 65% of width)
- Session info, agent status, pinned messages, votes, and activity log in the sidebar

**Persistence:**
- All sessions, messages, votes, and artifacts stored server-side in SQLite
- Four new database tables: `council_sessions`, `council_messages`, `council_votes`, `council_artifacts`
- Full REST API at `/fd/council/*` with auth
- Sessions can be reopened and reviewed after conclusion

**Architecture:**
- Frontend-driven orchestration via Zustand store (matches chatStore pattern)
- Separate WebSocket sessions per agent — messages go through the agent's normal LLM pipeline and persist in agent memory
- Council messages are NOT visible in the FD chat panel (clean separation)
- All instruction prompts externalized to `.md` template files in `instructions/council/`

### Other Improvements

- **Fixed agent response timing** — `_collectResponse` was reading `data.text` but agents send status in `data.status`, causing every speaker turn to wait the full 120-second timeout. Now reads both keys and detects empty status as idle. Agents respond in seconds instead of 2 minutes.

## Files Changed

**New files (21):**
- `captain_claw/flight_deck/council_routes.py` — REST API router (10 endpoints)
- `captain_claw/flight_deck/db.py` — 4 new tables, ~15 new DB methods
- `flight-deck/src/stores/councilStore.ts` — Core orchestration store (~1400 lines)
- `flight-deck/src/pages/CouncilPage.tsx` — Main page with resizable layout
- `flight-deck/src/components/council/CouncilSetup.tsx` — Session creation form
- `flight-deck/src/components/council/CouncilDiscussion.tsx` — Message thread with round headers
- `flight-deck/src/components/council/CouncilMessage.tsx` — Message bubble with badges
- `flight-deck/src/components/council/CouncilControls.tsx` — Input bar and round controls
- `flight-deck/src/components/council/CouncilSidebar.tsx` — Session info, agents, activity log
- `flight-deck/src/components/council/AgentPicker.tsx` — Multi-agent selector
- `flight-deck/src/components/council/VotingPanel.tsx` — Vote summary and cards
- `flight-deck/src/components/council/SynthesisView.tsx` — Synthesis document display
- `flight-deck/src/components/council/TldrPanel.tsx` — Collapsible TL;DR panel
- `flight-deck/src/components/council/SessionCard.tsx` — Session history card
- `flight-deck/src/instructions/council/btw_context.md` — Initial agent context template
- `flight-deck/src/instructions/council/turn_prompt.md` — Per-turn prompt template
- `flight-deck/src/instructions/council/moderator_select.md` — Moderator speaker selection
- `flight-deck/src/instructions/council/synthesis.md` — Synthesis generation template
- `flight-deck/src/instructions/council/vote.md` — Voting prompt template
- `flight-deck/src/instructions/council/suitability_check.md` — Suitability score check
- `flight-deck/src/instructions/council/tldr.md` — TL;DR generation template

**Modified files:**
- `captain_claw/flight_deck/server.py` — Added council router
- `flight-deck/src/App.tsx` — Added CouncilPage render
- `flight-deck/src/types/index.ts` — Added 'council' to ViewMode
- `flight-deck/src/components/layout/Sidebar.tsx` — Added Council nav item
