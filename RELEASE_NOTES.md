# Captain Claw v0.4.7 Release Notes

**Release title:** Flight Deck Upgrade — Director, Operations, Pipelines & Collaboration

**Release date:** 2026-03-29

## Highlights

This release is a major upgrade to **Flight Deck**, the multi-agent management dashboard. The Agent Desktop now supports free-form draggable layouts, embedded chat on agent cards, and resizable panels. New views include a **Director panel** for unified agent oversight with broadcast, an **Operations dashboard** for token usage and cost analytics, and **Agent Pipelines** for chaining agent outputs automatically. Collaboration features — pinned messages, shared clipboard, notification center, keyboard shortcuts, and dark/light theme — round out the experience.

## New Features

### Director Panel

A collapsible left panel (Cmd+D) providing unified oversight of all agents:

- **Agents tab** — all Docker and local agents listed with real-time status, current activity text, last interaction time
- **Activity Feed tab** — chronological feed of agent events
- **Broadcast** — send a message to all running agents simultaneously
- **Filter & sort** — filter by status (running/stopped/error), sort by name, status, last active, or type
- **Expandable rows** — click to reveal description, host info, creation time, and recent message previews
- **Quick actions** — Stop All, Restart All buttons
- **Resizable** — drag right edge to resize (220–500px, persisted to localStorage)

### Operations Dashboard

A dedicated analytics view (Cmd+2) for monitoring agent usage and costs:

- **Summary cards** — total tokens, estimated cost (USD), API calls, average latency, data transferred, cache hit rate
- **Token distribution bar** — visual breakdown of input, output, and cached tokens
- **Per-agent usage table** — sortable columns for tokens, cost, calls, latency per agent
- **Model breakdown table** — usage grouped by LLM model across all agents
- **Agent health grid** — status cards for all connected agents
- **Period filter** — last hour, today, yesterday, this week, this month, all time
- **Cost estimation** — uses published pricing for Claude, GPT, and Gemini models
- **Backend proxy** — new `/fd/agent-usage/{host}/{port}` endpoint proxies agent usage APIs with auth cookie handling

### Agent Pipelines

Chain agents together so output from one automatically flows to the next. Moved to the Workflows view (Cmd+3) with a redesigned visual builder:

- **Visual pipeline cards** — large rounded cards with status badges, enable/disable toggle, mini flow preview
- **Vertical flow diagram** — expanded view shows numbered step cards with gradient arrow connectors
- **Step editor** — add agents from a selector, set optional prompt prefixes, remove steps
- **Contextual forwarding** — forwarded messages include pipeline name, source agent name, and instructions for the receiving agent to process based on its playbooks, instructions, and persona
- **Auto-trigger** — subscribes to Zustand store changes; when an agent at step N responds, step N+1 receives the output automatically
- **Notifications** — pipeline forwarding events appear in the notification center

### Agent Desktop Overhaul

The desktop view now combines Docker and local agents into one unified stage:

- **Free-form layout** — drag agent cards anywhere on the canvas using pointer events (not HTML5 drag); positions persisted to localStorage
- **Grid layout** — traditional card grid as an alternative
- **Layout toggle** — switch between grid and free-form modes
- **Embedded chat** — collapsible chat panel directly on each agent card
- **Agent descriptions** — editable description field on each card
- **Container removal** — removing an agent now also removes its Docker container with a confirmation dialog

### Chat Enhancements

- **Pin messages** — hover any message and click the pin icon (amber) to save with tags
- **Copy to shared clipboard** — click the clipboard icon (cyan) to share snippets across agents
- **File attachments** — attach files and clipboard content to messages
- **Resizable panel** — drag the left edge to resize between 320–900px (persisted)
- **Full context forwarding** — removed the 2000-character truncation limit when sending context between agents
- **Width from parent** — chat panel width now controlled by the App layout, not hardcoded

### Pinned Messages Panel

Pin important chat messages for quick reference:

- Tag system with filtering by tag, agent name, or content
- Expandable message previews with full markdown rendering
- Copy, tag, and unpin actions per message

### Shared Clipboard

Cross-agent clipboard for sharing text between agents:

- Add entries manually or from chat messages
- Pin important items, edit inline, send to any online agent

### Notification Center

Bell icon in the top bar with:

- Unread badge count
- Type filters (info, success, warning, error)
- Auto-notifications for agent connect/disconnect and pipeline forwarding
- Mark read, clear individual or all

### Keyboard Shortcuts

Full keyboard navigation with Cmd/Ctrl modifiers:

| Shortcut | Action |
|---|---|
| Cmd/Ctrl+1–4 | Switch views (Desktop, Operations, Workflows, Spawner) |
| Cmd/Ctrl+D | Toggle Director panel |
| Cmd/Ctrl+J | Toggle Chat panel |
| Cmd/Ctrl+K | Toggle Shortcuts overlay |
| Cmd/Ctrl+[ / ] | Previous / Next chat tab |
| Escape | Close modals and panels |

### Dark/Light Theme

- Sun/Moon toggle in the top bar
- Full light mode with overrides for zinc palette, scrollbars, markdown styles, and chat top bar
- Theme preference persisted to localStorage

### Resizable Panels

All side panels support drag-to-resize with persisted widths:

| Panel | Min | Max | Default |
|---|---|---|---|
| Director (left) | 220px | 500px | 300px |
| Chat (right) | 320px | 900px | 480px |
| Tool panels | 280px | 500px | 340px |

## Backend Changes

- **`/fd/agent-usage/{host}/{port}`** — new proxy endpoint for fetching agent usage statistics, handles auth cookie from 302 redirects
- **Auth handling** — proxy endpoints now follow redirects, capture cookies from 302 responses, and retry with cookies

## New Stores (Zustand)

| Store | Purpose |
|---|---|
| `pinnedStore` | Pinned messages with tags, notes |
| `clipboardStore` | Shared clipboard entries with pin support |
| `pipelineStore` | Pipeline chains with steps and prompt prefixes |
| `notificationStore` | Notifications with types and read state |
| `groupStore` | Agent groups with colors |
| `themeStore` | Dark/light theme with `applyTheme()` |

## New Components

| Component | Description |
|---|---|
| `DirectorPanel` | Unified agent overview with broadcast and activity feed |
| `OperationsPage` | Token usage analytics dashboard |
| `PinnedMessages` | Pinned message viewer with tag filtering |
| `SharedClipboard` | Cross-agent clipboard panel |
| `PipelineBuilder` | Visual pipeline creation (now in WorkflowPage) |
| `NotificationCenter` | Bell dropdown with type filtering |
| `KeyboardShortcuts` | Shortcuts hook + overlay modal |
| `EmbeddedChat` | Inline chat on agent cards |
| `FileViewer` | Syntax-highlighted file viewer |

## Stats

- **~5,900 lines added** across 42 files
- 11 new components, 6 new stores
- 1 new backend endpoint
