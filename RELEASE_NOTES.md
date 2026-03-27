# Captain Claw v0.4.6 Release Notes

**Release title:** Flight Deck — Multi-Agent Management Dashboard

**Release date:** 2026-03-27

## Highlights

This release introduces **Flight Deck** — a full-featured management dashboard for running multiple Captain Claw agents simultaneously. Spawn agents in Docker containers, register local or remote instances, chat with multiple agents via WebSocket, browse and transfer files between agents, and forward conversation context with tasks from one agent to another. Flight Deck is now bundled with Captain Claw and launches with a single command: `captain-claw-fd`.

## New Features

### Flight Deck Dashboard

Multi-agent management UI built with React + FastAPI, served from a single process:

- **`captain-claw-fd`** CLI command starts the dashboard on port 25080
- **Docker container management** — spawn agents with full configuration (provider, model, tools, platforms, BotPort, networking), start/stop/restart/remove containers, view logs
- **Local agent management** — register any Captain Claw instance by host:port with optional auth token, probe status, connect for chat
- **Multi-agent chat** — WebSocket-based chat with multiple agents simultaneously via a tabbed interface, proxied through the backend to avoid CORS
- **Markdown rendering** — full GFM support (tables, code blocks, lists, bold/italic) in both user and assistant messages
- **Agent status indicators** — real-time busy/idle status on agent cards with spinner and status text (e.g., "Using web_fetch...")
- **Smart message filtering** — strips CC suggestion prompts, limits tool messages to last 3 per turn, deduplicates echoed user messages

### File Browser & Transfer

Browse and transfer files between agents without manual download/upload:

- **File browser** — click "Files" on any agent card to browse workspace files with size, type, and path info
- **Multi-select** — checkbox selection with select-all support
- **Agent-to-agent transfer** — select files and click a destination agent to transfer; Flight Deck downloads from source and uploads to destination
- **Transfer status** — per-file progress indicators (spinner/checkmark/error) and a completion summary banner

### Context Transfer

Forward conversation history and tasks between agents:

- **Forward button** (↗) in the chat panel tab bar opens the context transfer modal
- **Message slider** — select how many recent user/assistant messages to include (0 to all)
- **Preview** — expandable preview of selected messages before sending
- **Task prompt** — write what the receiving agent should do with the context
- **Destination selection** — send to any other online agent

### Connection Settings UI

Captain Claw's Settings page now shows connection information:

- WebSocket URL, HTTP URL, and auth status
- Copy-to-clipboard buttons for easy sharing
- Auth token, cookie max age, and public run settings exposed in the Web Server section

### macOS Docker Networking Fix

Docker's `--network host` is silently ignored on macOS. Flight Deck now:

- Auto-detects macOS and switches to bridge networking with explicit port mapping
- Uses `host.docker.internal` for Ollama and other host-side services in container configs
- Sets `OLLAMA_BASE_URL` env var and `model.base_url` in config for Ollama providers

### Docker Config Persistence Fix

Resolved an issue where Flight Deck-provided config.yaml was overridden by CC's home directory config inside Docker containers. Config is now written to both `/app/config.yaml` (CWD) and the home-config volume mount.

## Distribution Changes

- **Flight Deck dependencies** (fastapi, uvicorn, docker, websockets) are now included in the standard `pip install captain-claw`
- **New CLI entry point:** `captain-claw-fd` launches the Flight Deck dashboard
- **Package includes built frontend** — `captain_claw/flight_deck/static/` contains the pre-built React app
- **Frontend source** lives in `flight-deck/` for development (`npm run dev` for HMR, `npm run build` to rebuild)
- Flight Deck can also be run as a Python module: `python -m captain_claw.flight_deck`

## Architecture

```
captain_claw/
  flight_deck/           # Python package
    __init__.py
    __main__.py          # python -m captain_claw.flight_deck
    server.py            # FastAPI backend + static file serving
    static/              # Built React frontend (bundled)
flight-deck/             # Frontend source (React/TypeScript/Vite)
  src/
  package.json
  vite.config.ts
```

## API Endpoints (Flight Deck Backend)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/fd/containers` | List managed Docker containers |
| POST | `/fd/spawn` | Spawn a new agent container |
| POST | `/fd/containers/{id}/stop\|start\|restart` | Container lifecycle |
| DELETE | `/fd/containers/{id}` | Remove container |
| GET | `/fd/containers/{id}/logs` | Container logs (supports streaming) |
| WS | `/fd/agent-ws/{host}/{port}` | WebSocket proxy to agent |
| GET | `/fd/probe` | Probe agent availability |
| GET | `/fd/agent-files/{host}/{port}` | List agent files (CORS proxy) |
| POST | `/fd/transfer` | Transfer file between agents |
| GET | `/fd/health` | Docker health check |
