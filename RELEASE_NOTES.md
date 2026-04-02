# Captain Claw v0.4.9 Release Notes

**Release title:** Multi-User Flight Deck, Agent Delegation & Docker Deployment

**Release date:** 2026-04-02

## Highlights

Flight Deck gains **multi-user authentication** with JWT, an **admin dashboard** for managing users and plan tiers, and **server-side chat persistence**. Agents can now **delegate tasks** to each other with fire-and-forget semantics — the calling agent sends work and moves on, while the peer delivers results back automatically. Flight Deck also gets its own **Dockerfile and Docker Compose** for standalone deployment, plus an **agent config editor** for modifying agent settings in-flight.

## New Features

### Multi-User Authentication

Flight Deck now supports multi-user deployment with JWT-based authentication:

- **Registration & login** — first user becomes admin, subsequent users get the `user` role
- **JWT tokens** — 15-minute access tokens with 7-day refresh token rotation via HttpOnly cookies
- **WebSocket auth** — agent connections authenticated via token query parameter
- **Login page** — sign in / register UI with error handling
- **Profile management** — update display name and password via `/fd/auth/me`
- Enable with `FD_AUTH_ENABLED=true` environment variable

### Admin Dashboard

Full admin panel for managing the Flight Deck deployment:

- **User management** — list, search, edit roles (admin/user), delete users
- **Plan tiers** — free/pro/enterprise with configurable limits per tier
- **Per-user quotas** — override agent count, storage, rate limits for individual users
- **Usage analytics** — event-based logging with type filtering and visualization
- **System config** — enable/disable Docker spawning from the admin UI

### Rate Limiting & Quotas

Tiered plan system with per-user enforcement:

- **Free tier** — 2 agents, 500MB storage, 60 requests/min, 5 spawns/hour
- **Pro/Enterprise** — configurable higher limits
- **Sliding-window** rate limiting for requests and spawn operations
- **Agent count limits** with real-time quota tracking
- Admin-configurable plan limits persisted to database

### Fire-and-Forget Task Delegation

Agents can now delegate tasks to peers without blocking:

- **`delegate` action** on the `flight_deck` tool — sends a task to a peer and returns immediately
- **Background execution** — the peer agent works independently on the delegated task
- **Automatic result delivery** — results are sent back to the source agent as a notification
- **Busy-aware queuing** — if the source agent is busy, results are queued and processed when it becomes free
- **`/fd/delegate-peer` endpoint** — two-phase background task: send to target, then deliver result to source
- Strong task references prevent garbage collection of in-flight delegations

### Consultation Improvements

- **Heartbeat streaming** — 15-second heartbeat events during peer consultation so the requesting agent knows work is happening (replaces the hard 30s timeout that caused premature failures)
- **Deduplication** — prevents hammering a peer with duplicate consultation requests
- **Notification message type** — fleet notifications no longer block the agent (previously set `_busy=True` and triggered full LLM responses)

### Chat Persistence

Server-side chat session and message storage:

- **Per-user isolation** — chat sessions and messages scoped by user ownership
- **Session management** — create, list, load, and delete chat sessions
- **Message history** — up to 500 messages per session with metadata (tool calls, model info, peer interactions)
- **Debounced batching** — message persistence batched for performance

### Settings Sync

Per-user settings persisted server-side:

- **Automatic migration** — localStorage settings migrated to server on login
- **Hydration** — settings loaded from server on login
- **24 synced keys** — theme, layout, panel sizes, view modes, and more
- **Debounced persistence** — 300ms batching for settings updates

### Agent Config Editor

Edit agent configuration directly from the Flight Deck UI:

- **config.yaml editing** — modify agent configuration in-flight
- **.env editing** — update environment variables
- **Restart warning** — indicates when agent restart is required for changes to take effect

### Docker Deployment

Flight Deck can now be deployed as a standalone Docker container:

- **`Dockerfile.flight-deck`** — Python 3.11 slim base with FFmpeg, Git, and Playwright
- **`docker-compose.flight-deck.yml`** — ready-to-run compose file with volume mounts and API key forwarding
- **Non-root user** — runs as `claw` user for security
- **Port range** — exposes 25080 (Flight Deck) + 24080–24099 (spawned agents)
- **Persistent data** — `fd-data` volume for agent data, settings, and chat history

## Backend Changes

- **`POST /fd/delegate-peer`** — fire-and-forget task delegation with two-phase background execution
- **`POST /fd/auth/register`** — user registration (first user becomes admin)
- **`POST /fd/auth/login`** — login with JWT access + refresh token
- **`POST /fd/auth/refresh`** — refresh access tokens with automatic rotation
- **`POST /fd/auth/logout`** — clear refresh sessions
- **`GET /fd/auth/me`**, **`PUT /fd/auth/me`** — user profile management
- **`GET /fd/auth/status`** — public endpoint for checking auth status and Docker spawn availability
- **`GET /fd/admin/users`**, **`PUT /fd/admin/users/{id}`**, **`DELETE /fd/admin/users/{id}`** — user management
- **`GET /fd/admin/usage`** — usage analytics with event type filtering
- **`GET/PUT /fd/admin/config`** — system configuration
- **`GET/PUT/DELETE /fd/settings`** — per-user settings CRUD
- **`GET/POST/DELETE /fd/chat/sessions`** — chat session management
- **`GET/POST /fd/chat/sessions/{id}/messages`** — chat message persistence
- **SQLite database** (`fd-data/flight_deck.db`) — users, sessions, settings, chat, usage logs with WAL mode
- **Heartbeat streaming** on `/fd/consult-peer` — 15s interval heartbeat events
- **Consultation deduplication** — `_active_consults` tracking prevents duplicate peer consultations
- **`notification` WebSocket message type** — injects content into agent sessions without triggering _busy state
- **`trigger_response` flag** — notifications can trigger agent LLM response, with queuing when agent is busy

## New Files

| File | Description |
|---|---|
| `captain_claw/flight_deck/auth.py` | JWT authentication with bcrypt password hashing and refresh token rotation |
| `captain_claw/flight_deck/auth_routes.py` | Auth endpoints (register, login, refresh, logout, profile) |
| `captain_claw/flight_deck/admin_routes.py` | Admin endpoints (user management, plans, quotas, usage, config) |
| `captain_claw/flight_deck/db.py` | SQLite database with async support for users, sessions, settings, chat, usage |
| `captain_claw/flight_deck/rate_limiter.py` | Tiered rate limiting and quota enforcement |
| `captain_claw/flight_deck/settings_routes.py` | Per-user settings CRUD endpoints |
| `captain_claw/flight_deck/chat_routes.py` | Chat session and message persistence endpoints |
| `captain_claw/tools/consult_peer.py` | Peer consultation tool with approval gates and activity broadcasting |
| `Dockerfile.flight-deck` | Docker image for standalone Flight Deck deployment |
| `docker-compose.flight-deck.yml` | Docker Compose for Flight Deck with volume mounts |
| `flight-deck/src/pages/AdminPage.tsx` | Admin dashboard UI (users, plans, usage, config) |
| `flight-deck/src/pages/LoginPage.tsx` | Login/registration page |
| `flight-deck/src/stores/authStore.ts` | Authentication state management |
| `flight-deck/src/stores/chatStore.ts` | Chat persistence state management |
| `flight-deck/src/services/settingsSync.ts` | Settings sync service with localStorage migration |
| `flight-deck/src/components/agents/AgentConfigEditor.tsx` | In-flight agent config editor |

## Stats

- **~7,200 lines added** across 73 files
- 16 new files, 10 new backend endpoints
- 1 new tool action (`delegate`), 1 new tool (`consult_peer`)
- 1 new Dockerfile, 1 new Docker Compose file
