# Captain Claw v0.4.24 Release Notes

**Release title:** Centralised MCP

**Release date:** 2026-05-06

## Highlights

Captain Claw 0.4.24 lands the four Phase 2 items the 0.4.23 release notes flagged as "known limitations." Flight Deck's MCP control plane now supports **stdio-transport servers**, **per-agent allowlists**, **hot push of tool-list changes**, and **streaming tool calls** — the centralised MCP surface is now feature-complete enough to run real fleets against the wide ecosystem of MCP servers (filesystem, sqlite, github, postgres, etc.) without any per-agent config.

If you upgraded to 0.4.23, no migration is required — every Phase 2 feature is opt-in and existing HTTP-only configurations keep working unchanged.

## What changed

### Phase 2.1 — Per-agent MCP server allowlists

Each MCP server record now carries an optional `allowed_agents: list[str]`. Empty list = "every agent in the fleet" (unchanged Phase 1 behaviour). Once any slug is listed, only those agent slugs can see, list, or call the server.

- New `mcp_storage.is_agent_allowed(record, slug)` is the single permission check. Used by `/fd/mcp/agent/servers`, `/fd/mcp/<name>/tools`, `/fd/mcp/<name>/call`, the new `/fd/mcp/<name>/call_stream` and the new `/fd/mcp/agent/events`.
- Restricted servers return **HTTP 404 "No MCP server named '<name>'"** to disallowed agents — same shape as the "doesn't exist" response, so a restricted server's existence is opaque to agents that aren't entitled to it.
- Agents identify themselves via a new `X-Agent-Slug` header automatically populated from the `FD_AGENT_SLUG` env var (Flight Deck already injected this when spawning sub-agents). Agents that don't identify themselves are only allowed access to servers with empty allowlists.

### Phase 2.2 — stdio transport for MCP servers

Adding an MCP server in Flight Deck now lets you pick **`stdio`** alongside the original `http` transport. The stdio binding is the de-facto standard for local MCP servers shipped via `npx` / `uvx` (Anthropic's reference filesystem, sqlite, github, postgres servers, etc.).

A new `captain_claw/flight_deck/mcp_transport.py` module factors the wire protocol behind a tiny `Transport` ABC (`request` / `notify` / `close`):

- **`HttpTransport`** — the Phase 1 streamable-HTTP implementation, now self-contained: owns the OAuth `client_credentials` flow, captures and replays `Mcp-Session-Id`, parses `text/event-stream` responses.
- **`StdioTransport`** — spawns a long-lived child via `asyncio.create_subprocess_exec`, runs a single background reader task that JSON-decodes lines from stdout and dispatches to per-id `asyncio.Future`s. Concurrent requests on the same subprocess are correlated by JSON-RPC `id`. Stderr is drained at debug level. The 16 MiB stdout line limit accommodates large tool responses (base64 images, etc.) without letting a misbehaving child OOM the host. Process lifecycle: lazy spawn on first request, SIGTERM with a 2-second grace then SIGKILL on `close()`, automatic respawn if the child dies.

`MCPManager` no longer carries any HTTP-specific state — every server is a `Transport` built by the new `build_transport(record)` factory.

**Storage schema additions** (backward-compatible — older records load unchanged):

```yaml
transport: stdio              # "http" (default) or "stdio"
command:   uvx                # stdio only — executable to launch
args:      ["mcp-server-foo", "--flag"]
env:       {PATH: "/usr/bin"}
```

The admin form validates `url` for HTTP transports and `command` for stdio. The "Test connection" probe builds a transient transport, runs `initialize` + `tools/list`, then closes it cleanly so probing a stdio record never leaks a child process.

### Phase 2.3 — Hot push of tool-list changes

Captain-claw agents now hot-reload their MCP proxy tools the moment Flight Deck's admin UI changes a server — no agent restart required.

- New `mcp_events.py` event bus: a process-wide singleton with fan-out publish/subscribe, one bounded `asyncio.Queue` per subscriber. Slow subscribers drop frames silently rather than blocking publishers.
- Admin routes publish `server_added` / `server_updated` / `server_removed` after the storage write + manager state cleanup settle.
- The `Transport.on_notification` hook lets the manager subscribe to upstream `notifications/tools/list_changed` events. When the upstream stdio (or HTTP-SSE) server signals tool-list churn, the manager invalidates its `tools/list` cache and publishes a `tools_changed` event.
- New SSE endpoint **`GET /fd/mcp/agent/events`** filters events through the caller's allowlist and emits `ping` frames every 25 seconds to survive idle-TCP-killing proxies. The connection sends an initial `hello` so the client knows the channel is live before any real event arrives.
- `FDClient` gained an `async with fd.stream("GET", ...)` helper for long-lived SSE consumers (no per-request timeout).
- `mcp_connector.watch_mcp_events(registry)` is a fire-and-forget coroutine the agent boots in the background. It subscribes forever with exponential backoff capped at 30 s, tracks which captain-claw tool names were registered for each upstream server, unregisters them on `server_removed`, and re-runs registration on `server_added` / `server_updated` / `tools_changed`.

The agent-side bootstrap in `agent_context_mixin._register_mcp_tools_async_init` now starts that watcher task automatically when running under Flight Deck.

### Phase 2.4 — Streaming tool calls

Tool calls that emit progress (`notifications/progress` per the MCP spec) are now plumbed through Flight Deck. UIs that want to surface "tool is running, 30 %…" indicators can subscribe to a per-call SSE stream while the model waits for the final result.

- `Transport.request(method, params, on_progress=callable)` is the new opt-in shape. When `on_progress` is supplied, the transport stamps a unique `_meta.progressToken` into params and fans matching `notifications/progress` envelopes out to the callback. Servers that don't support progress simply ignore the token — fully backward-compatible.
- `HttpTransport` walks every SSE frame in the response body, dispatching progress notifications and any other server-initiated notification (e.g. `notifications/tools/list_changed`) before resolving on the final JSON-RPC result.
- `StdioTransport` registers per-token handlers for the lifetime of one call, dispatched from the same background reader that handles regular responses. Cleanup happens in the request's `finally` block whether the call succeeds, errors, or is cancelled.
- `MCPManager.call_tool(..., on_progress=...)` forwards the callback through to the transport.
- New SSE endpoint **`POST /fd/mcp/<name>/call_stream`** runs the upstream call as a background task, drains progress frames off a bounded queue and emits them to the client, then sends a single terminal `result` (or `error`) frame:
  - `data: {"type":"progress","params":{...}}` — one per upstream progress notification
  - `data: {"type":"result","server":"...","tool":"...","result":{...}}`
  - `data: {"type":"error","error":"..."}`
  - `data: {"type":"ping"}` every 25 s while idle
  Cancels the call cleanly when the client disconnects mid-stream.

The agent-side `tools/call` execution path still uses the one-shot `/call` endpoint — wiring streaming progress into the model's tool-result UX is a separate UI concern. The proxy infrastructure is fully in place when that work happens.

## REST surface

Two new endpoints on top of Phase 1:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/fd/mcp/agent/events` | SSE stream of MCP state changes (`server_added`, `server_updated`, `server_removed`, `tools_changed`), filtered by the caller's allowlist. Used by the agent-side watcher to hot-reload tools. |
| `POST` | `/fd/mcp/<name>/call_stream` | Streaming variant of `/call` — emits `progress` frames as they arrive, then a final `result` or `error` frame. Cancels on client disconnect. |

The Phase 1 surface (`/fd/mcp/servers`, `/fd/mcp/<name>/tools`, `/fd/mcp/<name>/call`, `/fd/mcp/probe`, `/fd/mcp/servers/<name>/test`, `/fd/mcp/agent/servers`) is unchanged but now also accepts `transport` / `command` / `args` / `env` / `allowed_agents` in the admin payloads.

## Migrating from 0.4.23

Nothing required. Existing HTTP server configs work unchanged with empty allowlists. To opt into Phase 2 features:

1. **Per-agent ACLs:** edit the server in Flight Deck → Connections → MCP servers and add the agent slugs you want to allow. An empty list keeps the fleet-wide-allow behaviour.
2. **stdio transport:** add a new server, choose **stdio** transport, fill in `command` (e.g. `uvx`) plus optional `args` and `env`. The "Test connection" probe spawns the child, runs `tools/list`, and tears down — same UX as HTTP probes.
3. **Hot reload:** automatic. As soon as agents are spawned with this build, they'll subscribe to `/fd/mcp/agent/events` on boot and re-register tools whenever you save a server.
4. **Streaming calls:** any caller (UI, integration, eventually the model itself) can `POST /fd/mcp/<name>/call_stream` with the same payload as `/call` to get an SSE-framed response.

## Known limitations / next steps

- **Streaming through the agent's tool execution loop.** The infrastructure is in place but the model still sees a single final `ToolResult`. Surfacing live progress to the model (and to the chat panel) is a separate UI / agent-runtime change.
- **HTTP server-initiated tool-list change notifications.** The transport supports them, but most HTTP-binding MCP servers don't actively push these mid-session. stdio servers that do (e.g. ones that watch a filesystem) trigger the hot-reload path today.
- **Allowlist UX.** Allowed-agents is currently a free-text list of slugs. A picker that pulls from Flight Deck's known agent fleet is a follow-up UI polish task.
