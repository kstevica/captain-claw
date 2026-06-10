# Captain Claw v0.4.23 Release Notes

**Release title:** Centralised MCP via Flight Deck

**Release date:** 2026-05-06

## Highlights

Captain Claw 0.4.23 moves Model Context Protocol (MCP) management out of every individual agent and into Flight Deck as a fleet-wide service. Add an MCP server once in Flight Deck → Connections → MCP servers and every agent in the fleet immediately has access to its tools — no more copying OAuth credentials into each agent's `config.yaml`, no more N×M connections fanning out to upstream servers, and a single chokepoint for tokens, sessions and observability.

This is the Phase 1 MVP of a larger roadmap. Phase 1 keeps the surface deliberately narrow: HTTP-only transport, fleet-wide allow (no per-agent ACLs yet), no hot push of tool-list changes. Phase 2 will layer those on once we have feedback from real fleets.

## What changed

### Flight Deck is now the MCP control plane

A new `MCPManager` singleton inside Flight Deck owns:

- **OAuth2 `client_credentials` token acquisition** — done once per server, cached, refreshed transparently on 401.
- **MCP `initialize` handshake** — single-flighted via a per-server `asyncio.Lock`, so a thundering-herd of agent boots never re-handshakes the same server multiple times.
- **`tools/list` cache** — 30-second TTL, force-refresh on demand. The Flight Deck UI's "Test" button bypasses the cache.
- **Session tracking** — `Mcp-Session-Id` headers are captured and replayed on subsequent calls.
- **SSE response parsing** — `text/event-stream` responses from upstream servers are folded back into the JSON-RPC result transparently.

A flat JSON store at `~/.captain-claw-fd/mcp_servers.json` persists configured servers. Writes are atomic (write-to-tmp + rename) so a crashed Flight Deck never leaves the file half-written.

### REST surface

Flight Deck exposes eight new endpoints under `/fd/mcp/...`:

**Admin (gated by Flight Deck user auth):**

- `GET /fd/mcp/servers` — list configured servers with secrets masked + runtime status (initialised? tool count? last error?)
- `POST /fd/mcp/servers` — insert or update a server by name. Empty/masked `client_secret` preserves the stored value so the UI can re-submit forms safely.
- `DELETE /fd/mcp/servers/{name}` — remove a server and drop its cached state.
- `POST /fd/mcp/servers/{name}/test` — end-to-end probe (re-init + tools/list).
- `POST /fd/mcp/probe` — probe an ad-hoc transient config without persisting it; powers the "Test connection" button on the Add-server form.

**Agent-facing (gated by loopback or `X-Agent-Secret`):**

- `GET /fd/mcp/agent/servers` — list of enabled server names for an agent's discovery loop.
- `GET /fd/mcp/{name}/tools?refresh={bool}` — proxy `tools/list`.
- `POST /fd/mcp/{name}/call` — proxy `tools/call`.

### Agent side

`captain_claw/tools/mcp_connector.py` now contains an `MCPProxyConnector` that talks exclusively to Flight Deck's REST proxy. Every captain-claw agent enumerates servers from `/fd/mcp/agent/servers` at boot and registers a proxy `Tool` per advertised upstream tool. When `FD_URL` is unset the registration is a clean no-op — MCP is exclusively a Flight Deck feature now.

The `FD_AGENT_SHARED_SECRET` env var is inherited by spawned agents automatically (both spawn paths in `flight_deck/server.py` start from `dict(os.environ)`).

A new `captain_claw/fd_client.py` module factors out the FD URL/secret helpers (`flight_deck_base()`, `flight_deck_headers()`, `flight_deck_slug()`, `is_under_flight_deck()`) plus a small `FDClient` async httpx wrapper used by both `CodexAuthManager` and the MCP proxy connector.

### Flight Deck UI

The Connections page gets a third card alongside Google + ChatGPT (Codex):

- **MCP servers** — collapsible card showing per-server status (Connected / Error / Not initialized), tool counts and an inline Add-server form with a live "Test connection" probe before saving.
- Each row has Test and Remove buttons; remove requires a click-to-confirm step.
- The form preserves the stored `client_secret` if the user re-submits without typing anything in the secret field.

### Cleanup

The agent's per-instance MCP UI is gone:

- The agent's settings page now shows an "MCP servers are managed in Flight Deck" notice instead of the old per-agent array editor.
- The home card linking to `/mcp-connectors` is removed and the static page returns HTTP 410 Gone with a redirect message.
- `MCPServerConfig` and `tools.mcp_servers` remain in the config schema for backward compatibility, but are now ignored at runtime. A one-shot deprecation warning fires at startup if any entries are present.

## Migrating from 0.4.22

If you had MCP servers configured in `config.yaml` under `tools.mcp_servers`:

1. Start Flight Deck.
2. Open **Connections → MCP servers → Add MCP server**.
3. Copy each entry's `name`, `url`, `client_id`, `client_secret`, `token_endpoint` and any custom `headers`.
4. Click **Test connection** to verify, then **Save**.
5. Remove the `tools.mcp_servers` block from `config.yaml` (optional — leaving it in place just emits a startup warning).

Every agent in your fleet picks up the new server on its next boot.

## Tests

A new `tests/test_flight_deck/test_mcp.py` module covers:

- `mcp_storage` round-trip CRUD with secret masking and atomic writes.
- `MCPManager` initialise + tools/list caching + tools/call session tracking, using `httpx.MockTransport` to fake the upstream MCP server.
- The full FastAPI route surface via `TestClient`, including agent-auth gating with a temporary `X-Agent-Secret`.

Nine tests, all green.

## Known limitations (Phase 2 candidates)

- **HTTP transport only.** stdio-transport MCP servers (the standard for local-only servers like the Anthropic-shipped filesystem server) aren't supported yet.
- **No tool-list change push.** When you add a new MCP server, agents pick it up on their next boot. Hot push (`notifications/tools/list_changed` from Flight Deck → agents) is Phase 2.
- **Fleet-wide allow.** Every agent sees every enabled server. Per-agent allowlists / scopes are Phase 2.
- **No streaming tool calls.** Long-running MCP tool calls return as a single JSON response; no progress events surfaced to the agent yet.
