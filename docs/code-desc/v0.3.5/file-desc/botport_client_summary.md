# Summary: botport_client.py

# BotPort Client Summary

**Summary:**
BotPort client maintains a persistent WebSocket connection between Captain Claw instances, enabling bidirectional concern routing and agent-to-agent communication. It handles dual roles: CC-A (originator) sends concerns and receives results; CC-B (handler) receives dispatches, spawns ephemeral agents, and returns responses. The client manages connection lifecycle, heartbeats, message routing, and automatic reconnection with exponential backoff.

**Purpose:**
Solves the problem of distributed agent coordination across multiple Captain Claw instances by providing a reliable, persistent communication channel that abstracts WebSocket complexity, manages concern lifecycle (submission, acknowledgment, result delivery), and enables dynamic agent spawning for handling remote requests while maintaining session context and persona information.

**Most Important Functions/Classes/Procedures:**

1. **`BotPortClient.__init__()` & lifecycle methods (`start()`, `stop()`)**
   - Initializes connection state, callbacks, and concern tracking dictionaries. `start()` establishes session and WebSocket, spawns background loops. `stop()` gracefully shuts down with task cancellation and future resolution.

2. **`_connect()` & `_reconnect_loop()`**
   - `_connect()` establishes WebSocket, sends registration with capabilities, waits for acknowledgment. `_reconnect_loop()` implements exponential backoff (up to 60s) for automatic recovery, restarts receive/heartbeat tasks on successful reconnection.

3. **`send_concern()` & `send_follow_up()`**
   - CC-A methods that send concerns/follow-ups to BotPort with context and expertise tags, create futures for result tracking, implement timeout handling (default 300s), return structured result dicts with ok/error/response/metadata.

4. **`_handle_dispatch()` & `_spawn_dispatch_agent()`**
   - `_handle_dispatch()` receives dispatched concerns, spawns ephemeral agents via `_spawn_dispatch_agent()`, injects agent-to-agent system prompt, runs agent.complete(), sends result back with persona/model metadata. Agents are isolated per concern with max 10 iterations, exclude botport tool to prevent multi-hop.

5. **`_receive_loop()` & `_handle_message()`**
   - `_receive_loop()` continuously reads WebSocket messages, parses JSON, routes to handlers. `_handle_message()` dispatches 10+ message types (heartbeat_ack, concern_result, dispatch, follow_up, context_request/reply, concern_closed, timeout_notice) with appropriate async task creation.

6. **`_build_capabilities()`**
   - Constructs capabilities manifest advertising max_concurrent, personas (global + user profiles with expertise tags), enabled tools, and available models (provider:model format) based on configuration flags.

**Architecture & Dependencies:**
- **Async I/O:** Built on `aiohttp.ClientSession` and `asyncio` for non-blocking WebSocket communication
- **Session Management:** Integrates with `captain_claw.session.SessionManager` to create isolated sessions per dispatch
- **Agent Integration:** Spawns `captain_claw.agent.Agent` instances with provider, callbacks, and instruction loaders
- **Configuration:** Reads from `BotPortClientConfig` (URL, instance name, key/secret, heartbeat interval, reconnect delay, concurrency limits, advertising flags)
- **Logging:** Uses centralized logger via `captain_claw.logging.get_logger()`
- **State Tracking:** Maintains three dictionaries for CC-B dispatch state (agents, sessions, personas) and one for CC-A pending results (concern_id → Future mapping)
- **Message Protocol:** JSON-based with concern_id UUIDs for correlation, metadata enrichment (model_used, tokens), and persona hints for multi-persona support