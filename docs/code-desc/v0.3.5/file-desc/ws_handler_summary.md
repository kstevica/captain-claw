# Summary: ws_handler.py

# ws_handler.py Summary

Manages WebSocket connections for the Captain Claw web UI, handling real-time bidirectional communication between the frontend and the AI agent backend. Implements connection lifecycle management, session state synchronization, and message routing for chat, commands, model switching, personality/playbook configuration, and user feedback.

## Purpose

Solves the problem of maintaining persistent, real-time communication channels between web clients and the AI agent system. Enables:
- Live chat message streaming and session replay
- Dynamic configuration changes (model selection, user personality, playbook overrides)
- User feedback collection on agent responses
- Command execution and monitoring tool output broadcasting
- Session state synchronization across multiple connected clients

## Most Important Functions/Classes

1. **`ws_handler(server: WebServer, request: web.Request) -> web.WebSocketResponse`**
   - Main WebSocket connection handler that manages client lifecycle. Prepares the WebSocket connection, adds client to server's active clients set, sends welcome payload with session info/models/commands/personalities/playbooks, replays existing session messages for new connections, and maintains the async message loop until disconnection.

2. **`handle_ws_message(server: WebServer, ws: web.WebSocketResponse, data: dict) -> None`**
   - Central message dispatcher that routes incoming WebSocket messages to appropriate handlers based on message type. Handles 10+ message types including: chat messages, slash commands, model switching, personality/playbook configuration, force script mode toggling, message feedback, cancellation signals, and approval responses.

3. **Session Replay Logic (within `ws_handler`)**
   - Iterates through existing session messages and reconstructs chat history for newly connected clients. Filters messages by role (user/assistant/tool), formats them appropriately (chat_message, monitor, rephrase), and sends replay_done signal. Ensures new clients have full context of prior conversation.

4. **Model/Personality/Playbook Configuration Handlers**
   - `set_model`: Switches the LLM model for current session via `server.agent.set_session_model()` and broadcasts session info updates
   - `set_personality`: Sets active user profile context (who the agent talks to, not agent identity), clears instruction caches to rebuild prompts with new user context
   - `set_playbook`: Overrides playbook selection (auto/none/specific ID) for task guidance injection

5. **Message Feedback Handler (`message_feedback`)**
   - Stores user like/dislike feedback on assistant messages by timestamp, persists to session storage via session manager, enables quality tracking and model improvement signals.

## Architecture & Dependencies

**Key Dependencies:**
- `aiohttp.web`: WebSocket protocol implementation and HTTP request handling
- `captain_claw.agent.Agent`: Core AI agent with session management and model selection
- `captain_claw.web_server.WebServer`: Server instance managing client connections and broadcasting
- `captain_claw.session`: Session manager for playbook/message persistence
- `captain_claw.personality`: User personality/profile management
- `captain_claw.web.chat_handler` & `slash_commands`: Message processing modules

**System Role:**
Acts as the real-time communication bridge between web frontend and backend agent. Maintains bidirectional state synchronization, handles client connection pooling, and routes all interactive commands/queries. Critical for multi-client support and live UI updates.

**Key Architectural Patterns:**
- Async/await for non-blocking I/O
- Broadcast mechanism for multi-client state updates (`server._broadcast()`)
- Message type dispatch pattern for extensible command handling
- Session replay for client state recovery
- Cache invalidation strategy for personality-dependent prompts