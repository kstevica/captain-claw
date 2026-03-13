# Summary: slack_bridge.py

# slack_bridge.py Summary

**Summary:** A minimal Slack Web API bridge providing DM-first bot interaction with polling-based message retrieval and send capabilities. Implements async HTTP communication using httpx with built-in message batching, user caching, and file upload support.

**Purpose:** Solves the problem of integrating a bot with Slack's Web API by abstracting away HTTP request handling, pagination, error management, and message normalization. Enables polling-based message consumption (rather than WebSocket/Events API) with support for sending text, audio files, and typing indicators to DM channels.

**Most Important Functions/Classes:**

1. **`SlackMessage` (dataclass)** – Normalized incoming message payload containing channel_id, message_ts, user_id, username, text, and optional thread_ts. Provides a clean contract for message handling across the system.

2. **`get_updates(offsets, limit_per_channel)` (async method)** – Core polling mechanism that fetches DM messages from all active channels, filters out bot/system messages, resolves usernames, and returns updates with refreshed offsets for stateful polling. Handles pagination and timestamp-based filtering.

3. **`send_message(channel_id, text, reply_to_message_ts)` (async method)** – Sends text to Slack DM with automatic chunking (max 3500 chars per message) and smart line-break splitting. Supports thread replies via `reply_to_message_ts`.

4. **`_api_get()` and `_api_post()` (async methods)** – Low-level HTTP wrappers handling Bearer token authentication, response validation (checking `ok` field), and error propagation. Support both JSON and multipart payloads.

5. **`_resolve_username(user_id)` (async method)** – Resolves Slack user IDs to display names with in-memory caching. Attempts display_name → real_name → handle fallback chain, reducing API calls for repeated users.

**Architecture & Dependencies:**
- **Async-first design:** Uses `httpx.AsyncClient` for non-blocking HTTP operations with 40-second timeout
- **Stateless polling:** Caller manages offset state (message timestamps) for resumable polling
- **User caching:** In-memory dict caches user ID → username mappings to reduce API calls
- **Pagination support:** Handles cursor-based pagination for `conversations.list` and message history
- **Error handling:** Validates all Slack API responses for `ok: true` and raises RuntimeError on failures
- **File upload:** Supports MP3 audio file uploads via multipart form data
- **No external dependencies beyond httpx** – minimal footprint for embedding in larger systems

**Key Design Decisions:**
- DM-only focus (`types: "im"`) simplifies channel discovery
- Polling-based (vs. Events API) trades latency for simplicity and cost
- Message chunking prevents Slack API rejection of oversized payloads
- Best-effort typing/read signals (Slack Web API limitations acknowledged)
- Defensive parsing with type checks throughout to handle malformed API responses