# Captain Claw — Frontend Integration Guide

> Comprehensive reference for coding agents and developers building a frontend for Captain Claw.
> Covers every WebSocket message, every REST endpoint, authentication, file uploads, and streaming patterns.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Quick Start](#2-quick-start)
3. [WebSocket Protocol — Main Chat (`/ws`)](#3-websocket-protocol--main-chat-ws)
4. [WebSocket Protocol — Speech-to-Text (`/ws/stt`)](#4-websocket-protocol--speech-to-text-wsstt)
5. [REST API Reference](#5-rest-api-reference)
6. [Authentication](#6-authentication)
7. [File Upload Flow](#7-file-upload-flow)
8. [Streaming Patterns](#8-streaming-patterns)
9. [State Management Recommendations](#9-state-management-recommendations)

---

## 1. Architecture Overview

Captain Claw exposes a **hybrid WebSocket + HTTP REST** backend built on **aiohttp** (Python async HTTP framework).

| Channel | Purpose |
|---------|---------|
| **WebSocket `/ws`** | Real-time bidirectional: chat messages, streaming LLM responses, tool output, status updates, thinking indicators, approval flow |
| **WebSocket `/ws/stt`** | Speech-to-text: browser streams raw PCM audio, server returns transcription |
| **HTTP REST** | CRUD for sessions, settings, contacts, todos, scripts, cron jobs, files, memory, playbooks, datastore, and more |
| **OpenAI-compatible API** | `POST /v1/chat/completions` — programmatic access using standard OpenAI format (when enabled in config) |

**Key characteristics:**
- All messages are **JSON** (WebSocket text frames)
- WebSocket max message size: **4 MB** (main), **10 MB** (STT)
- The server maintains a **single shared agent** for admin connections and **per-session isolated agents** for public mode
- Session history is **replayed** on WebSocket connect so the frontend doesn't need to persist chat state
- Streaming uses dedicated message types (`response_stream`, `tool_stream`, `thinking`) — the final assembled response is sent separately as a `chat_message`

---

## 2. Quick Start

### Minimal connection (JavaScript)

```javascript
// 1. Connect
const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${protocol}//${location.host}/ws`);

// 2. Handle messages
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  switch (msg.type) {
    case 'welcome':
      // Connection established. msg.session, msg.models, msg.commands available.
      console.log('Connected to session:', msg.session.id);
      break;

    case 'replay_done':
      // All historical messages have been replayed. UI is now up to date.
      break;

    case 'chat_message':
      // msg.role: "user" | "assistant" | "rephrase" | "image" | "audio" | "html_file"
      // msg.content: the text or file path
      // msg.replay: true if this is a historical message being replayed
      displayMessage(msg.role, msg.content);
      break;

    case 'response_stream':
      // Partial LLM response chunk. Append to a streaming buffer.
      appendToStreamBuffer(msg.text);
      break;

    case 'status':
      // msg.status: "ready" | "thinking" | ...
      updateStatusIndicator(msg.status);
      break;

    case 'error':
      showError(msg.message);
      break;
  }
};

// 3. Auto-reconnect
ws.onclose = () => setTimeout(() => connect(), 3000);

// 4. Send a chat message
function sendChat(text) {
  ws.send(JSON.stringify({ type: 'chat', content: text }));
}

// 5. Cancel a running task
function cancelTask() {
  ws.send(JSON.stringify({ type: 'cancel' }));
}
```

### Minimal REST call

```javascript
// List sessions
const sessions = await fetch('/api/sessions').then(r => r.json());

// Upload an image, then reference it in chat
const formData = new FormData();
formData.append('file', imageBlob, 'photo.png');
const { path } = await fetch('/api/image/upload', { method: 'POST', body: formData }).then(r => r.json());
ws.send(JSON.stringify({ type: 'chat', content: 'Analyze this image', image_path: path }));
```

---

## 3. WebSocket Protocol — Main Chat (`/ws`)

### 3.1 Connection Lifecycle

```
Client                          Server
  |                               |
  |-- WebSocket connect --------> |
  |                               |-- validate auth (public mode)
  |<--------- welcome ----------- |  (session, models, commands, personalities, playbooks)
  |<------ chat_message (replay)- |  (one per historical message)
  |<------ monitor (replay) ----- |  (one per historical tool call)
  |<------ replay_done ---------- |  (history complete, UI ready)
  |                               |
  |== bidirectional messages ====>|
  |<===== bidirectional messages ==|
```

**Reconnection:** On disconnect, wait 3 seconds and reconnect. The server will replay the full session again.

**Public mode:** If `web.public_run` is enabled, unauthenticated clients must have a valid session cookie (obtained via `POST /api/public/session/new`). Invalid sessions receive close code `4001`.

### 3.2 Welcome Message (Server → Client)

Sent immediately after WebSocket connection is established.

```json
{
  "type": "welcome",
  "session": {
    "id": "abc123",
    "name": "My Session"
  },
  "models": [
    {
      "id": "default",
      "provider": "anthropic",
      "model": "claude-sonnet-4-20250514",
      "description": "Default model",
      "model_type": "llm"
    }
  ],
  "commands": [
    {
      "command": "/new [name]",
      "description": "Create a new session",
      "category": "Sessions"
    }
  ],
  "personalities": [
    {
      "id": "user_123",
      "name": "John",
      "description": "Product manager",
      "is_telegram": false
    }
  ],
  "playbooks": [
    {
      "id": "pb_456",
      "name": "Code Review",
      "task_type": "review",
      "trigger_description": "When reviewing code..."
    }
  ]
}
```

**Notes:**
- `models` — array of model definitions the agent can use. Show in a model selector dropdown.
- `commands` — slash commands available. Used for autocompletion when user types `/`.
- `personalities` — user profiles. Show in a persona selector.
- `playbooks` — behavioral playbooks. Show in a playbook override selector.
- In **public mode**, `commands` and `playbooks` are empty arrays.

### 3.3 Session Replay (Server → Client)

After the welcome message, the server replays all messages from the active session. Each replayed message has `"replay": true`.

**Replayed message types:**
- `chat_message` with `role: "user"` or `role: "assistant"` — historical chat messages
- `chat_message` with `role: "rephrase"` — task rephrasing from the `task_rephrase` tool
- `monitor` — historical tool executions (tool name, arguments, output)

**End of replay:**
```json
{ "type": "replay_done" }
```

**Implementation tip:** During replay, batch DOM updates. After `replay_done`, scroll to bottom and mark UI as ready for interaction.

### 3.4 Client → Server Messages

#### `chat` — Send a Chat Message

The primary message type for user input.

```json
{
  "type": "chat",
  "content": "Explain how WebSockets work",
  "image_path": "/path/to/image.png",
  "file_path": "/path/to/data.csv",
  "image_paths": ["/path/to/img1.png", "/path/to/img2.png"],
  "file_paths": ["/path/to/a.csv", "/path/to/b.xlsx"],
  "rewind_to": "2026-03-17T10:30:00+00:00"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | Yes (unless files attached) | The user's message text. If starts with `/`, routed to slash command handler. |
| `image_path` | string | No | Single image path (backward compat). Obtain from `POST /api/image/upload`. |
| `file_path` | string | No | Single file path (backward compat). Obtain from `POST /api/file/upload`. |
| `image_paths` | string[] | No | Multiple image paths. |
| `file_paths` | string[] | No | Multiple file paths. |
| `rewind_to` | string | No | ISO-8601 timestamp. Truncates session history to this point before processing (enables "fork from here" / history branching). |

**Behavior:**
- If `content` starts with `/`, it's treated as a slash command (same as the `command` message type).
- Attachments are prepended as `[Attached image: /path]` or `[Attached file: /path]` lines.
- If only attachments and no text, a default message is generated ("Please analyze this image." / "I've attached a file." / "Please analyze these files.").
- Server immediately broadcasts the user message as a `chat_message` with `role: "user"`, sets status to `"thinking"`, and launches the agent.

#### `command` — Execute a Slash Command

```json
{
  "type": "command",
  "command": "/new My Project"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | string | Yes | Full slash command string including arguments. |

**Response:** Server sends a `command_result` message with the output.

#### `btw` — Inject Instructions Mid-Task

Send additional instructions while the agent is actively processing a task. The agent incorporates these into its remaining steps.

```json
{
  "type": "btw",
  "content": "Also make sure to add error handling"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | Yes | Additional instruction to inject. |

**Behavior:**
- Appended to the agent's `_btw_instructions` list.
- Agent checks this list during tool execution loop iterations.
- Instructions are cleared at the end of the current task.
- Server responds with a `command_result`:
  ```json
  {
    "type": "command_result",
    "command": "/btw",
    "content": "Got it — noted for the remaining steps: *Also make sure to add error handling*"
  }
  ```

**When to use:** When the user realizes they want to add requirements after submitting a task, without cancelling and re-submitting.

#### `set_model` — Change Active Model

```json
{
  "type": "set_model",
  "selector": "anthropic/claude-sonnet-4-20250514"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `selector` | string | Yes | Model ID or `provider/model` string. Must match one of the IDs from the `welcome.models` array. |

**Behavior:**
- **Admin only.** Public users receive an error message.
- On success, broadcasts `session_info` to all clients and returns a `command_result`.
- The model change persists across messages within the session.

#### `set_personality` — Change Active User Profile

```json
{
  "type": "set_personality",
  "personality_id": "user_123"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `personality_id` | string \| null | Yes | User profile ID from `welcome.personalities[].id`. Send empty string or null to clear (use default context). |

**Behavior:**
- **Admin only.** Public users receive an error.
- Clears instruction caches so the system prompt rebuilds with the new user context.
- Broadcasts `session_info` to all clients.

#### `set_playbook` — Override Playbook Selection

```json
{
  "type": "set_playbook",
  "playbook_id": "pb_456"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `playbook_id` | string | Yes | One of: empty string `""` → Auto mode (system selects), `"__none__"` → No playbook, or a specific playbook ID from `welcome.playbooks[].id`. |

**Behavior:**
- **Admin only.** Public users receive an error.
- Broadcasts `session_info` to all clients.

#### `set_force_script` — Toggle Force Script Mode

```json
{
  "type": "set_force_script",
  "enabled": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `enabled` | boolean | Yes | Whether to force the agent to use script-based execution. |

**Behavior:**
- **Admin only.** Silently ignored for public users.
- Broadcasts `session_info` to all clients.
- No explicit response sent.

#### `message_feedback` — Like/Dislike a Message

```json
{
  "type": "message_feedback",
  "timestamp": "2026-03-17T10:30:00+00:00",
  "feedback": "good"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timestamp` | string | Yes | ISO-8601 timestamp of the assistant message to rate. |
| `feedback` | string \| null | Yes | `"good"`, `"bad"`, or `null` to clear feedback. |

**Behavior:**
- Finds the assistant message matching the timestamp.
- Persists feedback to the session store.
- No response sent. Update UI optimistically.

#### `cancel` — Cancel Current Task

```json
{
  "type": "cancel"
}
```

**Behavior:**
- Sets the agent's `cancel_event`.
- Agent checks this event at the top of each iteration in the tool loop.
- Agent stops cleanly and sends `{"type": "status", "status": "ready"}`.
- No explicit response — the `status: "ready"` message signals completion.

#### `approval_response` — Respond to Approval Request

```json
{
  "type": "approval_response"
}
```

**Note:** This message type is currently defined but **not yet implemented** on the backend. The web UI auto-approves all approval requests. Future versions may support async approval dialogs.

### 3.5 Server → Client Messages

#### `status` — Agent Status Update

```json
{
  "type": "status",
  "status": "thinking"
}
```

| Value | Meaning |
|-------|---------|
| `"ready"` | Agent is idle, ready for input. |
| `"thinking"` | Agent is processing a request. |
| Custom strings | E.g. `"🎙️ listening... (release key to stop)"` for voice input. |

**When to use:** Show/hide a loading indicator. Disable/enable the input field.

#### `chat_message` — Chat Message

The main message type for conversation content.

```json
{
  "type": "chat_message",
  "role": "assistant",
  "content": "Here's how WebSockets work...",
  "timestamp": "2026-03-17T10:30:15+00:00",
  "model": "anthropic:claude-sonnet-4-20250514",
  "replay": false,
  "feedback": "good"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `role` | string | `"user"`, `"assistant"`, `"rephrase"`, `"image"`, `"audio"`, `"html_file"` |
| `content` | string | Message text (for user/assistant) or file path (for image/audio/html_file). |
| `timestamp` | string | ISO-8601 timestamp. Use as unique ID for feedback. |
| `model` | string | `"provider:model"` label. Only present on assistant messages. |
| `replay` | boolean | `true` during session history replay. |
| `feedback` | string \| undefined | `"good"` or `"bad"` if user has rated this message. |

**Role-specific rendering:**

| Role | Rendering |
|------|-----------|
| `user` | User message bubble. Content may contain `[Attached image: /path]` or `[Attached file: /path]` prefixes. |
| `assistant` | Assistant response. Render as markdown. Show model label and feedback buttons. |
| `rephrase` | Task rephrase panel. Display in a distinct UI element (e.g. a collapsible card above the response). |
| `image` | `content` is a file path. Render as `<img src="/api/media?path={encoded_path}">`. |
| `audio` | `content` is a file path. Render as `<audio controls src="/api/media?path={encoded_path}">`. |
| `html_file` | `content` is a file path. Render as an iframe: `<iframe src="/api/files/view?logical={encoded_path}">`. |

#### `monitor` — Tool Execution Result

```json
{
  "type": "monitor",
  "tool_name": "bash_run",
  "arguments": { "command": "ls -la" },
  "output": "total 42\ndrwxr-xr-x ...",
  "replay": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | string | Name of the tool that executed. |
| `arguments` | object | Arguments passed to the tool. |
| `output` | string | Tool output text. |
| `replay` | boolean | `true` during session replay. |

**When to use:** Display in a separate "monitor" or "tool output" panel. Not all tools generate monitor messages — internal/silent tools are excluded.

#### `thinking` — Reasoning/Thinking Updates

```json
{
  "type": "thinking",
  "text": "Analyzing the code structure...",
  "tool": "bash_run",
  "phase": "tool"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Thinking/reasoning text. |
| `tool` | string | Tool name (if phase is "tool"). |
| `phase` | string | `"reasoning"` for LLM reasoning, `"tool"` for tool execution. |

**When to use:** Show in an inline thinking indicator (e.g. animated text below the loading state). Replace on each new message.

#### `tool_output_inline` — Tool Summary for Thinking Indicator

```json
{
  "type": "tool_output_inline",
  "tool": "web_search",
  "summary": "Web Search: 'captain claw frontend api'",
  "output": "Results: 1. Captain Claw docs... [3000 chars total]"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `tool` | string | Tool name. |
| `summary` | string | Short human-readable summary of the tool action. |
| `output` | string | Truncated tool output (max 3000 chars). |

**When to use:** Update the thinking indicator with tool-specific context. Shows what the agent is doing while it works.

#### `tool_stream` — Live Tool Output Chunks

```json
{
  "type": "tool_stream",
  "chunk": "Installing dependencies...\n"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `chunk` | string | Partial tool output (e.g. live terminal output). |

**When to use:** Append to a streaming tool output display. These arrive during long-running tool executions (e.g. `bash_run` with a build process).

#### `response_stream` — Streaming LLM Response

```json
{
  "type": "response_stream",
  "text": "Here's"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Partial LLM response token(s). |

**When to use:** Append to a streaming response buffer. When the final `chat_message` with `role: "assistant"` arrives, replace the stream buffer with the complete response.

**Important:** `response_stream` messages arrive *before* the final `chat_message`. The final `chat_message` contains the complete response — you can either:
1. Stream tokens into a temporary element, then replace with the final content, or
2. Ignore `response_stream` and just display the final `chat_message` (no streaming UX).

#### `approval_notice` — Auto-Approval Notification

```json
{
  "type": "approval_notice",
  "message": "Execute bash command: rm -rf /tmp/build ?"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | The approval question that was auto-approved. |

**When to use:** Display as a non-blocking notification (e.g. toast) showing what was auto-approved. Currently the web backend always auto-approves.

#### `usage` — Token Usage Statistics

```json
{
  "type": "usage",
  "last": {
    "prompt_tokens": 1500,
    "completion_tokens": 300,
    "total_tokens": 1800,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 500
  },
  "total": {
    "prompt_tokens": 45000,
    "completion_tokens": 12000,
    "total_tokens": 57000
  },
  "context_window": {
    "context_budget_tokens": 200000,
    "prompt_tokens": 45000,
    "utilization": 0.225
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `last` | object | Token usage for the most recent LLM call. |
| `total` | object | Cumulative token usage for the session. |
| `context_window` | object | Context window budget and current utilization. |

**When to use:** Display a token usage meter or context window utilization bar. Sent after every agent `complete()` call.

#### `session_info` — Session Metadata Update

```json
{
  "type": "session_info",
  "id": "session_abc",
  "name": "My Session",
  "model": "claude-sonnet-4-20250514",
  "provider": "anthropic",
  "description": "Working on API docs",
  "message_count": 42,
  "tools": [
    { "name": "bash_run", "description": "Execute shell commands" }
  ],
  "skills": [
    { "name": "/analyze", "skill": "data_analysis", "description": "Analyze data files" }
  ],
  "personality_id": "user_123",
  "personality_name": "John",
  "playbook_id": "pb_456",
  "playbook_name": "Code Review",
  "force_script": false
}
```

**When to use:** Sync UI state — update the session name in the header, highlight the active model/personality/playbook in dropdowns, show available tools and skills.

**Broadcast on:** Model change, personality change, playbook change, force-script toggle, after every agent response.

#### `next_steps` — Suggested Follow-Up Actions

```json
{
  "type": "next_steps",
  "options": [
    {
      "text": "Run the test suite",
      "description": "Verify the changes work correctly"
    },
    {
      "text": "Deploy to staging",
      "description": "Push changes to the staging environment"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `options` | array | Suggested next actions. Each has `text` (short label) and `description`. |

**When to use:** Display as clickable suggestion chips below the assistant's response. When clicked, send the `text` as a new chat message.

#### `command_result` — Slash Command Output

```json
{
  "type": "command_result",
  "command": "/session",
  "content": "**Session:** My Project\n**Messages:** 42\n**Model:** claude-sonnet-4-20250514"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `command` | string | The command that was executed. |
| `content` | string | Command output (often markdown). |

**When to use:** Display inline in the chat as a system message.

#### `error` — Error Message

```json
{
  "type": "error",
  "message": "Agent is busy processing another request. Please wait."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Human-readable error description. |

**Common errors:**
- `"Invalid JSON"` — malformed WebSocket message
- `"Agent not initialized"` — server not ready
- `"Agent is busy processing another request. Please wait."` — send queue busy
- `"Your session is busy processing. Please wait."` — public mode per-session busy
- `"Session error: ..."` — public session load failure

#### `replay_done` — Session Replay Complete

```json
{
  "type": "replay_done"
}
```

**When to use:** Mark the UI as ready for interaction. Scroll to the bottom of the chat. Enable the input field.

---

## 4. WebSocket Protocol — Speech-to-Text (`/ws/stt`)

A separate WebSocket endpoint for voice input. Accepts raw PCM audio and returns transcription.

### Connection

```javascript
const sttWs = new WebSocket(`${protocol}//${location.host}/ws/stt`);
```

Max message size: **10 MB**.

### Protocol

```
Client                          Server
  |                               |
  |-- WebSocket connect --------> |
  |<------ stt_ready ------------ |  { "type": "stt_ready", "realtime": true/false }
  |                               |
  |-- (binary PCM chunks) ------> |
  |<------ stt_partial ---------- |  (realtime only, partial transcript)
  |-- { "type": "stop" } -------> |  (user stopped recording)
  |<------ stt_final ------------ |  (final transcript)
```

### Server → Client Messages

#### `stt_ready` — Connection Ready

```json
{
  "type": "stt_ready",
  "realtime": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `realtime` | boolean | `true` if Soniox realtime STT is available (streaming partial results). `false` for batch mode. |

#### `stt_partial` — Partial Transcript (Realtime Only)

```json
{
  "type": "stt_partial",
  "text": "Hello, I want to..."
}
```

Only sent when `realtime: true`. Shows progressively refined text as the user speaks.

#### `stt_final` — Final Transcript

```json
{
  "type": "stt_final",
  "text": "Hello, I want to build a new frontend"
}
```

Final transcription. Insert into the chat input field.

#### `stt_error` — Error

```json
{
  "type": "stt_error",
  "error": "Realtime STT error: connection lost"
}
```

### Client → Server Messages

| Message | Description |
|---------|-------------|
| Binary frames | Raw PCM int16 audio at 16 kHz, mono. Send chunks as they're recorded. |
| `{"type": "stop"}` | User stopped recording. Server completes transcription and sends `stt_final`. |

### Audio Recording Implementation

```javascript
// Record from microphone at 16 kHz, PCM int16
const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000 } });
const context = new AudioContext({ sampleRate: 16000 });
const source = context.createMediaStreamSource(stream);
const processor = context.createScriptProcessor(4096, 1, 1);

processor.onaudioprocess = (e) => {
  const float32 = e.inputBuffer.getChannelData(0);
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    int16[i] = Math.max(-32768, Math.min(32767, Math.round(float32[i] * 32767)));
  }
  sttWs.send(int16.buffer);
};

source.connect(processor);
processor.connect(context.destination);

// When user releases the mic button:
sttWs.send(JSON.stringify({ type: 'stop' }));
```

---

## 5. REST API Reference

All endpoints accept and return JSON unless noted otherwise. Errors return `{"error": "message"}` with appropriate HTTP status codes.

### 5.1 Sessions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/sessions` | List all sessions |
| GET | `/api/sessions/{id}` | Get session with full message history |
| PATCH | `/api/sessions/{id}` | Update session name/description |
| DELETE | `/api/sessions/{id}` | Delete a session |
| POST | `/api/sessions/{id}/auto-describe` | LLM-generate a session description |
| POST | `/api/sessions/bulk-delete` | Delete multiple sessions |
| GET | `/api/sessions/{id}/export` | Export session as markdown |

**GET `/api/sessions`**
```
Query: ?q=search+term (optional)
Response: [{ "id", "name", "message_count", "created_at", "updated_at", "description" }, ...]
```

**GET `/api/sessions/{id}`**
```
Response: { "id", "name", "description", "created_at", "updated_at", "message_count", "messages": [...] }
```

**PATCH `/api/sessions/{id}`**
```
Body: { "name": "New Name", "description": "Optional description" }
Response: { "ok": true }
```

**POST `/api/sessions/{id}/auto-describe`**
```
Response: { "description": "LLM-generated description of the session" }
```

**POST `/api/sessions/bulk-delete`**
```
Body: { "session_ids": ["id1", "id2", ...] }
Response: { "deleted_count": 2 }
```

**GET `/api/sessions/{id}/export`**
```
Query: ?format=chat|monitor|all (default: all)
Response: Markdown file download
```

### 5.2 Settings & Configuration

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/config` | Active configuration summary |
| GET | `/api/settings/schema` | Settings form schema (field types, labels, groups) |
| GET | `/api/settings` | Current setting values (secrets masked) |
| PUT | `/api/settings` | Save partial configuration changes |

**GET `/api/config`**
```
Response: { "model": { "provider", "model", "temperature", "max_tokens" }, "context": {...}, "tools": {...}, "guards": {...} }
```

**PUT `/api/settings`**
```
Body: { "key.subkey": "value", ... }  // Partial YAML config
Response: { "ok": true, "updated": ["key.subkey", ...] }
```

### 5.3 Instructions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/instructions` | List instruction files |
| GET | `/api/instructions/{name}` | Get instruction content |
| PUT | `/api/instructions/{name}` | Save instruction override |
| DELETE | `/api/instructions/{name}` | Revert to system default |

**GET `/api/instructions`**
```
Response: [{ "name": "system_prompt.md", "size": 4096, "overridden": true, "has_micro": false }, ...]
```

**PUT `/api/instructions/{name}`**
```
Body: { "content": "# My custom instruction\n..." }
Response: { "status": "saved", "name": "my_instruction.md", "size": 2048, "overridden": true }
```

### 5.4 Orchestrator

The orchestrator manages multi-session parallel task execution.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/orchestrator/status` | Current orchestration state |
| POST | `/api/orchestrator/reset` | Cancel and reset to idle |
| GET | `/api/orchestrator/skills` | Available skills and tools |
| POST | `/api/orchestrator/rephrase` | LLM-rephrase casual input into structured prompt |
| POST | `/api/orchestrator/prepare` | Pre-flight validation |
| POST | `/api/orchestrator/task/edit` | Edit active task |
| POST | `/api/orchestrator/task/update` | Persist task changes |
| POST | `/api/orchestrator/task/restart` | Restart failed/paused task |
| POST | `/api/orchestrator/task/pause` | Pause task |
| POST | `/api/orchestrator/task/resume` | Resume paused task |
| POST | `/api/orchestrator/task/postpone` | Delay task execution |
| GET | `/api/orchestrator/sessions` | Available orchestrator sessions |
| GET | `/api/orchestrator/models` | Available model options |
| GET | `/api/orchestrator/workflows` | List saved workflows |
| POST | `/api/orchestrator/workflows/save` | Save workflow |
| POST | `/api/orchestrator/workflows/load` | Load workflow |
| DELETE | `/api/orchestrator/workflows/{name}` | Delete workflow |

### 5.5 Cron / Scheduling

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/cron/jobs` | List all cron jobs |
| POST | `/api/cron/jobs` | Create new cron job |
| POST | `/api/cron/jobs/{id}/run` | Execute job immediately |
| POST | `/api/cron/jobs/{id}/pause` | Disable scheduling |
| POST | `/api/cron/jobs/{id}/resume` | Re-enable scheduling |
| PATCH | `/api/cron/jobs/{id}` | Update job parameters |
| DELETE | `/api/cron/jobs/{id}` | Delete job |
| GET | `/api/cron/jobs/{id}/history` | View execution history |

**POST `/api/cron/jobs`**
```json
{
  "kind": "prompt",
  "schedule": { "type": "cron", "expression": "0 9 * * 1-5" },
  "payload": { "prompt": "Check emails and summarize" },
  "session_id": "session_abc"
}
```

**kind values:** `"prompt"`, `"script"`, `"tool"`, `"orchestrate"`

### 5.6 Todos

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/todos` | List todos (filterable) |
| POST | `/api/todos` | Create todo |
| PATCH | `/api/todos/{id}` | Update todo |
| DELETE | `/api/todos/{id}` | Delete todo |

**GET `/api/todos`**
```
Query: ?status=pending&responsible=human&session_id=...
```

**POST `/api/todos`**
```json
{
  "content": "Review PR #42",
  "responsible": "human",
  "priority": "high",
  "source_session": "session_abc",
  "target_session": "session_def",
  "tags": "code-review,urgent"
}
```

### 5.7 Contacts

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/contacts` | List all contacts |
| GET | `/api/contacts/search?q=...` | Search contacts |
| POST | `/api/contacts` | Create contact |
| GET | `/api/contacts/{id}` | Get contact |
| PATCH | `/api/contacts/{id}` | Update contact |
| DELETE | `/api/contacts/{id}` | Delete contact |

### 5.8 Scripts

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/scripts` | List all scripts |
| GET | `/api/scripts/search?q=...` | Search scripts |
| POST | `/api/scripts` | Create script |
| GET | `/api/scripts/{id}` | Get script |
| PATCH | `/api/scripts/{id}` | Update script |
| DELETE | `/api/scripts/{id}` | Delete script |

### 5.9 APIs (Registered API Calls)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/apis` | List registered APIs |
| GET | `/api/apis/search?q=...` | Search APIs |
| POST | `/api/apis` | Register new API |
| GET | `/api/apis/{id}` | Get API definition |
| PATCH | `/api/apis/{id}` | Update API |
| DELETE | `/api/apis/{id}` | Delete API |

**POST `/api/apis`**
```json
{
  "name": "Get Weather",
  "url": "https://api.weather.com/v1/current",
  "method": "GET",
  "description": "Fetch current weather",
  "headers": "Content-Type: application/json",
  "auth_type": "bearer",
  "auth_token": "...",
  "app_name": "weather",
  "tags": "weather,api"
}
```

### 5.10 Personality & User Profiles

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/personality` | Get agent personality |
| PUT | `/api/personality` | Update agent personality |
| POST | `/api/personality/rephrase` | LLM-improve personality text |
| GET | `/api/user-personalities` | List user profiles |
| GET | `/api/user-personalities/{id}` | Get user profile |
| PUT | `/api/user-personalities/{id}` | Create/update user profile |
| DELETE | `/api/user-personalities/{id}` | Delete user profile |
| GET | `/api/telegram-users` | List Telegram users |

### 5.11 Playbooks

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/playbooks` | List playbooks |
| GET | `/api/playbooks/search?q=...` | Search playbooks |
| POST | `/api/playbooks` | Create playbook |
| GET | `/api/playbooks/{id}` | Get playbook |
| PATCH | `/api/playbooks/{id}` | Update playbook |
| DELETE | `/api/playbooks/{id}` | Delete playbook |

**POST `/api/playbooks`**
```json
{
  "name": "Deep Research",
  "task_type": "research",
  "do_pattern": "Always cite sources. Use multiple search queries.",
  "dont_pattern": "Don't make claims without evidence.",
  "trigger_description": "When the user asks for research or analysis",
  "tags": "research"
}
```

### 5.12 Files & Folder Browsing

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/files` | List workspace files |
| GET | `/api/files/session/{session_id}` | Files for a specific session |
| GET | `/api/files/content?logical=...` | Read file content (text, max 2MB) |
| GET | `/api/files/download?logical=...` | Download file (binary) |
| GET | `/api/files/view?logical=...` | View HTML/SVG in browser |
| POST | `/api/files/export` | Export content as markdown file |
| GET | `/api/browse?path=...` | Browse directory contents |
| GET | `/api/drives` | List filesystem drives |
| GET | `/api/folder-trees` | Get hierarchical folder structure |
| GET | `/api/read-folders` | List read-accessible folders |
| POST | `/api/read-folders` | Register read folder |
| DELETE | `/api/read-folders?path=...` | Unregister read folder |

### 5.13 Google Drive Integration

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/read-folders/gdrive` | List authorized Drive folders |
| POST | `/api/read-folders/gdrive` | Authorize Drive folder |
| DELETE | `/api/read-folders/gdrive?folder_id=...` | Revoke folder access |
| GET | `/api/read-folders/gdrive/browse?folder_id=...` | Browse Drive folder |
| GET | `/api/gws-status` | Google Workspace connectivity status |

### 5.14 Skills

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/skills` | List installed skills |
| POST | `/api/skills/install` | Install skill from URL/package |
| POST | `/api/skills/toggle` | Enable/disable skill |

### 5.15 Semantic Memory

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/semantic-memory/status` | Memory index status |
| GET | `/api/semantic-memory/documents` | List indexed documents |
| GET | `/api/semantic-memory/documents/{doc_id}` | Document details with chunks |
| GET | `/api/semantic-memory/search?q=...` | Full-text search |
| GET | `/api/semantic-memory/promote?doc_id=...&chunk_id=...` | Mark chunk as important |

### 5.16 Deep Memory (Typesense)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/deep-memory/status` | Typesense connection status |
| GET | `/api/deep-memory/facets` | Available facet dimensions |
| GET | `/api/deep-memory/documents` | List documents (paginated) |
| GET | `/api/deep-memory/documents/{doc_id}` | Document details |
| DELETE | `/api/deep-memory/documents/{doc_id}` | Remove document |
| POST | `/api/deep-memory/index` | Manually index documents |

### 5.17 Datastore (SQLite Database Browser)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/datastore/tables` | List all tables |
| POST | `/api/datastore/tables` | Create table |
| GET | `/api/datastore/tables/{name}` | Describe table schema |
| PATCH | `/api/datastore/tables/{name}` | Rename table |
| DELETE | `/api/datastore/tables/{name}` | Drop table |
| GET | `/api/datastore/tables/{name}/rows` | Query rows (paginated, filterable) |
| POST | `/api/datastore/tables/{name}/rows` | Insert rows |
| PATCH | `/api/datastore/tables/{name}/rows` | Update rows |
| DELETE | `/api/datastore/tables/{name}/rows?where=...` | Delete rows |
| POST | `/api/datastore/tables/{name}/columns` | Add column |
| DELETE | `/api/datastore/tables/{name}/columns/{col}` | Drop column |
| POST | `/api/datastore/sql` | Execute raw SQL |
| GET | `/api/datastore/tables/{name}/export?format=csv\|json` | Export table |
| POST | `/api/datastore/upload` | Import CSV/Excel as table |
| GET | `/api/datastore/tables/{name}/protections` | List protections |
| POST | `/api/datastore/tables/{name}/protections` | Add protection |
| DELETE | `/api/datastore/tables/{name}/protections` | Remove protection |

**GET `/api/datastore/tables/{name}/rows`**
```
Query: ?limit=100&offset=0&where=status='active'&order_by=created_at DESC
```

**POST `/api/datastore/sql`**
```json
{
  "query": "SELECT * FROM users WHERE active = ?",
  "params": [true]
}
```

### 5.18 Computer / Visualization

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/computer/visualize` | Generate themed HTML visualization |
| POST | `/api/computer/visualize/stream` | Stream HTML generation (SSE) |
| POST | `/api/computer/export-pdf` | Export HTML to PDF |
| POST | `/api/computer/exploration` | Save exploration snapshot |
| GET | `/api/computer/exploration` | List explorations |
| GET | `/api/computer/exploration/{id}` | Get exploration |
| PUT | `/api/computer/exploration/{id}/visual` | Update exploration HTML |
| DELETE | `/api/computer/exploration/{id}` | Delete exploration |

**POST `/api/computer/visualize`**
```json
{
  "prompt": "Show sales by region",
  "result": "Region A: $1M, Region B: $2M...",
  "theme": "dark-corporate",
  "theme_instructions": "Use blue accent colors",
  "token_tier": "standard",
  "model": "anthropic/claude-sonnet-4-20250514"
}
```
Response: `{ "html": "<div>...</div>", "tokens_used": 1500 }`

### 5.19 Browser Workflows

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/browser-workflows` | List workflows |
| POST | `/api/browser-workflows` | Create workflow |
| GET | `/api/browser-workflows/{id}` | Get workflow |
| PATCH | `/api/browser-workflows/{id}` | Update workflow |
| DELETE | `/api/browser-workflows/{id}` | Delete workflow |

### 5.20 Direct API Calls

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/direct-api-calls` | List API calls |
| POST | `/api/direct-api-calls` | Register API call |
| GET | `/api/direct-api-calls/{id}` | Get API call |
| PATCH | `/api/direct-api-calls/{id}` | Update API call |
| DELETE | `/api/direct-api-calls/{id}` | Delete API call |
| POST | `/api/direct-api-calls/{id}/execute` | Execute API call |

**POST `/api/direct-api-calls/{id}/execute`**
```json
{
  "variable_map": { "user_id": "123", "date": "2026-03-17" }
}
```
Response: `{ "status": 200, "response": {...}, "duration_ms": 245 }`

### 5.21 Loops (Workflow Execution)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/loops/start` | Start multi-iteration workflow |
| GET | `/api/loops/status` | Check loop progress |
| POST | `/api/loops/stop` | Cancel running loop |

### 5.22 Media Upload

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/image/upload` | Upload image (multipart form) |
| POST | `/api/file/upload` | Upload data file (multipart form) |
| POST | `/api/audio/transcribe` | Transcribe audio file |
| GET | `/api/media?path=...` | Serve media file |

**Supported formats:**
- Images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`
- Data files: `.csv`, `.xlsx`, `.xls`, `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.md`, `.txt`

**POST `/api/image/upload`**
```
Content-Type: multipart/form-data
Body: file=<image binary>
Response: { "path": "/full/path/to/saved/image.png", "filename": "image.png", "size": 102400 }
```

### 5.23 Visualization Style

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/visualization-style` | Get current theme |
| PUT | `/api/visualization-style` | Update theme |
| POST | `/api/visualization-style/analyze` | Analyze theme |
| POST | `/api/visualization-style/rephrase` | LLM-improve theme text |

### 5.24 Reflections

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/reflections?limit=50` | List reflections |
| GET | `/api/reflections/latest` | Get most recent reflection |
| POST | `/api/reflections/generate` | LLM-generate reflection |
| PUT | `/api/reflections/{timestamp}` | Update reflection |
| DELETE | `/api/reflections/{timestamp}` | Delete reflection |

### 5.25 Authentication (Google OAuth)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/auth/google/login` | Start Google OAuth2 flow (redirect) |
| GET | `/auth/google/callback` | OAuth2 callback (redirect back) |
| GET | `/auth/google/status` | Check connection status |
| POST | `/auth/google/logout` | Disconnect Google account |

**GET `/auth/google/status`**
```
Response: { "connected": true, "email": "user@gmail.com", "scopes": ["drive.readonly", ...] }
```

### 5.26 Public Mode

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/public/session/new` | Create anonymous session |
| POST | `/api/public/session/resume` | Resume session with code |
| GET | `/api/public/session/enter?code=...` | Enter session |
| POST | `/api/public/session/logout` | End session |

**POST `/api/public/session/new`**
```
Response: { "session_id": "pub_abc", "code": "ABCD1234" }
```

### 5.27 OpenAI-Compatible API

Only available when `api_enabled: true` in config.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (OpenAI format) |
| GET | `/v1/models` | List available models |

**POST `/v1/chat/completions`**
```
Headers: Authorization: Bearer <session_id>
Body: {
  "model": "default",
  "messages": [{ "role": "user", "content": "Hello" }],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 4096
}
Response: Standard OpenAI chat completion or SSE stream
```

### 5.28 Onboarding

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/onboarding/status` | Check setup completion |
| POST | `/api/onboarding/validate` | Validate config fields |
| POST | `/api/onboarding/save` | Complete initial setup |
| GET | `/api/onboarding/codex-auth` | Check alternate auth |

### 5.29 Miscellaneous

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/commands` | List slash commands |
| GET | `/api/usage` | Token usage statistics |
| GET | `/api/workflow-browser` | Browse workflow output files |
| GET | `/api/workflow-browser/output/{filename}` | Get workflow output |

---

## 6. Authentication

### Token-Based Auth (Default)

If `web.auth_token` is set in config:
- Pass token as query parameter: `?token=YOUR_TOKEN`
- Or as a signed cookie: `claw_session`

All API routes are protected. The WebSocket connection inherits the auth state.

### Public Mode

When `web.public_run` is enabled:
1. Non-authenticated users access only `/api/public/*` endpoints
2. Create a session: `POST /api/public/session/new` → get `session_id` and `code`
3. Session cookie is set automatically
4. WebSocket uses the cookie for session binding
5. Each public user gets an isolated agent instance

**Public mode restrictions:**
- Cannot change model, personality, playbook, or force-script mode
- No slash commands
- No playbook list
- Isolated sessions (no access to admin sessions)

### Google OAuth

For Google Drive, Gmail, and Google Workspace features:
1. Navigate user to `GET /auth/google/login`
2. User authorizes on Google's consent screen
3. Callback at `GET /auth/google/callback` sets auth cookie
4. Check status: `GET /auth/google/status`

---

## 7. File Upload Flow

Files must be uploaded via REST before being referenced in WebSocket chat messages.

### Step 1: Upload via REST

```javascript
// Image upload
const formData = new FormData();
formData.append('file', imageFile);
const { path } = await fetch('/api/image/upload', {
  method: 'POST',
  body: formData,
}).then(r => r.json());

// Data file upload
const formData = new FormData();
formData.append('file', csvFile);
const { path } = await fetch('/api/file/upload', {
  method: 'POST',
  body: formData,
}).then(r => r.json());
```

### Step 2: Reference in Chat Message

```javascript
// Single file
ws.send(JSON.stringify({
  type: 'chat',
  content: 'Analyze this data',
  file_path: path,
}));

// Multiple files
ws.send(JSON.stringify({
  type: 'chat',
  content: 'Compare these images',
  image_paths: [path1, path2, path3],
}));

// Mixed (images + files)
ws.send(JSON.stringify({
  type: 'chat',
  content: 'Process these',
  image_paths: [imagePath],
  file_paths: [csvPath, xlsxPath],
}));
```

### Step 3: Serving Uploaded Files

```html
<!-- Images -->
<img src="/api/media?path=ENCODED_PATH" />

<!-- Audio -->
<audio controls src="/api/media?path=ENCODED_PATH"></audio>

<!-- HTML/SVG -->
<iframe src="/api/files/view?logical=ENCODED_PATH"></iframe>

<!-- Download -->
<a href="/api/files/download?logical=ENCODED_PATH">Download</a>
```

---

## 8. Streaming Patterns

### LLM Response Streaming

The agent streams responses token-by-token. Here's how to handle it:

```javascript
let streamBuffer = '';
let isStreaming = false;

function handleMessage(msg) {
  switch (msg.type) {
    case 'status':
      if (msg.status === 'thinking') {
        isStreaming = true;
        streamBuffer = '';
        showStreamingPlaceholder();
      }
      if (msg.status === 'ready') {
        isStreaming = false;
      }
      break;

    case 'response_stream':
      // Append streaming token
      streamBuffer += msg.text;
      updateStreamingDisplay(streamBuffer);
      break;

    case 'chat_message':
      if (msg.role === 'assistant' && !msg.replay) {
        // Final complete response — replace stream buffer
        hideStreamingPlaceholder();
        displayFinalMessage(msg.content);
        streamBuffer = '';
      }
      break;
  }
}
```

### Tool Output Streaming

Long-running tools (e.g. shell commands) stream output in real time:

```javascript
case 'tool_stream':
  appendToToolConsole(msg.chunk);  // Append each chunk to a terminal-like display
  break;

case 'monitor':
  // Tool completed — show final result in monitor panel
  addMonitorEntry(msg.tool_name, msg.arguments, msg.output);
  break;
```

### Thinking Indicator

```javascript
case 'thinking':
  if (msg.phase === 'reasoning') {
    showThinkingText(msg.text);
  } else if (msg.phase === 'tool') {
    showThinkingText(`Using ${msg.tool}: ${msg.text}`);
  }
  break;

case 'tool_output_inline':
  showThinkingText(`${msg.summary}`);
  break;
```

### Full Message Lifecycle

For a typical user message, the server sends messages in this order:

```
1. status          { status: "thinking" }        — Agent started
2. thinking        { text: "...", phase: "reasoning" }  — LLM reasoning
3. thinking        { text: "...", tool: "web_search", phase: "tool" }  — Using a tool
4. tool_stream     { chunk: "..." }              — Live tool output (if applicable)
5. monitor         { tool_name, arguments, output }  — Tool completed
6. tool_output_inline { tool, summary, output }  — Tool summary for thinking
7. (repeat 3-6 for more tool calls)
8. response_stream { text: "Here" }              — LLM starts responding
9. response_stream { text: "'s how" }            — More tokens
10. response_stream { text: "..." }              — ...
11. chat_message   { role: "assistant", content: "full response" }  — Final response
12. next_steps     { options: [...] }            — Suggested follow-ups (optional)
13. usage          { last, total, context_window }  — Token stats
14. session_info   { id, name, model, ... }      — Updated session metadata
15. status         { status: "ready" }           — Agent done
```

---

## 9. State Management Recommendations

### Essential State to Track

```typescript
interface AppState {
  // Connection
  ws: WebSocket | null;
  isConnected: boolean;

  // Session
  sessionId: string;
  sessionName: string;

  // Agent state
  status: 'ready' | 'thinking' | string;
  streamBuffer: string;

  // Messages
  messages: Message[];        // Chat history (populated from replay + live)
  monitorEntries: Monitor[];  // Tool output history

  // Selectors (from welcome message)
  models: Model[];
  commands: Command[];
  personalities: Personality[];
  playbooks: Playbook[];

  // Active selections
  activeModel: string;
  activePersonalityId: string;
  activePlaybookId: string;
  forceScriptMode: boolean;

  // Attachments (pending upload)
  pendingImages: string[];    // Uploaded image paths
  pendingFiles: string[];     // Uploaded file paths

  // Usage
  tokenUsage: UsageData;
}
```

### Message Type

```typescript
interface Message {
  role: 'user' | 'assistant' | 'rephrase' | 'image' | 'audio' | 'html_file';
  content: string;
  timestamp?: string;
  model?: string;
  replay?: boolean;
  feedback?: 'good' | 'bad' | null;
}

interface Monitor {
  toolName: string;
  arguments: Record<string, any>;
  output: string;
  replay?: boolean;
}
```

### Key Implementation Patterns

1. **Replay vs Live:** During replay (`replay: true`), batch UI updates and don't scroll. After `replay_done`, scroll to bottom and enable input.

2. **Streaming buffer:** Keep a separate buffer for `response_stream` tokens. On final `chat_message`, replace the buffer contents with the complete response.

3. **Busy state:** Disable the send button when `status !== "ready"`. Show a cancel button instead.

4. **Optimistic feedback:** When the user clicks like/dislike, update the UI immediately without waiting for a response (the server doesn't send one).

5. **Session sync:** On every `session_info` message, update the active model/personality/playbook displayed in the UI.

6. **File attachments:** Stage uploads in a pending list with preview chips. Clear after sending the chat message.

7. **Error handling:** Display `error` messages as non-blocking notifications. Keep the chat functional.

8. **Markdown rendering:** Assistant messages contain markdown. Use a markdown renderer (e.g. `marked`, `markdown-it`, or a framework-specific component).

9. **Command autocomplete:** When the input starts with `/`, filter the `commands` array and show suggestions. Tab or Enter to select.

10. **Media paths:** All file paths from `image`, `audio`, and `html_file` roles need to be URL-encoded and served via `/api/media?path=...` or `/api/files/view?logical=...`.
