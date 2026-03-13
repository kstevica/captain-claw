# Captain Claw: Comprehensive System Summary

Captain Claw is a sophisticated, multi-modal AI agent framework built on Python that orchestrates complex workflows through LLM-powered task decomposition, parallel execution, and intelligent tool management. The system spans 137 files organized into core agent logic, tool ecosystem (40+ tools), web UI infrastructure, session/memory management, and platform integrations (Telegram, Discord, Slack, Google Workspace).

## System Architecture Overview

### Core Agent Engine

The agent system centers on a **mixin-based architecture** where `Agent` class inherits from 13 specialized mixins providing distinct capabilities:

1. **Orchestration** (`agent_orchestration_mixin.py`) — Main turn-level request processing loop managing iteration budgets, progress tracking, and completion gating
2. **Tool Loop** (`agent_tool_loop_mixin.py`) — LLM tool call extraction, execution, and result management with duplicate detection and scale-aware guards
3. **Completion** (`agent_completion_mixin.py`) — Multi-stage validation gates ensuring task requirements are met before response finalization
4. **Context** (`agent_context_mixin.py`) — Dynamic system prompt construction, semantic memory integration, and intelligent message selection within token budgets
5. **Session** (`agent_session_mixin.py`) — Token-aware message handling, context compaction, and runtime configuration synchronization
6. **File Operations** (`agent_file_ops_mixin.py`) — Script generation, execution, and structured result wrapping
7. **Guard** (`agent_guard_mixin.py`) — Input/output content filtering and approval workflows
8. **Model** (`agent_model_mixin.py`) — Runtime model selection and provider resolution
9. **Pipeline** (`agent_pipeline_mixin.py`) — DAG-based task pipeline construction with dependency resolution and timeout management
10. **Reasoning** (`agent_reasoning_mixin.py`) — Task contract generation, critic validation, and list-member extraction
11. **Research** (`agent_research_mixin.py`) — Multi-stage web research pipeline for entity extraction and content aggregation
12. **Scale Detection** (`agent_scale_detection_mixin.py`) — Large-scale list-processing task detection and advisory injection
13. **Scale Loop** (`agent_scale_loop_mixin.py`) — Per-item batch processing with constant-context isolation

### Tool Ecosystem (40+ Tools)

Tools are organized into functional categories:

**File & Text Operations:**
- `read.py` — Safe file reading with path resolution across multiple contexts
- `write.py` — Sandboxed file writing with session-based scoping
- `edit.py` — Surgical file editing with backup/undo capability
- `glob.py` — Pattern-based file discovery with case-insensitive matching

**Web & Data Integration:**
- `web_fetch.py` / `web_get.py` — HTTP content retrieval with text extraction or raw HTML
- `web_search.py` — Brave Search API integration for real-time web queries
- `google_drive.py` — Google Drive file operations with OAuth authentication
- `google_mail.py` — Read-only Gmail access with MIME parsing
- `google_calendar.py` — Calendar event management via REST API
- `gws.py` — Google Workspace CLI wrapper for Drive/Docs/Sheets/Slides/Gmail
- `typesense.py` — Vector search and document indexing via Typesense
- `datastore.py` — Relational database operations with protection rules

**Document Processing:**
- `document_extract.py` — Multi-format extraction (PDF, DOCX, XLSX, PPTX) to Markdown
- `image_ocr.py` / `image_gen.py` — OCR and image generation via vision-capable LLMs
- `summarize_files.py` — Batch file summarization with map-reduce pattern

**Browser Automation:**
- `browser.py` — Playwright-based browser with persistent sessions, credential management, workflow recording
- `pinchtab.py` — Token-efficient accessibility tree-based browser automation
- `browser_accessibility.py` — Semantic page structure extraction
- `browser_session.py` — Stateful browser instance management
- `browser_workflow.py` — Record-and-replay workflow automation
- `browser_api_replay.py` — Direct API execution from captured network traffic
- `browser_credentials.py` — Encrypted credential storage for automated login
- `browser_network.py` — Network traffic interception and API pattern discovery
- `browser_vision.py` — Vision-based page analysis

**System & Hardware:**
- `shell.py` — Secure shell command execution with timeout management
- `desktop_action.py` — Cross-platform GUI automation (mouse, keyboard, application launching)
- `screen_capture.py` — Screenshot capture with optional vision analysis
- `clipboard.py` — macOS clipboard read/write operations
- `termux.py` — Android device hardware control via Termux API
- `stt.py` — Speech-to-text with multi-provider support (Soniox, OpenAI, Gemini)
- `pocket_tts.py` — Local text-to-speech synthesis

**Productivity & Context:**
- `todo.py` — Cross-session task management with priority/responsibility tracking
- `contacts.py` — Address book with importance scoring and privacy tiers
- `scripts.py` — Script registry with usage tracking
- `apis.py` — API endpoint management with authentication
- `personality.py` — Agent/user personality profile management
- `playbooks.py` — Reusable task pattern library with LLM-based distillation
- `direct_api.py` — Direct HTTP API call management and execution
- `send_mail.py` — Email dispatch via Mailgun/SendGrid/SMTP

**Specialized:**
- `botport.py` — Distributed agent coordination via BotPort network
- `skills.py` — Modular skill discovery, installation, and invocation

### Session & Memory Management

**Session Layer** (`session.py`):
- SQLite-backed persistence for conversations, tasks, contacts, scripts, APIs, playbooks, workflows, credentials
- Message history with tool call metadata and token counting
- Cron job scheduling with execution history
- Cross-session state management via app_state table

**Memory Layers**:
1. **Working Memory** (`memory.py`) — In-turn context buffer with automatic compaction
2. **Semantic Memory** (`semantic_memory.py`) — SQLite FTS5 + vector embeddings with hybrid search
3. **Deep Memory** (`deep_memory.py`) — Typesense-backed long-term archive with chunking and embedding

**File Registry** (`file_registry.py`) — Logical-to-physical path mapping for cross-task artifact discovery

### LLM Provider Abstraction

**Multi-Provider Support** (`llm/__init__.py`):
- **Ollama** — Direct HTTP client for local models
- **OpenAI/ChatGPT** — Standard API + ChatGPT Responses API (SSE streaming)
- **Anthropic Claude** — With prompt caching support
- **Google Gemini** — Via LiteLLM with async/sync handling
- **xAI Grok** — Via LiteLLM

**Features**:
- Token rate limiting with sliding-window backpressure
- Provider-specific message/tool conversion
- Unified tool definition schema
- Streaming response collection
- Token counting and usage tracking

### Web UI Infrastructure

**Core Server** (`web_server.py`):
- aiohttp-based async web server
- 100+ HTTP/WebSocket route handlers
- Real-time callback routing to connected clients
- Multi-session state management
- Third-party integration orchestration

**WebSocket Communication**:
- `ws_handler.py` — Chat message routing and session state sync
- `ws_stt.py` — Live speech-to-text streaming
- `chat_handler.py` — Agent execution with concurrent task naming

**REST API Modules** (50+ endpoints):
- Session management (list, create, switch, export)
- Entity CRUD (todos, contacts, scripts, APIs, playbooks)
- Datastore operations (table/row management, import/export)
- Cron job scheduling and execution
- File browsing and media serving
- Configuration management with hot-reload
- Orchestrator control and workflow management
- Deep memory search and indexing
- Visualization style management
- OAuth authentication flows

**Static Pages** (`static_pages.py`):
- 23 HTML page templates with cache-busting
- Chat, orchestration, workflows, memory, settings, sessions, datastore, playbooks, skills, etc.

### Platform Integrations

**Telegram** (`telegram.py`):
- Per-user Agent instances with session isolation
- User approval workflow with pairing tokens
- Typing indicators and inline keyboards for next steps
- Image upload/download support
- Slash command execution
- Cron job management per-user

**Discord** (`discord_bridge.py`):
- DM-based polling interface
- Message normalization and bot mention detection
- Audio file upload support

**Slack** (`slack_bridge.py`):
- DM-first polling with pagination
- User caching and username resolution
- Thread reply support

**Google OAuth** (`google_oauth.py`, `google_oauth_manager.py`):
- PKCE-based authorization flow
- Token lifecycle management
- Credential injection into Vertex AI provider

**BotPort** (`botport_client.py`):
- Distributed agent coordination
- Concern-based task delegation
- Multi-hop agent communication

**Hotkey Daemon** (`hotkey_daemon.py`):
- Global keyboard listener for voice activation
- Double/triple-tap state machine
- Screenshot capture on demand
- Clipboard text selection detection

### Configuration & Utilities

**Configuration** (`config.py`):
- Pydantic v2 with nested models
- YAML persistence with local/home precedence
- Environment variable overrides for secrets
- 30+ subsystems (tools, skills, guards, memory, UI platforms, etc.)

**Logging** (`logging.py`):
- structlog-based structured logging
- Dynamic sink routing to TUI system panel
- Fallback to stderr

**Cron System** (`cron.py`, `cron_dispatch.py`):
- Human-readable schedule parsing (interval, daily, weekly)
- Job execution with trigger tracking
- History persistence

**Instruction Management** (`instructions.py`):
- Two-tier directory system (system defaults + personal overrides)
- Markdown template rendering with partial substitution
- Micro-template variants for context-specific prompts

**Personality System** (`personality.py`):
- Agent and per-user personality profiles
- Markdown-based configuration
- Prompt block injection

**Visualization Styles** (`visualization_style.py`):
- Design preference management
- LLM-powered style extraction from images/documents
- Cache invalidation on updates

**Datastore** (`datastore.py`):
- SQLite-backed relational database
- Multi-format import/export (CSV, XLSX, JSON)
- Granular protection system (table/column/row/cell)
- Type inference and schema evolution

**Orchestration** (`session_orchestrator.py`):
- DAG-based task decomposition and execution
- Parallel task activation with traffic-light gating
- Timeout management with warning/grace/restart flow
- Workflow persistence and templating
- Cross-task file registry for artifact sharing

**Task Graph** (`task_graph.py`):
- Topological sorting for dependency resolution
- Concurrency control via traffic lights
- Timeout/retry management
- Cascade failure handling

**Skills System** (`skills.py`):
- Skill discovery from multiple sources
- GitHub-based installation
- Dependency management (brew, npm, go, uv)
- LLM-based skill ranking

**File Tree Builder** (`file_tree_builder.py`):
- Local directory tree generation
- Google Drive folder browsing
- Caching with TTL

**Next Steps** (`next_steps.py`):
- LLM-based extraction of suggested actions
- Heuristic pre-filtering to avoid unnecessary LLM calls
- UI-friendly button generation

**Onboarding** (`onboarding.py`):
- Interactive setup wizard
- Provider validation
- Configuration persistence

**Reflections** (`reflections.py`):
- Self-improvement pattern extraction
- LLM-driven session analysis
- Playbook proposal generation

**Session Export** (`session_export.py`):
- Multi-format session history export (Markdown, JSONL)
- Pipeline trace collection and summarization

**Platform Lifecycle** (`platform_lifecycle.py`):
- Telegram/Slack/Discord bridge initialization
- Background polling loop management
- Graceful shutdown coordination

**Remote Command Handler** (`remote_command_handler.py`):
- 50+ slash command implementations
- Platform-agnostic command routing
- Entity CRUD via chat interface

**Prompt Execution** (`prompt_execution.py`):
- Queue-based task management
- Multi-lane priority system
- Follow-up prompt deduplication and consolidation

**Agent Pool** (`agent_pool.py`):
- Worker agent lifecycle management
- Idle eviction and capacity-based culling
- Per-session creation locks
- Shared resource caching

**Runtime Context** (`runtime_context.py`):
- Dependency injection container
- Shared state across modules
- Platform-specific state management

**CLI** (`cli.py`):
- Terminal UI abstraction
- Split-pane monitoring view
- Readline history integration
- Special command parsing

**Main Entry Point** (`main.py`):
- Argument parsing and configuration loading
- Interactive TUI vs. web server routing
- Onboarding wizard execution
- Signal handling and graceful shutdown

## Key Design Patterns

### 1. Mixin-Based Composition
The Agent class uses 13 mixins to provide distinct capabilities without deep inheritance hierarchies. Each mixin focuses on a specific concern (orchestration, tools, context, memory, etc.), enabling modular testing and feature toggling.

### 2. Callback-Driven Architecture
Agent execution feeds events through registered callbacks (status, thinking, tool_output, approval) that route to UI layers without tight coupling. Enables real-time monitoring and multi-client synchronization.

### 3. Async-First Design
All I/O operations use asyncio with non-blocking patterns. Long-running operations (LLM calls, file I/O, network requests) run in thread pools to prevent event loop blocking.

### 4. Token-Aware Context Management
The system tracks token consumption at multiple levels (message, turn, session) and implements intelligent context compaction, chunking, and message selection to stay within LLM context windows.

### 5. Multi-Layer Memory
Working memory (in-turn), semantic memory (session-scoped), and deep memory (long-term archive) provide different retrieval patterns optimized for recency, relevance, and scale.

### 6. Guard Rails & Approval Workflows
Input/output guards with configurable levels (stop_suspicious, ask_for_approval) and tool execution approval callbacks enable safe autonomous operation with human oversight.

### 7. Scale Detection & Micro-Loops
Automatic detection of large-scale list-processing tasks triggers a specialized micro-loop that processes items one-at-a-time with constant context, preventing context window overflow.

### 8. Orchestration & Parallelization
SessionOrchestrator decomposes complex requests into DAGs, executes tasks in parallel with dependency constraints, and manages timeout/retry policies for resilient multi-step workflows.

### 9. File Registry & Cross-Task Sharing
FileRegistry maps logical paths to physical locations, enabling downstream tasks to discover and reference upstream artifacts without knowledge of session IDs or directory structures.

### 10. Configuration Hierarchy
Environment variables → .env file → config.yaml (home) → config.yaml (local) → hardcoded defaults provide flexible configuration with security-sensitive overrides.

## Notable Technical Achievements

1. **Token-Efficient Browser Automation** — PinchTab uses accessibility trees (~800 tokens) instead of screenshots (~2K+ tokens), achieving 2-3x token efficiency for web automation.

2. **Chunked Processing Pipeline** — Automatically detects context overflow and splits large documents into chunks, processes independently, and combines results via LLM synthesis.

3. **Dual-Mode Orchestration** — Supports both fast "loop" mode (direct tool execution) and "contracts" mode (planner + critic validation) for different task complexity levels.

4. **Intelligent Scale Detection** — Automatically detects large-scale tasks and switches to per-item processing to prevent context explosion, with automatic list extraction after content fetch.

5. **Multi-Provider LLM Abstraction** — Unified interface supporting Ollama, OpenAI, Anthropic, Gemini, xAI with provider-specific quirks (Anthropic caching, Gemini streaming issues) handled transparently.

6. **Hybrid Memory Search** — Combines full-text search (BM25) with vector embeddings and temporal decay scoring for intelligent context retrieval across multiple timescales.

7. **Graceful Degradation** — Every failure point has a fallback (chunk LLM failure → skip chunk; combine overflow → concatenate; vision failure → return screenshot path; Soniox unavailable →