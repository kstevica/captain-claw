# Summary: agent_context_mixin.py

# agent_context_mixin.py

## Summary

This is a comprehensive context assembly and message building system for an AI agent. It handles dynamic system prompt construction, semantic memory integration, cross-session context resolution, tool management, and intelligent message selection within token budgets. The mixin provides sophisticated context injection mechanisms including to-do items, contacts, scripts, APIs, datastore tables, and deep memory search capabilities, while maintaining strict provider-specific message ordering constraints.

## Purpose

Solves the core problem of assembling coherent, token-efficient LLM conversation contexts by:
- Building dynamic system prompts with conditional tool sections and file listings
- Managing multiple memory layers (semantic, deep/archive, cross-session)
- Auto-capturing metadata (todos, contacts, scripts, APIs) from conversations
- Intelligently selecting historical messages within token budgets
- Normalizing messages for provider-specific constraints (OpenAI, Anthropic)
- Injecting contextual notes (planning, workspace state, task progress)

## Most Important Functions/Classes/Procedures

### 1. **`_build_messages()`** (Core Message Assembly)
Constructs the final message list for LLM calls by:
- Building system prompt with token accounting
- Collecting candidate messages from session history
- Filtering historical tool messages and skipping monitor-only tools
- Injecting context notes (memory, semantic, deep, cross-session, planning, todos, contacts, scripts, APIs, datastore)
- Selecting messages in reverse chronological order within token budget
- Normalizing for provider constraints
- Returns `list[Message]` with detailed context window metrics

### 2. **`_build_system_prompt()`** (Dynamic Prompt Construction)
Assembles the complete system prompt by:
- Loading base template and rendering with variables
- Injecting personality block (agent identity) and user context block (who agent talks to)
- Adding visualization style, self-reflection, and system info blocks
- Building extra read directories block with local/GDrive folder listings and file trees
- Conditionally including tool-specific sections (browser policy, direct API, termux, gws, datastore)
- Building tool list (full or micro format)
- Collapsing excess newlines and appending skills section
- Returns complete system prompt string

### 3. **`_normalize_selected_messages_for_provider()`** (Provider Compliance)
Ensures message sequences comply with strict provider rules (OpenAI, Anthropic):
- Tracks pending tool_call IDs and validates tool response chains
- Strips orphaned tool_calls from assistant messages
- Converts orphan tool messages to assistant context
- For Anthropic: ensures conversation ends with user message (no prefill)
- Returns normalized message list maintaining semantic integrity

### 4. **`_initialize_layered_memory()`** (Memory System Setup)
Initializes multi-layer memory architecture:
- Creates semantic memory manager (SQLite-backed) for session/workspace retrieval
- Initializes deep memory (Typesense-backed archive) for long-term indexed search
- Sets active session and schedules background sync
- Gracefully degrades if components unavailable
- Enables cross-session and deep memory context injection

### 5. **Auto-Capture Methods** (Metadata Extraction)
Pattern-based extraction from conversations:
- **`_auto_capture_todos()`** — extracts task items from user/assistant messages via regex patterns
- **`_auto_capture_contacts()`** — captures contact info from "remember that" patterns and send_mail tool usage
- **`_auto_capture_scripts()`** — tracks scripts from write tool calls and conversation mentions
- **`_auto_capture_apis()`** — extracts API endpoints from web_fetch usage and conversation patterns
- All methods integrate with session manager for persistence

### 6. **Context Note Builders** (Contextual Injection)
Construct specialized context blocks:
- **`_build_semantic_memory_note()`** — retrieves relevant past session snippets via semantic search
- **`_build_deep_memory_note()`** — searches Typesense archive for long-term indexed content
- **`_resolve_cross_session_context()`** — combines direct output extraction + semantic search from referenced sessions
- **`_build_tool_memory_note()`** — compacts historical tool outputs with term-overlap ranking
- **`_build_todo_context_note()`** — formats active to-do items with priority/responsible party
- **`_build_contacts_context_note()`** — injects relevant contacts when names mentioned in query
- **`_build_scripts_context_note()`** / **`_build_apis_context_note()`** — similar mention-based injection

### 7. **Tool Registration & Discovery** (Tool Management)
Manages tool lifecycle:
- **`_register_default_tools()`** — registers 40+ built-in tools from config, handles conditional registration
- **`_discover_plugin_tool_files()`** — scans configured plugin directories for custom tool Python files
- **`_register_plugin_tools()`** — dynamically loads plugin modules, instantiates Tool subclasses, registers with metadata
- **`reload_tools()`** — syncs registry with config changes, unregisters removed tools, re-registers updated ones

### 8. **URL & Domain Extraction** (Source Tracking)
Utilities for managing source URLs in context:
- **`_extract_urls()`** — regex-based URL extraction with deduplication
- **`_extract_source_links()`** — extracts URLs from tool arguments/content, skips web_fetch/web_get content bodies
- **`_extract_mentioned_domains()`** — parses domains from URLs and bare domain tokens
- **`_collect_recent_source_urls()`** — gathers recent URLs with optional domain filtering to prevent context pollution

### 9. **Tool Policy Management** (Access Control)
Handles tool permission policies:
- **`_normalize_tool_policy_payload()`** — validates and normalizes allow/deny/also_allow lists
- **`_session_tool_policy_payload()`** — loads session-level tool policies from metadata
- **`_active_task_tool_policy_payload()`** — extracts task-specific policies from planning pipeline
- Integrates with tool registry for enforcement

### 10. **Message Normalization** (Serialization)
Converts between formats:
- **`_normalize_session_tool_calls()`** — converts persisted tool_calls to OpenAI-compatible shape
- **`_serialize_tool_calls_for_session()`** — serializes ToolCall objects to JSON-safe dicts for persistence
- **`_ensure_user_message_last()`** — converts trailing assistant context to user role for Anthropic compatibility

## Architecture & Dependencies

**Key Dependencies:**
- `captain_claw.config` — configuration management
- `captain_claw.llm` — Message, ToolCall, provider management
- `captain_claw.memory` — semantic memory (SQLite)
- `captain_claw.deep_memory` — Typesense archive indexing
- `captain_claw.tools.registry` — tool registration system
- `captain_claw.session_manager` — session/todo/contact/script/API persistence
- `captain_claw.instructions` — template rendering for system prompts
- `captain_claw.personality` — agent/user personality profiles
- `captain_claw.file_tree_builder` — file listing generation

**Data Flow:**
1. User query → extract session references, domains, semantic signals
2. Build candidate messages from session history + context notes
3. Select messages within token budget (reverse chronological, must-include rules)
4. Normalize for provider constraints
5. Return Message list to LLM provider

**Role in System:**
This mixin is the **context assembly engine** — it bridges the gap between raw conversation history and what the LLM actually sees. It's responsible for intelligent context pruning, memory integration, metadata auto-capture, and provider compliance, enabling the agent to maintain coherent long-running sessions with limited token budgets.