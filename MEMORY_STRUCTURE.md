# Captain Claw — Memory System Architecture

A comprehensive reference for Captain Claw's memory, insights, reflections, and layered retrieval systems.

---

## Table of Contents

- [1. Architecture Overview](#1-architecture-overview)
- [2. Working Memory](#2-working-memory)
- [3. Semantic Memory](#3-semantic-memory)
- [4. Deep Memory](#4-deep-memory)
- [5. Datastore](#5-datastore)
- [6. L1, L2, L3 Memory Layers](#6-l1-l2-l3-memory-layers)
- [7. Insights System](#7-insights-system)
- [8. Reflections System](#8-reflections-system)
- [9. Memory Lifecycle Management](#9-memory-lifecycle-management)
- [10. Session Isolation](#10-session-isolation)
- [11. Configuration Reference](#11-configuration-reference)
- [12. Key Files Reference](#12-key-files-reference)

---

## 1. Architecture Overview

Captain Claw implements a **three-layer memory architecture** with two additional learning mechanisms (insights and reflections), unified via the `LayeredMemory` facade.

| Layer | Backend | Purpose | Persistence |
|-------|---------|---------|-------------|
| **Working Memory** | In-memory | Current conversation context | Serialized to session DB |
| **Semantic Memory** | SQLite (FTS5 + embeddings) | Workspace files + session history | `~/.captain-claw/memory.db` |
| **Deep Memory** | Typesense | Long-term searchable archive | External Typesense service |
| **Datastore** | SQLite | User-managed relational data | Configurable path |
| **Insights** | SQLite | Extracted factual knowledge | `~/.captain-claw/insights.db` |
| **Reflections** | Markdown files | Meta-cognitive self-improvement | `~/.captain-claw/reflections/` |

```
                    ┌──────────────────────────────┐
                    │        System Prompt          │
                    │  {reflection_block}           │  ← Latest reflection
                    │  {insights_block}             │  ← Insights tool info
                    │  insights_note (context)      │  ← Top 8 relevant insights
                    │  semantic_note (context)       │  ← Semantic memory matches
                    └──────────────────────────────┘
                              ↑
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
   Reflections           Insights              Semantic Memory
   (meta-learning)    (factual knowledge)    (hybrid search index)
   ~/.captain-claw/   insights.db            memory.db
   reflections/       ├─ auto-extracted      ├─ workspace files
                      ├─ deduped             ├─ session history
                      └─ categorized         └─ embeddings + FTS5
```

The `LayeredMemory` facade in `memory.py` exposes:

```python
LayeredMemory:
  .working: WorkingMemory
  .semantic: SemanticMemoryIndex | None
  .deep: DeepMemoryIndex | None
  .active_session_id: str | None
```

Key methods: `set_active_session()`, `record_message()`, `compact_working_memory()`, `search_semantic()`, `build_semantic_note()`, `promote()`, `clear_all()`.

---

## 2. Working Memory

**File:** `memory.py` — `WorkingMemory` class

Short-term, in-memory storage for the current conversation turn.

**Data Structure:**

```python
WorkingMemorySnapshot:
  - summary: str        # LLM-generated summary of older messages
  - messages: list[dict] # role, content, tool_name, etc.
```

**Behavior:**

- Max budget: **100K tokens** (configurable)
- Messages added via `record_message(role, content)`
- **Compaction**: When usage hits 80% of max_tokens, older messages are summarized by the LLM and replaced with a summary message, keeping ~40% recent messages
- Token counting uses provider-aware methods with fallback
- Serialized to the session database during persistence

**Key methods:**
- `record_message(role, content)` — add a message
- `snapshot(include_summary_message=True)` — get current state
- `clear()` — erase all messages and summary

---

## 3. Semantic Memory

**File:** `semantic_memory.py` — `SemanticMemoryIndex` class
**Storage:** SQLite at `~/.captain-claw/memory.db` (WAL mode)

Medium-term persistent memory with hybrid BM25 keyword + vector similarity search.

### Database Schema

**`memory_documents`** — Document metadata:
- `doc_id`, `source`, `reference`, `path`, `signature`, `updated_at`

**`memory_chunks`** — Text chunks with layered summaries:
- `chunk_id` (PK), `doc_id`, `source`, `reference`, `path`
- `chunk_index`, `start_line`, `end_line`
- `text` (L3 full text), `text_l1` (headline), `text_l2` (summary)
- `updated_at`

**`memory_chunks_fts`** — FTS5 virtual table indexing all three layers:
```sql
CREATE VIRTUAL TABLE memory_chunks_fts
USING fts5(chunk_id, text, text_l1, text_l2, path, source, reference)
```

**`memory_embeddings`** — Vector embeddings per chunk:
- `chunk_id`, `provider_key`, `dims`, `embedding`, `updated_at`

**`memory_sync_state`** — Sync metadata

### Indexing

- **Workspace indexing**: `_collect_workspace_documents()` scans `workspace_path` up to 400 files (max 256KB each)
- **Session indexing**: `_collect_session_documents()` extracts from session database (one doc per session with all messages)
- **Manual upsert**: `upsert_text(source, reference, path, text, updated_at)` for ad-hoc memory
- **Chunking**: Line-based, ~1,400 chars per chunk, 200-char overlap between consecutive chunks
- **Embeddings**: Generated by embedding chain (LiteLLM, Ollama, or local hash fallback)
- **Background sync**: Worker thread runs `_sync_once()` when scheduled, compares file signatures to detect changes

### Hybrid Retrieval

1. **Keyword path**: Tokenize query → FTS5 BM25 ranking (fallback to LIKE if FTS fails)
2. **Vector path**: Embedding chain encodes query → cosine similarity on L2-normalized vectors
3. **Merge**: `0.65 × vector_score + 0.35 × text_score` (configurable weights)
4. **Temporal decay**: `score *= exp(-lambda × age_days)` with 21-day half-life
5. **Filtering**: Min score threshold (0.1), max results (6), candidate limit (80)
6. **Result caching**: 45-second TTL
7. **Auto-sync**: Triggers background sync if data is stale (120 seconds)

### Embedding Chain (`_EmbeddingProviderChain`)

- **Provider priority**: LiteLLM → Ollama → local hash fallback
- **Models**: `text-embedding-3-small` (LiteLLM), `nomic-embed-text` (Ollama)
- **Normalization**: All vectors L2-normalized
- **Timeout**: 4 seconds per request

### Result Object

```python
SemanticMemoryResult:
  - chunk_id, source, reference, path
  - start_line, end_line
  - snippet: str        # Text at selected layer
  - score: float        # Combined score
  - text_score: float   # BM25 ranking
  - vector_score: float # Cosine similarity
  - text_l1, text_l2: str
  - updated_at: str
```

---

## 4. Deep Memory

**File:** `deep_memory.py` — `DeepMemoryIndex` class
**Storage:** Typesense collection (external service)

Long-term searchable archive for past content with hybrid BM25 + vector search.

### Collection Schema

```python
fields: [
    doc_id, source, reference, path,
    text, text_l1, text_l2,
    chunk_index, start_line, end_line,
    tags,          # Faceted filtering
    updated_at,
    embedding      # float array (1536 dims default)
]
```

Facets: `source`, `reference`, `tags`

### Indexing

- **Manual**: `index_document(doc_id, text, source, reference, tags)` or `index_batch()`
- **Scale loop integration**: Micro-loop `no_file` sink can auto-index
- **Typesense tool**: LLM-callable tool for manual indexing
- **Chunking**: Same algorithm as semantic memory
- **Upsert**: JSONL import to Typesense via HTTP

### Retrieval

- `search(query, max_results, filter_by, vector_query)` — hybrid search
- Auto-generates vector query if embedding chain available
- BM25 + optional vector similarity
- Facet filtering by source/tags
- Group by `doc_id` for unique documents

### Deletion

- `delete_document(doc_id)` — delete all chunks for a document
- `delete_by_filter(filter_by)` — Typesense filter-based deletion
- `clear_all()` — drop and recreate collection

---

## 5. Datastore

**File:** `datastore.py`

A separate relational system for structured, user-managed data. Not strictly a memory layer, but related to persistent data storage.

- User-defined schemas with typed columns: text, integer, real, boolean, date, datetime, json
- **Protection rules** at table/column/row/cell level
- Atomic writes via aiosqlite, WAL mode for concurrent access
- LLM-callable tool: `tools/datastore.py`
- `delete_rows(table, where)` for targeted deletion

### Session Isolation

When `public_run == "computer"`, each session gets its own datastore database.

---

## 6. L1, L2, L3 Memory Layers

A **three-tier hierarchical text representation** for each memory chunk, enabling progressive detail disclosure during retrieval.

### Layer Definitions

| Layer | Name | Max Size | Content |
|-------|------|----------|---------|
| **L1** | One-liner headline | ~100 chars | Ultra-condensed core idea |
| **L2** | Contextual summary | ~300 chars | 1-2 sentence summary for relevance assessment |
| **L3** | Full text | ~1,400 chars | Complete chunk text as stored |

### How L1/L2 Are Generated

The summarizer in `agent_context_mixin.py` sends each chunk to the LLM:

```
You are a memory indexer. Given the following text, produce exactly two lines:
Line 1: A one-liner headline (max 100 chars) capturing the core idea.
Line 2: A 1-2 sentence summary (max 300 chars) with enough context to assess relevance.

Rules:
- Output ONLY the two lines, nothing else.
- No labels, prefixes, or numbering.
```

- `temperature=0.0` (deterministic), `max_tokens=200`
- Response split on newline: line 1 → L1, line 2 → L2
- Post-processing: L1 truncated to 120 chars, L2 to 400 chars
- **Fallback** if LLM fails: L1 = first line of text (120 chars), L2 = first 300 chars
- Token usage tracked under label `memory_summarize_chunk`
- Batched: `summary_batch_size` chunks per batch (default 10)

### Layer Selection & Fallback

Both semantic and deep memory implement identical fallback logic:

```python
def _pick_layer_text(layer, *, text, text_l1, text_l2):
    if layer == "l1":
        return text_l1 or text_l2 or text   # L1 → L2 → L3
    if layer == "l2":
        return text_l2 or text              # L2 → L3
    return text                             # L3 always available
```

If L1/L2 haven't been generated yet (e.g., `layered_summaries=False`), the system gracefully degrades to full text.

### Context Note Building

When building context for the LLM prompt:

1. **Default layer is L2** — both `_build_semantic_memory_note()` and `_build_deep_memory_note()` default to `layer="l2"`
2. Search results formatted as markdown with citations:
   ```
   Semantic memory matches (all sessions + workspace):
   - [source] path:line_number (score=0.XXX) [L2 summary snippet]
   ```
3. Snippets truncated to `max_snippet_chars` (360 for semantic, 400 for deep)

**Key point:** Layer selection only affects display — **scoring always happens on L3 full text** (BM25 + vector similarity). This ensures accurate relevance ranking with compact output.

### The Promote Mechanism

Enables drill-down from compact summaries to full detail:

```python
def promote(self, chunk_ids: list[str], layer: str = "l3") -> list[SemanticMemoryResult]:
    """Fetch specific chunks at the requested detail layer.
    Use after an L1/L2 search to expand interesting hits to full detail."""
```

**Workflow:**
1. Search at L2 → scan results quickly
2. Identify interesting chunk IDs
3. Call `promote([chunk_id1, chunk_id2], layer="l3")` → get full text
4. Agent uses full context for detailed analysis

REST API: `GET /api/semantic-memory/promote?ids=id1,id2&layer=l3`

### End-to-End Flow

```
Document indexed
  ↓
Chunked (~1400 chars, 200 overlap)
  ↓
L3 stored immediately
  ↓
Background: LLM generates L1 + L2 per chunk
  ↓
All three layers stored in DB + FTS5 indexed
  ↓
Search query arrives
  ↓
Scoring on L3 (BM25 + vector + temporal decay)
  ↓
Display at requested layer (default L2)
  ↓
Optional: promote(chunk_ids) for L3 full text
```

### Configuration

```python
layered_summaries: bool = True    # Enable/disable L1/L2 generation
summary_batch_size: int = 10      # Chunks per summarization batch
```

When `layered_summaries=False`: chunks are indexed normally, L1/L2 remain empty strings, layer selection falls back to L3 — zero overhead.

---

## 7. Insights System

**File:** `insights.py` — `InsightsManager` class
**Storage:** SQLite at `~/.captain-claw/insights.db` with FTS5 full-text search

Automatically extracts and stores factual knowledge (contacts, decisions, preferences, deadlines) from conversations and tool outputs.

### Database Schema

```sql
CREATE TABLE insights (
    id TEXT PRIMARY KEY,          -- 12-char UUID
    content TEXT NOT NULL,        -- 1-2 sentence fact
    category TEXT NOT NULL,       -- contact, decision, preference, fact, deadline, project, workflow
    entity_key TEXT,              -- Dedup key (e.g., "contact:john@example.com")
    importance INTEGER DEFAULT 5, -- 1-10 scale (5=useful, 8=important, 10=critical)
    source_tool TEXT,             -- Which tool extracted it (e.g., "gws:mail_read")
    source_session TEXT,          -- Session ID that created it
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT,              -- Optional TTL
    tags TEXT                     -- Comma-separated metadata
)
```

FTS5 virtual table `insights_fts` enables semantic search + prefix matching.

### Automatic Extraction

**Triggers:**
1. **Periodic**: After every 8 messages (configurable), with 60-second cooldown
2. **Tool-specific**: After `gws:mail_read` or `gws:mail_read_thread` tool calls
3. **Manual**: Via insights tool or slash command

**Flow (`extract_insights`):**
1. Gathers last 15 messages from current conversation
2. Loads last 20 existing insights to prevent re-extraction
3. Calls LLM with `insight_extraction_system_prompt.md` + `insight_extraction_user_prompt.md`
4. Parses JSON response (max 5 insights per extraction)
5. **Deduplication**:
   - Entity key exact match → updates existing insight
   - FTS similarity check (BM25 rank threshold -8.0) → skips if too similar
6. Records LLM usage metrics
7. Opportunistically prunes expired insights
8. Runs as a **non-blocking background asyncio task**

### Prompt Integration

- **`insights_block`**: In system prompt — tells agent the insights tool is available
- **`insights_note`**: Per-turn context message with top 8 insights (ranked by importance + recency)
- **Cache**: `_refresh_insights_context_cache()` called during agent initialization

### Tool Interface

**File:** `tools/insights.py`
Actions: `search`, `list`, `add`, `update`, `delete`
Supports category filtering, importance scoring, entity key dedup, tags.

### REST API

**File:** `web/rest_insights.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/insights` | GET | List or search insights |
| `/api/insights/{id}` | GET | Get single insight |
| `/api/insights` | POST | Create new insight |
| `/api/insights/{id}` | PATCH | Update insight |
| `/api/insights/{id}` | DELETE | Delete insight |

### Configuration (`InsightsConfig`)

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Master toggle |
| `auto_extract` | `true` | Enable automatic extraction |
| `inject_in_context` | `true` | Show in system prompt |
| `max_items_in_prompt` | 8 | Max insights per context |
| `extraction_interval_messages` | 8 | Trigger every N messages |
| `extraction_cooldown_seconds` | 60 | Min time between extractions |
| `max_insights` | 500 | Storage limit |
| `db_path` | `~/.captain-claw/insights.db` | Database location |

---

## 8. Reflections System

**File:** `reflections.py`
**Storage:** Markdown files in `~/.captain-claw/reflections/`

Generates **self-improvement instructions** by analyzing conversation patterns, task outcomes, and prior reflections — enabling iterative meta-learning.

### Data Structure

```python
@dataclass
class Reflection:
    timestamp: str          # ISO 8601 creation time
    summary: str            # Self-improvement instructions (actionable principles)
    topics_reviewed: list   # What was analyzed
    token_usage: dict       # LLM token metrics
```

Stored as timestamped markdown files. Only the **latest** reflection is active.

### Markdown Format

```markdown
# Reflection

## Timestamp
2026-03-20T17:30:45+01:00

## Summary
[Actionable self-improvement bullet points in second person]

## Topics Reviewed
- recent conversation messages
- memory facts
- completed tasks/cron jobs

## Token Usage
- prompt_tokens: 450
- completion_tokens: 320
- total_tokens: 770
```

### Generation (`generate_reflection`)

**Context gathering:**
1. Load previous reflection (for comparison/iteration)
2. Gather recent messages (last 20 from current session)
3. Gather memory facts (if available via session manager)
4. Gather task/cron history since last reflection

**LLM call:**
- System prompt: `reflection_system_prompt.md`
- User prompt: `reflection_user_prompt.md`
- `max_tokens=1500`, labeled `"reflection"`

**Output rules:**
- Second person ("You should...", "Continue to...")
- Generalized — no specific task/session references
- Max 15 bullet points, no markdown headers

### Auto-Reflection Trigger (`maybe_auto_reflect`)

Conditions (all must be true):
- At least **10 messages** in current session
- At least **4 hours** (14,400s) since last reflection

Spawned as a non-blocking background asyncio task after agent turn completes.

### Prompt Integration

- Loaded via `load_latest_reflection()` with mtime-based caching
- Rendered via `reflection_to_prompt_block()`
- Injected into system prompt at `{reflection_block}` placeholder
- Only the newest file is used — keeps prompt lean

### REST API & Slash Commands

**File:** `web/rest_reflections.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/reflections` | GET | List all reflections |
| `/api/reflections/latest` | GET | Get active reflection |
| `/api/reflections/generate` | POST | Manually trigger generation |
| `/api/reflections/{timestamp}` | PUT | Update summary/topics |
| `/api/reflections/{timestamp}` | DELETE | Delete reflection |

Slash commands: `/reflection generate`, `/reflection list`, `/reflection latest`

---

## 9. Memory Lifecycle Management

### Initialization

1. Agent calls `_initialize_layered_memory()` on startup
2. Creates `WorkingMemory` (always)
3. Creates `SemanticMemoryIndex` if enabled in config
4. Creates `DeepMemoryIndex` if Typesense configured
5. Sets active session via `set_active_session()`
6. Loads insights cache via `_refresh_insights_context_cache()`
7. Loads latest reflection via `load_latest_reflection()`

### During Execution

1. Messages recorded to working memory via `record_message()`
2. Semantic sync scheduled (non-blocking background thread)
3. On search: auto-sync triggered if stale (120s)
4. Working memory compacted when ratio threshold hit (80% → keep 40%)
5. Insights extracted periodically (every 8 messages, 60s cooldown)
6. Insights extracted after specific tool calls

### Session Switch

1. Load new session via session manager
2. Call `memory.set_active_session(new_session_id)`
3. Clear search cache
4. Trigger background sync to index new session

### Post-Turn

1. Auto-reflection triggered if conditions met (10+ messages, 4+ hours)
2. Insights extraction triggered if interval met
3. Both run as non-blocking background tasks

### Shutdown

1. Graceful close: `memory.close()`
2. Semantic memory closes SQLite connection
3. Deep memory closes HTTP client
4. Working memory cleared

---

## 10. Session Isolation

### Standard Mode

- Each session has its own message history in the session DB
- Semantic memory indexes by `reference = session_id`
- `set_active_session()` restricts retrieval to active session + workspace
- **Cross-session retrieval** is opt-in via config

### Public Computer Mode (`public_run == "computer"`)

When enabled, multi-tenant safety is enforced:

| System | Isolation |
|--------|-----------|
| **Insights** | Per-session database: `insights_{session_id}.db` |
| **Datastore** | Per-session database |
| **Reflections** | Global (shared across all public users) |
| **Semantic Memory** | Session-scoped search (unless cross-session enabled) |

Manager resolution via `_resolve_manager()` checks `public_run` and `session_id`.

### Cross-Session Retrieval

```python
cross_session_retrieval: bool = False  # Config flag
```

- **Disabled** (default): Search filters to active session + workspace only
- **Enabled**: Search across all sessions + workspace
- `search_in_session(query, session_reference)` bypasses config for targeted queries

### Session Metadata

- Compaction history: `session.metadata["compaction"]`
- Memory protection flag: `session.metadata["memory_protection"]["enabled"]`
- Subagent memory: child session references parent in metadata
- File registry: orchestrator maps logical → physical paths per `orchestration_id + task_id`

---

## 11. Configuration Reference

### MemoryConfig

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Master toggle for semantic memory |
| `path` | `~/.captain-claw/memory.db` | SQLite database path |
| `index_workspace` | `true` | Index user workspace files |
| `index_sessions` | `true` | Index session messages |
| `cross_session_retrieval` | `false` | Search across all sessions |
| `auto_sync_on_search` | `true` | Trigger sync if data stale |
| `max_workspace_files` | 400 | Max files to index |
| `max_file_bytes` | 262,144 (256KB) | Max file size to index |
| `chunk_chars` | 1,400 | Chunk size (line-based) |
| `chunk_overlap_chars` | 200 | Overlap between chunks |
| `cache_ttl_seconds` | 45 | Search result cache TTL |
| `stale_after_seconds` | 120 | Auto-sync threshold |
| `layered_summaries` | `true` | Generate L1/L2 summaries |
| `summary_batch_size` | 10 | Chunks per summarization batch |

### MemoryEmbeddingsConfig

| Setting | Default | Description |
|---------|---------|-------------|
| `provider` | `"auto"` | `auto`, `litellm`, `ollama`, `none` |
| `litellm_model` | `text-embedding-3-small` | LiteLLM embedding model |
| `ollama_model` | `nomic-embed-text` | Ollama embedding model |
| `ollama_base_url` | `http://127.0.0.1:11434` | Ollama server URL |
| `request_timeout_seconds` | 4 | Embedding request timeout |
| `fallback_to_local_hash` | `true` | Use hash if providers fail |

### MemorySearchConfig

| Setting | Default | Description |
|---------|---------|-------------|
| `max_results` | 6 | Results returned per search |
| `candidate_limit` | 80 | Candidates before reranking |
| `min_score` | 0.1 | Minimum combined score |
| `vector_weight` | 0.65 | Weight for vector similarity |
| `text_weight` | 0.35 | Weight for BM25 keyword score |
| `temporal_decay_enabled` | `true` | Apply time-based score decay |
| `temporal_half_life_days` | 21.0 | Half-life for temporal decay |

### DeepMemoryConfig

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `false` | Master toggle |
| `protocol` | `http` | Typesense protocol |
| `host` | `localhost` | Typesense host |
| `port` | 8108 | Typesense port |
| `collection_name` | `captain_claw_deep_memory` | Collection name |
| `embedding_dims` | 1536 | Vector dimensions |
| `auto_embed` | `true` | Auto-generate embeddings |
| `layered_summaries` | `true` | Generate L1/L2 for deep memory |

### InsightsConfig

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Master toggle |
| `auto_extract` | `true` | Enable automatic extraction |
| `inject_in_context` | `true` | Show in system prompt |
| `max_items_in_prompt` | 8 | Max insights per context |
| `extraction_interval_messages` | 8 | Trigger every N messages |
| `extraction_cooldown_seconds` | 60 | Min time between extractions |
| `max_insights` | 500 | Storage limit |
| `db_path` | `~/.captain-claw/insights.db` | Database location |

### Context/Compaction Config

| Setting | Default | Description |
|---------|---------|-------------|
| `max_tokens` | 100,000 | Working memory budget |
| `compaction_threshold` | 0.8 | Trigger compaction at 80% |
| `compaction_ratio` | 0.4 | Keep 40% recent messages |

---

## 12. Key Files Reference

### Core Memory

| File | Role |
|------|------|
| `memory.py` | `LayeredMemory` facade + `WorkingMemory` |
| `semantic_memory.py` | SQLite-backed hybrid search index |
| `deep_memory.py` | Typesense-backed archive layer |
| `datastore.py` | Relational data manager |

### Insights & Reflections

| File | Role |
|------|------|
| `insights.py` | `InsightsManager` — CRUD + auto-extraction |
| `reflections.py` | Reflection generation, serialization, caching |
| `tools/insights.py` | LLM-callable insights tool |
| `tools/datastore.py` | LLM-callable datastore tool |

### Agent Integration

| File | Role |
|------|------|
| `agent_context_mixin.py` | Memory init, context note generation, summarizer wiring |
| `agent_session_mixin.py` | Compaction, session description generation |
| `agent_tool_loop_mixin.py` | Post-tool-call insight extraction hook |
| `session_orchestrator.py` | Multi-session context, file registry |

### Web / REST API

| File | Role |
|------|------|
| `web/rest_semantic_memory.py` | REST API for semantic memory browser |
| `web/rest_deep_memory.py` | REST API for deep memory browser |
| `web/rest_insights.py` | REST API for insights CRUD |
| `web/rest_reflections.py` | REST API + slash commands for reflections |
| `web/chat_handler.py` | Post-turn triggers for auto-reflect and insight extraction |

### Prompts

| File | Role |
|------|------|
| `instructions/insight_extraction_system_prompt.md` | Insight extraction system prompt |
| `instructions/insight_extraction_user_prompt.md` | Insight extraction user prompt |
| `instructions/reflection_system_prompt.md` | Reflection generation system prompt |
| `instructions/reflection_user_prompt.md` | Reflection generation user prompt |
| `instructions/system_prompt.md` | Main system prompt (`{reflection_block}`, `{insights_block}`) |

### Configuration & Tests

| File | Role |
|------|------|
| `config.py` | All memory/insights/deep config schemas |
| `tests/test_memory/test_layered_memory.py` | Layered memory tests |
