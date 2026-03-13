# Summary: deep_memory.py

# Deep Memory System (Typesense-Backed Long-Term Archive)

## Summary

This module implements a persistent, searchable deep memory layer built on Typesense (a vector search engine). It operates as a supplementary archive to SQLite-backed semantic memory, designed for on-demand retrieval of long-term content. The system supports hybrid search (BM25 full-text + optional vector similarity), automatic document chunking with overlap, embedding integration, and LLM context injection.

## Purpose

Solves the problem of maintaining a queryable long-term archive for agent memory that:
- Persists beyond short-term semantic memory constraints
- Supports both keyword and semantic (vector) search
- Integrates seamlessly with LLM prompts via context note generation
- Handles document chunking consistently across memory layers
- Provides programmatic and tool-based indexing interfaces

## Architecture & Dependencies

**External Dependencies:**
- `httpx` — HTTP client for Typesense REST API communication
- `captain_claw.logging` — Structured logging

**Key Design Patterns:**
- Lazy HTTP client initialization with connection pooling
- Collection schema auto-creation with idempotency
- Deterministic chunk ID generation via SHA1 hashing
- Hybrid scoring (combines BM25 text_match with inverted vector_distance)
- Embedding dimension validation to prevent schema mismatches

**Data Flow:**
- **Inbound:** Micro-loop sink (`no_file`), LLM-callable `typesense` tool, programmatic API
- **Outbound:** Context note injection into LLM prompts, typed `DeepMemoryResult` objects

---

## Most Important Functions/Classes

### 1. **`DeepMemoryIndex` (Main Class)**
   - **Purpose:** Typesense-backed index manager mirroring `SemanticMemoryIndex` public API
   - **Key Responsibilities:** Collection lifecycle, document indexing, hybrid search, context note generation, deletion operations
   - **Configuration:** Host/port/protocol, API key, collection name, embedding dimensions, chunking parameters

### 2. **`index_document(doc_id, text, *, source, reference, path, tags) → int`**
   - **Purpose:** Index a single document with automatic chunking and optional embedding
   - **Process:** 
     - Chunks text using `_chunk_text()` with configurable overlap
     - Generates deterministic chunk IDs via `_hash_id()`
     - Computes embeddings via `_embed()` if embedding chain available
     - Upserts all chunks to Typesense via `_upsert_batch()`
   - **Returns:** Count of successfully indexed chunks

### 3. **`search(query, *, max_results, filter_by, vector_query) → list[DeepMemoryResult]`**
   - **Purpose:** Hybrid search combining BM25 full-text and optional vector similarity
   - **Process:**
     - Auto-generates vector query from embedding chain if not provided
     - Queries Typesense with combined parameters
     - Combines text_match and vector_distance scores (inverts vector distance: `1/(1+distance)`)
     - Returns typed `DeepMemoryResult` objects with metadata
   - **Scoring:** `combined = max(text_score, 1/(1+vector_score))`

### 4. **`build_context_note(query, *, max_items, max_snippet_chars) → tuple[str, str]`**
   - **Purpose:** Generate LLM-injectable context note from search results
   - **Output:** Returns `(formatted_note, debug_block)` matching `SemanticMemoryIndex` contract
   - **Format:** Human-readable snippet list with source tags, locations, and scores
   - **Error Handling:** Gracefully degrades to empty strings on search failure

### 5. **`_chunk_text(text, chunk_chars, chunk_overlap_chars) → list[dict]`**
   - **Purpose:** Split text into overlapping line-based chunks (consistent with semantic memory)
   - **Algorithm:**
     - Line-based chunking (respects line boundaries)
     - Configurable character limit per chunk (default 1,400)
     - Configurable overlap (default 200 chars) for context preservation
     - Returns list of dicts with `chunk_index`, `start_line`, `end_line`, `text`
   - **Consistency:** Matches `SemanticMemoryIndex._chunk_document()` algorithm

### 6. **`_embed(texts) → list[list[float]]`**
   - **Purpose:** Compute embeddings via shared embedding chain with validation
   - **Validation:** Checks embedding dimensions match collection schema; discards mismatched vectors
   - **Fallback:** Returns empty list if embedding disabled, chain unavailable, or errors occur
   - **Providers Supported:** OpenAI, Ollama, local_hash (with dimension validation)

---

## Supporting Methods

- **`ensure_collection()`** — Idempotent collection creation with 409 conflict handling
- **`_upsert_batch(docs) → int`** — JSONL batch import to Typesense; logs partial failures
- **`_get_client() → httpx.Client`** — Lazy HTTP client with 30s timeout, 5s connect timeout
- **`delete_document(doc_id) → int`** — Delete all chunks for a document ID
- **`delete_by_filter(filter_by) → int`** — Delete via Typesense filter expression
- **`close()`** — Cleanup HTTP client connection

---

## Collection Schema

**Fields:**
- `doc_id`, `source`, `reference` (faceted for filtering)
- `path`, `text` (searchable content)
- `chunk_index`, `start_line`, `end_line` (positional metadata)
- `tags` (optional faceted array)
- `embedding` (optional float[] for vector search, auto-added if dims > 0)
- `updated_at` (int64, default sort field)

**Token Separators:** `.`, `/`, `-`, `_` (for better tokenization of paths/IDs)

---

## Error Handling & Robustness

- **HTTP Errors:** Raises on non-409 status codes; treats 409 (conflict) as success
- **Embedding Failures:** Logs at debug level; continues without vectors
- **Dimension Mismatches:** Logs warning; discards vectors to prevent schema violations
- **Partial Batch Failures:** Logs warning with first error; returns success count
- **Search Failures:** Returns empty results; logs at debug level for context note generation

---

## Integration Points

1. **Micro-loop Integration:** Receives indexed items via `no_file` sink
2. **LLM Tool:** Exposed as `typesense` callable for manual indexing
3. **Agent Context Mixin:** Used interchangeably with `SemanticMemoryIndex` for note generation
4. **Embedding Chain:** Pluggable embedding provider (OpenAI, Ollama, etc.)
5. **Typesense Server:** External dependency (must be running on configured host:port)