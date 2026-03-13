# Summary: semantic_memory.py

# Semantic Memory Index - Comprehensive Summary

## Overview

This module implements a sophisticated SQLite-backed semantic memory system with hybrid retrieval capabilities, combining full-text search (FTS) and vector embeddings. It indexes workspace files and session conversations, enabling intelligent context retrieval through keyword matching and semantic similarity. The system supports multiple embedding providers (LiteLLM, Ollama, local hash-based), background synchronization, temporal decay scoring, and cross-session retrieval with configurable weighting strategies.

## Purpose

Solves the problem of maintaining persistent, searchable context across development sessions and workspace files. Enables AI agents to retrieve relevant information through both lexical (keyword) and semantic (vector) search, with intelligent result ranking that combines multiple scoring dimensions including recency, relevance, and source type. Supports both workspace indexing (source code, documentation) and session memory (conversation history).

---

## Architecture & Key Components

### Core Data Structures

**SemanticMemoryResult** - Output dataclass representing a single search hit with chunk metadata, multiple relevance scores (combined, text-based, vector-based), and source location information.

**_Document** - Internal representation of indexable content (workspace file or session) with source type, reference ID, file path, content hash signature, and timestamp.

**_Chunk** - Subdivision of documents with line-number tracking, enabling precise source location references in search results.

---

## Most Important Functions/Classes

### 1. **SemanticMemoryIndex (Main Class)**
   - **Purpose**: Central orchestrator managing the entire semantic memory lifecycle
   - **Key Responsibilities**: 
     - Database initialization and schema management (SQLite with WAL mode)
     - Hybrid search coordination (keyword + vector)
     - Background document synchronization
     - Cache management with TTL
     - Session scoping and cross-session retrieval control
   - **Critical Methods**:
     - `search()` - Primary hybrid search interface with caching and auto-sync
     - `search_in_session()` - Targeted session-specific retrieval
     - `upsert_text()` - Manual document insertion for tests/ad-hoc memory
     - `schedule_sync()` - Non-blocking background indexing trigger
     - `build_context_note()` - Formats search results as LLM-ready prompt context

### 2. **_EmbeddingProviderChain (Provider Abstraction)**
   - **Purpose**: Manages multiple embedding providers with automatic failover
   - **Key Responsibilities**:
     - Provider rotation on failure (round-robin fallback)
     - Unified interface for different embedding backends
     - Thread-safe active provider tracking
   - **Supported Providers**:
     - `_LiteLLMEmbeddingProvider` - OpenAI, Anthropic, and 100+ models via LiteLLM
     - `_OllamaEmbeddingProvider` - Local Ollama instance (nomic-embed-text default)
     - `_LocalHashEmbeddingProvider` - Deterministic SHA1-based bag-of-words fallback (256-dim)
   - **Key Method**: `embed_batch()` - Returns provider ID and normalized embedding vectors

### 3. **Hybrid Search Pipeline (_keyword_search + _vector_search + _merge_hybrid)**
   - **Purpose**: Implements multi-modal relevance ranking
   - **_keyword_search()**: 
     - FTS5 full-text search with BM25 ranking
     - Fallback to LIKE queries if FTS fails
     - Filters by source type (workspace vs. session)
     - Returns text_score (0-1 normalized from BM25 rank)
   - **_vector_search()**:
     - Cosine similarity matching against stored embeddings
     - Provider key validation (ensures index freshness)
     - Session/workspace filtering
     - Returns vector_score (0-1 cosine similarity)
   - **_merge_hybrid()**:
     - Combines keyword and vector hits by chunk_id
     - Weighted score calculation: `(vector_weight × vector_score) + (text_weight × text_score)`
     - Applies temporal decay: `score × exp(-λ × age_days)` where λ = ln(2) / half_life_days
     - Filters by min_score threshold
     - Returns top-k results sorted by final score

### 4. **Document Synchronization (_sync_once + _collect_workspace_documents + _collect_session_documents)**
   - **Purpose**: Maintains index freshness without blocking search operations
   - **_sync_once()**:
     - Orchestrates workspace and session document collection
     - Compares signatures to detect changes (size + mtime for files, hash for session text)
     - Upserts new/modified documents, deletes stale entries
     - Triggers embedding generation for new chunks
   - **_collect_workspace_documents()**:
     - Walks filesystem respecting exclude_dirs (git, node_modules, etc.)
     - Filters by file extension (configurable, defaults to code/config/doc formats)
     - Enforces max_workspace_files and max_file_bytes limits
     - Computes relative paths and file modification timestamps
   - **_collect_session_documents()**:
     - Reads session database (separate SQLite DB)
     - Parses JSON message arrays (role + content format)
     - Generates safe filenames from session names
     - Handles missing/malformed session data gracefully

### 5. **Chunking & Embedding (_chunk_document + _upsert_embeddings_for_chunks)**
   - **Purpose**: Breaks large documents into semantically-sized pieces with overlap
   - **_chunk_document()**:
     - Line-based chunking with configurable chunk_chars (default 1400)
     - Overlap strategy: retains last N lines from previous chunk (chunk_overlap_chars)
     - Tracks start_line and end_line for precise source location
     - Generates deterministic chunk_ids from doc_id + index + text prefix
   - **_upsert_embeddings_for_chunks()**:
     - Batches chunks (24 per batch) for efficient API calls
     - Stores embeddings as JSON with provider_key and dimensions
     - Handles provider failures gracefully (logs warning, continues with keyword-only)
     - Normalizes all embeddings to unit vectors

---

## Database Schema

**memory_documents** - Document-level metadata (source, reference, path, signature, timestamp)

**memory_chunks** - Chunk-level records with line numbers and full text

**memory_chunks_fts** - FTS5 virtual table for full-text search

**memory_embeddings** - Vector embeddings with provider tracking and dimension info

**memory_sync_state** - Sync metadata (currently unused but reserved)

---

## Configuration & Initialization

**create_semantic_memory_index()** - Factory function accepting config object with:
- `path` - SQLite database location
- `index_workspace`, `index_sessions` - Boolean flags
- `cross_session_retrieval` - Enable all-sessions search
- `max_workspace_files`, `max_file_bytes` - Indexing limits
- `include_extensions`, `exclude_dirs` - File filtering
- `chunk_chars`, `chunk_overlap_chars` - Chunking parameters
- `cache_ttl_seconds`, `stale_after_seconds` - Cache/sync timing
- `search.max_results`, `search.candidate_limit` - Result limits
- `search.vector_weight`, `search.text_weight` - Hybrid weighting
- `search.temporal_decay_enabled`, `temporal_half_life_days` - Recency scoring
- `embeddings.provider` - "litellm" | "ollama" | "auto" | "none"
- `embeddings.litellm_model`, `litellm_api_key`, `litellm_base_url`
- `embeddings.ollama_model`, `ollama_base_url`

**_build_embedding_chain()** - Constructs provider chain with fallback logic (LiteLLM → Ollama → LocalHash)

---

## Threading & Concurrency

- **_sync_lock** - Prevents concurrent sync operations
- **_db_lock** - RLock protecting SQLite access (check_same_thread=False)
- **Background sync worker** - Daemon thread triggered by `schedule_sync()`, respects _dirty flag
- **Thread-safe cache** - Dictionary with TTL-based expiration

---

## Search Scoping & Session Management

- **set_active_session()** - Configures session context for subsequent searches
- **cross_session_retrieval flag** - Global setting for all-sessions vs. active-session-only
- **search_in_session()** - Override for targeted session retrieval regardless of global flag
- **Session filtering logic**: 
  - Workspace-only: excludes all session chunks
  - Active session: includes workspace + specific session
  - All sessions: includes workspace + all sessions

---

## Utility Functions

**_normalize_embedding()** - L2 normalization with zero-vector handling

**_cosine_similarity()** - Bounded similarity computation (-1 to 1) with NaN protection

**_tokenize_fts()** - Lowercases and extracts word tokens via regex

**_build_fts_query()** - Constructs FTS5 query with stopword filtering and length thresholds

**_hash_text()** - SHA1 hashing for signatures and deterministic IDs

**_parse_iso_to_timestamp()** - ISO8601 parsing with timezone handling

---

## Error Handling & Resilience

- **Embedding provider failover** - Chain automatically rotates to next provider on error
- **FTS fallback** - Reverts to LIKE queries if FTS5 fails
- **Graceful degradation** - Continues with keyword-only search if embeddings unavailable
- **Session DB errors** - Logs debug message, returns empty document list
- **File read errors** - Skips unreadable files, continues indexing
- **Malformed data** - Validates JSON, handles missing fields, filters invalid entries

---

## Performance Characteristics

- **Search latency**: O(candidate_limit) for both keyword and vector searches, merged in O(n log n)
- **Sync latency**: O(files × chunks) with batched embedding calls (24 chunks/batch)
- **Memory**: In-memory cache with configurable TTL; embeddings stored in SQLite
- **Database**: WAL mode for concurrent read/write; PRAGMA optimizations for speed
- **Embedding cost**: Only on document changes (signature-based detection)

---

## Key Design Decisions

1. **Hybrid scoring** - Combines lexical and semantic relevance with configurable weights, enabling both exact-match and semantic-similarity use cases
2. **Temporal decay** - Exponential decay function prioritizes recent information while preserving older context
3. **Chunking with overlap** - Prevents semantic fragmentation at document boundaries
4. **Provider chain abstraction** - Enables seamless switching between local, self-hosted, and cloud embedding services
5. **Background sync** - Non-blocking indexing prevents search latency spikes
6. **Signature-based change detection** - Avoids re-indexing unchanged files
7. **Session scoping** - Allows both global and targeted retrieval patterns for multi-turn conversations