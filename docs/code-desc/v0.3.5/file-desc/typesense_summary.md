# Summary: typesense.py

# Typesense Tool Summary

**Summary:**
A locked-down, LLM-safe interface to Typesense deep memory storage that restricts the language model to three operations: indexing text (with automatic chunking and embedding), searching (keyword + vector hybrid), and deleting documents. The tool enforces a single pre-configured collection and canonical schema, preventing LLM access to collection creation, naming, or schema definition.

**Purpose:**
Solves the problem of safely exposing vector search and document storage to LLMs without allowing them to misconfigure infrastructure, create arbitrary collections, or bypass security controls. Acts as a controlled gateway between the LLM and Typesense, delegating to `DeepMemoryIndex` when available (preferred path with full chunking/embedding support) or falling back to raw HTTP operations when standalone.

**Most Important Functions/Classes/Procedures:**

1. **`TypesenseTool` (class)**
   - Main tool class inheriting from `Tool`. Manages configuration, HTTP client lifecycle, and action dispatch. Enforces collection lockdown via `_get_collection()` which returns either the `DeepMemoryIndex` collection name or a configured default—no LLM choice allowed.

2. **`_ensure_collection()` (async method)**
   - Bootstrap handler that creates the deep memory collection with the canonical schema if it doesn't exist. Delegates to `DeepMemoryIndex.ensure_collection()` when available (sync, called at startup), otherwise performs async HTTP collection creation with embedding field injection based on configured `embedding_dims`.

3. **`execute()` (async method)**
   - Main entry point dispatching to action handlers (`_action_index`, `_action_search`, `_action_delete`). Handles runtime kwarg cleanup, validates API key and collection configuration, ensures collection exists, and wraps all operations with comprehensive HTTP error handling (connection errors, status errors, JSON decode failures).

4. **`_action_index()` (async method)**
   - Indexes text by generating a deterministic `doc_id` from reference + text hash, then routing through `DeepMemoryIndex.index_document()` (preferred) for proper chunking, embedding, and timestamp handling, or falling back to raw HTTP upsert. Supports source labeling, reference tracking, and comma-separated tags.

5. **`_action_search()` (async method)**
   - Searches the collection using `DeepMemoryIndex.search()` (hybrid keyword + vector when available) or raw HTTP keyword search. Formats results with source tags, scores, and truncated snippets (max 300 chars). Respects `filter_by` expressions and `max_results` (capped at 250).

6. **`_action_delete()` (async method)**
   - Deletes by `document_id` (removes all chunks for that doc) or by `filter_by` expression. Routes through `DeepMemoryIndex.delete_document()` or `delete_by_filter()` when available, otherwise uses raw HTTP filter-based deletion. Prevents collection-level deletion.

**Architecture & Dependencies:**
- **Async-first design** using `httpx.AsyncClient` for non-blocking HTTP operations with configurable timeouts and connection pooling.
- **Dual-path execution**: Preferred path delegates to `DeepMemoryIndex` (from `captain_claw.deep_memory`) for schema compliance and intelligent chunking; fallback path uses raw Typesense HTTP API when `DeepMemoryIndex` is unavailable.
- **Configuration-driven**: Reads from `captain_claw.config` (protocol, host, port, API key, default collection, timeouts, embedding dimensions).
- **Schema enforcement**: Uses canonical `_COLLECTION_SCHEMA_TEMPLATE` from `deep_memory` module; dynamically injects embedding field if `embedding_dims > 0`.
- **Error resilience**: Graceful degradation on collection bootstrap failures; comprehensive exception handling for connection, HTTP status, and JSON errors with user-friendly messages.
- **Security**: No LLM control over collection names, schema, or creation—all operations locked to a single pre-configured collection.