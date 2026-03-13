# Summary: rest_deep_memory.py

# rest_deep_memory.py Summary

**Summary:** REST API endpoint handler for a Typesense-backed deep memory (semantic search) dashboard. Provides CRUD operations, faceted search, and document indexing capabilities with fallback support for direct agent integration or standalone Typesense connectivity.

**Purpose:** Solves the problem of exposing deep memory functionality through HTTP endpoints, enabling dashboard UI interactions with a vector/semantic search backend. Handles document lifecycle management (browse, search, retrieve, delete), facet aggregation for filtering, and manual document indexing with automatic text chunking.

---

## Most Important Functions/Classes/Procedures

### 1. **`list_documents(server, request)`**
Retrieves paginated document list grouped by `doc_id` with full-text search, source/tag filtering, and relevance sorting. Returns document metadata including chunk counts and text snippets. Core browsing endpoint for the dashboard.

### 2. **`get_document(server, request)`**
Fetches all text chunks for a specific document ordered by chunk index. Returns complete document metadata (source, reference, path, tags, timestamps) plus individual chunk details (line numbers, text content). Enables detailed document inspection.

### 3. **`index_document(server, request)`**
Manually indexes new documents with automatic text chunking and Typesense import. Attempts to use agent's DeepMemoryIndex if available, falls back to direct Typesense API. Generates deterministic doc_id via SHA1 hashing and handles tag parsing.

### 4. **`get_status(server, request)`**
Health check endpoint returning Typesense connectivity status, collection statistics (document count, field schema), and creation timestamp. Gracefully handles missing collections and API key validation.

### 5. **`get_facets(server, request)`**
Aggregates facet counts for `source` and `tags` fields, enabling dashboard filter UI population. Returns value-count pairs for each facet dimension.

---

## Architecture & Dependencies

**Key Dependencies:**
- `httpx` (async HTTP client for Typesense API calls)
- `aiohttp.web` (async web framework for request/response handling)
- `captain_claw.config` (configuration management for Typesense connection params)
- `captain_claw.deep_memory._chunk_text()` (text chunking utility)
- `captain_claw.logging` (structured logging)

**Configuration Sources (Priority Order):**
1. Agent's `_deep_memory` object (if server has attached agent)
2. Global config via `get_config()` (fallback)

**System Role:** Middleware layer between dashboard UI and Typesense search backend. Handles:
- Connection pooling and timeout management (15s default, 5s connect)
- Request validation and parameter sanitization (pagination limits, filter escaping)
- Response transformation (grouping, snippet generation, facet aggregation)
- Error handling with graceful degradation (404 collection handling, fallback indexing paths)

**Notable Design Patterns:**
- Dual-path indexing: prefers agent integration but falls back to direct API
- Deterministic ID generation via SHA1 hashing for idempotency
- Group-by aggregation for document deduplication across chunks
- JSONL streaming for bulk imports (efficient for large document sets)