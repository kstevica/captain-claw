# Summary: file_registry.py

# File Registry Summary

**Summary:**
FileRegistry maintains a bidirectional mapping system between logical file paths (what LLMs request) and physical on-disk locations within session-scoped directories. It enables cross-task file access in orchestrated workflows without exposing internal folder structures or session IDs. The registry is thread-safe, supports fuzzy filename matching, and can persist mappings to SQLite for survival across server restarts.

**Purpose:**
Solves the problem of inconsistent file path references in multi-agent orchestration workflows. LLMs frequently refer to the same file using different path formats (e.g., `/output/results.json` vs `results.json`). The registry provides intelligent resolution with fallback strategies, allowing downstream tasks to access upstream artifacts without knowledge of session IDs or internal directory hierarchies. Also maintains a manifest of available files for injection into task prompts.

**Most Important Functions/Classes/Procedures:**

1. **`FileRegistry.__init__()`**
   - Initializes thread-safe registry with optional SQLite persistence callback. Sets up internal dictionaries for exact-match and filename-based fuzzy lookups. Stores orchestration ID and workflow metadata (start timestamp, shared run directory).

2. **`register(logical_path, physical_path, task_id="")`**
   - Records logical-to-physical mappings with thread-safe locking. Extracts filename for reverse index to enable fuzzy matching. Triggers async SQLite persistence callback if configured. Logs all registrations for diagnostics.

3. **`resolve(logical_path) -> str | None`**
   - Three-tier lookup strategy: (1) exact normalized path match, (2) stripped-prefix match (removes "saved/" prefixes), (3) filename-only fuzzy match (only if unambiguous). Returns physical path string or None. Thread-safe with lock protection.

4. **`build_manifest() -> str`**
   - Generates human-readable file inventory for LLM prompt injection. Formats as bulleted list showing logical paths (what LLM should use) with physical location hints. Prevents LLMs from attempting glob searches on non-existent workspace paths.

5. **`_normalize(path) -> str`**
   - Static utility that canonicalizes paths for consistent lookup: strips whitespace/quotes, resolves path components, removes leading slashes, normalizes separators. Ensures `/foo/bar` and `foo/bar` resolve identically.

**Architecture & Dependencies:**
- **Thread Safety:** Uses `threading.Lock()` for all state mutations; safe for concurrent task access within single orchestration run
- **Async Integration:** Fire-and-forget SQLite persistence via `asyncio.create_task()` with graceful fallback when no event loop exists
- **Indexing Strategy:** Dual-index design (exact path map + filename reverse index) enables O(1) exact lookups and O(1) fuzzy lookups when unambiguous
- **Serialization:** Supports `to_dict()`/`from_dict()` for state persistence and `merge_from()` for combining registries across orchestration phases
- **Logging:** Integrates with captain_claw logging system for diagnostic tracing

**Key Design Decisions:**
- First-writer-wins conflict resolution in `merge_from()` prevents accidental overwrites
- Fuzzy matching only triggers on single-candidate filenames to avoid ambiguity
- Prefix stripping handles common LLM path normalization patterns without explicit configuration
- Workflow metadata (`workflow_started_at`, `workflow_run_dir`) enables tools to filter stale files from prior runs