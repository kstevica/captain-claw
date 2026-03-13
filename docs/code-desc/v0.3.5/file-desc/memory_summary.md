# Summary: memory.py

# memory.py Summary

This module implements a three-layer memory architecture for Captain Claw, combining working memory (turn-level context), session memory (managed externally), and semantic memory (persistent SQLite-backed retrieval). It provides token-aware message buffering with automatic compaction, serialization capabilities, and unified search across memory layers.

## Purpose

Solves the problem of managing conversation context across multiple timescales: immediate turn-level interactions (working memory), session-scoped history (session memory), and long-term semantic knowledge retrieval (semantic memory). Addresses token budget constraints through intelligent message compaction while maintaining conversation coherence via summaries.

## Most Important Functions/Classes

1. **WorkingMemory class**
   - In-turn memory buffer maintaining a summary string plus recent detailed message window. Implements token estimation (1 token ≈ 4 characters), threshold-based compaction detection, and message addition. Core methods: `add_message()`, `compact()` (summarizes older messages while retaining recent ones), `get_token_count()`, `should_compact()`.

2. **LayeredMemory class**
   - Facade coordinating three memory layers (working, session via external manager, semantic). Routes message recording to both working and semantic memory, manages active session context, and provides unified search interface. Key methods: `record_message()`, `search_semantic()`, `search_in_session()`, `build_semantic_note()`, `compact_working_memory()`.

3. **WorkingMemorySnapshot dataclass**
   - Serializable representation of working memory state containing summary string and message list. Enables persistence and transmission of conversation context snapshots.

4. **create_layered_memory() factory function**
   - Instantiates complete memory stack from runtime configuration, handling graceful degradation if semantic memory initialization fails. Reads max token limits from config and conditionally enables semantic indexing based on configuration flags.

5. **compact() method (WorkingMemory)**
   - Implements intelligent message retention strategy: keeps recent messages (configurable ratio, default 40%), summarizes dropped older messages (last 10 for context), and appends new summary to existing summary. Prevents memory bloat while preserving conversation continuity through incremental summaries.

## Architecture & Dependencies

- **Dependencies**: `captain_claw.logging` (structured logging), `captain_claw.semantic_memory` (SemanticMemoryIndex, create_semantic_memory_index)
- **Token Management**: Simple estimation algorithm (text length / 4) with configurable max_tokens threshold (default 100K)
- **Compaction Strategy**: Ratio-based retention (5-95% bounds) with incremental summary building
- **Integration Points**: Semantic memory is optional; system degrades gracefully if initialization fails. Session management delegated to external SessionManager (not in this file)
- **Serialization**: WorkingMemorySnapshot enables checkpoint/restore of conversation state