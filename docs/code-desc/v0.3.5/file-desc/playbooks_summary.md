# Summary: playbooks.py

# PLAYBOOKS.PY SUMMARY

## Summary
PlaybooksTool is a persistent cross-session orchestration system that captures and manages proven patterns (do/don't pseudo-code) for recurring task types. It enables teams to build institutional knowledge by storing successful approaches and anti-patterns, automatically injecting relevant playbooks into planning contexts when similar tasks are detected. The tool supports full CRUD operations plus a sophisticated "rate" action that uses LLM-based distillation to automatically extract playbook patterns from completed sessions.

## Purpose
Solves the problem of knowledge loss across sessions and repeated task failures by creating a searchable, persistent library of proven orchestration patterns. Enables rapid task execution by providing pre-validated approaches for common problem types (batch-processing, web-research, code-generation, document-processing, data-transformation, orchestration, interactive, file-management). Particularly valuable in multi-agent or multi-user environments where institutional knowledge must survive beyond individual sessions.

## Most Important Functions/Classes/Procedures

### 1. **PlaybooksTool.execute() (async)**
Main entry point routing all six actions (add, list, search, info, update, remove, rate) to appropriate handlers. Manages session context extraction and error handling. Validates action parameter and delegates to specialized static methods. Critical for tool integration with the agent framework.

### 2. **PlaybooksTool._add() (async, static)**
Creates new playbook entries with comprehensive validation. Enforces required fields (name, task_type), validates task_type against PLAYBOOK_TASK_TYPES enum, requires at least one pattern (do_pattern or dont_pattern). Calls session manager's create_playbook() and returns formatted confirmation with playbook ID prefix.

### 3. **PlaybooksTool._rate() (async, static)**
Implements the sophisticated "rate session" workflow that converts completed session traces into reusable playbooks. Stores rating metadata in session, invokes _distill_session_standalone() for LLM-powered pattern extraction, and automatically creates playbook entries from distilled proposals. Bridges session evaluation with playbook creation.

### 4. **_distill_session_standalone() (async, module-level)**
Standalone LLM distillation engine that extracts playbook patterns from session message history without requiring agent reference. Loads session, extracts summary and tool trace, renders system/user prompts from instruction templates, calls LLMProvider.complete(), parses JSON response, and validates required fields (task_type, name, at least one pattern). Returns structured playbook proposal dict or None on failure.

### 5. **PlaybooksTool._list(), _search(), _info() (async, static)**
Query operations providing different access patterns. _list() shows all playbooks with optional task_type filtering (limit 50). _search() performs keyword matching across playbook fields (limit 20). _info() retrieves full playbook details including trigger conditions, patterns, reasoning, usage stats, and source session. All format output for human readability with indexed results.

### 6. **PlaybooksTool._update(), _remove() (async, static)**
Mutation operations for lifecycle management. _update() selectively modifies playbook fields (only provided parameters), validates task_type if changed, delegates to session manager. _remove() deletes playbook after existence verification. Both use playbook_id resolution (supports ID prefix, index, or name lookup via select_playbook()).

## Architecture & Dependencies

**Core Dependencies:**
- `captain_claw.session.get_session_manager()` – Persistent storage and retrieval of playbooks and sessions
- `captain_claw.logging.get_logger()` – Structured logging for errors and warnings
- `captain_claw.tools.registry.Tool, ToolResult` – Framework base class and result wrapper
- `captain_claw.llm.LLMProvider` – Language model for distillation (lazy import in _distill_session_standalone)
- `captain_claw.instructions.InstructionLoader` – Template rendering for distillation prompts
- `captain_claw.agent_playbook_mixin` – Session extraction utilities (_extract_session_summary, _extract_tool_trace)

**Data Model:**
Playbooks contain: id, name, task_type (enum), rating (good/bad), do_pattern, dont_pattern, trigger_description, reasoning, tags, use_count, last_used_at, created_at, source_session. Task types are fixed enum (9 categories) to enable semantic routing.

**Integration Points:**
- Operates as Tool subclass within agent framework (Tool registry pattern)
- Session manager provides persistence layer (abstracted storage backend)
- LLM provider enables intelligent pattern extraction from unstructured session traces
- Instruction templates decouple prompt engineering from code

**Key Design Patterns:**
- Static methods for pure business logic (testability, no state)
- Async/await throughout for non-blocking I/O
- Validation-first approach (fail fast with clear error messages)
- Flexible playbook_id resolution (supports multiple lookup strategies)
- Lazy imports for optional dependencies (LLM, instructions only loaded when needed)
- Structured output formatting for CLI/UI consumption