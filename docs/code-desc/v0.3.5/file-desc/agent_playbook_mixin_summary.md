# Summary: agent_playbook_mixin.py

# Summary: agent_playbook_mixin.py

## Overview
A mixin class that manages playbook retrieval, formatting, and distillation for an AI agent system. Playbooks are human-reviewed do/don't patterns extracted from successful sessions that improve orchestration decisions by providing concrete pseudo-code examples for recurring task types. The module handles task classification, playbook injection into prompts, session trace analysis, and LLM-driven playbook proposal generation.

## Purpose
Solves the problem of capturing and reusing proven patterns from past agent sessions. Enables the agent to learn from successful task executions by:
- Automatically classifying incoming tasks into predefined categories (batch-processing, web-research, code-generation, etc.)
- Retrieving relevant playbooks based on task type and keyword matching
- Injecting playbook context into LLM prompts to guide better decision-making
- Distilling session traces into structured playbook proposals via LLM analysis
- Persisting approved playbooks for future use

## Most Important Functions/Classes

### 1. **`classify_task_type(user_input: str) -> str | None`**
Lightweight heuristic-based task classifier using keyword matching against predefined categories (batch-processing, web-research, code-generation, document-processing, data-transformation, orchestration, file-management). Scores each category by keyword frequency and returns the highest-scoring match. No regex or ML required—enables fast, deterministic classification.

### 2. **`AgentPlaybookMixin._retrieve_playbooks(user_input, task_type, max_results=2)`**
Core retrieval engine with three-tier fallback strategy:
- Respects UI-driven overrides (`_playbook_override` attribute: `None`=auto, `"__none__"`=disabled, specific ID=force)
- Auto mode: classifies task type → searches by type → broadens keyword search → lists by type if needed
- Returns up to `max_results` most relevant `PlaybookEntry` objects from persistent store
- Handles graceful degradation when specific playbooks unavailable

### 3. **`AgentPlaybookMixin._distill_playbook_from_session(session_id, rating, user_note)`**
LLM-driven playbook proposal generator that:
- Loads session messages and extracts compact summary and tool trace via helper functions
- Renders system/user prompts from instruction templates with session context
- Calls `_complete_with_guards()` to run distillation LLM pass
- Parses JSON response to extract playbook fields (task_type, name, do_pattern, dont_pattern, trigger_description, reasoning)
- Validates required fields and returns structured dict or None on failure
- Enables human review before persistence

### 4. **`format_playbook_block(entries: list[PlaybookEntry]) -> str`**
Formats playbook entries into a structured text block for injection into planner/scale prompts. Includes pattern name, task type, trigger conditions, DO/DON'T pseudo-code, and reasoning. Returns empty string if no entries (safe for template concatenation).

### 5. **`AgentPlaybookMixin._rate_and_distill_session(session_id, rating, user_note)`**
Orchestrates the complete session rating and playbook proposal workflow:
- Defaults to current session if not specified
- Stores rating and user notes in session metadata
- Triggers distillation via `_distill_playbook_from_session()`
- Returns status dict indicating success ("proposed"), failure ("distill_failed"), or missing session ("no_session")
- Enables downstream persistence via `_save_playbook_from_proposal()`

## Architecture & Dependencies

**Key Dependencies:**
- `captain_claw.session`: `PlaybookEntry`, `get_session_manager()` for persistent playbook storage
- `captain_claw.instructions`: `InstructionLoader` for rendering distillation prompts
- `captain_claw.llm`: `Message` class for LLM communication
- `captain_claw.config`: `get_config()` for model token limits
- `captain_claw.logging`: structured logging

**Design Patterns:**
- **Mixin architecture**: Adds playbook capabilities to Agent class without inheritance bloat
- **Async-first with sync fallback**: `_build_playbook_context_note_sync()` bridges sync/async contexts using event loop detection
- **Best-effort error handling**: Gracefully degrades on missing playbooks, failed distillation, or LLM errors
- **Override mechanism**: UI-driven playbook selection via `_playbook_override` attribute
- **Multi-tier retrieval**: Task-type-specific search → keyword broadening → type-based listing

**Session Integration:**
- Playbooks stored in session metadata (`playbook_rating`, `playbook_rating_note`)
- Usage counters incremented on retrieval (popularity tracking)
- Source session linked to generated playbooks for traceability

**Prompt Injection Points:**
- `_build_playbook_block()`: Full-featured injection for planner/scale (async)
- `_build_playbook_context_note()`: Lighter injection for message-level context (async)
- `_build_playbook_context_note_sync()`: Synchronous wrapper with caching for sync contexts