# Summary: agent_skills_mixin.py

# agent_skills_mixin.py Summary

## Summary
This mixin provides comprehensive skill management and invocation capabilities for an AI agent system, including skill discovery, LLM-based ranking, manual invocation, GitHub installation, and dependency management. It handles skill snapshot caching with version tracking, environment variable overrides, and supports multiple dispatch modes (tool-based, script-based, and prompt rewriting).

## Purpose
Solves the problem of integrating a modular skill system into an AI agent by providing: (1) dynamic skill loading and caching with filesystem watching, (2) semantic skill discovery via LLM ranking combined with lexical search, (3) flexible skill invocation through multiple dispatch mechanisms, (4) skill installation from GitHub repositories, and (5) dependency installation orchestration. Enables agents to discover, select, and execute specialized capabilities on-demand.

## Most Important Functions/Classes/Procedures

### 1. **`_resolve_skills_snapshot(force_refresh: bool = False) -> SkillSnapshot`**
   - **Description**: Core caching mechanism that resolves the current workspace skills snapshot with session-aware persistence and version-based invalidation. Implements debounced filesystem watching to detect skill changes without constant recomputation. Restores snapshots from session metadata when available.
   - **Key Logic**: Checks cache validity via version hash, respects debounce intervals, falls back to session storage, and rebuilds snapshot only when necessary.

### 2. **`invoke_skill_command(name: str, args: str | None = None, turn_usage: dict[str, int] | None = None) -> dict[str, Any]`**
   - **Description**: Main entry point for manual skill execution via `/skill` commands. Resolves skill by name, dispatches via appropriate mechanism (tool, script, or prompt rewriting), executes with guards, logs results to session, and returns structured outcome.
   - **Key Logic**: Handles three dispatch modes—tool dispatch (delegates to `_execute_tool_with_guard`), script dispatch (resolves interpreter and arguments via `_resolve_script_dispatch_command`), and prompt rewriting (returns rewritten request for LLM handling).

### 3. **`search_skill_catalog(criteria: str) -> dict[str, Any]`**
   - **Description**: Implements two-stage skill discovery combining lexical ranking with LLM-based semantic ranking. Fetches catalog from configured source, applies lexical filtering to create a candidate pool, ranks pool with LLM for relevance, and falls back to lexical ranking if LLM ranking produces fewer results.
   - **Key Logic**: Lexical ranking generates `limit * 20` candidates (min 120), LLM ranks top `limit` from pool, fallback ensures result coverage via lexical ranking of full catalog.

### 4. **`_rank_skill_search_entries_with_llm(query: str, entries: list[SkillCatalogEntry], limit: int) -> list[SkillCatalogEntry]`**
   - **Description**: Uses LLM to semantically rank skill catalog entries by relevance to user query. Constructs structured JSON prompt with candidate metadata, parses LLM response for selected IDs, and returns ranked subset with deduplication.
   - **Key Logic**: Extracts selected IDs from multiple possible JSON keys (`selected_ids`, `top_ids`, `ids`, `selected`, `results`), validates indices, and falls back to `_extract_json_object` if initial JSON parsing fails.

### 5. **`_build_skills_system_prompt_section() -> str`**
   - **Description**: Generates the mandatory skills section appended to the system prompt. Retrieves snapshot prompt block and wraps it with OpenClaw-style instructions for skill selection and reading workflow.
   - **Key Logic**: Enforces single skill read-ahead constraint, requires explicit selection before reading SKILL.md, and prefers specific skills over generic utilities.

### 6. **`install_skill_from_github(github_url: str) -> dict[str, Any]`**
   - **Description**: Installs a skill from GitHub URL into the managed skills directory. Delegates to `install_skill_from_github_url`, refreshes snapshot, and returns installation metadata including aliases.
   - **Key Logic**: Validates URL, installs skill, refreshes snapshot to detect new commands, and extracts command aliases for the installed skill.

### 7. **`_resolve_script_dispatch_command(command: SkillCommandSpec, raw_args: str) -> tuple[str, str]`**
   - **Description**: Resolves script path and constructs shell command for script-based skill dispatch. Determines interpreter from config or file extension (.py → Python, .sh → bash), parses arguments via shlex, and returns quoted shell command.
   - **Key Logic**: Supports custom interpreters, auto-detects Python/bash, handles argument parsing with error recovery, and returns both shell command and resolved script path.

---

## Architecture & Dependencies

**Key Dependencies**:
- `captain_claw.skills`: SkillSnapshot, SkillCatalogEntry, SkillCommandSpec, skill loading/ranking utilities
- `captain_claw.llm`: Message class for LLM communication
- `captain_claw.config`: Configuration management
- `captain_claw.logging`: Structured logging

**System Role**: Mixin class integrated into Agent class, providing skill subsystem. Depends on agent having `provider` (LLM), `session`/`session_manager`, `workspace_base_path`, and methods like `_execute_tool_with_guard`, `_add_session_message`, `_emit_tool_output`, `_log_llm_call`, `_extract_json_object`.

**Caching Strategy**: Session-aware with version hashing and debounced filesystem watching. Snapshot cached in `_skills_snapshot_cache` and persisted in `session.metadata["skills"]["snapshot"]`.

**Dispatch Modes**: (1) Tool dispatch—delegates to registered tool handlers; (2) Script dispatch—executes shell commands with interpreter resolution; (3) Prompt rewriting—returns rewritten request for LLM to handle skill invocation.