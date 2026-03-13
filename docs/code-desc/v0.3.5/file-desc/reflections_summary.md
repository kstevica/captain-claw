# Summary: reflections.py

# reflections.py Summary

**Summary:**
Self-reflection system for Captain Claw that periodically generates self-improvement instructions by analyzing recent interactions, memory, and task history. Reflections are persisted as timestamped Markdown files in `~/.captain-claw/reflections/` with the latest reflection injected into the system prompt via a `{reflection_block}` placeholder. The module implements mtime-based caching to keep the prompt lean and follows the same pattern as `personality.py` (dataclass → markdown file → cache → prompt block).

**Purpose:**
Enables the agent to perform continuous self-assessment and improvement by reviewing recent conversation context, stored memory facts, and completed tasks. This creates a feedback loop where the LLM generates actionable self-improvement directives that inform future behavior. The system prevents reflection spam through cooldown timers and message thresholds while maintaining efficient disk I/O through intelligent caching.

**Most Important Functions/Classes/Procedures:**

1. **`Reflection` (dataclass)**
   - Core data model storing a single self-reflection record with timestamp (ISO 8601), summary (self-improvement instructions), topics reviewed, and token usage metrics. Serves as the canonical in-memory representation.

2. **`generate_reflection(agent: Agent) -> Reflection`**
   - Main async function that orchestrates LLM-based reflection generation. Gathers context from previous reflections, recent session messages (last 20), memory facts, and task history. Calls the LLM with system/user prompts, records token usage to session manager, and persists the result to disk. Returns the new Reflection object.

3. **`load_latest_reflection() -> Reflection | None`**
   - Loads the newest reflection from disk using mtime-based caching. Compares file modification time against cached value to avoid redundant disk reads. Returns None if no reflections exist or directory is missing.

4. **`reflection_to_markdown(r: Reflection) -> str` / `markdown_to_reflection(text: str) -> Reflection`**
   - Bidirectional serialization functions. `to_markdown` converts dataclass to canonical Markdown format with sections (Timestamp, Summary, Topics Reviewed, Token Usage). `from_markdown` parses Markdown back to dataclass, intelligently handling LLM-generated headers within the summary section while only treating known section headers as delimiters.

5. **`maybe_auto_reflect(agent: Agent) -> Reflection | None`**
   - Conditional trigger for automatic reflection after agent turns. Enforces two constraints: minimum 4-hour cooldown since last reflection and minimum 10 messages in current session. Returns new Reflection if triggered, None otherwise. Logs warnings on non-fatal failures without disrupting agent operation.

**Architecture & Dependencies:**
- **File I/O**: Uses `pathlib.Path` for cross-platform file handling; stores reflections in `~/.captain-claw/reflections/` with ISO-timestamp-derived filenames
- **Caching Strategy**: Global `_cached_reflection` and `_cached_mtime` variables enable efficient reload detection
- **LLM Integration**: Depends on `agent._complete_with_guards()` for LLM calls; uses `agent.instructions` for prompt template rendering
- **Session Management**: Integrates with `agent.session` for message history and `agent.session_manager` for memory facts and cron task history
- **Prompt Injection**: `reflection_to_prompt_block()` with `_esc()` escaping ensures safe string formatting in system prompts
- **Logging**: Uses module-level logger for info/warning events; records LLM usage metrics to session manager