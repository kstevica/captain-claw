# Summary: agent_reasoning_mixin.py

# agent_reasoning_mixin.py Summary

**Summary:**
A comprehensive mixin providing multi-layered reasoning, planning, and task validation infrastructure for an AI agent. Handles clarification context tracking, contract-based task planning with critic validation, source report generation, list-member extraction and coverage tracking, and intelligent task rephrasing. Bridges user intent detection with structured execution pipelines through heuristic analysis and LLM-driven planning.

**Purpose:**
Solves the problem of translating ambiguous, multi-part user requests into executable task contracts with validation gates. Manages conversation state across clarification loops, detects list-processing vs. single-action tasks, orchestrates prefetching of sources, validates completion against requirements, and provides fallback strategies when planning fails. Enables both direct tool execution and script-generation workflows.

---

## Most Important Functions/Classes/Procedures

### 1. **`_generate_task_contract()`** (async)
Generates a structured task contract from user input using an LLM planner. Accepts user input, recent source URLs, and list task context; calls the planner system prompt to produce tasks (with DAG dependencies), requirements, and prefetch URLs. Implements retry logic with escalating token limits. Returns normalized contract dict with summary, task tree, requirements list, and prefetch URLs. **Critical for:** Multi-step planning and task decomposition.

### 2. **`_evaluate_contract_completion()`** (async)
Critic pass that validates whether a candidate response satisfies all contract requirements. Sends requirements + response to critic LLM; extracts JSON checks for each requirement (ok/reason pairs). Handles scale micro-loop context (post-processing synthesis). Returns completion status and per-requirement feedback. **Critical for:** Quality gates and retry triggering.

### 3. **`_generate_list_task_plan()`** (async)
Extracts list members from user input and context using dedicated LLM extractor. Detects list-processing intent, normalizes member names, determines execution strategy (direct vs. script), and captures per-member action + output format specs. Returns plan dict with enabled flag, member list, strategy, and output metadata. **Critical for:** Batch/per-item processing workflows.

### 4. **`_resolve_effective_user_input()`**
Merges pending clarification context with current user message when appropriate. Detects if user is answering a previous clarification question; if so, appends last assistant response as context to avoid re-researching. Honors explicit refetch requests while providing context. Returns merged input + boolean flag. **Critical for:** Multi-turn clarification loops.

### 5. **`_normalize_contract_tasks()`** (static)
Validates and normalizes raw task tree from planner into DAG-capable structure. Enforces limits (max 8 root tasks, max 12 depth, max 36 total nodes). Extracts task title, children, dependencies, retries, timeouts. Builds stable task IDs. **Critical for:** Task tree validation and safety.

---

## Architecture & Dependencies

**Key Dependencies:**
- `captain_claw.config.get_config()` — Configuration access (model tokens, scale settings)
- `captain_claw.llm.Message` — LLM message objects
- `re`, `json` — Pattern matching and JSON parsing
- `datetime.UTC` — Timestamp tracking for clarification state

**Session/State Management:**
- `self.session` — Conversation session with metadata dict (clarification_state, etc.)
- `self.session.messages` — Message history for context collection
- `self.instructions` — Template loader/renderer for system/user prompts

**Tool Integration:**
- `self.tools.list_tools()` — Check available tools (e.g., web_fetch)
- `self._execute_tool_with_guard()` — Safe tool execution with usage tracking
- `self._emit_tool_output()` — Emit structured tool outputs for UI/logging
- `self._add_session_message()` — Append tool results to session

**LLM Interaction:**
- `self._complete_with_guards()` — Call LLM with token/safety guards
- `self._extract_json_object()` — Parse JSON from model responses

**Helper Methods (assumed from class):**
- `self._is_monitor_only_tool_name()` — Filter monitor-only tools from context
- `self._current_session_slug()` — Get session identifier
- `self._iter_pipeline_nodes()`, `self._iter_pipeline_leaves()` — Task tree traversal
- `self._merge_unique_urls()`, `self._extract_urls()` — URL utilities

---

## Key Patterns & Design Decisions

**Clarification Loop:**
- Tracks pending clarification state in session metadata
- Detects assistant questions (2+ question marks, clarification prompts)
- Validates user follow-ups (short, non-command, non-question)
- Merges context from last assistant response instead of re-posing original question

**Contract Pipeline:**
- Planner generates tasks + requirements + prefetch URLs
- Critic validates response against requirements
- Fallback contract provided if planner fails
- Supports both simple single-action tasks (bypass) and complex multi-step (use contract)

**List Processing:**
- Detects list-processing intent via regex patterns (per-item language, format specs, output tasks)
- Extracts members with confidence scoring
- Tracks coverage via alias matching (handles "Name — URL" format)
- Supports direct (tool-based) and script (Python worker) strategies

**Task Rephrasing:**
- Complexity scoring: URLs, list markers, format specs, line count, file naming
- Threshold: ≥3 signals justify rephrasing
- Sanity checks: non-empty, ≥30% of original length
- Separate script-mode rephraser for explicit script requests

**Source Report Pipeline:**
- Detects "check all sources" intent
- Prefetches all URLs via web_fetch
- Validates report structure (Source N headings, Conclusion section)
- Per-source tagging in tool outputs

**Safety & Limits:**
- Max task tree: 8 root tasks, 12 depth, 36 total nodes
- Max list members: 150 (configurable per function)
- Max requirements: 10
- Max prefetch URLs: 20
- Token escalation: first attempt 3200 tokens, retry up to 6400