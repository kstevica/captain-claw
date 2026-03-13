# Summary: agent.py

# Agent.py Summary

Core orchestration engine for Captain Claw that coordinates LLM interactions, tool execution, and multi-agent workflows through a mixin-based architecture. Manages session state, token usage tracking, callback routing, and runtime configuration while providing unified interface to 13+ specialized agent capabilities.

## Purpose

Solves the problem of coordinating complex AI agent behavior across multiple concerns: LLM provider abstraction, tool invocation loops, session management, guard rails, file operations, reasoning chains, and scaling detection. Acts as the central hub that delegates to specialized mixins while maintaining consistent state and metrics.

## Most Important Functions/Classes/Procedures

1. **`__init__()` Constructor**
   - Initializes agent with optional LLM provider override and 5 callback types (status, tool_output, approval, thinking, tool_stream)
   - Sets up tool registry, workspace paths, session manager, and runtime configuration
   - Configures pipeline mode ("loop" vs "contracts"), max iterations (40), and monitoring flags
   - Establishes cancel_event for graceful shutdown and llm_session_logging for audit trails

2. **`_set_runtime_status(status: str)` & `_emit_thinking(text, tool, phase)`**
   - Callback forwarding methods that safely route runtime updates and inline reasoning to UI layer
   - Enable real-time monitoring of agent execution without tight coupling to UI implementation
   - Include exception handling to prevent callback failures from crashing agent

3. **`_emit_tool_output(tool_name, arguments, output)` & `_log_llm_call()`**
   - Dual-path output handling: adds monitor-only tools (planning, task_contract, completion_gate) to session history while optionally forwarding to external callback
   - Generates pipeline_trace payloads for observability when monitor_trace_pipeline enabled
   - LLM call logging writes interaction details to file-based session log for audit/debugging

4. **`_accumulate_usage()` & `_finalize_turn_usage()`**
   - Token accounting system tracking prompt/completion/cache tokens across turns and globally
   - Separates last_usage (current turn) from total_usage (cumulative) for granular metrics
   - Supports Anthropic cache tokens via LiteLLM integration

5. **`_is_monitor_only_tool_name()` Static Method**
   - Filters tools that generate observability/tracing output (llm_trace, planning, task_contract, pipeline_trace, telegram) from model context
   - Prevents feedback loops where monitoring tools influence subsequent LLM decisions

## Architecture & Dependencies

**Mixin Inheritance Chain** (13 mixins):
- **Orchestration**: AgentOrchestrationMixin (primary coordination)
- **Execution**: AgentCompletionMixin, AgentToolLoopMixin, AgentScaleLoopMixin (task completion & tool loops)
- **State Management**: AgentSessionMixin, AgentContextMixin (session/context tracking)
- **LLM Integration**: AgentModelMixin, AgentCompletionMixin (model interaction)
- **Processing**: AgentChunkedProcessingMixin, AgentPipelineMixin (chunked/pipeline execution)
- **Intelligence**: AgentReasoningMixin, AgentResearchMixin, AgentScaleDetectionMixin (reasoning & research)
- **Capabilities**: AgentFileOpsMixin, AgentSkillsMixin, AgentPlaybookMixin (file ops, skills, playbooks)
- **Safety**: AgentGuardMixin (approval gates & safety checks)

**Key Dependencies**:
- `LLMProvider` (abstract provider for Claude/GPT/etc via LiteLLM)
- `FileRegistry` (cross-task file path resolution)
- `Session` & `SessionManager` (conversation history & persistence)
- `InstructionLoader` (dynamic instruction file management)
- `ToolRegistry` (tool discovery & execution with approval callbacks)
- `LLMSessionLogger` (audit trail to logs/<session>/session_log.md)
- `Config` (workspace paths, UI settings, logging configuration)

**State Variables**:
- `provider`: LLM provider instance (overridable)
- `session`: Current Session object (None until initialized)
- `tools`: ToolRegistry instance with approval callback wired
- `workspace_base_path`: Resolved workspace directory
- `cancel_event`: asyncio.Event for external cancellation
- `pipeline_mode`: "loop" (fast) or "contracts" (planner+critic)
- `max_iterations`: 40 tool calls per message limit
- `llm_session_logging`: Boolean flag for audit logging
- `_file_registry`: Optional FileRegistry for orchestration runs
- `_is_worker`: Boolean flag for orchestrator worker agents

**Configuration Flow**: Runtime config loaded via `get_config()` → workspace paths resolved → tool registry initialized with approval callback → session manager attached → LLM logger configured → runtime model details refreshed.