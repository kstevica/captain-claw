# Summary: agent_file_ops_mixin.py

# Summary: agent_file_ops_mixin.py

## Overview
A comprehensive mixin class providing file operations, script generation, execution, and output handling for an AI agent system. Handles code extraction, script synthesis, structured result wrapping, file verification, and automated script/file writing workflows. Integrates with an LLM-based agent to manage script lifecycle from generation through execution with result capture.

## Purpose
Solves the problem of reliably generating, persisting, executing, and capturing results from dynamically created scripts within an agent workflow. Provides deterministic path management, structured result protocols, verification mechanisms, and automatic script/file writing when users explicitly request these operations.

## Most Important Functions/Classes/Procedures

### 1. **`_build_structured_result_wrapper(code: str) -> str`**
Wraps user Python code in a structured execution harness that captures stdout, exceptions, and return values as JSON. Creates a deterministic protocol for extracting execution results regardless of script complexity. Essential for reliable result capture from arbitrary user code.

### 2. **`_run_python_worker_for_list_task(...) -> dict[str, Any]`**
Orchestrates end-to-end execution of generated Python workers for batch/list processing tasks. Synthesizes code, writes to disk with verification, executes with timeout handling, captures structured results, and updates execution context with artifacts and variables. Handles session scoping, tool allowlists, and task policies.

### 3. **`_write_file_with_verification(...) -> dict[str, Any]`**
Writes files via tool execution with retry logic and on-disk verification. Attempts to confirm file persistence before returning success. Handles multiple candidate paths (absolute, relative, saved directory variants) and emits detailed tool output for session tracking.

### 4. **`_maybe_auto_script_requested_output(...) -> str`**
Guarantees explicit script requests produce write+run tool actions. Extracts code blocks, synthesizes missing code, wraps Python scripts, writes to disk, and executes with result capture. Appends execution results to assistant output when successful.

### 5. **`_maybe_auto_write_requested_output(...) -> str`**
Auto-executes write tool when user explicitly requests file output. Handles both single-file and multi-file scenarios (extracting named file blocks from assistant output). Retries failed writes and provides detailed success/failure summaries.

### 6. **`_extract_code_block(text: str) -> tuple[str | None, str | None]`**
Parses first fenced code block from text, returning (language, code). Uses regex to handle markdown code fences with optional language tags.

### 7. **`_infer_script_extension(language: str | None, code: str) -> str`**
Determines file extension from language tag or shebang line. Maps common language identifiers (python, bash, javascript, ruby) to extensions with fallback to `.py`.

### 8. **`_build_python_runner_command(script_path: Path, result_path: Path | None = None) -> str`**
Constructs shell command using active Python interpreter with proper directory context and optional result file argument. Handles path quoting for safety.

### 9. **`_synthesize_script_content(user_input: str, turn_usage: dict) -> tuple[str, str]`**
Generates script content via LLM when model omits code block. Falls back to deterministic scaffold if synthesis fails. Returns (code, extension) tuple.

### 10. **`_build_python_worker_prompt(...) -> str`**
Constructs synthesis prompt for batch/list worker scripts with requirements for iteration, output paths, progress logging, and structured return values. Includes list members and per-member action context.

## Architecture & Dependencies

**Key Dependencies:**
- `captain_claw.llm.Message` - LLM message protocol
- `captain_claw.logging.get_logger` - Structured logging
- Standard library: `pathlib`, `json`, `re`, `shlex`, `datetime`, `sys`

**Session Integration:**
- Requires `self.session` (message history), `self.session_manager` (persistence)
- Requires `self.tools` (file/shell execution), `self.instructions` (prompt templates)
- Requires `self.runtime_base_path`, `self.memory` (optional background sync)

**Tool Execution Pattern:**
- Delegates actual file writes/shell execution to `_execute_tool_with_guard()` (inherited)
- Wraps tool calls with session message tracking via `_add_session_message()`
- Emits structured output via `_emit_tool_output()`

**Execution Context Management:**
- Tracks active task nodes, artifacts, variables, and execution events
- Supports session/task policy enforcement and tool allowlists
- Integrates with planning pipeline for multi-step workflows

**Result Capture Protocol:**
- Uses marker `# captain-claw: structured-result-protocol-v1` to identify wrapped scripts
- Writes JSON result payloads to deterministic paths: `saved/tmp/{session_slug}/script_results/{stem}_{timestamp}.result.json`
- Parses structured results to extract stdout, data, or error information

**Path Management:**
- Normalizes session IDs to safe folder slugs (alphanumeric, dots, hyphens)
- Supports relative paths (scripts/, tools/, saved/) and absolute paths
- Verifies written files against multiple candidate locations before confirming success