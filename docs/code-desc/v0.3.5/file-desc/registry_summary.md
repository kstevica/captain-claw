# Summary: registry.py

# Registry.py Summary

This module implements a comprehensive tool registry and execution framework for managing, validating, and executing tools with policy-based access control and shell command security filtering.

## Purpose

Solves the problem of safely managing a dynamic set of executable tools with:
- Policy-based access control (global, session, task-level filtering)
- Shell command security validation and blocking
- Async execution with timeout and abort handling
- Tool metadata and definition management for LLM integration
- Approval workflows for sensitive operations

## Architecture & Dependencies

**Key Dependencies:**
- `pydantic`: Data validation and model definitions
- `asyncio`: Async execution and timeout management
- `shlex`: Shell command tokenization
- `fnmatch`: Pattern matching for policy rules
- `pathlib`: File path handling
- Custom exceptions: `ToolBlockedError`, `ToolExecutionError`, `ToolNotFoundError`
- Custom logging: `get_logger`
- Custom config: `get_config` for tool policies

**System Role:** Central orchestration layer that bridges LLM tool calls with actual tool execution, enforcing security policies and managing lifecycle.

## Most Important Functions/Classes

### 1. **ToolRegistry** (Main Class)
Manages tool registration, policy resolution, and execution orchestration. Maintains three-level policy chain (global → session → task), handles tool filtering, and coordinates async execution with timeout/abort propagation. Key methods: `register()`, `execute()`, `list_tools()`, `get_definitions()`, `_resolve_policy_chain()`.

**Purpose:** Central registry providing policy-filtered tool access and safe execution with approval workflows.

### 2. **ToolPolicyChain**
Applies cascading policy rules to filter available tools. Normalizes tool names, applies allow/deny/also_allow rules in sequence, and resolves final tool set. Method `resolve()` returns filtered tool list after applying all policy steps.

**Purpose:** Implements hierarchical policy filtering logic to progressively restrict tool access based on context (global → session → task).

### 3. **Tool** (Abstract Base Class)
Base class for all executable tools with async `execute()` method, parameter validation, and LLM definition generation. Subclasses implement tool-specific logic. Provides `get_definition()` for OpenAI-style function schemas and `validate_arguments()` for input validation.

**Purpose:** Standardized interface for all tools ensuring consistent execution contract and LLM integration.

### 4. **Shell Command Security Functions** (Group)
- `extract_shell_base_commands()`: Parses shell commands into segments and extracts base executables
- `is_blocked_shell_command()`: Evaluates commands against blocked patterns with segment-level and base-command-level matching
- `_split_shell_segments()`: Tokenizes commands while preserving control operators (`;`, `&&`, `||`, `|`, `&`)
- `_extract_segment_base_command()`: Isolates executable token from shell segment, skipping wrappers and assignments

**Purpose:** Prevents execution of dangerous shell commands by parsing complex shell syntax and matching against configurable blocklists.

### 5. **ToolResult** (Pydantic Model)
Result container with `success` boolean, `content` string, and optional `error` message. Includes validator ensuring failed results always have error text (fallback to content if needed).

**Purpose:** Standardized tool execution result format with automatic error message normalization for consistent error handling.

## Critical Implementation Details

**Policy Resolution Flow:**
1. Global policy applied first (broadest scope)
2. Session policy applied second (narrows further)
3. Task policy applied last (most specific)
4. Each step filters allowed names, removes denied names, adds also_allowed names

**Shell Execution Security:**
- Commands parsed into segments separated by control operators
- Base commands extracted after skipping wrappers (`sudo`, `command`, `builtin`, `nohup`, `time`)
- Blocked patterns matched at both segment-level (full command) and base-command level
- Invalid regex patterns fall back to literal string matching
- Three-tier policy: deny patterns → allow patterns → default policy (allow/deny/ask)

**Async Execution Handling:**
- Tool execution wrapped in asyncio task with configurable timeout (default 30s)
- External abort event bridged to tool-local abort event
- Timeout triggers task cancellation with proper cleanup
- Approval callback integration for ask-mode policies
- File registry and workflow context passed to tools via kwargs

**Policy Coercion:**
- Accepts `ToolPolicy` objects, dicts, or None
- Normalizes tool names (lowercase, stripped) for case-insensitive comparison
- Maintains separate metadata dict per tool for extensibility

**Runtime Path Management:**
- Base path resolves to runtime directory (where Captain Claw launched)
- Saved directory (`<base>/saved` or custom name) created on-demand
- Path validation ensures saved dir stays within base path