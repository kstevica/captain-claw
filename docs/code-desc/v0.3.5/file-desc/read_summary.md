# Summary: read.py

# read.py Summary

A file reading tool that safely retrieves and returns file contents with support for partial reads (line-based offset and limit), intelligent path resolution across multiple contexts, and comprehensive error handling.

## Purpose

Solves the problem of reliably reading file contents in a multi-context execution environment where files may be located in different directories (workspace root, process CWD, workflow-run directories, or registered file locations). Provides bounded access through file size limits and line-based pagination to prevent resource exhaustion.

## Most Important Functions/Classes

1. **ReadTool class** — Main tool implementation inheriting from `Tool` base class. Defines the tool's metadata (name, description, timeout), JSON schema for parameters (path, limit, offset), and orchestrates the entire read operation.

2. **execute() method** — Core async execution function that:
   - Validates and blocks reads of temporary artifacts (download.bin)
   - Resolves file paths through a cascading fallback strategy (absolute paths → workspace-relative → CWD-relative → workflow-run directory → file registry)
   - Enforces file size limits via configuration
   - Implements line-based pagination (offset/limit)
   - Returns structured ToolResult with success status, content, and metadata

3. **Path resolution logic** — Multi-stage fallback mechanism handling:
   - User-expanded paths (`~` expansion)
   - Absolute vs. relative path distinction
   - Workspace root resolution via `_runtime_base_path`
   - CWD-based fallback for files written by other tools
   - Workflow-run directory lookup with both relative and flat (filename-only) matching
   - File registry resolution as final fallback

4. **Error handling and validation** — Comprehensive checks for:
   - File existence across multiple resolution contexts
   - File type validation (must be regular file, not directory)
   - File size enforcement against configured maximum
   - UTF-8 encoding compatibility
   - Exception catching with structured logging

5. **Content processing** — Line-based pagination supporting:
   - 1-indexed offset parameter (line number to start from)
   - Optional limit parameter (maximum lines to return)
   - Metadata annotation showing file path, character count, and line range

## Architecture & Dependencies

- **Framework**: Integrates with `captain_claw` tool registry system via `Tool` base class
- **Logging**: Uses centralized logger from `captain_claw.logging`
- **Configuration**: Reads max file size from `captain_claw.config`
- **Async**: Implements async/await pattern for non-blocking execution
- **Type hints**: Full Python 3.10+ type annotations including union types (`int | None`)

## Role in System

Acts as a foundational file access tool in an AI agent/automation framework, enabling safe, bounded file reading across distributed execution contexts with intelligent path resolution and resource protection.