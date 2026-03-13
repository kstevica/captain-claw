# Summary: edit.py

# Edit.py Summary

A surgical file editing tool that provides both string-match and line-based operations for precise text file modifications with automatic backup, undo capability, and atomic writes.

## Purpose

Solves the problem of safely modifying existing text files through an AI agent or automation system. Enables precise edits via multiple strategies (find-and-replace, anchor-based insertion, line-range operations) while maintaining data integrity through automatic backups, atomic writes, and undo functionality. Handles encoding detection, line-ending normalization, and prevents ambiguous matches.

## Architecture & Dependencies

**Core Dependencies:**
- `captain_claw.logging` — structured logging
- `captain_claw.tools.registry` — Tool base class and ToolResult
- `captain_claw.config` — configuration for backup settings and file size limits
- Standard library: `pathlib`, `tempfile`, `shutil`, `datetime`

**System Role:**
Operates as a registered tool in the captain_claw automation framework. Integrates with file registry and runtime path resolution systems. Supports workflow-run directory context and absolute/relative path resolution matching ReadTool conventions.

## Most Important Functions/Classes

### 1. **`EditTool.execute()` — Main entry point**
Orchestrates the entire edit workflow: validates action and parameters, resolves file paths, performs encoding/binary checks, normalizes line endings, creates backups, dispatches to action handlers, and performs atomic writes. Returns ToolResult with success status and context summary.

### 2. **`_dispatch()` — Action router**
Routes requests to 7 specialized action handlers based on the `action` parameter (replace_string, insert_after, insert_before, delete_string, insert_at_line, delete_lines, replace_lines). Centralizes parameter passing and error handling.

### 3. **String-match actions (5 handlers)**
- `_act_replace_string()` — Find exact text and replace; detects multiple matches and reports line numbers
- `_act_insert_after()` — Insert text after anchor string with line-ending normalization
- `_act_insert_before()` — Insert text before anchor string
- `_act_delete_string()` — Remove exact text match
All validate uniqueness of anchor/target strings and provide context-aware error messages.

### 4. **Line-based actions (3 handlers)**
- `_act_insert_at_line()` — Insert at 1-indexed line number
- `_act_delete_lines()` — Delete inclusive line range
- `_act_replace_lines()` — Replace line range with new content
All perform bounds checking and return line counts.

### 5. **Backup & undo system**
- `_create_backup()` — Creates timestamped backups in config-specified directory, prunes old backups
- `_get_latest_backup()` / `_undo()` — Restores most recent backup (itself backed up for undo-ability)
- `_backup_dir_for()` — Computes safe backup directory from relative file paths

### 6. **Safety mechanisms**
- `_atomic_write()` — Writes via temp file + rename to prevent corruption on failure
- `_resolve_path()` — Multi-strategy path resolution (absolute, relative to runtime base, workflow-run directory, file registry)
- `_normalize_line_endings()` — Converts all line endings to file's native style (CRLF vs LF)
- `_context_around()` — Generates annotated context output showing edit location with line numbers

## Key Design Patterns

**Validation-first:** Parameter validation, path resolution, encoding checks, and size limits occur before any modification.

**Uniqueness enforcement:** String-match operations require exact single matches; multiple matches report all line numbers to guide user refinement.

**Encoding resilience:** UTF-8 primary, falls back to latin-1, detects binary via null bytes in first 8KB.

**Line-ending agnostic:** Detects file's native line ending and normalizes input parameters to match.

**Immutable backups:** Each edit creates a timestamped backup; undo itself creates a backup, enabling undo chains.

**Atomic writes:** Prevents partial writes via temp file in same directory + atomic rename.

**Context-aware feedback:** All operations return surrounding lines with line numbers and edit markers (>>>) for verification.

## Configuration Integration

Reads from `get_config().tools.edit`:
- `max_file_bytes` — file size limit
- `backup_enabled` — enable/disable backups
- `backup_dir` — backup storage location
- `max_backups` — retention limit per file

## Error Handling

Comprehensive validation with specific error messages:
- Unknown actions, missing required parameters
- File not found, not a file, too large, binary, unreadable
- String not found, multiple matches (with line numbers)
- Line range out of bounds
- Empty anchor/old_string values