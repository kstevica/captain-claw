# Summary: rest_instructions.py

# rest_instructions.py Summary

Manages REST API endpoints for instruction file (Markdown) CRUD operations with a two-tier directory system supporting system defaults and personal overrides. Implements file listing with micro-variant detection, read/write/revert operations, and cache invalidation for the agent's instruction system.

## Purpose

Solves the problem of managing instruction documentation with layered configuration—allowing users to override system instructions while maintaining a clean separation between default and customized content. Prevents directory traversal attacks and maintains consistency between the filesystem and in-memory instruction cache.

## Most Important Functions/Classes

1. **`list_instructions(server, request)`**
   - Aggregates instruction files from both system and personal directories, discovers `micro_*` variant files, and returns a JSON list with metadata (name, size, override status, has_micro flag). Excludes micro files from direct listing but tags parent files that have micro counterparts for frontend UI.

2. **`get_instruction(server, request)`**
   - Retrieves instruction file content with personal override precedence over system defaults. Validates filename against path traversal attacks (`..`, `/`, `.md` extension), returns file content with override status metadata.

3. **`put_instruction(server, request)`**
   - Writes instruction content to the personal override directory, creating it if necessary. Validates input paths, invalidates agent instruction cache, and returns status indicating whether file was created or updated.

4. **`revert_instruction(server, request)`**
   - Deletes personal override file and returns the system default content, effectively reverting customizations. Validates that both personal override and system default exist before deletion, clears agent cache.

5. **Path Validation Pattern**
   - Consistent security checks across all endpoints: rejects `..`, `/`, `\` in filenames and validates `.md` extension; `put_instruction` additionally uses `resolve().relative_to()` to prevent symlink/path traversal exploits.

## Architecture & Dependencies

- **Framework**: aiohttp web handlers (async/await pattern)
- **Dependencies**: `WebServer` instance providing `_instructions_dir`, `_instructions_personal_dir`, and optional `agent` with instruction cache
- **Cache Integration**: Invalidates `server.agent.instructions._cache` on write/revert to keep in-memory state synchronized
- **File System**: Pathlib-based file operations with UTF-8 encoding; supports directory creation with `mkdir(parents=True, exist_ok=True)`