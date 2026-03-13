# Summary: write.py

# write.py Summary

**Summary:**
WriteTool is a file-writing utility that safely creates or appends content to files within a sandboxed workspace environment. It enforces session-based path scoping, sanitizes content (removing control characters and unescaping HTML entities), and provides detailed feedback about overwrites and file locations.

**Purpose:**
Solves the problem of allowing LLM agents to write files safely within a bounded workspace while preventing directory traversal attacks, enforcing session isolation, and maintaining a registry of logical-to-physical file mappings for cross-task file discovery.

**Most Important Functions/Classes:**

1. **WriteTool.execute()** — Main async entry point that orchestrates the entire write operation. Handles path resolution (either workflow-run or session-scoped), content sanitization, file I/O, overwrite detection, and file registry updates. Returns detailed success/failure feedback.

2. **WriteTool._normalize_under_saved()** — Core path security function that maps any requested path into the saved root directory while enforcing session scoping. Implements passthrough logic for `<workspace>/output/` paths, strips dangerous traversals (`..`), and automatically categorizes files into predefined folders (downloads, media, output, scripts, etc.). Returns a safe, resolved Path object.

3. **WriteTool._resolve_saved_root()** — Determines the base directory for tool-managed outputs by checking `_saved_base_path`, `_runtime_base_path`, or defaulting to `<cwd>/saved`. Creates the directory structure if it doesn't exist.

4. **WriteTool._normalize_session_id()** — Converts raw session identifiers into filesystem-safe strings by removing/replacing unsafe characters and defaulting to "default" for empty inputs. Ensures session IDs can be safely used in file paths.

5. **Content sanitization pipeline** — Removes C0/C1 control characters that LLMs emit when failing to reproduce Unicode, and unescapes HTML entities (`&lt;`, `&gt;`, `&amp;`, `&quot;`) for markup files (HTML, SVG, XML). Preserves normal whitespace.

**Architecture Notes:**
- Implements the Tool interface from captain_claw.tools.registry
- Uses late-import guard for FileRegistry to avoid circular dependencies
- Supports dual modes: workflow-run (direct path mapping) and session-scoped (sandboxed)
- Integrates with optional file registry for logical-to-physical path tracking
- 10-second timeout appropriate for local file operations
- Provides user-facing warnings about overwrites and hints to prevent wasteful read-after-write patterns