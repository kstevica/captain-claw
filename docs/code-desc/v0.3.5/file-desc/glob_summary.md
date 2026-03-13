# Summary: glob.py

# glob.py Summary

**Summary:**
A file discovery tool that locates files matching glob patterns across workspace and workflow directories with case-insensitive matching, timestamp filtering, and support for multiple search scopes. Integrates with the Captain Claw orchestration framework as a registered Tool and handles both standard glob patterns and custom read folder searches.

**Purpose:**
Solves the problem of flexible file discovery in multi-directory environments where users need to find files by pattern across workspace files, workflow-generated outputs, and configured extra read directories. Addresses case-sensitivity issues on macOS and provides intelligent filtering to avoid stale workflow artifacts while preserving pre-existing user files.

**Most Important Functions/Classes/Procedures:**

1. **`GlobTool.execute()` (async method)**
   - Main entry point for file discovery; orchestrates pattern resolution, scope handling, and result formatting
   - Manages three search scopes: workspace (default), workflow (task outputs), and custom root directories
   - Applies timestamp filtering to workflow outputs while preserving workspace inputs; returns workspace-relative paths for conciseness
   - Implements cross-scope hints to guide users when searches return no results

2. **`_case_insensitive_walk()` (function)**
   - Recursively traverses directories using `os.walk()` and matches filenames case-insensitively via `fnmatch`
   - Extracts only the filename portion of patterns (e.g., `*kartica*` from `**/*kartica*stranke*`) to match basenames
   - Solves the problem that Python's `glob.glob()` is case-sensitive even on macOS; critical for finding user files like "Kartica_stranke.pdf" when searching "*kartica*"

3. **`_file_modified_after()` (function)**
   - Checks if a file's modification time exceeds a cutoff timestamp using `os.path.getmtime()`
   - Used to filter out stale workflow artifacts from previous orchestration runs while preserving pre-existing workspace files
   - Gracefully handles inaccessible files by returning `True` (includes them) to avoid false negatives

**Architecture & Dependencies:**

- **Framework Integration:** Extends `Tool` base class from `captain_claw.tools.registry`; returns `ToolResult` objects
- **Async Design:** Uses `asyncio.get_event_loop().run_in_executor()` to run blocking filesystem operations without blocking the event loop
- **Configuration:** Dynamically loads extra read directories from `captain_claw.config.get_config().tools.read.extra_dirs`
- **Logging:** Integrates structured logging via `captain_claw.logging.get_logger()`
- **Key Parameters:** Pattern (required), root, scope ("workspace"/"workflow"), limit (default 100)
- **Timeout:** 10 seconds (appropriate for local filesystem scans)

**Role in System:**
Acts as a foundational discovery tool in the Captain Claw orchestration framework, enabling downstream tasks to locate input files, reference previous outputs, and search across multiple logical file domains. The scope mechanism allows tasks to distinguish between user-provided workspace files (immutable inputs) and task-generated workflow outputs (transient artifacts), with intelligent filtering to prevent cross-run contamination.