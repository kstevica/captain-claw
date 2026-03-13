# Summary: file_tree_builder.py

# file_tree_builder.py Summary

**Summary:**
This module generates compact Unicode tree representations of local directories and Google Drive folders for LLM context injection, eliminating the need for the LLM to call external commands. It provides caching, size formatting, and depth-limited traversal with entry count limits to keep output manageable.

**Purpose:**
Solves the problem of making file system structures visible to LLMs without requiring subprocess calls or repeated filesystem/API queries. Enables the LLM to understand available files and folders before making decisions about which resources to access, improving context awareness and reducing unnecessary API calls.

---

## Most Important Functions/Classes/Procedures

### 1. **build_local_tree(directory, max_entries=50, max_depth=2)**
Recursively walks a local directory and constructs a formatted Unicode tree string with file sizes. Handles permission errors gracefully, sorts entries (directories first), and truncates output when entry limits are exceeded. Returns tuple of (tree_string, entry_count) for caching purposes.

### 2. **build_gdrive_tree(folder_id, folder_name, max_entries=50, max_depth=2)** [async]
Asynchronously lists Google Drive folder contents via the `gws` CLI tool, building a tree structure with file IDs and sizes. Recursively traverses subfolders up to max_depth, respects entry limits, and handles API errors by returning error messages in the tree output. Supports shared drives and all drive types.

### 3. **browse_gdrive_folders(folder_id="root")** [async]
UI-focused function that lists immediate subfolders in a Google Drive location without recursive traversal. When folder_id is "root", also fetches accessible shared drives. Returns structured dict with folders, shared_drives, and error fields for UI consumption.

### 4. **_run_gws(binary, args)** [async]
Executes `gws` CLI commands with JSON output format, handling subprocess communication, timeouts (30s), and error parsing. Decodes stdout/stderr safely and returns either parsed JSON dict or error string, providing unified interface for all gws operations.

### 5. **Cache Management Functions (get_cached_tree, set_cached_tree, clear_cache)**
Simple in-memory cache with TTL support storing (timestamp, tree_str, entry_count) tuples. Enables fast repeated tree generation for the same paths without re-traversing filesystem or making repeated API calls.

---

## Architecture & Dependencies

**Key Dependencies:**
- `asyncio` – Async subprocess execution for gws CLI calls
- `pathlib.Path` – Cross-platform local filesystem operations
- `json` – Parsing gws CLI JSON output
- `shutil.which()` – Binary path resolution
- `captain_claw.logging` – Structured logging
- `captain_claw.config` – Configuration for custom gws binary paths

**System Role:**
Acts as a context preparation layer in the captain_claw system, pre-computing file tree representations that get injected into LLM prompts. Bridges local filesystem and Google Drive APIs through the gws CLI tool, providing unified tree formatting for both sources.

**Design Patterns:**
- **Lazy evaluation with caching** – Trees computed once, cached by key with TTL
- **Graceful degradation** – Missing gws binary, permission errors, and API failures return informative error strings rather than exceptions
- **Depth/entry limiting** – Prevents unbounded tree growth through max_depth and max_entries parameters
- **Async-first for I/O** – Google Drive operations use asyncio for non-blocking CLI calls