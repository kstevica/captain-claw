# Summary: rest_skills.py

# rest_skills.py Summary

Comprehensive REST API handler module for skills management, directory browsing, and file access configuration in the Captain Claw system. Provides endpoints for skill discovery/installation, configuration toggling, filesystem navigation, and integration with Google Drive for extended file access.

## Purpose

Solves the problem of exposing skills management and file system access through HTTP REST endpoints, enabling:
- Dynamic skill discovery, installation, and toggling from a web UI
- Safe filesystem browsing with security constraints (blocked system paths)
- Configuration persistence for skill enablement and extra readable directories
- Google Drive folder integration via the `gws` CLI tool
- File tree caching and enumeration for performance optimization

## Architecture & Dependencies

**Key Dependencies:**
- `aiohttp.web` — async HTTP request/response handling
- `yaml` — configuration serialization/deserialization
- `pathlib.Path` — cross-platform filesystem operations
- `captain_claw.config` — configuration loading/saving and validation
- `captain_claw.skills` — skill loading, filtering, and GitHub installation
- `captain_claw.file_tree_builder` — local and GDrive tree building with caching
- `captain_claw.logging` — structured logging

**System Design:**
- All handlers are async coroutines accepting `(server: WebServer, request: web.Request)`
- Configuration changes validated through `Config(**merged_data)` before persistence
- Path security enforced via blocked system directory lists (Unix/Windows)
- Caching layer for expensive tree-building operations with TTL support
- Cross-platform drive enumeration (Windows-specific)

## Most Important Functions/Classes/Procedures

### 1. **list_skills(server, request) → web.Response**
Retrieves all workspace skill entries with rich metadata including enablement state, requirements, and invocation settings. Filters active skills and resolves skill keys from metadata. Returns JSON array with name, description, source, emoji, homepage, invocation flags, enabled state, and dependency requirements (bins, env vars, config keys).

### 2. **toggle_skill(server, request) → web.Response**
Enables/disables a skill via YAML config modification. Handles skill_key resolution, config merging, and validation. Removes redundant config entries when enabling (default state) to keep config clean. Persists changes through `_save_config()` with full validation.

### 3. **install_skill(server, request) → web.Response**
Installs a skill from a GitHub URL by delegating to `install_skill_from_github_url()`. Extracts URL from JSON body, validates presence, executes installation, and returns skill name, destination path, and repo metadata. Error handling with detailed logging.

### 4. **browse_directory(server, request) → web.Response**
Safe filesystem directory browser returning subdirectories for a given path. Expands user paths, validates existence/type, filters hidden directories, and flags blocked system paths. Returns parent directory link and blocked status for UI security warnings.

### 5. **get_folder_trees(server, request) → web.Response**
Aggregates file tree listings for all configured local and GDrive folders with caching. Respects max_entries and max_depth limits. Implements TTL-based cache with fallback to live tree building. Returns mixed array of local and GDrive tree objects with error handling per folder.

### 6. **_is_blocked_path(p: Path) → bool**
Security validation function checking if a path is a filesystem root or known system directory. Resolves path and compares against platform-specific blocked sets (_BLOCKED_UNIX, _BLOCKED_WIN). Prevents accidental modification of critical system directories.

### 7. **_save_config(config_path, data) → None**
Atomic configuration persistence with validation. Merges provided data with existing local config, validates merged result through Config class (raises on error), creates parent directories, and writes YAML. Reloads global config state via `set_config()`.

### 8. **add_read_folder / remove_read_folder (server, request) → web.Response**
Manage extra readable directories in config. Add validates path existence, blocks system directories, prevents duplicates (by resolved path), and appends to config. Remove filters by resolved path and cleans up empty config sections. Both persist via `_save_config()`.

### 9. **add_gdrive_folder / remove_gdrive_folder / browse_gdrive (server, request) → web.Response**
Google Drive integration endpoints. Add/remove manage gdrive_folders list with duplicate prevention by folder ID. Browse delegates to `browse_gdrive_folders()` async helper, supporting hierarchical folder navigation with folder_id parameter (defaults to "root").

### 10. **list_drives(server, request) → web.Response**
Windows-only endpoint enumerating available drive letters (A-Z) by checking Path existence. Returns empty list on non-Windows platforms. Supports drive selection UI for Windows users.