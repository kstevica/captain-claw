# Summary: skills.py

# skills.py Summary

Comprehensive skill discovery, loading, filtering, and invocation system for the Captain Claw framework. Manages skill metadata parsing, GitHub-based skill installation, dependency resolution, command dispatch, and prompt injection. Handles multiple skill sources with precedence-based merging and supports complex installation workflows (brew, npm, go, uv, direct downloads).

## Purpose

Solves the problem of discovering, validating, installing, and managing reusable "skills" (task automation modules) across multiple sources (bundled, managed, plugins, workspace). Provides a unified interface for loading skill metadata from SKILL.md files, filtering by platform/requirements, building command specs for user/model invocation, and managing skill-specific environment/API key overrides.

## Most Important Functions/Classes/Procedures

### 1. **build_workspace_skill_snapshot()**
Orchestrates the complete skill discovery pipeline: loads all skill entries from configured roots, filters by requirements/platform/config, applies prompt limits, builds command specs, and returns a SkillSnapshot containing formatted prompt text and routable commands. Central function for preparing skills for LLM context injection and command routing.

### 2. **install_skill_from_github_url()**
Handles end-to-end skill installation from GitHub URLs. Parses GitHub URLs (supports /tree/<ref>/<path> and /blob/<ref>/SKILL.md formats), attempts GitHub API download with fallback to git clone, validates SKILL.md presence, sanitizes destination paths, and copies skill into managed directory. Returns SkillInstallResult with metadata.

### 3. **install_skill_dependencies()**
Resolves and executes runtime dependency installation for a skill. Selects appropriate installer (brew/npm/go/uv/download) based on platform and configuration preferences, handles archive extraction with path normalization and security checks, and returns detailed SkillDependencyInstallResult with stdout/stderr/exit codes.

### 4. **filter_skill_entries()**
Applies multi-criteria filtering to loaded skills: checks bundled skill allowlist, verifies enabled status in config, validates platform compatibility, checks OS requirements, evaluates metadata.requires (bins/env/config), and handles "always" metadata flag. Returns filtered list eligible for prompt/command inclusion.

### 5. **load_workspace_skill_entries()**
Discovers and loads raw skill entries from all configured roots with precedence-based deduplication. Resolves skill roots from config (extra_dirs, plugin_dirs, bundled, managed, agents, workspace), iterates SKILL.md candidates, parses frontmatter/metadata, and merges by skill name with later sources overriding earlier ones.

### 6. **parse_github_skill_source()**
Parses and validates GitHub URLs into GitHubSkillSource objects. Supports https://github.com/owner/repo, /tree/<ref>/<skill-path>, and /blob/<ref>/<path-to-SKILL.md> formats. Validates scheme/host, extracts path components, normalizes skill paths, and prevents directory traversal attacks.

### 7. **resolve_skill_command_invocation()**
Parses user command text (`/skill <name> [args]` or `/<command> [args]`) into SkillCommandSpec + args tuple. Handles both direct command invocation and `/skill` prefix syntax, normalizes skill name lookup, and returns None for non-skill commands.

## Key Data Structures

- **SkillEntry**: Parsed skill metadata (name, description, file_path, source, frontmatter, metadata, invocation policy)
- **SkillMetadata**: Structured metadata from SKILL.md frontmatter (emoji, homepage, requires, install specs, OS list, skill_key)
- **SkillInstallSpec**: Single installation method (kind: brew/node/go/uv/download, formula/package/module/url, archive extraction config)
- **SkillSnapshot**: Runtime skill state (prompt text, eligible skills, command specs, version hash)
- **SkillCommandSpec**: User-invocable command (name, skill_name, description, dispatch config)
- **SkillDependencyInstallResult**: Installation outcome (ok, stdout/stderr, exit code, error message)

## Architecture & Dependencies

**External Dependencies**: httpx (GitHub API/downloads), yaml (frontmatter parsing), tarfile/zipfile (archive extraction), subprocess (git/brew/npm/go/uv), pathlib (path resolution)

**Internal Dependencies**: captain_claw.config (Config, SkillEntryConfig), captain_claw.logging

**Source Precedence** (highest to lowest): extra → plugin → bundled → managed → agents-personal → agents-project → workspace. Later sources override earlier ones by skill name.

**Key Design Patterns**:
- Path normalization with security checks (prevents directory traversal in archives/GitHub paths)
- Platform-aware filtering (linux/darwin/win32)
- Graceful fallback (GitHub API → git clone)
- Lazy loading with candidate limits (max_candidates_per_root, max_skills_loaded_per_source)
- Environment override management with automatic restoration
- Deterministic versioning via SHA1 hash of SKILL.md paths/mtimes