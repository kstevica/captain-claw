# Summary: instructions.py

# instructions.py Summary

**Summary:**
This module implements a two-layer instruction template loading system with personal override support for the Captain Claw LLM framework. It provides transparent resolution between system defaults and user customizations, with optional micro-template variants that fall back to standard templates when unavailable.

**Purpose:**
Solves the problem of managing LLM instruction templates across multiple environments (system vs. personal) while allowing users to customize templates without modifying source code. Enables safe template rendering with partial variable substitution and tracks template usage for logging purposes.

**Most Important Functions/Classes:**

1. **InstructionLoader class**
   - Core orchestrator managing template resolution, caching, and rendering. Implements the four-layer resolution hierarchy (personal micro → base micro → personal standard → base standard) and handles auto-detection of micro mode from global config.

2. **_path(name: str) → Path**
   - Implements the resolution logic determining which template file to load based on micro mode and file existence. Checks personal directory before base directory at each layer.

3. **render(name: str, **variables) → str**
   - Renders templates using `str.format_map()` with safe placeholder handling via `_SafeFormatDict`, allowing partial substitution without raising KeyError on unknown variables.

4. **load(name: str) → str**
   - Loads template content with caching and recent-file tracking for LLM session logging. Raises FileNotFoundError with helpful context if template doesn't exist.

5. **_SafeFormatDict class**
   - Custom dict subclass overriding `__missing__()` to preserve unknown placeholders as literal strings (e.g., `{unknown}`) rather than raising exceptions, enabling graceful partial template rendering.

**Architecture Notes:**
- **Two-tier directory structure:** Personal overrides (`~/.captain-claw/instructions/`) take precedence over system defaults (project `instructions/` folder or `CAPTAIN_CLAW_INSTRUCTIONS_DIR` env var)
- **Caching strategy:** In-memory cache reduces disk I/O; `_recent_files` list tracks usage for drain-on-demand logging
- **Micro-template system:** Optional feature (controlled by `context.micro_instructions` config) that transparently tries `micro_<name>` variants before falling back to standard templates
- **Safe rendering:** `_SafeFormatDict` enables partial template substitution, leaving unresolved placeholders intact for downstream processing