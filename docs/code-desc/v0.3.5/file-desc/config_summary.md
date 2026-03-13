# Summary: config.py

# Summary: config.py

Comprehensive configuration management system for Captain Claw using Pydantic v2 with hierarchical nested models, YAML persistence, environment variable overrides, and legacy compatibility handling. Supports 30+ subsystems including LLM providers, tools, skills, memory backends, UI platforms (Telegram/Slack/Discord), and advanced features like chunked processing, deep memory, and orchestration.

## Purpose

Solves the problem of managing complex, multi-layered configuration across diverse subsystems (LLM providers, 30+ tools, skills ecosystem, memory backends, UI integrations, security guards) while maintaining:
- **Backward compatibility** with legacy flat config structures
- **Security-sensitive overrides** via environment variables for tokens/API keys
- **Flexible precedence** (local project config < home directory config < environment variables)
- **Type safety** through Pydantic validation
- **Extensibility** for new tools, skills, and providers without schema changes

## Most Important Functions/Classes

### 1. **Config (BaseSettings class)**
Main configuration root aggregating 25+ nested config sections (model, memory, tools, skills, guards, session, workspace, UI platforms, orchestrator, datastore, etc.). Implements `load()` class method for environment-aware initialization with security-sensitive overrides for tokens, `from_yaml()` for file-based loading with local/home precedence merging, `save()` for persistence, and `resolved_workspace_path()` for runtime path resolution. Handles legacy config migration via `_apply_legacy_top_level_config()`.

### 2. **Config.load() (class method)**
Orchestrates multi-source configuration assembly: YAML loading → environment variable overlays → security-sensitive token injection (Telegram, Slack, Discord, API keys) → provider auto-detection → provider_keys export to os.environ. Implements sophisticated fallback chains (e.g., OPENAI_API_KEY → provider_keys.openai → config.model.api_key) and auto-enables Google OAuth when credentials present. Critical for production deployments where secrets come from environment.

### 3. **Config._deep_merge() (static method)**
Recursive dictionary merger enabling local config.yaml (project defaults) + home config.yaml (user overrides) composition. Preserves nested structure integrity while allowing overlay precedence—essential for supporting both shipped defaults and user customization without duplication.

### 4. **ModelConfig + AllowedModelConfig (nested classes)**
Defines LLM provider configuration with per-model overrides: provider selection (ollama/openai/anthropic/gemini), base_url, temperature, max_tokens, rate limiting (tokens_per_minute), context windows, reasoning levels, and multi-model selection for live per-session switching. AllowedModelConfig enables dynamic model roster with task-specific descriptions and model_type classification (llm/image/video/audio/multimodal).

### 5. **ToolsConfig (class)**
Aggregates 20+ tool configurations (shell, read, write, web_fetch, web_search, browser, image_gen, image_ocr, send_mail, typesense, etc.) with per-tool settings. Enforces _ALWAYS_ENABLED frozenset to prevent accidental disabling of critical tools (read, write, edit, personality, playbooks, browser, datastore). Includes duplicate_call_max safety limit and require_confirmation list for sensitive operations.

### 6. **MemoryConfig + MemoryEmbeddingsConfig + MemorySearchConfig (nested classes)**
Layered memory subsystem: workspace/session file indexing with semantic embeddings (auto/litellm/ollama providers), hybrid retrieval scoring (vector_weight=0.65, text_weight=0.35), temporal decay, chunk overlap, cache TTL. Supports cross-session retrieval and auto-sync. Configurable file type filtering and directory exclusion for efficient indexing.

### 7. **BrowserToolConfig + PinchTabConfig (classes)**
Browser automation configuration: Playwright-based browser with screenshot capture, network recording, credential encryption, cookie persistence, login timeout handling. PinchTabConfig provides alternative token-efficient HTTP-based browser control via accessibility tree snapshots (Go binary) with stealth levels, ad/image blocking, and JavaScript evaluation controls.

### 8. **SkillsConfig + SkillEntryConfig + SkillsLoadConfig (nested classes)**
Skills ecosystem management: managed directory, bundled skill allowlist, per-skill overrides (enabled/api_key/env/config), dynamic loading with watch/debounce, dependency install preferences (brew/npm/pnpm/yarn/bun), prompt injection limits (max_skills_in_prompt=64, max_skills_prompt_chars=16000), and GitHub search integration for skill discovery.

### 9. **GuardConfig + GuardTypeConfig (classes)**
Safety guardrails for input/output/script-tool operations with configurable levels: "stop_suspicious" (block risky operations) or "ask_for_approval" (require user confirmation). Enables content filtering and execution control without blocking legitimate use.

### 10. **DeepMemoryConfig + DatastoreConfig (classes)**
Long-term searchable memory via Typesense (deep_memory) with auto-embedding, and user-facing relational datastore (SQLite) with table/row limits, query result caps, and export controls. Enables persistent knowledge accumulation and structured data management across sessions.

## Architecture & Dependencies

**Dependency Stack:**
- `pydantic` (v2): BaseModel/BaseSettings for validation, serialization, field aliasing
- `pydantic_settings`: SettingsConfigDict for env var prefix/nesting/file handling
- `yaml`: Safe YAML parsing/dumping with legacy enum sanitization
- `pathlib`: Cross-platform path handling with expanduser() for ~/ expansion
- `os`: Environment variable access
- `re`: Regex-based legacy enum tag sanitization

**Key Architectural Patterns:**
1. **Nested composition**: 25+ config classes compose into single Config root via Field(default_factory=...)
2. **Precedence chain**: Environment variables > YAML home directory > YAML local directory > hardcoded defaults
3. **Legacy migration**: Regex-based YAML tag sanitization + _apply_legacy_top_level_config() for backward compatibility
4. **Security-sensitive overrides**: Explicit env var checks for tokens (Telegram, Slack, Discord, API keys) bypass normal Pydantic precedence
5. **Aliasing**: populate_by_name=True enables both snake_case (Python) and camelCase (JSON/UI) field names
6. **Global singleton**: get_config()/set_config() pattern for application-wide access

**Role in System:**
Central configuration hub that decouples subsystem implementations from deployment-specific settings. Enables:
- Multi-environment deployments (dev/staging/prod) via .env files
- User customization via ~/.captain-claw/config.yaml without modifying project files
- Dynamic tool/skill/model selection at runtime
- Security isolation of credentials via environment variables
- Type-safe configuration with Pydantic validation preventing runtime errors