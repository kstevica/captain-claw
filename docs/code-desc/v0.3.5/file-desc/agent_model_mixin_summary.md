# Summary: agent_model_mixin.py

# agent_model_mixin.py Summary

**Summary:**
This mixin provides runtime model selection and provider resolution capabilities for an Agent class, enabling dynamic switching between different LLM providers (OpenAI, Anthropic, Gemini, XAI) and models at both session and configuration levels. It handles provider normalization, API key resolution from environment variables or configuration, and manages model metadata persistence across sessions.

**Purpose:**
Solves the problem of flexible LLM provider/model selection in multi-provider environments by abstracting provider instantiation, API key management, and session-specific model overrides. Enables users to switch between different models without restarting the agent while maintaining configuration hierarchy (environment → .env → config → session metadata).

**Most Important Functions/Classes/Procedures:**

1. **`_normalize_provider_key(provider: str) -> str`**
   - Normalizes provider aliases (e.g., "chatgpt" → "openai", "claude" → "anthropic") to canonical provider names for consistent lookups across the system.

2. **`_resolve_provider_api_key(normalized_provider: str) -> str | None`**
   - Multi-source API key resolution with priority order: environment variables → .env file → provider_keys config fallback. Supports provider-specific key names (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY).

3. **`get_allowed_models() -> list[dict[str, Any]]`**
   - Returns configured allowlist of available models with full specifications (provider, model name, temperature, max_tokens, context limits, reasoning level). Falls back to default config model if allowlist is empty. Supports both dict and object-style config entries.

4. **`_apply_model_option(option: dict[str, Any], source: str, model_id: str) -> None`**
   - Core provider instantiation logic that creates and activates a provider instance with resolved parameters. Handles base_url isolation (prevents config provider's base_url from leaking to different providers), temperature/token overrides, and per-model rate limiting (tokens_per_minute). Resolves correct API keys for provider switching and manages extra headers for auth.

5. **`set_session_model(selector: str, persist: bool = True) -> tuple[bool, str]`**
   - User-facing async method to select runtime model for active session. Supports multiple selector formats: model ID, #index, provider:model pattern, or "default". Persists selection to session metadata with timestamp and saves to session manager if requested.

**Architecture & Dependencies:**
- Depends on `captain_claw.config.Config` for configuration management and `captain_claw.llm.create_provider`/`set_provider` for provider instantiation
- Assumes agent instance has `self.provider`, `self.session`, `self.session_manager`, `self._provider_override`, and `self._runtime_model_details` attributes
- Uses configuration hierarchy: environment variables → .env file → config.provider_keys → session metadata
- Integrates with session persistence layer for model selection metadata storage with UTC timestamps