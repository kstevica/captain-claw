# Summary: onboarding.py

# onboarding.py Summary

Manages interactive first-time setup for Captain Claw, guiding users through LLM provider selection, API key configuration, and safety settings via a Rich-based terminal UI. Handles state persistence, provider validation, and configuration file generation with sensible defaults for multi-model support.

## Purpose

Solves the cold-start problem for new Captain Claw installations by providing a guided onboarding experience that:
- Detects whether onboarding should run (checks completion state, existing configs)
- Validates LLM provider connectivity before saving configuration
- Seeds default allowed models (OpenAI, Anthropic, Gemini variants + image/OCR/vision)
- Persists user choices to YAML config with optional API key storage
- Supports both interactive terminal and programmatic flows

## Most Important Functions/Classes/Procedures

1. **`run_onboarding_wizard(config_path, state_path, require_interactive)`**
   - Main entry point orchestrating the 9-step interactive wizard (welcome → config location → provider selection → model name → API key → validation → additional models → safety guards → summary → save). Uses Rich console for formatted TUI output. Returns saved config file path or None if skipped.

2. **`validate_provider_connection(provider, model, api_key, base_url)`**
   - Async validation testing LLM provider connectivity via lightweight API calls (5-token completion for cloud providers, `/api/tags` GET for Ollama). Returns tuple of (success: bool, error_message: str | None). Handles provider normalization and special Ollama logic.

3. **`save_onboarding_config(values, config_path, state_path)`**
   - Builds Config object from onboarding dictionary, applies user selections (provider, model, API keys, guards, allowed models), seeds defaults if needed, persists to YAML, marks onboarding completed. Handles provider_keys dict and OpenAI OAuth headers. Returns target config path.

4. **`should_run_onboarding(force, state_path, config_path, cwd, global_config_path)`**
   - Decision logic determining if onboarding should execute: checks force flag, completion marker, explicit config path existence, and local/global config presence. Returns boolean.

5. **`_select_provider(console, current_provider)` & `_select_config_path(console, config_path)`**
   - TUI helpers rendering Rich tables for user selection. Provider selector shows all 5 options (OpenAI, Anthropic, Gemini, xAI, Ollama) with suggested models and env var names. Config path selector chooses between global (~/.captain-claw) and local (cwd) storage.

## Architecture & Dependencies

**Key Dependencies:**
- `rich` — Console UI, tables, prompts, spinners, panels
- `httpx` — Async HTTP client for Ollama validation
- `captain_claw.config` — Config class, path constants, model definitions
- `captain_claw.llm` — Provider creation and Message interface for validation

**State Management:**
- Onboarding state stored in `~/.captain-claw/onboarding_state.json` (completion timestamp)
- Config saved to `~/.captain-claw/config.yaml` (global) or `./captain-claw.yaml` (local)

**Provider Metadata:**
- Hardcoded provider order, labels, default models, env var names, and aliases for normalization
- Default allowed models list (_DEFAULT_ALLOWED_MODELS) seeded into fresh configs with 12 pre-configured entries spanning OpenAI (gpt-5 variants), Anthropic (Sonnet/Opus/Haiku), Gemini (Flash/Pro/Lite), plus image generation and OCR models

**Validation Strategy:**
- Cloud providers: send minimal completion request (5 tokens, "Say OK" prompt)
- Ollama: HTTP GET to `/api/tags` endpoint (no auth required)
- Sync wrapper handles both running and non-running event loops via ThreadPoolExecutor

**Configuration Seeding:**
- Guards (input/output/script_tool) default to disabled; onboarding can enable with "ask_for_approval" level
- Multi-model support: users can add up to 3 additional models during wizard; defaults always included
- Workspace path, Telegram settings, and provider-specific keys all configurable through values dict