"""REST handlers for the web-based configuration editor.

Endpoints
---------
GET  /api/settings/schema  – field schema with types, defaults, and grouping
GET  /api/settings         – current values (secrets masked)
PUT  /api/settings         – partial update → save to ~/.captain-claw/config.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from aiohttp import web

from captain_claw.config import DEFAULT_CONFIG_PATH, LOCAL_CONFIG_FILENAME, Config, get_config, set_config
from captain_claw.logging import get_logger

log = get_logger(__name__)

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

# ── Secret mask ──────────────────────────────────────────────────────────────

SECRET_MASK = "\u2022" * 8  # ••••••••

# Fields whose values must be masked in GET and skipped if unchanged in PUT.
_SECRET_FIELDS: frozenset[str] = frozenset({
    "model.api_key",
    "provider_keys.openai",
    "provider_keys.anthropic",
    "provider_keys.gemini",
    "provider_keys.xai",
    "provider_keys.brave",
    "tools.web_search.api_key",
    "tools.send_mail.mailgun_api_key",
    "tools.send_mail.sendgrid_api_key",
    "tools.send_mail.smtp_password",
    "tools.typesense.api_key",
    "tools.pinchtab.token",
    "telegram.bot_token",
    "slack.bot_token",
    "slack.app_token",
    "discord.bot_token",
    "google_oauth.client_id",
    "google_oauth.client_secret",
    "deep_memory.api_key",
    "memory.embeddings.litellm_api_key",
    "botport.key",
    "botport.secret",
})

# Fields shown as readonly (require manual edit + restart).
_READONLY_FIELDS: frozenset[str] = frozenset({
    "web.host",
    "web.port",
})


# ── Schema definition ───────────────────────────────────────────────────────
# Hand-curated grouping with auto-detected field types from Pydantic.

def _field(key: str, label: str, **kw: Any) -> dict[str, Any]:
    """Helper to build a field descriptor."""
    d: dict[str, Any] = {"key": key, "label": label}
    d.update(kw)
    if key in _SECRET_FIELDS:
        d["type"] = "secret"
    if key in _READONLY_FIELDS:
        d["readonly"] = True
    return d


def _build_schema() -> list[dict[str, Any]]:
    """Return the full settings schema grouped for the UI."""
    return [
        # ── 0. Personality ──────────────────────────────────────
        {
            "id": "personality",
            "title": "Personality",
            "icon": "\U0001F3AD",
            "sections": [
                {
                    "id": "personality_editor",
                    "title": "Agent Personality",
                    "description": "Define the agent's default identity, background, and expertise areas.",
                    "type": "custom",
                    "custom_id": "personality",
                    "fields": [],
                },
                {
                    "id": "user_personalities_editor",
                    "title": "User Profiles",
                    "description": "User profiles for approved Telegram users. Tells the agent who it is talking to — their expertise, background, and perspective.",
                    "type": "custom",
                    "custom_id": "user_personalities",
                    "fields": [],
                },
            ],
        },
        # ── 1. Provider API Keys ───────────────────────────────────
        {
            "id": "provider_keys",
            "title": "Provider API Keys",
            "icon": "\U0001F511",
            "sections": [
                {
                    "id": "provider_keys_section",
                    "title": "API Keys",
                    "description": "Enter API keys for each provider. These are saved locally and used when environment variables are not set.",
                    "fields": [
                        _field("provider_keys.openai", "OpenAI"),
                        _field("provider_keys.openai_headers", "OpenAI Extra Headers",
                               type="tags", hint="Custom HTTP headers (Header-Name: value). When set, API key is ignored."),
                        _field("provider_keys.anthropic", "Anthropic"),
                        _field("provider_keys.anthropic_headers", "Anthropic Extra Headers",
                               type="tags", hint="Custom HTTP headers (Header-Name: value). When set, API key is ignored."),
                        _field("provider_keys.gemini", "Google Gemini"),
                        _field("provider_keys.gemini_headers", "Gemini Extra Headers",
                               type="tags", hint="Custom HTTP headers (Header-Name: value). When set, API key is ignored."),
                        _field("provider_keys.xai", "xAI (Grok)"),
                        _field("provider_keys.xai_headers", "xAI Extra Headers",
                               type="tags", hint="Custom HTTP headers (Header-Name: value). When set, API key is ignored."),
                        _field("provider_keys.brave", "Brave Search"),
                    ],
                },
            ],
        },
        # ── 2. Model & LLM ──────────────────────────────────────
        {
            "id": "model_llm",
            "title": "Model & LLM",
            "icon": "\U0001F916",
            "sections": [
                {
                    "id": "model",
                    "title": "Default Model",
                    "description": "Primary LLM provider and model for all sessions.",
                    "fields": [
                        _field("model.provider", "Provider", type="select",
                               options=["openai", "anthropic", "ollama", "gemini", "xai"]),
                        _field("model.model", "Model name", type="text"),
                        _field("model.temperature", "Temperature", type="range",
                               min=0, max=2, step=0.1),
                        _field("model.max_tokens", "Max tokens", type="number",
                               min=1, max=1000000),
                        _field("model.api_key", "API Key"),
                        _field("model.base_url", "Base URL", type="text",
                               placeholder="Leave empty for provider default"),
                        _field("model.tokens_per_minute", "Tokens per minute",
                               type="number", min=0, max=10000000,
                               hint="Provider rate limit (0 = unlimited)"),
                    ],
                },
                {
                    "id": "model_allowed",
                    "title": "Allowed Models",
                    "description": "Models available for per-session selection. Each model can override global defaults.",
                    "type": "array",
                    "layout": "cards",
                    "array_key": "model.allowed",
                    "item_fields": [
                        _field("id", "ID", type="text",
                               hint="Unique short name used to select this model"),
                        _field("provider", "Provider", type="select",
                               options=["openai", "anthropic", "ollama", "gemini", "xai"]),
                        _field("model", "Model", type="text",
                               hint="e.g. claude-sonnet-4-20250514, gpt-4.1-mini"),
                        _field("description", "Description", type="text",
                               hint="What tasks is this model best for",
                               placeholder="e.g. Fast coding tasks, deep reasoning"),
                        _field("model_type", "Type", type="select",
                               options=["llm", "image", "video", "audio", "multimodal", "ocr", "vision"],
                               hint="Model capability type"),
                        _field("tokens_per_minute", "Tokens / min", type="number",
                               min=0, max=10000000,
                               hint="Rate limit (0 = use global)"),
                        _field("max_context", "Max context", type="number",
                               min=0, max=2000000,
                               hint="Context window size (0 = use global)"),
                        _field("max_output_tokens", "Max output tokens", type="number",
                               min=0, max=1000000,
                               hint="Output limit (0 = use global)"),
                        _field("reasoning_level", "Reasoning", type="select",
                               options=["", "low", "medium", "high"],
                               hint="Extended thinking level (if supported)"),
                        _field("temperature", "Temperature", type="number",
                               min=0, max=2, step=0.1,
                               hint="Leave empty to use global default"),
                        _field("base_url", "Base URL", type="text",
                               placeholder="Leave empty for provider default"),
                    ],
                },
                {
                    "id": "context",
                    "title": "Context Window",
                    "description": "Controls how much conversation history the model sees.",
                    "fields": [
                        _field("context.max_tokens", "Max context tokens", type="number",
                               min=1000, max=2000000),
                        _field("context.compaction_threshold", "Compaction threshold",
                               type="range", min=0.1, max=1.0, step=0.05,
                               hint="Start compacting when context reaches this fraction."),
                        _field("context.compaction_ratio", "Compaction ratio",
                               type="range", min=0.1, max=1.0, step=0.05,
                               hint="Target size after compaction (fraction of max)."),
                        _field("context.micro_instructions", "Micro instructions",
                               type="toggle",
                               hint="Use compact instruction prompts (~66 % fewer tokens). Faster and cheaper but less detailed guidance for the LLM. Takes effect on new sessions."),
                    ],
                },
                {
                    "id": "chunked_processing",
                    "title": "Chunked Processing",
                    "description": "Process large content with small-context models by splitting it into sequential chunks. "
                                   "Activates automatically when content exceeds the model's available context window.",
                    "fields": [
                        _field("context.chunked_processing.enabled", "Enabled",
                               type="toggle",
                               hint="Master switch. Enable chunked processing regardless of model size."),
                        _field("context.chunked_processing.auto_threshold", "Auto-threshold (tokens)",
                               type="number", min=0, max=2000000,
                               hint="Auto-enable when context.max_tokens is at or below this value. "
                                    "E.g. 64000 to activate for any model with context <= 64k. 0 = off."),
                        _field("context.chunked_processing.output_reserve_tokens", "Output reserve (tokens)",
                               type="number", min=500, max=100000,
                               hint="Tokens reserved for LLM output in each chunk call. "
                                    "Larger values leave less room for content per chunk."),
                        _field("context.chunked_processing.chunk_overlap_tokens", "Chunk overlap (tokens)",
                               type="number", min=0, max=2000,
                               hint="Overlap between consecutive chunks to preserve continuity across boundaries."),
                        _field("context.chunked_processing.max_chunks", "Max chunks",
                               type="number", min=1, max=100,
                               hint="Hard cap on number of chunks per item. "
                                    "If content requires more, the last chunk absorbs the remainder."),
                        _field("context.chunked_processing.combine_strategy", "Combine strategy",
                               type="select",
                               options=["summarize", "concatenate"],
                               hint="How partial results are merged. 'summarize' uses an LLM synthesis call; "
                                    "'concatenate' joins with separators."),
                    ],
                },
            ],
        },
        # ── 2. Memory ────────────────────────────────────────────
        {
            "id": "memory",
            "title": "Memory",
            "icon": "\U0001F9E0",
            "sections": [
                {
                    "id": "memory_general",
                    "title": "General",
                    "fields": [
                        _field("memory.enabled", "Enable memory", type="toggle"),
                        _field("memory.path", "Database path", type="text"),
                        _field("memory.index_workspace", "Index workspace files", type="toggle"),
                        _field("memory.index_sessions", "Index sessions", type="toggle"),
                        _field("memory.cross_session_retrieval", "Cross-session retrieval", type="toggle"),
                        _field("memory.auto_sync_on_search", "Auto-sync on search", type="toggle"),
                        _field("memory.max_workspace_files", "Max workspace files", type="number"),
                        _field("memory.max_file_bytes", "Max file size (bytes)", type="number"),
                        _field("memory.chunk_chars", "Chunk size (chars)", type="number"),
                        _field("memory.chunk_overlap_chars", "Chunk overlap (chars)", type="number"),
                        _field("memory.cache_ttl_seconds", "Cache TTL (seconds)", type="number"),
                        _field("memory.stale_after_seconds", "Stale after (seconds)", type="number"),
                        _field("memory.include_extensions", "Included file extensions", type="tags"),
                        _field("memory.exclude_dirs", "Excluded directories", type="tags"),
                    ],
                },
                {
                    "id": "memory_embeddings",
                    "title": "Embeddings",
                    "description": "Vector embedding provider for semantic search.",
                    "fields": [
                        _field("memory.embeddings.provider", "Provider", type="select",
                               options=["auto", "litellm", "ollama", "none"]),
                        _field("memory.embeddings.litellm_model", "LiteLLM model", type="text"),
                        _field("memory.embeddings.litellm_api_key", "LiteLLM API key"),
                        _field("memory.embeddings.litellm_base_url", "LiteLLM base URL", type="text"),
                        _field("memory.embeddings.ollama_model", "Ollama model", type="text"),
                        _field("memory.embeddings.ollama_base_url", "Ollama base URL", type="text"),
                        _field("memory.embeddings.request_timeout_seconds", "Request timeout (s)", type="number"),
                        _field("memory.embeddings.fallback_to_local_hash", "Fallback to local hash", type="toggle"),
                    ],
                },
                {
                    "id": "memory_search",
                    "title": "Search",
                    "description": "Hybrid retrieval scoring parameters.",
                    "fields": [
                        _field("memory.search.max_results", "Max results", type="number"),
                        _field("memory.search.candidate_limit", "Candidate limit", type="number"),
                        _field("memory.search.min_score", "Min score", type="range",
                               min=0, max=1, step=0.05),
                        _field("memory.search.vector_weight", "Vector weight", type="range",
                               min=0, max=1, step=0.05),
                        _field("memory.search.text_weight", "Text weight", type="range",
                               min=0, max=1, step=0.05),
                        _field("memory.search.temporal_decay_enabled", "Temporal decay", type="toggle"),
                        _field("memory.search.temporal_half_life_days", "Half-life (days)", type="number",
                               min=1, max=365, step=0.5),
                    ],
                },
            ],
        },
        # ── 3. Tools ─────────────────────────────────────────────
        {
            "id": "tools",
            "title": "Tools",
            "icon": "\U0001F527",
            "sections": [
                {
                    "id": "tools_enabled",
                    "title": "Enabled Tools",
                    "description": "Select which tools the agent can use.",
                    "fields": [
                        _field("tools.enabled", "Enabled tools", type="tags"),
                        _field("tools.require_confirmation", "Require confirmation", type="tags"),
                        _field("tools.plugin_dirs", "Plugin directories", type="tags"),
                        _field("tools.duplicate_call_max", "Duplicate call limit", type="number",
                               min=1, max=10),
                    ],
                },
                {
                    "id": "tools_read",
                    "title": "Read Tool",
                    "fields": [
                        _field("tools.read.max_file_bytes", "Max file size (bytes)", type="number"),
                    ],
                },
                {
                    "id": "tools_web_fetch",
                    "title": "Web Fetch",
                    "fields": [
                        _field("tools.web_fetch.max_chars", "Max chars per fetch", type="number"),
                    ],
                },
                {
                    "id": "tools_web_search",
                    "title": "Web Search",
                    "fields": [
                        _field("tools.web_search.provider", "Provider", type="select",
                               options=["brave"]),
                        _field("tools.web_search.api_key", "API Key"),
                        _field("tools.web_search.base_url", "Base URL", type="text"),
                        _field("tools.web_search.max_results", "Max results", type="number",
                               min=1, max=20),
                        _field("tools.web_search.timeout", "Timeout (s)", type="number"),
                        _field("tools.web_search.safesearch", "SafeSearch", type="select",
                               options=["off", "moderate", "strict"]),
                    ],
                },
                {
                    "id": "tools_tts",
                    "title": "Text-to-Speech",
                    "fields": [
                        _field("tools.pocket_tts.max_chars", "Max chars", type="number"),
                        _field("tools.pocket_tts.default_voice", "Default voice", type="text"),
                        _field("tools.pocket_tts.sample_rate", "Sample rate", type="number"),
                        _field("tools.pocket_tts.mp3_bitrate_kbps", "MP3 bitrate (kbps)", type="number"),
                        _field("tools.pocket_tts.timeout_seconds", "Timeout (s)", type="number"),
                    ],
                },
                {
                    "id": "tools_typesense",
                    "title": "Typesense",
                    "fields": [
                        _field("tools.typesense.host", "Host", type="text"),
                        _field("tools.typesense.port", "Port", type="number"),
                        _field("tools.typesense.protocol", "Protocol", type="select",
                               options=["http", "https"]),
                        _field("tools.typesense.api_key", "API Key"),
                        _field("tools.typesense.default_collection", "Default collection", type="text"),
                        _field("tools.typesense.timeout", "Timeout (s)", type="number"),
                        _field("tools.typesense.connection_timeout", "Connection timeout (s)", type="number"),
                    ],
                },
                {
                    "id": "tools_browser",
                    "title": "Browser",
                    "fields": [
                        _field("tools.browser.screenshot_max_pixels", "Screenshot max pixels",
                               type="number", min=200, max=3840,
                               hint="Longest edge cap before sending to vision LLM. Lower = fewer tokens."),
                        _field("tools.browser.screenshot_jpeg_quality", "Screenshot JPEG quality",
                               type="range", min=10, max=100, step=5,
                               hint="JPEG compression quality for vision LLM. Lower = smaller images, fewer tokens."),
                        _field("tools.browser.headless", "Headless mode", type="toggle"),
                        _field("tools.browser.viewport_width", "Viewport width", type="number",
                               min=320, max=3840),
                        _field("tools.browser.viewport_height", "Viewport height", type="number",
                               min=240, max=2160),
                        _field("tools.browser.timeout_seconds", "Timeout (s)", type="number",
                               min=5, max=300),
                        _field("tools.browser.default_wait_seconds", "Default wait (s)", type="number",
                               min=0, max=30),
                    ],
                },
                {
                    "id": "tools_pinchtab",
                    "title": "PinchTab",
                    "description": "Token-efficient browser automation via PinchTab HTTP API.",
                    "fields": [
                        _field("tools.pinchtab.enabled", "Enabled", type="toggle",
                               hint="Enable the PinchTab browser tool (runs parallel to the Playwright browser)."),
                        _field("tools.pinchtab.host", "Host", type="text",
                               hint="PinchTab server bind address."),
                        _field("tools.pinchtab.port", "Port", type="number",
                               min=1024, max=65535),
                        _field("tools.pinchtab.token", "Bearer Token",
                               hint="Authentication token for the PinchTab API."),
                        _field("tools.pinchtab.auto_start", "Auto-start server", type="toggle",
                               hint="Automatically start PinchTab server if not running."),
                        _field("tools.pinchtab.binary_path", "Binary path", type="text",
                               hint="Custom path to pinchtab binary. Leave empty to find in PATH."),
                        _field("tools.pinchtab.headless", "Headless mode", type="toggle"),
                        _field("tools.pinchtab.stealth_level", "Stealth level", type="select",
                               options=["light", "medium", "full"],
                               hint="Anti-bot evasion level."),
                        _field("tools.pinchtab.default_profile", "Default profile", type="text",
                               hint="Persistent profile name. Cookies/localStorage survive restarts."),
                        _field("tools.pinchtab.timeout_seconds", "Timeout (s)", type="number",
                               min=5, max=300),
                        _field("tools.pinchtab.max_tabs", "Max tabs", type="number",
                               min=1, max=50),
                        _field("tools.pinchtab.block_ads", "Block ads", type="toggle"),
                        _field("tools.pinchtab.block_images", "Block images", type="toggle",
                               hint="Block image loading for faster browsing."),
                        _field("tools.pinchtab.allow_evaluate", "Allow JavaScript eval", type="toggle",
                               hint="Enable the eval action for running JavaScript in the browser. Disabled by default for security."),
                    ],
                },
            ],
        },
        # ── 4. Email ─────────────────────────────────────────────
        {
            "id": "email",
            "title": "Email",
            "icon": "\U0001F4E7",
            "sections": [
                {
                    "id": "email_general",
                    "title": "Email Settings",
                    "description": "Configure outbound email for the send_mail tool.",
                    "wizard": "email",
                    "fields": [
                        _field("tools.send_mail.provider", "Provider", type="select",
                               options=["smtp", "mailgun", "sendgrid"]),
                        _field("tools.send_mail.from_address", "From address", type="text"),
                        _field("tools.send_mail.from_name", "From name", type="text"),
                        _field("tools.send_mail.timeout", "Timeout (s)", type="number"),
                        _field("tools.send_mail.max_attachment_bytes", "Max attachment (bytes)", type="number"),
                    ],
                },
                {
                    "id": "email_mailgun",
                    "title": "Mailgun",
                    "description": "Mailgun provider credentials.",
                    "show_when": {"key": "tools.send_mail.provider", "value": "mailgun"},
                    "fields": [
                        _field("tools.send_mail.mailgun_api_key", "Mailgun API Key"),
                        _field("tools.send_mail.mailgun_domain", "Mailgun domain", type="text"),
                        _field("tools.send_mail.mailgun_base_url", "Mailgun base URL", type="text"),
                    ],
                },
                {
                    "id": "email_sendgrid",
                    "title": "SendGrid",
                    "description": "SendGrid provider credentials.",
                    "show_when": {"key": "tools.send_mail.provider", "value": "sendgrid"},
                    "fields": [
                        _field("tools.send_mail.sendgrid_api_key", "SendGrid API Key"),
                        _field("tools.send_mail.sendgrid_base_url", "SendGrid base URL", type="text"),
                    ],
                },
                {
                    "id": "email_smtp",
                    "title": "SMTP",
                    "description": "SMTP server credentials.",
                    "show_when": {"key": "tools.send_mail.provider", "value": "smtp"},
                    "fields": [
                        _field("tools.send_mail.smtp_host", "SMTP host", type="text"),
                        _field("tools.send_mail.smtp_port", "SMTP port", type="number"),
                        _field("tools.send_mail.smtp_username", "SMTP username", type="text"),
                        _field("tools.send_mail.smtp_password", "SMTP password"),
                        _field("tools.send_mail.smtp_use_tls", "Use TLS", type="toggle"),
                    ],
                },
            ],
        },
        # ── 5. Safety ────────────────────────────────────────────
        {
            "id": "safety",
            "title": "Safety",
            "icon": "\U0001F6E1",
            "sections": [
                {
                    "id": "guards",
                    "title": "Guards",
                    "description": "Input/output filtering and script safety.",
                    "fields": [
                        _field("guards.input.enabled", "Input guard enabled", type="toggle"),
                        _field("guards.input.level", "Input guard level", type="select",
                               options=["stop_suspicious", "ask_for_approval"]),
                        _field("guards.output.enabled", "Output guard enabled", type="toggle"),
                        _field("guards.output.level", "Output guard level", type="select",
                               options=["stop_suspicious", "ask_for_approval"]),
                        _field("guards.script_tool.enabled", "Script tool guard enabled", type="toggle"),
                        _field("guards.script_tool.level", "Script tool guard level", type="select",
                               options=["stop_suspicious", "ask_for_approval"]),
                    ],
                },
                {
                    "id": "shell_safety",
                    "title": "Shell Policy",
                    "description": "Execution policy and pattern matching for shell commands.",
                    "fields": [
                        _field("tools.shell.timeout", "Timeout (s)", type="number"),
                        _field("tools.shell.default_policy", "Default policy", type="select",
                               options=["ask", "allow", "deny"]),
                        _field("tools.shell.blocked", "Blocked commands", type="tags"),
                        _field("tools.shell.allowed_commands", "Allowed commands", type="tags"),
                        _field("tools.shell.allow_patterns", "Allow patterns", type="tags"),
                        _field("tools.shell.deny_patterns", "Deny patterns", type="tags"),
                    ],
                },
            ],
        },
        # ── 6. Integrations ──────────────────────────────────────
        {
            "id": "integrations",
            "title": "Integrations",
            "icon": "\U0001F4F1",
            "sections": [
                {
                    "id": "telegram",
                    "title": "Telegram",
                    "description": "Telegram bot interface.",
                    "wizard": "messaging",
                    "fields": [
                        _field("telegram.enabled", "Enable Telegram", type="toggle"),
                        _field("telegram.bot_token", "Bot Token"),
                        _field("telegram.api_base_url", "API base URL", type="text"),
                        _field("telegram.poll_timeout_seconds", "Poll timeout (s)", type="number"),
                        _field("telegram.pairing_ttl_minutes", "Pairing TTL (min)", type="number"),
                    ],
                },
                {
                    "id": "slack",
                    "title": "Slack",
                    "description": "Slack bot interface.",
                    "wizard": "messaging",
                    "fields": [
                        _field("slack.enabled", "Enable Slack", type="toggle"),
                        _field("slack.bot_token", "Bot Token"),
                        _field("slack.app_token", "App Token"),
                        _field("slack.api_base_url", "API base URL", type="text"),
                        _field("slack.poll_timeout_seconds", "Poll timeout (s)", type="number"),
                        _field("slack.pairing_ttl_minutes", "Pairing TTL (min)", type="number"),
                    ],
                },
                {
                    "id": "discord",
                    "title": "Discord",
                    "description": "Discord bot interface.",
                    "wizard": "messaging",
                    "fields": [
                        _field("discord.enabled", "Enable Discord", type="toggle"),
                        _field("discord.bot_token", "Bot Token"),
                        _field("discord.application_id", "Application ID", type="number"),
                        _field("discord.api_base_url", "API base URL", type="text"),
                        _field("discord.poll_timeout_seconds", "Poll timeout (s)", type="number"),
                        _field("discord.pairing_ttl_minutes", "Pairing TTL (min)", type="number"),
                        _field("discord.require_mention_in_guild", "Require @mention in guilds", type="toggle"),
                    ],
                },
                {
                    "id": "google_oauth",
                    "title": "Google OAuth",
                    "description": "Google/Gemini authentication via Vertex AI.",
                    "fields": [
                        _field("google_oauth.enabled", "Enable Google OAuth", type="toggle"),
                        _field("google_oauth.client_id", "Client ID"),
                        _field("google_oauth.client_secret", "Client Secret"),
                        _field("google_oauth.project_id", "GCP Project ID", type="text"),
                        _field("google_oauth.location", "Vertex AI region", type="text"),
                        _field("google_oauth.scopes", "OAuth Scopes", type="tags",
                               hint="Google API scopes requested during login. "
                                    "Required scopes are always included automatically."),
                    ],
                },
            ],
        },
        # ── 7. General ───────────────────────────────────────────
        {
            "id": "general",
            "title": "General",
            "icon": "\u2699\uFE0F",
            "sections": [
                {
                    "id": "ui",
                    "title": "UI",
                    "fields": [
                        _field("ui.theme", "Theme", type="select", options=["dark", "light"]),
                        _field("ui.show_tokens", "Show token counts", type="toggle"),
                        _field("ui.streaming", "Streaming responses", type="toggle"),
                        _field("ui.colors", "Colors in terminal", type="toggle"),
                        _field("ui.monitor_trace_llm", "Monitor: trace LLM calls", type="toggle"),
                        _field("ui.monitor_trace_pipeline", "Monitor: trace pipeline", type="toggle"),
                        _field("ui.monitor_full_output", "Monitor: full output", type="toggle"),
                    ],
                },
                {
                    "id": "logging",
                    "title": "Logging",
                    "fields": [
                        _field("logging.level", "Log level", type="select",
                               options=["DEBUG", "INFO", "WARNING", "ERROR"]),
                        _field("logging.format", "Log format", type="select",
                               options=["console", "json"]),
                        _field("logging.llm_session_logging", "LLM session logging", type="toggle"),
                    ],
                },
                {
                    "id": "session",
                    "title": "Session",
                    "fields": [
                        _field("session.storage", "Storage backend", type="text"),
                        _field("session.path", "Database path", type="text"),
                        _field("session.auto_save", "Auto-save", type="toggle"),
                    ],
                },
                {
                    "id": "workspace",
                    "title": "Workspace",
                    "fields": [
                        _field("workspace.path", "Workspace path", type="text"),
                    ],
                },
                {
                    "id": "execution_queue",
                    "title": "Execution Queue",
                    "description": "Follow-up message queue while sessions are busy.",
                    "fields": [
                        _field("execution_queue.mode", "Mode", type="select",
                               options=["collect", "drop"]),
                        _field("execution_queue.debounce_ms", "Debounce (ms)", type="number"),
                        _field("execution_queue.cap", "Queue cap", type="number"),
                        _field("execution_queue.drop", "Drop strategy", type="select",
                               options=["summarize", "oldest", "newest"]),
                    ],
                },
            ],
        },
        # ── 8. Features ──────────────────────────────────────────
        {
            "id": "features",
            "title": "Features",
            "icon": "\U0001F4CB",
            "sections": [
                {
                    "id": "todo",
                    "title": "To-Do Memory",
                    "fields": [
                        _field("todo.enabled", "Enabled", type="toggle"),
                        _field("todo.auto_capture", "Auto-capture", type="toggle"),
                        _field("todo.inject_on_session_load", "Inject on session load", type="toggle"),
                        _field("todo.max_items_in_prompt", "Max items in prompt", type="number"),
                        _field("todo.archive_after_days", "Archive after (days)", type="number"),
                    ],
                },
                {
                    "id": "addressbook",
                    "title": "Address Book",
                    "fields": [
                        _field("addressbook.enabled", "Enabled", type="toggle"),
                        _field("addressbook.auto_capture", "Auto-capture", type="toggle"),
                        _field("addressbook.inject_on_mention", "Inject on mention", type="toggle"),
                        _field("addressbook.max_items_in_prompt", "Max items in prompt", type="number"),
                    ],
                },
                {
                    "id": "scripts_memory",
                    "title": "Scripts Memory",
                    "fields": [
                        _field("scripts_memory.enabled", "Enabled", type="toggle"),
                        _field("scripts_memory.auto_capture", "Auto-capture", type="toggle"),
                        _field("scripts_memory.inject_on_mention", "Inject on mention", type="toggle"),
                        _field("scripts_memory.max_items_in_prompt", "Max items in prompt", type="number"),
                    ],
                },
                {
                    "id": "apis_memory",
                    "title": "APIs Memory",
                    "fields": [
                        _field("apis_memory.enabled", "Enabled", type="toggle"),
                        _field("apis_memory.auto_capture", "Auto-capture", type="toggle"),
                        _field("apis_memory.inject_on_mention", "Inject on mention", type="toggle"),
                        _field("apis_memory.max_items_in_prompt", "Max items in prompt", type="number"),
                    ],
                },
            ],
        },
        # ── 9. Scale & Orchestrator ──────────────────────────────
        {
            "id": "scale_orchestrator",
            "title": "Scale & Orchestrator",
            "icon": "\U0001F680",
            "sections": [
                {
                    "id": "scale",
                    "title": "Scale Loop",
                    "description": "Thresholds for list-processing and research extraction.",
                    "fields": [
                        _field("scale.scale_advisory_min_members", "Scale advisory min members", type="number"),
                        _field("scale.lightweight_progress_min_members", "Lightweight progress min members", type="number"),
                        _field("scale.task_rephrase_enabled", "Task rephrasing", type="toggle"),
                        _field("scale.task_rephrase_min_chars", "Rephrase min chars", type="number"),
                        _field("scale.research_search_results", "Research: search results per item", type="number"),
                        _field("scale.research_max_chars_per_fetch", "Research: max chars per fetch", type="number"),
                        _field("scale.research_query_keywords", "Research: max search keywords", type="number"),
                    ],
                },
                {
                    "id": "orchestrator",
                    "title": "Orchestrator",
                    "description": "Parallel session orchestration settings.",
                    "fields": [
                        _field("orchestrator.max_parallel", "Max parallel sessions", type="number"),
                        _field("orchestrator.max_agents", "Max agents", type="number"),
                        _field("orchestrator.idle_evict_seconds", "Idle eviction (s)", type="number"),
                        _field("orchestrator.worker_timeout_seconds", "Worker timeout (s)", type="number"),
                        _field("orchestrator.timeout_grace_seconds", "Timeout grace period (s)", type="number"),
                        _field("orchestrator.worker_max_retries", "Worker max retries", type="number"),
                    ],
                },
            ],
        },
        # ── 10. Web Server ───────────────────────────────────────
        {
            "id": "web",
            "title": "Web Server",
            "icon": "\U0001F310",
            "sections": [
                {
                    "id": "web_server",
                    "title": "Web Server",
                    "description": "Web UI and API settings.  Host and port require a restart.",
                    "fields": [
                        _field("web.enabled", "Enable web server", type="toggle"),
                        _field("web.host", "Host", type="text"),
                        _field("web.port", "Port", type="number"),
                        _field("web.api_enabled", "Enable OpenAI-compatible API", type="toggle"),
                        _field("web.api_pool_max_agents", "API pool max agents", type="number"),
                        _field("web.api_pool_idle_seconds", "API pool idle (s)", type="number"),
                    ],
                },
            ],
        },
        # ── 11. Deep Memory ──────────────────────────────────────
        {
            "id": "deep_memory",
            "title": "Deep Memory",
            "icon": "\U0001F4BE",
            "sections": [
                {
                    "id": "deep_memory",
                    "title": "Typesense Deep Memory",
                    "description": "Long-term searchable content via Typesense.",
                    "fields": [
                        _field("deep_memory.enabled", "Enabled", type="toggle"),
                        _field("deep_memory.host", "Host", type="text"),
                        _field("deep_memory.port", "Port", type="number"),
                        _field("deep_memory.protocol", "Protocol", type="select",
                               options=["http", "https"]),
                        _field("deep_memory.api_key", "API Key"),
                        _field("deep_memory.collection_name", "Collection name", type="text"),
                        _field("deep_memory.embedding_dims", "Embedding dimensions", type="number"),
                        _field("deep_memory.auto_embed", "Auto-embed", type="toggle"),
                    ],
                },
            ],
        },
        # ── 12. Skills ───────────────────────────────────────────
        {
            "id": "skills",
            "title": "Skills",
            "icon": "\U0001F3AF",
            "sections": [
                {
                    "id": "skills_general",
                    "title": "Skills",
                    "description": "Skill system configuration.",
                    "fields": [
                        _field("skills.managed_dir", "Managed directory", type="text"),
                        _field("skills.max_skills_in_prompt", "Max skills in prompt", type="number"),
                        _field("skills.max_skills_prompt_chars", "Max prompt chars", type="number"),
                        _field("skills.max_skill_file_bytes", "Max skill file (bytes)", type="number"),
                        _field("skills.search_source_url", "Search source URL", type="text"),
                        _field("skills.search_limit", "Search limit", type="number"),
                    ],
                },
                {
                    "id": "skills_install",
                    "title": "Install Preferences",
                    "fields": [
                        _field("skills.install.prefer_brew", "Prefer Homebrew", type="toggle"),
                        _field("skills.install.node_manager", "Node package manager", type="select",
                               options=["npm", "pnpm", "yarn", "bun"]),
                    ],
                },
            ],
        },
        # ── 13. BotPort ─────────────────────────────────────────
        {
            "id": "botport",
            "title": "BotPort",
            "icon": "\U0001F500",
            "sections": [
                {
                    "id": "botport_connection",
                    "title": "Connection",
                    "description": "Connect to a BotPort hub for agent-to-agent task routing.",
                    "fields": [
                        _field("botport.enabled", "Enable BotPort", type="toggle"),
                        _field("botport.url", "BotPort URL", type="text",
                               placeholder="ws://localhost:9800/ws"),
                        _field("botport.instance_name", "Instance name", type="text",
                               hint="Human-readable name for this instance on the BotPort network."),
                    ],
                },
                {
                    "id": "botport_auth",
                    "title": "Authentication",
                    "description": "Credentials for BotPort hub (leave empty if auth is disabled on the hub).",
                    "fields": [
                        _field("botport.key", "Key"),
                        _field("botport.secret", "Secret"),
                    ],
                },
                {
                    "id": "botport_advanced",
                    "title": "Advanced",
                    "description": "Capacity and timing settings.",
                    "fields": [
                        _field("botport.max_concurrent", "Max concurrent concerns", type="number",
                               min=1, max=50,
                               hint="How many dispatched concerns this instance handles simultaneously."),
                        _field("botport.reconnect_delay_seconds", "Reconnect delay (s)", type="number",
                               min=1, max=300),
                        _field("botport.heartbeat_interval_seconds", "Heartbeat interval (s)", type="number",
                               min=5, max=300),
                        _field("botport.advertise_personas", "Advertise personas", type="toggle",
                               hint="Share this instance's personas with BotPort for routing."),
                        _field("botport.advertise_tools", "Advertise tools", type="toggle",
                               hint="Share available tools list with BotPort."),
                        _field("botport.advertise_models", "Advertise models", type="toggle",
                               hint="Share available models list with BotPort."),
                    ],
                },
            ],
        },
    ]


# ── Value helpers ────────────────────────────────────────────────────────────

def _get_nested(obj: Any, dotted_key: str) -> Any:
    """Resolve ``'a.b.c'`` to ``obj.a.b.c`` (attribute or dict access)."""
    parts = dotted_key.split(".")
    for p in parts:
        if isinstance(obj, dict):
            obj = obj.get(p)
        else:
            obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``data['a']['b']['c'] = value``, creating intermediary dicts."""
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        data = data.setdefault(p, {})
    data[parts[-1]] = value


def _flatten_values(cfg: Config, schema: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten the config into ``{ "model.provider": "openai", ... }``."""
    values: dict[str, Any] = {}
    for group in schema:
        for section in group.get("sections", []):
            if section.get("type") == "array":
                arr_key = section["array_key"]
                arr_val = _get_nested(cfg, arr_key)
                if arr_val is not None:
                    # Convert list of Pydantic models / dicts to plain dicts
                    items = []
                    for item in arr_val:
                        if hasattr(item, "model_dump"):
                            items.append(item.model_dump(mode="json"))
                        elif isinstance(item, dict):
                            items.append(item)
                    values[arr_key] = items
            else:
                for field in section.get("fields", []):
                    key = field["key"]
                    val = _get_nested(cfg, key)
                    if key in _SECRET_FIELDS and val:
                        val = SECRET_MASK
                    values[key] = val
    return values


# ── Endpoints ────────────────────────────────────────────────────────────────

async def get_settings_schema(server: WebServer, request: web.Request) -> web.Response:
    """Return the full settings schema for the UI to render."""
    schema = _build_schema()
    return web.json_response({"groups": schema})


async def get_settings_values(server: WebServer, request: web.Request) -> web.Response:
    """Return current config values (secrets masked)."""
    cfg = get_config()
    schema = _build_schema()
    values = _flatten_values(cfg, schema)
    return web.json_response({"values": values, "source": "home"})


async def put_settings(server: WebServer, request: web.Request) -> web.Response:
    """Apply partial config changes.

    Reads the existing ``~/.captain-claw/config.yaml``, merges the
    submitted changes, validates via Pydantic, writes back, and reloads
    the in-memory config singleton.
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    changes: dict[str, Any] = body.get("changes", {})
    if not changes:
        return web.json_response({"error": "No changes provided"}, status=400)

    # ── Read existing YAML (or start empty) ───────────────
    config_path = DEFAULT_CONFIG_PATH
    if config_path.exists():
        raw = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
    else:
        data = {}

    # ── Merge changes ─────────────────────────────────────
    for dotted_key, value in changes.items():
        # Skip readonly fields
        if dotted_key in _READONLY_FIELDS:
            continue
        # Skip secrets that weren't actually changed
        if dotted_key in _SECRET_FIELDS and value == SECRET_MASK:
            continue
        _set_nested(data, dotted_key, value)

    # Always-on tools — re-inject if the user removed them.
    tools_enabled = _get_nested(data, "tools.enabled")
    if isinstance(tools_enabled, list):
        for always_on in ("personality", "botport"):
            if always_on not in tools_enabled:
                tools_enabled.append(always_on)

    # ── Validate against merged result (local base + home overlay) ──
    local_path = Path.cwd() / LOCAL_CONFIG_FILENAME
    local_data = Config._read_yaml_data(local_path)
    if local_data:
        merged_data = Config._deep_merge(local_data, data)
    else:
        merged_data = data
    try:
        Config(**merged_data)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=422)

    # ── Save to disk (only the home file) ─────────────────
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    # ── Reload in-memory config (merges local + home) ─────
    set_config(Config.load())

    # ── Hot-reload tools if the enabled list changed ──────
    if "tools.enabled" in changes and server.agent is not None:
        try:
            server.agent.reload_tools()
        except Exception as exc:
            log.warning("Tool hot-reload failed: %s", exc)

    return web.json_response({"ok": True})
