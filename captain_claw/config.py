"""Configuration management for Captain Claw."""

import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Paths
DEFAULT_CONFIG_PATH = Path("~/.captain-claw/config.yaml").expanduser()
DEFAULT_DB_PATH = Path("~/.captain-claw/sessions.db").expanduser()
LOCAL_CONFIG_FILENAME = "config.yaml"


class ModelConfig(BaseModel):
    """Model configuration."""

    class AllowedModelConfig(BaseModel):
        """Allowed model entry for live per-session selection."""

        id: str
        provider: str
        model: str
        base_url: str = ""
        temperature: float | None = None
        max_tokens: int | None = None
        tokens_per_minute: int = 0  # 0 = use global default
        max_context: int = 0  # 0 = use global context.max_tokens
        max_output_tokens: int = 0  # 0 = use model.max_tokens
        reasoning_level: str = ""  # "", "low", "medium", "high"
        description: str = ""  # what tasks this model is best for
        model_type: str = "llm"  # llm, image, video, audio, multimodal

    provider: str = "ollama"
    model: str = "minimax-m2.5:cloud"
    temperature: float = 0.7
    max_tokens: int = 32000
    api_key: str = ""
    base_url: str = ""
    tokens_per_minute: int = 0  # 0 = unlimited (no rate limiting)
    allowed: list[AllowedModelConfig] = Field(default_factory=list)


class ChunkedProcessingConfig(BaseModel):
    """Chunked processing pipeline for small-context models.

    When enabled, content that exceeds the model's context window is
    automatically split into sequential chunks, each processed with
    the full instruction set, and the partial results are combined
    into a single output.  This allows models with 20k–32k context
    windows to handle documents that would otherwise be truncated.
    """

    enabled: bool = False  # master switch; "auto" behavior via auto_threshold
    auto_threshold: int = 0  # auto-enable when context.max_tokens <= this (0 = manual only)
    output_reserve_tokens: int = 4000  # tokens reserved for model output per chunk call
    chunk_overlap_tokens: int = 200  # continuity overlap between sequential chunks
    max_chunks: int = 12  # safety cap to prevent runaway splitting
    combine_strategy: str = "summarize"  # summarize | concatenate


class ContextConfig(BaseModel):
    """Context window configuration."""

    max_tokens: int = 160000
    compaction_threshold: float = 0.8
    compaction_ratio: float = 0.4
    micro_instructions: bool = False
    chunked_processing: ChunkedProcessingConfig = Field(
        default_factory=ChunkedProcessingConfig,
    )


class MemoryEmbeddingsConfig(BaseModel):
    """Embedding provider settings for semantic memory."""

    provider: str = "auto"  # auto | litellm | ollama | none
    litellm_model: str = "text-embedding-3-small"
    litellm_api_key: str = ""
    litellm_base_url: str = ""
    ollama_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://127.0.0.1:11434"
    request_timeout_seconds: int = 4
    fallback_to_local_hash: bool = True


class MemorySearchConfig(BaseModel):
    """Hybrid retrieval scoring controls."""

    max_results: int = 6
    candidate_limit: int = 80
    min_score: float = 0.1
    vector_weight: float = 0.65
    text_weight: float = 0.35
    temporal_decay_enabled: bool = True
    temporal_half_life_days: float = 21.0


class MemoryConfig(BaseModel):
    """Layered memory settings."""

    enabled: bool = True
    path: str = "~/.captain-claw/memory.db"
    index_workspace: bool = True
    index_sessions: bool = True
    cross_session_retrieval: bool = False
    auto_sync_on_search: bool = True
    max_workspace_files: int = 400
    max_file_bytes: int = 262144
    chunk_chars: int = 1400
    chunk_overlap_chars: int = 200
    cache_ttl_seconds: int = 45
    stale_after_seconds: int = 120
    include_extensions: list[str] = Field(
        default_factory=lambda: [
            ".txt",
            ".md",
            ".markdown",
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".sql",
            ".csv",
            ".sh",
        ]
    )
    exclude_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git",
            ".hg",
            ".svn",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
        ]
    )
    layered_summaries: bool = True
    summary_batch_size: int = 10
    embeddings: MemoryEmbeddingsConfig = Field(default_factory=MemoryEmbeddingsConfig)
    search: MemorySearchConfig = Field(default_factory=MemorySearchConfig)


class ShellToolConfig(BaseModel):
    """Shell tool configuration."""

    class ExecPolicy(StrEnum):
        """Default shell execution behavior when no pattern matches."""

        ASK = "ask"
        ALLOW = "allow"
        DENY = "deny"

    timeout: int = 120
    blocked: list[str] = [
        "rm -rf /",
        "mkfs",
        ":(){:|:&};:",
    ]
    allowed_commands: list[str] = []
    default_policy: ExecPolicy = ExecPolicy.ASK
    allow_patterns: list[str] = ["python *", "pip install *"]
    deny_patterns: list[str] = ["rm -rf *", "sudo *"]


class GDriveFolderEntry(BaseModel):
    """A Google Drive folder configured for context injection."""

    id: str  # Google Drive folder ID
    name: str  # Display name (for UI and prompt)


class ReadToolConfig(BaseModel):
    """Read tool configuration."""

    model_config = ConfigDict(populate_by_name=True)

    max_file_bytes: int = 200_000  # 200 KB
    extra_dirs: list[str] = Field(default_factory=list, alias="extraDirs")
    gdrive_folders: list[GDriveFolderEntry] = Field(
        default_factory=list, alias="gdriveFolders"
    )
    file_tree_max_entries: int = Field(default=50, alias="fileTreeMaxEntries")
    file_tree_max_depth: int = Field(default=2, alias="fileTreeMaxDepth")
    file_tree_max_tokens: int = Field(default=2000, alias="fileTreeMaxTokens")
    file_tree_cache_ttl_seconds: int = Field(default=300, alias="fileTreeCacheTtlSeconds")


class WebFetchToolConfig(BaseModel):
    """Web fetch tool configuration."""

    max_chars: int = 100000


class WebSearchToolConfig(BaseModel):
    """Web search tool configuration."""

    provider: str = "brave"
    api_key: str = ""
    base_url: str = "https://api.search.brave.com/res/v1/web/search"
    max_results: int = 5
    timeout: int = 20
    safesearch: str = "moderate"


class PocketTTSToolConfig(BaseModel):
    """Pocket TTS tool configuration."""

    max_chars: int = 4000
    default_voice: str = ""
    sample_rate: int = 24000
    mp3_bitrate_kbps: int = 128
    timeout_seconds: int = 600


class ImageGenToolConfig(BaseModel):
    """Image generation tool configuration."""

    timeout_seconds: int = 120
    default_size: str = "1024x1024"
    default_quality: str = ""  # empty = let provider choose


class ImageOcrToolConfig(BaseModel):
    """Image OCR tool configuration."""

    timeout_seconds: int = 120
    max_chars: int = 120000
    default_prompt: str = ""  # empty = use built-in default
    max_pixels: int = 1568  # longest edge cap before sending to LLM (0 = no resize)
    jpeg_quality: int = 85  # JPEG compression quality (1-100) for resized images


class ImageVisionToolConfig(BaseModel):
    """Image vision tool configuration."""

    timeout_seconds: int = 120
    max_chars: int = 120000
    default_prompt: str = ""  # empty = use built-in default
    max_pixels: int = 1568  # longest edge cap before sending to LLM (0 = no resize)
    jpeg_quality: int = 85  # JPEG compression quality (1-100) for resized images


class SendMailToolConfig(BaseModel):
    """Send mail tool configuration."""

    provider: str = "smtp"  # mailgun | sendgrid | smtp
    from_address: str = ""
    from_name: str = ""
    # Mailgun
    mailgun_api_key: str = ""
    mailgun_domain: str = ""
    mailgun_base_url: str = "https://api.mailgun.net/v3"
    # SendGrid
    sendgrid_api_key: str = ""
    sendgrid_base_url: str = "https://api.sendgrid.com/v3/mail/send"
    # SMTP
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    # General
    timeout: int = 60
    max_attachment_bytes: int = 26214400  # 25 MB


class TypesenseToolConfig(BaseModel):
    """Typesense tool configuration."""

    host: str = "localhost"
    port: int = 8108
    protocol: str = "http"
    api_key: str = ""
    default_collection: str = ""
    timeout: int = 30
    connection_timeout: int = 5


class GwsToolConfig(BaseModel):
    """Google Workspace CLI (gws) tool configuration."""

    binary_path: str = ""  # custom path to the gws binary; empty = find in PATH


class EditToolConfig(BaseModel):
    """Edit tool configuration."""

    max_file_bytes: int = 500_000  # 500 KB max file size for editing
    backup_enabled: bool = True
    backup_dir: str = ".captain-claw/backups"  # relative to runtime_base_path
    max_backups: int = 5


class BrowserToolConfig(BaseModel):
    """Browser automation tool configuration."""

    timeout_seconds: int = 60
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    screenshot_max_pixels: int = 800  # longest edge cap before sending to vision LLM
    screenshot_jpeg_quality: int = 40
    user_agent: str = ""  # empty = Playwright default
    default_wait_seconds: float = 2.0
    max_session_duration_seconds: int = 1800  # 30 min auto-close safety
    network_capture_enabled: bool = True
    network_max_captures: int = 500
    network_max_body_bytes: int = 10000
    network_filter_static: bool = True
    network_auto_record: bool = True  # start recording automatically on navigate
    # Credential management
    credential_encryption_key: str = ""  # Fernet key — or CLAW_BROWSER_CREDENTIAL_KEY env var
    cookie_persistence: bool = True  # save/restore cookies across sessions
    login_timeout_seconds: int = 30  # max wait for login form fill + submit
    login_verify_wait_seconds: float = 3.0  # wait after submit before checking login success


class PinchTabConfig(BaseModel):
    """PinchTab browser automation configuration.

    PinchTab is an HTTP-based browser automation server (standalone Go binary)
    that provides token-efficient browser control via accessibility tree
    snapshots rather than expensive screenshots.
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 9867
    token: str = ""  # bearer token — or PINCHTAB_TOKEN env var
    auto_start: bool = True  # auto-start pinchtab server if not running
    binary_path: str = ""  # custom path to pinchtab binary; empty = find in PATH
    headless: bool = True
    stealth_level: str = "light"  # light | medium | full
    default_profile: str = ""  # persistent profile name; empty = no profile
    timeout_seconds: int = 60
    max_tabs: int = 10
    block_ads: bool = False
    block_images: bool = False
    allow_evaluate: bool = False  # JS eval disabled by default for security


class ScreenCaptureToolConfig(BaseModel):
    """Screen capture and voice command configuration."""

    # Hotkey settings
    hotkey_enabled: bool = False
    hotkey_trigger_key: str = "shift"  # key to double-tap (shift, ctrl, alt, caps_lock)
    hotkey_double_tap_ms: int = 400  # ms between taps to count as double
    hotkey_triple_tap_wait_ms: int = 300  # ms to wait after double-tap for a potential 3rd tap

    # Capture settings
    default_monitor: int = 0  # 0=all monitors, 1=primary, 2=secondary, ...
    timeout_seconds: int = 30

    # Audio settings
    max_recording_seconds: float = 30.0  # max voice recording duration
    audio_sample_rate: int = 16000  # Hz for mic recording
    save_audio: bool = False  # persist WAV files to workspace

    # STT settings
    stt_provider: str = ""  # "soniox", "openai", "gemini", or "" for auto-detect
    stt_model: str = ""  # explicit model ID; empty = auto-detect provider


class ClipboardToolConfig(BaseModel):
    """Clipboard tool configuration."""

    timeout_seconds: int = 10


class ToolsConfig(BaseModel):
    """Tools configuration."""

    enabled: list[str] = [
        "shell",
        "read",
        "write",
        "glob",
        "web_fetch",
        "web_search",
        "pdf_extract",
        "docx_extract",
        "xlsx_extract",
        "pptx_extract",
        "pocket_tts",
        "image_gen",
        "image_ocr",
        "image_vision",
        "send_mail",
        "google_drive",
        "google_calendar",
        "google_mail",
        "gws",
        "todo",
        "contacts",
        "scripts",
        "apis",
        "playbooks",
        "typesense",
        "datastore",
        "personality",
        "termux",
        "edit",
        "browser",
        "screen_capture",
        "desktop_action",
    ]
    shell: ShellToolConfig = Field(default_factory=ShellToolConfig)
    read: ReadToolConfig = Field(default_factory=ReadToolConfig)
    web_fetch: WebFetchToolConfig = Field(default_factory=WebFetchToolConfig)
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)
    pocket_tts: PocketTTSToolConfig = Field(default_factory=PocketTTSToolConfig)
    image_gen: ImageGenToolConfig = Field(default_factory=ImageGenToolConfig)
    image_ocr: ImageOcrToolConfig = Field(default_factory=ImageOcrToolConfig)
    image_vision: ImageVisionToolConfig = Field(default_factory=ImageVisionToolConfig)
    send_mail: SendMailToolConfig = Field(default_factory=SendMailToolConfig)
    typesense: TypesenseToolConfig = Field(default_factory=TypesenseToolConfig)
    gws: GwsToolConfig = Field(default_factory=GwsToolConfig)
    edit: EditToolConfig = Field(default_factory=EditToolConfig)
    browser: BrowserToolConfig = Field(default_factory=BrowserToolConfig)
    pinchtab: PinchTabConfig = Field(default_factory=PinchTabConfig)
    clipboard: ClipboardToolConfig = Field(default_factory=ClipboardToolConfig)
    screen_capture: ScreenCaptureToolConfig = Field(default_factory=ScreenCaptureToolConfig)
    require_confirmation: list[str] = ["shell", "write", "edit"]
    plugin_dirs: list[str] = ["skills/tools"]

    # These tools are always available — re-inject if removed by user.
    _ALWAYS_ENABLED: frozenset[str] = frozenset({"read", "write", "edit", "personality", "botport", "playbooks", "browser", "datastore", "direct_api"})

    @model_validator(mode="after")
    def _ensure_always_enabled(self) -> "ToolsConfig":
        for tool in self._ALWAYS_ENABLED:
            if tool not in self.enabled:
                self.enabled.append(tool)
        return self

    duplicate_call_max: int = Field(
        default=1,
        description=(
            "Maximum times a tool may be called on the same target "
            "(path/URL/pattern) per turn before the call is blocked. "
            "Set to 1 to block on the first repeat; 2 to allow one retry."
        ),
    )


class SkillEntryConfig(BaseModel):
    """Per-skill config overrides."""

    model_config = ConfigDict(populate_by_name=True)

    enabled: bool | None = None
    api_key: str = Field(default="", alias="apiKey")
    env: dict[str, str] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class SkillsLoadConfig(BaseModel):
    """Skill loading settings."""

    model_config = ConfigDict(populate_by_name=True)

    extra_dirs: list[str] = Field(default_factory=list, alias="extraDirs")
    plugin_dirs: list[str] = Field(default_factory=list, alias="pluginDirs")
    watch: bool = True
    watch_debounce_ms: int = Field(default=250, alias="watchDebounceMs")


class SkillsInstallConfig(BaseModel):
    """Skill dependency install preferences."""

    model_config = ConfigDict(populate_by_name=True)

    prefer_brew: bool = Field(default=True, alias="preferBrew")
    node_manager: Literal["npm", "pnpm", "yarn", "bun"] = Field(
        default="npm",
        alias="nodeManager",
    )


class SkillsConfig(BaseModel):
    """Skills subsystem configuration."""

    model_config = ConfigDict(populate_by_name=True)

    managed_dir: str = "~/.captain-claw/skills"
    allow_bundled: list[str] = Field(default_factory=list, alias="allowBundled")
    entries: dict[str, SkillEntryConfig] = Field(default_factory=dict)
    load: SkillsLoadConfig = Field(default_factory=SkillsLoadConfig)
    install: SkillsInstallConfig = Field(default_factory=SkillsInstallConfig)
    max_skills_in_prompt: int = 64
    max_skills_prompt_chars: int = 16000
    max_skill_file_bytes: int = 131072
    max_candidates_per_root: int = 2048
    max_skills_loaded_per_source: int = 256
    search_source_url: str = "https://github.com/VoltAgent/awesome-openclaw-skills"
    search_limit: int = 10
    search_max_candidates: int = 5000
    search_http_timeout_seconds: int = 20


class GuardTypeConfig(BaseModel):
    """Guard behavior for a single guard type."""

    enabled: bool = False
    level: Literal["stop_suspicious", "ask_for_approval"] = "stop_suspicious"


class GuardConfig(BaseModel):
    """Guarding configuration."""

    input: GuardTypeConfig = Field(default_factory=GuardTypeConfig)
    output: GuardTypeConfig = Field(default_factory=GuardTypeConfig)
    script_tool: GuardTypeConfig = Field(default_factory=GuardTypeConfig)


class SessionConfig(BaseModel):
    """Session configuration."""

    storage: str = "sqlite"
    path: str = str(DEFAULT_DB_PATH)
    auto_save: bool = True


class WorkspaceConfig(BaseModel):
    """Workspace/output root configuration."""

    path: str = "./workspace"


class UIConfig(BaseModel):
    """UI configuration."""

    theme: str = "dark"
    show_tokens: bool = True
    streaming: bool = True
    colors: bool = True
    monitor_trace_llm: bool = False
    monitor_trace_pipeline: bool = True
    monitor_full_output: bool = False


class ExecutionQueueConfig(BaseModel):
    """Follow-up queue behavior while sessions are busy."""

    mode: str = "collect"
    debounce_ms: int = 1000
    cap: int = 20
    drop: str = "summarize"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "console"
    llm_session_logging: bool = False


class TelegramConfig(BaseModel):
    """Telegram bot UI configuration."""

    enabled: bool = False
    bot_token: str = ""
    api_base_url: str = "https://api.telegram.org"
    poll_timeout_seconds: int = 25
    pairing_ttl_minutes: int = 30


class SlackConfig(BaseModel):
    """Slack bot UI configuration."""

    enabled: bool = False
    bot_token: str = ""
    app_token: str = ""
    api_base_url: str = "https://slack.com/api"
    poll_timeout_seconds: int = 25
    pairing_ttl_minutes: int = 30


class DiscordConfig(BaseModel):
    """Discord bot UI configuration."""

    enabled: bool = False
    bot_token: str = ""
    application_id: int = 0
    api_base_url: str = "https://discord.com/api/v10"
    poll_timeout_seconds: int = 25
    pairing_ttl_minutes: int = 30
    require_mention_in_guild: bool = True


class TodoConfig(BaseModel):
    """Cross-session to-do memory settings."""

    enabled: bool = True
    auto_capture: bool = True
    inject_on_session_load: bool = True
    max_items_in_prompt: int = 10
    archive_after_days: int = 30


class AddressBookConfig(BaseModel):
    """Cross-session address book / contact memory settings."""

    enabled: bool = True
    auto_capture: bool = True
    inject_on_mention: bool = True
    max_items_in_prompt: int = 5


class ScriptsMemoryConfig(BaseModel):
    """Cross-session script/file memory settings."""

    enabled: bool = True
    auto_capture: bool = True
    inject_on_mention: bool = True
    max_items_in_prompt: int = 5


class ApisMemoryConfig(BaseModel):
    """Cross-session API memory settings."""

    enabled: bool = True
    auto_capture: bool = True
    inject_on_mention: bool = True
    max_items_in_prompt: int = 5


class ScaleConfig(BaseModel):
    """Scale loop thresholds for list processing tasks."""

    # Minimum member count to activate the full scale advisory
    # (scale progress tracking, guards, micro-loop eligibility).
    scale_advisory_min_members: int = 7

    # Minimum member count to activate lightweight scale progress
    # (progress indicators only, no micro-loop).
    lightweight_progress_min_members: int = 3

    # Automatic task rephrasing — rewrites complex user prompts into
    # structured, agent-friendly format before execution.  Especially
    # beneficial for list-processing tasks with detailed formatting.
    task_rephrase_enabled: bool = True

    # Minimum user input length (chars) to consider rephrasing.
    # Very short prompts (e.g. "summarize this") don't benefit.
    task_rephrase_min_chars: int = 120

    # Research extraction settings — used when list items are plain-text
    # entities (company names, product names, etc.) that require web search
    # + web_fetch to gather information.
    research_search_results: int = 5       # Brave search results per item
    research_max_chars_per_fetch: int = 15000  # Max chars per fetched page
    research_query_keywords: int = 10     # Max keywords in search query (proper nouns + content words)


class DeepMemoryConfig(BaseModel):
    """Typesense-backed deep memory for long-term searchable content."""

    enabled: bool = False
    host: str = "localhost"
    port: int = 8108
    protocol: str = "http"
    api_key: str = ""
    collection_name: str = "captain_claw_deep_memory"
    embedding_dims: int = 1536
    auto_embed: bool = True
    layered_summaries: bool = True


class DatastoreConfig(BaseModel):
    """User-facing relational datastore settings."""

    enabled: bool = True
    path: str = "~/.captain-claw/datastore.db"
    inject_table_list: bool = True
    max_rows_per_table: int = 100_000
    max_tables: int = 50
    max_query_rows: int = 500
    max_export_rows: int = 50_000


class BotPortClientConfig(BaseModel):
    """BotPort agent-to-agent routing hub connection settings."""

    enabled: bool = False
    url: str = ""  # ws://botport-host:23180/ws
    instance_name: str = "default"
    key: str = ""
    secret: str = ""
    advertise_personas: bool = True
    advertise_tools: bool = True
    advertise_models: bool = True
    max_concurrent: int = 5
    reconnect_delay_seconds: float = 5.0
    heartbeat_interval_seconds: float = 30.0


class OrchestratorConfig(BaseModel):
    """Parallel session orchestration settings."""

    max_parallel: int = 5
    max_agents: int = 50
    idle_evict_seconds: float = 300.0
    worker_timeout_seconds: float = 600.0
    timeout_grace_seconds: float = 60.0
    worker_max_retries: int = 2


class GoogleOAuthConfig(BaseModel):
    """Google OAuth2 configuration for Gemini API access via Vertex AI."""

    enabled: bool = False
    client_id: str = ""
    client_secret: str = ""
    project_id: str = ""  # GCP project ID (required for Vertex AI)
    location: str = "us-central1"  # Vertex AI region
    scopes: list[str] = Field(default_factory=lambda: [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/gmail.readonly",
        "openid",
        "email",
    ])


class WebConfig(BaseModel):
    """Web UI configuration."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 23080
    # OpenAI-compatible API proxy settings
    api_enabled: bool = True
    api_pool_max_agents: int = 50
    api_pool_idle_seconds: float = 600.0
    # Authentication – set auth_token to enable; empty string = auth disabled
    auth_token: str = ""
    auth_cookie_max_age: int = 90  # days


class ProviderKeysConfig(BaseModel):
    """Per-provider API keys managed from the settings UI."""

    openai: str = ""
    openai_headers: list[str] = Field(default_factory=list)
    anthropic: str = ""
    anthropic_headers: list[str] = Field(default_factory=list)
    gemini: str = ""
    gemini_headers: list[str] = Field(default_factory=list)
    xai: str = ""
    xai_headers: list[str] = Field(default_factory=list)
    openrouter: str = ""
    openrouter_headers: list[str] = Field(default_factory=list)
    brave: str = ""

    def headers_for(self, provider: str) -> dict[str, str]:
        """Parse extra header tags for *provider* into a dict.

        Each tag is ``"Header-Name: value"``; entries without a colon are
        silently skipped.  Returns an empty dict when the provider has no
        headers configured.
        """
        raw: list[str] = getattr(self, f"{provider}_headers", []) or []
        result: dict[str, str] = {}
        for tag in raw:
            if ":" in tag:
                k, v = tag.split(":", 1)
                result[k.strip()] = v.strip()
        return result


class Config(BaseSettings):
    """Main configuration for Captain Claw."""

    provider_keys: ProviderKeysConfig = Field(default_factory=ProviderKeysConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    guards: GuardConfig = Field(default_factory=GuardConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    execution_queue: ExecutionQueueConfig = Field(default_factory=ExecutionQueueConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    google_oauth: GoogleOAuthConfig = Field(default_factory=GoogleOAuthConfig)
    todo: TodoConfig = Field(default_factory=TodoConfig)
    addressbook: AddressBookConfig = Field(default_factory=AddressBookConfig)
    scripts_memory: ScriptsMemoryConfig = Field(default_factory=ScriptsMemoryConfig)
    apis_memory: ApisMemoryConfig = Field(default_factory=ApisMemoryConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    botport: BotPortClientConfig = Field(default_factory=BotPortClientConfig)
    scale: ScaleConfig = Field(default_factory=ScaleConfig)
    deep_memory: DeepMemoryConfig = Field(default_factory=DeepMemoryConfig)
    datastore: DatastoreConfig = Field(default_factory=DatastoreConfig)

    model_config = SettingsConfigDict(
        env_prefix="CLAW_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @staticmethod
    def _apply_legacy_top_level_config(data: dict[str, Any]) -> dict[str, Any]:
        """Map legacy flat config keys into nested model structure."""
        payload = dict(data or {})
        model_payload = payload.get("model")
        if isinstance(model_payload, dict):
            model_data: dict[str, Any] = dict(model_payload)
        else:
            model_data = {}

        legacy_openai_key = str(payload.get("openai_api_key", "")).strip()
        legacy_ollama_base = str(payload.get("ollama_base_url", "")).strip()
        if legacy_openai_key and not str(model_data.get("api_key", "")).strip():
            model_data["api_key"] = legacy_openai_key
        if legacy_ollama_base and not str(model_data.get("base_url", "")).strip():
            model_data["base_url"] = legacy_ollama_base
        if model_data:
            payload["model"] = model_data

        payload.pop("openai_api_key", None)
        payload.pop("ollama_base_url", None)
        return payload

    @staticmethod
    def _read_dotenv_file(path: Path | str = ".env") -> dict[str, str]:
        """Read simple KEY=VALUE pairs from .env (best-effort, no interpolation)."""
        env_path = Path(path).expanduser()
        if not env_path.exists() or not env_path.is_file():
            return {}
        values: dict[str, str] = {}
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return {}
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_key = str(key).strip()
            env_value = str(value).strip()
            if not env_key:
                continue
            if (
                len(env_value) >= 2
                and env_value[0] == env_value[-1]
                and env_value[0] in {'"', "'"}
            ):
                env_value = env_value[1:-1]
            values[env_key] = env_value
        return values

    @classmethod
    def resolve_default_config_path(cls) -> Path:
        """Resolve default config path with local-first precedence."""
        local_path = Path.cwd() / LOCAL_CONFIG_FILENAME
        if local_path.exists():
            return local_path
        return DEFAULT_CONFIG_PATH

    @staticmethod
    def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge *overlay* into *base* (overlay wins on conflict).

        - Dicts are merged recursively.
        - Lists and scalars in *overlay* replace *base* entirely.
        """
        merged = dict(base)
        for key, overlay_val in overlay.items():
            base_val = merged.get(key)
            if isinstance(base_val, dict) and isinstance(overlay_val, dict):
                merged[key] = Config._deep_merge(base_val, overlay_val)
            else:
                merged[key] = overlay_val
        return merged

    @classmethod
    def _read_yaml_data(cls, config_path: Path) -> dict[str, Any]:
        """Read and parse a single YAML config file into a dict."""
        if not config_path.exists():
            return {}
        raw_text = config_path.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(raw_text) or {}
        except yaml.constructor.ConstructorError as e:
            message = str(e)
            if "python/object/apply:captain_claw.config." not in message:
                raise
            # Legacy configs could serialize enums as python/object/apply tags.
            sanitized = re.sub(
                r"(?m)^(\s*[\w_]+:\s*)!!python/object/apply:captain_claw\.config\.[^\n]+\n\s*-\s*([^\n]+)\s*$",
                r"\1\2",
                raw_text,
            )
            data = yaml.safe_load(sanitized) or {}
            if sanitized != raw_text:
                try:
                    config_path.write_text(sanitized, encoding="utf-8")
                except Exception:
                    pass
        return data

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "Config":
        """Load configuration from YAML file(s).

        When *path* is ``None`` (normal startup), both config sources are
        merged with the home-directory file (``~/.captain-claw/config.yaml``)
        taking precedence over the local project file (``./config.yaml``).
        This allows the settings page to write user overrides to the home
        file while the project ships its own defaults in the local file.

        When an explicit *path* is given, only that file is loaded.
        """
        if path is not None:
            data = cls._read_yaml_data(Path(path).expanduser())
            data = cls._apply_legacy_top_level_config(data)
            return cls(**data)

        # ── Merge: local (base) + home (overlay) ──────────
        local_path = Path.cwd() / LOCAL_CONFIG_FILENAME
        home_path = DEFAULT_CONFIG_PATH

        base_data = cls._read_yaml_data(local_path)
        home_data = cls._read_yaml_data(home_path)

        if base_data and home_data:
            data = cls._deep_merge(base_data, home_data)
        elif home_data:
            data = home_data
        else:
            data = base_data

        if not data:
            return cls()

        data = cls._apply_legacy_top_level_config(data)
        return cls(**data)

    @classmethod
    def load(cls) -> "Config":
        """Load configuration, preferring env vars over YAML."""
        # First load from YAML
        config = cls.from_yaml()
        dotenv_values = cls._read_dotenv_file(".env")

        # Security-sensitive override: prefer Telegram token from env/.env when present.
        # (YAML init kwargs otherwise take precedence over settings sources.)
        env_overlay = cls()
        env_token = str(getattr(env_overlay.telegram, "bot_token", "")).strip()
        if not env_token:
            # Also support a direct non-prefixed variable for operator convenience.
            env_token = (
                str(os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
                or str(dotenv_values.get("TELEGRAM_BOT_TOKEN", "")).strip()
            )
        if env_token:
            config.telegram.bot_token = env_token

        # Security-sensitive override: prefer Slack tokens from env/.env when present.
        slack_bot_token = str(getattr(env_overlay.slack, "bot_token", "")).strip()
        if not slack_bot_token:
            slack_bot_token = (
                str(os.getenv("SLACK_BOT_TOKEN", "")).strip()
                or str(dotenv_values.get("SLACK_BOT_TOKEN", "")).strip()
            )
        if slack_bot_token:
            config.slack.bot_token = slack_bot_token
        slack_app_token = str(getattr(env_overlay.slack, "app_token", "")).strip()
        if not slack_app_token:
            slack_app_token = (
                str(os.getenv("SLACK_APP_TOKEN", "")).strip()
                or str(dotenv_values.get("SLACK_APP_TOKEN", "")).strip()
            )
        if slack_app_token:
            config.slack.app_token = slack_app_token

        # Security-sensitive override: prefer Discord bot token from env/.env when present.
        discord_bot_token = str(getattr(env_overlay.discord, "bot_token", "")).strip()
        if not discord_bot_token:
            discord_bot_token = (
                str(os.getenv("DISCORD_BOT_TOKEN", "")).strip()
                or str(dotenv_values.get("DISCORD_BOT_TOKEN", "")).strip()
            )
        if discord_bot_token:
            config.discord.bot_token = discord_bot_token
        raw_app_id = getattr(env_overlay.discord, "application_id", 0)
        discord_application_id = ""
        try:
            if int(raw_app_id) > 0:
                discord_application_id = str(int(raw_app_id))
        except Exception:
            discord_application_id = ""
        if not discord_application_id:
            discord_application_id = (
                str(os.getenv("DISCORD_APPLICATION_ID", "")).strip()
                or str(dotenv_values.get("DISCORD_APPLICATION_ID", "")).strip()
            )
        if discord_application_id:
            try:
                config.discord.application_id = int(discord_application_id)
            except Exception:
                pass

        # Security-sensitive override: prefer Brave API key from env/.env when present.
        env_brave_key = str(getattr(env_overlay.tools.web_search, "api_key", "")).strip()
        if not env_brave_key:
            env_brave_key = (
                str(os.getenv("BRAVE_API_KEY", "")).strip()
                or str(dotenv_values.get("BRAVE_API_KEY", "")).strip()
            )
        if env_brave_key:
            config.tools.web_search.api_key = env_brave_key

        # Security-sensitive override: send_mail credentials from env/.env.
        # MAIL_PROVIDER is handled separately: it ALWAYS overrides the config
        # value because `provider` has a non-empty default ("smtp") that would
        # prevent the standard "fill if empty" logic from ever applying.
        _mail_provider_env = (
            str(os.getenv("MAIL_PROVIDER", "")).strip()
            or str(dotenv_values.get("MAIL_PROVIDER", "")).strip()
        )
        if _mail_provider_env:
            config.tools.send_mail.provider = _mail_provider_env

        _mail_env_map = {
            "MAILGUN_API_KEY": "mailgun_api_key",
            "MAILGUN_DOMAIN": "mailgun_domain",
            "SENDGRID_API_KEY": "sendgrid_api_key",
            "MAIL_FROM_ADDRESS": "from_address",
            "MAIL_FROM_NAME": "from_name",
            "SMTP_HOST": "smtp_host",
            "SMTP_PORT": "smtp_port",
            "SMTP_USERNAME": "smtp_username",
            "SMTP_PASSWORD": "smtp_password",
        }
        for env_key, cfg_attr in _mail_env_map.items():
            if not str(getattr(config.tools.send_mail, cfg_attr, "") or "").strip():
                val = (
                    str(os.getenv(env_key, "")).strip()
                    or str(dotenv_values.get(env_key, "")).strip()
                )
                if val:
                    # smtp_port needs int conversion.
                    if cfg_attr == "smtp_port":
                        try:
                            setattr(config.tools.send_mail, cfg_attr, int(val))
                        except ValueError:
                            pass
                    else:
                        setattr(config.tools.send_mail, cfg_attr, val)

        # Auto-detect provider from available credentials when the provider
        # is still the default "smtp" and no explicit MAIL_PROVIDER was set.
        sm = config.tools.send_mail
        if not _mail_provider_env and sm.provider == "smtp":
            if str(sm.mailgun_api_key or "").strip() and str(sm.mailgun_domain or "").strip():
                sm.provider = "mailgun"
            elif str(sm.sendgrid_api_key or "").strip():
                sm.provider = "sendgrid"

        # Google OAuth env overrides.
        for env_key, cfg_attr in [
            ("GOOGLE_OAUTH_CLIENT_ID", "client_id"),
            ("GOOGLE_OAUTH_CLIENT_SECRET", "client_secret"),
            ("GOOGLE_OAUTH_PROJECT_ID", "project_id"),
        ]:
            if not str(getattr(config.google_oauth, cfg_attr, "") or "").strip():
                val = (
                    str(os.getenv(env_key, "")).strip()
                    or str(dotenv_values.get(env_key, "")).strip()
                )
                if val:
                    setattr(config.google_oauth, cfg_attr, val)

        # Allow enabling via env: GOOGLE_OAUTH_ENABLED=true
        if not config.google_oauth.enabled:
            env_enabled = (
                str(os.getenv("GOOGLE_OAUTH_ENABLED", "")).strip().lower()
                or str(dotenv_values.get("GOOGLE_OAUTH_ENABLED", "")).strip().lower()
            )
            if env_enabled in {"true", "1", "yes"}:
                config.google_oauth.enabled = True

        # Auto-enable Google OAuth when client_id and client_secret are set.
        if (
            not config.google_oauth.enabled
            and str(config.google_oauth.client_id or "").strip()
            and str(config.google_oauth.client_secret or "").strip()
        ):
            config.google_oauth.enabled = True

        # Security-sensitive override: Typesense API key from env/.env.
        env_typesense_key = str(getattr(env_overlay.tools.typesense, "api_key", "")).strip()
        if not env_typesense_key:
            env_typesense_key = (
                str(os.getenv("TYPESENSE_API_KEY", "")).strip()
                or str(dotenv_values.get("TYPESENSE_API_KEY", "")).strip()
            )
        if env_typesense_key:
            config.tools.typesense.api_key = env_typesense_key
            # Also set on deep_memory if it doesn't have its own key.
            if not str(config.deep_memory.api_key or "").strip():
                config.deep_memory.api_key = env_typesense_key

        # Compatibility fallbacks for common provider env vars.
        provider = str(config.model.provider or "").strip().lower()
        if not str(config.model.api_key or "").strip():
            if provider in {"openai", "chatgpt"}:
                config.model.api_key = (
                    str(os.getenv("OPENAI_API_KEY", "")).strip()
                    or str(dotenv_values.get("OPENAI_API_KEY", "")).strip()
                )
            elif provider in {"anthropic", "claude"}:
                config.model.api_key = (
                    str(os.getenv("ANTHROPIC_API_KEY", "")).strip()
                    or str(dotenv_values.get("ANTHROPIC_API_KEY", "")).strip()
                )
            elif provider in {"gemini", "google"}:
                config.model.api_key = (
                    str(os.getenv("GEMINI_API_KEY", "")).strip()
                    or str(dotenv_values.get("GEMINI_API_KEY", "")).strip()
                    or str(os.getenv("GOOGLE_API_KEY", "")).strip()
                    or str(dotenv_values.get("GOOGLE_API_KEY", "")).strip()
                )
        # Provider-keys fallback (settings UI keys).
        if not str(config.model.api_key or "").strip():
            pk = config.provider_keys
            if provider in {"openai", "chatgpt"} and pk.openai:
                config.model.api_key = pk.openai
            elif provider in {"anthropic", "claude"} and pk.anthropic:
                config.model.api_key = pk.anthropic
            elif provider in {"gemini", "google"} and pk.gemini:
                config.model.api_key = pk.gemini
            elif provider in {"xai", "grok"} and pk.xai:
                config.model.api_key = pk.xai

        # Brave Search key from provider_keys.
        if not str(config.tools.web_search.api_key or "").strip() and config.provider_keys.brave:
            config.tools.web_search.api_key = config.provider_keys.brave

        # Export provider_keys to environment so that libraries which
        # check env vars directly (e.g. LiteLLM image_generation
        # validate_environment) find them.
        pk = config.provider_keys
        _env_exports: list[tuple[str, str]] = [
            ("OPENAI_API_KEY", pk.openai),
            ("ANTHROPIC_API_KEY", pk.anthropic),
            ("GEMINI_API_KEY", pk.gemini),
            ("GOOGLE_API_KEY", pk.gemini),
            ("XAI_API_KEY", pk.xai),
        ]
        for env_name, pk_value in _env_exports:
            if pk_value and not os.getenv(env_name):
                os.environ[env_name] = pk_value

        if provider == "ollama" and not str(config.model.base_url or "").strip():
            base_url = (
                str(os.getenv("OLLAMA_BASE_URL", "")).strip()
                or str(dotenv_values.get("OLLAMA_BASE_URL", "")).strip()
            )
            if base_url:
                config.model.base_url = base_url

        return config

    def save(self, path: Path | str | None = None) -> None:
        """Save configuration to YAML file."""
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH

        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-safe dict so enums/complex objects are plain scalars.
        data = self.model_dump(mode="json", exclude_none=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def resolved_workspace_path(self, runtime_base: Path | str | None = None) -> Path:
        """Resolve workspace path, anchoring relative paths to runtime base/cwd."""
        raw = Path(self.workspace.path).expanduser()
        if raw.is_absolute():
            return raw.resolve()
        anchor = Path(runtime_base).expanduser().resolve() if runtime_base is not None else Path.cwd().resolve()
        return (anchor / raw).resolve()


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
