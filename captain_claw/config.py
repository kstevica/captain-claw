"""Configuration management for Captain Claw."""

import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
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

    provider: str = "ollama"
    model: str = "minimax-m2.5:cloud"
    temperature: float = 0.7
    max_tokens: int = 32000
    api_key: str = ""
    base_url: str = ""
    allowed: list[AllowedModelConfig] = Field(default_factory=list)


class ContextConfig(BaseModel):
    """Context window configuration."""

    max_tokens: int = 160000
    compaction_threshold: float = 0.8
    compaction_ratio: float = 0.4


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
    embeddings: MemoryEmbeddingsConfig = Field(default_factory=MemoryEmbeddingsConfig)
    search: MemorySearchConfig = Field(default_factory=MemorySearchConfig)


class ShellToolConfig(BaseModel):
    """Shell tool configuration."""

    class ExecPolicy(StrEnum):
        """Default shell execution behavior when no pattern matches."""

        ASK = "ask"
        ALLOW = "allow"
        DENY = "deny"

    timeout: int = 30
    blocked: list[str] = [
        "rm -rf /",
        "mkfs",
        ":(){:|:&};:",
    ]
    allowed_commands: list[str] = []
    default_policy: ExecPolicy = ExecPolicy.ASK
    allow_patterns: list[str] = ["python *", "pip install *"]
    deny_patterns: list[str] = ["rm -rf *", "sudo *"]


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
    ]
    shell: ShellToolConfig = Field(default_factory=ShellToolConfig)
    web_fetch: WebFetchToolConfig = Field(default_factory=WebFetchToolConfig)
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)
    pocket_tts: PocketTTSToolConfig = Field(default_factory=PocketTTSToolConfig)
    require_confirmation: list[str] = ["shell", "write"]
    plugin_dirs: list[str] = ["skills/tools"]


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


class OrchestratorConfig(BaseModel):
    """Parallel session orchestration settings."""

    max_parallel: int = 5
    max_agents: int = 50
    idle_evict_seconds: float = 300.0
    worker_timeout_seconds: float = 300.0
    worker_max_retries: int = 2


class WebConfig(BaseModel):
    """Web UI configuration."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8340


class Config(BaseSettings):
    """Main configuration for Captain Claw."""

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
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)

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

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(path).expanduser() if path else cls.resolve_default_config_path()

        if not config_path.exists():
            return cls()

        raw_text = config_path.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(raw_text) or {}
        except yaml.constructor.ConstructorError as e:
            message = str(e)
            if "python/object/apply:captain_claw.config." not in message:
                raise
            # Legacy configs could serialize enums as python/object/apply tags.
            # Convert these to plain scalar values so safe_load can parse the file.
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
