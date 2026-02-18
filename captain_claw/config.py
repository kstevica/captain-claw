"""Configuration management for Captain Claw."""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
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


class ShellToolConfig(BaseModel):
    """Shell tool configuration."""

    timeout: int = 30
    blocked: list[str] = [
        "rm -rf /",
        "mkfs",
        ":(){:|:&};:",
    ]
    allowed_commands: list[str] = []


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
    ]
    shell: ShellToolConfig = Field(default_factory=ShellToolConfig)
    web_fetch: WebFetchToolConfig = Field(default_factory=WebFetchToolConfig)
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)
    require_confirmation: list[str] = ["shell", "write"]


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


class Config(BaseSettings):
    """Main configuration for Captain Claw."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    guards: GuardConfig = Field(default_factory=GuardConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    execution_queue: ExecutionQueueConfig = Field(default_factory=ExecutionQueueConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(
        env_prefix="CLAW_",
        env_file=".env",
        env_nested_delimiter="__",
    )

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
        
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        
        return cls(**data)

    @classmethod
    def load(cls) -> "Config":
        """Load configuration, preferring env vars over YAML."""
        # First load from YAML
        config = cls.from_yaml()
        
        # Override with environment variables if set
        # Pydantic-settings handles this automatically via BaseSettings
        return config

    def save(self, path: Path | str | None = None) -> None:
        """Save configuration to YAML file."""
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict, excluding defaults
        data = self.model_dump(exclude_none=True)
        
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

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
