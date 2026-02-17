"""Configuration management for Captain Claw."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Paths
DEFAULT_CONFIG_PATH = Path("~/.captain-claw/config.yaml").expanduser()
DEFAULT_DB_PATH = Path("~/.captain-claw/sessions.db").expanduser()


class ModelConfig(BaseModel):
    """Model configuration."""

    provider: str = "ollama"
    model: str = "minimax-m2.5:cloud"
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: str = ""


class ContextConfig(BaseModel):
    """Context window configuration."""

    max_tokens: int = 100000
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


class ToolsConfig(BaseModel):
    """Tools configuration."""

    enabled: list[str] = ["shell", "read", "write", "glob", "web_fetch"]
    shell: ShellToolConfig = Field(default_factory=ShellToolConfig)
    require_confirmation: list[str] = ["shell", "write"]


class SessionConfig(BaseModel):
    """Session configuration."""

    storage: str = "sqlite"
    path: str = str(DEFAULT_DB_PATH)
    auto_save: bool = True


class UIConfig(BaseModel):
    """UI configuration."""

    theme: str = "dark"
    show_tokens: bool = True
    streaming: bool = True
    colors: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "console"


class Config(BaseSettings):
    """Main configuration for Captain Claw."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(
        env_prefix="CLAW_",
        env_file=".env",
        env_nested_delimiter="__",
    )

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        
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
