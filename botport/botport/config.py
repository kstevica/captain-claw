"""BotPort configuration management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

try:
    DEFAULT_CONFIG_PATH = Path("~/.botport/config.yaml").expanduser()
except RuntimeError:
    DEFAULT_CONFIG_PATH = Path("/tmp/.botport/config.yaml")
LOCAL_CONFIG_FILENAME = "config.yaml"


class ServerConfig(BaseModel):
    """Web server settings."""

    host: str = "0.0.0.0"
    port: int = 23180
    dashboard_enabled: bool = True


class RoutingConfig(BaseModel):
    """Routing strategy settings."""

    strategy: str = "tag_match"  # tag_match | llm_assisted
    fallback: str = "reject"  # reject | queue_until_available


class ConcernConfig(BaseModel):
    """Concern lifecycle settings."""

    idle_timeout_seconds: int = 600
    max_follow_ups: int = 20
    max_concurrent_per_instance: int = 10
    timeout_check_interval_seconds: int = 30


class AuthKeyEntry(BaseModel):
    """A single authorized instance key."""

    instance: str = ""
    key: str = ""
    secret: str = ""


class AuthConfig(BaseModel):
    """Authentication settings."""

    enabled: bool = False
    keys: list[AuthKeyEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_null_keys(cls, data: Any) -> Any:
        """YAML parses 'keys:' with only comments as None — coerce to []."""
        if isinstance(data, dict) and data.get("keys") is None:
            data["keys"] = []
        return data


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = "info"
    concern_history: bool = True
    retention_days: int = 30


class LLMConfig(BaseModel):
    """LLM settings for smart routing via litellm.

    Model names use litellm's ``provider/model`` format, e.g.:
      - ``ollama/llama3.2``
      - ``gemini/gemini-2.0-flash``
      - ``openai/gpt-4o-mini``
      - ``anthropic/claude-3-haiku-20240307``

    For backward compatibility the old ``provider`` + ``model`` fields are
    merged automatically (e.g. provider="gemini", model="gemini-2.0-flash"
    → model="gemini/gemini-2.0-flash").
    """

    enabled: bool = False
    provider: str = ""  # legacy — merged into model if set
    model: str = "ollama/llama3.2"
    api_key: str = ""
    base_url: str = ""  # empty = let litellm use its provider defaults
    temperature: float = 0.3
    max_tokens: int = 1000
    timeout: int = 30

    @model_validator(mode="after")
    def _merge_provider_into_model(self) -> "LLMConfig":
        """Merge legacy ``provider`` field into ``model`` for litellm compat."""
        if self.provider and "/" not in self.model:
            self.model = f"{self.provider}/{self.model}"
            self.provider = ""
        return self


class BotPortConfig(BaseSettings):
    """Main BotPort configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    concerns: ConcernConfig = Field(default_factory=ConcernConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    model_config = SettingsConfigDict(
        env_prefix="BOTPORT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> BotPortConfig:
        """Load configuration from a YAML file."""
        p = Path(path).expanduser()
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def load(cls) -> BotPortConfig:
        """Load configuration with fallback chain: local -> home -> defaults."""
        # Try local config first.
        local = Path(LOCAL_CONFIG_FILENAME)
        if local.is_file():
            try:
                cfg = cls.from_yaml(local)
                log.info("Config loaded from %s", local.resolve())
                return cfg
            except Exception as exc:
                log.warning("Failed to parse local config %s: %s", local, exc)

        # Try home config.
        if DEFAULT_CONFIG_PATH.is_file():
            try:
                cfg = cls.from_yaml(DEFAULT_CONFIG_PATH)
                log.info("Config loaded from %s", DEFAULT_CONFIG_PATH)
                return cfg
            except Exception as exc:
                log.warning("Failed to parse home config %s: %s", DEFAULT_CONFIG_PATH, exc)
        else:
            log.debug("No config found at %s", DEFAULT_CONFIG_PATH)

        log.info("Using default configuration")
        return cls()

    def save(self, path: Path | str | None = None) -> None:
        """Save configuration to YAML."""
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json", exclude_none=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


# ── Global singleton ─────────────────────────────────────────────

_config: BotPortConfig | None = None


def get_config() -> BotPortConfig:
    global _config
    if _config is None:
        _config = BotPortConfig.load()
    return _config


def set_config(config: BotPortConfig) -> None:
    global _config
    _config = config
