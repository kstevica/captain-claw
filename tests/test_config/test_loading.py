from pathlib import Path

import captain_claw.config as config_module
from captain_claw.config import Config


def test_load_prefers_local_config_yaml(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    home_cfg = tmp_path / "home_config.yaml"
    home_cfg.write_text("model:\n  provider: ollama\n  model: llama3.2\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", home_cfg)

    local_cfg = tmp_path / "config.yaml"
    local_cfg.write_text(
        (
            "model:\n"
            "  provider: openai\n"
            "  model: gpt-4o-mini\n"
            "  allowed:\n"
            "    - id: chatgpt-fast\n"
            "      provider: openai\n"
            "      model: gpt-4o-mini\n"
        ),
        encoding="utf-8",
    )

    cfg = Config.load()

    assert cfg.model.provider == "openai"
    assert cfg.model.model == "gpt-4o-mini"
    assert len(cfg.model.allowed) == 1
    assert cfg.model.allowed[0].id == "chatgpt-fast"


def test_load_reads_skill_search_source_url_from_yaml(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    local_cfg = tmp_path / "config.yaml"
    local_cfg.write_text(
        (
            "skills:\n"
            "  search_source_url: https://github.com/example/custom-openclaw-skills\n"
        ),
        encoding="utf-8",
    )

    cfg = Config.load()

    assert cfg.skills.search_source_url == "https://github.com/example/custom-openclaw-skills"


def test_load_falls_back_to_default_path_when_no_local(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    home_cfg = tmp_path / "home_config.yaml"
    home_cfg.write_text(
        (
            "model:\n"
            "  provider: anthropic\n"
            "  model: claude-3-5-sonnet-latest\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", home_cfg)

    cfg = Config.load()

    assert cfg.model.provider == "anthropic"
    assert cfg.model.model == "claude-3-5-sonnet-latest"


def test_workspace_defaults_to_relative_workspace_folder():
    cfg = Config()
    assert cfg.workspace.path == "./workspace"


def test_resolved_workspace_path_anchors_relative_to_runtime_base(tmp_path: Path):
    cfg = Config()
    cfg.workspace.path = "./workspace"
    resolved = cfg.resolved_workspace_path(tmp_path)
    assert resolved == (tmp_path / "workspace").resolve()


def test_load_prefers_telegram_token_from_env_var(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    local_cfg = tmp_path / "config.yaml"
    local_cfg.write_text(
        (
            "telegram:\n"
            "  enabled: true\n"
            "  bot_token: insecure-yaml-token\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAW_TELEGRAM__BOT_TOKEN", "secure-env-token")

    cfg = Config.load()

    assert cfg.telegram.bot_token == "secure-env-token"


def test_load_prefers_telegram_token_from_dotenv(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    local_cfg = tmp_path / "config.yaml"
    local_cfg.write_text(
        (
            "telegram:\n"
            "  enabled: true\n"
            "  bot_token: insecure-yaml-token\n"
        ),
        encoding="utf-8",
    )
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("CLAW_TELEGRAM__BOT_TOKEN=secure-dotenv-token\n", encoding="utf-8")

    cfg = Config.load()

    assert cfg.telegram.bot_token == "secure-dotenv-token"


def test_load_prefers_telegram_token_from_plain_dotenv_key(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    local_cfg = tmp_path / "config.yaml"
    local_cfg.write_text(
        (
            "telegram:\n"
            "  enabled: true\n"
            "  bot_token: insecure-yaml-token\n"
        ),
        encoding="utf-8",
    )
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("TELEGRAM_BOT_TOKEN=secure-plain-dotenv-token\n", encoding="utf-8")

    cfg = Config.load()

    assert cfg.telegram.bot_token == "secure-plain-dotenv-token"


def test_load_maps_legacy_flat_yaml_model_keys(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    local_cfg = tmp_path / "config.yaml"
    local_cfg.write_text(
        (
            "model:\n"
            "  provider: openai\n"
            "  model: gpt-5-mini\n"
            "openai_api_key: legacy-openai-key\n"
            "ollama_base_url: http://localhost:11434\n"
        ),
        encoding="utf-8",
    )

    cfg = Config.load()

    assert cfg.model.api_key == "legacy-openai-key"
    assert cfg.model.base_url == "http://localhost:11434"


def test_load_uses_common_openai_env_key_when_model_api_key_missing(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    local_cfg = tmp_path / "config.yaml"
    local_cfg.write_text(
        (
            "model:\n"
            "  provider: openai\n"
            "  model: gpt-5-mini\n"
            "  api_key: \"\"\n"
        ),
        encoding="utf-8",
    )
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("OPENAI_API_KEY=env-openai-key\n", encoding="utf-8")

    cfg = Config.load()

    assert cfg.model.api_key == "env-openai-key"


def test_load_prefers_brave_api_key_from_env_var(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    local_cfg = tmp_path / "config.yaml"
    local_cfg.write_text(
        (
            "tools:\n"
            "  web_search:\n"
            "    api_key: insecure-yaml-brave-key\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BRAVE_API_KEY", "secure-env-brave-key")

    cfg = Config.load()

    assert cfg.tools.web_search.api_key == "secure-env-brave-key"


def test_load_prefers_brave_api_key_from_dotenv(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    local_cfg = tmp_path / "config.yaml"
    local_cfg.write_text(
        (
            "tools:\n"
            "  web_search:\n"
            "    api_key: insecure-yaml-brave-key\n"
        ),
        encoding="utf-8",
    )
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("BRAVE_API_KEY=secure-dotenv-brave-key\n", encoding="utf-8")

    cfg = Config.load()

    assert cfg.tools.web_search.api_key == "secure-dotenv-brave-key"


def test_load_prefers_slack_bot_token_from_env_var(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    local_cfg = tmp_path / "config.yaml"
    local_cfg.write_text(
        (
            "slack:\n"
            "  enabled: true\n"
            "  bot_token: insecure-yaml-slack-token\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SLACK_BOT_TOKEN", "secure-env-slack-token")

    cfg = Config.load()

    assert cfg.slack.bot_token == "secure-env-slack-token"


def test_load_prefers_discord_bot_token_from_dotenv(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    local_cfg = tmp_path / "config.yaml"
    local_cfg.write_text(
        (
            "discord:\n"
            "  enabled: true\n"
            "  bot_token: insecure-yaml-discord-token\n"
        ),
        encoding="utf-8",
    )
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("DISCORD_BOT_TOKEN=secure-dotenv-discord-token\n", encoding="utf-8")

    cfg = Config.load()

    assert cfg.discord.bot_token == "secure-dotenv-discord-token"


def test_load_handles_legacy_python_object_apply_enum_tag(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    local_cfg = tmp_path / "config.yaml"
    local_cfg.write_text(
        (
            "tools:\n"
            "  shell:\n"
            "    default_policy: !!python/object/apply:captain_claw.config.ExecPolicy\n"
            "    - ask\n"
        ),
        encoding="utf-8",
    )

    cfg = Config.load()

    assert cfg.tools.shell.default_policy == "ask"
    assert "python/object/apply" not in local_cfg.read_text(encoding="utf-8")


def test_save_uses_safe_yaml_scalars_for_enum(tmp_path: Path):
    config_path = tmp_path / "saved-config.yaml"
    cfg = Config()
    cfg.tools.shell.default_policy = cfg.tools.shell.ExecPolicy.DENY

    cfg.save(config_path)

    raw = config_path.read_text(encoding="utf-8")
    assert "python/object/apply" not in raw
    reloaded = Config.from_yaml(config_path)
    assert reloaded.tools.shell.default_policy == "deny"
