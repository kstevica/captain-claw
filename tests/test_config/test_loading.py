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
