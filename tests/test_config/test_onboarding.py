from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from captain_claw.main import _build_runtime_arg_parser
from captain_claw.onboarding import (
    _PROVIDER_DEFAULT_MODELS,
    is_onboarding_completed,
    mark_onboarding_completed,
    save_onboarding_config,
    should_run_onboarding,
    validate_provider_connection,
)


def test_should_run_onboarding_on_fresh_install(tmp_path: Path):
    state_path = tmp_path / "onboarding_state.json"
    global_cfg = tmp_path / "home-config.yaml"

    assert should_run_onboarding(
        force=False,
        state_path=state_path,
        cwd=tmp_path,
        global_config_path=global_cfg,
    )


def test_should_not_run_onboarding_when_local_config_exists(tmp_path: Path):
    state_path = tmp_path / "onboarding_state.json"
    global_cfg = tmp_path / "home-config.yaml"
    local_cfg = tmp_path / "config.yaml"
    local_cfg.write_text("model:\n  provider: openai\n", encoding="utf-8")

    assert not should_run_onboarding(
        force=False,
        state_path=state_path,
        cwd=tmp_path,
        global_config_path=global_cfg,
    )


def test_should_not_run_onboarding_when_explicit_config_exists(tmp_path: Path):
    state_path = tmp_path / "onboarding_state.json"
    explicit_cfg = tmp_path / "custom-config.yaml"
    explicit_cfg.write_text("model:\n  provider: ollama\n", encoding="utf-8")

    assert not should_run_onboarding(
        force=False,
        state_path=state_path,
        config_path=explicit_cfg,
        cwd=tmp_path,
        global_config_path=(tmp_path / "home-config.yaml"),
    )


def test_should_run_onboarding_when_forced(tmp_path: Path):
    assert should_run_onboarding(
        force=True,
        state_path=(tmp_path / "onboarding_state.json"),
        cwd=tmp_path,
        global_config_path=(tmp_path / "home-config.yaml"),
    )


def test_mark_onboarding_completed_persists_state(tmp_path: Path):
    state_path = tmp_path / "onboarding_state.json"

    assert not is_onboarding_completed(state_path=state_path)
    mark_onboarding_completed(state_path=state_path)
    assert is_onboarding_completed(state_path=state_path)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["completed"] is True
    assert "completed_at" in payload


def test_runtime_cli_parser_supports_onboarding_flag():
    parser = _build_runtime_arg_parser()
    parsed, unknown = parser.parse_known_args(["--onboarding"])

    assert parsed.onboarding is True
    assert unknown == []


def test_provider_default_models_are_current():
    assert _PROVIDER_DEFAULT_MODELS["openai"] == "gpt-4.1-mini"
    assert _PROVIDER_DEFAULT_MODELS["anthropic"] == "claude-sonnet-4-20250514"
    assert _PROVIDER_DEFAULT_MODELS["gemini"] == "gemini-2.5-flash-preview-05-20"
    assert _PROVIDER_DEFAULT_MODELS["ollama"] == "llama3.2"


# ── validate_provider_connection ─────────────────────────


@pytest.mark.asyncio
async def test_validate_provider_connection_success():
    mock_resp = MagicMock()
    mock_resp.content = "OK"

    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(return_value=mock_resp)

    with patch("captain_claw.llm.create_provider", return_value=mock_provider):
        ok, error = await validate_provider_connection(
            provider="openai", model="gpt-4.1-mini", api_key="sk-test"
        )

    assert ok is True
    assert error is None


@pytest.mark.asyncio
async def test_validate_provider_connection_failure():
    with patch(
        "captain_claw.llm.create_provider",
        side_effect=Exception("Invalid API key"),
    ):
        ok, error = await validate_provider_connection(
            provider="openai", model="gpt-4.1-mini", api_key="bad-key"
        )

    assert ok is False
    assert "Invalid API key" in error


@pytest.mark.asyncio
async def test_validate_ollama_success():
    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        ok, error = await validate_provider_connection(
            provider="ollama", model="llama3.2", base_url="http://localhost:11434"
        )

    assert ok is True
    assert error is None


@pytest.mark.asyncio
async def test_validate_ollama_failure():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        ok, error = await validate_provider_connection(
            provider="ollama", model="llama3.2", base_url="http://localhost:11434"
        )

    assert ok is False
    assert "Cannot reach Ollama" in error


# ── save_onboarding_config ───────────────────────────────


def test_save_onboarding_config_creates_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    state_path = tmp_path / "onboarding_state.json"

    result = save_onboarding_config(
        values={
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-ant-test",
            "base_url": "",
            "enable_guards": True,
        },
        config_path=config_path,
        state_path=state_path,
    )

    assert result == config_path
    assert config_path.exists()
    assert is_onboarding_completed(state_path=state_path)

    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["model"]["provider"] == "anthropic"
    assert data["model"]["model"] == "claude-sonnet-4-20250514"
    assert data["model"]["api_key"] == "sk-ant-test"
    assert data["guards"]["input"]["enabled"] is True
    assert data["guards"]["input"]["level"] == "ask_for_approval"


def test_save_onboarding_config_with_allowed_models(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    state_path = tmp_path / "onboarding_state.json"

    result = save_onboarding_config(
        values={
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "api_key": "",
            "base_url": "",
            "enable_guards": False,
            "allowed_models": [
                {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "api_key": ""},
                {"provider": "gemini", "model": "gemini-2.5-flash-preview-05-20", "api_key": ""},
            ],
        },
        config_path=config_path,
        state_path=state_path,
    )

    assert result == config_path

    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    allowed = data["model"]["allowed"]
    assert len(allowed) >= 2
    providers = [a["provider"] for a in allowed]
    assert "anthropic" in providers
    assert "gemini" in providers


def test_save_onboarding_config_guards_disabled(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    state_path = tmp_path / "onboarding_state.json"

    save_onboarding_config(
        values={
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "api_key": "",
            "base_url": "",
            "enable_guards": False,
        },
        config_path=config_path,
        state_path=state_path,
    )

    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["guards"]["input"]["enabled"] is False
    assert data["guards"]["output"]["enabled"] is False
    assert data["guards"]["script_tool"]["enabled"] is False
