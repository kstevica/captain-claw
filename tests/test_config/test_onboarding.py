from __future__ import annotations

import json
from pathlib import Path

from captain_claw.main import _build_runtime_arg_parser
from captain_claw.onboarding import (
    _PROVIDER_DEFAULT_MODELS,
    is_onboarding_completed,
    mark_onboarding_completed,
    should_run_onboarding,
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


def test_onboarding_openai_default_model_is_gpt5mini():
    assert _PROVIDER_DEFAULT_MODELS["openai"] == "gpt-5-mini"
