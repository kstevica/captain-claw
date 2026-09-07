"""Per-instance isolation of Flight Deck home-anchored state.

Several Flight Decks on one host must not share their code-app state
(``apps`` / ``app_data`` / ``app_files`` / ``app_manifests``) or their
``agent_secret``. All of these resolve through :func:`fd_home.fd_home`
with the precedence CAPTAIN_CLAW_FD_HOME > FD_DATA_DIR > legacy.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from captain_claw.flight_deck import (
    agent_secret,
    app_entities,
    app_files,
    app_manifests,
    app_runtime,
)
from captain_claw.flight_deck.fd_home import fd_home


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("CAPTAIN_CLAW_FD_HOME", "FD_DATA_DIR"):
        monkeypatch.delenv(k, raising=False)


def test_fd_home_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    home = tmp_path / "home"
    data = tmp_path / "data"

    # FD_HOME wins over FD_DATA_DIR and legacy.
    _clear(monkeypatch)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(home))
    monkeypatch.setenv("FD_DATA_DIR", str(data))
    assert fd_home(legacy) == home.resolve()

    # FD_DATA_DIR when FD_HOME unset.
    monkeypatch.delenv("CAPTAIN_CLAW_FD_HOME", raising=False)
    assert fd_home(legacy) == data.resolve()

    # Legacy fallback when neither is set.
    monkeypatch.delenv("FD_DATA_DIR", raising=False)
    assert fd_home(legacy) == legacy


def test_app_state_dirs_follow_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """apps / app_data / app_files / app_manifests land under FD_DATA_DIR."""
    _clear(monkeypatch)
    data = tmp_path / "deckA"
    monkeypatch.setenv("FD_DATA_DIR", str(data))

    assert app_runtime.apps_root() == data.resolve() / "apps"
    assert app_files._default_base() == data.resolve() / "app_files"
    assert app_manifests._manifests_dir() == data.resolve() / "app_manifests"
    assert app_entities._default_base() == data.resolve() / "app_data"


def test_two_decks_get_separate_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole point: deck A and deck B never resolve to the same dir."""
    a, b = tmp_path / "A", tmp_path / "B"
    _clear(monkeypatch)

    monkeypatch.setenv("FD_DATA_DIR", str(a))
    a_apps = app_runtime.apps_root()
    a_secret = agent_secret._secret_path()

    monkeypatch.setenv("FD_DATA_DIR", str(b))
    b_apps = app_runtime.apps_root()
    b_secret = agent_secret._secret_path()

    assert a_apps != b_apps
    assert a_secret != b_secret
    assert a_secret == a.resolve() / "agent_secret"
    assert b_secret == b.resolve() / "agent_secret"


def test_agent_secret_legacy_uses_passwd_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no isolation env, the secret stays anchored to the passwd home
    (not $HOME) so a sandboxed-HOME agent still converges with FD."""
    _clear(monkeypatch)
    monkeypatch.setenv("HOME", "/tmp/some-sandbox-home")  # noqa: S108
    expected = agent_secret._real_user_home() / ".captain-claw-fd" / "agent_secret"
    assert agent_secret._secret_path() == expected
