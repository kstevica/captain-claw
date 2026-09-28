"""file_tree_builder's gws calls follow the gws tool's identity rules.

The gdrive tree (injected into the system prompt) and the agent's folder
picker run the ``gws`` CLI directly. Under Flight Deck they must act as the
agent OWNER's Google account — the owner token injected, ambient operator
credentials scrubbed — and fail closed (return the "not connected" text,
spawn nothing) when FD has no token for the owner. Standalone they inherit
the process env as before.

No network, no gws binary: the subprocess and the token source are faked.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

import captain_claw.session as session_mod
from captain_claw import file_tree_builder as ftb
from captain_claw.google_oauth import GoogleOAuthTokens
from captain_claw.google_oauth_manager import GoogleOAuthManager
from captain_claw.tools._gws_runtime import FD_CONNECT_HINT


class _Proc:
    def __init__(self, out: bytes):
        self._out = out
        self.returncode = 0

    async def communicate(self):
        return self._out, b""

    def kill(self):  # pragma: no cover - timeout path not exercised
        pass


class _Spawner:
    """Stands in for asyncio.create_subprocess_exec; answers gws drive calls."""

    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, *cmd, **kwargs):
        self.calls.append({"cmd": list(cmd), **kwargs})
        if "drives" in cmd:
            out = {"drives": [{"id": "sd1", "name": "Team"}]}
        else:
            params = json.loads(cmd[cmd.index("--params") + 1])
            if params["q"].startswith("'root'"):
                out = {"files": [
                    {"id": "f1", "name": "Sub", "mimeType": "application/vnd.google-apps.folder"},
                    {"id": "d1", "name": "doc.txt", "mimeType": "text/plain", "size": "10"},
                ]}
            else:
                out = {"files": [{"id": "d2", "name": "inner.txt", "mimeType": "text/plain"}]}
        return _Proc(json.dumps(out).encode())


@pytest.fixture
def spawner(monkeypatch):
    sp = _Spawner()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", sp)
    monkeypatch.setattr(ftb, "resolve_gws_binary", lambda: "/fake/bin/gws")
    monkeypatch.setattr(session_mod, "get_session_manager", lambda: object())
    return sp


def _set_mode(monkeypatch, *, fd: bool, token: str | None = None) -> dict:
    calls = {"get_tokens": 0}
    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_base",
                        staticmethod(lambda: "http://localhost:25080" if fd else ""))

    async def _get_tokens(self):
        calls["get_tokens"] += 1
        if token is None:
            return None
        return GoogleOAuthTokens(access_token=token, refresh_token="",
                                 expires_at=time.time() + 3300, scope="")

    monkeypatch.setattr(GoogleOAuthManager, "get_tokens", _get_tokens)
    return calls


@pytest.fixture
def ambient(monkeypatch):
    """Credentials an operator exported deck-wide — every tenant inherited them."""
    monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_TOKEN", "OPERATOR-ACCESS")
    monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE", "/host/creds.json")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/host/adc.json")


async def test_tree_runs_as_the_owner_and_resolves_the_token_once(monkeypatch, spawner, ambient):
    calls = _set_mode(monkeypatch, fd=True, token="OWNER-ACCESS")
    tree, count = await ftb.build_gdrive_tree("root", "My Drive", max_depth=2)
    assert count == 3 and "inner.txt" in tree
    assert len(spawner.calls) == 2          # root + the subfolder
    assert calls["get_tokens"] == 1         # one FD round-trip per tree, not per folder
    for call in spawner.calls:
        env = call["env"]
        assert env["GOOGLE_WORKSPACE_CLI_TOKEN"] == "OWNER-ACCESS"
        assert "GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE" not in env
        assert "GOOGLE_APPLICATION_CREDENTIALS" not in env


async def test_tree_without_an_owner_token_fails_closed(monkeypatch, spawner, ambient):
    _set_mode(monkeypatch, fd=True, token=None)
    tree, count = await ftb.build_gdrive_tree("root", "My Drive")
    assert count == 0
    assert "My Drive" in tree and FD_CONNECT_HINT in tree
    assert spawner.calls == []              # never falls back to the operator's account


async def test_folder_picker_runs_as_the_owner(monkeypatch, spawner, ambient):
    _set_mode(monkeypatch, fd=True, token="OWNER-ACCESS")
    res = await ftb.browse_gdrive_folders("root")
    assert res["shared_drives"] == [{"id": "sd1", "name": "Team"}]
    assert len(spawner.calls) == 2          # folders + shared drives
    assert all(c["env"]["GOOGLE_WORKSPACE_CLI_TOKEN"] == "OWNER-ACCESS" for c in spawner.calls)


async def test_folder_picker_without_an_owner_token_fails_closed(monkeypatch, spawner, ambient):
    _set_mode(monkeypatch, fd=True, token=None)
    res = await ftb.browse_gdrive_folders("root")
    assert res["folders"] == [] and FD_CONNECT_HINT in res["error"]
    assert spawner.calls == []


async def test_run_gws_resolves_the_identity_itself_when_not_given(monkeypatch, spawner, ambient):
    _set_mode(monkeypatch, fd=True, token=None)
    out = await ftb._run_gws("/fake/bin/gws", ["drive", "drives", "list", "--params", "{}"])
    assert isinstance(out, str) and FD_CONNECT_HINT in out
    assert spawner.calls == []


async def test_standalone_inherits_the_process_env(monkeypatch, spawner):
    calls = _set_mode(monkeypatch, fd=False)
    tree, count = await ftb.build_gdrive_tree("root", "My Drive", max_depth=1)
    assert count == 2
    assert all(c["env"] is None for c in spawner.calls)
    assert calls["get_tokens"] == 0


async def test_any_identity_failure_fails_closed(monkeypatch, spawner, ambient):
    _set_mode(monkeypatch, fd=True, token="OWNER-ACCESS")

    async def _boom(self):
        raise RuntimeError("Flight Deck unreachable")

    monkeypatch.setattr(GoogleOAuthManager, "get_tokens", _boom)
    tree, count = await ftb.build_gdrive_tree("root", "My Drive")
    assert count == 0 and "Flight Deck unreachable" in tree
    res = await ftb.browse_gdrive_folders("root")
    assert "Flight Deck unreachable" in res["error"]
    assert spawner.calls == []
