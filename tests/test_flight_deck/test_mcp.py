"""End-to-end smoke tests for Flight Deck's MCP layer.

Covers:

* :mod:`captain_claw.flight_deck.mcp_storage` — round-trip CRUD on a
  temp file, including secret masking and atomic-write behaviour.
* :class:`captain_claw.flight_deck.mcp_manager.MCPManager` — handshake
  + tools/list + tools/call against a fake upstream server using
  :class:`httpx.MockTransport`.
* :mod:`captain_claw.flight_deck.mcp_routes` — the FastAPI routes via
  :class:`fastapi.testclient.TestClient` with auth dependency
  overridden, exercising both the admin and agent-facing surfaces.

These tests are deliberately self-contained: they avoid real network,
real disk outside ``tmp_path`` and real Flight Deck auth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import mcp_manager, mcp_routes, mcp_storage


# ── shared fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_storage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect mcp_storage to a temp file for every test."""
    target = tmp_path / "mcp_servers.json"
    monkeypatch.setenv("CAPTAIN_CLAW_FD_MCP_PATH", str(target))
    return target


@pytest.fixture
def reset_manager() -> None:
    """Tear down the global manager before/after each test."""
    if mcp_manager._manager is not None:  # noqa: SLF001
        mcp_manager._manager = None  # noqa: SLF001
    yield
    if mcp_manager._manager is not None:  # noqa: SLF001
        mcp_manager._manager = None  # noqa: SLF001


# ── fake upstream MCP server (httpx MockTransport) ──────────────────


class FakeMCPServer:
    """Tiny in-process MCP server used by manager + route tests.

    Implements just enough JSON-RPC to exercise initialize / tools/list
    / tools/call. The transport handler echoes ``Mcp-Session-Id`` back
    so the manager's session-tracking path is exercised too.
    """

    def __init__(self, tools: list[dict[str, Any]] | None = None) -> None:
        self.tools = tools or [
            {
                "name": "echo",
                "description": "Echo back the input.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                },
            }
        ]
        self.requests: list[dict[str, Any]] = []
        self.session_id = "session-abc"
        self.access_tokens_issued = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/oauth/token"):
            self.access_tokens_issued += 1
            return httpx.Response(200, json={"access_token": "tok-123", "token_type": "Bearer"})

        body = json.loads(request.content) if request.content else {}
        self.requests.append({"path": path, "body": body})
        method = body.get("method")

        if method == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "serverInfo": {"name": "fake", "version": "0"},
                    },
                },
                headers={"Mcp-Session-Id": self.session_id},
            )
        if method == "notifications/initialized":
            return httpx.Response(202, content=b"")
        if method == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {"tools": self.tools},
                },
            )
        if method == "tools/call":
            params = body.get("params") or {}
            tool = params.get("name")
            args = params.get("arguments") or {}
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"called {tool} with {json.dumps(args, sort_keys=True)}",
                            }
                        ]
                    },
                },
            )
        return httpx.Response(400, json={"error": f"unknown method {method}"})


# ── storage tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_storage_round_trip_with_masking(_isolate_storage: Path) -> None:
    assert mcp_storage.load_servers() == []

    record = await mcp_storage.upsert_server(
        {
            "name": "fricmcp",
            "url": "https://example.com/mcp",
            "client_id": "abc",
            "client_secret": "supersecret",
            "token_endpoint": "/oauth/token",
            "headers": {"X-Custom": "hi"},
        }
    )
    assert record["client_secret"] == "supersecret"
    assert record["added_at"] > 0

    on_disk = json.loads(_isolate_storage.read_text())
    assert len(on_disk) == 1
    assert on_disk[0]["client_secret"] == "supersecret"

    public = mcp_storage.list_servers_public()
    assert public[0]["client_secret"] != "supersecret"
    assert public[0]["client_secret_set"] is True

    # Update preserves added_at on re-insert
    earlier_added = record["added_at"]
    updated = await mcp_storage.upsert_server(
        {
            "name": "fricmcp",
            "url": "https://example.com/mcp/v2",
            "client_id": "abc",
            "client_secret": "newsecret",
        }
    )
    assert updated["url"] == "https://example.com/mcp/v2"
    assert updated["added_at"] == earlier_added

    # Delete returns True only when something was removed
    assert await mcp_storage.delete_server("fricmcp") is True
    assert await mcp_storage.delete_server("fricmcp") is False
    assert mcp_storage.load_servers() == []


@pytest.mark.asyncio
async def test_storage_rejects_records_missing_required_fields(
    _isolate_storage: Path,
) -> None:
    with pytest.raises(ValueError):
        await mcp_storage.upsert_server({"name": "", "url": "https://x"})
    with pytest.raises(ValueError):
        await mcp_storage.upsert_server({"name": "x", "url": ""})


# ── manager tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_manager_initializes_and_caches_tools_list(
    _isolate_storage: Path, reset_manager: None
) -> None:
    fake = FakeMCPServer()
    await mcp_storage.upsert_server(
        {
            "name": "fake",
            "url": "https://upstream.example/mcp",
            "client_id": "id",
            "client_secret": "sec",
            "token_endpoint": "/oauth/token",
        }
    )
    manager = mcp_manager.get_manager()
    manager._client = httpx.AsyncClient(transport=fake.transport())  # noqa: SLF001

    tools_a = await manager.list_tools("fake")
    tools_b = await manager.list_tools("fake")
    assert tools_a == tools_b
    assert tools_a[0]["name"] == "echo"

    # Initialise + token endpoint hit exactly once. Tools/list cached
    # so the second call doesn't re-issue an upstream request.
    rpc_calls = [r for r in fake.requests if r["body"].get("method") in {"initialize", "tools/list"}]
    assert sum(1 for r in rpc_calls if r["body"]["method"] == "initialize") == 1
    assert sum(1 for r in rpc_calls if r["body"]["method"] == "tools/list") == 1
    assert fake.access_tokens_issued == 1

    # Force-refresh skips the cache.
    await manager.list_tools("fake", force_refresh=True)
    assert sum(1 for r in fake.requests if r["body"].get("method") == "tools/list") == 2

    await manager.close()


@pytest.mark.asyncio
async def test_manager_call_tool_returns_result_and_tracks_session(
    _isolate_storage: Path, reset_manager: None
) -> None:
    fake = FakeMCPServer()
    await mcp_storage.upsert_server(
        {"name": "fake", "url": "https://upstream.example/mcp"}
    )
    manager = mcp_manager.get_manager()
    manager._client = httpx.AsyncClient(transport=fake.transport())  # noqa: SLF001

    result = await manager.call_tool("fake", "echo", {"message": "hi"})
    assert "called echo" in result["content"][0]["text"]

    state = manager._states["fake"]  # noqa: SLF001
    assert state.session_id == fake.session_id
    await manager.close()


@pytest.mark.asyncio
async def test_manager_probe_record_does_not_persist_state(
    _isolate_storage: Path, reset_manager: None
) -> None:
    fake = FakeMCPServer()
    manager = mcp_manager.get_manager()
    manager._client = httpx.AsyncClient(transport=fake.transport())  # noqa: SLF001

    record = {
        "name": "transient",
        "url": "https://upstream.example/mcp",
        "client_id": "",
        "client_secret": "",
        "token_endpoint": "",
        "headers": {},
        "enabled": True,
    }
    result = await manager.probe_record(record)
    assert result["ok"] is True
    assert result["tools_count"] == 1
    assert "transient" not in manager._states  # noqa: SLF001
    assert mcp_storage.load_servers() == []
    await manager.close()


@pytest.mark.asyncio
async def test_manager_test_server_reports_failure(
    _isolate_storage: Path, reset_manager: None
) -> None:
    failing_transport = httpx.MockTransport(
        lambda request: httpx.Response(500, json={"error": "boom"})
    )
    await mcp_storage.upsert_server(
        {"name": "broken", "url": "https://upstream.example/mcp"}
    )
    manager = mcp_manager.get_manager()
    manager._client = httpx.AsyncClient(transport=failing_transport)  # noqa: SLF001

    result = await manager.test_server("broken")
    assert result["ok"] is False
    assert result.get("status_code") == 500
    assert "error" in result
    await manager.close()


# ── route tests (FastAPI) ───────────────────────────────────────────


def _make_app(monkeypatch: pytest.MonkeyPatch, fake: FakeMCPServer) -> FastAPI:
    """Build a FastAPI app with the mcp router and auth bypass."""
    app = FastAPI()
    app.include_router(mcp_routes.router)
    # Override the auth dependency so admin endpoints don't require a real user.
    from captain_claw.flight_deck.auth import get_current_user

    app.dependency_overrides[get_current_user] = lambda: {"username": "test"}

    # Replace the global manager so its httpx client uses our mock.
    if mcp_manager._manager is not None:  # noqa: SLF001
        mcp_manager._manager = None  # noqa: SLF001
    manager = mcp_manager.get_manager()
    manager._client = httpx.AsyncClient(transport=fake.transport())  # noqa: SLF001
    return app


def test_routes_full_admin_flow(
    monkeypatch: pytest.MonkeyPatch, _isolate_storage: Path
) -> None:
    fake = FakeMCPServer()
    app = _make_app(monkeypatch, fake)
    client = TestClient(app)

    # Initially empty.
    resp = client.get("/fd/mcp/servers")
    assert resp.status_code == 200
    assert resp.json() == {"servers": []}

    # Add a server.
    payload = {
        "name": "fake",
        "url": "https://upstream.example/mcp",
        "client_id": "id",
        "client_secret": "sec",
        "token_endpoint": "/oauth/token",
    }
    resp = client.post("/fd/mcp/servers", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["server"]["name"] == "fake"
    assert body["server"]["client_secret"] != "sec"  # masked in public view

    # List shows it.
    resp = client.get("/fd/mcp/servers")
    assert len(resp.json()["servers"]) == 1

    # Test endpoint runs initialize + tools/list.
    resp = client.post("/fd/mcp/servers/fake/test")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["tools_count"] == 1

    # Re-saving with empty client_secret preserves the stored one.
    resp = client.post(
        "/fd/mcp/servers",
        json={
            "name": "fake",
            "url": "https://upstream.example/mcp",
            "client_id": "id",
            "client_secret": "",
            "token_endpoint": "/oauth/token",
        },
    )
    assert resp.status_code == 200
    on_disk = mcp_storage.get_server("fake")
    assert on_disk is not None and on_disk["client_secret"] == "sec"

    # Delete.
    resp = client.delete("/fd/mcp/servers/fake")
    assert resp.status_code == 200
    assert mcp_storage.get_server("fake") is None

    # 404 on missing server.
    resp = client.delete("/fd/mcp/servers/fake")
    assert resp.status_code == 404


def test_routes_agent_facing_endpoints(
    monkeypatch: pytest.MonkeyPatch, _isolate_storage: Path
) -> None:
    fake = FakeMCPServer()
    app = _make_app(monkeypatch, fake)
    client = TestClient(app)

    # Agent endpoints accept either loopback OR matching X-Agent-Secret.
    # TestClient doesn't show as loopback, so set the secret instead.
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "shh")
    agent_headers = {"X-Agent-Secret": "shh"}

    client.post(
        "/fd/mcp/servers",
        json={
            "name": "fake",
            "url": "https://upstream.example/mcp",
            "enabled": True,
        },
    )

    # Agent discovery.
    resp = client.get("/fd/mcp/agent/servers", headers=agent_headers)
    assert resp.status_code == 200
    assert resp.json() == {"servers": [{"name": "fake"}]}

    # Missing/wrong secret -> 401
    resp = client.get("/fd/mcp/agent/servers")
    assert resp.status_code == 401
    resp = client.get("/fd/mcp/agent/servers", headers={"X-Agent-Secret": "nope"})
    assert resp.status_code == 401

    # Tools list passthrough.
    resp = client.get("/fd/mcp/fake/tools", headers=agent_headers)
    assert resp.status_code == 200
    assert resp.json()["tools"][0]["name"] == "echo"

    # Tool call passthrough.
    resp = client.post(
        "/fd/mcp/fake/call",
        headers=agent_headers,
        json={"tool": "echo", "arguments": {"message": "hi"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["server"] == "fake"
    assert "called echo" in body["result"]["content"][0]["text"]

    # 404 path for unknown server.
    resp = client.get("/fd/mcp/nope/tools", headers=agent_headers)
    assert resp.status_code == 404


def test_routes_probe_does_not_persist(
    monkeypatch: pytest.MonkeyPatch, _isolate_storage: Path
) -> None:
    fake = FakeMCPServer()
    app = _make_app(monkeypatch, fake)
    client = TestClient(app)

    resp = client.post(
        "/fd/mcp/probe",
        json={
            "name": "probe-only",
            "url": "https://upstream.example/mcp",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    # Nothing was persisted.
    assert mcp_storage.load_servers() == []
