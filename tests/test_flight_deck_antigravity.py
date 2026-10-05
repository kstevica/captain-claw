"""The host subscription diagnostics never expose credentials or generate text."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import antigravity_routes as routes
from captain_claw.flight_deck.auth import get_current_user


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": "admin-test", "role": "admin"}
    return app


async def test_non_admin_cannot_check_host_account(app, monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: {"id": "viewer-test", "role": "user"}
    run = AsyncMock()
    monkeypatch.setattr(routes, "run_cli", run)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        assert (await client.get("/fd/antigravity/status")).status_code == 403
        assert (await client.post("/fd/antigravity/check")).status_code == 403
    run.assert_not_called()


async def test_check_only_reads_usage_and_models(app, monkeypatch):
    run = AsyncMock(side_effect=[json.dumps({"status": "SUCCESS", "response": "Account: test@example.org\nGemini: 100%"}).encode(),
                                b"gemini-3.8-flash-low\tGemini Flash\nclaude-opus-test\tClaude\n"])
    monkeypatch.setattr(routes, "run_cli", run)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.post("/fd/antigravity/check")
    assert response.status_code == 200
    assert response.json()["models"] == ["gemini-3.8-flash-low"]
    assert response.json()["extra_credits_enabled"] is False
    assert [call.args[0] for call in run.call_args_list] == [
        ["-p", "/usage", "--output-format", "json", "--print-timeout", "30s"], ["models"]]
    assert all("prompt" not in call.kwargs for call in run.call_args_list)


async def test_diagnostics_do_not_forward_tokens(app, monkeypatch):
    run = AsyncMock(side_effect=[b'{"status":"SUCCESS","response":"Bearer very-private"}', b""])
    monkeypatch.setattr(routes, "run_cli", run)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.post("/fd/antigravity/check")
    assert response.status_code == 400
    assert "very-private" not in response.text
