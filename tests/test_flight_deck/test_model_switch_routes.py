"""Models agents can switch to on a custom endpoint (model_switch_routes.py):
the admin's list, and what an agent is told for its endpoint."""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import auth as auth_mod
from captain_claw.flight_deck import model_switch_routes as routes


class FakeDB:
    def __init__(self):
        self.settings: dict[str, str] = {}

    async def get_system_setting(self, key):
        return self.settings.get(key)

    async def set_system_setting(self, key, value):
        self.settings[key] = value


@pytest.fixture
def deck(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(auth_mod, "_fd_auth_enabled", lambda: False)
    monkeypatch.setattr(routes, "get_db", lambda: db)
    app = FastAPI()
    app.include_router(routes.router)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fd.test")
    return client, db


def test_clean_endpoints_tidies_and_merges():
    rows = routes.clean_endpoints([
        {"provider": "OpenAI", "base_url": "http://gw:4000/v1/", "models": ["qwen", {"model": "kimi", "note": "smart"}]},
        {"provider": "openai", "base_url": "http://gw:4000/v1", "models": [{"model": "QWEN"}, {"model": "glm"}]},
        {"provider": "nonsense", "base_url": "", "models": ["x"]},
        {"provider": "ollama", "base_url": "http://localhost:11434", "models": []},
        {"provider": "claude", "base_url": "", "models": [" claude-opus-4-8 "]},
    ])
    assert rows == [
        {"provider": "openai", "base_url": "http://gw:4000/v1/", "models": [
            {"model": "qwen", "note": ""}, {"model": "kimi", "note": "smart"}, {"model": "glm", "note": ""}]},
        {"provider": "anthropic", "base_url": "", "models": [{"model": "claude-opus-4-8", "note": ""}]},
    ]


async def test_the_admin_saves_and_reads_the_list(deck):
    client, db = deck
    resp = await client.put("/fd/model-switch/endpoints", json={"endpoints": [
        {"provider": "openrouter", "base_url": "", "models": [{"model": "moonshotai/kimi-k3", "note": "cheap"}]},
    ]})
    assert resp.status_code == 200, resp.text
    stored = json.loads(db.settings[routes.SETTING])
    assert stored["endpoints"][0]["models"][0]["model"] == "moonshotai/kimi-k3"
    got = (await client.get("/fd/model-switch/endpoints")).json()
    assert got["endpoints"] == stored["endpoints"]


async def test_a_non_admin_can_t_touch_it(deck, monkeypatch):
    client, _db = deck
    from captain_claw.flight_deck import admin_routes

    async def not_admin():
        return {"id": "u1", "role": "user"}

    client._transport.app.dependency_overrides[admin_routes.get_current_user] = not_admin
    assert (await client.get("/fd/model-switch/endpoints")).status_code == 403
    assert (await client.put("/fd/model-switch/endpoints", json={"endpoints": []})).status_code == 403


async def test_an_agent_gets_its_endpoint_s_models_only(deck, monkeypatch):
    client, db = deck
    db.settings[routes.SETTING] = json.dumps({"endpoints": [
        {"provider": "openai", "base_url": "http://gw:4000/v1", "models": [{"model": "qwen", "note": ""}]},
        {"provider": "ollama", "base_url": "http://127.0.0.1:11434", "models": [{"model": "gemma4", "note": ""}]},
    ]})
    from captain_claw.flight_deck import google_oauth_routes, server

    monkeypatch.setattr(google_oauth_routes, "_authorize_agent_call", lambda _r: None)
    monkeypatch.setattr(server, "_find_agent_by_auth", lambda tok: (tok == "good", "", "a1"))
    params = {"provider": "ollama", "base_url": "http://localhost:11434/"}
    resp = await client.get("/fd/model-switch/agent/models", params=params, headers={"X-Agent-Auth": "good"})
    assert resp.status_code == 200 and resp.json() == {"models": [{"model": "gemma4", "note": ""}]}
    resp = await client.get("/fd/model-switch/agent/models", params={"provider": "anthropic"},
                            headers={"X-Agent-Auth": "good"})
    assert resp.json() == {"models": []}
    resp = await client.get("/fd/model-switch/agent/models", params=params, headers={"X-Agent-Auth": "bad"})
    assert resp.status_code == 403
