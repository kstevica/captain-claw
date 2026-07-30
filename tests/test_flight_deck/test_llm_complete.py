"""POST /fd/llm/complete — one completion on the owner's configured tier.

Resolves the owner's saved Library tier server-side (so a caller need not, and
must not, handle raw creds) and runs a single LLM call. Used for structured
one-shot generations where a multi-agent run would be the wrong tool.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import llm_routes
from captain_claw.flight_deck.auth import get_current_user


class _FakeDB:
    def __init__(self, forge_tiers_json: str = ""):
        self._settings = ({"fd:forge-tiers": forge_tiers_json}
                          if forge_tiers_json else {})

    async def get_all_settings(self, owner_id):
        return self._settings


_TIER_SET = (
    '{"activeSetId": "s1", "sets": [{"id": "s1", "tiers": '
    '{"reason": {"provider": "openai", "model": "deepseek-v4-pro", '
    '"base_url": "https://api.deepseek.com", "api_key": "sk-x"}}, '
    '"envVars": []}]}'
)


def _app(monkeypatch, db, provider_factory=None):
    monkeypatch.setattr(llm_routes, "get_db", lambda: db)
    app = FastAPI()
    app.include_router(llm_routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": "u1"}
    return app


def test_complete_uses_owner_reason_tier(monkeypatch):
    captured: dict = {}

    class _Resp:
        content = "```json\n{\"ok\": true}\n```"

    class _Provider:
        async def complete(self, messages, temperature, max_tokens):
            captured["messages"] = messages
            return _Resp()

    def _create_provider(**kwargs):
        captured["creds"] = kwargs
        return _Provider()

    import captain_claw.llm as llm
    monkeypatch.setattr(llm, "create_provider", _create_provider)

    app = _app(monkeypatch, _FakeDB(_TIER_SET))
    client = TestClient(app)
    r = client.post("/fd/llm/complete",
                    json={"prompt": "make a manifest", "system": "you configure",
                          "tier": "reason"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "deepseek-v4-pro"
    assert body["provider"] == "openai"
    assert "ok" in body["content"]
    # It resolved the owner's saved reason tier — not a hardcoded default.
    assert captured["creds"]["provider"] == "openai"
    assert captured["creds"]["model"] == "deepseek-v4-pro"
    assert captured["creds"]["base_url"] == "https://api.deepseek.com"
    # system + user messages, in order.
    roles = [m.role for m in captured["messages"]]
    assert roles == ["system", "user"]


def test_complete_400_when_tier_has_no_model(monkeypatch):
    # No saved tiers and (by monkeypatch) an empty registry tier → no model.
    from captain_claw.flight_deck import basna_routes
    monkeypatch.setattr(basna_routes, "_load_registry",
                        lambda: {"tiers": {}}, raising=False)
    app = _app(monkeypatch, _FakeDB(""))
    client = TestClient(app)
    r = client.post("/fd/llm/complete", json={"prompt": "x", "tier": "reason"})
    assert r.status_code == 400
    assert "no model" in r.json()["detail"]


def test_complete_502_surfaces_llm_error(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("missing Anthropic API key")

    import captain_claw.llm as llm
    monkeypatch.setattr(llm, "create_provider", _boom)
    app = _app(monkeypatch, _FakeDB(_TIER_SET))
    client = TestClient(app)
    r = client.post("/fd/llm/complete", json={"prompt": "x", "tier": "reason"})
    assert r.status_code == 502
    assert "missing Anthropic API key" in r.json()["detail"]
