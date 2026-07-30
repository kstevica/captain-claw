"""start_vatra must default to the owner's saved Library tiers when the caller
omits them — so the Group-0 Lead decompose uses the user's configured model,
not the registry-default (anthropic) one.

This mirrors the fallback /plan/approve already had; the UI-start path had been
missing it, so an API caller (e.g. a product BFF) that omits `tiers` saw the
Lead fail with a missing-Anthropic-key error even though the user had a working
tier set configured.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from captain_claw.flight_deck import vatra_routes as vr


class _FakeDB:
    def __init__(self, forge_tiers_json: str):
        self._settings = {"fd:forge-tiers": forge_tiers_json}

    async def get_all_settings(self, owner_id):
        return self._settings

    async def create_basna_session(self, user_id, intent, title="", config="{}"):
        return {"id": "sess-1", "user_id": user_id, "intent": intent,
                "title": title, "config": config}


_TIER_SET = (
    '{"activeSetId": "s1", "sets": [{"id": "s1", "tiers": '
    '{"reason": {"provider": "openai", "model": "deepseek-v4-pro", '
    '"base_url": "https://api.deepseek.com"}}, "envVars": []}]}'
)


async def _call_start(monkeypatch, *, tiers):
    """Drive start_vatra, capturing the ExecuteRequest handed to the planner."""
    captured: dict = {}

    async def _fake_planner(exec_req, request, user, gate=True):
        captured["exec_req"] = exec_req

    monkeypatch.setattr(vr, "get_db", lambda: _FakeDB(_TIER_SET))
    monkeypatch.setattr(vr, "plan_vatra_group0", _fake_planner)
    # Don't leak a real asyncio task into the loop — run the planner inline.
    created: list = []

    class _FakeTask:  # hashable (added to a set) with a no-op done callback
        def add_done_callback(self, _cb):
            pass

    def _fake_create_task(coro):
        created.append(coro)
        return _FakeTask()

    monkeypatch.setattr(vr.asyncio, "create_task", _fake_create_task)

    body = vr.VatraStartRequest(intent="Research about Captain Claw", tiers=tiers)
    req = SimpleNamespace(state=SimpleNamespace(user_id="u1"))
    await vr.start_vatra(body, req, {"id": "u1"})
    # Execute the captured planner coroutine so `captured` fills in.
    for coro in created:
        await coro
    return captured["exec_req"]


async def test_start_falls_back_to_owner_tiers_when_omitted(monkeypatch):
    exec_req = await _call_start(monkeypatch, tiers=None)
    assert exec_req.tiers == {
        "reason": {"provider": "openai", "model": "deepseek-v4-pro",
                   "base_url": "https://api.deepseek.com"}}


async def test_start_respects_explicit_tiers(monkeypatch):
    explicit = {"reason": {"provider": "anthropic", "model": "claude-opus-4-8"}}
    exec_req = await _call_start(monkeypatch, tiers=explicit)
    # An explicit tier set is used verbatim — the fallback never overrides it.
    assert exec_req.tiers == explicit
