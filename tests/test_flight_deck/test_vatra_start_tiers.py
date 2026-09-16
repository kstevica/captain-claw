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


class _RecordingDB(_FakeDB):
    def __init__(self, forge_tiers_json):
        super().__init__(forge_tiers_json)
        self.created_config = None

    async def create_basna_session(self, user_id, intent, title="", config="{}"):
        self.created_config = config
        return await super().create_basna_session(user_id, intent, title, config)


async def test_start_threads_and_persists_longform_fields(monkeypatch):
    import json as _json
    db = _RecordingDB(_TIER_SET)
    captured = {}

    async def _fake_planner(exec_req, request, user, gate=True):
        captured["exec_req"] = exec_req

    monkeypatch.setattr(vr, "get_db", lambda: db)
    monkeypatch.setattr(vr, "plan_vatra_group0", _fake_planner)
    created = []

    class _T:
        def add_done_callback(self, _cb):
            pass

    monkeypatch.setattr(vr.asyncio, "create_task", lambda c: (created.append(c), _T())[1])

    body = vr.VatraStartRequest(
        intent="Write a mystery novella", tiers={"reason": {"provider": "openai", "model": "m"}},
        deliverable={"path": "fair-measure.md", "kind": "fiction",
                     "parts": [{"path": "part-one.md", "order": 1}]},
        role_tiers={"reporter": "reason", "lead": "reason"},
        dispatch_timeout=1200.0,
        quality={"profile": "long_form"})
    req = SimpleNamespace(state=SimpleNamespace(user_id="u1"))
    await vr.start_vatra(body, req, {"id": "u1"})
    for c in created:
        await c

    exec_req = captured["exec_req"]
    assert exec_req.deliverable["path"] == "fair-measure.md"
    assert exec_req.role_tiers == {"reporter": "reason", "lead": "reason"}
    assert exec_req.dispatch_timeout == 1200.0
    # persisted into the session config so /plan/approve inherits them
    cfg = _json.loads(db.created_config)
    assert cfg["deliverable"]["kind"] == "fiction"
    assert cfg["role_tiers"]["reporter"] == "reason"
    assert cfg["quality"]["profile"] == "long_form"
    assert cfg["dispatch_timeout"] == 1200.0
