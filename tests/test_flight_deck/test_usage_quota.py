"""Daily plan allowances — ``usage_quota``: messages and LLM tokens per user
per UTC day, the pro → premium rename, the routes, and the places that refuse a
turn (chat proxy frames, scheduled tasks) or report tokens (the agent side)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import rate_limiter, usage_quota
from captain_claw.flight_deck.auth import get_current_user, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

LOOPBACK = ("127.0.0.1", 40001)


@pytest.fixture(autouse=True)
def _auth_on(monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setattr(usage_quota, "STANDALONE_MODE", False)


@pytest.fixture
async def fd_db(tmp_path: Path):
    from captain_claw.flight_deck import auth as fd_auth
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


async def _user(db, *, role="user", **meta) -> dict:
    return await db.create_user(f"{role}-{uuid.uuid4().hex[:8]}@example.com", "x",
                                role=role, metadata=meta)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LOOPBACK),
                             base_url="http://fd.test")


def _app(user: dict) -> FastAPI:
    app = FastAPI()
    app.include_router(usage_quota.router)
    app.dependency_overrides[get_current_user] = lambda: user
    return app


# ── Plans ────────────────────────────────────────────────────────────


def test_tiers_are_free_premium_enterprise_with_daily_limits():
    assert list(rate_limiter.DEFAULT_PLAN_LIMITS) == ["free", "premium", "enterprise"]
    for limits in rate_limiter.DEFAULT_PLAN_LIMITS.values():
        assert {"messages_per_day", "tokens_per_day"} <= set(limits)
    assert rate_limiter.DEFAULT_PLAN_LIMITS["enterprise"]["messages_per_day"] == 0  # unlimited


def test_pro_reads_as_premium():
    assert rate_limiter.canonical_plan("pro") == "premium"
    assert rate_limiter.canonical_plan("Premium") == "premium"
    assert rate_limiter.get_user_plan({"metadata": json.dumps({"plan": "pro"})}) == "premium"
    assert rate_limiter.get_user_plan({"metadata": json.dumps({"plan": "nope"})}) == "free"
    assert rate_limiter.get_user_plan({"metadata": None}) == "free"


def test_stored_pro_limits_load_into_premium(monkeypatch):
    plans = {k: {**v} for k, v in rate_limiter.DEFAULT_PLAN_LIMITS.items()}
    monkeypatch.setattr(rate_limiter, "PLAN_LIMITS", plans)
    rate_limiter.load_plan_limits_from_db_sync(json.dumps({"pro": {"max_agents": 7}}))
    assert plans["premium"]["max_agents"] == 7
    assert "pro" not in plans


async def test_pro_users_migrate_to_premium(tmp_path):
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    u = await db.create_user("old@example.com", "x", metadata={"plan": "pro", "kiosk": True})
    await db.close()
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    try:
        meta = json.loads((await db.get_user_by_id(u["id"]))["metadata"])
        assert meta == {"plan": "premium", "kiosk": True}
    finally:
        await db.close()


# ── Allowances ───────────────────────────────────────────────────────


async def test_messages_count_until_the_limit(fd_db):
    u = await _user(fd_db, messages_per_day=2, tokens_per_day=0)
    ok, st = await usage_quota.admit_message(u["id"], "hello")
    assert ok and st["messages"] == {"used": 1, "limit": 2, "remaining": 1}
    assert st["warning"] is None  # 50%
    ok, st = await usage_quota.admit_message(u["id"], "again")
    assert ok and st["messages"]["used"] == 2
    ok, st = await usage_quota.admit_message(u["id"], "one more")
    assert not ok and st["blocked"] == "messages"
    assert (await fd_db.get_daily_usage(u["id"], usage_quota.today()))["messages"] == 2
    text = usage_quota.limit_text(st)
    assert "2 messages" in text and "Free plan" in text and "midnight UTC" in text


async def test_free_commands_are_neither_counted_nor_refused(fd_db):
    u = await _user(fd_db, messages_per_day=1)
    assert (await usage_quota.admit_message(u["id"], "/new")) == (True, None)
    assert (await usage_quota.admit_message(u["id"], "/session switch x")) == (True, None)
    ok, _ = await usage_quota.admit_message(u["id"], "hi")
    assert ok
    assert (await usage_quota.admit_message(u["id"], "/clear")) == (True, None)
    # "/new <question>" starts a chat with that question: it counts.
    ok, st = await usage_quota.admit_message(u["id"], "/new what's the weather")
    assert not ok and st["blocked"] == "messages"
    # Commands that run a model count like messages.
    ok, _ = await usage_quota.admit_message(u["id"], "/orchestrate build it")
    assert not ok


async def test_tokens_block_new_messages_and_automation(fd_db):
    u = await _user(fd_db, messages_per_day=0, tokens_per_day=1000)
    await usage_quota.record_tokens(fd_db, u["id"], 850)
    st = await usage_quota.status(fd_db, u["id"])
    assert st["warning"] == "tokens" and st["blocked"] is None
    assert await usage_quota.automation_blocked(u["id"]) is None
    await usage_quota.record_tokens(fd_db, u["id"], 200)
    ok, st = await usage_quota.admit_message(u["id"], "hello")
    assert not ok and st["blocked"] == "tokens"
    assert "AI tokens" in usage_quota.limit_text(st)
    assert (await usage_quota.automation_blocked(u["id"]))["blocked"] == "tokens"


async def test_one_report_is_capped(fd_db):
    u = await _user(fd_db)
    await usage_quota.record_tokens(fd_db, u["id"], 10**12)
    used = (await fd_db.get_daily_usage(u["id"], usage_quota.today()))["tokens"]
    assert used == usage_quota._MAX_TOKENS_PER_REPORT


async def test_admins_are_counted_but_never_refused(fd_db):
    a = await _user(fd_db, role="admin", messages_per_day=1)
    for _ in range(3):
        ok, st = await usage_quota.admit_message(a["id"], "hi")
        assert ok and st["exempt"] and st["blocked"] is None
    assert (await fd_db.get_daily_usage(a["id"], usage_quota.today()))["messages"] == 3


async def test_no_signin_deck_is_exempt(fd_db, monkeypatch):
    u = await _user(fd_db, messages_per_day=1)
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    for _ in range(3):
        ok, _ = await usage_quota.admit_message(u["id"], "hi")
        assert ok


async def test_unknown_user_or_broken_bookkeeping_lets_messages_through(fd_db, monkeypatch):
    assert (await usage_quota.admit_message("", "hi")) == (True, None)
    assert (await usage_quota.admit_message("ghost", "hi")) == (True, None)
    u = await _user(fd_db, messages_per_day=1)

    async def boom(*_a, **_k):
        raise RuntimeError("db down")
    monkeypatch.setattr(fd_db, "get_daily_usage", boom)
    assert (await usage_quota.admit_message(u["id"], "hi")) == (True, None)


async def test_screen_chat_frame(fd_db):
    u = await _user(fd_db, messages_per_day=1)
    # Not a chat, not JSON, automated: forwarded untouched, uncounted.
    for raw in ('{"type":"cancel"}', "not json",
                '{"type":"chat","content":"x","automation":{"kind":"peer"}}'):
        assert (await usage_quota.screen_chat_frame(u["id"], raw)) == (True, None)
    ok, reply = await usage_quota.screen_chat_frame(u["id"], '{"type":"chat","content":"hi"}')
    assert ok and reply["type"] == "fd_quota" and reply["quota"]["messages"]["used"] == 1
    ok, reply = await usage_quota.screen_chat_frame(u["id"], '{"type":"chat","content":"hi"}')
    assert not ok and reply["type"] == "error" and reply["code"] == "quota_exceeded"
    assert reply["quota"]["blocked"] == "messages" and reply["message"]
    # No signed-in speaker (sign-in off): never screened.
    assert (await usage_quota.screen_chat_frame("", '{"type":"chat","content":"hi"}')) == (True, None)


# ── Routes ───────────────────────────────────────────────────────────


async def test_my_quota_route(fd_db):
    u = await _user(fd_db, plan="premium")
    await usage_quota.admit_message(u["id"], "hi")
    async with _client(_app(u)) as c:
        r = await c.get("/fd/quota")
    assert r.status_code == 200
    body = r.json()
    assert body["plan"] == "premium" and body["plan_label"] == "Premium"
    assert body["messages"]["used"] == 1
    assert body["messages"]["limit"] == rate_limiter.PLAN_LIMITS["premium"]["messages_per_day"]


async def test_admin_sees_and_resets_today(fd_db):
    admin = await _user(fd_db, role="admin")
    u = await _user(fd_db, messages_per_day=1)
    await usage_quota.admit_message(u["id"], "hi")
    async with _client(_app(admin)) as c:
        r = await c.get("/fd/admin/usage/today")
        assert r.json()["usage"][u["id"]] == {"messages": 1, "tokens": 0, "byok_tokens": 0}
        r = await c.delete(f"/fd/admin/users/{u['id']}/usage/today")
        assert r.status_code == 200
        assert (await c.delete("/fd/admin/users/nobody/usage/today")).status_code == 404
    ok, _ = await usage_quota.admit_message(u["id"], "hi again")
    assert ok
    async with _client(_app(u)) as c:
        assert (await c.get("/fd/admin/usage/today")).status_code == 403
        assert (await c.delete(f"/fd/admin/users/{u['id']}/usage/today")).status_code == 403


async def test_agent_usage_bills_owner_or_member(fd_db, monkeypatch):
    from captain_claw.flight_deck import deep_memory_routes, speaker_grants

    owner = await _user(fd_db)
    member = await _user(fd_db, messages_per_day=5)
    acting: list = [None]

    async def fake_acting(_request, **_k):
        return acting[0]
    monkeypatch.setattr(speaker_grants, "acting_member", fake_acting)
    monkeypatch.setattr(deep_memory_routes, "_agent_owner", lambda _r: owner["id"])
    app = FastAPI()
    app.include_router(usage_quota.router)
    async with _client(app) as c:
        r = await c.post("/fd/quota/agent/usage", json={"tokens": 120})
        assert r.status_code == 200, r.text
        acting[0] = speaker_grants.ActingMember(
            user_id=member["id"], owner=owner["id"], agent_ref="process:a", lane="A",
            turn="t", slug="a", name="A")
        r = await c.post("/fd/quota/agent/usage", json={"tokens": 30})
        assert r.status_code == 200
        assert (await c.post("/fd/quota/agent/usage", json={"tokens": -1})).status_code == 422
    day = usage_quota.today()
    assert (await fd_db.get_daily_usage(owner["id"], day))["tokens"] == 120
    assert (await fd_db.get_daily_usage(member["id"], day))["tokens"] == 30


def test_agent_usage_route_is_grant_aware():
    assert usage_quota.AGENT_USAGE_PATH in speaker_grants_paths()


def speaker_grants_paths():
    from captain_claw.flight_deck.speaker_grants import GRANT_AWARE_PATHS
    return GRANT_AWARE_PATHS


# ── Free resets ──────────────────────────────────────────────────────


async def test_grants_add_up_to_the_cap_and_never_take_away(fd_db):
    a, b = await _user(fd_db), await _user(fd_db)
    admin = await _user(fd_db, role="admin")
    assert await fd_db.grant_usage_resets([a["id"]], 2, 3) == {a["id"]: 2}
    assert await fd_db.grant_usage_resets([a["id"], b["id"]], 2, 3) == {a["id"]: 3, b["id"]: 2}
    # Over the cap already (the cap was lowered): a grant keeps what they hold.
    await fd_db.set_usage_resets(b["id"], 5)
    assert (await fd_db.grant_usage_resets(None, 1, 3))[b["id"]] == 5
    # Everyone: admins are skipped, whether named or not.
    assert admin["id"] not in await fd_db.grant_usage_resets(None, 1, 3)
    assert await fd_db.grant_usage_resets([admin["id"]], 1, 3) == {}
    assert await fd_db.list_usage_resets() == {a["id"]: 3, b["id"]: 5}


async def test_spending_a_reset_starts_today_over(fd_db):
    u = await _user(fd_db, messages_per_day=1, tokens_per_day=0)
    await usage_quota.admit_message(u["id"], "hi")
    ok, st = await usage_quota.admit_message(u["id"], "more")
    assert not ok and st["resets_left"] == 0
    assert "Ask your admin" in usage_quota.limit_text(st)
    await fd_db.grant_usage_resets([u["id"]], 1, 3)
    st = await usage_quota.status(fd_db, u["id"])
    assert st["resets_left"] == 1 and st["blocked"] == "messages"
    assert "1 free reset:" in usage_quota.limit_text(st)
    assert "1 free reset in the web chat" in usage_quota.short_limit_text(st).lower()
    async with _client(_app(u)) as c:
        r = await c.post("/fd/quota/reset")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["blocked"] is None and body["resets_left"] == 0
        assert body["messages"]["used"] == 0
        # None left.
        await usage_quota.admit_message(u["id"], "again")
        assert (await c.post("/fd/quota/reset")).status_code == 409
    ok, _ = await usage_quota.admit_message(u["id"], "and again")
    assert not ok
    logs = await fd_db.get_usage_logs(user_id=u["id"], event_type="usage_reset_used")
    assert len(logs) == 1 and json.loads(logs[0]["detail"])["messages"] == 1


async def test_a_reset_is_not_spent_on_an_unused_day_or_by_admins(fd_db):
    u = await _user(fd_db)
    admin = await _user(fd_db, role="admin")
    await fd_db.grant_usage_resets([u["id"]], 2, 3)
    async with _client(_app(u)) as c:
        assert (await c.post("/fd/quota/reset")).status_code == 409
    assert await fd_db.get_usage_resets(u["id"]) == 2
    async with _client(_app(admin)) as c:
        assert (await c.post("/fd/quota/reset")).status_code == 400


async def test_admin_grant_routes(fd_db):
    admin = await _user(fd_db, role="admin")
    a, b, c_ = await _user(fd_db), await _user(fd_db), await _user(fd_db)
    async with _client(_app(admin)) as c:
        r = await c.get("/fd/admin/usage/today")
        assert r.json()["resets"] == {} and r.json()["reset_cap"] == usage_quota.DEFAULT_RESET_CAP
        assert (await c.post("/fd/admin/usage/resets", json={"count": 1})).status_code == 400
        assert (await c.post("/fd/admin/usage/resets",
                             json={"count": 0, "all": True})).status_code == 422
        # Several users, with a new cap that the deck keeps.
        r = await c.post("/fd/admin/usage/resets",
                         json={"count": 5, "cap": 2, "user_ids": [a["id"], b["id"], "ghost"]})
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "cap": 2, "users": 2, "given": 2,
                            "balances": {a["id"]: 2, b["id"]: 2}}
        # Everyone: a and b are at the cap already.
        r = await c.post("/fd/admin/usage/resets", json={"count": 1, "all": True})
        assert r.json()["cap"] == 2 and r.json()["given"] == 1
        assert r.json()["balances"] == {a["id"]: 2, b["id"]: 2, c_["id"]: 1}
        r = await c.get("/fd/admin/usage/today")
        assert r.json()["reset_cap"] == 2 and r.json()["resets"][c_["id"]] == 1
        # Take one back.
        r = await c.put(f"/fd/admin/users/{a['id']}/usage/resets", json={"balance": 1})
        assert r.status_code == 200 and r.json()["balance"] == 1
        assert (await c.put("/fd/admin/users/ghost/usage/resets",
                            json={"balance": 1})).status_code == 404
    assert await fd_db.get_usage_resets(a["id"]) == 1
    async with _client(_app(a)) as c:
        assert (await c.post("/fd/admin/usage/resets",
                             json={"count": 1, "all": True})).status_code == 403
        assert (await c.put(f"/fd/admin/users/{a['id']}/usage/resets",
                            json={"balance": 9})).status_code == 403


async def test_resets_go_with_a_deleted_user(fd_db):
    u = await _user(fd_db)
    await fd_db.grant_usage_resets([u["id"]], 1, 3)
    await fd_db.delete_user(u["id"])
    assert await fd_db.list_usage_resets() == {}


# ── Scheduled tasks ──────────────────────────────────────────────────


async def test_scheduled_task_skips_when_tokens_are_spent(monkeypatch):
    from captain_claw.flight_deck import fd_scheduler

    async def spent(owner):
        return {"blocked": "tokens"} if owner == "u1" else None
    monkeypatch.setattr(usage_quota, "automation_blocked", spent)
    monkeypatch.setattr(fd_scheduler, "_in_quiet_hours", lambda: False)
    status, _ = await fd_scheduler.execute_job({"owner_id": "u1", "agent_slug": "a", "prompt": "x"})
    assert status == "skipped:limit"


# ── Agent side ───────────────────────────────────────────────────────


async def test_agent_reports_only_under_flight_deck(monkeypatch):
    import asyncio

    from captain_claw import fd_usage_report as rep

    sent: list = []

    class FakeClient:
        async def post(self, path, *, json=None, headers=None, params=None):
            sent.append((path, json, headers, params))
            return httpx.Response(200)
    monkeypatch.setattr(rep, "_client", lambda: FakeClient())
    for var in ("FD_URL", "FD_AGENT_SLUG", "FD_OWNER_ID"):
        monkeypatch.delenv(var, raising=False)
    rep.report_tokens(100)
    await asyncio.sleep(0)
    assert sent == []

    monkeypatch.setenv("FD_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("FD_AGENT_SLUG", "a")
    monkeypatch.setenv("FD_OWNER_ID", "u1")
    rep.report_tokens(0)
    rep.report_tokens(50, byok=True)  # the person's own key: nothing to bill
    rep.report_tokens(100)
    for _ in range(3):
        await asyncio.sleep(0)
    assert sent == [(rep.USAGE_PATH, {"tokens": 100}, {}, {})]


async def test_member_report_falls_back_to_owner_after_the_turn(monkeypatch):
    from captain_claw import fd_usage_report as rep

    calls: list = []

    class FakeClient:
        async def post(self, path, *, json=None, headers=None, params=None):
            calls.append(headers)
            return httpx.Response(403 if headers else 200)
    monkeypatch.setattr(rep, "_client", lambda: FakeClient())
    await rep._post(10, {"X-FD-Speaker-Grant": "g"}, {"fd_member": "1"})
    assert calls == [{"X-FD-Speaker-Grant": "g"}, None]


# ── Only tokens that missed the prompt cache count ───────────────────


def test_uncached_tokens_per_provider_shape():
    from captain_claw.llm import input_tokens_total, uncached_tokens

    # LiteLLM folds Anthropic / Gemini cache reads and writes into prompt_tokens.
    folded = {"prompt_tokens": 50_000, "completion_tokens": 300,
              "cache_read_input_tokens": 48_000, "cache_creation_input_tokens": 1_500}
    assert input_tokens_total("anthropic", folded) == 50_000
    assert uncached_tokens("anthropic", folded) == 2_300  # 500 fresh + 1500 written + 300 out
    assert uncached_tokens("gemini", folded) == 2_300
    # OpenAI-style: _extract_usage already took the cached part out of prompt_tokens.
    split = {"prompt_tokens": 2_000, "completion_tokens": 300, "cache_read_input_tokens": 48_000}
    assert input_tokens_total("openai", split) == 50_000
    assert uncached_tokens("openai", split) == 2_300
    # An Anthropic-named usage that is NOT folded (prompt < cache) is read as split.
    assert uncached_tokens("anthropic", {"prompt_tokens": 500, "completion_tokens": 300,
                                         "cache_read_input_tokens": 48_000}) == 800
    # No cache at all (Ollama): everything counts.
    assert uncached_tokens("ollama", {"prompt_tokens": 9_000, "completion_tokens": 40}) == 9_040
    assert uncached_tokens("ollama", None) == 0


def test_responses_api_cached_input_is_split_out():
    from captain_claw.llm import ChatGPTResponsesProvider, uncached_tokens

    provider = ChatGPTResponsesProvider.__new__(ChatGPTResponsesProvider)
    provider.model = "gpt-test"
    event = {"type": "response.completed", "response": {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
        "usage": {"input_tokens": 30_000, "output_tokens": 50, "total_tokens": 30_050,
                  "input_tokens_details": {"cached_tokens": 28_000}},
    }}
    usage = provider._parse_response_output(event).usage
    assert usage["prompt_tokens"] == 2_000 and usage["cache_read_input_tokens"] == 28_000
    assert uncached_tokens("chatgpt", usage) == 2_050


async def test_agent_reports_only_uncached_tokens(tmp_path, monkeypatch):
    from captain_claw import fd_usage_report
    from captain_claw.session import SessionManager

    reported: list = []
    monkeypatch.setattr(fd_usage_report, "report_tokens",
                        lambda tokens, *, byok=False: reported.append((tokens, byok)))
    sm = SessionManager(tmp_path / "s.db")
    await sm.record_llm_usage(provider="anthropic", prompt_tokens=50_000, completion_tokens=300,
                              total_tokens=50_300, cache_read_input_tokens=48_000,
                              cache_creation_input_tokens=1_500)
    await sm.record_llm_usage(provider="openai", prompt_tokens=100, completion_tokens=20,
                              total_tokens=120, byok=True)
    assert reported == [(2_300, False), (120, True)]
