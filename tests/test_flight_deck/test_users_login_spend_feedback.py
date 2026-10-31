"""The Users page's last login, spend and feedback request: ``usage_spend`` reads
the agents' ``llm_usage`` into a ledger, sign-ins stamp ``last_login_at``, and
``feedback_request_routes`` emails a user from the admin's own account."""

from __future__ import annotations

import base64
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import (
    admin_routes,
    auth_routes,
    feedback_request_routes,
    invite_routes,
    pricing,
    usage_spend,
)
from captain_claw.flight_deck.auth import get_current_user, hash_password, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

LOOPBACK = ("127.0.0.1", 40001)
GMAIL_COMPOSE = "https://www.googleapis.com/auth/gmail.compose"
THIS_MONTH = datetime.now(UTC).strftime("%Y-%m")
LONG_AGO = "2020-03-04"


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch):
    from captain_claw.flight_deck import auth as fd_auth

    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.delenv("FD_GMAIL_SEND", raising=False)
    monkeypatch.setattr(usage_spend, "_last_sync", 0.0)
    monkeypatch.setattr(usage_spend, "_sync_task", None)
    boss = await db.create_user("boss@example.com", hash_password("boss-pass"), "Boss", "admin")
    ana = await db.create_user("ana@example.com", hash_password("ana-pass"), "Ana", "user")
    try:
        yield SimpleNamespace(db=db, admin={**boss, "role": "admin"}, ana=ana, dir=tmp_path)
    finally:
        await db.close()
        fd_auth._db = prev


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LOOPBACK),
                             base_url="http://fd.test")


def _app(user: dict) -> FastAPI:
    app = FastAPI()
    app.include_router(usage_spend.router)
    app.include_router(feedback_request_routes.router)
    app.include_router(admin_routes.router)
    app.include_router(auth_routes.router)
    app.dependency_overrides[get_current_user] = lambda: user
    return app


# ── An agent's llm_usage ─────────────────────────────────────────────


def _agent_db(folder: Path, name: str = "sessions.db") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS llm_usage (
        id TEXT PRIMARY KEY, provider TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',
        prompt_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0,
        cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
        cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
        byok INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)""")
    conn.commit()
    conn.close()
    return path


def _call(path: Path, model="gpt-4o", provider="openai", *, prompt=0, output=0, created=0,
          cached=0, byok=0, day=None) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO llm_usage (id, provider, model, prompt_tokens, completion_tokens,"
        " cache_creation_input_tokens, cache_read_input_tokens, byok, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (uuid.uuid4().hex, provider, model, prompt, output, created, cached, byok,
         f"{day or datetime.now(UTC).date().isoformat()}T12:00:00+00:00"))
    conn.commit()
    conn.close()


def _agents(monkeypatch, **by_slug) -> None:
    monkeypatch.setattr(usage_spend, "_agents", lambda: by_slug)


# ── The ledger ───────────────────────────────────────────────────────


async def test_spend_splits_this_month_from_all_time(deck, monkeypatch):
    path = _agent_db(deck.dir / "a")
    _call(path, "gpt-4o", prompt=1_000_000, output=1_000_000)              # $2.50 + $10
    _call(path, "gpt-4o", prompt=1_000_000, day=LONG_AGO)                  # $2.50, long ago
    _agents(monkeypatch, a=(deck.ana["id"], path))

    assert await usage_spend.sync() == 2
    body = await usage_spend.totals(deck.db)

    assert body["month"] == THIS_MONTH
    assert body["since"] == LONG_AGO
    user = body["users"][deck.ana["id"]]
    assert user["month"]["usd"] == pytest.approx(12.5)
    assert user["month"]["tokens"] == 2_000_000
    assert user["all"]["usd"] == pytest.approx(15.0)
    assert user["all"]["tokens"] == 3_000_000


async def test_a_second_pass_takes_only_the_new_calls(deck, monkeypatch):
    path = _agent_db(deck.dir / "a")
    _call(path, prompt=1_000_000)
    _agents(monkeypatch, a=(deck.ana["id"], path))

    assert await usage_spend.sync() == 1
    assert await usage_spend.sync() == 0
    _call(path, prompt=1_000_000)
    assert await usage_spend.sync() == 1

    user = (await usage_spend.totals(deck.db))["users"][deck.ana["id"]]
    assert user["all"]["usd"] == pytest.approx(5.0)


async def test_what_was_taken_outlives_the_agent(deck, monkeypatch):
    path = _agent_db(deck.dir / "a")
    _call(path, prompt=1_000_000)
    _agents(monkeypatch, a=(deck.ana["id"], path))
    await usage_spend.sync()

    path.unlink()
    _agents(monkeypatch)
    await usage_spend.sync()

    user = (await usage_spend.totals(deck.db))["users"][deck.ana["id"]]
    assert user["all"]["usd"] == pytest.approx(2.5)


async def test_a_new_database_under_the_same_name_is_read_from_its_start(deck, monkeypatch):
    path = _agent_db(deck.dir / "a")
    for _ in range(3):
        _call(path, prompt=1_000_000)
    _agents(monkeypatch, a=(deck.ana["id"], path))
    await usage_spend.sync()

    path.unlink()
    path = _agent_db(deck.dir / "a")
    _call(path, prompt=1_000_000)
    assert await usage_spend.sync() == 1
    assert await usage_spend.sync() == 0


async def test_several_agents_add_up_and_each_bills_its_owner(deck, monkeypatch):
    one, two, other = _agent_db(deck.dir / "1"), _agent_db(deck.dir / "2"), _agent_db(deck.dir / "3")
    _call(one, prompt=1_000_000)
    _call(two, prompt=1_000_000)
    _call(other, prompt=1_000_000)
    _agents(monkeypatch, one=(deck.ana["id"], one), two=(deck.ana["id"], two),
            other=(deck.admin["id"], other))
    await usage_spend.sync()

    users = (await usage_spend.totals(deck.db))["users"]
    assert users[deck.ana["id"]]["all"]["usd"] == pytest.approx(5.0)
    assert users[deck.admin["id"]]["all"]["usd"] == pytest.approx(2.5)


async def test_a_model_with_no_price_stays_tokens_and_is_named(deck, monkeypatch):
    path = _agent_db(deck.dir / "a")
    _call(path, "Qwen3.6-local", prompt=700, output=300)
    _call(path, "gpt-4o", prompt=1_000_000)
    _agents(monkeypatch, a=(deck.ana["id"], path))
    await usage_spend.sync()

    user = (await usage_spend.totals(deck.db))["users"][deck.ana["id"]]
    assert user["all"]["usd"] == pytest.approx(2.5)
    assert user["all"]["unpriced_tokens"] == 1000
    assert user["unpriced_models"] == ["Qwen3.6-local"]


async def test_calls_on_their_own_key_are_kept_apart(deck, monkeypatch):
    path = _agent_db(deck.dir / "a")
    _call(path, prompt=1_000_000)
    _call(path, prompt=2_000_000, byok=1)
    _agents(monkeypatch, a=(deck.ana["id"], path))
    await usage_spend.sync()

    period = (await usage_spend.totals(deck.db))["users"][deck.ana["id"]]["all"]
    assert period["usd"] == pytest.approx(2.5)
    assert period["tokens"] == 1_000_000
    assert period["own_usd"] == pytest.approx(5.0)
    assert period["own_tokens"] == 2_000_000


async def test_anthropic_cache_inside_prompt_tokens_is_not_billed_twice(deck, monkeypatch):
    # LiteLLM's Anthropic usage folds cache reads and writes into prompt_tokens.
    path = _agent_db(deck.dir / "a")
    _call(path, "claude-opus-4-8", "anthropic", prompt=1_100_000, created=100_000, cached=900_000)
    _agents(monkeypatch, a=(deck.ana["id"], path))
    await usage_spend.sync()

    period = (await usage_spend.totals(deck.db))["users"][deck.ana["id"]]["all"]
    # 100k fresh at $5, 100k written at $6.25, 900k read at $0.50.
    assert period["usd"] == pytest.approx(0.5 + 0.625 + 0.45)
    assert period["tokens"] == 200_000


async def test_an_agent_from_before_own_keys_is_read_too(deck, monkeypatch):
    path = _agent_db(deck.dir / "a")
    _call(path, prompt=1_000_000)
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE llm_usage DROP COLUMN byok")
    conn.commit()
    conn.close()
    _agents(monkeypatch, a=(deck.ana["id"], path))

    assert await usage_spend.sync() == 1
    user = (await usage_spend.totals(deck.db))["users"][deck.ana["id"]]
    assert user["all"]["usd"] == pytest.approx(2.5)


async def test_a_busy_agent_waits_for_the_next_pass(deck, monkeypatch):
    path = _agent_db(deck.dir / "a")
    _call(path, prompt=1_000_000)
    broken = deck.dir / "b.db"
    broken.write_text("not a database")
    _agents(monkeypatch, b=(deck.admin["id"], broken), a=(deck.ana["id"], path))

    assert await usage_spend.sync() == 1  # the other agents are still read


def test_an_agents_folder_names_its_owner(tmp_path, monkeypatch):
    from captain_claw.flight_deck import server as srv

    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)
    for slug in ("in-registry", "only-folder", "nobody"):
        _agent_db(tmp_path / slug / "data" / "sessions")
    (tmp_path / "only-folder" / srv.FOLDER_OWNER_FILE).write_text("carol\n")
    (tmp_path / "in-registry" / srv.FOLDER_OWNER_FILE).write_text("old-owner\n")
    monkeypatch.setattr(srv, "_load_process_registry", lambda: {"in-registry": {"owner": "dan"}})

    assert {slug: owner for slug, (owner, _) in usage_spend._agents().items()} == {
        "in-registry": "dan", "only-folder": "carol"}


def test_new_models_have_prices():
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "deepseek-v4-flash", "deepseek-v4-pro", "gpt-5.5"):
        assert pricing.rate_for(model), model
    # A prefix of a dearer model must not price the cheaper one, nor the reverse.
    assert pricing.rate_for("gpt-5.5")["input"] == 5.0
    assert pricing.rate_for("gpt-5.5-pro")["input"] == 30.0
    assert pricing.rate_for("claude-opus-5-5")["input"] == 4.0
    assert pricing.rate_for("openai/deepseek-v4-flash")["input"] == 0.15


async def test_the_spend_route_is_for_admins_and_reads_the_agents(deck, monkeypatch):
    path = _agent_db(deck.dir / "a")
    _call(path, prompt=1_000_000)
    _agents(monkeypatch, a=(deck.ana["id"], path))

    async with _client(_app({**deck.ana, "role": "user"})) as c:
        assert (await c.get("/fd/admin/usage/spend")).status_code == 403
    async with _client(_app(deck.admin)) as c:
        r = await c.get("/fd/admin/usage/spend")
    assert r.status_code == 200, r.text
    assert r.json()["users"][deck.ana["id"]]["all"]["usd"] == pytest.approx(2.5)


# ── Last login ───────────────────────────────────────────────────────


async def _last_login(db, user_id: str) -> str:
    return (await db.get_user_by_id(user_id))["last_login_at"]


async def test_signing_in_and_coming_back_stamp_the_last_login(deck, monkeypatch):
    uid = deck.ana["id"]
    assert await _last_login(deck.db, uid) == ""

    app = FastAPI()
    app.include_router(auth_routes.router)
    async with _client(app) as c:
        assert (await c.post("/fd/auth/login", json={"email": "ana@example.com", "password": "nope"})).status_code == 401
        assert await _last_login(deck.db, uid) == ""

        r = await c.post("/fd/auth/login", json={"email": "ana@example.com", "password": "ana-pass"})
        assert r.status_code == 200, r.text
        first = await _last_login(deck.db, uid)
        assert first

        r = await c.post("/fd/auth/refresh")
        assert r.status_code == 200, r.text
        assert await _last_login(deck.db, uid) > first


async def test_the_user_list_has_last_login_and_the_user_payload_does_not(deck):
    await deck.db.touch_last_login(deck.ana["id"])
    async with _client(_app(deck.admin)) as c:
        users = (await c.get("/fd/admin/users")).json()["users"]
    ana = next(u for u in users if u["id"] == deck.ana["id"])
    assert ana["last_login_at"]
    assert ana["feedback_asked_at"] == ""

    payload = await auth_routes._user_payload(deck.db, await deck.db.get_user_by_id(deck.ana["id"]))
    assert "last_login_at" not in payload
    assert "feedback_asked_at" not in payload


async def test_a_deck_upgrading_takes_last_logins_from_its_sessions(tmp_path):
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    try:
        signed_in = await db.create_user("in@example.com", "x")
        never = await db.create_user("never@example.com", "x")
        await db.create_refresh_session(signed_in["id"], "h1", "2099-01-01T00:00:00+00:00")
        await db._db.execute("UPDATE user_sessions SET created_at = '2026-05-06T07:08:09+00:00'")
        await db._db.commit()
        await db.close()

        db = FlightDeckDB(tmp_path / "fd.db")
        await db.init()
        assert await _last_login(db, signed_in["id"]) == "2026-05-06T07:08:09+00:00"
        assert await _last_login(db, never["id"]) == ""
    finally:
        await db.close()


# ── Asking for feedback ──────────────────────────────────────────────


def _connect_google(monkeypatch):
    async def connected(uid):
        return True

    async def tokens(db, uid):
        return SimpleNamespace(scope=f"openid email {GMAIL_COMPOSE}")

    async def user(db, uid):
        return {"email": "boss@gmail.com"}

    async def access(uid):
        return "g-token"

    monkeypatch.setattr(invite_routes._google, "is_google_connected", connected)
    monkeypatch.setattr(invite_routes._google, "_load_tokens", tokens)
    monkeypatch.setattr(invite_routes._google, "_load_user", user)
    monkeypatch.setattr(invite_routes._google, "get_valid_google_access_token", access)


def _mock_http(monkeypatch, handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def wrapped(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return handler(req)

    monkeypatch.setattr(invite_routes, "_http",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(wrapped)))
    return seen


def _decoded(req: httpx.Request) -> str:
    raw = json.loads(req.content)["raw"]
    return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode().replace("=\r\n", "")


async def test_the_draft_asks_what_was_wanted_and_names_who_can_send_it(deck, monkeypatch):
    _connect_google(monkeypatch)
    async with _client(_app(deck.admin)) as c:
        r = await c.get(f"/fd/admin/users/{deck.ana['id']}/feedback-request")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["to"] == "ana@example.com"
    assert body["subject"] == "Ana, how is Captain working for you?"
    assert body["body"].startswith("Hi Ana,")
    assert body["body"].endswith("Boss")
    for question in ("use it for", "worked well", "confuses you", "missing", "make it better"):
        assert question in body["body"]
    assert body["asked_at"] == ""
    assert body["mail"]["google"] == {"available": True, "from": "boss@gmail.com", "reason": ""}
    assert body["mail"]["microsoft"]["available"] is False


def test_a_draft_for_someone_with_no_name_still_reads_well():
    subject, body = feedback_request_routes.compose_request(
        instance_name="Captain", admin_name="Boss", user_name="")
    assert subject == "How is Captain working for you?"
    assert body.startswith("Hi,\n")


async def test_sending_emails_the_user_and_marks_them_asked(deck, monkeypatch):
    _connect_google(monkeypatch)
    seen = _mock_http(monkeypatch, lambda req: httpx.Response(200, json={"id": "m1"}))
    async with _client(_app(deck.admin)) as c:
        r = await c.post(f"/fd/admin/users/{deck.ana['id']}/feedback-request", json={
            "subject": "Quick question\r\nBcc: evil@example.com", "body": "Hi Ana,\n\nHow's it going? <b>Tell me</b>",
            "send_via": "google",
        })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["email"]["from"] == "boss@gmail.com"
    assert body["asked_at"]

    [req] = seen
    assert req.url.path == "/gmail/v1/users/me/messages/send"
    message = _decoded(req)
    assert "To: ana@example.com" in message
    assert "Subject: Quick question Bcc: evil@example.com" in message  # one header, no injection
    assert "\nBcc:" not in message.replace("\r\n", "\n")
    assert "&lt;b&gt;Tell me&lt;/b&gt;" in message

    user = await deck.db.get_user_by_id(deck.ana["id"])
    assert user["feedback_asked_at"] == body["asked_at"]
    assert (await deck.db.get_usage_logs(event_type="feedback_requested"))[0]["user_id"] == deck.admin["id"]


async def test_a_failed_email_says_why_and_does_not_mark_them_asked(deck, monkeypatch):
    _connect_google(monkeypatch)
    _mock_http(monkeypatch, lambda req: httpx.Response(403, json={"error": {"message": "no"}}))
    async with _client(_app(deck.admin)) as c:
        r = await c.post(f"/fd/admin/users/{deck.ana['id']}/feedback-request", json={
            "subject": "Hello", "body": "Hi", "send_via": "google"})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is False
    assert "Gmail refused" in r.json()["email"]["error"]
    assert (await deck.db.get_user_by_id(deck.ana["id"]))["feedback_asked_at"] == ""


async def test_an_unconnected_account_is_told_how_to_connect(deck):
    async with _client(_app(deck.admin)) as c:
        r = await c.post(f"/fd/admin/users/{deck.ana['id']}/feedback-request", json={
            "subject": "Hello", "body": "Hi", "send_via": "microsoft"})
    assert r.json()["ok"] is False
    assert "Connections" in r.json()["email"]["error"]


async def test_feedback_requests_are_checked(deck):
    url = f"/fd/admin/users/{deck.ana['id']}/feedback-request"
    good = {"subject": "Hello", "body": "Hi", "send_via": "google"}
    async with _client(_app(deck.admin)) as c:
        assert (await c.post(url, json={**good, "send_via": "pigeon"})).status_code == 400
        assert (await c.post(url, json={**good, "subject": "  "})).status_code == 400
        assert (await c.post(url, json={**good, "body": "\n"})).status_code == 400
        assert (await c.get("/fd/admin/users/nobody/feedback-request")).status_code == 404
        assert (await c.post("/fd/admin/users/nobody/feedback-request", json=good)).status_code == 404
    async with _client(_app({**deck.ana, "role": "user"})) as c:
        assert (await c.get(url)).status_code == 403
        assert (await c.post(url, json=good)).status_code == 403
