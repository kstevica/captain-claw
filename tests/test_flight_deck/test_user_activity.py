"""The Admin console's Activity page (``user_activity``): who is active, from
the messages they typed and their agents' work, and whether an admin wrote to
them — the deck's own record and the admin's Gmail."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import (
    feedback_request_routes,
    invite_routes,
    usage_spend,
    user_activity,
)
from captain_claw.flight_deck.auth import get_current_user, hash_password, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.usage_quota import today

LOOPBACK = ("127.0.0.1", 40001)
GMAIL_COMPOSE = "https://www.googleapis.com/auth/gmail.compose"
GMAIL_MODIFY = "https://www.googleapis.com/auth/gmail.modify"


def _ago(n: int) -> str:
    return (date.fromisoformat(today()) - timedelta(days=n)).isoformat()


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
    monkeypatch.setattr(usage_spend, "_agents", lambda: {})
    user_activity._mailbox_cache.clear()
    boss = await db.create_user("boss@example.com", hash_password("boss-pass"), "Boss", "admin")
    ana = await db.create_user("ana@example.com", hash_password("ana-pass"), "Ana", "user")
    ivo = await db.create_user("ivo@example.com", hash_password("ivo-pass"), "Ivo", "user")
    mia = await db.create_user("mia@example.com", hash_password("mia-pass"), "Mia", "user")
    try:
        yield SimpleNamespace(db=db, admin={**boss, "role": "admin"}, ana=ana, ivo=ivo, mia=mia,
                              dir=tmp_path)
    finally:
        await db.close()
        fd_auth._db = prev


def _app(user: dict) -> FastAPI:
    app = FastAPI()
    app.include_router(user_activity.router)
    app.include_router(feedback_request_routes.router)
    app.dependency_overrides[get_current_user] = lambda: user
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LOOPBACK),
                             base_url="http://fd.test")


def _by_email(body: dict) -> dict[str, dict]:
    return {u["email"]: u for u in body["users"]}


async def _typed(db, user_id: str, day: str, n: int) -> None:
    await db.add_daily_usage(user_id, day, messages=n)


async def _agent_tokens(db, user_id: str, day: str, tokens: int) -> None:
    await db.add_usage_spend("agent-" + user_id[:6], user_id, [{
        "day": day, "model": "gpt-4o", "byok": 0, "input_tokens": tokens, "output_tokens": 0,
        "cache_write_tokens": 0, "cache_read_tokens": 0, "calls": 1}], 1)


# ── Who is active ───────────────────────────────────────────────────


async def test_status_comes_from_the_last_day_they_typed(deck):
    await _typed(deck.db, deck.ana["id"], _ago(0), 3)
    await _typed(deck.db, deck.ana["id"], _ago(2), 2)
    await _typed(deck.db, deck.ivo["id"], _ago(12), 5)
    await _typed(deck.db, deck.ivo["id"], _ago(60), 4)   # before the window

    async with _client(_app(deck.admin)) as c:
        r = await c.get("/fd/admin/activity")
    assert r.status_code == 200, r.text
    body = r.json()
    users = _by_email(body)

    assert "boss@example.com" not in users          # the admin looking isn't listed
    ana, ivo, mia = users["ana@example.com"], users["ivo@example.com"], users["mia@example.com"]
    assert ana["status"] == "active"
    t = ana["typed"]
    assert (t["messages"], t["messages_week"], t["days"], t["last_day"], t["messages_all"],
            t["days_all"]) == (5, 5, 2, _ago(0), 5, 2)
    assert [d["day"] for d in ana["days"]] == [_ago(2), _ago(0)]
    assert ivo["status"] == "quiet"
    assert ivo["typed"]["messages"] == 5 and ivo["typed"]["messages_week"] == 0
    assert ivo["typed"]["messages_all"] == 9 and ivo["typed"]["first_day"] == _ago(60)
    assert mia["status"] == "never"
    assert mia["days"] == [] and mia["typed"]["last_day"] == ""
    assert body["since"] == _ago(29) and body["days"] == 30


def test_a_week_counts_today_and_the_six_days_before():
    now = today()
    assert user_activity.status_for(_ago(6), now) == "active"
    assert user_activity.status_for(_ago(7), now) == "quiet"
    assert user_activity.status_for("", now) == "never"


async def test_agent_work_shows_without_counting_as_typing(deck):
    await _agent_tokens(deck.db, deck.mia["id"], _ago(1), 12_000)
    await _agent_tokens(deck.db, deck.mia["id"], _ago(3), 8_000)

    async with _client(_app(deck.admin)) as c:
        mia = _by_email((await c.get("/fd/admin/activity")).json())["mia@example.com"]
    assert mia["status"] == "never"
    assert mia["agents"] == {"last_day": _ago(1), "tokens": 20_000, "days": 2}
    assert [(d["day"], d["messages"], d["tokens"]) for d in mia["days"]] == [
        (_ago(3), 0, 8_000), (_ago(1), 0, 12_000)]


async def test_the_window_can_be_shorter_or_longer(deck):
    await _typed(deck.db, deck.ivo["id"], _ago(40), 2)
    async with _client(_app(deck.admin)) as c:
        wide = _by_email((await c.get("/fd/admin/activity?days=90")).json())["ivo@example.com"]
        narrow = _by_email((await c.get("/fd/admin/activity?days=7")).json())["ivo@example.com"]
        assert (await c.get("/fd/admin/activity?days=3")).status_code == 422
        assert (await c.get("/fd/admin/activity?days=365")).status_code == 422
    assert wide["typed"]["messages"] == 2
    assert narrow["typed"]["messages"] == 0 and narrow["status"] == "quiet"


async def test_activity_is_for_admins_on_decks_with_accounts(deck, monkeypatch):
    async with _client(_app({**deck.ana, "role": "user"})) as c:
        assert (await c.get("/fd/admin/activity")).status_code == 403
        assert (await c.get("/fd/admin/activity/mailbox")).status_code == 403
        assert (await c.post(f"/fd/admin/activity/{deck.ivo['id']}/contacts",
                             json={"channel": "WhatsApp"})).status_code == 403
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    async with _client(_app(deck.admin)) as c:
        r = await c.get("/fd/admin/activity")
    assert r.status_code == 403
    assert "without" in r.json()["detail"]


# ── What the deck knows about writing to them ───────────────────────


async def test_a_noted_contact_shows_and_can_be_taken_back(deck):
    async with _client(_app(deck.admin)) as c:
        r = await c.post(f"/fd/admin/activity/{deck.ana['id']}/contacts",
                         json={"channel": "  WhatsApp\n", "note": "Asked how the tender radar works"})
        assert r.status_code == 200, r.text
        contact = r.json()["contact"]
        assert contact["kind"] == "noted" and contact["channel"] == "WhatsApp"
        assert contact["admin_name"] == "Boss"

        ana = _by_email((await c.get("/fd/admin/activity")).json())["ana@example.com"]
        assert ana["contacts_total"] == 1
        [shown] = ana["contacts"]
        assert shown["note"] == "Asked how the tender radar works" and shown["admin_name"] == "Boss"

        assert (await c.delete(f"/fd/admin/activity/contacts/{contact['id']}")).status_code == 200
        ana = _by_email((await c.get("/fd/admin/activity")).json())["ana@example.com"]
        assert ana["contacts"] == []
        assert (await c.delete(f"/fd/admin/activity/contacts/{contact['id']}")).status_code == 404


async def test_noted_contacts_are_checked(deck):
    async with _client(_app(deck.admin)) as c:
        assert (await c.post(f"/fd/admin/activity/{deck.ana['id']}/contacts",
                             json={"channel": "  "})).status_code == 400
        assert (await c.post("/fd/admin/activity/nobody/contacts",
                             json={"channel": "Call"})).status_code == 404


async def test_a_removed_user_takes_their_contacts_along(deck):
    await deck.db.add_user_contact(deck.ivo["id"], deck.admin["id"], "noted", "Call")
    await deck.db.delete_user(deck.ivo["id"])
    assert await deck.db.list_user_contacts() == []


# ── Emails the deck sends ───────────────────────────────────────────


def _connect_google(monkeypatch, scope: str = GMAIL_COMPOSE):
    async def connected(uid):
        return True

    async def tokens(db, uid):
        return SimpleNamespace(scope=f"openid email {scope}")

    async def user(db, uid):
        return {"email": "boss@gmail.com"}

    async def access(uid):
        return "g-token"

    for mod in (invite_routes._google, user_activity._google):
        monkeypatch.setattr(mod, "is_google_connected", connected)
        monkeypatch.setattr(mod, "_load_tokens", tokens)
        monkeypatch.setattr(mod, "_load_user", user)
        monkeypatch.setattr(mod, "get_valid_google_access_token", access)


def _mock_send(monkeypatch, status: int = 200) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(status, json={"id": "m1"} if status < 400 else {"error": {"message": "no"}})

    monkeypatch.setattr(invite_routes, "_http",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return seen


async def test_a_message_from_the_page_is_emailed_and_recorded(deck, monkeypatch):
    _connect_google(monkeypatch)
    seen = _mock_send(monkeypatch)
    async with _client(_app(deck.admin)) as c:
        draft = (await c.get(f"/fd/admin/activity/{deck.ana['id']}/message")).json()
        assert draft["to"] == "ana@example.com"
        assert draft["body"].startswith("Hi Ana,") and draft["body"].endswith("Boss")
        assert draft["mail"]["google"]["available"] is True

        r = await c.post(f"/fd/admin/activity/{deck.ana['id']}/message", json={
            "subject": "Your tender radar", "body": "Hi Ana,\n\nHow is it going?\n\nBoss",
            "send_via": "google"})
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is True
        assert r.json()["contact"]["kind"] == "message"
        ana = _by_email((await c.get("/fd/admin/activity")).json())["ana@example.com"]
    [req] = seen
    assert req.url.path == "/gmail/v1/users/me/messages/send"
    [contact] = ana["contacts"]
    assert (contact["kind"], contact["channel"], contact["subject"]) == ("message", "google", "Your tender radar")


async def test_an_email_that_went_out_is_not_reported_failed_when_its_record_fails(deck, monkeypatch):
    _connect_google(monkeypatch)
    _mock_send(monkeypatch)

    async def broken(*a, **kw):
        raise RuntimeError("FOREIGN KEY constraint failed")

    monkeypatch.setattr(deck.db, "add_user_contact", broken)
    async with _client(_app(deck.admin)) as c:
        r = await c.post(f"/fd/admin/activity/{deck.ana['id']}/message", json={
            "subject": "Hello", "body": "Hi", "send_via": "google"})
        assert r.json() == {**r.json(), "ok": True, "contact": None}
        r = await c.post(f"/fd/admin/users/{deck.ana['id']}/feedback-request", json={
            "subject": "Hello", "body": "Hi", "send_via": "google"})
        assert r.json()["ok"] is True


def test_the_blank_message_greets_by_first_name():
    assert user_activity.compose_message(admin_name="Boss", user_name="Marta Horvat").startswith("Hi Marta,\n")
    assert user_activity.compose_message(admin_name="Boss", user_name="  ").startswith("Hi,\n")


async def test_a_failed_message_is_not_recorded(deck, monkeypatch):
    _connect_google(monkeypatch)
    _mock_send(monkeypatch, status=403)
    async with _client(_app(deck.admin)) as c:
        r = await c.post(f"/fd/admin/activity/{deck.ana['id']}/message", json={
            "subject": "Hello", "body": "Hi", "send_via": "google"})
        assert r.json()["ok"] is False
        assert "Gmail refused" in r.json()["email"]["error"]
        assert (await c.post(f"/fd/admin/activity/{deck.ana['id']}/message", json={
            "subject": "Hello", "body": "Hi", "send_via": "pigeon"})).status_code == 400
        assert (await c.post(f"/fd/admin/activity/{deck.ana['id']}/message", json={
            "subject": " ", "body": "Hi", "send_via": "google"})).status_code == 400
    assert await deck.db.list_user_contacts() == []


async def test_a_feedback_request_counts_as_writing_to_them_and_cant_be_taken_back(deck, monkeypatch):
    _connect_google(monkeypatch)
    _mock_send(monkeypatch)
    async with _client(_app(deck.admin)) as c:
        r = await c.post(f"/fd/admin/users/{deck.ivo['id']}/feedback-request", json={
            "subject": "Ivo, how is Captain working for you?", "body": "Hi Ivo", "send_via": "google"})
        assert r.json()["ok"] is True
        ivo = _by_email((await c.get("/fd/admin/activity")).json())["ivo@example.com"]
        [contact] = ivo["contacts"]
        assert contact["kind"] == "feedback" and contact["subject"].startswith("Ivo, how")
        assert contact["created_at"] == r.json()["asked_at"]
        r = await c.delete(f"/fd/admin/activity/contacts/{contact['id']}")
        assert r.status_code == 400


async def test_feedback_requests_from_before_are_taken_in_once(tmp_path):
    db = FlightDeckDB(tmp_path / "old.db")
    await db.init()
    try:
        boss = await db.create_user("boss@example.com", "x", "Boss", "admin")
        ana = await db.create_user("ana@example.com", "x", "Ana", "user")
        await db.log_usage(boss["id"], "feedback_requested", json.dumps({"user_id": ana["id"], "via": "google"}))
        await db.log_usage(boss["id"], "feedback_requested", json.dumps({"user_id": "gone", "via": "google"}))
        await db.log_usage(boss["id"], "feedback_requested", "not json")
    finally:
        await db.close()

    for _ in range(2):   # every start runs the catch-up
        db = FlightDeckDB(tmp_path / "old.db")
        await db.init()
        try:
            rows = await db.list_user_contacts()
        finally:
            await db.close()
        assert [(r["user_id"], r["admin_id"], r["kind"], r["channel"], r["admin_name"]) for r in rows] == [
            (ana["id"], boss["id"], "feedback", "google", "Boss")]


# ── The admin's own Gmail ───────────────────────────────────────────

GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me/messages"


def _mailbox(monkeypatch, found: dict[str, tuple[str, int, str]], status: int = 200) -> list[str]:
    """Gmail with *found*: a search → (message id, epoch ms, subject)."""
    queries: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["Authorization"] == "Bearer g-token"
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "nope"}})
        url = urlparse(str(req.url))
        if url.path.endswith("/messages"):
            q = parse_qs(url.query)["q"][0]
            queries.append(q)
            hit = found.get(q)
            return httpx.Response(200, json={"messages": [{"id": hit[0]}]} if hit else {"resultSizeEstimate": 0})
        mid = url.path.rsplit("/", 1)[1]
        _, ms, subject = next(v for v in found.values() if v[0] == mid)
        return httpx.Response(200, json={"id": mid, "internalDate": str(ms),
                                         "payload": {"headers": [{"name": "Subject", "value": subject}]}})

    monkeypatch.setattr(user_activity, "_http",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return queries


async def test_the_mailbox_shows_the_newest_email_each_way(deck, monkeypatch):
    _connect_google(monkeypatch, GMAIL_MODIFY)
    queries = _mailbox(monkeypatch, {
        "in:sent {to:ana@example.com cc:ana@example.com bcc:ana@example.com}": ("s1", 1_762_000_000_000, "How is it going?"),
        "from:ana@example.com": ("r1", 1_762_100_000_000, "Re: How is it going?"),
    })
    async with _client(_app(deck.admin)) as c:
        r = await c.get("/fd/admin/activity/mailbox")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is True and body["partial"] is False
    assert body["address"] == "boss@gmail.com"
    ana = body["users"][deck.ana["id"]]
    assert ana["sent"] == {"at": "2025-11-01T12:26:40+00:00", "subject": "How is it going?"}
    assert ana["received"]["subject"] == "Re: How is it going?"
    assert body["users"][deck.ivo["id"]] == {"sent": None, "received": None}
    assert deck.admin["id"] not in body["users"]
    assert len(queries) == 6   # three people, two searches each


async def test_the_mailbox_is_kept_a_while_unless_asked_again(deck, monkeypatch):
    _connect_google(monkeypatch, GMAIL_MODIFY)
    queries = _mailbox(monkeypatch, {})
    async with _client(_app(deck.admin)) as c:
        await c.get("/fd/admin/activity/mailbox")
        await c.get("/fd/admin/activity/mailbox")
        assert len(queries) == 6
        await c.get("/fd/admin/activity/mailbox?refresh=true")
    assert len(queries) == 12


async def test_a_google_connection_that_cannot_read_mail_says_so(deck, monkeypatch):
    _connect_google(monkeypatch, GMAIL_COMPOSE)
    queries = _mailbox(monkeypatch, {})
    async with _client(_app(deck.admin)) as c:
        body = (await c.get("/fd/admin/activity/mailbox")).json()
    assert body["available"] is False
    assert "gmail.readonly" in body["reason"]
    assert queries == []


async def test_no_google_connection_says_how_to_connect(deck):
    async with _client(_app(deck.admin)) as c:
        body = (await c.get("/fd/admin/activity/mailbox")).json()
    assert body["available"] is False
    assert "Connections → Google" in body["reason"]


async def test_gmail_refusing_the_read_says_why_and_keeps_nothing(deck, monkeypatch):
    _connect_google(monkeypatch, GMAIL_MODIFY)
    _mailbox(monkeypatch, {}, status=401)
    async with _client(_app(deck.admin)) as c:
        body = (await c.get("/fd/admin/activity/mailbox")).json()
    assert body["available"] is False and body["users"] == {}
    assert "reconnect" in body["reason"]
    assert user_activity._mailbox_cache == {}


async def test_addresses_gmail_search_could_misread_are_left_out(deck, monkeypatch):
    _connect_google(monkeypatch, GMAIL_MODIFY)
    odd = await deck.db.create_user('odd"{x}@example.com', "x", "Odd", "user")
    queries = _mailbox(monkeypatch, {})
    async with _client(_app(deck.admin)) as c:
        body = (await c.get("/fd/admin/activity/mailbox")).json()
    assert odd["id"] not in body["users"]
    assert not any("odd" in q for q in queries)
