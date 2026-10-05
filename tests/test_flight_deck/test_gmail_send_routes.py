"""Gmail sending through Flight Deck — /fd/google/gmail/send + the user policy.

FD is the one place a send is decided: the agent OWNER's opt-in policy (off by
default; allowlist; daily limit), the deck kill switch FD_GMAIL_SEND, duplicate
suppression, then the send with the owner's token, an audit row and a bell
notification. The agent route has /access_token's gate (agent transport +
X-Agent-Auth this deck issued, no browsers); the policy / history routes are the
signed-in user's own.

Google is never contacted: Gmail answers through httpx.MockTransport. The app is
a bare FastAPI with the two Google routers, driven by TestClient from loopback
(as test_google_tenant_isolation.py).
"""

import base64
import email
import email.policy
import json
import time

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import captain_claw.flight_deck.gmail_send_routes as gs
import captain_claw.flight_deck.google_oauth_routes as gr
from captain_claw.flight_deck import agent_secret, auth
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.google_oauth import GoogleOAuthTokens

ALICE = "user-alice"
BOB = "user-bob"
_RealAsyncClient = httpx.AsyncClient


def _tokens(access):
    # Far-future expiry so _refresh_if_needed never makes a network call.
    return GoogleOAuthTokens(access_token=access, refresh_token=f"{access}-refresh",
                             token_type="Bearer", expires_at=time.time() + 3600,
                             scope="openid email https://www.googleapis.com/auth/gmail.compose")


@pytest.fixture()
async def db(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for var in ("FD_LOCKDOWN", "FD_PUBLIC_URL", "FD_AGENT_SHARED_SECRET", "FD_GMAIL_SEND"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    agent_secret.reset_cache_for_tests()
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev_db = auth._db
    auth.set_auth_db(d)
    now = "2026-01-01T00:00:00Z"
    for uid, email_, role in [(ALICE, "alice@x.co", "admin"), (BOB, "bob@x.co", "user")]:
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, email_, "h", uid.title(), role, now, now),
        )
    await d._db.commit()
    await d.set_system_setting(gr._K_CLIENT_ID, "cid")
    await d.set_system_setting(gr._K_CLIENT_SECRET, "csecret")
    gr._primary_owner_cache.update(id=ALICE, at=time.time())
    gs._policy_locks.clear()
    gs._send_locks.clear()
    try:
        yield d
    finally:
        gr._primary_owner_cache.update(id=None, at=0.0)
        gs._policy_locks.clear()
        gs._send_locks.clear()
        agent_secret.reset_cache_for_tests()
        auth.set_auth_db(prev_db)
        await d.close()


@pytest.fixture()
def agents(monkeypatch):
    """This deck's agent registry: web_auth token → (owner, slug). Stands in for
    both ``server._resolve_agent_identity_by_auth`` (the gate) and
    ``server._find_agent_by_auth`` (the audit label)."""
    import captain_claw.flight_deck.server as srv
    registry = {"tok-alice": (ALICE, "alice-agent"), "tok-bob": (BOB, "bob-agent")}
    monkeypatch.setattr(
        srv, "_resolve_agent_identity_by_auth",
        lambda t: (t in registry, registry.get(t, ("", ""))[0]) if t else (False, ""),
    )
    monkeypatch.setattr(
        srv, "_find_agent_by_auth",
        lambda t: (True, *registry[t]) if t in registry else (False, "", ""),
    )
    return registry


class _Gmail:
    """Scripted Gmail API; records every request (with its JSON body)."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        # (method, path) → an httpx transport error class to raise instead.
        self.raises: dict[tuple[str, str], type[httpx.TransportError]] = {}
        self.routes: dict[tuple[str, str], tuple[int, dict]] = {
            ("GET", "/messages/m1"): (200, {"id": "m1", "threadId": "t1", "payload": {"headers": [
                {"name": "From", "value": "Carol <carol@ok.com>"},
                {"name": "Subject", "value": "Plans"},
                {"name": "Message-ID", "value": "<orig@ok.com>"},
            ]}}),
            ("POST", "/messages/send"): (200, {"id": "sent-1", "threadId": "t-new"}),
            ("GET", "/drafts/d1"): (200, {"id": "d1", "message": {"id": "m9", "threadId": "td", "payload": {
                "headers": [{"name": "To", "value": "Dave <dave@ok.com>"},
                            {"name": "Subject", "value": "From the draft"}]}}}),
            ("POST", "/drafts/send"): (200, {"id": "sent-d", "threadId": "td"}),
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.split("/gmail/v1/users/me", 1)[-1]
        if (request.method, path) in self.raises:
            raise self.raises[(request.method, path)]("boom", request=request)
        status, body = self.routes.get((request.method, path), (404, {"error": {"message": "Not Found"}}))
        return httpx.Response(status, json=body)

    def sends(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "POST"]


@pytest.fixture()
def gmail(monkeypatch):
    g = _Gmail()
    monkeypatch.setattr(
        gs.httpx, "AsyncClient",
        lambda **kw: _RealAsyncClient(transport=httpx.MockTransport(g), **kw),
    )
    return g


def _client(host="127.0.0.1"):
    app = FastAPI()
    app.include_router(gr.router)
    app.include_router(gs.router)
    return TestClient(app, client=(host, 50123), follow_redirects=False)


def _bearer(uid, role="user"):
    return {"Authorization": f"Bearer {auth.create_access_token(uid, role)}"}


def _agent(token="tok-alice", **extra):
    return {"X-Agent-Auth": token, **extra}


def _decode(raw: str):
    data = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    return email.message_from_bytes(data, policy=email.policy.default)


async def _enable(db, uid=ALICE, **policy):
    await db.set_settings(uid, {gs._K_GMAIL_SEND: json.dumps(
        {"enabled": True, "allowed_recipients": [], "daily_limit": 50, **policy})})


MSG = {"to": "Bob <bob@x.co>", "subject": "Hello", "body": "Hi Bob"}


# ── the user's policy ────────────────────────────────────────────────


class TestPolicyRoutes:
    async def test_defaults_are_off(self, db):
        r = _client().get("/fd/google/gmail-send", headers=_bearer(ALICE))
        assert r.status_code == 200
        assert r.json() == {"enabled": False, "allowed_recipients": [], "daily_limit": 50,
                            "deck_disabled": False, "sent_last_24h": 0}

    async def test_needs_a_signed_in_user(self, db):
        c = _client()
        assert c.get("/fd/google/gmail-send").status_code == 401
        assert c.put("/fd/google/gmail-send", json={"enabled": True}).status_code == 401
        assert c.get("/fd/google/gmail-sends").status_code == 401

    @pytest.mark.parametrize("body", [
        {"daily_limit": 0}, {"daily_limit": 501}, {"daily_limit": "50"}, {"daily_limit": True},
        {"daily_limit": 2.5}, {"enabled": "yes"},
        {"allowed_recipients": ["not an address"]}, {"allowed_recipients": "a@b.co"},
        {"allowed_recipients": ["a@@b.co"]},
        {"allowed_recipients": [f"u{i}@ok.com" for i in range(201)]},
    ])
    async def test_put_rejects_bad_values(self, db, body):
        r = _client().put("/fd/google/gmail-send", headers=_bearer(ALICE), json=body)
        assert r.status_code == 400, r.text
        assert r.json()["detail"]
        assert (await gs.load_gmail_send_policy(db, ALICE)) == gs._default_policy()

    async def test_put_normalizes_persists_and_is_per_user(self, db):
        c = _client()
        r = c.put("/fd/google/gmail-send", headers=_bearer(ALICE), json={
            "enabled": True, "daily_limit": 5,
            "allowed_recipients": [" Alice@OK.com ", "ok.org", "@ok.org", "@Corp.IO"],
        })
        assert r.status_code == 200, r.text
        want = {"enabled": True, "allowed_recipients": ["alice@ok.com", "@ok.org", "@corp.io"],
                "daily_limit": 5, "deck_disabled": False, "sent_last_24h": 0}
        assert r.json() == want
        assert c.get("/fd/google/gmail-send", headers=_bearer(ALICE)).json() == want
        # A partial update leaves the rest alone.
        r = c.put("/fd/google/gmail-send", headers=_bearer(ALICE), json={"daily_limit": 7})
        assert r.json() == {**want, "daily_limit": 7}
        # Bob's policy is his own.
        assert c.get("/fd/google/gmail-send", headers=_bearer(BOB)).json()["enabled"] is False

    async def test_policy_survives_google_disconnect_and_is_server_owned(self, db):
        from captain_claw.flight_deck.settings_routes import _server_owned

        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db, daily_limit=9)
        await gr._clear_oauth_state(db, ALICE)
        assert await gr._load_tokens(db, ALICE) is None
        assert (await gs.load_gmail_send_policy(db, ALICE))["daily_limit"] == 9
        assert _server_owned(gs._K_GMAIL_SEND)  # /fd/settings can't read or write it

    async def test_corrupt_policy_reads_as_off(self, db):
        for raw in ("not json", json.dumps([1]),
                    json.dumps({"enabled": True, "daily_limit": 99999}),
                    json.dumps({"enabled": True, "allowed_recipients": ["bad entry"]})):
            await db.set_settings(ALICE, {gs._K_GMAIL_SEND: raw})
            assert await gs.load_gmail_send_policy(db, ALICE) == gs._default_policy()

    async def test_deck_kill_switch_is_reported(self, db, monkeypatch):
        monkeypatch.setenv("FD_GMAIL_SEND", "off")
        r = _client().get("/fd/google/gmail-send", headers=_bearer(ALICE))
        assert r.json()["deck_disabled"] is True


# ── the agent route: refusals ────────────────────────────────────────


class TestAgentSendRefusals:
    async def test_off_by_default(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        r = _client().post("/fd/google/gmail/send", headers=_agent(), json=MSG)
        assert r.status_code == 403
        assert r.headers.get("X-FD-Gmail-Send") == "off"
        detail = r.json()["detail"]
        assert "turned off" in detail and "create_draft" in detail and "Email sending" in detail
        assert gmail.requests == []

    @pytest.mark.parametrize("value", ["off", "0", "false", "NO"])
    async def test_deck_kill_switch_beats_the_user_policy(self, db, agents, gmail, monkeypatch, value):
        monkeypatch.setenv("FD_GMAIL_SEND", value)
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        r = _client().post("/fd/google/gmail/send", headers=_agent(), json=MSG)
        assert r.status_code == 403
        assert r.headers.get("X-FD-Gmail-Send") == "deck-disabled"
        assert "FD_GMAIL_SEND" in r.json()["detail"]
        assert gmail.requests == []

    @pytest.mark.parametrize("extra", [{"Origin": "https://evil.example"}, {"Sec-Fetch-Mode": "cors"}])
    async def test_browser_request_is_refused(self, db, agents, gmail, extra):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        r = _client().post("/fd/google/gmail/send", headers=_agent(**extra), json=MSG)
        assert r.status_code == 403 and "browsers" in r.json()["detail"]
        assert gmail.requests == []

    async def test_unknown_or_missing_agent_is_refused(self, db, agents, gmail):
        await _enable(db)
        c = _client()
        assert c.post("/fd/google/gmail/send", headers=_agent("other-deck"), json=MSG).status_code == 403
        assert c.post("/fd/google/gmail/send", json=MSG).status_code == 403
        assert gmail.requests == []

    async def test_off_host_caller_without_the_secret_is_refused(self, db, agents, gmail):
        await _enable(db)
        r = _client("10.1.2.3").post("/fd/google/gmail/send", headers=_agent(), json=MSG)
        assert r.status_code == 401

    async def test_owner_without_google_gets_404(self, db, agents, gmail):
        await _enable(db, uid=BOB)
        r = _client().post("/fd/google/gmail/send", headers=_agent("tok-bob"), json=MSG)
        assert r.status_code == 404 and "Google not connected" in r.json()["detail"]
        assert gmail.requests == []

    async def test_auth_off_deck_answers_like_access_token(self, db, agents, gmail, monkeypatch):
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        c = _client()
        r = c.post("/fd/google/gmail/send", headers=_agent(), json=MSG)
        assert r.status_code == 403 and "auth is disabled" in r.json()["detail"]
        assert r.headers.get(gr.AUTH_OFF_HEADER) == gr.AUTH_OFF_VALUE
        for path in ("/fd/google/gmail-send", "/fd/google/gmail-sends"):
            assert c.get(path).status_code == 503, path


# ── the agent route: sending ─────────────────────────────────────────


class TestAgentSend:
    async def test_sends_with_the_owners_token_audits_and_notifies(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await gr._store_tokens(db, BOB, _tokens("bob-access"))
        await _enable(db)
        r = _client().post("/fd/google/gmail/send", headers=_agent(), json=MSG)

        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "message_id": "sent-1", "thread_id": "t-new",
                            "to": "Bob <bob@x.co>", "cc": "", "bcc": "", "subject": "Hello",
                            "sent_last_24h": 1, "daily_limit": 50}
        (send,) = gmail.sends()
        assert send.url.path.endswith("/users/me/messages/send")
        assert send.headers["Authorization"] == "Bearer alice-access"
        payload = json.loads(send.content)
        assert "threadId" not in payload
        msg = _decode(payload["raw"])
        assert msg["To"] == "Bob <bob@x.co>" and msg["Subject"] == "Hello"

        (row,) = await db.list_gmail_sends(ALICE)
        assert row["agent"] == "alice-agent" and row["gmail_message_id"] == "sent-1"
        assert row["to_addrs"] == "Bob <bob@x.co>" and row["content_hash"]
        assert await db.list_gmail_sends(BOB) == []
        (note,) = await db.list_notifications(ALICE)
        assert note["type"] == "email_sent" and note["title"] == "alice-agent sent an email"
        assert note["body"] == "To: Bob <bob@x.co> - Subject: Hello"
        assert (note["ref_type"], note["ref_id"]) == ("gmail_message", "sent-1")

    async def test_reply_threads_and_defaults_like_create_draft(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        r = _client().post("/fd/google/gmail/send", headers=_agent(),
                           json={"reply_to_message_id": "m1", "body": "Count me in"})
        assert r.status_code == 200, r.text
        assert r.json()["to"] == "Carol <carol@ok.com>"
        assert r.json()["subject"] == "Re: Plans"
        payload = json.loads(gmail.sends()[0].content)
        assert payload["threadId"] == "t1"
        msg = _decode(payload["raw"])
        assert msg["In-Reply-To"] == "<orig@ok.com>" and msg["References"] == "<orig@ok.com>"

    async def test_allowlist_refusal_names_the_outsiders(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db, allowed_recipients=["@ok.com", "boss@corp.io"])
        c = _client()
        r = c.post("/fd/google/gmail/send", headers=_agent(), json={
            **MSG, "to": "a@ok.com, evil@bad.com", "bcc": "x@mail.ok.com"})
        assert r.status_code == 403
        detail = r.json()["detail"]
        assert "evil@bad.com" in detail and "x@mail.ok.com" in detail  # no subdomains
        assert "a@ok.com," not in detail
        assert "X-FD-Gmail-Send" not in r.headers
        assert gmail.sends() == []
        ok = c.post("/fd/google/gmail/send", headers=_agent(),
                    json={**MSG, "to": "a@ok.com", "cc": "Boss <BOSS@corp.io>"})
        assert ok.status_code == 200, ok.text

    async def test_daily_limit(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db, daily_limit=2)
        c = _client()
        for i in range(2):
            r = c.post("/fd/google/gmail/send", headers=_agent(), json={**MSG, "subject": f"s{i}"})
            assert r.status_code == 200 and r.json()["sent_last_24h"] == i + 1
        r = c.post("/fd/google/gmail/send", headers=_agent(), json={**MSG, "subject": "s3"})
        assert r.status_code == 429 and "2 of 2" in r.json()["detail"]
        assert len(gmail.sends()) == 2
        # Older than 24h no longer counts.
        await db._db.execute("UPDATE gmail_sends SET created_at = '2020-01-01T00:00:00+00:00'")
        await db._db.commit()
        assert c.post("/fd/google/gmail/send", headers=_agent(),
                      json={**MSG, "subject": "s4"}).status_code == 200

    async def test_duplicate_within_ten_minutes(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        c = _client()
        assert c.post("/fd/google/gmail/send", headers=_agent(), json=MSG).status_code == 200
        again = {**MSG, "to": "bob@X.co", "subject": "  hello "}  # same email, normalized
        r = c.post("/fd/google/gmail/send", headers=_agent(), json=again)
        assert r.status_code == 409
        assert "already sent" in r.json()["detail"] and "sent-1" in r.json()["detail"]
        assert len(gmail.sends()) == 1
        await db._db.execute("UPDATE gmail_sends SET created_at = '2020-01-01T00:00:00+00:00'")
        await db._db.commit()
        assert c.post("/fd/google/gmail/send", headers=_agent(), json=again).status_code == 200

    async def test_send_draft(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db, allowed_recipients=["@ok.com"])
        r = _client().post("/fd/google/gmail/send", headers=_agent(), json={"draft_id": "d1"})
        assert r.status_code == 200, r.text
        assert r.json()["to"] == "Dave <dave@ok.com>" and r.json()["subject"] == "From the draft"
        assert r.json()["message_id"] == "sent-d" and r.json()["thread_id"] == "td"
        (meta,) = [q for q in gmail.requests if q.method == "GET"]
        assert meta.url.params["format"] == "metadata"
        (send,) = gmail.sends()
        assert send.url.path.endswith("/users/me/drafts/send")
        assert json.loads(send.content) == {"id": "d1"}
        (row,) = await db.list_gmail_sends(ALICE)
        assert row["draft_id"] == "d1" and row["content_hash"] == ""

    async def test_draft_errors(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db, allowed_recipients=["@corp.io"])
        c = _client()
        r = c.post("/fd/google/gmail/send", headers=_agent(), json={"draft_id": "gone"})
        assert r.status_code == 404 and "list_drafts" in r.json()["detail"]
        r = c.post("/fd/google/gmail/send", headers=_agent(), json={"draft_id": "d1", "to": "x@y.co"})
        assert r.status_code == 400 and "not both" in r.json()["detail"]
        r = c.post("/fd/google/gmail/send", headers=_agent(), json={"draft_id": "../messages/m1"})
        assert r.status_code == 400
        r = c.post("/fd/google/gmail/send", headers=_agent(), json={"draft_id": "d1"})
        assert r.status_code == 403 and "dave@ok.com" in r.json()["detail"]  # draft vs allowlist
        assert gmail.sends() == []

    @pytest.mark.parametrize("body, needle", [
        ({"subject": "s", "body": "b"}, "No recipient"),
        ({"to": "bob@x.co", "body": "b"}, "subject and a body"),
        ({"to": "bob@x.co", "subject": "s"}, "subject and a body"),
        ({"to": "Bob", "subject": "s", "body": "b"}, "Invalid recipient"),
        ({"to": "a@x.co; b@x.co", "subject": "s", "body": "b"}, "Invalid recipient"),
        ({"to": ", ".join(f"p{i}@x.co" for i in range(21)), "subject": "s", "body": "b"},
         "Too many recipients"),
        ({"to": "bob@x.co", "subject": "s", "body": 5}, "body must be a string"),
    ])
    async def test_validation(self, db, agents, gmail, body, needle):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        r = _client().post("/fd/google/gmail/send", headers=_agent(), json=body)
        assert r.status_code == 400, r.text
        assert needle in r.json()["detail"]
        assert gmail.sends() == []

    async def test_gmail_scope_error_is_explained(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        gmail.routes[("POST", "/messages/send")] = (
            403, {"error": {"message": "Request had insufficient authentication scopes."}})
        r = _client().post("/fd/google/gmail/send", headers=_agent(), json=MSG)
        assert r.status_code == 403
        assert "gmail.compose" in r.json()["detail"] and "Scopes" in r.json()["detail"]
        assert await db.list_gmail_sends(ALICE) == []

    async def test_other_gmail_errors_are_502_with_gmails_message(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        gmail.routes[("POST", "/messages/send")] = (400, {"error": {"message": "Invalid To header"}})
        r = _client().post("/fd/google/gmail/send", headers=_agent(), json=MSG)
        assert r.status_code == 502 and "Invalid To header" in r.json()["detail"]
        gmail.routes[("POST", "/messages/send")] = (401, {"error": {"message": "expired"}})
        r = _client().post("/fd/google/gmail/send", headers=_agent(), json=MSG)
        assert r.status_code == 502 and "reconnect" in r.json()["detail"]

    async def test_a_failing_notification_does_not_fail_the_send(self, db, agents, gmail, monkeypatch):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)

        async def boom(*a, **kw):
            raise RuntimeError("bell broken")

        monkeypatch.setattr(db, "add_notification", boom)
        r = _client().post("/fd/google/gmail/send", headers=_agent(), json=MSG)
        assert r.status_code == 200 and r.json()["message_id"] == "sent-1"
        assert len(await db.list_gmail_sends(ALICE)) == 1


# ── review fixes ─────────────────────────────────────────────────────


class TestSendReviewFixes:
    @pytest.mark.parametrize("to", ["josé@example.com", "info@münchen.de"])
    async def test_non_ascii_address_is_a_400_not_a_500(self, db, agents, gmail, to):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        r = _client().post("/fd/google/gmail/send", headers=_agent(), json={**MSG, "to": to})
        assert r.status_code == 400, r.text
        detail = r.json()["detail"]
        assert to in detail and "ASCII" in detail and "xn--" in detail
        assert gmail.sends() == [] and await db.list_gmail_sends(ALICE) == []

    async def test_non_ascii_display_name_reads_plainly_and_encodes_on_the_wire(
        self, db, agents, gmail,
    ):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        r = _client().post("/fd/google/gmail/send", headers=_agent(),
                           json={**MSG, "to": "Željko Horvat <z@x.co>", "cc": '"Horvat, Ana" <a@x.co>'})
        assert r.status_code == 200, r.text
        assert r.json()["to"] == "Željko Horvat <z@x.co>"
        assert r.json()["cc"] == '"Horvat, Ana" <a@x.co>'
        (row,) = await db.list_gmail_sends(ALICE)
        assert row["to_addrs"] == "Željko Horvat <z@x.co>"
        (note,) = await db.list_notifications(ALICE)
        assert "=?utf-8?" not in note["body"] and "Željko Horvat" in note["body"]
        raw = json.loads(gmail.sends()[0].content)["raw"]
        wire = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
        assert b"=?utf-8?" in wire and "Željko".encode() not in wire  # RFC 2047 on the wire
        msg = _decode(raw)
        assert [(a.display_name, a.addr_spec) for a in msg["To"].addresses] == [
            ("Željko Horvat", "z@x.co")]
        assert [(a.display_name, a.addr_spec) for a in msg["Cc"].addresses] == [
            ("Horvat, Ana", "a@x.co")]

    async def test_same_reply_to_two_different_emails_is_not_a_duplicate(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        headers = [{"name": "From", "value": "bookings@hotel.com"},
                   {"name": "Subject", "value": "Your booking"}]
        gmail.routes[("GET", "/messages/m1")] = (200, {"id": "m1", "threadId": "t1", "payload": {
            "headers": headers + [{"name": "Message-ID", "value": "<b1@hotel.com>"}]}})
        gmail.routes[("GET", "/messages/m2")] = (200, {"id": "m2", "threadId": "t2", "payload": {
            "headers": headers + [{"name": "Message-ID", "value": "<b2@hotel.com>"}]}})
        c = _client()
        for mid in ("m1", "m2"):
            r = c.post("/fd/google/gmail/send", headers=_agent(),
                       json={"reply_to_message_id": mid, "body": "Confirmed, thanks"})
            assert r.status_code == 200, (mid, r.text)
        assert [json.loads(q.content)["threadId"] for q in gmail.sends()] == ["t1", "t2"]
        # Retrying the same reply is still caught.
        r = c.post("/fd/google/gmail/send", headers=_agent(),
                   json={"reply_to_message_id": "m2", "body": "Confirmed, thanks"})
        assert r.status_code == 409 and "already sent" in r.json()["detail"]
        assert len(gmail.sends()) == 2

    async def test_reply_to_own_sent_email_goes_to_its_recipients(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        gmail.routes[("GET", "/messages/mine")] = (200, {
            "id": "mine", "threadId": "t1", "labelIds": ["SENT"], "payload": {"headers": [
                {"name": "From", "value": "Alice <alice@x.co>"},
                {"name": "To", "value": "Bob <bob@ok.com>"},
                {"name": "Subject", "value": "Q3"},
                {"name": "Message-ID", "value": "<mine@x.co>"},
            ]}})
        r = _client().post("/fd/google/gmail/send", headers=_agent(),
                           json={"reply_to_message_id": "mine", "body": "Following up"})
        assert r.status_code == 200, r.text
        assert r.json()["to"] == "Bob <bob@ok.com>"
        assert _decode(json.loads(gmail.sends()[0].content)["raw"])["To"] == "Bob <bob@ok.com>"

    @pytest.mark.parametrize("failure", ["timeout", "5xx"])
    async def test_unknown_outcome_is_recorded_and_blocks_a_blind_retry(
        self, db, agents, gmail, failure,
    ):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        if failure == "timeout":
            gmail.raises[("POST", "/messages/send")] = httpx.ReadTimeout
        else:
            gmail.routes[("POST", "/messages/send")] = (503, {"error": {"message": "Backend Error"}})
        c = _client()
        r = c.post("/fd/google/gmail/send", headers=_agent(), json=MSG)
        assert r.status_code == 502
        assert r.headers.get("X-FD-Gmail-Send-Outcome") == "unknown"
        assert "may or may not have been sent" in r.json()["detail"]
        assert "in:sent" in r.json()["detail"]

        (row,) = await db.list_gmail_sends(ALICE)
        assert row["status"] == "unknown" and row["gmail_message_id"] == ""
        assert row["content_hash"] and row["to_addrs"] == "Bob <bob@x.co>"
        (note,) = await db.list_notifications(ALICE)
        assert note["title"] == "alice-agent may have sent an email"
        assert "Sent folder" in note["body"]
        sends = c.get("/fd/google/gmail-sends", headers=_bearer(ALICE)).json()["sends"]
        assert [s["status"] for s in sends] == ["unknown"]
        assert c.get("/fd/google/gmail-send", headers=_bearer(ALICE)).json()["sent_last_24h"] == 1

        # The agent retries: refused as a duplicate, Gmail not called again.
        gmail.raises.clear()
        gmail.routes[("POST", "/messages/send")] = (200, {"id": "sent-1", "threadId": "t"})
        r = c.post("/fd/google/gmail/send", headers=_agent(), json=MSG)
        assert r.status_code == 409
        assert "never confirmed" in r.json()["detail"] and "in:sent" in r.json()["detail"]
        assert len(gmail.sends()) == 1

    async def test_unknown_outcome_of_a_draft_send_counts_toward_the_limit(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db, daily_limit=1)
        gmail.raises[("POST", "/drafts/send")] = httpx.RemoteProtocolError
        c = _client()
        r = c.post("/fd/google/gmail/send", headers=_agent(), json={"draft_id": "d1"})
        assert r.status_code == 502 and r.headers.get("X-FD-Gmail-Send-Outcome") == "unknown"
        (row,) = await db.list_gmail_sends(ALICE)
        assert (row["status"], row["draft_id"]) == ("unknown", "d1")
        r = c.post("/fd/google/gmail/send", headers=_agent(), json={**MSG, "subject": "other"})
        assert r.status_code == 429

    async def test_connect_failure_on_send_is_not_sent_and_not_recorded(self, db, agents, gmail):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        gmail.raises[("POST", "/messages/send")] = httpx.ConnectError
        r = _client().post("/fd/google/gmail/send", headers=_agent(), json=MSG)
        assert r.status_code == 502 and "was not sent" in r.json()["detail"]
        assert "X-FD-Gmail-Send-Outcome" not in r.headers
        assert await db.list_gmail_sends(ALICE) == []
        assert await db.list_notifications(ALICE) == []

    @pytest.mark.parametrize("route, body", [
        (("GET", "/drafts/d1"), {"draft_id": "d1"}),
        (("GET", "/messages/m1"), {"reply_to_message_id": "m1", "body": "x"}),
    ])
    @pytest.mark.parametrize("failure", ["timeout", "5xx"])
    async def test_a_failed_lookup_before_the_send_says_not_sent(
        self, db, agents, gmail, route, body, failure,
    ):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await _enable(db)
        if failure == "timeout":
            gmail.raises[route] = httpx.ReadTimeout
        else:
            gmail.routes[route] = (500, {"error": {"message": "Backend Error"}})
        r = _client().post("/fd/google/gmail/send", headers=_agent(), json=body)
        assert r.status_code == 502, r.text
        detail = r.json()["detail"]
        assert "was not sent" in detail and "may or may not" not in detail
        assert "X-FD-Gmail-Send-Outcome" not in r.headers
        assert gmail.sends() == [] and await db.list_gmail_sends(ALICE) == []


# ── history ──────────────────────────────────────────────────────────


class TestSendHistory:
    async def test_lists_only_own_rows_newest_first(self, db):
        rows = [(ALICE, "old", "2026-01-01T00:00:00+00:00"),
                (BOB, "bob's", "2026-01-02T00:00:00+00:00"),
                (ALICE, "new", "2026-01-03T00:00:00+00:00")]
        for owner, subject, at in rows:
            sid = await db.add_gmail_send(owner, agent="a", to_addrs="x@y.co", subject=subject,
                                          gmail_message_id=f"g-{subject}")
            await db._db.execute("UPDATE gmail_sends SET created_at = ? WHERE id = ?", (at, sid))
        await db._db.commit()

        c = _client()
        sends = c.get("/fd/google/gmail-sends", headers=_bearer(ALICE)).json()["sends"]
        assert [s["subject"] for s in sends] == ["new", "old"]
        assert set(sends[0]) == {"id", "status", "agent", "to", "cc", "bcc", "subject",
                                 "gmail_message_id", "thread_id", "draft_id", "created_at"}
        assert sends[0]["status"] == "sent"
        assert sends[0]["gmail_message_id"] == "g-new" and sends[0]["to"] == "x@y.co"
        one = c.get("/fd/google/gmail-sends?limit=0", headers=_bearer(ALICE)).json()["sends"]
        assert [s["subject"] for s in one] == ["new"]  # clamped to 1
        bobs = c.get("/fd/google/gmail-sends?limit=500", headers=_bearer(BOB)).json()["sends"]
        assert [s["subject"] for s in bobs] == ["bob's"]
