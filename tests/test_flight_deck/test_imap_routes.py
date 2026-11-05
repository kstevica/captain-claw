"""Email (IMAP) on Flight Deck — /fd/imap: a user's inboxes, the agent routes
and the send gate.

No mail server: the IMAP/SMTP core (captain_claw.imap_mail) is replaced by
stubs at the functions the routes call, so these tests pin what Flight Deck
decides — who may use which inbox, what is stored (passwords sealed, never
shown), and when a send is refused, counted, audited and rung. The core itself
is tested in tests/test_imap_mail.py (and live in test_imap_mail_live.py).
"""

import email
import email.policy
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import captain_claw.flight_deck.imap_routes as ir
from captain_claw import imap_mail
from captain_claw.flight_deck import agent_secret, auth, fd_crypto
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.settings_routes import _server_owned

ALICE = "user-alice"
BOB = "user-bob"


@pytest.fixture()
async def db(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for var in ("FD_LOCKDOWN", "FD_AGENT_SHARED_SECRET", "FD_IMAP_SEND", "FD_IMAP_REPEAT_CHECK_DAYS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    monkeypatch.setenv("FD_DATA_KEY_FILE", str(tmp_path / "data_key"))
    fd_crypto.reset_for_tests()
    agent_secret.reset_cache_for_tests()
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev = auth._db
    auth.set_auth_db(d)
    now = "2026-01-01T00:00:00Z"
    for uid, mail, role in [(ALICE, "alice@x.co", "admin"), (BOB, "bob@x.co", "user")]:
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)", (uid, mail, "h", uid.title(), role, now, now))
    await d._db.commit()
    for store in (ir._locks, ir._send_locks, ir._conn_limits):
        store.clear()
    try:
        yield d
    finally:
        auth.set_auth_db(prev)
        fd_crypto.reset_for_tests()
        agent_secret.reset_cache_for_tests()
        await d.close()


@pytest.fixture()
def agents(monkeypatch):
    import captain_claw.flight_deck.server as srv
    registry = {"tok-alice": (ALICE, "mailer"), "tok-alice-2": (ALICE, "other"), "tok-bob": (BOB, "bob-agent")}
    monkeypatch.setattr(srv, "_resolve_agent_identity_by_auth",
                        lambda t: (t in registry, registry.get(t, ("", ""))[0]) if t else (False, ""))
    monkeypatch.setattr(srv, "_find_agent_by_auth",
                        lambda t: (True, *registry[t]) if t in registry else (False, "", ""))
    return registry


@pytest.fixture()
def core(monkeypatch):
    """The IMAP/SMTP core, scripted. ``core.test`` decides whether a login works."""
    class Core:
        test = {"ok": True, "imap": {"ok": True, "folders": 5, "roles": ["inbox", "sent"]}, "smtp": {"ok": True}}
        delivered: list = []
        deliver_exc: Exception | None = None
        listed: list = []

    c = Core()
    c.delivered = []
    c.listed = []

    def test_account(acct):
        c.tested = acct
        return c.test

    def smtp_deliver(acct, rcpts, data):
        if c.deliver_exc:
            raise c.deliver_exc
        c.delivered.append((acct.email, rcpts, data))
        return "250 queued"

    def compose(acct, body, signature, atts, *, for_send, allowlist, save):
        what = "Nothing was sent." if for_send else "No draft was created."
        to = ir._recips(body.get("to"))
        ir._check_addresses(to, "", "", allowlist=allowlist, what=what)
        text, html = ir.mail_signature.apply(str(body.get("body") or ""), "", signature)
        data, mid = imap_mail.build_message(acct, to=to, cc="", bcc="", subject=body.get("subject") or "",
                                            body=text, html_body=html, attachments=atts or None)
        return ir._Composed(data, mid, to, "", "", body.get("subject") or "", text, html, None, atts), \
            ("work:1:9:Drafts" if save else "")

    def list_messages(acct, **kw):
        c.listed.append(acct.id)
        return {"folder": "INBOX", "total": 1, "messages": [
            {"id": f"{acct.id}:1:1:INBOX", "subject": f"hi from {acct.id}", "ts": len(c.listed)}]}

    monkeypatch.setattr(imap_mail, "test_account", test_account)
    monkeypatch.setattr(imap_mail, "smtp_deliver", smtp_deliver)
    monkeypatch.setattr(imap_mail, "after_send", lambda *a, **k: [])
    monkeypatch.setattr(imap_mail, "list_messages", list_messages)
    monkeypatch.setattr(ir, "_compose_work", compose)
    return c


def _client():
    app = FastAPI()
    app.include_router(ir.router)
    return TestClient(app, client=("127.0.0.1", 50123), follow_redirects=False)


def _bearer(uid, role="user"):
    return {"Authorization": f"Bearer {auth.create_access_token(uid, role)}"}


def _agent(token="tok-alice"):
    return {"X-Agent-Auth": token}


WORK = {"label": "Work", "email": "ana@firma.hr", "password": "s3cret!", "imap_host": "imap.firma.hr",
        "smtp_host": "smtp.firma.hr"}
SUPPORT = {"label": "Support", "email": "support@firma.hr", "password": "pw2", "imap_host": "imap.firma.hr",
           "smtp_host": "smtp.firma.hr"}


def _add(c, body=None, uid=ALICE):
    r = c.post("/fd/imap/accounts", json=body or WORK, headers=_bearer(uid))
    assert r.status_code == 200, r.text
    return r.json()["account"]


# ── the user's inboxes ──────────────────────────────────────────────


async def test_add_stores_password_sealed_and_never_shows_it(db, core):
    c = _client()
    acct = _add(c)
    assert acct["id"] == "work" and acct["sending"]["enabled"] is False
    assert "password" not in acct and "smtp_password" not in acct
    raw = await db.get_setting(ALICE, ir._K_ACCOUNTS)
    assert "s3cret!" not in raw
    stored = json.loads(raw)[0]
    assert fd_crypto.unseal(stored["password"]) == "s3cret!"
    listed = c.get("/fd/imap/accounts", headers=_bearer(ALICE)).json()
    assert "s3cret" not in json.dumps(listed)
    # Each user sees only their own.
    assert c.get("/fd/imap/accounts", headers=_bearer(BOB)).json()["accounts"] == []


async def test_add_refused_when_login_fails_and_nothing_kept(db, core):
    core.test = {"ok": False, "imap": {"ok": False, "error": "refused the login"}, "smtp": {"ok": True}}
    r = _client().post("/fd/imap/accounts", json=WORK, headers=_bearer(ALICE))
    assert r.status_code == 400 and "refused the login" in r.json()["detail"]
    assert await db.get_setting(ALICE, ir._K_ACCOUNTS) is None


async def test_several_inboxes_unique_ids_and_no_duplicates(db, core):
    c = _client()
    assert _add(c)["id"] == "work"
    assert _add(c, {**WORK, "email": "ana2@firma.hr"})["id"] == "work-2"
    assert _add(c, SUPPORT)["id"] == "support"
    dup = c.post("/fd/imap/accounts", json=WORK, headers=_bearer(ALICE))
    assert dup.status_code == 409


async def test_bad_settings_are_400(db, core):
    c = _client()
    for body in ({**WORK, "imap_security": "none"}, {**WORK, "password": ""}, {**WORK, "imap_port": 0}):
        assert c.post("/fd/imap/accounts", json=body, headers=_bearer(ALICE)).status_code == 400


async def test_update_keeps_password_when_blank_and_retests_on_change(db, core):
    c = _client()
    _add(c)
    r = c.put("/fd/imap/accounts/work", json={"imap_host": "mail.firma.hr", "password": ""}, headers=_bearer(ALICE))
    assert r.status_code == 200 and r.json()["account"]["imap_host"] == "mail.firma.hr"
    assert core.tested.password == "s3cret!"  # the stored one was used for the check
    core.test = {"ok": False, "imap": {"ok": False, "error": "nope"}, "smtp": {"ok": True}}
    bad = c.put("/fd/imap/accounts/work", json={"password": "wrong"}, headers=_bearer(ALICE))
    assert bad.status_code == 400
    stored = json.loads(await db.get_setting(ALICE, ir._K_ACCOUNTS))[0]
    assert fd_crypto.unseal(stored["password"]) == "s3cret!"


async def test_sending_policy_validation(db, core):
    c = _client()
    _add(c)
    assert c.put("/fd/imap/accounts/work", json={"sending": {"daily_limit": 0}}, headers=_bearer(ALICE)).status_code == 400
    assert c.put("/fd/imap/accounts/work", json={"sending": {"allowed_recipients": ["not an address"]}},
                 headers=_bearer(ALICE)).status_code == 400
    r = c.put("/fd/imap/accounts/work", json={"sending": {"enabled": True, "allowed_recipients": ["firma.hr"]},
                                              "signature": "— Ana's assistant"}, headers=_bearer(ALICE))
    assert r.json()["account"]["sending"]["allowed_recipients"] == ["@firma.hr"]
    assert r.json()["account"]["signature"] == "— Ana's assistant"


async def test_delete(db, core):
    c = _client()
    _add(c)
    assert c.delete("/fd/imap/accounts/work", headers=_bearer(BOB)).status_code == 404
    assert c.delete("/fd/imap/accounts/work", headers=_bearer(ALICE)).status_code == 200
    assert c.get("/fd/imap/accounts", headers=_bearer(ALICE)).json()["accounts"] == []


def test_settings_routes_cannot_touch_inboxes():
    assert _server_owned("imap_mail:accounts")


# ── the agent routes ────────────────────────────────────────────────


async def test_agent_status_lists_only_the_owners_inboxes_this_agent_may_use(db, core, agents):
    c = _client()
    _add(c)
    _add(c, SUPPORT)
    c.put("/fd/imap/accounts/support", json={"agents": ["other"]}, headers=_bearer(ALICE))
    mine = c.get("/fd/imap/agent_status", headers=_agent()).json()
    assert [a["id"] for a in mine["accounts"]] == ["work"]
    other = c.get("/fd/imap/agent_status", headers=_agent("tok-alice-2")).json()
    assert {a["id"] for a in other["accounts"]} == {"work", "support"}
    bob = c.get("/fd/imap/agent_status", headers=_agent("tok-bob")).json()
    assert bob == {"connected": False, "accounts": []}
    # An inbox this agent may not use is refused even by ID.
    r = c.post("/fd/imap/agent/read_message", json={"message_id": "support:1:1:INBOX"}, headers=_agent())
    assert r.status_code == 404 and "can't use" in r.json()["detail"]


async def test_agent_routes_refuse_browsers_and_unknown_agents(db, core, agents):
    c = _client()
    _add(c)
    assert c.get("/fd/imap/agent_status", headers={**_agent(), "Origin": "https://evil.test"}).status_code == 403
    assert c.get("/fd/imap/agent_status", headers=_agent("tok-forged")).status_code == 403
    assert c.get("/fd/imap/agent_status").status_code == 403


async def test_which_inbox(db, core, agents):
    c = _client()
    _add(c)
    _add(c, SUPPORT)
    r = c.post("/fd/imap/agent/create_draft", json={"to": "x@y.co", "body": "hi"}, headers=_agent())
    assert r.status_code == 400 and "Which inbox?" in r.json()["detail"]
    assert "work" in r.json()["detail"] and "support" in r.json()["detail"]
    r = c.post("/fd/imap/agent/create_draft", json={"account": "Support", "to": "x@y.co", "body": "hi"},
               headers=_agent())
    assert r.status_code == 200 and r.json()["account"] == "support"
    # Reading with no account reads every inbox, newest first.
    r = c.post("/fd/imap/agent/list_messages", json={}, headers=_agent()).json()
    assert set(r["accounts"]) == {"work", "support"} and len(r["messages"]) == 2
    r = c.post("/fd/imap/agent/send", json={"account": "all", "to": "x@y.co", "subject": "s", "body": "b"},
               headers=_agent())
    assert r.status_code == 400


# ── the send gate ───────────────────────────────────────────────────


MAIL = {"account": "work", "to": "Bob <bob@klijent.hr>", "subject": "Ponuda", "body": "Pozdrav"}


async def _sending(c, **policy):
    r = c.put("/fd/imap/accounts/work", json={"sending": {"enabled": True, **policy}}, headers=_bearer(ALICE))
    assert r.status_code == 200


async def test_send_off_by_default(db, core, agents):
    c = _client()
    _add(c)
    r = c.post("/fd/imap/agent/send", json=MAIL, headers=_agent())
    assert r.status_code == 403 and "turned off" in r.json()["detail"] and "create_draft" in r.json()["detail"]
    assert core.delivered == []


async def test_send_goes_out_audited_and_rung(db, core, agents):
    c = _client()
    _add(c, {**WORK, "display_name": "Ana Horvat"})
    await _sending(c)
    c.put("/fd/imap/accounts/work", json={"signature": "— Ana's assistant"}, headers=_bearer(ALICE))
    r = c.post("/fd/imap/agent/send", json=MAIL, headers=_agent())
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["from"] == "ana@firma.hr" and out["sent_last_24h"] == 1
    sender, rcpts, data = core.delivered[0]
    assert rcpts == ["bob@klijent.hr"]
    msg = email.message_from_bytes(data, policy=email.policy.default)
    assert msg["From"] == "Ana Horvat <ana@firma.hr>"
    assert "— Ana's assistant" in msg.get_body(("plain",)).get_content()
    rows = await db.list_imap_sends(ALICE, 10, "work")
    assert len(rows) == 1 and rows[0]["agent"] == "mailer" and rows[0]["status"] == "sent"
    notes = await db.list_notifications(ALICE, 10)
    assert any("mailer sent an email from ana@firma.hr" in n["title"] for n in notes)
    hist = c.get("/fd/imap/sends", params={"account_id": "work"}, headers=_bearer(ALICE)).json()["sends"]
    assert hist[0]["to"] == "Bob <bob@klijent.hr>"


async def test_duplicate_within_minutes_refused(db, core, agents):
    c = _client()
    _add(c)
    await _sending(c)
    assert c.post("/fd/imap/agent/send", json=MAIL, headers=_agent()).status_code == 200
    again = c.post("/fd/imap/agent/send", json=MAIL, headers=_agent())
    assert again.status_code == 409 and "Duplicate" in again.json()["detail"]
    assert len(core.delivered) == 1


async def test_allowlist(db, core, agents):
    c = _client()
    _add(c)
    await _sending(c, allowed_recipients=["@firma.hr"])
    r = c.post("/fd/imap/agent/send", json=MAIL, headers=_agent())
    assert r.status_code == 403 and "allowed-recipients" in r.json()["detail"]
    ok = c.post("/fd/imap/agent/send", json={**MAIL, "to": "kolega@firma.hr"}, headers=_agent())
    assert ok.status_code == 200


async def test_daily_limit(db, core, agents):
    c = _client()
    _add(c)
    await _sending(c, daily_limit=1)
    assert c.post("/fd/imap/agent/send", json=MAIL, headers=_agent()).status_code == 200
    r = c.post("/fd/imap/agent/send", json={**MAIL, "subject": "Other"}, headers=_agent())
    assert r.status_code == 429 and "1 of 1" in r.json()["detail"]


async def test_unknown_outcome_counted_and_caught_as_duplicate(db, core, agents):
    c = _client()
    _add(c)
    await _sending(c)
    core.deliver_exc = imap_mail.SendOutcomeUnknown("never confirmed — may have been sent")
    r = c.post("/fd/imap/agent/send", json=MAIL, headers=_agent())
    assert r.status_code == 504
    rows = await db.list_imap_sends(ALICE, 10, "work")
    assert rows[0]["status"] == "unknown"
    core.deliver_exc = None
    again = c.post("/fd/imap/agent/send", json=MAIL, headers=_agent())
    assert again.status_code == 409 and "may have gone out" in again.json()["detail"]


async def test_deck_kill_switch_and_receive_only(db, core, agents, monkeypatch):
    c = _client()
    _add(c)
    _add(c, {**SUPPORT, "smtp_host": ""})
    await _sending(c)
    monkeypatch.setenv("FD_IMAP_SEND", "off")
    r = c.post("/fd/imap/agent/send", json=MAIL, headers=_agent())
    assert r.status_code == 403 and "FD_IMAP_SEND" in r.json()["detail"]
    monkeypatch.delenv("FD_IMAP_SEND")
    r = c.post("/fd/imap/agent/send", json={**MAIL, "account": "support"}, headers=_agent())
    assert r.status_code == 403 and "receive-only" in r.json()["detail"]
    assert core.delivered == []


async def test_password_unreadable_after_key_change(db, core, agents, tmp_path, monkeypatch):
    c = _client()
    _add(c)
    monkeypatch.setenv("FD_DATA_KEY_FILE", str(tmp_path / "another_key"))
    fd_crypto.reset_for_tests()
    r = c.post("/fd/imap/agent/list_messages", json={"account": "work"}, headers=_agent())
    assert r.status_code == 502 and "re-enters" in r.json()["detail"]


def test_ids_from_two_inboxes_in_one_call_refused():
    recs = [{"id": "work"}, {"id": "support"}]
    with pytest.raises(Exception) as exc:
        ir._choose("archive", {"message_ids": ["work:1:1:INBOX", "support:1:1:INBOX"]}, recs)
    assert "different inboxes" in str(getattr(exc.value, "detail", exc.value))


