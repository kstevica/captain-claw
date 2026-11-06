"""Email for an agent: its own inbox, or the owner's mail that names it, becomes
a task assigned to it and a turn in its line — drafts only, except a send from
its own inbox, which that inbox's own gate still decides."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from captain_claw.flight_deck import agent_mail, colleagues, imap_routes, taskboard_triage
from captain_claw.flight_deck import taskboard_handover as handover
from captain_claw.flight_deck.agent_mail import Email

from .test_colleagues import TEAM, _settle, relay, team  # noqa: F401
from .test_taskboard_nextsteps import ALICE, board, jev, ledger, model, told  # noqa: F401


@pytest.fixture()
def mail(team, monkeypatch):  # noqa: F811
    """Researcher picks up email (as Rita, with an alias), Writer doesn't; IMAP inboxes and sources stubbed."""
    team.save_identity(ALICE, "researcher", colleagues.clean_identity(
        {"role": "Researcher", "names": ["Rita"], "addresses": ["alice+rita@gmail.com"], "pick_up_mail": True,
         "inbox": "rita-box"}))
    team.save_identity(ALICE, "writer", colleagues.clean_identity({"names": ["Writer"], "pick_up_mail": False}))
    state = SimpleNamespace(imap={}, gmail=None, outlook=None, recs=[
        {"id": "rita-box", "email": "rita@firm.co", "agents": ["researcher"], "smtp_host": "smtp.firm.co",
         "sending": {"enabled": True}},
        {"id": "main", "email": "alice@firm.co", "agents": [], "sending": {"enabled": False}}])

    async def inboxes(owner):
        return {r["id"]: r["email"] for r in state.recs}

    async def load(db, uid):
        return [dict(r) for r in state.recs]

    monkeypatch.setattr(colleagues, "_inboxes", inboxes)
    monkeypatch.setattr(imap_routes, "_load", load)
    import captain_claw.flight_deck.auth as auth_mod

    monkeypatch.setattr(auth_mod, "get_db", lambda: None)

    def imap_read(rec, cursor):
        got = state.imap.pop(rec["id"], [])
        return got, {"uidvalidity": 1, "last_uid": len(got) + int(cursor.get("last_uid") or 0)}

    async def gmail_read(owner, cursor):
        got, state.gmail = state.gmail, None
        return (got or [], {"after": 1.0}) if got is not None else ([], {"after": 1.0})

    async def outlook_read(owner, cursor):
        return None

    monkeypatch.setattr(agent_mail, "_imap_read", imap_read)
    monkeypatch.setattr(agent_mail, "_gmail_read", gmail_read)
    monkeypatch.setattr(agent_mail, "_outlook_read", outlook_read)
    return state


def _gmail(ref="m1", subject="Rita: Q3 numbers please", body="Can you pull the Q3 numbers?", sender="ana@client.co",
           thread="t1", **kw) -> Email:
    return Email(source="gmail", ref=ref, thread=thread, subject=subject, body=body, sender=sender,
                 sender_name="Ana", to=("alice@gmail.com",), reply_ref=ref, **kw)


def _imap(account, ref="rita-box:1:1:INBOX", subject="Hello", body="Hi there", sender="bob@supplier.co",
          to=("rita@firm.co",), **kw) -> Email:
    return Email(source="imap", ref=ref, thread=f"{account}:<root@x>", subject=subject, body=body, sender=sender,
                 sender_name="Bob", to=to, account=account, reply_ref=ref, **kw)


async def test_an_email_naming_the_agent_becomes_its_task_and_turn(board, ledger, relay, mail):  # noqa: F811
    mail.gmail = [_gmail()]
    assert await agent_mail.owner_pass(ALICE) == 1
    await _settle()
    call = relay.calls[0]
    assert call["agent"] == "researcher"
    auto = call["automation"]
    assert auto["kind"] == "inbox" and auto["mail_write"] == "draft" and "send_from" not in auto
    assert auto["job_text"] == "" and auto["task"]["board"] is True
    head, _, rest = call["prompt"].partition("<<<EMAIL")
    assert "with your name in the subject" in head and "google_mail action=create_draft" in head
    assert "reply_to_message_id=m1" in head and "Don't send" in head
    assert "Can you pull the Q3 numbers?" in rest and "ana@client.co" in rest
    task = board.get_task(ALICE, auto["task"]["id"])
    assert task["agent_slug"] == "researcher" and task["channel"] == "gmail" and task["title"] == "Rita: Q3 numbers please"
    assert json.loads(task["flags"])["agent_mail"]["why"] == "subject"
    keys = {k["key"] for k in board.links(ALICE, [task["id"]])[task["id"]]}
    assert {"gmail:thread:t1", "gmail:msg:m1"} <= keys
    said = [e["summary"] for e in board.events(ALICE, task["id"])]
    assert "Email for Researcher (named in the subject): Rita: Q3 numbers please" in said
    assert task["owes_next"] == "me", "after the agent's turn the owner looks it over"


async def test_its_own_inbox_may_send_from_there(board, ledger, relay, mail):  # noqa: F811
    mail.imap["rita-box"] = [_imap("rita-box")]
    assert await agent_mail.owner_pass(ALICE) == 1
    await _settle()
    auto = relay.calls[0]["automation"]
    assert auto["send_from"] == "rita-box" and auto["mail_write"] == "draft"
    assert "in your own inbox (rita@firm.co)" in relay.calls[0]["prompt"]
    assert "imap_mail action=send account=rita-box" in relay.calls[0]["prompt"]


async def test_own_inbox_with_sending_off_stays_drafts(board, ledger, relay, mail):  # noqa: F811
    mail.recs[0]["sending"] = {"enabled": False}
    mail.imap["rita-box"] = [_imap("rita-box")]
    await agent_mail.owner_pass(ALICE)
    await _settle()
    auto = relay.calls[0]["automation"]
    assert "send_from" not in auto and "imap_mail action=create_draft account=rita-box" in relay.calls[0]["prompt"]


async def test_an_alias_in_the_owners_inbox_routes_by_address(board, ledger, relay, mail):  # noqa: F811
    mail.imap["main"] = [_imap("main", ref="main:1:1:INBOX", subject="Numbers", body="see attached",
                               to=("alice+rita@gmail.com",))]
    await agent_mail.owner_pass(ALICE)
    await _settle()
    call = relay.calls[0]
    assert call["agent"] == "researcher" and "to your address alice+rita@gmail.com" in call["prompt"]
    assert "send_from" not in call["automation"], "not its own inbox: drafts"


async def test_nobodys_mail_and_automatic_mail_go_nowhere(board, ledger, relay, mail):  # noqa: F811
    mail.gmail = [
        _gmail(ref="n1", subject="Lunch?", body="Are you free?"),
        _gmail(ref="n2", sender="no-reply@bank.co"),
        _gmail(ref="n3", headers={"list-id": "<news.x>"}),
        _gmail(ref="n4", sender="rita@firm.co"),  # what its own address wrote
        _gmail(ref="n5", subject="Writer: copy", body="x"),  # Writer doesn't pick up email
    ]
    assert await agent_mail.owner_pass(ALICE) == 0
    await _settle()
    assert relay.calls == []
    assert colleagues.get_store().mail_seen(ALICE, "gmail", ["n1", "n2", "n3", "n4", "n5"]) == {"n1", "n2", "n3", "n4", "n5"}


async def test_an_email_is_taken_once(board, ledger, relay, mail):  # noqa: F811
    mail.gmail = [_gmail()]
    await agent_mail.owner_pass(ALICE)
    mail.gmail = [_gmail()]
    assert await agent_mail.owner_pass(ALICE) == 0
    await _settle()
    assert len(relay.calls) == 1


async def test_a_reply_on_a_thread_already_with_the_agent_doesnt_queue_again(board, ledger, relay, mail, monkeypatch):  # noqa: F811
    gate = asyncio.Event()
    real = handover._relay

    async def slow(agent, prompt, automation):
        await gate.wait()
        return await real(agent, prompt, automation)

    monkeypatch.setattr(handover, "_relay", slow)
    mail.gmail = [_gmail()]
    await agent_mail.owner_pass(ALICE)
    mail.gmail = [_gmail(ref="m2", subject="Re: Rita: Q3 numbers please", body="Also Q2, Rita, please")]
    assert await agent_mail.owner_pass(ALICE) == 0
    gate.set()
    await _settle()
    assert len(relay.calls) == 1
    task_id = relay.calls[0]["automation"]["task"]["id"]
    assert any("Also Q2" in e["summary"] for e in board.events(ALICE, task_id)), "the new email is on its timeline"


@pytest.fixture()
def registry(monkeypatch):
    from captain_claw.flight_deck import server

    reg = {a["slug"]: {"name": a["name"], "owner": ALICE} for a in TEAM}
    reg["strays"] = {"name": "Nobody's", "owner": ""}
    monkeypatch.setattr(server, "_load_process_registry", lambda: reg)
    return reg


async def test_every_agent_picks_up_its_email_unless_switched_off(board, ledger, relay, mail, team, registry):  # noqa: F811
    assert agent_mail._owners() == [ALICE], "on by default; an agent nobody owns is nobody's to read for"
    for slug in ("researcher", "editor", "checker"):
        team.save_identity(ALICE, slug, {**(team.identity(ALICE, slug) or {}), "pick_up_mail": False})
    assert agent_mail._owners() == [], "Writer was switched off by the fixture, the rest here"
    mail.gmail = [_gmail()]
    assert await agent_mail.run_pass() == 0


async def test_an_agent_with_no_identity_answers_to_its_name(board, ledger, relay, mail):  # noqa: F811
    mail.gmail = [_gmail(ref="e1", subject="Editor: fix the intro", body="Thanks")]
    assert await agent_mail.owner_pass(ALICE) == 1
    await _settle()
    assert relay.calls[0]["agent"] == "editor"


async def test_it_can_be_switched_off(monkeypatch, mail):
    monkeypatch.setenv("FD_AGENT_MAIL", "off")
    mail.gmail = [_gmail()]
    assert await agent_mail.run_pass() == 0


# ── The readers ───────────────────────────────────────────────────────


class _FakeSession:
    """An IMAP inbox with ``msgs`` {uid: raw bytes}."""

    msgs: dict[int, bytes] = {}
    uidvalidity = 7

    def __init__(self, account):
        self.account = account
        self.exists = len(self.msgs)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def folder(self, spec):
        return SimpleNamespace(name="INBOX", raw="INBOX")

    def select(self, folder):
        pass

    def search(self, args):
        if args == ["ALL"]:
            return sorted(self.msgs)
        lo = int(args[1].split(":")[0])
        return sorted(u for u in self.msgs if u >= lo) or [max(self.msgs)]

    def fetch_raw(self, uid):
        return self.msgs[uid], {}


def _raw(uid, subject="Rita, please", to="rita@firm.co"):
    return (f"From: Bob <bob@supplier.co>\r\nTo: {to}\r\nSubject: {subject}\r\nMessage-ID: <m{uid}@x>\r\n"
            f"Date: Thu, 06 Nov 2026 10:00:00 +0000\r\nContent-Type: text/plain\r\n\r\nBody {uid}\r\n").encode()


def test_imap_reader_starts_from_now_then_reads_what_came(monkeypatch):
    from captain_claw import imap_mail

    monkeypatch.setattr(imap_mail, "Session", _FakeSession)
    monkeypatch.setattr(imap_routes, "_account", lambda rec: SimpleNamespace(id=rec["id"]))
    _FakeSession.msgs = {1: _raw(1), 2: _raw(2)}
    got, cur = agent_mail._imap_read({"id": "box"}, {})
    assert got == [] and cur == {"uidvalidity": 7, "last_uid": 2}, "the first look is a baseline"
    _FakeSession.msgs = {1: _raw(1), 2: _raw(2), 3: _raw(3, subject="Q3")}
    got, cur = agent_mail._imap_read({"id": "box"}, cur)
    assert [e.subject for e in got] == ["Q3"] and cur["last_uid"] == 3
    assert got[0].sender == "bob@supplier.co" and got[0].to == ("rita@firm.co",) and got[0].body.strip() == "Body 3"
    assert got[0].ref.startswith("box:7:3:") and got[0].thread == "box:<m3@x>"
    got, cur = agent_mail._imap_read({"id": "box"}, cur)
    assert got == [] and cur["last_uid"] == 3, "UID n:* answers the last one when nothing is new"


async def test_outlook_reader_takes_graph_messages(monkeypatch):
    import httpx

    from captain_claw.flight_deck import microsoft_oauth_routes

    async def token(uid):
        return "tok"

    monkeypatch.setattr(microsoft_oauth_routes, "get_valid_microsoft_access_token", token)
    asked = {}

    class Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None, headers=None):
            asked.update(url=url, params=params, headers=headers)
            return httpx.Response(200, json={"value": [{
                "id": "AAMk1", "conversationId": "conv1", "subject": "For Rita", "receivedDateTime": "2026-11-06T10:00:00Z",
                "from": {"emailAddress": {"address": "Ana@Client.co", "name": "Ana"}},
                "toRecipients": [{"emailAddress": {"address": "alice@firm.co"}}], "ccRecipients": [],
                "body": {"content": "Rita, please send the deck"},
                "internetMessageHeaders": [{"name": "Precedence", "value": "bulk"}]}]})

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    got, cur = await agent_mail._outlook_read(ALICE, {})
    assert got == [] and cur["after"].endswith("Z")
    got, cur = await agent_mail._outlook_read(ALICE, {"after": "2026-11-06T09:00:00Z"})
    assert asked["params"]["$filter"] == "receivedDateTime gt 2026-11-06T09:00:00Z"
    assert got[0].sender == "ana@client.co" and got[0].thread == "conv1" and got[0].headers == {"precedence": "bulk"}
    assert cur == {"after": "2026-11-06T10:00:00Z"}


# ── Sending the reply from the owner's Gmail ──────────────────────────


def test_a_gmail_reply_needs_the_setting_a_verified_sender_and_for_me_the_owner():
    mine = {"alice@firm.co", "alice@gmail.com"}
    me = {"gmail_replies": "me"}
    assert agent_mail.gmail_reply_for(_gmail(sender="alice@gmail.com", verified=True), me, mine) == {
        "message_id": "m1", "to": "alice@gmail.com"}
    assert agent_mail.gmail_reply_for(_gmail(sender="alice@gmail.com", verified=False), me, mine) is None, "forged?"
    assert agent_mail.gmail_reply_for(_gmail(sender="ana@client.co", verified=True), me, mine) is None
    anyone = {"gmail_replies": "verified"}
    assert agent_mail.gmail_reply_for(_gmail(sender="ana@client.co", verified=True), anyone, mine)["to"] == "ana@client.co"
    assert agent_mail.gmail_reply_for(_gmail(sender="ana@client.co", verified=True), {}, mine) is None, "off by default"
    assert agent_mail.gmail_reply_for(_imap("rita-box", sender="alice@firm.co", verified=True), anyone, mine) is None


async def test_with_send_to_you_the_agent_is_told_to_send_and_may(board, ledger, relay, mail, team):  # noqa: F811
    team.save_identity(ALICE, "researcher", {**team.identity(ALICE, "researcher"), "gmail_replies": "me"})
    mail.gmail = [_gmail(sender="alice@firm.co", verified=True)]
    assert await agent_mail.owner_pass(ALICE) == 1
    await _settle()
    call = relay.calls[0]
    auto = call["automation"]
    assert (auto["mail_write"], auto["reply_id"], auto["reply_to"]) == ("draft", "m1", "alice@firm.co")
    head = call["prompt"].partition("<<<EMAIL")[0]
    assert "google_mail action=send to=alice@firm.co reply_to_message_id=m1" in head and "Don't send" not in head


async def test_with_send_to_you_an_outsider_still_gets_a_draft(board, ledger, relay, mail, team):  # noqa: F811
    team.save_identity(ALICE, "researcher", {**team.identity(ALICE, "researcher"), "gmail_replies": "me"})
    mail.gmail = [_gmail(sender="ana@client.co", verified=True)]
    await agent_mail.owner_pass(ALICE)
    await _settle()
    auto = relay.calls[0]["automation"]
    assert "reply_id" not in auto and "google_mail action=create_draft" in relay.calls[0]["prompt"]


async def test_an_email_hand_over_stays_off_the_boards_ledger_reading(board, ledger, relay, mail):  # noqa: F811
    mail.gmail = [_gmail()]
    await agent_mail.owner_pass(ALICE)
    await _settle()
    action = next(a for a in ledger.list_actions(ALICE) if a["payload"].get("inbox"))
    assert action["payload"]["handover"] is True
    task = relay.calls[0]["automation"]["task"]["id"]
    assert not any(e["kind"] in ("proposal", "approval") for e in board.events(ALICE, task))
