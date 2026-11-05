"""The imap_mail agent tool: what it asks Flight Deck, what it tells the model,
and what it refuses before asking anything. Flight Deck's answers are
scripted (ImapMailClient.call); the deck side is tests/test_flight_deck/
test_imap_routes.py."""

import time

import httpx
import pytest

from captain_claw import fabrication_guard, imap_mail_manager, mail_authority
from captain_claw.tools.imap_mail import ImapMailTool


@pytest.fixture()
def deck(monkeypatch):
    """Flight Deck, scripted: ``deck.answers[action] = (status, body)``."""
    class Deck:
        calls: list = []
        answers: dict = {}
        raises: Exception | None = None

    d = Deck()
    d.calls, d.answers = [], {}

    async def call(self, action, payload, *, timeout=120.0):
        d.calls.append((action, payload))
        if d.raises:
            raise d.raises
        return d.answers.get(action, (200, {}))

    monkeypatch.setattr(imap_mail_manager.ImapMailClient, "call", call)
    return d


SENT = {"ok": True, "account": "work", "from": "ana@firma.hr", "to": "bob@x.co", "cc": "", "bcc": "",
        "subject": "Ponuda", "message_id": "<m1@firma.hr>", "attachments": [], "sent_last_24h": 2,
        "daily_limit": 50, "notes": []}


async def test_send_answer_reads_as_sent_for_the_guard(deck):
    deck.answers["send"] = (200, SENT)
    res = await ImapMailTool().execute(action="send", account="work", to="bob@x.co", subject="Ponuda", body="Hi")
    assert res.success and res.content.startswith("Email sent.")
    assert fabrication_guard.mail_send_went_out(res.content)
    assert deck.calls == [("send", {"account": "work", "to": "bob@x.co", "subject": "Ponuda", "body": "Hi"})]
    assert "imap_mail" in fabrication_guard.MAIL_TOOLS


async def test_unknown_outcome_is_never_reported_as_not_sent(deck):
    deck.answers["send"] = (504, {"detail": "The SMTP server took the email but never confirmed it"})
    res = await ImapMailTool().execute(action="send", account="work", to="bob@x.co", subject="s", body="b")
    assert not res.success and res.error.startswith("Email may or may not have been sent")
    assert fabrication_guard.mail_send_went_out(res.error)


async def test_unanswered_send_tells_the_model_to_check_sent(deck):
    deck.raises = httpx.ReadTimeout("slow")
    res = await ImapMailTool().execute(action="send_draft", draft_id="work:1:2:Drafts")
    assert not res.success and "may or may not have been sent" in res.error and "Sent" in res.error


async def test_refused_send(deck):
    deck.answers["send"] = (403, {"detail": "Sending is turned off for ana@firma.hr"})
    res = await ImapMailTool().execute(action="send", account="work", to="bob@x.co", subject="s", body="b")
    assert not res.success and res.error.startswith("Email not sent: Sending is turned off")


async def test_automated_turn_refused_before_any_request(deck):
    with mail_authority.bound(mail_authority.automated("cron", "Summarize my inbox", "intent")):
        w = await ImapMailTool().execute(action="send", account="work", to="bob@x.co", subject="s", body="b")
        o = await ImapMailTool().execute(action="archive", message_ids=["work:1:1:INBOX"])
        r = await ImapMailTool().execute(action="list_messages", account="work")
    assert mail_authority.is_refusal(w.error) and mail_authority.is_refusal(o.error)
    assert r.success
    assert [a for a, _ in deck.calls] == ["list_messages"]


async def test_job_that_asks_to_email_may_draft(deck):
    deck.answers["create_draft"] = (200, {"draft_id": "work:1:9:Drafts", "to": "ana@x.co", "subject": "R",
                                          "account": "work", "attachments": []})
    with mail_authority.bound(mail_authority.automated("cron", "Email ana@x.co the weekly report", "intent")):
        res = await ImapMailTool().execute(action="create_draft", account="work", to="ana@x.co",
                                           subject="R", body="…")
    assert res.success and "Draft ID: work:1:9:Drafts" in res.content


async def test_member_turn_never_reaches_the_deck(monkeypatch):
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    monkeypatch.setattr(GoogleOAuthManager, "_member_call", staticmethod(lambda: True))
    res = await ImapMailTool().execute(action="list_messages")
    assert not res.success and "shared-agent member" in res.error


async def test_no_deck_no_inboxes(monkeypatch):
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_base", staticmethod(lambda: ""))
    res = await ImapMailTool().execute(action="list_accounts")
    assert not res.success and "needs Flight Deck" in res.error


@pytest.mark.parametrize("action", ["reply", "forward", "delete", "trash"])
async def test_actions_it_does_not_have(deck, action):
    res = await ImapMailTool().execute(action=action)
    assert not res.success and deck.calls == []


async def test_list_renders_ids_and_inboxes(deck):
    deck.answers["list_messages"] = (200, {"accounts": ["work", "support"], "folder": "every inbox", "total": 2,
                                           "errors": [{"account": "old", "error": "login failed"}],
                                           "messages": [
        {"id": "work:1:5:INBOX", "account": "work", "folder": "INBOX", "thread_id": "work:<a@x>",
         "from": "Ivana <i@k.hr>", "subject": "Ponuda", "date": "2026-11-05 09:00 +0100", "unread": True,
         "has_attachments": True, "preview": "Poštovani"},
        {"id": "support:3:7:INBOX", "account": "support", "folder": "INBOX", "thread_id": "",
         "from": "x@y.co", "subject": "Help", "date": "2026-11-04 10:00 +0100"}]})
    res = await ImapMailTool().execute(action="list_messages", account="all", include_body=True)
    assert "Message ID: work:1:5:INBOX · Inbox: work" in res.content
    assert "Thread ID: work:<a@x>" in res.content and "[unread, attachments]" in res.content
    assert "inbox old couldn't be read: login failed" in res.content
    assert deck.calls[0][1] == {"account": "all", "include_body": True}


async def test_list_accounts_says_which_can_send(deck):
    deck.answers["list_accounts"] = (200, {"accounts": [
        {"id": "work", "email": "ana@firma.hr", "label": "Work", "can_send": True, "sending_enabled": True,
         "allowed_recipients": ["@firma.hr"], "daily_limit": 20},
        {"id": "arhiva", "email": "arhiva@firma.hr", "label": "", "can_send": False, "sending_enabled": False,
         "allowed_recipients": [], "daily_limit": 50}]})
    res = await ImapMailTool().execute(action="list_accounts")
    assert "account=work: ana@firma.hr — Work · sending on (up to 20/day, only to @firma.hr)" in res.content
    assert "account=arhiva: arhiva@firma.hr · receive-only" in res.content


async def test_move_prints_new_ids(deck):
    deck.answers["move"] = (200, {"destination": "Klijenti/Šibenik", "skipped": [],
                                  "moved": [["work:1:5:INBOX", "work:9:1:Klijenti/Šibenik"]]})
    res = await ImapMailTool().execute(action="move", message_ids=["work:1:5:INBOX"], folder="Klijenti/Šibenik")
    assert "work:1:5:INBOX → work:9:1:Klijenti/Šibenik" in res.content


class TestRegistryGate:
    def test_hidden_until_an_inbox_is_connected(self):
        from captain_claw.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register(ImapMailTool(), metadata={"requires_imap": True})
        imap_mail_manager._mark(False)
        assert "imap_mail" not in [t.name for t in reg._resolve_tools()]
        assert "Email (IMAP)" in reg._unconnected_reason("imap_mail")
        imap_mail_manager._mark(True, [{"id": "work", "email": "a@b.co"}])
        assert "imap_mail" in [t.name for t in reg._resolve_tools()]
        imap_mail_manager._STATE = (True, time.time() - 3600, [])  # stale → hidden again
        assert "imap_mail" not in [t.name for t in reg._resolve_tools()]
        imap_mail_manager._mark(False)

    def test_members_never_get_it(self):
        from captain_claw import speaker

        assert "imap_mail" not in speaker.SPEAKER_TOOL_ALLOWLIST_MAX


def test_schema_paths_confined():
    from captain_claw import deck_confine

    props = set(ImapMailTool.parameters["properties"])
    assert ("attachments", "read") in deck_confine.PATH_RULES["imap_mail"]
    assert {"attachment", "remove_attachments", "folder"} <= deck_confine.NON_PATH_PARAMS["imap_mail"]
    assert "attachments" in props


