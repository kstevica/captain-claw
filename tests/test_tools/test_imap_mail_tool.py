"""The imap_mail agent tool: what it asks Flight Deck, what it tells the model,
and what it refuses before asking anything. Flight Deck's answers are
scripted (ImapMailClient.call); the deck side is tests/test_flight_deck/
test_imap_routes.py."""

import time

import httpx
import pytest

from captain_claw import fabrication_guard, imap_mail_manager, mail_authority
from captain_claw.imap_mail import STALE_ID_HINT
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


async def test_parallel_sends_in_a_scheduled_turn_refused_once_then_pointed_back(deck):
    """A run that wrote fifteen emails it may not send gets the whole refusal
    once, then a short pointer back to it — and the deck is never asked."""
    import asyncio

    tool = ImapMailTool()
    job = "Check the leads table and write each lead a short follow-up"
    with mail_authority.bound(mail_authority.automated("cron", job, "intent")):
        answers = await asyncio.gather(*(
            tool.execute(action="send", account="outreach", to=f"lead{i}@example.org",
                         subject=f"Following up {i}", body="Hello")
            for i in range(15)))
    errors = [a.error for a in answers]
    assert not any(a.success for a in answers) and all(mail_authority.is_refusal(e) for e in errors)
    assert errors.count(mail_authority.refusal_text("cron", "sent")) == 1
    short = [e for e in errors if "the same refusal as the first one" in e]
    assert len(short) == 14 and all(len(e) < 260 for e in short)
    assert deck.calls == []


STALE = ("That ID was never given out for INBOX: its folder number doesn't match the folder's, so "
         "it was mistyped, made up or copied from another folder. " + STALE_ID_HINT)


async def test_stale_id_is_not_asked_about_again(deck):
    deck.answers["read_message"] = (404, {"detail": STALE})
    tool = ImapMailTool()
    first = await tool.execute(action="read_message", message_id="work:1700000300:1:INBOX")
    again = await tool.execute(action="read_message", message_id="work:1700000300:1:INBOX")
    reply = await tool.execute(action="create_draft", reply_to_message_id="work:1700000300:1:INBOX", body="Hi")
    assert first.error == STALE
    assert again.error.startswith("Not asked again, nothing was done: message_id work:1700000300:1:INBOX")
    assert reply.error.startswith("Not asked again") and STALE in reply.error
    # one question to the deck, and no search for an ID this tool never printed
    assert [a for a, _ in deck.calls] == ["read_message"]


async def test_stale_send_draft_says_not_sent_and_is_remembered(deck):
    deck.answers["send_draft"] = (404, {"detail": STALE})
    tool = ImapMailTool()
    res = await tool.execute(action="send_draft", draft_id="work:1700000300:4:Drafts")
    again = await tool.execute(action="send_draft", draft_id="work:1700000300:4:Drafts")
    assert res.error == f"Email not sent: {STALE}" and again.error.startswith("Not asked again")
    assert [a for a, _ in deck.calls] == ["send_draft"]


async def test_renumbered_folder_found_again_by_message_id(monkeypatch):
    """An ID this tool printed went stale (the server renumbered INBOX): the
    read finds the same email by its own Message-ID and reads that."""
    calls = []
    old, new = "work:1700000000:42:INBOX", "work:1700000500:7:INBOX"
    listed = {"folder": "INBOX", "total": 1, "messages": [
        {"id": old, "account": "work", "folder": "INBOX", "message_id": "<offer-7@example.org>",
         "from": "Iva <iva@example.org>", "subject": "Offer", "date": "2026-11-14 09:00 +0100"}]}
    renumbered = ("The server renumbered the folder INBOX since that ID was made, so that ID won't "
                  "work again. " + STALE_ID_HINT)
    message = {"id": new, "account": "work", "folder": "INBOX", "message_id": "<offer-7@example.org>",
               "from": "Iva <iva@example.org>", "to": "me@example.com", "subject": "Offer",
               "date": "2026-11-14 09:00 +0100", "body": "See the offer."}

    async def call(self, action, payload, *, timeout=120.0):
        calls.append((action, dict(payload)))
        if action == "list_messages":
            return 200, listed
        if action == "search":
            return 200, {"folder": "INBOX", "total": 1, "messages": [dict(message)]}
        if action == "read_message":
            return (200, {"message": message}) if payload["message_id"] == new else (404, {"detail": renumbered})
        return 200, {}

    monkeypatch.setattr(imap_mail_manager.ImapMailClient, "call", call)
    tool = ImapMailTool()
    await tool.execute(action="list_messages", account="work")
    res = await tool.execute(action="read_message", message_id=old)
    assert res.success and "See the offer." in res.content
    assert res.content.startswith(f"Note: the server renumbered INBOX, so this email's Message ID changed "
                                  f"from {old} to {new}.")
    assert ("search", {"account": "work", "query": "header:Message-ID=<offer-7@example.org>",
                       "max_results": 5, "folder": "INBOX"}) in calls
    # later calls with the old ID go straight to the new one
    calls.clear()
    later = await tool.execute(action="read_message", message_id=old)
    assert later.success and later.content.startswith(f"Note: {old} is now {new}")
    assert calls == [("read_message", {"message_id": new})]


async def test_renumbered_but_not_found_again_answers_the_deck(monkeypatch):
    old = "work:1700000000:42:INBOX"
    renumbered = ("The server renumbered the folder INBOX since that ID was made, so that ID won't "
                  "work again. " + STALE_ID_HINT)

    async def call(self, action, payload, *, timeout=120.0):
        if action == "list_messages":
            return 200, {"folder": "INBOX", "total": 1, "messages": [
                {"id": old, "folder": "INBOX", "message_id": "<gone@example.org>"}]}
        if action == "search":
            return 200, {"folder": "INBOX", "total": 0, "messages": []}
        return 404, {"detail": renumbered}

    monkeypatch.setattr(imap_mail_manager.ImapMailClient, "call", call)
    tool = ImapMailTool()
    await tool.execute(action="list_messages", account="work")
    res = await tool.execute(action="read_message", message_id=old)
    assert not res.success and res.error == renumbered


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


