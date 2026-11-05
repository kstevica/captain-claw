"""captain_claw.imap_mail against a real IMAP/SMTP server — skipped unless
CAPTAIN_CLAW_IMAP_LIVE=1. Start the server first (GreenMail, users ana / bob):

    docker run -d --name cc-greenmail -p 13025:3025 -p 13143:3143 -p 13465:3465 -p 13993:3993 \\
      -e GREENMAIL_OPTS='-Dgreenmail.setup.test.all -Dgreenmail.hostname=0.0.0.0 \\
      -Dgreenmail.users=ana:secret@example.com,bob:secret2@example.com -Dgreenmail.users.login=email' \\
      greenmail/standalone:2.1.3

Each run uses fresh subjects, so it can run against the same server again.
"""

import os
import time
import uuid

import pytest

from captain_claw import imap_mail as m
from captain_claw.mail_attachments import Attachment

pytestmark = pytest.mark.skipif(os.environ.get("CAPTAIN_CLAW_IMAP_LIVE") != "1",
                                reason="needs a live IMAP/SMTP server (CAPTAIN_CLAW_IMAP_LIVE=1)")


def _acct(user, pw, aid, sec):
    ports = {"ssl": (13993, 13465), "none": (13143, 13025)}[sec]
    return m.account_from_settings({
        "email": user, "imap_host": "127.0.0.1", "imap_port": ports[0], "imap_security": sec,
        "smtp_host": "127.0.0.1", "smtp_port": ports[1], "smtp_security": sec,
    }, account_id=aid, password=pw)


def test_round_trip():
    ana = _acct("ana@example.com", "secret", "ana", "ssl")
    bob = _acct("bob@example.com", "secret2", "bob", "none")
    assert m.test_account(ana)["ok"] and m.test_account(bob)["ok"]
    assert not m.test_account(_acct("ana@example.com", "wrong", "ana", "ssl"))["ok"]

    tag = uuid.uuid4().hex[:8]
    subject = f"Račun {tag} — škola"
    data, mid = m.build_message(ana, to="Bob <bob@example.com>", cc="", bcc="", subject=subject,
                                body="Pozdrav Bob — čćžšđ", html_body="",
                                attachments=[Attachment("invoice.txt", "text/plain", b"100 EUR\n")])
    m.smtp_deliver(ana, ["bob@example.com"], m.without_bcc(data))
    assert m.after_send(ana, data, answered=None, draft=None) == []
    time.sleep(1)

    found = m.search(bob, query=f"subject:{tag}")["messages"]
    assert len(found) == 1 and found[0]["subject"] == subject and found[0]["has_attachments"]
    assert m.search(bob, query="škola")["total"] >= 1
    ref = m.parse_ref(found[0]["id"])
    detail = m.read_message(bob, ref)
    assert detail["body"].startswith("Pozdrav Bob") and detail["attachments"][0]["name"] == "invoice.txt"
    assert m.get_attachments(bob, ref, "1")[0].data == b"100 EUR\n"

    with m.Session(bob) as s:
        ctx = m.reply_context(s, ref, reply_all=False)
        assert ctx.to_default == "ana@example.com" and ctx.subject_default == f"Re: {subject}"
        reply, rmid = m.build_message(bob, to=ctx.to_default, cc="", bcc="", subject=ctx.subject_default,
                                      body="Hvala!", html_body="", in_reply_to=ctx.message_id,
                                      references=ctx.references)
        assert m.find_repeats(s, to=ctx.to_default, cc="", bcc="", subject=ctx.subject_default,
                              reply_msgid=ctx.message_id) == []
        draft_id = m.save_draft(s, reply, rmid)
        assert [x["kind"] for x in m.find_repeats(s, to=ctx.to_default, cc="", bcc="",
                                                   subject=ctx.subject_default,
                                                   reply_msgid=ctx.message_id)] == ["draft"]
    thread = m.get_thread(bob, ctx.message_id)["messages"]
    assert {t["folder"] for t in thread} >= {"INBOX", "Drafts"}

    assert m.set_seen(bob, [ref], [], True)["changed"] == [str(ref)]
    folder = f"Clients/Acme {tag}"
    created = m.create_folder(bob, folder)
    moved = m.move(bob, [ref], [], folder)
    assert moved["destination"] == created["folder"] and moved["moved"][0][1]
    with pytest.raises(m.MailError):
        m.move(bob, [m.parse_ref(moved["moved"][0][1])], [], "Drafts")
    with m.Session(bob) as s:
        m.delete_message(s, m.parse_ref(draft_id))
    assert subject in [x["subject"] for x in m.list_messages(ana, folder="sent", max_results=50)["messages"]]
