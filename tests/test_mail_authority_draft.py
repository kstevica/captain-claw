"""mail_authority's ``draft`` mode (an email's turn) and the colleague/inbox kinds."""

from __future__ import annotations

from captain_claw import mail_authority as ma


def _with(a):
    return ma.bound(a)


def test_draft_mode_writes_drafts_never_sends():
    with _with(ma.automated("inbox", "", "draft")):
        assert ma.check_mail_write("google_mail", "create_draft") is None
        assert ma.check_mail_write("microsoft_mail", "update_draft") is None
        assert ma.check_mail_write("imap_mail", "create_draft", account="box") is None
        for tool, action in (("google_mail", "send"), ("google_mail", "send_draft"), ("microsoft_mail", "send"),
                             ("imap_mail", "send"), ("send_mail", None), ("mcp_mail", None)):
            refusal = ma.check_mail_write(tool, action)
            assert refusal and ma.is_refusal(refusal) and "create_draft" in refusal, (tool, action)
        assert ma.check_notify_others("invitations") is not None


def test_its_own_inbox_may_send():
    with _with(ma.automated("inbox", "", "draft", send_from="rita-box")):
        assert ma.check_mail_write("imap_mail", "send", account="rita-box") is None
        assert ma.check_mail_write("imap_mail", "send_draft", account="RITA-BOX") is None
        assert ma.check_mail_write("imap_mail", "send", account="other") is not None
        assert ma.check_mail_write("imap_mail", "send") is not None
        assert ma.check_mail_write("google_mail", "send") is not None, "only that IMAP inbox"
        assert "account=rita-box" in ma.check_mail_write("google_mail", "send")


def test_send_from_only_in_draft_mode_and_well_formed():
    assert ma.automated("inbox", "", "allow", send_from="box").send_from == ""
    assert ma.automated("inbox", "", "draft", send_from="Bad Box!").send_from == ""
    a = ma.from_wire({"kind": "inbox", "mail_write": "draft", "send_from": "rita-box"}, default_kind="x")
    assert (a.kind, a.mail_write, a.send_from) == ("inbox", "draft", "rita-box")
    a = ma.from_wire({"kind": "colleague", "mail_write": "intent", "send_from": "rita-box"}, default_kind="x")
    assert a.send_from == ""


def test_the_new_kinds_have_their_own_words():
    assert "colleague" in ma.AUTOMATION_KINDS and "inbox" in ma.AUTOMATION_KINDS
    with _with(ma.automated("colleague", "email Ana the report", "intent")):
        assert "a colleague" in ma.automated_prefix()
        assert ma.check_mail_write("google_mail", "create_draft") is None, "the person's words asked for it"
    with _with(ma.automated("colleague", "research Q3", "intent")):
        assert ma.check_mail_write("google_mail", "create_draft") is not None
    with _with(ma.automated("inbox", "", "draft")):
        assert "an email that came to you" in ma.automated_prefix()


def test_the_one_gmail_reply_an_email_turn_may_send():
    a = ma.automated("inbox", "", "draft", reply_id="m1", reply_to="Ana@X.co")
    assert (a.reply_id, a.reply_to) == ("m1", "ana@x.co")
    with _with(a):
        assert ma.check_mail_write("google_mail", "send", reply_message_id="m1", reply_recipients=["ana@x.co"]) is None
        for rid, rcpt in (("m2", ["ana@x.co"]), ("m1", ["mallory@evil.co"]), ("m1", ["ana@x.co", "bob@x.co"]),
                          ("m1", []), ("", ["ana@x.co"])):
            assert ma.check_mail_write("google_mail", "send", reply_message_id=rid, reply_recipients=rcpt), (rid, rcpt)
        assert ma.check_mail_write("google_mail", "send_draft", reply_message_id="m1", reply_recipients=["ana@x.co"])
        assert ma.check_mail_write("microsoft_mail", "send", reply_message_id="m1", reply_recipients=["ana@x.co"])
        refusal = ma.check_mail_write("google_mail", "send")
        assert "google_mail action=send to=ana@x.co reply_to_message_id=m1" in refusal


def test_a_reply_needs_draft_mode_and_both_parts():
    assert ma.automated("inbox", "", "allow", reply_id="m1", reply_to="a@x.co").reply_id == ""
    assert ma.automated("inbox", "", "draft", reply_id="m1", reply_to="").reply_id == ""
    assert ma.automated("inbox", "", "draft", reply_id="bad id!", reply_to="a@x.co").reply_to == ""
    a = ma.from_wire({"kind": "inbox", "mail_write": "draft", "reply_id": "m1", "reply_to": "a@x.co"}, default_kind="x")
    assert (a.reply_id, a.reply_to) == ("m1", "a@x.co")
    a = ma.from_wire({"kind": "colleague", "mail_write": "intent", "reply_id": "m1", "reply_to": "a@x.co"},
                     default_kind="x")
    assert (a.reply_id, a.reply_to) == ("", "")
