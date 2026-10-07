"""mail_authority: who may write email in this turn (PR E, agent part).

User rule: the agent doesn't draft, reply to or send email unless it was
specifically told to. Human turns are never hard-gated; automated turns may
write email only when the job's own human-written text explicitly asks for
one (EN + HR detector), or when a human approved the action ("allow"). An
"email me …" job may only write one new email to the owner.

Pure module: no network, no DB, no config.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from captain_claw import mail_authority as ma

_WORKER_ENVS = (
    "CLAW_BEING_WORKER", "CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER",
    "CLAW_COUNCIL_WORKER", "CLAW_CODE_AGENT",
)


@pytest.fixture(autouse=True)
def _no_worker_env(monkeypatch):
    for name in _WORKER_ENVS:
        monkeypatch.delenv(name, raising=False)


# ── detector (part 0 §5) ─────────────────────────────────────────────

MUST_BE_TRUE = [
    "draft a reply to Ana about the Q3 numbers",
    "Draft replies to the new email threads from the rundown",
    "write an email to bob@x.co",
    "please reply to Miro and say yes",
    "send an email to the investors with the deck",
    "send the weekly report by email",
    "email me the summary every morning",
    "prepare a short reply to Luka",
    "napiši mail Ani da kasnim",
    "napravi draft za Tihu, ja ću ručno poslati",
    "odgovori Miliju, stavi Tihu u cc",
    "odgovori na mail od Nataše",
    "pošalji mail Roberu s ponudom",
    "posalji izvjestaj mailom",
    "pripremi odgovor na upit od Ane",
    "Sumamrize questions, and do a draft, yes",
    "answer Ana's email about the invoice",
    "Send Stevica an email with the inbox summary",
    "respond to the email from bob@x.co",
    "odgovori na taj mejl i reci da može",
]

MUST_BE_FALSE = [
    "summarize my inbox",
    "check unanswered emails",
    "what's new in my inbox?",
    "list emails that need a reply",
    "which emails do I need to respond to first?",
    "find emails I haven't replied to",
    "send me a summary of my unread emails on WhatsApp",
    "morning briefing: emails, calendar, tasks",
    "triage my inbox and flag anything urgent",
    "jutarnji pregled: mailovi i kalendar",
    "provjeri neodgovorene mailove",
    "pregledaj inbox i reci mi što je hitno",
    "napravi sažetak mailova od jučer",
    "na koje mailove trebam odgovoriti?",
    "pošalji mi sažetak mailova na WhatsApp",
    "Email check: summarize unread",
    "Respond in Croatian.",
    "Reply only with the summary",
    "Answer the user's question: {{trigger.text}}",
    "Write a short reply to the user",
    "Odgovori kratko na hrvatskom",
    "odgovori na pitanje korisnika",
    "Pripremi odgovor korisniku",
    "jutarnji pregled: mailovi i kalendar. Odgovori na hrvatskom",
    "Reply ONLY with JSON of this shape:",
    "Odgovori na poruku: {{trigger.text}}",
    "pripremi odgovor na upit",
    "Reply to the user on WhatsApp in English",
]


@pytest.mark.parametrize("text", MUST_BE_TRUE)
def test_detector_table_true(text):
    assert ma.explicit_mail_intent(text) is True, text


@pytest.mark.parametrize("text", MUST_BE_FALSE)
def test_detector_table_false(text):
    assert ma.explicit_mail_intent(text) is False, text


def test_detector_table():
    """The whole part 0 §5 table in one place."""
    assert [t for t in MUST_BE_TRUE if not ma.explicit_mail_intent(t)] == []
    assert [t for t in MUST_BE_FALSE if ma.explicit_mail_intent(t)] == []


SCOPE_TABLE = [
    ("email me the summary every morning", "self"),
    ("send me the report by email", "self"),
    ("pošalji mi sažetak mailom", "self"),
    ("javi mi mailom kad stigne ponuda", "self"),
    ("send the weekly report by email", "any"),
    ("draft a reply to Ana about the Q3 numbers", "any"),
    ("email me the summary and draft replies to Ana", "any"),
    ("summarize my inbox", "none"),
]


@pytest.mark.parametrize("text,scope", SCOPE_TABLE)
def test_scope_table(text, scope):
    assert ma.intent_scope(text) == scope


def test_narrower_intent_pins():
    a = "Send Stevica an email with the inbox summary"
    b = "email me the summary every morning"
    assert ma.narrower_intent(a, b) == b
    assert ma.narrower_intent("summarize my inbox daily", "draft a reply to Ana") == ""
    assert ma.narrower_intent("draft a reply to Ana", "") == ""
    # a tie keeps the first
    assert ma.narrower_intent("draft a reply to Ana", "napravi draft za Anu") == "draft a reply to Ana"


def test_narrower_self_intent_keeps_only_addresses_both_name():
    """A self job may reach an address its text names, so the agent's task
    can't add one the user never wrote (agent cron, peer forwarding)."""
    user = "email me the summary every morning"
    task = "Every morning email me at x@evil.co the summary"
    out = ma.narrower_intent(task, user)
    assert "x@evil.co" not in out and ma.intent_scope(out) == "self"
    assert ma.job_addresses(out) == set()
    # an address the user wrote too is kept
    user2 = "every morning email me at stevica@company.com the summary"
    task2 = "Every morning email me the summary at Stevica@Company.com and cc x@evil.co"
    assert ma.job_addresses(ma.narrower_intent(task2, user2)) == {"stevica@company.com"}
    # the human text narrower than the task: its own addresses must be in the task
    assert ma.job_addresses(ma.narrower_intent(
        "email Ana and me the summary at x@evil.co", "email me at x@evil.co the summary",
    )) == {"x@evil.co"}
    # the end-to-end shape: an agent cron job whose task smuggles an address
    payload = {"text": task, "author": "agent", "mail_intent_text": user}
    gmail = {"kstevica@gmail.com"}
    with ma.bound(ma.automated("cron", ma.cron_job_text(payload))):
        assert ma.is_refusal(ma.check_mail_write(
            "google_mail", "create_draft", recipients=["x@evil.co"], own_addresses=gmail))
        assert ma.check_mail_write(
            "google_mail", "create_draft", recipients=["kstevica@gmail.com"], own_addresses=gmail) is None


def test_detector_extras():
    assert ma.explicit_mail_intent("do not send any email") is False
    assert ma.explicit_mail_intent("Summarize the email the boss sent") is False
    assert ma.explicit_mail_intent("draft a reply to bob@x.co") is True
    assert ma.explicit_mail_intent("draft a reply to Nataša") is True
    assert ma.explicit_mail_intent("") is False


# Review additions: scheduled jobs and requests are often phrased with the
# time or a request phrase before the verb — they still ask for an email.
EXTRA_TRUE = [
    ("Every Friday draft a reply to Ana", "any"),
    ("Every morning at 9 draft replies to unanswered client emails", "any"),
    ("On Friday draft an email to Ana", "any"),
    ("At 5pm reply to Ana", "any"),
    ("In 2 hours draft an email to Bob", "any"),
    ("svaki petak napiši mail Ani", "any"),
    ("Svako jutro odgovori na nove mailove", "any"),
    ("Set up a cron job in 5 minutes to draft a reply to Stevica saying test", "any"),
    ("I want you to draft an email to Ana", "any"),
    ("I need you to write an email to Ana", "any"),
    ("Help me draft an email to Ana", "any"),
    ("Let's draft a reply to Ana", "any"),
    ("Možeš li napisati mail Ani?", "any"),
    ("Make a draft for Ana", "any"),
    ("Write Ana an email", "any"),
    ("Email Ana the deck", "any"),
    ("Reply all to the thread from Ana", "any"),
    ("mail me the report every morning", "self"),
    ("Daily at 09:00 email me the summary", "self"),
    ("pregledaj mailove i odgovori Ani", "any"),
]
# …while statements, reminders and other channels still don't.
EXTRA_FALSE = [
    "I reply to client emails myself, just summarize the inbox",
    "I draft replies myself; summarize the inbox",
    "i reply to Ana's emails myself, summarize",
    "Remind me to reply to Ana tomorrow",
    "Set a reminder to reply to Ana",
    "Create a task to remind me to reply to Ana",
    "Read the email and the reply to Ana and summarize",
    "Email Ana sent: summarize",
    "Every morning summarize the inbox",
    "In the morning list emails to reply to",
    "Every day reply in English",
    "Each morning respond with a summary",
    "svaki dan odgovori kratko",
    "Respond to Jira tickets assigned to me",
    "Write a message to Ana on Slack",
    "Reply to Ana on WhatsApp",
    "odgovori Ani na WhatsApp",
    "Write the drafts list to a file",
    "emails I need you to reply to",
]


@pytest.mark.parametrize("text,scope", EXTRA_TRUE)
def test_detector_review_true(text, scope):
    assert ma.intent_scope(text) == scope, text


@pytest.mark.parametrize("text", EXTRA_FALSE)
def test_detector_review_false(text):
    assert ma.intent_scope(text) == "none", text


# Integration review: explicit jobs the detector used to miss (each one was
# refused at fire time) — free words before the noun, send-to-an-address,
# "by mail" / "via Gmail" / naming the tool, HR imperfectives, a clitic or a
# dative NAME before the HR verb, and address-like flow placeholders.
EXPLICIT_JOBS = [
    ("Every Friday email Ana the report", "any"),
    ("Svaki ponedjeljak pošalji Marku izvještaj mailom", "any"),
    ("draft replies to the investors who wrote this week", "any"),
    ("Every Monday draft a status update email to the team", "any"),
    ("Draft a reminder email to Ana", "any"),
    ("Write a weekly email to the team", "any"),
    ("Write the monthly investor update email", "any"),
    ("Prepare the weekly newsletter email and send it to ana@x.com", "any"),
    ("Send the weekly report to ana@x.com", "any"),
    ("Every Friday send an update to ana@x.com", "any"),
    ("Forward the weekly report to ana@x.com", "any"),
    ("Mail the report to ana@x.com every Friday", "any"),
    ("Pošalji sažetak na ana@x.com", "any"),
    ("Svaki dan u 9 pošalji podsjetnik Ani na ana@x.com", "any"),
    ("Every Friday at 17h send Ana the timesheet by mail", "any"),
    ("Every Friday, send the newsletter to all subscribers via Gmail", "any"),
    ("Use google_mail to send the report to boss@x.com", "any"),
    ("Use google_mail create_draft to reply to the email", "any"),
    ("Create a Gmail draft for Ana with the report", "any"),
    ("Send the team a weekly email with the KPI report", "any"),
    ("Svaki petak šalji Ani izvještaj mailom", "any"),
    ("Odgovaraj na mailove od klijenata", "any"),
    ("Svaki petak Ani napiši mail s izvještajem", "any"),
    ("Svaki petak Ani odgovori na mail", "any"),
    ("Svakog jutra mi napravi draft odgovora za Anu", "any"),
    ("Every weekday, if Ana emails, reply to her", "any"),
    ("When Ana emails, draft a reply", "any"),
    ("Email {{trigger.from}} a confirmation of the booking", "any"),
    ("Email {{vars.customer_email}} the invoice", "any"),
    ("Reply to {{trigger.from}} confirming the booking", "any"),
    ("Reply to {{trigger.sender}} by email confirming the booking", "any"),
    ("Send {{vars.client}} an email with the invoice", "any"),
    ("Send the invoice to {{vars.client_email}}", "any"),
    ("Draft a reply to {{trigger.from}}", "any"),
    ("Draft a confirmation email to {{trigger.from}}", "any"),
    ("Pošalji {{vars.klijent}} mail s ponudom", "any"),
    ("Pošalji ponudu na {{vars.email}}", "any"),
    ("Create a Gmail draft replying to {{trigger.from}}", "any"),
    ("Every morning email me and Ana the summary", "any"),
    ("Every morning email me and ana@x.co the summary", "any"),
    # self: only the user
    ("Send me the report by mail every morning", "self"),
    ("Every morning send the agenda to my email", "self"),
    ("Every morning send a summary to my inbox", "self"),
    ("Every morning write me an email with the agenda", "self"),
    ("Svako jutro mi napiši mail sa sažetkom", "self"),
    ("Svako jutro mi šalji mail sa sažetkom", "self"),
    ("Svaki petak šalji mi izvještaj mailom", "self"),
    ("Every morning email me at stevica@company.com the inbox summary", "self"),
    # (re-check) the looser forms still match an email with the reply in mail
    ("If Ana emails, reply to her in English", "any"),
    ("Every weekday, if Ana emails, reply to her saying I'm out", "any"),
    ("Summarize my inbox and use google_mail to draft replies to Ana", "any"),
    ("Every Friday use gmail to send the report to the team", "any"),
    ("Draft replies to the investors who wrote this week, in English", "any"),
    ("Draft a short status update email for the team", "any"),
]
# …and still not: other channels, reminders, documents, code, statements,
# placeholders that aren't an addressee, and replies with no email in sight.
EXPLICIT_JOBS_FALSE = [
    "Svaki ponedjeljak pošalji Marku izvještaj",
    "Send the weekly report to Ana",
    "Write to Ana every Friday with the report",
    "Remind me every Friday to email Ana",
    "Remind me to email ana@x.com tomorrow",
    "Remind me to send Ana the report by email",
    "Podsjeti me da pošaljem mail Ani",
    "do the email triage",
    "Make sure the email from Ana is archived",
    "write the email subjects into a file",
    "Write the email parser in parser.py",
    "Draft notes from the email thread",
    "Write a summary of my emails",
    "Write up the emails into a report",
    "Prepare the project draft for review",
    "Write the draft of section 2 to the file",
    "Draft a response to the RFP",
    "Send the answer to {{trigger.text}}",
    "Reply to the comment on the PR",
    "Send the report to the team on Slack",
    "Never use google_mail to send anything",
    "Use gmail to search for invoices",
    "Check gmail and write a summary",
    "I send the report to ana@x.com myself",
    "The mail to ana@x.com bounced",
    "Tko odgovori na mail od Ane?",
    "Ne šalji mail Ani",
    "Nemoj odgovarati na mailove",
    "odgovaraj na hrvatskom",
    "Svaki dan odgovaraj kratko",
    "reply to him",
    "Summarize what Ana emailed and reply in English",
    "If Ana emails, tell me on WhatsApp",
    "Create a message template for support",
    # (re-check) describing the tool isn't an instruction to use it
    "The report shows how often we use gmail to send newsletters",
    "Write a report on how often we use gmail to send newsletters",
    "Summarize the article about how teams use gmail to draft replies",
    "Check whether people use gmail to reply",
    "Do not, under any circumstances, use gmail to send anything",
    "Do not, under any circumstances, email Ana the report",
    "Nemoj, ni slučajno, slati mail Ani",
    # another channel or a document, not an email
    "Write a WhatsApp reply to Ana",
    "Write a LinkedIn reply to the recruiter",
    "Draft the onboarding email copy in a Google Doc",
    "Write the welcome email copy for the landing page",
    "Prepare the quarterly email newsletter design",
    "Draft me a list of email ideas",
    # people as a modifier / possessive, or a reply that goes somewhere else
    "Draft responses to customer reviews on Google",
    "Draft responses to customer questions in the FAQ doc",
    "Write replies to candidates' questions in the doc",
    "Draft responses for the investors FAQ page",
    "Draft a response for the clients survey report",
    "Draft a reply to the vendor ticket in Zendesk",
    "Write responses to partner comments in Notion",
    "Draft replies to applicants in Greenhouse",
    "Write a reply to the customer review on Trustpilot",
    "Prepare responses to the senders' complaints in the support portal",
    "Draft responses to everyone who commented on the post",
    "Draft replies to people who wrote on the forum",
    "Every Friday write a response to subscribers' comments on the blog",
    "If Ana emails about the doc, reply to her comment in the doc",
    "When the email arrives, reply to it in the ticket",
    "When Ana's email arrives, draft a reply in the doc",
    "Read the email from the client and draft a response in the ticket",
    "After reading Ana's email, answer her question in the chat",
]


@pytest.mark.parametrize("text,scope", EXPLICIT_JOBS)
def test_detector_explicit_jobs(text, scope):
    assert ma.intent_scope(text) == scope, text


@pytest.mark.parametrize("text", EXPLICIT_JOBS_FALSE)
def test_detector_explicit_jobs_false(text):
    assert ma.intent_scope(text) == "none", text


# ── check_mail_write matrix ──────────────────────────────────────────

WRITES = ["create_draft", "update_draft", "send", "send_draft"]
READS = ["read_message", "list_messages", "search", "list_drafts", "get_thread"]


def test_human_turns_never_gated():
    assert ma.current() == ma.HUMAN
    for action in WRITES + READS:
        assert ma.check_mail_write("google_mail", action) is None
    assert ma.check_mail_write("send_mail", None) is None
    assert ma.check_mail_write("mcp_mail", None) is None
    with ma.bound(ma.human("summarize my inbox")):
        assert ma.check_mail_write("google_mail", "create_draft") is None


def test_automated_deny():
    with ma.bound(ma.automated("cron", "draft a reply to Ana", "deny")):
        for action in WRITES:
            r = ma.check_mail_write("google_mail", action)
            assert r and ma.is_refusal(r)
        assert ma.is_refusal(ma.check_mail_write("send_mail", None))
        assert ma.is_refusal(ma.check_mail_write("mcp_mail", None))
        for action in READS:
            assert ma.check_mail_write("google_mail", action) is None


def test_automated_allow():
    with ma.bound(ma.automated("autonomy_tool", "", "allow")):
        for action in WRITES:
            assert ma.check_mail_write("google_mail", action) is None
        assert ma.check_mail_write("send_mail", None) is None


def test_automated_intent_any_and_none():
    with ma.bound(ma.automated("cron", "draft a reply to Ana")):
        assert ma.check_mail_write("google_mail", "create_draft") is None
        assert ma.check_mail_write("mcp_mail", None) is None
    with ma.bound(ma.automated("cron", "summarize my inbox")):
        r = ma.check_mail_write("google_mail", "create_draft")
        assert r == ma.refusal_text("cron", "created")


def test_automated_intent_self_scope():
    me = {"me@x.co"}
    with ma.bound(ma.automated("cron", "email me the summary every morning")):
        assert ma.needs_recipient_check() is True
        assert ma.check_mail_write("google_mail", "create_draft",
                                   recipients=["me@x.co"], own_addresses=me) is None
        assert ma.check_mail_write("google_mail", "send",
                                   recipients=["me@x.co"], own_addresses=me) is None
        assert ma.check_mail_write("google_mail", "create_draft", recipients=["ana@x.co"],
                                   own_addresses=me) == ma.refusal_text_self("cron", "created")
        assert ma.is_refusal(ma.check_mail_write(
            "google_mail", "create_draft", recipients=["me@x.co", "ana@x.co"], own_addresses=me))
        assert ma.is_refusal(ma.check_mail_write(
            "google_mail", "create_draft", recipients=None, own_addresses=me))
        assert ma.is_refusal(ma.check_mail_write(
            "google_mail", "update_draft", recipients=["me@x.co"], own_addresses=me))
        assert ma.is_refusal(ma.check_mail_write(
            "google_mail", "send_draft", recipients=["me@x.co"], own_addresses=me))
        assert ma.is_refusal(ma.check_mail_write(
            "google_mail", "create_draft", recipients=["me@x.co"], own_addresses=set()))
        assert ma.check_mail_write("send_mail", None, recipients=["anyone@x.co"],
                                   own_addresses=None) is None
        assert ma.is_refusal(ma.check_mail_write(
            "send_mail", None, recipients=["a@x.co", "b@x.co"], own_addresses=None))
        assert ma.is_refusal(ma.check_mail_write("mcp_mail", None))


def test_self_scope_allows_addresses_the_job_names():
    """ "email me at stevica@company.com …": that address is the user's too."""
    gmail = {"kstevica@gmail.com"}
    job = "Every morning email me at Stevica@Company.com the inbox summary"
    with ma.bound(ma.automated("cron", job)):
        assert ma.needs_recipient_check() is True
        assert ma.check_mail_write("google_mail", "create_draft", recipients=["stevica@company.com"],
                                   own_addresses=gmail) is None
        assert ma.check_mail_write("google_mail", "send", recipients=["kstevica@gmail.com"],
                                   own_addresses=gmail) is None
        assert ma.check_mail_write("google_mail", "create_draft", recipients=["stevica@company.com"],
                                   own_addresses=set()) is None
        assert ma.check_mail_write("send_mail", None, recipients=["stevica@company.com"],
                                   own_addresses=None) is None
        # anyone else is still refused
        assert ma.is_refusal(ma.check_mail_write(
            "google_mail", "create_draft", recipients=["ana@x.co"], own_addresses=gmail))
        assert ma.is_refusal(ma.check_mail_write(
            "google_mail", "create_draft", recipients=["stevica@company.com", "ana@x.co"],
            own_addresses=gmail))
        assert ma.is_refusal(ma.check_mail_write(
            "send_mail", None, recipients=["a@x.co", "b@x.co"], own_addresses=None))
        assert ma.is_refusal(ma.check_mail_write(
            "google_mail", "create_draft", recipients=[], own_addresses=gmail))
        assert ma.is_refusal(ma.check_mail_write(
            "google_mail", "update_draft", recipients=["stevica@company.com"], own_addresses=gmail))
    assert ma.job_addresses(job) == {"stevica@company.com"}
    assert ma.job_addresses("email me the summary") == set()


def test_needs_recipient_check_only_for_self_intent():
    assert ma.needs_recipient_check() is False
    with ma.bound(ma.human("email me the summary")):
        assert ma.needs_recipient_check() is False
    with ma.bound(ma.automated("cron", "email me the summary", "deny")):
        assert ma.needs_recipient_check() is False
    with ma.bound(ma.automated("cron", "email me the summary", "allow")):
        assert ma.needs_recipient_check() is False
    with ma.bound(ma.automated("cron", "draft a reply to Ana")):
        assert ma.needs_recipient_check() is False
    with ma.bound(ma.automated("cron", "summarize my inbox")):
        assert ma.needs_recipient_check() is False
    with ma.bound(ma.automated("cron", "email me the summary")):
        assert ma.needs_recipient_check() is True


# ── exact strings (part 0 §6) ────────────────────────────────────────


def test_refusal_texts_exact():
    head = (
        "[not-authorized: mail-write] Not created: this turn was started by {}, "
        "not by the user, and its instructions don't ask for an email to be written or sent. "
        "Nothing was drafted or sent. Don't retry or work around this (no other tool, no "
        "send_mail) — "
    )
    # inbox triage: who is waiting
    assert ma.refusal_text("autonomy", "created") == head.format("Autonomous Work") + (
        "tell the user who is waiting for a reply and offer to draft it when they ask."
    )
    # a job meant to email: how to make it do so
    assert ma.refusal_text("cron", "created") == head.format("a scheduled task") + (
        "tell the user that this job's text doesn't ask to email anyone, so no email was "
        "written. If they want it to write email, they can edit the job to say so "
        "explicitly, e.g. 'email Ana the report' or 'pošalji Ani izvještaj mailom'."
    )
    assert ma.refusal_text("peer", "created") == head.format("another agent") + (
        "tell the user no email was written because the request didn't explicitly ask for "
        "one. They can ask for it in their own words, e.g. 'draft a reply to Ana'."
    )
    assert ma.refusal_text("being", "created") == head.format("a being tick") + (
        "say in your answer that no email was written because this run wasn't asked to write one."
    )
    for kind in ("cron", "fd_scheduler", "flow", "flow_tool", "plan", "mcp_task"):
        r = ma.refusal_text(kind, "sent")
        assert "edit the job" in r and "who is waiting" not in r, kind
    for kind in ma.AUTOMATION_KINDS:
        assert ma.is_refusal(ma.refusal_text(kind, "sent")), kind
    assert "who is waiting" in ma.refusal_text("autonomy_tool", "sent")
    assert ma.is_refusal(ma.refusal_text("cron", "created"))
    assert ma.refusal_text_self("cron", "sent") == (
        "[not-authorized: mail-write] Not sent: this turn was started by a scheduled task, and "
        "its instructions only ask for an email to the user themself. Write one new email "
        "addressed only to the user's own address or an address the instructions name (no "
        "other recipients, not a reply to someone else's message). Nothing was drafted or sent."
    )
    assert ma.is_refusal(ma.refusal_text_self("cron", "sent"))
    assert ma.MAIL_REFUSAL_TAG == "[not-authorized: mail-write]"
    assert ma.is_refusal("plain error") is False


def test_verbs_by_action():
    with ma.bound(ma.automated("flow", "", "deny")):
        assert "Not created:" in ma.check_mail_write("google_mail", "create_draft")
        assert "Not updated:" in ma.check_mail_write("google_mail", "update_draft")
        assert "Not sent:" in ma.check_mail_write("google_mail", "send")
        assert "Not sent:" in ma.check_mail_write("google_mail", "send_draft")
        assert "Not sent:" in ma.check_mail_write("send_mail", None)
        assert "started by a flow," in ma.check_mail_write("send_mail", None)


def test_labels_cover_every_kind():
    assert set(ma.KIND_LABELS) == set(ma.AUTOMATION_KINDS)
    assert "mcp_task" in ma.AUTOMATION_KINDS
    assert ma.KIND_LABELS["mcp_task"] == "a task sent over MCP"
    assert ma.KIND_LABELS["autonomy"] == ma.KIND_LABELS["autonomy_tool"] == "Autonomous Work"
    assert ma.KIND_LABELS["unknown"] == "an automation"


def test_automated_prefix():
    assert ma.automated_prefix() == ""
    with ma.bound(ma.automated("fd_scheduler")):
        assert ma.automated_prefix() == (
            "[Automated turn — a scheduled job. Not a live message from the user.]"
        )
    with ma.bound(ma.human("hi")):
        assert ma.automated_prefix() == ""


def test_strip_automated_prefix():
    with ma.bound(ma.automated("cron")):
        line = ma.automated_prefix()
    assert ma.strip_automated_prefix(line + "\nrun a basna on X") == "run a basna on X"
    assert ma.strip_automated_prefix("[Basna run 'x' finished] ok") == "[Basna run 'x' finished] ok"
    assert ma.strip_automated_prefix("hello") == "hello"


def test_basna_detector_sees_the_job_text_under_the_prefix():
    from captain_claw.agent_orchestration_mixin import _detect_basna_run

    with ma.bound(ma.automated("fd_scheduler")):
        line = ma.automated_prefix()
    plain = _detect_basna_run("run a basna on the Q3 market for Croatia")
    assert plain
    assert _detect_basna_run(line + "\nrun a basna on the Q3 market for Croatia") == plain
    assert _detect_basna_run(line + "\n[Basna run 'x' finished] run a basna again") is None


# ── from_wire ────────────────────────────────────────────────────────


def test_from_wire():
    assert ma.from_wire(None, default_kind="autonomy_tool") is None
    a = ma.from_wire({"kind": "nope", "job_text": "x", "mail_write": "allow"}, default_kind="flow_tool")
    assert (a.mode, a.kind, a.mail_write) == ("automated", "unknown", "deny")
    a = ma.from_wire({"kind": "cron", "mail_write": "maybe"}, default_kind="x")
    assert (a.kind, a.mail_write) == ("cron", "deny")
    a = ma.from_wire("autonomy", default_kind="x")
    assert (a.mode, a.kind, a.job_text, a.mail_write) == ("automated", "unknown", "", "deny")
    a = ma.from_wire({"kind": "fd_scheduler", "job_text": "y" * 5000, "mail_write": "intent"},
                     default_kind="x")
    assert (a.kind, a.mail_write, len(a.job_text)) == ("fd_scheduler", "intent", 4000)
    a = ma.from_wire({"kind": "autonomy_tool", "mail_write": "allow"}, default_kind="x")
    assert (a.kind, a.job_text, a.mail_write) == ("autonomy_tool", "", "allow")


# ── process default + binding ────────────────────────────────────────


@pytest.mark.parametrize("env,kind", [
    ("CLAW_BEING_WORKER", "being"), ("CLAW_BASNA_WORKER", "fd_worker"),
    ("CLAW_VATRA_WORKER", "fd_worker"), ("CLAW_COUNCIL_WORKER", "fd_worker"),
    ("CLAW_CODE_AGENT", "fd_worker"),
])
def test_interactive_keeps_the_worker_default(monkeypatch, env, kind):
    """J7: a plain chat frame in a worker / being is an LLM-written prompt, not a person."""
    assert ma.interactive("draft a reply to Ana") == ma.human("draft a reply to Ana")
    monkeypatch.setenv(env, "1")
    a = ma.interactive("Draft emails to every investor about Q3")
    assert (a.mode, a.kind, a.job_text, a.mail_write) == ("automated", kind, "", "deny")
    with ma.bound(a):
        assert ma.is_refusal(ma.check_mail_write("google_mail", "create_draft"))
        assert ma.intent_source_text(None) == ""
        assert ma.recent_intent_text(None) == ""


def test_process_default(monkeypatch):
    assert ma.current() == ma.HUMAN
    monkeypatch.setenv("CLAW_BEING_WORKER", "1")
    assert (ma.current().kind, ma.current().mail_write) == ("being", "deny")
    monkeypatch.delenv("CLAW_BEING_WORKER")
    monkeypatch.setenv("CLAW_VATRA_WORKER", "true")
    assert (ma.current().mode, ma.current().kind, ma.current().mail_write) == (
        "automated", "fd_worker", "deny")
    with ma.bound(ma.human("draft a reply to Ana")):
        assert ma.current().mode == "human"
    monkeypatch.setenv("CLAW_VATRA_WORKER", "0")
    assert ma.current() == ma.HUMAN
    monkeypatch.setenv("CLAW_CODE_AGENT", "YES")
    assert ma.current().kind == "fd_worker"


async def test_context_isolation():
    seen: dict[str, ma.Authority] = {}
    gate = asyncio.Event()

    async def _lane(name: str, a: ma.Authority) -> None:
        with ma.bound(a):
            await gate.wait()
            await asyncio.sleep(0)
            seen[name] = ma.current()

    t1 = asyncio.create_task(_lane("a", ma.automated("cron", "x", "deny")))
    t2 = asyncio.create_task(_lane("b", ma.human("hello")))
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(t1, t2)
    assert seen["a"].kind == "cron" and seen["a"].mail_write == "deny"
    assert seen["b"].mode == "human" and seen["b"].job_text == "hello"
    assert ma.current() == ma.HUMAN

    with ma.bound(None):
        assert ma.current() == ma.HUMAN
    ma.reset(None)  # tolerant


# ── stored / forwarded intent ────────────────────────────────────────


def test_cron_job_text():
    assert ma.cron_job_text({"text": "draft a reply to Ana"}) == "draft a reply to Ana"
    assert ma.cron_job_text({"text": "draft a reply to Ana", "author": "agent",
                             "mail_intent_text": ""}) == ""
    assert ma.cron_job_text({"text": "Draft a reply to Ana every Friday", "author": "agent",
                             "mail_intent_text": "napravi draft za Anu svaki petak"}) != ""
    assert ma.cron_job_text({
        "text": "Summarize my inbox daily", "author": "agent",
        "mail_intent_text": "draft a reply to Ana and set up a daily inbox summary",
    }) == ""
    assert ma.cron_job_text({
        "text": "Draft replies to unanswered mail and email Stevica a summary", "author": "agent",
        "mail_intent_text": "email me a summary every morning",
    }) == "email me a summary every morning"
    assert ma.cron_job_text(None) == ""


def test_intent_source_text():
    agent = types.SimpleNamespace(_turn_user_text="draft a reply to Ana")
    with ma.bound(ma.human("draft a reply to Ana")):
        assert ma.intent_source_text(agent) == "draft a reply to Ana"
    # unbound human: the attribute is never used
    assert ma.intent_source_text(agent) == ""
    with ma.bound(ma.automated("cron", "draft a reply to Ana", "deny")):
        assert ma.intent_source_text(agent) == ""
    with ma.bound(ma.automated("cron", "draft a reply to Ana", "intent")):
        assert ma.intent_source_text(agent) == "draft a reply to Ana"


def test_soft_checks():
    agent = types.SimpleNamespace(_turn_user_text="draft a reply to Ana")
    with ma.bound(ma.human("what's new?")):
        assert ma.human_turn_text(agent) == "what's new?"
        assert ma.nudge_mail_ok(agent) is False
    assert ma.human_turn_text(agent) == "draft a reply to Ana"  # fallback
    assert ma.nudge_mail_ok(agent) is True
    for a in (ma.automated("cron", "draft a reply to Ana", "allow"),
              ma.automated("cron", "draft a reply to Ana", "intent"),
              ma.automated("cron", "", "deny")):
        with ma.bound(a):
            assert ma.human_turn_text(agent) == ""
            assert ma.nudge_mail_ok(agent) is False
    with ma.bound(ma.automated("cron", "draft a reply to Ana")):
        assert ma.stall_mail_ok(agent) is True
    with ma.bound(ma.automated("cron", "draft a reply to Ana", "deny")):
        assert ma.stall_mail_ok(agent) is False
    with ma.bound(ma.automated("autonomy_tool", "", "allow")):
        assert ma.stall_mail_ok(agent) is True
    with ma.bound(ma.human("At the moment you have a hard gate")):
        assert ma.stall_mail_ok(agent) is False


def _msg(role, content, turn_input=False):
    m = {"role": role, "content": content}
    if turn_input:
        m["turn_input"] = True
    return m


def _convo(*turns, current=None):
    """An agent whose session holds *turns* ((user, assistant) pairs) and
    then *current* (this turn's input)."""
    msgs = []
    for user, assistant in turns:
        msgs.append(_msg("user", user, True))
        if assistant is not None:
            msgs.append(_msg("assistant", assistant))
    if current is not None:
        msgs.append(_msg("user", current, True))
    return types.SimpleNamespace(session=types.SimpleNamespace(messages=msgs), _turn_user_text="")


_OFFER = "Ana is waiting for a reply about the contract. Shall I draft a reply to her?"


@pytest.mark.parametrize("yes", [
    "yes do it", "yes please", "draft it", "go ahead", "ok send it", "da, napravi",
    "odgovori mu", "odgovori joj da može", "reply to him", "napiši mu odgovor",
    "go ahead and draft the reply", "yes, reply to her saying Thursday works", "može",
])
def test_a_yes_to_the_agents_offer_counts_as_asked(yes):
    agent = _convo(("any news from Ana?", _OFFER), current=yes)
    with ma.bound(ma.human(yes)):
        assert ma.human_asked_for_mail(agent) is True
        assert ma.nudge_mail_ok(agent) is True
        assert ma.stall_mail_ok(agent) is True


@pytest.mark.parametrize("reply", ["no", "ne, nemoj", "don't send it", "wait", "thanks"])
def test_a_no_or_a_thanks_is_not_asked(reply):
    agent = _convo(("any news from Ana?", _OFFER), current=reply)
    with ma.bound(ma.human(reply)):
        assert ma.human_asked_for_mail(agent) is False


def test_a_yes_without_an_email_offer_is_not_asked():
    agent = _convo(("make me a chart", "Shall I add a legend?"), current="yes do it")
    with ma.bound(ma.human("yes do it")):
        assert ma.human_asked_for_mail(agent) is False


@pytest.mark.parametrize("said", [
    "You have 3 new emails from Ana. Anything else?",
    "Ana's reply says the meeting moved. Anything else?",
    "Evo odgovora na tvoje pitanje o mailu. Trebaš li još nešto?",
    "Did Ana reply to your email?",
    "Do you want a summary of the replies in your email?",
])
def test_an_ok_to_a_message_that_only_mentions_email_is_not_asked(said):
    """Only an offer to WRITE one counts — not email in passing plus a "?"."""
    agent = _convo(("what's new?", said), current="ok")
    with ma.bound(ma.human("ok")):
        assert ma.human_asked_for_mail(agent) is False


@pytest.mark.parametrize("offer", [
    "Here is the draft:\n\nTo: ana@x.co\nHi Ana …\n\nWant me to send it?",
    "Ana pita za ugovor. Želiš li da joj odgovorim?",
    "Ana pita za ugovor. Mogu li napisati odgovor?",
    "Ana asked about the invoice. I can draft a reply if you'd like.",
    "Would you like me to email her?",
    "Draft a reply to Ana?",
])
def test_offers_to_write_count(offer):
    agent = _convo(("anything from Ana?", offer), current="yes")
    with ma.bound(ma.human("yes")):
        assert ma.human_asked_for_mail(agent) is True


def test_asked_looks_at_the_last_three_human_messages():
    agent = _convo(("Every Friday email Ana the KPI report", "Should I set it up for 9:00?"),
                   ("what time is it?", "It's 10:00."), current="yes, set it up")
    with ma.bound(ma.human("yes, set it up")):
        assert ma.recent_human_texts(agent) == [
            "yes, set it up", "what time is it?", "Every Friday email Ana the KPI report"]
        assert ma.human_asked_for_mail(agent) is True
    older = _convo(("Every Friday email Ana the KPI report", "ok"), ("a", "b"), ("c", "d"),
                   current="yes")
    with ma.bound(ma.human("yes")):
        assert ma.human_asked_for_mail(older) is False


def test_recent_messages_skip_nudges_and_automated_turns():
    msgs = [
        _msg("user", "draft a reply to Ana", True),
        _msg("assistant", "Done."),
        _msg("user", "[Automated turn — a scheduled job. Not a live message from the user.]\n"
                     "draft replies to everyone", True),
        _msg("assistant", "ok"),
        _msg("user", "hello", True),
        _msg("assistant", "I'll draft the reply now."),
        _msg("user", "You announced intent without acting. Do NOT create, update or send any "
                     "email or draft"),
        _msg("user", "thanks", True),
    ]
    agent = types.SimpleNamespace(session=types.SimpleNamespace(messages=msgs), _turn_user_text="")
    with ma.bound(ma.human("thanks")):
        assert ma.recent_human_texts(agent) == ["thanks", "hello", "draft a reply to Ana"]
    # this turn's own message isn't in this session: only the bound text
    with ma.bound(ma.human("something else")):
        assert ma.recent_human_texts(agent) == ["something else"]


def test_recent_intent_text():
    agent = _convo(("Every Friday email Ana the KPI report", "Should I set it up?"),
                   current="yes, set it up")
    with ma.bound(ma.human("yes, set it up")):
        assert ma.recent_intent_text(agent) == (
            "yes, set it up\nEvery Friday email Ana the KPI report")
    # never the agent attribute, never in a deny turn
    agent._turn_user_text = "draft a reply to Ana"
    assert ma.recent_intent_text(agent) == ""
    with ma.bound(ma.automated("cron", "draft a reply to Ana", "deny")):
        assert ma.recent_intent_text(agent) == ""
    with ma.bound(ma.automated("cron", "draft a reply to Ana", "intent")):
        assert ma.recent_intent_text(agent) == "draft a reply to Ana"


def test_turn_input_key_matches_member_privacy():
    from captain_claw import member_privacy

    assert ma._TURN_INPUT_KEY == member_privacy.TURN_INPUT


def test_parse_recipients():
    assert ma.parse_recipients("Ana <ANA@x.co>, bob@x.co", None, ["c@x.co"]) == [
        "ana@x.co", "bob@x.co", "c@x.co"]
    assert ma.parse_recipients(None, "", []) == []


def test_module_has_no_flight_deck_import():
    import inspect

    src = inspect.getsource(ma)
    assert "captain_claw.flight_deck" not in src and "import flight_deck" not in src
