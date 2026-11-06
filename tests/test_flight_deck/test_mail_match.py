"""Which agent an email is for: its own inbox, an address, the subject, or the body addressing it."""

from __future__ import annotations

import pytest

from captain_claw.flight_deck.mail_match import Agent, Mail, automated_reason, match, why_words

AGENTS = [
    Agent("researcher", ("Researcher", "Rita"), ("alice+rita@gmail.com",)),
    Agent("writer", ("Writer",), ()),
    Agent("istrazivac", ("Istraživač",), ()),
    Agent("ops", ("Ops Desk",), ()),
]


@pytest.mark.parametrize("mail, slug, why", [
    (Mail(subject="Researcher: Q3 numbers"), "researcher", "subject"),
    (Mail(subject="[Writer] copy for the site"), "writer", "subject"),
    (Mail(subject="For RITA — urgent"), "researcher", "subject"),
    (Mail(subject="Ask the ops desk"), "ops", "subject"),
    (Mail(subject="Hello", body="Hi Rita,\nplease find the file."), "researcher", "body"),
    (Mail(subject="x", body="Some context first.\n\nWriter, could you draft this?"), "writer", "body"),
    (Mail(subject="x", body="This is for Researcher."), "researcher", "body"),
    (Mail(subject="x", body="@writer take this one"), "writer", "body"),
    (Mail(subject="x", body="Bok Istraživaču? no — bok istrazivac, trebam podatke"), "istrazivac", "body"),
    (Mail(subject="x", body="Attn: Writer\nthe numbers"), "writer", "body"),
    (Mail(subject="x", to=("ALICE+Rita@gmail.com",)), "researcher", "address"),
    (Mail(subject="Writer: please", cc=("alice+rita@gmail.com",)), "researcher", "address"),
    (Mail(subject="Writer and Rita"), "writer", "subject"),
])
def test_it_finds_the_agent(mail, slug, why):
    m = match(mail, AGENTS)
    assert m is not None and (m.slug, m.why) == (slug, why)


@pytest.mark.parametrize("mail", [
    Mail(subject="Writers block"),
    Mail(subject="x", body="a good writer would know"),
    Mail(subject="x", body="mail rita@corp.com about it"),
    Mail(subject="x", body="We met the researchers yesterday"),
    Mail(subject=""),
])
def test_a_passing_word_is_not_an_address(mail):
    assert match(mail, AGENTS) is None


def test_its_own_inbox_is_its_own():
    assert match(Mail(subject="Writer: hi"), AGENTS, own_inbox="researcher").why == "own_inbox"


@pytest.mark.parametrize("headers, sender, why", [
    ({}, "no-reply@bank.co", "a no-reply sender"),
    ({}, "notifications@github.com", "a no-reply sender"),
    ({"Auto-Submitted": "auto-replied"}, "ana@x.co", "an automatic email"),
    ({"Precedence": "bulk"}, "ana@x.co", "bulk mail"),
    ({"List-Unsubscribe": "<mailto:x>"}, "ana@x.co", "a mailing list or newsletter"),
    ({"Auto-Submitted": "no"}, "ana@x.co", ""),
    ({}, "ana@x.co", ""),
])
def test_automatic_mail_is_nobodys(headers, sender, why):
    assert automated_reason(Mail(sender=sender, headers=headers)) == why


def test_how_the_agent_is_told():
    m = match(Mail(subject="Rita?"), AGENTS)
    assert why_words(m, where="your owner's Gmail") == "in your owner's Gmail, with your name in the subject"
    assert why_words(match(Mail(), AGENTS, own_inbox="r"), where="", own_email="r@x.co") == "in your own inbox (r@x.co)"
