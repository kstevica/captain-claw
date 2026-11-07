"""Custom tools, Phase 7: what a tool may ask the agent that ran it to do
(external_tool_uses). Publishing refuses what no tool may use; each request is
checked against what the tool declared and the risk of the real call."""

from __future__ import annotations

import pytest

from captain_claw.flight_deck import external_tool_uses as uses


def _one(**entry) -> dict:
    return uses.parse_uses([{"why": "it needs it", **entry}])[0]


def test_read_and_write_tools_can_be_declared():
    assert _one(tool="web_fetch", hosts=["API.Example.com."]) == {
        "tool": "web_fetch", "why": "it needs it", "hosts": ["api.example.com"]}
    assert _one(tool="google_mail", actions=["Search", "read"])["actions"] == ["search", "read"]
    assert _one(tool="write")["tool"] == "write"
    # Sending may be declared: it works only when the owner runs the tool.
    assert _one(tool="google_mail", actions=["send"])["actions"] == ["send"]
    assert uses.parse_uses(None) == ()


@pytest.mark.parametrize("entry, words", [
    ({"tool": "shell"}, "runs code"),
    ({"tool": "browser"}, "runs code"),
    ({"tool": "payments"}, "moves money"),
    ({"tool": "cron"}, "sets up later work"),
    ({"tool": "personality"}, "changes the agent"),
    ({"tool": "custom_tools"}, "sets up later work"),
    ({"tool": "no_such_tool"}, "doesn't know"),
    ({"tool": "mcp_github_search"}, "doesn't know"),
    ({"tool": "web_fetch"}, "hosts"),
    ({"tool": "web_fetch", "hosts": ["not a host"]}, "isn't a host name"),
    ({"tool": "write", "hosts": ["example.com"]}, "only for the tools that fetch"),
    ({"tool": "google_mail", "actions": ["send", "DROP TABLE"]}, "actions"),
])
def test_what_no_custom_tool_may_use(entry, words):
    with pytest.raises(uses.UsesError) as exc:
        uses.parse_uses([{"why": "because", **entry}])
    assert words in str(exc.value)


def test_each_use_says_why_once():
    with pytest.raises(uses.UsesError, match="why"):
        uses.parse_uses([{"tool": "write"}])
    with pytest.raises(uses.UsesError, match="twice"):
        uses.parse_uses([{"tool": "write", "why": "a"}, {"tool": "write", "why": "b"}])
    with pytest.raises(uses.UsesError, match="at most"):
        uses.parse_uses([{"tool": "write", "why": "a"}] * 21)


DECLARED = uses.parse_uses([
    {"tool": "web_fetch", "hosts": ["example.com"], "why": "rates"},
    {"tool": "google_mail", "actions": ["search", "send"], "why": "mail"},
    {"tool": "google_calendar", "why": "the diary"},
    {"tool": "write", "why": "a report"},
])


def test_a_request_must_be_declared():
    assert "isn't in this tool's uses" in uses.check_request(DECLARED, "read", {"path": "x"}, owner_runs=True)
    assert "only use google_mail search, send" in uses.check_request(
        DECLARED, "google_mail", {"action": "trash", "id": "1"}, owner_runs=True)
    assert uses.check_request(DECLARED, "google_mail", {"action": "search", "query": "x"}, owner_runs=False) is None
    assert uses.check_request(DECLARED, "write", {"path": "a.md", "content": "x"}, owner_runs=False) is None


def test_sending_and_invitations_only_when_the_owner_runs_it():
    send = {"action": "send", "to": "x@example.com"}
    assert uses.check_request(DECLARED, "google_mail", send, owner_runs=True) is None
    assert "reaches other people" in uses.check_request(DECLARED, "google_mail", send, owner_runs=False)
    invite = {"action": "create_event", "summary": "x", "attendees": ["a@example.com"]}
    assert uses.risk("google_calendar", invite) == "send"
    assert "reaches other people" in uses.check_request(DECLARED, "google_calendar", invite, owner_runs=False)
    assert uses.check_request(DECLARED, "google_calendar", {"action": "list_events"}, owner_runs=False) is None
    assert uses.check_request(DECLARED, "google_calendar", invite, owner_runs=True) is None


@pytest.mark.parametrize("url, ok", [
    ("https://example.com/rates", True),
    ("https://api.example.com/rates?x=1", True),
    ("https://example.com.evil.net/", False),
    ("https://evil.net/?d=example.com", False),
    ("ftp://example.com/x", False),
    ("https://user:pw@example.com/", False),
    ("javascript:alert(1)", False),
])
def test_web_reads_only_from_the_declared_hosts(url, ok):
    why = uses.check_request(DECLARED, "web_fetch", {"url": url}, owner_runs=False)
    assert (why is None) is ok, why


def test_every_url_of_a_batch_is_checked():
    assert uses.check_request(DECLARED, "web_fetch", {"urls": ["https://example.com/a"]}, owner_runs=False) is None
    batch = uses.parse_uses([{"tool": "web_fetch_batch", "hosts": ["example.com"], "why": "x"}])
    assert "isn't on this tool's hosts" in uses.check_request(
        batch, "web_fetch_batch", {"urls": ["https://example.com/a", "https://evil.net/"]}, owner_runs=True)
    assert "no URL" in uses.check_request(DECLARED, "web_fetch", {}, owner_runs=True)


def test_described_in_a_few_words():
    assert uses.describe(DECLARED) == ["web_fetch (example.com)", "google_mail (search, send)", "google_calendar", "write"]
