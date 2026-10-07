"""google_mail send / send_draft / list_drafts / update_draft + the shared
gmail_compose helpers.

Sending is opt-in. Under Flight Deck the tool never sends itself: it POSTs to
``/fd/google/gmail/send`` (FD's per-user policy decides, sends, audits) and
relays FD's answer. Standalone it sends directly, only with
``tools.google_mail.allow_send`` on. create_draft's reply threading moved into
``gmail_compose.fetch_reply_context`` — pinned here so the refactor changed
nothing.

No repeats: create_draft / send refuse an email already in Drafts or Sent
(``gmail_compose.find_repeats``) unless ``allow_repeat``; the rule itself is
pinned in every prompt text a model sees (section files, roster one-liners,
the tool description members get, the tool-avoidance nudge).

No network: Gmail and Flight Deck answer through httpx.MockTransport.
"""

from __future__ import annotations

import base64
import email
import email.policy
import json
import time
from pathlib import Path

import httpx
import pytest

import captain_claw.session as session_mod
from captain_claw import gmail_compose
from captain_claw.google_oauth import GoogleOAuthTokens
from captain_claw.google_oauth_manager import GoogleOAuthManager
from captain_claw.tools.google_mail import GoogleMailTool

_G = "https://www.googleapis.com/auth/"
READ = _G + "gmail.readonly"
COMPOSE = _G + "gmail.compose"
GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me"
_RealAsyncClient = httpx.AsyncClient  # tests patch httpx.AsyncClient itself


# ── fixtures (as tests/test_google_identity.py) ──────────────────────


@pytest.fixture(autouse=True)
def _no_real_session_manager(monkeypatch):
    # The tool builds GoogleOAuthManager(get_session_manager()); never touch a DB.
    monkeypatch.setattr(session_mod, "get_session_manager", lambda: object())


@pytest.fixture(autouse=True)
def _isolated_fd_home(monkeypatch, tmp_path):
    from captain_claw.flight_deck import agent_secret

    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    agent_secret.reset_cache_for_tests()
    yield
    agent_secret.reset_cache_for_tests()


@pytest.fixture(autouse=True)
def _config(monkeypatch):
    """A fresh default Config per test (sending off); tests flip knobs on it."""
    import captain_claw.config as config_mod
    from captain_claw.config import Config

    cfg = Config()
    monkeypatch.setattr(config_mod, "_config", cfg)
    return cfg


def _set_mode(monkeypatch, *, fd: bool, token: str | None = "T",
              scope: str = f"{READ} {COMPOSE}") -> dict:
    """Flight Deck client mode on/off + the token the manager hands out."""
    calls = {"get_tokens": 0}
    monkeypatch.setattr(
        GoogleOAuthManager, "_flight_deck_base",
        staticmethod(lambda: "http://localhost:25080" if fd else ""),
    )

    async def _get_tokens(self):
        calls["get_tokens"] += 1
        if token is None:
            return None
        return GoogleOAuthTokens(access_token=token, refresh_token="",
                                 expires_at=time.time() + 3300, scope=scope)

    monkeypatch.setattr(GoogleOAuthManager, "get_tokens", _get_tokens)
    return calls


class _Gmail:
    """Scripted Gmail API: (method, path suffix) → JSON; records requests."""

    def __init__(self, routes: dict[tuple[str, str], tuple[int, dict]]):
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.split("/gmail/v1/users/me", 1)[-1]
        status, body = self.routes.get((request.method, path), (404, {"error": {"message": "nope"}}))
        return httpx.Response(status, json=body)

    def json_of(self, method: str, path: str) -> dict:
        for r in self.requests:
            if r.method == method and r.url.path.endswith(path):
                return json.loads(r.content)
        raise AssertionError(f"no {method} {path}: {[(r.method, r.url.path) for r in self.requests]}")


def _tool_with(gmail: _Gmail) -> GoogleMailTool:
    tool = GoogleMailTool()
    tool._client = httpx.AsyncClient(transport=httpx.MockTransport(gmail))
    return tool


def _decode(raw: str) -> email.message.EmailMessage:
    data = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    return email.message_from_bytes(data, policy=email.policy.default)


ORIGINAL = {
    "id": "m1",
    "threadId": "t1",
    "payload": {"headers": [
        {"name": "From", "value": "Alice <alice@ok.com>"},
        {"name": "Reply-To", "value": "Team <team@ok.com>"},
        {"name": "Subject", "value": "Quarterly numbers"},
        {"name": "Message-ID", "value": "<orig@ok.com>"},
        {"name": "References", "value": "<root@ok.com>"},
    ]},
}


# ── gmail_compose helpers ────────────────────────────────────────────


def test_build_raw_message_round_trip():
    raw = gmail_compose.build_raw_message(
        to="Bob <bob@x.co>", cc="c@x.co", bcc="d@x.co", subject="Hi there",
        body="line one\nline two", in_reply_to="<orig@ok.com>",
        references="<root@ok.com> <orig@ok.com>",
    )
    assert "=" not in raw  # base64url, padding stripped
    msg = _decode(raw)
    assert msg["To"] == "Bob <bob@x.co>"
    assert msg["Cc"] == "c@x.co" and msg["Bcc"] == "d@x.co"
    assert msg["Subject"] == "Hi there"
    assert msg["In-Reply-To"] == "<orig@ok.com>"
    assert msg["References"] == "<root@ok.com> <orig@ok.com>"
    plain = msg.get_body(("plain",)).get_content()
    html = msg.get_body(("html",)).get_content()
    assert "line one\nline two" in plain.replace("\r\n", "\n")
    assert "line one<br>line two" in html
    # The tool's old static entry points still work (thin delegates).
    assert _decode(GoogleMailTool._build_raw_message(to="a@b.co", body="x"))["To"] == "a@b.co"
    assert GoogleMailTool._text_to_html("a\nb") == gmail_compose.text_to_html("a\nb")
    assert GoogleMailTool._html_to_text("<p>a</p>b") == "a\nb"


def test_allowlist_forms_and_matching():
    norm = gmail_compose.normalize_allowlist(
        [" Alice@OK.com ", "ok.org", "@ok.org", "@Corp.IO", ""],
    )
    assert norm == ["alice@ok.com", "@ok.org", "@corp.io"]
    allowed = gmail_compose.recipients_allowed(
        ["alice@ok.com", "bob@ok.org", "eve@mail.ok.org", "mallory@ok.com", "x@corp.io"], norm,
    )
    # Exact address; domain exactly — no subdomains, no other user of a listed address's domain.
    assert allowed == ["eve@mail.ok.org", "mallory@ok.com"]
    assert gmail_compose.recipients_allowed(["anyone@any.where"], []) == []
    for bad in (["not an address"], ["@"], ["a@b"], [42], "alice@ok.com"):
        with pytest.raises(ValueError):
            gmail_compose.normalize_allowlist(bad)


def test_recipient_parsing_and_rendering():
    assert gmail_compose.parse_addresses(
        "Alice <A@X.com>, bob@y.org", "", "BOB@y.org, c@z.io",
    ) == ["a@x.com", "bob@y.org", "c@z.io"]
    assert gmail_compose.invalid_recipients("a@x.com", "Bob") == ["Bob"]
    assert gmail_compose.invalid_recipients("a@x.com; b@y.com")  # unparseable value
    assert gmail_compose.invalid_recipients("a@ok.com\r\nBcc: e@bad.com")
    assert gmail_compose.invalid_recipients("Alice <a@x.com>", "", "c@z.io") == []
    assert gmail_compose.format_recipients('"Doe, John" <j@d.com>,, k@l.mn') == '"Doe, John" <j@d.com>, k@l.mn'


def test_content_hash_normalizes():
    a = gmail_compose.content_hash("Bob <BOB@x.co>, c@x.co", "", "", "Hello  there", "hi \r\nyou\n")
    b = gmail_compose.content_hash("c@x.co, bob@x.co", "", "", "hello there", "hi\nyou")
    c = gmail_compose.content_hash("c@x.co, bob@x.co", "", "", "hello there", "hi\nyou too")
    assert a == b != c


# ── create_draft: unchanged by the refactor ──────────────────────────


@pytest.mark.asyncio
async def test_create_draft_reply_threading_unchanged(monkeypatch):
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({
        ("GET", "/messages/m1"): (200, ORIGINAL),
        ("POST", "/drafts"): (200, {"id": "d9", "message": {"id": "m9"}}),
    })
    res = await _tool_with(gmail).execute("create_draft", reply_to_message_id="m1", body="Sounds good")

    assert res.success, res.error
    assert "Draft ID: d9" in res.content and "Message ID: m9" in res.content
    assert "Threaded under: t1 (reply to m1)" in res.content
    meta = gmail.requests[0]
    assert meta.url.params.get_list("metadataHeaders") == [
        "From", "To", "Cc", "Subject", "Message-ID", "References", "Reply-To",
    ]
    sent = gmail.json_of("POST", "/drafts")["message"]
    assert sent["threadId"] == "t1"
    msg = _decode(sent["raw"])
    assert msg["To"] == "Team <team@ok.com>"  # Reply-To wins over From
    assert msg["Subject"] == "Re: Quarterly numbers"
    assert msg["In-Reply-To"] == "<orig@ok.com>"
    assert msg["References"] == "<root@ok.com> <orig@ok.com>"


@pytest.mark.asyncio
async def test_create_draft_reply_lookup_error_is_reported(monkeypatch):
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({})  # 404 for the original
    res = await _tool_with(gmail).execute("create_draft", reply_to_message_id="gone", body="x")
    assert not res.success and res.error == "Message or thread not found."
    assert [r.method for r in gmail.requests] == ["GET"]


# ── standalone send ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_local_send_refused_when_allow_send_is_off(monkeypatch):
    calls = _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({})
    tool = _tool_with(gmail)
    for action, kw in (("send", {"to": "bob@x.co", "subject": "s", "body": "b"}),
                       ("send_draft", {"draft_id": "d1"})):
        res = await tool.execute(action, **kw)
        assert not res.success
        assert "tools.google_mail.allow_send" in res.error
        assert "create_draft" in res.error
    assert gmail.requests == [] and calls["get_tokens"] == 0


@pytest.mark.asyncio
async def test_local_send_posts_raw_and_thread_when_replying(monkeypatch, _config):
    _config.tools.google_mail.allow_send = True
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({
        ("GET", "/messages/m1"): (200, ORIGINAL),
        ("POST", "/messages/send"): (200, {"id": "s1", "threadId": "t1"}),
    })
    res = await _tool_with(gmail).execute("send", reply_to_message_id="m1", body="Thanks!")

    assert res.success, res.error
    assert "Email sent." in res.content and "Message ID: s1" in res.content
    assert "Thread ID: t1" in res.content and "Subject: Re: Quarterly numbers" in res.content
    sent = gmail.json_of("POST", "/messages/send")
    assert sent["threadId"] == "t1"
    msg = _decode(sent["raw"])
    assert msg["To"] == "Team <team@ok.com>"
    assert msg["Subject"] == "Re: Quarterly numbers"
    assert msg["In-Reply-To"] == "<orig@ok.com>"
    assert gmail.requests[-1].headers["Authorization"] == "Bearer T"


@pytest.mark.asyncio
async def test_local_send_validates_before_calling_gmail(monkeypatch, _config):
    _config.tools.google_mail.allow_send = True
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({("POST", "/messages/send"): (200, {"id": "s1"})})
    tool = _tool_with(gmail)
    many = ", ".join(f"p{i}@x.co" for i in range(gmail_compose.MAX_RECIPIENTS + 1))
    for kw, needle in (
        ({"to": "bob@x.co", "subject": "s"}, "subject and a body"),
        ({"subject": "s", "body": "b"}, "No recipient"),
        ({"to": "Bob", "subject": "s", "body": "b"}, "Invalid recipient"),
        ({"to": many, "subject": "s", "body": "b"}, "Too many recipients"),
    ):
        res = await tool.execute("send", **kw)
        assert not res.success and needle in res.error, (kw, res.error)
    assert gmail.requests == []


@pytest.mark.asyncio
async def test_local_send_allowlist_refusal(monkeypatch, _config):
    _config.tools.google_mail.allow_send = True
    _config.tools.google_mail.allowed_recipients = ["@ok.com"]
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({("POST", "/messages/send"): (200, {"id": "s1"})})
    res = await _tool_with(gmail).execute(
        "send", to="alice@ok.com", cc="evil@bad.com", subject="s", body="b",
    )
    assert not res.success
    assert "evil@bad.com" in res.error and "allowed_recipients" in res.error
    assert "alice@ok.com" not in res.error  # only the disallowed one is named
    assert gmail.requests == []


@pytest.mark.asyncio
async def test_local_send_needs_a_send_capable_scope(monkeypatch, _config):
    _config.tools.google_mail.allow_send = True
    _set_mode(monkeypatch, fd=False, scope=READ)
    gmail = _Gmail({})
    res = await _tool_with(gmail).execute("send", to="bob@x.co", subject="s", body="b")
    assert not res.success and "send scope not granted" in res.error
    assert gmail.requests == []


@pytest.mark.asyncio
async def test_local_send_draft_posts_drafts_send(monkeypatch, _config):
    _config.tools.google_mail.allow_send = True
    _set_mode(monkeypatch, fd=False, scope=COMPOSE)
    gmail = _Gmail({
        ("GET", "/drafts/d1"): (200, {"id": "d1", "message": {"id": "m5", "payload": {"headers": [
            {"name": "To", "value": "Bob <bob@x.co>"}, {"name": "Subject", "value": "Draft subj"},
        ]}}}),
        ("POST", "/drafts/send"): (200, {"id": "s5", "threadId": "t5"}),
    })
    res = await _tool_with(gmail).execute("send_draft", draft_id="d1")

    assert res.success, res.error
    assert gmail.json_of("POST", "/drafts/send") == {"id": "d1"}
    assert "To: Bob <bob@x.co>" in res.content and "Subject: Draft subj" in res.content
    assert "Message ID: s5" in res.content
    assert gmail.requests[0].url.params["format"] == "metadata"


@pytest.mark.asyncio
async def test_local_send_draft_not_found(monkeypatch, _config):
    _config.tools.google_mail.allow_send = True
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({})
    res = await _tool_with(gmail).execute("send_draft", draft_id="gone")
    assert not res.success and "Draft gone not found" in res.error and "list_drafts" in res.error


# ── list_drafts ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_drafts_formatting_with_compose_only_scope(monkeypatch):
    _set_mode(monkeypatch, fd=False, scope=COMPOSE)
    gmail = _Gmail({
        ("GET", "/drafts"): (200, {"drafts": [{"id": "d1", "message": {"id": "m1"}}]}),
        ("GET", "/drafts/d1"): (200, {"id": "d1", "message": {
            "id": "m1", "snippet": "Hi Bob, attached…",
            "payload": {"headers": [
                {"name": "To", "value": "bob@x.co"}, {"name": "Subject", "value": "Report"},
            ]},
        }}),
    })
    res = await _tool_with(gmail).execute("list_drafts", query="to:bob", max_results=5)

    assert res.success, res.error
    assert gmail.requests[0].url.params["q"] == "to:bob"
    assert gmail.requests[0].url.params["maxResults"] == "5"
    for needle in ("Report", "To: bob@x.co", "Draft ID: d1", "Message ID: m1",
                   "Preview: Hi Bob, attached…", "action=send_draft"):
        assert needle in res.content, needle


@pytest.mark.asyncio
async def test_list_drafts_empty(monkeypatch):
    _set_mode(monkeypatch, fd=False)
    res = await _tool_with(_Gmail({("GET", "/drafts"): (200, {})})).execute("list_drafts")
    assert res.success and res.content == "No drafts found."


# ── Flight Deck mode: FD sends ───────────────────────────────────────


def _fd_answers(monkeypatch, status: int, body: dict, headers: dict | None = None) -> list:
    """Real GoogleOAuthManager in FD client mode; FD answers every request with
    (*status*, *body*). Returns the requests it received."""
    import captain_claw.config as config_mod
    from captain_claw.config import Config

    monkeypatch.setattr(config_mod, "_config", Config(
        web={"auth_token": "agent-web-auth"},
        google_oauth={"flight_deck_url": "http://fd.test", "flight_deck_secret": ""},
    ))
    monkeypatch.delenv("FD_AGENT_SHARED_SECRET", raising=False)
    seen: list = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json=body, headers=headers or {})

    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kw: _RealAsyncClient(transport=httpx.MockTransport(handler), **kw),
    )
    return seen


_SENT = {"ok": True, "message_id": "s7", "thread_id": "t7", "to": "bob@x.co", "cc": "",
         "bcc": "", "subject": "Hello", "sent_last_24h": 3, "daily_limit": 50}


@pytest.mark.asyncio
async def test_fd_mode_send_posts_to_flight_deck_not_gmail(monkeypatch):
    seen = _fd_answers(monkeypatch, 200, _SENT)
    res = await GoogleMailTool().execute(
        "send", to="bob@x.co", subject="Hello", body="Hi", reply_to_message_id="m1",
    )

    assert res.success, res.error
    assert len(seen) == 1  # no /access_token fetch, no direct Gmail call
    req = seen[0]
    assert req.method == "POST" and str(req.url) == "http://fd.test/fd/google/gmail/send"
    assert req.headers["X-Agent-Auth"] == "agent-web-auth"
    assert req.headers.get("X-Agent-Secret")  # the per-deck secret file
    assert "origin" not in {k.lower() for k in req.headers}
    assert json.loads(req.content) == {
        "to": "bob@x.co", "cc": "", "bcc": "", "subject": "Hello", "body": "Hi",
        "html_body": "", "reply_to_message_id": "m1",
    }
    for needle in ("Email sent.", "To: bob@x.co", "Subject: Hello", "Message ID: s7",
                   "Thread ID: t7", "3 of 50 sends used in the last 24h"):
        assert needle in res.content, needle


@pytest.mark.asyncio
async def test_fd_mode_send_draft_sends_only_the_draft_id(monkeypatch):
    seen = _fd_answers(monkeypatch, 200, _SENT)
    res = await GoogleMailTool().execute("send_draft", draft_id="d1", to="ignored@x.co")
    assert res.success, res.error
    assert json.loads(seen[0].content) == {"draft_id": "d1"}


@pytest.mark.asyncio
async def test_fd_mode_off_surfaces_the_detail_and_points_at_drafts(monkeypatch):
    detail = "Email sending is turned off for this user, so nothing was sent."
    _fd_answers(monkeypatch, 403, {"detail": detail}, {gmail_compose.SEND_REFUSED_HEADER: "off"})
    res = await GoogleMailTool().execute("send", to="bob@x.co", subject="s", body="b")
    assert not res.success
    assert detail in res.error
    assert "create_draft" in res.error


@pytest.mark.asyncio
async def test_fd_mode_other_refusals_are_relayed_verbatim(monkeypatch):
    detail = "Unknown agent — it was not spawned by this Flight Deck"
    _fd_answers(monkeypatch, 403, {"detail": detail})
    res = await GoogleMailTool().execute("send", to="bob@x.co", subject="s", body="b")
    assert not res.success
    assert res.error == f"Email not sent: {detail}"  # no draft hint: not an "off" refusal

    _fd_answers(monkeypatch, 429, {"detail": "Daily send limit reached: 50 of 50"})
    res = await GoogleMailTool().execute("send", to="bob@x.co", subject="s", body="b")
    assert not res.success and "Daily send limit reached" in res.error


@pytest.mark.asyncio
async def test_fd_mode_unreachable_says_nothing_was_sent(monkeypatch):
    import captain_claw.config as config_mod
    from captain_claw.config import Config

    monkeypatch.setattr(config_mod, "_config", Config(
        web={"auth_token": "agent-web-auth"},
        google_oauth={"flight_deck_url": "http://fd.test"},
    ))

    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: _RealAsyncClient(transport=httpx.MockTransport(handler), **kw))
    res = await GoogleMailTool().execute("send", to="bob@x.co", subject="s", body="b")
    assert not res.success and "Nothing was sent" in res.error


# ── review fixes ─────────────────────────────────────────────────────


def test_non_ascii_addresses_are_invalid_and_named():
    for addr in ("josé@example.com", "info@münchen.de"):
        assert gmail_compose.invalid_recipients(f"Bob <{addr}>") == [addr]
        message = gmail_compose.invalid_recipients_message([addr])
        assert addr in message and "ASCII" in message and "Nothing was sent" in message
        with pytest.raises(ValueError):
            gmail_compose.format_recipients(addr)
    assert gmail_compose.invalid_recipients("Željko <z@xn--mnchen-3ya.de>") == []


def test_format_recipients_keeps_display_names_readable():
    assert gmail_compose.format_recipients("Željko Horvat <z@x.com>") == "Željko Horvat <z@x.com>"
    assert gmail_compose.format_recipients('"Horvat, Željko" <z@x.com>') == '"Horvat, Željko" <z@x.com>'
    rendered = gmail_compose.format_recipients("Željko Horvat <z@x.com>, k@l.mn")
    msg = _decode(gmail_compose.build_raw_message(to=rendered, subject="s", body="b"))
    assert [(a.display_name, a.addr_spec) for a in msg["To"].addresses] == [
        ("Željko Horvat", "z@x.com"), ("", "k@l.mn")]


def test_content_hash_includes_the_reply_target():
    args = ("bookings@hotel.com", "", "", "Re: Your booking", "Confirmed, thanks")
    plain = gmail_compose.content_hash(*args)
    assert gmail_compose.content_hash(*args, reply_to="") == plain
    m1 = gmail_compose.content_hash(*args, reply_to="m1")
    assert m1 == gmail_compose.content_hash(*args, reply_to=" m1 ")
    assert len({plain, m1, gmail_compose.content_hash(*args, reply_to="m2")}) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["SENT", "DRAFT"])
async def test_reply_to_own_email_defaults_to_its_recipients(label):
    own = {"id": "mine", "threadId": "t1", "labelIds": ["INBOX", label], "payload": {"headers": [
        {"name": "From", "value": "Owner <owner@x.co>"},
        {"name": "To", "value": "Bob <bob@ok.com>, carol@ok.com"},
        {"name": "Reply-To", "value": "owner@x.co"},
        {"name": "Subject", "value": "Q3"},
    ]}}
    gmail = _Gmail({("GET", "/messages/mine"): (200, own),
                    ("GET", "/messages/m1"): (200, {**ORIGINAL, "labelIds": ["INBOX"]})})
    async with httpx.AsyncClient(transport=httpx.MockTransport(gmail)) as client:
        ctx = await gmail_compose.fetch_reply_context(client, "T", "mine")
        assert ctx["reply_to_default"] == "Bob <bob@ok.com>, carol@ok.com"
        other = await gmail_compose.fetch_reply_context(client, "T", "m1")
        assert other["reply_to_default"] == "Team <team@ok.com>"  # someone else's: Reply-To


@pytest.mark.asyncio
async def test_local_send_reply_to_own_sent_email_goes_to_its_recipients(monkeypatch, _config):
    _config.tools.google_mail.allow_send = True
    _set_mode(monkeypatch, fd=False)
    own = {"id": "mine", "threadId": "t1", "labelIds": ["SENT"], "payload": {"headers": [
        {"name": "From", "value": "Owner <owner@x.co>"},
        {"name": "To", "value": "Bob <bob@ok.com>"},
        {"name": "Subject", "value": "Q3"},
    ]}}
    gmail = _Gmail({("GET", "/messages/mine"): (200, own),
                    ("POST", "/messages/send"): (200, {"id": "s1", "threadId": "t1"})})
    res = await _tool_with(gmail).execute("send", reply_to_message_id="mine", body="Following up")
    assert res.success, res.error
    assert _decode(gmail.json_of("POST", "/messages/send")["raw"])["To"] == "Bob <bob@ok.com>"
    assert "To: Bob <bob@ok.com>" in res.content


@pytest.mark.asyncio
async def test_local_send_non_ascii_address_is_refused_cleanly(monkeypatch, _config):
    _config.tools.google_mail.allow_send = True
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({("POST", "/messages/send"): (200, {"id": "s1"})})
    res = await _tool_with(gmail).execute("send", to="josé@example.com", subject="s", body="b")
    assert not res.success
    assert "josé@example.com" in res.error and "ASCII" in res.error
    assert gmail.requests == []


@pytest.mark.asyncio
async def test_local_send_summary_shows_plain_display_names(monkeypatch, _config):
    _config.tools.google_mail.allow_send = True
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({("POST", "/messages/send"): (200, {"id": "s1", "threadId": "t1"})})
    res = await _tool_with(gmail).execute(
        "send", to="Željko Horvat <z@x.co>", subject="Pozdrav", body="Bok",
    )
    assert res.success, res.error
    assert "To: Željko Horvat <z@x.co>" in res.content and "=?utf-8?" not in res.content


@pytest.mark.asyncio
async def test_fd_mode_unknown_outcome_is_not_reported_as_not_sent(monkeypatch):
    detail = "Gmail did not answer (ReadTimeout) — the email may or may not have been sent."
    _fd_answers(monkeypatch, 502, {"detail": detail}, {gmail_compose.SEND_OUTCOME_HEADER: "unknown"})
    res = await GoogleMailTool().execute("send", to="bob@x.co", subject="s", body="b")
    assert not res.success
    assert res.error == f"Email may or may not have been sent: {detail}"


# ── no repeats: the same email already drafted or sent ───────────────


def _hdrs(**headers) -> list[dict]:
    return [{"name": k.replace("_", "-").title(), "value": v} for k, v in headers.items()]


def _draft(draft_id: str, msg_id: str, **headers) -> tuple[int, dict]:
    """drafts.get (format=metadata) answer."""
    return 200, {"id": draft_id, "message": {"id": msg_id, "threadId": f"t-{msg_id}",
                                             "payload": {"headers": _hdrs(**headers)}}}


def _drafts_listing(*ids: str) -> tuple[int, dict]:
    return 200, {"drafts": [{"id": d, "message": {"id": f"m-{d}"}} for d in ids]}


def _sent(msg_id: str, **headers) -> tuple[int, dict]:
    """messages.get (format=metadata) answer for one of the user's sent emails."""
    return 200, {"id": msg_id, "threadId": f"t-{msg_id}", "labelIds": ["SENT"],
                 "payload": {"headers": _hdrs(**headers)}}


def _posts(gmail: _Gmail) -> list[httpx.Request]:
    return [r for r in gmail.requests if r.method in ("POST", "PUT")]


def _calls(gmail: _Gmail) -> list[tuple[str, str]]:
    return [(r.method, r.url.path.split("/gmail/v1/users/me", 1)[-1]) for r in gmail.requests]


def test_normalize_subject_and_matching():
    norm = gmail_compose.normalize_subject
    assert norm("Re: Fwd: RE:  Quarterly   Report ") == "quarterly report"
    assert norm("Odg: Ponuda za suradnju") == "ponuda za suradnju"
    assert norm("AW: SV: Fw: x") == norm("Re[2]: X") == "x"
    assert norm("Reality check") == "reality check"  # "re" without a colon stays
    assert norm("Re:") == "" and norm("") == ""
    match = gmail_compose.subjects_match
    assert match("Odg: Ponuda za suradnju", "ponuda za  SURADNJU")
    assert match("Meeting notes 5 Oct", "Meeting notes, 5 Oct")  # near-identical
    assert not match("Q3 report", "Q4 hiring plan")
    # Different numbers are different emails, however alike the rest is —
    # recurring invoices / reports / quarters must not block each other.
    assert not match("Invoice 1041", "Invoice 1042")
    assert not match("Q3 report", "Q4 report")
    assert not match("Weekly report 2026-10-06", "Weekly report 2026-09-29")
    assert not match("Proposal v2", "Proposal v3")
    assert match("Invoice 1041 for October", "Re: invoice 1041 for october")
    assert match("Project kickoff tomorow", "Project kickoff tomorrow")  # typo, same numbers
    assert not match("", "") and not match("Re:", "Re: ")  # nothing to compare


def test_repeat_refusal_wording():
    draft = {"kind": "draft", "draft_id": "r-1", "message_id": "m1",
             "to": "bob@x.co", "subject": "Report"}
    sent = {"kind": "sent", "message_id": "s1", "date": "2026-10-01 09:30",
            "to": "bob@x.co", "subject": "Report"}
    text = gmail_compose.repeat_refusal([draft, sent], lead="Not created")
    assert text.startswith('Not created — repeat of Draft ID r-1 ("Report" to bob@x.co) and 1 more')
    assert "send_draft draft_id=r-1" in text and "update_draft draft_id=r-1" in text
    assert "allow_repeat=true" in text and "Don't retry" in text
    assert gmail_compose.is_repeat_refusal(text)
    only_sent = gmail_compose.repeat_refusal([sent], lead="Not sent")
    assert "the email sent on 2026-10-01 09:30, Message ID s1" in only_sent
    assert "send_draft" not in only_sent  # no draft to point at
    assert not gmail_compose.is_repeat_refusal("Message or thread not found.")


@pytest.mark.asyncio
async def test_reply_context_carries_the_originals_date():
    gmail = _Gmail({("GET", "/messages/m1"): (200, {**ORIGINAL, "internalDate": "1700000000000"})})
    async with httpx.AsyncClient(transport=httpx.MockTransport(gmail)) as client:
        ctx = await gmail_compose.fetch_reply_context(client, "T", "m1")
    assert ctx["internal_date"] == "1700000000000"


@pytest.mark.asyncio
async def test_create_draft_repeat_of_a_draft_is_refused_without_a_post(monkeypatch):
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({
        ("GET", "/drafts"): _drafts_listing("r-1"),
        ("GET", "/drafts/r-1"): _draft("r-1", "m1", to="Bob <bob@x.co>", subject="Q3 report",
                                       date="Thu, 01 Oct 2026 09:30:00 +0000"),
        ("POST", "/drafts"): (200, {"id": "new", "message": {"id": "m-new"}}),
    })
    res = await _tool_with(gmail).execute(
        "create_draft", to="bob@x.co", subject="Re: Q3 Report", body="Here it is",
    )

    assert not res.success
    assert res.error.startswith('Not created — repeat of Draft ID r-1 ("Q3 report" to Bob <bob@x.co>)')
    assert "send_draft draft_id=r-1" in res.error and "allow_repeat=true" in res.error
    assert _posts(gmail) == []
    listing = gmail.requests[0]
    assert listing.url.path.endswith("/drafts")
    assert listing.url.params["q"] == "{to:bob@x.co cc:bob@x.co}"  # no age limit on drafts


@pytest.mark.asyncio
async def test_create_draft_repeat_of_a_sent_email_is_refused(monkeypatch, _config):
    _config.tools.google_mail.repeat_check_days = 3
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({
        ("GET", "/drafts"): (200, {}),
        ("GET", "/messages"): (200, {"messages": [{"id": "s1"}]}),
        ("GET", "/messages/s1"): _sent("s1", to="carol@ok.com", cc="bob@x.co",
                                       subject="Invoice 42", date="Mon, 05 Oct 2026 10:00:00 +0000"),
    })
    res = await _tool_with(gmail).execute(
        "create_draft", to="Bob <bob@x.co>", subject="invoice  42", body="x",
    )

    assert not res.success
    assert "repeat of the email sent on 2026-10-05 10:00, Message ID s1" in res.error
    assert _posts(gmail) == []
    sent_search = next(r for r in gmail.requests if r.url.path.endswith("/messages"))
    assert sent_search.url.params["q"] == "in:sent {to:bob@x.co cc:bob@x.co} newer_than:3d"
    meta = next(r for r in gmail.requests if r.url.path.endswith("/messages/s1"))
    assert meta.url.params.get_list("metadataHeaders") == ["To", "Cc", "Bcc", "Subject", "Date"]


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [True, "true"])
async def test_allow_repeat_skips_the_check(monkeypatch, flag):
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({
        ("GET", "/drafts"): _drafts_listing("r-1"),
        ("GET", "/drafts/r-1"): _draft("r-1", "m1", to="bob@x.co", subject="Q3 report"),
        ("POST", "/drafts"): (200, {"id": "r-2", "message": {"id": "m2"}}),
    })
    res = await _tool_with(gmail).execute(
        "create_draft", to="bob@x.co", subject="Q3 report", body="Another copy", allow_repeat=flag,
    )
    assert res.success, res.error
    assert "Draft ID: r-2" in res.content
    assert _calls(gmail) == [("POST", "/drafts")]


@pytest.mark.asyncio
async def test_check_off_when_repeat_check_days_is_zero(monkeypatch, _config):
    _config.tools.google_mail.repeat_check_days = 0
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({("POST", "/drafts"): (200, {"id": "r-2", "message": {"id": "m2"}})})
    res = await _tool_with(gmail).execute("create_draft", to="bob@x.co", subject="s", body="b")
    assert res.success and [r.method for r in gmail.requests] == ["POST"]


@pytest.mark.asyncio
async def test_same_subject_to_a_different_recipient_is_not_a_repeat(monkeypatch):
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({  # the fake ignores the query, so the other recipient's draft comes back
        ("GET", "/drafts"): _drafts_listing("r-1"),
        ("GET", "/drafts/r-1"): _draft("r-1", "m1", to="carol@x.co", subject="Team offsite"),
        ("GET", "/messages"): (200, {"messages": [{"id": "s1"}]}),
        ("GET", "/messages/s1"): _sent("s1", to="dave@x.co", subject="Team offsite"),
        ("POST", "/drafts"): (200, {"id": "r-2", "message": {"id": "m2"}}),
    })
    res = await _tool_with(gmail).execute("create_draft", to="bob@x.co", subject="Team offsite", body="b")
    assert res.success, res.error
    assert len(_posts(gmail)) == 1


@pytest.mark.asyncio
async def test_a_different_subject_to_the_same_recipient_is_not_a_repeat(monkeypatch):
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({
        ("GET", "/drafts"): _drafts_listing("r-1"),
        ("GET", "/drafts/r-1"): _draft("r-1", "m1", to="bob@x.co", subject="Q3 report"),
        ("POST", "/drafts"): (200, {"id": "r-2", "message": {"id": "m2"}}),
    })
    res = await _tool_with(gmail).execute("create_draft", to="bob@x.co", subject="Lunch Friday?", body="b")
    assert res.success, res.error


@pytest.mark.asyncio
async def test_compose_only_grant_403_on_sent_still_checks_drafts(monkeypatch):
    _set_mode(monkeypatch, fd=False, scope=COMPOSE)
    forbidden = (403, {"error": {"message": "Request had insufficient authentication scopes."}})
    gmail = _Gmail({
        ("GET", "/drafts"): _drafts_listing("r-1"),
        ("GET", "/drafts/r-1"): _draft("r-1", "m1", to="bob@x.co", subject="Odg: Ponuda"),
        ("GET", "/messages"): forbidden,
        ("POST", "/drafts"): (200, {"id": "r-2", "message": {"id": "m2"}}),
    })
    tool = _tool_with(gmail)
    res = await tool.execute("create_draft", to="bob@x.co", subject="Ponuda", body="b")
    assert not res.success and "repeat of Draft ID r-1" in res.error  # 'Odg:' folded away
    assert _posts(gmail) == []

    # Nothing in Drafts and Sent unreadable: fail open — the draft is made.
    gmail.routes[("GET", "/drafts")] = (200, {})
    res = await tool.execute("create_draft", to="bob@x.co", subject="Ponuda", body="b")
    assert res.success, res.error
    assert len(_posts(gmail)) == 1


@pytest.mark.asyncio
async def test_recipients_passed_as_a_list_are_still_checked(monkeypatch):
    # The message builder takes a list, so the check must too — not skip it.
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({
        ("GET", "/drafts"): _drafts_listing("r-1"),
        ("GET", "/drafts/r-1"): _draft("r-1", "m1", to="bob@x.co, carol@x.co", subject="Q3 report"),
        ("POST", "/drafts"): (200, {"id": "r-2", "message": {"id": "m2"}}),
    })
    res = await _tool_with(gmail).execute(
        "create_draft", to=["carol@x.co"], subject="Q3 report", body="b",
    )
    assert not res.success and "repeat of Draft ID r-1" in res.error
    assert _posts(gmail) == []
    assert gmail.requests[0].url.params["q"] == "{to:carol@x.co cc:carol@x.co}"


@pytest.mark.asyncio
async def test_check_errors_fail_open(monkeypatch):
    _set_mode(monkeypatch, fd=False)
    boom = (500, {"error": {"message": "Backend Error"}})
    gmail = _Gmail({("GET", "/drafts"): boom, ("GET", "/messages"): boom,
                    ("POST", "/drafts"): (200, {"id": "r-2", "message": {"id": "m2"}})})
    res = await _tool_with(gmail).execute("create_draft", to="bob@x.co", subject="s", body="b")
    assert res.success, res.error


def _thread(*messages: dict) -> tuple[int, dict]:
    return 200, {"id": "t1", "messages": list(messages)}


_ORIG_IN_THREAD = {"id": "m1", "threadId": "t1", "labelIds": ["INBOX"], "internalDate": "1000",
                   "payload": {"headers": ORIGINAL["payload"]["headers"]}}


@pytest.mark.asyncio
async def test_reply_with_a_draft_already_in_the_thread_is_refused(monkeypatch):
    _set_mode(monkeypatch, fd=False)
    reply_draft = {"id": "m2", "threadId": "t1", "labelIds": ["DRAFT"], "internalDate": "2000",
                   "payload": {"headers": _hdrs(to="Team <team@ok.com>",
                                                subject="Re: Quarterly numbers")}}
    gmail = _Gmail({
        ("GET", "/messages/m1"): (200, {**ORIGINAL, "internalDate": "1000"}),
        ("GET", "/threads/t1"): _thread(_ORIG_IN_THREAD, reply_draft),
        ("GET", "/drafts"): (200, {"drafts": [{"id": "r-9", "message": {"id": "m2", "threadId": "t1"}}]}),
        ("POST", "/drafts"): (200, {"id": "new", "message": {"id": "m-new"}}),
    })
    res = await _tool_with(gmail).execute("create_draft", reply_to_message_id="m1", body="Sounds good")

    assert not res.success
    assert 'repeat of Draft ID r-9 ("Re: Quarterly numbers" to Team <team@ok.com>)' in res.error
    assert "send_draft draft_id=r-9" in res.error
    assert _posts(gmail) == []
    # The reply lookup first (its defaults decide what is checked), then the thread.
    assert _calls(gmail)[:2] == [("GET", "/messages/m1"), ("GET", "/threads/t1")]


@pytest.mark.asyncio
async def test_a_thread_draft_to_someone_else_does_not_block_a_reply(monkeypatch):
    _set_mode(monkeypatch, fd=False)
    other = {"id": "m2", "threadId": "t1", "labelIds": ["DRAFT"], "internalDate": "2000",
             "payload": {"headers": _hdrs(to="boss@corp.io", subject="Fwd: Quarterly numbers")}}
    gmail = _Gmail({
        ("GET", "/messages/m1"): (200, {**ORIGINAL, "internalDate": "1000"}),
        ("GET", "/threads/t1"): _thread(_ORIG_IN_THREAD, other),
        ("POST", "/drafts"): (200, {"id": "new", "message": {"id": "m-new"}}),
    })
    res = await _tool_with(gmail).execute("create_draft", reply_to_message_id="m1", body="Sounds good")
    assert res.success, res.error  # the reply goes to team@ok.com
    assert len(_posts(gmail)) == 1


@pytest.mark.asyncio
async def test_reply_already_sent_after_the_original_is_refused_but_an_older_one_is_not(monkeypatch):
    _set_mode(monkeypatch, fd=False)

    def own_sent(mid: str, at: str) -> dict:
        return {"id": mid, "threadId": "t1", "labelIds": ["SENT"], "internalDate": at,
                "payload": {"headers": _hdrs(to="team@ok.com", subject="Re: Quarterly numbers",
                                             date="Fri, 02 Oct 2026 08:00:00 +0000")}}

    gmail = _Gmail({
        ("GET", "/messages/m1"): (200, {**ORIGINAL, "internalDate": "1000"}),
        ("GET", "/threads/t1"): _thread(_ORIG_IN_THREAD, own_sent("s2", "3000")),
        ("POST", "/drafts"): (200, {"id": "new", "message": {"id": "m-new"}}),
    })
    tool = _tool_with(gmail)
    res = await tool.execute("create_draft", reply_to_message_id="m1", body="Sounds good")
    assert not res.success and "the email sent on 2026-10-02 08:00, Message ID s2" in res.error

    # The user's own message from BEFORE the one being answered isn't a reply to it.
    gmail.routes[("GET", "/threads/t1")] = _thread(own_sent("s0", "500"), _ORIG_IN_THREAD)
    res = await tool.execute("create_draft", reply_to_message_id="m1", body="Sounds good")
    assert res.success, res.error
    assert len(_posts(gmail)) == 1


@pytest.mark.asyncio
async def test_local_send_repeat_is_refused_before_sending(monkeypatch, _config):
    _config.tools.google_mail.allow_send = True
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({
        ("GET", "/drafts"): _drafts_listing("r-1"),
        ("GET", "/drafts/r-1"): _draft("r-1", "m1", to="bob@x.co", subject="Hello"),
        ("POST", "/messages/send"): (200, {"id": "s1", "threadId": "t1"}),
    })
    tool = _tool_with(gmail)
    res = await tool.execute("send", to="bob@x.co", subject="Hello", body="Hi")
    assert not res.success
    assert res.error.startswith("Not sent — repeat of Draft ID r-1")
    assert "send_draft draft_id=r-1" in res.error
    assert _posts(gmail) == []

    res = await tool.execute("send", to="bob@x.co", subject="Hello", body="Hi", allow_repeat=True)
    assert res.success, res.error
    assert len(_posts(gmail)) == 1


@pytest.mark.asyncio
async def test_fd_mode_send_forwards_allow_repeat_and_relays_the_repeat_409(monkeypatch):
    seen = _fd_answers(monkeypatch, 200, _SENT)
    res = await GoogleMailTool().execute(
        "send", to="bob@x.co", subject="Hello", body="Hi", allow_repeat=True,
    )
    assert res.success, res.error
    assert json.loads(seen[0].content)["allow_repeat"] is True
    # Not passed → not sent (FD's default is to check).
    seen = _fd_answers(monkeypatch, 200, _SENT)
    await GoogleMailTool().execute("send", to="bob@x.co", subject="Hello", body="Hi")
    assert "allow_repeat" not in json.loads(seen[0].content)

    detail = gmail_compose.repeat_refusal(
        [{"kind": "sent", "message_id": "s1", "date": "2026-10-05 10:00",
          "to": "bob@x.co", "subject": "Hello"}], lead="Duplicate: not sent",
    )
    _fd_answers(monkeypatch, 409, {"detail": detail})
    res = await GoogleMailTool().execute("send", to="bob@x.co", subject="Hello", body="Hi")
    assert not res.success and res.error == f"Email not sent: {detail}"
    assert gmail_compose.is_repeat_refusal(res.error)


# ── update_draft ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_draft_replaces_the_draft_in_place(monkeypatch):
    _set_mode(monkeypatch, fd=False, scope=COMPOSE)
    gmail = _Gmail({
        ("GET", "/drafts/r-1"): (200, {"id": "r-1", "message": {"id": "m5", "threadId": "t1", "payload": {
            "headers": _hdrs(to="Team <team@ok.com>", cc="carol@ok.com",
                             subject="Re: Quarterly numbers", in_reply_to="<orig@ok.com>",
                             references="<root@ok.com> <orig@ok.com>"),
        }}}),
        ("PUT", "/drafts/r-1"): (200, {"id": "r-1", "message": {"id": "m6"}}),
    })
    res = await _tool_with(gmail).execute("update_draft", draft_id="r-1", body="Revised: numbers attached")

    assert res.success, res.error
    assert "Draft updated" in res.content and "Draft ID: r-1" in res.content
    assert "Message ID: m6" in res.content
    assert [r.method for r in gmail.requests] == ["GET", "PUT"]  # never a POST /drafts
    sent = gmail.json_of("PUT", "/drafts/r-1")
    assert sent["id"] == "r-1" and sent["message"]["threadId"] == "t1"
    msg = _decode(sent["message"]["raw"])
    assert msg["To"] == "Team <team@ok.com>" and msg["Cc"] == "carol@ok.com"
    assert msg["Subject"] == "Re: Quarterly numbers"
    assert msg["In-Reply-To"] == "<orig@ok.com>" and msg["References"] == "<root@ok.com> <orig@ok.com>"
    assert "Revised: numbers attached" in msg.get_body(("plain",)).get_content()


@pytest.mark.asyncio
async def test_update_draft_overrides_and_standalone_drafts_get_no_thread(monkeypatch):
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({
        ("GET", "/drafts/r-1"): _draft("r-1", "m5", to="bob@x.co", subject="Old subject"),
        ("PUT", "/drafts/r-1"): (200, {"id": "r-1", "message": {"id": "m6"}}),
    })
    res = await _tool_with(gmail).execute(
        "update_draft", draft_id="r-1", subject="New subject", body="New body",
    )
    assert res.success, res.error
    sent = gmail.json_of("PUT", "/drafts/r-1")
    assert "threadId" not in sent["message"]
    msg = _decode(sent["message"]["raw"])
    assert msg["To"] == "bob@x.co" and msg["Subject"] == "New subject"


@pytest.mark.asyncio
async def test_update_draft_errors(monkeypatch):
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({})
    tool = _tool_with(gmail)
    res = await tool.execute("update_draft", body="x")
    assert not res.success and "requires draft_id" in res.error
    res = await tool.execute("update_draft", draft_id="r-1", subject="only a subject")
    assert not res.success and "full new body" in res.error
    assert gmail.requests == []
    res = await tool.execute("update_draft", draft_id="gone", body="x")
    assert not res.success and "Draft gone not found" in res.error and "list_drafts" in res.error
    assert [r.method for r in gmail.requests] == ["GET"]


@pytest.mark.asyncio
async def test_update_draft_needs_the_compose_scope(monkeypatch):
    _set_mode(monkeypatch, fd=False, scope=READ)
    gmail = _Gmail({})
    res = await _tool_with(gmail).execute("update_draft", draft_id="r-1", body="x")
    assert not res.success and "compose scope not granted" in res.error
    assert gmail.requests == []


# ── output: what a repeat check needs to see ─────────────────────────


@pytest.mark.asyncio
async def test_list_drafts_shows_date_and_cc(monkeypatch):
    _set_mode(monkeypatch, fd=False)
    gmail = _Gmail({
        ("GET", "/drafts"): _drafts_listing("r-1"),
        ("GET", "/drafts/r-1"): _draft("r-1", "m1", to="bob@x.co", cc="carol@x.co", subject="Report",
                                       date="Thu, 01 Oct 2026 09:30:00 +0000"),
    })
    res = await _tool_with(gmail).execute("list_drafts", query="to:bob@x.co")
    assert res.success, res.error
    for needle in ("To: bob@x.co", "Cc: carol@x.co", "Date: 2026-10-01 09:30", "Draft ID: r-1",
                   "action=update_draft", "already drafted"):
        assert needle in res.content, needle


@pytest.mark.asyncio
async def test_search_summaries_show_recipients_of_own_mail(monkeypatch):
    _set_mode(monkeypatch, fd=False)

    def msg(mid: str, labels: list[str]) -> dict:
        return {"id": mid, "threadId": f"t-{mid}", "labelIds": labels, "internalDate": "1",
                "payload": {"headers": _hdrs(**{"from": "Me <me@x.co>"}, to="Bob <bob@x.co>",
                                             subject="Report", date="Thu, 01 Oct 2026 09:30:00 +0000")}}

    gmail = _Gmail({
        ("GET", "/threads"): (200, {"threads": [{"id": "t-s1"}]}),
        ("GET", "/threads/t-s1"): (200, {"id": "t-s1", "messages": [msg("s1", ["SENT"])]}),
        ("GET", "/messages"): (200, {"messages": [{"id": "d1"}, {"id": "i1"}]}),
        ("GET", "/messages/d1"): (200, msg("d1", ["DRAFT"])),
        ("GET", "/messages/i1"): (200, msg("i1", ["INBOX"])),
    })
    tool = _tool_with(gmail)
    res = await tool.execute("search", query="in:sent to:bob@x.co newer_than:14d")
    assert res.success, res.error
    assert "To: Bob <bob@x.co>  (sent)" in res.content
    assert gmail.requests[0].url.params["q"] == "in:sent to:bob@x.co newer_than:14d"  # not re-scoped

    res = await tool.execute("search", query="in:anywhere to:bob", group_by_thread=False)
    assert "To: Bob <bob@x.co>  (draft)" in res.content
    assert res.content.count("To: ") == 1  # not for someone else's inbox mail


# ── no repeats: the instructions every model path sees ───────────────


_INSTRUCTIONS = Path(gmail_compose.__file__).resolve().parent / "instructions"


@pytest.mark.parametrize("name", ["section_google.md", "micro_section_google.md"])
def test_section_files_carry_the_no_repeat_rule(name):
    text = (_INSTRUCTIONS / name).read_text(encoding="utf-8")
    for needle in ("in:sent", "list_drafts", "send_draft", "update_draft", "get_thread",
                   "Different recipients" if name.startswith("micro") else "different recipients"):
        assert needle in text, (name, needle)


def test_roster_one_liners_carry_the_no_repeat_rule():
    import captain_claw.agent_context_mixin as acm

    for descs in (acm._TOOL_PROMPT_DESCRIPTIONS, acm._TOOL_PROMPT_DESCRIPTIONS_MICRO):
        line = descs["google_mail"]
        assert "in:sent" in line and "list_drafts" in line and "repeat" in line, line


def test_tool_description_is_the_members_copy_of_the_rule():
    # Shared-agent members never get section_google.md or the roster: this is it.
    d = GoogleMailTool.description
    for needle in ("NO REPEATS", "in:sent", "list_drafts", "get_thread", "update_draft",
                   "allow_repeat", "Different recipients are not repeats", "repeat of"):
        assert needle in d, needle
    assert "retry now — do not reference past failures" not in d
    assert "never redo one that succeeded" in d
    params = GoogleMailTool.parameters["properties"]
    assert params["allow_repeat"]["type"] == "boolean"
    assert "update_draft" in params["action"]["enum"]


def test_nudge_no_longer_forces_a_draft_per_recipient():
    import captain_claw.agent_orchestration_mixin as aom

    nudge = aom._MAIL_TOOL_AVOIDANCE_NUDGE
    assert "for EACH recipient right now" not in nudge
    assert "doesn't already have this email" in nudge
    assert "list_drafts" in nudge and "in:sent" in nudge and "don't create it again" in nudge
    # Same detector as before.
    assert aom._MAIL_AS_TEXT_RE.search("**To:** bob@x.co\n**Subject:** Hi")
    assert not aom._MAIL_AS_TEXT_RE.search("I drafted the email to Bob.")


class _Turn:
    """Just enough agent for AgentToolLoopMixin._turn_has_mail_write."""

    def __init__(self, messages: list[dict]):
        self.session = type("S", (), {"messages": messages})()


def _tool_msg(action: str, content: str, tool: str = "google_mail") -> dict:
    return {"role": "tool", "tool_name": tool, "tool_arguments": {"action": action},
            "content": content}


@pytest.mark.parametrize("messages, expected", [
    ([], False),
    ([_tool_msg("create_draft", "Draft created.\n  Draft ID: r-1")], True),
    ([_tool_msg("send", "Email sent.")], True),
    ([_tool_msg("update_draft", "Draft updated (same draft — no new one was created).")], True),
    ([_tool_msg("create_draft", "Error: Not created — repeat of Draft ID r-1 (…)")], True),
    ([_tool_msg("create_draft", "Error: Google authentication expired.")], False),
    ([_tool_msg("list_drafts", "Drafts (1 shown):")], False),
    ([_tool_msg("create_draft", "Draft created.", tool="send_mail")], False),
    ([{"role": "user", "content": "draft it"}], False),
])
def test_turn_has_mail_write(messages, expected):
    from captain_claw.agent_tool_loop_mixin import AgentToolLoopMixin

    prior = [_tool_msg("create_draft", "Draft created.")]  # an earlier turn's
    turn = _Turn(prior + messages)
    assert AgentToolLoopMixin._turn_has_mail_write(turn, len(prior)) is expected
