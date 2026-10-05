"""google_mail send / send_draft / list_drafts + the shared gmail_compose helpers.

Sending is opt-in. Under Flight Deck the tool never sends itself: it POSTs to
``/fd/google/gmail/send`` (FD's per-user policy decides, sends, audits) and
relays FD's answer. Standalone it sends directly, only with
``tools.google_mail.allow_send`` on. create_draft's reply threading moved into
``gmail_compose.fetch_reply_context`` — pinned here so the refactor changed
nothing.

No network: Gmail and Flight Deck answer through httpx.MockTransport.
"""

from __future__ import annotations

import base64
import email
import email.policy
import json
import time

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
