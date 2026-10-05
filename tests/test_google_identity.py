"""Google tools — one Google account per tenant.

Under Flight Deck the google_* tools act as the agent OWNER's Google account
(the token FD hands this agent). Their "not connected" hints must point FD
users at FD's Connections page, not the agent-local OAuth flow whose tokens
FD mode discards; a Flight Deck refusal must reach the user with FD's reason
instead of a misleading "connect your Google account". The FD client sends
its identity headers only to the configured Flight Deck.

No network: the token source and Flight Deck are faked.
"""

from __future__ import annotations

import time

import pytest

import captain_claw.session as session_mod
from captain_claw.google_oauth import GoogleOAuthTokens
from captain_claw.google_oauth_manager import GoogleOAuthManager

# ── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_real_session_manager(monkeypatch):
    # The tools build GoogleOAuthManager(get_session_manager()); never touch a DB.
    monkeypatch.setattr(session_mod, "get_session_manager", lambda: object())


@pytest.fixture(autouse=True)
def _isolated_fd_home(monkeypatch, tmp_path):
    # The Google client falls back to the per-deck agent_secret file; keep any
    # it reads or mints in this test's tmp dir, never the real FD home.
    from captain_claw.flight_deck import agent_secret

    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    agent_secret.reset_cache_for_tests()
    yield
    agent_secret.reset_cache_for_tests()


def _set_mode(monkeypatch, *, fd: bool, token: str | None = None, scope: str = "") -> dict:
    """Flight Deck client mode on/off + what FD returns for this agent's owner."""
    calls = {"get_tokens": 0}
    monkeypatch.setattr(
        GoogleOAuthManager,
        "_flight_deck_base",
        staticmethod(lambda: "http://localhost:25080" if fd else ""),
    )

    async def _get_tokens(self):
        calls["get_tokens"] += 1
        if token is None:
            return None
        return GoogleOAuthTokens(
            access_token=token, refresh_token="",
            expires_at=time.time() + 3300, scope=scope,
        )

    monkeypatch.setattr(GoogleOAuthManager, "get_tokens", _get_tokens)
    return calls


# ── the auth-disabled marker on a Flight Deck refusal ────────────────


@pytest.mark.parametrize("header,expected", [
    ({"X-FD-Google-Unavailable": "auth-disabled"}, True),
    ({}, False),
    ({"X-FD-Google-Unavailable": "something-else"}, False),
])
def test_refusal_reads_the_auth_disabled_marker(header, expected):
    import httpx

    from captain_claw.google_oauth_manager import _fd_refusal

    resp = httpx.Response(403, json={"detail": "x"}, headers=header)
    assert _fd_refusal(resp).auth_disabled is expected


# ── google_* tools: "not connected" hint ─────────────────────────────


def _google_tools():
    from captain_claw.tools.google_calendar import GoogleCalendarTool
    from captain_claw.tools.google_drive import GoogleDriveTool
    from captain_claw.tools.google_mail import GoogleMailTool

    return [GoogleCalendarTool(), GoogleDriveTool(), GoogleMailTool()]


@pytest.mark.asyncio
async def test_google_tools_not_connected_fd_points_at_connections(monkeypatch):
    _set_mode(monkeypatch, fd=True, token=None)
    for tool in _google_tools():
        with pytest.raises(RuntimeError) as ei:
            await tool._get_access_token()
        msg = str(ei.value)
        assert "Flight Deck → Connections → Google" in msg, type(tool).__name__
        assert "/auth/google/login" not in msg, type(tool).__name__
        assert "Settings > Google OAuth" not in msg, type(tool).__name__


@pytest.mark.asyncio
async def test_google_tools_not_connected_standalone_points_at_local_oauth(monkeypatch):
    _set_mode(monkeypatch, fd=False, token=None)
    for tool in _google_tools():
        with pytest.raises(RuntimeError) as ei:
            await tool._get_access_token()
        msg = str(ei.value)
        assert "/auth/google/login" in msg, type(tool).__name__
        assert "Flight Deck" not in msg, type(tool).__name__


# ── Flight Deck refusals reach the user (not "connect your Google") ──


def _fd_answers(monkeypatch, status: int, body: dict, *, fd_secret: str = "") -> list:
    """Real GoogleOAuthManager in FD client mode; FD answers every request with
    (*status*, *body*). Returns the list of requests it received."""
    import httpx

    import captain_claw.config as config_mod
    import captain_claw.google_oauth_manager as gom
    from captain_claw.config import Config

    monkeypatch.setattr(config_mod, "_config", Config(
        web={"auth_token": "agent-web-auth"},
        google_oauth={"flight_deck_url": "http://fd.test", "flight_deck_secret": fd_secret},
    ))
    monkeypatch.delenv("FD_AGENT_SHARED_SECRET", raising=False)
    seen: list = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json=body)

    real = httpx.AsyncClient
    monkeypatch.setattr(
        gom.httpx, "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    return seen


REFUSAL = "Flight Deck can't attribute this agent to a user — respawn it from Flight Deck"


@pytest.mark.asyncio
async def test_agent_request_identifies_the_agent_and_looks_nothing_like_a_browser(monkeypatch):
    # FD refuses Origin / Sec-Fetch-* as browser traffic; the agent client must
    # never send them, and must send its X-Agent-Auth.
    seen = _fd_answers(monkeypatch, 200, {"access_token": "OWNER", "expires_at": time.time() + 600})
    tokens = await GoogleOAuthManager(object()).get_tokens()
    assert tokens.access_token == "OWNER"
    (req,) = seen
    assert str(req.url) == "http://fd.test/fd/google/access_token"
    assert req.headers["X-Agent-Auth"] == "agent-web-auth"
    names = {k.lower() for k in req.headers.keys()}
    assert "origin" not in names
    assert not any(n.startswith("sec-fetch-") for n in names)


@pytest.mark.asyncio
async def test_agent_secret_falls_back_to_the_per_deck_file(monkeypatch):
    # FD_LOCKDOWN decks that rely on the auto-bootstrapped per-deck secret (no
    # FD_AGENT_SHARED_SECRET in the env) refuse loopback without it: the
    # Google client sends it, as tools.flight_deck._fd_agent_headers does.
    from captain_claw.flight_deck.agent_secret import get_or_create_agent_secret

    seen = _fd_answers(monkeypatch, 200, {"access_token": "OWNER", "expires_at": time.time() + 600})
    await GoogleOAuthManager(object()).get_tokens()
    deck_secret = get_or_create_agent_secret()
    assert deck_secret and seen[0].headers["X-Agent-Secret"] == deck_secret


@pytest.mark.asyncio
async def test_agent_secret_order_is_config_then_env_then_file(monkeypatch):
    ok = {"access_token": "OWNER", "expires_at": time.time() + 600}
    seen = _fd_answers(monkeypatch, 200, ok, fd_secret="from-config")
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "from-env")
    await GoogleOAuthManager(object()).get_tokens()
    assert seen[-1].headers["X-Agent-Secret"] == "from-config"

    import captain_claw.config as config_mod
    from captain_claw.config import Config

    monkeypatch.setattr(config_mod, "_config", Config(
        web={"auth_token": "agent-web-auth"},
        google_oauth={"flight_deck_url": "http://fd.test"},
    ))
    await GoogleOAuthManager(object()).get_tokens()
    assert seen[-1].headers["X-Agent-Secret"] == "from-env"


@pytest.mark.asyncio
async def test_identity_headers_go_only_to_the_configured_flight_deck(monkeypatch):
    # The FD URL is config / the env FD pins at spawn — nothing a session or a
    # websocket message can set — and every request goes there.
    seen = _fd_answers(monkeypatch, 200, {"access_token": "OWNER", "expires_at": time.time() + 600})
    monkeypatch.setenv("FD_URL", "http://elsewhere.test")  # config wins
    mgr = GoogleOAuthManager(object())
    await mgr.get_tokens()
    await mgr.get_vertex_credentials()
    assert {r.url.host for r in seen} == {"fd.test"}
    assert all(r.headers.get("X-Agent-Auth") == "agent-web-auth" for r in seen)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_fd_refusal_is_raised_with_fds_reason(monkeypatch, status):
    from captain_claw.google_oauth_manager import FlightDeckRefused

    _fd_answers(monkeypatch, status, {"detail": REFUSAL})
    mgr = GoogleOAuthManager(object())
    with pytest.raises(FlightDeckRefused) as ei:
        await mgr.get_tokens()
    assert ei.value.status == status and ei.value.detail == REFUSAL
    assert REFUSAL in str(ei.value)
    # Status checks stay a plain bool; the Vertex/LLM path stays None.
    assert await mgr.is_connected() is False
    assert await mgr.get_vertex_credentials() is None


@pytest.mark.asyncio
async def test_not_connected_upstream_is_still_none(monkeypatch):
    _fd_answers(monkeypatch, 404, {"detail": "No stored Google OAuth tokens"})
    assert await GoogleOAuthManager(object()).get_tokens() is None


@pytest.mark.asyncio
async def test_google_tools_report_the_refusal_instead_of_reconnect(monkeypatch):
    _fd_answers(monkeypatch, 403, {"detail": REFUSAL})
    for tool in _google_tools():
        with pytest.raises(RuntimeError) as ei:
            await tool._get_access_token()
        msg = str(ei.value)
        assert "Flight Deck refused" in msg and REFUSAL in msg, type(tool).__name__
        assert "Connect your Google account" not in msg, type(tool).__name__
    # …and through each tool's execute(), which is what the agent sees.
    from captain_claw.tools.google_mail import GoogleMailTool
    res = await GoogleMailTool().execute("list_messages")
    assert not res.success and REFUSAL in res.error


@pytest.mark.asyncio
async def test_google_tools_still_say_connect_when_fd_has_no_tokens(monkeypatch):
    _fd_answers(monkeypatch, 404, {"detail": "No stored Google OAuth tokens"})
    for tool in _google_tools():
        with pytest.raises(RuntimeError) as ei:
            await tool._get_access_token()
        assert "Flight Deck → Connections → Google" in str(ei.value)


@pytest.mark.asyncio
async def test_drive_paths_treat_the_refusal_as_not_connected(monkeypatch):
    # Drive/VFS callers catch DriveError; the refusal must stay one of those.
    from captain_claw.drive_client import DriveNotConnected, global_token_provider

    _fd_answers(monkeypatch, 403, {"detail": REFUSAL})
    with pytest.raises(DriveNotConnected) as ei:
        await global_token_provider()
    assert REFUSAL in str(ei.value)
