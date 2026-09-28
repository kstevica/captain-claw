"""gws / Google tools — one Google account per tenant.

Under Flight Deck the ``gws`` CLI must act as the agent OWNER's Google
account (the token FD hands this agent), never an ambient credential the
subprocess happens to see — and must fail closed when FD has no token.
Standalone it keeps its own ``gws auth login``. The google_* tools' "not
connected" hints must point FD users at FD's Connections page, not the
agent-local OAuth flow whose tokens FD mode discards.

No network, no gws binary: the subprocess and the token source are faked.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest

import captain_claw.session as session_mod
from captain_claw.google_oauth import GoogleOAuthTokens
from captain_claw.google_oauth_manager import GoogleOAuthManager
from captain_claw.tools import _gws_runtime
from captain_claw.tools.gws import GwsTool

# ── fakes ────────────────────────────────────────────────────────────


class _FakeProc:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:  # pragma: no cover - timeout path not exercised
        pass


class _Spawner:
    """Stands in for asyncio.create_subprocess_exec; records every spawn."""

    def __init__(self, outputs: list[tuple[bytes, bytes, int]] | None = None):
        self.calls: list[dict] = []
        self._outputs = list(outputs or [])

    async def __call__(self, *cmd, **kwargs):
        self.calls.append({"cmd": list(cmd), **kwargs})
        out, err, rc = self._outputs.pop(0) if self._outputs else (b"{}", b"", 0)
        return _FakeProc(out, err, rc)


@pytest.fixture
def spawner(monkeypatch):
    sp = _Spawner()
    monkeypatch.setattr(_gws_runtime.asyncio, "create_subprocess_exec", sp)
    return sp


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


def _tool(monkeypatch) -> GwsTool:
    monkeypatch.setattr(GwsTool, "_resolve_binary", lambda self: "/fake/bin/gws")
    return GwsTool()


# ── gws identity: Flight Deck mode ───────────────────────────────────


@pytest.mark.asyncio
async def test_fd_mode_injects_owner_token_and_scrubs_ambient_credentials(monkeypatch, spawner):
    _set_mode(monkeypatch, fd=True, token="OWNER-B-ACCESS")
    # Ambient credentials an operator might have exported deck-wide.
    monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_TOKEN", "OPERATOR-ACCESS")
    monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE", "/host/creds.json")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/host/adc.json")
    monkeypatch.setenv("SOME_UNRELATED_VAR", "kept")

    tool = _tool(monkeypatch)
    res = await tool.execute("drive_info", file_id="abc")

    assert res.success, res.error
    assert len(spawner.calls) == 1
    env = spawner.calls[0]["env"]
    assert env is not None, "FD mode must pass an explicit env, not inherit"
    assert env["GOOGLE_WORKSPACE_CLI_TOKEN"] == "OWNER-B-ACCESS"
    assert "GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE" not in env
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env
    assert env["SOME_UNRELATED_VAR"] == "kept"
    assert env.get("PATH") == os.environ.get("PATH")


@pytest.mark.asyncio
async def test_fd_mode_without_owner_token_fails_closed(monkeypatch, spawner):
    _set_mode(monkeypatch, fd=True, token=None)
    # An ambient credential exists — it must NOT be used as a fallback.
    monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_TOKEN", "OPERATOR-ACCESS")

    tool = _tool(monkeypatch)
    res = await tool.execute("drive_list")

    assert not res.success
    assert "Flight Deck → Connections → Google" in res.error
    assert spawner.calls == [], "gws must not run without the owner's token"


@pytest.mark.asyncio
async def test_fd_mode_empty_access_token_fails_closed(monkeypatch, spawner):
    _set_mode(monkeypatch, fd=True, token="")
    tool = _tool(monkeypatch)
    res = await tool.execute("calendar_list")
    assert not res.success
    assert "Connections → Google" in res.error
    assert spawner.calls == []


def _fd_refuses(monkeypatch, *, auth_disabled: bool) -> None:
    """Flight Deck mode, and FD answers the token request with a 403."""
    from captain_claw.google_oauth_manager import FlightDeckRefused

    monkeypatch.setattr(
        GoogleOAuthManager, "_flight_deck_base", staticmethod(lambda: "http://localhost:25080"),
    )

    async def _refuse(self):
        raise FlightDeckRefused(403, "nope", auth_disabled=auth_disabled)

    monkeypatch.setattr(GoogleOAuthManager, "get_tokens", _refuse)


@pytest.mark.asyncio
async def test_auth_disabled_deck_keeps_gws_own_credentials(monkeypatch, spawner):
    # An auth-disabled deck has one tenant and no Google via FD: gws runs with
    # its own / inherited credentials, as on main.
    _fd_refuses(monkeypatch, auth_disabled=True)
    monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_TOKEN", "DESKTOP-USER-ACCESS")
    res = await _tool(monkeypatch).execute("drive_info", file_id="abc")
    assert res.success, res.error
    assert len(spawner.calls) == 1
    assert spawner.calls[0]["env"] is None  # inherit the process env


@pytest.mark.asyncio
async def test_any_other_fd_refusal_still_fails_closed(monkeypatch, spawner):
    _fd_refuses(monkeypatch, auth_disabled=False)
    monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_TOKEN", "OPERATOR-ACCESS")
    res = await _tool(monkeypatch).execute("drive_info", file_id="abc")
    assert not res.success and "nope" in res.error
    assert spawner.calls == []


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


@pytest.mark.asyncio
async def test_run_gws_outside_execute_still_fails_closed(monkeypatch, spawner):
    # Defence in depth: the runner itself never spawns gws with an ambient
    # identity under FD, even when called without execute() resolving first.
    _set_mode(monkeypatch, fd=True, token=None)
    tool = _tool(monkeypatch)
    res = await tool._run_gws("/fake/bin/gws", ["drive", "files", "list"])
    assert not res.success
    assert "Connections → Google" in res.error
    assert spawner.calls == []


@pytest.mark.asyncio
async def test_fd_mode_one_token_fetch_per_call_and_not_retained(monkeypatch, spawner):
    calls = _set_mode(monkeypatch, fd=True, token="OWNER-ACCESS")
    spawner._outputs = [
        (json.dumps({"files": [{"id": "1", "name": "a"}], "nextPageToken": "p2"}).encode(), b"", 0),
        (json.dumps({"files": [{"id": "2", "name": "b"}]}).encode(), b"", 0),
    ]
    tool = _tool(monkeypatch)
    res = await tool.execute("drive_list")

    assert res.success, res.error
    assert len(spawner.calls) == 2  # two pages
    assert all(c["env"]["GOOGLE_WORKSPACE_CLI_TOKEN"] == "OWNER-ACCESS" for c in spawner.calls)
    assert calls["get_tokens"] == 1, "a paginated call costs one FD round-trip"
    assert tool._gws_env_cache is None, "owner token must not linger on the tool"

    # The next call re-resolves (picks up a reconnect / different grant).
    await tool.execute("drive_info", file_id="x")
    assert calls["get_tokens"] == 2


@pytest.mark.asyncio
async def test_fd_mode_blocks_raw_auth_subcommands(monkeypatch, spawner):
    _set_mode(monkeypatch, fd=True, token="OWNER-ACCESS")
    tool = _tool(monkeypatch)
    for raw in ("auth login", "auth export --unmasked", "--verbose auth status"):
        res = await tool.execute("raw", raw_args=raw)
        assert not res.success, raw
        assert "'gws auth' is not used under Flight Deck" in res.error
    assert spawner.calls == []

    # Non-auth raw commands still run, as the owner.
    res = await tool.execute("raw", raw_args="drive files list")
    assert res.success, res.error
    assert spawner.calls[0]["env"]["GOOGLE_WORKSPACE_CLI_TOKEN"] == "OWNER-ACCESS"


@pytest.mark.asyncio
async def test_fd_mode_auth_error_hint_points_at_connections(monkeypatch, spawner):
    _set_mode(monkeypatch, fd=True, token="OWNER-ACCESS")
    spawner._outputs = [(b"", b"Request had invalid authentication credentials: token expired", 1)]
    tool = _tool(monkeypatch)
    res = await tool.execute("drive_info", file_id="x")
    assert not res.success
    assert "Flight Deck → Connections → Google" in res.error
    assert "gws auth login" not in res.error


@pytest.mark.asyncio
async def test_fd_mode_missing_binary_message(monkeypatch):
    _set_mode(monkeypatch, fd=True, token="OWNER-ACCESS")
    monkeypatch.setattr(GwsTool, "_resolve_binary", lambda self: None)
    res = await GwsTool().execute("drive_list")
    assert not res.success
    assert "not found" in res.error
    assert "Connections → Google" in res.error
    assert "gws auth setup" not in res.error


# ── gws identity: standalone mode (unchanged behaviour) ──────────────


@pytest.mark.asyncio
async def test_standalone_inherits_env_and_gws_login(monkeypatch, spawner):
    calls = _set_mode(monkeypatch, fd=False, token="SHOULD-NOT-BE-USED")
    monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_TOKEN", "USER-OWN-TOKEN")
    tool = _tool(monkeypatch)
    res = await tool.execute("drive_info", file_id="abc")

    assert res.success, res.error
    assert spawner.calls[0]["env"] is None, "standalone keeps inheriting the env"
    assert calls["get_tokens"] == 0, "standalone never asks for an FD token"


@pytest.mark.asyncio
async def test_standalone_allows_raw_auth_and_keeps_login_hint(monkeypatch, spawner):
    _set_mode(monkeypatch, fd=False)
    tool = _tool(monkeypatch)
    res = await tool.execute("raw", raw_args="auth status")
    assert res.success, res.error
    assert spawner.calls[0]["cmd"][1:3] == ["auth", "status"]

    spawner._outputs = [(b"", b"No credentials found", 1)]
    res = await tool.execute("drive_list")
    assert not res.success
    assert "gws auth login" in res.error


@pytest.mark.asyncio
async def test_standalone_missing_binary_message(monkeypatch):
    _set_mode(monkeypatch, fd=False)
    monkeypatch.setattr(GwsTool, "_resolve_binary", lambda self: None)
    res = await GwsTool().execute("drive_list")
    assert not res.success
    assert "gws auth setup && gws auth login" in res.error


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


_DRIVE_FILE = "https://www.googleapis.com/auth/drive.file"
_DRIVE_RO = "https://www.googleapis.com/auth/drive.readonly"
_CAL_RO = "https://www.googleapis.com/auth/calendar.readonly"
_CAL = "https://www.googleapis.com/auth/calendar"


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
async def test_gws_reports_the_refusal_and_does_not_spawn(monkeypatch, spawner):
    _fd_answers(monkeypatch, 403, {"detail": REFUSAL})
    tool = _tool(monkeypatch)
    res = await tool.execute("drive_list")
    assert not res.success
    assert REFUSAL in res.error and "Connect your Google account" not in res.error
    assert spawner.calls == []


@pytest.mark.asyncio
async def test_drive_paths_treat_the_refusal_as_not_connected(monkeypatch):
    # Drive/VFS callers catch DriveError; the refusal must stay one of those.
    from captain_claw.drive_client import DriveNotConnected, global_token_provider

    _fd_answers(monkeypatch, 403, {"detail": REFUSAL})
    with pytest.raises(DriveNotConnected) as ei:
        await global_token_provider()
    assert REFUSAL in str(ei.value)


# ── gws under FD: the deck's scopes, which only an admin can change ──


@pytest.mark.asyncio
async def test_drive_listing_with_per_file_scope_runs_and_says_what_it_leaves_out(
    monkeypatch, spawner
):
    # drive.file doesn't fail a listing — it returns just the files this app
    # created (e.g. the agent's own uploads). Run it; note the gap and who can
    # widen it, beside the output (callers json.loads the content).
    _set_mode(monkeypatch, fd=True, token="OWNER", scope=f"openid email {_DRIVE_FILE}")
    tool = _tool(monkeypatch)
    spawner._outputs = [(b'{"files": [{"id": "mine", "name": "report.md"}]}', b"", 0)]
    res = await tool.execute("drive_list")
    assert res.success, res.error
    assert json.loads(res.content)["files"] == [{"id": "mine", "name": "report.md"}]
    spawner._outputs = [(b'{"files": []}', b"", 0)]
    assert (await tool.execute("drive_search", query="report")).success
    assert len(spawner.calls) == 2

    spawner._outputs = [(b'{"files": []}', b"", 0)]
    res = await tool.execute("raw", raw_args="drive files list --params {}")
    assert res.success and json.loads(res.content) == {"files": []}
    note = res.system_hint
    assert "files this app created" in note and "drive.readonly" in note
    assert "admin must add" in note and "Connections → Google → Scopes" in note


@pytest.mark.asyncio
async def test_drive_listing_with_a_broad_scope_has_no_note(monkeypatch, spawner):
    _set_mode(monkeypatch, fd=True, token="OWNER", scope=f"openid {_DRIVE_FILE} {_DRIVE_RO}")
    res = await _tool(monkeypatch).execute("raw", raw_args="drive files list")
    assert res.success and res.system_hint is None


@pytest.mark.asyncio
async def test_drive_listing_without_any_drive_scope_is_refused(monkeypatch, spawner):
    _set_mode(monkeypatch, fd=True, token="OWNER", scope=f"openid email {_CAL}")
    for action, kw in (("drive_list", {}), ("drive_search", {"query": "x"})):
        res = await _tool(monkeypatch).execute(action, **kw)
        assert not res.success
        assert "drive.readonly" in res.error and "admin must add that scope" in res.error
    assert spawner.calls == []


@pytest.mark.asyncio
async def test_per_file_scope_still_serves_single_file_commands(monkeypatch, spawner):
    _set_mode(monkeypatch, fd=True, token="OWNER", scope=f"openid {_DRIVE_FILE}")
    res = await _tool(monkeypatch).execute("drive_info", file_id="abc")
    assert res.success, res.error
    assert len(spawner.calls) == 1


@pytest.mark.asyncio
async def test_calendar_without_a_calendar_scope_names_the_admin_fix(monkeypatch, spawner):
    _set_mode(monkeypatch, fd=True, token="OWNER", scope=f"openid {_DRIVE_RO}")
    res = await _tool(monkeypatch).execute("calendar_list")
    assert not res.success
    assert "calendar.readonly" in res.error and "admin must add that scope" in res.error
    assert "granting" not in res.error  # users can't grant a scope the deck lacks
    assert spawner.calls == []


@pytest.mark.asyncio
async def test_calendar_write_needs_write_scope(monkeypatch, spawner):
    _set_mode(monkeypatch, fd=True, token="OWNER", scope=f"openid {_CAL_RO}")
    tool = _tool(monkeypatch)
    assert (await tool.execute("calendar_agenda")).success
    res = await tool.execute("calendar_create", summary="x", start="2026-10-01T10:00")
    assert not res.success and "admin must add that scope" in res.error
    assert len(spawner.calls) == 1  # only the agenda ran

    _set_mode(monkeypatch, fd=True, token="OWNER", scope=f"openid {_CAL}")
    res = await tool.execute("calendar_create", summary="x", start="2026-10-01T10:00")
    assert res.success, res.error


@pytest.mark.asyncio
async def test_google_scope_error_hint_names_the_admin_not_a_reconnect(monkeypatch, spawner):
    _set_mode(monkeypatch, fd=True, token="OWNER", scope="")  # scope unknown → gws decides
    spawner._outputs = [(b"", b"Error 403: Request had insufficient authentication scopes.", 1)]
    res = await _tool(monkeypatch).execute("docs_append", file_id="d", content="hi")
    assert not res.success
    assert "Flight Deck → Connections → Google" in res.error
    assert "admin must add that scope" in res.error
    assert "gws auth login" not in res.error
