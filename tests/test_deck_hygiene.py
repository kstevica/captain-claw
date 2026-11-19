"""Deck hygiene from the cohort001 read (2026-11-19).

- A Google sign-in refused for good (invalid_grant) was refreshed every
  minute for a month; a Drive-only sign-in had its Gmail polled every minute
  (403). Nobody was told.
- Each step of an agent's turn asked Flight Deck for ~10 connection flags; a
  not-connected Microsoft answered 404 every two seconds.
- Flight Deck's log lived only in tmux scrollback (about eleven minutes,
  filled by polling lines); agents' process.log files never shrank.
- "New version for this deck" went out for every new upstream commit.
- Every agent start on a headless server warned about pynput.
"""

from __future__ import annotations

import logging
import sys
import time
import types

import httpx
import pytest

from captain_claw.flight_deck import agent_mail, deck_update, fd_logfile
from captain_claw.flight_deck import google_oauth_routes as gor
from captain_claw.google_oauth import GoogleOAuthTokens

# ── Google: a refused sign-in is left alone, and the owner is told ─────


def _expired(refresh: str = "rt-1") -> GoogleOAuthTokens:
    return GoogleOAuthTokens(access_token="old", refresh_token=refresh, expires_at=time.time() - 10,
                             scope="openid https://www.googleapis.com/auth/drive.file")


def _http_error(status: int, body: dict) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://oauth2.googleapis.com/token")
    return httpx.HTTPStatusError("refused", request=req, response=httpx.Response(status, json=body, request=req))


@pytest.fixture
def google(monkeypatch):
    gor._DEAD_REFRESH.clear()
    gor._REFRESH_PAUSED.clear()
    state = {"refreshes": 0, "error": None, "told": []}

    async def refresh(**kw):
        state["refreshes"] += 1
        if state["error"] is not None:
            raise state["error"]
        return GoogleOAuthTokens(access_token="new", refresh_token=kw["refresh_token"],
                                 expires_at=time.time() + 3600)

    async def store(db, uid, tokens):
        return None

    async def post_once(user_id, kind, title, body="", **kw):
        state["told"].append((user_id, title, kw.get("ref_id")))
        return "item"

    monkeypatch.setattr(gor, "refresh_access_token", refresh)
    monkeypatch.setattr(gor, "_store_tokens", store)
    from captain_claw.flight_deck import line

    monkeypatch.setattr(line, "post_once", post_once)
    yield state
    gor._DEAD_REFRESH.clear()
    gor._REFRESH_PAUSED.clear()


CLIENT = {"client_id": "c", "client_secret": "s"}


async def test_a_refused_sign_in_is_tried_once_and_the_owner_told_once(google):
    google["error"] = _http_error(400, {"error": "invalid_grant", "error_description": "Token has been expired"})
    for _ in range(3):
        assert await gor._refresh_if_needed(None, "u1", CLIENT, _expired()) is None
    assert google["refreshes"] == 1
    assert len(google["told"]) == 1 and google["told"][0][1] == "Reconnect Google"
    assert gor.google_needs_reconnect("u1")


async def test_reconnecting_brings_a_new_token_that_is_tried(google):
    google["error"] = _http_error(400, {"error": "invalid_grant"})
    await gor._refresh_if_needed(None, "u1", CLIENT, _expired("rt-old"))
    google["error"] = None
    fresh = await gor._refresh_if_needed(None, "u1", CLIENT, _expired("rt-new"))
    assert fresh is not None and fresh.access_token == "new"
    assert google["refreshes"] == 2 and not gor.google_needs_reconnect("u1")


async def test_another_failure_waits_before_the_next_try(google):
    google["error"] = _http_error(503, {"error": "backend_error"})
    await gor._refresh_if_needed(None, "u1", CLIENT, _expired())
    await gor._refresh_if_needed(None, "u1", CLIENT, _expired())
    assert google["refreshes"] == 1 and google["told"] == []
    gor._REFRESH_PAUSED["u1"] = time.time() - 1
    await gor._refresh_if_needed(None, "u1", CLIENT, _expired())
    assert google["refreshes"] == 2


# ── agent mail: no Gmail scope, no Gmail poll; failures back off ───────


async def test_a_drive_only_sign_in_has_no_inbox_to_read(monkeypatch):
    async def scopes(owner):
        return {"openid", "https://www.googleapis.com/auth/drive.file"}

    async def token(owner):
        raise AssertionError("no token fetch for a sign-in without Gmail")

    monkeypatch.setattr(gor, "granted_google_scopes", scopes)
    monkeypatch.setattr(gor, "get_valid_google_access_token", token)
    assert await agent_mail._gmail_read("u1", {}) is None


def test_a_failing_source_waits_longer_each_time():
    agent_mail._FAILS.clear()
    try:
        agent_mail._failed("u1", "gmail", "Gmail answered 403")
        first = agent_mail._FAILS[("u1", "gmail")][1] - time.time()
        agent_mail._failed("u1", "gmail", "Gmail answered 403")
        agent_mail._failed("u1", "gmail", "Gmail answered 403")
        third = agent_mail._FAILS[("u1", "gmail")][1] - time.time()
        assert agent_mail._backing_off("u1", "gmail")
        assert third > 3 * first
        for _ in range(20):
            agent_mail._failed("u1", "gmail", "x")
        assert agent_mail._FAILS[("u1", "gmail")][1] - time.time() <= agent_mail._BACKOFF_MAX_S + 1
        assert not agent_mail._backing_off("u1", "outlook")
    finally:
        agent_mail._FAILS.clear()


# ── connection flags: a turn's start, then every 30 s ─────────────────


def test_connection_flags_are_refreshed_at_a_turns_start_then_rarely():
    from captain_claw.agent_orchestration_mixin import AgentOrchestrationMixin

    agent = types.SimpleNamespace()
    due = AgentOrchestrationMixin._connection_flags_due.__get__(agent)
    agent._CONNECTION_FLAGS_REFRESH_S = AgentOrchestrationMixin._CONNECTION_FLAGS_REFRESH_S
    assert due(0)
    assert not due(1) and not due(2)
    agent._connection_flags_at -= 31
    assert due(3)
    assert due(0)          # a new turn always asks


# ── the log: a file, without the polling ──────────────────────────────


def _access(path: str, status: int) -> logging.LogRecord:
    return logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                             '%s - "%s %s HTTP/%s" %d', ("1.2.3.4:0", "GET", path, "1.1", status), None)


@pytest.mark.parametrize("path", [
    "/fd/asks?wait=25&known=", "/fd/system/update/notice", "/fd/crew/tasks/crw_1/events?after=5",
    "/fd/line/unread", "/fd/crew/agent/status", "/fd/imap/agent_status", "/scheduler/jobs",
])
def test_a_successful_poll_is_not_logged(path, monkeypatch):
    monkeypatch.delenv("FD_ACCESS_LOG", raising=False)
    f = fd_logfile.QuietPollingFilter()
    assert not f.filter(_access(path, 200))
    assert f.filter(_access(path, 500))               # a failing poll still shows


def test_other_requests_and_all_mode_are_logged(monkeypatch):
    f = fd_logfile.QuietPollingFilter()
    assert f.filter(_access("/fd/chat/sessions/x", 200))
    monkeypatch.setenv("FD_ACCESS_LOG", "all")
    assert f.filter(_access("/fd/asks", 200))


def test_the_log_goes_to_a_rotating_file_with_structlog_lines(tmp_path, monkeypatch):
    from captain_claw import logging as cc_logging

    monkeypatch.delenv("FD_LOG_FILE", raising=False)
    root = logging.getLogger()
    before = list(root.handlers)
    old_level = root.level
    root.setLevel(logging.INFO)             # as _configure_fd_logging does
    old_sink = cc_logging._system_log_sink
    import structlog

    old_structlog = structlog.get_config()
    try:
        path = fd_logfile.install(tmp_path)
        assert path == tmp_path / "logs" / fd_logfile.LOG_NAME
        assert fd_logfile.install(tmp_path) == path       # once per process
        logging.getLogger("flight_deck").info("deck says %s", "hello")
        cc_logging._system_log_sink("\x1b[2m2026-11-19\x1b[0m [info] Spend ledger read calls=7")
        # A module's structlog logger, as Flight Deck uses it (structlog never
        # configured there): on a live deck these reached only the terminal.
        cc_logging.get_logger("captain_claw.flight_deck.google_oauth_routes").warning(
            "Google sign-in refused for good", owner="u1", error="invalid_grant")
        for h in root.handlers:
            h.flush()
        text = path.read_text()
        assert "deck says hello" in text
        assert "2026-11-19 [info] Spend ledger read calls=7" in text and "\x1b[" not in text
        assert "Google sign-in refused for good" in text and "error=invalid_grant" in text
    finally:
        structlog.configure(**old_structlog)
        for h in list(root.handlers):
            if h not in before:
                root.removeHandler(h)
                h.close()
        cc_logging.set_system_log_sink(old_sink)
        root.setLevel(old_level)


def test_the_log_file_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv("FD_LOG_FILE", "off")
    assert fd_logfile.install(tmp_path) is None


def test_a_big_process_log_moves_aside_at_the_next_start(tmp_path, monkeypatch):
    from captain_claw.flight_deck import server

    monkeypatch.setattr(server, "_PROCESS_LOG_MAX_BYTES", 100)
    (tmp_path / "process.log").write_text("x" * 200)
    log = server._rotated_process_log(tmp_path)
    assert log == tmp_path / "process.log" and not log.exists()
    assert (tmp_path / "process.log.1").stat().st_size == 200
    log.write_text("small")
    assert server._rotated_process_log(tmp_path).read_text() == "small"


# ── one notice per version ────────────────────────────────────────────


def test_a_new_version_is_told_once_per_deck_version_then_weekly():
    now = time.time()
    assert deck_update.should_tell({}, "aaa", now)
    told = {"told_for": "aaa", "told_at": now}
    assert not deck_update.should_tell(told, "aaa", now + 3600)      # another commit upstream
    assert deck_update.should_tell(told, "bbb", now + 3600)          # the deck was updated
    assert deck_update.should_tell(told, "aaa", now + 8 * 86400)     # a week on, still behind


# ── no hotkey warning without a display ───────────────────────────────


def test_a_linux_box_without_a_display_is_headless(monkeypatch):
    from captain_claw.web import hotkey_daemon

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert hotkey_daemon._headless()[0]
    monkeypatch.setenv("DISPLAY", ":0")
    assert not hotkey_daemon._headless()[0]
