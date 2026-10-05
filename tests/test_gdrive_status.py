"""GET /api/gdrive-status — the Read Folders modal's Google Drive tab.

``available`` is true only when this agent's Google identity (the owner's
under Flight Deck) has a token whose scope can list the user's folders. No
token, a Flight Deck refusal, an unreachable deck or a scope without folder
access are all simply "not available" — never an error. Nothing here looks
for a ``gws`` binary any more.
"""

from __future__ import annotations

import json
import time

import pytest

import captain_claw.session as session_mod
from captain_claw.google_oauth import GoogleOAuthTokens
from captain_claw.google_oauth_manager import FlightDeckRefused, GoogleOAuthManager
from captain_claw.web.rest_skills import gdrive_status

DRIVE = "https://www.googleapis.com/auth/drive"
DRIVE_RO = "https://www.googleapis.com/auth/drive.readonly"
DRIVE_FILE = "https://www.googleapis.com/auth/drive.file"


@pytest.fixture
def identity(monkeypatch):
    """Set what GoogleOAuthManager.get_tokens answers: tokens, None, or an exception."""
    monkeypatch.setattr(session_mod, "get_session_manager", lambda: object())
    state: dict = {"answer": None}

    async def _get_tokens(self):
        answer = state["answer"]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(GoogleOAuthManager, "get_tokens", _get_tokens)
    return state


def _tokens(scope: str, access: str = "ACCESS") -> GoogleOAuthTokens:
    return GoogleOAuthTokens(access_token=access, refresh_token="",
                             expires_at=time.time() + 3300, scope=scope)


async def _available() -> bool:
    resp = await gdrive_status(None, None)
    assert resp.status == 200
    body = json.loads(resp.text)
    assert set(body) == {"available"}
    return body["available"]


@pytest.mark.parametrize("scope", [
    f"openid email {DRIVE}",
    f"openid {DRIVE_RO} https://www.googleapis.com/auth/calendar",
    f"{DRIVE_FILE} {DRIVE_RO}",
])
async def test_connected_with_a_folder_listing_drive_scope(identity, scope):
    identity["answer"] = _tokens(scope)
    assert await _available() is True


async def test_unreported_scope_gets_the_benefit_of_the_doubt(identity):
    # Same policy as DriveClient: an empty scope string is "unknown", not "none".
    identity["answer"] = _tokens("")
    assert await _available() is True


async def test_no_tokens(identity):
    identity["answer"] = None
    assert await _available() is False


async def test_token_without_an_access_token(identity):
    identity["answer"] = _tokens(DRIVE, access="")
    assert await _available() is False


@pytest.mark.parametrize("scope", [
    "openid email profile",
    "https://www.googleapis.com/auth/gmail.readonly",
    # Per-file Drive access can't see the user's existing folders.
    f"openid email {DRIVE_FILE}",
])
async def test_no_folder_listing_drive_scope(identity, scope):
    identity["answer"] = _tokens(scope)
    assert await _available() is False


@pytest.mark.parametrize("exc", [
    FlightDeckRefused(403, "agent not spawned by this deck"),
    FlightDeckRefused(401, "", auth_disabled=True),
    RuntimeError("Flight Deck unreachable"),
    OSError("connection refused"),
])
async def test_any_identity_failure_is_not_available(identity, exc):
    identity["answer"] = exc
    assert await _available() is False


def test_route_is_registered_and_the_gws_route_is_gone():
    import inspect

    from captain_claw import web_server

    src = inspect.getsource(web_server)
    assert '"/api/gdrive-status", self._gdrive_status' in src
    assert "/api/gws-status" not in src and "gws_status" not in src


@pytest.mark.parametrize("script", ["app.js", "computer.js"])
def test_web_ui_asks_the_new_route(script):
    from pathlib import Path

    import captain_claw.web as web_pkg

    src = (Path(web_pkg.__file__).parent / "static" / script).read_text(encoding="utf-8")
    assert "/api/gdrive-status" in src
    assert "gws" not in src.lower()
