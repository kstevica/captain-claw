"""file_tree_builder lists Google Drive through the native Drive API.

The gdrive tree (injected into the system prompt) and the agent's folder
picker read Drive as the agent's own Google identity — the OWNER's token
under Flight Deck — resolved once per tree (one FD round-trip, not one per
folder), and fail closed (the error text, no HTTP call) without one. A folder
that fails to list renders an ``[error: ...]`` line; nothing raises.

No network: the token source and the httpx layer under DriveClient are faked,
so the real DriveClient code (headers, scope check, pagination) runs.
"""

from __future__ import annotations

import re
import time

import httpx
import pytest

import captain_claw.session as session_mod
from captain_claw import file_tree_builder as ftb
from captain_claw.drive_client import FOLDER_MIME, DriveClient
from captain_claw.google_oauth import GoogleOAuthTokens
from captain_claw.google_oauth_manager import FlightDeckRefused, GoogleOAuthManager

DRIVE_RO = "https://www.googleapis.com/auth/drive.readonly"

TREE = {
    "root": [
        {"id": "f1", "name": "Sub", "mimeType": FOLDER_MIME},
        {"id": "d1", "name": "doc.txt", "mimeType": "text/plain", "size": "10"},
        {"id": "e1", "name": "empty.txt", "mimeType": "text/plain", "size": "0"},
        {"id": "g1", "name": "Notes", "mimeType": "application/vnd.google-apps.document"},
    ],
    "f1": [{"id": "d2", "name": "inner.txt", "mimeType": "text/plain", "size": "2048"}],
}


def _resp(status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("GET", "https://example.test"))


class _FakeHTTP:
    """Stands in for DriveClient's httpx.AsyncClient; answers from TREE."""

    def __init__(self):
        self.calls: list[dict] = []
        self.is_closed = False
        self.status_for: dict[str, int] = {}   # folder id -> forced error status
        self.drives_status = 200

    async def request(self, method, url, headers=None, params=None, **kwargs):
        params = dict(params or {})
        self.calls.append({"url": url, "headers": dict(headers or {}), "params": params})
        if url.endswith("/drives"):
            if self.drives_status != 200:
                return _resp(self.drives_status, {"error": {"message": "drives boom"}})
            return _resp(200, {"drives": [{"id": "sd1", "name": "Team"}]})
        parent = re.match(r"'(.+?)' in parents", params["q"]).group(1)
        if parent in self.status_for:
            return _resp(self.status_for[parent], {"error": {"message": "nope"}})
        files = TREE.get(parent, [])
        page = int(params.get("pageSize", 1000))
        body: dict = {"files": files[:page]}
        if len(files) > page:
            body["nextPageToken"] = "more"
        return _resp(200, body)


@pytest.fixture
def http(monkeypatch):
    fake = _FakeHTTP()
    monkeypatch.setattr(DriveClient, "_http", lambda self: fake)
    monkeypatch.setattr(session_mod, "get_session_manager", lambda: object())
    return fake


def _set_identity(monkeypatch, *, token: str | None = "OWNER-ACCESS",
                  scope: str = DRIVE_RO, raises: Exception | None = None) -> dict:
    """Fake the Google identity source; counts token resolutions."""
    calls = {"get_tokens": 0}

    async def _get_tokens(self):
        calls["get_tokens"] += 1
        if raises is not None:
            raise raises
        if token is None:
            return None
        return GoogleOAuthTokens(access_token=token, refresh_token="",
                                 expires_at=time.time() + 3300, scope=scope)

    monkeypatch.setattr(GoogleOAuthManager, "get_tokens", _get_tokens)
    return calls


# ── tree ─────────────────────────────────────────────────────────────

async def test_tree_renders_folders_files_sizes_and_ids(monkeypatch, http):
    _set_identity(monkeypatch)
    tree, count = await ftb.build_gdrive_tree("root", "My Drive", max_depth=2)
    assert count == 5
    assert tree == "\n".join([
        "Google Drive: My Drive (5 entries)",
        "  ├── Sub/ [id:f1]",
        "  │   └── inner.txt (2.0 KB) [id:d2]",
        "  ├── doc.txt (10 B) [id:d1]",
        "  ├── empty.txt (0 B) [id:e1]",
        "  └── Notes [id:g1]",
    ])
    first = http.calls[0]["params"]
    assert first["orderBy"] == "folder,name"         # folders first, by name
    assert first["corpora"] == "allDrives"           # a shared-drive folder lists too
    assert first["includeItemsFromAllDrives"] == "true"


async def test_tree_stops_at_max_depth(monkeypatch, http):
    _set_identity(monkeypatch)
    tree, count = await ftb.build_gdrive_tree("root", "My Drive", max_depth=1)
    assert count == 4 and "inner.txt" not in tree
    assert len(http.calls) == 1


async def test_tree_truncates_at_max_entries(monkeypatch, http):
    _set_identity(monkeypatch)
    tree, count = await ftb.build_gdrive_tree("root", "My Drive", max_entries=2)
    assert count == 2
    assert tree.splitlines()[0] == "Google Drive: My Drive (2 entries) [truncated at 2 entries]"
    assert "inner.txt" in tree and "  ... and 1 more" in tree
    assert "Notes" not in tree
    assert http.calls[0]["params"]["pageSize"] == 2  # never fetches past the budget


async def test_tree_resolves_the_owner_token_once(monkeypatch, http):
    calls = _set_identity(monkeypatch)
    await ftb.build_gdrive_tree("root", "My Drive", max_depth=2)
    assert len(http.calls) == 2                      # root + the subfolder
    assert calls["get_tokens"] == 1                  # one FD round-trip per tree
    assert all(c["headers"]["Authorization"] == "Bearer OWNER-ACCESS" for c in http.calls)


async def test_tree_without_an_identity_fails_closed(monkeypatch, http):
    _set_identity(monkeypatch, token=None)
    tree, count = await ftb.build_gdrive_tree("root", "My Drive")
    assert count == 0
    assert tree.startswith("Google Drive: My Drive [error: ")
    assert "not connected" in tree
    assert http.calls == []                          # no listing, no HTTP at all


async def test_tree_without_a_drive_scope_fails_closed(monkeypatch, http):
    _set_identity(monkeypatch, scope="openid email")
    tree, count = await ftb.build_gdrive_tree("root", "My Drive")
    assert count == 0 and "no Drive scope" in tree
    assert http.calls == []


@pytest.mark.parametrize("exc", [
    FlightDeckRefused(403, "agent not spawned by this deck"),
    RuntimeError("Flight Deck unreachable"),
])
async def test_tree_identity_failure_fails_closed_and_never_raises(monkeypatch, http, exc):
    _set_identity(monkeypatch, raises=exc)
    tree, count = await ftb.build_gdrive_tree("root", "My Drive")
    assert count == 0 and str(exc) in tree
    assert http.calls == []


async def test_a_folder_that_fails_renders_an_error_line(monkeypatch, http):
    _set_identity(monkeypatch)
    http.status_for["f1"] = 403
    tree, count = await ftb.build_gdrive_tree("root", "My Drive", max_depth=2)
    assert count == 4                                # the rest of the tree is intact
    assert "  │   [error: Drive API error (403): nope]" in tree.splitlines()


async def test_an_unknown_root_folder_renders_an_error_line(monkeypatch, http):
    _set_identity(monkeypatch)
    http.status_for["gone"] = 404
    tree, count = await ftb.build_gdrive_tree("gone", "Old")
    assert count == 0
    assert tree.splitlines() == [
        "Google Drive: Old (0 entries)",
        "  [error: Not found (check the file or folder id).]",
    ]


async def test_an_expired_token_mid_tree_fails_the_whole_tree_closed(monkeypatch, http):
    _set_identity(monkeypatch)
    http.status_for["f1"] = 401
    tree, count = await ftb.build_gdrive_tree("root", "My Drive", max_depth=2)
    assert count == 0 and "[error: Google authentication expired" in tree
    assert "doc.txt" not in tree


# ── folder picker ────────────────────────────────────────────────────

async def test_picker_root_lists_folders_and_shared_drives(monkeypatch, http):
    calls = _set_identity(monkeypatch)
    res = await ftb.browse_gdrive_folders("root")
    assert res == {
        "folders": [{"id": "f1", "name": "Sub"}],    # files are filtered out
        "shared_drives": [{"id": "sd1", "name": "Team"}],
    }
    assert len(http.calls) == 2 and calls["get_tokens"] == 1
    assert http.calls[0]["params"]["corpora"] == "allDrives"
    assert all(c["headers"]["Authorization"] == "Bearer OWNER-ACCESS" for c in http.calls)


async def test_picker_below_root_skips_shared_drives(monkeypatch, http):
    _set_identity(monkeypatch)
    res = await ftb.browse_gdrive_folders("f1")
    assert res == {"folders": [], "shared_drives": []}
    assert len(http.calls) == 1


async def test_picker_keeps_folders_when_shared_drives_fail(monkeypatch, http):
    _set_identity(monkeypatch)
    http.drives_status = 403
    res = await ftb.browse_gdrive_folders("root")
    assert res == {"folders": [{"id": "f1", "name": "Sub"}], "shared_drives": []}


async def test_picker_without_an_identity_fails_closed(monkeypatch, http):
    _set_identity(monkeypatch, token=None)
    res = await ftb.browse_gdrive_folders("root")
    assert res["folders"] == [] and res["shared_drives"] == []
    assert "not connected" in res["error"]
    assert http.calls == []


async def test_picker_surfaces_a_drive_error(monkeypatch, http):
    _set_identity(monkeypatch)
    http.status_for["gone"] = 404
    res = await ftb.browse_gdrive_folders("gone")
    assert res["folders"] == [] and "Not found" in res["error"]


async def test_picker_identity_failure_never_raises(monkeypatch, http):
    _set_identity(monkeypatch, raises=RuntimeError("Flight Deck unreachable"))
    res = await ftb.browse_gdrive_folders("root")
    assert "Flight Deck unreachable" in res["error"]
    assert http.calls == []


# ── DriveClient.list_folder(all_drives=...) ──────────────────────────

class TestListFolderCorpora:
    def _client(self, http):
        async def provider():
            return "tok", DRIVE_RO
        return DriveClient(provider)

    async def test_default_corpus_is_untouched(self, http):
        await self._client(http).list_folder("root")
        assert "corpora" not in http.calls[0]["params"]

    async def test_all_drives_searches_every_corpus(self, http):
        await self._client(http).list_folder("f1", all_drives=True)
        p = http.calls[0]["params"]
        assert p["corpora"] == "allDrives" and "driveId" not in p
        assert p["supportsAllDrives"] == "true" and p["includeItemsFromAllDrives"] == "true"

    async def test_drive_id_wins_over_all_drives(self, http):
        await self._client(http).list_folder("f1", drive_id="sd1", all_drives=True)
        p = http.calls[0]["params"]
        assert p["corpora"] == "drive" and p["driveId"] == "sd1"
