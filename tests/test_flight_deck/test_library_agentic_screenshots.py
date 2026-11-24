"""The browser tool's screenshots stay out of the File library's own lists.

The browser tool saves a picture each time it looks at a page
(saved/media/<session>/browser_screenshot_<ms>.jpg, browser_chrome_<ms>.jpg in
the person's Chrome, browser_zoom_<ms>.jpg for a close-up). A busy agent takes
hundreds, and they buried the person's real files. They are left out of the
library's list, counts and search, listed on their own with
``section=screenshots`` ("Agentic screenshots"), and filed without a model.
Only that exact name in that exact place counts.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import file_library as fl
from captain_claw.flight_deck import rate_limiter, server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

OWNER = "user-owner"
SESSION = "3f1c9a52-7d40-4e8b-9b1a-2c6d0e5f7a18"
SHOTS = {
    f"saved/media/{SESSION}/browser_screenshot_1795526291270.jpg",
    f"saved/media/{SESSION}/browser_chrome_1795526299001.jpg",
    f"saved/media/{SESSION}/browser_zoom_1795526300412.jpg",
}
FILES = {
    "notes.md",
    # Says "browser", but isn't a picture the browser tool saved.
    f"saved/media/{SESSION}/browser_notes.md",
    "saved/reports/browser-comparison.md",
    # A screenshot the agent saved on purpose, somewhere else: a library file.
    "saved/reports/browser_screenshot_1795526291270.jpg",
}


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Auth-enabled deck: one stopped agent whose workspace holds the
    browser tool's three kinds of screenshot and four files that aren't."""
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    await db._db.execute(
        "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
        " VALUES (?, ?, 'h', ?, 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        (OWNER, "owner@x.co", "Owner"))
    await db._db.commit()

    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(fl, "_STORE", fl.LibraryStore(tmp_path / "file_library.db"))

    async def no_tier(*_a, **_kw):
        return None

    async def no_media(owner, tier):
        return fl.MediaReach(owner=owner, tier=tier)

    monkeypatch.setattr(fl, "_resolve_tier", no_tier)
    monkeypatch.setattr(fl, "media_reach", no_media)

    server._save_process_registry({
        "web-agent": {"slug": "web-agent", "name": "Web agent", "web_port": 24101,
                      "web_auth": "tok", "owner": OWNER, "pid": None},
    })
    ws = data / "web-agent" / "data" / "workspace"
    for rel in SHOTS | FILES:
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff\xe0 not a real jpeg" if rel.endswith(".jpg") else b"# A page\n")
    try:
        yield ws
    finally:
        await db.close()
        fd_auth._db = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=("203.0.113.9", 40001)), base_url="http://fd.test")


def _auth() -> dict:
    return {"Authorization": f"Bearer {create_access_token(OWNER)}"}


async def _scan() -> None:
    await fl.run_scan({"id": OWNER}, fl.ScanJob())


async def _files(**params) -> dict:
    async with _client() as c:
        r = await c.get("/fd/file-library/files", params=params, headers=_auth())
    assert r.status_code == 200, r.text
    return r.json()


def _paths(body: dict) -> set[str]:
    return {i["rel_path"] for i in body["items"]}


# ── What counts ──


@pytest.mark.parametrize("rel", sorted(SHOTS) + [
    "saved/media/default/browser_screenshot_1795526291270.jpg",
    "saved/media/s1/browser_screenshot_1795526291270.png",
])
def test_the_browser_tools_pictures_are_agentic_screenshots(rel: str):
    assert fl.is_agentic_screenshot(rel)


@pytest.mark.parametrize("rel", sorted(FILES) + [
    f"saved/media/{SESSION}/browser_screenshot_final.jpg",
    f"saved/media/{SESSION}/browser_screenshot_1795526291270.pdf",
    f"saved/media/{SESSION}/my_browser_screenshot_1795526291270.jpg",
    f"saved/media/{SESSION}/deeper/browser_screenshot_1795526291270.jpg",
    "browser_screenshot_1795526291270.jpg",
    "",
])
def test_other_files_are_not(rel: str):
    assert not fl.is_agentic_screenshot(rel)


def test_the_paths_the_browser_tools_build_match(tmp_path: Path):
    """Each place the browser tool names a picture, asked for a name: the
    library knows every one as an agentic screenshot."""
    from captain_claw.tools import browser_chrome
    from captain_claw.tools.browser import BrowserTool
    from captain_claw.tools.browser_page import HeadlessPage

    kwargs = {"_runtime_base_path": str(tmp_path), "_session_id": SESSION}
    built = [
        BrowserTool._screenshot_path(kwargs),
        browser_chrome._screenshot_path(kwargs),
        browser_chrome._screenshot_path(kwargs, "zoom"),
        HeadlessPage(None, kwargs)._picture_path("zoom"),
    ]
    for path in built:
        assert fl.is_agentic_screenshot(path.relative_to(tmp_path).as_posix()), path


# ── The library ──


async def test_the_library_leaves_them_out(deck: Path):
    await _scan()
    body = await _files()
    assert _paths(body) == FILES
    assert body["total"] == len(FILES)
    assert body["sections"] == {"files": len(FILES), "screenshots": len(SHOTS)}
    # Counts and filters don't see them either.
    assert body["facets"]["agents"] == [{"id": "web-agent", "label": "Web agent", "count": len(FILES)}]
    assert all(f["id"] != "image" or f["count"] == 1 for f in body["facets"]["kinds"])
    async with _client() as c:
        status = (await c.get("/fd/file-library/status", headers=_auth())).json()
    assert status["counts"]["total"] == len(FILES)


async def test_search_leaves_them_out(deck: Path):
    await _scan()
    body = await _files(q="browser")
    assert _paths(body) == FILES - {"notes.md"}
    assert body["sections"] == {"files": 3, "screenshots": len(SHOTS)}
    assert _paths(await _files(q="screenshot")) == {"saved/reports/browser_screenshot_1795526291270.jpg"}


async def test_agentic_screenshots_are_listed_on_their_own(deck: Path):
    await _scan()
    body = await _files(section="screenshots")
    assert _paths(body) == SHOTS
    assert body["total"] == len(SHOTS)
    assert body["sections"] == {"files": len(FILES), "screenshots": len(SHOTS)}
    assert body["facets"]["agents"] == [{"id": "web-agent", "label": "Web agent", "count": len(SHOTS)}]
    by_name = {i["filename"]: i for i in body["items"]}
    assert {i["status"] for i in body["items"]} == {"skipped"}
    assert {i["summary"] for i in body["items"]} == {fl._SHOT_SUMMARY}
    assert by_name["browser_screenshot_1795526291270.jpg"]["title"] == "Browser screenshot"
    assert by_name["browser_chrome_1795526299001.jpg"]["title"] == "Chrome screenshot"
    assert by_name["browser_zoom_1795526300412.jpg"]["title"] == "Browser close-up"
    # A category or kind left over from the files side doesn't empty it.
    assert _paths(await _files(section="screenshots", category="report", kind="document")) == SHOTS
    # Searched within, and filtered by agent.
    assert _paths(await _files(section="screenshots", q="chrome")) == {
        f"saved/media/{SESSION}/browser_chrome_1795526299001.jpg"}
    assert _paths(await _files(section="screenshots", agent="web-agent")) == SHOTS
    # An unknown section is the library's own list.
    assert _paths(await _files(section="nonsense")) == FILES


async def test_one_still_opens(deck: Path):
    await _scan()
    item = next(i for i in (await _files(section="screenshots"))["items"]
                if i["filename"].startswith("browser_screenshot_"))
    async with _client() as c:
        r = await c.get(f"/fd/file-library/files/{item['id']}/view", headers=_auth())
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"


# ── Filed without a model ──


async def test_a_scan_never_shows_them_to_a_model(deck: Path, monkeypatch: pytest.MonkeyPatch):
    read: list[str] = []

    async def a_tier(*_a, **_kw):
        return {"provider": "openai", "model": "small", "base_url": ""}

    async def summarize(row, tier, owner_id, reach=None):
        read.append(row["rel_path"])
        return {"status": "done", "title": row["filename"], "summary": "Read.", "category": "notes",
                "tags": "[]", "excerpt": "", "error": "", "model": "small", "tokens": 1,
                "summarized_at": 1.0, "waiting_for": ""}

    async def never_spent(_uid):
        return False

    monkeypatch.setattr(fl, "_resolve_tier", a_tier)
    monkeypatch.setattr(fl, "summarize_row", summarize)
    monkeypatch.setattr(fl, "_quota_spent", never_spent)
    job = fl.ScanJob()
    await fl.run_scan({"id": OWNER}, job)
    assert set(read) == FILES
    assert job.total == len(FILES)
    body = await _files(section="screenshots")
    assert {i["status"] for i in body["items"]} == {"skipped"}


async def test_one_set_aside_is_filed_and_one_described_on_request_is_kept(deck: Path):
    await _scan()
    store = fl.get_store()
    by_name = {i["filename"]: i for i in (await _files(section="screenshots"))["items"]}
    waiting = by_name["browser_chrome_1795526299001.jpg"]
    described = by_name["browser_zoom_1795526300412.jpg"]
    # From before: one set aside for want of vision, one described on request.
    row = store.get(waiting["id"])
    store.save_result(row["id"], row["fingerprint"], status="skipped", waiting_for="vision",
                      summary="Not described: its agent's vision is off.")
    row = store.get(described["id"])
    store.save_result(row["id"], row["fingerprint"], status="done", title="Checkout page",
                      summary="The shop's checkout with two items in the cart.")
    await _scan()
    assert store.get(waiting["id"])["summary"] == fl._SHOT_SUMMARY
    assert store.get(waiting["id"])["waiting_for"] == ""
    kept = store.get(described["id"])
    assert kept["status"] == "done" and kept["title"] == "Checkout page"
