"""Smart-glasses HUD routes (hud_routes.py) as wired into the Flight Deck app.

Pinned here:

* /hud, /hud/ and /hud/<anything> (incl. /hud/pair) serve the built hud.html,
  never cached; a missing build is a 503 that says how to build it;
* the manifest (PNG icons, /hud/ scope) and the service worker's headers and
  rules (never intercepts /fd/*);
* /fd/hud/config is signed-in only and returns the glasses rendering rules;
* GET / sends Meta Ray-Ban Display's WebView to /hud/ unless the wearer opted
  out with /?ui=full (cookie fd_ui=full); /?ui=hud forgets the opt-out; other
  user agents and other paths are untouched.

The page is a fake hud.html in a tmp STATIC_DIR, so these tests don't depend on
the frontend build.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import glasses_bridge, hud_routes, server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

USER = "user-hud"
GLASSES_UA = ("Mozilla/5.0 (Linux; Android 14; Greatwhite Build/UKQ1.250303.001; wv) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/146.0.7680.177 "
              "Mobile Safari/537.36")
DESKTOP_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/146.0.0.0 Safari/537.36")
# A Chrome WebView on some other Android phone: not the Display.
PHONE_WEBVIEW_UA = ("Mozilla/5.0 (Linux; Android 14; Pixel 8; wv) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Version/4.0 Chrome/146.0.0.0 Mobile Safari/537.36")

HUD_HTML = ('<!doctype html><html><head><meta name="mrbd-web-app-capable" content="yes">'
            '<title>Captain Claw</title></head><body><div id="root">HUD</div></body></html>')
INDEX_HTML = "<!doctype html><html><body>DASHBOARD</body></html>"


@pytest.fixture
def static_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "static"
    d.mkdir()
    (d / "hud.html").write_text(HUD_HTML, encoding="utf-8")
    (d / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    monkeypatch.setattr(hud_routes, "STATIC_DIR", d)
    monkeypatch.setattr(server, "STATIC_DIR", d)
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.delenv("FD_COOKIE_SECURE", raising=False)
    return d


@pytest.fixture
async def db(tmp_path: Path):
    prev = fd_auth._db
    fdb = FlightDeckDB(tmp_path / "fd.db")
    await fdb.init()
    set_auth_db(fdb)
    await fdb._db.execute(
        "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
        " VALUES (?, 'hud@x.co', 'h', 'Hud User', 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        (USER,))
    await fdb._db.commit()
    try:
        yield fdb
    finally:
        await fdb.close()
        fd_auth._db = prev


def _client(**kw) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=("203.0.113.7", 40000)),
        base_url="http://fd.test", **kw)


def _no_store(r: httpx.Response) -> bool:
    return "no-store" in r.headers.get("cache-control", "")


# ── The page ────────────────────────────────────────────────────────────────


class TestPage:
    @pytest.mark.parametrize("path", ["/hud", "/hud/", "/hud/pair", "/hud/pair?code=BCDF-GHJK",
                                      "/hud/?v=agent&a=x&t=chat", "/hud/deep/link"])
    async def test_serves_the_built_page_uncached(self, static_dir, path):
        async with _client() as c:
            r = await c.get(path)
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/html")
        assert '<meta name="mrbd-web-app-capable" content="yes">' in r.text
        assert _no_store(r)

    async def test_missing_build_is_a_503_that_says_how_to_build(self, static_dir):
        (static_dir / "hud.html").unlink()
        async with _client() as c:
            r = await c.get("/hud/")
        assert r.status_code == 503
        assert "npm run build" in r.text

    async def test_glasses_ua_reaches_the_page_without_redirect_loops(self, static_dir):
        async with _client() as c:
            r = await c.get("/hud/", headers={"User-Agent": GLASSES_UA})
        assert r.status_code == 200 and "HUD" in r.text


# ── PWA assets ──────────────────────────────────────────────────────────────


class TestPwa:
    async def test_manifest(self, static_dir):
        async with _client() as c:
            r = await c.get("/hud/manifest.webmanifest")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/manifest+json")
        assert "max-age=3600" in r.headers["cache-control"]
        m = r.json()
        assert m["name"] == "Captain Claw" and m["short_name"] == "Claw"
        assert m["start_url"] == "/hud/" and m["scope"] == "/hud/"
        assert m["display"] == "standalone"
        assert m["background_color"] == "#000000" and m["theme_color"] == "#000000"
        assert m["description"]
        assert m["icons"] == glasses_bridge._PNG_ICONS
        assert all(i["type"] == "image/png" for i in m["icons"])  # Meta rejects SVG

    async def test_service_worker_headers(self, static_dir):
        async with _client() as c:
            r = await c.get("/hud/sw.js")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/javascript")
        assert r.headers["cache-control"] == "no-cache"
        assert r.headers["service-worker-allowed"] == "/hud/"

    async def test_service_worker_rules(self, static_dir):
        async with _client() as c:
            js = (await c.get("/hud/sw.js")).text
        assert "const CACHE = 'hud-v1';" in js
        assert "skipWaiting" in js and "clients.claim" in js
        assert "n.startsWith('hud-') && n !== CACHE" in js           # old HUD caches only
        assert "request.method !== 'GET'" in js
        assert "url.origin !== self.location.origin" in js
        assert "startsWith('/assets/')" in js                         # cache-first assets
        assert "request.mode === 'navigate' && url.pathname.startsWith('/hud')" in js
        assert "cache.put(SHELL" in js and "const SHELL = '/hud/';" in js
        assert "if (res.ok)" in js                                    # never caches errors
        # The API and auth are never intercepted or cached.
        assert "/fd/" not in js.replace("// Everything else (notably /fd/*)", "")


# ── Config ──────────────────────────────────────────────────────────────────


class TestConfig:
    async def test_requires_a_session(self, static_dir, db):
        async with _client() as c:
            r = await c.get("/fd/hud/config")
        assert r.status_code == 401

    async def test_returns_the_surface_rules(self, static_dir, db):
        async with _client() as c:
            r = await c.get("/fd/hud/config",
                            headers={"Authorization": f"Bearer {create_access_token(USER)}"})
        assert r.status_code == 200, r.text
        assert r.json() == {"surface_rules": glasses_bridge._GLASSES_SYSTEM_CONTEXT}
        assert _no_store(r)


# ── GET / for the Display's WebView ─────────────────────────────────────────


class TestRootRedirect:
    def test_display_webview_detection(self):
        assert hud_routes.is_display_webview(GLASSES_UA)
        assert not hud_routes.is_display_webview(DESKTOP_UA)
        assert not hud_routes.is_display_webview(PHONE_WEBVIEW_UA)
        # Greatwhite without the WebView marker (e.g. a desktop UA spoof test) — no.
        assert not hud_routes.is_display_webview("Mozilla/5.0 (Linux; Android 14; Greatwhite) Chrome/146")

    async def test_glasses_go_to_the_hud(self, static_dir):
        async with _client() as c:
            r = await c.get("/", headers={"User-Agent": GLASSES_UA})
        assert r.status_code == 302
        assert r.headers["location"] == "/hud/"
        assert _no_store(r)

    @pytest.mark.parametrize("ua", [DESKTOP_UA, PHONE_WEBVIEW_UA, ""])
    async def test_other_browsers_get_the_dashboard(self, static_dir, ua):
        async with _client() as c:
            r = await c.get("/", headers={"User-Agent": ua})
        assert r.status_code == 200 and "DASHBOARD" in r.text
        assert "fd_ui" not in r.headers.get("set-cookie", "")

    async def test_only_the_root_redirects(self, static_dir):
        async with _client() as c:
            spa = await c.get("/agents/whatever", headers={"User-Agent": GLASSES_UA})
            api = await c.get("/fd/definitely-not-a-route", headers={"User-Agent": GLASSES_UA})
        assert spa.status_code == 200 and "DASHBOARD" in spa.text
        assert api.status_code == 404  # API prefixes still 404, never HTML

    async def test_ui_full_opts_out_with_a_cookie(self, static_dir):
        async with _client() as c:
            r = await c.get("/?ui=full", headers={"User-Agent": GLASSES_UA})
            assert r.status_code == 200 and "DASHBOARD" in r.text
            cookie = r.headers["set-cookie"]
            assert cookie.startswith("fd_ui=full;")
            low = cookie.lower()
            assert "path=/;" in low or low.endswith("path=/")
            assert "max-age=31536000" in low and "httponly" in low and "samesite=lax" in low
            # The client now carries fd_ui=full: the glasses stay on the dashboard.
            again = await c.get("/", headers={"User-Agent": GLASSES_UA})
        assert again.status_code == 200 and "DASHBOARD" in again.text

    async def test_opt_out_cookie_is_respected(self, static_dir):
        async with _client(cookies={"fd_ui": "full"}) as c:
            r = await c.get("/", headers={"User-Agent": GLASSES_UA})
        assert r.status_code == 200 and "DASHBOARD" in r.text

    async def test_ui_hud_forgets_the_opt_out(self, static_dir):
        async with _client() as c:
            opted_out = await c.get("/?ui=full", headers={"User-Agent": GLASSES_UA})
            assert opted_out.status_code == 200 and c.cookies.get("fd_ui") == "full"
            r = await c.get("/?ui=hud", headers={"User-Agent": DESKTOP_UA})
            assert r.status_code == 302 and r.headers["location"] == "/hud/"
            cookie = r.headers["set-cookie"].lower()
            assert cookie.startswith("fd_ui=") and "max-age=0" in cookie and "path=/" in cookie
            again = await c.get("/", headers={"User-Agent": GLASSES_UA})
        assert again.status_code == 302 and again.headers["location"] == "/hud/"
