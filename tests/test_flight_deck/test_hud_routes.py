"""Smart-glasses HUD routes (hud_routes.py) as wired into the Flight Deck app.

Pinned here:

* /hud, /hud/ and /hud/<anything> (incl. /hud/pair) serve the built hud.html,
  never cached, never framed, marked X-HUD-Shell; a missing build is a 503
  that says how to build it (no marker);
* the manifest (PNG icons, /hud/ scope) and the service worker's headers and
  rules (never intercepts /fd/*) — and, run under Node with fake caches/fetch,
  its behaviour: the last good shell answers network errors, a slow link and a
  tunnel's 5xx; only the marked page is ever cached; redirects, auth
  challenges and other 2xx pass through; the manifest/sw.js aren't pages;
* /fd/hud/config is signed-in only and returns the glasses rendering rules;
* GET / sends Meta Ray-Ban Display's WebView to /hud/ unless the wearer opted
  out with /?ui=full (cookie fd_ui=full); /?ui=hud forgets the opt-out; other
  user agents and other paths are untouched; every answer at / is revalidated
  (no-cache/no-store + Vary: User-Agent, Cookie);
* HudDeliveryMiddleware: gzip for /hud* and the built text assets (not small
  ones, not Range requests, never /fd/*); immutable caching for hashed assets.

The page is a fake hud.html in a tmp STATIC_DIR (and /assets/ a tmp dir), so
these tests don't depend on the frontend build.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
from starlette.routing import Mount

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


def _not_framable(r: httpx.Response) -> bool:
    return (r.headers.get("x-frame-options") == "DENY"
            and r.headers.get("content-security-policy") == "frame-ancestors 'none'")


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
        # Never framed (the /hud/pair approval page is a clickjacking target)…
        assert _not_framable(r)
        # …and marked, so the service worker only ever caches this page.
        assert r.headers[hud_routes.HUD_SHELL_HEADER] == "1"

    async def test_missing_build_is_a_503_that_says_how_to_build(self, static_dir):
        (static_dir / "hud.html").unlink()
        async with _client() as c:
            r = await c.get("/hud/")
        assert r.status_code == 503
        assert "npm run build" in r.text
        assert _no_store(r) and _not_framable(r)
        assert hud_routes.HUD_SHELL_HEADER.lower() not in r.headers

    async def test_only_the_page_is_marked(self, static_dir):
        async with _client() as c:
            for path in ("/hud/manifest.webmanifest", "/hud/sw.js"):
                r = await c.get(path)
                assert r.status_code == 200
                assert hud_routes.HUD_SHELL_HEADER.lower() not in r.headers

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
        assert "const CACHE = 'hud-v2';" in js                       # bumped with the new rules
        assert "skipWaiting" in js and "clients.claim" in js
        assert "n.startsWith('hud-') && n !== CACHE" in js           # old HUD caches only
        assert "request.method !== 'GET'" in js
        assert "url.origin !== self.location.origin" in js
        assert "startsWith('/assets/')" in js                         # cache-first assets
        assert "request.mode === 'navigate'" in js
        assert "url.pathname === '/hud' || url.pathname.startsWith('/hud/')" in js
        assert "const NOT_PAGES = ['/hud/sw.js', '/hud/manifest.webmanifest'];" in js
        assert "!NOT_PAGES.includes(url.pathname)" in js
        assert "cache.put(SHELL" in js and "const SHELL = '/hud/';" in js
        # Only the marked 200 HTML page becomes the shell…
        assert ("res.status === 200 && res.headers.get('X-HUD-Shell') === '1'" in js
                and ".startsWith('text/html')" in js)
        assert "if (isShell(res))" in js
        # …and a gateway's error page, a network error or a slow link get the
        # last good shell, while 2xx, redirects and auth challenges pass through.
        assert "res.ok || res.type === 'opaqueredirect' || res.status === 401" in js
        assert "if (passThrough(res)) return res;" in js
        assert "return (await lastShell()) || res;" in js
        assert "return (await lastShell()) || network;" in js
        assert "const SHELL_TIMEOUT_MS = 4000;" in js and "Promise.race([network, timeout])" in js
        # The API and auth are never intercepted or cached.
        assert "/fd/" not in js.replace("// Everything else (notably /fd/*)", "")

    @pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the service worker")
    def test_service_worker_behaviour(self, tmp_path):
        sw = tmp_path / "sw.js"
        sw.write_text(hud_routes._HUD_SW, encoding="utf-8")
        harness = tmp_path / "harness.cjs"
        harness.write_text(_SW_HARNESS, encoding="utf-8")
        run = subprocess.run([shutil.which("node"), str(harness), str(sw), hud_routes.HUD_CACHE],
                             capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stderr
        got = json.loads(run.stdout)
        shell = {"handled": True, "status": 200, "body": "SHELL-1"}

        # First launch, nothing cached: whatever the network gives.
        assert got["first_down"]["handled"] and "error" in got["first_down"]
        assert got["first_502"] == {"handled": True, "status": 502, "body": "tunnel 502"}
        assert got["first_shell"] == shell and got["cached_after_first"] == "SHELL-1"
        # Flight Deck down behind a tunnel, or no network: the last good shell.
        for case in ("down_502", "down_530", "down_503", "ngrok_404", "network_error", "slow_link"):
            assert got[case] == shell, case
        # Passed through as is, never cached.
        assert got["interstitial"] == {"handled": True, "status": 200, "body": "ngrok warning"}
        assert got["access_gate"] == {"handled": True, "status": 0, "body": "opaqueredirect"}
        assert got["auth_challenge"] == {"handled": True, "status": 401, "body": "who are you"}
        assert got["cached_after_passthrough"] == "SHELL-1"
        # Not pages / not the HUD / not navigations: never handled.
        for case in ("manifest_nav", "sw_nav", "fd_nav", "other_page", "post_nav"):
            assert got[case] == {"handled": False}, case
        # A late answer (after the timeout) still refreshes the shell.
        assert got["late_shell"] == shell and got["cached_after_late"] == "SHELL-2"
        # Deep links are the same page.
        assert got["deep_link"] == {"handled": True, "status": 200, "body": "SHELL-3"}
        assert got["cached_after_deep_link"] == "SHELL-3"


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


def _varies(r: httpx.Response) -> bool:
    vary = {v.strip().lower() for v in r.headers.get("vary", "").split(",")}
    return {"user-agent", "cookie"} <= vary


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
        assert _no_store(r) and _varies(r)

    @pytest.mark.parametrize("ua", [DESKTOP_UA, PHONE_WEBVIEW_UA, ""])
    async def test_other_browsers_get_the_dashboard(self, static_dir, ua):
        async with _client() as c:
            r = await c.get("/", headers={"User-Agent": ua})
        assert r.status_code == 200 and "DASHBOARD" in r.text
        assert "fd_ui" not in r.headers.get("set-cookie", "")
        # Depends on the UA and fd_ui: a cached copy must never answer for the
        # server, or the Display would keep the dashboard instead of the redirect.
        assert r.headers["cache-control"] == "no-cache" and _varies(r)

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
            assert r.headers["cache-control"] == "no-cache" and _varies(r)
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
            assert _no_store(r) and _varies(r)
            cookie = r.headers["set-cookie"].lower()
            assert cookie.startswith("fd_ui=") and "max-age=0" in cookie and "path=/" in cookie
            again = await c.get("/", headers={"User-Agent": GLASSES_UA})
        assert again.status_code == 302 and again.headers["location"] == "/hud/"


# ── Compression and asset caching (HudDeliveryMiddleware) ───────────────────

BIG_JS = "export const x = " + json.dumps(["captain claw hud"] * 400) + ";\n"  # ~8 KB, compressible


@pytest.fixture
def assets_dir(tmp_path: Path, static_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the app's /assets mount at a tmp dir."""
    mount = next((r for r in server.app.routes if isinstance(r, Mount) and r.path == "/assets"), None)
    if mount is None:
        pytest.skip("no /assets mount (flight_deck/static missing)")
    d = tmp_path / "assets"
    d.mkdir()
    (d / "hud-AbCd12_-.js").write_text(BIG_JS, encoding="utf-8")
    (d / "tiny-Zz9_Yy8-.js").write_text("export {};\n", encoding="utf-8")
    (d / "font-QwErTy12.woff2").write_bytes(b"\x00wOF2" + bytes(4000))
    (d / "plain.js").write_text(BIG_JS, encoding="utf-8")
    monkeypatch.setattr(mount.app, "all_directories", [d])
    return d


GZ = {"Accept-Encoding": "gzip, deflate, br"}


class TestDelivery:
    async def test_hashed_assets_are_gzipped_and_immutable(self, assets_dir):
        async with _client() as c:
            async with c.stream("GET", "/assets/hud-AbCd12_-.js", headers=GZ) as r:
                wire = b"".join([chunk async for chunk in r.aiter_raw()])
            raw = await c.get("/assets/hud-AbCd12_-.js", headers={"Accept-Encoding": "identity"})
        assert r.status_code == 200
        assert r.headers["content-encoding"] == "gzip"
        assert "accept-encoding" in r.headers["vary"].lower()
        assert gzip.decompress(wire) == BIG_JS.encode()
        assert len(wire) < len(BIG_JS) // 4
        assert r.headers["cache-control"] == "public, max-age=31536000, immutable"
        # Without gzip in Accept-Encoding: the same file, identity.
        assert raw.status_code == 200 and "content-encoding" not in raw.headers
        assert raw.text == BIG_JS
        assert raw.headers["cache-control"] == "public, max-age=31536000, immutable"

    async def test_small_binary_and_unhashed_assets(self, assets_dir):
        async with _client() as c:
            tiny = await c.get("/assets/tiny-Zz9_Yy8-.js", headers=GZ)
            font = await c.get("/assets/font-QwErTy12.woff2", headers=GZ)
            plain = await c.get("/assets/plain.js", headers=GZ)
            missing = await c.get("/assets/gone-AbCd1234.js", headers=GZ)
        assert tiny.status_code == 200 and "content-encoding" not in tiny.headers  # < 1 KB
        assert font.status_code == 200 and "content-encoding" not in font.headers  # already compressed
        assert font.headers["cache-control"] == "public, max-age=31536000, immutable"
        # No content hash in the name: compressed, but not cached for good.
        assert plain.headers["content-encoding"] == "gzip"
        assert "immutable" not in plain.headers.get("cache-control", "")
        assert missing.status_code == 404 and "immutable" not in missing.headers.get("cache-control", "")

    async def test_range_requests_are_not_compressed(self, assets_dir):
        async with _client() as c:
            r = await c.get("/assets/hud-AbCd12_-.js", headers={**GZ, "Range": "bytes=0-99"})
        assert r.status_code == 206 and "content-encoding" not in r.headers
        assert r.content == BIG_JS.encode()[:100]

    async def test_the_hud_page_is_compressed(self, static_dir):
        big = HUD_HTML.replace("HUD", "HUD" + " <p>glasses</p>" * 200)
        (static_dir / "hud.html").write_text(big, encoding="utf-8")
        async with _client() as c:
            r = await c.get("/hud/", headers=GZ)
            sw = await c.get("/hud/sw.js", headers=GZ)
        assert r.status_code == 200 and r.headers["content-encoding"] == "gzip"
        assert r.text == big and _not_framable(r) and _no_store(r)
        assert r.headers[hud_routes.HUD_SHELL_HEADER] == "1"
        assert sw.headers["content-encoding"] == "gzip" and sw.text == hud_routes._HUD_SW
        assert sw.headers["cache-control"] == "no-cache"  # not "immutable": no hash in the name

    async def test_fd_routes_are_never_compressed(self, static_dir, db):
        async with _client() as c:
            r = await c.get("/fd/hud/config", headers={
                **GZ, "Authorization": f"Bearer {create_access_token(USER)}"})
        assert r.status_code == 200
        assert len(r.content) > 1024  # big enough that a global GZip would have kicked in
        assert "content-encoding" not in r.headers
        assert "accept-encoding" not in r.headers.get("vary", "").lower()

    async def test_root_and_spa_pages_are_untouched(self, static_dir):
        (static_dir / "index.html").write_text(INDEX_HTML * 100, encoding="utf-8")
        async with _client() as c:
            root = await c.get("/", headers={**GZ, "User-Agent": DESKTOP_UA})
            spa = await c.get("/agents/x", headers={**GZ, "User-Agent": DESKTOP_UA})
        for r in (root, spa):
            assert r.status_code == 200 and "content-encoding" not in r.headers

    def test_middleware_is_installed(self):
        assert any(m.cls is hud_routes.HudDeliveryMiddleware for m in server.app.user_middleware)

    async def test_middleware_passes_other_paths_straight_through(self):
        seen: list[str] = []

        async def app(scope, receive, send):
            seen.append(scope["path"])
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"x" * 5000})

        mw = hud_routes.HudDeliveryMiddleware(app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mw),
                                     base_url="http://fd.test") as c:
            hud = await c.get("/hud/x", headers=GZ)
            hudson = await c.get("/hudson", headers=GZ)
            stream = await c.get("/fd/agent-ws/x", headers=GZ)
        assert hud.headers["content-encoding"] == "gzip"
        assert "content-encoding" not in hudson.headers and "content-encoding" not in stream.headers
        assert seen == ["/hud/x", "/hudson", "/fd/agent-ws/x"]


# Runs the service worker (hud_routes._HUD_SW) in a Node vm with fake caches,
# fetch and FetchEvents; prints what each navigation got as JSON.
_SW_HARNESS = r"""
const vm = require('vm');
const fs = require('fs');
const [srcPath, CACHE] = process.argv.slice(2);
const SRC = fs.readFileSync(srcPath, 'utf8');
if (!SRC.includes('const SHELL_TIMEOUT_MS = 4000;')) throw new Error('timeout constant moved');
const src = SRC.replace('const SHELL_TIMEOUT_MS = 4000;', 'const SHELL_TIMEOUT_MS = 50;');
const ORIGIN = 'http://fd.test';

class FakeCache {
  constructor() { this.map = new Map(); }
  key(r) { return new URL(typeof r === 'string' ? r : r.url, ORIGIN).pathname; }
  async match(r) { const hit = this.map.get(this.key(r)); return hit ? hit.clone() : undefined; }
  async put(r, res) {
    const body = await res.text();
    this.map.set(this.key(r), new Response(body, { status: res.status, headers: res.headers }));
  }
  async keys() { return [...this.map.keys()].map((k) => ({ url: ORIGIN + k })); }
  async delete(r) { return this.map.delete(this.key(r)); }
}
const stores = new Map();
const caches = {
  async open(name) { if (!stores.has(name)) stores.set(name, new FakeCache()); return stores.get(name); },
  async keys() { return [...stores.keys()]; },
  async delete(name) { return stores.delete(name); },
};
const handlers = {};
let network = async () => { throw new TypeError('no network'); };
const self = {
  location: { origin: ORIGIN },
  addEventListener: (type, fn) => { handlers[type] = fn; },
  skipWaiting() {},
  clients: { claim: async () => {} },
};
vm.runInContext(src, vm.createContext({
  self, caches, fetch: (req) => network(req), URL, Response, Headers, setTimeout, clearTimeout, console,
}));

const page = (body) => async () => new Response(body, {
  status: 200, headers: { 'Content-Type': 'text/html; charset=utf-8', 'X-HUD-Shell': '1' } });
const html = (status, body) => async () => new Response(body, {
  status, headers: { 'Content-Type': 'text/html' } });
const down = async () => { throw new TypeError('Failed to fetch'); };
const hang = () => new Promise(() => {});
const later = (ms, respond) => () => new Promise((resolve) => setTimeout(() => resolve(respond()), ms));
const opaqueRedirect = async () => ({ type: 'opaqueredirect', status: 0, ok: false, headers: new Headers() });

async function go(path, respond, { method = 'GET', mode = 'navigate', settle = true } = {}) {
  network = respond;
  const waits = [];
  let responded = null;
  handlers.fetch({
    request: { url: ORIGIN + path, method, mode },
    respondWith(p) { responded = Promise.resolve(p); },
    waitUntil(p) { waits.push(p); },
  });
  if (!responded) return { handled: false };
  let res;
  try { res = await responded; } catch (e) { return { handled: true, error: String(e) }; }
  if (settle) await Promise.all(waits);
  const body = res.type === 'opaqueredirect' ? 'opaqueredirect' : await res.text();
  return { handled: true, status: res.status, body };
}
const cachedShell = async () => {
  const hit = await (await caches.open(CACHE)).match('/hud/');
  return hit ? hit.text() : null;
};

(async () => {
  const out = {};
  out.first_down = await go('/hud/', down);
  out.first_502 = await go('/hud/', html(502, 'tunnel 502'));
  out.first_shell = await go('/hud/', page('SHELL-1'));
  out.cached_after_first = await cachedShell();
  out.down_502 = await go('/hud/', html(502, 'tunnel 502'));
  out.down_530 = await go('/hud/', html(530, 'cloudflare 530'));
  out.down_503 = await go('/hud/', html(503, 'HUD not built'));
  out.ngrok_404 = await go('/hud/', html(404, 'ERR_NGROK_3200'));
  out.network_error = await go('/hud/', down);
  out.slow_link = await go('/hud/', hang, { settle: false });
  out.interstitial = await go('/hud/', html(200, 'ngrok warning'));
  out.access_gate = await go('/hud/', opaqueRedirect);
  out.auth_challenge = await go('/hud/', html(401, 'who are you'));
  out.cached_after_passthrough = await cachedShell();
  out.manifest_nav = await go('/hud/manifest.webmanifest', html(200, '{}'));
  out.sw_nav = await go('/hud/sw.js', html(200, '//'));
  out.fd_nav = await go('/fd/auth/status', html(200, '{}'));
  out.other_page = await go('/hudson', html(200, 'x'));
  out.post_nav = await go('/hud/', page('POSTED'), { method: 'POST' });
  out.late_shell = await go('/hud/', later(150, page('SHELL-2')));
  out.cached_after_late = await cachedShell();
  out.deep_link = await go('/hud/pair?code=BCDF-GHJK', page('SHELL-3'));
  out.cached_after_deep_link = await cachedShell();
  process.stdout.write(JSON.stringify(out));
  process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
"""
