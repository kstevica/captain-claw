"""Smart-glasses HUD: the page, its PWA assets, and its config.

The HUD is a separate Vite entry (``flight-deck/hud.html`` →
``src/hud/main.tsx``) built into ``static/hud.html`` next to the dashboard's
``index.html``, so the glasses never download the dashboard bundle. It runs
on Meta Ray-Ban Display (600x600 WebView) and other glasses web hosts.

Routes (this router is included before ``spa_catch_all`` in ``server.py``):

* ``GET /hud/manifest.webmanifest`` — the launcher manifest (PNG icons only:
  Meta's launcher rejects SVG).
* ``GET /hud/sw.js`` — a small service worker: cache-first for hashed
  ``/assets/*``, network-first for ``/hud*`` navigations (falls back to the
  last good shell when Flight Deck doesn't answer: a network error, a slow
  link, or a tunnel's 5xx page). It never touches ``/fd/*``.
* ``GET /hud``, ``/hud/``, ``/hud/{rest}`` — the built page, never cached by
  the browser (the SW is the only cache) and never framed. ``/hud/pair`` (the
  phone/desktop approval page for device pairing) is one of these client-side
  routes.
* ``GET /fd/hud/config`` — signed-in: the rendering rules the HUD prepends to
  the first chat message of a connection.

Plus ``root_override`` — called by ``spa_catch_all`` for ``GET /`` only — which
sends the Display's WebView to ``/hud/`` unless the wearer opted out with
``/?ui=full``; and ``HudDeliveryMiddleware`` (registered in ``server.py``),
which gzips the HUD page and the built ``/assets/*`` and lets browsers keep
the content-hashed assets for good.

The HUD's data comes from the owner-checked ``/fd/*`` API and
``/fd/agent-ws``; nothing here uses the unauthenticated ``/glasses/*`` bridge.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.auth_routes import _cookie_secure
from captain_claw.flight_deck.glasses_bridge import (
    _GLASSES_SYSTEM_CONTEXT,
    _NO_CACHE,
    _PNG_ICONS,
)

router = APIRouter(tags=["hud"])

# Where the built ``hud.html`` lives. Module-level so tests can monkeypatch it.
STATIC_DIR = Path(__file__).parent / "static"

HUD_PAGE = "hud.html"

# ── Manifest ─────────────────────────────────────────────────────────

_HUD_MANIFEST = {
    "name": "Captain Claw",
    "short_name": "Claw",
    "description": (
        "Captain Claw on smart glasses: chat with your agents, read their "
        "markdown files and browse their datastore tables."
    ),
    "start_url": "/hud/",
    "scope": "/hud/",
    "display": "standalone",
    "background_color": "#000000",
    "theme_color": "#000000",
    "icons": _PNG_ICONS,
}


@router.get("/hud/manifest.webmanifest")
async def hud_manifest() -> Response:
    return Response(
        content=json.dumps(_HUD_MANIFEST).encode("utf-8"),
        media_type="application/manifest+json",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ── Service worker ───────────────────────────────────────────────────
# Versioned cache: bump HUD_CACHE when the caching rules change. Built assets
# are content-hashed, so cache-first is safe for them; the page itself is
# network-first, so a redeploy is picked up on the next online launch (Meta:
# "cache-first can serve a stale build after redeploy"). Only GET + same-origin
# requests are considered; /fd/* (API, auth) is never intercepted.
#
# The deck normally sits behind a tunnel or reverse proxy (Meta needs public
# HTTPS), so "Flight Deck is down" usually arrives as the proxy's 502/530 page,
# not as a network error. Any such answer — or none within SHELL_TIMEOUT_MS —
# boots the last good shell instead, whose own loop retries Flight Deck. Only
# the real page (marked X-HUD-Shell by _hud_page) is ever stored as the shell.

HUD_CACHE = "hud-v2"
HUD_SHELL_HEADER = "X-HUD-Shell"

_HUD_SW = """// Captain Claw HUD service worker (served by hud_routes.py).
'use strict';
const CACHE = '__HUD_CACHE__';
const SHELL = '/hud/';
const MAX_ASSETS = 60;
// How long a launch waits for the page before booting the cached shell.
const SHELL_TIMEOUT_MS = 4000;
// Files under /hud/ that are not the page: never handled as the shell.
const NOT_PAGES = ['/hud/sw.js', '/hud/manifest.webmanifest'];

self.addEventListener('install', () => { self.skipWaiting(); });

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names
      .filter((n) => n.startsWith('hud-') && n !== CACHE)
      .map((n) => caches.delete(n)));
    await self.clients.claim();
  })());
});

async function trimAssets(cache) {
  const keys = (await cache.keys()).filter((r) => new URL(r.url).pathname.startsWith('/assets/'));
  for (let i = 0; i < keys.length - MAX_ASSETS; i++) await cache.delete(keys[i]);
}

async function assetFirst(request) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(request);
  if (hit) return hit;
  const res = await fetch(request);
  if (res.ok) {
    await cache.put(request, res.clone());
    trimAssets(cache).catch(() => {});
  }
  return res;
}

// The HUD page itself (Flight Deck marks it) — never a proxy's error page, a
// tunnel's interstitial, or the manifest opened in a tab.
function isShell(res) {
  return res.status === 200 && res.headers.get('__HUD_SHELL_HEADER__') === '1'
    && (res.headers.get('Content-Type') || '').startsWith('text/html');
}

// Answers the wearer must see as they are: any 2xx (a tunnel's interstitial),
// redirects (an access gate's sign-in: navigations fetch with redirect
// 'manual') and auth challenges. Any other error (502/503/504, a tunnel's
// 530, ngrok's offline 404) means Flight Deck isn't answering.
function passThrough(res) {
  return res.ok || res.type === 'opaqueredirect' || res.status === 401 || res.status === 407;
}

function shellNetworkFirst(event) {
  const network = fetch(event.request).then(async (res) => {
    if (isShell(res)) {
      const cache = await caches.open(CACHE);
      await cache.put(SHELL, res.clone()).catch(() => {});
    }
    return res;
  });
  // An answer that comes after the timeout still refreshes the cached shell.
  event.waitUntil(network.catch(() => {}));
  const lastShell = async () => (await caches.open(CACHE)).match(SHELL);
  return (async () => {
    let timer;
    const timeout = new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('timeout')), SHELL_TIMEOUT_MS);
    });
    let res;
    try {
      res = await Promise.race([network, timeout]);
    } catch (err) {
      // Network error or no answer in time: the last good shell; nothing
      // cached yet (first launch): keep waiting for the network.
      return (await lastShell()) || network;
    } finally {
      clearTimeout(timer);
    }
    if (passThrough(res)) return res;
    return (await lastShell()) || res;
  })();
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith('/assets/')) {
    event.respondWith(assetFirst(request));
    return;
  }
  if (request.mode === 'navigate' && (url.pathname === '/hud' || url.pathname.startsWith('/hud/'))
      && !NOT_PAGES.includes(url.pathname)) {
    event.respondWith(shellNetworkFirst(event));
  }
  // Everything else (notably /fd/*) goes straight to the network.
});
""".replace("__HUD_CACHE__", HUD_CACHE).replace("__HUD_SHELL_HEADER__", HUD_SHELL_HEADER)


@router.get("/hud/sw.js")
async def hud_service_worker() -> Response:
    return Response(
        content=_HUD_SW.encode("utf-8"),
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/hud/"},
    )


# ── The page ─────────────────────────────────────────────────────────
# Registered after the two routes above so /hud/{rest} never shadows them.

_NOT_BUILT = "HUD not built — run npm run build in flight-deck"

# Never framed: /hud/pair is a one-click "sign this device in" page, and a
# framing page on the same site (another port, a sibling subdomain) would get
# the approver's cookie and could clickjack it. Same as the OAuth consent pages.
_HUD_PAGE_HEADERS = {**_NO_CACHE, "X-Frame-Options": "DENY",
                     "Content-Security-Policy": "frame-ancestors 'none'"}


def _hud_page() -> Response:
    page = STATIC_DIR / HUD_PAGE
    try:
        body = page.read_bytes()
    except OSError:
        return Response(_NOT_BUILT, status_code=503, media_type="text/plain; charset=utf-8",
                        headers=_HUD_PAGE_HEADERS)
    # The marker is how the service worker tells the page from anything else.
    return Response(body, media_type="text/html; charset=utf-8",
                    headers={**_HUD_PAGE_HEADERS, HUD_SHELL_HEADER: "1"})


@router.get("/hud")
async def hud_root() -> Response:
    return _hud_page()


@router.get("/hud/")
async def hud_index() -> Response:
    return _hud_page()


@router.get("/hud/{rest:path}")
async def hud_any(rest: str) -> Response:
    # Client-side routes (/hud/pair, …); the screen lives in the query string.
    return _hud_page()


# ── Config ───────────────────────────────────────────────────────────


@router.get("/fd/hud/config")
async def hud_config(user: dict = Depends(get_current_user)) -> Response:
    return Response(
        content=json.dumps({"surface_rules": _GLASSES_SYSTEM_CONTEXT}).encode("utf-8"),
        media_type="application/json",
        headers=_NO_CACHE,
    )


# ── Root redirect for the Display's WebView ──────────────────────────
# Meta Ray-Ban Display's WebView identifies as Android WebView ("; wv)") on a
# "Greatwhite" build. Undocumented by Meta, so it only drives this convenience
# redirect; /hud/ is the URL to register in the Meta AI app.

UI_COOKIE = "fd_ui"
UI_COOKIE_MAX_AGE = 365 * 24 * 3600


def is_display_webview(user_agent: str) -> bool:
    return "Greatwhite" in user_agent and "; wv)" in user_agent


# What GET / returns depends on the User-Agent and the fd_ui cookie, so no
# cached copy may answer for the server: a WebView that once got the dashboard
# at / would otherwise keep it and never see the redirect (or /?ui=hud).
ROOT_HEADERS = {"Cache-Control": "no-cache", "Vary": "User-Agent, Cookie"}


def _to_hud() -> RedirectResponse:
    return RedirectResponse("/hud/", status_code=302,
                            headers={**ROOT_HEADERS, "Cache-Control": "no-store"})


def root_override(request: Request, index_file: Path) -> Response | None:
    """For ``GET /`` only: a response that replaces the dashboard, or None.

    * ``/?ui=full`` — serve the dashboard and remember the choice (cookie
      ``fd_ui=full``) so the glasses stop being redirected;
    * ``/?ui=hud`` — forget that choice and go to ``/hud/``;
    * the Display's WebView without the opt-out cookie — ``302 /hud/``.

    Any other request: None (the caller serves ``index.html`` with
    ``ROOT_HEADERS``).
    """
    ui = request.query_params.get("ui", "")
    if ui == "full":
        resp = FileResponse(index_file, headers=ROOT_HEADERS)
        resp.set_cookie(UI_COOKIE, "full", max_age=UI_COOKIE_MAX_AGE, path="/",
                        samesite="lax", httponly=True,
                        # Same rule as the refresh cookie (FD_COOKIE_SECURE / FD_LOCKDOWN).
                        secure=_cookie_secure())
        return resp
    if ui == "hud":
        resp = _to_hud()
        resp.delete_cookie(UI_COOKIE, path="/")
        return resp
    if (is_display_webview(request.headers.get("user-agent", ""))
            and request.cookies.get(UI_COOKIE) != "full"):
        return _to_hud()
    return None


# ── Compression and caching for the HUD's first load ─────────────────
# Meta's budget for a first load is under 300 KB at ~500 Kbps. The HUD's page,
# JS and CSS are ~490 KB raw but ~150 KB gzipped, and many tunnels (ngrok, an
# ssh tunnel, stock nginx) don't compress. So Flight Deck gzips them itself,
# scoped by path: only /hud* and the built text files in /assets/, never /fd/*
# (a compressor must not buffer SSE, ndjson or chat streams). Content-hashed
# assets are also cacheable for good: a new build has new names.

_ASSET_PREFIX = "/assets/"
_COMPRESSIBLE_ASSET = re.compile(r"\.(?:js|mjs|css|json|map|svg|html|txt)$")
_HASHED_ASSET = re.compile(r"-[A-Za-z0-9_-]{8}\.[A-Za-z0-9]+$")  # Vite's [name]-[hash].[ext]
_IMMUTABLE = "public, max-age=31536000, immutable"
_GZIP_MIN_BYTES = 1024


def _with_immutable_cache(send: Send) -> Send:
    async def send_cached(message: Message) -> None:
        if message["type"] == "http.response.start" and message["status"] in (200, 304):
            MutableHeaders(scope=message)["Cache-Control"] = _IMMUTABLE
        await send(message)
    return send_cached


class HudDeliveryMiddleware:
    """gzip for ``/hud*`` and the text files in ``/assets/`` (no Range
    requests), plus ``Cache-Control: immutable`` for content-hashed
    ``/assets/*``. Every other path goes straight through, untouched."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.gzip = GZipMiddleware(app, minimum_size=_GZIP_MIN_BYTES, compresslevel=6)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope["path"]
        asset = path.startswith(_ASSET_PREFIX)
        if asset and _HASHED_ASSET.search(path):
            send = _with_immutable_cache(send)
        hud = path == "/hud" or path.startswith("/hud/")
        if ((hud or (asset and _COMPRESSIBLE_ASSET.search(path)))
                and "range" not in Headers(scope=scope)):
            await self.gzip(scope, receive, send)
        else:
            await self.app(scope, receive, send)
