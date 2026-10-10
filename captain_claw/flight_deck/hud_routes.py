"""Smart-glasses HUD: the page, its PWA assets, and its config.

The HUD is a separate Vite entry (``flight-deck/hud.html`` →
``src/hud/main.tsx``) built into ``static/hud.html`` next to the dashboard's
``index.html``, so the glasses never download the dashboard bundle. It runs
on Meta Ray-Ban Display (600x600 WebView) and other glasses web hosts.

Routes (this router is included before ``spa_catch_all`` in ``server.py``):

* ``GET /hud/manifest.webmanifest`` — the launcher manifest (PNG icons only:
  Meta's launcher rejects SVG).
* ``GET /hud/sw.js`` — a small service worker: cache-first for hashed
  ``/assets/*``, network-first for ``/hud*`` navigations (offline fallback to
  the last good shell). It never touches ``/fd/*``.
* ``GET /hud``, ``/hud/``, ``/hud/{rest}`` — the built page, never cached by
  the browser (the SW is the only cache). ``/hud/pair`` (the phone/desktop
  approval page for device pairing) is one of these client-side routes.
* ``GET /fd/hud/config`` — signed-in: the rendering rules the HUD prepends to
  the first chat message of a connection.

Plus ``root_override`` — called by ``spa_catch_all`` for ``GET /`` only — which
sends the Display's WebView to ``/hud/`` unless the wearer opted out with
``/?ui=full``.

The HUD's data comes from the owner-checked ``/fd/*`` API and
``/fd/agent-ws``; nothing here uses the unauthenticated ``/glasses/*`` bridge.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, RedirectResponse, Response

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

HUD_CACHE = "hud-v1"

_HUD_SW = """// Captain Claw HUD service worker (served by hud_routes.py).
'use strict';
const CACHE = '__HUD_CACHE__';
const SHELL = '/hud/';
const MAX_ASSETS = 60;

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

async function shellNetworkFirst(request) {
  const cache = await caches.open(CACHE);
  try {
    const res = await fetch(request);
    if (res.ok) await cache.put(SHELL, res.clone());
    return res;
  } catch (err) {
    const hit = await cache.match(SHELL);
    if (hit) return hit;
    throw err;
  }
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
  if (request.mode === 'navigate' && url.pathname.startsWith('/hud')) {
    event.respondWith(shellNetworkFirst(request));
  }
  // Everything else (notably /fd/*) goes straight to the network.
});
""".replace("__HUD_CACHE__", HUD_CACHE)


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


def _hud_page() -> Response:
    page = STATIC_DIR / HUD_PAGE
    try:
        body = page.read_bytes()
    except OSError:
        return Response(_NOT_BUILT, status_code=503, media_type="text/plain; charset=utf-8",
                        headers=_NO_CACHE)
    return Response(body, media_type="text/html; charset=utf-8", headers=_NO_CACHE)


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


def _to_hud() -> RedirectResponse:
    return RedirectResponse("/hud/", status_code=302, headers={"Cache-Control": "no-store"})


def root_override(request: Request, index_file: Path) -> Response | None:
    """For ``GET /`` only: a response that replaces the dashboard, or None.

    * ``/?ui=full`` — serve the dashboard and remember the choice (cookie
      ``fd_ui=full``) so the glasses stop being redirected;
    * ``/?ui=hud`` — forget that choice and go to ``/hud/``;
    * the Display's WebView without the opt-out cookie — ``302 /hud/``.

    Any other request: None (the caller serves ``index.html`` as before).
    """
    ui = request.query_params.get("ui", "")
    if ui == "full":
        resp = FileResponse(index_file)
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
