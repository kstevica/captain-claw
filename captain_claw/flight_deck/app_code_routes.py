"""HTTP routes for agent-coded apps.

This is the new (code-app) sibling of ``app_routes.py`` (manifest apps).
A "code app" is an agent-authored directory containing
``backend.py`` + ``frontend.html`` under
``~/.captain-claw-fd/apps/<slug>/``.

Three categories of routes live here:

1. **Discovery / lifecycle** (``/fd/code-apps``) — list, get, delete,
   restart. The agent's authoring loop and the FD client use these.

2. **Frontend serving** (``GET /fd/code-apps/{slug}/page``) — returns
   the app's ``frontend.html`` to be embedded in a sandboxed iframe.

3. **Backend proxy** (``ANY /fd/code-apps/{slug}/api/{tail:path}``) —
   forwards every request to the per-app subprocess managed by
   :mod:`app_runtime`.

Authentication: every route requires a logged-in Flight Deck user via
``get_current_user``. Inside the iframe the proxied requests carry
the same auth cookie, so the backend subprocess inherits the user
context for free.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from captain_claw.flight_deck import app_runtime
from captain_claw.flight_deck.agent_secret import get_or_create_agent_secret
from captain_claw.flight_deck.app_runtime import AppRuntimeError
from captain_claw.flight_deck.auth import decode_access_token


log = logging.getLogger(__name__)


router = APIRouter(prefix="/fd/code-apps", tags=["code-apps"])


# Name of the per-app auth cookie. Scoped to ``/fd/code-apps/<slug>/``
# so the iframe and its sub-calls see it but it doesn't leak elsewhere.
_APP_COOKIE_NAME = "fd_app_token"


# Header an agent-process tool sends when calling these routes
# server-to-server. The value must match the FD-side env var
# ``FD_AGENT_SHARED_SECRET``. Lets the FD process tell "real user
# coming in through the UI" apart from "agent in another process
# calling on the user's behalf" without minting JWTs for agents.
_AGENT_SECRET_HEADER = "x-fd-agent-secret"

# Header that a shared-secret caller uses to identify *what* it is:
#
# - ``app:<slug>``  — one code-app calling another. The target must
#   declare a non-empty ``data_api`` block in its manifest, otherwise
#   the proxy returns 403. This is the cross-app discovery gate.
# - ``chat`` (or absent) — the user's chat agent. No gating; the
#   chat agent is acting on the user's behalf and the user already
#   has full access to all of their apps.
#
# Both arrive over the same shared-secret channel so this header is
# self-asserted, not verified — see the SECURITY note in
# :func:`_get_code_app_user` and ``app_sdk.py``. Good enough for
# single-user / single-trust-domain v1.
_AGENT_AS_HEADER = "x-fd-agent-as"


def _fd_auth_enabled() -> bool:
    return os.environ.get("FD_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")


def _agent_shared_secret() -> str:
    """Return the auto-managed shared secret for agent → FD calls.

    Resolved via :func:`get_or_create_agent_secret` so FD and any
    agent on the same host converge on the same value without the
    operator setting env vars.
    """
    return get_or_create_agent_secret()


async def _get_code_app_user(request: Request) -> dict:
    """Auth dependency tailored for code-app routes.

    Accepts authentication from any of these sources (priority order):

    0. ``FD_AUTH_ENABLED=false`` (server-side env) — all FD auth is
       disabled in this deployment, so we return a synthetic user
       just like the rest of FD does via ``_no_user``. Keeps this
       dep consistent with the shared deps in :mod:`server.py`.
    1. ``X-FD-Agent-Secret`` header matching the server's
       ``FD_AGENT_SHARED_SECRET`` env var — used by the
       :class:`AppRunnerTool` (agent-side) to call these routes
       server-to-server, on the user's behalf, without minting a
       JWT for the agent process. Constant-time compare.
    2. ``Authorization: Bearer …`` header — used by the FD UI's
       ``/fd/code-apps`` management calls (list, restart, logs).
    3. ``?fd_token=…`` query string — used when loading the iframe
       page URL directly.
    4. ``fd_app_token`` cookie — set by the ``/page`` handler when
       it serves an app's HTML. The iframe's relative ``./api/*``
       calls carry this cookie back automatically, even though the
       agent-written ``frontend.html`` doesn't know anything about
       auth.

    Why a separate dep instead of reusing :func:`get_current_user`?
    The shared dep doesn't read cookies, and we don't want to add
    a cookie path to it system-wide — code-app proxying is the
    only place where the *iframe's own JS* needs to authenticate
    without being able to set request headers. The agent-secret
    path is also code-app-specific: only this surface is hit by
    agent processes calling FD.
    """
    # 0. Global FD auth off → synthetic user, matches FD shared deps.
    if not _fd_auth_enabled():
        request.state.user_id = "anonymous"
        request.state.fd_token = ""
        request.state.agent_as = ""
        return {"id": "anonymous"}

    # 1. Server-to-server: shared secret from agent process.
    agent_secret = _agent_shared_secret()
    header_secret = request.headers.get(_AGENT_SECRET_HEADER) or request.headers.get(
        _AGENT_SECRET_HEADER.title()
    ) or ""
    if agent_secret and header_secret and hmac.compare_digest(
        agent_secret, header_secret
    ):
        # Agents act on behalf of *some* user; the agent should pass
        # the user id in ``X-FD-Agent-User`` so logs / data scoping
        # stay correct. Falls back to a stable sentinel if missing.
        on_behalf_of = (
            request.headers.get("x-fd-agent-user")
            or request.headers.get("X-Fd-Agent-User")
            or "agent"
        )
        request.state.user_id = on_behalf_of
        # No JWT to re-issue as a cookie — the iframe path would
        # never reach here (the iframe uses cookie auth).
        request.state.fd_token = ""
        # Stash who the caller claims to be (app:<slug> vs chat).
        # Route-level handlers (esp. proxy_api) read this to decide
        # whether to enforce the cross-app data_api gate.
        request.state.agent_as = (
            request.headers.get(_AGENT_AS_HEADER)
            or request.headers.get(_AGENT_AS_HEADER.title())
            or ""
        )
        return {"id": on_behalf_of}

    # 2-4. JWT paths (header / query / cookie).
    token_str: str | None = None
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        token_str = auth.split(None, 1)[1].strip()
    if not token_str:
        token_str = request.query_params.get("fd_token")
    if not token_str:
        token_str = request.cookies.get(_APP_COOKIE_NAME)
    if not token_str:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_access_token(token_str)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    request.state.user_id = user_id
    request.state.fd_token = token_str
    # JWT path = real user via UI. No agent-as tag.
    request.state.agent_as = ""
    return {"id": user_id}


# ── helpers ───────────────────────────────────────────────────────────


def _require_app(slug: str) -> None:
    """404 if the slug has no on-disk directory yet."""
    if app_runtime.read_app_manifest(slug) is None:
        # Allow apps that have a backend.py but no manifest yet (during
        # authoring iteration) — treat the directory's existence as
        # ground truth and let proxy() complain if backend.py is
        # missing.
        if not (app_runtime.app_dir(slug) / "backend.py").exists():
            raise HTTPException(status_code=404, detail=f"No code-app '{slug}'")


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ── discovery / lifecycle ─────────────────────────────────────────────


@router.get("")
async def list_apps(_user: dict = Depends(_get_code_app_user)) -> dict[str, Any]:
    """Return every code-app on disk with a quick liveness summary."""
    running = {p["slug"]: p for p in app_runtime.get_runtime().list_running()}
    out: list[dict[str, Any]] = []
    for entry in app_runtime.list_code_apps():
        slug = entry["slug"]
        live = running.get(slug)
        out.append({
            **entry,
            "running": live is not None,
            "pid": live["pid"] if live else None,
            "idle_seconds": live["idle_seconds"] if live else None,
            "has_error": bool(live and live["has_error"]),
        })
    return {"apps": out}


@router.get("/{slug}")
async def get_app(slug: str, _user: dict = Depends(_get_code_app_user)) -> dict[str, Any]:
    """Return the manifest + file presence summary for one code-app."""
    _require_app(slug)
    manifest = app_runtime.read_app_manifest(slug) or {}
    d = app_runtime.app_dir(slug)
    return {
        "slug": slug,
        "manifest": manifest,
        "has_backend": (d / "backend.py").exists(),
        "has_frontend": (d / "frontend.html").exists(),
    }


@router.post("/{slug}/restart")
async def restart_app(slug: str, _user: dict = Depends(_get_code_app_user)) -> dict[str, Any]:
    """Stop the subprocess (if running) and spawn fresh.

    Used by the self-repair loop after the agent rewrites ``backend.py``.
    """
    _require_app(slug)
    runtime = app_runtime.get_runtime()
    try:
        proc = await runtime.restart(slug)
    except AppRuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "pid": proc.pid, "started_at": proc.started_at}


@router.post("/{slug}/stop")
async def stop_app(slug: str, _user: dict = Depends(_get_code_app_user)) -> dict[str, Any]:
    """Stop the subprocess if running. Idempotent."""
    _require_app(slug)
    killed = await app_runtime.get_runtime().stop(slug)
    return {"ok": True, "was_running": killed}


@router.get("/{slug}/logs")
async def tail_logs(
    slug: str,
    n: int = 200,
    _user: dict = Depends(_get_code_app_user),
) -> dict[str, Any]:
    """Return the last ``n`` stderr/stdout lines + last_error.

    This endpoint is the self-repair surface: after a 5xx from the
    proxy, the agent calls here to read the traceback and decide what
    to patch.
    """
    _require_app(slug)
    n = max(1, min(int(n), 2000))
    return app_runtime.get_runtime().tail_logs(slug, n=n)


@router.delete("/{slug}")
async def delete_app(slug: str, _user: dict = Depends(_get_code_app_user)) -> dict[str, Any]:
    """Stop the subprocess and remove the app's directory.

    Soft-irreversible: data records under the slug's entity namespace
    are *not* deleted here; the agent must clean those up explicitly
    if desired. That asymmetry is deliberate — code can be rewritten,
    data can't.
    """
    _require_app(slug)
    runtime = app_runtime.get_runtime()
    await runtime.stop(slug)
    import shutil
    target = app_runtime.app_dir(slug)
    try:
        shutil.rmtree(target)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"rmtree failed: {exc}")
    return {"ok": True}


# ── scaffolding (a thin convenience for the agent) ────────────────────


@router.post("/{slug}/scaffold")
async def scaffold_app(
    slug: str,
    payload: dict = Body(...),
    user: dict = Depends(_get_code_app_user),
) -> dict[str, Any]:
    """Create or overwrite a code-app's files in one call.

    Body::

        {
          "name":      str,                      # display name
          "version":   str,            optional, default "0.1.0"
          "backend":   str,                      # backend.py contents
          "frontend":  str,                      # frontend.html contents
        }

    This is the *write-once* helper. For iterative edits the agent uses
    its normal ``write`` / ``edit`` tools against the on-disk paths.
    """
    name = str(payload.get("name") or "").strip()
    backend = payload.get("backend")
    frontend = payload.get("frontend")
    version = str(payload.get("version") or "0.1.0").strip() or "0.1.0"
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if not isinstance(backend, str) or not backend.strip():
        raise HTTPException(status_code=400, detail="backend (Python source) is required")
    if not isinstance(frontend, str) or not frontend.strip():
        raise HTTPException(status_code=400, detail="frontend (HTML source) is required")

    try:
        d = app_runtime.app_dir(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    (d / "backend.py").write_text(backend, encoding="utf-8")
    (d / "frontend.html").write_text(frontend, encoding="utf-8")
    manifest = app_runtime.read_app_manifest(slug) or {}
    manifest.update({
        "name": name,
        "version": version,
        "slug": slug,
        "updated_at": _now_iso(),
        "updated_by": str(user.get("id") or user.get("email") or ""),
    })
    manifest.setdefault("created_at", _now_iso())
    app_runtime.write_app_manifest(slug, manifest)

    # If the app was already running with the old code, restart so the
    # next request picks up the fresh backend.py.
    runtime = app_runtime.get_runtime()
    if any(p["slug"] == slug for p in runtime.list_running()):
        try:
            await runtime.restart(slug)
        except AppRuntimeError as exc:
            log.warning("Scaffold restart failed for %s: %s", slug, exc)
    return {"ok": True, "slug": slug, "manifest": manifest}


# ── source read + surgical edit ───────────────────────────────────────


# Files an agent is allowed to read / edit through these routes.
# ``manifest.json`` is intentionally NOT here: it's controlled
# server-side via scaffold's name/version fields. Letting an agent
# patch arbitrary manifest bytes invites slug drift and version
# tampering for no upside.
_EDITABLE_FILES: dict[str, str] = {
    "backend": "backend.py",
    "frontend": "frontend.html",
}


@router.get("/{slug}/source")
async def get_source(
    slug: str,
    _user: dict = Depends(_get_code_app_user),
) -> dict[str, Any]:
    """Return the current source of a code-app.

    Lets a cold agent (no prior context) load an app's actual code
    before editing — without this, the only way to modify an
    existing app was a full ``scaffold`` rewrite from scratch.

    Returns ``backend`` and ``frontend`` as raw strings (empty
    string if the file is absent), plus the manifest. The response
    is structured rather than concatenated so the tool layer can
    surface whichever pieces the LLM needs.
    """
    _require_app(slug)
    d = app_runtime.app_dir(slug)
    backend_p = d / "backend.py"
    frontend_p = d / "frontend.html"
    try:
        backend = backend_p.read_text(encoding="utf-8") if backend_p.exists() else ""
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"read backend.py: {exc}")
    try:
        frontend = frontend_p.read_text(encoding="utf-8") if frontend_p.exists() else ""
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"read frontend.html: {exc}")
    return {
        "slug": slug,
        "backend": backend,
        "frontend": frontend,
        "manifest": app_runtime.read_app_manifest(slug) or {},
    }


@router.post("/{slug}/edit")
async def edit_file(
    slug: str,
    payload: dict = Body(...),
    user: dict = Depends(_get_code_app_user),
) -> dict[str, Any]:
    """Surgically edit one file of a code-app.

    Two modes, mutually exclusive:

    1. **Replace one occurrence** of ``old`` with ``new`` (preferred
       for small targeted changes — fast and audit-friendly):

           {"file": "backend", "old": "...", "new": "..."}

       Fails with HTTP 400 if ``old`` is missing, appears zero
       times, or appears more than once — mirrors the safety of
       the standard ``edit`` tool and protects against silent
       corruption when the agent guesses a non-unique snippet.

    2. **Replace the whole file** with ``content`` (use this for
       large rewrites of one file when ``scaffold`` would be
       overkill because the other file shouldn't change):

           {"file": "frontend", "content": "..."}

    Both modes auto-restart the subprocess if it's running, so the
    next request sees the new code immediately.
    """
    file_key = str(payload.get("file") or "").strip().lower()
    if file_key not in _EDITABLE_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"'file' must be one of {sorted(_EDITABLE_FILES)}",
        )
    filename = _EDITABLE_FILES[file_key]
    target = app_runtime.app_dir(slug) / filename

    # Decide mode: full-content replace vs. surgical old→new.
    content_full = payload.get("content")
    old = payload.get("old")
    new = payload.get("new")

    if isinstance(content_full, str):
        if old is not None or new is not None:
            raise HTTPException(
                status_code=400,
                detail="provide either 'content' OR ('old' + 'new'), not both",
            )
        if not content_full.strip():
            raise HTTPException(
                status_code=400,
                detail=f"'content' is empty — use scaffold to delete files",
            )
        target.write_text(content_full, encoding="utf-8")
        change_summary = f"replaced entire {filename} ({len(content_full)} chars)"
    else:
        if not isinstance(old, str) or not old:
            raise HTTPException(
                status_code=400,
                detail="'old' (non-empty string) is required for surgical edit",
            )
        if not isinstance(new, str):
            raise HTTPException(
                status_code=400,
                detail="'new' (string) is required for surgical edit",
            )
        if not target.exists():
            raise HTTPException(
                status_code=404,
                detail=f"{filename} doesn't exist — use scaffold to create it",
            )
        try:
            current = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"read {filename}: {exc}")
        count = current.count(old)
        if count == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'old' not found in {filename}. Call action='read_source' "
                    "to see the current contents before editing."
                ),
            )
        if count > 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'old' appears {count} times in {filename} — must be "
                    "unique. Include more surrounding context to disambiguate."
                ),
            )
        updated = current.replace(old, new, 1)
        target.write_text(updated, encoding="utf-8")
        change_summary = (
            f"edited {filename}: 1 occurrence replaced "
            f"({len(old)} → {len(new)} chars)"
        )

    # Bump manifest's updated_at so listings reflect the change.
    manifest = app_runtime.read_app_manifest(slug) or {}
    manifest["updated_at"] = _now_iso()
    manifest["updated_by"] = str(user.get("id") or user.get("email") or "")
    app_runtime.write_app_manifest(slug, manifest)

    # Auto-restart if running, same contract as scaffold.
    runtime = app_runtime.get_runtime()
    restarted = False
    if any(p["slug"] == slug for p in runtime.list_running()):
        try:
            await runtime.restart(slug)
            restarted = True
        except AppRuntimeError as exc:
            log.warning("Edit restart failed for %s: %s", slug, exc)

    return {
        "ok": True,
        "slug": slug,
        "file": filename,
        "change": change_summary,
        "restarted": restarted,
    }


# ── frontend serving ──────────────────────────────────────────────────


@router.get("/{slug}/page", response_class=HTMLResponse)
async def get_frontend(
    slug: str,
    request: Request,
    _user: dict = Depends(_get_code_app_user),
) -> HTMLResponse:
    """Return the app's ``frontend.html`` for embedding in an iframe.

    The HTML is served as-is — no template substitution, no script
    injection. The iframe sandbox on the FD side is the security
    boundary.

    Auth side effect: we set a path-scoped ``fd_app_token`` cookie so
    the iframe's ``./api/*`` calls inherit the user's session without
    the agent-written frontend.html needing to know anything about
    auth. Cookie path is ``/fd/code-apps/<slug>/`` so it only flows
    back to the same app's proxy.
    """
    _require_app(slug)
    html = app_runtime.read_frontend_html(slug)
    if html is None:
        # Return a stub page rather than 404 so the iframe shows a
        # helpful message instead of a broken-image-style error.
        html = _placeholder_page(slug)
    # CSP hardening: deny everything except same-origin so a stray
    # agent-written <script src="https://evil.example"> can't reach
    # outbound. Inline scripts ARE allowed because that's the whole
    # point of a self-contained HTML bundle.
    headers = {
        "Content-Security-Policy": (
            "default-src 'self' data: blob:; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com https://cdn.jsdelivr.net https://esm.sh; "
            "style-src 'self' 'unsafe-inline' https://unpkg.com https://cdn.jsdelivr.net; "
            "img-src 'self' data: blob: https:; "
            "connect-src 'self'; "
            "frame-ancestors 'self';"
        ),
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-store",
    }
    response = HTMLResponse(content=html, headers=headers)
    token = getattr(request.state, "fd_token", "") or ""
    if token:
        response.set_cookie(
            key=_APP_COOKIE_NAME,
            value=token,
            # Scope tightly to this app's URL space so the cookie
            # doesn't leak to other FD routes.
            path=f"/fd/code-apps/{slug}/",
            httponly=True,
            samesite="strict",
            # Short-lived: matches the typical access-token lifetime.
            # The FD UI re-renders the iframe periodically anyway.
            max_age=60 * 60,
        )
    return response


def _placeholder_page(slug: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{slug} — no frontend yet</title>"
        "<style>body{font-family:system-ui,sans-serif;color:#666;padding:32px;}</style>"
        "</head><body>"
        f"<h2>App <code>{slug}</code> has no <code>frontend.html</code> yet.</h2>"
        "<p>Ask the agent to scaffold one, or write it directly.</p>"
        "</body></html>"
    )


# ── backend proxy ─────────────────────────────────────────────────────


# Methods we forward to the subprocess. WebSockets are intentionally
# NOT supported in v1 — keep the surface small until we have a real
# use case.
_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]


@router.api_route("/{slug}/api/{tail:path}", methods=_PROXY_METHODS)
async def proxy_api(
    slug: str,
    tail: str,
    request: Request,
    _user: dict = Depends(_get_code_app_user),
) -> Response:
    """Forward a request to the app's subprocess.

    The subprocess receives the path *relative to ``/api/``* with a
    leading slash, so a request to ``/fd/code-apps/notes/api/items``
    arrives at the user's ``handle()`` as path ``"/items"``.

    **Cross-app gate.** If the caller authenticated via the shared
    agent secret AND claims to be another code-app
    (``X-FD-Agent-As: app:<other_slug>``), the target slug must
    declare a non-empty ``data_api`` block in its manifest. Apps
    without that opt-in are private to their own frontend and to
    the user's chat agent. Returns 403 otherwise so the failure is
    visible at call time, not silently empty.
    """
    _require_app(slug)

    # Enforce the cross-app data_api opt-in before doing any work.
    agent_as = getattr(request.state, "agent_as", "") or ""
    if agent_as.startswith("app:"):
        caller_slug = agent_as.split(":", 1)[1].strip()
        # Self-calls are pointless but harmless — allow them so an app
        # can be refactored to use the SDK against itself without
        # special-casing the loopback path.
        if caller_slug != slug:
            target_manifest = app_runtime.read_app_manifest(slug) or {}
            data_api = target_manifest.get("data_api") or {}
            if not isinstance(data_api, dict) or not data_api:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"App '{slug}' does not expose a data_api — "
                        f"app '{caller_slug}' cannot read from it. "
                        "To allow cross-app reads, add a 'data_api' "
                        "object to the target's manifest.json listing "
                        "the endpoints other apps may call."
                    ),
                )

    body = await request.body()
    headers = {k: v for k, v in request.headers.items()}
    method = request.method
    # Pass the user-visible part of the path through. Backend writers
    # think in terms of "/items", not "/fd/code-apps/notes/api/items".
    sub_path = "/" + tail.lstrip("/") if tail else "/"
    query_string = request.url.query

    started = time.time()
    try:
        status, out_headers, out_body = await app_runtime.get_runtime().proxy(
            slug,
            method=method,
            path=sub_path,
            headers=headers,
            body=body,
            query_string=query_string,
        )
    except AppRuntimeError as exc:
        # Surface the subprocess error to the client (typically the
        # iframe) so the user sees something coherent, and keep the
        # full traceback in logs/last_error for the self-repair loop.
        msg = {
            "error": "app_backend_unavailable",
            "detail": str(exc),
        }
        return Response(
            content=json.dumps(msg),
            status_code=502,
            media_type="application/json",
        )
    finally:
        log.debug(
            "Proxied %s /fd/code-apps/%s%s -> %dms",
            method, slug, sub_path, int((time.time() - started) * 1000),
        )

    # Strip any Content-Length the subprocess set — FastAPI/Starlette
    # recomputes it from the body anyway, and mismatched lengths break
    # the response.
    out_headers.pop("Content-Length", None)
    out_headers.pop("content-length", None)
    media_type = out_headers.pop("Content-Type", None) or out_headers.pop("content-type", None)
    return Response(
        content=out_body,
        status_code=status,
        headers=out_headers,
        media_type=media_type or "application/octet-stream",
    )
