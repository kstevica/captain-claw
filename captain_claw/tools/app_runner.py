"""Agent-facing tool for authoring and operating code-apps.

A "code-app" is an agent-authored ``backend.py`` + ``frontend.html``
pair stored under ``~/.captain-claw-fd/apps/<slug>/`` and served by
the per-app subprocess runtime in
``captain_claw.flight_deck.app_runtime``.

This tool is how an agent's planning loop interacts with that
runtime. It thinly wraps the ``/fd/code-apps`` HTTP routes so the
same machinery the FD UI uses (scaffold, restart, logs, proxy) is
also reachable from a CC agent's tool-call shell.

Actions
-------
- ``list``      — enumerate all code-apps with liveness summary.
- ``get``       — manifest + file presence for one slug.
- ``scaffold``  — write ``backend.py`` + ``frontend.html`` for a slug
                  (creates or overwrites; auto-restarts if running).
- ``restart``   — stop + respawn the subprocess (used after edits).
- ``stop``      — terminate the subprocess (idempotent).
- ``logs``      — last N stderr / stdout lines + ``last_error``. This
                  is the **self-repair surface**: after a 5xx from a
                  smoke-test, call this to read the traceback.
- ``delete``    — remove the app directory (data records preserved).
- ``proxy``     — issue any HTTP method against the app's backend via
                  the FD proxy. The smoke-test step uses this.

Auth
----
The runtime routes require a logged-in Flight Deck user. Callers
supply credentials via (in priority order):

1. The session-metadata key ``fd_token`` (set by FD when it spawns
   the agent).
2. The agent attribute ``_fd_token``.
3. The ``FD_TOKEN`` environment variable.

If none of those are set and FD has auth enabled, every request
will 401. With ``FD_AUTH_ENABLED=false`` no token is needed.
"""

from __future__ import annotations

import json
import os
from typing import Any

import structlog

from captain_claw.tools.registry import Tool, ToolResult


log = structlog.get_logger(__name__)


_TIMEOUT_SECONDS = 60.0
_PROXY_TIMEOUT_SECONDS = 30.0


def _resolve_fd_url(**kwargs: Any) -> str:
    """Resolve the FD URL from session metadata, agent attrs, or env."""
    session = kwargs.get("_session")
    agent = kwargs.get("_agent")
    metadata = getattr(session, "metadata", {}) or {} if session else {}
    fd_url = metadata.get("fd_url", "")
    if not fd_url and agent:
        fd_url = getattr(agent, "_fd_url", "") or ""
    if not fd_url:
        fd_url = os.environ.get("FD_URL", "") or os.environ.get("FD_INTERNAL_URL", "")
    return str(fd_url or "").rstrip("/")


def _resolve_fd_token(**kwargs: Any) -> str:
    """Resolve the FD bearer token. Empty string if none available."""
    session = kwargs.get("_session")
    agent = kwargs.get("_agent")
    metadata = getattr(session, "metadata", {}) or {} if session else {}
    token = metadata.get("fd_token", "")
    if not token and agent:
        token = getattr(agent, "_fd_token", "") or ""
    if not token:
        token = os.environ.get("FD_TOKEN", "")
    return str(token or "")


def _resolve_agent_secret() -> str:
    """Return the shared secret used for agent → FD calls.

    Agents don't hold user JWTs; FD treats this header value as a
    service credential. The secret is auto-managed by
    :mod:`captain_claw.flight_deck.agent_secret`: same file under
    ``~/.captain-claw-fd/agent_secret`` that the FD process reads.
    No env-var setup is required on a single-host setup.

    Import is deferred so an agent process that runs without the
    ``flight_deck`` package on its path still loads this tool;
    the call would just return an empty string and surface a clear
    error at request time.
    """
    try:
        from captain_claw.flight_deck.agent_secret import (
            get_or_create_agent_secret,
        )
    except Exception:
        return os.environ.get("FD_AGENT_SHARED_SECRET", "") or ""
    return get_or_create_agent_secret()


def _resolve_agent_user(**kwargs: Any) -> str:
    """Best effort: identify the user whose agent we're acting for.

    Sent alongside the shared secret as ``X-FD-Agent-User`` so FD
    can attribute the call to a real account. Falls back to empty
    (FD will record the generic ``"agent"`` sentinel).
    """
    session = kwargs.get("_session")
    agent = kwargs.get("_agent")
    metadata = getattr(session, "metadata", {}) or {} if session else {}
    uid = metadata.get("user_id") or metadata.get("owner_id") or ""
    if not uid and agent:
        uid = (
            getattr(agent, "_user_id", "")
            or getattr(agent, "_active_personality_id", "")
            or ""
        )
    return str(uid or "")


def _auth_headers(token: str, **kwargs: Any) -> dict[str, str]:
    """Build the auth headers for an agent → FD call.

    Two complementary paths:

    - ``Authorization: Bearer <token>`` if we somehow have a JWT
      (rare in practice for agent-side callers — most agents don't
      hold user tokens, so this stays empty).
    - ``X-FD-Agent-Secret`` + ``X-FD-Agent-User`` when the shared
      secret is configured. This is the normal agent-side auth.

    If neither is available we send nothing and FD returns 401 —
    the tool surfaces that with a clear hint.
    """
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    secret = _resolve_agent_secret()
    if secret:
        headers["X-FD-Agent-Secret"] = secret
        user = _resolve_agent_user(**kwargs)
        if user:
            headers["X-FD-Agent-User"] = user
    return headers


class AppRunnerTool(Tool):
    """Author / inspect / control agent-coded apps under Flight Deck."""

    name = "app_runner"
    description = (
        "PREFERRED tool for any user request to 'build / make / create / "
        "scaffold an app' — notes app, todo app, trackers, kanban boards, "
        "mini-CRMs, bookmark managers, etc. Anything with persistent state "
        "+ multiple screens + interactive CRUD belongs here. Do NOT write "
        "a standalone *.html file with localStorage via the ``write`` tool "
        "for these requests; that hides the app from Flight Deck's Code "
        "Apps page and loses data across devices.\n\n"
        "A code-app is a Python backend (``backend.py``) + self-contained "
        "HTML frontend (``frontend.html``) that runs in a sandboxed "
        "subprocess and is embedded in the FD UI as an iframe. The "
        "backend gets a shared datastore for free, so apps share data "
        "across devices and survive restarts. Actions: ``scaffold`` "
        "(write backend + frontend from scratch), ``read_source`` "
        "(load current backend + frontend so you can modify them), "
        "``edit_file`` (surgically replace one snippet in one file, "
        "or replace one whole file), ``restart`` (respawn after edits), "
        "``logs`` (tail stderr/stdout + last_error), ``proxy`` "
        "(smoke-test by issuing HTTP against the backend), "
        "``list``/``get``/``stop``/``delete``.\n\n"
        "MODIFYING AN EXISTING APP: do NOT call ``scaffold`` to change "
        "an app that already exists — that wipes the whole source "
        "and loses any details you don't reconstruct. Instead: "
        "(1) call ``read_source`` to load the current backend + "
        "frontend, (2) call ``edit_file`` for each targeted change "
        "(it auto-restarts the subprocess), (3) optionally smoke-test "
        "with ``proxy``. Only fall back to ``scaffold`` if the change "
        "is a full rewrite of both files.\n\n"
        "READING ANOTHER APP'S DATA: each app can publish a public "
        "data API by adding a ``data_api`` block to its "
        "``manifest.json`` (endpoint name → {path, method, "
        "description}). To read across apps from chat, call "
        "``list`` to see which apps expose what, then ``query_app`` "
        "to hit a specific endpoint. Apps without a ``data_api`` "
        "block are private — chat can still read them via "
        "``query_app`` (the chat agent is treated as the user), but "
        "OTHER apps calling sibling() will get a 403 until you add "
        "the data_api block. When scaffolding apps you expect to be "
        "queried by siblings, include a data_api in the manifest.\n\n"
        "Self-repair loop: after scaffold/edit, call ``proxy`` to "
        "smoke-test. On 5xx, call ``logs`` to read the subprocess "
        "traceback, fix with ``edit_file`` or ``scaffold``, then "
        "``restart`` (if not already auto-restarted) and retry. "
        "Iterate until the smoke test passes — do NOT fall back to a "
        "localStorage SPA."
    )
    timeout_seconds = _TIMEOUT_SECONDS

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list", "get", "read_source", "scaffold", "edit_file",
                    "restart", "stop", "logs", "delete", "proxy", "query_app",
                ],
                "description": (
                    "What to do. 'scaffold' writes backend.py + "
                    "frontend.html from scratch (use only for NEW apps or "
                    "full rewrites). 'read_source' returns current backend "
                    "+ frontend source (use before editing an existing app "
                    "so you don't lose detail). 'edit_file' surgically "
                    "patches one file (auto-restarts). 'restart' respawns "
                    "the subprocess. 'logs' tails subprocess stderr/stdout "
                    "+ last_error (call this when a request returned 5xx). "
                    "'proxy' issues any HTTP method against the app backend "
                    "(use for smoke tests of an app you just authored). "
                    "'query_app' reads data from another app's published "
                    "data_api on the user's behalf (use this when chat "
                    "needs facts FROM an app, e.g. 'how many notes do I "
                    "have' or 'list contacts whose email is gmail')."
                ),
            },
            "slug": {
                "type": "string",
                "description": (
                    "App slug — required for everything except 'list'. "
                    "Lowercase letters/digits/underscores/dashes, leading "
                    "letter, ≤48 chars. Example: 'notes_demo'."
                ),
            },
            "name": {
                "type": "string",
                "description": "Display name (scaffold only).",
            },
            "version": {
                "type": "string",
                "description": "Version string (scaffold only, defaults to '0.1.0').",
            },
            "backend": {
                "type": "string",
                "description": (
                    "Full source of backend.py (scaffold only). Must define "
                    "``async def handle(method, path, headers, body) -> "
                    "{status, headers, body}``."
                ),
            },
            "frontend": {
                "type": "string",
                "description": (
                    "Full source of frontend.html (scaffold only). Self-"
                    "contained HTML + inline JS. API calls use relative "
                    "paths under './api/'."
                ),
            },
            "n": {
                "type": "integer",
                "description": "Number of log lines to tail (logs only, default 200).",
            },
            "method": {
                "type": "string",
                "description": (
                    "HTTP method for 'proxy' / 'query_app': GET / POST / "
                    "PUT / PATCH / DELETE / OPTIONS / HEAD. Defaults to "
                    "GET (or the value declared in the target app's "
                    "data_api when using 'query_app' with 'endpoint')."
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "Path under the app backend for 'proxy' / 'query_app'. "
                    "The leading slash is optional; '/items' and 'items' "
                    "both arrive at the backend as '/items'. For "
                    "'query_app' this is optional if you pass 'endpoint' "
                    "instead."
                ),
            },
            "endpoint": {
                "type": "string",
                "description": (
                    "(query_app only) Named endpoint from the target "
                    "app's manifest.json 'data_api' block. Use this "
                    "instead of 'path' when you saw the endpoint name "
                    "in the 'list' output — saves you from memorising "
                    "URLs. Example: endpoint='list' or 'search'."
                ),
            },
            "body": {
                "type": "string",
                "description": (
                    "Request body (string) for 'proxy' / 'query_app'. "
                    "JSON callers should pre-serialize and set "
                    "'Content-Type: application/json' in 'headers'."
                ),
            },
            "headers": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": (
                    "Extra request headers for 'proxy' / 'query_app'."
                ),
            },
            "file": {
                "type": "string",
                "enum": ["backend", "frontend"],
                "description": (
                    "Which file to edit (edit_file only). 'backend' = "
                    "backend.py, 'frontend' = frontend.html."
                ),
            },
            "old_string": {
                "type": "string",
                "description": (
                    "Exact snippet to find and replace (edit_file only, "
                    "surgical mode). Must appear EXACTLY ONCE in the file "
                    "— include enough surrounding context to be unique. "
                    "Mutually exclusive with 'content'."
                ),
            },
            "new_string": {
                "type": "string",
                "description": (
                    "Replacement text for 'old_string' (edit_file only, "
                    "surgical mode). May be empty to delete the snippet."
                ),
            },
            "content": {
                "type": "string",
                "description": (
                    "Full replacement contents for the chosen file "
                    "(edit_file only, full-file mode). Mutually exclusive "
                    "with 'old_string'/'new_string'. Use this only when "
                    "rewriting one file but leaving the other alone — if "
                    "both files change, call 'scaffold' instead."
                ),
            },
        },
        "required": ["action"],
    }

    # ── dispatch ─────────────────────────────────────────────────────

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action") or "").strip().lower()
        fd_url = _resolve_fd_url(**kwargs)
        if not fd_url:
            return ToolResult(
                success=False,
                error=(
                    "Flight Deck URL not configured. Set FD_URL env, "
                    "or session metadata 'fd_url', or agent._fd_url."
                ),
            )
        token = _resolve_fd_token(**kwargs)

        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, error="httpx is required")

        slug = str(kwargs.get("slug") or "").strip()
        # Validate slug presence early for actions that need it.
        if action != "list" and not slug:
            return ToolResult(
                success=False,
                error=f"'slug' is required for action '{action}'",
            )

        if action == "list":
            return await self._list(httpx, fd_url, token)
        if action == "get":
            return await self._get(httpx, fd_url, token, slug)
        if action == "read_source":
            return await self._read_source(httpx, fd_url, token, slug)
        if action == "scaffold":
            return await self._scaffold(httpx, fd_url, token, slug, kwargs)
        if action == "edit_file":
            return await self._edit_file(httpx, fd_url, token, slug, kwargs)
        if action == "restart":
            return await self._restart(httpx, fd_url, token, slug)
        if action == "stop":
            return await self._stop(httpx, fd_url, token, slug)
        if action == "logs":
            n = int(kwargs.get("n") or 200)
            return await self._logs(httpx, fd_url, token, slug, n)
        if action == "delete":
            return await self._delete(httpx, fd_url, token, slug)
        if action == "proxy":
            return await self._proxy(httpx, fd_url, token, slug, kwargs)
        if action == "query_app":
            return await self._query_app(httpx, fd_url, token, slug, kwargs)

        return ToolResult(
            success=False,
            error=f"Unknown action: {action!r}",
        )

    # ── action handlers ──────────────────────────────────────────────

    async def _list(self, httpx, fd_url: str, token: str) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.get(
                    f"{fd_url}/fd/code-apps", headers=_auth_headers(token),
                )
        except Exception as e:
            return ToolResult(success=False, error=f"Cannot reach FD: {e}")
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                error=f"FD returned HTTP {resp.status_code}: {resp.text[:200]}",
            )
        data = resp.json()
        apps = data.get("apps") or []
        if not apps:
            return ToolResult(success=True, content="No code-apps registered.")
        lines = [f"Code-apps ({len(apps)}):"]
        for a in apps:
            running = "●" if a.get("running") else "○"
            err = " [last_error]" if a.get("has_error") else ""
            backend = "" if a.get("has_backend") else " [no backend.py]"
            lines.append(
                f"- {running} `{a.get('slug')}` "
                f"v{a.get('version', '?')} — {a.get('name', '?')}"
                f"{err}{backend}"
            )
            # Surface any published data_api so the agent knows what
            # it can call via ``query_app`` without a second request.
            # Compact 1-line summary per endpoint to keep the listing
            # readable when many apps are registered.
            data_api = a.get("data_api") or {}
            if isinstance(data_api, dict) and data_api:
                for ep_name, ep in data_api.items():
                    if not isinstance(ep, dict):
                        continue
                    method = str(ep.get("method") or "GET").upper()
                    path = str(ep.get("path") or "")
                    desc = str(ep.get("description") or "").strip()
                    suffix = f" — {desc}" if desc else ""
                    lines.append(
                        f"    · data_api.{ep_name}: {method} {path}{suffix}"
                    )
        return ToolResult(success=True, content="\n".join(lines))

    async def _get(self, httpx, fd_url: str, token: str, slug: str) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.get(
                    f"{fd_url}/fd/code-apps/{slug}",
                    headers=_auth_headers(token),
                )
        except Exception as e:
            return ToolResult(success=False, error=f"Cannot reach FD: {e}")
        if resp.status_code == 404:
            return ToolResult(success=False, error=f"No code-app '{slug}'")
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                error=f"FD returned HTTP {resp.status_code}: {resp.text[:200]}",
            )
        return ToolResult(success=True, content=resp.text)

    async def _read_source(
        self, httpx, fd_url: str, token: str, slug: str,
    ) -> ToolResult:
        """Fetch the current backend.py + frontend.html for editing.

        The response is intentionally verbose — for an LLM about to
        modify the source, raw text with clear file headers is far
        easier to ground edits against than nested JSON.
        """
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.get(
                    f"{fd_url}/fd/code-apps/{slug}/source",
                    headers=_auth_headers(token),
                )
        except Exception as e:
            return ToolResult(success=False, error=f"Cannot reach FD: {e}")
        if resp.status_code == 404:
            return ToolResult(success=False, error=f"No code-app '{slug}'")
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                error=f"FD returned HTTP {resp.status_code}: {resp.text[:200]}",
            )
        data = resp.json()
        backend = data.get("backend") or ""
        frontend = data.get("frontend") or ""
        manifest = data.get("manifest") or {}
        name = manifest.get("name") or slug
        version = manifest.get("version") or "?"
        parts = [
            f"# Code-app source: {slug} ({name}, v{version})",
            "",
            "## backend.py",
            f"({len(backend)} chars)" if backend else "(empty)",
            "",
            "```python",
            backend or "# (no backend.py)",
            "```",
            "",
            "## frontend.html",
            f"({len(frontend)} chars)" if frontend else "(empty)",
            "",
            "```html",
            frontend or "<!-- (no frontend.html) -->",
            "```",
            "",
            "When making changes, prefer ``edit_file`` for targeted "
            "patches (auto-restarts). Only call ``scaffold`` again if "
            "both files are being rewritten end-to-end.",
        ]
        return ToolResult(success=True, content="\n".join(parts))

    async def _edit_file(
        self, httpx, fd_url: str, token: str, slug: str, kwargs: dict[str, Any],
    ) -> ToolResult:
        """Surgically edit one file (or replace it whole) and restart.

        Two modes mirror the server route:
          * ``old_string`` + ``new_string`` → exact-string replace
            (must match exactly once).
          * ``content`` → replace the entire file.
        """
        file_key = str(kwargs.get("file") or "").strip().lower()
        if file_key not in ("backend", "frontend"):
            return ToolResult(
                success=False,
                error="'file' must be 'backend' or 'frontend'",
            )
        old_str = kwargs.get("old_string")
        new_str = kwargs.get("new_string")
        content = kwargs.get("content")

        body: dict[str, Any] = {"file": file_key}
        if isinstance(content, str) and content:
            if old_str is not None or new_str is not None:
                return ToolResult(
                    success=False,
                    error=(
                        "Provide either 'content' OR "
                        "'old_string'+'new_string', not both."
                    ),
                )
            body["content"] = content
        else:
            if not isinstance(old_str, str) or not old_str:
                return ToolResult(
                    success=False,
                    error=(
                        "Surgical edit needs a non-empty 'old_string'. "
                        "Tip: call action='read_source' first to copy "
                        "the exact snippet you want to replace, with "
                        "enough surrounding context to be unique."
                    ),
                )
            if not isinstance(new_str, str):
                return ToolResult(
                    success=False,
                    error="'new_string' is required (may be empty to delete).",
                )
            body["old"] = old_str
            body["new"] = new_str

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    f"{fd_url}/fd/code-apps/{slug}/edit",
                    headers={**_auth_headers(token), "Content-Type": "application/json"},
                    content=json.dumps(body),
                )
        except Exception as e:
            return ToolResult(success=False, error=f"Cannot reach FD: {e}")
        if resp.status_code == 404:
            return ToolResult(success=False, error=f"No code-app '{slug}'")
        if resp.status_code == 400:
            # Edit-specific 4xx (snippet not unique / missing / etc.)
            # come back as JSON {"detail": "..."} — surface verbatim so
            # the LLM can fix its next attempt.
            try:
                detail = resp.json().get("detail") or resp.text
            except Exception:
                detail = resp.text
            return ToolResult(
                success=False,
                error=f"Edit rejected: {detail}",
            )
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                error=f"FD returned HTTP {resp.status_code}: {resp.text[:200]}",
            )
        data = resp.json()
        restarted = " (subprocess restarted)" if data.get("restarted") else ""
        return ToolResult(
            success=True,
            content=f"{data.get('change') or 'edited'}{restarted}",
        )

    async def _scaffold(
        self, httpx, fd_url: str, token: str, slug: str, kwargs: dict[str, Any],
    ) -> ToolResult:
        name = str(kwargs.get("name") or "").strip()
        backend = kwargs.get("backend")
        frontend = kwargs.get("frontend")
        version = str(kwargs.get("version") or "0.1.0").strip() or "0.1.0"
        if not name:
            return ToolResult(success=False, error="'name' is required for scaffold")
        if not isinstance(backend, str) or not backend.strip():
            return ToolResult(success=False, error="'backend' (Python source) is required")
        if not isinstance(frontend, str) or not frontend.strip():
            return ToolResult(success=False, error="'frontend' (HTML source) is required")

        payload = {
            "name": name,
            "version": version,
            "backend": backend,
            "frontend": frontend,
        }
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    f"{fd_url}/fd/code-apps/{slug}/scaffold",
                    json=payload,
                    headers=_auth_headers(token),
                )
        except Exception as e:
            return ToolResult(success=False, error=f"Cannot reach FD: {e}")
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                error=f"Scaffold failed HTTP {resp.status_code}: {resp.text[:400]}",
            )
        data = resp.json()
        return ToolResult(
            success=True,
            content=(
                f"Scaffolded '{slug}'. Manifest:\n"
                + json.dumps(data.get("manifest") or {}, indent=2)
            ),
        )

    async def _restart(self, httpx, fd_url: str, token: str, slug: str) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    f"{fd_url}/fd/code-apps/{slug}/restart",
                    headers=_auth_headers(token),
                )
        except Exception as e:
            return ToolResult(success=False, error=f"Cannot reach FD: {e}")
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                error=f"Restart failed HTTP {resp.status_code}: {resp.text[:400]}",
            )
        data = resp.json()
        return ToolResult(
            success=True,
            content=f"Restarted '{slug}' (pid {data.get('pid')}).",
        )

    async def _stop(self, httpx, fd_url: str, token: str, slug: str) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    f"{fd_url}/fd/code-apps/{slug}/stop",
                    headers=_auth_headers(token),
                )
        except Exception as e:
            return ToolResult(success=False, error=f"Cannot reach FD: {e}")
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                error=f"Stop failed HTTP {resp.status_code}: {resp.text[:400]}",
            )
        data = resp.json()
        was = "stopped" if data.get("was_running") else "was not running"
        return ToolResult(success=True, content=f"'{slug}' {was}.")

    async def _logs(
        self, httpx, fd_url: str, token: str, slug: str, n: int,
    ) -> ToolResult:
        n = max(1, min(int(n), 2000))
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.get(
                    f"{fd_url}/fd/code-apps/{slug}/logs",
                    params={"n": n},
                    headers=_auth_headers(token),
                )
        except Exception as e:
            return ToolResult(success=False, error=f"Cannot reach FD: {e}")
        if resp.status_code == 404:
            return ToolResult(success=False, error=f"No code-app '{slug}'")
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                error=f"Logs failed HTTP {resp.status_code}: {resp.text[:200]}",
            )
        data = resp.json()
        stderr = data.get("stderr") or []
        stdout = data.get("stdout") or []
        last_err = data.get("last_error") or ""
        parts: list[str] = []
        if last_err:
            parts.append(f"## last_error\n\n```\n{last_err}\n```")
        if stderr:
            parts.append(
                f"## stderr (last {len(stderr)})\n\n```\n"
                + "\n".join(stderr) + "\n```"
            )
        if stdout:
            parts.append(
                f"## stdout (last {len(stdout)})\n\n```\n"
                + "\n".join(stdout) + "\n```"
            )
        if not parts:
            parts.append("No logs yet.")
        return ToolResult(success=True, content="\n\n".join(parts))

    async def _delete(self, httpx, fd_url: str, token: str, slug: str) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.delete(
                    f"{fd_url}/fd/code-apps/{slug}",
                    headers=_auth_headers(token),
                )
        except Exception as e:
            return ToolResult(success=False, error=f"Cannot reach FD: {e}")
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                error=f"Delete failed HTTP {resp.status_code}: {resp.text[:400]}",
            )
        return ToolResult(success=True, content=f"Deleted '{slug}'.")

    async def _proxy(
        self, httpx, fd_url: str, token: str, slug: str, kwargs: dict[str, Any],
    ) -> ToolResult:
        method = str(kwargs.get("method") or "GET").strip().upper()
        if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"):
            return ToolResult(success=False, error=f"Unsupported method: {method}")
        path = str(kwargs.get("path") or "/")
        if not path.startswith("/"):
            path = "/" + path
        body = kwargs.get("body")
        extra_headers = kwargs.get("headers") or {}
        if not isinstance(extra_headers, dict):
            return ToolResult(success=False, error="'headers' must be an object")

        merged_headers = {
            str(k): str(v) for k, v in extra_headers.items()
            if k and v is not None
        }
        merged_headers.update(_auth_headers(token))

        url = f"{fd_url}/fd/code-apps/{slug}/api{path}"
        try:
            async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT_SECONDS) as client:
                resp = await client.request(
                    method,
                    url,
                    content=body.encode("utf-8") if isinstance(body, str) else body,
                    headers=merged_headers,
                )
        except Exception as e:
            return ToolResult(success=False, error=f"Proxy call failed: {e}")

        # Surface the response as a single block so the agent can read it.
        text = resp.text
        if len(text) > 8000:
            text = text[:8000] + f"\n... (truncated; full length {len(resp.text)})"
        # Treat 5xx as a failure so the planner's verify-step can branch
        # to the repair path. 4xx is *not* failure — the agent may have
        # intentionally sent a bad request to test error handling.
        is_failure = resp.status_code >= 500
        return ToolResult(
            success=not is_failure,
            content=(
                f"{method} {path} → HTTP {resp.status_code}\n\n"
                f"```\n{text}\n```"
            ),
            error=(
                f"Backend returned HTTP {resp.status_code} — "
                f"call action='logs' to inspect the traceback."
                if is_failure else None
            ),
        )

    async def _query_app(
        self, httpx, fd_url: str, token: str, slug: str, kwargs: dict[str, Any],
    ) -> ToolResult:
        """Read another app's data on the user's behalf.

        Functionally similar to ``proxy`` (same FD route, same auth),
        but framed for read-from-another-app rather than smoke-test-
        my-own. Two extra things on top of ``proxy``:

        1. Best-effort discovery: if ``path`` is omitted but the
           target app's manifest declares a ``data_api`` block with
           exactly one endpoint named via ``endpoint``, we resolve
           the path/method from there. Lets the agent say
           ``query_app slug=contacts endpoint=list`` without
           memorizing routes.

        2. Pretty-printed JSON in the result so the agent can ground
           directly on the data without re-parsing a raw blob.

        The chat agent calls FD without an ``X-FD-Agent-As`` tag, so
        the cross-app data_api gate doesn't apply — chat is treated
        like the user, who already owns every app.
        """
        method_in = kwargs.get("method")
        path_in = kwargs.get("path")
        endpoint = str(kwargs.get("endpoint") or "").strip()
        method = str(method_in or "GET").strip().upper()
        path = str(path_in or "").strip()

        # Endpoint lookup mode.
        if endpoint and not path:
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                    resp = await client.get(
                        f"{fd_url}/fd/code-apps/{slug}",
                        headers=_auth_headers(token),
                    )
            except Exception as e:
                return ToolResult(success=False, error=f"Cannot reach FD: {e}")
            if resp.status_code != 200:
                return ToolResult(
                    success=False,
                    error=(
                        f"Lookup of '{slug}' failed HTTP "
                        f"{resp.status_code}: {resp.text[:200]}"
                    ),
                )
            manifest = (resp.json() or {}).get("manifest") or {}
            data_api = manifest.get("data_api") or {}
            ep = data_api.get(endpoint) if isinstance(data_api, dict) else None
            if not isinstance(ep, dict):
                available = (
                    sorted(data_api.keys()) if isinstance(data_api, dict) else []
                )
                return ToolResult(
                    success=False,
                    error=(
                        f"App '{slug}' has no data_api endpoint "
                        f"named '{endpoint}'. "
                        + (
                            f"Available: {', '.join(available)}."
                            if available
                            else "It declares no data_api at all."
                        )
                    ),
                )
            path = str(ep.get("path") or "")
            if not method_in:
                method = str(ep.get("method") or "GET").upper()

        if not path:
            return ToolResult(
                success=False,
                error=(
                    "query_app needs either 'path' (e.g. '/contacts') or "
                    "'endpoint' (a name from the target app's data_api)."
                ),
            )
        if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"):
            return ToolResult(success=False, error=f"Unsupported method: {method}")
        if not path.startswith("/"):
            path = "/" + path

        body = kwargs.get("body")
        extra_headers = kwargs.get("headers") or {}
        if not isinstance(extra_headers, dict):
            return ToolResult(success=False, error="'headers' must be an object")
        merged_headers = {
            str(k): str(v) for k, v in extra_headers.items()
            if k and v is not None
        }
        merged_headers.update(_auth_headers(token))

        url = f"{fd_url}/fd/code-apps/{slug}/api{path}"
        try:
            async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT_SECONDS) as client:
                resp = await client.request(
                    method,
                    url,
                    content=body.encode("utf-8") if isinstance(body, str) else body,
                    headers=merged_headers,
                )
        except Exception as e:
            return ToolResult(success=False, error=f"query_app failed: {e}")

        # Try to pretty-print JSON so the agent grounds on structured
        # data, not a one-line blob. Fall through to raw text on
        # parse failure.
        raw = resp.text
        rendered = raw
        try:
            parsed = resp.json()
            rendered = json.dumps(parsed, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass
        if len(rendered) > 12000:
            rendered = rendered[:12000] + f"\n... (truncated; full length {len(raw)})"

        is_failure = resp.status_code >= 500
        # 403 from cross-app gate (only fires for app-to-app, but
        # surface it cleanly in case the user reused query_app from
        # inside an app context).
        if resp.status_code == 403:
            return ToolResult(
                success=False,
                error=(
                    f"App '{slug}' refused the query (HTTP 403). It "
                    "likely needs a 'data_api' block in its manifest. "
                    f"Body: {raw[:200]}"
                ),
            )
        return ToolResult(
            success=not is_failure,
            content=(
                f"{method} {slug}{path} → HTTP {resp.status_code}\n\n"
                f"```json\n{rendered}\n```"
            ),
            error=(
                f"App '{slug}' returned HTTP {resp.status_code}."
                if is_failure else None
            ),
        )
