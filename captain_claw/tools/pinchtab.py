"""PinchTab browser automation tool for Captain Claw.

Provides a ``pinchtab`` tool that controls a headless Chrome browser via the
PinchTab HTTP server.  PinchTab is a standalone Go binary that gives AI agents
token-efficient browser control through accessibility tree snapshots (~800
tokens per page) instead of expensive screenshots.

Key advantages over the Playwright-based ``browser`` tool:
  - Token-efficient: text/snapshot at ~800 tokens vs screenshots at ~2K+
  - Persistent profiles: cookies/localStorage survive restarts
  - Stealth mode: three levels (light/medium/full) for anti-bot evasion
  - Multi-instance: manages multiple Chrome instances with port auto-allocation
  - No Python runtime dependency: standalone Go binary + HTTP calls

Both ``browser`` and ``pinchtab`` tools coexist.  The agent picks based on
the task: ``pinchtab`` for most browsing, ``browser`` when vision LLM
analysis, workflow recording, or API replay is needed.

PinchTab must be installed separately:
  npm install -g pinchtab
  # or: curl -fsSL https://pinchtab.com/install.sh | bash
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from typing import Any

import aiohttp

from captain_claw.config import PinchTabConfig, get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# Module-level state keyed by session ID.
_PINCHTAB_SESSIONS: dict[str, "_PinchTabSession"] = {}


class _PinchTabSession:
    """Manages connection state to a PinchTab server instance."""

    def __init__(self, config: PinchTabConfig) -> None:
        self._config = config
        self._base_url = f"http://{config.host}:{config.port}"
        self._token = config.token or os.environ.get("PINCHTAB_TOKEN", "")
        self._http: aiohttp.ClientSession | None = None
        self._server_proc: asyncio.subprocess.Process | None = None
        self._instance_id: str | None = None
        self._active_tab_id: str | None = None
        self._started_at: float = 0.0

    @staticmethod
    def _as_dict(resp: Any) -> dict[str, Any]:
        """Coerce an API response into a dict.

        PinchTab endpoints return either a JSON object or a JSON array.
        When the response is a list, wrap it so callers can use ``.get()``
        safely.
        """
        if isinstance(resp, dict):
            return resp
        if isinstance(resp, list):
            return {"items": resp}
        return {"value": resp}

    @staticmethod
    def _as_list(resp: Any, key: str = "items") -> list[Any]:
        """Extract a list from an API response.

        Handles both ``[...]`` (raw list) and ``{"key": [...]}`` shapes.
        """
        if isinstance(resp, list):
            return resp
        if isinstance(resp, dict):
            return resp.get(key) or []
        return []

    @property
    def is_alive(self) -> bool:
        return self._http is not None and not self._http.closed

    @property
    def base_url(self) -> str:
        return self._base_url

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _ensure_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
            self._http = aiohttp.ClientSession(
                base_url=self._base_url,
                headers=self._headers(),
                timeout=timeout,
            )
        return self._http

    async def request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make an HTTP request to the PinchTab server.

        Returns the parsed JSON body — may be a dict, list, or scalar
        depending on the endpoint.
        """
        http = await self._ensure_http()
        try:
            async with http.request(method, path, json=json, params=params) as resp:
                if resp.content_type == "application/json":
                    body = await resp.json()
                else:
                    text = await resp.text()
                    body = {"text": text}
                if resp.status >= 400:
                    if isinstance(body, dict):
                        error_msg = body.get("error", body.get("text", f"HTTP {resp.status}"))
                    else:
                        error_msg = str(body) if body else f"HTTP {resp.status}"
                    raise PinchTabError(f"PinchTab API error ({resp.status}): {error_msg}")
                return body
        except aiohttp.ClientError as exc:
            raise PinchTabError(f"PinchTab connection error: {exc}") from exc

    async def request_bytes(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> bytes:
        """Make an HTTP request that returns raw bytes (screenshots, PDFs)."""
        http = await self._ensure_http()
        try:
            async with http.request(method, path, params=params) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise PinchTabError(f"PinchTab API error ({resp.status}): {text}")
                return await resp.read()
        except aiohttp.ClientError as exc:
            raise PinchTabError(f"PinchTab connection error: {exc}") from exc

    async def health_check(self) -> dict[str, Any]:
        """Check if the server is reachable."""
        return await self.request("GET", "/health")

    async def start_server(self) -> None:
        """Auto-start the PinchTab server as a background process."""
        binary = self._config.binary_path or shutil.which("pinchtab")
        if not binary:
            raise PinchTabError(
                "PinchTab binary not found. Install with: "
                "npm install -g pinchtab  "
                "or: curl -fsSL https://pinchtab.com/install.sh | bash"
            )

        env = {**os.environ}
        env["PINCHTAB_PORT"] = str(self._config.port)
        env["PINCHTAB_BIND"] = self._config.host
        if self._token:
            env["PINCHTAB_TOKEN"] = self._token

        log.info("Starting PinchTab server", binary=binary, port=self._config.port)
        self._server_proc = await asyncio.create_subprocess_exec(
            binary, "server",
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        # Wait briefly for server to become ready.
        for _ in range(20):
            await asyncio.sleep(0.5)
            try:
                await self.health_check()
                self._started_at = time.time()
                log.info("PinchTab server ready", pid=self._server_proc.pid)
                return
            except (PinchTabError, Exception):
                continue
        raise PinchTabError("PinchTab server failed to start within 10 seconds")

    async def ensure_server(self) -> None:
        """Ensure the PinchTab server is running (auto-start if configured)."""
        try:
            await self.health_check()
            return
        except (PinchTabError, Exception):
            pass

        if not self._config.auto_start:
            raise PinchTabError(
                "PinchTab server is not running and auto_start is disabled. "
                f"Start manually: pinchtab server --port {self._config.port}"
            )

        await self.start_server()

    async def _probe_instance(self, instance_id: str) -> bool:
        """Check if an instance is truly ready by listing its tabs.

        A status of ``running`` only means PinchTab registered the
        instance — Chrome's CDP connection may not be established yet.
        Listing tabs exercises the actual connection.

        As a side-effect, sets ``_active_tab_id`` to the first tab found
        (if not already set).
        """
        try:
            resp = await self.request("GET", f"/instances/{instance_id}/tabs")
            tabs = self._as_list(resp, "tabs")
            # Capture the first available tab ID if we don't have one.
            if not self._active_tab_id and tabs:
                first = tabs[0]
                if isinstance(first, dict):
                    self._active_tab_id = first.get("id") or first.get("tabId")
                elif isinstance(first, str):
                    self._active_tab_id = first
            return True
        except PinchTabError:
            return False

    async def _stop_instance(self, instance_id: str) -> None:
        """Best-effort stop of an instance."""
        try:
            await self.request("POST", f"/instances/{instance_id}/stop")
        except PinchTabError:
            pass

    async def ensure_instance(self) -> str:
        """Ensure a browser instance is running, return its ID."""
        if self._instance_id:
            try:
                resp = await self.request("GET", f"/instances/{self._instance_id}")
                if resp.get("status") == "running" and await self._probe_instance(self._instance_id):
                    return self._instance_id
            except PinchTabError:
                pass
            self._instance_id = None

        # Check if any instance is already running.
        resp = await self.request("GET", "/instances")
        instances = self._as_list(resp, "instances")
        for inst in instances:
            if isinstance(inst, dict) and inst.get("status") == "running":
                iid = inst["id"]
                if await self._probe_instance(iid):
                    self._instance_id = iid
                    return self._instance_id
                # Stale instance — stop it so we can start fresh.
                log.warning("Stopping stale PinchTab instance", instance_id=iid)
                await self._stop_instance(iid)

        # Start a new instance.
        start_params: dict[str, Any] = {
            "mode": "headless" if self._config.headless else "headed",
        }
        if self._config.default_profile:
            start_params["profileId"] = self._config.default_profile

        resp = self._as_dict(await self.request("POST", "/instances/start", json=start_params))
        self._instance_id = resp.get("id") or resp.get("instanceId", "")
        if not self._instance_id:
            raise PinchTabError("Failed to start PinchTab instance — no ID returned")
        log.info("PinchTab instance started", instance_id=self._instance_id)

        # Wait for the instance to be truly ready (CDP connection live).
        for attempt in range(15):
            await asyncio.sleep(1.0)
            if await self._probe_instance(self._instance_id):
                log.info("PinchTab instance ready", instance_id=self._instance_id, attempts=attempt + 1)
                return self._instance_id

        raise PinchTabError(
            f"PinchTab instance {self._instance_id} started but Chrome "
            "did not become ready within 15 seconds"
        )

    async def get_active_tab(self) -> str | None:
        """Return the active tab ID, or None."""
        return self._active_tab_id

    async def close(self) -> None:
        """Shut down the session."""
        if self._instance_id:
            try:
                await self.request("POST", f"/instances/{self._instance_id}/stop")
            except PinchTabError:
                pass
            self._instance_id = None
            self._active_tab_id = None

        if self._server_proc and self._server_proc.returncode is None:
            self._server_proc.terminate()
            try:
                await asyncio.wait_for(self._server_proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._server_proc.kill()
            self._server_proc = None

        if self._http and not self._http.closed:
            await self._http.close()
            self._http = None


class PinchTabError(Exception):
    """PinchTab-specific errors."""


class PinchTabTool(Tool):
    """Control a headless browser via PinchTab HTTP API.

    Token-efficient browser automation using accessibility tree snapshots
    (~800 tokens) instead of screenshots (~2K+ tokens).  Supports persistent
    profiles, stealth mode, and multi-instance orchestration.

    Core actions:
        navigate   – Go to a URL (auto-starts server + instance).
        snapshot   – Get page accessibility tree (element refs: e0, e5, e12…).
                     USE THIS as the primary way to read and understand pages.
        click      – Click element by ref from snapshot (or by text via 'text' param).
                     Automatically detects navigation (link clicks) and waits for page load.
        wait       – Wait for specified seconds (use wait_seconds param). Good for dynamic pages.
        type       – Type text into element by ref.
        fill       – Replace input value by ref.
        press      – Press keyboard key.
        scroll     – Scroll the page.
        hover      – Hover over element by ref.
        select     – Select dropdown option by ref.
        text       – Extract raw page text (~800 tokens, most token-efficient).
        find       – Natural language element discovery.
        links      – Extract all links with their URLs and text from the page.
                     USE THIS to find the actual URL before navigating to an article.
        screenshot – Capture page image (fallback, use snapshot first).
        pdf        – Export page as PDF.
        eval       – Run JavaScript (disabled by default).

    Tab management:
        tabs       – List open tabs.
        tab_open   – Open a new tab.
        tab_close  – Close a tab.
        cookies    – Get cookies for active tab.

    Profile management:
        profiles       – List persistent profiles.
        profile_create – Create a new profile.

    Lifecycle:
        health – Check PinchTab server status.
        status – Current session info (instance, tab, URL).
        close  – Stop the instance and shut down.

    WORKFLOW: navigate → snapshot → click/type/fill → snapshot → text.
    To navigate to an article: navigate → links (find URL) → navigate (to URL).
    """

    name = "pinchtab"
    description = (
        "Token-efficient browser automation via PinchTab HTTP API. "
        "Uses accessibility tree snapshots (~800 tokens) instead of screenshots (~2K+). "
        "Actions: navigate, snapshot (get element refs e0/e5/e12), click (by ref or text), "
        "wait, type, fill, press, scroll, hover, select, text (extract page text), "
        "find (natural language), links (list link elements). "
        "Extras: screenshot, pdf, eval, tabs, tab_open, tab_close, cookies, "
        "profiles, profile_create, health, status, close. "
        "Click auto-detects navigation and tries hover+Enter if needed. "
        "Supports persistent profiles (login state survives restarts) and stealth mode. "
        "WORKFLOW: navigate → snapshot → click ref → (auto-navigates) → text. "
        "IMPORTANT: Always click link refs from snapshot — NEVER guess or construct URLs. "
        "If click doesn't navigate, try: snapshot to find cookie/accept buttons, dismiss them, then retry click."
    )
    timeout_seconds = 60.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "navigate",
                    "snapshot",
                    "click",
                    "wait",
                    "type",
                    "fill",
                    "press",
                    "scroll",
                    "hover",
                    "select",
                    "text",
                    "find",
                    "links",
                    "screenshot",
                    "pdf",
                    "eval",
                    "tabs",
                    "tab_open",
                    "tab_close",
                    "cookies",
                    "profiles",
                    "profile_create",
                    "health",
                    "status",
                    "close",
                ],
                "description": "PinchTab action to perform.",
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (for 'navigate', 'tab_open').",
            },
            "ref": {
                "type": "string",
                "description": (
                    "Element reference from snapshot (for 'click', 'type', 'fill', "
                    "'hover', 'select', 'scroll'). Examples: 'e0', 'e5', 'e12'."
                ),
            },
            "text": {
                "type": "string",
                "description": (
                    "Text to type or fill (for 'type', 'fill'). "
                    "For 'select': the option value or visible text."
                ),
            },
            "key": {
                "type": "string",
                "description": "Key to press (for 'press'). Examples: 'Enter', 'Tab', 'Escape'.",
            },
            "query": {
                "type": "string",
                "description": (
                    "Natural language query (for 'find'). "
                    "Example: 'search bar', 'login button', 'email input'."
                ),
            },
            "scroll_direction": {
                "type": "string",
                "enum": ["up", "down"],
                "description": "Scroll direction (for 'scroll'). Default: 'down'.",
            },
            "scroll_amount": {
                "type": "number",
                "description": "Pixels to scroll (for 'scroll'). Default: 500.",
            },
            "filter": {
                "type": "string",
                "enum": ["all", "interactive"],
                "description": (
                    "Snapshot filter (for 'snapshot'). "
                    "'interactive' returns only buttons/links/inputs — most token-efficient. "
                    "Default: 'interactive'."
                ),
            },
            "selector": {
                "type": "string",
                "description": (
                    "CSS selector to scope the snapshot (for 'snapshot'). "
                    "Example: '#main-content', '.search-results'."
                ),
            },
            "expression": {
                "type": "string",
                "description": "JavaScript expression to evaluate (for 'eval').",
            },
            "tab_id": {
                "type": "string",
                "description": "Tab ID (for 'tab_close', or to target a specific tab).",
            },
            "profile_name": {
                "type": "string",
                "description": "Profile name (for 'profile_create').",
            },
            "profile_description": {
                "type": "string",
                "description": "Profile description (for 'profile_create').",
            },
            "new_tab": {
                "type": "boolean",
                "description": "Open URL in a new tab (for 'navigate'). Default: false.",
            },
            "wait_seconds": {
                "type": "number",
                "description": (
                    "Seconds to wait after navigation or click before returning "
                    "(for 'navigate', 'click', 'wait'). "
                    "Recommended: 3+ for dynamic pages. Default: 2."
                ),
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        pass  # sessions stored in module-level _PINCHTAB_SESSIONS

    def _get_session(self, session_id: str = "default") -> _PinchTabSession:
        """Return or lazily create the PinchTab session for *session_id*."""
        session_id = session_id or "default"
        if session_id not in _PINCHTAB_SESSIONS:
            cfg = get_config()
            log.info("Creating new PinchTabSession", session_id=session_id)
            _PINCHTAB_SESSIONS[session_id] = _PinchTabSession(config=cfg.tools.pinchtab)
        return _PINCHTAB_SESSIONS[session_id]

    @staticmethod
    def _session_key(kwargs: dict[str, Any]) -> str:
        return str(kwargs.get("_session_id", "")).strip() or "default"

    # -- execute (dispatch) ---------------------------------------------------

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action handler."""
        log.info("PinchTab tool called", action=action, kwargs_keys=list(kwargs.keys()))

        cfg = get_config()
        if not cfg.tools.pinchtab.enabled:
            return ToolResult(
                success=False,
                error=(
                    "PinchTab is not enabled. "
                    "Set tools.pinchtab.enabled: true in config.yaml"
                ),
            )

        action = action.strip().lower()

        _DISPATCH: dict[str, str] = {
            "navigate": "_navigate",
            "snapshot": "_snapshot",
            "click": "_click",
            "wait": "_wait",
            "type": "_type",
            "fill": "_fill",
            "press": "_press",
            "scroll": "_scroll",
            "hover": "_hover",
            "select": "_select",
            "text": "_text",
            "find": "_find",
            "links": "_links",
            "screenshot": "_screenshot",
            "pdf": "_pdf",
            "eval": "_eval",
            "tabs": "_tabs",
            "tab_open": "_tab_open",
            "tab_close": "_tab_close",
            "cookies": "_cookies",
            "profiles": "_profiles",
            "profile_create": "_profile_create",
            "health": "_health",
            "status": "_status",
            "close": "_close",
        }

        handler_name = _DISPATCH.get(action)
        if not handler_name:
            return ToolResult(success=False, error=f"Unknown pinchtab action: {action}")

        try:
            handler = getattr(self, handler_name)
            result = await handler(**kwargs)
            log.info(
                "PinchTab action completed",
                action=action,
                success=result.success,
                content_len=len(result.content or ""),
            )
            return result
        except PinchTabError as exc:
            log.error("PinchTab error", action=action, error=str(exc))
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:
            log.error("PinchTab tool error", action=action, error=str(exc), exc_info=True)
            return ToolResult(success=False, error=f"PinchTab {action} failed: {exc}")

    # -- helpers --------------------------------------------------------------

    async def _ensure_ready(self, kwargs: dict[str, Any]) -> tuple[_PinchTabSession, str]:
        """Ensure server + instance are running. Return (session, instance_id)."""
        session = self._get_session(self._session_key(kwargs))
        await session.ensure_server()
        instance_id = await session.ensure_instance()
        return session, instance_id

    def _tab_path(self, session: _PinchTabSession, tab_id: str | None = None) -> str:
        """Build /tabs/{id} path.

        In server mode PinchTab requires tab-specific paths for all
        browser endpoints.  The shorthand paths (``/navigate``, ``/snapshot``,
        etc.) only work in bridge mode.
        """
        tid = tab_id or session._active_tab_id
        if tid:
            return f"/tabs/{tid}"
        return ""

    def _require_tab_path(self, session: _PinchTabSession, tab_id: str | None = None) -> str:
        """Like ``_tab_path`` but raises if no tab is available."""
        path = self._tab_path(session, tab_id)
        if not path:
            raise PinchTabError(
                "No active tab. Navigate to a URL first with: "
                "pinchtab(action='navigate', url='...')"
            )
        return path

    # -- action handlers ------------------------------------------------------

    async def _navigate(self, **kwargs: Any) -> ToolResult:
        url = str(kwargs.get("url", "")).strip()
        if not url:
            return ToolResult(success=False, error="URL is required for navigate")

        session, _ = await self._ensure_ready(kwargs)

        params: dict[str, Any] = {"url": url}
        if kwargs.get("new_tab"):
            params["newTab"] = True
        if kwargs.get("tab_id"):
            params["tabId"] = kwargs["tab_id"]

        # Use tab-specific path when we have an active tab.
        tab_path = self._tab_path(session, kwargs.get("tab_id"))
        nav_path = f"{tab_path}/navigate" if tab_path else "/navigate"

        # Retry once on transient Chrome errors (context canceled, target closed).
        last_error: str = ""
        for attempt in range(2):
            try:
                resp = session._as_dict(await session.request("POST", nav_path, json=params))
                break
            except PinchTabError as exc:
                last_error = str(exc)
                if attempt == 0 and ("context canceled" in last_error or "target closed" in last_error):
                    log.warning("Navigate failed, restarting instance", error=last_error)
                    # Instance is stale — restart and retry.
                    if session._instance_id:
                        await session._stop_instance(session._instance_id)
                        session._instance_id = None
                        session._active_tab_id = None
                    await session.ensure_instance()
                    continue
                raise
        else:
            return ToolResult(success=False, error=f"Navigate failed after retry: {last_error}")

        # Track active tab.
        tab_id = resp.get("tabId") or resp.get("tab_id")
        if tab_id:
            session._active_tab_id = tab_id

        # Optional wait for dynamic content.
        wait_secs = kwargs.get("wait_seconds", 2)
        if wait_secs and float(wait_secs) > 0:
            await asyncio.sleep(float(wait_secs))

        title = resp.get("title", "")
        final_url = resp.get("url", url)
        return ToolResult(
            success=True,
            content=(
                f"Navigated to: {final_url}\n"
                f"Title: {title}\n"
                f"Tab: {tab_id or 'default'}"
            ),
        )

    async def _snapshot(self, **kwargs: Any) -> ToolResult:
        session, _ = await self._ensure_ready(kwargs)

        params: dict[str, str] = {}
        filter_val = kwargs.get("filter", "interactive")
        if filter_val:
            params["filter"] = filter_val
        params["format"] = "compact"
        if kwargs.get("selector"):
            params["selector"] = kwargs["selector"]

        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))
        path = f"{tab_path}/snapshot"
        resp = session._as_dict(await session.request("GET", path, params=params))

        # The response is typically the snapshot text or a JSON with 'snapshot' key.
        snapshot = resp.get("snapshot") or resp.get("text") or str(resp)
        url = resp.get("url", "")
        title = resp.get("title", "")

        header = ""
        if url or title:
            header = f"Page: {title} ({url})\n\n"

        return ToolResult(success=True, content=f"{header}{snapshot}")

    async def _get_current_url(self, session: _PinchTabSession, tab_path: str) -> str:
        """Get the current page URL for the active tab (lightweight check)."""
        try:
            resp = session._as_dict(await session.request("GET", f"{tab_path}/text"))
            return resp.get("url", "")
        except PinchTabError:
            return ""

    async def _list_tab_ids(self, session: _PinchTabSession, instance_id: str) -> list[str]:
        """Return a list of tab IDs for the given instance."""
        try:
            resp = await session.request("GET", f"/instances/{instance_id}/tabs")
            tabs = session._as_list(resp, "tabs")
            ids: list[str] = []
            for t in tabs:
                if isinstance(t, dict):
                    tid = t.get("id") or t.get("tabId")
                    if tid:
                        ids.append(tid)
            return ids
        except PinchTabError:
            return []

    async def _click(self, **kwargs: Any) -> ToolResult:
        ref = kwargs.get("ref", "")
        text_query = kwargs.get("text", "")

        # If no ref but text provided, use find to locate the element first.
        if not ref and text_query:
            # Delegate to the _find handler which has full fallback logic.
            find_result = await self._find(query=text_query, **{
                k: v for k, v in kwargs.items() if k not in ("ref", "text", "query", "action")
            })
            if find_result.success and find_result.content:
                # Extract ref from the first line: "Best match: e38 ..."
                first_line = find_result.content.splitlines()[0]
                for token in first_line.split():
                    if token.startswith("e") and token[1:].isdigit():
                        ref = token
                        break
                if not ref:
                    return ToolResult(
                        success=False,
                        error=f"Could not find clickable element matching '{text_query}'. "
                              "Use snapshot to see available elements and their refs.",
                    )
                log.info("Click: resolved text to ref via find", text=text_query, ref=ref)
            else:
                return ToolResult(
                    success=False,
                    error=f"Could not find element matching '{text_query}': "
                          f"{find_result.error or 'no matches'}. "
                          "Use snapshot to see available elements and click by ref.",
                )

        if not ref:
            return ToolResult(
                success=False,
                error="'ref' is required (e.g. 'e5'). Or provide 'text' to click by visible text.",
            )

        session, instance_id = await self._ensure_ready(kwargs)
        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))

        # Capture state before click.
        url_before = await self._get_current_url(session, tab_path)
        tabs_before = await self._list_tab_ids(session, instance_id)

        # Perform the click — if "not focusable" or "not an Element", try to
        # find a nearby link element with matching text from the snapshot.
        try:
            await session.request("POST", f"{tab_path}/action", json={"kind": "click", "ref": ref})
        except PinchTabError as exc:
            err_msg = str(exc).lower()
            if "not focusable" in err_msg or "not an element" in err_msg:
                log.info("Click ref not focusable, searching for nearby link", ref=ref, error=str(exc))
                alt_ref = await self._find_nearby_link(session, tab_path, ref)
                if alt_ref:
                    log.info("Retrying click with link ref", original=ref, link_ref=alt_ref)
                    ref = alt_ref
                    await session.request("POST", f"{tab_path}/action", json={"kind": "click", "ref": ref})
                else:
                    raise
            else:
                raise

        # Wait for potential navigation or new tab.
        wait_secs = float(kwargs.get("wait_seconds", 2))
        if wait_secs > 0:
            await asyncio.sleep(wait_secs)

        # Check if navigation occurred in the same tab.
        url_after = await self._get_current_url(session, tab_path)

        if url_after and url_before and url_after != url_before:
            return ToolResult(
                success=True,
                content=(
                    f"Clicked {ref} → navigated to: {url_after}\n"
                    "Use snapshot or text to read the new page."
                ),
            )

        # Check if a new tab was opened (target="_blank" links).
        tabs_after = await self._list_tab_ids(session, instance_id)
        new_tabs = [t for t in tabs_after if t not in tabs_before]
        if new_tabs:
            new_tab_id = new_tabs[0]
            session._active_tab_id = new_tab_id
            log.info("Click opened new tab, switching", old_tab=tab_path, new_tab=new_tab_id)
            new_tab_path = f"/tabs/{new_tab_id}"
            new_url = await self._get_current_url(session, new_tab_path)
            return ToolResult(
                success=True,
                content=(
                    f"Clicked {ref} → opened new tab: {new_url or new_tab_id}\n"
                    "Switched to the new tab. Use snapshot or text to read the page."
                ),
            )

        # Click didn't navigate — try alternative activation methods.
        # Method 1: Hover + Enter (keyboard activation bypasses cookie overlays).
        log.info("Click didn't navigate, trying hover+Enter", ref=ref)
        url_before_retry = url_after or url_before  # current URL
        activated = await self._activate_link(session, tab_path, ref)
        if activated:
            url_after_enter = await self._get_current_url(session, tab_path)
            if url_after_enter and url_after_enter != url_before_retry:
                return ToolResult(
                    success=True,
                    content=(
                        f"Clicked {ref} → navigated to: {url_after_enter}\n"
                        "Use snapshot or text to read the new page."
                    ),
                )
            # Check for new tab opened by Enter.
            tabs_after_enter = await self._list_tab_ids(session, instance_id)
            new_tabs_enter = [t for t in tabs_after_enter if t not in tabs_before]
            if new_tabs_enter:
                new_tab_id = new_tabs_enter[0]
                session._active_tab_id = new_tab_id
                new_tab_path = f"/tabs/{new_tab_id}"
                new_url = await self._get_current_url(session, new_tab_path)
                return ToolResult(
                    success=True,
                    content=(
                        f"Clicked {ref} → opened new tab: {new_url or new_tab_id}\n"
                        "Switched to the new tab. Use snapshot or text to read the page."
                    ),
                )

        # Method 2: Try to extract the link's href via evaluate and navigate.
        href = await self._resolve_link_href(session, tab_path, ref)
        if href and href != url_before_retry and not href.startswith("javascript:"):
            log.info("Click didn't navigate, using href fallback", ref=ref, href=href)
            try:
                await session.request("POST", f"{tab_path}/navigate", json={"url": href})
                await asyncio.sleep(1.5)
                return ToolResult(
                    success=True,
                    content=(
                        f"Clicked {ref} → navigated to: {href}\n"
                        "Use snapshot or text to read the new page."
                    ),
                )
            except PinchTabError as exc:
                log.warning("Href fallback navigate failed", href=href, error=str(exc))

        return ToolResult(
            success=True,
            content=(
                f"Clicked {ref} (page did not navigate). "
                "The link may require cookie consent dismissal first. "
                "Try: snapshot to find cookie/accept buttons, click them, then retry."
            ),
        )

    async def _find_nearby_link(
        self, session: _PinchTabSession, tab_path: str, ref: str
    ) -> str:
        """When a ref is not focusable (e.g. heading inside a link), find the
        parent link element by looking for a link ref with matching text in
        the interactive snapshot.

        Returns the link ref string (e.g. 'e66') or empty string.
        """
        try:
            snap = session._as_dict(
                await session.request(
                    "GET",
                    f"{tab_path}/snapshot",
                    params={"filter": "interactive", "format": "compact"},
                )
            )
            snapshot = snap.get("snapshot") or snap.get("text") or ""

            # Find the text of the non-focusable element.
            target_text = ""
            for line in snapshot.splitlines():
                stripped = line.strip()
                if stripped.startswith(f"[{ref}]"):
                    # Extract text: everything after role word, or inside quotes.
                    after = stripped[stripped.find("]") + 1:].strip()
                    if '"' in after:
                        parts = after.split('"')
                        if len(parts) >= 2:
                            target_text = parts[1].lower()
                    else:
                        tokens = after.split(None, 1)
                        if len(tokens) > 1:
                            target_text = tokens[1].lower()
                    break

            if not target_text:
                return ""

            # Find a link element whose text contains the same content.
            for line in snapshot.splitlines():
                stripped = line.strip()
                if not stripped.startswith("[e"):
                    continue
                bracket_end = stripped.find("]")
                if bracket_end < 0:
                    continue
                candidate_ref = stripped[1:bracket_end]
                if candidate_ref == ref:
                    continue  # Skip the original ref.
                rest = stripped[bracket_end + 1:].strip()
                if not rest.lower().startswith("link "):
                    continue  # Only consider link elements.
                if target_text[:30] in rest.lower():
                    log.info("Found nearby link for non-focusable ref", original=ref, link=candidate_ref)
                    return candidate_ref

            return ""
        except (PinchTabError, Exception) as exc:
            log.debug("_find_nearby_link failed", ref=ref, error=str(exc))
            return ""

    async def _activate_link(
        self, session: _PinchTabSession, tab_path: str, ref: str
    ) -> bool:
        """Try to activate a link element using hover + Enter keypress.

        This bypasses the issue where CDP click succeeds at the protocol level
        but doesn't trigger navigation (e.g. cookie overlays intercepting,
        JS click handlers not firing).  Keyboard activation (Enter on focused
        element) goes directly to the focused element regardless of overlays.
        """
        try:
            await session.request(
                "POST", f"{tab_path}/action", json={"kind": "hover", "ref": ref}
            )
            await asyncio.sleep(0.3)
            await session.request(
                "POST", f"{tab_path}/action", json={"kind": "press", "key": "Enter"}
            )
            await asyncio.sleep(2)
            return True
        except PinchTabError as exc:
            log.debug("_activate_link failed", ref=ref, error=str(exc))
            return False

    async def _resolve_link_href(
        self, session: _PinchTabSession, tab_path: str, ref: str
    ) -> str:
        """Try to find the href of a link element via JavaScript evaluate.

        Returns the URL string, or empty string if evaluate is unavailable
        or the ref isn't a link.
        """
        try:
            # Get snapshot to find the text of the clicked element.
            snap = session._as_dict(
                await session.request(
                    "GET", f"{tab_path}/snapshot", params={"format": "compact"}
                )
            )
            snapshot_text = snap.get("snapshot") or snap.get("text") or ""

            # Find the line for this ref: e.g. [e66] link "Article Title"
            element_text = ""
            for line in snapshot_text.splitlines():
                stripped = line.strip()
                if not stripped.startswith(f"[{ref}]"):
                    continue
                if '"' in stripped:
                    parts = stripped.split('"')
                    if len(parts) >= 2:
                        element_text = parts[1]
                else:
                    after_bracket = stripped[len(f"[{ref}]"):].strip()
                    tokens = after_bracket.split(None, 1)
                    if len(tokens) > 1:
                        element_text = tokens[1]
                break

            if not element_text:
                return ""

            # Try evaluate on multiple possible endpoint paths.
            search_text = element_text[:60].replace("\\", "\\\\").replace('"', '\\"')
            js = (
                f'(function(){{'
                f'var els = document.querySelectorAll("a[href]");'
                f'for(var i=0;i<els.length;i++){{'
                f'if(els[i].textContent.trim().includes("{search_text}")){{'
                f'return els[i].href;}}}}'
                f'return "";}})();'
            )
            for eval_path in (f"{tab_path}/evaluate", f"{tab_path}/exec", f"{tab_path}/js"):
                try:
                    eval_resp = session._as_dict(
                        await session.request(
                            "POST", eval_path, json={"expression": js}
                        )
                    )
                    href = str(eval_resp.get("result") or eval_resp.get("value") or "").strip('"')
                    if href:
                        return href
                except PinchTabError as exc:
                    if "404" in str(exc):
                        continue
                    raise
            return ""
        except (PinchTabError, Exception) as exc:
            log.debug("_resolve_link_href failed", ref=ref, error=str(exc))
            return ""

    async def _wait(self, **kwargs: Any) -> ToolResult:
        """Wait for a specified number of seconds."""
        wait_secs = float(kwargs.get("wait_seconds", 2))
        if wait_secs <= 0:
            return ToolResult(success=False, error="wait_seconds must be > 0")
        if wait_secs > 30:
            wait_secs = 30  # Cap at 30 seconds.

        await asyncio.sleep(wait_secs)
        return ToolResult(success=True, content=f"Waited {wait_secs} seconds.")

    async def _type(self, **kwargs: Any) -> ToolResult:
        ref = kwargs.get("ref", "")
        text = kwargs.get("text", "")
        if not ref:
            return ToolResult(success=False, error="'ref' is required (e.g. 'e5')")
        if not text:
            return ToolResult(success=False, error="'text' is required")

        session, _ = await self._ensure_ready(kwargs)
        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))
        resp = await session.request("POST", f"{tab_path}/action", json={"kind": "type", "ref": ref, "text": text})
        return ToolResult(success=True, content=f"Typed into {ref}")

    async def _fill(self, **kwargs: Any) -> ToolResult:
        ref = kwargs.get("ref", "")
        text = kwargs.get("text", "")
        if not ref:
            return ToolResult(success=False, error="'ref' is required (e.g. 'e5')")

        session, _ = await self._ensure_ready(kwargs)
        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))
        resp = await session.request("POST", f"{tab_path}/action", json={"kind": "fill", "ref": ref, "value": text})
        return ToolResult(success=True, content=f"Filled {ref}")

    async def _press(self, **kwargs: Any) -> ToolResult:
        key = kwargs.get("key", "")
        if not key:
            return ToolResult(success=False, error="'key' is required (e.g. 'Enter')")

        session, _ = await self._ensure_ready(kwargs)
        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))
        resp = await session.request("POST", f"{tab_path}/action", json={"kind": "press", "key": key})
        return ToolResult(success=True, content=f"Pressed {key}")

    async def _scroll(self, **kwargs: Any) -> ToolResult:
        session, _ = await self._ensure_ready(kwargs)
        direction = kwargs.get("scroll_direction", "down")
        amount = int(kwargs.get("scroll_amount", 500))

        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))

        payload: dict[str, Any] = {"kind": "scroll", "direction": direction}
        if kwargs.get("ref"):
            payload["ref"] = kwargs["ref"]
        if amount != 500:
            payload["pixels"] = amount

        resp = await session.request("POST", f"{tab_path}/action", json=payload)
        return ToolResult(success=True, content=f"Scrolled {direction} {amount}px")

    async def _hover(self, **kwargs: Any) -> ToolResult:
        ref = kwargs.get("ref", "")
        if not ref:
            return ToolResult(success=False, error="'ref' is required (e.g. 'e5')")

        session, _ = await self._ensure_ready(kwargs)
        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))
        resp = await session.request("POST", f"{tab_path}/action", json={"kind": "hover", "ref": ref})
        return ToolResult(success=True, content=f"Hovered over {ref}")

    async def _select(self, **kwargs: Any) -> ToolResult:
        ref = kwargs.get("ref", "")
        text = kwargs.get("text", "")
        if not ref:
            return ToolResult(success=False, error="'ref' is required")

        session, _ = await self._ensure_ready(kwargs)
        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))
        resp = await session.request(
            "POST", f"{tab_path}/action", json={"kind": "select", "ref": ref, "value": text}
        )
        return ToolResult(success=True, content=f"Selected '{text}' on {ref}")

    async def _text(self, **kwargs: Any) -> ToolResult:
        session, _ = await self._ensure_ready(kwargs)

        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))
        resp = session._as_dict(await session.request("GET", f"{tab_path}/text", params={"raw": "true"}))

        url = resp.get("url", "")
        title = resp.get("title", "")
        text = resp.get("text", "")
        truncated = resp.get("truncated", False)

        header = f"Page: {title} ({url})\n"
        if truncated:
            header += "(truncated)\n"
        return ToolResult(success=True, content=f"{header}\n{text}")

    async def _find(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(success=False, error="'query' is required for find")

        session, _ = await self._ensure_ready(kwargs)
        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))

        # Try tab-scoped find, then shorthand, then fall back to snapshot search.
        resp: Any = None
        for find_path in (f"{tab_path}/find", "/find"):
            try:
                resp = session._as_dict(
                    await session.request("POST", find_path, json={"query": query})
                )
                break
            except PinchTabError as exc:
                if "404" in str(exc):
                    continue
                raise

        if resp is not None:
            matches = resp.get("matches") or []
            best_ref = resp.get("best_ref", "")
            confidence = resp.get("confidence", "")
            lines = [f"Best match: {best_ref} (confidence: {confidence})"]
            for m in matches[:5]:
                lines.append(f"  {m.get('ref', '?')} — {m.get('text', '')[:80]} (score: {m.get('score', '')})")
            return ToolResult(success=True, content="\n".join(lines))

        # Fallback: get *interactive* snapshot and do text matching.
        # Using filter=interactive ensures we only return clickable elements
        # (links, buttons, inputs) — avoids "Element is not focusable" errors.
        log.info("Find endpoint unavailable, falling back to snapshot search", query=query)
        try:
            snap_resp = session._as_dict(
                await session.request(
                    "GET",
                    f"{tab_path}/snapshot",
                    params={"filter": "interactive", "format": "compact"},
                )
            )
            snapshot = snap_resp.get("snapshot") or snap_resp.get("text") or str(snap_resp)
        except PinchTabError:
            return ToolResult(
                success=False,
                error="Find endpoint not available and snapshot failed. Use snapshot action manually.",
            )

        # Text search through interactive elements only.
        # IMPORTANT: filter=interactive returns elements *within* interactive
        # subtrees (e.g. headings inside links). Only match directly-clickable
        # roles to avoid "Element is not focusable" errors.
        _CLICKABLE_ROLES = {
            "link", "button", "textbox", "combobox", "checkbox", "radio",
            "menuitem", "tab", "switch", "searchbox", "option", "spinbutton",
            "slider", "scrollbar",
        }
        query_lower = query.lower()
        query_words = query_lower.split()
        matches_found: list[tuple[str, str, int, int]] = []  # (ref, text, score, role_priority)
        for line in snapshot.splitlines():
            line_stripped = line.strip()
            # Look for lines like: [e38] link "Some Article Title"
            if not line_stripped.startswith("[e"):
                continue
            bracket_end = line_stripped.find("]")
            if bracket_end < 0:
                continue
            ref_str = line_stripped[1:bracket_end]
            rest = line_stripped[bracket_end + 1:].strip()
            rest_lower = rest.lower()

            # Extract the role (first word after the bracket).
            role = rest_lower.split()[0] if rest_lower else ""
            # Skip non-clickable roles (headings, static text, images, etc.
            # inside interactive subtrees).
            if role not in _CLICKABLE_ROLES:
                continue

            score = sum(1 for w in query_words if w in rest_lower)
            if score > 0:
                # Priority: link=2 (most useful for navigation), button=1, others=0
                role_priority = 2 if role == "link" else (1 if role == "button" else 0)
                matches_found.append((ref_str, rest, score, role_priority))

        # Sort by: role priority descending, then by word-match score descending.
        matches_found.sort(key=lambda x: (x[3], x[2]), reverse=True)

        if not matches_found:
            return ToolResult(
                success=False,
                error=f"No clickable elements matching '{query}' found. Use snapshot to see all elements.",
            )

        lines = [f"Best match: {matches_found[0][0]} (text match)"]
        for ref_str, text_val, score, role_prio in matches_found[:5]:
            role_label = text_val.split()[0] if text_val else "element"
            lines.append(f"  {ref_str} — [{role_label}] {text_val[:80]} (words matched: {score})")
        return ToolResult(success=True, content="\n".join(lines))

    async def _links(self, **kwargs: Any) -> ToolResult:
        """Extract all links from the page.

        Tries JavaScript evaluate to get hrefs.  If evaluate is unavailable
        (404), falls back to the interactive snapshot which lists link elements
        with their refs — the agent can click refs to navigate instead of
        guessing URLs.
        """
        session, _ = await self._ensure_ready(kwargs)
        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))

        # Try evaluate on multiple possible endpoint paths.
        js = (
            "JSON.stringify(Array.from(document.querySelectorAll('a[href]')).map(a => ({"
            "text: (a.textContent || '').trim().substring(0, 120),"
            "href: a.href"
            "})).filter(l => l.text && l.href && !l.href.startsWith('javascript:')))"
        )
        links: list[dict[str, str]] | None = None
        for eval_path in (f"{tab_path}/evaluate", f"{tab_path}/exec", f"{tab_path}/js"):
            try:
                resp = session._as_dict(
                    await session.request("POST", eval_path, json={"expression": js})
                )
                result_str = resp.get("result") or resp.get("value") or "[]"
                links = json.loads(result_str) if isinstance(result_str, str) else result_str
                break
            except PinchTabError as exc:
                if "404" in str(exc):
                    continue
                log.warning("Links evaluate error", path=eval_path, error=str(exc))
            except (json.JSONDecodeError, Exception) as exc:
                log.warning("Links evaluate parse error", error=str(exc))

        if links is None:
            # Fallback: interactive snapshot — only link elements with refs.
            log.info("Links: evaluate unavailable, falling back to interactive snapshot")
            try:
                snap_resp = session._as_dict(
                    await session.request(
                        "GET",
                        f"{tab_path}/snapshot",
                        params={"filter": "interactive", "format": "compact"},
                    )
                )
                snapshot = snap_resp.get("snapshot") or snap_resp.get("text") or str(snap_resp)
                link_lines: list[str] = []
                for line in snapshot.splitlines():
                    stripped = line.strip()
                    # Match lines like: [e66] link "Article Title"
                    if not stripped.startswith("[e"):
                        continue
                    rest_lower = stripped[stripped.find("]") + 1:].strip().lower()
                    if rest_lower.startswith("link "):
                        link_lines.append(stripped)
                if link_lines:
                    # Filter by query if provided.
                    query = kwargs.get("query", "").lower()
                    if query:
                        query_words = query.split()
                        link_lines = [
                            l for l in link_lines
                            if any(w in l.lower() for w in query_words)
                        ]
                    return ToolResult(
                        success=True,
                        content=(
                            f"Links found: {len(link_lines)} (refs only — click ref to navigate)\n\n"
                            + "\n".join(link_lines[:40])
                        ),
                    )
            except PinchTabError:
                pass
            return ToolResult(
                success=False,
                error="Could not extract links. Use snapshot action to see page elements.",
            )

        if not links:
            return ToolResult(success=True, content="No links found on the page.")

        # Filter and format.
        query = kwargs.get("query", "").lower()
        if query:
            query_words = query.split()
            links = [
                l for l in links
                if any(w in l.get("text", "").lower() for w in query_words)
            ]

        lines = [f"Links found: {len(links)}"]
        for link in links[:40]:
            text = link.get("text", "").replace("\n", " ")[:80]
            href = link.get("href", "")
            lines.append(f"  [{text}]({href})")

        if len(links) > 40:
            lines.append(f"  ... and {len(links) - 40} more links")

        return ToolResult(success=True, content="\n".join(lines))

    async def _screenshot(self, **kwargs: Any) -> ToolResult:
        session, _ = await self._ensure_ready(kwargs)

        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))
        data = await session.request_bytes("GET", f"{tab_path}/screenshot")

        # Save to file system.
        runtime_base = kwargs.get("_runtime_base_path")
        sid = str(kwargs.get("_session_id", "default"))
        if runtime_base:
            base_dir = os.path.join(runtime_base, "saved", "media", sid)
        else:
            base_dir = os.path.join("saved", "media", sid)
        os.makedirs(base_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        filepath = os.path.join(base_dir, f"pinchtab_screenshot_{ts}.png")
        with open(filepath, "wb") as f:
            f.write(data)

        return ToolResult(
            success=True,
            content=f"Screenshot saved: {filepath} ({len(data)} bytes)",
        )

    async def _pdf(self, **kwargs: Any) -> ToolResult:
        session, _ = await self._ensure_ready(kwargs)

        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))
        data = await session.request_bytes("GET", f"{tab_path}/pdf")

        runtime_base = kwargs.get("_runtime_base_path")
        sid = str(kwargs.get("_session_id", "default"))
        if runtime_base:
            base_dir = os.path.join(runtime_base, "saved", "media", sid)
        else:
            base_dir = os.path.join("saved", "media", sid)
        os.makedirs(base_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        filepath = os.path.join(base_dir, f"pinchtab_page_{ts}.pdf")
        with open(filepath, "wb") as f:
            f.write(data)

        return ToolResult(
            success=True,
            content=f"PDF saved: {filepath} ({len(data)} bytes)",
        )

    async def _eval(self, **kwargs: Any) -> ToolResult:
        cfg = get_config()
        if not cfg.tools.pinchtab.allow_evaluate:
            return ToolResult(
                success=False,
                error="JavaScript eval is disabled. Set tools.pinchtab.allow_evaluate: true",
            )

        expression = kwargs.get("expression", "")
        if not expression:
            return ToolResult(success=False, error="'expression' is required for eval")

        session, _ = await self._ensure_ready(kwargs)
        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))
        resp = session._as_dict(await session.request("POST", f"{tab_path}/evaluate", json={"expression": expression}))

        result_val = resp.get("result", resp.get("value", str(resp)))
        return ToolResult(success=True, content=str(result_val))

    async def _tabs(self, **kwargs: Any) -> ToolResult:
        session, instance_id = await self._ensure_ready(kwargs)
        resp = await session.request("GET", f"/instances/{instance_id}/tabs")

        tabs = session._as_list(resp, "tabs")
        if not tabs:
            return ToolResult(success=True, content="No open tabs.")

        lines = []
        for tab in tabs:
            tid = tab.get("id", "?")
            url = tab.get("url", "")
            title = tab.get("title", "")
            active = " (active)" if tid == session._active_tab_id else ""
            lines.append(f"  [{tid}] {title[:60]} — {url}{active}")

        return ToolResult(success=True, content=f"Open tabs ({len(tabs)}):\n" + "\n".join(lines))

    async def _tab_open(self, **kwargs: Any) -> ToolResult:
        session, instance_id = await self._ensure_ready(kwargs)

        params: dict[str, Any] = {}
        url = kwargs.get("url", "")
        if url:
            params["url"] = url

        resp = session._as_dict(await session.request("POST", f"/instances/{instance_id}/tabs/open", json=params))
        tab_id = resp.get("id") or resp.get("tabId", "")
        session._active_tab_id = tab_id
        return ToolResult(success=True, content=f"Opened new tab: {tab_id}")

    async def _tab_close(self, **kwargs: Any) -> ToolResult:
        session, _ = await self._ensure_ready(kwargs)
        tab_id = kwargs.get("tab_id") or session._active_tab_id
        if not tab_id:
            return ToolResult(success=False, error="No tab to close (provide tab_id)")

        await session.request("POST", f"/tabs/{tab_id}/close")
        if session._active_tab_id == tab_id:
            session._active_tab_id = None
        return ToolResult(success=True, content=f"Closed tab: {tab_id}")

    async def _cookies(self, **kwargs: Any) -> ToolResult:
        session, _ = await self._ensure_ready(kwargs)
        tab_path = self._require_tab_path(session, kwargs.get("tab_id"))
        resp = await session.request("GET", f"{tab_path}/cookies")
        cookies = session._as_list(resp, "cookies")

        if not cookies:
            return ToolResult(success=True, content="No cookies.")

        lines = []
        for c in cookies[:20]:
            name = c.get("name", "?")
            domain = c.get("domain", "")
            lines.append(f"  {name} (domain: {domain})")
        suffix = f"\n  ... and {len(cookies) - 20} more" if len(cookies) > 20 else ""
        return ToolResult(
            success=True,
            content=f"Cookies ({len(cookies)}):\n" + "\n".join(lines) + suffix,
        )

    async def _profiles(self, **kwargs: Any) -> ToolResult:
        session = self._get_session(self._session_key(kwargs))
        await session.ensure_server()
        resp = await session.request("GET", "/profiles")

        profiles = session._as_list(resp, "profiles")
        if not profiles:
            return ToolResult(success=True, content="No profiles.")

        lines = []
        for p in profiles:
            pid = p.get("id", "?")
            name = p.get("name", "")
            desc = p.get("description", "")
            lines.append(f"  [{pid}] {name} — {desc}")
        return ToolResult(success=True, content=f"Profiles ({len(profiles)}):\n" + "\n".join(lines))

    async def _profile_create(self, **kwargs: Any) -> ToolResult:
        name = kwargs.get("profile_name", "")
        if not name:
            return ToolResult(success=False, error="'profile_name' is required")

        session = self._get_session(self._session_key(kwargs))
        await session.ensure_server()

        payload: dict[str, Any] = {"name": name}
        if kwargs.get("profile_description"):
            payload["description"] = kwargs["profile_description"]

        resp = session._as_dict(await session.request("POST", "/profiles", json=payload))
        pid = resp.get("id", "?")
        return ToolResult(success=True, content=f"Created profile: {name} (id: {pid})")

    async def _health(self, **kwargs: Any) -> ToolResult:
        session = self._get_session(self._session_key(kwargs))
        try:
            resp = await session.health_check()
            return ToolResult(success=True, content=f"PinchTab server healthy: {resp}")
        except PinchTabError as exc:
            return ToolResult(success=False, error=str(exc))

    async def _status(self, **kwargs: Any) -> ToolResult:
        session = self._get_session(self._session_key(kwargs))

        try:
            health = await session.health_check()
        except PinchTabError:
            return ToolResult(
                success=True,
                content="PinchTab server is not running.",
            )

        lines = ["PinchTab server: running"]
        if session._instance_id:
            lines.append(f"Instance: {session._instance_id}")
        if session._active_tab_id:
            lines.append(f"Active tab: {session._active_tab_id}")

        profiles = health.get("profiles", 0)
        instances = health.get("instances", 0)
        lines.append(f"Profiles: {profiles}, Instances: {instances}")

        return ToolResult(success=True, content="\n".join(lines))

    async def _close(self, **kwargs: Any) -> ToolResult:
        session_key = self._session_key(kwargs)
        session = _PINCHTAB_SESSIONS.get(session_key)
        if not session:
            return ToolResult(success=True, content="No active PinchTab session.")

        await session.close()
        _PINCHTAB_SESSIONS.pop(session_key, None)
        return ToolResult(success=True, content="PinchTab session closed.")
