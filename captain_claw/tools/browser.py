"""Browser automation tool for Captain Claw.

Provides a single ``browser`` tool with action-based dispatch for controlling
a headless Playwright browser.  The browser session persists across tool calls
within the same Captain Claw session, enabling multi-step workflows such as
logging into web apps, navigating pages, capturing screenshots, and
interacting with UI elements.

Playwright is an **optional** dependency.  If it is not installed the tool
returns a clear error message with install instructions.
"""

from __future__ import annotations

import json as _json
import time
from pathlib import Path
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.browser_accessibility import AccessibilityExtractor
from captain_claw.tools.browser_api_replay import ApiReplayEngine, ApiReplayResult
from captain_claw.tools.browser_credentials import CredentialStore
from captain_claw.tools.browser_network import NetworkInterceptor
from captain_claw.tools.browser_session import BrowserSession, has_playwright
from captain_claw.tools.browser_vision import BrowserVision
from captain_claw.tools.browser_workflow import WorkflowRecorder, WorkflowReplayEngine
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# Module-level session/recorder stores keyed by _session_id.
# Survives BrowserTool re-instantiation (e.g. when Telegram agents
# call _register_default_tools and replace the global registry entry).
_BROWSER_SESSIONS: dict[str, BrowserSession] = {}
_WORKFLOW_RECORDERS: dict[str, WorkflowRecorder] = {}


class BrowserTool(Tool):
    """Control a headless browser for interacting with web applications.

    Actions:
        open       – Launch the browser (called automatically on first use).
        navigate   – Go to a URL.
        screenshot – Capture the current page as a JPEG image.
        click      – Click an element by CSS selector, visible text, or ARIA role.
                     Supports ``nth`` for disambiguation when multiple elements match.
        type       – Enter text into a form field.
                     Supports ``nth`` for disambiguation.
        press_key  – Press a keyboard key (Enter, Tab, Escape, …).
        scroll     – Scroll the page up or down.
        wait       – Pause for dynamic content to load.
        status     – Show current browser state (URL, uptime, viewport).
        close      – Shut down the browser session.

    Page understanding actions:
        observe            – Rich page analysis: screenshot + vision LLM + accessibility tree + interactive elements.
        act                – Goal-directed page analysis: observe + recommended next actions.
                             USE THIS in multi-step workflows to decide what to do next.
        accessibility_tree – Extract the page's accessibility tree (semantic structure).
        find_element       – Find interactive elements with suggested Playwright selectors.

    Credential & login actions:
        credentials_store  – Store credentials for a web app (encrypted).
        credentials_list   – List stored credentials (passwords masked).
        credentials_delete – Delete stored credentials.
        login              – Automated login: restore cookies or fill login form.

    Network capture actions:
        network_start   – Start recording network traffic.
        network_stop    – Stop recording (keeps captured data).
        network_list    – List captured API calls.
        network_capture – Analyze captured traffic and store APIs in memory.
        network_clear   – Discard all captured network data.

    API replay actions:
        api_replay     – Execute a captured API directly via HTTP (skip the browser).
        api_test       – Test a captured API endpoint and show the response.

    Multi-app session actions:
        switch_app     – Switch to a different app's browser context.
        list_sessions  – List all active app sessions.

    Workflow recording actions:
        workflow_record_start – Start recording user interactions in the browser.
        workflow_record_stop  – Stop recording and show captured steps.
        workflow_save         – Save recorded steps as a named, replayable workflow.
        workflow_list         – List all saved workflows.
        workflow_show         – Show full details of a saved workflow.
        workflow_run          – Replay a workflow with variable substitution.
        workflow_delete       – Delete a saved workflow.
    """

    name = "browser"
    description = (
        "Control a headless browser for interacting with web applications. "
        "Actions: open, navigate, screenshot, click (with nth), type (with nth), "
        "press_key, scroll, wait, status, close. "
        "Page understanding: observe, act (goal-directed). "
        "Credentials: credentials_store, credentials_list, credentials_delete, login. "
        "Network: network_start, network_stop, network_list, network_capture, network_clear. "
        "API replay: api_replay (execute captured API directly — skip the browser!), api_test. "
        "Multi-app: switch_app, list_sessions (manage multiple app sessions simultaneously). "
        "Workflow recording: workflow_record_start, workflow_record_stop, workflow_save, "
        "workflow_list, workflow_show, workflow_run (replay with variables), workflow_delete. "
        "WORKFLOW: observe/act → click/type → network_capture → api_replay for speed."
    )
    timeout_seconds = 60.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "open",
                    "navigate",
                    "screenshot",
                    "click",
                    "type",
                    "press_key",
                    "scroll",
                    "wait",
                    "status",
                    "close",
                    "observe",
                    "act",
                    "accessibility_tree",
                    "find_element",
                    "credentials_store",
                    "credentials_list",
                    "credentials_delete",
                    "login",
                    "network_start",
                    "network_stop",
                    "network_list",
                    "network_capture",
                    "network_clear",
                    "api_replay",
                    "api_test",
                    "switch_app",
                    "list_sessions",
                    "workflow_record_start",
                    "workflow_record_stop",
                    "workflow_save",
                    "workflow_list",
                    "workflow_show",
                    "workflow_run",
                    "workflow_delete",
                ],
                "description": "Browser action to perform.",
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (for 'navigate' and 'open').",
            },
            "selector": {
                "type": "string",
                "description": (
                    "CSS selector to target an element (for 'click', 'type'). "
                    "Examples: '#login-btn', 'input[name=email]', '.submit-button'."
                ),
            },
            "text": {
                "type": "string",
                "description": (
                    "For 'type': the text to enter into the field. "
                    "For 'click': visible text content to click on "
                    "(alternative to CSS selector, uses Playwright get_by_text)."
                ),
            },
            "role": {
                "type": "string",
                "description": (
                    "ARIA role for element targeting (for 'click', 'type'). "
                    "Examples: 'button', 'textbox', 'link', 'checkbox'. "
                    "Use with 'text' to specify the element name."
                ),
            },
            "key": {
                "type": "string",
                "description": (
                    "Keyboard key to press (for 'press_key'). "
                    "Examples: 'Enter', 'Tab', 'Escape', 'ArrowDown'."
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
            "wait_seconds": {
                "type": "number",
                "description": "Seconds to wait (for 'wait'). Default: 2.",
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture full page screenshot (for 'screenshot'). Default: false.",
            },
            "headless": {
                "type": "boolean",
                "description": "Run in headless mode (for 'open'). Default: true.",
            },
            "nth": {
                "type": "integer",
                "description": (
                    "Zero-based index to select a specific match when multiple elements "
                    "have the same role/text (for 'click', 'type'). "
                    "Essential for React/SPA apps with duplicate labels. "
                    "Example: nth=0 for first match, nth=1 for second. "
                    "Use find_element or observe to discover element indices."
                ),
            },
            "goal": {
                "type": "string",
                "description": (
                    "What you're trying to accomplish on this page (for 'observe' and 'act'). "
                    "For 'act': REQUIRED — drives the analysis and recommended actions. "
                    "For 'observe': optional — focuses the vision analysis. "
                    "Example: 'find the sprint report', 'fill out the search form'."
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Custom prompt for the vision model (for 'observe'). "
                    "Overrides the default page analysis prompt."
                ),
            },
            "app_name": {
                "type": "string",
                "description": (
                    "Unique friendly name for the web app (for credential/login actions). "
                    "Examples: 'jira', 'confluence', 'github'."
                ),
            },
            "username": {
                "type": "string",
                "description": "Login username or email (for 'credentials_store').",
            },
            "password": {
                "type": "string",
                "description": "Login password (for 'credentials_store'). Will be encrypted before storage.",
            },
            "api_id": {
                "type": "string",
                "description": (
                    "API identifier for 'api_replay' and 'api_test'. "
                    "Use a direct ID, '#N' index from apis(action='list'), or a fuzzy name match. "
                    "Examples: 'abc123', '#1', 'jira'."
                ),
            },
            "endpoint": {
                "type": "string",
                "description": (
                    "API endpoint path for 'api_replay' and 'api_test'. "
                    "Examples: '/rest/api/2/search', '/api/users'."
                ),
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "description": "HTTP method for 'api_replay' and 'api_test'. Default: 'GET'.",
            },
            "query_params": {
                "type": "string",
                "description": (
                    "URL query parameters as JSON string for 'api_replay' and 'api_test'. "
                    "Example: '{\"jql\": \"project=PROJ\", \"maxResults\": \"50\"}'."
                ),
            },
            "body_json": {
                "type": "string",
                "description": (
                    "Request body as JSON string for 'api_replay' and 'api_test' (POST/PUT/PATCH). "
                    "Example: '{\"summary\": \"New issue\", \"description\": \"Details\"}'."
                ),
            },
            "workflow_id": {
                "type": "string",
                "description": (
                    "Workflow identifier for 'workflow_show', 'workflow_run', 'workflow_delete'. "
                    "Use the ID from workflow_save or workflow_list. "
                    "Supports direct ID, '#N' index, or fuzzy name match."
                ),
            },
            "workflow_name": {
                "type": "string",
                "description": "Name for the workflow (for 'workflow_save'). Example: 'jira-search'.",
            },
            "workflow_description": {
                "type": "string",
                "description": "Description of what the workflow does (for 'workflow_save').",
            },
            "workflow_variables": {
                "type": "string",
                "description": (
                    "JSON string — meaning depends on action. "
                    "For 'workflow_save': array of variable definitions, e.g. "
                    "'[{\"name\": \"query\", \"step_index\": 2, \"field\": \"value\", \"description\": \"Search query\"}]'. "
                    "For 'workflow_run': object of variable values, e.g. "
                    "'{\"query\": \"my search term\"}'."
                ),
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        pass  # sessions stored in module-level _BROWSER_SESSIONS

    def _get_session(self, session_id: str = "default") -> BrowserSession:
        """Return or lazily create the browser session for *session_id*."""
        session_id = session_id or "default"
        if session_id not in _BROWSER_SESSIONS:
            log.info("Creating new BrowserSession", session_id=session_id)
            cfg = get_config()
            _BROWSER_SESSIONS[session_id] = BrowserSession(config=cfg.tools.browser)
            log.info("BrowserSession created", session_id=session_id, headless=cfg.tools.browser.headless)
        return _BROWSER_SESSIONS[session_id]

    @staticmethod
    def _session_key(kwargs: dict[str, Any]) -> str:
        """Extract session key from tool kwargs."""
        return str(kwargs.get("_session_id", "")).strip() or "default"

    # -- screenshot path helper -----------------------------------------------

    @staticmethod
    def _screenshot_path(kwargs: dict[str, Any]) -> Path:
        """Build the output path for a screenshot file."""
        runtime_base = kwargs.get("_runtime_base_path")
        session_id = str(kwargs.get("_session_id", "default"))

        if runtime_base:
            base = Path(runtime_base) / "saved" / "media" / session_id
        else:
            base = Path("saved") / "media" / session_id

        base.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        return base / f"browser_screenshot_{timestamp}.jpg"

    # -- execute (dispatch) ---------------------------------------------------

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action handler."""
        log.info("Browser tool called", action=action, kwargs_keys=list(kwargs.keys()))

        if not has_playwright():
            log.error("Playwright not installed")
            return ToolResult(
                success=False,
                error=(
                    "Playwright is not installed. "
                    "Install with: pip install 'captain-claw[browser]' && playwright install chromium"
                ),
            )

        action = action.strip().lower()
        log.info("Dispatching browser action", action=action)

        _DISPATCH: dict[str, str] = {
            "open": "_open", "navigate": "_navigate", "screenshot": "_screenshot",
            "click": "_click", "type": "_type", "press_key": "_press_key",
            "scroll": "_scroll", "wait": "_wait", "status": "_status", "close": "_close",
            "observe": "_observe", "act": "_act",
            "accessibility_tree": "_accessibility_tree", "find_element": "_find_element",
            "credentials_store": "_credentials_store", "credentials_list": "_credentials_list",
            "credentials_delete": "_credentials_delete", "login": "_login",
            "network_start": "_network_start", "network_stop": "_network_stop",
            "network_list": "_network_list", "network_capture": "_network_capture",
            "network_clear": "_network_clear",
            "api_replay": "_api_replay", "api_test": "_api_test",
            "switch_app": "_switch_app", "list_sessions": "_list_sessions",
            "workflow_record_start": "_workflow_record_start",
            "workflow_record_stop": "_workflow_record_stop",
            "workflow_save": "_workflow_save",
            "workflow_list": "_workflow_list",
            "workflow_show": "_workflow_show",
            "workflow_run": "_workflow_run",
            "workflow_delete": "_workflow_delete",
        }

        handler_name = _DISPATCH.get(action)
        if not handler_name:
            log.error("Unknown browser action", action=action)
            return ToolResult(success=False, error=f"Unknown browser action: {action}")

        try:
            handler = getattr(self, handler_name)
            result = await handler(**kwargs)
            log.info(
                "Browser action completed",
                action=action,
                success=result.success,
                content_len=len(result.content or ""),
                error=result.error,
            )
            return result
        except Exception as e:
            log.error("Browser tool error", action=action, error=str(e), exc_info=True)
            return ToolResult(success=False, error=f"Browser {action} failed: {e}")

    # -- action handlers ------------------------------------------------------

    @staticmethod
    def _is_google_drive_url(url: str) -> bool:
        """Return True if URL is a Google Drive/Docs/Sheets/Slides URL."""
        _gdrive_hosts = (
            "docs.google.com", "drive.google.com",
            "sheets.google.com", "slides.google.com",
        )
        try:
            from urllib.parse import urlparse
            host = urlparse(url).hostname or ""
            return any(host == h or host.endswith("." + h) for h in _gdrive_hosts)
        except Exception:
            return False

    _GDRIVE_BLOCK_MSG = (
        "Cannot open Google Drive/Docs URLs in the browser (requires authentication). "
        "Use the gws tool instead:\n"
        "  - gws(action='docs_read', file_id='...') to read document content\n"
        "  - gws(action='drive_download', file_id='...') to download files\n"
        "  - gws(action='drive_info', file_id='...') for file metadata\n"
        "The docs_read action returns the full document text inline."
    )

    async def _open(self, **kwargs: Any) -> ToolResult:
        """Launch the browser."""
        log.info("browser._open called", headless=kwargs.get("headless"), url=kwargs.get("url"))

        # Block Google Drive URLs — gws tool should be used instead.
        url = str(kwargs.get("url", "")).strip()
        if url and self._is_google_drive_url(url):
            return ToolResult(success=False, error=self._GDRIVE_BLOCK_MSG)

        session = self._get_session(self._session_key(kwargs))

        if session.is_alive:
            return ToolResult(
                success=True,
                content=(
                    "Browser is already running.\n"
                    f"Current URL: {session.current_url}\n"
                    f"Uptime: {session.uptime_seconds:.0f}s"
                ),
            )

        headless = kwargs.get("headless")
        await session.start(headless=headless)

        url = str(kwargs.get("url", "")).strip()
        if url:
            nav_info = await session.navigate(url)
            return ToolResult(
                success=True,
                content=(
                    f"Browser launched and navigated to: {nav_info['url']}\n"
                    f"Page title: {nav_info['title']}\n"
                    f"HTTP status: {nav_info['status']}"
                ),
            )

        return ToolResult(
            success=True,
            content="Browser launched successfully (headless mode). Ready to navigate.",
        )

    async def _navigate(self, **kwargs: Any) -> ToolResult:
        """Navigate to a URL."""
        log.info("browser._navigate called", url=kwargs.get("url"))
        url = str(kwargs.get("url", "")).strip()
        if not url:
            return ToolResult(success=False, error="'url' parameter is required for navigate.")

        # Block Google Drive URLs — gws tool should be used instead.
        if self._is_google_drive_url(url):
            return ToolResult(success=False, error=self._GDRIVE_BLOCK_MSG)

        session = self._get_session(self._session_key(kwargs))
        nav_info = await session.navigate(url)

        return ToolResult(
            success=nav_info["ok"] or nav_info["status"] == 0,
            content=(
                f"Navigated to: {nav_info['url']}\n"
                f"Page title: {nav_info['title']}\n"
                f"HTTP status: {nav_info['status']}"
            ),
        )

    async def _screenshot(self, **kwargs: Any) -> ToolResult:
        """Take a screenshot of the current page."""
        log.info("browser._screenshot called", full_page=kwargs.get("full_page"))
        session = self._get_session(self._session_key(kwargs))
        full_page = bool(kwargs.get("full_page", False))

        path = self._screenshot_path(kwargs)
        screenshot_bytes = await session.screenshot(path=path, full_page=full_page)

        # Register with file registry if available
        file_registry = kwargs.get("_file_registry")
        if file_registry is not None:
            try:
                file_registry.register(
                    logical_path=str(path.name),
                    physical_path=str(path),
                )
            except Exception:
                pass  # non-critical

        title = await session.get_title()
        return ToolResult(
            success=True,
            content=(
                f"Path: {path}\n"
                f"Image size: {len(screenshot_bytes):,} bytes\n"
                f"Current URL: {session.current_url}\n"
                f"Page title: {title}\n"
                f"Full page: {full_page}\n\n"
                "Use image_vision or image_ocr on this path to analyze the screenshot content."
            ),
        )

    async def _click(self, **kwargs: Any) -> ToolResult:
        """Click an element.

        Supports targeting by CSS selector, visible text, or ARIA role.
        Use ``nth`` (zero-based) to disambiguate when multiple elements match.
        """
        log.info("browser._click called", selector=kwargs.get("selector"), text=kwargs.get("text"), role=kwargs.get("role"), nth=kwargs.get("nth"))
        session = self._get_session(self._session_key(kwargs))
        selector = str(kwargs.get("selector", "")).strip()
        text = str(kwargs.get("text", "")).strip()
        role = str(kwargs.get("role", "")).strip()
        nth_raw = kwargs.get("nth")
        nth: int | None = int(nth_raw) if nth_raw is not None else None

        nth_label = f", nth={nth}" if nth is not None else ""

        try:
            if role:
                # ARIA role-based targeting (best for React apps)
                await session.click_by_role(role, name=text, nth=nth)
                return ToolResult(
                    success=True,
                    content=(
                        f"Clicked element with role='{role}'"
                        + (f", name='{text}'" if text else "")
                        + nth_label
                        + f"\nCurrent URL: {session.current_url}"
                    ),
                )
            elif text and not selector:
                # Text-based targeting
                await session.click_by_text(text, nth=nth)
                return ToolResult(
                    success=True,
                    content=(
                        f"Clicked element with text: '{text}'{nth_label}\n"
                        f"Current URL: {session.current_url}"
                    ),
                )
            elif selector:
                # CSS selector targeting (nth not applicable — use :nth-child in CSS)
                await session.click(selector)
                return ToolResult(
                    success=True,
                    content=(
                        f"Clicked element: {selector}\n"
                        f"Current URL: {session.current_url}"
                    ),
                )
            else:
                return ToolResult(
                    success=False,
                    error="Click requires 'selector', 'text', or 'role' parameter.",
                )
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "waiting" in error_msg.lower():
                return ToolResult(
                    success=False,
                    error=(
                        f"Element not found (timed out): "
                        f"selector={selector or 'N/A'}, text={text or 'N/A'}, "
                        f"role={role or 'N/A'}{nth_label}. "
                        "Try browser(action='observe') to see the page structure and "
                        "find the correct selector. "
                        "If multiple elements match, use nth=0, nth=1, etc. to select."
                    ),
                )
            raise

    async def _type(self, **kwargs: Any) -> ToolResult:
        """Type text into a form field.

        Supports targeting by CSS selector or ARIA role.
        Use ``nth`` (zero-based) to disambiguate when multiple fields match.
        """
        log.info("browser._type called", selector=kwargs.get("selector"), role=kwargs.get("role"), text_len=len(str(kwargs.get("text", ""))))
        session = self._get_session(self._session_key(kwargs))
        selector = str(kwargs.get("selector", "")).strip()
        text = str(kwargs.get("text", "")).strip()
        role = str(kwargs.get("role", "")).strip()
        nth_raw = kwargs.get("nth")
        nth: int | None = int(nth_raw) if nth_raw is not None else None

        nth_label = f", nth={nth}" if nth is not None else ""

        if not text:
            return ToolResult(success=False, error="'text' parameter is required for type.")

        try:
            if role:
                # ARIA role-based targeting
                name = str(kwargs.get("selector", "")).strip()  # use selector as name hint
                await session.type_by_role(role, name=name, text=text, nth=nth)
                return ToolResult(
                    success=True,
                    content=(
                        f"Typed {len(text)} chars into role='{role}'"
                        + (f", name='{name}'" if name else "")
                        + nth_label
                    ),
                )
            elif selector:
                await session.type_text(selector, text)
                return ToolResult(
                    success=True,
                    content=f"Typed {len(text)} chars into: {selector}",
                )
            else:
                return ToolResult(
                    success=False,
                    error="Type requires 'selector' or 'role' parameter to identify the field.",
                )
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "waiting" in error_msg.lower():
                return ToolResult(
                    success=False,
                    error=(
                        f"Input field not found (timed out): "
                        f"selector={selector or 'N/A'}, role={role or 'N/A'}{nth_label}. "
                        "Try browser(action='observe') to see the page structure and "
                        "find the correct selector. "
                        "If multiple fields match, use nth=0, nth=1, etc. to select."
                    ),
                )
            raise

    async def _press_key(self, **kwargs: Any) -> ToolResult:
        """Press a keyboard key."""
        log.info("browser._press_key called", key=kwargs.get("key"))
        session = self._get_session(self._session_key(kwargs))
        key = str(kwargs.get("key", "")).strip()

        if not key:
            return ToolResult(success=False, error="'key' parameter is required for press_key.")

        await session.press_key(key)
        return ToolResult(
            success=True,
            content=f"Pressed key: {key}",
        )

    async def _scroll(self, **kwargs: Any) -> ToolResult:
        """Scroll the page."""
        log.info("browser._scroll called", direction=kwargs.get("scroll_direction"), amount=kwargs.get("scroll_amount"))
        session = self._get_session(self._session_key(kwargs))
        direction = str(kwargs.get("scroll_direction", "down")).strip().lower()
        amount = int(kwargs.get("scroll_amount", 500))

        await session.scroll(direction=direction, amount=amount)
        return ToolResult(
            success=True,
            content=f"Scrolled {direction} by {amount}px.\nCurrent URL: {session.current_url}",
        )

    async def _wait(self, **kwargs: Any) -> ToolResult:
        """Wait for a specified duration."""
        log.info("browser._wait called", seconds=kwargs.get("wait_seconds"))
        session = self._get_session(self._session_key(kwargs))
        seconds = float(kwargs.get("wait_seconds", 2.0))
        seconds = min(seconds, 30.0)  # safety cap

        await session.wait(seconds)
        return ToolResult(
            success=True,
            content=f"Waited {seconds:.1f} seconds.\nCurrent URL: {session.current_url}",
        )

    async def _status(self, **kwargs: Any) -> ToolResult:
        """Show current browser state."""
        log.info("browser._status called")
        session = self._get_session(self._session_key(kwargs))
        info = session.status_info()

        if not info["alive"]:
            return ToolResult(
                success=True,
                content="Browser is not running. Use browser(action='open') to launch.",
            )

        title = await session.get_title()
        return ToolResult(
            success=True,
            content=(
                f"Browser status:\n"
                f"  Running: {info['alive']}\n"
                f"  URL: {info['url']}\n"
                f"  Title: {title}\n"
                f"  Uptime: {info['uptime_seconds']}s\n"
                f"  Headless: {info['headless']}\n"
                f"  Viewport: {info['viewport']}"
            ),
        )

    async def _close(self, **kwargs: Any) -> ToolResult:
        """Shut down the browser."""
        log.info("browser._close called")
        sk = self._session_key(kwargs)
        session = self._get_session(sk)

        if not session.is_alive:
            return ToolResult(
                success=True,
                content="Browser was not running.",
            )

        uptime = session.uptime_seconds
        capture_count = session.network.capture_count
        await session.close()
        _BROWSER_SESSIONS.pop(sk, None)
        _WORKFLOW_RECORDERS.pop(sk, None)

        extra = ""
        if capture_count:
            extra = (
                f"\nNetwork captures: {capture_count} API calls were recorded. "
                "Use network_capture before closing next time to save them."
            )

        return ToolResult(
            success=True,
            content=f"Browser closed after {uptime:.0f}s.{extra}",
        )

    # -- page understanding actions -------------------------------------------

    async def _observe(self, **kwargs: Any) -> ToolResult:
        """Rich page analysis: screenshot + vision + accessibility tree + interactive elements.

        This is the primary action for understanding a page before acting on it.
        Combines multiple signals for comprehensive page understanding.
        """
        log.info("browser._observe called", goal=kwargs.get("goal"), prompt=bool(kwargs.get("prompt")))
        session = self._get_session(self._session_key(kwargs))
        page = await session.ensure_page()
        title = await session.get_title()
        url = session.current_url

        parts: list[str] = [
            f"URL: {url}",
            f"Title: {title}",
            "",
        ]

        # 1. Screenshot
        path = self._screenshot_path(kwargs)
        screenshot_bytes = await session.screenshot(path=path, full_page=False)

        # Register with file registry
        file_registry = kwargs.get("_file_registry")
        if file_registry is not None:
            try:
                file_registry.register(
                    logical_path=str(path.name),
                    physical_path=str(path),
                )
            except Exception:
                pass

        parts.append(f"Path: {path}")
        parts.append(f"Image size: {len(screenshot_bytes):,} bytes")
        parts.append("")

        # 2. Vision analysis (if a vision model is configured)
        goal = str(kwargs.get("goal", "")).strip()
        custom_prompt = str(kwargs.get("prompt", "")).strip()

        if goal and not custom_prompt:
            # Build a goal-aware prompt
            vision_prompt = (
                f"Analyze this web page screenshot. The user's goal is: {goal}\n\n"
                "Describe:\n"
                "1. What is currently visible on the page\n"
                "2. Where the relevant elements for the goal are located\n"
                "3. What actions would help achieve the goal\n"
                "4. Any forms, buttons, or interactive elements related to the goal\n"
                "Be specific about element locations and names."
            )
        else:
            vision_prompt = custom_prompt  # empty string = default prompt in BrowserVision

        vision_text = await BrowserVision.analyze_screenshot(
            screenshot_bytes,
            prompt=vision_prompt,
        )

        if vision_text:
            parts.append("--- Visual Analysis ---")
            parts.append(vision_text)
            parts.append("")
        else:
            parts.append("--- Visual Analysis ---")
            parts.append("(no vision model configured — using accessibility tree only)")
            parts.append("")

        # 3. Accessibility tree
        tree_text = await AccessibilityExtractor.extract_tree(
            page, max_depth=6, max_lines=150,
        )
        parts.append("--- Page Structure (Accessibility Tree) ---")
        parts.append(tree_text)
        parts.append("")

        # 4. Interactive elements
        interactive = await AccessibilityExtractor.find_interactive_elements(
            page, max_items=40,
        )
        parts.append("--- Interactive Elements ---")
        parts.append(AccessibilityExtractor.format_interactive_list(interactive))

        # 5. Login form detection hint
        login_hint = await self._detect_login_hint(url, interactive)
        if login_hint:
            parts.append("")
            parts.append(login_hint)

        return ToolResult(
            success=True,
            content="\n".join(parts),
        )

    async def _act(self, **kwargs: Any) -> ToolResult:
        """Goal-directed page analysis with recommended next actions.

        Like ``observe`` but more focused and action-oriented.  Takes a ``goal``
        parameter and returns:
        1. Concise page state summary
        2. Interactive elements relevant to the goal
        3. Specific recommended next browser tool call(s)

        This is the primary action for multi-step observe-think-act workflows.
        The existing agent tool loop drives iteration — no inner loop here.
        """
        goal = str(kwargs.get("goal", "")).strip()
        if not goal:
            return ToolResult(
                success=False,
                error=(
                    "'goal' parameter is required for 'act'. "
                    "Describe what you're trying to accomplish, e.g.: "
                    "goal='find and click the Sprint 23 report link'."
                ),
            )

        log.info("browser._act called", goal=goal)
        session = self._get_session(self._session_key(kwargs))
        page = await session.ensure_page()
        title = await session.get_title()
        url = session.current_url

        parts: list[str] = [
            f"Goal: {goal}",
            f"URL: {url}",
            f"Title: {title}",
            "",
        ]

        # 1. Screenshot (always — provides visual context)
        path = self._screenshot_path(kwargs)
        screenshot_bytes = await session.screenshot(path=path, full_page=False)

        file_registry = kwargs.get("_file_registry")
        if file_registry is not None:
            try:
                file_registry.register(
                    logical_path=str(path.name),
                    physical_path=str(path),
                )
            except Exception:
                pass

        parts.append(f"Path: {path}")
        parts.append("")

        # 2. Goal-focused vision analysis
        vision_prompt = (
            f"You are helping automate a browser task. The current goal is: {goal}\n\n"
            "Analyze this screenshot and provide:\n"
            "1. CURRENT STATE: Brief description of what's on the page (1-2 sentences)\n"
            "2. GOAL RELEVANT: Elements, links, or areas related to the goal\n"
            "3. RECOMMENDED ACTION: The single most useful next action to take\n"
            "4. BLOCKERS: Anything preventing progress (login wall, loading, error, etc.)\n\n"
            "Be very specific about element names, text labels, and positions. "
            "Keep the response concise and actionable."
        )

        vision_text = await BrowserVision.analyze_screenshot(
            screenshot_bytes,
            prompt=vision_prompt,
        )

        if vision_text:
            parts.append("--- Analysis ---")
            parts.append(vision_text)
            parts.append("")

        # 3. Interactive elements (filtered for relevance when possible)
        interactive = await AccessibilityExtractor.find_interactive_elements(
            page, max_items=40,
        )

        if interactive:
            parts.append("--- Available Actions ---")
            parts.append(AccessibilityExtractor.format_interactive_list(interactive))
            parts.append("")

        # 4. Compact accessibility tree (fewer lines than full observe)
        tree_text = await AccessibilityExtractor.extract_tree(
            page, max_depth=4, max_lines=80,
        )
        parts.append("--- Page Structure ---")
        parts.append(tree_text)
        parts.append("")

        # 5. Action hints based on page state
        hints = self._generate_action_hints(url, title, interactive, goal)
        if hints:
            parts.append("--- Suggested Next Steps ---")
            parts.append(hints)

        return ToolResult(
            success=True,
            content="\n".join(parts),
        )

    @staticmethod
    def _generate_action_hints(
        url: str,
        title: str,
        interactive: list[dict[str, str]],
        goal: str,
    ) -> str:
        """Generate actionable hints based on page state and goal."""
        hints: list[str] = []
        goal_lower = goal.lower()
        url_lower = url.lower()
        title_lower = title.lower()

        # Detect if we're on a login page
        login_indicators = ["login", "sign in", "signin", "log in", "authenticate"]
        on_login_page = any(ind in url_lower or ind in title_lower for ind in login_indicators)

        if on_login_page:
            hints.append(
                "Page appears to be a login form. "
                "Use browser(action='login', app_name=...) if credentials are stored, "
                "or fill the form with browser(action='type', ...) + browser(action='click', ...)."
            )
            return "\n".join(hints)

        # Detect error/blocked states
        error_indicators = ["error", "denied", "forbidden", "not found", "404", "500"]
        if any(ind in title_lower for ind in error_indicators):
            hints.append(
                "Page shows an error. Consider navigating back or to a different URL."
            )

        # Suggest based on interactive elements and goal
        goal_words = set(goal_lower.split())
        relevant_elements: list[dict[str, str]] = []
        for elem in interactive:
            name_lower = elem["name"].lower()
            name_words = set(name_lower.split())
            # Check for word overlap between goal and element name
            if goal_words & name_words or any(w in name_lower for w in goal_words if len(w) > 3):
                relevant_elements.append(elem)

        if relevant_elements:
            hints.append("Elements matching your goal:")
            for elem in relevant_elements[:5]:
                hints.append(
                    f"  → browser(action='click', role='{elem['role']}', "
                    f"text='{elem['name']}')"
                )
        elif interactive:
            hints.append(
                "No elements directly match the goal text. "
                "Try scrolling, navigating to a different section, or using "
                "browser(action='observe') for a more detailed page analysis."
            )

        # Suggest network capture if the page looks like it has data
        if any(elem["role"] in ("table", "grid", "list") for elem in interactive):
            hints.append(
                "Page contains data tables/lists. After browsing, use "
                "browser(action='network_capture') to save discovered API patterns."
            )

        return "\n".join(hints)

    async def _accessibility_tree(self, **kwargs: Any) -> ToolResult:
        """Extract the page's accessibility tree (semantic structure)."""
        log.info("browser._accessibility_tree called")
        session = self._get_session(self._session_key(kwargs))
        page = await session.ensure_page()
        title = await session.get_title()

        tree_text = await AccessibilityExtractor.extract_tree(
            page, max_depth=6, max_lines=200,
        )

        return ToolResult(
            success=True,
            content=(
                f"URL: {session.current_url}\n"
                f"Title: {title}\n\n"
                f"Accessibility Tree:\n{tree_text}"
            ),
        )

    async def _find_element(self, **kwargs: Any) -> ToolResult:
        """Find interactive elements with suggested Playwright selectors."""
        log.info("browser._find_element called")
        session = self._get_session(self._session_key(kwargs))
        page = await session.ensure_page()

        interactive = await AccessibilityExtractor.find_interactive_elements(
            page, max_items=60,
        )

        formatted = AccessibilityExtractor.format_interactive_list(interactive)

        return ToolResult(
            success=True,
            content=(
                f"URL: {session.current_url}\n"
                f"Found {len(interactive)} interactive element(s):\n\n"
                f"{formatted}\n\n"
                "Use the suggested selectors with click(role=..., text=...) "
                "or type(role=..., text=...) to interact."
            ),
        )

    # -- credential & login actions -------------------------------------------

    def _get_credential_store(self) -> CredentialStore:
        """Return or lazily create the credential store."""
        if not hasattr(self, "_cred_store"):
            self._cred_store = CredentialStore()
        return self._cred_store

    async def _credentials_store(self, **kwargs: Any) -> ToolResult:
        """Store credentials for a web app (encrypted)."""
        log.info("browser._credentials_store called", app_name=kwargs.get("app_name"))
        app_name = str(kwargs.get("app_name", "")).strip()
        url = str(kwargs.get("url", "")).strip()
        username = str(kwargs.get("username", "")).strip()
        password = str(kwargs.get("password", "")).strip()

        if not app_name:
            return ToolResult(success=False, error="'app_name' is required for credentials_store.")
        if not url:
            return ToolResult(success=False, error="'url' is required for credentials_store.")
        if not username:
            return ToolResult(success=False, error="'username' is required for credentials_store.")
        if not password:
            return ToolResult(success=False, error="'password' is required for credentials_store.")

        store = self._get_credential_store()
        session_id = str(kwargs.get("_session_id", "") or "").strip() or None

        summary = await store.store_credential(
            app_name=app_name,
            url=url,
            username=username,
            password=password,
            session_id=session_id,
        )

        encrypted_note = "encrypted (Fernet)" if store.is_encrypted else "obfuscated (Base64)"

        return ToolResult(
            success=True,
            content=(
                f"Credentials stored for '{app_name}':\n"
                f"  URL: {url}\n"
                f"  Username: {username}\n"
                f"  Password: {encrypted_note}\n\n"
                f"Use browser(action='login', app_name='{app_name}') to log in."
            ),
        )

    async def _credentials_list(self, **kwargs: Any) -> ToolResult:
        """List stored credentials (passwords masked)."""
        log.info("browser._credentials_list called")
        store = self._get_credential_store()
        creds = await store.list_credentials()

        if not creds:
            return ToolResult(
                success=True,
                content=(
                    "No browser credentials stored.\n"
                    "Use browser(action='credentials_store', app_name=..., url=..., "
                    "username=..., password=...) to add one."
                ),
            )

        lines: list[str] = [f"Stored credentials ({len(creds)}):"]
        for cred in creds:
            cookie_status = "cookies saved" if cred.get("has_cookies") else "no cookies"
            lines.append(
                f"\n  [{cred['app_name']}]\n"
                f"    URL: {cred['url']}\n"
                f"    Username: {cred['username']}\n"
                f"    Auth type: {cred['auth_type']}\n"
                f"    Status: {cookie_status}"
            )

        return ToolResult(success=True, content="\n".join(lines))

    async def _credentials_delete(self, **kwargs: Any) -> ToolResult:
        """Delete stored credentials."""
        log.info("browser._credentials_delete called", app_name=kwargs.get("app_name"))
        app_name = str(kwargs.get("app_name", "")).strip()
        if not app_name:
            return ToolResult(success=False, error="'app_name' is required for credentials_delete.")

        store = self._get_credential_store()
        deleted = await store.delete_credential(app_name)

        if deleted:
            return ToolResult(
                success=True,
                content=f"Credentials for '{app_name}' deleted (including saved cookies).",
            )
        return ToolResult(
            success=False,
            error=f"No credentials found for app '{app_name}'.",
        )

    async def _login(self, **kwargs: Any) -> ToolResult:
        """Automated login flow: restore cookies or fill login form.

        Login strategy:
        1. Look up credential by app_name
        2. If saved cookies exist, restore them and navigate to URL
           → check if we're logged in (URL didn't redirect to login)
        3. If no cookies or cookies expired:
           a. Navigate to login URL
           b. Use accessibility tree to find form fields
           c. Fill username + password
           d. Submit the form
           e. Wait and verify login success
           f. Save cookies for next time
        """
        app_name = str(kwargs.get("app_name", "")).strip()
        if not app_name:
            return ToolResult(success=False, error="'app_name' is required for login.")

        log.info("browser._login called", app_name=app_name)
        store = self._get_credential_store()
        cred = await store.get_credential(app_name)

        if not cred:
            return ToolResult(
                success=False,
                error=(
                    f"No credentials found for '{app_name}'. "
                    "Use browser(action='credentials_store', ...) first."
                ),
            )

        if cred.get("error"):
            return ToolResult(
                success=False,
                error=f"Cannot decrypt credentials for '{app_name}': {cred['error']}",
            )

        session = self._get_session(self._session_key(kwargs))
        login_url = cred["url"]
        username = cred["username"]
        password = cred["password"]
        saved_cookies = cred.get("cookies")
        cfg = get_config().tools.browser

        # Strategy 1: Try saved cookies
        if saved_cookies and cfg.cookie_persistence:
            log.info("Attempting cookie-based login", app_name=app_name)
            await session.ensure_page()

            # Load cookies before navigating
            await session.load_cookies(saved_cookies)
            nav_info = await session.navigate(login_url)

            # Check if we landed on the expected page (not redirected to login)
            current_url = session.current_url.rstrip("/")
            login_url_base = login_url.rstrip("/")

            # Heuristic: if URL changed significantly from login URL, cookies worked
            if current_url != login_url_base and not self._looks_like_login_page(current_url):
                title = await session.get_title()
                return ToolResult(
                    success=True,
                    content=(
                        f"Logged into '{app_name}' using saved cookies.\n"
                        f"URL: {current_url}\n"
                        f"Title: {title}\n\n"
                        "Use browser(action='observe') to see the page."
                    ),
                )
            else:
                log.info("Saved cookies expired, falling back to form login", app_name=app_name)

        # Strategy 2: Form-based login
        log.info("Performing form-based login", app_name=app_name)
        nav_info = await session.navigate(login_url)
        page = await session.ensure_page()

        # Use accessibility tree to find form fields
        interactive = await AccessibilityExtractor.find_interactive_elements(page, max_items=30)

        # Find username and password fields
        username_field = self._find_field(interactive, field_type="username")
        password_field = self._find_field(interactive, field_type="password")
        submit_button = self._find_field(interactive, field_type="submit")

        if not username_field:
            # Fallback: try common selectors
            username_field = {"role": "textbox", "name": "", "selector": ""}

        if not password_field:
            password_field = {"role": "textbox", "name": "", "selector": ""}

        # Fill the form
        filled_fields: list[str] = []

        # Fill username — try CSS selectors first (most reliable), then role-based
        username_filled = False
        for sel in [
            'input[type="email"]', 'input[name="username"]',
            'input[name="email"]', 'input[name="login"]',
            'input[type="text"]:not([type="password"])',
        ]:
            try:
                await session.type_text(sel, username, timeout=3000)
                filled_fields.append(f"username → {sel}")
                username_filled = True
                break
            except Exception:
                continue
        if not username_filled and username_field and username_field.get("name"):
            try:
                await session.type_by_role(
                    username_field["role"], name=username_field["name"], text=username,
                )
                filled_fields.append(f"username → {username_field['name']}")
                username_filled = True
            except Exception as e:
                log.warning("Failed to fill username by role", error=str(e))
        if not username_filled:
            log.warning("Could not fill username field with any strategy")

        # Fill password — try CSS selectors first, then role-based
        password_filled = False
        for sel in [
            'input[type="password"]', 'input[name="password"]',
        ]:
            try:
                await session.type_text(sel, password, timeout=3000)
                filled_fields.append(f"password → {sel}")
                password_filled = True
                break
            except Exception:
                continue
        if not password_filled and password_field and password_field.get("name"):
            try:
                await session.type_by_role(
                    password_field["role"], name=password_field["name"], text=password,
                )
                filled_fields.append(f"password → {password_field['name']}")
                password_filled = True
            except Exception as e:
                log.warning("Failed to fill password by role", error=str(e))
        if not password_filled:
            log.warning("Could not fill password field with any strategy")

        if not filled_fields:
            # Return diagnostic info instead of failing silently
            tree_text = await AccessibilityExtractor.extract_tree(page, max_depth=4, max_lines=50)
            return ToolResult(
                success=False,
                error=(
                    f"Could not find login form fields on {login_url}.\n"
                    f"Page structure:\n{tree_text}\n\n"
                    "You can manually fill the form using browser(action='type', ...) "
                    "and browser(action='click', ...) instead."
                ),
            )

        # Submit the form
        submitted = False
        if submit_button and submit_button.get("name"):
            try:
                await session.click_by_role(
                    submit_button["role"], name=submit_button["name"],
                )
                submitted = True
            except Exception:
                pass

        if not submitted:
            # Try pressing Enter or clicking common submit buttons
            for approach in ["enter_key", "submit_button_css"]:
                try:
                    if approach == "enter_key":
                        await session.press_key("Enter")
                        submitted = True
                        break
                    elif approach == "submit_button_css":
                        for sel in [
                            'button[type="submit"]', 'input[type="submit"]',
                            'button:has-text("Sign in")', 'button:has-text("Log in")',
                            'button:has-text("Login")',
                        ]:
                            try:
                                await session.click(sel, timeout=2000)
                                submitted = True
                                break
                            except Exception:
                                continue
                        if submitted:
                            break
                except Exception:
                    continue

        # Wait for navigation after submit
        await session.wait(cfg.login_verify_wait_seconds)

        # Verify login
        post_login_url = session.current_url
        title = await session.get_title()
        url_changed = post_login_url.rstrip("/") != login_url.rstrip("/")

        # Save cookies for next time
        if cfg.cookie_persistence:
            cookies = await session.save_cookies()
            if cookies:
                await store.save_cookies(app_name, cookies)
                log.info("Saved login cookies", app_name=app_name, count=len(cookies))

        status_msg = "Login appears successful" if url_changed else "Login submitted (verify manually)"

        return ToolResult(
            success=True,
            content=(
                f"{status_msg} for '{app_name}'.\n"
                f"URL: {post_login_url}\n"
                f"Title: {title}\n"
                f"Fields filled: {', '.join(filled_fields)}\n"
                f"Form submitted: {submitted}\n"
                f"URL changed: {url_changed}\n"
                f"Cookies saved: {cfg.cookie_persistence}\n\n"
                "Use browser(action='observe') to see the current page."
            ),
        )

    @staticmethod
    def _looks_like_login_page(url: str) -> bool:
        """Heuristic check if a URL looks like a login page."""
        login_indicators = [
            "/login", "/signin", "/sign-in", "/auth",
            "/sso", "/oauth", "/cas/login", "login.php",
            "signin", "authenticate",
        ]
        url_lower = url.lower()
        return any(indicator in url_lower for indicator in login_indicators)

    @staticmethod
    def _find_field(
        interactive: list[dict[str, str]],
        field_type: str,
    ) -> dict[str, str] | None:
        """Find a login-related field from interactive elements list.

        Args:
            interactive: List from AccessibilityExtractor.find_interactive_elements.
            field_type: One of 'username', 'password', 'submit'.
        """
        if field_type == "username":
            # Look for textbox with name hints
            username_hints = [
                "email", "username", "user", "login", "account",
                "e-mail", "user name", "sign in",
            ]
            for elem in interactive:
                if elem["role"] in ("textbox", "searchbox"):
                    name_lower = elem["name"].lower()
                    if any(hint in name_lower for hint in username_hints):
                        return elem
            # Fall back to first textbox
            for elem in interactive:
                if elem["role"] == "textbox":
                    return elem

        elif field_type == "password":
            password_hints = ["password", "pass", "pwd", "secret"]
            for elem in interactive:
                if elem["role"] == "textbox":
                    name_lower = elem["name"].lower()
                    if any(hint in name_lower for hint in password_hints):
                        return elem

        elif field_type == "submit":
            submit_hints = [
                "sign in", "log in", "login", "submit", "continue",
                "next", "enter", "go",
            ]
            for elem in interactive:
                if elem["role"] == "button":
                    name_lower = elem["name"].lower()
                    if any(hint in name_lower for hint in submit_hints):
                        return elem
            # Fall back to first button
            for elem in interactive:
                if elem["role"] == "button":
                    return elem

        return None

    async def _detect_login_hint(
        self,
        url: str,
        interactive: list[dict[str, str]],
    ) -> str | None:
        """Detect if the current page is a login form and matching credentials exist.

        Returns an actionable hint string, or None if not a login page or no
        credentials match.
        """
        # Check if the page has a password field — strongest login indicator
        has_password_field = any(
            elem["role"] == "textbox"
            and any(
                kw in elem["name"].lower()
                for kw in ("password", "pass", "pwd")
            )
            for elem in interactive
        )

        # Also check for login-related submit buttons
        login_button_hints = [
            "sign in", "log in", "login", "sign-in", "log-in",
        ]
        has_login_button = any(
            elem["role"] == "button"
            and any(kw in elem["name"].lower() for kw in login_button_hints)
            for elem in interactive
        )

        # Also check if the URL looks like a login page
        url_looks_login = self._looks_like_login_page(url)

        # Need at least one strong signal
        is_login_page = has_password_field or (has_login_button and url_looks_login)

        if not is_login_page:
            return None

        # Check stored credentials for a matching domain
        try:
            from urllib.parse import urlparse

            current_domain = urlparse(url).netloc.lower()
            if not current_domain:
                return None

            store = self._get_credential_store()
            creds = await store.list_credentials()

            for cred in creds:
                cred_domain = urlparse(cred.get("url", "")).netloc.lower()
                # Match on domain (e.g. "fricv3.filrougecapital.com" matches)
                if cred_domain and (
                    cred_domain == current_domain
                    or current_domain.endswith("." + cred_domain)
                    or cred_domain.endswith("." + current_domain)
                ):
                    app_name = cred["app_name"]
                    log.info(
                        "Login form detected with matching credentials",
                        url=url,
                        app_name=app_name,
                        cred_url=cred.get("url"),
                    )
                    return (
                        "--- Action Required ---\n"
                        f"LOGIN FORM DETECTED on this page.\n"
                        f"Stored credentials found for '{app_name}' "
                        f"(user: {cred.get('username', '?')}).\n"
                        f"→ Use browser(action='login', app_name='{app_name}') "
                        f"to log in automatically."
                    )

            # Login page detected but no matching credentials
            log.info(
                "Login form detected but no matching credentials",
                url=url,
                current_domain=current_domain,
                stored_domains=[
                    urlparse(c.get("url", "")).netloc for c in creds
                ],
            )
            return (
                "--- Action Required ---\n"
                "LOGIN FORM DETECTED on this page.\n"
                "No stored credentials match this domain.\n"
                "→ Use browser(action='credentials_store', app_name=..., "
                "url=..., username=..., password=...) to store credentials,\n"
                "  then browser(action='login', app_name=...) to log in."
            )

        except Exception as e:
            log.warning("Failed to check credentials for login hint", error=str(e))
            return None

    # -- network capture actions ----------------------------------------------

    async def _network_start(self, **kwargs: Any) -> ToolResult:
        """Start recording network traffic."""
        log.info("browser._network_start called")
        session = self._get_session(self._session_key(kwargs))
        network = session.network

        if network.is_recording:
            return ToolResult(
                success=True,
                content=(
                    f"Network recording is already active. "
                    f"{network.capture_count} call(s) captured so far."
                ),
            )

        network.start_recording()
        return ToolResult(
            success=True,
            content="Network recording started. All XHR/fetch API calls will be captured.",
        )

    async def _network_stop(self, **kwargs: Any) -> ToolResult:
        """Stop recording network traffic."""
        log.info("browser._network_stop called")
        session = self._get_session(self._session_key(kwargs))
        network = session.network

        if not network.is_recording:
            return ToolResult(
                success=True,
                content=f"Network recording was already stopped. {network.capture_count} call(s) in buffer.",
            )

        network.stop_recording()
        return ToolResult(
            success=True,
            content=(
                f"Network recording stopped. "
                f"{network.capture_count} API call(s) in buffer.\n"
                "Use browser(action='network_list') to review or "
                "browser(action='network_capture') to store in APIs memory."
            ),
        )

    async def _network_list(self, **kwargs: Any) -> ToolResult:
        """List captured API calls."""
        log.info("browser._network_list called")
        session = self._get_session(self._session_key(kwargs))
        network = session.network

        listing = network.format_capture_list(max_items=50)
        recording_status = "Recording" if network.is_recording else "Stopped"

        return ToolResult(
            success=True,
            content=f"Network status: {recording_status}\n\n{listing}",
        )

    async def _network_capture(self, **kwargs: Any) -> ToolResult:
        """Analyze captured traffic and store API patterns in APIs memory."""
        log.info("browser._network_capture called")
        session = self._get_session(self._session_key(kwargs))
        network = session.network

        summaries = network.summarize_apis()
        if not summaries:
            return ToolResult(
                success=True,
                content=(
                    "No API endpoints to capture. "
                    "Browse some pages first to generate API traffic, "
                    "then use network_capture to store the patterns."
                ),
            )

        # Store each discovered API group in the session manager
        from captain_claw.session import get_session_manager

        sm = get_session_manager()
        session_id = str(kwargs.get("_session_id", "") or "").strip() or None

        stored: list[str] = []
        for summary in summaries:
            try:
                endpoints_json = _json.dumps(summary.endpoints, indent=2)
                entry = await sm.create_api(
                    name=summary.name,
                    base_url=summary.base_url,
                    endpoints=endpoints_json,
                    auth_type=summary.auth_type if summary.auth_type != "none" else None,
                    credentials=summary.auth_header_value or None,
                    description=summary.description,
                    purpose=f"Auto-captured from browser session on {summary.base_url}",
                    tags="browser-captured,auto-discovered",
                    source_session=session_id,
                )
                stored.append(
                    f"  API #{entry.id[:8]}: {summary.name} ({summary.base_url})\n"
                    f"    {len(summary.endpoints)} endpoint(s), "
                    f"auth={summary.auth_type}, "
                    f"{summary.sample_requests} request(s) observed"
                )
            except Exception as e:
                stored.append(f"  FAILED: {summary.name} — {e}")
                log.error("Failed to store captured API", name=summary.name, error=str(e))

        return ToolResult(
            success=True,
            content=(
                f"Captured and stored {len(summaries)} API(s) in memory:\n\n"
                + "\n".join(stored)
                + "\n\nUse apis(action='list') to see all stored APIs."
            ),
        )

    async def _network_clear(self, **kwargs: Any) -> ToolResult:
        """Discard all captured network data."""
        log.info("browser._network_clear called")
        session = self._get_session(self._session_key(kwargs))
        count = session.network.capture_count
        session.network.clear()
        return ToolResult(
            success=True,
            content=f"Cleared {count} captured network request(s).",
        )

    # -- API replay actions ----------------------------------------------------

    async def _api_replay(self, **kwargs: Any) -> ToolResult:
        """Execute a captured API directly via HTTP — skip the browser.

        Looks up the API in the APIs memory, resolves auth headers from the
        stored credentials or the browser credential store, then executes
        the request via ``ApiReplayEngine``.

        Increments the API's usage counter on success.
        """
        log.info("browser._api_replay called", api_id=kwargs.get("api_id"), endpoint=kwargs.get("endpoint"), method=kwargs.get("method"))
        api_id = str(kwargs.get("api_id", "")).strip()
        endpoint = str(kwargs.get("endpoint", "")).strip()
        method = str(kwargs.get("method", "GET")).strip().upper()

        if not api_id:
            return ToolResult(
                success=False,
                error=(
                    "'api_id' is required for api_replay. "
                    "Use apis(action='list') to see available APIs, then pass the ID or '#N' index."
                ),
            )
        if not endpoint:
            return ToolResult(
                success=False,
                error=(
                    "'endpoint' is required for api_replay. "
                    "Example: endpoint='/rest/api/2/search'. "
                    "Use apis(action='info', api_id=...) to see available endpoints."
                ),
            )

        # Look up the API entry via SessionManager
        from captain_claw.session import get_session_manager

        sm = get_session_manager()
        api_entry = await sm.select_api(api_id)

        if not api_entry:
            return ToolResult(
                success=False,
                error=(
                    f"API '{api_id}' not found. "
                    "Use apis(action='list') to see available APIs."
                ),
            )

        # Parse optional query_params and body_json
        query_params: dict[str, str] | None = None
        body_json: Any | None = None

        raw_params = str(kwargs.get("query_params", "")).strip()
        if raw_params:
            try:
                query_params = _json.loads(raw_params)
            except _json.JSONDecodeError as e:
                return ToolResult(
                    success=False,
                    error=f"Invalid JSON in 'query_params': {e}",
                )

        raw_body = str(kwargs.get("body_json", "")).strip()
        if raw_body:
            try:
                body_json = _json.loads(raw_body)
            except _json.JSONDecodeError as e:
                return ToolResult(
                    success=False,
                    error=f"Invalid JSON in 'body_json': {e}",
                )

        # Resolve auth headers
        # Priority: API entry credentials → browser credential store → no auth
        auth_headers = ApiReplayEngine.resolve_auth_headers(
            api_entry.auth_type,
            api_entry.credentials,
        )

        if not auth_headers:
            # Try browser credential store for matching domain
            try:
                from urllib.parse import urlparse

                domain = urlparse(api_entry.base_url).hostname or ""
                store = self._get_credential_store()
                creds = await store.list_credentials()
                for cred in creds:
                    cred_domain = urlparse(cred.get("url", "")).hostname or ""
                    if cred_domain == domain:
                        # Found matching credential — get the full entry with password
                        full_cred = await store.get_credential(cred["app_name"])
                        if full_cred and not full_cred.get("error"):
                            # Use stored cookies to extract auth tokens would be complex;
                            # for now just note we found matching creds
                            log.info(
                                "Found browser credentials for API domain",
                                domain=domain,
                                app_name=cred["app_name"],
                            )
                        break
            except Exception:
                pass  # Non-critical — proceed without auth

        # Check endpoint definition from stored API
        ep_info = ApiReplayEngine.find_endpoint_in_api(
            api_entry.endpoints, endpoint, method,
        )
        if ep_info:
            log.info(
                "Matched endpoint definition",
                path=ep_info.get("path"),
                method=ep_info.get("method"),
            )

        # Execute the API call
        result = await ApiReplayEngine.replay(
            base_url=api_entry.base_url,
            endpoint=endpoint,
            method=method,
            headers=auth_headers or None,
            query_params=query_params,
            body_json=body_json,
        )

        # Increment usage counter on success
        if result.success:
            try:
                await sm.increment_api_usage(api_entry.id)
            except Exception:
                pass  # Non-critical

        summary = result.to_summary(max_body=3000)

        # Add helpful context
        lines = [summary]
        if not result.success and result.status_code in (401, 403):
            lines.append(
                "\n⚠️  Authentication failed. The stored token may be expired. "
                "Try browser(action='login', app_name=...) to refresh credentials, "
                "then browser(action='network_capture') to capture a fresh token."
            )
        elif result.success:
            lines.append(
                f"\n✅ API call successful. Used API: {api_entry.name} ({api_entry.base_url})"
            )

        return ToolResult(
            success=result.success,
            content="\n".join(lines),
            error=result.error if not result.success else None,
        )

    async def _api_test(self, **kwargs: Any) -> ToolResult:
        """Test a captured API endpoint — lighter preview of api_replay.

        Shows the API info + a dry run or actual test call with truncated
        response, useful for verifying an API is working before using
        it in production workflows.
        """
        log.info("browser._api_test called", api_id=kwargs.get("api_id"), endpoint=kwargs.get("endpoint"))
        api_id = str(kwargs.get("api_id", "")).strip()
        if not api_id:
            return ToolResult(
                success=False,
                error=(
                    "'api_id' is required for api_test. "
                    "Use apis(action='list') to see available APIs."
                ),
            )

        # Look up the API
        from captain_claw.session import get_session_manager

        sm = get_session_manager()
        api_entry = await sm.select_api(api_id)

        if not api_entry:
            return ToolResult(
                success=False,
                error=f"API '{api_id}' not found. Use apis(action='list') to see available APIs.",
            )

        endpoint = str(kwargs.get("endpoint", "")).strip()
        method = str(kwargs.get("method", "GET")).strip().upper()

        # Build API info summary
        lines: list[str] = [
            f"API: {api_entry.name}",
            f"Base URL: {api_entry.base_url}",
            f"Auth type: {api_entry.auth_type or 'none'}",
            f"Has credentials: {'yes' if api_entry.credentials else 'no'}",
            f"Use count: {api_entry.use_count}",
            "",
        ]

        # Show available endpoints
        ep_list = ApiReplayEngine.format_endpoints_list(api_entry.endpoints)
        lines.append("Available endpoints:")
        lines.append(ep_list)
        lines.append("")

        # If an endpoint was specified, do a test call
        if endpoint:
            lines.append(f"--- Test call: {method} {endpoint} ---")

            # Parse optional params
            query_params: dict[str, str] | None = None
            body_json: Any | None = None

            raw_params = str(kwargs.get("query_params", "")).strip()
            if raw_params:
                try:
                    query_params = _json.loads(raw_params)
                except _json.JSONDecodeError:
                    pass

            raw_body = str(kwargs.get("body_json", "")).strip()
            if raw_body:
                try:
                    body_json = _json.loads(raw_body)
                except _json.JSONDecodeError:
                    pass

            auth_headers = ApiReplayEngine.resolve_auth_headers(
                api_entry.auth_type,
                api_entry.credentials,
            )

            result = await ApiReplayEngine.replay(
                base_url=api_entry.base_url,
                endpoint=endpoint,
                method=method,
                headers=auth_headers or None,
                query_params=query_params,
                body_json=body_json,
            )

            lines.append(result.to_summary(max_body=1000))
        else:
            lines.append(
                "No endpoint specified — showing API info only. "
                "Add endpoint='/path/here' to run a test call."
            )

        return ToolResult(success=True, content="\n".join(lines))

    # -- multi-app session actions ---------------------------------------------

    async def _switch_app(self, **kwargs: Any) -> ToolResult:
        """Switch the active browser session to a different app.

        Creates a new isolated browser context for the app if one doesn't
        exist yet.  Each app context has its own cookies, storage, and page.
        """
        log.info("browser._switch_app called", app_name=kwargs.get("app_name"))
        app_name = str(kwargs.get("app_name", "")).strip()
        if not app_name:
            return ToolResult(
                success=False,
                error=(
                    "'app_name' is required for switch_app. "
                    "Examples: 'jira', 'confluence', 'github'."
                ),
            )

        session = self._get_session(self._session_key(kwargs))

        # Create context if it doesn't exist
        if app_name not in session._app_contexts:
            await session.create_app_context(app_name)

        # Switch to it
        info = await session.switch_app(app_name)

        # If the app has stored credentials and the page is blank, suggest login
        hint = ""
        url = info.get("url", "")
        if not url or url == "about:blank":
            try:
                store = self._get_credential_store()
                cred = await store.get_credential(app_name)
                if cred and not cred.get("error"):
                    hint = (
                        f"\n\nCredentials found for '{app_name}'. "
                        f"Use browser(action='login', app_name='{app_name}') to log in."
                    )
            except Exception:
                pass

        sessions = session.list_app_sessions()
        session_list = ", ".join(
            f"{s['app_name']}{'*' if s['active'] == 'yes' else ''}"
            for s in sessions
        )

        return ToolResult(
            success=True,
            content=(
                f"Switched to app session: '{app_name}'\n"
                f"URL: {info.get('url', 'about:blank')}\n"
                f"Title: {info.get('title', '(empty)')}\n"
                f"Active sessions: [{session_list}] (* = current)"
                + hint
            ),
        )

    async def _list_sessions(self, **kwargs: Any) -> ToolResult:
        """List all active app sessions."""
        log.info("browser._list_sessions called")
        session = self._get_session(self._session_key(kwargs))
        sessions = session.list_app_sessions()

        if not sessions:
            return ToolResult(
                success=True,
                content=(
                    "No multi-app sessions active.\n"
                    f"Default context URL: {session.current_url}\n\n"
                    "Use browser(action='switch_app', app_name='...') to create an isolated session."
                ),
            )

        lines: list[str] = [f"Active app sessions ({len(sessions)}):"]
        for s in sessions:
            active_marker = " ← active" if s["active"] == "yes" else ""
            lines.append(f"  [{s['app_name']}]{active_marker}")
            lines.append(f"    URL: {s['url']}")

        lines.append("")
        lines.append(
            "Use browser(action='switch_app', app_name='...') to switch between sessions."
        )

        return ToolResult(success=True, content="\n".join(lines))

    # -- workflow recording actions ------------------------------------------

    def _get_workflow_recorder(self, session_id: str = "default") -> WorkflowRecorder | None:
        return _WORKFLOW_RECORDERS.get(session_id or "default")

    def _ensure_workflow_recorder(self, session_id: str = "default") -> WorkflowRecorder:
        session_id = session_id or "default"
        if session_id not in _WORKFLOW_RECORDERS:
            _WORKFLOW_RECORDERS[session_id] = WorkflowRecorder()
        return _WORKFLOW_RECORDERS[session_id]

    async def _workflow_record_start(self, **kwargs: Any) -> ToolResult:
        sk = self._session_key(kwargs)
        session = self._get_session(sk)
        page = await session.ensure_page()

        recorder = self._ensure_workflow_recorder(sk)
        if not recorder._page:
            await recorder.attach(page)

        if recorder.is_recording:
            return ToolResult(
                success=True,
                content=(
                    f"Already recording. {recorder.step_count} steps captured so far.\n"
                    "Use workflow_record_stop to stop."
                ),
            )

        recorder.start_recording()
        return ToolResult(
            success=True,
            content=(
                "Workflow recording started.\n"
                "Interact with the browser — clicks, typing, and navigation will be captured.\n"
                "Use browser(action='workflow_record_stop') when done."
            ),
        )

    async def _workflow_record_stop(self, **kwargs: Any) -> ToolResult:
        recorder = self._get_workflow_recorder(self._session_key(kwargs))
        if recorder is None or not recorder.is_recording:
            return ToolResult(
                success=False,
                error="No recording in progress. Start with browser(action='workflow_record_start').",
            )

        recorder.stop_recording()
        summary = recorder.summary()
        return ToolResult(
            success=True,
            content=(
                f"Recording stopped.\n\n{summary}\n\n"
                "Use browser(action='workflow_save', workflow_name='...') to save this workflow.\n"
                "Use browser(action='workflow_record_start') to continue recording more steps."
            ),
        )

    async def _workflow_save(self, **kwargs: Any) -> ToolResult:
        from captain_claw.session import get_session_manager

        sk = self._session_key(kwargs)
        recorder = self._get_workflow_recorder(sk)
        if recorder is None or recorder.step_count == 0:
            return ToolResult(
                success=False,
                error="No recorded steps to save. Record a workflow first.",
            )

        name = str(kwargs.get("workflow_name", "")).strip()
        if not name:
            return ToolResult(
                success=False,
                error="'workflow_name' is required for workflow_save.",
            )

        description = str(kwargs.get("workflow_description", "")).strip()
        app_name = str(kwargs.get("app_name", "")).strip()

        # determine start_url from first step
        steps = recorder.steps_as_dicts()
        start_url = steps[0]["url"] if steps else ""

        # variable parameterisation
        variables_raw = str(kwargs.get("workflow_variables", "")).strip()
        variables: list[dict[str, Any]] = []
        if variables_raw:
            try:
                variables = _json.loads(variables_raw)
            except _json.JSONDecodeError:
                return ToolResult(
                    success=False,
                    error=f"Invalid JSON in workflow_variables: {variables_raw[:200]}",
                )

        # apply variable placeholders to steps
        for var_def in variables:
            idx = var_def.get("step_index")
            fld = var_def.get("field", "value")
            vname = var_def.get("name", "")
            if idx is not None and 0 <= idx < len(steps) and vname:
                old_val = steps[idx].get(fld, "")
                if old_val:
                    steps[idx][fld] = "{{" + vname + "}}"

        sm = get_session_manager()
        entry = await sm.create_workflow(
            name=name,
            description=description,
            app_name=app_name,
            start_url=start_url,
            steps=_json.dumps(steps),
            variables=_json.dumps(variables),
        )

        # clear recorder after saving
        recorder.clear()

        return ToolResult(
            success=True,
            content=(
                f"Workflow saved: {entry.name}\n"
                f"ID: {entry.id}\n"
                f"Steps: {len(steps)}\n"
                f"Variables: {len(variables)}\n"
                f"Start URL: {start_url}\n\n"
                f"Replay with: browser(action='workflow_run', workflow_id='{entry.id}')"
            ),
        )

    async def _workflow_list(self, **kwargs: Any) -> ToolResult:
        from captain_claw.session import get_session_manager

        sm = get_session_manager()
        app_name = str(kwargs.get("app_name", "")).strip() or None
        entries = await sm.list_workflows(app_name=app_name)

        if not entries:
            return ToolResult(
                success=True,
                content="No saved workflows.\nRecord one with browser(action='workflow_record_start').",
            )

        lines: list[str] = [f"Saved workflows ({len(entries)}):"]
        for i, e in enumerate(entries, 1):
            step_count = len(_json.loads(e.steps)) if e.steps else 0
            var_count = len(_json.loads(e.variables)) if e.variables else 0
            lines.append(f"  #{i} {e.name} (id: {e.id[:8]}...)")
            if e.app_name:
                lines.append(f"      App: {e.app_name}")
            lines.append(f"      Steps: {step_count}, Variables: {var_count}, Used: {e.use_count}x")
            if e.description:
                lines.append(f"      {e.description}")

        return ToolResult(success=True, content="\n".join(lines))

    async def _workflow_show(self, **kwargs: Any) -> ToolResult:
        from captain_claw.session import get_session_manager

        wf_id = str(kwargs.get("workflow_id", "")).strip()
        if not wf_id:
            return ToolResult(success=False, error="'workflow_id' is required.")

        sm = get_session_manager()
        entry = await sm.select_workflow(wf_id)
        if not entry:
            return ToolResult(success=False, error=f"Workflow not found: {wf_id}")

        steps = _json.loads(entry.steps)
        variables = _json.loads(entry.variables)

        lines: list[str] = [
            f"Workflow: {entry.name}",
            f"ID: {entry.id}",
        ]
        if entry.description:
            lines.append(f"Description: {entry.description}")
        if entry.app_name:
            lines.append(f"App: {entry.app_name}")
        lines.append(f"Start URL: {entry.start_url}")
        lines.append(f"Used: {entry.use_count}x")
        lines.append("")

        # Steps
        lines.append(f"Steps ({len(steps)}):")
        for s in steps:
            val_part = f" → {s['value'][:60]}" if s.get("value") else ""
            sel_parts: list[str] = []
            sels = s.get("selectors", {})
            if sels.get("role") and sels.get("role_name"):
                sel_parts.append(f'role={sels["role"]}("{sels["role_name"]}")')
            elif sels.get("text"):
                sel_parts.append(f'text="{sels["text"][:40]}"')
            elif sels.get("css"):
                sel_parts.append(f'css="{sels["css"][:40]}"')
            sel_str = f" [{', '.join(sel_parts)}]" if sel_parts else ""
            lines.append(f"  {s['seq']}. {s['action']}{sel_str}{val_part}")

        # Variables
        if variables:
            lines.append("")
            lines.append(f"Variables ({len(variables)}):")
            for v in variables:
                desc = f" — {v['description']}" if v.get("description") else ""
                lines.append(f"  {v['name']}: step {v.get('step_index', '?')}, field={v.get('field', 'value')}{desc}")

        return ToolResult(success=True, content="\n".join(lines))

    async def _workflow_run(self, **kwargs: Any) -> ToolResult:
        from captain_claw.session import get_session_manager

        wf_id = str(kwargs.get("workflow_id", "")).strip()
        if not wf_id:
            return ToolResult(success=False, error="'workflow_id' is required.")

        sm = get_session_manager()
        entry = await sm.select_workflow(wf_id)
        if not entry:
            return ToolResult(success=False, error=f"Workflow not found: {wf_id}")

        steps = _json.loads(entry.steps)
        if not steps:
            return ToolResult(success=False, error="Workflow has no steps.")

        # parse variable values
        variables: dict[str, str] = {}
        vars_raw = str(kwargs.get("workflow_variables", "")).strip()
        if vars_raw:
            try:
                variables = _json.loads(vars_raw)
            except _json.JSONDecodeError:
                return ToolResult(
                    success=False,
                    error=f"Invalid JSON in workflow_variables: {vars_raw[:200]}",
                )

        session = self._get_session(self._session_key(kwargs))

        # navigate to start_url first (with variable substitution)
        start_url = entry.start_url
        for vname, vval in variables.items():
            start_url = start_url.replace("{{" + vname + "}}", vval)
        if start_url:
            await session.navigate(start_url)

        # replay
        result = await WorkflowReplayEngine.replay(
            session=session,
            steps=steps,
            variables=variables,
        )

        # increment usage on success
        if result.success:
            await sm.increment_workflow_usage(entry.id)

        return ToolResult(
            success=result.success,
            content=result.to_summary(),
            error=result.error if not result.success else None,
        )

    async def _workflow_delete(self, **kwargs: Any) -> ToolResult:
        from captain_claw.session import get_session_manager

        wf_id = str(kwargs.get("workflow_id", "")).strip()
        if not wf_id:
            return ToolResult(success=False, error="'workflow_id' is required.")

        sm = get_session_manager()
        # resolve first to support #index and fuzzy name
        entry = await sm.select_workflow(wf_id)
        if not entry:
            return ToolResult(success=False, error=f"Workflow not found: {wf_id}")

        deleted = await sm.delete_workflow(entry.id)
        if deleted:
            return ToolResult(success=True, content=f"Deleted workflow: {entry.name} ({entry.id})")
        return ToolResult(success=False, error=f"Failed to delete workflow: {wf_id}")
