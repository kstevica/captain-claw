"""Browser session lifecycle manager using Playwright.

Manages a single Playwright browser instance with async lifecycle.
One instance per Captain Claw session.  Persists cookies and state
between tool calls.  Handles startup, page management, and cleanup.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from captain_claw.config import BrowserToolConfig, get_config
from captain_claw.logging import get_logger
from captain_claw.tools.browser_network import NetworkInterceptor

log = get_logger(__name__)

# ---------- optional dependency guard ----------------------------------------

try:
    from playwright.async_api import (
        async_playwright,
        Browser,
        BrowserContext,
        Page,
        Playwright,
    )

    _HAS_PLAYWRIGHT = True
except ImportError:  # pragma: no cover
    _HAS_PLAYWRIGHT = False
    async_playwright = None  # type: ignore[assignment,misc]
    Browser = None  # type: ignore[assignment,misc]
    BrowserContext = None  # type: ignore[assignment,misc]
    Page = None  # type: ignore[assignment,misc]
    Playwright = None  # type: ignore[assignment,misc]


def has_playwright() -> bool:
    """Return True if Playwright is installed."""
    return _HAS_PLAYWRIGHT


# ---------- BrowserSession ---------------------------------------------------


class BrowserSession:
    """Manages a Playwright browser instance with async lifecycle.

    One instance per Captain Claw session.  Persists cookies and state
    between tool calls.  Handles startup, page management, and cleanup.

    Supports **multi-app sessions**: each named app (e.g. ``jira``,
    ``confluence``) gets its own ``BrowserContext`` + ``Page`` with
    isolated cookies.  The ``_active_app`` tracks which context is
    currently in use.  ``switch_app()`` swaps the active context.
    The default (unnamed) context is keyed as ``"_default"``.
    """

    _DEFAULT_APP = "_default"

    def __init__(self, config: BrowserToolConfig | None = None) -> None:
        self._config = config or get_config().tools.browser
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._context: Any | None = None
        self._page: Any | None = None
        self._started_at: float = 0.0
        self._network = NetworkInterceptor(
            max_captures=self._config.network_max_captures,
            max_body_bytes=self._config.network_max_body_bytes,
            filter_static=self._config.network_filter_static,
        )

        # Multi-app session state
        self._app_contexts: dict[str, dict[str, Any]] = {}  # app_name → {context, page, url}
        self._active_app: str = self._DEFAULT_APP

    # -- lifecycle ------------------------------------------------------------

    async def start(self, headless: bool | None = None) -> None:
        """Launch browser if not already running."""
        log.info("BrowserSession.start called", headless=headless)
        if not _HAS_PLAYWRIGHT:
            log.error("Playwright not installed — cannot start browser")
            raise RuntimeError(
                "Playwright is not installed. "
                "Install it with: pip install 'captain-claw[browser]' && playwright install chromium"
            )

        if self.is_alive:
            log.info("Browser already running, skipping start")
            return  # already running

        effective_headless = headless if headless is not None else self._config.headless
        log.info("Launching Playwright chromium", headless=effective_headless)

        self._playwright = await async_playwright().start()
        log.info("Playwright started")
        self._browser = await self._playwright.chromium.launch(headless=effective_headless)
        log.info("Chromium browser launched")

        context_kwargs: dict[str, Any] = {
            "viewport": {
                "width": self._config.viewport_width,
                "height": self._config.viewport_height,
            },
        }
        if self._config.user_agent:
            context_kwargs["user_agent"] = self._config.user_agent

        self._context = await self._browser.new_context(**context_kwargs)
        self._page = await self._context.new_page()
        self._started_at = time.time()

        # Attach network interceptor and auto-start if configured
        if self._config.network_capture_enabled:
            await self._network.attach(self._page)
            if self._config.network_auto_record:
                self._network.start_recording()

        log.info(
            "Browser started",
            headless=effective_headless,
            viewport=f"{self._config.viewport_width}x{self._config.viewport_height}",
            network_capture=self._config.network_capture_enabled,
        )

    @property
    def network(self) -> NetworkInterceptor:
        """Return the network interceptor."""
        return self._network

    async def close(self) -> None:
        """Shut down browser and clean up resources."""
        log.info("BrowserSession.close called")
        errors: list[str] = []

        # Detach network interceptor first
        try:
            await self._network.detach()
        except Exception as e:
            errors.append(f"network detach: {e}")

        if self._page is not None:
            try:
                await self._page.close()
            except Exception as e:
                errors.append(f"page close: {e}")
            self._page = None

        if self._context is not None:
            try:
                await self._context.close()
            except Exception as e:
                errors.append(f"context close: {e}")
            self._context = None

        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception as e:
                errors.append(f"browser close: {e}")
            self._browser = None

        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as e:
                errors.append(f"playwright stop: {e}")
            self._playwright = None

        self._started_at = 0.0

        if errors:
            log.warning("Browser close had errors", errors=errors)
        else:
            log.info("Browser closed")

    # -- properties -----------------------------------------------------------

    @property
    def is_alive(self) -> bool:
        """Return True if the browser is running and responsive."""
        return (
            self._browser is not None
            and self._page is not None
            and not self._browser.is_connected() is False
        )

    @property
    def current_url(self) -> str:
        """Return the URL of the current page."""
        if self._page is not None:
            try:
                return self._page.url
            except Exception:
                return ""
        return ""

    @property
    def uptime_seconds(self) -> float:
        """Return how long the browser has been running."""
        if self._started_at:
            return time.time() - self._started_at
        return 0.0

    # -- page management ------------------------------------------------------

    async def ensure_page(self) -> Any:
        """Return the active page, starting the browser if needed."""
        if not self.is_alive:
            log.info("Browser not alive, auto-starting")
            await self.start()
        log.debug("ensure_page returning page", url=self.current_url)
        return self._page

    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> dict[str, Any]:
        """Navigate to a URL and wait for the page to load.

        Returns a dict with navigation info.
        """
        log.info("BrowserSession.navigate called", url=url, wait_until=wait_until)
        page = await self.ensure_page()

        log.info("Calling page.goto", url=url, timeout=30000)
        response = await page.goto(url, wait_until=wait_until, timeout=30000)

        status = response.status if response else 0
        ok = response.ok if response else False

        # Wait a bit for dynamic content
        await asyncio.sleep(self._config.default_wait_seconds)

        title = await page.title()

        log.info("Navigated", url=url, status=status, title=title)

        return {
            "url": page.url,
            "title": title,
            "status": status,
            "ok": ok,
        }

    async def screenshot(
        self,
        path: Path,
        full_page: bool = False,
        quality: int | None = None,
    ) -> bytes:
        """Take a screenshot and save it to *path*.

        Returns the raw JPEG bytes.
        """
        page = await self.ensure_page()

        effective_quality = quality or self._config.screenshot_jpeg_quality
        path.parent.mkdir(parents=True, exist_ok=True)

        screenshot_bytes: bytes = await page.screenshot(
            path=str(path),
            type="jpeg",
            quality=effective_quality,
            full_page=full_page,
        )

        log.info("Screenshot taken", path=str(path), size=len(screenshot_bytes))
        return screenshot_bytes

    async def click(self, selector: str, timeout: int = 10000) -> None:
        """Click an element by CSS selector or text-based selector."""
        page = await self.ensure_page()
        await page.click(selector, timeout=timeout)
        await asyncio.sleep(0.5)  # brief pause after click
        log.info("Clicked", selector=selector)

    async def click_by_text(
        self, text: str, exact: bool = False, nth: int | None = None, timeout: int = 10000,
    ) -> None:
        """Click an element by its visible text content.

        Args:
            text: Visible text to match.
            exact: Require exact text match (default: substring).
            nth: Zero-based index when multiple elements match (for React apps
                 with duplicate labels).  ``None`` = first match.
            timeout: Max wait in ms.
        """
        page = await self.ensure_page()
        locator = page.get_by_text(text, exact=exact)
        if nth is not None:
            locator = locator.nth(nth)
        await locator.click(timeout=timeout)
        await asyncio.sleep(0.5)
        log.info("Clicked by text", text=text, nth=nth)

    async def click_by_role(
        self, role: str, name: str = "", nth: int | None = None, timeout: int = 10000,
    ) -> None:
        """Click an element by ARIA role and optional name.

        Args:
            role: ARIA role (e.g. ``button``, ``link``, ``textbox``).
            name: Accessible name filter.
            nth: Zero-based index when multiple elements match.
            timeout: Max wait in ms.
        """
        page = await self.ensure_page()
        kwargs: dict[str, Any] = {}
        if name:
            kwargs["name"] = name
        locator = page.get_by_role(role, **kwargs)
        if nth is not None:
            locator = locator.nth(nth)
        await locator.click(timeout=timeout)
        await asyncio.sleep(0.5)
        log.info("Clicked by role", role=role, name=name, nth=nth)

    async def type_text(
        self, selector: str, text: str, delay: int = 50, timeout: int = 10000
    ) -> None:
        """Type text into an element identified by selector."""
        page = await self.ensure_page()
        await page.fill(selector, text, timeout=timeout)
        log.info("Typed text", selector=selector, chars=len(text))

    async def type_by_role(
        self, role: str, name: str, text: str, nth: int | None = None, timeout: int = 10000,
    ) -> None:
        """Type text into an element identified by ARIA role and name.

        Args:
            role: ARIA role (e.g. ``textbox``).
            name: Accessible name filter.
            text: Text to enter.
            nth: Zero-based index when multiple elements match.
            timeout: Max wait in ms.
        """
        page = await self.ensure_page()
        kwargs: dict[str, Any] = {}
        if name:
            kwargs["name"] = name
        locator = page.get_by_role(role, **kwargs)
        if nth is not None:
            locator = locator.nth(nth)
        await locator.fill(text, timeout=timeout)
        log.info("Typed by role", role=role, name=name, nth=nth, chars=len(text))

    async def press_key(self, key: str) -> None:
        """Press a keyboard key (e.g. 'Enter', 'Tab', 'Escape')."""
        page = await self.ensure_page()
        await page.keyboard.press(key)
        log.info("Pressed key", key=key)

    async def scroll(self, direction: str = "down", amount: int = 500) -> None:
        """Scroll the page up or down."""
        page = await self.ensure_page()
        delta = amount if direction == "down" else -amount
        await page.mouse.wheel(0, delta)
        await asyncio.sleep(0.3)
        log.info("Scrolled", direction=direction, amount=amount)

    async def wait(self, seconds: float = 2.0) -> None:
        """Wait for a specified duration (for dynamic content to load)."""
        await asyncio.sleep(seconds)
        log.info("Waited", seconds=seconds)

    async def wait_for_load(self, state: str = "networkidle", timeout: int = 30000) -> None:
        """Wait for page load state."""
        page = await self.ensure_page()
        await page.wait_for_load_state(state, timeout=timeout)
        log.info("Load state reached", state=state)

    async def get_title(self) -> str:
        """Return the page title."""
        if self._page is not None:
            return await self._page.title()
        return ""

    async def get_content(self) -> str:
        """Return the page's visible text content (via innerText on body)."""
        page = await self.ensure_page()
        try:
            return await page.inner_text("body", timeout=5000)
        except Exception:
            return ""

    # -- cookie management ----------------------------------------------------

    async def save_cookies(self) -> list[dict[str, Any]]:
        """Return all cookies from the current browser context."""
        if self._context is None:
            return []
        return await self._context.cookies()

    async def load_cookies(self, cookies: list[dict[str, Any]]) -> None:
        """Load cookies into the current browser context."""
        if self._context is not None and cookies:
            await self._context.add_cookies(cookies)
            log.info("Loaded cookies", count=len(cookies))

    # -- multi-app session management -----------------------------------------

    async def create_app_context(self, app_name: str) -> None:
        """Create a new isolated browser context for an app.

        Each app context has its own cookies, storage, and page — completely
        isolated from other app sessions.  If a context already exists for
        ``app_name``, this is a no-op.
        """
        if not self.is_alive:
            await self.start()

        if app_name in self._app_contexts:
            log.info("App context already exists", app_name=app_name)
            return

        context_kwargs: dict[str, Any] = {
            "viewport": {
                "width": self._config.viewport_width,
                "height": self._config.viewport_height,
            },
        }
        if self._config.user_agent:
            context_kwargs["user_agent"] = self._config.user_agent

        new_context = await self._browser.new_context(**context_kwargs)
        new_page = await new_context.new_page()

        self._app_contexts[app_name] = {
            "context": new_context,
            "page": new_page,
            "url": new_page.url,
        }

        log.info("Created app context", app_name=app_name)

    async def switch_app(self, app_name: str) -> dict[str, Any]:
        """Switch the active session to a different app's context.

        Saves the current context's state, then activates the target
        app's context.  Returns info about the activated context.
        """
        if app_name not in self._app_contexts:
            raise ValueError(
                f"No session for app '{app_name}'. "
                f"Available: {', '.join(self._app_contexts.keys()) or '(none)'}"
            )

        # Save current context state before switching
        if self._active_app != self._DEFAULT_APP and self._active_app in self._app_contexts:
            entry = self._app_contexts[self._active_app]
            entry["url"] = self._page.url if self._page else ""
        elif self._active_app == self._DEFAULT_APP:
            # Save the default context's page URL
            pass  # default context stays on self._context / self._page

        # Activate the target app's context
        target = self._app_contexts[app_name]
        self._context = target["context"]
        self._page = target["page"]
        self._active_app = app_name

        # Detach/re-attach network interceptor to new page
        try:
            await self._network.detach()
        except Exception:
            pass

        if self._config.network_capture_enabled:
            await self._network.attach(self._page)

        url = self._page.url if self._page else ""
        title = await self._page.title() if self._page else ""

        log.info("Switched app context", app_name=app_name, url=url)

        return {
            "app_name": app_name,
            "url": url,
            "title": title,
        }

    async def switch_to_default(self) -> None:
        """Switch back to the default (unnamed) browser context."""
        if self._active_app == self._DEFAULT_APP:
            return

        # Restore default context (from __init__ / start())
        # We need to re-find the original context and page
        # The original context is the one NOT in _app_contexts
        if hasattr(self, "_default_context") and self._default_context is not None:
            self._context = self._default_context
            self._page = self._default_page
        # else: the default context is already on self._context/_page
        # (it wasn't saved to _app_contexts)

        self._active_app = self._DEFAULT_APP
        log.info("Switched back to default context")

    def list_app_sessions(self) -> list[dict[str, str]]:
        """Return info about all active app sessions."""
        sessions: list[dict[str, str]] = []

        for app_name, entry in self._app_contexts.items():
            page = entry.get("page")
            url = page.url if page else entry.get("url", "")
            sessions.append({
                "app_name": app_name,
                "url": url,
                "active": "yes" if app_name == self._active_app else "no",
            })

        return sessions

    @property
    def active_app(self) -> str:
        """Return the name of the currently active app session."""
        return self._active_app

    # -- status ---------------------------------------------------------------

    def status_info(self) -> dict[str, Any]:
        """Return a status summary of the browser session."""
        return {
            "alive": self.is_alive,
            "url": self.current_url,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "headless": self._config.headless,
            "viewport": f"{self._config.viewport_width}x{self._config.viewport_height}",
            "active_app": self._active_app,
            "app_sessions": len(self._app_contexts),
        }
