"""Tests for the browser automation tool — Phases 1-6.

Mocks Playwright and vision so tests run without a real browser.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from captain_claw.tools.browser import BrowserTool
from captain_claw.tools.browser_session import BrowserSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_aria_snapshot() -> str:
    """Return a realistic aria_snapshot YAML-like string."""
    return (
        "- heading \"Dashboard\" [level=1]\n"
        "- navigation \"Main Menu\":\n"
        "  - link \"Home\"\n"
        "  - link \"Users\"\n"
        "  - link \"Reports\"\n"
        "- main:\n"
        "  - textbox \"Search users\"\n"
        "  - button \"Add User\"\n"
        "  - button \"Export\"\n"
        "  - table \"Users\":\n"
        "    - row:\n"
        "      - cell \"Alice\"\n"
        "      - cell \"admin\"\n"
    )


def _fake_login_snapshot() -> str:
    """Return a snapshot that looks like a login page."""
    return (
        "- heading \"Sign In\" [level=1]\n"
        "- textbox \"Email address\"\n"
        "- textbox \"Password\"\n"
        "- button \"Sign in\"\n"
        "- link \"Forgot password?\"\n"
    )


def _mock_page(aria_text: str = "") -> MagicMock:
    """Create a mock Playwright Page with aria_snapshot support."""
    page = AsyncMock()
    page.url = "https://example.com/dashboard"
    page.title = AsyncMock(return_value="Dashboard — Example App")

    # Mock locator chain: page.locator("body").aria_snapshot()
    body_locator = AsyncMock()
    body_locator.aria_snapshot = AsyncMock(return_value=aria_text or _fake_aria_snapshot())
    page.locator = MagicMock(return_value=body_locator)

    # Mock interactions
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.keyboard = AsyncMock()
    page.keyboard.press = AsyncMock()
    page.mouse = AsyncMock()
    page.mouse.wheel = AsyncMock()

    # get_by_role / get_by_text locators
    role_locator = AsyncMock()
    role_locator.click = AsyncMock()
    role_locator.fill = AsyncMock()
    role_locator.nth = MagicMock(return_value=role_locator)  # nth returns same mock
    page.get_by_role = MagicMock(return_value=role_locator)

    text_locator = AsyncMock()
    text_locator.click = AsyncMock()
    text_locator.nth = MagicMock(return_value=text_locator)
    page.get_by_text = MagicMock(return_value=text_locator)

    # Screenshot
    page.screenshot = AsyncMock(return_value=b"\xff\xd8\xff\xe0fake-jpeg-data")

    # Navigation
    response_mock = MagicMock()
    response_mock.status = 200
    response_mock.ok = True
    page.goto = AsyncMock(return_value=response_mock)

    # Load state
    page.wait_for_load_state = AsyncMock()
    page.inner_text = AsyncMock(return_value="Dashboard content here")

    return page


def _mock_session(page: MagicMock | None = None) -> BrowserSession:
    """Create a BrowserSession with mocked internals."""
    session = BrowserSession.__new__(BrowserSession)
    session._config = MagicMock()
    session._config.headless = True
    session._config.viewport_width = 1280
    session._config.viewport_height = 800
    session._config.default_wait_seconds = 0.0
    session._config.screenshot_jpeg_quality = 60
    session._config.network_max_captures = 100
    session._config.network_max_body_bytes = 10000
    session._config.network_filter_static = True
    session._config.network_capture_enabled = False
    session._config.network_auto_record = False
    session._config.cookie_persistence = True
    session._config.login_timeout_seconds = 30
    session._config.login_verify_wait_seconds = 0.01

    from captain_claw.tools.browser_network import NetworkInterceptor
    session._network = NetworkInterceptor()

    if page is None:
        page = _mock_page()

    session._playwright = MagicMock()
    session._browser = MagicMock()
    session._browser.is_connected = MagicMock(return_value=True)
    session._context = AsyncMock()
    session._context.cookies = AsyncMock(return_value=[{"name": "sid", "value": "abc123"}])
    session._context.add_cookies = AsyncMock()
    session._page = page
    session._started_at = 1000.0

    # Multi-app session state (Phase 6)
    session._app_contexts = {}
    session._active_app = "_default"

    return session


def _make_tool(session: BrowserSession | None = None) -> BrowserTool:
    """Create a BrowserTool with a pre-configured mock session."""
    tool = BrowserTool()
    if session is None:
        session = _mock_session()
    tool._session = session
    return tool


# ---------------------------------------------------------------------------
# Phase 1 tests: Basic actions
# ---------------------------------------------------------------------------

class TestBasicActions:
    """Test basic browser actions: open, navigate, screenshot, click, type, etc."""

    @pytest.mark.asyncio
    async def test_status_shows_info(self):
        tool = _make_tool()
        result = await tool.execute(action="status")
        assert result.success
        assert "Running: True" in result.content
        assert "URL:" in result.content

    @pytest.mark.asyncio
    async def test_navigate(self):
        tool = _make_tool()
        result = await tool.execute(action="navigate", url="https://example.com")
        assert result.success
        assert "Navigated to:" in result.content

    @pytest.mark.asyncio
    async def test_navigate_requires_url(self):
        tool = _make_tool()
        result = await tool.execute(action="navigate")
        assert not result.success
        assert "url" in result.error.lower()

    @pytest.mark.asyncio
    async def test_screenshot(self):
        tool = _make_tool()
        result = await tool.execute(action="screenshot")
        assert result.success
        assert "Path:" in result.content

    @pytest.mark.asyncio
    async def test_click_by_role(self):
        tool = _make_tool()
        result = await tool.execute(action="click", role="button", text="Submit")
        assert result.success
        assert "role='button'" in result.content

    @pytest.mark.asyncio
    async def test_click_by_text(self):
        tool = _make_tool()
        result = await tool.execute(action="click", text="Login")
        assert result.success
        assert "text: 'Login'" in result.content

    @pytest.mark.asyncio
    async def test_click_by_selector(self):
        tool = _make_tool()
        result = await tool.execute(action="click", selector="#btn")
        assert result.success
        assert "Clicked element: #btn" in result.content

    @pytest.mark.asyncio
    async def test_click_requires_target(self):
        tool = _make_tool()
        result = await tool.execute(action="click")
        assert not result.success
        assert "requires" in result.error.lower()

    @pytest.mark.asyncio
    async def test_type_by_selector(self):
        tool = _make_tool()
        result = await tool.execute(action="type", selector="input#email", text="user@test.com")
        assert result.success
        assert "Typed" in result.content

    @pytest.mark.asyncio
    async def test_type_requires_text(self):
        tool = _make_tool()
        result = await tool.execute(action="type", selector="input")
        assert not result.success
        assert "text" in result.error.lower()

    @pytest.mark.asyncio
    async def test_press_key(self):
        tool = _make_tool()
        result = await tool.execute(action="press_key", key="Enter")
        assert result.success
        assert "Pressed key: Enter" in result.content

    @pytest.mark.asyncio
    async def test_scroll(self):
        tool = _make_tool()
        result = await tool.execute(action="scroll", scroll_direction="down", scroll_amount=300)
        assert result.success
        assert "Scrolled down" in result.content

    @pytest.mark.asyncio
    async def test_wait(self):
        tool = _make_tool()
        result = await tool.execute(action="wait", wait_seconds=0.01)
        assert result.success
        assert "Waited" in result.content

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = _make_tool()
        result = await tool.execute(action="fly_to_moon")
        assert not result.success
        assert "Unknown" in result.error

    @pytest.mark.asyncio
    async def test_close(self):
        tool = _make_tool()
        session = tool._session
        session.close = AsyncMock()
        result = await tool.execute(action="close")
        assert result.success
        assert "Browser closed" in result.content


# ---------------------------------------------------------------------------
# Phase 3 tests: Page understanding
# ---------------------------------------------------------------------------

class TestPageUnderstanding:
    """Test observe, accessibility_tree, and find_element actions."""

    @pytest.mark.asyncio
    async def test_observe_returns_combined_output(self):
        tool = _make_tool()
        with patch(
            "captain_claw.tools.browser.BrowserVision.analyze_screenshot",
            new_callable=AsyncMock,
            return_value="This is a dashboard with user management.",
        ):
            result = await tool.execute(action="observe")

        assert result.success
        assert "URL:" in result.content
        assert "Visual Analysis" in result.content
        assert "Page Structure" in result.content
        assert "Interactive Elements" in result.content

    @pytest.mark.asyncio
    async def test_observe_with_goal(self):
        tool = _make_tool()
        with patch(
            "captain_claw.tools.browser.BrowserVision.analyze_screenshot",
            new_callable=AsyncMock,
            return_value="The search box is at the top.",
        ) as mock_vision:
            result = await tool.execute(action="observe", goal="find the search box")

        assert result.success
        # The goal should have been passed to the vision prompt (as keyword arg)
        call_kwargs = mock_vision.call_args
        prompt_used = call_kwargs.kwargs.get("prompt", "")
        if not prompt_used and len(call_kwargs.args) > 1:
            prompt_used = call_kwargs.args[1]
        assert "search box" in prompt_used.lower() or "goal" in prompt_used.lower()

    @pytest.mark.asyncio
    async def test_observe_without_vision_model(self):
        tool = _make_tool()
        with patch(
            "captain_claw.tools.browser.BrowserVision.analyze_screenshot",
            new_callable=AsyncMock,
            return_value="",
        ):
            result = await tool.execute(action="observe")

        assert result.success
        assert "no vision model configured" in result.content.lower()
        # Accessibility tree should still be present
        assert "Page Structure" in result.content

    @pytest.mark.asyncio
    async def test_accessibility_tree(self):
        tool = _make_tool()
        result = await tool.execute(action="accessibility_tree")
        assert result.success
        assert "Accessibility Tree:" in result.content
        assert "heading" in result.content.lower()

    @pytest.mark.asyncio
    async def test_find_element(self):
        tool = _make_tool()
        result = await tool.execute(action="find_element")
        assert result.success
        assert "interactive element" in result.content.lower()
        # Should find buttons and textbox from the mock snapshot
        assert "button" in result.content.lower()


# ---------------------------------------------------------------------------
# Phase 5 tests: act action and nth-match targeting
# ---------------------------------------------------------------------------

class TestActAction:
    """Test the goal-directed 'act' action."""

    @pytest.mark.asyncio
    async def test_act_requires_goal(self):
        tool = _make_tool()
        result = await tool.execute(action="act")
        assert not result.success
        assert "goal" in result.error.lower()

    @pytest.mark.asyncio
    async def test_act_returns_goal_directed_output(self):
        tool = _make_tool()
        with patch(
            "captain_claw.tools.browser.BrowserVision.analyze_screenshot",
            new_callable=AsyncMock,
            return_value="The Reports link is in the navigation menu on the left.",
        ):
            result = await tool.execute(action="act", goal="find the reports section")

        assert result.success
        assert "Goal: find the reports section" in result.content
        assert "URL:" in result.content
        assert "Analysis" in result.content
        assert "Available Actions" in result.content
        assert "Page Structure" in result.content

    @pytest.mark.asyncio
    async def test_act_generates_action_hints(self):
        tool = _make_tool()
        with patch(
            "captain_claw.tools.browser.BrowserVision.analyze_screenshot",
            new_callable=AsyncMock,
            return_value="Reports link visible in navigation.",
        ):
            result = await tool.execute(action="act", goal="click Reports")

        assert result.success
        # Should have suggested next steps section
        assert "Suggested Next Steps" in result.content
        # "Reports" is in the mock snapshot as a link, and "Reports" is in the goal
        assert "Reports" in result.content

    @pytest.mark.asyncio
    async def test_act_detects_login_page(self):
        """When on a login page, act should suggest using the login action."""
        page = _mock_page(_fake_login_snapshot())
        page.url = "https://example.com/login"
        page.title = AsyncMock(return_value="Sign In — Example App")
        session = _mock_session(page)
        tool = _make_tool(session)

        with patch(
            "captain_claw.tools.browser.BrowserVision.analyze_screenshot",
            new_callable=AsyncMock,
            return_value="This is a login page with email and password fields.",
        ):
            result = await tool.execute(action="act", goal="log into the dashboard")

        assert result.success
        assert "login" in result.content.lower()
        # Should suggest using credentials/login action
        assert "credentials" in result.content.lower() or "login" in result.content.lower()

    @pytest.mark.asyncio
    async def test_act_without_vision_still_works(self):
        """Act should still provide useful output even without a vision model."""
        tool = _make_tool()
        with patch(
            "captain_claw.tools.browser.BrowserVision.analyze_screenshot",
            new_callable=AsyncMock,
            return_value="",  # no vision model
        ):
            result = await tool.execute(action="act", goal="find users")

        assert result.success
        assert "Goal: find users" in result.content
        assert "Available Actions" in result.content
        assert "Page Structure" in result.content

    @pytest.mark.asyncio
    async def test_act_vision_prompt_is_goal_focused(self):
        """The vision prompt for act should be focused on actionable output."""
        tool = _make_tool()
        with patch(
            "captain_claw.tools.browser.BrowserVision.analyze_screenshot",
            new_callable=AsyncMock,
            return_value="analysis text",
        ) as mock_vision:
            await tool.execute(action="act", goal="export user data")

        # Check the vision prompt includes goal-directed instructions
        call_args = mock_vision.call_args
        prompt = call_args.kwargs.get("prompt", "")
        assert "export user data" in prompt
        assert "RECOMMENDED ACTION" in prompt


class TestNthMatch:
    """Test nth-match disambiguation for click and type actions."""

    @pytest.mark.asyncio
    async def test_click_with_nth_by_role(self):
        tool = _make_tool()
        result = await tool.execute(action="click", role="button", text="Submit", nth=1)
        assert result.success
        assert "nth=1" in result.content

        # Verify nth was called on the locator
        page = tool._session._page
        page.get_by_role.assert_called_with("button", name="Submit")
        role_locator = page.get_by_role.return_value
        role_locator.nth.assert_called_with(1)

    @pytest.mark.asyncio
    async def test_click_with_nth_by_text(self):
        tool = _make_tool()
        result = await tool.execute(action="click", text="Delete", nth=2)
        assert result.success
        assert "nth=2" in result.content

        page = tool._session._page
        page.get_by_text.assert_called_with("Delete", exact=False)
        text_locator = page.get_by_text.return_value
        text_locator.nth.assert_called_with(2)

    @pytest.mark.asyncio
    async def test_click_without_nth_does_not_call_nth(self):
        tool = _make_tool()
        result = await tool.execute(action="click", role="button", text="OK")
        assert result.success
        assert "nth" not in result.content

        page = tool._session._page
        role_locator = page.get_by_role.return_value
        role_locator.nth.assert_not_called()

    @pytest.mark.asyncio
    async def test_type_with_nth(self):
        tool = _make_tool()
        result = await tool.execute(
            action="type", role="textbox", selector="Name", text="Alice", nth=0,
        )
        assert result.success
        assert "nth=0" in result.content

    @pytest.mark.asyncio
    async def test_click_timeout_suggests_nth(self):
        """When click times out, error message should suggest nth."""
        page = _mock_page()
        role_locator = page.get_by_role.return_value
        role_locator.click = AsyncMock(side_effect=Exception("Timeout waiting for element"))
        session = _mock_session(page)
        tool = _make_tool(session)

        result = await tool.execute(action="click", role="button", text="Save")
        assert not result.success
        assert "nth" in result.error.lower()
        assert "observe" in result.error.lower()


class TestGenerateActionHints:
    """Test the _generate_action_hints static method."""

    def test_login_page_hint(self):
        hints = BrowserTool._generate_action_hints(
            url="https://app.com/login",
            title="Sign In",
            interactive=[],
            goal="access dashboard",
        )
        assert "login" in hints.lower()

    def test_matching_elements_hint(self):
        interactive = [
            {"role": "link", "name": "Reports", "selector": 'get_by_role("link", name="Reports")'},
            {"role": "button", "name": "Add User", "selector": 'get_by_role("button", name="Add User")'},
        ]
        hints = BrowserTool._generate_action_hints(
            url="https://app.com/dashboard",
            title="Dashboard",
            interactive=interactive,
            goal="view Reports",
        )
        assert "Reports" in hints
        assert "click" in hints.lower()

    def test_no_matching_elements_hint(self):
        interactive = [
            {"role": "button", "name": "Save", "selector": 'get_by_role("button", name="Save")'},
        ]
        hints = BrowserTool._generate_action_hints(
            url="https://app.com/page",
            title="Some Page",
            interactive=interactive,
            goal="find the billing section",
        )
        assert "scroll" in hints.lower() or "navigate" in hints.lower()

    def test_error_page_hint(self):
        hints = BrowserTool._generate_action_hints(
            url="https://app.com/missing",
            title="404 Not Found",
            interactive=[],
            goal="find users",
        )
        assert "error" in hints.lower()

    def test_empty_goal_words(self):
        """Should not crash with minimal goal."""
        hints = BrowserTool._generate_action_hints(
            url="https://app.com",
            title="App",
            interactive=[],
            goal="go",
        )
        # Should return something (possibly empty) without error
        assert isinstance(hints, str)


# ---------------------------------------------------------------------------
# Phase 4 tests: Credential and login actions (basic coverage)
# ---------------------------------------------------------------------------

class TestCredentialActions:
    """Test credential management and login actions."""

    @pytest.mark.asyncio
    async def test_credentials_store_requires_fields(self):
        tool = _make_tool()
        result = await tool.execute(action="credentials_store")
        assert not result.success
        assert "app_name" in result.error.lower()

    @pytest.mark.asyncio
    async def test_credentials_store_requires_all_fields(self):
        tool = _make_tool()
        result = await tool.execute(
            action="credentials_store", app_name="test", url="https://test.com",
        )
        assert not result.success
        assert "username" in result.error.lower()

    @pytest.mark.asyncio
    async def test_login_requires_app_name(self):
        tool = _make_tool()
        result = await tool.execute(action="login")
        assert not result.success
        assert "app_name" in result.error.lower()

    @pytest.mark.asyncio
    async def test_login_no_credentials(self):
        tool = _make_tool()
        with patch.object(
            tool, "_get_credential_store",
        ) as mock_store_getter:
            mock_store = AsyncMock()
            mock_store.get_credential = AsyncMock(return_value=None)
            mock_store_getter.return_value = mock_store

            result = await tool.execute(action="login", app_name="unknown_app")
        assert not result.success
        assert "No credentials" in result.error

    @pytest.mark.asyncio
    async def test_credentials_delete_requires_app_name(self):
        tool = _make_tool()
        result = await tool.execute(action="credentials_delete")
        assert not result.success
        assert "app_name" in result.error.lower()


# ---------------------------------------------------------------------------
# Integration: tool schema validation
# ---------------------------------------------------------------------------

class TestToolSchema:
    """Test that the tool schema is well-formed."""

    def test_all_actions_in_enum(self):
        tool = BrowserTool()
        actions = tool.parameters["properties"]["action"]["enum"]
        assert "act" in actions
        assert "observe" in actions
        assert "login" in actions
        assert "network_capture" in actions
        assert len(actions) == 27  # 23 from Phase 5 + 4 new (api_replay, api_test, switch_app, list_sessions)

    def test_nth_parameter_exists(self):
        tool = BrowserTool()
        props = tool.parameters["properties"]
        assert "nth" in props
        assert props["nth"]["type"] == "integer"

    def test_goal_parameter_mentions_act(self):
        tool = BrowserTool()
        goal_desc = tool.parameters["properties"]["goal"]["description"]
        assert "act" in goal_desc.lower()

    def test_required_is_only_action(self):
        tool = BrowserTool()
        assert tool.parameters["required"] == ["action"]

    def test_tool_name(self):
        tool = BrowserTool()
        assert tool.name == "browser"
        assert tool.timeout_seconds == 60.0

    def test_phase6_actions_in_enum(self):
        tool = BrowserTool()
        actions = tool.parameters["properties"]["action"]["enum"]
        assert "api_replay" in actions
        assert "api_test" in actions
        assert "switch_app" in actions
        assert "list_sessions" in actions

    def test_phase6_parameters_exist(self):
        tool = BrowserTool()
        props = tool.parameters["properties"]
        assert "api_id" in props
        assert "endpoint" in props
        assert "method" in props
        assert "query_params" in props
        assert "body_json" in props

    def test_method_enum(self):
        tool = BrowserTool()
        method_prop = tool.parameters["properties"]["method"]
        assert method_prop["type"] == "string"
        assert set(method_prop["enum"]) == {"GET", "POST", "PUT", "DELETE", "PATCH"}


# ---------------------------------------------------------------------------
# Phase 6 tests: API Replay
# ---------------------------------------------------------------------------

class TestApiReplay:
    """Test api_replay and api_test actions."""

    @pytest.mark.asyncio
    async def test_api_replay_requires_api_id(self):
        tool = _make_tool()
        result = await tool.execute(action="api_replay")
        assert not result.success
        assert "api_id" in result.error.lower()

    @pytest.mark.asyncio
    async def test_api_replay_requires_endpoint(self):
        tool = _make_tool()
        result = await tool.execute(action="api_replay", api_id="test123")
        assert not result.success
        assert "endpoint" in result.error.lower()

    @pytest.mark.asyncio
    async def test_api_replay_api_not_found(self):
        tool = _make_tool()
        with patch(
            "captain_claw.session.get_session_manager",
        ) as mock_sm_getter:
            mock_sm = AsyncMock()
            mock_sm.select_api = AsyncMock(return_value=None)
            mock_sm_getter.return_value = mock_sm

            result = await tool.execute(
                action="api_replay", api_id="nonexistent", endpoint="/api/test",
            )
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_api_replay_success(self):
        """Test successful API replay with mocked SessionManager and httpx."""
        tool = _make_tool()

        # Create a mock ApiEntry
        mock_api = MagicMock()
        mock_api.id = "abc123"
        mock_api.name = "Test API"
        mock_api.base_url = "https://api.example.com"
        mock_api.endpoints = json.dumps([
            {"method": "GET", "path": "/api/users", "description": "List users"},
        ])
        mock_api.auth_type = "bearer"
        mock_api.credentials = "test-token-123"
        mock_api.use_count = 5

        # Mock ApiReplayResult
        from captain_claw.tools.browser_api_replay import ApiReplayResult
        mock_result = ApiReplayResult(
            success=True,
            status_code=200,
            url="https://api.example.com/api/users",
            method="GET",
            response_body='{"users": []}',
            response_headers={"content-type": "application/json"},
            elapsed_ms=42.0,
        )

        with patch(
            "captain_claw.session.get_session_manager",
        ) as mock_sm_getter, patch(
            "captain_claw.tools.browser.ApiReplayEngine.replay",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            mock_sm = AsyncMock()
            mock_sm.select_api = AsyncMock(return_value=mock_api)
            mock_sm.increment_api_usage = AsyncMock()
            mock_sm_getter.return_value = mock_sm

            result = await tool.execute(
                action="api_replay", api_id="#1", endpoint="/api/users", method="GET",
            )

        assert result.success
        assert "200" in result.content
        assert "Test API" in result.content

    @pytest.mark.asyncio
    async def test_api_replay_invalid_query_params(self):
        tool = _make_tool()

        mock_api = MagicMock()
        mock_api.id = "abc123"
        mock_api.name = "Test API"
        mock_api.base_url = "https://api.example.com"
        mock_api.endpoints = None
        mock_api.auth_type = None
        mock_api.credentials = None

        with patch(
            "captain_claw.session.get_session_manager",
        ) as mock_sm_getter:
            mock_sm = AsyncMock()
            mock_sm.select_api = AsyncMock(return_value=mock_api)
            mock_sm_getter.return_value = mock_sm

            result = await tool.execute(
                action="api_replay", api_id="#1", endpoint="/api/test",
                query_params="not valid json{",
            )
        assert not result.success
        assert "json" in result.error.lower()

    @pytest.mark.asyncio
    async def test_api_replay_invalid_body_json(self):
        tool = _make_tool()

        mock_api = MagicMock()
        mock_api.id = "abc123"
        mock_api.name = "Test API"
        mock_api.base_url = "https://api.example.com"
        mock_api.endpoints = None
        mock_api.auth_type = None
        mock_api.credentials = None

        with patch(
            "captain_claw.session.get_session_manager",
        ) as mock_sm_getter:
            mock_sm = AsyncMock()
            mock_sm.select_api = AsyncMock(return_value=mock_api)
            mock_sm_getter.return_value = mock_sm

            result = await tool.execute(
                action="api_replay", api_id="#1", endpoint="/api/test",
                body_json="broken{json",
            )
        assert not result.success
        assert "json" in result.error.lower()

    @pytest.mark.asyncio
    async def test_api_replay_auth_failure_hint(self):
        """When API returns 401, should suggest re-login."""
        tool = _make_tool()

        mock_api = MagicMock()
        mock_api.id = "abc123"
        mock_api.name = "Test API"
        mock_api.base_url = "https://api.example.com"
        mock_api.endpoints = None
        mock_api.auth_type = "bearer"
        mock_api.credentials = "expired-token"
        mock_api.use_count = 3

        from captain_claw.tools.browser_api_replay import ApiReplayResult
        mock_result = ApiReplayResult(
            success=False,
            status_code=401,
            url="https://api.example.com/api/test",
            method="GET",
            response_body="Unauthorized",
            elapsed_ms=15.0,
            error="HTTP 401",
        )

        with patch(
            "captain_claw.session.get_session_manager",
        ) as mock_sm_getter, patch(
            "captain_claw.tools.browser.ApiReplayEngine.replay",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            mock_sm = AsyncMock()
            mock_sm.select_api = AsyncMock(return_value=mock_api)
            mock_sm_getter.return_value = mock_sm

            result = await tool.execute(
                action="api_replay", api_id="#1", endpoint="/api/test",
            )

        assert not result.success
        assert "expired" in result.content.lower() or "login" in result.content.lower()


class TestApiTest:
    """Test the api_test action."""

    @pytest.mark.asyncio
    async def test_api_test_requires_api_id(self):
        tool = _make_tool()
        result = await tool.execute(action="api_test")
        assert not result.success
        assert "api_id" in result.error.lower()

    @pytest.mark.asyncio
    async def test_api_test_api_not_found(self):
        tool = _make_tool()
        with patch(
            "captain_claw.session.get_session_manager",
        ) as mock_sm_getter:
            mock_sm = AsyncMock()
            mock_sm.select_api = AsyncMock(return_value=None)
            mock_sm_getter.return_value = mock_sm

            result = await tool.execute(action="api_test", api_id="missing")
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_api_test_info_only(self):
        """api_test without endpoint should show API info only."""
        tool = _make_tool()

        mock_api = MagicMock()
        mock_api.id = "abc123"
        mock_api.name = "Jira API"
        mock_api.base_url = "https://jira.example.com"
        mock_api.endpoints = json.dumps([
            {"method": "GET", "path": "/rest/api/2/search", "description": "Search issues"},
            {"method": "POST", "path": "/rest/api/2/issue", "description": "Create issue"},
        ])
        mock_api.auth_type = "bearer"
        mock_api.credentials = "tok123"
        mock_api.use_count = 10

        with patch(
            "captain_claw.session.get_session_manager",
        ) as mock_sm_getter:
            mock_sm = AsyncMock()
            mock_sm.select_api = AsyncMock(return_value=mock_api)
            mock_sm_getter.return_value = mock_sm

            result = await tool.execute(action="api_test", api_id="#1")

        assert result.success
        assert "Jira API" in result.content
        assert "/rest/api/2/search" in result.content
        assert "No endpoint specified" in result.content

    @pytest.mark.asyncio
    async def test_api_test_with_endpoint(self):
        """api_test with endpoint should do a test call."""
        tool = _make_tool()

        mock_api = MagicMock()
        mock_api.id = "abc123"
        mock_api.name = "Test API"
        mock_api.base_url = "https://api.example.com"
        mock_api.endpoints = json.dumps([
            {"method": "GET", "path": "/api/status"},
        ])
        mock_api.auth_type = None
        mock_api.credentials = None
        mock_api.use_count = 0

        from captain_claw.tools.browser_api_replay import ApiReplayResult
        mock_result = ApiReplayResult(
            success=True,
            status_code=200,
            url="https://api.example.com/api/status",
            method="GET",
            response_body='{"status": "ok"}',
            response_headers={"content-type": "application/json"},
            elapsed_ms=20.0,
        )

        with patch(
            "captain_claw.session.get_session_manager",
        ) as mock_sm_getter, patch(
            "captain_claw.tools.browser.ApiReplayEngine.replay",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            mock_sm = AsyncMock()
            mock_sm.select_api = AsyncMock(return_value=mock_api)
            mock_sm_getter.return_value = mock_sm

            result = await tool.execute(
                action="api_test", api_id="#1", endpoint="/api/status",
            )

        assert result.success
        assert "Test call" in result.content
        assert "200" in result.content


# ---------------------------------------------------------------------------
# Phase 6 tests: Multi-App Sessions
# ---------------------------------------------------------------------------

class TestMultiAppSessions:
    """Test switch_app and list_sessions actions."""

    @pytest.mark.asyncio
    async def test_switch_app_requires_app_name(self):
        tool = _make_tool()
        result = await tool.execute(action="switch_app")
        assert not result.success
        assert "app_name" in result.error.lower()

    @pytest.mark.asyncio
    async def test_switch_app_creates_and_switches(self):
        tool = _make_tool()
        session = tool._session

        # Mock the multi-context methods
        session.create_app_context = AsyncMock()
        session.switch_app = AsyncMock(return_value={
            "app_name": "jira",
            "url": "about:blank",
            "title": "",
        })
        session.list_app_sessions = MagicMock(return_value=[
            {"app_name": "jira", "url": "about:blank", "active": "yes"},
        ])

        result = await tool.execute(action="switch_app", app_name="jira")
        assert result.success
        assert "jira" in result.content
        session.create_app_context.assert_called_once_with("jira")
        session.switch_app.assert_called_once_with("jira")

    @pytest.mark.asyncio
    async def test_switch_app_existing_context(self):
        """If the context already exists, don't recreate it."""
        tool = _make_tool()
        session = tool._session

        # Simulate existing context
        session._app_contexts = {"confluence": {"context": MagicMock(), "page": MagicMock(), "url": "https://wiki.example.com"}}
        session.create_app_context = AsyncMock()
        session.switch_app = AsyncMock(return_value={
            "app_name": "confluence",
            "url": "https://wiki.example.com",
            "title": "Wiki",
        })
        session.list_app_sessions = MagicMock(return_value=[
            {"app_name": "confluence", "url": "https://wiki.example.com", "active": "yes"},
        ])

        result = await tool.execute(action="switch_app", app_name="confluence")
        assert result.success
        # Should NOT create a new context since it already exists
        session.create_app_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_switch_app_shows_sessions(self):
        """switch_app result should list all active sessions."""
        tool = _make_tool()
        session = tool._session

        session.create_app_context = AsyncMock()
        session.switch_app = AsyncMock(return_value={
            "app_name": "github",
            "url": "about:blank",
            "title": "",
        })
        session.list_app_sessions = MagicMock(return_value=[
            {"app_name": "jira", "url": "https://jira.co", "active": "no"},
            {"app_name": "github", "url": "about:blank", "active": "yes"},
        ])

        result = await tool.execute(action="switch_app", app_name="github")
        assert result.success
        assert "jira" in result.content
        assert "github*" in result.content  # asterisk marks active

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self):
        tool = _make_tool()
        session = tool._session
        session.list_app_sessions = MagicMock(return_value=[])

        result = await tool.execute(action="list_sessions")
        assert result.success
        assert "No multi-app sessions" in result.content

    @pytest.mark.asyncio
    async def test_list_sessions_with_apps(self):
        tool = _make_tool()
        session = tool._session
        session.list_app_sessions = MagicMock(return_value=[
            {"app_name": "jira", "url": "https://jira.company.com/board", "active": "yes"},
            {"app_name": "confluence", "url": "https://wiki.company.com", "active": "no"},
        ])

        result = await tool.execute(action="list_sessions")
        assert result.success
        assert "2" in result.content
        assert "jira" in result.content
        assert "confluence" in result.content
        assert "active" in result.content.lower()


# ---------------------------------------------------------------------------
# Phase 6 tests: ApiReplayEngine unit tests
# ---------------------------------------------------------------------------

class TestApiReplayEngine:
    """Unit tests for the ApiReplayEngine."""

    def test_resolve_auth_headers_bearer(self):
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        headers = ApiReplayEngine.resolve_auth_headers("bearer", "mytoken123")
        assert headers == {"Authorization": "Bearer mytoken123"}

    def test_resolve_auth_headers_bearer_strips_prefix(self):
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        headers = ApiReplayEngine.resolve_auth_headers("bearer", "Bearer already-prefixed")
        assert headers == {"Authorization": "Bearer already-prefixed"}

    def test_resolve_auth_headers_api_key(self):
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        headers = ApiReplayEngine.resolve_auth_headers("api_key", "key-abc-123")
        assert headers == {"X-API-Key": "key-abc-123"}

    def test_resolve_auth_headers_basic(self):
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        headers = ApiReplayEngine.resolve_auth_headers("basic", "dXNlcjpwYXNz")
        assert headers == {"Authorization": "Basic dXNlcjpwYXNz"}

    def test_resolve_auth_headers_custom(self):
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        headers = ApiReplayEngine.resolve_auth_headers("custom", "X-Custom-Auth: secret-value")
        assert headers == {"X-Custom-Auth": "secret-value"}

    def test_resolve_auth_headers_none(self):
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        headers = ApiReplayEngine.resolve_auth_headers("none", None)
        assert headers == {}

        headers2 = ApiReplayEngine.resolve_auth_headers(None, None)
        assert headers2 == {}

    def test_find_endpoint_in_api_exact_match(self):
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        endpoints = json.dumps([
            {"method": "GET", "path": "/api/users"},
            {"method": "POST", "path": "/api/users"},
            {"method": "GET", "path": "/api/issues"},
        ])
        result = ApiReplayEngine.find_endpoint_in_api(endpoints, "/api/users", "GET")
        assert result is not None
        assert result["method"] == "GET"
        assert result["path"] == "/api/users"

    def test_find_endpoint_in_api_parameterized(self):
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        endpoints = json.dumps([
            {"method": "GET", "path": "/api/users/{id}"},
        ])
        result = ApiReplayEngine.find_endpoint_in_api(endpoints, "/api/users/42", "GET")
        assert result is not None
        assert result["path"] == "/api/users/{id}"

    def test_find_endpoint_in_api_no_match(self):
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        endpoints = json.dumps([
            {"method": "GET", "path": "/api/users"},
        ])
        result = ApiReplayEngine.find_endpoint_in_api(endpoints, "/api/projects", "GET")
        assert result is None

    def test_find_endpoint_in_api_invalid_json(self):
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        result = ApiReplayEngine.find_endpoint_in_api("not json{", "/api/test")
        assert result is None

    def test_find_endpoint_in_api_none(self):
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        result = ApiReplayEngine.find_endpoint_in_api(None, "/api/test")
        assert result is None

    def test_format_endpoints_list(self):
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        endpoints = json.dumps([
            {"method": "GET", "path": "/api/users", "description": "List users"},
            {"method": "POST", "path": "/api/users", "description": "Create user"},
        ])
        formatted = ApiReplayEngine.format_endpoints_list(endpoints)
        assert "GET" in formatted
        assert "POST" in formatted
        assert "/api/users" in formatted
        assert "List users" in formatted

    def test_format_endpoints_list_empty(self):
        from captain_claw.tools.browser_api_replay import ApiReplayEngine

        assert "no endpoints" in ApiReplayEngine.format_endpoints_list(None).lower()
        assert "no endpoints" in ApiReplayEngine.format_endpoints_list("[]").lower()

    def test_api_replay_result_summary_success(self):
        from captain_claw.tools.browser_api_replay import ApiReplayResult

        result = ApiReplayResult(
            success=True,
            status_code=200,
            url="https://api.example.com/test",
            method="GET",
            response_body='{"data": "ok"}',
            response_headers={"content-type": "application/json"},
            elapsed_ms=50.0,
        )
        summary = result.to_summary()
        assert "200" in summary
        assert "GET" in summary
        assert "api.example.com" in summary
        assert "50ms" in summary

    def test_api_replay_result_summary_failure(self):
        from captain_claw.tools.browser_api_replay import ApiReplayResult

        result = ApiReplayResult(
            success=False,
            error="Connection timed out",
        )
        summary = result.to_summary()
        assert "FAILED" in summary
        assert "timed out" in summary


# ---------------------------------------------------------------------------
# Phase 6 tests: BrowserSession multi-context
# ---------------------------------------------------------------------------

class TestBrowserSessionMultiContext:
    """Test multi-app context management in BrowserSession."""

    def test_initial_state(self):
        session = _mock_session()
        # Need to set multi-context attributes
        session._app_contexts = {}
        session._active_app = "_default"

        assert session.active_app == "_default"
        assert session.list_app_sessions() == []

    @pytest.mark.asyncio
    async def test_create_app_context(self):
        session = _mock_session()
        session._app_contexts = {}
        session._active_app = "_default"

        # Mock browser.new_context
        new_context_mock = AsyncMock()
        new_page_mock = AsyncMock()
        new_page_mock.url = "about:blank"
        new_context_mock.new_page = AsyncMock(return_value=new_page_mock)
        session._browser.new_context = AsyncMock(return_value=new_context_mock)

        await session.create_app_context("jira")
        assert "jira" in session._app_contexts
        assert session._app_contexts["jira"]["page"] == new_page_mock

    @pytest.mark.asyncio
    async def test_create_app_context_idempotent(self):
        session = _mock_session()
        session._app_contexts = {}
        session._active_app = "_default"

        new_context_mock = AsyncMock()
        new_page_mock = AsyncMock()
        new_page_mock.url = "about:blank"
        new_context_mock.new_page = AsyncMock(return_value=new_page_mock)
        session._browser.new_context = AsyncMock(return_value=new_context_mock)

        await session.create_app_context("jira")
        await session.create_app_context("jira")  # second call should be no-op
        # Should only create once
        assert session._browser.new_context.call_count == 1

    @pytest.mark.asyncio
    async def test_switch_app(self):
        session = _mock_session()
        session._app_contexts = {}
        session._active_app = "_default"

        # Set up a mock app context
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_page.url = "https://jira.example.com"
        mock_page.title = AsyncMock(return_value="Jira Board")
        session._app_contexts["jira"] = {
            "context": mock_context,
            "page": mock_page,
            "url": "https://jira.example.com",
        }

        info = await session.switch_app("jira")
        assert info["app_name"] == "jira"
        assert session.active_app == "jira"
        assert session._page == mock_page

    @pytest.mark.asyncio
    async def test_switch_app_unknown_raises(self):
        session = _mock_session()
        session._app_contexts = {}
        session._active_app = "_default"

        with pytest.raises(ValueError, match="No session for app"):
            await session.switch_app("unknown")

    def test_list_app_sessions(self):
        session = _mock_session()
        session._active_app = "jira"

        mock_page_1 = MagicMock()
        mock_page_1.url = "https://jira.example.com"
        mock_page_2 = MagicMock()
        mock_page_2.url = "https://wiki.example.com"

        session._app_contexts = {
            "jira": {"context": MagicMock(), "page": mock_page_1, "url": "https://jira.example.com"},
            "confluence": {"context": MagicMock(), "page": mock_page_2, "url": "https://wiki.example.com"},
        }

        sessions = session.list_app_sessions()
        assert len(sessions) == 2
        jira_session = next(s for s in sessions if s["app_name"] == "jira")
        assert jira_session["active"] == "yes"
        conf_session = next(s for s in sessions if s["app_name"] == "confluence")
        assert conf_session["active"] == "no"

    def test_status_info_includes_multi_app(self):
        session = _mock_session()
        session._app_contexts = {"jira": {}, "confluence": {}}
        session._active_app = "jira"

        info = session.status_info()
        assert info["active_app"] == "jira"
        assert info["app_sessions"] == 2
