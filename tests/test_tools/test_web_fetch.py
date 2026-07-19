from unittest.mock import AsyncMock, patch

import pytest

from captain_claw.config import get_config, set_config
from captain_claw.tools.web_fetch import WebFetchTool, WebGetTool


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, response: _FakeResponse):
        self._response = response

    async def get(self, url: str) -> _FakeResponse:
        return self._response


@pytest.mark.asyncio
async def test_web_fetch_defaults_to_readable_text_extraction():
    html = """
    <html>
      <head>
        <title>Example Page</title>
        <script>var should_not_show = true;</script>
      </head>
      <body>
        <h1>Hello World</h1>
        <p>This is readable text.</p>
      </body>
    </html>
    """
    tool = WebFetchTool()
    tool.client = _FakeClient(_FakeResponse(html))

    result = await tool.execute(url="https://example.com")

    assert result.success is True
    assert "[Mode: text]" in result.content
    assert "Example Page" in result.content
    assert "Hello World" in result.content
    assert "This is readable text." in result.content
    assert "should_not_show" not in result.content
    assert "<html>" not in result.content


@pytest.mark.asyncio
async def test_web_fetch_text_mode_preserves_links():
    html = """
    <html>
      <body>
        <p>Read <a href="/guide">the guide</a> for details.</p>
        <p>External <a href="https://docs.example.org/ref">reference</a>.</p>
      </body>
    </html>
    """
    tool = WebFetchTool()
    tool.client = _FakeClient(_FakeResponse(html))

    result = await tool.execute(url="https://example.com/start")

    assert result.success is True
    assert "the guide (https://example.com/guide)" in result.content
    assert "reference (https://docs.example.org/ref)" in result.content


@pytest.mark.asyncio
async def test_web_fetch_ignores_legacy_extract_mode():
    """extract_mode is silently stripped — web_fetch always returns text."""
    html = "<html><body><h1>Hello</h1></body></html>"
    tool = WebFetchTool()
    tool.client = _FakeClient(_FakeResponse(html))

    result = await tool.execute(url="https://example.com", extract_mode="html")

    assert result.success is True
    assert "[Mode: text]" in result.content
    assert "Hello" in result.content
    # Should NOT contain raw HTML — extract_mode is ignored.
    assert "<html>" not in result.content


@pytest.mark.asyncio
async def test_web_fetch_uses_configurable_default_max_chars():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.tools.web_fetch.max_chars = 20
    set_config(cfg)
    try:
        html = "<html><body><p>abcdefghijklmnopqrstuvwxyz</p></body></html>"
        tool = WebFetchTool()
        tool.client = _FakeClient(_FakeResponse(html))

        result = await tool.execute(url="https://example.com")

        assert result.success is True
        assert "... [truncated]" in result.content
        assert "abcdefghijklmnopqrst" in result.content
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_web_fetch_deep_fetch_calls_playwright():
    """deep_fetch=True should delegate to _deep_fetch and return deep mode."""
    final_html = "<html><body><h1>Full Content</h1><p>Loaded via JS.</p></body></html>"

    with patch("captain_claw.tools.web_fetch._deep_fetch", new_callable=AsyncMock) as mock_df:
        mock_df.return_value = final_html
        tool = WebFetchTool()

        result = await tool.execute(url="https://example.com/lazy", deep_fetch=True)

    assert result.success is True
    assert "[Mode: deep]" in result.content
    assert "Full Content" in result.content
    assert "Loaded via JS." in result.content
    assert "<html>" not in result.content  # Still extracts text, not raw HTML.
    mock_df.assert_awaited_once_with("https://example.com/lazy")


# ── web_get: text on the first call per URL, raw HTML on the repeat ──

_GET_HTML = """
<html>
  <head><title>Shop</title><script>var hidden = 1;</script></head>
  <body><div class="price">$42</div></body>
</html>
"""


@pytest.mark.asyncio
async def test_web_get_first_call_returns_stripped_text():
    tool = WebGetTool()
    tool.client = _FakeClient(_FakeResponse(_GET_HTML))

    result = await tool.execute(url="https://example.com/shop")

    assert result.success is True
    assert "[Mode: text (first web_get on this URL)]" in result.content
    assert "Shop" in result.content
    assert "$42" in result.content
    assert "<div" not in result.content
    assert "hidden" not in result.content
    assert "call web_get on this same URL once more" in result.content


@pytest.mark.asyncio
async def test_web_get_second_call_on_same_url_returns_raw_html():
    tool = WebGetTool()
    tool.client = _FakeClient(_FakeResponse(_GET_HTML))

    await tool.execute(url="https://example.com/shop")
    result = await tool.execute(url="https://example.com/shop")

    assert result.success is True
    assert "[Mode: html]" in result.content
    assert '<div class="price">$42</div>' in result.content
    assert "raw HTML withheld" not in result.content


@pytest.mark.asyncio
async def test_web_get_tracks_urls_independently():
    """A second call on a DIFFERENT URL is still that URL's first look."""
    tool = WebGetTool()
    tool.client = _FakeClient(_FakeResponse(_GET_HTML))

    await tool.execute(url="https://example.com/shop")
    result = await tool.execute(url="https://example.com/other")

    assert "[Mode: text (first web_get on this URL)]" in result.content
    assert "<div" not in result.content


@pytest.mark.asyncio
async def test_web_get_failed_fetch_does_not_burn_the_text_pass():
    """A URL that errored was never seen — the next call still gets text."""
    class _FailingOnceClient:
        def __init__(self, response):
            self._response = response
            self.calls = 0

        async def get(self, url: str):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return self._response

    tool = WebGetTool()
    tool.client = _FailingOnceClient(_FakeResponse(_GET_HTML))

    failed = await tool.execute(url="https://example.com/shop")
    assert failed.success is False

    result = await tool.execute(url="https://example.com/shop")
    assert result.success is True
    assert "[Mode: text (first web_get on this URL)]" in result.content
