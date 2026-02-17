import pytest

from captain_claw.config import get_config, set_config
from captain_claw.tools.web_fetch import WebFetchTool


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
async def test_web_fetch_html_mode_returns_raw_html():
    html = "<html><body><h1>Hello</h1></body></html>"
    tool = WebFetchTool()
    tool.client = _FakeClient(_FakeResponse(html))

    result = await tool.execute(url="https://example.com", extract_mode="html")

    assert result.success is True
    assert "[Mode: html]" in result.content
    assert "<html><body><h1>Hello</h1></body></html>" in result.content


@pytest.mark.asyncio
async def test_web_fetch_rejects_unsupported_extract_mode():
    html = "<html><body>hello</body></html>"
    tool = WebFetchTool()
    tool.client = _FakeClient(_FakeResponse(html))

    result = await tool.execute(url="https://example.com", extract_mode="xml")

    assert result.success is False
    assert "Unsupported extract_mode" in (result.error or "")


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
