import httpx
import pytest

from captain_claw.config import get_config, set_config
from captain_claw.tools.web_search import WebSearchTool


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)
        self.request = httpx.Request("GET", "https://api.search.brave.com/res/v1/web/search")

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                message=f"status={self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request, text=self.text),
            )


class _FakeClient:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.calls: list[dict] = []

    async def get(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self._response


@pytest.mark.asyncio
async def test_web_search_formats_brave_results_and_uses_config_defaults(monkeypatch):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.tools.web_search.api_key = "cfg-key"
    cfg.tools.web_search.max_results = 3
    cfg.tools.web_search.safesearch = "strict"
    set_config(cfg)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    try:
        tool = WebSearchTool()
        tool.client = _FakeClient(
            _FakeResponse(
                {
                    "web": {
                        "results": [
                            {
                                "title": "Zagreb travel guide",
                                "url": "https://example.com/zagreb",
                                "description": "Visit Zagreb old town and museums.",
                            },
                            {
                                "title": "Split city profile",
                                "url": "https://example.com/split",
                                "description": "Split overview and key landmarks.",
                            },
                        ]
                    }
                }
            )
        )

        result = await tool.execute(query="croatia top cities")

        assert result.success is True
        assert "[SEARCH ENGINE: Brave]" in result.content
        assert "[QUERY: croatia top cities]" in result.content
        assert "1. Zagreb travel guide" in result.content
        assert "URL: https://example.com/zagreb" in result.content
        assert "Snippet: Visit Zagreb old town and museums." in result.content

        call = tool.client.calls[0]
        assert call["headers"]["X-Subscription-Token"] == "cfg-key"
        assert call["params"]["count"] == 3
        assert call["params"]["safesearch"] == "strict"
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_web_search_uses_brave_api_key_from_env_when_config_missing(monkeypatch):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.tools.web_search.api_key = ""
    set_config(cfg)
    monkeypatch.setenv("BRAVE_API_KEY", "env-key")
    try:
        tool = WebSearchTool()
        tool.client = _FakeClient(_FakeResponse({"web": {"results": []}}))

        result = await tool.execute(query="zagreb", count=50, offset=2)

        assert result.success is True
        call = tool.client.calls[0]
        assert call["headers"]["X-Subscription-Token"] == "env-key"
        # Brave max is 20, ensure clamped.
        assert call["params"]["count"] == 20
        assert call["params"]["offset"] == 2
    finally:
        set_config(old_cfg)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)


@pytest.mark.asyncio
async def test_web_search_fails_when_api_key_missing(monkeypatch):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.tools.web_search.api_key = ""
    set_config(cfg)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    try:
        tool = WebSearchTool()

        result = await tool.execute(query="zagreb")

        assert result.success is False
        assert "Missing Brave API key" in (result.error or "")
    finally:
        set_config(old_cfg)
