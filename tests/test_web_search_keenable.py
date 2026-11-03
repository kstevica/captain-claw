"""Tests for the Keenable provider of the web_search tool."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from captain_claw.tools.web_search import WebSearchTool


def _response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _config(**fields):
    search = SimpleNamespace(provider="keenable", keenable_api_key="", max_results=5, timeout=20)
    for key, value in fields.items():
        setattr(search, key, value)
    return SimpleNamespace(tools=SimpleNamespace(web_search=search))


def _run(tool, cfg, **kwargs):
    with patch("captain_claw.tools.web_search.get_config", return_value=cfg):
        return asyncio.run(tool.execute(**kwargs))


def test_keyless_uses_public_endpoint(monkeypatch):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    tool = WebSearchTool()
    tool.client = MagicMock()
    tool.client.post = AsyncMock(
        return_value=_response(
            {
                "results": [
                    {"title": "K", "url": "https://k.example", "description": "desc", "snippet": "s"},
                    {"title": "No URL", "description": "dropped"},
                    {"title": "L", "url": "https://l.example", "snippet": "page text"},
                ]
            }
        )
    )

    result = _run(tool, _config(), query="captain claw", count=3)

    assert result.success
    args, kwargs = tool.client.post.call_args
    assert args[0] == "https://api.keenable.ai/v1/search/public"
    assert kwargs["json"] == {"query": "captain claw", "max_results": 3}
    assert kwargs["headers"]["X-Keenable-Title"] == "Captain Claw"
    assert "X-API-Key" not in kwargs["headers"]
    assert "[SEARCH ENGINE: Keenable]" in result.content
    assert "[RESULTS: 2]" in result.content
    assert "Snippet: desc" in result.content
    assert "Snippet: page text" in result.content
    assert "No URL" not in result.content


def test_key_uses_keyed_endpoint(monkeypatch):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    tool = WebSearchTool()
    tool.client = MagicMock()
    tool.client.post = AsyncMock(return_value=_response({"results": []}))

    result = _run(tool, _config(keenable_api_key="kn"), query="q")

    assert result.success
    args, kwargs = tool.client.post.call_args
    assert args[0] == "https://api.keenable.ai/v1/search"
    assert kwargs["headers"]["X-API-Key"] == "kn"
