"""Web fetch tools for retrieving web page content."""

import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import httpx

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


def _make_http_client() -> httpx.AsyncClient:
    """Create a shared-style HTTP client."""
    return httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={
            "User-Agent": "Captain Claw/0.1.0 (Web Fetch Tool)",
        },
    )


def _extract_readable_text(html: str, base_url: str | None = None) -> str:
    """Extract human-readable text from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "template", "svg", "canvas"]):
        tag.decompose()

    # Preserve links in readable text so later turns can reference sources.
    for anchor in soup.find_all("a"):
        href = (anchor.get("href") or "").strip()
        label = anchor.get_text(" ", strip=True)
        if not href:
            continue
        absolute = urljoin(base_url, href) if base_url else href
        if label:
            anchor.replace_with(f"{label} ({absolute})")
        else:
            anchor.replace_with(absolute)

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    raw_text = soup.get_text(separator="\n")
    lines: list[str] = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)

    text = "\n".join(lines)
    if title and not text.startswith(title):
        return f"{title}\n\n{text}" if text else title
    return text


class WebFetchTool(Tool):
    """Fetch web page content as clean readable text (ALWAYS text mode)."""

    name = "web_fetch"
    description = (
        "Fetch a URL and return clean readable text (no HTML). "
        "This is the default tool for reading web pages. Always extracts text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch",
            },
            "max_chars": {
                "type": "number",
                "description": "Maximum characters to return (default from config, typically 100000)",
            },
        },
        "required": ["url"],
    }

    def __init__(self):
        self.client = _make_http_client()

    async def execute(
        self,
        url: str,
        max_chars: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Fetch a web page and extract readable text via BeautifulSoup.

        Args:
            url: URL to fetch
            max_chars: Max characters to return

        Returns:
            ToolResult with extracted readable text
        """
        # Hard guard: strip any extract_mode — web_fetch ALWAYS returns text.
        kwargs.pop("extract_mode", None)

        try:
            log.info("Fetching URL (text mode)", url=url)
            cfg = get_config()
            configured_max = int(getattr(cfg.tools.web_fetch, "max_chars", 100000))
            effective_max_chars = configured_max if max_chars is None else int(max_chars)
            effective_max_chars = max(1, effective_max_chars)

            response = await self.client.get(url)
            response.raise_for_status()

            content = _extract_readable_text(response.text, base_url=url)

            if len(content) > effective_max_chars:
                content = content[:effective_max_chars] + "\n... [truncated]"

            output = f"[URL: {url}]\n"
            output += f"[Status: {response.status_code}]\n"
            output += f"[Mode: text]\n"
            output += f"[Size: {len(response.text)} chars]\n\n"
            output += content

            return ToolResult(
                success=True,
                content=output,
            )

        except httpx.HTTPError as e:
            log.error("HTTP fetch failed", url=url, error=str(e))
            return ToolResult(
                success=False,
                error=f"HTTP error: {e}",
            )
        except Exception as e:
            log.error("Fetch failed", url=url, error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class WebGetTool(Tool):
    """Fetch raw HTML from a URL (for scraping, parsing, DOM inspection)."""

    name = "web_get"
    description = (
        "Fetch a URL and return the raw HTML source. "
        "Use ONLY when you need the actual HTML markup (scraping, DOM analysis, CSS selectors). "
        "For normal page reading, use web_fetch instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch",
            },
            "max_chars": {
                "type": "number",
                "description": "Maximum characters of raw HTML to return (default from config, typically 100000)",
            },
        },
        "required": ["url"],
    }

    def __init__(self):
        self.client = _make_http_client()

    async def execute(
        self,
        url: str,
        max_chars: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Fetch a web page and return raw HTML.

        Args:
            url: URL to fetch
            max_chars: Max characters to return

        Returns:
            ToolResult with raw HTML content
        """
        try:
            log.info("Fetching URL (raw HTML mode)", url=url)
            cfg = get_config()
            configured_max = int(getattr(cfg.tools.web_fetch, "max_chars", 100000))
            effective_max_chars = configured_max if max_chars is None else int(max_chars)
            effective_max_chars = max(1, effective_max_chars)

            response = await self.client.get(url)
            response.raise_for_status()

            content = response.text
            if len(content) > effective_max_chars:
                content = content[:effective_max_chars] + "\n... [truncated]"

            output = f"[URL: {url}]\n"
            output += f"[Status: {response.status_code}]\n"
            output += f"[Mode: html]\n"
            output += f"[Size: {len(response.text)} chars]\n\n"
            output += content

            return ToolResult(
                success=True,
                content=output,
            )

        except httpx.HTTPError as e:
            log.error("HTTP fetch failed (raw)", url=url, error=str(e))
            return ToolResult(
                success=False,
                error=f"HTTP error: {e}",
            )
        except Exception as e:
            log.error("Fetch failed (raw)", url=url, error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
