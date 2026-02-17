"""Web fetch tool for retrieving web page content."""

import asyncio
from typing import Any

import httpx

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class WebFetchTool(Tool):
    """Fetch web page content."""

    name = "web_fetch"
    description = "Fetch and extract readable content from a URL."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch",
            },
            "max_chars": {
                "type": "number",
                "description": "Maximum characters to return (default: 10000)",
            },
            "extract_mode": {
                "type": "string",
                "description": " Extraction mode: 'markdown' or 'text' (default: markdown)",
            },
        },
        "required": ["url"],
    }

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Captain Claw/0.1.0 (Web Fetch Tool)",
            },
        )

    async def execute(
        self,
        url: str,
        max_chars: int = 10000,
        extract_mode: str = "markdown",
        **kwargs: Any,
    ) -> ToolResult:
        """Fetch a web page.
        
        Args:
            url: URL to fetch
            max_chars: Max characters to return
            extract_mode: Extraction mode
        
        Returns:
            ToolResult with page content
        """
        try:
            log.info("Fetching URL", url=url)
            
            response = await self.client.get(url)
            response.raise_for_status()
            
            content = response.text
            
            # Simple extraction - in production would use proper HTML parsing
            # For now, just return raw content trimmed
            if len(content) > max_chars:
                content = content[:max_chars] + "\n... [truncated]"
            
            # Add header with URL info
            output = f"[URL: {url}]\n"
            output += f"[Status: {response.status_code}]\n"
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
