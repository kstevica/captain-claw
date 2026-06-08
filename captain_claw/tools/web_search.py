"""Web search tool powered by Brave Search API or Tavily."""

import os
import re
from typing import Any

import httpx

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# Brave Search rejects the whole request (HTTP 422) if `country` isn't one of
# its supported codes — e.g. "HR" (Croatia) is NOT supported, which silently
# breaks every regional query. Validate against this set and drop anything else
# (search globally) instead of forwarding an invalid value.
_BRAVE_COUNTRIES = frozenset({
    "ALL", "AR", "AU", "AT", "BE", "BR", "CA", "CL", "DK", "FI", "FR", "DE",
    "GR", "HK", "IN", "ID", "IT", "JP", "KR", "MY", "MX", "NL", "NZ", "NO",
    "CN", "PL", "PT", "PH", "RU", "SA", "ZA", "ES", "SE", "CH", "TW", "TR",
    "GB", "US",
})


class WebSearchTool(Tool):
    """Search the web using Brave Search API or Tavily."""

    name = "web_search"
    description = "Search the web and return ranked results with titles, links, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text",
            },
            "count": {
                "type": "number",
                "description": "Maximum results to return (default from config, max 20)",
            },
            "offset": {
                "type": "number",
                "description": "Result offset for pagination (default 0)",
            },
            "country": {
                "type": "string",
                "description": (
                    "Optional 2-letter country for regional ranking. Brave supports a "
                    "limited set (US, GB, DE, FR, IT, ES, etc.); unsupported codes (e.g. HR) "
                    "are ignored. For local results, prefer search_lang (e.g. 'hr') and "
                    "query terms instead."
                ),
            },
            "search_lang": {
                "type": "string",
                "description": "Search language code (example: en, hr)",
            },
            "freshness": {
                "type": "string",
                "description": "Freshness filter (example: pd, pw, pm, py)",
            },
            "safesearch": {
                "type": "string",
                "description": "SafeSearch level: off, moderate, strict",
            },
        },
        "required": ["query"],
    }

    def __init__(self):
        self.client = httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": "Captain Claw/0.1.0 (Web Search Tool)"},
        )

    @staticmethod
    def _clean_text(value: str, max_chars: int = 500) -> str:
        """Normalize whitespace and bound output size."""
        cleaned = re.sub(r"\s+", " ", (value or "")).strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "... [truncated]"

    async def execute(
        self,
        query: str,
        count: int | None = None,
        offset: int = 0,
        country: str = "",
        search_lang: str = "",
        freshness: str = "",
        safesearch: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Brave web search."""
        q = (query or "").strip()
        if not q:
            return ToolResult(success=False, error="Missing required query")

        cfg = get_config()
        search_cfg = getattr(cfg.tools, "web_search", None)
        provider = str(getattr(search_cfg, "provider", "brave") or "brave").strip().lower()

        if provider == "tavily":
            if any([offset, country.strip(), freshness.strip(), safesearch.strip()]):
                log.debug("Tavily provider ignores offset/country/freshness/safesearch parameters")
            return await self._execute_tavily(q, search_cfg, count=count)

        if provider != "brave":
            return ToolResult(success=False, error=f"Unsupported web_search provider: {provider}")

        api_key = (
            str(getattr(search_cfg, "api_key", "") or "").strip()
            or str(os.environ.get("BRAVE_API_KEY", "")).strip()
        )
        if not api_key:
            return ToolResult(
                success=False,
                error=(
                    "Missing Brave API key. Set tools.web_search.api_key in config "
                    "or BRAVE_API_KEY environment variable."
                ),
            )

        base_url = str(
            getattr(search_cfg, "base_url", "https://api.search.brave.com/res/v1/web/search")
            or "https://api.search.brave.com/res/v1/web/search"
        ).strip()
        timeout = float(getattr(search_cfg, "timeout", 20) or 20)
        default_count = int(getattr(search_cfg, "max_results", 5) or 5)
        effective_count = default_count if count is None else int(count)
        effective_count = min(max(effective_count, 1), 20)
        effective_offset = max(0, int(offset))
        safe_value = (safesearch or str(getattr(search_cfg, "safesearch", "moderate") or "moderate")).strip().lower()
        if safe_value not in {"off", "moderate", "strict"}:
            safe_value = "moderate"

        params: dict[str, Any] = {
            "q": q,
            "count": effective_count,
            "offset": effective_offset,
            "safesearch": safe_value,
        }
        if country.strip():
            _cc = country.strip().upper()
            if _cc in _BRAVE_COUNTRIES:
                params["country"] = _cc
            else:
                # Unsupported by Brave (e.g. HR) — drop it and search globally
                # rather than 422 the whole request.
                log.debug("web_search: dropping unsupported Brave country %r", country.strip())
        if search_lang.strip():
            params["search_lang"] = search_lang.strip()
        if freshness.strip():
            params["freshness"] = freshness.strip()

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }

        try:
            response = await self.client.get(
                base_url,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()

            web_block = payload.get("web", {}) if isinstance(payload, dict) else {}
            results = web_block.get("results", []) if isinstance(web_block, dict) else []
            if not isinstance(results, list):
                results = []

            lines = [
                f"[SEARCH ENGINE: Brave]",
                f"[QUERY: {q}]",
                f"[RESULTS: {len(results)}]",
                "",
            ]

            if not results:
                lines.append("No results found.")
            else:
                for idx, item in enumerate(results, start=1):
                    if not isinstance(item, dict):
                        continue
                    title = self._clean_text(str(item.get("title", "") or "Untitled"), max_chars=180)
                    link = str(item.get("url", "") or "").strip()
                    desc = self._clean_text(str(item.get("description", "") or ""))
                    lines.append(f"{idx}. {title}")
                    lines.append(f"   URL: {link or '-'}")
                    lines.append(f"   Snippet: {desc or '-'}")
                    lines.append("")

            return ToolResult(success=True, content="\n".join(lines).strip())
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = (e.response.text or "").strip()
            except Exception:
                body = ""
            detail = f"HTTP {e.response.status_code}" if e.response is not None else str(e)
            if body:
                detail = f"{detail}: {self._clean_text(body, max_chars=300)}"
            log.error("Brave web search failed", query=q, error=detail)
            return ToolResult(success=False, error=detail)
        except Exception as e:
            log.error("Web search failed", query=q, error=str(e))
            return ToolResult(success=False, error=str(e))

    async def _execute_tavily(
        self,
        query: str,
        search_cfg: Any,
        count: int | None = None,
    ) -> ToolResult:
        """Execute a Tavily web search."""
        api_key = (
            str(getattr(search_cfg, "tavily_api_key", "") or "").strip()
            or str(os.environ.get("TAVILY_API_KEY", "")).strip()
        )
        if not api_key:
            return ToolResult(
                success=False,
                error=(
                    "Missing Tavily API key. Set tools.web_search.tavily_api_key in config "
                    "or TAVILY_API_KEY environment variable."
                ),
            )

        timeout = float(getattr(search_cfg, "timeout", 20) or 20)
        default_count = int(getattr(search_cfg, "max_results", 5) or 5)
        effective_count = default_count if count is None else int(count)
        effective_count = min(max(effective_count, 1), 20)

        payload: dict[str, Any] = {
            "query": query,
            "max_results": effective_count,
            "search_depth": "basic",
            "include_answer": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        try:
            response = await self.client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", []) if isinstance(data, dict) else []
            if not isinstance(results, list):
                results = []

            lines = [
                "[SEARCH ENGINE: Tavily]",
                f"[QUERY: {query}]",
                f"[RESULTS: {len(results)}]",
                "",
            ]

            if not results:
                lines.append("No results found.")
            else:
                for idx, item in enumerate(results, start=1):
                    if not isinstance(item, dict):
                        continue
                    title = self._clean_text(str(item.get("title", "") or "Untitled"), max_chars=180)
                    link = str(item.get("url", "") or "").strip()
                    desc = self._clean_text(str(item.get("content", "") or ""))
                    lines.append(f"{idx}. {title}")
                    lines.append(f"   URL: {link or '-'}")
                    lines.append(f"   Snippet: {desc or '-'}")
                    lines.append("")

            return ToolResult(success=True, content="\n".join(lines).strip())
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = (e.response.text or "").strip()
            except Exception:
                body = ""
            detail = f"HTTP {e.response.status_code}" if e.response is not None else str(e)
            if body:
                detail = f"{detail}: {self._clean_text(body, max_chars=300)}"
            log.error("Tavily web search failed", query=query, error=detail)
            return ToolResult(success=False, error=detail)
        except Exception as e:
            log.error("Web search failed", query=query, error=str(e))
            return ToolResult(success=False, error=str(e))

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

