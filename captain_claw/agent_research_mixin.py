"""Research extraction pipeline for Agent.

This mixin handles web-search-based research for scale tasks:
- Keyword extraction from task descriptions
- Search query building with disambiguation hints
- URL parsing from search results
- Embedded URL splitting (entity name + URL)
- Full research pipeline: embedded URL fetch → web search → supplementary fetches
"""

import re
from typing import Any

from captain_claw.config import get_config
from captain_claw.llm import Message
from captain_claw.logging import get_logger


log = get_logger(__name__)


class AgentResearchMixin:
    """Research extraction: web search, URL fetch, and content synthesis."""

    # Regex to extract URLs from Brave search output.
    _SEARCH_RESULT_URL_RE = re.compile(r"^\s*URL:\s*(https?://\S+)", re.MULTILINE)

    # ------------------------------------------------------------------
    # Keyword extraction (once per task, cached)
    # ------------------------------------------------------------------

    async def _extract_research_keywords(
        self,
        task_description: str,
        all_items: list[str],
        turn_usage: dict[str, int],
    ) -> list[str]:
        """Ask the LLM to extract search keywords from the task description.

        Called **once** before the micro-loop starts (not per item).
        The result is cached in ``_scale_progress["_research_keywords"]``
        so every item reuses the same keyword set.

        Returns a list of keyword strings (may include quoted phrases).
        """
        cfg = get_config()
        max_kw = cfg.scale.research_query_keywords

        # Build a compact sample of item names so the LLM knows what
        # the list members look like (and can exclude them).
        sample = all_items[:15]
        items_str = ", ".join(sample)
        if len(all_items) > 15:
            items_str += f" … ({len(all_items)} total)"

        messages = [
            Message(
                role="system",
                content=(
                    "You are a search-query keyword extractor. "
                    "Given a task description and a list of entities that will be "
                    "searched individually, extract the most useful web-search "
                    "keywords that should accompany EVERY entity search.\n\n"
                    "Rules:\n"
                    "- Return ONLY the keywords, one per line, nothing else.\n"
                    "- Multi-word proper nouns should be wrapped in double quotes "
                    '(e.g. "Silicon Gardens").\n'
                    "- Include contextual identifiers (parent company, fund name, "
                    "industry, geography) that help narrow results.\n"
                    "- Include useful data-source names mentioned in the task "
                    "(e.g. Crunchbase, PitchBook, LinkedIn).\n"
                    "- EXCLUDE the entity names themselves — they will be added "
                    "separately.\n"
                    "- EXCLUDE generic instruction words (search, extract, find, "
                    "compile, report, format, write, etc.).\n"
                    "- EXCLUDE output field labels (company, founders, country, "
                    "date, amount, etc.) — these describe output columns, not "
                    "search terms.\n"
                    f"- Return at most {max_kw} keywords, ordered by importance.\n"
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Task description:\n{task_description}\n\n"
                    f"Entity list (will be searched one by one): {items_str}\n\n"
                    f"Extract up to {max_kw} search keywords:"
                ),
            ),
        ]

        try:
            response = await self._complete_with_guards(
                messages=messages,
                tools=None,
                interaction_label="research_keyword_extraction",
                turn_usage=turn_usage,
                max_tokens=min(200, int(cfg.model.max_tokens)),
            )
            raw = (response.content or "").strip()
        except Exception as e:
            log.warning("Research keyword extraction failed", error=str(e))
            return []

        # Parse: one keyword per line, strip whitespace and bullets.
        keywords: list[str] = []
        for line in raw.splitlines():
            line = line.strip().lstrip("-•*123456789. )")
            if not line:
                continue
            keywords.append(line)
            if len(keywords) >= max_kw:
                break

        log.info(
            "Research keywords extracted",
            keywords=keywords,
            count=len(keywords),
        )
        return keywords

    # ------------------------------------------------------------------
    # Search query building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_research_search_query(
        item: str,
        keywords: list[str],
    ) -> str:
        """Build a web-search query for a plain-text entity.

        Combines the item name with pre-extracted keywords
        (produced once by ``_extract_research_keywords``).
        The item is NOT quoted to allow fuzzy/keyword matching —
        exact-match quotes cause zero results for obscure entities.
        """
        parts: list[str] = [item.strip()] + keywords

        query = " ".join(parts)
        # Cap at ~150 chars to stay within search engine limits.
        if len(query) > 150:
            query = query[:150].rsplit(" ", 1)[0]
        return query.strip()

    # ------------------------------------------------------------------
    # URL parsing from search results
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_search_result_urls(search_content: str) -> list[str]:
        """Extract URLs from Brave web-search tool output.

        The output format is::

            1. Title
               URL: https://example.com/...
               Snippet: ...
        """
        urls: list[str] = []
        for match in AgentResearchMixin._SEARCH_RESULT_URL_RE.finditer(search_content):
            url = match.group(1).strip().rstrip(".")
            if url and url not in urls:
                urls.append(url)
        return urls

    # ------------------------------------------------------------------
    # Embedded URL splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_embedded_url(item: str) -> tuple[str, str]:
        """Split an item into (entity_name, embedded_url).

        For items like ``"Company Name — https://example.com"`` returns
        ``("Company Name", "https://example.com")``.
        For plain URLs or plain text returns ``("", url)`` or ``(item, "")``.
        """
        # Pure URL — no entity name
        stripped = item.strip()
        if stripped.startswith(("http://", "https://")):
            return "", stripped
        # Entity with embedded URL
        url_match = re.search(r"https?://[^\s)\]}>\"']+", stripped)
        if url_match:
            entity = stripped[:url_match.start()].strip().rstrip("—–-:.|,").strip()
            return entity, url_match.group(0)
        # Plain text entity — no URL
        return stripped, ""

    # ------------------------------------------------------------------
    # Full research pipeline
    # ------------------------------------------------------------------

    async def _extract_research_item(
        self,
        item: str,
        task_description: str,
        item_num: int,
        total: int,
        turn_usage: dict[str, int],
        session_policy: dict[str, Any] | None = None,
        task_policy: dict[str, Any] | None = None,
        source_context: str = "",
    ) -> str:
        """Research a single entity using provided URLs and/or web search.

        When the item contains an embedded URL (e.g. ``"Company — https://…"``),
        that URL is fetched FIRST as the primary source.  Web search is then
        used to find supplementary sources.  When no embedded URL is present,
        falls back to pure web-search-based research.

        ``source_context`` optionally provides metadata from the original
        article (e.g. country, brief description) which is used to refine
        the search query and help disambiguate common entity names.

        Returns concatenated content from all sources ready for the LLM
        processing step.  On total failure returns a short error string.
        """
        cfg = get_config()
        max_results = cfg.scale.research_search_results
        max_chars = cfg.scale.research_max_chars_per_fetch

        entity_name, embedded_url = self._extract_embedded_url(item)
        # Use entity name for search queries; fall back to full item text
        search_entity = entity_name or item.strip()
        # If source context is available, append it to help disambiguate
        # common names in web search (e.g. "Acme Corp Germany" instead
        # of just "Acme Corp" which might find a different company).
        # Keep it short — truncate to avoid bloating the search query.
        if source_context:
            _ctx_hint = source_context.strip()[:80]
            search_entity = f"{search_entity} {_ctx_hint}"

        parts: list[str] = [f"# Research: {item}\n"]
        fetched_ok = 0

        # ── 1. Fetch embedded URL first (primary source) ─────────
        if embedded_url:
            self._emit_thinking(
                f"scale_micro_loop: Fetching provided URL ({item_num}/{total})\n"
                f"{item}\n"
                f"{embedded_url}",
                tool="scale_micro_loop",
                phase="tool",
            )
            try:
                primary_result = await self._execute_tool_with_guard(
                    name="web_fetch",
                    arguments={"url": embedded_url, "max_chars": max_chars},
                    interaction_label=f"scale_research_primary_{item_num}",
                    turn_usage=turn_usage,
                    session_policy=session_policy,
                    task_policy=task_policy,
                )
                if primary_result.success and primary_result.content:
                    parts.append(
                        f"\n--- Primary source: {embedded_url} ---\n"
                        f"{primary_result.content[:max_chars]}\n"
                    )
                    fetched_ok += 1
                else:
                    parts.append(
                        f"\n--- Primary source: {embedded_url} ---\n"
                        f"Fetch failed: {primary_result.error or 'empty content'}\n"
                    )
            except Exception as exc:
                parts.append(
                    f"\n--- Primary source: {embedded_url} ---\n"
                    f"Fetch error: {exc}\n"
                )

            self._emit_tool_output(
                "scale_micro_loop",
                {"item": item, "step": "research_primary", "url": embedded_url},
                f"[{item_num}/{total}] Primary fetch: {embedded_url}",
            )

        # ── 2. Web search for supplementary sources ──────────────
        keywords = (
            self._scale_progress.get("_research_keywords", [])
            if self._scale_progress
            else []
        )
        query = self._build_research_search_query(search_entity, keywords)

        self._emit_thinking(
            f"scale_micro_loop: Searching ({item_num}/{total})\n"
            f"{item}\n"
            f"query: {query}",
            tool="scale_micro_loop",
            phase="tool",
        )

        search_result = await self._execute_tool_with_guard(
            name="web_search",
            arguments={"query": query, "count": max_results},
            interaction_label=f"scale_research_search_{item_num}",
            turn_usage=turn_usage,
            session_policy=session_policy,
            task_policy=task_policy,
        )

        if not search_result.success or not search_result.content:
            if fetched_ok > 0:
                # Primary URL was fetched — search failure is not fatal
                parts.append(f"\nSupplementary search failed: {search_result.error or 'no results'}\n")
                return "\n".join(parts)
            return f"Web search failed for '{item}': {search_result.error or 'no results'}"

        self._emit_tool_output(
            "scale_micro_loop",
            {"item": item, "step": "research_search", "query": query},
            f"[{item_num}/{total}] Search: {query}",
        )

        # ── 3. Parse URLs from search results ────────────────────
        urls = self._parse_search_result_urls(search_result.content)
        # Exclude the embedded URL to avoid fetching it twice
        if embedded_url:
            urls = [u for u in urls if u != embedded_url]
        if not urls and fetched_ok == 0:
            # No URLs at all — fall back to search snippets
            return f"# Research: {item}\nSearch query: {query}\n\n{search_result.content}"

        # ── 4. Fetch supplementary URLs ──────────────────────────
        # Reduce supplementary fetches when we already have a primary source
        _supp_limit = max(1, max_results - 1) if embedded_url else max_results

        for fetch_idx, url in enumerate(urls[:_supp_limit], start=1):
            self._emit_thinking(
                f"scale_micro_loop: Fetching source ({fetch_idx}/{min(len(urls), _supp_limit)}) "
                f"for item {item_num}/{total}\n"
                f"{item}\n"
                f"{url}",
                tool="scale_micro_loop",
                phase="tool",
            )
            try:
                fetch_result = await self._execute_tool_with_guard(
                    name="web_fetch",
                    arguments={"url": url, "max_chars": max_chars},
                    interaction_label=f"scale_research_fetch_{item_num}_{fetch_idx}",
                    turn_usage=turn_usage,
                    session_policy=session_policy,
                    task_policy=task_policy,
                )
                if fetch_result.success and fetch_result.content:
                    parts.append(
                        f"\n--- Source {fetch_idx}: {url} ---\n"
                        f"{fetch_result.content[:max_chars]}\n"
                    )
                    fetched_ok += 1
                else:
                    parts.append(
                        f"\n--- Source {fetch_idx}: {url} ---\n"
                        f"Fetch failed: {fetch_result.error or 'empty content'}\n"
                    )
            except Exception as exc:
                parts.append(
                    f"\n--- Source {fetch_idx}: {url} ---\n"
                    f"Fetch error: {exc}\n"
                )

        self._emit_tool_output(
            "scale_micro_loop",
            {"item": item, "step": "research_fetch", "fetched": fetched_ok, "total_urls": len(urls[:_supp_limit]), "had_primary": bool(embedded_url)},
            f"[{item_num}/{total}] Fetched {fetched_ok} sources{' (incl. primary URL)' if embedded_url else ''} for: {item}",
        )

        # If all fetches failed, fall back to search snippets.
        if fetched_ok == 0:
            return f"# Research: {item}\nSearch query: {query}\n\n{search_result.content}"

        return "\n".join(parts)
