# Summary: agent_research_mixin.py

# agent_research_mixin.py Summary

## Summary
This mixin implements a multi-stage web research pipeline for extracting information about entities in scale tasks. It orchestrates keyword extraction, search query building, URL parsing, and content fetching to compile comprehensive research materials from web sources. The pipeline prioritizes embedded URLs as primary sources, supplements with web search results, and gracefully degrades to search snippets when fetches fail.

## Purpose
Solves the problem of systematically researching multiple entities (companies, people, funds, etc.) at scale by automating the discovery and aggregation of relevant web content. Handles the complexity of query disambiguation, URL extraction from search results, and intelligent source prioritization while managing API rate limits and content size constraints.

## Most Important Functions/Classes/Procedures

1. **`_extract_research_keywords(task_description, all_items, turn_usage)`**
   - LLM-driven keyword extraction called once per task and cached for reuse across all items. Analyzes task context and entity list to generate 3-15 contextual search keywords (company names, fund identifiers, geographies, data sources) while excluding generic terms and entity names themselves. Returns list of keyword strings for query augmentation.

2. **`_extract_research_item(item, task_description, item_num, total, turn_usage, ...)`**
   - Core research pipeline orchestrator. Executes three-stage fetch strategy: (1) fetches embedded URL if present as primary source, (2) performs web search with augmented query, (3) fetches top supplementary URLs from search results. Includes fallback logic to return search snippets if all fetches fail. Returns concatenated content from all sources or error message.

3. **`_build_research_search_query(item, keywords)`**
   - Constructs optimized web search queries by combining entity name with pre-extracted keywords. Avoids quoting entity names to enable fuzzy matching for obscure entities. Caps query length at ~150 characters to respect search engine limits.

4. **`_parse_search_result_urls(search_content)`**
   - Regex-based URL extraction from Brave search tool output format. Parses structured search results containing title, URL, and snippet fields. Returns deduplicated list of valid URLs.

5. **`_extract_embedded_url(item)`**
   - Utility for splitting items into entity name and embedded URL components. Handles three cases: pure URLs (returns empty entity), entity with embedded URL (parses and returns both), and plain text (returns entity with empty URL). Supports various URL delimiters (em-dash, colon, etc.).

## Architecture & Dependencies

**Key Dependencies:**
- `captain_claw.config`: Configuration access for research limits (max keywords, search results, character limits)
- `captain_claw.llm`: Message class for LLM communication
- `captain_claw.logging`: Structured logging
- Mixin expects host class to provide: `_complete_with_guards()`, `_execute_tool_with_guard()`, `_emit_thinking()`, `_emit_tool_output()`, `_scale_progress` state dictionary

**System Role:**
Operates as a research layer within a larger agent scale-task system. Sits between task description parsing and LLM-based content synthesis. Manages tool interactions (web_search, web_fetch) with guard rails and usage tracking. Caches keyword extraction results to optimize repeated searches across entity lists.

**Design Patterns:**
- Single-call caching: Keywords extracted once, reused for all items
- Graceful degradation: Falls back from fetches → search snippets → error messages
- Source prioritization: Embedded URLs > search results > fallback content
- Query optimization: Context-aware disambiguation hints appended to search queries