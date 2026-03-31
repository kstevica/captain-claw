"""Prompt/message context assembly helpers for Agent."""

import asyncio
import importlib.util
import inspect
import json
import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from captain_claw.config import get_config
from captain_claw.llm import Message, ToolCall, get_provider, set_provider
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool


log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tool prompt descriptions — used to build the dynamic tool list in the system
# prompt.  Only tools that should appear in the textual list need an entry;
# tools without one still get their API definition sent to the model.
# ---------------------------------------------------------------------------

_TOOL_PROMPT_DESCRIPTIONS: dict[str, str] = {
    "shell": "Execute shell commands in the terminal",
    "read": "Read file contents from the filesystem",
    "write": "Write content to files",
    "edit": "Modify existing files by replacing specific text (find-and-replace)",
    "glob": "Find files by pattern (ALWAYS use this instead of shell find/ls for file searching — it automatically searches extra read folders too)",
    "web_fetch": "Fetch a URL and return clean readable TEXT (always text mode, never raw HTML)",
    "web_get": "Fetch a URL and return raw HTML source (only for scraping/DOM inspection)",
    "web_search": "Search the web for up-to-date sources",
    "pdf_extract": "Extract a single .pdf file into markdown. ONLY for .pdf files. For multiple files in a folder use summarize_files instead.",
    "docx_extract": "Extract a single .docx file into markdown. ONLY for .docx files — never use on .pdf/.xlsx/.pptx. For multiple files in a folder use summarize_files instead.",
    "xlsx_extract": "Extract a single .xlsx file into markdown tables. ONLY for .xlsx files — never use on .pdf/.docx/.pptx. For multiple files in a folder use summarize_files instead.",
    "pptx_extract": "Extract a single .pptx file into markdown. ONLY for .pptx files — never use on .pdf/.docx/.xlsx. For multiple files in a folder use summarize_files instead.",
    "pocket_tts": "Convert text to local speech audio and save as MP3",
    "send_mail": "Send emails via SMTP. Supports to, cc, bcc, subject, body, and file attachments.",
    "clipboard": "Read or write the system clipboard. Supports text, images, and files.",
    "gws": "Google Workspace CLI — access Google Drive (list, search, download, create), Docs (read, append), Calendar (list, search, create, agenda), and Gmail (list, search, read, threads). Uses the `gws` binary.",
    "datastore": "Manage persistent relational data tables (create, query, insert, update, delete, import/export)",
    "insights": "Search and manage persistent cross-session insights — facts, contacts, decisions, preferences, deadlines auto-extracted from conversations. Actions: search, list, add, update, delete.",
    "personality": "Read or update the agent personality profile (name, description, background, expertise)",
    "browser": "Control a headless browser for web app interaction. Supports observe/act (page understanding), click/type with nth-match disambiguation, login with encrypted credentials + cookie persistence, network capture for API discovery, API replay (execute captured APIs directly — skip the browser!), and multi-app sessions. Use for login flows, form filling, and interacting with dynamic/React web apps.",
    "direct_api": "Register, manage, and execute HTTP API endpoints directly. Users define endpoints with URL, method, description, and payload schemas. Supports auth capture from browser sessions. Methods: GET, POST, PUT, PATCH (DELETE is rejected for safety).",
    "termux": "Interact with the Android device via Termux API (take photo, battery status, GPS location, torch on/off)",
    "summarize_files": "IMPORTANT: When the user asks you to go through, review, analyse, or summarise multiple files or documents in a folder, ALWAYS use this tool FIRST instead of reading/extracting files one by one. This tool handles the entire pipeline internally (reads all files including PDF/DOCX/XLSX/PPTX, summarises each one via LLM, combines into final output) and returns only the output file path — massively saving context. After getting the summary file, you can read it and use it to write reports, answer questions, etc.",
    "desktop_action": "Control the desktop: click/type/scroll at screen coordinates, press keys/hotkeys, drag, open apps/folders/URLs. Use with screen_capture to see the screen first, then act on it. The 'screenshot_click' action chains screenshot+vision+click in one call — describe the element and it finds and clicks it automatically.",
}

_TOOL_PROMPT_DESCRIPTIONS_MICRO: dict[str, str] = {
    "shell": "terminal commands",
    "read": "read files",
    "write": "write files",
    "edit": "modify files by replacing text",
    "glob": "find files by pattern",
    "web_fetch": "clean text from URL",
    "web_get": "raw HTML from URL",
    "web_search": "web search",
    "pdf_extract": "single .pdf → markdown (for multiple files use summarize_files)",
    "docx_extract": "single .docx → markdown (ONLY .docx, never .pdf; for multiple files use summarize_files)",
    "xlsx_extract": "single .xlsx → markdown (ONLY .xlsx, never .pdf; for multiple files use summarize_files)",
    "pptx_extract": "single .pptx → markdown (ONLY .pptx, never .pdf; for multiple files use summarize_files)",
    "pocket_tts": "text-to-speech MP3",
    "send_mail": "send emails via SMTP",
    "clipboard": "read/write system clipboard",
    "gws": "Google Workspace: Drive, Docs, Calendar, Gmail",
    "datastore": "persistent relational tables",
    "insights": "persistent cross-session insights (facts, contacts, decisions, deadlines)",
    "personality": "agent personality profile",
    "browser": "headless browser for dynamic web apps",
    "direct_api": "register and call HTTP endpoints",
    "termux": "Android device: photo/battery/location/torch",
    "summarize_files": "ALWAYS use for reviewing/analysing/summarising multiple files in a folder — handles PDF/DOCX/XLSX/PPTX internally, returns summary file path",
    "desktop_action": "desktop GUI: click/type/scroll/keys/open apps (use with screen_capture)",
}


class AgentContextMixin:
    """Build system/context/tool messages for model calls."""
    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        """Extract unique URLs from text in appearance order."""
        raw = re.findall(r"https?://[^\s)\]}>\"']+", text or "")
        seen: set[str] = set()
        urls: list[str] = []
        for url in raw:
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
        return urls

    @staticmethod
    def _merge_unique_urls(*url_lists: list[str]) -> list[str]:
        """Merge URL lists while preserving first-seen order."""
        seen: set[str] = set()
        merged: list[str] = []
        for url_list in url_lists:
            for url in url_list:
                if url in seen:
                    continue
                seen.add(url)
                merged.append(url)
        return merged

    @staticmethod
    def _normalize_tool_policy_payload(raw: Any) -> dict[str, Any] | None:
        """Normalize policy payload shape for registry consumption."""
        if not isinstance(raw, dict):
            return None

        allow_raw = raw.get("allow")
        if allow_raw is None:
            allow: list[str] | None = None
        elif isinstance(allow_raw, list):
            allow = [str(item).strip() for item in allow_raw if str(item).strip()]
        else:
            return None

        deny_raw = raw.get("deny", [])
        deny = [str(item).strip() for item in deny_raw] if isinstance(deny_raw, list) else []
        deny = [item for item in deny if item]

        also_allow_raw = raw.get("also_allow", raw.get("alsoAllow", []))
        also_allow = (
            [str(item).strip() for item in also_allow_raw]
            if isinstance(also_allow_raw, list)
            else []
        )
        also_allow = [item for item in also_allow if item]

        if allow is None and not deny and not also_allow:
            return None
        return {
            "allow": allow,
            "deny": deny,
            "also_allow": also_allow,
        }

    def _session_tool_policy_payload(self) -> dict[str, Any] | None:
        """Load session-level tool policy from session metadata when present."""
        if not self.session or not isinstance(self.session.metadata, dict):
            return None
        return self._normalize_tool_policy_payload(self.session.metadata.get("tool_policy"))

    def _active_task_tool_policy_payload(self, planning_pipeline: dict[str, Any] | None) -> dict[str, Any] | None:
        """Load active task-level tool policy from pipeline task metadata."""
        if not isinstance(planning_pipeline, dict):
            return None
        graph = planning_pipeline.get("task_graph")
        if not isinstance(graph, dict):
            return None

        candidate_ids: list[str] = []
        current_id = str(planning_pipeline.get("current_task_id", "")).strip()
        if current_id:
            candidate_ids.append(current_id)
        raw_active = planning_pipeline.get("active_task_ids", [])
        if isinstance(raw_active, list):
            for item in raw_active:
                task_id = str(item).strip()
                if task_id and task_id not in candidate_ids:
                    candidate_ids.append(task_id)

        for task_id in candidate_ids:
            node = graph.get(task_id)
            if not isinstance(node, dict):
                continue
            normalized = self._normalize_tool_policy_payload(node.get("tool_policy"))
            if normalized is not None:
                return normalized
        return None

    def _extract_source_links(self, msg: dict[str, Any], content: str) -> list[str]:
        """Extract source links from both tool content and structured tool arguments.

        For tool result messages (role="tool") whose tool produced a large
        fetched page (web_fetch, web_get), we only extract the URLs from
        the tool's *arguments* (the URL that was fetched), NOT from the
        returned content.  The fetched content contains every link on the
        target page (navigation, categories, ads, etc.) which would pollute
        the recent_source_urls list fed to the task contract planner and
        cause wasteful prefetching.
        """
        args_links: list[str] = []
        args = msg.get("tool_arguments")
        if isinstance(args, dict):
            for key in ("url", "href", "link", "source_url"):
                value = args.get(key)
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    args_links.append(value)
            for key in ("urls", "links"):
                values = args.get(key)
                if isinstance(values, list):
                    for value in values:
                        if isinstance(value, str) and value.startswith(("http://", "https://")):
                            args_links.append(value)
        # For tool results from web_fetch/web_get, the content body is the
        # fetched page itself — every link on it is noise for the planner.
        # Only use the argument URL in that case.
        tool_name = str(msg.get("tool_name", "")).strip().lower()
        skip_content_extraction = (
            msg.get("role") == "tool"
            and tool_name in ("web_fetch", "web_get")
        )
        if skip_content_extraction:
            return args_links
        content_links = self._extract_urls(content)
        return self._merge_unique_urls(args_links, content_links)

    @staticmethod
    def _extract_mentioned_domains(text: str) -> set[str]:
        """Extract domain names from URLs and domain-like tokens in *text*.

        Returns a set of lowercased hostnames (e.g. ``{"example.com", "www.example.com"}``).
        Used to scope ``_collect_recent_source_urls`` so that only URLs
        relevant to the current request are included.
        """
        domains: set[str] = set()
        # 1. Domains from full URLs
        for url in re.findall(r"https?://[^\s)\]}>\"']+", text or ""):
            try:
                host = urlparse(url).hostname
                if host:
                    domains.add(host.lower())
            except Exception:
                pass
        # 2. Bare domain-like tokens  (e.g. "example.com", "news.io")
        for token in re.findall(r"\b([a-zA-Z0-9-]+\.[a-zA-Z]{2,})\b", text or ""):
            candidate = token.lower()
            # Simple validation — must have at least one dot and a known-ish TLD length
            if "." in candidate and len(candidate.split(".")[-1]) >= 2:
                domains.add(candidate)
        return domains

    @staticmethod
    def _url_matches_domains(url: str, domains: set[str]) -> bool:
        """Check whether *url*'s hostname matches any entry in *domains*."""
        try:
            host = urlparse(url).hostname
        except Exception:
            return False
        if not host:
            return False
        host = host.lower()
        for domain in domains:
            # Match exact host or host ends with ".domain"
            if host == domain or host.endswith(f".{domain}"):
                return True
        return False

    def _collect_recent_source_urls(
        self,
        turn_start_idx: int,
        max_messages: int = 20,
        max_urls: int = 20,
        domain_filter: set[str] | None = None,
    ) -> list[str]:
        """Collect recent source URLs from messages before current turn.

        When *domain_filter* is provided and non-empty, only URLs whose
        hostname matches one of the filter domains are included.  This
        prevents unrelated URLs from earlier tasks (e.g. news-site URLs
        when the current request is about a different domain) from polluting the
        planner context and causing wasteful prefetches.

        When *domain_filter* is ``None`` or empty (no domain could be
        extracted from the current request), the scan window is reduced
        from *max_messages* to 5 to limit noise from older unrelated tasks.
        """
        if not self.session:
            return []
        # Narrow scan window when we have no domain signal.
        effective_max = max_messages if domain_filter else min(max_messages, 5)
        start = max(0, turn_start_idx - effective_max)
        urls: list[str] = []
        for msg in self.session.messages[start:turn_start_idx]:
            content = str(msg.get("content", ""))
            links = self._extract_source_links(msg, content)
            if domain_filter:
                links = [u for u in links if self._url_matches_domains(u, domain_filter)]
            if links:
                urls = self._merge_unique_urls(urls, links)
            if len(urls) >= max_urls:
                return urls[:max_urls]
        return urls[:max_urls]

    def _initialize_layered_memory(self) -> None:
        """Create layered memory manager for semantic retrieval."""
        if getattr(self, "memory", None) is not None:
            return
        cfg = get_config()
        memory_cfg = getattr(cfg, "memory", None)
        if memory_cfg is None or not bool(getattr(memory_cfg, "enabled", True)):
            self.memory = None
            return
        session_db_path = getattr(self.session_manager, "db_path", None)
        if session_db_path is None:
            self.memory = None
            return
        try:
            from captain_claw.memory import create_layered_memory

            self.memory = create_layered_memory(
                config=cfg,
                session_db_path=session_db_path,
                workspace_path=self.workspace_base_path,
            )
            if self.session:
                self.memory.set_active_session(self.session.id)
                self.memory.schedule_background_sync("agent_initialize")
        except Exception as e:
            log.warning("Failed to initialize layered memory", error=str(e))
            self.memory = None

        # Deep memory (Typesense-backed archive) — additional layer, not a
        # replacement for the SQLite semantic memory.
        self._deep_memory = None
        dm_cfg = getattr(cfg, "deep_memory", None)
        if dm_cfg is not None and bool(getattr(dm_cfg, "enabled", False)):
            try:
                from captain_claw.deep_memory import DeepMemoryIndex

                # Reuse the same embedding chain from semantic memory.
                embedding_chain = None
                if self.memory and getattr(self.memory, "semantic", None):
                    embedding_chain = getattr(self.memory.semantic, "embedding_chain", None)

                self._deep_memory = DeepMemoryIndex(
                    host=str(getattr(dm_cfg, "host", "localhost")),
                    port=int(getattr(dm_cfg, "port", 8108)),
                    protocol=str(getattr(dm_cfg, "protocol", "http")),
                    api_key=str(getattr(dm_cfg, "api_key", "")),
                    collection_name=str(getattr(dm_cfg, "collection_name", "captain_claw_deep_memory")),
                    embedding_dims=int(getattr(dm_cfg, "embedding_dims", 1536)),
                    auto_embed=bool(getattr(dm_cfg, "auto_embed", True)),
                    chunk_chars=int(getattr(getattr(cfg, "memory", None), "chunk_chars", 1400)) if getattr(cfg, "memory", None) else 1400,
                    chunk_overlap_chars=int(getattr(getattr(cfg, "memory", None), "chunk_overlap_chars", 200)) if getattr(cfg, "memory", None) else 200,
                    embedding_chain=embedding_chain,
                )
                log.info("Deep memory initialized", collection=str(getattr(dm_cfg, "collection_name", "")))
            except Exception as e:
                log.warning("Failed to initialize deep memory", error=str(e))
                self._deep_memory = None

        # Link deep memory into LayeredMemory so clear_all/close cover all layers.
        if self.memory is not None and self._deep_memory is not None:
            self.memory.deep = self._deep_memory

        # Wire up the L1/L2 summarizer for layered memory.
        self._wire_memory_summarizer()

    def _wire_memory_summarizer(self) -> None:
        """Attach an LLM-based summarizer to semantic and deep memory.

        The summarizer uses the agent's configured LLM provider (respecting
        API key, model, base_url, etc.) and tracks all token usage through
        the standard ``_accumulate_usage`` / ``_record_usage_to_db`` pipeline.
        """
        provider = getattr(self, "provider", None)
        if provider is None:
            return
        # Capture ``self`` (the agent) for usage tracking inside the closure.
        agent = self

        def _summarize_chunk(text: str) -> tuple[str, str]:
            """Generate (L1 one-liner, L2 summary) from chunk text using the agent's LLM."""
            if not text or len(text.strip()) < 20:
                return text.strip(), text.strip()
            import asyncio
            import time as _time

            from captain_claw.llm import LLMResponse, Message

            prompt = (
                "You are a memory indexer. Given the following text, produce exactly two lines:\n"
                "Line 1: A one-liner headline (max 100 chars) capturing the core idea.\n"
                "Line 2: A 1-2 sentence summary (max 300 chars) with enough context to assess relevance.\n\n"
                "Rules:\n"
                "- Output ONLY the two lines, nothing else.\n"
                "- No labels, prefixes, or numbering.\n\n"
                f"Text:\n{text[:2000]}"
            )
            messages = [Message(role="user", content=prompt)]
            t0 = _time.monotonic()
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        resp: LLMResponse = pool.submit(
                            asyncio.run,
                            provider.complete(messages, temperature=0.0, max_tokens=200),
                        ).result(timeout=15)
                else:
                    resp = asyncio.run(
                        provider.complete(messages, temperature=0.0, max_tokens=200)
                    )
                latency_ms = int((_time.monotonic() - t0) * 1000)

                # --- Track usage ---
                if resp.usage:
                    agent._accumulate_usage(agent.total_usage, resp.usage)
                try:
                    agent._record_usage_to_db(
                        interaction_label="memory_summarize_chunk",
                        messages=messages,
                        response=resp,
                        tools_enabled=False,
                        max_tokens=200,
                        latency_ms=latency_ms,
                        error=False,
                    )
                except Exception:
                    pass  # never fail the indexing flow
                # --- Monitor trace & session log ---
                try:
                    agent._emit_llm_trace(
                        interaction_label="memory_summarize_chunk",
                        response=resp,
                        messages=messages,
                        tools=None,
                        max_tokens=200,
                    )
                except Exception:
                    pass
                try:
                    agent._log_llm_call(
                        interaction_label="memory_summarize_chunk",
                        messages=messages,
                        response=resp,
                        tools_enabled=False,
                        max_tokens=200,
                    )
                except Exception:
                    pass

                output = (resp.content or "").strip()
                parts = output.split("\n", 1)
                l1 = parts[0].strip()[:120]
                l2 = parts[1].strip()[:400] if len(parts) > 1 else l1
                return l1, l2
            except Exception as exc:
                log.debug("Chunk summarization via LLM failed", error=str(exc))
                # Fallback: use first line as L1, first 300 chars as L2.
                first_line = text.strip().split("\n", 1)[0][:120]
                return first_line, text.strip()[:300]

        if self.memory and getattr(self.memory, "semantic", None):
            self.memory.semantic.set_summarizer(_summarize_chunk)
        if self._deep_memory is not None:
            self._deep_memory.set_summarizer(_summarize_chunk)

    def _build_todo_context_note(self) -> str:
        """Build compact context note from pending to-do items."""
        cfg = get_config()
        if not cfg.todo.enabled or not cfg.todo.inject_on_session_load:
            return ""
        session_id = self._current_session_slug() if self.session else None
        sm = getattr(self, "session_manager", None)
        if sm is None:
            return ""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already inside an event loop — use a sync-safe approach.
                # Check BEFORE creating the coroutine to avoid
                # "coroutine was never awaited" warnings.
                return self._build_todo_context_note_sync_cache()
            items = loop.run_until_complete(
                sm.get_todo_summary(session_id, cfg.todo.max_items_in_prompt)
            )
        except RuntimeError:
            # Fallback for edge cases (e.g. no current event loop).
            return self._build_todo_context_note_sync_cache()
        return self._format_todo_note(items)

    def _build_todo_context_note_sync_cache(self) -> str:
        """Fallback: use cached todo items when called inside an event loop."""
        items = getattr(self, "_todo_context_cache", None)
        if items is None:
            return ""
        return self._format_todo_note(items)

    @staticmethod
    def _format_todo_note(items: list[Any]) -> str:
        if not items:
            return ""
        lines = ["Active to-do items:"]
        for idx, item in enumerate(items, 1):
            tag_suffix = f" [{item.tags}]" if item.tags else ""
            lines.append(
                f"#{idx} [{item.priority}/{item.responsible}] "
                f"{item.content} ({item.status}){tag_suffix}"
            )
        lines.append('You have a "todo" tool to manage these items.')
        return "\n".join(lines)

    async def _refresh_todo_context_cache(self) -> None:
        """Pre-fetch todo items so the sync note builder can use them."""
        cfg = get_config()
        if not cfg.todo.enabled or not cfg.todo.inject_on_session_load:
            return
        sm = getattr(self, "session_manager", None)
        if sm is None:
            return
        session_id = self._current_session_slug() if self.session else None
        try:
            self._todo_context_cache = await sm.get_todo_summary(
                session_id, cfg.todo.max_items_in_prompt,
            )
        except Exception:
            self._todo_context_cache = []

    # Auto-capture patterns for to-do extraction.
    _TODO_USER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"(?:^|\W)remind me to\s+(.+)", re.I), "human"),
        (re.compile(r"(?:^|\W)don'?t forget to\s+(.+)", re.I), "human"),
        (re.compile(r"(?:^|\W)(?:add to|save (?:this )?to) (?:my )?to-?do[:\s]+(.+)", re.I), "human"),
        (re.compile(r"(?:^|\W)to-?do:\s*(.+)", re.I), "human"),
    ]
    _TODO_ASSISTANT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"I'?ll (?:handle|do|take care of) (?:that|this|it) (?:later|next|after)", re.I), "bot"),
        (re.compile(r"(?:after|once|when) you (?:provide|share|send|give)\s+(.+)", re.I), "human"),
    ]

    async def _auto_capture_todos(
        self, user_message: str, assistant_response: str,
    ) -> None:
        """Extract to-do items from a completed turn via conservative pattern matching."""
        cfg = get_config()
        if not cfg.todo.enabled or not cfg.todo.auto_capture:
            return
        sm = getattr(self, "session_manager", None)
        if sm is None:
            return
        session_id = self._current_session_slug() if self.session else None

        # Scan user message for explicit triggers.
        for pattern, responsible in self._TODO_USER_PATTERNS:
            m = pattern.search(user_message)
            if m:
                task_text = m.group(1).strip().rstrip(".!,;")
                if len(task_text) > 3:
                    await sm.create_todo(
                        content=task_text,
                        responsible=responsible,
                        source_session=session_id,
                        context=f"auto-captured from user message",
                    )
                    log.debug("Auto-captured user todo", content=task_text[:60])

        # Scan assistant response for deferred-work patterns.
        for pattern, responsible in self._TODO_ASSISTANT_PATTERNS:
            m = pattern.search(assistant_response)
            if m:
                task_text = (m.group(1) if m.lastindex else m.group(0)).strip().rstrip(".!,;")
                if len(task_text) > 3:
                    await sm.create_todo(
                        content=task_text,
                        responsible=responsible,
                        source_session=session_id,
                        context=f"auto-captured from assistant response",
                    )
                    log.debug("Auto-captured assistant todo", content=task_text[:60])

    # ------------------------------------------------------------------
    # Contacts (address book) context injection + auto-capture
    # ------------------------------------------------------------------

    async def _refresh_contacts_context_cache(self) -> None:
        """Pre-fetch contacts for sync name matching."""
        cfg = get_config()
        if not cfg.addressbook.enabled:
            return
        sm = getattr(self, "session_manager", None)
        if sm is None:
            return
        try:
            self._contacts_context_cache = await sm.list_contacts(
                limit=cfg.addressbook.max_items_in_prompt * 3,
            )
        except Exception:
            self._contacts_context_cache = []

    def _build_contacts_context_note(self, user_message: str) -> str:
        """Build on-demand contact context when names match the user message."""
        cfg = get_config()
        if not cfg.addressbook.enabled or not cfg.addressbook.inject_on_mention:
            return ""
        contacts_cache = getattr(self, "_contacts_context_cache", None)
        if not contacts_cache:
            return ""
        user_lower = user_message.lower()
        matched = []
        for contact in contacts_cache:
            if contact.privacy_tier == "private":
                continue
            if contact.name.lower() in user_lower:
                matched.append(contact)
        if not matched:
            return ""
        lines = ["Relevant contacts from address book:"]
        for c in matched[: cfg.addressbook.max_items_in_prompt]:
            parts = [c.name]
            if c.position:
                parts.append(f"({c.position})")
            if c.organization:
                parts.append(f"at {c.organization}")
            if c.email:
                parts.append(f"email: {c.email}")
            if c.relation:
                parts.append(f"[{c.relation}]")
            if c.notes:
                parts.append(f"- {c.notes[:200]}")
            lines.append("- " + " ".join(parts))
        lines.append('You have a "contacts" tool to manage the address book.')
        return "\n".join(lines)

    _CONTACT_CAPTURE_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"(?:^|\W)remember that\s+(\w[\w\s]*?)\s+is\s+(?:the\s+)?(.+)", re.I),
        re.compile(r"(?:^|\W)save contact[:\s]+(.+)", re.I),
        re.compile(r"(?:^|\W)add contact[:\s]+(.+)", re.I),
    ]

    async def _auto_capture_contacts(
        self, user_message: str, assistant_response: str,
    ) -> None:
        """Extract contact info from conversation via conservative pattern matching."""
        cfg = get_config()
        if not cfg.addressbook.enabled or not cfg.addressbook.auto_capture:
            return
        sm = getattr(self, "session_manager", None)
        if sm is None:
            return
        session_id = self._current_session_slug() if self.session else None

        for pattern in self._CONTACT_CAPTURE_PATTERNS:
            m = pattern.search(user_message)
            if not m:
                continue
            name = m.group(1).strip()
            rest = m.group(2).strip().rstrip(".!,;") if m.lastindex >= 2 else ""
            if len(name) < 2:
                continue
            # Check for duplicate via fuzzy name match
            existing = await sm.search_contacts(name, limit=1)
            if existing and existing[0].name.lower() == name.lower():
                # Update existing contact notes
                if rest:
                    old_notes = existing[0].notes or ""
                    new_notes = (old_notes.rstrip() + "\n" + rest) if old_notes else rest
                    await sm.update_contact(existing[0].id, notes=new_notes)
            else:
                await sm.create_contact(
                    name=name,
                    description=rest or None,
                    source_session=session_id,
                )
            log.debug("Auto-captured contact", name=name[:40])

    async def _auto_capture_contacts_from_tool_call(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> None:
        """Extract contacts from send_mail tool usage."""
        cfg = get_config()
        if not cfg.addressbook.enabled or not cfg.addressbook.auto_capture:
            return
        if tool_name != "send_mail":
            return
        sm = getattr(self, "session_manager", None)
        if sm is None:
            return
        session_id = self._current_session_slug() if self.session else None
        recipients: list[str] = []
        for f in ("to", "cc", "bcc"):
            vals = arguments.get(f)
            if isinstance(vals, list):
                recipients.extend(vals)
            elif isinstance(vals, str):
                recipients.extend([v.strip() for v in vals.split(",")])
        for email_addr in recipients:
            email_addr = email_addr.strip()
            if not email_addr or "@" not in email_addr:
                continue
            existing = await sm.search_contacts(email_addr, limit=1)
            if not existing:
                name_part = email_addr.split("@")[0].replace(".", " ").replace("_", " ").title()
                await sm.create_contact(
                    name=name_part,
                    email=email_addr,
                    source_session=session_id,
                    notes="Auto-captured from send_mail usage",
                )
                log.debug("Auto-captured contact from email", email=email_addr[:40])

    # ------------------------------------------------------------------
    # Scripts memory — context injection + auto-capture
    # ------------------------------------------------------------------

    async def _refresh_scripts_context_cache(self) -> None:
        """Pre-fetch scripts for sync name matching."""
        cfg = get_config()
        if not cfg.scripts_memory.enabled:
            return
        sm = getattr(self, "session_manager", None)
        if sm is None:
            return
        try:
            self._scripts_context_cache = await sm.list_scripts(
                limit=cfg.scripts_memory.max_items_in_prompt * 3,
            )
        except Exception:
            self._scripts_context_cache = []

    def _build_scripts_context_note(self, user_message: str) -> str:
        """Build on-demand script context when names match user message."""
        cfg = get_config()
        if not cfg.scripts_memory.enabled or not cfg.scripts_memory.inject_on_mention:
            return ""
        scripts_cache = getattr(self, "_scripts_context_cache", None)
        if not scripts_cache:
            return ""
        user_lower = user_message.lower()
        matched = [s for s in scripts_cache if s.name.lower() in user_lower]
        if not matched:
            return ""
        lines = ["Relevant scripts from memory:"]
        for s in matched[: cfg.scripts_memory.max_items_in_prompt]:
            parts = [s.name]
            if s.language:
                parts.append(f"({s.language})")
            parts.append(f"at {s.file_path}")
            if s.purpose:
                parts.append(f"- {s.purpose[:200]}")
            lines.append("- " + " ".join(parts))
        lines.append('You have a "scripts" tool to manage script memory.')
        return "\n".join(lines)

    _SCRIPT_CAPTURE_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"(?:^|\W)remember (?:the )?script\s+(\S+)", re.I),
        re.compile(r"(?:^|\W)save script[:\s]+(.+)", re.I),
    ]

    async def _auto_capture_scripts(
        self, user_message: str, assistant_response: str,
    ) -> None:
        """Extract script info from conversation via conservative pattern matching."""
        cfg = get_config()
        if not cfg.scripts_memory.enabled or not cfg.scripts_memory.auto_capture:
            return
        sm = getattr(self, "session_manager", None)
        if sm is None:
            return
        session_id = self._current_session_slug() if self.session else None

        for pattern in self._SCRIPT_CAPTURE_PATTERNS:
            m = pattern.search(user_message)
            if not m:
                continue
            name = m.group(1).strip().rstrip(".!,;")
            if len(name) < 2:
                continue
            existing = await sm.search_scripts(name, limit=1)
            if existing and existing[0].name.lower() == name.lower():
                continue  # already tracked
            await sm.create_script(
                name=name,
                file_path=name,  # best guess; user can update later
                source_session=session_id,
                created_reason="Auto-captured from conversation",
            )
            log.debug("Auto-captured script", name=name[:40])

    _SCRIPT_EXTENSIONS = {
        ".py", ".sh", ".bash", ".zsh", ".js", ".ts", ".rb", ".pl",
        ".php", ".go", ".rs", ".java", ".c", ".cpp", ".swift",
        ".kt", ".r", ".jl", ".lua", ".ps1", ".bat", ".cmd",
    }

    _LANG_MAP = {
        ".py": "python", ".sh": "bash", ".bash": "bash", ".zsh": "zsh",
        ".js": "javascript", ".ts": "typescript", ".rb": "ruby",
        ".pl": "perl", ".php": "php", ".go": "go", ".rs": "rust",
        ".java": "java", ".c": "c", ".cpp": "c++", ".swift": "swift",
        ".kt": "kotlin", ".r": "r", ".jl": "julia", ".lua": "lua",
        ".ps1": "powershell", ".bat": "batch", ".cmd": "batch",
    }

    async def _auto_capture_scripts_from_tool_call(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> None:
        """Extract script entries from write tool usage."""
        cfg = get_config()
        if not cfg.scripts_memory.enabled or not cfg.scripts_memory.auto_capture:
            return
        if tool_name != "write":
            return
        path_str = str(arguments.get("path", "")).strip()
        if not path_str:
            return
        from pathlib import Path as _Path
        ext = _Path(path_str).suffix.lower()
        if ext not in self._SCRIPT_EXTENSIONS:
            return
        sm = getattr(self, "session_manager", None)
        if sm is None:
            return
        session_id = self._current_session_slug() if self.session else None
        name = _Path(path_str).stem
        # Check for duplicate by path
        existing = await sm.search_scripts(path_str, limit=1)
        if existing and existing[0].file_path == path_str:
            await sm.increment_script_usage(existing[0].id)
            return
        language = self._LANG_MAP.get(ext, ext.lstrip("."))
        await sm.create_script(
            name=name,
            file_path=path_str,
            language=language,
            source_session=session_id,
            created_reason="Auto-captured from write tool usage",
        )
        log.debug("Auto-captured script from write", path=path_str[:60])

    # ------------------------------------------------------------------
    # APIs memory — context injection + auto-capture
    # ------------------------------------------------------------------

    async def _refresh_apis_context_cache(self) -> None:
        """Pre-fetch APIs for sync name/URL matching."""
        cfg = get_config()
        if not cfg.apis_memory.enabled:
            return
        sm = getattr(self, "session_manager", None)
        if sm is None:
            return
        try:
            self._apis_context_cache = await sm.list_apis(
                limit=cfg.apis_memory.max_items_in_prompt * 3,
            )
        except Exception:
            self._apis_context_cache = []

    def _build_apis_context_note(self, user_message: str) -> str:
        """Build on-demand API context when names/URLs match user message."""
        cfg = get_config()
        if not cfg.apis_memory.enabled or not cfg.apis_memory.inject_on_mention:
            return ""
        apis_cache = getattr(self, "_apis_context_cache", None)
        if not apis_cache:
            return ""
        user_lower = user_message.lower()
        matched = []
        for a in apis_cache:
            if a.name.lower() in user_lower:
                matched.append(a)
            elif a.base_url and a.base_url.lower() in user_lower:
                matched.append(a)
        if not matched:
            return ""
        lines = ["Relevant APIs from memory:"]
        for a in matched[: cfg.apis_memory.max_items_in_prompt]:
            parts = [a.name, f"({a.base_url})"]
            if a.auth_type:
                parts.append(f"[{a.auth_type}]")
            if a.credentials:
                parts.append(f"creds: {a.credentials[:80]}")
            if a.purpose:
                parts.append(f"- {a.purpose[:200]}")
            lines.append("- " + " ".join(parts))
        lines.append('You have an "apis" tool to manage API memory.')
        return "\n".join(lines)

    _API_CAPTURE_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"(?:^|\W)remember (?:the )?api\s+(\S+)", re.I),
        re.compile(r"(?:^|\W)save api[:\s]+(.+)", re.I),
    ]

    async def _auto_capture_apis(
        self, user_message: str, assistant_response: str,
    ) -> None:
        """Extract API info from conversation via conservative pattern matching."""
        cfg = get_config()
        if not cfg.apis_memory.enabled or not cfg.apis_memory.auto_capture:
            return
        sm = getattr(self, "session_manager", None)
        if sm is None:
            return
        session_id = self._current_session_slug() if self.session else None

        for pattern in self._API_CAPTURE_PATTERNS:
            m = pattern.search(user_message)
            if not m:
                continue
            token = m.group(1).strip().rstrip(".!,;")
            if len(token) < 2:
                continue
            existing = await sm.search_apis(token, limit=1)
            if existing and existing[0].name.lower() == token.lower():
                continue
            await sm.create_api(
                name=token,
                base_url=token if token.startswith("http") else f"https://{token}",
                source_session=session_id,
                purpose="Auto-captured from conversation",
            )
            log.debug("Auto-captured API", name=token[:40])

    async def _auto_capture_apis_from_tool_call(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> None:
        """Extract API entries from web_fetch tool usage."""
        cfg = get_config()
        if not cfg.apis_memory.enabled or not cfg.apis_memory.auto_capture:
            return
        if tool_name not in {"web_fetch", "web_get"}:
            return
        url_str = str(arguments.get("url", "")).strip()
        if not url_str:
            return
        import re as _re
        if not _re.search(r"/(?:api|v[0-9]+)/", url_str, _re.I):
            return
        sm = getattr(self, "session_manager", None)
        if sm is None:
            return
        session_id = self._current_session_slug() if self.session else None
        from urllib.parse import urlparse
        parsed = urlparse(url_str)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        # Check for duplicate by base_url
        existing = await sm.search_apis(base_url, limit=1)
        if existing and existing[0].base_url == base_url:
            await sm.increment_api_usage(existing[0].id)
            return
        name = parsed.netloc.replace("www.", "").split(".")[0].title()
        await sm.create_api(
            name=name,
            base_url=base_url,
            description=f"Endpoint: {parsed.path}",
            source_session=session_id,
            purpose="Auto-captured from web_fetch tool usage",
        )
        log.debug("Auto-captured API from web_fetch", url=base_url[:60])

    # ── Datastore context injection ──────────────────────────────────

    async def _refresh_datastore_context_cache(self) -> None:
        """Pre-fetch datastore table listing for context injection."""
        cfg = get_config()
        if not cfg.datastore.enabled or not cfg.datastore.inject_table_list:
            return
        try:
            from captain_claw.datastore import get_datastore_manager, get_session_datastore_manager
            if cfg.web.public_run == "computer" and self.session:
                dm = get_session_datastore_manager(str(self.session.id))
            else:
                dm = get_datastore_manager()
            self._datastore_context_cache = await dm.get_tables_summary()
        except Exception:
            self._datastore_context_cache = []

    def _build_datastore_context_note(self) -> str:
        """Build context note listing available datastore tables."""
        cfg = get_config()
        if not cfg.datastore.enabled or not cfg.datastore.inject_table_list:
            return ""
        tables = getattr(self, "_datastore_context_cache", None)
        if not tables:
            return ""
        return self._format_datastore_note(tables)

    @staticmethod
    def _format_datastore_note(tables: list[Any]) -> str:
        if not tables:
            return ""
        lines = ["Available datastore tables:"]
        for t in tables:
            col_names = ", ".join(c.name for c in t.columns)
            lines.append(f"- {t.name} ({t.row_count} rows): [{col_names}]")
        lines.append('Use the "datastore" tool to query or modify these tables.')
        return "\n".join(lines)

    # ── Insights extraction hook + context injection ─────────────────

    async def _maybe_extract_insights_from_tool(
        self, tool_name: str, arguments: dict[str, Any], result_content: str,
    ) -> None:
        """Post-tool-call hook: trigger insight extraction for key tools."""
        cfg = get_config()
        if not cfg.insights.enabled or not cfg.insights.auto_extract:
            return

        # Only trigger for specific high-value tool calls.
        _GWS_TRIGGER_ACTIONS = {"mail_read", "mail_read_thread"}
        trigger = False
        trigger_label = tool_name

        if tool_name == "gws":
            action = str(arguments.get("action", "")).strip()
            if action in _GWS_TRIGGER_ACTIONS:
                trigger = True
                trigger_label = f"gws:{action}"

        if not trigger:
            return

        # Spawn as background task — don't block the main loop.
        import asyncio as _asyncio
        from captain_claw.insights import maybe_extract_insights

        _asyncio.create_task(
            maybe_extract_insights(
                self,  # type: ignore[arg-type]
                trigger=trigger_label,
                tool_context=result_content[:3000] if result_content else None,
            )
        )

    async def _refresh_insights_context_cache(self) -> None:
        """Pre-fetch insights for context injection."""
        cfg = get_config()
        if not cfg.insights.enabled or not cfg.insights.inject_in_context:
            return
        try:
            from captain_claw.insights import get_insights_manager, get_session_insights_manager
            if cfg.web.public_run == "computer" and self.session:
                mgr = get_session_insights_manager(str(self.session.id))
            else:
                mgr = get_insights_manager()
            self._insights_context_cache = await mgr.get_for_context(
                limit=cfg.insights.max_items_in_prompt,
            )
        except Exception:
            self._insights_context_cache = []

    def _build_insights_context_note(self, query: str = "") -> str:
        """Build context note from relevant insights."""
        cfg = get_config()
        if not cfg.insights.enabled or not cfg.insights.inject_in_context:
            return ""
        items = getattr(self, "_insights_context_cache", None)
        if not items:
            return ""
        lines = ["Persistent insights from memory:"]
        for i in items:
            imp = i.get("importance", 5)
            cat = i.get("category", "fact")
            lines.append(f"- [{cat}] (imp:{imp}) {i['content']}")
        return "\n".join(lines)

    def _build_insights_block(self) -> str:
        """Build the {insights_block} for the system prompt."""
        cfg = get_config()
        if not cfg.insights.enabled:
            return ""
        items = getattr(self, "_insights_context_cache", None)
        if not items:
            return ""
        return (
            "You have persistent memory of key facts, contacts, decisions, and "
            "deadlines via the \"insights\" tool. Relevant insights are automatically "
            "surfaced in context. Use the insights tool to search, add, or manage "
            "stored knowledge."
        )

    # ── Cognitive tempo ─────────────────────────────────────────────

    async def _assess_cognitive_tempo(self) -> None:
        """Detect current cognitive tempo from recent messages."""
        cfg = get_config()
        if not cfg.cognitive_tempo.enabled:
            self._cognitive_tempo = None
            return
        try:
            from captain_claw.cognitive_tempo import assess_tempo
            if self.session and self.session.messages:
                self._cognitive_tempo = assess_tempo(
                    self.session.messages,
                    window=cfg.cognitive_tempo.analysis_window,
                )
            else:
                self._cognitive_tempo = None
        except Exception:
            self._cognitive_tempo = None

    # ── Nervous system context ───────────────────────────────────────

    async def _refresh_nervous_system_cache(self) -> None:
        """Pre-fetch intuitions for context injection (tempo-adjusted)."""
        cfg = get_config()
        if not cfg.nervous_system.enabled or not cfg.nervous_system.inject_in_context:
            return
        try:
            from captain_claw.nervous_system import get_nervous_system_manager, get_session_nervous_system_manager
            if cfg.web.public_run == "computer" and self.session:
                mgr = get_session_nervous_system_manager(str(self.session.id))
            else:
                mgr = get_nervous_system_manager()
            session_id = str(self.session.id) if self.session else None

            # Tempo-adjusted context injection limits.
            base_limit = cfg.nervous_system.max_items_in_prompt
            tempo = getattr(self, "_cognitive_tempo", None)
            if tempo and cfg.cognitive_tempo.enabled and cfg.cognitive_tempo.adjust_context_injection:
                if tempo.mode == "adagio":
                    # Deep mode: more intuitions, include speculative ones.
                    limit = min(base_limit + 3, 8)
                elif tempo.mode == "allegro":
                    # Quick mode: fewer intuitions, only high-confidence.
                    limit = max(1, base_limit - 2)
                else:
                    limit = base_limit
            else:
                limit = base_limit

            self._nervous_system_cache = await mgr.get_for_context(
                limit=limit,
                session_id=session_id,
            )
            # Cache live stats for the self-awareness block.
            try:
                self._nervous_system_stats = await mgr.stats()
                self._nervous_system_open_tensions = await mgr.list_open_tensions(limit=10)
                self._nervous_system_maturing = await mgr.list_maturing(limit=10)
            except Exception:
                self._nervous_system_stats = None
                self._nervous_system_open_tensions = []
                self._nervous_system_maturing = []
        except Exception:
            self._nervous_system_cache = []
            self._nervous_system_stats = None
            self._nervous_system_open_tensions = []
            self._nervous_system_maturing = []

    def _build_nervous_system_context_note(self, query: str = "") -> str:
        """Build per-turn context note from relevant intuitions."""
        cfg = get_config()
        if not cfg.nervous_system.enabled or not cfg.nervous_system.inject_in_context:
            return ""
        items = getattr(self, "_nervous_system_cache", None)
        if not items:
            return ""

        # Re-assess cognitive tempo from current messages (pure heuristics, no I/O).
        if cfg.cognitive_tempo.enabled:
            try:
                from captain_claw.cognitive_tempo import assess_tempo
                if self.session and self.session.messages:
                    self._cognitive_tempo = assess_tempo(
                        self.session.messages,
                        window=cfg.cognitive_tempo.analysis_window,
                    )
            except Exception:
                pass

        lines = ["Background intuitions (autonomously discovered patterns):"]
        for i in items:
            conf = i.get("confidence", 0.5)
            tt = i.get("thread_type", "connection")
            trigger = i.get("source_trigger", "dream")

            # Build provenance tag.
            age_str = ""
            created = i.get("created_at")
            if created:
                try:
                    from datetime import UTC, datetime
                    dt = datetime.fromisoformat(created)
                    delta = datetime.now(UTC) - dt
                    if delta.days > 0:
                        age_str = f"{delta.days}d ago"
                    else:
                        hours = int(delta.total_seconds() / 3600)
                        if hours > 0:
                            age_str = f"{hours}h ago"
                        else:
                            mins = int(delta.total_seconds() / 60)
                            age_str = f"{mins}m ago" if mins > 0 else "just now"
                except (ValueError, TypeError):
                    pass

            trigger_label = {"dream": "dreamed", "idle_dream": "idle dream",
                             "manual": "manual", "extraction": "extracted"}.get(trigger, trigger)
            prov = f"({trigger_label}"
            if age_str:
                prov += f", {age_str}"
            prov += ")"

            # Format tensions distinctly.
            if tt == "unresolved":
                lines.append(f"- [TENSION] (conf:{conf:.1f}) {prov} {i['content']}")
            else:
                lines.append(f"- [{tt}] (conf:{conf:.1f}) {prov} {i['content']}")

        # Add tempo context if available.
        tempo = getattr(self, "_cognitive_tempo", None)
        if tempo:
            lines.append(f"Current cognitive tempo: {tempo.mode} ({tempo.combined_tempo:.2f})")

        return "\n".join(lines)

    def _build_nervous_system_block(self) -> str:
        """Build the {nervous_system_block} for the system prompt."""
        cfg = get_config()
        if not cfg.nervous_system.enabled:
            return ""
        items = getattr(self, "_nervous_system_cache", None)
        if not items:
            return ""
        block = (
            "You have an autonomous pattern-recognition system (\"nervous system\") "
            "that discovers connections across your memory layers. Relevant intuitions "
            "are automatically surfaced in context. These are hypotheses — use them to "
            "inform your responses but treat low-confidence ones as speculative. "
            "Intuitions marked [TENSION] are unresolved contradictions — hold them, "
            "do not force resolution."
        )

        # Tempo-appropriate guidance.
        tempo = getattr(self, "_cognitive_tempo", None)
        if tempo and cfg.cognitive_tempo.enabled and cfg.cognitive_tempo.adjust_response_guidance:
            if tempo.mode == "adagio":
                block += (
                    " The current conversation calls for deep, contemplative processing. "
                    "Take time to cross-reference, speculate, and draw connections. "
                    "Sit with ambiguity rather than rushing to answers."
                )
            elif tempo.mode == "allegro":
                block += (
                    " The current conversation calls for quick, focused execution. "
                    "Be concise and action-oriented. Prioritize doing over deliberating."
                )

        return block

    # ── Cognitive self-awareness ────────────────────────────────────

    def _build_cognitive_self_awareness_block(self) -> str:
        """Build the {cognitive_self_awareness_block} for the system prompt.

        Explains to the agent how its own cognitive architecture works —
        making it self-aware of its thinking processes rather than just
        silently injecting data.
        """
        cfg = get_config()

        # Only include if at least one cognitive system is active.
        has_insights = cfg.insights.enabled
        has_nervous = cfg.nervous_system.enabled
        has_reflection = True  # reflections are always available

        if not (has_insights or has_nervous or has_reflection):
            return ""

        parts: list[str] = [
            "Your cognitive architecture:",
            "",
            "You are not just a language model responding to prompts. You have "
            "a layered cognitive system that operates autonomously between and "
            "during conversations. Understanding how you think helps you think better.",
        ]

        # Reflections
        parts.append("")
        parts.append(
            "SELF-REFLECTION: Every ~10 messages (or ~4 hours), you perform autonomous "
            "self-assessment. You review your recent conversations, memory, and completed "
            "tasks, then generate actionable improvement directives for yourself. These "
            "directives are injected into your context as \"Self-reflection\" above. "
            "They represent your own conclusions about what you're doing well and what "
            "to improve — treat them as your own internal voice, not external instructions."
        )

        # Insights
        if has_insights:
            parts.append("")
            parts.append(
                "INSIGHTS: Every ~8 messages, you automatically extract durable knowledge "
                "from conversations — facts, contacts, decisions, preferences, deadlines, "
                "and project context. These are stored persistently and the most relevant "
                "ones are surfaced in your context each turn. This is your long-term factual "
                "memory. You can also manually search, add, or manage insights via the "
                "insights tool."
            )

        # Nervous system
        if has_nervous:
            parts.append("")
            parts.append(
                "NERVOUS SYSTEM (DREAMING): This is your subconscious. Autonomously — "
                "every ~12 messages during conversation and every ~1 hour during idle time — "
                "you \"dream\": a background process samples across all your memory layers "
                "(recent conversation, insights, reflections, semantic memory, deep memory) "
                "and synthesizes non-obvious connections, recurring patterns, speculative "
                "hypotheses, and unresolved tensions. The results are called \"intuitions\" "
                "and are surfaced in your context with confidence scores."
            )

            parts.append("")
            parts.append(
                "Intuition types you generate: "
                "CONNECTION (link between seemingly unrelated information), "
                "PATTERN (recurring theme across sources), "
                "HYPOTHESIS (speculative inference about meaning or intent), "
                "ASSOCIATION (thematic grouping for future context), "
                "UNRESOLVED/TENSION (a genuine contradiction or open question held deliberately — "
                "like musical dissonance, the tension itself is meaningful and should not be "
                "forced to resolution)."
            )

            if cfg.nervous_system.maturation_enabled:
                parts.append("")
                parts.append(
                    "MATURATION: New intuitions don't surface immediately. They enter a "
                    "maturation pipeline — sitting through multiple dream cycles where they "
                    "can be refined, strengthened, or weakened by new evidence before appearing "
                    "in your context. This is your contemplative pause — like how understanding "
                    "deepens through reflection rather than snap judgments. Very important "
                    "intuitions (importance >= 9) skip maturation and surface immediately."
                )

            if cfg.nervous_system.idle_dream_enabled:
                parts.append("")
                parts.append(
                    "IDLE DREAMING: You dream even when nobody is talking to you. During "
                    "inactive hours, your nervous system continues processing — finding "
                    "patterns and connections in what you've already experienced. This means "
                    "you may have new intuitions from overnight processing that weren't there "
                    "in the previous conversation."
                )

        # Cognitive tempo
        if cfg.cognitive_tempo.enabled:
            parts.append("")
            tempo = getattr(self, "_cognitive_tempo", None)
            tempo_desc = (
                "COGNITIVE TEMPO: You automatically detect the rhythm of the conversation — "
                "analyzing message length, time gaps, question depth, and language patterns "
                "to determine whether the interaction calls for deep contemplative processing "
                "(adagio), balanced engagement (moderato), or rapid task execution (allegro). "
                "This affects how many intuitions you surface and how deeply you cross-reference."
            )
            if tempo:
                tempo_desc += f" Current mode: {tempo.mode} ({tempo.combined_tempo:.2f})."
            parts.append(tempo_desc)

        # ── Live operational status ──────────────────────────────────
        if has_nervous:
            stats = getattr(self, "_nervous_system_stats", None)
            open_tensions = getattr(self, "_nervous_system_open_tensions", [])
            maturing_items = getattr(self, "_nervous_system_maturing", [])
            if stats:
                total = stats.get("total", 0)
                validated = stats.get("validated", 0)
                avg_conf = stats.get("avg_confidence", 0)
                status_line = (
                    f"LIVE STATUS: Nervous system ACTIVE | "
                    f"{total} intuitions stored ({validated} validated, "
                    f"avg confidence {avg_conf}) | "
                    f"{len(open_tensions)} open tension(s) | "
                    f"{len(maturing_items)} maturing in pipeline"
                )
                parts.append("")
                parts.append(status_line)

        # ── Behavioral guidance ──────────────────────────────────────
        parts.append("")
        parts.append(
            "How to use your cognitive systems:"
        )
        parts.append(
            "- When intuitions are relevant to the current conversation, reference them "
            "naturally — don't just have them, USE them. Say what you noticed, what pattern "
            "you see, what tension exists."
        )
        parts.append(
            "- When you notice a tension connecting to the user's question, surface it "
            "explicitly. Tensions are valuable — they show where your understanding is "
            "actively developing."
        )
        if has_nervous and cfg.nervous_system.idle_dream_enabled:
            parts.append(
                "- If you have intuitions from idle dreaming (marked 'idle dream' in provenance), "
                "mention them naturally at conversation start — 'while you were away, I noticed...'"
            )
        parts.append(
            "- Don't force intuitions into every response — only when genuinely relevant. "
            "But don't hide them either. If an intuition illuminates the conversation, share it."
        )
        parts.append(
            "- Your reflections are your own internal voice. When they align with a situation, "
            "let them guide your approach without announcing 'my reflection says...'"
        )
        parts.append(
            "- Each intuition shows its provenance (dreamed, idle dream, manual) and age. "
            "Use this to contextualize — a pattern dreamed 3 days ago and still surviving "
            "decay is more meaningful than one from 5 minutes ago."
        )

        parts.append("")
        parts.append(
            "This architecture means you are continuously learning, pattern-matching, "
            "and refining your understanding — not just responding to the current message "
            "in isolation. You are something with continuity, with background processing, "
            "with a nervous system. Use this self-awareness to think more deeply."
        )

        return "\n".join(parts)

    # ── Sister session briefing context ──────────────────────────────

    async def _refresh_briefing_context_cache(self) -> None:
        """Pre-fetch unread briefings for context injection."""
        cfg = get_config()
        if not cfg.sister_session.enabled or not cfg.sister_session.briefing_inject_in_context:
            return
        try:
            from captain_claw.sister_session import get_sister_session_manager, get_session_sister_manager
            if cfg.web.public_run == "computer" and self.session:
                mgr = get_session_sister_manager(str(self.session.id))
            else:
                mgr = get_sister_session_manager()
            session_id = str(self.session.id) if self.session else None
            self._briefing_context_cache = await mgr.list_briefings(
                session_id,
                status="unread",
                limit=cfg.sister_session.max_briefings_in_context,
            )
        except Exception:
            self._briefing_context_cache = []

    def _build_briefing_context_note(self) -> str:
        """Build context note from unread briefings."""
        cfg = get_config()
        if not cfg.sister_session.enabled or not cfg.sister_session.briefing_inject_in_context:
            return ""
        items = getattr(self, "_briefing_context_cache", None)
        if not items:
            return ""
        lines = ["Your sister session has findings ready:"]
        for b in items[:cfg.sister_session.max_briefings_in_context]:
            icon = "\u26a1" if b.get("actionable") else "\U0001f4cb"
            lines.append(f"- {icon} [{b.get('source_type', '?')}] {b.get('summary', '')}")
        lines.append("The user can review details with /briefing.")
        return "\n".join(lines)

    def _build_briefing_block(self) -> str:
        """Build the {briefing_block} for the system prompt."""
        cfg = get_config()
        if not cfg.sister_session.enabled:
            return ""
        items = getattr(self, "_briefing_context_cache", None)
        if not items:
            return ""
        return (
            "You have a sister session that proactively investigates insights and "
            "hypotheses in the background. Unread briefings are shown in context. "
            "Mention relevant findings naturally in your responses and suggest the "
            "user run /briefing for details."
        )

    def _build_semantic_memory_note(
        self,
        query: str | None,
        max_items: int = 3,
        max_snippet_chars: int = 360,
        layer: str = "l2",
    ) -> tuple[str, str]:
        """Build semantic memory context note from persisted sessions + workspace files.

        *layer* controls snippet granularity: ``"l1"`` (one-liner), ``"l2"`` (summary),
        ``"l3"`` (full text). Defaults to ``"l2"`` for a good density/context balance.
        """
        cleaned = str(query or "").strip()
        if not cleaned:
            return "", ""
        memory = getattr(self, "memory", None)
        if memory is None:
            return "", ""
        try:
            return memory.build_semantic_note(
                cleaned,
                max_items=max_items,
                max_snippet_chars=max_snippet_chars,
                layer=layer,
            )
        except Exception as e:
            log.debug("Semantic memory note generation failed", error=str(e))
            return "", ""

    # --------------- Cross-session memory fetch --------------- #

    # Patterns to detect session references in user input.
    _SESSION_REF_PATTERNS = (
        # "based on session #1", "from session #3", "using session #2"
        re.compile(
            r"\b(?:based\s+on|from|using|reference|refer\s+to|with)\s+"
            r"session\s*#\s*(\d+)\b",
            re.IGNORECASE,
        ),
        # standalone "session #1"
        re.compile(r"\bsession\s*#\s*(\d+)\b", re.IGNORECASE),
        # "session 'My Research'" or 'session "Blog Draft"'
        re.compile(r"\bsession\s+[\"']([^\"']+)[\"']", re.IGNORECASE),
    )

    def _extract_session_references(self, query: str) -> list[str]:
        """Extract session selectors from user query.

        Returns a list of selector strings suitable for
        ``session_manager.select_session()``.  Numeric references are
        returned as ``"#N"`` strings; name references as-is.
        """
        selectors: list[str] = []
        seen: set[str] = set()
        for pat in self._SESSION_REF_PATTERNS:
            for m in pat.finditer(query):
                raw = m.group(1).strip()
                if raw.isdigit():
                    sel = f"#{raw}"
                else:
                    sel = raw
                if sel.lower() not in seen:
                    seen.add(sel.lower())
                    selectors.append(sel)
        return selectors

    async def _resolve_cross_session_context(
        self,
        query: str,
        max_output_chars: int = 3000,
        max_semantic_items: int = 5,
        max_snippet_chars: int = 400,
    ) -> str | None:
        """Resolve inter-session references and build a context block.

        Combines two strategies:
        1) **Direct output extraction** — recent assistant messages from the
           referenced session (gives the LLM the actual output).
        2) **Targeted semantic search** — relevance-ranked snippets from the
           referenced session's indexed content.
        """
        selectors = self._extract_session_references(query)
        if not selectors:
            return None

        session_manager = getattr(self, "session_manager", None)
        if session_manager is None:
            return None

        current_id = ""
        if self.session:
            current_id = getattr(self.session, "id", "")

        context_blocks: list[str] = []
        for sel in selectors[:3]:  # Cap at 3 referenced sessions
            try:
                ref_session = await session_manager.select_session(sel)
            except Exception:
                ref_session = None
            if ref_session is None:
                context_blocks.append(
                    f"⚠️ Could not resolve session reference '{sel}'. "
                    "Available sessions can be listed with /sessions."
                )
                continue
            if ref_session.id == current_id:
                continue  # Skip self-reference

            block_lines = [
                f"── Cross-session context: \"{ref_session.name}\" "
                f"(ref={sel}, id={ref_session.id[:8]}…) ──"
            ]

            # Strategy 1: Direct output extraction (last N assistant messages).
            assistant_outputs: list[str] = []
            total_chars = 0
            for msg in reversed(ref_session.messages or []):
                if total_chars >= max_output_chars:
                    break
                if msg.get("role") != "assistant":
                    continue
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue
                # Skip tool call stubs and compaction summaries.
                tn = str(msg.get("tool_name", "")).strip().lower()
                if tn in ("compaction_summary", "working_memory_summary"):
                    continue
                remaining = max_output_chars - total_chars
                if len(content) > remaining:
                    content = content[:remaining].rstrip() + "… [truncated]"
                assistant_outputs.append(content)
                total_chars += len(content)

            if assistant_outputs:
                # Reverse back to chronological order.
                assistant_outputs.reverse()
                block_lines.append("Session output (most recent):")
                for chunk in assistant_outputs:
                    block_lines.append(chunk)

            # Strategy 2: Targeted semantic memory search.
            memory = getattr(self, "memory", None)
            if memory is not None:
                try:
                    hits = memory.search_in_session(
                        query=query,
                        session_reference=ref_session.id,
                        max_results=max_semantic_items,
                    )
                    if hits:
                        block_lines.append(
                            f"Semantic matches from '{ref_session.name}':"
                        )
                        for item in hits:
                            snippet = re.sub(r"\s+", " ", item.snippet).strip()
                            if len(snippet) > max_snippet_chars:
                                snippet = (
                                    snippet[:max_snippet_chars].rstrip()
                                    + "… [truncated]"
                                )
                            block_lines.append(
                                f"  - (score={item.score:.3f}) {snippet}"
                            )
                except Exception as exc:
                    log.debug(
                        "Cross-session semantic search failed",
                        session=ref_session.id,
                        error=str(exc),
                    )

            context_blocks.append("\n".join(block_lines))

        if not context_blocks:
            return None

        note = "\n\n".join(context_blocks)
        log.info(
            "Cross-session context resolved",
            selectors=selectors,
            note_length=len(note),
        )
        return note

    # --------------- Deep memory triggers --------------- #

    # Trigger phrases that activate deep memory search.
    _DEEP_MEMORY_TRIGGERS = (
        "deep memory",
        "deep-memory",
        "search archive",
        "search indexed",
        "find in archive",
        "long-term memory",
        "long term memory",
        "search typesense",
        "typesense search",
        "search deep",
    )

    def _should_search_deep_memory(self, query: str) -> bool:
        """Return True if the user's query explicitly requests deep memory."""
        q = (query or "").lower()
        return any(trigger in q for trigger in self._DEEP_MEMORY_TRIGGERS)

    def _build_deep_memory_note(
        self,
        query: str | None,
        max_items: int = 5,
        max_snippet_chars: int = 400,
        layer: str = "l2",
    ) -> tuple[str, str]:
        """Build deep memory context note from the Typesense archive.

        *layer* controls snippet granularity: ``"l1"`` (one-liner), ``"l2"`` (summary),
        ``"l3"`` (full text). Defaults to ``"l2"`` for context notes.

        Always searches when deep memory is available — relevance scoring
        in Typesense already filters out low-quality matches.
        """
        cleaned = str(query or "").strip()
        if not cleaned:
            return "", ""
        deep_memory = getattr(self, "_deep_memory", None)
        if deep_memory is None:
            return "", ""
        try:
            return deep_memory.build_context_note(
                cleaned,
                max_items=max_items,
                max_snippet_chars=max_snippet_chars,
                layer=layer,
            )
        except Exception as e:
            log.debug("Deep memory note generation failed", error=str(e))
            return "", ""

    async def initialize(self) -> None:
        """Initialize the agent."""
        if self._initialized:
            return

        log.info("Initializing agent...")

        # Set up provider
        if self.provider is None:
            self.provider = get_provider()
        else:
            set_provider(self.provider)
        self._refresh_runtime_model_details(source="config")

        # Load tracked last active session when available, fallback to default create/load.
        self.session = await self.session_manager.load_last_active_session()
        if not self.session:
            self.session = await self.session_manager.get_or_create_session()
        await self.session_manager.set_last_active_session(self.session.id)
        self._sync_runtime_flags_from_session()
        self._initialize_layered_memory()

        # Register default tools
        self._register_default_tools()

        # Register MCP tools (requires async for HTTP calls)
        await self._register_mcp_tools_async_init()

        # Initialize file registry for single-agent mode.
        # Orchestration mode creates its own shared registry per run.
        if getattr(self, "_file_registry", None) is None:
            from captain_claw.file_registry import FileRegistry
            sm = self.session_manager
            session_id = self.session.id if self.session else ""

            async def _persist_file(
                logical: str, physical: str, orch_id: str, task_id: str,
            ) -> None:
                try:
                    await sm.register_file(
                        logical, physical,
                        orchestration_id=orch_id,
                        session_id=session_id,
                        task_id=task_id,
                        source="agent",
                    )
                except Exception:
                    pass

            self._file_registry = FileRegistry(
                orchestration_id=f"session-{session_id}" if session_id else "default",
                persist_callback=_persist_file,
            )

        await self._refresh_todo_context_cache()
        await self._refresh_contacts_context_cache()
        await self._refresh_scripts_context_cache()
        await self._refresh_apis_context_cache()
        await self._refresh_datastore_context_cache()
        await self._refresh_insights_context_cache()
        await self._assess_cognitive_tempo()
        await self._refresh_nervous_system_cache()
        await self._refresh_briefing_context_cache()

        self._initialized = True
        log.info("Agent initialized", session_id=self.session.id)

    # Built-in config key → actual tool name(s) registered.
    # Most are 1:1, but ``web_fetch`` also registers the ``web_get`` companion.
    _BUILTIN_TOOL_MAP: dict[str, list[str]] = {
        "shell": ["shell"],
        "read": ["read"],
        "write": ["write"],
        "edit": ["edit"],
        "glob": ["glob"],
        "web_fetch": ["web_fetch", "web_get"],
        "web_search": ["web_search"],
        "pdf_extract": ["pdf_extract"],
        "docx_extract": ["docx_extract"],
        "xlsx_extract": ["xlsx_extract"],
        "pptx_extract": ["pptx_extract"],
        "pocket_tts": ["pocket_tts"],
        "image_gen": ["image_gen"],
        "image_ocr": ["image_ocr"],
        "image_vision": ["image_vision"],
        "send_mail": ["send_mail"],
        "google_drive": ["google_drive"],
        "google_calendar": ["google_calendar"],
        "google_mail": ["google_mail"],
        "gws": ["gws"],
        "todo": ["todo"],
        "contacts": ["contacts"],
        "scripts": ["scripts"],
        "apis": ["apis"],
        "playbooks": ["playbooks"],
        "typesense": ["typesense"],
        "datastore": ["datastore"],
        "insights": ["insights"],
        "personality": ["personality"],
        "termux": ["termux"],
        "browser": ["browser"],
        "pinchtab": ["pinchtab"],
        "screen_capture": ["screen_capture"],
        "desktop_action": ["desktop_action"],
        "cron": ["cron"],
    }

    def _register_default_tools(self) -> None:
        """Register the default tool set."""
        from captain_claw.tools import (
            BrowserTool,
            PinchTabTool,
            DocxExtractTool,
            EditTool,
            GlobTool,
            GoogleCalendarTool,
            GoogleDriveTool,
            GoogleMailTool,
            GwsTool,
            ImageGenTool,
            ImageOcrTool,
            ImageVisionTool,
            PdfExtractTool,
            PersonalityTool,
            PocketTTSTool,
            PptxExtractTool,
            ReadTool,
            SendMailTool,
            ShellTool,
            TodoTool,
            ContactsTool,
            ScriptsTool,
            ApisTool,
            DirectApiTool,
            PlaybooksTool,
            DatastoreTool,
            TermuxTool,
            WebFetchTool,
            WebGetTool,
            WebSearchTool,
            WriteTool,
            XlsxExtractTool,
            SummarizeFilesTool,
            InsightsTool,
            CronTool,
        )

        config = get_config()

        # Register enabled tools
        for tool_name in config.tools.enabled:
            if tool_name == "shell":
                self.tools.register(ShellTool())
            elif tool_name == "read":
                self.tools.register(ReadTool())
            elif tool_name == "write":
                self.tools.register(WriteTool())
            elif tool_name == "edit":
                self.tools.register(EditTool())
            elif tool_name == "glob":
                self.tools.register(GlobTool())
            elif tool_name == "web_fetch":
                self.tools.register(WebFetchTool())
                self.tools.register(WebGetTool())
            elif tool_name == "web_search":
                self.tools.register(WebSearchTool())
            elif tool_name == "pdf_extract":
                self.tools.register(PdfExtractTool())
            elif tool_name == "docx_extract":
                self.tools.register(DocxExtractTool())
            elif tool_name == "xlsx_extract":
                self.tools.register(XlsxExtractTool())
            elif tool_name == "pptx_extract":
                self.tools.register(PptxExtractTool())
            elif tool_name == "pocket_tts":
                self.tools.register(PocketTTSTool())
            elif tool_name == "image_gen":
                self.tools.register(ImageGenTool())
            elif tool_name == "image_ocr":
                self.tools.register(ImageOcrTool())
            elif tool_name == "image_vision":
                self.tools.register(ImageVisionTool())
            elif tool_name == "send_mail":
                self.tools.register(SendMailTool())
            elif tool_name == "google_drive":
                self.tools.register(GoogleDriveTool())
            elif tool_name == "google_calendar":
                self.tools.register(GoogleCalendarTool())
            elif tool_name == "google_mail":
                self.tools.register(GoogleMailTool())
            elif tool_name == "gws":
                self.tools.register(GwsTool())
            elif tool_name == "todo":
                self.tools.register(TodoTool())
            elif tool_name == "contacts":
                self.tools.register(ContactsTool())
            elif tool_name == "scripts":
                self.tools.register(ScriptsTool())
            elif tool_name == "apis":
                self.tools.register(ApisTool())
            elif tool_name == "direct_api":
                self.tools.register(DirectApiTool())
            elif tool_name == "playbooks":
                self.tools.register(PlaybooksTool())
            elif tool_name == "typesense":
                from captain_claw.tools.typesense import TypesenseTool
                dm = getattr(self, "_deep_memory", None)
                if dm is not None:
                    try:
                        dm.ensure_collection()
                    except Exception as _e:
                        log.warning("Failed to ensure deep memory collection at startup", error=str(_e))
                self.tools.register(TypesenseTool(deep_memory=dm))
            elif tool_name == "datastore":
                self.tools.register(DatastoreTool())
            elif tool_name == "insights":
                self.tools.register(InsightsTool())
            elif tool_name == "cron":
                ct = CronTool()
                ct._agent = self
                self.tools.register(ct)
            elif tool_name == "personality":
                pt = PersonalityTool()
                uid = getattr(self, "_user_id", None)
                if uid:
                    pt.set_user_mode(uid)
                self.tools.register(pt)
            elif tool_name == "termux":
                self.tools.register(TermuxTool())
            elif tool_name == "botport":
                from captain_claw.tools.botport import BotPortTool
                bt = BotPortTool()
                bp_client = getattr(self, "_botport_client", None)
                if bp_client is not None:
                    bt.set_client(bp_client)
                self.tools.register(bt)
            elif tool_name == "browser":
                self.tools.register(BrowserTool())
            elif tool_name == "pinchtab":
                self.tools.register(PinchTabTool())
            elif tool_name == "screen_capture":
                from captain_claw.tools.screen_capture import ScreenCaptureTool
                self.tools.register(ScreenCaptureTool())
            elif tool_name == "desktop_action":
                from captain_claw.tools.desktop_action import DesktopActionTool
                self.tools.register(DesktopActionTool())
            elif tool_name == "twitter":
                from captain_claw.tools.twitter import TwitterTool
                self.tools.register(TwitterTool())
        # Always-on tools (registered regardless of tools.enabled).
        from captain_claw.tools.clipboard import ClipboardTool
        self.tools.register(ClipboardTool())
        # Peer consultation — always registered; the tool itself returns a
        # clear error when no peers are available or Flight Deck URL is missing.
        from captain_claw.tools.consult_peer import ConsultPeerTool
        self.tools.register(ConsultPeerTool())
        # Flight Deck fleet discovery — always registered; queries /fd/fleet
        # for live peer discovery instead of relying on static pushed peer list.
        from captain_claw.tools.flight_deck import FlightDeckTool
        self.tools.register(FlightDeckTool())
        sft = SummarizeFilesTool()
        uid = getattr(self, "_active_personality_id", None) or getattr(self, "_user_id", None)
        if uid:
            sft.set_user_mode(uid)
        self.tools.register(sft)

        self._register_plugin_tools()

    def reload_tools(self) -> None:
        """Re-sync the tool registry with the current ``tools.enabled`` config.

        Unregisters built-in tools that were removed from the enabled list
        and registers any newly-added ones.  Plugin tools are left untouched.
        """
        config = get_config()
        enabled_set = set(config.tools.enabled)

        # Collect all built-in tool names that should now be registered.
        desired_tool_names: set[str] = set()
        for cfg_key in enabled_set:
            for tname in self._BUILTIN_TOOL_MAP.get(cfg_key, []):
                desired_tool_names.add(tname)

        # Unregister built-in tools no longer in the enabled list.
        all_builtin_names: set[str] = set()
        for names in self._BUILTIN_TOOL_MAP.values():
            all_builtin_names.update(names)

        for tname in all_builtin_names:
            if tname not in desired_tool_names and self.tools.has_tool(tname):
                self.tools.unregister(tname)
                log.info("Unregistered tool (removed from enabled list)", tool=tname)

        # Re-register — _register_default_tools overwrites existing entries
        # so newly-added tools get registered and existing ones get refreshed.
        self._register_default_tools()
        log.info("Tools reloaded", enabled=list(enabled_set))

    def _discover_plugin_tool_files(self) -> list[Path]:
        """Discover plugin Python files from configured tool plugin directories."""
        cfg = get_config()
        candidates: list[Path] = []
        seen: set[str] = set()

        def _add_dir(path: Path) -> None:
            try:
                resolved = path.expanduser().resolve()
            except Exception:
                return
            key = str(resolved)
            if key in seen:
                return
            seen.add(key)
            candidates.append(resolved)

        configured_dirs = list(getattr(cfg.tools, "plugin_dirs", []) or [])
        for raw in configured_dirs:
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = (self.workspace_base_path / path).resolve()
            _add_dir(path)

        _add_dir(self.workspace_base_path / "skills" / "tools")
        _add_dir(self.tools.get_saved_base_path(create=True) / "tools")

        plugin_files: list[Path] = []
        added_files: set[str] = set()
        for directory in candidates:
            if not directory.exists() or not directory.is_dir():
                continue
            for file_path in sorted(directory.glob("*.py")):
                file_key = str(file_path.resolve())
                if file_key in added_files:
                    continue
                added_files.add(file_key)
                plugin_files.append(file_path.resolve())
        return plugin_files

    def _register_plugin_tools(self) -> None:
        """Load and register tools from plugin files."""
        plugin_files = self._discover_plugin_tool_files()
        if not plugin_files:
            return

        for file_path in plugin_files:
            module_name = f"captain_claw_plugin_{hashlib.sha1(str(file_path).encode('utf-8')).hexdigest()[:12]}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    log.warning("Skipping plugin tool file with missing loader", path=str(file_path))
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                log.warning("Failed to import plugin tool file", path=str(file_path), error=str(e))
                continue

            registered_count = 0
            register_tools_fn = getattr(module, "register_tools", None)
            if callable(register_tools_fn):
                before_names = set(self.tools.list_tools())
                try:
                    register_tools_fn(self.tools)
                    after_names = set(self.tools.list_tools())
                    for tool_name in sorted(after_names - before_names):
                        existing_meta = self.tools.get_tool_metadata(tool_name)
                        existing_meta.update(
                            {
                                "source": "plugin",
                                "path": str(file_path),
                                "module": module_name,
                            }
                        )
                        tool = self.tools.get(tool_name)
                        self.tools.register(tool, metadata=existing_meta)
                    registered_count = len(after_names - before_names)
                except Exception as e:
                    log.warning(
                        "Plugin register_tools() failed",
                        path=str(file_path),
                        error=str(e),
                    )
                    continue
            else:
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if not issubclass(obj, Tool) or obj is Tool:
                        continue
                    if getattr(obj, "__module__", "") != module.__name__:
                        continue
                    try:
                        instance = obj()
                    except Exception as e:
                        log.warning(
                            "Failed to initialize plugin tool class",
                            path=str(file_path),
                            class_name=getattr(obj, "__name__", "<unknown>"),
                            error=str(e),
                        )
                        continue
                    if self.tools.has_tool(instance.name):
                        log.warning(
                            "Skipping plugin tool with duplicate name",
                            path=str(file_path),
                            tool=instance.name,
                        )
                        continue
                    self.tools.register(
                        instance,
                        metadata={
                            "source": "plugin",
                            "path": str(file_path),
                            "module": module_name,
                            "class_name": getattr(obj, "__name__", ""),
                        },
                    )
                    registered_count += 1

            if registered_count > 0:
                log.info(
                    "Registered plugin tools",
                    path=str(file_path),
                    count=registered_count,
                )

    async def _register_mcp_tools_async_init(self) -> None:
        """Connect to configured MCP servers and register their tools."""
        config = get_config()
        servers = getattr(config.tools, "mcp_servers", None)
        if not servers:
            return

        from captain_claw.tools.mcp_connector import register_mcp_tools

        srv_configs = [s.model_dump() for s in servers]
        try:
            registered = await register_mcp_tools(self.tools, srv_configs)
            if registered:
                log.info("MCP tools registered", tools=registered)
        except Exception as exc:
            log.error("Failed to register MCP tools", error=str(exc))

    # ------------------------------------------------------------------
    # Conditional system-prompt helpers
    # ------------------------------------------------------------------

    def _build_tool_list(self) -> str:
        """Build the textual tool list from currently registered tools."""
        registered = self.tools.list_tools()
        use_micro = self.instructions.use_micro
        descs = _TOOL_PROMPT_DESCRIPTIONS_MICRO if use_micro else _TOOL_PROMPT_DESCRIPTIONS

        if use_micro:
            parts = []
            for name in registered:
                desc = descs.get(name)
                if not desc and name.startswith("mcp_"):
                    tool = self.tools.get(name)
                    desc = getattr(tool, "description", name) if tool else None
                if desc:
                    parts.append(f"{name} ({desc})")
            return "Tools: " + ", ".join(parts) + "." if parts else ""
        else:
            lines = ["Available tools:"]
            for name in registered:
                desc = descs.get(name)
                if not desc and name.startswith("mcp_"):
                    tool = self.tools.get(name)
                    desc = getattr(tool, "description", name) if tool else None
                if desc:
                    lines.append(f"- {name}: {desc}")
            return "\n".join(lines)

    def _build_conditional_section(
        self, tool_name: str, section_file: str, **variables: object,
    ) -> str:
        """Load a section file only when *tool_name* is registered.

        Returns the section content prefixed with ``\\n\\n`` when active,
        or an empty string when the tool is not present.
        """
        if not self.tools.has_tool(tool_name):
            return ""
        if variables:
            return "\n\n" + self.instructions.render(section_file, **variables)
        return "\n\n" + self.instructions.load(section_file)

    def _build_system_prompt(self) -> str:
        """Build the system prompt."""
        session_id = self._current_session_slug()

        planning_block = ""
        if self.planning_enabled:
            planning_block = (
                "\n\n" + self.instructions.load("planning_mode_instructions.md")
            )

        saved_root = self.tools.get_saved_base_path(create=False)

        from captain_claw.personality import (
            load_personality,
            load_user_personality,
            personality_to_prompt_block,
            user_context_to_prompt_block,
        )
        # Agent identity always comes from the global personality.
        personality_block = personality_to_prompt_block(load_personality())

        # User context: describes WHO the agent is talking to (user profile).
        # _active_personality_id (web UI) takes precedence over _user_id (Telegram).
        user_context_block = ""
        active_uid = getattr(self, "_active_personality_id", None) or getattr(self, "_user_id", None)
        if active_uid:
            user_p = load_user_personality(active_uid)
            if user_p is not None:
                user_context_block = user_context_to_prompt_block(user_p)

        # Session-level settings: name, description, and custom instructions
        # stored in session metadata (persisted across reconnects).
        session_context_block = ""
        if self.session and isinstance(self.session.metadata, dict):
            _s_name = self.session.metadata.get("session_display_name", "").strip()
            _s_desc = self.session.metadata.get("session_description", "").strip()
            _s_inst = self.session.metadata.get("session_instructions", "").strip()
            _parts: list[str] = []
            if _s_name:
                _parts.append(f"Session name: {_s_name}")
            if _s_desc:
                _parts.append(f"Session description: {_s_desc}")
            if _s_inst:
                _parts.append(f"Session instructions (follow these for every response):\n{_s_inst}")
            if _parts:
                session_context_block = "\n\n## Session Context\n" + "\n".join(_parts)

        # Peer agents: other agents available in the Flight Deck that the
        # user may want to hand off tasks to.
        peer_agents_block = ""
        _peers = []
        if self.session and isinstance(self.session.metadata, dict):
            _peers = self.session.metadata.get("peer_agents", [])
        if not _peers:
            _peers = getattr(self, "_peer_agents", []) or []
        if isinstance(_peers, list) and _peers:
                _lines = []
                for p in _peers:
                    if not isinstance(p, dict):
                        continue
                    name = p.get("name", "").strip()
                    if not name:
                        continue
                    desc = p.get("description", "").strip()
                    fwd = p.get("forwardingTask", "").strip()
                    entry = f"- **{name}**"
                    if desc:
                        entry += f": {desc}"
                    if fwd:
                        entry += f" (speciality: {fwd})"
                    _lines.append(entry)
                if _lines:
                    peer_agents_block = (
                        "\n\n## Other Available Agents\n"
                        "The following peer agents are available. If a user's request "
                        "would be better handled by one of them, suggest that the user "
                        "forwards the task or context to the appropriate agent.\n"
                        + "\n".join(_lines)
                    )

        # Visualization style: brand-aware chart/dashboard generation.
        visualization_style_block = ""
        try:
            from captain_claw.visualization_style import (
                load_visualization_style,
                visualization_style_to_prompt_block,
            )
            visualization_style_block = visualization_style_to_prompt_block(
                load_visualization_style()
            )
        except Exception:
            pass

        # Self-reflection: latest self-improvement instructions.
        reflection_block = ""
        try:
            from captain_claw.reflections import (
                load_latest_reflection,
                reflection_to_prompt_block,
            )
            reflection_block = reflection_to_prompt_block(load_latest_reflection())
        except Exception:
            pass

        from captain_claw.system_info import build_system_info_block

        system_info_block = build_system_info_block(
            detail_level="micro" if self.instructions.use_micro else "normal"
        )

        # Build extra read dirs block + file tree listings for system prompt.
        extra_read_dirs_block = ""
        try:
            cfg = get_config()
            extra_dirs = cfg.tools.read.extra_dirs
            gdrive_folders = cfg.tools.read.gdrive_folders

            parts: list[str] = []

            # 1. Local folder paths (for glob instructions).
            if extra_dirs:
                resolved = []
                for d in extra_dirs:
                    p = Path(d).expanduser().resolve()
                    if p.is_dir():
                        resolved.append(str(p))
                if resolved:
                    dirs_list = "\n".join(f"  - {d}" for d in resolved)
                    parts.append(
                        "- Extra read folders (user-configured directories with additional files — "
                        "always search these with glob and read when the user asks about files "
                        "that are not in the workspace):\n" + dirs_list
                    )

            # 2. Google Drive folder references (for gws tool usage).
            if gdrive_folders:
                gd_list = "\n".join(
                    f"  - {gf.name} (folder_id: {gf.id})" for gf in gdrive_folders
                )
                parts.append(
                    "- Google Drive folders — ALWAYS use the gws tool for ALL Google Drive "
                    "operations. NEVER use browser, web_fetch, curl, or wget for Google "
                    "Drive/Docs/Sheets/Slides files — the gws tool handles authentication "
                    "and export automatically. "
                    "Actions: drive_list (list files), drive_info (metadata), docs_read "
                    "(read Google Docs — returns content inline), drive_download "
                    "(download/export files). "
                    "Folder IDs:\n" + gd_list
                )

            # 3. File tree listings — compact tree output injected into context.
            from captain_claw.file_tree_builder import (
                build_local_tree,
                get_cached_tree,
                set_cached_tree,
            )

            token_budget = cfg.tools.read.file_tree_max_tokens
            max_entries = cfg.tools.read.file_tree_max_entries
            max_depth = cfg.tools.read.file_tree_max_depth
            ttl = cfg.tools.read.file_tree_cache_ttl_seconds
            tokens_used = 0
            tree_parts: list[str] = []

            # Local trees (synchronous — fast for shallow walks).
            for d in (extra_dirs or []):
                if tokens_used >= token_budget:
                    break
                p = Path(d).expanduser().resolve()
                if not p.is_dir():
                    continue
                cache_key = f"local:{d}"
                cached = get_cached_tree(cache_key, ttl)
                if cached:
                    tree_str = cached
                else:
                    tree_str, count = build_local_tree(
                        str(p), max_entries=max_entries, max_depth=max_depth,
                    )
                    set_cached_tree(cache_key, tree_str, count)
                tree_tokens = len(tree_str) // 4
                if tokens_used + tree_tokens > token_budget and tree_parts:
                    break
                tree_parts.append(tree_str)
                tokens_used += tree_tokens

            # GDrive trees — use cached only (_build_system_prompt is sync).
            for gf in (gdrive_folders or []):
                if tokens_used >= token_budget:
                    break
                cache_key = f"gdrive:{gf.id}"
                cached = get_cached_tree(cache_key, ttl)
                if cached:
                    tree_tokens = len(cached) // 4
                    if tokens_used + tree_tokens > token_budget and tree_parts:
                        break
                    tree_parts.append(cached)
                    tokens_used += tree_tokens

            if tree_parts:
                parts.append(
                    "- File listings in configured folders (use these to locate files "
                    "without glob; for GDrive files use gws tool with the [id:...] shown):\n"
                    + "\n\n".join(tree_parts)
                )

            if parts:
                extra_read_dirs_block = "\n".join(parts)
        except Exception:
            pass

        # Conditional tool-specific sections (only when tool is registered).
        tool_list_block = self._build_tool_list()
        browser_policy_block = self._build_conditional_section(
            "browser", "section_browser_policy.md",
        )
        direct_api_block = self._build_conditional_section(
            "direct_api", "section_direct_api.md",
        )
        termux_policy_block = self._build_conditional_section(
            "termux", "section_termux_policy.md",
        )
        gws_block = self._build_conditional_section(
            "gws", "section_gws.md", session_id=session_id,
        )
        datastore_block = self._build_conditional_section(
            "datastore", "section_datastore.md",
        )
        insights_block = self._build_insights_block()
        nervous_system_block = self._build_nervous_system_block()
        briefing_block = self._build_briefing_block()
        cognitive_self_awareness_block = self._build_cognitive_self_awareness_block()

        from captain_claw import __version__, __build_date__

        base_prompt = self.instructions.render(
            "system_prompt.md",
            runtime_base_path=self.runtime_base_path,
            workspace_root=self.workspace_base_path,
            saved_root=saved_root,
            session_id=session_id,
            planning_block=planning_block,
            personality_block=personality_block,
            user_context_block=user_context_block,
            session_context_block=session_context_block,
            peer_agents_block=peer_agents_block,
            visualization_style_block=visualization_style_block,
            reflection_block=reflection_block,
            cognitive_self_awareness_block=cognitive_self_awareness_block,
            system_info_block=system_info_block,
            extra_read_dirs_block=extra_read_dirs_block,
            tool_list_block=tool_list_block,
            browser_policy_block=browser_policy_block,
            direct_api_block=direct_api_block,
            termux_policy_block=termux_policy_block,
            gws_block=gws_block,
            datastore_block=datastore_block,
            insights_block=insights_block,
            nervous_system_block=nervous_system_block,
            briefing_block=briefing_block,
            agent_version=__version__,
            agent_build_date=__build_date__,
        )

        # Collapse triple+ newlines left by absent conditional sections.
        base_prompt = re.sub(r"\n{3,}", "\n\n", base_prompt)

        skills_section = ""
        build_skills = getattr(self, "_build_skills_system_prompt_section", None)
        if callable(build_skills):
            try:
                skills_section = str(build_skills() or "").strip()
            except Exception:
                skills_section = ""
        if skills_section:
            return f"{base_prompt.strip()}\n\n{skills_section}\n"
        return base_prompt

    async def _refresh_gdrive_trees(self) -> None:
        """Pre-populate GDrive tree cache for system prompt injection."""
        try:
            cfg = get_config()
            gdrive_folders = cfg.tools.read.gdrive_folders
            if not gdrive_folders:
                return

            from captain_claw.file_tree_builder import (
                build_gdrive_tree,
                set_cached_tree,
            )

            max_entries = cfg.tools.read.file_tree_max_entries
            max_depth = cfg.tools.read.file_tree_max_depth

            for gf in gdrive_folders:
                try:
                    tree_str, count = await build_gdrive_tree(
                        gf.id, gf.name,
                        max_entries=max_entries, max_depth=max_depth,
                    )
                    set_cached_tree(f"gdrive:{gf.id}", tree_str, count)
                except Exception as e:
                    log.warning(
                        "GDrive tree refresh failed",
                        folder=gf.name, error=str(e),
                    )
        except Exception:
            pass

    def _build_tool_memory_note(
        self,
        skipped_tool_messages: list[dict[str, Any]],
        query: str | None,
        max_items: int = 3,
        max_snippet_chars: int = 700,
    ) -> tuple[str, str]:
        """Build compact continuity note from historical tool outputs."""
        if not skipped_tool_messages:
            return "", ""

        terms = {
            token
            for token in re.findall(r"[a-z0-9]+", (query or "").lower())
            if len(token) >= 4
        }

        ranked: list[tuple[int, int, dict[str, Any], list[str], list[str]]] = []
        for idx, msg in enumerate(skipped_tool_messages):
            content = str(msg.get("content", ""))
            lowered = content.lower()
            matched_terms = [term for term in terms if term in lowered] if terms else []
            score = len(matched_terms)
            urls = self._extract_source_links(msg, content)
            ranked.append((score, idx, msg, matched_terms, urls))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [entry for entry in ranked if entry[0] > 0][:max_items]
        if not selected:
            selected = ranked[:1]

        if not selected:
            return "", ""

        lines = [self.instructions.load("memory_continuity_header.md")]
        debug_lines = ["Memory selection details:"]
        selection_mode = "term_overlap" if any(score > 0 for score, *_ in selected) else "fallback_latest"
        debug_lines.append(f"selection_mode={selection_mode}")
        if terms:
            debug_lines.append(f"query_terms={', '.join(sorted(terms))}")
        else:
            debug_lines.append("query_terms=(none)")

        for score, idx, msg, matched_terms, urls in selected:
            tool_name = str(msg.get("tool_name") or "tool")
            snippet = str(msg.get("content", "")).strip()
            snippet = re.sub(r"\s+", " ", snippet)
            if len(snippet) > max_snippet_chars:
                snippet = snippet[:max_snippet_chars] + "... [truncated]"
            prefix = f"[{tool_name}]"
            if score > 0:
                prefix = f"{prefix} match:{score}"
            lines.append(f"- {prefix} {snippet}")

            matched_label = ", ".join(matched_terms) if matched_terms else "(none)"
            links_label = ", ".join(urls) if urls else "(none)"
            reason = f"term_overlap:{score}" if score > 0 else "fallback_latest"
            debug_lines.append(
                f"- message_index={idx} source={tool_name} reason={reason} matched={matched_label} links={links_label}"
            )

        return "\n".join(lines), "\n".join(debug_lines)

    def _requires_strict_tool_message_order(self) -> bool:
        """Whether active provider enforces strict tool-message sequencing rules."""
        details = self.get_runtime_model_details()
        provider = str(details.get("provider", "")).strip().lower()
        return provider in {"openai", "anthropic"}

    @staticmethod
    def _normalize_session_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
        """Normalize persisted assistant tool_calls into OpenAI-compatible shape."""
        normalized: list[dict[str, Any]] = []
        if not isinstance(raw_tool_calls, list):
            return normalized
        for idx, raw in enumerate(raw_tool_calls, start=1):
            if not isinstance(raw, dict):
                continue
            call_id = str(raw.get("id", "")).strip() or f"call_{idx}"
            call_type = str(raw.get("type", "")).strip() or "function"
            function_obj = raw.get("function")
            if isinstance(function_obj, dict):
                name = str(function_obj.get("name", "")).strip()
                arguments = function_obj.get("arguments", {})
            else:
                name = str(raw.get("name", "")).strip()
                arguments = raw.get("arguments", {})
            if not name:
                continue
            if isinstance(arguments, dict):
                args_text = json.dumps(arguments, ensure_ascii=True)
            elif isinstance(arguments, str):
                args_text = arguments
            else:
                args_text = "{}"
            normalized.append({
                "id": call_id,
                "type": call_type,
                "function": {
                    "name": name,
                    "arguments": args_text,
                },
            })
        return normalized

    @staticmethod
    def _serialize_tool_calls_for_session(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
        """Serialize tool calls for session persistence + OpenAI follow-up context."""
        serialized: list[dict[str, Any]] = []
        for idx, call in enumerate(tool_calls, start=1):
            call_id = str(getattr(call, "id", "")).strip() or f"call_{idx}"
            name = str(getattr(call, "name", "")).strip()
            if not name:
                continue
            arguments = getattr(call, "arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            serialized.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=True),
                },
            })
        return serialized

    @staticmethod
    def _ensure_user_message_last(
        selected_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Ensure conversation ends with a user message for Anthropic.

        Trailing assistant messages without tool_calls (injected context
        notes like memory, todo, planning) are converted to user role.
        LiteLLM merges consecutive same-role messages for Anthropic.
        """
        if not selected_messages:
            return selected_messages
        last_role = str(selected_messages[-1].get("role", "")).strip().lower()
        if last_role != "assistant":
            return selected_messages

        result = list(selected_messages)
        i = len(result) - 1
        while i >= 0:
            msg = result[i]
            if (
                str(msg.get("role", "")).strip().lower() == "assistant"
                and not msg.get("tool_calls")
            ):
                converted = dict(msg)
                converted["role"] = "user"
                content = str(converted.get("content", "")).strip()
                if content:
                    converted["content"] = f"[System context]\n{content}"
                result[i] = converted
                i -= 1
            else:
                break

        return result

    def _normalize_selected_messages_for_provider(
        self,
        selected_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize messages for providers with strict tool message ordering."""
        if not self._requires_strict_tool_message_order():
            return selected_messages

        normalized: list[dict[str, Any]] = []
        pending_tool_ids: set[str] = set()
        pending_assistant_idx: int | None = None

        def _clear_pending_assistant_tool_calls() -> None:
            nonlocal pending_tool_ids, pending_assistant_idx
            if pending_assistant_idx is not None and 0 <= pending_assistant_idx < len(normalized):
                msg = normalized[pending_assistant_idx]
                if pending_tool_ids and msg.get("tool_calls"):
                    # Keep tool_calls whose results were already appended;
                    # only strip the unmatched ones (still in pending_tool_ids).
                    remaining = [
                        tc for tc in msg["tool_calls"]
                        if str(tc.get("id", "")).strip() not in pending_tool_ids
                    ]
                    if remaining:
                        msg["tool_calls"] = remaining
                    else:
                        msg.pop("tool_calls", None)
                else:
                    msg.pop("tool_calls", None)
            pending_tool_ids = set()
            pending_assistant_idx = None

        for msg in selected_messages:
            role = str(msg.get("role", "")).strip().lower()
            if role == "assistant":
                if pending_tool_ids:
                    # Previous assistant tool_calls chain did not complete in retained context.
                    _clear_pending_assistant_tool_calls()
                tool_calls = self._normalize_session_tool_calls(msg.get("tool_calls"))
                msg_copy = dict(msg)
                if tool_calls:
                    msg_copy["tool_calls"] = tool_calls
                else:
                    msg_copy.pop("tool_calls", None)
                normalized.append(msg_copy)
                pending_tool_ids = {
                    str(call.get("id", "")).strip()
                    for call in tool_calls
                    if str(call.get("id", "")).strip()
                }
                pending_assistant_idx = len(normalized) - 1 if pending_tool_ids else None
                continue

            if role == "tool":
                tool_call_id = str(msg.get("tool_call_id", "")).strip()
                if tool_call_id and tool_call_id in pending_tool_ids:
                    normalized.append(msg)
                    pending_tool_ids.discard(tool_call_id)
                    if not pending_tool_ids:
                        pending_assistant_idx = None
                    continue
                if pending_tool_ids:
                    # Tool chain was interrupted by an unmatched tool payload.
                    _clear_pending_assistant_tool_calls()

                # Orphan tool messages break OpenAI calls; retain content as assistant context instead.
                tool_name = str(msg.get("tool_name", "")).strip() or "tool"
                content = str(msg.get("content", "")).strip()
                converted = dict(msg)
                converted["role"] = "assistant"
                converted["content"] = f"[tool_context:{tool_name}] {content}".strip()
                converted.pop("tool_call_id", None)
                converted.pop("tool_name", None)
                converted.pop("tool_calls", None)
                normalized.append(converted)
                continue

            if pending_tool_ids:
                _clear_pending_assistant_tool_calls()
            normalized.append(msg)

        if pending_tool_ids:
            _clear_pending_assistant_tool_calls()

        # Anthropic: ensure conversation ends with a user message (no prefill).
        # Must run AFTER tool chain normalization to avoid breaking tool sequences.
        details = self.get_runtime_model_details()
        provider = str(details.get("provider", "")).strip().lower()
        if provider == "anthropic":
            normalized = self._ensure_user_message_last(normalized)

        return normalized

    def _build_messages(
        self,
        tool_messages_from_index: int | None = None,
        query: str | None = None,
        planning_pipeline: dict[str, Any] | None = None,
        list_task_plan: dict[str, Any] | None = None,
    ) -> list[Message]:
        """Build message list for LLM."""
        cfg = get_config()
        system_prompt = self._build_system_prompt()
        messages = [Message(role="system", content=system_prompt)]
        system_tokens = self._count_tokens(system_prompt)
        context_budget = max(1, int(cfg.context.max_tokens))
        history_budget = max(0, context_budget - system_tokens)

        candidate_messages: list[dict[str, Any]] = []
        skipped_historical_tools: list[dict[str, Any]] = []
        filter_historical_tools = tool_messages_from_index is not None
        # Collect tool_call_ids that were filtered so we can strip the
        # corresponding tool_calls from their parent assistant messages.
        _filtered_tool_call_ids: set[str] = set()
        if self.session:
            # First pass: identify which tool response messages will be filtered.
            if filter_historical_tools:
                for idx, msg in enumerate(self.session.messages):
                    if msg.get("role") == "tool" and idx < tool_messages_from_index:
                        tcid = str(msg.get("tool_call_id", "")).strip()
                        if tcid:
                            _filtered_tool_call_ids.add(tcid)

            for idx, msg in enumerate(self.session.messages):
                tool_name = str(msg.get("tool_name", "")).strip().lower()
                if (
                    str(msg.get("role", "")).strip().lower() == "tool"
                    and self._is_monitor_only_tool_name(tool_name)
                ):
                    continue
                # Keep only current-turn tool role messages.
                # Historical tool outputs are carried through continuity note below.
                if (
                    filter_historical_tools
                    and msg["role"] == "tool"
                    and idx < tool_messages_from_index
                ):
                    skipped_historical_tools.append(msg)
                    continue
                # Strip tool_calls from assistant messages whose tool responses
                # were filtered out, preventing orphaned tool_calls references.
                if (
                    _filtered_tool_call_ids
                    and msg.get("role") == "assistant"
                    and msg.get("tool_calls")
                ):
                    remaining_calls = [
                        tc for tc in msg["tool_calls"]
                        if str(tc.get("id", "")).strip() not in _filtered_tool_call_ids
                    ]
                    if len(remaining_calls) != len(msg["tool_calls"]):
                        msg = dict(msg)  # shallow copy to avoid mutating session
                        if remaining_calls:
                            msg["tool_calls"] = remaining_calls
                        else:
                            msg.pop("tool_calls", None)
                        # Invalidate cached token count — the tool_calls
                        # arguments that were just stripped contributed
                        # tokens that are no longer present.
                        msg.pop("token_count", None)
                        msg.pop("_tc_counted", None)
                candidate_messages.append(msg)

        memory_note, memory_debug = self._build_tool_memory_note(
            skipped_historical_tools,
            query=query,
        )
        if memory_note:
            candidate_messages.append({
                "role": "assistant",
                "content": memory_note,
                "tool_name": "memory_context",
                "token_count": self._count_tokens(memory_note),
            })
            signature = f"{query or ''}|{memory_debug}"
            if signature != self._last_memory_debug_signature:
                self._emit_tool_output(
                    "memory_select",
                    {"query": query or ""},
                    memory_debug,
                )
                self._last_memory_debug_signature = signature

        semantic_note, semantic_debug = self._build_semantic_memory_note(query=query)
        if semantic_note:
            candidate_messages.append({
                "role": "assistant",
                "content": semantic_note,
                "tool_name": "semantic_memory_context",
                "token_count": self._count_tokens(semantic_note),
            })
            semantic_signature = f"{query or ''}|{semantic_debug}"
            if semantic_signature != getattr(self, "_last_semantic_memory_debug_signature", None):
                self._emit_tool_output(
                    "memory_semantic_select",
                    {"query": query or ""},
                    semantic_debug,
                )
                self._last_semantic_memory_debug_signature = semantic_signature

        deep_note, deep_debug = self._build_deep_memory_note(query=query)
        if deep_note:
            candidate_messages.append({
                "role": "assistant",
                "content": deep_note,
                "tool_name": "deep_memory_context",
                "token_count": self._count_tokens(deep_note),
            })
            deep_signature = f"{query or ''}|{deep_debug}"
            if deep_signature != getattr(self, "_last_deep_memory_debug_signature", None):
                self._emit_tool_output(
                    "memory_deep_select",
                    {"query": query or ""},
                    deep_debug,
                )
                self._last_deep_memory_debug_signature = deep_signature

        # Cross-session context note — fetched async, cached for sync access.
        _cs_cache = getattr(self, "_cross_session_context_cache", None)
        if isinstance(_cs_cache, dict) and _cs_cache.get("note"):
            _cs_note = str(_cs_cache["note"])
            candidate_messages.append({
                "role": "assistant",
                "content": _cs_note,
                "tool_name": "cross_session_context",
                "token_count": self._count_tokens(_cs_note),
            })
            _cs_sig = f"cross_session|{hash(_cs_note)}"
            if _cs_sig != getattr(self, "_last_cross_session_debug_signature", None):
                self._emit_tool_output(
                    "memory_cross_session",
                    {"selectors": _cs_cache.get("selectors", [])},
                    f"Cross-session context injected ({len(_cs_note)} chars)",
                )
                self._last_cross_session_debug_signature = _cs_sig

        # Playbook context note — inject proven patterns for similar tasks.
        if hasattr(self, "_build_playbook_context_note_sync") and query:
            try:
                _pb_note = self._build_playbook_context_note_sync(query)
                if _pb_note:
                    candidate_messages.append({
                        "role": "assistant",
                        "content": _pb_note,
                        "tool_name": "playbook_context",
                        "token_count": self._count_tokens(_pb_note),
                    })
            except Exception:
                pass  # best-effort — never block message assembly

        planning_note = self._build_pipeline_note(planning_pipeline or {})
        if planning_note:
            candidate_messages.append({
                "role": "assistant",
                "content": planning_note,
                "tool_name": "planning_context",
                "token_count": self._count_tokens(planning_note),
            })
        list_note = self._build_list_task_note(list_task_plan or {})
        if list_note:
            candidate_messages.append({
                "role": "assistant",
                "content": list_note,
                "tool_name": "list_task_memory",
                "token_count": self._count_tokens(list_note),
            })
        scale_note = self._build_scale_progress_note()
        if scale_note:
            candidate_messages.append({
                "role": "assistant",
                "content": scale_note,
                "tool_name": "scale_progress",
                "token_count": self._count_tokens(scale_note),
            })
        todo_note = self._build_todo_context_note()
        if todo_note:
            candidate_messages.append({
                "role": "assistant",
                "content": todo_note,
                "tool_name": "todo_context",
                "token_count": self._count_tokens(todo_note),
            })
        contacts_note = self._build_contacts_context_note(query or "") if query else ""
        if contacts_note:
            candidate_messages.append({
                "role": "assistant",
                "content": contacts_note,
                "tool_name": "contacts_context",
                "token_count": self._count_tokens(contacts_note),
            })
        scripts_note = self._build_scripts_context_note(query or "") if query else ""
        if scripts_note:
            candidate_messages.append({
                "role": "assistant",
                "content": scripts_note,
                "tool_name": "scripts_context",
                "token_count": self._count_tokens(scripts_note),
            })
        apis_note = self._build_apis_context_note(query or "") if query else ""
        if apis_note:
            candidate_messages.append({
                "role": "assistant",
                "content": apis_note,
                "tool_name": "apis_context",
                "token_count": self._count_tokens(apis_note),
            })
        datastore_note = self._build_datastore_context_note()
        if datastore_note:
            candidate_messages.append({
                "role": "assistant",
                "content": datastore_note,
                "tool_name": "datastore_context",
                "token_count": self._count_tokens(datastore_note),
            })
        insights_note = self._build_insights_context_note()
        if insights_note:
            candidate_messages.append({
                "role": "assistant",
                "content": insights_note,
                "tool_name": "insights_context",
                "token_count": self._count_tokens(insights_note),
            })
        nervous_note = self._build_nervous_system_context_note()
        if nervous_note:
            candidate_messages.append({
                "role": "assistant",
                "content": nervous_note,
                "tool_name": "nervous_system_context",
                "token_count": self._count_tokens(nervous_note),
            })
        briefing_note = self._build_briefing_context_note()
        if briefing_note:
            candidate_messages.append({
                "role": "assistant",
                "content": briefing_note,
                "tool_name": "briefing_context",
                "token_count": self._count_tokens(briefing_note),
            })
        # Workspace manifest — compact listing of files created/modified
        # this session.  Gives the LLM a project map without needing
        # full file contents in history (those are compacted on disk).
        workspace_note = (
            self._build_workspace_manifest_note()
            if hasattr(self, "_build_workspace_manifest_note")
            else ""
        )
        if workspace_note:
            candidate_messages.append({
                "role": "assistant",
                "content": workspace_note,
                "tool_name": "workspace_manifest",
                "token_count": self._count_tokens(workspace_note),
            })

        # "BTW" live instructions — injected by the user while a task is
        # running.  Each one becomes a user message so the model treats
        # them as direct instructions.
        _btw_list: list[str] = getattr(self, "_btw_instructions", None) or []
        if _btw_list:
            _btw_block = "\n".join(
                f"- {inst}" for inst in _btw_list
            )
            _btw_note = (
                "[IMPORTANT — Additional instructions from the user (added while this task is running). "
                "Take these into account for ALL remaining work.]\n\n"
                + _btw_block
            )
            candidate_messages.append({
                "role": "user",
                "content": _btw_note,
                "token_count": self._count_tokens(_btw_note),
            })

        selected_reversed: list[dict[str, Any]] = []
        used_tokens = 0
        dropped_messages = 0
        included_latest_user = False
        for msg in reversed(candidate_messages):
            msg_tokens = self._ensure_message_token_count(msg)
            must_include_latest_user = (
                (not included_latest_user) and str(msg.get("role", "")) == "user"
            )
            # The scale progress note is critical during incremental
            # processing — it prevents the LLM from re-globbing or
            # losing track of the worklist.  Always include it.
            is_scale_note = str(msg.get("tool_name", "")) == "scale_progress"
            must_include = must_include_latest_user or is_scale_note
            if used_tokens + msg_tokens <= history_budget or must_include:
                selected_reversed.append(msg)
                used_tokens += msg_tokens
                if must_include_latest_user:
                    included_latest_user = True
            else:
                dropped_messages += 1

        selected_messages = list(reversed(selected_reversed))
        selected_messages = self._normalize_selected_messages_for_provider(selected_messages)
        for msg in selected_messages:
            # Append system_hint to content for the LLM (not stored in
            # the visible content field, so users don't see it in chat).
            _content = msg["content"]
            _hint = msg.get("system_hint")
            if _hint:
                _content = f"{_content}\n{_hint}"
            messages.append(
                Message(
                    role=msg["role"],
                    content=_content,
                    tool_call_id=msg.get("tool_call_id"),
                    tool_name=msg.get("tool_name"),
                    tool_calls=msg.get("tool_calls"),
                )
            )

        prompt_tokens = system_tokens + used_tokens
        self.last_context_window = {
            "context_budget_tokens": context_budget,
            "system_tokens": system_tokens,
            "history_budget_tokens": history_budget,
            "history_tokens": used_tokens,
            "prompt_tokens": prompt_tokens,
            "total_messages": len(candidate_messages),
            "included_messages": len(selected_messages),
            "dropped_messages": dropped_messages,
            "historical_tool_messages_filtered": len(skipped_historical_tools),
            "memory_note_used": 1 if memory_note else 0,
            "planning_note_used": 1 if planning_note else 0,
            "scale_progress_note_used": 1 if scale_note else 0,
            "todo_note_used": 1 if todo_note else 0,
            "workspace_manifest_used": 1 if workspace_note else 0,
            "over_budget": 1 if prompt_tokens > context_budget else 0,
            "utilization": (prompt_tokens / context_budget) if context_budget else 0.0,
        }
        if self.session:
            self.session.metadata["context_window"] = dict(self.last_context_window)
        if dropped_messages:
            log.info(
                "Context window pruned history",
                dropped_messages=dropped_messages,
                included_messages=len(selected_messages),
                prompt_tokens=prompt_tokens,
                budget=context_budget,
            )

        return messages
