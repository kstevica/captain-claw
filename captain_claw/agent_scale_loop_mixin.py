"""Scale micro-loop, context trimming, progress tracking, and scale guard for Agent.

This mixin handles all aspects of the scale (batch-processing) pipeline:
- Context trimming to keep the session bounded during scale tasks
- Scale guard that intercepts off-track tool calls
- Progress note injection for LLM awareness
- Item classification and extraction mode detection
- Micro-turn scale loop with isolated per-item LLM calls
- Scale loop readiness checks and task description extraction
"""

import asyncio
import json
import os
import re
import shutil
from typing import Any

from captain_claw.config import get_config
from captain_claw.llm import Message, ToolCall
from captain_claw.logging import get_logger


log = get_logger(__name__)


class AgentScaleLoopMixin:
    """Scale loop orchestration: micro-turn processing, context trimming, guards."""

    # Tool names whose output can be trimmed after the LLM writes the
    # processed result.  These are the "extractors" and fetchers that
    # produce large raw content the LLM summarizes and then no longer needs.
    _TRIMMABLE_TOOL_NAMES = frozenset({
        "pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract",
        "web_fetch", "web_get",
    })

    # Minimum content length to bother trimming — smaller results aren't
    # worth the overhead and won't meaningfully free context.
    _TRIM_MIN_CHARS = 1500

    # How many recent extract→write message pairs to keep in full
    # context.  Everything older is compressed to minimal placeholders
    # to prevent the context from growing linearly with item count.
    _SCALE_KEEP_RECENT_PAIRS = 3

    # Maximum items to show in the progress note.  For very large lists
    # (50–200+ items) we only show the next handful so the note stays
    # compact and doesn't eat up the context budget on every iteration.
    _SCALE_NOTE_MAX_VISIBLE = 15

    # Map item type to the tool that extracts its content.
    _SCALE_ITEM_EXTRACTOR: dict[str, tuple[str, str]] = {
        # extension → (tool_name, path_arg_name)
        ".pdf": ("pdf_extract", "path"),
        ".docx": ("docx_extract", "path"),
        ".xlsx": ("xlsx_extract", "path"),
        ".pptx": ("pptx_extract", "path"),
    }

    # Regex to find the first URL embedded anywhere in a string.
    _URL_RE = re.compile(r"https?://[^\s)\]}>\"']+")

    # Regex to extract an output filename from a task description.
    # Matches patterns like:
    #   "write results to report-2025-07-30.md"
    #   "output to report.csv"
    #   "save to summary.json"
    #   "name the output file report.md"
    _OUTPUT_FILE_RE = re.compile(
        r"(?:"
        r"write\s+(?:(?:the\s+)?results?\s+)?(?:in)?to"
        r"|output\s+(?:in)?to"
        r"|save\s+(?:(?:the\s+)?results?\s+)?(?:in)?to"
        r"|(?:name|call)\s+(?:the\s+)?(?:output\s+)?file"
        r"|(?:in)?to\s+(?:a\s+)?(?:file\s+)?(?:named|called)"
        r")\s+['\"]?([^\s'\",:;]+\.(?:md|csv|json|txt|html|xml|yaml|yml|tsv))['\"]?",
        re.IGNORECASE,
    )

    # File extensions recognised as "file" items during classification.
    _KNOWN_FILE_EXTS = frozenset({
        ".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".csv",
        ".json", ".xml", ".html", ".htm", ".yaml", ".yml", ".toml",
        ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp",
    })

    # Google Workspace MIME types that can be read via ``gws docs_read``
    # (Drive export API).  All other files require ``drive_download``
    # followed by a local extraction tool (read, pdf_extract, etc.).
    _GOOGLE_NATIVE_MIMETYPES = frozenset({
        "application/vnd.google-apps.document",
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.google-apps.presentation",
    })

    # ------------------------------------------------------------------
    # Context trimming
    # ------------------------------------------------------------------

    def _trim_processed_extracts_in_session(self) -> None:
        """Aggressively trim the session context after a successful
        ``write(append=true)`` during scale-progress tasks.

        The context window grows linearly with each processed item
        because every iteration adds assistant messages (LLM reasoning),
        tool results (extracted content, write confirmations), and guard
        redirects.  Even with extract-output trimming, the accumulated
        assistant messages and write results slow down each LLM call.

        This method now performs **two levels** of trimming:

        1. **Extract outputs** — large extractor/fetcher tool results
           are replaced with a short placeholder (original behavior).
        2. **Full-pair compression** — assistant messages, write results,
           and guard redirects older than the last N extract→write pairs
           are compressed to a single-line placeholder.  This keeps the
           context at a roughly constant size regardless of how many
           items have been processed.

        The last ``_SCALE_KEEP_RECENT_PAIRS`` pairs are kept intact so
        the LLM can see its recent pattern and continue.
        """
        if not self.session:
            return
        messages = self.session.messages
        if not messages:
            return

        extract_placeholder = (
            "[Content processed and written to output file. "
            "Do NOT re-extract this file — move to the next item.]"
        )

        # ── Phase 1: Trim large extractor outputs (original) ─────
        extract_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            tool_name = str(msg.get("tool_name", "")).strip().lower()
            if tool_name not in self._TRIMMABLE_TOOL_NAMES:
                continue
            content = str(msg.get("content", ""))
            if len(content) < self._TRIM_MIN_CHARS:
                continue
            if content.startswith("Error:"):
                continue
            extract_indices.append(idx)

        if len(extract_indices) > 1:
            trimmed_count = 0
            for idx in extract_indices[:-1]:
                msg = messages[idx]
                old_content = str(msg.get("content", ""))
                if old_content == extract_placeholder:
                    continue
                msg["content"] = extract_placeholder
                msg["token_count"] = self._count_tokens(extract_placeholder)
                trimmed_count += 1
            if trimmed_count > 0:
                log.debug(
                    "Trimmed processed extracts from session",
                    trimmed=trimmed_count,
                    remaining=len(extract_indices) - trimmed_count,
                )

        # ── Phase 2: Compress old message pairs ──────────────────
        # Find write(append=true) tool results — each marks the end of
        # a completed extract→write pair.  Keep the last N pairs in full
        # context; compress everything before that.
        write_result_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            tool_name = str(msg.get("tool_name", "")).strip().lower()
            if tool_name != "write":
                continue
            args = msg.get("tool_arguments")
            if isinstance(args, dict) and args.get("append") is True:
                write_result_indices.append(idx)

        # Only compress if we have more pairs than the keep-window.
        keep = self._SCALE_KEEP_RECENT_PAIRS
        if len(write_result_indices) <= keep:
            return

        # The cutoff index: everything at or before this write result's
        # index is "old" and can be compressed.
        cutoff_idx = write_result_indices[-(keep + 1)]
        assistant_placeholder = "[Previous iteration — processed and written. See scale progress note for status.]"
        write_placeholder = "[Item written to output file.]"
        guard_placeholder = "[Guard redirect — resolved.]"
        compressed = 0

        for idx in range(cutoff_idx + 1):
            msg = messages[idx]
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", ""))

            if role == "assistant":
                # Compress assistant reasoning messages from old pairs.
                # Skip very short messages (already compact) and the
                # original user message context.
                if len(content) > 200:
                    msg["content"] = assistant_placeholder
                    msg["token_count"] = self._count_tokens(assistant_placeholder)
                    # Strip tool_calls from compressed assistant messages
                    # to avoid orphaned references.
                    msg.pop("tool_calls", None)
                    compressed += 1

            elif role == "tool":
                tool_name = str(msg.get("tool_name", "")).strip().lower()
                # Compress write tool results.
                if tool_name == "write" and len(content) > 60:
                    msg["content"] = write_placeholder
                    msg["token_count"] = self._count_tokens(write_placeholder)
                    compressed += 1
                # Compress guard redirect messages.
                elif content.startswith("SCALE GUARD:"):
                    msg["content"] = guard_placeholder
                    msg["token_count"] = self._count_tokens(guard_placeholder)
                    compressed += 1
                # Compress duplicate-blocked messages.
                elif content.startswith("DUPLICATE CALL BLOCKED:"):
                    msg["content"] = guard_placeholder
                    msg["token_count"] = self._count_tokens(guard_placeholder)
                    compressed += 1

        if compressed > 0:
            log.debug(
                "Compressed old scale-loop messages",
                compressed=compressed,
                cutoff_idx=cutoff_idx,
                total_messages=len(messages),
                kept_pairs=keep,
            )

    # ------------------------------------------------------------------
    # Scale guard — hard redirect for off-track tool calls
    # ------------------------------------------------------------------
    # When the scale progress system is active, the LLM sometimes "forgets"
    # the item list after 60+ messages and starts re-globbing or re-reading
    # the output file.  The soft hints (progress note, write reminder) are
    # not always enough.  This guard actively blocks off-track tool calls
    # and returns a synthetic tool result that tells the LLM what to do
    # next, acting as a hard state-machine that enforces the
    # extract→write→next pattern.

    def _scale_guard_intercept(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        """Check whether a tool call should be intercepted by the scale guard.

        Returns a redirect message string if the call should be blocked,
        or ``None`` if the call should proceed normally.

        Blocked scenarios:
        1. ``glob`` when items are already known — the LLM is trying to
           rediscover the file list.
        2. ``read`` on the output file — the LLM is trying to review what
           it already wrote.
        3. Repeated extraction of an already-done item.
        """
        sp = getattr(self, "_scale_progress", None)
        if sp is None:
            return None
        # Passthrough mode: the main LLM handles items directly — don't
        # intercept its tool calls with scale guard redirects.
        if str(sp.get("_extraction_mode", "")).strip() == "passthrough":
            return None
        items: list[str] = sp.get("items", [])
        if not items:
            return None
        done_items: set[str] = sp.get("done_items", set())
        remaining = [item for item in items if item not in done_items]
        if not remaining:
            return None  # all done, let the LLM finalize

        tool_lower = str(tool_name or "").strip().lower()

        # ── Guard 1: Block glob when items are already populated ──
        # Once the item list is known from a previous glob or direct URL
        # list, re-globbing is never needed.  Only activate after the
        # first successful glob (indicated by _glob_completed flag) or
        # when items are URLs (no glob discovery needed).
        has_urls = any(
            "http://" in item or "https://" in item for item in items[:5]
        )
        glob_completed = sp.get("_glob_completed", False)
        if tool_lower == "glob" and (glob_completed or has_urls):
            next_item = remaining[0]
            short = next_item
            if "/workspace/" in next_item:
                short = next_item.split("/workspace/", 1)[-1]
            return (
                f"SCALE GUARD: Item list is already known ({len(items)} items, "
                f"{len(remaining)} remaining). Do NOT re-glob.\n"
                f"Your NEXT item to process is:\n  {short}\n"
                "Process it now (fetch/read, then write its result)."
            )

        # ── Guard 2: Block read on output file(s) ──
        # Track the output file from the first write call.
        # For file_per_item mode, also block reads on any previously
        # written per-item files.
        output_file = sp.get("_output_file", "")
        output_files_set: set[str] = sp.get("_output_files", set())
        if tool_lower == "read" and isinstance(arguments, dict):
            read_path = str(arguments.get("path", arguments.get("file_path", ""))).strip()
            if read_path:
                read_abs = os.path.abspath(read_path)
                # Check against single output file
                is_output = False
                if output_file:
                    output_abs = os.path.abspath(output_file)
                    if read_abs == output_abs:
                        is_output = True
                # Check against all per-item output files
                if not is_output and output_files_set:
                    for of in output_files_set:
                        if read_abs == os.path.abspath(of):
                            is_output = True
                            break
                if is_output:
                    next_item = remaining[0]
                    short = next_item
                    if "/workspace/" in next_item:
                        short = next_item.split("/workspace/", 1)[-1]
                    done = len(done_items)
                    total = len(items)
                    return (
                        f"SCALE GUARD: Do NOT re-read the output file. "
                        f"You have written {done} of {total} items — "
                        f"trust the write. {len(remaining)} items remain.\n"
                        f"Your NEXT item to process is:\n  {short}\n"
                        "Extract/read it now, then write its result."
                    )

        # ── Guard 3: Block re-extraction of already-done items ──
        _PATH_TOOLS = {"read", "pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract"}
        _URL_TOOLS = {"web_fetch", "web_get"}
        if tool_lower in (_PATH_TOOLS | _URL_TOOLS) and isinstance(arguments, dict):
            target = ""
            if tool_lower in _PATH_TOOLS:
                target = str(arguments.get("path", arguments.get("file_path", ""))).strip()
                if target:
                    target = os.path.abspath(target)
            elif tool_lower in _URL_TOOLS:
                target = str(arguments.get("url", "")).strip()
            if target:
                # Check if this target matches any done item
                for done_item in done_items:
                    done_abs = done_item
                    if "/" in done_item and not done_item.startswith(("http://", "https://")):
                        done_abs = os.path.abspath(done_item)
                    if target == done_abs or target.endswith("/" + done_item.rsplit("/", 1)[-1]):
                        next_item = remaining[0]
                        short = next_item
                        if "/workspace/" in next_item:
                            short = next_item.split("/workspace/", 1)[-1]
                        return (
                            f"SCALE GUARD: This item has already been processed and "
                            f"written to the output file. Do NOT re-extract it.\n"
                            f"Your NEXT unprocessed item is:\n  {short}\n"
                            "Extract/read it now, then write(append=true) its summary."
                        )

        # ── Guard 4: Block batch-fetching of multiple remaining items ──
        # When the LLM tries to fetch/extract multiple list items in a
        # single batch (instead of processing one-at-a-time), allow only
        # the first extraction and block subsequent ones.  This forces
        # the incremental fetch→process→write pattern.
        _EXTRACTION_TOOLS = _PATH_TOOLS | _URL_TOOLS
        if tool_lower in _EXTRACTION_TOOLS and isinstance(arguments, dict):
            target = ""
            if tool_lower in _URL_TOOLS:
                target = str(arguments.get("url", "")).strip()
            else:
                target = str(arguments.get("path", arguments.get("file_path", ""))).strip()
            # Check if this target is one of the remaining list items
            is_list_item = False
            if target:
                for item in remaining:
                    if target == item or (
                        not item.startswith(("http://", "https://"))
                        and os.path.abspath(target) == os.path.abspath(item)
                    ):
                        is_list_item = True
                        break
            if is_list_item:
                batch_extractions = sp.get("_batch_extractions", 0)
                if batch_extractions >= 1:
                    next_item = remaining[0]
                    short = next_item
                    if "/workspace/" in next_item:
                        short = next_item.split("/workspace/", 1)[-1]
                    return (
                        f"SCALE GUARD: Process ONE item at a time. You are trying "
                        f"to batch-fetch multiple list items at once — this will "
                        f"overflow the context window.\n"
                        f"Finish processing the current item first (extract → write "
                        f"its result), then move to the next.\n"
                        f"Your current/next item is:\n  {short}"
                    )
                sp["_batch_extractions"] = batch_extractions + 1

        return None

    # ------------------------------------------------------------------
    # Progress note injection
    # ------------------------------------------------------------------

    def _build_scale_progress_note(self) -> str:
        """Build a progress note injected into every LLM call during
        scale-progress tasks.

        This note gives the LLM a fresh, up-to-date worklist at every
        iteration so it never needs to re-glob or re-read the output
        file to figure out what's left.  It replaces the LLM's need
        to "remember" the file list from a glob 40+ messages ago.

        Works for any kind of list: file paths, URLs, entity names, etc.

        For very large lists (>15 remaining) only the next 15 items are
        shown to keep the injected note compact.
        """
        sp = getattr(self, "_scale_progress", None)
        if sp is None:
            return ""
        items: list[str] = sp.get("items", [])
        if not items:
            return ""
        # Passthrough mode: the main LLM handles items directly (e.g.
        # datastore tool calls) — don't inject the scale progress note
        # as it would confuse the LLM into thinking it should follow the
        # scale extract→write pattern.
        if str(sp.get("_extraction_mode", "")).strip() == "passthrough":
            return ""
        done_items: set[str] = sp.get("done_items", set())
        total = len(items)
        done = len(done_items)
        remaining = [item for item in items if item not in done_items]

        # Detect item type to tailor the instructions.
        has_paths = any("/" in item and not item.startswith("http") for item in items[:5])
        has_urls = any(item.startswith(("http://", "https://")) for item in items[:5])
        extraction_mode = str(sp.get("_extraction_mode", "file")).strip()

        # Detect output strategy
        output_strategy = str(sp.get("_output_strategy", "single_file")).strip().lower()
        filename_template = str(sp.get("_output_filename_template", "")).strip()

        lines = [
            "--- SCALE PROGRESS (auto-tracked) ---",
            f"Total items: {total}  |  Completed: {done}  |  Remaining: {len(remaining)}",
        ]
        if output_strategy == "file_per_item":
            lines.append(
                f"OUTPUT: SEPARATE file per item (template: {filename_template or 'per-item filename'}). "
                "Use write(append=false) for each item."
            )
        elif output_strategy == "no_file":
            final_action = str(sp.get("_final_action", "reply")).strip()
            lines.append(
                f"OUTPUT: No file output — final action is: {final_action}."
            )
        else:
            lines.append("OUTPUT: Single file with append=true.")
        if has_urls:
            lines.append(
                "The item list is already known (URLs). "
                "Skip any discovery/glob step — start processing directly."
            )
        lines.append("")

        # Show up to _SCALE_NOTE_MAX_VISIBLE items; beyond that just
        # indicate how many more exist.  This keeps the note at a
        # bounded size (~2-3K chars) even for 100+ item lists.
        visible = remaining[:self._SCALE_NOTE_MAX_VISIBLE]
        overflow = len(remaining) - len(visible)

        lines.append("NEXT items to process:")
        for idx, item in enumerate(visible, start=1):
            # Show just the relative portion after the workspace root
            # to keep lines short for file paths.
            short = item
            if "/workspace/" in item:
                short = item.split("/workspace/", 1)[-1]
            lines.append(f"  {idx}. {short}")
        if overflow > 0:
            lines.append(f"  ... and {overflow} more items after these.")
        lines.append("")
        lines.append(
            "\u26a1 GUARD ACTIVE: glob, output-file reads, and re-extraction of "
            "completed items will be BLOCKED automatically."
        )

        # Write instruction varies by output strategy
        if output_strategy == "file_per_item":
            write_instr = (
                "write its result to a NEW per-item file "
                f"(template: {filename_template or 'per-item filename'}, append=false)."
            )
        elif output_strategy == "no_file":
            write_instr = "deliver its result (no file output needed)."
        else:
            write_instr = "write(append=true) its summary."

        if extraction_mode == "research":
            lines.append(
                f"Process the NEXT entity in this list. Search the web for it, "
                f"synthesize the findings, then immediately {write_instr}"
            )
        elif extraction_mode == "inline":
            lines.append(
                f"Process the NEXT item in this list using ONLY the information "
                f"already available from the source page — do NOT fetch any URLs "
                f"or search the web. Then immediately {write_instr}"
            )
        elif has_paths:
            lines.append(
                f"Process the NEXT item in this list. Extract/read it, "
                f"then immediately {write_instr}"
            )
        elif has_urls:
            lines.append(
                f"Process the NEXT URL in this list. Fetch it, "
                f"then immediately {write_instr}"
            )
        else:
            lines.append(
                f"Process the NEXT item in this list. Process it, "
                f"then immediately {write_instr}"
            )
        lines.append("--- END SCALE PROGRESS ---")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Item classification and extraction detection
    # ------------------------------------------------------------------

    # Regex that matches the ENTIRE prefix when it consists exclusively of
    # label-like tokens: dates, keywords, numbers, and separators — but NO
    # entity-name words.  Used to distinguish:
    #   "Date: 18.07.2025. — URL:" → label (url mode)
    #   "Acme Corp —"              → entity name (research mode)
    #
    # Strategy: after stripping separators, the prefix is checked against
    # this "only labels" pattern.  If it matches, the item is a labelled URL.
    _LABEL_ONLY_PREFIX_RE = re.compile(
        r"^("
        # Any mix of: label keywords, dates, numbers, separators, whitespace
        r"(?:"
        # Label keywords (case-insensitive)
        r"(?:date|url|item|link|page|entry|nr|no|num|number)"
        r"|"
        # DD.MM.YYYY or DD/MM/YYYY dates
        r"\d{1,2}[./]\d{1,2}[./]\d{2,4}\.?"
        r"|"
        # YYYY-MM-DD or YYYY/MM/DD dates
        r"\d{4}[-/]\d{2}[-/]\d{2}"
        r"|"
        # Pure numbers (with optional # prefix)
        r"#?\d{1,4}"
        r"|"
        # Separator characters (dash, em-dash, colon, pipe, etc.)
        r"[\s:.\-—–|,;#/]+"
        r")+"
        r")$",
        re.IGNORECASE,
    )

    # Signals in per_member_action that indicate "just fetch the URL"
    _FETCH_ONLY_RE = re.compile(
        r"\b(?:fetch(?:es)?|download|read|scrape|extract from|get the page|"
        r"visit the (?:provided |given )?(?:url|page|link)|"
        r"do not (?:follow|visit|fetch|search)|"
        r"only (?:the )?provided|single page)\b",
        re.IGNORECASE,
    )
    # Signals in per_member_action that indicate research is needed
    _RESEARCH_SIGNAL_RE = re.compile(
        r"\b(?:research|investigate|look up|find (?:information|details|data) about|"
        r"search (?:for|the web)|gather (?:information|data)|"
        r"explore|deep dive)\b",
        re.IGNORECASE,
    )
    # User-level signals that forbid external fetching / link following.
    # Checked against the raw user input / task description so that explicit
    # instructions like "do not follow any links" suppress research mode even
    # when items look like named entities with URLs.
    _NO_EXTERNAL_FETCH_RE = re.compile(
        r"(?:"
        r"do\s+not\s+(?:follow|visit|fetch|open|click|search|browse)(?:\s+or\s+(?:follow|visit|fetch|open|click|search|browse))?\s+(?:any\s+)?(?:other\s+)?(?:links?|urls?|pages?|external)"
        r"|(?:don'?t|never)\s+(?:follow|visit|fetch|open|click|search|browse)(?:\s+or\s+(?:follow|visit|fetch|open|click|search|browse))?\s+(?:any\s+)?(?:other\s+)?(?:links?|urls?|pages?|external)"
        r"|read\s+only\s+(?:that|the|this)\s+(?:single\s+)?page"
        r"|only\s+(?:read|use|extract\s+from)\s+(?:that|the|this)\s+(?:single\s+)?page"
        r"|(?:no|without)\s+(?:external|additional|extra)\s+(?:fetching|searching|browsing|requests)"
        r"|single[- ]page\s+(?:only|extraction)"
        r"|do\s+not\s+(?:fetch|visit|open)\s+(?:or\s+(?:fetch|visit|open)\s+)?any\s+other"
        r")",
        re.IGNORECASE,
    )
    # User-level signals that indicate save/store/create intent — the user
    # wants to PERSIST already-known data, NOT research or search the web.
    # When this matches user_input, research mode is suppressed for plain-text
    # entities because the items are already the data to be saved.
    _SAVE_INTENT_RE = re.compile(
        r"(?:"
        # "save this/these/it/them/files", "store this", "create a table"
        r"(?:save|store|persist|keep|insert|put|index)\s+(?:this|these|it|them|files?|the\s+(?:data|results?|items?|list|info|files?))"
        # "create a [optional words] table" — allows "create a shopping list table"
        r"|create\s+(?:a\s+)?(?:\w+\s+){0,3}(?:table|datastore|dataset|database|spreadsheet|csv)"
        r"|(?:save|store|write|add|import|insert|index)\s+(?:to|into|in|on)\s+(?:a\s+)?(?:table|datastore|dataset|database|spreadsheet|file|csv|memory|deep\s+memory|typesense)"
        # "make/build a [optional words] table from/with/of"
        r"|(?:make|build)\s+(?:a\s+)?(?:\w+\s+){0,3}(?:table|spreadsheet|csv)\s+(?:from|with|of)"
        # "put/add this/these [optional words] to/into/in" — allows "add these items to it"
        r"|(?:put|add)\s+(?:this|these|it|them|the\s+(?:data|results?|items?|list))(?:\s+\w+){0,3}\s+(?:to|into|in)\b"
        # "save to datastore", "write to table", "index to memory"
        r"|(?:save|write|export|index)\s+(?:it\s+)?(?:to|into|on)\s+(?:a\s+)?(?:table|datastore|file|memory|deep\s+memory|typesense)"
        # "just save/store", "only save/store"
        r"|(?:just|only)\s+(?:save|store|create|write|keep|index)"
        # "to/on memory", "to/on deep memory" — covers "index X to/on memory"
        r"|(?:to|into|on)\s+(?:deep\s+)?memory"
        r")",
        re.IGNORECASE,
    )

    @staticmethod
    def _classify_item_extraction_mode(
        items: list[str],
        per_member_action: str = "",
        user_input: str = "",
    ) -> str:
        """Classify list items into an extraction mode.

        Samples up to 10 items and votes:
        - ``"url"``         — majority are pure HTTP(S) URLs (no name prefix),
          OR labelled URLs where the prefix is a date/metadata label
        - ``"inline"``      — same as ``"url"`` but the user explicitly forbade
          external fetching; each item is processed using only the
          ``_member_context`` snippet from the already-fetched page
        - ``"file"``        — majority contain path separators or known extensions
        - ``"research"``    — majority are plain-text entities (names, titles),
          OR entities with accompanying URLs (``"Name — https://…"``),
          AND the user or per_member_action explicitly requested web
          research.  Research mode is NEVER auto-enabled — it requires
          an explicit opt-in signal.
        - ``"passthrough"`` — plain-text entities where no explicit research
          signal is present.  The micro-loop is skipped and the main LLM
          handles the items directly with tool calls (datastore, todo,
          etc.).  This is the DEFAULT for entity lists.

        Items that have BOTH a text prefix and an embedded URL are examined
        further: if the prefix is a date, numeric label, or metadata tag
        (e.g. ``"Date: 18.07.2025. — URL: https://…"``), the item is a
        *labelled URL* and counts toward ``"url"`` mode, NOT research.  Only
        items where the prefix looks like a meaningful entity name
        (e.g. ``"Company Name — https://…"``) trigger ``"research"`` mode.

        When ``per_member_action`` is provided (from the list task plan), it
        can override ambiguous cases: if the action text says to "fetch" or
        "extract from page" without mentioning "research"/"search", labelled
        URLs are classified as ``"url"`` mode.

        When ``user_input`` is provided, it is checked for explicit
        no-external-fetch signals (e.g. "do not follow any links",
        "read only that page").  These override entity-name heuristics
        so that named URLs are treated as plain URLs (no research).

        When ``user_input`` contains save/store/create intent signals
        (e.g. "create a table and save this", "store to datastore",
        "index to memory"), research mode is suppressed because the user
        wants to persist already-known data, not search the web.

        **IMPORTANT**: Research mode is NEVER auto-enabled.  It requires
        an explicit research signal — either from ``user_input`` (e.g.
        "research these companies") or from ``per_member_action`` (e.g.
        "research each company and summarize findings").  Without such a
        signal, plain-text entity lists default to ``"passthrough"``.

        The result is stored once in ``_scale_progress["_extraction_mode"]``
        so the micro-loop knows which extraction strategy to use.
        """
        if not items:
            return "file"

        # ── Check per_member_action for explicit fetch-only or research hints ──
        _action = per_member_action.strip().lower()
        _action_says_fetch_only = bool(
            AgentScaleLoopMixin._FETCH_ONLY_RE.search(_action)
        ) if _action else False
        _action_says_research = bool(
            AgentScaleLoopMixin._RESEARCH_SIGNAL_RE.search(_action)
        ) if _action else False

        # ── Check user_input for no-external-fetch signals ──
        # The user's raw instructions override automatic heuristics: if they
        # say "do not follow any links" or "read only that page", we suppress
        # research mode even when items look like named entities with URLs.
        _user_forbids_external = bool(
            AgentScaleLoopMixin._NO_EXTERNAL_FETCH_RE.search(user_input)
        ) if user_input else False
        if _user_forbids_external:
            _action_says_fetch_only = True

        # ── Check user_input for save/store/create intent ──
        # When the user wants to persist existing data (e.g. "create a table
        # and save this"), research mode is suppressed.  The items are already
        # the data to store — no need to web-search each entity.
        _user_wants_save = bool(
            AgentScaleLoopMixin._SAVE_INTENT_RE.search(user_input)
        ) if user_input else False

        # ── Check user_input for explicit research intent ──
        # Research mode must be explicitly requested.  We check the user's
        # raw instructions (not just per_member_action) so that phrases like
        # "research these companies" or "search the web for each" opt in.
        _user_wants_research = bool(
            AgentScaleLoopMixin._RESEARCH_SIGNAL_RE.search(user_input)
        ) if user_input else False

        # When the user explicitly wants to save/store/index data, suppress
        # research mode.  The user's save intent takes precedence over the
        # LLM-generated per_member_action — which may hallucinate research
        # keywords even when the user only wanted to persist existing data.
        # Only override if the USER themselves didn't also ask for research.
        if _user_wants_save and not _user_wants_research:
            _action_says_fetch_only = True
            _action_says_research = False

        sample = items[:10]
        pure_url_count = 0
        labelled_url_count = 0   # Date/label + URL → still url mode
        named_url_count = 0      # Entity name + URL → research mode
        file_count = 0
        entity_count = 0

        for raw in sample:
            item = raw.strip()
            if not item:
                continue
            # Pure URL check — item IS a URL (no name prefix)
            if item.startswith(("http://", "https://")):
                pure_url_count += 1
                continue
            # Item has an embedded URL — classify the prefix
            url_match = re.search(r"https?://", item)
            if url_match:
                prefix = item[:url_match.start()].strip().rstrip("—–-:.,;|").strip()
                # If the prefix is a date/label/number, this is a labelled URL
                # (the URL is the target, the prefix is just metadata).
                if (
                    not prefix
                    or AgentScaleLoopMixin._LABEL_ONLY_PREFIX_RE.match(prefix)
                ):
                    labelled_url_count += 1
                elif _action_says_fetch_only and not _action_says_research:
                    # The task explicitly says "fetch only" — treat as URL
                    labelled_url_count += 1
                else:
                    # Prefix looks like an entity name → research
                    named_url_count += 1
                continue
            # Path separator check
            if "/" in item or "\\" in item:
                file_count += 1
                continue
            # Known file extension check
            ext = os.path.splitext(item)[-1].lower()
            if ext and ext in AgentScaleLoopMixin._KNOWN_FILE_EXTS:
                file_count += 1
                continue
            # Everything else is a plain-text entity
            entity_count += 1

        n = len(sample)
        # Labelled URLs are effectively URLs — count them with pure URLs.
        url_count = pure_url_count + labelled_url_count
        # Only entity names (+ named URLs) count toward research.
        research_count = entity_count + named_url_count
        if url_count >= n * 0.5:
            # User explicitly said "do not fetch/follow links" — use inline
            # extraction from already-fetched page content (_member_context)
            # instead of fetching each URL individually.
            if _user_forbids_external:
                return "inline"
            return "url"
        if research_count >= n * 0.5:
            # Research mode requires an EXPLICIT opt-in signal — either
            # the user asked for web research in their input, or the
            # LLM-generated per_member_action explicitly says "research".
            # Without this signal, the agent must NOT autonomously search
            # the web for each entity.
            #
            # When save intent is detected, return "passthrough" so the
            # micro-loop is skipped entirely and the main LLM handles the
            # items directly (e.g. via datastore tool calls).
            # Note: _action_says_research is already cleared by the save
            # intent guard above, so only _user_wants_research matters.
            if _user_wants_save and not _user_wants_research:
                return "passthrough"
            if _action_says_fetch_only and not _action_says_research:
                return "file"
            # Only enable research mode if explicitly requested.
            if _action_says_research or _user_wants_research:
                return "research"
            # Default: treat plain entities as passthrough — let the main
            # LLM decide how to handle them without autonomous web searching.
            return "passthrough"
        return "file"

    @staticmethod
    def _derive_member_label(item: str) -> str:
        """Derive a short label from an item string for use in per-item filenames.

        For items like ``"18.07.2025. https://example.com/page"`` → ``"18.07.2025"``
        For items like ``"https://example.com/page"`` → ``"page"`` (last URL path segment)
        For file paths → the filename without extension.
        """
        item = item.strip()
        # Try extracting a date-like prefix (e.g. "18.07.2025. https://...")
        date_match = re.match(r"^(\d{1,2}\.\d{1,2}\.\d{4})", item)
        if date_match:
            return date_match.group(1)
        # Try extracting a date prefix with dashes (e.g. "2025-07-18 https://...")
        date_match2 = re.match(r"^(\d{4}-\d{2}-\d{2})", item)
        if date_match2:
            return date_match2.group(1)
        # For plain URLs, use the last meaningful path segment
        if item.startswith(("http://", "https://")):
            try:
                from urllib.parse import urlparse
                path = urlparse(item).path.strip("/")
                if path:
                    return path.rsplit("/", 1)[-1].split(".")[0] or path
            except Exception:
                pass
            return item.rsplit("/", 1)[-1][:40]
        # For "prefix - url" or "prefix url" items with embedded URLs
        url_match = re.search(r"https?://", item)
        if url_match:
            prefix = item[:url_match.start()].strip().rstrip(".-:").strip()
            if prefix:
                return prefix
        # For file paths, use filename without extension
        if "/" in item or "\\" in item:
            basename = os.path.basename(item)
            name, _ = os.path.splitext(basename)
            return name or basename
        # Fallback: the item itself, cleaned up
        return re.sub(r"[^\w.\-]", "_", item)[:60]

    def _detect_item_extractor(self, item: str) -> tuple[str, dict[str, Any]]:
        """Determine which tool and arguments to use for extracting an item.

        Returns ``(tool_name, arguments_dict)``.

        Handles plain URLs, URLs with prefixes (``"18.07.2025 - https://…"``),
        file paths, and document extensions.
        """
        # URL items → web_fetch
        # Check both "starts with URL" and "contains a URL somewhere"
        # (e.g. "18.07.2025 - https://example.com/page")
        if item.startswith(("http://", "https://")):
            return "web_fetch", {"url": item.strip()}
        url_match = self._URL_RE.search(item)
        if url_match:
            return "web_fetch", {"url": url_match.group(0)}
        # File path items → check extension
        ext = os.path.splitext(item)[-1].lower()
        if ext in self._SCALE_ITEM_EXTRACTOR:
            tool_name, arg_name = self._SCALE_ITEM_EXTRACTOR[ext]
            return tool_name, {arg_name: item}
        # If the item has no path separators and no file extension it is
        # likely a plain-text entity (company name, product, etc.).
        # Return a no-op sentinel ("_passthrough") so the micro-loop uses
        # the item text itself as content without triggering a web search.
        # Web search is only used in explicit "research" extraction mode,
        # which has its own dedicated path in the micro-loop.
        if "/" not in item and "\\" not in item:
            ext = os.path.splitext(item)[-1].lower()
            if not ext:
                return "_passthrough", {"text": item}
        # Default: plain file read
        return "read", {"path": item}

    # ------------------------------------------------------------------
    # Google Drive file map for scale extraction
    # ------------------------------------------------------------------

    def _build_gdrive_file_map(self) -> dict[str, dict[str, str]]:
        """Scan session messages for ``gws drive_list`` / ``drive_search``
        results and build a file-name → ``{id, mimeType}`` map for
        Google-Drive-aware extraction in the scale micro-loop.

        Returns an empty dict when ``gws`` is not available or no
        Drive file listing results exist in the session.
        """
        if not shutil.which("gws"):
            return {}

        session = getattr(self, "session", None)
        if session is None:
            return {}

        file_map: dict[str, dict[str, str]] = {}
        for msg in session.messages:
            if msg.get("role") != "tool":
                continue
            if msg.get("tool_name") != "gws":
                continue
            args = msg.get("tool_arguments")
            if not isinstance(args, dict):
                continue
            action = args.get("action", "")
            if action not in ("drive_list", "drive_search"):
                continue
            content = msg.get("content", "")
            if not content:
                continue
            try:
                data = json.loads(content)
                files = data.get("files", [])
                if not isinstance(files, list):
                    continue
                for f in files:
                    name = f.get("name", "")
                    file_id = f.get("id", "")
                    mime_type = f.get("mimeType", "")
                    if name and file_id:
                        file_map[name] = {"id": file_id, "mimeType": mime_type}
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue

        return file_map

    @staticmethod
    def _lookup_gdrive_file(
        item: str,
        gdrive_map: dict[str, dict[str, str]],
    ) -> dict[str, str] | None:
        """Look up a scale item in the Google Drive file map.

        Tries exact match first, then case-insensitive, then substring
        matching (Drive file name contained in item text or vice versa).
        Returns ``{id, mimeType}`` or ``None``.
        """
        if not gdrive_map:
            return None

        # Exact match
        if item in gdrive_map:
            return gdrive_map[item]

        # Case-insensitive match
        item_lower = item.strip().lower()
        for name, info in gdrive_map.items():
            if name.strip().lower() == item_lower:
                return info

        # Substring match — one name contained in the other
        for name, info in gdrive_map.items():
            name_lower = name.strip().lower()
            if name_lower in item_lower or item_lower in name_lower:
                return info

        return None

    # ------------------------------------------------------------------
    # Prompt building for micro-loop
    # ------------------------------------------------------------------

    def _build_scale_item_prompt(
        self,
        task_description: str,
        item_label: str,
        extracted_content: str,
        is_first_item: bool = False,
        previous_summary_sample: str = "",
        output_strategy: str = "single_file",
        extraction_mode: str = "file",
        source_context: str = "",
    ) -> list[Message]:
        """Build a minimal message list for processing a single scale item.

        The resulting prompt is self-contained and does not reference any
        prior conversation.  It contains:
        - A focused system message
        - A user message with the extracted content and task

        ``previous_summary_sample`` optionally provides the last written
        summary so the LLM can match the formatting pattern.

        ``output_strategy`` controls the write instructions:
        - ``single_file``: output is appended to a shared file
        - ``file_per_item``: output is the complete content for a standalone file

        ``extraction_mode`` adjusts the system prompt:
        - ``research``: web-search-based extraction (synthesize multiple sources)
        - ``file`` / ``url``: standard document extraction

        ``source_context`` optionally provides context about this item from the
        original source article (e.g. country, brief description).  This is the
        AUTHORITATIVE source for the entity's identity and should take precedence
        over web-search results when there are conflicts (e.g. multiple companies
        sharing the same name or domain).
        """
        if output_strategy == "file_per_item":
            output_instruction = (
                "- Just output the COMPLETE file content for this single item.\n"
                "- The output will be written as a standalone file (not appended).\n"
                "- Include any necessary headers (e.g. CSV headers) in the output."
            )
        elif output_strategy == "no_file":
            output_instruction = (
                "- Just output the processed result for this single item.\n"
                "- The output will be indexed to a search engine (not written to a file).\n"
                "- Keep the output structured and self-contained."
            )
        else:
            output_instruction = (
                "- Just output the processed content that should be appended to the output file."
            )

        if extraction_mode == "research":
            _source_ctx_rule = ""
            if source_context:
                _source_ctx_rule = (
                    "- IMPORTANT — SOURCE CONTEXT: The item comes from a specific "
                    "article/source that provides authoritative context about this "
                    "entity (see SOURCE CONTEXT below). This context is the ground "
                    "truth for the entity's IDENTITY: its name, country/location, "
                    "and what it is. When web search results describe a DIFFERENT "
                    "entity that happens to share the same name or domain, you MUST "
                    "use the source context to identify the correct entity and "
                    "DISCARD information about the wrong one. For example, if the "
                    "source says 'Acme Corp, Germany' but web results describe an "
                    "unrelated American company, the German entity is the correct one.\n"
                )
            system_text = (
                "You are a research assistant. Your job is to synthesize web "
                "search results about ONE entity and produce a factual summary.\n\n"
                "Rules:\n"
                "- The EXTRACTED CONTENT below contains web search results from "
                "multiple sources about this entity.\n"
                "- Synthesize the information into the format requested by the TASK.\n"
                "- Focus ONLY on facts mentioned in the sources — do NOT fabricate.\n"
                f"{_source_ctx_rule}"
                "- If sources conflict, prefer more authoritative/recent sources.\n"
                "- If no relevant information was found, leave the field blank as "
                "instructed by the task.\n"
                "- Produce ONLY the processed result for this single entity.\n"
                "- Do NOT include any preamble, commentary, or meta-discussion.\n"
                "- Do NOT ask the user any questions.\n"
                "- Do NOT mention other items, the total count, or the overall task.\n"
                "\n"
                "OUTPUT FORMAT — strict rules:\n"
                "- Use a Markdown heading (## Entity Name) followed by labeled fields, "
                "one per line: **Field**: value\n"
                "- NEVER output Markdown tables (no | pipes). Tables are hard to read "
                "and break when values are long.\n"
                "- NEVER output CSV or TSV rows.\n"
                "- Keep it human-readable: headings, bold labels, short paragraphs.\n"
                f"{output_instruction}"
            )
        elif extraction_mode == "inline":
            _has_table_ref_inline = (
                previous_summary_sample
                and not is_first_item
                and "|" in previous_summary_sample
            )
            if _has_table_ref_inline:
                _inline_format_rule = (
                    "- You MUST match the FORMAT REFERENCE exactly. If the "
                    "reference is a Markdown table row, output a single table "
                    "row (starting with |) with the same columns.\n"
                    "- Keep each cell concise (1-3 sentences max).\n"
                )
            else:
                _inline_format_rule = (
                    "- Use Markdown with headings and **bold labels** for structure. "
                    "NEVER output Markdown tables (no | pipes) or CSV/TSV rows — "
                    "use labeled fields instead (e.g. **Field**: value).\n"
                )
            system_text = (
                "You are a document processing assistant. Your job is to process "
                "ONE item and produce a focused summary or processed result.\n\n"
                "Rules:\n"
                "- The EXTRACTED CONTENT below is the FULL source page that contains "
                "information about MANY items. Focus ONLY on the specific ITEM "
                "identified above — find the section about that item and extract "
                "every detail available for it.\n"
                "- This page content is ALL the information available — do NOT "
                "indicate that you need to fetch more information.\n"
                "- If a field is not mentioned on the page for this item, write "
                "'N/A' — do not guess or fabricate.\n"
                "- Produce ONLY the processed result/summary for this single item.\n"
                "- Do NOT include any preamble, commentary, or meta-discussion.\n"
                "- Do NOT ask the user any questions.\n"
                "- Do NOT mention other items, the total count, or the overall task.\n"
                "- Do NOT echo or reproduce the extracted content — summarize it.\n"
                "- NEVER include 'EXTRACTED CONTENT' sections in your output.\n"
                f"{_inline_format_rule}"
                f"{output_instruction}"
            )
        else:
            # When a format reference is available and uses a markdown table
            # (pipes), the LLM must match that style.  Otherwise, default to
            # headings + bold-label format (which is safer for long content).
            _has_table_ref = (
                previous_summary_sample
                and not is_first_item
                and "|" in previous_summary_sample
            )
            if _has_table_ref:
                _format_rule = (
                    "- You MUST match the FORMAT REFERENCE exactly. If the "
                    "reference is a Markdown table row, output a single table "
                    "row (starting with |) with the same columns.\n"
                    "- Keep each cell concise (1-3 sentences max).\n"
                )
            else:
                _format_rule = (
                    "- Use Markdown with headings and **bold labels** for structure. "
                    "NEVER output Markdown tables (no | pipes) or CSV/TSV rows — "
                    "use labeled fields instead (e.g. **Field**: value).\n"
                )
            system_text = (
                "You are a document processing assistant. Your job is to process "
                "ONE item and produce a focused summary or processed result.\n\n"
                "Rules:\n"
                "- Produce ONLY the processed result/summary for this single item.\n"
                "- Do NOT include any preamble, commentary, or meta-discussion.\n"
                "- Do NOT ask the user any questions.\n"
                "- Do NOT mention other items, the total count, or the overall task.\n"
                "- Do NOT echo or reproduce the extracted content — summarize it.\n"
                "- NEVER include 'EXTRACTED CONTENT' sections in your output.\n"
                f"{_format_rule}"
                f"{output_instruction}"
            )

        user_parts = [f"TASK: {task_description}\n"]
        user_parts.append(f"ITEM: {item_label}\n")

        if source_context:
            user_parts.append(
                f"SOURCE CONTEXT (authoritative — from the original article/source):\n"
                f"{source_context}\n"
            )

        if previous_summary_sample and not is_first_item:
            # Strip any leaked extracted-content echoes from the reference.
            _ref = previous_summary_sample
            for _marker in ("EXTRACTED CONTENT:", "===\n#"):
                _cut = _ref.find(_marker)
                if _cut > 0:
                    _ref = _ref[:_cut].rstrip()
            user_parts.append(
                "FORMAT REFERENCE (match this style):\n"
                "---\n"
                f"{_ref[:1500]}\n"
                "---\n"
            )

        # Truncate extracted content to keep the prompt bounded.
        # Most documents are well under 80K chars; this is a safety cap.
        max_content = 80000
        content_truncated = extracted_content[:max_content]
        if len(extracted_content) > max_content:
            content_truncated += "\n... [content truncated for processing]"
        user_parts.append(
            "EXTRACTED CONTENT:\n"
            "===\n"
            f"{content_truncated}\n"
            "===\n"
        )
        user_parts.append(
            "Now produce the processed result for this single item. "
            "Output ONLY the content to be appended."
        )

        return [
            Message(role="system", content=system_text),
            Message(role="user", content="\n".join(user_parts)),
        ]

    def _resolve_per_item_filename(
        self,
        item: str,
        template: str,
        output_dir: str,
    ) -> str:
        """Resolve the per-item output filename from a template.

        ``template`` uses ``{member_label}`` as the placeholder.
        ``output_dir`` is the directory where files should be written.
        """
        label = self._derive_member_label(item)
        filename = template.replace("{member_label}", label)
        if output_dir:
            return os.path.join(output_dir, filename)
        return filename

    @staticmethod
    def _extract_output_filename(text: str) -> str:
        """Try to extract an output filename from a task description.

        Looks for patterns like "write results to FILENAME.md" in the text.
        Returns the extracted filename or empty string if not found.
        """
        m = AgentScaleLoopMixin._OUTPUT_FILE_RE.search(text)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _derive_output_filename_from_context(text: str) -> str:
        """Derive an output filename from the task context text.

        Extracts the task title from a worker prompt (format:
        ``Task: {title}\\n``) and sanitizes it into a valid filename.
        Common action prefixes (e.g. "Fetch and extract", "Process")
        and filler words (e.g. "page", "data for") are stripped to
        produce cleaner filenames like ``ProjectName-2025-07-31.md``
        instead of ``Fetch_and_extract_ProjectName_page_2025-07-31.md``.

        Returns a sanitized filename or empty string if no title can
        be extracted.
        """
        # Try to extract task title from worker prompt format.
        title_match = re.match(r"Task:\s*(.+)", text.strip())
        if not title_match:
            return ""
        title = title_match.group(1).strip()
        if not title:
            return ""

        # ---- Strip common action-verb prefixes ----
        # These are typical task-title prefixes generated by the LLM
        # decomposer that add no value to the filename.
        _PREFIX_RE = re.compile(
            r"^(?:"
            r"fetch\s+and\s+extract"
            r"|fetch\s+and\s+process"
            r"|fetch\s+and\s+analyze"
            r"|fetch\s+and\s+summarize"
            r"|extract\s+and\s+process"
            r"|extract\s+and\s+analyze"
            r"|extract\s+and\s+summarize"
            r"|collect\s+and\s+process"
            r"|collect\s+and\s+extract"
            r"|scrape\s+and\s+extract"
            r"|download\s+and\s+extract"
            r"|download\s+and\s+process"
            r"|fetch|extract|process|analyze|summarize"
            r"|collect|scrape|download|parse|retrieve"
            r"|gather|compile|generate|create|produce"
            r"|build|prepare|get|read|pull"
            r")\s+",
            re.IGNORECASE,
        )
        title = _PREFIX_RE.sub("", title)

        # ---- Strip filler words that commonly wrap the core topic ----
        # e.g. "Report page 2025-07-31" → "Report 2025-07-31"
        #       "data for Report 2025-07-31" → "Report 2025-07-31"
        _FILLER_RE = re.compile(
            r"\b(?:page|pages|data\s+for|data\s+from|details?\s+from"
            r"|details?\s+for|info\s+from|info\s+for"
            r"|information\s+from|information\s+for"
            r"|content\s+from|content\s+for|results?\s+from"
            r"|results?\s+for|entries?\s+from|entries?\s+for)\b",
            re.IGNORECASE,
        )
        title = _FILLER_RE.sub("", title)

        # Collapse whitespace left after stripping.
        title = re.sub(r"\s+", " ", title).strip()
        if not title:
            return ""

        # Truncate to a reasonable length for a filename.
        title = title[:80]
        # Sanitize: keep alphanumeric, dots, hyphens; replace spaces
        # with hyphens for readability, everything else with underscores.
        safe = re.sub(r"\s+", "-", title)
        safe = re.sub(r"[^\w.\-]", "_", safe)
        # Collapse multiple underscores/hyphens and strip edges.
        safe = re.sub(r"[-_]{2,}", "-", safe).strip("-_")
        if not safe:
            return ""
        return f"{safe}.md"

    def _resolve_scale_output_path(self, filename_or_path: str) -> str:
        """Normalise a scale output filename for the write tool.

        If the path is already absolute, return it as-is.  Otherwise
        return the bare filename so the write tool's normal session
        scoping places it into ``<workspace>/saved/tmp/<session>/``,
        consistent with all other tool-generated files.
        """
        from pathlib import Path as _Path

        if not filename_or_path:
            return filename_or_path

        # Already absolute — respect it as-is.
        if _Path(filename_or_path).is_absolute():
            return filename_or_path

        # Return bare filename — the write tool will scope it into
        # saved/tmp/<session>/ via _normalize_under_saved().
        return filename_or_path

    # ------------------------------------------------------------------
    # Micro-turn scale loop — isolated per-item processing
    # ------------------------------------------------------------------
    # Instead of feeding the entire growing conversation to the LLM at
    # each iteration (which causes O(n) context growth, latency creep,
    # and eventual LLM confusion), the micro-turn loop processes each
    # item with a single isolated LLM call.
    #
    # Flow for each item:
    # 1. Extract/fetch the item content directly via tool execution
    # 2. Make one LLM call with a minimal prompt:
    #    system + task description + extracted content → summary
    # 3. Write the summary to the output file directly via tool execution
    # 4. Discard the per-item LLM conversation entirely
    #
    # Result: constant context size (~system + task + 1 item's content),
    # no message accumulation, no growing latency, no LLM confusion.

    async def _run_scale_micro_loop(
        self,
        task_description: str,
        output_file: str,
        turn_usage: dict[str, int],
        session_policy: dict[str, Any] | None = None,
        task_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Drive the scale loop with isolated per-item LLM calls.

        This method takes over from the main iteration loop when the
        scale-progress system has all the information it needs:
        - ``_scale_progress["items"]`` — the full item list
        - ``_scale_progress["_output_file"]`` — where to write (single_file)
        - ``_scale_progress["_output_strategy"]`` — output strategy
        - ``_scale_progress["_output_filename_template"]`` — template for
          per-item filenames (file_per_item strategy)

        For each unprocessed item:
        1. Execute the extractor tool directly (no LLM needed)
        2. Make a single LLM call with minimal context to process it
        3. Write/sink the result:
           - single_file: execute write(append=true) to the shared output file
           - file_per_item: execute write(append=false) to a per-item file
           - no_file: route to sink (typesense, email, or accumulate for reply)
        4. Update scale progress tracking

        Returns a summary dict with ``success``, ``processed``, ``failed``,
        ``total``, and ``errors``.
        """
        sp = getattr(self, "_scale_progress", None)
        if sp is None:
            return {"success": False, "error": "no scale progress active"}

        items: list[str] = sp.get("items", [])
        done_items: set[str] = sp.get("done_items", set())
        total = len(items)
        remaining = [item for item in items if item not in done_items]

        # Refuse to start with degenerate items (source URL repeated).
        if self._items_are_source_urls_only(items):
            log.warning(
                "Scale micro-loop refused: items are just source URLs",
                total=total,
                items_preview=items[:3],
            )
            return {
                "success": False,
                "error": "items are source URLs only — real list not yet extracted",
                "processed": 0,
                "failed": 0,
                "total": total,
                "errors": [],
            }

        if not remaining:
            return {
                "success": True,
                "processed": total,
                "failed": 0,
                "total": total,
                "errors": [],
            }

        import time as _time

        _loop_start = _time.monotonic()

        # Determine output strategy and extraction mode
        _output_strategy = str(sp.get("_output_strategy", "single_file")).strip().lower()
        _filename_template = str(sp.get("_output_filename_template", "")).strip()
        _is_file_per_item = _output_strategy == "file_per_item" and bool(_filename_template)
        _is_no_file = _output_strategy == "no_file"
        _final_action = str(sp.get("_final_action", "reply")).strip()
        _extraction_mode = str(sp.get("_extraction_mode", "file")).strip()
        _processing_mode = str(sp.get("_processing_mode", "summarize")).strip().lower()
        _sink_collection = str(sp.get("_sink_collection", "")).strip()
        # For file_per_item, resolve the output directory from the first
        # written file or from the output_file path.
        _output_dir = ""
        if _is_file_per_item:
            # Use the directory of the first output file if available
            _first_output = ""
            output_files_set: set[str] = sp.get("_output_files", set())
            if output_files_set:
                _first_output = next(iter(output_files_set))
            if _first_output:
                _output_dir = os.path.dirname(_first_output)
            elif output_file:
                _output_dir = os.path.dirname(output_file)

        log.info(
            "Starting micro-turn scale loop",
            total=total,
            remaining=len(remaining),
            output_file=output_file,
            output_strategy=_output_strategy,
            extraction_mode=_extraction_mode,
            processing_mode=_processing_mode,
            is_file_per_item=_is_file_per_item,
            filename_template=_filename_template,
        )
        if _processing_mode == "raw":
            log.info(
                "Tool-to-tool mode active — no LLM calls per item",
                remaining=len(remaining),
                extraction=_extraction_mode,
                sink=_final_action,
            )

        # ── Build Google Drive file map for GDrive-aware extraction ──
        _gdrive_file_map = self._build_gdrive_file_map()
        if _gdrive_file_map:
            log.info(
                "GDrive file map built for scale loop",
                files=len(_gdrive_file_map),
                names=list(_gdrive_file_map.keys())[:5],
            )

        # ── Research mode: extract search keywords once via LLM ──
        if _extraction_mode == "research" and not sp.get("_research_keywords"):
            self._emit_thinking(
                "scale_micro_loop: Extracting search keywords from task…",
                tool="scale_micro_loop",
                phase="tool",
            )
            _research_kw = await self._extract_research_keywords(
                task_description=task_description,
                all_items=items,
                turn_usage=turn_usage,
            )
            sp["_research_keywords"] = _research_kw

        processed = 0
        failed = 0
        errors: list[dict[str, Any]] = []
        _item_durations: list[float] = []  # per-item wall times for avg/ETA
        _total_prompt_tokens = 0
        _total_response_tokens = 0
        _total_extract_chars = 0

        # Capture the last written summary from the LLM's initial
        # processing so the micro-loop can provide it as a format
        # reference for subsequent items.
        last_summary = ""
        if self.session:
            for msg in reversed(self.session.messages):
                if msg.get("role") != "tool":
                    continue
                tool_name = str(msg.get("tool_name", "")).strip().lower()
                if tool_name != "write":
                    continue
                args = msg.get("tool_arguments")
                if isinstance(args, dict):
                    # Accept both append=True (single_file) and append=False
                    # (file_per_item) as valid format references.
                    last_summary = str(args.get("content", "")).strip()
                    break

        def _check_cancel() -> bool:
            """Return True if the user has requested cancellation."""
            ev: asyncio.Event | None = getattr(self, "cancel_event", None)
            return ev is not None and ev.is_set()

        def _consume_cancel() -> None:
            """Clear the cancel event after acknowledging it."""
            ev: asyncio.Event | None = getattr(self, "cancel_event", None)
            if ev is not None:
                ev.clear()

        _cancelled = False
        for idx, item in enumerate(remaining):
            # ── Cancellation check (between items) ────────────
            if _check_cancel():
                log.info("Scale micro-loop cancelled by user", processed=processed)
                _cancelled = True
                _consume_cancel()
                break

            _item_start = _time.monotonic()
            item_num = len(done_items) + 1
            item_label = item
            if "/workspace/" in item:
                # Use the LAST /workspace/ segment to get the relative path
                # e.g. "/a/b/workspace/c/workspace/pdf-test/f.pdf" → "pdf-test/f.pdf"
                item_label = item.rsplit("/workspace/", 1)[-1]

            # ETA calculation
            _avg_sec = (
                sum(_item_durations) / len(_item_durations)
                if _item_durations
                else 0
            )
            _items_left = len(remaining) - idx
            _eta_sec = round(_avg_sec * _items_left, 1) if _avg_sec else 0
            _eta_str = (
                f"~{int(_eta_sec // 60)}m{int(_eta_sec % 60):02d}s"
                if _eta_sec > 60
                else f"~{int(_eta_sec)}s"
            ) if _eta_sec else ""
            _eta_suffix = f" | ETA {_eta_str}" if _eta_str else ""

            self._emit_thinking(
                f"scale_micro_loop: Extracting ({item_num}/{total})\n"
                f"{item_label}{_eta_suffix}",
                tool="scale_micro_loop",
                phase="tool",
            )

            # ── Step 1: Extract content ─────────────────────────
            _t_extract_start = _time.monotonic()

            if _extraction_mode == "research":
                # Research mode: web_search → web_fetch → combine
                # Look up per-item source context early for search query hints.
                _member_ctx_map_r: dict[str, str] = sp.get("_member_context", {}) if sp else {}
                _item_src_ctx = _member_ctx_map_r.get(item, "")
                if not _item_src_ctx and item_label != item:
                    _item_src_ctx = _member_ctx_map_r.get(item_label, "")
                try:
                    extracted_content = await self._extract_research_item(
                        item=item,
                        task_description=task_description,
                        item_num=item_num,
                        total=total,
                        turn_usage=turn_usage,
                        session_policy=session_policy,
                        task_policy=task_policy,
                        source_context=_item_src_ctx,
                    )
                except Exception as e:
                    log.warning(
                        "Scale micro-loop research extract failed",
                        item=item_label,
                        error=str(e),
                    )
                    self._emit_tool_output(
                        "scale_micro_loop",
                        {"item": item_label, "step": "extract", "mode": "research"},
                        f"[{item_num}/{total}] RESEARCH FAILED: {item_label}\nError: {e}",
                    )
                    errors.append({"item": item_label, "phase": "extract", "error": str(e)})
                    failed += 1
                    done_items.add(item)
                    sp["done_items"] = done_items
                    continue

            elif _extraction_mode == "inline":
                # Inline mode: user explicitly forbade external fetching.
                # Use the full source page content (stored during deferred
                # scale init) so the LLM has maximum context for extraction.
                # Fall back to the _member_context snippet if no full page
                # content is available.
                _full_source = str(sp.get("_source_page_content", "")).strip() if sp else ""
                if _full_source:
                    extracted_content = _full_source
                else:
                    _member_ctx_map_i: dict[str, str] = sp.get("_member_context", {}) if sp else {}
                    extracted_content = _member_ctx_map_i.get(item, "")
                    if not extracted_content and item_label != item:
                        extracted_content = _member_ctx_map_i.get(item_label, "")

            else:
                # Standard mode: single-tool extraction (read, web_fetch, pdf_extract, etc.)
                tool_name, tool_args = self._detect_item_extractor(item)

                # ── Google Drive file override ──
                # When the detected tool would try a local-file operation
                # (read, pdf_extract) or _passthrough, check if this item
                # exists in the Google Drive file map.  If yes, use gws
                # docs_read (for Google-native Docs/Sheets/Slides) or
                # drive_download + local extract (for uploaded files).
                if _gdrive_file_map and tool_name in (
                    "read", "pdf_extract", "docx_extract",
                    "xlsx_extract", "pptx_extract", "_passthrough",
                ):
                    _gd_info = self._lookup_gdrive_file(item, _gdrive_file_map)
                    if _gd_info:
                        _gd_id = _gd_info["id"]
                        _gd_mime = _gd_info.get("mimeType", "")
                        if _gd_mime in self._GOOGLE_NATIVE_MIMETYPES:
                            # Google Docs/Sheets/Slides → docs_read
                            tool_name = "gws"
                            tool_args = {"action": "docs_read", "file_id": _gd_id}
                            log.info(
                                "GDrive override → docs_read",
                                item=item_label[:60], file_id=_gd_id,
                            )
                        else:
                            # Uploaded files → download first, then extract
                            log.info(
                                "GDrive override → drive_download",
                                item=item_label[:60], file_id=_gd_id, mime=_gd_mime,
                            )
                            try:
                                _dl_result = await self._execute_tool_with_guard(
                                    name="gws",
                                    arguments={"action": "drive_download", "file_id": _gd_id},
                                    interaction_label=f"scale_gdrive_dl_{item_num}",
                                    turn_usage=turn_usage,
                                    session_policy=session_policy,
                                    task_policy=task_policy,
                                )
                            except Exception as _dl_err:
                                log.warning("GDrive download failed", item=item_label, error=str(_dl_err))
                                self._emit_tool_output(
                                    "scale_micro_loop",
                                    {"item": item_label, "step": "extract", "mode": "gdrive"},
                                    f"[{item_num}/{total}] GDRIVE DOWNLOAD FAILED: {item_label}\nError: {_dl_err}",
                                )
                                errors.append({"item": item_label, "phase": "extract", "error": str(_dl_err)})
                                failed += 1
                                done_items.add(item)
                                sp["done_items"] = done_items
                                continue

                            if not _dl_result.success:
                                log.warning("GDrive download error", item=item_label, error=_dl_result.error)
                                self._emit_tool_output(
                                    "scale_micro_loop",
                                    {"item": item_label, "step": "extract", "mode": "gdrive"},
                                    f"[{item_num}/{total}] GDRIVE DOWNLOAD ERROR: {item_label}\nError: {_dl_result.error}",
                                )
                                errors.append({"item": item_label, "phase": "extract", "error": _dl_result.error})
                                failed += 1
                                done_items.add(item)
                                sp["done_items"] = done_items
                                continue

                            # Parse local path from download result.
                            _dl_content = _dl_result.content or ""
                            _path_match = re.search(r'read\(path="([^"]+)"\)', _dl_content)
                            if not _path_match:
                                # Fallback: try "to /absolute/path" pattern
                                _path_match = re.search(r"to\s+(/\S+)", _dl_content)
                            if _path_match:
                                _local_path = _path_match.group(1)
                                _local_ext = os.path.splitext(_local_path)[-1].lower()
                                if _local_ext == ".pdf":
                                    tool_name = "pdf_extract"
                                    tool_args = {"path": _local_path}
                                elif _local_ext == ".docx":
                                    tool_name = "docx_extract"
                                    tool_args = {"path": _local_path}
                                elif _local_ext == ".xlsx":
                                    tool_name = "xlsx_extract"
                                    tool_args = {"path": _local_path}
                                elif _local_ext == ".pptx":
                                    tool_name = "pptx_extract"
                                    tool_args = {"path": _local_path}
                                else:
                                    tool_name = "read"
                                    tool_args = {"path": _local_path}
                            else:
                                # Could not parse downloaded path — use raw
                                # download output as extracted content.
                                extracted_content = _dl_content
                                tool_name = None

                # _passthrough: the item is a plain-text entity (not a file
                # or URL).  Use the item text itself + any member_context as
                # content — no external tool call needed.
                if tool_name == "_passthrough":
                    _member_ctx_map_pt: dict[str, str] = sp.get("_member_context", {}) if sp else {}
                    _pt_ctx = _member_ctx_map_pt.get(item, "")
                    if not _pt_ctx and item_label != item:
                        _pt_ctx = _member_ctx_map_pt.get(item_label, "")
                    extracted_content = _pt_ctx if _pt_ctx else tool_args.get("text", item)
                    # Jump past the tool-execution block below.
                    _t_extract_end = _time.monotonic()
                    _extract_sec = round(_t_extract_end - _t_extract_start, 2)
                    _extract_chars = len(extracted_content)
                    self._emit_tool_output(
                        "scale_micro_loop",
                        {"item": item_label, "step": "extract", "mode": _extraction_mode},
                        f"[{item_num}/{total}] Passthrough {_extract_chars:,} chars — {item_label}",
                    )
                    # Skip the empty-content check for passthrough — the LLM
                    # will receive the item name and any available context.
                    if not extracted_content.strip():
                        extracted_content = item
                    # Skip to Step 2 (LLM call) by jumping past extract block.
                    # We use a flag to avoid executing the tool-call block.
                    tool_name = None  # sentinel to skip tool execution below

                if tool_name is not None:
                    try:
                        extract_result = await self._execute_tool_with_guard(
                            name=tool_name,
                            arguments=tool_args,
                            interaction_label=f"scale_extract_{item_num}",
                            turn_usage=turn_usage,
                            session_policy=session_policy,
                            task_policy=task_policy,
                        )
                    except Exception as e:
                        log.warning(
                            "Scale micro-loop extract failed",
                            item=item_label,
                            error=str(e),
                        )
                        self._emit_tool_output(
                            "scale_micro_loop",
                            {"item": item_label, "step": "extract", "tool": tool_name},
                            f"[{item_num}/{total}] EXTRACT FAILED: {item_label}\nError: {e}",
                        )
                        errors.append({"item": item_label, "phase": "extract", "error": str(e)})
                        failed += 1
                        done_items.add(item)
                        sp["done_items"] = done_items
                        continue

                    if not extract_result.success:
                        log.warning(
                            "Scale micro-loop extract error",
                            item=item_label,
                            error=extract_result.error,
                        )
                        self._emit_tool_output(
                            "scale_micro_loop",
                            {"item": item_label, "step": "extract", "tool": tool_name},
                            f"[{item_num}/{total}] EXTRACT ERROR: {item_label}\nError: {extract_result.error}",
                        )
                        errors.append({"item": item_label, "phase": "extract", "error": extract_result.error})
                        failed += 1
                        done_items.add(item)
                        sp["done_items"] = done_items
                        continue

                    extracted_content = extract_result.content or ""

            _t_extract_end = _time.monotonic()
            _extract_sec = round(_t_extract_end - _t_extract_start, 2)
            _extract_chars = len(extracted_content)

            self._emit_tool_output(
                "scale_micro_loop",
                {"item": item_label, "step": "extract", "mode": _extraction_mode},
                f"[{item_num}/{total}] Extracted {_extract_chars:,} chars in {_extract_sec}s — {item_label}",
            )

            if not extracted_content.strip():
                log.warning("Scale micro-loop: empty extract", item=item_label)
                self._emit_tool_output(
                    "scale_micro_loop",
                    {"item": item_label, "step": "extract", "mode": _extraction_mode},
                    f"[{item_num}/{total}] EMPTY CONTENT: {item_label}",
                )
                errors.append({"item": item_label, "phase": "extract", "error": "empty content"})
                failed += 1
                done_items.add(item)
                sp["done_items"] = done_items
                continue

            # ── Cancellation check (after extract) ────────────
            if _check_cancel():
                log.info("Scale micro-loop cancelled after extract", item=item_label, processed=processed)
                _cancelled = True
                _consume_cancel()
                break

            # ── Step 2: Process content (LLM call or raw passthrough) ─
            _t_llm_start = _time.monotonic()

            if _processing_mode == "raw":
                # ── Raw passthrough: skip LLM, use extracted content as-is ──
                self._emit_thinking(
                    f"scale_micro_loop: Passthrough ({item_num}/{total})\n"
                    f"{item_label}\n"
                    f"extracted {_extract_chars:,} chars in {_extract_sec}s — raw mode, no LLM{_eta_suffix}",
                    tool="scale_micro_loop",
                    phase="tool",
                )
                summary_text = extracted_content
                _llm_sec = 0.0
                _item_prompt_tokens = 0
                _response_tokens = 0
                log.info(
                    "Tool-to-tool passthrough (no LLM)",
                    item_num=item_num,
                    total=total,
                    item=item_label[-80:],
                    extract_chars=_extract_chars,
                    llm_tokens=0,
                )
                self._emit_tool_output(
                    "scale_micro_loop",
                    {"item": item_label, "step": "passthrough"},
                    f"[{item_num}/{total}] Passthrough ({_extract_chars:,} chars) — {item_label}",
                )

            else:
                # ── Standard mode: isolated LLM call to process/summarize ──
                self._emit_thinking(
                    f"scale_micro_loop: Processing ({item_num}/{total})\n"
                    f"{item_label}\n"
                    f"extracted {_extract_chars:,} chars in {_extract_sec}s{_eta_suffix}",
                    tool="scale_micro_loop",
                    phase="tool",
                )

                # Look up per-item context from the source article (if available).
                _member_ctx_map: dict[str, str] = sp.get("_member_context", {}) if sp else {}
                _item_source_context = _member_ctx_map.get(item, "")
                # Also try matching by item_label (which may differ from item)
                if not _item_source_context and item_label != item:
                    _item_source_context = _member_ctx_map.get(item_label, "")

                item_messages = self._build_scale_item_prompt(
                    task_description=task_description,
                    item_label=item_label,
                    extracted_content=extracted_content,
                    is_first_item=(item_num == 1),
                    previous_summary_sample=last_summary,
                    output_strategy=_output_strategy,
                    extraction_mode=_extraction_mode,
                    source_context=_item_source_context,
                )

                # Log context size for this micro-call
                _item_prompt_tokens = sum(
                    self._count_tokens(m.content) for m in item_messages
                )

                # ── Chunked processing guard ──────────────────
                # Check whether the content overflows the context budget
                # and should be processed via the chunked pipeline instead
                # of a single LLM call.
                _content_tokens = self._count_tokens(extracted_content)
                _instruction_tokens = _item_prompt_tokens - _content_tokens
                _use_chunked = self._chunked_processing_needed(
                    instruction_tokens=_instruction_tokens,
                    content_tokens=_content_tokens,
                )

                if _use_chunked:
                    # ── Chunked pipeline path ─────────────────
                    log.info(
                        "Scale micro-loop: routing to chunked pipeline",
                        item_num=item_num,
                        total=total,
                        item=item_label[-80:],
                        extract_chars=_extract_chars,
                        instruction_tokens=_instruction_tokens,
                        content_tokens=_content_tokens,
                        prompt_tokens=_item_prompt_tokens,
                    )
                    self._emit_tool_output(
                        "scale_micro_loop",
                        {
                            "item": item_label,
                            "step": "chunked_pipeline",
                            "instruction_tokens": _instruction_tokens,
                            "content_tokens": _content_tokens,
                        },
                        (
                            f"[{item_num}/{total}] Content exceeds context budget "
                            f"({_content_tokens} content tok + {_instruction_tokens} "
                            f"instruction tok) — routing to chunked pipeline"
                        ),
                    )

                    # Decompose the prompt built by _build_scale_item_prompt
                    # into system_text (for the chunked pipeline's system message)
                    # and task_preamble + task_suffix (wrapping the content).
                    _sys_text = item_messages[0].content
                    _user_text = item_messages[1].content if len(item_messages) > 1 else ""

                    # The user message contains sections separated by
                    # "EXTRACTED CONTENT:\n===\n" and closed by "\n===\n".
                    # Split at the content markers to isolate the preamble
                    # and suffix.
                    _content_start_marker = "EXTRACTED CONTENT:\n===\n"
                    _content_end_marker = "\n===\n"
                    _cs_idx = _user_text.find(_content_start_marker)
                    if _cs_idx >= 0:
                        _task_preamble = _user_text[:_cs_idx].rstrip()
                        _after_marker = _user_text[_cs_idx + len(_content_start_marker):]
                        _ce_idx = _after_marker.rfind(_content_end_marker)
                        if _ce_idx >= 0:
                            _task_suffix = _after_marker[_ce_idx + len(_content_end_marker):].strip()
                        else:
                            _task_suffix = ""
                    else:
                        # Fallback: use the whole user text as preamble
                        _task_preamble = _user_text
                        _task_suffix = ""

                    try:
                        summary_text, _chunked_stats = await self._chunked_process_content(
                            system_text=_sys_text,
                            task_preamble=_task_preamble,
                            extracted_content=extracted_content,
                            task_suffix=_task_suffix,
                            item_label=item_label,
                            interaction_label=f"scale_process_{item_num}",
                            turn_usage=turn_usage,
                        )
                    except Exception as e:
                        _t_llm_end = _time.monotonic()
                        _llm_fail_sec = round(_t_llm_end - _t_llm_start, 2)
                        log.warning(
                            "Scale micro-loop chunked pipeline failed",
                            item=item_label,
                            error=str(e),
                            llm_sec=_llm_fail_sec,
                        )
                        self._emit_tool_output(
                            "scale_micro_loop",
                            {"item": item_label, "step": "chunked_pipeline_error"},
                            f"[{item_num}/{total}] CHUNKED PIPELINE FAILED in {_llm_fail_sec}s: {item_label}\nError: {e}",
                        )
                        errors.append({"item": item_label, "phase": "chunked_pipeline", "error": str(e)})
                        failed += 1
                        done_items.add(item)
                        sp["done_items"] = done_items
                        continue

                    _t_llm_end = _time.monotonic()
                    _llm_sec = round(_t_llm_end - _t_llm_start, 2)
                    summary_text = (summary_text or "").strip()
                    _response_tokens = self._count_tokens(summary_text)

                    # Override prompt/response tokens from chunked stats
                    _item_prompt_tokens = _chunked_stats.get("total_prompt_tokens", _item_prompt_tokens)
                    _chunked_info = (
                        f" [chunked: {_chunked_stats.get('num_chunks', '?')} chunks, "
                        f"strategy={_chunked_stats.get('combine_strategy', '?')}]"
                    )

                    self._emit_tool_output(
                        "scale_micro_loop",
                        {
                            "item": item_label,
                            "step": "llm",
                            "prompt_tokens": _item_prompt_tokens,
                            "chunked": True,
                            "num_chunks": _chunked_stats.get("num_chunks"),
                        },
                        f"[{item_num}/{total}] LLM done in {_llm_sec}s — {_item_prompt_tokens} prompt / {_response_tokens} response tokens — {item_label}{_chunked_info}",
                    )

                else:
                    # ── Normal single-call path (unchanged) ───
                    log.info(
                        "Scale micro-loop LLM call",
                        item_num=item_num,
                        total=total,
                        item=item_label[-80:],
                        extract_chars=_extract_chars,
                        extract_sec=_extract_sec,
                        prompt_tokens=_item_prompt_tokens,
                        prompt_kb=round(_item_prompt_tokens * 4 / 1024, 1),
                    )

                    try:
                        response = await self._complete_with_guards(
                            messages=item_messages,
                            tools=None,  # No tools — just produce text
                            interaction_label=f"scale_process_{item_num}",
                            turn_usage=turn_usage,
                        )
                    except Exception as e:
                        _t_llm_end = _time.monotonic()
                        _llm_fail_sec = round(_t_llm_end - _t_llm_start, 2)
                        log.warning(
                            "Scale micro-loop LLM call failed",
                            item=item_label,
                            error=str(e),
                            llm_sec=_llm_fail_sec,
                        )
                        self._emit_tool_output(
                            "scale_micro_loop",
                            {"item": item_label, "step": "llm"},
                            f"[{item_num}/{total}] LLM FAILED in {_llm_fail_sec}s: {item_label}\nError: {e}",
                        )
                        errors.append({"item": item_label, "phase": "llm", "error": str(e)})
                        failed += 1
                        done_items.add(item)
                        sp["done_items"] = done_items
                        continue

                    _t_llm_end = _time.monotonic()
                    _llm_sec = round(_t_llm_end - _t_llm_start, 2)

                    summary_text = (response.content or "").strip()
                    _response_tokens = self._count_tokens(summary_text)

                    self._emit_tool_output(
                        "scale_micro_loop",
                        {"item": item_label, "step": "llm", "prompt_tokens": _item_prompt_tokens},
                        f"[{item_num}/{total}] LLM done in {_llm_sec}s — {_item_prompt_tokens} prompt / {_response_tokens} response tokens — {item_label}",
                    )

                if not summary_text:
                    log.warning("Scale micro-loop: empty LLM response", item=item_label)
                    self._emit_tool_output(
                        "scale_micro_loop",
                        {"item": item_label, "step": "llm"},
                        f"[{item_num}/{total}] EMPTY LLM RESPONSE: {item_label}",
                    )
                    errors.append({"item": item_label, "phase": "llm", "error": "empty response"})
                    failed += 1
                    done_items.add(item)
                    sp["done_items"] = done_items
                    continue

            # ── Cancellation check (after LLM) ───────────────
            if _check_cancel():
                log.info("Scale micro-loop cancelled after LLM", item=item_label, processed=processed)
                _cancelled = True
                _consume_cancel()
                break

            # ── Step 3: Write summary directly ──────────────────
            self._emit_thinking(
                f"scale_micro_loop: Writing ({item_num}/{total})\n"
                f"{item_label}\n"
                f"extract={_extract_sec}s ({_extract_chars:,} chars) | llm={_llm_sec}s ({_item_prompt_tokens}\u2192{_response_tokens} tok)",
                tool="scale_micro_loop",
                phase="tool",
            )

            # Clean up LLM output: strip leading/trailing --- separators
            # to avoid accumulating multiple --- between items.
            _clean = summary_text.strip()
            while _clean.startswith("---"):
                _clean = _clean[3:].lstrip("\n")
            while _clean.endswith("---"):
                _clean = _clean[:-3].rstrip("\n")
            _clean = _clean.strip()

            # Determine write path and mode based on output strategy
            _t_write_start = _time.monotonic()

            if _is_no_file:
                # ── no_file: route to sink (typesense, email, reply) ──
                _sink_label = "sink"
                _sink_ok = False
                try:
                    if _final_action == "api_call":
                        _sink_label = "typesense"
                        sink_result = await self._execute_tool_with_guard(
                            name="typesense",
                            arguments={
                                "action": "index",
                                "text": _clean,
                                "source": "scale_loop",
                                "reference": item_label,
                            },
                            interaction_label=f"scale_sink_{item_num}",
                            turn_usage=turn_usage,
                            session_policy=session_policy,
                            task_policy=task_policy,
                        )
                        _sink_ok = sink_result.success
                        if not _sink_ok:
                            raise RuntimeError(sink_result.error or "Typesense index failed")

                    elif _final_action == "email":
                        _sink_label = "email"
                        _sink_email_to = str(sp.get("_sink_email_to", "")).strip()
                        if _sink_email_to:
                            sink_result = await self._execute_tool_with_guard(
                                name="send_mail",
                                arguments={
                                    "to": _sink_email_to,
                                    "subject": f"Processed: {item_label}",
                                    "body": _clean,
                                },
                                interaction_label=f"scale_sink_{item_num}",
                                turn_usage=turn_usage,
                                session_policy=session_policy,
                                task_policy=task_policy,
                            )
                            _sink_ok = sink_result.success
                            if not _sink_ok:
                                raise RuntimeError(sink_result.error or "Send mail failed")
                        else:
                            # No email target — fall through to reply accumulation.
                            _sink_label = "reply"
                            _sink_ok = True

                    else:
                        # "reply" — just accumulate; no external write needed.
                        _sink_label = "reply"
                        _sink_ok = True

                except Exception as e:
                    _t_write_end = _time.monotonic()
                    _write_fail_sec = round(_t_write_end - _t_write_start, 2)
                    log.warning(
                        "Scale micro-loop sink failed",
                        item=item_label,
                        sink=_sink_label,
                        error=str(e),
                        write_sec=_write_fail_sec,
                    )
                    self._emit_tool_output(
                        "scale_micro_loop",
                        {"item": item_label, "step": "sink", "sink": _sink_label},
                        f"[{item_num}/{total}] SINK FAILED in {_write_fail_sec}s: {item_label}\nError: {e}",
                    )
                    errors.append({"item": item_label, "phase": "sink", "error": str(e)})
                    failed += 1
                    done_items.add(item)
                    sp["done_items"] = done_items
                    continue

                _t_write_end = _time.monotonic()
                _write_sec = round(_t_write_end - _t_write_start, 2)
                write_path_arg = f"[{_sink_label}]"

            else:
                # ── file-based output strategies ──
                if _is_file_per_item:
                    # Per-item file: write to a unique file per item
                    write_path_arg = self._resolve_per_item_filename(
                        item, _filename_template, _output_dir,
                    )
                    write_content = _clean
                    write_append = False
                else:
                    # Single file: append with separator.
                    # Skip the separator for the first item.
                    if processed == 0 and not last_summary:
                        write_content = _clean
                    elif last_summary and "|" in last_summary:
                        # Table format: rows are separated by newline,
                        # not horizontal rules.
                        write_content = _clean if _clean.startswith("|") else f"| {_clean}"
                    else:
                        write_content = f"\n\n---\n\n{_clean}"
                    # Prefer the already-resolved absolute path (_output_file)
                    # over the original relative argument (_output_file_arg).
                    # Using the relative arg and then calling
                    # _resolve_scale_output_path would double-resolve it:
                    #   "saved/showcase/{id}/file.md" → "output/{id}/saved/showcase/{id}/file.md"
                    # The absolute path is correct as-is and the write tool
                    # passes absolute paths through without remapping.
                    _known_abs = sp.get("_output_file", "")
                    if _known_abs and os.path.isabs(_known_abs):
                        write_path_arg = _known_abs
                    else:
                        write_path_arg = sp.get("_output_file_arg", output_file)
                        # Fallback chain when no output file is known yet
                        # (early takeover before the LLM wrote anything):
                        # 1. Filename template from scale progress (file_per_item)
                        # 2. output_file from plan (LLM-derived for single_file)
                        # 3. Extract filename from task description regex
                        # 4. Derive from worker task title (unique per worker)
                        # 5. Last resort: "scale_output.md"
                        if not write_path_arg:
                            write_path_arg = (
                                str(sp.get("_output_filename_template", "")).strip()
                                or str(sp.get("_output_file_from_plan", "")).strip()
                                or self._extract_output_filename(task_description)
                                or self._derive_output_filename_from_context(task_description)
                                or "scale_output.md"
                            )
                        # Resolve to absolute path under <workspace>/output/<session>/
                        # only when we don't have a known absolute path.
                        write_path_arg = self._resolve_scale_output_path(write_path_arg)
                    write_append = True

                try:
                    write_result = await self._execute_tool_with_guard(
                        name="write",
                        arguments={
                            "path": write_path_arg,
                            "content": write_content,
                            "append": write_append,
                        },
                        interaction_label=f"scale_write_{item_num}",
                        turn_usage=turn_usage,
                        session_policy=session_policy,
                        task_policy=task_policy,
                    )
                except Exception as e:
                    _t_write_end = _time.monotonic()
                    _write_fail_sec = round(_t_write_end - _t_write_start, 2)
                    log.warning(
                        "Scale micro-loop write failed",
                        item=item_label,
                        error=str(e),
                        write_sec=_write_fail_sec,
                    )
                    self._emit_tool_output(
                        "scale_micro_loop",
                        {"item": item_label, "step": "write", "path": write_path_arg},
                        f"[{item_num}/{total}] WRITE FAILED in {_write_fail_sec}s: {item_label}\nError: {e}",
                    )
                    errors.append({"item": item_label, "phase": "write", "error": str(e)})
                    failed += 1
                    done_items.add(item)
                    sp["done_items"] = done_items
                    continue

                _t_write_end = _time.monotonic()
                _write_sec = round(_t_write_end - _t_write_start, 2)

                # Capture the resolved output path for the summary.
                if write_result.success and not sp.get("_output_file"):
                    # Parse the resolved path from the write result content,
                    # e.g. "Wrote 123 bytes to /abs/path (requested: rel)"
                    _wr_content = str(write_result.content or "")
                    _to_idx = _wr_content.find(" to ")
                    if _to_idx > 0:
                        _after_to = _wr_content[_to_idx + 4:].strip()
                        _resolved = _after_to.split(" (requested:")[0].strip()
                        if _resolved:
                            sp["_output_file"] = _resolved
                    if not sp.get("_output_file"):
                        sp["_output_file"] = os.path.abspath(write_path_arg)
                    sp["_output_file_arg"] = write_path_arg

                if not write_result.success:
                    log.warning(
                        "Scale micro-loop write error",
                        item=item_label,
                        error=write_result.error,
                        write_sec=_write_sec,
                    )
                    self._emit_tool_output(
                        "scale_micro_loop",
                        {"item": item_label, "step": "write", "path": write_path_arg},
                        f"[{item_num}/{total}] WRITE ERROR in {_write_sec}s: {item_label}\nError: {write_result.error}",
                    )
                    errors.append({"item": item_label, "phase": "write", "error": write_result.error})
                    failed += 1
                    done_items.add(item)
                    sp["done_items"] = done_items
                    continue

            # ── Step 4: Update tracking & timing ─────────────────
            done_items.add(item)
            sp["done_items"] = done_items
            sp["completed"] = len(done_items)
            processed += 1

            # Signal progress to the orchestrator so it can auto-postpone
            # timeout warnings for workers with active scale loops.
            self._scale_last_progress_at = _time.monotonic()
            # Store a clean version for FORMAT REFERENCE — strip any
            # leaked extracted content echoes.
            _last = _clean
            for _m in ("EXTRACTED CONTENT:", "===\n#"):
                _c = _last.find(_m)
                if _c > 0:
                    _last = _last[:_c].rstrip()
            last_summary = _last

            _item_total_sec = round(_time.monotonic() - _item_start, 2)
            _item_durations.append(_item_total_sec)
            _total_prompt_tokens += _item_prompt_tokens
            _total_response_tokens += _response_tokens
            _total_extract_chars += _extract_chars

            _avg_item_sec = round(sum(_item_durations) / len(_item_durations), 2)

            pct = int(len(done_items) / total * 100)

            # ── Monitor: per-item summary ─────────────────────
            self._emit_tool_output(
                "scale_micro_loop",
                {"item": item_label, "step": "done", "item_num": item_num, "total": total},
                (
                    f"[{item_num}/{total}] \u2713 {item_label}\n"
                    f"  extract={_extract_sec}s ({_extract_chars:,} chars) | "
                    f"llm={_llm_sec}s ({_item_prompt_tokens}\u2192{_response_tokens} tok) | "
                    f"write={_write_sec}s | total={_item_total_sec}s | avg={_avg_item_sec}s"
                ),
            )

            self._emit_thinking(
                f"scale_micro_loop: \u2713 {pct}% ({len(done_items)}/{total})\n"
                f"{item_label}\n"
                f"extract={_extract_sec}s | llm={_llm_sec}s | write={_write_sec}s | total={_item_total_sec}s{_eta_suffix}",
                tool="scale_micro_loop",
                phase="tool",
            )

            log.info(
                "Scale micro-loop item done",
                item_num=item_num,
                total=total,
                pct=pct,
                item=item_label[-80:],
                extract_sec=_extract_sec,
                extract_chars=_extract_chars,
                llm_sec=_llm_sec,
                write_sec=_write_sec,
                total_sec=_item_total_sec,
                avg_sec=_avg_item_sec,
                prompt_tokens=_item_prompt_tokens,
                response_tokens=_response_tokens,
            )

        # ── Loop-level summary ────────────────────────────────────
        _loop_total_sec = round(_time.monotonic() - _loop_start, 2)
        _loop_avg_sec = (
            round(sum(_item_durations) / len(_item_durations), 2)
            if _item_durations
            else 0
        )
        _items_per_min = (
            round(len(_item_durations) / (_loop_total_sec / 60), 1)
            if _loop_total_sec > 0
            else 0
        )

        result = {
            "success": failed == 0 and not _cancelled,
            "processed": processed,
            "failed": failed,
            "total": total,
            "completed_total": len(done_items),
            "cancelled": _cancelled,
            "errors": errors,
        }
        log.info(
            "Scale micro-loop finished",
            **{k: v for k, v in result.items() if k != "errors"},
            error_count=len(errors),
            loop_total_sec=_loop_total_sec,
            avg_per_item_sec=_loop_avg_sec,
            items_per_min=_items_per_min,
            total_prompt_tokens=_total_prompt_tokens,
            total_response_tokens=_total_response_tokens,
            total_extract_chars=_total_extract_chars,
        )

        _loop_min = int(_loop_total_sec // 60)
        _loop_s = int(_loop_total_sec % 60)
        _status_label = "stopped by user" if _cancelled else "finished"
        self._emit_tool_output(
            "scale_micro_loop",
            {"step": _status_label, "processed": processed, "failed": failed, "total": total, "cancelled": _cancelled},
            (
                f"Scale micro-loop {_status_label}: {processed}/{total} processed, {failed} failed\n"
                f"  total={_loop_min}m{_loop_s:02d}s | avg={_loop_avg_sec}s/item | {_items_per_min} items/min\n"
                f"  tokens: {_total_prompt_tokens:,} prompt + {_total_response_tokens:,} response\n"
                f"  extracted: {_total_extract_chars:,} chars total"
            ),
        )

        return result

    # ------------------------------------------------------------------
    # Scale loop readiness & status
    # ------------------------------------------------------------------

    def _scale_loop_ready(self) -> bool:
        """Check whether the micro-turn scale loop can take over.

        The loop is ready when:
        1. Scale progress is active with items populated
        2. The output file path is known (first write has happened)
        3. At least the first item has been processed by the LLM
           (so we know the format/structure)
        4. There are remaining items to process
        5. Output strategy is compatible:
           - single_file: must be writing to a SINGLE file (original behavior)
           - file_per_item: always compatible (micro-loop creates per-item files)
           - no_file: compatible when sink is available (typesense, email, reply)
        """
        sp = getattr(self, "_scale_progress", None)
        if sp is None:
            return False
        items = sp.get("items", [])
        if len(items) < 2:
            # Need at least 2 items for a meaningful scale loop.
            # A single item likely means the extractor only found the
            # source URL, not the actual list members.
            return False
        # Do NOT take over when all items are just the source URL repeated.
        # This means the real list members haven't been extracted yet.
        if self._items_are_source_urls_only(items):
            return False
        # Do NOT take over when extraction mode is "passthrough" — the
        # user wants to save/store items (e.g. create a datastore table)
        # and the main LLM should handle it directly with tool calls.
        if str(sp.get("_extraction_mode", "")).strip() == "passthrough":
            return False

        output_strategy = str(sp.get("_output_strategy", "single_file")).strip().lower()

        # no_file output strategy — the micro-loop can take over when the
        # final action has a compatible sink (typesense, email, or reply).
        if output_strategy == "no_file":
            final_action = str(sp.get("_final_action", "reply")).strip()
            if final_action == "api_call":
                # Allow if typesense tool is registered.
                if not self.tools.has_tool("typesense"):
                    return False
            elif final_action not in ("reply", "email"):
                return False

        if output_strategy == "file_per_item":
            # For file-per-item: we need the filename template and at least
            # one item done so we know the format.
            filename_template = str(sp.get("_output_filename_template", "")).strip()
            if not filename_template:
                # No template — need at least one write to infer the pattern.
                # Check if the LLM has written at least one per-item file.
                output_files: set[str] = sp.get("_output_files", set())
                if not output_files:
                    return False
        elif output_strategy == "no_file":
            # no_file mode — no output file needed, sink was validated above.
            pass
        else:
            # single_file mode (original behavior)
            output_file = sp.get("_output_file", "") or sp.get("_output_file_arg", "")
            if not output_file:
                return False
            # If the LLM is creating separate files per item but we expected
            # single_file, the micro-loop cannot take over.
            output_files = sp.get("_output_files", set())
            if len(output_files) > 1:
                return False

        # At least 1 item done — the LLM has established the format.
        # Use both done_items (path-matched) and completed (counter)
        # as indicators since done_items matching can be fuzzy.
        done_count = max(
            len(sp.get("done_items", set())),
            sp.get("completed", 0),
        )
        if done_count < 1:
            return False
        # Must have remaining items to process
        remaining = len(items) - done_count
        return remaining > 0

    def _get_scale_task_description(self) -> str:
        """Extract the per-item task description from the scale progress
        or list_task_plan context.

        Falls back to a generic description if nothing specific is available.
        """
        # Try list_task_plan per_member_action first — it's the most
        # focused description of what to do per item.
        sp = getattr(self, "_scale_progress", None)
        if sp is not None:
            action = sp.get("_per_member_action", "")
            if action:
                return action
        return "Summarize or process this item and produce a structured result."
