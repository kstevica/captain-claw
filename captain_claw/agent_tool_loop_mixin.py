"""Tool-call parsing/execution helpers for Agent."""

import asyncio
import json
import os
import re
from typing import Any

from captain_claw.config import get_config
from captain_claw.llm import Message, ToolCall
from captain_claw.logging import get_logger


log = get_logger(__name__)


class AgentToolLoopMixin:
    """Extract commands, parse embedded tool calls, and execute tool loops."""

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

    # Maximum items to show in the progress note.  For very large lists
    # (50–200+ items) we only show the next handful so the note stays
    # compact and doesn't eat up the context budget on every iteration.
    _SCALE_NOTE_MAX_VISIBLE = 15

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

        return None

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
        done_items: set[str] = sp.get("done_items", set())
        total = len(items)
        done = len(done_items)
        remaining = [item for item in items if item not in done_items]

        # Detect item type to tailor the instructions.
        has_paths = any("/" in item and not item.startswith("http") for item in items[:5])
        has_urls = any(item.startswith(("http://", "https://")) for item in items[:5])

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

        if has_paths:
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
        # Default: plain file read
        return "read", {"path": item}

    def _build_scale_item_prompt(
        self,
        task_description: str,
        item_label: str,
        extracted_content: str,
        is_first_item: bool = False,
        previous_summary_sample: str = "",
        output_strategy: str = "single_file",
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
            f"{output_instruction}"
        )

        user_parts = [f"TASK: {task_description}\n"]
        user_parts.append(f"ITEM: {item_label}\n")

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

        # Determine output strategy
        _output_strategy = str(sp.get("_output_strategy", "single_file")).strip().lower()
        _filename_template = str(sp.get("_output_filename_template", "")).strip()
        _is_file_per_item = _output_strategy == "file_per_item" and bool(_filename_template)
        _is_no_file = _output_strategy == "no_file"
        _final_action = str(sp.get("_final_action", "reply")).strip()
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
            is_file_per_item=_is_file_per_item,
            filename_template=_filename_template,
        )

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

        for idx, item in enumerate(remaining):
            # ── Cancellation check ──────────────────────────────
            cancel_ev: asyncio.Event | None = getattr(self, "cancel_event", None)
            if cancel_ev is not None and cancel_ev.is_set():
                log.info("Scale micro-loop cancelled by user", processed=processed)
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

            # ── Step 1: Extract content directly ────────────────
            _t_extract_start = _time.monotonic()
            tool_name, tool_args = self._detect_item_extractor(item)
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
                {"item": item_label, "step": "extract", "tool": tool_name},
                f"[{item_num}/{total}] Extracted {_extract_chars:,} chars in {_extract_sec}s — {item_label}",
            )

            if not extracted_content.strip():
                log.warning("Scale micro-loop: empty extract", item=item_label)
                self._emit_tool_output(
                    "scale_micro_loop",
                    {"item": item_label, "step": "extract", "tool": tool_name},
                    f"[{item_num}/{total}] EMPTY CONTENT: {item_label}",
                )
                errors.append({"item": item_label, "phase": "extract", "error": "empty content"})
                failed += 1
                done_items.add(item)
                sp["done_items"] = done_items
                continue

            # ── Step 2: Isolated LLM call to process ────────────
            _t_llm_start = _time.monotonic()
            self._emit_thinking(
                f"scale_micro_loop: Processing ({item_num}/{total})\n"
                f"{item_label}\n"
                f"extracted {_extract_chars:,} chars in {_extract_sec}s{_eta_suffix}",
                tool="scale_micro_loop",
                phase="tool",
            )

            item_messages = self._build_scale_item_prompt(
                task_description=task_description,
                item_label=item_label,
                extracted_content=extracted_content,
                is_first_item=(item_num == 1),
                previous_summary_sample=last_summary,
                output_strategy=_output_strategy,
            )

            # Log context size for this micro-call
            _item_prompt_tokens = sum(
                self._count_tokens(m.content) for m in item_messages
            )
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
                    # Single file: append with separator
                    write_content = f"\n\n---\n\n{_clean}"
                    # Use the original write argument (before the write tool
                    # resolves it under the saved root) so the tool doesn't
                    # double-resolve an already-absolute path.
                    write_path_arg = sp.get("_output_file_arg", output_file)
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
                    f"[{item_num}/{total}] ✓ {item_label}\n"
                    f"  extract={_extract_sec}s ({_extract_chars:,} chars) | "
                    f"llm={_llm_sec}s ({_item_prompt_tokens}→{_response_tokens} tok) | "
                    f"write={_write_sec}s | total={_item_total_sec}s | avg={_avg_item_sec}s"
                ),
            )

            self._emit_thinking(
                f"scale_micro_loop: ✓ {pct}% ({len(done_items)}/{total})\n"
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
            "success": failed == 0,
            "processed": processed,
            "failed": failed,
            "total": total,
            "completed_total": len(done_items),
            "errors": errors,
        }
        log.info(
            "Scale micro-loop finished",
            **result,
            loop_total_sec=_loop_total_sec,
            avg_per_item_sec=_loop_avg_sec,
            items_per_min=_items_per_min,
            total_prompt_tokens=_total_prompt_tokens,
            total_response_tokens=_total_response_tokens,
            total_extract_chars=_total_extract_chars,
        )

        _loop_min = int(_loop_total_sec // 60)
        _loop_s = int(_loop_total_sec % 60)
        self._emit_tool_output(
            "scale_micro_loop",
            {"step": "finished", "processed": processed, "failed": failed, "total": total},
            (
                f"Scale micro-loop finished: {processed}/{total} processed, {failed} failed\n"
                f"  total={_loop_min}m{_loop_s:02d}s | avg={_loop_avg_sec}s/item | {_items_per_min} items/min\n"
                f"  tokens: {_total_prompt_tokens:,} prompt + {_total_response_tokens:,} response\n"
                f"  extracted: {_total_extract_chars:,} chars total"
            ),
        )

        return result

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
        if not items:
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

    @staticmethod
    def _tool_thinking_summary(tool_name: str, arguments: dict[str, Any]) -> str:
        """Derive a short human-readable summary of a tool call for the thinking indicator."""
        name = str(tool_name or "").strip().lower()
        # ── Core tools ──────────────────────────────────────
        if name == "read":
            path = str(arguments.get("path", arguments.get("file_path", ""))).strip()
            return f"Reading: {path.rsplit('/', 1)[-1]}" if path else "Reading file"
        if name == "write":
            path = str(arguments.get("path", arguments.get("file_path", ""))).strip()
            return f"Writing: {path.rsplit('/', 1)[-1]}" if path else "Writing file"
        if name == "shell":
            cmd = str(arguments.get("command", "")).strip()
            return f"Running: {cmd[:60]}" if cmd else "Running shell command"
        if name == "web_fetch":
            url = str(arguments.get("url", "")).strip()
            return f"Fetching: {url[:60]}" if url else "Fetching URL"
        if name == "web_get":
            url = str(arguments.get("url", "")).strip()
            return f"Fetching HTML: {url[:60]}" if url else "Fetching raw HTML"
        if name == "web_search":
            query = str(arguments.get("query", "")).strip()
            return f"Searching: {query[:60]}" if query else "Searching the web"
        if name == "send_mail":
            to = str(arguments.get("to", "")).strip()
            return f"Sending email to {to}" if to else "Sending email"
        if name in {"todos", "contacts", "scripts", "apis"}:
            action = str(arguments.get("action", "list")).strip()
            return f"{action.capitalize()}: {name} memory"
        # ── Document extractors ───────────────────────────────
        if name in {"pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract"}:
            path = str(arguments.get("path", "")).strip()
            label = name.replace("_extract", "").upper()
            return f"Extracting {label}: {path.rsplit('/', 1)[-1]}" if path else f"Extracting {label}"
        # ── Google Drive ─────────────────────────────────────
        if name == "google_drive":
            action = str(arguments.get("action", "")).strip()
            return f"Google Drive: {action}" if action else "Google Drive"
        # ── Memory & context ────────────────────────────────
        if name == "memory_select":
            query = str(arguments.get("query", "")).strip()
            return f"Selecting memory context: {query[:50]}" if query else "Selecting memory context"
        if name == "memory_semantic_select":
            query = str(arguments.get("query", "")).strip()
            return f"Semantic memory search: {query[:50]}" if query else "Semantic memory search"
        # ── Pipeline & planning ─────────────────────────────
        if name == "task_contract":
            step = str(arguments.get("step", "")).strip()
            return f"Task planner: {step}" if step else "Generating task contract"
        if name == "completion_gate":
            step = str(arguments.get("step", "")).strip()
            return f"Completion check: {step}" if step else "Evaluating completion"
        if name == "planning":
            event = str(arguments.get("event", "")).strip()
            # Show the current task path (human-readable) instead of the
            # raw pipeline event name like "tool_calls_completed".
            current_path = str(arguments.get("current_path", "")).strip()
            if current_path:
                label = current_path.split(" > ")[-1][:60]
                if event and "completed" in event:
                    return f"✓ {label}"
                return f"▸ {label}"
            return f"Planning: {event}" if event else "Updating plan"
        if name == "pipeline_trace":
            return "Pipeline trace"
        # ── Guards ──────────────────────────────────────────
        if name.startswith("guard_"):
            guard_type = name[6:]
            decision = str(arguments.get("decision", "")).strip()
            return f"Guard ({guard_type}): {decision}" if decision else f"Checking guard: {guard_type}"
        # ── Session management ──────────────────────────────
        if name == "compaction":
            trigger = str(arguments.get("trigger", "")).strip()
            return f"Compacting messages: {trigger}" if trigger else "Compacting session messages"
        if name == "session_procreate":
            step = str(arguments.get("step", "")).strip()
            return f"Session procreation: {step}" if step else "Procreating session"
        # ── LLM tracing ────────────────────────────────────
        if name == "llm_trace":
            return "LLM trace"
        # ── Media ──────────────────────────────────────────
        if name == "pocket_tts":
            return "Text-to-speech"
        # ── Approval ───────────────────────────────────────
        if name == "approval":
            return "Auto-approved action"
        # ── Fallback ───────────────────────────────────────
        return f"Running: {tool_name}"

    def _extract_command_from_response(self, content: str) -> str | None:
        """Extract shell command from model response.
        
        Looks for:
        - ```bash\ncommand\n``` or ```shell\ncommand\n```
        - "I'll run: command"
        - "Running: command"
        """
        import re
        
        if not content:
            return None
        
        # Only match explicit shell/code blocks or explicit commands
        patterns = [
            r'```(?:bash|shell|sh)\s*\n(.*?)\n```',  # ```bash\ncommand\n```
            r'```\s*\n(.*?)\n```(?:\s|$)',  # ```\ncommand\n``` followed by whitespace or end
            r"I'(?:ll| will) run[:\s]+[`\"]?(.+?)[`\"]?(?:\n|$)",  # I'll run: `command`
            r"(?:exec|execute)[:\s]+[`\"](.+?)[`\"]",  # exec: `command`
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                cmd = match.group(1).strip()
                # Must look like a shell command (has spaces or special chars)
                if cmd and len(cmd) > 2 and (' ' in cmd or '|' in cmd or '&' in cmd or '/' in cmd):
                    return cmd
        
        return None

    def _supports_tool_result_followup(self) -> bool:
        """Whether model can handle a follow-up turn with tool result messages."""
        details = self.get_runtime_model_details()
        provider = str(details.get("provider", "")).lower()
        model = str(details.get("model", "")).lower()

        # Known issue: Ollama cloud models often return HTTP 500 when tool role
        # messages are included in the follow-up request.
        if provider == "ollama" and model.endswith(":cloud"):
            return False

        return True

    def _collect_turn_tool_output(self, turn_start_idx: int) -> str:
        """Collect tool outputs for the current turn."""
        if not self.session:
            return ""
        outputs: list[str] = []
        for msg in self.session.messages[turn_start_idx:]:
            if msg.get("role") != "tool":
                continue
            if self._is_monitor_only_tool_name(str(msg.get("tool_name", ""))):
                continue
            content = str(msg.get("content", "")).strip()
            if content:
                outputs.append(content)
        return "\n\n".join(outputs)

    def _turn_has_successful_tool(self, turn_start_idx: int, tool_name: str) -> bool:
        """Check whether a successful tool result exists in current turn."""
        if not self.session:
            return False
        target = (tool_name or "").strip().lower()
        for msg in self.session.messages[turn_start_idx:]:
            if msg.get("role") != "tool":
                continue
            if str(msg.get("tool_name", "")).strip().lower() != target:
                continue
            content = str(msg.get("content", "")).strip().lower()
            if not content.startswith("error:"):
                return True
        return False

    @staticmethod
    def _clip_tool_output_for_rewrite(raw: str, max_chars: int) -> tuple[str, str]:
        """Clip tool output while preserving coverage across multiple blocks/sources."""
        text = (raw or "").strip()
        if len(text) <= max_chars:
            return text, ""

        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        if len(blocks) <= 1:
            return text[:max_chars], "\n\n[Tool output truncated before rewrite due to size limits.]"

        target_blocks = min(len(blocks), 24)
        per_block = max(350, max_chars // target_blocks)
        kept: list[str] = []
        used = 0
        for block in blocks:
            snippet = block
            if len(snippet) > per_block:
                snippet = snippet[:per_block].rstrip() + "... [truncated]"
            projected = used + len(snippet) + (2 if kept else 0)
            if projected > max_chars:
                break
            kept.append(snippet)
            used = projected
        if not kept:
            kept = [text[:max_chars]]
        clipped = "\n\n".join(kept)
        return clipped, "\n\n[Tool output truncated before rewrite due to size limits.]"

    async def _friendly_tool_output_response(
        self,
        user_input: str,
        tool_output: str,
        turn_usage: dict[str, int],
    ) -> str:
        """Ask model to rewrite raw tool output into a user-friendly answer."""
        raw = (tool_output or "").strip() or "[no output]"
        max_tool_output_chars = 45000
        clipped, clipped_note = self._clip_tool_output_for_rewrite(
            raw,
            max_chars=max_tool_output_chars,
        )

        rewrite_messages = [
            Message(
                role="system",
                content=self.instructions.load("tool_output_rewrite_system_prompt.md"),
            ),
            Message(
                role="user",
                content=self.instructions.render(
                    "tool_output_rewrite_user_prompt.md",
                    user_input=user_input,
                    tool_output=clipped,
                    clipped_note=clipped_note,
                ),
            ),
        ]

        self._set_runtime_status("thinking")
        try:
            response = await self._complete_with_guards(
                messages=rewrite_messages,
                tools=None,
                interaction_label="tool_output_rewrite",
                turn_usage=turn_usage,
            )
            friendly = (response.content or "").strip()
            if friendly:
                return friendly
        except Exception as e:
            log.warning("Friendly tool rewrite failed", error=str(e))

        return f"Tool executed:\n{raw}"

    def _extract_tool_calls_from_content(self, content: str) -> list[ToolCall]:
        """Extract tool calls from response content text.
        
        Looks for various formats:
        - @shell\ncommand: value
        - {tool => "shell", args => { --command "ls -la" }}
        - ```tool\ncommand\n```
        - <invoke name="shell"><command>value</command></invoke>
        """
        import re
        
        tool_calls: list[ToolCall] = []
        if not content:
            return tool_calls

        max_calls = 8
        seen: set[str] = set()

        def _append_tool_call(name: str, arguments: dict[str, Any]) -> None:
            if len(tool_calls) >= max_calls:
                return
            normalized_name = str(name or "").strip().lower()
            if not normalized_name or not isinstance(arguments, dict):
                return
            signature = json.dumps(
                {"name": normalized_name, "arguments": arguments},
                ensure_ascii=True,
                sort_keys=True,
            )
            if signature in seen:
                return
            seen.add(signature)
            tool_calls.append(
                ToolCall(
                    id=f"embedded_{len(tool_calls)}",
                    name=normalized_name,
                    arguments=arguments,
                )
            )
        
        # Pattern 1: @tool\ncommand: value
        pattern1 = r'@(\w+)\s*\n\s*command:\s*(.+?)(?:\n\n|\n\*|$)'
        
        # Pattern 2: {tool => "name", args => { --key "value" }}
        pattern2 = r'\{tool\s*=>\s*"([^"]+)"[^}]*args\s*=>\s*\{([^}]+)\}\}'
        
        # Pattern 3: ```tool\ncommand\n```
        pattern3 = r'```(\w+)\s*\n(.*?)\n```'
        
        # Pattern 4: <invoke name="shell"><command>value</command></invoke>
        pattern4 = r'<invoke\s+name="(\w+)">\s*<command>(.+?)</command>\s*</invoke>'
        
        all_patterns = [pattern1, pattern2, pattern3, pattern4]
        
        for pattern in all_patterns:
            for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
                if pattern == pattern4:
                    # Pattern 4: <invoke name="..."><command>...</command></invoke>
                    tool_name = match.group(1).strip().lower()
                    command = match.group(2).strip()
                    if tool_name and command:
                        _append_tool_call(tool_name, {"command": command})
                elif pattern == pattern1:
                    tool_name = match.group(1).strip().lower()
                    command = match.group(2).strip()
                    if tool_name and command:
                        _append_tool_call(tool_name, {"command": command})
                elif pattern == pattern2:
                    tool_name = match.group(1).strip()
                    args_str = match.group(2).strip()
                    args = {}
                    arg_pattern = r'--(\w+)\s+"([^"]+)"'
                    for arg_match in re.finditer(arg_pattern, args_str):
                        key = arg_match.group(1)
                        value = arg_match.group(2)
                        args[key] = value
                    if tool_name and args:
                        _append_tool_call(tool_name, args)
                elif pattern == pattern3:
                    tool_name = match.group(1).strip().lower()
                    command = match.group(2).strip()
                    if tool_name and command:
                        _append_tool_call(tool_name, {"command": command})

        # Pattern 5: JSON tool calls and pseudo-tool argument objects.
        # Supports:
        # - {"tool":"web_search","args":{"query":"..."}}
        # - {"name":"web_fetch","arguments":{"url":"https://..."}}
        # - {"tool":"web_search","input":"..."}
        # - {"query":"..."}  -> web_search
        # - {"url":"https://..."} -> web_fetch
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line.startswith("{") or not line.endswith("}"):
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            explicit_tool = str(payload.get("tool", payload.get("name", "")) or "").strip().lower()
            explicit_args = payload.get("args", payload.get("arguments"))
            if explicit_tool and isinstance(explicit_args, dict):
                args_obj = dict(explicit_args)
                if explicit_tool == "web_search" and "max_results" in args_obj and "count" not in args_obj:
                    try:
                        args_obj["count"] = int(args_obj.get("max_results"))
                    except Exception:
                        pass
                _append_tool_call(explicit_tool, args_obj)
                continue
            if explicit_tool:
                args_obj: dict[str, Any] = {}
                input_text = str(payload.get("input", "") or "").strip()
                if explicit_tool == "web_search":
                    query = str(payload.get("query", "") or "").strip()
                    if not query and input_text:
                        query = input_text
                    if query:
                        args_obj["query"] = query
                    if "count" in payload:
                        args_obj["count"] = payload.get("count")
                    elif "max_results" in payload:
                        args_obj["count"] = payload.get("max_results")
                    for key in ("offset", "country", "search_lang", "freshness", "safesearch"):
                        if key in payload:
                            args_obj[key] = payload.get(key)
                elif explicit_tool == "web_fetch":
                    url = str(payload.get("url", "") or "").strip()
                    if not url and input_text.startswith(("http://", "https://")):
                        url = input_text
                    if url.startswith(("http://", "https://")):
                        args_obj["url"] = url
                    if "max_chars" in payload:
                        args_obj["max_chars"] = payload.get("max_chars")
                    if "extract_mode" in payload:
                        args_obj["extract_mode"] = payload.get("extract_mode")
                elif explicit_tool == "shell":
                    command = str(payload.get("command", "") or "").strip()
                    if not command and input_text:
                        command = input_text
                    if command:
                        args_obj["command"] = command
                elif explicit_tool == "pocket_tts":
                    text = str(payload.get("text", "") or "").strip()
                    if not text and input_text:
                        text = input_text
                    if text:
                        args_obj["text"] = text
                    voice = str(payload.get("voice", "") or "").strip()
                    if voice:
                        args_obj["voice"] = voice
                    output_path = str(payload.get("output_path", "") or "").strip()
                    if output_path:
                        args_obj["output_path"] = output_path
                    if "sample_rate" in payload:
                        args_obj["sample_rate"] = payload.get("sample_rate")
                else:
                    for key, value in payload.items():
                        if key in {"tool", "name", "id", "input"}:
                            continue
                        args_obj[str(key)] = value
                    if input_text and "input" not in args_obj:
                        args_obj["input"] = input_text
                if args_obj:
                    _append_tool_call(explicit_tool, args_obj)
                    continue

            # Heuristic fallback: common pseudo-tool argument blobs.
            if "query" in payload:
                query = str(payload.get("query", "")).strip()
                if query:
                    args_obj: dict[str, Any] = {"query": query}
                    if "count" in payload:
                        args_obj["count"] = payload.get("count")
                    elif "max_results" in payload:
                        args_obj["count"] = payload.get("max_results")
                    for key in ("offset", "country", "search_lang", "freshness", "safesearch"):
                        if key in payload:
                            args_obj[key] = payload.get(key)
                    _append_tool_call("web_search", args_obj)
                    continue

            if "url" in payload:
                url = str(payload.get("url", "")).strip()
                if url.startswith(("http://", "https://")):
                    args_obj = {"url": url}
                    if "max_chars" in payload:
                        args_obj["max_chars"] = payload.get("max_chars")
                    if "extract_mode" in payload:
                        args_obj["extract_mode"] = payload.get("extract_mode")
                    _append_tool_call("web_fetch", args_obj)
        
        return tool_calls

    async def _handle_tool_calls(
        self,
        tool_calls: list[ToolCall],
        turn_usage: dict[str, int] | None = None,
        session_policy: dict[str, Any] | None = None,
        task_policy: dict[str, Any] | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> list[dict[str, Any]]:
        """Handle tool calls from LLM.
        
        Args:
            tool_calls: List of tool calls to execute
        
        Returns:
            List of tool results
        """
        results = []
        
        for tc in tool_calls:
            self._set_runtime_status("running script")
            log.info("Executing tool", tool=tc.name, call_id=tc.id)

            # Parse arguments (could be string or dict)
            arguments = tc.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}

            # ── Scale guard: hard redirect for off-track calls ─────
            # Must run before dup detection and execution. When the guard
            # fires, we skip execution entirely and return a redirect
            # message that tells the LLM what to process next.
            guard_msg = self._scale_guard_intercept(
                tc.name,
                arguments if isinstance(arguments, dict) else {},
            )
            if guard_msg is not None:
                log.info(
                    "Scale guard blocked tool call",
                    tool=tc.name,
                    call_id=tc.id,
                )
                self._emit_thinking(
                    f"🛡️ Scale guard: redirecting from {tc.name}",
                    tool=tc.name,
                    phase="tool",
                )
                self._add_session_message(
                    role="tool",
                    content=guard_msg,
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    tool_arguments=arguments if isinstance(arguments, dict) else None,
                )
                self._emit_tool_output(
                    tc.name,
                    arguments if isinstance(arguments, dict) else {},
                    guard_msg,
                )
                results.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": guard_msg,
                })
                continue

            # Emit inline thinking indicator for the tool being executed.
            args_dict = arguments if isinstance(arguments, dict) else {}
            summary = self._tool_thinking_summary(tc.name, args_dict)
            # When scale-progress is active, prefix extract/read calls
            # with progress info like "📄 3 of 27 — ".
            sp = getattr(self, "_scale_progress", None)
            if sp is not None and sp.get("total", 0) > 0:
                tool_lower = str(tc.name or "").strip().lower()
                _extractors = {"pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract", "read"}
                if tool_lower in _extractors:
                    done = sp.get("completed", 0)
                    total = sp["total"]
                    # +1 because the next write(append) will complete this item
                    current = min(done + 1, total)
                    summary = f"📄 {current} of {total} — {summary}"
            self._emit_thinking(summary, tool=tc.name, phase="tool")

            # ── Duplicate tool call detection ─────────────────────
            # If the LLM keeps requesting the same tool within a single
            # turn (e.g. re-fetching the same URL 10 times, or re-extracting
            # the same PDF with different max_chars), skip execution after
            # the Nth call and return a warning instead.
            # For path-based tools (read, pdf_extract, etc.) and URL-based
            # tools (web_fetch, web_get), the key dimension is the path/URL
            # — not the full args.  Re-reading the same file with different
            # limits is still a duplicate.
            _dup_max = max(1, int(get_config().tools.duplicate_call_max))
            _tool_lower = str(tc.name or "").strip().lower()
            _PATH_TOOLS = {"read", "pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract"}
            _URL_TOOLS = {"web_fetch", "web_get"}
            # Stateful tools that modify data between calls — exempt from
            # duplicate detection because index→search sequences are normal.
            _STATEFUL_TOOLS = {"typesense", "todo", "contacts", "scripts", "apis", "send_mail"}
            # During scale-progress tasks, give extra headroom so the LLM
            # can recover from a failed first attempt or a write-before-read
            # situation without being permanently locked out of a file.
            _sp = getattr(self, "_scale_progress", None)
            if _sp is not None and _tool_lower in (_PATH_TOOLS | _URL_TOOLS | {"glob"}):
                _dup_max = max(_dup_max, 2)
            if _tool_lower in _PATH_TOOLS and isinstance(arguments, dict):
                _sig_key = str(arguments.get("path", "")).strip()
                # Normalize to absolute path so relative and absolute
                # references to the same file share one dup-counter.
                if _sig_key:
                    _sig_key = os.path.abspath(_sig_key)
                _dup_sig = f"{tc.name}|path={_sig_key}"
            elif _tool_lower in _URL_TOOLS and isinstance(arguments, dict):
                _sig_key = str(arguments.get("url", "")).strip()
                _dup_sig = f"{tc.name}|url={_sig_key}"
            elif _tool_lower == "glob" and isinstance(arguments, dict):
                _sig_key = str(arguments.get("pattern", "")).strip()
                _dup_sig = f"{tc.name}|pattern={_sig_key}"
            else:
                try:
                    _sig_args = json.dumps(arguments, sort_keys=True, ensure_ascii=True) if isinstance(arguments, dict) else str(arguments)
                except Exception:
                    _sig_args = str(arguments)
                _dup_sig = f"{tc.name}|{_sig_args}"
            _dup_counts: dict[str, int] = getattr(self, "_turn_tool_call_counts", {})
            _dup_count = _dup_counts.get(_dup_sig, 0)
            # Stateful tools (index→search, CRUD operations) skip duplicate
            # detection — repeated calls are expected and legitimate.
            if _tool_lower in _STATEFUL_TOOLS:
                _dup_count = 0
            if _dup_count >= _dup_max:
                # During scale tasks, give a more actionable message that
                # tells the LLM to write what it has or skip the item.
                if _sp is not None and _tool_lower in (_PATH_TOOLS | _URL_TOOLS):
                    dup_msg = (
                        f"DUPLICATE CALL BLOCKED: You have already called `{tc.name}` on "
                        f"this target {_dup_count} times this turn. "
                        "If you have content from a previous call, write its summary "
                        "now (write, append=true). If you cannot recall the content, SKIP "
                        "this item and move on to the NEXT unprocessed item in the list. "
                        "Do NOT attempt to re-fetch this item again."
                    )
                else:
                    dup_msg = (
                        f"DUPLICATE CALL BLOCKED: You have already called `{tc.name}` on "
                        f"this target {_dup_count} times this turn. The content has not "
                        "changed. Use the data you already have and move on to the next "
                        "item. Do NOT re-read, re-extract, or re-glob the same target."
                    )
                log.info("Blocking duplicate tool call", tool=tc.name, call_id=tc.id, count=_dup_count)
                self._add_session_message(
                    role="tool",
                    content=dup_msg,
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    tool_arguments=arguments if isinstance(arguments, dict) else None,
                )
                self._emit_tool_output(
                    tc.name,
                    arguments if isinstance(arguments, dict) else {},
                    dup_msg,
                )
                results.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": dup_msg,
                })
                continue
            _dup_counts[_dup_sig] = _dup_count + 1
            if not hasattr(self, "_turn_tool_call_counts"):
                self._turn_tool_call_counts = _dup_counts

            # ── Scale-progress: track last action for write hint ──
            # Instead of hard-blocking read-before-write (which caused
            # deadlocks when interacting with the dup detector), we
            # append a soft reminder to the tool result after execution
            # if the LLM skipped a write step.  The LLM can still
            # proceed — the hint nudges it without creating unrecoverable
            # states.
            sp = getattr(self, "_scale_progress", None)
            _scale_write_hint = False
            _content_tools = _PATH_TOOLS | _URL_TOOLS
            if sp is not None and _tool_lower in _content_tools:
                if sp.get("_last_action") == "extract":
                    _scale_write_hint = True  # will append hint after execution
            elif sp is not None and _tool_lower == "write":
                sp["_last_action"] = "write"

            try:
                # Execute tool
                result = await self._execute_tool_with_guard(
                    name=tc.name,
                    arguments=arguments,
                    interaction_label=f"tool_call:{tc.name}",
                    turn_usage=turn_usage,
                    session_policy=session_policy,
                    task_policy=task_policy,
                    abort_event=abort_event,
                )

                # Add result to session
                _result_content = result.content if result.success else f"Error: {result.error}"
                # Append a soft write-reminder when the LLM skipped writing
                # the previous item's result before reading a new one.
                # This is a HINT, not a hard block — execution still happened.
                if _scale_write_hint and result.success:
                    _result_content += (
                        "\n\n⚠️ REMINDER: You have processed content from TWO items "
                        "without writing a summary in between. Append the summary for "
                        "the PREVIOUS item to the output first (write, append=true), "
                        "then append the summary for THIS item. Do not skip any items."
                    )
                self._add_session_message(
                    role="tool",
                    content=_result_content,
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    tool_arguments=arguments if isinstance(arguments, dict) else None,
                )
                self._emit_tool_output(
                    tc.name,
                    arguments if isinstance(arguments, dict) else {},
                    _result_content,
                )

                results.append({
                    "tool_call_id": tc.id,
                    "tool_name": tc.name,
                    "success": result.success,
                    "content": _result_content if result.success else result.error,
                })

                # ── Dup-counter rollback on failure ──────────────
                # If the call failed (e.g. wrong path), roll back the
                # duplicate counter so the LLM can retry with a
                # corrected argument (e.g. absolute path) without
                # being blocked by the dup detector.
                if not result.success:
                    _dup_counts = getattr(self, "_turn_tool_call_counts", {})
                    if _dup_sig in _dup_counts and _dup_counts[_dup_sig] > 0:
                        _dup_counts[_dup_sig] -= 1

                # ── Scale-progress tracking ───────────────────────
                # When a large-scale incremental task is running, track
                # glob results (to learn total count) and write(append)
                # calls (to count completed items) and emit a progress
                # indicator to the thinking line.
                #
                # Also: set _last_action="extract" on success so the
                # soft write-hint fires if the LLM reads another file
                # before writing.  Only set on success — a failed
                # extract should not trigger the hint on retry.
                sp = getattr(self, "_scale_progress", None)
                if result.success and sp is not None:
                    tool_lower = str(tc.name or "").strip().lower()
                    if tool_lower in _PATH_TOOLS:
                        sp["_last_action"] = "extract"
                    if tool_lower in _URL_TOOLS:
                        sp["_last_action"] = "extract"
                    if tool_lower == "glob" and result.content:
                        # Count lines in glob output to determine total items
                        # AND store the full list so we can inject a progress
                        # note into every LLM call (preventing the LLM from
                        # "forgetting" the list as context grows).
                        # Filter out non-path lines like "Found 27 file(s):"
                        # or other header/summary text the glob tool may emit.
                        lines = [
                            ln.strip() for ln in result.content.strip().splitlines()
                            if ln.strip()
                            and not re.match(r"^Found \d+", ln.strip())
                            and (
                                "/" in ln
                                or "\\" in ln
                                or "." in ln.strip().rsplit("/", 1)[-1]
                            )
                        ]
                        if lines:
                            sp["total"] = len(lines)
                            sp["completed"] = 0
                            sp["items"] = list(lines)
                            sp["done_items"] = set()
                            # Mark glob as completed so the scale guard
                            # blocks any subsequent re-glob attempts.
                            sp["_glob_completed"] = True
                    elif tool_lower == "write" and isinstance(arguments, dict):
                        # Track the output file path so the scale guard
                        # can block re-reads of it during the loop.
                        # We store TWO paths:
                        #   _output_file: the resolved absolute path (for
                        #     comparison in scale guard read-blocking)
                        #   _output_file_arg: the original arg passed to
                        #     the write tool (for the micro-loop to reuse,
                        #     since the write tool re-resolves its input)
                        # Extract the real resolved path from the tool result.
                        real_path = ""
                        if result.content and " to " in result.content:
                            after_to = result.content.split(" to ", 1)[-1]
                            real_path = after_to.split(" (requested:")[0].strip()
                        if not real_path:
                            write_path = str(arguments.get("path", "")).strip()
                            if write_path:
                                real_path = os.path.abspath(write_path)
                        if not sp.get("_output_file"):
                            if real_path:
                                sp["_output_file"] = real_path
                            # Also keep the original argument for micro-loop
                            original_arg = str(arguments.get("path", "")).strip()
                            if original_arg:
                                sp["_output_file_arg"] = original_arg
                        # Track distinct output files.  When the LLM writes
                        # to MULTIPLE different files (one per item), the
                        # micro-loop cannot take over because it assumes a
                        # single output file with appends.
                        if real_path:
                            output_files: set[str] = sp.setdefault("_output_files", set())
                            output_files.add(real_path)
                        # Track item completion.
                        # For append=True: match written content against items.
                        # For append=False: also try matching — the LLM may
                        # write separate files per item (e.g. "FinSMEs-18.07.2025.csv"
                        # for item "18.07.2025. https://…").
                        is_append = arguments.get("append") is True
                        if is_append:
                            sp["completed"] = sp.get("completed", 0) + 1
                        # Build a search text from both file content and path.
                        written_content = str(arguments.get("content", ""))
                        write_path = str(arguments.get("path", "")).strip().lower()
                        # Check a broader portion of written content —
                        # not just the first line — to catch items
                        # mentioned in markdown headers, sub-headings, etc.
                        content_head = written_content[:500].lower()
                        # For non-append writes, also search in the file path
                        # (the LLM may embed the item identifier in the filename).
                        search_text = content_head + " " + write_path
                        done_items: set[str] = sp.get("done_items", set())
                        items: list[str] = sp.get("items", [])
                        for item in items:
                            if item in done_items:
                                continue
                            # Build match candidates for this item:
                            # - filename (last path component)
                            # - URL domain+path for web URLs
                            # - the item itself (lowered)
                            candidates: list[str] = []
                            if "/" in item:
                                # Could be a file path or URL
                                candidates.append(item.rsplit("/", 1)[-1].lower())
                            # The item itself (e.g. "index.hr/...", "Company Name")
                            candidates.append(item.lower())
                            # For URLs, also match just the path portion.
                            # Handle both plain URLs and items with
                            # embedded URLs (e.g. "18.07.2025 - https://…")
                            _url_in_item = None
                            if item.startswith(("http://", "https://")):
                                _url_in_item = item
                            else:
                                _um = re.search(r"https?://[^\s)\]}>\"']+", item)
                                if _um:
                                    _url_in_item = _um.group(0)
                            if _url_in_item:
                                try:
                                    from urllib.parse import urlparse
                                    parsed = urlparse(_url_in_item)
                                    path_part = parsed.path.strip("/")
                                    if path_part:
                                        candidates.append(path_part.lower())
                                    # hostname + path for partial matching
                                    if parsed.hostname:
                                        candidates.append(
                                            f"{parsed.hostname}{parsed.path}".lower()
                                        )
                                except Exception:
                                    pass
                            # Extract date-like tokens from items
                            # (e.g. "18.07.2025" from "18.07.2025. https://…")
                            _date_m = re.search(r"\d{2}\.\d{2}\.\d{4}", item)
                            if _date_m:
                                candidates.append(_date_m.group(0))
                            # For non-path items (e.g. "Company Name"),
                            # also try normalized form
                            if " " in item:
                                candidates.append(
                                    re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
                                )
                            matched = any(
                                c and c in search_text
                                for c in candidates
                                if len(c) >= 3
                            )
                            if matched:
                                done_items.add(item)
                                if not is_append:
                                    sp["completed"] = sp.get("completed", 0) + 1
                                break
                        sp["done_items"] = done_items
                        total = sp.get("total", 0)
                        done = sp["completed"]
                        path = str(arguments.get("path", "")).strip()
                        filename = path.rsplit("/", 1)[-1] if path else ""
                        if total > 0:
                            pct = int(done / total * 100)
                            progress_text = f"{done} of {total} ({pct}%)"
                        else:
                            progress_text = f"{done} items written"
                        # Show last-written filename in the indicator.
                        # Look at the content to extract a section title
                        # if it starts with "## " or "# ".
                        content_str = str(arguments.get("content", ""))
                        label = ""
                        for cline in content_str.splitlines():
                            cline = cline.strip()
                            if cline.startswith("#"):
                                label = cline.lstrip("#").strip()[:60]
                                break
                        if not label and filename:
                            label = filename[:60]
                        display = f"📄 {progress_text}"
                        if label:
                            display += f" — {label}"
                        self._emit_thinking(display, tool="progress", phase="tool")
                        # ── Context trimming ─────────────────────
                        # After successfully writing a summary, the
                        # full extracted content of previous items is
                        # no longer needed.  Trim large tool results
                        # from earlier in the turn to keep the context
                        # lean and prevent the LLM from "forgetting"
                        # where it is in the list.
                        self._trim_processed_extracts_in_session()

                # Auto-capture contacts from send_mail usage.
                if result.success and hasattr(self, "_auto_capture_contacts_from_tool_call"):
                    try:
                        await self._auto_capture_contacts_from_tool_call(
                            tc.name, arguments if isinstance(arguments, dict) else {},
                        )
                    except Exception as _ac_err:
                        log.warning("Auto-capture contacts failed", tool=tc.name, error=str(_ac_err))

                # Auto-capture scripts from write tool usage.
                if result.success and hasattr(self, "_auto_capture_scripts_from_tool_call"):
                    try:
                        await self._auto_capture_scripts_from_tool_call(
                            tc.name, arguments if isinstance(arguments, dict) else {},
                        )
                    except Exception as _ac_err:
                        log.warning("Auto-capture scripts failed", tool=tc.name, error=str(_ac_err))

                # Auto-capture APIs from web_fetch tool usage.
                if result.success and hasattr(self, "_auto_capture_apis_from_tool_call"):
                    try:
                        await self._auto_capture_apis_from_tool_call(
                            tc.name, arguments if isinstance(arguments, dict) else {},
                        )
                    except Exception as _ac_err:
                        log.warning("Auto-capture APIs failed", tool=tc.name, error=str(_ac_err))

            except Exception as e:
                log.error("Tool execution failed", tool=tc.name, error=str(e))
                
                self._add_session_message(
                    role="tool",
                    content=f"Error: {str(e)}",
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    tool_arguments=arguments if isinstance(arguments, dict) else None,
                )
                self._emit_tool_output(
                    tc.name,
                    arguments if isinstance(arguments, dict) else {},
                    f"Error: {str(e)}",
                )
                
                results.append({
                    "tool_call_id": tc.id,
                    "tool_name": tc.name,
                    "success": False,
                    "error": str(e),
                })
                # Roll back dup counter on exception so a retry is allowed.
                _dup_counts = getattr(self, "_turn_tool_call_counts", {})
                if _dup_sig in _dup_counts and _dup_counts[_dup_sig] > 0:
                    _dup_counts[_dup_sig] -= 1

        self._set_runtime_status("thinking")
        return results
