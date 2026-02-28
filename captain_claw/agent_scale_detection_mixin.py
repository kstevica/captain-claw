"""Scale detection, advisory injection, and deferred scale init for Agent.

This mixin handles the pre-flight detection of large-scale list tasks and
the initialization of scale-progress tracking:

- Scale advisory template selection and injection
- Pattern-based detection of list-processing requests
- Source-URL-only item detection (degenerate extraction guard)
- Deferred scale init after web_fetch brings article content
- Scale-progress initialization from list_task_plan
- Micro-loop summary building (de-duplicated across takeover paths)
- Advisory-stripping helper

These were extracted from ``agent_orchestration_mixin.py`` to keep the
main orchestration loop focused on flow control.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger


log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pre-flight scale advisory — injected into the system/user context when the
# orchestration layer detects a large-item-count task so the LLM chooses the
# incremental append-to-file strategy instead of trying to hold everything in
# context.
# ---------------------------------------------------------------------------
_SCALE_ADVISORY_SINGLE_FILE = (
    "\n\n--- SCALE ADVISORY (auto-detected) ---\n"
    "This task involves approximately {item_count} items. "
    "That is TOO MANY to hold in the context window at once.\n"
    "MANDATORY strategy — you MUST follow this exactly:\n"
    "1. Glob/list all items first to get the full file list.\n"
    "2. Create the output file with a header (write tool, append=false).\n"
    "3. Process items ONE AT A TIME in strict read-then-write pairs:\n"
    "   - Read/extract one item.\n"
    "   - Immediately in the next response, APPEND its processed result to the "
    "output file (write tool, append=true).\n"
    "   - Only then move to the next item.\n"
    "4. NEVER read more than one item before writing. Pattern: "
    "read → write → read → write → ... until done.\n"
    "5. After all items: the file is complete. Give the user a short summary.\n"
    "PROHIBITED actions during the loop:\n"
    "- Do NOT re-read the output file. You wrote it — trust the append.\n"
    "- Do NOT re-run glob. You have the list.\n"
    "- Do NOT re-extract the same file with different parameters.\n"
    "- Extract once, summarize, append, move on.\n"
    "If the item count exceeds 100, FIRST tell the user the count and ask for "
    "confirmation before starting.\n"
    "--- END SCALE ADVISORY ---\n"
)

_SCALE_ADVISORY_FILE_PER_ITEM = (
    "\n\n--- SCALE ADVISORY (auto-detected) ---\n"
    "This task involves approximately {item_count} items. "
    "That is TOO MANY to hold in the context window at once.\n"
    "OUTPUT: Each item produces its OWN SEPARATE file "
    "(filename template: {filename_template}).\n"
    "MANDATORY strategy — you MUST follow this exactly:\n"
    "1. Glob/list all items first to get the full item list.\n"
    "2. Process items ONE AT A TIME in strict read-then-write pairs:\n"
    "   - Read/extract one item.\n"
    "   - Immediately write its processed result to a NEW file named using "
    "the template (write tool, append=false). Each item gets its own file.\n"
    "   - Only then move to the next item.\n"
    "3. NEVER read more than one item before writing. Pattern: "
    "read → write → read → write → ... until done.\n"
    "4. After all items: all files are complete. Give the user a short summary.\n"
    "CRITICAL: Do NOT append everything to one file. "
    "Each item MUST be written to a SEPARATE file.\n"
    "PROHIBITED actions during the loop:\n"
    "- Do NOT re-read any output file. You wrote it — trust the write.\n"
    "- Do NOT re-run glob. You have the list.\n"
    "- Do NOT re-extract the same item with different parameters.\n"
    "- Extract once, process, write to per-item file, move on.\n"
    "If the item count exceeds 100, FIRST tell the user the count and ask for "
    "confirmation before starting.\n"
    "--- END SCALE ADVISORY ---\n"
)

_SCALE_ADVISORY_NO_FILE = (
    "\n\n--- SCALE ADVISORY (auto-detected) ---\n"
    "This task involves approximately {item_count} items. "
    "That is TOO MANY to hold in the context window at once.\n"
    "OUTPUT: Results should NOT be written to files. "
    "The final action is: {final_action}.\n"
    "MANDATORY strategy — you MUST follow this exactly:\n"
    "1. Glob/list all items first to get the full item list.\n"
    "2. Process items ONE AT A TIME:\n"
    "   - Read/extract one item.\n"
    "   - Immediately process/deliver its result ({final_action}).\n"
    "   - Only then move to the next item.\n"
    "3. NEVER read more than one item before processing.\n"
    "4. After all items: give the user a short summary.\n"
    "PROHIBITED actions during the loop:\n"
    "- Do NOT re-run glob. You have the list.\n"
    "- Do NOT re-extract the same item with different parameters.\n"
    "- Extract once, process, deliver, move on.\n"
    "If the item count exceeds 100, FIRST tell the user the count and ask for "
    "confirmation before starting.\n"
    "--- END SCALE ADVISORY ---\n"
)


def _build_scale_advisory(
    item_count: int | str,
    output_strategy: str = "single_file",
    filename_template: str = "",
    final_action: str = "write_file",
) -> str:
    """Build the appropriate scale advisory based on output strategy."""
    if output_strategy == "file_per_item":
        return _SCALE_ADVISORY_FILE_PER_ITEM.format(
            item_count=item_count,
            filename_template=filename_template or "(per-item filename)",
        )
    elif output_strategy == "no_file":
        return _SCALE_ADVISORY_NO_FILE.format(
            item_count=item_count,
            final_action=final_action,
        )
    else:
        return _SCALE_ADVISORY_SINGLE_FILE.format(item_count=item_count)


# Patterns that suggest a task touching many files/items in a folder tree.
_LARGE_SCALE_INPUT_PATTERNS = [
    re.compile(r"\b(?:all|every)\s+files?\b", re.I),
    re.compile(r"\bgo\s+through\b.*\bfiles?\b", re.I),
    re.compile(r"\beach\s+file\b", re.I),
    re.compile(r"\bfor\s+each\b.*\bfiles?\b", re.I),
    re.compile(r"\bprocess\s+(?:all|every|each)\b", re.I),
    re.compile(r"\blist\s+of\s+files\b", re.I),
    re.compile(r"\bgenerate\b.*\bfor\s+(?:all|every|each)\b", re.I),
    re.compile(r"\bsummar(?:y|ize|ise)\b.*\b(?:all|every|each)\b.*\bfiles?\b", re.I),
    re.compile(r"\bfolder\b.*\band\b.*\bsubfolders?\b", re.I),
    re.compile(r"\brecursive(?:ly)?\b.*\bfiles?\b", re.I),
    # Explicit list-providing language
    re.compile(r"\bhere (?:is|are) the\s+(?:list|urls?|links?|files?|pages?|items?)\b", re.I),
    re.compile(r"\bfrom (?:these|the following)\s+(?:urls?|links?|files?|pages?)\b", re.I),
    re.compile(r"\bthe following\s+(?:urls?|links?|files?|pages?|items?)\b", re.I),
    re.compile(r"\bthese\s+(?:urls?|links?|files?|pages?)\b", re.I),
    # Deferred list language — article/page contains a list to be processed
    re.compile(r"\bthere\s+(?:is|are)\s+(?:a\s+)?(?:list|set|collection)\b", re.I),
    re.compile(r"\b(?:contains?|includes?|has)\s+(?:a\s+)?(?:list|set|collection)\s+of\b", re.I),
    re.compile(r"\bresearch\b.*\b(?:all|every|each)\b", re.I),
    re.compile(r"\b(?:about|report\s+(?:on|about))\s+(?:all|every|each)\b", re.I),
    # List-producing language
    re.compile(r"\b(?:create|compile|build|make|generate|produce|prepare)\s+(?:a\s+)?(?:list|csv|spreadsheet|table)\b", re.I),
    re.compile(r"\bpopulate\s+(?:a\s+)?(?:csv|spreadsheet|table)\b", re.I),
]

# Count inline URLs to detect user-provided lists even without keywords.
_INLINE_URL_RE = re.compile(r"https?://[^\s)\]}>\"']+")

# Tasks whose primary purpose is NOT list-processing.  When the task
# description matches one of these patterns the deferred scale init
# should NOT fire — otherwise it hijacks the worker and prevents it
# from completing its actual job (e.g. sending an email, combining
# files into a single summary, etc.).
_SKIP_SCALE_DETECTION_RE = re.compile(
    r"(?:"
    r"send\s+(?:an?\s+)?(?:email|e-mail|message|notification)"
    r"|send\s+(?:via|using|through)\s+(?:mailgun|smtp|sendgrid|ses|postmark)"
    r"|(?:email|e-mail)\s+(?:via|using|through)\b"
    r"|mailgun|sendgrid|smtp|send_mail"
    r"|send\b.*\bvia\s+(?:internal|the)\s+\w+\s+tool"
    r"|combine\s+(?:per-day|all|the|both|two)\s+.*?(?:markdowns?|files?|results)"
    r"|combine\s+(?:the\s+)?(?:\w+\s+){0,3}(?:markdowns?|files?|results)\s+into\b"
    r"|merge\s+(?:all\s+)?(?:the\s+)?(?:files|results|outputs)"
    r"|assemble\s+(?:a\s+)?(?:single|one|final|combined|unified)\s+(?:markdown|file|document|report)"
    r"|assemble\s+.*\bsection\s+per\b"
    r"|produce\s+\d+-paragraph\s+summary"
    # File-discovery-only tasks: the worker just lists files and returns
    # the list — it should NOT enter the scale micro-loop because there
    # is no per-item processing to do.
    r"|(?:use\s+(?:the\s+)?glob\b.*return\s+(?:the\s+)?(?:complete\s+)?(?:list|paths?|files?))"
    r"|(?:find\s+(?:all|every)\s+.*files?\b.*return\s+(?:the\s+)?(?:list|paths?))"
    r"|(?:locate\s+(?:all|every)\s+.*files?\b)"
    r"|(?:return\s+(?:the\s+)?(?:complete\s+)?list\s+of\b.*(?:file|relative)\s*(?:paths?|names?))"
    r")",
    re.IGNORECASE,
)

# Marker that starts the dependency-output section injected by the
# orchestrator into worker prompts.  Everything from this marker onward
# is upstream task output and must be stripped before checking the skip
# regex — otherwise upstream task titles (e.g. "Locate all PDF files")
# falsely trigger skip patterns meant for the current task.
_DEP_OUTPUT_MARKER = "Results from previous steps:"


def _strip_dep_output_section(text: str) -> str:
    """Return *text* with the dependency-output section removed."""
    idx = text.find(_DEP_OUTPUT_MARKER)
    if idx < 0:
        return text
    return text[:idx]


class AgentScaleDetectionMixin:
    """Scale detection, advisory injection, and deferred scale init."""

    # ------------------------------------------------------------------
    # Pre-flight scale detection
    # ------------------------------------------------------------------

    @staticmethod
    def _input_suggests_large_scale(user_input: str) -> bool:
        """Return True if the user input matches patterns that typically
        produce a very large number of items, or the user has explicitly
        provided a list of items (URLs, file paths, numbered entries).

        Fires on:
        1. Large-scope phrasing ("all files in folder and subfolders")
        2. Explicit list-providing language ("here are the urls", "these files")
        3. List-producing language ("create a csv", "compile a list")
        4. Inline item count: 3+ URLs or 5+ numbered/bulleted lines
        """
        text = (user_input or "").strip()
        if not text:
            return False
        # Keyword patterns
        if any(p.search(text) for p in _LARGE_SCALE_INPUT_PATTERNS):
            return True
        # Inline URL count — if the user pasted 3+ URLs, it's a list task
        if len(_INLINE_URL_RE.findall(text)) >= 3:
            return True
        # Numbered / bulleted list lines (e.g. "1. ...", "- ...", "* ...")
        list_lines = re.findall(r"^\s*(?:\d+[\.\)]\s+|[-*•]\s+)", text, re.MULTILINE)
        if len(list_lines) >= 5:
            return True
        return False

    @staticmethod
    def _items_are_source_urls_only(items: list[str]) -> bool:
        """Return True if all items appear to be the same source URL (or trivial variants).

        When the initial list extraction runs BEFORE the article is fetched,
        it may produce items that are just the source article URL repeated
        (possibly with trailing punctuation).  These are NOT real list members
        and should not block deferred re-extraction.

        Heuristics:
        - All items start with http:// or https://
        - After normalizing (strip trailing punctuation, lowercase), there
          are 2 or fewer unique URLs (duplicates of the source URL)
        """
        if not items:
            return True
        all_urls = all(
            item.strip().startswith(("http://", "https://"))
            for item in items
            if item.strip()
        )
        if not all_urls:
            return False
        # Normalize: lowercase, strip trailing comma/period/space
        normalized = set()
        for item in items:
            clean = item.strip().lower().rstrip(".,;:!?/ ")
            if clean:
                normalized.add(clean)
        # If all items resolve to ≤2 unique URLs, they're source URLs
        return len(normalized) <= 2

    def _preflight_scale_check(
        self,
        effective_user_input: str,
        list_task_plan: dict[str, Any],
    ) -> str:
        """Detect large-scale tasks and return a scale advisory string.

        The advisory is appended to the effective user input that gets
        passed to the LLM so it adopts the incremental append-to-file
        strategy instead of trying to hold all results in context.

        Fires when:
        - Member count meets the configured threshold (scale_advisory_min_members), OR
        - The input explicitly provides or requests a list (detected by
          ``_input_suggests_large_scale``), even with fewer members.

        Returns an empty string when no advisory is needed.
        """
        member_count = len(list_task_plan.get("members", []))
        _scale_cfg = get_config().scale
        large_from_members = member_count >= _scale_cfg.scale_advisory_min_members
        # IMPORTANT: check the STRIPPED text (without dependency output)
        # for large-scale input patterns.  The dependency output from
        # upstream tasks can contain dozens of URLs / file paths that
        # falsely trigger the inline-URL-count heuristic, causing
        # non-scale tasks (combine, send_mail) to enter the scale path.
        _stripped_input = _strip_dep_output_section(effective_user_input)
        large_from_input = self._input_suggests_large_scale(_stripped_input)

        if not large_from_members and not large_from_input:
            return ""

        # Skip scale detection for tasks whose primary purpose is not
        # per-item processing (e.g. file discovery, sending email, merging).
        # Strip the "Results from previous steps:" dependency-output section
        # before checking, otherwise titles of upstream tasks (e.g. "Locate
        # all PDF files") can falsely match the skip patterns and prevent
        # the scale loop from firing for the current task.
        _skip_check_text = _stripped_input
        if _SKIP_SCALE_DETECTION_RE.search(_skip_check_text):
            # But DON'T skip when the stripped text also contains per-item
            # processing language.  This handles combined tasks like
            # "For each PDF, extract and summarize, then assemble a single
            # Markdown file" — the "assemble" hits the skip regex, but the
            # "for each PDF" signals real list-processing that needs the
            # scale loop.
            if not (
                self._input_suggests_large_scale(_skip_check_text)
                or self._is_list_processing_request(_skip_check_text)
            ):
                return ""

        # Use the known member count if available; otherwise use a
        # placeholder hint that prompts the LLM to discover the count
        # itself via glob before processing.
        if member_count > 0:
            estimated_count = member_count
        else:
            # We don't know the exact count yet — signal "many".
            estimated_count = "many (exact count unknown — discover via glob first)"

        output_strategy = str(list_task_plan.get("output_strategy", "single_file")).strip().lower()
        filename_template = str(list_task_plan.get("output_filename_template", "")).strip()
        final_action = str(list_task_plan.get("final_action", "write_file")).strip()

        advisory = _build_scale_advisory(
            item_count=estimated_count,
            output_strategy=output_strategy,
            filename_template=filename_template,
            final_action=final_action,
        )

        self._emit_tool_output(
            "task_contract",
            {
                "step": "preflight_scale_advisory",
                "detected_from": "list_members" if large_from_members else "input_pattern",
                "estimated_items": member_count if large_from_members else -1,
            },
            (
                "step=preflight_scale_advisory\n"
                f"detected_from={'list_members' if large_from_members else 'input_pattern'}\n"
                f"estimated_items={member_count if large_from_members else 'unknown'}\n"
                "action=injecting_scale_advisory_into_context"
            ),
        )
        return advisory

    async def _deferred_scale_init(
        self,
        effective_user_input: str,
        list_task_plan: dict[str, Any],
        turn_usage: dict[str, int],
    ) -> dict[str, Any]:
        """Re-run list extraction after a web_fetch brings in new content.

        When the user says "fetch article X, there is a list of Y, research all",
        the initial list extraction can't find members because the article isn't
        fetched yet.  This method re-runs extraction with the now-available
        fetched content in session messages.

        Returns the updated ``list_task_plan`` (unchanged if no new members).
        """
        # ── Skip scale detection for non-scalable tasks ──
        # Tasks like "send an email" or "combine all markdowns into one file"
        # should NEVER be hijacked by the scale micro-loop.  Without this
        # guard, the deferred init detects file paths/names in the session
        # as "members" and starts a useless micro-loop that prevents the
        # worker from completing its real job.
        # Strip dependency output section so upstream task titles don't
        # falsely match skip patterns for the current task.
        _skip_check_text = _strip_dep_output_section(effective_user_input)
        if _SKIP_SCALE_DETECTION_RE.search(_skip_check_text):
            # But DON'T skip when the text also contains per-item processing
            # language — this is a combined process+assemble task that needs
            # the scale loop (e.g. "for each PDF, extract and assemble into
            # a single Markdown").
            if not (
                self._input_suggests_large_scale(_skip_check_text)
                or self._is_list_processing_request(_skip_check_text)
            ):
                return list_task_plan

        # Limit attempts to avoid wasting tokens.  Allow up to 3 tries
        # because the LLM list extraction can randomly fail on the first
        # attempt (especially when two workers run in parallel and the
        # source content is large or complex).
        _attempts = getattr(self, "_deferred_scale_attempts", 0)
        if _attempts >= 3:
            return list_task_plan
        sp = getattr(self, "_scale_progress", None)
        _existing_items = sp.get("items", []) if sp else []
        if (
            sp is not None
            and len(_existing_items) >= 2
            and not self._items_are_source_urls_only(_existing_items)
        ):
            # Scale already initialized with enough diverse items — nothing to do.
            return list_task_plan
        # Only attempt re-detection if the TASK's own input (without
        # dependency output) suggests list processing.  Using the full
        # effective_user_input would let upstream task output (which may
        # contain "for each" language or dozens of URLs) falsely trigger
        # re-extraction for non-scale tasks like combine/send.
        if not (
            self._input_suggests_large_scale(_skip_check_text)
            or self._is_list_processing_request(_skip_check_text)
        ):
            return list_task_plan

        # Re-collect context with generous limits.  The default
        # per_message_chars (1400) truncates fetched articles before the
        # list of items appears, so we use a much larger budget here.
        # Articles can be 30-60K chars; we allow up to 50K per message
        # and 80K total to capture full listicles even when the list
        # appears late in the article.
        new_context = self._collect_list_extraction_context(
            max_messages=10,
            max_chars=80000,
            per_message_chars=50000,
        )
        if not new_context or len(new_context) < 200:
            log.info(
                "Deferred scale init: context too short, skipping",
                context_len=len(new_context or ""),
            )
            return list_task_plan

        # Increment attempt counter to limit retries.
        self._deferred_scale_attempts = _attempts + 1

        log.info(
            "Deferred scale init: re-running list extraction",
            context_chars=len(new_context),
        )
        new_plan = await self._generate_list_task_plan(
            user_input=effective_user_input,
            context_excerpt=new_context,
            turn_usage=turn_usage,
            # Allow more tokens for deferred re-extraction: the context is
            # much larger (full article) and may contain 20–100+ list members
            # with URLs and per-member context.
            max_tokens_override=min(6000, int(get_config().model.max_tokens)),
        )
        new_members = new_plan.get("members", [])
        # Accept 2+ real members — the source-URL guard (below and in
        # _needs_deferred_scale_init) already catches the degenerate case
        # where extraction just echoed the source URLs.  A threshold of 3
        # incorrectly rejects legitimate 2-item tasks.
        if len(new_members) < 2:
            log.info(
                "Deferred scale init: too few members from re-extraction",
                members=len(new_members),
                members_preview=new_members[:5],
            )
            # If we still have stale source-URL items in scale_progress,
            # clear them so the main LLM loop doesn't try to use them.
            _stale_sp = getattr(self, "_scale_progress", None)
            _stale_items = _stale_sp.get("items", []) if _stale_sp else []
            if _stale_items and self._items_are_source_urls_only(_stale_items):
                log.info(
                    "Deferred scale init: clearing stale source-URL scale_progress",
                    stale_items=len(_stale_items),
                )
                self._scale_progress = None
            return list_task_plan

        # Initialize scale progress with the newly discovered members.
        self._scale_progress = self._init_scale_progress_from_plan(
            new_plan, user_input=effective_user_input,
        )

        # For inline extraction mode, store the full source page content
        # so the micro-loop can feed it (rather than tiny _member_context
        # snippets) to the per-item LLM calls.
        if self._scale_progress.get("_extraction_mode") == "inline" and new_context:
            self._scale_progress["_source_page_content"] = new_context

        # Inject scale advisory into effective user input.
        _out_strategy = str(new_plan.get("output_strategy", "single_file")).strip().lower()
        advisory = _build_scale_advisory(
            item_count=len(new_members),
            output_strategy=_out_strategy,
            filename_template=str(new_plan.get("output_filename_template", "")).strip(),
            final_action=str(new_plan.get("final_action", "write_file")).strip(),
        )

        self._emit_tool_output(
            "task_contract",
            {
                "step": "deferred_scale_init",
                "members": len(new_members),
                "output_strategy": _out_strategy,
            },
            (
                "step=deferred_scale_init\n"
                f"members={len(new_members)}\n"
                f"output_strategy={_out_strategy}\n"
                "note=scale loop initialized after web_fetch provided article content"
            ),
        )
        log.info(
            "Deferred scale init: members discovered after fetch",
            members=len(new_members),
            output_strategy=_out_strategy,
        )
        return new_plan

    # ------------------------------------------------------------------
    # File-path member repair
    # ------------------------------------------------------------------

    _FILE_PATH_EXTS: frozenset[str] = frozenset({
        ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
        ".txt", ".csv", ".md", ".json", ".xml", ".html", ".htm",
        ".yaml", ".yml", ".toml", ".rtf", ".odt", ".ods",
    })

    def _repair_file_path_members(self, members: list[str]) -> list[str]:
        """Fix file-path members that are missing a directory prefix.

        The LLM member extractor sometimes strips a common directory prefix
        from file paths (e.g. ``pdf-test/subdir/file.pdf`` becomes
        ``subdir/file.pdf``).  Detect and repair by:

        1. Checking whether the first few members resolve against the
           workspace base path.
        2. If they don't, scanning recent session messages for a prior
           tool call that used a longer path ending with the same member
           string — the difference is the missing prefix.
        3. As a fallback, searching the filesystem (rglob) for the first
           member's filename and deriving the prefix from the result.

        Returns the (possibly repaired) member list.
        """
        if not members or len(members) < 2:
            return members

        # --- Are these file paths? ---
        _file_count = sum(
            1 for m in members
            if os.path.splitext(m)[-1].lower() in self._FILE_PATH_EXTS
        )
        if _file_count < len(members) * 0.5:
            return members  # Not predominantly file paths

        base = getattr(self, "workspace_base_path", None)
        if not base:
            tools = getattr(self, "tools", None)
            if tools:
                base = getattr(tools, "runtime_base_path", None)
        if not base:
            return members
        base = Path(base)

        # --- Already correct? ---
        test_member = next(
            (m for m in members if os.path.splitext(m)[-1].lower() in self._FILE_PATH_EXTS),
            members[0],
        )
        if (base / test_member).exists():
            return members

        # --- Strategy 1: find prefix from prior tool calls in session ---
        prefix = self._find_prefix_from_session(test_member)
        if prefix and (base / (prefix + test_member)).exists():
            repaired = [prefix + m for m in members]
            log.info(
                "Repaired member file paths (from session)",
                prefix=prefix,
                count=len(members),
            )
            return repaired

        # --- Strategy 2: rglob for the first member's filename ---
        test_filename = Path(test_member).name
        try:
            found_list = list(base.rglob(test_filename))
        except Exception:
            found_list = []
        for found_path in found_list:
            try:
                found_rel = str(found_path.relative_to(base))
            except ValueError:
                continue
            if found_rel.endswith(test_member) and found_rel != test_member:
                prefix = found_rel[: -len(test_member)]
                # Verify prefix works for a second member too
                if len(members) > 1:
                    verify = next(
                        (m for m in members[1:4]
                         if os.path.splitext(m)[-1].lower() in self._FILE_PATH_EXTS),
                        None,
                    )
                    if verify and not (base / (prefix + verify)).exists():
                        continue
                repaired = [prefix + m for m in members]
                log.info(
                    "Repaired member file paths (from rglob)",
                    prefix=prefix,
                    count=len(members),
                )
                return repaired

        return members  # Could not repair

    def _find_prefix_from_session(self, member: str) -> str:
        """Scan session messages for a prior tool call whose path ends with *member*."""
        session = getattr(self, "session", None)
        if not session:
            return ""
        for msg in reversed(getattr(session, "messages", [])):
            role = str(msg.get("role", "")).strip().lower()
            if role == "assistant":
                for tc in msg.get("tool_calls", []) or []:
                    args = tc.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            continue
                    path_arg = str(args.get("path", "")).strip()
                    if path_arg and path_arg.endswith(member) and path_arg != member:
                        return path_arg[: -len(member)]
            # Also check tool result content — the dependency output from
            # locate_pdfs may list full paths in the text.
            if role in ("user", "system"):
                content = str(msg.get("content", ""))
                idx = content.find(member)
                if idx > 0:
                    # Walk backwards to find the start of the path segment
                    start = idx - 1
                    while start >= 0 and content[start] not in ("\n", "\r", "\t", " ", ",", "[", '"', "'"):
                        start -= 1
                    candidate = content[start + 1: idx]
                    if candidate and "/" in candidate + member:
                        return candidate
        return ""

    # ------------------------------------------------------------------
    # Scale-progress initialization helper
    # ------------------------------------------------------------------

    def _init_scale_progress_from_plan(
        self,
        plan: dict[str, Any],
        user_input: str = "",
    ) -> dict[str, Any]:
        """Build the ``_scale_progress`` dict from a list_task_plan.

        Centralizes the duplicated scale-progress initialization that
        previously appeared in multiple places in ``complete()``.

        Args:
            plan: The list_task_plan dict with members, per_member_action, etc.
            user_input: Raw user input / task description — used to detect
                explicit no-external-fetch signals (e.g. "do not follow links").
        """
        members = plan.get("members", [])

        # Repair file-path members that the LLM extractor may have
        # shortened by stripping a common directory prefix.
        members = self._repair_file_path_members(members)
        plan["members"] = members

        _out_strategy = str(plan.get("output_strategy", "single_file")).strip().lower()
        if _out_strategy not in ("file_per_item", "single_file", "no_file"):
            _out_strategy = "single_file"

        # For single_file strategy, the plan may include an output_file field
        # that the LLM derived from the task description.
        _plan_output_file = str(plan.get("output_file", "")).strip()

        progress: dict[str, Any] = {
            "total": len(members),
            "completed": 0,
            "items": list(members),
            "done_items": set(),
            "_output_strategy": _out_strategy,
            "_output_filename_template": str(plan.get("output_filename_template", "")).strip(),
            "_output_file_from_plan": _plan_output_file,
            "_final_action": str(plan.get("final_action", "write_file")).strip(),
            "_extraction_mode": self._classify_item_extraction_mode(
                members,
                per_member_action=str(plan.get("per_member_action", "")),
                user_input=user_input,
            ),
            "_member_context": plan.get("member_context") or {},
            "_processing_mode": str(plan.get("processing_mode", "summarize")).strip().lower(),
        }
        if _out_strategy == "no_file":
            progress["_sink_collection"] = ""
            progress["_sink_email_to"] = ""
        log.info(
            "Scale progress initialized",
            total=len(members),
            extraction_mode=progress["_extraction_mode"],
            output_strategy=_out_strategy,
            processing_mode=progress["_processing_mode"],
        )
        return progress

    # ------------------------------------------------------------------
    # Advisory stripping helper
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_scale_advisory(text: str) -> str:
        """Remove the ``--- SCALE ADVISORY ---`` block from text.

        The advisory is injected into effective_user_input for the outer
        LLM loop but should be stripped before passing text to the micro-
        loop or deriving the per-member action.
        """
        start = text.find("\n\n--- SCALE ADVISORY")
        if start <= 0:
            return text
        end = text.find("--- END SCALE ADVISORY ---")
        if end > start:
            return text[:start].rstrip()
        return text

    # ------------------------------------------------------------------
    # Micro-loop summary builder (de-duplicated)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_micro_loop_summary(
        micro_result: dict[str, Any],
        scale_progress: dict[str, Any] | None,
        output_file: str = "",
    ) -> str:
        """Build a human-readable summary from micro-loop results.

        This was duplicated across 4+ takeover paths in ``complete()``.
        Now it's a single method called from each path.
        """
        sp_total = micro_result.get("total", 0)
        sp_processed = micro_result.get("processed", 0)
        sp_completed = micro_result.get("completed_total", sp_processed)
        sp_failed = micro_result.get("failed", 0)
        sp_cancelled = micro_result.get("cancelled", False)
        errors = micro_result.get("errors", [])

        status_word = "stopped by user" if sp_cancelled else "complete"
        summary_lines = [
            f"Scale processing {status_word}: {sp_completed} of {sp_total} items processed.",
        ]

        out_strategy = (
            str(scale_progress.get("_output_strategy", "single_file")).strip().lower()
            if scale_progress
            else "single_file"
        )

        if out_strategy == "file_per_item":
            fn_template = (
                str(scale_progress.get("_output_filename_template", "")).strip()
                if scale_progress
                else ""
            )
            summary_lines.append(
                f"Output: {sp_completed} separate files "
                f"(template: {fn_template or 'per-item'})"
            )
        elif out_strategy == "no_file":
            final_act = (
                str(scale_progress.get("_final_action", "reply")).strip()
                if scale_progress
                else "reply"
            )
            if final_act == "api_call":
                summary_lines.append(f"Output: indexed {sp_completed} items to Typesense")
            elif final_act == "email":
                summary_lines.append(f"Output: emailed {sp_completed} items")
            else:
                summary_lines.append(f"Output: {sp_completed} items processed (no file)")
        else:
            # single_file
            _file = output_file
            if not _file and scale_progress:
                _file = scale_progress.get("_output_file", "") or scale_progress.get("_output_file_arg", "")
            if _file:
                summary_lines.append(f"Output file: {_file}")
            else:
                summary_lines.append("Output: single file")

        if sp_failed > 0:
            summary_lines.append(f"Failed: {sp_failed} items")
            for err in errors[:5]:
                summary_lines.append(f"  - {err.get('item', '?')}: {err.get('error', '?')}")

        return "\n".join(summary_lines)

    # ------------------------------------------------------------------
    # Micro-loop takeover orchestration (de-duplicated)
    # ------------------------------------------------------------------

    async def _run_micro_loop_and_summarize(
        self,
        *,
        effective_user_input: str,
        list_task_plan: dict[str, Any],
        turn_usage: dict[str, int],
        session_tool_policy: dict[str, Any] | None,
        planning_pipeline: dict[str, Any] | None,
        step_label: str = "scale_micro_loop_takeover",
        output_file: str = "",
    ) -> dict[str, Any]:
        """Run the scale micro-loop and build a summary.

        Consolidates the repeated pattern of:
        1. Strip advisory from user input → derive per-member action
        2. Emit takeover trace
        3. Run ``_run_scale_micro_loop``
        4. Build summary via ``_build_micro_loop_summary``
        5. Add session message

        Returns the micro-loop result dict augmented with ``"summary"`` key.
        """
        sp = getattr(self, "_scale_progress", None)

        # Derive per-member action
        task_input = self._strip_scale_advisory(effective_user_input)
        per_member_action = task_input[:2000]
        if not per_member_action:
            per_member_action = (
                str(list_task_plan.get("per_member_action", "")).strip()
                or effective_user_input[:500]
            )
        if sp is not None:
            sp["_per_member_action"] = per_member_action

        # Resolve output file
        if not output_file and sp is not None:
            output_file = sp.get("_output_file", "") or ""

        self._emit_tool_output(
            "task_contract",
            {
                "step": step_label,
                "items": len(sp.get("items", [])) if sp else 0,
                "output_strategy": sp.get("_output_strategy", "single_file") if sp else "single_file",
            },
            (
                f"step={step_label}\n"
                f"items={len(sp.get('items', [])) if sp else 0}\n"
                f"note=entering micro loop ({step_label})"
            ),
        )

        active_task_tool_policy = self._active_task_tool_policy_payload(planning_pipeline)
        micro_result = await self._run_scale_micro_loop(
            task_description=per_member_action,
            output_file=output_file,
            turn_usage=turn_usage,
            session_policy=session_tool_policy,
            task_policy=active_task_tool_policy,
        )

        micro_summary = self._build_micro_loop_summary(
            micro_result=micro_result,
            scale_progress=sp,
            output_file=output_file,
        )
        self._add_session_message(role="assistant", content=micro_summary)

        self._emit_tool_output(
            "task_contract",
            {
                "step": f"{step_label}_done",
                "processed": micro_result.get("processed", 0),
                "failed": micro_result.get("failed", 0),
                "total": micro_result.get("total", 0),
                "cancelled": micro_result.get("cancelled", False),
            },
            micro_summary,
        )

        micro_result["summary"] = micro_summary
        return micro_result

    # ------------------------------------------------------------------
    # Deferred scale check helper
    # ------------------------------------------------------------------

    def _needs_deferred_scale_init(self) -> bool:
        """Check whether deferred scale initialization should be attempted.

        Returns True when scale_progress is absent, empty, has fewer than
        2 items, or all items are just source URLs.
        """
        # The orchestrator sets this flag on worker agents for non-scale
        # tasks (combine, send, assemble) to avoid wasting LLM calls on
        # list extraction that will never produce useful results.
        if getattr(self, "_skip_deferred_scale", False):
            return False
        sp = getattr(self, "_scale_progress", None)
        items = sp.get("items", []) if sp else []
        return (
            sp is None
            or not items
            or len(items) < 2
            or self._items_are_source_urls_only(items)
        )
