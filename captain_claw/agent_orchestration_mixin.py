"""Main request orchestration (complete/stream) for Agent."""

import asyncio
import json
import re
from typing import Any, AsyncIterator

from captain_claw.config import get_config
from captain_claw.exceptions import GuardBlockedError
from captain_claw.llm import Message
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


class AgentOrchestrationMixin:

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
        large_from_input = self._input_suggests_large_scale(effective_user_input)

        if not large_from_members and not large_from_input:
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
        # Only attempt once per turn to avoid wasting tokens.
        if getattr(self, "_deferred_scale_attempted", False):
            return list_task_plan
        sp = getattr(self, "_scale_progress", None)
        _existing_items = sp.get("items", []) if sp else []
        if (
            sp is not None
            and len(_existing_items) >= 3
            and not self._items_are_source_urls_only(_existing_items)
        ):
            # Scale already initialized with enough diverse items — nothing to do.
            return list_task_plan
        # Only attempt re-detection if the input suggested list processing.
        if not (
            self._input_suggests_large_scale(effective_user_input)
            or self._is_list_processing_request(effective_user_input)
        ):
            return list_task_plan

        # Re-collect context with generous limits.  The default
        # per_message_chars (1400) truncates fetched articles before the
        # list of items appears, so we use a much larger budget here.
        new_context = self._collect_list_extraction_context(
            max_messages=10,
            max_chars=40000,
            per_message_chars=20000,
        )
        if not new_context or len(new_context) < 200:
            return list_task_plan

        # Mark as attempted so we don't re-run on every iteration.
        self._deferred_scale_attempted = True

        new_plan = await self._generate_list_task_plan(
            user_input=effective_user_input,
            context_excerpt=new_context,
            turn_usage=turn_usage,
        )
        new_members = new_plan.get("members", [])
        if len(new_members) < 3:
            return list_task_plan

        # Initialize scale progress with the newly discovered members.
        _scale_cfg = get_config().scale
        _out_strategy = str(new_plan.get("output_strategy", "single_file")).strip().lower()
        if _out_strategy not in ("file_per_item", "single_file", "no_file"):
            _out_strategy = "single_file"
        self._scale_progress = {
            "total": len(new_members),
            "completed": 0,
            "items": list(new_members),
            "done_items": set(),
            "_output_strategy": _out_strategy,
            "_output_filename_template": str(new_plan.get("output_filename_template", "")).strip(),
            "_final_action": str(new_plan.get("final_action", "write_file")).strip(),
            "_extraction_mode": self._classify_item_extraction_mode(new_members),
            "_member_context": new_plan.get("member_context") or {},
        }
        if _out_strategy == "no_file":
            self._scale_progress["_sink_collection"] = ""
            self._scale_progress["_sink_email_to"] = ""

        # Inject scale advisory into effective user input.
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

    async def complete(self, user_input: str) -> str:
        """Process user input and return response.
        
        Args:
            user_input: User's message
        
        Returns:
            Agent's response
        """
        if not self._initialized:
            await self.initialize()
        self._last_memory_debug_signature = None
        self._last_semantic_memory_debug_signature = None
        restore_skill_env = self._apply_skill_env_overrides_for_run()
        skill_env_restored = False

        turn_usage = self._empty_usage()
        self.last_usage = self._empty_usage()
        # Clear any stale cancel signal from a previous turn so it doesn't
        # immediately abort this new turn.
        cancel_ev = getattr(self, "cancel_event", None)
        if cancel_ev is not None:
            cancel_ev.clear()
        # Reset per-turn duplicate tool call tracker.  This dict maps
        # (tool_name, canonical_args_json) → execution count so that we can
        # detect the LLM re-requesting the exact same tool call and stop it
        # before wasting resources on an infinite re-fetch loop.
        self._turn_tool_call_counts: dict[str, int] = {}
        # Scale-progress tracker: populated when the scale advisory fires.
        # The tool loop uses this to emit "3 of 27 (11%)" progress.
        self._scale_progress: dict[str, Any] | None = None
        self._deferred_scale_attempted: bool = False
        planning_pipeline: dict[str, Any] | None = None
        recent_source_urls: list[str] = []
        effective_user_input = user_input
        effective_user_input, clarification_context_applied = self._resolve_effective_user_input(user_input)
        require_all_sources = self._request_references_all_sources(effective_user_input)
        is_worker = getattr(self, "_is_worker", False)
        use_contract_pipeline = self._should_use_contract_pipeline(
            effective_user_input,
            self.planning_enabled,
            pipeline_mode=self.pipeline_mode,
        )
        if clarification_context_applied:
            # Clarification follow-ups usually represent partially-specified
            # continuations of a larger request; keep strict completion gating.
            use_contract_pipeline = True
        # Workers should never use contract pipelines — they execute a
        # single focused task and should return as soon as it's done.
        if is_worker:
            use_contract_pipeline = False
        explicit_script_request = self._is_explicit_script_request(effective_user_input)
        enforce_python_worker_mode = explicit_script_request and not is_worker
        session_id = self._current_session_slug()
        session_tool_policy = self._session_tool_policy_payload()
        turn_abort_event = asyncio.Event()
        available_tools = {
            name.strip().lower()
            for name in self.tools.list_tools(
                session_id=session_id,
                session_policy=session_tool_policy,
            )
        }
        python_worker_tools_available = {"write", "shell"}.issubset(available_tools)
        python_worker_attempted = False
        list_task_plan: dict[str, Any] = {
            "enabled": False,
            "members": [],
            "strategy": "none",
            "per_member_action": "",
            "confidence": "low",
        }
        task_contract: dict[str, Any] | None = None
        completion_requirements: list[dict[str, Any]] = []
        completion_feedback: str = ""

        def _restore_skill_env_once() -> None:
            nonlocal skill_env_restored
            if skill_env_restored:
                return
            skill_env_restored = True
            try:
                restore_skill_env()
            except Exception:
                pass

        def finish(text: str, success: bool = True) -> str:
            self._emit_thinking("", phase="done")
            if planning_pipeline is not None:
                self._finalize_pipeline(planning_pipeline, success=success)
            self._finalize_turn_usage(turn_usage)
            _restore_skill_env_once()
            return text

        async def attempt_finalize_response(
            output_text: str,
            iteration: int,
            finish_success: bool = True,
        ) -> tuple[bool, str, bool]:
            """Apply auto-write + completion gate before returning final output."""
            nonlocal completion_feedback, python_worker_attempted, list_task_plan
            final_response = await self._maybe_auto_write_requested_output(
                user_input=effective_user_input,
                output_text=output_text,
                turn_start_idx=turn_start_idx,
                turn_usage=turn_usage,
                session_policy=session_tool_policy,
                task_policy=self._active_task_tool_policy_payload(planning_pipeline),
            )
            if not str(final_response or "").strip():
                tool_output_fallback = self._collect_turn_tool_output(turn_start_idx)
                if str(tool_output_fallback or "").strip():
                    final_response = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=tool_output_fallback,
                        turn_usage=turn_usage,
                    )
            if enforce_python_worker_mode:
                worker_ran_this_iteration = False
                has_write = self._turn_has_successful_tool(turn_start_idx, "write")
                has_shell = self._turn_has_successful_tool(turn_start_idx, "shell")
                if not (has_write and has_shell) and not python_worker_attempted:
                    python_worker_attempted = True
                    worker_result = await self._run_python_worker_for_list_task(
                        user_input=effective_user_input,
                        turn_usage=turn_usage,
                        list_task_plan=list_task_plan,
                        planning_pipeline=planning_pipeline,
                        session_policy=session_tool_policy,
                        task_policy=self._active_task_tool_policy_payload(planning_pipeline),
                    )
                    self._emit_tool_output(
                        "task_contract",
                        {
                            "step": "python_worker_autorun",
                            "success": bool(worker_result.get("success", False)),
                            "attempted": True,
                        },
                        json.dumps(worker_result, ensure_ascii=True),
                    )
                    worker_ran_this_iteration = bool(worker_result.get("success", False))
                    has_write = self._turn_has_successful_tool(turn_start_idx, "write")
                    has_shell = self._turn_has_successful_tool(turn_start_idx, "shell")
                if worker_ran_this_iteration and iteration < (hard_turn_iterations - 1):
                    completion_feedback = (
                        "Python worker executed successfully.\n"
                        "Now provide the final answer covering the complete processed list and saved outputs."
                    )
                    return False, "", finish_success
                if not (has_write and has_shell):
                    completion_feedback = (
                        "Completion gate: execute Python worker workflow via tools before finalizing.\n"
                        "- Generate or refine a Python script/tool that handles the full item list.\n"
                        "- Run it through shell.\n"
                        "- Then provide final concise summary."
                    )
                    if iteration < (hard_turn_iterations - 1):
                        return False, "", finish_success
            if bool(list_task_plan.get("enabled", False)):
                members = list_task_plan.get("members")
                if isinstance(members, list) and members:
                    # When scale progress is active and has tracked items
                    # (e.g. from glob), prefer its done_items tracking
                    # over text-based list member coverage.  The scale
                    # progress system tracks real file paths / URLs from
                    # glob output, whereas list_task_plan members may
                    # contain LLM-extracted descriptions like "All PDF
                    # files under pdf-test (including subfolders)" which
                    # are not actual items.
                    _sp = getattr(self, "_scale_progress", None)
                    _sp_items = _sp.get("items", []) if _sp else []
                    _sp_done = _sp.get("done_items", set()) if _sp else set()
                    _sp_total = len(_sp_items)
                    _sp_completed = len(_sp_done)
                    _sp_all_done = (
                        _sp_total > 0
                        and _sp_completed >= _sp_total
                    )
                    if _sp_all_done:
                        # Scale progress confirms all items done — skip
                        # the text-based coverage check which may have
                        # stale / aggregate member names.
                        covered_members = [str(m) for m in members]
                        missing_members: list[str] = []
                    else:
                        # Use scale-progress items for coverage when
                        # they differ from list_task_plan members (e.g.
                        # glob discovered real paths while the LLM
                        # extractor produced descriptive text).
                        eval_members = (
                            [str(m) for m in _sp_items]
                            if _sp_items and len(_sp_items) > len(members)
                            else [str(m) for m in members]
                        )
                        covered_members, missing_members = self._evaluate_list_member_coverage(
                            members=eval_members,
                            candidate_response=final_response,
                            turn_start_idx=turn_start_idx,
                        )
                        # Also cross-reference with scale progress done_items:
                        # items that are in done_items should count as covered
                        # even if the text-based check missed them.
                        if _sp_done and missing_members:
                            still_missing: list[str] = []
                            for m in missing_members:
                                if m in _sp_done:
                                    covered_members.append(m)
                                else:
                                    still_missing.append(m)
                            missing_members = still_missing
                    self._emit_tool_output(
                        "completion_gate",
                        {
                            "step": "list_member_coverage",
                            "covered": len(covered_members),
                            "missing": len(missing_members),
                            "members": len(members),
                            "scale_progress_total": _sp_total,
                            "scale_progress_done": _sp_completed,
                        },
                        (
                            "step=list_member_coverage\n"
                            f"covered={len(covered_members)}\n"
                            f"missing={len(missing_members)}\n"
                            f"members={len(members)}\n"
                            f"scale_progress_total={_sp_total}\n"
                            f"scale_progress_done={_sp_completed}"
                        ),
                    )
                    if missing_members:
                        completion_feedback = self._build_list_coverage_feedback(
                            missing_members=missing_members,
                            strategy=str(list_task_plan.get("strategy", "direct")).strip().lower(),
                            per_member_action=str(list_task_plan.get("per_member_action", "")).strip(),
                        )
                        self._emit_tool_output(
                            "task_contract",
                            {
                                "step": "list_member_retry",
                                "missing": len(missing_members),
                            },
                            completion_feedback,
                        )
                        if iteration < (hard_turn_iterations - 1):
                            return False, "", finish_success
            if completion_requirements and task_contract is not None:
                critique = await self._evaluate_contract_completion(
                    user_input=effective_user_input,
                    candidate_response=final_response,
                    contract=task_contract,
                    turn_usage=turn_usage,
                )
                checks_ok = bool(critique.get("complete", False))
                raw_check_results = critique.get("checks", [])
                check_results = raw_check_results if isinstance(raw_check_results, list) else []
                failed_items = [item for item in check_results if not bool(item.get("ok", False))]
                failure_reasons = ", ".join(
                    f"{item.get('id', '')}: {item.get('reason', '')}" for item in failed_items
                ) or "none"
                self._emit_tool_output(
                    "completion_gate",
                    {
                        "step": "validation",
                        "passed": checks_ok,
                        "failed_count": len(failed_items),
                    },
                    (
                        f"step=validation\n"
                        f"passed={checks_ok}\n"
                        f"failed_count={len(failed_items)}\n"
                        f"reasons={failure_reasons}"
                    ),
                )
                if planning_pipeline is not None:
                    task_order = self._refresh_pipeline_task_order(planning_pipeline)
                    if task_order:
                        self._set_pipeline_progress(
                            planning_pipeline,
                            current_index=len(task_order) - 1,
                            current_status="completed" if checks_ok else "in_progress",
                        )
                    self._update_pipeline_checks(planning_pipeline, check_results)
                    self._emit_pipeline_update(
                        "validation_passed" if checks_ok else "validation_retry",
                        planning_pipeline,
                    )
                if not checks_ok:
                    completion_feedback = self._build_completion_feedback(
                        contract=task_contract,
                        critique=critique,
                    )
                    self._emit_tool_output(
                        "task_contract",
                        {
                            "step": "validation_retry",
                            "missing": len(failed_items),
                        },
                        completion_feedback,
                    )
                    if iteration < (hard_turn_iterations - 1):
                        return False, "", finish_success

            self._update_clarification_state(
                user_input=user_input,
                effective_user_input=effective_user_input,
                assistant_response=final_response,
            )
            await self._persist_assistant_response(final_response)
            await self._auto_capture_todos(effective_user_input, final_response)
            await self._auto_capture_contacts(effective_user_input, final_response)
            await self._auto_capture_scripts(effective_user_input, final_response)
            await self._auto_capture_apis(effective_user_input, final_response)
            return True, final_response, finish_success

        turn_start_idx = len(self.session.messages) if self.session else 0

        # Compute a domain filter so that _collect_recent_source_urls only
        # returns URLs relevant to the current request.  We extract domains
        # from the effective user input *and* the last assistant response
        # (which contains the specific items the user is referring to in
        # follow-up / clarification scenarios).
        domain_filter = self._extract_mentioned_domains(effective_user_input)
        if self.session and self.session.messages:
            for msg in reversed(self.session.messages):
                if msg.get("role") == "assistant":
                    assistant_text = str(msg.get("content", ""))
                    domain_filter |= self._extract_mentioned_domains(assistant_text)
                    break
        recent_source_urls = self._collect_recent_source_urls(
            turn_start_idx, domain_filter=domain_filter or None,
        )
        allowed_user_input, input_guard_error = await self._enforce_guard(
            guard_type="input",
            interaction_label="user_turn",
            content=user_input,
            turn_usage=turn_usage,
        )
        if not allowed_user_input:
            return finish(input_guard_error, success=False)

        # Add user message to session
        self._add_session_message("user", user_input)
        await self._auto_compact_if_needed()
        await self._refresh_todo_context_cache()
        await self._refresh_contacts_context_cache()
        await self._refresh_scripts_context_cache()
        await self._refresh_apis_context_cache()
        if clarification_context_applied:
            self._emit_tool_output(
                "task_contract",
                {"step": "clarification_context_applied"},
                "step=clarification_context_applied\nstatus=merged_pending_anchor_into_current_turn",
            )
        # ── Automatic task rephrasing ──────────────────────────────
        # For complex user prompts (list processing with formatting
        # details, multiple URLs, output specifications), rephrase into
        # a structured format that downstream components can parse more
        # reliably.  Skip for workers and clarification follow-ups.
        task_was_rephrased = False
        if not is_worker and not clarification_context_applied:
            effective_user_input, task_was_rephrased = await self._rephrase_task(
                user_input=effective_user_input,
                turn_usage=turn_usage,
            )
            if task_was_rephrased:
                # Re-check require_all_sources with the rephrased input
                # (unlikely to change, but keeps things consistent).
                require_all_sources = self._request_references_all_sources(effective_user_input)
        list_context_excerpt = self._collect_list_extraction_context()
        # Workers execute a single focused task — skip the heavyweight list
        # task extraction / coverage pipeline which can cause endless loops
        # on simple fetch-and-summarize instructions.
        if getattr(self, "_is_worker", False):
            list_task_plan = list_task_plan  # keep default (disabled)
        else:
            list_task_plan = await self._generate_list_task_plan(
                user_input=effective_user_input,
                context_excerpt=list_context_excerpt,
                turn_usage=turn_usage,
            )
        # Direct URL extraction fallback: when the user pastes many URLs
        # in their message, the LLM list extractor (1000 max_tokens) may
        # not be able to return all of them as JSON members.  Detect this
        # and augment the list_task_plan with directly-extracted URLs.
        # IMPORTANT: use the *original* user input when a clarification
        # context was merged, otherwise the assistant's previous response
        # (which may list many URLs) leaks into the member list and causes
        # unwanted scale-loop processing of all items.
        url_extraction_source = user_input if clarification_context_applied else effective_user_input
        input_urls = self._extract_urls(url_extraction_source)
        if len(input_urls) > len(list_task_plan.get("members", [])):
            existing_members = set(
                str(m).strip() for m in list_task_plan.get("members", [])
            )
            augmented = list(list_task_plan.get("members", []))
            for url in input_urls:
                if url not in existing_members:
                    augmented.append(url)
                    existing_members.add(url)
            if len(augmented) > len(list_task_plan.get("members", [])):
                list_task_plan["members"] = augmented[:150]
                list_task_plan["enabled"] = True
                if not list_task_plan.get("per_member_action"):
                    list_task_plan["per_member_action"] = "fetch and process"
                self._emit_tool_output(
                    "task_contract",
                    {
                        "step": "list_members_augmented_from_input_urls",
                        "llm_extracted": len(existing_members),
                        "augmented_total": len(augmented),
                    },
                    (
                        "step=list_members_augmented_from_input_urls\n"
                        f"llm_extracted={len(existing_members)}\n"
                        f"augmented_total={len(augmented)}"
                    ),
                )
        extracted_strategy = str(list_task_plan.get("strategy", "none")).strip().lower()
        if extracted_strategy == "script" and not explicit_script_request:
            self._emit_tool_output(
                "task_contract",
                {"step": "python_worker_mode_skipped", "reason": "prefer_internal_tools"},
                "step=python_worker_mode_skipped\nreason=prefer_internal_tools\nmode=direct",
            )
        if enforce_python_worker_mode and not python_worker_tools_available:
            enforce_python_worker_mode = False
            self._emit_tool_output(
                "task_contract",
                {"step": "python_worker_mode_skipped", "reason": "missing_tools"},
                "step=python_worker_mode_skipped\nreason=missing_tools\nrequired=write,shell",
            )
        if bool(list_task_plan.get("enabled", False)):
            self._emit_tool_output(
                "task_contract",
                {
                    "step": "list_task_memory_enabled",
                    "members": len(list_task_plan.get("members", [])),
                    "strategy": extracted_strategy,
                },
                (
                    "step=list_task_memory_enabled\n"
                    f"members={len(list_task_plan.get('members', []))}\n"
                    f"strategy={extracted_strategy}"
                ),
            )
        if enforce_python_worker_mode:
            self._emit_tool_output(
                "task_contract",
                {"step": "python_worker_mode_enabled", "strategy": extracted_strategy or "script"},
                "step=python_worker_mode_enabled\nmode=python_worker_tool_execution",
            )
        # --- Pre-flight scale check ---
        # Detect tasks that will produce a large number of items and inject
        # an advisory so the LLM (and planner) adopt the incremental
        # append-to-file strategy rather than trying to hold everything in
        # the context window.
        scale_advisory = self._preflight_scale_check(effective_user_input, list_task_plan)
        if scale_advisory:
            effective_user_input = effective_user_input + scale_advisory
            # Activate scale-progress tracking so _handle_tool_calls can
            # count glob results and write(append) calls and emit "3 of 27"
            # progress indicators to the thinking line.
            self._scale_progress = {"total": 0, "completed": 0}
            # Store output strategy so the micro-loop knows whether to
            # write per-item files, a single file, or skip file output.
            _out_strategy = str(list_task_plan.get("output_strategy", "single_file")).strip().lower()
            self._scale_progress["_output_strategy"] = _out_strategy
            self._scale_progress["_output_filename_template"] = str(
                list_task_plan.get("output_filename_template", "")
            ).strip()
            self._scale_progress["_final_action"] = str(
                list_task_plan.get("final_action", "write_file")
            ).strip()
            # Sink metadata for no_file output strategy.
            if _out_strategy == "no_file":
                self._scale_progress["_sink_collection"] = ""  # resolved at runtime
                self._scale_progress["_sink_email_to"] = ""     # resolved at runtime
            # Pre-populate items from list_task_plan members when available.
            # This gives the scale progress note an initial worklist so the
            # LLM sees "REMAINING items to process" even before glob runs.
            # When glob later discovers the actual file list, it overwrites
            # these items with the real paths.
            list_members = list_task_plan.get("members", [])
            if list_members:
                self._scale_progress["items"] = list(list_members)
                self._scale_progress["done_items"] = set()
                self._scale_progress["total"] = len(list_members)
                self._scale_progress["_extraction_mode"] = self._classify_item_extraction_mode(list_members)
                self._scale_progress["_member_context"] = list_task_plan.get("member_context") or {}
        if use_contract_pipeline:
            # When a clarification context was merged, the relevant URLs are
            # already embedded in effective_user_input.  Passing the full
            # recent_source_urls (which may contain dozens of navigation/
            # category links from previous web_fetch content) pollutes the
            # planner and causes wasteful prefetching of unrelated URLs.
            planner_source_urls = [] if clarification_context_applied else recent_source_urls
            task_contract = await self._generate_task_contract(
                user_input=effective_user_input,
                recent_source_urls=planner_source_urls,
                require_all_sources=require_all_sources,
                turn_usage=turn_usage,
                list_task_plan=list_task_plan,
            )
            completion_requirements = self._apply_list_requirements(
                base_requirements=list(task_contract.get("requirements", [])),
                list_task_plan=list_task_plan,
            )
            task_contract["requirements"] = completion_requirements
            prefetch_urls = [
                url
                for url in list(task_contract.get("prefetch_urls", []))
                if isinstance(url, str) and url.startswith(("http://", "https://"))
            ]
            # Skip prefetch when scale progress already has items — the
            # micro loop will fetch each item one at a time, so batch-
            # prefetching would be redundant and waste context/tokens.
            sp = getattr(self, "_scale_progress", None)
            _skip_prefetch = sp is not None and bool(sp.get("items"))
            if prefetch_urls and not _skip_prefetch:
                await self._run_source_report_prefetch(
                    source_urls=prefetch_urls,
                    turn_usage=turn_usage,
                    pipeline_label="task_contract",
                )
            elif prefetch_urls and _skip_prefetch:
                self._emit_tool_output(
                    "task_contract",
                    {
                        "step": "prefetch_skipped",
                        "reason": "scale_progress_has_items",
                        "prefetch_urls": len(prefetch_urls),
                        "scale_items": len(sp.get("items", [])),
                    },
                    (
                        "step=prefetch_skipped\n"
                        f"reason=scale_progress_has_items\n"
                        f"prefetch_urls={len(prefetch_urls)}\n"
                        f"scale_items={len(sp.get('items', []))}\n"
                        "note=micro loop will fetch each item individually"
                    ),
                )
            # If scale progress is active but has no items yet (e.g. list
            # members were empty, waiting for glob), and the contract
            # produced prefetch_urls, seed items from those URLs.  This
            # covers URL-based list tasks where the "items" are web pages.
            if sp is not None and not sp.get("items") and prefetch_urls:
                sp["items"] = list(prefetch_urls)
                sp["done_items"] = set()
                sp["total"] = len(prefetch_urls)
                sp["_extraction_mode"] = self._classify_item_extraction_mode(prefetch_urls)
        if self.planning_enabled or task_contract is not None:
            planning_pipeline = self._build_task_pipeline(
                effective_user_input,
                tasks_override=(task_contract or {}).get("tasks"),
                completion_checks=completion_requirements,
            )
            if self.planning_enabled and task_contract is not None:
                planning_pipeline["mode"] = "manual_with_contract"
            elif self.planning_enabled:
                planning_pipeline["mode"] = "manual"
            else:
                planning_pipeline["mode"] = "auto_contract"
            self._emit_pipeline_update("created", planning_pipeline)
            created_children = await self.ensure_pipeline_subagent_contexts(planning_pipeline)
            if created_children:
                self._emit_tool_output(
                    "planning",
                    {"event": "subagent_contexts_spawned", "count": len(created_children)},
                    (
                        "event=subagent_contexts_spawned\n"
                        f"count={len(created_children)}"
                    ),
                )
        
        # Activate lightweight scale-progress tracking for moderate-size
        # list tasks (≥5 members) even when the full scale advisory didn't
        # fire.  The progress note keeps the LLM on track by showing
        # remaining items at every iteration — preventing re-glob and
        # context-loss issues.  This generalizes the scale progress system
        # beyond large-item-count tasks and glob-based lists.
        _lw_min = get_config().scale.lightweight_progress_min_members
        if (
            self._scale_progress is None
            and bool(list_task_plan.get("enabled", False))
            and len(list_task_plan.get("members", [])) >= _lw_min
        ):
            members = list_task_plan.get("members", [])
            _lw_out_strategy = str(list_task_plan.get("output_strategy", "single_file")).strip().lower()
            self._scale_progress = {
                "total": len(members),
                "completed": 0,
                "items": list(members),
                "done_items": set(),
                "_output_strategy": _lw_out_strategy,
                "_output_filename_template": str(list_task_plan.get("output_filename_template", "")).strip(),
                "_final_action": str(list_task_plan.get("final_action", "write_file")).strip(),
                "_extraction_mode": self._classify_item_extraction_mode(members),
                "_member_context": list_task_plan.get("member_context") or {},
            }
            if _lw_out_strategy == "no_file":
                self._scale_progress["_sink_collection"] = ""
                self._scale_progress["_sink_email_to"] = ""
            self._emit_tool_output(
                "task_contract",
                {
                    "step": "scale_progress_from_list_task",
                    "members": len(members),
                },
                (
                    "step=scale_progress_from_list_task\n"
                    f"members={len(members)}\n"
                    "note=activated lightweight progress tracking for moderate list"
                ),
            )

        # ── Early micro-loop takeover ────────────────────────────
        # When scale progress is active with pre-populated items and a
        # clear per-member action, skip the main LLM loop entirely and
        # go straight into the micro loop.  This avoids wasting context
        # on a full LLM pass that would only process 1 item before the
        # completion gate retries.
        #
        # Requirements:
        # - scale_progress active with items
        # - per_member_action known (from list_task_plan)
        # - output strategy is file_per_item with a template, OR
        #   any strategy where the micro loop can handle the first item
        #   without prior LLM output (is_first_item=True path)
        _sp_early = getattr(self, "_scale_progress", None)
        _early_items = _sp_early.get("items", []) if _sp_early else []
        # Use the full user input as the task description so the micro-
        # loop LLM receives complete formatting instructions (CSV headers,
        # column mappings, etc.).  The short per_member_action is kept as
        # a prefix hint but the full prompt ensures fidelity.
        _per_action = str(list_task_plan.get("per_member_action", "")).strip()
        # Strip the scale advisory from effective_user_input (it's between
        # "--- SCALE ADVISORY" and "--- END SCALE ADVISORY ---") since
        # it contains instructions for the outer loop, not the micro loop.
        _task_input = effective_user_input
        _adv_start = _task_input.find("\n\n--- SCALE ADVISORY")
        if _adv_start > 0:
            _adv_end = _task_input.find("--- END SCALE ADVISORY ---")
            if _adv_end > _adv_start:
                _task_input = _task_input[:_adv_start].rstrip()
        _early_action = _task_input[:2000]
        if not _early_action:
            _early_action = _per_action or "Process each item"
        _early_output_strategy = str(
            _sp_early.get("_output_strategy", "single_file")
        ).strip().lower() if _sp_early else "single_file"
        _early_fn_template = str(
            _sp_early.get("_output_filename_template", "")
        ).strip() if _sp_early else ""

        _can_early_takeover = (
            _sp_early is not None
            and len(_early_items) >= 3
            and bool(_early_action)
            # file_per_item with template is fully self-sufficient
            # single_file and no_file are also OK — the micro loop's
            # is_first_item=True path handles the first item without
            # needing a prior LLM write.
            #
            # IMPORTANT: Do NOT take over when items are just source URLs
            # (the article URL repeated).  The real list members haven't
            # been extracted yet — the article needs to be fetched first.
            and not self._items_are_source_urls_only(_early_items)
        )
        if _can_early_takeover:
            # Store the per_member_action so the micro loop can use it.
            _sp_early["_per_member_action"] = _early_action

            self._emit_tool_output(
                "task_contract",
                {
                    "step": "early_scale_micro_loop_takeover",
                    "items": len(_early_items),
                    "output_strategy": _early_output_strategy,
                    "filename_template": _early_fn_template,
                },
                (
                    "step=early_scale_micro_loop_takeover\n"
                    f"items={len(_early_items)}\n"
                    f"output_strategy={_early_output_strategy}\n"
                    f"filename_template={_early_fn_template}\n"
                    "note=skipping main LLM loop, entering micro loop directly"
                ),
            )

            active_task_tool_policy = self._active_task_tool_policy_payload(planning_pipeline)
            micro_result = await self._run_scale_micro_loop(
                task_description=_early_action,
                output_file="",  # micro loop handles file creation
                turn_usage=turn_usage,
                session_policy=session_tool_policy,
                task_policy=active_task_tool_policy,
            )

            _sp_total = micro_result.get("total", 0)
            _sp_processed = micro_result.get("processed", 0)
            _sp_failed = micro_result.get("failed", 0)
            _sp_completed = micro_result.get("completed_total", _sp_processed)
            _sp_cancelled = micro_result.get("cancelled", False)
            _micro_errors = micro_result.get("errors", [])

            _status_word = "stopped by user" if _sp_cancelled else "complete"
            summary_lines = [
                f"Scale processing {_status_word}: {_sp_completed} of {_sp_total} items processed.",
            ]
            if _early_output_strategy == "file_per_item":
                summary_lines.append(
                    f"Output: {_sp_completed} separate files "
                    f"(template: {_early_fn_template or 'per-item'})"
                )
            elif _early_output_strategy == "no_file":
                _final_act = str(_sp_early.get("_final_action", "reply")).strip()
                summary_lines.append(f"Output: {_sp_completed} items ({_final_act})")
            else:
                _output_file = _sp_early.get("_output_file", "") or _sp_early.get("_output_file_arg", "")
                summary_lines.append(f"Output file: {_output_file}")
            if _sp_failed > 0:
                summary_lines.append(f"Failed: {_sp_failed} items")
                for err in _micro_errors[:5]:
                    summary_lines.append(f"  - {err.get('item', '?')}: {err.get('error', '?')}")
            micro_summary = "\n".join(summary_lines)

            self._add_session_message(role="assistant", content=micro_summary)

            self._emit_tool_output(
                "task_contract",
                {
                    "step": "early_scale_micro_loop_done",
                    "processed": _sp_processed,
                    "failed": _sp_failed,
                    "total": _sp_total,
                    "cancelled": _sp_cancelled,
                },
                micro_summary,
            )

            self._update_clarification_state(
                user_input=user_input,
                effective_user_input=effective_user_input,
                assistant_response=micro_summary,
            )
            await self._persist_assistant_response(micro_summary)
            return finish(micro_summary, success=micro_result.get("success", False))

        # Main agent loop
        base_turn_iterations = self.max_iterations + (2 if completion_requirements else 0)
        planned_turn_iterations = self._compute_turn_iteration_budget(
            base_iterations=base_turn_iterations,
            planning_pipeline=planning_pipeline,
            completion_requirements=completion_requirements,
        )
        # When the scale advisory fired, the task is known to involve many
        # items. Boost the iteration budget so the LLM has enough room to
        # process each item (read + append ≈ 2 iterations per item, plus
        # overhead for glob/setup/finalize).
        if scale_advisory:
            member_count = len(list_task_plan.get("members", []))
            # If we know the count from list extraction, use it;
            # otherwise assume a generous default (we'll rely on the
            # extension mechanism for the rest).
            estimated_items = member_count if member_count > 15 else 50
            # Budget: ~2-3 iterations per item + overhead.  Cap at 400
            # to allow lists up to ~150 items within a single turn.
            scale_budget = min(400, 10 + estimated_items * 3)
            if scale_budget > planned_turn_iterations:
                planned_turn_iterations = scale_budget
        hard_turn_iterations = max(planned_turn_iterations, min(500, planned_turn_iterations * 3))
        soft_turn_iterations = planned_turn_iterations
        extension_step = max(6, min(24, max(1, planned_turn_iterations // 3)))
        max_stagnant_iterations = 6
        stagnant_iterations = 0
        progress_window: list[bool] = []
        previous_progress_snapshot: dict[str, Any] | None = None
        last_completion_feedback_signature = ""
        if planned_turn_iterations != base_turn_iterations:
            self._emit_tool_output(
                "completion_gate",
                {
                    "step": "iteration_budget",
                    "base_limit": base_turn_iterations,
                    "effective_limit": planned_turn_iterations,
                    "hard_limit": hard_turn_iterations,
                },
                (
                    "step=iteration_budget\n"
                    f"base_limit={base_turn_iterations}\n"
                    f"effective_limit={planned_turn_iterations}\n"
                    f"hard_limit={hard_turn_iterations}"
                ),
            )
        for iteration in range(hard_turn_iterations):
            # ── External cancellation check ──────────────────────────
            # If the UI layer (TUI Ctrl+C / web cancel) has signalled
            # cancellation, break out cleanly rather than running all
            # remaining iterations.
            cancel_ev: asyncio.Event | None = getattr(self, "cancel_event", None)
            if cancel_ev is not None and cancel_ev.is_set():
                self._set_runtime_status("waiting")
                self._emit_thinking("Cancelled", phase="done")
                cancel_ev.clear()  # reset for the next turn
                return finish(
                    "Request cancelled by user.",
                    success=False,
                )
            if planning_pipeline is not None:
                runtime_update = self._tick_pipeline_runtime(
                    planning_pipeline,
                    event=f"runtime_tick_{iteration + 1}",
                )
                if bool(runtime_update.get("changed", False)):
                    activated = runtime_update.get("activated", [])
                    if isinstance(activated, list) and activated:
                        await self.ensure_pipeline_subagent_contexts(
                            planning_pipeline,
                            task_ids=[str(item) for item in activated],
                        )
                    self._emit_pipeline_update("runtime_update", planning_pipeline)
                if str(planning_pipeline.get("state", "")).strip().lower() == "failed":
                    self._set_runtime_status("waiting")
                    return finish(
                        "Task pipeline failed after timeout/retry exhaustion. Could not complete the request.",
                        success=False,
                    )
            if iteration >= soft_turn_iterations:
                recent_progress = any(progress_window[-4:])
                remaining_work = (
                    self._pipeline_has_remaining_work(planning_pipeline)
                    or bool(completion_feedback)
                    or bool(completion_requirements)
                )
                if recent_progress and remaining_work and soft_turn_iterations < hard_turn_iterations:
                    previous_limit = soft_turn_iterations
                    soft_turn_iterations = min(hard_turn_iterations, soft_turn_iterations + extension_step)
                    self._emit_tool_output(
                        "completion_gate",
                        {
                            "step": "iteration_budget_extended",
                            "previous_limit": previous_limit,
                            "new_limit": soft_turn_iterations,
                            "hard_limit": hard_turn_iterations,
                        },
                        (
                            "step=iteration_budget_extended\n"
                            f"previous_limit={previous_limit}\n"
                            f"new_limit={soft_turn_iterations}\n"
                            f"hard_limit={hard_turn_iterations}"
                        ),
                    )
                else:
                    self._emit_tool_output(
                        "completion_gate",
                        {
                            "step": "iteration_budget_exhausted",
                            "soft_limit": soft_turn_iterations,
                            "hard_limit": hard_turn_iterations,
                            "recent_progress": recent_progress,
                            "remaining_work": remaining_work,
                        },
                        (
                            "step=iteration_budget_exhausted\n"
                            f"soft_limit={soft_turn_iterations}\n"
                            f"hard_limit={hard_turn_iterations}\n"
                            f"recent_progress={recent_progress}\n"
                            f"remaining_work={remaining_work}"
                        ),
                    )
                    self._set_runtime_status("waiting")
                    return finish("Max iterations reached. Could not complete the request.", success=False)
            current_snapshot = self._capture_turn_progress_snapshot(turn_start_idx, planning_pipeline)
            if previous_progress_snapshot is not None:
                snapshot_progress = self._has_turn_progress(previous_progress_snapshot, current_snapshot)
                completion_feedback_signature = completion_feedback.strip()
                feedback_progress = bool(
                    completion_feedback_signature
                    and completion_feedback_signature != last_completion_feedback_signature
                )
                if feedback_progress:
                    last_completion_feedback_signature = completion_feedback_signature
                progressed = snapshot_progress or feedback_progress
                progress_window.append(progressed)
                if progressed:
                    stagnant_iterations = 0
                else:
                    stagnant_iterations += 1
                    if (
                        stagnant_iterations >= max_stagnant_iterations
                        and iteration >= max(2, min(base_turn_iterations, soft_turn_iterations) // 2)
                    ):
                        self._emit_tool_output(
                            "completion_gate",
                            {
                                "step": "stuck_detected",
                                "stagnant_iterations": stagnant_iterations,
                                "iteration": iteration + 1,
                            },
                            (
                                "step=stuck_detected\n"
                                f"iteration={iteration + 1}\n"
                                f"stagnant_iterations={stagnant_iterations}"
                            ),
                        )
                        self._set_runtime_status("waiting")
                        return finish(
                            "Stopped after repeated non-progress iterations. Could not complete the request.",
                            success=False,
                        )
            else:
                progress_window.append(False)
            previous_progress_snapshot = current_snapshot
            self._set_runtime_status("thinking")
            # Build messages for LLM
            messages = self._build_messages(
                tool_messages_from_index=turn_start_idx,
                query=effective_user_input,
                planning_pipeline=planning_pipeline,
                list_task_plan=list_task_plan,
            )
            if completion_feedback:
                messages.append(
                    Message(
                        role="user",
                        content=completion_feedback,
                    )
                )

            active_task_tool_policy = self._active_task_tool_policy_payload(planning_pipeline)
            tool_defs = self.tools.get_definitions(
                session_id=session_id,
                session_policy=session_tool_policy,
                task_policy=active_task_tool_policy,
            )
            log.debug(
                "Tool definitions available",
                count=len(
                    self.tools.list_tools(
                        session_id=session_id,
                        session_policy=session_tool_policy,
                        task_policy=active_task_tool_policy,
                    )
                ),
                tools_sent=bool(tool_defs),
            )
            
            # Call LLM — log context size for diagnostics.
            _ctx = getattr(self, "last_context_window", {})
            _ctx_tokens = int(_ctx.get("prompt_tokens", 0))
            _ctx_budget = int(_ctx.get("context_budget_tokens", 1))
            _ctx_pct = round(_ctx_tokens / _ctx_budget * 100, 1) if _ctx_budget else 0
            _ctx_kb = round(_ctx_tokens * 4 / 1024, 1)  # ~4 bytes/token estimate
            _session_msgs = len(self.session.messages) if self.session else 0
            log.info(
                "Calling LLM",
                iteration=iteration + 1,
                message_count=len(messages),
                session_messages=_session_msgs,
                context_tokens=_ctx_tokens,
                context_kb=_ctx_kb,
                context_budget=_ctx_budget,
                context_pct=f"{_ctx_pct}%",
                dropped=int(_ctx.get("dropped_messages", 0)),
            )
            try:
                response = await self._complete_with_guards(
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                    interaction_label=f"turn_{iteration + 1}",
                    turn_usage=turn_usage,
                )
            except GuardBlockedError as e:
                final = str(e)
                self._update_clarification_state(
                    user_input=user_input,
                    effective_user_input=effective_user_input,
                    assistant_response=final,
                )
                await self._persist_assistant_response(final)
                return finish(final, success=False)
            except Exception as e:
                # Check if this is a 500 error after tool execution
                error_str = str(e)
                tool_output = self._collect_turn_tool_output(turn_start_idx)
                
                # If we have tool messages AND got a 500 error, return tool output
                # This handles the case where Ollama can't process tool results in context
                if tool_output and "500" in error_str:
                    log.warning("Tool result call failed (500), returning tool output")
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=tool_output,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=final,
                        iteration=iteration,
                        finish_success=False,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                
                log.error("LLM call failed", error=str(e), exc_info=True)
                _restore_skill_env_once()
                raise
            self._record_pipeline_task_usage(
                planning_pipeline,
                response.usage if isinstance(getattr(response, "usage", None), dict) else {},
            )
            
            # Check for explicit tool calls (for models that support it)
            if response.tool_calls:
                log.info("Tool calls detected", count=len(response.tool_calls), calls=response.tool_calls)
                self._add_session_message(
                    role="assistant",
                    content=str(response.content or ""),
                    tool_calls=self._serialize_tool_calls_for_session(response.tool_calls),
                )
                await self._handle_tool_calls(
                    response.tool_calls,
                    turn_usage=turn_usage,
                    session_policy=session_tool_policy,
                    task_policy=active_task_tool_policy,
                    abort_event=turn_abort_event,
                )
                if planning_pipeline is not None:
                    activated = self._advance_pipeline(planning_pipeline, event="tool_calls_completed")
                    if activated:
                        await self.ensure_pipeline_subagent_contexts(
                            planning_pipeline,
                            task_ids=[str(item) for item in activated],
                        )
                # ── Deferred scale init after fetch ──────────────────
                # When the user says "fetch article, there is a list",
                # the article content is now in session messages.  Re-run
                # list extraction if scale wasn't initialized yet, has
                # no items (preflight set total=0 shell), or has very
                # few items (< 3) which likely means the initial
                # extractor only saw the source URL, not the real list.
                _sp_pre = getattr(self, "_scale_progress", None)
                _pre_items = _sp_pre.get("items", []) if _sp_pre else []
                _needs_deferred = (
                    _sp_pre is None
                    or not _pre_items
                    or len(_pre_items) < 3
                    # Also re-extract when existing items are just the source
                    # URL repeated — they're not real list members.
                    or self._items_are_source_urls_only(_pre_items)
                )
                if _needs_deferred:
                    list_task_plan = await self._deferred_scale_init(
                        effective_user_input=effective_user_input,
                        list_task_plan=list_task_plan,
                        turn_usage=turn_usage,
                    )
                # If deferred init just populated scale_progress with items,
                # jump straight into the micro loop (like early takeover).
                _sp_deferred = getattr(self, "_scale_progress", None)
                if (
                    _needs_deferred
                    and _sp_deferred is not None
                    and len(_sp_deferred.get("items", [])) >= 3
                ):
                    _def_task = effective_user_input
                    _def_sa = _def_task.find("\n\n--- SCALE ADVISORY")
                    if _def_sa > 0:
                        _def_end = _def_task.find("--- END SCALE ADVISORY ---")
                        if _def_end > _def_sa:
                            _def_task = _def_task[:_def_sa].rstrip()
                    _def_action = _def_task[:2000] or str(list_task_plan.get("per_member_action", "")).strip() or "Process each item"
                    _sp_deferred["_per_member_action"] = _def_action
                    self._emit_tool_output(
                        "task_contract",
                        {
                            "step": "deferred_scale_micro_loop_takeover",
                            "items": len(_sp_deferred.get("items", [])),
                            "output_strategy": _sp_deferred.get("_output_strategy", "single_file"),
                        },
                        (
                            "step=deferred_scale_micro_loop_takeover\n"
                            f"items={len(_sp_deferred.get('items', []))}\n"
                            "note=entering micro loop after deferred scale init"
                        ),
                    )
                    active_task_tool_policy = self._active_task_tool_policy_payload(planning_pipeline)
                    micro_result = await self._run_scale_micro_loop(
                        task_description=_def_action,
                        output_file="",
                        turn_usage=turn_usage,
                        session_policy=session_tool_policy,
                        task_policy=active_task_tool_policy,
                    )
                    _sp_total = micro_result.get("total", 0)
                    _sp_completed = micro_result.get("completed_total", 0)
                    _sp_cancelled = micro_result.get("cancelled", False)
                    _sp_failed = micro_result.get("failed", 0)
                    _status_word = "stopped by user" if _sp_cancelled else "complete"
                    _out_strategy = str(_sp_deferred.get("_output_strategy", "single_file")).strip().lower()
                    summary_lines = [
                        f"Scale processing {_status_word}: {_sp_completed} of {_sp_total} items processed.",
                    ]
                    if _out_strategy == "file_per_item":
                        _fn_template = str(_sp_deferred.get("_output_filename_template", "")).strip()
                        summary_lines.append(f"Output: {_sp_completed} separate files (template: {_fn_template or 'per-item'})")
                    elif _out_strategy == "no_file":
                        _final_act = str(_sp_deferred.get("_final_action", "reply")).strip()
                        summary_lines.append(f"Output: {_sp_completed} items ({_final_act})")
                    else:
                        _def_outfile = _sp_deferred.get("_output_file", "")
                        summary_lines.append(f"Output file: {_def_outfile}" if _def_outfile else "Output: single file")
                    if _sp_failed > 0:
                        summary_lines.append(f"Failed: {_sp_failed} items")
                    micro_summary = "\n".join(summary_lines)
                    self._add_session_message(role="assistant", content=micro_summary)
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=micro_summary,
                        iteration=iteration,
                        finish_success=micro_result.get("success", False),
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                # ── Micro-turn scale loop takeover ──────────────────
                # After the LLM has established the format (1+ items done,
                # output file known), switch to direct-execution mode for
                # all remaining items.  This prevents the context from
                # growing linearly and keeps each LLM call at constant size.
                if self._scale_loop_ready():
                    sp = getattr(self, "_scale_progress", None)
                    output_file = sp.get("_output_file", "") if sp else ""
                    # Use the full user input as the task description so
                    # the micro-loop LLM gets complete formatting instructions.
                    # Strip the scale advisory portion if present.
                    _task_src = effective_user_input
                    _sa_start = _task_src.find("\n\n--- SCALE ADVISORY")
                    if _sa_start > 0:
                        _sa_end = _task_src.find("--- END SCALE ADVISORY ---")
                        if _sa_end > _sa_start:
                            _task_src = _task_src[:_sa_start].rstrip()
                    per_member_action = _task_src[:2000]
                    if not per_member_action:
                        per_member_action = str(
                            list_task_plan.get("per_member_action", "")
                        ).strip() or effective_user_input[:500]
                    # Store on scale_progress so the micro-loop can access it.
                    if sp is not None:
                        sp["_per_member_action"] = per_member_action

                    self._emit_tool_output(
                        "task_contract",
                        {
                            "step": "scale_micro_loop_takeover",
                            "remaining": len(sp.get("items", [])) - len(sp.get("done_items", set())),
                            "total": len(sp.get("items", [])),
                            "output_file": output_file,
                        },
                        (
                            "step=scale_micro_loop_takeover\n"
                            f"remaining={len(sp.get('items', [])) - len(sp.get('done_items', set()))}\n"
                            f"total={len(sp.get('items', []))}\n"
                            f"output_file={output_file}"
                        ),
                    )
                    log.info(
                        "Scale micro-loop takeover",
                        remaining=len(sp.get("items", [])) - len(sp.get("done_items", set())),
                        total=len(sp.get("items", [])),
                    )

                    micro_result = await self._run_scale_micro_loop(
                        task_description=per_member_action,
                        output_file=output_file,
                        turn_usage=turn_usage,
                        session_policy=session_tool_policy,
                        task_policy=active_task_tool_policy,
                    )

                    # Build a final summary from the micro-loop results.
                    _sp_total = micro_result.get("total", 0)
                    _sp_processed = micro_result.get("processed", 0)
                    _sp_failed = micro_result.get("failed", 0)
                    _sp_completed = micro_result.get("completed_total", _sp_processed)
                    _sp_cancelled = micro_result.get("cancelled", False)
                    _micro_errors = micro_result.get("errors", [])

                    # Record a synthetic session message so the completion
                    # gate can see that the work was done.
                    _out_strategy = str(sp.get("_output_strategy", "single_file")).strip().lower()
                    _status_word = "stopped by user" if _sp_cancelled else "complete"
                    summary_lines = [
                        f"Scale processing {_status_word}: {_sp_completed} of {_sp_total} items processed.",
                    ]
                    if _out_strategy == "file_per_item":
                        _fn_template = str(sp.get("_output_filename_template", "")).strip()
                        summary_lines.append(
                            f"Output: {_sp_completed} separate files "
                            f"(template: {_fn_template or 'per-item'})"
                        )
                    elif _out_strategy == "no_file":
                        _final_act = str(sp.get("_final_action", "reply")).strip()
                        if _final_act == "api_call":
                            summary_lines.append(f"Output: indexed {_sp_completed} items to Typesense")
                        elif _final_act == "email":
                            summary_lines.append(f"Output: emailed {_sp_completed} items")
                        else:
                            summary_lines.append(f"Output: {_sp_completed} items (no file)")
                    else:
                        summary_lines.append(f"Output file: {output_file}")
                    if _sp_failed > 0:
                        summary_lines.append(f"Failed: {_sp_failed} items")
                        for err in _micro_errors[:5]:
                            summary_lines.append(f"  - {err.get('item', '?')}: {err.get('error', '?')}")
                    micro_summary = "\n".join(summary_lines)

                    self._add_session_message(
                        role="assistant",
                        content=micro_summary,
                    )

                    self._emit_tool_output(
                        "task_contract",
                        {
                            "step": "scale_micro_loop_done",
                            "processed": _sp_processed,
                            "failed": _sp_failed,
                            "total": _sp_total,
                        },
                        micro_summary,
                    )

                    # Finalize: let the completion gate evaluate.
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=micro_summary,
                        iteration=iteration,
                        finish_success=micro_result.get("success", False),
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    # If not finalized (gate wants more), continue the
                    # normal loop — but remaining should be 0 at this point.
                    continue

                if not self._supports_tool_result_followup():
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=output,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=final,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                # Continue loop with normal tool-enabled call on next iteration.
                # This avoids prematurely finalizing after a single tool (e.g. skill read).
                continue

            # Check for tool calls embedded in response text (fallback)
            # Looking for patterns like: {tool => "shell", args => {...}}
            embedded_calls = self._extract_tool_calls_from_content(response.content)
            if embedded_calls:
                log.info("Tool calls found in response text", count=len(embedded_calls))
                self._add_session_message(
                    role="assistant",
                    content=str(response.content or ""),
                    tool_calls=self._serialize_tool_calls_for_session(embedded_calls),
                )
                await self._handle_tool_calls(
                    embedded_calls,
                    turn_usage=turn_usage,
                    session_policy=session_tool_policy,
                    task_policy=active_task_tool_policy,
                    abort_event=turn_abort_event,
                )
                if planning_pipeline is not None:
                    activated = self._advance_pipeline(planning_pipeline, event="embedded_tool_calls_completed")
                    if activated:
                        await self.ensure_pipeline_subagent_contexts(
                            planning_pipeline,
                            task_ids=[str(item) for item in activated],
                        )
                # ── Deferred scale init after fetch (embedded path) ──
                _sp_pre2 = getattr(self, "_scale_progress", None)
                _pre_items2 = _sp_pre2.get("items", []) if _sp_pre2 else []
                _needs_deferred2 = (
                    _sp_pre2 is None
                    or not _pre_items2
                    or len(_pre_items2) < 3
                    or self._items_are_source_urls_only(_pre_items2)
                )
                if _needs_deferred2:
                    list_task_plan = await self._deferred_scale_init(
                        effective_user_input=effective_user_input,
                        list_task_plan=list_task_plan,
                        turn_usage=turn_usage,
                    )
                _sp_deferred2 = getattr(self, "_scale_progress", None)
                if (
                    _needs_deferred2
                    and _sp_deferred2 is not None
                    and len(_sp_deferred2.get("items", [])) >= 3
                ):
                    _def_task2 = effective_user_input
                    _def_sa2 = _def_task2.find("\n\n--- SCALE ADVISORY")
                    if _def_sa2 > 0:
                        _def_end2 = _def_task2.find("--- END SCALE ADVISORY ---")
                        if _def_end2 > _def_sa2:
                            _def_task2 = _def_task2[:_def_sa2].rstrip()
                    _def_action2 = _def_task2[:2000] or str(list_task_plan.get("per_member_action", "")).strip() or "Process each item"
                    _sp_deferred2["_per_member_action"] = _def_action2
                    self._emit_tool_output(
                        "task_contract",
                        {
                            "step": "deferred_scale_micro_loop_takeover",
                            "items": len(_sp_deferred2.get("items", [])),
                            "output_strategy": _sp_deferred2.get("_output_strategy", "single_file"),
                        },
                        (
                            "step=deferred_scale_micro_loop_takeover\n"
                            f"items={len(_sp_deferred2.get('items', []))}\n"
                            "note=entering micro loop after deferred scale init (embedded)"
                        ),
                    )
                    active_task_tool_policy = self._active_task_tool_policy_payload(planning_pipeline)
                    micro_result = await self._run_scale_micro_loop(
                        task_description=_def_action2,
                        output_file="",
                        turn_usage=turn_usage,
                        session_policy=session_tool_policy,
                        task_policy=active_task_tool_policy,
                    )
                    _sp_total = micro_result.get("total", 0)
                    _sp_completed = micro_result.get("completed_total", 0)
                    _sp_cancelled = micro_result.get("cancelled", False)
                    _sp_failed = micro_result.get("failed", 0)
                    _status_word = "stopped by user" if _sp_cancelled else "complete"
                    _out_strategy = str(_sp_deferred2.get("_output_strategy", "single_file")).strip().lower()
                    summary_lines = [
                        f"Scale processing {_status_word}: {_sp_completed} of {_sp_total} items processed.",
                    ]
                    if _out_strategy == "file_per_item":
                        _fn_template = str(_sp_deferred2.get("_output_filename_template", "")).strip()
                        summary_lines.append(f"Output: {_sp_completed} separate files (template: {_fn_template or 'per-item'})")
                    elif _out_strategy == "no_file":
                        _final_act = str(_sp_deferred2.get("_final_action", "reply")).strip()
                        summary_lines.append(f"Output: {_sp_completed} items ({_final_act})")
                    else:
                        _def_outfile2 = _sp_deferred2.get("_output_file", "")
                        summary_lines.append(f"Output file: {_def_outfile2}" if _def_outfile2 else "Output: single file")
                    if _sp_failed > 0:
                        summary_lines.append(f"Failed: {_sp_failed} items")
                    micro_summary = "\n".join(summary_lines)
                    self._add_session_message(role="assistant", content=micro_summary)
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=micro_summary,
                        iteration=iteration,
                        finish_success=micro_result.get("success", False),
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                # ── Micro-turn scale loop takeover (embedded path) ──
                if self._scale_loop_ready():
                    sp = getattr(self, "_scale_progress", None)
                    output_file = sp.get("_output_file", "") if sp else ""
                    _task_src2 = effective_user_input
                    _sa2_start = _task_src2.find("\n\n--- SCALE ADVISORY")
                    if _sa2_start > 0:
                        _sa2_end = _task_src2.find("--- END SCALE ADVISORY ---")
                        if _sa2_end > _sa2_start:
                            _task_src2 = _task_src2[:_sa2_start].rstrip()
                    per_member_action = _task_src2[:2000]
                    if not per_member_action:
                        per_member_action = str(
                            list_task_plan.get("per_member_action", "")
                        ).strip() or effective_user_input[:500]
                    if sp is not None:
                        sp["_per_member_action"] = per_member_action
                    log.info("Scale micro-loop takeover (embedded path)")
                    micro_result = await self._run_scale_micro_loop(
                        task_description=per_member_action,
                        output_file=output_file,
                        turn_usage=turn_usage,
                        session_policy=session_tool_policy,
                        task_policy=active_task_tool_policy,
                    )
                    _sp_total = micro_result.get("total", 0)
                    _sp_completed = micro_result.get("completed_total", 0)
                    _sp_failed = micro_result.get("failed", 0)
                    _sp_cancelled = micro_result.get("cancelled", False)
                    _status_word = "stopped by user" if _sp_cancelled else "complete"
                    _out_strategy = str(sp.get("_output_strategy", "single_file")).strip().lower() if sp else "single_file"
                    if _out_strategy == "file_per_item":
                        _fn_template = str(sp.get("_output_filename_template", "")).strip() if sp else ""
                        micro_summary = (
                            f"Scale processing {_status_word}: {_sp_completed} of {_sp_total} items.\n"
                            f"Output: {_sp_completed} separate files "
                            f"(template: {_fn_template or 'per-item'})"
                        )
                    elif _out_strategy == "no_file":
                        _final_act = str(sp.get("_final_action", "reply")).strip() if sp else "reply"
                        if _final_act == "api_call":
                            micro_summary = (
                                f"Scale processing {_status_word}: {_sp_completed} of {_sp_total} items.\n"
                                f"Output: indexed {_sp_completed} items to Typesense"
                            )
                        elif _final_act == "email":
                            micro_summary = (
                                f"Scale processing {_status_word}: {_sp_completed} of {_sp_total} items.\n"
                                f"Output: emailed {_sp_completed} items"
                            )
                        else:
                            micro_summary = (
                                f"Scale processing {_status_word}: {_sp_completed} of {_sp_total} items.\n"
                                f"Output: {_sp_completed} items processed (no file)"
                            )
                    else:
                        micro_summary = (
                            f"Scale processing {_status_word}: {_sp_completed} of {_sp_total} items.\n"
                            f"Output file: {output_file}"
                        )
                    if _sp_failed:
                        micro_summary += f"\nFailed: {_sp_failed} items"
                    self._add_session_message(role="assistant", content=micro_summary)
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=micro_summary,
                        iteration=iteration,
                        finish_success=micro_result.get("success", False),
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue

                if not self._supports_tool_result_followup():
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=output,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=final,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                # Continue loop with normal tool-enabled call on next iteration.
                continue

            # Check for inline commands in response (fallback for models without tool calling)
            # This works by extracting commands from markdown code blocks in the response
            command = self._extract_command_from_response(response.content)
            if command:
                log.info("Executing inline command", command=command)
                self._emit_thinking(f"Running: {command[:60]}", tool="shell", phase="tool")
                try:
                    result = await self._execute_tool_with_guard(
                        name="shell",
                        arguments={"command": command},
                        interaction_label="inline_command",
                        turn_usage=turn_usage,
                        session_policy=session_tool_policy,
                        task_policy=active_task_tool_policy,
                        abort_event=turn_abort_event,
                    )
                    tool_result = result.content if result.success else f"Error: {result.error}"
                except Exception as e:
                    result = None
                    tool_result = f"Error: {str(e)}"
                
                # Add tool result to session
                self._add_session_message(
                    role="tool",
                    content=tool_result,
                    tool_name="shell",
                    tool_arguments={"command": command},
                )
                self._emit_tool_output("shell", {"command": command}, tool_result)
                if planning_pipeline is not None:
                    activated = self._advance_pipeline(planning_pipeline, event="inline_command_completed")
                    if activated:
                        await self.ensure_pipeline_subagent_contexts(
                            planning_pipeline,
                            task_ids=[str(item) for item in activated],
                        )
                if not self._supports_tool_result_followup():
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=tool_result,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=final,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                
                # Get final response
                messages = self._build_messages(
                    tool_messages_from_index=turn_start_idx,
                    query=effective_user_input,
                    planning_pipeline=planning_pipeline,
                    list_task_plan=list_task_plan,
                )
                try:
                    response = await self._complete_with_guards(
                        messages=messages,
                        tools=None,
                        interaction_label="inline_command_followup",
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=response.content,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                except Exception:
                    # Return tool output directly
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=tool_result,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=final,
                        iteration=iteration,
                        finish_success=False,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
            
            # No tool calls - this is the final response
            finalized, final_text, finish_success = await attempt_finalize_response(
                output_text=response.content,
                iteration=iteration,
                finish_success=True,
            )
            if finalized:
                return finish(final_text, success=finish_success)
            continue
        
        # Hard iteration cap reached
        self._set_runtime_status("waiting")
        return finish("Max iterations reached. Could not complete the request.", success=False)

    async def stream(self, user_input: str) -> AsyncIterator[str]:
        """Stream response for user input.
        
        Args:
            user_input: User's message
        
        Yields:
            Response chunks
        """
        if not self._initialized:
            await self.initialize()
        self._last_memory_debug_signature = None
        self._last_semantic_memory_debug_signature = None
        self.last_usage = self._empty_usage()

        # Tool-calling and streaming over a single pass is currently limited.
        # Preserve tool behavior and guard checks by using complete() and
        # yielding chunked output when tools/guards are enabled.
        if self.tools.list_tools() or self.guards_enabled():
            self._set_runtime_status("thinking")
            content = await self.complete(user_input)
            chunk_size = 24
            self._set_runtime_status("streaming")
            for idx in range(0, len(content), chunk_size):
                yield content[idx : idx + chunk_size]
            self._set_runtime_status("waiting")
            return
        
        # Add user message to session
        self._add_session_message("user", user_input)
        await self._auto_compact_if_needed()
        planning_pipeline: dict[str, Any] | None = None
        if self.planning_enabled:
            planning_pipeline = self._build_task_pipeline(user_input)
            self._emit_pipeline_update("created", planning_pipeline)
            await self.ensure_pipeline_subagent_contexts(planning_pipeline)
        
        # Get tool definitions
        tool_defs = self.tools.get_definitions()
        
        # For streaming, we currently don't support tool calling
        # This is a limitation - full streaming with tools needs more work
        messages = self._build_messages(query=user_input, planning_pipeline=planning_pipeline)
        
        # Stream the response
        full_content = ""
        self._set_runtime_status("streaming")
        async for chunk in self.provider.complete_streaming(
            messages=messages,
            tools=tool_defs if tool_defs else None,
        ):
            full_content += chunk
            yield chunk
        
        # Add assistant response to session
        if self.session:
            self._add_session_message("assistant", full_content)
            await self.session_manager.save_session(self.session)
            memory = getattr(self, "memory", None)
            if memory is not None:
                memory.schedule_background_sync("assistant_stream_saved")

        if planning_pipeline is not None:
            self._finalize_pipeline(planning_pipeline, success=True)

        prompt_tokens = sum(self._count_tokens(m.content) for m in messages)
        completion_tokens = self._count_tokens(full_content)
        self._finalize_turn_usage({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })
        self._set_runtime_status("waiting")
