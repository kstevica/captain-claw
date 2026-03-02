"""Main request orchestration (complete/stream) for Agent.

This mixin contains the core ``complete()`` and ``stream()`` methods that
drive the agent's turn-level loop.  Scale detection, advisory injection,
deferred init, and micro-loop summary building have been moved to
``AgentScaleDetectionMixin``.  Completion gate, finalization, and coverage
validation have been moved to ``AgentCompletionMixin``.
"""

import asyncio
import json
from typing import Any, AsyncIterator

from captain_claw.config import get_config
from captain_claw.exceptions import GuardBlockedError
from captain_claw.llm import Message
from captain_claw.logging import get_logger


log = get_logger(__name__)

# Tools that bring in new external / document content and may warrant
# deferred scale re-extraction.  Lightweight tools like datastore, glob,
# shell, todo do NOT qualify — they return structured local data that
# should never trigger the research micro-loop.
_CONTENT_FETCH_TOOLS: frozenset[str] = frozenset({
    "web_fetch", "web_get", "read",
    "pdf_extract", "docx_extract", "pptx_extract",
    "xlsx_extract", "pocket_tts",
})


class AgentOrchestrationMixin:
    """Core request orchestration: complete() and stream()."""

    # ------------------------------------------------------------------
    # complete() — main entry point
    # ------------------------------------------------------------------

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
        # Reset per-turn success flag (updated by finish()).
        self._last_complete_success = True
        # Scale-progress tracker: populated when the scale advisory fires.
        # The tool loop uses this to emit "3 of 27 (11%)" progress.
        self._scale_progress: dict[str, Any] | None = None
        self._deferred_scale_attempts: int = 0
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
            self._last_complete_success = success
            return text

        async def attempt_finalize(
            output_text: str,
            iteration: int,
            finish_success: bool = True,
        ) -> tuple[bool, str, bool]:
            """Wrapper around _attempt_finalize_response that updates closure vars."""
            nonlocal completion_feedback, python_worker_attempted, list_task_plan
            (
                finalized,
                final_text,
                fin_success,
                completion_feedback,
                python_worker_attempted,
            ) = await self._attempt_finalize_response(
                output_text=output_text,
                iteration=iteration,
                hard_turn_iterations=hard_turn_iterations,
                finish_success=finish_success,
                effective_user_input=effective_user_input,
                user_input=user_input,
                turn_start_idx=turn_start_idx,
                turn_usage=turn_usage,
                session_tool_policy=session_tool_policy,
                planning_pipeline=planning_pipeline,
                list_task_plan=list_task_plan,
                task_contract=task_contract,
                completion_requirements=completion_requirements,
                completion_feedback=completion_feedback,
                enforce_python_worker_mode=enforce_python_worker_mode,
                python_worker_attempted=python_worker_attempted,
            )
            return finalized, final_text, fin_success

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
        await self._refresh_datastore_context_cache()
        if clarification_context_applied:
            self._emit_tool_output(
                "task_contract",
                {"step": "clarification_context_applied"},
                "step=clarification_context_applied\nstatus=merged_pending_anchor_into_current_turn",
            )
        # ── Automatic task rephrasing ──────────────────────────────
        task_was_rephrased = False
        if not is_worker and not clarification_context_applied:
            effective_user_input, task_was_rephrased = await self._rephrase_task(
                user_input=effective_user_input,
                turn_usage=turn_usage,
            )
            if task_was_rephrased:
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
        # Skip for workers — they execute a single focused task; URL
        # extraction from the worker prompt would re-enable the list task
        # plan that was intentionally disabled above, causing endless
        # coverage-check loops on simple fetch-and-summarize tasks.
        if not is_worker:
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

        # ── Pre-flight scale check ────────────────────────────────
        # The scale micro-loop accelerates large list-processing tasks by
        # taking over the extract→write loop.  The preflight check uses
        # _SKIP_SCALE_DETECTION_RE to avoid firing for discovery-only tasks
        # (e.g. "find all files and return the list") so the scale loop
        # only activates for genuine per-item processing tasks.
        scale_advisory = self._preflight_scale_check(effective_user_input, list_task_plan)
        if scale_advisory:
            effective_user_input = effective_user_input + scale_advisory
            self._scale_progress = {"total": 0, "completed": 0}
            _out_strategy = str(list_task_plan.get("output_strategy", "single_file")).strip().lower()
            self._scale_progress["_output_strategy"] = _out_strategy
            self._scale_progress["_output_filename_template"] = str(
                list_task_plan.get("output_filename_template", "")
            ).strip()
            self._scale_progress["_final_action"] = str(
                list_task_plan.get("final_action", "write_file")
            ).strip()
            if _out_strategy == "no_file":
                self._scale_progress["_sink_collection"] = ""
                self._scale_progress["_sink_email_to"] = ""
            list_members = list_task_plan.get("members", [])
            if list_members:
                self._scale_progress["items"] = list(list_members)
                self._scale_progress["done_items"] = set()
                self._scale_progress["total"] = len(list_members)
                self._scale_progress["_extraction_mode"] = self._classify_item_extraction_mode(
                    list_members,
                    per_member_action=str(list_task_plan.get("per_member_action", "")),
                    user_input=effective_user_input,
                )
                self._scale_progress["_member_context"] = list_task_plan.get("member_context") or {}
                # For inline mode, store source page content for per-item LLM calls
                if self._scale_progress.get("_extraction_mode") == "inline" and list_context_excerpt:
                    self._scale_progress["_source_page_content"] = list_context_excerpt

        # ── Contract pipeline ─────────────────────────────────────
        if use_contract_pipeline:
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
            if sp is not None and not sp.get("items") and prefetch_urls:
                sp["items"] = list(prefetch_urls)
                sp["done_items"] = set()
                sp["total"] = len(prefetch_urls)
                sp["_extraction_mode"] = self._classify_item_extraction_mode(
                    prefetch_urls,
                    per_member_action=str(list_task_plan.get("per_member_action", "")),
                    user_input=effective_user_input,
                )

        # ── Planning pipeline setup ───────────────────────────────
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

        # ── Lightweight scale-progress for moderate lists ─────────
        _lw_min = get_config().scale.lightweight_progress_min_members
        if (
            self._scale_progress is None
            and bool(list_task_plan.get("enabled", False))
            and len(list_task_plan.get("members", [])) >= _lw_min
        ):
            self._scale_progress = self._init_scale_progress_from_plan(
                list_task_plan, user_input=effective_user_input,
            )
            # For inline mode, store the full source page content so the
            # micro-loop can feed the entire page (not just tiny snippets)
            # to per-item LLM calls.
            if self._scale_progress.get("_extraction_mode") == "inline" and list_context_excerpt:
                self._scale_progress["_source_page_content"] = list_context_excerpt
            self._emit_tool_output(
                "task_contract",
                {
                    "step": "scale_progress_from_list_task",
                    "members": len(list_task_plan.get("members", [])),
                },
                (
                    "step=scale_progress_from_list_task\n"
                    f"members={len(list_task_plan.get('members', []))}\n"
                    "note=activated lightweight progress tracking for moderate list"
                ),
            )

        # ── Early micro-loop takeover ─────────────────────────────
        _sp_early = getattr(self, "_scale_progress", None)
        _early_items = _sp_early.get("items", []) if _sp_early else []
        # Skip micro-loop when extraction mode is "passthrough" — the user
        # wants to save/store items (e.g. create a datastore table) and the
        # main LLM should handle it directly with tool calls.
        _early_passthrough = (
            _sp_early is not None
            and str(_sp_early.get("_extraction_mode", "")).strip() == "passthrough"
        )
        _can_early_takeover = (
            _sp_early is not None
            and len(_early_items) >= 2
            and not self._items_are_source_urls_only(_early_items)
            and not _early_passthrough
        )
        if _can_early_takeover:
            log.info("Early scale micro-loop takeover", items=len(_early_items))
            micro_result = await self._run_micro_loop_and_summarize(
                effective_user_input=effective_user_input,
                list_task_plan=list_task_plan,
                turn_usage=turn_usage,
                session_tool_policy=session_tool_policy,
                planning_pipeline=planning_pipeline,
                step_label="early_scale_micro_loop_takeover",
            )
            micro_summary = micro_result["summary"]
            self._update_clarification_state(
                user_input=user_input,
                effective_user_input=effective_user_input,
                assistant_response=micro_summary,
            )
            await self._persist_assistant_response(micro_summary)
            return finish(micro_summary, success=micro_result.get("success", False))

        # ── Main agent loop ───────────────────────────────────────
        base_turn_iterations = self.max_iterations + (2 if completion_requirements else 0)
        planned_turn_iterations = self._compute_turn_iteration_budget(
            base_iterations=base_turn_iterations,
            planning_pipeline=planning_pipeline,
            completion_requirements=completion_requirements,
        )
        if scale_advisory:
            member_count = len(list_task_plan.get("members", []))
            estimated_items = member_count if member_count > 15 else 50
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
            # ── External cancellation check ───────────────────────
            cancel_ev: asyncio.Event | None = getattr(self, "cancel_event", None)
            if cancel_ev is not None and cancel_ev.is_set():
                self._set_runtime_status("waiting")
                self._emit_thinking("Cancelled", phase="done")
                cancel_ev.clear()
                return finish("Request cancelled by user.", success=False)

            # ── Pipeline runtime tick ─────────────────────────────
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

            # ── Iteration budget management ───────────────────────
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

            # ── Progress / stagnation tracking ────────────────────
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

            # ── LLM call ──────────────────────────────────────────
            self._set_runtime_status("thinking")
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

            _ctx = getattr(self, "last_context_window", {})
            _ctx_tokens = int(_ctx.get("prompt_tokens", 0))
            _ctx_budget = int(_ctx.get("context_budget_tokens", 1))
            _ctx_pct = round(_ctx_tokens / _ctx_budget * 100, 1) if _ctx_budget else 0
            _ctx_kb = round(_ctx_tokens * 4 / 1024, 1)
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
                error_str = str(e)
                tool_output = self._collect_turn_tool_output(turn_start_idx)
                if tool_output and "500" in error_str:
                    log.warning("Tool result call failed (500), returning tool output")
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=tool_output,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize(
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

            # ── Explicit tool calls ───────────────────────────────
            if response.tool_calls:
                log.info("Tool calls detected", count=len(response.tool_calls), calls=response.tool_calls)
                self._add_session_message(
                    role="assistant",
                    content=str(response.content or ""),
                    tool_calls=self._serialize_tool_calls_for_session(response.tool_calls),
                )
                _tool_results = await self._handle_tool_calls(
                    response.tool_calls,
                    turn_usage=turn_usage,
                    session_policy=session_tool_policy,
                    task_policy=active_task_tool_policy,
                    abort_event=turn_abort_event,
                )

                # ── Free iteration for successful glob ────────────
                # When glob finds files, the iteration was purely
                # informational — give the budget back so the agent
                # can spend it on real work.
                if _tool_results and any(
                    str(r.get("tool_name", "")).lower() == "glob"
                    and r.get("success")
                    and "No files found" not in str(r.get("content", ""))
                    for r in _tool_results
                ):
                    soft_turn_iterations += 1
                    hard_turn_iterations += 1
                if planning_pipeline is not None:
                    activated = self._advance_pipeline(planning_pipeline, event="tool_calls_completed")
                    if activated:
                        await self.ensure_pipeline_subagent_contexts(
                            planning_pipeline,
                            task_ids=[str(item) for item in activated],
                        )

                # ── Deferred scale init after fetch ───────────────
                # Only consider deferred scale init when a content-
                # fetching tool ran (web_fetch, web_get, read,
                # pdf_extract, etc.).  Datastore / glob / shell / todo
                # results don't bring in new article content that
                # warrants list re-extraction.
                _had_content_fetch = _tool_results and any(
                    str(r.get("tool_name", "")).lower() in _CONTENT_FETCH_TOOLS
                    for r in _tool_results
                )
                _needs_def = (
                    _had_content_fetch and self._needs_deferred_scale_init()
                )
                log.info(
                    "Post-tool deferred check",
                    needs_deferred=_needs_def,
                    had_content_fetch=_had_content_fetch,
                    scale_items=len(
                        (getattr(self, "_scale_progress", None) or {}).get("items", [])
                    ),
                )
                if _needs_def:
                    list_task_plan = await self._deferred_scale_init(
                        effective_user_input=effective_user_input,
                        list_task_plan=list_task_plan,
                        turn_usage=turn_usage,
                    )
                    # If deferred init populated REAL items (not just source
                    # URLs), enter micro-loop.  The source-URL guard prevents
                    # entering the micro-loop with the article URL repeated.
                    # Skip when extraction mode is "passthrough" (save/store intent).
                    _sp_deferred = getattr(self, "_scale_progress", None)
                    _deferred_items = _sp_deferred.get("items", []) if _sp_deferred else []
                    _deferred_passthrough = (
                        _sp_deferred is not None
                        and str(_sp_deferred.get("_extraction_mode", "")).strip() == "passthrough"
                    )
                    if (
                        _sp_deferred is not None
                        and len(_deferred_items) >= 2
                        and not self._items_are_source_urls_only(_deferred_items)
                        and not _deferred_passthrough
                    ):
                        micro_result = await self._run_micro_loop_and_summarize(
                            effective_user_input=effective_user_input,
                            list_task_plan=list_task_plan,
                            turn_usage=turn_usage,
                            session_tool_policy=session_tool_policy,
                            planning_pipeline=planning_pipeline,
                            step_label="deferred_scale_micro_loop_takeover",
                        )
                        finalized, final_text, finish_success = await attempt_finalize(
                            output_text=micro_result["summary"],
                            iteration=iteration,
                            finish_success=micro_result.get("success", False),
                        )
                        if finalized:
                            return finish(final_text, success=finish_success)
                        continue

                # ── Micro-loop takeover (tool call path) ──────────
                if self._scale_loop_ready():
                    log.info("Scale micro-loop takeover (tool call path)")
                    sp = getattr(self, "_scale_progress", None)
                    output_file = sp.get("_output_file", "") if sp else ""
                    micro_result = await self._run_micro_loop_and_summarize(
                        effective_user_input=effective_user_input,
                        list_task_plan=list_task_plan,
                        turn_usage=turn_usage,
                        session_tool_policy=session_tool_policy,
                        planning_pipeline=planning_pipeline,
                        step_label="scale_micro_loop_takeover",
                        output_file=output_file,
                    )
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=micro_result["summary"],
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
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=final,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                continue

            # ── Embedded tool calls (fallback) ────────────────────
            embedded_calls = self._extract_tool_calls_from_content(response.content)
            if embedded_calls:
                log.info("Tool calls found in response text", count=len(embedded_calls))
                self._add_session_message(
                    role="assistant",
                    content=str(response.content or ""),
                    tool_calls=self._serialize_tool_calls_for_session(embedded_calls),
                )
                _emb_tool_results = await self._handle_tool_calls(
                    embedded_calls,
                    turn_usage=turn_usage,
                    session_policy=session_tool_policy,
                    task_policy=active_task_tool_policy,
                    abort_event=turn_abort_event,
                )

                # Free iteration for successful glob (embedded path).
                if _emb_tool_results and any(
                    str(r.get("tool_name", "")).lower() == "glob"
                    and r.get("success")
                    and "No files found" not in str(r.get("content", ""))
                    for r in _emb_tool_results
                ):
                    soft_turn_iterations += 1
                    hard_turn_iterations += 1

                if planning_pipeline is not None:
                    activated = self._advance_pipeline(planning_pipeline, event="embedded_tool_calls_completed")
                    if activated:
                        await self.ensure_pipeline_subagent_contexts(
                            planning_pipeline,
                            task_ids=[str(item) for item in activated],
                        )

                # ── Deferred scale init (embedded path) ───────────
                if self._needs_deferred_scale_init():
                    list_task_plan = await self._deferred_scale_init(
                        effective_user_input=effective_user_input,
                        list_task_plan=list_task_plan,
                        turn_usage=turn_usage,
                    )
                    _sp_deferred2 = getattr(self, "_scale_progress", None)
                    _deferred_items2 = _sp_deferred2.get("items", []) if _sp_deferred2 else []
                    _deferred_passthrough2 = (
                        _sp_deferred2 is not None
                        and str(_sp_deferred2.get("_extraction_mode", "")).strip() == "passthrough"
                    )
                    if (
                        _sp_deferred2 is not None
                        and len(_deferred_items2) >= 2
                        and not self._items_are_source_urls_only(_deferred_items2)
                        and not _deferred_passthrough2
                    ):
                        micro_result = await self._run_micro_loop_and_summarize(
                            effective_user_input=effective_user_input,
                            list_task_plan=list_task_plan,
                            turn_usage=turn_usage,
                            session_tool_policy=session_tool_policy,
                            planning_pipeline=planning_pipeline,
                            step_label="deferred_scale_micro_loop_takeover",
                        )
                        finalized, final_text, finish_success = await attempt_finalize(
                            output_text=micro_result["summary"],
                            iteration=iteration,
                            finish_success=micro_result.get("success", False),
                        )
                        if finalized:
                            return finish(final_text, success=finish_success)
                        continue

                # ── Micro-loop takeover (embedded path) ───────────
                if self._scale_loop_ready():
                    log.info("Scale micro-loop takeover (embedded path)")
                    sp = getattr(self, "_scale_progress", None)
                    output_file = sp.get("_output_file", "") if sp else ""
                    micro_result = await self._run_micro_loop_and_summarize(
                        effective_user_input=effective_user_input,
                        list_task_plan=list_task_plan,
                        turn_usage=turn_usage,
                        session_tool_policy=session_tool_policy,
                        planning_pipeline=planning_pipeline,
                        step_label="scale_micro_loop_takeover",
                        output_file=output_file,
                    )
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=micro_result["summary"],
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
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=final,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue

                # ── Worker finalization for embedded tool calls ────
                # Workers execute a single focused task.  When the LLM
                # response contained text *and* embedded tool-like
                # patterns, the text portion is the actual answer but
                # the embedded extraction prevented the text-only
                # finalization path from being reached.  Try to
                # finalize using the response text so workers don't
                # loop endlessly on spurious embedded tool matches.
                if is_worker and str(response.content or "").strip():
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=response.content,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)

                continue

            # ── Inline command fallback ────────────────────────────
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
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=final,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue

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
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=response.content,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                except Exception:
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=tool_result,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=final,
                        iteration=iteration,
                        finish_success=False,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue

            # ── No tool calls — final response ────────────────────
            finalized, final_text, finish_success = await attempt_finalize(
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

    # ------------------------------------------------------------------
    # stream() — streaming wrapper
    # ------------------------------------------------------------------

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
