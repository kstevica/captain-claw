"""Main request orchestration (complete/stream) for Agent."""

import json
from typing import Any, AsyncIterator

from captain_claw.exceptions import GuardBlockedError
from captain_claw.llm import Message
from captain_claw.logging import get_logger


log = get_logger(__name__)


class AgentOrchestrationMixin:
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

        turn_usage = self._empty_usage()
        self.last_usage = self._empty_usage()
        planning_pipeline: dict[str, Any] | None = None
        recent_source_urls: list[str] = []
        effective_user_input = user_input
        effective_user_input, clarification_context_applied = self._resolve_effective_user_input(user_input)
        require_all_sources = self._request_references_all_sources(effective_user_input)
        use_contract_pipeline = self._should_use_contract_pipeline(
            effective_user_input,
            self.planning_enabled,
            pipeline_mode=self.pipeline_mode,
        )
        explicit_script_request = self._is_explicit_script_request(effective_user_input)
        enforce_python_worker_mode = explicit_script_request
        available_tools = {name.strip().lower() for name in self.tools.list_tools()}
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

        def finish(text: str, success: bool = True) -> str:
            if planning_pipeline is not None:
                self._finalize_pipeline(planning_pipeline, success=success)
            self._finalize_turn_usage(turn_usage)
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
                    covered_members, missing_members = self._evaluate_list_member_coverage(
                        members=[str(member) for member in members],
                        candidate_response=final_response,
                        turn_start_idx=turn_start_idx,
                    )
                    self._emit_tool_output(
                        "completion_gate",
                        {
                            "step": "list_member_coverage",
                            "covered": len(covered_members),
                            "missing": len(missing_members),
                            "members": len(members),
                        },
                        (
                            "step=list_member_coverage\n"
                            f"covered={len(covered_members)}\n"
                            f"missing={len(missing_members)}\n"
                            f"members={len(members)}"
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
            return True, final_response, finish_success

        turn_start_idx = len(self.session.messages) if self.session else 0
        recent_source_urls = self._collect_recent_source_urls(turn_start_idx)
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
        if clarification_context_applied:
            self._emit_tool_output(
                "task_contract",
                {"step": "clarification_context_applied"},
                "step=clarification_context_applied\nstatus=merged_pending_anchor_into_current_turn",
            )
        list_context_excerpt = self._collect_list_extraction_context()
        list_task_plan = await self._generate_list_task_plan(
            user_input=effective_user_input,
            context_excerpt=list_context_excerpt,
            turn_usage=turn_usage,
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
        if use_contract_pipeline:
            task_contract = await self._generate_task_contract(
                user_input=effective_user_input,
                recent_source_urls=recent_source_urls,
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
            if prefetch_urls:
                await self._run_source_report_prefetch(
                    source_urls=prefetch_urls,
                    turn_usage=turn_usage,
                    pipeline_label="task_contract",
                )
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
        
        # Send tool definitions so the model can issue structured tool calls.
        tool_defs = self.tools.get_definitions()
        log.debug("Tool definitions available", count=len(self.tools.list_tools()), tools_sent=bool(tool_defs))
        
        # Main agent loop
        base_turn_iterations = self.max_iterations + (2 if completion_requirements else 0)
        planned_turn_iterations = self._compute_turn_iteration_budget(
            base_iterations=base_turn_iterations,
            planning_pipeline=planning_pipeline,
            completion_requirements=completion_requirements,
        )
        hard_turn_iterations = max(planned_turn_iterations, min(320, planned_turn_iterations * 3))
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
            
            # Call LLM
            log.info("Calling LLM", iteration=iteration + 1, message_count=len(messages))
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
                raise
            
            # Check for explicit tool calls (for models that support it)
            if response.tool_calls:
                log.info("Tool calls detected", count=len(response.tool_calls), calls=response.tool_calls)
                self._add_session_message(
                    role="assistant",
                    content=str(response.content or ""),
                    tool_calls=self._serialize_tool_calls_for_session(response.tool_calls),
                )
                await self._handle_tool_calls(response.tool_calls, turn_usage=turn_usage)
                if planning_pipeline is not None:
                    self._advance_pipeline(planning_pipeline, event="tool_calls_completed")
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
                # Try to get final response
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
                        interaction_label="tool_followup",
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
                    # Model doesn't support tool results - return tool output
                    log.warning("Model doesn't support tool results", error=str(e))
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=output,
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
                finalized, final_text, finish_success = await attempt_finalize_response(
                    output_text=response.content,
                    iteration=iteration,
                    finish_success=True,
                )
                if finalized:
                    return finish(final_text, success=finish_success)
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
                await self._handle_tool_calls(embedded_calls, turn_usage=turn_usage)
                if planning_pipeline is not None:
                    self._advance_pipeline(planning_pipeline, event="embedded_tool_calls_completed")
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
                # Try to get final response
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
                        interaction_label="embedded_tool_followup",
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
                    log.warning("Model doesn't support tool results", error=str(e))
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=output,
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
                # If successful, return the response normally
                finalized, final_text, finish_success = await attempt_finalize_response(
                    output_text=response.content,
                    iteration=iteration,
                    finish_success=True,
                )
                if finalized:
                    return finish(final_text, success=finish_success)
                continue
            
            # Check for inline commands in response (fallback for models without tool calling)
            # This works by extracting commands from markdown code blocks in the response
            command = self._extract_command_from_response(response.content)
            if command:
                log.info("Executing inline command", command=command)
                try:
                    result = await self._execute_tool_with_guard(
                        name="shell",
                        arguments={"command": command},
                        interaction_label="inline_command",
                        turn_usage=turn_usage,
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
                    self._advance_pipeline(planning_pipeline, event="inline_command_completed")
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
