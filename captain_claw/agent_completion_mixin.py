"""Completion gate, finalization, and response validation for Agent.

This mixin handles the logic that decides whether a turn's output is
"done" or needs more iterations:

- Auto-write of requested output files
- Datastore save verification
- Python worker mode enforcement
- List member coverage evaluation
- Task contract completion validation
- Clarification state updates and session persistence
- Auto-capture of todos, contacts, scripts, APIs

Extracted from the ``attempt_finalize_response`` nested function inside
``AgentOrchestrationMixin.complete()`` to reduce the size of the main
orchestration loop and make the completion logic testable in isolation.
"""

import json
from typing import Any

from captain_claw.logging import get_logger


log = get_logger(__name__)


class AgentCompletionMixin:
    """Completion gate and response finalization."""

    async def _verify_datastore_saves(
        self,
        saves: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Verify that datastore save operations actually persisted.

        Queries the datastore to confirm that tables exist and contain
        the expected data.  Returns a list of failed verifications,
        each with keys ``action``, ``table``, and ``reason``.
        """
        from captain_claw.config import get_config
        from captain_claw.datastore import get_datastore_manager, get_session_datastore_manager

        cfg = get_config()
        if cfg.web.public_run == "computer" and self.session:
            dm = get_session_datastore_manager(str(self.session.id))
        else:
            dm = get_datastore_manager()
        failures: list[dict[str, Any]] = []

        # Deduplicate: only verify the last save per table.
        tables_to_verify: dict[str, dict[str, Any]] = {}
        for save in saves:
            tables_to_verify[save["table"]] = save

        for table_name, save in tables_to_verify.items():
            action = save["action"]
            try:
                info = await dm.describe_table(table_name)
                # For insert/import_file, the table must have rows.
                if action in ("insert", "import_file") and info.row_count == 0:
                    failures.append({
                        "action": action,
                        "table": table_name,
                        "reason": (
                            f"Table '{table_name}' exists but has 0 rows "
                            f"after {action}"
                        ),
                    })
            except Exception as exc:
                failures.append({
                    "action": action,
                    "table": table_name,
                    "reason": (
                        f"Table '{table_name}' could not be verified "
                        f"after {action}: {exc}"
                    ),
                })

        return failures

    async def _attempt_finalize_response(
        self,
        *,
        output_text: str,
        iteration: int,
        hard_turn_iterations: int,
        finish_success: bool,
        effective_user_input: str,
        user_input: str,
        turn_start_idx: int,
        turn_usage: dict[str, int],
        session_tool_policy: dict[str, Any] | None,
        planning_pipeline: dict[str, Any] | None,
        list_task_plan: dict[str, Any],
        task_contract: dict[str, Any] | None,
        completion_requirements: list[dict[str, Any]],
        completion_feedback: str,
        enforce_python_worker_mode: bool,
        python_worker_attempted: bool,
    ) -> tuple[bool, str, bool, str, bool]:
        """Apply auto-write + completion gate before returning final output.

        Returns:
            Tuple of (finalized, final_text, finish_success,
                       updated_completion_feedback, updated_python_worker_attempted).

        When ``finalized`` is True the orchestration loop should return
        ``final_text`` to the caller.  When False the loop should continue
        with ``updated_completion_feedback`` injected as a user message.
        """
        log.info(
            "attempt_finalize_response called",
            iteration=iteration,
            hard_turn_iterations=hard_turn_iterations,
            has_task_contract=task_contract is not None,
            completion_requirements_count=len(completion_requirements),
            list_task_plan_enabled=bool(list_task_plan.get("enabled", False)),
            enforce_python_worker_mode=enforce_python_worker_mode,
            output_text_len=len(output_text or ""),
        )
        # Skip auto-write when the scale micro-loop just completed.
        # The micro-loop already wrote the real content to output files
        # via _execute_tool_with_guard (which does NOT add tool messages
        # to the session).  Without this guard,
        # _maybe_auto_write_requested_output would detect a filename in
        # the user input, fail to find a write-tool session message, and
        # auto-write the scale *summary* text as a marker file that
        # shadows the real output for downstream tasks.
        _sp = getattr(self, "_scale_progress", None)
        _scale_completed = (
            _sp is not None
            and _sp.get("items")
            and len(_sp.get("done_items", set())) >= len(_sp["items"])
        )
        if _scale_completed:
            final_response = output_text
        else:
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

        # ── final_action=write_file enforcement ────────────────────────
        # When the list task plan requires file output, verify that a
        # file was actually produced this turn.  Accepted patterns:
        #   a) datastore(action='export') succeeded (single-table export)
        #   b) write tool succeeded (direct file creation — markdown,
        #      text, or any non-tabular content)
        #   c) A Python script was written to a .py file and executed
        #      via shell (script-generated output like PDF, charts)
        #   d) Media tools (browser screenshot, image_gen) produced output
        # Without this gate the LLM can claim it saved a file while
        # never invoking the tool, causing hallucinated file paths.
        _final_action = str(list_task_plan.get("final_action", "")).strip().lower()
        _output_strategy = str(list_task_plan.get("output_strategy", "")).strip().lower()
        if (
            _final_action == "write_file"
            and _output_strategy in ("single_file", "file_per_item")
            and bool(list_task_plan.get("enabled", False))
            and not _scale_completed
        ):
            has_ds_export = self._turn_has_successful_datastore_export(turn_start_idx)
            has_write = (
                self._turn_has_successful_tool(turn_start_idx, "write")
                or self._turn_has_successful_tool(turn_start_idx, "edit")
            )
            has_script_exec = self._turn_has_successful_script_execution(turn_start_idx)
            # Tools that directly produce file output (screenshots, images)
            # satisfy write_file without needing write+shell.
            has_media_output = (
                self._turn_has_successful_tool(turn_start_idx, "browser")
                or self._turn_has_successful_tool(turn_start_idx, "image_gen")
            )
            # Typesense index is a valid output sink — no file needed.
            has_index_output = self._turn_has_successful_tool(turn_start_idx, "typesense")
            # Any single file-producing action is sufficient.  The write
            # or edit tool proves a file was created/modified (e.g. markdown,
            # text).  Script execution alone proves a script generated output.
            # Typesense indexing proves data was persisted to memory.
            file_produced = has_ds_export or has_write or has_script_exec or has_media_output or has_index_output
            if not file_produced and iteration < (hard_turn_iterations - 1):
                completion_feedback = (
                    "Completion gate: final_action is write_file but no output file was "
                    "actually produced this turn.\n"
                    "You MUST produce the file using one of these approaches:\n"
                    "1. write(path='<output_path>', content='<content>') — for text, "
                    "markdown, or any non-tabular content.\n"
                    "2. datastore(action='export') — for tabular data (XLSX/CSV). "
                    "For JOINs: datastore(action='export', format='xlsx', "
                    "sql_query='SELECT ... FROM t1 JOIN t2 ON ...', table='output_name').\n"
                    "3. write a Python script to scripts/ then execute it via "
                    "shell(command='python3 <script_path>') — for non-tabular output "
                    "(PDF, charts, custom formats). IMPORTANT: all output paths in the "
                    "script must be relative to the workspace root (not the script's "
                    "directory). Do NOT cd into the script directory.\n"
                    "Do NOT just describe the file path — actually invoke the tools."
                )
                self._emit_tool_output(
                    "completion_gate",
                    {
                        "step": "write_file_enforcement",
                        "passed": False,
                        "final_action": _final_action,
                        "output_strategy": _output_strategy,
                    },
                    completion_feedback,
                )
                log.info("Finalize BLOCKED by write_file_enforcement gate", iteration=iteration)
                return False, "", finish_success, completion_feedback, python_worker_attempted

        # ── Script execution enforcement ─────────────────────────────
        # If a Python script was written to scripts/ this turn but never
        # successfully executed, force the LLM to run it before finalizing.
        # This catches cases where the LLM writes a generation script
        # (e.g. for PDF/XLSX) but returns text without actually running it.
        has_unexecuted, unexecuted_path = self._turn_has_unexecuted_script(turn_start_idx)
        if has_unexecuted and iteration < (hard_turn_iterations - 1):
            script_name = unexecuted_path.rsplit("/", 1)[-1]
            completion_feedback = (
                f"Completion gate: script '{script_name}' was written but never executed.\n"
                f"You MUST run the script via shell before responding:\n"
                f"  shell(command='python3 {unexecuted_path}')\n"
                "The output file will not exist until the script is executed."
            )
            self._emit_tool_output(
                "completion_gate",
                {
                    "step": "script_execution_enforcement",
                    "passed": False,
                    "script_path": unexecuted_path,
                },
                completion_feedback,
            )
            log.info("Finalize BLOCKED by script_execution_enforcement gate", iteration=iteration, script=unexecuted_path)
            return False, "", finish_success, completion_feedback, python_worker_attempted

        # ── Failed shell + pip install without retry ─────────────────
        # When a shell command failed (e.g. missing module), the LLM
        # installed the dependency via pip, but never re-ran the
        # original command.  Block finalization so the actual output
        # gets produced.
        if iteration < (hard_turn_iterations - 1) and self.session:
            _shell_msgs = [
                m for m in self.session.messages[turn_start_idx:]
                if m.get("role") == "tool"
                and str(m.get("tool_name", "")).strip().lower() == "shell"
            ]
            _has_shell_failure = any(
                str(m.get("content", "")).strip().lower().startswith("error:")
                for m in _shell_msgs
            )
            _has_pip_install = any(
                "pip install" in str((m.get("tool_arguments") or {}).get("command", "")).lower()
                and not str(m.get("content", "")).strip().lower().startswith("error:")
                for m in _shell_msgs
            )
            # Check if there was a successful non-pip shell call AFTER
            # the pip install (meaning the LLM did re-run something).
            _has_post_pip_success = False
            _pip_seen = False
            for _sm in _shell_msgs:
                _cmd = str((_sm.get("tool_arguments") or {}).get("command", "")).lower()
                _is_err = str(_sm.get("content", "")).strip().lower().startswith("error:")
                if "pip install" in _cmd and not _is_err:
                    _pip_seen = True
                elif _pip_seen and not _is_err and "pip" not in _cmd:
                    _has_post_pip_success = True
                    break
            if _has_shell_failure and _has_pip_install and not _has_post_pip_success:
                completion_feedback = (
                    "Completion gate: a shell command failed earlier this turn and "
                    "a dependency was installed via pip, but the original command "
                    "was never re-executed.\n"
                    "You MUST re-run the original script/command that failed to "
                    "actually produce the output file. The pip install alone does "
                    "NOT create the output."
                )
                self._emit_tool_output(
                    "completion_gate",
                    {
                        "step": "pip_install_retry_enforcement",
                        "passed": False,
                    },
                    completion_feedback,
                )
                log.info("Finalize BLOCKED by pip_install_retry_enforcement gate", iteration=iteration)
                return False, "", finish_success, completion_feedback, python_worker_attempted

        # ── Datastore save verification ────────────────────────────
        # When the LLM performed datastore write operations (insert,
        # import_file, create_table, update, update_column) this turn,
        # verify the data actually persisted by querying the datastore.
        # If verification fails, block finalization and tell the LLM
        # to retry the save operation.
        ds_saves = self._turn_collect_datastore_saves(turn_start_idx)
        if ds_saves and iteration < (hard_turn_iterations - 1):
            ds_failures = await self._verify_datastore_saves(ds_saves)
            if ds_failures:
                failed_details = "; ".join(
                    f"{f['table']} ({f['action']}): {f['reason']}"
                    for f in ds_failures
                )
                completion_feedback = (
                    "Completion gate: datastore save verification FAILED.\n"
                    f"The following saves did not persist: {failed_details}\n"
                    "You MUST retry the datastore save operation to ensure the "
                    "data is actually stored.\n"
                    "Call the datastore tool again with the same action and data."
                )
                self._emit_tool_output(
                    "completion_gate",
                    {
                        "step": "datastore_save_verification",
                        "passed": False,
                        "failed_tables": [f["table"] for f in ds_failures],
                    },
                    completion_feedback,
                )
                log.info("Finalize BLOCKED by datastore_save_verification gate", iteration=iteration)
                return False, "", finish_success, completion_feedback, python_worker_attempted
            else:
                self._emit_tool_output(
                    "completion_gate",
                    {
                        "step": "datastore_save_verification",
                        "passed": True,
                        "verified_tables": [s["table"] for s in ds_saves],
                    },
                    (
                        "step=datastore_save_verification\n"
                        f"passed=True\n"
                        f"verified_tables={[s['table'] for s in ds_saves]}"
                    ),
                )

        # ── Python worker enforcement ────────────────────────────────
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
                log.info("Finalize BLOCKED by python_worker_ran gate", iteration=iteration)
                return False, "", finish_success, completion_feedback, python_worker_attempted
            if not (has_write and has_shell):
                # ── Stuck-loop safety valve ──────────────────────────
                # Track consecutive blocks where write+shell never succeed.
                # After _PW_MAX_BLOCKS, allow finalization to prevent
                # infinite loops on conversational / non-script messages.
                _pw_attr = "_pw_enforcement_streak"
                _pw_streak: int = getattr(self, _pw_attr, 0) + 1
                setattr(self, _pw_attr, _pw_streak)
                _PW_MAX_BLOCKS = 3
                if _pw_streak > _PW_MAX_BLOCKS:
                    log.warning(
                        "python_worker_enforcement stuck — allowing finalization",
                        streak=_pw_streak,
                        iteration=iteration,
                    )
                    setattr(self, _pw_attr, 0)
                    # Fall through — do NOT block.
                elif iteration < (hard_turn_iterations - 1):
                    completion_feedback = (
                        "Completion gate: execute Python worker workflow via tools before finalizing.\n"
                        "- Generate or refine a Python script/tool that handles the full item list.\n"
                        "- Run it through shell.\n"
                        "- Then provide final concise summary."
                    )
                    log.info("Finalize BLOCKED by python_worker_enforcement gate", iteration=iteration)
                    return False, "", finish_success, completion_feedback, python_worker_attempted
            else:
                # write+shell succeeded — reset streak.
                setattr(self, "_pw_enforcement_streak", 0)

        # ── List member coverage ─────────────────────────────────────
        if bool(list_task_plan.get("enabled", False)):
            members = list_task_plan.get("members")
            if isinstance(members, list) and members:
                # When a Python script was successfully executed this
                # turn, it handled the entire task outside the scale
                # system.  The script's output lives in a file (not in
                # the LLM response), so _evaluate_list_member_coverage
                # would never find textual evidence of each member and
                # would block finalization forever.  Skip the gate.
                # ── Bypass: script execution covers the entire task ──
                _skip_coverage = False
                if self._turn_has_successful_script_execution(turn_start_idx):
                    log.info(
                        "Skipping list_member_coverage — script execution covered task",
                        member_count=len(members),
                    )
                    _skip_coverage = True

                # When the browser tool was successfully used this turn,
                # URL-only members are satisfied — the browser already
                # visited/processed them.  The list task plan often
                # false-positives on browser automation prompts that
                # contain URLs.
                if not _skip_coverage:
                    _browser_used = self._turn_has_successful_tool(turn_start_idx, "browser")
                    if _browser_used:
                        _all_urls = all(
                            str(m).strip().startswith(("http://", "https://"))
                            for m in members
                        )
                        if _all_urls:
                            log.info(
                                "Skipping list_member_coverage — browser tool covered URL members",
                                member_count=len(members),
                            )
                            _skip_coverage = True

                # When the write tool produced a file AND final_action
                # is write_file, the file content lives on disk — not in
                # the conversation messages that the coverage gate scans.
                # The write-file enforcement gate (above) already verified
                # a file was produced, so skip the member-text check.
                if not _skip_coverage:
                    _fa = str(list_task_plan.get("final_action", "")).strip().lower()
                    if _fa == "write_file" and self._turn_has_successful_tool(turn_start_idx, "write"):
                        log.info(
                            "Skipping list_member_coverage — write tool produced output file",
                            member_count=len(members),
                        )
                        _skip_coverage = True

                if _skip_coverage:
                    members = []  # prevent downstream evaluation

                _sp = getattr(self, "_scale_progress", None)
                _sp_items = _sp.get("items", []) if _sp else []
                _sp_done = _sp.get("done_items", set()) if _sp else set()
                _sp_total = len(_sp_items)
                _sp_completed = len(_sp_done)
                _sp_all_done = _sp_total > 0 and _sp_completed >= _sp_total

                if _skip_coverage:
                    # Bypass already logged; treat all members as covered.
                    covered_members = list(members)
                    missing_members: list[str] = []
                elif _sp_all_done:
                    covered_members = [str(m) for m in members]
                    missing_members = []
                else:
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
                    # ── Stuck-loop detection ──────────────────────────
                    # If the gate blocks repeatedly with the same missing
                    # count the agent is stuck (e.g. file content is not
                    # in the conversation haystack).  After a few retries
                    # with zero progress, let finalisation through.
                    _streak_attr = "_coverage_gate_streak"
                    _prev_attr = "_coverage_gate_prev_missing"
                    _streak: int = getattr(self, _streak_attr, 0)
                    _prev: int = getattr(self, _prev_attr, -1)
                    _cur = len(missing_members)
                    if _cur == _prev:
                        _streak += 1
                    else:
                        _streak = 1
                    setattr(self, _streak_attr, _streak)
                    setattr(self, _prev_attr, _cur)

                    _max_retries = 3
                    if _streak > _max_retries:
                        log.warning(
                            "Coverage gate stuck — same missing count for %d consecutive blocks; allowing finalization",
                            _streak,
                            missing=_cur,
                        )
                        # Reset streak and fall through to finalize.
                        setattr(self, _streak_attr, 0)
                        setattr(self, _prev_attr, -1)
                    else:
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
                            log.info("Finalize BLOCKED by list_member_coverage gate", iteration=iteration, missing=len(missing_members), streak=_streak)
                            return False, "", finish_success, completion_feedback, python_worker_attempted
                else:
                    # Reset streak on success.
                    if hasattr(self, "_coverage_gate_streak"):
                        self._coverage_gate_streak = 0
                        self._coverage_gate_prev_missing = -1

        # ── Contract completion validation ───────────────────────────
        if completion_requirements and task_contract is not None:
            critique = await self._evaluate_contract_completion(
                user_input=effective_user_input,
                candidate_response=final_response,
                contract=task_contract,
                turn_usage=turn_usage,
                scale_completed=_scale_completed,
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
                # Post-scale synthesis: cap validation retries.  The scale
                # micro-loop already did the heavy lifting; the LLM is just
                # combining pre-researched results.  Allowing many retries
                # wastes LLM calls when the critic oscillates between
                # contradictory feedback (e.g. "deliver sequentially" vs
                # "missing topic X").
                _post_scale_retry_cap = 1
                # Also accept after just 1 retry for any task (non-scale)
                # that already produced a substantive response.  The contract
                # validation is advisory — blocking indefinitely causes
                # pipeline timeouts for straightforward requests.
                _general_retry_cap = 2

                # Cognitive mode completion strictness modifier (Layer 2).
                # High strictness (>0.7) → more retries allowed.
                # Low strictness (<0.3) → fewer retries, accept sooner.
                _mode_params = getattr(self, "_cognitive_mode_params", None)
                if _mode_params and _mode_params.completion_strictness != 0.5:
                    cs = _mode_params.completion_strictness
                    if cs > 0.7:
                        _post_scale_retry_cap = 2
                        _general_retry_cap = 3
                    elif cs < 0.3:
                        _post_scale_retry_cap = 0
                        _general_retry_cap = 1
                if _scale_completed and iteration >= _post_scale_retry_cap:
                    log.info(
                        "Post-scale validation retry cap reached — accepting response",
                        iteration=iteration,
                        failed=len(failed_items),
                        cap=_post_scale_retry_cap,
                    )
                elif iteration >= _general_retry_cap:
                    log.info(
                        "General validation retry cap reached — accepting response (advisory)",
                        iteration=iteration,
                        failed=len(failed_items),
                        cap=_general_retry_cap,
                    )
                else:
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
                        log.info("Finalize BLOCKED by contract_validation gate", iteration=iteration, failed=len(failed_items))
                        return False, "", finish_success, completion_feedback, python_worker_attempted

        # ── Finalize: persist and return ─────────────────────────────
        log.info("All completion gates passed — finalizing response", iteration=iteration)
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

        # ── Playbook rating hint (once per session, complex tasks only) ──
        final_response = self._maybe_append_playbook_hint(
            final_response,
            task_contract=task_contract,
            scale_completed=_scale_completed,
            planning_pipeline=planning_pipeline,
            turn_start_idx=turn_start_idx,
        )

        return True, final_response, finish_success, completion_feedback, python_worker_attempted

    # ------------------------------------------------------------------
    # Playbook rating hint
    # ------------------------------------------------------------------

    _PLAYBOOK_HINT = (
        "\n\n---\n"
        "💡 *If this worked well or could be improved, say "
        "\"rate good\" or \"rate bad\" to save this pattern for future tasks.*"
    )

    def _maybe_append_playbook_hint(
        self,
        response: str,
        *,
        task_contract: dict | None,
        scale_completed: bool,
        planning_pipeline: dict | None,
        turn_start_idx: int,
    ) -> str:
        """Append a one-time playbook rating hint after complex tasks.

        Criteria (all must be true):
        1. Complex task: used a task contract, scale loop, or pipeline.
        2. Hint not yet shown this session (``_playbook_hint_shown``).
        3. At least 3 tool calls happened this turn (real work, not a
           quick answer).
        """
        # Already shown this session?
        if getattr(self, "_playbook_hint_shown", False):
            return response

        # Was this a complex task?
        is_complex = bool(task_contract) or scale_completed or bool(planning_pipeline)
        if not is_complex:
            return response

        # Count tool calls this turn.
        tool_call_count = 0
        if self.session:
            for msg in self.session.messages[turn_start_idx:]:
                if msg.get("role") == "tool":
                    tool_call_count += 1
        if tool_call_count < 3:
            return response

        self._playbook_hint_shown = True
        return response + self._PLAYBOOK_HINT
