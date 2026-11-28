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
import re
from typing import Any

from captain_claw import answer_caveats, drive, file_format_check
from captain_claw.agent_stuck import MSG_EMPTY_RESPONSE
from captain_claw.logging import get_logger


log = get_logger(__name__)

# What the check before delivering shows its checker: the last tool calls of
# the turn, each shortened, and the start of the request and of the reply.
_DELIVERY_LOG_CALLS = 40
_DELIVERY_LOG_ARGS = 200
_DELIVERY_LOG_RESULT = 400
_DELIVERY_REQUEST_CHARS = 6000
_DELIVERY_REPLY_CHARS = 12000
_DELIVERY_EARLIER_CHARS = 2000  # the exchange before the request, each side
# The checker's answer, and the room on top for a model that thinks first.
_DELIVERY_ANSWER_TOKENS = 1500
_DELIVERY_THINKING_ROOM = 8000
# Tools that open a page (the search tool only lists snippets).
_PAGE_TOOLS = frozenset({"web_fetch", "web_fetch_batch", "web_get", "browser"})
_DOWNLOAD_RE = re.compile(r"\b(?:curl|wget)\b[^\n]*https?://", re.IGNORECASE)
_SNIPPET_FEEDBACK = (
    "[Check before delivering] Your answer rests on search results you didn't open: this turn ran "
    "web_search but opened no page. Search snippets are fragments, and a summary can paraphrase its "
    "source loosely. Open the pages behind the facts your answer turns on (web_fetch, or "
    "web_fetch_batch for several) and check those facts against them. Where sources disagree, read the "
    "original they cite (the law, the standard, the paper) and follow it, and say in one line where they "
    "differ. Then give your complete answer again. Do not ask the person a question, and do not mention "
    "this check."
)
# How much of the extra-work step's own words is shown under the answer.
_EXTRA_WORK_NOTE_CHARS = 1500


# How many times a zero-length reply may be re-rolled before the turn gives
# up honestly. Mirrors MAX_STALL_RETRIES: two re-rolls convert a transient
# empty generation into a real answer, and a model that is still empty on
# the third attempt is not going to converge on the fourth.
MAX_EMPTY_ANSWER_RETRIES = 2


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
        from captain_claw.datastore import resolve_datastore_manager

        # Verify against the SAME store the tool wrote to — a shared VFS-folder run
        # (CLAW_DATASTORE_VFS) writes there, not the global DB; using the wrong one
        # made this gate report false "save didn't persist" and the agent thrashed.
        dm = resolve_datastore_manager(str(self.session.id) if self.session else None)
        failures: list[dict[str, Any]] = []

        # Deduplicate: only verify the last save per table.
        tables_to_verify: dict[str, dict[str, Any]] = {}
        for save in saves:
            tables_to_verify[save["table"]] = save

        for table_name, save in tables_to_verify.items():
            action = save["action"]
            try:
                info = await dm.describe_table(table_name)
                # For insert/upsert/import_file, the table must have rows.
                if action in ("insert", "upsert", "import_file") and info.row_count == 0:
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
        # ── Extra work: its reply is not a new answer ─────────────────
        # The answer was fixed when the extra-work step began. What comes
        # back now only says what was added, so it passes none of the gates
        # below (no auto-write, no contract, no second check): it is put
        # under the answer and delivered.
        _extra = getattr(self, "_extra_work", None)
        if isinstance(_extra, dict) and _extra.get("active"):
            _extra["active"] = False
            final_response = self._extra_work_reply(output_text)
            self._emit_tool_output(
                "completion_gate",
                {"step": "extra_work", "phase": "end", "calls": int(_extra.get("calls", 0)),
                 "refused": int(_extra.get("refused", 0)), "added": final_response != _extra.get("reply")},
                f"step=extra_work\nphase=end\ncalls={_extra.get('calls', 0)}\nrefused={_extra.get('refused', 0)}",
            )
            final_response = await self._deliver_final_response(
                final_response=final_response,
                user_input=user_input,
                effective_user_input=effective_user_input,
                task_contract=None,
                scale_completed=False,
                planning_pipeline=planning_pipeline,
                turn_start_idx=turn_start_idx,
            )
            return True, final_response, finish_success, "", python_worker_attempted

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
                # The file to save to comes from the person's message as sent:
                # the effective input may carry the previous reply (merged in
                # for a clarification follow-up), and a path that reply names
                # was asked for in an earlier turn, not this one.
                request_text=user_input,
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
                or self._turn_has_successful_tool(turn_start_idx, "docx")
                or self._turn_has_successful_tool(turn_start_idx, "xlsx")
                or self._turn_has_successful_tool(turn_start_idx, "pptx")
            )
            has_script_exec = self._turn_has_successful_script_execution(turn_start_idx)
            # Tools that directly produce file output (screenshots, images)
            # satisfy write_file without needing write+shell.
            has_media_output = (
                self._turn_has_successful_tool(turn_start_idx, "browser")
                or self._turn_has_successful_tool(turn_start_idx, "image_gen")
                or self._turn_has_successful_tool(turn_start_idx, "image_manipulation")
                or self._turn_has_successful_tool(turn_start_idx, "visualize")
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
                    "2. xlsx(action='create') — for a workbook with formulas and formats; "
                    "pptx(action='create') — for a deck; "
                    "datastore(action='export') — for a plain dump of tabular data (XLSX/CSV). "
                    "For JOINs: datastore(action='export', format='xlsx', "
                    "sql_query='SELECT ... FROM t1 JOIN t2 ON ...', table='output_name').\n"
                    + ("3. write a Python script to scripts/ then execute it via "
                       "shell(command='python3 <script_path>') — for non-tabular output "
                       "(PDF, charts, custom formats). IMPORTANT: all output paths in the "
                       "script must be relative to the workspace root (not the script's "
                       "directory). Do NOT cd into the script directory.\n"
                       if self._shell_usable(turn_start_idx) else "")
                    + "Do NOT just describe the file path — actually invoke the tools."
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
        # Once per script, and only when this agent can run commands at all.
        has_unexecuted, unexecuted_path = self._turn_has_unexecuted_script(turn_start_idx)
        if (has_unexecuted and iteration < (hard_turn_iterations - 1)
                and self._script_gate_may_ask(turn_start_idx, unexecuted_path)):
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
                # A verified BULK data write (insert/upsert/import) in "passthrough"
                # list mode covers ALL members at once — mark them done so the list
                # coverage gate below doesn't then block a worker that already saved
                # its data (the create/drop thrash post-mortem). Per-item micro-loop
                # modes mark their own done_items and are unaffected.
                _sp = getattr(self, "_scale_progress", None)
                _data_saved = any(
                    s.get("action") in ("insert", "upsert", "import_file") for s in ds_saves)
                if (
                    _sp is not None and _data_saved and _sp.get("items")
                    and str(_sp.get("_extraction_mode", "")).strip().lower() == "passthrough"
                ):
                    _sp["done_items"] = set(_sp.get("items", []))
                    log.info(
                        "Scale members marked done after verified bulk datastore save",
                        members=len(_sp.get("items", [])))

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
        # Master switch: latched-off agents (code agents, non-scale workers)
        # never face the coverage gate — their completion is judged
        # externally (orchestrator/reviewers), not by reply-text scanning.
        if bool(list_task_plan.get("enabled", False)) and not self._scale_system_disabled():
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

                # When the agent saved rows to the datastore this turn (bulk
                # insert/upsert/import), the evidence lives in the DB — not echoed
                # member-by-member in the reply. The datastore_save_verification gate
                # above already confirmed it persisted, so skip the member-text check
                # (else a worker that bulk-saved gets blocked → the create/drop thrash).
                if not _skip_coverage and any(
                    s.get("action") in ("insert", "upsert", "import_file") for s in ds_saves
                ):
                    log.info(
                        "Skipping list_member_coverage — datastore save persisted the members",
                        member_count=len(members),
                    )
                    _skip_coverage = True

                # When the edit tool successfully mutated files this turn,
                # the work's evidence lives in the files on disk — never in
                # the reply text the coverage gate scans.  Fix prompts
                # ("apply these 5 fixes") false-positive as list tasks;
                # without this bypass the gate blocks a correctly finished
                # agent forever (SENKO2 hang: covered=0 despite all fixes
                # applied, because edits don't echo members into the reply).
                if not _skip_coverage and self._turn_has_successful_tool(turn_start_idx, "edit"):
                    log.info(
                        "Skipping list_member_coverage — edit tool mutated files (evidence on disk)",
                        member_count=len(members),
                    )
                    _skip_coverage = True

                # The full member list, kept for the report: a bypass empties
                # `members` below, which made a bypassed gate log covered=0.
                _all_members = [str(m) for m in members]
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
                    covered_members = list(_all_members)
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

                # Scale-loop items that failed (marked done so the loop moved on)
                # and that no later tool call opened: still missing, whatever the
                # bypasses above concluded — six unread spreadsheets must not pass
                # as covered. The stuck-loop valve below still bounds the blocks.
                _unread_failed = self._unread_failed_scale_items(turn_start_idx)
                if _unread_failed:
                    _unread_set = set(_unread_failed)
                    covered_members = [m for m in covered_members if m not in _unread_set]
                    missing_members = list(missing_members) + [
                        m for m in _unread_failed if m not in set(missing_members)]

                self._emit_tool_output(
                    "completion_gate",
                    {
                        "step": "list_member_coverage",
                        "covered": len(covered_members),
                        "missing": len(missing_members),
                        "members": len(_all_members),
                        "bypassed": _skip_coverage,
                        "unread_failed": len(_unread_failed),
                        "scale_progress_total": _sp_total,
                        "scale_progress_done": _sp_completed,
                    },
                    (
                        "step=list_member_coverage\n"
                        f"covered={len(covered_members)}\n"
                        f"missing={len(missing_members)}\n"
                        f"members={len(_all_members)}\n"
                        f"bypassed={_skip_coverage}\n"
                        f"unread_failed={len(_unread_failed)}\n"
                        f"scale_progress_total={_sp_total}\n"
                        f"scale_progress_done={_sp_completed}"
                    ),
                )
                if missing_members:
                    # ── Stuck-loop detection ──────────────────────────
                    # The gate can block a finished agent forever when the
                    # member "evidence" can never appear in the reply text
                    # (it lives in files on disk).  Two independent trips,
                    # both scoped to THIS turn:
                    #  1. No-net-progress streak: a block only counts as
                    #     progress when the missing count strictly improves
                    #     on the best (lowest) seen this turn.  The old
                    #     same-count check let OSCILLATING counts (5→4→5→4,
                    #     e.g. from line-shifted re-reads) reset the streak
                    #     forever — the SENKO2 hang.
                    #  2. Absolute cap on total blocks per turn: whatever
                    #     the counts do, the gate never blocks finalization
                    #     more than _max_blocks times.
                    # Counters are keyed to turn_start_idx so stale state
                    # from a previous turn can never leak in.
                    if getattr(self, "_coverage_gate_turn_idx", None) != turn_start_idx:
                        self._coverage_gate_turn_idx = turn_start_idx
                        self._coverage_gate_streak = 0
                        self._coverage_gate_best_missing = None
                        self._coverage_gate_blocks = 0
                    _cur = len(missing_members)
                    _best = getattr(self, "_coverage_gate_best_missing", None)
                    if _best is None or _cur < _best:
                        # Net progress — strictly fewer missing than ever
                        # before this turn.
                        self._coverage_gate_best_missing = _cur
                        _streak = 1
                    else:
                        _streak = getattr(self, "_coverage_gate_streak", 0) + 1
                    self._coverage_gate_streak = _streak
                    _blocks = getattr(self, "_coverage_gate_blocks", 0) + 1
                    self._coverage_gate_blocks = _blocks

                    _max_retries = 3
                    _max_blocks = 8
                    if _streak > _max_retries or _blocks > _max_blocks:
                        log.warning(
                            "Coverage gate stuck — %d consecutive blocks without net progress, %d total blocks this turn; allowing finalization",
                            _streak,
                            _blocks,
                            missing=_cur,
                        )
                        self._emit_tool_output(
                            "completion_gate",
                            {
                                "step": "coverage_valve_release",
                                "streak": _streak,
                                "blocks": _blocks,
                                "missing": _cur,
                            },
                            (
                                "step=coverage_valve_release\n"
                                f"streak={_streak}\n"
                                f"blocks={_blocks}\n"
                                f"missing={_cur}\n"
                                "note=coverage gate made no net progress; forcing finalization"
                            ),
                        )
                        # Reset and fall through to finalize.
                        self._coverage_gate_streak = 0
                        self._coverage_gate_best_missing = None
                        self._coverage_gate_blocks = 0
                    else:
                        completion_feedback = self._build_list_coverage_feedback(
                            missing_members=missing_members,
                            strategy=str(list_task_plan.get("strategy", "direct")).strip().lower(),
                            per_member_action=str(list_task_plan.get("per_member_action", "")).strip(),
                        )
                        if _unread_failed:
                            completion_feedback += (
                                "\nThese inputs failed in the scale loop and were never read: "
                                + ", ".join(_unread_failed[:8])
                                + (f", … (+{len(_unread_failed) - 8} more)" if len(_unread_failed) > 8 else "")
                                + ". Open each with its extractor (pdf_extract / xlsx_extract / "
                                "docx_extract / pptx_extract) or read, use the content, then finish."
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
                    # Reset valve counters on success.
                    self._coverage_gate_streak = 0
                    self._coverage_gate_best_missing = None
                    self._coverage_gate_blocks = 0

        # ── Scale-loop reply coverage gate ───────────────────────────
        # When the scale_micro_loop processed N items, the user-facing reply
        # MUST acknowledge each one. Caught a real swallow: glasses-mode agent
        # processed 2 chassis options but emitted only 1 to honour brevity. We
        # check coverage on the final text and, if it's missing items, feed the
        # existing scale completion_feedback back so the next iteration enumerates
        # them all. Bypasses past the iteration cap (same safety valve as the
        # other gates).
        try:
            _missing_scale = self._scale_reply_missing_items(_sp, final_response)
        except Exception:
            _missing_scale = []
        if _missing_scale and iteration < (hard_turn_iterations - 1):
            # Same no-net-progress valve as the list-coverage gate above:
            # only a strictly-lower missing count resets the streak, so
            # oscillating counts can't dodge the release; an absolute
            # per-turn block cap backstops everything.  Turn-keyed.
            if getattr(self, "_scale_reply_gate_turn_idx", None) != turn_start_idx:
                self._scale_reply_gate_turn_idx = turn_start_idx
                self._scale_reply_gate_streak = 0
                self._scale_reply_gate_best_missing = None
                self._scale_reply_gate_blocks = 0
            _cur = len(_missing_scale)
            _best = getattr(self, "_scale_reply_gate_best_missing", None)
            if _best is None or _cur < _best:
                self._scale_reply_gate_best_missing = _cur
                _streak = 1
            else:
                _streak = getattr(self, "_scale_reply_gate_streak", 0) + 1
            self._scale_reply_gate_streak = _streak
            _blocks = getattr(self, "_scale_reply_gate_blocks", 0) + 1
            self._scale_reply_gate_blocks = _blocks
            if _streak > 3 or _blocks > 8:
                log.warning(
                    "Scale-reply gate stuck — %d consecutive blocks without net progress, %d total blocks this turn; allowing finalization",
                    _streak, _blocks, missing=_cur,
                )
                self._scale_reply_gate_streak = 0
                self._scale_reply_gate_best_missing = None
                self._scale_reply_gate_blocks = 0
            else:
                _show = ", ".join(f"'{m}'" for m in _missing_scale[:5])
                completion_feedback = (
                    f"[Reply incomplete] The scale loop processed "
                    f"{len(_sp.get('items') or [])} item(s) this turn but your "
                    f"reply did not mention: {_show}.\n\nRewrite your reply so it "
                    "enumerates ALL processed items as short bullets — never drop a "
                    "result to stay under a length target. A complete short list "
                    "beats an incomplete shorter one. Keep each bullet tight, but "
                    "include every item."
                )
                log.info(
                    "Finalize BLOCKED by scale_reply_coverage gate",
                    iteration=iteration, missing=_cur, streak=_streak,
                )
                return False, "", finish_success, completion_feedback, python_worker_attempted
        else:
            # Reset valve counters on success.
            self._scale_reply_gate_streak = 0
            self._scale_reply_gate_best_missing = None
            self._scale_reply_gate_blocks = 0

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
                #
                # How many retries is the turn's Drive level (drive.py): none
                # at Relaxed, 2 (1 post-scale) at Diligent, one more above.
                _level = drive.current(self)
                _post_scale_retry_cap = _level.validation_retries_after_list
                _general_retry_cap = _level.validation_retries
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

        # ── Empty-answer floor ───────────────────────────────────────
        # A blank reply is not an answer. Weak local models can return
        # zero-length content over and over (a decoding grammar or a
        # reasoning block eats the generation, or an oversized KV cache
        # starves it). Without this floor the empty string passes every
        # gate above, `_persist_assistant_response` drops it as unsavable,
        # and the turn is reported as a SUCCESS — the user just sees an
        # empty chat bubble with no indication anything went wrong.
        #
        # Bounded by MAX_EMPTY_ANSWER_RETRIES rather than the iteration
        # budget: a model returning empty every time is not converging, and
        # re-rolling it to the hard ceiling would spend 80+ LLM calls to
        # reach the same conclusion three calls would.
        if not str(final_response or "").strip():
            _empty_retries = int(getattr(self, "_empty_answer_retries", 0))
            if (
                _empty_retries < MAX_EMPTY_ANSWER_RETRIES
                and iteration < (hard_turn_iterations - 1)
            ):
                self._empty_answer_retries = _empty_retries + 1
                log.warning(
                    "Finalize BLOCKED — model returned an empty response",
                    iteration=iteration,
                )
                completion_feedback = (
                    "Your last response was completely empty — the user saw nothing. "
                    "Produce the answer now, as plain text. If you cannot complete the "
                    "task, say so explicitly and explain what is blocking you."
                )
                return False, "", finish_success, completion_feedback, python_worker_attempted
            log.error(
                "Model returned an empty response and the iteration budget is spent",
                iteration=iteration,
            )
            return True, MSG_EMPTY_RESPONSE, False, completion_feedback, python_worker_attempted

        # ── Files the turn wrote that their own format can't read (same Drive switch) ──
        # Once per turn: the data files changed on disk during the turn
        # (iCalendar, JSON, XML, YAML, TOML — often written by a script, not
        # the write tool) are parsed, and one that fails sends the turn back
        # once with the place and the reason. No model call.
        if (
            drive.current(self).delivery_check
            and not getattr(self, "_format_check_done", False)
            and iteration < (hard_turn_iterations - 1)
            and self._turn_did_work(turn_start_idx)
        ):
            self._format_check_done = True
            import asyncio

            try:
                report = await asyncio.to_thread(
                    file_format_check.check,
                    getattr(getattr(self, "tools", None), "runtime_base_path", None),
                    float(getattr(self, "_turn_started_wall", 0.0) or 0.0),
                )
            except Exception as exc:  # noqa: BLE001 — a check must never end the turn
                log.warning("Format check failed", error=str(exc))
                report = []
            if report:
                completion_feedback = file_format_check.feedback(report)
                self._emit_tool_output(
                    "completion_gate",
                    {"step": "format_check", "passed": False, "files": len(report)},
                    "step=format_check\npassed=False\n" + "\n".join(f"file={line}" for line in report),
                )
                log.info("Finalize BLOCKED by format check", iteration=iteration, files=len(report))
                return False, "", finish_success, completion_feedback, python_worker_attempted

        # ── An answer built on search results nobody opened (same Drive switch) ──
        # Once per turn: the turn searched the web but opened no page, so the
        # answer stands on snippets (a model once took a report's loose
        # paraphrase of a law over the law itself that way). It goes back once
        # to open the pages its facts come from. Decided from the turn's tool
        # calls, no model call.
        if (
            drive.current(self).delivery_check
            and not getattr(self, "_snippet_check_done", False)
            and iteration < (hard_turn_iterations - 1)
            and self._turn_searched_without_opening(turn_start_idx)
        ):
            self._snippet_check_done = True
            completion_feedback = _SNIPPET_FEEDBACK
            self._emit_tool_output(
                "completion_gate", {"step": "snippet_check", "passed": False},
                "step=snippet_check\npassed=False",
            )
            log.info("Finalize BLOCKED by snippet check", iteration=iteration)
            return False, "", finish_success, completion_feedback, python_worker_attempted

        # ── An answer that takes back its own result (same Drive switch) ──
        # Once per turn, for a turn that did work with tools: when the reply
        # or a document the turn wrote qualifies its own result ("read
        # strictly, it would be…", "if X is a hard requirement…"), it goes
        # back once to settle on the reading its evidence supports. Found by
        # wording (answer_caveats), no model call.
        if (
            drive.current(self).delivery_check
            and not getattr(self, "_caveat_check_done", False)
            and iteration < (hard_turn_iterations - 1)
            and self._turn_did_work(turn_start_idx)
        ):
            self._caveat_check_done = True
            caveats = answer_caveats.find_caveats(
                [str(final_response or ""), *self._turn_written_documents(turn_start_idx)])
            if caveats:
                completion_feedback = answer_caveats.feedback(caveats)
                self._emit_tool_output(
                    "completion_gate",
                    {"step": "caveat_check", "passed": False, "caveats": len(caveats)},
                    "step=caveat_check\npassed=False\n" + "\n".join(f"caveat={c}" for c in caveats),
                )
                log.info("Finalize BLOCKED by caveat check", iteration=iteration, caveats=len(caveats))
                return False, "", finish_success, completion_feedback, python_worker_attempted

        # ── Check before delivering (Drive: Persistent and above) ────
        # Once per turn, for a turn that did work with tools and was not
        # already judged against a contract: compare the reply with what was
        # asked, and send it back when something asked for is missing.
        if (
            drive.current(self).delivery_check
            and not getattr(self, "_delivery_check_done", False)
            and not (completion_requirements and task_contract is not None)
            and iteration < (hard_turn_iterations - 1)
            and self._turn_did_work(turn_start_idx)
        ):
            self._delivery_check_done = True
            gaps = await self._delivery_check(
                user_input=effective_user_input,
                candidate_response=final_response,
                turn_start_idx=turn_start_idx,
                turn_usage=turn_usage,
            )
            if gaps:
                completion_feedback = (
                    "[Check before delivering] Compared with what was asked, your reply "
                    "is missing or does not support the following:\n"
                    + "\n".join(f"- {gap}" for gap in gaps)
                    + "\n\nIf a point is right, finish it now with your tools. If it is "
                    "wrong, or cannot be done, leave it. Then give your complete answer "
                    "again, and say plainly what is still missing or unverified. Do not "
                    "ask the person a question, and do not mention this check."
                )
                self._emit_tool_output(
                    "completion_gate",
                    {"step": "delivery_check", "passed": False, "gaps": len(gaps)},
                    "step=delivery_check\npassed=False\n" + "\n".join(f"gap={gap}" for gap in gaps),
                )
                log.info("Finalize BLOCKED by delivery check", iteration=iteration, gaps=len(gaps))
                return False, "", finish_success, completion_feedback, python_worker_attempted
            self._emit_tool_output(
                "completion_gate",
                {"step": "delivery_check", "passed": True, "gaps": 0},
                "step=delivery_check\npassed=True",
            )

        # ── Extra work (Drive: Ambitious, where switched on) ──────────
        # The asked work is done and has passed every gate. Before it is
        # delivered, the agent gets one bounded chance to add the obvious
        # next step; its answer stays as it is, and whatever it adds is
        # shown under it, marked as not asked for.
        if self._extra_work_may_start(iteration, hard_turn_iterations, turn_start_idx, final_response):
            limit = self._extra_work_limit()
            self._extra_work = {"active": True, "reply": final_response, "calls": 0, "refused": 0, "limit": limit}
            completion_feedback = self.instructions.render("drive_extra_work_prompt.md", limit=limit)
            self._emit_tool_output(
                "completion_gate", {"step": "extra_work", "phase": "start", "limit": limit},
                f"step=extra_work\nphase=start\nlimit={limit}",
            )
            log.info("Extra work offered before delivering", iteration=iteration, limit=limit)
            return False, "", finish_success, completion_feedback, python_worker_attempted

        # ── Finalize: persist and return ─────────────────────────────
        log.info("All completion gates passed — finalizing response", iteration=iteration)
        final_response = await self._deliver_final_response(
            final_response=final_response,
            user_input=user_input,
            effective_user_input=effective_user_input,
            task_contract=task_contract,
            scale_completed=_scale_completed,
            planning_pipeline=planning_pipeline,
            turn_start_idx=turn_start_idx,
        )
        return True, final_response, finish_success, completion_feedback, python_worker_attempted

    async def _deliver_final_response(
        self,
        *,
        final_response: str,
        user_input: str,
        effective_user_input: str,
        task_contract: dict[str, Any] | None,
        scale_completed: Any,
        planning_pipeline: dict[str, Any] | None,
        turn_start_idx: int,
    ) -> str:
        """Persist the reply every gate let through and do what follows a
        delivered answer; returns the text to deliver."""
        self._update_clarification_state(
            user_input=user_input,
            effective_user_input=effective_user_input,
            assistant_response=final_response,
        )
        await self._persist_assistant_response(final_response)
        # The owner's todo/contacts/scripts/APIs stores are never written
        # from a shared-agent member's turn — nor (PR D) from a turn that
        # read members' private data.
        # Nor from an onboarding exam's question, which keeps nothing.
        from captain_claw import exam_turn, member_privacy
        if (getattr(self, "_speaker_scoped", False) is not True and not member_privacy.private_turn(self)
                and not exam_turn.active(self)):
            # A reply carry-over flagged (it restated member text from
            # context) is not scanned: a to-do lands in every later prompt.
            _capture_reply = "" if member_privacy.output_level(self) else final_response
            await self._auto_capture_todos(effective_user_input, _capture_reply)
            await self._auto_capture_contacts(effective_user_input, final_response)
            await self._auto_capture_scripts(effective_user_input, final_response)
            await self._auto_capture_apis(effective_user_input, final_response)

        # ── Playbook rating hint (once per session, complex tasks only) ──
        return self._maybe_append_playbook_hint(
            final_response,
            task_contract=task_contract,
            scale_completed=scale_completed,
            planning_pipeline=planning_pipeline,
            turn_start_idx=turn_start_idx,
        )

    # ------------------------------------------------------------------
    # Extra work (Drive)
    # ------------------------------------------------------------------

    def _extra_work_limit(self) -> int:
        """Tool calls an extra-work step may make: the level's share of a
        turn's base step budget, at least 3."""
        base = int(getattr(self, "max_iterations", 40) or 40)
        return max(3, round(base * drive.current(self).initiative_share))

    def _extra_work_may_start(
        self, iteration: int, hard_turn_iterations: int, turn_start_idx: int, final_response: str,
    ) -> bool:
        """Whether this finished turn gets the extra-work step: Ambitious with
        the agent's switch on, once, for a person's own turn that did real
        work and still has steps left."""
        if getattr(self, "_extra_work", None) is not None:
            return False                      # once per turn
        if not (drive.current(self).initiative and drive.initiative_enabled()):
            return False
        if not str(final_response or "").strip():
            return False
        # Room for the step and its reply, with a margin: a turn near its
        # ceiling delivers what it has.
        if hard_turn_iterations - iteration < self._extra_work_limit() + 5:
            return False
        # Unasked work belongs to a person at the keyboard: a scheduled run
        # or an automation's message would repeat it unseen. Nor on a shared
        # agent member's turn — the workspace is the owner's.
        origin = getattr(self, "_turn_origin", None)
        if not (isinstance(origin, tuple) and origin and str(origin[0] or "") == "human"):
            return False
        if getattr(self, "_speaker_scoped", False) is True:
            return False
        return self._turn_did_work(turn_start_idx)

    def _extra_work_rescue(self, text: Any, success: bool) -> tuple[Any, bool]:
        """What a turn delivers when it ends with the extra-work step still
        open: the finished answer, as a success. Anything else is unchanged."""
        state = getattr(self, "_extra_work", None)
        if not (isinstance(state, dict) and state.get("active")):
            return text, success
        state["active"] = False
        answer = str(state.get("reply", "") or "")
        if not answer.strip():
            return text, success
        log.warning("Turn ended during extra work — delivering the finished answer",
                    calls=state.get("calls", 0), ended_with=str(text or "")[:120])
        try:
            self._add_session_message("assistant", answer)
        except Exception as exc:
            log.warning("Finished answer not added to the session", error=str(exc))
        return answer, True

    def _extra_work_reply(self, extra_text: str) -> str:
        """The answer as delivered, with what the extra-work step added shown
        under it — or the answer alone when the step added nothing."""
        state = self._extra_work or {}
        answer = str(state.get("reply", "") or "")
        added = str(extra_text or "").strip()
        nothing = (
            not added
            or int(state.get("calls", 0)) == 0          # words alone are not work
            or added.upper().rstrip(". ").startswith(drive.NOTHING_TO_ADD)
        )
        if nothing:
            return answer
        return f"{answer}\n\n---\n**Also done (not asked for)**\n\n{added[:_EXTRA_WORK_NOTE_CHARS]}"

    # ------------------------------------------------------------------
    # Check before delivering (Drive)
    # ------------------------------------------------------------------

    def _turn_work_messages(self, turn_start_idx: int) -> list[dict[str, Any]]:
        """This turn's tool results, leaving out tools that only watch."""
        session = getattr(self, "session", None)
        messages = getattr(session, "messages", None) or []
        out: list[dict[str, Any]] = []
        for msg in messages[turn_start_idx:]:
            if not isinstance(msg, dict) or str(msg.get("role", "")).strip().lower() != "tool":
                continue
            name = str(msg.get("tool_name", "")).strip().lower()
            if name and self._is_monitor_only_tool_name(name):
                continue
            out.append(msg)
        return out

    def _turn_searched_without_opening(self, turn_start_idx: int) -> bool:
        """Whether this turn ran a web search that returned results but tried
        to open no page, while the agent has a tool that could open one."""
        tools = getattr(self, "tools", None)
        try:
            can_open = tools is not None and any(tools.has_tool(name) for name in _PAGE_TOOLS)
        except Exception:  # noqa: BLE001
            can_open = False
        if not can_open:
            return False
        searched = False
        for msg in self._turn_work_messages(turn_start_idx):
            name = str(msg.get("tool_name", "")).strip().lower()
            if name in _PAGE_TOOLS or name == "pdf_extract":
                return False
            # A source downloaded and read another way (curl the PDF, then read it).
            if name == "shell" and _DOWNLOAD_RE.search(str((msg.get("tool_arguments") or {}).get("command", ""))):
                return False
            if name == "web_search" and not str(msg.get("content", "")).lstrip().startswith(("Error", "❌")):
                searched = True
        return searched

    def _turn_written_documents(self, turn_start_idx: int) -> list[str]:
        """The prose this turn wrote to files (a write's content, an edit's
        new text), for the caveat check: reports and answers, not code or data."""
        from pathlib import PurePath

        session = getattr(self, "session", None)
        messages = getattr(session, "messages", None) or []
        texts: list[str] = []
        for msg in messages[turn_start_idx:]:
            if not isinstance(msg, dict) or str(msg.get("role", "")).strip().lower() != "tool":
                continue
            if str(msg.get("tool_name", "")).strip().lower() not in ("write", "file_write", "edit"):
                continue
            if str(msg.get("content", "")).lstrip().startswith(("Error", "❌", "Not written", "Refused")):
                continue
            args = msg.get("tool_arguments") or {}
            if not isinstance(args, dict):
                continue
            if PurePath(str(args.get("path") or "")).suffix.lower() not in answer_caveats.PROSE_SUFFIXES:
                continue
            text = args.get("content") if args.get("content") is not None else args.get("new_string")
            if isinstance(text, str) and text.strip():
                texts.append(text)
        return texts

    def _turn_did_work(self, turn_start_idx: int) -> bool:
        """Whether the turn called a tool: a reply made of words alone (small
        talk, a simple answer) is not checked."""
        return bool(self._turn_work_messages(turn_start_idx))

    def _delivery_work_log(self, turn_start_idx: int) -> str:
        """What the turn did, shortened for the checker: each tool call with
        the start of its arguments and of its result."""
        lines: list[str] = []
        work = self._turn_work_messages(turn_start_idx)
        for msg in work[-_DELIVERY_LOG_CALLS:]:
            args = msg.get("tool_arguments")
            try:
                arg_text = json.dumps(args, ensure_ascii=False, default=str) if args else ""
            except Exception:
                arg_text = str(args)
            result = " ".join(str(msg.get("content", "")).split())
            lines.append(
                f"- {str(msg.get('tool_name', '') or 'tool')}"
                f"({arg_text[:_DELIVERY_LOG_ARGS]}) -> {result[:_DELIVERY_LOG_RESULT] or '(no output)'}"
            )
        if len(work) > _DELIVERY_LOG_CALLS:
            lines.insert(0, f"({len(work) - _DELIVERY_LOG_CALLS} earlier tool call(s) not shown)")
        return "\n".join(lines) or "(no tool calls)"

    def _delivery_earlier(self, turn_start_idx: int) -> str:
        """The exchange just before this turn — the person's previous message
        and the reply to it, each cut short — so the checker can tell what a
        short request ("compare them") points back to. Notes other lanes put
        in the conversation ("[Autonomous Work — …]") are skipped."""
        from captain_claw import msg_origin

        def _plain(content: Any) -> str:
            if isinstance(content, list):
                content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
            return " ".join(str(content or "").split())

        messages = list(getattr(getattr(self, "session", None), "messages", None) or [])
        earlier = messages[: max(0, min(int(turn_start_idx or 0), len(messages)))]
        asked = answered = ""
        for msg in reversed(earlier):
            if not isinstance(msg, dict):
                continue
            if msg_origin.is_human_input(msg):
                asked = _plain(msg.get("content"))
                break
            if (not answered and str(msg.get("role", "")).strip().lower() == "assistant"
                    and not msg.get("tool_calls")):
                text = _plain(msg.get("content"))
                if text and not text.startswith("["):
                    answered = text
        if not asked and not answered:
            return "(nothing before it)"
        cut = _DELIVERY_EARLIER_CHARS
        return (f"The person: {asked[:cut] + (' …' if len(asked) > cut else '') or '(not found)'}\n"
                f"The reply: {answered[:cut] + (' …' if len(answered) > cut else '') or '(none)'}")

    async def _delivery_check(
        self,
        *,
        user_input: str,
        candidate_response: str,
        turn_start_idx: int,
        turn_usage: dict[str, int],
    ) -> list[str]:
        """What the reply still owes the request, as the checker sees it: up
        to five short statements, or none. Anything that goes wrong (the call
        fails, the answer is not the JSON asked for) means none — the check
        may add a round of work, never block a reply."""
        try:
            from captain_claw.llm import Message

            messages = [
                Message(role="system", content=self.instructions.load("delivery_check_system_prompt.md")),
                Message(role="user", content=self.instructions.render(
                    "delivery_check_user_prompt.md",
                    user_input=str(user_input or "")[:_DELIVERY_REQUEST_CHARS],
                    earlier=self._delivery_earlier(turn_start_idx),
                    work_log=self._delivery_work_log(turn_start_idx),
                    candidate_response=str(candidate_response or "")[:_DELIVERY_REPLY_CHARS],
                )),
            ]
            from captain_claw.config import get_config

            cap = int(get_config().model.max_tokens)
            # The check thinks at the turn's effort, and a thinking model
            # counts its thinking against max_tokens: with 1500 alone, Qwen 3.8
            # at xhigh ran out mid-thought on 33 of 50 checks (WildClawBench,
            # 2026-11-26), and a cut-off check passes everything. So it gets
            # room to think; if it still runs out before the JSON, it is asked
            # once more with thinking off.
            response = await self._complete_with_guards(
                messages=messages,
                tools=None,
                interaction_label="delivery_check",
                turn_usage=turn_usage,
                max_tokens=min(_DELIVERY_ANSWER_TOKENS + _DELIVERY_THINKING_ROOM, cap),
            )
            payload = self._extract_json_object(response.content or "")
            if not isinstance(payload, dict) and getattr(response, "finish_reason", "") == "length":
                log.warning("Delivery check ran out of tokens — asking again with thinking off")
                response = await self._complete_with_guards(
                    messages=messages,
                    tools=None,
                    interaction_label="delivery_check_quiet",
                    turn_usage=turn_usage,
                    max_tokens=min(_DELIVERY_ANSWER_TOKENS, cap),
                )
                payload = self._extract_json_object(response.content or "")
        except Exception as exc:
            log.warning("Delivery check failed — delivering as is", error=str(exc))
            return []
        if not isinstance(payload, dict) or payload.get("complete") is not False:
            return []
        gaps = payload.get("gaps")
        if not isinstance(gaps, list):
            return []
        return [" ".join(str(g).split())[:300] for g in gaps if str(g or "").strip()][:5]

    # ------------------------------------------------------------------
    # Playbook rating hint
    # ------------------------------------------------------------------

    _PLAYBOOK_HINT = (
        "\n\n---\n"
        "💡 *If this worked well or could be improved, say "
        "\"rate good\" or \"rate bad\" to save this pattern for future tasks.*"
    )

    def _scale_reply_missing_items(self, sp: dict | None, reply: str) -> list[str]:
        """Items the scale loop processed this turn that the reply doesn't mention.

        Only fires when the loop fully completed (done_items == items) and there
        were ≥2 items — single-item tasks don't need an explicit enumeration. The
        match is forgiving: a meaningful token from each item label must appear
        somewhere in the reply (case-insensitive, ignoring punctuation). This
        catches the "two chassis → one mentioned" swallow without false-positives
        when the agent legitimately rephrases each item."""
        if not isinstance(sp, dict):
            return []
        items = sp.get("items") or []
        if not items or len(items) < 2:
            return []
        done = sp.get("done_items") or set()
        if len(done) < len(items):
            return []  # loop hasn't finished — other gates handle that
        text = (reply or "").lower()
        if not text:
            return list(items)
        import re as _re

        _stop = {"with", "from", "that", "this", "your", "yours", "have", "into",
                 "about", "their", "these", "those", "where", "which", "when",
                 "than", "what", "want", "need", "more", "less", "some", "many",
                 "item", "items", "task", "tasks", "list", "step", "steps"}

        def _tokens(label: str) -> list[str]:
            return [w for w in _re.findall(r"[a-z0-9]{4,}", str(label).lower()) if w not in _stop]

        # Drop tokens that appear in EVERY item — they don't fingerprint a
        # specific item (e.g. "chassis" in both "wheel chassis" and "bipedal
        # chassis" would otherwise let "wheel chassis: PiCar" cover both).
        all_token_sets = [set(_tokens(it)) for it in items]
        common = set.intersection(*all_token_sets) if all_token_sets else set()
        missing: list[str] = []
        for raw, toks_set in zip(items, all_token_sets):
            distinct = [t for t in toks_set if t not in common][:5]
            if not distinct:
                # Nothing distinctive to match — give the benefit of the doubt.
                continue
            if not any(t in text for t in distinct):
                missing.append(str(raw)[:80])
        return missing

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

        # Count tool calls this turn (monitor rows and the scale loop's
        # progress cards are not tool calls).
        from captain_claw.msg_origin import is_model_hidden_tool

        tool_call_count = 0
        if self.session:
            for msg in self.session.messages[turn_start_idx:]:
                if msg.get("role") == "tool" and not is_model_hidden_tool(msg):
                    tool_call_count += 1
        if tool_call_count < 3:
            return response

        self._playbook_hint_shown = True
        return response + self._PLAYBOOK_HINT

    # Tools whose output only LISTS file names — a listing after the failure is
    # not an opening of the file.
    _LISTING_TOOLS = frozenset({"glob", "vfs"})

    def _unread_failed_scale_items(self, turn_start_idx: int) -> list[str]:
        """Scale-loop items that failed during this turn and that no successful
        tool call since has opened.

        The loop marks a failed item done so it can move on; on its own that let
        unread files pass the coverage gate. An item counts as opened once a
        non-error tool call AFTER its failure names the file in its arguments
        (the extractor, a ``read``, a shell script) or its output (a script that
        printed it) — a plain listing (glob, vfs) doesn't count.
        """
        sp = getattr(self, "_scale_progress", None)
        failed_at: dict = (sp or {}).get("failed_items_at") or {}
        failed = [(item, int(failed_at.get(item, turn_start_idx)))
                  for item in ((sp or {}).get("failed_items") or {})]
        failed = [(item, at) for item, at in failed if at >= turn_start_idx]
        if not failed or not self.session:
            return []
        from captain_claw.msg_origin import is_model_hidden_tool

        messages = self.session.messages
        blobs: list[tuple[int, str]] = []
        for idx in range(min(at for _, at in failed), len(messages)):
            msg = messages[idx]
            if msg.get("role") != "tool" or is_model_hidden_tool(msg):
                continue
            if str(msg.get("tool_name", "")).strip().lower() in self._LISTING_TOOLS:
                continue
            content = str(msg.get("content", "")).strip()
            if content.lower().startswith("error:"):
                continue
            try:
                args = json.dumps(msg.get("tool_arguments") or {}, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                args = str(msg.get("tool_arguments") or "")
            blobs.append((idx, f"{args}\n{content[:20000]}".casefold()))
        unread: list[str] = []
        for item, at in failed:
            name = str(item).rstrip("/\\").rsplit("/", 1)[-1].strip().casefold()
            if not name or not any(idx >= at and name in blob for idx, blob in blobs):
                unread.append(item)
        return unread
