"""Reasoning/contract/list planning helpers for Agent."""

from datetime import UTC, datetime
import json
import re
from typing import Any

from captain_claw.config import get_config
from captain_claw.llm import Message


class AgentReasoningMixin:
    """Clarification handling, contract planning, and list-task reasoning."""
    @staticmethod
    def _request_references_all_sources(user_input: str) -> bool:
        """Detect intent to cover all referenced links/sources."""
        text = (user_input or "").strip().lower()
        if not text:
            return False
        patterns = (
            r"\bcheck all (?:those )?(?:sources|links)\b",
            r"\ball (?:those )?(?:sources|links)\b",
            r"\beach source\b",
            r"\bevery source\b",
            r"\bper source\b",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _assistant_requests_clarification(response_text: str) -> bool:
        """Heuristic: whether assistant is asking user to clarify before execution."""
        text = re.sub(r"\s+", " ", (response_text or "").strip())
        if not text:
            return False
        lowered = text.lower()
        if lowered.count("?") >= 2:
            return True
        prompts = (
            "which would you like",
            "do you want me to",
            "would you like me to",
            "tell me your choices",
            "quick questions",
            "so i proceed correctly",
            "confirm",
        )
        if any(phrase in lowered for phrase in prompts) and "?" in lowered:
            return True
        return False

    @staticmethod
    def _should_apply_pending_clarification(user_input: str) -> bool:
        """Whether current user text looks like a clarification answer."""
        text = (user_input or "").strip()
        if not text:
            return False
        if text.startswith("/"):
            return False
        if text.count("\n") > 4:
            return False
        words = re.findall(r"\S+", text)
        if not words:
            return False
        if len(words) > 40:
            return False
        if text.endswith("?"):
            return False
        return True

    def _resolve_effective_user_input(self, user_input: str) -> tuple[str, bool]:
        """Merge pending clarification anchor with current message when appropriate."""
        if not self.session or not isinstance(self.session.metadata, dict):
            return user_input, False
        state = self.session.metadata.get("clarification_state")
        if not isinstance(state, dict) or not bool(state.get("pending", False)):
            return user_input, False
        anchor = str(state.get("anchor_request", "")).strip()
        if not anchor:
            return user_input, False
        if not self._should_apply_pending_clarification(user_input):
            return user_input, False
        merged = (
            f"{anchor}\n\n"
            "Clarifications/preferences from user:\n"
            f"{user_input.strip()}\n\n"
            "Execute the full original request using these clarification details."
        )
        return merged, True

    def _update_clarification_state(
        self,
        user_input: str,
        effective_user_input: str,
        assistant_response: str,
    ) -> None:
        """Track unresolved clarification context across turns."""
        if not self.session:
            return
        meta = self.session.metadata.setdefault("clarification_state", {})
        if not isinstance(meta, dict):
            self.session.metadata["clarification_state"] = {}
            meta = self.session.metadata["clarification_state"]
        now_iso = datetime.now(UTC).isoformat()
        if self._assistant_requests_clarification(assistant_response):
            meta["pending"] = True
            meta["anchor_request"] = str(effective_user_input or user_input).strip()[:12000]
            meta["updated_at"] = now_iso
            return
        if bool(meta.get("pending", False)):
            meta["pending"] = False
            meta.pop("anchor_request", None)
            meta["updated_at"] = now_iso

    @staticmethod
    def _should_run_source_report_pipeline(user_input: str, source_urls: list[str]) -> bool:
        """Whether request asks for all/each sources to be checked and reported."""
        if not source_urls:
            return False
        text = (user_input or "").strip().lower()
        if not text:
            return False
        trigger = any(
            re.search(pattern, text)
            for pattern in (
                r"\bcheck all (?:those )?sources\b",
                r"\ball sources\b",
                r"\beach source\b",
                r"\bper source\b",
                r"\bsource[- ]distinguish(?:ed)?\b",
            )
        )
        report_intent = any(word in text for word in ("report", "summar", "compile"))
        return trigger or (report_intent and "source" in text)

    async def _run_source_report_prefetch(
        self,
        source_urls: list[str],
        turn_usage: dict[str, int],
        max_chars_per_source: int = 4500,
        pipeline_label: str = "source_report_pipeline",
    ) -> dict[str, Any]:
        """Prefetch all source URLs via web_fetch for source-report tasks."""
        if not source_urls:
            return {"requested": 0, "fetched": 0, "failed": 0}
        if "web_fetch" not in self.tools.list_tools():
            return {"requested": len(source_urls), "fetched": 0, "failed": len(source_urls), "reason": "web_fetch_disabled"}

        fetched = 0
        failed = 0
        total = len(source_urls)
        self._emit_tool_output(
            pipeline_label,
            {"step": "prefetch_start", "sources": total},
            f"step=prefetch_start\nsources={total}",
        )
        for idx, url in enumerate(source_urls, start=1):
            args = {
                "url": url,
                "extract_mode": "text",
                "max_chars": max_chars_per_source,
            }
            try:
                result = await self._execute_tool_with_guard(
                    name="web_fetch",
                    arguments=args,
                    interaction_label=f"source_report_prefetch_{idx}",
                    turn_usage=turn_usage,
                )
                output = result.content if result.success else f"Error: {result.error}"
            except Exception as e:
                result = None
                output = f"Error: {str(e)}"

            tagged_output = f"[SOURCE {idx}/{total}] {url}\n{output}"
            self._add_session_message(
                role="tool",
                content=tagged_output,
                tool_name="web_fetch",
                tool_arguments=args,
            )
            self._emit_tool_output("web_fetch", args, tagged_output)
            if result and result.success:
                fetched += 1
            else:
                failed += 1

        self._emit_tool_output(
            pipeline_label,
            {"step": "prefetch_done", "sources": total, "fetched": fetched, "failed": failed},
            (
                "step=prefetch_done\n"
                f"sources={total}\n"
                f"fetched={fetched}\n"
                f"failed={failed}"
            ),
        )
        return {"requested": total, "fetched": fetched, "failed": failed}

    @staticmethod
    def _count_source_sections(text: str) -> int:
        """Count `Source <n>` headings in a report text."""
        if not text:
            return 0
        return len(re.findall(r"(?im)^\s{0,3}(?:#+\s*)?source\s+\d+\b", text))

    @staticmethod
    def _has_conclusion_section(text: str) -> bool:
        """Detect whether report contains a `Conclusion` section heading."""
        if not text:
            return False
        return bool(re.search(r"(?im)^\s{0,3}(?:#+\s*)?conclusion\b", text))

    def _validate_source_report_response(
        self,
        response_text: str,
        expected_sources: int,
    ) -> tuple[bool, str]:
        """Validate report completeness for source-by-source requests."""
        expected = max(1, int(expected_sources))
        actual_sources = self._count_source_sections(response_text)
        has_conclusion = self._has_conclusion_section(response_text)
        if actual_sources < expected:
            return False, f"source sections missing ({actual_sources}/{expected})"
        if not has_conclusion:
            return False, "missing conclusion section"
        return True, ""

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
        """Extract the first valid JSON object from model text."""
        text = (raw_text or "").strip()
        if not text:
            return None

        candidates: list[str] = [text]
        fenced_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(fenced_matches)
        inline_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if inline_match:
            candidates.append(inline_match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    @staticmethod
    def _should_use_contract_pipeline(
        user_input: str,
        planning_enabled: bool,
        pipeline_mode: str | None = None,
    ) -> bool:
        """Use explicit user-selected mode only (no automatic switching)."""
        del user_input  # mode-driven decision for now
        mode = str(pipeline_mode or "").strip().lower()
        if mode == "contracts":
            return True
        if mode == "loop":
            # Backward compatibility: some call sites/tests toggle planning_enabled
            # directly without updating pipeline_mode.
            return bool(planning_enabled)
        # Backward-compat fallback for call sites/session metadata not yet migrated.
        return bool(planning_enabled)

    @staticmethod
    def _normalize_contract_tasks(
        raw_tasks: Any,
        max_tasks: int = 8,
        max_depth: int = 4,
        max_total_nodes: int = 36,
    ) -> list[dict[str, Any]]:
        """Normalize planner task items into a nested task tree."""
        if isinstance(raw_tasks, dict):
            source_tasks: list[Any] = [raw_tasks]
        elif isinstance(raw_tasks, list):
            source_tasks = list(raw_tasks)
        else:
            return []

        normalized: list[dict[str, Any]] = []
        next_id = 0

        def _extract_title(item: Any) -> str:
            if isinstance(item, dict):
                for key in ("title", "task", "name", "step", "summary", "description"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                return ""
            return str(item).strip()

        def _extract_children(item: Any) -> list[Any]:
            if not isinstance(item, dict):
                return []
            for key in ("children", "tasks", "subtasks", "steps", "items"):
                value = item.get(key)
                if isinstance(value, list):
                    return value
            return []

        def _visit(item: Any, depth: int) -> dict[str, Any] | None:
            nonlocal next_id
            if next_id >= max_total_nodes:
                return None
            title = _extract_title(item)
            if not title:
                return None
            next_id += 1
            node: dict[str, Any] = {"id": f"task_{next_id}", "title": title[:180]}
            if depth >= max_depth:
                return node
            child_nodes: list[dict[str, Any]] = []
            for child in _extract_children(item):
                if next_id >= max_total_nodes:
                    break
                normalized_child = _visit(child, depth + 1)
                if normalized_child:
                    child_nodes.append(normalized_child)
            if child_nodes:
                node["children"] = child_nodes
            return node

        for item in source_tasks[:max_tasks]:
            if next_id >= max_total_nodes:
                break
            normalized_item = _visit(item, depth=1)
            if normalized_item:
                normalized.append(normalized_item)
        return normalized

    @staticmethod
    def _normalize_contract_requirements(raw_requirements: Any, max_items: int = 10) -> list[dict[str, Any]]:
        """Normalize planner requirements into stable ids + titles."""
        normalized: list[dict[str, Any]] = []
        if isinstance(raw_requirements, list):
            for idx, item in enumerate(raw_requirements[:max_items], start=1):
                if isinstance(item, dict):
                    title = str(item.get("title", "")).strip()
                    req_id = str(item.get("id", "")).strip()
                else:
                    title = str(item).strip()
                    req_id = ""
                if not title:
                    continue
                if not req_id:
                    req_id = f"req_{idx}"
                req_id = re.sub(r"[^a-zA-Z0-9_]+", "_", req_id).strip("_") or f"req_{idx}"
                normalized.append({"id": req_id[:48], "title": title[:220]})
        return normalized

    @staticmethod
    def _default_task_contract(user_input: str) -> dict[str, Any]:
        """Fallback contract when planner output is unavailable."""
        cleaned = re.sub(r"\s+", " ", (user_input or "").strip())
        return {
            "summary": cleaned[:320],
            "tasks": [
                {"id": "task_1", "title": "Understand the request and constraints"},
                {"id": "task_2", "title": "Execute needed tools/actions"},
                {"id": "task_3", "title": "Produce final response aligned with request"},
            ],
            "requirements": [
                {"id": "req_user_request", "title": "Fully satisfy the user request before finalizing"},
            ],
            "prefetch_urls": [],
        }

    async def _generate_task_contract(
        self,
        user_input: str,
        recent_source_urls: list[str],
        require_all_sources: bool,
        turn_usage: dict[str, int],
        list_task_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Use planner prompt to generate a task contract for this turn."""
        source_lines = "\n".join(f"{idx}. {url}" for idx, url in enumerate(recent_source_urls, start=1))
        if not source_lines:
            source_lines = "(none)"
        list_member_lines = "(none)"
        if isinstance(list_task_plan, dict) and bool(list_task_plan.get("enabled", False)):
            members = list_task_plan.get("members")
            if isinstance(members, list) and members:
                list_member_lines = "\n".join(f"{idx}. {str(member)}" for idx, member in enumerate(members, start=1))
        base_messages = [
            Message(
                role="system",
                content=self.instructions.load("task_contract_planner_system_prompt.md"),
            ),
            Message(
                role="user",
                content=self.instructions.render(
                    "task_contract_planner_user_prompt.md",
                    user_input=user_input,
                    recent_source_urls=source_lines,
                    require_all_sources=str(bool(require_all_sources)).lower(),
                    extracted_list_members=list_member_lines,
                ),
            ),
        ]
        cfg_max_tokens = max(1, int(get_config().model.max_tokens))
        first_max_tokens = min(1200, cfg_max_tokens)
        retry_max_tokens = min(max(first_max_tokens * 2, 1800), cfg_max_tokens)
        attempts: list[tuple[str, int]] = [("task_contract_planner", first_max_tokens)]
        if retry_max_tokens > first_max_tokens:
            attempts.append(("task_contract_planner_retry", retry_max_tokens))

        payload: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt_idx, (interaction_label, planner_max_tokens) in enumerate(attempts, start=1):
            try:
                response = await self._complete_with_guards(
                    messages=base_messages,
                    tools=None,
                    interaction_label=interaction_label,
                    turn_usage=turn_usage,
                    max_tokens=planner_max_tokens,
                )
                payload = self._extract_json_object(response.content or "")
                if isinstance(payload, dict):
                    break

                usage = response.usage if isinstance(response.usage, dict) else {}
                completion_tokens = int(usage.get("completion_tokens", 0))
                at_cap = completion_tokens >= planner_max_tokens
                empty_output = not str(response.content or "").strip()
                should_retry = attempt_idx < len(attempts) and (empty_output or at_cap)
                if should_retry:
                    retry_reason = "empty_output" if empty_output else "hit_max_tokens"
                    self._emit_tool_output(
                        "task_contract",
                        {
                            "step": "planner_retry",
                            "attempt": attempt_idx,
                            "reason": retry_reason,
                            "completion_tokens": completion_tokens,
                            "max_tokens": planner_max_tokens,
                        },
                        (
                            "step=planner_retry\n"
                            f"attempt={attempt_idx}\n"
                            f"reason={retry_reason}\n"
                            f"completion_tokens={completion_tokens}\n"
                            f"max_tokens={planner_max_tokens}"
                        ),
                    )
                    continue
                payload = None
                break
            except Exception as e:
                last_error = e
                if attempt_idx < len(attempts):
                    self._emit_tool_output(
                        "task_contract",
                        {
                            "step": "planner_retry",
                            "attempt": attempt_idx,
                            "reason": "planner_error",
                        },
                        (
                            "step=planner_retry\n"
                            f"attempt={attempt_idx}\n"
                            "reason=planner_error\n"
                            f"error={str(e)}"
                        ),
                    )
                    continue
                payload = None
                break

        if payload is None and last_error is not None:
            self._emit_tool_output(
                "task_contract",
                {"step": "planner_error"},
                f"Planner contract generation failed: {str(last_error)}",
            )

        if not isinstance(payload, dict):
            contract = self._default_task_contract(user_input)
            self._emit_tool_output(
                "task_contract",
                {"step": "planner_fallback"},
                "Planner output was not valid JSON. Using fallback contract.",
            )
            return contract

        tasks = self._normalize_contract_tasks(payload.get("tasks"))
        requirements = self._normalize_contract_requirements(payload.get("requirements"))
        summary = str(payload.get("summary", "")).strip()[:320]
        if not tasks:
            tasks = self._default_task_contract(user_input)["tasks"]
        if not requirements:
            requirements = self._default_task_contract(user_input)["requirements"]
        prefetch_urls: list[str] = []
        raw_prefetch = payload.get("prefetch_urls")
        if isinstance(raw_prefetch, list):
            for url in raw_prefetch:
                if not isinstance(url, str):
                    continue
                clean = url.strip()
                if clean.startswith(("http://", "https://")):
                    prefetch_urls.append(clean)
        if require_all_sources and recent_source_urls:
            prefetch_urls = self._merge_unique_urls(recent_source_urls, prefetch_urls)
        else:
            prefetch_urls = self._merge_unique_urls(prefetch_urls, recent_source_urls)
        prefetch_urls = prefetch_urls[:20]

        if require_all_sources and recent_source_urls:
            requirement_urls: set[str] = set()
            for req in requirements:
                if not isinstance(req, dict):
                    continue
                title = str(req.get("title", ""))
                for url in self._extract_urls(title):
                    requirement_urls.add(url)
            missing_urls = [url for url in recent_source_urls if url not in requirement_urls]
            base_count = len(requirements)
            for offset, url in enumerate(missing_urls, start=1):
                req_id = f"req_source_{base_count + offset}"
                requirements.append({
                    "id": req_id,
                    "title": f"Cover source: {url}",
                })

        contract = {
            "summary": summary or self._default_task_contract(user_input)["summary"],
            "tasks": tasks,
            "requirements": requirements,
            "prefetch_urls": prefetch_urls,
        }
        task_nodes = sum(1 for _ in self._iter_pipeline_nodes(tasks))
        task_leaves = sum(1 for _ in self._iter_pipeline_leaves(tasks))
        self._emit_tool_output(
            "task_contract",
            {
                "step": "planner_done",
                "tasks": len(tasks),
                "task_nodes": task_nodes,
                "task_leaves": task_leaves,
                "requirements": len(requirements),
                "prefetch_urls": len(prefetch_urls),
                "require_all_sources": require_all_sources,
                "recent_sources": len(recent_source_urls),
            },
            (
                f"step=planner_done\n"
                f"tasks={len(tasks)}\n"
                f"task_nodes={task_nodes}\n"
                f"task_leaves={task_leaves}\n"
                f"requirements={len(requirements)}\n"
                f"prefetch_urls={len(prefetch_urls)}\n"
                f"require_all_sources={require_all_sources}\n"
                f"recent_sources={len(recent_source_urls)}"
            ),
        )
        return contract

    async def _evaluate_contract_completion(
        self,
        user_input: str,
        candidate_response: str,
        contract: dict[str, Any],
        turn_usage: dict[str, int],
    ) -> dict[str, Any]:
        """Critic pass: evaluate whether candidate satisfies contract requirements."""
        requirements = contract.get("requirements")
        if not isinstance(requirements, list) or not requirements:
            return {"complete": True, "checks": []}

        requirements_json = json.dumps(requirements, ensure_ascii=True)
        messages = [
            Message(
                role="system",
                content=self.instructions.load("task_contract_critic_system_prompt.md"),
            ),
            Message(
                role="user",
                content=self.instructions.render(
                    "task_contract_critic_user_prompt.md",
                    user_input=user_input,
                    requirements_json=requirements_json,
                    candidate_response=candidate_response,
                ),
            ),
        ]
        try:
            response = await self._complete_with_guards(
                messages=messages,
                tools=None,
                interaction_label="task_contract_critic",
                turn_usage=turn_usage,
                max_tokens=min(1200, int(get_config().model.max_tokens)),
            )
            payload = self._extract_json_object(response.content or "")
        except Exception as e:
            self._emit_tool_output(
                "completion_gate",
                {"step": "critic_error"},
                f"Contract critic failed: {str(e)}",
            )
            payload = None

        if not isinstance(payload, dict):
            return {
                "complete": True,
                "checks": [],
                "feedback": "",
                "error": "critic_non_json",
            }

        req_ids = {str(req.get("id", "")).strip() for req in requirements if isinstance(req, dict)}
        checks: list[dict[str, Any]] = []
        raw_checks = payload.get("checks")
        if isinstance(raw_checks, list):
            for entry in raw_checks:
                if not isinstance(entry, dict):
                    continue
                check_id = str(entry.get("id", "")).strip()
                if not check_id or check_id not in req_ids:
                    continue
                checks.append({
                    "id": check_id,
                    "ok": bool(entry.get("ok", False)),
                    "reason": str(entry.get("reason", "")).strip(),
                })

        for req_id in req_ids:
            if not any(item.get("id") == req_id for item in checks):
                checks.append({"id": req_id, "ok": False, "reason": "missing critic evaluation"})

        complete = bool(payload.get("complete", False))
        if checks and not all(bool(item.get("ok", False)) for item in checks):
            complete = False
        feedback = str(payload.get("feedback", "")).strip()
        return {
            "complete": complete,
            "checks": checks,
            "feedback": feedback,
        }

    def _build_completion_feedback(
        self,
        contract: dict[str, Any],
        critique: dict[str, Any],
    ) -> str:
        """Build retry feedback based on model critic output."""
        feedback = str(critique.get("feedback", "")).strip()
        checks = critique.get("checks")
        req_map = {
            str(req.get("id", "")).strip(): str(req.get("title", "")).strip()
            for req in (contract.get("requirements") or [])
            if isinstance(req, dict)
        }
        failed = []
        if isinstance(checks, list):
            failed = [entry for entry in checks if isinstance(entry, dict) and not bool(entry.get("ok", False))]
        if feedback and failed:
            lines = [feedback, "Missing requirements to fix:"]
            for item in failed:
                req_id = str(item.get("id", "")).strip()
                title = req_map.get(req_id, req_id) or req_id
                reason = str(item.get("reason", "")).strip() or "not satisfied"
                lines.append(f"- {title}: {reason}")
            lines.append("Return only the corrected final answer.")
            return "\n".join(lines)
        if feedback:
            return feedback + "\nReturn only the corrected final answer."
        if failed:
            lines = [
                "The previous draft is incomplete. Fix all missing requirements before finalizing.",
                "Missing requirements:",
            ]
            for item in failed:
                req_id = str(item.get("id", "")).strip()
                title = req_map.get(req_id, req_id) or req_id
                reason = str(item.get("reason", "")).strip() or "not satisfied"
                lines.append(f"- {title}: {reason}")
            lines.append("Return only the corrected final answer.")
            return "\n".join(lines)
        return "Re-check the task contract and return a complete final answer."

    @staticmethod
    def _is_explicit_script_request(user_input: str) -> bool:
        """Detect explicit user requests to generate/build a script."""
        text = (user_input or "").strip().lower()
        if not text:
            return False
        return bool(
            re.search(
                r"\b(generate|create|build|write|make)\b.{0,40}\bscript\b"
                r"|\bscript\b.{0,40}\b(generate|create|build|write|make)\b",
                text,
            )
        )

    @staticmethod
    def _is_list_processing_request(user_input: str) -> bool:
        """Detect requests that imply processing multiple items/entities."""
        text = (user_input or "").strip().lower()
        if not text:
            return False
        if re.search(r"\btop\s+\d+\b", text):
            return True
        list_markers = (
            r"\bfor each\b",
            r"\beach\b",
            r"\bevery\b",
            r"\ball\b",
            r"\bper\b",
            r"\blist\b",
            r"\bextract\b.{0,30}\bnames?\b",
            r"\bcompanies?\b",
            r"\bcities?\b",
            r"\bsources?\b",
        )
        return any(re.search(pattern, text) for pattern in list_markers)

    @staticmethod
    def _should_enforce_python_worker_mode(user_input: str) -> bool:
        """Whether this turn should enforce Python worker mode."""
        return AgentReasoningMixin._is_explicit_script_request(user_input)

    @staticmethod
    def _normalize_list_members(raw_members: Any, max_members: int = 40) -> list[str]:
        """Normalize extracted list members into a stable ordered unique list."""
        members: list[str] = []
        seen: set[str] = set()
        items: list[Any]
        if isinstance(raw_members, list):
            items = raw_members
        elif isinstance(raw_members, dict):
            items = [raw_members]
        else:
            items = []
        for item in items:
            if len(members) >= max_members:
                break
            if isinstance(item, dict):
                candidate = str(
                    item.get("name")
                    or item.get("member")
                    or item.get("item")
                    or item.get("title")
                    or ""
                ).strip()
            else:
                candidate = str(item or "").strip()
            if not candidate:
                continue
            candidate = re.sub(r"\s+", " ", candidate).strip(" -\t\r\n")
            if len(candidate) < 2:
                continue
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            members.append(candidate[:160])
        return members

    @staticmethod
    def _choose_list_execution_strategy(
        user_input: str,
        members_count: int,
        recommended: str = "",
    ) -> str:
        """Choose execution strategy for per-member work.

        Default policy is direct internal tool usage. Script mode is opt-in only.
        """
        rec = str(recommended or "").strip().lower()
        text = (user_input or "").strip().lower()
        if rec == "direct":
            return "direct"
        if AgentReasoningMixin._is_explicit_script_request(text):
            return "script"
        if rec == "script":
            return "direct"
        del members_count
        return "direct"

    def _collect_list_extraction_context(
        self,
        max_messages: int = 18,
        max_chars: int = 12000,
        per_message_chars: int = 1400,
    ) -> str:
        """Collect compact recent context to help list-member extraction."""
        if not self.session:
            return ""
        start = max(0, len(self.session.messages) - max_messages)
        lines: list[str] = []
        total_chars = 0
        for msg in self.session.messages[start:]:
            role = str(msg.get("role", "")).strip().lower() or "unknown"
            tool_name = str(msg.get("tool_name", "")).strip()
            if role == "tool" and self._is_monitor_only_tool_name(tool_name):
                continue
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if len(content) > per_message_chars:
                content = content[:per_message_chars] + "... [truncated]"
            prefix = f"[{role}]"
            if role == "tool" and tool_name:
                prefix = f"[{role}:{tool_name}]"
            line = f"{prefix} {content}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line) + 1
        return "\n".join(lines)

    @staticmethod
    def _list_member_aliases(member: str) -> set[str]:
        """Build simple aliases for matching member coverage in outputs."""
        base = str(member or "").strip().lower()
        if not base:
            return set()
        aliases = {base}
        normalized_words = re.sub(r"[^a-z0-9]+", " ", base).strip()
        if normalized_words:
            aliases.add(normalized_words)
            aliases.add(normalized_words.replace(" ", "-"))
            aliases.add(normalized_words.replace(" ", "_"))
            aliases.add(normalized_words.replace(" ", ""))
        return {alias for alias in aliases if len(alias) >= 2}

    def _evaluate_list_member_coverage(
        self,
        members: list[str],
        candidate_response: str,
        turn_start_idx: int,
    ) -> tuple[list[str], list[str]]:
        """Evaluate which list members are covered in this turn outputs."""
        if not members:
            return [], []
        text_parts: list[str] = [str(candidate_response or "")]
        if self.session:
            for msg in self.session.messages[turn_start_idx:]:
                role = str(msg.get("role", "")).strip().lower()
                if role not in {"assistant", "tool"}:
                    continue
                tool_name = str(msg.get("tool_name", "")).strip().lower()
                if role == "tool" and self._is_monitor_only_tool_name(tool_name):
                    continue
                content = str(msg.get("content", "")).strip()
                if content:
                    text_parts.append(content)
        haystack = "\n".join(text_parts).lower()

        showcase_root = (
            self.tools.get_saved_base_path(create=False)
            / "showcase"
            / self._current_session_slug()
        )
        showcase_names: list[str] = []
        if showcase_root.exists():
            try:
                showcase_names = [
                    path.name.lower()
                    for path in showcase_root.rglob("*")
                    if path.is_file()
                ]
            except Exception:
                showcase_names = []

        covered: list[str] = []
        missing: list[str] = []
        for member in members:
            aliases = self._list_member_aliases(member)
            in_text = any(alias in haystack for alias in aliases)
            in_artifacts = False
            if not in_text and showcase_names:
                in_artifacts = any(
                    any(alias in filename for alias in aliases)
                    for filename in showcase_names
                )
            if in_text or in_artifacts:
                covered.append(member)
            else:
                missing.append(member)
        return covered, missing

    @staticmethod
    def _build_list_coverage_feedback(
        missing_members: list[str],
        strategy: str,
        per_member_action: str,
    ) -> str:
        """Build retry guidance when not all extracted list members are covered."""
        if not missing_members:
            return ""
        preview = ", ".join(missing_members[:8])
        if len(missing_members) > 8:
            preview += f", ... (+{len(missing_members) - 8} more)"
        action_line = f"Requested per-member action: {per_member_action}" if per_member_action else ""
        if strategy == "script":
            return (
                "Completion gate: extracted list members are still missing.\n"
                f"Missing members: {preview}\n"
                f"{action_line}\n"
                "Regenerate or adjust the Python worker to process all missing members, execute it, "
                "and then provide final concise output."
            ).strip()
        return (
            "Completion gate: extracted list members are still missing.\n"
            f"Missing members: {preview}\n"
            f"{action_line}\n"
            "Continue in direct loop mode: process each missing member one-by-one using tools as needed, "
            "then return final concise output."
        ).strip()

    @staticmethod
    def _apply_list_requirements(
        base_requirements: list[dict[str, Any]],
        list_task_plan: dict[str, Any],
        max_members: int = 30,
    ) -> list[dict[str, Any]]:
        """Augment completion requirements with extracted list-member coverage checks."""
        requirements = [dict(item) for item in base_requirements if isinstance(item, dict)]
        if not isinstance(list_task_plan, dict) or not bool(list_task_plan.get("enabled", False)):
            return requirements
        members = list_task_plan.get("members")
        if not isinstance(members, list) or not members:
            return requirements
        action = str(list_task_plan.get("per_member_action", "")).strip()
        existing_ids = {str(req.get("id", "")).strip() for req in requirements}
        for idx, member in enumerate(members[:max_members], start=1):
            base_id = re.sub(r"[^a-zA-Z0-9_]+", "_", str(member).strip().lower()).strip("_")[:28]
            if not base_id:
                base_id = f"member_{idx}"
            req_id = f"req_member_{base_id}"
            if req_id in existing_ids:
                continue
            existing_ids.add(req_id)
            title = f"Cover list member: {member}"
            if action:
                title = f"{title} ({action})"
            requirements.append({"id": req_id, "title": title[:220]})
        return requirements

    async def _generate_list_task_plan(
        self,
        user_input: str,
        context_excerpt: str,
        turn_usage: dict[str, int],
    ) -> dict[str, Any]:
        """Extract list members from context and select direct vs script strategy."""
        fallback = {
            "enabled": False,
            "members": [],
            "strategy": "none",
            "per_member_action": "",
            "confidence": "low",
        }
        if not self._is_list_processing_request(user_input):
            return fallback

        messages = [
            Message(
                role="system",
                content=self.instructions.load("list_task_extractor_system_prompt.md"),
            ),
            Message(
                role="user",
                content=self.instructions.render(
                    "list_task_extractor_user_prompt.md",
                    user_input=user_input,
                    context_excerpt=context_excerpt or "(empty)",
                ),
            ),
        ]
        payload: dict[str, Any] | None = None
        try:
            response = await self._complete_with_guards(
                messages=messages,
                tools=None,
                interaction_label="list_task_extractor",
                turn_usage=turn_usage,
                max_tokens=min(1000, int(get_config().model.max_tokens)),
            )
            payload = self._extract_json_object(response.content or "")
        except Exception as e:
            self._emit_tool_output(
                "task_contract",
                {"step": "list_extract_error"},
                f"step=list_extract_error\nerror={str(e)}",
            )
            payload = None

        has_list_work = False
        members: list[str] = []
        per_member_action = ""
        recommended_strategy = ""
        confidence = "low"
        if isinstance(payload, dict):
            has_list_work = bool(payload.get("has_list_work", False))
            members = self._normalize_list_members(payload.get("members"))
            per_member_action = str(payload.get("per_member_action", "")).strip()[:220]
            recommended_strategy = str(payload.get("recommended_strategy", "")).strip().lower()
            confidence = str(payload.get("confidence", "low")).strip().lower()[:16] or "low"

        if not has_list_work and not members:
            return fallback
        strategy = self._choose_list_execution_strategy(
            user_input=user_input,
            members_count=len(members),
            recommended=recommended_strategy,
        )
        plan = {
            "enabled": bool(members),
            "members": members,
            "strategy": strategy,
            "per_member_action": per_member_action,
            "confidence": confidence,
        }
        preview = ", ".join(members[:8]) if members else "(none)"
        if len(members) > 8:
            preview += f", ... (+{len(members) - 8} more)"
        self._emit_tool_output(
            "task_contract",
            {
                "step": "list_extract_done",
                "enabled": bool(plan["enabled"]),
                "members": len(members),
                "strategy": strategy,
                "confidence": confidence,
            },
            (
                "step=list_extract_done\n"
                f"enabled={plan['enabled']}\n"
                f"members={len(members)}\n"
                f"strategy={strategy}\n"
                f"confidence={confidence}\n"
                f"members_preview={preview}"
            ),
        )
        return plan
