"""Reasoning/contract/list planning helpers for Agent."""

from datetime import UTC, datetime
import json
import os
import re
from typing import Any

from captain_claw.config import get_config
from captain_claw.llm import Message


# An agent spawned by one of the Flight Deck multi-agent modes stamps a marker in
# its environment. In those modes the orchestrator already frames each agent's
# task (role + Lead brief / router `why` + shared contract, at reason tier, once),
# so the agent's own generic task-rephrase is redundant overhead + a scope-drift
# risk — we skip it for these workers.
_FD_WORKER_MARKERS = (
    "CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER", "CLAW_COUNCIL_WORKER", "CLAW_CODE_AGENT",
    # An Iskra being's tick prompt is fully pre-framed by compose_tick_prompt
    # (vitals, drives, task, digest schema) — a per-turn self-rephrase would
    # rewrite that carefully-built prompt through a weaker pass and drift the
    # digest contract, and next-steps is wasted on a headless being. Skip both.
    "CLAW_BEING_WORKER",
)


def _is_fd_spawned_worker() -> bool:
    return any(
        str(os.environ.get(m, "")).strip().lower() in ("1", "true", "yes")
        for m in _FD_WORKER_MARKERS
    )


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
        """Heuristic: whether the assistant is asking the user to choose/clarify.

        Catches an explicit clarification phrase OR any short question (so terse
        asks like "Which file?" still count). The crucial guard against social
        false-positives ("How's your day going?") lives at the call site in
        ``_update_clarification_state``: a question only pins as a clarification
        when the message that PROMPTED it was a real task, not chit-chat. That
        lets us keep genuine terse clarifications without re-introducing the
        "pretty good!" → hallucinated-pipeline regression."""
        text = re.sub(r"\s+", " ", (response_text or "").strip())
        if not text or not text.endswith("?"):
            return False
        lowered = text.lower()
        prompts = (
            "which would you like",
            "do you want me to",
            "would you like me to",
            "should i proceed",
            "tell me your choices",
            "quick questions",
            "so i proceed correctly",
            "could you clarify",
            "can you clarify",
            "could you specify",
            "can you specify",
            "which option",
            "which one",
            "which of",
            "what would you like me to",
        )
        if any(phrase in lowered for phrase in prompts):
            return True
        # A short reply that ends in a question is plausibly a real ask; a long
        # answer that merely tails off into one is not. The task-vs-chit-chat
        # gate at the call site neutralizes social-question false positives.
        return len(text) <= 200

    # Greetings / acknowledgments / social one-liners — a message that is NONE
    # of a task request. Used to decide a question back is social, not a
    # clarification, and that a follow-up shouldn't force the task pipeline.
    # NOTE: bare affirmatives (yes/yeah/ok/sure) are deliberately EXCLUDED —
    # "yeah" answering "do you want me to proceed?" is a genuine clarification
    # answer, and treating it as filler would drop the context the user wants
    # kept. Only unambiguous chit-chat is matched.
    _FILLER_RE = re.compile(
        r"^(?:(?:"
        r"hi+|hey+|hello+|yo+|hiya|sup|heya|"
        r"good\s+(morning|afternoon|evening|night)|"
        r"how\s+(are|r)\s+(you|ya|u)|how'?s\s+(it\s+going|your\s+day|things|life|everything)|"
        r"what'?s\s+up|wassup|long\s+time|"
        r"thanks?|thank\s+you|thx|cheers|much\s+appreciated|appreciate\s+it|"
        r"cool|nice|awesome|perfect|sweet|excellent|"
        r"sounds?\s+good|got\s+it|gotcha|makes\s+sense|fair\s+enough|"
        r"lol+|haha+|hehe+|nice\s+one|"
        r"(?:i'?m|i\s+am)\s+(good|fine|great|ok|okay|well|alright|doing\s+\w+)|"
        r"pretty\s+(good|nice)|not\s+bad|all\s+good|doing\s+(good|well|fine|great)|"
        r"good|fine|great|same|likewise"
        r")[\s.!?,……🙂😊👍👋🎉]*){1,3}$",
        re.I,
    )

    @staticmethod
    def _is_conversational_filler(text: str) -> bool:
        """True when the message is pure chit-chat (greeting, ack, social) and
        carries no task request — so a question back to it is social, and a
        follow-up to it should never force the heavyweight task pipeline."""
        t = (text or "").strip()
        if not t:
            return True
        if not re.search(r"[A-Za-z0-9]", t):
            return True  # emoji/punctuation only
        if len(re.findall(r"\S+", t)) <= 6 and AgentReasoningMixin._FILLER_RE.match(t):
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

    @staticmethod
    def _looks_like_topic_switch(user_input: str, anchor: str) -> bool:
        """True when the new message looks like a fresh self-contained request
        rather than an answer to the pending clarification.

        Two signals must agree: (a) the message opens like a new command or
        question (an imperative verb or a wh-/"can you" opener), and (b) it
        shares almost no content words with the pending anchor. A genuine
        clarification answer is usually a short fragment ("the second one",
        "make it formal") that neither opens like a command nor introduces a
        wholly new vocabulary."""
        text = (user_input or "").strip().lower()
        if not text:
            return False
        _NEW_TASK_OPENERS = (
            "make ", "create ", "write ", "build ", "generate ", "draw ",
            "design ", "show ", "list ", "find ", "search ", "fetch ", "get ",
            "read ", "open ", "play ", "explain ", "summarize ", "summarise ",
            "translate ", "calculate ", "send ", "email ", "schedule ", "plan ",
            "fix ", "debug ", "add ", "install ", "run ", "give me ", "tell me ",
            "how ", "what ", "why ", "when ", "where ", "who ", "can you ",
            "could you ", "i want ", "i need ", "let's ", "lets ", "help me ",
        )
        if not text.startswith(_NEW_TASK_OPENERS):
            return False
        def _content_words(s: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]{4,}", (s or "").lower()))
        anchor_words = _content_words(anchor)
        new_words = _content_words(user_input)
        if not anchor_words or not new_words:
            # No anchor vocabulary to compare against — the fresh opener alone
            # is enough to treat it as a new task.
            return True
        return len(anchor_words & new_words) < 2

    @staticmethod
    def _user_requests_refetch(user_input: str) -> bool:
        """Detect whether the user's reply explicitly asks for fresh fetching."""
        text = (user_input or "").strip().lower()
        if not text:
            return False
        patterns = (
            r"\bfetch\s+(?:new|again|fresh|live|now)\b",
            r"\bre-?fetch\b",
            r"\brefresh\b",
            r"\bfetch\s+(?:it|the|that)\s+(?:again|now)\b",
            r"\bgo\s+(?:fetch|get)\b",
            r"\bget\s+(?:new|fresh|latest|live)\b",
            r"\bre-?(?:search|research|check|load|read)\b",
            r"\bdo\s+(?:a\s+)?(?:new|fresh|live)\s+(?:fetch|search|check)\b",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    def _resolve_effective_user_input(self, user_input: str) -> tuple[str, bool]:
        """Merge pending clarification anchor with current message when appropriate.

        When the assistant previously asked a clarification question and the user
        now replies with a short follow-up, we merge the context so the planner
        understands what the user wants.  Instead of re-posing the *original*
        question (which can trigger a full re-research pipeline), we attach the
        last assistant response as context — that response already contains the
        specific items (URLs, article titles, numbered options) the user is
        referring to.

        If the user explicitly requests a fresh fetch/research (e.g. "fetch new",
        "refresh", "re-search"), the merge still provides the context but does NOT
        forbid re-fetching so the user's intent is honoured.
        """
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
        # Pure chit-chat follow-up ("pretty good!", "no thanks", "cool") is NOT
        # an answer to a task clarification — don't merge stale context or let
        # it force the task pipeline. Clear the anchor and treat it plainly.
        if self._is_conversational_filler(user_input):
            state["pending"] = False
            state.pop("anchor_request", None)
            return user_input, False
        # Topic switch: if the new message reads as a fresh, self-contained
        # request rather than an answer to the pending question, do NOT merge
        # the stale context (which would both contaminate the new task and
        # force the heavyweight contract pipeline). Clear the pending anchor
        # and treat the message as a brand-new turn.
        if self._looks_like_topic_switch(user_input, anchor):
            state["pending"] = False
            state.pop("anchor_request", None)
            return user_input, False

        wants_refetch = self._user_requests_refetch(user_input)

        # Try to find the last assistant response — it contains the specific
        # items the user is referring to (URLs, articles, options offered).
        last_assistant_text = ""
        if self.session.messages:
            for msg in reversed(self.session.messages):
                if msg.get("role") == "assistant":
                    last_assistant_text = str(msg.get("content", "")).strip()
                    break

        if last_assistant_text and len(last_assistant_text) > 50:
            # Use the assistant's previous response as context instead of the
            # broad original question.  This keeps the planner focused on the
            # specific items already surfaced rather than re-researching.
            context_excerpt = last_assistant_text[-2000:]
            if wants_refetch:
                # User explicitly asked for a fresh fetch — provide the context
                # for reference but do NOT forbid re-fetching.
                merged = (
                    f"{user_input.strip()}\n\n"
                    "Context from the previous assistant response:\n"
                    f"{context_excerpt}\n\n"
                    "The user explicitly requested a fresh fetch/research. "
                    "Perform a live fetch or search as they asked — do NOT "
                    "reuse cached or previously saved content."
                )
            else:
                merged = (
                    f"{user_input.strip()}\n\n"
                    "Context from the previous assistant response:\n"
                    f"{context_excerpt}\n\n"
                    "Execute the user's follow-up request using the context above. "
                    "Do NOT re-research the original question — the context already "
                    "contains the needed information."
                )
        else:
            # Fallback: no usable assistant response, use the original anchor.
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
        # Only treat a question-back as a real clarification when the message
        # that prompted it was an actual TASK request — not chit-chat. This is
        # what stops "hi!" → "How's your day going?" from pinning (and then
        # hijacking the next casual reply into the task pipeline), while keeping
        # genuine clarifications after a real request ("edit the config" →
        # "Which file?") working.
        if (
            self._assistant_requests_clarification(assistant_response)
            and not self._is_conversational_filler(user_input)
        ):
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

    # Regex for requests that are simple single-action tasks (list, search,
    # read, check, show).  These don't benefit from the contract pipeline's
    # multi-step planning and validation overhead.
    _SIMPLE_TASK_RE = re.compile(
        r"^(?:list|show|search|find|check|get|read|display|what(?:'s| is| are))"
        r"\b",
        re.IGNORECASE,
    )

    # Trivial single-value edits: "change X to Y", "replace A with B",
    # "set X to Y", "on slide 2, change $2.2M to $4.5M". One substitution =
    # one or two tool calls; the contract → critic → completion-gate pipeline
    # adds ~10s of LLM round-trips for nothing. A verb followed (within a short
    # window) by a to/with/from/into connector — searched anywhere so a leading
    # locator like "on slide 2," still matches.
    _TRIVIAL_EDIT_RE = re.compile(
        r"\b(?:change|replace|rename|swap|set|update)\b[^\n]{0,40}\b(?:to|with|from|into)\b",
        re.IGNORECASE,
    )
    # Words that signal a multi-part request — never treat such a turn as
    # trivial even if it contains an edit verb. (A comma alone is fine: "on
    # slide 2, change X to Y" is still a single edit.)
    _COMPOUND_RE = re.compile(
        r"\b(?:and|then|also|plus|while|each|every|both)\b", re.IGNORECASE
    )

    @staticmethod
    def _is_simple_request(user_input: str) -> bool:
        """Single-action turns that don't justify the contract pipeline:
        read-only lookups (list/show/search…) and trivial inline edits
        (change X to Y). Must be short and free of compound conjunctions."""
        text = (user_input or "").strip()
        if not text or len(text) >= 120:
            return False
        if AgentReasoningMixin._SIMPLE_TASK_RE.search(text):
            return True
        if (
            AgentReasoningMixin._TRIVIAL_EDIT_RE.search(text)
            and not AgentReasoningMixin._COMPOUND_RE.search(text)
        ):
            return True
        return False

    @staticmethod
    def _should_use_contract_pipeline(
        user_input: str,
        planning_enabled: bool,
        pipeline_mode: str | None = None,
    ) -> bool:
        """Use explicit user-selected mode only (no automatic switching).

        Simple single-action requests (list, search, read, trivial edits)
        bypass the contract pipeline entirely — the overhead of generating a
        contract, running a critic, and retrying validation is not justified
        for tasks that translate to one or two tool calls.
        """
        mode = str(pipeline_mode or "").strip().lower()
        if mode == "contracts":
            # Explicit contracts mode — but still skip for trivially simple tasks.
            if AgentReasoningMixin._is_simple_request(user_input):
                return False
            return True
        if mode == "loop":
            return bool(planning_enabled)

        # Default / fallback — skip for simple tasks.
        if AgentReasoningMixin._is_simple_request(user_input):
            return False
        return bool(planning_enabled)

    @staticmethod
    def _normalize_contract_tasks(
        raw_tasks: Any,
        max_tasks: int = 8,
        max_depth: int = 12,
        max_total_nodes: int = 36,
    ) -> list[dict[str, Any]]:
        """Normalize planner task items into a validated DAG-capable task tree."""
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

        def _extract_depends_on(item: Any) -> list[str]:
            if not isinstance(item, dict):
                return []
            raw_depends = item.get("depends_on")
            if raw_depends is None:
                raw_depends = item.get("depends")
            if raw_depends is None:
                raw_depends = item.get("after")
            if not isinstance(raw_depends, list):
                return []
            deps: list[str] = []
            for dep in raw_depends:
                dep_id = re.sub(r"[^a-zA-Z0-9_]+", "_", str(dep or "").strip()).strip("_")
                if dep_id and dep_id not in deps:
                    deps.append(dep_id[:64])
            return deps

        def _extract_max_retries(item: Any) -> int:
            if not isinstance(item, dict):
                return 2
            for key in ("max_retries", "retries", "retry_limit"):
                value = item.get(key)
                if value is None:
                    continue
                try:
                    return max(0, min(8, int(value)))
                except Exception:
                    continue
            return 2

        def _extract_timeout_seconds(item: Any) -> float | None:
            if not isinstance(item, dict):
                return 180.0
            for key in ("timeout_seconds", "timeout_sec", "timeout"):
                value = item.get(key)
                if value in {None, ""}:
                    continue
                try:
                    parsed = float(value)
                except Exception:
                    continue
                if parsed <= 0:
                    return None
                return min(parsed, 3600.0)
            return 180.0

        def _visit(item: Any, depth: int) -> dict[str, Any] | None:
            nonlocal next_id
            if next_id >= max_total_nodes:
                return None
            title = _extract_title(item)
            if not title:
                return None
            next_id += 1
            raw_id = ""
            if isinstance(item, dict):
                raw_id = re.sub(r"[^a-zA-Z0-9_]+", "_", str(item.get("id", "")).strip()).strip("_")
            node: dict[str, Any] = {
                "id": raw_id or f"task_{next_id}",
                "title": title[:220],
                "depends_on": _extract_depends_on(item),
                "retries": 0,
                "max_retries": _extract_max_retries(item),
                "timeout_seconds": _extract_timeout_seconds(item),
            }
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
        # Retrieve relevant playbook patterns for planner context.
        playbook_block = ""
        if hasattr(self, "_build_playbook_block"):
            try:
                playbook_block = await self._build_playbook_block(user_input)
            except Exception as _pb_err:
                log.debug("Playbook retrieval skipped", error=str(_pb_err))
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
                    playbook_block=playbook_block,
                ),
            ),
        ]
        cfg_max_tokens = max(1, int(get_config().model.max_tokens))
        first_max_tokens = min(3200, cfg_max_tokens)
        retry_max_tokens = min(max(first_max_tokens * 2, 6400), cfg_max_tokens)
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
        scale_completed: bool = False,
    ) -> dict[str, Any]:
        """Critic pass: evaluate whether candidate satisfies contract requirements."""
        requirements = contract.get("requirements")
        if not isinstance(requirements, list) or not requirements:
            return {"complete": True, "checks": []}

        requirements_json = json.dumps(requirements, ensure_ascii=True)
        user_content = self.instructions.render(
            "task_contract_critic_user_prompt.md",
            user_input=user_input,
            requirements_json=requirements_json,
            candidate_response=candidate_response,
        )
        if scale_completed:
            user_content += (
                "\n\nIMPORTANT: A scale micro-loop has already researched/processed "
                "all list items independently. The candidate response is a POST-PROCESSING "
                "synthesis that COMBINES pre-researched results. "
                "Do NOT require sequential or one-at-a-time delivery — a single "
                "combined response covering all items is correct and expected. "
                "Focus only on whether the content for each requirement is present."
            )
        messages = [
            Message(
                role="system",
                content=self.instructions.load("task_contract_critic_system_prompt.md"),
            ),
            Message(
                role="user",
                content=user_content,
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

    # Nouns that turn a bare quantifier ("all/each/every/per") into a real
    # list of items to process. Without one of these, "all"/"each"/"per" is
    # almost always ordinary prose ("all files need to be written", "as per
    # the spec", "each game has its own logic") — NOT list-extraction work.
    _LIST_NOUN = (
        r"(?:files?|urls?|links?|pages?|articles?|items?|entries|records?|rows?|"
        r"documents?|docs?|pdfs?|sources?|companies|cities|names?|sites?|"
        r"products?|images?|photos?|videos?|messages?|emails?|tickets?|"
        r"results?|sections?|entities|people|users?|customers?|accounts?)"
    )

    @staticmethod
    def _is_creation_request(user_input: str) -> bool:
        """Detect tasks that CREATE or modify artifacts (code, games, apps,
        components) rather than consume a list of existing inputs.

        These must never enter the input-extraction scale loop: their "items"
        are OUTPUTS to produce, not sources to read — feeding deliverable
        filenames into the micro-loop just burns tokens to processed=0 and
        then re-fires. Returns False for list-PRODUCING tasks ("create a
        CSV/list/table"), which the scale system legitimately handles.
        """
        text = (user_input or "").strip().lower()
        if not text:
            return False
        # List-PRODUCING tasks stay in the scale system — not artifact creation.
        if re.search(
            r"\b(?:create|compile|build|make|generate|produce|prepare|populate)\s+"
            r"(?:an?\s+|the\s+)?(?:\w+\s+){0,2}(?:list|csv|spreadsheet|table|report|index|dataset)\b",
            text,
        ):
            return False
        artifact = (
            r"(?:game|app|application|web\s*app|website|web\s*page|webpage|component|"
            r"feature|function|class|method|module|script|program|launcher|engine|"
            r"library|api|endpoint|server|backend|frontend|plugin|extension|ui|"
            r"interface|page|screen|form|dashboard|widget|suite|bot|cli|tool|"
            r"implementation|codebase)s?\b"
        )
        # Strong code-creation/modification verbs — creation regardless of object.
        if re.search(r"\b(?:implement|develop|scaffold|refactor|rewrite|reimplement|program|debug)\b", text):
            return True
        # Other create/modify verbs only count as creation with a code/artifact object.
        if re.search(
            r"\b(?:build|create|make|write|code|design|add|fix|test|update|modify|enhance|improve|port|wire)\b"
            r"[\s\S]{0,40}?" + artifact,
            text,
        ):
            return True
        return False

    @staticmethod
    def _is_list_processing_request(user_input: str) -> bool:
        """Detect requests that imply processing multiple existing items/entities.

        Covers three categories:
        1. Per-item processing language (for each FILE/URL/ITEM, ...)
        2. User explicitly providing a list (here are the urls, these files, ...)
        3. Task that will produce a list (create a list, compile a list, ...)

        A bare quantifier ("all/each/every/per") no longer fires on its own —
        it must be paired with a list noun — and artifact-CREATION tasks are
        excluded outright (their items are outputs, not inputs).
        """
        text = (user_input or "").strip().lower()
        if not text:
            return False
        # Creation/modification of code or artifacts is never input-list work.
        if AgentReasoningMixin._is_creation_request(text):
            return False
        if re.search(r"\btop\s+\d+\b", text):
            return True
        _noun = AgentReasoningMixin._LIST_NOUN
        list_markers = (
            # Per-item processing — quantifier MUST be followed by a list noun.
            r"\bfor\s+each\s+(?:\w+\s+){0,2}" + _noun,
            r"\b(?:each|every|all|per)\s+(?:\w+\s+){0,3}" + _noun,
            r"\b(?:loop|iterate)\s+(?:over|through)\b",
            r"\blist\s+(?:all|of|out|every|each|the|them|these|those)\s+(?:\w+\s+){0,3}" + _noun,
            r"\bextract\b.{0,30}\bnames?\b",
            # User explicitly providing items
            r"\bhere (?:is|are) the\s+(?:list|" + _noun + r")",
            r"\bthese\s+(?:urls?|links?|files?|pages?|items?|articles?|documents?)\b",
            r"\bthe following\s+(?:urls?|links?|files?|pages?|items?|articles?|list)\b",
            r"\bfrom (?:these|the following)\s+" + _noun,
            r"\bbelow (?:is|are)\s+(?:a\s+)?(?:list|" + _noun + r")",
            # Task producing a list
            r"\b(?:create|compile|build|make|generate|produce|prepare)\s+(?:a\s+)?(?:list|csv|spreadsheet|table|report)\b",
            r"\b(?:list|csv|spreadsheet|table)\s+(?:of|with|containing)\b",
            r"\bpopulate\s+(?:a\s+)?(?:csv|spreadsheet|table)\b",
        )
        return any(re.search(pattern, text) for pattern in list_markers)

    @staticmethod
    def _should_rephrase_task(user_input: str) -> bool:
        """Decide whether a user prompt would benefit from automatic rephrasing.

        Rephrasing helps when the user provides complex, free-form instructions
        that mix list items, formatting details, and output specifications in an
        unstructured way.  The rephraser converts these into a clean, structured
        prompt that downstream components (list extractor, planner, micro-loop)
        can parse more reliably.

        Returns True when the prompt has enough complexity to justify a
        rephrasing LLM call.  Short or already-structured prompts are skipped.
        """
        text = (user_input or "").strip()
        if not text:
            return False

        cfg = get_config().scale
        if not cfg.task_rephrase_enabled:
            return False

        # Flight Deck workers (Basna/Vatra/Council/Code) get their task pre-framed
        # by the orchestrator, so a per-agent self-rephrase just re-does that job —
        # worse (weaker model), N times over, and risks drifting from the Lead's
        # scoped brief. Skip it for these; standalone agents still rephrase.
        if _is_fd_spawned_worker():
            return False

        # Too short — rephrasing adds latency for no benefit.
        if len(text) < cfg.task_rephrase_min_chars:
            return False

        # Skip slash commands and very short follow-ups.
        if text.startswith("/"):
            return False

        lowered = text.lower()

        # Complexity signals — count how many apply.
        signals = 0

        # 1. Contains inline URLs (data sources to process)
        url_count = len(re.findall(r"https?://[^\s)\]}>\"']+", text))
        if url_count >= 2:
            signals += 2  # strong signal
        elif url_count >= 1:
            signals += 1

        # 2. List-processing language
        if AgentReasoningMixin._is_list_processing_request(text):
            signals += 1

        # 3. Output format specifications (CSV, columns, fields, format)
        format_patterns = (
            r"\bcsv\b",
            r"\bcolumns?\b",
            r"\bfields?\b",
            r"\bformat\b",
            r"\bheader\b",
            r"\bspreadsheet\b",
            r"\btable\b",
            r"\btemplate\b",
        )
        format_hits = sum(1 for p in format_patterns if re.search(p, lowered))
        if format_hits >= 2:
            signals += 2
        elif format_hits >= 1:
            signals += 1

        # 4. Multiple lines / paragraphs (complex multi-part instruction)
        line_count = text.count("\n")
        if line_count >= 8:
            signals += 2
        elif line_count >= 4:
            signals += 1

        # 5. Numbered / bulleted list of items
        list_lines = re.findall(r"^\s*(?:\d+[\.\)]\s+|[-*•]\s+)", text, re.MULTILINE)
        if len(list_lines) >= 3:
            signals += 1

        # 6. File naming instructions
        if re.search(r"\bname\s+(?:the\s+)?(?:output|file|result)", lowered):
            signals += 1
        if re.search(r"\.\w{2,4}\b", text) and re.search(r"\bfile\b|\boutput\b|\bsave\b|\bwrite\b", lowered):
            signals += 1

        # Threshold: need at least 3 complexity signals to justify rephrasing.
        return signals >= 3

    async def _rephrase_task(
        self,
        user_input: str,
        turn_usage: dict[str, int],
    ) -> tuple[str, bool]:
        """Rephrase a complex user prompt into a structured, agent-friendly format.

        Returns (rephrased_text, was_rephrased).
        If rephrasing fails or is not needed, returns (user_input, False).
        """
        # Externally-orchestrated code agents get precise prompts written by
        # the code orchestrator — an LLM rewrite only distorts them (SENKO2:
        # a 5-fix prompt was rephrased 1968→2271 chars before misfiring the
        # list pipeline).  Never rephrase.
        if self._scale_system_disabled():
            return user_input, False
        if not self._should_rephrase_task(user_input):
            return user_input, False

        self._emit_tool_output(
            "task_contract",
            {"step": "task_rephrase_start"},
            "step=task_rephrase_start\nnote=rephrasing user prompt for better agent execution",
        )

        messages = [
            Message(
                role="system",
                content=self.instructions.load("task_rephrase_system_prompt.md"),
            ),
            Message(
                role="user",
                content=self.instructions.render(
                    "task_rephrase_user_prompt.md",
                    user_input=user_input,
                ),
            ),
        ]
        try:
            response = await self._complete_with_guards(
                messages=messages,
                tools=None,
                interaction_label="task_rephrase",
                turn_usage=turn_usage,
                max_tokens=min(4000, int(get_config().model.max_tokens)),
            )
            rephrased = (response.content or "").strip()
        except Exception as e:
            self._emit_tool_output(
                "task_contract",
                {"step": "task_rephrase_error"},
                f"step=task_rephrase_error\nerror={str(e)}",
            )
            return user_input, False

        # Sanity checks: rephrased must be non-empty and not drastically
        # shorter than the original (which would indicate a bad rephrase).
        if not rephrased or len(rephrased) < len(user_input) * 0.3:
            self._emit_tool_output(
                "task_contract",
                {"step": "task_rephrase_rejected", "reason": "too_short"},
                (
                    "step=task_rephrase_rejected\n"
                    f"reason=too_short\n"
                    f"original_len={len(user_input)}\n"
                    f"rephrased_len={len(rephrased)}"
                ),
            )
            return user_input, False

        # Strip any accidental code fences wrapping the rephrased output.
        if rephrased.startswith("```") and rephrased.endswith("```"):
            rephrased = re.sub(r"^```\w*\n?", "", rephrased)
            rephrased = re.sub(r"\n?```$", "", rephrased).strip()

        self._emit_tool_output(
            "task_contract",
            {
                "step": "task_rephrase_done",
                "original_len": len(user_input),
                "rephrased_len": len(rephrased),
            },
            (
                "step=task_rephrase_done\n"
                f"original_len={len(user_input)}\n"
                f"rephrased_len={len(rephrased)}"
            ),
        )
        # Emit the full rephrased content as a dedicated tool output so the
        # web UI can display it in a visible panel for the user.
        self._emit_tool_output(
            "task_rephrase",
            {"original_len": len(user_input), "rephrased_len": len(rephrased)},
            rephrased,
        )
        return rephrased, True

    async def _rephrase_for_script_mode(
        self,
        user_input: str,
        turn_usage: dict[str, int],
    ) -> tuple[str, bool]:
        """Rephrase user prompt into a script-oriented specification.

        Always runs (no complexity check) — called when force-script mode
        is ON.  Uses a dedicated system prompt that restructures the user's
        request as a script spec.

        Returns (rephrased_text, was_rephrased).
        """
        self._emit_tool_output(
            "task_contract",
            {"step": "script_rephrase_start"},
            "step=script_rephrase_start\nnote=rephrasing for script-mode execution",
        )

        messages = [
            Message(
                role="system",
                content=self.instructions.load("script_rephrase_system_prompt.md"),
            ),
            Message(
                role="user",
                content=self.instructions.render(
                    "task_rephrase_user_prompt.md",
                    user_input=user_input,
                ),
            ),
        ]
        try:
            response = await self._complete_with_guards(
                messages=messages,
                tools=None,
                interaction_label="script_rephrase",
                turn_usage=turn_usage,
                max_tokens=min(4000, int(get_config().model.max_tokens)),
            )
            rephrased = (response.content or "").strip()
        except Exception as e:
            self._emit_tool_output(
                "task_contract",
                {"step": "script_rephrase_error"},
                f"step=script_rephrase_error\nerror={str(e)}",
            )
            return user_input, False

        if not rephrased or len(rephrased) < len(user_input) * 0.3:
            self._emit_tool_output(
                "task_contract",
                {"step": "script_rephrase_rejected", "reason": "too_short"},
                (
                    "step=script_rephrase_rejected\n"
                    f"reason=too_short\n"
                    f"original_len={len(user_input)}\n"
                    f"rephrased_len={len(rephrased)}"
                ),
            )
            return user_input, False

        # Strip accidental code fences.
        if rephrased.startswith("```") and rephrased.endswith("```"):
            rephrased = re.sub(r"^```\w*\n?", "", rephrased)
            rephrased = re.sub(r"\n?```$", "", rephrased).strip()

        self._emit_tool_output(
            "task_contract",
            {
                "step": "script_rephrase_done",
                "original_len": len(user_input),
                "rephrased_len": len(rephrased),
            },
            (
                "step=script_rephrase_done\n"
                f"original_len={len(user_input)}\n"
                f"rephrased_len={len(rephrased)}"
            ),
        )
        self._emit_tool_output(
            "task_rephrase",
            {"original_len": len(user_input), "rephrased_len": len(rephrased)},
            rephrased,
        )
        return rephrased, True

    @staticmethod
    def _should_enforce_python_worker_mode(user_input: str) -> bool:
        """Whether this turn should enforce Python worker mode."""
        return AgentReasoningMixin._is_explicit_script_request(user_input)

    @staticmethod
    def _normalize_list_members(raw_members: Any, max_members: int = 150) -> list[str]:
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
            # URLs can be very long (200+ chars); truncating them causes 404s.
            # Use a generous limit when a URL is present (standalone or embedded
            # in a label like "Title — https://example.com/very-long-slug").
            _has_url = "http://" in candidate or "https://" in candidate
            max_len = 512 if _has_url else 160
            members.append(candidate[:max_len])
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
        """Build simple aliases for matching member coverage in outputs.

        Handles several member formats:
        - Plain text: ``"Bird Buddy"``
        - Entity + URL: ``"Bird Buddy — https://mybirdbuddy.com/"``
        - Pure URL: ``"https://example.com"``

        For "Name — URL" members the entity name is extracted and used
        as the primary matching alias so that a response mentioning just
        "Bird Buddy" satisfies coverage for the full member string.
        """
        base = str(member or "").strip().lower()
        if not base:
            return set()

        aliases: set[str] = set()

        # ── Extract entity name from "Name — URL" format ──
        # If the member contains an embedded URL, the text before the URL
        # separator is the meaningful entity name to match against.
        url_match = re.search(r"https?://", base)
        if url_match:
            name_part = base[:url_match.start()].strip().rstrip("—–\u2014-:.,;| ").strip()
            if name_part:
                # Entity name is the primary match target.
                aliases.add(name_part)
                name_normalized = re.sub(r"[^a-z0-9]+", " ", name_part).strip()
                if name_normalized:
                    aliases.add(name_normalized)
                    aliases.add(name_normalized.replace(" ", "-"))
                    aliases.add(name_normalized.replace(" ", "_"))
                    aliases.add(name_normalized.replace(" ", ""))
            # Also add the domain name as a fallback alias (e.g. "mybirdbuddy").
            domain_match = re.search(r"https?://(?:www\.)?([^/.:]+)", base)
            if domain_match:
                aliases.add(domain_match.group(1))

        # ── Extract entity name from "Name — description" format ──
        # Members often have a short entity name followed by a dash
        # separator and a description (e.g. "Accenture — exposed internal
        # security tools").  The entity name is the key matching target;
        # the description may differ from the actual output wording.
        if not url_match:
            for sep in (" — ", " – ", " \u2014 ", " - "):
                if sep in base:
                    name_part = base.split(sep, 1)[0].strip()
                    if name_part and len(name_part) >= 2:
                        aliases.add(name_part)
                        name_normalized = re.sub(r"[^a-z0-9]+", " ", name_part).strip()
                        if name_normalized and len(name_normalized) >= 2:
                            aliases.add(name_normalized)
                            aliases.add(name_normalized.replace(" ", "-"))
                            aliases.add(name_normalized.replace(" ", "_"))
                            aliases.add(name_normalized.replace(" ", ""))
                    break

        # ── Full-string aliases (backward compatibility) ──
        aliases.add(base)
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
        max_individual_members: int = 20,
    ) -> list[dict[str, Any]]:
        """Augment completion requirements with extracted list-member coverage checks.

        For small/moderate lists (≤ *max_individual_members*), one requirement
        per member is added so the critic can verify each.  For large lists,
        a single aggregate requirement is added instead — the per-member
        coverage is still tracked by ``_evaluate_list_member_coverage`` at
        the completion gate, which is more reliable than the critic for
        100-item lists.
        """
        requirements = [dict(item) for item in base_requirements if isinstance(item, dict)]
        if not isinstance(list_task_plan, dict) or not bool(list_task_plan.get("enabled", False)):
            return requirements
        members = list_task_plan.get("members")
        if not isinstance(members, list) or not members:
            return requirements
        action = str(list_task_plan.get("per_member_action", "")).strip()
        existing_ids = {str(req.get("id", "")).strip() for req in requirements}

        if len(members) > max_individual_members:
            # Large list: single aggregate requirement to keep the critic
            # prompt manageable.  The scale-progress system and list-member
            # coverage gate handle individual tracking.
            agg_id = "req_all_list_members"
            if agg_id not in existing_ids:
                title = f"All {len(members)} list members must be processed"
                if action:
                    title += f" ({action})"
                requirements.append({"id": agg_id, "title": title[:220]})
        else:
            for idx, member in enumerate(members[:max_individual_members], start=1):
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
        max_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        """Extract list members from context and select direct vs script strategy.

        Args:
            max_tokens_override: Optional override for the LLM max_tokens.
                Useful for deferred re-extraction where the context is larger
                and more members may need to be returned.
        """
        fallback = {
            "enabled": False,
            "members": [],
            "strategy": "none",
            "per_member_action": "",
            "confidence": "low",
            "output_strategy": "single_file",
            "output_filename_template": "",
            "final_action": "reply",
        }
        # Master switch: code agents / latched-off workers never extract
        # list members — a numbered fix prompt is instructions, not a list.
        if self._scale_system_disabled():
            return fallback
        if not self._is_list_processing_request(user_input):
            return fallback

        _default_max_tokens = min(3000, int(get_config().model.max_tokens))
        _effective_max_tokens = max_tokens_override if max_tokens_override else _default_max_tokens

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
                max_tokens=_effective_max_tokens,
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
        member_context: dict[str, str] = {}
        per_member_action = ""
        recommended_strategy = ""
        confidence = "low"
        output_strategy = "single_file"
        output_filename_template = ""
        final_action = "reply"
        processing_mode = "summarize"
        if isinstance(payload, dict):
            has_list_work = bool(payload.get("has_list_work", False))
            members = self._normalize_list_members(payload.get("members"))
            per_member_action = str(payload.get("per_member_action", "")).strip()[:220]
            recommended_strategy = str(payload.get("recommended_strategy", "")).strip().lower()
            confidence = str(payload.get("confidence", "low")).strip().lower()[:16] or "low"
            output_strategy = str(payload.get("output_strategy", "single_file")).strip().lower()
            if output_strategy not in ("file_per_item", "single_file", "no_file"):
                output_strategy = "single_file"
            output_filename_template = str(payload.get("output_filename_template", "")).strip()[:200]
            final_action = str(payload.get("final_action", "reply")).strip().lower()
            if final_action not in ("write_file", "reply", "email", "api_call"):
                final_action = "reply"
            processing_mode = str(payload.get("processing_mode", "summarize")).strip().lower()
            if processing_mode not in ("raw", "summarize"):
                processing_mode = "summarize"
            # member_context: optional dict mapping member → brief context from
            # the source article (country, description, etc.)
            _raw_ctx = payload.get("member_context")
            if isinstance(_raw_ctx, dict):
                member_context = {
                    str(k): str(v)[:200] for k, v in _raw_ctx.items() if v
                }

        if not has_list_work and not members:
            return fallback
        strategy = self._choose_list_execution_strategy(
            user_input=user_input,
            members_count=len(members),
            recommended=recommended_strategy,
        )
        # Enable the plan when the extractor found list work OR concrete
        # members.  Output-list tasks ("create a list of files") have
        # has_list_work=True but members=[] because the items must be
        # discovered at runtime (e.g. via glob).  We still need
        # enabled=True so the scale advisory and progress systems know
        # this is a list task.
        plan = {
            "enabled": bool(has_list_work or members),
            "members": members,
            "member_context": member_context,
            "strategy": strategy,
            "per_member_action": per_member_action,
            "confidence": confidence,
            "output_strategy": output_strategy,
            "output_filename_template": output_filename_template,
            "final_action": final_action,
            "processing_mode": processing_mode,
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
                "output_strategy": output_strategy,
                "output_filename_template": output_filename_template,
                "final_action": final_action,
                "processing_mode": processing_mode,
            },
            (
                "step=list_extract_done\n"
                f"enabled={plan['enabled']}\n"
                f"members={len(members)}\n"
                f"strategy={strategy}\n"
                f"confidence={confidence}\n"
                f"output_strategy={output_strategy}\n"
                f"output_filename_template={output_filename_template}\n"
                f"final_action={final_action}\n"
                f"processing_mode={processing_mode}\n"
                f"members_preview={preview}"
            ),
        )
        return plan
