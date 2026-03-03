"""Playbook retrieval, distillation, and context injection for Agent.

This mixin handles:
- Retrieving relevant playbooks from the persistent store
- Formatting playbook blocks for injection into planner / scale / context
- Building playbook context notes for message assembly
- Distilling session traces into playbook proposals
- Session rating flow

Playbooks are human-reviewed do/don't patterns that improve orchestration
decisions by providing concrete pseudo-code examples of what works (and
what doesn't) for recurring task types.
"""

import asyncio
import json
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.session import PlaybookEntry, get_session_manager

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Task-type classification heuristics (lightweight, regex-free)
# ---------------------------------------------------------------------------
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "batch-processing": [
        "all files", "every file", "each file", "for each", "process all",
        "go through", "batch", "bulk", "folder", "recursive",
    ],
    "web-research": [
        "research", "articles", "search the web", "web_search", "web_fetch",
        "summarize article", "news", "find information",
    ],
    "code-generation": [
        "write code", "create script", "implement", "generate code",
        "function that", "class that", "build a", "refactor",
    ],
    "document-processing": [
        "pdf", "docx", "pptx", "xlsx", "extract from", "convert document",
        "parse document", "read document",
    ],
    "data-transformation": [
        "csv", "json", "transform", "convert format", "parse data",
        "spreadsheet", "table", "mapping",
    ],
    "orchestration": [
        "orchestrate", "pipeline", "workflow", "multi-step", "coordinate",
        "parallel tasks",
    ],
    "file-management": [
        "rename files", "move files", "organize", "copy files", "delete files",
        "file structure", "directory",
    ],
}


def classify_task_type(user_input: str) -> str | None:
    """Return the most likely task type for *user_input*, or ``None``."""
    text = user_input.lower()
    scores: dict[str, int] = {}
    for task_type, keywords in _TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score:
            scores[task_type] = score
    if not scores:
        return None
    return max(scores, key=scores.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_playbook_block(entries: list[PlaybookEntry]) -> str:
    """Render a concise playbook context block for prompt injection.

    Returns an empty string when *entries* is empty so callers can safely
    concatenate or pass as a template variable.
    """
    if not entries:
        return ""

    parts: list[str] = [
        "\n--- PLAYBOOK (from previous sessions) ---"
    ]
    for entry in entries:
        parts.append(f"\nPattern: {entry.name} [{entry.task_type}]")
        if entry.trigger_description:
            parts.append(f"When: {entry.trigger_description}")
        if entry.do_pattern:
            parts.append(f"DO:\n{entry.do_pattern}")
        if entry.dont_pattern:
            parts.append(f"DON'T:\n{entry.dont_pattern}")
        if entry.reasoning:
            parts.append(f"Why: {entry.reasoning}")
    parts.append("--- END PLAYBOOK ---\n")
    return "\n".join(parts)


def format_playbook_context_note(entries: list[PlaybookEntry]) -> str:
    """Build a lighter context note suitable for message-level injection."""
    if not entries:
        return ""
    lines = ["[Playbook context — proven patterns from past sessions]"]
    for entry in entries:
        lines.append(f"• {entry.name} ({entry.task_type})")
        if entry.do_pattern:
            lines.append(f"  DO: {entry.do_pattern[:300]}")
        if entry.dont_pattern:
            lines.append(f"  DON'T: {entry.dont_pattern[:300]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session trace extraction helpers
# ---------------------------------------------------------------------------

def _extract_session_summary(messages: list[dict[str, Any]], max_chars: int = 2000) -> str:
    """Extract a compact summary of the session from its message list."""
    parts: list[str] = []
    char_count = 0
    for msg in messages:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))[:400]
        if role == "user":
            line = f"USER: {content}"
        elif role == "assistant":
            line = f"ASSISTANT: {content}"
        elif role == "tool":
            tool_name = msg.get("tool_name", "unknown")
            line = f"TOOL({tool_name}): {content[:200]}"
        else:
            continue
        if char_count + len(line) > max_chars:
            parts.append("... (truncated)")
            break
        parts.append(line)
        char_count += len(line)
    return "\n".join(parts)


def _extract_tool_trace(messages: list[dict[str, Any]], max_entries: int = 30) -> str:
    """Extract an ordered list of tool calls from the session."""
    entries: list[str] = []
    for msg in messages:
        # Tool calls from assistant messages
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "unknown")
                args_str = fn.get("arguments", "")
                if isinstance(args_str, str) and len(args_str) > 200:
                    args_str = args_str[:200] + "..."
                elif isinstance(args_str, dict):
                    args_str = json.dumps(args_str)[:200]
                entries.append(f"  {len(entries)+1}. {name}({args_str})")
                if len(entries) >= max_entries:
                    entries.append(f"  ... ({max_entries}+ tool calls, truncated)")
                    return "\n".join(entries)
        # Tool results
        if msg.get("role") == "tool":
            tool_name = msg.get("tool_name", "")
            if tool_name:
                content = str(msg.get("content", ""))[:150]
                entries.append(f"     → {tool_name} result: {content}")
                if len(entries) >= max_entries * 2:
                    break
    return "\n".join(entries) if entries else "(no tool calls recorded)"


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class AgentPlaybookMixin:
    """Mixin that adds playbook retrieval, distillation, and injection to Agent."""

    # -------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------

    async def _retrieve_playbooks(
        self,
        user_input: str,
        task_type: str | None = None,
        *,
        max_results: int = 2,
    ) -> list[PlaybookEntry]:
        """Retrieve the most relevant playbook entries for the current task.

        Respects the ``_playbook_override`` attribute set via the web UI:
        - ``None`` or ``""`` → auto mode (default behaviour below)
        - ``"__none__"`` → disabled, return empty list
        - any other string → force that specific playbook ID

        Auto strategy:
        1. Classify the task type from *user_input* (if not provided).
        2. Search by task_type first for a focused result set.
        3. Fall back to keyword search across all playbooks.
        4. Return at most *max_results* entries.
        """
        sm = get_session_manager()

        # ── Override handling ─────────────────────────────────
        override = getattr(self, "_playbook_override", None)
        if override == "__none__":
            return []
        if override and override != "__none__":
            entry = await sm.load_playbook(override)
            if entry:
                return [entry]
            log.warning("Playbook override ID not found, falling back to auto",
                        playbook_id=override)

        # ── Auto mode ─────────────────────────────────────────
        # Step 1: classify
        effective_type = task_type or classify_task_type(user_input)

        # Step 2: search within type
        candidates: list[PlaybookEntry] = []
        if effective_type:
            candidates = await sm.search_playbooks(
                user_input[:200], limit=max_results, task_type=effective_type,
            )

        # Step 3: broaden if needed
        if len(candidates) < max_results:
            broader = await sm.search_playbooks(
                user_input[:200], limit=max_results - len(candidates),
            )
            seen = {c.id for c in candidates}
            for b in broader:
                if b.id not in seen:
                    candidates.append(b)
                    if len(candidates) >= max_results:
                        break

        # Step 4: also try listing by type (for short user inputs that
        # don't match keywords but the type is correct)
        if not candidates and effective_type:
            candidates = await sm.list_playbooks(
                limit=max_results, task_type=effective_type,
            )

        return candidates[:max_results]

    # -------------------------------------------------------------------
    # Injection (planner + scale)
    # -------------------------------------------------------------------

    async def _build_playbook_block(
        self,
        user_input: str,
        task_type: str | None = None,
    ) -> str:
        """Retrieve and format playbooks for planner / scale injection.

        Returns empty string if no relevant playbooks found.
        Side effect: increments usage counters on matched playbooks.
        """
        entries = await self._retrieve_playbooks(user_input, task_type)
        if not entries:
            return ""

        # Increment usage
        sm = get_session_manager()
        for entry in entries:
            try:
                await sm.increment_playbook_usage(entry.id)
            except Exception:
                pass  # best-effort

        return format_playbook_block(entries)

    async def _build_playbook_context_note(
        self,
        user_input: str,
        task_type: str | None = None,
    ) -> str:
        """Retrieve and format playbooks as a lighter context note.

        Used for message-level injection (fallback path).
        """
        entries = await self._retrieve_playbooks(user_input, task_type)
        if not entries:
            return ""

        sm = get_session_manager()
        for entry in entries:
            try:
                await sm.increment_playbook_usage(entry.id)
            except Exception:
                pass

        return format_playbook_context_note(entries)

    def _build_playbook_context_note_sync(
        self,
        user_input: str,
        task_type: str | None = None,
    ) -> str:
        """Synchronous wrapper for ``_build_playbook_context_note``.

        Used by the sync ``_build_messages`` method in AgentContextMixin.
        Runs the async retrieval on the current event loop if available,
        otherwise returns empty (best-effort).
        """
        try:
            loop = asyncio.get_running_loop()
            # We're inside an event loop — create a task and use a
            # thread-safe approach.  Since ``_build_messages`` is called
            # from within an already-running async context, we can use
            # a cached result from the current turn if available.
            cached = getattr(self, "_playbook_context_cache", None)
            if cached is not None and cached.get("query") == user_input:
                return cached.get("note", "")
            # Schedule the async call to populate cache for next time.
            # For now return empty — the planner/scale paths will
            # still inject via their async injection points.
            return ""
        except RuntimeError:
            # No running loop — we can safely run_until_complete.
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    self._build_playbook_context_note(user_input, task_type)
                )
            finally:
                loop.close()

    # -------------------------------------------------------------------
    # Distillation: session trace → playbook proposal
    # -------------------------------------------------------------------

    async def _distill_playbook_from_session(
        self,
        session_id: str,
        rating: str,
        user_note: str = "",
    ) -> dict[str, Any] | None:
        """Run an LLM distillation pass on a session trace.

        Returns a parsed JSON dict with playbook fields, or None on failure.
        The caller is responsible for presenting the proposal to the user
        and persisting it after approval.
        """
        sm = get_session_manager()
        session = await sm.load_session(session_id)
        if not session:
            log.warning("Distill: session not found", session_id=session_id)
            return None

        messages = session.messages or []
        if not messages:
            log.warning("Distill: session has no messages", session_id=session_id)
            return None

        session_summary = _extract_session_summary(messages)
        tool_trace = _extract_tool_trace(messages)

        # Build the distillation prompt via InstructionLoader.
        instructions = getattr(self, "instructions", None)
        if instructions is None:
            from captain_claw.instructions import InstructionLoader
            instructions = InstructionLoader()

        note_block = f"\nUser note: {user_note}" if user_note else ""

        system_content = instructions.render(
            "playbook_distill_system_prompt.md",
            rating=rating,
            user_note=note_block,
        )
        user_content = instructions.render(
            "playbook_distill_user_prompt.md",
            session_summary=session_summary,
            tool_trace=tool_trace,
        )

        from captain_claw.llm import Message as LLMMessage

        distill_messages = [
            LLMMessage(role="system", content=system_content),
            LLMMessage(role="user", content=user_content),
        ]

        try:
            cfg_max = max(1, int(get_config().model.max_tokens))
            response = await self._complete_with_guards(
                messages=distill_messages,
                tools=None,
                interaction_label="playbook_distill",
                turn_usage={},
                max_tokens=min(2048, cfg_max),
            )
            raw = response.content or ""
        except Exception as e:
            log.error("Playbook distillation LLM call failed", error=str(e))
            return None

        # Parse JSON from response.
        payload = self._extract_json_object(raw) if hasattr(self, "_extract_json_object") else None
        if payload is None:
            # Fallback: try direct json.loads
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                log.warning("Distill: could not parse JSON from LLM response")
                return None

        if not isinstance(payload, dict):
            return None

        # Validate required fields.
        required = {"task_type", "name", "do_pattern", "dont_pattern"}
        missing = required - set(payload.keys())
        if missing:
            log.warning("Distill: missing fields in proposal", missing=missing)
            return None

        return payload

    async def _rate_and_distill_session(
        self,
        session_id: str | None,
        rating: str,
        user_note: str = "",
    ) -> dict[str, Any]:
        """Rate a session and run distillation to propose a playbook.

        Returns a dict with:
            - "proposal": the distilled playbook fields (or None)
            - "session_id": the rated session
            - "rating": good/bad
            - "status": "proposed" | "distill_failed" | "no_session"
        """
        # Default to current session if not specified.
        if not session_id:
            current = getattr(self, "session", None)
            if current:
                session_id = current.id
            else:
                return {"status": "no_session", "proposal": None, "session_id": None, "rating": rating}

        # Store rating in session metadata.
        sm = get_session_manager()
        session = await sm.load_session(session_id)
        if session:
            session.metadata["playbook_rating"] = rating
            if user_note:
                session.metadata["playbook_rating_note"] = user_note
            await sm.save_session(session)

        # Run distillation.
        proposal = await self._distill_playbook_from_session(session_id, rating, user_note)

        if proposal:
            return {
                "status": "proposed",
                "proposal": proposal,
                "session_id": session_id,
                "rating": rating,
            }
        return {
            "status": "distill_failed",
            "proposal": None,
            "session_id": session_id,
            "rating": rating,
        }

    async def _save_playbook_from_proposal(
        self,
        proposal: dict[str, Any],
        rating: str = "good",
        source_session: str | None = None,
    ) -> PlaybookEntry:
        """Persist a distilled playbook proposal to the database."""
        sm = get_session_manager()
        return await sm.create_playbook(
            name=str(proposal.get("name", "Unnamed playbook")),
            task_type=str(proposal.get("task_type", "other")),
            rating=rating,
            do_pattern=str(proposal.get("do_pattern", "")),
            dont_pattern=str(proposal.get("dont_pattern", "")),
            trigger_description=str(proposal.get("trigger_description", "")),
            reasoning=str(proposal.get("reasoning", "")),
            source_session=source_session,
        )
