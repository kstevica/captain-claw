"""Playbook retrieval, distillation, and context injection for Agent.

This mixin handles:
- Retrieving relevant playbooks from the persistent store
- Formatting playbook blocks for injection into planner / scale / context
- Building playbook context notes for message assembly
- Distilling session traces into playbook proposals
- Session rating flow

Playbooks are human-reviewed do/don't patterns, concrete examples, and
linked scripts that improve orchestration decisions by providing reusable
knowledge of what works (and what doesn't) for recurring task types.
"""

import asyncio
import json
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.session import PlaybookEntry, ScriptEntry, get_session_manager

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

def format_playbook_block(
    entries: list[PlaybookEntry],
    resolved_scripts: dict[str, list[ScriptEntry]] | None = None,
) -> str:
    """Render a concise playbook context block for prompt injection.

    Returns an empty string when *entries* is empty so callers can safely
    concatenate or pass as a template variable.

    *resolved_scripts* maps playbook IDs to pre-fetched ScriptEntry lists
    (populated by the async caller that has access to SessionManager).
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
        if entry.examples:
            parts.append(f"EXAMPLES:\n{entry.examples}")
        if entry.reasoning:
            parts.append(f"Why: {entry.reasoning}")
        # Linked scripts (pre-resolved by caller).
        scripts = (resolved_scripts or {}).get(entry.id, [])
        if scripts:
            script_lines = ["SCRIPTS:"]
            for s in scripts:
                lang = f" ({s.language})" if s.language else ""
                script_lines.append(f"  - {s.name}{lang}: {s.file_path}")
                if s.purpose:
                    script_lines.append(f"    {s.purpose}")
            parts.append("\n".join(script_lines))
    parts.append("--- END PLAYBOOK ---\n")
    return "\n".join(parts)


def format_playbook_context_note(
    entries: list[PlaybookEntry],
    resolved_scripts: dict[str, list[ScriptEntry]] | None = None,
) -> str:
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
        if entry.examples:
            lines.append(f"  EXAMPLES: {entry.examples[:300]}")
        scripts = (resolved_scripts or {}).get(entry.id, [])
        if scripts:
            names = ", ".join(f"{s.name} ({s.file_path})" for s in scripts)
            lines.append(f"  SCRIPTS: {names}")
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
    # Monitor helper
    # -------------------------------------------------------------------

    def _emit_playbook_event(self, action: str, details: dict | None = None, output: str = "") -> None:
        """Emit a playbook activity event to the monitor panel."""
        callback = getattr(self, "tool_output_callback", None)
        if callback is None:
            return
        try:
            args = {"action": action}
            if details:
                args.update(details)
            callback("playbook", args, output)
        except Exception:
            pass

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
        self._emit_playbook_event(
            "search",
            {"task_type": effective_type or "auto", "query": user_input[:100]},
            f"Searching playbooks (type: {effective_type or 'any'})…",
        )

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

        result = candidates[:max_results]
        if result:
            names = ", ".join(e.name for e in result)
            self._emit_playbook_event(
                "matched",
                {"count": len(result), "names": names},
                f"Found {len(result)} playbook(s): {names}",
            )
        else:
            self._emit_playbook_event("no_match", {}, "No matching playbooks found")
        return result

    # -------------------------------------------------------------------
    # Playbook approval
    # -------------------------------------------------------------------

    async def _request_playbook_approval(
        self,
        entries: list[PlaybookEntry],
    ) -> bool:
        """Request user approval before injecting playbooks.

        Uses the async ``playbook_approval_callback`` if configured.
        When no callback is set, playbooks are used without approval.
        Returns True if approved (or no callback), False if declined.
        """
        callback = getattr(self, "playbook_approval_callback", None)
        if callback is None:
            return True
        try:
            descriptions = []
            for e in entries:
                desc = f"**{e.name}** [{e.task_type}]"
                if e.trigger_description:
                    desc += f" — {e.trigger_description}"
                descriptions.append(desc)
            message = "Playbook match found:\n" + "\n".join(
                f"  {d}" for d in descriptions
            )
            names = ", ".join(e.name for e in entries)
            self._emit_playbook_event(
                "approval_requested",
                {"names": names},
                f"Requesting approval for: {names}",
            )
            approved = await callback(message)
            self._emit_playbook_event(
                "approval_resolved",
                {"approved": approved, "names": names},
                f"Playbook {'approved' if approved else 'declined'}: {names}",
            )
            return approved
        except Exception:
            log.warning("Playbook approval callback failed, proceeding without approval")
            return True

    # -------------------------------------------------------------------
    # Injection (planner + scale)
    # -------------------------------------------------------------------

    @staticmethod
    async def _resolve_playbook_scripts(
        entries: list[PlaybookEntry],
    ) -> dict[str, list[ScriptEntry]]:
        """Resolve linked script_ids for a list of playbook entries.

        Returns a dict mapping playbook ID → list of resolved ScriptEntry
        objects.  Unknown IDs are silently skipped.
        """
        result: dict[str, list[ScriptEntry]] = {}
        sm = get_session_manager()
        for entry in entries:
            if not entry.script_ids:
                continue
            scripts: list[ScriptEntry] = []
            for sid in (s.strip() for s in entry.script_ids.split(",") if s.strip()):
                try:
                    script = await sm.load_script(sid)
                    if script:
                        scripts.append(script)
                except Exception:
                    pass
            if scripts:
                result[entry.id] = scripts
        return result

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

        # Notify user which playbooks are being used.
        names = ", ".join(e.name for e in entries)
        self._set_runtime_status(f"Using playbook: {names}")

        # Request approval if callback is configured.
        approved = await self._request_playbook_approval(entries)
        if not approved:
            self._set_runtime_status("Playbook usage declined by user")
            return ""

        # Increment usage
        sm = get_session_manager()
        for entry in entries:
            try:
                await sm.increment_playbook_usage(entry.id)
            except Exception:
                pass  # best-effort

        resolved_scripts = await self._resolve_playbook_scripts(entries)
        block = format_playbook_block(entries, resolved_scripts=resolved_scripts)
        self._emit_playbook_event(
            "injected",
            {"names": names, "target": "planner", "block_chars": len(block)},
            f"Injected playbook into planner: {names} ({len(block)} chars)",
        )
        return block

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

        # Notify user which playbooks are being used.
        names = ", ".join(e.name for e in entries)
        self._set_runtime_status(f"Using playbook: {names}")

        # Request approval if callback is configured.
        approved = await self._request_playbook_approval(entries)
        if not approved:
            self._set_runtime_status("Playbook usage declined by user")
            return ""

        sm = get_session_manager()
        for entry in entries:
            try:
                await sm.increment_playbook_usage(entry.id)
            except Exception:
                pass

        resolved_scripts = await self._resolve_playbook_scripts(entries)
        return format_playbook_context_note(entries, resolved_scripts=resolved_scripts)

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

        self._emit_playbook_event(
            "distill_start",
            {"session_id": session_id, "rating": rating, "message_count": len(messages)},
            f"Distilling session {session_id[:8]}… ({len(messages)} messages, rated {rating})",
        )

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
            self._emit_playbook_event(
                "distill_failed",
                {"session_id": session_id, "reason": "invalid_json"},
                "Distillation failed: LLM response was not valid JSON",
            )
            return None

        # Validate required fields.
        required = {"task_type", "name", "do_pattern", "dont_pattern"}
        missing = required - set(payload.keys())
        if missing:
            log.warning("Distill: missing fields in proposal", missing=missing)
            self._emit_playbook_event(
                "distill_failed",
                {"session_id": session_id, "reason": "missing_fields", "missing": list(missing)},
                f"Distillation failed: missing fields {missing}",
            )
            return None

        pb_name = payload.get("name", "?")
        self._emit_playbook_event(
            "distill_complete",
            {"session_id": session_id, "name": pb_name, "task_type": payload.get("task_type", "?")},
            f"Distilled playbook proposal: {pb_name} [{payload.get('task_type', '?')}]",
        )
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
        entry = await sm.create_playbook(
            name=str(proposal.get("name", "Unnamed playbook")),
            task_type=str(proposal.get("task_type", "other")),
            rating=rating,
            do_pattern=str(proposal.get("do_pattern", "")),
            dont_pattern=str(proposal.get("dont_pattern", "")),
            trigger_description=str(proposal.get("trigger_description", "")),
            reasoning=str(proposal.get("reasoning", "")),
            examples=str(proposal["examples"]) if proposal.get("examples") else None,
            source_session=source_session,
        )
        self._emit_playbook_event(
            "saved",
            {"id": entry.id, "name": entry.name, "task_type": entry.task_type},
            f"Playbook saved: {entry.name} [{entry.task_type}] (id: {entry.id[:8]}…)",
        )
        return entry
