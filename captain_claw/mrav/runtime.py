"""MravRuntime — the micro agentic loop (plan → act → observe → digest).

Every LLM call fits the hard input cap; state lives on the Blackboard; each
ACT step performs exactly one action, grammar-constrained where the provider
supports it. Parallel to `Agent` — never imported by the classic loop.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.mrav import prompts
from captain_claw.mrav.digest import describe_result, digest_text
from captain_claw.mrav.ledger import (
    LedgerOverflowError,
    PromptLedger,
    Section,
    estimate_tokens,
    truncate_tokens,
)
from captain_claw.mrav.protocol import (
    ACT_RESPONSE_SCHEMA,
    PLAN_RESPONSE_SCHEMA,
    StepAction,
    parse_json_object,
    parse_plan,
    strip_thinking,
    validate_action,
)
from captain_claw.mrav.state import Blackboard, TraceWriter
from captain_claw.mrav.toolpack import ToolpackContext, build_toolpack

log = get_logger(__name__)

# Fractions of the usable budget per ACT section (sums < 1.0; the remainder
# is slack on top of the explicit ledger reserve). Measured against the real
# 53-tool registry at cap 8192: contract ~256 tok, core pack ~241 tok,
# full index ~860 tok — index needs 0.12 so no tool silently falls off it.
_BUDGET_FRACTIONS: dict[str, float] = {
    "contract": 0.08,
    "persona": 0.02,
    "tools": 0.22,
    "index": 0.12,
    "task": 0.05,
    "plan": 0.08,
    "summary": 0.08,
    "observations": 0.31,
    "now": 0.03,
}

_GIVE_UP_STREAK = 4
_ESCALATE_STREAK = 2


class MravRuntime:
    """One micro agent: a provider, a tool registry view, and a blackboard."""

    def __init__(
        self,
        *,
        provider: Any,
        tools: Any,
        config: Any,
        session_key: str,
        state_dir: Path,
        session_id: str | None = None,
        escalate_provider: Any | None = None,
        status_callback: Callable[[str], None] | None = None,
        tool_output_callback: Callable[[str, dict[str, Any], str], None] | None = None,
        llm_observer: Callable[[str, Any, list[Any], int, int], None] | None = None,
    ):
        self.provider = provider
        self.tools = tools
        self.cfg = config
        self.session_key = session_key or "default"
        self.session_id = session_id
        self.escalate_provider = escalate_provider
        self.status_callback = status_callback
        self.tool_output_callback = tool_output_callback
        # (label, response, messages, max_tokens, latency_ms) — the Agent
        # shell points this at _emit_llm_trace + _record_usage_to_db so a
        # mrav call is as visible as a classic one.
        self.llm_observer = llm_observer

        self.input_cap = int(getattr(config, "input_cap", 8192))
        self.output_cap = int(getattr(config, "output_cap", 1024))
        self.observation_cap = int(getattr(config, "observation_cap", 2500))
        self.digest_target = int(getattr(config, "digest_target", 400))
        self.max_steps = int(getattr(config, "max_steps", 24))
        self.act_retries = int(getattr(config, "act_retries", 2))
        self.replan_every = int(getattr(config, "replan_every", 6))
        self.max_pinned = int(getattr(config, "max_pinned_tools", 3))
        self.temperature = float(getattr(config, "temperature", 0.2))
        self.persona = str(getattr(config, "persona", "") or "").strip()
        self.escalate_enabled = bool(getattr(config, "escalate", False)) and escalate_provider is not None

        self.ledger = PromptLedger(self.input_cap, reserve=512)
        self.budgets = {
            name: max(64, int(self.ledger.usable * frac))
            for name, frac in _BUDGET_FRACTIONS.items()
        }

        safe_key = "".join(c if c.isalnum() or c in "-_" else "-" for c in self.session_key)[:80]
        self.state_path = state_dir / f"{safe_key}.state.json"
        self.trace = TraceWriter(state_dir / f"{safe_key}.trace.jsonl")
        self.board = Blackboard.load(self.state_path)
        self.last_usage: dict[str, int] = {}

    # ── plumbing ──

    def _status(self, text: str) -> None:
        if self.status_callback:
            try:
                self.status_callback(text)
            except Exception:
                pass

    def _add_usage(self, usage: dict[str, int] | None) -> None:
        for key, value in (usage or {}).items():
            try:
                self.last_usage[key] = self.last_usage.get(key, 0) + int(value)
            except (TypeError, ValueError):
                continue

    async def _llm(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        provider: Any | None = None,
        step_kind: str = "act",
    ) -> str:
        """One capped LLM call: final fit gate, structured when schema given."""
        import time

        from captain_claw.llm import Message

        fits, total = self.ledger.check_messages([system, user])
        if not fits:
            raise LedgerOverflowError(f"{step_kind} prompt {total} tokens exceeds cap {self.input_cap}")

        prov = provider or self.provider
        messages = [Message(role="system", content=system), Message(role="user", content=user)]
        out_tokens = max_tokens or self.output_cap
        started = time.monotonic()
        if schema is not None:
            response = await prov.complete_structured(
                messages, schema, temperature=self.temperature, max_tokens=out_tokens
            )
        else:
            response = await prov.complete(
                messages, None, temperature=self.temperature, max_tokens=out_tokens
            )
        latency_ms = int((time.monotonic() - started) * 1000)
        usage = getattr(response, "usage", None) or {}
        self._add_usage(usage)
        content = strip_thinking(getattr(response, "content", "") or "")
        in_tok = int(usage.get("prompt_tokens") or 0) or total
        out_tok = int(usage.get("completion_tokens") or 0) or estimate_tokens(content)
        self.trace.write(
            "llm",
            kind=step_kind,
            model=getattr(prov, "model", "?"),
            in_tokens=in_tok,
            out_tokens=out_tok,
            latency_ms=latency_ms,
            escalated=prov is not self.provider,
        )
        # Live token ticker — always on, like the classic status line.
        self._status(f"mrav: {step_kind} · {in_tok}→{out_tok} tok · {latency_ms/1000:.1f}s")
        # Same visibility as the classic loop: llm_trace card + usage-to-DB,
        # both handled by the observer the Agent shell wires in (it applies
        # the same ui.monitor_trace_llm gating as normal mode).
        if self.llm_observer is not None:
            try:
                self.llm_observer(f"mrav:{step_kind}", response, messages, out_tokens, latency_ms)
            except Exception as exc:
                log.debug("mrav llm observer failed", error=str(exc))
        return content

    async def _digest_call(self, system: str, user: str, max_tokens: int) -> str:
        return await self._llm(system, user, max_tokens=max_tokens, step_kind="digest")

    # ── prompt assembly ──

    def _toolpack(self) -> ToolpackContext:
        definitions = self.tools.get_definitions(session_id=self.session_id)
        return build_toolpack(definitions, pinned=self.board.pinned_tools)

    def _observations_text(self) -> str:
        lines = []
        for obs in self.board.observations:
            lines.append(f"[step {obs.step} {obs.kind}] {obs.label}\n{obs.text}")
        return "\n\n".join(lines)

    def _act_sections(self, pack: ToolpackContext, error: str) -> list[Section]:
        board = self.board
        plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(board.plan))
        facts_text = "\n".join(f"- {f}" for f in board.facts)
        plan_block = "\n".join(x for x in (plan_text, facts_text and prompts.H_FACTS, facts_text) if x)
        error_block = (
            f"{prompts.H_ERROR}\nYour previous reply was rejected: {error}\n"
            "Do not repeat it — choose a different action.\n\n"
            if error
            else ""
        )
        now_block = error_block + f"{prompts.H_NOW}\n{prompts.ACT_NOW}"
        return [
            Section("contract", prompts.ACT_CONTRACT, self.budgets["contract"], keep="head"),
            Section("persona", f"Your role: {self.persona}" if self.persona else "", self.budgets["persona"], keep="head"),
            Section("tools", f"{prompts.H_TOOLS}\n{pack.defs_text}", self.budgets["tools"], keep="head"),
            Section("index", f"{prompts.H_INDEX}\n{pack.index_text}", self.budgets["index"], keep="head"),
            Section("task", f"{prompts.H_TASK}\n{board.task}", self.budgets["task"], keep="head"),
            Section("plan", f"{prompts.H_PLAN}\n{plan_block}" if plan_block else "", self.budgets["plan"], keep="head", flex=True),
            Section("summary", f"{prompts.H_SUMMARY}\n{board.summary}" if board.summary else "", self.budgets["summary"], keep="tail", flex=True),
            Section("observations", f"{prompts.H_OBSERVATIONS}\n{self._observations_text()}" if board.observations else "", self.budgets["observations"], keep="tail", flex=True),
            Section("now", now_block, self.budgets["now"], keep="tail"),
        ]

    def _assemble(self, sections: list[Section]) -> tuple[str, str]:
        fitted, report = self.ledger.fit(sections)
        system = fitted.get("contract", "")
        body = "\n\n".join(fitted[s.name] for s in sections if s.name != "contract" and fitted.get(s.name))
        self.trace.write("prompt", total_tokens=report.total_tokens, trimmed=report.trimmed, squeezed=report.squeezed)
        return system, body

    # ── steps ──

    async def _plan_step(self, pack: ToolpackContext, replan: bool) -> None:
        board = self.board
        tool_names = ", ".join(sorted(pack.all_names)) or "(no tools)"
        parts = [f"{prompts.H_TASK}\n{truncate_tokens(board.task, self.budgets['task'], keep='head')}"]
        parts.append(f"Available tools: {truncate_tokens(tool_names, 300, keep='head')}")
        if board.summary:
            parts.append(f"{prompts.H_SUMMARY}\n{truncate_tokens(board.summary, self.budgets['summary'], keep='tail')}")
        if replan and board.observations:
            parts.append(
                f"{prompts.H_OBSERVATIONS}\n"
                + truncate_tokens(self._observations_text(), self.budgets["observations"] // 2, keep="tail")
            )
        parts.append(f"{prompts.H_NOW}\n{prompts.PLAN_NOW}")
        try:
            content = await self._llm(
                prompts.PLAN_CONTRACT,
                "\n\n".join(parts),
                schema=PLAN_RESPONSE_SCHEMA,
                max_tokens=min(self.output_cap, 512),
                step_kind="replan" if replan else "plan",
            )
            plan = parse_plan(content)
            if plan:
                board.plan = plan
                self.trace.write("plan", steps=plan, replan=replan)
        except Exception as exc:
            # A missing plan is survivable; ACT works without one.
            log.debug("mrav plan step failed", error=str(exc))
            self.trace.write("plan_failed", error=str(exc), replan=replan)

    async def _act_step(self, pack: ToolpackContext, use_escalation: bool) -> tuple[StepAction | None, str]:
        """Run one ACT with parse/validation retries. Returns (action, last_error)."""
        error = ""
        provider = self.escalate_provider if use_escalation else None
        for attempt in range(self.act_retries + 1):
            sections = self._act_sections(pack, error)
            system, body = self._assemble(sections)
            content = await self._llm(
                system, body, schema=ACT_RESPONSE_SCHEMA, provider=provider, step_kind="act"
            )
            action, error = validate_action(
                parse_json_object(content), pack.visible_names, pack.all_names
            )
            if action is not None:
                return action, ""
            self.trace.write("act_retry", attempt=attempt + 1, error=error, raw=content[:400])
        return None, error

    async def _run_tool(self, action: StepAction) -> tuple[bool, str]:
        """Execute one tool; returns (success, observation_text)."""
        try:
            result = await self.tools.execute(
                action.tool,
                action.args,
                session_id=self.session_id,
            )
        except Exception as exc:
            return False, f"TOOL FAILED: {type(exc).__name__}: {exc}"
        text = describe_result(result)
        return bool(getattr(result, "success", True)), text

    async def _maybe_compress(self) -> None:
        board = self.board
        budget = self.budgets["observations"]
        if board.observation_tokens() <= int(budget * 0.9) or len(board.observations) <= 2:
            return
        keep_from = max(1, len(board.observations) // 2)
        old, board.observations = board.observations[:keep_from], board.observations[keep_from:]
        old_text = "\n\n".join(f"[{o.label}] {o.text}" for o in old)
        words = max(60, int(self.budgets["summary"] * 0.7))
        user = (
            f"{prompts.H_SUMMARY}\n{board.summary or '(empty)'}\n\n"
            f"OLDER OBSERVATIONS:\n{truncate_tokens(old_text, budget, keep='tail')}"
        )
        try:
            new_summary = await self._llm(
                prompts.COMPRESS_INSTRUCTION.format(words=words),
                user,
                max_tokens=min(self.output_cap, self.budgets["summary"] + 128),
                step_kind="compress",
            )
            if new_summary.strip():
                board.summary = truncate_tokens(new_summary.strip(), self.budgets["summary"], keep="head")
                self.trace.write("compress", folded=len(old), summary_tokens=estimate_tokens(board.summary))
                return
        except Exception as exc:
            log.debug("mrav compress failed", error=str(exc))
        # Model unavailable for compression → keep facts crudely rather than lose them.
        board.summary = truncate_tokens(
            (board.summary + "\n" + old_text).strip(), self.budgets["summary"], keep="tail"
        )

    async def _observe_tool(self, action: StepAction, success: bool, text: str) -> None:
        board = self.board
        label = f"{action.tool}({json.dumps(action.args, ensure_ascii=False, default=str)[:160]})"
        if estimate_tokens(text) > self.observation_cap:
            self._status(f"mrav: digesting {action.tool} output")
            text = await digest_text(
                self._digest_call,
                board.task,
                f"{action.tool} output",
                text,
                self.digest_target,
            )
            kind = "tool"
        else:
            kind = "tool" if success else "error"
        board.add_observation(kind, label, text)
        if self.tool_output_callback:
            try:
                self.tool_output_callback(action.tool, action.args, text)
            except Exception:
                pass

    def _finish(self, reply: str, outcome: str) -> str:
        board = self.board
        board.tasks_completed += 1
        line = f"Task: {truncate_tokens(board.task, 60, keep='head')} → {outcome}: {truncate_tokens(reply, 80, keep='head')}"
        board.summary = truncate_tokens((board.summary + "\n" + line).strip(), self.budgets["summary"], keep="tail")
        board.save(self.state_path)
        self.trace.write("task_end", outcome=outcome, steps=board.step, usage=self.last_usage)
        self._status("mrav: done")
        return reply

    # ── the loop ──

    async def run(self, user_input: str, *, cancel_event: asyncio.Event | None = None) -> str:
        board = self.board
        board.new_task(user_input.strip())
        self.last_usage = {}
        self.trace.write("task_start", task=board.task, model=getattr(self.provider, "model", "?"))
        self._status("mrav: planning")

        pack = self._toolpack()
        await self._plan_step(pack, replan=False)

        last_call_sig = ""
        escalation_used_for_streak = False

        while board.step < self.max_steps:
            if cancel_event is not None and cancel_event.is_set():
                return self._finish("Cancelled.", "cancelled")
            board.step += 1
            pack = self._toolpack()

            use_escalation = (
                self.escalate_enabled
                and board.consecutive_failures >= _ESCALATE_STREAK
                and not escalation_used_for_streak
            )
            if use_escalation:
                escalation_used_for_streak = True
                self.trace.write("escalate", step=board.step, streak=board.consecutive_failures)
                self._status("mrav: escalating one step")
            else:
                self._status(f"mrav: step {board.step}/{self.max_steps}")

            try:
                action, error = await self._act_step(pack, use_escalation)
            except LedgerOverflowError:
                raise
            except Exception as exc:
                board.consecutive_failures += 1
                board.add_observation("error", "llm", f"LLM call failed: {exc}")
                self.trace.write("act_error", step=board.step, error=str(exc))
                if board.consecutive_failures >= _GIVE_UP_STREAK:
                    return self._finish(
                        f"I could not continue: the model kept failing ({exc}).", "error"
                    )
                continue

            if action is None:
                board.consecutive_failures += 1
                board.add_observation("error", "protocol", f"Invalid step response: {error}")
                if board.consecutive_failures >= _GIVE_UP_STREAK:
                    return self._finish(
                        "I could not complete this: the model repeatedly produced invalid steps. "
                        f"Last problem: {error}",
                        "protocol_failure",
                    )
                continue

            if action.kind == "final":
                return self._finish(action.text, "final")

            if action.kind == "give_up":
                return self._finish(f"I could not complete this: {action.reason}", "give_up")

            if action.kind == "open_tool":
                sig = f"open_tool:{action.name}"
                if sig == last_call_sig:
                    board.consecutive_failures += 1
                    board.add_observation(
                        "error",
                        "loop_guard",
                        f"You already did open_tool on '{action.name}'. It is in TOOLS — "
                        "call it now with its args, or pick another action.",
                    )
                    self.trace.write("repeat_call", step=board.step, tool=sig)
                    continue
                last_call_sig = sig
                if action.name in pack.visible_names:
                    # Forgiving no-op: 2B-class models "open" core tools out
                    # of habit; a rejection here loops them (seen live, E2B).
                    # Bait the correct next reply with a concrete template —
                    # small models follow examples far better than prose.
                    compact = pack.visible.get(action.name)
                    args_hint = ", ".join(
                        f'"{p}":...' for p in (compact.param_names[:3] if compact else [])
                    )
                    board.add_observation(
                        "note",
                        "open_tool",
                        f"Tool '{action.name}' is already in TOOLS. Call it now, like: "
                        f'{{"action":"tool","tool":"{action.name}","args":{{{args_hint}}}}}',
                    )
                    self.trace.write("open_tool_noop", step=board.step, name=action.name)
                    continue
                board.pin_tool(action.name, self.max_pinned)
                board.add_observation("note", "open_tool", f"Loaded schema for tool '{action.name}'; it is now in TOOLS.")
                board.consecutive_failures = 0
                escalation_used_for_streak = False
                self.trace.write("open_tool", step=board.step, name=action.name)
                board.save(self.state_path)
                continue

            # action.kind == "tool"
            call_sig = f"{action.tool}:{json.dumps(action.args, sort_keys=True, default=str)}"
            if call_sig == last_call_sig:
                board.consecutive_failures += 1
                board.add_observation(
                    "error",
                    "loop_guard",
                    f"You already ran exactly this {action.tool} call. Do something different or finish.",
                )
                self.trace.write("repeat_call", step=board.step, tool=action.tool)
                continue
            last_call_sig = call_sig

            self._status(f"mrav: step {board.step} — {action.tool}")
            success, text = await self._run_tool(action)
            self.trace.write(
                "tool",
                step=board.step,
                tool=action.tool,
                success=success,
                out_tokens=estimate_tokens(text),
            )
            await self._observe_tool(action, success, text)
            if success:
                board.consecutive_failures = 0
                escalation_used_for_streak = False
            else:
                board.consecutive_failures += 1
                if board.consecutive_failures >= _GIVE_UP_STREAK:
                    return self._finish(
                        f"I could not complete this: tool '{action.tool}' kept failing. Last output:\n{truncate_tokens(text, 300, keep='head')}",
                        "tool_failure",
                    )

            await self._maybe_compress()
            if self.replan_every > 0 and board.step % self.replan_every == 0:
                await self._plan_step(pack, replan=True)
            board.save(self.state_path)

        gist = truncate_tokens(self._observations_text(), 400, keep="tail")
        return self._finish(
            "I ran out of steps before finishing. What I found so far:\n"
            + (board.summary + "\n\n" if board.summary else "")
            + gist,
            "steps_exhausted",
        )
