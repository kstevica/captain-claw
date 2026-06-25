"""Background watcher for interactive terminal sessions.

When the agent opens an interactive terminal on the user's machine and then
goes idle, the program it launched may be **parked at a prompt** waiting for
a choice (``claude``'s "Do you want to create index.html? 1/2/3", a
``(y/n)``, an ``ssh`` host-key prompt, …). Nothing would move until the next
time the user pokes the agent.

This watcher closes that gap. Per open session it polls the daemon's
non-advancing ``/peek`` (so it never steals output from the agent's own
reads); when output has been **stable and prompt-shaped** for a few seconds,
it makes ONE pure (tool-less) LLM call that — given the terminal tail *and*
the recent task context — plays the agent's role and decides:

  - **answer**   → the choice is obvious and safe given the task; the watcher
                   types it into the terminal and tells the user what it did.
  - **ask_user** → a real decision (or anything destructive/irreversible);
                   the watcher relays the question to the user's channel.
  - **wait**     → not actually waiting; keep polling.

Safety by construction: the watcher NEVER touches ``agent.session`` (there's
no turn lock, so a re-entrant ``agent.complete()`` could corrupt a concurrent
user turn). It only does pure-LLM calls, direct daemon writes, and
origin delivery — all read-only with respect to the agent.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

_POLL_INTERVAL = float(os.environ.get("CLAW_TERMINAL_WATCH_INTERVAL", "3"))
_QUIET_SECONDS = float(os.environ.get("CLAW_TERMINAL_WATCH_QUIET", "6"))
_LLM_COOLDOWN = float(os.environ.get("CLAW_TERMINAL_WATCH_COOLDOWN", "20"))
_MAX_PEEK_FAILS = 5
# Hard backstop against a runaway answer→prompt→answer loop: after this many
# auto-answers in a row (no intervening "wait"/"ask_user"), stop auto-answering
# and hand the decision to the user instead.
_MAX_CONSECUTIVE_AUTO = int(os.environ.get("CLAW_TERMINAL_WATCH_MAX_AUTO", "3"))

_ANSI_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]|\x1b\[[0-?]*[ -/]*[@-~]|\r")

# Words that make a prompt too risky to auto-answer — force ask_user even if
# the model judged it answerable. Defense in depth around the LLM.
_DESTRUCTIVE_RE = re.compile(
    r"\brm\s"                                          # rm <anything>
    r"|\b(?:delete|remove|overwrite|truncate|format|wipe|erase|deploy|"
    r"production|prod|sudo|payment|purchase|charge|transfer|revoke|grant|"
    r"credential|secret|password|shutdown|reboot|kill)\b"
    r"|drop\s+table|force[- ]?push|--force|reset\s+--hard|chmod\s+777|"
    r"send\s+money|:\(\)\{",                           # also catch a fork bomb
    re.IGNORECASE,
)


def _enabled() -> bool:
    return os.environ.get("CLAW_TERMINAL_WATCH", "1").lower() not in ("0", "false", "no", "off")


def _auto_allowed() -> bool:
    return os.environ.get("CLAW_TERMINAL_WATCH_AUTO", "1").lower() not in ("0", "false", "no", "off")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


_SYSTEM_PROMPT = (
    "You are the AI agent driving an interactive terminal on the USER'S OWN "
    "machine on their behalf. A background watcher noticed the terminal went "
    "quiet and may be blocked waiting for input. Look at the terminal output "
    "and the recent task context, then decide what to do.\n\n"
    "Reply with ONLY a JSON object:\n"
    "{\n"
    '  "waiting": true|false,        // is the program blocked, waiting for the user to type?\n'
    '  "question": "<one line: what it is asking>",\n'
    '  "decision": "answer" | "ask_user" | "wait",\n'
    '  "answer": "<exact characters to type if decision=answer, e.g. \\"1\\" or \\"y\\" or a name>",\n'
    '  "send_enter": true|false,     // press Enter after the answer\n'
    '  "user_message": "<if decision=ask_user: the question to send to the user>",\n'
    '  "reason": "<brief>"\n'
    "}\n\n"
    "Rules:\n"
    "- decision \"answer\" ONLY when the answer is obvious AND safe given the task, "
    "and the action is reversible (confirming a file creation, picking the clearly "
    "intended menu option, accepting a sane default).\n"
    "- decision \"ask_user\" for any real choice only the user can make, or anything "
    "destructive / irreversible / risky (deleting, overwriting, force-push, deploys, "
    "payments, sending messages, granting access, credentials) — and whenever you are "
    "unsure. Prefer ask_user when in doubt.\n"
    "- decision \"wait\" if it is NOT actually waiting for input (still working, or just "
    "an idle shell prompt with nothing pending).\n"
    "- \"answer\" must be the literal characters to type, nothing else."
)


class _SessState:
    __slots__ = ("last_total", "last_change", "handled_total", "last_llm", "fails", "auto_count")

    def __init__(self) -> None:
        self.last_total = -1
        self.last_change = time.time()
        self.handled_total: int | None = None
        self.last_llm = 0.0
        self.fails = 0
        self.auto_count = 0  # consecutive auto-answers without a wait/ask break


class TerminalWatcher:
    """Owns the polling loop for one TerminalTool's open sessions."""

    def __init__(self, tool: Any) -> None:
        self.tool = tool  # TerminalTool: provides _call() + _agent
        self._sessions: dict[str, _SessState] = {}
        self._task: asyncio.Task | None = None

    # ── registration (called by the tool) ────────────────────────────
    def track(self, session_id: str) -> None:
        if not _enabled() or getattr(self.tool, "_agent", None) is None:
            return
        if session_id and session_id not in self._sessions:
            self._sessions[session_id] = _SessState()
            log.info("terminal watcher tracking session %s", session_id)
        self._ensure_loop()

    def untrack(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _ensure_loop(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            self._task = asyncio.get_running_loop().create_task(self._run())
        except RuntimeError:
            self._task = None  # no running loop (e.g. sync context)

    # ── loop ─────────────────────────────────────────────────────────
    async def _run(self) -> None:
        try:
            while self._sessions:
                await asyncio.sleep(_POLL_INTERVAL)
                for sid in list(self._sessions):
                    try:
                        await self._check(sid)
                    except Exception as exc:  # never let one session kill the loop
                        log.debug("terminal watcher check failed", session=sid, error=str(exc))
        finally:
            self._task = None

    async def _check(self, sid: str) -> None:
        st = self._sessions.get(sid)
        if st is None:
            return
        try:
            peek = await self.tool._call("/peek", {"session_id": sid})
            st.fails = 0
        except Exception:
            st.fails += 1
            if st.fails >= _MAX_PEEK_FAILS:
                self.untrack(sid)  # session gone or daemon unreachable
            return

        if not peek.get("alive", True):
            self.untrack(sid)
            return

        total = int(peek.get("total", 0))
        now = time.time()
        if total != st.last_total:
            # new output → reset the quiet timer and allow a fresh decision
            st.last_total = total
            st.last_change = now
            st.handled_total = None
            return

        if not peek.get("prompt_like"):
            return
        if now - st.last_change < _QUIET_SECONDS:
            return
        if st.handled_total == total:
            return  # already decided on this exact stable prompt
        if now - st.last_llm < _LLM_COOLDOWN:
            return

        st.last_llm = now
        st.handled_total = total  # mark first, so a slow LLM call can't double-fire
        await self._decide(sid, str(peek.get("tail", "")))

    # ── decision (pure LLM) ──────────────────────────────────────────
    async def _decide(self, sid: str, tail: str) -> None:
        agent = getattr(self.tool, "_agent", None)
        st = self._sessions.get(sid)
        if agent is None or st is None:
            return
        clean = _strip(tail)[-1500:]
        if not clean.strip():
            return

        from captain_claw.llm import Message

        user = (
            f"Recent task context:\n{self._recent_context(agent)}\n\n"
            f"The terminal you are driving on the user's machine currently shows:\n"
            f"```\n{clean}\n```\n\nDecide and reply with the JSON object only."
        )
        try:
            resp = await agent._complete_with_guards(
                messages=[
                    Message(role="system", content=_SYSTEM_PROMPT),
                    Message(role="user", content=user),
                ],
                tools=None,
                interaction_label="terminal_watcher",
                max_tokens=400,
            )
        except Exception as exc:
            log.debug("terminal watcher LLM call failed", error=str(exc))
            return

        data = _parse_json(getattr(resp, "content", "") or "")
        if not data or not data.get("waiting"):
            return

        decision = str(data.get("decision", "")).strip().lower()
        question = str(data.get("question", "")).strip()[:300]
        answer = str(data.get("answer", ""))
        send_enter = bool(data.get("send_enter", True))

        # Guard rails: never auto-answer anything destructive, honour the global
        # "no auto-answer" switch, and stop a runaway answer→prompt→answer loop.
        risky = bool(_DESTRUCTIVE_RE.search(clean + " " + question + " " + answer))
        too_many = st.auto_count >= _MAX_CONSECUTIVE_AUTO
        if decision == "answer" and (risky or too_many or not _auto_allowed()):
            decision = "ask_user"
            if too_many:
                data["user_message"] = (
                    f"The terminal keeps prompting — I've auto-answered {st.auto_count} in a "
                    f"row and stopped. It's now asking: {question}. How should I proceed?"
                )
            else:
                data["user_message"] = data.get("user_message") or (
                    f"The terminal is asking: {question}. I didn't want to answer this one "
                    "automatically — how should I respond?"
                )

        # Reset the consecutive-auto counter on any non-answer outcome.
        if decision != "answer":
            st.auto_count = 0

        if decision == "answer":
            try:
                await self.tool._call(
                    "/input", {"session_id": sid, "data": answer, "enter": send_enter}
                )
            except Exception as exc:
                log.debug("terminal watcher auto-answer failed", error=str(exc))
                return
            st.auto_count += 1
            log.info("terminal watcher auto-answered %s: %r", sid, answer)
            await self._notify(
                agent,
                f"🤖 The terminal was waiting — {question}\nI answered: {answer or '<enter>'}",
            )
        elif decision == "ask_user":
            msg = str(data.get("user_message") or f"The terminal is waiting for input: {question}")
            await self._notify(agent, f"⏳ {msg}")
        # decision == "wait" (or anything else): do nothing, keep polling.

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _recent_context(agent: Any, limit: int = 8, maxlen: int = 400) -> str:
        sess = getattr(agent, "session", None)
        msgs = getattr(sess, "messages", None) or []
        out: list[str] = []
        for m in list(msgs)[-limit:]:
            if isinstance(m, dict):
                role, content = m.get("role", ""), m.get("content", "")
            else:
                role, content = getattr(m, "role", ""), getattr(m, "content", "")
            if not isinstance(content, str):
                content = str(content)
            if content.strip():
                out.append(f"{role}: {content[:maxlen]}")
        return "\n".join(out) or "(no recent context)"

    async def _notify(self, agent: Any, text: str) -> None:
        try:
            from captain_claw.delivery import deliver_to_origin

            sid = getattr(getattr(agent, "session", None), "id", None)
            if sid:
                await deliver_to_origin(agent, str(sid), text)
        except Exception as exc:
            log.debug("terminal watcher notify failed", error=str(exc))


def _parse_json(raw: str) -> dict | None:
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        fence = text.find("\n```")
        if fence != -1:
            text = text[:fence]
    # tolerate prose around the object
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None
