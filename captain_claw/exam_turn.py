"""An onboarding exam's turn — Flight Deck asks the agent a question about
its documents (flight_deck/agent_exam.py) and grades the answer.

The question comes as an automated turn of kind ``exam`` on the agent's exam
lane, in a session of its own that is deleted when the turn ends, so no
question sees another's answer. The turn answers as the agent always would —
its instructions, its surfaced memory, its knowledge tool — but:

- it keeps nothing: no insights, reflections, intuitions, topics, captured
  to-dos or contacts, and its session is never indexed into semantic memory;
- it can't send or change anything: only the tools (and actions) that read
  run (:data:`EXAM_TOOLS`); anything else is refused, MCP and plugin tools
  included (fail closed). The web isn't among them: the exam asks what its
  documents say, and a grader judging against them can't tell a page it
  found from something made up;
- what it looked up goes back with the reply (:func:`report`), for the
  grader to tell "didn't look" from "looked in the wrong place".

The flag is the agent's ``_exam_turn``, set for the turn by the chat handler
and cleared when it ends.
"""

from __future__ import annotations

from typing import Any

KIND = "exam"
LANE = "EXAM"
CAPABILITY = "onboarding_exam"     # /api/version "capabilities": it knows this kind (Flight Deck asks first)
ATTR = "_exam_turn"
_CALLS_ATTR = "_exam_calls"
_RESULT_CHARS = 4000       # of a lookup's result, what goes back with the reply
_CALLS_MAX = 40

_DATASTORE_READS = frozenset({"list_tables", "describe", "query", "sql", "list_protections"})

# What an exam turn may run (tool → actions; None = every action). Reading only.
EXAM_TOOLS: dict[str, frozenset[str] | None] = {
    "knowledge": frozenset({"search", "section", "outline", "list"}),
    "read": None, "workspace_read": None, "glob": None, "grep": None, "history": None,
    "pdf_extract": None, "docx_extract": None, "xlsx_extract": None, "pptx_extract": None,
    "image_ocr": None, "image_vision": None,
    "topics": frozenset({"recall", "search", "list", "get"}),
    "insights": frozenset({"search", "list"}),
    "playbooks": frozenset({"list", "search", "info"}),
    "contacts": frozenset({"list", "search", "info"}),
    "todo": frozenset({"list"}),
    "datastore": _DATASTORE_READS,
    "vfs": frozenset({"info", "list_projects", "ls", "tree", "stat"}),
    "typesense": frozenset({"search"}),
    "docx": frozenset({"read", "changes", "comments"}),
    "xlsx": frozenset({"read", "check"}),
    "pptx": frozenset({"read", "check"}),
    "reasoning_effort": frozenset({"get"}),
}

REFUSED = ("{tool} isn't available in this conversation: it is an onboarding exam, where nothing is sent, saved "
           "or changed, and the web isn't searched. Answer from your documents (knowledge) — say so if you can't "
           "find it there.")


def active(agent: Any) -> bool:
    """Whether *agent* is answering an exam question right now."""
    return getattr(agent, ATTR, False) is True


def begin(agent: Any) -> None:
    setattr(agent, ATTR, True)
    setattr(agent, _CALLS_ATTR, [])


def end(agent: Any) -> None:
    setattr(agent, ATTR, False)
    setattr(agent, _CALLS_ATTR, [])


def tool_block(name: str, arguments: Any, agent: Any) -> str | None:
    """The refusal for a tool call in an exam turn, else None."""
    if agent is None or not active(agent):
        return None
    args = arguments if isinstance(arguments, dict) else {}
    action = str(args.get("action") or "").strip().lower()
    allowed = EXAM_TOOLS.get(str(name), False)
    if allowed is False or (allowed is not None and action not in allowed):
        return REFUSED.format(tool=f"{name} {action}" if action else str(name))
    return None


def note_call(agent: Any, name: str, arguments: Any, success: bool, output: Any) -> None:
    """Keep a call the exam turn made, for :func:`report`."""
    if not active(agent):
        return
    calls = getattr(agent, _CALLS_ATTR, None)
    if not isinstance(calls, list) or len(calls) >= _CALLS_MAX:
        return
    args = {k: v for k, v in (arguments if isinstance(arguments, dict) else {}).items()
            if not str(k).startswith("_") and isinstance(v, (str, int, float, bool))}
    calls.append({"tool": str(name), "args": {k: (v[:500] if isinstance(v, str) else v) for k, v in args.items()},
                  "ok": bool(success), "result": str(output or "")[:_RESULT_CHARS]})


def report(agent: Any) -> dict:
    """What the exam turn looked up, and the tokens it spent."""
    calls = getattr(agent, _CALLS_ATTR, None)
    usage = getattr(agent, "last_usage", None)
    return {"lookups": list(calls) if isinstance(calls, list) else [],
            "usage": {k: int(v) for k, v in usage.items() if isinstance(v, (int, float))}
            if isinstance(usage, dict) else {}}
