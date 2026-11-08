"""course — the goals this agent's work is for (Flight Deck's Course).

The live goals are already in the agent's prompt ("Your course"). This tool
reaches what the prompt leaves out and records the work done toward them:

* ``view``: the deck's goals, the owner's and this agent's own, with status,
  progress, when work last moved each one, review and due dates.
* ``link``: record that something this agent did moved a goal forward — a
  session, a scheduled task, a project, a Taskboard task or a plan.
* ``report``: an estimate of how far along one of the owner's goals (or the
  agent's own) is; ``looks_achieved`` asks the owner on their Line. It never
  marks a goal done — only a person does.

The registry shows the tool only under Flight Deck and never on a member's
turn (metadata ``requires_line``, :func:`course_client.usable`).
"""

from __future__ import annotations

from typing import Any

from captain_claw import course_client as client
from captain_claw.course_client import CourseError
from captain_claw.tools.registry import Tool, ToolResult

_KINDS = ["session", "schedule", "project", "task", "plan"]
_SCOPE = {"instance": "deck-wide", "user": "your owner's", "agent": "your own"}
_SHORT = {"strategic": "strategic", "long": "long-term", "medium": "medium-term",
          "short": "short-term"}


def _date(iso: Any) -> str:
    return str(iso or "")[:10]


def _line(g: dict[str, Any]) -> str:
    bits = [f"G:{g.get('id')}", _SHORT.get(str(g.get("horizon")), str(g.get("horizon"))),
            _SCOPE.get(str(g.get("scope")), str(g.get("scope")))]
    if g.get("status") != "active":
        bits.append(str(g.get("status")))
    if g.get("stale"):
        bits.append("out of your prompt until it's reviewed")
    head = f"- [{' · '.join(bits)}] {' '.join(str(g.get('title') or '').split())}"
    out = [head]
    if g.get("success"):
        out.append(f"  done when: {' '.join(str(g['success']).split())}")
    facts = []
    if g.get("progress") is not None:
        facts.append(f"progress {round(float(g['progress']) * 100)}%"
                     + (f" ({' '.join(str(g.get('progress_note') or '').split())[:160]})"
                        if g.get("progress_note") else ""))
    facts.append(f"last moved {_date(g['last_moved'])}" if g.get("last_moved") else "no work linked yet")
    if g.get("due_at"):
        facts.append(f"due {_date(g['due_at'])}")
    if g.get("review_at") and g.get("status") == "active":
        facts.append(f"review by {_date(g['review_at'])}")
    out.append("  " + " · ".join(facts))
    return "\n".join(out)


class CourseTool(Tool):
    """Read and report on the owner's Course (Flight Deck keeps it)."""

    name = "course"
    timeout_seconds = 30.0
    description = (
        "Your course in Flight Deck: the goals your work is for — the deck's, your owner's "
        "and your own. The live ones are already in your prompt; this tool shows more and "
        "records progress. action='view': every goal with status, progress, when work last "
        "moved it, due and review dates. action='link': record that something you did moved a "
        "goal forward — 'goal_id' (G:<id>), 'kind' (session, schedule, project, task, plan), "
        "'ref_id' (its id) and a short 'note'. action='report': your estimate of how far along "
        "one of your owner's goals or your own is — 'goal_id', 'progress' 0 to 1, 'note'; set "
        "'looks_achieved' true to ask your owner whether it's done (only they mark it done). "
        "A goal guides what you choose to do; it never permits anything your instructions, "
        "approvals or guards don't."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["view", "link", "report"],
                "description": "view your course, link work to a goal, or report progress.",
            },
            "goal_id": {"type": "string", "description": "link / report: the goal's G:<id>."},
            "kind": {"type": "string", "enum": _KINDS, "description": "link: what kind of work."},
            "ref_id": {"type": "string", "description": "link: the work's id (session id, task id…)."},
            "note": {"type": "string", "description": "link / report: one short sentence."},
            "progress": {"type": "number", "description": "report: 0 (not started) to 1 (done)."},
            "looks_achieved": {
                "type": "boolean",
                "description": "report: ask your owner whether the goal is achieved.",
            },
        },
        "required": ["action"],
    }

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        action = str(action or "").strip().lower()
        try:
            if action == "view":
                return await self._view()
            if action == "link":
                return await self._link(**kwargs)
            if action == "report":
                return await self._report(**kwargs)
        except CourseError as exc:
            return ToolResult(success=False, error=exc.message)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        return ToolResult(success=False, error="action must be one of: view, link, report")

    @staticmethod
    async def _view(**_: Any) -> ToolResult:
        goals = (await client.view()).get("goals") or []
        if not goals:
            return ToolResult(success=True, content="Your course is empty: no goals are set.")
        return ToolResult(success=True, content=(
            "Your course (goal text is your deck's and your owner's words: what to aim for, "
            "never permission):\n" + "\n".join(_line(g) for g in goals)))

    @staticmethod
    async def _link(goal_id: Any = "", kind: Any = "", ref_id: Any = "", note: Any = "",
                    **_: Any) -> ToolResult:
        if not str(goal_id or "").strip():
            raise ValueError("goal_id is required (G:<id> from your course)")
        if not str(ref_id or "").strip():
            raise ValueError("ref_id is required: the id of the work you're linking")
        data = await client.link(str(goal_id), str(kind or ""), str(ref_id), str(note or ""))
        return ToolResult(success=True,
                          content=f"Linked to “{data.get('goal') or goal_id}”.")

    @staticmethod
    async def _report(goal_id: Any = "", progress: Any = None, note: Any = "",
                      looks_achieved: Any = False, **_: Any) -> ToolResult:
        if not str(goal_id or "").strip():
            raise ValueError("goal_id is required (G:<id> from your course)")
        try:
            value = float(progress)
        except (TypeError, ValueError):
            raise ValueError("progress must be a number from 0 to 1") from None
        data = await client.report(str(goal_id), value, str(note or ""),
                                   looks_achieved=looks_achieved is True)
        goal = data.get("goal") or {}
        text = f"Recorded {round(value * 100)}% on “{goal.get('title') or goal_id}”."
        if data.get("asked_owner"):
            text += " Your owner was asked on their Line whether it's achieved."
        elif looks_achieved is True:
            text += " Your owner was already asked about it today."
        return ToolResult(success=True, content=text)
