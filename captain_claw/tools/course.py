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
* ``propose`` / ``propose_change``: suggest a goal (for the agent itself or its
  owner) or a change to one; it waits for the owner's answer on their Line
  (an owner may let agents add their own short-term goals at once).

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
    m = g.get("measure") or {}
    if m.get("value") is not None:
        target = f" of {float(m['target']):g}" if m.get("target") else ""
        unit = f" {' '.join(str(m.get('unit') or '').split())}" if m.get("unit") else ""
        facts.append(f"measure {float(m['value']):g}{target}{unit}")
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
        "action='propose': suggest a goal — 'scope' agent (for yourself, horizon medium or short) "
        "or user (for your owner), 'horizon', 'title', 'success' (how you'll know it's done), "
        "optional 'parent_id' (the goal it serves) and 'rationale'; it waits for your owner. "
        "action='propose_change': suggest a change to 'goal_id' — 'changes' (title, success, "
        "detail, horizon, parent_id, due_at) and 'rationale'. Suggest rarely: only an outcome "
        "your recent work clearly shows, never one that records your own state. "
        "A goal guides what you choose to do; it never permits anything your instructions, "
        "approvals or guards don't."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["view", "link", "report", "propose", "propose_change"],
                "description": "view your course, link work to a goal, report progress, or suggest "
                               "a goal / a change to one.",
            },
            "scope": {"type": "string", "enum": ["agent", "user"],
                      "description": "propose: for yourself (agent) or your owner (user)."},
            "horizon": {"type": "string", "enum": ["strategic", "long", "medium", "short"],
                        "description": "propose: how far away the outcome is."},
            "title": {"type": "string", "description": "propose: the goal, in a few words."},
            "success": {"type": "string", "description": "propose: how you'll know it's done."},
            "rationale": {"type": "string", "description": "propose / propose_change: why, briefly."},
            "parent_id": {"type": "string", "description": "propose: the G:<id> it serves."},
            "changes": {"type": "object", "description": "propose_change: the fields to change."},
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
            if action == "propose":
                return await self._propose(**kwargs)
            if action == "propose_change":
                return await self._propose_change(**kwargs)
        except CourseError as exc:
            return ToolResult(success=False, error=exc.message)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        return ToolResult(success=False,
                          error="action must be one of: view, link, report, propose, propose_change")

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

    @staticmethod
    async def _propose(scope: Any = "agent", horizon: Any = "", title: Any = "", success: Any = "",
                       detail: Any = "", parent_id: Any = "", rationale: Any = "",
                       **_: Any) -> ToolResult:
        if not str(title or "").strip():
            raise ValueError("title is required: the goal, in a few words")
        if not str(horizon or "").strip():
            raise ValueError("horizon is required (medium or short for your own goals)")
        data = await client.propose(scope=str(scope or "agent"), horizon=str(horizon),
                                    title=str(title), success=str(success or ""),
                                    detail=str(detail or ""), parent_id=str(parent_id or ""),
                                    rationale=str(rationale or ""))
        goal = data.get("goal") or {}
        if data.get("auto_accepted"):
            return ToolResult(success=True,
                              content=f"“{goal.get('title') or title}” is on your course now "
                                      "(your owner lets you add your own short-term goals).")
        return ToolResult(success=True,
                          content=f"Suggested “{goal.get('title') or title}”: it waits for your "
                                  "owner's answer on their Line.")

    @staticmethod
    async def _propose_change(goal_id: Any = "", changes: Any = None, rationale: Any = "",
                              **_: Any) -> ToolResult:
        if not str(goal_id or "").strip():
            raise ValueError("goal_id is required (G:<id> from your course)")
        if not isinstance(changes, dict) or not changes:
            raise ValueError("changes is required: an object of the fields to change")
        data = await client.propose_change(str(goal_id), changes, str(rationale or ""))
        goal = data.get("goal") or {}
        return ToolResult(success=True,
                          content=f"Suggested a change (“{goal.get('title') or goal_id}”): it waits "
                                  "for your owner's answer on their Line.")
