"""The `team` tool — Team up: split a task among helpers, copies of this agent.

See ``captain_claw/team_up.py`` and docs/team-up-plan.md. The lead proposes a
team; the person taps Team up (or it starts at once when the agent's setting
is "On"); the helpers work in the background while the person keeps talking to
the lead; the lead checks on them, tells them things and stops them; when they
are done the lead gets their results and writes the reply.

Inside a helper only ``status`` works (it shows what the other helpers do).
"""

from __future__ import annotations

from typing import Any

from captain_claw import team_up
from captain_claw.tools.registry import Tool, ToolResult


class TeamTool(Tool):
    name = "team"
    description = (
        "Team up: split a big task among 2–4 helpers — copies of you with your memory, tools and "
        "model — that work on their parts at the same time, in the background, while the user can "
        "keep talking to you. Propose a team ONLY when the request has 2+ independent, substantial "
        "parts (research several companies or options, go through a large folder in parts, gather "
        "data for different sections). Never for a short answer, for steps that depend on each "
        "other, or for work on one account (one inbox).\n"
        "'propose' — offer the team (`title`, `parts`: [{name, goal, produces}]). The user usually "
        "taps Team up on a card; then end your turn with one short line — don't do the work too.\n"
        "'start' — start the team you offered, only when the user's message says go ahead.\n"
        "'status' — what every helper is doing (`helper` for one in detail, with its result).\n"
        "'tell' — a message to a helper (`helper`, or 'all') shown at its next step (`text`).\n"
        "'stop' — stop one helper (`helper`) or the whole team.\n"
        "Helpers can't send, publish, pay or change memory: when they finish you get their results "
        "and write the user one reply, and you do any such action once."
    )
    timeout_seconds = 60.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["propose", "start", "status", "tell", "stop"],
                "description": "propose | start | status | tell | stop",
            },
            "title": {"type": "string", "description": "For propose: the whole task in a few words."},
            "parts": {
                "type": "array",
                "description": "For propose: 2–4 independent parts, one per helper.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "A short name for the part, e.g. 'Prices'."},
                        "goal": {"type": "string", "description": "What this helper does, self-contained."},
                        "produces": {"type": "string", "description": "What it hands back."},
                    },
                    "required": ["name", "goal"],
                },
            },
            "helper": {"type": "string", "description": "For status/tell/stop: a helper's name ('all' = every one)."},
            "text": {"type": "string", "description": "For tell: the message."},
        },
        "required": ["action"],
    }

    async def execute(self, action: str = "status", title: str = "", parts: Any = None,
                      helper: str = "", text: str = "", **kwargs: Any) -> ToolResult:
        agent = kwargs.get("_agent")
        act = str(action or "status").strip().lower()
        found = team_up.helper_of(agent)
        if found is not None:
            team, me = found
            if act != "status":
                return ToolResult(success=False, error=(
                    "Only the lead starts, steers or stops the team. Use team status to see the others."))
            return ToolResult(content=team_up.status_text(team, helper, for_helper=me))

        if act == "propose":
            team, error = await team_up.propose(agent, title, parts)
            if team is None:
                return ToolResult(success=False, error=error)
            names = ", ".join(h.name for h in team.helpers)
            if team.state == "running":
                return ToolResult(content=(
                    f"Team started ({team.id}): {names}. They work in the background and the user sees "
                    "their progress. End your turn now with one short line saying the helpers are on it "
                    "and the user can keep talking to you. You'll get their results when they finish."))
            return ToolResult(content=(
                f"Team offered ({team.id}): {names}. The user sees a card with Team up / Do it alone. "
                "End your turn now with one short line (don't repeat the parts, don't start the work). "
                "If they tap Team up the helpers start; if they say go ahead in a message, call team start."))

        team = team_up.active_for(agent) or self._latest(agent)
        if team is None:
            return ToolResult(success=False, error="You have no team. Use team propose to offer one.")

        if act == "start":
            if team.state != "proposed":
                return ToolResult(success=False, error=f"Your team is {team.state}, not waiting to start.")
            why = team_up.why_unavailable(agent)
            if why and "already working" not in why:
                return ToolResult(success=False, error=why)
            error = await team_up.start(team, by="agent")
            if error:
                return ToolResult(success=False, error=error)
            return ToolResult(content=(
                f"Team started: {', '.join(h.name for h in team.helpers)}. End your turn with one short "
                "line; you'll get their results when they finish."))
        if act == "status":
            return ToolResult(content=team_up.status_text(team, helper))
        if act == "tell":
            error = team_up.tell(team, helper, text)
            if error:
                return ToolResult(success=False, error=error)
            who = "every working helper" if str(helper or "").strip().lower() in ("", "all") else helper
            return ToolResult(content=f"Told {who}; it sees the message at its next step.")
        if act == "stop":
            error = team_up.stop(team, helper)
            if error:
                return ToolResult(success=False, error=error)
            if str(helper or "").strip().lower() in ("", "all"):
                return ToolResult(content="Stopping the whole team. Nothing more comes from it.")
            return ToolResult(content=(
                f"Stopping {helper}. The others go on; you get the results when they finish."))
        return ToolResult(success=False, error=f"Unknown action {action!r} — propose, start, status, tell or stop.")

    @staticmethod
    def _latest(agent: Any) -> Any:
        """The agent's most recent team on its lane (finished ones too, for status)."""
        lane = team_up.lane_of(agent)
        mine = [t for t in team_up.teams_for_lane(lane) if t.lead is agent] if lane else []
        return mine[-1] if mine else None
