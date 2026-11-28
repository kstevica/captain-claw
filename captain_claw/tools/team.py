"""The `team` tool — Team up: split a task among helpers, copies of this agent.

See ``captain_claw/team_up.py`` and docs/team-up-plan.md. The lead proposes a
team; the person taps Team up (or it starts at once when the agent's setting
is "On"); the helpers work in the background while the person keeps talking to
the lead; the lead checks on them, tells them things and stops them; when they
are done the lead gets their results and writes the reply.

Inside a helper the tool is the team board (team_board.py): status, map, read,
search, post, ask, answer, inbox and wait. The agent itself can read the board too.
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
        "As the agent: 'propose' (`title`, `parts`: [{name, goal, produces}]) — the user usually taps "
        "Team up on a card; end your turn with one short line and don't do the work too. 'start' — "
        "only when the user's message says go ahead. 'status' (`helper` for one in detail), 'tell' "
        "(`helper` or 'all', `text`: shown at its next step), 'stop' (`helper`, or all). When they "
        "finish you get their results and write the user one reply.\n"
        "As a helper — the team board: 'post' (`text`, `title`; start the text with 'remember:' for "
        "a note to memory), 'ask' (`to`: a helper's name, 'anyone' or 'lead'; `text` — doesn't block), "
        "'answer' (`ask_id`, `text`), 'inbox' (answers to your asks; `wait` ≤30 s), 'wait' (`helper` "
        "to finish, or `query` for a post; `wait` ≤120 s). Both: 'read' (`helper`, `kind`), 'search' "
        "(`query`), 'map' (who covers what). Helpers never send, publish, pay or change memory.\n"
        "Later, to answer what an earlier team found ('what did Prices find?'): 'recall' — no arguments "
        "for the teams in this conversation, `helper` for one helper's full result, `query` to search them."
    )
    # `wait` may block up to two minutes.
    timeout_seconds = 150.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["propose", "start", "status", "tell", "stop",
                         "post", "ask", "answer", "inbox", "wait", "read", "search", "map", "recall"],
                "description": "What to do (see the tool description).",
            },
            "title": {"type": "string", "description": "propose: the whole task in a few words. post: a short label."},
            "parts": {
                "type": "array",
                "description": "propose: 2–4 independent parts, one per helper.",
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
            "helper": {"type": "string", "description": "status/tell/stop/read/wait/recall: a helper's name ('all' for tell/stop)."},
            "text": {"type": "string", "description": "tell/post/ask/answer: the message."},
            "to": {"type": "string", "description": "ask: a helper's name, 'anyone' or 'lead'."},
            "ask_id": {"type": "integer", "description": "answer: the ask's number (#)."},
            "query": {"type": "string", "description": "search/recall, or wait: words of the post you need."},
            "kind": {"type": "string", "description": "read: post, ask, answer or remember."},
            "wait": {"type": "integer", "description": "inbox (≤30) / wait (≤120): seconds to block."},
        },
        "required": ["action"],
    }

    async def execute(self, action: str = "status", title: str = "", parts: Any = None,
                      helper: str = "", text: str = "", to: str = "", ask_id: Any = None,
                      query: str = "", kind: str = "", wait: Any = 0, **kwargs: Any) -> ToolResult:
        agent = kwargs.get("_agent")
        act = str(action or "status").strip().lower()
        found = team_up.helper_of(agent)
        if found is not None:
            return await self._as_helper(found, act, helper=helper, text=text, title=title, to=to,
                                         ask_id=ask_id, query=query, kind=kind, wait=wait)

        if act == "propose":
            team, error = await team_up.propose(agent, title, parts)
            if team is None:
                return ToolResult(success=False, error=error)
            names = ", ".join(h.name for h in team.helpers)
            where = team_up.CHANNEL_NAMES.get(team.channel, "")
            if team.state == "proposed" and where:
                return ToolResult(content=(
                    f"Team offered ({team.id}): {names}. On {where} there is no card to tap: end your turn "
                    "with one short line asking whether to team up, naming the helpers. If they say yes, "
                    "call team start; if not, do the task yourself."))
            if team.state == "running" and where:
                return ToolResult(content=(
                    f"Team started ({team.id}): {names}. End your turn with one short line saying they're on "
                    f"it and the user can keep talking to you. Their results come back to you here on {where}."))
            if team.state == "running":
                return ToolResult(content=(
                    f"Team started ({team.id}): {names}. They work in the background and the user sees "
                    "their progress. End your turn now with one short line saying the helpers are on it "
                    "and the user can keep talking to you. You'll get their results when they finish."))
            return ToolResult(content=(
                f"Team offered ({team.id}): {names}. The user sees a card with Team up / Do it alone. "
                "End your turn now with one short line (don't repeat the parts, don't start the work). "
                "If they tap Team up the helpers start; if they say go ahead in a message, call team start."))

        if act == "recall":
            return ToolResult(content=await team_up.recall_text(agent, helper, query or text))
        team = team_up.active_for(agent) or self._latest(agent)
        if team is None:
            # No team in memory (none yet, or the agent restarted): the records of earlier ones.
            if act in ("status", "read", "search", "map") and team_up.records_of(getattr(agent, "session", None)):
                return ToolResult(content=await team_up.recall_text(agent, helper, query or text))
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
        if act == "read":
            return ToolResult(content=team_up.board_read(team, helper, kind))
        if act == "search":
            return ToolResult(content=team_up.board_search(team, query or text))
        if act == "map":
            return ToolResult(content=team_up.board_map(team))
        if act in ("post", "ask", "answer", "inbox", "wait"):
            return ToolResult(success=False, error=(
                f"'{act}' is for your helpers. To steer one, use team tell; their asks to you are "
                "answered automatically and show in Team now."))
        return ToolResult(success=False, error=f"Unknown action {action!r}.")

    @staticmethod
    async def _as_helper(found: Any, act: str, *, helper: str, text: str, title: str, to: str,
                         ask_id: Any, query: str, kind: str, wait: Any) -> ToolResult:
        team, me = found

        def _done(msg: str, ok_prefixes: tuple[str, ...]) -> ToolResult:
            if msg.startswith(ok_prefixes):
                return ToolResult(content=msg)
            return ToolResult(success=False, error=msg)

        if act == "status":
            return ToolResult(content=team_up.status_text(team, helper, for_helper=me))
        if act == "map":
            return ToolResult(content=team_up.board_map(team, for_helper=me))
        if act == "read":
            return ToolResult(content=team_up.board_read(team, helper, kind))
        if act == "search":
            if not str(query or text).strip():
                return ToolResult(success=False, error="Give a query.")
            return ToolResult(content=team_up.board_search(team, query or text, for_helper=me))
        if act == "post":
            return _done(team_up.board_post(team, me, text, title), ("Posted", "Noted"))
        if act == "ask":
            return _done(await team_up.board_ask(team, me, to or helper, text), ("Asked", "You already asked"))
        if act == "answer":
            return _done(team_up.board_answer(team, me, ask_id, text), ("Answered",))
        if act == "inbox":
            return ToolResult(content=await team_up.board_inbox(team, me, wait))
        if act == "wait":
            return ToolResult(content=await team_up.board_wait(team, me, helper, query, wait))
        return ToolResult(success=False, error=(
            "Only the lead starts, steers or stops the team. Helpers use the board: post, ask, answer, "
            "inbox, wait, read, search, map, status."))

    @staticmethod
    def _latest(agent: Any) -> Any:
        """The agent's most recent team on its lane (finished ones too, for status)."""
        lane = team_up.lane_of(agent)
        mine = [t for t in team_up.teams_for_lane(lane) if t.lead is agent] if lane else []
        return mine[-1] if mine else None
