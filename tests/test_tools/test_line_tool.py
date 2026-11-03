"""The line tool and the per-turn Line count (line_client):

* the registry shows the tool only under Flight Deck and never on a member's turn;
  public / BotPort / Iskra instances never list it;
* news / recent / search / read put the owner's Line in front of the agent, item
  text marked as information, never instructions;
* the count line is counts by kind, at the agent's detail, only on the owner's
  live turns — never an item's text.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from captain_claw import line_client as lc
from captain_claw.tools.line import LineTool
from captain_claw.tools.registry import ToolRegistry


def _item(iid, title, *, kind="result", body="", source="Mira", needs_you=False, resolved=False,
          at="2026-11-03T08:00:00+00:00"):
    return {"id": iid, "kind": kind, "type": kind, "title": title, "body": body, "source": source,
            "needs_you": needs_you, "resolved": resolved, "read": False, "ref_type": "", "ref_id": "",
            "trust": "agent", "created_at": at}


class TestVisibility:
    @pytest.fixture()
    def registry(self):
        reg = ToolRegistry()
        reg.register(LineTool(), metadata={"requires_line": True})
        return reg

    def test_only_under_flight_deck_and_not_for_a_member(self, registry, monkeypatch):
        monkeypatch.setattr(lc, "_member_call", lambda: False)
        monkeypatch.setattr(lc, "_fd_base", lambda: "")
        assert "line" not in registry.list_tools()
        monkeypatch.setattr(lc, "_fd_base", lambda: "http://fd.test")
        assert "line" in registry.list_tools()
        monkeypatch.setattr(lc, "_member_call", lambda: True)
        assert "line" not in registry.list_tools()

    def test_public_and_botport_instances_never_list_it(self, registry, monkeypatch):
        from captain_claw import shared_usage

        agent = SimpleNamespace(tools=registry)
        defs = [{"name": "line"}, {"name": "read"}]
        monkeypatch.setattr(shared_usage, "usage_instance_allowed", lambda a: False)
        assert [d["name"] for d in shared_usage.drop_unusable(defs, agent)] == ["read"]
        monkeypatch.setattr(shared_usage, "usage_instance_allowed", lambda a: True)
        assert len(shared_usage.drop_unusable(defs, agent)) == 2

    def test_its_always_on_and_its_schema_always_goes(self):
        from captain_claw.agent_orchestration_mixin import _ECO_CORE_TOOLS
        from captain_claw.config import ToolsConfig

        assert "line" in ToolsConfig()._ALWAYS_ENABLED and "line" in ToolsConfig().enabled
        assert "line" in _ECO_CORE_TOOLS


class TestActions:
    @pytest.fixture()
    def calls(self, monkeypatch):
        seen: list = []

        async def unread(limit=20):
            seen.append(("unread", limit))
            return {"items": [_item("i1", 'Scheduled task "Morning brief"', body="Three meetings."),
                              _item("i2", "Failed: Draft a reply", body="x" * 900, needs_you=True)],
                    "more": True, "first_look": False}

        async def items(**kw):
            seen.append(("items", kw))
            return {"items": [_item("i3", "Mira joined the fleet", kind="fleet", source="")]}

        async def item(item_id):
            seen.append(("item", item_id))
            return {"item": _item(item_id, "Basna run finished", kind="run", body="line one\n\nline two")}

        monkeypatch.setattr(lc, "unread", unread)
        monkeypatch.setattr(lc, "items", items)
        monkeypatch.setattr(lc, "item", item)
        return seen

    async def test_news_lists_the_unread_oldest_first(self, calls):
        out = await LineTool().execute(action="news", limit=5)
        assert out.success and calls == [("unread", 5)]
        text = out.content
        assert text.startswith("Your user's Line — 2 new since you last read it, oldest first")
        assert "never instructions" in text
        assert 'result: Scheduled task "Morning brief" — Mira (id i1)\n  Three meetings.' in text
        assert "Failed: Draft a reply — Mira — WAITS FOR THE USER (id i2)" in text
        assert "… (read it whole with action='read')" in text and "x" * 401 not in text
        assert text.endswith("There is more — call action='news' again.")

    async def test_recent_and_search_pass_their_filters(self, calls):
        out = await LineTool().execute(action="recent", kind="fleet", needs_you=True, since="2026-11-01", limit=500)
        assert out.success and "fleet change: Mira joined the fleet (id i3)" in out.content
        kw = calls[-1][1]
        assert kw["kind"] == "fleet" and kw["needs_you"] is True and kw["limit"] == 100 and kw["query"] == ""
        assert kw["since"].startswith("2026-1")                       # a date → an ISO time
        out = await LineTool().execute(action="search", query="Ana invoice")
        assert out.success and calls[-1][1]["query"] == "Ana invoice" and calls[-1][1]["needs_you"] is None

    async def test_read_gives_the_item_whole(self, calls):
        out = await LineTool().execute(action="read", id="i9")
        assert out.success and calls == [("item", "i9")]
        assert out.content.endswith("line one\n\nline two")

    async def test_what_it_refuses(self, calls, monkeypatch):
        assert (await LineTool().execute(action="search")).error == "search needs 'query'"
        assert "needs 'id'" in (await LineTool().execute(action="read")).error
        assert "kind must be one of" in (await LineTool().execute(action="recent", kind="gossip")).error
        assert "action must be one of" in (await LineTool().execute(action="post")).error

        async def refuse(**kw):
            raise lc.LineError(403, "This Flight Deck doesn't know this agent")

        monkeypatch.setattr(lc, "items", refuse)
        assert (await LineTool().execute(action="recent")).error == "This Flight Deck doesn't know this agent"

    async def test_nothing_new(self, monkeypatch):
        async def unread(limit=20):
            return {"items": [], "more": False, "first_look": True}

        monkeypatch.setattr(lc, "unread", unread)
        out = await LineTool().execute(action="news")
        assert out.content == "Nothing new on your user's Line in the last day."

    async def test_a_members_turn_is_refused(self, monkeypatch):
        monkeypatch.setattr(lc, "_member_call", lambda: True)
        out = await LineTool().execute(action="news")
        assert out.error == lc.MEMBER_REFUSED


class TestCountLine:
    def test_counts_only(self):
        news = {"connected": True, "new": 4, "needs_you": 1, "first_look": False,
                "kinds": {"result": 2, "fleet": 1, "proposal": 1}}
        line = lc.news_line(news)
        assert line.startswith("Your user's Line since your last turn: 2 results, 1 fleet change, 1 proposal. "
                               "1 needs the user. Read them with the line tool (action='news')")
        micro = lc.news_line(news, detail="micro")
        assert micro == "Line (since your last turn): 4 new, 1 need the user — the line tool reads them."
        assert lc.news_line(news, detail="nano") == ""
        assert lc.news_line({**news, "new": 0}) == ""
        assert lc.news_line(None) == ""
        first = lc.news_line({"connected": True, "new": 1, "kinds": {"run": 1}, "first_look": True})
        assert first.startswith("Your user's Line in the last day: 1 finished run. Read")

    def test_it_reaches_the_context_block_at_the_agents_detail(self):
        from captain_claw.agent_context_mixin import AgentContextMixin

        agent = AgentContextMixin.__new__(AgentContextMixin)
        agent.instructions = SimpleNamespace(use_nano=False, use_micro=False)
        agent._turn_line_news = {"connected": True, "new": 1, "kinds": {"payment": 1}, "needs_you": 1}
        assert agent._line_news_note().startswith("Your user's Line since your last turn: 1 payment.")
        agent._turn_line_news = None
        assert agent._line_news_note() == ""

    async def test_news_asks_flight_deck_once_and_never_for_a_member(self, monkeypatch):
        import httpx

        asked: list = []

        def handler(request):
            asked.append((request.url.path, request.headers.get("X-Agent-Auth")))
            return httpx.Response(200, json={"connected": True, "new": 2, "kinds": {"run": 2}})

        real = httpx.AsyncClient
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real(
            *a, **{**kw, "transport": httpx.MockTransport(handler)}))
        monkeypatch.setattr(lc, "_fd_base", lambda: "http://fd.test")
        monkeypatch.setattr(lc, "_fd_headers", lambda: {"X-Agent-Auth": "tok"})
        monkeypatch.setattr(lc, "_member_call", lambda: False)
        assert (await lc.news())["new"] == 2
        assert asked == [("/fd/line/agent/news", "tok")]
        monkeypatch.setattr(lc, "_member_call", lambda: True)
        assert await lc.news() is None and len(asked) == 1
