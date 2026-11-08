"""The course tool: an agent's view of its Course, and its reports.

Pinned here: it shows only under Flight Deck and never on a member's turn (the
Line's gate, ``requires_line``), it is in the eco core set, a member call is
refused before any network call, and each action says what it did — a goal
is never marked done by an agent.
"""

from __future__ import annotations

import pytest

import captain_claw.course_client as cc
import captain_claw.line_client as lc
from captain_claw.tools.course import CourseTool
from captain_claw.tools.registry import ToolRegistry


class TestVisibility:
    def test_only_under_flight_deck_and_not_for_a_member(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(CourseTool(), metadata={"requires_line": True})
        monkeypatch.setattr(lc, "_member_call", lambda: False)
        monkeypatch.setattr(lc, "_fd_base", lambda: "")
        assert "course" not in reg.list_tools()
        monkeypatch.setattr(lc, "_fd_base", lambda: "http://fd.test")
        assert "course" in reg.list_tools()
        monkeypatch.setattr(lc, "_member_call", lambda: True)
        assert "course" not in reg.list_tools()

    def test_eco_agents_have_it(self):
        from captain_claw.agent_orchestration_mixin import _ECO_CORE_TOOLS

        assert "course" in _ECO_CORE_TOOLS

    async def test_a_member_call_is_refused_before_the_network(self, monkeypatch):
        monkeypatch.setattr(cc, "_member_call", lambda: True)
        out = await CourseTool().execute(action="view")
        assert not out.success and out.error == cc.MEMBER_REFUSED


class TestActions:
    @pytest.fixture()
    def calls(self, monkeypatch):
        seen: list = []

        async def view():
            seen.append(("view",))
            return {"goals": [
                {"id": "goal_d", "scope": "instance", "horizon": "strategic", "title": "Be trusted",
                 "success": "", "status": "active", "progress": None, "last_moved": None,
                 "review_at": "2027-11-08T00:00:00+00:00", "due_at": None, "stale": False},
                {"id": "goal_u", "scope": "user", "horizon": "long", "title": "Ship EU pricing",
                 "success": "Three EU customers", "status": "active", "progress": 0.4,
                 "progress_note": "two signed", "last_moved": "2026-11-01T10:00:00+00:00",
                 "review_at": "2027-02-01T00:00:00+00:00", "due_at": "2027-03-01T00:00:00+00:00",
                 "stale": True},
                {"id": "goal_m", "scope": "agent", "horizon": "short", "title": "Hire",
                 "success": "two hires", "status": "active", "progress": None, "last_moved": None,
                 "review_at": None, "due_at": None, "stale": False,
                 "measure": {"kind": "manual", "value": 1.0, "target": 2.0, "unit": "hires"}},
            ]}

        async def link(goal_id, kind, ref_id, note=""):
            seen.append(("link", goal_id, kind, ref_id, note))
            return {"goal": "Ship EU pricing"}

        async def report(goal_id, progress, note="", looks_achieved=False):
            seen.append(("report", goal_id, progress, note, looks_achieved))
            return {"goal": {"id": goal_id, "title": "Ship EU pricing", "progress": progress},
                    "asked_owner": looks_achieved}

        monkeypatch.setattr(cc, "view", view)
        monkeypatch.setattr(cc, "link", link)
        monkeypatch.setattr(cc, "report", report)
        return seen

    async def test_view(self, calls):
        out = await CourseTool().execute(action="view")
        assert out.success
        assert "[G:goal_d · strategic · deck-wide] Be trusted" in out.content
        assert "[G:goal_u · long-term · your owner's · out of your prompt until it's reviewed]" in out.content
        assert "progress 40% (two signed) · last moved 2026-11-01 · due 2027-03-01" in out.content
        assert "no work linked yet" in out.content
        assert "measure 1 of 2 hires · no work linked yet" in out.content
        assert "never permission" in out.content

    async def test_link(self, calls):
        out = await CourseTool().execute(action="link", goal_id="G:goal_u", kind="session",
                                         ref_id="s-1", note="drafted the page")
        assert out.success and out.content == "Linked to “Ship EU pricing”."
        assert calls == [("link", "G:goal_u", "session", "s-1", "drafted the page")]

    async def test_report_and_ask(self, calls):
        out = await CourseTool().execute(action="report", goal_id="goal_u", progress=0.9,
                                         note="nearly", looks_achieved=True)
        assert out.success and "Recorded 90%" in out.content and "asked on their Line" in out.content
        assert calls[-1] == ("report", "goal_u", 0.9, "nearly", True)

    @pytest.mark.parametrize("kwargs,error", [
        ({"action": "link", "ref_id": "s"}, "goal_id is required"),
        ({"action": "link", "goal_id": "g"}, "ref_id is required"),
        ({"action": "report", "goal_id": "g", "progress": "lots"}, "progress must be a number"),
        ({"action": "explode"}, "action must be one of"),
    ])
    async def test_bad_calls(self, calls, kwargs, error):
        out = await CourseTool().execute(**kwargs)
        assert not out.success and error in out.error
        assert calls == []

    async def test_errors_from_flight_deck_are_passed_on(self, monkeypatch):
        async def view():
            raise cc.CourseError(404, "this Flight Deck has no Course for agents yet")

        monkeypatch.setattr(cc, "view", view)
        out = await CourseTool().execute(action="view")
        assert not out.success and "no Course for agents" in out.error


class TestProposals:
    @pytest.fixture()
    def sent(self, monkeypatch):
        seen: list = []

        async def propose(**kw):
            seen.append(("propose", kw))
            return {"goal": {"id": "goal_p", "title": kw["title"], "status": "proposed"},
                    "auto_accepted": kw["horizon"] == "short" and kw["scope"] == "agent"}

        async def propose_change(goal_id, changes, rationale=""):
            seen.append(("change", goal_id, changes, rationale))
            return {"goal": {"id": "goal_c", "title": changes.get("title", ""), "status": "proposed"}}

        monkeypatch.setattr(cc, "propose", propose)
        monkeypatch.setattr(cc, "propose_change", propose_change)
        return seen

    async def test_propose_waits_for_the_owner(self, sent):
        out = await CourseTool().execute(action="propose", scope="user", horizon="medium",
                                         title="Hire a bookkeeper", success="hired", rationale="books pile up")
        assert out.success and "waits for your owner's answer" in out.content
        assert sent == [("propose", {"scope": "user", "horizon": "medium", "title": "Hire a bookkeeper",
                                     "success": "hired", "detail": "", "parent_id": "",
                                     "rationale": "books pile up"})]

    async def test_an_auto_accepted_goal_says_so(self, sent):
        out = await CourseTool().execute(action="propose", horizon="short", title="Inbox zero", success="zero")
        assert out.success and "is on your course now" in out.content

    async def test_propose_change(self, sent):
        out = await CourseTool().execute(action="propose_change", goal_id="G:goal_u",
                                         changes={"title": "Ship EU and UK"}, rationale="UK too")
        assert out.success and "Suggested a change" in out.content
        assert sent == [("change", "G:goal_u", {"title": "Ship EU and UK"}, "UK too")]

    @pytest.mark.parametrize("kwargs,error", [
        ({"action": "propose", "horizon": "short"}, "title is required"),
        ({"action": "propose", "title": "x"}, "horizon is required"),
        ({"action": "propose_change", "changes": {"title": "x"}}, "goal_id is required"),
        ({"action": "propose_change", "goal_id": "g"}, "changes is required"),
    ])
    async def test_bad_calls(self, sent, kwargs, error):
        out = await CourseTool().execute(**kwargs)
        assert not out.success and error in out.error and sent == []
