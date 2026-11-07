# ruff: noqa: F811 — `data`, `deck` and `as_agent` are fixtures imported from the other Custom tools tests
"""Custom tools, Phase 7b: a tool runs another of its owner's tools, through the
agent that ran it, as the same person. Only what it declared; no cycles; at
most three deep; inside the outer run's time; people need the tools it runs."""

from __future__ import annotations

import json
import sys
import time

import pytest

from captain_claw import custom_tool_callbacks, custom_tools_client
from captain_claw.flight_deck import external_tools as xt
from captain_claw.flight_deck import external_tools_routes as xr
from captain_claw.tools.custom_tools import CustomToolsTool
from captain_claw.tools.registry import ToolResult

from .test_external_tools import _commit, _manifest, _switch_on, data, deck  # noqa: F401
from .test_external_tools_agent import _agent_client, as_agent  # noqa: F401

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="the line to the agent is POSIX only")

TOTAL = """import json, os, sys
d = json.load(sys.stdin)
total = sum(d.get("numbers", []))
open(os.path.join(os.environ["XT_OUT"], "total.txt"), "w").write(str(total))
print(json.dumps({"total": total}))
"""
REPORT = """import json, sys, captain
d = json.load(sys.stdin)
try:
    out = captain.custom_tool(d.get("other", "total"), {"numbers": [1, 2, 3]})
    files = list(captain.last_files)
    print(json.dumps({"total": out["total"], "files": len(files), "text": open(files[0]).read() if files else ""}))
except captain.CaptainError as exc:
    print(json.dumps({"error": str(exc)}))
"""
RUNS_TOTAL = [{"custom_tool": "total", "why": "adds the numbers"}]
RECORDED_TOTAL = [{"input": {}, "expect": {"total": 6},
                   "calls": [{"custom_tool": "total", "reply": {"ok": True, "output": {"total": 6}}}]}]


async def _publish(uid: str, slug: str, main_py: str, **manifest) -> None:
    await xt.create_tool(uid, slug, title=slug.title(), description="A test tool.")
    _commit(uid, slug, {"main.py": main_py, "tool.json": _manifest(slug, **manifest), "tests/test_main.py": None})
    state = await xt.publish(uid, slug, by=uid)
    assert state["state"] == "passed", state["steps"]


class Agent:
    """The agent that ran the tool: runs what it's asked through its own custom_tools tool."""

    async def _execute_tool_with_guard(self, name, arguments, interaction_label="", session_id_override=None, **_):
        if name == "custom_tools":
            return await CustomToolsTool().execute(**arguments, _agent=self, _session_id=session_id_override or "")
        return ToolResult(success=True, content=f"{name} ran")


@pytest.fixture
def wired(monkeypatch):
    """Flight Deck's relay reaches the agent's handler, and the agent's
    custom_tools tool reaches Flight Deck's agent routes, all in-process."""
    custom_tool_callbacks.reset_for_tests()

    async def post(self, body, timeout):
        status, answer = await custom_tool_callbacks.handle(body)
        if status != 200:
            raise xt.ToolError("agent_refused", answer.get("error", ""))
        return answer

    async def call(method, path, body=None):
        async with _agent_client() as c:
            resp = await c.request(method, f"/fd/external-tools/agent/{path}", json=body)
        return resp.status_code, resp.json()

    monkeypatch.setattr(xr.AgentLink, "_post", post)
    monkeypatch.setattr(custom_tools_client, "_fd_base", lambda: "http://deck.test")
    monkeypatch.setattr(custom_tools_client, "call", call)
    yield
    custom_tool_callbacks.reset_for_tests()


async def _run(slug: str, data_in: dict | None = None) -> dict:
    result = await CustomToolsTool()._run(slug, data_in or {}, agent=Agent(), session_id="chat", raw=True)
    assert result.success, result.error
    return json.loads(result.content)


async def test_a_tool_runs_another_through_the_agent(deck, as_agent, wired):
    await _switch_on(deck)
    uid = deck.user["id"]
    as_agent["owner"] = uid
    await _publish(uid, "total", TOTAL, input={"type": "object"}, examples=[{"input": {"numbers": [2, 2]}}])
    await _publish(uid, "report", REPORT, uses=RUNS_TOTAL, examples=RECORDED_TOTAL)
    res = await _run("report")
    assert res["ok"], res
    assert res["output"] == {"total": 6, "files": 1, "text": "6"}
    # Only what it declared.
    other = await _run("report", {"other": "nope"})
    assert "isn't in this tool's uses" in other["output"]["error"]
    assert xt._RUNS == {}


async def test_no_cycles_and_at_most_three_deep(deck, as_agent, wired):
    await _switch_on(deck)
    uid = deck.user["id"]
    as_agent["owner"] = uid
    chain = """import json, sys, captain
d = json.load(sys.stdin)
try:
    print(json.dumps({"inner": captain.custom_tool(NEXT, {})}))
except captain.CaptainError as exc:
    print(json.dumps({"error": str(exc)}))
"""
    recorded = [{"input": {}, "calls": [{"custom_tool": "{next}", "reply": {"ok": True, "output": {"x": 1}}}]}]

    async def link(slug: str, nxt: str) -> None:
        ex = json.loads(json.dumps(recorded).replace("{next}", nxt))
        await _publish(uid, slug, chain.replace("NEXT", repr(nxt)),
                       uses=[{"custom_tool": nxt, "why": "the next step"}], examples=ex)

    await link("ping", "pong")
    await link("pong", "ping")
    res = await _run("ping")
    assert "already running further up this chain" in res["output"]["inner"]["error"]
    for a, b in (("one", "two"), ("two", "three"), ("three", "four"), ("four", "five")):
        await link(a, b)
    deep = await _run("one")
    assert "at most 3 deep" in deep["output"]["inner"]["inner"]["error"]


async def test_the_inner_run_fits_inside_the_outer_ones_time(data):
    await xt.create_tool("u1", "total", title="Total", description="A test tool.")
    _commit("u1", "total", {"main.py": TOTAL, "tool.json": _manifest("total"), "tests/test_main.py": None})
    assert (await xt.publish("u1", "total", by="u1"))["state"] == "passed"
    outer = xt.CallContext(owner="u1", slug="report", title="Report", version="v", caller="u1", call_id="xc_1",
                           call_dir=data, manifest=xt.manifest_of("u1", "total"), deadline=time.monotonic() + 3)
    with pytest.raises(xt.ToolError) as exc:
        await xt.run_tool("u1", "total", "u1", {"numbers": [1]}, parent=outer)
    assert exc.value.kind == "no_time"


async def test_people_need_the_tools_it_runs(deck, as_agent, wired):
    await _switch_on(deck)
    owner, other = deck.user["id"], deck.other["id"]
    await _publish(owner, "total", TOTAL, examples=[{"input": {}}])
    await _publish(owner, "report", REPORT, uses=RUNS_TOTAL, examples=RECORDED_TOTAL)
    db = deck.db
    await db.create_share(xr.SHARE_TYPE, xr.share_id(owner, "report"), owner, other)
    await db.set_settings(other, {xr.added_key(owner, "report"): json.dumps({"at": 1})})
    assert [d["state"] for d in await xr.dependencies_for(other, owner, "report")] == ["not_offered"]
    as_agent["owner"] = other
    res = await _run("report")
    assert "you haven't added" in res["output"]["error"]
    # The owner offers what it runs the same way; then the person adds it.
    from .test_external_tools import _client

    async with _client(deck.user) as c:
        offered = await c.post("/fd/external-tools/report/offer-dependencies")
    assert offered.status_code == 200 and offered.json()["offered"] == ["total"]
    assert [d["state"] for d in await xr.dependencies_for(other, owner, "report")] == ["offered"]
    await db.set_settings(other, {xr.added_key(owner, "total"): json.dumps({"at": 1})})
    assert [d["state"] for d in await xr.dependencies_for(other, owner, "report")] == ["added"]
    res = await _run("report")
    assert res["output"]["total"] == 6, res


def test_a_tool_cant_use_itself_or_an_odd_entry():
    base = json.loads(_manifest("report"))
    with pytest.raises(xt.ToolError, match="can't use itself"):
        xt.parse_manifest({**base, "uses": [{"custom_tool": "report", "why": "x"}]})
    with pytest.raises(xt.ToolError, match="custom_tool"):
        xt.parse_manifest({**base, "uses": [{"custom_tool": "Bad Name", "why": "x"}]})
    assert xt.parse_manifest({**base, "uses": RUNS_TOTAL}).uses == ({"custom_tool": "total", "why": "adds the numbers"},)
