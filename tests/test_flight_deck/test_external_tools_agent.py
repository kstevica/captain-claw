# ruff: noqa: F811 — the `data` and `deck` fixtures are imported from test_external_tools
"""Custom tools, Phase 1: agents call them through Flight Deck. The person a call
runs as is the agent's owner on Flight Deck's record, or on a shared agent the
member whose turn it is (their tools, secrets and files, never the owner's)."""

from __future__ import annotations

import json
import time
import types

import httpx
import pytest
from fastapi import FastAPI

from captain_claw import custom_tools_client, speaker
from captain_claw.flight_deck import agent_identity, external_tools_routes, speaker_grants
from captain_claw.flight_deck import external_tools as xt
from captain_claw.rehearsal_turn import EXEC_REFUSED_TOOLS
from captain_claw.tools.custom_tools import CustomToolsTool
from captain_claw.tools.registry import ToolRegistry

from .test_external_tools import (  # noqa: F401 — fixtures
    LOOPBACK,
    _commit,
    _manifest,
    _switch_on,
    data,
    deck,
)

ECHO = "import json,sys,os\nd=json.load(sys.stdin)\nprint(json.dumps({'who': os.environ.get('WHO'), 'got': d}))\n"


def _agent_app() -> FastAPI:
    app = FastAPI()
    app.include_router(external_tools_routes.agent_router)
    app.include_router(external_tools_routes.router)
    return app


def _agent_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=_agent_app(), client=LOOPBACK), base_url="http://deck.test")


@pytest.fixture
def as_agent(monkeypatch):
    """Who's calling: ``who["owner"]`` is the agent's owner; ``who["member"]`` a
    member whose grant the call carries (None: the owner's own turn)."""
    who: dict = {"owner": "", "member": None}
    monkeypatch.setattr(agent_identity, "require_agent_caller",
                        lambda request, what="": agent_identity.AgentCaller(slug="helper", owner=who["owner"]))
    monkeypatch.setattr(speaker_grants, "member_request", lambda request: who["member"] is not None)

    async def acting(request, **_):
        return speaker_grants.ActingMember(user_id=who["member"], owner=who["owner"], agent_ref="process:helper",
                                           lane="A", turn="t", slug="helper", name="Helper")

    monkeypatch.setattr(speaker_grants, "acting_member", acting)
    return who


async def _make(uid: str, slug: str, main_py: str = ECHO, **manifest) -> None:
    await xt.create_tool(uid, slug, title=slug.title(), description=f"The {slug} tool.")
    _commit(uid, slug, {"main.py": main_py, "tool.json": _manifest(slug, **manifest), "tests/test_main.py": None})


async def _publish(uid: str, slug: str, secrets: dict | None = None) -> None:
    state = await xt.publish(uid, slug, by=uid, secret_values=secrets)
    assert state["state"] == "passed", state


# ── Flight Deck's agent routes ───────────────────────────────────────────


def test_the_member_routes_are_grant_aware():
    assert external_tools_routes.GRANT_AWARE <= speaker_grants.GRANT_AWARE_PATHS
    assert "/fd/external-tools/agent/status" not in speaker_grants.GRANT_AWARE_PATHS


async def test_status_says_off_then_lists_the_owners_tools(deck, as_agent):
    as_agent["owner"] = deck.admin["id"]
    async with _agent_client() as c:
        assert (await c.get("/fd/external-tools/agent/status")).json() == {"enabled": False}
        assert (await c.get("/fd/external-tools/agent/tools")).status_code == 403
    await _switch_on(deck)
    await _make(deck.admin["id"], "live-one")
    await _publish(deck.admin["id"], "live-one")
    await _make(deck.admin["id"], "draft-one")  # never published: not offered
    async with _agent_client() as c:
        status = (await c.get("/fd/external-tools/agent/status")).json()
        assert status["enabled"] and status["can_build"] is True
        assert [t["ref"] for t in status["tools"]] == ["live-one"]
        tools = (await c.get("/fd/external-tools/agent/tools")).json()["tools"]
        assert [t["ref"] for t in tools] == ["live-one"] and tools[0]["secrets_missing"] == []


async def test_the_owner_runs_with_their_secret(deck, as_agent):
    await _switch_on(deck)
    uid = deck.admin["id"]
    as_agent["owner"] = uid
    await _make(uid, "who", secrets=[{"name": "WHO", "why": "says who"}],
                input={"type": "object", "properties": {"n": {"type": "integer"}}})
    await _publish(uid, "who", {"WHO": "owner"})
    async with _agent_client() as c:
        tools = (await c.get("/fd/external-tools/agent/tools")).json()["tools"]
        assert tools[0]["secrets_missing"] == ["WHO"]
        resp = await c.post("/fd/external-tools/agent/run", json={"tool": "who", "input": {"n": 1}})
        assert resp.status_code == 400 and "WHO isn't set" in resp.json()["detail"]
        key = xt.secret_key(uid, "who", "WHO")
        from captain_claw.flight_deck import fd_crypto
        await deck.db.set_settings(uid, {key: fd_crypto.seal("the owner")})
        out = (await c.post("/fd/external-tools/agent/run", json={"tool": "who", "input": {"n": 2}})).json()
        assert out["ok"] and out["output"] == {"who": "the owner", "got": {"n": 2}}
        described = (await c.post("/fd/external-tools/agent/describe", json={"tool": "who"})).json()
        assert described["secrets"] == [{"name": "WHO", "why": "says who", "set": True}]
        assert described["input"]["properties"] == {"n": {"type": "integer"}}
        assert (await c.post("/fd/external-tools/agent/run", json={"tool": "nope", "input": {}})).status_code == 404
        assert (await c.post("/fd/external-tools/agent/run", json={"tool": "../x", "input": {}})).status_code == 404


async def test_a_members_turn_runs_as_the_member(deck, as_agent):
    await _switch_on(deck)
    owner, member = deck.admin["id"], deck.user["id"]
    as_agent["owner"] = owner
    await _make(owner, "owners", secrets=[{"name": "WHO", "why": ""}])
    await _publish(owner, "owners", {"WHO": "o"})
    await _make(member, "mine", secrets=[{"name": "WHO", "why": ""}])
    await _publish(member, "mine", {"WHO": "m"})
    from captain_claw.flight_deck import fd_crypto
    await deck.db.set_settings(owner, {xt.secret_key(owner, "owners", "WHO"): fd_crypto.seal("owner-secret"),
                                       xt.secret_key(member, "mine", "WHO"): fd_crypto.seal("owner-on-members")})
    await deck.db.set_settings(member, {xt.secret_key(member, "mine", "WHO"): fd_crypto.seal("member-secret")})
    as_agent["member"] = member
    async with _agent_client() as c:
        tools = (await c.get("/fd/external-tools/agent/tools")).json()
        assert [t["ref"] for t in tools["tools"]] == ["mine"] and tools["can_build"] is False
        out = (await c.post("/fd/external-tools/agent/run", json={"tool": "mine", "input": {}})).json()
        assert out["output"]["who"] == "member-secret"
        # The owner's tool isn't the member's to run (sharing comes in Phase 3).
        assert (await c.post("/fd/external-tools/agent/run", json={"tool": "owners", "input": {}})).status_code == 404
    as_agent["member"] = None
    async with _agent_client() as c:
        out = (await c.post("/fd/external-tools/agent/run", json={"tool": "owners", "input": {}})).json()
        assert out["output"]["who"] == "owner-secret"


async def test_agent_routes_win_over_a_tool_name(deck, as_agent):
    await _switch_on(deck)
    as_agent["owner"] = deck.admin["id"]
    with pytest.raises(xt.ToolError):
        await xt.create_tool(deck.admin["id"], "agent", title="A", description="d")
    async with _agent_client() as c:
        assert (await c.get("/fd/external-tools/agent/status")).json()["enabled"] is True


async def test_files_come_from_the_runners_vfs(deck, as_agent):
    await _switch_on(deck)
    uid = deck.admin["id"]
    as_agent["owner"] = uid
    main_py = ("import json,os,sys\nfrom pathlib import Path\nd=json.load(sys.stdin)\n"
               "t=Path(d['f']).read_text() if d.get('f') else ''\n"
               "Path(os.environ['XT_OUT'],'n.txt').write_text(str(len(t)))\nprint(json.dumps({'n':len(t)}))\n")
    await _make(uid, "count", main_py, input={"type": "object", "properties": {"f": {"type": "string", "format": "file"}}})
    await _publish(uid, "count")
    root = xt.root().parent / "vfs" / uid / "docs"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("four")
    async with _agent_client() as c:
        out = (await c.post("/fd/external-tools/agent/run", json={"tool": "count", "input": {"f": "vfs:docs/a.txt"}})).json()
    assert out["output"] == {"n": 4}
    assert out["files"][0]["path"].startswith("vfs:custom-tools/count/")


# ── The agent's tool ─────────────────────────────────────────────────────


@pytest.fixture
def wired(monkeypatch):
    """The real tool and client, talking to the deck app in-process."""
    monkeypatch.setattr(custom_tools_client, "_fd_base", lambda: "http://deck.test")
    monkeypatch.setattr(custom_tools_client, "_fd_headers", lambda: {"X-Agent-Auth": "t"})
    real = httpx.AsyncClient
    monkeypatch.setattr(custom_tools_client, "httpx", types.SimpleNamespace(
        AsyncClient=lambda **kw: real(transport=httpx.ASGITransport(app=_agent_app(), client=LOOPBACK),
                                      base_url="http://deck.test", timeout=kw.get("timeout")),
        Response=httpx.Response))
    custom_tools_client.reset_for_tests()
    yield
    custom_tools_client.reset_for_tests()


async def test_the_tool_over_http(deck, as_agent, wired):
    await _switch_on(deck)
    uid = deck.admin["id"]
    as_agent["owner"] = uid
    await _make(uid, "echo", input={"type": "object", "properties": {"text": {"type": "string", "description": "what to say"}},
                                    "required": ["text"]}, examples=[{"input": {"text": "x"}}])
    await _publish(uid, "echo")
    assert await custom_tools_client.refresh(force=True) is True
    assert custom_tools_client.owner_tools() == [{"ref": "echo", "title": "Echo", "description": "A test tool."}]
    tool = CustomToolsTool()
    listed = await tool.execute(action="list")
    assert listed.success and "- echo: Echo." in listed.content
    described = await tool.execute(action="describe", tool="echo")
    assert "- text: string — what to say" in described.content and "Example input" in described.content
    ran = await tool.execute(action="run", tool="echo", input={"text": "hi"})
    assert ran.success and '"text": "hi"' in ran.content
    ran = await tool.execute(action="run", tool="echo", input='{"text": "as a string"}')
    assert ran.success and "as a string" in ran.content
    bad = await tool.execute(action="run", tool="echo", input={"text": 3})
    assert not bad.success and "doesn't fit this tool" in bad.error
    assert not (await tool.execute(action="run", tool="echo", input="not json")).success
    assert not (await tool.execute(action="build")).success


async def test_a_failed_run_says_why(deck, as_agent, wired):
    await _switch_on(deck)
    uid = deck.admin["id"]
    as_agent["owner"] = uid
    await _make(uid, "boom", "import json,sys\nd=json.load(sys.stdin)\nif d.get('x'):\n    raise SystemExit('bad thing')\nprint('{}')\n")
    await _publish(uid, "boom")
    out = await CustomToolsTool().execute(action="run", tool="boom", input={"x": 1})
    assert not out.success and "crashed" in out.error and "bad thing" in out.error


async def test_refresh_says_off_when_the_deck_has_them_off(deck, as_agent, wired):
    as_agent["owner"] = deck.admin["id"]
    assert await custom_tools_client.refresh(force=True) is False
    assert not custom_tools_client.usable()


async def test_a_member_without_a_grant_is_refused_before_any_call(monkeypatch):
    calls = []
    monkeypatch.setattr(custom_tools_client, "_fd_base", lambda: "http://deck.test")

    def no_grant():
        raise speaker.SpeakerGrantMissing("none")

    monkeypatch.setattr(speaker, "grant_headers", no_grant)
    monkeypatch.setattr(custom_tools_client, "httpx", types.SimpleNamespace(AsyncClient=lambda **kw: calls.append(kw)))
    status, data = await custom_tools_client.call("GET", "tools")
    assert status == 403 and data["detail"] == custom_tools_client.NO_GRANT and calls == []


def test_the_registry_shows_the_tool_only_while_on(monkeypatch):
    registry = ToolRegistry()
    registry.register(CustomToolsTool(), metadata={"requires_custom_tools": True})
    custom_tools_client.reset_for_tests()
    assert "custom_tools" not in {t["name"] for t in registry.get_definitions()}
    monkeypatch.setitem(custom_tools_client._STATE, "usable", True)
    monkeypatch.setitem(custom_tools_client._STATE, "at", time.time())
    assert "custom_tools" in {t["name"] for t in registry.get_definitions()}
    custom_tools_client.reset_for_tests()


def test_a_member_may_list_describe_and_run_only():
    for action in ("list", "describe", "run"):
        assert speaker.apply_tool_rules("custom_tools", {"action": action}, None)[1] is None
    for action in ("build", "improve", "propose", ""):
        assert speaker.apply_tool_rules("custom_tools", {"action": action}, None)[1] is not None
    assert "custom_tools" in speaker.SPEAKER_TOOL_ALLOWLIST_MAX


def test_a_rehearsal_runs_no_tool():
    assert "custom_tools" in EXEC_REFUSED_TOOLS


def test_the_tool_schema_has_no_path_arguments():
    assert set(CustomToolsTool.parameters["properties"]) == {"action", "tool", "input", "title", "what", "why",
                                                             "harness", "model"}
    assert json.dumps(CustomToolsTool.parameters)
