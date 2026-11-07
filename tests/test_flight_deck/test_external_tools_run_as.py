# ruff: noqa: F811 — `data` and `deck` are fixtures imported from test_external_tools
"""Custom tools, Phase 7c: Try it on the page runs a tool that calls back into an
agent as one of the person's own agents ("Run as"), which takes the run's
requests as the person, in a session of its own. What a tool asks of the agent
shows before anyone adds it, and a new version that asks for more rings."""

from __future__ import annotations

import json
import sys

import httpx
import pytest
from fastapi import FastAPI

from captain_claw import custom_tool_callbacks
from captain_claw.flight_deck import agent_idle, external_tool_review
from captain_claw.flight_deck import external_tools as xt
from captain_claw.flight_deck import external_tools_routes as xr
from captain_claw.flight_deck import server as srv
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.tools.registry import ToolResult

from .test_external_tools import LOOPBACK, _commit, _manifest, _switch_on, data, deck  # noqa: F401

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="the line to the agent is POSIX only")

def _client(user: dict) -> httpx.AsyncClient:
    """The routers in server.py's order: fixed paths before /{slug}."""
    app = FastAPI()
    app.include_router(xr.catalog_router)
    app.include_router(xr.router)
    app.dependency_overrides[get_current_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LOOPBACK), base_url="http://deck.test")


ASKS = """import json, captain
print(json.dumps({"got": captain.tool("write", path="vfs:notes/a.md", content="hi")["content"]}))
"""
USES = [{"tool": "write", "why": "saves a note"}]
RECORDED = [{"input": {}, "calls": [{"tool": "write", "reply": {"ok": True, "content": "saved"}}]}]


class Agent:
    def __init__(self) -> None:
        self.ran: list[tuple[str, str]] = []

    async def _execute_tool_with_guard(self, name, arguments, interaction_label="", session_id_override=None, **_):
        self.ran.append((name, str(session_id_override)))
        return ToolResult(success=True, content=f"{name} done")


@pytest.fixture
def agents(deck, monkeypatch):
    """The person's agents in the registry, and each one's /api/custom-tools/host
    and /callback, in-process."""
    custom_tool_callbacks.reset_for_tests()
    uid = deck.user["id"]
    state = {"alive": {"helper"}, "agent": Agent(), "woken": []}
    registry = {
        "helper": {"owner": uid, "name": "Helper", "web_port": 1, "web_auth": "t"},
        "sleepy": {"owner": uid, "name": "Sleepy", "web_port": 2, "web_auth": "t", "asleep": True},
        "off": {"owner": uid, "name": "Off", "web_port": 3, "web_auth": "t", "stopped": True},
        "theirs": {"owner": deck.other["id"], "name": "Theirs", "web_port": 4, "web_auth": "t"},
    }
    monkeypatch.setattr(srv, "_load_process_registry", lambda: registry)
    monkeypatch.setattr(srv, "_process_is_alive", lambda slug: slug in state["alive"])

    async def wake(slug, timeout=0):
        state["woken"].append(slug)
        state["alive"].add(slug)
        return True

    monkeypatch.setattr(agent_idle, "wake", wake)

    async def agent_post(agent_slug, path, body):
        assert path == xr.HOST_PATH
        if body.get("close"):
            custom_tool_callbacks.close_wait(body["close"])
            return 200, {"ok": True}
        cb = custom_tool_callbacks.open_wait(state["agent"], custom_tool_callbacks.TRY_SESSION, tool=body["tool"],
                                             ttl=body["ttl"])
        state["callback"] = cb
        return 200, {"ok": True, "callback": cb}

    async def post(self, body, timeout):
        state["identity"] = body["identity"]
        status, answer = await custom_tool_callbacks.handle(body)
        if status != 200:
            raise xt.ToolError("agent_refused", answer.get("error", ""))
        return answer

    monkeypatch.setattr(xr, "_agent_post", agent_post)
    monkeypatch.setattr(xr.AgentLink, "_post", post)
    yield state
    custom_tool_callbacks.reset_for_tests()


async def _publish(uid: str, slug: str, main_py: str = ASKS, **manifest) -> None:
    await xt.create_tool(uid, slug, title=slug.title(), description="A test tool.")
    _commit(uid, slug, {"main.py": main_py, "tool.json": _manifest(slug, **{"uses": USES, "examples": RECORDED, **manifest}),
                        "tests/test_main.py": None})
    state = await xt.publish(uid, slug, by=uid)
    assert state["state"] == "passed", state["steps"]


async def test_try_it_runs_as_one_of_your_agents(deck, agents):
    await _switch_on(deck)
    await _publish(deck.user["id"], "notes")
    async with _client(deck.user) as c:
        mine = (await c.get("/fd/external-tools/agents")).json()["agents"]
        assert [(a["slug"], a["status"]) for a in mine] == [("helper", "running"), ("off", "stopped"), ("sleepy", "asleep")]
        need = await c.post("/fd/external-tools/notes/run", json={"input": {}})
        assert need.status_code == 409 and "pick which of your agents" in need.json()["detail"]
        res = await c.post("/fd/external-tools/notes/run", json={"input": {}, "agent": "helper"})
        assert res.status_code == 200, res.text
        assert res.json()["output"] == {"got": "write done"}
        # Not someone else's agent; not one that's off; one asleep is woken.
        assert (await c.post("/fd/external-tools/notes/run", json={"input": {}, "agent": "theirs"})).status_code == 404
        off = await c.post("/fd/external-tools/notes/run", json={"input": {}, "agent": "off"})
        assert off.status_code == 409 and "Off is off" in off.json()["detail"]
        woke = await c.post("/fd/external-tools/notes/run", json={"input": {}, "agent": "sleepy"})
        assert woke.status_code == 200 and agents["woken"] == ["sleepy"]
    assert agents["agent"].ran[0] == ("write", custom_tool_callbacks.TRY_SESSION)
    assert agents["identity"] == {"user_id": deck.user["id"], "member": False, "grant": "", "owner_runs": True}
    # The slot closes with the run.
    assert custom_tool_callbacks.waiting(agents["callback"]) is None


async def test_what_it_asks_of_the_agent_shows_and_rings_when_it_grows(deck, agents):
    await _switch_on(deck)
    owner, other = deck.user["id"], deck.other["id"]
    await _publish(owner, "notes")
    declares = xr._declares(owner, "notes")
    assert declares["uses"] == ["write"] and declares["runs"] == []
    await deck.db.set_settings(other, {xr.added_key(owner, "notes"): json.dumps({"at": 1})})
    more = USES + [{"tool": "web_fetch", "hosts": ["example.com"], "why": "reads a page"}]
    recorded = [{"input": {}, "calls": [{"tool": "write", "reply": {"ok": True}}]}]
    _commit(owner, "notes", {"tool.json": _manifest("notes", uses=more, examples=recorded)})
    assert (await xt.publish(owner, "notes", by=owner))["state"] == "passed"
    bells = [n for n in await deck.db.list_notifications(other) if n.get("ref_type") == "custom_tool"]
    assert bells and "to use your agent's web_fetch (example.com)" in bells[0]["title"]
    assert xr.declares_more({"uses": [], "runs": [], "network": False, "secrets": [], "timeout_s": 1, "memory_mb": 1},
                            {"uses": [], "runs": ["total"], "network": False, "secrets": [], "timeout_s": 1,
                             "memory_mb": 1}) == ["to run total"]


def test_the_spec_check_knows_a_tool_can_use_its_agent():
    assert "uses" in external_tool_review.SYSTEM and "Prefer that to network access" in external_tool_review.SYSTEM
    assert "agents" in xt.RESERVED_SLUGS and "asks" in xt.RESERVED_SLUGS
