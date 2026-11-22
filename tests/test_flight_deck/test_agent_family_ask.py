"""Agent family: asking, or giving a task to, several of the family at once
(``POST /fd/family/agent/send``, the ``flight_deck`` tool's ``ask`` / ``task``).

* ``agent_family.recipients``: parent, children, siblings, family, a group, a
  name; never itself, a worker or someone else's agent; at most 12.
* ``ask``: every recipient asked at once, each told who's asking; the answers
  come back together, with the ones that didn't answer.
* ``task``: answered at once; one turn for the asker on all the results later.
* The guards: an agent being asked can't ask back; two going at once; an hourly
  cap. Groups only where the deck keeps them.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from captain_claw.flight_deck import agent_family as fam
from captain_claw.flight_deck import server

from .test_agent_family import bob_agent  # noqa: F401 — fixture
from .test_spawn_owner_integrity import (  # noqa: F401 — fixtures
    BOB, LOOPBACK, _FakePopen, deck, fd_db, spawn_env, users,
)


def _reg(**entries) -> dict:
    out = {}
    for slug, (parent, *rest) in entries.items():
        e = {"slug": slug, "name": slug.title(), "owner": rest[0] if rest else "u"}
        if parent:
            e["parent"] = parent
        out[slug] = e
    return out


FAMILY = dict(top=("",), me=("top",), sib=("top",), sib2=("top",), kid=("me",), other=("",),
              theirs=("top", "v"))


class TestGroupNames:
    def test_a_group_is_reached_by_the_name_the_agent_was_shown(self):
        reg = _reg(**FAMILY)
        groups = [{"id": "g", "name": "  **Deal_flow**\n", "type": "group", "agentIds": ["proc-sib"]},
                  {"id": "r", "name": "Reviewers", "type": "role", "agentIds": ["proc-sib2"]}]
        assert fam.folders(groups) == [("Deal flow", ["sib"])]  # roles aren't groups
        assert fam.recipients(reg, "me", "group:deal flow", groups=groups) == ["sib"]
        assert fam.recipients(reg, "me", "group:Deal_flow", groups=groups) == ["sib"]
        with pytest.raises(fam.FamilyError, match="no agent group called 'Reviewers'"):
            fam.recipients(reg, "me", "group:Reviewers", groups=groups)

    def test_groups_of(self):
        reg = _reg(**FAMILY)
        reg["w"] = {"slug": "w", "owner": "u", "parent": "top", "role": "worker"}
        groups = [{"name": "Ops", "type": "group", "agentIds": ["proc-top"]},
                  {"name": "Research", "type": "group",
                   "agentIds": ["proc-sib", "proc-me", "proc-w", "proc-theirs", "proc-gone", "proc-sib"]}]
        assert fam.groups_of(reg, "me", groups) == [
            {"name": "Research", "members": ["sib"], "mine": True},   # own first; no worker, stranger or ghost
            {"name": "Ops", "members": ["top"], "mine": False},
        ]
        assert fam.groups_of(reg, "w", groups) == [] and fam.groups_of(reg, "nobody", groups) == []
        assert fam.groups_of(reg, "me", None) == [] and fam.groups_of(reg, "me", "junk") == []


class TestRecipients:
    def test_by_relation(self):
        reg = _reg(**FAMILY)
        reg["w"] = {"slug": "w", "owner": "u", "parent": "top", "role": "worker"}
        assert fam.recipients(reg, "me", "parent") == ["top"]
        assert fam.recipients(reg, "me", "Siblings") == ["sib", "sib2"]  # never theirs, never a worker
        assert fam.recipients(reg, "me", "children") == ["kid"]
        assert fam.recipients(reg, "me", "family") == ["top", "sib", "sib2", "kid"]
        assert fam.recipients(reg, "me", "Other") == ["other"]

    def test_what_isnt_there_says_so(self):
        reg = _reg(**FAMILY)
        for to, why in (("parent", "no parent"), ("siblings", "sit under nobody"), ("children", "No agents sit")):
            with pytest.raises(fam.FamilyError, match=why):
                fam.recipients(reg, "other", to)
        with pytest.raises(fam.FamilyError, match="no agent called"):
            fam.recipients(reg, "me", "nobody")
        with pytest.raises(fam.FamilyError, match="but you"):
            fam.recipients(reg, "me", "me")

    def test_groups(self):
        reg = _reg(**FAMILY)
        groups = [{"id": "g", "name": "Research", "type": "group", "agentIds": ["proc-sib", "proc-me", "abc123", "proc-theirs"]}]
        assert fam.recipients(reg, "me", "group:research", groups=groups) == ["sib"]
        with pytest.raises(fam.FamilyError, match="no agent group called 'Ops'. Their groups: Research."):
            fam.recipients(reg, "me", "group:Ops", groups=groups)
        with pytest.raises(fam.FamilyError, match="live in the browser"):
            fam.recipients(reg, "me", "group:Research", groups=None)

    def test_at_most_twelve(self):
        reg = _reg(top=("",), **{f"k{i}": ("top",) for i in range(fam.FANOUT_MAX + 1)})
        with pytest.raises(fam.FamilyError, match="at most 12"):
            fam.recipients(reg, "top", "children")


# ── through the deck ─────────────────────────────────────────────────


@pytest.fixture
def family(deck, users, monkeypatch):
    """bob-agent sits under bob-top, with two siblings; the peer calls are stood in for."""
    reg = server._load_process_registry()
    for slug, parent, port in (("bob-top", "", 24160), ("bob-sib", "bob-top", 24161), ("bob-sib2", "bob-top", 24162)):
        reg[slug] = {"slug": slug, "name": slug.replace("-", " ").title(), "owner": BOB, "web_port": port,
                     "web_auth": f"{slug}-token", **({"parent": parent} if parent else {})}
    reg["bob-agent"]["parent"] = "bob-top"
    server._save_process_registry(reg)
    for k in ("_fanout_times", "_fanout_active", "_being_asked"):
        getattr(server, k).clear()
    seen: list[dict] = []
    told: list[tuple] = []

    async def _consult(host, port, auth, message, **kw):
        seen.append({"port": port, "auth": auth, "message": message, "kw": kw})
        await asyncio.sleep(0)
        if port == 24162:
            yield {"ok": False, "error": "busy after 180s"}
            return
        yield {"event": "status", "data": {}}
        yield {"ok": True, "done": True, "response": f"answer from {port}"}

    async def _notify(port, auth, content):
        told.append((port, auth, content))
        return True

    monkeypatch.setattr(server, "_consult_peer_events", _consult)
    monkeypatch.setattr(server, "_notify_agent_turn", _notify)
    return {"seen": seen, "told": told}


async def _send(token: str, **body):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=LOOPBACK),
                                 base_url="http://fd.test") as c:
        return await c.post("/fd/family/agent/send", json=body, headers={"X-Agent-Auth": token})


class TestAsk:
    async def test_siblings_answer_together(self, family):
        r = await _send("bob-agent-token", to="siblings", message="Which client is this?", mode="ask")
        assert r.status_code == 200, r.text
        got = r.json()
        assert [x["name"] for x in got["results"]] == ["Bob Sib", "Bob Sib2"]
        assert got["results"][0] == {"slug": "bob-sib", "name": "Bob Sib", "relation": "your sibling, also under Bob Top",
                                     "ok": True, "answer": "answer from 24161"}
        assert got["results"][1]["ok"] is False and "busy" in got["results"][1]["error"]
        assert got["message"].startswith("[Answers from siblings: 1 of 2 answered]")
        assert "### Bob Sib (your sibling)\nanswer from 24161" in got["message"]
        sent = family["seen"][0]
        assert sent["auth"] == "bob-sib-token"
        assert sent["message"].startswith("[From bob-agent, your sibling, also under Bob Top]\n"
                                          "It asked these agents the same question")
        assert sent["message"].endswith("Which client is this?")
        assert sent["kw"]["automation"]["kind"] == "peer"
        assert server._being_asked == {} and server._fanout_active.get("bob-agent") == 0

    async def test_one_by_name_and_the_parent(self, family):
        r = await _send("bob-agent-token", to="Bob Top", message="ok?", mode="ask")
        assert r.status_code == 200 and [x["slug"] for x in r.json()["results"]] == ["bob-top"]
        assert "Answer it briefly." in family["seen"][-1]["message"]
        assert family["seen"][-1]["message"].startswith("[From bob-agent, your child")

    async def test_refusals(self, family, monkeypatch):
        r = await _send("bob-agent-token", to="children", message="x", mode="ask")
        assert r.status_code == 409 and "No agents sit under you" in r.json()["detail"]
        r = await _send("bob-agent-token", to="siblings", message="x", mode="shout")
        assert r.status_code == 400
        server._being_asked["bob-agent"] = "bob-top"
        r = await _send("bob-agent-token", to="siblings", message="x", mode="ask")
        assert r.status_code == 409 and "Bob Top is asking you something right now" in r.json()["detail"]
        server._being_asked.clear()
        server._fanout_active["bob-agent"] = 2
        assert (await _send("bob-agent-token", to="siblings", message="x", mode="ask")).status_code == 429
        server._fanout_active.clear()
        monkeypatch.setattr(server, "_FANOUTS_PER_HOUR", 1)
        assert (await _send("bob-agent-token", to="parent", message="x", mode="ask")).status_code == 200
        r = await _send("bob-agent-token", to="parent", message="x", mode="ask")
        assert r.status_code == 429 and "1 times in the last hour" in r.json()["detail"]
        assert (await _send("not-a-token", to="parent", message="x")).status_code == 403


class TestTask:
    async def test_one_summary_comes_back(self, family):
        r = await _send("bob-agent-token", to="family", message="Draft the Monday notes.", mode="task")
        assert r.status_code == 200, r.text
        got = r.json()
        assert got["sent_to"] == ["Bob Top", "Bob Sib", "Bob Sib2"] and got["message"].startswith("Given to Bob Top")
        for _ in range(20):
            await asyncio.sleep(0)
            if family["told"]:
                break
        [(port, auth, content)] = family["told"]
        assert port == 24102 and auth == "bob-agent-token"
        assert content.startswith("[Results of the task you gave family: 2 of 3 answered]")
        assert "It gives you this task." in family["seen"][0]["message"]
        assert server._fanout_active.get("bob-agent") == 0


class TestGroups:
    async def test_a_group_on_a_deck_that_keeps_them(self, family, users):
        await users.set_settings(BOB, {"fd:agent-groups": json.dumps(
            [{"id": "g", "name": "Clients", "type": "group", "agentIds": ["proc-bob-sib2", "proc-bob-top"]}])})
        r = await _send("bob-agent-token", to="group:clients", message="x", mode="ask")
        assert r.status_code == 200, r.text
        assert [x["slug"] for x in r.json()["results"]] == ["bob-sib2", "bob-top"]

    async def test_not_without_sign_in(self, family, monkeypatch):
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        assert await server._owner_agent_groups(BOB) is None


class TestTool:
    async def test_ask_through_the_tool(self, bob_agent, users, monkeypatch):
        from captain_claw.tools.flight_deck import FlightDeckTool

        reg = server._load_process_registry()
        reg["bob-top"] = {"slug": "bob-top", "name": "Bob Top", "owner": BOB, "web_port": 24160, "web_auth": "t"}
        reg["bob-agent"]["parent"] = "bob-top"
        server._save_process_registry(reg)
        for k in ("_fanout_times", "_fanout_active", "_being_asked"):
            getattr(server, k).clear()

        async def _consult(host, port, auth, message, **kw):
            yield {"ok": True, "done": True, "response": "yes"}

        monkeypatch.setattr(server, "_consult_peer_events", _consult)
        res = await FlightDeckTool().execute("ask", message="ok?", to="parent", fd_url="http://fd.test")
        assert res.success, res.error
        assert "[Answers from parent: 1 of 1 answered]" in res.content and "yes" in res.content
        res = await FlightDeckTool().execute("ask", message="ok?", fd_url="http://fd.test")
        assert not res.success and "to is required" in res.error
        res = await FlightDeckTool().execute("task", message="x", to="siblings", fd_url="http://fd.test")
        assert not res.success and "no siblings" in res.error
