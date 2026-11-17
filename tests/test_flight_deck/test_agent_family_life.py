"""Agent family, Phase 5 (docs/agent-family-plan.md): what happens to a family
when agents come and go.

* Removing an agent: the ones under it move up, or (``family=remove``) go too,
  deepest first.
* A clone sits beside its source (under the same parent when there's room).
* Each agent keeps a short family history, served with its family.
* A Basna/Vatra run an agent started: its workers sit under that agent.
* A bundle carries ``made_for`` but not where the agent sat.
"""

from __future__ import annotations

import types

import httpx
import pytest

from captain_claw.flight_deck import agent_family as fam
from captain_claw.flight_deck import server
from captain_claw.flight_deck.auth import create_access_token

from .test_spawn_owner_integrity import (  # noqa: F401 — fixtures
    BOB, LOOPBACK, REMOTE, _FakePopen, deck, fd_db, spawn_env, users,
)


def _bob():
    return {"Authorization": f"Bearer {create_access_token(BOB)}"}


def _client(addr=REMOTE):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=addr), base_url="http://fd.test")


def _add(**entries):
    reg = server._load_process_registry()
    for slug, parent in entries.items():
        reg[slug] = {"slug": slug, "name": slug.title(), "owner": BOB, "web_port": 0,
                     **({"parent": parent} if parent else {})}
    server._save_process_registry(reg)


class TestRemoval:
    async def test_the_ones_under_it_move_up(self, deck, users):
        _add(top="", mid="top", leaf="mid")
        async with _client() as c:
            r = await c.delete("/fd/processes/mid", headers=_bob())
        assert r.status_code == 200, r.text
        reg = server._load_process_registry()
        assert "mid" not in reg and fam.parent_of(reg, "leaf") == "top"
        assert reg["leaf"]["family_log"][-1]["text"] == "Mid was removed: moved up under Top"

    async def test_or_go_too_deepest_first(self, deck, users, monkeypatch):
        _add(top="", mid="top", leaf="mid", other="top", keep="")
        order = []
        real = server._remove_process_agent

        async def _spy(slug, **kw):
            order.append(slug)
            return await real(slug, **kw)

        monkeypatch.setattr(server, "_remove_process_agent", _spy)
        async with _client() as c:
            r = await c.delete("/fd/processes/top?family=remove", headers=_bob())
            assert r.status_code == 200, r.text
            assert r.json()["message"].endswith("and the 3 under it")
            assert (await c.delete("/fd/processes/keep?family=everything", headers=_bob())).status_code == 400
        assert order.index("leaf") < order.index("mid") < order.index("top") and order[-1] == "top"
        assert set(server._load_process_registry()) >= {"keep"} and not {"top", "mid", "leaf", "other"} & set(server._load_process_registry())

    def test_removal_order(self):
        reg = {s: {"slug": s, "owner": "u", **({"parent": p} if p else {})}
               for s, p in (("t", ""), ("a", "t"), ("b", "t"), ("a1", "a"), ("w", "t"))}
        reg["w"]["role"] = "worker"
        got = fam.removal_order(reg, "t")
        assert got[0] == "a1" and set(got) == {"a1", "a", "b"}  # never the run's worker


class TestClone:
    @pytest.fixture
    def src(self, deck, monkeypatch):
        d = deck / "kid"
        (d / "data" / "home-config").mkdir(parents=True)
        cfg = "web:\n  port: 24150\n  auth_token: kid-token\n"
        (d / "config.yaml").write_text(cfg)
        (d / "data" / "home-config" / "config.yaml").write_text(cfg)
        _add(top="", kid="top")
        reg = server._load_process_registry()
        reg["kid"].update(web_port=24150, web_auth="kid-token", made_for="checks facts")
        server._save_process_registry(reg)
        monkeypatch.setattr(server, "_find_available_port", lambda start: 24778)
        return d

    async def test_beside_its_source(self, src):
        res = await server.clone_process("kid", server.CloneRequest(new_name="kid 2"),
                                         types.SimpleNamespace(state=types.SimpleNamespace(user_id=BOB)), None)
        assert res.ok
        reg = server._load_process_registry()
        clone = reg["kid-2"]
        assert fam.parent_of(reg, "kid-2") == "top" and clone["made_for"] == "checks facts"
        assert "made_by" not in clone and clone["family_log"][-1]["text"] == "Cloned from Kid by its owner"

    async def test_top_level_when_its_parent_is_full(self, src):
        _add(**{f"k{i}": "top" for i in range(fam.MAX_CHILDREN - 1)})
        res = await server.clone_process("kid", server.CloneRequest(new_name="kid 3"),
                                         types.SimpleNamespace(state=types.SimpleNamespace(user_id=BOB)), None)
        assert res.ok and fam.parent_of(server._load_process_registry(), "kid-3") == ""


class TestHistory:
    async def test_spawn_move_and_the_route(self, spawn_env, users):
        async with _client(LOOPBACK) as c:
            r = await c.post("/fd/spawn-process", json={"name": "kid", "made_for": "checks facts"},
                             headers={"X-Agent-Auth": "bob-agent-token"})
            assert r.status_code == 200, r.text
            _add(top="")
            r = await c.put("/fd/processes/kid/parent", json={"parent": "top"}, headers=_bob())
            assert r.status_code == 200
            r = await c.post("/fd/spawn-process", json={"name": "kid"}, headers=_bob())  # a re-spawn keeps it
            got = (await c.get("/fd/processes/kid/family", headers=_bob())).json()
            top = (await c.get("/fd/processes/top/family", headers=_bob())).json()
            other = await c.get("/fd/processes/alice-agent/family", headers=_bob())
        assert [x["text"] for x in got["log"]] == ["Made by bob-agent", "Put under Top by its owner"]
        assert got["parent"] == "top" and got["parent_name"] == "Top"
        assert got["made_by"] == "bob-agent" and got["made_by_name"] == "bob-agent" and got["made_for"] == "checks facts"
        assert top["children"] == [{"slug": "kid", "name": "kid"}] and top["log"] == []
        assert other.status_code == 404

    def test_capped(self):
        e: dict = {}
        for i in range(fam.LOG_MAX + 5):
            fam.log(e, f"line {i}")
        assert len(e["family_log"]) == fam.LOG_MAX and e["family_log"][-1]["text"] == f"line {fam.LOG_MAX + 4}"


class _Cfg:
    def __init__(self, name, env):
        self.name, self.made_for, self._family = name, "", {}
        self.env_vars = [{"key": k, "value": v} for k, v in env.items()]


class TestRunStarters:
    def test_workers_sit_under_the_agent_that_started_the_run(self, monkeypatch):
        monkeypatch.setattr(fam, "_RUN_STARTERS", {})
        reg = {"glasses": {"slug": "glasses", "owner": "u", "web_port": 24001},
               "theirs": {"slug": "theirs", "owner": "v", "web_port": 24002}}
        assert fam.note_run_starter("vatra:s1", reg, port=24001, owner="u") == "glasses"
        assert fam.note_run_starter("basna:abcd1234", reg, port=24001, owner="u") == "glasses"
        assert fam.note_run_starter("vatra:s2", reg, port=24002, owner="u") == ""  # not theirs
        got = fam.for_spawn(_Cfg("vatra-s1-x-y", {"CLAW_VATRA_WORKER": "1", "CLAW_VATRA_SESSION": "s1"}),
                            prior=None, caller="")
        assert got["role"] == "worker" and got["parent"] == got["made_by"] == "glasses" and got["run"] == "vatra:s1"
        got = fam.for_spawn(_Cfg("basna-abcd1234-r1-researcher", {"CLAW_BASNA_WORKER": "1"}), prior=None, caller="")
        assert got["parent"] == "glasses" and got["run"] == "basna:abcd1234"
        got = fam.for_spawn(_Cfg("vatra-s9-x", {"CLAW_VATRA_WORKER": "1", "CLAW_VATRA_SESSION": "s9"}),
                            prior=None, caller="")
        assert "parent" not in got  # a run the person started


def test_a_bundle_carries_made_for_not_the_parent():
    from captain_claw.flight_deck.agent_bundle.restore import spawn_config

    cfg = spawn_config({"name": "Kid", "made_for": "checks facts", "parent": "top"}, "Kid")
    assert cfg.made_for == "checks facts"
    assert fam.for_spawn(cfg, prior=None, caller="").get("parent") is None
