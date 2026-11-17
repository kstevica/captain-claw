"""Agent family (docs/agent-family-plan.md, Phase 1): who made which agent and
where it sits now.

* The pure rules in ``agent_family``: roles of old entries, parents that are
  gone or someone else's, the limits on a move (loop, depth, children, workers,
  owners), moving up when a parent leaves, and what a (re)spawn records.
* ``POST /fd/spawn-process`` end to end (Popen faked): an agent's spawn makes
  its child, a re-spawn keeps its place, the family limits refuse before
  anything is written, ``/fd/processes`` shows the family.
* ``PUT /fd/processes/{slug}/parent``: the owner moves agents, with reasons.
* Removing a parent moves what sat under it up a level.
* Bat's workers: First Mate's agent for a factory's run, the starting agent
  for an agent's run; ``bat-bat-…`` is a worker by name too.
* The agent-side ``spawn_agent`` sends ``made_for`` and stops on a 409.
"""

from __future__ import annotations

import httpx
import pytest

from captain_claw.flight_deck import agent_family as fam
from captain_claw.flight_deck import server
from captain_claw.flight_deck.agent_sharing import is_managed_agent
from captain_claw.flight_deck.auth import create_access_token

from .test_spawn_owner_integrity import (  # noqa: F401 — fixtures
    ALICE, BOB, LOOPBACK, REMOTE, _FakePopen, deck, fd_db, spawn_env, users,
)


def _reg(**entries) -> dict:
    """slug=(owner, parent[, role]) → a registry dict."""
    out = {}
    for slug, spec in entries.items():
        owner, parent, *rest = spec
        e = {"slug": slug, "name": slug.title(), "owner": owner}
        if parent:
            e["parent"] = parent
        if rest:
            e["role"] = rest[0]
        out[slug] = e
    return out


def _chain(n: int, owner: str = "u") -> dict:
    """a0 ← a1 ← … ← a(n-1): a(n-1) is n-1 levels down."""
    return _reg(**{f"a{i}": (owner, f"a{i - 1}" if i else "") for i in range(n)})


# ── the rules ────────────────────────────────────────────────────────


class TestRules:
    def test_old_entries_are_standing_unless_named_like_a_worker(self):
        assert fam.role_of("glasses", {}) == fam.STANDING
        assert fam.role_of("bat-bat-ab12-research", {"description": "Bat worker"}) == fam.WORKER
        assert fam.role_of("basna-0123abcd-x-y", {}) == fam.WORKER
        assert fam.role_of("glasses", {"role": "worker"}) == fam.WORKER

    def test_bat_workers_are_managed_and_a_bat_named_agent_is_not(self):
        assert is_managed_agent("bat-bat-ab12-step-1", "Bat worker")
        assert not is_managed_agent("bat-cave", "")
        assert not is_managed_agent("bat-bat-cave-notes", "")

    def test_a_parent_that_is_gone_or_someone_elses_counts_as_none(self):
        reg = _reg(kid=("u", "gone"), other=("v", ""), kid2=("u", "other"), me=("u", "me"))
        assert fam.parent_of(reg, "kid") == ""
        assert fam.parent_of(reg, "kid2") == ""
        assert fam.parent_of(reg, "me") == ""

    def test_with_sign_in_off_one_persons_owner_values_all_match(self, monkeypatch):
        reg = _reg(top=("", ""), kid=("local", "top"))
        assert fam.parent_of(reg, "kid") == ""
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        assert fam.parent_of(reg, "kid") == "top"
        fam.check_move(reg, "top", "")

    def test_children_depth_height_descendants(self):
        reg = _reg(top=("u", ""), a=("u", "top"), b=("u", "top"), w=("u", "top", "worker"),
                   aa=("u", "a"))
        assert sorted(fam.children(reg, "top")) == ["a", "b"]
        assert sorted(fam.children(reg, "top", workers=True)) == ["a", "b", "w"]
        assert fam.depth(reg, "aa") == 2 and fam.depth(reg, "top") == 0
        assert fam.height(reg, "top") == 2 and fam.height(reg, "b") == 0
        assert fam.descendants(reg, "top") == {"a", "b", "w", "aa"}
        assert fam.ancestors(reg, "aa") == ["a", "top"]

    def test_a_loop_in_old_data_never_hangs(self):
        reg = _reg(x=("u", "y"), y=("u", "x"))
        assert fam.ancestors(reg, "x") == ["y"]
        assert fam.depth(reg, "x") == 1

    def test_move_refusals_say_why(self):
        reg = _reg(top=("u", ""), kid=("u", "top"), w=("u", "", "worker"), theirs=("v", ""))
        cases = [
            ("w", "top", "worker can't be moved"),
            ("top", "w", "worker can't have agents"),
            ("top", "top", "under itself"),
            ("top", "kid", "loop"),
            ("top", "theirs", "have to be yours"),
        ]
        for slug, parent, why in cases:
            with pytest.raises(fam.FamilyError, match=why):
                fam.check_move(reg, slug, parent)
        with pytest.raises(fam.FamilyError) as e:
            fam.check_move(reg, "nobody", "")
        assert e.value.status == 404

    def test_depth_counts_what_moves_along(self):
        reg = _chain(4)  # a3 is 3 levels down: the deepest allowed
        reg.update(_reg(solo=("u", ""), pair=("u", ""), pair_kid=("u", "pair")))
        fam.check_move(reg, "solo", "a2")        # becomes level 3
        with pytest.raises(fam.FamilyError, match="past 3 levels"):
            fam.check_move(reg, "solo", "a3")    # would be level 4
        with pytest.raises(fam.FamilyError, match="past 3 levels"):
            fam.check_move(reg, "pair", "a2")    # its child would be level 4
        fam.check_move(reg, "pair", "a1")

    def test_at_most_twelve_standing_agents_under_one(self):
        reg = _reg(top=("u", ""), new=("u", ""), w=("u", "top", "worker"))
        reg.update(_reg(**{f"k{i}": ("u", "top") for i in range(fam.MAX_CHILDREN)}))
        with pytest.raises(fam.FamilyError, match="already has 12"):
            fam.check_move(reg, "new", "top")
        fam.check_move(reg, "k0", "top")  # already there: not one more
        with pytest.raises(fam.FamilyError, match="already has 12"):
            fam.check_child(reg, "top")

    def test_an_agent_three_levels_down_makes_no_agents(self):
        reg = _chain(4)
        fam.check_child(reg, "a2")
        with pytest.raises(fam.FamilyError, match="3 levels down"):
            fam.check_child(reg, "a3")

    def test_move_and_back(self):
        reg = _reg(top=("u", ""), kid=("u", "", ))
        reg["kid"]["made_by"] = "top"
        got = fam.move(reg, "kid", "top", made_for="  drafts   follow-ups ")
        assert got["parent"] == "top" and got["made_for"] == "drafts follow-ups"
        assert reg["kid"]["parent_set_by"] == "person" and reg["kid"]["role"] == fam.STANDING
        got = fam.move(reg, "kid", "")
        assert got["parent"] == "" and "parent" not in reg["kid"]
        assert got["made_for"] == "drafts follow-ups"     # None leaves it
        assert reg["kid"]["made_by"] == "top"              # never changes
        fam.move(reg, "kid", "", made_for="")
        assert "made_for" not in reg["kid"]

    def test_removing_a_parent_moves_its_agents_up(self):
        reg = _reg(top=("u", ""), mid=("u", "top"), a=("u", "mid"), w=("u", "mid", "worker"),
                   b=("u", ""))
        assert sorted(fam.on_removed(reg, "mid")) == ["a", "w"]
        reg.pop("mid")
        assert reg["a"]["parent"] == "top" and reg["w"]["parent"] == "top"
        reg2 = _reg(top=("u", ""), a=("u", "top"))
        fam.on_removed(reg2, "top")
        assert "parent" not in reg2["a"]


class _Cfg:
    def __init__(self, made_for="", env=None, family=None):
        self.made_for = made_for
        self.env_vars = [{"key": k, "value": v} for k, v in (env or {}).items()]
        self._family = family or {}


class TestForSpawn:
    def test_a_new_agent_an_agent_makes_is_its_child(self):
        got = fam.for_spawn(_Cfg("checks facts"), prior=None, caller="glasses")
        assert got["parent"] == got["made_by"] == "glasses"
        assert got["parent_set_by"] == "spawn" and got["role"] == fam.STANDING
        assert got["made_for"] == "checks facts" and got["made_at"].endswith("Z")

    def test_a_maker_cant_hide_its_child_as_a_worker(self):
        got = fam.for_spawn(_Cfg(env={"CLAW_BAT_WORKER": "1"}), prior=None, caller="glasses")
        assert got["role"] == fam.STANDING and got["parent"] == "glasses"

    def test_a_runs_worker(self):
        got = fam.for_spawn(_Cfg(env={"CLAW_VATRA_WORKER": "1", "CLAW_VATRA_SESSION": "s1"}),
                            prior=None, caller="")
        assert got["role"] == fam.WORKER and got["run"] == "vatra:s1" and "parent" not in got
        prior = {"role": "worker", "run": "vatra:old", "made_at": "2026-01-01T00:00:00Z"}
        got = fam.for_spawn(_Cfg(env={"CLAW_VATRA_WORKER": "1", "CLAW_VATRA_SESSION": "s2"}),
                            prior=prior, caller="")
        assert got["run"] == "vatra:s2" and got["made_at"] == "2026-01-01T00:00:00Z"

    def test_a_respawn_keeps_its_place_whoever_restarts_it(self):
        prior = {"parent": "top", "made_by": "top", "made_for": "x", "parent_set_by": "person",
                 "role": "standing", "made_at": "2026-01-01T00:00:00Z"}
        got = fam.for_spawn(_Cfg(), prior=prior, caller="someone-else")
        assert got == prior

    def test_fd_code_says_first(self):
        got = fam.for_spawn(_Cfg(family={"role": "worker", "run": "bat:1", "bogus": "x"}),
                            prior={"parent": "p"}, caller="c")
        assert got["role"] == "worker" and got["run"] == "bat:1" and "parent" not in got
        assert "bogus" not in got

    def test_ones_own(self):
        got = fam.for_spawn(_Cfg(), prior=None, caller="")
        assert set(got) == {"role", "made_at"} and got["role"] == fam.STANDING


# ── through the deck ─────────────────────────────────────────────────


def _client(addr=REMOTE) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=addr),
                             base_url="http://fd.test")


def _bob() -> dict:
    return {"Authorization": f"Bearer {create_access_token(BOB)}"}


def _family(slug: str) -> dict:
    return fam.view(server._load_process_registry(), slug)


class TestSpawnRoute:
    async def test_an_agents_spawn_makes_its_child(self, spawn_env, users):
        async with _client(LOOPBACK) as c:
            r = await c.post("/fd/spawn-process", json={"name": "kid", "made_for": "drafts follow-ups"},
                             headers={"X-Agent-Auth": "bob-agent-token"})
            assert r.status_code == 200, r.text
            procs = (await c.get("/fd/processes", headers=_bob())).json()
        kid = next(p for p in procs if p["slug"] == "kid")
        assert kid["parent"] == "bob-agent" and kid["made_by"] == "bob-agent"
        assert kid["made_for"] == "drafts follow-ups" and kid["role"] == "standing"
        bob = next(p for p in procs if p["slug"] == "bob-agent")
        assert bob["parent"] == "" and bob["role"] == "standing"

    async def test_the_owners_own_spawn_is_top_level_and_a_respawn_keeps_its_place(self, spawn_env, users,
                                                                                    monkeypatch):
        async def _no_cap(*a, **k):
            return None

        monkeypatch.setattr(server, "check_agent_count_limit", _no_cap)  # the free plan's 2 agents
        reg = server._load_process_registry()
        reg["alice-kid"] = {"slug": "alice-kid", "name": "alice-kid", "web_port": 24120, "web_auth": "ak",
                            "owner": ALICE, "parent": "alice-agent", "made_by": "alice-agent",
                            "role": "standing"}
        server._save_process_registry(reg)
        alice = {"Authorization": f"Bearer {create_access_token(ALICE)}"}
        async with _client() as c:
            r = await c.post("/fd/spawn-process", json={"name": "fresh"}, headers=alice)
            assert r.status_code == 200, r.text
            r = await c.post("/fd/spawn-process", json={"name": "alice-kid"}, headers=alice)
            assert r.status_code == 200, r.text
        assert _family("fresh")["parent"] == ""
        assert _family("alice-kid")["parent"] == "alice-agent"

    async def test_past_the_limits_nothing_is_written(self, spawn_env, users):
        reg = server._load_process_registry()
        for i, parent in enumerate(["", "l0", "l1"]):
            reg[f"l{i}"] = {"slug": f"l{i}", "name": f"l{i}", "web_port": 24130 + i, "web_auth": f"l{i}-t",
                            "owner": BOB, **({"parent": parent} if parent else {})}
        reg["bob-agent"]["parent"] = "l2"   # bob-agent is 3 levels down now
        server._save_process_registry(reg)
        async with _client(LOOPBACK) as c:
            r = await c.post("/fd/spawn-process", json={"name": "too-deep"},
                             headers={"X-Agent-Auth": "bob-agent-token"})
        assert r.status_code == 409 and "3 levels down" in r.json()["detail"]
        assert "too-deep" not in server._load_process_registry()
        assert not (spawn_env / "too-deep").exists() and _FakePopen.calls == []


class TestMoveRoute:
    @pytest.fixture
    def family(self, spawn_env, users):
        reg = server._load_process_registry()
        for slug, parent in (("glasses", ""), ("digest", "")):
            reg[slug] = {"slug": slug, "name": slug.title(), "web_port": 24200, "web_auth": f"{slug}-t",
                         "owner": BOB, **({"parent": parent} if parent else {})}
        server._save_process_registry(reg)
        return reg

    async def test_drag_under_and_back_out(self, family):
        async with _client() as c:
            r = await c.put("/fd/processes/digest/parent", json={"parent": "glasses", "made_for": "Monday digest"},
                            headers=_bob())
            assert r.status_code == 200, r.text
            assert r.json() == {"ok": True, "slug": "digest", "parent": "glasses", "made_for": "Monday digest",
                                "message": "Digest is now under Glasses"}
            assert server._load_process_registry()["digest"]["parent_set_by"] == "person"
            r = await c.put("/fd/processes/glasses/parent", json={"parent": "digest"}, headers=_bob())
            assert r.status_code == 409 and "loop" in r.json()["detail"]
            r = await c.put("/fd/processes/digest/parent", json={"parent": ""}, headers=_bob())
            assert r.status_code == 200 and r.json()["parent"] == ""
            assert r.json()["message"] == "Digest is one of your own agents again"

    async def test_never_someone_elses_agent(self, family):
        async with _client() as c:
            r = await c.put("/fd/processes/digest/parent", json={"parent": "alice-agent"}, headers=_bob())
            assert r.status_code == 404
            r = await c.put("/fd/processes/alice-agent/parent", json={"parent": "glasses"}, headers=_bob())
            assert r.status_code == 404
        assert "parent" not in server._load_process_registry()["alice-agent"]

    async def test_removing_a_parent_moves_its_agents_up(self, family):
        async with _client() as c:
            await c.put("/fd/processes/digest/parent", json={"parent": "glasses"}, headers=_bob())
            await c.put("/fd/processes/glasses/parent", json={"parent": "bob-agent"}, headers=_bob())
            r = await c.delete("/fd/processes/glasses", headers=_bob())
            assert r.status_code == 200, r.text
        assert _family("digest")["parent"] == "bob-agent"


# ── Bat's workers ───────────────────────────────────────────────────


class TestBatWorkers:
    @pytest.fixture
    def reg(self, deck):
        reg = server._load_process_registry()
        reg["bat-bat-ab12-research"] = {"slug": "bat-bat-ab12-research", "name": "bat-bat_ab12-research",
                                        "description": "Bat worker", "web_port": 24301, "owner": BOB}
        reg["fm"] = {"slug": "fm", "name": "First Mate", "web_port": 24302, "owner": BOB}
        server._save_process_registry(reg)
        return reg

    async def test_a_factory_runs_worker_is_first_mates(self, reg, monkeypatch):
        from captain_claw.flight_deck import bat_routes
        from captain_claw.flight_deck import first_mate_store as fms

        class _Store:
            def get(self, fid):
                return {"title": "Inbox triage"} if fid == "fac_1" else None

            def prefs(self, owner):
                return {"agent": "fm"} if owner == fms.norm_user(BOB) else {"agent": ""}

        monkeypatch.setattr(fms, "get_store", lambda: _Store())
        run = {"id": "bat_ab12cd", "owner_id": BOB, "config": {"source": "first_mate", "factory_id": "fac_1"}}
        await bat_routes._label_worker("bat-bat-ab12-research", run, "find-sources")
        got = _family("bat-bat-ab12-research")
        assert got == {"parent": "fm", "made_by": "first_mate", "made_for": "Inbox triage · find sources",
                       "role": "worker", "run": "bat:bat_ab12cd", "factory": "fac_1"}

    async def test_an_agents_run_hangs_under_that_agent(self, reg):
        from captain_claw.flight_deck import bat_routes

        run = {"id": "bat_ab12cd", "owner_id": BOB, "source_port": 24102,
               "config": {"source": "agent"}}
        await bat_routes._label_worker("bat-bat-ab12-research", run, "write")
        got = _family("bat-bat-ab12-research")
        assert got["parent"] == "bob-agent" and got["made_by"] == "bob-agent"
        assert got["role"] == "worker" and got["made_for"] == "write"
        # Alice's agent on that port would never be taken for Bob's run's parent.
        run["source_port"] = 24101
        await bat_routes._label_worker("bat-bat-ab12-research", run, "write")
        assert _family("bat-bat-ab12-research")["parent"] == ""

    async def test_a_worker_that_is_gone_is_left_alone(self, reg):
        from captain_claw.flight_deck import bat_routes

        await bat_routes._label_worker("nobody", {"id": "bat_x", "owner_id": BOB, "config": {}}, "s")
        assert "nobody" not in server._load_process_registry()


# ── the agent-side tool ─────────────────────────────────────────────


@pytest.fixture
def bob_agent(spawn_env, monkeypatch, tmp_path):
    """This agent IS bob-agent, its tool's requests go into the deck (as
    test_spawn_owner_integrity's TestFlightDeckToolSpawn)."""
    from captain_claw import config as config_mod
    from captain_claw.config import Config
    from captain_claw.flight_deck import agent_secret

    cfg = Config()
    cfg.web.auth_token = "bob-agent-token"
    monkeypatch.setattr(config_mod, "_config", cfg)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    monkeypatch.setenv("FD_LOCKDOWN", "1")
    monkeypatch.setenv("FD_OWNER_ID", BOB)
    monkeypatch.setenv("FD_URL", "http://fd.test")
    agent_secret.reset_cache_for_tests()
    real_client = httpx.AsyncClient

    def _to_fd(*args, **kwargs):
        return real_client(transport=httpx.ASGITransport(app=server.app, client=LOOPBACK),
                           timeout=30.0)

    monkeypatch.setattr(httpx, "AsyncClient", _to_fd)
    yield cfg
    agent_secret.reset_cache_for_tests()


class TestTool:
    @pytest.fixture
    def agent(self, bob_agent):
        return bob_agent

    async def test_made_for_reaches_the_deck(self, agent, users):
        from captain_claw.tools.flight_deck import FlightDeckTool

        res = await FlightDeckTool()._spawn_agent("http://fd.test", "kid", made_for="checks the numbers")
        assert res.success, res.error
        got = _family("kid")
        assert got["parent"] == "bob-agent" and got["made_for"] == "checks the numbers"

    async def test_a_family_refusal_is_not_retried_through_docker(self, agent, users):
        from captain_claw.tools.flight_deck import FlightDeckTool

        reg = server._load_process_registry()
        for i in range(fam.MAX_CHILDREN):
            reg[f"k{i}"] = {"slug": f"k{i}", "name": f"k{i}", "web_port": 24400 + i, "owner": BOB,
                            "parent": "bob-agent"}
        server._save_process_registry(reg)
        res = await FlightDeckTool()._spawn_agent("http://fd.test", "kid")
        # The deck's own reason, not what the Docker endpoint would have said.
        assert not res.success and "already has 12" in res.error
        assert "kid" not in server._load_process_registry()
