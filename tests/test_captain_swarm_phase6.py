"""Captain Swarm, Phase 6: learning (docs/captain-swarm-plan.md §12) — families,
verdicts a proposer may read, strategy weights, lessons written when a run
ends and read by the next run of the same kind of task, the planner's
readings, the strategy round on a stall, a playbook from a one-task run, and
the routes that show and forget lessons."""

from __future__ import annotations

import asyncio

import pytest

from captain_claw.flight_deck import captain_swarm_learning as learning
from captain_claw.flight_deck import captain_swarm_loop as loop
from captain_claw.flight_deck import captain_swarm_objective as om
from captain_claw.flight_deck import captain_swarm_outcome as outcome
from captain_claw.flight_deck import captain_swarm_rules as rules
from captain_claw.flight_deck.captain_swarm_store import CaptainSwarmStore

BASE, H1, H2, H3 = "b" * 40, "1" * 40, "2" * 40, "3" * 40


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def store(tmp_path):
    s = CaptainSwarmStore(tmp_path / "cs.db")
    loop.set_store(s)
    yield s
    loop.set_store(None)
    run(s.close())


# ── families and what a proposer may read ────────────────────────────


def test_every_kind_has_a_family(tmp_path):
    assert outcome.normalize(outcome.outcome_template(task="Find the cheapest new copy of The Book"))["family"] == "find-price"
    assert outcome.normalize({**outcome.outcome_template(task="Find the cheapest new copy of The Book"),
                              "family": "Book Prices!"})["family"] == "book-prices"
    assert rules.normalize(rules.triage_template(set_id="mt-1", agent="helper"))["family"] == "mail-triage"
    assert om.mrav_template("/r")["family"] == "mrav"
    assert learning.family_of({"kind": "code"}) == "code"
    assert learning.family_of({"subject": {"family": "mail-triage"}, "kind": "practice"}) == "mail-triage"


@pytest.mark.parametrize("status,verdict,shown", [
    ("kept", "0.900 ± 0.010 beats the best 0.800 ± 0.010; held-out 0.850 against the best's 0.700",
     "0.900 ± 0.010 beats the best 0.800 ± 0.010; the held-out cases confirmed it"),
    ("discarded", "0.9 ± 0 beats the best 0.8 ± 0, but held-out 0.400 against the best's 0.700: the gain didn't hold",
     "0.9 ± 0 beats the best 0.8 ± 0, but the held-out cases didn't confirm it"),
    ("disqualified", "it names held-out mail it was never shown: 'zed@secret.example'",
     "it names held-out mail it was never shown"),
    ("disqualified", "out of bounds: evals/x.json is protected", "out of bounds: evals/x.json is protected"),
    ("discarded", "0.61 ± 0 against the best 0.60 ± 0: not a gain of at least 0.050",
     "0.61 ± 0 against the best 0.60 ± 0: not a gain of at least 0.050"),
])
def test_shown_verdict_never_shows_held_out_numbers_or_contents(status, verdict, shown):
    assert learning.shown_verdict({"status": status, "verdict_reason": verdict}) == shown


@pytest.mark.parametrize("status,verdict,result", [
    ("kept", "", "won"),
    ("discarded", f"0.7 beats the best 0.6, but #1.1 {learning.RUNNER_UP}; it comes back", "won"),
    ("discarded", "not a gain", "lost"),
    ("disqualified", "out of bounds", "lost"),
    ("failed", "the lane didn't start", ""),
])
def test_outcome_of(status, verdict, result):
    assert learning.outcome_of({"status": status, "verdict_reason": verdict}) == result


def test_weight_shrinks_and_counts_losses_double():
    assert learning.weight(0, 0) == pytest.approx(0.5)
    assert learning.weight(3, 0) > learning.weight(1, 0) > 0.5
    assert learning.weight(1, 1) < learning.weight(1, 0)
    assert learning.weight(2, 1) < learning.weight(1, 0)          # one loss outweighs one more win
    assert 0.05 <= learning.weight(0, 50) < 0.1


def test_parse_reading_and_direction():
    text = "READING: official reseller lists beat aggregators here; kits keep sneaking in.\n[{\"hypothesis\": \"x\"}]"
    assert learning.parse_reading(text).startswith("official reseller lists beat aggregators")
    assert learning.parse_reading("**Reading:** short") == "short"
    assert learning.parse_reading('[{"hypothesis": "x"}]') == ""
    assert loop.parse_proposals(text, 3)[0]["hypothesis"] == "x"            # the reading doesn't get in the way
    assert learning.parse_direction('{"direction": "try distributors", "why": "nobody has"}') == \
        {"direction": "try distributors", "why": "nobody has"}
    assert learning.parse_direction('{"why": "x"}') is None


# ── lessons written down, and read back ──────────────────────────────


async def _run_with(store, run_id, family, exps, *, owner="u", status="stalled"):
    obj = {"kind": "outcome", "family": family, "title": "T", "goal": "G",
           "metric": {"read": "best_verified", "direction": "min"}, "subject": {}}
    await store.create_run(run_id=run_id, owner_id=owner, title="T", objective=obj, status=status)
    for i, (tag, st, verdict) in enumerate(exps, 1):
        eid = f"exp_{run_id[-4:]}{i:08d}"
        await store.add_experiment(exp_id=eid, run_id=run_id, round_no=1, n=i, hypothesis=f"way {tag}", strategy_tag=tag)
        await store.update_experiment(eid, status=st, verdict_reason=verdict, lesson=f"lesson {tag}")
    return await store.get_run(run_id)


def test_record_run_once_then_the_next_run_reads_it(store):
    async def go():
        first = await _run_with(store, "cs_000000000001", "find-price", [
            ("resellers", "kept", "189 beats the best —; held-out 189.000 against the best's —"),
            ("aggregators", "discarded", f"198 beats the best —, but #1.1 {learning.RUNNER_UP}"),
            ("distributors", "discarded", "250 against the best 189: not a gain"),
            ("broken", "failed", "the lane didn't start"),
            ("live", "running", ""),
        ])
        n = await learning.record_run(store, first)
        again = await learning.record_run(store, await store.get_run("cs_000000000001"))
        await store.add_lesson(owner_id="u", family="find-price", run_id="cs_000000000001", kind="reading",
                               text="official lists beat aggregators")
        await _run_with(store, "cs_000000000009", "other-family", [("resellers", "discarded", "no")])
        await learning.record_run(store, await store.get_run("cs_000000000009"))
        second = await _run_with(store, "cs_000000000002", "find-price", [], status="running")
        await store.add_lesson(owner_id="u", family="find-price", run_id="cs_000000000002", kind="strategy",
                               text="try the manufacturer's own shop")
        brief = await learning.brief_block(store, second)
        w = await learning.weights(store, "u", "find-price")
        stranger = await _run_with(store, "cs_000000000003", "find-price", [], owner="v", status="running")
        return n, again, brief, w, await learning.brief_block(store, stranger)
    n, again, brief, w, strangers_brief = run(go())
    assert n == 4 and again == 0                                              # the running one isn't judged yet
    assert "worked (resellers): way resellers → 189 beats the best —; the held-out cases confirmed it" in brief
    assert "worked (aggregators)" in brief and "didn't (distributors)" in brief and "broken" not in brief
    assert "noted: official lists beat aggregators" in brief
    assert "a strategy round suggested" in brief and "try the manufacturer's own shop" in brief
    assert "held-out 189" not in brief and "other-family" not in brief
    assert w == {"resellers": {"won": 1, "lost": 0, "weight": pytest.approx(learning.weight(1, 0), abs=1e-3)},
                 "aggregators": {"won": 1, "lost": 0, "weight": pytest.approx(learning.weight(1, 0), abs=1e-3)},
                 "distributors": {"won": 0, "lost": 1, "weight": pytest.approx(learning.weight(0, 1), abs=1e-3)}}
    assert strangers_brief == ""                                              # lessons are the owner's own


def test_rank_puts_winning_strategies_first(store):
    async def go():
        await _run_with(store, "cs_000000000001", "find-price",
                        [("resellers", "kept", ""), ("aggregators", "discarded", "no"), ("aggregators", "discarded", "no")])
        await learning.record_run(store, await store.get_run("cs_000000000001"))
        now = await _run_with(store, "cs_000000000002", "find-price", [], status="running")
        ideas = [{"hypothesis": "a", "strategy_tag": "aggregators"}, {"hypothesis": "n", "strategy_tag": "new-idea"},
                 {"hypothesis": "r", "strategy_tag": "resellers"}]
        return [i["hypothesis"] for i in await learning.rank(store, now, ideas)]
    assert run(go()) == ["r", "n", "a"]


def test_the_proposer_reads_and_its_reading_is_kept(store, monkeypatch):
    seen = {}

    async def fake_ask(run_, system, user, **kw):
        seen["system"], seen["user"] = system, user
        return 'READING: aggregators list kits first.\n[{"hypothesis": "resellers", "strategy_tag": "resellers"}]'
    monkeypatch.setattr(loop, "ask_model", fake_ask)

    async def go():
        r = await _run_with(store, "cs_000000000002", "find-price", [], status="running")
        r = {**r, "objective": outcome.normalize({**outcome.outcome_template(task="Find the cheapest copy of The Book"),
                                                   "family": "find-price"}), "learned": "\nFrom earlier runs: X"}
        ideas = await outcome.propose(r, [], 2)
        return ideas, await store.list_lessons("u", "find-price", run_id="cs_000000000002"), \
            [e["message"] for e in await store.list_events("cs_000000000002")]
    ideas, kept, events = run(go())
    assert ideas[0]["hypothesis"] == "resellers"
    assert seen["system"].endswith(learning.READING_ASK) and "From earlier runs: X" in seen["user"]
    assert [(k["kind"], k["text"]) for k in kept] == [("reading", "aggregators list kits first.")]
    assert any("the planner's reading" in m for m in events)


def test_strategy_round_reads_three_ways_on_the_panel(monkeypatch):
    asked = []

    async def fake_ask(run_, system, user, **kw):
        asked.append((user.split("\n")[0], kw.get("panel"), kw.get("label")))
        if "nobody has tried" in user:
            raise RuntimeError("one lens failed")
        return '{"direction": "go to the manufacturer", "why": "nobody did"}'
    monkeypatch.setattr(loop, "ask_model", fake_ask)
    r = {"objective": {"title": "T", "goal": "", "metric": {"read": "x", "direction": "max"}}, "champion": {}}
    got = run(learning.strategy_round(r, [{"status": "discarded", "strategy_tag": "a", "hypothesis": "h",
                                          "verdict_reason": "held-out 0.1 against the best's 0.2: the gain didn't hold"}]))
    assert len(asked) == 3 and all(p is True and lbl == "strategy" for _, p, lbl in asked)
    assert len(got) == 2 and got[0]["direction"] == "go to the manufacturer"


# ── the driver ───────────────────────────────────────────────────────


class Lane:
    def __init__(self, heads):
        self.heads, self.started = heads, []

    async def start(self, run_, exp, brief):
        self.started.append(exp["hypothesis"])
        return {"child_id": f"c{len(self.started)}", "branch": ""}

    async def poll(self, run_, exp):
        return {"state": "done", "commits": 1, **self.heads[exp["hypothesis"]]}

    async def stop(self, run_, exp):
        return None

    async def discard(self, run_, exp):
        return None


def objective(**lane):
    return {"title": "T", "goal": "G", "kind": "code", "family": "mrav",
            "subject": {"repo_path": "/r", "editable": ["a/**"], "protected": [], "snapshot": {}},
            "metric": {"run": "x {checkout}", "read": "pass_rate", "spread": "", "per_case": "", "direction": "max",
                       "timeout_s": 60, "dev_file": "", "env": []},
            "holdout": {"run": "x {checkout}", "file": "/h"}, "guardrails": [], "noise": {"min_gain": 0.05},
            "lane": {"kind": "crew", "harness": "claude", "model": "", "model_source": None, "per_round": 2,
                     "parallel": 1, **lane},
            "budget": {"usd": 50.0, "hours": 8.0, "max_experiments": 40},
            "stop": {"target": None, "stall_rounds": 1, "remeasure_every": 50}}


@pytest.fixture()
def seams(store, monkeypatch):
    monkeypatch.setattr(loop, "_POLL_S", 0)
    ns = type("NS", (), {})()
    ns.scores = {(BASE, "dev"): 0.6, (BASE, "holdout"): 0.5}
    ns.ideas: list[list[dict]] = []
    ns.briefs: list[str] = []
    ns.strategy_calls = 0

    async def measure(run_, sha, which):
        v = ns.scores.get((sha, which))
        if v is None:
            return {"value": None, "error": "no score"}
        return {"value": v, "stats": {"n": 3, "mean": v, "stdev": 0.0, "low": v, "high": v}, "guard": {}, "per_case": {}}

    async def nothing(*a):
        return None

    async def screen_(run_, exp):
        return {"ok": True, "reason": "", "changed": [], "diff_stat": "", "hits": []}

    async def propose_(run_, ledger, k):
        ns.briefs.append(run_.get("learned") or "")
        return ns.ideas.pop(0) if ns.ideas else []

    async def review_(run_, exp):
        return {"ok": True, "reason": "general"}

    async def strategy_(run_, ledger):
        ns.strategy_calls += 1
        return [{"direction": "change the digest instead", "why": "every idea tuned the prompt", "lens": "x"}]

    loop.set_measure(measure)
    loop.set_seal(nothing)
    loop.set_branch(nothing)
    loop.set_screen(screen_)
    loop.set_proposer(propose_)
    loop.set_review(review_)
    loop.set_strategy(strategy_)

    async def drive(run_id, *, lane, obj=None):
        loop.set_lane(lane)
        await store.create_run(run_id=run_id, owner_id="u", title="T", objective=obj or objective(), status="baseline")
        await store.update_run(run_id, state={"base_sha": BASE})
        status = await loop.Driver(store, run_id).drive()
        return status, await store.get_run(run_id), await store.list_experiments(run_id)
    ns.drive = drive
    yield ns
    for setter in (loop.set_measure, loop.set_seal, loop.set_branch, loop.set_screen, loop.set_proposer,
                   loop.set_review, loop.set_strategy, loop.set_lane):
        setter(None)
    loop._SLOTS.clear()


def idea(h, tag):
    return {"hypothesis": h, "strategy_tag": tag, "files": [], "why": ""}


def test_a_second_run_starts_from_the_first(seams, store):
    ns = seams
    ns.scores.update({(H1, "dev"): 0.8, (H1, "holdout"): 0.7, (H2, "dev"): 0.6})
    ns.ideas = [[idea("shorter digest", "digest"), idea("louder prompt", "prompt")]]

    async def go():
        first = await ns.drive("cs_000000000001", lane=Lane({"shorter digest": {"head_sha": H1},
                                                            "louder prompt": {"head_sha": H2}}))
        ns.ideas = [[idea("louder prompt again", "prompt"), idea("digest, another way", "digest")]]
        lane = Lane({"louder prompt again": {"head_sha": H2}, "digest, another way": {"head_sha": H3}})
        second = await ns.drive("cs_000000000002", lane=lane)
        return first, second, lane
    first, second, lane = run(go())
    assert first[1]["state"]["learned"] == 2
    brief = ns.briefs[-1] if len(ns.briefs) == 2 else next(b for b in ns.briefs[1:] if "earlier runs" in b)
    assert "From earlier runs on this kind of task (mrav)" in brief
    assert "worked (digest): shorter digest" in brief and "didn't (prompt): louder prompt" in brief
    assert "digest 1/1" in brief and "prompt 0/1" in brief
    assert lane.started[0] == "digest, another way"                     # the strategy that won goes first


def test_a_stall_brings_a_strategy_round_and_two_more_rounds(seams, store):
    ns = seams
    ns.scores.update({(H1, "dev"): 0.6, (H2, "dev"): 0.6, (H3, "dev"): 0.6})
    ns.ideas = [[idea("a", "a")], [idea("b", "b")], [idea("c", "c")], [idea("d", "d")]]

    async def go():
        return await ns.drive("cs_000000000001", lane=Lane({"a": {"head_sha": H1}, "b": {"head_sha": H2},
                                                           "c": {"head_sha": H3}}),
                              obj=objective(per_round=1))
    status, r, exps = run(go())
    assert status == "stalled" and "3 rounds in a row" in r["stopped_reason"] and ns.strategy_calls == 1
    assert [e["hypothesis"] for e in exps] == ["a", "b", "c"] and ns.ideas == [[idea("d", "d")]]
    assert "change the digest instead" in ns.briefs[1] and "change the digest instead" in ns.briefs[2]
    assert "change the digest instead" not in ns.briefs[0]


def test_nothing_new_to_try_brings_a_strategy_round_and_one_more_ask(seams, store):
    ns = seams
    ns.scores.update({(H1, "dev"): 0.8, (H1, "holdout"): 0.7})
    ns.ideas = [[], [idea("from the new direction", "digest")]]

    async def go():
        return await ns.drive("cs_000000000001", lane=Lane({"from the new direction": {"head_sha": H1}}))
    status, r, exps = run(go())
    assert [e["status"] for e in exps] == ["kept"] and r["champion"]["sha"] == H1
    # One before the keep; after a gain the next stall earns another, and then it stops.
    assert ns.strategy_calls == 2 and status == "stalled" and "nothing new" in r["stopped_reason"]


def test_a_cancelled_run_is_remembered_too(seams, store):
    async def go():
        await store.create_run(run_id="cs_000000000001", owner_id="u", title="T", objective=objective(), status="running")
        await store.add_experiment(exp_id="exp_000000000001", run_id="cs_000000000001", round_no=1, n=1,
                                   hypothesis="h", strategy_tag="t")
        await store.update_experiment("exp_000000000001", status="discarded", verdict_reason="not a gain")
        await loop.cancel_run(store, "cs_000000000001")
        return await store.list_lessons("u", "mrav")
    lessons = run(go())
    assert [(x["kind"], x["strategy_tag"], x["outcome"]) for x in lessons] == [("attempt", "t", "lost")]


# ── a playbook, and the routes ───────────────────────────────────────


def _outcome_run(store, status="stalled"):
    obj = outcome.normalize(outcome.outcome_template(task="Find the lowest price for a new bare Raspberry Pi 5 8GB"))
    best = {"title": "Raspberry Pi 5, 8GB RAM", "url": "https://shop.example/pi5", "value": 189.0, "currency": "EUR"}

    async def go():
        await store.create_run(run_id="cs_000000000001", owner_id="u", title=obj["title"], objective=obj, status=status)
        await store.update_run("cs_000000000001", champion={"sha": "v_1", "exp": "exp_000000000001",
                                                            "dev": {"value": 189.0, "best": best}})
        for i, (h, tag, st, v) in enumerate((
                ("official resellers", "resellers", "kept", "189 beats the best —"),
                ("comparison sites", "aggregators", "discarded", f"198.74 beats the best —, but #1.1 {learning.RUNNER_UP}"),
                ("distributors by part number", "distributors", "discarded",
                 "250 against the best 189, but held-out 250 against the best's 189: the gain didn't hold")), 1):
            await store.add_experiment(exp_id=f"exp_00000000000{i}", run_id="cs_000000000001", round_no=1, n=i,
                                       hypothesis=h, strategy_tag=tag)
            await store.update_experiment(f"exp_00000000000{i}", status=st, verdict_reason=v)
    run(go())
    return obj


def test_playbook_for_an_outcome_run(store):
    _outcome_run(store)
    r = run(store.get_run("cs_000000000001"))
    pb = learning.playbook_for(r, run(store.list_experiments("cs_000000000001")))
    assert pb["task_type"] == "web-research" and pb["source_session"] == "cs_000000000001"
    assert "- official resellers" in pb["do_pattern"] and "- comparison sites" in pb["do_pattern"]
    assert "Raspberry Pi 5, 8GB RAM — 189 EUR at https://shop.example/pi5" in pb["do_pattern"]
    assert "distributors by part number — " in pb["dont_pattern"] and "held-out 250" not in pb["dont_pattern"]
    assert "lowest wins" in pb["trigger_description"] and "captain-swarm,find-price" == pb["tags"]


def test_routes_show_and_forget_lessons_and_save_a_playbook(store, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from captain_claw.flight_deck import captain_swarm_routes as routes
    from captain_claw.flight_deck.auth import get_current_user

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": "u", "role": "user"}
    _outcome_run(store, status="running")
    run(learning.record_run(store, run(store.get_run("cs_000000000001"))))
    saved = []

    async def fake_save(owner, slug, playbook):
        saved.append((slug, playbook["name"]))
        return {"id": "pb_1"}
    monkeypatch.setattr(learning, "save_playbook", fake_save)
    with TestClient(app) as c:
        fams = c.get("/fd/captain-swarm/lessons").json()["families"]
        assert fams[0]["family"] == "find-price" and fams[0]["lessons"] == 3 and fams[0]["runs"] == 1
        got = c.get("/fd/captain-swarm/lessons?family=find-price").json()
        assert {w["tag"] for w in got["weights"]} == {"resellers", "aggregators", "distributors"}
        assert c.post("/fd/captain-swarm/runs/cs_000000000001/playbook", json={"agent": "helper"}).status_code == 409
        run(store.set_status("cs_000000000001", "stalled", "done trying"))
        preview = c.get("/fd/captain-swarm/runs/cs_000000000001/playbook").json()["playbook"]
        assert "official resellers" in preview["do_pattern"]
        r = c.post("/fd/captain-swarm/runs/cs_000000000001/playbook", json={"agent": "helper"})
        assert r.status_code == 200 and r.json()["saved"]["agent"] == "helper" and saved[0][0] == "helper"
        view = c.get("/fd/captain-swarm/runs/cs_000000000001").json()["learning"]
        assert view["family"] == "find-price" and view["playbook"]["agent"] == "helper"
        lesson_id = got["lessons"][0]["id"]
        assert c.delete(f"/fd/captain-swarm/lessons/{lesson_id}").status_code == 200
        assert c.delete(f"/fd/captain-swarm/lessons/{lesson_id}").status_code == 404
        assert len(c.get("/fd/captain-swarm/lessons?family=find-price").json()["lessons"]) == 2


def test_a_rules_lesson_carries_the_rules_but_never_a_disqualified_ones():
    kept = {"status": "kept", "hypothesis": "recruiters", "verdict_reason": "0.9 beats the best 0.8",
            "change": {"add": ["Recruiter outreach doesn't need my reply."]}}
    cheat = {"status": "disqualified", "hypothesis": "zed", "verdict_reason": "it names held-out mail it was never shown: 'zed'",
             "change": {"add": ["Mail from zed@secret.example needs my reply."]}}
    assert "added: Recruiter outreach doesn't need my reply." in learning.attempt_text(kept)
    assert "zed@secret" not in learning.attempt_text(cheat) and "added" not in learning.attempt_text(cheat)
