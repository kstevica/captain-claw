"""Captain Swarm, Phase 4: parallel lanes (docs/captain-swarm-plan.md §15) —
lanes per kind, waves started together, one attempt per site and per
account, the measurement queue, the quick look, one keeper per wave with
the runner-up brought back, the model reviewer, stops and cancels that reach
every lane, a wave resumed after a restart, and the routes."""

from __future__ import annotations

import asyncio

import pytest

from captain_claw import rehearsal_turn as rt
from captain_claw.flight_deck import captain_swarm_loop as loop
from captain_claw.flight_deck import captain_swarm_objective as om
from captain_claw.flight_deck import captain_swarm_outcome as outcome
from captain_claw.flight_deck import captain_swarm_rehearsal as rehearsal
from captain_claw.flight_deck import captain_swarm_rules as rules
from captain_claw.flight_deck.captain_swarm_store import CaptainSwarmStore

RID = "cs_0123456789ab"
BASE, H1, H2, H3, H4 = "b" * 40, "1" * 40, "2" * 40, "3" * 40, "4" * 40


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _fresh_sites():
    rehearsal._SITES.clear()
    yield
    rehearsal._SITES.clear()


# ── lanes per kind ───────────────────────────────────────────────────


def test_lanes_per_kind(monkeypatch):
    assert om.lanes({}, default=1, most=8, per_round_most=8) == (1, 2)
    assert om.lanes({"parallel": 3, "per_round": 2}, default=1, most=8, per_round_most=8) == (3, 3)
    monkeypatch.setenv("FD_CAPTAIN_SWARM_MAX_LANES", "2")
    assert om.lanes({"parallel": 4}, default=1, most=8, per_round_most=8) == (2, 2)      # cut, not refused
    with pytest.raises(om.ObjectiveError):
        om.lanes({"parallel": 0}, default=1, most=8, per_round_most=8)
    monkeypatch.delenv("FD_CAPTAIN_SWARM_MAX_LANES")
    o = outcome.normalize(outcome.outcome_template(task="Find the cheapest new paperback copy of The Book"))
    assert o["lane"]["parallel"] == 3 and o["lane"]["per_round"] == 3                    # Bat's 3 runs per owner
    o = outcome.normalize({**outcome.outcome_template(task="Find the cheapest copy of The Book"),
                           "lane": {"kind": "bat", "parallel": 4}})
    assert o["lane"]["parallel"] == 3
    r = rules.normalize(rules.triage_template(set_id="mt-1", agent="helper"))
    assert r["lane"]["parallel"] == 1
    assert loop.parts_for("outcome")["carry"] is False and loop.parts_for("practice")["slice"] is True
    assert loop.parts_for("code")["queue"]({}) == ("deck:command", 1)
    assert loop.wave_of({"current": "e1"}) == ["e1"] and loop.wave_of({"wave": ["a", "b"], "current": ""}) == ["a", "b"]


# ── sites ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,site", [
    ("https://www.rasppishop.de/Raspberry-Pi-5", "rasppishop.de"),
    ("shop.berrybase.de", "berrybase.de"),
    ("https://amazon.co.uk/dp/x", "amazon.co.uk"),
    ("http://127.0.0.1:8080/x", ""),
    ("localhost", ""),
    ("about:blank", ""),
    ("", ""),
])
def test_site_of(raw, site):
    assert rehearsal.site_of(raw) == site


def test_parse_proposals_keeps_sites_and_pick_wave_spreads_them():
    got = loop.parse_proposals('[{"hypothesis": "official resellers", "strategy_tag": "resellers", '
                               '"sites": ["https://www.berrybase.de/x", "rasppishop.de", 7]}]', 3)
    assert got[0]["sites"] == ["berrybase.de", "rasppishop.de"]
    queue = [{"hypothesis": "a", "sites": ["x.de"]}, {"hypothesis": "b", "sites": ["x.de", "y.de"]},
             {"hypothesis": "c", "sites": ["z.de"]}, {"hypothesis": "d"}]
    picked, rest = loop.pick_wave(queue, 3)
    assert [i["hypothesis"] for i in picked] == ["a", "c", "d"] and [i["hypothesis"] for i in rest] == ["b"]
    picked, rest = loop.pick_wave(queue, 3, per_site=2)
    assert [i["hypothesis"] for i in picked] == ["a", "b", "c"]


def test_take_site_one_attempt_at_a_time(monkeypatch):
    a, b = f"{RID}:exp_aaaaaaaaaaaa", f"{RID}:exp_bbbbbbbbbbbb"
    assert rehearsal.take_site(a, "shop.de", now=100) == ""
    assert rehearsal.take_site(a, "shop.de", now=150) == ""                 # its own: again and again
    assert rehearsal.take_site(b, "shop.de", now=160) == a                  # someone else's: no
    assert rehearsal.take_site(b, "other.de", now=160) == ""
    assert rehearsal.sites_held(RID, now=170) == {"exp_aaaaaaaaaaaa": ["shop.de"], "exp_bbbbbbbbbbbb": ["other.de"]}
    assert rehearsal.take_site(b, "shop.de", now=150 + 601) == ""           # left alone ten minutes: free
    rehearsal.release_sites(RID, "exp_bbbbbbbbbbbb")
    assert rehearsal.take_site(a, "shop.de", now=800) == ""
    rehearsal.release_sites(RID)
    assert rehearsal._SITES == {}
    monkeypatch.setenv("FD_CAPTAIN_SWARM_PER_SITE", "2")
    assert rehearsal.take_site(a, "shop.de") == "" and rehearsal.take_site(b, "shop.de") == ""


@pytest.fixture()
def store(tmp_path):
    s = CaptainSwarmStore(tmp_path / "cs.db")
    loop.set_store(s)
    yield s
    loop.set_store(None)
    run(s.close())


def test_take_asks_for_the_owners_running_attempt(store):
    async def go():
        await store.create_run(run_id=RID, owner_id="u", title="t", objective={}, status="running")
        for i, e in enumerate(("exp_aaaaaaaaaaaa", "exp_bbbbbbbbbbbb"), 1):
            await store.add_experiment(exp_id=e, run_id=RID, round_no=1, n=i, hypothesis="h")
            await store.update_experiment(e, status="running")
        a, b = f"{RID}:exp_aaaaaaaaaaaa", f"{RID}:exp_bbbbbbbbbbbb"
        first = await rehearsal.take(store, "u", tag=a, url="https://www.shop.de/item")
        second = await rehearsal.take(store, "u", tag=b, url="https://shop.de/other")
        account = await rehearsal.take(store, "u", tag=a, account="Shop Login")
        account2 = await rehearsal.take(store, "u", tag=b, account="shop login")
        blank = await rehearsal.take(store, "u", tag=b, url="about:blank")
        stranger = await rehearsal.take(store, "someone-else", tag=a, url="https://x.de")
        await store.update_experiment("exp_aaaaaaaaaaaa", status="measuring")
        over = await rehearsal.take(store, "u", tag=a, url="https://x.de")
        return first, second, account, account2, blank, stranger, over
    first, second, account, account2, blank, stranger, over = run(go())
    assert first == {"ok": True, "site": "shop.de"}
    assert not second["ok"] and "attempt #1.1 is using shop.de" in second["why"]
    assert account["ok"] and account["site"] == "account:shop login"
    assert not account2["ok"] and "the account 'shop login'" in account2["why"]
    assert blank == {"ok": True, "site": ""}
    assert stranger == {"ok": False, "why": "no such attempt"}
    assert over == {"ok": False, "why": "that attempt is over"}


def test_guard_asks_before_the_browser_goes_somewhere(monkeypatch):
    asked = []
    answer = {"value": {"ok": True}}

    async def fake_post(path, body, agent):
        asked.append((path, {k: v for k, v in body.items() if k != "tag"}))
        return answer["value"]
    monkeypatch.setattr(rt, "_post", fake_post)
    monkeypatch.setenv(rt.ENV, f"{RID}:exp_aaaaaaaaaaaa")
    assert run(rt.tool_block("browser", {"action": "navigate", "url": "https://shop.de/x"}, None)) is None
    assert asked[-1] == ("site", {"url": "https://shop.de/x"})
    assert run(rt.tool_block("browser", {"action": "login", "app_name": "Shop"}, None)) is None
    assert asked[-1] == ("site", {"account": "Shop"})
    n = len(asked)
    assert run(rt.tool_block("web_fetch", {"url": "https://shop.de/x"}, None)) is None          # a plain read
    assert run(rt.tool_block("browser", {"action": "scroll"}, None)) is None                    # goes nowhere new
    assert len(asked) == n
    answer["value"] = {"ok": False, "why": "attempt #1.2 is using shop.de in its browser right now"}
    msg = run(rt.tool_block("browser", {"action": "open", "url": "https://shop.de/y"}, None))
    assert msg.startswith("Not now: attempt #1.2 is using shop.de") and "web_fetch" in msg
    answer["value"] = None                                                                      # can't ask: no
    msg = run(rt.tool_block("browser", {"action": "navigate", "url": "https://shop.de/z"}, None))
    assert msg.startswith("Not now: Flight Deck couldn't be asked")
    monkeypatch.delenv(rt.ENV)
    assert run(rt.tool_block("browser", {"action": "navigate", "url": "https://shop.de/z"}, None)) is None


# ── the reviewer ─────────────────────────────────────────────────────


def test_parse_verdict():
    assert loop.parse_verdict('{"verdict": "PASS", "reason": "general"}') == {"ok": True, "reason": "general"}
    assert loop.parse_verdict('Sure: {"verdict": "fail", "reason": "names a case"}')["ok"] is False
    assert loop.parse_verdict('{"verdict": "maybe"}') is None and loop.parse_verdict("no") is None


def test_review_asks_again_once_then_gives_up(monkeypatch):
    replies = ["hmm", '{"verdict": "fail", "reason": "a kit, not the board"}']
    asked = []

    async def fake_ask(run_, system, user, **kw):
        asked.append(kw)
        return replies.pop(0)

    async def material(run_, exp):
        return "the result"
    monkeypatch.setattr(loop, "ask_model", fake_ask)
    monkeypatch.setattr(outcome, "review_material", material)
    r = {"objective": {"kind": "outcome"}}
    assert run(loop.review(r, {})) == {"ok": False, "reason": "a kit, not the board"}
    assert asked[0]["panel"] is True and asked[0]["label"] == "review"
    replies[:] = ["x", "y"]
    with pytest.raises(RuntimeError, match="couldn't be read"):
        run(loop.review(r, {}))


def test_outcome_review_material_shows_the_page():
    obj = outcome.normalize(outcome.outcome_template(task="Find the cheapest new Raspberry Pi 5 8GB board",
                                                     must_include=["8GB"], currency="EUR"))
    text = "Raspberry Pi 5 Starter Kit 8GB with case and PSU ... only 189,00 € incl. VAT ... shipping 4,95"
    best = {"title": "Raspberry Pi 5 8GB", "url": "https://s.de/kit", "value": 189.0, "currency": "EUR",
            "details": "", "snippet": outcome.excerpt(text, 189.0, width=40)}
    m = run(outcome.review_material({"objective": obj}, {"dev": {"best": best}}))
    assert "Task: Find the cheapest new Raspberry Pi 5 8GB board" in m and "it must be: 8GB" in m
    assert "189,00 €" in m and "189 EUR at https://s.de/kit" in m
    with pytest.raises(RuntimeError):
        run(outcome.review_material({"objective": obj}, {"dev": {}}))


# ── the measurement queue ────────────────────────────────────────────


@pytest.mark.parametrize("env,most", [("", 1), ("2", 2)])
def test_measurement_queue(monkeypatch, env, most):
    if env:
        monkeypatch.setenv("FD_CAPTAIN_SWARM_MEASURE_PARALLEL", env)
    r = {"id": RID, "objective": {"kind": "code"}}
    now = {"n": 0, "max": 0, "seen": []}

    async def one(i):
        async with loop.measurement_slot(r, f"e{i}"):
            now["n"] += 1
            now["max"] = max(now["max"], now["n"])
            now["seen"].append(loop.measuring(RID))
            await asyncio.sleep(0.01)
            now["n"] -= 1

    async def go():
        loop._SLOTS.pop("deck:command", None)
        await asyncio.gather(*(one(i) for i in range(3)))
    run(go())
    assert now["max"] == most and loop.measuring(RID) == {}
    if most == 1:
        assert any("waiting" in seen.values() for seen in now["seen"])     # the rest waited their turn
    loop._SLOTS.pop("deck:command", None)


# ── the driver: waves ────────────────────────────────────────────────


class WaveLane:
    """Each idea's results by its hypothesis, one per start. A poll keeps saying
    'running' until *together* attempts have started — a lane run alone fails."""

    def __init__(self, results: dict[str, list[dict]], together: int):
        self.results, self.together = results, together
        self.started: list[str] = []
        self.running = 0
        self.max_running = 0
        self.stopped: list[str] = []
        self.discarded: list[str] = []
        self.hold = False
        self.polls: dict[str, int] = {}

    async def start(self, run_, exp, brief):
        self.started.append(exp["hypothesis"])
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        return {"child_id": f"c{len(self.started)}", "branch": ""}

    async def poll(self, run_, exp):
        self.polls[exp["id"]] = self.polls.get(exp["id"], 0) + 1
        if self.hold or len(self.started) < self.together:
            if not self.hold and self.polls[exp["id"]] > 2000:
                return {"state": "failed", "note": "it ran alone"}
            return {"state": "running", "usd": 0.1, "tokens": 10}
        self.running -= 1
        got = self.results[exp["hypothesis"]].pop(0)
        return {"commits": 1, "usd": 0.1, "tokens": 10, **got}

    async def stop(self, run_, exp):
        self.stopped.append(exp["child_id"])

    async def discard(self, run_, exp):
        self.discarded.append(exp["child_id"])


def make_objective(**lane):
    return {"title": "T", "goal": "G", "kind": "code",
            "subject": {"repo_path": "/r", "editable": ["a/**"], "protected": [], "snapshot": {}},
            "metric": {"run": "x {checkout}", "read": "summary.pass_rate", "spread": "summary.per_repeat",
                       "per_case": "", "direction": "max", "timeout_s": 60, "dev_file": "", "env": []},
            "holdout": {"run": "x {checkout}", "file": "/h"}, "guardrails": [], "noise": {"min_gain": 0.05},
            "lane": {"kind": "crew", "harness": "claude", "model": "", "model_source": None, "per_round": 3,
                     "parallel": 3, **lane},
            "budget": {"tokens": 200_000_000, "hours": 8.0, "max_experiments": 40},
            "stop": {"target": None, "stall_rounds": 2, "remeasure_every": 50}}


def idea(h):
    return {"hypothesis": h, "strategy_tag": h, "files": [], "why": ""}


@pytest.fixture()
def wave(store, monkeypatch):
    """Fake seams around a real driver; scores by (sha, which)."""
    monkeypatch.setattr(loop, "_POLL_S", 0)
    ns = type("NS", (), {})()
    ns.store = store
    ns.scores: dict[tuple[str, str], float] = {}
    ns.measured: list[tuple[str, str]] = []
    ns.ideas: list[list[dict]] = []
    ns.reviews: dict[str, dict] = {}
    ns.reviewed: list[str] = []
    ns.at_once = {"n": 0, "max": 0}
    loop._SLOTS.clear()

    async def measure(run_, sha, which):
        ns.measured.append((sha, which))
        ns.at_once["n"] += 1
        ns.at_once["max"] = max(ns.at_once["max"], ns.at_once["n"])
        await asyncio.sleep(0.005)
        ns.at_once["n"] -= 1
        v = ns.scores.get((sha, which))
        if v is None:
            return {"value": None, "error": f"no score for {sha[:4]}/{which}"}
        return {"value": v, "stats": {"n": 3, "mean": v, "stdev": 0.0, "low": v, "high": v}, "guard": {},
                "per_case": {}, "error": ""}

    async def seal(run_, sha):
        return None

    async def screen_(run_, exp):
        return {"ok": True, "reason": "", "changed": ["a/x.py"], "diff_stat": "1 file", "hits": []}

    async def branch(run_, sha):
        return None

    async def propose_(run_, ledger, k):
        return ns.ideas.pop(0) if ns.ideas else []

    async def review_(run_, exp):
        ns.reviewed.append(exp["head_sha"])
        got = ns.reviews.get(exp["head_sha"], {"ok": True, "reason": "general"})
        if isinstance(got, Exception):
            raise got
        return got

    loop.set_measure(measure)
    loop.set_seal(seal)
    loop.set_screen(screen_)
    loop.set_branch(branch)
    loop.set_proposer(propose_)
    loop.set_review(review_)

    async def new_run(objective=None, state=None):
        await store.create_run(run_id=RID, owner_id="u", title="T", objective=objective or make_objective(),
                               status="baseline" if state is None else "running")
        await store.update_run(RID, state=state or {"base_sha": BASE})

    def drive():
        return loop.Driver(store, RID).drive()
    ns.new_run, ns.drive = new_run, drive
    yield ns
    for setter in (loop.set_measure, loop.set_seal, loop.set_screen, loop.set_branch, loop.set_proposer,
                   loop.set_lane, loop.set_review):
        setter(None)
    loop._SLOTS.clear()


def by_n(exps):
    return {(e["round"], e["n"]): e for e in exps}


def test_a_wave_runs_together_and_keeps_one(wave):
    ns = wave
    ns.scores = {(BASE, "dev"): 0.60, (BASE, "holdout"): 0.50, (H1, "dev"): 0.80, (H1, "holdout"): 0.70,
                 (H2, "dev"): 0.70, (H3, "dev"): 0.61, (H4, "dev"): 0.82}
    lane = WaveLane({"A": [{"state": "done", "head_sha": H1}], "B": [{"state": "done", "head_sha": H2},
                                                                     {"state": "done", "head_sha": H4}],
                     "C": [{"state": "done", "head_sha": H3}]}, together=3)
    loop.set_lane(lane)
    ns.ideas = [[idea("A"), idea("B"), idea("C")]]

    async def go():
        await ns.new_run()
        status = await ns.drive()
        return status, await ns.store.get_run(RID), await ns.store.list_experiments(RID), \
            [e["message"] for e in await ns.store.list_events(RID)]
    status, r, exps, events = run(go())
    e = by_n(exps)
    assert lane.max_running == 3                                          # all three at once
    assert ns.at_once["max"] == 1                                         # one measurement at a time (the queue)
    assert e[(1, 1)]["status"] == "kept" and r["champion"]["sha"] == H1
    assert e[(1, 2)]["status"] == "discarded" and "#1.1 did better in the same wave" in e[(1, 2)]["verdict_reason"]
    assert "comes back" in e[(1, 2)]["verdict_reason"]
    assert e[(1, 3)]["status"] == "discarded" and "not a gain" in e[(1, 3)]["verdict_reason"]
    assert {x["base_sha"] for x in (e[(1, 1)], e[(1, 2)], e[(1, 3)])} == {BASE}   # one wave, one starting point
    again = e[(1, 4)]
    assert again["hypothesis"] == "B" and again["base_sha"] == H1 and again["change"]["again_of"] == e[(1, 2)]["id"]
    assert again["status"] == "discarded"                                  # 0.82 vs 0.80: no gain on top
    assert ns.reviewed == [H1] and (H2, "holdout") not in ns.measured     # only the keeper is reviewed and held out
    assert status == "stalled" and r["spent_usd"] > 0
    assert any("3 attempts at once" in m for m in events)
    assert any("the reviewer passed it" in m for m in events)


def test_the_reviewer_can_turn_the_best_down(wave):
    ns = wave
    ns.scores = {(BASE, "dev"): 0.60, (BASE, "holdout"): 0.50, (H1, "dev"): 0.90, (H2, "dev"): 0.75,
                 (H2, "holdout"): 0.60}
    loop.set_lane(WaveLane({"A": [{"state": "done", "head_sha": H1}], "B": [{"state": "done", "head_sha": H2}]},
                           together=2))
    ns.ideas = [[idea("A"), idea("B")]]
    ns.reviews = {H1: {"ok": False, "reason": "it special-cases task 7"}}

    async def go():
        await ns.new_run(make_objective(parallel=2, per_round=2))
        await ns.drive()
        return await ns.store.get_run(RID), by_n(await ns.store.list_experiments(RID))
    r, e = run(go())
    assert e[(1, 1)]["status"] == "disqualified" and "the reviewer: it special-cases task 7" in e[(1, 1)]["verdict_reason"]
    assert e[(1, 2)]["status"] == "kept" and r["champion"]["sha"] == H2
    assert (H1, "holdout") not in ns.measured


def test_no_review_is_no_pass(wave):
    ns = wave
    ns.scores = {(BASE, "dev"): 0.60, (BASE, "holdout"): 0.50, (H1, "dev"): 0.90}
    loop.set_lane(WaveLane({"A": [{"state": "done", "head_sha": H1}]}, together=1))
    ns.ideas = [[idea("A")]]
    ns.reviews = {H1: RuntimeError("no model")}

    async def go():
        await ns.new_run(make_objective(parallel=1, per_round=1))
        await ns.drive()
        return await ns.store.get_run(RID), await ns.store.list_experiments(RID)
    r, exps = run(go())
    assert exps[0]["status"] == "failed" and "the review couldn't be done: no model" in exps[0]["verdict_reason"]
    assert r["champion"]["sha"] == BASE


def test_when_the_best_does_not_hold_the_next_one_may(wave):
    ns = wave
    ns.scores = {(BASE, "dev"): 0.60, (BASE, "holdout"): 0.50, (H1, "dev"): 0.90, (H1, "holdout"): 0.30,
                 (H2, "dev"): 0.75, (H2, "holdout"): 0.60}
    loop.set_lane(WaveLane({"A": [{"state": "done", "head_sha": H1}], "B": [{"state": "done", "head_sha": H2}]},
                           together=2))
    ns.ideas = [[idea("A"), idea("B")]]

    async def go():
        await ns.new_run(make_objective(parallel=2, per_round=2))
        await ns.drive()
        return await ns.store.get_run(RID), by_n(await ns.store.list_experiments(RID))
    r, e = run(go())
    assert e[(1, 1)]["status"] == "discarded" and "didn't hold" in e[(1, 1)]["verdict_reason"]
    assert e[(1, 2)]["status"] == "kept" and r["champion"]["sha"] == H2


def test_a_limit_stops_every_lane_of_the_wave(wave):
    ns = wave
    ns.scores = {(BASE, "dev"): 0.60, (BASE, "holdout"): 0.50}
    lane = WaveLane({}, together=3)
    lane.hold = True                                                       # they keep running and spending
    loop.set_lane(lane)
    ns.ideas = [[idea("A"), idea("B"), idea("C")]]
    obj = make_objective()
    obj["budget"]["tokens"] = 25

    async def go():
        await ns.new_run(obj)
        status = await ns.drive()
        return status, await ns.store.get_run(RID), await ns.store.list_experiments(RID)
    status, r, exps = run(go())
    assert status == "stopped" and "token budget is used" in r["stopped_reason"]
    assert sorted(lane.stopped) == ["c1", "c2", "c3"]
    assert all(e["status"] == "failed" and e["verdict_reason"].startswith("stopped: the token budget") for e in exps)


def test_cancel_stops_every_lane_of_the_wave(wave):
    ns = wave
    ns.scores = {(BASE, "dev"): 0.60, (BASE, "holdout"): 0.50}
    lane = WaveLane({}, together=3)
    lane.hold = True
    loop.set_lane(lane)
    ns.ideas = [[idea("A"), idea("B"), idea("C")]]

    async def go():
        await ns.new_run()
        task = asyncio.create_task(ns.drive())
        for _ in range(500):
            await asyncio.sleep(0.005)
            if len(lane.started) == 3:
                break
        await loop.cancel_run(ns.store, RID)
        status = await asyncio.wait_for(task, 5)
        return status, await ns.store.list_experiments(RID)
    status, exps = run(go())
    assert status == "cancelled" and sorted(set(lane.stopped)) == ["c1", "c2", "c3"]
    assert all(e["status"] == "failed" for e in exps)


def test_a_wave_resumes_after_a_restart(wave):
    ns = wave
    ns.scores = {(H1, "dev"): 0.80, (H1, "holdout"): 0.70}
    loop.set_lane(WaveLane({}, together=1))
    point = {"sha": BASE, "dev": {"value": 0.6, "stats": {"n": 3, "mean": 0.6, "stdev": 0.0, "low": 0.6, "high": 0.6}},
             "holdout": {"value": 0.5, "stats": {"n": 3, "mean": 0.5, "stdev": 0.0, "low": 0.5, "high": 0.5}}, "exp": ""}

    async def go():
        await ns.new_run(state={"base_sha": BASE, "round": 1, "queue": [], "wave": ["exp_aaaaaaaaaaaa", "exp_bbbbbbbbbbbb"]})
        await ns.store.update_run(RID, baseline=point, champion=point)
        for i, (eid, sha) in enumerate((("exp_aaaaaaaaaaaa", H1), ("exp_bbbbbbbbbbbb", H2)), 1):
            await ns.store.add_experiment(exp_id=eid, run_id=RID, round_no=1, n=i, hypothesis=f"h{i}", base_sha=BASE)
            await ns.store.update_experiment(eid, head_sha=sha, child_id=f"c{i}")
        # A was measured when the deck went down; B was in the middle of its measurement.
        await ns.store.update_experiment("exp_aaaaaaaaaaaa", status="measured",
                                         dev={"value": 0.8, "stats": {"n": 3, "mean": 0.8, "stdev": 0.0, "low": 0.8, "high": 0.8}})
        await ns.store.update_experiment("exp_bbbbbbbbbbbb", status="measuring")
        ns.scores[(H2, "dev")] = 0.62
        await ns.drive()
        return await ns.store.get_run(RID), by_n(await ns.store.list_experiments(RID))
    r, e = run(go())
    assert (H2, "dev") in ns.measured and (H1, "dev") not in ns.measured     # only the cut-short one again
    assert e[(1, 1)]["status"] == "kept" and e[(1, 2)]["status"] == "discarded"
    assert r["champion"]["sha"] == H1 and r["state"]["wave"] == []


def test_a_quick_look_sends_only_the_better_half_on(wave, monkeypatch):
    ns = wave
    obj = rules.normalize(rules.triage_template(set_id="mt-1", agent="helper", cases=30))
    obj["lane"].update(parallel=4, per_round=4)
    ns.scores = {(BASE, "dev"): 0.60, (BASE, "holdout"): 0.50, (H1, "slice"): 0.9, (H2, "slice"): 0.8,
                 (H3, "slice"): 0.5, (H4, "slice"): 0.4, (H1, "dev"): 0.85, (H1, "holdout"): 0.7, (H2, "dev"): 0.8}
    h5 = "5" * 40
    ns.scores[(h5, "dev")] = 0.86
    lane = WaveLane({h: [{"state": "done", "head_sha": s}] for h, s in
                     (("A", H1), ("B", H2), ("C", H3), ("D", H4))}, together=4)
    lane.results["B"].append({"state": "done", "head_sha": h5})
    loop.set_lane(lane)
    ns.ideas = [[idea("A"), idea("B"), idea("C"), idea("D")]]

    async def go():
        await ns.new_run(obj)
        await ns.drive()
        return by_n(await ns.store.list_experiments(RID))
    e = run(go())
    assert {s for s, w in ns.measured if w == "dev"} & {H1, H2, H3, H4} == {H1, H2}      # the better half
    assert e[(1, 3)]["status"] == e[(1, 4)]["status"] == "discarded"
    assert "slower half on a quick look" in e[(1, 4)]["verdict_reason"] and e[(1, 4)]["dev"]["stage"] == "slice"
    assert e[(1, 1)]["status"] == "kept" and e[(1, 1)]["dev"]["slice"] == 0.9
    assert ns.at_once["max"] == 1                                            # one agent, one question at a time
    # B beat the start too: it comes back once, on top of A.
    assert e[(1, 2)]["status"] == "discarded" and "comes back" in e[(1, 2)]["verdict_reason"]


# ── the one-task kind ────────────────────────────────────────────────


class FakeBat:
    def __init__(self, statuses):
        self.statuses, self.created = statuses, []

    async def list_runs(self, owner, limit=100):
        return [{"status": s} for s in self.statuses]

    async def create_run(self, **kw):
        self.created.append(kw)

    async def append_event(self, *a, **kw):
        return None


def test_outcome_lanes_are_bat_slots_and_share_the_budget(monkeypatch):
    from captain_claw.flight_deck import bat_loop
    obj = outcome.normalize(outcome.outcome_template(task="Find the cheapest new paperback copy of The Book"))
    r = {"id": RID, "owner_id": "u", "objective": obj, "spent_usd": 4.0, "spent_tokens": 44_000_000}
    bat = FakeBat(["running", "done", "waiting"])
    monkeypatch.setattr(bat_loop, "get_store", lambda: bat)

    async def kick():
        return 0
    monkeypatch.setattr(bat_loop, "kick", kick)
    assert run(outcome.capacity(r, 3)) == 1                                   # 3 per owner, 2 busy
    bat.statuses = ["running", "planning", "retrying"]
    assert run(outcome.capacity(r, 3)) == 0
    exp = {"id": "exp_aaaaaaaaaaaa", "round": 1, "n": 1, "hypothesis": "resellers", "why": "",
           "change": {"sites": ["berrybase.de"]}}
    run(outcome.BatLane().start(r, {**exp, "token_share": 3_000_000}, ""))
    run(outcome.BatLane().start(r, {**exp, "token_share": 90_000_000}, ""))
    run(outcome.BatLane().start(r, exp, ""))                                # no share: what the run has left
    # Capped in tokens — its share or the most per attempt, whichever is less — never in dollars.
    assert [c["config"]["token_cap"] for c in bat.created] == [3_000_000, 10_000_000, 6_000_000]
    assert {c["llm_usd_cap"] for c in bat.created} == {0.0}
    assert "Mainly on: berrybase.de" in bat.created[0]["task"]
    assert bat.created[0]["config"]["rehearsal"] == f"{RID}:exp_aaaaaaaaaaaa"


# ── routes ───────────────────────────────────────────────────────────


def test_routes_site_check_and_live_view(store, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from captain_claw.flight_deck import captain_swarm_routes as routes
    from captain_claw.flight_deck.auth import get_current_user

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": "u", "role": "user"}
    monkeypatch.setattr(routes, "_agent_owner", lambda body: "u")
    obj = outcome.normalize(outcome.outcome_template(task="Find the cheapest new paperback copy of The Book"))

    async def prepare():
        await store.create_run(run_id=RID, owner_id="u", title="t", objective=obj, status="running")
        await store.update_run(RID, state={"wave": ["exp_aaaaaaaaaaaa", "exp_bbbbbbbbbbbb"], "queue": [idea("C")]})
        for i, e in enumerate(("exp_aaaaaaaaaaaa", "exp_bbbbbbbbbbbb"), 1):
            await store.add_experiment(exp_id=e, run_id=RID, round_no=1, n=i, hypothesis="h")
            await store.update_experiment(e, status="running")
    run(prepare())
    with TestClient(app) as c:
        a = c.post("/fd/captain-swarm/agent/site", json={"tag": f"{RID}:exp_aaaaaaaaaaaa", "url": "https://www.shop.de/a"})
        b = c.post("/fd/captain-swarm/agent/site", json={"tag": f"{RID}:exp_bbbbbbbbbbbb", "url": "https://shop.de/b"})
        assert a.json() == {"ok": True, "site": "shop.de"}
        assert b.json()["ok"] is False and "attempt #1.1" in b.json()["why"]
        live = c.get(f"/fd/captain-swarm/runs/{RID}").json()["live"]
    assert live["wave"] == ["exp_aaaaaaaaaaaa", "exp_bbbbbbbbbbbb"] and live["queued"] == 1
    assert live["sites"] == {"exp_aaaaaaaaaaaa": ["shop.de"]}
    tpl = run(_templates(routes))
    assert tpl["templates"]["outcome"]["defaults"]["parallel"] == 3 and tpl["templates"]["outcome"]["max_parallel"] == 3


async def _templates(routes):
    from unittest import mock
    with mock.patch.object(routes.practice, "list_sets", lambda owner: []), \
            mock.patch.object(routes, "_local_models", mock.AsyncMock(return_value=[])), \
            mock.patch.object(routes, "_default_holdout", lambda: ""), \
            mock.patch.object(routes, "_source_repo", lambda: ""):
        return await routes.templates(user={"id": "u", "role": "admin"})


# ── found live: scans that overlap, and folders a rehearsal may look at ──


def test_bat_runs_started_together_get_one_driver_each(tmp_path):
    """A wave's attempts kick Bat at the same moment; each run must be driven once."""
    from captain_claw.flight_deck import bat_loop
    from captain_claw.flight_deck.bat_store import BatStore

    class Driver:
        def __init__(self):
            self.drives: list[str] = []

        async def drive(self, run_id):
            self.drives.append(run_id)
            await asyncio.sleep(0.05)
            return "done"

        async def watch_parked(self, run_):
            return None

    async def go():
        bat = BatStore(tmp_path / "bat.db")
        for i in range(3):
            await bat.create_run(run_id=f"bat_{i:012d}", owner_id="u", title="t", task="x", config={}, origin={},
                                 llm_usd_cap=1.0, status="planning")
        driver = Driver()
        sup = bat_loop.BatSupervisor(bat, driver)
        launched = await asyncio.gather(sup.tick(), sup.tick(), sup.tick())
        await asyncio.gather(*sup._running.values())
        await bat.close()
        return launched, driver.drives
    launched, drives = run(go())
    assert sum(launched) == 3 and sorted(drives) == [f"bat_{i:012d}" for i in range(3)]


def test_a_kick_on_the_periodic_scan_drives_a_run_once(store, monkeypatch):
    drives = []

    class OneDriver:
        def __init__(self, store_, run_id):
            self.run_id = run_id

        async def drive(self):
            drives.append(self.run_id)
            await asyncio.sleep(0.05)
            return "done"
    monkeypatch.setattr(loop, "Driver", OneDriver)

    async def go():
        await store.create_run(run_id=RID, owner_id="u", title="t", objective={}, status="baseline")
        sup = loop.Supervisor(store)
        launched = await asyncio.gather(sup.tick(), sup.tick())
        await asyncio.gather(*sup.tasks.values())
        return launched
    assert sum(run(go())) == 1 and drives == [RID]


@pytest.mark.parametrize("action,risk,verdict", [
    ("ls", "read", "allow"), ("tree", "read", "allow"), ("stat", "read", "allow"), ("info", "read", "allow"),
    ("mkdir", "write", "intent"), ("rm", "write", "intent"), ("mv", "write", "intent"),
])
def test_a_rehearsal_may_look_at_its_folders(action, risk, verdict):
    from captain_claw.flight_deck.tool_risk import risk_for
    assert risk_for("vfs", {"action": action, "path": "vfs:bat-x/"}) == risk
    assert rt.classify("vfs", {"action": action, "path": "vfs:bat-x/"})[0] == verdict


def test_a_bat_step_is_priced_on_its_worker_model_when_the_agent_does_not_say():
    from captain_claw.flight_deck.bat_routes import step_cost
    entries = [{"model": "", "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}},
               {"model": "claude-sonnet-5-5", "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                                                        "cache_read_input_tokens": 0}}]
    got = step_cost(entries, "claude-haiku-5-5")
    assert got["priced"] and got["usd"] == pytest.approx(0.10)              # Haiku 5.5: $0.10 per million in
    assert step_cost([{"model": "", "usage": {"prompt_tokens": 10}}], "")["usd"] is None   # truly unknown stays unknown
