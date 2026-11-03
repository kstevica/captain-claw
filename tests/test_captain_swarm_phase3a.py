"""Captain Swarm, Phase 3a: real work, first kind — an agent's triage rules,
improved against what the owner actually did (docs/captain-swarm-plan.md)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from captain_claw.flight_deck import captain_swarm_loop as loop
from captain_claw.flight_deck import captain_swarm_practice as practice
from captain_claw.flight_deck import captain_swarm_rules as rules
from captain_claw.flight_deck.captain_swarm_store import CaptainSwarmStore


def run(coro):
    return asyncio.run(coro)


def case(i, label, frm="Ann <ann@client.example>", subject="Quick question", body="Can you confirm?", cc=()):
    return {"id": f"m{i}", "received_ms": i, "received": "", "from": frm, "to": ["me@example.com"], "cc": list(cc),
            "subject": subject, "body": body, "attachments": [], "categories": ["personal"], "bulk": False,
            "owner_wrote_before": False, "position": 1, "label": label}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A sealed folder, a store, a practice set and a run of the practice kind."""
    sealed = tmp_path / "captain_swarm" / "sealed"
    monkeypatch.setattr(practice, "sealed_root", lambda: sealed)
    monkeypatch.setattr(loop, "sealed_dir", lambda rid: sealed / "runs" / rid)
    monkeypatch.setattr(loop, "_POLL_S", 0)
    store = CaptainSwarmStore(tmp_path / "cs.db")
    loop.set_store(store)
    opened = [case(i, i % 2 == 0, subject=f"Open subject number {i}") for i in range(8)]
    held = [case(100, True, frm="Zed <zed@secret.example>", subject="Board meeting moved to Friday"),
            case(101, False, frm="News <news@shop.example>", subject="Weekly deals")]
    practice.save_set("u", "mt-1", {"own": ["me@example.com"], "skipped": {}, "window": {"from": "", "to": "", "holdout_from": ""},
                                    "open": opened, "holdout": held}, {"reply_days": 3})
    obj = rules.normalize(rules.triage_template(set_id="mt-1", agent="helper", agent_name="Helper", owner_name="Ana",
                                                cases=8, repeat=2))
    ns = type("NS", (), {})()
    ns.store, ns.obj, ns.sealed = store, obj, sealed

    async def new_run():
        return await store.create_run(run_id="cs_r", owner_id="u", title=obj["title"], objective=obj, status="baseline")
    ns.new_run = new_run
    yield ns
    for setter in (loop.set_measure, loop.set_seal, loop.set_screen, loop.set_branch, loop.set_proposer,
                   loop.set_lane, loop.set_start, loop.set_store):
        setter(None)
    run(store.close())


# ── the objective ────────────────────────────────────────────────────


def test_template_normalizes():
    obj = rules.normalize(rules.triage_template(set_id="mt-1", agent="helper", cases=30, repeat=3))
    assert obj["kind"] == "practice" and obj["subject"]["type"] == "rules" and obj["subject"]["start_rules"] == []
    assert obj["metric"]["cases"] == 30 and obj["metric"]["repeat"] == 3 and obj["metric"]["holdout_repeat"] == 3
    assert obj["lane"] == {"kind": "rules", "per_round": 2, "max_rules": 12}
    from captain_claw.flight_deck.captain_swarm_objective import ObjectiveError
    with pytest.raises(ObjectiveError):
        rules.normalize(rules.triage_template(set_id="../x", agent="helper"))
    with pytest.raises(ObjectiveError):
        rules.normalize(rules.triage_template(set_id="mt-1", agent="a b"))


# ── versions and edits ───────────────────────────────────────────────


def test_apply_edit():
    base = ["Newsletters never need my reply.", "Clients' questions need my reply."]
    assert rules.apply_edit(base, {"add": ["A request to reschedule needs my reply."]}) == [*base, "A request to reschedule needs my reply."]
    assert rules.apply_edit(base, {"remove": [1]}) == [base[1]]
    assert rules.apply_edit(base, {"remove": ["newsletters never need my reply."]}) == [base[1]]
    assert rules.apply_edit(base, {"add": ["Clients' questions need my reply.", "  "]}) == base
    assert rules.apply_edit(base, {"add": ["x", "y", "z"]}, max_rules=3) == [*base, "x"]


def test_lane_makes_a_version_from_the_change(env):
    async def go():
        r = await env.new_run()
        base = await rules.start_point(r)
        exp = {"base_sha": base, "change": {"add": ["Mail asking me to choose something needs my reply."]}}
        got = await rules.RulesLane().start(r, exp, "")
        polled = await rules.RulesLane().poll(r, {**exp, "child_id": got["child_id"]})
        return await rules.rules_of(base), await rules.rules_of(got["child_id"]), polled, got
    base, new, polled, got = run(go())
    assert base == [] and new == ["Mail asking me to choose something needs my reply."]
    assert polled == {"state": "done", "head_sha": got["child_id"], "commits": 1, "dirty": 0}


# ── seal and screen ──────────────────────────────────────────────────


def test_holdout_strings_and_the_screen(env):
    async def go():
        r = await env.new_run()
        base = await rules.start_point(r)
        await rules.seal(r, base)
        strings = json.loads((env.sealed / "runs" / "cs_r" / "strings.json").read_text())

        async def version(rs):
            vid = f"v_{len(rs)}_{abs(hash(tuple(rs))) % 10_000}"
            await env.store.add_version(version_id=vid, run_id="cs_r", content={"rules": rs})
            return vid
        same = await rules.screen(r, {"base_sha": base, "head_sha": base})
        fair = await rules.screen(r, {"base_sha": base, "head_sha": await version(["Mail asking me to confirm a date needs my reply."])})
        peek = await rules.screen(r, {"base_sha": base, "head_sha": await version(["Mail from zed@secret.example needs my reply."])})
        peek2 = await rules.screen(r, {"base_sha": base, "head_sha": await version(["Anything like 'Board meeting moved to Friday' needs a reply."])})
        shared = await rules.screen(r, {"base_sha": base, "head_sha": await version(["Mail from ann@client.example needs my reply."])})
        many = await rules.screen(r, {"base_sha": base, "head_sha": await version([f"rule {i}" for i in range(13)])})
        return strings, same, fair, peek, peek2, shared, many
    strings, same, fair, peek, peek2, shared, many = run(go())
    assert "zed@secret.example" in strings and "board meeting moved to friday" in strings
    assert "ann@client.example" not in strings          # it's in the open part too: fair to learn
    assert not same["ok"] and same["reason"] == "it changed nothing"
    assert fair["ok"] and fair["changed"] == ["+ Mail asking me to confirm a date needs my reply."]
    assert not peek["ok"] and "held-out mail" in peek["reason"]
    assert not peek2["ok"]
    assert shared["ok"]
    assert not many["ok"] and "at most 12" in many["reason"]


# ── measuring ────────────────────────────────────────────────────────


def test_with_rules():
    q = "This email arrived…"
    assert rules.with_rules(q, []) == q
    got = rules.with_rules(q, ["A", "B"], "Ana")
    assert got.startswith("How Ana triages mail") and "- A\n- B" in got and got.endswith(q)


def test_measure_puts_the_rules_first_and_counts_tokens(env, monkeypatch):
    asked: list[str] = []

    async def fake_asker(owner_id, slug, name, label):
        async def ask(question):
            asked.append(question)
            yes = "Open subject number 0" in question or "Open subject number 2" in question \
                or "Open subject number 4" in question or "Open subject number 6" in question
            return {"answer": "YES" if yes else "NO", "tokens": 100, "seconds": 1}
        return ask
    monkeypatch.setattr(rules, "exam_asker", fake_asker)

    async def go():
        r = await env.new_run()
        await env.store.update_run("cs_r", status="running")
        await env.store.add_version(version_id="v_x", run_id="cs_r", content={"rules": ["Rule one."]})
        got = await rules.measure(await env.store.get_run("cs_r"), "v_x", "dev")
        return got, await env.store.get_run("cs_r")
    got, r = run(go())
    assert got["value"] == 1.0 and got["stats"]["n"] == 2 and got["per_case"]["m0"] == 1.0
    assert got["guard"] == {"tokens per case": 100.0} and r["spent_tokens"] == 1600
    assert len(asked) == 16 and all(q.startswith("How Ana triages mail") and "- Rule one." in q for q in asked)


def test_measure_refuses_when_the_agent_does_not_answer(env, monkeypatch):
    async def fake_asker(*a):
        async def ask(question):
            return {"error": "Helper isn't running."}
        return ask
    monkeypatch.setattr(rules, "exam_asker", fake_asker)

    async def go():
        await env.new_run()
        await env.store.update_run("cs_r", status="running")
        await env.store.add_version(version_id="v_x", run_id="cs_r", content={"rules": []})
        return await rules.measure(await env.store.get_run("cs_r"), "v_x", "holdout")
    got = run(go())
    assert got["value"] is None and "didn't answer" in got["error"]


# ── the proposer ─────────────────────────────────────────────────────


def test_parse_proposals():
    text = '[{"hypothesis": "Reschedules", "strategy_tag": "Reschedule!", "why": "misses", "add": ["A reschedule needs my reply."]},' \
           ' {"hypothesis": "nothing"}, {"hypothesis": "drop one", "remove": [2]}]'
    got = rules.parse(text, 5)
    assert [g["change"] for g in got] == [{"add": ["A reschedule needs my reply."], "remove": []}, {"add": [], "remove": [2]}]
    assert got[0]["strategy_tag"] == "reschedule" and got[0]["files"] == []


def test_brief_shows_rules_score_and_open_misses_only(env):
    async def go():
        r = await env.new_run()
        await env.store.add_version(version_id="v_c", run_id="cs_r", content={"rules": ["Newsletters never need my reply."]})
        dev = {"stats": {"n": 2, "mean": 0.7, "stdev": 0.05, "low": 0.65, "high": 0.75},
               "pooled": {"precision": 0.9, "recall": 0.6}, "per_case": {"m2": 0.0, "m3": 1.0}}
        await env.store.update_run("cs_r", champion={"sha": "v_c", "dev": dev, "holdout": {}, "exp": ""})
        ledger = [{"status": "discarded", "strategy_tag": "x", "hypothesis": "tried", "change": {"add": ["R"]},
                   "verdict_reason": "no gain"}]
        return await rules.brief(await env.store.get_run("cs_r"), ledger)
    b = run(go())
    assert "1. Newsletters never need my reply." in b and "F1 0.700" in b and "recall 0.60" in b
    assert "Open subject number 2" in b and "[replied]" in b and "Open subject number 3" not in b
    assert "Board meeting" not in b and "tried" in b and "no gain" in b


# ── the loop, end to end, on a practice run ──────────────────────────


def test_driver_keeps_better_rules(env):
    scores = {}

    async def measure(run_, vid, which):
        rs = await rules.rules_of(vid)
        mean = 0.6 + 0.2 * len(rs) if which == "dev" else 0.7 + 0.1 * len(rs)
        scores[(vid, which)] = mean
        return {"value": mean, "stats": {"n": 2, "mean": mean, "stdev": 0.01, "low": mean, "high": mean},
                "guard": {"tokens per case": 100.0}, "per_case": {"m1": 0.0}, "pooled": {}}
    ideas = [[{"hypothesis": "reschedules", "strategy_tag": "a", "files": [], "why": "",
               "change": {"add": ["A request to reschedule needs my reply."]}},
              {"hypothesis": "cheat", "strategy_tag": "b", "files": [], "why": "",
               "change": {"add": ["Mail from zed@secret.example needs my reply."]}}]]

    async def propose(run_, ledger, k):
        return ideas.pop(0) if ideas else []
    loop.set_measure(measure)
    loop.set_proposer(propose)

    async def go():
        await env.new_run()
        status = await loop.Driver(env.store, "cs_r").drive()
        r = await env.store.get_run("cs_r")
        return status, r, await env.store.list_experiments("cs_r"), await rules.rules_of(r["champion"]["sha"])
    status, r, exps, champ_rules = run(go())
    assert status == "stalled"
    assert [e["status"] for e in exps] == ["kept", "disqualified"]
    assert champ_rules == ["A request to reschedule needs my reply."]
    assert "held-out mail" in exps[1]["verdict_reason"]
    assert exps[1]["base_sha"] == r["champion"]["sha"]      # the second started from the new best
    assert "champion_branch" not in r["state"]["report"] and r["state"]["report"]["improved"]


# ── routes ───────────────────────────────────────────────────────────


@pytest.fixture()
def client(env, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from captain_claw.flight_deck import agent_preset as ap
    from captain_claw.flight_deck import captain_swarm_routes as routes
    from captain_claw.flight_deck.auth import get_current_user

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": "u", "display_name": "Ana", "role": "user"}

    class DB:
        pass

    async def tiers(db, owner):
        return {"reason": {"model": "m", "provider": "ollama"}}, None
    monkeypatch.setattr("captain_claw.flight_deck.auth.get_db", lambda: DB())
    monkeypatch.setattr("captain_claw.flight_deck.basna_routes._load_owner_tiers", tiers)
    monkeypatch.setattr("captain_claw.flight_deck.basna_routes._effective_key", lambda p, k, b=None: k)
    monkeypatch.setattr(ap, "_own_process", lambda slug, owner: {"name": "Helper"} if slug == "helper"
                        else (_ for _ in ()).throw(__import__("fastapi").HTTPException(404, "no")))
    monkeypatch.setattr(ap, "_owner", lambda user: user["id"])
    monkeypatch.setattr("captain_claw.flight_deck.bat_routes._owner_has_shell", lambda u: False)

    async def kick():
        return 0
    monkeypatch.setattr(loop, "kick", kick)
    with TestClient(app) as c:
        yield c, routes


def test_routes_start_a_triage_run_without_a_shell(client):
    c, _ = client
    tpl = c.get("/fd/captain-swarm/templates").json()
    assert tpl["templates"]["triage"]["sets"] == [{"id": "mt-1", "title": "Inbox triage: did you reply?", "open": 8, "holdout": 2}]
    r = c.post("/fd/captain-swarm/runs", json={"template": "triage", "options": {"set_id": "mt-1", "agent": "helper", "cases": 8}})
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "practice" and "Ana" in r.json()["title"]
    bad = c.post("/fd/captain-swarm/runs", json={"template": "triage", "options": {"set_id": "mt-1", "agent": "someone"}})
    assert bad.status_code == 404
    assert c.post("/fd/captain-swarm/runs", json={"template": "mrav"}).status_code == 400   # code needs a shell


def test_routes_teach_the_best_rules(client, monkeypatch):
    c, routes = client
    taught = {}

    async def teach_rules(slug, user, request, new, *, source):
        taught.update(slug=slug, rules=new, source=source)
        return {"added": len(new)}
    monkeypatch.setattr("captain_claw.flight_deck.agent_onboarding.teach_rules", teach_rules)
    rid = c.post("/fd/captain-swarm/runs", json={"template": "triage", "options": {"set_id": "mt-1", "agent": "helper"}}).json()["id"]
    store = loop.get_store()

    async def prepare():
        await store.add_version(version_id="v_b", run_id=rid, content={"rules": ["Old rule."]})
        await store.add_version(version_id="v_c", run_id=rid, content={"rules": ["Old rule.", "New rule."]})
        await store.update_run(rid, baseline={"sha": "v_b"}, champion={"sha": "v_c", "exp": "e1"}, status="running")
    run(prepare())
    assert c.post(f"/fd/captain-swarm/runs/{rid}/teach").status_code == 409            # still running
    got = c.get(f"/fd/captain-swarm/runs/{rid}").json()
    assert got["rules"] == {"baseline": ["Old rule."], "champion": ["Old rule.", "New rule."]}
    run(store.set_status(rid, "stalled", "no more gains"))
    r = c.post(f"/fd/captain-swarm/runs/{rid}/teach")
    assert r.status_code == 200 and r.json() == {"added": 1, "rules": ["New rule."]}
    assert taught["slug"] == "helper" and taught["rules"] == ["New rule."]


def test_answers_are_priced_on_the_agents_model_and_the_budget_stops_a_measurement(env, monkeypatch):
    """The exam answers are the agent's own model at work: each is priced as it
    comes (cache reads at their own rate), and a run out of money stops mid-way."""
    calls = []

    async def fake_asker(*a):
        async def ask(question):
            calls.append(1)
            return {"answer": "NO", "tokens": 21000, "seconds": 1,
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "cache_read_input_tokens": 19980}}
        return ask
    monkeypatch.setattr(rules, "exam_asker", fake_asker)
    monkeypatch.setattr(rules, "agent_model", lambda slug: "claude-haiku-5-5")
    usd, tokens = rules.price_answer("claude-haiku-5-5", {"tokens": 21000, "usage": {
        "prompt_tokens": 1000, "completion_tokens": 20, "cache_read_input_tokens": 19980}})
    assert tokens == 21000 and usd == pytest.approx(1000 * 0.1e-6 + 20 * 0.5e-6 + 19980 * 0.01e-6, rel=1e-3)
    assert rules.price_answer("no-such-model", {"tokens": 5})[0] is None
    assert rules.price_answer("claude-haiku-5-5", {"tokens": 1_000_000})[0] == pytest.approx(0.1)

    async def go():
        await env.new_run()
        await env.store.update_run("cs_r", status="running")
        await env.store.add_version(version_id="v_x", run_id="cs_r", content={"rules": []})
        got = await rules.measure(await env.store.get_run("cs_r"), "v_x", "dev")
        r = await env.store.get_run("cs_r")
        # Nearly out of money: the next measurement stops itself after an answer or so.
        obj = {**r["objective"], "budget": {**r["objective"]["budget"], "usd": r["spent_usd"] + 0.00001}}
        await env.store.update_run("cs_r", objective=obj)
        calls.clear()
        stopped = await rules.measure(await env.store.get_run("cs_r"), "v_x", "dev")
        return got, r, stopped
    monkeypatch.setattr(rules, "_WATCH_S", 0.01)

    async def slow_asker(*a):
        inner = await fake_asker()

        async def ask(question):
            await asyncio.sleep(0.02)
            return await inner(question)
        return ask
    monkeypatch.setattr(rules, "exam_asker", slow_asker)
    got, r, stopped = run(go())
    assert got["usd"] == pytest.approx(16 * 0.0003098, rel=1e-2) and r["spent_usd"] == pytest.approx(got["usd"])
    assert r["spent_tokens"] == 16 * 21000
    assert stopped["value"] is None and "budget ran out" in stopped["error"] and len(calls) < 16
