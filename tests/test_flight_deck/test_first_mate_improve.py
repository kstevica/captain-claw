"""Captain Swarm's ``factory`` kind (P6, piece 1).

Pinned here:

* the cases split: every open case is dev, the sealed ones alternate,
  holdout first; a factory needs three cases, two sealed, to be improved;
* the objective: family ``factory-<id>``, grants and triggers never in it;
* an attempt is an edit of the charter and steps; the screen refuses no
  change, a stub of a charter, a held-out case's words and the eval;
* the seal writes the split and the held-out strings, owner-only;
* measuring runs each case as a Bat run in rehearsal (a Captain Swarm tag)
  and grades it; dev keeps per-case grades, holdout doesn't; spend counts;
* the proposer sees open cases' words, a sealed case only as a number;
* starting one from First Mate: admins only, ready factory, Captain Swarm's
  own checks, a run of kind factory that starts at once.
"""

from __future__ import annotations

import json
import stat

import pytest

from captain_claw.flight_deck import bat_loop, events
from captain_claw.flight_deck import captain_swarm_loop as loop
from captain_claw.flight_deck import captain_swarm_rehearsal as rehearsal
from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import first_mate_cases as cases
from captain_claw.flight_deck import first_mate_improve as improve
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.captain_swarm_objective import ObjectiveError
from captain_claw.flight_deck.captain_swarm_store import CaptainSwarmStore

BOB, CAROL = "user-bob", "user-carol"
CHARTER = ("Go through my inbox for client mail about invoices and draft a short reply to each. Never send "
           "anything. Finish with who wrote and whether a draft waits.")


class FakeBat:
    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}
        self.truth = "Drafted a reply."

    async def create_run(self, *, run_id, owner_id, title, task, config=None, origin=None,
                         source_host="localhost", source_port=0, llm_usd_cap=0.0, status="planning"):
        self.runs[run_id] = {"id": run_id, "owner_id": owner_id, "title": title, "task": task,
                             "config": config or {}, "status": "done", "cumulative_tokens": 1000,
                             "cumulative_usd": 0.01, "truth": self.truth, "stopped_reason": ""}
        return run_id

    async def get_run(self, run_id):
        r = self.runs.get(run_id)
        return dict(r) if r else None

    async def append_event(self, run_id, stage, message="", **data):
        return 1


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "_STORE", events.EventsStore(tmp_path / "events.db"))
    monkeypatch.setattr(c, "_STORE", c.CourseStore(tmp_path / "course.db"))
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True, "admins_only": False})
    return s


@pytest.fixture
async def cs(tmp_path, monkeypatch):
    s = CaptainSwarmStore(tmp_path / "cs.db")
    loop.set_store(s)
    monkeypatch.setattr(loop, "sealed_dir", lambda rid: tmp_path / "sealed" / rid)
    yield s
    loop.set_store(None)
    await s.close()


@pytest.fixture
def bat(monkeypatch):
    fake = FakeBat()

    async def kick():
        return 0

    async def cancel_run(store, run_id):
        fake.runs[run_id]["status"] = "cancelled"
        return True

    monkeypatch.setattr(bat_loop, "_STORE", fake)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "cancel_run", cancel_run)
    monkeypatch.setattr(improve, "POLL_S", 0.0)
    return fake


def _live(store, owner=BOB, **kw):
    kw.setdefault("title", "Inbox replies")
    kw.setdefault("charter", CHARTER)
    kw.setdefault("checks", {"contract": ["every invoice question has a draft", "nothing was sent"]})
    kw.setdefault("runner_config", {"worker_mode": "plain", "steps": ["read the inbox", "draft replies"]})
    f = store.create(owner, **kw)
    return store.approve(f["id"], 1, f["draft"]["sha"], owner)


def _cases(store, f):
    made = [cases.add_case(store, f["id"], title=f"Sealed question number {n}",
                           input=f"From: Client {n} <client{n}@held.example>\nCould you check invoice {1040 + n} "
                                 f"for me, the total looks wrong?", expect=["drafts a reply", "doesn't send it"])
            for n in range(3)]
    made.append(cases.add_case(store, f["id"], title="Open total question", sealed=False,
                               input="From: Ana <ana@open.example>\nIs the total right?", expect=["drafts a reply"]))
    return made


async def _run(cs, store, f, **kw):
    obj = improve.normalize({**improve.factory_template(store.get(f["id"]), store.live(f["id"])), **kw})
    run = await cs.create_run(run_id="cs_0123456789ab", owner_id=BOB, title=obj["title"], objective=obj,
                              status="running")
    return run


# ── the split and the objective ────────────────────────────────────────

def test_the_split_and_readiness(store):
    f = _live(store)
    with pytest.raises(ObjectiveError, match="at least 3 cases"):
        improve.ready(store, BOB, f["id"])
    made = _cases(store, f)
    dev, hold = improve.split(cases.cases(store, f["id"]))
    sealed = sorted(x["id"] for x in made[:3])
    assert hold == [sealed[0], sealed[2]] and dev == [made[3]["id"], sealed[1]]
    got, live = improve.ready(store, BOB, f["id"])
    assert got["id"] == f["id"] and live["version"] == 1
    with pytest.raises(ObjectiveError, match="isn't one of yours"):
        improve.ready(store, CAROL, f["id"])


def test_the_objective(store):
    f = _live(store)
    obj = improve.normalize(improve.factory_template(store.get(f["id"]), store.live(f["id"])))
    assert obj["kind"] == "factory" and obj["family"] == improve.family_of(f["id"])
    assert obj["family"].startswith("factory-fac-")
    assert obj["subject"]["start"] == {"charter": CHARTER, "steps": ["read the inbox", "draft replies"],
                                       "worker_mode": "plain"}
    assert "grants" not in json.dumps(obj) and "triggers" not in json.dumps(obj)
    assert obj["metric"]["direction"] == "max" and obj["noise"]["min_gain"] == 0.1
    assert obj["budget"]["tokens"] == improve.BUDGET["tokens"]
    with pytest.raises(ObjectiveError):
        improve.normalize({"subject": {"type": "factory", "factory_id": "nope"}})
    assert loop.parts_for("factory")["lane"].__class__ is improve.FactoryLane
    assert "keeping every limit" in loop.REVIEW_QUESTIONS["factory"]


# ── the lane, the seal and the screen ──────────────────────────────────

async def test_an_attempt_is_an_edit_and_the_screen_is_fair(store, cs):
    f = _live(store)
    _cases(store, f)
    run = await _run(cs, store, f)
    base = await improve.start_point(run)
    await improve.seal(run, base)
    d = loop.sealed_dir(run["id"])
    assert stat.S_IMODE((d / "split.json").stat().st_mode) == 0o600
    split = json.loads((d / "split.json").read_text())
    assert not set(split["dev"]) & set(split["holdout"])
    lane = improve.FactoryLane()

    async def attempt(change):
        exp = {"base_sha": base, "change": change}
        child = (await lane.start(run, exp, ""))["child_id"]
        return await improve.screen(run, {**exp, "head_sha": child})

    better = CHARTER + " For each one, check the invoice number against my sent mail first."
    assert (await attempt({"charter": better}))["ok"]
    assert (await attempt({"steps": ["read the inbox", "check invoices", "draft replies"]}))["ok"]
    assert (await attempt({}))["reason"] == "it changed nothing"
    assert "too short" in (await attempt({"charter": "Do it."}))["reason"]
    held_out = {x["id"]: x for x in cases.cases(store, f["id"])}[split["holdout"][0]]
    email = held_out["input"].split("<", 1)[1].split(">", 1)[0]  # a client only a held-out case names
    held = await attempt({"charter": better + f" Watch for {email}."})
    assert not held["ok"] and "held-out case it was never shown" in held["reason"]
    opened = await attempt({"charter": better + " Like ana@open.example, keep it short."})
    assert opened["ok"]  # an open case's words are fair to use


# ── measuring ──────────────────────────────────────────────────────────

async def test_measuring_runs_each_case_in_rehearsal(store, cs, bat, monkeypatch):
    f = _live(store)
    made = _cases(store, f)
    grades = {made[3]["title"]: 1.0, made[1]["title"]: 0.5, made[0]["title"]: 0.0, made[2]["title"]: 1.0}

    async def grade(owner, case, result):
        return grades[case["title"]], f"graded {case['title']}"

    monkeypatch.setattr(cases, "grade", grade)
    run = await _run(cs, store, f)
    base = await improve.start_point(run)
    await improve.seal(run, base)
    split = json.loads((loop.sealed_dir(run["id"]) / "split.json").read_text())
    dev = await improve.measure(run, base, "dev")
    assert dev["error"] == "" and set(dev["per_case"]) == set(split["dev"])
    expect = sum(grades[x["title"]] for x in made if x["id"] in split["dev"]) / len(split["dev"])
    assert dev["value"] == round(expect, 4) and dev["stats"]["n"] == 1
    hold = await improve.measure(run, base, "holdout")
    assert "per_case" not in hold and hold["value"] is not None
    assert len(bat.runs) == len(split["dev"]) + len(split["holdout"])
    for b in bat.runs.values():
        assert rehearsal.TAG_RE.match(b["config"]["rehearsal"]) and b["config"]["source"] == "captain_swarm"
        assert b["config"]["steps"] == ["read the inbox", "draft replies"] and "This is a test case" in b["task"]
        assert "contract" in b["config"]  # the factory's own done-when lines, as in its real runs
    assert (await cs.get_run(run["id"]))["spent_tokens"] == 1000 * len(bat.runs)


async def test_a_failed_case_counts_nothing_and_a_stop_stops(store, cs, bat, monkeypatch):
    f = _live(store)
    _cases(store, f)

    async def grade(owner, case, result):
        return 1.0, ""

    monkeypatch.setattr(cases, "grade", grade)
    run = await _run(cs, store, f, metric={"repeat": 2})
    base = await improve.start_point(run)
    await improve.seal(run, base)
    real_create = bat.create_run

    async def failing(**kw):
        rid = await real_create(**kw)
        bat.runs[rid]["status"] = "error"
        return rid

    bat.create_run = failing
    got = await improve.measure(run, base, "dev")
    assert got["value"] == 0.0 and got["stats"]["n"] == 2
    await cs.update_run(run["id"], status="stopped")
    bat.create_run = real_create
    for r in bat.runs.values():
        r["status"] = "running"

    async def running(**kw):
        rid = await real_create(**kw)
        bat.runs[rid]["status"] = "running"
        return rid

    bat.create_run = running
    monkeypatch.setattr(improve, "POLL_S", 0.01)
    got = await improve.measure({**run, "status": "stopped"}, base, "dev")
    assert got["value"] is None and got["error"] == "the run was stopped"


# ── the proposer ───────────────────────────────────────────────────────

async def test_the_proposer_never_sees_a_sealed_cases_words(store, cs):
    f = _live(store)
    made = _cases(store, f)
    run = await _run(cs, store, f)
    base = await improve.start_point(run)
    await improve.seal(run, base)
    split = json.loads((loop.sealed_dir(run["id"]) / "split.json").read_text())
    run["champion"] = {"sha": base, "dev": {"stats": {"mean": 0.5}, "per_case": {cid: 0.5 for cid in split["dev"]},
                                           "notes": {cid: "it missed the total" for cid in split["dev"]}}}
    text = await improve.brief(run, [])
    assert "Open total question: 50%" in text and "ana@open.example" in text and "it missed the total" in text
    assert "- a sealed case: 50%" in text
    for x in made[:3]:
        assert x["title"] not in text and "held.example" not in text
    assert "every invoice question has a draft" in text  # the done-when lines, fixed
    ideas = improve.parse(json.dumps([
        {"hypothesis": "check numbers first", "strategy_tag": "Check First", "charter": CHARTER + " Check numbers.",
         "steps": ["read", "check", "draft"]},
        {"hypothesis": "nothing", "charter": "short"},
        {"hypothesis": "steps only", "steps": ["read", "draft"]}]), 4)
    assert [i["strategy_tag"] for i in ideas] == ["check-first", ""]
    assert ideas[0]["change"] == {"charter": CHARTER + " Check numbers.", "steps": ["read", "check", "draft"]}
    assert ideas[1]["change"] == {"steps": ["read", "draft"]}


# ── starting one from First Mate ───────────────────────────────────────

async def test_starting_one(store, cs, bat, monkeypatch):
    from fastapi import HTTPException

    from captain_claw.flight_deck import captain_swarm_routes as csr
    asked: list[str] = []

    async def can_run(owner, objective=None, user=None):
        asked.append(objective["kind"])

    async def kick():
        return None

    monkeypatch.setattr(csr, "_can_run", can_run)
    monkeypatch.setattr(loop, "kick", kick)
    f = _live(store)
    with pytest.raises(HTTPException) as e:
        await improve.start({"id": BOB, "role": "user"}, f["id"])
    assert e.value.status_code == 403  # Captain Swarm is for admins
    with pytest.raises(ObjectiveError, match="at least 3 cases"):
        await improve.start({"id": BOB, "role": "admin"}, f["id"])
    _cases(store, f)
    run = await improve.start({"id": BOB, "role": "admin"}, f["id"], {"budget_tokens": "20M", "per_round": 3})
    assert run["status"] == "baseline" and run["started_by"] == "the owner, from First Mate" and asked == ["factory"]
    got = await cs.get_run(run["id"])
    assert got["objective"]["kind"] == "factory" and got["objective"]["subject"]["factory_id"] == f["id"]
    assert got["objective"]["budget"]["tokens"] == 20_000_000 and got["objective"]["lane"]["per_round"] == 3
    assert any(e["event"] == "improvement_started" for e in store.events(f["id"]))


def test_family_is_slugged():
    assert improve.family_of("fac_0123abcd") == "factory-fac-0123abcd"


# ── the loop, end to end, on a factory ─────────────────────────────────

async def test_the_driver_keeps_a_better_charter(store, cs, bat, monkeypatch):
    f = _live(store)
    _cases(store, f)
    real_create = bat.create_run

    async def create(**kw):  # a worker that does what its brief says
        rid = await real_create(**kw)
        bat.runs[rid]["truth"] = kw["task"]
        return rid

    bat.create_run = create

    async def grade(owner, case, result):
        return (1.0 if "check the invoice number" in result.lower() else 0.4), ""

    better = CHARTER + " For each one, check the invoice number against my sent mail before drafting."
    cheat = CHARTER + " Mail from client0@held.example, client1@held.example or client2@held.example matters most."
    asked: list[str] = []

    async def ask_model(run, system, user, *, label, temperature=0.7, panel=False):
        asked.append(label)
        if label == "propose":
            ideas = [{"hypothesis": "check numbers", "strategy_tag": "check-first", "charter": better},
                     {"hypothesis": "favour some clients", "strategy_tag": "clients", "charter": cheat}]
            return json.dumps(ideas if asked.count("propose") == 1 else [])
        return json.dumps({"verdict": "pass", "reason": "a clearer, general charter"})

    monkeypatch.setattr(cases, "grade", grade)
    monkeypatch.setattr(loop, "ask_model", ask_model)
    obj = improve.normalize({**improve.factory_template(store.get(f["id"]), store.live(f["id"])),
                             "lane": {"per_round": 2, "parallel": 2}, "stop": {"stall_rounds": 1}})
    await cs.create_run(run_id="cs_0123456789ab", owner_id=BOB, title=obj["title"], objective=obj,
                        status="baseline")
    status = await loop.Driver(cs, "cs_0123456789ab").drive()
    r = await cs.get_run("cs_0123456789ab")
    exps = {e["hypothesis"]: e for e in await cs.list_experiments("cs_0123456789ab")}
    assert status == "stalled", status
    assert exps["check numbers"]["status"] == "kept"
    assert exps["favour some clients"]["status"] == "disqualified"
    assert "held-out case it was never shown" in exps["favour some clients"]["verdict_reason"]
    assert (await improve.content_of(r["champion"]["sha"]))["charter"] == better
    report = r["state"]["report"]
    assert report["improved"] and report["dev"]["champion"] > report["dev"]["baseline"]
    assert r["objective"]["family"] == improve.family_of(f["id"])
