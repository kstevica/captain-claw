"""Course, Phase 2: goals wired into the rest of the deck.

Pinned here:

* evidence links belong to someone (``owner_id``): a deck-wide goal shows each
  user only their own, and an old ``course_links`` table gains the column;
* progress reports are 0..1 on active or paused goals; ``last_moved`` is the
  newest link;
* the Arbiter reads ``goal_id`` (``G:`` stripped) and keeps only a goal it was
  shown; ``goal_id`` never changes whether an action auto-fires; a done action
  that served a goal links it (a failed one doesn't); the proposal's bell says
  which goal it serves;
* plans: a ``goal_id`` column (migrated), "Plan a route" plans from the goal's
  title and success text, and a done step links the goal;
* the Taskboard: the extraction prompt lists the owner's goals in the trusted
  part, ``serves_goal`` is parsed and kept only when it names one of them, the
  ``course`` sort puts goal tasks first, and closing a goal task as done (by a
  source or by the person) links it;
* Forge agents: the team's goal and one goal per agent, within the admin's
  horizons, serving it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import course as c

BOB, CAROL = "user-bob", "user-carol"


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = c.CourseStore(tmp_path / "course.db")
    monkeypatch.setattr(c, "_STORE", s)
    return s


def _goal(store, scope="user", owner=BOB, horizon="long", title="Ship EU pricing", **kw):
    kw.setdefault("success", "three EU customers")
    if scope == "instance":
        owner = ""
    return store.create(scope=scope, owner_id=owner, horizon=horizon, title=title, **kw)


# ── store: owners, progress, last moved ──────────────────────────────


def test_links_belong_to_someone(store):
    deck = _goal(store, "instance", title="Deck goal")
    store.add_link(deck["id"], "project", "p1", note="Bob's", owner_id=BOB)
    store.add_link(deck["id"], "project", "p2", note="Carol's", owner_id=CAROL)
    assert [lk["note"] for lk in store.links(deck["id"], owner_id=BOB)] == ["Bob's"]
    assert len(store.links(deck["id"])) == 2
    assert [lk["goal_id"] for lk in store.links_for("project", "p2")] == [deck["id"]]
    assert store.remove_link(deck["id"], "project", "p2", owner_id=BOB) is False
    assert store.remove_link(deck["id"], "project", "p2", owner_id=CAROL) is True


def test_old_links_table_gains_owner(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE course_links (goal_id TEXT NOT NULL, kind TEXT NOT NULL, ref_id TEXT NOT NULL,"
        " note TEXT NOT NULL DEFAULT '', added_by TEXT NOT NULL DEFAULT '', at TEXT NOT NULL,"
        " PRIMARY KEY (goal_id, kind, ref_id));")
    conn.close()
    s = c.CourseStore(path)
    have = {r["name"] for r in s._c().execute("PRAGMA table_info(course_links)")}
    assert "owner_id" in have


def test_progress(store):
    g = _goal(store)
    assert store.set_progress(g["id"], 0.4, "halfway-ish")["progress"] == 0.4
    for bad in (1.5, -0.1, "0.5", True):
        with pytest.raises(c.CourseError):
            store.set_progress(g["id"], bad)
    store.set_status(g["id"], "achieved")
    with pytest.raises(c.CourseError, match="takes no progress"):
        store.set_progress(g["id"], 1.0)
    assert store.events(g["id"])[-2]["event"] == "progress"


def test_last_moved(store):
    g = _goal(store)
    assert store.last_moved([g["id"]]) == {}
    store.add_link(g["id"], "task", "t1", owner_id=BOB)
    moved = store.last_moved([g["id"]])
    assert set(moved) == {g["id"]}
    assert store.last_moved([g["id"]], owner_id=CAROL) == {}


def test_helpers(store):
    deck = _goal(store, "instance", title="Deck goal")
    mine = _goal(store, title="Mine")
    _goal(store, owner=CAROL, title="Carol's")
    stale = _goal(store, horizon="short", title="Old", review_every_days=1)
    later = datetime.now(UTC) + timedelta(days=10)
    live = c.live_goals(BOB, now=later)
    assert {g["id"] for g in live} == {deck["id"], mine["id"]}           # stale one gone
    assert stale["id"] in {g["id"] for g in c.live_goals(BOB, include_stale=True, now=later)}
    text = c.prompt_lines(c.live_goals(BOB))
    assert f"- G:{mine['id']} (long-term): Mine — done when: three EU customers" in text
    assert "Carol's" not in text
    assert c.goal_ref("G:goal_x") == "goal_x" and c.goal_ref("none") == "" and c.goal_ref(None) == ""
    assert c.usable_goal(f"G:{mine['id']}", BOB)["id"] == mine["id"]
    assert c.usable_goal(mine["id"], CAROL) is None
    assert c.link_quietly("goal_gone", "task", "t1") is False
    assert c.link_quietly(mine["id"], "task", "t1", note="x" * 900, owner_id=BOB) is True
    assert len(store.links(mine["id"])[0]["note"]) == c.TEXT_CAPS["note"]


# ── the Arbiter and Autonomous Work ──────────────────────────────────


def test_arbiter_reads_goal_id():
    from captain_claw.flight_deck.arbiter import _parse_actions

    got = _parse_actions(json.dumps([{"kind": "nudge", "title": "Ping Ana", "score": 0.8,
                                      "goal_id": "G:goal_abc"}]))
    assert got[0]["goal_id"] == "goal_abc"
    assert _parse_actions('[{"kind": "nudge", "title": "x"}]')[0]["goal_id"] == ""


def test_arbiter_prompt_wording():
    from captain_claw.flight_deck import arbiter

    assert "candidate goals" not in arbiter._SYSTEM_PROMPT
    assert '"goal_id"' in arbiter._SYSTEM_PROMPT
    assert "never replaces the user's approval" in arbiter._SYSTEM_PROMPT


@pytest.mark.parametrize("level,kind,risk", [
    ("act_low_risk", "nudge", "low"), ("act_low_risk", "run_prompt", "normal"),
    ("act", "run_prompt", "normal"), ("act", "run_prompt", "high"), ("propose", "nudge", "low"),
])
def test_goal_never_changes_auto_dispatch(level, kind, risk):
    from captain_claw.flight_deck.autonomy import merge_config
    from captain_claw.flight_deck.fd_dispatch import should_auto_dispatch

    cfg = {**merge_config({}), "autonomy_level": level, "max_autonomy_level": "act"}
    row = {"id": "act_1", "user_id": BOB, "kind": kind, "risk": risk, "payload": {}, "domain": "ops"}
    assert should_auto_dispatch(cfg, row) == should_auto_dispatch(cfg, {**row, "goal_id": "goal_x"})


def test_autonomy_goal_column_and_done_link(store, tmp_path, monkeypatch):
    from captain_claw.flight_deck import autonomy, line

    monkeypatch.setattr(line, "soon", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_status_changed", lambda uid: None)
    # An old ledger without the column gains it.
    path = tmp_path / "autonomy.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE autonomous_actions (id TEXT PRIMARY KEY, user_id TEXT NOT NULL,"
                 " source TEXT NOT NULL DEFAULT 'heartbeat', kind TEXT NOT NULL DEFAULT 'nudge',"
                 " title TEXT NOT NULL DEFAULT '', rationale TEXT NOT NULL DEFAULT '',"
                 " risk TEXT NOT NULL DEFAULT 'normal', domain TEXT NOT NULL DEFAULT 'general',"
                 " score REAL NOT NULL DEFAULT 0.0, status TEXT NOT NULL DEFAULT 'candidate',"
                 " target TEXT NOT NULL DEFAULT '', ref_id TEXT NOT NULL DEFAULT '',"
                 " payload TEXT NOT NULL DEFAULT '{}', outcome TEXT, outcome_note TEXT NOT NULL DEFAULT '',"
                 " created_at TEXT NOT NULL, dispatched_at TEXT, completed_at TEXT)")
    conn.commit()
    conn.close()
    ledger = autonomy.AutonomyStore(path)
    g = _goal(store)
    ok = ledger.add_action(BOB, kind="nudge", title="Ask the accountant", goal_id=g["id"])
    failed = ledger.add_action(BOB, kind="nudge", title="Failed one", goal_id=g["id"])
    plain = ledger.add_action(BOB, kind="nudge", title="No goal")
    assert ok["goal_id"] == g["id"] and plain["goal_id"] == ""
    ledger.update_status(ok["id"], "done", outcome="success")
    ledger.update_status(failed["id"], "done", outcome="fail")
    ledger.update_status(plain["id"], "done", outcome="success")
    links = store.links(g["id"])
    assert [(lk["kind"], lk["ref_id"], lk["owner_id"]) for lk in links] == [("autonomy_action", ok["id"], BOB)]


def test_proposal_bell_says_which_goal(store):
    from captain_claw.flight_deck import owner_notify

    g = _goal(store, title="Ship EU pricing")
    _t, _title, body = owner_notify._proposal_bell({"title": "Ask Ana", "rationale": "Why", "goal_id": g["id"]})
    assert body == "Ask Ana\nWhy\nServes: Ship EU pricing"
    _t, _title, body = owner_notify._proposal_bell({"title": "Ask Ana", "goal_id": "goal_gone"})
    assert body == "Ask Ana"


# ── plans ────────────────────────────────────────────────────────────


@pytest.fixture
def plans(tmp_path, monkeypatch):
    from captain_claw.flight_deck import plans as plans_mod

    s = plans_mod.PlansStore(tmp_path / "plans.db")
    monkeypatch.setattr(plans_mod, "_STORE", s)
    return s


def test_plans_goal_column(plans, store):
    p = plans.create_plan(BOB, "Ship it", [{"title": "Step one", "kind": "nudge"}], goal_id="goal_x")
    assert p["goal_id"] == "goal_x"
    assert plans.create_plan(BOB, "Other", [])["goal_id"] == ""


async def test_a_done_step_links_its_goal(plans, store, monkeypatch):
    from captain_claw.flight_deck import plans as plans_mod

    g = _goal(store)
    p = plans.create_plan(BOB, "Ship it", [{"title": "Draft the page", "kind": "nudge"},
                                          {"title": "Publish", "kind": "nudge"}], goal_id=g["id"])

    async def ran(user_id, step, **kw):
        return {"ok": True, "content": "draft.md saved"}

    async def no_line(*a, **k):
        return None

    monkeypatch.setattr(plans_mod, "_run_step", ran)
    monkeypatch.setattr(plans_mod, "_step_to_line", no_line)
    res = await plans_mod.advance_one(BOB, p["id"])
    assert res["ok"] is True
    links = store.links(g["id"])
    assert [(lk["kind"], lk["ref_id"], lk["owner_id"]) for lk in links] == [("plan", p["id"], BOB)]
    assert links[0]["note"] == "Draft the page: draft.md saved"


async def test_plan_a_route(plans, store, monkeypatch):
    from captain_claw.flight_deck import autonomy_routes
    from captain_claw.flight_deck import plans as plans_mod

    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    g = _goal(store, owner="local", title="Ship EU pricing")
    asked = []

    async def decompose(user_id, goal):
        asked.append(goal)
        return [{"title": "Draft the page", "kind": "nudge"}]

    monkeypatch.setattr(plans_mod, "decompose_goal", decompose)
    app = FastAPI()
    app.include_router(autonomy_routes.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fd.test") as cl:
        r = await cl.post("/fd/autonomy/plans", json={"goal_id": g["id"]})
        missing = await cl.post("/fd/autonomy/plans", json={"goal_id": "goal_nope"})
    assert r.status_code == 200, r.text
    assert asked == ["Ship EU pricing — done when: three EU customers"]
    assert r.json()["plan"]["goal_id"] == g["id"]
    assert missing.status_code == 404
    assert [lk["kind"] for lk in store.links(g["id"])] == ["plan"]


# ── the Taskboard ────────────────────────────────────────────────────


def _view_and_facts():
    view = {"task": {"direction": "received", "agent_name": "Mira", "created_at": 0, "title": "t", "ask": "a"},
            "turns": [], "events": []}
    facts = {"channel": "web", "outcome": "", "failed_calls": 0, "turn": None}
    return view, facts


def test_extraction_prompt_lists_goals_in_the_trusted_part():
    from captain_claw.flight_deck import taskboard_triage as tt

    view, facts = _view_and_facts()
    prompt = tt.extraction_prompt(view, facts, "N1", course="- G:goal_a (long-term): Ship EU pricing")
    head = prompt.split("<<<DATA N1>>>")[0]
    assert "Course (the owner's goals" in head and "G:goal_a" in head
    assert "Course (" not in tt.extraction_prompt(view, facts, "N1")
    assert '"serves_goal": ""' in tt._EXTRACT_SYSTEM.format(nonce="N1")


def test_serves_goal_is_parsed_and_written():
    from captain_claw.flight_deck import taskboard_triage as tt

    got = tt.parse_extraction('{"title": "Q3 invoice", "serves_goal": "G:goal_a"}')
    assert got["serves_goal"] == "goal_a"
    assert tt.extraction_fields(got)["goal_id"] == "goal_a"
    none = tt.parse_extraction('{"title": "Q3 invoice", "serves_goal": "none"}')
    assert none["serves_goal"] == "" and "goal_id" not in tt.extraction_fields(none)


async def test_extract_keeps_only_a_goal_it_was_shown(store, monkeypatch):
    from captain_claw.flight_deck import taskboard_triage as tt

    g = _goal(store, title="Ship EU pricing")
    seen = {}

    async def reader(task, user_id):
        return {"model": "m"}

    async def complete(reader_, system, prompt, user_id):
        seen["prompt"] = prompt
        return seen["answer"], "m"

    monkeypatch.setattr(tt, "reader_for", reader)
    monkeypatch.setattr(tt, "complete", complete)
    view, facts = _view_and_facts()
    seen["answer"] = json.dumps({"title": "Euro page", "serves_goal": f"G:{g['id']}"})
    got, _model, _h = await tt.extract(view, facts, BOB)
    assert got["serves_goal"] == g["id"] and f"G:{g['id']}" in seen["prompt"]
    seen["answer"] = json.dumps({"title": "Euro page", "serves_goal": "G:goal_made_up"})
    got, _model, _h = await tt.extract(view, facts, BOB)
    assert got["serves_goal"] == ""
    got, _model, _h = await tt.extract(view, facts, CAROL)          # not Carol's goal
    assert f"G:{g['id']}" not in seen["prompt"]


def test_course_sort_and_done_links(tmp_path, store):
    from captain_claw.flight_deck import taskboard_store as ts

    assert "course" in ts._SORTS
    board = ts.TaskboardStore(tmp_path / "taskboard.db")
    g = _goal(store, title="Ship EU pricing")
    cols = {r["name"] for r in board._conn.execute("PRAGMA table_info(tasks)")}
    assert "goal_id" in cols
    now = 1_700_000_000.0
    for tid, goal in (("t-goal", g["id"]), ("t-plain", "")):
        board._conn.execute(
            "INSERT INTO tasks (id, user_id, title, status, goal_id, urgency, created_at, updated_at)"
            " VALUES (?, ?, ?, 'open', ?, 0, ?, ?)", (tid, BOB, tid, goal, now, now))
    board._conn.commit()
    # A source closes the goal task: evidence. The other one: nothing.
    assert board._machine_write("t-goal", {"status": "done"}, now + 1) is True
    assert board._machine_write("t-plain", {"status": "done"}, now + 1) is True
    assert [(lk["kind"], lk["ref_id"], lk["owner_id"]) for lk in store.links(g["id"])] == [("task", "t-goal", BOB)]
    # The person marks another goal task done: evidence too.
    board._conn.execute(
        "INSERT INTO tasks (id, user_id, title, status, goal_id, urgency, created_at, updated_at)"
        " VALUES ('t-mine', ?, 't-mine', 'open', ?, 0, ?, ?)", (BOB, g["id"], now, now))
    board._conn.commit()
    assert board.correct(BOB, "t-mine", "done") is not None
    assert {lk["ref_id"] for lk in store.links(g["id"])} == {"t-goal", "t-mine"}


# ── Forge agents ─────────────────────────────────────────────────────


async def test_forge_makes_the_team_goal_and_agent_goals(store, monkeypatch):
    from captain_claw.flight_deck import agent_forge, course_carrier, server

    registry = {"ana": {"slug": "ana", "web_auth": "t1", "owner": BOB},
                "ben": {"slug": "ben", "web_auth": "t2", "owner": BOB}}
    monkeypatch.setattr(server, "_load_process_registry", lambda: registry)

    async def no_refresh(db, owner_id=None, **kw):
        return 0

    monkeypatch.setattr(course_carrier, "refresh_agents", no_refresh)
    monkeypatch.setattr(agent_forge, "get_db", lambda: None)
    made = {
        "a1": {"ok": True, "slug": "ana", "name": "Ana", "mandate": "Find EU pricing benchmarks",
               "produces": "A benchmark table"},
        "a2": {"ok": True, "slug": "ben", "name": "Ben", "mandate": "Draft the euro price page " * 10,
               "produces": ""},
        "a3": {"ok": False, "slug": "", "name": "Cid", "mandate": "never made"},
    }
    out = await agent_forge._make_course_goals(
        BOB, {"title": "Launch EU pricing", "success": "Three EU customers", "horizon": "long",
              "agent_goals": True}, made)
    assert out["errors"] == []
    team = store.get(out["team"])
    assert (team["scope"], team["horizon"], team["owner_id"]) == ("user", "long", BOB)
    assert set(out["agents"]) == {"a1", "a2"}
    ana = store.get(out["agents"]["a1"])
    assert (ana["scope"], ana["horizon"], ana["parent_id"]) == ("agent", "medium", team["id"])
    assert ana["success"] == "A benchmark table"
    ben = store.get(out["agents"]["a2"])
    assert len(ben["title"]) <= c.TEXT_CAPS["title"] and ben["title"].endswith("…")
    assert "Launch EU pricing" in ben["success"]


async def test_forge_respects_the_admins_horizons(store, monkeypatch):
    from captain_claw.flight_deck import agent_forge, course_carrier, server

    monkeypatch.setattr(server, "_load_process_registry", lambda: {"ana": {"slug": "ana", "web_auth": "t1"}})

    async def no_refresh(db, owner_id=None, **kw):
        return 0

    monkeypatch.setattr(course_carrier, "refresh_agents", no_refresh)
    monkeypatch.setattr(agent_forge, "get_db", lambda: None)
    store.set_settings(horizons={"user": ["medium"], "agent": []})
    out = await agent_forge._make_course_goals(
        BOB, {"title": "Launch", "success": "done", "horizon": "long", "agent_goals": True},
        {"a1": {"ok": True, "slug": "ana", "name": "Ana", "mandate": "Benchmarks", "produces": "x"}})
    assert store.get(out["team"])["horizon"] == "medium"          # the one the admin allows
    assert out["agents"] == {}
    assert out["errors"] == ["Your admin has switched off agent goals that fit under the team's goal."]
