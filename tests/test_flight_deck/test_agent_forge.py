"""Forge agents: the plan stands until approved, approval makes what was left in it,
and everything afterwards is the person's to choose. Vatra's planner, spawning, Crew
and the model are stood in for; the routes, state and scheduler store are real."""

from __future__ import annotations

import asyncio
import json
import types

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from captain_claw.flight_deck import agent_forge as af
from captain_claw.flight_deck import agent_handshake, crew_routes, crew_runner, crew_store, fd_scheduler
from captain_claw.flight_deck import basna_routes as br
from captain_claw.flight_deck import server as srv
from captain_claw.flight_deck import vatra_routes as vr
from captain_claw.flight_deck.auth import get_current_user

pytestmark = pytest.mark.asyncio

ALICE = {"id": "alice", "email": "a@x.test", "role": "user"}
BOB = {"id": "bob", "email": "b@x.test", "role": "user"}

ARCHETYPES = [
    {"id": "data-analyst", "role": "Data Analyst", "family": "Data", "description": "Numbers", "fleet_instructions": "BASE-ANALYST"},
    {"id": "editor-writer", "role": "Editor", "family": "Writing", "description": "Words", "fleet_instructions": "BASE-WRITER"},
]

ROUTE = {
    "mode": "vatra", "domain": "aif-risk", "rationale": "Two kinds of work.", "shared_context": "Call a fund an AIF.",
    "subtasks": [
        {"id": "s1", "title": "AIF analyst", "owner_archetype_id": "data-analyst", "brief": "Reads reports", "depends_on": []},
        {"id": "s2", "title": "Drafter", "owner_archetype_id": "editor-writer", "brief": "Writes drafts", "depends_on": ["s1"]},
    ],
    "selected": [],
}
PLAN = {
    "overview": "The analyst feeds the drafter.",
    "agents": [
        {"subtask_id": "s1", "agent_id": "data-analyst", "mandate": "Own the fund facts", "produces": "A register",
         "consumes_from": [], "hand_off_notes": "In the shared store"},
        {"subtask_id": "s2", "agent_id": "editor-writer", "mandate": "Draft decisions", "produces": "Word drafts",
         "consumes_from": ["s1"], "hand_off_notes": ""},
    ],
    "questions": [{"id": "q1", "question": "Outlook or Gmail?", "multi": False, "options": ["Outlook", "Gmail"], "selected": [], "other": ""}],
}
RECS = {
    "usage": "Talk to the analyst first.",
    "schedules": [
        {"name": "Morning summary", "agent": "aif-analyst", "schedule": "weekdays 07:30", "prompt": "Summarise new reports.",
         "why": "Saves reading.", "usage": "It arrives in chat.", "delivery": ["chat", "email"]},
    ],
    "apps": [
        {"name": "AIF register", "slug": "AIF Register!", "summary": "A register.", "why": "It holds data.", "brief": "Build a register.",
         "agents": ["aif-analyst", "made-up"], "agent_use": {"aif-analyst": "Files positions."},
         "endpoints": [{"method": "get", "path": "/api/agent/funds", "purpose": "List funds"}, {"method": "GET", "path": "no-slash"}],
         "events": [{"name": "report.received", "purpose": "New report"}], "connect": "both"},
    ],
}


class _Db:
    """The session table and the bell, kept in a dict."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.bell: list[tuple] = []
        self.n = 0

    async def create_basna_session(self, user_id, intent, title="", config="{}"):
        self.n += 1
        row = {"id": f"plan-{self.n}", "user_id": user_id, "intent": intent, "title": title, "config": config,
               "route": "{}", "files": "[]", "status": "pending", "updated_at": "now", "created_at": "now"}
        self.rows[row["id"]] = row
        return dict(row)

    async def get_basna_session(self, sid, uid):
        row = self.rows.get(sid)
        return dict(row) if row and row["user_id"] == uid else None

    async def update_basna_session(self, sid, uid, **fields):
        self.rows[sid].update(fields)

    async def list_basna_sessions(self, uid):
        return [dict(r) for r in self.rows.values() if r["user_id"] == uid]

    async def delete_basna_session(self, sid, uid):
        return self.rows.pop(sid, None) is not None

    async def add_notification(self, user, kind, title, body, ref_type="", ref_id=""):
        self.bell.append((user, kind, title, ref_type, ref_id))

    async def list_shares_for_grantee(self, uid, kind):
        return []


class _Forge:
    """What the tests set and look at."""

    def __init__(self):
        self.db = _Db()
        self.user = ALICE
        self.lead_error: Exception | None = None
        self.planner_error: Exception | None = None
        self.brief = "The files say quarterly reports."
        self.lead_calls: list[dict] = []
        self.planner_calls: list[dict] = []
        self.spawned: list = []
        self.spawn_fail: set[str] = set()
        self.recs: object = RECS
        self.told: list[tuple] = []
        self.saved_inputs: list[str] = []


class _Resp:
    def __init__(self, content, finish="stop"):
        self.content, self.finish_reason, self.model, self.usage = content, finish, "m", {}


@pytest.fixture
def forge(monkeypatch, tmp_path):
    f = _Forge()
    monkeypatch.setattr(af, "get_db", lambda: f.db)
    monkeypatch.setattr(br, "get_db", lambda: f.db)
    monkeypatch.setattr(vr, "get_db", lambda: f.db)
    monkeypatch.setattr(af, "merged_archetypes", _async(ARCHETYPES))
    monkeypatch.setattr(af, "_room", lambda user: 8)
    monkeypatch.setattr(af, "_refuse_when_spent", _async(None))
    monkeypatch.setattr(vr, "_load_owner_tiers", _async(({}, [])))
    monkeypatch.setattr(vr, "_load_registry", lambda: {})
    monkeypatch.setattr(vr, "_resolve_creds", lambda *a, **k: {"provider": "x", "model": "m"})
    monkeypatch.setattr(vr, "_session_files_dir", lambda sid: _mkdir(tmp_path / "files" / sid))

    async def examine(request, user, *, name, intent, input_files, **kw):
        return f.brief

    async def build(db, uid, intent, max_agents, creds, **kw):
        f.lead_calls.append({"intent": intent, "max_agents": max_agents, **kw})
        if f.lead_error:
            raise f.lead_error
        return json.loads(json.dumps(ROUTE))

    async def planner(request, user, sid, **kw):
        f.planner_calls.append(kw)
        if f.planner_error:
            raise f.planner_error
        return json.loads(json.dumps(PLAN))

    monkeypatch.setattr(vr, "_examine_files_brief", examine)
    monkeypatch.setattr(vr, "_build_plan", build)
    monkeypatch.setattr(vr, "_run_group0_planner", planner)

    def inputs(uid, project, src, files):
        f.saved_inputs.append(project)
        return len(files)

    monkeypatch.setattr(vr, "_save_inputs_to_vfs", inputs)

    class Provider:
        async def complete(self, messages, **kw):
            if isinstance(f.recs, Exception):
                raise f.recs
            return _Resp(f.recs if isinstance(f.recs, str) else json.dumps(f.recs))

    monkeypatch.setattr(vr, "_provider_call", lambda *a, **k: (Provider(), 8192))

    from captain_claw.flight_deck import vfs_routes
    monkeypatch.setattr(vfs_routes, "_user_root", lambda uid: _mkdir(tmp_path / "vfs" / uid))

    monkeypatch.setattr(srv, "_load_process_registry", lambda: {})

    async def spawn(config, request, user):
        if config.name in f.spawn_fail:
            raise HTTPException(403, "Agent limit reached.")
        f.spawned.append(config)
        return types.SimpleNamespace(ok=True, slug=srv._slug(config.name), message="")

    monkeypatch.setattr(srv, "spawn_process", spawn)

    async def tell(uid, slug, instructions=""):
        f.told.append((uid, slug))
        return True

    monkeypatch.setattr(af, "hand_identity", tell)
    monkeypatch.setattr(af, "_crew_line", _async("They can use Crew."))
    return f


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _async(value):
    async def call(*a, **k):
        return value
    return call


def _client(forge) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(af.router)
    app.include_router(br.router)
    app.dependency_overrides[get_current_user] = lambda: forge.user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck.test")


async def _settle(sid: str) -> None:
    for _ in range(200):
        task = vr._live_run_task(sid)
        if task is None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("a background step never ended")


async def _at_gate(client, forge, *, files=None) -> str:
    resp = await client.post("/fd/agent-forge/plans", data={"request": "Quarterly look-through for Risk."}, files=files or [])
    assert resp.status_code == 201, resp.text
    sid = resp.json()["id"]
    await _settle(sid)
    return sid


def _drafts(plan: dict) -> list[dict]:
    return plan["agents"]


async def _ready(client, forge) -> str:
    sid = await _at_gate(client, forge)
    plan = (await client.get(f"/fd/agent-forge/plans/{sid}")).json()
    resp = await client.post(f"/fd/agent-forge/plans/{sid}/approve", json={"agents": _drafts(plan), "overview": plan["overview"]})
    assert resp.status_code == 200, resp.text
    await _settle(sid)
    return sid


# ── Planning ──────────────────────────────────────────────────────────


async def test_planning_stops_at_the_plan_with_nothing_created(forge):
    async with _client(forge) as c:
        sid = await _at_gate(c, forge, files=[("files", ("report.pdf", b"%PDF", "application/pdf"))])
        plan = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
    assert plan["status"] == "awaiting_plan"
    assert [a["name"] for a in plan["agents"]] == ["AIF analyst", "Drafter"]
    assert plan["agents"][0]["mandate"] == "Own the fund facts"
    assert plan["questions"][0]["question"] == "Outlook or Gmail?"
    assert plan["files"][0]["name"] == "report.pdf"
    assert forge.spawned == []
    assert forge.db.bell[-1][3:] == ("agent_forge", sid)
    call = forge.lead_calls[0]
    assert call["lead_prompt"] == "forge/lead.md"
    assert "The files say quarterly reports." in call["intent"] and "report.pdf" in call["intent"]
    assert "STANDING agents" in forge.planner_calls[0]["intent"]


async def test_a_request_is_required(forge):
    async with _client(forge) as c:
        resp = await c.post("/fd/agent-forge/plans", data={"request": "  "})
    assert resp.status_code == 400


async def test_no_room_for_agents_is_said_before_planning(forge, monkeypatch):
    monkeypatch.setattr(af, "_room", lambda user: 0)
    async with _client(forge) as c:
        resp = await c.post("/fd/agent-forge/plans", data={"request": "A team"})
    assert resp.status_code == 403 and "limit" in resp.json()["detail"]
    assert not forge.db.rows


async def test_a_dead_examiner_or_planner_does_not_stop_the_plan(forge, monkeypatch):
    async def boom(*a, **k):
        raise HTTPException(403, "Agent limit reached.")

    monkeypatch.setattr(vr, "_examine_files_brief", boom)
    forge.planner_error = HTTPException(403, "Agent limit reached.")
    async with _client(forge) as c:
        sid = await _at_gate(c, forge, files=[("files", ("a.txt", b"x", "text/plain"))])
        plan = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
    assert plan["status"] == "awaiting_plan"
    assert plan["agents"][0]["mandate"] == "Reads reports"  # the Lead's own words


async def test_a_lead_that_fails_says_why_and_can_be_retried(forge):
    forge.lead_error = HTTPException(502, "The Lead's plan was cut off.")
    async with _client(forge) as c:
        sid = await _at_gate(c, forge)
        plan = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
        assert plan["status"] == "error" and "cut off" in plan["error"]
        assert forge.db.bell[-1][1] == "agent_forge_error"
        forge.lead_error = None
        assert (await c.post(f"/fd/agent-forge/plans/{sid}/replan", json={})).status_code == 200
        await _settle(sid)
        assert (await c.get(f"/fd/agent-forge/plans/{sid}")).json()["status"] == "awaiting_plan"


async def test_replan_folds_in_answers_and_the_note(forge):
    async with _client(forge) as c:
        sid = await _at_gate(c, forge)
        questions = [{**PLAN["questions"][0], "selected": ["Outlook"]}]
        resp = await c.post(f"/fd/agent-forge/plans/{sid}/replan", json={"note": "Add a risk reviewer.", "questions": questions})
        assert resp.status_code == 200
        await _settle(sid)
    second = forge.lead_calls[1]["intent"]
    assert "Add a risk reviewer." in second and "Outlook" in second
    assert "AIF analyst" in second  # what it proposed last time
    assert "Outlook" in forge.planner_calls[1]["clarifications"]


async def test_planning_again_starts_from_the_persons_edits(forge):
    async with _client(forge) as c:
        sid = await _at_gate(c, forge)
        edited = [{"name": "Fund analyst", "mandate": "Only the listed funds"}]
        await c.post(f"/fd/agent-forge/plans/{sid}/replan", json={"note": "Fewer agents.", "agents": edited})
        await _settle(sid)
    second = forge.lead_calls[1]["intent"]
    assert "Fund analyst: Only the listed funds" in second and "AIF analyst" not in second


async def test_the_feed_says_which_step_it_is_and_who_was_chosen(forge):
    async with _client(forge) as c:
        sid = await _at_gate(c, forge, files=[("files", ("a.txt", b"x", "text/plain"))])
    events = br._PROGRESS[sid]["events"]
    assert [e["step"] for e in events if e.get("step")] == ["read", "lead", "plan"]
    chosen = next(e["agents"] for e in events if e.get("agents"))
    assert chosen == [{"name": "AIF analyst", "archetype": "data-analyst"}, {"name": "Drafter", "archetype": "editor-writer"}]


async def test_a_plan_is_its_owners_only(forge):
    async with _client(forge) as c:
        sid = await _at_gate(c, forge)
        forge.user = BOB
        assert (await c.get(f"/fd/agent-forge/plans/{sid}")).status_code == 404
        assert (await c.post(f"/fd/agent-forge/plans/{sid}/approve", json={"agents": []})).status_code == 404
        assert (await c.get("/fd/agent-forge/plans")).json()["plans"] == []


async def test_plans_stay_off_the_run_lists(forge):
    async with _client(forge) as c:
        await _at_gate(c, forge)
        assert (await c.get("/fd/basna/sessions")).json() == []
        assert len((await c.get("/fd/agent-forge/plans")).json()["plans"]) == 1


# ── Approving ─────────────────────────────────────────────────────────


async def test_approval_makes_what_was_left_in_the_plan(forge):
    async with _client(forge) as c:
        sid = await _at_gate(c, forge, files=[("files", ("a.pdf", b"x", "application/pdf"))])
        plan = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
        first = {**plan["agents"][0], "name": "Fund analyst", "mandate": "Own every fund fact", "consumes_from": ["s2"]}
        resp = await c.post(f"/fd/agent-forge/plans/{sid}/approve", json={"agents": [first], "overview": "Just one."})
        assert resp.status_code == 200
        await _settle(sid)
        plan = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
    assert [a.name for a in forge.spawned] == ["Fund analyst"]
    assert plan["approved"] == [{"id": "s1", "name": "Fund analyst", "archetype": "data-analyst"}]
    config = forge.spawned[0]
    assert config.archetype == "data-analyst"
    assert config.fleet_instructions.startswith("BASE-ANALYST")
    assert "Own every fund fact" in config.fleet_instructions
    assert "Drafter" not in config.fleet_instructions  # left out, and not named to anyone
    assert "Just one." in config.fleet_instructions
    assert "vfs:aif-risk/" in config.fleet_instructions
    keys = {e["key"]: e["value"] for e in config.env_vars}
    assert keys == {"CLAW_DATASTORE_VFS": "aif-risk"}  # CLAW_VFS_PROJECT would mark it an ephemeral worker
    assert forge.saved_inputs == ["aif-risk/inputs"]
    assert plan["status"] == "ready"
    assert forge.told == [("alice", "fund-analyst")]


async def test_teammates_are_told_about_each_other(forge):
    async with _client(forge) as c:
        await _ready(c, forge)
    analyst, drafter = forge.spawned
    assert "Drafter" in analyst.fleet_instructions and "consult_peer" in analyst.fleet_instructions
    assert "AIF analyst" in drafter.fleet_instructions
    assert "**You build on the work of:** AIF analyst" in drafter.fleet_instructions
    assert "Call a fund an AIF." in drafter.fleet_instructions


async def test_an_edit_is_checked(forge):
    async with _client(forge) as c:
        sid = await _at_gate(c, forge)
        plan = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
        bad = [{**plan["agents"][0], "archetype": "invented"}]
        assert (await c.post(f"/fd/agent-forge/plans/{sid}/approve", json={"agents": bad})).status_code == 400
        assert (await c.post(f"/fd/agent-forge/plans/{sid}/approve", json={"agents": []})).status_code == 400
        stranger = [{**plan["agents"][0], "id": "s99"}]
        assert (await c.post(f"/fd/agent-forge/plans/{sid}/approve", json={"agents": stranger})).status_code == 400
    assert forge.spawned == []


async def test_one_agent_failing_does_not_stop_the_rest(forge):
    forge.spawn_fail = {"AIF analyst"}
    async with _client(forge) as c:
        sid = await _ready(c, forge)
        plan = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
    assert plan["status"] == "ready"
    by_name = {a["name"]: a for a in plan["created"]}
    assert by_name["AIF analyst"]["ok"] is False and "limit" in by_name["AIF analyst"]["error"]
    assert by_name["Drafter"]["ok"] is True
    assert [a.name for a in forge.spawned] == ["Drafter"]


async def test_every_agent_failing_stops_and_going_on_makes_only_the_rest(forge):
    forge.spawn_fail = {"AIF analyst", "Drafter"}
    async with _client(forge) as c:
        sid = await _ready(c, forge)
        plan = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
        assert plan["status"] == "error"
        forge.spawn_fail = {"Drafter"}
        assert (await c.post(f"/fd/agent-forge/plans/{sid}/approve", json={})).status_code == 200
        await _settle(sid)
        forge.spawn_fail = set()
        plan = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
    assert plan["status"] == "ready"
    assert [a.name for a in forge.spawned] == ["AIF analyst"]


async def test_instructions_only_come_back_where_the_browser_keeps_them(forge, monkeypatch):
    from captain_claw.flight_deck import auth

    monkeypatch.setattr(auth, "_fd_auth_enabled", lambda: True)
    async with _client(forge) as c:
        sid = await _ready(c, forge)
        with_accounts = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
        monkeypatch.setattr(auth, "_fd_auth_enabled", lambda: False)
        without = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
    assert "instructions" not in with_accounts["created"][0]
    assert without["created"][0]["instructions"].startswith("BASE-ANALYST")


# ── Recommendations ───────────────────────────────────────────────────


AGENTS = [{"slug": "aif-analyst", "name": "AIF analyst"}, {"slug": "drafter", "name": "Drafter"}]


async def test_recommendations_are_held_to_what_exists():
    recs = af.coerce_recommendations(RECS, AGENTS)
    schedule = recs["schedules"][0]
    assert schedule["agent"] == "aif-analyst" and schedule["schedule_problem"] == ""
    assert schedule["delivery"] == {"chat": True, "email": True, "telegram": False, "email_to": ""}
    app = recs["apps"][0]
    assert app["slug"] == "aif-register"
    assert app["agents"] == ["aif-analyst"]  # a made-up agent is dropped
    assert [e["path"] for e in app["endpoints"]] == ["/api/agent/funds"]
    assert app["endpoints"][0]["method"] == "GET"


async def test_a_schedule_the_scheduler_cannot_read_is_kept_and_marked():
    raw = {"schedules": [{"name": "Monthly", "agent": "ghost", "schedule": "first monday at nine", "prompt": "Do it"}]}
    schedule = af.coerce_recommendations(raw, AGENTS)["schedules"][0]
    assert "unrecognized schedule" in schedule["schedule_problem"]
    assert schedule["agent"] == "aif-analyst"  # someone real, for them to change
    assert schedule["delivery"]["chat"] is True


async def test_junk_recommendations_are_nothing_and_the_counts_are_capped():
    assert af.coerce_recommendations("nope", AGENTS) == {"usage": "", "schedules": [], "apps": []}
    many = {"schedules": [{"name": f"n{i}", "prompt": "p", "schedule": "every 1d"} for i in range(40)],
            "apps": [{"name": "Same", "brief": "b"} for _ in range(40)]}
    recs = af.coerce_recommendations(many, AGENTS)
    assert len(recs["schedules"]) == af.MAX_SCHEDULES and len(recs["apps"]) == af.MAX_APPS
    assert len({a["slug"] for a in recs["apps"]}) == af.MAX_APPS


async def test_recommendations_that_fail_leave_the_team_standing(forge):
    forge.recs = "this isn't JSON"
    async with _client(forge) as c:
        sid = await _ready(c, forge)
        plan = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
        assert plan["status"] == "ready" and "unreadable" in plan["recommend_error"]
        assert len(plan["created"]) == 2
        forge.recs = RECS
        assert (await c.post(f"/fd/agent-forge/plans/{sid}/recommend")).status_code == 200
        await _settle(sid)
        plan = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
    assert plan["recommend_error"] == "" and plan["recommendations"]["schedules"][0]["name"] == "Morning summary"


# ── Scheduled tasks ───────────────────────────────────────────────────


@pytest.fixture
def jobs(monkeypatch, tmp_path):
    store = fd_scheduler.SchedulerStore(tmp_path / "scheduler.db")
    monkeypatch.setattr(fd_scheduler, "get_store", lambda: store)
    return store


async def test_kept_tasks_are_made_once_as_they_were_left(forge, jobs):
    async with _client(forge) as c:
        sid = await _ready(c, forge)
        rec = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()["recommendations"]["schedules"][0]
        item = {**rec, "agent": "drafter", "schedule": "daily 08:00", "name": "Edited"}
        body = {"items": [item], "tz": "Europe/Zagreb"}
        first = (await c.post(f"/fd/agent-forge/plans/{sid}/schedules", json=body)).json()["results"][0]
        again = (await c.post(f"/fd/agent-forge/plans/{sid}/schedules", json=body)).json()["results"][0]
        plan = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
    assert first["ok"] and again["job_id"] == first["job_id"]
    made = jobs.list()
    assert len(made) == 1
    job = made[0]
    assert (job["name"], job["agent_slug"], job["schedule"], job["tz"]) == ("Edited", "drafter", "daily 08:00", "Europe/Zagreb")
    assert job["delivery_kind"] == "agent" and job["prompt_author"] == "human"
    assert plan["recommendations"]["schedules"][0]["job_id"] == first["job_id"]


async def test_a_bad_task_says_why_and_an_outside_agent_is_refused(forge, jobs):
    async with _client(forge) as c:
        sid = await _ready(c, forge)
        rec = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()["recommendations"]["schedules"][0]
        bad = (await c.post(f"/fd/agent-forge/plans/{sid}/schedules",
                            json={"items": [{**rec, "schedule": "whenever"}]})).json()["results"][0]
        stranger = (await c.post(f"/fd/agent-forge/plans/{sid}/schedules",
                                 json={"items": [{**rec, "agent": "someone-elses"}]})).json()["results"][0]
    assert bad["ok"] is False and "unrecognized schedule" in bad["error"]
    assert stranger["ok"] is False
    assert jobs.list() == []


# ── Apps ──────────────────────────────────────────────────────────────


async def test_the_contract_follows_how_agents_connect():
    both, api, wss = af._contract("both"), af._contract("api"), af._contract("wss")
    assert "/api/agent/manifest" in both and "/agent/ws" in both and "AGENT_API_KEY" in both
    assert "Agent socket" not in api and "Agent API" in api
    assert "Agent API (HTTP)" not in wss and '"type": "hello"' in wss
    for text in (both, api, wss):
        assert "agent-connection.json" in text and "PORT" in text


async def test_a_brief_says_who_uses_it_and_never_leaves_the_contract_out():
    app = af.coerce_recommendations(RECS, AGENTS)["apps"][0]
    brief = af.app_brief(app, "aif-risk", {"aif-analyst": "AIF analyst"}, True)
    assert "Build a register." in brief and "AIF analyst: Files positions." in brief
    assert "GET /api/agent/funds" in brief and "report.received" in brief
    assert "`aif-risk` folder" in brief and '"type": "hello"' in brief
    only_api = af.app_brief({**app, "connect": "api"}, "aif-risk", {}, False)
    assert "report.received" not in only_api and "folder" not in only_api.split("## How agents connect")[0]


@pytest.fixture
def crew(monkeypatch):
    seen: dict = {}
    grant = types.SimpleNamespace(allowed=True)

    async def approved(user):
        return {"id": user["id"]}, grant

    async def dispatch(row, g, body):
        seen.update(body)
        return {"id": "crw_1", "status": "queued", "title": body["title"]}

    tasks: dict = {}

    class Store:
        def get(self, uid, task_id):
            return tasks.get(task_id)

    monkeypatch.setattr(crew_routes, "_approved", approved)
    monkeypatch.setattr(crew_runner, "dispatch", dispatch)
    monkeypatch.setattr(crew_store, "get_store", lambda: Store())
    monkeypatch.setattr(crew_runner, "view", lambda t, **k: {"project_name": "aif-register"})
    seen["_tasks"] = tasks
    return seen


async def test_building_goes_through_crew_with_the_contract(forge, crew):
    async with _client(forge) as c:
        sid = await _ready(c, forge)
        resp = await c.post(f"/fd/agent-forge/plans/{sid}/apps/a1/build", json={"harness": "claude", "connect": "api", "brief": "Build it my way."})
        assert resp.status_code == 200, resp.text
        crew["_tasks"]["crw_1"] = {"id": "crw_1", "status": "running", "note": "On it", "branch": "crew/x"}
        plan = (await c.get(f"/fd/agent-forge/plans/{sid}")).json()
    assert crew["harness"] == "claude" and crew["project"] == "aif-register" and crew["kind"] == "ship"
    assert crew["agent_slug"] == "aif-analyst"  # the one called back
    assert "Build it my way." in crew["brief"] and "/api/agent/manifest" in crew["brief"]
    assert '"type": "hello"' not in crew["brief"]
    app = plan["recommendations"]["apps"][0]
    assert app["connect"] == "api" and app["task"]["status"] == "running"


async def test_a_second_build_while_one_runs_is_that_one(forge, crew):
    async with _client(forge) as c:
        sid = await _ready(c, forge)
        await c.post(f"/fd/agent-forge/plans/{sid}/apps/a1/build", json={"harness": "claude"})
        crew["_tasks"]["crw_1"] = {"id": "crw_1", "status": "running"}
        crew.pop("harness")
        again = await c.post(f"/fd/agent-forge/plans/{sid}/apps/a1/build", json={"harness": "codex"})
    assert again.json()["task"]["id"] == "crw_1"
    assert "harness" not in crew


async def test_crew_not_being_approved_is_said(forge, monkeypatch):
    async def refused(user):
        raise HTTPException(403, "Crew isn't approved for this account.")

    monkeypatch.setattr(crew_routes, "_approved", refused)
    async with _client(forge) as c:
        sid = await _ready(c, forge)
        resp = await c.post(f"/fd/agent-forge/plans/{sid}/apps/a1/build", json={"harness": "claude"})
    assert resp.status_code == 403 and "approved" in resp.json()["detail"]


def _links(monkeypatch):
    from captain_claw.flight_deck import agent_link

    made: list = []
    changed: list = []

    async def create(body, user):
        made.append(body)
        return {"id": f"lnk_{len(made)}"}

    async def change(link_id, body, user):
        changed.append((link_id, body))
        return {"id": link_id}

    monkeypatch.setattr(agent_link, "create_link", create)
    monkeypatch.setattr(agent_link, "change_link", change)
    return made, changed


@pytest.fixture
def stored(monkeypatch):
    texts: dict[str, str] = {}

    async def read(owner, kind):
        return dict(texts)

    async def write(owner, kind, identifier, text, *, strict=False):
        texts[identifier] = text

    monkeypatch.setattr(srv, "_owner_agent_instructions", read)
    monkeypatch.setattr(srv, "_set_owner_agent_instructions", write)
    return texts


async def test_connecting_links_the_socket_registers_the_api_and_keeps_no_key(forge, monkeypatch, stored):
    made, changed = _links(monkeypatch)
    calls: list = []

    async def manifest(base, key):
        assert key == "secret-key"
        return {"ws": "/agent/ws", "endpoints": [{"method": "GET", "path": "/api/agent/funds"}]}

    async def register(slug, app, base, key, endpoints):
        calls.append((slug, base, key, len(endpoints)))
        return len(endpoints)

    monkeypatch.setattr(af, "_read_manifest", manifest)
    monkeypatch.setattr(af, "_register_calls", register)
    body = {"url": "https://apps.test/vfs-apps/reg/", "key": "secret-key"}
    async with _client(forge) as c:
        sid = await _ready(c, forge)
        resp = await c.post(f"/fd/agent-forge/plans/{sid}/apps/a1/connect", json=body)
        assert resp.status_code == 200, resp.text
        first = resp.json()
        again = await c.post(f"/fd/agent-forge/plans/{sid}/apps/a1/connect", json=body)
        assert again.status_code == 200
    assert first["results"][0] == {"slug": "aif-analyst", "name": "AIF analyst", "linked": True, "api_calls": 1, "error": ""}
    link = made[0]
    assert link.direction == "outbound" and link.remote_key == "secret-key" and link.agent_slug == "aif-analyst"
    assert link.remote_url == "wss://apps.test/vfs-apps/reg/agent/ws"
    assert changed and changed[0][0] == "lnk_1" and len(made) == 1  # connecting again changes that link
    assert calls[0] == ("aif-analyst", "https://apps.test/vfs-apps/reg", "secret-key", 1)
    assert "secret-key" not in forge.db.rows[sid]["route"]
    text = stored["aif-analyst"]
    assert text.count("<!-- forge-app:aif-register -->") == 1 and "direct_api" in text and "linked_system" in text
    assert "Files positions." in text
    assert "secret-key" not in text
    assert first["connection"]["api_calls"] == 1


async def test_a_socket_address_needs_no_manifest(forge, monkeypatch, stored):
    made, _ = _links(monkeypatch)

    async def no_manifest(base, key):
        raise AssertionError("a socket address has no manifest to read")

    monkeypatch.setattr(af, "_read_manifest", no_manifest)
    async with _client(forge) as c:
        sid = await _ready(c, forge)
        resp = await c.post(f"/fd/agent-forge/plans/{sid}/apps/a1/connect", json={"url": "wss://x.test/agent/ws", "key": "k"})
    assert resp.status_code == 200 and made[0].remote_url == "wss://x.test/agent/ws"
    assert "direct_api" not in stored["aif-analyst"]


async def test_connecting_needs_an_address_a_key_and_an_agent_of_the_team(forge, monkeypatch, stored):
    _links(monkeypatch)
    async with _client(forge) as c:
        sid = await _ready(c, forge)
        path = f"/fd/agent-forge/plans/{sid}/apps/a1/connect"
        assert (await c.post(path, json={"url": "", "key": "k"})).status_code == 400
        assert (await c.post(path, json={"url": "wss://x.test/ws", "key": ""})).status_code == 400
        assert (await c.post(path, json={"url": "ftp://x.test", "key": "k"})).status_code == 400
        assert (await c.post(path, json={"url": "wss://x.test/ws", "key": "k", "agents": ["not-ours"]})).status_code == 400
        assert (await c.post(f"/fd/agent-forge/plans/{sid}/apps/a9/connect", json={"url": "wss://x", "key": "k"})).status_code == 404


async def test_a_manifest_is_only_read_from_an_allowed_address(forge, monkeypatch):
    from captain_claw.flight_deck import agent_link_http

    monkeypatch.setattr(agent_link_http, "address_problem", _async("Use a public address."))
    with pytest.raises(HTTPException) as err:
        await af._read_manifest("http://169.254.169.254", "k")
    assert "public address" in err.value.detail


# ── Telling an agent who it is ────────────────────────────────────────


async def test_the_frame_carries_its_job_and_its_owners_running_agents(monkeypatch):
    registry = {
        "me": {"owner": "alice", "name": "Me", "web_port": 1},
        "mate": {"owner": "alice", "name": "Mate", "web_port": 2, "description": "Helps"},
        "off": {"owner": "alice", "name": "Off", "web_port": 3},
        "other": {"owner": "bob", "name": "Other", "web_port": 4},
    }
    monkeypatch.setattr(srv, "_load_process_registry", lambda: registry)
    monkeypatch.setattr(srv, "_process_is_alive", lambda slug: slug != "off")
    monkeypatch.setattr(srv, "_owner_agent_instructions", _async({"me": "Do the job."}))
    frame = await agent_handshake.identity_frame("alice", "me")
    assert frame["type"] == "peer_agents" and frame["self"]["fleet_instructions"] == "Do the job."
    assert [p["name"] for p in frame["agents"]] == ["Mate"]


async def test_nothing_stored_sends_nothing_so_a_chats_instructions_survive(monkeypatch):
    monkeypatch.setattr(srv, "_owner_agent_instructions", _async({}))
    assert await agent_handshake.identity_frame("", "me") is None


async def test_a_scheduled_turn_tells_the_agent_first(monkeypatch):
    sent: list = []
    subscribers: list = []

    class Ws:
        async def send(self, text):
            frame = json.loads(text)
            sent.append(frame)
            if frame["type"] == "chat":
                for cb in subscribers:
                    await cb({"type": "agent", "text": "Done."})

    ch = types.SimpleNamespace(callback_subscribers=subscribers, send_lock=asyncio.Lock(), agent_ws=Ws())
    monkeypatch.setattr(fd_scheduler, "_get_or_create_channel", _async(ch))
    monkeypatch.setattr(fd_scheduler, "_ensure_agent_binding", _async(None))
    monkeypatch.setattr(fd_scheduler, "await_agent_link", _async(True))
    monkeypatch.setattr(fd_scheduler, "_remove_channel", _async(None))
    reply = await fd_scheduler.run_prompt_and_capture(
        host="localhost", port=1, auth="", prompt="Go.", identity={"type": "peer_agents", "agents": []})
    assert reply == "Done."
    assert [f["type"] for f in sent] == ["peer_agents", "chat"]


async def test_a_link_follows_its_agent_to_a_new_port_and_tells_it_who_it_is(monkeypatch):
    from captain_claw.flight_deck import agent_link

    agent = {"slug": "me", "port": 2, "name": "Me"}
    monkeypatch.setattr(agent_link, "_link_agent", lambda user_id, slug: agent)
    conn = agent_link._LinkConn(types.SimpleNamespace(), ALICE, {"id": "l1", "agent_slug": "me"})
    conn.ch = types.SimpleNamespace(agent_ws=object())
    conn.agent = {"slug": "me", "port": 1}
    bound: list = []

    async def bind(selector):
        bound.append(selector)
        conn.agent = agent

    monkeypatch.setattr(conn, "bind", bind)
    await conn.before_send()
    await conn.before_send()  # the same port needs nothing
    assert bound == [""]

    frames: list = []

    async def frame(owner, slug, **kw):
        frames.append((owner, slug))
        return {"type": "peer_agents"}

    monkeypatch.setattr(agent_handshake, "identity_frame", frame)
    assert await conn.first_frame(conn.ch) == {"type": "peer_agents"}
    assert await conn.first_frame(conn.ch) is None  # once per connection to the agent
    conn.ch.agent_ws = object()  # it restarted
    assert await conn.first_frame(conn.ch) == {"type": "peer_agents"}
    assert frames == [("alice", "me"), ("alice", "me")]
