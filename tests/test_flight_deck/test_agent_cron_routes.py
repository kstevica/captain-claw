"""Agents' own schedules in Scheduled tasks (agent_cron_routes).

A job an agent set up with its ``cron`` tool lives in the agent's own
sessions.db. Flight Deck lists it next to its own scheduled tasks — for the
agent's owner only (an admin, every agent's) — and pauses, resumes, removes
and runs it: through the agent's REST while it runs, in its file while it is
down. The files here are made by the agent's real SessionManager.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import captain_claw.flight_deck.server as srv
from captain_claw.cron import compute_next_run, to_utc_iso
from captain_claw.flight_deck import agent_cron_routes as acr
from captain_claw.flight_deck import agent_idle, agent_secret, auth
from captain_claw.session import SessionManager

ALICE, BOB, ADMIN = "user-alice", "user-bob", "user-admin"
SECRET = "deck-agent-secret"


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


async def _make_job(db, *, text="Check the outreach inbox", schedule=None, history=None, **update):
    sm = SessionManager(db)
    try:
        schedule = schedule or {"type": "daily", "hour": 10, "minute": 0}
        job = await sm.create_cron_job(
            kind="prompt", payload={"text": text, "author": "agent"}, schedule=schedule,
            session_id="sess-1", next_run_at=to_utc_iso(compute_next_run(schedule)),
        )
        for ev in history or []:
            await sm.append_cron_job_history(job.id, **ev)
        if update:
            await sm.update_cron_job(job.id, **update)
        return job.id
    finally:
        await sm.close()


async def _load(db, job_id):
    sm = SessionManager(db)
    try:
        return await sm.load_cron_job(job_id)
    finally:
        await sm.close()


@pytest.fixture()
def deck(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for var in ("FD_LOCKDOWN", "FD_GLASSES_BRIDGE_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", SECRET)
    agent_secret.reset_cache_for_tests()
    registry = {
        "alice-agent": {"name": "Scout", "owner": ALICE, "web_port": 41001, "web_auth": "tok-a"},
        "bob-agent": {"name": "Bobby", "owner": BOB, "web_port": 41002, "web_auth": "tok-b"},
    }
    alive: set[str] = set()
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)
    monkeypatch.setattr(srv, "_load_process_registry", lambda: registry)
    monkeypatch.setattr(srv, "_process_is_alive", lambda slug: slug in alive)
    rest: list[tuple[str, str, str]] = []

    async def fake_rest(entry, method, path, name):
        rest.append((entry.get("web_auth"), method, path))
        return {"ok": True}

    monkeypatch.setattr(acr, "_agent_rest", fake_rest)
    yield {"registry": registry, "alive": alive, "rest": rest, "dir": tmp_path}
    agent_secret.reset_cache_for_tests()


def _db(deck, slug):
    return deck["dir"] / slug / "data" / "sessions" / "sessions.db"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(acr.router)
    return TestClient(app, client=("10.0.0.5", 50123))


def _bearer(uid: str, role: str = "user") -> dict[str, str]:
    return {"Authorization": f"Bearer {auth.create_access_token(uid, role=role)}"}


# ── The list ──────────────────────────────────────────────────────────


def test_owner_sees_their_agents_jobs_only(deck):
    a = _run(_make_job(_db(deck, "alice-agent"), text="Scout outreach follow-up pass"))
    _run(_make_job(_db(deck, "bob-agent"), text="Bob's digest"))
    c = _client()

    mine = c.get("/scheduler/agent-jobs", headers=_bearer(ALICE)).json()
    assert [j["id"] for j in mine] == [a]
    job = mine[0]
    assert job["agent_slug"] == "alice-agent"
    assert job["task"] == "Scout outreach follow-up pass"
    assert job["schedule"] == "daily 10:00"
    assert job["author"] == "agent"
    assert job["enabled"] is True
    assert job["session_id"] == "sess-1"
    assert job["next_run_at"] > datetime.now(UTC).timestamp()
    assert job["agent_state"] == "off"
    assert job["running"] is False

    everyone = c.get("/scheduler/agent-jobs", headers=_bearer(ADMIN, role="admin")).json()
    assert {j["agent_slug"] for j in everyone} == {"alice-agent", "bob-agent"}


def test_agents_without_a_file_or_table_list_nothing(deck):
    db = _db(deck, "alice-agent")
    db.parent.mkdir(parents=True)
    import sqlite3
    sqlite3.connect(db).close()  # a file with no cron table yet
    assert _client().get("/scheduler/agent-jobs", headers=_bearer(ALICE)).json() == []


def test_an_agent_caller_is_refused(deck):
    _run(_make_job(_db(deck, "alice-agent")))
    r = _client().get("/scheduler/agent-jobs", headers={"X-Agent-Secret": SECRET})
    assert r.status_code == 403


def test_running_only_while_the_agent_is_up_and_the_start_is_fresh(deck):
    now = datetime.now(UTC)
    start = {"monitor_event": {"timestamp": _iso(now - timedelta(minutes=2)), "step": "job_start"}}
    _run(_make_job(_db(deck, "alice-agent"), history=[start]))
    c = _client()
    assert c.get("/scheduler/agent-jobs", headers=_bearer(ALICE)).json()[0]["running"] is False
    deck["alive"].add("alice-agent")
    job = c.get("/scheduler/agent-jobs", headers=_bearer(ALICE)).json()[0]
    assert job["running"] is True and job["agent_state"] == "on"


def test_a_finished_or_stale_run_is_not_running(deck):
    now = datetime.now(UTC)
    deck["alive"].add("alice-agent")
    _run(_make_job(_db(deck, "alice-agent"), history=[
        {"monitor_event": {"timestamp": _iso(now - timedelta(minutes=3)), "step": "job_start"}},
        {"monitor_event": {"timestamp": _iso(now - timedelta(minutes=1)), "step": "job_done"}},
    ]))
    _run(_make_job(_db(deck, "alice-agent"), text="old", history=[
        {"monitor_event": {"timestamp": _iso(now - timedelta(hours=5)), "step": "job_start"}},
    ]))
    jobs = _client().get("/scheduler/agent-jobs", headers=_bearer(ALICE)).json()
    assert [j["running"] for j in jobs] == [False, False]


def test_asleep_agent_state(deck, monkeypatch):
    _run(_make_job(_db(deck, "alice-agent")))
    monkeypatch.setattr(agent_idle, "is_asleep", lambda slug, entry=None: slug == "alice-agent")
    job = _client().get("/scheduler/agent-jobs", headers=_bearer(ALICE)).json()[0]
    assert job["agent_state"] == "asleep"


# ── Pause, resume, remove ─────────────────────────────────────────────


def test_down_agent_edits_its_file(deck):
    db = _db(deck, "alice-agent")
    jid = _run(_make_job(db))
    c, h = _client(), _bearer(ALICE)

    assert c.post(f"/scheduler/agent-jobs/alice-agent/{jid}/pause", headers=h).json() == {"ok": True}
    job = _run(_load(db, jid))
    assert job.enabled is False and job.last_status == "paused"

    # Long paused: resume schedules from now, not the stale time.
    _run(_make_job(db, text="x"))  # another job stays as it is
    assert c.post(f"/scheduler/agent-jobs/alice-agent/{jid}/resume", headers=h).status_code == 200
    job = _run(_load(db, jid))
    assert job.enabled is True and job.last_status == "pending"
    assert datetime.fromisoformat(job.next_run_at) > datetime.now(UTC)

    assert c.delete(f"/scheduler/agent-jobs/alice-agent/{jid}", headers=h).json() == {"ok": True}
    assert _run(_load(db, jid)) is None
    assert len(c.get("/scheduler/agent-jobs", headers=h).json()) == 1
    assert deck["rest"] == []


def test_running_agent_is_asked_through_its_rest(deck):
    db = _db(deck, "alice-agent")
    jid = _run(_make_job(db))
    deck["alive"].add("alice-agent")
    c, h = _client(), _bearer(ALICE)
    c.post(f"/scheduler/agent-jobs/alice-agent/{jid}/pause", headers=h)
    c.delete(f"/scheduler/agent-jobs/alice-agent/{jid}", headers=h)
    assert deck["rest"] == [
        ("tok-a", "POST", f"/api/cron/jobs/{jid}/pause"),
        ("tok-a", "DELETE", f"/api/cron/jobs/{jid}"),
    ]
    assert _run(_load(db, jid)).enabled is True  # the agent changes its own file


def test_another_persons_agent_is_not_found(deck):
    jid = _run(_make_job(_db(deck, "alice-agent")))
    c, h = _client(), _bearer(BOB)
    assert c.post(f"/scheduler/agent-jobs/alice-agent/{jid}/pause", headers=h).status_code == 404
    assert c.delete(f"/scheduler/agent-jobs/alice-agent/{jid}", headers=h).status_code == 404
    assert c.get(f"/scheduler/agent-jobs/alice-agent/{jid}/runs", headers=h).status_code == 404
    assert _run(_load(_db(deck, "alice-agent"), jid)).enabled is True
    # An admin reaches it.
    r = c.post(f"/scheduler/agent-jobs/alice-agent/{jid}/pause", headers=_bearer(ADMIN, role="admin"))
    assert r.status_code == 200


def test_unknown_job_or_action_or_odd_id(deck):
    _run(_make_job(_db(deck, "alice-agent")))
    c, h = _client(), _bearer(ALICE)
    assert c.post("/scheduler/agent-jobs/alice-agent/nope/pause", headers=h).status_code == 404
    assert c.delete("/scheduler/agent-jobs/alice-agent/nope", headers=h).status_code == 404
    assert c.post("/scheduler/agent-jobs/alice-agent/x.y/pause", headers=h).status_code == 404
    jid = c.get("/scheduler/agent-jobs", headers=h).json()[0]["id"]
    assert c.post(f"/scheduler/agent-jobs/alice-agent/{jid}/explode", headers=h).status_code == 404


# ── Run now ───────────────────────────────────────────────────────────


def test_run_now_readies_the_agent_then_asks_it(deck, monkeypatch):
    jid = _run(_make_job(_db(deck, "alice-agent")))
    readied: list[str] = []

    async def ready(slug, timeout=0):
        readied.append(slug)
        deck["registry"]["alice-agent"]["web_port"] = 41009  # came back on another port
        return True

    monkeypatch.setattr(agent_idle, "ready_for_task", ready)
    r = _client().post(f"/scheduler/agent-jobs/alice-agent/{jid}/run", headers=_bearer(ALICE))
    assert r.json() == {"ok": True, "status": "started"}
    assert readied == ["alice-agent"]
    assert deck["rest"] == [("tok-a", "POST", f"/api/cron/jobs/{jid}/run")]


def test_run_now_when_the_agent_cannot_start(deck, monkeypatch):
    jid = _run(_make_job(_db(deck, "alice-agent")))

    async def not_ready(slug, timeout=0):
        return False

    monkeypatch.setattr(agent_idle, "ready_for_task", not_ready)
    r = _client().post(f"/scheduler/agent-jobs/alice-agent/{jid}/run", headers=_bearer(ALICE))
    assert r.status_code == 503 and "Scout" in r.json()["detail"]
    assert deck["rest"] == []


# ── Runs ──────────────────────────────────────────────────────────────


def _ev(minutes_ago: float, role: str, content: str, **meta):
    at = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    return {"timestamp": _iso(at), "role": role, "content": content, **meta}


def test_runs_from_history():
    history = [
        _ev(60, "system", "[CRON] job start trigger=scheduled kind=prompt session=s", trigger="scheduled"),
        _ev(60, "user", "Check the inbox"),
        _ev(58, "assistant", "Two replies: Linus and Devang."),
        _ev(58, "system", "[CRON] job done trigger=scheduled kind=prompt session=s"),
        _ev(40, "system", "[CRON] job start trigger=manual kind=prompt session=s", trigger="manual"),
        _ev(40, "system", "[CRON] queued followup for busy session=s mode=queue"),
        _ev(40, "system", "[CRON] job queued trigger=manual kind=prompt session=s"),
        _ev(35, "assistant", "No new replies."),
        _ev(20, "system", "[CRON] job start trigger=scheduled kind=prompt session=s"),
        _ev(20, "system", "[CRON] job failed: model timed out"),
        _ev(10, "system", "[CRON] job start trigger=scheduled kind=prompt session=s"),
        _ev(5, "system", "[CRON] job start trigger=scheduled kind=prompt session=s"),
    ]
    runs = acr.runs_from_history(history, alive=True)
    assert [r["status"] for r in runs] == ["running", "interrupted", "error", "ok", "ok"]
    assert [r["n"] for r in runs] == [5, 4, 3, 2, 1]
    assert runs[2]["error"] == "model timed out"
    assert runs[3]["result"] == "No new replies." and runs[3]["trigger"] == "manual"
    assert runs[4]["result"] == "Two replies: Linus and Devang."
    assert runs[4]["finished_at"] - runs[4]["started_at"] == pytest.approx(120, abs=2)
    # The agent is down: the run that never ended didn't finish.
    assert acr.runs_from_history(history, alive=False)[0]["status"] == "interrupted"


def test_runs_route(deck):
    db = _db(deck, "alice-agent")
    jid = _run(_make_job(db, history=[
        {"chat_event": _ev(3, "system", "[CRON] job start trigger=scheduled kind=prompt session=s")},
        {"chat_event": _ev(2, "assistant", "Done: 5 leads contacted.")},
        {"chat_event": _ev(2, "system", "[CRON] job done trigger=scheduled kind=prompt session=s")},
    ]))
    runs = _client().get(f"/scheduler/agent-jobs/alice-agent/{jid}/runs", headers=_bearer(ALICE)).json()
    assert len(runs) == 1 and runs[0]["status"] == "ok"
    assert runs[0]["result"] == "Done: 5 leads contacted."


# ── Where the agent keeps its file ────────────────────────────────────


def test_the_agents_config_says_where_its_file_is(deck):
    agent = deck["dir"] / "alice-agent"
    elsewhere = agent / "custom" / "s.db"
    _run(_make_job(elsewhere, text="from the configured file"))
    _run(_make_job(_db(deck, "alice-agent"), text="from the default file"))
    (agent / "config.yaml").write_text(f"session:\n  path: {elsewhere}\n")
    jobs = _client().get("/scheduler/agent-jobs", headers=_bearer(ALICE)).json()
    assert [j["task"] for j in jobs] == ["from the configured file"]
    assert jobs[0]["agent_name"] == "Scout"
    # The home config wins, as when the agent starts; "~" is the agent's HOME.
    home_cc = agent / "data" / "home-config-parent" / ".captain-claw"
    home_cc.mkdir(parents=True)
    _run(_make_job(home_cc / "sessions.db", text="from its home"))
    (home_cc / "config.yaml").write_text("session:\n  path: ~/.captain-claw/sessions.db\n")
    jobs = _client().get("/scheduler/agent-jobs", headers=_bearer(ALICE)).json()
    assert [j["task"] for j in jobs] == ["from its home"]


def test_without_a_config_the_home_default_is_used(deck):
    home_db = deck["dir"] / "alice-agent" / "data" / "home-config-parent" / ".captain-claw" / "sessions.db"
    _run(_make_job(home_db, text="no config"))
    jobs = _client().get("/scheduler/agent-jobs", headers=_bearer(ALICE)).json()
    assert [j["task"] for j in jobs] == ["no config"]


def test_without_a_path_the_agents_home_default_wins(deck):
    agent = deck["dir"] / "alice-agent"
    _run(_make_job(_db(deck, "alice-agent"), text="fd default"))
    (agent / "config.yaml").write_text("model:\n  provider: ollama\n")  # no session.path
    home_db = agent / "data" / "home-config-parent" / ".captain-claw" / "sessions.db"
    _run(_make_job(home_db, text="home default"))
    jobs = _client().get("/scheduler/agent-jobs", headers=_bearer(ALICE)).json()
    assert [j["task"] for j in jobs] == ["home default"]
