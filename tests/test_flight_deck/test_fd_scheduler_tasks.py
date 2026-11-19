"""Scheduler tasks (the kiosk's Scheduled tasks): zoned / weekday schedules,
"agent" delivery (chat, email, Telegram), the run log and its search, owner
checks on the agent a task drives, and the preview / options / run-now
routes."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import agent_secret, auth
from captain_claw.flight_deck import fd_scheduler as sched

ALICE, BOB = "user-alice", "user-bob"
SECRET = "deck-agent-secret"
BASE = datetime(2026, 5, 28, 12, 0, 0)  # a Thursday, noon


# ── schedules ─────────────────────────────────────────────────────────


def test_weekdays_skip_the_weekend():
    assert sched.compute_next_run("weekdays 08:00", base=BASE) == datetime(2026, 5, 29, 8).timestamp()
    fri_eve = datetime(2026, 5, 29, 18, 0)
    assert sched.compute_next_run("weekdays 08:00", base=fri_eve) == datetime(2026, 6, 1, 8).timestamp()


def test_weekly_on_several_days():
    assert sched.compute_next_run("weekly mon,wed 09:00", base=BASE) == datetime(2026, 6, 1, 9).timestamp()
    assert sched.compute_next_run("weekly tue, thu 18:00", base=BASE) == datetime(2026, 5, 28, 18).timestamp()
    with pytest.raises(sched.ScheduleError):
        sched.compute_next_run("weekly mon,xyz 09:00", base=BASE)


def test_wall_clock_times_are_in_the_jobs_zone():
    ny = sched.compute_next_run("daily 08:00", base=BASE, tz="America/New_York")
    zg = sched.compute_next_run("daily 08:00", base=BASE, tz="Europe/Zagreb")
    assert ny - zg == 6 * 3600                      # 08:00 in New York is six hours later
    once = sched.compute_next_run("once 2026-12-25T09:00:00", base=BASE, tz="Asia/Tokyo")
    assert once == datetime.fromisoformat("2026-12-25T09:00:00+09:00").timestamp()
    with pytest.raises(sched.ScheduleError):
        sched.validate_schedule("daily 08:00", tz="Mars/Olympus")


# ── delivery config ───────────────────────────────────────────────────


def test_delivery_is_normalized_and_validated():
    d = sched.norm_delivery({"chat": 1, "email": True, "email_to": "a@x.com; b@y.org, a@x.com"})
    assert d == {"chat": True, "email": True, "telegram": False, "email_to": "a@x.com, b@y.org"}
    assert sched.norm_delivery("garbage") == {"chat": False, "email": False, "telegram": False, "email_to": ""}
    with pytest.raises(ValueError):
        sched.norm_delivery({"email_to": "not-an-address"})


def test_an_agent_task_needs_no_target_but_one_delivery(tmp_path):
    store = sched.SchedulerStore(db_path=tmp_path / "s.db")
    row = store.create(name="Brief", schedule="daily 08:00", agent_slug="a", prompt="brief",
                       delivery_kind="agent", delivery={"chat": True}, tz="Europe/Zagreb")
    assert row["delivery_target"] == "" and row["tz"] == "Europe/Zagreb"
    with pytest.raises(ValueError):
        store.create(schedule="daily 08:00", agent_slug="a", prompt="p",
                     delivery_kind="agent", delivery={})
    with pytest.raises(ValueError):
        store.update(row["id"], delivery={"chat": False})
    with pytest.raises(sched.ScheduleError):  # a one-shot in the past
        store.create(schedule="once 2020-01-01T08:00:00", agent_slug="a", prompt="p",
                     delivery_kind="agent", delivery={"chat": True})


# ── the run itself ────────────────────────────────────────────────────


@pytest.fixture()
def store(monkeypatch, tmp_path):
    s = sched.SchedulerStore(db_path=tmp_path / "scheduler.db")
    monkeypatch.setattr(sched, "_STORE", s)
    return s


@pytest.fixture()
def agent_up(monkeypatch):
    """A running agent that answers, recording what the scheduler sent."""
    seen: dict = {}
    monkeypatch.setattr(sched, "resolve_agent_by_slug", lambda *a, **k: ("127.0.0.1", 1234, "tok"))

    async def _capture(*, host, port, auth, prompt, automation=None, meta=None, **kw):
        seen.update(prompt=prompt, automation=automation)
        if meta is not None:
            meta["task_session"] = {"id": "sess-7", "name": "⏰ Brief"}
        return "Three meetings today."

    monkeypatch.setattr(sched, "run_prompt_and_capture", _capture)
    return seen


def _task(store, **over):
    fields = {"name": "Brief", "schedule": "daily 08:00", "agent_slug": "alice-agent",
              "prompt": "Brief me", "delivery_kind": "agent", "delivery": {"chat": True},
              **over}
    return store.create(owner_id=ALICE, **fields)


async def test_a_chat_task_lets_the_agent_post_it_and_logs_the_run(store, agent_up):
    job = _task(store)
    status, text, run_id = await sched.run_job(job, trigger="manual", force=True)
    assert status == "ok" and text == "Three meetings today."
    auto = agent_up["automation"]
    assert auto["fd_delivers"] is False                         # the mirror IS the delivery
    assert auto["task"] == {"id": job["id"], "name": "Brief", "session_id": ""}
    assert "This is a scheduled task" in agent_up["prompt"] and agent_up["prompt"].endswith("Brief me")
    run = store.get_run(run_id)
    assert run["status"] == "ok" and run["result"] == "Three meetings today."
    assert run["deliveries"][0]["via"] == "chat" and run["session_id"] == "sess-7"
    assert store.get(job["id"])["agent_session_id"] == "sess-7"   # the next run goes there
    assert store.get(job["id"])["last_status"] == "ok"


async def test_an_email_only_task_is_not_mirrored_and_reports_each_delivery(store, agent_up, monkeypatch):
    sent: list = []

    async def _email(job, delivery, text):
        sent.append(("email", text))
        return {"via": "email", "ok": True, "note": "sent from a@x.com", "to": "a@x.com"}

    async def _tg(owner, text):
        return {"via": "telegram", "ok": False, "note": "Telegram isn't linked"}

    monkeypatch.setattr(sched, "_deliver_email", _email)
    monkeypatch.setattr(sched, "_deliver_telegram_owner", _tg)
    job = _task(store, delivery={"email": True, "telegram": True})
    status, _, run_id = await sched.run_job(job, force=True)
    assert agent_up["automation"]["fd_delivers"] is True
    assert agent_up["automation"]["task"]["mirror"] is False
    assert sent == [("email", "Three meetings today.")]
    assert status == "partial: telegram failed"
    assert [d["via"] for d in store.get_run(run_id)["deliveries"]] == ["email", "telegram"]


async def test_a_task_never_drives_someone_elses_agent(store, agent_up, monkeypatch):
    monkeypatch.setattr(sched, "_agent_owner", lambda slug: BOB)

    async def _not_admin(uid):
        return False
    monkeypatch.setattr(sched, "_user_is_admin", _not_admin)
    status, _, _ = await sched.run_job(_task(store), force=True)
    assert status == "error:agent-not-yours" and "automation" not in agent_up


async def test_a_long_run_is_claimed_before_it_runs(store, monkeypatch):
    job = store.create(owner_id=ALICE, name="t", schedule="every 1h", agent_slug="a",
                       prompt="p", delivery_kind="agent", delivery={"chat": True})
    store.claim(job["id"], next_run_at=1.0)               # due
    gate = asyncio.Event()

    async def _slow(job, *, force=False):
        await gate.wait()
        return ("ok", "done")

    monkeypatch.setattr(sched, "execute_job", _slow)
    (due,) = store.list_due(10.0)
    task = asyncio.create_task(sched._run_due_job(store, due))
    await asyncio.sleep(0.05)
    assert store.list_due(10.0) == []                       # not due again while running
    assert store.run_stats()[job["id"]]["running"] is True
    gate.set()
    await task
    assert store.list_runs(job_id=job["id"])[0]["status"] == "ok"


async def test_runs_on_one_agent_queue_but_other_agents_run_alongside(store, monkeypatch):
    active: dict[str, int] = {}
    peak: dict[str, int] = {}

    async def _work(job, *, force=False):
        slug = job["agent_slug"]
        active[slug] = active.get(slug, 0) + 1
        peak[slug] = max(peak.get(slug, 0), active[slug])
        await asyncio.sleep(0.05)
        active[slug] -= 1
        return ("ok", "done")

    monkeypatch.setattr(sched, "execute_job", _work)
    jobs = [_task(store, agent_slug=s) for s in ("a1", "a1", "a2")]
    started = asyncio.get_running_loop().time()
    await asyncio.gather(*(sched.run_job(j, force=True) for j in jobs))
    assert peak == {"a1": 1, "a2": 1}
    assert asyncio.get_running_loop().time() - started < 0.15   # a2 didn't wait for a1


def test_runs_search_and_pruning(store):
    a = _task(store, name="Morning briefing")
    b = _task(store, name="Inbox sweep")
    for jid, text in ((a["id"], "Rain in Zagreb, 3 meetings"), (b["id"], "Two invoices from Acme_Co")):
        store.finish_run(store.start_run(store.get(jid), "scheduled"), status="ok", result=text)
    assert [r["job_id"] for r in store.list_runs(q="zagreb")] == [a["id"]]
    assert [r["job_id"] for r in store.list_runs(q="inbox")] == [b["id"]]     # by task name
    assert store.list_runs(q="acme_co")[0]["job_id"] == b["id"]
    assert store.list_runs(q="acme%")  == []                                 # % is literal
    assert store.list_runs(owner_id=BOB) == []
    monkey_kept = sched._RUNS_KEPT_PER_JOB
    try:
        sched._RUNS_KEPT_PER_JOB = 2
        for _ in range(3):
            store.finish_run(store.start_run(store.get(a["id"]), "scheduled"), status="ok", result="x")
        assert len(store.list_runs(job_id=a["id"])) == 2
    finally:
        sched._RUNS_KEPT_PER_JOB = monkey_kept
    store.delete(a["id"])
    assert store.list_runs(job_id=a["id"]) == []


def test_an_interrupted_run_is_closed_on_restart(tmp_path):
    s = sched.SchedulerStore(db_path=tmp_path / "s.db")
    job = s.create(schedule="daily 08:00", agent_slug="a", prompt="p",
                   delivery_kind="agent", delivery={"chat": True})
    rid = s.start_run(job, "scheduled")
    s2 = sched.SchedulerStore(db_path=tmp_path / "s.db")
    assert s2.get_run(rid)["status"] == "error:interrupted"


# ── routes ────────────────────────────────────────────────────────────


@pytest.fixture()
def api(monkeypatch, store):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for var in ("FD_LOCKDOWN", "FD_GLASSES_BRIDGE_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", SECRET)
    agent_secret.reset_cache_for_tests()
    owners = {"alice-agent": ALICE, "bob-agent": BOB}
    monkeypatch.setattr(sched, "_agent_owner", lambda slug: owners.get(slug))
    app = FastAPI()
    app.include_router(sched.router)
    yield TestClient(app, client=("10.0.0.5", 50123))
    agent_secret.reset_cache_for_tests()


def _bearer(uid: str, role: str = "user") -> dict[str, str]:
    return {"Authorization": f"Bearer {auth.create_access_token(uid, role=role)}"}


_BODY = {"name": "Brief", "schedule": "weekdays 08:00", "tz": "Europe/Zagreb",
         "agent_slug": "alice-agent", "prompt": "Brief me", "delivery_kind": "agent",
         "delivery": {"chat": True, "email": True, "email_to": "boss@example.com"}}


def test_create_returns_the_delivery_and_refuses_another_users_agent(api):
    r = api.post("/scheduler/jobs", json=_BODY, headers=_bearer(ALICE))
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["delivery"]["email_to"] == "boss@example.com" and job["run_count"] == 0
    assert job["running"] is False
    r = api.post("/scheduler/jobs", json={**_BODY, "agent_slug": "bob-agent"}, headers=_bearer(ALICE))
    assert r.status_code == 403
    r = api.patch(f"/scheduler/jobs/{job['id']}", json={"agent_slug": "bob-agent"}, headers=_bearer(ALICE))
    assert r.status_code == 403
    # nor may an admin: Bob's agent is Bob's to schedule
    assert api.post("/scheduler/jobs", json={**_BODY, "agent_slug": "bob-agent"},
                    headers=_bearer("adm", role="admin")).status_code == 403


def test_an_agent_cannot_choose_who_a_task_emails(api, store, monkeypatch):
    import captain_claw.flight_deck.server as srv
    monkeypatch.setattr(srv, "_find_agent_by_auth",
                        lambda t: (True, ALICE, "alice-agent") if t == "tok-a" else (False, "", ""))
    r = api.post("/scheduler/jobs", json=_BODY,
                 headers={"X-Agent-Secret": SECRET, "X-Agent-Auth": "tok-a"})
    assert r.status_code == 200, r.text
    assert store.get(r.json()["id"])["owner_id"] == ALICE
    assert r.json()["delivery"]["email_to"] == ""          # the owner's own address


def test_runs_are_owner_scoped_and_searchable(api, store):
    mine = _task(store, name="Mine")
    store.finish_run(store.start_run(mine, "scheduled"), status="ok", result="weather is sunny")
    theirs = store.create(owner_id=BOB, schedule="daily 08:00", agent_slug="bob-agent", prompt="p",
                          delivery_kind="agent", delivery={"chat": True})
    store.finish_run(store.start_run(theirs, "scheduled"), status="ok", result="sunny too")
    rows = api.get("/scheduler/runs?q=sunny", headers=_bearer(ALICE)).json()
    assert [r["job_id"] for r in rows] == [mine["id"]]
    assert api.get(f"/scheduler/runs?job_id={theirs['id']}", headers=_bearer(ALICE)).status_code == 404
    # An admin sees neither person's runs — only the deck's ownerless (system) ones.
    assert api.get("/scheduler/runs?q=sunny", headers=_bearer("adm", role="admin")).json() == []
    system = store.create(schedule="daily 09:00", prompt="p", delivery_kind="channel", delivery_target="c")
    store.finish_run(store.start_run(system, "scheduled"), status="ok", result="sunny system")
    rows = api.get("/scheduler/runs?q=sunny", headers=_bearer("adm", role="admin")).json()
    assert [r["job_id"] for r in rows] == [system["id"]]


def test_preview_lists_the_next_runs(api):
    r = api.post("/scheduler/preview", json={"schedule": "weekdays 08:00", "tz": "Europe/Zagreb"},
                 headers=_bearer(ALICE)).json()
    assert r["ok"] and len(r["next"]) == 3 and r["one_shot"] is False
    assert all(datetime.fromtimestamp(t).weekday() < 5 for t in r["next"])
    bad = api.post("/scheduler/preview", json={"schedule": "daily 25:00"}, headers=_bearer(ALICE)).json()
    assert bad["ok"] is False and bad["error"]
    past = api.post("/scheduler/preview", json={"schedule": "once 2020-01-01T08:00:00"},
                    headers=_bearer(ALICE)).json()
    assert past["ok"] is False and "passed" in past["error"]


def test_delivery_options(api, monkeypatch):
    async def _route(owner):
        return {"available": owner == ALICE, "provider": "google", "from": "a@x.com",
                "to": "a@x.com", "reason": "" if owner == ALICE else "Connect Google"}
    monkeypatch.setattr(sched, "_email_route", _route)
    monkeypatch.setattr(sched, "_owner_telegram_chat", lambda owner: "42" if owner == ALICE else "")
    r = api.get("/scheduler/delivery-options", headers=_bearer(ALICE)).json()
    assert r["chat"]["available"] and r["email"]["available"] and r["telegram"]["available"]
    r = api.get("/scheduler/delivery-options", headers=_bearer(BOB)).json()
    assert not r["email"]["available"] and not r["telegram"]["available"] and r["telegram"]["reason"]


def test_delivery_options_say_when_tasks_pause_for_an_away_owner(api, monkeypatch):
    from captain_claw.flight_deck import idle_pause

    async def _route(owner):
        return {"available": False, "provider": "", "from": "", "to": "", "reason": ""}
    monkeypatch.setattr(sched, "_email_route", _route)
    monkeypatch.setattr(sched, "_owner_telegram_chat", lambda owner: "")
    days = {"n": 5}

    async def _pause_days():
        return days["n"]
    monkeypatch.setattr(idle_pause, "pause_days", _pause_days)
    assert api.get("/scheduler/delivery-options", headers=_bearer(ALICE)).json()["idle_pause_days"] == 5
    days["n"] = 0
    assert api.get("/scheduler/delivery-options", headers=_bearer(ALICE)).json()["idle_pause_days"] == 0


def test_run_now_in_the_background_answers_at_once(api, store, monkeypatch):
    async def _fake(job, *, force=False):
        return ("ok", "done")
    monkeypatch.setattr(sched, "execute_job", _fake)
    job = _task(store)
    r = api.post(f"/scheduler/jobs/{job['id']}/run?background=1", headers=_bearer(ALICE)).json()
    assert r["status"] == "running" and r["run_id"]
    # The legacy (waiting) form still answers with the result.
    r = api.post(f"/scheduler/jobs/{job['id']}/run", headers=_bearer(ALICE)).json()
    assert r["status"] == "ok" and r["result_preview"] == "done" and r["run_id"]


def test_email_parts_render_markdown():
    subject, plain, html = sched._email_parts("Brief", "**Rain** today\n\n- a\n- b", datetime(2026, 5, 28, 8, 0))
    assert subject == "⏰ Brief — Thu 28 May"
    assert "<strong>Rain</strong>" in html and "<li>a</li>" in html
    assert plain.startswith("**Rain** today") and "Scheduled task “Brief”" in plain
