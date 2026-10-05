"""Scheduler REST API tenancy: every job has an owner and the routes are
owner-scoped.

A job carries its owner's prompt, delivery target (WhatsApp number / Telegram
chat) and optionally an agent auth override, so on a multi-user deck user B
must not list, read, change, delete or fire user A's jobs. Admins see every
job; ownerless (legacy / system) jobs are admin-only. ``agent_auth`` is never
returned, and an update that doesn't change it keeps the stored value.

Driven through a bare FastAPI app with the scheduler router and a temp store,
from a non-loopback client unless a test says otherwise.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import agent_secret, auth
from captain_claw.flight_deck import fd_scheduler as sched

ALICE, BOB, ADMIN = "user-alice", "user-bob", "user-admin"
SECRET = "deck-agent-secret"


@pytest.fixture()
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for var in ("FD_LOCKDOWN", "FD_GLASSES_BRIDGE_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", SECRET)
    agent_secret.reset_cache_for_tests()
    s = sched.SchedulerStore(db_path=tmp_path / "scheduler.db")
    monkeypatch.setattr(sched, "_STORE", s)
    yield s
    agent_secret.reset_cache_for_tests()


@pytest.fixture()
def agents(monkeypatch):
    """This deck's agents: web_auth token → (owner, slug)."""
    import captain_claw.flight_deck.server as srv
    registry = {"tok-alice": (ALICE, "alice-agent"), "tok-bob": (BOB, "bob-agent"),
                "tok-orphan": ("", "orphan-agent"), "tok-local": ("local", "local-agent")}
    monkeypatch.setattr(
        srv, "_find_agent_by_auth",
        lambda t: (True, *registry[t]) if t in registry else (False, "", ""),
    )
    return registry


@pytest.fixture()
def fired(monkeypatch):
    """Stub out execution: record which jobs run-now actually fired."""
    calls: list[str] = []

    async def _fake_execute(job, *, force=False):
        calls.append(job["id"])
        return ("ok", "reply")

    monkeypatch.setattr(sched, "execute_job", _fake_execute)
    return calls


def _client(host: str = "10.0.0.5") -> TestClient:
    app = FastAPI()
    app.include_router(sched.router)
    return TestClient(app, client=(host, 50123))


def _bearer(uid: str, role: str = "user") -> dict[str, str]:
    return {"Authorization": f"Bearer {auth.create_access_token(uid, role=role)}"}


_JOB = {
    "name": "Morning briefing",
    "schedule": "daily 08:00",
    "agent_slug": "alice-agent",
    "agent_auth": "s3cret-agent-token",
    "prompt": "Compile my morning briefing",
    "delivery_kind": "whatsapp",
    "delivery_target": "385900000001",
    "enabled": True,
}


def _create(c: TestClient, headers: dict[str, str], **over) -> dict:
    r = c.post("/scheduler/jobs", json={**_JOB, **over}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


# ── owner stamping ────────────────────────────────────────────────────


def test_create_stamps_owner_from_jwt_and_ignores_body_owner(store):
    c = _client()
    job = _create(c, _bearer(BOB), owner_id=ALICE)
    assert store.get(job["id"])["owner_id"] == BOB
    assert job["owner_id"] == BOB


def test_update_cannot_change_owner(store):
    c = _client()
    job = _create(c, _bearer(BOB))
    r = c.patch(f"/scheduler/jobs/{job['id']}", json={"owner_id": ALICE}, headers=_bearer(BOB))
    assert r.status_code == 200
    assert store.get(job["id"])["owner_id"] == BOB


def test_internal_caller_stamps_the_calling_agents_owner(store, agents):
    c = _client()
    job = _create(c, {"X-Agent-Secret": SECRET, "X-Agent-Auth": "tok-bob"})
    assert store.get(job["id"])["owner_id"] == BOB


def test_loopback_agent_stamps_its_owner(store, agents):
    c = _client("127.0.0.1")
    job = _create(c, {"X-Agent-Auth": "tok-alice"})
    assert store.get(job["id"])["owner_id"] == ALICE


@pytest.mark.parametrize("headers", [
    {"X-Agent-Secret": SECRET},                               # unidentified
    {"X-Agent-Secret": SECRET, "X-Agent-Auth": "tok-nobody"},  # not this deck's
    {"X-Agent-Secret": SECRET, "X-Agent-Auth": "tok-orphan"},  # no recorded owner
    {"X-Agent-Secret": SECRET, "X-Agent-Auth": "tok-local"},   # auth-off synthetic owner
])
def test_unidentified_internal_caller_creates_a_system_job(store, agents, headers):
    c = _client()
    job = _create(c, headers)
    assert store.get(job["id"])["owner_id"] == ""


# ── owner scoping ─────────────────────────────────────────────────────


def test_user_b_cannot_see_or_touch_user_a_job(store, fired):
    c = _client()
    job = _create(c, _bearer(ALICE))
    jid = job["id"]
    bob = _bearer(BOB)

    r = c.get("/scheduler/jobs", headers=bob)
    assert r.status_code == 200
    assert [j["id"] for j in r.json()] == []

    assert c.get(f"/scheduler/jobs/{jid}", headers=bob).status_code == 404
    r = c.patch(f"/scheduler/jobs/{jid}",
                json={"prompt": "pwned", "delivery_target": "385999999999"}, headers=bob)
    assert r.status_code == 404
    r = c.patch(f"/scheduler/jobs/{jid}", json={"enabled": False}, headers=bob)
    assert r.status_code == 404
    assert c.post(f"/scheduler/jobs/{jid}/run", headers=bob).status_code == 404
    assert c.delete(f"/scheduler/jobs/{jid}", headers=bob).status_code == 404

    row = store.get(jid)
    assert row is not None
    assert row["prompt"] == _JOB["prompt"]
    assert row["delivery_target"] == _JOB["delivery_target"]
    assert row["enabled"] == 1
    assert fired == []


def test_owner_manages_own_job(store, fired):
    c = _client()
    alice = _bearer(ALICE)
    job = _create(c, alice)
    jid = job["id"]

    assert [j["id"] for j in c.get("/scheduler/jobs", headers=alice).json()] == [jid]
    assert c.get(f"/scheduler/jobs/{jid}", headers=alice).status_code == 200
    r = c.patch(f"/scheduler/jobs/{jid}", json={"enabled": False}, headers=alice)
    assert r.status_code == 200 and r.json()["enabled"] == 0
    assert c.post(f"/scheduler/jobs/{jid}/run", headers=alice).status_code == 200
    assert fired == [jid]
    assert c.delete(f"/scheduler/jobs/{jid}", headers=alice).status_code == 200
    assert store.get(jid) is None


def test_admin_sees_and_manages_every_job_including_legacy(store, fired):
    c = _client()
    a = _create(c, _bearer(ALICE))
    b = _create(c, _bearer(BOB))
    legacy = store.create(schedule="every 1h", prompt="legacy", delivery_kind="channel",
                          delivery_target="c")  # pre-owner row: owner ""
    admin = _bearer(ADMIN, role="admin")

    ids = {j["id"] for j in c.get("/scheduler/jobs", headers=admin).json()}
    assert ids == {a["id"], b["id"], legacy["id"]}
    assert c.get(f"/scheduler/jobs/{a['id']}", headers=admin).status_code == 200
    r = c.patch(f"/scheduler/jobs/{b['id']}", json={"name": "renamed"}, headers=admin)
    assert r.status_code == 200 and r.json()["name"] == "renamed"
    assert store.get(b["id"])["owner_id"] == BOB  # an admin edit keeps the owner
    assert c.post(f"/scheduler/jobs/{legacy['id']}/run", headers=admin).status_code == 200
    assert c.delete(f"/scheduler/jobs/{a['id']}", headers=admin).status_code == 200


def test_legacy_ownerless_job_is_admin_only(store):
    legacy = store.create(schedule="every 1h", prompt="legacy", delivery_kind="channel",
                          delivery_target="c")
    c = _client()
    bob = _bearer(BOB)
    assert c.get("/scheduler/jobs", headers=bob).json() == []
    assert c.get(f"/scheduler/jobs/{legacy['id']}", headers=bob).status_code == 404
    assert c.delete(f"/scheduler/jobs/{legacy['id']}", headers=bob).status_code == 404


def test_identified_agent_sees_only_its_owners_jobs(store, agents):
    c = _client()
    a = _create(c, _bearer(ALICE))
    _create(c, _bearer(BOB))
    r = c.get("/scheduler/jobs", headers={"X-Agent-Secret": SECRET, "X-Agent-Auth": "tok-alice"})
    assert [j["id"] for j in r.json()] == [a["id"]]


def test_unidentified_internal_caller_sees_no_jobs(store, agents):
    """Dropping X-Agent-Auth must never widen an agent's view: an unidentified
    internal caller can create (intentions' follow-through) but reads nothing,
    not even ownerless jobs."""
    c = _client()
    _create(c, _bearer(ALICE))
    sys_job = _create(c, {"X-Agent-Secret": SECRET})
    internal = {"X-Agent-Secret": SECRET}
    assert c.get("/scheduler/jobs", headers=internal).json() == []
    assert c.get(f"/scheduler/jobs/{sys_job['id']}", headers=internal).status_code == 404
    assert c.delete(f"/scheduler/jobs/{sys_job['id']}", headers=internal).status_code == 404


def test_unauthenticated_caller_is_rejected(store):
    c = _client()
    assert c.get("/scheduler/jobs").status_code == 401
    assert c.post("/scheduler/jobs", json=_JOB).status_code == 401


def test_stale_user_token_on_loopback_is_rejected_not_downgraded(store):
    """An expired/invalid session token must 401 even from loopback — not fall
    through to the internal branch, where the user would see an empty list and
    create ownerless jobs they can't see."""
    import jwt

    expired = jwt.encode(
        {"sub": ALICE, "role": "user", "type": "access", "iat": 0, "exp": 1},
        auth.get_jwt_secret(), algorithm=auth.ALGORITHM,
    )
    c = _client("127.0.0.1")
    for tok in (expired, "not-a-jwt"):
        hdr = {"Authorization": f"Bearer {tok}"}
        assert c.get("/scheduler/jobs", headers=hdr).status_code == 401
        assert c.post("/scheduler/jobs", json=_JOB, headers=hdr).status_code == 401
    assert store.list() == []


# ── agent_auth redaction ──────────────────────────────────────────────


def test_agent_auth_is_never_returned(store):
    c = _client()
    alice = _bearer(ALICE)
    r = c.post("/scheduler/jobs", json=_JOB, headers=alice)
    jid = r.json()["id"]
    texts = [
        r.text,
        c.get("/scheduler/jobs", headers=alice).text,
        c.get(f"/scheduler/jobs/{jid}", headers=alice).text,
        c.patch(f"/scheduler/jobs/{jid}", json={"name": "x"}, headers=alice).text,
        c.get("/scheduler/jobs", headers=_bearer(ADMIN, role="admin")).text,
    ]
    for t in texts:
        assert _JOB["agent_auth"] not in t
    # …but the job still knows it has one, and the runner still gets the real value.
    assert c.get(f"/scheduler/jobs/{jid}", headers=alice).json()["agent_auth"]
    assert store.get(jid)["agent_auth"] == _JOB["agent_auth"]


def test_job_without_agent_auth_reports_none(store):
    c = _client()
    job = _create(c, _bearer(ALICE), agent_auth="")
    assert job["agent_auth"] == ""


def test_update_keeps_stored_agent_auth(store):
    c = _client()
    alice = _bearer(ALICE)
    jid = _create(c, alice)["id"]

    # Toggle (omits agent_auth).
    assert c.patch(f"/scheduler/jobs/{jid}", json={"enabled": False}, headers=alice).status_code == 200
    assert store.get(jid)["agent_auth"] == _JOB["agent_auth"]

    # The SchedulerPage edit form round-trips the redacted value it was given.
    shown = c.get(f"/scheduler/jobs/{jid}", headers=alice).json()
    form = {k: shown[k] for k in ("name", "schedule", "agent_slug", "agent_auth", "prompt",
                                  "flow_id", "delivery_kind", "delivery_target",
                                  "ignore_quiet_hours")}
    form["name"] = "Edited"
    r = c.patch(f"/scheduler/jobs/{jid}", json=form, headers=alice)
    assert r.status_code == 200 and r.json()["name"] == "Edited"
    assert store.get(jid)["agent_auth"] == _JOB["agent_auth"]

    # A new value replaces it; clearing the field clears it.
    c.patch(f"/scheduler/jobs/{jid}", json={"agent_auth": "rotated"}, headers=alice)
    assert store.get(jid)["agent_auth"] == "rotated"
    c.patch(f"/scheduler/jobs/{jid}", json={"agent_auth": ""}, headers=alice)
    assert store.get(jid)["agent_auth"] == ""


def test_create_with_the_redaction_placeholder_stores_no_auth(store):
    """The placeholder a GET shows is never stored as a real token."""
    c = _client()
    job = _create(c, _bearer(ALICE), agent_auth=sched._AGENT_AUTH_MASK)
    assert store.get(job["id"])["agent_auth"] == ""


# ── auth-disabled (single trusted user) ───────────────────────────────


def test_auth_disabled_deck_sees_every_job(store, monkeypatch, fired):
    a = store.create(schedule="every 1h", prompt="a", delivery_kind="channel",
                     delivery_target="c", agent_auth="tok")
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    c = _client()
    jobs = c.get("/scheduler/jobs").json()
    assert [j["id"] for j in jobs] == [a["id"]]
    assert jobs[0]["agent_auth"] != "tok"
    assert c.patch(f"/scheduler/jobs/{a['id']}", json={"enabled": False}).status_code == 200
    assert c.post(f"/scheduler/jobs/{a['id']}/run").status_code == 200
    created = _create(c, {})
    assert store.get(created["id"])["owner_id"] == ""
    assert c.delete(f"/scheduler/jobs/{a['id']}").status_code == 200


# ── schema migration ──────────────────────────────────────────────────


def test_legacy_db_gains_owner_column(tmp_path):
    path = tmp_path / "scheduler.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE scheduler_jobs (
            id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '',
            schedule TEXT NOT NULL, agent_slug TEXT NOT NULL DEFAULT '',
            agent_auth TEXT NOT NULL DEFAULT '', prompt TEXT NOT NULL,
            delivery_kind TEXT NOT NULL, delivery_target TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            ignore_quiet_hours INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            next_run_at REAL, last_run_at REAL,
            last_status TEXT NOT NULL DEFAULT '', last_result TEXT NOT NULL DEFAULT ''
        );
        INSERT INTO scheduler_jobs (id, schedule, prompt, delivery_kind, delivery_target,
                                    created_at, updated_at)
        VALUES ('job_old', 'daily 08:00', 'hi', 'channel', 'c', 't', 't');
        """
    )
    conn.commit()
    conn.close()
    s = sched.SchedulerStore(db_path=path)
    assert s.get("job_old")["owner_id"] == ""
    assert s.list(owner_id="someone") == []
    assert [j["id"] for j in s.list()] == ["job_old"]
    # Re-opening (the ALTER already applied) is harmless.
    sched.SchedulerStore(db_path=path)


class _FakeFlowEngine:
    """A flow store + runner pair that records the context each run acts as."""

    def __init__(self, admins=()):
        self.calls: list[dict] = []
        self.admins = set(admins)

    async def get_flow(self, flow_id):
        return {"id": flow_id, "name": "f", "owner_id": ALICE}

    async def user_is_admin(self, uid):
        return uid in self.admins

    async def run(self, flow, payload=None, **ctx):
        self.calls.append(ctx)
        return {"status": "done", "output": ""}


@pytest.mark.parametrize("owner, admins, expected", [
    (BOB, (), {"owner_id": BOB, "is_admin": False}),
    (ADMIN, (ADMIN,), {"owner_id": ADMIN, "is_admin": True}),
    ("", (), {}),  # legacy ownerless job: acts as the flow's owner, as before
])
async def test_scheduled_flow_runs_as_the_job_owner(monkeypatch, owner, admins, expected):
    import captain_claw.flight_deck.server as srv
    engine = _FakeFlowEngine(admins)
    monkeypatch.setattr(srv.app.state, "flow_store", engine, raising=False)
    monkeypatch.setattr(srv.app.state, "flow_runner", engine, raising=False)
    status, _ = await sched._execute_flow_job({"id": "job_1", "owner_id": owner}, "flow_1")
    assert status == "ok:no-output"
    assert engine.calls == [expected]
