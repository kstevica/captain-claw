"""Judge vs human labels on Basna runs.

`basna_runs.success` is the judge's automatic label; a human thumbs vote lands in
`human_success` / `human_feedback_at` instead of overwriting it, so judge/human
pairs survive for agreement metrics. Consumers keep seeing the human override:
the run API's `success` is the effective label and reliability still follows
the vote.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import basna_routes
from captain_claw.flight_deck.auth import get_current_user, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

ARCH = "analyst"
DOMAIN = "finance"


@pytest.fixture
async def fd_db(tmp_path: Path):
    from captain_claw.flight_deck import auth as fd_auth
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


@pytest.fixture
async def uid(fd_db) -> str:
    return (await fd_db.create_user("u1@test.local", "x"))["id"]


def _client(user_id: str) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(basna_routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": user_id}
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://localhost:25080")


async def _session_with_run(db: FlightDeckDB, user_id: str) -> tuple[str, int]:
    sess = await db.create_basna_session(user_id, "size the EU market")
    await db.update_basna_session(sess["id"], user_id, domain=DOMAIN)
    [rid] = await db.add_basna_runs(sess["id"], user_id, [{
        "archetype_id": ARCH, "role": "Analyst", "output": "42B EUR", "success": None,
    }])
    return sess["id"], rid


async def _judge(db: FlightDeckDB, user_id: str, rid: int, success: bool) -> None:
    """What the execute path does once the truth is compiled."""
    await db.score_basna_run(rid, user_id, success)
    await db.record_archetype_outcome(user_id, ARCH, DOMAIN, success)


async def _raw_labels(db: FlightDeckDB, rid: int) -> tuple:
    async with db._db.execute(
        "SELECT success, human_success, human_feedback_at FROM basna_runs WHERE id = ?",
        (rid,),
    ) as cur:
        return tuple(await cur.fetchone())


async def _counts(db: FlightDeckDB, user_id: str) -> tuple[int, int]:
    [rel] = await db.get_archetype_reliability(user_id, DOMAIN)
    return rel["successes"], rel["fails"]


async def test_thumbs_vote_preserves_judge_label(fd_db, uid):
    sid, rid = await _session_with_run(fd_db, uid)
    await _judge(fd_db, uid, rid, True)

    async with _client(uid) as c:
        r = await c.post(f"/fd/basna/runs/{rid}/feedback", json={"success": False})
    assert r.status_code == 200
    body = r.json()
    assert body["changed"] is True
    assert body["success"] is False and body["judge_success"] == 1

    judge, human, at = await _raw_labels(fd_db, rid)
    assert judge == 1  # the judge's verdict survives the vote
    assert human == 0
    assert at


async def test_run_api_exposes_both_labels_with_human_override(fd_db, uid):
    sid, rid = await _session_with_run(fd_db, uid)
    await _judge(fd_db, uid, rid, True)
    async with _client(uid) as c:
        await c.post(f"/fd/basna/runs/{rid}/feedback", json={"success": False})
        [run] = (await c.get(f"/fd/basna/sessions/{sid}/runs")).json()
    assert run["success"] == 0  # effective label: the human vote wins
    assert run["judge_success"] == 1
    assert run["human_success"] == 0
    assert run["human_feedback_at"]


async def test_agent_runs_route_sees_human_override(fd_db, uid, monkeypatch):
    sid, rid = await _session_with_run(fd_db, uid)
    await _judge(fd_db, uid, rid, True)
    await fd_db.set_basna_run_human_label(rid, uid, False)
    monkeypatch.setattr(basna_routes, "_resolve_owner", lambda body: uid)
    async with _client(uid) as c:
        r = await c.post("/fd/basna/agent/runs", json={"session_id": sid})
    [run] = r.json()["runs"]
    assert run["success"] == 0
    assert run["judge_success"] == 1


async def test_unvoted_run_reports_judge_label(fd_db, uid):
    sid, rid = await _session_with_run(fd_db, uid)
    await _judge(fd_db, uid, rid, False)
    [run] = await fd_db.list_basna_runs(sid, uid)
    assert run["success"] == 0 and run["judge_success"] == 0
    assert run["human_success"] is None and run["human_feedback_at"] is None


async def test_vote_moves_reliability_like_before(fd_db, uid):
    sid, rid = await _session_with_run(fd_db, uid)
    await _judge(fd_db, uid, rid, True)
    assert await _counts(fd_db, uid) == (1, 0)

    async with _client(uid) as c:
        r = await c.post(f"/fd/basna/runs/{rid}/feedback", json={"success": False})
        assert r.json()["reliability"]["fails"] == 1
        assert await _counts(fd_db, uid) == (0, 1)  # moved, not double-counted
        # Flip back: the outcome returns to the success bucket.
        await c.post(f"/fd/basna/runs/{rid}/feedback", json={"success": True})
    assert await _counts(fd_db, uid) == (1, 0)
    judge, human, _ = await _raw_labels(fd_db, rid)
    assert (judge, human) == (1, 1)


async def test_vote_agreeing_with_judge_is_recorded_without_recount(fd_db, uid):
    sid, rid = await _session_with_run(fd_db, uid)
    await _judge(fd_db, uid, rid, True)
    async with _client(uid) as c:
        r = await c.post(f"/fd/basna/runs/{rid}/feedback", json={"success": True})
        assert r.json()["changed"] is True
        assert r.json()["reliability"] is None
        # The same vote again is a no-op.
        r2 = await c.post(f"/fd/basna/runs/{rid}/feedback", json={"success": True})
        assert r2.json()["changed"] is False
    assert await _counts(fd_db, uid) == (1, 0)
    judge, human, at = await _raw_labels(fd_db, rid)
    assert (judge, human) == (1, 1) and at  # an agreeing pair, kept for kappa


async def test_vote_on_unscored_run_then_judge_keeps_human_override(fd_db, uid):
    sid, rid = await _session_with_run(fd_db, uid)
    async with _client(uid) as c:
        await c.post(f"/fd/basna/runs/{rid}/feedback", json={"success": True})
    assert await _counts(fd_db, uid) == (1, 0)

    # A later judge pass (e.g. a resumed run) writes only its own label.
    await fd_db.score_basna_run(rid, uid, False)
    [run] = await fd_db.list_basna_runs(sid, uid)
    assert run["judge_success"] == 0
    assert run["human_success"] == 1
    assert run["success"] == 1


async def test_feedback_404_for_another_users_run(fd_db, uid):
    sid, rid = await _session_with_run(fd_db, uid)
    other = (await fd_db.create_user("u2@test.local", "x"))["id"]
    async with _client(other) as c:
        r = await c.post(f"/fd/basna/runs/{rid}/feedback", json={"success": False})
    assert r.status_code == 404
    assert await fd_db.set_basna_run_human_label(rid, other, False) is False
    assert (await _raw_labels(fd_db, rid))[1] is None


async def test_migration_adds_human_label_columns(tmp_path: Path):
    """A deck whose basna_runs predates the human columns gets them on init."""
    path = tmp_path / "old.db"
    async with aiosqlite.connect(path) as conn:
        await conn.execute("""
            CREATE TABLE basna_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                archetype_id TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                tier TEXT NOT NULL DEFAULT '',
                weight_at_run REAL NOT NULL DEFAULT 0.0,
                output TEXT NOT NULL DEFAULT '',
                success INTEGER,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )""")
        await conn.execute(
            "INSERT INTO basna_runs (session_id, success, created_at) VALUES ('s', 1, 'now')")
        await conn.commit()

    db = FlightDeckDB(path)
    await db.init()
    try:
        async with db._db.execute("PRAGMA table_info(basna_runs)") as cur:
            cols = {r["name"] for r in await cur.fetchall()}
        assert {"human_success", "human_feedback_at"} <= cols
        async with db._db.execute(
            "SELECT success, human_success FROM basna_runs") as cur:
            assert tuple(await cur.fetchone()) == (1, None)  # existing label untouched
    finally:
        await db.close()
