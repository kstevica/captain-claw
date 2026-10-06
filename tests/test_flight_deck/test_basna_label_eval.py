"""Judge-vs-human eval exporter: agreement math and the export routes."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import basna_routes, label_eval
from captain_claw.flight_deck.auth import get_current_user, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

# ── agreement math ───────────────────────────────────────────────────


def _pairs(both_success=0, both_fail=0, judge_only=0, human_only=0):
    return ([(1, 1)] * both_success + [(0, 0)] * both_fail
            + [(1, 0)] * judge_only + [(0, 1)] * human_only)


def test_kappa_matches_hand_computed_example():
    # p_o = 35/50 = .7; judge success 25/50, human 30/50 → p_e = .5·.6 + .5·.4 = .5
    a = label_eval.agreement(_pairs(20, 15, 5, 10))
    assert a["n"] == 50
    assert a["agreement"] == 0.7
    assert a["expected_agreement"] == 0.5
    assert a["kappa"] == 0.4
    assert a["judge_success_rate"] == 0.5 and a["human_success_rate"] == 0.6
    assert a["confusion"] == {"both_success": 20, "both_fail": 15,
                              "judge_success_human_fail": 5, "judge_fail_human_success": 10}


def test_kappa_perfect_and_worse_than_chance():
    assert label_eval.agreement(_pairs(3, 2))["kappa"] == 1.0
    assert label_eval.agreement(_pairs(0, 0, 3, 2))["kappa"] < 0


def test_kappa_undefined_when_every_label_is_the_same():
    a = label_eval.agreement(_pairs(both_success=4))
    assert a["agreement"] == 1.0 and a["kappa"] is None


def test_agreement_with_no_pairs():
    a = label_eval.agreement([])
    assert a["n"] == 0 and a["kappa"] is None and a["agreement"] is None


def test_export_row_shapes_mode_and_agree():
    row = {"id": 7, "session_id": "s", "config": json.dumps({"mode": "vatra"}),
           "domain": "", "judge_success": 1, "human_success": 0, "output": "x"}
    out = label_eval.export_row(row)
    assert out["mode"] == "vatra" and out["domain"] == "general"
    assert out["agree"] is False and "output" not in out
    assert label_eval.export_row({**row, "config": "not json"})["mode"] == "basna"
    assert label_eval.export_row({**row, "judge_success": None})["agree"] is None
    assert label_eval.export_row(row, include_text=True)["output"] == "x"


def test_summarize_groups_and_counts_unpaired():
    rows = [
        {"mode": "basna", "domain": "finance", "archetype_id": "a", "judge_success": 1, "human_success": 1},
        {"mode": "basna", "domain": "finance", "archetype_id": "b", "judge_success": 1, "human_success": 0},
        {"mode": "vatra", "domain": "law", "archetype_id": "a", "judge_success": 0, "human_success": 0},
        {"mode": "vatra", "domain": "law", "archetype_id": "a", "judge_success": None, "human_success": 1},
    ]
    s = label_eval.summarize(rows)
    assert s["overall"]["n"] == 3 and s["unpaired"] == 1
    assert [g["mode"] for g in s["by_mode"]] == ["basna", "vatra"]
    assert {g["archetype_id"]: g["n"] for g in s["by_archetype"]} == {"a": 2, "b": 1}


# ── routes ───────────────────────────────────────────────────────────


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


async def _seed(db: FlightDeckDB, user_id: str, labels: list[tuple], *,
                vatra: bool = False, domain: str = "finance") -> list[int]:
    """One session; a run per (judge, human) label — None leaves it unset."""
    cfg = json.dumps({"mode": "vatra"}) if vatra else "{}"
    sess = await db.create_basna_session(user_id, "size the market", config=cfg)
    await db.update_basna_session(sess["id"], user_id, domain=domain, truth="42B EUR")
    ids = await db.add_basna_runs(sess["id"], user_id, [
        {"archetype_id": f"arch{i}", "role": f"Role {i}", "output": f"answer {i}", "success": None}
        for i in range(len(labels))])
    for rid, (judge, human) in zip(ids, labels):
        if judge is not None:
            await db.score_basna_run(rid, user_id, bool(judge))
        if human is not None:
            await db.set_basna_run_human_label(rid, user_id, bool(human))
    return ids


def _jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


async def test_export_jsonl_has_only_complete_pairs(fd_db, uid):
    await _seed(fd_db, uid, [(1, 1), (1, 0), (None, 1), (1, None)])
    async with _client(uid) as c:
        r = await c.get("/fd/basna/eval/label-pairs")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    rows = _jsonl(r.text)
    assert [(x["judge_success"], x["human_success"], x["agree"]) for x in rows] == [
        (1, 1, True), (1, 0, False)]
    assert list(rows[0]) == label_eval.EXPORT_COLUMNS
    assert rows[0]["domain"] == "finance" and rows[0]["mode"] == "basna"


async def test_export_include_unpaired_and_text(fd_db, uid):
    await _seed(fd_db, uid, [(1, 0), (None, 1)])
    async with _client(uid) as c:
        r = await c.get("/fd/basna/eval/label-pairs",
                        params={"include_unpaired": "true", "include_text": "true"})
    rows = _jsonl(r.text)
    assert [x["judge_success"] for x in rows] == [1, None]
    assert rows[0]["output"] == "answer 0" and rows[0]["truth"] == "42B EUR"
    assert rows[0]["intent"] == "size the market"


async def test_export_csv(fd_db, uid):
    await _seed(fd_db, uid, [(0, 0), (1, 0)])
    async with _client(uid) as c:
        r = await c.get("/fd/basna/eval/label-pairs", params={"format": "csv"})
    assert r.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert list(rows[0]) == label_eval.EXPORT_COLUMNS
    assert [(x["judge_success"], x["human_success"]) for x in rows] == [("0", "0"), ("1", "0")]


async def test_vote_through_feedback_route_lands_in_export(fd_db, uid):
    [rid] = await _seed(fd_db, uid, [(1, None)])
    async with _client(uid) as c:
        await c.post(f"/fd/basna/runs/{rid}/feedback", json={"success": False})
        rows = _jsonl((await c.get("/fd/basna/eval/label-pairs")).text)
    assert [(x["run_id"], x["judge_success"], x["human_success"]) for x in rows] == [(rid, 1, 0)]
    assert rows[0]["human_feedback_at"]


async def test_agreement_endpoint(fd_db, uid):
    await _seed(fd_db, uid, [(1, 1), (0, 0), (1, 0), (None, 1)])
    await _seed(fd_db, uid, [(1, 1)], vatra=True, domain="law")
    async with _client(uid) as c:
        s = (await c.get("/fd/basna/eval/agreement")).json()
    assert s["overall"]["n"] == 4 and s["unpaired"] == 1
    assert s["overall"]["agreement"] == 0.75
    assert {g["mode"]: g["n"] for g in s["by_mode"]} == {"basna": 3, "vatra": 1}
    assert {g["domain"]: g["n"] for g in s["by_domain"]} == {"finance": 3, "law": 1}


async def test_since_drops_older_runs(fd_db, uid):
    old, new = await _seed(fd_db, uid, [(1, 1), (1, 0)])
    await fd_db._db.execute("UPDATE basna_runs SET created_at = ? WHERE id = ?",
                            ("2026-09-01T08:00:00+00:00", old))
    await fd_db._db.commit()
    async with _client(uid) as c:
        rows = _jsonl((await c.get("/fd/basna/eval/label-pairs",
                                   params={"since": "2026-10-01"})).text)
        assert [x["run_id"] for x in rows] == [new]
        # An offset timestamp is compared in UTC: 09:30+02:00 is 07:30Z, before `old`.
        rows = _jsonl((await c.get("/fd/basna/eval/label-pairs",
                                   params={"since": "2026-09-01T09:30:00+02:00"})).text)
        assert [x["run_id"] for x in rows] == [old, new]
        s = (await c.get("/fd/basna/eval/agreement", params={"since": "2026-10-01"})).json()
    assert s["overall"]["n"] == 1 and s["since"] == "2026-10-01"


async def test_export_is_owner_scoped(fd_db, uid):
    await _seed(fd_db, uid, [(1, 1)])
    other = (await fd_db.create_user("u2@test.local", "x"))["id"]
    async with _client(other) as c:
        assert (await c.get("/fd/basna/eval/label-pairs")).text == ""
        assert (await c.get("/fd/basna/eval/agreement")).json()["overall"]["n"] == 0


async def test_bad_params_are_rejected(fd_db, uid):
    async with _client(uid) as c:
        assert (await c.get("/fd/basna/eval/label-pairs", params={"format": "xlsx"})).status_code == 400
        assert (await c.get("/fd/basna/eval/agreement", params={"since": "last week"})).status_code == 400
