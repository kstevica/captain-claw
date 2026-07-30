"""Lupa BFF end-to-end against a fake Flight Deck.

The fake FD implements exactly the endpoints the BFF consumes (the "engine
contract" from docs/research-desk-product-plan.md). These tests prove the
whole commission flow — login proxy, stream, start → plan gate → approve →
progress → facts → report file — without a running Captain.
"""

from __future__ import annotations

import time

import httpx
import jwt as pyjwt
import pytest
from fastapi import FastAPI, Header, HTTPException, Request, Response

from lupa_api.server import create_app

SECRET = "test-jwt-secret"


def make_token(sub: str = "u1", role: str = "user") -> str:
    return pyjwt.encode({"sub": sub, "role": role, "type": "access",
                         "iat": int(time.time()), "exp": int(time.time()) + 3600},
                        SECRET, algorithm="HS256")


def make_fake_fd() -> tuple[FastAPI, dict]:
    """A minimal FD implementing the BFF's engine contract. Returns (app, log)
    — `log` records what FD received, for contract assertions."""
    fd = FastAPI()
    log: dict = {"quality": [], "sessions": {}}

    def _need_auth(authorization: str | None):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing token")

    @fd.post("/fd/auth/login")
    async def login(body: dict):
        resp = Response(
            content='{"access_token": "fd-access", "user": {"id": "u1"}}',
            media_type="application/json")
        resp.headers.append(
            "set-cookie",
            "fd_refresh=rt1; HttpOnly; Max-Age=604800; Path=/fd/auth; SameSite=lax")
        return resp

    @fd.get("/fd/auth/me")
    async def me(authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        return {"id": "u1", "email": "u1@test", "role": "user"}

    @fd.post("/fd/vatra/start")
    async def vatra_start(body: dict, authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        log["quality"].append(("start", body.get("quality")))
        sid = "sess-aaaa1111"
        log["sessions"][sid] = {
            "id": sid, "status": "planning", "intent": body["intent"],
            "title": body.get("title", ""), "truth": "",
            "route": {"group0_plan": {"steps": [{"agent": "deep-researcher",
                                                 "does": "scope the market"}]}},
            "analysis": {}, "config": {"mode": "vatra"}}
        return {"session_id": sid, "title": body.get("title", ""), "status": "planning"}

    @fd.post("/fd/vatra/plan/approve")
    async def plan_approve(body: dict, authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        log["quality"].append(("approve", body.get("quality")))
        sess = log["sessions"].get(body["session_id"])
        if not sess:
            raise HTTPException(404, "session not found")
        sess["status"] = "running"
        log["approved_plan"] = body.get("plan")
        return {"ok": True, "session_id": body["session_id"], "status": "running"}

    @fd.post("/fd/vatra/plan/cancel")
    async def plan_cancel(body: dict, authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        sess = log["sessions"].get(body["session_id"])
        if not sess or sess["status"] != "planning":
            raise HTTPException(409, "not at the plan gate")
        sess["status"] = "cancelled"
        return {"ok": True}

    @fd.post("/fd/vatra/sessions/{sid}/continue")
    async def vatra_continue(sid: str, body: dict,
                             authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        if sid not in log["sessions"]:
            raise HTTPException(404, "session not found")
        new_sid = "sess-bbbb2222"
        log["sessions"][new_sid] = {
            "id": new_sid, "status": "running", "intent": body["instruction"],
            "title": "round 2", "truth": "", "route": {}, "analysis": {},
            "config": {"mode": "vatra", "vfs_project": "vatra-sess-aaa",
                       "round": 2}}
        log["continued"] = {"parent": sid, **body}
        return {"ok": True, "session_id": new_sid, "round": 2}

    @fd.get("/fd/basna/sessions/{sid}")
    async def session_detail(sid: str, authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        sess = log["sessions"].get(sid)
        if not sess:
            raise HTTPException(404, "session not found")
        return sess

    @fd.get("/fd/basna/sessions/{sid}/progress")
    async def progress(sid: str, authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        if sid not in log["sessions"]:
            raise HTTPException(404, "session not found")
        return {"events": [{"i": 0, "stage": "phase", "message": "Planning"},
                           {"i": 1, "stage": "cost", "message": "$0.12",
                            "usd": 0.12, "hourly_usd": 3.4}],
                "active": log["sessions"][sid]["status"] == "running"}

    @fd.get("/fd/basna/sessions/{sid}/facts")
    async def facts(sid: str, authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        return {"project": f"vatra-{sid[:8]}", "count": 1, "conflicts": [],
                "facts": [{"key": "eu_market_size", "value": "42", "unit": "B EUR",
                           "status": "verified"}]}

    @fd.get("/fd/vfs/list")
    async def vfs_list(project: str, path: str = "",
                       authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        log["vfs_list_project"] = project
        return {"entries": [{"name": "r1-report.md", "dir": False, "size": 10}]}

    @fd.get("/fd/vfs/read")
    async def vfs_read(project: str, path: str,
                       authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        return {"text": f"# Report\nfrom {project}/{path}", "binary": False}

    @fd.get("/fd/costs")
    async def costs(authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        return {"costs": [{"run_kind": "vatra", "usd": 0.12}], "count": 1,
                "priced": 1, "total_usd": 0.12}

    return fd, log


@pytest.fixture
async def bff(tmp_path, monkeypatch):
    """(client, fd_log) — a lifespan-running BFF wired to the fake FD."""
    monkeypatch.setenv("LUPA_DATA_DIR", str(tmp_path / "lupa-data"))
    monkeypatch.setenv("FD_JWT_SECRET", SECRET)
    fake_fd, log = make_fake_fd()
    app = create_app(fd_transport=httpx.ASGITransport(app=fake_fd))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://lupa.test") as client:
            yield client, log


def _auth(sub: str = "u1") -> dict:
    return {"Authorization": f"Bearer {make_token(sub)}"}


async def test_pack_is_served_and_vertical_free(bff):
    client, _ = bff
    r = await client.get("/api/pack")
    assert r.status_code == 200
    pack = r.json()
    assert pack["slug"] == "research-desk"
    assert pack["vocabulary"]["stream"]
    assert pack["quality"] == {"profile": "thorough"}
    assert "onboarding_md" in pack


async def test_login_proxy_rewrites_cookie_path(bff):
    client, _ = bff
    r = await client.post("/api/auth/login", json={"email": "u1@test", "password": "x"})
    assert r.status_code == 200
    assert r.json()["access_token"] == "fd-access"
    sc = r.headers.get("set-cookie", "")
    assert "Path=/api/auth" in sc
    assert "Path=/fd/auth" not in sc


async def test_product_routes_require_token(bff):
    client, _ = bff
    assert (await client.get("/api/streams")).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert (await client.get("/api/streams", headers=bad)).status_code == 401


async def test_full_commission_flow(bff):
    client, log = bff
    h = _auth()

    # Stream.
    r = await client.post("/api/streams", json={"title": "EU heat-pump market"}, headers=h)
    assert r.status_code == 200
    stream_id = r.json()["id"]
    assert r.json()["pack"] == "research-desk"

    # Round 1: commission → FD start, pack quality attached, folder derived.
    r = await client.post(f"/api/streams/{stream_id}/commissions",
                          json={"brief": "Size the EU heat-pump market to 2030"},
                          headers=h)
    assert r.status_code == 200
    body = r.json()
    sid = body["session_id"]
    assert (body["round"], body["kind"], body["status"]) == (1, "initial", "planning")
    assert ("start", {"profile": "thorough"}) in log["quality"]

    # Plan gate: detail carries the drafted plan.
    r = await client.get(f"/api/commissions/{sid}", headers=h)
    assert r.json()["route"]["group0_plan"]["steps"][0]["agent"] == "deep-researcher"

    # Approve with an edited plan; pack quality rides approve too.
    edited = {"steps": [{"agent": "deep-researcher", "does": "scope EU27 only"}]}
    r = await client.post(f"/api/commissions/{sid}/approve",
                          json={"plan": edited}, headers=h)
    assert r.status_code == 200 and r.json()["status"] == "running"
    assert log["approved_plan"] == edited
    assert ("approve", {"profile": "thorough"}) in log["quality"]

    # Progress poll (cost event included).
    r = await client.get(f"/api/commissions/{sid}/progress", headers=h)
    assert any(e["stage"] == "cost" for e in r.json()["events"])

    # Facts (the receipts).
    r = await client.get(f"/api/commissions/{sid}/facts", headers=h)
    assert r.json()["facts"][0]["status"] == "verified"

    # Report reader: stream folder was derived from the round-1 session.
    r = await client.get(f"/api/streams/{stream_id}", headers=h)
    assert r.json()["vfs_project"] == f"vatra-{sid[:8]}"
    r = await client.get(f"/api/streams/{stream_id}/files", headers=h)
    assert r.json()["entries"][0]["name"] == "r1-report.md"
    assert log["vfs_list_project"] == f"vatra-{sid[:8]}"
    r = await client.get(f"/api/streams/{stream_id}/file",
                         params={"path": "r1-report.md"}, headers=h)
    assert r.json()["text"].startswith("# Report")

    # Round 2: continues the last session in the same folder.
    r = await client.post(f"/api/streams/{stream_id}/commissions",
                          json={"brief": "Deepen the Nordics"}, headers=h)
    assert r.status_code == 200
    assert (r.json()["round"], r.json()["kind"]) == (2, "continue")
    assert log["continued"]["parent"] == sid
    assert log["continued"]["same_cast"] is True

    # Costs passthrough.
    r = await client.get("/api/costs", headers=h)
    assert r.json()["total_usd"] == 0.12


async def test_commissions_are_owner_scoped(bff):
    client, _ = bff
    r = await client.post("/api/streams", json={"title": "mine"}, headers=_auth("u1"))
    stream_id = r.json()["id"]
    r = await client.post(f"/api/streams/{stream_id}/commissions",
                          json={"brief": "x"}, headers=_auth("u1"))
    sid = r.json()["session_id"]

    # Another user can see neither the stream nor the commission.
    other = _auth("u2")
    assert (await client.get(f"/api/streams/{stream_id}", headers=other)).status_code == 404
    assert (await client.get(f"/api/commissions/{sid}", headers=other)).status_code == 404
    assert (await client.post(f"/api/commissions/{sid}/approve", json={},
                              headers=other)).status_code == 404


async def test_cancel_at_plan_gate(bff):
    client, _ = bff
    h = _auth()
    r = await client.post("/api/streams", json={"title": "s"}, headers=h)
    stream_id = r.json()["id"]
    r = await client.post(f"/api/streams/{stream_id}/commissions",
                          json={"brief": "x"}, headers=h)
    sid = r.json()["session_id"]
    r = await client.post(f"/api/commissions/{sid}/cancel", headers=h)
    assert r.status_code == 200 and r.json()["ok"] is True
