"""Lupa BFF end-to-end against a fake Flight Deck.

The fake FD implements exactly the endpoints the BFF consumes (the "engine
contract" from docs/research-desk-product-plan.md). These tests prove the
whole commission flow — login proxy, stream, start → plan gate → approve →
progress → facts → report file — without a running Captain.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import jwt as pyjwt
import pytest
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile

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
        log["start_body"] = body
        sid = "sess-aaaa1111"
        log["sessions"][sid] = {
            "id": sid, "status": "planning", "intent": body["intent"],
            "title": body.get("title", ""), "truth": "",
            "route": {"group0_plan": {"steps": [{"agent": "deep-researcher",
                                                 "does": "scope the market"}]}},
            "analysis": {
                "quality_verdict": "pass",
                "blocking": {"rounds": 1, "verdict": "pass"},
                "quality_metrics": {"claims_checked": 10, "claims_confirmed": 8,
                                    "claims_refuted": 1, "claims_unverifiable": 1,
                                    "quality_verdict": "pass"},
                "consistency": {"values_checked": 12, "critical": 0, "major": 1},
                "gaps": [{"severity": "major", "text": "Nordics not covered"}],
            },
            "config": {"mode": "vatra"}}
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
        log["approve_body"] = body
        return {"ok": True, "session_id": body["session_id"], "status": "running"}

    @fd.post("/fd/vatra/plan/cancel")
    async def plan_cancel(body: dict, authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        sess = log["sessions"].get(body["session_id"])
        if not sess or sess["status"] not in ("planning", "awaiting_plan"):
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

    KALUP_TRUTH = ('Configured the desk.\n```json\n'
                   '{"name": "Tender Desk", "tagline": "RFPs, decoded",'
                   ' "theme": {"accent": "#2fb2a0"},'
                   ' "vocabulary": {"stream": "Tender"},'
                   ' "quality": {"profile": "thorough"},'
                   ' "evals": [{"brief": "Analyze a sample RFP"}]}\n```')

    _FAIL_MARKER = "MAKE_IT_FAIL"
    _FAIL_MSG = "Vatra Lead failed: missing Anthropic API key"

    @fd.post("/fd/llm/complete")
    async def llm_complete(body: dict, authorization: str | None = Header(default=None)):
        # Manifest generation is a single completion on the owner's tier.
        _need_auth(authorization)
        log["llm_complete"] = body
        if _FAIL_MARKER in (body.get("prompt", "") + body.get("system", "")):
            raise HTTPException(502, "LLM call failed: missing Anthropic API key")
        return {"content": KALUP_TRUTH, "provider": "openai",
                "model": "deepseek-v4-pro"}

    @fd.get("/fd/basna/sessions/{sid}")
    async def session_detail(sid: str, authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        sess = log["sessions"].get(sid)
        if not sess:
            raise HTTPException(404, "session not found")
        # A run whose intent carries the marker fails at the Lead (like a
        # missing API key) — status error before it ever reaches the gate.
        if _FAIL_MARKER in sess.get("intent", ""):
            sess["status"] = "error"
            return sess
        # Poll-driven progression so headless (factory) runs can complete:
        # planning → awaiting_plan on first read; running → done after 2 reads.
        if sess["status"] == "planning":
            sess["status"] = "awaiting_plan"
        elif sess["status"] == "running":
            sess["polls"] = sess.get("polls", 0) + 1
            if sess["polls"] >= 2:
                sess["status"] = "done"
                if not sess.get("truth"):
                    sess["truth"] = (KALUP_TRUTH
                                     if sess["intent"].startswith("KALUP PACK DRAFT")
                                     else "# Golden run\nAll checks passed.")
        return sess

    @fd.get("/fd/basna/sessions/{sid}/progress")
    async def progress(sid: str, authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        sess = log["sessions"].get(sid)
        if sess is None:
            raise HTTPException(404, "session not found")
        if _FAIL_MARKER in sess.get("intent", ""):
            return {"events": [
                {"i": 0, "stage": "phase", "message": "Group 0 · Long Horizon Planner"},
                {"i": 1, "stage": "route", "message": _FAIL_MSG, "ok": False}],
                "active": False}
        return {"events": [{"i": 0, "stage": "phase", "message": "Planning"},
                           {"i": 1, "stage": "dispatch", "message": "deep-researcher working"},
                           {"i": 2, "stage": "cost", "message": "$0.12",
                            "usd": 0.12, "hourly_usd": 3.4}],
                "active": sess["status"] == "running"}

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
        import json as _json
        _need_auth(authorization)
        if path == ".contract.json":
            return {"text": _json.dumps({"constraints": [
                {"id": "c1", "text": "All values in EUR", "severity": "critical",
                 "status": "pass"}]}), "binary": False}
        return {"text": f"# Report\nfrom {project}/{path}", "binary": False}

    @fd.get("/fd/costs")
    async def costs(authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        return {"costs": [{"run_kind": "vatra", "run_id": "sess-aaaa1111",
                           "usd": 0.42, "elapsed_seconds": 300.0,
                           "usage": {"prompt_tokens": 1000},
                           "at": "2026-07-30T10:00:00+00:00"}],
                "count": 1, "priced": 1, "total_usd": 0.42}

    log["archetypes"] = {}

    @fd.post("/fd/archetypes/forge")
    async def forge(instructions: str = Form(""), count: str = Form("0"),
                    files: list[UploadFile] = File(default=[]),
                    authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        log["forge"] = {"instructions": instructions,
                        "files": [f.filename for f in files]}
        return {"archetypes": [
            {"id": "house-analyst", "role": "House Analyst",
             "instructions": "Analyze in the house style."},
            {"id": "house-writer", "role": "House Writer",
             "instructions": "Write in the house voice."}]}

    @fd.put("/fd/archetypes/{aid}")
    async def put_archetype(aid: str, body: dict,
                            authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        log["archetypes"][aid] = {**body, "id": aid}
        return log["archetypes"][aid]

    @fd.get("/fd/archetypes/mine")
    async def my_archetypes(authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        return [{**a, "source": "user"} for a in log["archetypes"].values()]

    @fd.get("/fd/archetypes")
    async def merged_archetypes(authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        base = [{"id": "deep-researcher", "role": "Deep Researcher", "source": "base"},
                {"id": "analyst", "role": "Analyst", "source": "base"}]
        mine = [{**a, "source": "user"} for a in log["archetypes"].values()]
        return {"archetypes": base + mine}

    @fd.post("/fd/basna/route")
    async def basna_route(body: dict, authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        log["basna_route"] = body
        sid = "basna-cccc3333"
        log["sessions"][sid] = {"id": sid, "status": "routed",
                                "intent": body["intent"], "truth": "",
                                "confidence": 0.0, "route": {}, "analysis": {},
                                "config": {"mode": "basna"}}
        return {"session_id": sid, "selected": ["deep-researcher", "analyst"]}

    @fd.post("/fd/basna/execute")
    async def basna_execute(body: dict, authorization: str | None = Header(default=None)):
        _need_auth(authorization)
        s = log["sessions"].get(body["session_id"])
        if not s:
            raise HTTPException(404, "session not found")
        s["status"] = "done"
        s["truth"] = ("# Second read\nLargely agrees with the desk; flags the "
                      "2030 CAGR as optimistic.")
        s["confidence"] = 0.78
        return {"session_id": s["id"], "truth": s["truth"], "confidence": 0.78}

    return fd, log


@pytest.fixture
async def bff(tmp_path, monkeypatch):
    """(client, fd_log, app) — a lifespan-running BFF wired to the fake FD."""
    monkeypatch.setenv("LUPA_DATA_DIR", str(tmp_path / "lupa-data"))
    monkeypatch.setenv("FD_JWT_SECRET", SECRET)
    # Park the scheduler loop so tests drive fire_due_briefs deterministically.
    monkeypatch.setenv("LUPA_BRIEF_TICK_SECONDS", "3600")
    # Fast factory polling for the Studio generate/evaluate tests.
    monkeypatch.setenv("LUPA_FACTORY_POLL_SECONDS", "0.01")
    fake_fd, log = make_fake_fd()
    app = create_app(fd_transport=httpx.ASGITransport(app=fake_fd))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://lupa.test") as client:
            yield client, log, app


def _auth(sub: str = "u1") -> dict:
    return {"Authorization": f"Bearer {make_token(sub)}"}


async def test_pack_is_served_and_vertical_free(bff):
    client, _, _ = bff
    r = await client.get("/api/pack")
    assert r.status_code == 200
    pack = r.json()
    assert pack["slug"] == "research-desk"
    assert pack["vocabulary"]["stream"]
    assert pack["quality"] == {"profile": "thorough"}
    assert "onboarding_md" in pack


async def test_login_proxy_rewrites_cookie_path(bff):
    client, _, _ = bff
    r = await client.post("/api/auth/login", json={"email": "u1@test", "password": "x"})
    assert r.status_code == 200
    assert r.json()["access_token"] == "fd-access"
    sc = r.headers.get("set-cookie", "")
    assert "Path=/api/auth" in sc
    assert "Path=/fd/auth" not in sc


async def test_product_routes_require_token(bff):
    client, _, _ = bff
    assert (await client.get("/api/streams")).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert (await client.get("/api/streams", headers=bad)).status_code == 401


async def test_execution_groups_default_and_override(bff):
    """The research-desk default runs grouped; a stream can override to flat."""
    client, log, _ = bff
    h = _auth()
    # Pack default (research-desk → run.execution_groups: true) reaches approve.
    stream_id = (await client.post("/api/streams", json={"title": "grouped"},
                                   headers=h)).json()["id"]
    sid = (await client.post(f"/api/streams/{stream_id}/commissions",
                             json={"brief": "x"}, headers=h)).json()["session_id"]
    await client.post(f"/api/commissions/{sid}/approve", json={}, headers=h)
    assert log["approve_body"]["execution_groups"] is True

    # An explicit stream override to flat wins over the pack default.
    s2 = (await client.post("/api/streams", json={"title": "flat"}, headers=h)).json()["id"]
    await client.patch(f"/api/streams/{s2}/settings",
                       json={"execution_groups": False}, headers=h)
    sid3 = (await client.post(f"/api/streams/{s2}/commissions",
                              json={"brief": "z"}, headers=h)).json()["session_id"]
    await client.post(f"/api/commissions/{sid3}/approve", json={}, headers=h)
    assert log["approve_body"]["execution_groups"] is False


async def test_full_commission_flow(bff):
    client, log, _ = bff
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
    assert r.json()["total_usd"] == 0.42


async def test_commissions_are_owner_scoped(bff):
    client, _, _ = bff
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


async def test_stream_settings_drive_commissions(bff):
    """Stream overrides (quality, cast continuity, max agents) reach FD."""
    client, log, _ = bff
    h = _auth()
    stream_id = (await client.post("/api/streams", json={"title": "s"},
                                   headers=h)).json()["id"]

    r = await client.patch(f"/api/streams/{stream_id}/settings",
                           json={"quality_profile": "balanced",
                                 "same_cast": False, "max_agents": 4}, headers=h)
    assert r.status_code == 200
    assert r.json()["settings"] == {"quality_profile": "balanced",
                                    "same_cast": False, "max_agents": 4}
    bad = await client.patch(f"/api/streams/{stream_id}/settings",
                             json={"quality_profile": "extreme"}, headers=h)
    assert bad.status_code == 400

    # Round 1 runs with the stream's quality + max_agents.
    r = await client.post(f"/api/streams/{stream_id}/commissions",
                          json={"brief": "x"}, headers=h)
    sid = r.json()["session_id"]
    assert log["start_body"]["quality"] == {"profile": "balanced"}
    assert log["start_body"]["max_agents"] == 4

    # Round 2 as an explicit revise, carrying same_cast=False.
    r = await client.post(f"/api/streams/{stream_id}/commissions",
                          json={"brief": "tighten it", "kind": "revise"}, headers=h)
    assert r.status_code == 200 and r.json()["kind"] == "revise"
    assert log["continued"]["kind"] == "revise"
    assert log["continued"]["same_cast"] is False
    assert log["continued"]["parent"] == sid

    # Unknown kinds are rejected; explicit null resets an override.
    assert (await client.post(f"/api/streams/{stream_id}/commissions",
                              json={"brief": "x", "kind": "weird"},
                              headers=h)).status_code == 400
    r = await client.patch(f"/api/streams/{stream_id}/settings",
                           json={"quality_profile": None}, headers=h)
    assert "quality_profile" not in r.json()["settings"]


async def test_receipts_aggregation(bff):
    """One call assembles verdict + metrics + facts + contract + cost + ROI."""
    client, _, _ = bff
    h = _auth()
    r = await client.post("/api/streams", json={"title": "s"}, headers=h)
    stream_id = r.json()["id"]
    r = await client.post(f"/api/streams/{stream_id}/commissions",
                          json={"brief": "x"}, headers=h)
    sid = r.json()["session_id"]

    r = await client.get(f"/api/commissions/{sid}/receipts", headers=h)
    assert r.status_code == 200
    rec = r.json()
    assert rec["verdict"] == "pass"
    assert rec["metrics"]["claims_confirmed"] == 8
    assert rec["consistency"]["values_checked"] == 12
    assert rec["gaps"][0]["severity"] == "major"
    assert rec["facts"][0]["status"] == "verified"
    assert rec["contract"]["constraints"][0]["id"] == "c1"
    # hourly recomputed from usd/elapsed: 0.42 / (300/3600) = 5.04
    assert rec["cost"]["usd"] == 0.42
    assert rec["cost"]["hourly_usd"] == pytest.approx(5.04)
    assert rec["roi"]["analyst_hourly_usd"] == 60
    # Owner-scoped like everything else.
    assert (await client.get(f"/api/commissions/{sid}/receipts",
                             headers=_auth("u2"))).status_code == 404


async def test_standing_brief_lifecycle(bff, monkeypatch):
    """PUT/GET/DELETE + the scheduler pass: running-guard, framed delta
    instruction, minted owner token, round recorded as 'brief', inbox."""
    from lupa_api.server import fire_due_briefs
    client, log, app = bff
    h = _auth()
    stream_id = (await client.post("/api/streams", json={"title": "watch"},
                                   headers=h)).json()["id"]

    # No rounds yet → a brief has nothing to continue from.
    r = await client.put(f"/api/streams/{stream_id}/brief",
                         json={"instruction": "watch EU subsidies",
                               "cadence_hours": 1}, headers=h)
    assert r.status_code == 409

    # Round 1 + approve (fake leaves it "running").
    sid = (await client.post(f"/api/streams/{stream_id}/commissions",
                             json={"brief": "baseline"}, headers=h)).json()["session_id"]
    await client.post(f"/api/commissions/{sid}/approve", json={}, headers=h)

    r = await client.put(f"/api/streams/{stream_id}/brief",
                         json={"instruction": "watch EU subsidies",
                               "cadence_hours": 1}, headers=h)
    assert r.status_code == 200
    assert r.json()["brief"]["enabled"] == 1

    # Due immediately, but the tip round is still running → the guard holds.
    assert await fire_due_briefs(app.state) == 0

    # Tip finishes → the brief fires: framed instruction, brief-kind round.
    log["sessions"][sid]["status"] = "done"
    assert await fire_due_briefs(app.state) == 1
    assert log["continued"]["parent"] == sid
    assert log["continued"]["instruction"].startswith("Standing brief")
    assert "watch EU subsidies" in log["continued"]["instruction"]

    rounds = (await client.get(f"/api/streams/{stream_id}", headers=h)).json()["rounds"]
    assert rounds[-1]["kind"] == "brief"

    brief = (await client.get(f"/api/streams/{stream_id}/brief",
                              headers=h)).json()["brief"]
    assert brief["last_session_id"] == rounds[-1]["session_id"]
    assert brief["next_run_at"] > brief["last_run_at"]  # pushed a cadence ahead

    # Not due anymore.
    assert await fire_due_briefs(app.state) == 0

    # Inbox lists the scheduled round with its stream title.
    inbox = (await client.get("/api/inbox", headers=h)).json()["rounds"]
    assert inbox and inbox[0]["stream_title"] == "watch"
    assert inbox[0]["kind"] == "brief"

    # Without the shared secret the scheduler stands down entirely.
    monkeypatch.delenv("FD_JWT_SECRET")
    assert await fire_due_briefs(app.state) == 0
    monkeypatch.setenv("FD_JWT_SECRET", SECRET)

    # Delete → gone.
    assert (await client.delete(f"/api/streams/{stream_id}/brief",
                                headers=h)).status_code == 200
    assert (await client.get(f"/api/streams/{stream_id}/brief",
                             headers=h)).json()["brief"] is None


async def test_house_style_forge_save_and_pinned_cast(bff):
    """Forge proxy → drafts; save via upsert; pinned cast reaches vatra/start."""
    client, log, _ = bff
    h = _auth()

    r = await client.post(
        "/api/forge", headers=h,
        data={"instructions": "our diligence playbook, terse memos"},
        files=[("files", ("playbook.md", b"# Playbook\nBe terse.", "text/markdown"))])
    assert r.status_code == 200
    drafts = r.json()["drafts"]
    assert [d["id"] for d in drafts] == ["house-analyst", "house-writer"]
    assert log["forge"]["instructions"].startswith("our diligence")
    assert log["forge"]["files"] == ["playbook.md"]

    r = await client.post("/api/archetypes", json=drafts[0], headers=h)
    assert r.status_code == 200
    # The default cast pool is the MERGED registry: system (source=base) +
    # the saved house archetype (source=user), each tagged.
    pool = (await client.get("/api/archetypes", headers=h)).json()["archetypes"]
    by_source = {a["source"] for a in pool}
    assert by_source == {"base", "user"}
    assert any(a["id"] == "house-analyst" and a["source"] == "user" for a in pool)
    assert any(a["source"] == "base" for a in pool)
    # ?mine=true narrows to just the user's own (the House-style save target).
    mine = (await client.get("/api/archetypes?mine=true", headers=h)).json()["archetypes"]
    assert [a["id"] for a in mine] == ["house-analyst"]

    stream_id = (await client.post("/api/streams", json={"title": "cast"},
                                   headers=h)).json()["id"]
    await client.patch(f"/api/streams/{stream_id}/settings",
                       json={"archetype_ids": ["house-analyst"]}, headers=h)
    await client.post(f"/api/streams/{stream_id}/commissions",
                      json={"brief": "x"}, headers=h)
    assert log["start_body"]["archetype_ids"] == ["house-analyst"]

    # Clearing the cast removes the pin (and start omits the field).
    r = await client.patch(f"/api/streams/{stream_id}/settings",
                           json={"archetype_ids": []}, headers=h)
    assert "archetype_ids" not in r.json()["settings"]


async def test_second_opinion_lifecycle(bff):
    """Route → background execute → poll shows the ensemble's truth; idempotent."""
    client, log, _ = bff
    h = _auth()
    stream_id = (await client.post("/api/streams", json={"title": "s"},
                                   headers=h)).json()["id"]
    sid = (await client.post(f"/api/streams/{stream_id}/commissions",
                             json={"brief": "size the market"},
                             headers=h)).json()["session_id"]

    r = await client.post(f"/api/commissions/{sid}/second-opinion", headers=h)
    assert r.status_code == 200
    rec = r.json()
    assert rec["basna_session_id"] == "basna-cccc3333"
    assert log["basna_route"]["intent"] == "size the market"

    # Same call again → the same record, no second ensemble.
    again = await client.post(f"/api/commissions/{sid}/second-opinion", headers=h)
    assert again.json()["basna_session_id"] == rec["basna_session_id"]

    # The background execute finishes; poll until the ensemble's truth lands.
    got: dict = {}
    for _ in range(50):
        got = (await client.get(f"/api/commissions/{sid}/second-opinion",
                                headers=h)).json()
        if got["second_opinion"]["status"] == "done":
            break
        await asyncio.sleep(0.02)
    assert got["second_opinion"]["status"] == "done"
    assert "Second read" in got["truth"]
    assert got["confidence"] == 0.78

    # Owner-scoped.
    assert (await client.post(f"/api/commissions/{sid}/second-opinion",
                              headers=_auth("u2"))).status_code == 404


async def test_cancel_at_plan_gate(bff):
    client, _, _ = bff
    h = _auth()
    r = await client.post("/api/streams", json={"title": "s"}, headers=h)
    stream_id = r.json()["id"]
    r = await client.post(f"/api/streams/{stream_id}/commissions",
                          json={"brief": "x"}, headers=h)
    sid = r.json()["session_id"]
    r = await client.post(f"/api/commissions/{sid}/cancel", headers=h)
    assert r.status_code == 200 and r.json()["ok"] is True


# ── Kalup Part II: pack registry + Studio ────────────────────────────


async def _wait_for(client, url, headers, path: list[str], value: str,
                    tries: int = 200) -> dict:
    """Poll a GET until a nested field reaches `value` (background factory)."""
    body: dict = {}
    for _ in range(tries):
        body = (await client.get(url, headers=headers)).json()
        cur = body
        for key in path:
            cur = (cur or {}).get(key) if isinstance(cur, dict) else None
        if cur == value:
            return body
        await asyncio.sleep(0.02)
    raise AssertionError(f"{url} never reached {'.'.join(path)}={value}: {body}")


async def test_registry_seeds_and_permissions(bff):
    client, _, _ = bff
    user, admin = _auth("u1"), {"Authorization": f"Bearer {make_token('boss', 'admin')}"}

    # The repo pack was imported as a published system pack.
    packs = (await client.get("/api/packs", headers=user)).json()["packs"]
    assert any(p["slug"] == "research-desk" and p["status"] == "published"
               for p in packs)

    # Plain users can't create drafts; admins can; granted creators can.
    assert (await client.post("/api/packs", json={"slug": "x-desk", "name": "X"},
                              headers=user)).status_code == 403
    assert (await client.post("/api/creators", json={"user_id": "u1"},
                              headers=user)).status_code == 403
    assert (await client.post("/api/creators", json={"user_id": "u1"},
                              headers=admin)).status_code == 200
    assert (await client.get("/api/creators/me", headers=user)).json()["creator"] is True
    r = await client.post("/api/packs", json={"slug": "x-desk", "name": "X Desk"},
                          headers=user)
    assert r.status_code == 200
    # The scaffold copies the default pack so a fresh draft renders.
    assert r.json()["pack"]["vocabulary"]["stream"] == "Stream"

    # Drafts are invisible to others (u2 sees only published).
    others = (await client.get("/api/packs", headers=_auth("u2"))).json()["packs"]
    assert all(p["slug"] != "x-desk" for p in others)


async def test_draft_detail_401_not_404_when_unauthed(bff):
    """A draft's detail returns 401 (not 404) without a token, so the SPA
    silently refreshes and retries — a long Studio poll must not wedge when the
    access token's TTL lapses. Authed-but-unauthorized still gets 404 (no leak);
    a published pack is public."""
    client, _, _ = bff
    admin = {"Authorization": f"Bearer {make_token('boss', 'admin')}"}
    await client.post("/api/packs", json={"slug": "d1", "name": "D"}, headers=admin)

    assert (await client.get("/api/packs/d1")).status_code == 401          # refresh path
    assert (await client.get("/api/packs/d1", headers=_auth("u2"))).status_code == 404
    assert (await client.get("/api/packs/d1", headers=admin)).status_code == 200
    # Published packs render pre-login (public branding).
    assert (await client.get("/api/packs/research-desk")).status_code == 200


async def test_factory_generate_evaluate_publish(bff):
    """The 2-3-day factory line, compressed: draft → generate (Vatra writes the
    manifest) → ship-gate blocks → evaluate (golden run) → green → publish."""
    client, log, _ = bff
    admin = {"Authorization": f"Bearer {make_token('boss', 'admin')}"}

    await client.post("/api/packs", json={"slug": "tender-desk", "name": "Tenders"},
                      headers=admin)

    # The gate is closed before any evaluation.
    r = await client.post("/api/packs/tender-desk/publish", headers=admin)
    assert r.status_code == 409

    # Generate: a SINGLE completion (not a Vatra run) drafts the manifest.
    r = await client.post("/api/packs/tender-desk/generate",
                          json={"instructions": "public-sector RFP analysis"},
                          headers=admin)
    assert r.status_code == 200
    body = await _wait_for(client, "/api/packs/tender-desk", admin,
                           ["generation", "status"], "done")
    # Generation went through /fd/llm/complete on the reason tier, not vatra.
    assert log["llm_complete"]["tier"] == "reason"
    assert "public-sector RFP analysis" in log["llm_complete"]["prompt"]
    pack = body["pack"]
    assert pack["name"] == "Tender Desk"           # generated
    assert pack["vocabulary"]["stream"] == "Tender"
    assert pack["evals"][0]["brief"] == "Analyze a sample RFP"

    # Evaluate: the golden commission runs and passes its own receipts.
    r = await client.post("/api/packs/tender-desk/evaluate", headers=admin)
    assert r.status_code == 200
    body = await _wait_for(client, "/api/packs/tender-desk", admin,
                           ["eval", "verdict"], "green")
    assert log["start_body"]["intent"] == "Analyze a sample RFP"

    # Publish: the gate opens on green; the desk appears for everyone.
    r = await client.post("/api/packs/tender-desk/publish", headers=admin)
    assert r.status_code == 200
    packs = (await client.get("/api/packs", headers=_auth("u2"))).json()["packs"]
    tender = next(p for p in packs if p["slug"] == "tender-desk")
    assert tender["status"] == "published" and tender["version"] == 1

    # Streams scope to the new desk and its quality profile drives runs.
    h = _auth("u2")
    r = await client.post("/api/streams",
                          json={"title": "City RFP", "pack": "tender-desk"},
                          headers=h)
    stream_id = r.json()["id"]
    await client.post(f"/api/streams/{stream_id}/commissions",
                      json={"brief": "review the city RFP"}, headers=h)
    assert log["start_body"]["quality"] == {"profile": "thorough"}
    scoped = (await client.get("/api/streams", params={"pack": "tender-desk"},
                               headers=h)).json()["streams"]
    assert [s["id"] for s in scoped] == [stream_id]
    assert (await client.get("/api/streams", params={"pack": "research-desk"},
                             headers=h)).json()["streams"] == []


async def test_factory_streams_eval_progress(bff):
    """The Studio streams the EVALUATE golden run's live feed (the real
    multi-agent run); its session id is stored so the feed is reachable."""
    client, _, _ = bff
    admin = {"Authorization": f"Bearer {make_token('boss', 'admin')}"}
    await client.post("/api/packs", json={"slug": "feed-desk", "name": "Feed"},
                      headers=admin)
    await client.post("/api/packs/feed-desk/generate",
                      json={"instructions": "a monitoring desk"}, headers=admin)
    await _wait_for(client, "/api/packs/feed-desk", admin,
                    ["generation", "status"], "done")
    await client.post("/api/packs/feed-desk/evaluate", headers=admin)
    await _wait_for(client, "/api/packs/feed-desk", admin, ["eval", "verdict"], "green")
    # The eval run's session id was recorded; progress proxies its feed.
    r = await client.get("/api/packs/feed-desk/progress?phase=eval", headers=admin)
    assert r.status_code == 200
    events = r.json()["events"]
    assert any(e["stage"] == "phase" for e in events)
    assert any("deep-researcher" in e["message"] for e in events)
    # Owner-scoped like the rest of the registry.
    assert (await client.get("/api/packs/feed-desk/progress",
                             headers=_auth("u2"))).status_code == 404


async def test_generation_surfaces_specific_error(bff):
    """A failed generation reports the SPECIFIC LLM error (the completion
    endpoint's detail), not a generic 'error'."""
    client, _, _ = bff
    admin = {"Authorization": f"Bearer {make_token('boss', 'admin')}"}
    await client.post("/api/packs", json={"slug": "fail-desk", "name": "Fail"},
                      headers=admin)
    await client.post("/api/packs/fail-desk/generate",
                      json={"instructions": "a desk MAKE_IT_FAIL please"}, headers=admin)
    body = await _wait_for(client, "/api/packs/fail-desk", admin,
                           ["generation", "status"], "error")
    assert "missing Anthropic API key" in body["generation"]["message"]


async def test_cancel_unsticks_a_stuck_evaluation(bff):
    """A stuck 'running' eval blocks re-run (409); cancel resets it so the
    creator can run evaluation again."""
    client, _, app = bff
    admin = {"Authorization": f"Bearer {make_token('boss', 'admin')}"}
    await client.post("/api/packs", json={"slug": "stuck-desk", "name": "Stuck"},
                      headers=admin)
    # Simulate a run wedged 'running' by a since-dead task.
    await app.state.db.update_pack(
        "stuck-desk", eval_state={"status": "running", "run_id": "old"})
    assert (await client.post("/api/packs/stuck-desk/evaluate",
                              headers=admin)).status_code == 409

    # Cancel clears it.
    r = await client.post("/api/packs/stuck-desk/cancel?phase=eval", headers=admin)
    assert r.status_code == 200
    detail = (await client.get("/api/packs/stuck-desk", headers=admin)).json()
    assert detail["eval"]["status"] == "cancelled"
    # Re-run is allowed again.
    assert (await client.post("/api/packs/stuck-desk/evaluate",
                              headers=admin)).status_code == 200
    # Owner-scoped.
    assert (await client.post("/api/packs/stuck-desk/cancel",
                              headers=_auth("u2"))).status_code == 404


async def test_startup_resets_stale_running_packs(bff):
    """A pack left 'running' at boot is stale — reset_running_packs flips it to
    a terminal state so it isn't wedged forever."""
    client, _, app = bff
    admin = {"Authorization": f"Bearer {make_token('boss', 'admin')}"}
    await client.post("/api/packs", json={"slug": "boot-desk", "name": "Boot"},
                      headers=admin)
    await app.state.db.update_pack(
        "boot-desk", generation={"status": "running", "run_id": "x"},
        eval_state={"status": "running", "run_id": "y"})

    n = await app.state.db.reset_running_packs()
    assert n == 1
    detail = (await client.get("/api/packs/boot-desk", headers=admin)).json()
    assert detail["generation"]["status"] == "error"
    assert detail["eval"]["status"] == "error"
    assert "interrupted" in detail["eval"]["message"]


async def test_seed_refreshes_unedited_system_pack_only(bff):
    """Startup re-seed refreshes an UNEDITED system pack from the repo, but
    preserves a runtime-edited one and never touches a user pack."""
    client, _, app = bff
    db = app.state.db

    # A fresh system pack: seeded, then the repo changes → refreshed.
    await db.upsert_seed_pack("sys-a", {"name": "A", "v": 1})
    await db.upsert_seed_pack("sys-a", {"name": "A", "v": 2})
    row = await db.get_pack("sys-a")
    assert json.loads(row["manifest"])["v"] == 2  # repo change landed

    # A system pack an admin edited in the Studio → NOT clobbered by re-seed.
    await db.upsert_seed_pack("sys-b", {"name": "B", "v": 1})
    await db.update_pack("sys-b", manifest={"name": "B edited", "v": 1})
    await db.upsert_seed_pack("sys-b", {"name": "B", "v": 9})
    row = await db.get_pack("sys-b")
    assert json.loads(row["manifest"])["name"] == "B edited"  # runtime wins

    # A user/creator pack is never touched by seeding, even at the same slug.
    await db.create_pack("mine", "u1", {"name": "Mine", "v": 1})
    await db.upsert_seed_pack("mine", {"name": "Seed", "v": 9})
    row = await db.get_pack("mine")
    assert row["owner_id"] == "u1"
    assert json.loads(row["manifest"])["name"] == "Mine"


def test_job_token_outlasts_the_factory_timeout(monkeypatch):
    """A background job's minted token must not lapse before the run's own
    timeout — otherwise the FD poll 401s and the run dies at the finish line."""
    from lupa_api.server import _job_token_ttl, _factory_timeout, _mint_owner_token
    monkeypatch.setenv("LUPA_FACTORY_TIMEOUT_SECONDS", "2700")
    assert _job_token_ttl() > _factory_timeout()
    # The ttl is honored in the token's exp.
    tok = _mint_owner_token("u1", SECRET, ttl=1234)
    payload = pyjwt.decode(tok, SECRET, algorithms=["HS256"])
    assert payload["exp"] - payload["iat"] == 1234


def test_eval_verdict_is_strict():
    from lupa_api.server import _eval_verdict
    green, _ = _eval_verdict({"quality_verdict": "pass",
                              "quality_metrics": {"contract_failed_critical": 0}})
    assert green == "green"
    red, _ = _eval_verdict({"quality_verdict": "pass",
                            "quality_metrics": {"contract_failed_critical": 1}})
    assert red == "red"
    assert _eval_verdict({})[0] == "red"
