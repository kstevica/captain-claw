"""First Mate routes: /fd/first-mate*, /fd/admin/first-mate/settings.

Pinned here:

* a session is needed; the deck switch is off by default; while it's set to
  admins only, members get a 403 that says why; ``GET /fd/first-mate`` always
  answers, so the page can say it;
* the deck's settings are admin-only and validated;
* someone else's factory or run is a 404; an admin's ``X-FD-Act-As`` is ignored;
* a factory from a template, a draft, approval against the hash shown, decline;
* owners set running · paused · stopped · retired, never paused_budget;
* trust and pins; no ``direct`` with outward grants;
* Run now starts a Bat run, the list follows it, stop cancels it; no free Bat
  slot is a 400 that says why;
* a factory may serve only the caller's own or a deck-wide Course goal;
* delete only a draft or a retired factory.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import bat_loop, rate_limiter, server
from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import ACT_AS_HEADER, create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

ADMIN, BOB, CAROL = "user-admin", "user-bob", "user-carol"
REMOTE = ("203.0.113.9", 40001)


class FakeBat:
    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}

    async def create_run(self, *, run_id, owner_id, title, task, config=None, origin=None,
                         source_host="localhost", source_port=0, llm_usd_cap=0.0, status="planning"):
        self.runs[run_id] = {"id": run_id, "owner_id": owner_id, "title": title, "task": task,
                             "config": config or {}, "status": status, "cumulative_tokens": 0,
                             "cumulative_usd": 0.0, "truth": "", "stopped_reason": ""}
        return run_id

    async def append_event(self, run_id, stage, message="", **data):
        return 1

    async def get_run(self, run_id):
        r = self.runs.get(run_id)
        return dict(r) if r else None

    async def list_runs(self, owner_id, limit=50):
        return [dict(r) for r in self.runs.values() if r["owner_id"] == owner_id][:limit]

    async def set_status(self, run_id, status, **kw):
        self.runs[run_id].update(status=status, **kw)


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Auth-enabled deck with an admin, Bob and Carol; First Mate's and
    Course's stores in tmp; a stand-in Bat."""
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid, role in ((ADMIN, "admin"), (BOB, "user"), (CAROL, "user")):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid, role))
    await db._db.commit()
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(fm, "_STORE", fm.FirstMateStore(tmp_path / "first_mate.db"))
    monkeypatch.setattr(c, "_STORE", c.CourseStore(tmp_path / "course.db"))
    from captain_claw.flight_deck import events
    monkeypatch.setattr(events, "_STORE", events.EventsStore(tmp_path / "events.db"))  # never the real one
    bat = FakeBat()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", bat)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    try:
        yield bat
    finally:
        await db.close()
        fd_auth._db = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=REMOTE), base_url="http://fd.test")


def _hdr(caller: str, act_as: str | None = None) -> dict:
    h = {"Authorization": f"Bearer {create_access_token(caller)}"}
    if act_as is not None:
        h[ACT_AS_HEADER] = act_as
    return h


def _open(admins_only: bool = False) -> None:
    fm.get_store().set_settings({"enabled": True, "admins_only": admins_only})


async def _factory(cl, who=BOB, **body):
    body.setdefault("template", "weekly-report")
    r = await cl.post("/fd/first-mate/factories", headers=_hdr(who), json=body)
    assert r.status_code == 200, r.text
    return r.json()


async def _approved(cl, who=BOB, **body):
    f = await _factory(cl, who, **body)
    r = await cl.post(f"/fd/first-mate/factories/{f['id']}/approve", headers=_hdr(who),
                      json={"version": 1, "sha": f["draft"]["sha"]})
    assert r.status_code == 200, r.text
    return r.json()


# ── who may use it ─────────────────────────────────────────────────────

async def test_needs_a_session_and_starts_off(deck):
    async with _client() as cl:
        assert (await cl.get("/fd/first-mate")).status_code == 401
        r = await cl.get("/fd/first-mate", headers=_hdr(BOB))
        assert r.status_code == 200
        assert r.json()["available"] is False and "off on this deck" in r.json()["reason"]
        r = await cl.get("/fd/first-mate/factories", headers=_hdr(ADMIN))
        assert r.status_code == 403 and "off on this deck" in r.json()["detail"]


async def test_admins_only_then_everyone(deck):
    async with _client() as cl:
        assert (await cl.put("/fd/admin/first-mate/settings", headers=_hdr(BOB),
                             json={"enabled": True})).status_code == 403
        r = await cl.put("/fd/admin/first-mate/settings", headers=_hdr(ADMIN), json={"enabled": True})
        assert r.status_code == 200 and r.json()["admins_only"] is True
        assert (await cl.get("/fd/first-mate", headers=_hdr(ADMIN))).json()["available"] is True
        r = await cl.get("/fd/first-mate/templates", headers=_hdr(BOB))
        assert r.status_code == 403 and "admins only" in r.json()["detail"]
        await cl.put("/fd/admin/first-mate/settings", headers=_hdr(ADMIN), json={"admins_only": False})
        r = await cl.get("/fd/first-mate/templates", headers=_hdr(BOB))
        assert r.status_code == 200 and len(r.json()["templates"]) == 6
        r = await cl.put("/fd/admin/first-mate/settings", headers=_hdr(ADMIN), json={"max_factories": 0})
        assert r.status_code == 400


async def test_prefs(deck):
    _open()
    async with _client() as cl:
        r = await cl.put("/fd/first-mate/prefs", headers=_hdr(BOB), json={"weekly_tokens": 10_000_000})
        assert r.status_code == 200 and r.json()["weekly_tokens"] == 10_000_000
        assert (await cl.put("/fd/first-mate/prefs", headers=_hdr(BOB), json={"weekly_tokens": 5})).status_code == 400
        assert (await cl.get("/fd/first-mate", headers=_hdr(BOB))).json()["prefs"]["weekly_tokens"] == 10_000_000
        assert (await cl.get("/fd/first-mate", headers=_hdr(CAROL))).json()["prefs"]["weekly_tokens"] == 25_000_000


# ── factories and versions ─────────────────────────────────────────────

async def test_create_from_a_template_and_fields(deck):
    _open()
    async with _client() as cl:
        f = await _factory(cl, template="price-watch", title="GPU watch")
        assert f["state"] == "draft" and f["title"] == "GPU watch" and f["template"] == "price-watch"
        assert f["draft"]["triggers"][0]["schedule"] == "daily 09:00"
        r = await cl.post("/fd/first-mate/factories", headers=_hdr(BOB), json={"template": "nope"})
        assert r.status_code == 400 and "template" in r.json()["detail"]
        r = await cl.post("/fd/first-mate/factories", headers=_hdr(BOB), json={"title": "x", "colour": "red"})
        assert r.status_code == 400 and "unknown fields" in r.json()["detail"]
        r = await cl.post("/fd/first-mate/factories", headers=_hdr(BOB), json={"title": "No charter"})
        assert r.status_code == 400
        r = await cl.post("/fd/first-mate/factories", headers=_hdr(BOB),
                          json={"title": "t", "charter": "c", "runner": "vatra", "runner_config": {"api_key": "k"}})
        assert r.status_code == 400 and "secret" in r.json()["detail"]
        r = await cl.get("/fd/first-mate/factories", headers=_hdr(BOB))
        assert [x["title"] for x in r.json()["factories"]] == ["GPU watch"]


async def test_someone_elses_factory_is_a_404_and_act_as_is_ignored(deck):
    _open()
    async with _client() as cl:
        f = await _factory(cl)
        for method, path, body in (
            ("get", f"/fd/first-mate/factories/{f['id']}", None),
            ("post", f"/fd/first-mate/factories/{f['id']}/approve", {"version": 1, "sha": f["draft"]["sha"]}),
            ("post", f"/fd/first-mate/factories/{f['id']}/run", {}),
            ("delete", f"/fd/first-mate/factories/{f['id']}", None),
        ):
            kw = {"json": body} if body is not None else {}
            r = await getattr(cl, method)(path, headers=_hdr(CAROL), **kw)
            assert r.status_code == 404, (method, path, r.text)
        r = await cl.get("/fd/first-mate/factories", headers=_hdr(ADMIN, act_as=BOB))
        assert r.json()["factories"] == []
        r = await cl.get(f"/fd/first-mate/factories/{f['id']}", headers=_hdr(ADMIN, act_as=BOB))
        assert r.status_code == 404


async def test_approval_goes_by_the_hash_shown(deck):
    _open()
    async with _client() as cl:
        f = await _factory(cl)
        fid, sha = f["id"], f["draft"]["sha"]
        r = await cl.post(f"/fd/first-mate/factories/{fid}/submit", headers=_hdr(BOB))
        assert r.json()["state"] == "awaiting_charter"
        r = await cl.patch(f"/fd/first-mate/factories/{fid}/draft", headers=_hdr(BOB),
                           json={"charter": "Two pages this time."})
        assert r.status_code == 200 and r.json()["draft"]["sha"] != sha
        r = await cl.post(f"/fd/first-mate/factories/{fid}/approve", headers=_hdr(BOB),
                          json={"version": 1, "sha": sha})
        assert r.status_code == 400 and "changed since you saw it" in r.json()["detail"]
        fresh = (await cl.get(f"/fd/first-mate/factories/{fid}", headers=_hdr(BOB))).json()["draft"]["sha"]
        r = await cl.post(f"/fd/first-mate/factories/{fid}/approve", headers=_hdr(BOB),
                          json={"version": 1, "sha": fresh})
        assert r.status_code == 200 and r.json()["state"] == "running"
        detail = (await cl.get(f"/fd/first-mate/factories/{fid}", headers=_hdr(BOB))).json()
        assert detail["live"]["charter"] == "Two pages this time."
        assert [v["status"] for v in detail["versions"]] == ["live"]
        assert detail["events"][0]["event"] == "approved" and detail["satisfaction"] is None


async def test_new_draft_and_decline(deck):
    _open()
    async with _client() as cl:
        f = await _approved(cl)
        fid = f["id"]
        r = await cl.post(f"/fd/first-mate/factories/{fid}/draft", headers=_hdr(BOB),
                          json={"grants": {"email": True}, "note": "let it send"})
        assert r.status_code == 200 and r.json()["draft_version"] == 2
        r = await cl.post(f"/fd/first-mate/factories/{fid}/draft", headers=_hdr(BOB), json={})
        assert r.status_code == 400 and "already has a draft" in r.json()["detail"]
        r = await cl.post(f"/fd/first-mate/factories/{fid}/decline", headers=_hdr(BOB),
                          json={"version": 2, "reason": "not yet"})
        assert r.status_code == 200 and r.json()["state"] == "running" and r.json()["draft"] is None
        r = await cl.post(f"/fd/first-mate/factories/{fid}/draft", headers=_hdr(BOB), json={"title": "x"})
        assert r.status_code == 400


async def test_patch_factory_and_course_goals(deck):
    _open()
    mine = c.get_store().create(scope="user", owner_id=BOB, horizon="long", title="Calmer weeks",
                                success="fewer late nights")
    theirs = c.get_store().create(scope="user", owner_id=CAROL, horizon="long", title="Carol's",
                                  success="done")
    async with _client() as cl:
        f = await _approved(cl)
        r = await cl.patch(f"/fd/first-mate/factories/{f['id']}", headers=_hdr(BOB),
                           json={"title": "Monday page", "share": 2, "goal_id": mine["id"]})
        assert r.status_code == 200 and (r.json()["title"], r.json()["share"]) == ("Monday page", 2)
        assert r.json()["goal_id"] == mine["id"] and r.json()["version"] == 1
        r = await cl.patch(f"/fd/first-mate/factories/{f['id']}", headers=_hdr(BOB), json={"goal_id": theirs["id"]})
        assert r.status_code == 400 and "isn't one of yours" in r.json()["detail"]
        r = await cl.patch(f"/fd/first-mate/factories/{f['id']}", headers=_hdr(BOB), json={"charter": "x"})
        assert r.status_code == 400 and "unknown fields" in r.json()["detail"]


# ── state, trust, delete ───────────────────────────────────────────────

async def test_state_and_trust(deck):
    _open()
    async with _client() as cl:
        f = await _approved(cl, grants={"email": True})
        fid = f["id"]
        r = await cl.post(f"/fd/first-mate/factories/{fid}/state", headers=_hdr(BOB),
                          json={"state": "paused", "reason": "holiday"})
        assert r.status_code == 200 and r.json()["state"] == "paused"
        r = await cl.post(f"/fd/first-mate/factories/{fid}/state", headers=_hdr(BOB), json={"state": "paused_budget"})
        assert r.status_code == 400
        r = await cl.post(f"/fd/first-mate/factories/{fid}/trust", headers=_hdr(BOB), json={"level": "commit", "pin": True})
        assert r.status_code == 200 and r.json()["trust"] == "commit" and r.json()["trust_pinned"] is True
        r = await cl.post(f"/fd/first-mate/factories/{fid}/trust", headers=_hdr(BOB), json={"level": "direct"})
        assert r.status_code == 400 and "direct isn't allowed" in r.json()["detail"]
        r = await cl.post(f"/fd/first-mate/factories/{fid}/trust", headers=_hdr(BOB), json={"level": "commit", "pin": "yes"})
        assert r.status_code == 400


async def test_delete_only_drafts_and_retired(deck):
    _open()
    async with _client() as cl:
        draft = await _factory(cl)
        assert (await cl.delete(f"/fd/first-mate/factories/{draft['id']}", headers=_hdr(BOB))).json() == {"ok": True}
        f = await _approved(cl)
        r = await cl.delete(f"/fd/first-mate/factories/{f['id']}", headers=_hdr(BOB))
        assert r.status_code == 400 and "retire" in r.json()["detail"]
        await cl.post(f"/fd/first-mate/factories/{f['id']}/state", headers=_hdr(BOB), json={"state": "retired"})
        assert (await cl.delete(f"/fd/first-mate/factories/{f['id']}", headers=_hdr(BOB))).status_code == 200
        assert (await cl.get(f"/fd/first-mate/factories/{f['id']}", headers=_hdr(BOB))).status_code == 404


# ── runs ───────────────────────────────────────────────────────────────

async def test_run_now_follow_and_stop(deck):
    _open()
    bat = deck
    async with _client() as cl:
        f = await _approved(cl)
        fid = f["id"]
        r = await cl.post(f"/fd/first-mate/factories/{fid}/run", headers=_hdr(BOB), json={"note": "Short."})
        assert r.status_code == 200, r.text
        run = r.json()
        assert run["status"] == "running" and run["child_id"] in bat.runs
        assert "Short." in bat.runs[run["child_id"]]["task"]
        r = await cl.post(f"/fd/first-mate/factories/{fid}/run", headers=_hdr(BOB), json={})
        assert r.status_code == 400 and "already has a run going" in r.json()["detail"]
        bat.runs[run["child_id"]].update(status="running", cumulative_tokens=1234)
        r = await cl.get(f"/fd/first-mate/factories/{fid}/runs", headers=_hdr(BOB))
        assert r.json()["runs"][0]["tokens"] == 1234
        assert (await cl.post(f"/fd/first-mate/runs/{run['id']}/stop", headers=_hdr(CAROL))).status_code == 404
        r = await cl.post(f"/fd/first-mate/runs/{run['id']}/stop", headers=_hdr(BOB))
        assert r.status_code == 200 and r.json()["status"] == "cancelled"
        assert bat.runs[run["child_id"]]["status"] == "cancelled"
        cards = (await cl.get("/fd/first-mate/factories", headers=_hdr(BOB))).json()["factories"]
        assert cards[0]["last_run"]["status"] == "cancelled"


async def test_run_now_when_the_list_finishes_a_run(deck):
    _open()
    bat = deck
    async with _client() as cl:
        f = await _approved(cl)
        run = (await cl.post(f"/fd/first-mate/factories/{f['id']}/run", headers=_hdr(BOB), json={})).json()
        bat.runs[run["child_id"]].update(status="done", truth="One page.")
        cards = (await cl.get("/fd/first-mate/factories", headers=_hdr(BOB))).json()["factories"]
        assert cards[0]["last_run"]["status"] == "done" and cards[0]["last_run"]["result"] == "One page."


async def test_the_owners_switch_pauses_and_resumes(deck):
    _open()
    bat = deck
    async with _client() as cl:
        await cl.put("/fd/first-mate/prefs", headers=_hdr(BOB), json={"on": True})
        f = await _approved(cl)
        run = (await cl.post(f"/fd/first-mate/factories/{f['id']}/run", headers=_hdr(BOB), json={})).json()
        r = await cl.put("/fd/first-mate/prefs", headers=_hdr(BOB), json={"on": False})
        assert r.status_code == 200 and r.json()["on"] is False
        g = (await cl.get(f"/fd/first-mate/factories/{f['id']}", headers=_hdr(BOB))).json()
        assert g["state"] == "paused" and g["state_reason"] == "First Mate is off"
        assert g["runs"][0]["status"] == "cancelled" and bat.runs[run["child_id"]]["status"] == "cancelled"
        await cl.put("/fd/first-mate/prefs", headers=_hdr(BOB), json={"on": True})
        g = (await cl.get(f"/fd/first-mate/factories/{f['id']}", headers=_hdr(BOB))).json()
        assert g["state"] == "running"


async def test_the_week_and_each_cards_numbers(deck):
    _open()
    from captain_claw.flight_deck import events
    async with _client() as cl:
        await cl.put("/fd/first-mate/prefs", headers=_hdr(BOB), json={"on": True, "weekly_tokens": 10_000_000})
        a = await _approved(cl, template="price-watch", share=1)
        b = await _approved(cl, share=4)
        run = fm.get_store().start_run(a["id"])
        fm.get_store().finish_run(run["id"], "done", tokens=1_500_000)
        ev = events.get_store().add_event(BOB, source="gmail", event_type="new_email", summary="Email from x: y")
        events.get_store().claim_for(ev["id"], {"fm_factory": b["id"], "fm_trigger": "k"})
        status = (await cl.get("/fd/first-mate", headers=_hdr(BOB))).json()
        assert status["week"]["budget"] == 10_000_000 and status["week"]["used"] == 1_500_000
        cards = {c["id"]: c for c in (await cl.get("/fd/first-mate/factories", headers=_hdr(BOB))).json()["factories"]}
        assert (cards[a["id"]]["share_tokens"], cards[a["id"]]["week_tokens"]) == (2_000_000, 1_500_000)
        assert cards[b["id"]]["share_tokens"] == 8_000_000 and cards[b["id"]]["pending_events"] == 1
        assert cards[a["id"]]["next_at"]  # daily 09:00, worked out before the loop saw it
        assert cards[b["id"]]["next_at"]  # weekly mon 07:30


async def test_run_now_refusals(deck):
    _open()
    bat = deck
    async with _client() as cl:
        draft = await _factory(cl)
        r = await cl.post(f"/fd/first-mate/factories/{draft['id']}/run", headers=_hdr(BOB), json={})
        assert r.status_code == 400 and "approved" in r.json()["detail"]
        f = await _approved(cl)
        for i in range(2):
            await bat.create_run(run_id=f"bat_own{i}", owner_id=BOB, title="mine", task="t", status="running")
        r = await cl.post(f"/fd/first-mate/factories/{f['id']}/run", headers=_hdr(BOB), json={})
        assert r.status_code == 400 and "no free slot" in r.json()["detail"]
        r = await cl.post(f"/fd/first-mate/factories/{f['id']}/run", headers=_hdr(BOB), json={"note": 5})
        assert r.status_code == 400
