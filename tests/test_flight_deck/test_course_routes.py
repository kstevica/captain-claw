"""Course routes and the files they write: /fd/course*, /fd/admin/course/settings.

Pinned here:

* the routes need a session; an admin's ``X-FD-Act-As`` is ignored; someone
  else's goal (or agent) is a 404; deck-wide goals are readable by everyone and
  changed only by an admin (403);
* a save rewrites the course files of exactly the agents it reaches: a user goal
  → the owner's agents; an agent goal → that agent; a deck-wide goal → every
  agent, in both the owner's and the members' file;
* paused, finished and stale goals leave the files; an empty course removes them;
* the admin's settings are admin-only, validated and audited;
* the review sweep asks once per review cycle (owners; admins for deck-wide
  goals), the answer resolves it, and a goal going stale rewrites the files;
* a deleted user and a removed agent take their goals with them; a new agent
  gets the deck's and its owner's goals at spawn;
* auth-off decks: the local user owns every agent and edits the deck's goals.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import agent_sharing, line, rate_limiter, server
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import course_carrier as carrier
from captain_claw.flight_deck import tenant_profile as tp
from captain_claw.flight_deck.auth import ACT_AS_HEADER, create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

ADMIN = "user-admin"
BOB = "user-bob"
CAROL = "user-carol"
REMOTE = ("203.0.113.9", 40001)

_REGISTRY = {
    "bob-agent": {"slug": "bob-agent", "name": "Bob's agent", "web_port": 24101,
                  "web_auth": "bob-tok", "owner": BOB, "pid": None},
    "bob-two": {"slug": "bob-two", "name": "Bob's second", "web_port": 24103,
                "web_auth": "bob2-tok", "owner": BOB, "pid": None},
    "carol-agent": {"slug": "carol-agent", "name": "Carol's agent", "web_port": 24102,
                    "web_auth": "carol-tok", "owner": CAROL, "pid": None},
}


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Auth-enabled deck: an admin, Bob (two agents) and Carol (one)."""
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid, role, name in ((ADMIN, "admin", "Ada Admin"), (BOB, "user", "Bob Builder"),
                            (CAROL, "user", "Carol")):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", name, role))
    await db._db.commit()

    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)

    def _no_docker():
        raise RuntimeError("docker unavailable in tests")

    monkeypatch.setattr(server, "get_docker", _no_docker)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(c, "_STORE", c.CourseStore(tmp_path / "course.db"))
    monkeypatch.setattr(carrier, "_last_stale", None)

    server._save_process_registry(dict(_REGISTRY))
    for slug in _REGISTRY:
        (data / slug).mkdir()
    try:
        yield db
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


def _ref(slug: str) -> str:
    return agent_sharing.process_ref(slug, _REGISTRY[slug])


def _file(slug: str, name: str = carrier.FULL_FILE) -> str | None:
    path = tp.context_dir(server.DATA_DIR / slug, "process") / name
    return path.read_text(encoding="utf-8") if path.is_file() else None


async def _post(client, who, **body):
    body.setdefault("success", "done when it's done")
    return await client.post("/fd/course/goals", headers=_hdr(who), json=body)


# ── reading ──────────────────────────────────────────────────────────


class TestRead:
    async def test_needs_a_session(self, deck):
        async with _client() as cl:
            assert (await cl.get("/fd/course")).status_code == 401
            assert (await cl.post("/fd/course/goals", json={})).status_code == 401

    async def test_empty_course(self, deck):
        async with _client() as cl:
            r = await cl.get("/fd/course", headers=_hdr(BOB))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["goals"]["deck"] == [] and body["goals"]["mine"] == []
        assert sorted(a["slug"] for a in body["agents"]) == ["bob-agent", "bob-two"]
        assert set(body["goals"]["agents"]) == {_ref("bob-agent"), _ref("bob-two")}
        assert body["is_admin"] is False
        assert body["settings"]["horizons"]["agent"] == ["medium", "short"]

    async def test_users_see_deck_goals_and_only_their_own(self, deck):
        async with _client() as cl:
            await _post(cl, ADMIN, scope="instance", horizon="strategic", title="Deck way")
            await _post(cl, BOB, scope="user", horizon="long", title="Bob's")
            r = await cl.get("/fd/course", headers=_hdr(CAROL))
        body = r.json()
        assert [g["title"] for g in body["goals"]["deck"]] == ["Deck way"]
        assert body["goals"]["deck"][0]["can_edit"] is False
        assert body["goals"]["mine"] == []

    async def test_act_as_is_ignored(self, deck):
        async with _client() as cl:
            await _post(cl, BOB, scope="user", horizon="long", title="Bob's")
            r = await cl.get("/fd/course", headers=_hdr(ADMIN, act_as=BOB))
        assert r.json()["goals"]["mine"] == []

    async def test_flags(self, deck):
        async with _client() as cl:
            await _post(cl, BOB, scope="user", horizon="long", title="Orphan")
            r = await cl.get("/fd/course", headers=_hdr(BOB))
        flags = r.json()["goals"]["mine"][0]["flags"]
        assert flags == {"due": False, "stale": False, "horizon_retired": False,
                         "serves_nothing": True, "parent_gone": False}


# ── writing and the files ────────────────────────────────────────────


class TestWrite:
    async def test_a_user_goal_reaches_the_owners_agents_only(self, deck):
        async with _client() as cl:
            r = await _post(cl, BOB, scope="user", horizon="long", title="Ship the EU plan",
                            detail="Customers asked.", success="3 EU customers")
        assert r.status_code == 200, r.text
        assert r.json()["agents_updated"] == 2
        for slug in ("bob-agent", "bob-two"):
            text = _file(slug)
            assert "### Bob Builder's goals (your owner)" in text
            assert "  > Ship the EU plan" in text and "  > Done when: 3 EU customers" in text
            assert "a goal is not permission" in text
            assert _file(slug, carrier.DECK_FULL_FILE) is None      # members: nothing of Bob's
        assert _file("carol-agent") is None

    async def test_an_agent_goal_reaches_that_agent_only(self, deck):
        async with _client() as cl:
            r = await _post(cl, BOB, scope="agent", agent_ref=_ref("bob-agent"),
                            horizon="short", title="Clear the inbox backlog")
        assert r.status_code == 200, r.text
        assert r.json()["agents_updated"] == 1
        assert "### Your own goals" in _file("bob-agent")
        assert _file("bob-two") is None

    async def test_someone_elses_agent_is_a_404(self, deck):
        async with _client() as cl:
            r = await _post(cl, BOB, scope="agent", agent_ref=_ref("carol-agent"),
                            horizon="short", title="Sneaky")
        assert r.status_code == 404

    async def test_deck_goals_are_admin_only_and_reach_everyone(self, deck):
        async with _client() as cl:
            denied = await _post(cl, BOB, scope="instance", horizon="strategic", title="Mine now")
            r = await _post(cl, ADMIN, scope="instance", horizon="strategic",
                            title="Be the trusted partner", success="")
        assert denied.status_code == 403
        assert r.status_code == 200, r.text and r.json()["agents_updated"] == 3
        for slug in _REGISTRY:
            assert "### Deck-wide (set by the deck's admin)" in _file(slug)
            member = _file(slug, carrier.DECK_FULL_FILE)
            assert member.startswith("## This deck's course")
            assert "  > Be the trusted partner" in member
            assert _file(slug, carrier.DECK_COMPACT_FILE)

    async def test_members_never_get_the_owners_goals(self, deck):
        async with _client() as cl:
            await _post(cl, ADMIN, scope="instance", horizon="long", title="Deck goal")
            await _post(cl, BOB, scope="user", horizon="long", title="Private plan")
            await _post(cl, BOB, scope="agent", agent_ref=_ref("bob-agent"),
                        horizon="short", title="Agent plan")
        for name in (carrier.DECK_FULL_FILE, carrier.DECK_COMPACT_FILE):
            text = _file("bob-agent", name)
            assert "Deck goal" in text
            assert "Private plan" not in text and "Agent plan" not in text

    @pytest.mark.parametrize("body,status", [
        ({"scope": "team", "horizon": "long", "title": "x"}, 400),
        ({"scope": "user", "horizon": "forever", "title": "x"}, 400),
        ({"scope": "user", "horizon": "long", "title": ""}, 400),
        ({"scope": "user", "horizon": "long", "title": "x", "success": ""}, 400),
        ({"scope": "user", "horizon": "long", "title": "x" * 121}, 400),
        ({"scope": "instance", "horizon": "short", "title": "x"}, 403),
    ])
    async def test_bad_goals(self, deck, body, status):
        async with _client() as cl:
            body = {"success": "s", **body}
            r = await cl.post("/fd/course/goals", headers=_hdr(BOB), json=body)
        assert r.status_code == status, r.text

    async def test_parents_are_checked_for_visibility(self, deck):
        async with _client() as cl:
            carols = (await _post(cl, CAROL, scope="user", horizon="strategic",
                                  title="Carol's")).json()["goal"]
            deckg = (await _post(cl, ADMIN, scope="instance", horizon="strategic",
                                 title="Deck")).json()["goal"]
            r1 = await _post(cl, BOB, scope="user", horizon="long", title="x",
                             parent_id=carols["id"])
            r2 = await _post(cl, BOB, scope="user", horizon="long", title="y",
                             parent_id=deckg["id"])
        assert r1.status_code == 404
        assert r2.status_code == 200 and r2.json()["goal"]["parent_id"] == deckg["id"]
        assert f"serves G:{deckg['id']}" in _file("bob-agent")

    async def test_someone_elses_goal_is_a_404(self, deck):
        async with _client() as cl:
            g = (await _post(cl, BOB, scope="user", horizon="long", title="Bob's")).json()["goal"]
            gid = g["id"]
            assert (await cl.get(f"/fd/course/goals/{gid}", headers=_hdr(CAROL))).status_code == 404
            assert (await cl.patch(f"/fd/course/goals/{gid}", headers=_hdr(CAROL),
                                   json={"title": "x"})).status_code == 404
            assert (await cl.post(f"/fd/course/goals/{gid}/drop",
                                  headers=_hdr(CAROL))).status_code == 404
            assert (await cl.post(f"/fd/course/goals/{gid}/drop",
                                  headers=_hdr(ADMIN))).status_code == 404

    async def test_non_admins_cant_change_deck_goals(self, deck):
        async with _client() as cl:
            g = (await _post(cl, ADMIN, scope="instance", horizon="long",
                             title="Deck")).json()["goal"]
            read = await cl.get(f"/fd/course/goals/{g['id']}", headers=_hdr(BOB))
            edit = await cl.patch(f"/fd/course/goals/{g['id']}", headers=_hdr(BOB),
                                  json={"title": "x"})
            drop = await cl.post(f"/fd/course/goals/{g['id']}/drop", headers=_hdr(BOB))
        assert read.status_code == 200 and read.json()["can_edit"] is False
        assert edit.status_code == 403 and drop.status_code == 403

    async def test_edit_and_lifecycle_rewrite_the_files(self, deck):
        async with _client() as cl:
            g = (await _post(cl, BOB, scope="user", horizon="long", title="Old")).json()["goal"]
            r = await cl.patch(f"/fd/course/goals/{g['id']}", headers=_hdr(BOB),
                               json={"title": "New title"})
            assert r.status_code == 200 and "New title" in _file("bob-agent")
            r = await cl.post(f"/fd/course/goals/{g['id']}/pause", headers=_hdr(BOB))
            assert r.json()["goal"]["status"] == "paused"
            assert _file("bob-agent") is None                      # nothing live: no file
            await cl.post(f"/fd/course/goals/{g['id']}/resume", headers=_hdr(BOB))
            assert "New title" in _file("bob-agent")
            bad = await cl.post(f"/fd/course/goals/{g['id']}/accept", headers=_hdr(BOB))
            unknown = await cl.post(f"/fd/course/goals/{g['id']}/explode", headers=_hdr(BOB))
            detail = await cl.get(f"/fd/course/goals/{g['id']}", headers=_hdr(BOB))
        assert bad.status_code == 409 and unknown.status_code == 404
        events = [e["event"] for e in detail.json()["events"]]
        assert events == ["status", "status", "edited", "created"]

    async def test_links(self, deck):
        async with _client() as cl:
            g = (await _post(cl, BOB, scope="user", horizon="long", title="G")).json()["goal"]
            r = await cl.post(f"/fd/course/goals/{g['id']}/links", headers=_hdr(BOB),
                              json={"kind": "task", "ref_id": "t-1", "note": "the launch task"})
            assert r.status_code == 200
            d = await cl.delete(f"/fd/course/goals/{g['id']}/links/task/t-1", headers=_hdr(BOB))
            again = await cl.delete(f"/fd/course/goals/{g['id']}/links/task/t-1",
                                    headers=_hdr(BOB))
        assert d.status_code == 200 and again.status_code == 404

    async def test_preview_is_what_the_agent_gets(self, deck):
        async with _client() as cl:
            await _post(cl, BOB, scope="user", horizon="long", title="Previewed")
            r = await cl.get("/fd/course/preview", headers=_hdr(BOB),
                             params={"agent_ref": _ref("bob-agent")})
            other = await cl.get("/fd/course/preview", headers=_hdr(BOB),
                                 params={"agent_ref": _ref("carol-agent")})
        assert r.json()["owner"]["full"] == _file("bob-agent")
        assert r.json()["member"]["full"] == ""
        assert other.status_code == 404


# ── the admin's settings ─────────────────────────────────────────────


class TestSettings:
    async def test_admin_only(self, deck):
        async with _client() as cl:
            assert (await cl.get("/fd/admin/course/settings",
                                 headers=_hdr(BOB))).status_code == 403
            assert (await cl.put("/fd/admin/course/settings", headers=_hdr(BOB),
                                 json={"caps": {"short": 9}})).status_code == 403

    async def test_save_validate_and_audit(self, deck):
        async with _client() as cl:
            r = await cl.put("/fd/admin/course/settings", headers=_hdr(ADMIN),
                             json={"horizons": {"agent": ["long", "medium", "short"]},
                                   "caps": {"short": 4}})
            bad = await cl.put("/fd/admin/course/settings", headers=_hdr(ADMIN),
                               json={"caps": {"short": 99}})
            got = await cl.get("/fd/admin/course/settings", headers=_hdr(ADMIN))
            course = await cl.get("/fd/course", headers=_hdr(BOB))
        assert r.status_code == 200, r.text
        assert bad.status_code == 400
        assert got.json()["horizons"]["agent"] == ["long", "medium", "short"]
        assert got.json()["caps"]["short"] == 4
        assert course.json()["settings"]["horizons"]["agent"] == ["long", "medium", "short"]
        rows = await deck.get_usage_logs(user_id=ADMIN, event_type="course_settings_update")
        assert len(rows) == 1

    async def test_a_retired_horizon_is_flagged_and_kept(self, deck):
        async with _client() as cl:
            await _post(cl, BOB, scope="user", horizon="short", title="This week")
            await cl.put("/fd/admin/course/settings", headers=_hdr(ADMIN),
                         json={"horizons": {"user": ["strategic", "long"]}})
            r = await cl.get("/fd/course", headers=_hdr(BOB))
            new = await _post(cl, BOB, scope="user", horizon="short", title="Another")
        goal = r.json()["goals"]["mine"][0]
        assert goal["flags"]["horizon_retired"] is True and goal["status"] == "active"
        assert "This week" in _file("bob-agent")
        assert new.status_code == 400


# ── the review sweep ─────────────────────────────────────────────────


class TestReview:
    async def test_due_goals_are_asked_about_once_and_answered_on_review(self, deck):
        async with _client() as cl:
            g = (await _post(cl, BOB, scope="user", horizon="short", title="Weekly")).json()["goal"]
            due = datetime.fromisoformat(g["review_at"]) + timedelta(minutes=1)
            await carrier.review_round(deck, now=due)
            await carrier.review_round(deck, now=due)              # once per cycle
            items = await deck.list_line_items(BOB, needs_you=True)
            assert [i["title"] for i in items] == ["Still on course? Weekly"]
            assert items[0]["ref_type"] == "course_review" and not items[0]["resolved_at"]
            await deck.mark_line_item_read(items[0]["id"], BOB)    # reading doesn't answer it
            assert not (await deck.list_line_items(BOB, needs_you=True))[0]["resolved_at"]
            r = await cl.post(f"/fd/course/goals/{g['id']}/review", headers=_hdr(BOB))
        assert r.status_code == 200
        assert (await deck.list_line_items(BOB, needs_you=True))[0]["resolved_at"]

    async def test_deck_goals_are_asked_of_the_admins(self, deck):
        async with _client() as cl:
            g = (await _post(cl, ADMIN, scope="instance", horizon="long",
                             title="Deck review")).json()["goal"]
        due = datetime.fromisoformat(g["review_at"]) + timedelta(minutes=1)
        await carrier.review_round(deck, now=due)
        assert [i["title"] for i in await deck.list_line_items(ADMIN)] == [
            "Still on course? Deck review"]
        assert await deck.list_line_items(BOB) == []

    async def test_a_stale_goal_leaves_the_files(self, deck):
        async with _client() as cl:
            g = (await _post(cl, BOB, scope="user", horizon="short", title="Weekly")).json()["goal"]
            g2 = (await _post(cl, BOB, scope="user", horizon="long", title="Quarter")).json()["goal"]
        at = datetime.fromisoformat(g["review_at"])
        await carrier.review_round(deck, now=at + timedelta(minutes=1))   # baseline
        stale_at = at + timedelta(days=4)
        assert c.is_stale(c.get_store().get(g["id"]), stale_at)
        assert not c.is_stale(c.get_store().get(g2["id"]), stale_at)
        # The files are composed against real time; pretend it is stale_at.
        real = carrier.goals_for_agent
        carrier_now = {"t": stale_at}

        def at_stale(subject, ref, *, now=None):
            return real(subject, ref, now=carrier_now["t"])

        import unittest.mock as um
        with um.patch.object(carrier, "goals_for_agent", at_stale):
            out = await carrier.review_round(deck, now=stale_at)
        assert out["refreshed"] == 2
        text = _file("bob-agent")
        assert "Quarter" in text and "Weekly" not in text


# ── forgetting and spawning ──────────────────────────────────────────


class TestLifecycle:
    async def test_deleting_a_user_forgets_their_goals(self, deck, monkeypatch):
        from captain_claw.flight_deck import payments_routes

        # The payments cleanup opens an aiosqlite store nothing closes in a test,
        # and its worker thread then keeps pytest from exiting. Not under test here.
        async def _no_payments(user_id):
            return None

        monkeypatch.setattr(payments_routes, "forget_user", _no_payments)
        async with _client() as cl:
            await _post(cl, ADMIN, scope="instance", horizon="long", title="Deck")
            await _post(cl, BOB, scope="user", horizon="long", title="Bob's")
            await _post(cl, BOB, scope="agent", agent_ref=_ref("bob-agent"),
                        horizon="short", title="Agent's")
            r = await cl.delete(f"/fd/admin/users/{BOB}", headers=_hdr(ADMIN))
        assert r.status_code == 200, r.text
        left = c.get_store().list_goals()
        assert [g["title"] for g in left] == ["Deck"]

    async def test_a_removed_agent_takes_its_goals(self, deck):
        async with _client() as cl:
            await _post(cl, BOB, scope="agent", agent_ref=_ref("bob-agent"),
                        horizon="short", title="Agent's")
            await _post(cl, BOB, scope="user", horizon="long", title="Bob's")
        await server._forget_shared_agent(_ref("bob-agent"), BOB)
        assert [g["title"] for g in c.get_store().list_goals()] == ["Bob's"]

    async def test_a_new_agent_gets_the_deck_and_owner_goals(self, deck):
        async with _client() as cl:
            await _post(cl, ADMIN, scope="instance", horizon="long", title="Deck")
            await _post(cl, BOB, scope="user", horizon="long", title="Bob's")
            await _post(cl, BOB, scope="agent", agent_ref=_ref("bob-agent"),
                        horizon="short", title="Not for the new one")
        new_dir = server.DATA_DIR / "bob-new"
        new_dir.mkdir()
        await carrier.write_on_spawn(new_dir, BOB, "process")
        text = (tp.context_dir(new_dir, "process") / carrier.FULL_FILE).read_text()
        assert "Deck" in text and "Bob's" in text and "Not for the new one" not in text

    async def test_a_failed_write_leaves_no_files(self, deck, monkeypatch):
        async with _client() as cl:
            await _post(cl, BOB, scope="user", horizon="long", title="Bob's")
        assert _file("bob-agent")

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(carrier, "write_files", boom)
        await carrier.refresh_agents(deck, BOB)
        assert _file("bob-agent") is None


class TestAuthOff:
    async def test_the_local_user_owns_every_agent_and_the_deck(self, deck, monkeypatch):
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        async with _client() as cl:
            d = await cl.post("/fd/course/goals", json={
                "scope": "instance", "horizon": "strategic", "title": "Home lab direction"})
            u = await cl.post("/fd/course/goals", json={
                "scope": "user", "horizon": "long", "title": "Mine", "success": "s"})
            r = await cl.get("/fd/course")
        assert d.status_code == 200 and u.status_code == 200, (d.text, u.text)
        body = r.json()
        assert body["is_admin"] is True and body["auth_enabled"] is False
        assert len(body["agents"]) == 3
        assert body["goals"]["deck"][0]["can_edit"] is True
        assert u.json()["goal"]["owner_id"] == "local"
        for slug in _REGISTRY:
            text = _file(slug)
            assert "Home lab direction" in text and "Mine" in text


def test_review_ref_is_per_cycle():
    g = {"id": "goal_x", "review_at": "2026-01-01T00:00:00+00:00"}
    assert carrier.review_ref(g) == "goal_x@2026-01-01T00:00:00+00:00"


def test_course_review_is_answered_elsewhere():
    assert "course_review" in line.ANSWERED_REFS
    assert line.classify("course_review") == ("ask", True, "deck")


def test_now_is_utc():
    assert datetime.now(UTC).tzinfo is not None
