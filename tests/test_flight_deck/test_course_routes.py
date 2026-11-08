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
                         "serves_nothing": True, "parent_gone": False, "off_course": False}


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
                              json={"kind": "project", "ref_id": "p-1", "note": "the launch project"})
            assert r.status_code == 200
            # A Taskboard task that isn't Bob's (there is none): refused.
            task = await cl.post(f"/fd/course/goals/{g['id']}/links", headers=_hdr(BOB),
                                 json={"kind": "task", "ref_id": "t-1"})
            assert task.status_code == 404
            d = await cl.delete(f"/fd/course/goals/{g['id']}/links/project/p-1", headers=_hdr(BOB))
            again = await cl.delete(f"/fd/course/goals/{g['id']}/links/project/p-1",
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


# ── Phase 2: evidence, "Serves a goal", the agent's side ─────────────


class TestServed:
    async def test_set_read_and_clear_the_goal_an_item_serves(self, deck):
        async with _client() as cl:
            g = (await _post(cl, BOB, scope="user", horizon="long", title="Ship EU")).json()["goal"]
            g2 = (await _post(cl, BOB, scope="user", horizon="long", title="Other")).json()["goal"]
            none = await cl.get("/fd/course/served", headers=_hdr(BOB),
                                params={"kind": "project", "ref_id": "p1"})
            put = await cl.put("/fd/course/served", headers=_hdr(BOB), json={
                "kind": "project", "ref_id": "p1", "goal_id": g["id"], "note": "EU launch project"})
            moved = await cl.put("/fd/course/served", headers=_hdr(BOB), json={
                "kind": "project", "ref_id": "p1", "goal_id": g2["id"], "note": "EU launch project"})
            got = await cl.get("/fd/course/served", headers=_hdr(BOB),
                               params={"kind": "project", "ref_id": "p1"})
            cleared = await cl.put("/fd/course/served", headers=_hdr(BOB), json={
                "kind": "project", "ref_id": "p1", "goal_id": ""})
        assert none.json() == {"goal": None}
        assert put.json()["goal"]["id"] == g["id"] and moved.json()["goal"]["id"] == g2["id"]
        assert got.json()["goal"]["title"] == "Other"
        assert cleared.json() == {"goal": None}
        assert c.get_store().links_for("project", "p1") == []

    async def test_only_your_own_items_and_goals(self, deck):
        async with _client() as cl:
            carols = (await _post(cl, CAROL, scope="user", horizon="long", title="C")).json()["goal"]
            mine = (await _post(cl, BOB, scope="user", horizon="long", title="B")).json()["goal"]
            other_goal = await cl.put("/fd/course/served", headers=_hdr(BOB), json={
                "kind": "project", "ref_id": "p1", "goal_id": carols["id"]})
            not_mine = await cl.put("/fd/course/served", headers=_hdr(BOB), json={
                "kind": "schedule", "ref_id": "job_nope", "goal_id": mine["id"]})
            bad_kind = await cl.get("/fd/course/served", headers=_hdr(BOB),
                                    params={"kind": "session", "ref_id": "s1"})
        assert other_goal.status_code == 404 and not_mine.status_code == 404
        assert bad_kind.status_code == 400

    async def test_on_a_deck_goal_each_user_sees_only_their_own_work(self, deck):
        async with _client() as cl:
            g = (await _post(cl, ADMIN, scope="instance", horizon="long", title="Deck")).json()["goal"]
            await cl.put("/fd/course/served", headers=_hdr(BOB), json={
                "kind": "project", "ref_id": "p-bob", "goal_id": g["id"], "note": "Bob's secret project"})
            await cl.put("/fd/course/served", headers=_hdr(CAROL), json={
                "kind": "project", "ref_id": "p-carol", "goal_id": g["id"], "note": "Carol's project"})
            bob = await cl.get(f"/fd/course/goals/{g['id']}", headers=_hdr(BOB))
            carol = await cl.get(f"/fd/course/goals/{g['id']}", headers=_hdr(CAROL))
            sneaky = await cl.delete(f"/fd/course/goals/{g['id']}/links/project/p-bob",
                                     headers=_hdr(CAROL))
            page = await cl.get("/fd/course", headers=_hdr(CAROL))
        assert [lk["note"] for lk in bob.json()["links"]] == ["Bob's secret project"]
        assert [lk["note"] for lk in carol.json()["links"]] == ["Carol's project"]
        assert all(e["event"] != "linked" or e["detail"]["owner_id"] == CAROL
                   for e in carol.json()["events"])
        assert sneaky.status_code == 404
        assert page.json()["goals"]["deck"][0]["last_moved"]


def _agent(monkeypatch, token: str):
    from captain_claw.flight_deck import google_oauth_routes

    monkeypatch.setattr(google_oauth_routes, "_authorize_agent_call", lambda request: None)
    owners = {e["web_auth"]: e["owner"] for e in _REGISTRY.values()}

    async def owner_of(request, service=""):
        return owners[request.headers.get("X-Agent-Auth", "")]

    monkeypatch.setattr(google_oauth_routes, "_agent_owner", owner_of)
    return {"X-Agent-Auth": token}


class TestAgentSide:
    async def test_view_link_report(self, deck, monkeypatch):
        hdr = _agent(monkeypatch, "bob-tok")
        async with _client() as cl:
            await _post(cl, ADMIN, scope="instance", horizon="long", title="Deck")
            mine = (await _post(cl, BOB, scope="user", horizon="long", title="Bob's")).json()["goal"]
            own = (await _post(cl, BOB, scope="agent", agent_ref=_ref("bob-agent"), horizon="short",
                               title="Agent's")).json()["goal"]
            await _post(cl, BOB, scope="agent", agent_ref=_ref("bob-two"), horizon="short",
                        title="The other agent's")
            await _post(cl, CAROL, scope="user", horizon="long", title="Carol's")
            view = await cl.post("/fd/course/agent/view", headers=hdr)
            link = await cl.post("/fd/course/agent/link", headers=hdr, json={
                "goal_id": f"G:{own['id']}", "kind": "session", "ref_id": "s-42", "note": "drafted it"})
            report = await cl.post("/fd/course/agent/report", headers=hdr, json={
                "goal_id": mine["id"], "progress": 0.5, "note": "half there", "looks_achieved": True})
            again = await cl.post("/fd/course/agent/report", headers=hdr, json={
                "goal_id": mine["id"], "progress": 0.9, "looks_achieved": True})
        titles = {g["title"] for g in view.json()["goals"]}
        assert titles == {"Deck", "Bob's", "Agent's"}
        assert link.status_code == 200 and link.json()["goal"] == "Agent's"
        assert c.get_store().links(own["id"])[0]["owner_id"] == BOB
        assert report.status_code == 200 and report.json()["asked_owner"] is True
        assert again.json()["asked_owner"] is False                     # once a day
        assert c.get_store().get(mine["id"])["progress"] == 0.9
        items = await deck.list_line_items(BOB, needs_you=True)
        assert [i["title"] for i in items] == ["Looks achieved? Bob's"]
        assert items[0]["type"] == "course_looks_achieved"

    async def test_refusals(self, deck, monkeypatch):
        hdr = _agent(monkeypatch, "bob-tok")
        async with _client() as cl:
            deckg = (await _post(cl, ADMIN, scope="instance", horizon="long", title="Deck")).json()["goal"]
            carols = (await _post(cl, CAROL, scope="user", horizon="long", title="C")).json()["goal"]
            mine = (await _post(cl, BOB, scope="user", horizon="long", title="B")).json()["goal"]
            on_deck = await cl.post("/fd/course/agent/report", headers=hdr,
                                    json={"goal_id": deckg["id"], "progress": 0.5})
            on_carol = await cl.post("/fd/course/agent/link", headers=hdr, json={
                "goal_id": carols["id"], "kind": "session", "ref_id": "s"})
            bad_kind = await cl.post("/fd/course/agent/link", headers=hdr, json={
                "goal_id": mine["id"], "kind": "line_item", "ref_id": "x"})
            not_owned = await cl.post("/fd/course/agent/link", headers=hdr, json={
                "goal_id": mine["id"], "kind": "plan", "ref_id": "plan_nope"})
            bad_progress = await cl.post("/fd/course/agent/report", headers=hdr,
                                         json={"goal_id": mine["id"], "progress": 2})
            stranger = await cl.post("/fd/course/agent/view", headers={"X-Agent-Auth": "nope"})
        assert on_deck.status_code == 400 and on_carol.status_code == 404
        assert bad_kind.status_code == 400 and not_owned.status_code == 404
        assert bad_progress.status_code == 400 and stranger.status_code == 403


# ── Phase 3: suggestions ─────────────────────────────────────────────


class TestProposals:
    async def test_an_agent_suggests_and_the_owner_accepts(self, deck, monkeypatch):
        hdr = _agent(monkeypatch, "bob-tok")
        async with _client() as cl:
            r = await cl.post("/fd/course/agent/propose", headers=hdr, json={
                "scope": "agent", "horizon": "short", "title": "Clear the backlog",
                "success": "zero open", "rationale": "it keeps growing"})
            assert r.status_code == 200, r.text
            gid = r.json()["goal"]["id"]
            assert r.json()["auto_accepted"] is False
            page = await cl.get("/fd/course", headers=_hdr(BOB))
            mine = page.json()["goals"]["agents"][_ref("bob-agent")]
            assert [(g["status"], g["proposer"]) for g in mine] == [("proposed", "Bob's agent")]
            assert _file("bob-agent") is None or "Clear the backlog" not in _file("bob-agent")
            items = await deck.list_line_items(BOB, needs_you=True)
            assert [i["title"] for i in items] == ["Goal proposed: Clear the backlog"]
            await deck.mark_line_item_read(items[0]["id"], BOB)                   # reading doesn't answer it
            assert not (await deck.list_line_items(BOB, needs_you=True))[0]["resolved_at"]
            acc = await cl.post(f"/fd/course/goals/{gid}/accept", headers=_hdr(BOB))
        assert acc.status_code == 200 and acc.json()["goal"]["status"] == "active"
        assert (await deck.list_line_items(BOB, needs_you=True))[0]["resolved_at"]
        assert "Clear the backlog" in _file("bob-agent")

    async def test_decline_and_its_memory(self, deck, monkeypatch):
        hdr = _agent(monkeypatch, "bob-tok")
        async with _client() as cl:
            gid = (await cl.post("/fd/course/agent/propose", headers=hdr, json={
                "scope": "user", "horizon": "medium", "title": "Hire a bookkeeper",
                "success": "hired"})).json()["goal"]["id"]
            dec = await cl.post(f"/fd/course/goals/{gid}/decline", headers=_hdr(BOB))
            again = await cl.post("/fd/course/agent/propose", headers=hdr, json={
                "scope": "user", "horizon": "medium", "title": "Hire a bookkeeper", "success": "hired"})
            carol = await cl.post(f"/fd/course/goals/{gid}/accept", headers=_hdr(CAROL))
        assert dec.json()["goal"]["status"] == "declined"
        assert again.status_code == 400 and "declined recently" in again.json()["detail"]
        assert carol.status_code == 404
        assert (await deck.list_line_items(BOB, needs_you=True))[0]["resolved_at"]

    async def test_a_suggested_change(self, deck, monkeypatch):
        hdr = _agent(monkeypatch, "bob-tok")
        async with _client() as cl:
            g = (await _post(cl, BOB, scope="user", horizon="long", title="Ship EU")).json()["goal"]
            deckg = (await _post(cl, ADMIN, scope="instance", horizon="long", title="Deck")).json()["goal"]
            r = await cl.post("/fd/course/agent/propose_change", headers=hdr, json={
                "goal_id": f"G:{g['id']}", "changes": {"title": "Ship EU and UK"}, "rationale": "UK too"})
            on_deck = await cl.post("/fd/course/agent/propose_change", headers=hdr, json={
                "goal_id": deckg["id"], "changes": {"title": "Mine now"}})
            acc = await cl.post(f"/fd/course/goals/{r.json()['goal']['id']}/accept", headers=_hdr(BOB))
        assert r.status_code == 200, r.text
        assert on_deck.status_code == 400
        assert acc.json()["goal"]["title"] == "Ship EU and UK"
        assert c.get_store().get(g["id"])["status"] == "superseded"
        assert "Ship EU and UK" in _file("bob-agent")

    async def test_auto_accept_and_prefs(self, deck, monkeypatch):
        hdr = _agent(monkeypatch, "bob-tok")
        async with _client() as cl:
            p0 = await cl.get("/fd/course/prefs", headers=_hdr(BOB))
            put = await cl.put("/fd/course/prefs", headers=_hdr(BOB), json={"auto_accept_agent_short": True})
            bad = await cl.put("/fd/course/prefs", headers=_hdr(BOB), json={"auto_accept_agent_short": 1})
            r = await cl.post("/fd/course/agent/propose", headers=hdr, json={
                "scope": "agent", "horizon": "short", "title": "Inbox zero", "success": "zero"})
        assert p0.json() == {"auto_accept_agent_short": False} and put.status_code == 200
        assert bad.status_code == 400
        assert r.json()["auto_accepted"] is True and r.json()["goal"]["status"] == "active"
        items = await deck.list_line_items(BOB)
        assert [(i["title"], i["needs_you"]) for i in items] == [("Added to Bob's agent's course: Inbox zero", 0)]
        assert "Inbox zero" in _file("bob-agent")

    async def test_the_admins_switch_and_limits(self, deck, monkeypatch):
        hdr = _agent(monkeypatch, "bob-tok")
        async with _client() as cl:
            put = await cl.put("/fd/admin/course/settings", headers=_hdr(ADMIN),
                               json={"proposals": {"recommendations": False}})
            got = await cl.get("/fd/admin/course/settings", headers=_hdr(ADMIN))
            off = await cl.post("/fd/course/agent/propose", headers=hdr, json={
                "scope": "agent", "horizon": "short", "title": "x", "success": "y"})
            page = await cl.get("/fd/course", headers=_hdr(BOB))
            bad = await cl.put("/fd/admin/course/settings", headers=_hdr(ADMIN),
                               json={"proposals": {"deck_per_week": -1}})
        assert put.status_code == 200 and got.json()["proposals"]["recommendations"] is False
        assert got.json()["per_week_range"] == [0, 20]
        assert off.status_code == 400 and "switched off" in off.json()["detail"]
        assert page.json()["recommendations"] is False
        assert bad.status_code == 400

    async def test_agent_proposal_refusals(self, deck, monkeypatch):
        hdr = _agent(monkeypatch, "bob-tok")
        async with _client() as cl:
            carols = (await _post(cl, CAROL, scope="user", horizon="long", title="C")).json()["goal"]
            bad_scope = await cl.post("/fd/course/agent/propose", headers=hdr, json={
                "scope": "instance", "horizon": "long", "title": "x", "success": "y"})
            other_parent = await cl.post("/fd/course/agent/propose", headers=hdr, json={
                "scope": "agent", "horizon": "short", "title": "x", "success": "y",
                "parent_id": carols["id"]})
            no_success = await cl.post("/fd/course/agent/propose", headers=hdr, json={
                "scope": "agent", "horizon": "short", "title": "x"})
        assert bad_scope.status_code == 400 and other_parent.status_code == 404
        assert no_success.status_code == 400


# ── Phase 4: progress, off course, conflicts ─────────────────────────


@pytest.fixture
def checks(monkeypatch):
    """The background conflict check, recorded instead of run."""
    from captain_claw.flight_deck import course_conflicts

    seen: list[tuple[str, str, str]] = []
    monkeypatch.setattr(course_conflicts, "check_soon",
                        lambda db, goal, *, thinker: seen.append((goal["id"], goal["status"], thinker)))
    return seen


def _backdate(goal_id: str, days: int) -> None:
    store = c.get_store()
    at = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    with store._lock:
        store._c().execute("UPDATE course_goals SET decided_at = ?, created_at = ? WHERE id = ?",
                           (at, at, goal_id))
        store._c().commit()


class TestProgress:
    async def test_a_measured_goal(self, deck, checks):
        async with _client() as cl:
            g = (await _post(cl, BOB, scope="user", horizon="long", title="Hire",
                             metric={"kind": "manual", "target": 4, "unit": "hires"})).json()["goal"]
            other = (await _post(cl, BOB, scope="user", horizon="long", title="Linked",
                                 metric={"kind": "link_count"})).json()["goal"]
            value = await cl.put(f"/fd/course/goals/{g['id']}/value", headers=_hdr(BOB), json={"value": 3})
            not_manual = await cl.put(f"/fd/course/goals/{other['id']}/value", headers=_hdr(BOB),
                                      json={"value": 3})
            carols = await cl.put(f"/fd/course/goals/{g['id']}/value", headers=_hdr(CAROL),
                                  json={"value": 4})
            bad = await cl.put(f"/fd/course/goals/{g['id']}/value", headers=_hdr(BOB),
                               json={"value": "lots"})
            page = await cl.get("/fd/course", headers=_hdr(BOB))
            detail = await cl.get(f"/fd/course/goals/{g['id']}", headers=_hdr(BOB))
            deck_bad = await _post(cl, ADMIN, scope="instance", horizon="long", title="Deck",
                                   metric={"kind": "taskboard_done"})
            deck_goal = (await _post(cl, ADMIN, scope="instance", horizon="long", title="Deck",
                                     metric={"kind": "manual", "target": 100})).json()["goal"]
            not_admin = await cl.put(f"/fd/course/goals/{deck_goal['id']}/value", headers=_hdr(BOB),
                                     json={"value": 1})
        assert value.status_code == 200 and value.json()["measure"]["progress"] == 0.75
        assert not_manual.status_code == 400 and carols.status_code == 404 and bad.status_code == 400
        shown = {x["title"]: x for x in page.json()["goals"]["mine"]}
        assert shown["Hire"]["measure"]["value"] == 3.0 and shown["Linked"]["measure"]["value"] == 0
        assert page.json()["metric_kinds"]["manual"] == "A number you enter"
        assert detail.json()["measure"]["target"] == 4.0
        assert deck_bad.status_code == 400 and "deck-wide" in deck_bad.json()["detail"]
        assert not_admin.status_code == 403

    async def test_off_course(self, deck, checks):
        async with _client() as cl:
            idle = (await _post(cl, BOB, scope="user", horizon="long", title="Idle")).json()["goal"]
            _backdate(idle["id"], 20)
            orphan = (await _post(cl, BOB, scope="agent", agent_ref=_ref("bob-agent"),
                                  horizon="short", title="Orphan")).json()["goal"]
            page = await cl.get("/fd/course", headers=_hdr(BOB))
        mine = page.json()["goals"]["mine"][0]
        assert mine["flags"]["off_course"] is True
        assert mine["off_course_reason"].startswith("No agent goal serves it")
        agent = page.json()["goals"]["agents"][_ref("bob-agent")][0]
        assert agent["id"] == orphan["id"] and agent["flags"]["off_course"] is True

    async def test_conflicts_are_shown_only_to_whom_they_concern(self, deck, checks):
        async with _client() as cl:
            d = (await _post(cl, ADMIN, scope="instance", horizon="long", title="Cut costs")).json()["goal"]
            g = (await _post(cl, BOB, scope="user", horizon="long", title="Hire five")).json()["goal"]
            cf = c.get_store().add_conflict(g["id"], d["id"], "hiring costs money")
            from captain_claw.flight_deck import course_conflicts

            assert await course_conflicts.announce(deck, cf) == 1
            bob = await cl.get("/fd/course", headers=_hdr(BOB))
            carol = await cl.get("/fd/course", headers=_hdr(CAROL))
            detail = await cl.get(f"/fd/course/goals/{d['id']}", headers=_hdr(BOB))
            carol_detail = await cl.get(f"/fd/course/goals/{d['id']}", headers=_hdr(CAROL))
            not_hers = await cl.post(f"/fd/course/conflicts/{cf['id']}/dismiss", headers=_hdr(CAROL))
            not_admins = await cl.post(f"/fd/course/conflicts/{cf['id']}/dismiss", headers=_hdr(ADMIN))
            waiting = await deck.list_line_items(BOB, needs_you=True)
            dismissed = await cl.post(f"/fd/course/conflicts/{cf['id']}/dismiss", headers=_hdr(BOB))
            after = await cl.get("/fd/course", headers=_hdr(BOB))
        on_deck = bob.json()["goals"]["deck"][0]["conflicts"]
        assert [(x["id"], x["other_title"], x["reason"]) for x in on_deck] == \
            [(cf["id"], "Hire five", "hiring costs money")]
        assert bob.json()["goals"]["mine"][0]["conflicts"][0]["other_title"] == "Cut costs"
        assert carol.json()["goals"]["deck"][0]["conflicts"] == []
        assert detail.json()["conflicts"][0]["other_id"] == g["id"]
        assert carol_detail.json()["conflicts"] == []
        assert not_hers.status_code == 404 and not_admins.status_code == 404
        assert [(i["ref_type"], i["resolved_at"]) for i in waiting] == [("course_conflict", "")]
        assert dismissed.status_code == 200 and dismissed.json()["conflict"]["status"] == "dismissed"
        assert after.json()["goals"]["mine"][0]["conflicts"] == []
        assert all(i["resolved_at"] for i in await deck.list_line_items(BOB, needs_you=True))

    async def test_when_the_check_runs(self, deck, checks, monkeypatch):
        from captain_claw.flight_deck import course_conflicts

        resolved: list[str] = []

        async def resolve_for_goal(db, goal):
            resolved.append(goal["id"])
            return 0

        monkeypatch.setattr(course_conflicts, "resolve_for_goal", resolve_for_goal)
        async with _client() as cl:
            g = (await _post(cl, BOB, scope="user", horizon="long", title="Hire five")).json()["goal"]
            await cl.patch(f"/fd/course/goals/{g['id']}", headers=_hdr(BOB), json={"due_at": "2027-01-01"})
            await cl.patch(f"/fd/course/goals/{g['id']}", headers=_hdr(BOB), json={"title": "Hire six"})
            p = c.get_store().propose_change(g["id"], {"title": "Hire seven"}, origin="deck",
                                             proposed_by="deck")
            await cl.post(f"/fd/course/goals/{p['id']}/accept", headers=_hdr(BOB))
            await cl.post(f"/fd/course/goals/{p['id']}/drop", headers=_hdr(BOB))
        # Added, reworded and accepted: checked; a new due date: not.
        assert [x[0] for x in checks] == [g["id"], g["id"], p["id"]]
        assert all(x[2] == BOB for x in checks)
        # The replaced goal, then the dropped one, leave their conflicts.
        assert resolved == [g["id"], p["id"]]

    async def test_the_admins_automation_switches(self, deck):
        async with _client() as cl:
            put = await cl.put("/fd/admin/course/settings", headers=_hdr(ADMIN),
                               json={"automation": {"goal_driven_candidates": True}})
            got = await cl.get("/fd/admin/course/settings", headers=_hdr(ADMIN))
            bad = await cl.put("/fd/admin/course/settings", headers=_hdr(ADMIN),
                               json={"automation": {"weekly_estimate": "sometimes"}})
            denied = await cl.put("/fd/admin/course/settings", headers=_hdr(BOB),
                                  json={"automation": {"conflict_check": False}})
        assert put.status_code == 200 and put.json()["automation"] == {
            "weekly_estimate": True, "conflict_check": True, "goal_driven_candidates": True}
        assert got.json()["defaults"]["automation"]["goal_driven_candidates"] is False
        assert bad.status_code == 400 and denied.status_code == 403
