"""Course store: the rules every goal follows, whoever sends it.

Pinned here:

* scopes and the admin's horizons: a horizon a scope doesn't use is refused for
  new goals, kept for existing ones, and blocks reactivation until moved;
* the tree: a goal serves its own scope or a wider one, the same owner's or
  agent's, a horizon no shorter than its own, an active parent, and no cycle;
* caps per horizon count active goals only; paused goals don't count;
* text caps and the success requirement (strategic goals are exempt);
* the lifecycle, the review clock, and when a goal is due and stale;
* forgetting a user or an agent.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from captain_claw.flight_deck import course as c

BOB, CAROL = "user-bob", "user-carol"
REF_B = "process:bob-agent:" + "a" * 16
REF_B2 = "process:bob-two:" + "b" * 16


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = c.CourseStore(tmp_path / "course.db")
    monkeypatch.setattr(c, "_STORE", s)
    return s


def _user(store, horizon="long", owner=BOB, **kw):
    kw.setdefault("title", f"{horizon} goal")
    kw.setdefault("success", "measurably done")
    return store.create(scope="user", owner_id=owner, horizon=horizon, **kw)


def _deck(store, horizon="strategic", **kw):
    kw.setdefault("title", f"deck {horizon}")
    kw.setdefault("success", "done" if horizon != "strategic" else "")
    return store.create(scope="instance", horizon=horizon, origin="admin", **kw)


# ── settings ─────────────────────────────────────────────────────────


def test_defaults(store):
    s = store.settings()
    assert s["horizons"] == {"instance": ["strategic", "long"],
                             "user": ["strategic", "long", "medium", "short"],
                             "agent": ["medium", "short"]}
    assert s["caps"] == {"strategic": 3, "long": 5, "medium": 7, "short": 3}


def test_settings_are_validated_and_canonical(store):
    s = store.set_settings(horizons={"instance": ["long", "strategic", "long"], "agent": []})
    assert s["horizons"]["instance"] == ["strategic", "long"]
    assert s["horizons"]["agent"] == []
    assert s["horizons"]["user"] == list(c.HORIZONS)       # absent → default
    with pytest.raises(c.CourseError):
        store.set_settings(horizons={"user": ["forever"]})
    with pytest.raises(c.CourseError):
        store.set_settings(caps={"short": 0})
    with pytest.raises(c.CourseError):
        store.set_settings(caps={"short": True})
    assert store.set_settings(caps={"short": 5})["caps"]["short"] == 5


def test_a_scope_uses_only_the_admins_horizons(store):
    with pytest.raises(c.CourseError, match="aren't used at the instance level"):
        _deck(store, "short")
    with pytest.raises(c.CourseError):
        store.create(scope="agent", owner_id=BOB, agent_ref=REF_B, horizon="strategic",
                     title="role-like")
    store.set_settings(horizons={"agent": ["strategic", "medium", "short"]})
    assert store.create(scope="agent", owner_id=BOB, agent_ref=REF_B, horizon="strategic",
                        title="now allowed")["status"] == "active"


def test_retiring_a_horizon_keeps_existing_goals(store):
    g = _user(store, "short")
    store.set_settings(horizons={"user": ["strategic", "long", "medium"]})
    assert store.get(g["id"])["status"] == "active"        # kept, still active
    with pytest.raises(c.CourseError):
        _user(store, "short", title="a new one")
    store.set_status(g["id"], "paused")
    with pytest.raises(c.CourseError, match="change its horizon first"):
        store.set_status(g["id"], "active")
    store.update(g["id"], {"horizon": "medium"})
    assert store.set_status(g["id"], "active")["status"] == "active"


# ── scopes and text ──────────────────────────────────────────────────


def test_instance_goals_have_no_owner(store):
    g = store.create(scope="instance", horizon="long", owner_id=BOB, agent_ref=REF_B,
                     title="t", success="s")
    assert (g["owner_id"], g["agent_ref"]) == ("", "")


def test_agent_goals_need_their_agent(store):
    with pytest.raises(c.CourseError, match="needs its agent"):
        store.create(scope="agent", owner_id=BOB, horizon="short", title="t", success="s")


def test_empty_owner_is_local(store):
    assert _user(store, owner="")["owner_id"] == "local"


def test_text_rules(store):
    with pytest.raises(c.CourseError, match="title is required"):
        _user(store, title="   ")
    with pytest.raises(c.CourseError, match="too long"):
        _user(store, title="x" * 121)
    with pytest.raises(c.CourseError, match="success"):
        _user(store, success="")
    with pytest.raises(c.CourseError, match="valid text"):
        _user(store, title="bad \ud800")
    with pytest.raises(c.CourseError, match="must be text"):
        _user(store, title=5)
    assert _user(store, "strategic", success="")["success"] == ""   # strategic: optional


def test_unknown_values_are_refused(store):
    with pytest.raises(c.CourseError):
        store.create(scope="team", horizon="long", title="t", success="s")
    with pytest.raises(c.CourseError):
        _user(store, "forever")
    with pytest.raises(c.CourseError):
        _user(store, due_at="not a date")
    with pytest.raises(c.CourseError):
        _user(store, review_every_days=0)


# ── the tree ─────────────────────────────────────────────────────────


def test_a_goal_serves_its_own_scope_or_a_wider_one(store):
    deck = _deck(store, "strategic")
    mine = _user(store, "long", parent_id=deck["id"])
    agent = store.create(scope="agent", owner_id=BOB, agent_ref=REF_B, horizon="medium",
                         title="a", success="s", parent_id=mine["id"])
    assert agent["parent_id"] == mine["id"]
    with pytest.raises(c.CourseError, match="can't serve a user goal"):
        _deck(store, "long", parent_id=mine["id"])
    with pytest.raises(c.CourseError, match="can't serve a agent goal"):
        _user(store, "short", parent_id=agent["id"])


def test_a_goal_never_serves_someone_elses(store):
    carols = _user(store, "strategic", owner=CAROL)
    with pytest.raises(c.CourseError, match="doesn't exist"):
        _user(store, "long", parent_id=carols["id"])
    other_agent = store.create(scope="agent", owner_id=BOB, agent_ref=REF_B2,
                               horizon="medium", title="x", success="s")
    with pytest.raises(c.CourseError, match="doesn't exist"):
        store.create(scope="agent", owner_id=BOB, agent_ref=REF_B, horizon="short",
                     title="y", success="s", parent_id=other_agent["id"])


def test_a_goal_never_serves_a_shorter_horizon(store):
    short = _user(store, "short")
    with pytest.raises(c.CourseError, match="same horizon or a longer one"):
        _user(store, "long", parent_id=short["id"])
    assert _user(store, "short", parent_id=short["id"], title="same horizon")


def test_parent_must_be_active(store):
    p = _user(store, "long")
    store.set_status(p["id"], "achieved")
    with pytest.raises(c.CourseError, match="isn't active"):
        _user(store, "short", parent_id=p["id"])


def test_no_cycles(store):
    a = _user(store, "long", title="a")
    b = _user(store, "long", title="b", parent_id=a["id"])
    with pytest.raises(c.CourseError, match="can't serve itself"):
        store.update(a["id"], {"parent_id": b["id"]})
    with pytest.raises(c.CourseError, match="can't serve itself"):
        store.update(a["id"], {"parent_id": a["id"]})


def test_horizon_change_keeps_children_valid(store):
    p = _user(store, "long")
    _user(store, "long", title="child", parent_id=p["id"])
    with pytest.raises(c.CourseError, match="move them first"):
        store.update(p["id"], {"horizon": "short"})


# ── caps ─────────────────────────────────────────────────────────────


def test_caps_count_active_goals_per_level(store):
    store.set_settings(caps={"short": 2})
    a = _user(store, "short", title="1")
    _user(store, "short", title="2")
    with pytest.raises(c.CourseError, match="most it may hold"):
        _user(store, "short", title="3")
    assert _user(store, "short", owner=CAROL, title="carol's own level")
    store.set_status(a["id"], "paused")
    c3 = _user(store, "short", title="3")
    with pytest.raises(c.CourseError, match="most it may hold"):
        store.set_status(a["id"], "active")
    store.set_status(c3["id"], "dropped")
    assert store.set_status(a["id"], "active")["status"] == "active"


def test_proposals_dont_count_toward_caps(store):
    store.set_settings(caps={"short": 1})
    _user(store, "short", title="1")
    p = store.create(scope="user", owner_id=BOB, horizon="short", title="idea",
                     success="s", status="proposed", origin="agent", proposed_by=REF_B)
    assert p["status"] == "proposed" and p["review_at"] is None
    with pytest.raises(c.CourseError, match="most it may hold"):
        store.set_status(p["id"], "active")


# ── lifecycle and review ─────────────────────────────────────────────


def test_lifecycle(store):
    g = _user(store, "medium")
    with pytest.raises(c.CourseError, match="can't become declined"):
        store.set_status(g["id"], "declined")
    store.set_status(g["id"], "achieved")
    store.set_status(g["id"], "active")              # reopened
    store.set_status(g["id"], "dropped")
    assert [e["event"] for e in store.events(g["id"])][-1] == "created"
    assert store.events(g["id"])[0]["detail"] == {"from": "active", "to": "dropped"}
    with pytest.raises(c.CourseNotFoundError):
        store.set_status("goal_nope", "paused")


def test_review_clock(store):
    g = _user(store, "short")
    now = datetime.now(UTC)
    assert g["review_every_days"] == 7
    at = datetime.fromisoformat(g["review_at"])
    assert timedelta(days=6, hours=23) < at - now < timedelta(days=7, minutes=1)
    assert not c.is_due(g, now) and not c.is_stale(g, now)
    assert c.is_due(g, at + timedelta(seconds=1))
    assert not c.is_stale(g, at + timedelta(days=2))          # grace: max(3, 7 // 4) = 3 days
    assert c.is_stale(g, at + timedelta(days=3, seconds=1))
    assert [x["id"] for x in store.due_for_review(at + timedelta(seconds=1))] == [g["id"]]
    reviewed = store.review(g["id"])
    assert datetime.fromisoformat(reviewed["review_at"]) > at
    assert [e["event"] for e in store.events(g["id"])][0] == "reviewed"


def test_grace_scales_with_the_review_period():
    assert c.grace_days(7) == 3
    assert c.grace_days(90) == 22
    assert c.grace_days(365) == 91


def test_paused_goals_are_never_due(store):
    g = _user(store, "short")
    store.set_status(g["id"], "paused")
    later = datetime.now(UTC) + timedelta(days=100)
    assert not c.is_due(store.get(g["id"]), later)
    assert store.due_for_review(later) == []
    with pytest.raises(c.CourseError, match="Only an active goal"):
        store.review(g["id"])


def test_edit_review_period_restarts_the_clock(store):
    g = _user(store, "long")
    g2 = store.update(g["id"], {"review_every_days": 10})
    at = datetime.fromisoformat(g2["review_at"])
    assert at - datetime.now(UTC) < timedelta(days=10, minutes=1)


def test_unknown_edit_fields_are_refused(store):
    g = _user(store)
    with pytest.raises(c.CourseError, match="can't be edited"):
        store.update(g["id"], {"owner_id": CAROL})
    with pytest.raises(c.CourseError, match="can't be edited"):
        store.update(g["id"], {"status": "achieved"})


# ── links and forgetting ─────────────────────────────────────────────


def test_links(store):
    g = _user(store)
    store.add_link(g["id"], "task", "t1", note="first")
    store.add_link(g["id"], "task", "t1", note="again")       # upsert
    assert [(lk["ref_id"], lk["note"]) for lk in store.links(g["id"])] == [("t1", "again")]
    with pytest.raises(c.CourseError):
        store.add_link(g["id"], "rumour", "x")
    assert store.remove_link(g["id"], "task", "t1") is True
    assert store.links(g["id"]) == []


def test_forget_user_and_agent(store):
    deck = _deck(store)
    mine = _user(store)
    store.add_link(mine["id"], "task", "t1")
    ag = store.create(scope="agent", owner_id=BOB, agent_ref=REF_B, horizon="short",
                      title="a", success="s")
    carols = _user(store, owner=CAROL)
    assert store.forget_agent(REF_B) == 1
    assert store.get(ag["id"]) is None and store.get(mine["id"])
    assert store.forget_user(BOB) == 1
    assert store.get(mine["id"]) is None and store.links(mine["id"]) == []
    assert store.get(deck["id"]) and store.get(carols["id"])
    assert c.forget_user("") == 0 and c.forget_agent("") == 0
