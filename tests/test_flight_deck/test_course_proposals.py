"""Course, Phase 3: goals the agents and the deck suggest.

Pinned here:

* a suggestion is a ``proposed`` goal, never active until a person accepts it;
  only user and agent scopes; the admin's switch turns suggestions off;
* the rules: a weekly limit per suggester (per owner for the deck), at most
  five waiting per scope, no title already waiting or on the course, none
  declined in the last 90 days;
* a suggested change is a new row that replaces its goal when accepted: the old
  one superseded, its children, evidence and progress moved;
* a proposal only becomes active through ``accept``;
* submit: the owner's Line is asked — or, when the owner let agents add their
  own short-term goals, it's added at once and the Line just says so;
* the agent's self-reflection takes a ``GOAL_PROPOSAL`` line off the
  reflection; the deck's heartbeat suggests at most one user goal.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from captain_claw.flight_deck import course as c

BOB, CAROL = "user-bob", "user-carol"
REF_A = "process:ana:" + "a" * 16
REF_B = "process:ben:" + "b" * 16


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = c.CourseStore(tmp_path / "course.db")
    monkeypatch.setattr(c, "_STORE", s)
    return s


def _suggest(store, title="Clear the backlog", *, by=REF_A, scope="agent", horizon="short",
             owner=BOB, origin="agent", **kw):
    kw.setdefault("success", "zero open")
    agent_ref = (kw.pop("agent_ref", by) if scope == "agent" else "")
    return store.propose(scope=scope, horizon=horizon, title=title, owner_id=owner, origin=origin,
                         proposed_by=by, agent_ref=agent_ref, **kw)


def _goal(store, title="Ship EU pricing", horizon="long", **kw):
    kw.setdefault("success", "three EU customers")
    return store.create(scope="user", owner_id=BOB, horizon=horizon, title=title, **kw)


# ── proposing ────────────────────────────────────────────────────────


def test_a_suggestion_waits(store):
    p = _suggest(store, rationale="I keep falling behind")
    assert (p["status"], p["origin"], p["proposed_by"], p["review_at"]) == ("proposed", "agent", REF_A, None)
    assert p["rationale"] == "I keep falling behind"
    with pytest.raises(c.CourseError, match="Only a person"):
        store.propose(scope="instance", horizon="long", title="x", owner_id="", origin="deck",
                      proposed_by="deck", success="s")
    with pytest.raises(c.CourseError, match="Unknown origin"):
        _suggest(store, "Other", origin="user")


def test_the_admin_can_switch_suggestions_off(store):
    store.set_settings(proposals={"recommendations": False})
    with pytest.raises(c.CourseError, match="switched off"):
        _suggest(store)
    assert store.settings()["proposals"]["agent_per_week"] == 2     # a partial save keeps the rest


def test_weekly_limit_per_suggester(store):
    store.set_settings(proposals={"agent_per_week": 2, "deck_per_week": 1})
    _suggest(store, "One")
    _suggest(store, "Two", scope="user", horizon="medium")          # the same agent, for its owner
    with pytest.raises(c.CourseError, match="weekly limit of 2"):
        _suggest(store, "Three")
    assert _suggest(store, "Ben's", by=REF_B)                        # another agent has its own
    _suggest(store, "Deck one", by="deck", origin="deck", scope="user", horizon="medium")
    with pytest.raises(c.CourseError, match="weekly limit of 1 suggestion "):
        _suggest(store, "Deck two", by="deck", origin="deck", scope="user", horizon="medium")
    assert _suggest(store, "For Carol", by="deck", origin="deck", scope="user", horizon="medium",
                    owner=CAROL)                                    # per person for the deck


def test_at_most_five_waiting_per_scope(store):
    store.set_settings(proposals={"agent_per_week": 20})
    for i in range(5):
        _suggest(store, f"Goal {i}")
    with pytest.raises(c.CourseError, match="5 suggestions are already waiting"):
        _suggest(store, "Goal 5")
    assert _suggest(store, "For the owner", scope="user", horizon="medium")   # another scope


def test_duplicates_and_the_decline_memory(store):
    store.set_settings(proposals={"agent_per_week": 20})
    p = _suggest(store, "Clear the backlog")
    with pytest.raises(c.CourseError, match="already waiting"):
        _suggest(store, "  clear THE backlog! ")
    store.set_status(p["id"], "declined")
    with pytest.raises(c.CourseError, match="declined recently"):
        _suggest(store, "Clear the backlog")
    old = (datetime.now(UTC) - timedelta(days=c.DECLINE_MEMORY_DAYS + 1)).isoformat()
    store._c().execute("UPDATE course_goals SET decided_at = ? WHERE id = ?", (old, p["id"]))
    store._c().commit()
    again = _suggest(store, "Clear the backlog")
    store.accept(again["id"])
    with pytest.raises(c.CourseError, match="already on the course"):
        _suggest(store, "Clear the backlog")


# ── accepting, declining, changing ───────────────────────────────────


def test_accept_and_decline(store):
    p = _suggest(store)
    with pytest.raises(c.CourseError, match="can't become active"):
        store.set_status(p["id"], "active")                          # only accept() activates
    g = store.accept(p["id"], actor="user:bob")
    assert g["status"] == "active" and g["review_at"] and g["decided_by"] == "user:bob"
    assert store.events(p["id"])[0]["event"] == "accepted"
    with pytest.raises(c.CourseError, match="not a proposal"):
        store.accept(p["id"])
    q = _suggest(store, "Another")
    assert store.set_status(q["id"], "declined")["status"] == "declined"


def test_a_change_replaces_its_goal(store):
    parent = _goal(store, "Ship EU pricing")
    child = _goal(store, "Draft the price page", horizon="short", parent_id=parent["id"])
    store.add_link(parent["id"], "project", "p1", owner_id=BOB)
    store.set_progress(parent["id"], 0.3, "a third")
    change = store.propose_change(parent["id"], {"title": "Ship EU and UK pricing",
                                                 "success": "five customers"},
                                  origin="agent", proposed_by=REF_A, rationale="UK asked too")
    assert change["status"] == "proposed" and change["supersedes_id"] == parent["id"]
    assert change["horizon"] == "long" and change["review_every_days"] == parent["review_every_days"]
    with pytest.raises(c.CourseError, match="already waiting"):
        store.propose_change(parent["id"], {"title": "Something else"}, origin="agent", proposed_by=REF_B)
    new = store.accept(change["id"])
    assert new["status"] == "active" and new["progress"] == 0.3 and new["progress_note"] == "a third"
    assert store.get(parent["id"])["status"] == "superseded"
    assert store.get(child["id"])["parent_id"] == new["id"]
    assert [lk["ref_id"] for lk in store.links(new["id"])] == ["p1"]


def test_change_rules(store):
    g = _goal(store)
    deck = store.create(scope="instance", horizon="long", title="Deck", success="s")
    for changes, msg in (({}, "Say what should change"), ({"status": "achieved"}, "can't be changed"),
                         ({"title": g["title"]}, "changes nothing")):
        with pytest.raises(c.CourseError, match=msg):
            store.propose_change(g["id"], changes, origin="agent", proposed_by=REF_A)
    with pytest.raises(c.CourseError, match="Only an admin"):
        store.propose_change(deck["id"], {"title": "Mine"}, origin="agent", proposed_by=REF_A)
    _goal(store, "Child", horizon="long", parent_id=g["id"])
    with pytest.raises(c.CourseError, match="longer horizon"):
        store.propose_change(g["id"], {"horizon": "short"}, origin="agent", proposed_by=REF_A)


def test_a_change_to_a_goal_that_moved_on_is_refused(store):
    g = _goal(store)
    change = store.propose_change(g["id"], {"title": "New"}, origin="deck", proposed_by="deck")
    store.set_status(g["id"], "achieved")
    with pytest.raises(c.CourseError, match="no longer active"):
        store.accept(change["id"])


def test_accepting_a_change_doesnt_count_its_goal_twice(store):
    store.set_settings(caps={"long": 1})
    g = _goal(store)
    change = store.propose_change(g["id"], {"title": "Renamed"}, origin="agent", proposed_by=REF_A)
    assert store.accept(change["id"])["status"] == "active"


def test_prefs(store):
    assert store.prefs(BOB) == {"auto_accept_agent_short": False}
    assert store.set_prefs(BOB, {"auto_accept_agent_short": True}) == {"auto_accept_agent_short": True}
    assert store.prefs(CAROL)["auto_accept_agent_short"] is False
    with pytest.raises(c.CourseError):
        store.set_prefs(BOB, {"auto_accept_agent_short": "yes"})
    with pytest.raises(c.CourseError):
        store.set_settings(proposals={"agent_per_week": 99})


# ── submit: the Line, or straight on ─────────────────────────────────


@pytest.fixture
def posted(monkeypatch):
    from captain_claw.flight_deck import course_carrier, line

    seen: list = []

    async def post(user_id, kind, title, body="", **kw):
        seen.append({"user": user_id, "kind": kind, "title": title, "body": body, **kw})
        return "item-1"

    async def refresh(db, goal):
        seen.append({"refresh": goal["id"]})
        return 1

    monkeypatch.setattr(line, "post", post)
    monkeypatch.setattr(course_carrier, "refresh_for_goal", refresh)
    monkeypatch.setattr("captain_claw.flight_deck.auth.get_db", lambda: None)
    return seen


async def test_submit_asks_the_owner(store, posted):
    from captain_claw.flight_deck import course_proposals as cp

    out = await cp.submit(owner_id=BOB, origin="agent", proposed_by=REF_A, proposer="Ana",
                          scope="agent", agent_ref=REF_A, horizon="medium", title="Benchmarks",
                          success="a table", rationale="asked twice")
    assert out["auto_accepted"] is False and out["goal"]["status"] == "proposed"
    item = posted[0]
    assert (item["user"], item["kind"], item["bell_type"], item["ref_type"], item["ref_id"], item["trust"]) == (
        BOB, "proposal", "course_proposal", "course_goal", out["goal"]["id"], "agent")
    assert item["title"] == "Goal proposed: Benchmarks"
    assert "Ana suggests a goal for an agent's course." in item["body"] and "Why: asked twice" in item["body"]


async def test_a_suggested_change_says_what_it_changes(store, posted):
    from captain_claw.flight_deck import course_proposals as cp

    g = _goal(store)
    await cp.submit(owner_id=BOB, origin="deck", proposed_by="deck", proposer="Flight Deck",
                    change_of=g["id"], changes={"title": "Ship EU and UK pricing"})
    item = posted[0]
    assert item["title"] == "Change proposed: Ship EU and UK pricing" and item["trust"] == "deck"
    assert "It changes: title (was: Ship EU pricing)." in item["body"]


async def test_auto_accept_only_an_agents_own_short_goal(store, posted):
    from captain_claw.flight_deck import course_proposals as cp

    store.set_prefs(BOB, {"auto_accept_agent_short": True})
    store.set_settings(proposals={"agent_per_week": 20})
    own = await cp.submit(owner_id=BOB, origin="agent", proposed_by=REF_A, proposer="Ana",
                          scope="agent", agent_ref=REF_A, horizon="short", title="Inbox zero",
                          success="zero")
    assert own["auto_accepted"] is True and own["goal"]["status"] == "active"
    assert posted[0]["bell_type"] == "course_auto_added" and posted[0]["title"] == "Added to Ana's course: Inbox zero"
    assert posted[1] == {"refresh": own["goal"]["id"]}
    medium = await cp.submit(owner_id=BOB, origin="agent", proposed_by=REF_A, proposer="Ana",
                             scope="agent", agent_ref=REF_A, horizon="medium", title="Benchmarks",
                             success="table")
    for_owner = await cp.submit(owner_id=BOB, origin="agent", proposed_by=REF_A, proposer="Ana",
                                scope="user", horizon="short", title="Call the accountant",
                                success="called")
    assert medium["auto_accepted"] is False and for_owner["auto_accepted"] is False


# ── the agent's reflection and the deck's heartbeat ──────────────────


def test_reflection_takes_the_goal_line_off():
    from captain_claw import reflections as r

    text = ('Be terser.\nGOAL_PROPOSAL: {"horizon": "short", "title": "Clear the backlog", '
            '"success": "zero open", "rationale": "it grew", "parent_id": "G:goal_a"}\n')
    summary, found = r.take_goal_proposal(text)
    assert summary == "Be terser."
    assert found == {"horizon": "short", "title": "Clear the backlog", "success": "zero open",
                     "rationale": "it grew", "parent_id": "G:goal_a"}
    for bad in ('GOAL_PROPOSAL: {"horizon": "long", "title": "x"}', 'GOAL_PROPOSAL: {"horizon": "short"}',
                "GOAL_PROPOSAL: not json"):
        summary, found = r.take_goal_proposal(f"Keep going.\n{bad}")
        assert found is None and summary == "Keep going."               # the line never stays


def test_reflection_asks_only_under_flight_deck(monkeypatch):
    from captain_claw import course_client, reflections

    monkeypatch.setattr(course_client, "usable", lambda: False)
    assert reflections.course_ask() == ""
    monkeypatch.setattr(course_client, "usable", lambda: True)
    monkeypatch.setattr("captain_claw.tenant_context.load_course_context", lambda compact: "## Your course\nG")
    ask = reflections.course_ask()
    assert "GOAL_PROPOSAL:" in ask and "## Your course\nG" in ask and "Your owner decides" in ask


def test_heartbeat_prompt_lists_the_course():
    from captain_claw.flight_deck import consciousness as cs

    delta = {"new_messages": 3, "new_sessions": 1, "reachable": 2}
    prompt = cs._build_reflection_user_prompt(delta, "", [], course="- G:goal_a (long-term): Ship")
    assert "The user's Course (their goals):\n- G:goal_a (long-term): Ship" in prompt
    assert "(none yet)" in cs._build_reflection_user_prompt(delta, "", [])
    assert '"course": null|{' in cs._SYSTEM_PROMPT
    parsed = cs._parse_reflection('{"thought": "t", "course": {"horizon": "long", "title": "Open a UK office"}}')
    assert parsed["course"]["title"] == "Open a UK office"


async def test_heartbeat_suggestion(store, posted):
    from captain_claw.flight_deck import consciousness as cs

    g = await cs._suggest_goal(BOB, {"horizon": "long", "title": "Open a UK office",
                                     "success": "a signed lease", "why": "three UK leads"})
    assert g["status"] == "proposed" and g["origin"] == "deck" and g["scope"] == "user"
    assert g["rationale"] == "three UK leads" and posted[0]["source"] == "Flight Deck"
    assert await cs._suggest_goal(BOB, {"horizon": "strategic", "title": "x"}) is None
    assert await cs._suggest_goal(BOB, None) is None
    assert await cs._suggest_goal(BOB, {"horizon": "long", "title": "Open a UK office"}) is None  # waiting already
