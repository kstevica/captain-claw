"""A strong model plans and checks; a cheaper one works the steps.

Bat planned and its workers worked on the same tier (reasoning, else fast), so a
stronger model on the reasoning tier would have done the searching too, at its
price. The deck's "workers use" setting names the tier a step's worker runs on.
Pinned here:

* the setting is one of the tier names, or empty for the planning tier; it's kept
  and in force at once, and the console sees the admin's own model per tier;
* a step's worker runs on that tier when the owner's set has a model there, else on
  the planning tier; the first step, which writes the conventions, on the planning
  tier; a run may name its own;
* the reviewers pick a model the workers don't run on;
* the planner is told when a smaller model works the steps, and a note says which
  model does what.
"""

from __future__ import annotations

import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import bat_routes
from captain_claw.flight_deck.auth import set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

TIERS = {
    "reason": {"provider": "anthropic", "model": "big-model"},
    "fast": {"provider": "openai", "model": "small-model"},
    "balanced": {"provider": "openai", "model": "small-model"},
}


@pytest.fixture
async def deck_db(tmp_path):
    prev = fd_auth._db
    d = FlightDeckDB(tmp_path / "fd.db")
    await d.init()
    set_auth_db(d)
    try:
        yield d
    finally:
        bat_routes._apply_settings({})
        await d.close()
        fd_auth._db = prev


@pytest.fixture(autouse=True)
def _clean():
    bat_routes._apply_settings({})
    yield
    bat_routes._apply_settings({})


async def test_the_setting_is_a_tier_or_the_planning_one(deck_db, monkeypatch):
    assert bat_routes.deck_settings()["worker_tier"] == ""
    got = await bat_routes.save_settings({"worker_tier": "fast"})
    assert got["worker_tier"] == "fast" and bat_routes.deck_settings()["worker_tier"] == "fast"
    for bad in ({"worker_tier": "huge"}, {"worker_tier": 3}):
        with pytest.raises(ValueError):
            await bat_routes.save_settings(bad)
    assert (await bat_routes.save_settings({"worker_tier": ""}))["worker_tier"] == ""
    # kept: read again at start
    await bat_routes.save_settings({"worker_tier": "fast", "call_minutes": 30})
    bat_routes._apply_settings({})
    assert (await bat_routes.load_settings())["worker_tier"] == "fast"

    async def tiers(_db, _owner):
        return TIERS, []

    from captain_claw.flight_deck import basna_routes
    monkeypatch.setattr(basna_routes, "_load_owner_tiers", tiers)
    view = await bat_routes.get_deck_settings(admin={"id": "admin", "role": "admin"})
    assert view["tiers"] == list(bat_routes.TIER_NAMES)
    assert view["your_tiers"] == {"reason": "big-model", "balanced": "small-model", "fast": "small-model"}


def test_who_works_a_step():
    step = {"step_key": "step-3", "title": "Find 8 firms", "seq": 2, "round": 0}
    conventions = {"step_key": "step-1", "title": "Write shared/conventions.md for the list", "seq": 0, "round": 0}
    run = {"config": {}}
    # not set: everything on the planning tier, as before
    assert bat_routes.step_tier(run, step, TIERS) == "reason"
    bat_routes._apply_settings({"worker_tier": "fast"})
    assert bat_routes.step_tier(run, step, TIERS) == "fast"
    assert bat_routes.step_tier(run, conventions, TIERS) == "reason"
    # a fix round's first step isn't the conventions
    assert bat_routes.step_tier(run, {**conventions, "round": 1}, TIERS) == "fast"
    # a tier the owner's set leaves empty falls back to the planning one
    bat_routes._apply_settings({"worker_tier": "micro"})
    assert bat_routes.step_tier(run, step, TIERS) == "reason"
    # a run may name its own
    assert bat_routes.worker_tier(TIERS, {"config": {"worker_tier": "balanced"}}) == "balanced"


def test_the_reviewers_use_a_model_the_workers_dont():
    creds, independent = bat_routes._panel_creds(TIERS)
    assert not independent or creds["model"] != "big-model"  # workers on reason: reviewers elsewhere
    bat_routes._apply_settings({"worker_tier": "fast"})
    creds, independent = bat_routes._panel_creds(TIERS)
    assert independent and creds["model"] == "big-model"


def test_the_planner_is_told_when_a_smaller_model_works():
    assert bat_routes.workers_note(TIERS) == ""
    bat_routes._apply_settings({"worker_tier": "fast"})
    note = bat_routes.workers_note(TIERS)
    assert "smaller model (small-model)" in note and "what counts as proof" in note
    one_model = {k: {"provider": "openai", "model": "same"} for k in ("reason", "fast")}
    assert bat_routes.workers_note(one_model) == ""


async def test_the_worker_spawns_on_the_workers_tier(monkeypatch):
    """The attempt hands the worker the chosen tier (the spawn is stopped right there)."""
    seen = {}

    class Stop(Exception):
        pass

    async def spawn(stub, user, *, name, tier, tiers, env_vars, extra_env):
        seen["tier"] = tier
        raise Stop()

    async def tiers(_db, _owner):
        return TIERS, []

    class Db:
        async def get_user_by_id(self, _id):
            return {"id": "u1", "role": "user"}

    from captain_claw.flight_deck import auth, basna_routes
    monkeypatch.setattr(basna_routes, "_load_owner_tiers", tiers)
    monkeypatch.setattr(auth, "get_db", lambda: Db())
    monkeypatch.setattr(bat_routes, "_spawn_bat_worker", spawn)
    bat_routes._apply_settings({"worker_tier": "fast"})
    run = {"id": "bat_tier0001", "owner_id": "u1", "config": {}, "task": "t"}
    with pytest.raises(Stop):
        await bat_routes._bat_attempt(run, {"step_key": "step-2", "title": "Find 8 firms", "seq": 1, "round": 0})
    assert seen["tier"] == "fast"
    with pytest.raises(Stop):
        await bat_routes._bat_attempt(run, {"step_key": "step-1", "title": "Write shared/conventions.md",
                                            "seq": 0, "round": 0})
    assert seen["tier"] == "reason"
