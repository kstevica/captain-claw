"""Bat's own model calls get time and room for a thinking model.

A thinking model on one machine (~90 tokens a second) thought for 15.7k tokens
while Bat shaped a factory's checks: past the provider's own 180 s, so the call was
cut off and asked again from the start, and the run sat in planning. Pinned here:
Bat's calls (planning, its checks, the reviewers, the step check's review) carry
the admin's time limit (20 minutes by default, 3 to 120) and may write 32k tokens,
or the tier's own limit when that is lower; First Mate's drafts take the same limit.
"""

from __future__ import annotations

import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import basna_routes, bat_routes
from captain_claw.flight_deck.auth import set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB


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


def test_bats_calls_get_time_and_room(monkeypatch):
    seen = {}

    def provider(**kw):
        seen.update(kw)

        class P:
            request_timeout = None
        return P()

    monkeypatch.setattr("captain_claw.llm.create_provider", provider)
    prov, mt = bat_routes.think_call({"provider": "openai", "model": "m", "output_ctx": 65_536}, temperature=0.0)
    assert mt == bat_routes.CALL_TOKENS == 32_768 and prov.request_timeout == 20 * 60.0
    _, small = bat_routes.think_call({"provider": "openai", "model": "m", "output_ctx": 8_000}, temperature=0.0)
    assert small == 16_384  # never under the default, whatever the tier says
    assert seen["max_tokens"] == 16_384


async def test_the_admin_sets_the_time_limit(deck_db):
    assert bat_routes.deck_settings()["call_minutes"] == 20
    got = await bat_routes.save_settings({"call_minutes": 45})
    assert got["call_minutes"] == 45 and bat_routes.call_timeout() == 45 * 60.0
    for bad in ({"call_minutes": 2}, {"call_minutes": 500}):
        with pytest.raises(ValueError):
            await bat_routes.save_settings(bad)
    view = await bat_routes.get_deck_settings(admin={"id": "admin", "role": "admin"})
    assert view["ranges"]["call_minutes"] == [3, 120] and view["defaults"]["call_minutes"] == 20


def test_no_bat_call_keeps_the_old_limits():
    import inspect
    from captain_claw.flight_deck import bat_course
    src = inspect.getsource(bat_routes) + inspect.getsource(bat_course)
    assert "cap=16384" not in src and "cap=8192" not in src
    assert basna_routes._provider_call  # what think_call builds on
