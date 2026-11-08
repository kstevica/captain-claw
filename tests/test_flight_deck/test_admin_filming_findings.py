"""Fixes found while planning the admin tutorial video (docs/tutorial/admin-video-plan.md).

* F1 — a deck without sign-in (the desktop app's) goes by its admin's settings
  (branding, Docker …), all but the ones about other people.
* F6 — /fd/auth/status says when the deck has no account yet.
* F4 — deleting a user stops and removes their agents.
* F11 — a reset gives the day's allowance back without undoing what was typed.
* F3 — limit messages ask the admin; nobody can upgrade their own plan.
* F10 — the users list says whether Crew is offered; the update page knows its countdown.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from captain_claw.flight_deck import admin_routes, rate_limiter
from captain_claw.flight_deck.auth import get_current_user, hash_password, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

LOOPBACK = ("127.0.0.1", 40001)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LOOPBACK), base_url="http://fd.test")


@pytest.fixture
async def db(tmp_path: Path):
    from captain_claw.flight_deck import auth as fd_auth
    prev = fd_auth._db
    d = FlightDeckDB(tmp_path / "fd.db")
    await d.init()
    set_auth_db(d)
    try:
        yield d
    finally:
        await d.close()
        fd_auth._db = prev


def _status_app(server) -> FastAPI:
    app = FastAPI()
    app.add_api_route("/fd/auth/status", server.auth_status)
    return app


# ── F1 ──

async def test_a_deck_without_sign_in_keeps_its_admins_settings(db, monkeypatch):
    from captain_claw.flight_deck import server

    await db.set_system_setting("fd:system-config", json.dumps({
        "instance_name": "Northwind", "instance_description": "Our agents",
        "docker_spawn_enabled": False, "kiosk_mode": "all", "waitlist_enabled": False,
    }))
    monkeypatch.setattr(server, "AUTH_ENABLED", False)
    monkeypatch.setattr(server.app.state, "fd_db", db, raising=False)

    cfg = await server._get_system_config()
    assert cfg["instance_name"] == "Northwind" and cfg["docker_spawn_enabled"] is False
    # …but not the ones about other people: there is no one else.
    defaults = admin_routes._load_system_config(None)
    assert cfg["kiosk_mode"] == defaults["kiosk_mode"]
    assert cfg["waitlist_enabled"] == defaults["waitlist_enabled"]

    async with _client(_status_app(server)) as c:
        st = (await c.get("/fd/auth/status")).json()
    assert st["instance_name"] == "Northwind" and st["instance_description"] == "Our agents"
    assert st["docker_spawn_enabled"] is False
    assert st["simple_chat_only"] is (defaults["kiosk_mode"] == "all")
    assert st["first_run"] is False and st["registration_open"] is False


# ── F6 ──

async def test_status_says_when_the_deck_has_no_account_yet(db, monkeypatch):
    from captain_claw.flight_deck import server

    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setattr(server.app.state, "fd_db", db, raising=False)
    async with _client(_status_app(server)) as c:
        st = (await c.get("/fd/auth/status")).json()
        assert st["first_run"] is True and st["registration_open"] is True
        await db.create_user("lena@example.com", hash_password("pw-123456"), "Lena", "admin")
        st = (await c.get("/fd/auth/status")).json()
        assert st["first_run"] is False


# ── F4 ──

async def test_deleting_a_user_removes_their_agents(db, monkeypatch):
    from captain_claw.flight_deck import server

    admin = await db.create_user("lena@example.com", hash_password("pw-123456"), "Lena", "admin")
    maya = await db.create_user("maya@example.com", hash_password("pw-123456"), "Maya", "user")
    asked: list[str] = []

    async def fake_remove(owner_id: str) -> list[str]:
        asked.append(owner_id)
        return ["inbox-assistant"]

    monkeypatch.setattr(server, "remove_agents_of", fake_remove)
    # The route's other clean-ups use deck-wide stores: kept out of a test.
    from captain_claw.flight_deck import (
        context_packs, course, crew_runner, payments_routes, profile_sources, taskboard, tenant_profile,
    )

    async def anull(*_a, **_kw):
        return None

    monkeypatch.setattr(payments_routes, "forget_user", anull)
    monkeypatch.setattr(taskboard, "forget_user", lambda *_a, **_kw: None)
    monkeypatch.setattr(course, "forget_user", lambda *_a, **_kw: None)
    monkeypatch.setattr(crew_runner, "forget_user", anull)
    monkeypatch.setattr(tenant_profile, "refresh_agents", anull)
    monkeypatch.setattr(profile_sources, "remove_all", lambda *_a, **_kw: None)
    monkeypatch.setattr(context_packs, "refresh_refs", anull)
    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[get_current_user] = lambda: admin
    async with _client(app) as c:
        r = await c.delete(f"/fd/admin/users/{maya['id']}")
        assert r.status_code == 200, r.text
        assert r.json()["removed_agents"] == ["inbox-assistant"]
        # Nobody's agents go for an account that isn't there.
        r = await c.delete(f"/fd/admin/users/{maya['id']}")
        assert r.status_code == 404
    assert asked == [maya["id"]]
    assert await db.get_user_by_id(maya["id"]) is None


async def test_remove_agents_of_takes_only_that_owners_agents(monkeypatch):
    from captain_claw.flight_deck import server

    monkeypatch.setattr(server, "_load_process_registry", lambda: {
        "inbox": {"owner": "u-maya"}, "research": {"owner": "u-jonas"}, "old": {"owner": ""},
    })
    removed: list[tuple[str, bool]] = []

    async def fake_remove(slug: str, *, purge: bool = False):
        removed.append((slug, purge))

    def no_docker(**_kw):
        raise RuntimeError("no Docker here")

    monkeypatch.setattr(server, "_remove_process_agent", fake_remove)
    monkeypatch.setattr(server, "_deck_containers", no_docker)
    assert await server.remove_agents_of("u-maya") == ["inbox"]
    assert removed == [("inbox", True)]
    assert await server.remove_agents_of("") == []


# ── F11 ──

async def test_a_reset_gives_the_allowance_back_but_keeps_what_was_typed(db):
    user = await db.create_user("maya@example.com", hash_password("pw-123456"), "Maya", "user")
    day = "2026-11-08"
    await db.add_daily_usage(user["id"], day, messages=3, tokens=900)
    await db.reset_daily_usage(user["id"], day)

    assert (await db.get_daily_usage(user["id"], day))["messages"] == 0   # the allowance is back
    typed = {r["day"]: r["messages"] for r in await db.list_message_days("2026-11-01")}
    assert typed == {day: 3}                                               # Activity still sees them

    await db.add_daily_usage(user["id"], day, messages=2)
    await db.reset_daily_usage(user["id"], day)                            # a second reset adds up
    await db.add_daily_usage(user["id"], day, messages=1)
    typed = {r["day"]: r["messages"] for r in await db.list_message_days("2026-11-01")}
    assert typed == {day: 6}
    assert (await db.message_day_bounds())[user["id"]]["messages"] == 6


# ── F3 ──

def test_limit_messages_ask_the_admin(monkeypatch):
    monkeypatch.setattr(rate_limiter, "STANDALONE_MODE", False)
    monkeypatch.setattr(rate_limiter, "_spawn_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(rate_limiter, "get_user_limits", lambda _u: {"spawns_per_hour": 1, "requests_per_minute": 1})
    user = {"id": "u-maya"}
    rate_limiter.check_spawn_rate_limit(user)
    with pytest.raises(HTTPException) as err:
        rate_limiter.check_spawn_rate_limit(user)
    assert "ask your admin" in err.value.detail and "Upgrade" not in err.value.detail


# ── F10 ──

async def test_the_users_list_says_whether_crew_is_offered(db, monkeypatch):
    admin = await db.create_user("lena@example.com", hash_password("pw-123456"), "Lena", "admin")
    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[get_current_user] = lambda: admin
    async with _client(app) as c:
        monkeypatch.setenv("FD_CREW", "off")
        crew = (await c.get("/fd/admin/users")).json()["crew"]
        assert crew == {"offered": False, "terminal": False}
        monkeypatch.setenv("FD_CREW", "on")
        monkeypatch.setenv("FD_CREW_TERMINAL", "off")
        crew = (await c.get("/fd/admin/users")).json()["crew"]
        assert crew["terminal"] is False


async def test_the_update_page_knows_the_countdown(monkeypatch):
    from captain_claw.flight_deck import deck_update, deck_update_routes

    async def status(*, fetch: bool = False):
        return {"available": False, "why": "not a checkout", "job": None}

    async def get_mode(_db):
        return "off"

    monkeypatch.setattr(deck_update, "status", status)
    monkeypatch.setattr(deck_update, "get_mode", get_mode)
    monkeypatch.setattr(deck_update_routes, "get_db", lambda: None)
    monkeypatch.setenv("FD_UPDATE_COUNTDOWN_S", "90")
    out = await deck_update_routes._status()
    assert out["countdown_s"] == 90 and out["mode"] == "off"
