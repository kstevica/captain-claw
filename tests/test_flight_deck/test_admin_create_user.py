"""Admin account provisioning — ``POST /fd/admin/users``.

Team deployments close public self-registration (``FD_REGISTRATION_OPEN``
unset), so an admin creates teammate accounts here. These tests cover the
happy path, the validation gates, duplicate-email conflict, and — crucially —
that a non-admin cannot reach the endpoint at all.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import admin_routes
from captain_claw.flight_deck.auth import get_current_user, set_auth_db, verify_password
from captain_claw.flight_deck.db import FlightDeckDB

LOOPBACK = ("127.0.0.1", 40001)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=LOOPBACK),
        base_url="http://fd.test")


@pytest.fixture
async def fd_db(tmp_path: Path):
    from captain_claw.flight_deck import auth as fd_auth
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


def _app(user: dict) -> FastAPI:
    """Admin app with the current user pinned via dependency override."""
    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[get_current_user] = lambda: user
    return app


ADMIN = {"id": "admin-1", "role": "admin"}


async def test_admin_creates_user(fd_db):
    app = _app(ADMIN)
    async with _client(app) as c:
        r = await c.post("/fd/admin/users", json={
            "email": "Teammate@Example.com", "password": "hunter2",
            "display_name": "Tammy", "role": "user",
        })
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["user"]["email"] == "teammate@example.com"  # normalized
    assert body["user"]["display_name"] == "Tammy"
    assert body["user"]["role"] == "user"

    # Persisted, hashed, and loginable-shaped.
    row = await fd_db.get_user_by_email("teammate@example.com")
    assert row is not None
    assert row["password_hash"] != "hunter2"
    assert verify_password("hunter2", row["password_hash"])


async def test_display_name_defaults_to_email_local_part(fd_db):
    app = _app(ADMIN)
    async with _client(app) as c:
        r = await c.post("/fd/admin/users", json={
            "email": "solo@example.com", "password": "sixchars",
        })
    assert r.status_code == 201, r.text
    assert r.json()["user"]["display_name"] == "solo"
    assert r.json()["user"]["role"] == "user"  # default role


async def test_can_create_another_admin(fd_db):
    app = _app(ADMIN)
    async with _client(app) as c:
        r = await c.post("/fd/admin/users", json={
            "email": "boss@example.com", "password": "sixchars", "role": "admin",
        })
    assert r.status_code == 201, r.text
    row = await fd_db.get_user_by_email("boss@example.com")
    assert row["role"] == "admin"


async def test_non_admin_forbidden(fd_db):
    app = _app({"id": "u-2", "role": "user"})
    async with _client(app) as c:
        r = await c.post("/fd/admin/users", json={
            "email": "sneaky@example.com", "password": "sixchars",
        })
    assert r.status_code == 403
    # And nothing was created.
    assert await fd_db.get_user_by_email("sneaky@example.com") is None


async def test_duplicate_email_conflict(fd_db):
    await fd_db.create_user("dupe@example.com", "x", "Existing", "user")
    app = _app(ADMIN)
    async with _client(app) as c:
        r = await c.post("/fd/admin/users", json={
            "email": "Dupe@example.com", "password": "sixchars",  # case-insensitive
        })
    assert r.status_code == 409


async def test_short_password_rejected(fd_db):
    app = _app(ADMIN)
    async with _client(app) as c:
        r = await c.post("/fd/admin/users", json={
            "email": "short@example.com", "password": "12345",
        })
    assert r.status_code == 400
    assert await fd_db.get_user_by_email("short@example.com") is None


async def test_bad_email_rejected(fd_db):
    app = _app(ADMIN)
    async with _client(app) as c:
        r = await c.post("/fd/admin/users", json={
            "email": "not-an-email", "password": "sixchars",
        })
    assert r.status_code == 400


async def test_invalid_role_rejected(fd_db):
    app = _app(ADMIN)
    async with _client(app) as c:
        r = await c.post("/fd/admin/users", json={
            "email": "role@example.com", "password": "sixchars", "role": "superadmin",
        })
    assert r.status_code == 400
    assert await fd_db.get_user_by_email("role@example.com") is None
