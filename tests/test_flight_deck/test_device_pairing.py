"""Device pairing (RFC 8628-style) under /fd/auth/pair/*.

Pinned here:

* the full approve flow: start (public) → lookup + approve (signed in on a
  phone/desktop) → poll (public) returns a normal session — access token and
  the fd_refresh cookie — and that cookie works with /fd/auth/refresh;
* deny; single use; unknown and expired codes; codes are accepted in any
  case, with or without the dash;
* lookup/approve need a session; an auth-disabled deck refuses pairing;
* rate limits (per IP for start/poll, per user for lookup/approve) and the
  cap on pending pairings;
* only sha256(device_code) is stored; labels/user agents are cleaned before
  the approver reads them.

Real FlightDeckDB in a tmp dir; the module's clock is monkeypatched for expiry.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import auth_routes, server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.rate_limiter import _SlidingWindow

ALICE = "user-alice"
BOB = "user-bob"
GLASSES_IP = "198.51.100.20"
PHONE_IP = "198.51.100.30"
GLASSES_UA = ("Mozilla/5.0 (Linux; Android 14; Greatwhite Build/UKQ1.250303.001; wv) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/146.0.7680.177 "
              "Mobile Safari/537.36")
CODE_RE = re.compile(r"^[BCDFGHJKLMNPQRSTVWXZ]{4}-[BCDFGHJKLMNPQRSTVWXZ]{4}$")


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid, role, name in ((ALICE, "admin", "Alice"), (BOB, "user", "Bob")):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", name, role))
    await db._db.commit()

    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.setenv("FD_COOKIE_SECURE", "0")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setattr(auth_routes, "_pairings", {})
    monkeypatch.setattr(auth_routes, "_pairings_by_device", {})
    monkeypatch.setattr(auth_routes, "_pair_start_limiter", _SlidingWindow())
    monkeypatch.setattr(auth_routes, "_pair_poll_limiter", _SlidingWindow())
    monkeypatch.setattr(auth_routes, "_pair_approver_limiter", _SlidingWindow())
    clock = {"now": 1_800_000_000.0}
    monkeypatch.setattr(auth_routes, "_now", lambda: clock["now"])
    try:
        yield clock
    finally:
        await db.close()
        fd_auth._db = prev


def _client(ip: str = GLASSES_IP) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=(ip, 40000)),
        base_url="http://fd.test")


def _hdr(uid: str) -> dict:
    return {"Authorization": f"Bearer {create_access_token(uid)}"}


async def _start(c: httpx.AsyncClient, label: str | None = "Ray-Ban Display") -> dict:
    body = {} if label is None else {"label": label}
    r = await c.post("/fd/auth/pair/start", json=body, headers={"User-Agent": GLASSES_UA})
    assert r.status_code == 200, r.text
    return r.json()


async def _poll(c: httpx.AsyncClient, device_code: str) -> httpx.Response:
    return await c.post("/fd/auth/pair/poll", json={"device_code": device_code})


# ── Happy path ──────────────────────────────────────────────────────────────


async def test_full_approve_flow_gives_the_device_a_normal_session(deck):
    async with _client(GLASSES_IP) as glasses, _client(PHONE_IP) as phone:
        started = await _start(glasses)
        assert set(started) == {"device_code", "user_code", "expires_in", "interval",
                                "verification_path"}
        assert CODE_RE.match(started["user_code"])
        assert started["expires_in"] == 600 and started["interval"] == 3
        assert started["verification_path"] == "/hud/pair"
        assert len(started["device_code"]) >= 40

        assert (await _poll(glasses, started["device_code"])).json() == {"status": "pending"}

        info = await phone.get("/fd/auth/pair/lookup", params={"code": started["user_code"]},
                               headers=_hdr(BOB))
        assert info.status_code == 200, info.text
        info = info.json()
        assert info["user_code"] == started["user_code"]
        assert info["label"] == "Ray-Ban Display"
        assert info["user_agent"] == GLASSES_UA
        assert info["ip"] == GLASSES_IP
        assert info["created_at"].startswith("2027-01-15T")  # ISO, from the module clock
        assert info["expires_in"] == 600

        ok = await phone.post("/fd/auth/pair/approve",
                              json={"user_code": started["user_code"], "approve": True},
                              headers=_hdr(BOB))
        assert ok.status_code == 200 and ok.json() == {"ok": True, "status": "approved"}

        r = await _poll(glasses, started["device_code"])
        assert r.status_code == 200, r.text
        got = r.json()
        assert got["status"] == "approved" and got["token_type"] == "bearer"
        assert got["user"] == {"id": BOB, "email": f"{BOB}@x.co", "display_name": "Bob",
                               "role": "user"}
        cookie = r.headers["set-cookie"]
        assert cookie.startswith(f"{fd_auth.REFRESH_COOKIE}=")
        assert "path=/fd/auth" in cookie.lower() and "httponly" in cookie.lower()

        # The access token is Bob's…
        me = await glasses.get("/fd/auth/me",
                               headers={"Authorization": f"Bearer {got['access_token']}"})
        assert me.status_code == 200 and me.json()["id"] == BOB
        # …and the refresh cookie is a real session.
        refreshed = await glasses.post("/fd/auth/refresh")
        assert refreshed.status_code == 200, refreshed.text
        assert refreshed.json()["user"]["id"] == BOB

        # Single use: the pairing is gone.
        assert (await _poll(glasses, started["device_code"])).json() == {"status": "expired"}
        gone = await phone.get("/fd/auth/pair/lookup", params={"code": started["user_code"]},
                               headers=_hdr(BOB))
        assert gone.status_code == 404


async def test_deny_flow(deck):
    async with _client(GLASSES_IP) as glasses, _client(PHONE_IP) as phone:
        started = await _start(glasses)
        r = await phone.post("/fd/auth/pair/approve",
                             json={"user_code": started["user_code"], "approve": False},
                             headers=_hdr(ALICE))
        assert r.status_code == 200 and r.json() == {"ok": True, "status": "denied"}
        denied = await _poll(glasses, started["device_code"])
        assert denied.json() == {"status": "denied"}
        assert "set-cookie" not in denied.headers
        assert (await _poll(glasses, started["device_code"])).json() == {"status": "expired"}


async def test_a_decided_code_cannot_be_decided_again(deck):
    async with _client(GLASSES_IP) as glasses, _client(PHONE_IP) as phone:
        started = await _start(glasses)
        code = started["user_code"]
        first = await phone.post("/fd/auth/pair/approve", json={"user_code": code, "approve": False},
                                 headers=_hdr(ALICE))
        assert first.status_code == 200
        # Someone else can't flip a denial into an approval before the device polls.
        flip = await phone.post("/fd/auth/pair/approve", json={"user_code": code, "approve": True},
                                headers=_hdr(BOB))
        assert flip.status_code == 404
        assert (await _poll(glasses, started["device_code"])).json() == {"status": "denied"}


async def test_codes_are_normalized(deck):
    async with _client(GLASSES_IP) as glasses, _client(PHONE_IP) as phone:
        started = await _start(glasses)
        raw = started["user_code"].replace("-", "")
        sloppy = f" {raw[:4].lower()} {raw[4:].lower()} "
        info = await phone.get("/fd/auth/pair/lookup", params={"code": raw.lower()}, headers=_hdr(BOB))
        assert info.status_code == 200 and info.json()["user_code"] == started["user_code"]
        ok = await phone.post("/fd/auth/pair/approve", json={"user_code": sloppy, "approve": True},
                              headers=_hdr(BOB))
        assert ok.status_code == 200
        assert (await _poll(glasses, started["device_code"])).json()["status"] == "approved"


def test_normalize_and_format_helpers():
    assert auth_routes.normalize_user_code("bcdf-ghjk") == "BCDFGHJK"
    assert auth_routes.normalize_user_code(" BcDf GhJk\n") == "BCDFGHJK"
    assert auth_routes.normalize_user_code("A1E-O0") == ""  # not in the alphabet
    assert auth_routes.format_user_code("BCDFGHJK") == "BCDF-GHJK"


# ── Unknown / expired ───────────────────────────────────────────────────────


async def test_unknown_codes(deck):
    async with _client(PHONE_IP) as phone:
        r = await phone.get("/fd/auth/pair/lookup", params={"code": "BCDF-GHJK"}, headers=_hdr(BOB))
        assert r.status_code == 404
        r = await phone.post("/fd/auth/pair/approve", json={"user_code": "BCDF-GHJK", "approve": True},
                             headers=_hdr(BOB))
        assert r.status_code == 404
        r = await phone.get("/fd/auth/pair/lookup", params={"code": ""}, headers=_hdr(BOB))
        assert r.status_code == 404
        r = await _poll(phone, "not-a-device-code")
        assert r.status_code == 200 and r.json() == {"status": "expired"}


async def test_expiry(deck):
    async with _client(GLASSES_IP) as glasses, _client(PHONE_IP) as phone:
        started = await _start(glasses)
        deck["now"] += 599
        info = await phone.get("/fd/auth/pair/lookup", params={"code": started["user_code"]},
                               headers=_hdr(BOB))
        assert info.status_code == 200 and info.json()["expires_in"] == 1
        deck["now"] += 2
        assert (await phone.get("/fd/auth/pair/lookup", params={"code": started["user_code"]},
                                headers=_hdr(BOB))).status_code == 404
        assert (await phone.post("/fd/auth/pair/approve",
                                 json={"user_code": started["user_code"], "approve": True},
                                 headers=_hdr(BOB))).status_code == 404
        assert (await _poll(glasses, started["device_code"])).json() == {"status": "expired"}
        assert auth_routes._pairings == {} and auth_routes._pairings_by_device == {}


async def test_an_approval_not_claimed_in_time_expires(deck):
    async with _client(GLASSES_IP) as glasses, _client(PHONE_IP) as phone:
        started = await _start(glasses)
        await phone.post("/fd/auth/pair/approve",
                         json={"user_code": started["user_code"], "approve": True}, headers=_hdr(BOB))
        deck["now"] += 601
        r = await _poll(glasses, started["device_code"])
        assert r.json() == {"status": "expired"} and "set-cookie" not in r.headers


# ── Auth requirements ───────────────────────────────────────────────────────


async def test_lookup_and_approve_need_a_session(deck):
    async with _client(GLASSES_IP) as glasses:
        started = await _start(glasses)
        r = await glasses.get("/fd/auth/pair/lookup", params={"code": started["user_code"]})
        assert r.status_code == 401
        r = await glasses.post("/fd/auth/pair/approve",
                               json={"user_code": started["user_code"], "approve": True})
        assert r.status_code == 401
        r = await glasses.post("/fd/auth/pair/approve",
                               json={"user_code": started["user_code"], "approve": True},
                               headers={"Authorization": "Bearer not-a-jwt"})
        assert r.status_code == 401
        assert (await _poll(glasses, started["device_code"])).json() == {"status": "pending"}


async def test_auth_disabled_deck_refuses_pairing(deck, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    async with _client(GLASSES_IP) as c:
        assert (await c.post("/fd/auth/pair/start", json={})).status_code == 400
        assert (await _poll(c, "x")).status_code == 400
        assert (await c.get("/fd/auth/pair/lookup", params={"code": "BCDF-GHJK"})).status_code == 400
        assert (await c.post("/fd/auth/pair/approve",
                             json={"user_code": "BCDF-GHJK", "approve": True})).status_code == 400
    assert auth_routes._pairings == {}


# ── Limits ──────────────────────────────────────────────────────────────────


async def test_start_is_rate_limited_per_ip(deck):
    async with _client(GLASSES_IP) as glasses, _client(PHONE_IP) as other:
        for _ in range(10):
            await _start(glasses)
        r = await glasses.post("/fd/auth/pair/start", json={})
        assert r.status_code == 429
        assert (await other.post("/fd/auth/pair/start", json={})).status_code == 200


async def test_poll_is_rate_limited_per_ip(deck, monkeypatch):
    monkeypatch.setattr(auth_routes, "PAIR_POLL_LIMIT", (3, 60.0))
    async with _client(GLASSES_IP) as glasses:
        started = await _start(glasses)
        codes = [(await _poll(glasses, started["device_code"])).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


async def test_approver_is_rate_limited_per_user(deck, monkeypatch):
    monkeypatch.setattr(auth_routes, "PAIR_APPROVER_LIMIT", (2, 60.0))
    async with _client(PHONE_IP) as phone:
        got = [(await phone.get("/fd/auth/pair/lookup", params={"code": "BCDFGHJK"},
                                headers=_hdr(BOB))).status_code for _ in range(3)]
        assert got == [404, 404, 429]
        # Another user has their own budget.
        alice = await phone.get("/fd/auth/pair/lookup", params={"code": "BCDFGHJK"},
                                headers=_hdr(ALICE))
        assert alice.status_code == 404
        tries = [(await phone.post("/fd/auth/pair/approve",
                                   json={"user_code": "BCDFGHJK", "approve": True},
                                   headers=_hdr(BOB))).status_code for _ in range(3)]
        assert tries == [404, 404, 429]


async def test_pending_pairings_are_capped(deck, monkeypatch):
    monkeypatch.setattr(auth_routes, "PAIR_MAX_PENDING", 2)
    async with _client("192.0.2.1") as a, _client("192.0.2.2") as b, _client("192.0.2.3") as c:
        await _start(a)
        await _start(b)
        assert (await c.post("/fd/auth/pair/start", json={})).status_code == 429
        deck["now"] += 601  # expired ones are purged and make room
        assert (await c.post("/fd/auth/pair/start", json={})).status_code == 200


# ── Storage / input hygiene ─────────────────────────────────────────────────


async def test_only_the_device_code_hash_is_stored(deck):
    async with _client(GLASSES_IP) as glasses:
        started = await _start(glasses)
    (p,) = auth_routes._pairings.values()
    assert p.device_hash == hashlib.sha256(started["device_code"].encode()).hexdigest()
    assert started["device_code"] not in repr(auth_routes._pairings)
    assert started["device_code"] not in repr(auth_routes._pairings_by_device)


async def test_label_and_user_agent_are_cleaned(deck):
    async with _client(GLASSES_IP) as glasses, _client(PHONE_IP) as phone:
        started = await _start(glasses, label="  Ray‮Ban\n\x00Display\t" + "x" * 200)
        no_label = await _start(glasses, label=None)
        r = await glasses.post("/fd/auth/pair/start", content=b"",
                               headers={"User-Agent": "UA\x07" + "y" * 500})
        assert r.status_code == 200
        bare = r.json()
        info = (await phone.get("/fd/auth/pair/lookup", params={"code": started["user_code"]},
                                headers=_hdr(BOB))).json()
        assert info["label"].startswith("RayBan Display xxx") and len(info["label"]) == 60
        info2 = (await phone.get("/fd/auth/pair/lookup", params={"code": no_label["user_code"]},
                                 headers=_hdr(BOB))).json()
        assert info2["label"] == ""
        info3 = (await phone.get("/fd/auth/pair/lookup", params={"code": bare["user_code"]},
                                 headers=_hdr(BOB))).json()
        assert info3["user_agent"].startswith("UAyyy") and len(info3["user_agent"]) == 300


async def test_deleted_approver_does_not_mint_a_session(deck):
    async with _client(GLASSES_IP) as glasses, _client(PHONE_IP) as phone:
        started = await _start(glasses)
        await phone.post("/fd/auth/pair/approve",
                         json={"user_code": started["user_code"], "approve": True}, headers=_hdr(BOB))
        db = fd_auth.get_db()
        await db._db.execute("DELETE FROM users WHERE id = ?", (BOB,))
        await db._db.commit()
        r = await _poll(glasses, started["device_code"])
        assert r.json() == {"status": "denied"} and "set-cookie" not in r.headers
