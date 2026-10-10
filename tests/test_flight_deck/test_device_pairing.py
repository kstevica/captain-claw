"""Device pairing (RFC 8628-style) under /fd/auth/pair/*.

Pinned here:

* the full approve flow: start (public) → lookup + approve (signed in on a
  phone/desktop) → poll (public) returns a normal session — access token and
  the fd_refresh cookie — and that cookie works with /fd/auth/refresh;
* login still issues the same session (shared _issue_session);
* deny; single use (also under racing polls); unknown and expired codes;
  codes are accepted in any case, with or without the dash;
* lookup/approve need a full sign-in: the Authorization header AND this
  browser's own live refresh cookie for the same user — a bare access token
  (header or ?fd_token=) can't turn itself into a lasting device session;
  an auth-disabled deck refuses pairing;
* fairness: start/poll are limited per client network (IPv4 address, IPv6
  /64); one network holds at most PAIR_MAX_PER_NETWORK pending codes (a new
  code replaces its oldest), so a flood can't lock other devices out; the
  global cap is only a backstop; per-user limits for lookup/approve;
* the limiters' idle-key sweep is amortised (never a full scan per request);
* only sha256(device_code) is stored; labels/user agents are cleaned before
  the approver reads them.

Real FlightDeckDB in a tmp dir; the module's clock is monkeypatched for expiry.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import bcrypt
import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import auth_routes, server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

ALICE = "user-alice"
BOB = "user-bob"
PASSWORD = "correct-horse-battery"
# Low-cost bcrypt: the tests sign in a lot, and verify_password reads the cost from the hash.
PASSWORD_HASH = bcrypt.hashpw(PASSWORD.encode(), bcrypt.gensalt(rounds=4)).decode()
GLASSES_IP = "198.51.100.20"
PHONE_IP = "198.51.100.30"
GLASSES_UA = ("Mozilla/5.0 (Linux; Android 14; Greatwhite Build/UKQ1.250303.001; wv) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/146.0.7680.177 "
              "Mobile Safari/537.36")
CODE_RE = re.compile(r"^[BCDFGHJKLMNPQRSTVWXZ]{4}-[BCDFGHJKLMNPQRSTVWXZ]{4}$")
FULL_SIGN_IN = auth_routes._FULL_SIGN_IN_NEEDED


def _email(uid: str) -> str:
    return f"{uid}@x.co"


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid, role, name in ((ALICE, "admin", "Alice"), (BOB, "user", "Bob")):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, _email(uid), PASSWORD_HASH, name, role))
    await db._db.commit()

    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.setenv("FD_COOKIE_SECURE", "0")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setattr(auth_routes, "_pairings", OrderedDict())
    monkeypatch.setattr(auth_routes, "_pairings_by_device", {})
    monkeypatch.setattr(auth_routes, "_pairings_by_network", {})
    monkeypatch.setattr(auth_routes, "_pair_start_limiter", auth_routes._PairLimiter())
    monkeypatch.setattr(auth_routes, "_pair_poll_limiter", auth_routes._PairLimiter())
    monkeypatch.setattr(auth_routes, "_pair_approver_limiter", auth_routes._PairLimiter())
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


async def _login(c: httpx.AsyncClient, uid: str) -> str:
    """Sign *uid* in on *c* (its jar keeps the fd_refresh cookie); the access token."""
    r = await c.post("/fd/auth/login", json={"email": _email(uid), "password": PASSWORD})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@asynccontextmanager
async def _approver(uid: str, ip: str = PHONE_IP):
    """A phone/desktop browser where *uid* signed in: refresh cookie + Bearer header."""
    async with _client(ip) as c:
        c.headers["Authorization"] = f"Bearer {await _login(c, uid)}"
        yield c


def _hdr(uid: str) -> dict:
    return {"Authorization": f"Bearer {create_access_token(uid)}"}


async def _start(c: httpx.AsyncClient, label: str | None = "Ray-Ban Display") -> dict:
    body = {} if label is None else {"label": label}
    r = await c.post("/fd/auth/pair/start", json=body, headers={"User-Agent": GLASSES_UA})
    assert r.status_code == 200, r.text
    return r.json()


async def _poll(c: httpx.AsyncClient, device_code: str) -> httpx.Response:
    return await c.post("/fd/auth/pair/poll", json={"device_code": device_code})


async def _sessions(uid: str) -> int:
    db = fd_auth.get_db()
    async with db._db.execute("SELECT COUNT(*) FROM user_sessions WHERE user_id = ?", (uid,)) as cur:
        return (await cur.fetchone())[0]


# ── Happy path ──────────────────────────────────────────────────────────────


async def test_full_approve_flow_gives_the_device_a_normal_session(deck):
    async with _client(GLASSES_IP) as glasses, _approver(BOB) as phone:
        started = await _start(glasses)
        assert set(started) == {"device_code", "user_code", "expires_in", "interval",
                                "verification_path"}
        assert CODE_RE.match(started["user_code"])
        assert started["expires_in"] == 600 and started["interval"] == 3
        assert started["verification_path"] == "/hud/pair"
        assert len(started["device_code"]) >= 40

        assert (await _poll(glasses, started["device_code"])).json() == {"status": "pending"}

        info = await phone.get("/fd/auth/pair/lookup", params={"code": started["user_code"]})
        assert info.status_code == 200, info.text
        info = info.json()
        assert info["user_code"] == started["user_code"]
        assert info["label"] == "Ray-Ban Display"
        assert info["user_agent"] == GLASSES_UA
        assert info["ip"] == GLASSES_IP
        assert info["created_at"].startswith("2027-01-15T")  # ISO, from the module clock
        assert info["expires_in"] == 600

        ok = await phone.post("/fd/auth/pair/approve",
                              json={"user_code": started["user_code"], "approve": True})
        assert ok.status_code == 200 and ok.json() == {"ok": True, "status": "approved"}

        r = await _poll(glasses, started["device_code"])
        assert r.status_code == 200, r.text
        got = r.json()
        assert got["status"] == "approved" and got["token_type"] == "bearer"
        assert got["user"] == {"id": BOB, "email": _email(BOB), "display_name": "Bob",
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
        gone = await phone.get("/fd/auth/pair/lookup", params={"code": started["user_code"]})
        assert gone.status_code == 404


async def test_login_sets_the_refresh_cookie_and_returns_the_user(deck):
    """login() and pairing share _issue_session: login's response must not drift."""
    async with _client(PHONE_IP) as c:
        r = await c.post("/fd/auth/login", json={"email": _email(ALICE), "password": PASSWORD})
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body) == {"access_token", "token_type", "user"}
        assert body["token_type"] == "bearer"
        assert body["user"] == {"id": ALICE, "email": _email(ALICE), "display_name": "Alice",
                                "role": "admin"}
        assert fd_auth.decode_access_token(body["access_token"])["sub"] == ALICE
        cookie = r.headers["set-cookie"]
        assert cookie.startswith(f"{fd_auth.REFRESH_COOKIE}=")
        attrs = {a.strip() for a in cookie.lower().split(";")[1:]}
        assert {"path=/fd/auth", "httponly", "samesite=lax",
                f"max-age={int(fd_auth.REFRESH_TOKEN_TTL.total_seconds())}"} <= attrs
        assert "secure" not in attrs  # FD_COOKIE_SECURE=0
        assert await _sessions(ALICE) == 1
        refreshed = await c.post("/fd/auth/refresh")
        assert refreshed.status_code == 200 and refreshed.json()["user"]["id"] == ALICE

        bad = await c.post("/fd/auth/login", json={"email": _email(ALICE), "password": "nope"})
        assert bad.status_code == 401 and bad.json() == {"detail": "Invalid credentials"}
        unknown = await c.post("/fd/auth/login", json={"email": "who@x.co", "password": PASSWORD})
        assert unknown.status_code == 401 and "set-cookie" not in unknown.headers


async def test_deny_flow(deck):
    async with _client(GLASSES_IP) as glasses, _approver(ALICE) as phone:
        started = await _start(glasses)
        r = await phone.post("/fd/auth/pair/approve",
                             json={"user_code": started["user_code"], "approve": False})
        assert r.status_code == 200 and r.json() == {"ok": True, "status": "denied"}
        denied = await _poll(glasses, started["device_code"])
        assert denied.json() == {"status": "denied"}
        assert "set-cookie" not in denied.headers
        assert (await _poll(glasses, started["device_code"])).json() == {"status": "expired"}


async def test_a_decided_code_cannot_be_decided_again(deck):
    async with _client(GLASSES_IP) as glasses, _approver(ALICE) as alice, _approver(BOB) as bob:
        started = await _start(glasses)
        code = started["user_code"]
        first = await alice.post("/fd/auth/pair/approve", json={"user_code": code, "approve": False})
        assert first.status_code == 200
        # Someone else can't flip a denial into an approval before the device polls.
        flip = await bob.post("/fd/auth/pair/approve", json={"user_code": code, "approve": True})
        assert flip.status_code == 404
        assert (await _poll(glasses, started["device_code"])).json() == {"status": "denied"}


async def test_racing_polls_claim_the_session_once(deck):
    async with _client(GLASSES_IP) as glasses, _approver(BOB) as phone:
        started = await _start(glasses)
        await phone.post("/fd/auth/pair/approve",
                         json={"user_code": started["user_code"], "approve": True})
        before = await _sessions(BOB)
        polls = await asyncio.gather(*(_poll(glasses, started["device_code"]) for _ in range(5)))
    statuses = sorted(r.json()["status"] for r in polls)
    assert statuses == ["approved"] + ["expired"] * 4
    assert sum("set-cookie" in r.headers for r in polls) == 1
    assert await _sessions(BOB) == before + 1


async def test_codes_are_normalized(deck):
    async with _client(GLASSES_IP) as glasses, _approver(BOB) as phone:
        started = await _start(glasses)
        raw = started["user_code"].replace("-", "")
        sloppy = f" {raw[:4].lower()} {raw[4:].lower()} "
        info = await phone.get("/fd/auth/pair/lookup", params={"code": raw.lower()})
        assert info.status_code == 200 and info.json()["user_code"] == started["user_code"]
        ok = await phone.post("/fd/auth/pair/approve", json={"user_code": sloppy, "approve": True})
        assert ok.status_code == 200
        assert (await _poll(glasses, started["device_code"])).json()["status"] == "approved"


def test_normalize_and_format_helpers():
    assert auth_routes.normalize_user_code("bcdf-ghjk") == "BCDFGHJK"
    assert auth_routes.normalize_user_code(" BcDf GhJk\n") == "BCDFGHJK"
    assert auth_routes.normalize_user_code("A1E-O0") == ""  # not in the alphabet
    assert auth_routes.format_user_code("BCDFGHJK") == "BCDF-GHJK"


# ── Unknown / expired ───────────────────────────────────────────────────────


async def test_unknown_codes(deck):
    async with _approver(BOB) as phone:
        r = await phone.get("/fd/auth/pair/lookup", params={"code": "BCDF-GHJK"})
        assert r.status_code == 404
        r = await phone.post("/fd/auth/pair/approve", json={"user_code": "BCDF-GHJK", "approve": True})
        assert r.status_code == 404
        r = await phone.get("/fd/auth/pair/lookup", params={"code": ""})
        assert r.status_code == 404
        r = await _poll(phone, "not-a-device-code")
        assert r.status_code == 200 and r.json() == {"status": "expired"}


async def test_expiry(deck):
    async with _client(GLASSES_IP) as glasses, _approver(BOB) as phone:
        started = await _start(glasses)
        deck["now"] += 599
        info = await phone.get("/fd/auth/pair/lookup", params={"code": started["user_code"]})
        assert info.status_code == 200 and info.json()["expires_in"] == 1
        deck["now"] += 2
        assert (await phone.get("/fd/auth/pair/lookup",
                                params={"code": started["user_code"]})).status_code == 404
        assert (await phone.post("/fd/auth/pair/approve",
                                 json={"user_code": started["user_code"], "approve": True})).status_code == 404
        assert (await _poll(glasses, started["device_code"])).json() == {"status": "expired"}
        assert auth_routes._pairings == {} and auth_routes._pairings_by_device == {}
        assert auth_routes._pairings_by_network == {}


async def test_an_approval_not_claimed_in_time_expires(deck):
    async with _client(GLASSES_IP) as glasses, _approver(BOB) as phone:
        started = await _start(glasses)
        await phone.post("/fd/auth/pair/approve",
                         json={"user_code": started["user_code"], "approve": True})
        deck["now"] += 601
        r = await _poll(glasses, started["device_code"])
        assert r.json() == {"status": "expired"} and "set-cookie" not in r.headers


async def test_expired_pairings_are_purged_oldest_first(deck):
    async with _client("192.0.2.1") as a, _client("192.0.2.2") as b:
        old = await _start(a)
        deck["now"] += 300
        young = await _start(b)
        deck["now"] += 301  # the first one is past its TTL, the second isn't
        assert (await _poll(a, old["device_code"])).json() == {"status": "expired"}
        assert (await _poll(b, young["device_code"])).json() == {"status": "pending"}
        assert list(auth_routes._pairings) == [young["user_code"].replace("-", "")]


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


async def test_an_access_token_alone_cannot_mint_a_device_session(deck):
    """A leaked 15-minute access token (logs, ?fd_token= URLs) must not become a
    lasting refresh session by pairing a device of its own."""
    async with _client(GLASSES_IP) as glasses, _client(PHONE_IP) as attacker:
        started = await _start(glasses)
        code = started["user_code"]
        r = await attacker.get("/fd/auth/pair/lookup", params={"code": code}, headers=_hdr(BOB))
        assert r.status_code == 403 and r.json() == {"detail": FULL_SIGN_IN}
        r = await attacker.post("/fd/auth/pair/approve", json={"user_code": code, "approve": True},
                                headers=_hdr(BOB))
        assert r.status_code == 403 and r.json() == {"detail": FULL_SIGN_IN}
        # Nor through the ?fd_token= fallback.
        r = await attacker.post("/fd/auth/pair/approve", json={"user_code": code, "approve": True},
                                params={"fd_token": create_access_token(BOB)})
        assert r.status_code == 403
        # A cookie that doesn't map to a session is no better.
        attacker.cookies.set(fd_auth.REFRESH_COOKIE, "forged", domain="fd.test", path="/fd/auth")
        r = await attacker.post("/fd/auth/pair/approve", json={"user_code": code, "approve": True},
                                headers=_hdr(BOB))
        assert r.status_code == 403
        assert (await _poll(glasses, started["device_code"])).json() == {"status": "pending"}
    assert await _sessions(BOB) == 0


async def test_the_cookie_must_be_a_live_session_of_the_same_user(deck):
    async with _client(GLASSES_IP) as glasses, _client(PHONE_IP) as phone:
        started = await _start(glasses)
        code = started["user_code"]
        alice_token = await _login(phone, ALICE)  # the jar now holds Alice's cookie
        # Bob's access token riding on Alice's browser session: refused.
        r = await phone.post("/fd/auth/pair/approve", json={"user_code": code, "approve": True},
                             headers=_hdr(BOB))
        assert r.status_code == 403 and r.json() == {"detail": FULL_SIGN_IN}
        # Alice's own header + cookie, but only through ?fd_token=: refused.
        r = await phone.get("/fd/auth/pair/lookup", params={"code": code, "fd_token": alice_token})
        assert r.status_code == 403
        # An expired session: refused.
        db = fd_auth.get_db()
        await db._db.execute("UPDATE user_sessions SET expires_at = '2020-01-01T00:00:00+00:00'"
                             " WHERE user_id = ?", (ALICE,))
        await db._db.commit()
        alice = {"Authorization": f"Bearer {alice_token}"}
        r = await phone.get("/fd/auth/pair/lookup", params={"code": code}, headers=alice)
        assert r.status_code == 403
        # Signed in again: fine.
        alice = {"Authorization": f"Bearer {await _login(phone, ALICE)}"}
        r = await phone.get("/fd/auth/pair/lookup", params={"code": code}, headers=alice)
        assert r.status_code == 200, r.text
        # Signed out (the session row is gone): refused again.
        await phone.post("/fd/auth/logout")
        assert phone.cookies.get(fd_auth.REFRESH_COOKIE) is None
        r = await phone.get("/fd/auth/pair/lookup", params={"code": code}, headers=alice)
        assert r.status_code == 403


async def test_auth_disabled_deck_refuses_pairing(deck, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    async with _client(GLASSES_IP) as c:
        assert (await c.post("/fd/auth/pair/start", json={})).status_code == 400
        assert (await _poll(c, "x")).status_code == 400
        assert (await c.get("/fd/auth/pair/lookup", params={"code": "BCDF-GHJK"})).status_code == 400
        assert (await c.post("/fd/auth/pair/approve",
                             json={"user_code": "BCDF-GHJK", "approve": True})).status_code == 400
    assert auth_routes._pairings == {}


# ── Limits and fairness ─────────────────────────────────────────────────────


def test_client_network_keys():
    net = auth_routes._client_network
    assert net("198.51.100.20") == "198.51.100.20"
    assert net("2001:db8:1:2:aaaa:bbbb:cccc:dddd") == "2001:db8:1:2::/64"
    assert net("2001:db8:1:2::1") == net("2001:db8:1:2:ffff::9")
    assert net("2001:db8:1:3::1") != net("2001:db8:1:2::1")
    assert net("::ffff:198.51.100.20") == "198.51.100.20"  # IPv4-mapped
    assert net("fe80::1%eth0") == "fe80::/64"
    assert net("") == "" and net("not-an-ip") == "not-an-ip"


async def test_start_is_rate_limited_per_network(deck):
    async with _client(GLASSES_IP) as glasses, _client(PHONE_IP) as other:
        for _ in range(10):
            await _start(glasses)
        r = await glasses.post("/fd/auth/pair/start", json={})
        assert r.status_code == 429
        assert (await other.post("/fd/auth/pair/start", json={})).status_code == 200
    # Every address of an IPv6 /64 shares one budget.
    for i in range(10):
        async with _client(f"2001:db8:1:2::{i + 1:x}") as c:
            await _start(c)
    async with _client("2001:db8:1:2::ffff") as c:
        assert (await c.post("/fd/auth/pair/start", json={})).status_code == 429
    async with _client("2001:db8:1:3::1") as c:  # the next /64 is someone else
        assert (await c.post("/fd/auth/pair/start", json={})).status_code == 200


async def test_poll_is_rate_limited_per_network(deck, monkeypatch):
    monkeypatch.setattr(auth_routes, "PAIR_POLL_LIMIT", (3, 60.0))
    async with _client(GLASSES_IP) as glasses:
        started = await _start(glasses)
        codes = [(await _poll(glasses, started["device_code"])).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    async with _client("2001:db8:9:9::1") as a, _client("2001:db8:9:9::2") as b:
        got = [(await _poll(c, "x")).status_code for c in (a, b, a, b)]
    assert got == [200, 200, 200, 429]


async def test_one_network_holds_at_most_three_pending_codes(deck):
    async with _client(GLASSES_IP) as glasses, _approver(BOB) as phone:
        codes = [await _start(glasses) for _ in range(4)]
        # The oldest was replaced by the newest: a wearer pressing "New code"
        # keeps working, a flood from one network keeps three slots.
        assert (await _poll(glasses, codes[0]["device_code"])).json() == {"status": "expired"}
        r = await phone.get("/fd/auth/pair/lookup", params={"code": codes[0]["user_code"]})
        assert r.status_code == 404
        for c in codes[1:]:
            assert (await _poll(glasses, c["device_code"])).json() == {"status": "pending"}
        assert len(auth_routes._pairings) == 3
        assert auth_routes.PAIR_MAX_PER_NETWORK == 3


async def test_an_approved_code_is_not_replaced_by_new_ones(deck):
    async with _client(GLASSES_IP) as glasses, _approver(BOB) as phone:
        approved = await _start(glasses)
        await phone.post("/fd/auth/pair/approve",
                         json={"user_code": approved["user_code"], "approve": True})
        for _ in range(4):
            await _start(glasses)
        r = await _poll(glasses, approved["device_code"])
        assert r.json()["status"] == "approved"


async def test_a_flood_from_many_networks_cannot_lock_out_a_new_device(deck, monkeypatch):
    # The old global-cap DoS: 50 sources x 10 starts filled a 500-slot table.
    # Per-network caps now bind long before the (backstop) global cap.
    monkeypatch.setattr(auth_routes, "PAIR_MAX_PENDING", 200)
    for i in range(50):
        async with _client(f"203.0.113.{i + 1}") as v4:
            for _ in range(10):
                await _start(v4)
        async with _client(f"2001:db8:1:2::{i + 1:x}") as v6:  # all one /64
            r = await v6.post("/fd/auth/pair/start", json={})
            assert r.status_code in (200, 429)
    assert len(auth_routes._pairings) == 50 * 3 + 3
    async with _client("198.51.100.99") as wearer:
        assert (await wearer.post("/fd/auth/pair/start", json={})).status_code == 200


async def test_pending_pairings_have_a_global_backstop(deck, monkeypatch):
    monkeypatch.setattr(auth_routes, "PAIR_MAX_PENDING", 2)
    async with _client("192.0.2.1") as a, _client("192.0.2.2") as b, _client("192.0.2.3") as c:
        await _start(a)
        await _start(b)
        assert (await c.post("/fd/auth/pair/start", json={})).status_code == 429
        deck["now"] += 601  # expired ones are purged and make room
        assert (await c.post("/fd/auth/pair/start", json={})).status_code == 200


async def test_a_refused_start_keeps_the_codes_it_would_replace(deck, monkeypatch):
    async with _client(GLASSES_IP) as glasses, _client("192.0.2.9") as other:
        mine = [await _start(glasses) for _ in range(3)]
        await _start(other)
        monkeypatch.setattr(auth_routes, "PAIR_MAX_PENDING", 3)  # the table is now over-full
        assert (await glasses.post("/fd/auth/pair/start", json={})).status_code == 429
        for c in mine:
            assert (await _poll(glasses, c["device_code"])).json() == {"status": "pending"}
        monkeypatch.setattr(auth_routes, "PAIR_MAX_PENDING", 4)  # one replaced, one added: fits
        assert (await glasses.post("/fd/auth/pair/start", json={})).status_code == 200
        assert (await _poll(glasses, mine[0]["device_code"])).json() == {"status": "expired"}


async def test_approver_is_rate_limited_per_user(deck, monkeypatch):
    monkeypatch.setattr(auth_routes, "PAIR_APPROVER_LIMIT", (2, 60.0))
    async with _approver(BOB) as bob, _approver(ALICE) as alice:
        got = [(await bob.get("/fd/auth/pair/lookup", params={"code": "BCDFGHJK"})).status_code
               for _ in range(3)]
        assert got == [404, 404, 429]
        # Another user has their own budget.
        r = await alice.get("/fd/auth/pair/lookup", params={"code": "BCDFGHJK"})
        assert r.status_code == 404
        tries = [(await bob.post("/fd/auth/pair/approve",
                                 json={"user_code": "BCDFGHJK", "approve": True})).status_code
                 for _ in range(3)]
        assert tries == [404, 404, 429]


def _seed(limiter: auth_routes._PairLimiter, keys: list[str], at: float) -> None:
    for k in keys:
        limiter._requests[k] = [at]


def test_limiter_sweeps_idle_keys_at_most_every_few_seconds(monkeypatch):
    monkeypatch.setattr(auth_routes, "_LIMITER_GC_AT", 3)
    limiter = auth_routes._PairLimiter()
    sweeps: list[float] = []
    real_sweep = limiter.sweep
    monkeypatch.setattr(limiter, "sweep", lambda cutoff: (sweeps.append(cutoff), real_sweep(cutoff)))
    long_ago = time.monotonic() - 3600
    _seed(limiter, [f"idle-{i}" for i in range(5)], long_ago)
    _seed(limiter, ["live"], time.monotonic())

    auth_routes._rate_limit(limiter, "new", (10, 60.0))
    assert set(limiter._requests) == {"live", "new"} and len(sweeps) == 1

    # More idle keys right away: no new sweep yet (amortised), so they stay…
    _seed(limiter, [f"idle2-{i}" for i in range(5)], long_ago)
    auth_routes._rate_limit(limiter, "new", (10, 60.0))
    assert len(sweeps) == 1 and len(limiter._requests) == 7
    # …until the sweep interval has passed.
    limiter.swept_at -= auth_routes._LIMITER_SWEEP_EVERY_S
    auth_routes._rate_limit(limiter, "new", (10, 60.0))
    assert len(sweeps) == 2 and set(limiter._requests) == {"live", "new"}


def test_limiter_never_scans_every_live_key_on_every_request(monkeypatch):
    # A spray of distinct live sources (nothing is idle, so a sweep frees
    # nothing): the old GC rescanned all of them on every call.
    limiter = auth_routes._PairLimiter()
    sweeps: list[float] = []
    real_sweep = limiter.sweep
    monkeypatch.setattr(limiter, "sweep", lambda cutoff: (sweeps.append(cutoff), real_sweep(cutoff)))
    _seed(limiter, [f"net:{i}" for i in range(auth_routes._LIMITER_GC_AT + 20_000)], time.monotonic())
    for i in range(2000):
        auth_routes._rate_limit(limiter, f"spray:{i}", auth_routes.PAIR_START_LIMIT)
    assert len(sweeps) == 1
    assert len(limiter._requests) == auth_routes._LIMITER_GC_AT + 22_000


# ── Storage / input hygiene ─────────────────────────────────────────────────


async def test_only_the_device_code_hash_is_stored(deck):
    async with _client(GLASSES_IP) as glasses:
        started = await _start(glasses)
    (p,) = auth_routes._pairings.values()
    assert p.device_hash == hashlib.sha256(started["device_code"].encode()).hexdigest()
    assert started["device_code"] not in repr(auth_routes._pairings)
    assert started["device_code"] not in repr(auth_routes._pairings_by_device)


async def test_label_and_user_agent_are_cleaned(deck):
    async with _client(GLASSES_IP) as glasses, _approver(BOB) as phone:
        started = await _start(glasses, label="  Ray‮Ban\n\x00Display\t" + "x" * 200)
        no_label = await _start(glasses, label=None)
        r = await glasses.post("/fd/auth/pair/start", content=b"",
                               headers={"User-Agent": "UA\x07" + "y" * 500})
        assert r.status_code == 200
        bare = r.json()
        info = (await phone.get("/fd/auth/pair/lookup", params={"code": started["user_code"]})).json()
        assert info["label"].startswith("RayBan Display xxx") and len(info["label"]) == 60
        info2 = (await phone.get("/fd/auth/pair/lookup", params={"code": no_label["user_code"]})).json()
        assert info2["label"] == ""
        info3 = (await phone.get("/fd/auth/pair/lookup", params={"code": bare["user_code"]})).json()
        assert info3["user_agent"].startswith("UAyyy") and len(info3["user_agent"]) == 300


async def test_deleted_approver_does_not_mint_a_session(deck):
    async with _client(GLASSES_IP) as glasses, _approver(BOB) as phone:
        started = await _start(glasses)
        await phone.post("/fd/auth/pair/approve",
                         json={"user_code": started["user_code"], "approve": True})
        db = fd_auth.get_db()
        await db._db.execute("DELETE FROM users WHERE id = ?", (BOB,))
        await db._db.commit()
        r = await _poll(glasses, started["device_code"])
        assert r.json() == {"status": "denied"} and "set-cookie" not in r.headers
