"""A1 — chat-only shared agents, Flight Deck side.

Pinned here:

* identifiers: ``agent_ref`` parsing, the derived instance id (contract test
  vector), instance ids across spawn/re-spawn, resolution by instance (slug
  collisions across owners and runtimes), the managed-worker rule (parity with
  SimpleLayout.tsx) and ``check_shareable``;
* the ``X-FD-Speaker`` assertion and ack reproduce the contract test vector;
* membership: ``member_check`` caches only True, for at most ``max_age``;
* routes: the ``agent`` share type exists only with ``FD_AGENT_SHARING`` on;
  shares are chat-only ('view'), refused for unowned / managed / token-less
  agents, persist the process instance id, and notify; ``/fd/shared-agents``
  lists only live agents still owned by their sharer and never leaks the
  token, port or host; revoking (DELETE / leave / agent removal) closes the
  member's sockets; process and container rows carry ``agent_ref``; shared chat
  history ids are per-user server-side;
* the member socket: accept-then-close with ``fd_close`` for every refusal; FD
  connects to the recorded port with the recorded token and a signed header
  the (fake) agent verifies; no ack → 4426; the profile context goes upstream
  before any client frame; the frame allowlist; the outstanding-turn cap;
  revocation (immediately via DELETE, within a watchdog tick via the DB); JWT
  expiry and ``fd_auth``; the per-member socket cap.

Real FlightDeckDB + process registry in tmp dirs; Docker and agent processes
are faked; the agent's socket is a real ``websockets`` server on an ephemeral
port in its own thread.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, urlsplit

import anyio
import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from captain_claw.flight_deck import agent_sharing as sharing
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import server
from captain_claw.flight_deck import tenant_profile as tp
from captain_claw.flight_deck.auth import create_access_token, get_jwt_secret, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

OWNER, MEMBER, OTHER, ADMIN = "u-owner", "u-member", "u-other", "u-admin"
USERS = ((OWNER, "user", "Olga Owner"), (MEMBER, "user", "Mia Member"),
         (OTHER, "user", "Oscar Other"), (ADMIN, "admin", "Ada Admin"))
INST = "0123456789abcdef"
REF = f"process:helper:{INST}"
DOCKER_INST = "fedcba9876543210"
DOCKER_REF = f"docker:helper:{DOCKER_INST}"
HELPER_TOK = "helper-tok"
REMOTE = ("203.0.113.9", 40001)
WS_FD = "ws://localhost:25080"
HTTP_FD = "http://localhost:25080"
NOT_ALLOWED = {"type": "error", "code": "not_allowed", "message": "Not available on a shared agent"}

# Contract part 0 §5 test vector.
VECTOR_PAYLOAD = {"v": 1, "sub": "u-member", "name": "Ana", "owner": "u-owner",
                  "owner_name": "Olga", "ref": "process:helper:0123456789abcdef", "lane": "A",
                  "conn": "c0ffee00c0ffee00", "iat": 1760000000, "exp": 1760000060,
                  "nonce": "00112233445566aa"}
VECTOR_KEY = "43ef71203472b43d8204b4b75e823784c90669104c1ce40e80adfc07682850c2"
VECTOR_HEADER = (
    "v1.eyJjb25uIjoiYzBmZmVlMDBjMGZmZWUwMCIsImV4cCI6MTc2MDAwMDA2MCwiaWF0IjoxNzYwMDAwMDAwLCJsYW5l"
    "IjoiQSIsIm5hbWUiOiJBbmEiLCJub25jZSI6IjAwMTEyMjMzNDQ1NTY2YWEiLCJvd25lciI6InUtb3duZXIiLCJvd25l"
    "cl9uYW1lIjoiT2xnYSIsInJlZiI6InByb2Nlc3M6aGVscGVyOjAxMjM0NTY3ODlhYmNkZWYiLCJzdWIiOiJ1LW1lbWJl"
    "ciIsInYiOjF9.ylLbFYmtrikkdnr6BjLD0BEIO2RxWIW8A5X7zpnivHc")
VECTOR_ACK = "941895124a6e13f5"


# ── Agent-side verification, as the contract specifies it (part 0 §5) ──────


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def verify_speaker(header: str, web_auth: str, seen: set, now: float | None = None) -> dict | None:
    parts = header.split(".")
    if len(parts) != 3 or parts[0] != "v1" or not web_auth:
        return None
    key = hmac.new(web_auth.encode("utf-8"), b"captain-claw/fd-speaker/v1", hashlib.sha256).digest()
    want = hmac.new(key, ("v1." + parts[1]).encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(want, _unb64(parts[2])):
        return None
    payload = json.loads(_unb64(parts[1]))
    now = time.time() if now is None else now
    if payload.get("v") != 1 or not (payload["iat"] - 30 <= now <= payload["exp"]):
        return None
    if payload["exp"] - payload["iat"] > 120 or payload.get("lane") not in ("A", "B", "C"):
        return None
    if payload["nonce"] in seen:
        return None
    seen.add(payload["nonce"])
    return payload


class FakeAgent:
    """A Captain Claw agent's /ws as far as FD can tell: checks the token and
    the X-FD-Speaker assertion, acks it in ``welcome`` (per ``mode``) and
    records every frame it receives."""

    def __init__(self, web_auth: str):
        self.web_auth = web_auth
        self.mode = "ack"   # ack | noack | early | silent | close4429 | close4401
        self.requests: list[dict] = []
        self.frames: list[tuple[int, dict]] = []
        self.conns: list = []
        self._seen: set = set()
        self._cv = threading.Condition()
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._ready.wait(5), "fake agent did not start"

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start())
        self._ready.set()
        self.loop.run_forever()

    async def _start(self) -> None:
        from websockets.asyncio.server import serve

        self.server = await serve(self._handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def _handler(self, ws) -> None:
        header = ws.request.headers.get("X-FD-Speaker", "")
        token = parse_qs(urlsplit(ws.request.path).query).get("token", [""])[0]
        payload = verify_speaker(header, self.web_auth, self._seen)
        with self._cv:
            self.requests.append({"path": ws.request.path, "host": ws.request.headers.get("Host"),
                                  "header": header, "payload": payload})
            self._cv.notify_all()
        if self.mode == "close4429":
            await ws.close(4429, "speaker capacity")
            return
        if token != self.web_auth or payload is None or self.mode == "close4401":
            await ws.close(4401, "bad assertion")
            return
        if self.mode == "silent":
            async for _ in ws:
                pass
            return
        if self.mode == "early":
            await ws.send(json.dumps({"type": "status", "status": "thinking"}))
            await ws.send("not json at all")
        welcome = {"type": "welcome", "session": {"name": "default"}, "models": [],
                   "speaker": {"id": payload["sub"], "name": payload["name"],
                               "owner_name": payload["owner_name"], "lane": payload["lane"]}}
        if self.mode != "noack":
            welcome["speaker_ack"] = hashlib.sha256(header.encode("ascii")).hexdigest()[:16]
        await ws.send(json.dumps(welcome))
        with self._cv:
            idx = len(self.conns)
            self.conns.append(ws)
        try:
            async for msg in ws:
                with self._cv:
                    self.frames.append((idx, json.loads(msg)))
                    self._cv.notify_all()
        except Exception:
            pass

    def wait_frames(self, n: int, timeout: float = 5.0) -> list[dict]:
        with self._cv:
            assert self._cv.wait_for(lambda: len(self.frames) >= n, timeout), self.frames
            return [f for _, f in self.frames]

    def wait_requests(self, n: int, timeout: float = 5.0) -> list[dict]:
        with self._cv:
            assert self._cv.wait_for(lambda: len(self.requests) >= n, timeout), self.requests
            return list(self.requests)

    def push(self, conn_idx: int, frame: dict) -> None:
        fut = asyncio.run_coroutine_threadsafe(
            self.conns[conn_idx].send(json.dumps(frame)), self.loop)
        fut.result(5)

    def stop(self) -> None:
        async def _stop():
            self.server.close()
            await self.server.wait_closed()

        try:
            asyncio.run_coroutine_threadsafe(_stop(), self.loop).result(5)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(5)
            self.loop.close()


# ── Deck setup ────────────────────────────────────────────────────────────


class FakeContainer:
    def __init__(self, store: list, name: str, owner: str, web_auth: str, port: int,
                 instance: str | None = None, status: str = "running"):
        self._store = store
        self.name, self.id, self.short_id, self.status = name, f"full-{name}", f"sid-{name}", status
        self.labels = {server.CONTAINER_LABEL: "true", server.OWNER_LABEL: owner,
                       "flight-deck.agent-name": name, "flight-deck.description": "",
                       "flight-deck.web-auth": web_auth, "flight-deck.web-port": str(port)}
        if instance:
            self.labels[sharing.INSTANCE_LABEL] = instance
        self.attrs = {"Created": "", "NetworkSettings": {"Ports": {}}}
        self.image = SimpleNamespace(tags=["kstevica/captain-claw:latest"], short_id="img")

    def remove(self, force: bool = False) -> None:
        self._store.remove(self)


async def _add_users(db: FlightDeckDB) -> None:
    for uid, role, name in USERS:
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", name, role))
    await db._db.commit()


def _patch_deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, helper_port: int) -> SimpleNamespace:
    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setattr(sharing, "SHARING_ENABLED", True)
    monkeypatch.setattr(sharing, "_SOCKETS", {})
    monkeypatch.setattr(sharing, "_MEMBER_CACHE", {})

    running = {"helper"}
    monkeypatch.setattr(server, "_process_is_alive", lambda slug: slug in running)
    killed: list = []
    monkeypatch.setattr(server, "_kill_pid", lambda pid, timeout=5.0: killed.append(pid))

    def _no_docker():
        raise RuntimeError("docker unavailable in tests")

    containers: list = []
    monkeypatch.setattr(server, "get_docker", _no_docker)
    monkeypatch.setattr(server, "_deck_containers", lambda all=False, client=None: list(containers))
    containers.append(FakeContainer(containers, "helper", OTHER, "dock-tok", 24990,
                                    instance=DOCKER_INST))

    def entry(slug, tok, owner, inst=None, **kw):
        e = {"slug": slug, "name": kw.pop("name", slug.title()), "description": kw.pop("description", ""),
             "web_port": kw.pop("port", 24901), "web_auth": tok, "owner": owner, "pid": None}
        if inst:
            e["instance_id"] = inst
        return e

    registry = {
        "helper": entry("helper", HELPER_TOK, OWNER, INST, name="Helper", port=helper_port,
                        description="Answers questions"),
        "legacy": entry("legacy", "legacy-tok", OWNER),
        "basna-1a2b3c4d-worker": entry("basna-1a2b3c4d-worker", "w-tok", OWNER, "1111111111111111"),
        "orphan": entry("orphan", "o-tok", "", "2222222222222222"),
        "notoken": entry("notoken", "", OWNER, "3333333333333333"),
        "sleepy": entry("sleepy", "sleepy-tok", OWNER, "4444444444444444"),
        "others": entry("others", "x-tok", OTHER, "5555555555555555"),
    }
    server._save_process_registry(registry)
    for slug in registry:
        (data / slug).mkdir()
    return SimpleNamespace(containers=containers, running=running, killed=killed, data=data)


def _ref(slug: str, inst: str) -> str:
    return f"process:{slug}:{inst}"


def _hdr(uid: str) -> dict:
    return {"Authorization": f"Bearer {create_access_token(uid)}"}


def _token(uid: str, exp_in: int) -> str:
    now = int(time.time())
    return jwt.encode({"sub": uid, "role": "user", "iat": now, "exp": now + exp_in,
                       "type": "access"}, get_jwt_secret(), algorithm="HS256")


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    await _add_users(db)
    env = _patch_deck(tmp_path, monkeypatch, 24987)
    env.db = db
    try:
        yield env
    finally:
        await db.close()
        fd_auth._db = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=REMOTE), base_url="http://fd.test")


# ── Unit: identifiers ─────────────────────────────────────────────────────


class TestIdentifiers:
    def test_ref_round_trip(self):
        assert sharing.format_ref("process", "helper", INST) == REF
        assert sharing.parse_ref(REF) == ("process", "helper", INST)
        assert sharing.parse_ref(DOCKER_REF) == ("docker", "helper", DOCKER_INST)
        rec = sharing.AgentRecord("docker", "a-1", DOCKER_INST, "o", "n", "", 1, "t", True)
        assert sharing.parse_ref(rec.ref) == ("docker", "a-1", DOCKER_INST)

    @pytest.mark.parametrize("bad", [
        "lambda:helper:0123456789abcdef",       # runtime
        "process:Helper:0123456789abcdef",      # uppercase slug
        "process:helper:0123456789ABCDEF",      # uppercase instance
        "process:helper:0123456789abcde",       # 15 hex
        "process:helper:0123456789abcdef0",     # 17 hex
        "process:-helper:0123456789abcdef",     # slug must start alnum
        "process:helper:0123456789abcdef\n",    # trailing newline
        "process::0123456789abcdef", "", "process:helper",
    ])
    def test_parse_ref_rejects(self, bad):
        with pytest.raises(ValueError):
            sharing.parse_ref(bad)

    def test_parse_ref_rejects_non_strings(self):
        with pytest.raises(ValueError):
            sharing.parse_ref(None)  # type: ignore[arg-type]

    def test_derived_instance_vector(self):
        assert sharing.derived_instance_id("process", "helper", "test-web-auth") == "af056979c84f8684"
        assert sharing.derived_instance_id("process", "helper", "") == ""
        assert len(sharing.new_instance_id()) == 16
        assert sharing.new_instance_id() != sharing.new_instance_id()

    def test_process_instance_id(self):
        assert sharing.process_instance_id("helper", {"instance_id": INST, "web_auth": "x"}) == INST
        derived = sharing.derived_instance_id("process", "helper", "test-web-auth")
        assert sharing.process_instance_id("helper", {"web_auth": "test-web-auth"}) == derived
        # a malformed stored id is ignored, not trusted
        assert sharing.process_instance_id("helper", {"instance_id": "NOPE",
                                                      "web_auth": "test-web-auth"}) == derived
        assert sharing.process_instance_id("helper", {}) == ""

    def test_process_instance_for_spawn(self):
        prior = {"slug": "helper", "owner": OWNER, "instance_id": INST, "web_auth": "a"}
        assert sharing.process_instance_for_spawn(prior, OWNER, "helper") == INST
        legacy = {"slug": "helper", "owner": OWNER, "web_auth": "test-web-auth"}
        assert sharing.process_instance_for_spawn(legacy, OWNER) == "af056979c84f8684"
        fresh = sharing.process_instance_for_spawn(prior, OTHER, "helper")
        assert fresh != INST and len(fresh) == 16
        assert sharing.process_instance_for_spawn({}, OWNER, "helper") not in ("", INST)

    @pytest.mark.parametrize("slug,description,managed", [
        ("basna-1a2b3c4d-x", "", True),
        ("vatra-1a2b3c4d-lead", "", True),
        ("council-abcdef-y", "", True),
        ("iskra-foo-1a2b", "", True),
        ("dubina-x", "Dubina ephemeral worker", True),
        ("dubina-x", "my research buddy", False),
        ("council-notes", "", False),
        ("council notes", "", False),
        ("basna-1a2b3c4-x", "", False),
        ("iskra-foo-1a2b-extra", "", False),
        ("helper", "", False),
    ])
    def test_is_managed_agent_parity(self, slug, description, managed):
        assert sharing.is_managed_agent(slug, description) is managed

    def test_check_shareable(self):
        def rec(**kw):
            base = dict(runtime="process", slug="helper", instance=INST, owner=OWNER, name="Helper",
                        description="", port=1, web_auth="t", running=True)
            base.update(kw)
            return sharing.AgentRecord(**base)

        assert sharing.check_shareable(rec(), OWNER) is None
        assert sharing.check_shareable(None, OWNER) == "Agent not found"
        assert sharing.check_shareable(rec(), OTHER) == "Agent not found"
        assert sharing.check_shareable(rec(owner=""), "") == "Unowned agents can't be shared"
        assert sharing.check_shareable(rec(slug="basna-1a2b3c4d-w"), OWNER) == (
            "Flight Deck–managed workers can't be shared")
        assert sharing.check_shareable(rec(web_auth=""), OWNER) == (
            "This agent has no access token; respawn it to share it")


class TestSpeakerAssertion:
    def test_contract_vector(self):
        assert sharing._speaker_key("test-web-auth").hex() == VECTOR_KEY
        header = sharing.sign_speaker_assertion("test-web-auth", dict(VECTOR_PAYLOAD))
        assert header == VECTOR_HEADER
        assert sharing.speaker_ack_for(header) == VECTOR_ACK

    def test_agent_side_verification_accepts_vector_once(self):
        seen: set = set()
        assert verify_speaker(VECTOR_HEADER, "test-web-auth", seen, now=1760000010) == VECTOR_PAYLOAD
        assert verify_speaker(VECTOR_HEADER, "test-web-auth", seen, now=1760000010) is None  # replay
        assert verify_speaker(VECTOR_HEADER, "other-token", set(), now=1760000010) is None
        assert verify_speaker(VECTOR_HEADER, "test-web-auth", set(), now=1760000061) is None

    def test_payload_shape(self):
        p = sharing.speaker_payload(speaker_id=MEMBER, name="Mia", owner=OWNER, owner_name="Olga",
                                    ref=REF, lane="B", conn="c0ffee00c0ffee00", now=1000)
        assert p["exp"] - p["iat"] == 60 and p["v"] == 1 and len(p["nonce"]) == 16
        assert {k for k in p} == {"v", "sub", "name", "owner", "owner_name", "ref", "lane",
                                  "conn", "iat", "exp", "nonce"}

    def test_refuses_without_token(self):
        with pytest.raises(ValueError):
            sharing.sign_speaker_assertion("", dict(VECTOR_PAYLOAD))


class TestResolve:
    async def test_same_slug_different_owner_and_runtime(self, deck):
        proc = sharing.resolve_agent_record(REF)
        dock = sharing.resolve_agent_record(DOCKER_REF)
        assert proc is not None and dock is not None
        assert (proc.runtime, proc.owner, proc.web_auth, proc.port, proc.running) == (
            "process", OWNER, HELPER_TOK, 24987, True)
        assert (dock.runtime, dock.owner, dock.web_auth, dock.port, dock.running) == (
            "docker", OTHER, "dock-tok", 24990, True)
        assert proc.ref == REF and dock.ref == DOCKER_REF

    async def test_stale_or_crossed_instance(self, deck):
        assert sharing.resolve_agent_record("process:helper:aaaaaaaaaaaaaaaa") is None
        assert sharing.resolve_agent_record(f"process:helper:{DOCKER_INST}") is None
        assert sharing.resolve_agent_record(f"docker:helper:{INST}") is None
        assert sharing.resolve_agent_record("process:nobody:aaaaaaaaaaaaaaaa") is None
        assert sharing.resolve_agent_record("garbage") is None

    async def test_legacy_entry_resolves_by_derived_id(self, deck):
        ref = _ref("legacy", sharing.derived_instance_id("process", "legacy", "legacy-tok"))
        rec = sharing.resolve_agent_record(ref)
        assert rec is not None and rec.slug == "legacy" and rec.running is False

    async def test_docker_label_absent_uses_derived(self, deck):
        deck.containers.append(FakeContainer(deck.containers, "dockless", OWNER, "dl-tok", 24991))
        inst = sharing.derived_instance_id("docker", "dockless", "dl-tok")
        rec = sharing.resolve_agent_record(f"docker:dockless:{inst}")
        assert rec is not None and rec.owner == OWNER

    async def test_ensure_process_instance_persisted(self, deck):
        derived = sharing.derived_instance_id("process", "legacy", "legacy-tok")
        sharing.ensure_process_instance_persisted("legacy")
        assert server._load_process_registry()["legacy"]["instance_id"] == derived
        sharing.ensure_process_instance_persisted("helper")  # stored id untouched
        assert server._load_process_registry()["helper"]["instance_id"] == INST


class _CountingDB:
    def __init__(self):
        self.users = {MEMBER}
        self.members = {(REF, OWNER, MEMBER)}
        self.calls = 0

    async def get_user_by_id(self, uid):
        return {"id": uid} if uid in self.users else None

    async def is_agent_member(self, ref, owner, uid):
        self.calls += 1
        return (ref, owner, uid) in self.members


class TestMemberCheck:
    async def test_caches_true_within_max_age(self, monkeypatch):
        monkeypatch.setattr(sharing, "_MEMBER_CACHE", {})
        db = _CountingDB()
        assert await sharing.member_check(db, REF, OWNER, MEMBER, max_age=0.2)
        assert await sharing.member_check(db, REF, OWNER, MEMBER, max_age=0.2)
        assert db.calls == 1
        await asyncio.sleep(0.25)
        assert await sharing.member_check(db, REF, OWNER, MEMBER, max_age=0.2)
        assert db.calls == 2
        # max_age=0 always asks
        assert await sharing.member_check(db, REF, OWNER, MEMBER, max_age=0)
        assert db.calls == 3

    async def test_never_caches_false(self, monkeypatch):
        monkeypatch.setattr(sharing, "_MEMBER_CACHE", {})
        db = _CountingDB()
        db.members.clear()
        for _ in range(3):
            assert not await sharing.member_check(db, REF, OWNER, MEMBER)
        assert db.calls == 3
        db.users.clear()
        assert not await sharing.member_check(db, REF, OWNER, MEMBER)

    async def test_invalidate(self, monkeypatch):
        monkeypatch.setattr(sharing, "_MEMBER_CACHE", {})
        db = _CountingDB()
        assert await sharing.member_check(db, REF, OWNER, MEMBER)
        db.members.clear()
        assert await sharing.member_check(db, REF, OWNER, MEMBER)  # cached
        sharing.invalidate_member_cache(REF, MEMBER)
        assert not await sharing.member_check(db, REF, OWNER, MEMBER)

    async def test_fails_closed(self, monkeypatch):
        monkeypatch.setattr(sharing, "_MEMBER_CACHE", {})

        class Broken(_CountingDB):
            async def is_agent_member(self, *a):
                raise RuntimeError("db down")

        assert not await sharing.member_check(Broken(), REF, OWNER, MEMBER)
        assert not await sharing.member_check(_CountingDB(), REF, "", MEMBER)


class _SlowBrowserWS:
    """A browser socket whose sends take a moment (a real one can yield)."""

    def __init__(self):
        self.sent: list[dict] = []
        self.closed_with: tuple | None = None

    async def send_text(self, text: str) -> None:
        await asyncio.sleep(0.05)
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await asyncio.sleep(0.05)
        self.closed_with = (code, reason)


class TestMemberConnClose:
    async def test_a_close_from_another_task_reaches_the_browser_first(self):
        # DELETE / leave / agent removal close the socket from their own task.
        # The socket's handler must not return (→ an abrupt 1006) before that
        # close frame is out: neither the watchdog's wake-up (`done`) nor the
        # handler's final wait may run ahead of it.
        from captain_claw.flight_deck import agent_sharing_routes as routes

        ws = _SlowBrowserWS()
        conn = routes._MemberConn(ws)
        closing = asyncio.create_task(conn.close(4403, "Access removed"))
        await asyncio.sleep(0)  # close() is now mid-send
        assert conn.closed and not conn.done.is_set()
        await conn.wait_client_closed()
        assert ws.closed_with == (4403, "Access removed")
        assert ws.sent == [{"type": "fd_close", "code": 4403, "reason": "Access removed"}]
        assert conn.done.is_set()
        await closing

    async def test_browser_gone_needs_no_wait(self):
        from captain_claw.flight_deck import agent_sharing_routes as routes

        ws = _SlowBrowserWS()
        conn = routes._MemberConn(ws)
        await conn.shutdown()
        await asyncio.wait_for(conn.wait_client_closed(), timeout=0.01)
        assert ws.sent == [] and ws.closed_with is None
        await conn.close(4403, "late")  # already closed: nothing more is sent
        assert ws.sent == [] and ws.closed_with is None


class TestSocketRegistry:
    async def test_register_count_close(self, monkeypatch):
        monkeypatch.setattr(sharing, "_SOCKETS", {})
        closed: list = []

        def closer(tag):
            async def _close(code, reason):
                closed.append((tag, code, reason))
            return _close

        a = sharing.register_member_socket(REF, MEMBER, "A", closer("a"))
        sharing.register_member_socket(REF, MEMBER, "B", closer("b"))
        sharing.register_member_socket(REF, OTHER, "A", closer("c"))
        sharing.register_member_socket(DOCKER_REF, MEMBER, "A", closer("d"))
        assert len(a) == 16 and sharing.live_conn_count(REF, MEMBER) == 2
        assert await sharing.close_member_sockets(REF, MEMBER) == 2
        assert sorted(closed) == [("a", 4403, "Access removed"), ("b", 4403, "Access removed")]
        assert sharing.live_conn_count(REF, MEMBER) == 0
        assert await sharing.close_member_sockets(REF, code=4404, reason="Agent removed") == 1
        assert closed[-1] == ("c", 4404, "Agent removed")
        sharing.unregister_member_socket(a)  # already gone: no error
        assert sharing.live_conn_count(DOCKER_REF, MEMBER) == 1


# ── Speaker profile block ─────────────────────────────────────────────────


class TestSpeakerProfile:
    async def test_member_block(self, deck):
        await tp.save_profile(deck.db, MEMBER, {"about_me": "I design bridges.",
                                                "company": "", "instructions": "Be brief."})
        await tp.save_profile(deck.db, OWNER, {"about_me": "OWNER SECRET", "company": "",
                                               "instructions": "OWNER PREFS"})
        await tp.save_deck(deck.db, {"company": "Acme", "instructions": "Use metric."})
        full, compact = await tp.compose_for_speaker(deck.db, MEMBER, OWNER)
        assert full.startswith("## Who you are talking to\n")
        assert ("You are talking with Mia Member, a Flight Deck user Olga Owner shared this "
                "agent with — not your owner.") in full
        assert "### About them\n> I design bridges." in full
        assert "### About their company\n> Acme" in full
        assert "From the person you're talking to:\n> Be brief." in full
        assert tp.DECK_PREFS_LABEL + "\n> Use metric." in full
        assert "OWNER" not in full and "OWNER" not in compact
        assert tp.OWNER_PREFS_LABEL not in full and tp.PRIVATE_LINE not in full
        assert "anyone but Mia Member" in full
        assert compact.startswith("## Who you are talking to\n") and len(compact) <= tp.COMPACT_MAX
        assert "About: I design bridges." in compact

    async def test_compact_is_capped(self, deck):
        await tp.save_profile(deck.db, MEMBER, {"about_me": "x" * 1500, "company": "y" * 4000,
                                                "instructions": "z" * 2000})
        full, compact = await tp.compose_for_speaker(deck.db, MEMBER, OWNER)
        assert len(compact) <= tp.COMPACT_MAX and "…" in compact
        assert full

    async def test_nothing_to_say(self, deck):
        assert await tp.compose_for_speaker(deck.db, MEMBER, OWNER) == ("", "")
        assert await tp.compose_for_speaker(deck.db, "u-ghost", OWNER) == ("", "")
        assert await tp.compose_for_speaker(deck.db, "", OWNER) == ("", "")


# ── Routes ────────────────────────────────────────────────────────────────


def _share_body(ref: str = REF, grantee: str = MEMBER, perm: str = "edit") -> dict:
    return {"resource_type": "agent", "resource_id": ref, "grantee_id": grantee, "permission": perm}


class TestFlagOff:
    async def test_everything_off(self, deck, monkeypatch):
        monkeypatch.setattr(sharing, "SHARING_ENABLED", False)
        async with _client() as c:
            r = await c.post("/fd/shares", json=_share_body(), headers=_hdr(OWNER))
            assert r.status_code == 400
            r = await c.get("/fd/shares", params={"resource_type": "agent", "resource_id": REF},
                            headers=_hdr(OWNER))
            assert r.status_code == 400
            r = await c.get("/fd/shared-agents", headers=_hdr(MEMBER))
            assert r.json() == {"enabled": False, "host_warning": "", "agents": []}
        assert await deck.db.list_agent_members(REF, OWNER) == []

    async def test_auth_off_means_off(self, deck, monkeypatch):
        monkeypatch.setattr(server, "AUTH_ENABLED", False)
        assert not sharing.sharing_active()


class TestShareCreate:
    async def test_chat_only_notified_and_persisted(self, deck):
        legacy_ref = _ref("legacy", sharing.derived_instance_id("process", "legacy", "legacy-tok"))
        assert "instance_id" not in server._load_process_registry()["legacy"]
        async with _client() as c:
            r = await c.post("/fd/shares", json=_share_body(legacy_ref), headers=_hdr(OWNER))
            assert r.status_code == 200, r.text
            assert r.json()["share"]["permission"] == "view"
            r = await c.get("/fd/shares", params={"resource_type": "agent",
                                                  "resource_id": legacy_ref}, headers=_hdr(OWNER))
            assert [s["grantee_id"] for s in r.json()["shares"]] == [MEMBER]
        entry = server._load_process_registry()["legacy"]
        assert entry["instance_id"] == legacy_ref.rsplit(":", 1)[1]
        assert await deck.db.is_agent_member(legacy_ref, OWNER, MEMBER)
        notes = await deck.db.list_notifications(MEMBER)
        assert notes[0]["title"] == "Olga Owner shared the agent “Legacy” with you"
        assert (notes[0]["body"], notes[0]["ref_type"], notes[0]["ref_id"]) == (
            "Legacy", "agent", legacy_ref)

    @pytest.mark.parametrize("slug,inst,owner,status,detail", [
        ("orphan", "2222222222222222", OWNER, 400, "Unowned agents can't be shared"),
        ("basna-1a2b3c4d-worker", "1111111111111111", OWNER, 400,
         "Flight Deck–managed workers can't be shared"),
        ("notoken", "3333333333333333", OWNER, 400,
         "This agent has no access token; respawn it to share it"),
        ("others", "5555555555555555", OWNER, 404, None),         # not yours
        ("helper", "aaaaaaaaaaaaaaaa", OWNER, 404, None),          # stale instance
        ("helper", INST, MEMBER, 404, None),                       # member can't re-share
    ])
    async def test_refusals(self, deck, slug, inst, owner, status, detail):
        grantee = OTHER if owner == MEMBER else MEMBER
        async with _client() as c:
            r = await c.post("/fd/shares", json=_share_body(_ref(slug, inst), grantee),
                             headers=_hdr(owner))
        assert r.status_code == status, r.text
        if detail:
            assert r.json()["detail"] == detail
        assert await deck.db.list_shares_for_grantee(grantee, "agent") == []

    async def test_docker_agent(self, deck):
        async with _client() as c:
            r = await c.post("/fd/shares", json=_share_body(DOCKER_REF), headers=_hdr(OTHER))
        assert r.status_code == 200, r.text
        assert await deck.db.is_agent_member(DOCKER_REF, OTHER, MEMBER)


class TestSharedAgentsList:
    async def test_lists_only_live_owner_matching(self, deck):
        db = deck.db
        await db.create_share("agent", REF, OWNER, MEMBER, "view")
        await db.create_share("agent", DOCKER_REF, OTHER, MEMBER, "view")
        await db.create_share("agent", "process:helper:aaaaaaaaaaaaaaaa", OWNER, MEMBER, "view")
        await db.create_share("agent", REF, OTHER, MEMBER, "view")  # not OTHER's agent
        await db.create_share("agent", _ref("basna-1a2b3c4d-worker", "1111111111111111"),
                              OWNER, MEMBER, "view")
        await db.create_share("agent", _ref("sleepy", "4444444444444444"), OWNER, MEMBER, "view")
        async with _client() as c:
            r = await c.get("/fd/shared-agents", headers=_hdr(MEMBER))
            mine = await c.get("/fd/shared-agents", headers=_hdr(OTHER))
        body = r.json()
        assert body["enabled"] is True and body["host_warning"] == sharing.HOST_TRUST_WARNING
        rows = {a["agent_ref"]: a for a in body["agents"]}
        assert set(rows) == {REF, DOCKER_REF, _ref("sleepy", "4444444444444444")}
        helper = rows[REF]
        assert helper == {
            "agent_ref": REF, "runtime": "process", "slug": "helper", "name": "Helper",
            "description": "Answers questions", "status": "running", "owner_id": OWNER,
            "owner_name": "Olga Owner", "owner_email": f"{OWNER}@x.co",
            "shared_at": helper["shared_at"]}
        assert helper["shared_at"]
        assert rows[_ref("sleepy", "4444444444444444")]["status"] == "stopped"
        for row in body["agents"]:
            assert not {"web_auth", "web_port", "port", "host", "pid", "container_id"} & set(row)
        for secret in (HELPER_TOK, "dock-tok", "sleepy-tok", "24987", "24990"):
            assert secret not in r.text
        assert mine.json()["agents"] == []


class TestRevocationRoutes:
    @pytest.fixture
    def spy(self, monkeypatch):
        calls: list = []

        async def fake_close(ref, user_id=None, *, code=4403, reason="Access removed"):
            calls.append((ref, user_id, code, reason))
            return 0

        monkeypatch.setattr(sharing, "close_member_sockets", fake_close)
        return calls

    async def test_delete_closes_member_sockets(self, deck, spy):
        await deck.db.create_share("agent", REF, OWNER, MEMBER, "view")
        async with _client() as c:
            r = await c.delete("/fd/shares", params={"resource_type": "agent", "resource_id": REF,
                                                     "grantee_id": MEMBER}, headers=_hdr(OWNER))
            assert r.json() == {"ok": True}
            # nothing deleted → nobody's sockets touched
            r = await c.delete("/fd/shares", params={"resource_type": "agent", "resource_id": REF,
                                                     "grantee_id": MEMBER}, headers=_hdr(OTHER))
            assert r.json() == {"ok": False}
        assert spy == [(REF, MEMBER, 4403, "Access removed")]
        notes = await deck.db.list_notifications(MEMBER)
        assert notes[0]["title"] == "Olga Owner stopped sharing “Helper” with you"
        assert (notes[0]["type"], notes[0]["ref_type"]) == ("share", "agent")

    async def test_leave_closes_member_sockets(self, deck, spy):
        await deck.db.create_share("agent", REF, OWNER, MEMBER, "view")
        async with _client() as c:
            r = await c.delete("/fd/shares/leave", params={"resource_type": "agent",
                                                           "resource_id": REF, "owner_id": OWNER},
                               headers=_hdr(MEMBER))
        assert r.json() == {"ok": True}
        assert spy == [(REF, MEMBER, 4403, "You left this shared agent")]
        assert await deck.db.list_notifications(MEMBER) == []
        assert not await deck.db.is_agent_member(REF, OWNER, MEMBER)

    async def test_remove_process_purges_and_closes(self, deck, spy):
        await deck.db.create_share("agent", REF, OWNER, MEMBER, "view")
        await deck.db.create_share("agent", REF, OWNER, OTHER, "view")
        async with _client() as c:
            r = await c.delete("/fd/processes/helper", headers=_hdr(OWNER))
        assert r.status_code == 200, r.text
        assert await deck.db.list_agent_members(REF, OWNER) == []
        assert spy == [(REF, None, 4404, "Agent removed")]
        assert deck.killed == []

    async def test_remove_container_purges_and_closes(self, deck, spy):
        await deck.db.create_share("agent", DOCKER_REF, OTHER, MEMBER, "view")
        async with _client() as c:
            r = await c.delete("/fd/containers/sid-helper", headers=_hdr(OTHER))
        assert r.status_code == 200, r.text
        assert await deck.db.list_agent_members(DOCKER_REF, OTHER) == []
        assert spy == [(DOCKER_REF, None, 4404, "Agent removed")]


class TestAgentRefRows:
    async def test_processes_and_containers_carry_agent_ref(self, deck):
        async with _client() as c:
            procs = (await c.get("/fd/processes", headers=_hdr(OWNER))).json()
            conts = (await c.get("/fd/containers", headers=_hdr(OTHER))).json()
        refs = {p["slug"]: p["agent_ref"] for p in procs}
        assert refs["helper"] == REF
        assert refs["legacy"] == _ref(
            "legacy", sharing.derived_instance_id("process", "legacy", "legacy-tok"))
        assert refs["sleepy"] == _ref("sleepy", "4444444444444444")
        assert [c_["agent_ref"] for c_ in conts] == [DOCKER_REF]

    async def test_spawn_and_clone_instance_ids(self, deck):
        # A respawn of one's own agent keeps its id; a clone gets a fresh one.
        registry = server._load_process_registry()
        prior = registry["helper"]
        assert sharing.process_instance_for_spawn(prior, OWNER, "helper") == INST
        async with _client() as c:
            r = await c.post("/fd/processes/sleepy/clone", json={"new_name": "Sleepy Two"},
                             headers=_hdr(OWNER))
        assert r.status_code == 200, r.text
        clone = server._load_process_registry()["sleepy-two"]
        assert len(clone["instance_id"]) == 16 and clone["instance_id"] != "4444444444444444"

    async def test_process_spawn_keeps_own_respawn_id(self, deck, monkeypatch):
        from captain_claw.flight_deck import rate_limiter

        class _FakePopen:
            def __init__(self, args, cwd=None, env=None, stdout=None, stderr=None,
                         start_new_session=None):
                self.pid = 434343
                if stdout is not None:
                    stdout.close()

            def poll(self):
                return None

        monkeypatch.setattr(server.subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(server, "_is_port_available", lambda port: True)
        monkeypatch.setattr(server, "_schedule_fleet_notify", lambda *a, **k: None)

        async def _no_cap(user, count):
            return None

        monkeypatch.setattr(server, "check_agent_count_limit", _no_cap)
        monkeypatch.setattr(rate_limiter, "_spawn_limiter", rate_limiter._SlidingWindow())
        monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
        monkeypatch.setenv("FD_SPAWN_SETTLE_S", "0")
        monkeypatch.setenv("FD_PORT", "25999")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fd-env-key")
        async with _client() as c:
            r = await c.post("/fd/spawn-process", headers=_hdr(OWNER),
                             json={"name": "Sleepy", "web_port": 24310})
            assert r.status_code == 200, r.text
            r = await c.post("/fd/spawn-process", headers=_hdr(OWNER),
                             json={"name": "Brand New", "web_port": 24311})
            assert r.status_code == 200, r.text
        registry = server._load_process_registry()
        assert registry["sleepy"]["instance_id"] == "4444444444444444"
        assert len(registry["brand-new"]["instance_id"]) == 16

    async def test_docker_rebuild_keeps_and_clone_renews(self, deck, monkeypatch):
        import docker as docker_mod

        runs: list[dict] = []

        def _get(name):
            raise docker_mod.errors.NotFound("no such container")

        client = SimpleNamespace(
            images=SimpleNamespace(pull=lambda image: None),
            containers=SimpleNamespace(
                run=lambda **kw: runs.append(kw) or SimpleNamespace(short_id="sid-new"),
                get=_get, list=lambda **kw: list(deck.containers)))
        monkeypatch.setattr(server, "get_docker", lambda: client)
        legacy = FakeContainer(deck.containers, "dockless", OWNER, "dl-tok", 24991)
        deck.containers.append(legacy)
        for c_ in deck.containers:
            c_.stop = lambda timeout=5: None
        async with _client() as c:
            r = await c.post("/fd/containers/sid-helper/rebuild", json={}, headers=_hdr(OTHER))
            assert r.status_code == 200, r.text
            r = await c.post("/fd/containers/sid-dockless/rebuild", json={}, headers=_hdr(OWNER))
            assert r.status_code == 200, r.text
        assert runs[0]["labels"][sharing.INSTANCE_LABEL] == DOCKER_INST  # same agent, same ref
        assert runs[1]["labels"][sharing.INSTANCE_LABEL] == sharing.derived_instance_id(
            "docker", "dockless", "dl-tok")  # legacy: its derived id is pinned

        deck.containers.append(FakeContainer(deck.containers, "helper", OTHER, "dock-tok", 24990,
                                             instance=DOCKER_INST))
        async with _client() as c:
            r = await c.post("/fd/containers/sid-helper/clone", json={"new_name": "Helper Two"},
                             headers=_hdr(OTHER))
        assert r.status_code == 200, r.text
        inst = runs[-1]["labels"][sharing.INSTANCE_LABEL]
        assert len(inst) == 16 and inst != DOCKER_INST


class TestSharedChatHistory:
    async def test_per_user_storage(self, deck):
        sid = f"shared:{REF}"
        async with _client() as c:
            for uid in (MEMBER, OTHER):
                r = await c.post("/fd/chat/sessions", json={"id": sid, "agent_id": sid,
                                                            "agent_name": "Helper"},
                                 headers=_hdr(uid))
                assert r.json()["id"] == sid
                r = await c.post(f"/fd/chat/sessions/{sid}/messages",
                                 json={"messages": [{"role": "user", "content": f"hi from {uid}"}]},
                                 headers=_hdr(uid))
                assert r.status_code == 200, r.text
            rows = await deck.db.list_chat_sessions(MEMBER)
            assert [row["id"] for row in rows] == [f"{sid}@{MEMBER}"]
            listed = (await c.get("/fd/chat/sessions", headers=_hdr(MEMBER))).json()
            assert [row["id"] for row in listed] == [sid]
            msgs = (await c.get(f"/fd/chat/sessions/{sid}/messages", headers=_hdr(MEMBER))).json()
            assert [m["content"] for m in msgs] == [f"hi from {MEMBER}"]

            # OTHER can't pre-create or write MEMBER's row by naming the suffix
            r = await c.post("/fd/chat/sessions", json={"id": f"{sid}@{MEMBER}"},
                             headers=_hdr(OTHER))
            assert r.status_code == 200
            assert await deck.db.get_chat_session(f"{sid}@{MEMBER}", OTHER) is None
            r = await c.post(f"/fd/chat/sessions/{sid}@{MEMBER}/messages",
                             json={"messages": [{"role": "user", "content": "sneaky"}]},
                             headers=_hdr(OTHER))
            msgs = (await c.get(f"/fd/chat/sessions/{sid}/messages", headers=_hdr(MEMBER))).json()
            assert [m["content"] for m in msgs] == [f"hi from {MEMBER}"]

            r = await c.delete(f"/fd/chat/sessions/{sid}", headers=_hdr(MEMBER))
            assert r.status_code == 200
            assert await deck.db.get_chat_session(f"{sid}@{OTHER}", OTHER) is not None

    async def test_plain_ids_unchanged(self, deck):
        async with _client() as c:
            await c.post("/fd/chat/sessions", json={"id": "proc-helper"}, headers=_hdr(OWNER))
        assert [r["id"] for r in await deck.db.list_chat_sessions(OWNER)] == ["proc-helper"]


# ── Member socket (WS proxy) ──────────────────────────────────────────────


@pytest.fixture
def ws_deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One event loop (a blocking portal) runs FD's requests, member sockets
    and the DB, as uvicorn would; the fake agent runs in its own thread."""
    agent = FakeAgent(HELPER_TOK)
    prev = fd_auth._db
    try:
        with anyio.from_thread.start_blocking_portal() as portal:
            db = FlightDeckDB(tmp_path / "fd.db")
            portal.call(db.init)
            set_auth_db(db)
            portal.call(_add_users, db)
            env = _patch_deck(tmp_path, monkeypatch, agent.port)
            portal.call(db.create_share, "agent", REF, OWNER, MEMBER, "view")
            portal.call(tp.save_profile, db, MEMBER,
                        {"about_me": "I design bridges.", "company": "", "instructions": ""})
            client = TestClient(server.app)
            client.portal = portal
            env.db, env.portal, env.client, env.agent = db, portal, client, agent
            try:
                yield env
            finally:
                portal.call(db.close)
    finally:
        fd_auth._db = prev
        agent.stop()


def _ws_url(ref: str = REF, token: str | None = None, lane: str | None = None,
            extra: str = "") -> str:
    q = f"ref={quote(ref, safe='')}"
    if token is not None:
        q += f"&fd_token={token}"
    if lane is not None:
        q += f"&lane={lane}"
    return f"{WS_FD}/fd/agent-ws-shared?{q}{extra}"


def _recv(sess, timeout: float = 5.0) -> dict:
    async def _r():
        with anyio.fail_after(timeout):
            return await sess._send_rx.receive()

    return sess.portal.call(_r)


def _recv_json(sess, timeout: float = 5.0) -> dict:
    msg = _recv(sess, timeout)
    assert msg["type"] == "websocket.send", msg
    return json.loads(msg["text"])


def _expect_close(sess, code: int, timeout: float = 5.0) -> list[dict]:
    """Frames up to the close; the last one must be the matching fd_close."""
    frames: list[dict] = []
    while True:
        msg = _recv(sess, timeout)
        if msg["type"] == "websocket.close":
            assert msg["code"] == code, (msg, frames)
            assert frames and frames[-1] == {"type": "fd_close", "code": code,
                                             "reason": msg["reason"]}, frames
            return frames
        frames.append(json.loads(msg["text"]))


@contextlib.contextmanager
def _member(env, uid: str = MEMBER, **kw):
    kw.setdefault("token", create_access_token(uid))
    with env.client.websocket_connect(_ws_url(**kw)) as sess:
        yield sess


def _open(env, uid: str = MEMBER, **kw):
    """A member socket past the handshake: returns (session, welcome)."""
    cm = _member(env, uid, **kw)
    sess = cm.__enter__()
    try:
        welcome = _recv_json(sess)
        assert welcome["type"] == "welcome", welcome
    except BaseException:
        # A refused socket must still be torn down, or the portal hangs on exit.
        cm.__exit__(None, None, None)
        raise
    return cm, sess, welcome


class TestMemberSocketRefusals:
    def test_flag_off(self, ws_deck, monkeypatch):
        monkeypatch.setattr(sharing, "SHARING_ENABLED", False)
        with _member(ws_deck) as s:
            assert _expect_close(s, 4503) == [
                {"type": "fd_close", "code": 4503, "reason": "Agent sharing is off on this Flight Deck"}]
        assert ws_deck.agent.requests == []

    @pytest.mark.parametrize("uid,kw,code", [
        (OTHER, {}, 4403),                                  # not a member
        (ADMIN, {}, 4403),                                  # admins get no bypass
        (OWNER, {}, 4400),                                  # the owner uses their own route
        (MEMBER, {"token": "not-a-jwt"}, 4001),
        (MEMBER, {"token": ""}, 4001),
        (MEMBER, {"lane": "D"}, 4400),
        (MEMBER, {"ref": "process:helper"}, 4400),
        (MEMBER, {"ref": "process:helper:aaaaaaaaaaaaaaaa"}, 4404),
    ])
    def test_accept_then_close(self, ws_deck, uid, kw, code):
        with _member(ws_deck, uid, **kw) as s:
            frames = _expect_close(s, code)
        assert len(frames) == 1  # nothing but fd_close
        assert ws_deck.agent.requests == []

    def test_expired_jwt(self, ws_deck):
        with _member(ws_deck, token=_token(MEMBER, -5)) as s:
            _expect_close(s, 4001)

    def test_deleted_user(self, ws_deck):
        tok = create_access_token(MEMBER)
        ws_deck.portal.call(ws_deck.db.delete_user, MEMBER)
        with _member(ws_deck, token=tok) as s:
            _expect_close(s, 4001)

    def test_owner_reason(self, ws_deck):
        with _member(ws_deck, OWNER) as s:
            frames = _expect_close(s, 4400)
        assert frames[-1]["reason"] == "You own this agent — open it from your agents list"

    def test_stopped_agent(self, ws_deck):
        ws_deck.running.discard("helper")
        with _member(ws_deck) as s:
            _expect_close(s, 4409)

    def test_unreachable_agent(self, ws_deck):
        reg = server._load_process_registry()
        reg["helper"]["web_port"] = 1  # nothing listens there
        server._save_process_registry(reg)
        with _member(ws_deck) as s:
            _expect_close(s, 4502)


class TestMemberSocketHandshake:
    def test_upstream_is_recorded_port_and_token(self, ws_deck):
        url_extra = "&token=evil-token&host=evil.example&port=1"
        cm, s, welcome = _open(ws_deck, lane="b", extra=url_extra)
        try:
            req = ws_deck.agent.wait_requests(1)[0]
            assert req["path"] == f"/ws?token={HELPER_TOK}"
            assert req["host"] == f"localhost:{ws_deck.agent.port}"
            p = req["payload"]
            assert p is not None, "the agent could not verify the assertion"
            assert (p["sub"], p["name"], p["owner"], p["owner_name"], p["ref"], p["lane"]) == (
                MEMBER, "Mia Member", OWNER, "Olga Owner", REF, "B")
            assert len(p["conn"]) == 16 and p["exp"] - p["iat"] == 60
            assert welcome["speaker_ack"] == sharing.speaker_ack_for(req["header"])
        finally:
            cm.__exit__(None, None, None)

    def test_no_ack_closes_4426_and_relays_nothing(self, ws_deck):
        ws_deck.agent.mode = "noack"
        with _member(ws_deck) as s:
            frames = _expect_close(s, 4426)
        assert len(frames) == 1

    def test_silent_agent_times_out_4426(self, ws_deck, monkeypatch):
        monkeypatch.setattr(sharing, "ACK_TIMEOUT_S", 0.3)
        ws_deck.agent.mode = "silent"
        with _member(ws_deck) as s:
            assert len(_expect_close(s, 4426)) == 1

    def test_early_frames_dropped_then_welcome(self, ws_deck):
        ws_deck.agent.mode = "early"
        cm, s, welcome = _open(ws_deck)  # first relayed frame IS the welcome
        cm.__exit__(None, None, None)

    def test_agent_capacity_relayed(self, ws_deck):
        ws_deck.agent.mode = "close4429"
        with _member(ws_deck) as s:
            assert len(_expect_close(s, 4429)) == 1

    def test_agent_refusal_maps_to_4502(self, ws_deck):
        ws_deck.agent.mode = "close4401"
        with _member(ws_deck) as s:
            assert len(_expect_close(s, 4502)) == 1

    def test_wrong_key_maps_to_4502(self, ws_deck):
        ws_deck.agent.web_auth = "rotated-token"  # FD's record no longer matches the agent
        with _member(ws_deck) as s:
            _expect_close(s, 4502)

    def test_speaker_name_is_one_bounded_line(self, ws_deck):
        # Display names have no length limit; the assertion's name is one line
        # of at most 120 characters, so the header can't outgrow the agent's limit.
        async def _rename():
            await ws_deck.db._db.execute("UPDATE users SET display_name = ? WHERE id = ?",
                                         ("Mia\n  Member " + "x" * 9000, MEMBER))
            await ws_deck.db._db.commit()

        ws_deck.portal.call(_rename)
        cm, s, _ = _open(ws_deck)
        try:
            p = ws_deck.agent.wait_requests(1)[0]["payload"]
            assert p is not None
            assert p["name"].startswith("Mia Member xxx") and len(p["name"]) == 120
        finally:
            cm.__exit__(None, None, None)

    def test_profile_context_before_any_client_frame(self, ws_deck):
        cm, s, _ = _open(ws_deck)
        try:
            s.send_json({"type": "chat", "content": "hello"})
            frames = ws_deck.agent.wait_frames(2)
            assert frames[0]["type"] == "fd_speaker_context"
            assert set(frames[0]) == {"type", "profile_full", "profile_compact"}
            assert "## Who you are talking to" in frames[0]["profile_full"]
            assert "> I design bridges." in frames[0]["profile_full"]
            assert frames[1]["type"] == "chat"
            assert [f["type"] for f in frames].count("fd_speaker_context") == 1
        finally:
            cm.__exit__(None, None, None)


class TestMemberSocketFrames:
    def test_frame_filter(self, ws_deck):
        cm, s, _ = _open(ws_deck)
        try:
            for ftype in ("run_tool", "peer_agents", "set_model", "notification", "command",
                          "fd_speaker_context", "fd_auth_ok", None):
                s.send_json({"type": ftype, "content": "x", "profile_full": "pwned"})
                assert _recv_json(s) == NOT_ALLOWED
            s.send_text("{not json")
            assert _recv_json(s)["code"] == "invalid"
            s.send_json({"type": "chat", "content": "hi", "origin": "whatsapp",
                         "whatsapp_waid": "385991234567", "image_path": "/etc/passwd",
                         "no_flow": True, "_fd_turn": "aaaaaaaaaaaaaaaa", "no_next_steps": True,
                         "rewind_to": "2026-10-05T10:00:00", "no_rephrase": "yes"})
            s.send_json({"type": "cancel", "content": "extra"})
            frames = ws_deck.agent.wait_frames(3)
            assert [f["type"] for f in frames] == ["fd_speaker_context", "chat", "cancel"]
            chat = frames[1]
            turn = chat.pop("_fd_turn")
            assert chat == {"type": "chat", "content": "hi", "no_next_steps": True,
                            "rewind_to": "2026-10-05T10:00:00"}
            assert len(turn) == 16 and turn != "aaaaaaaaaaaaaaaa"
            int(turn, 16)
            assert frames[2] == {"type": "cancel"}
        finally:
            cm.__exit__(None, None, None)

    def test_oversized_chat_refused_locally(self, ws_deck, monkeypatch):
        monkeypatch.setattr(sharing, "MAX_CHAT_CONTENT", 10)
        cm, s, _ = _open(ws_deck)
        try:
            s.send_json({"type": "chat", "content": "x" * 11})
            assert _recv_json(s)["code"] == "invalid"
            end = _recv_json(s)
            assert end["type"] == "status" and end["status"] == "ready" and len(end["turn_end"]) == 16
            s.send_json({"type": "chat", "content": "ok"})
            assert [f["type"] for f in ws_deck.agent.wait_frames(2)] == [
                "fd_speaker_context", "chat"]
        finally:
            cm.__exit__(None, None, None)

    def test_oversized_btw_refused_locally(self, ws_deck, monkeypatch):
        # The agent keeps every btw until its next turn ends: it is size-capped here.
        monkeypatch.setattr(sharing, "MAX_CHAT_CONTENT", 10)
        refused = {"type": "error", "code": "invalid",
                   "message": "A note must be text of at most 10 characters"}
        cm, s, _ = _open(ws_deck)
        try:
            s.send_json({"type": "btw", "content": "x" * 11})
            assert _recv_json(s) == refused
            s.send_json({"type": "btw"})
            assert _recv_json(s) == refused
            s.send_json({"type": "btw", "content": ["x"]})
            assert _recv_json(s) == refused
            s.send_json({"type": "btw", "content": "x" * 10})
            s.send_json({"type": "cancel"})
            frames = ws_deck.agent.wait_frames(3)
            assert frames[0]["type"] == "fd_speaker_context"
            assert frames[1:] == [{"type": "btw", "content": "x" * 10}, {"type": "cancel"}]
        finally:
            cm.__exit__(None, None, None)

    def test_btw_is_one_a_second_per_socket(self, ws_deck, monkeypatch):
        assert sharing.BTW_MIN_INTERVAL_S == 1.0
        monkeypatch.setattr(sharing, "BTW_MIN_INTERVAL_S", 0.5)
        cm, s, _ = _open(ws_deck)
        cm2, s2, _ = _open(ws_deck)
        try:
            s.send_json({"type": "btw", "content": "one"})
            s.send_json({"type": "btw", "content": "two"})
            assert _recv_json(s) == {"type": "error", "code": "busy",
                                     "message": "One note a second — try again"}
            s2.send_json({"type": "btw", "content": "other socket"})  # its own window
            notes = [f["content"] for f in ws_deck.agent.wait_frames(4) if f["type"] == "btw"]
            assert sorted(notes) == ["one", "other socket"]
            time.sleep(0.6)
            s.send_json({"type": "btw", "content": "three"})
            notes = [f["content"] for f in ws_deck.agent.wait_frames(5) if f["type"] == "btw"]
            assert sorted(notes) == ["one", "other socket", "three"]
        finally:
            cm2.__exit__(None, None, None)
            cm.__exit__(None, None, None)

    def test_session_settings_are_capped(self, ws_deck):
        # The agent persists these and puts them in every prompt of the session.
        assert set(sharing.SESSION_SETTING_MAX) == set(
            sharing.MEMBER_FRAME_ALLOWLIST["session_settings"]) - {"type"}
        cm, s, _ = _open(ws_deck)
        try:
            for key, cap, label in (("session_name", 200, "Session name"),
                                    ("session_description", 2000, "Session description"),
                                    ("session_instructions", 8000, "Session instructions")):
                s.send_json({"type": "session_settings", "session_name": "fine", key: "x" * (cap + 1)})
                assert _recv_json(s) == {"type": "error", "code": "invalid",
                                         "message": f"{label} must be at most {cap:,} characters"}
            at_cap = {"type": "session_settings", "session_name": "n" * 200,
                      "session_description": "d" * 2000, "session_instructions": "i" * 8000}
            s.send_json(at_cap)
            s.send_json({"type": "session_settings", "session_name": "Plan",
                         "session_description": {"x": "y" * 9000}, "session_instructions": 5})
            frames = ws_deck.agent.wait_frames(3)
            assert frames[0]["type"] == "fd_speaker_context"
            assert frames[1:] == [at_cap, {"type": "session_settings", "session_name": "Plan"}]
        finally:
            cm.__exit__(None, None, None)

    def test_third_concurrent_chat_is_busy(self, ws_deck):
        cm, s, _ = _open(ws_deck)
        try:
            for n in ("one", "two", "three"):
                s.send_json({"type": "chat", "content": n})
            assert _recv_json(s) == {"type": "error", "code": "busy",
                                     "message": "Wait for the current reply"}
            end = _recv_json(s)
            assert end["type"] == "status" and end["status"] == "ready"
            chats = [f for f in ws_deck.agent.wait_frames(3) if f["type"] == "chat"]
            assert [c["content"] for c in chats] == ["one", "two"]
            assert end["turn_end"] not in {c["_fd_turn"] for c in chats}
            # the agent ends turn one (relayed verbatim) → room for another
            done = {"type": "status", "status": "ready", "turn_end": chats[0]["_fd_turn"]}
            ws_deck.agent.push(0, done)
            assert _recv_json(s) == done
            s.send_json({"type": "chat", "content": "four"})
            chats = [f for f in ws_deck.agent.wait_frames(4) if f["type"] == "chat"]
            assert [c["content"] for c in chats] == ["one", "two", "four"]
        finally:
            cm.__exit__(None, None, None)

    def test_agent_frames_relayed_verbatim(self, ws_deck):
        cm, s, _ = _open(ws_deck)
        try:
            frame = {"type": "chat_message", "role": "assistant", "content": "hello",
                     "web_auth": "whatever the agent says"}
            ws_deck.agent.push(0, frame)
            assert _recv_json(s) == frame
        finally:
            cm.__exit__(None, None, None)


class TestMemberSocketRevocation:
    def test_delete_share_closes_immediately(self, ws_deck):
        cm, s, _ = _open(ws_deck)
        try:
            r = ws_deck.client.delete(f"{HTTP_FD}/fd/shares", params={
                "resource_type": "agent", "resource_id": REF, "grantee_id": MEMBER},
                headers=_hdr(OWNER))
            assert r.json() == {"ok": True}
            frames = _expect_close(s, 4403, timeout=3)
            assert frames[-1]["reason"] == "Access removed"
        finally:
            cm.__exit__(None, None, None)
        assert sharing.live_conn_count(REF, MEMBER) == 0

    def test_leave_closes_immediately(self, ws_deck):
        cm, s, _ = _open(ws_deck)
        try:
            ws_deck.client.delete(f"{HTTP_FD}/fd/shares/leave", params={
                "resource_type": "agent", "resource_id": REF, "owner_id": OWNER},
                headers=_hdr(MEMBER))
            assert _expect_close(s, 4403, timeout=3)[-1]["reason"] == "You left this shared agent"
        finally:
            cm.__exit__(None, None, None)

    def test_direct_db_delete_within_a_watchdog_tick(self, ws_deck, monkeypatch):
        monkeypatch.setattr(sharing, "RECHECK_INTERVAL_S", 0.1)
        cm, s, _ = _open(ws_deck)
        try:
            ws_deck.portal.call(ws_deck.db.delete_share, "agent", REF, OWNER, MEMBER)
            _expect_close(s, 4403, timeout=2)
        finally:
            cm.__exit__(None, None, None)

    def test_frame_after_revocation_is_not_forwarded(self, ws_deck):
        cm, s, _ = _open(ws_deck)  # watchdog at its default 10 s: the frame check catches it
        try:
            ws_deck.portal.call(ws_deck.db.delete_share, "agent", REF, OWNER, MEMBER)
            sharing.invalidate_member_cache(REF, MEMBER)
            s.send_json({"type": "chat", "content": "still there?"})
            _expect_close(s, 4403, timeout=3)
            ws_deck.agent.wait_frames(1)
            time.sleep(0.2)  # anything forwarded would have reached the agent by now
            assert [f["type"] for _, f in ws_deck.agent.frames] == ["fd_speaker_context"]
        finally:
            cm.__exit__(None, None, None)

    def test_owner_change_closes(self, ws_deck, monkeypatch):
        monkeypatch.setattr(sharing, "RECHECK_INTERVAL_S", 0.1)
        cm, s, _ = _open(ws_deck)
        try:
            reg = server._load_process_registry()
            reg["helper"]["owner"] = OTHER
            server._save_process_registry(reg)
            _expect_close(s, 4403, timeout=2)
        finally:
            cm.__exit__(None, None, None)

    def test_agent_removed_closes_4404(self, ws_deck):
        cm, s, _ = _open(ws_deck)
        try:
            r = ws_deck.client.delete(f"{HTTP_FD}/fd/processes/helper", headers=_hdr(OWNER))
            assert r.status_code == 200, r.text
            assert _expect_close(s, 4404, timeout=3)[-1]["reason"] == "Agent removed"
        finally:
            cm.__exit__(None, None, None)
        assert ws_deck.portal.call(ws_deck.db.list_agent_members, REF, OWNER) == []
        assert ws_deck.killed == []

    def test_agent_gone_from_records_closes_4404(self, ws_deck, monkeypatch):
        monkeypatch.setattr(sharing, "RECHECK_INTERVAL_S", 0.1)
        cm, s, _ = _open(ws_deck)
        try:
            reg = server._load_process_registry()
            reg.pop("helper")
            server._save_process_registry(reg)
            _expect_close(s, 4404, timeout=2)
        finally:
            cm.__exit__(None, None, None)

    def test_upstream_lost_is_4502(self, ws_deck):
        cm, s, _ = _open(ws_deck)
        try:
            ws_deck.agent.wait_frames(1)
            fut = asyncio.run_coroutine_threadsafe(ws_deck.agent.conns[0].close(), ws_deck.agent.loop)
            fut.result(5)
            assert _expect_close(s, 4502, timeout=3)[-1]["reason"] == "Agent connection lost"
        finally:
            cm.__exit__(None, None, None)


class TestMemberSocketSession:
    def test_jwt_expiry_closes_4001(self, ws_deck, monkeypatch):
        monkeypatch.setattr(sharing, "RECHECK_INTERVAL_S", 0.1)
        monkeypatch.setattr(sharing, "JWT_GRACE_S", 0)
        cm, s, _ = _open(ws_deck, token=_token(MEMBER, 2))
        try:
            frames = _expect_close(s, 4001, timeout=6)
            assert frames[-1]["reason"] == "Session expired"
        finally:
            cm.__exit__(None, None, None)

    def test_fd_auth_extends_and_is_never_forwarded(self, ws_deck, monkeypatch):
        monkeypatch.setattr(sharing, "RECHECK_INTERVAL_S", 0.1)
        monkeypatch.setattr(sharing, "JWT_GRACE_S", 0)
        cm, s, _ = _open(ws_deck, token=_token(MEMBER, 2))
        try:
            fresh = create_access_token(MEMBER)
            s.send_json({"type": "fd_auth", "fd_token": fresh})
            ok = _recv_json(s)
            assert ok["type"] == "fd_auth_ok" and ok["exp"] > time.time() + 60
            time.sleep(2.5)  # past the first token's expiry
            s.send_json({"type": "chat", "content": "still here"})
            frames = ws_deck.agent.wait_frames(2)
            assert [f["type"] for f in frames] == ["fd_speaker_context", "chat"]
        finally:
            cm.__exit__(None, None, None)

    def test_fd_auth_of_another_user_is_refused(self, ws_deck, monkeypatch):
        monkeypatch.setattr(sharing, "RECHECK_INTERVAL_S", 0.1)
        monkeypatch.setattr(sharing, "JWT_GRACE_S", 0)
        cm, s, _ = _open(ws_deck, token=_token(MEMBER, 2))
        try:
            s.send_json({"type": "fd_auth", "fd_token": create_access_token(OTHER)})
            assert _recv_json(s) == {"type": "error", "code": "invalid", "message": "Token refused"}
            _expect_close(s, 4001, timeout=6)
        finally:
            cm.__exit__(None, None, None)

    def test_seventh_socket_is_refused(self, ws_deck):
        opened = []
        try:
            for _ in range(sharing.MAX_MEMBER_SOCKETS_PER_AGENT):
                opened.append(_open(ws_deck)[0])
            assert sharing.live_conn_count(REF, MEMBER) == 6
            with _member(ws_deck) as s:
                assert len(_expect_close(s, 4429)) == 1
            assert len(ws_deck.agent.requests) == 6
            # the cap is per member and agent, whatever the lane
            with _member(ws_deck, lane="C") as s:
                _expect_close(s, 4429)
        finally:
            for cm in opened:
                cm.__exit__(None, None, None)
