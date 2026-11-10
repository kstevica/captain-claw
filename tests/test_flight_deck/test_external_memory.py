"""External memory on Flight Deck (``/fd/external-memory/*``) with Membase as
the provider, through the real routes, service, outbox and provider:

* the connection is per FD user (the deck's with sign-in off), checked with the
  provider before it is saved, sealed at rest, and never returned — only a hint;
  ``external_memory:*`` settings are server-owned;
* what a connection can do comes from the provider's capabilities: a read-only
  key offers no ``remember`` / ``forget``; the agent manifest carries the
  provider's actions and rules plus the person's own;
* an agent acts with its OWNER's connection, resolved from the deck's records;
  mirrored items are queued once and written by the drainer, held (not
  failed) while the key is refused or the provider rate-limits;
* lessons taught in Flight Deck are mirrored from there.

Membase is never contacted: every ``httpx.AsyncClient`` gets a mock transport.
"""

from __future__ import annotations

import json
import types

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import agent_secret, auth, fd_crypto, settings_routes
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.external_memory import routes as emr
from captain_claw.flight_deck.external_memory import service as svc
from captain_claw.flight_deck.external_memory.base import (
    ConnectionInfo,
    MemoryItem,
    ProviderAuthError,
    ProviderError,
    ProviderInputError,
    UnknownAction,
)
from captain_claw.flight_deck.external_memory.membase import MembaseProvider
from captain_claw.flight_deck.external_memory.outbox import MAX_ATTEMPTS, Outbox

ALICE = "user-alice"
BOB = "user-bob"
KEY_BOB = "mbk_bob_0000bbbb1111"
KEY_RO = "mbk_readonly_0000rrrr"
KEY_ALICE = "mbk_alice_0000aaaa2222"

FULL_ACTIONS = [
    "search", "memories.list", "memories.get", "memories.add", "memories.forget",
    "profile_facts.list", "profile_facts.add", "documents.list", "documents.get", "documents.add",
    "documents.delete",
]
RO_ACTIONS = ["search", "memories.list", "memories.get", "documents.list", "documents.get"]


# ── fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
async def db(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for var in ("FD_LOCKDOWN", "FD_AGENT_SHARED_SECRET", "FD_MEMBASE_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    agent_secret.reset_cache_for_tests()
    fd_crypto.reset_for_tests()
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev_db = auth._db
    auth.set_auth_db(d)
    now = "2026-01-01T00:00:00Z"
    for uid, email, role in [(ALICE, "alice@x.co", "admin"), (BOB, "bob@x.co", "user")]:
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, email, "h", uid.title(), role, now, now),
        )
    await d._db.commit()
    box = Outbox(tmp_path / "external_memory.db")
    svc.set_outbox(box)
    try:
        yield d
    finally:
        svc.set_outbox(None)
        await box.close()
        agent_secret.reset_cache_for_tests()
        fd_crypto.reset_for_tests()
        auth.set_auth_db(prev_db)
        await d.close()


@pytest.fixture()
def membase(monkeypatch):
    """A fake Membase behind every ``httpx.AsyncClient``."""
    state = types.SimpleNamespace(requests=[], fail=None, fail_body=None, rules={}, partial=[])

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        key = request.headers.get("authorization", "").removeprefix("Bearer ")
        state.requests.append({
            "method": request.method, "path": request.url.path, "body": body,
            "params": dict(request.url.params), "key": key,
        })
        if state.fail == "network":
            raise httpx.ConnectError("boom", request=request)
        if isinstance(state.fail, int):
            return httpx.Response(state.fail, json=state.fail_body or {"error": {"code": "x", "message": "nope"}})
        if key not in (KEY_BOB, KEY_RO, KEY_ALICE):
            return httpx.Response(403, json={"error": {"code": "unauthorized", "message": "unknown key"}})
        path = request.url.path
        if path == "/v1/capabilities":
            return httpx.Response(200, json={
                "actions": RO_ACTIONS if key == KEY_RO else FULL_ACTIONS,
                "memories": [{"name": "Work", "kind": "text"}, {"name": "Photos", "kind": "media"}],
                "rules": {"citation_required": False, "retention_notice": "", "version": "1", **state.rules},
                "features": [],
            })
        if path == "/v1/search":
            return httpx.Response(200, json={"items": [
                {"type": "memory", "id": "mem_1", "content": "Acme needs the signed NDA before the pilot.",
                 "memory_name": "Work", "score": 0.9},
            ], "partial": state.partial})
        if path == "/v1/memories/add":
            return httpx.Response(202, json={"id": f"doc_n{len(state.requests)}", "kind": "note", "status": "pending"})
        if path == "/v1/documents/add":
            return httpx.Response(202, json={"id": "doc_f1", "kind": "file", "status": "processing"})
        if path.endswith("/forget") or path.endswith("/delete"):
            if request.url.params.get("confirm") != "true":
                return httpx.Response(428, json={"error": {
                    "code": "confirmation_required", "message": "confirm first",
                    "details": {"id": path.split("/")[3], "how": "Ask the person, then pass confirm=true."},
                }})
            return httpx.Response(200, json={"id": path.split("/")[3], "status": "forgotten"})
        if path == "/v1/memories":
            return httpx.Response(200, json={"items": [
                {"id": "mem_9", "content": "Bob prefers short replies.", "memory_name": "Work",
                 "recorded_at": "2026-11-01T10:00:00Z"},
            ], "has_more": False, "partial": []})
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "no"}})

    real_client = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real_client(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return state


@pytest.fixture()
def agents(monkeypatch):
    """This deck's agents: web_auth token → (owner, slug)."""
    import captain_claw.flight_deck.server as srv

    registry = {"tok-alice": (ALICE, "alice-bot"), "tok-bob": (BOB, "bob-bot"), "tok-local": ("", "desk-bot")}

    def find(token):
        if token in registry:
            owner, slug = registry[token]
            return True, owner, slug
        return False, "", ""

    monkeypatch.setattr(srv, "_find_agent_by_auth", find)
    monkeypatch.setattr(srv, "_resolve_agent_identity_by_auth", lambda t: find(t)[:2])
    monkeypatch.setattr(srv, "_load_process_registry", lambda: {
        "bob-bot": {"name": "Bob's Researcher"}, "alice-bot": {"name": "Alice Bot"},
    })
    return registry


def _client():
    app = FastAPI()
    app.include_router(emr.router)
    app.include_router(settings_routes.router)
    return TestClient(app, client=("127.0.0.1", 50123))


def _bearer(uid, role="user"):
    return {"Authorization": f"Bearer {auth.create_access_token(uid, role)}"}


def _connect(c, uid, key):
    return c.put("/fd/external-memory/connection", headers=_bearer(uid),
                 json={"provider": "membase", "credentials": {"api_key": key}})


def _calls(membase, path):
    return [r for r in membase.requests if r["path"] == path]


# ── the provider itself ─────────────────────────────────────────────


class TestMembaseProvider:
    async def test_capabilities_become_connection_info(self, membase):
        membase.rules = {"citation_required": True, "retention_notice": "Kept 90 days."}
        info = await MembaseProvider({"api_key": KEY_BOB}).verify()
        assert info.allowed == sorted(FULL_ACTIONS)
        assert info.reaches == ["Work", "Photos (media)"]
        assert any("requires citations" in r for r in info.rules)
        assert any("Kept 90 days." in r for r in info.rules)

    async def test_a_refused_key_is_an_auth_error_when_verifying(self, membase):
        with pytest.raises(ProviderAuthError):
            await MembaseProvider({"api_key": "mbk_nope_12345678"}).verify()

    async def test_a_403_on_an_action_is_not_an_auth_error(self, membase):
        membase.fail = 403
        p = MembaseProvider({"api_key": KEY_BOB}, ConnectionInfo(allowed=FULL_ACTIONS), max_retries=0)
        with pytest.raises(ProviderError) as exc:
            await p.run("search", {"query": "x"})
        assert not isinstance(exc.value, ProviderAuthError) and "access level" in str(exc.value)

    async def test_a_read_only_key_offers_no_writes(self):
        names = [a.name for a in MembaseProvider({"api_key": KEY_RO}, ConnectionInfo(allowed=RO_ACTIONS)).actions()]
        assert "search" in names and "list" in names and "get" in names
        assert "remember" not in names and "forget" not in names and "add_document" not in names
        with pytest.raises(UnknownAction):
            MembaseProvider({"api_key": KEY_RO}, ConnectionInfo(allowed=RO_ACTIONS)).action("remember")

    async def test_search_names_the_memory_and_a_partial_answer(self, membase):
        membase.partial = [{"memory_name": "Archive", "reason": "error"}]
        p = MembaseProvider({"api_key": KEY_BOB}, ConnectionInfo(allowed=FULL_ACTIONS))
        out = await p.run("search", {"query": "Acme pilot", "limit": 3})
        req = _calls(membase, "/v1/search")[-1]
        assert req["body"] == {"q": "Acme pilot", "limit": 3, "mode": "full"}
        assert "[Work]" in out.text and "(mem_1)" in out.text
        assert "Archive" in out.text and "incomplete" in out.text

    async def test_remember_checks_size_before_sending(self, membase):
        p = MembaseProvider({"api_key": KEY_BOB}, ConnectionInfo(allowed=FULL_ACTIONS))
        with pytest.raises(ProviderInputError):
            await p.run("remember", {"content": "x" * 20_001})
        assert membase.requests == []
        out = await p.run("remember", {"content": "Bob's dentist is Dr. Horvat.", "title": "Dentist"})
        assert _calls(membase, "/v1/memories/add")[-1]["body"] == {
            "content": "Bob's dentist is Dr. Horvat.", "title": "Dentist",
        }
        assert "learned in the background" in out.text

    async def test_forget_needs_confirm_and_routes_by_id(self, membase):
        p = MembaseProvider({"api_key": KEY_BOB}, ConnectionInfo(allowed=FULL_ACTIONS), max_retries=0)
        with pytest.raises(ProviderError) as exc:
            await p.run("forget", {"id": "mem_7"})
        assert exc.value.status == 428 and "Ask the person" in str(exc.value)
        out = await p.run("forget", {"id": "doc_7", "confirm": True})
        req = membase.requests[-1]
        assert req["path"] == "/v1/documents/doc_7/delete" and req["params"] == {"confirm": "true"}
        assert out.text == "Deleted: doc_7."
        with pytest.raises(ProviderInputError):
            await p.run("forget", {"id": "xyz", "confirm": True})

    @pytest.mark.parametrize("code,retryable", [("quota_exceeded", False), ("rate_limited", True)])
    async def test_429s(self, membase, code, retryable):
        membase.fail, membase.fail_body = 429, {"error": {"code": code, "message": "slow down"}}
        p = MembaseProvider({"api_key": KEY_BOB}, ConnectionInfo(allowed=FULL_ACTIONS), max_retries=0)
        with pytest.raises(ProviderError) as exc:
            await p.run("search", {"query": "x"})
        assert exc.value.retryable is retryable

    async def test_mirror_formats(self, membase):
        p = MembaseProvider({"api_key": KEY_BOB})
        rid = await p.mirror(MemoryItem(
            kind="insight", source_id="ab12", content="Bob prefers mornings for calls.",
            tags=["schedule"], agent="Bob's Researcher", created_at="2026-11-06T09:00:00",
            meta={"category": "preference", "why": "He said so twice."},
        ))
        req = _calls(membase, "/v1/memories/add")[-1]
        assert req["body"]["title"] == "Insight · preference"
        assert "Why: He said so twice." in req["body"]["content"]
        assert "Tags: schedule" in req["body"]["content"]
        assert "Bob's Researcher" in req["body"]["content"] and rid.startswith("doc_")
        await p.mirror(MemoryItem(kind="reflection", source_id="2026-11-06T09:00", content="Be brief."))
        doc = _calls(membase, "/v1/documents/add")[-1]["body"]
        assert doc["custom_id"] == "captain-reflection-2026-11-06T09:00"


# ── the connection (user routes) ────────────────────────────────────


class TestConnection:
    async def test_status_without_a_connection(self, db):
        r = _client().get("/fd/external-memory/status", headers=_bearer(BOB))
        assert r.status_code == 200
        body = r.json()
        assert body["connected"] is False and body["scope"] == "user"
        assert [p["name"] for p in body["providers"]] == ["membase", "hindsight", "qdrant", "arcadedb"]
        assert body["memories"] == [] and body["max_memories"] == svc.MAX_MEMORIES
        assert body["default_mirror"] == {"insight": True, "intention": True, "lesson": True,
                                          "reflection": False, "intuition": False}

    async def test_connect_checks_seals_and_never_returns_the_key(self, db, membase):
        c = _client()
        r = _connect(c, BOB, f"  {KEY_BOB} ")
        assert r.status_code == 200, r.text
        body = r.json()
        mem = body["memories"][0]
        assert body["connected"] and mem["verified"] is True and mem["label"] == "Membase"
        assert mem["name"] == "Membase" and mem["source"] == "own" and body["memory_id"] == mem["id"]
        assert mem["credential_hints"] == {"api_key": "mbk_…1111"}
        assert "remember" in [a["name"] for a in mem["actions"]]
        assert mem["reaches"] == ["Work", "Photos (media)"]
        assert KEY_BOB not in r.text
        raw = await db.get_setting(BOB, "external_memory:memories")
        assert KEY_BOB not in raw and "sealed" in json.loads(raw)[0]["credentials"]
        conn = await svc.load_connection(BOB)
        assert conn["credentials"]["api_key"] == KEY_BOB
        assert await svc.load_connection(ALICE) is None

    async def test_a_key_membase_refuses_is_a_400_and_nothing_is_saved(self, db, membase):
        r = _connect(_client(), BOB, "mbk_wrong_00000000")
        assert r.status_code == 400 and "Nothing was saved" in r.json()["detail"]
        assert "mbk_wrong_00000000" not in r.text
        assert await svc.load_connection(BOB) is None

    async def test_an_unreachable_membase_saves_unverified_with_a_warning(self, db, membase):
        membase.fail = "network"
        r = _connect(_client(), BOB, KEY_BOB)
        assert r.status_code == 200
        assert r.json()["memories"][0]["verified"] is False
        assert r.json()["warning"].startswith("Saved, but couldn't check it")

    async def test_bad_input_is_a_422_before_membase(self, db, membase):
        c = _client()
        for creds in ({}, {"api_key": "  "}, {"api_key": "mbk_​x12345678"}):
            r = c.put("/fd/external-memory/connection", headers=_bearer(BOB),
                      json={"provider": "membase", "credentials": creds})
            assert r.status_code == 422
        r = c.put("/fd/external-memory/connection", headers=_bearer(BOB),
                  json={"provider": "nope", "credentials": {"api_key": KEY_BOB}})
        assert r.status_code == 422
        assert membase.requests == []

    async def test_settings_are_server_owned(self, db):
        r = _client().put("/fd/settings", headers=_bearer(BOB),
                          json={"settings": {"external_memory:connection": "{}"}})
        assert r.status_code in (400, 403, 422)

    async def test_disconnect_drops_the_queue(self, db, membase):
        c = _client()
        _connect(c, BOB, KEY_BOB)
        await svc.mirror_for_owner(BOB, MemoryItem(kind="insight", source_id="i1", content="x"))
        assert (await svc.outbox().counts(BOB))["pending"] == 1
        r = c.delete("/fd/external-memory/connection", headers=_bearer(BOB))
        assert r.json()["connected"] is False
        assert (await svc.outbox().counts(BOB))["pending"] == 0

    async def test_mirror_settings(self, db, membase):
        c = _client()
        mid = _connect(c, BOB, KEY_BOB).json()["memory_id"]
        r = c.put(f"/fd/external-memory/memories/{mid}", headers=_bearer(BOB),
                  json={"mirror": {"reflection": True, "insight": False, "bogus": True}})
        mirror = r.json()["memories"][0]["mirror"]
        assert mirror["reflection"] is True and mirror["insight"] is False and mirror["lesson"] is True
        assert "bogus" not in mirror
        assert c.put("/fd/external-memory/memories/m_nope", headers=_bearer(BOB),
                     json={"priority": "first"}).status_code == 404

    async def test_playground_run(self, db, membase):
        c = _client()
        _connect(c, BOB, KEY_BOB)
        r = c.post("/fd/external-memory/run", headers=_bearer(BOB),
                   json={"action": "search", "args": {"query": "Acme"}})
        assert r.status_code == 200 and "NDA" in r.json()["text"]
        logs = await db.get_usage_logs(user_id=BOB, event_type="external_memory_call")
        assert json.loads(logs[0]["detail"])["action"] == "search"


# ── agents ──────────────────────────────────────────────────────────


class TestAgents:
    async def test_manifest_carries_actions_and_rules(self, db, membase, agents):
        c = _client()
        assert c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json() == {
            "connected": False,
        }
        membase.rules = {"retention_notice": "Deleted after a year."}
        _connect(c, BOB, KEY_BOB)
        m = c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json()
        assert m["connected"] and m["provider"] == "membase"
        assert {"remember", "search", "forget"} <= {a["name"] for a in m["actions"]}
        assert any("shared by ALL" in r for r in m["rules"])            # common
        assert any("memory_name" in r for r in m["rules"])              # Membase's own
        assert any("Deleted after a year." in r for r in m["rules"])    # the person's
        assert any(r.startswith("Copied here automatically") for r in m["rules"])
        assert m["mirror"] == ["insight", "intention"]                  # lessons go from FD
        assert KEY_BOB not in json.dumps(m)
        # Alice's agent never sees Bob's memory.
        assert c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-alice"}).json() == {
            "connected": False,
        }

    async def test_agent_runs_with_its_owners_connection(self, db, membase, agents):
        c = _client()
        _connect(c, BOB, KEY_BOB)
        r = c.post("/fd/external-memory/agent/run", headers={"X-Agent-Auth": "tok-bob"},
                   json={"action": "remember", "args": {"content": "Bob's flight is on Friday."}})
        assert r.status_code == 200, r.text
        assert membase.requests[-1]["key"] == KEY_BOB
        r = c.post("/fd/external-memory/agent/run", headers={"X-Agent-Auth": "tok-alice"},
                   json={"action": "search", "args": {"query": "flight"}})
        assert r.status_code == 404

    async def test_unknown_or_missing_identity_is_refused(self, db, membase, agents):
        c = _client()
        for headers in ({}, {"X-Agent-Auth": "other-deck"}):
            assert c.get("/fd/external-memory/agent/status", headers=headers).status_code == 403

    async def test_confirmation_and_unknown_actions_reach_the_agent(self, db, membase, agents):
        c = _client()
        _connect(c, BOB, KEY_BOB)
        r = c.post("/fd/external-memory/agent/run", headers={"X-Agent-Auth": "tok-bob"},
                   json={"action": "forget", "args": {"id": "mem_1"}})
        assert r.status_code == 409 and "confirm=true" in r.json()["detail"]
        r = c.post("/fd/external-memory/agent/run", headers={"X-Agent-Auth": "tok-bob"},
                   json={"action": "teleport", "args": {}})
        assert r.status_code == 422 and "offers" in r.json()["detail"]

    async def test_mirror_is_queued_once_with_the_decks_agent_name(self, db, membase, agents):
        c = _client()
        _connect(c, BOB, KEY_BOB)
        item = {"kind": "insight", "source_id": "ins-1", "content": "Bob's sister is Ana.",
                "agent": "I am lying", "meta": {"category": "contact"}}
        r = c.post("/fd/external-memory/agent/mirror", headers={"X-Agent-Auth": "tok-bob"}, json=item)
        assert r.json() == {"queued": True, "reason": "", "memories": ["Membase"]}
        r = c.post("/fd/external-memory/agent/mirror", headers={"X-Agent-Auth": "tok-bob"}, json=item)
        assert r.json() == {"queued": False, "reason": "already_queued", "memories": []}
        rows = await svc.outbox().recent(BOB)
        assert len(rows) == 1 and rows[0]["agent"] == "Bob's Researcher"
        # A kind the person turned off isn't queued; lessons never come from agents.
        r = c.post("/fd/external-memory/agent/mirror", headers={"X-Agent-Auth": "tok-bob"},
                   json={**item, "kind": "intuition", "source_id": "t1"})
        assert r.json() == {"queued": False, "reason": "kind_off", "memories": []}
        r = c.post("/fd/external-memory/agent/mirror", headers={"X-Agent-Auth": "tok-bob"},
                   json={**item, "kind": "lesson", "source_id": "l1"})
        assert r.status_code == 422
        # No connection: nothing queued.
        r = c.post("/fd/external-memory/agent/mirror", headers={"X-Agent-Auth": "tok-alice"}, json=item)
        assert r.json() == {"queued": False, "reason": "not_connected", "memories": []}


# ── the drainer ─────────────────────────────────────────────────────


class TestDrain:
    async def test_drain_writes_and_marks_sent(self, db, membase, agents, monkeypatch):
        monkeypatch.setattr(svc, "SEND_GAP_S", 0)
        _connect(_client(), BOB, KEY_BOB)
        for n in range(3):
            await svc.mirror_for_owner(BOB, MemoryItem(kind="insight", source_id=f"i{n}", content=f"fact {n}"))
        assert await svc.drain_once() == 3
        counts = await svc.outbox().counts(BOB)
        assert counts["sent"] == 3 and counts["pending"] == 0 and counts["last_sent_at"]
        assert len(_calls(membase, "/v1/memories/add")) == 3
        assert await svc.drain_once() == 0

    async def test_a_refused_key_holds_the_queue_without_spending_attempts(self, db, membase, agents, monkeypatch):
        monkeypatch.setattr(svc, "SEND_GAP_S", 0)
        _connect(_client(), BOB, KEY_BOB)
        await svc.mirror_for_owner(BOB, MemoryItem(kind="insight", source_id="i1", content="fact"))
        await svc.mirror_for_owner(BOB, MemoryItem(kind="insight", source_id="i2", content="fact 2"))
        membase.fail = 401
        await svc.drain_once()
        rows = await svc.outbox().recent(BOB)
        assert {r["state"] for r in rows} == {"pending"} and all(r["attempts"] == 0 for r in rows)
        assert len(_calls(membase, "/v1/memories/add")) == 1  # the owner is held after the first
        assert (await svc.load_connection(BOB))["verify_error"] == "refused"
        assert await svc.drain_once() == 0  # held for an hour
        membase.fail = None
        await svc.outbox().retry_owner(BOB)
        await svc.drain_once()
        assert (await svc.outbox().counts(BOB))["sent"] == 2

    async def test_retryable_errors_back_off_and_bad_ones_fail(self, db, membase, agents, monkeypatch):
        monkeypatch.setattr(svc, "SEND_GAP_S", 0)
        _connect(_client(), BOB, KEY_BOB)
        await svc.mirror_for_owner(BOB, MemoryItem(kind="insight", source_id="i1", content="fact"))
        membase.fail = 503
        await svc.drain_once()
        row = (await svc.outbox().recent(BOB))[0]
        assert row["state"] == "pending" and row["attempts"] == 1 and "Membase" in row["last_error"]
        await svc.outbox().retry_owner(BOB)
        membase.fail, membase.fail_body = 400, {"error": {"code": "validation", "message": "bad"}}
        await svc.drain_once()
        assert (await svc.outbox().recent(BOB))[0]["state"] == "failed"

    async def test_outbox_fails_after_max_attempts(self, tmp_path):
        box = Outbox(tmp_path / "o.db")
        await box.enqueue("u", "membase", "insight", "s1", {"kind": "insight"})
        row = (await box.due())[0]
        state = ""
        for _ in range(MAX_ATTEMPTS):
            state = await box.mark_retry(row["id"], "err", 0)
        assert state == "failed"
        await box.close()


# ── lessons from Flight Deck ────────────────────────────────────────


async def test_taught_lessons_are_mirrored(db, membase, agents):
    from captain_claw.flight_deck import agent_onboarding as ob

    _connect(_client(), BOB, KEY_BOB)
    await ob._mirror_lessons(BOB, "bob-bot", [("rule", "Always CC Marko on invoices."),
                                               ("ask_first", "Before booking travel.")], "you")
    rows = await svc.outbox().recent(BOB)
    assert {r["kind"] for r in rows} == {"lesson"} and len(rows) == 2
    assert all(r["agent"] == "Bob's Researcher" for r in rows)
    payload = json.loads((await svc.outbox().due())[0]["payload"])
    assert payload["meta"]["category"] in ("rule", "ask first")


# ── sign-in off: one connection for the deck ────────────────────────


async def test_auth_off_deck_keeps_one_connection(db, membase, agents, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    c = _client()
    r = _connect(c, BOB, KEY_BOB)
    assert r.status_code == 200 and r.json()["scope"] == "deck"
    assert await db.get_system_setting("external_memory:memories")
    m = c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-local"}).json()
    assert m["connected"] is True
