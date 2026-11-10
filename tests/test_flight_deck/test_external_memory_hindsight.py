"""Hindsight as an external memory, and the per-person scoping it brings:

* the provider speaks Hindsight's REST API (paths and bodies as in its own
  generated client): retain with a document id that replaces, recall, reflect,
  list, documents, forget only with confirm;
* a *scoped* provider can be the deck's connection (admin only); every user
  without their own then works in a bank of their own — never another's — and
  their own connection wins; a *key* provider (Membase) can't be the deck's;
* a user who isn't an admin can't point their own connection at this machine
  or its network; an admin can;
* mirrored items go to the user's bank, keyed by what they are, so an insight
  that replaces another (same entity_key) replaces its document.

Hindsight is never contacted: every ``httpx.AsyncClient`` gets a fake server.
"""

from __future__ import annotations

import json
import types

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import agent_secret, auth, fd_crypto
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.external_memory import base as em_base
from captain_claw.flight_deck.external_memory import routes as emr
from captain_claw.flight_deck.external_memory import service as svc
from captain_claw.flight_deck.external_memory.base import MemoryItem, ProviderError, scope_id
from captain_claw.flight_deck.external_memory.hindsight import HindsightProvider
from captain_claw.flight_deck.external_memory.outbox import Outbox

ALICE = "user-alice"   # admin
BOB = "user-bob"
CAROL = "user-carol"
HS = "https://hindsight.example.com"
HS_KEY = "hs_secret_key_0000abcd"


@pytest.fixture()
async def db(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for var in ("FD_LOCKDOWN", "FD_AGENT_SHARED_SECRET", "FD_HINDSIGHT_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    agent_secret.reset_cache_for_tests()
    fd_crypto.reset_for_tests()
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev_db = auth._db
    auth.set_auth_db(d)
    now = "2026-01-01T00:00:00Z"
    for uid, role in [(ALICE, "admin"), (BOB, "user"), (CAROL, "user")]:
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, f"{uid}@x.co", "h", uid.title(), role, now, now),
        )
    await d._db.commit()
    box = Outbox(tmp_path / "external_memory.db")
    svc.set_outbox(box)
    svc._host_checks.clear()
    # Names never resolve in tests: public unless written as a private address.
    monkeypatch.setattr(em_base.socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0))])
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
def hindsight(monkeypatch):
    """A fake Hindsight: banks of documents, each document's text as its facts."""
    state = types.SimpleNamespace(requests=[], banks={}, key=HS_KEY, fail=None)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        state.requests.append({"method": request.method, "path": request.url.path, "body": body,
                               "params": dict(request.url.params),
                               "auth": request.headers.get("authorization", "")})
        if isinstance(state.fail, int):
            return httpx.Response(state.fail, json={"detail": "nope"})
        path = request.url.path
        if path == "/version":
            return httpx.Response(200, json={"api_version": "0.9.1", "features": {"observations": True}})
        if state.key and request.headers.get("authorization") != f"Bearer {state.key}":
            return httpx.Response(401, json={"detail": "Unauthorized"})
        if path == "/v1/default/banks":
            return httpx.Response(200, json={"banks": [{"bank_id": b} for b in state.banks]})
        parts = path.split("/")  # ['', 'v1', 'default', 'banks', bank, ...]
        bank = parts[4]
        docs = state.banks.setdefault(bank, {})
        rest = "/".join(parts[5:])
        if rest == "memories" and request.method == "POST":
            for item in body["items"]:
                docs[item.get("document_id") or f"rand-{len(docs)}"] = item
            return httpx.Response(200, json={"success": True, "bank_id": bank, "items_count": len(body["items"]),
                                             "async": body.get("async", False), "operation_id": "op-1"})
        if rest == "memories/recall":
            words = body["query"].lower().split()
            hits = [{"id": f"f-{d}", "text": i["content"].split("\n")[0], "type": "world",
                     "document_id": d, "mentioned_at": "2026-11-05T10:00:00Z"}
                    for d, i in docs.items() if any(w in i["content"].lower() for w in words)]
            return httpx.Response(200, json={"results": hits})
        if rest == "reflect":
            return httpx.Response(200, json={"text": f"From {len(docs)} document(s): it's about Acme.",
                                             "based_on": {"memories": [{"text": "x"}] * len(docs)}})
        if rest == "memories/list":
            return httpx.Response(200, json={"items": [{"id": d, "text": i["content"][:50]} for d, i in docs.items()],
                                             "total": len(docs), "limit": 20, "offset": 0})
        if rest == "documents" and request.method == "GET":
            return httpx.Response(200, json={"items": [{"id": d, "memory_unit_count": 1, "text_length": 10}
                                                       for d in docs], "total": len(docs), "limit": 20, "offset": 0})
        if rest.startswith("documents/") and request.method == "DELETE":
            doc = rest.split("/", 1)[1]
            if doc not in docs:
                return httpx.Response(404, json={"detail": "Document not found"})
            del docs[doc]
            return httpx.Response(200, json={"success": True, "message": "deleted", "document_id": doc,
                                             "memory_units_deleted": 1})
        if rest == "stats":
            return httpx.Response(200, json={"bank_id": bank, "total_nodes": len(docs), "total_documents": len(docs),
                                             "pending_operations": 0, "total_observations": 0})
        return httpx.Response(404, json={"detail": "Not Found"})

    real_client = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real_client(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return state


@pytest.fixture()
def agents(monkeypatch):
    import captain_claw.flight_deck.server as srv

    registry = {"tok-bob": (BOB, "bob-bot"), "tok-carol": (CAROL, "carol-bot")}

    def find(token):
        if token in registry:
            owner, slug = registry[token]
            return True, owner, slug
        return False, "", ""

    monkeypatch.setattr(srv, "_find_agent_by_auth", find)
    monkeypatch.setattr(srv, "_resolve_agent_identity_by_auth", lambda t: find(t)[:2])
    monkeypatch.setattr(srv, "_load_process_registry", lambda: {"bob-bot": {"name": "Bob Bot"},
                                                                "carol-bot": {"name": "Carol Bot"}})
    return registry


def _client():
    app = FastAPI()
    app.include_router(emr.router)
    return TestClient(app, client=("127.0.0.1", 50123))


def _bearer(uid, role="user"):
    return {"Authorization": f"Bearer {auth.create_access_token(uid, role)}"}


def _deck(c, url=HS, key=HS_KEY):
    return c.put("/fd/external-memory/deck-connection", headers=_bearer(ALICE, "admin"),
                 json={"provider": "hindsight", "credentials": {"base_url": url, "api_key": key}})


def _hs(bank="captain-bob", key=HS_KEY):
    return HindsightProvider({"base_url": HS, "api_key": key}, scope=bank, max_retries=0)


# ── the provider ────────────────────────────────────────────────────


class TestHindsightProvider:
    async def test_verify_checks_the_key_on_a_call_that_needs_one(self, hindsight):
        info = await _hs().verify()
        assert info.extra["api_version"] == "0.9.1"
        assert [r["path"] for r in hindsight.requests] == ["/version", "/v1/default/banks"]
        with pytest.raises(em_base.ProviderAuthError):
            await _hs(key="wrong-key-123456").verify()

    async def test_a_server_without_a_key(self, hindsight):
        hindsight.key = ""
        p = HindsightProvider({"base_url": HS}, scope="captain-bob")
        await p.verify()
        assert hindsight.requests[-1]["auth"] == ""

    async def test_no_bank_no_call(self, hindsight):
        p = HindsightProvider({"base_url": HS, "api_key": HS_KEY})
        with pytest.raises(em_base.ProviderInputError):
            await p.run("search", {"query": "x"})
        assert hindsight.requests == []

    async def test_remember_replaces_itself_and_search_shows_dates(self, hindsight):
        p = _hs()
        await p.run("remember", {"content": "Acme's pilot starts in March.", "context": "call with Acme"})
        await p.run("remember", {"content": "acme's  pilot starts in march."})  # same fact, other spacing
        assert len(hindsight.banks["captain-bob"]) == 1
        req = hindsight.requests[0]
        assert req["path"] == "/v1/default/banks/captain-bob/memories" and req["body"]["async"] is True
        assert req["body"]["items"][0]["context"] == "call with Acme"
        out = await p.run("search", {"query": "pilot", "budget": "high", "limit": 5})
        recall = hindsight.requests[-1]
        assert recall["path"] == "/v1/default/banks/captain-bob/memories/recall"
        assert recall["body"]["budget"] == "high"
        assert "pilot starts in march" in out.text.lower() and "said 2026-11-05" in out.text
        assert "[document captain-note-" in out.text

    async def test_reflect_list_documents_capabilities(self, hindsight):
        p = _hs()
        await p.run("remember", {"content": "Acme wants SSO."})
        assert "it's about Acme" in (await p.run("reflect", {"question": "What does Acme need?"})).text
        assert "Acme wants SSO." in (await p.run("list", {})).text
        assert "1 fact(s)" in (await p.run("documents", {})).text
        cap = await p.run("capabilities", {})
        assert "bank “captain-bob”" in cap.text and "Holds 1 fact(s)" in cap.text

    async def test_forget_needs_confirm(self, hindsight):
        p = _hs()
        await p.run("remember", {"content": "Old phone 555-0100."})
        doc = next(iter(hindsight.banks["captain-bob"]))
        with pytest.raises(ProviderError) as exc:
            await p.run("forget", {"id": doc})
        assert exc.value.status == 428 and "confirm=true" in str(exc.value)
        assert doc in hindsight.banks["captain-bob"]  # nothing sent
        out = await p.run("forget", {"id": doc, "confirm": True})
        assert out.text.startswith(f"Forgotten: {doc}") and not hindsight.banks["captain-bob"]

    async def test_mirror_is_keyed_by_what_it_is(self, hindsight):
        p = _hs()
        await p.mirror(MemoryItem(kind="insight", source_id="i1", content="Lead on Acme is Ana.",
                                  agent="Bob Bot", created_at="2026-11-01T09:00:00",
                                  meta={"category": "contact", "entity_key": "contact:acme-lead"}))
        await p.mirror(MemoryItem(kind="insight", source_id="i2", content="Lead on Acme is Marko now.",
                                  meta={"category": "contact", "entity_key": "contact:acme-lead"}))
        docs = hindsight.banks["captain-bob"]
        assert list(docs) == ["captain-insight-contact:acme-lead"]  # replaced, not piled up
        item = docs["captain-insight-contact:acme-lead"]
        assert item["content"].startswith("Lead on Acme is Marko now.")
        assert item["metadata"]["source_id"] == "i2" and item["tags"][:2] == ["captain", "insight"]
        first = hindsight.requests[0]["body"]["items"][0]
        assert first["timestamp"] == "2026-11-01T09:00:00" and first["metadata"]["agent"] == "Bob Bot"
        # Only the person's world in the text: on a real server, provenance lines
        # came back as facts ("Bob Bot is a Captain agent").
        assert first["content"] == "Lead on Acme is Ana."
        assert "Captain" not in first["context"] and "contact" not in first["context"]
        assert first["metadata"]["category"] == "contact"

    def test_bad_credentials(self):
        with pytest.raises(em_base.ProviderInputError):
            HindsightProvider.clean_credentials({"base_url": "ftp://x"})
        with pytest.raises(em_base.ProviderInputError):
            HindsightProvider.clean_credentials({"base_url": HS, "bank": "../alice"})


# ── the deck's connection: one Hindsight, a bank per person ─────────


class TestDeckConnection:
    async def test_only_admins_and_only_scoped_providers(self, db, hindsight):
        c = _client()
        r = c.put("/fd/external-memory/deck-connection", headers=_bearer(BOB),
                  json={"provider": "hindsight", "credentials": {"base_url": HS, "api_key": HS_KEY}})
        assert r.status_code == 403
        r = c.put("/fd/external-memory/deck-connection", headers=_bearer(ALICE, "admin"),
                  json={"provider": "membase", "credentials": {"api_key": "mbk_x_12345678"}})
        assert r.status_code == 422 and "one person's memory" in r.json()["detail"]
        r = _deck(c)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["connected"] and body["provider"] == "hindsight" and HS_KEY not in r.text
        assert body["scope_example"] == "captain-user-id"

    async def test_each_user_gets_their_own_bank(self, db, hindsight, agents):
        c = _client()
        _deck(c)
        st = c.get("/fd/external-memory/status", headers=_bearer(BOB)).json()
        (mem,) = st["memories"]
        assert st["connected"] and mem["source"] == "deck" and mem["scope_id"] == scope_id(BOB)
        assert mem["id"] == "deck" and mem["name"] == "Hindsight (shared)"
        assert mem["credential_hints"] == {}  # the admin's key isn't Bob's to see
        c.post("/fd/external-memory/agent/run", headers={"X-Agent-Auth": "tok-bob"},
               json={"action": "remember", "args": {"content": "Bob's sister is Ana."}})
        c.post("/fd/external-memory/agent/run", headers={"X-Agent-Auth": "tok-carol"},
               json={"action": "remember", "args": {"content": "Carol prefers mornings."}})
        assert set(hindsight.banks) == {scope_id(BOB), scope_id(CAROL)}
        r = c.post("/fd/external-memory/agent/run", headers={"X-Agent-Auth": "tok-carol"},
                   json={"action": "search", "args": {"query": "sister Ana"}})
        assert "nothing" in r.json()["text"]  # Bob's fact is not in Carol's bank
        m = c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json()
        assert m["provider"] == "hindsight" and m["reaches"] == [f"Hindsight bank “{scope_id(BOB)}”"]
        assert "reflect" in {a["name"] for a in m["actions"]}
        assert not any(p["name"] == "bank" for a in m["actions"] for p in a["params"])  # agents never pick one

    async def test_a_users_own_memory_sits_next_to_the_decks(self, db, hindsight):
        c = _client()
        _deck(c)
        r = c.put("/fd/external-memory/connection", headers=_bearer(BOB),
                  json={"provider": "hindsight",
                        "credentials": {"base_url": HS, "api_key": HS_KEY, "bank": "bob-private"}})
        assert r.status_code == 200, r.text
        own, deck = r.json()["memories"]
        assert own["source"] == "own" and own["scope_id"] == "bob-private" and own["name"] == "Hindsight"
        assert own["credential_hints"]["bank"] == "bob-private"
        assert deck["source"] == "deck" and deck["enabled"] is True  # still used, next to their own
        st = c.delete(f"/fd/external-memory/connection?memory={own['id']}", headers=_bearer(BOB)).json()
        assert [m["source"] for m in st["memories"]] == ["deck"]
        r = c.delete("/fd/external-memory/connection?memory=deck", headers=_bearer(BOB))
        assert r.status_code == 422  # the shared one is switched off, not removed
        st = c.put("/fd/external-memory/memories/deck", headers=_bearer(BOB), json={"enabled": False}).json()
        assert st["connected"] is False and st["memories"][0]["enabled"] is False

    async def test_mirrors_go_to_the_owners_bank(self, db, hindsight, agents, monkeypatch):
        monkeypatch.setattr(svc, "SEND_GAP_S", 0)
        c = _client()
        _deck(c)
        for tok in ("tok-bob", "tok-carol"):
            c.post("/fd/external-memory/agent/mirror", headers={"X-Agent-Auth": tok},
                   json={"kind": "insight", "source_id": "i1", "content": f"fact from {tok}"})
        await svc.drain_once()
        assert (await svc.outbox().counts(BOB))["sent"] == 1
        assert list(hindsight.banks[scope_id(BOB)].values())[0]["content"].startswith("fact from tok-bob")
        assert list(hindsight.banks[scope_id(CAROL)].values())[0]["content"].startswith("fact from tok-carol")

    async def test_removing_the_deck_connection(self, db, hindsight):
        c = _client()
        _deck(c)
        assert c.delete("/fd/external-memory/deck-connection",
                        headers=_bearer(ALICE, "admin")).json()["connected"] is False
        assert c.get("/fd/external-memory/status", headers=_bearer(BOB)).json()["connected"] is False


# ── private addresses ───────────────────────────────────────────────


class TestPrivateAddresses:
    @pytest.mark.parametrize("url", ["http://127.0.0.1:8888", "http://localhost:8888", "http://10.0.0.5",
                                     "http://169.254.169.254", "http://[::1]:8888", "http://box.local"])
    async def test_a_user_cant_point_at_this_machine_or_network(self, db, hindsight, url):
        r = _client().put("/fd/external-memory/connection", headers=_bearer(BOB),
                          json={"provider": "hindsight", "credentials": {"base_url": url}})
        assert r.status_code == 422 and "only an admin" in r.json()["detail"]
        assert hindsight.requests == [] and await svc.load_connection(BOB) is None

    async def test_a_name_that_resolves_inside_is_refused_too(self, db, hindsight, monkeypatch):
        monkeypatch.setattr(em_base.socket, "getaddrinfo", lambda h, p: [(2, 1, 6, "", ("192.168.1.20", 0))])
        r = _client().put("/fd/external-memory/connection", headers=_bearer(BOB),
                          json={"provider": "hindsight", "credentials": {"base_url": "https://sneaky.example"}})
        assert r.status_code == 422

    async def test_an_admin_may(self, db, hindsight):
        r = _client().put("/fd/external-memory/connection", headers=_bearer(ALICE, "admin"),
                          json={"provider": "hindsight",
                                "credentials": {"base_url": "http://127.0.0.1:8888", "api_key": HS_KEY}})
        assert r.status_code == 200, r.text
        assert (await svc.load_connection(ALICE))["private_ok"] is True

    async def test_a_name_that_later_points_inside_is_refused_at_call_time(self, db, hindsight, agents, monkeypatch):
        c = _client()
        r = c.put("/fd/external-memory/connection", headers=_bearer(BOB),
                  json={"provider": "hindsight", "credentials": {"base_url": HS, "api_key": HS_KEY}})
        assert r.status_code == 200
        svc._host_checks.clear()
        monkeypatch.setattr(em_base.socket, "getaddrinfo", lambda h, p: [(2, 1, 6, "", ("127.0.0.1", 0))])
        before = len(hindsight.requests)
        r = c.post("/fd/external-memory/agent/run", headers={"X-Agent-Auth": "tok-bob"},
                   json={"action": "search", "args": {"query": "x"}})
        assert r.status_code == 400 and "only an admin" in r.json()["detail"]
        assert len(hindsight.requests) == before

    async def test_membase_server_address_is_guarded_too(self, db, hindsight):
        r = _client().put("/fd/external-memory/connection", headers=_bearer(BOB),
                          json={"provider": "membase",
                                "credentials": {"api_key": "mbk_bob_12345678", "base_url": "http://127.0.0.1:8787"}})
        assert r.status_code == 422
