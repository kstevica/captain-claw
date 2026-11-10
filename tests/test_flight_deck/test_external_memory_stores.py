"""Qdrant and ArcadeDB as external memories — two connectors, one shape:

* both are *scoped*: one collection / database serves many people, each in a
  space of their own; EVERY request that touches stored items carries the
  person's space, and the agent never names it;
* Flight Deck makes the vectors (a model on the deck by default, or an
  OpenAI-compatible endpoint), and a connection that changes model is refused
  rather than mixing vectors that can't be compared;
* saving the same thing again replaces it (a fact by its text, a mirrored item
  by what it is), and forget needs the person's yes;
* the deck's connection never keeps a space; a user who isn't an admin can't
  point the embeddings address (any more than the server's) at this machine;
* Qdrant searches by meaning and words (falls back to meaning when the server
  can't do words); ArcadeDB adds the topic graph (related).

Neither server is contacted: every ``httpx.AsyncClient`` gets a fake, and the
deck's own embedding model is swapped for a word-hashing one. Against real
servers: ``test_external_memory_stores_live.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import types

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import agent_secret, auth, fd_crypto
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.external_memory import base as em_base
from captain_claw.flight_deck.external_memory import embeddings, providers
from captain_claw.flight_deck.external_memory import routes as emr
from captain_claw.flight_deck.external_memory import service as svc
from captain_claw.flight_deck.external_memory.arcadedb import ArcadeDBProvider, _ft_query
from captain_claw.flight_deck.external_memory.base import (
    ConnectionInfo,
    MemoryItem,
    ProviderError,
    ProviderInputError,
    RecallHit,
    scope_id,
)
from captain_claw.flight_deck.external_memory.hindsight import HindsightProvider
from captain_claw.flight_deck.external_memory.membase import MembaseProvider
from captain_claw.flight_deck.external_memory.outbox import Outbox
from captain_claw.flight_deck.external_memory.qdrant import QdrantProvider

ALICE = "user-alice"   # admin
BOB = "user-bob"
CAROL = "user-carol"
QD = "https://qdrant.example.com"
QD_KEY = "qd_secret_key_0000abcd"
ADB = "https://arcade.example.com"
ADB_PASS = "arcade-secret-pass"
EMB = "https://embed.example.com/v1"
DIMS = 64


def word_vectors(model: str, texts: list[str]) -> list[list[float]]:
    """Stand-in for the deck's model: texts sharing content words point the
    same way (stopwords carry no meaning here, as in a real model)."""
    from captain_claw import retrieval

    out = []
    for text in texts:
        v = [0.0] * DIMS
        for w in retrieval.query_terms(text, max_terms=64, keep_identifiers=True):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIMS] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


# ── fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
async def db(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for var in ("FD_LOCKDOWN", "FD_AGENT_SHARED_SECRET", "FD_QDRANT_BASE_URL", "FD_ARCADEDB_BASE_URL"):
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


@pytest.fixture(autouse=True)
def local_model(monkeypatch):
    calls: list[str] = []

    def fake(model, texts):
        calls.append(model)
        if "missing" in model:
            raise OSError("no such model")
        return word_vectors(model, texts)

    monkeypatch.setattr(embeddings, "_local_vectors", fake)
    return calls


class FakeQdrant:
    """Collections of points; filters, dense / text / fused queries, scroll,
    count, delete, facet — enough of Qdrant 1.20's REST API."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.collections: dict[str, dict] = {}
        self.key = QD_KEY
        self.version = "1.20.0"

    @staticmethod
    def matches(payload: dict, flt: dict | None) -> bool:
        for cond in (flt or {}).get("must", []):
            got, want = payload.get(cond["key"]), cond["match"]["value"]
            if (want not in got) if isinstance(got, list) else (got != want):
                return False
        return True

    def rank(self, col: dict, q: dict) -> list[tuple[str, float]]:
        pts = [(pid, p) for pid, p in col["points"].items() if self.matches(p["payload"], q.get("filter"))]
        query = q["query"]
        if isinstance(query, list):
            scored = [(pid, _cos(query, p["vector"][q["using"]])) for pid, p in pts]
            if "score_threshold" in q:
                scored = [s for s in scored if s[1] >= q["score_threshold"]]
        elif isinstance(query, dict) and "text" in query:
            if not (col["schema"].get("content") or {}).get("scoring"):
                raise ValueError("text query needs a scoring full-text index")
            words = set(re.findall(r"\w+", query["text"].lower()))
            scored = [(pid, float(len(words & set(re.findall(r"\w+", p["payload"]["content"].lower())))))
                      for pid, p in pts]
            scored = [s for s in scored if s[1] > 0]
        else:
            raise ValueError("unknown query")
        return sorted(scored, key=lambda s: s[1], reverse=True)[: q.get("limit", 10)]

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.requests.append({"method": request.method, "path": path, "body": body,
                              "key": request.headers.get("api-key", "")})
        if path == "/":
            return httpx.Response(200, json={"title": "qdrant", "version": self.version})
        if self.key and request.headers.get("api-key") != self.key:
            return httpx.Response(401, json={"status": {"error": "Must provide an API key"}})
        if path == "/collections":
            return httpx.Response(200, json={"result": {"collections": [{"name": n} for n in self.collections]}})
        parts = path.split("/")  # ['', 'collections', name, ...]
        name, rest = parts[2], "/".join(parts[3:])
        col = self.collections.get(name)
        ok = lambda result: httpx.Response(200, json={"result": result, "status": "ok"})  # noqa: E731
        if rest == "" and request.method == "PUT":
            self.collections[name] = {"vectors": body["vectors"], "hnsw": body.get("hnsw_config"),
                                      "schema": {}, "points": {}}
            return ok(True)
        if col is None:
            return httpx.Response(404, json={"status": {"error": f"Not found: Collection `{name}` doesn't exist!"}})
        if rest == "" and request.method == "GET":
            return ok({"config": {"params": {"vectors": col["vectors"]}},
                       "payload_schema": {k: {"params": v} for k, v in col["schema"].items()}})
        if rest == "index":
            schema = dict(body["field_schema"])
            if schema.get("scoring") and self.version < "1.20":
                return httpx.Response(400, json={"status": {"error": "unknown field `scoring`"}})
            col["schema"][body["field_name"]] = schema
            return ok({"status": "completed"})
        if rest == "points" and request.method == "PUT":
            for p in body["points"]:
                col["points"][p["id"]] = {"vector": p["vector"], "payload": p["payload"]}
            return ok({"status": "completed"})
        if rest == "points/query":
            try:
                if "prefetch" in body:
                    ranks = [[pid for pid, _ in self.rank(col, pf)] for pf in body["prefetch"]]
                    ranked = em_base.rrf(*ranks)[: body["limit"]]
                else:
                    ranked = self.rank(col, body)
            except ValueError as exc:
                return httpx.Response(400, json={"status": {"error": str(exc)}})
            return ok({"points": [{"id": pid, "score": s, "payload": col["points"][pid]["payload"]}
                                  for pid, s in ranked]})
        if rest == "points/scroll":
            pts = [p["payload"] for p in col["points"].values() if self.matches(p["payload"], body.get("filter"))]
            pts.sort(key=lambda p: p.get("created_at", ""), reverse=True)
            return ok({"points": [{"id": "x", "payload": p} for p in pts[: body["limit"]]]})
        if rest == "points/count":
            return ok({"count": sum(self.matches(p["payload"], body["filter"]) for p in col["points"].values())})
        if rest == "points/delete":
            for pid in [pid for pid, p in col["points"].items() if self.matches(p["payload"], body["filter"])]:
                del col["points"][pid]
            return ok({"status": "completed"})
        if rest == "facet":
            counts: dict[str, int] = {}
            for p in col["points"].values():
                if self.matches(p["payload"], body.get("filter")):
                    for t in p["payload"].get(body["key"]) or []:
                        counts[t] = counts.get(t, 0) + 1
            return ok({"hits": [{"value": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]})
        return httpx.Response(404, json={"status": {"error": "no such route"}})

    def item_requests(self) -> list[dict]:
        """The requests that read or change stored items."""
        return [r for r in self.requests if "/points" in r["path"] or r["path"].endswith("/facet")]


class FakeArcadeDB:
    """Records what is sent, keeps memories per (tenant, doc_id) and answers
    the queries the provider makes. Real SQL runs in the live tests."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.databases: set[str] = set()
        self.rows: dict[tuple[str, str], dict] = {}
        self.password = ADB_PASS
        self.may_create = True

    def _items(self, p: dict, where_topic: bool = False) -> list[dict]:
        rows = [r for (t, _d), r in self.rows.items() if t == p["t"]]
        if "topic" in p:
            rows = [r for r in rows if p["topic"] in r["topics"]]
        if "kind" in p:
            rows = [r for r in rows if r["kind"] == p["kind"]]
        return rows

    @staticmethod
    def _view(r: dict) -> dict:
        return {k: r[k] for k in ("doc_id", "content", "kind", "created_at", "topics")}

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.requests.append({"method": request.method, "path": path, "body": body})
        if request.headers.get("authorization") != httpx.BasicAuth("root", self.password)._auth_header:
            return httpx.Response(401, json={"error": "Authentication failed", "detail": "User/Password not valid"})
        ok = lambda result: httpx.Response(200, json={"result": result})  # noqa: E731
        if path == "/api/v1/server" and request.method == "GET":
            return httpx.Response(200, json={"version": "26.11.1 (build x)", "user": "root"})
        if path.startswith("/api/v1/exists/"):
            return ok(path.rsplit("/", 1)[1] in self.databases)
        if path == "/api/v1/server":
            if not self.may_create:
                return httpx.Response(403, json={"error": "Forbidden"})
            self.databases.add(body["command"].split()[-1])
            return ok("ok")
        db = path.rsplit("/", 1)[1]
        if db not in self.databases:
            return httpx.Response(404, json={"error": "Database not found",
                                             "exception": "com.arcadedb.exception.DatabaseNotFoundException"})
        sql, p = body["command"], body.get("params") or {}
        if sql.startswith("CREATE VERTEX TYPE CaptainMemory"):
            return ok([{"value": "ok"}])
        if "UPDATE CaptainMemory SET" in sql:
            topics = [p[k] for k in sorted(k for k in p if re.fullmatch(r"n\d+", k))]
            self.rows[(p["t"], p["d"])] = {"doc_id": p["d"], "content": p["c"], "kind": p["kind"],
                                          "created_at": p["created"], "topics": topics, "embedder": p["emb"],
                                          "embedding": p["vec"], "agent": p["agent"], "category": p["category"]}
            return ok([{"doc_id": p["d"]}])
        if "DELETE FROM CaptainMemory" in sql:
            self.rows.pop((p["t"], p["d"]), None)
            return ok([{"value": 1}])
        if "SEARCH_INDEX" in sql:
            words = set(re.findall(r"\w+", p["q"]))
            hits = [(r["doc_id"], len(words & set(re.findall(r"\w+", r["content"].lower()))))
                    for r in self._items(p)]
            return ok([{"doc_id": d, "score": s} for d, s in sorted(hits, key=lambda h: -h[1]) if s])
        if "vectorCosineSimilarity" in sql:
            rows = [r for r in self._items(p) if r["embedder"] == p["emb"]]
            rows.sort(key=lambda r: -_cos(p["v"], r["embedding"]))
            return ok([{"doc_id": r["doc_id"], "sim": _cos(p["v"], r["embedding"])} for r in rows])
        if "doc_id IN :ids" in sql:
            return ok([self._view(r) for (t, d), r in self.rows.items() if t == p["t"] and d in p["ids"]])
        if sql.startswith("SELECT count(*) AS n FROM CaptainMemory WHERE tenant = :t AND doc_id = :d"):
            return ok([{"n": int((p["t"], p["d"]) in self.rows)}])
        if sql.startswith("SELECT count(*) AS n FROM CaptainMemory"):
            return ok([{"n": len(self._items(p))}])
        if sql.startswith("SELECT count(*) AS n FROM CaptainTopic"):
            return ok([{"n": len({t for r in self._items(p) for t in r["topics"]})}])
        if "count(*) AS shared" in sql:
            about = [r for r in self._items(p) if p["n"] in r["topics"]]
            shared: dict[str, int] = {}
            for r in about:
                for t in r["topics"]:
                    if t != p["n"]:
                        shared[t] = shared.get(t, 0) + 1
            return ok([{"name": k, "shared": v} for k, v in shared.items()])
        if "expand(in('CaptainAbout'))" in sql:
            return ok([self._view(r) for r in self._items(p) if p["n"] in r["topics"]])
        if "FROM CaptainTopic" in sql:
            counts: dict[str, int] = {}
            for r in self._items(p):
                for t in r["topics"]:
                    counts[t] = counts.get(t, 0) + 1
            return ok([{"name": k, "n": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])])
        if "FROM CaptainMemory WHERE" in sql:
            rows = sorted(self._items(p), key=lambda r: r["created_at"], reverse=True)
            return ok([self._view(r) for r in rows])
        return httpx.Response(400, json={"error": "unexpected", "detail": sql})

    def item_requests(self) -> list[dict]:
        """The SQL sent about stored items (not the schema, not the server)."""
        return [r for r in self.requests if r["path"].startswith(("/api/v1/query/", "/api/v1/command/"))
                and not r["body"]["command"].startswith("CREATE VERTEX TYPE")]


@pytest.fixture()
def servers(monkeypatch):
    """Fake Qdrant, ArcadeDB and an OpenAI-compatible embeddings endpoint."""
    st = types.SimpleNamespace(qd=FakeQdrant(), adb=FakeArcadeDB(), emb_requests=[], emb_status=200)

    def embed(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        st.emb_requests.append({"path": request.url.path, "body": body,
                                "auth": request.headers.get("authorization", "")})
        if st.emb_status != 200:
            return httpx.Response(st.emb_status, json={"error": {"message": "bad key"}})
        vecs = word_vectors(body["model"], body["input"])
        rows = [{"index": i, "embedding": [x * 3 for x in v]} for i, v in enumerate(vecs)]
        return httpx.Response(200, json={"data": list(reversed(rows))})  # any order, by index

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "qdrant.example.com":
            return st.qd.handler(request)
        if host == "arcade.example.com":
            return st.adb.handler(request)
        if host == "embed.example.com":
            return embed(request)
        return httpx.Response(599)

    real_client = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real_client(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return st


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


QD_CREDS = {"base_url": QD, "api_key": QD_KEY}
ADB_CREDS = {"base_url": ADB, "password": ADB_PASS}
STORES = {"qdrant": QD_CREDS, "arcadedb": ADB_CREDS}


async def _store(name: str, space: str = "captain-bob", **extra) -> em_base.ExternalMemoryProvider:
    """A provider for *space*, checked once (as a saved connection would be)."""
    creds = {**STORES[name], **extra}
    info = await providers.build(name, creds, scope="captain-check").verify()
    return providers.build(name, creds, info, scope=space, max_retries=0)


def _fake(servers, name):
    return servers.qd if name == "qdrant" else servers.adb


# ── both stores ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["qdrant", "arcadedb"])
class TestBothStores:
    async def test_catalog_entry(self, name):
        entry = providers.provider_class(name).catalog_entry()
        assert entry["tenancy"] == "scoped" and entry["scope_field"] == "space" and entry["scope_label"] == "space"
        fields = {f["name"]: f for f in entry["credential_fields"]}
        assert {"embedding_url", "embedding_model", "embedding_key"} <= set(fields)
        assert fields["embedding_url"]["advanced"] and not fields["base_url"]["advanced"]
        assert providers.provider_class(name).address_fields() == ("base_url", "embedding_url")

    async def test_remember_replaces_itself_and_search_finds_it(self, servers, name):
        p = await _store(name)
        await p.run("remember", {"content": "Acme's pilot starts in March.", "topics": "Acme, pilot"})
        out = await p.run("remember", {"content": "acme's  pilot starts in march."})  # same fact, other spacing
        assert "captain-note-" in out.text
        found = await p.run("search", {"query": "when does the acme pilot start"})
        assert found.text.count("pilot starts in march") == 1  # replaced, not piled up
        assert "[id captain-note-" in found.text

    async def test_every_item_request_is_in_the_persons_space(self, servers, name):
        bob = await _store(name, "captain-bob")
        carol = await _store(name, "captain-carol")
        await bob.run("remember", {"content": "Bob's sister is Ana.", "topics": "ana"})
        await carol.run("remember", {"content": "Carol prefers mornings."})
        out = await carol.run("search", {"query": "sister Ana"})
        assert "Bob's sister" not in out.text  # Bob's fact is not in Carol's space
        for action, args in (("list", {}), ("topics", {}), ("capabilities", {})):
            await carol.run(action, args)
        with pytest.raises(ProviderError) as exc:
            await carol.run("forget", {"id": "captain-note-x", "confirm": True})
        assert exc.value.status == 404
        fake = _fake(servers, name)
        sent = fake.item_requests()
        assert sent
        for r in sent:
            if name == "qdrant":
                flt = (r["body"] or {}).get("filter") or (r["body"] or {}).get("prefetch", [{}])[0].get("filter")
                points = (r["body"] or {}).get("points")
                if points:
                    assert all(pt["payload"]["tenant"] in ("captain-bob", "captain-carol") for pt in points)
                else:
                    assert flt["must"][0] == {"key": "tenant", "match": {"value": flt["must"][0]["match"]["value"]}}
                    assert flt["must"][0]["match"]["value"] in ("captain-bob", "captain-carol")
                    for pf in (r["body"] or {}).get("prefetch", []):
                        assert pf["filter"]["must"][0]["key"] == "tenant"
            else:
                assert r["body"]["params"]["t"] in ("captain-bob", "captain-carol")
                assert "tenant = :t" in r["body"]["command"]

    async def test_forget_needs_confirm(self, servers, name):
        p = await _store(name)
        out = await p.run("remember", {"content": "Old phone 555-0100."})
        doc = out.data["id"]
        with pytest.raises(ProviderError) as exc:
            await p.run("forget", {"id": doc})
        assert exc.value.status == 428 and "confirm=true" in str(exc.value)
        assert "555-0100" in (await p.run("list", {})).text  # nothing removed
        assert (await p.run("forget", {"id": doc, "confirm": True})).text == f"Forgotten: {doc}."
        assert "555-0100" not in (await p.run("list", {})).text

    async def test_mirror_is_keyed_by_what_it_is(self, servers, name):
        p = await _store(name)
        await p.mirror(MemoryItem(kind="insight", source_id="i1", content="Lead on Acme is Ana.",
                                  agent="Bob Bot", created_at="2026-11-01T09:00:00", tags=["acme"],
                                  meta={"category": "contact", "entity_key": "contact:acme-lead",
                                        "why": "She runs the pilot."}))
        await p.mirror(MemoryItem(kind="insight", source_id="i2", content="Lead on Acme is Marko now.",
                                  agent="Bob Bot", meta={"category": "contact", "entity_key": "contact:acme-lead"}))
        listed = await p.run("list", {})
        assert "Marko" in listed.text and "Ana" not in listed.text  # replaced, not piled up
        assert "[id captain-insight-contact:acme-lead]" in listed.text
        if name == "qdrant":
            pts = list(servers.qd.collections["captain_memory"]["points"].values())
            first = pts[0]["payload"]
            assert first["kind"] == "insight" and first["agent"] == "Bob Bot" and first["category"] == "contact"
        else:
            row = servers.adb.rows[("captain-bob", "captain-insight-contact:acme-lead")]
            first = row
            assert row["agent"] == "Bob Bot" and row["category"] == "contact"
        # Only the person's world in the text; who kept it is beside it.
        assert first["content"] == "Lead on Acme is Marko now."
        assert "Captain" not in first["content"] and "Bob Bot" not in first["content"]

    async def test_a_mirrored_items_topics_are_its_tags_not_its_category(self, servers, name):
        p = await _store(name)
        await p.mirror(MemoryItem(kind="insight", source_id="i9", content="Ana leads the Acme pilot.",
                                  tags=["Ana", "acme"], meta={"category": "contact"}))
        assert "ana (1), acme (1)" in (await p.run("topics", {})).text
        assert "contact" not in (await p.run("topics", {})).text

    async def test_mirror_text_keeps_why(self, servers, name):
        p = await _store(name)
        await p.mirror(MemoryItem(kind="lesson", source_id="l1", content="Invoices go out on the 1st.",
                                  meta={"why": "The client pays on the 5th."}))
        out = await p.run("search", {"query": "invoices"})
        assert "Invoices go out on the 1st.\nThe client pays on the 5th." in out.text

    async def test_topics_and_topic_filter(self, servers, name):
        p = await _store(name)
        await p.run("remember", {"content": "Atlas ships in March.", "topics": "Atlas, Ana Kovač"})
        await p.run("remember", {"content": "Ana prefers Tuesday meetings.", "topics": "ana kovač, meetings"})
        out = await p.run("topics", {})
        assert "ana kovač (2)" in out.text
        found = await p.run("search", {"query": "march meetings", "topic": "MEETINGS"})
        assert "Tuesday" in found.text and "Atlas ships" not in found.text

    async def test_no_space_no_call(self, servers, name):
        p = providers.build(name, STORES[name])
        with pytest.raises(ProviderInputError):
            await p.run("search", {"query": "x"})
        assert not _fake(servers, name).requests

    async def test_bad_credentials(self, name):
        cls = providers.provider_class(name)
        good = STORES[name]
        with pytest.raises(ProviderInputError):
            cls.clean_credentials({**good, "base_url": "ftp://x"})
        with pytest.raises(ProviderInputError):
            cls.clean_credentials({**good, "space": "../alice"})
        with pytest.raises(ProviderInputError, match="name its model"):
            cls.clean_credentials({**good, "embedding_url": EMB})
        with pytest.raises(ProviderInputError, match="needs the embeddings address"):
            cls.clean_credentials({**good, "embedding_key": "sk-abcdefgh"})
        out = cls.clean_credentials({**good, "embedding_url": EMB + "/", "embedding_model": "m"})
        assert out["embedding_url"] == EMB

    async def test_a_model_that_cant_be_made_is_refused_before_anything_is_made(self, servers, name):
        creds = {**STORES[name], "embedding_model": "minishlab/missing-model"}
        with pytest.raises(ProviderInputError, match="couldn't make vectors"):
            await providers.build(name, creds, scope="captain-check").verify()
        fake = _fake(servers, name)
        if name == "qdrant":
            assert not fake.collections
        else:
            assert not fake.databases

    async def test_vectors_from_an_embeddings_endpoint(self, servers, name):
        p = await _store(name, embedding_url=EMB, embedding_model="text-embedding-3-small",
                         embedding_key="sk-test-123456")
        await p.run("remember", {"content": "Dentist every second Friday."})
        out = await p.run("search", {"query": "dentist"})
        assert "Dentist" in out.text
        req = servers.emb_requests[0]
        assert req["path"] == "/v1/embeddings" and req["auth"] == "Bearer sk-test-123456"
        assert req["body"]["model"] == "text-embedding-3-small" and isinstance(req["body"]["input"], list)
        assert p.info.extra["vector"] == "text-embedding-3-small"
        servers.emb_status = 401
        with pytest.raises(ProviderError, match="refused the key"):
            await p.run("remember", {"content": "Something else."})


# ── Qdrant ──────────────────────────────────────────────────────────


class TestQdrant:
    async def test_verify_makes_the_collection_for_the_model(self, servers):
        info = await QdrantProvider(QD_CREDS, scope="captain-check").verify()
        col = servers.qd.collections["captain_memory"]
        assert col["vectors"] == {"m2v-potion-base-8m": {"size": DIMS, "distance": "Cosine"}}
        assert col["hnsw"] == {"payload_m": 16, "m": 0}
        assert col["schema"]["tenant"] == {"type": "keyword", "is_tenant": True}
        assert col["schema"]["content"]["scoring"] is True
        assert info.extra == {"version": "1.20.0", "collection": "captain_memory", "vector": "m2v-potion-base-8m",
                              "dims": DIMS, "hybrid": True,
                              "embeddings": "minishlab/potion-base-8M on Flight Deck"}
        assert all(r["key"] == QD_KEY for r in servers.qd.requests if r["path"] != "/")
        # Checked again: nothing is made twice.
        n = len(servers.qd.requests)
        await QdrantProvider(QD_CREDS, scope="captain-check").verify()
        assert not any(r["method"] == "PUT" for r in servers.qd.requests[n:])

    async def test_the_key_is_checked_on_a_call_that_needs_one(self, servers):
        with pytest.raises(em_base.ProviderAuthError):
            await QdrantProvider({"base_url": QD, "api_key": "wrong-key-123"}, scope="x").verify()

    async def test_a_collection_made_for_another_model_is_refused(self, servers):
        await QdrantProvider(QD_CREDS, scope="captain-check").verify()
        other = QdrantProvider({**QD_CREDS, "embedding_model": "minishlab/potion-base-32M"}, scope="x")
        with pytest.raises(ProviderInputError, match="made for other vectors"):
            await other.verify()
        servers.qd.collections["captain_memory"]["vectors"]["m2v-potion-base-8m"]["size"] = 8
        with pytest.raises(ProviderInputError, match="holds 8-number vectors"):
            await QdrantProvider(QD_CREDS, scope="x").verify()

    async def test_search_fuses_meaning_and_words(self, servers):
        p = await _store("qdrant")
        await p.run("remember", {"content": "Invoice INV-2207 is overdue."})
        await p.run("search", {"query": "INV-2207"})
        body = servers.qd.requests[-1]["body"]
        assert body["query"] == {"fusion": "rrf"}
        assert [pf["using"] for pf in body["prefetch"]] == ["m2v-potion-base-8m", "content"]
        assert body["prefetch"][1]["query"] == {"text": "INV-2207"}

    async def test_older_qdrant_searches_by_meaning_only(self, servers):
        servers.qd.version = "1.15.4"
        p = await _store("qdrant")
        assert p.info.extra["hybrid"] is False
        assert "scoring" not in servers.qd.collections["captain_memory"]["schema"]["content"]
        await p.run("remember", {"content": "Acme renewal is in June."})
        out = await p.run("search", {"query": "acme renewal"})
        assert "June" in out.text and "prefetch" not in servers.qd.requests[-1]["body"]
        assert "word search needs Qdrant 1.20+" in (await p.run("capabilities", {})).text

    async def test_word_search_refused_falls_back_to_meaning(self, servers):
        p = await _store("qdrant")
        await p.run("remember", {"content": "Acme renewal is in June."})
        servers.qd.collections["captain_memory"]["schema"]["content"].pop("scoring")  # an index without BM25
        out = await p.run("search", {"query": "acme renewal"})
        assert "June" in out.text
        assert "prefetch" in servers.qd.requests[-2]["body"] and "prefetch" not in servers.qd.requests[-1]["body"]

    async def test_a_deleted_collection_is_made_again_on_write(self, servers):
        p = await _store("qdrant")
        servers.qd.collections.clear()
        await p.run("remember", {"content": "Still saved."})
        assert len(servers.qd.collections["captain_memory"]["points"]) == 1

    async def test_the_same_document_in_two_spaces_is_two_points(self, servers):
        bob, carol = await _store("qdrant", "captain-bob"), await _store("qdrant", "captain-carol")
        await bob.run("remember", {"content": "Same words."})
        await carol.run("remember", {"content": "Same words."})
        assert len(servers.qd.collections["captain_memory"]["points"]) == 2


# ── ArcadeDB ────────────────────────────────────────────────────────


class TestArcadeDB:
    async def test_verify_makes_the_database_and_schema(self, servers):
        info = await ArcadeDBProvider(ADB_CREDS, scope="captain-check").verify()
        assert servers.adb.databases == {"captain_memory"}
        made = [r["body"] for r in servers.adb.requests if r["path"] == "/api/v1/server" and r["method"] == "POST"]
        assert made == [{"command": "create database captain_memory"}]
        schema = next(r["body"] for r in servers.adb.requests if r["path"] == "/api/v1/command/captain_memory")
        assert schema["language"] == "sqlscript"
        for piece in ("ON CaptainMemory (tenant, doc_id) UNIQUE", "ON CaptainMemory (content) FULL_TEXT",
                      "ON CaptainTopic (tenant, name) UNIQUE", "CREATE EDGE TYPE CaptainAbout IF NOT EXISTS"):
            assert piece in schema["command"]
        assert info.extra["version"] == "26.11.1" and info.extra["dims"] == DIMS
        # Checked again: the database isn't made twice.
        await ArcadeDBProvider(ADB_CREDS, scope="captain-check").verify()
        assert len([r for r in servers.adb.requests if r["path"] == "/api/v1/server" and r["method"] == "POST"]) == 1

    async def test_wrong_password_and_no_right_to_create(self, servers):
        with pytest.raises(em_base.ProviderAuthError):
            await ArcadeDBProvider({**ADB_CREDS, "password": "nope"}, scope="x").verify()
        servers.adb.may_create = False
        with pytest.raises(ProviderInputError, match="may not make one"):
            await ArcadeDBProvider(ADB_CREDS, scope="x").verify()

    async def test_topics_are_a_graph(self, servers):
        p = await _store("arcadedb")
        await p.run("remember", {"content": "Atlas ships in March.", "topics": "Atlas, Ana Kovač"})
        await p.run("remember", {"content": "Ana prefers Tuesday meetings.", "topics": "ana kovač, meetings"})
        out = await p.run("related", {"topic": "Ana Kovač"})
        assert "about “ana kovač”" in out.text and "Atlas ships" in out.text and "Tuesday" in out.text
        assert "Comes up together with: " in out.text and "atlas (1)" in out.text
        none = await p.run("related", {"topic": "nobody"})
        assert "nothing about “nobody”" in none.text and "ana kovač" in none.text
        script = next(r["body"] for r in servers.adb.requests
                      if r["body"] and "UPDATE CaptainMemory SET" in r["body"].get("command", ""))
        assert script["language"] == "sqlscript" and script["command"].startswith("BEGIN;")
        assert "COMMIT RETRY 3;" in script["command"] and script["params"]["n1"] == "ana kovač"

    async def test_search_words_are_quoted_for_lucene(self):
        assert _ft_query('Ana OR "x: (y') == '"ana" "or" "x" "y"'
        assert _ft_query("?!") == ""

    async def test_search_fuses_words_and_meaning(self, servers):
        p = await _store("arcadedb")
        await p.run("remember", {"content": "Invoice 2207 is overdue."})
        await p.run("remember", {"content": "The office closes at six."})
        out = await p.run("search", {"query": "invoice 2207", "limit": 1})
        assert out.text.startswith("ArcadeDB — 1 item(s)") and "Invoice 2207" in out.text
        assert out.data["words"][0] == out.data["meaning"][0]


# ── through Flight Deck: the deck's connection, private addresses ───


@pytest.mark.parametrize("name", ["qdrant", "arcadedb"])
class TestThroughFlightDeck:
    async def test_the_deck_connection_gives_each_user_a_space(self, db, servers, agents, name):
        c = _client()
        r = c.put("/fd/external-memory/deck-connection", headers=_bearer(ALICE, "admin"),
                  json={"provider": name, "credentials": {**STORES[name], "space": "everyone"}})
        assert r.status_code == 200, r.text
        assert "space" not in r.json().get("credential_hints", {})  # each person gets their own
        for tok, fact in (("tok-bob", "Bob's sister is Ana."), ("tok-carol", "Carol prefers mornings.")):
            r = c.post("/fd/external-memory/agent/run", headers={"X-Agent-Auth": tok},
                       json={"action": "remember", "args": {"content": fact}})
            assert r.status_code == 200, r.text
        r = c.post("/fd/external-memory/agent/run", headers={"X-Agent-Auth": "tok-carol"},
                   json={"action": "search", "args": {"query": "sister Ana"}})
        assert "Bob's sister" not in r.json()["text"]
        m = c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json()
        assert m["provider"] == name and m["reaches"][0].endswith(f"space “{scope_id(BOB)}”")
        assert not any(p["name"] == "space" for a in m["actions"] for p in a["params"])  # agents never pick one
        if name == "qdrant":
            tenants = {p["payload"]["tenant"] for p in servers.qd.collections["captain_memory"]["points"].values()}
        else:
            tenants = {t for t, _ in servers.adb.rows}
        assert tenants == {scope_id(BOB), scope_id(CAROL)}

    async def test_a_users_own_connection_names_its_space(self, db, servers, name):
        c = _client()
        r = c.put("/fd/external-memory/connection", headers=_bearer(BOB),
                  json={"provider": name, "credentials": {**STORES[name], "space": "bob-private"}})
        assert r.status_code == 200, r.text
        st = r.json()
        assert st["source"] == "own" and st["scope_id"] == "bob-private"
        assert st["credential_hints"]["space"] == "bob-private"

    async def test_a_private_embeddings_address_is_for_admins_only(self, db, servers, name):
        c = _client()
        creds = {**STORES[name], "embedding_url": "http://127.0.0.1:11434/v1", "embedding_model": "nomic-embed-text"}
        r = c.put("/fd/external-memory/connection", headers=_bearer(BOB), json={"provider": name, "credentials": creds})
        assert r.status_code == 422 and "only an admin" in r.json()["detail"]
        assert not servers.emb_requests

    async def test_the_embeddings_address_is_checked_again_at_call_time(self, db, servers, monkeypatch, name):
        c = _client()
        creds = {**STORES[name], "embedding_url": EMB, "embedding_model": "m"}
        assert c.put("/fd/external-memory/connection", headers=_bearer(BOB),
                     json={"provider": name, "credentials": creds}).status_code == 200
        # The name now points into the network.
        monkeypatch.setattr(em_base.socket, "getaddrinfo",
                            lambda host, port: [(2, 1, 6, "", ("10.0.0.5" if "embed" in host else "93.184.216.34", 0))])
        svc._host_checks.clear()
        r = c.post("/fd/external-memory/run", headers=_bearer(BOB),
                   json={"action": "search", "args": {"query": "x"}})
        assert r.status_code == 400 and "only an admin" in r.json()["detail"]

    async def test_mirrors_go_to_the_owners_space(self, db, servers, agents, monkeypatch, name):
        monkeypatch.setattr(svc, "SEND_GAP_S", 0)
        c = _client()
        c.put("/fd/external-memory/deck-connection", headers=_bearer(ALICE, "admin"),
              json={"provider": name, "credentials": STORES[name]})
        for tok in ("tok-bob", "tok-carol"):
            r = c.post("/fd/external-memory/agent/mirror", headers={"X-Agent-Auth": tok},
                       json={"kind": "insight", "source_id": "i1", "content": f"fact from {tok}"})
            assert r.status_code == 200, r.text
        await svc.drain_once()
        assert (await svc.outbox().counts(BOB))["sent"] == 1
        if name == "qdrant":
            got = {p["payload"]["tenant"]: p["payload"]["content"]
                   for p in servers.qd.collections["captain_memory"]["points"].values()}
        else:
            got = {t: r["content"] for (t, _d), r in servers.adb.rows.items()}
        assert got == {scope_id(BOB): "fact from tok-bob", scope_id(CAROL): "fact from tok-carol"}

    async def test_a_store_unusable_as_configured_is_a_422_on_test(self, db, servers, name):
        c = _client()
        assert c.put("/fd/external-memory/connection", headers=_bearer(BOB),
                     json={"provider": name, "credentials": STORES[name]}).status_code == 200
        if name == "qdrant":
            servers.qd.collections["captain_memory"]["vectors"] = {"other-model": {"size": 4, "distance": "Cosine"}}
        else:
            servers.adb.databases.clear()
            servers.adb.may_create = False
        r = c.post("/fd/external-memory/test", headers=_bearer(BOB))
        assert r.status_code == 422, r.text


# ── embeddings ──────────────────────────────────────────────────────


class TestEmbedder:
    def test_vector_names(self):
        assert embeddings.Embedder({}).vector_name == "m2v-potion-base-8m"
        assert embeddings.Embedder({"embedding_model": "minishlab/potion-base-32M"}).vector_name == "m2v-potion-base-32m"
        e = embeddings.Embedder({"embedding_url": EMB, "embedding_model": "nomic-embed-text:v1.5"})
        assert e.vector_name == "nomic-embed-text-v1-5" and e.label == f"nomic-embed-text:v1.5 at {EMB}"

    async def test_remote_vectors_are_unit_length_and_in_order(self, servers):
        e = embeddings.Embedder({"embedding_url": EMB, "embedding_model": "m"})
        a, b = await e.embed(["alpha beta", "gamma"])
        assert abs(sum(x * x for x in a) - 1) < 1e-9 and a == pytest.approx(word_vectors("m", ["alpha beta"])[0])
        assert b == pytest.approx(word_vectors("m", ["gamma"])[0])
        assert servers.emb_requests[0]["auth"] == ""

    async def test_local_model_is_used_without_an_address(self, local_model):
        await embeddings.Embedder({}).embed(["x"])
        assert local_model == [embeddings.DEFAULT_LOCAL_MODEL]

    def test_info_survives_a_round_trip(self):
        info = ConnectionInfo(extra={"dims": 16, "hybrid": True})
        assert ConnectionInfo.from_dict(info.to_dict()).extra == {"dims": 16, "hybrid": True}


# ── automatic recall ────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["qdrant", "arcadedb"])
class TestRecall:
    async def _filled(self, name):
        p = await _store(name)
        for fact, topics in (("Project Atlas ships to the client in March.", "atlas"),
                             ("Ana prefers meetings on Tuesday mornings.", "ana"),
                             ("The office wifi password is kept in the vault.", "office")):
            await p.run("remember", {"content": fact, "topics": topics})
        return p

    async def test_only_what_bears_on_the_message(self, servers, name):
        p = await self._filled(name)
        hits = await p.recall("when does atlas ship?", 5)
        assert hits and hits[0].text == "Project Atlas ships to the client in March."
        assert all(isinstance(h, RecallHit) and h.id.startswith("captain-note-") and h.when for h in hits)
        assert "Ana" not in " ".join(h.text for h in hits)

    async def test_nothing_for_a_message_about_nothing_it_holds(self, servers, name):
        p = await self._filled(name)
        assert await p.recall("write me a haiku", 5) == []
        assert await p.recall("", 5) == []

    async def test_stopwords_never_count_as_a_match(self, servers, name):
        p = await self._filled(name)
        # "the", "is", "to" are in every stored fact; none of these words is.
        assert await p.recall("what is the plan for tonight?", 5) == []

    async def test_the_threshold_is_the_connections_own(self, servers, name):
        p = await _store(name, recall_floor="0")
        await p.run("remember", {"content": "Ana prefers meetings on Tuesday mornings."})
        assert [h.text for h in await p.recall("write me a haiku", 5)] == [
            "Ana prefers meetings on Tuesday mornings."]  # a threshold of 0 lets anything through
        if name == "qdrant":
            dense = servers.qd.requests[-1]["body"]["prefetch"][0]
            assert dense["score_threshold"] == 0.0

    async def test_recall_stays_in_the_persons_space(self, servers, name):
        bob, carol = await _store(name, "captain-bob"), await _store(name, "captain-carol")
        await bob.run("remember", {"content": "Bob's sister Ana lives in Split."})
        assert await carol.recall("where does Ana live", 5) == []
        assert [h.text for h in await bob.recall("where does Ana live", 5)] == ["Bob's sister Ana lives in Split."]


class TestRecallOthers:
    async def test_qdrant_without_word_search_recalls_by_meaning(self, servers):
        servers.qd.version = "1.15.4"
        p = await _store("qdrant")
        await p.run("remember", {"content": "Project Atlas ships in March."})
        assert [h.text for h in await p.recall("atlas ships when", 3)] == ["Project Atlas ships in March."]
        assert len(servers.qd.requests[-1]["body"]["prefetch"]) == 1

    async def test_hindsight_uses_its_own_cheapest_recall(self, monkeypatch):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"results": [
                {"text": "Ana leads the Acme pilot.", "type": "world", "document_id": "captain-note-1",
                 "occurred_start": "2026-11-02T00:00:00Z"},
                {"text": "Acme pays on the 5th.", "type": "world"}]})

        p = HindsightProvider({"base_url": "https://hs.example.com"}, scope="captain-bob",
                              transport=httpx.MockTransport(handler), max_retries=0)
        hits = await p.recall("who leads acme", 1)
        assert seen[0]["budget"] == "low" and seen[0]["max_tokens"] == 1500
        assert [h.to_dict() for h in hits] == [{"text": "Ana leads the Acme pilot.", "id": "captain-note-1",
                                                "when": "2026-11-02", "kind": "world"}]

    async def test_membase_fast_search_and_only_with_search_allowed(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"items": [
                {"id": "mem_1", "title": "Acme", "content": "Pilot starts in March.", "recorded_at": "2026-11-01T09:00Z"}]})

        p = MembaseProvider({"api_key": "mbk_test_12345678"}, ConnectionInfo(allowed=["search"]),
                            transport=httpx.MockTransport(handler), max_retries=0)
        hits = await p.recall("acme pilot", 4)
        assert seen == [{"q": "acme pilot", "limit": 4, "mode": "fast"}]
        assert hits[0].text == "Acme: Pilot starts in March." and hits[0].when == "2026-11-01"
        blind = MembaseProvider({"api_key": "mbk_test_12345678"}, ConnectionInfo(allowed=["memories.add"]),
                                transport=httpx.MockTransport(handler))
        assert await blind.recall("acme", 4) == [] and len(seen) == 1

    def test_the_threshold_field(self):
        with pytest.raises(ProviderInputError, match="from 0 to 1"):
            QdrantProvider.clean_credentials({**QD_CREDS, "recall_floor": "2"})
        with pytest.raises(ProviderInputError, match="from 0 to 1"):
            QdrantProvider.clean_credentials({**QD_CREDS, "recall_floor": "high"})
        assert QdrantProvider.clean_credentials({**QD_CREDS, "recall_floor": "0.40"})["recall_floor"] == "0.4"
        assert embeddings.Embedder({}).floor == embeddings.DEFAULT_FLOOR_LOCAL
        assert embeddings.Embedder({"embedding_url": EMB, "embedding_model": "m"}).floor == 0.3
        assert embeddings.Embedder({"recall_floor": "0.5"}).floor == 0.5


# ── reading priority, through Flight Deck ───────────────────────────


def _connect_qdrant(c, uid=BOB):
    r = c.put("/fd/external-memory/connection", headers=_bearer(uid),
              json={"provider": "qdrant", "credentials": QD_CREDS})
    assert r.status_code == 200, r.text


class TestReadingPriority:
    async def test_alongside_by_default_and_set_per_person(self, db, servers):
        c = _client()
        _connect_qdrant(c)
        st = c.get("/fd/external-memory/status", headers=_bearer(BOB)).json()
        assert st["priority"] == "alongside" and st["priorities"] == ["first", "alongside", "on_demand"]
        assert st["capture_every"] == svc.CAPTURE_EVERY
        r = c.put("/fd/external-memory/priority", headers=_bearer(BOB), json={"priority": "first"})
        assert r.status_code == 200 and r.json()["priority"] == "first"
        assert any("MAIN memory" in rule for rule in r.json()["rules"])
        assert c.get("/fd/external-memory/status", headers=_bearer(CAROL)).json()["priority"] == "alongside"
        r = c.put("/fd/external-memory/priority", headers=_bearer(BOB), json={"priority": "always"})
        assert r.status_code == 422

    async def test_the_manifest_tells_the_agent(self, db, servers, agents):
        c = _client()
        _connect_qdrant(c)
        m = c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json()
        assert (m["priority"], m["recall_limit"], m["capture_every"]) == ("alongside", 5, svc.CAPTURE_EVERY)
        assert any("put in front of you" in rule for rule in m["rules"])
        c.put("/fd/external-memory/priority", headers=_bearer(BOB), json={"priority": "first"})
        m = c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json()
        assert (m["priority"], m["recall_limit"]) == ("first", 8)
        assert any("search it FIRST" in rule for rule in m["rules"])
        c.put("/fd/external-memory/priority", headers=_bearer(BOB), json={"priority": "on_demand"})
        m = c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json()
        assert (m["priority"], m["recall_limit"]) == ("on_demand", 0)

    async def test_agent_recall(self, db, servers, agents):
        c = _client()
        _connect_qdrant(c)
        c.post("/fd/external-memory/agent/run", headers={"X-Agent-Auth": "tok-bob"},
               json={"action": "remember", "args": {"content": "Project Atlas ships in March."}})
        r = c.post("/fd/external-memory/agent/recall", headers={"X-Agent-Auth": "tok-bob"},
                   json={"query": "when does atlas ship"})
        body = r.json()
        assert r.status_code == 200 and body["priority"] == "alongside" and body["label"] == "Qdrant"
        assert [h["text"] for h in body["hits"]] == ["Project Atlas ships in March."]
        # Carol has no memory of her own and the deck shares none.
        assert c.post("/fd/external-memory/agent/recall", headers={"X-Agent-Auth": "tok-carol"},
                      json={"query": "atlas"}).json() == {"connected": False, "hits": []}

    async def test_on_demand_asks_nothing(self, db, servers, agents):
        c = _client()
        _connect_qdrant(c)
        c.put("/fd/external-memory/priority", headers=_bearer(BOB), json={"priority": "on_demand"})
        n = len(servers.qd.requests)
        body = c.post("/fd/external-memory/agent/recall", headers={"X-Agent-Auth": "tok-bob"},
                      json={"query": "atlas"}).json()
        assert body["hits"] == [] and body["priority"] == "on_demand"
        assert len(servers.qd.requests) == n  # the memory wasn't asked

    async def test_a_slow_or_failing_memory_never_fails_the_turn(self, db, servers, agents, monkeypatch):
        c = _client()
        _connect_qdrant(c)

        async def slow(self, query, limit):
            import asyncio
            await asyncio.sleep(5)

        monkeypatch.setitem(svc.RECALL, "alongside", (5, 0.05))
        monkeypatch.setattr(QdrantProvider, "recall", slow)
        body = c.post("/fd/external-memory/agent/recall", headers={"X-Agent-Auth": "tok-bob"},
                      json={"query": "atlas"}).json()
        assert body["hits"] == [] and "in time" in body["error"]

        async def broken(self, query, limit):
            raise ProviderError("Qdrant: boom", status=500)

        monkeypatch.setattr(QdrantProvider, "recall", broken)
        body = c.post("/fd/external-memory/agent/recall", headers={"X-Agent-Auth": "tok-bob"},
                      json={"query": "atlas"}).json()
        assert body["hits"] == [] and body["error"] == "Qdrant: boom"

    async def test_recall_is_logged(self, db, servers, agents):
        c = _client()
        _connect_qdrant(c)
        c.post("/fd/external-memory/agent/recall", headers={"X-Agent-Auth": "tok-bob"}, json={"query": "atlas"})
        cur = await db._db.execute("SELECT detail FROM usage_logs WHERE event_type = 'external_memory_call'")
        rows = [json.loads(r[0]) for r in await cur.fetchall()]
        assert any(r["action"] == "recall" and r["source"] == "recall" and r["agent"] == "bob-bot" for r in rows)
