"""Qdrant and ArcadeDB connectors against REAL servers — skipped unless asked:

    docker run -d -p 16333:6333 qdrant/qdrant
    docker run -d -p 12480:2480 -e JAVA_OPTS="-Darcadedb.server.rootPassword=captaintest1" arcadedata/arcadedb
    CAPTAIN_LIVE_QDRANT_URL=http://localhost:16333 \\
    CAPTAIN_LIVE_ARCADEDB_URL=http://localhost:12480 CAPTAIN_LIVE_ARCADEDB_PASSWORD=captaintest1 \\
    .venv/bin/python -m pytest tests/test_flight_deck/test_external_memory_stores_live.py -q

Every run uses a collection / database of its own and removes it after. The
deck's embedding model is swapped for a word-hashing one (no model download);
everything else — REST paths, SQL, filters, indexes, upserts — is the real
server's. The fakes in ``test_external_memory_stores.py`` can't prove the SQL
parses or the filters isolate; these do.
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

from captain_claw.flight_deck.external_memory import embeddings
from captain_claw.flight_deck.external_memory.arcadedb import ArcadeDBProvider
from captain_claw.flight_deck.external_memory.base import MemoryItem, ProviderError
from captain_claw.flight_deck.external_memory.qdrant import QdrantProvider

from .test_external_memory_stores import DIMS, word_vectors

QD_URL = os.environ.get("CAPTAIN_LIVE_QDRANT_URL", "")
QD_KEY = os.environ.get("CAPTAIN_LIVE_QDRANT_KEY", "")
ADB_URL = os.environ.get("CAPTAIN_LIVE_ARCADEDB_URL", "")
ADB_USER = os.environ.get("CAPTAIN_LIVE_ARCADEDB_USER", "root")
ADB_PASS = os.environ.get("CAPTAIN_LIVE_ARCADEDB_PASSWORD", "")


@pytest.fixture(autouse=True)
def local_model(monkeypatch):
    monkeypatch.setattr(embeddings, "_local_vectors", word_vectors)


@pytest.fixture()
async def qdrant():
    if not QD_URL:
        pytest.skip("set CAPTAIN_LIVE_QDRANT_URL to run against a real Qdrant")
    creds = {"base_url": QD_URL, "collection": f"captain_live_{uuid.uuid4().hex[:10]}"}
    if QD_KEY:
        creds["api_key"] = QD_KEY
    yield creds
    async with httpx.AsyncClient(base_url=QD_URL, headers={"api-key": QD_KEY} if QD_KEY else {}) as c:
        await c.delete(f"/collections/{creds['collection']}")


@pytest.fixture()
async def arcadedb():
    if not (ADB_URL and ADB_PASS):
        pytest.skip("set CAPTAIN_LIVE_ARCADEDB_URL and CAPTAIN_LIVE_ARCADEDB_PASSWORD to run against a real ArcadeDB")
    creds = {"base_url": ADB_URL, "username": ADB_USER, "password": ADB_PASS,
             "database": f"captain_live_{uuid.uuid4().hex[:10]}"}
    yield creds
    async with httpx.AsyncClient(base_url=ADB_URL, auth=(ADB_USER, ADB_PASS)) as c:
        await c.post("/api/v1/server", json={"command": f"drop database {creds['database']}"})


async def _pair(cls, creds):
    info = await cls(creds, scope="captain-check").verify()
    return info, cls(creds, info, scope="captain-alice"), cls(creds, info, scope="captain-bob")


async def _exercise(alice, bob) -> None:
    await alice.run("remember", {"content": "Ana prefers meetings on Tuesday mornings.", "topics": "Ana, meetings"})
    await alice.run("remember", {"content": "Project Atlas ships in March.", "topics": "Atlas, Ana"})
    await alice.run("remember", {"content": "Project Atlas ships in March.", "topics": "Atlas, Ana"})  # replaces
    await bob.run("remember", {"content": "Bob's dentist is every second Friday.", "topics": "dentist"})
    item = MemoryItem(kind="insight", source_id="1", content="Ana's daughter starts school in September.",
                      tags=["family"], agent="Bob Bot", created_at="2026-11-09T08:00:00",
                      meta={"category": "contact", "entity_key": "ana-school"})
    await alice.mirror(item)
    item.content, item.source_id = "Ana's daughter starts school in October.", "2"
    await alice.mirror(item)  # same entity_key: replaces

    listed = await alice.run("list", {"limit": 50})
    assert listed.text.count("[id ") == 3, listed.text
    assert "October" in listed.text and "September" not in listed.text
    found = await alice.run("search", {"query": "when does atlas ship"})
    assert found.text.splitlines()[1].startswith("1. Project Atlas ships in March."), found.text
    assert "Bob's dentist" not in (await alice.run("search", {"query": "dentist friday"})).text
    bobs = (await bob.run("search", {"query": "Ana atlas meetings"})).text.split("\n", 1)[1]  # past the header
    assert "Ana" not in bobs and "Bob's dentist" in bobs
    topic = await alice.run("search", {"query": "march", "topic": "meetings"})
    assert "Tuesday" in topic.text and "Atlas" not in topic.text.split("\n", 1)[1]
    assert "ana (2)" in (await alice.run("topics", {})).text

    with pytest.raises(ProviderError) as exc:
        await alice.run("forget", {"id": "captain-insight-ana-school"})
    assert exc.value.status == 428
    with pytest.raises(ProviderError) as exc:
        await bob.run("forget", {"id": "captain-insight-ana-school", "confirm": True})
    assert exc.value.status == 404  # not in Bob's space
    await alice.run("forget", {"id": "captain-insight-ana-school", "confirm": True})
    assert "October" not in (await alice.run("list", {})).text
    assert "Holds 2 item(s)" in (await alice.run("capabilities", {})).text
    assert "Holds 1 item(s)" in (await bob.run("capabilities", {})).text


async def test_qdrant_live(qdrant):
    info, alice, bob = await _pair(QdrantProvider, qdrant)
    assert info.extra["dims"] == DIMS and info.extra["vector"] == "m2v-potion-base-8m"
    await _exercise(alice, bob)
    # Checked again on an existing collection: accepted as it is.
    assert (await QdrantProvider(qdrant, scope="x").verify()).extra["hybrid"] == info.extra["hybrid"]


async def test_arcadedb_live(arcadedb):
    info, alice, bob = await _pair(ArcadeDBProvider, arcadedb)
    assert info.extra["dims"] == DIMS
    await _exercise(alice, bob)
    related = await alice.run("related", {"topic": "ana"})
    assert "Tuesday" in related.text and "Atlas" in related.text and "atlas (1)" in related.text
    # Orphaned topics went with the forgotten insight.
    assert "family" not in (await alice.run("topics", {})).text
    await ArcadeDBProvider(arcadedb, scope="x").verify()  # the schema again: idempotent
