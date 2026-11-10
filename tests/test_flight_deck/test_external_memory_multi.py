"""Several external memories per person:

* a person adds up to MAX_MEMORIES of their own, each with a name of its own
  (the same service twice is fine); the deck's shared memory is one more in
  their list, which they can switch off but not remove;
* each memory has its own reading priority and its own "copied in on its
  own" choices; recall asks every memory not on_demand and returns one group
  per memory; a mirrored item is queued once per memory that takes its kind;
* the agent's tool spans them: search and other reading actions look in all
  of them, remember saves to every one that can save, and anything else
  (forget…) names one — unless only one memory offers it;
* a connection from before several memories is moved into the list on first
  read, with its choices; someone who had their own keeps the shared one off,
  as before; queued items from then still reach that memory.

Qdrant and ArcadeDB are the fakes from ``test_external_memory_stores``.
"""

# The fixtures are imported from test_external_memory_stores (not a conftest:
# its autouse stand-in model must not reach other tests); pytest passes them
# in by name, which ruff reads as a redefinition.
# ruff: noqa: F811

from __future__ import annotations

import json

from captain_claw.flight_deck.external_memory import service as svc
from captain_claw.flight_deck.external_memory.base import ConnectionInfo

from .test_external_memory_stores import (  # noqa: F401 — fixtures
    ADB_CREDS,
    ALICE,
    BOB,
    CAROL,
    QD_CREDS,
    _bearer,
    _client,
    agents,
    db,
    local_model,
    servers,
)


def _add(c, provider, creds, uid=BOB, name=""):
    body = {"provider": provider, "credentials": creds}
    if name:
        body["name"] = name
    r = c.put("/fd/external-memory/connection", headers=_bearer(uid), json=body)
    assert r.status_code == 200, r.text
    return r.json()["memory_id"]


def _set(c, mid, uid=BOB, **data):
    return c.put(f"/fd/external-memory/memories/{mid}", headers=_bearer(uid), json=data)


def _agent_run(c, action, args, tok="tok-bob", memory=None):
    body = {"action": action, "args": args}
    if memory:
        body["memory"] = memory
    return c.post("/fd/external-memory/agent/run", headers={"X-Agent-Auth": tok}, json=body)


def _two(c):
    """Bob: a Qdrant called Work and an ArcadeDB called Notes."""
    return _add(c, "qdrant", QD_CREDS, name="Work"), _add(c, "arcadedb", ADB_CREDS, name="Notes")


class TestList:
    async def test_add_several_with_names_of_their_own(self, db, servers):
        c = _client()
        first = _add(c, "qdrant", QD_CREDS)
        second = _add(c, "qdrant", {**QD_CREDS, "collection": "second"})
        third = _add(c, "arcadedb", ADB_CREDS, name="Notes")
        st = c.get("/fd/external-memory/status", headers=_bearer(BOB)).json()
        assert [(m["id"], m["name"], m["provider"]) for m in st["memories"]] == [
            (first, "Qdrant", "qdrant"), (second, "Qdrant 2", "qdrant"), (third, "Notes", "arcadedb")]
        assert len({first, second, third}) == 3 and st["connected"]
        r = c.put("/fd/external-memory/connection", headers=_bearer(BOB),
                  json={"provider": "qdrant", "credentials": QD_CREDS, "name": "notes"})
        assert r.status_code == 422 and "already have a memory called" in r.json()["detail"]

    async def test_at_most_max_memories(self, db, servers, monkeypatch):
        monkeypatch.setattr(svc, "MAX_MEMORIES", 2)
        c = _client()
        _two(c)
        r = c.put("/fd/external-memory/connection", headers=_bearer(BOB),
                  json={"provider": "qdrant", "credentials": QD_CREDS})
        assert r.status_code == 422 and "remove one" in r.json()["detail"]

    async def test_rename_replace_remove(self, db, servers):
        c = _client()
        work, notes = _two(c)
        st = _set(c, work, name="Work stuff").json()
        assert [m["name"] for m in st["memories"]] == ["Work stuff", "Notes"]
        assert _set(c, work, name="notes").status_code == 422
        assert _set(c, work, name="x" * 80).status_code == 422
        # New credentials for one memory keep its id, name and choices.
        _set(c, notes, priority="first")
        r = c.put("/fd/external-memory/connection", headers=_bearer(BOB),
                  json={"provider": "arcadedb", "memory": notes,
                        "credentials": {**ADB_CREDS, "database": "other_db"}})
        mem = r.json()["memories"][1]
        assert (mem["id"], mem["name"], mem["priority"]) == (notes, "Notes", "first")
        assert "other_db" in mem["reaches"][0]
        st = c.delete(f"/fd/external-memory/connection?memory={work}", headers=_bearer(BOB)).json()
        assert [m["name"] for m in st["memories"]] == ["Notes"]
        assert c.delete("/fd/external-memory/connection?memory=m_gone", headers=_bearer(BOB)).status_code == 404

    async def test_choices_are_per_memory_and_per_person(self, db, servers):
        c = _client()
        work, notes = _two(c)
        _set(c, work, priority="first", mirror={"reflection": True})
        st = _set(c, notes, priority="on_demand", mirror={"insight": False}, enabled=False).json()
        w, n = st["memories"]
        assert (w["priority"], w["mirror"]["reflection"], w["mirror"]["insight"], w["enabled"]) == (
            "first", True, True, True)
        assert (n["priority"], n["mirror"]["insight"], n["enabled"]) == ("on_demand", False, False)
        assert _set(c, notes, enabled="no").status_code == 422
        assert c.get("/fd/external-memory/status", headers=_bearer(CAROL)).json()["memories"] == []


class TestSharedMemory:
    def _deck(self, c):
        r = c.put("/fd/external-memory/deck-connection", headers=_bearer(ALICE, "admin"),
                  json={"provider": "qdrant", "credentials": QD_CREDS})
        assert r.status_code == 200, r.text

    async def test_one_more_memory_with_this_persons_choices(self, db, servers, agents):
        c = _client()
        self._deck(c)
        notes = _add(c, "arcadedb", ADB_CREDS, name="Notes")
        st = c.get("/fd/external-memory/status", headers=_bearer(BOB)).json()
        assert [(m["id"], m["name"], m["source"]) for m in st["memories"]] == [
            (notes, "Notes", "own"), ("deck", "Qdrant (shared)", "deck")]
        _set(c, "deck", priority="first")
        assert c.get("/fd/external-memory/status", headers=_bearer(CAROL)).json()["memories"][0]["priority"] == \
            "alongside"  # Bob's choice, not Carol's
        assert _set(c, "deck", name="Mine").status_code == 422
        m = c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json()
        assert [x["name"] for x in m["memories"]] == ["Notes", "Qdrant (shared)"]
        # Switched off: not used, still listed.
        _set(c, "deck", enabled=False)
        m = c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json()
        assert [x["name"] for x in m["memories"]] == ["Notes"]
        st = c.get("/fd/external-memory/status", headers=_bearer(BOB)).json()
        assert st["memories"][1]["enabled"] is False
        r = c.put("/fd/external-memory/connection", headers=_bearer(BOB),
                  json={"provider": "qdrant", "memory": "deck", "credentials": QD_CREDS})
        assert r.status_code == 422


class TestTheAgentsTool:
    async def test_the_schema_spans_the_memories(self, db, servers, agents):
        c = _client()
        _two(c)
        _set(c, _add(c, "qdrant", {**QD_CREDS, "collection": "x"}, name="Spare"), enabled=False)
        m = c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json()
        assert m["label"] == "Work, Notes" and m["provider"] == "several"
        actions = {a["name"]: a for a in m["actions"]}
        assert "related" in actions and "(Only in: Notes.)" in actions["related"]["description"]
        memory = next(p for p in actions["search"]["params"] if p["name"] == "memory")
        assert memory["enum"] == ["Work", "Notes"] and not memory["required"]
        assert "every one that can save" in next(
            p for p in actions["remember"]["params"] if p["name"] == "memory")["description"]
        assert any(r.startswith("The person has 2 external memories: Work (Qdrant") for r in m["rules"])
        assert any(r.startswith("Notes: ArcadeDB keeps") for r in m["rules"])
        assert sum("Shared by ALL" in r or "shared by ALL" in r for r in m["rules"]) == 1  # common rules once

    async def test_one_memory_reads_as_before(self, db, servers, agents):
        c = _client()
        _add(c, "qdrant", QD_CREDS)
        m = c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json()
        assert m["label"] == "Qdrant" and m["provider"] == "qdrant"
        assert not any(p["name"] == "memory" for a in m["actions"] for p in a["params"])
        text = _agent_run(c, "remember", {"content": "Atlas ships in March."}).json()["text"]
        assert text.startswith("Saved to Qdrant (id captain-note-")

    async def test_remember_saves_to_every_memory_unless_one_is_named(self, db, servers, agents):
        c = _client()
        _two(c)
        r = _agent_run(c, "remember", {"content": "Project Atlas ships in March.", "topics": "atlas"})
        assert r.status_code == 200, r.text
        text = r.json()["text"]
        assert "── Work (Qdrant) ──\nSaved to Qdrant" in text and "── Notes (ArcadeDB) ──\nSaved to ArcadeDB" in text
        assert len(servers.qd.collections["captain_memory"]["points"]) == 1 and len(servers.adb.rows) == 1
        _agent_run(c, "remember", {"content": "Only for notes."}, memory="notes")  # by name, any case
        assert len(servers.qd.collections["captain_memory"]["points"]) == 1 and len(servers.adb.rows) == 2
        _agent_run(c, "remember", {"content": "Only for work.", "memory": "Work"})  # inside args works too
        assert len(servers.qd.collections["captain_memory"]["points"]) == 2

    async def test_search_looks_in_all_and_changes_name_one(self, db, servers, agents):
        c = _client()
        _two(c)
        _agent_run(c, "remember", {"content": "Ana leads Atlas."}, memory="Work")
        _agent_run(c, "remember", {"content": "Atlas ships in March.", "topics": "atlas"}, memory="Notes")
        text = _agent_run(c, "search", {"query": "atlas"}).json()["text"]
        assert "── Work (Qdrant) ──" in text and "Ana leads Atlas." in text
        assert "── Notes (ArcadeDB) ──" in text and "Atlas ships in March." in text
        only = _agent_run(c, "search", {"query": "atlas"}, memory="Notes").json()["text"]
        assert "Ana leads" not in only and "──" not in only
        r = _agent_run(c, "forget", {"id": "x", "confirm": True})
        assert r.status_code == 422 and "say which with memory: Work, Notes" in r.json()["detail"]
        # Only one memory offers it: no need to name it.
        related = _agent_run(c, "related", {"topic": "atlas"}).json()["text"]
        assert related.startswith("ArcadeDB — about “atlas”") and "Atlas ships in March." in related
        r = _agent_run(c, "search", {"query": "x"}, memory="Nope")
        assert r.status_code == 422 and "No memory called “Nope”" in r.json()["detail"]
        r = _agent_run(c, "reflect", {"question": "x"})
        assert r.status_code == 422 and "No memory offers 'reflect'" in r.json()["detail"]

    async def test_one_failing_memory_doesnt_hide_the_others(self, db, servers, agents):
        c = _client()
        _two(c)
        _agent_run(c, "remember", {"content": "Atlas ships in March."}, memory="Notes")
        servers.adb.password = "changed"  # Notes now refuses
        text = _agent_run(c, "search", {"query": "atlas"}).json()["text"]
        assert "── Notes (ArcadeDB) — failed: ArcadeDB refused" in text and "── Work (Qdrant) ──" in text


class TestRecall:
    async def test_one_group_per_memory_with_its_priority(self, db, servers, agents):
        c = _client()
        work, notes = _two(c)
        spare = _add(c, "qdrant", {**QD_CREDS, "collection": "spare"}, name="Spare")
        _set(c, work, priority="first")
        _set(c, spare, priority="on_demand")
        _agent_run(c, "remember", {"content": "Project Atlas ships in March."})
        body = c.post("/fd/external-memory/agent/recall", headers={"X-Agent-Auth": "tok-bob"},
                      json={"query": "when does atlas ship"}).json()
        assert [(g["name"], g["priority"]) for g in body["groups"]] == [("Work", "first"), ("Notes", "alongside")]
        assert all(g["hits"][0]["text"] == "Project Atlas ships in March." for g in body["groups"])
        assert body["priority"] == "first" and len(body["hits"]) == 2
        m = c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json()
        assert (m["priority"], m["recall_limit"], m["recall_timeout"]) == ("first", 8, 6.0)


class TestMirroring:
    async def test_each_memory_takes_the_kinds_it_chose(self, db, servers, agents, monkeypatch):
        monkeypatch.setattr(svc, "SEND_GAP_S", 0)
        c = _client()
        work, notes = _two(c)
        _set(c, notes, mirror={"insight": False, "reflection": True})
        r = c.post("/fd/external-memory/agent/mirror", headers={"X-Agent-Auth": "tok-bob"},
                   json={"kind": "insight", "source_id": "i1", "content": "Ana leads Atlas."})
        assert r.json() == {"queued": True, "reason": "", "memories": ["Work"]}
        r = c.post("/fd/external-memory/agent/mirror", headers={"X-Agent-Auth": "tok-bob"},
                   json={"kind": "reflection", "source_id": "r1", "content": "Keep answers short for Bob."})
        assert r.json()["memories"] == ["Notes"]
        r = c.post("/fd/external-memory/agent/mirror", headers={"X-Agent-Auth": "tok-bob"},
                   json={"kind": "intention", "source_id": "t1", "content": "Call Ana on Friday."})
        assert r.json()["memories"] == ["Work", "Notes"]
        m = c.get("/fd/external-memory/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json()
        assert m["mirror"] == ["insight", "intention", "reflection"]  # what any memory takes
        await svc.drain_once()
        qd = {p["payload"]["content"] for p in servers.qd.collections["captain_memory"]["points"].values()}
        adb = {r["content"] for r in servers.adb.rows.values()}
        assert qd == {"Ana leads Atlas.", "Call Ana on Friday."}
        assert adb == {"Keep answers short for Bob.", "Call Ana on Friday."}
        st = c.get("/fd/external-memory/status", headers=_bearer(BOB)).json()
        assert [m["outbox"]["sent"] for m in st["memories"]] == [2, 2] and st["outbox"]["sent"] == 4

    async def test_a_refusing_memory_holds_only_its_own_queue(self, db, servers, agents, monkeypatch):
        monkeypatch.setattr(svc, "SEND_GAP_S", 0)
        c = _client()
        work, notes = _two(c)
        servers.adb.password = "changed"
        c.post("/fd/external-memory/agent/mirror", headers={"X-Agent-Auth": "tok-bob"},
               json={"kind": "insight", "source_id": "i1", "content": "Ana leads Atlas."})
        await svc.drain_once()
        st = c.get("/fd/external-memory/status", headers=_bearer(BOB)).json()
        w, n = st["memories"]
        assert w["outbox"]["sent"] == 1 and n["outbox"]["pending"] == 1 and n["verify_error"] == "refused"

    async def test_removing_a_memory_drops_only_its_queue(self, db, servers, agents):
        c = _client()
        work, notes = _two(c)
        c.post("/fd/external-memory/agent/mirror", headers={"X-Agent-Auth": "tok-bob"},
               json={"kind": "insight", "source_id": "i1", "content": "Ana leads Atlas."})
        c.delete(f"/fd/external-memory/connection?memory={notes}", headers=_bearer(BOB))
        rows = await svc.outbox().recent(BOB)
        assert [r["target"] for r in rows] == [work]

    async def test_switched_off_takes_nothing(self, db, servers, agents):
        c = _client()
        work, notes = _two(c)
        _set(c, notes, enabled=False)
        r = c.post("/fd/external-memory/agent/mirror", headers={"X-Agent-Auth": "tok-bob"},
                   json={"kind": "insight", "source_id": "i1", "content": "Ana leads Atlas."})
        assert r.json()["memories"] == ["Work"]


class TestMovingOver:
    async def _legacy(self, db, uid, conn, *, mirror=None, priority=None):
        await db.set_settings(uid, {"external_memory:connection": json.dumps(svc._closed(conn))})
        if mirror is not None:
            await db.set_settings(uid, {"external_memory:mirror": json.dumps(mirror)})
        if priority is not None:
            await db.set_settings(uid, {"external_memory:priority": priority})

    async def test_an_old_connection_becomes_the_first_memory(self, db, servers, agents, monkeypatch):
        monkeypatch.setattr(svc, "SEND_GAP_S", 0)
        c = _client()
        c.put("/fd/external-memory/deck-connection", headers=_bearer(ALICE, "admin"),
              json={"provider": "arcadedb", "credentials": ADB_CREDS})
        info = await svc.providers.build("qdrant", QD_CREDS, scope="x").verify()
        await self._legacy(db, BOB, {"provider": "qdrant", "credentials": QD_CREDS, "info": info,
                                     "verified": True, "private_ok": False},
                           mirror={"insight": False, "reflection": True}, priority="first")
        # Something queued before, for the Qdrant: no target yet.
        await svc.outbox().enqueue(BOB, "qdrant", "reflection", "r0", {
            "kind": "reflection", "source_id": "r0", "content": "Queued before the move."})
        st = c.get("/fd/external-memory/status", headers=_bearer(BOB)).json()
        own, deck = st["memories"]
        assert (own["name"], own["provider"], own["priority"], own["verified"]) == ("Qdrant", "qdrant", "first", True)
        assert own["mirror"]["insight"] is False and own["mirror"]["reflection"] is True
        assert deck["enabled"] is False  # their own used to replace the shared one: still does
        assert await db.get_setting(BOB, "external_memory:connection") in (None, "")
        assert json.loads(await db.get_setting(BOB, "external_memory:memories"))[0]["id"] == own["id"]
        await svc.drain_once()
        assert "Queued before the move." in {
            p["payload"]["content"] for p in servers.qd.collections["captain_memory"]["points"].values()}
        # Read again: moved once, not twice.
        assert len(c.get("/fd/external-memory/status", headers=_bearer(BOB)).json()["memories"]) == 2

    async def test_someone_on_the_shared_memory_keeps_their_choices_there(self, db, servers):
        c = _client()
        c.put("/fd/external-memory/deck-connection", headers=_bearer(ALICE, "admin"),
              json={"provider": "qdrant", "credentials": QD_CREDS})
        await db.set_settings(CAROL, {"external_memory:priority": "first",
                                      "external_memory:mirror": json.dumps({"lesson": False})})
        (deck,) = c.get("/fd/external-memory/status", headers=_bearer(CAROL)).json()["memories"]
        assert (deck["enabled"], deck["priority"], deck["mirror"]["lesson"]) == (True, "first", False)

    def test_info_survives(self):
        assert ConnectionInfo.from_dict(ConnectionInfo(extra={"dims": 3}).to_dict()).extra == {"dims": 3}
