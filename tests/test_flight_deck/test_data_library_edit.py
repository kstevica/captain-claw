"""Editing a Data library table by hand: change cells, add a row, delete one.

Every write goes through the agent's own datastore manager, so it obeys what
the agent's writes obey (protected rows and cells, a unique key) and stamps
the table's last change; the library's row count and last change follow at
once. An edit carries what the person saw in the cells it changes: a cell
that holds something else now (the agent wrote it meanwhile) refuses the
edit instead of being overwritten. Values typed by hand are stored as their
column's type, or refused in plain words. Another user's table is refused.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import httpx
import pytest

from captain_claw.datastore import DatastoreManager
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import data_library as dl
from captain_claw.flight_deck import rate_limiter, server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

BOB = "user-bob"
CAROL = "user-carol"


def _ds_path(data: Path) -> Path:
    return data / "bob-agent" / "data" / "home-config-parent" / ".captain-claw" / "datastore.db"


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Auth-enabled deck: Bob's stopped agent keeps a ``leads`` table (with
    a unique email); Carol owns an agent of her own. Yields the table's id."""
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid in (BOB, CAROL):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid))
    await db._db.commit()

    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(dl, "_STORE", dl.DataStore(tmp_path / "data_library.db"))

    server._save_process_registry({
        "bob-agent": {"slug": "bob-agent", "name": "Bob's agent", "web_port": 24101,
                      "web_auth": "bob-tok", "owner": BOB, "pid": None},
        "carol-agent": {"slug": "carol-agent", "name": "Carol's agent", "web_port": 24102,
                        "web_auth": "carol-tok", "owner": CAROL, "pid": None},
    })
    mgr = DatastoreManager(db_path=_ds_path(data))
    await mgr.create_table("leads", [
        {"name": "org", "type": "text"}, {"name": "email", "type": "text"},
        {"name": "score", "type": "integer"}, {"name": "weight", "type": "real"},
        {"name": "excluded", "type": "boolean"}, {"name": "last_touch", "type": "date"},
        {"name": "meta", "type": "json"},
    ], unique=["email"])
    await mgr.insert_rows("leads", [
        {"org": "Acme", "email": "a@acme.co", "score": 7, "weight": 1.5, "excluded": 0,
         "last_touch": "2026-11-01", "meta": '{"tier": 1}'},
        {"org": "Globex", "email": "g@globex.co", "score": None, "weight": None, "excluded": 1,
         "last_touch": None, "meta": None},
    ])
    await mgr.close()
    dl.get_store().sync_agent("bob-agent", dl.scan_agent("bob-agent", {}))
    try:
        yield dl.DataStore.table_id("bob-agent", "leads")
    finally:
        await db.close()
        fd_auth._db = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=("203.0.113.9", 40001)), base_url="http://fd.test")


def _auth(uid: str = BOB) -> dict:
    return {"Authorization": f"Bearer {create_access_token(uid)}"}


async def _patch(table_id: str, rowid: int, values: dict, expect: dict | None = None, uid: str = BOB):
    body: dict = {"values": values}
    if expect is not None:
        body["expect"] = expect
    async with _client() as c:
        return await c.patch(f"/fd/data-library/tables/{table_id}/rows/{rowid}", json=body, headers=_auth(uid))


async def _post(table_id: str, values: dict, uid: str = BOB):
    async with _client() as c:
        return await c.post(f"/fd/data-library/tables/{table_id}/rows", json={"values": values}, headers=_auth(uid))


async def _delete(table_id: str, rowid: int, uid: str = BOB):
    async with _client() as c:
        return await c.delete(f"/fd/data-library/tables/{table_id}/rows/{rowid}", headers=_auth(uid))


def _stored(rowid: int) -> dict | None:
    path = dl.datastore_path("bob-agent")
    assert path is not None
    with closing(sqlite3.connect(path)) as c:
        c.row_factory = sqlite3.Row
        r = c.execute("SELECT * FROM ds_leads WHERE _id = ?", (rowid,)).fetchone()
        return dict(r) if r else None


def _write(sql: str, *args) -> None:
    """Change the file straight on disk, as the running agent would."""
    path = dl.datastore_path("bob-agent")
    assert path is not None
    with closing(sqlite3.connect(path)) as c:
        c.execute(sql, args)
        c.commit()


async def test_a_cell_changes_and_the_table_follows(deck: str):
    before = dl.get_store().get(deck)
    r = await _patch(deck, 1, {"org": "Acme d.o.o."}, expect={"org": "Acme"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["row"]["org"] == "Acme d.o.o."
    assert body["row"]["__rowid"] == 1
    assert _stored(1)["org"] == "Acme d.o.o."
    # The table's last change is stamped, and the library shows it at once.
    assert body["table"]["updated_at"] > before["updated_at"]
    assert dl.get_store().get(deck)["updated_at"] == body["table"]["updated_at"]


async def test_values_are_stored_as_their_column_type(deck: str):
    r = await _patch(deck, 2, {"score": "12", "weight": "2.25", "excluded": "no",
                               "last_touch": "2026-11-11", "meta": '["a", 1]'})
    assert r.status_code == 200, r.text
    row = _stored(2)
    assert row["score"] == 12 and isinstance(row["score"], int)
    assert row["weight"] == 2.25
    assert row["excluded"] == 0
    assert row["last_touch"] == "2026-11-11"
    assert row["meta"] == '["a", 1]'


async def test_emptying_a_cell_stores_no_value(deck: str):
    r = await _patch(deck, 1, {"org": "  ", "score": ""})
    assert r.status_code == 200, r.text
    assert _stored(1)["org"] is None
    assert _stored(1)["score"] is None


@pytest.mark.parametrize(("values", "words"), [
    ({"score": "seven"}, "Score takes a whole number."),
    ({"weight": "heavy"}, "Weight takes a number."),
    ({"excluded": "maybe"}, "Excluded takes yes or no."),
    ({"last_touch": "11/11/2026"}, "Last touch takes a date like 2026-11-11."),
    ({"meta": "{oops"}, "Meta takes JSON"),
    ({"nope": "x"}, "The table has no column 'nope'."),
])
async def test_a_value_that_doesnt_fit_is_refused_in_plain_words(deck: str, values: dict, words: str):
    r = await _patch(deck, 1, values)
    assert r.status_code == 400
    assert words in r.json()["detail"]
    assert _stored(1)["org"] == "Acme"


async def test_a_cell_the_agent_changed_meanwhile_is_not_overwritten(deck: str):
    _write("UPDATE ds_leads SET org = ? WHERE _id = 1", "Acme (agent)")
    r = await _patch(deck, 1, {"org": "Acme (mine)"}, expect={"org": "Acme"})
    assert r.status_code == 409
    assert "changed since you opened it" in r.json()["detail"]
    assert _stored(1)["org"] == "Acme (agent)"


async def test_an_empty_cell_seen_as_empty_still_matches(deck: str):
    r = await _patch(deck, 2, {"score": 3}, expect={"score": None})
    assert r.status_code == 200, r.text
    assert _stored(2)["score"] == 3


async def test_only_the_changed_cells_are_checked(deck: str):
    # The agent changed another cell of the row: this edit doesn't touch it.
    _write("UPDATE ds_leads SET score = 99 WHERE _id = 1")
    r = await _patch(deck, 1, {"org": "Acme 2"}, expect={"org": "Acme", "score": 7})
    assert r.status_code == 200, r.text
    assert _stored(1)["org"] == "Acme 2"
    assert _stored(1)["score"] == 99


async def test_a_deleted_row_is_gone(deck: str):
    _write("DELETE FROM ds_leads WHERE _id = 1")
    r = await _patch(deck, 1, {"org": "x"})
    assert r.status_code == 404
    assert "gone" in r.json()["detail"]


async def test_a_protected_row_refuses_a_hand_edit(deck: str):
    mgr = DatastoreManager(db_path=dl.datastore_path("bob-agent"))
    await mgr.protect("leads", "row", row_id=1, reason="signed off")
    await mgr.close()
    r = await _patch(deck, 1, {"org": "x"})
    assert r.status_code == 409
    assert "protected" in r.json()["detail"]
    assert _stored(1)["org"] == "Acme"


async def test_a_unique_value_cant_be_repeated(deck: str):
    r = await _patch(deck, 2, {"email": "a@acme.co"})
    assert r.status_code == 409
    assert "unique" in r.json()["detail"]


async def test_a_row_is_added_with_its_id_and_the_count_follows(deck: str):
    r = await _post(deck, {"org": "Initech", "score": "4", "excluded": True})
    assert r.status_code == 200, r.text
    body = r.json()
    rowid = body["row"]["__rowid"]
    assert body["row"]["org"] == "Initech"
    assert body["row"]["email"] is None
    assert _stored(rowid)["score"] == 4
    assert _stored(rowid)["excluded"] == 1
    assert body["table"]["row_count"] == 3
    assert dl.get_store().get(deck)["row_count"] == 3


async def test_a_blank_row_is_a_row_too(deck: str):
    r = await _post(deck, {})
    assert r.status_code == 200, r.text
    assert r.json()["table"]["row_count"] == 3


async def test_a_row_is_deleted_and_the_count_follows(deck: str):
    r = await _delete(deck, 2)
    assert r.status_code == 200, r.text
    assert r.json()["table"]["row_count"] == 1
    assert _stored(2) is None
    again = await _delete(deck, 2)
    assert again.status_code == 404


async def test_a_hand_edit_has_the_next_scan_summarise_again(deck: str):
    store = dl.get_store()
    store.save_result(deck, store.get(deck)["fingerprint"], status="done", summary="Old summary")
    await _patch(deck, 1, {"org": "Acme 3"})
    store.sync_agent("bob-agent", dl.scan_agent("bob-agent", store.fingerprints("bob-agent")))
    t = store.get(deck)
    assert t["status"] == "pending"
    assert t["summary"] == "Old summary"


@pytest.mark.parametrize("call", ["patch", "post", "delete"])
async def test_another_users_table_is_refused(deck: str, call: str):
    if call == "patch":
        r = await _patch(deck, 1, {"org": "x"}, uid=CAROL)
    elif call == "post":
        r = await _post(deck, {"org": "x"}, uid=CAROL)
    else:
        r = await _delete(deck, 1, uid=CAROL)
    assert r.status_code == 404
    assert _stored(1)["org"] == "Acme"
    assert dl.get_store().get(deck)["row_count"] == 2
