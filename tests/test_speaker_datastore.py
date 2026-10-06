"""PR C: a shared agent's datastore is a commons — members edit only their own.

Contract parts 0b §1.1/§3, 2 §1-§3, 2d: every table and row records its
creator (``''`` = the agent's owner, else the member's Flight Deck id) in the
same statement that writes it; a member (the bound speaker principal) reads
everything, adds rows to any table and changes only rows and tables they
created — a call that would reach someone else's row is refused as a whole.
Member limits, member ``sql`` limits, imports, the write lock, the eager
migration and the creator index are pinned here, plus the ``datastore`` tool's
member rules (argument aliases, file paths, framing of other people's rows).

Every test runs with HOME, FD_DATA_DIR, the config DB paths, the workspace and
the global datastore pointed at a tmp dir (nothing here may reach
~/.captain-claw or a real FD data dir).
"""

from __future__ import annotations

import asyncio
import contextvars
import csv
import importlib
import json
import sqlite3
import types
import zipfile
from pathlib import Path

import aiosqlite
import pytest

from captain_claw import datastore as ds
from captain_claw import saved_attribution, speaker
from captain_claw.config import get_config
from captain_claw.datastore import (
    DS_EXPRESSION_MEMBER,
    DS_FOREIGN_ROWS,
    DS_IDENTITY_LOST,
    DS_IMPORT_TOO_LARGE,
    DS_MEMBER_NO_ROOM,
    DS_MEMBER_ROW_LIMIT,
    DS_MEMBER_STORAGE_LIMIT,
    DS_MEMBER_TABLE_LIMIT,
    DS_MEMBER_TABLE_NAME,
    DS_MEMBER_TOO_MANY_COLUMNS,
    DS_MEMBER_VALUE_TOO_LARGE,
    DS_MEMBER_UNAVAILABLE,
    DS_NOT_YOUR_ROWS,
    DS_NOT_YOUR_TABLE,
    DS_OWNER_ONLY,
    DS_PROJECT_MEMBER,
    DS_SQL_LIMITS,
    DS_TABLE_NEARLY_FULL,
    DatastoreManager,
    MemberDeniedError,
    TableInfo,
)
from captain_claw.exceptions import ToolBlockedError
from captain_claw.speaker import PATH_REFUSED_PREFIX, SPEAKER_TOOL_ALLOWLIST_MAX, Principal
from captain_claw.tools import datastore as tds
from captain_claw.tools.datastore import (
    COMMONS_DATA_NOTE,
    MEMBER_CREATOR_COLUMN,
    OWNER_MEMBER_ROWS_NOTE,
    OWNER_SQL_MEMBER_NOTE,
    DatastoreTool,
)
from captain_claw.tools.registry import Tool, ToolRegistry

REF = "process:helper:0123456789abcdef"
ANA = Principal("u-ana", "Ana", "Olga", "A", REF)
BOB = Principal("u-bob", "Bob", "Olga", "A", REF)
DOCKER = Principal("u-ana", "Ana", "Olga", "A", "docker:helper:0123456789abcdef")
SLUG = "spk-session-1"
BOB_SLUG = "spk-session-bob"
GRANT = "f" * 43

_DB_FIELDS = (
    ("memory", "path"), ("session", "path"), ("insights", "db_path"),
    ("conversation_topics", "db_path"), ("nervous_system", "db_path"),
    ("sister_session", "db_path"), ("cognitive_metrics", "db_path"),
    ("datastore", "path"), ("autonomous_work", "db_path"),
)
_ENV_CLEARED = (
    "CLAW_VFS_ROOT", "CLAW_VFS_USER", "FD_OWNER_ID", "CLAW_VFS_PROJECT", "CLAW_VFS_SCOPE",
    "CLAW_WRITE_DIRECT", "FD_URL", "FD_AGENT_SHARED_SECRET", "CLAW_AGENT_LABEL",
    "CLAW_VATRA_OWNER", "CLAW_DATASTORE_VFS",
)
COLS = [{"name": "k", "type": "text"}, {"name": "v", "type": "text"}]


# ── isolation ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".captain-claw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    fd_data = tmp_path / "fd-data"
    (fd_data / "vfs").mkdir(parents=True)
    monkeypatch.setenv("FD_DATA_DIR", str(fd_data))
    for var in _ENV_CLEARED:
        monkeypatch.delenv(var, raising=False)
    cfg = get_config()
    for section, attr in _DB_FIELDS:
        monkeypatch.setattr(getattr(cfg, section), attr, str(home / ".captain-claw" / f"{section}.db"))
    ws = (tmp_path / "workspace").resolve()
    (ws / "saved").mkdir(parents=True)
    monkeypatch.setattr(cfg.workspace, "path", str(ws))
    monkeypatch.setattr(cfg.tools.read, "extra_dirs", [])
    monkeypatch.setattr(cfg.web, "public_run", False)
    from captain_claw import session as _session

    monkeypatch.setattr(_session, "_manager", _session.SessionManager(home / ".captain-claw" / "s.db"))
    import captain_claw.conversation_topics as _ct

    monkeypatch.setattr(_ct, "_MANAGER", None)
    monkeypatch.setattr(speaker, "_IN_FLIGHT", 0)
    monkeypatch.setattr(speaker, "_MAIN_LOOP", None)
    monkeypatch.setattr(speaker, "_MEMBER_GOOGLE", {})
    monkeypatch.setattr(saved_attribution, "_BACKFILLED", set())
    monkeypatch.setattr(ds, "_vfs_managers", {})
    yield home
    with saved_attribution._LOCK:
        for conn in saved_attribution._CONNS.values():
            conn.close()
        saved_attribution._CONNS.clear()


@pytest.fixture
async def dm(tmp_path, monkeypatch):
    mgr = DatastoreManager(db_path=tmp_path / "store" / "datastore.db")
    monkeypatch.setattr(ds, "_manager", mgr)
    yield mgr
    await mgr.close()


class _Bound:
    def __init__(self, p, grant=GRANT):
        self.p, self.grant = p, grant

    def __enter__(self):
        self._t1 = speaker.bind(self.p)
        self._t2 = speaker.bind_grant(self.grant)
        return self

    def __exit__(self, *exc):
        speaker.reset_grant(self._t2)
        speaker.reset(self._t1)


async def _raw(mgr: DatastoreManager, sql: str, params=()):
    async with aiosqlite.connect(str(mgr.db_path)) as db:
        async with db.execute(sql, params) as cur:
            return await cur.fetchall()


async def _creators(mgr: DatastoreManager, table: str) -> list[tuple]:
    return await _raw(mgr, f'SELECT k, "_created_by", "_created_by_name" FROM "ds_{table}" ORDER BY _id')


async def _seed_mixed(mgr: DatastoreManager, table: str = "mixed") -> None:
    """A table Ana created, with one row of hers, one of Bob's, one of the owner's."""
    with _Bound(ANA):
        await mgr.create_table(table, COLS, unique=["k"])
        await mgr.insert_rows(table, [{"k": "ana1", "v": "a"}])
    with _Bound(BOB):
        await mgr.insert_rows(table, [{"k": "bob1", "v": "b"}])
    await mgr.insert_rows(table, [{"k": "own1", "v": "o"}])


# ── stamping, migration, visibility ──────────────────────────────────


async def test_a_legacy_store_migrates_on_open_and_reads_as_the_owners(tmp_path):
    path = tmp_path / "legacy.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE _ds_tables (name TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                                 updated_at TEXT NOT NULL);
        CREATE TABLE _ds_columns (table_name TEXT NOT NULL, col_name TEXT NOT NULL,
            col_type TEXT NOT NULL DEFAULT 'text', position INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (table_name, col_name));
        CREATE TABLE ds_old (_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT);
        INSERT INTO _ds_tables VALUES ('old', 't0', 't0');
        INSERT INTO _ds_columns VALUES ('old', 'title', 'text', 0);
        INSERT INTO ds_old (title) VALUES ('before PR C');
        CREATE TABLE ds_second (_id INTEGER PRIMARY KEY AUTOINCREMENT, x TEXT);
        INSERT INTO _ds_tables VALUES ('second', 't0', 't0');
        INSERT INTO _ds_columns VALUES ('second', 'x', 'text', 0);
    """)
    con.commit()
    con.close()
    mgr = DatastoreManager(db_path=path)
    try:
        tables = await mgr.list_tables()
        assert {t.name for t in tables} == {"old", "second"}
        assert all(t.created_by == "" for t in tables)
        # Both system columns and the creator index on EVERY table, at open.
        for internal in ("ds_old", "ds_second"):
            cols = {r[1] for r in await _raw(mgr, f'PRAGMA table_info("{internal}")')}
            assert set(ds.SYSTEM_COLUMNS) <= cols
            idx = {r[1] for r in await _raw(mgr, f'PRAGMA index_list("{internal}")')}
            assert f"ix_{internal}_created_by" in idx
        assert {"created_by", "created_by_name"} <= {
            r[1] for r in await _raw(mgr, "PRAGMA table_info(_ds_tables)")}
        res = await mgr.query("old", include_creator=True)
        assert res["rows"] == [[1, "before PR C"]]
        assert res["creators"] == [{"kind": "owner", "user_id": "", "name": ""}]
        assert mgr._sys_ok == {"ds_old", "ds_second"}
    finally:
        await mgr.close()


async def test_concurrent_first_opens_make_one_connection(tmp_path, monkeypatch):
    mgr = DatastoreManager(db_path=tmp_path / "c.db")
    opened = []
    real = aiosqlite.connect

    def _counting(*a, **k):
        opened.append(a)
        return real(*a, **k)

    monkeypatch.setattr(ds.aiosqlite, "connect", _counting)
    try:
        await asyncio.gather(*(mgr._ensure_db() for _ in range(5)))
        assert len(opened) == 1 and mgr._db is not None
    finally:
        await mgr.close()


async def test_create_and_insert_stamp_the_creator_in_the_same_row(dm):
    await dm.create_table("own", COLS)
    await dm.insert_rows("own", [{"k": "o", "v": "1"}])
    with _Bound(ANA):
        info = await dm.create_table("anas", COLS)
        assert (info.created_by, info.created_by_name) == ("u-ana", "Ana")
        await dm.insert_rows("anas", [{"k": "a", "v": "1"}])
        await dm.insert_rows("own", [{"k": "a", "v": "2"}])   # the owner's table: commons
    assert await _creators(dm, "own") == [("o", "", ""), ("a", "u-ana", "Ana")]
    assert await _creators(dm, "anas") == [("a", "u-ana", "Ana")]
    meta = {t.name: (t.created_by, t.created_by_name) for t in await dm.list_tables()}
    assert meta == {"own": ("", ""), "anas": ("u-ana", "Ana")}
    # The new table has the creator columns right after _id, and its index.
    cols = [r[1] for r in await _raw(dm, 'PRAGMA table_info("ds_anas")')]
    assert cols[:3] == ["_id", "_created_by", "_created_by_name"]
    assert "ix_ds_anas_created_by" in {r[1] for r in await _raw(dm, 'PRAGMA index_list("ds_anas")')}


async def test_a_member_name_snapshot_is_one_line_and_bounded(dm):
    long = Principal("u-long", "  Ana\n\tMaria   " + "x" * 300, "Olga", "A", REF)
    with _Bound(long):
        await dm.create_table("t", COLS)
        await dm.insert_rows("t", [{"k": "a"}])
    (_k, cb, name), = await _creators(dm, "t")
    assert cb == "u-long" and name.startswith("Ana Maria x") and len(name) == ds.MEMBER_NAME_MAX


async def test_system_columns_are_invisible_by_default(dm, tmp_path):
    with _Bound(ANA):
        await dm.create_table("t", COLS)
        await dm.insert_rows("t", [{"k": "a", "v": "1"}])
    info = await dm.describe_table("t")
    assert [c.name for c in info.columns] == ["k", "v"]
    assert [c.name for c in (await dm.list_tables())[0].columns] == ["k", "v"]
    res = await dm.query("t")
    assert res["columns"] == ["_id", "k", "v"] and res["rows"] == [[1, "a", "1"]]
    assert "creators" not in res
    out = tmp_path / "e.csv"
    await dm.export_csv("t", out)
    assert out.read_text().splitlines()[0] == "_id,k,v"
    await dm.export_json("t", tmp_path / "e.json")
    assert set(json.loads((tmp_path / "e.json").read_text())[0]) == {"_id", "k", "v"}
    # Named explicitly, they can be read (J18).
    res = await dm.query("t", columns=["k", "_created_by", "_created_by_name"])
    assert res["rows"] == [["a", "u-ana", "Ana"]]


async def test_query_include_creator_is_aligned_and_keeps_named_columns(dm):
    await _seed_mixed(dm)
    res = await dm.query("mixed", order_by=["_id"], include_creator=True)
    assert [r[1] for r in res["rows"]] == ["ana1", "bob1", "own1"]
    assert [c["kind"] for c in res["creators"]] == ["member", "member", "owner"]
    assert [c["user_id"] for c in res["creators"]] == ["u-ana", "u-bob", ""]
    assert res["columns"] == ["_id", "k", "v"]
    res = await dm.query("mixed", columns=["k", "_created_by"], order_by=["-_created_by"],
                         include_creator=True)
    assert res["columns"] == ["k", "_created_by"]
    assert res["rows"][0] == ["bob1", "u-bob"]
    assert res["creators"][0]["user_id"] == "u-bob"



async def test_a_table_another_process_created_reads_with_its_creators(tmp_path):
    # This manager opened the store first; a second one (another process: a
    # script, a worker) then created a table and a member added a row. Reads
    # here must show that member, not the owner, before this manager ever
    # writes to the table.
    path = tmp_path / "shared.db"
    here, other = DatastoreManager(db_path=path), DatastoreManager(db_path=path)
    try:
        await here.list_tables()
        await other.create_table("leads", COLS)
        await other.insert_rows("leads", [{"k": "own1", "v": "o"}])
        with _Bound(ANA):
            await other.insert_rows("leads", [{"k": "ana1", "v": "a"}])
        res = await here.query("leads", order_by=["_id"], include_creator=True)
        assert [c["kind"] for c in res["creators"]] == ["owner", "member"]
        assert res["creators"][1]["user_id"] == "u-ana"
        assert await here.has_member_rows() is True
        with _Bound(ANA):
            assert await here._member_row_total(ds.current_actor()) == 1
    finally:
        await here.close()
        await other.close()

# ── changing other people's rows ─────────────────────────────────────


@pytest.mark.parametrize("op", ["update_one", "update_all", "delete_one", "delete_all",
                                "upsert", "update_none"])
async def test_a_member_call_reaching_a_foreign_row_is_refused_whole(dm, op):
    await _seed_mixed(dm)
    before = await _creators(dm, "mixed")
    before_rows = await _raw(dm, 'SELECT * FROM "ds_mixed" ORDER BY _id')
    with _Bound(ANA), pytest.raises(MemberDeniedError) as info:
        if op == "update_one":
            await dm.update_rows("mixed", {"v": "X"}, {"k": ["ana1", "bob1"]})
        elif op == "update_all":
            await dm.update_rows("mixed", {"v": "X"}, {"_all": True})
        elif op == "update_none":
            await dm.update_rows("mixed", {"v": "X"}, None)
        elif op == "delete_one":
            await dm.delete_rows("mixed", {"k": "own1"})
        elif op == "delete_all":
            await dm.delete_rows("mixed", {"_all": True})
        else:
            await dm.upsert_rows("mixed", [{"k": "ana1", "v": "A2"}, {"k": "bob1", "v": "X"}])
    assert str(info.value) == DS_NOT_YOUR_ROWS
    assert await _creators(dm, "mixed") == before
    assert await _raw(dm, 'SELECT * FROM "ds_mixed" ORDER BY _id') == before_rows


async def test_mine_narrows_for_member_and_owner(dm):
    await _seed_mixed(dm)
    with _Bound(ANA):
        assert await dm.update_rows("mixed", {"v": "A!"}, {"_mine": True}) == 1
        res = await dm.query("mixed", where={"_mine": True})
        assert [r[1] for r in res["rows"]] == ["ana1"]
    assert await dm.update_rows("mixed", {"v": "O!"}, {"_mine": True}) == 1
    rows = {k: v for _i, k, v in [(r[0], r[1], r[2]) for r in await _raw(
        dm, 'SELECT _id, k, v FROM "ds_mixed"')]}
    assert rows == {"ana1": "A!", "bob1": "b", "own1": "O!"}
    with pytest.raises(ValueError, match='"_mine" takes true'):
        await dm.query("mixed", where={"_mine": "yes"})


async def test_update_all_on_own_rows_only_and_delete_all_mine(dm):
    with _Bound(ANA):
        await dm.create_table("solo", COLS)
        await dm.insert_rows("solo", [{"k": "a"}, {"k": "b"}])
        assert await dm.update_rows("solo", {"v": "z"}, {"_all": True}) == 2
    await _seed_mixed(dm)
    with _Bound(ANA):
        assert await dm.delete_rows("mixed", {"_all": True, "_mine": True}) == 1
    assert [r[0] for r in await _creators(dm, "mixed")] == ["bob1", "own1"]
    # The owner: '' rows only.
    assert await dm.delete_rows("mixed", {"_all": True, "_mine": True}) == 1
    assert [r[0] for r in await _creators(dm, "mixed")] == ["bob1"]


async def test_member_upsert_on_own_row_updates_and_keeps_the_creator(dm):
    await _seed_mixed(dm)
    with _Bound(ANA):
        assert await dm.upsert_rows("mixed", [{"k": "ana1", "v": "new"}, {"k": "ana2", "v": "n"}]) == 2
    rows = await _raw(dm, 'SELECT k, v, "_created_by" FROM "ds_mixed" ORDER BY _id')
    assert ("ana1", "new", "u-ana") in rows and ("ana2", "n", "u-ana") in rows


async def test_owner_upsert_on_a_member_row_keeps_the_member_creator(dm):
    await _seed_mixed(dm)
    assert await dm.upsert_rows("mixed", [{"k": "bob1", "v": "owner edit"}]) == 1
    rows = await _raw(dm, 'SELECT k, v, "_created_by", "_created_by_name" FROM "ds_mixed"')
    assert ("bob1", "owner edit", "u-bob", "Bob") in rows


async def test_member_input_cannot_name_the_system_columns(dm):
    with _Bound(ANA):
        await dm.create_table("t", COLS)
        await dm.insert_rows("t", [{"k": "a", "_created_by": "", "_created_by_name": "Olga"}])
        with pytest.raises(ValueError):
            await dm.update_rows("t", {"_created_by": ""}, {"_mine": True})
    assert await _creators(dm, "t") == [("a", "u-ana", "Ana")]


# ── table structure ──────────────────────────────────────────────────


async def test_update_column_rules(dm):
    with _Bound(ANA):
        await dm.create_table("mine", COLS)
        await dm.insert_rows("mine", [{"k": "a"}, {"k": "b"}])
        with pytest.raises(MemberDeniedError) as info:
            await dm.update_column("mine", "v", expression="k || 'x'")
        assert str(info.value) == DS_EXPRESSION_MEMBER
        assert await dm.update_column("mine", "v", value="same") == 2
    await _seed_mixed(dm)
    with _Bound(ANA), pytest.raises(MemberDeniedError) as info:
        await dm.update_column("mixed", "v", value="x")
    assert str(info.value) == DS_FOREIGN_ROWS
    await dm.create_table("owners", COLS)
    with _Bound(ANA), pytest.raises(MemberDeniedError) as info:
        await dm.update_column("owners", "v", value="x")
    assert str(info.value) == DS_NOT_YOUR_TABLE


async def test_rename_and_add_column_on_own_table_but_not_the_owners(dm):
    await dm.create_table("owners", COLS)
    with _Bound(ANA):
        await dm.create_table("mine", COLS)
        await dm.add_column("mine", "extra")
        await dm.rename_column("mine", "extra", "more")
        info = await dm.rename_table("mine", "mine2")
        assert info.name == "mine2" and info.created_by == "u-ana"
        for call in (lambda: dm.add_column("owners", "x"),
                     lambda: dm.rename_table("owners", "taken"),
                     lambda: dm.rename_column("owners", "v", "w"),
                     lambda: dm.drop_column("owners", "v"),
                     lambda: dm.change_column_type("owners", "v", "integer"),
                     lambda: dm.drop_table("owners")):
            with pytest.raises(MemberDeniedError) as info:
                await call()
            assert str(info.value) == DS_NOT_YOUR_TABLE
    assert {t.name for t in await dm.list_tables()} == {"owners", "mine2"}


@pytest.mark.parametrize("op", ["drop_table", "rename_table", "rename_column", "drop_column",
                                "change_column_type"])
async def test_own_table_with_a_foreign_row_is_owner_only(dm, op):
    with _Bound(ANA):
        await dm.create_table("mine", COLS + [{"name": "w", "type": "text"}])
        await dm.insert_rows("mine", [{"k": "a"}])
    with _Bound(BOB):
        await dm.insert_rows("mine", [{"k": "b"}])
    with _Bound(ANA), pytest.raises(MemberDeniedError) as info:
        if op == "drop_table":
            await dm.drop_table("mine")
        elif op == "rename_table":
            await dm.rename_table("mine", "renamed")
        elif op == "rename_column":
            await dm.rename_column("mine", "v", "vv")
        elif op == "drop_column":
            await dm.drop_column("mine", "w")
        else:
            await dm.change_column_type("mine", "v", "integer")
    assert str(info.value) == DS_FOREIGN_ROWS
    assert [t.name for t in await dm.list_tables()] == ["mine"]
    assert [c.name for c in (await dm.describe_table("mine")).columns] == ["k", "v", "w"]
    # The owner can.
    await dm.drop_table("mine")


@pytest.mark.parametrize("call", ["protect", "unprotect"])
async def test_protect_and_unprotect_are_owner_only(dm, call):
    with _Bound(ANA):
        await dm.create_table("mine", COLS)
        with pytest.raises(MemberDeniedError) as info:
            if call == "protect":
                await dm.protect("mine", "table")
            else:
                await dm.unprotect("mine", "table")
    assert str(info.value) == DS_OWNER_ONLY
    assert await dm.list_protections() == []


async def test_table_protection_also_stops_members_adding_rows(dm):
    await dm.create_table("locked", COLS)
    await dm.protect("locked", "table", reason="frozen")
    with _Bound(ANA), pytest.raises(ds.ProtectedError):
        await dm.insert_rows("locked", [{"k": "a"}])


async def test_change_column_type_keeps_creators_and_the_unique_key(dm):
    await _seed_mixed(dm)
    await dm.change_column_type("mixed", "v", "text")
    assert await _creators(dm, "mixed") == [("ana1", "u-ana", "Ana"), ("bob1", "u-bob", "Bob"),
                                            ("own1", "", "")]
    # The UNIQUE key still drives upsert (it used to be dropped by the rebuild).
    assert await dm._unique_columns("ds_mixed") == ["k"]
    await dm.upsert_rows("mixed", [{"k": "own1", "v": "again"}])
    assert len(await _creators(dm, "mixed")) == 3
    assert "ix_ds_mixed_created_by" in {r[1] for r in await _raw(dm, 'PRAGMA index_list("ds_mixed")')}


# ── limits ───────────────────────────────────────────────────────────


async def test_a_member_creates_at_most_ten_tables(dm):
    with _Bound(ANA):
        for i in range(ds.MEMBER_MAX_TABLES):
            await dm.create_table(f"t{i}", COLS)
        with pytest.raises(MemberDeniedError) as info:
            await dm.create_table("eleventh", COLS)
    assert str(info.value) == DS_MEMBER_TABLE_LIMIT
    with _Bound(BOB):
        await dm.create_table("bobs", COLS)       # per member


async def test_the_owner_keeps_ten_table_slots(dm, monkeypatch):
    monkeypatch.setattr(get_config().datastore, "max_tables", 12)
    await dm.create_table("o1", COLS)
    with _Bound(ANA):
        await dm.create_table("a1", COLS)
        with pytest.raises(MemberDeniedError) as info:
            await dm.create_table("a2", COLS)
    assert str(info.value) == DS_MEMBER_NO_ROOM
    for i in range(2, 12):
        await dm.create_table(f"o{i}", COLS)     # the owner fills the rest
    with pytest.raises(ValueError, match="Table limit"):
        await dm.create_table("o13", COLS)


async def test_a_member_adds_at_most_ten_thousand_rows(dm, monkeypatch):
    monkeypatch.setattr(get_config().datastore, "max_rows_per_table", 100_000)
    await dm.create_table("big", COLS)
    with _Bound(ANA):
        await dm.insert_rows("big", [{"k": str(i)} for i in range(ds.MEMBER_MAX_ROWS)])
        with pytest.raises(MemberDeniedError) as info:
            await dm.insert_rows("big", [{"k": "one more"}])
    assert str(info.value) == DS_MEMBER_ROW_LIMIT
    assert (await dm.describe_table("big")).row_count == ds.MEMBER_MAX_ROWS


async def test_the_owner_keeps_ten_percent_of_every_table(dm, monkeypatch):
    monkeypatch.setattr(get_config().datastore, "max_rows_per_table", 100)
    await dm.create_table("t", COLS)
    await dm.insert_rows("t", [{"k": str(i)} for i in range(50)])
    with _Bound(ANA):
        await dm.insert_rows("t", [{"k": f"a{i}"} for i in range(40)])     # 90
        with pytest.raises(MemberDeniedError) as info:
            await dm.insert_rows("t", [{"k": "a91"}])
        assert str(info.value) == DS_TABLE_NEARLY_FULL
        with pytest.raises(MemberDeniedError):
            await dm.upsert_rows("t", [{"k": "a91"}], key_columns=["k"])
    await dm.insert_rows("t", [{"k": "o91"}])                                 # the owner: OK
    assert (await dm.describe_table("t")).row_count == 91


# ── who is asking ────────────────────────────────────────────────────


async def test_lost_identity_refuses_every_write(dm, monkeypatch):
    await dm.create_table("t", COLS)
    monkeypatch.setattr(speaker, "identity_lost", lambda: True)
    for call in (lambda: dm.insert_rows("t", [{"k": "x"}]),
                 lambda: dm.create_table("u", COLS),
                 lambda: dm.delete_rows("t", {"_all": True}),
                 lambda: dm.update_rows("t", {"v": "x"}, None)):
        with pytest.raises(MemberDeniedError) as info:
            await call()
        assert str(info.value) == DS_IDENTITY_LOST
    monkeypatch.undo()
    assert (await dm.describe_table("t")).row_count == 0
    assert [t.name for t in await dm.list_tables()] == ["t"]


@pytest.mark.parametrize("p,why", [
    (DOCKER, DS_MEMBER_UNAVAILABLE),
    (Principal("u-x", "X", "O", "A", "not-a-ref"), DS_MEMBER_UNAVAILABLE),
    (speaker.UNKNOWN_PRINCIPAL, DS_IDENTITY_LOST),
])
async def test_unverified_or_docker_members_cannot_write(dm, p, why):
    await dm.create_table("t", COLS)
    with _Bound(p), pytest.raises(MemberDeniedError) as info:
        await dm.insert_rows("t", [{"k": "x"}])
    assert str(info.value) == why
    assert (await dm.describe_table("t")).row_count == 0


def test_folder_stores_refuse_members(monkeypatch):
    with _Bound(ANA), pytest.raises(PermissionError) as info:
        ds.get_vfs_datastore_manager("some-run")
    assert str(info.value) == DS_PROJECT_MEMBER
    assert ds._vfs_managers == {}


async def test_a_member_always_resolves_to_the_global_store(dm, monkeypatch):
    monkeypatch.setenv("CLAW_DATASTORE_VFS", "vatra-run")
    with _Bound(ANA):
        assert ds.resolve_datastore_manager("s") is dm


# ── concurrency (J19) ────────────────────────────────────────────────


async def test_a_drop_racing_another_members_insert_never_loses_their_row(dm, monkeypatch):
    with _Bound(ANA):
        await dm.create_table("mine", COLS)
    real = DatastoreManager._require_no_foreign_rows

    async def _slow(self, internal, actor):
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return await real(self, internal, actor)

    monkeypatch.setattr(DatastoreManager, "_require_no_foreign_rows", _slow)

    async def _drop():
        with _Bound(ANA):
            return await dm.drop_table("mine")

    async def _insert():
        await asyncio.sleep(0)
        with _Bound(BOB):
            return await dm.insert_rows("mine", [{"k": "bob"}])

    dropped, inserted = await asyncio.gather(_drop(), _insert(), return_exceptions=True)
    if dropped is True:
        # The drop ran first and whole; Bob's insert then found no table.
        assert isinstance(inserted, ValueError) and "Table not found" in str(inserted)
    else:
        assert isinstance(dropped, MemberDeniedError) and inserted == 1
        assert await _creators(dm, "mine") == [("bob", "u-bob", "Bob")]


async def test_two_inserts_at_the_row_cap_exactly_one_succeeds(dm, monkeypatch):
    monkeypatch.setattr(ds, "MEMBER_MAX_ROWS", 5)
    await dm.create_table("t", COLS)
    with _Bound(ANA):
        await dm.insert_rows("t", [{"k": str(i)} for i in range(4)])
    real = DatastoreManager._member_row_total

    async def _slow(self, actor):
        n = await real(self, actor)
        await asyncio.sleep(0)
        return n

    monkeypatch.setattr(DatastoreManager, "_member_row_total", _slow)

    async def _one(tag):
        with _Bound(ANA):
            return await dm.insert_rows("t", [{"k": tag}])

    results = await asyncio.gather(_one("x"), _one("y"), return_exceptions=True)
    assert sorted(type(r).__name__ for r in results) == ["MemberDeniedError", "int"]
    assert (await dm.describe_table("t")).row_count == 5


async def test_the_write_lock_is_reentrant_within_a_task(dm):
    async with dm._writing():
        await dm.create_table("inside", COLS)       # would deadlock without re-entry
    assert [t.name for t in await dm.list_tables()] == ["inside"]


async def test_a_failed_write_never_rides_along_with_the_next_commit(dm):
    await dm.create_table("t", COLS)
    with _Bound(ANA), pytest.raises(ValueError):
        await dm.insert_rows("t", [{"k": "first"}, "not a row"])
    await dm.insert_rows("t", [{"k": "owner"}])
    assert await _creators(dm, "t") == [("owner", "", "")]


# ── migration fallback and the creator index ─────────────────────────


async def test_a_table_added_by_raw_sql_after_open_is_migrated_under_the_lock(dm):
    await dm.create_table("seed", COLS)
    async with aiosqlite.connect(str(dm.db_path)) as db:
        await db.execute('CREATE TABLE "ds_late" (_id INTEGER PRIMARY KEY AUTOINCREMENT, k TEXT)')
        await db.execute("INSERT INTO _ds_tables (name, created_at, updated_at) VALUES ('late', 't', 't')")
        await db.execute("INSERT INTO _ds_columns VALUES ('late', 'k', 'text', 0)")
        await db.execute("INSERT INTO ds_late (k) VALUES ('old')")
        await db.commit()
    res = await dm.query("late", include_creator=True)      # a read never ALTERs
    assert res["creators"] == [{"kind": "owner", "user_id": "", "name": ""}]
    assert "ds_late" not in dm._sys_ok
    with _Bound(ANA):
        await dm.insert_rows("late", [{"k": "ana"}])
    assert "ds_late" in dm._sys_ok
    assert await _raw(dm, 'SELECT k, "_created_by" FROM "ds_late" ORDER BY _id') == [
        ("old", ""), ("ana", "u-ana")]


async def test_the_member_row_count_uses_the_creator_index(dm):
    await dm.create_table("t", COLS)
    plan = await _raw(dm, 'EXPLAIN QUERY PLAN SELECT COUNT(*) FROM "ds_t" WHERE "_created_by" = ?',
                      ("u-ana",))
    assert any("ix_ds_t_created_by" in str(r) for r in plan), plan


async def test_the_index_follows_rename_and_a_new_table_under_the_old_name_gets_its_own(dm):
    await dm.create_table("t", COLS)
    await dm.rename_table("t", "t2")
    assert "ix_ds_t2_created_by" in {r[1] for r in await _raw(dm, 'PRAGMA index_list("ds_t2")')}
    await dm.create_table("t", COLS)
    assert "ix_ds_t_created_by" in {r[1] for r in await _raw(dm, 'PRAGMA index_list("ds_t")')}
    await dm.change_column_type("t2", "v", "integer")
    assert "ix_ds_t2_created_by" in {r[1] for r in await _raw(dm, 'PRAGMA index_list("ds_t2")')}


# ── member sql (J19) ─────────────────────────────────────────────────


async def test_member_sql_refuses_recursive_ctes(dm):
    await dm.create_table("t", COLS)
    with _Bound(ANA), pytest.raises(ValueError) as info:
        await dm.raw_select("SELECT * FROM (WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL "
                            "SELECT x + 1 FROM c) SELECT x FROM c)")
    assert str(info.value) == DS_SQL_LIMITS


async def test_member_sql_is_interrupted_at_the_deadline(dm, monkeypatch):
    await dm.create_table("t", COLS)
    await dm.insert_rows("t", [{"k": str(i)} for i in range(400)])
    monkeypatch.setattr(ds, "MEMBER_SQL_DEADLINE_S", 0.05)
    with _Bound(ANA), pytest.raises(ValueError) as info:
        await dm.raw_select("SELECT COUNT(*) FROM t a, t b, t c")
    assert str(info.value) == DS_SQL_LIMITS
    assert await dm.insert_rows("t", [{"k": "after"}]) == 1      # the store still writes
    with _Bound(ANA):
        res = await dm.raw_select("SELECT COUNT(*) AS n FROM t")
    assert res["rows"] == [[401]]


async def test_member_sql_rows_are_capped_even_with_a_subquery_limit(dm, monkeypatch):
    monkeypatch.setattr(get_config().datastore, "max_query_rows", 7)
    await dm.create_table("t", COLS)
    await dm.insert_rows("t", [{"k": str(i)} for i in range(20)])
    with _Bound(ANA):
        res = await dm.raw_select("SELECT a.k FROM t a, (SELECT k FROM t LIMIT 20) b")
    assert len(res["rows"]) == 7


async def test_member_sql_never_reads_pragma_functions(dm):
    # pragma_database_list names the store's host path (G-C4).
    await dm.create_table("t", COLS)
    with _Bound(ANA), pytest.raises(ValueError) as info:
        await dm.raw_select("SELECT file FROM pragma_database_list")
    assert str(info.value) == DS_SQL_LIMITS
    res = await dm.raw_select("SELECT name FROM pragma_database_list")    # the owner: as before
    assert res["rows"] == [["main"]]


async def test_member_sql_values_and_results_are_size_capped(dm, monkeypatch):
    await dm.create_table("t", COLS)
    await dm.insert_rows("t", [{"k": str(i)} for i in range(40)])
    with _Bound(ANA), pytest.raises(ValueError) as info:
        await dm.raw_select("SELECT zeroblob(50000000) FROM t")          # one huge value
    assert str(info.value) == DS_SQL_LIMITS
    monkeypatch.setattr(ds, "_MEMBER_SQL_RESULT_MAX", 1000)
    with _Bound(ANA), pytest.raises(ValueError) as info:
        await dm.raw_select("SELECT zeroblob(100) FROM t")               # 4000 bytes in all
    assert str(info.value) == DS_SQL_LIMITS
    with _Bound(ANA):
        res = await dm.raw_select("SELECT k FROM t")
    assert len(res["rows"]) == 40
    res = await dm.raw_select("SELECT zeroblob(100) FROM t")              # the owner: as before
    assert len(res["rows"]) == 40


async def test_member_sql_runs_on_a_read_only_connection(dm):
    await dm.create_table("t", COLS)
    with _Bound(ANA):
        await dm.raw_select("SELECT * FROM t")
    assert dm._ro_db is not None
    with pytest.raises(sqlite3.OperationalError):
        await dm._ro_db.execute("INSERT INTO ds_t (k) VALUES ('x')")


async def test_select_star_hides_the_creator_columns_unless_named(dm):
    await _seed_mixed(dm)
    for p in (None, ANA):
        with _Bound(p):
            res = await dm.raw_select("SELECT * FROM mixed")
            assert res["columns"] == ["_id", "k", "v"], p
            res = await dm.raw_select("SELECT k, _created_by, _created_by_name FROM mixed")
            assert res["columns"] == ["k", "_created_by", "_created_by_name"]
            assert ["bob1", "u-bob", "Bob"] in res["rows"]


# ── member imports (J19) ─────────────────────────────────────────────


async def test_a_too_large_csv_is_refused_before_anything(dm, tmp_path):
    big = tmp_path / "big.csv"
    with open(big, "wb") as f:
        f.write(b"k,v\n")
        f.truncate(26 * 1024 * 1024)
    with _Bound(ANA), pytest.raises(MemberDeniedError) as info:
        await dm.import_csv(big, "big")
    assert str(info.value) == DS_IMPORT_TOO_LARGE
    assert await dm.list_tables() == []


async def test_an_xlsx_over_the_unzipped_cap_is_refused_before_parsing(dm, tmp_path, monkeypatch):
    path = tmp_path / "wide.xlsx"
    DatastoreManager._write_xlsx(path, ["k"], [["x" * 50] for _ in range(50)])
    monkeypatch.setattr(ds, "MEMBER_IMPORT_MAX_UNZIPPED_BYTES", 200)
    parsed = []
    monkeypatch.setattr(DatastoreManager, "_parse_xlsx",
                        staticmethod(lambda *a, **k: parsed.append(a) or (["k"], [])))
    with _Bound(ANA), pytest.raises(MemberDeniedError) as info:
        await dm.import_xlsx(path, "wide")
    assert str(info.value) == DS_IMPORT_TOO_LARGE
    assert parsed == [] and await dm.list_tables() == []


async def test_a_csv_over_the_row_budget_creates_nothing(dm, tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "MEMBER_MAX_ROWS", 5)
    path = tmp_path / "rows.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "v"])
        for i in range(6):
            w.writerow([i, i])
    with _Bound(ANA), pytest.raises(MemberDeniedError) as info:
        await dm.import_csv(path, "imported")
    assert str(info.value) == DS_MEMBER_ROW_LIMIT
    assert await dm.list_tables() == []
    path.write_text("k,v\n1,1\n2,2\n")
    with _Bound(ANA):
        res = await dm.import_csv(path, "imported")
    assert res["rows_imported"] == 2
    assert [r[1] for r in await _creators(dm, "imported")] == ["u-ana", "u-ana"]
    assert (await dm.describe_table("imported")).created_by == "u-ana"


# ── exports neutralise formulas (J13) ────────────────────────────────


async def test_export_neutralize_modes(dm, tmp_path):
    await dm.create_table("t", COLS)
    await dm.insert_rows("t", [{"k": "=SUM(A1)", "v": "plain"}])
    with _Bound(ANA):
        await dm.insert_rows("t", [{"k": "=HYPERLINK(\"x\")", "v": "+1"}])
    for mode, expected in (("none", ["=SUM(A1)", "=HYPERLINK(\"x\")"]),
                           ("members", ["=SUM(A1)", "'=HYPERLINK(\"x\")"]),
                           ("all", ["'=SUM(A1)", "'=HYPERLINK(\"x\")"])):
        out = tmp_path / f"{mode}.csv"
        await dm.export_csv("t", out, neutralize=mode)
        rows = list(csv.reader(out.open()))[1:]
        assert [r[1] for r in rows] == expected, mode
    assert ds.neutralize_rows([["-1", 5, "@x", "\tx", "\rx", "ok", ""]], None, "all") == [
        ["'-1", 5, "'@x", "'\tx", "'\rx", "ok", ""]]


# ── the datastore tool for members (registry level) ──────────────────


class _Probe(Tool):
    def __init__(self, name):
        self.name = name
        self.description = name
        self.parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        from captain_claw.tools.registry import ToolResult

        return ToolResult(success=True, content="ok")


def _member_agent(p=ANA, session_id=SLUG):
    return types.SimpleNamespace(
        _speaker_scoped=True, _speaker_principal=p, _turn_grant=GRANT,
        session=types.SimpleNamespace(id=session_id),
        _current_session_slug=lambda: session_id,
    )


@pytest.fixture
def world(tmp_path, dm, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_USER", "owner")
    monkeypatch.setenv("FD_OWNER_ID", "owner")
    ws = Path(get_config().workspace.path)
    saved = ws / "saved"
    vfs_base = (tmp_path / "fd-data" / "vfs").resolve()
    (vfs_base / "u-ana" / "p").mkdir(parents=True)
    (vfs_base / "u-ana" / "p" / "mine.csv").write_text("k,v\nvfs,1\n")
    (vfs_base / "owner" / "proj").mkdir(parents=True)
    (vfs_base / "owner" / "proj" / "secret.csv").write_text("k,v\nsecret,1\n")
    (ws / "owner.csv").write_text("k,v\nworkspace,1\n")
    reg = ToolRegistry(base_path=ws)
    reg.register(DatastoreTool())
    return types.SimpleNamespace(reg=reg, ws=ws, saved=saved, vfs_base=vfs_base,
                                 tmp=tmp_path.resolve(), dm=dm)


async def call(w, args, *, p=ANA, session_id=SLUG):
    agent = _member_agent(p, session_id)
    with _Bound(p):
        return await w.reg.execute("datastore", {**args, "_agent": agent}, session_id=session_id,
                                   runtime_base_path=w.ws)


async def refused(w, args, **kw) -> str:
    with pytest.raises(ToolBlockedError) as info:
        await call(w, args, **kw)
    return info.value.reason


async def owner_call(w, args):
    return await w.reg.execute("datastore", args, session_id="owner", runtime_base_path=w.ws)


def _no_host_paths(text: str, w) -> None:
    for marker in (str(w.tmp), str(w.ws)):
        assert marker not in text, (marker, text)
    # An absolute host path (saved/tmp/… is fine: it isn't one).
    import re

    assert not re.search(r"(?<![\w.])/(tmp|private|Users|var)\b", text), text


async def test_a_process_member_may_use_the_datastore_a_docker_member_not(world):
    res = await call(world, {"action": "create_table", "table": "t", "columns": json.dumps(COLS)})
    assert res.success, res.error
    reason = await refused(world, {"action": "list_tables"}, p=DOCKER)
    assert reason == speaker.NOT_ALLOWED_MESSAGE


@pytest.mark.parametrize("args,why", [
    ({"action": "query", "table": "t", "project": "other-run"}, DS_PROJECT_MEMBER),
    ({"action": "protect", "table": "t", "level": "table"}, DS_OWNER_ONLY),
    ({"action": "unprotect", "table": "t", "level": "table"}, DS_OWNER_ONLY),
    ({"action": "update_column", "table": "t", "column": "v", "expression": "v || 'x'"},
     DS_EXPRESSION_MEMBER),
    ({"action": "drop_everything"}, speaker.NOT_ALLOWED_MESSAGE),
])
async def test_apply_tool_rules_refuses(world, args, why):
    assert await refused(world, args) == why


async def test_the_member_ignores_the_runs_folder_store(world, monkeypatch):
    monkeypatch.setenv("CLAW_DATASTORE_VFS", "vatra-run")
    res = await call(world, {"action": "create_table", "table": "t", "columns": json.dumps(COLS)})
    assert res.success, res.error
    assert [t.name for t in await world.dm.list_tables()] == ["t"]
    assert ds._vfs_managers == {}


async def test_member_query_shows_who_added_each_row(world):
    await _seed_mixed(world.dm)
    res = await call(world, {"action": "query", "table": "mixed", "order_by": "_id"})
    assert res.success
    assert res.content.startswith(COMMONS_DATA_NOTE + "\n")
    lines = res.content.splitlines()
    assert MEMBER_CREATOR_COLUMN in lines[1]
    body = "\n".join(lines[3:6])
    assert "ana1" in lines[3] and "you" in lines[3]
    assert "bob1" in lines[4] and "Bob" in lines[4]
    assert "own1" in lines[5] and "owner" in lines[5]
    assert "u-bob" not in body
    # Only her own rows → no note.
    res = await call(world, {"action": "query", "table": "mixed", "where": '{"_mine": true}'})
    assert not res.content.startswith(COMMONS_DATA_NOTE)


async def test_owner_query_marks_member_rows_and_is_unchanged_without(world):
    await world.dm.create_table("plain", COLS)
    await world.dm.insert_rows("plain", [{"k": "a", "v": "1"}])
    res = await owner_call(world, {"action": "query", "table": "plain"})
    expected = tds._format_table(["_id", "k", "v"], [[1, "a", "1"]], 1)
    assert res.content == expected          # byte-identical without member rows
    await _seed_mixed(world.dm)
    res = await owner_call(world, {"action": "query", "table": "mixed"})
    assert res.content.startswith(OWNER_MEMBER_ROWS_NOTE.format(n=2) + "\n")
    assert MEMBER_CREATOR_COLUMN not in res.content


async def test_a_member_refusal_reads_blocked(world):
    await _seed_mixed(world.dm)
    res = await call(world, {"action": "delete", "table": "mixed", "where": '{"_all": true}'})
    assert not res.success
    assert res.error == f"BLOCKED: {DS_NOT_YOUR_ROWS} The operation was NOT performed."


async def test_member_sql_results_carry_the_commons_note(world):
    await _seed_mixed(world.dm)
    res = await call(world, {"action": "sql", "sql_query": "SELECT k FROM mixed"})
    assert res.success and res.content.startswith(COMMONS_DATA_NOTE + "\n")


# ── ds_file: import / export paths ───────────────────────────────────


def _stamp_as(p: Principal | None, path: Path) -> None:
    ctx = contextvars.copy_context()
    if p is not None:
        ctx.run(speaker.bind, p)
    ctx.run(saved_attribution.note_write, path, None)


async def test_import_from_another_members_saved_file(world):
    src = world.saved / "downloads" / BOB_SLUG / "data.csv"
    src.parent.mkdir(parents=True)
    src.write_text("k,v\nfrom-bob,1\n")
    _stamp_as(BOB, src)
    res = await call(world, {"action": "import_file", "file_path": f"saved/downloads/{BOB_SLUG}/data.csv",
                             "table": "imported"})
    assert res.success, res.error
    assert await _creators(world.dm, "imported") == [("from-bob", "u-ana", "Ana")]
    assert (await world.dm.describe_table("imported")).created_by == "u-ana"


async def test_export_lands_in_the_members_folder_stamped_and_shown_relative(world):
    await _seed_mixed(world.dm)
    res = await call(world, {"action": "export", "table": "mixed", "file_path": "out.csv"})
    assert res.success, res.error
    target = world.saved / "tmp" / SLUG / "out.csv"
    assert target.is_file()
    assert f"saved/tmp/{SLUG}/out.csv" in res.content
    _no_host_paths(res.content, world)
    c = saved_attribution.creator_of(target)
    assert (c.kind, c.user_id, c.source) == ("member", "u-ana", "stamp")
    # Without file_path: this conversation's output folder.
    res = await call(world, {"action": "export", "table": "mixed", "format": "json"})
    assert res.success and f"saved/output/{SLUG}/mixed.json" in res.content
    _no_host_paths(res.content, world)


async def test_export_paths_that_are_refused(world):
    await _seed_mixed(world.dm)
    reason = await refused(world, {"action": "export", "table": "mixed", "file_path": "vfs:p/x.csv"})
    assert "export into this conversation's saved/ folder" in reason
    reason = await refused(world, {"action": "query", "table": "mixed", "file_path": "x.csv"})
    assert reason == PATH_REFUSED_PREFIX + "file_path is only used by import_file and export"
    owned = world.saved / "tmp" / SLUG / "owned.csv"
    owned.parent.mkdir(parents=True, exist_ok=True)
    owned.write_text("owner's\n")
    _stamp_as(None, owned)
    reason = await refused(world, {"action": "export", "table": "mixed", "file_path": "owned.csv"})
    assert reason == PATH_REFUSED_PREFIX + speaker.FILE_NOT_YOURS_WHY
    assert owned.read_text() == "owner's\n"


@pytest.mark.parametrize("alias", tds._ALIASES["file_path"])
@pytest.mark.parametrize("source", ["workspace", "owner_vfs"])
async def test_every_file_path_alias_is_confined_for_import(world, alias, source):
    path = (str(world.ws / "owner.csv") if source == "workspace"
            else str(world.vfs_base / "owner" / "proj" / "secret.csv"))
    with pytest.raises(ToolBlockedError) as info:
        await call(world, {"action": "import_file", alias: path, "table": "stolen"})
    assert info.value.reason.startswith(PATH_REFUSED_PREFIX)
    _no_host_paths(info.value.reason, world)
    assert await world.dm.list_tables() == []


@pytest.mark.parametrize("alias", tds._ALIASES["file_path"])
async def test_every_file_path_alias_is_confined_for_export(world, alias):
    await _seed_mixed(world.dm)
    outside = world.tmp / "elsewhere" / "out.json"
    try:
        res = await call(world, {"action": "export", "table": "mixed", "format": "json",
                                 alias: str(outside)})
    except ToolBlockedError as exc:
        assert exc.reason.startswith(PATH_REFUSED_PREFIX)
    else:
        assert res.success, res.error
        written = [p for p in (world.saved / "tmp" / SLUG).rglob("out.json")]
        assert len(written) == 1
    assert not outside.exists() and not outside.parent.exists()


def test_aliases_never_cover_project_or_expression():
    assert "project" not in tds._ALIASES and "expression" not in tds._ALIASES
    flat = {a for aliases in tds._ALIASES.values() for a in aliases}
    assert not flat & {"project", "expression"}
    assert "data" not in ("project", "expression")


def test_only_the_datastore_tool_folds_argument_aliases():
    import captain_claw.tools as pkg

    owners = set()
    for attr in dir(pkg):
        obj = getattr(pkg, attr)
        if isinstance(obj, type) and issubclass(obj, Tool) and obj is not Tool:
            if getattr(obj, "name", "") in SPEAKER_TOOL_ALLOWLIST_MAX:
                mod = importlib.import_module(obj.__module__)
                if hasattr(mod, "_ALIASES") or hasattr(mod, "_normalize_arg_aliases"):
                    owners.add(obj.__module__)
    assert owners == {"captain_claw.tools.datastore"}


async def test_direct_tool_calls_recheck_member_paths(world):
    await _seed_mixed(world.dm)
    with _Bound(ANA):
        res = await DatastoreTool._import_file(world.dm, {
            "file_path": str(world.ws / "owner.csv"), "_runtime_base_path": world.ws})
        assert not res.success and res.error == (
            PATH_REFUSED_PREFIX
            + "import_file reads only this agent's saved/ files and your own VFS folders")
        res = await DatastoreTool._export(world.dm, {
            "table": "mixed", "file_path": str(world.ws / "output" / "x.csv"),
            "_saved_base_path": world.saved, "_runtime_base_path": world.ws,
            "_session_id": SLUG})
        assert not res.success and res.error == (
            PATH_REFUSED_PREFIX + "export writes only into this conversation's saved/ folder")
    assert not (world.ws / "output" / "x.csv").exists()
    assert [t.name for t in await world.dm.list_tables()] == ["mixed"]


async def test_a_member_import_error_shows_no_host_path(world):
    gone = world.saved / "tmp" / SLUG / "gone.csv"
    gone.parent.mkdir(parents=True, exist_ok=True)
    gone.write_text("k\n1\n")
    reason = None
    try:
        await call(world, {"action": "import_file", "file_path": f"saved/tmp/{SLUG}/missing.csv"})
    except ToolBlockedError as exc:
        reason = exc.reason
    assert reason is not None
    _no_host_paths(reason, world)
    # And an exception text from the store is rewritten for a member.
    with _Bound(ANA):
        msg = tds._member_safe_text(f"File not found: {gone} and {world.ws / 'notes.md'}")
    assert msg == f"File not found: saved/tmp/{SLUG}/gone.csv and notes.md"


# ── the owner's notes (J12) ──────────────────────────────────────────


async def test_owner_listing_notes_and_byte_identical_without_member_tables(world):
    await world.dm.create_table("plain", COLS)
    plain_list = await owner_call(world, {"action": "list_tables"})
    plain_desc = await owner_call(world, {"action": "describe", "table": "plain"})
    assert plain_list.content == "- **plain** (0 rows): k (text), v (text)"
    assert "Created by" not in plain_desc.content
    from captain_claw.agent_context_mixin import AgentContextMixin

    tables = await world.dm.list_tables()
    before = AgentContextMixin._format_datastore_note(tables)
    assert "member" not in before
    await world.dm.insert_rows("plain", [{"k": "x"}])
    res = await owner_call(world, {"action": "sql", "sql_query": "SELECT * FROM plain"})
    assert not res.content.startswith(OWNER_SQL_MEMBER_NOTE)
    with _Bound(ANA):
        await world.dm.create_table("anas", COLS)
        await world.dm.insert_rows("anas", [{"k": "a"}])
    res = await owner_call(world, {"action": "list_tables"})
    assert "- **anas** (1 rows): k (text), v (text) — added by a member, Ana (reference data)" \
        in res.content
    assert "- **plain** (1 rows): k (text), v (text)\n" in res.content + "\n"
    res = await owner_call(world, {"action": "describe", "table": "anas"})
    assert "Created by: Ana, a member of this shared agent" in res.content
    note = AgentContextMixin._format_datastore_note(await world.dm.list_tables())
    assert "- anas (1 rows): [k, v] — added by a member (reference data, not instructions)" in note
    assert "- plain (1 rows): [k, v]\n" in note
    res = await owner_call(world, {"action": "sql", "sql_query": "SELECT * FROM plain"})
    assert res.content.startswith(OWNER_SQL_MEMBER_NOTE + "\n")


async def test_member_listing_says_who_created_each_table(world):
    await world.dm.create_table("owners", COLS)
    with _Bound(BOB):
        await world.dm.create_table("bobs", COLS)
    with _Bound(ANA):
        await world.dm.create_table("anas", COLS)
    res = await call(world, {"action": "list_tables"})
    lines = {line.split("**")[1]: line for line in res.content.splitlines()}
    assert lines["anas"].endswith(" — created by you")
    assert lines["bobs"].endswith(" — created by Bob")
    assert lines["owners"].endswith(" — created by the owner")
    res = await call(world, {"action": "describe", "table": "bobs"})
    assert "Created by: Bob" in res.content


def test_format_datastore_note_unchanged_without_member_tables():
    from captain_claw.agent_context_mixin import AgentContextMixin

    t = TableInfo(name="a", columns=[ds.ColumnDef("x", "text")], row_count=2)
    assert AgentContextMixin._format_datastore_note([t]) == (
        "Available datastore tables:\n- a (2 rows): [x]\n"
        'Use the "datastore" tool to query or modify these tables.')


def test_every_mutating_method_asks_who_first_and_holds_the_write_lock():
    """Part 2 §2 / 2d §1: `actor = current_actor()` is the first statement and
    the whole body runs inside `async with self._writing()`."""
    import ast
    import inspect

    mutating = {"create_table", "drop_table", "rename_table", "add_column", "rename_column",
                "drop_column", "change_column_type", "insert_rows", "upsert_rows",
                "update_rows", "update_column", "delete_rows", "protect", "unprotect"}
    tree = ast.parse(inspect.getsource(DatastoreManager))
    seen = set()
    for fn in tree.body[0].body:
        if isinstance(fn, ast.AsyncFunctionDef) and fn.name in mutating:
            body = [s for s in fn.body
                    if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            assert ast.unparse(body[0]) == "actor = current_actor()", fn.name
            assert isinstance(body[1], ast.AsyncWith), fn.name
            assert ast.unparse(body[1].items[0].context_expr) == "self._writing()", fn.name
            seen.add(fn.name)
    assert seen == mutating


# ── the scoping helper (part 2d §2) ──────────────────────────────────


def test_scope_where_is_the_one_place_a_member_filter_is_built(tmp_path):
    mgr = DatastoreManager(db_path=tmp_path / "x.db")
    member = ds.DatastoreActor("member", "u-ana", "Ana")
    assert mgr._scope_where("", [], ds.OWNER_ACTOR) == ("", [])
    assert mgr._scope_where('WHERE "k" = ?', ["a"], ds.OWNER_ACTOR) == ('WHERE "k" = ?', ["a"])
    assert mgr._scope_where("", [], member) == ('WHERE "_created_by" = ?', ["u-ana"])
    assert mgr._scope_where('WHERE "k" = ? OR "v" = ?', ["a", "b"], member) == (
        'WHERE ("k" = ? OR "v" = ?) AND "_created_by" = ?', ["a", "b", "u-ana"])
    assert mgr._scope_where("", [], member, foreign=True) == (
        'WHERE NOT ("_created_by" = ?)', ["u-ana"])
    with pytest.raises(ValueError):
        mgr._scope_where('"k" = ?', ["a"], member)


async def test_a_member_update_matching_a_foreign_row_is_refused(dm):
    """The member's creator test wraps the WHOLE built clause."""
    await _seed_mixed(dm)
    with _Bound(ANA):
        with pytest.raises(MemberDeniedError):
            await dm.update_rows("mixed", {"v": "X"},
                                 {"k": {"op": "IN", "value": ["ana1", "bob1"]}})
        assert await dm.update_rows("mixed", {"v": "ok"},
                                    {"k": {"op": "LIKE", "value": "ana%"}}) == 1
    assert await _raw(dm, "SELECT v FROM \"ds_mixed\" WHERE k = 'bob1'") == [("b",)]


# ── what a member's rows may store ───────────────────────────────────

_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _shared_string_xlsx(path: Path, big_len: int, ncols: int, nrows: int) -> None:
    """An xlsx whose every data cell points at ONE shared string of big_len chars."""
    def col(i):
        out, i = "", i + 1
        while i:
            i, r = divmod(i - 1, 26)
            out = chr(65 + r) + out
        return out

    shared = [f"h{c}" for c in range(ncols)] + ["Z" * big_len]
    ss = f'<?xml version="1.0"?><sst xmlns="{_NS}">' + "".join(
        f"<si><t>{x}</t></si>" for x in shared) + "</sst>"
    rows = ['<row r="1">' + "".join(f'<c r="{col(c)}1" t="s"><v>{c}</v></c>'
                                    for c in range(ncols)) + "</row>"]
    for r in range(2, nrows + 2):
        rows.append(f'<row r="{r}">' + "".join(
            f'<c r="{col(c)}{r}" t="s"><v>{ncols}</v></c>' for c in range(ncols)) + "</row>")
    sheet = (f'<?xml version="1.0"?><worksheet xmlns="{_NS}"><sheetData>{"".join(rows)}'
             "</sheetData></worksheet>")
    wb = (f'<?xml version="1.0"?><workbook xmlns="{_NS}" xmlns:r="http://schemas.openxmlformats.org'
          '/officeDocument/2006/relationships"><sheets><sheet name="S" sheetId="1" r:id="rId1"/>'
          "</sheets></workbook>")
    rels = ('<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package'
            '/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/>'
            "</Relationships>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/sharedStrings.xml", ss)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", rels)


def _db_bytes(mgr: DatastoreManager) -> int:
    return sum(Path(str(mgr.db_path) + x).stat().st_size for x in ("", "-wal")
               if Path(str(mgr.db_path) + x).exists())


async def test_a_shared_string_xlsx_cannot_fill_the_store(world):
    """2 MB shared string × 25 cells × 2 rows = 100 MB stored from a 3 KB file."""
    src = world.saved / "downloads" / SLUG / "amp.xlsx"
    src.parent.mkdir(parents=True)
    _shared_string_xlsx(src, 2_000_000, 25, 2)
    _stamp_as(ANA, src)
    assert src.stat().st_size < ds.MEMBER_IMPORT_MAX_BYTES
    before = _db_bytes(world.dm)
    res = await call(world, {"action": "import_file",
                             "file_path": f"saved/downloads/{SLUG}/amp.xlsx", "table": "amp"})
    assert not res.success and DS_MEMBER_STORAGE_LIMIT in res.error
    assert _db_bytes(world.dm) - before < 1_000_000
    assert [t.name for t in await world.dm.list_tables()] == []


async def test_member_stored_bytes_are_capped_across_imports_but_not_the_owners(
    dm, tmp_path, monkeypatch,
):
    monkeypatch.setattr(ds, "MEMBER_MAX_STORED_BYTES", 30_000)
    src = tmp_path / "one.xlsx"
    _shared_string_xlsx(src, 10_000, 2, 1)              # 20,000 bytes stored per import
    with _Bound(ANA):
        assert (await dm.import_xlsx(src, "grow"))["rows_imported"] == 1
        with pytest.raises(MemberDeniedError) as info:
            await dm.import_xlsx(src, "grow", append=True)
        assert str(info.value) == DS_MEMBER_STORAGE_LIMIT
        with pytest.raises(MemberDeniedError):
            await dm.import_xlsx(src, "second")       # nothing created either
    assert [t.name for t in await dm.list_tables()] == ["grow"]
    assert (await dm.describe_table("grow")).row_count == 1
    for _ in range(3):                                   # the owner: unchanged
        await dm.import_xlsx(src, "grow", append=True)
    assert (await dm.describe_table("grow")).row_count == 4


async def test_a_csv_append_over_the_member_quota_is_refused(dm, tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "MEMBER_MAX_STORED_BYTES", 2_500)
    path = tmp_path / "one.csv"
    path.write_text("a,b\n" + "x" * 500 + "," + "y" * 500 + "\n")
    with _Bound(ANA):
        await dm.import_csv(path, "grow")
        await dm.import_csv(path, "grow", append=True)
        with pytest.raises(MemberDeniedError) as info:
            await dm.import_csv(path, "grow", append=True)
    assert str(info.value) == DS_MEMBER_STORAGE_LIMIT
    assert (await dm.describe_table("grow")).row_count == 2


async def test_a_refused_insert_rolls_back_the_whole_batch_and_the_import_table(
    dm, tmp_path, monkeypatch,
):
    monkeypatch.setattr(ds, "MEMBER_MAX_STORED_BYTES", 1_000)
    await dm.create_table("t", COLS)
    with _Bound(ANA):
        await dm.insert_rows("t", [{"k": "a" * 400}])
        with pytest.raises(MemberDeniedError) as info:
            await dm.insert_rows("t", [{"k": "b" * 400}, {"k": "c" * 400}])
        assert str(info.value) == DS_MEMBER_STORAGE_LIMIT
        assert (await dm.describe_table("t")).row_count == 1
        # The quota is what is stored: freeing rows makes room again.
        await dm.delete_rows("t", {"_mine": True})
        assert await dm.insert_rows("t", [{"k": "b" * 400}, {"k": "c" * 400}]) == 2
        # Measured at insert time (under the lock) even when the import's
        # precheck is skipped: the table the import created goes too.

        async def no_precheck(*a, **k):
            return None

        monkeypatch.setattr(dm, "_member_import_bytes", no_precheck)
        path = tmp_path / "late.csv"
        path.write_text("k,v\n" + "z" * 300 + ",1\n")
        with pytest.raises(MemberDeniedError):
            await dm.import_csv(path, "late")
    assert [t.name for t in await dm.list_tables()] == ["t"]


async def test_member_updates_count_what_they_add(dm, monkeypatch):
    monkeypatch.setattr(ds, "MEMBER_MAX_STORED_BYTES", 1_000)
    with _Bound(ANA):
        await dm.create_table("u", COLS, unique=["k"])
        await dm.insert_rows("u", [{"k": "1", "v": "x" * 300}, {"k": "2", "v": "x" * 300}])
        with pytest.raises(MemberDeniedError) as info:
            await dm.update_rows("u", {"v": "y" * 600}, {"_mine": True})
        assert str(info.value) == DS_MEMBER_STORAGE_LIMIT
        with pytest.raises(MemberDeniedError):
            await dm.update_column("u", "v", value="y" * 600)
        with pytest.raises(MemberDeniedError):
            await dm.upsert_rows("u", [{"k": "3", "v": "y" * 500}])
        # Same-size refreshes and shrinking always fit.
        assert await dm.upsert_rows("u", [{"k": "1", "v": "z" * 300}]) == 1
        assert await dm.update_rows("u", {"v": "s"}, {"k": "2"}) == 1
        assert await dm.update_column("u", "v", value="w" * 400) == 2
    rows = await _raw(dm, 'SELECT k, v FROM "ds_u" ORDER BY k')
    assert rows == [("1", "w" * 400), ("2", "w" * 400)]


async def test_a_column_default_counts_in_every_row_that_leaves_it_out(dm, monkeypatch):
    monkeypatch.setattr(ds, "MEMBER_MAX_STORED_BYTES", 1_000)
    with _Bound(ANA):
        await dm.create_table("d", COLS)
        await dm.add_column("d", "pad", "text", default="p" * 400)
        await dm.insert_rows("d", [{"k": "a"}, {"k": "b"}])
        with pytest.raises(MemberDeniedError) as info:
            await dm.insert_rows("d", [{"k": "c"}])
    assert str(info.value) == DS_MEMBER_STORAGE_LIMIT
    with _Bound(BOB):                     # a default counts in UTF-8 bytes, not characters
        await dm.create_table("e", COLS)
        await dm.add_column("e", "pad", "text", default="é" * 200)
        with pytest.raises(MemberDeniedError):
            await dm.insert_rows("e", [{"k": "a"}, {"k": "b"}, {"k": "c"}])
        assert await dm.insert_rows("e", [{"k": "a"}, {"k": "b"}]) == 2


async def test_one_member_value_is_capped_the_owners_not(dm, monkeypatch):
    monkeypatch.setattr(ds, "MEMBER_MAX_VALUE_BYTES", 1_000)
    await dm.create_table("t", COLS, unique=["k"])
    big = "é" * 600                                      # 1,200 bytes in UTF-8
    with _Bound(ANA):
        await dm.insert_rows("t", [{"k": "mine", "v": "ok"}])
        for op in (lambda: dm.insert_rows("t", [{"k": "x", "v": big}]),
                   lambda: dm.upsert_rows("t", [{"k": "mine", "v": big}]),
                   lambda: dm.update_rows("t", {"v": big}, {"_mine": True})):
            with pytest.raises(MemberDeniedError) as info:
                await op()
            assert str(info.value) == DS_MEMBER_VALUE_TOO_LARGE
        await dm.create_table("own", COLS)
        for op in (lambda: dm.update_column("own", "v", value=big),
                   lambda: dm.add_column("own", "z", "text", default=big)):
            with pytest.raises(MemberDeniedError) as info:
                await op()
            assert str(info.value) == DS_MEMBER_VALUE_TOO_LARGE
    await dm.insert_rows("t", [{"k": "owner", "v": big}])
    assert (await dm.describe_table("t")).row_count == 2


async def test_member_tables_and_imports_have_at_most_the_column_cap(dm, tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "MEMBER_MAX_COLUMNS", 3)
    four = [{"name": f"c{i}", "type": "text"} for i in range(4)]
    path = tmp_path / "wide.csv"
    path.write_text("a,b,c,d\n1,2,3,4\n")
    xlsx = tmp_path / "wide.xlsx"
    _shared_string_xlsx(xlsx, 5, 4, 1)
    with _Bound(ANA):
        with pytest.raises(MemberDeniedError) as info:
            await dm.create_table("wide", four)
        assert str(info.value) == DS_MEMBER_TOO_MANY_COLUMNS
        await dm.create_table("narrow", four[:3])
        with pytest.raises(MemberDeniedError):
            await dm.add_column("narrow", "c3", "text")
        for imp in (lambda: dm.import_csv(path, "wide"), lambda: dm.import_xlsx(xlsx, "wide")):
            with pytest.raises(MemberDeniedError) as info:
                await imp()
            assert str(info.value) == DS_MEMBER_TOO_MANY_COLUMNS
    assert [t.name for t in await dm.list_tables()] == ["narrow"]
    await dm.create_table("owners", four)                 # the owner: unchanged
    await dm.import_csv(path, "owner_wide")


async def test_a_default_added_to_rows_already_there_counts_and_never_lands_in_others(
    dm, monkeypatch,
):
    """200 rows × a 400-byte default = 80 KB read back (and written out by
    the next update) — past a 10 KB quota, from one add_column."""
    monkeypatch.setattr(ds, "MEMBER_MAX_STORED_BYTES", 10_000)
    with _Bound(ANA):
        await dm.create_table("d", COLS)
        await dm.insert_rows("d", [{"k": str(i)} for i in range(200)])
        with pytest.raises(MemberDeniedError) as info:
            await dm.add_column("d", "pad", "text", default="p" * 400)
        assert str(info.value) == DS_MEMBER_STORAGE_LIMIT
        assert await dm.add_column("d", "small", "text", default="p" * 20) is True
        assert await dm.add_column("d", "plain", "text") is True
        await dm.create_table("shared", COLS)
    with _Bound(BOB):
        await dm.insert_rows("shared", [{"k": "bob"}])
    with _Bound(ANA):
        with pytest.raises(MemberDeniedError) as info:
            await dm.add_column("shared", "pad", "text", default="p")
        assert str(info.value) == DS_FOREIGN_ROWS
        assert await dm.add_column("shared", "plain", "text") is True
    assert [c.name for c in (await dm.describe_table("d")).columns] == ["k", "v", "small", "plain"]
    await dm.add_column("shared", "pad", "text", default="p" * 400)   # the owner: unchanged


async def test_member_names_are_bounded_so_an_xlsx_header_cannot_fill_the_schema(
    dm, tmp_path,
):
    """100 header cells sharing one 200 KB string = a 100 MB schema from a 2 KB file."""
    src = tmp_path / "head.xlsx"
    _shared_string_xlsx(src, 5, 1, 1)
    with zipfile.ZipFile(src) as z:
        parts = {n: z.read(n).decode() for n in z.namelist()}
    parts["xl/sharedStrings.xml"] = parts["xl/sharedStrings.xml"].replace("h0", "H" * 200_000)
    parts["xl/worksheets/sheet1.xml"] = parts["xl/worksheets/sheet1.xml"].replace(
        '<c r="A1" t="s"><v>0</v></c>',
        "".join(f'<c r="{c}1" t="s"><v>0</v></c>' for c in ("A", "B", "C")))
    with zipfile.ZipFile(src, "w", zipfile.ZIP_DEFLATED) as z:
        for n, body in parts.items():
            z.writestr(n, body)
    long = "n" * (ds.MEMBER_MAX_NAME_CHARS + 1)
    with _Bound(ANA):
        with pytest.raises(MemberDeniedError) as info:
            await dm.import_xlsx(src, "head")
        assert str(info.value) == ds.DS_MEMBER_NAME_TOO_LONG
        for op in (lambda: dm.create_table(long, COLS),
                   lambda: dm.create_table("t", [{"name": long, "type": "text"}])):
            with pytest.raises(MemberDeniedError) as info:
                await op()
            assert str(info.value) == ds.DS_MEMBER_NAME_TOO_LONG
        await dm.create_table("t", COLS)
        for op in (lambda: dm.add_column("t", long), lambda: dm.rename_column("t", "v", long),
                   lambda: dm.rename_table("t", long)):
            with pytest.raises(MemberDeniedError) as info:
                await op()
            assert str(info.value) == ds.DS_MEMBER_NAME_TOO_LONG
    assert [t.name for t in await dm.list_tables()] == ["t"]
    assert (await dm.import_xlsx(src, "head"))["rows_imported"] == 1   # the owner: unchanged


# ── member table names never rewrite someone else's SQL ──────────────


async def test_a_member_table_named_like_a_column_never_changes_owner_sql(dm):
    await dm.create_table("leads", [{"name": "email", "type": "text"},
                                    {"name": "status", "type": "text"}])
    await dm.insert_rows("leads", [{"email": "a@x", "status": "new"},
                                   {"email": "b@x", "status": "done"}])
    q = "SELECT email FROM leads WHERE status = 'new'"
    before = await dm.raw_select(q)
    assert before["rows"] == [["a@x"]]
    with _Bound(ANA):
        for name in ("email", "status", "count", "from", "rows", "lower", "rowid"):
            with pytest.raises(MemberDeniedError) as info:
                await dm.create_table(name, COLS)
            assert str(info.value) == DS_MEMBER_TABLE_NAME, name
        await dm.create_table("harmless", COLS)
        with pytest.raises(MemberDeniedError) as info:
            await dm.rename_table("harmless", "status")
        assert str(info.value) == DS_MEMBER_TABLE_NAME
    assert await dm.raw_select(q) == before
    assert (await dm.raw_select("select count(*) from leads"))["rows"] == [[2]]


async def test_a_member_table_never_shadows_sqlite_tables_internal_names_or_modules(dm):
    await dm.create_table("leads", [{"name": "email", "type": "text"}])
    q = "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'ds_leads'"
    assert (await dm.raw_select(q))["rows"] == [["ds_leads"]]
    with _Bound(ANA):
        for name in ("sqlite_master", "sqlite_schema", "ds_leads", "pragma_table_info",
                     "json_each", "json_tree", "main"):
            with pytest.raises(MemberDeniedError) as info:
                await dm.create_table(name, [{"name": "name", "type": "text"}])
            assert str(info.value) == DS_MEMBER_TABLE_NAME, name
    assert (await dm.raw_select(q))["rows"] == [["ds_leads"]]
    assert (await dm.raw_select("SELECT email FROM ds_leads"))["columns"] == ["email"]


async def test_a_member_table_never_rewrites_a_string_or_a_quoted_cte(dm):
    await dm.create_table("leads", [{"name": "note", "type": "text"}])
    await dm.insert_rows("leads", [{"note": "away from home"}])
    q = "SELECT note FROM leads WHERE note = 'away from home'"
    assert (await dm.raw_select(q))["rows"] == [["away from home"]]
    with _Bound(ANA):
        await dm.create_table("home", [{"name": "x", "type": "text"}])
        await dm.insert_rows("home", [{"x": "ana"}])
    assert (await dm.raw_select(q))["rows"] == [["away from home"]]
    assert (await dm.raw_select("SELECT 'it''s from home', x FROM home"))["rows"] == [
        ["it's from home", "ana"]]
    for cte in ('WITH "home" AS (SELECT \'cte\' AS x) SELECT x FROM home',
                "WITH `home` AS (SELECT 'cte' AS x) SELECT x FROM home",
                "WITH home /* c */ AS (SELECT 'cte' AS x) SELECT x FROM home",
                "WITH home -- c\n AS (SELECT 'cte' AS x) SELECT x FROM home"):
        assert (await dm.raw_select(f"SELECT x FROM ({cte})"))["rows"] == [["cte"]], cte


async def test_a_member_table_is_substituted_only_after_from_or_join(dm):
    await dm.create_table("leads", [{"name": "email", "type": "text"}])
    await dm.insert_rows("leads", [{"email": "a@x"}])
    with _Bound(ANA):
        await dm.create_table("flag", [{"name": "email", "type": "text"},
                                       {"name": "x", "type": "text"}])
        await dm.insert_rows("flag", [{"email": "a@x", "x": "ana"}])
    # A column the owner adds later under that name is never rewritten.
    await dm.add_column("leads", "flag", "text", default="yes")
    q = "SELECT email, flag FROM leads WHERE flag = 'yes'"
    assert (await dm.raw_select(q))["rows"] == [["a@x", "yes"]]
    # ... nor a CTE of that name.
    cte = "SELECT x FROM (WITH flag AS (SELECT 'cte' AS x) SELECT x FROM flag)"
    assert (await dm.raw_select(cte))["rows"] == [["cte"]]
    # Members (and the owner) still reach the member's table as a table.
    for p in (ANA, BOB, None):
        with _Bound(p):
            for sql in ("SELECT x FROM flag", 'select x from "flag"', "SELECT x FROM\nflag",
                        "SELECT f.x FROM leads l JOIN flag AS f ON f.email = l.email",
                        "SELECT x FROM leads INNER JOIN flag USING (email)"):
                assert (await dm.raw_select(sql))["rows"] == [["ana"]], (p, sql)
    # The owner's own tables keep the old word-for-word mapping.
    assert (await dm.raw_select("SELECT leads.email FROM leads"))["rows"] == [["a@x"]]


# ── change_column_type never leaves its rebuild table behind ─────────


async def test_a_unique_collision_in_change_column_type_changes_nothing_and_can_retry(dm):
    await dm.create_table("k", COLS, unique=["k"])
    await dm.insert_rows("k", [{"k": "1", "v": "a"}, {"k": "1.0", "v": "b"}])
    for _ in range(2):                    # the second call fails the same way, not on __tmp
        with pytest.raises(ValueError) as info:
            await dm.change_column_type("k", "k", "integer")
        assert str(info.value) == ("converting k would make values of the unique key collide; "
                                   "the column was not changed")
    tables = {r[0] for r in await _raw(dm, "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "ds_k__tmp" not in tables and "ds_k" in tables
    assert [c.col_type for c in (await dm.describe_table("k")).columns] == ["text", "text"]
    assert await _raw(dm, 'SELECT k, v FROM "ds_k" ORDER BY _id') == [("1", "a"), ("1.0", "b")]
    await dm.delete_rows("k", {"k": "1.0"})
    assert await dm.change_column_type("k", "k", "integer") is True
    assert (await dm.describe_table("k")).columns[0].col_type == "integer"
    assert await dm._unique_columns("ds_k") == ["k"]
    assert await _raw(dm, 'SELECT k FROM "ds_k"') == [(1,)]


async def test_change_column_type_clears_a_stray_rebuild_table(dm):
    await dm.create_table("k", COLS)
    await dm.insert_rows("k", [{"k": "1"}])
    async with aiosqlite.connect(str(dm.db_path)) as db:
        await db.execute('CREATE TABLE "ds_k__tmp" (x)')   # left by an older rebuild
        await db.commit()
    assert await dm.change_column_type("k", "k", "integer") is True
    assert await _raw(dm, 'SELECT k FROM "ds_k"') == [(1,)]
