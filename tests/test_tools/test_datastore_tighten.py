"""Datastore writes should land in one call, not three.

Read off a real agent's log: a leads table with a unique key on `email`,
and the agent never once used `upsert`. Adding leads went insert →
"UNIQUE constraint failed: ds_leads.email" (nothing inserted) → query the
table to see which already exist → insert the rest. Other agents on the same
deck sent `upsert` with a `key` that wasn't the table's unique key, got
"ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE constraint", and
guessed again. An `update` whose row id slipped out of `where` reached every
row. These tests pin the one-call answers.
"""

import pytest

from captain_claw import datastore as ds
from captain_claw.agent_context_mixin import AgentContextMixin
from captain_claw.datastore import DatastoreManager
from captain_claw.tools import datastore as ds_tool
from captain_claw.tools.datastore import DatastoreTool

LEADS = [
    {"name": "org", "type": "text"},
    {"name": "email", "type": "text"},
    {"name": "status", "type": "text"},
]


@pytest.fixture
async def dm(tmp_path, monkeypatch):
    mgr = DatastoreManager(db_path=tmp_path / "store.db")
    await mgr.create_table("leads", LEADS, unique=["email"])
    await mgr.insert_rows("leads", [
        {"org": "Acme", "email": "ana@acme.test", "status": "contacted"},
        {"org": "Bolt", "email": "ivo@bolt.test", "status": "new"},
    ])
    monkeypatch.setattr(ds_tool, "_resolve_datastore_manager", lambda _sid: mgr)
    yield mgr
    await mgr.close()


async def call(**kwargs):
    return await DatastoreTool().execute(**kwargs)


async def rows(dm, table="leads"):
    res = await dm.query(table, None, None, ["_id"], None, 0)
    return [dict(zip(res["columns"], r)) for r in res["rows"]]


# ── insert into a table with a unique key ────────────────────────────


async def test_insert_clash_names_the_rows_and_the_one_field_fix(dm):
    res = await call(action="insert", table="leads", rows=[
        {"org": "Cobalt", "email": "mia@cobalt.test", "status": "new"},
        {"org": "Acme", "email": "ana@acme.test", "status": "new"},
    ])
    assert not res.success
    assert "Nothing was inserted" in res.error
    assert '"ana@acme.test" (row 1)' in res.error
    assert '"on_conflict": "skip"' in res.error and '"action": "upsert"' in res.error
    assert len(await rows(dm)) == 2


async def test_insert_skip_adds_the_new_rows_and_leaves_existing_alone(dm):
    res = await call(action="insert", table="leads", on_conflict="skip", rows=[
        {"org": "Cobalt", "email": "mia@cobalt.test", "status": "new"},
        {"org": "Acme", "email": "ana@acme.test", "status": "new"},
        {"org": "Cobalt 2", "email": "mia@cobalt.test", "status": "new"},
    ])
    assert res.success, res.error
    assert "Inserted 1 row(s)" in res.content and "Skipped 2" in res.content
    assert "repeated in this call" in res.content
    by_email = {r["email"]: r for r in await rows(dm)}
    assert by_email["ana@acme.test"]["status"] == "contacted"     # untouched
    assert by_email["mia@cobalt.test"]["org"] == "Cobalt"          # the first one wins


async def test_insert_on_conflict_update_runs_as_upsert(dm):
    res = await call(action="insert", table="leads", on_conflict="update", rows=[
        {"org": "Acme", "email": "ana@acme.test", "status": "replied"},
    ])
    assert res.success, res.error
    assert "1 updated" in res.content
    assert {r["email"]: r["status"] for r in await rows(dm)}["ana@acme.test"] == "replied"


async def test_insert_rows_without_a_key_value_still_go_in(dm):
    """NULL never collides with a UNIQUE key — leads without an email yet."""
    res = await call(action="insert", table="leads", rows=[{"org": "Delta"}, {"org": "Echo"}])
    assert res.success, res.error
    assert len(await rows(dm)) == 4


async def test_insert_says_which_columns_it_could_not_store(dm):
    res = await call(action="insert", table="leads",
                     rows=[{"org": "Fox", "email": "f@fox.test", "phone": "123"}])
    assert res.success
    assert "`phone`" in res.content and "add_column" in res.content


# ── upsert ───────────────────────────────────────────────────────────


async def test_upsert_reports_what_it_inserted_and_updated(dm):
    res = await call(action="upsert", table="leads", rows=[
        {"org": "Acme", "email": "ana@acme.test", "status": "replied"},
        {"org": "Cobalt", "email": "mia@cobalt.test", "status": "new"},
    ])
    assert res.success, res.error
    assert "on **email**" in res.content
    assert "1 inserted" in res.content and "1 updated" in res.content


async def test_upsert_with_the_wrong_key_matches_on_the_unique_key(dm):
    res = await call(action="upsert", table="leads", key="org", rows=[
        {"org": "Acme Ltd", "email": "ana@acme.test", "status": "replied"},
    ])
    assert res.success, res.error
    assert "not this table's unique key" in res.content
    got = await rows(dm)
    assert len(got) == 2 and got[0]["org"] == "Acme Ltd"


async def test_upsert_wrong_key_and_rows_without_the_unique_key_says_so(dm):
    res = await call(action="upsert", table="leads", key="org",
                     rows=[{"org": "Acme", "status": "x"}])
    assert not res.success
    assert "unique key (email)" in res.error


async def test_upsert_on_a_table_without_unique_key_matches_on_key(dm):
    await dm.create_table("tenders", [{"name": "ref", "type": "integer"},
                                      {"name": "title", "type": "text"}])
    await dm.insert_rows("tenders", [{"ref": 101, "title": "old"}])
    res = await call(action="upsert", table="tenders", key="ref", rows=[
        {"ref": 101, "title": "new"}, {"ref": 102, "title": "fresh"},
    ])
    assert res.success, res.error
    assert "1 inserted" in res.content and "1 updated" in res.content
    assert {r["ref"]: r["title"] for r in await rows(dm, "tenders")} == {101: "new", 102: "fresh"}


async def test_upsert_without_any_key_explains_what_to_send(dm):
    await dm.create_table("notes", [{"name": "body", "type": "text"}])
    res = await call(action="upsert", table="notes", rows=[{"body": "x"}])
    assert not res.success
    assert "no unique key" in res.error and '"key"' in res.error


async def test_upsert_rows_missing_the_key_are_all_named_at_once(dm):
    res = await call(action="upsert", table="leads", rows=[
        {"org": "A", "email": "a@a.test"}, {"org": "B"}, {"org": "C", "email": None},
    ])
    assert not res.success
    assert "2 of 3" in res.error and "1, 2" in res.error and "action=insert" in res.error
    assert len(await rows(dm)) == 2


# ── update ───────────────────────────────────────────────────────────


async def test_update_without_where_is_refused(dm):
    res = await call(action="update", table="leads", set_values={"status": "lost"})
    assert not res.success
    assert "ALL 2 rows" in res.error and '"_all": true' in res.error
    assert {r["status"] for r in await rows(dm)} == {"contacted", "new"}


async def test_update_all_rows_when_asked(dm):
    res = await call(action="update", table="leads", set_values={"status": "lost"},
                     where={"_all": True})
    assert res.success and "Updated 2" in res.content


async def test_update_with_a_stray_top_level_id_touches_only_that_row(dm):
    first = (await rows(dm))[0]["_id"]
    res = await call(action="update", table="leads", _id=first,
                     set_values={"status": "replied"})
    assert res.success, res.error
    assert "Updated 1" in res.content and "`where`" in res.content
    assert [r["status"] for r in await rows(dm)] == ["replied", "new"]


async def test_update_value_object_is_taken_as_set_values(dm):
    res = await call(action="update", table="leads", value='{"status": "won"}',
                     where={"email": "ivo@bolt.test"})
    assert res.success, res.error
    assert {r["email"]: r["status"] for r in await rows(dm)}["ivo@bolt.test"] == "won"


async def test_update_onto_another_rows_key_names_that_row(dm):
    got = await rows(dm)
    res = await call(action="update", table="leads", where={"_id": got[1]["_id"]},
                     set_values={"email": "ana@acme.test", "status": "x"})
    assert not res.success
    assert f'_id {got[0]["_id"]}' in res.error and "Acme" in res.error
    assert "drop email from set_values" in res.error


# ── reads ────────────────────────────────────────────────────────────


async def test_query_id_means_the_row_id(dm):
    res = await call(action="query", table="leads", columns="id, org", where={"id": 1})
    assert res.success, res.error
    assert "`_id`" in res.content and "Acme" in res.content and "Bolt" not in res.content


async def test_sql_id_is_retried_as_row_id(dm):
    res = await call(action="sql", sql_query="SELECT id, org FROM leads ORDER BY id")
    assert res.success, res.error
    assert "`id` → `_id`" in res.content and "Acme" in res.content


async def test_sql_ilike_runs_as_like(dm):
    res = await call(action="sql", sql_query="SELECT org FROM leads WHERE org ILIKE '%ACME%'")
    assert res.success, res.error
    assert "Acme" in res.content and "Bolt" not in res.content


async def test_query_with_sql_runs_the_sql_instead_of_dropping_it(dm):
    res = await call(action="query", sql="SELECT org FROM leads WHERE org = 'Bolt'")
    assert res.success, res.error
    assert 'action="sql"' in res.content
    assert "Bolt" in res.content and "Acme" not in res.content


async def test_action_aliases(dm):
    res = await call(action="sql_query", sql_query="SELECT COUNT(*) AS n FROM leads")
    assert res.success and 'action="sql"' in res.content


async def test_unknown_column_lists_the_real_ones(dm):
    res = await call(action="query", table="leads", columns="org, phone")
    assert not res.success
    assert "Columns of leads: _id, org, email, status" in res.error


async def test_sql_unknown_column_lists_the_real_ones(dm):
    res = await call(action="sql", sql_query="SELECT phone FROM leads")
    assert not res.success
    assert "Columns of leads:" in res.error


# ── telling the model about the key up front ─────────────────────────


async def test_listing_describe_and_prompt_note_show_the_unique_key(dm):
    res = await call(action="list_tables")
    assert "unique key: email" in res.content
    res = await call(action="describe", table="leads")
    assert "Unique key: email" in res.content and "_id" in res.content
    note = AgentContextMixin._format_datastore_note(await dm.list_tables())
    assert "- leads (2 rows): [org, email, status] · unique key: email" in note


async def test_create_table_takes_plain_column_names(dm):
    res = await call(action="create_table", table="contacts",
                     columns='["name", "age integer", "city:text"]')
    assert res.success, res.error
    info = await dm.describe_table("contacts")
    assert [(c.name, c.col_type) for c in info.columns] == [
        ("name", "text"), ("age", "integer"), ("city", "text")]


# ── import ───────────────────────────────────────────────────────────


async def test_import_csv_with_an_unquoted_comma_names_the_line(dm, tmp_path):
    f = tmp_path / "tenders.csv"
    f.write_text("ref,title,buyer\n1,Roof,City\n2,Desks, chairs,School\n3,Paint,Port\n")
    res = await call(action="import_file", file_path=str(f))
    assert not res.success
    assert "line 3 (4 fields)" in res.error and "double quotes" in res.error
    assert "tenders" not in {t.name for t in await dm.list_tables()}


def test_tableinfo_unique_defaults_empty():
    assert ds.TableInfo(name="t").unique == []
