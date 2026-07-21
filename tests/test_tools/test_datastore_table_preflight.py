"""A dropped `table` argument should recover or explain itself.

"Error: 'table' is required." was the datastore tool's most-seen failure —
a dead end that named no tables and threw away the payload the model had
just built. These tests pin the two halves of the answer: infer where a
wrong guess is cheap, and where it isn't, fail with a retryable message.
"""

import re

import pytest

from captain_claw.datastore import DatastoreManager
from captain_claw.tools.datastore import DatastoreTool, _normalize_arg_aliases


@pytest.fixture
async def dm(tmp_path):
    mgr = DatastoreManager(db_path=tmp_path / "store.db")
    await mgr.create_table(
        "fund_portfolio",
        [{"name": "id", "type": "integer"}, {"name": "nm", "type": "text"}],
        unique=["id"],
    )
    await mgr.create_table("notes", [{"name": "body", "type": "text"}])
    yield mgr
    await mgr.close()


async def preflight(dm, action, kwargs, session="s1"):
    return await DatastoreTool()._preflight_table(dm, action, kwargs, session)


async def test_missing_table_error_lists_tables_and_a_retry(dm):
    note, err = await preflight(dm, "upsert", {"rows": "[]"})
    assert note is None and err is not None
    # Everything the model needs to fix the call on the next turn.
    assert "fund_portfolio" in err.error and "notes" in err.error
    assert '"action": "upsert"' in err.error
    assert '"table": "fund_portfolio"' in err.error


async def test_mid_loop_drop_recovers_to_the_table_in_use(dm):
    # The model names the table once…
    await preflight(dm, "upsert", {"table": "fund_portfolio", "rows": "[]"})
    # …then a later call in the same session loses it.
    kwargs = {"rows": "[]"}
    note, err = await preflight(dm, "upsert", kwargs)
    assert err is None
    assert kwargs["table"] == "fund_portfolio"
    assert "fund_portfolio" in note  # the correction is visible to the model


async def test_history_does_not_leak_across_sessions(dm):
    await preflight(dm, "upsert", {"table": "fund_portfolio", "rows": "[]"}, session="a")
    kwargs = {}
    note, err = await preflight(dm, "query", kwargs, session="b")
    assert err is not None and "table" not in kwargs


async def test_single_table_store_infers_without_history(tmp_path):
    mgr = DatastoreManager(db_path=tmp_path / "solo.db")
    await mgr.create_table("only", [{"name": "a", "type": "text"}])
    kwargs = {}
    note, err = await preflight(mgr, "query", kwargs)
    assert err is None and kwargs["table"] == "only"


@pytest.mark.parametrize("action", ["delete", "update", "drop_table", "rename_table",
                                    "drop_column", "change_column_type"])
async def test_destructive_actions_never_infer(dm, action):
    """A wrong guess here mangles data — history must not be enough."""
    await preflight(dm, "upsert", {"table": "fund_portfolio", "rows": "[]"})
    kwargs = {"where": "{}"}
    note, err = await preflight(dm, action, kwargs)
    assert err is not None and "table" not in kwargs


async def test_create_table_error_explains_it_names_the_new_table(dm):
    note, err = await preflight(dm, "create_table", {"columns": "[]"})
    assert err is not None and "names the table to create" in err.error


async def test_empty_store_says_create_one_first(tmp_path):
    mgr = DatastoreManager(db_path=tmp_path / "empty.db")
    note, err = await preflight(mgr, "query", {})
    assert err is not None and "create_table" in err.error


@pytest.mark.parametrize("action,kwargs", [
    ("list_tables", {}),
    ("sql", {"sql_query": "SELECT 1"}),
    ("export", {"sql_query": "SELECT 1"}),   # a query stands in for the table
    ("import_file", {"file_path": "x.csv"}),  # table derived from the filename
])
async def test_actions_that_need_no_table_pass_through(dm, action, kwargs):
    note, err = await preflight(dm, action, kwargs)
    assert note is None and err is None


# ── argument aliases ──
# Names taken from a real agent log: data, table_name, select, order, sql,
# column_type, column_name.

@pytest.mark.parametrize("action,sent,canonical,value", [
    ("update", "data", "set_values", {"nm": "Transmetrics"}),
    ("upsert", "data", "rows", [{"id": 1}]),
    ("query", "table_name", "table", "fund_portfolio"),
    ("query", "select", "columns", "id, nm"),
    ("query", "order", "order_by", "id"),
    ("sql", "sql", "sql_query", "SELECT 1"),
    ("add_column", "column_type", "col_type", "text"),
    ("add_column", "column_name", "column", "sector"),
    ("update", "filter", "where", {"id": 1}),
])
def test_alias_is_renamed_to_the_real_parameter(action, sent, canonical, value):
    kwargs = {sent: value}
    note = _normalize_arg_aliases(action, kwargs)
    assert kwargs.get(canonical) == value
    assert sent not in kwargs
    assert canonical in note  # the model is told what changed


def test_a_real_argument_always_wins_over_its_alias():
    kwargs = {"table": "real", "table_name": "alias"}
    _normalize_arg_aliases("query", kwargs)
    assert kwargs["table"] == "real"


def test_untouched_call_gets_no_note():
    kwargs = {"table": "t", "rows": [{"a": 1}]}
    assert _normalize_arg_aliases("upsert", kwargs) is None


def test_single_row_object_is_wrapped_in_an_array():
    kwargs = {"table": "t", "rows": {"a": 1}}
    _normalize_arg_aliases("upsert", kwargs)
    assert kwargs["rows"] == [{"a": 1}]


def test_set_values_wrapped_in_an_array_is_unwrapped():
    kwargs = {"table": "t", "set_values": [{"a": 1}]}
    _normalize_arg_aliases("update", kwargs)
    assert kwargs["set_values"] == {"a": 1}


async def test_missing_payload_error_shows_what_did_arrive(dm):
    """The tell is usually in the arguments the model DID send."""
    res = await DatastoreTool()._update(dm, {"table": "fund_portfolio", "where": {"id": 1}})
    assert not res.success
    assert "`where`" in res.error and "`table`" in res.error
    assert '"set_values"' in res.error


# ── SQL writes get translated, not just refused ──
# `action="sql"` stays SELECT-only (protection rules live in the structured
# write paths), but the refusal now carries the call that would have worked.

def _sql_error(sql):
    from captain_claw.tools.datastore import _sql_write_error
    return _sql_write_error(sql)


def test_update_statement_comes_back_as_an_update_call():
    err = _sql_error("UPDATE fund_portfolio SET activity_score = 8.5, "
                     "comment = 'strong' WHERE id = 220")
    assert '"action": "update"' in err
    assert '"table": "fund_portfolio"' in err
    assert '"activity_score": 8.5' in err
    assert '"comment": "strong"' in err
    assert '"where": {"id": 220}' in err


def test_insert_statement_comes_back_as_rows():
    err = _sql_error("INSERT INTO notes (body, n) VALUES ('hi, there', 3)")
    assert '"action": "insert"' in err
    assert '"rows": [{"body": "hi, there", "n": 3}]' in err  # comma inside the string survives


def test_delete_statement_keeps_its_filter():
    err = _sql_error("DELETE FROM notes WHERE id = 4 AND body = 'x'")
    assert '"action": "delete"' in err and '"id": 4' in err and '"body": "x"' in err


def test_unparseable_write_still_names_the_right_action():
    err = _sql_error("UPDATE t SET a = b WHERE id > 5")  # `>` isn't a simple filter
    assert 'action="update"' in err


def test_ddl_points_at_the_structured_equivalent():
    assert 'action="create_table"' in _sql_error("CREATE TABLE x (a TEXT)")
    assert 'action="drop_table"' in _sql_error("DROP TABLE x")


async def test_sql_action_refuses_a_write_with_a_usable_retry(dm):
    res = await DatastoreTool()._sql(dm, {"sql_query": "UPDATE fund_portfolio SET nm = 'a' WHERE id = 1"})
    assert not res.success
    assert '"action": "update"' in res.error


async def test_sql_action_still_runs_a_select(dm):
    res = await DatastoreTool()._sql(dm, {"sql_query": "SELECT * FROM ds_fund_portfolio"})
    assert res.success


# ── the context-eater ──
# A repeated `where` key silently became `id <= 379`, returning 379 rows of
# prose: 1.38M characters, 71% of a 200k context, in ONE tool message.

@pytest.fixture
async def wide(tmp_path):
    mgr = DatastoreManager(db_path=tmp_path / "wide.db")
    await mgr.create_table("t", [{"name": "id", "type": "integer"},
                                 {"name": "body", "type": "text"}], unique=["id"])
    await mgr.upsert_rows("t", [{"id": i, "body": "x" * 2000} for i in range(1, 60)])
    yield mgr
    await mgr.close()


async def test_a_range_needs_one_key_with_a_list(wide):
    res = await wide.query("t", where={"id": [{"op": ">=", "value": 10},
                                              {"op": "<=", "value": 12}]})
    assert res["total"] == 3


async def test_a_plain_list_means_in(wide):
    res = await wide.query("t", where={"id": [4, 7, 9]})
    assert res["total"] == 3


def test_repeated_where_key_is_rescued_not_dropped():
    from captain_claw.tools.datastore import _parse_where
    where, note = _parse_where(
        '{"id": {"op": ">=", "value": 370}, "id": {"op": "<=", "value": 379}}')
    assert where["id"] == [{"op": ">=", "value": 370}, {"op": "<=", "value": 379}]
    assert "`id`" in note  # and the model is told why


async def test_the_rescued_filter_selects_the_range_it_asked_for(wide):
    from captain_claw.tools.datastore import _parse_where
    where, _ = _parse_where('{"id": {"op": ">=", "value": 10}, "id": {"op": "<=", "value": 19}}')
    res = await wide.query("t", where=where)
    assert res["total"] == 10          # not "everything up to 19"


def test_a_clean_where_is_left_alone():
    from captain_claw.tools.datastore import _parse_where
    where, note = _parse_where('{"status": "active"}')
    assert where == {"status": "active"} and note is None


async def test_one_result_cannot_eat_the_context(wide):
    res = await DatastoreTool()._query(wide, {"table": "t"})
    assert res.success
    assert len(res.content) < 50_000          # 59 rows × 2KB would be ~120k
    assert "did not fit" in res.content       # never a silent cap
    assert "`offset`" in res.content          # and it says exactly how to resume


async def test_a_small_result_gets_no_truncation_notice(wide):
    res = await DatastoreTool()._query(wide, {"table": "t", "columns": "id", "limit": 5})
    assert res.success and "NOT shown" not in res.content


# ── A read after a write is not a duplicate ──
# query → upsert → query-to-verify is the prescribed workflow, and the verify
# query is byte-identical to the first. Blocked as a duplicate, the model is
# told "the content has not changed" about a table it just wrote to — and goes
# hunting for "a fresh query pattern to avoid the duplicate guard" until the
# turn dies. Observed: 31 tool calls, all of them that hunt.

def test_datastore_counts_as_a_stateful_tool():
    import inspect
    from captain_claw import agent_tool_loop_mixin as m

    src = inspect.getsource(m)
    stateful = src[src.index("_STATEFUL_TOOLS = {"):]
    stateful = stateful[:stateful.index("}")]
    assert '"datastore"' in stateful


def test_a_write_clears_the_turn_s_datastore_read_history():
    import inspect
    from captain_claw import agent_tool_loop_mixin as m

    src = inspect.getsource(m)
    assert "_DATASTORE_WRITE_ACTIONS" in src
    # Only datastore keys are cleared — other tools keep their counters.
    assert 'k.startswith("datastore|")' in src
    for action in ("upsert", "update", "insert", "delete"):
        assert f'"{action}"' in src[src.index("_DATASTORE_WRITE_ACTIONS"):][:600]


def test_stateful_block_message_does_not_claim_the_data_is_unchanged():
    import inspect
    from captain_claw import agent_tool_loop_mixin as m

    src = inspect.getsource(m)
    block = src[src.index("elif _tool_lower in _STATEFUL_TOOLS:"):]
    # The message itself, not the comment above it (which quotes the old text).
    msg = block[block.index("dup_msg = ("):block.index("else:")]
    # Splice the adjacent string literals back together so an assertion isn't
    # at the mercy of where the source happens to wrap.
    flat = re.sub(r'"\s*f?"', "", " ".join(msg.split()))
    assert "nothing written in between" in flat
    assert "content has not changed" not in flat
    # And it names the failure mode we actually saw.
    assert "reword" in flat


async def test_a_batch_sized_read_is_not_truncated(tmp_path):
    """~10 wide rows is the common case and must arrive whole — a partial
    answer reads as a failed query and starts the rephrasing hunt."""
    mgr = DatastoreManager(db_path=tmp_path / "wide.db")
    cols = [{"name": "id", "type": "integer"}] + [
        {"name": f"c{i}", "type": "text"} for i in range(19)
    ]
    await mgr.create_table("t", cols, unique=["id"])
    await mgr.upsert_rows("t", [
        {"id": i, **{f"c{j}": "x" * 180 for j in range(19)}} for i in range(1, 11)
    ])
    res = await DatastoreTool()._query(mgr, {"table": "t"})
    assert res.success
    assert "did not fit" not in res.content        # all 10 rows survived
    await mgr.close()


async def test_truncation_gives_one_concrete_next_step(wide):
    res = await DatastoreTool()._query(wide, {"table": "t"})
    if "did not fit" in res.content:
        assert "the query was CORRECT" in res.content
        assert "`offset`" in res.content


# ── Every way a model reaches for a range ──
# Observed in the wild, each one a separate dead end at the time:
#   two "id" keys           → JSON drops one, filter silently widens
#   {"op": [">=", "<="]}    → "Unsupported operator: ['>=', '<=']"
#   BETWEEN                 → "Unsupported operator: BETWEEN"

async def test_paired_operators(wide):
    res = await wide.query("t", where={"id": {"op": [">=", "<="], "value": [10, 12]}})
    assert res["total"] == 3


async def test_between(wide):
    res = await wide.query("t", where={"id": {"op": "BETWEEN", "value": [10, 12]}})
    assert res["total"] == 3


async def test_not_between(wide):
    res = await wide.query("t", where={"id": {"op": "NOT BETWEEN", "value": [3, 59]}})
    assert res["total"] == 2          # ids 1 and 2


async def test_all_four_range_forms_agree(wide):
    forms = [
        {"id": {"op": "BETWEEN", "value": [10, 19]}},
        {"id": {"op": [">=", "<="], "value": [10, 19]}},
        {"id": [{"op": ">=", "value": 10}, {"op": "<=", "value": 19}]},
    ]
    totals = [(await wide.query("t", where=f))["total"] for f in forms]
    assert totals == [10, 10, 10]


async def test_mismatched_pairs_say_how_to_pair_them(wide):
    with pytest.raises(ValueError, match="pair up"):
        await wide.query("t", where={"id": {"op": [">=", "<="], "value": [10]}})


async def test_between_needs_two_bounds(wide):
    with pytest.raises(ValueError, match="exactly two values"):
        await wide.query("t", where={"id": {"op": "BETWEEN", "value": 10}})


async def test_unknown_operator_shows_the_range_forms(wide):
    with pytest.raises(ValueError) as e:
        await wide.query("t", where={"id": {"op": "~~", "value": 1}})
    assert "BETWEEN" in str(e.value) and "Allowed:" in str(e.value)
