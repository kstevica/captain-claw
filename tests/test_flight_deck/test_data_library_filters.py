"""Column filters on a Data library table's rows.

The table view filters any column — text, number, date or yes/no, the way
the view reads it — and as many columns as the person likes, every filter
narrowing the rows further: values picked from the column's own, a range,
a condition, the top or bottom N, above or below the average, repeats.
``_filter_where`` turns them into SQL; these run it against a small table
the way ``_read_rows`` does, and check what it refuses (a column the table
doesn't have, an operator the kind doesn't take, a number filter without a
number). ``_column_values`` lists what a column holds for the picker.
"""

from __future__ import annotations

import sqlite3

import pytest

from captain_claw.flight_deck import data_library as dl

NAMES = ["name", "country", "score", "amount", "seen", "active"]
ROWS = [
    ("Ana", "Netherlands", 7, "1 234,5", "2026-03-01", 1),
    ("Ivo", "Croatia", 3, "12%", "2026-05-20 10:30", 0),
    ("Mia", "netherlands", None, "", "2025-12-31", None),
    ("Leo", None, 10, "abc", None, 1),
    ("Eva", "Germany", 7.5, "99", "", "true"),
]


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.create_function("cf", 1, dl._casefold, deterministic=True)
    c.create_function("fnum", 1, dl._to_number, deterministic=True)
    c.execute("CREATE TABLE t (name TEXT, country TEXT, score REAL, amount TEXT, seen TEXT, active)")
    c.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)", ROWS)
    yield c
    c.close()


def names_where(c: sqlite3.Connection, *filters: dict) -> list[str]:
    where, args = dl._filter_where(list(filters), NAMES, "t")
    sql = "SELECT name FROM t" + (f" WHERE {' AND '.join(where)}" if where else "") + " ORDER BY rowid"
    return [r[0] for r in c.execute(sql, args)]


def test_text_filters_ignore_case(conn):
    assert names_where(conn, {"col": "country", "op": "contains", "value": "NETHER"}) == ["Ana", "Mia"]
    assert names_where(conn, {"col": "country", "op": "eq", "value": " croatia "}) == ["Ivo"]
    assert names_where(conn, {"col": "country", "op": "starts", "value": "ger"}) == ["Eva"]


def test_negative_text_filters_keep_empty_cells(conn):
    assert names_where(conn, {"col": "country", "op": "not_contains", "value": "nether"}) == ["Ivo", "Leo", "Eva"]
    assert names_where(conn, {"col": "country", "op": "ne", "value": "croatia"}) == ["Ana", "Mia", "Leo", "Eva"]


def test_like_wildcards_are_literal(conn):
    assert names_where(conn, {"col": "amount", "op": "contains", "value": "%"}) == ["Ivo"]
    assert names_where(conn, {"col": "amount", "op": "contains", "value": "_"}) == []


def test_number_filters_read_numbers_as_written(conn):
    assert names_where(conn, {"col": "score", "op": "gte", "value": "7", "kind": "number"}) == ["Ana", "Leo", "Eva"]
    assert names_where(conn, {"col": "score", "op": "between", "value": "8", "value2": "3", "kind": "number"}) == ["Ana", "Ivo", "Eva"]
    # "1 234,5" and "12%" are numbers; "abc" and "" aren't.
    assert names_where(conn, {"col": "amount", "op": "gt", "value": "50", "kind": "number"}) == ["Ana", "Eva"]
    assert names_where(conn, {"col": "amount", "op": "eq", "value": "12", "kind": "number"}) == ["Ivo"]
    assert names_where(conn, {"col": "amount", "op": "ne", "value": "99", "kind": "number"}) == ["Ana", "Ivo", "Mia", "Leo"]


def test_date_filters_compare_the_day(conn):
    assert names_where(conn, {"col": "seen", "op": "eq", "value": "2026-05-20", "kind": "date"}) == ["Ivo"]
    assert names_where(conn, {"col": "seen", "op": "lt", "value": "2026-04-01", "kind": "date"}) == ["Ana", "Mia"]
    assert names_where(conn, {"col": "seen", "op": "between", "value": "2026-12-01", "value2": "2026-01-01", "kind": "date"}) == ["Ana", "Ivo"]


def test_yes_no_and_empty(conn):
    assert names_where(conn, {"col": "active", "op": "yes", "kind": "boolean"}) == ["Ana", "Leo", "Eva"]
    assert names_where(conn, {"col": "active", "op": "no", "kind": "boolean"}) == ["Ivo"]
    assert names_where(conn, {"col": "active", "op": "empty", "kind": "boolean"}) == ["Mia"]
    assert names_where(conn, {"col": "seen", "op": "empty"}) == ["Leo", "Eva"]
    assert names_where(conn, {"col": "seen", "op": "not_empty"}) == ["Ana", "Ivo", "Mia"]


def test_filters_combine(conn):
    assert names_where(
        conn,
        {"col": "country", "op": "contains", "value": "nether"},
        {"col": "score", "op": "gt", "value": "5", "kind": "number"},
    ) == ["Ana"]


@pytest.mark.parametrize("bad", [
    {"col": "nope", "op": "contains", "value": "x"},
    {"col": "name", "op": "gt", "value": "x"},
    {"col": "score", "op": "gt", "value": "many", "kind": "number"},
    {"col": "seen", "op": "eq", "value": "May 20", "kind": "date"},
    {"col": "name", "op": "contains", "value": "  "},
    {"col": "name", "op": "contains", "value": "x", "kind": "colour"},
    "name contains x",
    {"col": "country", "op": "in", "values": []},
    {"col": "country", "op": "in", "values": "Norway"},
    {"col": "score", "op": "range", "kind": "number"},
    {"col": "score", "op": "range", "value": "lots", "kind": "number"},
    {"col": "seen", "op": "range", "value": "May", "kind": "date"},
    {"col": "score", "op": "top", "value": "0", "kind": "number"},
    {"col": "score", "op": "top", "value": "ten", "kind": "number"},
    {"col": "country", "op": "top", "value": "3"},
])
def test_bad_filters_are_refused(bad):
    with pytest.raises(ValueError):
        dl._filter_where([bad], NAMES, "t")


def test_to_number():
    assert dl._to_number("1,234.5") == 1234.5
    assert dl._to_number("1234,5") == 1234.5
    assert dl._to_number("-3") == -3
    assert dl._to_number("12 %") == 12
    assert dl._to_number(True) is None
    assert dl._to_number("1.2.3") is None
    assert dl._to_number("") is None
    assert dl._to_number(None) is None


def test_picked_values(conn):
    assert names_where(conn, {"col": "country", "op": "in", "values": ["netherlands", "Croatia"]}) == ["Ana", "Ivo", "Mia"]
    # "" stands for an empty cell.
    assert names_where(conn, {"col": "country", "op": "in", "values": ["Germany", ""]}) == ["Leo", "Eva"]
    assert names_where(conn, {"col": "country", "op": "not_in", "values": ["NETHERLANDS"]}) == ["Ivo", "Leo", "Eva"]
    assert names_where(conn, {"col": "country", "op": "not_in", "values": ["Germany", ""]}) == ["Ana", "Ivo", "Mia"]
    # Numbers as their text: what the picker lists.
    assert names_where(conn, {"col": "score", "op": "in", "values": ["7.0", "10.0"], "kind": "number"}) == ["Ana", "Leo"]


def test_ranges_take_one_end_or_both(conn):
    assert names_where(conn, {"col": "score", "op": "range", "value": "5", "kind": "number"}) == ["Ana", "Leo", "Eva"]
    assert names_where(conn, {"col": "score", "op": "range", "value2": "7", "kind": "number"}) == ["Ana", "Ivo"]
    assert names_where(conn, {"col": "score", "op": "range", "value": "8", "value2": "3", "kind": "number"}) == ["Ana", "Ivo", "Eva"]
    assert names_where(conn, {"col": "seen", "op": "range", "value": "2026-01-01", "kind": "date"}) == ["Ana", "Ivo"]
    assert names_where(conn, {"col": "seen", "op": "range", "value2": "2026-03-01", "kind": "date"}) == ["Ana", "Mia"]


def test_top_bottom_and_average(conn):
    # score: 7, 3, None, 10, 7.5 — the average is 6.875.
    assert names_where(conn, {"col": "score", "op": "top", "value": "2", "kind": "number"}) == ["Leo", "Eva"]
    assert names_where(conn, {"col": "score", "op": "bottom", "value": "1", "kind": "number"}) == ["Ivo"]
    # More than the column holds: every row with a number.
    assert names_where(conn, {"col": "score", "op": "top", "value": "50", "kind": "number"}) == ["Ana", "Ivo", "Leo", "Eva"]
    assert names_where(conn, {"col": "score", "op": "above_avg", "kind": "number"}) == ["Ana", "Leo", "Eva"]
    assert names_where(conn, {"col": "score", "op": "below_avg", "kind": "number"}) == ["Ivo"]


def test_repeats(conn):
    # "Netherlands" and "netherlands" are one value.
    assert names_where(conn, {"col": "country", "op": "dupes"}) == ["Ana", "Mia"]
    assert names_where(conn, {"col": "country", "op": "unique"}) == ["Ivo", "Eva"]
    assert names_where(conn, {"col": "name", "op": "ends", "value": "O"}) == ["Ivo", "Leo"]


def test_column_values(conn):
    got = dl._column_values(conn, "t", "country", "text", [], [])
    assert got["values"][0] == {"value": "Netherlands", "count": 2}
    assert {v["value"] for v in got["values"]} == {"Netherlands", "Croatia", "Germany"}
    assert (got["distinct"], got["empty"], got["total"]) == (3, 1, 5)
    assert [v["value"] for v in dl._column_values(conn, "t", "country", "text", [], [], search="CRO")["values"]] == ["Croatia"]
    # Among the rows other filters leave.
    where, args = dl._filter_where([{"col": "score", "op": "gte", "value": "7", "kind": "number"}], NAMES, "t")
    got = dl._column_values(conn, "t", "country", "text", where, args)
    assert (got["total"], got["empty"], got["distinct"]) == (3, 1, 2)
    nums = dl._column_values(conn, "t", "amount", "number", [], [])
    assert (nums["min"], nums["max"]) == (12, 1234.5)
    days = dl._column_values(conn, "t", "seen", "date", [], [])
    assert (days["min"], days["max"]) == ("2025-12-31", "2026-05-20")
