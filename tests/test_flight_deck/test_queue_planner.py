"""Queue Task Planner — phase 1: facts, expansion, guards.

The design this pins (docs/queue-task-planner-plan.md): the model returns ONE
template plus a list of ranges, and Flight Deck expands them. A model asked to
write twenty-five near-identical messages paraphrases and drops clauses; the
one it drops ("never do +1 on the id!") corrupts a table. Expansion in Python
means every message is byte-identical except its range.
"""

import pytest
from fastapi import HTTPException

from captain_claw.flight_deck.queue_planner import (
    MAX_TASKS_CEILING,
    build_user_prompt,
    clamp_batches,
    expand_template,
    facts_summary,
    parse_plan,
    unresolved_placeholders,
)

# The real thing, trimmed — every message must carry all of it, every time.
TEMPLATE = (
    "always start fresh, data you need to enrich is not in your memory. "
    "enrich data in table fund_portfolio, _id from {from} to {to}. "
    "Use only these _ids, do not proceed automatically to the next batch. "
    "_id and id are identical, never do +1 on the id!"
)


def test_expansion_differs_only_in_the_range():
    """The whole point: no clause can drift between batch 1 and batch 3."""
    msgs = expand_template(TEMPLATE, [
        {"from": 241, "to": 250}, {"from": 251, "to": 260}, {"from": 261, "to": 270},
    ])
    assert len(msgs) == 3
    assert "_id from 241 to 250" in msgs[0]
    assert "_id from 251 to 260" in msgs[1]
    # Strip the ranges and the three messages are the same string.
    stripped = {m.replace("241", "").replace("250", "")
                 .replace("251", "").replace("260", "")
                 .replace("261", "").replace("270", "") for m in msgs}
    assert len(stripped) == 1
    for m in msgs:
        assert "never do +1 on the id!" in m


@pytest.mark.parametrize("spelling", ["{from}/{to}", "{start}/{end}", "{from_id}/{to_id}"])
def test_the_obvious_placeholder_spellings_all_work(spelling):
    """The model writes the template; rejecting {start} costs a retry for nothing."""
    a, b = spelling.split("/")
    msgs = expand_template(f"rows {a} to {b}", [{"from": 1, "to": 10}])
    assert msgs == ["rows 1 to 10"]


def test_batch_can_carry_extra_fields():
    msgs = expand_template("enrich {table} rows {from}-{to}",
                           [{"from": 1, "to": 5, "table": "fund_portfolio"}])
    assert msgs == ["enrich fund_portfolio rows 1-5"]


def test_index_and_total_are_available():
    msgs = expand_template("batch {index} of {total}: {from}-{to}",
                           [{"from": 1, "to": 5}, {"from": 6, "to": 10}])
    assert msgs == ["batch 1 of 2: 1-5", "batch 2 of 2: 6-10"]


def test_unfilled_placeholders_are_reported_not_shipped():
    """`_id from {from} to {to}` reaching the agent is a task that cannot work."""
    msgs = expand_template("rows {from} to {to} of {table}", [{"from": 1, "to": 5}])
    assert unresolved_placeholders(msgs) == ["{table}"]


def test_a_clean_expansion_reports_nothing():
    assert unresolved_placeholders(["rows 1 to 5"]) == []


# ── Guards ──

def test_batches_are_capped_and_say_so():
    batches = [{"from": i, "to": i + 9} for i in range(1, 1000, 10)]
    kept, note = clamp_batches(batches, 50)
    assert len(kept) == 50
    assert "dropped" in note and "50" in note      # never a silent cap


def test_a_plan_within_the_cap_is_untouched():
    batches = [{"from": 1, "to": 10}]
    kept, note = clamp_batches(batches, 50)
    assert kept == batches and note is None


def test_the_ceiling_is_far_below_a_whole_table():
    assert MAX_TASKS_CEILING <= 200


# ── Parsing what the model sends back ──

def test_parse_plain_json():
    assert parse_plan('{"template": "x", "batches": []}')["template"] == "x"


def test_parse_fenced_json():
    raw = 'Here is the plan:\n```json\n{"template": "x", "batches": [{"from": 1}]}\n```\n'
    assert parse_plan(raw)["batches"] == [{"from": 1}]


def test_parse_json_with_prose_around_it():
    assert parse_plan('Sure!\n{"template": "x"}\nHope that helps.')["template"] == "x"


def test_unparseable_reply_is_a_502_not_a_crash():
    with pytest.raises(HTTPException) as e:
        parse_plan("I'd be happy to help with that!")
    assert e.value.status_code == 502


# ── Facts ──

FACTS = {
    "tables": [{"name": "fund_portfolio", "rows": 1818,
                "columns": ["id", "portfolio_company_name", "company_description"]}],
    "table": "fund_portfolio", "key": "_id",
    "key_min": 1, "key_max": 1818, "key_count": 1818,
}


def test_facts_summary_states_the_real_bounds():
    """Ranges must be facts: a plan for 1..500 against a table ending at 318
    is 18 tasks that can only fail."""
    out = facts_summary(FACTS)
    assert "fund_portfolio (1818 rows)" in out
    assert "min=1, max=1818" in out
    assert "MUST stay inside this range" in out


def test_facts_summary_admits_when_it_could_not_read():
    out = facts_summary({"error": "connection refused", "tables": []})
    assert "do not invent ranges" in out


def test_prompt_carries_facts_batch_size_and_shape():
    p = build_user_prompt("enrich the portfolio", FACTS, 10, "_id")
    assert "enrich the portfolio" in p
    assert "min=1, max=1818" in p
    assert "Batch size: 10" in p
    assert '"batches"' in p and '"template"' in p


def test_prompt_includes_file_notes_only_when_present():
    assert "Attached files" not in build_user_prompt("x", FACTS, 10, "_id")
    assert "Attached files" in build_user_prompt("x", FACTS, 10, "_id", "ids.xlsx: 40 rows")


# ── Phase 2: re-expansion for the review UI ──
# Editing the template must re-render every task WITHOUT another LLM call, and
# without a second copy of the expansion in TypeScript that could drift from
# the one that produced the plan.

async def test_expand_endpoint_rerenders_an_edited_template():
    from captain_claw.flight_deck.queue_planner import ExpandRequest, expand_plan

    body = ExpandRequest(
        template="enrich rows {from}-{to}. ALWAYS write in English.",
        batches=[{"from": 1, "to": 10}, {"from": 11, "to": 20}],
    )
    out = await expand_plan(body, user={"id": 1})
    assert out["messages"] == [
        "enrich rows 1-10. ALWAYS write in English.",
        "enrich rows 11-20. ALWAYS write in English.",
    ]
    assert out["warnings"] == []


async def test_expand_endpoint_applies_the_same_cap():
    from captain_claw.flight_deck.queue_planner import ExpandRequest, expand_plan

    body = ExpandRequest(template="rows {from}-{to}",
                         batches=[{"from": i, "to": i} for i in range(100)],
                         max_tasks=5)
    out = await expand_plan(body, user={"id": 1})
    assert len(out["messages"]) == 5
    assert any("dropped" in w for w in out["warnings"])


async def test_expand_endpoint_reports_unfilled_placeholders():
    from captain_claw.flight_deck.queue_planner import ExpandRequest, expand_plan

    out = await expand_plan(
        ExpandRequest(template="rows {from}-{to} of {table}", batches=[{"from": 1, "to": 5}]),
        user={"id": 1})
    assert any("{table}" in w for w in out["warnings"])


async def test_expand_endpoint_needs_a_template():
    from captain_claw.flight_deck.queue_planner import ExpandRequest, expand_plan

    with pytest.raises(HTTPException) as e:
        await expand_plan(ExpandRequest(template="  ", batches=[]), user={"id": 1})
    assert e.value.status_code == 400


# ── Phase 3: attachments ──
# An attached file serves two purposes, and conflating them is how this goes
# wrong: the TASKS need its path (the agent opens it at run time), the PLANNER
# needs a peek inside (or it guesses at the ranges we just grounded).

def test_csv_preview_is_the_first_rows():
    from captain_claw.flight_deck.queue_planner import preview_file

    blob = ("id,name\n" + "\n".join(f"{i},Company {i}" for i in range(1, 200))).encode()
    out = preview_file("ids.csv", blob)
    assert "id,name" in out and "1,Company 1" in out
    assert "150,Company 150" not in out          # capped, not the whole file
    assert len(out) <= 3000


def test_xlsx_preview_uses_the_document_extractor():
    """One understanding of what an .xlsx is, not a second one here."""
    import io, zipfile
    from captain_claw.flight_deck.queue_planner import preview_file

    # A minimal but real xlsx: one sheet, two cells.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
                   'package/2006/content-types"/>')
        z.writestr("xl/worksheets/sheet1.xml",
                   '<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/'
                   'spreadsheetml/2006/main"><sheetData><row r="1">'
                   '<c r="A1" t="inlineStr"><is><t>portfolio_id</t></is></c>'
                   '<c r="B1"><v>241</v></c></row></sheetData></worksheet>')
    out = preview_file("ids.xlsx", buf.getvalue())
    # Either it extracted the cells, or it degraded honestly — never a crash.
    assert isinstance(out, str) and out


def test_unreadable_attachment_degrades_to_a_note():
    from captain_claw.flight_deck.queue_planner import preview_file

    out = preview_file("photo.heic", b"\x00\x01\x02\x03" * 100)
    assert "binary or unreadable" in out and "400" in out


def test_a_broken_file_never_fails_the_plan():
    """A preview is a nicety; the path is the part the tasks need."""
    from captain_claw.flight_deck.queue_planner import preview_file

    assert preview_file("truncated.xlsx", b"PK\x03\x04 not really a zip")


def test_prompt_tells_the_planner_a_path_must_reach_the_task():
    from captain_claw.flight_deck.queue_planner import _SYSTEM

    assert "cannot see your preview" in _SYSTEM


async def test_file_notes_lead_with_the_path(monkeypatch):
    from captain_claw.flight_deck import queue_planner as qp

    async def fake_fetch(host, port, auth, path):
        return b"id,name\n241,Xolo\n242,DrHouse\n"

    monkeypatch.setattr(qp, "_fetch_agent_file", fake_fetch)
    notes = await qp.build_file_notes("h", 1, "t", [
        {"path": "saved/uploads/ids.csv", "filename": "ids.csv"},
    ])
    assert "`saved/uploads/ids.csv`" in notes      # the agent opens THIS
    assert "241,Xolo" in notes                     # the planner sees THIS


async def test_a_file_that_cannot_be_read_back_still_contributes_its_path(monkeypatch):
    from captain_claw.flight_deck import queue_planner as qp

    async def fake_fetch(host, port, auth, path):
        return None

    monkeypatch.setattr(qp, "_fetch_agent_file", fake_fetch)
    notes = await qp.build_file_notes("h", 1, "t", [{"path": "saved/x.bin", "filename": "x.bin"}])
    assert "saved/x.bin" in notes and "could not be read back" in notes


# ── Phase 4: continuing where a plan stopped ──
# A 1,818-row table is not one sitting. Continuing needs no model: the template
# was written and approved once, and "the next 25 batches of 10 from 491" is
# arithmetic. A continuation that called the model again would also risk a
# DIFFERENT template for the second half of one job.

def test_batches_run_consecutively_from_the_start():
    from captain_claw.flight_deck.queue_planner import make_batches

    batches, note = make_batches(491, 10, 3)
    assert batches == [{"from": 491, "to": 500}, {"from": 501, "to": 510},
                       {"from": 511, "to": 520}]
    assert note is None


def test_batches_never_overlap_or_skip():
    from captain_claw.flight_deck.queue_planner import make_batches

    batches, _ = make_batches(1, 7, 20)
    for a, b in zip(batches, batches[1:]):
        assert b["from"] == a["to"] + 1          # the bug this whole feature exists to avoid


def test_continuation_stops_at_the_real_end_of_the_table():
    """Planning past the last row is tasks that can only fail."""
    from captain_claw.flight_deck.queue_planner import make_batches

    batches, note = make_batches(1800, 10, 10, key_max=1818)
    assert batches == [{"from": 1800, "to": 1809}, {"from": 1810, "to": 1818}]
    assert "Stopped at 1818" in note


def test_starting_past_the_end_makes_nothing():
    from captain_claw.flight_deck.queue_planner import make_batches

    batches, note = make_batches(2000, 10, 5, key_max=1818)
    assert batches == [] and "Stopped at 1818" in note


def test_batch_size_is_clamped_like_everywhere_else():
    from captain_claw.flight_deck.queue_planner import make_batches

    batches, _ = make_batches(1, 9999, 1)
    assert batches[0]["to"] - batches[0]["from"] + 1 <= 50


async def test_continue_endpoint_reuses_the_template_verbatim():
    from captain_claw.flight_deck.queue_planner import ContinueRequest, continue_plan

    tpl = "enrich _id from {from} to {to}. never do +1 on the id!"
    out = await continue_plan(
        ContinueRequest(template=tpl, start=491, batch_size=10, count=2), user={"id": 1})
    assert out["template"] == tpl                 # not re-written
    assert out["messages"][0] == "enrich _id from 491 to 500. never do +1 on the id!"
    assert out["messages"][1] == "enrich _id from 501 to 510. never do +1 on the id!"


async def test_continue_endpoint_says_when_there_is_nothing_left(monkeypatch):
    from captain_claw.flight_deck import queue_planner as qp

    async def facts(host, port, auth, table, key):
        return {"key_max": 100, "table": "t", "tables": []}

    monkeypatch.setattr(qp, "gather_datastore_facts", facts)
    out = await qp.continue_plan(
        qp.ContinueRequest(template="rows {from}-{to}", start=200, count=5, port=1234),
        user={"id": 1})
    assert out["messages"] == []
    assert any("past the last row" in w for w in out["warnings"])


async def test_continue_endpoint_needs_a_template():
    from captain_claw.flight_deck.queue_planner import ContinueRequest, continue_plan

    with pytest.raises(HTTPException) as e:
        await continue_plan(ContinueRequest(template="", start=1), user={"id": 1})
    assert e.value.status_code == 400
