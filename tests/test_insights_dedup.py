"""Insight dedup: an exact repeat, or the same fact reworded, is not stored again.

``find_similar`` used to build its FTS MATCH from raw words, so a "." or a
"15:00" made the query fail and it found nothing: a fact arriving twice (setup
applied twice, or from the waiting file and live) landed twice. Facts that
differ in a number or a negation must still be stored apart.

Every test uses a tmp DB (nothing here may reach ~/.captain-claw).
"""

from __future__ import annotations

import pytest

WAREHOUSE = "The warehouse closes at 15:00 on Fridays."


@pytest.fixture
async def insights(tmp_path):
    from captain_claw.insights import InsightsManager

    mgr = InsightsManager(tmp_path / "insights.db")
    rows = [
        ("Always answer in Croatian unless asked otherwise", "preference"),
        ("Stevica prefers short WhatsApp replies", "preference"),
        ("The accountant is Biserka; invoices go to her by the 5th", "contact"),
        ("Munich hotel: Adler near the main station, booked for May", "fact"),
        ("The VC term sheet with Vesna Ventures caps the valuation at 12M", "decision"),
        ("Dentist appointments are on Friday mornings", "fact"),
        ("Quarterly VAT returns are filed by the accountant", "workflow"),
        ("Project Iskra ships its demo in November", "project"),
    ]
    for content, category in rows:
        assert await mgr.add(content=content, category=category)
    yield mgr
    await mgr.close()


async def _matching(mgr, text: str) -> list[dict]:
    return [r for r in await mgr.list_recent(100) if r["content"].lower() == text.lower()]


# ── exact repeats ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_same_fact_twice_is_stored_once(insights):
    first = await insights.add(content=WAREHOUSE, category="fact")
    assert first
    assert await insights.add(content=WAREHOUSE, category="fact") is None
    assert [r["id"] for r in await _matching(insights, WAREHOUSE)] == [first]
    assert await insights.count() == 9


@pytest.mark.asyncio
async def test_case_and_whitespace_do_not_make_a_new_fact(insights):
    first = await insights.add(content=WAREHOUSE, category="fact")
    again = "  the WAREHOUSE closes at\n15:00   on fridays. "
    assert await insights.add(content=again, category="fact") is None
    assert (await insights.find_exact(again))["id"] == first


@pytest.mark.asyncio
async def test_a_repeat_in_another_category_is_still_a_repeat(insights):
    assert await insights.add(content=WAREHOUSE, category="fact")
    assert await insights.add(content=WAREHOUSE, category="workflow") is None


@pytest.mark.asyncio
async def test_entity_key_still_supersedes_the_old_value(insights):
    old = await insights.add(content="Office opens at 8:00", category="fact", entity_key="office-hours")
    new = await insights.add(content="Office opens at 8:00", category="fact", entity_key="office-hours")
    assert new and new != old
    assert await insights.get(old) is None
    assert (await insights.get(new))["supersedes_id"] == old


@pytest.mark.asyncio
async def test_import_counts_a_repeat_as_deduped(insights):
    stats = await insights.import_items([
        {"content": WAREHOUSE, "category": "fact"},
        {"content": WAREHOUSE.upper(), "category": "fact"},
    ])
    assert stats["stored"] == 1 and stats["deduped"] == 1


# ── the same fact reworded ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_punctuation_no_longer_breaks_the_similarity_query(insights):
    first = await insights.add(content=WAREHOUSE, category="fact")
    assert [r["id"] for r in await insights.find_similar(WAREHOUSE)] == [first]
    odd = 'Tag files "urgent" (never *draft*) - content: final, AND/OR NOT^2'
    tagged = await insights.add(content=odd, category="workflow")
    assert [r["id"] for r in await insights.find_similar(odd)] == [tagged]


@pytest.mark.asyncio
async def test_a_reworded_fact_is_deduped(insights):
    assert await insights.add(content="Warehouse closes at 15:00 on Fridays!", category="fact")
    assert await insights.add(content=WAREHOUSE, category="fact") is None
    assert await insights.add(
        content="Invoices go to the accountant, Biserka, by the 5th.", category="contact") is None
    assert await insights.count() == 9


@pytest.mark.asyncio
async def test_different_numbers_are_a_different_fact(insights):
    short = await insights.add(content="Refunds are accepted within 14 days.", category="fact")
    long = await insights.add(content="Refunds are accepted within 30 days.", category="fact")
    assert short and long and short != long
    assert await insights.add(content="The warehouse closes at 16:00 on Fridays.", category="fact")
    assert await insights.add(content=WAREHOUSE, category="fact")


@pytest.mark.asyncio
async def test_a_negation_is_a_different_fact(insights):
    assert await insights.add(content="The client wants weekly reports.", category="preference")
    assert await insights.add(content="The client does not want weekly reports.", category="preference")
    assert await insights.add(content="The client doesn't want weekly reports", category="preference") is None
    assert await insights.add(content="Klijent nije htio tjedne izvještaje", category="fact")
    assert await insights.add(content="Klijent htio tjedne izvještaje", category="fact")


@pytest.mark.asyncio
async def test_a_fact_with_more_detail_or_shared_words_is_kept(insights):
    assert await insights.add(content="User prefers dark mode", category="preference")
    assert await insights.add(content="User prefers dark mode except when presenting slides",
                              category="preference")
    assert await insights.add(content="Team standups are on Friday mornings", category="fact")


@pytest.mark.asyncio
async def test_import_stages_a_reworded_decision_for_review(insights):
    stats = await insights.import_items(
        [{"content": "Term sheet with Vesna Ventures caps the valuation at 12M.", "category": "decision"}],
        stage_conflicts=True,
    )
    assert stats["staged"] == 1 and stats["stored"] == 0
    [pending] = await insights.list_pending()
    assert pending["conflict_reason"] == "fts similarity"
