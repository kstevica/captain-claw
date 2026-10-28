"""Intuition dedup: the same thought, repeated or reworded, is not stored again.

``NervousSystemManager.find_similar`` built its FTS MATCH from raw words, so a
"." or a "15:00" made the query fail and it found nothing: every dream stored
its repeats again. Thoughts that differ in a number or a negation must still
be stored apart.

Every test uses a tmp DB, default config and a stub metrics store (nothing
here may reach ~/.captain-claw).
"""

from __future__ import annotations

import pytest

INVOICES = "User tends to send invoices on Mondays around 9:30."


class _Metrics:
    async def record_event(self, *args, **kwargs) -> None:
        pass


@pytest.fixture
async def ns(tmp_path, monkeypatch):
    from captain_claw import cognitive_metrics, nervous_system
    from captain_claw.config import Config

    monkeypatch.setattr(nervous_system, "get_config", Config)
    monkeypatch.setattr(cognitive_metrics, "get_cognitive_metrics_manager", _Metrics)
    mgr = nervous_system.NervousSystemManager(tmp_path / "intuitions.db")
    rows = [
        (INVOICES, "pattern"),
        ("Vesna Ventures and Biserka both ask about the Q3 budget.", "connection"),
        ("The Munich trip may clash with the dentist appointment.", "hypothesis"),
        ("Short WhatsApp replies get faster answers from clients.", "association"),
    ]
    for content, thread_type in rows:
        assert await mgr.add(content=content, thread_type=thread_type)
    yield mgr
    await mgr.close()


@pytest.mark.asyncio
async def test_the_same_thought_twice_is_stored_once(ns):
    assert [r["content"] for r in await ns.find_similar(INVOICES)] == [INVOICES]
    assert await ns.add(content=INVOICES, thread_type="pattern") is None
    assert await ns.add(content=INVOICES.upper(), thread_type="association") is None
    assert await ns.count() == 4


@pytest.mark.asyncio
async def test_a_reworded_thought_is_deduped(ns):
    assert await ns.add(content="Around 9:30 on Mondays, the user tends to send invoices!",
                        thread_type="pattern") is None
    assert await ns.add(content="Short WhatsApp replies get faster answers from clients",
                        thread_type="pattern") is None
    assert await ns.count() == 4


@pytest.mark.asyncio
async def test_a_different_number_or_a_negation_is_a_different_thought(ns):
    assert await ns.add(content="Vesna Ventures and Biserka both ask about the Q4 budget.",
                        thread_type="connection")
    assert await ns.add(content="User tends to send invoices on Mondays around 10:30.",
                        thread_type="pattern")
    assert await ns.add(content="Short WhatsApp replies don't get faster answers from clients.",
                        thread_type="association")
    assert await ns.count() == 7


@pytest.mark.asyncio
async def test_a_thought_with_more_detail_or_shared_words_is_kept(ns):
    assert await ns.add(content="The Munich trip may clash with the dentist appointment and the board call.",
                        thread_type="hypothesis")
    assert await ns.add(content="The Zagreb trip may clash with the board call.", thread_type="hypothesis")


@pytest.mark.asyncio
async def test_fts_special_characters_do_not_break_the_query(ns):
    odd = 'Tagging "urgent" (never *draft*) - content: final, AND/OR NOT^2 helps triage'
    assert await ns.add(content=odd, thread_type="pattern")
    assert [r["content"] for r in await ns.find_similar(odd)] == [odd]
