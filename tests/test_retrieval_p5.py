"""Context engine P5: shared retrieval scoring and query-relevant insights.

Every test uses a tmp DB (nothing here may reach ~/.captain-claw).
"""

from __future__ import annotations

import pytest

from captain_claw import retrieval


def test_terms_stems_and_match_expression():
    terms = retrieval.query_terms("Možeš li mi poslati putovanja za Split i invoices, molim?")
    assert terms == ["poslati", "putovanja", "split", "invoices"]
    assert retrieval.fts_match(["split", "putovanja", "art"]) == '"split"* OR "putovan"* OR "art"'
    assert retrieval.fts_match(["bise"], prefix_all=True) == '"bise"*'
    assert retrieval.matched_terms(["putovanja", "hotel"], "Putovanje u Split") == ["putovanja"]


def test_scores():
    assert retrieval.bm25_relevance(-4.0) == 0.5
    assert retrieval.bm25_relevance(0.0) == 0.0
    fused = retrieval.rrf([["a", "b"], ["b", "c"]])
    assert max(fused, key=fused.get) == "b"
    assert retrieval.decayed(1.0, 30, 30) == 0.5
    assert retrieval.decayed(1.0, 30, 0) == 1.0


@pytest.fixture
async def insights(tmp_path):
    from captain_claw.insights import InsightsManager

    mgr = InsightsManager(tmp_path / "insights.db")
    rows = [
        ("Always answer in Croatian unless asked otherwise", "preference", 10),
        ("Stevica prefers short WhatsApp replies", "preference", 9),
        ("The accountant is Biserka; invoices go to her by the 5th", "contact", 6),
        ("Munich hotel: Adler near the main station, booked for May", "fact", 5),
        ("The VC term sheet with Vesna Ventures caps the valuation at 12M", "decision", 7),
        ("Dentist appointments are on Friday mornings", "fact", 4),
    ]
    for content, category, importance in rows:
        await mgr.add(content=content, category=category, importance=importance)
    yield mgr
    await mgr.close()


@pytest.mark.asyncio
async def test_insights_follow_the_turn_not_a_fixed_top_list(insights):
    picked = await insights.relevant_for_context("which hotel did we book in Munich?", limit=5, core=2)
    contents = [i["content"] for i in picked]
    assert contents[:2] == ["Always answer in Croatian unless asked otherwise",
                            "Stevica prefers short WhatsApp replies"]       # core rules stay
    assert "Munich hotel: Adler near the main station, booked for May" in contents
    assert not any("Dentist" in c or "Vesna" in c for c in contents)        # unrelated: out


@pytest.mark.asyncio
async def test_a_turn_that_matches_nothing_gets_only_the_core(insights):
    picked = await insights.relevant_for_context("write a poem about autumn leaves", limit=5, core=2)
    assert len(picked) == 2


@pytest.mark.asyncio
async def test_one_word_counts_only_when_it_is_rare(insights):
    people = ["Ana", "Marko", "Ivana", "Petra", "Luka", "Josip", "Maja", "Tomislav", "Sara", "Filip",
              "Nina", "Davor"]
    subjects = ["budget", "hiring", "roadmap", "pricing", "legal review", "marketing", "office move",
                "security audit", "partnership", "onboarding", "quarterly goals", "vendor contracts"]
    for person, subject in zip(people, subjects):
        await insights.add(content=f"Meeting with {person} covered {subject}", category="fact", importance=3)
    assert await insights._term_rows("meeting") >= 10
    def matches(picked):          # rules of importance 8+ always ride; look past them
        return [i["content"] for i in picked if i["category"] not in ("preference", "feedback")]

    # "meeting" is in half the store: one loose word, not a match.
    assert matches(await insights.relevant_for_context(
        "meeting about the weather forecast", limit=8, core=0)) == []
    # "station" is in one insight: a rare word, a match.
    assert matches(await insights.relevant_for_context(
        "what about the station and the forecast", limit=8, core=0)) == [
        "Munich hotel: Adler near the main station, booked for May"]


@pytest.mark.asyncio
async def test_rules_match_on_when_they_apply_and_important_ones_always_show(tmp_path):
    from captain_claw.insights import InsightsManager

    mgr = InsightsManager(tmp_path / "rules.db")
    try:
        for n in range(3):
            await mgr.add(content=f"Top fact {n}", category="fact", importance=10)
        await mgr.add(content="Never create email drafts unless asked", category="feedback", importance=9,
                      how_to_apply="When processing the inbox or replying to mail")
        await mgr.add(content="Prefer metric units", category="preference", importance=6,
                      how_to_apply="Distances, weights and recipes")
        await mgr.add(content="Old board meeting time", category="deadline", importance=10,
                      expires_at="2020-01-01T00:00:00+00:00")
        picked = [i["content"] for i in await mgr.relevant_for_context(
            "write a poem", limit=8, core=3)]
        assert "Never create email drafts unless asked" in picked      # a rule of 8+, always
        assert "Old board meeting time" not in picked                  # expired
        picked = [i["content"] for i in await mgr.relevant_for_context(
            "convert the recipe weights for me", limit=8, core=3)]
        assert "Prefer metric units" in picked                          # matched on how_to_apply
    finally:
        await mgr.close()


def test_semantic_memory_keeps_paths_and_ids():
    from captain_claw.semantic_memory import _build_fts_query

    assert '"a890fa40"' in _build_fts_query("what changed in commit a890fa40?")
    query = _build_fts_query("summarize reports/q3/budget_review.md")
    assert '"repor"*' in query and "budget" in query


# ── In the agent ───────────────────────────────────────────────────────


class _Words:
    provider = "openai"
    model = "stub"

    def count_tokens(self, text):
        return len(str(text or "").split()) or 1


def _cfg(**context):
    from captain_claw.config import get_config, set_config

    old = get_config().model_copy(deep=True)
    cfg = old.model_copy(deep=True)
    for key, value in context.items():
        setattr(cfg.context, key, value)
    set_config(cfg)
    return old


def test_capped_allocator_cuts_a_large_note_to_its_share():
    from captain_claw.agent import Agent
    from captain_claw.config import set_config

    agent = Agent(provider=_Words())
    big = "\n".join(f"memory line {n} " + "word " * 8 for n in range(200))
    notes = [("memory_context", big), ("todo_context", "buy milk"), ("fleet_changes", "x joined")]
    old = _cfg(notes_allocator="trim")
    try:
        trimmed = agent._fit_context_notes(notes, 1000, "lead")
        assert [k for k, _ in trimmed] == ["todo_context", "fleet_changes"]   # the big note is gone
        _cfg(notes_allocator="capped")
        capped = agent._fit_context_notes(notes, 1000, "lead")
        assert [k for k, _ in capped] == ["memory_context", "todo_context", "fleet_changes"]
        assert agent._count_tokens(capped[0][1]) <= 300
        assert capped[0][1].endswith("share of the context]")
    finally:
        set_config(old)


@pytest.mark.asyncio
async def test_the_insights_note_follows_the_turn(insights, monkeypatch):
    from captain_claw.agent import Agent

    agent = Agent(provider=_Words())
    monkeypatch.setattr("captain_claw.insights.get_insights_manager", lambda: insights)
    await agent._refresh_insights_context_cache(query="which hotel did we book in Munich?")
    note = agent._build_insights_context_note()
    assert "Munich hotel" in note and "Dentist" not in note
    await agent._refresh_insights_context_cache()                 # session load: the top list
    assert "Dentist" in agent._build_insights_context_note()


def test_capped_allocator_leaves_notes_alone_when_they_fit_and_keeps_items_whole():
    from captain_claw.agent import Agent
    from captain_claw.config import set_config

    agent = Agent(provider=_Words())
    rules = "\n".join(
        f"- [feedback] rule {n} " + "word " * 10 + f"\n    Why: reason {n}\n    How to apply: when {n}"
        for n in range(30))
    fleet = "Fleet changes: " + "agent joined " * 150
    old = _cfg(notes_allocator="capped")
    try:
        small = [("todo_context", "buy milk"), ("fleet_changes", "x joined")]
        assert agent._fit_context_notes(small, 1000, "lead") == small            # all fit: untouched
        fitted = dict(agent._fit_context_notes(
            [("insights_context", rules), ("fleet_changes", fleet)], 600, "lead"))
        assert "fleet_changes" in fitted                                         # cut, not dropped
        kept_rules = [b for b in fitted["insights_context"].split("\n- ") if "rule" in b]
        assert kept_rules and all("How to apply" in b for b in kept_rules)        # whole items
    finally:
        set_config(old)
