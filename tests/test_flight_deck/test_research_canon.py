"""Increment 7: the pure canon continuity verifier + patch application."""

import pytest

from captain_claw.flight_deck import research_canon as rc


def test_fixed_fact_conflict():
    canon = {"entities": [
        {"name": "Ana", "fixed": {"age": "34"}, "quote": "Ana was 34"},
        {"name": "Ana", "fixed": {"age": "41"}, "quote": "Ana, 41,"},
    ], "scenes": [], "amounts": [], "tallies": [], "alibis": [], "dated_facts": []}
    f = rc.verify(canon)
    assert any(x["kind"] == "fixed_fact" for x in f)


def test_whereabouts_and_duplicate_scene():
    canon = {"entities": [], "amounts": [], "tallies": [], "alibis": [], "dated_facts": [],
             "scenes": [
                 {"id": "sc1", "order": 1, "day": "Fri", "hour": "9pm", "place": "archive",
                  "present": [{"name": "Ana", "whereabouts": "archive"},
                              {"name": "Ana", "whereabouts": "the docks"}]},
                 {"id": "sc2", "order": 2, "day": "Fri", "hour": "9pm", "place": "archive",
                  "present": [{"name": "Ana", "whereabouts": "archive"}]},
                 {"id": "sc3", "order": 3, "day": "Fri", "hour": "9pm", "place": "archive",
                  "present": [{"name": "Ana", "whereabouts": "archive"}]},
             ]}
    f = rc.verify(canon)
    kinds = {x["kind"] for x in f}
    assert "whereabouts" in kinds       # Ana in two places in sc1
    assert "duplicate_scene" in kinds   # sc2 and sc3 identical


def test_alibi_amount_tally_chronology():
    canon = {"entities": [], "scenes": [],
             "alibis": [{"suspect": "Bob", "alibi": "home"}, {"suspect": "Bob", "alibi": "the pub"}],
             "amounts": [{"label": "bribe", "paid": "12000", "attempted": "10000"}],
             "tallies": [{"claim": "four suspects", "count": 4, "items": ["a", "b", "c"]}],
             "dated_facts": [{"scene_year": "1990", "date_cited": "2001", "fact": "the merger"}]}
    kinds = {x["kind"] for x in rc.verify(canon)}
    assert {"alibi", "amount", "tally", "chronology"} <= kinds


def test_clean_canon_no_findings():
    canon = {"entities": [{"name": "Ana", "fixed": {"age": "34"}}],
             "scenes": [{"id": "s1", "present": [{"name": "Ana", "whereabouts": "home"}]}],
             "amounts": [{"label": "x", "paid": "5", "attempted": "10"}],
             "tallies": [{"claim": "three", "count": 3, "items": ["a", "b", "c"]}],
             "alibis": [{"suspect": "Bob", "alibi": "home"}],
             "dated_facts": [{"scene_year": "2001", "date_cited": "1999", "fact": "x"}]}
    assert rc.verify(canon) == []


def test_apply_patches():
    text = "Ana was 34 years old. Later, Ana, 41, walked in."
    patches = [{"find": "Ana, 41,", "replace": "Ana, 34,"}]
    out, applied, unapplied = rc.apply_patches(text, patches)
    assert applied == 1 and unapplied == []
    assert "41" not in out
    # a non-matching patch is reported unapplied
    out2, applied2, un2 = rc.apply_patches(text, [{"find": "NOPE", "replace": "x"}])
    assert applied2 == 0 and len(un2) == 1


def test_chunk_text_on_headings():
    text = "# Chapter One\n\n" + "a" * 20000 + "\n\n# Chapter Two\n\n" + "b" * 20000
    chunks = rc.chunk_text(text, max_chars=24000)
    assert len(chunks) == 2
    assert chunks[0].startswith("# Chapter One")
    assert chunks[1].startswith("# Chapter Two")


def test_parse_canon_and_patches_tolerant():
    assert rc.parse_canon("not json") == {k: [] for k in rc._LIST_KEYS}
    assert rc.parse_patches("garbage") == []
    good = '[{"find": "a", "replace": "b"}]'
    assert rc.parse_patches(good) == [{"find": "a", "replace": "b"}]


@pytest.mark.asyncio
async def test_run_check_extract_verify_patch():
    # scripted extractor: chunk yields a fixed_fact conflict; reviser patches it.
    extract_calls = {"n": 0}

    async def extract_fn(prompt):
        extract_calls["n"] += 1
        # first pass sees the conflict; after the patch the age is consistent
        if "Ana, 41," in prompt.split("TEXT:", 1)[-1]:
            return ('{"entities": [{"name": "Ana", "fixed": {"age": "34"}, "quote": "Ana was 34"},'
                    '{"name": "Ana", "fixed": {"age": "41"}, "quote": "Ana, 41,"}],'
                    '"scenes": [], "amounts": [], "tallies": [], "alibis": [], "dated_facts": []}')
        return '{"entities": [{"name": "Ana", "fixed": {"age": "34"}}], "scenes": [], "amounts": [], "tallies": [], "alibis": [], "dated_facts": []}'

    async def revise_fn(prompt):
        return '[{"find": "Ana, 41,", "replace": "Ana, 34,"}]'

    text = "# Ch1\n\nAna was 34 years old.\n\n# Ch2\n\nAna, 41, walked in."
    res = await rc.run_check(text, extract_fn=extract_fn, revise_fn=revise_fn)
    assert res["revised"] is True
    assert "41" not in res["text"]
    assert res["findings"] == []
    assert res["patched"] == 1
