"""Story Integrity P0 — the six deterministic passes over the story-state store."""

import pytest

from captain_claw.flight_deck import story_state as ss


# ── Pass A: chronology ────────────────────────────────────────────────

def test_future_evidence_cited_before_created():
    store = {"timeline": [
        {"id": "e1", "order": 1, "cites_evidence": ["dna_report"], "quote": "she read the DNA report"},
        {"id": "e2", "order": 5, "evidence_created": ["dna_report"], "quote": "the lab produced the DNA report"},
    ]}
    f = ss.check_chronology(store)
    assert any(x["kind"] == "future_evidence" and x["severity"] == "hard" for x in f)


def test_two_places_at_once():
    store = {"timeline": [
        {"id": "a", "order": 3, "location": "the archive", "participants": ["Ana"]},
        {"id": "b", "order": 3, "location": "the docks", "participants": ["Ana"]},
    ]}
    f = ss.check_chronology(store)
    assert any(x["kind"] == "two_places" and x["severity"] == "hard" for x in f)


def test_chronology_clean():
    store = {"timeline": [
        {"id": "e1", "order": 5, "cites_evidence": ["dna_report"]},
        {"id": "e2", "order": 1, "evidence_created": ["dna_report"]},
        {"id": "a", "order": 2, "location": "home", "participants": ["Ana"]},
    ]}
    assert ss.check_chronology(store) == []


# ── Pass B: knowledge ─────────────────────────────────────────────────

def test_acts_on_unknown_fact():
    store = {
        "characters": [{"name": "Bob", "knows": [{"fact": "the safe code", "learned_order": 8}]}],
        "timeline": [{"id": "z", "order": 3, "actor": "Bob", "uses_facts": ["the safe code"]}],
    }
    f = ss.check_knowledge(store)
    assert any(x["kind"] == "knowledge_gap" and x["severity"] == "hard" for x in f)


def test_knowledge_ok_when_learned_earlier():
    store = {
        "characters": [{"name": "Bob", "knows": [{"fact": "the safe code", "learned_order": 2}]}],
        "timeline": [{"id": "z", "order": 3, "actor": "Bob", "uses_facts": ["the safe code"]}],
    }
    assert ss.check_knowledge(store) == []


def test_knowledge_failsafe_on_unknown_fact():
    # actor uses a fact never recorded in knows[] → no finding (fail-safe)
    store = {"characters": [{"name": "Bob", "knows": []}],
             "timeline": [{"id": "z", "order": 3, "actor": "Bob", "uses_facts": ["something"]}]}
    assert ss.check_knowledge(store) == []


# ── Pass D: provenance ────────────────────────────────────────────────

def test_two_disappearance_histories():
    store = {"evidence": [{"id": "the knife",
                           "disappearance_accounts": ["thrown in the river", "burned in the yard"]}]}
    assert any(x["kind"] == "two_disappearances" for x in ss.check_provenance(store))


def test_generic_evidence_as_identity():
    store = {"evidence": [{"id": "a hair", "specificity": "generic",
                           "proves": "identifies the killer as Bob"}]}
    assert any(x["kind"] == "generic_as_identity" and x["severity"] == "hard"
               for x in ss.check_provenance(store))


def test_confession_sole_support():
    store = {"claims": [{"proposition": "Bob did it", "class": "suspect",
                         "presented_as_fact": True, "sole_support_for_case": True,
                         "supporting_fact_ids": []}]}
    f = ss.check_provenance(store)
    assert any(x["kind"] == "confession_sole_support" and x["severity"] == "hard" for x in f)


def test_unsupported_assertion_is_major():
    store = {"claims": [{"proposition": "the door was forced", "class": "witness",
                         "presented_as_fact": True, "supporting_fact_ids": []}]}
    f = ss.check_provenance(store)
    assert any(x["kind"] == "unsupported_assertion" and x["severity"] == "major" for x in f)


def test_supported_claim_ok():
    store = {"claims": [{"proposition": "x", "class": "inference", "presented_as_fact": True,
                         "supporting_fact_ids": ["e1"]}]}
    assert ss.check_provenance(store) == []


# ── Pass E: hypothesis scope ──────────────────────────────────────────

def test_cross_role_elimination():
    store = {"hypotheses": [
        {"role": "murder", "eliminations": [{"candidate": "Ana", "evidence_id": "alibi_x"}]},
        {"role": "leak", "eliminations": [{"candidate": "Ana", "evidence_id": "alibi_x"}]},
    ]}
    assert any(x["kind"] == "cross_role_elimination" and x["severity"] == "hard"
               for x in ss.check_hypothesis_scope(store))


def test_pool_overclaim():
    store = {"hypotheses": [{"role": "murder", "candidates": ["Ana", "Bob", "Cy"],
                            "eliminations": [{"candidate": "Ana", "evidence_id": "e"}],
                            "concluded_single": "Bob"}]}
    # Bob concluded, but Cy also remains → overclaim
    assert any(x["kind"] == "pool_overclaim" and x["severity"] == "hard"
               for x in ss.check_hypothesis_scope(store))


def test_pool_ok_when_one_remains():
    store = {"hypotheses": [{"role": "murder", "candidates": ["Ana", "Bob"],
                            "eliminations": [{"candidate": "Ana", "evidence_id": "e"}],
                            "concluded_single": "Bob"}]}
    assert ss.check_hypothesis_scope(store) == []


# ── Pass G: quantities & identity ─────────────────────────────────────

def test_age_date_mismatch():
    store = {"story_year": "2020",
             "characters": [{"name": "Ana", "age": "30", "birth_year": "1970"}]}  # implies 50
    assert any(x["kind"] == "age_date_mismatch" and x["severity"] == "hard"
               for x in ss.check_quantities(store))


def test_age_stated_two_ways():
    store = {"characters": [{"name": "Ana", "age": "34"}, {"name": "Ana", "age": "41"}]}
    assert any(x["kind"] == "quantity_mismatch" for x in ss.check_quantities(store))


def test_age_ok():
    store = {"story_year": "2020", "characters": [{"name": "Ana", "age": "50", "birth_year": "1970"}]}
    assert ss.check_quantities(store) == []


# ── Pass H: clue payoff ───────────────────────────────────────────────

def test_unpaid_high_clue():
    store = {"clues": [{"id": "the locket", "salience": "high", "status": "unresolved"}]}
    assert any(x["kind"] == "unpaid_clue" and x["severity"] == "major"
               for x in ss.check_clue_payoff(store))


def test_clue_ok_when_paid_or_open_with_reason():
    store = {"clues": [
        {"id": "a", "salience": "high", "status": "paid_off"},
        {"id": "b", "salience": "high", "status": "open", "open_reason": "sequel hook"},
        {"id": "c", "salience": "low", "status": "unresolved"},
    ]}
    assert ss.check_clue_payoff(store) == []


# ── parse / bucket / apply_patches ────────────────────────────────────

def test_parse_store_tolerant():
    empty = ss.parse_store("not json")
    assert all(empty[k] == [] for k in ("characters", "timeline", "evidence"))
    good = ss.parse_store('{"characters": [{"name": "Ana"}], "story_year": "2020"}')
    assert good["characters"][0]["name"] == "Ana" and good["story_year"] == "2020"


def test_bucket_and_apply_patches():
    findings = [{"severity": "hard"}, {"severity": "major"}, {"severity": "soft"}]
    b = ss.bucket(findings)
    assert len(b["hard"]) == 1 and len(b["major"]) == 1 and len(b["soft"]) == 1
    out, applied, un = ss.apply_patches("Ana was 41.", [{"find": "41", "replace": "34"}])
    assert applied == 1 and "41" not in out and un == []


# ── run_validator ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_validator_extracts_verifies_and_patches():
    seq = {"n": 0}

    async def extract_fn(prompt):
        # before the patch the age conflicts; after, it's consistent
        body = prompt.split("TEXT:", 1)[-1]
        if "41" in body:
            return '{"characters": [{"name": "Ana", "age": "34"}, {"name": "Ana", "age": "41"}]}'
        return '{"characters": [{"name": "Ana", "age": "34"}]}'

    async def revise_fn(prompt):
        return '[{"find": "Ana, 41,", "replace": "Ana, 34,"}]'

    text = "# Ch1\n\nAna was 34.\n\n# Ch2\n\nAna, 41, arrived."
    res = await ss.run_validator(text, extract_fn=extract_fn, revise_fn=revise_fn)
    assert res["revised"] is True
    assert "41" not in res["text"]
    assert res["hard"] == []
    assert res["patched"] == 1


@pytest.mark.asyncio
async def test_run_validator_reports_hard_without_reviser():
    async def extract_fn(prompt):
        return '{"characters": [{"name": "Ana", "age": "34"}, {"name": "Ana", "age": "41"}]}'

    res = await ss.run_validator("Ana 34 then 41.", extract_fn=extract_fn, revise_fn=None)
    assert len(res["hard"]) >= 1
    assert res["revised"] is False
    b = ss.blocking_analysis(res)
    assert b["hard"] and "pass" in b["hard"][0]


# ── Fail-safe regressions (from the adversarial verification) ─────────

def test_pool_overclaim_failsafe_on_absent_eliminations():
    # concluded_single + candidates but NO captured eliminations → under-extracted,
    # must NOT fire (was a hard false-block).
    store = {"hypotheses": [{"role": "murder", "candidates": ["Alice", "Bob", "Carol"],
                            "eliminations": [], "concluded_single": "Alice"}]}
    assert ss.check_hypothesis_scope(store) == []


def test_pool_overclaim_unions_across_merged_entries():
    # eliminations in one same-role entry, the conclusion in another (as merge()
    # produces across chunks) → must NOT fire.
    store = {"hypotheses": [
        {"role": "murder", "candidates": ["Alice", "Bob"], "eliminations": []},
        {"role": "murder", "eliminations": [{"candidate": "Bob", "evidence_id": "e"}],
         "concluded_single": "Alice"},
    ]}
    assert ss.check_hypothesis_scope(store) == []


def test_pool_overclaim_still_fires_when_truly_overclaimed():
    store = {"hypotheses": [{"role": "murder", "candidates": ["A", "B", "C"],
                            "eliminations": [{"candidate": "A", "evidence_id": "e"}],
                            "concluded_single": "B"}]}  # C also remains
    assert any(x["kind"] == "pool_overclaim" for x in ss.check_hypothesis_scope(store))


def test_quantity_numeric_equal_does_not_fire():
    for a, b in [("34", "34.0"), ("34", "34 years old"), ("34", "thirty-four (34)")]:
        store = {"characters": [{"name": "Ana", "age": a}, {"name": "Ana", "age": b}]}
        assert ss.check_quantities(store) == [], f"{a!r} vs {b!r} should not fire"


def test_quantity_numeric_difference_fires():
    store = {"characters": [{"name": "Ana", "age": "34"}, {"name": "Ana", "age": "41"}]}
    assert any(x["kind"] == "quantity_mismatch" for x in ss.check_quantities(store))


def test_merge_rebases_chunk_local_order():
    # two chunks each numbering order from 1; same actor at different places in each →
    # after merge the orders are globally distinct so NO false two_places.
    c1 = {"timeline": [{"id": "s1", "order": 1, "location": "London", "participants": ["Ana"]}]}
    c2 = {"timeline": [{"id": "s2", "order": 1, "location": "Paris", "participants": ["Ana"]}]}
    merged = ss.merge([c1, c2])
    orders = sorted(e["order"] for e in merged["timeline"])
    assert orders[0] != orders[1]  # rebased
    assert ss.check_chronology(merged) == []


def test_generic_identity_negation_does_not_fire():
    for proves in ["cannot identify any individual", "the blood is the common type O",
                   "does not identify the wearer", "only narrows it to a group"]:
        store = {"evidence": [{"id": "x", "specificity": "generic", "proves": proves}]}
        assert ss.check_provenance(store) == [], f"{proves!r} should not fire"


def test_generic_identity_affirmative_fires():
    store = {"evidence": [{"id": "x", "specificity": "generic",
                           "proves": "identifies the killer as Bob"}]}
    assert any(x["kind"] == "generic_as_identity" for x in ss.check_provenance(store))


def test_stringified_false_booleans_do_not_fire():
    # a weak model emitting JSON bools as strings must not mis-fire / mis-escalate
    store = {"claims": [{"proposition": "x", "class": "suspect",
                         "presented_as_fact": "false", "sole_support_for_case": "false",
                         "supporting_fact_ids": []}]}
    assert ss.check_provenance(store) == []
    store2 = {"evidence": [{"id": "y", "custody_gap": "false", "key_proof": "true"}]}
    assert ss.check_provenance(store2) == []


def test_stringified_true_booleans_fire():
    store = {"claims": [{"proposition": "Bob did it", "class": "suspect",
                         "presented_as_fact": "true", "sole_support_for_case": "true",
                         "supporting_fact_ids": []}]}
    assert any(x["kind"] == "confession_sole_support" and x["severity"] == "hard"
               for x in ss.check_provenance(store))


def test_malformed_nested_shapes_do_not_crash():
    # nested arrays of strings (not dicts) — the passes must skip, not crash
    store = {
        "characters": [{"name": "Sarah", "location_by_time": ["home"], "knows": ["a fact"]}],
        "timeline": ["not a dict", {"order": 1, "actor": "Sarah", "uses_facts": ["x"]}],
        "hypotheses": [{"role": "murder", "eliminations": ["nope"], "candidates": ["A"]}],
        "clues": ["nope"], "evidence": ["nope"], "claims": ["nope"],
    }
    # verify() wraps each check; none should raise
    assert isinstance(ss.verify(store), list)
    assert isinstance(ss.check_chronology(store), list)
    assert isinstance(ss.check_knowledge(store), list)
    assert isinstance(ss.check_hypothesis_scope(store), list)


def test_clue_open_reason_any_status():
    store = {"clues": [{"id": "x", "salience": "high", "status": "unresolved",
                        "open_reason": "sequel hook"}]}
    assert ss.check_clue_payoff(store) == []
    store2 = {"clues": [{"id": "y", "salience": "high", "status": "paid off"}]}  # space
    assert ss.check_clue_payoff(store2) == []
