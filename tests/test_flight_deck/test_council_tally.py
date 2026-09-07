"""R7 tests — the deterministic council vote tally (decision node)."""

from __future__ import annotations

from captain_claw.flight_deck.quality_profile import QualityProfile, tally_votes


def _v(vote, agent="a"):
    return {"vote": vote, "agent_id": agent}


def test_majority_agree_and_disagree():
    assert tally_votes([_v("agree", "a"), _v("agree", "b"), _v("disagree", "c")])["verdict"] == "agree"
    assert tally_votes([_v("disagree", "a"), _v("disagree", "b"), _v("agree", "c")])["verdict"] == "disagree"


def test_empty_is_no_quorum():
    for empty in (None, []):
        t = tally_votes(empty)
        assert t["verdict"] == "no_quorum"
        assert t["margin"] == 0.0
        assert t["weighted_used"] is False
        assert t["total_votes"] == 0


def test_tie_when_equal():
    t = tally_votes([_v("agree", "a"), _v("disagree", "b")])
    assert t["verdict"] == "tie"
    assert t["margin"] == 0.0


def test_abstains_never_tip():
    t = tally_votes([_v("agree", "a"), _v("abstain", "b"), _v("abstain", "c"), _v("abstain", "d")])
    assert t["verdict"] == "agree"
    assert t["counts"] == {"agree": 1, "disagree": 0, "abstain": 3}
    # All-abstain: decided==0 with votes present -> tie, margin 0.
    t2 = tally_votes([_v("abstain", "a"), _v("abstain", "b")])
    assert t2["verdict"] == "tie"
    assert t2["margin"] == 0.0


def test_unknown_vote_counts_as_abstain():
    t = tally_votes([_v("maybe", "a"), _v("agree", "b")])
    assert t["counts"] == {"agree": 1, "disagree": 0, "abstain": 1}
    assert t["verdict"] == "agree"


def test_no_weights_mirrors_counts():
    t = tally_votes([_v("agree", "a"), _v("disagree", "b")])
    assert t["weighted_used"] is False
    assert t["weighted"] == {"agree": 1.0, "disagree": 1.0, "abstain": 0.0}


def test_weighted_flip_and_margin():
    # a (weight 3) agrees; b,c (weight 1 each) disagree -> weighted agree wins.
    t = tally_votes(
        [_v("agree", "a"), _v("disagree", "b"), _v("disagree", "c")],
        weights={"a": 3.0},
    )
    assert t["verdict"] == "agree"
    assert t["weighted_used"] is True
    assert t["margin"] == 0.2  # (3 - 2) / 5
    assert t["counts"] == {"agree": 1, "disagree": 2, "abstain": 0}


def test_margin_sign_and_magnitude():
    t = tally_votes([_v("agree", "a"), _v("agree", "b"), _v("agree", "c"), _v("disagree", "d")])
    assert t["margin"] == 0.5  # (3 - 1) / 4


def test_council_tally_is_explicit_opt_in():
    assert QualityProfile.from_dict({}).council_tally is False
    for preset in ("off", "balanced", "thorough"):
        assert QualityProfile.from_dict({"profile": preset}).council_tally is False, preset
    on = QualityProfile.from_dict({"council_tally": True})
    assert on.council_tally is True
    assert on.any_enabled is True
    assert "council_tally" in QualityProfile._BOOL_FLAGS
