"""Bat Phase 3 — the honest done-judge (Invariant B).

The judge is the one verdict the stubborn loop cannot overrule, so every path is
fail-closed: ambiguity, errors, abstentions and missing deliverables are NOT
done. Done is returned only on positive evidence (green deterministic checks +
an agree-majority panel), with a bounded anti-deadlock override.
"""

from __future__ import annotations

import pytest

from captain_claw.flight_deck.bat_judge import Check, evaluate, tally


def _panel(*scripts):
    """A panel whose i-th member returns scripts[i] (a {vote,reason} or an
    Exception to raise)."""
    async def fn(deliverable, task, idx):
        s = scripts[idx % len(scripts)]
        if isinstance(s, Exception):
            raise s
        return s
    return fn


# ── tally ────────────────────────────────────────────────────────────

def test_tally_requires_real_agree_majority():
    assert tally([{"vote": "agree"}, {"vote": "agree"}, {"vote": "disagree"}])["verdict"] == "agree"
    assert tally([{"vote": "agree"}, {"vote": "disagree"}])["verdict"] == "tie"
    assert tally([{"vote": "abstain"}, {"vote": "abstain"}])["verdict"] == "no_quorum"
    assert tally([{"vote": "disagree"}, {"vote": "agree"}, {"vote": "disagree"}])["verdict"] == "disagree"
    # abstentions never tip the balance
    assert tally([{"vote": "agree"}, {"vote": "abstain"}, {"vote": "abstain"}])["verdict"] == "agree"


# ── Layer 1: deterministic criticals block, fail-closed ───────────────

async def test_critical_failure_blocks_without_panel():
    panel = _panel({"vote": "agree"}, {"vote": "agree"}, {"vote": "agree"})
    v = await evaluate(
        task="t", deliverable="a full result",
        deterministic=[Check("tests pass", passed=False, critical=True, detail="2 failing")],
        panel_vote_fn=panel,
    )
    assert not v.done
    assert "tests pass" in v.reason and v.det_criticals
    assert not v.panel  # panel never consulted once a critical fails


async def test_missing_deliverable_is_not_done():
    v = await evaluate(task="t", deliverable="   ",
                       deterministic=[Check("gate", passed=True)],
                       panel_vote_fn=_panel({"vote": "agree"}))
    assert not v.done and "no deliverable" in v.reason


async def test_advisory_failures_do_not_block():
    v = await evaluate(
        task="t", deliverable="result",
        deterministic=[Check("style", passed=False, critical=False, detail="nit"),
                       Check("tests", passed=True, critical=True)],
        panel_vote_fn=_panel({"vote": "agree"}, {"vote": "agree"}, {"vote": "agree"}),
    )
    assert v.done and v.det_advisories and "style" in v.det_advisories[0]


# ── Layer 2: the independent panel, fail-closed ───────────────────────

async def test_panel_agree_majority_is_done():
    v = await evaluate(task="t", deliverable="result", deterministic=[Check("g", True)],
                       panel_vote_fn=_panel({"vote": "agree", "reason": "complete"},
                                            {"vote": "agree"}, {"vote": "disagree"}))
    assert v.done and v.panel["verdict"] == "agree"


async def test_panel_tie_is_not_done():
    v = await evaluate(task="t", deliverable="result", deterministic=[Check("g", True)],
                       panel_vote_fn=_panel({"vote": "agree"}, {"vote": "disagree"}), panel_size=2)
    assert not v.done and v.panel["verdict"] == "tie"


async def test_panel_exception_counts_as_disagree():
    # One agree, one raising judge → the raise becomes disagree → tie → not done.
    v = await evaluate(task="t", deliverable="result", deterministic=[Check("g", True)],
                       panel_vote_fn=_panel({"vote": "agree"}, RuntimeError("model down")),
                       panel_size=2)
    assert not v.done
    assert any("fail-closed" in vote["reason"] for vote in v.panel["votes"])


async def test_all_abstain_is_not_done():
    v = await evaluate(task="t", deliverable="result", deterministic=[Check("g", True)],
                       panel_vote_fn=_panel({"vote": "abstain"}, {"vote": "abstain"}, {"vote": "abstain"}))
    assert not v.done and v.panel["verdict"] == "no_quorum"


async def test_unparseable_vote_becomes_abstain_not_agree():
    v = await evaluate(task="t", deliverable="result", deterministic=[Check("g", True)],
                       panel_vote_fn=_panel({"vote": "maybe?"}, {"vote": "garbage"}), panel_size=2)
    assert not v.done  # garbage never reads as agree


# ── no panel configured ───────────────────────────────────────────────

async def test_no_panel_requires_green_deterministic():
    done = await evaluate(task="t", deliverable="r", deterministic=[Check("tests", True)])
    assert done.done and "all green" in done.reason
    none = await evaluate(task="t", deliverable="r", deterministic=[])
    assert not none.done  # no evidence at all → fail closed


# ── anti-deadlock override ────────────────────────────────────────────

async def test_model_veto_overridden_only_when_green_and_streak_reached():
    green = [Check("tests", True), Check("contract", True)]
    reject = _panel({"vote": "disagree", "reason": "nit"}, {"vote": "disagree"})

    # First veto round: not yet at the ceiling → still not done.
    v1 = await evaluate(task="t", deliverable="r", deterministic=green, panel_vote_fn=reject,
                        panel_size=2, prior_model_veto_rounds=0, max_model_veto_rounds=2)
    assert not v1.done and not v1.overridden

    # Streak reached → override, and it's recorded.
    v2 = await evaluate(task="t", deliverable="r", deterministic=green, panel_vote_fn=reject,
                        panel_size=2, prior_model_veto_rounds=1, max_model_veto_rounds=2)
    assert v2.done and v2.overridden and "overridden" in v2.reason


async def test_no_override_without_green_deterministic():
    # Deterministic layer empty → not "all green" → a model veto is never
    # overridden, no matter the streak.
    reject = _panel({"vote": "disagree"}, {"vote": "disagree"})
    v = await evaluate(task="t", deliverable="r", deterministic=[], panel_vote_fn=reject,
                       panel_size=2, prior_model_veto_rounds=99, max_model_veto_rounds=2)
    assert not v.done and not v.overridden
