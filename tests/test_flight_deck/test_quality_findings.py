"""Increment 6 of the quality-tightening plan: the blocking gate.

The rules under test: only text-re-verifiable consistency criticals drive
rounds; contract/ledger criticals ride the checklist and the verdict but never
loop; a collapsed or check-worsening revision is rejected; cap/budget
exhaustion ends the loop with the best text and an honest verdict — work is
never lost.
"""

from __future__ import annotations

from captain_claw.flight_deck.quality_findings import (
    blocking_checklist,
    criticals,
    from_claim_check,
    from_consistency,
    from_contract,
    loop_drivers,
    run_gate,
)
from captain_claw.flight_deck.quality_profile import TokenBudget


def _driver(detail: str = "“total” conflicts: 549620 vs 157000") -> dict:
    return {"source": "consistency", "kind": "identity",
            "severity": "critical", "detail": detail}


def _ledger_finding() -> dict:
    return {"source": "consistency", "kind": "ledger", "severity": "critical",
            "detail": "text says 350000 but ledger records 300000"}


def _contract_finding() -> dict:
    return {"source": "contract", "kind": "deterministic", "severity": "critical",
            "detail": "grant exceeds the 150000 cap"}


# ── mappers + classification ─────────────────────────────────────────

def test_mappers_produce_the_unified_shape():
    c = from_consistency([{"kind": "relation", "severity": "critical", "detail": "sum broken"}])
    assert c == [{"source": "consistency", "kind": "relation",
                  "severity": "critical", "detail": "sum broken"}]
    k = from_contract([{"text": "grant in range", "severity": "critical",
                        "how": "deterministic", "note": "over cap"}])
    assert k[0]["source"] == "contract" and "over cap" in k[0]["detail"]
    cc = from_claim_check([
        {"verdict": "refuted", "claim": "X", "correction": "Y"},
        {"verdict": "unverifiable", "claim": "Z", "hedge": "reportedly Z"},
        {"verdict": "confirmed", "claim": "W"},
    ])
    assert [f["severity"] for f in cc] == ["critical", "major"]


def test_only_text_reverifiable_consistency_criticals_drive_the_loop():
    findings = [_driver(), _ledger_finding(), _contract_finding(),
                {"source": "consistency", "kind": "identity",
                 "severity": "minor", "detail": "small drift"}]
    assert len(criticals(findings)) == 3
    drivers = loop_drivers(findings)
    assert drivers == [findings[0]]  # ledger + contract are passengers


def test_checklist_dedupes_numbers_and_hints():
    text = blocking_checklist([_driver(), _driver(), _contract_finding()])
    assert text.startswith("These CRITICAL findings block finalization.")
    assert "1. [consistency]" in text and "2. [contract]" in text
    assert "3." not in text  # duplicate collapsed
    assert blocking_checklist([]) == ""


# ── the gate loop ────────────────────────────────────────────────────

def _revise(outputs: list[str], calls: list[str]):
    async def fn(prompt: str) -> str:
        calls.append(prompt)
        return outputs[min(len(calls) - 1, len(outputs) - 1)]
    return fn


def _recheck(sequences: list[list[dict]]):
    state = {"i": 0}

    async def fn(text: str) -> list[dict]:
        out = sequences[min(state["i"], len(sequences) - 1)]
        state["i"] += 1
        return out
    return fn


_LONG = "solid deliverable content. " * 60  # > 800 chars


async def test_clean_findings_skip_the_gate_entirely():
    calls: list[str] = []
    res = await run_gate(_LONG, findings=[], revise_fn=_revise(["x"], calls))
    assert res["verdict"] == "clean" and res["rounds"] == 0 and calls == []
    assert res["text"] == _LONG


async def test_driver_cleared_in_one_round_is_clean():
    calls: list[str] = []
    res = await run_gate(
        _LONG, findings=[_driver()],
        revise_fn=_revise([_LONG.replace("solid", "fixed")], calls),
        consistency_recheck_fn=_recheck([[]]))  # re-check comes back clean
    assert res["verdict"] == "clean" and res["revised"] is True
    assert res["rounds"] == 1 and "fixed" in res["text"]
    assert res["remaining"] == []


async def test_persistent_driver_stops_at_the_cap_with_honest_verdict():
    calls: list[str] = []
    still = [{"kind": "identity", "severity": "critical", "detail": "still broken"}]
    res = await run_gate(
        _LONG, findings=[_driver()],
        revise_fn=_revise([_LONG + " r1", _LONG + " r2"], calls),
        consistency_recheck_fn=_recheck([still, still]),
        max_rounds=2)
    assert res["rounds"] == 2 and len(calls) == 2
    assert res["verdict"] == "critical_findings_remain"
    assert len(res["remaining"]) == 1
    assert res["text"].endswith("r2")  # best-effort text kept, never discarded


async def test_collapsed_revision_keeps_prior_text():
    res = await run_gate(
        _LONG, findings=[_driver()],
        revise_fn=_revise(["too short"], []),
        consistency_recheck_fn=_recheck([[]]))
    assert res["text"] == _LONG and res["revised"] is False
    assert res["verdict"] == "critical_findings_remain"


async def test_worsening_revision_is_reverted():
    worse = [{"kind": "identity", "severity": "critical", "detail": f"broken {i}"}
             for i in range(3)]
    res = await run_gate(
        _LONG, findings=[_driver()],
        revise_fn=_revise([_LONG + " mangled"], []),
        consistency_recheck_fn=_recheck([worse]))
    assert res["text"] == _LONG and res["revised"] is False
    assert res["verdict"] == "critical_findings_remain"
    assert len(res["remaining"]) == 1  # the original driver, not the 3 new ones


async def test_passenger_only_criticals_get_one_revision_no_loop():
    calls: list[str] = []
    res = await run_gate(
        _LONG, findings=[_contract_finding(), _ledger_finding()],
        revise_fn=_revise([_LONG + " amended"], calls),
        consistency_recheck_fn=_recheck([[]]),
        max_rounds=3)
    assert len(calls) == 1 and res["rounds"] == 1
    assert res["revised"] is True and res["text"].endswith("amended")
    # Passengers can't be re-verified — they stay in the verdict, annotated.
    assert res["verdict"] == "critical_findings_remain"
    assert all("not deterministically re-verifiable" in f["note"]
               for f in res["remaining"])


async def test_budget_exhaustion_stops_before_spending():
    b = TokenBudget(100)  # can't afford anything
    calls: list[str] = []
    res = await run_gate(
        _LONG, findings=[_driver()],
        revise_fn=_revise(["x"], calls), budget=b, est=1000)
    assert calls == [] and res["rounds"] == 0
    assert res["verdict"] == "critical_findings_remain"


async def test_no_recheck_fn_demotes_drivers_to_applied_unverified():
    res = await run_gate(
        _LONG, findings=[_driver()],
        revise_fn=_revise([_LONG + " corrected"], []),
        consistency_recheck_fn=None)
    assert res["revised"] is True and res["text"].endswith("corrected")
    assert res["verdict"] == "critical_findings_remain"
    assert res["remaining"][0]["note"] == "revision applied — not re-verified"
