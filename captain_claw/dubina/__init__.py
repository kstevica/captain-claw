"""Dubina ("depth") — Frontier Horizon engine.

Simulates top-frontier behaviour (north star: Fable 5, GPT-5.6 max) on a cheaper
paid model by spending test-time compute, gated by verifiers, escalating up the
paid tier ladder only when a verifier demands it. See FRONTIER_HORIZON_DESIGN.md.

Phase 0 ships the substrate-agnostic engine: the budget-bounded
``decompose -> step -> verify -> recover/escalate -> advance`` loop, the
``Verifier``/``Generator`` plug-points, and the escalation-ladder controller.
The two verifier plug-ins (coder = ground-truth tests; reasoning = self-consistency
+ diverse-lens critics) and the Flight Deck surface land in later phases.
"""

from __future__ import annotations

from captain_claw.dubina.engine import (
    CODER_LADDER,
    REASON_LADDER,
    Budget,
    Candidate,
    EngineConfig,
    Generator,
    HorizonEngine,
    RunResult,
    Step,
    StepOutcome,
    Tier,
    Verdict,
    Verifier,
    any_pass_aggregator,
    majority_agreement_aggregator,
    resolve_ladder,
)

__all__ = [
    "CODER_LADDER",
    "REASON_LADDER",
    "Budget",
    "Candidate",
    "EngineConfig",
    "Generator",
    "HorizonEngine",
    "RunResult",
    "Step",
    "StepOutcome",
    "Tier",
    "Verdict",
    "Verifier",
    "any_pass_aggregator",
    "majority_agreement_aggregator",
    "resolve_ladder",
]
