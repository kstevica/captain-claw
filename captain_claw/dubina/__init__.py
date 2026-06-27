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

from captain_claw.dubina.coder import (
    CoderVerifier,
    CommandRunner,
    ProviderForTier,
    Workspace,
    ensure_tests,
    extract_code_blocks,
    make_coder_generator,
    provider_for_tier_from_config,
    shell_command_runner,
)
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
from captain_claw.dubina.reasoning import (
    DEFAULT_CRITIC_MODES,
    CriticVerdict,
    ReasoningJudge,
    ReasonVerifier,
    agreement_score,
    extract_answer,
    load_critic_modes,
    make_mode_critic,
    make_reasoning_generator,
)

__all__ = [
    "CODER_LADDER",
    "REASON_LADDER",
    "Budget",
    "Candidate",
    "CoderVerifier",
    "CommandRunner",
    "CriticVerdict",
    "DEFAULT_CRITIC_MODES",
    "EngineConfig",
    "Generator",
    "HorizonEngine",
    "ProviderForTier",
    "ReasonVerifier",
    "ReasoningJudge",
    "RunResult",
    "Step",
    "StepOutcome",
    "Tier",
    "Verdict",
    "Verifier",
    "Workspace",
    "agreement_score",
    "any_pass_aggregator",
    "ensure_tests",
    "extract_answer",
    "extract_code_blocks",
    "load_critic_modes",
    "majority_agreement_aggregator",
    "make_coder_generator",
    "make_mode_critic",
    "make_reasoning_generator",
    "provider_for_tier_from_config",
    "resolve_ladder",
    "shell_command_runner",
]
