"""Mrav — micro agentic runtime for small models (hard 8k input cap per call).

A parallel system to the 16-mixin `Agent`: same tools, same providers, same
transport shell, but prompt assembly is budget-enforced (TokenLedger), state
lives outside the model (Blackboard), and the loop is decomposed into small
single-purpose steps (PLAN / ACT / DIGEST / COMPRESS) that each fit a
2-4B-class model. Plan: docs/mrav-micro-agent-plan.md.
"""

from captain_claw.mrav.ledger import PromptLedger, Section, estimate_tokens, truncate_tokens
from captain_claw.mrav.protocol import StepAction, parse_json_object, validate_action
from captain_claw.mrav.runtime import MravRuntime
from captain_claw.mrav.state import Blackboard, Observation

__all__ = [
    "Blackboard",
    "MravRuntime",
    "Observation",
    "PromptLedger",
    "Section",
    "StepAction",
    "estimate_tokens",
    "parse_json_object",
    "truncate_tokens",
    "validate_action",
]
