"""Dubina reasoning track — the statistical verifier plug-in.

Reasoning has no ground truth, so the verifier is statistical and uses two signals,
**sequenced by cost** (design §"agreement vs critics"):

1. **Self-consistency (agreement)** — sample the step N times and measure how often
   the samples converge on the same answer. Cheap (no extra LLM calls; the engine's
   N-sample vote already produced them) and always-on. It measures *confidence /
   precision* and is the difficulty trigger.
2. **Diverse-lens critics** — run the leading answer past 2–3 cognitive modes
   (`phrygian` adversarial, `aeolian` depth, `locrian` deconstruction) as refuters.
   Expensive (one LLM call each) and conditional: fire only when agreement is low or
   the step is high-stakes. They measure *correctness / accuracy* — catching the
   "confidently wrong" answers agreement is blind to.

Because self-consistency is inherently a property of the whole sample set, the gate
lives in the engine's **aggregator** seam (`ReasoningJudge`), not the per-candidate
verifier. The per-candidate `ReasonVerifier` only does a cheap well-formedness check.

Critics' own token cost is not yet charged against the engine budget — that wiring
lands with the Flight Deck layer in Phase 3.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from captain_claw.cognitive_mode import cognitive_mode_to_prompt_block, get_mode
from captain_claw.dubina.engine import Candidate, Step, Verdict
from captain_claw.llm import LLMProvider, Message
from captain_claw.logging import get_logger

log = get_logger(__name__)

DEFAULT_CRITIC_MODES: tuple[str, ...] = ("phrygian", "aeolian", "locrian")
DEFAULT_AGREEMENT_THRESHOLD = 0.6  # fraction of samples that must converge
MIN_SAMPLES_FOR_AGREEMENT = 2      # a lone sample can't establish self-consistency

# Candidate.metadata keys the judge reads (set by the reasoning generator).
PROMPT_KEY = "prompt"
STAKES_KEY = "stakes"

ProviderForTier = Callable[[str], LLMProvider]


# ── Answer extraction & agreement ────────────────────────────────────

_ANSWER_RE = re.compile(r"(?im)^\s*(?:final answer|answer)\s*[:\-]\s*(.+?)\s*$")
_PUNCT = " \t\n.!?,;:"


def _normalize(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip(_PUNCT).lower()


def extract_answer(text: str) -> str:
    """Normalize a candidate's final answer for agreement clustering.

    Prefers the last ``Answer:``/``Final answer:`` line; falls back to the last
    non-empty line. Lower-cased, whitespace-collapsed, trailing punctuation stripped.
    """
    matches = _ANSWER_RE.findall(text or "")
    if matches:
        return _normalize(matches[-1])
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return _normalize(lines[-1]) if lines else ""


def agreement_score(answers: Sequence[str]) -> tuple[str, float]:
    """Return the majority answer and the fraction of samples that agree with it."""
    if not answers:
        return "", 0.0
    majority, count = Counter(answers).most_common(1)[0]
    return majority, count / len(answers)


# ── Per-candidate verifier (cheap) ───────────────────────────────────

class ReasonVerifier:
    """Cheap per-candidate well-formedness check; the real gate is the judge.

    Returns higher confidence when the candidate carries an explicit answer marker,
    which the judge uses to pick the representative of the majority cluster.
    """

    async def check(self, step: Step, candidate: Candidate) -> Verdict:
        text = candidate.content or ""
        if not text.strip():
            return Verdict(passed=False, confidence=0.0, feedback="empty answer")
        has_marker = bool(_ANSWER_RE.search(text))
        return Verdict(passed=True, confidence=0.8 if has_marker else 0.4)


# ── Critics (diverse lenses) ─────────────────────────────────────────

@dataclass
class CriticVerdict:
    refuted: bool
    reason: str


# A critic: ``async (question, answer) -> CriticVerdict``.
Critic = Callable[[str, str], Awaitable[CriticVerdict]]

_CRITIC_INSTRUCTION = (
    "You are an adversarial reviewer. Try hard to REFUTE the candidate answer to the "
    "question — find a real error, false assumption, or gap. Respond on the first "
    "line with exactly 'REFUTED: <reason>' if you find a genuine flaw, or "
    "'SOUND: <why it holds>' if it withstands scrutiny."
)


def make_mode_critic(
    provider: LLMProvider,
    mode_name: str,
    instruction_loader: Any | None = None,
    *,
    max_tokens: int = 1200,
) -> Critic:
    """Build a critic that reviews through one cognitive mode's lens."""
    persona = cognitive_mode_to_prompt_block(get_mode(mode_name), instruction_loader)

    async def critic(question: str, answer: str) -> CriticVerdict:
        system = f"{persona}\n\n{_CRITIC_INSTRUCTION}" if persona else _CRITIC_INSTRUCTION
        messages = [
            Message(role="system", content=system),
            Message(
                role="user",
                content=f"Question:\n{question}\n\nCandidate answer:\n{answer}",
            ),
        ]
        response = await provider.complete(messages, max_tokens=max_tokens)
        text = (response.content or "").strip()
        refuted = text.upper().startswith("REFUTED")
        return CriticVerdict(refuted=refuted, reason=text)

    return critic


def load_critic_modes(
    provider: LLMProvider,
    modes: Sequence[str] = DEFAULT_CRITIC_MODES,
    instruction_loader: Any | None = None,
) -> list[Critic]:
    return [make_mode_critic(provider, m, instruction_loader) for m in modes]


# ── The judge (engine aggregator) ────────────────────────────────────

class ReasoningJudge:
    """Engine aggregator implementing the two-stage statistical gate.

    Sequenced by cost: agreement first (free), critics only when agreement is below
    ``agreement_threshold`` or the step is high-stakes.
    """

    def __init__(
        self,
        critics: Sequence[Critic],
        *,
        agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
        min_samples: int = MIN_SAMPLES_FOR_AGREEMENT,
    ):
        self._critics = list(critics)
        self._threshold = agreement_threshold
        self._min_samples = min_samples

    async def __call__(
        self, candidates: Sequence[Candidate], verdicts: Sequence[Verdict]
    ) -> tuple[Candidate, Verdict]:
        def _best_overall() -> int:
            return max(range(len(verdicts)), key=lambda i: verdicts[i].confidence)

        # A single sample can't establish self-consistency — force the engine to
        # escalate to the N-sample vote rung.
        if len(candidates) < self._min_samples:
            i = _best_overall()
            return candidates[i], Verdict(
                False, verdicts[i].confidence,
                "insufficient samples for self-consistency",
            )

        answers = [extract_answer(c.content) for c in candidates]
        majority, ratio = agreement_score(answers)

        # Representative = highest-confidence candidate in the majority cluster.
        in_majority = [i for i, a in enumerate(answers) if a == majority]
        rep = max(in_majority, key=lambda i: verdicts[i].confidence)
        candidate = candidates[rep]

        stakes = candidate.metadata.get(STAKES_KEY, "normal")
        question = candidate.metadata.get(PROMPT_KEY, "")

        run_critics = ratio < self._threshold or stakes == "high"
        if not run_critics or not self._critics:
            # Agreement gate alone — the cheap path.
            passed = ratio >= self._threshold
            fb = "" if passed else f"low self-consistency ({ratio:.2f})"
            return candidate, Verdict(passed, ratio, fb)

        # Expensive path: diverse-lens critics must majority-survive.
        results = await asyncio.gather(
            *(c(question, majority) for c in self._critics)
        )
        survived = sum(1 for r in results if not r.refuted)
        passed = survived * 2 > len(self._critics)
        confidence = survived / len(self._critics)
        feedback = ""
        if not passed:
            feedback = " | ".join(r.reason for r in results if r.refuted)
        return candidate, Verdict(passed, confidence, feedback)


# ── Generator ────────────────────────────────────────────────────────

_REASON_SYSTEM = (
    "You are a careful reasoner. Work through the problem step by step, then end your "
    "reply with a single line: 'Answer: <concise final answer>'."
)


def make_reasoning_generator(
    provider_for_tier: ProviderForTier,
    *,
    max_tokens: int = 4000,
):
    """Build an engine ``Generator`` producing reasoned answers.

    Each candidate carries the question and stakes in its metadata so the judge can
    cluster answers and decide whether to invoke critics.
    """

    async def generate(step: Step, tier: str, feedback: str, sample: int) -> Candidate:
        parts = [step.prompt]
        if feedback:
            parts.append(
                f"\nA previous attempt was judged unsound. Reconsider. Critique:\n{feedback}"
            )
        messages = [
            Message(role="system", content=_REASON_SYSTEM),
            Message(role="user", content="\n".join(parts)),
        ]
        provider = provider_for_tier(tier)
        response = await provider.complete(messages, max_tokens=max_tokens)
        return Candidate(
            step_id=step.id,
            content=response.content or "",
            tier=tier,
            metadata={PROMPT_KEY: step.prompt, STAKES_KEY: step.stakes},
        )

    return generate
