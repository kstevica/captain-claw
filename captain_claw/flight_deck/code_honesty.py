"""A2 — completion-honesty guard for Code agents.

Research's honesty guard (``quality_profile.UNVERIFIED_GUARD_DIRECTIVE``) stops a
model asserting an unconfirmable *fact*. Code's failure mode is the sibling: a
model claiming *work it did not do* — "tests pass", "the build is green", "done" —
without running anything, or reporting a fix it never wrote to disk (the SW10
run's core failure). Code already has deterministic catches (the ``_acted`` check,
the C1 test gate); this is the prompt-side prevention that stops the false claim
being made in the first place.

Domain-agnostic, prompt-only (zero tokens beyond the added instruction). Paired
with ``QualityProfile.honesty_guard`` (default on) — but the route applies it only
when the run has some quality profile active, so a bare off-profile Code run stays
byte-for-byte today's prompts.
"""

from __future__ import annotations

CODE_HONESTY_DIRECTIVE = (
    "\n\nCLAIM ONLY WHAT YOU VERIFIED: never report that tests pass, the build "
    "succeeds, a server runs, or the task is done unless you actually ran the "
    "command THIS turn and saw it succeed — name the command and its result. If "
    "you did not run it, say so plainly (\"not run\") rather than implying success. "
    "A change exists only when you wrote it to disk with a write/edit tool call; "
    "do not describe an edit you did not make. If you could not finish or verify "
    "something, state exactly what remains and why — an honest \"unverified\" or "
    "\"blocked on X\" is worth more than a confident false \"done\"."
)

# Output modes — the completeness-vs-caution posture, code-flavored.
CODE_CONSERVATIVE_DIRECTIVE = (
    "\n\nOUTPUT MODE — CAUTIOUS: correctness outranks apparent completeness. "
    "Implement only what you can make actually work; do not stub, fake, or "
    "hard-code a result to make a check pass. Where a requirement is genuinely "
    "blocked (missing dependency, ambiguous spec, unavailable service), leave a "
    "clearly-marked TODO with the reason and report it, rather than shipping "
    "code that only looks finished."
)

CODE_COMPLETE_DIRECTIVE = (
    "\n\nOUTPUT MODE — FULL BUILD: implement every part of the task end to end. "
    "Where a real value or integration is unavailable, keep momentum with a "
    "clearly-labeled placeholder and note it — but a placeholder is never "
    "presented as working, and completeness never justifies faking a passing "
    "test or a result you did not actually produce."
)


def output_mode_directive(mode: str) -> str:
    """The prompt block for an ``output_mode`` value ("" → nothing, i.e. today)."""
    if mode == "conservative":
        return CODE_CONSERVATIVE_DIRECTIVE
    if mode == "complete":
        return CODE_COMPLETE_DIRECTIVE
    return ""


def guard_directive(honesty: bool, output_mode: str = "") -> str:
    """The combined honesty + output-mode suffix appended to build/fix prompts.
    Empty when honesty is off and no output mode is set — today's prompt."""
    parts = []
    if honesty:
        parts.append(CODE_HONESTY_DIRECTIVE)
    parts.append(output_mode_directive(output_mode))
    return "".join(parts)
