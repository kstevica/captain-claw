"""Token ledger — hard input-budget enforcement for Mrav prompts.

Every Mrav LLM call must fit ``input_cap`` tokens INCLUDING system prompt,
toolpack, state and instruction. Counting is deliberately conservative
(chars/3.6 instead of the usual chars/4) so the reserve absorbs tokenizer
variance across model families — the cap is a contract, not a hope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

TRIM_MARKER = "…[trimmed]…"
CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for budget math (never underestimates hard)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def truncate_tokens(text: str, max_tokens: int, keep: str = "tail") -> str:
    """Trim *text* to ~max_tokens, keeping the head, tail, or both ends.

    ``keep="split"`` keeps half from each end — useful for tool output where
    both the command echo (head) and the result (tail) matter.
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    budget_chars = max(0, int(max_tokens * CHARS_PER_TOKEN) - len(TRIM_MARKER))
    if budget_chars <= 0:
        return TRIM_MARKER
    if keep == "head":
        return text[:budget_chars] + TRIM_MARKER
    if keep == "split":
        half = budget_chars // 2
        return text[:half] + TRIM_MARKER + text[-(budget_chars - half):]
    return TRIM_MARKER + text[-budget_chars:]


@dataclass
class Section:
    """One prompt section with its own token budget.

    ``flex=True`` marks sections the ledger may squeeze further if the
    assembled prompt still exceeds the global cap (misconfigured budgets);
    frozen sections (contract, toolpack) are never squeezed post-fit so the
    prompt prefix stays byte-stable across steps for KV/prefix caching.
    """

    name: str
    text: str
    budget: int
    keep: str = "tail"
    flex: bool = False


@dataclass
class LedgerReport:
    """What the ledger actually did — for traces and tests."""

    total_tokens: int = 0
    cap: int = 0
    trimmed: dict[str, int] = field(default_factory=dict)
    squeezed: dict[str, int] = field(default_factory=dict)


class LedgerOverflowError(Exception):
    """Raised when a prompt cannot be made to fit the cap."""


class PromptLedger:
    """Assembles prompt sections under a hard global token cap."""

    def __init__(self, input_cap: int = 8192, reserve: int = 512):
        self.input_cap = max(256, int(input_cap))
        self.reserve = max(0, int(reserve))

    @property
    def usable(self) -> int:
        return self.input_cap - self.reserve

    def fit(self, sections: list[Section]) -> tuple[dict[str, str], LedgerReport]:
        """Fit every section to its budget, then the whole to the cap.

        Returns section-name → fitted text, plus a report. Raises
        ``LedgerOverflowError`` only when even squeezing every flex section to
        zero cannot fit — that is a configuration bug, not runtime bad luck.
        """
        report = LedgerReport(cap=self.input_cap)
        fitted: dict[str, str] = {}
        for sec in sections:
            text = sec.text or ""
            if estimate_tokens(text) > sec.budget:
                text = truncate_tokens(text, sec.budget, keep=sec.keep)
                report.trimmed[sec.name] = estimate_tokens(text)
            fitted[sec.name] = text

        total = sum(estimate_tokens(t) for t in fitted.values())
        if total <= self.usable:
            report.total_tokens = total
            return fitted, report

        # Over the global cap despite per-section budgets: squeeze flex
        # sections newest-last-first until it fits.
        overflow = total - self.usable
        for sec in reversed(sections):
            if overflow <= 0:
                break
            if not sec.flex:
                continue
            current = estimate_tokens(fitted[sec.name])
            take = min(current, overflow)
            new_budget = current - take
            fitted[sec.name] = truncate_tokens(fitted[sec.name], new_budget, keep=sec.keep)
            report.squeezed[sec.name] = new_budget
            overflow = sum(estimate_tokens(t) for t in fitted.values()) - self.usable

        total = sum(estimate_tokens(t) for t in fitted.values())
        report.total_tokens = total
        if total > self.usable:
            raise LedgerOverflowError(
                f"prompt does not fit: {total} tokens > usable {self.usable} "
                f"(cap {self.input_cap}, reserve {self.reserve}); "
                f"frozen sections alone exceed the cap — fix section budgets"
            )
        return fitted, report

    def check_messages(self, texts: list[str]) -> tuple[bool, int]:
        """Final gate right before a provider call: (fits, total_tokens)."""
        total = sum(estimate_tokens(t or "") for t in texts)
        return total <= self.input_cap, total
