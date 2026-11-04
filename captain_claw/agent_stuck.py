"""Canonical "give-up" replies the agent loop returns when it cannot proceed,
plus the substrings used to detect them downstream.

Single source of truth, shared by the agent runtime (which produces the messages)
and Flight Deck (which serves the detection markers to the Council UI so it can
auto-nudge a stuck agent). Keep messages and markers in sync here — do not inline
these strings anywhere else.
"""

# Shared tail of every bare give-up reply.
RETRY_HINT = "Please try again or simplify the task."

# Bare give-up replies (no salvaged partial content).
MSG_RETRIES_EXHAUSTED = (
    f"I wasn't able to complete the request — retries exhausted. {RETRY_HINT}"
)
# Running out of steps is not a failure of the work: the turn stopped while it
# was still going. Say so, and say how to go on ("continue" picks the work up
# from the session) — not "try again or simplify", which made users restart
# from nothing. "budget exhausted" stays in both texts: STUCK_MARKERS and the
# Council auto-nudge key on it.
CONTINUE_HINT = "Reply **continue** and I'll pick up where I left off."
MSG_BUDGET_EXHAUSTED = (
    "I reached this message's step limit (iteration budget exhausted) before "
    f"finishing. {CONTINUE_HINT}"
)


def budget_partial(partial: str, steps: int | None = None) -> str:
    """The reply when the step limit ends a turn that has something to show."""
    used = f" ({steps} steps)" if steps else ""
    return (
        f"⏸️ I reached this message's step limit{used} before finishing (iteration "
        f"budget exhausted). Here's where it stands — {CONTINUE_HINT[0].lower()}"
        f"{CONTINUE_HINT[1:]}\n\n{partial}"
    )


MSG_STUCK = (
    f"I got stuck and couldn't make further progress. {RETRY_HINT}"
)
# The model returned zero-length content until the iteration budget ran
# out. Distinct from the three above: nothing failed and nothing was
# refused — the model simply produced nothing, which weak local models do
# when a decoding grammar, a reasoning block, or an oversized KV cache
# eats the whole generation. Ends with RETRY_HINT, so the existing
# "simplify the task" marker already covers it downstream.
MSG_EMPTY_RESPONSE = (
    f"The model returned an empty response. {RETRY_HINT}"
)

# Lowercase substrings that identify a give-up reply — both the bare messages
# above and the "⚠️ …here's what I have so far" partial variants. "simplify the
# task" covers all three bare give-ups; the other two catch the partial variants.
STUCK_MARKERS = [
    "simplify the task",
    "got stuck and couldn",
    "budget exhausted",
]
