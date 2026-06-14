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
MSG_BUDGET_EXHAUSTED = (
    f"I wasn't able to complete the request — iteration budget exhausted. {RETRY_HINT}"
)
MSG_STUCK = (
    f"I got stuck and couldn't make further progress. {RETRY_HINT}"
)

# Lowercase substrings that identify a give-up reply — both the bare messages
# above and the "⚠️ …here's what I have so far" partial variants. "simplify the
# task" covers all three bare give-ups; the other two catch the partial variants.
STUCK_MARKERS = [
    "simplify the task",
    "got stuck and couldn",
    "budget exhausted",
]
