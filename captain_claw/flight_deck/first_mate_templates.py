"""Starter factories: a charter, done-checks and a schedule to begin from.

See docs/first-mate-plan.md §4. A template only fills in a new factory's
first draft. The owner edits it and approves it like any other, and the
store checks it like any other. No template grants anything: Bat's start gate
asks before a run would send, spend or sign up, until Phase 3 adds carry-outs.
"""

from __future__ import annotations

import copy
from typing import Any

TEMPLATES: dict[str, dict[str, Any]] = {
    "inbox-drafts": {
        "title": "Inbox drafts",
        "blurb": "Drafts replies to mail that needs one, for you to send.",
        "charter": (
            "Go through my inbox for mail from the last day that needs a reply from me. "
            "For each one, write a short draft reply in my usual tone and leave it as a draft: "
            "never send anything. Skip newsletters, notifications and mail where I'm only in Cc. "
            "Finish with a list: who wrote, what they want, and whether a draft is waiting."
        ),
        "triggers": [{"kind": "schedule", "schedule": "weekdays 08:00"}],
        "checks": {"contract": ["every mail that needs a reply has a draft or a reason it has none",
                                "nothing was sent"]},
    },
    "follow-ups": {
        "title": "Follow-up chaser",
        "blurb": "Finds threads where you're waiting on someone and drafts a nudge.",
        "charter": (
            "Find threads from the last two weeks where I asked something and nobody answered "
            "in five working days. For each, draft a short, friendly follow-up and leave it as a "
            "draft. Finish with the list of threads, how long each has waited, and the drafts."
        ),
        "triggers": [{"kind": "schedule", "schedule": "weekly mon,thu 09:00"}],
        "checks": {"contract": ["each thread listed has waited five working days or more",
                                "nothing was sent"]},
    },
    "weekly-report": {
        "title": "Weekly report",
        "blurb": "One page on last week: what moved, what's stuck, what's next.",
        "charter": (
            "Write a one-page report on last week from my mail, calendar and files: what moved "
            "forward, what is stuck and why, and what is planned for this week. Name the sources "
            "for every claim. Save it as a document in my files."
        ),
        "triggers": [{"kind": "schedule", "schedule": "weekly mon 07:30"}],
        "checks": {"contract": ["it fits on one page", "every claim names its source"]},
    },
    "price-watch": {
        "title": "Price watch",
        "blurb": "Checks prices or availability of something you want to buy.",
        "charter": (
            "Check the current price and availability of the item below at the shops that sell "
            "it in my country. Open every shop's page yourself. Report the three best offers with "
            "price, delivery cost, stock and a link. Never buy anything.\n\nItem: (describe it here)"
        ),
        "triggers": [{"kind": "schedule", "schedule": "daily 09:00"}],
        "checks": {"contract": ["every offer links to a shop page that was opened",
                                "nothing was bought"]},
    },
    "meeting-prep": {
        "title": "Meeting prep",
        "blurb": "A short brief for each of tomorrow's meetings.",
        "charter": (
            "For each meeting on my calendar tomorrow, write a short brief: who is coming, what "
            "we last discussed with them (from mail and notes), open questions, and what I should "
            "decide. One section per meeting."
        ),
        "triggers": [{"kind": "schedule", "schedule": "weekdays 17:00"}],
        "checks": {"contract": ["every meeting on tomorrow's calendar has a section"]},
    },
    "research-digest": {
        "title": "Research digest",
        "blurb": "What's new on a topic you follow, with sources.",
        "charter": (
            "Find what was published in the last week on the topic below: news, papers, releases. "
            "Keep only what is new and matters. For each item, two sentences and a link you "
            "opened yourself.\n\nTopic: (describe it here)"
        ),
        "triggers": [{"kind": "schedule", "schedule": "weekly fri 08:00"}],
        "checks": {"contract": ["every item links to a source that was opened",
                                "nothing older than a week"]},
    },
}


def listing() -> list[dict[str, Any]]:
    """Every template, for the New factory picker."""
    return [{"id": tid, **copy.deepcopy(t)} for tid, t in TEMPLATES.items()]


def fields_for(template_id: str) -> dict[str, Any] | None:
    """A template's fields for ``FirstMateStore.create``, or None for an unknown id."""
    t = TEMPLATES.get(template_id)
    if t is None:
        return None
    return {"title": t["title"], "charter": t["charter"], "runner": "bat",
            "runner_config": {"worker_mode": "plain", "steps": []},
            "triggers": copy.deepcopy(t["triggers"]), "checks": copy.deepcopy(t["checks"]),
            "template": template_id}
