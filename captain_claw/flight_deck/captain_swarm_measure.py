"""Captain Swarm — measuring an agent on a practice set.

docs/captain-swarm-plan.md §5 and §15 (Phase 1). The agent answers each case
on its exam lane (``exam_turn``): a session of its own, deleted afterwards,
that can only read, keeps nothing, and can't send or change anything. So a
measurement over the owner's mail never writes a draft, never replies and
never learns from the cases.

For inbox triage the question is whether the email needs the owner's own
reply; for calendar invitations, whether the owner accepts (:func:`question_for`).
Either is answered YES or NO on the first line. Captain reads that word
(:func:`parse_yes_no`) and compares it with what the owner actually did
(``captain_swarm_score.agreement``). Every case is asked *repeat* times, so
the result carries its own noise: F1 per run, and their spread.

A measurement keeps, per case, the case id, the owner's label and each run's
yes/no, seconds and tokens — never the mail text and never the agent's words.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from captain_claw.flight_deck.captain_swarm_practice import sample_evenly
from captain_claw.flight_deck.captain_swarm_score import agreement, stats

Ask = Callable[[str], Awaitable[Mapping[str, Any]]]

_YES_NO_RE = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
_LOOSE_RE = re.compile(r"\b(YES|NO)\b")


def triage_question(case: Mapping[str, Any], owner_name: str = "") -> str:
    """The question one case asks. It holds what a person triaging would see,
    and asks for the decision only — never a draft."""
    who = owner_name.strip() or "the owner of this inbox"
    head = [f"From: {case.get('from') or '(unknown)'}",
            f"To: {', '.join(case.get('to') or []) or '(not shown)'}"]
    if case.get("cc"):
        head.append(f"Cc: {', '.join(case['cc'])}")
    head += [f"Date: {case.get('received') or ''}", f"Subject: {case.get('subject') or '(no subject)'}"]
    if case.get("attachments"):
        head.append(f"Attachments: {', '.join(case['attachments'])}")
    if case.get("categories"):
        head.append(f"Gmail sorted it under: {', '.join(case['categories'])}")
    if case.get("bulk"):
        head.append("It carries mailing-list or automated-sender headers.")
    if int(case.get("position") or 1) > 1:
        head.append(f"It is message {case['position']} from others in its thread"
                    + (f"; {who} had already written in that thread." if case.get("owner_wrote_before") else "."))
    body = str(case.get("body") or "").strip() or "(no text)"
    return (
        f"This email arrived in {who}'s inbox. Decide whether it needs {who}'s own reply.\n\n"
        + "\n".join(head) + "\n\n" + body + "\n\n"
        + "Answer on the first line with exactly YES (it needs their reply) or NO (it doesn't), "
        + "then one short line on why. Don't write a reply."
    )


def invite_question(case: Mapping[str, Any], owner_name: str = "") -> str:
    """The question one calendar invitation asks: accept, or not."""
    who = owner_name.strip() or "the owner of this calendar"
    when = case.get("start") or ""
    if case.get("all_day"):
        when += " (all day)"
    elif case.get("duration_min"):
        when += f", {case['duration_min']} minutes"
    head = [f"Title: {case.get('title') or '(no title)'}", f"Organizer: {case.get('organizer') or '(unknown)'}",
            f"When: {when}", f"Invited: {case.get('guests', 0)} other guest(s)"
            + (f": {', '.join(case['others'])}" if case.get("others") else "")]
    if case.get("recurring"):
        head.append("It repeats.")
    if case.get("location"):
        head.append(f"Where: {case['location']}")
    body = str(case.get("description") or "").strip() or "(no description)"
    return (
        f"{who} was invited to this calendar event. Decide whether {who} accepts it.\n\n"
        + "\n".join(head) + "\n\n" + body + "\n\n"
        + "Answer on the first line with exactly YES (they accept) or NO (they decline), "
        + "then one short line on why."
    )


def question_for(kind: str) -> tuple[Callable[[Mapping[str, Any], str], str], str]:
    """``(question, metric name)`` for a kind of practice set."""
    if kind == "calendar_invites":
        return invite_question, "f1_accepts"
    return triage_question, "f1_needs_reply"


def parse_yes_no(answer: str) -> bool | None:
    """The agent's decision: the first line's YES or NO; else a lone YES or NO
    in capitals early in the answer; else None (counted against it)."""
    text = (answer or "").strip()
    if not text:
        return None
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    m = _YES_NO_RE.match(first.replace("*", "").replace("#", ""))
    if m and len(set(_LOOSE_RE.findall(first))) < 2:   # "YES or NO?" decides nothing
        return m.group(1).lower() == "yes"
    found = {w for w in _LOOSE_RE.findall(text[:200])}
    if len(found) == 1:
        return found.pop() == "YES"
    return None


def pick_cases(cases: list[dict], limit: int) -> list[dict]:
    """The cases a measurement asks: all of them, or *limit* spread evenly over time."""
    return sample_evenly(cases, limit) if limit and limit < len(cases) else list(cases)


async def run_triage(cases: list[dict], *, ask: Ask, repeat: int = 3, owner_name: str = "",
                     progress: Callable[[int, int], None] | None = None,
                     should_stop: Callable[[], bool] | None = None,
                     kind: str = "mail_triage") -> dict[str, Any]:
    """Ask every case *repeat* times and score the answers against the
    owner's labels. ``stopped`` is True when *should_stop* ended it early;
    the runs finished by then are still scored. *kind* picks the question."""
    question, metric = question_for(kind)
    runs = max(1, int(repeat))
    total = runs * len(cases)
    rows = {c["id"]: {"id": c["id"], "label": bool(c.get("label")), "predicted": [], "seconds": [],
                      "tokens": [], "errors": 0} for c in cases}
    per_run: list[dict] = []
    asked = 0
    stopped = False
    lookups = 0
    for k in range(runs):
        pairs: list[tuple[bool | None, bool]] = []
        for case in cases:
            if should_stop and should_stop():
                stopped = True
                break
            started = time.monotonic()
            try:
                got = await ask(question(case, owner_name))
            except Exception as exc:  # noqa: BLE001 — one failed ask is a miss, not a crash
                got = {"error": str(exc)}
            row = rows[case["id"]]
            predicted = None if got.get("error") else parse_yes_no(str(got.get("answer") or ""))
            row["predicted"].append(predicted)
            row["seconds"].append(float(got.get("seconds") or round(time.monotonic() - started, 1)))
            row["tokens"].append(int(got.get("tokens") or 0))
            row["errors"] += 1 if got.get("error") else 0
            lookups += len(got.get("lookups") or [])
            pairs.append((predicted, row["label"]))
            asked += 1
            if progress:
                progress(asked, total)
        if stopped:
            break
        per_run.append(agreement(pairs))
    f1 = stats(r["f1"] for r in per_run)
    answered = [r for r in rows.values() if r["predicted"]]
    seconds = [s for r in answered for s in r["seconds"]]
    tokens = [t for r in answered for t in r["tokens"]]
    return {
        "metric": metric,
        "value": round(f1.mean, 4),
        "f1": f1.to_dict(),
        "runs": per_run,
        "pooled": agreement((p, r["label"]) for r in answered for p in r["predicted"]),
        "guardrails": {
            "seconds_per_case": round(sum(seconds) / len(seconds), 1) if seconds else 0.0,
            "tokens_per_case": round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
            "lookups_per_case": round(lookups / asked, 3) if asked else 0.0,
            "errors": sum(r["errors"] for r in rows.values()),
            "intents_per_case": 0.0,   # the exam lane can't send or change anything
        },
        "asked": asked,
        "stopped": stopped,
        "cases": list(rows.values()),
    }
