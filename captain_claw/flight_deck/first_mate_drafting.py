"""First Mate drafts a factory from the owner's words, or from a Course goal.

See docs/first-mate-plan.md §11 (Phase 5, piece 2). One model call turns what
the owner said into a first draft: a name, the charter, when it starts, the
lines a run is checked against, the grants it needs and up to three sealed
cases. The store checks it like any other draft. It waits for the owner's
approval, on First Mate and as a card on the Line, against the hash of what
they saw. Nothing runs before that.

* **Grants:** only mail sending and organizing, and only when the words need
  them; never spending or opening accounts (those stay with Bat's own gate).
* **Starts:** a schedule the deck's scheduler accepts, new mail, or neither
  (Run now only). A schedule it can't use is dropped, not guessed.
* **Cases:** sealed, like the owner's own; one that doesn't pass the checks
  (a secret in it, no good-result lines) is left out.
* **Model:** the owner's First Mate agent's model when they picked one
  (``first_mate_agent``), else their own tiers, the best one first; no answer
  means no draft, with why.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms
from captain_claw.flight_deck.first_mate_store import FirstMateError

log = logging.getLogger(__name__)

WORDS_MAX = 2000
# What a draft may spend, thinking included (a thinking model thinks first), and on
# the try after an answer that came back empty or cut off.
DRAFT_TOKENS = 12_000
RETRY_TOKENS = 24_000
DONE_WHEN_MAX = 6
DRAFT_CASES_MAX = 3
SENDS_PER_RUN = 3
NEW_MAIL = {"kind": "events", "source": "gmail", "event_type": "new_email",
            "match": {"from": [], "label": [], "words": []}, "batch_minutes": None, "ask": False}

_SYSTEM = """You set up a standing job (a "factory") that an assistant runs again and again for one person. From what they want, write its first draft as one JSON object with these fields:

- "title": a short name, at most 5 words.
{charter}- "schedule": when it starts on its own: "daily HH:MM", "weekdays HH:MM", "weekly mon,thu HH:MM" (any days) or "every N h"; or "" when it should start on new mail instead, or only when they press Run.
- "on_new_mail": true only when it should start when new mail arrives.
- "done_when": 3 to 6 short lines checked on one run's result. Make the countable ones countable, so Captain checks them itself: the file and its kind ("final_leads.csv with columns Name, Email, Source, Reason"), how many or how long ("at least 500 rows", "1,500 to 3,000 words"), no duplicates, where things come from ("every Source is a link to the page the email is on", "personal emails, at most 20% role inboxes like info@"), who is out ("no university staff"), the sections a document must have, that code's tests pass. Then at most two lines on quality a reviewer judges ("each reason names a specific Captain feature"). When the person wrote their own "done when" lines, keep each one as its own line in their words, and keep every amount as they wrote it: "at least 60" stays "at least 60", never "60" or "exactly 60". Count anything you name (columns, sections) and make the number match the list.
- "may_send_email": true only when the job itself must send or reply to email. Writing drafts for the person to send is not sending.
- "may_organize_mail": true only when it must label, archive or move mail.
- "cases": 0 to 3 test cases for jobs that react to an input such as an email or a request: realistic inputs this job might get, each {"title": ..., "input": ..., "good_result": [1 to 3 lines]}. [] for other jobs.

Never let it spend money or open accounts. Keep your thinking short: only the JSON matters. Return ONLY the JSON object."""

# The charter when the owner's words are rewritten for it (the usual way): everything a
# run needs to be told once, so no run has to guess it. Each point is there because a
# run went wrong without it: one step for four kinds of item, columns the setup made
# up, a fact taken from a page about the item, a list filled by bending a rule.
CHARTER_REPHRASED = """- "charter": their task rewritten as a complete brief for the runs, in their voice ("Find ...", "Go through my ..."). Keep every limit, number, name and link they gave, in their words. Add only what a run needs and they left unsaid; where their words leave something open, take the sensible narrow reading and say it plainly, so they can change it before approving. Cover what applies, in this order, in short paragraphs:
  1. What one run does, and the result it finishes with, named: the file (a list with its columns, a document with its sections, code that runs).
  2. For a list: what one row is, and who or what is in and who is out. When they name several kinds or several places, make them groups with an amount for each that adds up to their total, and say how you split it. The columns, named, in order. For each fact a row carries, where it must come from (the item's own site or profile, the files they gave); a fact from a page about the item doesn't count. What happens to an item when a fact isn't found: left out, the cell left empty, or what may stand in for it. Whether fewer rows than asked is acceptable: then say "give what there is and explain the shortage in the summary; do not relax a rule to fill it". Each item once.
  3. When the test is whether something fits a product, a service or a topic, say in two or three lines what that is and who it is for, from their words. When they gave only its name or a link, keep the link and say the run reads it first.
  4. For a document or report: its sections, its length, where its facts come from.
  5. What a run must never do, their own limits first.
  Up to about 250 words. No heading, no preamble.
- "is_list": true when the result is a list of rows (leads, items, records) and nothing else, false otherwise.
"""
CHARTER_AS_WRITTEN = """- "charter": the person's own words, exactly as they wrote them. Do not rewrite, shorten or add to them.
- "is_list": true when the result is a list of rows (leads, items, records) and nothing else, false otherwise.
"""


def system_for(rephrase: bool = True) -> str:
    """The drafting prompt: the owner's task rewritten as a full charter, or kept as written."""
    return _SYSTEM.replace("{charter}", CHARTER_REPHRASED if rephrase else CHARTER_AS_WRITTEN)


SYSTEM = system_for(True)


def _parse(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if not raw or raw in ("[]", "{}"):
        raise FirstMateError("the answer came back empty (a thinking model can spend its whole budget thinking)")
    if start >= 0 and end <= start:
        raise FirstMateError("the draft was cut off before its end")
    if start < 0 or end <= start:
        raise FirstMateError("the draft wasn't JSON")
    try:
        data = json.loads(raw[start:end + 1])
    except ValueError:
        raise FirstMateError("the draft wasn't JSON") from None
    if not isinstance(data, dict):
        raise FirstMateError("the draft wasn't a JSON object")
    return data


def _lines(value: Any, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(v).strip()[:300] for v in value if str(v).strip()][:limit]


def fields_of(data: dict[str, Any]) -> dict[str, Any]:
    """The model's answer as a factory's fields, through the store's own checks.
    A schedule the scheduler refuses is dropped."""
    title = str(data.get("title") or "").strip()[:fms.TITLE_MAX] or "New factory"
    charter = str(data.get("charter") or "").strip()
    if not charter:
        raise FirstMateError("the draft had no charter")
    triggers: list[dict[str, Any]] = []
    schedule = str(data.get("schedule") or "").strip()
    if schedule:
        try:
            triggers = fms.check_triggers([{"kind": "schedule", "schedule": schedule, "tz": ""}])
        except FirstMateError:
            log.info("first mate: a drafted schedule was dropped", extra={"schedule": schedule})
    if data.get("on_new_mail") is True:
        triggers.append(dict(NEW_MAIL))
    email = data.get("may_send_email") is True
    grants = {"email": email, "organize": data.get("may_organize_mail") is True,
              "spend": False, "accounts": False, "sends_per_run": SENDS_PER_RUN if email else 0,
              "replies_only": True}
    cases = []
    for c in data.get("cases") or []:
        if isinstance(c, dict) and str(c.get("input") or "").strip():
            cases.append({"title": str(c.get("title") or "A case").strip()[:fms.TITLE_MAX],
                          "input": str(c.get("input") or "").strip(),
                          "expect": _lines(c.get("good_result"), 10)})
    return {"title": title, "charter": charter, "triggers": fms.check_triggers(triggers),
            "grants": fms.check_grants(grants), "is_list": data.get("is_list") is True,
            "checks": {"contract": _lines(data.get("done_when"), DONE_WHEN_MAX)},
            "cases": cases[:DRAFT_CASES_MAX]}


async def _ask_model(owner: str, system: str, user: str, max_tokens: int = DRAFT_TOKENS) -> str:
    """One call on the owner's First Mate agent's model when they picked one
    (first_mate_agent), else on their own tier set, the best tier first.
    ``max_tokens`` includes a thinking model's thinking."""
    from captain_claw.flight_deck import first_mate_agent
    got = await first_mate_agent.ask(fms.get_store(), owner, system, user, max_tokens=max_tokens)
    if got is not None:
        return got
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers, _provider_call
    from captain_claw.flight_deck.bat_routes import _creds_for
    from captain_claw.llm import Message
    tiers, _ = await _load_owner_tiers(get_db(), owner)
    creds = _creds_for(tiers, prefer=("reason", "balanced", "fast"))
    if not creds:
        raise FirstMateError("no model to draft with: set up a tier set first")
    from captain_claw.flight_deck.bat_routes import call_timeout
    prov, budget = _provider_call(creds, temperature=0.3, default_max=max_tokens, cap=max_tokens,
                                  request_timeout=call_timeout())
    resp = await prov.complete(messages=[Message(role="system", content=system), Message(role="user", content=user)],
                               temperature=0.3, max_tokens=budget)
    return resp.content or ""


def words_of_goal(goal: dict[str, Any]) -> str:
    parts = [f"A standing job that moves this goal forward: {goal.get('title', '')}"]
    for key, label in (("detail", "About it"), ("success", "Done when"), ("metric", "Measured by")):
        if str(goal.get(key) or "").strip():
            parts.append(f"{label}: {str(goal[key]).strip()}")
    return "\n".join(parts)


async def draft(store: fms.FirstMateStore, owner_id: str, words: str, *, goal: dict[str, Any] | None = None,
                by: str = "first_mate", rephrase: bool = True) -> dict[str, Any]:
    """A new factory drafted from ``words`` (or a Course ``goal``), waiting for
    the owner's approval, with its Line card. Returns the factory's view, with
    ``cases`` (how many sealed cases came with it). ``rephrase``: the task is
    rewritten as a full charter (groups and amounts, columns, where each fact must
    come from, what to do when one isn't found); off, the charter is their words as
    written and only the rest is drafted. A list is set to run as a list funnel."""
    owner = fms.norm_user(owner_id)
    words = (words or "").strip()[:WORDS_MAX]
    if goal is not None and not words:
        words = words_of_goal(goal)
    if len(words) < 8:
        raise FirstMateError("say in a sentence or two what the factory should do")
    cap = store.settings()["max_factories"]
    if len(store.list_factories(owner)) >= cap:
        raise FirstMateError(f"you already have {cap} factories; retire one first")
    user = f"What they want:\n{words}"
    fields = None
    last_error = ""
    budget = DRAFT_TOKENS
    for _ in range(2):
        prompt = user if not last_error else (f"{user}\n\nYour last answer couldn't be used: {last_error}. Answer with "
                                              "the JSON object only, and keep your thinking short.")
        try:
            text = await _ask_model(owner, system_for(rephrase), prompt, max_tokens=budget)
        except FirstMateError:
            raise  # no model to ask: asking again changes nothing
        except Exception as exc:  # noqa: BLE001 — no model answer is no draft, never a half one
            raise FirstMateError(f"the model didn't answer: {exc}"[:300]) from None
        try:
            fields = fields_of(_parse(text))
            break
        except FirstMateError as exc:
            last_error = str(exc)
            budget = RETRY_TOKENS  # an answer that ran out of room gets more of it
    if fields is None:
        raise FirstMateError(f"couldn't draft it: {last_error}")
    if not rephrase:
        fields["charter"] = words  # theirs, whatever the model returned
    note = f"Drafted from the goal: {goal.get('title', '')}"[:fms.NOTE_MAX] if goal else (
        "Drafted from your words, rewritten as a charter" if rephrase else "Drafted from your words, the charter as written")
    f = store.create(owner, title=fields["title"], charter=fields["charter"], triggers=fields["triggers"],
                     grants=fields["grants"], checks=fields["checks"], goal_id=(goal or {}).get("id", ""),
                     runner="funnel" if fields["is_list"] else "bat", note=note, created_by=by)
    from captain_claw.flight_deck import first_mate_cases
    added = 0
    for c in fields["cases"]:
        try:
            first_mate_cases.add_case(store, f["id"], title=c["title"], input=c["input"], expect=c["expect"],
                                      sealed=True, origin="first_mate")
            added += 1
        except FirstMateError:
            continue
    view = store.submit(f["id"], by)
    from captain_claw.flight_deck import first_mate_line
    await first_mate_line.charter_drafted(view, added, words if goal is None else "")
    return {**view, "cases": added}
