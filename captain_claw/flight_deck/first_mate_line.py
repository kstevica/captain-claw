"""First Mate on the Line: one thread per factory (``fm:<factory id>``).

See docs/first-mate-plan.md §11 and §14.

* **A run that finished** (done or failed) is a Runs item: not "needs you",
  not sent to phones. A rehearsal that would do something is a **proposal**
  instead: "needs you", to phones, settled once it's approved, rejected or
  replaced by a newer rehearsal. Bat's own "Bat finished" item is skipped for these runs
  (``bat_routes._bat_on_finish``), so each run is told once. A run stopped by
  its owner says nothing.
* **A factory First Mate paused** after a failure streak is "needs you". It
  reaches the owner's phones at their needs-you level. Over budget, it's a
  plain item, since it runs again once it has tokens.
* **Resuming a factory** settles its "needs you" item.

Every call here never raises: a post is news, never a reason for a run or a
scan to fail.
"""

from __future__ import annotations

import logging
from typing import Any

from captain_claw.flight_deck import first_mate_store as fms

log = logging.getLogger(__name__)

SOURCE = "First Mate"
EXCERPT_MAX = 1_200


def thread_of(factory_id: str) -> str:
    return f"fm:{factory_id}"


def _excerpt(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= EXCERPT_MAX else text[:EXCERPT_MAX].rstrip() + "…"


async def run_finished(run: dict[str, Any]) -> None:
    """A finished run on its factory's thread. A rehearsal that would do
    something is a proposal: "needs you", with Approve on First Mate."""
    if run.get("status") not in ("done", "failed") or (run.get("kind") == "check" and run.get("verdict") != "leak"):
        return  # a test case is told once, when its check ends; one that reached outside, now
    try:
        from captain_claw.flight_deck import line
        store = fms.get_store()
        factory = store.get(run["factory_id"])
        name = factory["title"] if factory else "A factory"
        kind = run.get("kind")
        needs_you, ref_type, to_phones = False, "first_mate_run", False
        if run["status"] == "done" and kind == "rehearsal":
            waiting = [i for i in store.intents(run["id"]) if i["status"] == "open"]
            if waiting:
                title = f"{name} would do {len(waiting)} {'thing' if len(waiting) == 1 else 'things'}"
                body = (_would(waiting) + "\n\nApprove it on First Mate to have it carried out once, "
                        "or reject it.\n\n" + _excerpt(run.get("result", ""))).strip()
                needs_you, ref_type, to_phones = True, "first_mate_proposal", True
            else:
                title = f"{name}: rehearsed"
                body = (_excerpt(run.get("result", "")) or "It finished, with nothing to show.") + \
                    "\n\n" + _would([])
        elif run["status"] == "done":
            title = f"{name}: carried out" if kind == "carry_out" else f"{name}: done"
            body = _excerpt(run.get("result", "")) or "It finished, with nothing to show."
        else:
            title = f"{name}: the carry-out failed" if kind == "carry_out" else f"{name}: the run failed"
            body = run.get("error") or "It stopped without saying why."
            if run.get("result"):
                body += "\n\n" + _excerpt(run["result"])
        body += "\n\nOpen First Mate to see the whole run."
        await line.post(run["owner_id"], "run", title, body, bell_type="first_mate_run", source=SOURCE,
                        needs_you=needs_you, ref_type=ref_type, ref_id=run["id"], to_phones=to_phones,
                        thread_id=thread_of(run["factory_id"]))
    except Exception:  # noqa: BLE001
        log.debug("first mate: run post failed", exc_info=True)


async def proof_doubt(factory: dict[str, Any], run: dict[str, Any], note: str) -> None:
    """A carry-out the record doesn't bear out: "needs you", to phones."""
    try:
        from captain_claw.flight_deck import line
        trust = {"propose": "asks first", "commit": "acts within its grants", "direct": "acts directly"}
        body = (f"{note}\n\nIts trust is now: {trust.get(factory['trust'], factory['trust'])}. "
                "Open First Mate to see what the carry-out did.")
        await line.post(factory["owner_id"], "run", f"{factory['title']}: check its carry-out", body,
                        bell_type="first_mate_proof", source=SOURCE, needs_you=True, ref_type="first_mate_proof",
                        ref_id=run["id"], to_phones=True, thread_id=thread_of(factory["id"]))
    except Exception:  # noqa: BLE001
        log.debug("first mate: proof post failed", exc_info=True)


async def proofs_seen(owner_id: str, run_ids: list[str]) -> None:
    """The owner looked at their factory's trust again: its doubts leave Needs you."""
    try:
        from captain_claw.flight_deck import line
        for rid in run_ids:
            await line.settle(fms.norm_user(owner_id), "first_mate_proof", rid)
    except Exception:  # noqa: BLE001
        log.debug("first mate: proof settle failed", exc_info=True)


async def _check_post(factory: dict[str, Any], title: str, body: str, *, needs_you: bool) -> None:
    try:
        from captain_claw.flight_deck import line
        await line.post(factory["owner_id"], "run", title, body + "\n\nOpen First Mate to see its cases.",
                        bell_type="first_mate_check", source=SOURCE, needs_you=needs_you,
                        ref_type="first_mate_factory", ref_id=factory["id"], to_phones=needs_you,
                        thread_id=thread_of(factory["id"]))
    except Exception:  # noqa: BLE001
        log.debug("first mate: check post failed", exc_info=True)


async def check_passed(factory: dict[str, Any], version: int, note: str) -> None:
    await _check_post(factory, f"{factory['title']}: version {version} is live", note, needs_you=False)


async def check_held(factory: dict[str, Any], version: int, note: str) -> None:
    await _check_post(factory, f"{factory['title']}: version {version} didn't go live", note, needs_you=True)


async def check_dropped(factory: dict[str, Any], score: float, previous: float) -> None:
    await _check_post(factory, f"{factory['title']} scores lower on its cases",
                      f"This week it scored {score:.0%} on its cases, down from {previous:.0%}.", needs_you=False)


def _cut(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def charter_ref(factory_id: str, sha: str) -> str:
    """A charter card's ref: the factory and the start of the draft's hash, so
    the card approves only the draft it showed."""
    return f"{factory_id}:{(sha or '')[:12]}"


async def charter_drafted(factory: dict[str, Any], cases: int, words: str = "") -> None:
    """A drafted factory waits for the owner: one card, approved against its hash."""
    draft = factory.get("draft") or {}
    if not draft:
        return
    grants = [w for k, w in (("email", "send mail"), ("organize", "organize mail")) if draft["grants"].get(k)]
    starts = [t["schedule"] for t in draft.get("triggers") or [] if t.get("kind") == "schedule"]
    starts += ["on new mail" for t in draft.get("triggers") or [] if t.get("kind") == "events"]
    parts = []
    if words:
        parts += [f"You asked: {_cut(words, 300)}", ""]
    parts += ["**What it does, each run**", _cut(draft["charter"], 900), ""]
    if draft["checks"].get("contract"):
        parts += ["**Done when**"] + [f"- {c}" for c in draft["checks"]["contract"]] + [""]
    parts.append(f"**Starts:** {', '.join(starts) if starts else 'only when you press Run now'}")
    parts.append(f"**May:** {', '.join(grants) if grants else 'nothing outward: it asks before anything leaves'}")
    if cases:
        parts.append(f"**Test cases:** {cases}, sealed (it never sees them).")
    parts += ["", "Approve it to start it, or open it on First Mate to change it first. "
                  "It asks first: nothing goes out without your yes."]
    try:
        from captain_claw.flight_deck import line
        await line.post(factory["owner_id"], "proposal", f"A new factory to approve: {factory['title']}",
                        "\n".join(parts), bell_type="first_mate_charter", source=SOURCE, needs_you=True,
                        ref_type="first_mate_charter", ref_id=charter_ref(factory["id"], draft["sha"]),
                        to_phones=False, thread_id=thread_of(factory["id"]))
    except Exception:  # noqa: BLE001
        log.debug("first mate: charter post failed", exc_info=True)


def _score(point: dict[str, Any] | None, part: str) -> float | None:
    s = ((point or {}).get(part) or {}).get("stats") or {}
    return s.get("mean")


async def improvement_drafted(factory: dict[str, Any], run: dict[str, Any], kept: list[str]) -> None:
    """An improvement run's best is a draft: one card, approved against its hash.
    Approving it checks it on the cases against the live version first."""
    draft = factory.get("draft") or {}
    if not draft:
        return
    before, after = _score(run.get("baseline"), "dev"), _score(run.get("champion"), "dev")
    parts = [f"Captain Swarm tried {run.get('experiments', 0)} ideas on its cases"
             + (f": the best scored {after:.0%}, against {before:.0%} before" if before is not None
                and after is not None else "") + ", and the held-out cases confirmed it.", ""]
    if kept:
        parts += ["**What changed**"] + [f"- {_cut(k, 200)}" for k in kept[:5]] + [""]
    parts += ["**The new charter**", _cut(draft["charter"], 900), ""]
    steps = (draft.get("runner_config") or {}).get("steps") or []
    if steps:
        parts += ["**Steps:** " + "; ".join(_cut(x, 80) for x in steps[:8]), ""]
    parts.append("Approving it checks it on its cases against the live version first: it goes live only if it "
                 "holds up. Its grants and when it starts don't change.")
    try:
        from captain_claw.flight_deck import line
        await line.post(factory["owner_id"], "proposal", f"{factory['title']}: a better charter to approve",
                        "\n".join(parts), bell_type="first_mate_improvement", source=SOURCE, needs_you=True,
                        ref_type="first_mate_charter", ref_id=charter_ref(factory["id"], draft["sha"]),
                        to_phones=False, thread_id=thread_of(factory["id"]))
    except Exception:  # noqa: BLE001
        log.debug("first mate: improvement post failed", exc_info=True)


async def improvement_ended(factory: dict[str, Any], run: dict[str, Any], *, kept: bool, note: str) -> None:
    """An improvement run ended with nothing for the owner to approve: it found
    nothing better, or its best couldn't become a draft (and why)."""
    if kept:
        title = f"{factory['title']}: a better charter waits"
        body = (f"The improvement run found a better charter, but it couldn't become a draft: {note.rstrip('.')}. "
                "Then open First Mate and make it a draft from the factory's Checks tab.")
    else:
        title = f"{factory['title']}: the improvement run found nothing better"
        body = (f"Captain Swarm tried {run.get('experiments', 0)} ideas on its cases and kept none"
                + (f" ({note.rstrip('.')})" if note else "") + ". The factory stays as it is.")
    try:
        from captain_claw.flight_deck import line
        await line.post(factory["owner_id"], "run", title, body, bell_type="first_mate_improvement", source=SOURCE,
                        needs_you=kept, ref_type="first_mate_factory", ref_id=factory["id"], to_phones=False,
                        thread_id=thread_of(factory["id"]))
    except Exception:  # noqa: BLE001
        log.debug("first mate: improvement post failed", exc_info=True)


async def charter_settled(owner_id: str, factory_id: str, sha: str) -> None:
    """The draft a card showed was approved, declined or changed: it leaves Needs you."""
    try:
        from captain_claw.flight_deck import line
        await line.settle(fms.norm_user(owner_id), "first_mate_charter", charter_ref(factory_id, sha))
    except Exception:  # noqa: BLE001
        log.debug("first mate: charter settle failed", exc_info=True)


_TRUST_WORDS = {"commit": "acts within its grants", "direct": "acts directly"}
_TRUST_HOW = {"commit": "It will carry out what its grants allow without asking you first, and ask about the rest.",
              "direct": "It won't rehearse any more: it does its work itself."}


async def trust_raised(factory: dict[str, Any], record: dict[str, Any], calibration: dict[str, Any]) -> None:
    """The record moved a factory up a level: the owner hears it, with why."""
    level = factory["trust"]
    body = (f"Its last {record['counted']} results scored {record['value']:.0%}, with no doubtful carry-out, and "
            f"its own checks agree with your labels (kappa {calibration['kappa']:.2f} on {calibration['n']}). "
            f"{_TRUST_HOW.get(level, '')}\n\nTo keep it asking first, set its trust back on First Mate and pin it.")
    try:
        from captain_claw.flight_deck import line
        await line.post(factory["owner_id"], "run", f"{factory['title']} now {_TRUST_WORDS.get(level, level)}", body,
                        bell_type="first_mate_trust", source=SOURCE, needs_you=False,
                        ref_type="first_mate_factory", ref_id=factory["id"], to_phones=True,
                        thread_id=thread_of(factory["id"]))
    except Exception:  # noqa: BLE001
        log.debug("first mate: trust post failed", exc_info=True)


async def proposal_settled(run: dict[str, Any] | None) -> None:
    """A proposal answered (approved, rejected) or replaced by a newer
    rehearsal: it leaves Needs you."""
    if not run:
        return
    try:
        from captain_claw.flight_deck import line
        await line.settle(fms.norm_user(run["owner_id"]), "first_mate_proposal", run["id"])
    except Exception:  # noqa: BLE001
        log.debug("first mate: proposal settle failed", exc_info=True)


WOULD_SHOWN = 8


def _would(intents: list[dict[str, Any]]) -> str:
    """A rehearsal's intents in words, for its post."""
    if not intents:
        return "It needed nothing outside."
    lines = [f"- {i['what']}" for i in intents[:WOULD_SHOWN]]
    if len(intents) > WOULD_SHOWN:
        lines.append(f"- and {len(intents) - WOULD_SHOWN} more")
    return "It would have done this, and nothing of it was done (a rehearsal):\n" + "\n".join(lines)


async def factory_paused(factory: dict[str, Any], reason: str, *, needs_you: bool) -> None:
    try:
        from captain_claw.flight_deck import line
        if needs_you:
            body = (f"First Mate paused it: {reason}. Open First Mate to see what went wrong, "
                    f"then resume it when it's fixed.")
        else:
            body = f"It {reason}, so it waits until it has tokens again: next Monday, or sooner if you give it a bigger share."
        await line.post(factory["owner_id"], "run", f"{factory['title']} is paused", body,
                        bell_type="first_mate_paused", source=SOURCE, needs_you=needs_you,
                        ref_type="first_mate_factory", ref_id=factory["id"], to_phones=needs_you,
                        thread_id=thread_of(factory["id"]))
    except Exception:  # noqa: BLE001
        log.debug("first mate: pause post failed", exc_info=True)


async def factory_resumed(owner_id: str, factory_id: str) -> None:
    try:
        from captain_claw.flight_deck import line
        await line.settle(fms.norm_user(owner_id), "first_mate_factory", factory_id)
    except Exception:  # noqa: BLE001
        log.debug("first mate: settle failed", exc_info=True)
