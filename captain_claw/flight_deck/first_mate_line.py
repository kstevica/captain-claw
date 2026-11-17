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
