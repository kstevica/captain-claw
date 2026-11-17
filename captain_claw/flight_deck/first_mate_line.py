"""First Mate on the Line: one thread per factory (``fm:<factory id>``).

See docs/first-mate-plan.md §11 and §14.

* **A run that finished** (done or failed) is a Runs item: not "needs you",
  not sent to phones. Bat's own "Bat finished" item is skipped for these runs
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
    if run.get("status") not in ("done", "failed"):
        return
    try:
        from captain_claw.flight_deck import line
        factory = fms.get_store().get(run["factory_id"])
        name = factory["title"] if factory else "A factory"
        if run["status"] == "done":
            title = f"{name}: done"
            body = _excerpt(run.get("result", "")) or "It finished, with nothing to show."
        else:
            title = f"{name}: the run failed"
            body = run.get("error") or "It stopped without saying why."
            if run.get("result"):
                body += "\n\n" + _excerpt(run["result"])
        body += "\n\nOpen First Mate to see the whole run."
        await line.post(run["owner_id"], "run", title, body, bell_type="first_mate_run", source=SOURCE,
                        needs_you=False, ref_type="first_mate_run", ref_id=run["id"], to_phones=False,
                        thread_id=thread_of(run["factory_id"]))
    except Exception:  # noqa: BLE001
        log.debug("first mate: run post failed", exc_info=True)


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
