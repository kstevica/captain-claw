"""Crew — who may run coding CLIs on this deck: an admin approves it, per user.

The approval lives in ``users.metadata.crew``, set by an admin on the Users
page and read fresh on every Crew call, so revoking it takes effect at once.
A non-admin has no Crew until then. An admin, and the one user of a deck with
sign-in off, may always use it: they already have a shell on this computer.

A worker has a shell of its own, as the deck's OS user, and no OS sandbox holds
it to its worktree yet (docs/agent-os-sandbox-plan.md). So approving someone
trusts them with this computer, which is what the approval dialog says.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime

from captain_claw.flight_deck.crew_harness import HARNESS_IDS

DEFAULT_PARALLEL = 2
MAX_PARALLEL = 8
_TRUSTED_PARALLEL = 4
_DECK_MAX_RUNNING = 8

NOT_APPROVED = ("Crew isn't approved for this account. An admin can approve it on the Users page.")
NOT_OFFERED = "Crew is turned off on this Flight Deck."


def offered() -> bool:
    """``FD_CREW=off`` turns Crew off for the whole deck. It also needs POSIX
    process groups, so a worker can outlive a Flight Deck restart."""
    return (os.environ.get("FD_CREW", "").strip().lower() not in ("off", "0", "false", "no")
            and os.name == "posix")


def terminal_offered() -> bool:
    """``FD_CREW_TERMINAL=off`` takes the browser terminal away from everyone on this deck."""
    return offered() and os.environ.get("FD_CREW_TERMINAL", "").strip().lower() not in ("off", "0", "false", "no")


def deck_max_running() -> int:
    try:
        return max(1, int(os.environ.get("FD_CREW_MAX_RUNNING", _DECK_MAX_RUNNING)))
    except ValueError:
        return _DECK_MAX_RUNNING


@dataclass(frozen=True)
class Grant:
    allowed: bool
    why: str = ""  # "local" · "admin" · "approved"
    harnesses: tuple[str, ...] = ()
    max_parallel: int = 0
    terminal: bool = False  # may open a shell in their projects from the browser

    @property
    def trusted(self) -> bool:
        """Already has a shell here: runs on this computer's own logins."""
        return self.why in ("local", "admin")


def _meta(user: dict) -> dict:
    try:
        meta = json.loads(user.get("metadata") or "{}")
    except (TypeError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def stored(user: dict) -> dict:
    """The approval on record, as the admin set it ({} when none)."""
    entry = _meta(user).get("crew")
    return entry if isinstance(entry, dict) else {}


def _auth_enabled() -> bool:
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    return _fd_auth_enabled()


def grant_of(user: dict | None) -> Grant:
    """What *user* may do. *user* needs ``role`` and ``metadata`` (a
    get_user_by_id row). Fails closed: anything unreadable is no approval."""
    if not offered() or not user:
        return Grant(False)
    if not _auth_enabled():
        return Grant(True, "local", HARNESS_IDS, _TRUSTED_PARALLEL, terminal_offered())
    if user.get("role") == "admin":
        return Grant(True, "admin", HARNESS_IDS, _TRUSTED_PARALLEL, terminal_offered())
    entry = stored(user)
    if entry.get("approved") is not True:
        return Grant(False)
    harnesses = tuple(h for h in HARNESS_IDS if h in (entry.get("harnesses") or ()))
    if not harnesses:
        return Grant(False)
    try:
        parallel = int(entry.get("max_parallel") or DEFAULT_PARALLEL)
    except (TypeError, ValueError):
        parallel = DEFAULT_PARALLEL
    return Grant(True, "approved", harnesses, max(1, min(parallel, MAX_PARALLEL)),
                 entry.get("terminal") is True and terminal_offered())


def clean_entry(body: dict, *, admin_id: str, previous: dict | None = None) -> dict | None:
    """An admin's approval as stored; None clears it. Which CLIs must be named:
    approving none of them approves nothing."""
    if not body.get("approved"):
        return None
    harnesses = [h for h in HARNESS_IDS if h in (body.get("harnesses") or ())]
    if not harnesses:
        raise ValueError("Choose at least one coding CLI to approve.")
    try:
        parallel = int(body.get("max_parallel") or DEFAULT_PARALLEL)
    except (TypeError, ValueError):
        raise ValueError("Parallel runs must be a number.") from None
    previous = previous or {}
    first = previous.get("approved") is not True
    return {
        "approved": True, "harnesses": harnesses, "max_parallel": max(1, min(parallel, MAX_PARALLEL)),
        "terminal": body.get("terminal") is True,
        "by": admin_id if first else previous.get("by", admin_id),
        "at": datetime.now(UTC).isoformat() if first else previous.get("at", ""),
    }
