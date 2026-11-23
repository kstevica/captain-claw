"""A robot check in the person's Chrome, handed to them — and the agent waits.

An agent's web search or web_fetch through the person's own Chrome
(``browser_search.py``, ``browser_fetch.py``) can land on a site asking to
prove a person is there: Google's "unusual traffic" page, Cloudflare,
reCAPTCHA, hCaptcha, DataDome, PerimeterX … Captain for Chrome then brings
that tab to the front instead of closing it (``lib/robot-check.js``) and says
so over its socket::

  ← {"type": "waiting", id, site, vendor, why, for_s}      the person is asked
  ← {"type": "waiting", id, site, over: "solved" | "timeout" | "closed" |
                                        "cancelled" | "declined", waited_s}

Once the person answered it, the search or read goes on in that tab. That can
take minutes — longer than one request of the agent's — so a search or read
that may wait runs as a **job**: the agent's first request (with ``job``, an id
it made) waits for the answer as before, but returns ``{"pending": id,
"waiting": {...}}`` as soon as a check is handed over (its chat says it's
waiting for the person), and the agent follows it (``/wait/follow``) until it's
done. Without ``job`` (an agent from before this) nothing is handed over: the
check sends the agent to its API engine or its own fetch, as before.

How long: :data:`HAND_OVER_S` when someone is there for this turn (a chat),
:data:`HAND_OVER_UNATTENDED_S` for a run that started on its own (they may be
at the computer anyway). Not answered in time, the agent goes on without it —
and the tab stays open for the person. A job the agent stopped following ends
(its act is cancelled; the tab stays).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

HAND_OVER_S = 300.0            # a person in the chat: five minutes to answer it
HAND_OVER_UNATTENDED_S = 120.0   # a run that started on its own
FOLLOW_GAP_S = 75.0            # the agent asks again within this, or its job ends
DONE_KEEP_S = 120.0            # a finished job's answer waits this long to be collected
FOLLOW_WAIT_MAX_S = 30.0
FIRST_WAIT_MAX_S = 60.0


def _now() -> float:
    return time.monotonic()


@dataclass(eq=False)
class Job:
    id: str
    owner: str
    slug: str
    hand_over_s: float
    task: asyncio.Task | None = None
    waiting: dict | None = None        # the check the person is asked to answer now
    announced: bool = False            # the agent heard about it
    touched: float = field(default_factory=_now)
    following: int = 0                 # the agent's requests open on it now
    done_at: float = 0.0
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    def stale(self) -> bool:
        """The agent stopped following it (gone, or its turn stopped)."""
        return not self.following and _now() - self.touched > FOLLOW_GAP_S


_JOBS: dict[str, Job] = {}
_CURRENT: contextvars.ContextVar[Job | None] = contextvars.ContextVar("browser_wait_job", default=None)


def current() -> Job | None:
    """The job a search or read runs in — None for an agent's plain request."""
    return _CURRENT.get()


def hand_over_s() -> float:
    """Seconds the extension may wait for the person on a check: 0 = don't hand it over."""
    job = current()
    return job.hand_over_s if job else 0.0


def reset_for_tests() -> None:
    for job in list(_JOBS.values()):
        if job.task and not job.task.done():
            job.task.cancel()
    _JOBS.clear()


def _reap() -> None:
    now = _now()
    for job in list(_JOBS.values()):
        if job.done_at and now - job.done_at > DONE_KEEP_S:
            _JOBS.pop(job.id, None)
        elif not job.done_at and job.stale() and job.task and not job.task.done():
            job.task.cancel()


async def run(owner: str, slug: str, job_id: str, attended: bool,
              work: Callable[[], Awaitable[dict]], wait_s: float) -> dict:
    """Start *work* as job *job_id* and answer as :func:`follow` does."""
    _reap()
    known = _JOBS.get(job_id)
    if known is not None:
        if known.owner != owner or known.slug != slug:
            return {"ok": False, "why": "page", "error": "That request id is taken."}
        return await _answer(known, wait_s)
    job = Job(id=job_id, owner=owner, slug=slug,
              hand_over_s=HAND_OVER_S if attended else HAND_OVER_UNATTENDED_S)
    token = _CURRENT.set(job)
    try:
        job.task = asyncio.create_task(work())   # carries the job in its context
    finally:
        _CURRENT.reset(token)
    job.task.add_done_callback(lambda _t: _finished(job))
    _JOBS[job.id] = job
    return await _answer(job, wait_s)


def _finished(job: Job) -> None:
    job.done_at = _now()
    job.changed.set()


async def follow(job_id: str, owner: str, slug: str, wait_s: float) -> dict | None:
    """The job's answer, or ``{"pending", "waiting"}`` after *wait_s* — None
    when there's no such job (ended a while ago, or not this agent's)."""
    _reap()
    job = _JOBS.get(job_id)
    if job is None or job.owner != owner or job.slug != slug:
        return None
    return await _answer(job, min(max(wait_s, 1.0), FOLLOW_WAIT_MAX_S))


async def cancel(job_id: str, owner: str, slug: str) -> bool:
    job = _JOBS.get(job_id)
    if job is None or job.owner != owner or job.slug != slug:
        return False
    _JOBS.pop(job_id, None)
    if job.task and not job.task.done():
        job.task.cancel()
        return True
    return False


async def _answer(job: Job, wait_s: float) -> dict:
    job.following += 1
    try:
        return await _answer_now(job, wait_s)
    finally:
        job.following -= 1
        job.touched = _now()


async def _answer_now(job: Job, wait_s: float) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + min(max(wait_s, 1.0), FIRST_WAIT_MAX_S)
    while True:
        assert job.task is not None
        if job.task.done():
            _JOBS.pop(job.id, None)
            try:
                return job.task.result()
            except asyncio.CancelledError:
                return {"ok": False, "why": "page", "error": "Stopped."}
            except Exception as exc:   # noqa: BLE001 — the agent goes on its own way
                log.warning("A wait in the person's browser failed", exc_info=True)
                return {"ok": False, "why": "page", "error": f"It failed in Flight Deck ({type(exc).__name__})."}
        if job.waiting and not job.announced:
            job.announced = True   # the agent says so in its chat now, then follows
            return _pending(job)
        left = deadline - loop.time()
        if left <= 0:
            return _pending(job)
        job.changed.clear()
        try:
            await asyncio.wait_for(job.changed.wait(), left)
        except TimeoutError:
            pass


def _pending(job: Job) -> dict:
    return {"pending": job.id, "waiting": dict(job.waiting) if job.waiting else None}


# ── the extension's word about an act ────────────────────────────────


def note(act: Any, msg: dict, page_wait_s: float) -> None:
    """``{"type": "waiting"}`` for *act*: the person is asked (the act may take
    as long as they're given, plus a page's load after), or it's over."""
    job = getattr(act, "job", None)
    if job is None:
        return   # nothing was handed over for this act (no job asked for it)
    loop = asyncio.get_running_loop()
    site = str(msg.get("site") or "")[:120]
    if "over" in msg:
        outcome = str(msg.get("over") or "")[:20]
        act.wait_until = loop.time() + page_wait_s   # answered: the page loads, the read goes on
        if not msg.get("joined"):
            was = act.robot or {}
            act.robot = {"site": site or was.get("site", ""), "vendor": was.get("vendor", ""), "outcome": outcome}
        if job.waiting and job.waiting.get("act") == act.id:
            job.waiting = None
        job.changed.set()
        return
    try:
        for_s = float(msg.get("for_s") or 0)
    except (TypeError, ValueError):
        for_s = 0.0
    for_s = max(0.0, min(for_s, job.hand_over_s + 5))
    act.wait_until = loop.time() + for_s + page_wait_s
    if not msg.get("joined"):   # (joined: waits on another search's check of that site, already in front)
        act.robot = {"site": site, "vendor": str(msg.get("vendor") or "")[:40], "outcome": "waiting"}
    # Either way the agent waits on the person — and hears so.
    job.waiting = {"act": act.id, "site": site, "vendor": str(msg.get("vendor") or "")[:40],
                   "why": str(msg.get("why") or "")[:200], "for_s": int(for_s), "since": time.time()}
    job.changed.set()


async def await_act(act: Any, wait_s: float, cancel: Callable[[], Awaitable[None]]) -> dict | None:
    """The browser's answer to *act* within *wait_s* — or longer while the
    person is asked to answer a check (:func:`note`). None on no answer in time
    (the browser is told to cancel)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_s
    try:
        while True:
            left = max(deadline, getattr(act, "wait_until", 0.0)) - loop.time()
            job = getattr(act, "job", None)
            if left <= 0 or (job is not None and job.stale()):
                break
            try:
                return await asyncio.wait_for(asyncio.shield(act.future), min(left, 1.0))
            except TimeoutError:
                continue
    except asyncio.CancelledError:
        await cancel()
        raise
    await cancel()
    return None
