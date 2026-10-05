"""Ignored proposals must not freeze the autonomy loop.

Before: proposals left in ``awaiting_approval`` counted toward
``max_concurrent_actions`` (2), nothing ever expired them, and the cap skip was
logged as routine (hidden on the 180s pulse) — so two unanswered proposals
silently stopped the Arbiter for good. Now pending proposals have their own cap
(``max_pending_proposals``), unanswered ones expire after ``proposal_ttl_hours``,
an expired title stays deduped so it isn't re-proposed straight away, and a full
pending queue is logged visibly as a paused loop."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw.config import AutonomousWorkConfig
from captain_claw.flight_deck import action_catalog, arbiter, autonomy, events

UID = "user-alice"
_REFLECTION = {"intentions": ["Check in with the user about the Q3 report"]}
_AUTHOR = {"host": "localhost", "port": 1, "auth": "", "name": "a"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    monkeypatch.setattr(autonomy, "_STORE", s)
    monkeypatch.setattr(autonomy, "global_defaults", lambda: AutonomousWorkConfig().model_dump())
    # Loop on, propose-only, never in quiet hours.
    s.set_overrides(UID, {"enabled": True, "autonomy_level": "propose",
                          "quiet_hours_start": 0, "quiet_hours_end": 0})
    return s


@pytest.fixture
def ranker(monkeypatch):
    """The ranking LLM proposes ``ranker.title`` (a nudge, score 0.8)."""
    import captain_claw.games.remote_provider as rp

    class _Resp:
        def __init__(self, content):
            self.content = content

    class _Provider:
        def __init__(self, **kw):
            pass

        async def complete(self, **kw):
            return _Resp(json.dumps([{"kind": "nudge", "title": _Provider.title, "rationale": "r",
                                      "risk": "low", "domain": "general", "score": 0.8}]))

    _Provider.title = "Check in about the Q3 report"
    monkeypatch.setattr(rp, "RemoteLLMProvider", _Provider)

    def _no_events():
        raise RuntimeError("no events store in this test")

    async def _no_runs(uid):
        return []

    monkeypatch.setattr(events, "get_store", _no_events)
    monkeypatch.setattr(arbiter, "_gather_active_runs", _no_runs)
    monkeypatch.setattr(action_catalog, "list_catalog", lambda **kw: [])
    return _Provider


def _run_coro(coro):
    """Run *coro* on a private loop. ``asyncio.run`` would clear the current
    event loop afterwards, breaking later tests that use ``get_event_loop()``."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _run(trigger="pulse"):
    return _run_coro(arbiter.maybe_run_arbiter(UID, _REFLECTION, _AUTHOR, [], trigger=trigger))


def _backdate(store, action_id, hours):
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with store._lock:
        store._c().execute("UPDATE autonomous_actions SET created_at = ? WHERE id = ?",
                           (ts, action_id))
        store._c().commit()


def _pending(store, title, *, hours_old=1.0):
    row = store.add_action(UID, kind="nudge", title=title, status="awaiting_approval")
    _backdate(store, row["id"], hours_old)
    return row


def test_config_defaults():
    cfg = AutonomousWorkConfig()
    assert cfg.max_pending_proposals == 5
    assert cfg.proposal_ttl_hours == 48
    assert cfg.max_concurrent_actions == 2


def test_two_ignored_proposals_no_longer_block_the_loop(store, ranker):
    _pending(store, "Old idea one")
    _pending(store, "Old idea two")
    res = _run()
    assert res["ran"] is True and res.get("proposed") == 1, res
    assert len(store.list_actions(UID, status="awaiting_approval")) == 3


def test_in_flight_work_still_hits_the_concurrency_cap(store, ranker):
    store.add_action(UID, kind="run_prompt", title="Running one", status="dispatched")
    store.add_action(UID, kind="run_prompt", title="Running two", status="queued")
    res = _run()
    assert res == {"ran": False, "reason": "concurrent-cap"}


def test_full_pending_queue_pauses_loop_and_says_so_on_a_pulse(store, ranker):
    for i in range(5):
        _pending(store, f"Idea {i}")
    res = _run(trigger="pulse")
    assert res == {"ran": False, "reason": "pending-cap"}
    log = store.list_log(UID)
    assert log[0]["event"] == "loop paused: 5 proposals awaiting approval"
    assert log[0]["level"] == "warn"
    # A paused loop on the 180s pulse shouldn't flood the trace (it's capped at
    # 500 rows): the same pause isn't re-logged back to back.
    _run(trigger="pulse")
    paused = [r for r in store.list_log(UID) if r["event"].startswith("loop paused")]
    assert len(paused) == 1


def test_stale_proposals_expire_before_the_caps(store, ranker):
    stale = [_pending(store, f"Stale {i}", hours_old=49) for i in range(5)]
    fresh = _pending(store, "Fresh idea", hours_old=1)
    res = _run()
    # The sweep cleared the queue first, so the pass ran instead of pausing.
    assert res["ran"] is True and res.get("proposed") == 1, res
    for row in stale:
        got = store.get_action(row["id"])
        assert got["status"] == "expired" and got["completed_at"]
    assert store.get_action(fresh["id"])["status"] == "awaiting_approval"
    assert any(r["event"] == "expired stale proposals" for r in store.list_log(UID))


def test_expired_title_is_not_immediately_re_proposed(store, ranker):
    # Proposed 49h ago (outside the 24h lookback by created_at), expired now.
    _pending(store, ranker.title, hours_old=49)
    res = _run(trigger="manual")
    assert res["reason"] == "nothing-viable", res
    assert [r for r in store.list_log(UID) if r["event"] == "dropped: already proposed"]
    statuses = [a["status"] for a in store.list_actions(UID)]
    assert statuses == ["expired"]


def test_store_expiry_is_scoped_to_user_and_status(store):
    mine = store.add_action(UID, kind="nudge", title="mine", status="awaiting_approval")
    theirs = store.add_action("user-bob", kind="nudge", title="theirs", status="awaiting_approval")
    running = store.add_action(UID, kind="nudge", title="running", status="dispatched")
    for row in (mine, theirs, running):
        _backdate(store, row["id"], 100)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    assert store.expire_stale_proposals(UID, cutoff) == 1
    assert store.get_action(mine["id"])["status"] == "expired"
    assert store.get_action(theirs["id"])["status"] == "awaiting_approval"
    assert store.get_action(running["id"])["status"] == "dispatched"
