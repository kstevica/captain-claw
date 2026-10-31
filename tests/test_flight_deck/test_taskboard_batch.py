"""Several tasks at once on the Taskboard: one set of words handed to one agent, which
takes them strictly one at a time, each with its own details. What a task is by its
turn decides whether it runs; a line survives a deck restart without repeating a
run; the user can stop the rest; one correction reaches several tasks at once."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import auth, taskboard, taskboard_triage
from captain_claw.flight_deck import taskboard_handover as handover

from .test_taskboard_nextsteps import ALICE, _task, board, jev, ledger, model, told  # noqa: F401

HELPER = {"slug": "helper", "name": "Helper", "asleep": False}
OTHER = {"slug": "writer", "name": "Writer", "asleep": True}
WORDS = "Thank them and politely decline."


@pytest.fixture()
def relay(monkeypatch):
    """The agent: each run waits at its own gate when ``hold`` says so, and is
    recorded when it starts and ends."""
    state = SimpleNamespace(calls=[], running=0, most=0, hold=False, gates=[], reply="Declined it, politely.",
                            meta={"task_session": {"id": "s-1", "name": "x"}})

    async def run(agent, prompt, automation):
        state.calls.append({"agent": agent, "prompt": prompt, "automation": automation})
        state.running += 1
        state.most = max(state.most, state.running)
        try:
            if state.hold:
                gate = asyncio.Event()
                state.gates.append(gate)
                await gate.wait()
            else:
                await asyncio.sleep(0)
            return state.reply, state.meta
        finally:
            state.running -= 1

    monkeypatch.setattr(handover, "_relay", run)
    monkeypatch.setattr(handover, "agents_for", lambda uid: [HELPER, OTHER])
    monkeypatch.setattr(handover, "_RUNNING", {})
    monkeypatch.setattr(handover, "_LINES", {})
    monkeypatch.setattr(handover, "_LINE_AGAIN", set())

    async def spent(uid):
        return False

    monkeypatch.setattr(taskboard_triage, "_quota_spent", spent)
    return state


async def _settle():
    """Until every line has been worked through."""
    for _ in range(200):
        lines = list(handover._LINES.values())
        if not lines:
            return
        await asyncio.wait(lines, timeout=2)
    raise AssertionError("a line never ended")


async def _started(relay, n):  # noqa: F811
    """Until the agent has been given *n* runs."""
    for _ in range(500):
        if len(relay.calls) >= n:
            return
        await asyncio.sleep(0.002)
    raise AssertionError(f"only {len(relay.calls)} of {n} started")


def _tasks(board, n=3):  # noqa: F811
    ids = [_task(board, flags={}, pending=False) for _ in range(n)]
    for i, t in enumerate(ids):
        board.update_fields(ALICE, t, {"title": f"Invitation {i}", "owes_next": "me"})
    return ids


def _lane(board, t):  # noqa: F811
    return board.lane_of(ALICE, t)


def _said(board, t):  # noqa: F811
    return [e["summary"] for e in board.events(ALICE, t)]


# ── One set of words, one at a time ───────────────────────────────────


async def test_they_run_in_the_order_given_and_never_two_at_once(board, ledger, relay):  # noqa: F811
    ts = _tasks(board)
    res = await handover.hand_over_many(ALICE, [ts[2], ts[0], ts[1]], "helper", WORDS, actor="Ada")
    assert [r["id"] for r in res["queued"]] == [ts[2], ts[0], ts[1]] and res["refused"] == []
    await _settle()
    assert relay.most == 1, "one at a time"
    assert [c["automation"]["task"]["id"] for c in relay.calls] == [ts[2], ts[0], ts[1]]


async def test_the_next_waits_until_the_last_has_ended(board, ledger, relay):  # noqa: F811
    relay.hold = True
    ts = _tasks(board, 2)
    await handover.hand_over_many(ALICE, ts, "helper", WORDS)
    await _started(relay, 1)
    await asyncio.sleep(0.01)
    assert len(relay.calls) == 1, "the second doesn't start while the first runs"
    relay.gates[0].set()
    await _started(relay, 2)
    relay.gates[1].set()
    await _settle()
    assert [t for t in ts if board.get_task(ALICE, t)["owes_next"] == "me"] == ts


async def test_each_gets_the_same_words_and_only_its_own_details(board, ledger, relay):  # noqa: F811
    ts = _tasks(board, 2)
    await handover.hand_over_many(ALICE, ts, "helper", WORDS)
    await _settle()
    first, second = relay.calls
    assert WORDS in first["prompt"] and WORDS in second["prompt"]
    assert "Invitation 0" in first["prompt"] and "Invitation 1" not in first["prompt"]
    assert "Invitation 1" in second["prompt"] and "Invitation 0" not in second["prompt"]
    for call in (first, second):
        auto = call["automation"]
        assert auto["job_text"] == WORDS and auto["mail_write"] == "allow" and auto["kind"] == "taskboard"
    assert first["automation"]["ref"] != second["automation"]["ref"], "an action of its own"
    assert first["automation"]["task"]["id"] != second["automation"]["task"]["id"], "a session of its own"
    head, _, _ = first["prompt"].partition("<<<DATA")
    assert "Invitation" not in head, "what the task says stays fenced as data"


async def test_they_leave_needs_me_at_once_and_say_where_they_stand(board, ledger, relay):  # noqa: F811
    relay.hold = True
    ts = _tasks(board)
    assert {_lane(board, t) for t in ts} == {"needs_me"}
    await handover.hand_over_many(ALICE, ts, "writer", WORDS)
    assert {_lane(board, t) for t in ts} == {"working"}
    assert all(f"Handed to Writer: {WORDS}" in _said(board, t) for t in ts)
    await _started(relay, 1)
    assert board.line_places(ALICE) == {ts[1]: {"agent": "Writer", "ahead": 1}, ts[2]: {"agent": "Writer", "ahead": 2}}
    assert any(s == "Writer started on what you handed over" for s in _said(board, ts[0]))
    assert not any(s == "Writer started on what you handed over" for s in _said(board, ts[1]))
    for gate_at in range(3):
        await _started(relay, gate_at + 1)
        relay.gates[gate_at].set()
    await _settle()
    assert board.line_places(ALICE) == {}


async def test_each_comes_back_to_the_user_with_its_own_answer(board, ledger, relay):  # noqa: F811
    ts = _tasks(board, 2)
    await handover.hand_over_many(ALICE, ts, "helper", WORDS)
    await _settle()
    for t in ts:
        row = board.get_task(ALICE, t)
        assert row["owes_next"] == "me" and row["result_text"] == "Declined it, politely."
        assert "Helper finished what you handed over" in _said(board, t)
    actions = ledger.list_actions(ALICE)
    assert len(actions) == 2 and {a["outcome"] for a in actions} == {"success"}


async def test_a_run_that_fails_stops_nothing_else(board, ledger, relay, monkeypatch):  # noqa: F811
    ts = _tasks(board, 3)
    seen = []

    async def run(agent, prompt, automation):
        seen.append(automation["task"]["id"])
        if len(seen) == 1:
            raise RuntimeError("boom")
        if len(seen) == 2:
            return None, {"error": "it isn't running"}
        return "Done.", {}

    monkeypatch.setattr(handover, "_relay", run)
    await handover.hand_over_many(ALICE, ts, "helper", WORDS)
    await _settle()
    assert seen == ts, "every one had its turn"
    assert any("didn't finish" in s for s in _said(board, ts[0]))
    assert "Helper didn't finish what you handed over: it isn't running" in _said(board, ts[1])
    assert board.get_task(ALICE, ts[2])["result_text"] == "Done."
    batch = board.line_batches(ALICE, ended_since=0)[0]
    assert (batch["done"], batch["failed"], batch["waiting"]) == (1, 2, 0)


# ── What can't go ─────────────────────────────────────────────────────


async def test_what_cant_go_is_refused_and_the_rest_goes(board, ledger, relay):  # noqa: F811
    ok, closed, gone = *_tasks(board, 2), "tb_not_mine"
    board.correct(ALICE, closed, "done")
    res = await handover.hand_over_many(ALICE, [closed, ok, gone], "helper", WORDS)
    assert [r["id"] for r in res["queued"]] == [ok]
    why = {r["id"]: r["why"] for r in res["refused"]}
    assert "open task" in why[closed] and "isn't on your board" in why[gone]
    await _settle()
    assert [c["automation"]["task"]["id"] for c in relay.calls] == [ok]


async def test_someone_elses_task_isnt_touched(board, ledger, relay):  # noqa: F811
    t = _tasks(board, 1)[0]
    with pytest.raises(ValueError, match="isn't on your board"):
        await handover.hand_over_many("someone-else", [t], "helper", WORDS)
    assert relay.calls == [] and board.get_task(ALICE, t)["owes_next"] != "agent"


async def test_already_in_line_isnt_handed_again(board, ledger, relay):  # noqa: F811
    relay.hold = True
    a, b = _tasks(board, 2)
    await handover.hand_over_many(ALICE, [a, b], "helper", WORDS)
    await _started(relay, 1)
    with pytest.raises(ValueError, match="still working|already in line"):
        await handover.hand_over(ALICE, a, "writer", "Something else.")
    with pytest.raises(ValueError, match="already in line for Helper"):
        await handover.hand_over(ALICE, b, "writer", "Something else.")
    with pytest.raises(ValueError, match="still working"):
        await handover.hand_over_many(ALICE, [a], "writer", "Something else.")
    with pytest.raises(ValueError, match="already in line for Helper"):
        await handover.hand_over_many(ALICE, [b], "writer", "Something else.")
    relay.gates[0].set()
    await _started(relay, 2)
    relay.gates[1].set()
    await _settle()


@pytest.mark.parametrize("words", ["", "   "])
async def test_it_needs_words(board, ledger, relay, words):  # noqa: F811
    ts = _tasks(board, 2)
    with pytest.raises(ValueError, match="should do"):
        await handover.hand_over_many(ALICE, ts, "helper", words)
    assert ledger.list_actions(ALICE) == [] and board.line_unfinished() == []


async def test_only_the_users_available_agent_and_a_limit(board, ledger, relay):  # noqa: F811
    ts = _tasks(board, 2)
    with pytest.raises(ValueError, match="isn't available"):
        await handover.hand_over_many(ALICE, ts, "someone-elses", WORDS)
    with pytest.raises(ValueError, match="at most"):
        await handover.hand_over_many(ALICE, [f"t{i}" for i in range(handover.BATCH_MAX + 1)], "helper", WORDS)
    with pytest.raises(ValueError, match="Pick"):
        await handover.hand_over_many(ALICE, [], "helper", WORDS)
    assert board.line_unfinished() == []


async def test_a_spent_allowance_refuses_before_anything_moves(board, ledger, relay, monkeypatch):  # noqa: F811
    async def spent(uid):
        return True

    monkeypatch.setattr(taskboard_triage, "_quota_spent", spent)
    ts = _tasks(board, 2)
    with pytest.raises(ValueError, match="allowance"):
        await handover.hand_over_many(ALICE, ts, "helper", WORDS)
    assert board.get_task(ALICE, ts[0])["owes_next"] != "agent" and relay.calls == []


# ── By its turn ───────────────────────────────────────────────────────


async def _with_first_held(board, relay, n=2):  # noqa: F811
    relay.hold = True
    ts = _tasks(board, n)
    await handover.hand_over_many(ALICE, ts, "helper", WORDS)
    await _started(relay, 1)
    return ts


async def _let_go(relay, then_hold=False):  # noqa: F811
    relay.hold = then_hold
    for gate in relay.gates:
        gate.set()
    await _settle()


async def test_closed_before_its_turn_is_never_sent(board, ledger, relay):  # noqa: F811
    a, b = await _with_first_held(board, relay)
    board.correct(ALICE, b, "done")
    await _let_go(relay)
    assert [c["automation"]["task"]["id"] for c in relay.calls] == [a], "the agent was never told"
    assert "Helper didn't finish what you handed over: it was closed before its turn" in _said(board, b)
    assert board.get_task(ALICE, b)["status"] == "done"
    assert board.line_batches(ALICE, ended_since=0)[0]["skipped"] == 1


async def test_moved_on_by_the_user_before_its_turn_is_never_sent(board, ledger, relay):  # noqa: F811
    a, b = await _with_first_held(board, relay)
    board.correct(ALICE, b, "owes_next", "them")
    await _let_go(relay)
    assert [c["automation"]["task"]["id"] for c in relay.calls] == [a]
    assert board.get_task(ALICE, b)["owes_next"] == "them", "what the user set stands"


async def test_merged_before_its_turn_is_never_sent(board, ledger, relay):  # noqa: F811
    a, b, c = await _with_first_held(board, relay, 3)
    board.merge(ALICE, b, c)
    await _let_go(relay)
    ran = [call["automation"]["task"]["id"] for call in relay.calls]
    assert b not in ran and a in ran and c in ran


async def test_the_agent_off_by_its_turn_comes_back_saying_so(board, ledger, relay, monkeypatch):  # noqa: F811
    a, b = await _with_first_held(board, relay)
    monkeypatch.setattr(handover, "agents_for", lambda uid: [OTHER])
    await _let_go(relay)
    assert len(relay.calls) == 1
    row = board.get_task(ALICE, b)
    assert row["owes_next"] == "me" and row["status"] == "open"
    assert "Helper didn't finish what you handed over: it was off when its turn came" in _said(board, b)


async def test_the_allowance_spent_by_its_turn_comes_back_saying_so(board, ledger, relay, monkeypatch):  # noqa: F811
    a, b = await _with_first_held(board, relay)

    async def spent(uid):
        return True

    monkeypatch.setattr(taskboard_triage, "_quota_spent", spent)
    await _let_go(relay)
    assert len(relay.calls) == 1
    assert any("allowance" in s for s in _said(board, b)) and board.get_task(ALICE, b)["owes_next"] == "me"


async def test_what_came_in_meanwhile_reaches_the_agent(board, ledger, relay):  # noqa: F811
    """Its details are read when its turn comes, not when it was handed over."""
    a, b = await _with_first_held(board, relay)
    board.ingest(ALICE, keys=[board.links(ALICE, [b])[b][0]["key"]], event={
        "source": "agent_turns", "source_ref": "late", "at": time.time(), "kind": "received", "actor_kind": "party",
        "actor": "Ana", "summary": "Actually we moved the date"})
    await _let_go(relay)
    assert "Actually we moved the date" in relay.calls[1]["prompt"]


async def test_triage_leaves_it_alone_while_in_line_and_reads_it_after(board, ledger, relay):  # noqa: F811
    """Judged by what it was before the user handed it over, it could close before its turn."""
    a, b = await _with_first_held(board, relay)
    for t in (a, b):
        board.update_fields(ALICE, t, {"triage_state": "pending", "triage_after": 1.0})
    assert board.triage_queue(ALICE) == [] and board.triage_users() == []
    await _let_go(relay)
    assert {r["id"] for r in board.triage_queue(ALICE)} == {a, b} and board.triage_users() == [ALICE]


# ── Stopping the rest ─────────────────────────────────────────────────


async def test_stopping_the_rest_keeps_them_from_running_and_brings_them_back(board, ledger, relay):  # noqa: F811
    a, b, c = await _with_first_held(board, relay, 3)
    batch_id = board.line_batches(ALICE, ended_since=0)[0]["id"]
    assert await handover.stop_batch(ALICE, batch_id) == 2
    for t in (b, c):
        row = board.get_task(ALICE, t)
        assert row["owes_next"] == "me" and row["status"] == "open"
        assert f"Helper didn't finish what you handed over: {handover._STOPPED_NOTE}" in _said(board, t)
    await _let_go(relay)
    assert [call["automation"]["task"]["id"] for call in relay.calls] == [a], "only the one it was on ran"
    assert board.get_task(ALICE, a)["result_text"] == "Declined it, politely.", "that one still finishes"
    done = board.line_batches(ALICE, ended_since=0)[0]
    assert (done["done"], done["stopped"], done["waiting"], done["on"]) == (1, 2, 0, None) and done["ended_at"]
    assert {a2["outcome"] for a2 in ledger.list_actions(ALICE)} == {"success", "fail"}


async def test_someone_else_cant_stop_it(board, ledger, relay):  # noqa: F811
    await _with_first_held(board, relay)
    batch_id = board.line_batches(ALICE, ended_since=0)[0]["id"]
    assert await handover.stop_batch("someone-else", batch_id) == 0
    assert board.line_batches(ALICE, ended_since=0)[0]["waiting"] == 1
    await _let_go(relay)


# ── Several batches ───────────────────────────────────────────────────


async def test_a_second_batch_for_the_same_agent_waits_behind_the_first(board, ledger, relay):  # noqa: F811
    relay.hold = True
    a, b, c = _tasks(board, 3)
    await handover.hand_over_many(ALICE, [a, b], "helper", WORDS)
    await handover.hand_over_many(ALICE, [c], "helper", "Something else.")
    await _started(relay, 1)
    assert board.line_places(ALICE)[c] == {"agent": "Helper", "ahead": 2}
    relay.hold = False
    relay.gates[0].set()
    await _settle()
    assert relay.most == 1 and [x["automation"]["task"]["id"] for x in relay.calls] == [a, b, c]
    assert "Something else." in relay.calls[2]["prompt"]


async def test_another_agents_line_goes_alongside(board, ledger, relay):  # noqa: F811
    relay.hold = True
    a, b = _tasks(board, 2)
    await handover.hand_over_many(ALICE, [a], "helper", WORDS)
    await handover.hand_over_many(ALICE, [b], "writer", WORDS)
    await _started(relay, 2)
    assert relay.running == 2, "one at a time per agent"
    await _let_go(relay)


async def test_a_batch_added_as_the_line_ends_isnt_stranded(board, ledger, relay, monkeypatch):  # noqa: F811
    """The worker found nothing just as more was added: it looks again."""
    a, b = _tasks(board, 2)
    real = board.line_next
    added = {"done": False}

    def next_then_more(uid, slug):
        got = real(uid, slug)
        if got is None and not added["done"]:
            added["done"] = True
            board.hand_over_started(ALICE, b, "act_late", {"slug": "helper", "name": "Helper"}, WORDS, batch_id="hb_late")
            handover._LINE_AGAIN.add((uid, slug))
        return got

    monkeypatch.setattr(board, "line_next", next_then_more)
    monkeypatch.setattr(handover.autonomy_store(), "update_status", lambda *a, **k: None)
    await handover.hand_over_many(ALICE, [a], "helper", WORDS)
    await _settle()
    assert [c["automation"]["task"]["id"] for c in relay.calls] == [a, b]


# ── A deck restart ────────────────────────────────────────────────────


async def test_a_restart_goes_on_without_repeating_what_was_running(board, ledger, relay):  # noqa: F811
    """The run the restart cut may have sent its email: it isn't sent again, and its task says so."""
    ts = _tasks(board, 3)
    res = await handover.hand_over_many(ALICE, ts, "helper", WORDS)
    assert len(res["queued"]) == 3
    # The first was on when the deck went down.
    handover._LINES[(ALICE, "helper")].cancel()
    await asyncio.gather(*handover._LINES.values(), return_exceptions=True)
    first = board.line_next(ALICE, "helper")
    assert board.line_claim(first["id"])
    relay.calls.clear()
    await handover._resume(wait=0)
    await _settle()
    assert [c["automation"]["task"]["id"] for c in relay.calls] == [ts[1], ts[2]], "not the one that was on"
    row = board.get_task(ALICE, ts[0])
    assert row["owes_next"] == "me" and row["status"] == "open"
    assert any(handover._CUT_NOTE in s for s in _said(board, ts[0]))
    batch = board.line_batches(ALICE, ended_since=0)[0]
    assert (batch["done"], batch["failed"], batch["waiting"]) == (2, 1, 0)


async def test_nothing_in_line_and_a_restart_does_nothing(board, ledger, relay):  # noqa: F811
    await handover._resume(wait=0)
    assert handover._LINES == {} and relay.calls == []


async def test_the_line_is_on_disk(board, ledger, relay, tmp_path):  # noqa: F811
    from captain_claw.flight_deck.taskboard_store import TaskboardStore

    ts = _tasks(board, 2)
    await handover.hand_over_many(ALICE, ts, "helper", WORDS)
    handover._LINES[(ALICE, "helper")].cancel()
    await asyncio.gather(*handover._LINES.values(), return_exceptions=True)
    again = TaskboardStore(tmp_path / "taskboard.db")
    try:
        assert [r["task_id"] for r in again.line_unfinished()] == ts
    finally:
        again.close()


# ── Keeping it tidy ───────────────────────────────────────────────────


async def test_a_batch_says_how_it_stands(board, ledger, relay):  # noqa: F811
    a, b = await _with_first_held(board, relay)
    going = board.line_batches(ALICE, ended_since=time.time())[0]
    assert going["total"] == 2 and going["waiting"] == 1 and going["on"]["id"] == a and going["on"]["title"] == "Invitation 0"
    assert going["agent"] == "Helper" and going["words"] == WORDS and going["ended_at"] is None
    await _let_go(relay)
    ended = board.line_batches(ALICE, ended_since=0)[0]
    assert ended["done"] == 2 and ended["ended_at"] and ended["on"] is None
    assert board.line_batches(ALICE, ended_since=time.time() + 60) == [], "and leaves the banner a while after"


async def test_what_ended_long_ago_is_forgotten_and_a_task_that_goes_takes_its_row(board, ledger, relay):  # noqa: F811
    a, b = _tasks(board, 2)
    await handover.hand_over_many(ALICE, [a, b], "helper", WORDS)
    await _settle()
    board.line_forget(time.time() + 60)
    assert board.line_batches(ALICE, ended_since=0) == []
    c = _tasks(board, 1)[0]
    relay.hold = True
    await handover.hand_over_many(ALICE, [c], "helper", WORDS)
    await _started(relay, 3)
    board.delete_user(ALICE)
    assert board.line_unfinished() == []
    await _let_go(relay)


async def test_it_is_still_the_single_hand_over_for_one(board, ledger, relay):  # noqa: F811
    t = _tasks(board, 1)[0]
    await handover.hand_over(ALICE, t, "helper", WORDS)
    assert board.line_places(ALICE) == {} and board.line_of(ALICE, t) is None
    for task in list(handover._RUNNING.values()):
        await task


# ── The routes ────────────────────────────────────────────────────────


@pytest.fixture()
async def api(board, ledger, relay):  # noqa: F811
    app = FastAPI()
    app.include_router(taskboard.router)
    app.include_router(handover.router)
    current = {"user": {"id": ALICE, "email": "a@x.co", "display_name": "Alice", "role": "user"}}
    app.dependency_overrides[auth.get_current_user] = lambda: current["user"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck") as client:
        yield SimpleNamespace(client=client, be=lambda uid: current.update(user={"id": uid, "email": "x@x.co", "role": "user"}))


async def test_the_batch_route_answers_with_the_batch_and_the_refused(board, api, relay):  # noqa: F811
    relay.hold = True
    ok, closed = _tasks(board, 2)
    board.correct(ALICE, closed, "done")
    r = await api.client.post("/fd/taskboard/hand_over_batch", json={"tasks": [ok, closed], "agent": "helper", "instructions": WORDS})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["queued"] == [ok] and body["refused"][0]["id"] == closed and "open task" in body["refused"][0]["why"]
    assert body["batch"]["total"] == 1 and body["batch"]["agent"] == "Helper"

    listed = (await api.client.get("/fd/taskboard/hand_over_batches")).json()["batches"]
    assert [b["id"] for b in listed] == [body["batch"]["id"]]
    row = (await api.client.get(f"/fd/taskboard/tasks/{ok}")).json()["task"]
    assert row["lane"] == "working"
    await _started(relay, 1)
    await _let_go(relay)


async def test_a_task_waiting_says_its_place_on_the_list(board, api, relay):  # noqa: F811
    relay.hold = True
    a, b = _tasks(board, 2)
    await api.client.post("/fd/taskboard/hand_over_batch", json={"tasks": [a, b], "agent": "helper", "instructions": WORDS})
    await _started(relay, 1)
    items = {i["id"]: i for i in (await api.client.get("/fd/taskboard/tasks", params={"view": "working"})).json()["items"]}
    assert items[a]["in_line"] is None and items[b]["in_line"] == {"agent": "Helper", "ahead": 1}
    await _let_go(relay)


async def test_the_batch_route_refuses_when_nothing_can_go(board, api):  # noqa: F811
    t = _tasks(board, 1)[0]
    board.correct(ALICE, t, "done")
    r = await api.client.post("/fd/taskboard/hand_over_batch", json={"tasks": [t], "agent": "helper", "instructions": WORDS})
    assert r.status_code == 409 and "open task" in r.json()["detail"]
    r = await api.client.post("/fd/taskboard/hand_over_batch", json={"tasks": [t], "agent": "nope", "instructions": WORDS})
    assert r.status_code == 409 and "isn't available" in r.json()["detail"]


async def test_the_stop_route_is_for_the_batchs_owner(board, api, relay):  # noqa: F811
    relay.hold = True
    a, b = _tasks(board, 2)
    r = await api.client.post("/fd/taskboard/hand_over_batch", json={"tasks": [a, b], "agent": "helper", "instructions": WORDS})
    batch_id = r.json()["batch"]["id"]
    await _started(relay, 1)
    api.be("someone-else")
    assert (await api.client.post(f"/fd/taskboard/hand_over_batches/{batch_id}/stop")).status_code == 404
    assert (await api.client.get("/fd/taskboard/hand_over_batches")).json()["batches"] == []
    api.be(ALICE)
    r = await api.client.post(f"/fd/taskboard/hand_over_batches/{batch_id}/stop")
    assert r.status_code == 200 and r.json()["stopped"] == 1 and r.json()["batch"]["stopped"] == 1
    await _let_go(relay)


async def test_one_correction_on_several_each_as_on_its_own(board, api):  # noqa: F811
    a, b, c = _tasks(board, 3)
    r = await api.client.post("/fd/taskboard/batch_action", json={"ids": [a, b, a], "action": "done"})
    assert r.status_code == 200 and r.json() == {"changed": [a, b], "refused": []}
    for t in (a, b):
        row = board.get_task(ALICE, t)
        assert row["status"] == "done" and row["closed_by"] == "user" and "status" in json.loads(row["user_locks"])
        assert "Marked done" in _said(board, t)
    assert board.get_task(ALICE, c)["status"] == "open"

    r = await api.client.post("/fd/taskboard/batch_action", json={"ids": [a, b], "action": "undo"})
    assert r.json()["changed"] == [a, b]
    assert {board.get_task(ALICE, t)["status"] for t in (a, b)} == {"open"}


async def test_one_that_cant_take_it_doesnt_hold_up_the_rest(board, api):  # noqa: F811
    a, b = _tasks(board, 2)
    r = await api.client.post("/fd/taskboard/batch_action", json={"ids": [a, "tb_nope", b], "action": "not_task"})
    body = r.json()
    assert body["changed"] == [a, b] and body["refused"] == [{"id": "tb_nope", "why": "task not found"}]
    r = await api.client.post("/fd/taskboard/batch_action", json={"ids": [a], "action": "ack_failed"})
    assert r.json()["changed"] == [] and "failed" in r.json()["refused"][0]["why"]


async def test_someone_elses_task_is_refused_on_a_correction_too(board, api):  # noqa: F811
    t = _tasks(board, 1)[0]
    api.be("someone-else")
    r = await api.client.post("/fd/taskboard/batch_action", json={"ids": [t], "action": "done"})
    assert r.json()["changed"] == [] and board.get_task(ALICE, t)["status"] == "open"


async def test_snooze_takes_one_time_for_all(board, api):  # noqa: F811
    a, b = _tasks(board, 2)
    until = time.time() + 3600
    r = await api.client.post("/fd/taskboard/batch_action", json={"ids": [a, b], "action": "snooze", "value": until})
    assert r.json()["changed"] == [a, b]
    assert {board.get_task(ALICE, t)["snooze_until"] for t in (a, b)} == {until}
    r = await api.client.post("/fd/taskboard/batch_action", json={"ids": [a], "action": "snooze", "value": 1})
    assert r.json()["changed"] == [] and "to come" in r.json()["refused"][0]["why"]


@pytest.mark.parametrize("body,status", [
    ({"ids": ["x"], "action": "rename", "value": "All the same"}, 400),
    ({"ids": ["x"], "action": "drop_table"}, 400),
    ({"ids": [f"t{i}" for i in range(201)], "action": "done"}, 400),
])
async def test_what_a_batch_doesnt_take(api, body, status):
    assert (await api.client.post("/fd/taskboard/batch_action", json=body)).status_code == status
