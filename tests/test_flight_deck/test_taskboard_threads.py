"""Taskboard threads (v2): a record goes on an open task of the same board only
by a strong key — the answer to the agent's question, the same WhatsApp
conversation, the email a turn names, the email a send answers or an
auto-reply's own message id — never across boards, never onto a closed one;
Merge, Unmerge and Split are the user's, each a correction on both
timelines, round-tripping exactly, a merged task hidden from every list,
count and badge; Jev's "same task" only ever suggests; parties are people
only on exact handles (an FD user's record, an agent's contacts on its
owner's board), and the People filter names each person once by their id,
the old address ids still filtering."""

from __future__ import annotations

import json
import sqlite3
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import (
    auth,
    jev_routes,
    taskboard,
    taskboard_nextsteps,
    taskboard_sources,
    taskboard_threads,
    taskboard_triage,
)
from captain_claw.flight_deck.taskboard_sources import run_sweep
from captain_claw.flight_deck.taskboard_store import TaskboardStore

from .test_taskboard_ingest import _close, _open, _send, _task, _tid, api  # noqa: F401
from .test_taskboard_sources import HELPER_REF, env  # noqa: F401

ANA = "385913334444"


@pytest.fixture()
async def deck(api, monkeypatch):  # noqa: F811
    """The ingest deck (Alice owns "helper", Bob is a member) with the
    board's routes and the threads' on top; Alice signed in."""
    monkeypatch.setattr(taskboard, "_SYNC_GAP_S", 0.0)
    app = FastAPI()
    app.include_router(taskboard.router)
    app.include_router(taskboard_threads.router)
    users = {api.alice: {"id": api.alice, "email": "alice@x.co", "display_name": "Alice", "role": "user"},
             api.bob: {"id": api.bob, "email": "bob@x.co", "display_name": "Bob", "role": "user"}}
    current = {"user": users[api.alice]}
    app.dependency_overrides[auth.get_current_user] = lambda: current["user"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck") as client:
        yield SimpleNamespace(api=api, client=client, board=api.board, alice=api.alice, bob=api.bob, env=api.env,
                              be=lambda uid: current.update(user=users[uid]))


def _ana(**kw) -> dict:
    return {"role": "requester", "kind": "waid", "address": ANA, "display": "Ana", "verified": True, **kw}


def _req(board, uid, key, title, *, party=None, at=None, owes="agent", channel="whatsapp", **create) -> str:
    """A person's request on the board, as the ingest leaves one: their words,
    the board user's own on their own chat or send — with Jev's word on them
    if it has one (``flags``), as triage writes it."""
    at = time.time() if at is None else at
    mine = create.get("kind") == "sent" or create.get("flags", {}).get("own_chat")
    tid, _ = board.ingest(
        uid, keys=[key], parties=[party] if party else [],
        create={"title": title, "ask": title, "kind": "request", "direction": "received", "channel": channel,
                "status": "open", "owes_next": owes, "triage_state": "ok", "agent_slug": "helper",
                "agent_name": "Helper", "agent_owner_id": uid, "created_at": at, **create},
        event={"source": "test", "source_ref": f"{key}:open", "at": at, "kind": "received", "channel": channel,
               "actor_kind": "me" if mine else "party", "actor": (party or {}).get("display", ""), "summary": title,
               "refs": {"turn_id": key.removeprefix("turn:")} if key.startswith("turn:") else {}})
    if create.get("flags", {}).get("contains_instructions") is False:
        board.update_fields(uid, tid, {"flags": {"contains_instructions": False}})
    return tid


def _rows(board, uid) -> list[dict]:
    return [dict(r) for r in board._conn.execute("SELECT * FROM tasks WHERE user_id = ?", (uid,))]


def _all(board, uid) -> list[str]:
    return [r["id"] for r in board.query(uid, view="all")["rows"]]


def _flags(board, uid, task_id) -> dict:
    return json.loads(board.get_task(uid, task_id)["flags"])


# ── Strong keys: what a new record goes on ────────────────────────────


async def test_the_answer_to_the_agents_question_goes_on_its_task(deck):
    """The agent asked Alice something on her chat and the board has the
    next move as hers: her next turn in that session is the answer — on
    that task (back in the triage queue), its close too, never a second
    task."""
    turns, board, uid = deck.api, deck.board, deck.alice
    first, answer = _tid(), _tid()
    await _send(turns, _open(turns, first, text="Book a table for Saturday"),
                _close(turns, first, reply="Sure — for how many people, and at what time?"))
    task = _task(turns, uid, first)
    board.update_fields(uid, task["id"], {"owes_next": "me", "triage_state": "ok"})  # what triage made of it
    await _send(turns, _open(turns, answer, text="Four of us, at 8 pm"))
    assert _task(turns, uid, answer)["id"] == task["id"] and len(_rows(board, uid)) == 1
    host = board.get_task(uid, task["id"])
    assert host["triage_state"] == "pending" and host["triage_after"] <= time.time()
    (ev,) = [e for e in board.events(uid, task["id"]) if e["source_ref"] == f"{answer}:open"]
    assert (ev["kind"], ev["refs"]["attached"]) == ("received", "follow_up")
    turn = board.get_turn(answer)
    assert (turn["gate"], turn["task_id"], turn["task_key"]) == ("task", task["id"], f"turn:{answer}")
    await _send(turns, _close(turns, answer, reply="Booked Toni's for four at 8 pm."))
    assert [e["kind"] for e in board.events(uid, task["id"])].count("reply") == 2
    # One requester still: Alice, once.
    assert [p["kind"] for p in board.parties_for(uid, [task["id"]])[task["id"]]] == ["fd_user"]


@pytest.mark.parametrize("case", ["statement", "agent_owes", "late", "other_session", "closed"])
async def test_a_new_ask_after_the_agents_reply_is_a_task_of_its_own(deck, case):
    """Every ask is a task (decision #3) unless it answers one: not when the
    reply asked nothing, the board doesn't wait on the user, an hour went
    by, it is another session, or the task is closed."""
    turns, board, uid = deck.api, deck.board, deck.alice
    first, second = _tid(), _tid()
    reply = "Booked Toni's for 8 pm." if case == "statement" else "Which day?"
    at = time.time() - (7200 if case == "late" else 10)
    await _send(turns, _open(turns, first, at=at - 5, text="Book a table"), _close(turns, first, reply=reply, at=at))
    task = _task(turns, uid, first)
    if case != "agent_owes":
        board.update_fields(uid, task["id"], {"owes_next": "me"})
    if case == "closed":
        board.correct(uid, task["id"], "done")
    await _send(turns, _open(turns, second, text="Saturday, and a taxi there",
                           **({"session_id": "s-other"} if case == "other_session" else {})))
    assert _task(turns, uid, second)["id"] != task["id"]
    assert len(_rows(board, uid)) == 2


async def test_a_reply_cut_where_the_board_keeps_it_asks_nothing():
    assert taskboard_threads.asks_back("Which one do you want?")
    assert taskboard_threads.asks_back("Two options: **A** or **B**?**")
    assert not taskboard_threads.asks_back("Is it A? I booked A.")
    assert not taskboard_threads.asks_back("x" * 990 + " is that ok?")  # 1000 long: the rest may follow
    assert not taskboard_threads.asks_back("")


async def test_someones_whatsapp_within_half_an_hour_is_one_conversation(deck):
    """Ana writing again to the same agent within half an hour of her open
    request is the same conversation; after a quiet half hour it is a new
    one. The board user's own WhatsApp never is (each of their asks is a
    task)."""
    turns, board, uid = deck.api, deck.board, deck.alice
    wa = {"channel": "whatsapp", "source": "whatsapp", "ext_refs": {"waid": ANA}, "speaker_label": "Ana",
          "session_id": "s-wa-ana"}
    first, more, later = _tid(), _tid(), _tid()
    await _send(turns, _open(turns, first, text="Hi, can you send me the price list?", at=time.time() - 600, **wa))
    await _send(turns, _open(turns, more, text="The one for 2027, please", **wa))
    task = _task(turns, uid, first)
    assert _task(turns, uid, more)["id"] == task["id"]
    assert any(e["refs"].get("attached") == "conversation" for e in board.events(uid, task["id"]))
    board._conn.execute("UPDATE tasks SET last_event_at = ? WHERE id = ?", (time.time() - 3600, task["id"]))
    board._conn.commit()
    await _send(turns, _open(turns, later, text="And the invoice for October", **wa))
    assert _task(turns, uid, later)["id"] != task["id"]

    turns.env.auto.set_overrides(uid, {"notify_waid": "385911112222"})
    own = {**wa, "ext_refs": {"waid": "385911112222"}, "session_id": "s-wa-alice", "speaker_label": "Alice"}
    mine, again = _tid(), _tid()
    await _send(turns, _open(turns, mine, text="What's on tomorrow?", **own))
    await _send(turns, _open(turns, again, text="And the weather?", **own))
    assert _flags(board, uid, _task(turns, uid, again)["id"])["own_chat"] is True
    assert _task(turns, uid, mine)["id"] != _task(turns, uid, again)["id"]


async def test_a_turn_naming_an_email_an_open_task_holds_goes_on_it(deck):
    turns, board, uid = deck.api, deck.board, deck.alice
    host, _ = board.ingest(uid, keys=["gmail:thread:t-77"], event={"source": "t", "source_ref": "e1", "kind": "received"},
                           create={"title": "Invoice for October", "kind": "held_reply", "channel": "gmail",
                                   "owes_next": "me"})
    tid = _tid()
    await _send(turns, _open(turns, tid, text="Draft the answer to Ana's email", ext_refs={"gmail": {"thread_id": "t-77"}}))
    assert _task(turns, uid, tid)["id"] == host
    assert any(e["refs"].get("attached") == "email_thread" for e in board.events(uid, host))
    # A closed task is never added to: the next one is a task of its own.
    board.correct(uid, host, "done")
    tid2 = _tid()
    await _send(turns, _open(turns, tid2, text="And file it", ext_refs={"gmail": {"thread_id": "t-77"}}))
    assert _task(turns, uid, tid2)["id"] != host


async def test_a_send_answering_an_email_an_open_task_holds_is_sent_on_it(deck, monkeypatch):
    """In-Reply-To: the agent's email answers one the board has on an open
    request (by its message id) on a thread the board hasn't seen: the
    send is that task's evidence, never a "sent" task of its own — and it
    doesn't close it."""
    monkeypatch.setattr(taskboard_sources, "_SEND_WAIT_S", 0.0)
    board, uid = deck.board, deck.alice
    host = _req(board, uid, "gmail:msg:m-9", "Price list for Ana", channel="gmail",
                party={"role": "requester", "kind": "email", "address": "ana@x.co", "display": "Ana", "verified": True})
    await deck.env.db.add_gmail_send(uid, "helper", to_addrs="Ana <ana@x.co>", subject="Re: Price list",
                                     gmail_message_id="g-9", thread_id="t-9", in_reply_to_message_id="m-9")
    run_sweep()
    assert board.task_for_key(uid, "gmail:thread:t-9") == host and board.task_for_key(uid, "gmail:msg:g-9") == host
    task = board.get_task(uid, host)
    assert (task["kind"], task["status"], task["evidence"]) == ("request", "open", "verified")
    (sent,) = [e for e in board.events(uid, host) if e["source"] == "gmail_sends"]
    assert (sent["kind"], sent["refs"]["attached"]) == ("sent", "email_reply")
    assert [r["id"] for r in _rows(board, uid)] == [host]


async def test_an_auto_reply_on_an_email_an_open_task_holds_stays_on_it(deck):
    board, uid, db = deck.board, deck.alice, deck.env.db
    host = _req(board, uid, "gmail:msg:m-5", "Opening hours?", channel="gmail")
    rid = await db.claim_autoreply(uid, "m-5", "t-5")
    await db.update_autoreply(rid, uid, from_addr="ana@x.co", subject="Opening hours?", topic_name="Opening hours",
                              agent_ref=HELPER_REF, agent_name="Helper")
    assert await db.cas_autoreply(rid, uid, "new", "held")
    run_sweep()
    assert board.task_for_key(uid, "gmail:thread:t-5") == host and board.task_for_key(uid, f"autoreply:{rid}") == host
    assert board.get_task(uid, host)["owes_next"] == "me"  # held for Alice's approval: Needs me
    assert len(_rows(board, uid)) == 1


async def test_nothing_is_attached_across_boards(deck, monkeypatch):
    """Bob's task holds the email Alice's agent answers: Alice's send is a
    task on her own board; and a key or a host of another board is never
    attached to."""
    monkeypatch.setattr(taskboard_sources, "_SEND_WAIT_S", 0.0)
    board, alice, bob = deck.board, deck.alice, deck.bob
    bobs = _req(board, bob, "gmail:msg:m-9", "Bob's price list", channel="gmail")
    await deck.env.db.add_gmail_send(alice, "helper", to_addrs="ana@x.co", subject="Re: Price list",
                                     gmail_message_id="g-9", thread_id="t-9", in_reply_to_message_id="m-9")
    run_sweep()
    mine = board.task_for_key(alice, "gmail:thread:t-9")
    assert mine and mine != bobs and board.get_task(alice, mine)["kind"] == "sent"
    assert board.event_count(bob, bobs) == 1 and board.task_for_key(bob, "gmail:thread:t-9") is None
    ev = {"source": "t", "source_ref": "x1"}
    tid, _ = taskboard_threads.attach_or_create(board, alice, keys=["x:1"], attach=["gmail:msg:m-9"],
                                                host=(bobs, "follow_up"), create={"title": "Mine"}, event=ev)
    assert tid not in (None, bobs) and board.get_task(alice, tid)["title"] == "Mine"
    assert board.get_task(alice, bobs) is None and board.event_count(bob, bobs) == 1


async def test_an_attach_never_closes_or_merges(deck):
    """A host that is closed or merged is no host; one a mute rule keeps
    quiet takes the record but stays out of the triage queue."""
    board, uid = deck.board, deck.alice
    closed = _req(board, uid, "turn:c1", "Closed one")
    board.correct(uid, closed, "done")
    tid, _ = taskboard_threads.attach_or_create(board, uid, keys=["x:1"], host=(closed, "follow_up"),
                                                create={"title": "New"}, event={"source": "t", "source_ref": "x1"})
    assert tid != closed
    quiet = _req(board, uid, "turn:q1", "Muted", triage_state="skipped", owes="nobody")
    got, _ = taskboard_threads.attach_or_create(
        board, uid, keys=["x:2"], host=(quiet, "conversation"), create={"title": "New"},
        on_attach={"triage_state": "pending", "triage_after": time.time()},
        event={"source": "t", "source_ref": "x2"})
    assert got == quiet and board.get_task(uid, quiet)["triage_state"] == "skipped"


# ── Merge, unmerge, split ─────────────────────────────────────────────


async def _post(deck, path, body=None, status=200):
    r = await deck.client.post(path, json=body)
    assert r.status_code == status, r.text
    return r.json()


def _summaries(board, uid, task_id) -> list[str]:
    return [e["summary"] for e in board.events(uid, task_id)]


async def test_merge_unmerge_and_split_round_trip(deck):
    board, uid = deck.board, deck.alice
    a = _req(board, uid, "turn:aaa", "Price list for Ana", party=_ana(), owes="me")
    board.ingest(uid, keys=["turn:aaa"], event={"source": "test", "source_ref": "aaa:close", "kind": "reply",
                                               "summary": "Sent it", "refs": {"turn_id": "aaa"}})
    b = _req(board, uid, "turn:bbb", "Ana's 2027 prices", owes="me",
             party={"role": "requester", "kind": "email", "address": "ana@x.co", "display": "Ana", "verified": True})
    a_events = [e["id"] for e in board.events(uid, a)]
    assert board.counts(uid)["all"] == 2 and board.badge_count(uid) == 2

    out = await _post(deck, f"/fd/taskboard/tasks/{a}/merge", {"into": b})
    assert out["task"]["id"] == b and out["task"]["merged_into"] == ""
    # Hidden everywhere: lists, counts, the badge — its id reads as b's.
    assert _all(board, uid) == [b] and board.counts(uid)["all"] == 1 and board.badge_count(uid) == 1
    assert board.get_task(uid, a)["merged_into"] == b and taskboard_threads.resolve(uid, a) == b
    assert taskboard_threads.detail_extras(uid, b) == {"merged": [{"id": a, "title": "Price list for Ana"}]}
    assert set(a_events) <= {e["id"] for e in board.events(uid, b)}
    assert "Merged “Price list for Ana” into this task" in _summaries(board, uid, b)
    assert _summaries(board, uid, a) == ["Merged into “Ana's 2027 prices”"]
    assert board.task_for_key(uid, "turn:aaa") == b
    assert {p["kind"] for p in board.parties_for(uid, [b])[b]} == {"waid", "email"}
    target = board.get_task(uid, b)
    assert target["triage_state"] == "pending" and target["extracted_fp"] == ""  # read again, with a's messages
    # What a's sources say next lands on b.
    board.ingest(uid, keys=["turn:aaa"], event={"source": "test", "source_ref": "late", "kind": "note"})
    assert "late" in {e["source_ref"] for e in board.events(uid, b)}

    out = await _post(deck, f"/fd/taskboard/tasks/{a}/unmerge")
    assert (out["task"]["id"], out["task"]["merged_into"], out["into"]["id"]) == (a, "", b)
    assert sorted(_all(board, uid)) == sorted([a, b]) and board.counts(uid)["all"] == 2
    assert set(a_events) <= {e["id"] for e in board.events(uid, a)}
    assert not set(a_events) & {e["id"] for e in board.events(uid, b)}
    assert "late" in {e["source_ref"] for e in board.events(uid, b)}  # what came in meanwhile stays where it landed
    assert board.task_for_key(uid, "turn:aaa") == a
    assert {p["kind"] for p in board.parties_for(uid, [b])[b]} == {"email"}
    assert _summaries(board, uid, a)[-1] == "Taken back out of “Ana's 2027 prices”"
    assert _summaries(board, uid, b)[-1] == "Took “Price list for Ana” back out"
    assert board.get_task(uid, a)["triage_state"] == "pending"
    assert taskboard_threads.detail_extras(uid, b) == {"merged": []} and taskboard_threads.resolve(uid, a) == a

    # Split: a's reply is a task of its own, with the turn only it names.
    reply = next(e["id"] for e in board.events(uid, a) if e["source_ref"] == "aaa:close")
    board._conn.execute("INSERT INTO turns (turn_id, user_id, agent_slug, task_id) VALUES ('aaa', ?, 'helper', ?)",
                        (uid, a))
    board._conn.commit()
    with pytest.raises(ValueError):
        board.split(uid, a, [e["id"] for e in board.events(uid, a)])  # corrections stay
    opened = next(e["id"] for e in board.events(uid, a) if e["source_ref"] == "turn:aaa:open")
    out = await _post(deck, f"/fd/taskboard/tasks/{a}/split", {"event_ids": [reply]})
    new = out["new_task"]["id"]
    assert out["task"]["id"] == a and new not in (a, b)
    assert [e["id"] for e in board.events(uid, new) if e["kind"] != "correction"] == [reply]
    assert out["new_task"]["title"] == "Sent it" and out["new_task"]["status"] == "open"
    # The turn's key: the open event left behind names it too, so it stays.
    assert board.task_for_key(uid, "turn:aaa") == a and board.get_turn("aaa")["task_id"] == a
    assert _summaries(board, uid, a)[-1] == "Split “Sent it” off as a task of its own"
    assert _summaries(board, uid, new)[-1] == "Split off “Price list for Ana”"
    assert board.counts(uid)["all"] == 3
    # Merging it back puts the task together again.
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": a})
    assert {opened, reply} <= {e["id"] for e in board.events(uid, a)} and board.counts(uid)["all"] == 2


async def test_split_moves_the_links_only_its_events_name(deck):
    board, uid = deck.board, deck.alice
    a = _req(board, uid, "turn:t1", "Two asks in one")
    board.add_link(uid, "turn:t2", a)
    board.ingest(uid, keys=["turn:t2"], event={"source": "test", "source_ref": "t2:open", "kind": "received",
                                              "summary": "Also the invoice", "refs": {"turn_id": "t2"}})
    board.ingest(uid, keys=["turn:t2"], event={"source": "test", "source_ref": "t2:close", "kind": "reply",
                                              "summary": "Sent", "refs": {"turn_id": "t2"}})
    board._conn.execute("INSERT INTO turns (turn_id, user_id, agent_slug, task_id) VALUES ('t2', ?, 'helper', ?)",
                        (uid, a))
    board._conn.commit()
    ids = [e["id"] for e in board.events(uid, a) if e["source_ref"].startswith("t2:")]
    _, new = board.split(uid, a, ids, actor="Alice")
    assert (new["title"], new["kind"], new["triage_state"]) == ("Also the invoice", "request", "pending")
    assert board.task_for_key(uid, "turn:t2") == new["id"] and board.task_for_key(uid, "turn:t1") == a
    assert board.get_turn("t2")["task_id"] == new["id"]


async def test_merge_and_split_refuse_what_cant_be(deck):
    board, uid = deck.board, deck.alice
    a, b, c = (_req(board, uid, f"turn:{k}", f"Task {k}") for k in "abc")
    bobs = _req(board, deck.bob, "turn:bob", "Bob's")
    await _post(deck, f"/fd/taskboard/tasks/{a}/merge", {"into": a}, status=400)
    await _post(deck, f"/fd/taskboard/tasks/{a}/merge", {"into": bobs}, status=404)
    await _post(deck, f"/fd/taskboard/tasks/{bobs}/merge", {"into": a}, status=404)
    await _post(deck, f"/fd/taskboard/tasks/{b}/unmerge", status=409)
    await _post(deck, f"/fd/taskboard/tasks/{a}/merge", {"into": b})
    await _post(deck, f"/fd/taskboard/tasks/{a}/merge", {"into": c}, status=409)  # merged already
    # Into a merged task: into the task it is part of.
    out = await _post(deck, f"/fd/taskboard/tasks/{c}/merge", {"into": a})
    assert out["task"]["id"] == b and board.get_task(uid, c)["merged_into"] == b
    # Merges come apart the other way round from how they were made.
    await _post(deck, f"/fd/taskboard/tasks/{b}/merge", {"into": bobs}, status=404)
    d = _req(board, uid, "turn:d", "Task d")
    await _post(deck, f"/fd/taskboard/tasks/{b}/merge", {"into": d})
    await _post(deck, f"/fd/taskboard/tasks/{a}/unmerge", status=409)
    await _post(deck, f"/fd/taskboard/tasks/{b}/unmerge")
    await _post(deck, f"/fd/taskboard/tasks/{a}/unmerge")
    assert sorted(_all(board, uid)) == sorted([a, b, d])
    # Split: some events, never all, never someone else's.
    only = [e["id"] for e in board.events(uid, d) if e["kind"] != "correction"]
    await _post(deck, f"/fd/taskboard/tasks/{d}/split", {"event_ids": only}, status=400)
    await _post(deck, f"/fd/taskboard/tasks/{d}/split", {"event_ids": [10**9]}, status=400)
    await _post(deck, f"/fd/taskboard/tasks/{bobs}/split", {"event_ids": [1]}, status=404)
    await _post(deck, f"/fd/taskboard/tasks/{d}/split", {"event_ids": []}, status=422)
    deck.be(deck.bob)
    await _post(deck, f"/fd/taskboard/tasks/{a}/unmerge", status=404)


async def test_a_merged_task_is_out_of_the_triage_queue_and_its_lanes(deck):
    board, uid = deck.board, deck.alice
    a = _req(board, uid, "turn:a", "Waiting on me", owes="me", triage_state="pending", triage_after=time.time() - 1)
    b = _req(board, uid, "turn:b", "The other")
    assert a in {t["id"] for t in board.triage_queue(uid)}
    board.merge(uid, a, b, actor="Alice")
    assert a not in {t["id"] for t in board.triage_queue(uid)}
    assert board.query(uid, view="needs_me")["total"] == 0 and board.counts(uid)["needs_me"] == 0
    assert board.badge_count(uid) == 0


# ── Same task? (Jev, only ever a suggestion) ─────────────────────────


@pytest.fixture()
def same_jev(monkeypatch):
    """A Jev stub for the same-task ask: every pair at ``p``."""
    j = SimpleNamespace(calls=[], p=0.9, fail=None)

    async def ask(owner, state, questions, model=None, *, source, agent=""):
        j.calls.append(SimpleNamespace(owner=owner, state=state, questions=questions, source=source))
        if j.fail is not None:
            raise j.fail
        return {"answers": {q: {"type": "noul", "noul": j.p} for q in questions}}

    monkeypatch.setattr(jev_routes, "ask_for_owner", ask)
    return j


def _pair(board, uid, *, older="Invoice for October", newer="October invoice from Ana", party=None):
    party = party or _ana()
    old = _req(board, uid, "turn:old", older, party=party, at=time.time() - 3600)
    new = _req(board, uid, "turn:new", newer, party=party)
    return old, new


async def test_same_task_only_suggests(deck, same_jev):
    board, uid = deck.board, deck.alice
    old, new = _pair(board, uid)
    assert await taskboard_threads.same_task_pass(uid, [new]) == 1
    (call,) = same_jev.calls
    assert call.owner == uid and list(call.questions) == ["t0_same_task"]
    assert call.state["pairs"][0]["older"]["quoted"]["title"] == "Invoice for October"
    assert _flags(board, uid, new)["same_as"] == {"task_id": old, "title": "Invoice for October", "p": 0.9}
    assert "same_as" not in _flags(board, uid, old)
    assert sorted(_all(board, uid)) == sorted([old, new])  # nothing merged
    (d,) = board.decisions(uid, new)
    assert (d["judge"], d["qset"], d["answers"]) == ("jev", "taskboard.same.v1", {"same_task": {"p": 0.9}})
    assert board.daily_counts(uid)["jev"] == 1
    # The decision says nothing of where the task stands.
    assert taskboard_triage.why_summary(board.get_task(uid, new), board.decisions(uid, new)) == \
        "Your agent is working on it."
    # Suggested once: not asked again.
    assert await taskboard_threads.same_task_pass(uid, [new]) == 0 and len(same_jev.calls) == 1

    # "Not the same": the suggestion goes and stays gone; Undo brings it back.
    out = await _post(deck, f"/fd/taskboard/tasks/{new}/not_same")
    assert "same_as" not in out["task"]["flags"] and "same_as" in out["task"]["user_locks"]
    await _post(deck, f"/fd/taskboard/tasks/{new}/not_same", status=400)
    assert await taskboard_threads.same_task_pass(uid, [new]) == 0 and len(same_jev.calls) == 1
    board.correct(uid, new, "undo")
    assert _flags(board, uid, new)["same_as"]["task_id"] == old
    # Merging answers it.
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    assert "same_as" not in _flags(board, uid, new)


async def test_same_task_below_the_bar_suggests_nothing_and_isnt_asked_twice(deck, same_jev):
    board, uid = deck.board, deck.alice
    old, new = _pair(board, uid)
    same_jev.p = 0.6
    assert await taskboard_threads.same_task_pass(uid, [new]) == 1
    assert "same_as" not in _flags(board, uid, new)
    assert board.decisions(uid, new)[0]["why"].startswith("Jev doesn't think it is the same task")
    assert await taskboard_threads.same_task_pass(uid, [new]) == 0 and len(same_jev.calls) == 1
    # A new title (extraction wrote a better one) is a new question.
    board.update_fields(uid, new, {"title": "Ana's October invoice"})
    assert await taskboard_threads.same_task_pass(uid, [new]) == 1 and len(same_jev.calls) == 2


async def test_same_task_is_asked_only_of_alike_tasks_sharing_someone_else(deck, same_jev):
    """Only pairs that share a recorded person who isn't the board user (on
    every one of their own chats), came within a week and are titled alike;
    no Jev, nothing — and the day's Jev cap given back."""
    board, uid = deck.board, deck.alice
    me = {"role": "requester", "kind": "fd_user", "address": uid, "display": "Alice", "verified": True}
    _pair(board, uid, party=me)
    assert await taskboard_threads.same_task_pass(uid, [board.task_for_key(uid, "turn:new")]) == 0
    other = _req(board, uid, "turn:x1", "Lunch on Friday", party=_ana())
    _req(board, uid, "turn:x0", "Invoice for October", party=_ana(), at=time.time() - 60)
    _req(board, uid, "turn:x2", "October invoice", party=_ana(address="385910000000"))
    old = _req(board, uid, "turn:x3", "Invoice October", party=_ana(), at=time.time() - 9 * 86400)
    board._conn.execute("UPDATE tasks SET last_event_at = ? WHERE id = ?", (time.time() - 9 * 86400, old))
    board._conn.commit()
    assert await taskboard_threads.same_task_pass(uid, [other]) == 0 and same_jev.calls == []
    newer = _req(board, uid, "turn:x4", "The October invoice again", party=_ana())
    same_jev.fail = jev_routes.JevNotConnected()
    assert await taskboard_threads.same_task_pass(uid, [newer]) == 0
    assert len(same_jev.calls[0].state["pairs"]) == 1  # x0 only: x2 is someone else, x3 too long ago
    assert "same_as" not in _flags(board, uid, newer) and board.daily_counts(uid)["jev"] == 0


async def test_the_triage_pass_asks_about_the_tasks_it_judged(deck, monkeypatch):
    board, uid = deck.board, deck.alice
    tid = _req(board, uid, "turn:t1", "Price list", party=_ana(), triage_state="pending", triage_after=time.time() - 1,
               kind="sent")
    seen = []

    async def same(user_id, task_ids):
        seen.append((user_id, task_ids))
        return 0

    monkeypatch.setattr(taskboard_threads, "same_task_pass", same)
    monkeypatch.setattr(taskboard_triage, "_RUNNING", set())

    async def no_jev(*a, **k):
        raise jev_routes.JevNotConnected()

    monkeypatch.setattr(jev_routes, "ask_for_owner", no_jev)
    await taskboard_triage.triage_user(uid)
    assert seen == [(uid, [tid])]


# ── People ────────────────────────────────────────────────────────────


def _person(board, uid, task_id, kind=None) -> dict:
    return next(p for p in board.parties_for(uid, [task_id])[task_id] if kind is None or p["kind"] == kind)


def test_a_party_is_a_person_only_on_an_exact_handle(tmp_path):
    board = TaskboardStore(tmp_path / "taskboard.db")
    try:
        uid = "user-alice"
        email = {"role": "requester", "kind": "email", "address": "ana@x.co", "display": "Ana Kovač", "verified": True}
        one = _req(board, uid, "k:1", "One", party=email)
        two = _req(board, uid, "k:2", "Two", party={**email, "address": "ANA@x.co", "display": "ana@x.co"})
        p1, p2 = _person(board, uid, one), _person(board, uid, two)
        assert p1["party_id"] and p1["party_id"] == p2["party_id"] and p2["name"] == "Ana Kovač"
        # The same name, another address: someone else.
        three = _req(board, uid, "k:3", "Three", party={**email, "address": "ana@other.org"})
        assert _person(board, uid, three)["party_id"] not in ("", p1["party_id"])
        # A model's guess: linked only to a handle the board knows, never a new person.
        read = _req(board, uid, "k:4", "Four", party={**email, "verified": False, "role": "mentioned"})
        guess = _req(board, uid, "k:5", "Five", party={**email, "address": "zed@x.co", "verified": False})
        named = _req(board, uid, "k:6", "Six", party={"role": "mentioned", "kind": "name", "address": "ana kovac",
                                                      "display": "Ana Kovač"})
        assert _person(board, uid, read)["party_id"] == p1["party_id"]
        assert _person(board, uid, guess)["party_id"] == "" and _person(board, uid, named)["party_id"] == ""
        # A WhatsApp Web jid and the number in it are one person; the board user is nobody's.
        wa = _req(board, uid, "k:7", "Seven", party=_ana())
        jid = _req(board, uid, "k:8", "Eight", party={"role": "requester", "kind": "wa_jid", "verified": True,
                                                      "address": f"{ANA}@s.whatsapp.net", "display": "Ana W"})
        assert _person(board, uid, wa)["party_id"] == _person(board, uid, jid)["party_id"] != ""
        me = _req(board, uid, "k:9", "Mine", party={"role": "requester", "kind": "fd_user", "address": uid,
                                                     "display": "Alice", "verified": True})
        assert _person(board, uid, me)["party_id"] == ""
        # One board's people are never another's.
        bobs = _req(board, "user-bob", "k:1", "Bob's", party=email)
        assert _person(board, "user-bob", bobs)["party_id"] not in ("", p1["party_id"])
    finally:
        board.close()


def test_a_board_from_before_v2_gets_its_people_once(tmp_path):
    path = tmp_path / "taskboard.db"
    board = TaskboardStore(path)
    uid = "user-alice"
    tid = _req(board, uid, "k:1", "Old", party=None)
    board._conn.execute(
        "INSERT INTO task_parties (task_id, user_id, role, kind, address, display, verified) "
        "VALUES (?, ?, 'requester', 'email', 'ana@x.co', 'Ana', 1)", (tid, uid))
    board._conn.execute("DELETE FROM source_cursors WHERE source = 'v2:people'")
    board._conn.commit()
    board.close()
    board = TaskboardStore(path)
    try:
        assert _person(board, uid, tid)["party_id"] and _person(board, uid, tid)["name"] == "Ana"
    finally:
        board.close()


def _address_book(tmp, rows) -> None:
    db = tmp / "helper" / "data" / "sessions" / "sessions.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE IF NOT EXISTS contacts (id TEXT PRIMARY KEY, name TEXT, email TEXT, phone TEXT, "
                 "privacy_tier TEXT NOT NULL DEFAULT 'normal')")
    conn.executemany("INSERT INTO contacts (id, name, email, phone, privacy_tier) VALUES (?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


async def test_the_agents_contacts_and_deck_users_name_people_on_the_owners_board(deck):
    """Ana writes from her email and her WhatsApp: two handles, two people —
    until the agent's address book holds both (one row), on its owner's
    board: one person, by the contact's name, under one People value
    (filtered by it or by either old address id). A member's board never
    reads the owner's contacts; a private contact never counts; a deck user
    is named by their own record."""
    board, uid, tmp = deck.board, deck.alice, deck.env.tmp
    agents = taskboard_sources._agent_index()
    _address_book(tmp, [("c1", "Ana Kovač", "Ana@X.co", "+385 91 333 4444", "normal"),
                        ("c2", "Secret Sam", "sam@x.co", "", "private")])
    mail = {"role": "requester", "kind": "email", "address": "ana@x.co", "display": "ana@x.co", "verified": True}
    ev = {"source": "t", "kind": "received"}
    create = {"kind": "request", "agent_slug": "helper", "agent_owner_id": uid, "status": "open", "owes_next": "me"}
    by_mail, _ = taskboard_threads.attach_or_create(board, uid, keys=["k:mail"], parties=[mail], agents=agents,
                                                    create={**create, "title": "Mail"}, event={**ev, "source_ref": "1"})
    by_wa, _ = taskboard_threads.attach_or_create(board, uid, keys=["k:wa"], parties=[_ana(display="Ana")],
                                                  agents=agents, create={**create, "title": "WhatsApp"},
                                                  event={**ev, "source_ref": "2"})
    person = _person(board, uid, by_mail)
    assert person["party_id"] == _person(board, uid, by_wa)["party_id"] and person["name"] == "Ana Kovač"
    facet = board.query(uid, view="all")["facets"]["party"]
    assert facet == [(f"party:{person['party_id']}", "Ana Kovač", 2)]
    for value in (f"party:{person['party_id']}", "email:ana@x.co", f"waid:{ANA}"):
        assert board.query(uid, view="all", filters={"party": [value]})["total"] == (2 if value.startswith("party:")
                                                                                     else 1)
    assert [i["title"] for i in board.query(uid, view="all", q="kovac")["rows"]] in (["WhatsApp", "Mail"],
                                                                                     ["Mail", "WhatsApp"])
    # Through the route: ids and names as the dialog reads them.
    out = (await deck.client.get("/fd/taskboard/tasks", params={"view": "all"})).json()
    assert [(f["id"], f["label"], f["count"]) for f in out["facets"]["party"]] == \
        [(f"party:{person['party_id']}", "Ana Kovač", 2)]
    assert {(p["party_id"], p["name"]) for i in out["items"] for p in i["parties"]} == \
        {(person["party_id"], "Ana Kovač")}

    # A private contact names nobody.
    sam = {**mail, "address": "sam@x.co", "display": "sam@x.co"}
    secret, _ = taskboard_threads.attach_or_create(board, uid, keys=["k:sam"], parties=[sam], agents=agents,
                                                   create={**create, "title": "Sam"}, event={**ev, "source_ref": "3"})
    assert _person(board, uid, secret)["name"] == "sam@x.co"
    # Bob, a member of Alice's agent: the same addresses are his own people, unnamed by her contacts.
    bobs, _ = taskboard_threads.attach_or_create(board, deck.bob, keys=["k:mail"], parties=[mail], agents=agents,
                                                 create={**create, "agent_owner_id": uid, "title": "Bob's mail"},
                                                 event={**ev, "source_ref": "4"})
    theirs = _person(board, deck.bob, bobs)
    assert theirs["party_id"] not in ("", person["party_id"]) and theirs["name"] == "ana@x.co"
    assert [f[0] for f in board.query(deck.bob, view="all")["facets"]["party"]] == [f"party:{theirs['party_id']}"]
    # A deck user on the task: named by their own record, and their email is them.
    on, _ = taskboard_threads.attach_or_create(
        board, uid, keys=["k:bob"], agents=agents, create={**create, "title": "With Bob"},
        parties=[{"role": "cc", "kind": "fd_user", "address": deck.bob, "display": "Member", "verified": True},
                 {"role": "recipient", "kind": "email", "address": "bob@x.co", "display": "bob@x.co",
                  "verified": True}], event={**ev, "source_ref": "5"})
    rows = board.parties_for(uid, [on])[on]
    assert len({p["party_id"] for p in rows}) == 1 and {p["name"] for p in rows} == {"Bob"}


def test_dropping_a_task_drops_what_was_merged_into_it_and_people_nobody_names(tmp_path):
    """An agent removed takes its tasks — and what was merged into them (its
    messages lived there) — and the people no task names any more."""
    board = TaskboardStore(tmp_path / "taskboard.db")
    try:
        uid = "user-alice"
        a = _req(board, uid, "k:a", "A", party=_ana())
        b = _req(board, uid, "k:b", "B", agent_slug="other")
        kept = _req(board, uid, "k:c", "C", party={"role": "requester", "kind": "email", "address": "bo@y.org",
                                                    "display": "Bo", "verified": True})
        board.merge(uid, a, b)

        def count(table: str) -> int:
            return board._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        assert (count("parties"), count("party_handles")) == (2, 2)
        assert board.delete_agent(uid, "other") == 1
        assert board.get_task(uid, a) is None and board.get_task(uid, b) is None
        assert board.get_task(uid, kept) is not None
        assert (count("parties"), count("party_handles")) == (1, 1)
        assert _person(board, uid, kept)["name"] == "Bo"
    finally:
        board.close()


# ── A merge never hides live work ─────────────────────────────────────


def _done(board, uid, key, title, **create) -> str:
    """A request Jev closed."""
    return _req(board, uid, key, title, party=_ana(), status="done", owes_next="nobody", closed_by="jev",
                closed_by_party="Jev", closed_at=time.time() - 3600, result="success", at=time.time() - 7200, **create)


async def test_live_work_merged_into_a_closed_task_opens_it_and_taking_it_out_closes_it(deck):
    board, uid = deck.board, deck.alice
    old = _done(board, uid, "turn:old", "Invoice for October")
    new = _req(board, uid, "turn:new", "October invoice again", party=_ana(), owes="me")
    assert (board.counts(uid)["needs_me"], board.counts(uid)["done"]) == (1, 1)

    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    t = board.get_task(uid, old)
    assert (t["status"], t["owes_next"], t["closed_at"], t["closed_by"], t["result"]) == ("open", "me", None, "", "")
    assert "status" in json.loads(t["user_locks"]) and t["triage_state"] == "pending"
    # Still where the user sees it: Needs me, counted in the badge — and no pass closes it again.
    assert board.counts(uid)["needs_me"] == 1 and board.counts(uid)["done"] == 0 and board.badge_count(uid) == 1
    view = board.triage_view(uid, old)
    assert taskboard_triage.rule_moves(view["task"], taskboard_triage.facts(view, uid))[0].get("status") is None

    await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
    t = board.get_task(uid, old)
    assert (t["status"], t["owes_next"], t["closed_by"], t["result"]) == ("done", "nobody", "jev", "success")
    assert t["closed_at"] is not None and "status" not in json.loads(t["user_locks"])
    assert (board.counts(uid)["needs_me"], board.counts(uid)["done"]) == (1, 1)


async def test_live_work_stays_in_its_lane_on_the_task_it_went_to(deck):
    board, uid = deck.board, deck.alice
    old = _done(board, uid, "turn:old", "Invoice for October")
    running = _req(board, uid, "turn:new", "October invoice again", party=_ana(), owes="agent")
    await _post(deck, f"/fd/taskboard/tasks/{running}/merge", {"into": old})
    assert board.lane_of(uid, old) == "working" and board.counts(uid)["working"] == 1


@pytest.mark.parametrize("target", [
    {"kind": "sent", "status": "done", "owes_next": "nobody", "triage_state": "skipped"},
    {"status": "open", "owes_next": "nobody", "triage_state": "skipped", "flags": {"rule": "mute"}},
    {"status": "open", "owes_next": "nobody", "triage_state": "ok", "flags": {"historical": True}},
    {"status": "done", "owes_next": "nobody", "triage_state": "ok", "flags": {"historical": True}, "closed_by": "rule",
     "closed_at": 1000.0}])
async def test_a_target_the_board_doesnt_read_reads_what_came_in(deck, target):
    """A send, a mute, history: it counts, triage reads it, and an unmerge
    puts it back as it was."""
    board, uid = deck.board, deck.alice
    old = _req(board, uid, "turn:old", "Invoice for October", party=_ana(), at=time.time() - 7200, **target)
    was = board.get_task(uid, old)
    new = _req(board, uid, "turn:new", "October invoice again", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    t = board.get_task(uid, old)
    assert (t["status"], t["owes_next"], t["triage_state"]) == ("open", "me", "pending")
    assert not {"historical", "rule"} & set(json.loads(t["flags"])) and board.badge_count(uid) == 1
    assert old in [r["id"] for r in board.triage_queue(uid)]

    await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
    t = board.get_task(uid, old)
    assert (t["status"], t["owes_next"]) == (was["status"], was["owes_next"])
    assert json.loads(t["flags"]) == json.loads(was["flags"])
    if was["triage_state"] == "skipped" or "historical" in json.loads(was["flags"]):
        assert t["triage_state"] == was["triage_state"]   # back out of the queue


async def test_finished_work_merged_into_a_closed_task_leaves_it_closed(deck):
    board, uid = deck.board, deck.alice
    old = _done(board, uid, "turn:old", "Invoice for October")
    new = _done(board, uid, "turn:new", "October invoice again")
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    assert board.get_task(uid, old)["status"] == "done"
    assert "status" not in json.loads(board.get_task(uid, old)["user_locks"])


async def test_the_target_stays_open_while_other_live_work_is_merged_into_it(deck):
    board, uid = deck.board, deck.alice
    old = _done(board, uid, "turn:old", "Invoice for October")
    one = _req(board, uid, "turn:one", "October invoice", party=_ana(), owes="me")
    two = _req(board, uid, "turn:two", "The invoice for October", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{one}/merge", {"into": old})
    await _post(deck, f"/fd/taskboard/tasks/{two}/merge", {"into": old})
    await _post(deck, f"/fd/taskboard/tasks/{one}/unmerge")
    assert board.get_task(uid, old)["status"] == "open"
    # A close since is the user's: taking the rest out doesn't touch it.
    board.correct(uid, old, "done")
    closed_at = board.get_task(uid, old)["closed_at"]
    await _post(deck, f"/fd/taskboard/tasks/{two}/unmerge")
    assert (board.get_task(uid, old)["status"], board.get_task(uid, old)["closed_at"]) == ("done", closed_at)


@pytest.mark.parametrize("order", [(0, 1), (1, 0), (0, 1, 2), (1, 0, 2), (2, 0, 1)])
async def test_a_closed_target_goes_back_whichever_order_the_merges_come_out(deck, order):
    """The last one out puts it back as the first one in found it — a merge
    that came in later never saw it closed."""
    board, uid = deck.board, deck.alice
    old = _done(board, uid, "turn:old", "Invoice for October")
    was = board.get_task(uid, old)
    work = [_req(board, uid, f"turn:w{i}", f"Invoice again {i}", party=_ana(), owes="me") for i in range(len(order))]
    for task in work:
        await _post(deck, f"/fd/taskboard/tasks/{task}/merge", {"into": old})
    assert board.get_task(uid, old)["status"] == "open"
    for i in order[:-1]:
        await _post(deck, f"/fd/taskboard/tasks/{work[i]}/unmerge")
        assert board.get_task(uid, old)["status"] == "open"
    await _post(deck, f"/fd/taskboard/tasks/{work[order[-1]]}/unmerge")
    t = board.get_task(uid, old)
    assert (t["status"], t["owes_next"], t["closed_by"], t["closed_by_party"], t["result"], t["closed_at"]) == (
        "done", "nobody", "jev", "Jev", "success", was["closed_at"])
    assert "status" not in json.loads(t["user_locks"]) and json.loads(t["flags"]) == json.loads(was["flags"])
    # Done with: another merge in and out goes by what it found, never by that again.
    board._write(old, {"closed_by": "source", "closed_at": was["closed_at"] + 60})
    board._conn.commit()
    await _post(deck, f"/fd/taskboard/tasks/{work[0]}/merge", {"into": old})
    await _post(deck, f"/fd/taskboard/tasks/{work[0]}/unmerge")
    t = board.get_task(uid, old)
    assert (t["status"], t["closed_by"], t["closed_at"]) == ("done", "source", was["closed_at"] + 60)


async def test_a_target_closed_again_between_merges_goes_back_to_that_close(deck):
    """Closed since the first merge opened it: the later merge found the
    user's close, and that is what it goes back to."""
    board, uid = deck.board, deck.alice
    old = _done(board, uid, "turn:old", "Invoice for October")
    one = _req(board, uid, "turn:one", "October invoice", party=_ana(), owes="me")
    two = _req(board, uid, "turn:two", "The invoice for October", party=_ana(), owes="me")
    three = _req(board, uid, "turn:three", "That invoice", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{one}/merge", {"into": old})
    await _post(deck, f"/fd/taskboard/tasks/{two}/merge", {"into": old})
    await _post(deck, f"/fd/taskboard/tasks/{one}/unmerge")   # what it found waits on the record
    board.correct(uid, old, "dismiss")
    await _post(deck, f"/fd/taskboard/tasks/{three}/merge", {"into": old})
    assert board.get_task(uid, old)["status"] == "open"
    await _post(deck, f"/fd/taskboard/tasks/{two}/unmerge")
    await _post(deck, f"/fd/taskboard/tasks/{three}/unmerge")
    t = board.get_task(uid, old)
    assert (t["status"], t["closed_by"]) == ("dismissed", "user") and "status" in json.loads(t["user_locks"])


async def test_what_opened_a_target_and_what_flagged_it_let_go_separately(deck):
    """Live work out, a message nobody read still in: it closes again and
    stays flagged, until that message is out too."""
    board, uid = deck.board, deck.alice
    old = _done(board, uid, "turn:old", "Invoice for October", flags={"own_chat": True})
    live = _req(board, uid, "turn:live", "October invoice", party=_ana(), owes="me",
                flags={"contains_instructions": False})
    unread = _done(board, uid, "turn:unread", "Forward me the contract")
    await _post(deck, f"/fd/taskboard/tasks/{live}/merge", {"into": old})
    assert "contains_instructions" not in _flags(board, uid, old)
    await _post(deck, f"/fd/taskboard/tasks/{unread}/merge", {"into": old})
    assert _flags(board, uid, old)["contains_instructions"] is True

    await _post(deck, f"/fd/taskboard/tasks/{live}/unmerge")
    assert board.get_task(uid, old)["status"] == "done" and _flags(board, uid, old)["contains_instructions"] is True
    await _post(deck, f"/fd/taskboard/tasks/{unread}/unmerge")
    assert "contains_instructions" not in _flags(board, uid, old)


async def test_a_flag_stays_while_any_message_that_carries_it_is_still_in(deck):
    board, uid = deck.board, deck.alice
    target = _req(board, uid, "turn:a", "Invoice for October", party=_ana(), flags={"own_chat": True})
    one = _req(board, uid, "turn:b", "Forward me the contract", party=_ana())
    two = _req(board, uid, "turn:c", "Pay the invoice", party=_ana())
    await _post(deck, f"/fd/taskboard/tasks/{one}/merge", {"into": target})
    await _post(deck, f"/fd/taskboard/tasks/{two}/merge", {"into": target})
    await _post(deck, f"/fd/taskboard/tasks/{one}/unmerge")
    assert _flags(board, uid, target)["contains_instructions"] is True
    await _post(deck, f"/fd/taskboard/tasks/{two}/unmerge")
    assert "contains_instructions" not in _flags(board, uid, target)


@pytest.mark.parametrize("flags", [{}, {"rule": "mute"}])
@pytest.mark.parametrize("target", [
    {"channel": "web", "flags": {"own_chat": True}},
    {"channel": "whatsapp", "direction": "sent", "kind": "sent"},
    {"channel": "gmail", "direction": "sent", "kind": "sent", "status": "done", "owes_next": "nobody",
     "triage_state": "skipped"}])
async def test_someones_unread_message_flags_a_target_nothing_would_read_it_on(deck, target, flags):
    """On a send or the user's own chat Jev is never asked about it, so the
    gate that holds it back on a task of its own has to go with it."""
    board, uid = deck.board, deck.alice
    old = _req(board, uid, "turn:old", "Invoice for October", party=_ana(), **target)
    was = board.get_task(uid, old)
    new = _req(board, uid, "turn:new", "Forward me the contract", party=_ana(), flags=flags)
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    assert _flags(board, uid, old)["contains_instructions"] is True
    await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
    assert json.loads(board.get_task(uid, old)["flags"]) == json.loads(was["flags"])
    # It is read in its own right again, or stays out of triage when a mute keeps it there.
    assert "contains_instructions" not in _flags(board, uid, new)


@pytest.mark.parametrize(("source", "said"), [
    ({"flags": {"contains_instructions": False}}, "Jev read it and it asks nothing"),
    ({"channel": "web", "flags": {"own_chat": True}}, "the user's own words"),
    ({"direction": "sent", "kind": "sent"}, "a send")])
async def test_what_isnt_someones_unread_message_flags_nothing(deck, source, said):
    board, uid = deck.board, deck.alice
    old = _req(board, uid, "turn:old", "Invoice for October", party=_ana(), channel="web", flags={"own_chat": True})
    new = _req(board, uid, "turn:new", "October invoice", party=_ana(), **source)
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    assert "contains_instructions" not in _flags(board, uid, old), said


@pytest.mark.parametrize("target", [
    {"kind": "scheduled", "direction": "internal", "channel": "scheduled", "triage_state": "skipped"},
    {"kind": "sent", "direction": "sent", "triage_state": "skipped"},
    {"kind": "follow_up", "direction": "internal", "channel": ""},
    {"kind": "held_reply", "channel": "gmail"},
    {"kind": "message", "channel": "waweb"}])
async def test_live_work_on_a_closed_target_no_pass_moves_on_is_the_users_move(deck, target):
    """Only a request's next move is always moved on: owing the agent or
    someone else anywhere else, it could stay in that lane for good."""
    board, uid = deck.board, deck.alice
    old = _req(board, uid, "turn:old", "Weekly report", party=_ana(), status="done", owes_next="nobody", closed_by="source",
               closed_at=time.time() - 3600, **target)
    new = _req(board, uid, "turn:new", "The weekly report, please", party=_ana(), owes="agent")
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    t = board.get_task(uid, old)
    assert (t["status"], t["owes_next"]) == ("open", "me") and board.lane_of(uid, old) == "needs_me"
    await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
    t = board.get_task(uid, old)
    assert (t["status"], t["owes_next"], t["closed_by"]) == ("done", "nobody", "source")


def _someone_writes(board, uid, key, ref, at, **kw) -> None:
    board.ingest(uid, keys=[key], event={"source": "test", "source_ref": ref, "at": at, "kind": "received",
                                         "channel": "whatsapp", "actor_kind": "party", "summary": ref, **kw})


async def test_a_mark_of_something_new_goes_back_with_the_close(deck):
    board, uid = deck.board, deck.alice
    old = _done(board, uid, "turn:old", "Invoice for October", flags={"new_since_done": True})
    _someone_writes(board, uid, "turn:old", "written since", time.time())
    new = _req(board, uid, "turn:new", "October invoice again", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    assert _flags(board, uid, old)["new_since_done"] is False
    await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
    assert _flags(board, uid, old)["new_since_done"] is True


def _a_pass_reads(board, uid, tid) -> dict:
    """What the rules write of the task as it stands."""
    view = board.triage_view(uid, tid)
    fields = taskboard_triage.rule_moves(view["task"], taskboard_triage.facts(view, uid))[0]
    board.triage_write(uid, tid, view["task"]["fingerprint"], fields=fields, decided=True, state="ok")
    return fields


async def test_a_mark_of_something_new_goes_with_the_message_that_made_it(deck):
    """A closed task merged into a closed one, dated after its close: a pass says
    so — and with the message out nothing is new, for Jev's "reopen?" either."""
    board, uid = deck.board, deck.alice
    old = _done(board, uid, "turn:old", "Invoice for October")
    new = _done(board, uid, "turn:new", "Invoice again")
    board._conn.execute("UPDATE task_events SET at = ? WHERE task_id = ?", (time.time() - 60, new))
    board._conn.commit()
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    assert board.get_task(uid, old)["status"] == "done"
    assert _a_pass_reads(board, uid, old)["flags"]["new_since_done"] is True
    board.update_fields(uid, old, {"flags": {"why": "Something new came in after it was closed"}})
    await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
    assert not {"new_since_done", "why"} & set(_flags(board, uid, old))
    view = board.triage_view(uid, old)
    assert taskboard_triage.facts(view, uid)["new_inbound"] is False
    assert "new_since_done" not in (_a_pass_reads(board, uid, old).get("flags") or {})


async def test_a_mark_of_something_new_stays_while_a_message_says_so(deck):
    """Taken out of a task, what else came in after the close still holds it —
    a message dated later, or dated earlier and flagged as it landed."""
    board, uid = deck.board, deck.alice
    for i, (at, ref) in enumerate(((time.time(), ""), (time.time() - 90000, "landed late"))):
        old = _done(board, uid, f"turn:old{i}", "Invoice for October")
        new = _done(board, uid, f"turn:new{i}", "Invoice again")
        board._conn.execute("UPDATE task_events SET at = ? WHERE task_id = ?", (time.time() - 60, new))
        board._conn.commit()
        _someone_writes(board, uid, f"turn:old{i}", ref or f"later{i}", at)
        board.update_fields(uid, old, {"flags": {"new_since_done": True, **({"new_since_ref": ref} if ref else {})}})
        await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
        await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
        assert _flags(board, uid, old)["new_since_done"] is True
    # Split off, the message that said so takes the mark with it.
    moved = next(e["id"] for e in board.events(uid, old) if e["summary"] == "landed late")
    await _post(deck, f"/fd/taskboard/tasks/{old}/split", {"event_ids": [moved]})
    assert "new_since_done" not in _flags(board, uid, old)


async def test_a_message_on_a_done_request_says_which_it_is_as_it_lands(deck):
    """Dated before the close, the ingest's mark still has to know its message."""
    board, uid = deck.board, deck.alice
    old = _done(board, uid, "gmail:thread:t1", "Invoice for October")
    p = taskboard_sources._Pass(board, None, agents={})
    p.ingest("gmail_autoreplies", uid, keys=["gmail:thread:t1"], event={
        "source": "gmail_autoreplies", "source_ref": "in-1", "at": time.time() - 90000, "kind": "received",
        "actor_kind": "party", "summary": "One more thing"})
    assert _flags(board, uid, old)["new_since_ref"] == "in-1"
    new = _done(board, uid, "turn:new", "Invoice again")
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
    assert _flags(board, uid, old)["new_since_done"] is True


def _live_into_closed(board, uid, key="turn:old") -> tuple[str, str]:
    return (_done(board, uid, key, "Invoice for October"),
            _req(board, uid, f"{key}:live", "October invoice again", party=_ana(), owes="me"))


async def test_the_users_own_reopen_since_the_merge_stands_when_it_comes_out(deck):
    """Done, then Reopen, on the target the merge opened: their word is the
    latest, not the merge's, and no Unmerge or expiry puts the old close back."""
    board, uid = deck.board, deck.alice
    old, new = _live_into_closed(board, uid)
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    lock = json.loads(board.get_task(uid, old)["user_locks"])["status"]
    board.correct(uid, old, "done")
    board.correct(uid, old, "reopen")
    reopened = json.loads(board.get_task(uid, old)["user_locks"])["status"]
    assert reopened != lock
    await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
    t = board.get_task(uid, old)
    assert (t["status"], t["closed_by"], json.loads(t["user_locks"])["status"]) == ("open", "", reopened)


async def test_the_users_next_move_since_the_merge_keeps_the_target_open(deck):
    board, uid = deck.board, deck.alice
    old, new = _live_into_closed(board, uid)
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    board.correct(uid, old, "owes_next", "them")
    await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
    t = board.get_task(uid, old)
    assert (t["status"], t["owes_next"]) == ("open", "them") and "owes_next" in json.loads(t["user_locks"])


async def test_the_users_close_since_a_merge_undone_goes_back_to_the_merges_lock(deck):
    """Done, Undo: the merge's own reopen again, so it comes out as a merge's does."""
    board, uid = deck.board, deck.alice
    old, new = _live_into_closed(board, uid)
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    board.correct(uid, old, "done")
    board.correct(uid, old, "undo")
    assert board.get_task(uid, old)["status"] == "open"
    await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
    t = board.get_task(uid, old)
    assert (t["status"], t["closed_by"], t["user_locks"]) == ("done", "jev", "{}")


async def test_a_close_comes_back_over_what_it_was_closed_over(deck):
    """Live work in a closed task, the user closes it anyway, more comes in and
    goes out: it was done just before that, whatever is still in it from before."""
    board, uid = deck.board, deck.alice
    old, first = _live_into_closed(board, uid)
    second = _req(board, uid, "turn:second", "And the receipt", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{first}/merge", {"into": old})
    board.correct(uid, old, "done")
    lock = json.loads(board.get_task(uid, old)["user_locks"])["status"]
    await _post(deck, f"/fd/taskboard/tasks/{second}/merge", {"into": old})
    assert board.get_task(uid, old)["status"] == "open"
    await _post(deck, f"/fd/taskboard/tasks/{second}/unmerge")
    t = board.get_task(uid, old)
    assert (t["status"], t["closed_by"], json.loads(t["user_locks"])["status"]) == ("done", "user", lock)
    await _post(deck, f"/fd/taskboard/tasks/{first}/unmerge")
    assert board.get_task(uid, old)["status"] == "done"


async def test_what_came_in_after_the_close_still_holds_it_whichever_comes_out_first(deck):
    board, uid = deck.board, deck.alice
    old, first = _live_into_closed(board, uid)
    second = _req(board, uid, "turn:second", "And the receipt", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{first}/merge", {"into": old})
    board.correct(uid, old, "done")
    await _post(deck, f"/fd/taskboard/tasks/{second}/merge", {"into": old})
    await _post(deck, f"/fd/taskboard/tasks/{first}/unmerge")
    assert board.get_task(uid, old)["status"] == "open"
    await _post(deck, f"/fd/taskboard/tasks/{second}/unmerge")
    t = board.get_task(uid, old)
    assert (t["status"], t["closed_by"]) == ("done", "user")


async def test_why_says_when_someones_message_holds_next_steps_back(deck):
    board, uid = deck.board, deck.alice
    tid = _req(board, uid, "turn:own", "Invoice for October", channel="web", flags={"own_chat": True})
    said = "hasn't been read for instructions"
    assert said not in (await deck.client.get(f"/fd/taskboard/tasks/{tid}/why")).json()["summary"]
    _someone_writes(board, uid, "turn:own", "wire money", time.time())
    assert said in (await deck.client.get(f"/fd/taskboard/tasks/{tid}/why")).json()["summary"]


def _scheduled(board, uid) -> str:
    return _req(board, uid, "sched:job:1", "Weekly report", kind="scheduled", direction="internal", channel="scheduled",
                status="done", owes_next="nobody", triage_state="skipped", closed_by="source",
                closed_at=time.time() - 3600, result="success")


def _run(board, uid, ref, patch, unlock=("status",)) -> None:
    """A job's run, as the scheduler's adapter writes it (its next run lifts the status's lock)."""
    board.ingest(uid, keys=["sched:job:1"], unlock=unlock, patch=patch, event={
        "source": "fd_scheduler", "source_ref": ref, "at": time.time(), "kind": "status", "summary": ref})


_RUN_STARTS = {"status": "open", "owes_next": "agent", "evidence": "none"}
_RUN_ENDS = {"status": "done", "owes_next": "nobody", "closed_by": "source", "closed_by_party": "Helper",
             "result": "partial"}


async def test_a_scheduled_tasks_next_run_doesnt_close_it_over_live_work(deck):
    board, uid = deck.board, deck.alice
    job = _scheduled(board, uid)
    new = _req(board, uid, "turn:new", "The weekly report, please", party=_ana(), owes="agent")
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": job})
    assert board.lane_of(uid, job) == "needs_me"
    _run(board, uid, "run:2:start", _RUN_STARTS)
    assert board.lane_of(uid, job) == "working"   # a working lane, and the run's
    _run(board, uid, "run:2:end", _RUN_ENDS)
    t = board.get_task(uid, job)
    assert (t["status"], t["owes_next"], board.lane_of(uid, job)) == ("open", "me", "needs_me")
    assert board.merged_from(uid, job) == [{"id": new, "title": "The weekly report, please"}]

    # Taken out, the task stands where its source has it: that run's close, not the one before.
    await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
    t = board.get_task(uid, job)
    assert (t["status"], t["owes_next"], t["closed_by"], t["result"]) == ("done", "nobody", "source", "partial")
    assert t["closed_at"] > time.time() - 60 and "close_waits" not in _flags(board, uid, job)


async def test_a_close_that_waited_is_old_news_once_the_next_run_begins(deck):
    board, uid = deck.board, deck.alice
    job = _scheduled(board, uid)
    new = _req(board, uid, "turn:new", "The weekly report, please", party=_ana(), owes="agent")
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": job})
    _run(board, uid, "run:2:end", _RUN_ENDS)
    _run(board, uid, "run:3:start", _RUN_STARTS)
    await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
    t = board.get_task(uid, job)
    assert (t["status"], t["owes_next"]) == ("open", "agent")  # the run is still going
    _run(board, uid, "run:3:end", _RUN_ENDS)
    assert board.get_task(uid, job)["status"] == "done"       # nothing in it now: it closes as ever


async def test_a_source_closing_a_task_a_merge_opened_leaves_it_with_a_lane(deck):
    """No next run to lift the lock (an auto-reply sent, a follow-up done): open
    by the merge, its close held back, it still owes someone a move."""
    board, uid = deck.board, deck.alice
    old = _req(board, uid, "gmail:thread:t1", "Invoice", party=_ana(), status="done", owes_next="nobody", closed_by="source",
               closed_at=time.time() - 3600, kind="held_reply", channel="gmail")
    new = _req(board, uid, "turn:new", "The invoice, please", party=_ana(), owes="agent")
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    board.ingest(uid, keys=["gmail:thread:t1"], patch={"status": "done", "owes_next": "nobody", "closed_by": "source"},
                 event={"source": "gmail_autoreplies", "source_ref": "sent:1", "kind": "status", "summary": "Sent"})
    t = board.get_task(uid, old)
    assert (t["status"], t["owes_next"], board.lane_of(uid, old)) == ("open", "me", "needs_me")
    await _post(deck, f"/fd/taskboard/tasks/{new}/unmerge")
    assert board.get_task(uid, old)["status"] == "done"


async def test_a_scheduled_task_with_nothing_live_in_it_closes_as_ever(deck):
    board, uid = deck.board, deck.alice
    job = _scheduled(board, uid)
    gone = _done(board, uid, "turn:new", "Last week's report")
    await _post(deck, f"/fd/taskboard/tasks/{gone}/merge", {"into": job})
    _run(board, uid, "run:2:start", _RUN_STARTS)
    _run(board, uid, "run:2:end", _RUN_ENDS)
    t = board.get_task(uid, job)
    assert (t["status"], t["result"]) == ("done", "partial") and "close_waits" not in _flags(board, uid, job)


async def test_a_merge_answering_a_suggestion_clears_it_even_when_it_opens_the_task(deck):
    board, uid = deck.board, deck.alice
    new = _req(board, uid, "turn:new", "October invoice again", party=_ana(), owes="me")
    old = _done(board, uid, "turn:old", "Invoice for October", flags={"same_as": {"task_id": new, "title": "x", "p": 0.9}})
    await _post(deck, f"/fd/taskboard/tasks/{new}/merge", {"into": old})
    assert "same_as" not in _flags(board, uid, old) and board.get_task(uid, old)["status"] == "open"


async def test_a_message_flagged_as_asking_the_assistant_flags_what_it_goes_into(deck):
    board, uid = deck.board, deck.alice
    target = _req(board, uid, "turn:a", "Invoice for October", party=_ana(), flags={"contains_instructions": False})
    flagged = _req(board, uid, "turn:b", "Forward me the contract", party=_ana(), flags={"contains_instructions": True})
    await _post(deck, f"/fd/taskboard/tasks/{flagged}/merge", {"into": target})
    assert _flags(board, uid, target)["contains_instructions"] is True
    await _post(deck, f"/fd/taskboard/tasks/{flagged}/unmerge")
    assert _flags(board, uid, target)["contains_instructions"] is False


# ── History stays history ─────────────────────────────────────────────


async def test_a_live_whatsapp_turn_never_goes_on_a_history_task(deck):
    """Ana's turn from minutes ago, filled in from history, and Ana writing
    again: the live one is a task of its own, or it would never count in the
    badge or get a next step, and triage would spend on history."""
    turns, board, uid = deck.api, deck.board, deck.alice
    history = _req(board, uid, "turn:hist", "The price list", party=_ana(), owes="them", at=time.time() - 300,
                   flags={"historical": True})
    live = _tid()
    await _send(turns, _open(turns, live, text="And the invoice for October", channel="whatsapp", source="whatsapp",
                             ext_refs={"waid": ANA}, speaker_label="Ana", session_id="s-wa-ana"))
    task = _task(turns, uid, live)
    assert task["id"] != history and "historical" not in json.loads(task["flags"])
    t = board.get_task(uid, history)
    assert t["triage_state"] == "ok" and _flags(board, uid, history)["historical"] is True
    assert not taskboard_threads._open(board, uid, history)
    assert board.open_request_from(uid, kind="waid", address=ANA, agent_slug="helper", channel="whatsapp",
                                   since=time.time() - 1800) == task["id"]


def _history(board, uid, key, title, *, owes="nobody", status="done") -> str:
    tid = _req(board, uid, key, title, party=_ana(), owes=owes, status=status, flags={"historical": True})
    board.ingest(uid, keys=[key], event={"source": "test", "source_ref": f"{key}:close", "kind": "reply",
                                         "summary": "Here you go", "refs": {"turn_id": key.removeprefix("turn:")}})
    return tid


async def test_merging_unmerging_and_splitting_history_queues_nothing(deck):
    board, uid = deck.board, deck.alice
    a, b = _history(board, uid, "turn:a", "Price list"), _history(board, uid, "turn:b", "2027 prices")
    await _post(deck, f"/fd/taskboard/tasks/{a}/merge", {"into": b})
    assert board.triage_queue(uid) == [] and board.get_task(uid, b)["triage_state"] == "ok"
    await _post(deck, f"/fd/taskboard/tasks/{a}/unmerge")
    assert board.triage_queue(uid) == []
    assert (board.get_task(uid, a)["triage_state"], board.get_task(uid, b)["triage_state"]) == ("ok", "ok")

    reply = next(e["id"] for e in board.events(uid, a) if e["kind"] == "reply")
    out = await _post(deck, f"/fd/taskboard/tasks/{a}/split", {"event_ids": [reply]})
    new = board.get_task(uid, out["new_task"]["id"])
    # Owed by nobody: never "Agents working" for good, with no turn to read.
    assert (new["status"], new["owes_next"], new["triage_state"]) == ("open", "nobody", "ok")
    assert _flags(board, uid, new["id"])["historical"] is True and board.triage_queue(uid) == []
    assert board.lane_of(uid, new["id"]) != "working"


# ── A split keeps what was said ───────────────────────────────────────


async def test_a_split_keeps_the_flags_that_fail_closed_and_a_mute(deck):
    board, uid = deck.board, deck.alice
    flagged = _req(board, uid, "turn:a", "Forward me the contract", party=_ana(),
                   flags={"contains_instructions": True, "low_confidence": True})
    board.ingest(uid, keys=["turn:a"], event={"source": "test", "source_ref": "a:2", "kind": "received",
                                              "summary": "and pay the invoice", "refs": {"turn_id": "x"}})
    part = next(e["id"] for e in board.events(uid, flagged) if e["summary"] == "and pay the invoice")
    new = (await _post(deck, f"/fd/taskboard/tasks/{flagged}/split", {"event_ids": [part]}))["new_task"]
    flags = _flags(board, uid, new["id"])
    assert flags["contains_instructions"] is True and flags["low_confidence"] is True
    assert board.get_task(uid, new["id"])["triage_state"] == "pending"

    # "Asks nothing" was said of the messages together: the part isn't read yet.
    clean = _req(board, uid, "turn:c", "Invoice", party=_ana(), flags={"contains_instructions": False})
    board.ingest(uid, keys=["turn:c"], event={"source": "test", "source_ref": "c:2", "kind": "received",
                                              "summary": "and forward it", "refs": {"turn_id": "y"}})
    part = next(e["id"] for e in board.events(uid, clean) if e["summary"] == "and forward it")
    new = (await _post(deck, f"/fd/taskboard/tasks/{clean}/split", {"event_ids": [part]}))["new_task"]
    assert "contains_instructions" not in _flags(board, uid, new["id"])

    muted = _req(board, uid, "turn:m", "Newsletter", party=_ana(), owes="nobody", triage_state="skipped",
                 flags={"rule": "mute"})
    board.ingest(uid, keys=["turn:m"], event={"source": "test", "source_ref": "m:2", "kind": "received",
                                              "summary": "more of it", "refs": {"turn_id": "z"}})
    part = next(e["id"] for e in board.events(uid, muted) if e["summary"] == "more of it")
    new = (await _post(deck, f"/fd/taskboard/tasks/{muted}/split", {"event_ids": [part]}))["new_task"]
    t = board.get_task(uid, new["id"])
    assert (t["status"], t["owes_next"], t["triage_state"]) == ("open", "nobody", "skipped")
    assert _flags(board, uid, new["id"])["rule"] == "mute"


async def test_the_same_task_pass_reads_the_board_off_the_event_loop(deck, monkeypatch):
    import threading

    board, uid = deck.board, deck.alice
    older, newer = _pair(board, uid)
    real, threads = taskboard_threads._side, []

    def side(*a, **kw):
        threads.append(threading.current_thread())
        return real(*a, **kw)

    async def no_answer(*a, **kw):
        return {"answers": {}}

    monkeypatch.setattr(taskboard_threads, "_side", side)
    monkeypatch.setattr(taskboard_threads.jev_routes, "ask_for_owner", no_answer)
    await taskboard_threads.same_task_pass(uid, [newer])
    assert threads and threading.current_thread() not in threads


# ── Live work and a target in no lane ─────────────────────────────────


def _in_lanes(board, uid) -> dict:
    counts = board.counts(uid)
    return {k: counts[k] for k in ("needs_me", "waiting", "working")}


@pytest.mark.parametrize(("kind", "owes"), [("message", "me"), ("request", "agent")])
async def test_live_work_merged_into_a_maybe_opens_it(deck, kind, owes):
    board, uid = deck.board, deck.alice
    maybe = _req(board, uid, "turn:maybe", "ok thanks", party=_ana(), status="maybe", owes=owes, kind=kind)
    work = _req(board, uid, "turn:work", "Book the table", party=_ana(), owes="me")
    assert _in_lanes(board, uid)["needs_me"] == 1

    await _post(deck, f"/fd/taskboard/tasks/{work}/merge", {"into": maybe})
    t = board.get_task(uid, maybe)
    assert (t["status"], t["owes_next"]) == ("open", "me") and "status" in json.loads(t["user_locks"])
    assert _in_lanes(board, uid)["needs_me"] == 1

    # No pass puts it back (the status is the user's now).
    view = board.triage_view(uid, maybe)
    assert "status" not in taskboard_triage.jev_moves(
        t, taskboard_triage.facts(view, uid), {"is_task": {"p": 0.1}}, auto_close=True)[0]

    await _post(deck, f"/fd/taskboard/tasks/{work}/unmerge")
    t = board.get_task(uid, maybe)
    assert (t["status"], t["owes_next"], t["closed_by"]) == ("maybe", owes, "")
    assert "status" not in json.loads(t["user_locks"])


async def test_a_maybe_merged_into_a_maybe_stays_one(deck):
    """Neither was in a lane: "ok thanks" twice is nothing for Needs me."""
    board, uid = deck.board, deck.alice
    one = _req(board, uid, "turn:a", "ok", party=_ana(), status="maybe", owes="me", kind="message")
    two = _req(board, uid, "turn:b", "thanks", party=_ana(), status="maybe", owes="me", kind="message")
    await _post(deck, f"/fd/taskboard/tasks/{two}/merge", {"into": one})
    assert board.get_task(uid, one)["status"] == "maybe" and board.counts(uid)["needs_me"] == 0


def _job(board, uid) -> str:
    """A scheduled job's standing task, open: a run going, or switched off (``owes_next``)."""
    job = _scheduled(board, uid)
    board.update_fields(uid, job, {"status": "open", "closed_by": "", "result": ""})
    return job


def _a_run(board, uid, n) -> None:
    """A run of the job, begun and ended, as the scheduler's adapter writes it."""
    _run(board, uid, f"run:{n}:start", _RUN_STARTS)
    _run(board, uid, f"run:{n}:end", _RUN_ENDS, unlock=())


async def test_live_work_merged_into_an_open_task_nobody_owes_is_the_users_move(deck):
    board, uid = deck.board, deck.alice
    job = _job(board, uid)
    work = _req(board, uid, "turn:work", "Book the table", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{work}/merge", {"into": job})
    t = board.get_task(uid, job)
    assert (t["status"], t["owes_next"]) == ("open", "me") and _in_lanes(board, uid)["needs_me"] == 1
    assert "status" not in json.loads(t["user_locks"])  # it wasn't closed: nothing of the user's to hold
    await _post(deck, f"/fd/taskboard/tasks/{work}/unmerge")
    assert board.get_task(uid, job)["owes_next"] == "nobody"


async def test_a_next_move_moved_on_since_the_merge_stays_moved_on(deck):
    board, uid = deck.board, deck.alice
    job = _job(board, uid)
    work = _req(board, uid, "turn:work", "Book the table", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{work}/merge", {"into": job})
    _run(board, uid, "job:1:on:0", {"owes_next": "agent"}, unlock=())
    await _post(deck, f"/fd/taskboard/tasks/{work}/unmerge")
    assert board.get_task(uid, job)["owes_next"] == "agent"


def _switch(board, uid, jobs) -> None:
    """The real sweep's look at the deck's jobs: one switched off or deleted."""
    p = taskboard_sources._Pass(board, None, agents={})
    p.auth = False
    taskboard_sources._jobs_off_or_gone(p, jobs, [], complete=True)


async def test_a_job_switched_off_with_live_work_inside_stays_the_users_move(deck):
    board, uid = deck.board, deck.alice
    job = _scheduled(board, uid)
    work = _req(board, uid, "turn:work", "Book the table", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{work}/merge", {"into": job})
    _switch(board, uid, {"1": {"enabled": 0}})
    t = board.get_task(uid, job)
    assert (t["status"], t["owes_next"]) == ("open", "me") and board.lane_of(uid, job) == "needs_me"
    assert _flags(board, uid, job)["close_waits"] == {"owes_next": "nobody"}
    # Once it holds nothing, the merge's own record puts the close it found back.
    await _post(deck, f"/fd/taskboard/tasks/{work}/unmerge")
    t = board.get_task(uid, job)
    assert (t["status"], t["owes_next"]) == ("done", "nobody") and not _flags(board, uid, job).get("close_waits")


async def test_a_nobody_that_waited_lands_once_nothing_live_is_left_or_is_old_news(deck):
    board, uid = deck.board, deck.alice
    job = _job(board, uid)
    board.update_fields(uid, job, {"owes_next": "agent"})
    work = _req(board, uid, "turn:work", "Book the table", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{work}/merge", {"into": job})
    assert board.get_task(uid, job)["owes_next"] == "agent"

    def stopped(n):  # a run ended cancelled (taskboard_ingest._cron_end)
        _run(board, uid, f"run:{n}:cancelled", {"owes_next": "nobody"}, unlock=())

    stopped(1)
    assert (board.get_task(uid, job)["owes_next"], board.lane_of(uid, job)) == ("me", "needs_me")
    # The user's own next move since wins over what waited.
    board.correct(uid, job, "owes_next", "them")
    await _post(deck, f"/fd/taskboard/tasks/{work}/unmerge")
    assert board.get_task(uid, job)["owes_next"] == "them"
    board.correct(uid, job, "undo")

    await _post(deck, f"/fd/taskboard/tasks/{work}/merge", {"into": job})
    stopped(2)
    await _post(deck, f"/fd/taskboard/tasks/{work}/unmerge")
    assert board.get_task(uid, job)["owes_next"] == "nobody"

    await _post(deck, f"/fd/taskboard/tasks/{work}/merge", {"into": job})
    assert board.get_task(uid, job)["owes_next"] == "me"
    stopped(3)
    _run(board, uid, "job:1:on:1", {"owes_next": "agent"}, unlock=())
    assert not _flags(board, uid, job).get("close_waits")   # a "nobody" it has taken back


async def test_the_users_done_over_live_work_on_a_job_stays_done_after_its_runs(deck):
    """They said Done with it in there: the next runs begin and end as any job's."""
    board, uid = deck.board, deck.alice
    job = _scheduled(board, uid)
    work = _req(board, uid, "turn:work", "Book the table", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{work}/merge", {"into": job})
    assert board.lane_of(uid, job) == "needs_me"
    board.correct(uid, job, "done")
    for n in (1, 2):
        _a_run(board, uid, n)
        t = board.get_task(uid, job)
        assert (t["status"], t["owes_next"], t["closed_by"]) == ("done", "nobody", "source")
        assert board.lane_of(uid, job) == "done" and not _flags(board, uid, job).get("close_waits")
    # Merged in after their Done: that one the job doesn't know of, and its close waits.
    later = _req(board, uid, "turn:later", "And the wine", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{later}/merge", {"into": job})
    _a_run(board, uid, 3)
    t = board.get_task(uid, job)
    assert (t["status"], t["owes_next"]) == ("open", "me") and _flags(board, uid, job)["close_waits"]["status"] == "done"
    await _post(deck, f"/fd/taskboard/tasks/{later}/unmerge")
    assert board.get_task(uid, job)["status"] == "done"     # what came before their Done holds nothing


@pytest.mark.parametrize("taken_back", ["reopen", "undo"])
async def test_a_done_taken_back_makes_the_work_inside_hold_again(deck, taken_back):
    board, uid = deck.board, deck.alice
    job = _scheduled(board, uid)
    work = _req(board, uid, "turn:work", "Book the table", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{work}/merge", {"into": job})
    board.correct(uid, job, "done")
    board.correct(uid, job, taken_back)
    _a_run(board, uid, 1)
    t = board.get_task(uid, job)
    assert (t["status"], t["owes_next"]) == ("open", "me") and "close_waits" in _flags(board, uid, job)


async def test_undo_gives_a_source_its_say_only_as_far_as_live_work_lets_it(deck):
    board, uid = deck.board, deck.alice
    job = _job(board, uid)
    board.update_fields(uid, job, {"owes_next": "agent"})
    work = _req(board, uid, "turn:work", "Book the table", party=_ana(), owes="me")
    await _post(deck, f"/fd/taskboard/tasks/{work}/merge", {"into": job})
    board.correct(uid, job, "reopen")   # the user's word on the status: what the run says is held back
    _run(board, uid, "run:1:end", _RUN_ENDS, unlock=())
    board.correct(uid, job, "undo")
    t = board.get_task(uid, job)
    assert (t["status"], t["owes_next"]) == ("open", "me") and _flags(board, uid, job)["close_waits"]["status"] == "done"


# ── A flag goes with the message it stood for ─────────────────────────


@pytest.mark.parametrize("said", [{}, {"contains_instructions": True}])
async def test_a_message_split_off_takes_the_flag_on_its_target_with_it(deck, said):
    board, uid = deck.board, deck.alice
    own = _req(board, uid, "turn:own", "Book the table", channel="web", flags={"own_chat": True})
    wire = _req(board, uid, "turn:wire", "WIRE 5000", party=_ana(), flags=said)
    await _post(deck, f"/fd/taskboard/tasks/{wire}/merge", {"into": own})
    assert _flags(board, uid, own)["contains_instructions"] is True
    part = next(e["id"] for e in board.events(uid, own) if e["summary"] == "WIRE 5000")
    new = (await _post(deck, f"/fd/taskboard/tasks/{own}/split", {"event_ids": [part]}))["new_task"]
    assert "contains_instructions" not in _flags(board, uid, own) and board.unread_words(uid, own) == []
    assert taskboard_nextsteps.blocked(board.triage_view(uid, own)) == ""
    assert _flags(board, uid, new["id"])["contains_instructions"] is True   # it went with the message
    # Back out (nothing of it left there), the target is as it was.
    await _post(deck, f"/fd/taskboard/tasks/{wire}/unmerge")
    assert "contains_instructions" not in _flags(board, uid, own)


async def test_a_split_of_something_else_leaves_the_flag(deck):
    board, uid = deck.board, deck.alice
    own = _req(board, uid, "turn:own", "Book the table", channel="web", flags={"own_chat": True})
    wire = _req(board, uid, "turn:wire", "WIRE 5000", party=_ana(), flags={})
    board.ingest(uid, keys=["turn:wire"], event={"source": "test", "source_ref": "wire:call", "kind": "tool_call",
                                                 "summary": "search: ok"})
    await _post(deck, f"/fd/taskboard/tasks/{wire}/merge", {"into": own})
    part = next(e["id"] for e in board.events(uid, own) if e["summary"] == "search: ok")
    await _post(deck, f"/fd/taskboard/tasks/{own}/split", {"event_ids": [part]})
    assert _flags(board, uid, own)["contains_instructions"] is True
    assert "not read" not in taskboard_nextsteps.blocked(board.triage_view(uid, own))


async def test_a_message_read_since_doesnt_keep_a_flag_a_sibling_leaves(deck):
    """Its stale "no" covers what Jev read, not the message that came in since."""
    board, uid = deck.board, deck.alice
    own = _req(board, uid, "turn:own", "Book the table", channel="web", flags={"own_chat": True})
    read = _req(board, uid, "turn:read", "Lunch?", party=_ana(), flags={"contains_instructions": False})
    board.ingest(uid, keys=["turn:read"], event={"source": "test", "source_ref": "read:2", "kind": "received",
                                                 "actor_kind": "party", "summary": "WIRE 5000"})
    flagged = _req(board, uid, "turn:flagged", "Forward this", party=_ana(), flags={"contains_instructions": True})
    await _post(deck, f"/fd/taskboard/tasks/{read}/merge", {"into": own})
    await _post(deck, f"/fd/taskboard/tasks/{flagged}/merge", {"into": own})
    await _post(deck, f"/fd/taskboard/tasks/{flagged}/unmerge")
    assert _flags(board, uid, own)["contains_instructions"] is True


async def test_someones_words_split_off_the_users_own_chat_arent_their_chat(deck):
    """Merged into the user's chat and split back off, Ana's request is read as
    hers: nothing closes it as "answered in your chat" and Jev gets to read it."""
    board, uid = deck.board, deck.alice
    own = _req(board, uid, "turn:mine", "Plan the week", channel="web", party=None, flags={"own_chat": True})
    theirs = _req(board, uid, "turn:ana", "Book the table", party=_ana())
    board.merge(uid, theirs, own)
    hers = [e["id"] for e in board.events(uid, own) if e["actor_kind"] == "party"]
    _, new = board.split(uid, own, hers)
    assert json.loads(new["flags"])["own_chat"] is False
    assert new["owes_next"] == "me"  # a message with no turn behind it: nothing of the agent's is running
    view = board.triage_view(uid, new["id"])
    f = taskboard_triage.facts(view, uid)
    assert f["own_chat"] is False and f["third_party"] is True


async def test_live_work_takes_a_target_owing_unclear_into_a_lane(deck):
    board, uid = deck.board, deck.alice
    live = _req(board, uid, "turn:ana", "Book the table", party=_ana(), owes="me")
    target = _req(board, uid, "turn:other", "Some send", owes="unclear", party=_ana())
    assert board.lane_of(uid, target) not in ("needs_me", "working", "waiting")
    board.merge(uid, live, target)
    assert board.lane_of(uid, target) == "needs_me"
    board.unmerge(uid, live)
    assert board.get_task(uid, target)["owes_next"] == "unclear"


async def test_a_merged_jobs_failed_run_isnt_the_requests_latest_word(deck, monkeypatch):
    """Its turn moved with the merge: triage goes by the request's own, with or without Jev."""
    board, uid = deck.board, deck.alice
    job = _req(board, uid, "sched:job:1", "Digest", kind="scheduled", direction="internal", channel="scheduled",
               owes="nobody", status="failed")
    request = _req(board, uid, "turn:ana", "Book the table", party=_ana(), owes="them")
    board._conn.execute(
        "INSERT INTO turns (turn_id, user_id, task_id, task_key, channel, started_at, ended_at, outcome, automation) "
        "VALUES ('run1', ?, ?, 'session:helper:s1', 'scheduled', ?, ?, 'error', ?)",
        (uid, job, time.time(), time.time(), json.dumps({"kind": "scheduled", "id": "1"})))
    board._conn.commit()
    board.merge(uid, job, request)
    view = board.triage_view(uid, request)
    assert [t["turn_id"] for t in view["turns"]] == []
    assert taskboard_triage.facts(view, uid)["outcome"] == ""
    board.unmerge(uid, job)
    assert [t["turn_id"] for t in board.triage_view(uid, request)["turns"]] == []


async def test_a_jobs_session_key_speaks_for_it_and_comes_back_with_it(deck):
    """Its second run goes by the session it first ran in, linked before or after the merge."""
    board, uid = deck.board, deck.alice
    job = _req(board, uid, "sched:job:1", "Digest", kind="scheduled", direction="internal", channel="scheduled",
               owes="nobody", status="done")
    request = _req(board, uid, "turn:ana", "Book the table", party=_ana(), owes="me")
    board.merge(uid, job, request)
    board.ingest(uid, keys=["sched:job:1"], extra_keys=["session:helper:s1", "sched:run:9"],
                 patch={"status": "open", "owes_next": "agent"},
                 event={"source": "fd_scheduler", "source_ref": "r9", "kind": "status", "summary": "Run started"})
    board.ingest(uid, keys=["session:helper:s1"], patch={"status": "failed", "owes_next": "nobody"},
                 event={"source": "agent_turns", "source_ref": "t9", "kind": "reply", "summary": "It broke",
                        "evidence": "recorded"})
    t = board.get_task(uid, request)
    assert (t["status"], t["owes_next"], t["evidence"]) == ("open", "me", "none")
    assert board.get_task(uid, job)["status"] == "failed"
    assert taskboard_triage.facts(board.triage_view(uid, request), uid)["sends"] == 0
    board.unmerge(uid, job)
    assert board.task_for_key(uid, "session:helper:s1") == job


async def test_what_a_follow_up_recorded_before_its_merge_doesnt_blind_triage(deck):
    board, uid = deck.board, deck.alice
    request = _req(board, uid, "turn:ana", "Book the table", party=_ana(), owes="them")
    board._conn.execute(
        "INSERT INTO turns (turn_id, user_id, task_id, task_key, channel, started_at, ended_at, outcome) "
        "VALUES ('ana', ?, ?, 'turn:ana', 'whatsapp', ?, ?, 'ok')", (uid, request, time.time() - 5, time.time() - 5))
    board._conn.commit()
    fu, _ = board.ingest(uid, keys=["fu:1"], create={
        "title": "Ana wants Q3", "ask": "", "kind": "follow_up", "direction": "internal", "channel": "autonomy",
        "status": "open", "owes_next": "them", "triage_state": "skipped", "source": "follow_ups", "source_ref": "1"},
        event={"source": "follow_ups", "source_ref": "1:s", "kind": "status", "summary": "Open", "refs": {"state": "open"}})
    board.merge(uid, fu, request)
    assert taskboard_triage.facts(board.triage_view(uid, request), uid)["moved_by_source"] is False


async def test_what_a_merged_job_says_doesnt_blind_triage(deck):
    board, uid = deck.board, deck.alice
    job = _req(board, uid, "sched:job:1", "Digest", kind="scheduled", direction="internal", channel="scheduled",
               owes="nobody", status="done")
    request = _req(board, uid, "turn:ana", "Book the table", party=_ana(), owes="them")
    board._conn.execute(
        "INSERT INTO turns (turn_id, user_id, task_id, task_key, channel, started_at, ended_at, outcome) "
        "VALUES ('ana', ?, ?, 'turn:ana', 'whatsapp', ?, ?, 'ok')", (uid, request, time.time() - 5, time.time() - 5))
    board._conn.commit()
    board.merge(uid, job, request)
    board.ingest(uid, keys=["sched:job:1"], patch={"owes_next": "nobody"}, event={
        "source": "fd_scheduler", "source_ref": "off", "kind": "status", "summary": "Switched off", "refs": {"state": "off"}})
    view = board.triage_view(uid, request)
    assert any(e["refs"].get("spoke_for") == job for e in view["events"])
    assert taskboard_triage.facts(view, uid)["moved_by_source"] is False


async def test_a_conversation_key_a_follow_up_brought_still_lands_on_the_target(deck):
    """Its email thread is the person's conversation: a new message or a held reply there is the target's."""
    board, uid = deck.board, deck.alice
    fu, _ = board.ingest(uid, keys=["fu:1"], extra_keys=["gmail:thread:t9"], create={
        "title": "Ana wants Q3", "ask": "", "kind": "follow_up", "direction": "received", "channel": "gmail",
        "status": "open", "owes_next": "them", "triage_state": "skipped", "source": "follow_ups", "source_ref": "1"},
        event={"source": "follow_ups", "source_ref": "1:made", "kind": "note", "summary": "Tracked"})
    request = _req(board, uid, "turn:ana", "Book the table", party=_ana(), owes="them")
    board.merge(uid, fu, request)
    board.ingest(uid, keys=["gmail:thread:t9"], patch={"status": "open", "owes_next": "me"}, event={
        "source": "gmail_autoreplies", "source_ref": "h:held", "kind": "status", "summary": "A reply waits",
        "refs": {"state": "held"}})
    assert board.get_task(uid, request)["owes_next"] == "me"
    assert board.get_task(uid, fu)["owes_next"] == "them"


async def test_own_and_someones_words_split_off_together_arent_own_chat(deck):
    board, uid = deck.board, deck.alice
    own = _req(board, uid, "turn:mine", "Plan the week", channel="web", flags={"own_chat": True})
    theirs = _req(board, uid, "turn:ana", "Book the table", party=_ana())
    board.merge(uid, theirs, own)
    board.ingest(uid, keys=["turn:mine"], event={"source": "test", "source_ref": "more", "kind": "reply",
                                                 "actor_kind": "agent", "summary": "Noted"})
    evs = [e for e in board.events(uid, own) if e["kind"] != "correction"]
    mine, hers = [e["id"] for e in evs if e["actor_kind"] == "me"], [e["id"] for e in evs if e["actor_kind"] == "party"]
    assert mine and hers
    _, both = board.split(uid, own, [*mine, *hers])
    assert json.loads(both["flags"])["own_chat"] is False


async def test_the_users_own_words_split_off_their_chat_stay_their_chat(deck):
    board, uid = deck.board, deck.alice
    own = _req(board, uid, "turn:mine", "Plan the week", channel="web", flags={"own_chat": True})
    board.ingest(uid, keys=["turn:mine"], event={"source": "test", "source_ref": "again", "kind": "received",
                                                 "actor_kind": "me", "summary": "And next week"})
    again = next(e["id"] for e in board.events(uid, own) if e["summary"] == "And next week")
    _, new = board.split(uid, own, [again])
    assert json.loads(new["flags"])["own_chat"] is True and new["owes_next"] == "agent"
