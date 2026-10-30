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
    """A person's request on the board, as the ingest leaves one."""
    at = time.time() if at is None else at
    tid, _ = board.ingest(
        uid, keys=[key], parties=[party] if party else [],
        create={"title": title, "ask": title, "kind": "request", "direction": "received", "channel": channel,
                "status": "open", "owes_next": owes, "triage_state": "ok", "agent_slug": "helper",
                "agent_name": "Helper", "agent_owner_id": uid, "created_at": at, **create},
        event={"source": "test", "source_ref": f"{key}:open", "at": at, "kind": "received", "channel": channel,
               "actor_kind": "party", "actor": (party or {}).get("display", ""), "summary": title,
               "refs": {"turn_id": key.removeprefix("turn:")} if key.startswith("turn:") else {}})
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
