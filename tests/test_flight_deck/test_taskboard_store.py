"""The Taskboard store (taskboard.db): one task per record however often it is
read, every read and write on one board only, a user's correction locks its
field against every later machine pass (until Undo), accent-folded search,
facets counted under the other filters, and the lanes."""

from __future__ import annotations

import sqlite3
import time

import pytest

from captain_claw.flight_deck.taskboard_store import TaskboardStore

ALICE = "user-alice"
BOB = "user-bob"


@pytest.fixture()
def store(tmp_path):
    s = TaskboardStore(tmp_path / "taskboard.db")
    yield s
    s.close()


def _ev(ref: str, **kw) -> dict:
    return {"source": "test", "source_ref": ref, "at": kw.pop("at", time.time()), **kw}


def _task(store, uid=ALICE, key="k:1", ref=None, patch=None, **fields) -> str:
    fields = {"title": "A task", "kind": "request", "channel": "web", "owes_next": "agent", **fields}
    tid, _ = store.ingest(uid, keys=[key], create=fields, patch=patch, event=_ev(ref or f"{uid}:{key}"))
    return tid


# ── ingest ────────────────────────────────────────────────────────────


def test_a_record_read_twice_is_one_task_and_one_event(store):
    first = store.ingest(ALICE, keys=["k:1"], create={"title": "x"}, event=_ev("r1"),
                         patch={"status": "done", "closed_by": "source"})
    again = store.ingest(ALICE, keys=["k:1"], create={"title": "x"}, event=_ev("r1"),
                         patch={"status": "open"})
    assert first[1] is True and again == (first[0], False)
    task = store.get_task(ALICE, first[0])
    assert task["status"] == "done" and task["closed_by"] == "source" and task["closed_at"]
    assert len(store.events(ALICE, first[0])) == 1


def test_a_new_event_moves_the_task_and_reopening_clears_who_closed_it(store):
    tid = _task(store, patch={"status": "done", "closed_by": "source", "result": "success"})
    store.ingest(ALICE, keys=["k:1"], event=_ev("r2"), patch={"status": "open", "owes_next": "agent"})
    task = store.get_task(ALICE, tid)
    assert (task["status"], task["closed_by"], task["closed_at"], task["result"]) == ("open", "", None, "")
    assert task["fingerprint"].endswith(":2")


def test_without_create_nothing_is_made(store):
    assert store.ingest(ALICE, keys=["nope"], event=_ev("r1")) == (None, False)


def test_a_recorded_event_makes_the_evidence_verified_and_claimed_never_overrides_it(store):
    tid = _task(store)
    store.ingest(ALICE, keys=["k:1"], event=_ev("sent", evidence="recorded"))
    assert store.get_task(ALICE, tid)["evidence"] == "verified"
    store.ingest(ALICE, keys=["k:1"], event=_ev("done"), patch={"status": "done", "evidence": "claimed"})
    assert store.get_task(ALICE, tid)["evidence"] == "verified"


def test_extra_keys_go_to_the_first_task_that_takes_them(store):
    a = _task(store, key="a")
    store.ingest(ALICE, keys=["a"], event=_ev("e1"), extra_keys=["gmail:thread:t1"])
    b = store.ingest(ALICE, keys=["b"], create={"title": "b"}, event=_ev("e2"), extra_keys=["gmail:thread:t1"])[0]
    assert a != b and store.task_for_key(ALICE, "gmail:thread:t1") == a


def test_next_state_ref_counts_only_a_change_and_a_return(store):
    _task(store)
    ref = store.next_state_ref("test", "row1", "held")
    assert ref == "row1:held:0"
    store.ingest(ALICE, keys=["k:1"], event=_ev(ref, refs={"state": "held"}))
    assert store.next_state_ref("test", "row1", "held") is None
    ref = store.next_state_ref("test", "row1", "approved")
    store.ingest(ALICE, keys=["k:1"], event=_ev(ref, refs={"state": "approved"}))
    assert store.next_state_ref("test", "row1", "held") == "row1:held:2"
    # A row whose id another one starts with is a row of its own.
    assert store.next_state_ref("test", "row", "held") == "row:held:0"


# ── per user ──────────────────────────────────────────────────────────


def test_another_user_sees_and_touches_nothing(store):
    tid = _task(store)
    store.ingest(ALICE, keys=["k:1"], event=_ev("p"),
                 parties=[{"role": "requester", "kind": "email", "address": "ana@x.co", "display": "Ana"}])
    assert store.get_task(BOB, tid) is None
    assert store.events(BOB, tid) == [] and store.parties_for(BOB, [tid]) == {tid: []}
    assert store.links(BOB, [tid]) == {tid: []} and store.task_for_key(BOB, "k:1") is None
    assert store.correct(BOB, tid, "done") is None
    assert store.get_task(ALICE, tid)["status"] == "open"
    res = store.query(BOB, view="all")
    assert res["total"] == 0 and res["rows"] == [] and all(not v for v in res["facets"].values())
    assert store.counts(BOB) == {"needs_me": 0, "waiting": 0, "working": 0, "done": 0, "all": 0}
    # The same key on Bob's board is Bob's own task.
    assert _task(store, uid=BOB) != tid


# ── corrections and locks ─────────────────────────────────────────────


def test_a_correction_locks_its_field_against_the_sources(store):
    tid = _task(store)
    store.correct(ALICE, tid, "rename", "My name for it", actor="Alice")
    store.correct(ALICE, tid, "done", actor="Alice")
    store.ingest(ALICE, keys=["k:1"], event=_ev("run2"), touch={"title": "Source title"},
                 patch={"status": "failed", "result": "failed", "closed_by": "source", "owes_next": "me"})
    task = store.get_task(ALICE, tid)
    assert task["title"] == "My name for it"
    assert (task["status"], task["closed_by"], task["closed_by_party"], task["result"]) == ("done", "user", "Alice", "")
    assert task["owes_next"] == "me"  # not locked: the source still moves it
    kinds = [(e["kind"], e["actor_kind"]) for e in store.events(ALICE, tid)]
    assert kinds.count(("correction", "me")) == 2


def test_undo_lifts_the_last_correction_and_its_lock(store):
    tid = _task(store)
    store.correct(ALICE, tid, "done")
    store.correct(ALICE, tid, "undo")
    task = store.get_task(ALICE, tid)
    assert task["status"] == "open" and task["closed_by"] == "" and "status" not in task["user_locks"]
    store.ingest(ALICE, keys=["k:1"], event=_ev("later"), patch={"status": "done", "closed_by": "source"})
    assert store.get_task(ALICE, tid)["status"] == "done"
    with pytest.raises(ValueError):
        store.correct(ALICE, tid, "undo")


def test_undo_goes_back_one_correction_at_a_time(store):
    tid = _task(store)
    store.correct(ALICE, tid, "rename", "First")
    store.correct(ALICE, tid, "rename", "Second")
    store.correct(ALICE, tid, "undo")
    assert store.get_task(ALICE, tid)["title"] == "First"
    store.correct(ALICE, tid, "undo")
    task = store.get_task(ALICE, tid)
    assert task["title"] == "A task" and "title" not in task["user_locks"]


def test_undo_gives_the_source_its_say_back(store):
    """What a source said while the user's lock kept it out stands after Undo
    — not the state from before the correction."""
    tid = _task(store, owes_next="me")
    store.correct(ALICE, tid, "done")
    store.ingest(ALICE, keys=["k:1"], event=_ev("rejected", at=time.time() + 1),
                 patch={"status": "dismissed", "owes_next": "nobody", "closed_by": "source"})
    task = store.get_task(ALICE, tid)
    assert (task["status"], task["owes_next"]) == ("done", "nobody")
    held = [e["refs"].get("held") for e in store.events(ALICE, tid) if e["source_ref"] == "rejected"]
    assert held == [{"status": "dismissed", "closed_by": "source"}]
    store.correct(ALICE, tid, "undo")
    task = store.get_task(ALICE, tid)
    assert (task["status"], task["closed_by"], task["user_locks"]) == ("dismissed", "source", "{}")
    assert task["closed_at"] is not None


def test_undo_with_nothing_from_the_sources_restores_what_was(store):
    tid = _task(store, owes_next="me")
    store.correct(ALICE, tid, "owes_next", "them")
    store.ingest(ALICE, keys=["k:1"], event=_ev("note"))  # no move
    store.correct(ALICE, tid, "undo")
    assert store.get_task(ALICE, tid)["owes_next"] == "me"


def test_a_new_event_can_lift_the_users_lock_and_its_undo_is_settled(store):
    tid = _task(store, patch={"status": "done", "closed_by": "source"})
    store.correct(ALICE, tid, "reopen")
    store.ingest(ALICE, keys=["k:1"], event=_ev("run2"), unlock=("status",),
                 patch={"status": "failed", "result": "failed", "closed_by": "source"})
    task = store.get_task(ALICE, tid)
    assert (task["status"], task["user_locks"]) == ("failed", "{}")
    # Nothing of the Reopen holds a field now: there is nothing to undo.
    with pytest.raises(ValueError):
        store.correct(ALICE, tid, "undo")
    # Read again, the event lifts nothing more.
    store.correct(ALICE, tid, "done")
    store.ingest(ALICE, keys=["k:1"], event=_ev("run2"), unlock=("status",), patch={"status": "open"})
    assert store.get_task(ALICE, tid)["status"] == "done"


def test_undo_of_a_seen_failure_puts_back_only_that_flag(store):
    tid = _task(store, patch={"status": "failed"})
    store.correct(ALICE, tid, "ack_failed")
    store.ingest(ALICE, keys=["k:1"], event=_ev("stale"), patch={"flags": {"stale": True}})
    store.correct(ALICE, tid, "undo")
    flags = store.get_task(ALICE, tid)["flags"]
    assert '"stale": true' in flags and "failed_ack" not in flags


# Jev's close, as triage writes it (taskboard_triage._close: nobody owes it then).
_JEV_CLOSE = {"status": "done", "closed_by": "jev", "closed_by_party": "Jev", "result": "success",
              "owes_next": "nobody", "flags": {"probably_done": False}}


def _jev_close(store, tid):
    assert store.triage_write(ALICE, tid, store.get_task(ALICE, tid)["fingerprint"], fields=_JEV_CLOSE)


def test_undo_after_jevs_close_opens_it_again(store):
    """Decision #2: Jev's close comes with Undo — even with no correction of
    the user's to take back."""
    tid = _task(store)
    _jev_close(store, tid)
    task = store.correct(ALICE, tid, "undo")
    assert (task["status"], task["closed_by"], task["closed_at"], task["result"]) == ("open", "", None, "")
    assert task["owes_next"] == "me" and "status" in task["user_locks"]
    assert store.lane_of(ALICE, tid) == "needs_me"
    assert [e["summary"] for e in store.events(ALICE, tid) if e["kind"] == "correction"] == ["Undid Jev's close"]
    # Jev can't close it again; Undo once more puts Jev's close back.
    _jev_close(store, tid)
    assert store.get_task(ALICE, tid)["status"] == "open"
    task = store.correct(ALICE, tid, "undo")
    assert (task["status"], task["closed_by"], task["user_locks"]) == ("done", "jev", "{}")


def test_undo_after_jevs_close_leaves_an_earlier_correction_alone(store):
    tid = _task(store)
    store.correct(ALICE, tid, "rename", "Book the flight")
    _jev_close(store, tid)
    task = store.correct(ALICE, tid, "undo")
    assert (task["status"], task["title"]) == ("open", "Book the flight")
    # A correction after Jev's close is the latest change: Undo takes that back first.
    other = _task(store, key="k:2")
    _jev_close(store, other)
    store.correct(ALICE, other, "rename", "Lunch with Ana")
    task = store.correct(ALICE, other, "undo")
    assert (task["status"], task["closed_by"], task["title"]) == ("done", "jev", "A task")


@pytest.mark.parametrize("action", ["not_done", "reopen"])
def test_a_closed_task_opened_again_is_the_users_move(store, action):
    """A close that left nobody owing a move (Jev's, a source's): opened
    again by the user, it is in Needs me — never in no lane at all."""
    tid = _task(store)
    _jev_close(store, tid)
    task = store.correct(ALICE, tid, action)
    assert (task["status"], task["owes_next"], task["closed_by"]) == ("open", "me", "")
    assert store.lane_of(ALICE, tid) == "needs_me"
    # Who owes it before the close stays when it wasn't nobody.
    kept = _task(store, key="k:2", owes_next="them", patch={"status": "done", "closed_by": "source"})
    assert store.correct(ALICE, kept, action)["owes_next"] == "them"


def test_the_timeline_is_the_latest_events_and_their_count(store):
    tid, _ = store.ingest(ALICE, keys=["k:1"], create={"title": "x"}, event=_ev("e-first", at=1000.0))
    for i in range(30):
        store.ingest(ALICE, keys=["k:1"], event=_ev(f"e{i}", at=2000.0 + i))
    events = store.events(ALICE, tid, limit=10)
    assert [e["source_ref"] for e in events] == [f"e{i}" for i in range(20, 30)]
    assert store.event_count(ALICE, tid) == 31 and store.event_count(BOB, tid) == 0


@pytest.mark.parametrize("action,value", [
    ("snooze", None), ("snooze", time.time() - 10), ("snooze", float("inf")), ("snooze", float("nan")),
    ("snooze", 1e300), ("snooze", 10 ** 400), ("snooze", time.time() + 400 * 86400),
    ("rename", "  "), ("owes_next", "the cat"), ("ack_failed", None), ("fly", None),
])
def test_a_correction_that_cannot_apply_is_refused(store, action, value):
    tid = _task(store)
    with pytest.raises(ValueError):
        store.correct(ALICE, tid, action, value)


# ── search, facets, lanes ─────────────────────────────────────────────


def test_search_folds_accents_on_both_sides_and_wants_every_word(store):
    _task(store, key="a", title="Ručak s Đurićem", ask="ćevapi u petak")
    _task(store, key="b", title="Invoice")
    store.ingest(ALICE, keys=["b"], event=_ev("pb"),
                 parties=[{"role": "requester", "kind": "email", "address": "zoe@x.co", "display": "Zoë Šarić"}])
    titles = lambda q: [r["title"] for r in store.query(ALICE, view="all", q=q)["rows"]]  # noqa: E731
    assert titles("duric") == ["Ručak s Đurićem"]
    assert titles("CEVAPI petak") == ["Ručak s Đurićem"]
    assert titles("ĐURIĆ") == ["Ručak s Đurićem"]
    assert titles("cevapi invoice") == []
    assert titles("saric") == ["Invoice"] and titles("zoe") == ["Invoice"]
    assert titles("100%") == []  # LIKE wildcards are matched as text


def test_facets_are_counted_under_the_other_filters(store):
    _task(store, key="a", channel="gmail", owes_next="me")
    _task(store, key="b", channel="gmail", owes_next="them")
    _task(store, key="c", channel="telegram", owes_next="me")
    _task(store, key="d", channel="web", patch={"status": "done"})
    res = store.query(ALICE, view="all", filters={"channel": ["gmail"]})
    assert res["total"] == 2
    channels = {v: n for v, _, n in res["facets"]["channel"]}
    assert channels == {"gmail": 2, "telegram": 1, "web": 1}  # the channel filter itself left out
    statuses = {v: n for v, _, n in res["facets"]["status"]}
    assert statuses == {"open": 2}
    res = store.query(ALICE, view="all", filters={"channel": ["gmail", "telegram"], "status": ["open"]})
    assert res["total"] == 3
    assert {v: n for v, _, n in res["facets"]["status"]} == {"open": 3}
    assert {v: n for v, _, n in res["facets"]["channel"]} == {"gmail": 2, "telegram": 1}


def test_party_facet_and_filter(store):
    for key, addr in (("a", "ana@x.co"), ("b", "ana@x.co"), ("c", "bo@y.org")):
        _task(store, key=key)
        store.ingest(ALICE, keys=[key], event=_ev(f"p{key}"),
                     parties=[{"role": "requester", "kind": "email", "address": addr, "display": addr[:2]}])
    res = store.query(ALICE, view="all", filters={"party": ["email:ana@x.co"]})
    assert res["total"] == 2
    assert {v: n for v, _, n in res["facets"]["party"]} == {"email:ana@x.co": 2, "email:bo@y.org": 1}


def test_views_lanes_and_counts(store):
    now = time.time()
    me = _task(store, key="me", owes_next="me")
    them = _task(store, key="them", owes_next="them")
    agent = _task(store, key="agent", owes_next="agent")
    done = _task(store, key="done", patch={"status": "done"})
    failed = _task(store, key="failed", patch={"status": "failed"})
    overdue = _task(store, key="overdue", owes_next="them", due_at=now - 60)
    proposal = _task(store, key="proposal", owes_next="agent")
    snoozed = _task(store, key="snoozed", owes_next="me")
    store.correct(ALICE, snoozed, "snooze", now + 3600)
    not_task = _task(store, key="nt")
    store.correct(ALICE, not_task, "not_task")

    def view(name):
        return {r["id"]: r["lane"] for r in store.query(ALICE, view=name, proposal_tasks=[proposal])["rows"]}

    assert set(view("needs_me")) == {me, failed, overdue, proposal}
    assert set(view("waiting")) == {them}
    assert set(view("working")) == {agent}
    assert set(view("done")) == {done}
    lanes = view("all")
    assert not_task not in lanes and lanes[snoozed] == "snoozed" and lanes[them] == "waiting"
    assert store.counts(ALICE, proposal_tasks=[proposal]) == \
        {"needs_me": 4, "waiting": 1, "working": 1, "done": 1, "all": 8}
    # Asked for by the Status facet, the not-tasks are listed.
    rows = store.query(ALICE, view="all", filters={"status": ["not_task"]})["rows"]
    assert [r["id"] for r in rows] == [not_task]
    # A failure seen leaves Needs me; a new failure brings it back.
    store.correct(ALICE, failed, "ack_failed")
    assert failed not in view("needs_me")
    time.sleep(0.01)
    store.ingest(ALICE, keys=["failed"], event=_ev("again"), patch={"status": "failed"})
    assert failed in view("needs_me")


def test_paging_sorting_and_dates(store):
    base = time.time() - 1000
    for i in range(5):
        store.ingest(ALICE, keys=[f"k{i}"], create={"title": f"t{i}", "urgency": i % 3},
                     event=_ev(f"e{i}", at=base + i * 100))
    recent = store.query(ALICE, view="all", limit=2)
    assert recent["total"] == 5 and [r["title"] for r in recent["rows"]] == ["t4", "t3"]
    assert [r["title"] for r in store.query(ALICE, view="all", limit=2, offset=4)["rows"]] == ["t0"]
    assert [r["title"] for r in store.query(ALICE, view="all", sort="oldest", limit=1)["rows"]] == ["t0"]
    assert store.query(ALICE, view="all", sort="urgency", limit=1)["rows"][0]["urgency"] == 2
    window = store.query(ALICE, view="all", date_from=base + 150, date_to=base + 350)
    assert sorted(r["title"] for r in window["rows"]) == ["t2", "t3"]


# ── cleanup, settings, schema ─────────────────────────────────────────


def test_delete_agent_drops_its_owners_tasks_on_every_board(store):
    a = _task(store, key="a", agent_slug="helper", agent_owner_id=ALICE)
    b = _task(store, uid=BOB, key="b", agent_slug="helper", agent_owner_id=ALICE)   # a member's turn
    c = _task(store, key="c", agent_slug="other", agent_owner_id=ALICE)
    d = _task(store, key="d", agent_slug="helper", agent_owner_id="someone-else")
    assert store.delete_agent(ALICE, "helper") == 2
    assert store.get_task(ALICE, a) is None and store.get_task(BOB, b) is None
    assert store.task_for_key(ALICE, "a") is None and store.events(ALICE, a) == []
    assert store.get_task(ALICE, c) and store.get_task(ALICE, d)
    assert store.delete_agent(None, "helper") == 1


def test_delete_agent_drops_ownerless_tasks_on_the_removers_board_only(store):
    """A task recorded without an agent owner (the agent wasn't known) may be
    someone else's agent of the same name."""
    mine = _task(store, key="a", agent_slug="researcher", agent_owner_id="")
    theirs = _task(store, uid=BOB, key="b", agent_slug="researcher", agent_owner_id="")
    assert store.delete_agent(ALICE, "researcher") == 1
    assert store.get_task(ALICE, mine) is None and store.get_task(BOB, theirs)


def test_keyed_tasks(store):
    a = _task(store, key="sched:job:1")
    b = _task(store, uid=BOB, key="sched:job:2", patch={"status": "done"})
    _task(store, key="fu:1")
    got = {(r["user_id"], r["key"], r["task_id"]) for r in store.keyed_tasks("sched:job:")}
    assert got == {(ALICE, "sched:job:1", a), (BOB, "sched:job:2", b)}
    assert [r["task_id"] for r in store.keyed_tasks("sched:job:", status="open")] == [a]
    assert [r["task_id"] for r in store.keyed_tasks("sched:job:", user_id=BOB)] == [b]


def test_delete_user_drops_their_whole_board(store):
    a = _task(store)
    b = _task(store, uid=BOB)
    store.save_settings(ALICE, jev_per_day=5)
    assert store.delete_user(ALICE) == 1
    assert store.get_task(ALICE, a) is None and store.get_task(BOB, b)
    assert store.get_settings(ALICE)["jev_per_day"] == 200


def test_settings_default_and_saved_per_user(store):
    assert store.get_settings(ALICE)["auto_close"] is True
    saved = store.save_settings(ALICE, auto_close=0, jev_per_day=10, passive={"gmail": {"on": True}})
    assert (saved["auto_close"], saved["jev_per_day"], saved["passive"]) == (False, 10, {"gmail": {"on": True}})
    assert store.get_settings(BOB)["jev_per_day"] == 200


def test_cursors(store):
    assert store.get_cursor("x") == ""
    store.set_cursor("x", "5")
    store.set_cursor("x", "6")
    assert store.get_cursor("x") == "6"


def test_an_older_board_gets_the_columns_it_lacks(tmp_path):
    path = tmp_path / "taskboard.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL DEFAULT '')")
    conn.execute("INSERT INTO tasks (id, user_id, title) VALUES ('tb_old', ?, 'Old')", (ALICE,))
    conn.commit()
    conn.close()
    s = TaskboardStore(path)
    try:
        cols = {r[1] for r in s._conn.execute("PRAGMA table_info(tasks)")}
        assert {"search_text", "user_locks", "next_steps", "retention_until", "merged_into"} <= cols
        task = s.get_task(ALICE, "tb_old")
        assert task["title"] == "Old" and task["status"] == "open"
        tables = {r[0] for r in s._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"task_links", "task_events", "task_decisions", "parties", "party_handles", "task_parties",
                "task_rules", "turns", "source_cursors", "taskboard_settings"} <= tables
    finally:
        s.close()
