"""First Mate signals, scorecard and health (P4, piece 1).

Pinned here:

* a draft's text is compared with what was sent, the quoted reply left out,
  only its start when the register kept only that;
* a draft no longer a draft was sent as written, sent after edits (worth
  how much stayed), or deleted; one still a draft after two weeks expires and
  counts for nothing; one that can't be checked now waits for the next look;
* a run's drafts are picked up from the register; the watcher waits 15
  minutes and looks at most once an hour;
* thumbs: one per run, the latest wins, only on a finished run, owner only;
* the scorecard's numbers and health: new, green, amber, red; a second
  doubtful carry-out in two weeks pauses the factory;
* the supervisor checks drafts on its scan.
"""

from __future__ import annotations

import base64
import json
from datetime import timedelta

import pytest

from captain_claw.flight_deck import bat_loop, events, taskboard_proof
from captain_claw.flight_deck import first_mate_drafts as drafts
from captain_claw.flight_deck import first_mate_loop as loop
from captain_claw.flight_deck import first_mate_proof as proof
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_score as score
from captain_claw.flight_deck import first_mate_store as fm

BOB, CAROL = "user-bob", "user-carol"


class FakeBat:
    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}

    async def create_run(self, *, run_id, owner_id, title, task, config=None, origin=None,
                         source_host="localhost", source_port=0, llm_usd_cap=0.0, status="planning"):
        self.runs[run_id] = {"id": run_id, "owner_id": owner_id, "title": title, "task": task,
                             "config": config or {}, "status": status, "cumulative_tokens": 0,
                             "cumulative_usd": 0.0, "truth": "", "stopped_reason": ""}
        return run_id

    async def append_event(self, run_id, stage, message="", **data):
        return 1

    async def get_run(self, run_id):
        r = self.runs.get(run_id)
        return dict(r) if r else None

    async def list_runs(self, owner_id, limit=50):
        return [dict(r) for r in self.runs.values() if r["owner_id"] == owner_id][:limit]

    async def set_status(self, run_id, status, **kw):
        self.runs[run_id].update(status=status, **kw)


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "_STORE", events.EventsStore(tmp_path / "events.db"))
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True})
    return s


@pytest.fixture
def bat(monkeypatch):
    fake = FakeBat()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", fake)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    return fake


def _live(store, **kw):
    kw.setdefault("title", "Inbox drafts")
    kw.setdefault("charter", "Draft replies to mail that needs one.")
    kw.setdefault("grants", {"email": True, "sends_per_run": 3})
    f = store.create(BOB, **kw)
    return store.approve(f["id"], 1, f["draft"]["sha"], BOB)


def _b64(text):
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def _msg(text, labels=("SENT",), at=None):
    return {"labelIds": list(labels), "internalDate": str(int(((at or fm._utcnow()).timestamp()) * 1000)),
            "payload": {"mimeType": "multipart/alternative",
                        "parts": [{"mimeType": "text/html", "body": {"data": _b64("<p>x</p>")}},
                                  {"mimeType": "text/plain", "body": {"data": _b64(text)}}]}}


DRAFT = "Hi Ana,\n\nI checked the invoice: the total is right.\n\nBest, Bob"


# ── text and verdicts ──────────────────────────────────────────────────

def test_text_compared_without_the_quoted_reply():
    sent = DRAFT + "\n\nOn Mon, 16 Nov 2026 at 09:00, Ana <ana@x.co> wrote:\n> Is the total right?\n> Thanks"
    assert drafts.similarity(DRAFT, sent) == 1.0
    assert drafts.similarity(DRAFT, "Hi Ana, wrong total, will send a new invoice.") < 0.9
    long = "x " * 200
    assert drafts.similarity("x x x", long, partial=True) == 1.0  # only the start the register kept
    assert drafts.message_text({"snippet": "just a snippet", "payload": {"mimeType": "text/html"}}) == "just a snippet"
    assert drafts.message_text(_msg("plain body")) == "plain body"


def test_what_became_of_a_draft():
    now = fm._utcnow()
    d = {"created_at": (now - timedelta(hours=2)).isoformat(), "body": DRAFT, "partial": 0}
    assert drafts.verdict(d, {"messages": [_msg(DRAFT)]}) == ("sent", 1.0)
    outcome, sim = drafts.verdict(d, {"messages": [_msg("Hi Ana, the total is right after all, but I added a note. Bob")]})
    assert outcome == "edited" and 0 < sim < drafts.AS_WRITTEN
    assert drafts.verdict(d, {"messages": [_msg("Old reply", at=now - timedelta(days=3))]}) == ("deleted", None)
    assert drafts.verdict(d, {"messages": [_msg(DRAFT, labels=("DRAFT",))]}) == ("deleted", None)


def test_drafts_read_from_register_rows():
    rows = [
        {"tool": "google_mail", "outcome": "ok", "params": json.dumps({"action": "create_draft", "to": "ana@x.co",
                                                                         "subject": "Re: invoice", "body": DRAFT}),
         "refs": json.dumps({"gmail_draft_id": "r-1", "gmail_message_id": "m1", "gmail_thread_id": "t1"})},
        {"tool": "google_mail", "outcome": "ok", "params": json.dumps({"action": "create_draft",
                                                                         "body": {"_head": "Hi Ben", "_len": 5000}}),
         "refs": json.dumps({"gmail_draft_id": "r-2"})},
        {"tool": "google_mail", "outcome": "error", "params": json.dumps({"action": "create_draft"}),
         "refs": json.dumps({"gmail_draft_id": "r-3"})},
        {"tool": "google_mail", "outcome": "ok", "params": json.dumps({"action": "send"}), "refs": "{}"},
    ]
    got = drafts.drafts_in(rows)
    assert [(d["draft_id"], d["thread_id"], d["partial"]) for d in got] == [("r-1", "t1", False), ("r-2", "", True)]
    assert got[0]["subject"] == "Re: invoice" and got[1]["body"] == "Hi Ben"


# ── the watcher ────────────────────────────────────────────────────────

@pytest.fixture
def gmail(monkeypatch):
    """A Gmail that answers per draft id and thread id."""
    state = {"token": "tok", "drafts": {}, "threads": {}}

    async def token(uid):
        return state["token"]

    async def fetch(token_, url, params=None):
        key = url.rsplit("/", 1)[-1]
        if "/drafts/" in url:
            return state["drafts"].get(key, (404, {}))
        return state["threads"].get(key, (404, {}))

    monkeypatch.setattr(taskboard_proof, "_token", token)
    monkeypatch.setattr(taskboard_proof, "_fetch", fetch)
    return state


async def test_the_watcher_turns_drafts_into_signals(store, gmail):
    f = _live(store)
    for n, did in enumerate(("as-written", "edited", "deleted", "still")):
        store.add_draft(f["id"], BOB, run_id="fmr_x", draft_id=did, thread_id=f"t{n}", subject=did, body=DRAFT)
    gmail["threads"] = {"t0": (200, {"messages": [_msg(DRAFT)]}),
                        "t1": (200, {"messages": [_msg("Hi Ana, actually the total is wrong; new invoice attached.")]}),
                        "t2": (200, {"messages": []})}
    gmail["drafts"] = {"still": (200, {"id": "still"})}
    now = fm._utcnow()
    assert await drafts.check_due(store, now) == 0  # not 15 minutes old yet
    later = now + timedelta(minutes=20)
    assert await drafts.check_due(store, later) == 3
    by_id = {d["draft_id"]: d for d in store.drafts(f["id"])}
    assert by_id["as-written"]["status"] == "sent" and by_id["edited"]["status"] == "edited"
    assert by_id["deleted"]["status"] == "deleted" and by_id["still"]["status"] == "waiting"
    kinds = {s["kind"]: s["value"] for s in store.signals(f["id"])}
    assert kinds["sent_as_written"] == 1.0 and kinds["draft_deleted"] == 0.0 and 0 < kinds["sent_edited"] < 0.9
    assert await drafts.check_due(store, later + timedelta(minutes=10)) == 0  # at most once an hour
    gmail["drafts"] = {"still": (200, {"id": "still"})}
    await drafts.check_due(store, now + timedelta(days=15))
    assert {d["draft_id"]: d["status"] for d in store.drafts(f["id"])}["still"] == "expired"
    assert len(store.signals(f["id"])) == 3  # silence is never a signal


async def test_a_draft_that_cant_be_checked_waits(store, gmail):
    f = _live(store)
    store.add_draft(f["id"], BOB, run_id="fmr_x", draft_id="d1", thread_id="t1", body=DRAFT)
    gmail["token"] = None
    later = fm._utcnow() + timedelta(minutes=20)
    assert await drafts.check_due(store, later) == 0
    [d] = store.drafts(f["id"])
    assert d["status"] == "waiting" and d["checked_at"]
    gmail["token"] = "tok"
    gmail["drafts"] = {"d1": (503, {})}
    assert await drafts.check_due(store, later + timedelta(hours=2)) == 0
    assert store.drafts(f["id"])[0]["status"] == "waiting"


async def test_a_carry_outs_drafts_are_picked_up(store, bat, monkeypatch):
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    store.add_intent(run["id"], kind="intent", tool="google_mail", what="Draft a reply to Ana",
                     args={"action": "create_draft", "to": "ana@x.co"})
    bat.runs[run["child_id"]].update(status="done", truth="ok")
    await runs.refresh(run["id"])
    carry = await runs.carry_out(run["id"], BOB)
    rows = [{"tool": "google_mail", "outcome": "ok", "risk": "write", "reason": "", "agent_slug": "bat-x",
             "params": json.dumps({"action": "create_draft", "to": "ana@x.co", "subject": "Re", "body": DRAFT}),
             "refs": json.dumps({"gmail_draft_id": "r-9", "gmail_thread_id": "t9"})}]

    async def fake_rows(r):
        return rows

    async def no_google(uid, actions):
        return None

    monkeypatch.setattr(proof, "_rows", fake_rows)
    monkeypatch.setattr(taskboard_proof, "confirm", no_google)
    bat.runs[carry["child_id"]].update(status="done", truth="Left a draft for Ana.")
    await runs.refresh(carry["id"])
    [d] = store.drafts(f["id"])
    assert (d["draft_id"], d["thread_id"], d["run_id"], d["status"]) == ("r-9", "t9", carry["id"], "waiting")


# ── thumbs ─────────────────────────────────────────────────────────────

def test_thumbs_one_per_run(store):
    f = _live(store)
    r = store.start_run(f["id"])
    store.finish_run(r["id"], "done")
    store.set_thumbs(f["id"], r["id"], "thumbs_up")
    store.set_thumbs(f["id"], r["id"], "thumbs_down")
    assert store.thumbs(r["id"]) == "thumbs_down" and [s["kind"] for s in store.signals(f["id"])] == ["thumbs_down"]
    store.set_thumbs(f["id"], r["id"], "")
    assert store.thumbs(r["id"]) == "" and store.signals(f["id"]) == []
    with pytest.raises(fm.FirstMateError):
        store.set_thumbs(f["id"], r["id"], "meh")


async def test_the_thumbs_route(store):
    from captain_claw.flight_deck import first_mate_routes as routes
    f = _live(store)
    going = store.start_run(f["id"])
    with pytest.raises(Exception) as early:
        await routes.thumbs(going["id"], {"kind": "up"}, user={"id": BOB})
    assert getattr(early.value, "status_code", 0) == 400
    store.finish_run(going["id"], "done")
    got = await routes.thumbs(going["id"], {"kind": "up"}, user={"id": BOB})
    assert got == {"thumbs": "thumbs_up", "satisfaction": 1.0}
    with pytest.raises(Exception) as other:
        await routes.thumbs(going["id"], {"kind": "down"}, user={"id": CAROL})
    assert getattr(other.value, "status_code", 0) == 404


# ── scorecard and health ───────────────────────────────────────────────

def test_the_scorecard_and_health(store):
    f = _live(store)
    now = fm._utcnow()
    assert score.scorecard(store, store.get(f["id"]), now)["health"] == "new"
    for verdict in ("pass", "pass", "pass"):
        r = store.start_run(f["id"])
        store.mark_running(r["id"], "bat_x")
        store.finish_run(r["id"], "done", verdict=verdict, tokens=30_000)
    card = score.scorecard(store, store.get(f["id"]), now)
    assert card["health"] == "green" and card["week"]["runs"] == 3 and card["week"]["passed"] == 3
    assert card["week"]["tokens_per_run"] == 30_000
    for _ in range(5):
        store.add_signal(f["id"], "thumbs_down")
    card = score.scorecard(store, store.get(f["id"]), now)
    assert card["health"] == "amber" and "under its target" in card["health_why"] and card["satisfaction"] == 0.0
    store.set_state(f["id"], "paused", "deck", loop.STREAK_REASON)
    assert score.scorecard(store, store.get(f["id"]), now)["health"] == "red"


async def test_a_second_doubt_in_two_weeks_pauses(store, bat, monkeypatch):
    f = _live(store)  # asks first: each carry-out is approved by hand below

    async def fake_rows(r):
        return [{"tool": "google_calendar", "outcome": "ok", "risk": "write", "reason": "",
                 "params": json.dumps({"action": "create_event", "summary": "x"}), "refs": "{}", "agent_slug": "bat-x"}]

    async def no_google(uid, actions):
        return None

    monkeypatch.setattr(proof, "_rows", fake_rows)
    monkeypatch.setattr(taskboard_proof, "confirm", no_google)
    for n in range(2):
        run = await runs.start(f["id"], BOB, watch=False)
        store.add_intent(run["id"], kind="intent", tool="google_mail", what=f"Reply {n}", args={"action": "send"})
        bat.runs[run["child_id"]].update(status="done", truth="ok")
        await runs.refresh(run["id"])
        carry = await runs.carry_out(run["id"], BOB)
        bat.runs[carry["child_id"]].update(status="done", truth="Done.")
        await runs.refresh(carry["id"])
    g = store.get(f["id"])
    assert g["state"] == "paused" and g["state_reason"] == proof.TWO_DOUBTS_REASON
    assert score.scorecard(store, g, fm._utcnow())["health"] == "red"


async def test_the_supervisor_checks_drafts(store, monkeypatch):
    seen = []

    async def check(st, now):
        seen.append(now)
        return 0

    monkeypatch.setattr(drafts, "check_due", check)
    await loop.Supervisor().tick(fm._utcnow())
    assert len(seen) == 1
