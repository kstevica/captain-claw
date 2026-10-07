"""Autonomous Work proposes email, it never writes it (PR E, part 1).

The glasses deck's Arbiter auto-fired ``mail.draft`` 40 times: it passed
``should_auto_dispatch`` through the grant AND through self-reinforcing earned
trust, the arbiter prompt told it to prefer drafts, and old / repeated email
threads kept resurfacing. Now:

* every email action is ``human_only`` (``mail.draft`` is also ``proposable``):
  never auto-fired by a grant or by earned trust, never learned as trust, and
  the dispatch rail itself holds an unapproved one (belt);
* an email that may need a reply becomes a "<sender> is waiting for a reply"
  nudge (or a track), or at most a mail.draft PROPOSAL the user approves;
* approve is compare-and-set; approved drafts are threaded and addressed to the
  real Reply-To / sender;
* Gmail events older than 48h are ignored and one thread surfaces once;
* every autonomy chat turn carries the ``autonomy``/``deny`` marker and the
  no-mail line, auto-fired or approved;
* the learned trust of ``mail.*`` actions is swept on every start.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from captain_claw.config import AutonomousWorkConfig
from captain_claw.flight_deck import (
    action_catalog,
    actions,
    arbiter,
    autonomy,
    autonomy_routes,
    basna_routes,
    events,
    fd_dispatch,
    plans,
)

UID = "user-alice"
_REFLECTION = {"intentions": []}
_AUTHOR = {"host": "localhost", "port": 1, "auth": "", "name": "a"}
_AGENT = {"slug": "alice-main", "host": "localhost", "port": 24001, "auth": "tok", "name": "main"}
_DENY = {"kind": "autonomy", "job_text": "", "mail_write": "deny"}


# ── isolation (required: the singletons default to ~/.captain-claw/*.db) ──

@pytest.fixture(autouse=True)
def _isolated_fd_data(tmp_path, monkeypatch):
    monkeypatch.setenv("FD_DATA_DIR", str(tmp_path / "fd-data"))
    from captain_claw.flight_deck import autonomy, consciousness, events, fd_scheduler, plans
    for mod in (autonomy, events, plans, consciousness, fd_scheduler):
        monkeypatch.setattr(mod, "_STORE", None, raising=False)


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    monkeypatch.setattr(autonomy, "_STORE", s)
    monkeypatch.setattr(autonomy, "global_defaults", lambda: AutonomousWorkConfig().model_dump())
    return s


@pytest.fixture
def evstore(tmp_path, monkeypatch):
    s = events.EventsStore(tmp_path / "events.db")
    monkeypatch.setattr(events, "_STORE", s)
    return s


def _iso(hours_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()


def _gmail_event(evstore, mid: str, tid: str, *, received_hours_ago: float | None = 1.0,
                 frm: str = '"Ana Kovač" <ana@x.co>', subject: str = "Q3 report",
                 **md) -> dict:
    meta = {"message_id": mid, "thread_id": tid, "from": frm, "subject": subject, **md}
    if received_hours_ago is not None:
        meta["received_at"] = _iso(received_hours_ago)
    return evstore.add_event(UID, source="gmail", event_type="new_email",
                             summary=f"Email from {frm}: {subject}",
                             metadata=meta, dedup_key=f"gmail:{mid}")


def _backdate_event(evstore, event_id: str, hours: float) -> None:
    with evstore._lock:
        evstore._c().execute("UPDATE external_events SET ingested_at = ? WHERE id = ?",
                             (_iso(hours), event_id))
        evstore._c().commit()


def _backdate_action(store, action_id: str, hours: float) -> None:
    with store._lock:
        store._c().execute("UPDATE autonomous_actions SET created_at = ? WHERE id = ?",
                           (_iso(hours), action_id))
        store._c().commit()


def _put_trust(store, user_id: str, domain: str, *, weight=0.97, runs=49) -> None:
    with store._lock:
        store._c().execute(
            "INSERT OR REPLACE INTO autonomy_reliability"
            " (user_id, kind, domain, successes, fails, runs, weight, updated_at)"
            " VALUES (?, 'tool_action', ?, ?, 0, ?, ?, ?)",
            (user_id, domain, runs, runs, weight, _iso(0)))
        store._c().commit()


# ── 1–2. catalog ─────────────────────────────────────────────────────

def test_mail_draft_is_human_only_but_proposable():
    spec = action_catalog.get_action("mail.draft")
    assert spec["human_only"] is True and spec["proposable"] is True
    assert "reply_to_message_id" in spec["optional"]
    send = action_catalog.get_action("mail.send")
    assert send["human_only"] is True and not send.get("proposable")
    assert action_catalog.may_propose(spec) is True
    assert action_catalog.may_propose(send) is False
    assert action_catalog.may_propose(None) is False
    by_id = {a["id"]: a for a in action_catalog.list_catalog()}
    assert by_id["mail.draft"]["proposable"] is True and by_id["mail.draft"]["mail_write"] is True
    assert by_id["mail.send"]["proposable"] is False and by_id["mail.send"]["mail_write"] is True
    assert by_id["calendar.hold"]["mail_write"] is False


def test_is_mail_write_and_custom_email_actions_are_forced_human_only(store):
    assert action_catalog.is_mail_write(action_catalog.get_action("mail.draft"))
    assert action_catalog.is_mail_write(action_catalog.get_action("mail.send"))
    assert action_catalog.is_mail_write({"tool": "google_mail"})          # action from args
    assert action_catalog.is_mail_write({"tool": "send_mail"})
    assert not action_catalog.is_mail_write({"tool": "google_mail", "base_args": {"action": "list"}})
    assert not action_catalog.is_mail_write(action_catalog.get_action("calendar.hold"))
    assert not action_catalog.is_mail_write(None)
    store.set_overrides(UID, {"custom_actions": [
        {"id": "my.reply", "label": "Quick reply", "tool": "google_mail",
         "base_args": {"action": "create_draft"}, "human_only": False,
         "risk": "low", "reversibility": "reversible"},
    ]})
    spec = action_catalog.resolve_catalog(UID)["my.reply"]
    assert spec["human_only"] is True and not spec.get("proposable")
    assert action_catalog.may_propose(spec) is False


def test_custom_mcp_mail_tool_is_a_mail_write_and_never_auto_fires(store):
    for tool in ("mcp_gmail_create_draft", "mcp_gmail_send_message", "mcp_outlook_reply",
                 "mcp_gws_send_email", "mcp_work_mail_forward"):
        assert action_catalog.is_mail_write({"tool": tool}), tool
    for tool in ("mcp_gmail_search_threads", "mcp_slack_send_message", "mcp_gmail_get_thread"):
        assert not action_catalog.is_mail_write({"tool": tool}), tool
    store.set_overrides(UID, {"custom_actions": [
        {"id": "my.mcpdraft", "label": "MCP draft", "tool": "mcp_gmail_create_draft",
         "human_only": False, "risk": "low", "reversibility": "reversible"},
    ]})
    spec = action_catalog.resolve_catalog(UID)["my.mcpdraft"]
    assert spec["human_only"] is True and action_catalog.may_propose(spec) is False
    cfg = AutonomousWorkConfig().model_dump()
    cfg.update({"enabled": True, "allow_auto_dispatch": True, "autonomy_level": "act_low_risk",
                "granted_actions": ["my.mcpdraft", "custom"]})
    row = {"kind": "tool_action", "user_id": UID, "payload": {"action_id": "my.mcpdraft"}}
    assert fd_dispatch.should_auto_dispatch(cfg, row) is False


# ── 3. never auto-dispatched ─────────────────────────────────────────

def _act_cfg(**over) -> dict:
    cfg = AutonomousWorkConfig().model_dump()
    cfg.update({"enabled": True, "allow_auto_dispatch": True, "autonomy_level": "act_low_risk",
                "granted_actions": ["mail.draft", "mail", "calendar.hold", "my.reply", "custom"]})
    cfg.update(over)
    return cfg


def test_should_auto_dispatch_never_fires_email(store, monkeypatch):
    _put_trust(store, UID, "mail.draft")
    cfg = _act_cfg()
    row = {"kind": "tool_action", "user_id": UID, "payload": {"action_id": "mail.draft"}}
    assert fd_dispatch.should_auto_dispatch(cfg, row) is False
    # The plan rail's pseudo-action (plans.advance_one auto path).
    pseudo = {"kind": "tool_action", "user_id": UID, "payload": {"action_id": "mail.draft"}}
    assert fd_dispatch.should_auto_dispatch(cfg, pseudo) is False
    # A user's custom email action, granted and "not human-only".
    store.set_overrides(UID, {"custom_actions": [
        {"id": "my.reply", "label": "Quick reply", "tool": "google_mail",
         "base_args": {"action": "create_draft"}, "human_only": False,
         "risk": "low", "reversibility": "reversible"},
    ]})
    _put_trust(store, UID, "my.reply")
    custom = {"kind": "tool_action", "user_id": UID, "payload": {"action_id": "my.reply"}}
    assert fd_dispatch.should_auto_dispatch(cfg, custom) is False
    # No regression: a granted reversible low-risk action still auto-fires.
    hold = {"kind": "tool_action", "user_id": UID, "payload": {"action_id": "calendar.hold"}}
    assert fd_dispatch.should_auto_dispatch(cfg, hold) is True


# ── 4–6. arbiter ─────────────────────────────────────────────────────

@pytest.fixture
def ranker(monkeypatch, store):
    """A fake ranking LLM: returns ``ranker.reply`` and records the prompts."""
    import captain_claw.games.remote_provider as rp

    class _Resp:
        def __init__(self, content):
            self.content = content

    class _Provider:
        reply: list = []
        prompts: list = []

        def __init__(self, **kw):
            pass

        async def complete(self, **kw):
            msgs = kw.get("messages") or []
            _Provider.prompts.append({m.role: m.content for m in msgs})
            return _Resp(json.dumps(_Provider.reply))

    _Provider.reply, _Provider.prompts = [], []
    monkeypatch.setattr(rp, "RemoteLLMProvider", _Provider)

    async def _no_runs(uid):
        return []

    monkeypatch.setattr(arbiter, "_gather_active_runs", _no_runs)
    store.set_overrides(UID, {"enabled": True, "autonomy_level": "act_low_risk",
                              "allow_auto_dispatch": True,
                              "granted_actions": ["mail.draft", "mail"],
                              "quiet_hours_start": 0, "quiet_hours_end": 0})
    return _Provider


@pytest.fixture
def dispatched(monkeypatch):
    calls: list = []

    async def fake_dispatch(user_id, action, **kw):
        calls.append((action, kw))
        return {"ok": True, "target": "x", "note": ""}

    monkeypatch.setattr(fd_dispatch, "dispatch_action", fake_dispatch)
    return calls


async def _run_arbiter(trigger="manual"):
    return await arbiter.maybe_run_arbiter(UID, _REFLECTION, _AUTHOR, [], trigger=trigger)


def _user_prompt(ranker) -> str:
    return ranker.prompts[-1]["user"]


def _offered(ranker, ev) -> bool:
    """Whether the event reached the ranking LLM (no candidates → no LLM call)."""
    return any(ev["id"] in p["user"] for p in ranker.prompts)


async def test_arbiter_mail_draft_is_only_proposed(store, evstore, ranker, dispatched):
    _put_trust(store, UID, "mail.draft")
    ev = _gmail_event(evstore, "m1", "t1")
    ranker.reply = [{
        "kind": "tool_action", "action_id": "mail.draft", "title": "Reply to Ana about Q3",
        "rationale": "she asked", "risk": "low", "score": 0.9, "event_ref": ev["id"],
        "args": {"to": "ana@x.co", "subject": "Re: Q3 report", "body": "The Q3 numbers are attached."},
    }]
    res = await _run_arbiter()
    assert res["ran"] is True and res["dispatched"] is False, res
    row = store.get_action(res["action_id"])
    assert row["status"] == "awaiting_approval"
    assert row["payload"]["event_ref"] == ev["id"]
    assert row["payload"]["thread_key"] == "gmail:t1"
    assert dispatched == []
    assert any(r["event"] == "proposed" for r in store.list_log(UID))


async def test_arbiter_prompt_pins(store, evstore, ranker, dispatched):
    sp = arbiter._SYSTEM_PROMPT
    assert "PREFER that tool_action" not in sp
    assert "a draft, a reminder" not in sp
    assert "stop_run and tool_action are held for approval" not in sp
    assert "is waiting for a reply" in sp
    assert "nothing is drafted or sent until the user approves it" in sp
    _gmail_event(evstore, "m1", "t1")
    await _run_arbiter()
    up = _user_prompt(ranker)
    line = next(ln for ln in up.splitlines() if ln.startswith("- mail.draft:"))
    assert line.endswith(" · needs the user's approval")
    assert "- mail.send:" not in up


async def test_candidate_build_ignores_old_email(store, evstore, ranker, dispatched):
    old = _gmail_event(evstore, "m-old", "t-old", received_hours_ago=72)
    legacy = _gmail_event(evstore, "m-legacy", "t-legacy", received_hours_ago=None)
    _backdate_event(evstore, legacy["id"], 60)
    fresh = _gmail_event(evstore, "m-new", "t-new")
    await _run_arbiter()
    up = _user_prompt(ranker)
    assert old["id"] not in up and legacy["id"] not in up
    assert fresh["id"] in up
    assert evstore.get_event(old["id"])["status"] == "ignored"
    assert evstore.get_event(legacy["id"])["status"] == "ignored"
    logs = [r for r in store.list_log(UID) if r["event"] == "events too old"]
    assert logs and "2 email(s) older than 48h ignored" in logs[0]["detail"]


async def test_candidate_build_cutoff_can_be_disabled(store, evstore, ranker, dispatched):
    store.set_overrides(UID, {**store.get_overrides(UID), "gmail_event_max_age_hours": 0})
    old = _gmail_event(evstore, "m-old", "t-old", received_hours_ago=72)
    await _run_arbiter()
    assert old["id"] in _user_prompt(ranker)
    assert evstore.get_event(old["id"])["status"] != "ignored"


async def test_one_candidate_per_thread_the_newest_one(store, evstore, ranker, dispatched):
    newer = _gmail_event(evstore, "m2", "t1", received_hours_ago=1)   # ingested first
    older = _gmail_event(evstore, "m1", "t1", received_hours_ago=2)   # ingested second
    await _run_arbiter()
    up = _user_prompt(ranker)
    assert newer["id"] in up and older["id"] not in up
    assert evstore.get_event(older["id"])["status"] == "surfaced"


async def test_thread_with_an_open_action_is_skipped(store, evstore, ranker, dispatched):
    store.add_action(UID, kind="nudge", title="Ana is waiting", status="awaiting_approval",
                     payload={"thread_key": "gmail:t1"})
    ev = _gmail_event(evstore, "m2", "t1")
    await _run_arbiter()
    assert not _offered(ranker, ev)
    assert evstore.get_event(ev["id"])["status"] == "surfaced"
    assert any(r["event"] == "event skipped: thread already handled" for r in store.list_log(UID))


async def test_thread_already_covered_by_a_done_action_is_skipped(store, evstore, ranker, dispatched):
    ev = _gmail_event(evstore, "m2", "t1", received_hours_ago=5)
    done = store.add_action(UID, kind="nudge", title="Ana is waiting", status="done",
                            payload={"thread_key": "gmail:t1"})
    _backdate_action(store, done["id"], 2)          # created AFTER the message arrived
    await _run_arbiter()
    assert not _offered(ranker, ev)
    assert evstore.get_event(ev["id"])["status"] == "surfaced"


@pytest.mark.parametrize("status", ["done", "rejected"])
async def test_newer_message_after_a_resolved_action_is_surfaced(store, evstore, ranker, dispatched, status):
    old = store.add_action(UID, kind="nudge", title="Ana is waiting", status=status,
                           payload={"thread_key": "gmail:t1"})
    _backdate_action(store, old["id"], 25)          # a day before the new message
    ev = _gmail_event(evstore, "m3", "t1", received_hours_ago=1)
    await _run_arbiter()
    assert ev["id"] in _user_prompt(ranker)


async def test_thread_action_older_than_the_window_does_not_dedup(store, evstore, ranker, dispatched):
    old = store.add_action(UID, kind="nudge", title="Ana is waiting", status="done",
                           payload={"thread_key": "gmail:t1"})
    _backdate_action(store, old["id"], 8 * 24)
    ev = _gmail_event(evstore, "m3", "t1", received_hours_ago=1)
    await _run_arbiter()
    assert ev["id"] in _user_prompt(ranker)


def _waiting_nudge(ev: dict, title: str = "Ana Kovač is waiting for a reply: Re: Ugovor") -> dict:
    return {"kind": "nudge", "title": title, "rationale": "she wrote again", "risk": "low",
            "score": 0.8, "event_ref": ev["id"]}


async def test_a_newer_message_with_the_same_title_is_not_title_deduped(store, evstore, ranker, dispatched):
    # A back-and-forth keeps one subject → the deterministic nudge title repeats.
    title = "Ana Kovač is waiting for a reply: Re: Ugovor"
    done = store.add_action(UID, kind="nudge", title=title, status="done",
                            payload={"thread_key": "gmail:t1"})
    _backdate_action(store, done["id"], 4)          # inside the 24h title lookback
    ev = _gmail_event(evstore, "m3", "t1", received_hours_ago=1, subject="Re: Ugovor")
    ranker.reply = [_waiting_nudge(ev, title)]
    res = await _run_arbiter()
    assert res.get("proposed") == 1, res
    row = store.get_action(res["action_id"])
    assert row["payload"]["event_ref"] == ev["id"] and row["payload"]["thread_key"] == "gmail:t1"
    assert not [r for r in store.list_log(UID) if r["event"] == "dropped: already proposed"]
    assert "new information" in arbiter._SYSTEM_PROMPT


async def test_title_dedup_still_applies_to_non_email_actions(store, evstore, ranker, dispatched):
    done = store.add_action(UID, kind="nudge", title="Stretch your legs", status="done")
    _backdate_action(store, done["id"], 4)
    _gmail_event(evstore, "m1", "t1")               # a candidate, so the ranker runs
    ranker.reply = [{"kind": "nudge", "title": "Stretch your legs", "rationale": "r",
                     "risk": "low", "score": 0.9}]
    res = await _run_arbiter()
    assert res["reason"] == "nothing-viable", res
    assert [r for r in store.list_log(UID) if r["event"] == "dropped: already proposed"]


async def test_a_gmail_action_on_an_already_covered_message_is_still_dropped(store, evstore, ranker, dispatched):
    ev = _gmail_event(evstore, "m2", "t1", received_hours_ago=5)
    _backdate_event(evstore, ev["id"], 5)
    done = store.add_action(UID, kind="nudge", title="x", status="done",
                            payload={"thread_key": "gmail:t1"})
    _backdate_action(store, done["id"], 2)          # covered m2 (created after it arrived)
    _gmail_event(evstore, "m9", "t9")               # another candidate, so the ranker runs
    ranker.reply = [_waiting_nudge(ev)]             # a stale event_ref from the model
    res = await _run_arbiter()
    assert res["reason"] == "nothing-viable", res
    assert [r for r in store.list_log(UID) if r["event"] == "dropped: thread already handled"]


async def test_a_newer_message_refreshes_the_open_waiting_nudge(store, evstore, ranker, dispatched):
    old = _gmail_event(evstore, "m1", "t1", received_hours_ago=5)
    evstore.mark([old["id"]], "surfaced")
    nudge = store.add_action(UID, kind="nudge", title="Ana Kovač is waiting for a reply: Ugovor",
                             status="awaiting_approval",
                             payload={"event_ref": old["id"], "thread_key": "gmail:t1"})
    new = _gmail_event(evstore, "m2", "t1", received_hours_ago=1, subject="Re: Ugovor")
    await _run_arbiter()
    assert not _offered(ranker, new)               # still one open item per thread
    assert evstore.get_event(new["id"])["status"] == "surfaced"
    row = store.get_action(nudge["id"])
    assert row["status"] == "awaiting_approval"
    assert row["payload"] == {"event_ref": new["id"], "thread_key": "gmail:t1"}
    assert any(r["event"] == "open proposal updated: newer email in thread"
               for r in store.list_log(UID))
    # Approving it now tells the user about the NEWEST message.
    assert "Re: Ugovor" in fd_dispatch._reply_waiting_line(fd_dispatch._gmail_event_md(row))


async def test_an_older_message_does_not_rewind_the_open_nudge(store, evstore, ranker, dispatched):
    cur = _gmail_event(evstore, "m2", "t1", received_hours_ago=1)
    evstore.mark([cur["id"]], "surfaced")
    nudge = store.add_action(UID, kind="nudge", title="Ana is waiting", status="awaiting_approval",
                             payload={"event_ref": cur["id"], "thread_key": "gmail:t1"})
    older = _gmail_event(evstore, "m1", "t1", received_hours_ago=3)
    await _run_arbiter()
    assert store.get_action(nudge["id"])["payload"]["event_ref"] == cur["id"]
    assert evstore.get_event(older["id"])["status"] == "surfaced"


async def test_an_open_mail_draft_proposal_is_not_retargeted(store, evstore, ranker, dispatched):
    old = _gmail_event(evstore, "m1", "t1", received_hours_ago=5)
    evstore.mark([old["id"]], "surfaced")
    draft = store.add_action(UID, kind="tool_action", title="Reply to Ana", status="awaiting_approval",
                             payload={"action_id": "mail.draft", "args": {"to": "ana@x.co"},
                                      "event_ref": old["id"], "thread_key": "gmail:t1"})
    new = _gmail_event(evstore, "m2", "t1", received_hours_ago=1)
    await _run_arbiter()
    assert store.get_action(draft["id"])["payload"]["event_ref"] == old["id"]
    assert evstore.get_event(new["id"])["status"] == "surfaced"


@pytest.mark.parametrize("status", ["queued", "dispatched", "awaiting_approval"])
async def test_an_open_row_older_than_the_window_no_longer_hides_the_thread(
        store, evstore, ranker, dispatched, status):
    parked = store.add_action(UID, kind="nudge", title="Ana is waiting", status=status,
                              payload={"thread_key": "gmail:t1"})
    _backdate_action(store, parked["id"], 8 * 24)   # reply_thread_dedup_days = 7
    ev = _gmail_event(evstore, "m3", "t1", received_hours_ago=1)
    await _run_arbiter()
    assert ev["id"] in _user_prompt(ranker)


def test_has_open_thread_action_is_bounded_by_since(store):
    row = store.add_action(UID, kind="nudge", title="t", status="queued",
                           payload={"thread_key": "gmail:t1"})
    _backdate_action(store, row["id"], 8 * 24)
    week = _iso(7 * 24)
    assert store.has_open_thread_action(UID, "gmail:t1") is True          # unbounded
    assert store.has_open_thread_action(UID, "gmail:t1", since_iso=week) is False
    assert store.open_thread_action(UID, "gmail:t1")["id"] == row["id"]
    fresh = store.add_action(UID, kind="nudge", title="t2", status="dispatched",
                             payload={"thread_key": "gmail:t1"})
    assert store.open_thread_action(UID, "gmail:t1", since_iso=week)["id"] == fresh["id"]
    assert store.has_open_thread_action(UID, "gmail:t2", since_iso=week) is False


async def test_events_not_acted_on_stay_new_for_the_next_pass(store, evstore, ranker, dispatched):
    evs = [_gmail_event(evstore, f"m{i}", f"t{i}", frm=f"P{i} <p{i}@x.co>") for i in range(3)]
    ranker.reply = [_waiting_nudge(evs[0], "P0 is waiting for a reply: Q3 report")]
    res = await _run_arbiter()
    assert res.get("proposed") == 1, res
    assert evstore.get_event(evs[0]["id"])["status"] == "surfaced"
    for ev in evs[1:]:
        got = evstore.get_event(ev["id"])
        assert got["status"] == "new" and got["surface_count"] == 1
    log = [r for r in store.list_log(UID) if r["event"] == "events carried over"]
    assert log and log[0]["detail"].startswith("2 event(s)")
    # The next pass offers the two others and nudges about one of them.
    ranker.reply = [_waiting_nudge(evs[1], "P1 is waiting for a reply: Q3 report")]
    res = await _run_arbiter()
    assert res.get("proposed") == 1, res
    up = _user_prompt(ranker)
    assert evs[1]["id"] in up and evs[2]["id"] in up and evs[0]["id"] not in up
    assert evstore.get_event(evs[1]["id"])["status"] == "surfaced"
    assert evstore.get_event(evs[2]["id"])["status"] == "new"


async def test_carried_over_events_are_given_up_after_the_attempt_cap(store, evstore, ranker, dispatched):
    store.set_overrides(UID, {**store.get_overrides(UID), "event_max_surface_attempts": 1})
    a = _gmail_event(evstore, "m1", "t1")
    b = _gmail_event(evstore, "m2", "t2", frm="Bo <bo@x.co>")
    ranker.reply = [_waiting_nudge(a, "Ana is waiting for a reply: Q3 report")]
    await _run_arbiter()
    assert evstore.get_event(a["id"])["status"] == "surfaced"
    assert evstore.get_event(b["id"])["status"] == "ignored"


async def test_a_burst_larger_than_the_attempt_cap_gets_a_nudge_per_email(store, evstore, ranker, dispatched):
    # 5 important emails at once, default cap of 4 attempts: an email the model
    # wanted to act on but that lost the one slot must not spend an attempt.
    store.set_overrides(UID, {**store.get_overrides(UID), "max_pending_proposals": 50,
                              "max_actions_per_day": 50})
    evs = [_gmail_event(evstore, f"m{i}", f"t{i}", frm=f"P{i} <p{i}@x.co>") for i in range(5)]
    nudged = []
    for _ in range(6):
        offered = [e for e in evs if evstore.get_event(e["id"])["status"] == "new"]
        if not offered:
            break
        ranker.reply = [_waiting_nudge(e, f"P{i} is waiting for a reply: Q3 report")
                        for i, e in enumerate(offered)]
        res = await _run_arbiter()
        nudged.append(store.get_action(res["action_id"])["payload"]["event_ref"])
    assert sorted(nudged) == sorted(e["id"] for e in evs)
    assert not [r for r in store.list_log(UID) if r["event"] == "events given up"]


async def test_an_event_no_action_wanted_still_spends_an_attempt(store, evstore, ranker, dispatched):
    a = _gmail_event(evstore, "m1", "t1")
    b = _gmail_event(evstore, "m2", "t2", frm="Bo <bo@x.co>")
    c = _gmail_event(evstore, "m3", "t3", frm="Cy <cy@x.co>")
    ranker.reply = [_waiting_nudge(a, "Ana is waiting for a reply: Q3 report"),
                    {**_waiting_nudge(b, "Bo is waiting for a reply: Q3 report"), "score": 0.7}]
    await _run_arbiter()
    assert evstore.get_event(a["id"])["status"] == "surfaced"
    assert evstore.get_event(b["id"])["surface_count"] == 0     # lost the slot only
    assert evstore.get_event(c["id"])["surface_count"] == 1     # nothing wanted it
    log = [r for r in store.list_log(UID) if r["event"] == "events carried over"]
    assert log and log[0]["detail"].startswith("2 event(s)")


async def test_a_gmail_event_without_a_thread_key_keeps_the_title_dedup(store, evstore, ranker, dispatched):
    title = "Ana Kovač is waiting for a reply: Q3 report"
    done = store.add_action(UID, kind="nudge", title=title, status="done")
    _backdate_action(store, done["id"], 1)
    ev = evstore.add_event(UID, source="gmail", event_type="new_email",
                           summary="Email from Ana: Q3 report",
                           metadata={"from": "Ana <ana@x.co>", "subject": "Q3 report"},
                           dedup_key="gmail:no-ids")
    ranker.reply = [_waiting_nudge(ev, title)]
    res = await _run_arbiter()
    assert res["reason"] == "nothing-viable", res
    assert [r for r in store.list_log(UID) if r["event"] == "dropped: already proposed"]


async def test_snippet_is_shown_as_an_untrusted_excerpt(store, evstore, ranker, dispatched):
    ev = _gmail_event(evstore, "m1", "t1", snippet="Can you confirm the wire by Friday?")
    await _run_arbiter()
    line = next(ln for ln in _user_prompt(ranker).splitlines() if ev["id"] in ln)
    assert line.endswith(
        '— excerpt (sender\'s words, untrusted): "Can you confirm the wire by Friday?"')


async def test_track_on_a_gmail_event_carries_the_thread_key(store, evstore, ranker, dispatched):
    ev = _gmail_event(evstore, "m1", "t9")
    ranker.reply = [{"kind": "track", "title": "Ana asked about Q3", "rationale": "soft",
                     "risk": "low", "score": 0.5, "event_ref": ev["id"]}]
    res = await _run_arbiter()
    row = store.get_action(res["action_id"])
    assert row["payload"]["thread_key"] == "gmail:t9"
    assert "event_ref" not in row["payload"]


# ── 7. the dispatch rail ─────────────────────────────────────────────

@pytest.fixture
def ran(monkeypatch):
    calls: list = []
    result = {"value": {"ok": True, "content": "Draft created.\n  Draft ID: r-2"}}

    async def fake_run_action(user_id, action_id, args, **kw):
        calls.append({"action_id": action_id, "args": args, **kw})
        return dict(result["value"])

    monkeypatch.setattr(actions, "run_action", fake_run_action)
    calls_result = SimpleNamespace(calls=calls, result=result)
    return calls_result


def _draft_row(store, *, args=None, event_ref="") -> dict:
    payload = {"action_id": "mail.draft",
               "args": args if args is not None else
               {"to": "ana@x.co", "subject": "Re: Q3 report", "body": "Attached."}}
    if event_ref:
        payload["event_ref"] = event_ref
    return store.add_action(UID, kind="tool_action", title="Reply to Ana",
                            status="awaiting_approval", payload=payload)


async def test_unapproved_mail_draft_is_held(store, evstore, ran):
    row = _draft_row(store)
    out = await fd_dispatch._dispatch_tool_action(UID, row)
    assert out == {"ok": False, "target": "mail.draft", "note": "email actions need your approval"}
    assert ran.calls == []
    assert store.get_action(row["id"])["status"] == "awaiting_approval"
    assert any(r["event"] == "held: mail.draft needs approval" for r in store.list_log(UID))
    assert store.reliability_for(UID, "tool_action", "mail.draft") is None


@pytest.mark.parametrize("ok", [True, False])
async def test_approved_mail_draft_runs_threaded_and_is_never_learned(store, evstore, ran, ok):
    ev = _gmail_event(evstore, "m77", "t77")
    ran.result["value"] = ({"ok": True, "content": "Draft created."} if ok
                           else {"ok": False, "error": "Google authentication expired."})
    row = _draft_row(store, event_ref=ev["id"])
    await fd_dispatch._dispatch_tool_action(UID, row, approved_by_human=True)
    (call,) = ran.calls
    assert call["approved_by_human"] is True
    assert call["args"]["reply_to_message_id"] == "m77"
    assert store.get_action(row["id"])["outcome"] == ("success" if ok else "fail")
    assert store.reliability_for(UID, "tool_action", "mail.draft") is None


async def test_enrichment_addresses_reply_to_then_sender_never_overrides(store, evstore):
    ev = _gmail_event(evstore, "m1", "t1", frm="Ana <ana@x.co>", reply_to="list@x.co")
    out = fd_dispatch._enrich_args_from_event("mail.draft", {"subject": "Re: x", "body": "b"}, ev["id"])
    assert out["to"] == "list@x.co" and out["reply_to_message_id"] == "m1"
    ev2 = _gmail_event(evstore, "m2", "t2", frm="Ana <ana@x.co>")
    out2 = fd_dispatch._enrich_args_from_event("mail.draft", {"body": "b"}, ev2["id"])
    assert out2["to"] == "ana@x.co" and out2["subject"] == "Re: Q3 report"
    out3 = fd_dispatch._enrich_args_from_event("mail.draft", {"to": "boss@x.co", "body": "b"}, ev["id"])
    assert out3["to"] == "boss@x.co"


async def test_agent_refusal_is_a_fail_never_learned(store, evstore, ran):
    ran.result["value"] = {"ok": False, "error": "[not-authorized: mail-write] Not created: this turn …"}
    row = _draft_row(store)
    await fd_dispatch._dispatch_tool_action(UID, row, approved_by_human=True)
    assert store.get_action(row["id"])["outcome"] == "fail"
    assert store.reliability_for(UID, "tool_action", "mail.draft") is None
    (entry, *_) = store.list_log(UID)
    assert "refused: not authorized" in entry["event"] and entry["level"] == "warn"


# ── 8. approve route ─────────────────────────────────────────────────

def _req():
    return SimpleNamespace(state=SimpleNamespace(user_id=UID))


@pytest.fixture
def approve_spies(monkeypatch):
    seen = {"dispatch": [], "feedback": []}
    real_feedback = autonomy_routes.record_human_feedback

    def spy_feedback(uid, action, approved):
        seen["feedback"].append(approved)
        return real_feedback(uid, action, approved)

    async def spy_dispatch(uid, action, **kw):
        seen["dispatch"].append(kw)
        return seen.get("disp_result") or {"ok": True, "target": "x", "note": ""}

    monkeypatch.setattr(autonomy_routes, "record_human_feedback", spy_feedback)
    monkeypatch.setattr(fd_dispatch, "dispatch_action", spy_dispatch)
    return seen


async def test_approving_twice_is_a_409(store, approve_spies):
    row = _draft_row(store)
    out = await autonomy_routes.approve_action_route(row["id"], _req(), None)
    assert out["reliability"] is None             # an approval never earns email trust
    with pytest.raises(HTTPException) as exc:
        await autonomy_routes.approve_action_route(row["id"], _req(), None)
    assert exc.value.status_code == 409
    assert approve_spies["dispatch"] == [{"approved_by_human": True}]
    assert approve_spies["feedback"] == [True]
    assert store.reliability_for(UID, "tool_action", "mail.draft") is None


@pytest.mark.parametrize("status", ["done", "rejected", "expired"])
async def test_approving_a_resolved_row_is_a_409(store, approve_spies, status):
    row = store.add_action(UID, kind="nudge", title="x", status=status)
    with pytest.raises(HTTPException) as exc:
        await autonomy_routes.approve_action_route(row["id"], _req(), None)
    assert exc.value.status_code == 409
    assert approve_spies["dispatch"] == [] and approve_spies["feedback"] == []


async def test_a_queued_row_can_be_approved_and_a_failed_dispatch_requeues(store, approve_spies):
    row = store.add_action(UID, kind="nudge", title="x", status="queued")
    approve_spies["disp_result"] = {"ok": False, "target": "", "note": "no running agent"}
    await autonomy_routes.approve_action_route(row["id"], _req(), None)
    assert approve_spies["dispatch"] == [{"approved_by_human": True}]
    assert store.get_action(row["id"])["status"] == "queued"


async def test_a_crashed_dispatch_does_not_strand_the_claimed_row(store, monkeypatch):
    row = _draft_row(store)

    async def boom(uid, action, **kw):
        raise RuntimeError("agent socket blew up")

    monkeypatch.setattr(fd_dispatch, "dispatch_action", boom)
    with pytest.raises(RuntimeError):
        await autonomy_routes.approve_action_route(row["id"], _req(), None)
    assert store.get_action(row["id"])["status"] == "queued"


async def test_reject_records_one_fail(store):
    row = _draft_row(store)
    await autonomy_routes.reject_action_route(row["id"], _req(), None)
    rel = store.reliability_for(UID, "tool_action", "mail.draft")
    assert rel and rel["fails"] == 1 and rel["successes"] == 0


@pytest.mark.parametrize("ok", [True, False])
async def test_approve_does_not_double_count(store, ran, ok):
    ran.result["value"] = {"ok": True, "content": "Event created. id: ev1"} if ok else \
        {"ok": False, "error": "calendar down"}
    row = store.add_action(UID, kind="tool_action", title="Hold 3pm", status="awaiting_approval",
                           payload={"action_id": "calendar.hold",
                                    "args": {"summary": "x", "start": "s", "end": "e"}})
    await autonomy_routes.approve_action_route(row["id"], _req(), None)
    rel = store.reliability_for(UID, "tool_action", "calendar.hold")
    assert rel["successes"] == 1 and rel["fails"] == (0 if ok else 1)


# ── 9. trust sweep ───────────────────────────────────────────────────

def _log_rows(path, user_id):
    s = autonomy.AutonomyStore(path)
    return [r for r in s.list_log(user_id) if r["event"] == "mail trust reset"], s


def test_trust_sweep_runs_on_every_start(tmp_path):
    path = tmp_path / "shared.db"
    s1 = autonomy.AutonomyStore(path)
    for uid in ("u1", "u2"):
        _put_trust(s1, uid, "mail.draft")
        _put_trust(s1, uid, "mail.send")
        _put_trust(s1, uid, "calendar.hold")
    s2 = autonomy.AutonomyStore(path)
    for uid in ("u1", "u2"):
        assert s2.reliability_for(uid, "tool_action", "mail.draft") is None
        assert s2.reliability_for(uid, "tool_action", "mail.send") is None
        assert s2.reliability_for(uid, "tool_action", "calendar.hold") is not None
        logs = [r for r in s2.list_log(uid) if r["event"] == "mail trust reset"]
        assert len(logs) == 1 and logs[0]["level"] == "info"
        assert logs[0]["detail"] == ("Email actions now always wait for your approval; "
                                     "their learned trust was cleared.")
    # Idempotent: nothing to clear → nothing deleted, nothing logged.
    s3 = autonomy.AutonomyStore(path)
    assert len([r for r in s3.list_log("u1") if r["event"] == "mail trust reset"]) == 1
    # An old-code deck re-learned trust in between → the next start clears it again.
    _put_trust(s3, "u1", "mail.draft")
    s4 = autonomy.AutonomyStore(path)
    assert s4.reliability_for("u1", "tool_action", "mail.draft") is None
    assert len([r for r in s4.list_log("u1") if r["event"] == "mail trust reset"]) == 2
    assert len([r for r in s4.list_log("u2") if r["event"] == "mail trust reset"]) == 1


# ── 10. instructions ─────────────────────────────────────────────────

def test_gmail_nudge_says_who_is_waiting_and_forbids_mail(store, evstore):
    ev = _gmail_event(evstore, "m1", "t1", frm='"Ana Kovač" <ana@x.co>', subject="Q3 report")
    nudge = {"kind": "nudge", "title": "Ana is waiting", "rationale": "Asked yesterday.",
             "payload": {"event_ref": ev["id"]}}
    text = fd_dispatch._instruction_for(nudge)
    assert 'Ana Kovač is waiting for a reply — "Q3 report"' in text
    assert fd_dispatch._NO_MAIL_LINE in text
    assert fd_dispatch._NO_MAIL_LINE in fd_dispatch._instruction_for(nudge, approved_by_human=True)
    rp = {"kind": "run_prompt", "title": "Look into Q3", "rationale": "r", "payload": {}}
    assert fd_dispatch._NO_MAIL_LINE in fd_dispatch._instruction_for(rp)
    assert fd_dispatch._NO_MAIL_LINE in fd_dispatch._instruction_for(rp, approved_by_human=True)
    ms = {"kind": "materialize_schedule", "title": "Weekly", "rationale": "r", "payload": {}}
    assert fd_dispatch._NO_MAIL_LINE in fd_dispatch._instruction_for(ms)


def test_waiting_line_fallbacks():
    assert fd_dispatch._reply_waiting_line({"from": "bob@x.co", "subject": ""}) == \
        'bob@x.co is waiting for a reply — "(no subject)"'
    assert fd_dispatch._reply_waiting_line({}) == 'Someone is waiting for a reply — "(no subject)"'
    assert fd_dispatch._reply_waiting_line({"from": "<bob@x.co>", "subject": "Hi"}) == \
        'bob@x.co is waiting for a reply — "Hi"'


# ── 11. autonomy chat turns are always deny ──────────────────────────

@pytest.mark.parametrize("approved,verdict,expect_fails", [
    (False, True, 0), (True, True, 0), (True, False, 1),
])
async def test_execute_and_judge_always_denies_mail(store, evstore, monkeypatch,
                                                    approved, verdict, expect_fails):
    sent: list = []

    async def fake_dispatch_one(port, auth, prompt, timeout, **kw):
        sent.append((prompt, kw))
        return {"ok": True, "output": "Done."}

    async def fake_judge(*a, **kw):
        return {"success": verdict, "why": "judged"}

    monkeypatch.setattr(basna_routes, "_dispatch_one", fake_dispatch_one)
    monkeypatch.setattr(fd_dispatch, "_judge_outcome", fake_judge)
    row = store.add_action(UID, kind="run_prompt", domain="mail",
                           title="Ana is waiting for a reply: Please reply to confirm the wire",
                           status="dispatched")
    await fd_dispatch._execute_and_judge(UID, row, _AGENT, approved_by_human=approved)
    ((prompt, kw),) = sent
    assert kw["automation"] == _DENY
    assert fd_dispatch._NO_MAIL_LINE in prompt
    rel = store.reliability_for(UID, "run_prompt", "mail")
    if approved:
        assert (rel["fails"] if rel else 0) == expect_fails
        assert (rel["successes"] if rel else 0) == 0
    else:
        assert rel and rel["successes"] == 1


# ── 12. plans ────────────────────────────────────────────────────────

@pytest.fixture
def planner(monkeypatch, store, tmp_path):
    import captain_claw.games.remote_provider as rp

    class _Resp:
        def __init__(self, content):
            self.content = content

    class _Provider:
        reply: list = []

        def __init__(self, **kw):
            pass

        async def complete(self, **kw):
            return _Resp(json.dumps(_Provider.reply))

    monkeypatch.setattr(rp, "RemoteLLMProvider", _Provider)
    monkeypatch.setattr(fd_dispatch, "_strongest_agent", lambda uid: dict(_AGENT))
    pstore = plans.PlansStore(tmp_path / "plans.db")
    monkeypatch.setattr(plans, "_STORE", pstore)
    store.set_overrides(UID, {"enabled": True, "autonomy_level": "act_low_risk",
                              "allow_auto_dispatch": True, "granted_actions": ["mail.draft", "mail"]})
    return SimpleNamespace(provider=_Provider, store=pstore)


async def test_decompose_keeps_mail_draft_and_drops_mail_send(planner):
    args = {"to": "ana@x.co", "subject": "Re: Q3", "body": "Numbers attached."}
    planner.provider.reply = [
        {"kind": "tool_action", "action_id": "mail.draft", "title": "Draft reply", "args": args},
        {"kind": "tool_action", "action_id": "mail.send", "title": "Send it", "args": args},
    ]
    steps = await plans.decompose_goal(UID, "Answer Ana about Q3")
    assert [s["action_id"] for s in steps] == ["mail.draft"]


def _draft_plan(planner, goal="Draft a reply to Ana about Q3"):
    return planner.store.create_plan(UID, goal, [{
        "kind": "tool_action", "action_id": "mail.draft", "title": "Draft reply",
        "args": {"to": "ana@x.co", "subject": "Re: Q3", "body": "Numbers attached."},
    }])


async def test_auto_advance_pauses_on_a_mail_draft(planner, ran):
    plan = _draft_plan(planner)
    out = await plans.advance_one(UID, plan["id"], auto=True)
    assert out.get("paused") is True
    assert ran.calls == []
    assert "awaiting approval" in planner.store.get_plan(plan["id"])["note"]


async def test_manual_advance_is_an_approval(planner, ran):
    plan = _draft_plan(planner)
    await plans.advance_one(UID, plan["id"], auto=False)
    (call,) = ran.calls
    assert call["action_id"] == "mail.draft" and call["approved_by_human"] is True


async def test_plan_run_prompt_step_carries_the_goal(planner, monkeypatch):
    sent: list = []

    async def fake_dispatch_one(port, auth, prompt, timeout, **kw):
        sent.append(kw)
        return {"ok": True, "output": "done"}

    monkeypatch.setattr(basna_routes, "_dispatch_one", fake_dispatch_one)
    goal = "Email me the Q3 summary"
    plan = planner.store.create_plan(UID, goal, [{"kind": "run_prompt", "title": "Summarize Q3"}])
    await plans.advance_one(UID, plan["id"], auto=False)
    assert sent[0]["automation"] == {"kind": "plan", "job_text": goal, "mail_write": "intent"}
