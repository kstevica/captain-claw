"""First Mate's store: the rules every factory follows, whoever calls it.

Pinned here:

* a version's fields are checked and normalized, and never hold a secret;
* the factory cap per user (retired factories don't count);
* the lifecycle: draft → awaiting charter → running, approval by the owner
  only and against the hash they saw, one open draft, decline, set_state;
* trust: who may raise it, pins, no ``direct`` with outward grants, back to
  ``propose`` when a version adds a grant;
* runs: only after approval, one at a time, the daily cap, which states allow
  which triggers, totals that only grow, a run that ends once;
* intents and signals, satisfaction;
* a refused write leaves nothing behind;
* deleting a factory, forgetting a user.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from captain_claw.flight_deck import first_mate_store as fm

BOB, CAROL = "user-bob", "user-carol"


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    return s


def _make(store, owner=BOB, **kw):
    kw.setdefault("title", "Inbox drafts")
    kw.setdefault("charter", "Draft replies to mail that needs one.")
    return store.create(owner, **kw)


def _live(store, owner=BOB, **kw):
    f = _make(store, owner, **kw)
    store.submit(f["id"], "owner")
    return store.approve(f["id"], 1, f["draft"]["sha"], owner)


# ── versions: fields ───────────────────────────────────────────────────

def test_create_starts_as_a_draft_with_normalized_fields(store):
    f = _make(store)
    assert f["state"] == "draft" and f["version"] == 0 and f["draft_version"] == 1
    assert f["live"] is None and f["trust"] == "propose"
    d = f["draft"]
    assert d["status"] == "draft" and d["runner"] == "funnel"  # the list funnel, when nothing says otherwise
    assert d["runner_config"] == {"leftovers": "agent", "loosen": "short"}
    # settings that are plainly Bat's name Bat; a runner named is the runner
    assert _make(store, title="B", runner_config={"worker_mode": "plain"})["draft"]["runner"] == "bat"
    assert _make(store, title="C", runner="bat")["draft"]["runner_config"] == {"worker_mode": "plain", "steps": []}
    assert d["grants"] == fm.DEFAULT_GRANTS
    assert d["checks"] == fm.DEFAULT_CHECKS
    assert d["budget"] == {"run_tokens": None}
    assert d["sha"] == fm.content_sha(d)
    assert [e["event"] for e in store.events(f["id"])] == ["created"]


@pytest.mark.parametrize("bad", [
    {"api_key": "x"},
    {"nested": {"password": "x"}},
    {"steps": ["sk-abcdefghijklmnop"]},
    {"headers": [{"Auth_Token": "y"}]},
])
def test_secrets_never_go_into_a_version(store, bad):
    with pytest.raises(fm.FirstMateError, match="secret"):
        _make(store, runner="vatra", runner_config=bad)


def test_runner_checks(store):
    with pytest.raises(fm.FirstMateError, match="runner must be"):
        _make(store, runner="robot")
    with pytest.raises(fm.FirstMateError, match="unknown fields"):
        _make(store, runner_config={"worker_mode": "plain", "model": "x"})
    with pytest.raises(fm.FirstMateError, match="worker_mode"):
        _make(store, runner_config={"worker_mode": "wild"})
    with pytest.raises(fm.FirstMateError, match="flow_id is required"):
        _make(store, runner="flow")
    f = _make(store, runner="flow", runner_config={"flow_id": "weekly"})
    assert f["draft"]["runner_config"] == {"flow_id": "weekly"}


def test_triggers(store):
    f = _make(store, triggers=[
        {"kind": "schedule", "schedule": "daily 08:00"},
        {"kind": "events", "source": "gmail", "match": {"from": ["a@example.com"]}},
    ])
    sched, ev = f["draft"]["triggers"]
    assert sched == {"kind": "schedule", "schedule": "daily 08:00", "tz": ""}
    assert ev["match"] == {"from": ["a@example.com"], "label": [], "words": []}
    assert ev["batch_minutes"] is None and ev["ask"] is False
    with pytest.raises(fm.FirstMateError, match="schedule"):
        _make(store, triggers=[{"kind": "schedule", "schedule": "whenever"}])
    with pytest.raises(fm.FirstMateError, match="source"):
        _make(store, triggers=[{"kind": "events", "source": "Gmail Inbox!"}])
    with pytest.raises(fm.FirstMateError, match="unknown fields"):
        _make(store, triggers=[{"kind": "events", "source": "gmail", "run": "now"}])
    with pytest.raises(fm.FirstMateError, match="kind must be"):
        _make(store, triggers=[{"kind": "webhook"}])
    with pytest.raises(fm.FirstMateError, match="at most"):
        _make(store, triggers=[{"kind": "schedule", "schedule": "every 1h"}] * 6)


def test_grants(store):
    g = _make(store, grants={"email": True})["draft"]["grants"]
    assert g["email"] and g["sends_per_run"] == 10 and g["replies_only"]
    with pytest.raises(fm.FirstMateError, match="sends_per_run"):
        _make(store, grants={"email": True, "sends_per_run": 0})
    with pytest.raises(fm.FirstMateError, match="weekly cap"):
        _make(store, grants={"spend": True})
    with pytest.raises(fm.FirstMateError, match="per_item_usd"):
        _make(store, grants={"spend": True, "real_usd_week": 20, "per_item_usd": 50})
    with pytest.raises(fm.FirstMateError, match="unknown fields"):
        _make(store, grants={"post": True})
    # Limits of a grant that isn't given are dropped.
    g = _make(store, grants={"sends_per_run": 5, "per_item_usd": 3})["draft"]["grants"]
    assert g["sends_per_run"] == 0 and g["per_item_usd"] == 0.0


def test_text_and_number_caps(store):
    with pytest.raises(fm.FirstMateError, match="charter is required"):
        _make(store, charter="  ")
    with pytest.raises(fm.FirstMateError, match="too long"):
        _make(store, charter="x" * (fm.CHARTER_MAX + 1))
    with pytest.raises(fm.FirstMateError, match="run_tokens"):
        _make(store, budget={"run_tokens": 10})
    with pytest.raises(fm.FirstMateError, match="satisfaction_target"):
        _make(store, checks={"satisfaction_target": 0.2})
    with pytest.raises(fm.FirstMateError, match="share"):
        _make(store, share=0)


def test_factory_cap_counts_only_factories_not_retired(store):
    store.set_settings({"max_factories": 2})
    a = _make(store)
    _make(store)
    with pytest.raises(fm.FirstMateError, match="at most 2"):
        _make(store)
    _make(store, CAROL)  # another user has their own count
    store.set_state(a["id"], "retired", "owner")
    _make(store)
    assert len(store.list_factories(BOB)) == 2
    assert len(store.list_factories(BOB, include_retired=True)) == 3


# ── lifecycle ──────────────────────────────────────────────────────────

def test_submit_and_approve_start_the_factory(store):
    f = _make(store)
    f = store.submit(f["id"], "owner")
    assert f["state"] == "awaiting_charter" and f["draft"]["status"] == "awaiting"
    f = store.approve(f["id"], 1, f["draft"]["sha"], BOB)
    assert f["state"] == "running" and f["version"] == 1 and f["draft_version"] == 0
    assert f["live"]["status"] == "live" and f["live"]["decided_by"] == BOB
    assert f["draft"] is None


def test_only_the_owner_approves_and_only_what_they_saw(store):
    f = _make(store)
    sha = f["draft"]["sha"]
    with pytest.raises(fm.FirstMateError, match="owner"):
        store.approve(f["id"], 1, sha, CAROL)
    with pytest.raises(fm.FirstMateError, match="owner"):
        store.decline(f["id"], 1, CAROL)
    store.edit_draft(f["id"], "first_mate", charter="Draft replies, and archive newsletters.")
    with pytest.raises(fm.FirstMateError, match="changed since you saw it"):
        store.approve(f["id"], 1, sha, BOB)
    fresh = store.view(f["id"])["draft"]["sha"]
    assert fresh != sha
    assert store.approve(f["id"], 1, fresh, BOB)["state"] == "running"


def test_a_local_owner_approves_with_an_empty_id(store):
    f = _make(store, owner="")
    assert f["owner_id"] == "local"
    assert store.approve(f["id"], 1, f["draft"]["sha"], "")["state"] == "running"


def test_one_open_draft_and_history_kept(store):
    f = _live(store)
    f = store.new_draft(f["id"], "first_mate", note="tighter", checks={"contract": ["a reply per thread"]})
    assert f["draft_version"] == 2 and f["draft"]["charter"] == f["live"]["charter"]
    assert f["draft"]["checks"]["contract"] == ["a reply per thread"]
    with pytest.raises(fm.FirstMateError, match="already has a draft"):
        store.new_draft(f["id"], "first_mate", charter="other")
    f = store.approve(f["id"], 2, f["draft"]["sha"], BOB)
    assert f["version"] == 2 and f["state"] == "running"
    assert [(v["version"], v["status"]) for v in store.versions(f["id"])] == [(2, "live"), (1, "replaced")]
    with pytest.raises(fm.FirstMateError, match="not part of a version"):
        store.new_draft(f["id"], "first_mate", title="renamed")


def test_decline(store):
    f = _make(store)
    store.submit(f["id"], "owner")
    f = store.decline(f["id"], 1, BOB, "too broad")
    assert f["state"] == "draft" and f["draft"] is None
    with pytest.raises(fm.FirstMateError, match="no draft"):
        store.submit(f["id"], "owner")
    f = store.new_draft(f["id"], "owner", charter="Draft replies to my clients only.")
    assert f["draft_version"] == 2 and f["draft"]["status"] == "draft"
    # Declining a later version leaves a live factory running.
    f = store.approve(f["id"], 2, f["draft"]["sha"], BOB)
    f = store.new_draft(f["id"], "first_mate", grants={"email": True})
    f = store.decline(f["id"], 3, BOB)
    assert f["state"] == "running" and f["version"] == 2 and f["draft"] is None


def test_set_state(store):
    f = _make(store)
    with pytest.raises(fm.FirstMateError, match="can't become"):
        store.set_state(f["id"], "running", "owner")
    f = _live(store)
    f = store.set_state(f["id"], "paused", "owner", "trying something")
    assert f["state"] == "paused" and f["state_reason"] == "trying something"
    assert store.set_state(f["id"], "running", "owner")["state"] == "running"
    store.set_state(f["id"], "paused_budget", "deck", "used its share")
    assert store.set_state(f["id"], "running", "deck")["state"] == "running"
    store.set_state(f["id"], "retired", "owner")
    for state in ("running", "paused", "draft"):
        with pytest.raises(fm.FirstMateError):
            store.set_state(f["id"], state, "owner")
    with pytest.raises(fm.FirstMateError, match="retired"):
        store.update_factory(f["id"], "owner", title="again")


def test_update_factory_needs_no_approval(store):
    f = _live(store)
    f = store.update_factory(f["id"], "owner", title="Mail drafts", share=3, priority=1, goal_id="goal_1")
    assert (f["title"], f["share"], f["priority"], f["goal_id"]) == ("Mail drafts", 3, 1, "goal_1")
    assert f["version"] == 1 and f["draft"] is None


# ── trust ──────────────────────────────────────────────────────────────

def test_who_moves_trust(store):
    f = _live(store)
    with pytest.raises(fm.FirstMateError, match="only lowers"):
        store.set_trust(f["id"], "commit", "deck")
    f = store.set_trust(f["id"], "commit", "owner", pin=True)
    assert f["trust"] == "commit" and f["trust_pinned"]
    with pytest.raises(fm.FirstMateError, match="pinned"):
        store.set_trust(f["id"], "direct", "record")
    with pytest.raises(fm.FirstMateError, match="only the owner pins"):
        store.set_trust(f["id"], "commit", "record", pin=False)
    # A proof doubt lowers it and unpins it.
    f = store.set_trust(f["id"], "propose", "deck")
    assert f["trust"] == "propose" and not f["trust_pinned"]
    assert store.set_trust(f["id"], "commit", "record")["trust"] == "commit"
    with pytest.raises(fm.FirstMateError, match="trust must be"):
        store.set_trust(f["id"], "yolo", "owner")


def test_no_direct_trust_with_outward_grants(store):
    f = _live(store, grants={"email": True})
    with pytest.raises(fm.FirstMateError, match="direct isn't allowed"):
        store.set_trust(f["id"], "direct", "owner")
    g = _live(store, grants={"organize": True})
    assert store.set_trust(g["id"], "direct", "owner")["trust"] == "direct"


def test_a_version_that_adds_a_grant_resets_trust(store):
    f = _live(store)
    store.set_trust(f["id"], "direct", "owner", pin=True)
    f = store.new_draft(f["id"], "first_mate", grants={"email": True})
    f = store.approve(f["id"], 2, f["draft"]["sha"], BOB)
    assert f["trust"] == "propose" and not f["trust_pinned"]
    assert store.events(f["id"])[0]["detail"]["grants_added"] == ["email"]
    # A version that changes nothing about grants keeps trust.
    store.set_trust(f["id"], "commit", "owner")
    f = store.new_draft(f["id"], "first_mate", charter="Draft replies, briefly.")
    assert store.approve(f["id"], 3, f["draft"]["sha"], BOB)["trust"] == "commit"


# ── runs ───────────────────────────────────────────────────────────────

def test_runs_only_after_approval_and_one_at_a_time(store):
    f = _make(store)
    with pytest.raises(fm.FirstMateError, match="approved"):
        store.start_run(f["id"])
    f = _live(store)
    run = store.start_run(f["id"], trigger="manual")
    assert run["status"] == "queued" and run["version"] == 1
    assert run["version_sha"] == f["live"]["sha"] and run["runner"] == "funnel"
    with pytest.raises(fm.FirstMateError, match="already has a run going"):
        store.start_run(f["id"])
    store.mark_running(run["id"], "bat_abc123")
    with pytest.raises(fm.FirstMateError, match="already has a run going"):
        store.start_run(f["id"])
    store.finish_run(run["id"], "done", verdict="pass", result="3 drafts")
    assert store.start_run(f["id"])["status"] == "queued"


def test_which_states_start_which_runs(store):
    f = _live(store)
    store.set_state(f["id"], "paused", "owner")
    with pytest.raises(fm.FirstMateError, match="on its own"):
        store.start_run(f["id"], trigger="schedule")
    run = store.start_run(f["id"], trigger="manual")  # the owner trying it
    store.finish_run(run["id"], "cancelled")
    with pytest.raises(fm.FirstMateError, match="can't become"):
        store.set_state(f["id"], "paused_budget", "deck")  # the owner's pause comes first
    store.set_state(f["id"], "running", "owner")
    store.set_state(f["id"], "paused_budget", "deck")
    with pytest.raises(fm.FirstMateError, match="paused budget"):
        store.start_run(f["id"], trigger="manual")
    with pytest.raises(fm.FirstMateError, match="trigger must be"):
        store.start_run(f["id"], trigger="cron")


def test_runs_per_day(store):
    store.set_settings({"runs_per_day": 2})
    f = _live(store)
    now = fm._utcnow()
    for _ in range(2):
        r = store.start_run(f["id"], now=now)
        store.finish_run(r["id"], "done")
    with pytest.raises(fm.FirstMateError, match="at most 2"):
        store.start_run(f["id"], now=now)
    assert store.start_run(f["id"], now=now + timedelta(days=1, minutes=1))["status"] == "queued"


def test_run_totals_only_grow_and_a_run_ends_once(store):
    f = _live(store)
    run = store.start_run(f["id"])
    with pytest.raises(fm.FirstMateError, match="child_id is required"):
        store.mark_running(run["id"], "")
    store.mark_running(run["id"], "bat_1")
    with pytest.raises(fm.FirstMateError, match="already running"):
        store.mark_running(run["id"], "bat_2")
    store.update_run(run["id"], tokens=500, usd=0.1)
    assert store.update_run(run["id"], tokens=200)["tokens"] == 500
    done = store.finish_run(run["id"], "failed", error="judge said no", tokens=900)
    assert (done["status"], done["tokens"], done["usd"]) == ("failed", 900, 0.1)
    again = store.finish_run(run["id"], "done", tokens=5000)
    assert again["status"] == "failed" and again["tokens"] == 900
    assert store.update_run(run["id"], tokens=9999)["tokens"] == 900
    with pytest.raises(fm.FirstMateError, match="ends as one of"):
        store.finish_run(run["id"], "running")
    assert store.tokens_since(f["id"], fm._utcnow() - timedelta(hours=1)) == 900
    assert store.active_runs(BOB) == []


def test_active_runs(store):
    a, b = _live(store), _live(store, CAROL)
    ra, rb = store.start_run(a["id"]), store.start_run(b["id"])
    assert {r["id"] for r in store.active_runs()} == {ra["id"], rb["id"]}
    assert [r["id"] for r in store.active_runs(CAROL)] == [rb["id"]]


# ── intents and signals ────────────────────────────────────────────────

def test_intents(store):
    f = _live(store, grants={"email": True})
    run = store.start_run(f["id"], kind="rehearsal")
    i1 = store.add_intent(run["id"], kind="mail.send", tool="google_mail", what="reply to a thread",
                          args={"thread": "t1"})
    i2 = store.add_intent(run["id"], kind="mail.send", what="reply to another")
    assert store.decide_intents(run["id"], "carried_out") == 0  # nothing approved yet
    assert store.decide_intents(run["id"], "approved", ids=[i1]) == 1
    assert store.decide_intents(run["id"], "rejected") == 1  # only i2 was still open
    assert store.decide_intents(run["id"], "carried_out") == 1
    by_id = {i["id"]: i for i in store.intents(run["id"])}
    assert by_id[i1]["status"] == "carried_out" and by_id[i1]["args"] == {"thread": "t1"}
    assert by_id[i2]["status"] == "rejected"
    with pytest.raises(fm.FirstMateError, match="no such run"):
        store.add_intent("fmr_nope", kind="mail.send")


def test_signals_and_satisfaction(store):
    f = _live(store)
    assert store.satisfaction(f["id"]) is None
    store.add_signal(f["id"], "sent_as_written")
    store.add_signal(f["id"], "rejected")
    store.add_signal(f["id"], "sent_edited", value=0.7)
    assert store.satisfaction(f["id"]) == pytest.approx(1.7 / 3)
    assert store.satisfaction(f["id"], last=1) == pytest.approx(0.7)
    with pytest.raises(fm.FirstMateError, match="needs a value"):
        store.add_signal(f["id"], "sent_edited")
    with pytest.raises(fm.FirstMateError, match="signal is one of"):
        store.add_signal(f["id"], "ignored")
    with pytest.raises(fm.FirstMateError, match="value"):
        store.add_signal(f["id"], "sent_edited", value=2)


# ── settings ───────────────────────────────────────────────────────────

def test_settings_and_prefs(store):
    assert store.settings() == fm.DEFAULT_DECK and store.prefs(BOB) == fm.DEFAULT_PREFS
    assert store.set_settings({"enabled": True})["enabled"] is True
    assert store.settings()["admins_only"] is True
    p = store.set_prefs(BOB, {"on": True, "agent": "first-mate"})
    assert p["on"] and p["agent"] == "first-mate" and store.prefs(CAROL) == fm.DEFAULT_PREFS
    for bad in ({"weekly_tokens": 10}, {"agent": "no spaces please"}, {"colour": "red"}):
        with pytest.raises(fm.FirstMateError):
            store.set_prefs(BOB, bad)
    # A damaged saved value falls back to the defaults.
    store._put_setting("deck", {"max_factories": "lots"})
    assert store.settings() == fm.DEFAULT_DECK


# ── safety ─────────────────────────────────────────────────────────────

def test_a_refused_write_leaves_nothing_behind(store, monkeypatch):
    def boom(*a, **kw):
        raise fm.FirstMateError("disk said no")

    monkeypatch.setattr(store, "_insert_version", boom)
    with pytest.raises(fm.FirstMateError, match="disk said no"):
        _make(store)
    monkeypatch.undo()
    _make(store)  # the next write commits only its own factory
    assert len(store.list_factories(BOB)) == 1


def test_delete_only_drafts_and_retired_and_their_sealed_cases(store):
    f = _live(store)
    sealed = store.sealed_dir(f["id"])
    sealed.mkdir(parents=True)
    (sealed / "case.json").write_text("{}")
    with pytest.raises(fm.FirstMateError, match="owner"):
        store.delete(f["id"], CAROL)
    with pytest.raises(fm.FirstMateError, match="archive the factory before deleting it"):
        store.delete(f["id"], BOB)
    store.set_state(f["id"], "retired", "owner")
    assert store.delete(f["id"], BOB)
    assert store.get(f["id"]) is None and not sealed.exists()
    assert store.sealed_dir("../../etc").name == "etc"
    assert store.sealed_dir("../../etc").parent == store.db_path.parent / "first_mate" / "sealed"


def test_forget_user(store):
    f = _live(store)
    store.start_run(f["id"])
    store.set_prefs(BOB, {"on": True})
    store.sealed_dir(f["id"]).mkdir(parents=True)
    keep = _live(store, CAROL)
    assert fm.forget_user(BOB) == 1
    assert store.list_factories(BOB, include_retired=True) == []
    assert store.active_runs(BOB) == [] and store.prefs(BOB) == fm.DEFAULT_PREFS
    assert not store.sealed_dir(f["id"]).exists()
    assert store.get(keep["id"]) is not None


def test_forgetting_on_a_deck_without_first_mate_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "_STORE", None)
    monkeypatch.setenv("FD_DATA_DIR", str(tmp_path))
    assert fm.forget_user(BOB) == 0
    assert not (tmp_path / "first_mate.db").exists() and fm._STORE is None


def test_a_new_factory_starts_with_the_cap_picked_last(tmp_path):
    s = fm.FirstMateStore(tmp_path / "fm.db")
    first = s.create("u1", title="A", charter="Do a.")
    assert first["run_tokens"] == 0  # nothing picked yet: the owner's default
    s.update_factory(first["id"], "owner", run_tokens=fm.UNLIMITED)
    assert s.create("u1", title="B", charter="Do b.")["run_tokens"] == fm.UNLIMITED
    s.update_factory(first["id"], "owner", run_tokens=8_000_000)
    assert s.create("u1", title="C", charter="Do c.")["run_tokens"] == 8_000_000
    assert s.create("u2", title="D", charter="Do d.")["run_tokens"] == 0  # someone else's pick isn't theirs
    s.update_factory(first["id"], "deck", run_tokens=5_000_000)  # not the owner's pick: not remembered
    assert s.create("u1", title="E", charter="Do e.")["run_tokens"] == 8_000_000
    s.update_factory(first["id"], "owner", run_tokens=0)  # back to their default
    assert s.create("u1", title="F", charter="Do f.")["run_tokens"] == 0
