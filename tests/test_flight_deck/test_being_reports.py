"""Iskra reports — the deterministic scoop and the Deep-Researcher run.

The scooper never touches a model, so it is tested against a seeded ledger;
the run is tested with an injected FAKE provider (canned markdown) so the whole
collect → research → save → row-done path is exercised without a network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_reports as R
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 22, 18, 0, tzinfo=timezone.utc)
DAY = datetime(2026, 7, 22, 9, 0, tzinfo=timezone.utc)      # a Wednesday morning
OWNER = "user-1"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    monkeypatch.delenv("CLAW_BEINGS_TZ", raising=False)      # keep periods in UTC
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


def _being(store, name="Zora", stage="child", now=DAY):
    b = store.conceive(OWNER, name, preset="explorer",
                       allowance_preset="5M", now=now)
    store.hatch(OWNER, b["slug"], now=now)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=now)
    return store.get(OWNER, b["slug"])


def _ev(store, bid, kind, data, at):
    store.record_event(bid, kind, data, now=at)


class _Resp:
    def __init__(self, content):
        self.content = content
        self.usage = {"prompt_tokens": 400, "completion_tokens": 200}


class _FakeProvider:
    """Records every call; returns a canned report with a real `#` heading."""
    def __init__(self, content="# The Village Today\nA quiet day.\n\n## Health\nAll well."):
        self.content = content
        self.calls: list = []

    async def complete(self, messages, **kw):
        self.calls.append(messages)
        return _Resp(self.content)


class _BoomProvider:
    async def complete(self, messages, **kw):
        raise RuntimeError("model unreachable")


# ═══ Period resolution ════════════════════════════════════════════════════

def test_period_windows_are_drawn_correctly():
    r = R.resolve_period
    s, e, disp = r("today", now=NOW)
    assert s.startswith("2026-07-22T00:00") and e.startswith("2026-07-22T18:00")
    assert disp == "today"
    s, e, _ = r("yesterday", now=NOW)
    assert s.startswith("2026-07-21T00:00") and e.startswith("2026-07-22T00:00")
    s, e, _ = r("this week", now=NOW)          # NOW is a Wednesday → Monday 20th
    assert s.startswith("2026-07-20T00:00")
    s, e, _ = r("last 7 days", now=NOW)         # rolling from the instant
    assert s.startswith("2026-07-15T18:00")
    s, e, _ = r("this month", now=NOW)
    assert s.startswith("2026-07-01T00:00")
    s, e, _ = r("last 30 days", now=NOW)
    assert s.startswith("2026-06-22T18:00")


def test_custom_period_and_unknown_label():
    s, e, disp = R.resolve_period("custom", now=NOW,
                                  start="2026-07-10", end="2026-07-12")
    assert s.startswith("2026-07-10T00:00")
    assert e.startswith("2026-07-13T00:00")     # end date is inclusive (+1 day)
    assert "2026-07-10" in disp
    with pytest.raises(ValueError):
        R.resolve_period("last century", now=NOW)


# ═══ events_between — the whole window, not a truncated tail ═══════════════

def test_events_between_returns_the_whole_window_untruncated(store):
    # Born an hour before the window, so its lifecycle events don't count.
    b = _being(store, now=DAY - timedelta(hours=1))
    bid = b["id"]
    # 600 events inside the window — more than the old events(limit=500) cap.
    for i in range(600):
        _ev(store, bid, "tick", {"act": "rest", "n": i},
            DAY + timedelta(seconds=i))
    # …and one clearly outside it.
    _ev(store, bid, "tick", {"act": "OLD"}, DAY - timedelta(days=5))
    got = store.events_between(OWNER, DAY.isoformat(),
                               (DAY + timedelta(hours=1)).isoformat())
    assert len(got) == 600                       # nothing dropped (no 500 cap)
    assert all(e["data"].get("act") != "OLD" for e in got)
    assert got[0]["at"] <= got[-1]["at"]         # oldest-first
    assert got[0]["slug"] == b["slug"]           # slug tagged on


def test_events_between_scopes_to_the_owner(store):
    a = _being(store, name="Ana")
    _ev(store, a["id"], "tick", {"act": "x"}, DAY)
    # a second owner's being must never leak in
    store.conceive("other", "Bex", preset="explorer", allowance_preset="5M",
                   now=DAY)
    got = store.events_between(OWNER, (DAY - timedelta(hours=1)).isoformat(),
                               (DAY + timedelta(hours=1)).isoformat())
    assert {e["slug"] for e in got} == {a["slug"]}


# ═══ The scoop ════════════════════════════════════════════════════════════

def _seed_day(store, b):
    bid = b["id"]
    _ev(store, bid, "tick", {"kind": "wake", "act": "create", "served": "grow",
        "mood_engine": "calm", "summary": "write about the Meadow as it meets me",
        "tokens_weighted": 1200}, DAY)
    _ev(store, bid, "instinct", {"act": "go", "retried": True},
        DAY + timedelta(minutes=6))
    _ev(store, bid, "departed", {"from": "home", "to": "meadow", "by": "feet"},
        DAY + timedelta(minutes=6))
    _ev(store, bid, "arrived", {"place": "meadow", "by": "feet"},
        DAY + timedelta(minutes=20))
    _ev(store, bid, "tick", {"kind": "wake", "act": "tend", "served": "survive",
        "mood_engine": "orderly", "summary": "update INDEX",
        "tokens_weighted": 900}, DAY + timedelta(minutes=65))
    _ev(store, bid, "body_rebound", {"port": 24877}, DAY + timedelta(hours=2))
    _ev(store, bid, "milestone", {"name": "first_visit_meadow"},
        DAY + timedelta(minutes=20))


def test_collect_report_data_aggregates_a_day(store):
    b = _being(store)
    _seed_day(store, b)
    # an out-of-window tick that must not appear
    _ev(store, b["id"], "tick", {"act": "OLD", "summary": "long ago"},
        DAY - timedelta(days=3))
    s, e, disp = R.resolve_period("today", now=NOW)
    data = R.collect_report_data(store, OWNER, s, e, display=disp, now=NOW)
    f = data["beings"][0]
    assert f["name"] == "Zora"
    assert f["ticks"] == 2 and f["acts"] == {"create": 1, "tend": 1}
    assert f["instinct"] == {"calls": 1, "acts": {"go": 1}, "retried": 1}
    assert f["walks"]["departures"] == 1
    assert f["health"] == {"body_rebound": 1}
    assert f["milestones"] == ["first_visit_meadow"]
    # the arrive→next tell: arrival paired with the following tick
    assert f["stays"] and f["stays"][0]["place"] == "meadow"
    assert f["stays"][0]["then"] == "tick"
    assert data["health_rollup"] == {"body_rebound": 1}
    md = R.render_data_markdown(data)
    assert "Zora" in md and "MEADOW".lower() in md.lower()
    assert "OLD" not in md and "long ago" not in md      # window respected


def test_letters_are_scooped_in_window_with_names(store):
    a = _being(store, name="Ana")
    c = _being(store, name="Cvijeta")
    store.send_letter(OWNER, a["slug"], c["slug"], "Draga Cvijeta, hvalo.",
                      now=DAY + timedelta(minutes=30))
    s, e, disp = R.resolve_period("today", now=NOW)
    data = R.collect_report_data(store, OWNER, s, e, display=disp, now=NOW)
    assert len(data["letters"]) == 1
    lt = data["letters"][0]
    assert lt["from"] == "Ana" and lt["to"] == "Cvijeta"
    assert "hvalo" in lt["body"]
    assert "hvalo" in R.render_data_markdown(data)


# ═══ run_report — the whole path with a fake model ════════════════════════

async def test_run_report_quick_end_to_end(store, tmp_path):
    b = _being(store)
    _seed_day(store, b)
    s, e, disp = R.resolve_period("today", now=NOW)
    rep = store.create_report(OWNER, label=disp, period_start=s, period_end=e,
                              depth="quick", now=NOW)
    fp = _FakeProvider()
    await R.run_report(store, OWNER, rep["id"], provider=fp,
                       tier_name="reason", now=NOW)
    got = store.get_report(OWNER, rep["id"])
    assert got["status"] == "done"
    assert got["title"] == "The Village Today"
    assert got["tokens"] == 600                  # one call, 400+200
    assert got["report_md"].startswith("# The Village Today")
    assert len(fp.calls) == 1                     # quick = single pass
    # the researcher was handed the real scooped blob, not an empty prompt
    user_msg = fp.calls[0][1].content
    assert "Zora" in user_msg and "meadow" in user_msg.lower()
    # both artefacts landed in the owner's VFS
    root = Path(tmp_path) / "vfs" / OWNER / R.REPORTS_PROJECT
    files = sorted(p.name for p in root.glob("*.md"))
    assert len(files) == 2 and any(f.endswith(".data.md") for f in files)
    assert got["vfs_path"].startswith("vfs:iskra-reports/")
    assert [p["msg"] for p in got["progress"]][-1] == "report ready"


async def test_run_report_deep_fans_out_then_synthesises(store):
    b = _being(store)
    _seed_day(store, b)
    s, e, disp = R.resolve_period("today", now=NOW)
    rep = store.create_report(OWNER, label=disp, period_start=s, period_end=e,
                              depth="deep", now=NOW)
    fp = _FakeProvider()
    await R.run_report(store, OWNER, rep["id"], provider=fp,
                       tier_name="reason", now=NOW)
    got = store.get_report(OWNER, rep["id"])
    assert got["status"] == "done"
    assert len(fp.calls) == 4                     # 3 facets + 1 synthesiser
    assert got["tokens"] == 4 * 600
    # the synthesiser saw the three briefings folded into its user prompt
    synth_user = fp.calls[-1][1].content
    assert "SPECIALIST BRIEFINGS" in synth_user
    assert "Behaviour briefing" in synth_user


async def test_a_failed_run_lands_on_the_row_not_a_crash(store):
    b = _being(store)
    _seed_day(store, b)
    s, e, disp = R.resolve_period("today", now=NOW)
    rep = store.create_report(OWNER, label=disp, period_start=s, period_end=e,
                              depth="quick", now=NOW)
    await R.run_report(store, OWNER, rep["id"], provider=_BoomProvider(),
                       tier_name="reason", now=NOW)          # must not raise
    got = store.get_report(OWNER, rep["id"])
    assert got["status"] == "failed"
    assert "model unreachable" in got["error"]
    assert any("failed" in p["msg"] for p in got["progress"])


# ═══ Store CRUD + tenancy ═════════════════════════════════════════════════

def test_reports_are_owner_scoped(store):
    r = store.create_report(OWNER, label="today", period_start=DAY.isoformat(),
                            period_end=NOW.isoformat(), now=NOW)
    assert store.list_reports(OWNER) and not store.list_reports("intruder")
    with pytest.raises(BeingError):
        store.get_report("intruder", r["id"])
    with pytest.raises(BeingError):
        store.delete_report("intruder", r["id"])
    assert store.delete_report(OWNER, r["id"])["ok"]
    assert not store.list_reports(OWNER)
