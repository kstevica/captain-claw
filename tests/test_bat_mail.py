"""Bat Phase 5 — the run-scoped email exception.

The exception is granted ONLY when the owner approved a mail plan (the Phase-4
start gate), travels ONLY on Bat's own dispatch frame, and routes through the
audited FD Gmail gate (never the uncapped send_mail path). A Bat worker with no
grant stays denied, exactly like PR #56.
"""

from __future__ import annotations

import uuid

import pytest

from captain_claw import mail_authority as ma
from captain_claw.flight_deck import bat_loop, bat_routes, human_ask
from captain_claw.flight_deck.bat_loop import BatDriver
from captain_claw.flight_deck.bat_routes import _email_receipt_block
from captain_claw.flight_deck.bat_store import BatStore


# ── mail_authority: the `bat` kind + worker default ───────────────────

def test_bat_is_a_known_automation_kind_with_a_label():
    assert "bat" in ma.AUTOMATION_KINDS
    assert ma.KIND_LABELS["bat"] == "a Bat run"
    assert set(ma.KIND_LABELS) == set(ma.AUTOMATION_KINDS)  # the pinned invariant
    assert ma.is_refusal(ma.refusal_text("bat", "sent"))    # still a valid refusal string


def test_bat_frame_allows_but_only_as_allow():
    # The per-turn frame the attempt_runner sends when the owner approved mail.
    auth = ma.from_wire({"kind": "bat", "job_text": "email the team", "mail_write": "allow"},
                        default_kind="bat")
    assert auth.mode == "automated" and auth.kind == "bat" and auth.mail_write == "allow"
    with ma.bound(auth):
        assert ma.check_mail_write("google_mail", "send") is None  # allowed

    # Deny frame (no grant) → refused, like any worker.
    deny = ma.from_wire({"kind": "bat", "job_text": "x", "mail_write": "deny"}, default_kind="bat")
    with ma.bound(deny):
        assert ma.is_refusal(ma.check_mail_write("google_mail", "send"))


def test_bat_worker_process_default_is_deny(monkeypatch):
    # A Bat worker with no bound frame falls back to the process default, which
    # is deny — the grant can only come from the explicit allow frame.
    assert "CLAW_BAT_WORKER" in ma._WORKER_ENVS
    monkeypatch.setenv("CLAW_BAT_WORKER", "1")
    for n in ("CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER", "CLAW_COUNCIL_WORKER", "CLAW_CODE_AGENT"):
        monkeypatch.delenv(n, raising=False)
    d = ma._process_default()
    assert d.mode == "automated" and d.mail_write == "deny"


# ── receipt block (Layer-3 evidence) ──────────────────────────────────

def test_email_receipt_block_reports_real_sends():
    sends = [
        {"agent": "bat-abcd1234-step-1", "to": "a@x.com", "subject": "Report", "status": "sent"},
        {"agent": "bat-abcd1234-step-2", "to": "b@x.com", "subject": "", "status": "sent"},
        {"agent": "vatra-zzzz-1", "to": "c@x.com", "subject": "other", "status": "sent"},  # not this run
    ]
    block = _email_receipt_block(sends, "abcd1234")
    assert "2 email(s) actually sent" in block
    assert "a@x.com" in block and "b@x.com" in block
    assert "c@x.com" not in block  # another run's send is excluded


def test_email_receipt_block_flags_nothing_sent():
    block = _email_receipt_block([], "abcd1234")
    assert "NO emails" in block and "unverified" in block


# ── the grant is tied to owner approval of a mail plan ─────────────────

@pytest.fixture
async def store(tmp_path):
    s = BatStore(tmp_path / "bat.db")
    await s.init()
    yield s
    await s.close()


@pytest.fixture(autouse=True)
def _wire():
    async def rec(ask):
        pass
    saved = (bat_loop._GATE_CHECK, bat_loop._PLANNER, bat_loop._JUDGE, human_ask._NOTIFY)
    bat_loop.set_gate_check(bat_routes._gate_check)
    bat_loop._PLANNER = bat_loop._default_planner
    bat_loop._JUDGE = bat_loop._default_judge
    human_ask.set_notifier(rec)
    yield
    bat_loop._GATE_CHECK, bat_loop._PLANNER, bat_loop._JUDGE, _old = saved
    human_ask.set_notifier(_old)


async def test_email_allowed_only_after_approving_a_mail_plan(store):
    rid = f"bat_{uuid.uuid4().hex[:8]}"
    await store.create_run(run_id=rid, owner_id="u1", title="t",
                           task="email the weekly report to the team",
                           config={"steps": ["draft it", "email it"]}, status="planning")
    sent = {}

    async def runner(run, step):
        sent["email_allowed"] = (run.get("config") or {}).get("email_allowed")
        return {"ok": True, "output": "ok"}

    drv = BatDriver(store, attempt_runner=runner)
    assert await drv.drive(rid) == "awaiting_plan"
    # gate recorded that mail was the reason
    assert "send email" in (await store.get_run(rid))["config"]["gate_reason"]
    # approve → email_allowed gets set, workers run with the grant
    await human_ask.answer(store, (await store.latest_open_ask(rid))["id"], "approve")
    assert await drv.drive(rid) == "done"
    assert (await store.get_run(rid))["config"]["email_allowed"] is True
    assert sent["email_allowed"] is True


async def test_non_mail_run_never_gets_email_allowed(store):
    rid = f"bat_{uuid.uuid4().hex[:8]}"
    await store.create_run(run_id=rid, owner_id="u1", title="t",
                           task="summarize the research and write it up",
                           config={"steps": ["read", "write"]}, status="planning")

    async def runner(run, step):
        assert not (run.get("config") or {}).get("email_allowed")  # never granted
        return {"ok": True, "output": "ok"}

    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "done"
    assert not (await store.get_run(rid))["config"].get("email_allowed")
