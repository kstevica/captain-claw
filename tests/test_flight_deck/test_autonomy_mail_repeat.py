"""A repeat refusal on the autonomous action rail is a skipped success.

The Arbiter's ``mail.draft`` runs google_mail create_draft with no LLM in the
loop, so the tool's own repeat check is what stops a second draft of an email
already in Drafts / Sent. That refusal comes back as a failed ToolResult; the
rail must ledger it as done-and-fine (the goal is met) and keep it out of
reliability learning, or every correct refusal would cost mail.draft trust.

Email actions only ever run after a human Approve (``approved_by_human``) and
never learn trust at all — not on success, not on failure (PR E, J2).
"""

from __future__ import annotations

import pytest

from captain_claw import gmail_compose
from captain_claw.config import AutonomousWorkConfig
from captain_claw.flight_deck import actions, autonomy, fd_dispatch

UID = "user-alice"

_REPEAT = gmail_compose.repeat_refusal(
    [{"kind": "draft", "draft_id": "r-1", "message_id": "m1",
      "to": "bob@x.co", "subject": "Re: Q3 report"}],
    lead="Not created",
)


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
    # Learning on, auto judge — the rail records reliability for real outcomes.
    monkeypatch.setattr(autonomy, "global_defaults", lambda: AutonomousWorkConfig().model_dump())
    return s


def _answer(monkeypatch, result: dict) -> list:
    calls: list = []

    async def fake_run_action(user_id, action_id, args, **kw):
        calls.append((user_id, action_id, args))
        return dict(result)

    monkeypatch.setattr(actions, "run_action", fake_run_action)
    return calls


def _mail_draft(store) -> dict:
    return store.add_action(
        UID, kind="tool_action", title="Draft a reply to Bob", status="approved",
        payload={"action_id": "mail.draft",
                 "args": {"to": "bob@x.co", "subject": "Re: Q3 report", "body": "Here it is"}},
    )


async def test_repeat_refusal_is_a_skipped_success_without_learning(store, monkeypatch):
    calls = _answer(monkeypatch, {"ok": False, "content": "", "error": _REPEAT})
    action = _mail_draft(store)
    out = await fd_dispatch._dispatch_tool_action(UID, action, approved_by_human=True)

    assert out["ok"] is True
    assert calls and calls[0][1] == "mail.draft"
    row = store.get_action(action["id"])
    assert row["status"] == "done" and row["outcome"] == "success"
    assert row["outcome_note"].startswith("skipped — already drafted or sent: Not created — repeat of")
    assert store.reliability_for(UID, "tool_action", "mail.draft") is None  # no trust lost
    (entry, *_) = store.list_log(UID)
    assert "skipped: repeat" in entry["event"] and entry["level"] == "info"


async def test_fd_send_repeat_409_is_skipped_too(store, monkeypatch):
    detail = gmail_compose.repeat_refusal(
        [{"kind": "sent", "message_id": "s1", "date": "2026-10-05 10:00",
          "to": "bob@x.co", "subject": "Hello"}], lead="Duplicate: not sent",
    )
    _answer(monkeypatch, {"ok": False, "error": f"Email not sent: {detail}"})
    action = store.add_action(UID, kind="tool_action", title="Send", status="approved",
                              payload={"action_id": "mail.send", "args": {}})
    await fd_dispatch._dispatch_tool_action(UID, action, approved_by_human=True)
    assert store.get_action(action["id"])["outcome"] == "success"
    assert store.reliability_for(UID, "tool_action", "mail.send") is None


async def test_a_real_failure_is_a_failure_but_never_learned_for_mail(store, monkeypatch):
    _answer(monkeypatch, {"ok": False, "error": "Google authentication expired."})
    action = _mail_draft(store)
    await fd_dispatch._dispatch_tool_action(UID, action, approved_by_human=True)

    row = store.get_action(action["id"])
    assert row["outcome"] == "fail" and row["outcome_note"] == "Google authentication expired."
    assert store.reliability_for(UID, "tool_action", "mail.draft") is None


async def test_a_created_draft_is_a_success_never_learned(store, monkeypatch):
    _answer(monkeypatch, {"ok": True, "content": "Draft created.\n  Draft ID: r-2"})
    action = _mail_draft(store)
    await fd_dispatch._dispatch_tool_action(UID, action, approved_by_human=True)

    row = store.get_action(action["id"])
    assert row["outcome"] == "success" and not row["outcome_note"].startswith("skipped")
    assert store.reliability_for(UID, "tool_action", "mail.draft") is None
