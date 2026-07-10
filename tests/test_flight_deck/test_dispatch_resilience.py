"""A dispatch that hits an agent-side error (e.g. context overflow on a local
model) must PRESERVE the work it did and surface the real reason — not raise it
all away as a mute ✗ · 0 actions."""

from __future__ import annotations

import asyncio

from captain_claw.flight_deck import basna_routes as b


def _run_dispatch(monkeypatch, fake_collect):
    monkeypatch.setattr(b, "_send_chat_and_collect", fake_collect)
    return asyncio.run(b._dispatch_one(1234, "tok", "do the thing", 60.0))


def test_agent_error_is_flagged_but_work_is_kept(monkeypatch):
    async def fake(port, token, prompt, timeout, *, error_sink=None, usage_sink=None, **kw):
        # Simulate the agent doing real work, then its LLM call erroring.
        if error_sink is not None:
            error_sink["message"] = "context length exceeded (171000 > 131072)"
        if usage_sink is not None:
            usage_sink.update({"prompt_tokens": 171000, "completion_tokens": 500, "model": "qwen"})
        return "partial findings so far", [{"tool": "web_fetch", "detail": "udruga-mi.hr"},
                                           {"tool": "shell", "detail": "curl"}]
    d = _run_dispatch(monkeypatch, fake)
    assert d["ok"] is False                       # marked failed…
    assert "context length exceeded" in d["error"]  # …with the REAL reason
    assert len(d["actions"]) == 2                  # work preserved, not 0 actions
    assert d["output"] == "partial findings so far"
    assert d["usage"]["prompt_tokens"] == 171000   # spend still counted


def test_clean_run_is_ok_with_no_error(monkeypatch):
    async def fake(port, token, prompt, timeout, *, error_sink=None, usage_sink=None, **kw):
        if usage_sink is not None:
            usage_sink.update({"prompt_tokens": 100, "completion_tokens": 50, "model": "qwen"})
        return "the answer", [{"tool": "web_fetch", "detail": "x"}]
    d = _run_dispatch(monkeypatch, fake)
    assert d["ok"] is True
    assert "error" not in d
    assert d["output"] == "the answer" and len(d["actions"]) == 1


def test_hard_exception_still_returns_a_failed_result(monkeypatch):
    async def boom(port, token, prompt, timeout, **kw):
        raise RuntimeError("socket exploded")
    d = _run_dispatch(monkeypatch, boom)
    assert d["ok"] is False and "socket exploded" in d["error"]


# ── dispatch time-budget: honest flag + bounded auto-extension ────────
# A worker cut at its (possibly extended) budget must read ⏱ in the log, not ✓,
# while its partial work stays usable (ok=True, no error).

def test_timed_out_dispatch_is_flagged_but_stays_usable(monkeypatch):
    async def fake(port, token, prompt, timeout, *, error_sink=None, usage_sink=None, **kw):
        if error_sink is not None:
            error_sink["timed_out"] = True  # budget spent mid-synthesis
        return "partial extraction notes", [{"tool": "pdf_extract", "detail": "annex.pdf"}]
    d = _run_dispatch(monkeypatch, fake)
    assert d["ok"] is True          # partial work is NOT discarded…
    assert "error" not in d
    assert d["timed_out"] is True   # …but the truth is on the record
    assert d["output"] == "partial extraction notes"


def test_extend_deadline_grants_while_active_and_under_cap():
    # now=600 (budget wall), active 10s ago, cap at 1800 → extend by a slice.
    new = b._extend_deadline(600.0, 1800.0, 590.0)
    assert new == 600.0 + b._EXTEND_SLICE_S


def test_extend_deadline_clamps_to_the_hard_ceiling():
    new = b._extend_deadline(1700.0, 1800.0, 1690.0)
    assert new == 1800.0  # slice would overshoot — clamped


def test_extend_deadline_refuses_idle_and_capped_agents():
    # Idle past the activity window → a hung agent still times out.
    assert b._extend_deadline(600.0, 1800.0, 600.0 - b._EXTEND_ACTIVITY_S - 1) is None
    # At/past the hard ceiling → no further extension, ever.
    assert b._extend_deadline(1800.0, 1800.0, 1799.0) is None
    # No ceiling armed (task never sent) → no extension.
    assert b._extend_deadline(600.0, None, 599.0) is None
