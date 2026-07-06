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
