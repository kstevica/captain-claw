"""A Vatra run must ALWAYS reach a terminal status.

Regression for the glasses "Story run 5" incident: a story-integrity run crashed with
an unhandled exception AFTER the reporter finished; execute_vatra had no top-level
handler and ran as a fire-and-forget task whose exception was dropped on GC, so the
session sat on status='running' forever with the UI spinning and no agent running. The
guard wrapper must catch the crash, log it, and persist status='error' (continuable /
resumable) — while letting HTTPException and CancelledError propagate unchanged.
"""

import asyncio
import json
import types

import pytest

from fastapi import HTTPException

from captain_claw.flight_deck import vatra_routes as vr


class _FakeDB:
    def __init__(self, session=None):
        self._session = session if session is not None else {"status": "running"}
        self.updates = []

    async def get_basna_session(self, sid, uid):
        return dict(self._session)

    async def update_basna_session(self, sid, uid, **kwargs):
        self.updates.append(kwargs)


def _body(sid="sid1"):
    return types.SimpleNamespace(session_id=sid)


@pytest.mark.asyncio
async def test_crash_persists_error_not_stuck_running(monkeypatch):
    monkeypatch.setattr(vr, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(vr, "_progress_done", lambda *a, **k: None)
    db = _FakeDB()
    monkeypatch.setattr(vr, "get_db", lambda: db)

    async def _boom(body, request, user):
        raise ValueError("kaboom in the post-reporter path")

    monkeypatch.setattr(vr, "_execute_vatra_inner", _boom)

    result = await vr.execute_vatra(_body(), None, {"id": "u1"})

    # Returned a terminal error dict rather than raising / hanging.
    assert result["status"] == "error"
    assert result["error"]["type"] == "ValueError"

    # Persisted status='error' with the traceback captured in analysis.
    assert db.updates, "expected the crash to be persisted"
    up = db.updates[-1]
    assert up["status"] == "error"
    persisted = json.loads(up["analysis"])
    assert persisted["error"]["type"] == "ValueError"
    assert "kaboom" in persisted["error"]["message"]
    assert "Traceback" in persisted["error"]["traceback"]


@pytest.mark.asyncio
async def test_crash_persist_preserves_prior_analysis(monkeypatch):
    monkeypatch.setattr(vr, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(vr, "_progress_done", lambda *a, **k: None)
    db = _FakeDB(session={"status": "running",
                          "analysis": json.dumps({"quality_metrics": {"acted": 3}})})
    monkeypatch.setattr(vr, "get_db", lambda: db)

    async def _boom(body, request, user):
        raise RuntimeError("late crash")

    monkeypatch.setattr(vr, "_execute_vatra_inner", _boom)
    result = await vr.execute_vatra(_body(), None, {"id": "u1"})
    assert result["status"] == "error"
    persisted = json.loads(db.updates[-1]["analysis"])
    # existing analysis kept, error appended
    assert persisted["quality_metrics"]["acted"] == 3
    assert persisted["error"]["type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_http_exception_propagates(monkeypatch):
    monkeypatch.setattr(vr, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(vr, "_progress_done", lambda *a, **k: None)
    db = _FakeDB()
    monkeypatch.setattr(vr, "get_db", lambda: db)

    async def _raise_http(body, request, user):
        raise HTTPException(404, "session not found")

    monkeypatch.setattr(vr, "_execute_vatra_inner", _raise_http)
    with pytest.raises(HTTPException):
        await vr.execute_vatra(_body(), None, {"id": "u1"})
    # a request-path 4xx must NOT be turned into a persisted 'error' run
    assert not db.updates


@pytest.mark.asyncio
async def test_cancellation_propagates(monkeypatch):
    monkeypatch.setattr(vr, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(vr, "_progress_done", lambda *a, **k: None)
    db = _FakeDB()
    monkeypatch.setattr(vr, "get_db", lambda: db)

    async def _cancel(body, request, user):
        raise asyncio.CancelledError()

    monkeypatch.setattr(vr, "_execute_vatra_inner", _cancel)
    with pytest.raises(asyncio.CancelledError):
        await vr.execute_vatra(_body(), None, {"id": "u1"})


@pytest.mark.asyncio
async def test_crash_persist_never_reraises_even_if_db_fails(monkeypatch):
    # Even if persistence itself fails, the guard must return an error dict (the task
    # ends cleanly rather than dropping an exception on GC).
    monkeypatch.setattr(vr, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(vr, "_progress_done", lambda *a, **k: None)

    class _BadDB:
        async def get_basna_session(self, sid, uid):
            raise OSError("db unavailable")

        async def update_basna_session(self, sid, uid, **kwargs):
            raise OSError("db unavailable")

    monkeypatch.setattr(vr, "get_db", lambda: _BadDB())

    async def _boom(body, request, user):
        raise ValueError("boom")

    monkeypatch.setattr(vr, "_execute_vatra_inner", _boom)
    result = await vr.execute_vatra(_body(), None, {"id": "u1"})
    assert result["status"] == "error"
