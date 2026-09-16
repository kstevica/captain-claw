"""Increment 4: the done-gate persists status='error' with truth kept."""

import json

import pytest

from captain_claw.flight_deck import vatra_routes as vr


class _FakeDB:
    def __init__(self):
        self.updates = []

    async def update_basna_session(self, sid, uid, **kwargs):
        self.updates.append(kwargs)


@pytest.mark.asyncio
async def test_finish_blocked_persists_error_with_truth(monkeypatch):
    monkeypatch.setattr(vr, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(vr, "_progress_done", lambda *a, **k: None)
    db = _FakeDB()
    truth = "# Best assembly so far\n\nsome content"
    files = [{"name": "part-one.md", "kind": "generated"}]
    analysis = {"deliverable": {"verdict": "failed", "reasons": ["deliverable_missing"]},
                "quality_verdict": "deliverable_missing"}
    result = await vr._finish_blocked(
        db, "sid1", {"id": "u1"}, reason="part-two.md was never written",
        truth=truth, files=files, analysis=analysis, confidence=0.5, domain="fiction")
    # returned dict is error-shaped but keeps truth
    assert result["status"] == "error"
    assert result["truth"] == truth
    assert result["blocked"] == "part-two.md was never written"
    # persisted status='error' with truth + analysis kept
    assert db.updates, "expected a persist"
    up = db.updates[-1]
    assert up["status"] == "error"
    assert up["truth"] == truth
    persisted = json.loads(up["analysis"])
    assert persisted["quality_verdict"] == "deliverable_missing"


def test_gate_analysis_shape():
    # A passing gate over a real deliverable → verdict ok; a broken one → failed.
    from captain_claw.flight_deck import deliverable_manifest as dm
    subs = [{"id": "s4", "depends_on": []}, {"id": "s5", "depends_on": ["s4"]}]
    m = dm.parse({"path": "book.md", "min_bytes": 100, "parts": [
        {"path": "p1.md", "owner": "s4"}, {"path": "p2.md", "owner": "s5"}]}, subs, "proj")
    good = "# Chapter One\n\n" + "x" * 200 + "\n\n# Chapter Two\n\n" + "y" * 200
    g = dm.gate(m, good)
    assert g["ok"] is True
    bad = dm.gate(m, "")
    assert bad["ok"] is False and "deliverable_missing" in bad["reasons"]
