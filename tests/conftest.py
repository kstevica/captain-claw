"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _record_of_written_files_in_tmp(tmp_path, monkeypatch):
    """The write tool remembers the files the agent wrote in a small database
    next to the session DB (written_by_agent): every test gets its own, so no
    test writes into a real ~/.captain-claw."""
    from captain_claw import written_by_agent

    monkeypatch.setattr(written_by_agent, "DB_PATH_OVERRIDE", tmp_path / "written_by_agent.db")
    yield
    with written_by_agent._LOCK:
        for conn in written_by_agent._CONNS.values():
            conn.close()
        written_by_agent._CONNS.clear()
