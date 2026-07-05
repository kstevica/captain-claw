"""Tests for R1 — the Research Map (prose FTS index over a shared VFS folder)."""

from __future__ import annotations

from captain_claw.flight_deck import research_map as rm


def _write(d, name, text):
    (d / name).write_text(text)


def test_empty_folder_has_no_preamble(tmp_path):
    assert rm.preamble(tmp_path) == ""
    assert rm.stats(tmp_path) == {"files": 0, "chunks": 0}


def test_index_and_search(tmp_path):
    _write(tmp_path, "findings.md",
           "# Market size\nThe TAM is $4B growing 20% YoY.\n\n"
           "# Competitors\nAcme and Globex dominate the enterprise tier.")
    _write(tmp_path, "sources.md", "# Sources\n- Gartner 2025 report on the widget market.")
    res = rm.reindex(tmp_path)
    assert res["files"] == 2
    assert res["chunks"] >= 3

    hits = rm.search(tmp_path, "competitors enterprise")
    assert any(h["path"] == "findings.md" for h in hits)
    hits2 = rm.search(tmp_path, "Gartner")
    assert any(h["path"] == "sources.md" for h in hits2)


def test_incremental_reindex_only_touches_changed(tmp_path):
    _write(tmp_path, "a.md", "# A\nalpha content here")
    rm.reindex(tmp_path)
    # No change → nothing re-chunked.
    res = rm.reindex(tmp_path)
    assert res["changed"] == []
    # Change one file → only it re-chunks.
    _write(tmp_path, "a.md", "# A\nalpha content here, now updated with beta")
    res2 = rm.reindex(tmp_path)
    assert res2["changed"] == ["a.md"]


def test_deleted_file_drops_from_index(tmp_path):
    _write(tmp_path, "gone.md", "# Gone\nsoon to vanish")
    rm.reindex(tmp_path)
    assert rm.search(tmp_path, "vanish")
    (tmp_path / "gone.md").unlink()
    rm.reindex(tmp_path)
    assert rm.search(tmp_path, "vanish") == []


def test_preamble_present_once_indexed(tmp_path):
    _write(tmp_path, "x.md", "# Topic\nsome established finding")
    rm.reindex(tmp_path)
    pre = rm.preamble(tmp_path)
    assert "Research Map available" in pre
    rm.write_overview(tmp_path, "We established that X leads to Y.")
    pre2 = rm.preamble(tmp_path)
    assert "established that X" in pre2


def test_skip_dirs_are_ignored(tmp_path):
    (tmp_path / ".researchmap").mkdir()
    _write(tmp_path / ".researchmap", "map.md", "should not be indexed")
    _write(tmp_path, "real.md", "# Real\nindex me")
    rm.reindex(tmp_path)
    assert rm.search(tmp_path, "index")
    assert rm.search(tmp_path, "should not be indexed") == []
