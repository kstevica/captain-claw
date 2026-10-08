"""Semantic-memory scoring: graded keyword scores, the passive note leaving
out the live session, and relevance floors that age doesn't undercut."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from captain_claw.semantic_memory import SemanticMemoryIndex


def _index(tmp_path: Path, **kwargs) -> SemanticMemoryIndex:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    session_db = tmp_path / "sessions.db"
    session_db.touch()
    defaults = dict(
        db_path=tmp_path / "memory.db",
        session_db_path=session_db,
        workspace_path=workspace,
        index_workspace=False,
        index_sessions=False,
        auto_sync_on_search=False,
        layered_summaries=False,
    )
    defaults.update(kwargs)
    return SemanticMemoryIndex(**defaults)


def test_keyword_scores_are_graded_not_constant(tmp_path: Path):
    index = _index(tmp_path)
    try:
        index.upsert_text(source="workspace", reference="a.md", path="a.md",
                          text="Kubernetes cluster autoscaling latency regression after the upgrade.")
        index.upsert_text(source="workspace", reference="b.md", path="b.md",
                          text="The cluster of flowers in the garden bloomed early.")
        index.upsert_text(source="workspace", reference="c.md", path="c.md",
                          text="Weekly grocery list: apples, bread, coffee.")
        hits = index._keyword_search(
            "kubernetes cluster autoscaling latency", limit=10,
            active_session_reference=None, include_all_sessions=False,
        )
    finally:
        index.close()

    scores = {hit["path"]: hit["text_score"] for hit in hits}
    assert set(scores) == {"a.md", "b.md"}
    assert 0.0 < scores["b.md"] < scores["a.md"] < 1.0
    assert scores["b.md"] < 0.5


def test_passive_note_leaves_out_the_live_session_even_across_sessions(tmp_path: Path):
    for cross in (False, True):
        index = _index(tmp_path / f"cross-{cross}", cross_session_retrieval=cross,
                       min_score=0.0, temporal_decay_enabled=False)
        try:
            index.upsert_text(source="session", reference="live", path="sessions/live.txt",
                              text="Quarterly revenue forecast discussion for the board.")
            index.upsert_text(source="session", reference="other", path="sessions/other.txt",
                              text="Quarterly revenue forecast numbers from last spring.")
            index.upsert_text(source="workspace", reference="notes.md", path="notes.md",
                              text="Revenue forecast template and quarterly checklist.")
            index.set_active_session("live")

            excluded = index.search("quarterly revenue forecast", max_results=6,
                                    exclude_active_session=True)
            included = index.search("quarterly revenue forecast", max_results=6)
        finally:
            index.close()

        assert excluded
        assert not any(r.source == "session" and r.reference == "live" for r in excluded), cross
        assert any(r.reference == "notes.md" for r in excluded)
        assert any(r.reference == "other" for r in excluded) is cross
        # The two searches don't share a cache entry.
        assert any(r.source == "session" and r.reference == "live" for r in included), cross


def test_old_but_relevant_hits_pass_the_floors(tmp_path: Path):
    index = _index(tmp_path, min_score=0.1, history_min_score=0.35,
                   temporal_decay_enabled=True, temporal_half_life_days=21.0)
    old = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    try:
        merged = index._merge_hybrid(
            [{"chunk_id": "h1", "source": "session_history", "reference": "hist", "path": "h",
              "start_line": 1, "end_line": 1, "snippet": "old but on point", "updated_at": old,
              "text_score": 0.9}],
            [{"chunk_id": "h1", "source": "session_history", "reference": "hist", "path": "h",
              "start_line": 1, "end_line": 1, "snippet": "old but on point", "updated_at": old,
              "vector_score": 0.8}],
            max_results=5,
        )
    finally:
        index.close()

    assert len(merged) == 1
    hit = merged[0]
    assert hit.relevance >= 0.35          # passes the history floor on relevance…
    assert hit.score < hit.relevance       # …while age still lowers its rank score
    assert hit.score < 0.35
