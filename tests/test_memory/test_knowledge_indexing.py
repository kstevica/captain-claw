"""A workspace's knowledge/ folder — the documents an owner's setup gives the
agent, possibly thousands of parts — is indexed first, under its own cap, so
other files never crowd it out; and a file already indexed and unchanged is
not read again on every sync."""

from __future__ import annotations

from pathlib import Path

from captain_claw.semantic_memory import SemanticMemoryIndex


def _index(tmp_path: Path, **kwargs) -> SemanticMemoryIndex:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    session_db = tmp_path / "sessions.db"
    session_db.touch()
    defaults = dict(db_path=tmp_path / "memory.db", session_db_path=session_db, workspace_path=workspace,
                    index_workspace=True, index_sessions=False, auto_sync_on_search=False,
                    layered_summaries=False)
    defaults.update(kwargs)
    return SemanticMemoryIndex(**defaults)


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_knowledge_comes_first_under_its_own_cap(tmp_path):
    index = _index(tmp_path, max_workspace_files=2, max_knowledge_files=10)
    try:
        ws = index.workspace_path
        for n in range(5):
            _write(ws, f"aaa/notes-{n}.md", f"note {n}")      # sorts before knowledge/
        for n in range(3):
            _write(ws, f"knowledge/manual-part-{n:02d}.md", f"part {n} of the manual")
        docs = index._collect_workspace_documents()
        refs = [d.reference for d in docs]
        assert refs[:3] == [f"knowledge/manual-part-{n:02d}.md" for n in range(3)]
        assert len(refs) == 5 and sum(r.startswith("aaa/") for r in refs) == 2   # the rest keep their own cap
        assert len(set(refs)) == len(refs)                                        # knowledge isn't walked twice
    finally:
        index.close()


def test_an_unchanged_file_is_not_read_again(tmp_path, monkeypatch):
    index = _index(tmp_path)
    try:
        ws = index.workspace_path
        _write(ws, "knowledge/a.md", "Alpha section about discount rates.")
        changed = _write(ws, "knowledge/b.md", "Beta section about growth.")
        first = {d.reference: d.signature for d in index._collect_workspace_documents()}
        changed.write_text("Beta section about terminal growth, rewritten.", encoding="utf-8")
        reads: list[str] = []
        real = Path.read_text

        def counting(self, *a, **k):
            reads.append(self.name)
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", counting)
        again = {d.reference: d for d in index._collect_workspace_documents(first)}
        assert reads == ["b.md"]
        assert again["knowledge/a.md"].text == "" and "terminal growth" in again["knowledge/b.md"].text
    finally:
        index.close()


def test_the_sync_keeps_unchanged_files_and_picks_up_changes(tmp_path):
    index = _index(tmp_path)
    try:
        ws = index.workspace_path
        _write(ws, "knowledge/a.md", "The warehouse closes at fifteen hundred on Fridays.")
        b = _write(ws, "knowledge/b.md", "Invoices are sent on the fifth of each month.")
        index._sync_once()

        def refs() -> set[str]:
            return {r[0] for r in index._conn_or_raise().execute(
                "SELECT reference FROM memory_documents WHERE source = 'workspace'").fetchall()}

        def chunks(ref: str) -> list:
            return index._conn_or_raise().execute(
                "SELECT c.text FROM memory_chunks c JOIN memory_documents d ON d.doc_id = c.doc_id "
                "WHERE d.reference = ?", (ref,)).fetchall()

        assert refs() == {"knowledge/a.md", "knowledge/b.md"}
        b.write_text("Invoices are sent on the tenth of each month.", encoding="utf-8")
        index._sync_once()
        assert refs() == {"knowledge/a.md", "knowledge/b.md"}
        assert "fifteen hundred" in chunks("knowledge/a.md")[0][0]     # kept, though it wasn't read
        assert "tenth" in chunks("knowledge/b.md")[0][0]
        b.unlink()
        index._sync_once()
        assert refs() == {"knowledge/a.md"}
    finally:
        index.close()
