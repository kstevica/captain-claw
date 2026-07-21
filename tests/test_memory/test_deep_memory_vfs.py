"""VFS → deep memory: eligibility, references, tenancy filters, opt-in registry.

Pure-unit: no Typesense, no FD server. The end-to-end path (index → search →
rewrite → delete) is exercised against a live Typesense separately.
"""

import os
import tempfile
from pathlib import Path

import pytest

from captain_claw.deep_memory import DeepMemoryIndex


@pytest.fixture()
def vfs_root(monkeypatch):
    """An isolated VFS root so tests never touch the real fd-data tree.

    ``CLAW_VFS_ROOT`` alone is NOT enough: ``deep_memory_service.user_root()``
    deliberately defers to Flight Deck's ``server.DATA_DIR`` so the freshness
    hooks can never index a different tree than the VFS panel writes to, and
    ``DATA_DIR`` is resolved at import. Patch the resolver itself, or these
    tests write into the developer's real ``fd-data/vfs/`` — which is exactly
    what happened before this fixture was tightened.
    """
    from captain_claw.flight_deck import deep_memory_service

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("CLAW_VFS_ROOT", tmp)
        monkeypatch.setattr(
            deep_memory_service, "user_root", lambda owner_id: Path(tmp) / (owner_id or "local")
        )
        yield Path(tmp)


@pytest.fixture()
def svc(vfs_root):
    from captain_claw.flight_deck import deep_memory_service

    deep_memory_service.reset_index()
    return deep_memory_service


class TestVfsReference:
    def test_reference_is_a_readable_vfs_uri(self, svc):
        """The archive key must be the exact string the read tool accepts, so a
        search hit is a pointer the agent can open."""
        assert svc.vfs_reference("notes", "sub/file.md") == "vfs:notes/sub/file.md"

    def test_leading_slash_is_normalised(self, svc):
        assert svc.vfs_reference("notes", "/file.md") == "vfs:notes/file.md"


class TestIndexable:
    def _f(self, root: Path, rel: str, body: str = "content") -> Path:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        return p

    def test_plain_text_is_indexable(self, svc, vfs_root):
        assert svc.indexable(self._f(vfs_root, "a.md"))[0] is True

    def test_unknown_binary_suffix_is_rejected(self, svc, vfs_root):
        ok, reason = svc.indexable(self._f(vfs_root, "a.bin"))
        assert ok is False and "unsupported" in reason

    def test_heavy_directories_are_skipped(self, svc, vfs_root):
        for junk in ("node_modules/a.md", ".git/a.md", "__pycache__/a.md"):
            ok, reason = svc.indexable(self._f(vfs_root, junk))
            assert ok is False, junk
            assert "excluded" in reason

    def test_hidden_files_are_skipped(self, svc, vfs_root):
        assert svc.indexable(self._f(vfs_root, ".secret.md"))[0] is False

    def test_empty_file_is_skipped(self, svc, vfs_root):
        ok, reason = svc.indexable(self._f(vfs_root, "empty.md", ""))
        assert ok is False and reason == "empty"

    def test_oversized_file_is_skipped(self, svc, vfs_root):
        big = self._f(vfs_root, "big.md", "x" * (svc._MAX_BYTES + 1))
        ok, reason = svc.indexable(big)
        assert ok is False and "too large" in reason

    def test_directory_is_not_a_file(self, svc, vfs_root):
        d = vfs_root / "adir"
        d.mkdir()
        assert svc.indexable(d)[0] is False


class TestOptInRegistry:
    def test_indexing_is_off_until_asked_for(self, svc):
        assert svc.indexing_enabled("alice", "notes") is False

    def test_toggle_round_trips(self, svc):
        svc.set_indexing("alice", "notes", True)
        assert svc.indexing_enabled("alice", "notes") is True
        svc.set_indexing("alice", "notes", False)
        assert svc.indexing_enabled("alice", "notes") is False

    def test_registries_are_per_owner(self, svc):
        svc.set_indexing("alice", "notes", True)
        assert svc.indexing_enabled("bob", "notes") is False

    def test_missing_registry_never_raises(self, svc):
        assert svc.read_registry("nobody-at-all") == {}

    def test_hooks_no_op_while_disabled(self, svc, monkeypatch):
        """on_write must not reach the index for a project that never opted in."""
        called = []
        monkeypatch.setattr(svc, "index_file", lambda *a, **k: called.append(a))
        svc.on_write("alice", "notes", "a.md")
        assert called == []
        svc.set_indexing("alice", "notes", True)
        svc.on_write("alice", "notes", "a.md")
        assert len(called) == 1


class TestFilterEscaping:
    def test_reference_is_quoted(self):
        """A vfs: reference contains ':' and '/', both structural in Typesense's
        filter grammar — unquoted, they produce a malformed filter."""
        out = DeepMemoryIndex.escape_filter_value("vfs:notes/a b,c.md")
        assert out == "`vfs:notes/a b,c.md`"

    def test_backticks_cannot_break_out(self):
        """There is no escape for a backtick inside a backtick-quoted literal,
        so it is stripped rather than allowed to terminate the quote early."""
        assert "`" not in DeepMemoryIndex.escape_filter_value("a`b")[1:-1]


class TestOwnerScoping:
    """The tenant filter must AND onto a caller's filter, never replace it —
    otherwise a caller could widen its own scope by supplying one."""

    def _captured_filter(self, monkeypatch, **kwargs):
        idx = DeepMemoryIndex(api_key="k")
        idx._collection_ensured = True
        seen = {}

        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"results": [{"hits": []}]}

        class Client:
            def post(self, url, json=None, **_):
                seen["filter"] = json["searches"][0].get("filter_by", "")
                return Resp()

        monkeypatch.setattr(idx, "_get_client", lambda: Client())
        monkeypatch.setattr(idx, "_embed", lambda texts: [])
        idx.search("q", **kwargs)
        return seen["filter"]

    def test_owner_filter_is_applied(self, monkeypatch):
        assert self._captured_filter(monkeypatch, owner_id="alice") == "owner_id:=`alice`"

    def test_caller_filter_is_anded_not_replaced(self, monkeypatch):
        got = self._captured_filter(
            monkeypatch, owner_id="alice", filter_by="source:=vfs"
        )
        assert got == "(source:=vfs) && owner_id:=`alice`"

    def test_caller_cannot_widen_scope_with_an_or(self, monkeypatch):
        """A caller-supplied OR is parenthesised, so it cannot escape the AND."""
        got = self._captured_filter(
            monkeypatch, owner_id="alice", filter_by="owner_id:=bob || source:=vfs"
        )
        assert got.startswith("(owner_id:=bob || source:=vfs) && ")
        assert got.endswith("owner_id:=`alice`")

    def test_no_owner_leaves_the_filter_alone(self, monkeypatch):
        assert self._captured_filter(monkeypatch, filter_by="source:=vfs") == "source:=vfs"


class TestPerFileSummarisation:
    """Summarising is one LLM call per chunk — it must never happen implicitly."""

    def _index_with_summarizer(self, monkeypatch, **kwargs):
        idx = DeepMemoryIndex(api_key="k")
        idx._collection_ensured = True
        calls = []
        idx.set_summarizer(lambda text: (calls.append(text), ("l1", "l2"))[1])
        monkeypatch.setattr(idx, "_embed", lambda texts: [])
        monkeypatch.setattr(idx, "_upsert_batch", lambda docs: len(docs))
        idx.index_document("d", "line one\nline two", **kwargs)
        return calls

    def test_off_by_default(self, monkeypatch):
        assert self._index_with_summarizer(monkeypatch) == []

    def test_on_when_the_document_asks(self, monkeypatch):
        assert len(self._index_with_summarizer(monkeypatch, summarize=True)) == 1
