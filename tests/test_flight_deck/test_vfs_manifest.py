"""The continuation manifest must list prior DELIVERABLES only — never the .git
tree R6 creates, dotfiles/internal folders, or the whole R10 sources/ corpus."""

from __future__ import annotations

from captain_claw.flight_deck.basna_routes import _manifest_body


def _seed(root):
    # A real deliverable from a prior round.
    (root / "r1-findings.md").write_text("prior findings")
    (root / "report.md").write_text("the report")
    # .git tree (R6 git-snapshots) — the reported bug.
    (root / ".git" / "hooks").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (root / ".git" / "config").write_text("[core]")
    (root / ".git" / "hooks" / "pre-commit.sample").write_text("#!/bin/sh")
    # Other internal / noise.
    (root / ".vfs-meta.jsonl").write_text("{}")
    (root / ".code").mkdir()
    (root / ".code" / "state.json").write_text("{}")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("//")
    # R10 corpus — many saved pages.
    (root / "sources").mkdir()
    for i in range(12):
        (root / "sources" / f"page-{i}.md").write_text("src")


def test_git_and_internal_dirs_are_hidden(tmp_path):
    _seed(tmp_path)
    body = _manifest_body(tmp_path, "proj")
    assert ".git" not in body
    assert "HEAD" not in body and "pre-commit" not in body and "config" not in body
    assert ".vfs-meta" not in body
    assert ".code" not in body and "state.json" not in body
    assert "node_modules" not in body


def test_real_deliverables_are_listed(tmp_path):
    _seed(tmp_path)
    body = _manifest_body(tmp_path, "proj")
    assert "vfs:proj/r1-findings.md" in body
    assert "vfs:proj/report.md" in body


def test_sources_corpus_is_collapsed_not_itemised(tmp_path):
    _seed(tmp_path)
    body = _manifest_body(tmp_path, "proj")
    # Not one line per page…
    assert "page-0.md" not in body and "page-11.md" not in body
    # …but a single pointer that names the count + how to reach them.
    assert "12 saved source page(s)" in body
    assert "researchmap" in body


def test_empty_or_only_noise_folder_yields_nothing(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("x")
    (tmp_path / ".vfs-meta.jsonl").write_text("{}")
    assert _manifest_body(tmp_path, "proj") == ""


def test_only_sources_still_points_to_them(tmp_path):
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "a.md").write_text("x")
    body = _manifest_body(tmp_path, "proj")
    assert "1 saved source page(s)" in body
    assert "already holds the prior" in body
