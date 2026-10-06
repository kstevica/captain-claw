"""vfs.resolve_under re-checks a project's link target at read time.

The link registry (.vfs-links.json) is a plain file in the user's VFS root, so
one edited on disk could point a project at Flight Deck's data dir (every
user's tokens and DBs). Server-side callers (Basna, Vatra, beings) resolve
through resolve_under; a link into the data dir — by any spelling — must
behave like a missing link, while ordinary external links keep working.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from captain_claw import vfs


@pytest.fixture
def deck(tmp_path, monkeypatch):
    data = tmp_path / "fd-data"
    root = data / "vfs" / "u1"
    root.mkdir(parents=True)
    (data / "flightdeck.db").write_text("secret")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "notes.md").write_text("hello")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAW_VFS_ROOT", str(data / "vfs"))
    return data, root, outside


def _links(root: Path, **targets: str) -> None:
    (root / ".vfs-links.json").write_text(
        json.dumps({k: {"path": v, "mode": "ro"} for k, v in targets.items()}))


def test_link_into_the_data_dir_is_ignored(deck):
    data, root, outside = deck
    _links(root, evil=str(data), up=str(data.parent), ok=str(outside))
    assert vfs.resolve_under("u1", "evil", "evil/flightdeck.db") == (root / "evil" / "flightdeck.db").resolve()
    assert vfs.resolve_under("u1", "up", "up/fd-data/flightdeck.db") == (root / "up" / "fd-data" / "flightdeck.db").resolve()
    assert vfs.resolve_under("u1", "ok", "ok/notes.md") == (outside / "notes.md").resolve()


def test_another_spelling_of_the_data_dir_is_ignored(deck):
    data, root, _ = deck
    upper = data.parent / data.name.upper()
    if not upper.exists():  # case-sensitive filesystem: nothing to prove here
        pytest.skip("case-sensitive filesystem")
    _links(root, evil=str(upper))
    assert vfs.resolve_under("u1", "evil", "evil/flightdeck.db") == (root / "evil" / "flightdeck.db").resolve()


def test_a_link_inside_the_users_own_root_is_kept(deck):
    _, root, _ = deck
    mount = root / ".drive" / "docs"
    mount.mkdir(parents=True)
    _links(root, docs=str(mount))
    assert vfs.safe_link_target_at(root, "docs") == mount.resolve()
