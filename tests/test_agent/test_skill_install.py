from pathlib import Path

import pytest

from captain_claw.config import get_config
from captain_claw.skills import (
    install_skill_from_github_url,
    parse_github_skill_source,
)


def test_parse_github_skill_source_supports_tree_and_blob_urls():
    tree = parse_github_skill_source(
        "https://github.com/openai/skills/tree/main/skills/.curated/source-brief"
    )
    assert tree.owner == "openai"
    assert tree.repo == "skills"
    assert tree.ref == "main"
    assert tree.skill_path == "skills/.curated/source-brief"

    blob = parse_github_skill_source(
        "https://github.com/openai/skills/blob/main/skills/.curated/source-brief/SKILL.md"
    )
    assert blob.owner == "openai"
    assert blob.repo == "skills"
    assert blob.ref == "main"
    assert blob.skill_path == "skills/.curated/source-brief"

    tree_root = parse_github_skill_source("https://github.com/openai/skills/tree/main")
    assert tree_root.owner == "openai"
    assert tree_root.repo == "skills"
    assert tree_root.ref == "main"
    assert tree_root.skill_path == ""

    tree_file = parse_github_skill_source(
        "https://github.com/openclaw/skills/tree/main/skills/udiedrichsen/event-planner/SKILL.md"
    )
    assert tree_file.owner == "openclaw"
    assert tree_file.repo == "skills"
    assert tree_file.ref == "main"
    assert tree_file.skill_path == "skills/udiedrichsen/event-planner"


def test_install_skill_from_github_url_copies_skill_into_managed_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = get_config().model_copy(deep=True)
    managed_dir = tmp_path / "managed-skills"
    cfg.skills.managed_dir = str(managed_dir)

    def _fake_download(*args, **kwargs):
        raise RuntimeError("download disabled in clone-path test")

    def _fake_clone(repo_url: str, destination: Path, ref: str | None, timeout_seconds: int) -> None:
        assert repo_url == "https://github.com/openai/skills.git"
        assert ref == "main"
        assert timeout_seconds == 120
        skill_dir = destination / "skills" / ".curated" / "source-brief"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            (
                "---\n"
                "name: source-brief\n"
                "description: Build source-grounded briefings.\n"
                "---\n\n"
                "# Source Brief\n"
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("captain_claw.skills._download_skill_from_github_api", _fake_download)
    monkeypatch.setattr("captain_claw.skills._run_git_clone", _fake_clone)

    result = install_skill_from_github_url(
        "https://github.com/openai/skills/tree/main/skills/.curated/source-brief",
        cfg,
    )

    destination = Path(result.destination)
    assert destination == managed_dir / "source-brief"
    assert (destination / "SKILL.md").is_file()
    assert result.skill_name == "source-brief"
    assert result.repo == "openai/skills"
    assert result.ref == "main"
    assert result.skill_path == "skills/.curated/source-brief"


def test_install_skill_from_github_url_fails_when_destination_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = get_config().model_copy(deep=True)
    managed_dir = tmp_path / "managed-skills"
    cfg.skills.managed_dir = str(managed_dir)
    existing = managed_dir / "source-brief"
    existing.mkdir(parents=True, exist_ok=True)

    def _fake_download(*args, **kwargs):
        raise RuntimeError("download disabled in clone-path test")

    def _fake_clone(repo_url: str, destination: Path, ref: str | None, timeout_seconds: int) -> None:
        skill_dir = destination / "skills" / ".curated" / "source-brief"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("# skill", encoding="utf-8")

    monkeypatch.setattr("captain_claw.skills._download_skill_from_github_api", _fake_download)
    monkeypatch.setattr("captain_claw.skills._run_git_clone", _fake_clone)

    with pytest.raises(FileExistsError):
        install_skill_from_github_url(
            "https://github.com/openai/skills/tree/main/skills/.curated/source-brief",
            cfg,
        )


def test_install_skill_from_github_url_downloads_tree_skill_md_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = get_config().model_copy(deep=True)
    managed_dir = tmp_path / "managed-skills"
    cfg.skills.managed_dir = str(managed_dir)

    def _fake_download(source, repo_root: Path, timeout_seconds: int) -> Path:
        assert source.owner == "openclaw"
        assert source.repo == "skills"
        assert source.ref == "main"
        assert source.skill_path == "skills/udiedrichsen/event-planner"
        assert timeout_seconds == 120
        skill_dir = repo_root / source.skill_path
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            (
                "---\n"
                "name: event-planner\n"
                "description: Build event planning timelines and checklists.\n"
                "---\n\n"
                "# Event Planner\n"
            ),
            encoding="utf-8",
        )
        (skill_dir / "README.md").write_text("# docs", encoding="utf-8")
        return skill_dir

    def _unexpected_clone(*args, **kwargs):
        raise AssertionError("git clone should not run when programmatic download succeeds")

    monkeypatch.setattr("captain_claw.skills._download_skill_from_github_api", _fake_download)
    monkeypatch.setattr("captain_claw.skills._run_git_clone", _unexpected_clone)

    result = install_skill_from_github_url(
        "https://github.com/openclaw/skills/tree/main/skills/udiedrichsen/event-planner/SKILL.md",
        cfg,
    )

    destination = Path(result.destination)
    assert destination == managed_dir / "event-planner"
    assert (destination / "SKILL.md").is_file()
    assert (destination / "README.md").is_file()
    assert result.skill_name == "event-planner"
    assert result.repo == "openclaw/skills"
    assert result.ref == "main"
    assert result.skill_path == "skills/udiedrichsen/event-planner"
