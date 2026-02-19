from pathlib import Path

from captain_claw.config import get_config
from captain_claw.skills import (
    SOURCE_PLUGIN,
    build_workspace_skill_snapshot,
    install_skill_dependencies,
    load_workspace_skill_entries,
)


def _write_skill(
    base_dir: Path,
    name: str,
    description: str = "demo",
    extra_frontmatter: str = "",
    body: str = "# Demo\n",
) -> Path:
    skill_dir = base_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter_lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
    ]
    if extra_frontmatter.strip():
        frontmatter_lines.extend(extra_frontmatter.strip().splitlines())
    frontmatter_lines.append("---")
    frontmatter = "\n".join(frontmatter_lines)
    (skill_dir / "SKILL.md").write_text(f"{frontmatter}\n\n{body}", encoding="utf-8")
    return skill_dir


def test_load_workspace_skill_entries_parses_nested_vendor_metadata(tmp_path: Path):
    cfg = get_config().model_copy(deep=True)
    cfg.skills.managed_dir = str(tmp_path / "managed")

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    _write_skill(
        workspace / "skills",
        "event-planner",
        extra_frontmatter=(
            'metadata: {"clawdbot":{"requires":{"bins":["uv"],"env":["GOOGLE_PLACES_API_KEY"]},'
            '"install":[{"id":"uv-brew","kind":"brew","formula":"uv"}]}}'
        ),
    )

    entries = load_workspace_skill_entries(workspace, cfg)
    event_planner = next((entry for entry in entries if entry.name == "event-planner"), None)
    assert event_planner is not None
    assert event_planner.metadata is not None
    assert event_planner.metadata.requires is not None
    assert event_planner.metadata.requires.bins == ["uv"]
    assert event_planner.metadata.install
    assert event_planner.metadata.install[0].kind == "brew"
    assert event_planner.metadata.install[0].id == "uv-brew"


def test_load_workspace_skill_entries_includes_plugin_manifest_skill_dirs(tmp_path: Path):
    cfg = get_config().model_copy(deep=True)
    cfg.skills.managed_dir = str(tmp_path / "managed")

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    plugin_root = workspace / "plugins" / "demo-plugin"
    plugin_root.mkdir(parents=True, exist_ok=True)
    (plugin_root / "openclaw.plugin.json").write_text(
        '{"skills":["skills"]}',
        encoding="utf-8",
    )
    _write_skill(plugin_root / "skills", "plugin-skill")

    cfg.skills.load.plugin_dirs = [str(plugin_root)]
    entries = load_workspace_skill_entries(workspace, cfg)
    plugin_entry = next((entry for entry in entries if entry.name == "plugin-skill"), None)
    assert plugin_entry is not None
    assert plugin_entry.source == SOURCE_PLUGIN


def test_build_workspace_skill_snapshot_supports_script_dispatch(tmp_path: Path):
    cfg = get_config().model_copy(deep=True)
    cfg.skills.managed_dir = str(tmp_path / "managed")

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    skill_dir = _write_skill(
        workspace / "skills",
        "scripted-skill",
        extra_frontmatter=(
            "command-dispatch: script\n"
            "command-script: scripts/run.sh\n"
        ),
    )
    script_dir = skill_dir / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    (script_dir / "run.sh").write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")

    snapshot = build_workspace_skill_snapshot(workspace, cfg)
    command = next((item for item in snapshot.commands if item.skill_name == "scripted-skill"), None)
    assert command is not None
    assert command.dispatch is not None
    assert command.dispatch.kind == "script"
    assert command.dispatch.script_path == str((script_dir / "run.sh").resolve())


def test_install_skill_dependencies_executes_declared_installer(
    tmp_path: Path,
    monkeypatch,
):
    cfg = get_config().model_copy(deep=True)
    cfg.skills.managed_dir = str(tmp_path / "managed")

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    _write_skill(
        workspace / "skills",
        "deps-skill",
        extra_frontmatter=(
            'metadata: {"openclaw":{"install":[{"id":"node","kind":"node","package":"example-cli"}]}}'
        ),
    )

    captured: dict[str, object] = {}

    def _fake_run(argv, timeout_seconds, env=None):
        captured["argv"] = list(argv)
        captured["timeout_seconds"] = timeout_seconds
        captured["env"] = env
        return 0, "ok", ""

    monkeypatch.setattr("captain_claw.skills._run_install_command", _fake_run)

    result = install_skill_dependencies(
        skill_name="deps-skill",
        workspace_dir=workspace,
        cfg=cfg,
    )

    assert result.ok is True
    assert result.skill_name == "deps-skill"
    assert result.install_id == "node"
    assert captured["argv"] == ["npm", "install", "-g", "--ignore-scripts", "example-cli"]
