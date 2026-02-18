"""Skill discovery, filtering, prompt construction, and command invocation metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable

import yaml

from captain_claw.config import Config, SkillEntryConfig
from captain_claw.logging import get_logger

log = get_logger(__name__)

SOURCE_EXTRA = "captain-extra"
SOURCE_BUNDLED = "captain-bundled"
SOURCE_MANAGED = "captain-managed"
SOURCE_AGENTS_PERSONAL = "agents-skills-personal"
SOURCE_AGENTS_PROJECT = "agents-skills-project"
SOURCE_WORKSPACE = "captain-workspace"

SOURCE_PRECEDENCE: tuple[str, ...] = (
    SOURCE_EXTRA,
    SOURCE_BUNDLED,
    SOURCE_MANAGED,
    SOURCE_AGENTS_PERSONAL,
    SOURCE_AGENTS_PROJECT,
    SOURCE_WORKSPACE,
)
_BUNDLED_SOURCES = {SOURCE_BUNDLED}

_DEFAULT_MAX_DESCRIPTION_CHARS = 200
_SKILL_COMMAND_NAME_MAX = 32
_SKILL_COMMAND_DESC_MAX = 120


@dataclass
class SkillRequires:
    bins: list[str] = field(default_factory=list)
    any_bins: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    config: list[str] = field(default_factory=list)


@dataclass
class SkillMetadata:
    always: bool | None = None
    emoji: str | None = None
    homepage: str | None = None
    skill_key: str | None = None
    primary_env: str | None = None
    os: list[str] = field(default_factory=list)
    requires: SkillRequires | None = None


@dataclass
class SkillInvocationPolicy:
    user_invocable: bool = True
    disable_model_invocation: bool = False


@dataclass
class SkillEntry:
    name: str
    description: str
    file_path: str
    base_dir: str
    source: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    metadata: SkillMetadata | None = None
    invocation: SkillInvocationPolicy = field(default_factory=SkillInvocationPolicy)


@dataclass
class SkillCommandDispatch:
    kind: str
    tool_name: str
    arg_mode: str = "raw"


@dataclass
class SkillCommandSpec:
    name: str
    skill_name: str
    description: str
    dispatch: SkillCommandDispatch | None = None


@dataclass
class SkillSnapshotSkill:
    name: str
    skill_key: str
    primary_env: str | None = None


@dataclass
class SkillSnapshot:
    prompt: str
    skills: list[SkillSnapshotSkill]
    commands: list[SkillCommandSpec]
    version: str = ""


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _parse_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return fallback


def _parse_frontmatter(content: str) -> dict[str, Any]:
    text = (content or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    block = text[4:end]
    try:
        parsed = yaml.safe_load(block)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _parse_metadata_block(frontmatter: dict[str, Any]) -> dict[str, Any] | None:
    raw_metadata = frontmatter.get("metadata")
    metadata_obj: dict[str, Any] | None = None
    if isinstance(raw_metadata, dict):
        metadata_obj = raw_metadata
    elif isinstance(raw_metadata, str):
        parsed: Any = None
        try:
            parsed = json.loads(raw_metadata)
        except Exception:
            try:
                parsed = yaml.safe_load(raw_metadata)
            except Exception:
                parsed = None
        if isinstance(parsed, dict):
            metadata_obj = parsed
    if not metadata_obj:
        return None

    for key in ("openclaw", "captainclaw", "captain_claw"):
        candidate = metadata_obj.get(key)
        if isinstance(candidate, dict):
            return candidate
    # Support already-unwrapped block.
    return metadata_obj


def _parse_metadata(frontmatter: dict[str, Any]) -> SkillMetadata | None:
    meta = _parse_metadata_block(frontmatter)
    if not isinstance(meta, dict):
        return None
    requires_raw = meta.get("requires")
    requires: SkillRequires | None = None
    if isinstance(requires_raw, dict):
        requires = SkillRequires(
            bins=_normalize_string_list(requires_raw.get("bins")),
            any_bins=_normalize_string_list(requires_raw.get("anyBins")),
            env=_normalize_string_list(requires_raw.get("env")),
            config=_normalize_string_list(requires_raw.get("config")),
        )
    return SkillMetadata(
        always=meta.get("always") if isinstance(meta.get("always"), bool) else None,
        emoji=str(meta.get("emoji")).strip() if isinstance(meta.get("emoji"), str) else None,
        homepage=str(meta.get("homepage")).strip() if isinstance(meta.get("homepage"), str) else None,
        skill_key=str(meta.get("skillKey")).strip() if isinstance(meta.get("skillKey"), str) else None,
        primary_env=(
            str(meta.get("primaryEnv")).strip() if isinstance(meta.get("primaryEnv"), str) else None
        ),
        os=_normalize_string_list(meta.get("os")),
        requires=requires,
    )


def _resolve_invocation_policy(frontmatter: dict[str, Any]) -> SkillInvocationPolicy:
    user_invocable_raw = frontmatter.get("user-invocable", frontmatter.get("user_invocable"))
    disable_model_raw = frontmatter.get(
        "disable-model-invocation",
        frontmatter.get("disable_model_invocation"),
    )
    return SkillInvocationPolicy(
        user_invocable=_parse_bool(user_invocable_raw, True),
        disable_model_invocation=_parse_bool(disable_model_raw, False),
    )


def _resolve_bundled_skills_dir() -> Path:
    return (Path(__file__).resolve().parent / "skills").resolve()


def _resolve_skill_roots(workspace_dir: Path, cfg: Config) -> list[tuple[str, Path]]:
    managed_dir = Path(cfg.skills.managed_dir).expanduser().resolve()
    extra_roots = [
        Path(entry).expanduser().resolve()
        for entry in cfg.skills.load.extra_dirs
        if str(entry).strip()
    ]
    bundled_dir = _resolve_bundled_skills_dir()
    personal_agents = (Path.home() / ".agents" / "skills").resolve()
    project_agents = (workspace_dir / ".agents" / "skills").resolve()
    workspace_skills = (workspace_dir / "skills").resolve()

    roots: list[tuple[str, Path]] = []
    roots.extend((SOURCE_EXTRA, root) for root in extra_roots)
    roots.append((SOURCE_BUNDLED, bundled_dir))
    roots.append((SOURCE_MANAGED, managed_dir))
    roots.append((SOURCE_AGENTS_PERSONAL, personal_agents))
    roots.append((SOURCE_AGENTS_PROJECT, project_agents))
    roots.append((SOURCE_WORKSPACE, workspace_skills))
    return roots


def _iter_skill_md_candidates(root: Path, cfg: Config) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []

    candidates: list[Path] = []
    direct = root / "SKILL.md"
    if direct.is_file():
        candidates.append(direct)

    child_dirs = [entry for entry in root.iterdir() if entry.is_dir()]
    child_dirs.sort(key=lambda item: item.name.lower())
    if len(child_dirs) > cfg.skills.max_candidates_per_root:
        child_dirs = child_dirs[: cfg.skills.max_candidates_per_root]

    for child in child_dirs:
        skill_md = child / "SKILL.md"
        if skill_md.is_file():
            candidates.append(skill_md)

    if len(candidates) > cfg.skills.max_skills_loaded_per_source:
        candidates = candidates[: cfg.skills.max_skills_loaded_per_source]
    return candidates


def _load_skill_entry(skill_md: Path, source: str, cfg: Config) -> SkillEntry | None:
    try:
        size = skill_md.stat().st_size
    except Exception:
        return None
    if size > cfg.skills.max_skill_file_bytes:
        return None

    try:
        raw = skill_md.read_text(encoding="utf-8")
    except Exception:
        return None

    frontmatter = _parse_frontmatter(raw)
    name = str(frontmatter.get("name", "")).strip() or skill_md.parent.name.strip()
    if not name:
        return None
    raw_description = str(frontmatter.get("description", "")).strip()
    description = raw_description or f"Skill instructions from {skill_md.parent.name}"
    description = re.sub(r"\s+", " ", description)[:_DEFAULT_MAX_DESCRIPTION_CHARS]
    metadata = _parse_metadata(frontmatter)
    invocation = _resolve_invocation_policy(frontmatter)
    return SkillEntry(
        name=name,
        description=description,
        file_path=str(skill_md.resolve()),
        base_dir=str(skill_md.parent.resolve()),
        source=source,
        frontmatter=frontmatter,
        metadata=metadata,
        invocation=invocation,
    )


def load_workspace_skill_entries(workspace_dir: str | Path, cfg: Config) -> list[SkillEntry]:
    """Load raw skill entries from all configured roots with precedence merge."""
    workspace = Path(workspace_dir).expanduser().resolve()
    merged: dict[str, SkillEntry] = {}
    for source, root in _resolve_skill_roots(workspace, cfg):
        for skill_md in _iter_skill_md_candidates(root, cfg):
            entry = _load_skill_entry(skill_md, source, cfg)
            if not entry:
                continue
            merged[entry.name] = entry
    return list(merged.values())


def _resolve_runtime_platform() -> str:
    platform = sys.platform.lower()
    if platform.startswith("linux"):
        return "linux"
    if platform.startswith("darwin"):
        return "darwin"
    if platform.startswith("win"):
        return "win32"
    return platform


def _resolve_skill_key(entry: SkillEntry) -> str:
    if entry.metadata and entry.metadata.skill_key:
        return entry.metadata.skill_key
    return entry.name


def _resolve_skill_config(cfg: Config, skill_key: str) -> SkillEntryConfig | None:
    entries = cfg.skills.entries or {}
    if skill_key in entries:
        return entries[skill_key]
    lowered = skill_key.lower()
    for key, value in entries.items():
        if str(key).strip().lower() == lowered:
            return value
    return None


def _is_bundled_skill_allowed(entry: SkillEntry, allowlist: list[str]) -> bool:
    if not allowlist:
        return True
    if entry.source not in _BUNDLED_SOURCES:
        return True
    key = _resolve_skill_key(entry)
    return key in allowlist or entry.name in allowlist


def _is_config_path_truthy(cfg_payload: dict[str, Any], path: str) -> bool:
    text = str(path or "").strip()
    if not text:
        return False
    current: Any = cfg_payload
    for part in text.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return False
    if isinstance(current, str):
        return bool(current.strip())
    return bool(current)


def _meets_requires(
    entry: SkillEntry,
    cfg: Config,
    cfg_payload: dict[str, Any],
    entry_cfg: SkillEntryConfig | None,
) -> bool:
    requires = entry.metadata.requires if entry.metadata else None
    if not requires:
        return True

    for bin_name in requires.bins:
        if shutil.which(bin_name) is None:
            return False
    if requires.any_bins:
        if not any(shutil.which(bin_name) is not None for bin_name in requires.any_bins):
            return False
    for env_name in requires.env:
        has_env = bool(os.environ.get(env_name))
        has_cfg_env = bool(entry_cfg and entry_cfg.env.get(env_name))
        has_cfg_key = bool(
            entry_cfg
            and entry_cfg.api_key
            and entry.metadata
            and entry.metadata.primary_env == env_name
        )
        if not (has_env or has_cfg_env or has_cfg_key):
            return False
    for cfg_path in requires.config:
        if not _is_config_path_truthy(cfg_payload, cfg_path):
            return False
    return True


def filter_skill_entries(
    entries: list[SkillEntry],
    cfg: Config,
    skill_filter: list[str] | None = None,
) -> list[SkillEntry]:
    """Filter loaded skills by config/metadata requirements."""
    cfg_payload = cfg.model_dump(mode="python")
    platform = _resolve_runtime_platform()
    allowlist = [entry.strip() for entry in cfg.skills.allow_bundled if str(entry).strip()]
    normalized_filter = {name.strip() for name in (skill_filter or []) if str(name).strip()}
    filtered: list[SkillEntry] = []

    for entry in entries:
        if normalized_filter and entry.name not in normalized_filter:
            continue
        if not _is_bundled_skill_allowed(entry, allowlist):
            continue

        skill_key = _resolve_skill_key(entry)
        entry_cfg = _resolve_skill_config(cfg, skill_key)
        if entry_cfg and entry_cfg.enabled is False:
            continue

        os_list = entry.metadata.os if entry.metadata else []
        if os_list and platform not in os_list:
            continue

        if entry.metadata and entry.metadata.always is True:
            filtered.append(entry)
            continue

        if not _meets_requires(entry, cfg, cfg_payload, entry_cfg):
            continue
        filtered.append(entry)
    return filtered


def _format_skills_for_prompt(entries: list[SkillEntry]) -> str:
    if not entries:
        return ""
    lines = ["<available_skills>"]
    for entry in entries:
        lines.append("  <skill>")
        lines.append(f"    <name>{html.escape(entry.name)}</name>")
        lines.append(f"    <description>{html.escape(entry.description)}</description>")
        lines.append(f"    <location>{html.escape(entry.file_path)}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _apply_skills_prompt_limits(entries: list[SkillEntry], cfg: Config) -> tuple[list[SkillEntry], bool]:
    limited = list(entries[: max(0, int(cfg.skills.max_skills_in_prompt))])
    truncated = len(entries) > len(limited)
    if not limited:
        return limited, truncated

    max_chars = max(0, int(cfg.skills.max_skills_prompt_chars))
    if len(_format_skills_for_prompt(limited)) <= max_chars:
        return limited, truncated

    lo = 0
    hi = len(limited)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = limited[:mid]
        if len(_format_skills_for_prompt(candidate)) <= max_chars:
            lo = mid
        else:
            hi = mid - 1
    limited = limited[:lo]
    return limited, True


def _sanitize_skill_command_name(raw_name: str) -> str:
    lowered = str(raw_name).strip().lower()
    lowered = re.sub(r"[^a-z0-9_\-\s]+", "", lowered)
    lowered = re.sub(r"[\s_]+", "-", lowered).strip("-")
    if not lowered:
        lowered = "skill"
    return lowered[:_SKILL_COMMAND_NAME_MAX]


def _resolve_unique_skill_command_name(base_name: str, used: set[str]) -> str:
    if base_name not in used:
        used.add(base_name)
        return base_name
    suffix = 2
    while True:
        candidate = f"{base_name}-{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        suffix += 1


def build_workspace_skill_command_specs(entries: list[SkillEntry]) -> list[SkillCommandSpec]:
    """Build user-invocable skill command specs from eligible entries."""
    commands: list[SkillCommandSpec] = []
    used: set[str] = set()
    for entry in entries:
        if entry.invocation.user_invocable is False:
            continue
        base_name = _sanitize_skill_command_name(entry.name)
        command_name = _resolve_unique_skill_command_name(base_name, used)
        description = (entry.description or entry.name).strip()
        if len(description) > _SKILL_COMMAND_DESC_MAX:
            description = description[: _SKILL_COMMAND_DESC_MAX - 1].rstrip() + "..."

        dispatch: SkillCommandDispatch | None = None
        dispatch_kind = str(
            entry.frontmatter.get("command-dispatch", entry.frontmatter.get("command_dispatch", ""))
        ).strip().lower()
        if dispatch_kind == "tool":
            tool_name = str(
                entry.frontmatter.get("command-tool", entry.frontmatter.get("command_tool", ""))
            ).strip()
            if tool_name:
                arg_mode_raw = str(
                    entry.frontmatter.get(
                        "command-arg-mode",
                        entry.frontmatter.get("command_arg_mode", "raw"),
                    )
                ).strip().lower()
                arg_mode = "raw" if arg_mode_raw in {"", "raw"} else "raw"
                dispatch = SkillCommandDispatch(kind="tool", tool_name=tool_name, arg_mode=arg_mode)
            else:
                log.warning(
                    "Skill command requested tool dispatch but command-tool is missing",
                    skill_name=entry.name,
                )

        commands.append(
            SkillCommandSpec(
                name=command_name,
                skill_name=entry.name,
                description=description,
                dispatch=dispatch,
            )
        )
    return commands


def _normalize_skill_lookup_name(value: str) -> str:
    return re.sub(r"[\s_]+", "-", str(value).strip().lower())


def find_skill_command(commands: list[SkillCommandSpec], raw_name: str) -> SkillCommandSpec | None:
    """Find command spec by command name or skill name."""
    lowered = str(raw_name).strip().lower()
    normalized = _normalize_skill_lookup_name(raw_name)
    if not lowered:
        return None
    for command in commands:
        if command.name.lower() == lowered or command.skill_name.lower() == lowered:
            return command
        if _normalize_skill_lookup_name(command.name) == normalized:
            return command
        if _normalize_skill_lookup_name(command.skill_name) == normalized:
            return command
    return None


def resolve_skill_command_invocation(
    command_body: str,
    commands: list[SkillCommandSpec],
) -> tuple[SkillCommandSpec, str | None] | None:
    """Resolve `/skill <name> [args]` or `/<command> [args]` invocations."""
    text = str(command_body).strip()
    if not text.startswith("/"):
        return None
    match = re.match(r"^/([^\s]+)(?:\s+([\s\S]+))?$", text)
    if not match:
        return None

    command_name = str(match.group(1) or "").strip().lower()
    args = str(match.group(2) or "").strip() or None
    if command_name == "skill":
        if not args:
            return None
        nested = re.match(r"^([^\s]+)(?:\s+([\s\S]+))?$", args)
        if not nested:
            return None
        skill_name = str(nested.group(1) or "").strip()
        nested_args = str(nested.group(2) or "").strip() or None
        command = find_skill_command(commands, skill_name)
        if not command:
            return None
        return command, nested_args

    command = find_skill_command(commands, command_name)
    if not command:
        return None
    return command, args


def _serialize_snapshot(snapshot: SkillSnapshot) -> dict[str, Any]:
    return {
        "version": snapshot.version,
        "prompt": snapshot.prompt,
        "skills": [
            {"name": item.name, "skill_key": item.skill_key, "primary_env": item.primary_env}
            for item in snapshot.skills
        ],
        "commands": [
            {
                "name": command.name,
                "skill_name": command.skill_name,
                "description": command.description,
                "dispatch": (
                    {
                        "kind": command.dispatch.kind,
                        "tool_name": command.dispatch.tool_name,
                        "arg_mode": command.dispatch.arg_mode,
                    }
                    if command.dispatch
                    else None
                ),
            }
            for command in snapshot.commands
        ],
    }


def snapshot_from_dict(payload: dict[str, Any]) -> SkillSnapshot | None:
    """Restore a skill snapshot from session metadata payload."""
    if not isinstance(payload, dict):
        return None
    skills_raw = payload.get("skills")
    commands_raw = payload.get("commands")
    if not isinstance(skills_raw, list) or not isinstance(commands_raw, list):
        return None

    skills: list[SkillSnapshotSkill] = []
    for item in skills_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        skill_key = str(item.get("skill_key", "")).strip() or name
        if not name:
            continue
        primary_env = item.get("primary_env")
        if primary_env is not None:
            primary_env = str(primary_env).strip() or None
        skills.append(SkillSnapshotSkill(name=name, skill_key=skill_key, primary_env=primary_env))

    commands: list[SkillCommandSpec] = []
    for item in commands_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        skill_name = str(item.get("skill_name", "")).strip()
        description = str(item.get("description", "")).strip()
        if not name or not skill_name:
            continue
        dispatch = None
        dispatch_raw = item.get("dispatch")
        if isinstance(dispatch_raw, dict):
            tool_name = str(dispatch_raw.get("tool_name", "")).strip()
            if tool_name:
                dispatch = SkillCommandDispatch(
                    kind=str(dispatch_raw.get("kind", "tool")).strip() or "tool",
                    tool_name=tool_name,
                    arg_mode=str(dispatch_raw.get("arg_mode", "raw")).strip() or "raw",
                )
        commands.append(
            SkillCommandSpec(
                name=name,
                skill_name=skill_name,
                description=description,
                dispatch=dispatch,
            )
        )

    return SkillSnapshot(
        prompt=str(payload.get("prompt", "")),
        skills=skills,
        commands=commands,
        version=str(payload.get("version", "")),
    )


def snapshot_to_dict(snapshot: SkillSnapshot) -> dict[str, Any]:
    """Serialize snapshot for storing in session metadata."""
    return _serialize_snapshot(snapshot)


def build_workspace_skill_snapshot(
    workspace_dir: str | Path,
    cfg: Config,
    skill_filter: list[str] | None = None,
    version: str | None = None,
) -> SkillSnapshot:
    """Build skill snapshot used for prompt injection and command routing."""
    entries = load_workspace_skill_entries(workspace_dir, cfg)
    eligible = filter_skill_entries(entries, cfg, skill_filter=skill_filter)
    prompt_entries = [entry for entry in eligible if not entry.invocation.disable_model_invocation]
    limited_entries, truncated = _apply_skills_prompt_limits(prompt_entries, cfg)
    prompt = _format_skills_for_prompt(limited_entries)
    if truncated and prompt:
        prompt = (
            "⚠️ Skills truncated: included "
            f"{len(limited_entries)} of {len(prompt_entries)} entries.\n{prompt}"
        )
    commands = build_workspace_skill_command_specs(eligible)
    skills = [
        SkillSnapshotSkill(
            name=entry.name,
            skill_key=_resolve_skill_key(entry),
            primary_env=(entry.metadata.primary_env if entry.metadata else None),
        )
        for entry in eligible
    ]
    return SkillSnapshot(
        prompt=prompt,
        skills=skills,
        commands=commands,
        version=(version or compute_skills_snapshot_version(workspace_dir, cfg)),
    )


def compute_skills_snapshot_version(workspace_dir: str | Path, cfg: Config) -> str:
    """Compute deterministic skills version hash from discovered SKILL.md files."""
    workspace = Path(workspace_dir).expanduser().resolve()
    digest = hashlib.sha1()
    count = 0
    for _, root in _resolve_skill_roots(workspace, cfg):
        for skill_md in _iter_skill_md_candidates(root, cfg):
            try:
                stat = skill_md.stat()
            except Exception:
                continue
            digest.update(str(skill_md.resolve()).encode("utf-8", errors="ignore"))
            digest.update(str(stat.st_mtime_ns).encode("ascii", errors="ignore"))
            digest.update(str(stat.st_size).encode("ascii", errors="ignore"))
            count += 1
    digest.update(str(count).encode("ascii", errors="ignore"))
    return digest.hexdigest()


def apply_skill_env_overrides(snapshot: SkillSnapshot | None, cfg: Config) -> Callable[[], None]:
    """Apply per-skill env/apiKey overrides for one run and return reverter."""
    updates: list[tuple[str, str | None]] = []
    if not snapshot:
        return lambda: None

    for skill in snapshot.skills:
        entry_cfg = _resolve_skill_config(cfg, skill.skill_key) or _resolve_skill_config(cfg, skill.name)
        if not entry_cfg:
            continue
        for key, value in (entry_cfg.env or {}).items():
            if not key or not value:
                continue
            if os.environ.get(key):
                continue
            updates.append((key, os.environ.get(key)))
            os.environ[key] = value
        if skill.primary_env and entry_cfg.api_key and not os.environ.get(skill.primary_env):
            updates.append((skill.primary_env, os.environ.get(skill.primary_env)))
            os.environ[skill.primary_env] = entry_cfg.api_key

    def _restore() -> None:
        for key, previous in updates:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    return _restore
