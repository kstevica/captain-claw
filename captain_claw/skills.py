"""Skill discovery, filtering, prompt construction, and command invocation metadata."""

from __future__ import annotations

import hashlib
import html
import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urljoin, urlparse

import httpx
import yaml

from captain_claw.config import Config, SkillEntryConfig
from captain_claw.logging import get_logger

log = get_logger(__name__)

SOURCE_EXTRA = "captain-extra"
SOURCE_PLUGIN = "captain-plugin"
SOURCE_BUNDLED = "captain-bundled"
SOURCE_MANAGED = "captain-managed"
SOURCE_AGENTS_PERSONAL = "agents-skills-personal"
SOURCE_AGENTS_PROJECT = "agents-skills-project"
SOURCE_WORKSPACE = "captain-workspace"

SOURCE_PRECEDENCE: tuple[str, ...] = (
    SOURCE_EXTRA,
    SOURCE_PLUGIN,
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
_GITHUB_API_BASE_URL = "https://api.github.com"
_DEFAULT_INSTALL_TIMEOUT_SECONDS = 300
_SKILLS_INSTALL_TOOLS_DIR = "~/.captain-claw/tools"


@dataclass
class SkillInstallSpec:
    kind: str
    id: str | None = None
    label: str | None = None
    bins: list[str] = field(default_factory=list)
    os: list[str] = field(default_factory=list)
    formula: str | None = None
    package: str | None = None
    module: str | None = None
    url: str | None = None
    archive: str | None = None
    extract: bool | None = None
    strip_components: int | None = None
    target_dir: str | None = None


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
    install: list[SkillInstallSpec] = field(default_factory=list)


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
    tool_name: str = ""
    arg_mode: str = "raw"
    script_path: str = ""
    script_interpreter: str = ""


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


@dataclass
class GitHubSkillSource:
    owner: str
    repo: str
    ref: str | None = None
    skill_path: str = ""
    url: str = ""


@dataclass
class SkillInstallResult:
    skill_name: str
    destination: str
    repo: str
    ref: str | None = None
    skill_path: str = ""
    source_url: str = ""


@dataclass
class SkillCatalogEntry:
    name: str
    url: str
    description: str = ""


@dataclass
class SkillDependencyInstallResult:
    ok: bool
    skill_name: str
    install_id: str
    kind: str
    command: str
    message: str
    stdout: str = ""
    stderr: str = ""
    code: int | None = None


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

    # Compatibility: some skill packs namespace metadata under a single custom vendor key
    # (e.g. {"clawdbot": {...}}). If there's exactly one top-level dict entry, unwrap it.
    nested_dict_values = [
        value
        for value in metadata_obj.values()
        if isinstance(value, dict)
    ]
    if len(nested_dict_values) == 1 and len(metadata_obj) == 1:
        return nested_dict_values[0]
    # Support already-unwrapped block.
    return metadata_obj


def _parse_install_spec(value: Any) -> SkillInstallSpec | None:
    if not isinstance(value, dict):
        return None
    kind = str(value.get("kind", value.get("type", ""))).strip().lower()
    if kind not in {"brew", "node", "go", "uv", "download"}:
        return None
    strip_components: int | None = None
    raw_strip = value.get("stripComponents")
    if isinstance(raw_strip, int):
        strip_components = max(0, raw_strip)
    return SkillInstallSpec(
        kind=kind,
        id=str(value.get("id")).strip() if isinstance(value.get("id"), str) else None,
        label=str(value.get("label")).strip() if isinstance(value.get("label"), str) else None,
        bins=_normalize_string_list(value.get("bins")),
        os=_normalize_string_list(value.get("os")),
        formula=str(value.get("formula")).strip() if isinstance(value.get("formula"), str) else None,
        package=str(value.get("package")).strip() if isinstance(value.get("package"), str) else None,
        module=str(value.get("module")).strip() if isinstance(value.get("module"), str) else None,
        url=str(value.get("url")).strip() if isinstance(value.get("url"), str) else None,
        archive=str(value.get("archive")).strip() if isinstance(value.get("archive"), str) else None,
        extract=value.get("extract") if isinstance(value.get("extract"), bool) else None,
        strip_components=strip_components,
        target_dir=(
            str(value.get("targetDir")).strip()
            if isinstance(value.get("targetDir"), str)
            else None
        ),
    )


def _parse_install_specs(value: Any) -> list[SkillInstallSpec]:
    if not isinstance(value, list):
        return []
    specs: list[SkillInstallSpec] = []
    for item in value:
        parsed = _parse_install_spec(item)
        if parsed:
            specs.append(parsed)
    return specs


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
        install=_parse_install_specs(meta.get("install")),
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
    package_dir = Path(__file__).resolve().parent
    project_skills_dir = (package_dir.parent / "skills").resolve()
    legacy_skills_dir = (package_dir / "skills").resolve()
    if project_skills_dir.exists():
        return project_skills_dir
    if legacy_skills_dir.exists():
        return legacy_skills_dir
    return project_skills_dir


def _normalize_repo_subpath(path_text: str) -> str:
    cleaned = str(path_text or "").replace("\\", "/").strip().strip("/")
    if not cleaned:
        return ""
    raw_parts = [part for part in cleaned.split("/") if part]
    if any(part in {".", ".."} for part in raw_parts):
        raise ValueError("Skill path cannot contain '.' or '..' path segments.")
    normalized = posixpath.normpath(cleaned)
    if normalized in {"", "."}:
        return ""
    parts = [part for part in normalized.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        raise ValueError("Skill path cannot escape repository boundaries.")
    return "/".join(parts)


def _strip_skill_md_suffix(path_text: str) -> str:
    normalized = _normalize_repo_subpath(path_text)
    if not normalized:
        return ""
    if normalized.lower() == "skill.md":
        return ""
    if normalized.lower().endswith("/skill.md"):
        return normalized[: -len("/skill.md")]
    return normalized


def parse_github_skill_source(url: str) -> GitHubSkillSource:
    """Parse supported GitHub URLs for skill installation."""
    raw_url = str(url or "").strip()
    if not raw_url:
        raise ValueError("GitHub URL is required.")
    parsed = urlparse(raw_url)
    scheme = parsed.scheme.lower().strip()
    host = parsed.netloc.lower().strip()
    if scheme not in {"https", "http"}:
        raise ValueError("GitHub URL must start with http:// or https://.")
    if host not in {"github.com", "www.github.com"}:
        raise ValueError("Only github.com URLs are supported.")

    path_parts = [unquote(part).strip() for part in parsed.path.split("/") if part.strip()]
    if len(path_parts) < 2:
        raise ValueError("GitHub URL must include owner and repository.")
    owner = path_parts[0]
    repo = path_parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        raise ValueError("GitHub URL must include a valid owner and repository.")

    ref: str | None = None
    skill_path = ""
    if len(path_parts) >= 3:
        marker = path_parts[2].lower()
        if marker == "tree":
            if len(path_parts) < 4:
                raise ValueError("Tree URL must include /tree/<ref>[/<skill-path>].")
            ref = path_parts[3].strip()
            skill_path = "/".join(part for part in path_parts[4:] if part)
        elif marker == "blob":
            if len(path_parts) < 5:
                raise ValueError("Blob URL must include /blob/<ref>/<path-to-SKILL.md>.")
            ref = path_parts[3].strip()
            blob_parts = [part for part in path_parts[4:] if part]
            if not blob_parts or blob_parts[-1].lower() != "skill.md":
                raise ValueError("Blob URL must point to SKILL.md.")
            skill_path = "/".join(blob_parts[:-1])
        else:
            raise ValueError(
                "Unsupported GitHub URL format. Use repo URL or /tree/<ref>/<skill-path>."
            )
    skill_path = _strip_skill_md_suffix(skill_path)
    return GitHubSkillSource(
        owner=owner,
        repo=repo,
        ref=ref,
        skill_path=skill_path,
        url=raw_url,
    )


def _run_git_clone(repo_url: str, destination: Path, ref: str | None, timeout_seconds: int) -> None:
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd.extend(["--branch", ref])
    cmd.extend([repo_url, str(destination)])
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=max(1, int(timeout_seconds)),
    )
    if completed.returncode == 0:
        return
    details = (completed.stderr or completed.stdout or "").strip()
    if details:
        raise RuntimeError(f"git clone failed: {details}")
    raise RuntimeError("git clone failed.")


def _build_github_api_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Captain Claw/0.1.0 (Skill Installer)",
    }
    token = (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_contents_api_url(owner: str, repo: str, repo_path: str) -> str:
    base = f"{_GITHUB_API_BASE_URL}/repos/{owner}/{repo}/contents"
    normalized_path = _normalize_repo_subpath(repo_path)
    if not normalized_path:
        return base
    return f"{base}/{quote(normalized_path, safe='/')}"


def _fetch_github_contents(
    client: httpx.Client,
    source: GitHubSkillSource,
    repo_path: str,
) -> list[dict[str, Any]] | dict[str, Any]:
    request_url = _github_contents_api_url(source.owner, source.repo, repo_path)
    params: dict[str, str] = {}
    if source.ref:
        params["ref"] = source.ref
    response = client.get(request_url, params=params or None)
    if response.status_code == 404:
        raise FileNotFoundError(f"GitHub path not found: {repo_path or '/'}")
    try:
        response.raise_for_status()
    except Exception as exc:
        details = (response.text or "").strip()
        if details:
            raise RuntimeError(f"GitHub API request failed: {details[:500]}") from exc
        raise
    payload: Any = response.json()
    if isinstance(payload, dict) or isinstance(payload, list):
        return payload
    raise RuntimeError("Unexpected GitHub API response format.")


def _download_file_to_path(client: httpx.Client, download_url: str, destination: Path) -> None:
    response = client.get(download_url)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)


def _download_github_directory(
    client: httpx.Client,
    source: GitHubSkillSource,
    repo_path: str,
    destination_dir: Path,
) -> None:
    payload = _fetch_github_contents(client, source, repo_path)
    if isinstance(payload, dict):
        if str(payload.get("type", "")).strip().lower() != "file":
            raise RuntimeError("Expected a file entry from GitHub API.")
        name = str(payload.get("name", "")).strip()
        download_url = str(payload.get("download_url", "")).strip()
        if not name or not download_url:
            raise RuntimeError("GitHub API file entry is missing required fields.")
        _download_file_to_path(client, download_url, destination_dir / name)
        return

    if not isinstance(payload, list):
        raise RuntimeError("Expected a directory listing from GitHub API.")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for item in payload:
        if not isinstance(item, dict):
            continue
        entry_type = str(item.get("type", "")).strip().lower()
        entry_name = str(item.get("name", "")).strip()
        entry_path = str(item.get("path", "")).strip()
        if not entry_name or not entry_path:
            continue
        if entry_type == "file":
            download_url = str(item.get("download_url", "")).strip()
            if not download_url:
                continue
            _download_file_to_path(client, download_url, destination_dir / entry_name)
            continue
        if entry_type == "dir":
            _download_github_directory(client, source, entry_path, destination_dir / entry_name)


def _download_skill_from_github_api(
    source: GitHubSkillSource,
    repo_root: Path,
    timeout_seconds: int,
) -> Path:
    repo_path = _strip_skill_md_suffix(source.skill_path)
    if not repo_path:
        raise ValueError("A /tree/<ref>/<skill-path> or /blob/<ref>/<path-to-SKILL.md> URL is required.")
    destination = (repo_root / repo_path).resolve()
    try:
        destination.relative_to(repo_root.resolve())
    except Exception as exc:
        raise ValueError("Resolved skill path is outside temporary workspace.") from exc

    timeout_value = max(1, int(timeout_seconds))
    with httpx.Client(
        timeout=timeout_value,
        follow_redirects=True,
        headers=_build_github_api_headers(),
    ) as client:
        _download_github_directory(client, source, repo_path, destination)

    skill_md = destination / "SKILL.md"
    if not skill_md.is_file():
        raise ValueError("Selected GitHub path does not contain SKILL.md.")
    return destination


def _infer_skill_dir_from_repo_root(repo_root: Path) -> Path:
    direct = repo_root / "SKILL.md"
    if direct.is_file():
        return repo_root
    candidates = [
        item
        for item in sorted(repo_root.iterdir(), key=lambda entry: entry.name.lower())
        if item.is_dir() and (item / "SKILL.md").is_file()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        sample = ", ".join(item.name for item in candidates[:5])
        if len(candidates) > 5:
            sample += ", ..."
        raise ValueError(
            "Repository contains multiple skill folders. Use a /tree/<ref>/<skill-path> URL. "
            f"Candidates: {sample}"
        )
    raise ValueError(
        "No SKILL.md found in repository root. Use a /tree/<ref>/<skill-path> URL that points to the skill folder."
    )


def _sanitize_skill_install_dir_name(raw_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(raw_name or "").strip())
    cleaned = cleaned.strip("-._")
    if not cleaned:
        return ""
    return cleaned[:128]


def install_skill_from_github_url(
    url: str,
    cfg: Config,
    timeout_seconds: int = 120,
) -> SkillInstallResult:
    """Install a skill into managed skills directory from a GitHub URL."""
    source = parse_github_skill_source(url)
    managed_dir = Path(cfg.skills.managed_dir).expanduser().resolve()
    managed_dir.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://github.com/{source.owner}/{source.repo}.git"

    with tempfile.TemporaryDirectory(prefix="captain-claw-skill-") as temp_dir:
        repo_root = (Path(temp_dir) / "repo").resolve()
        selected_dir: Path | None = None
        download_error: Exception | None = None
        if source.skill_path:
            try:
                selected_dir = _download_skill_from_github_api(source, repo_root, timeout_seconds)
            except Exception as exc:
                download_error = exc
                log.warning(
                    "GitHub API skill download failed; falling back to git clone",
                    repo=f"{source.owner}/{source.repo}",
                    skill_path=source.skill_path,
                    error=str(exc),
                )

        if selected_dir is None:
            try:
                _run_git_clone(repo_url, repo_root, source.ref, timeout_seconds)
            except Exception as exc:
                if download_error:
                    raise RuntimeError(
                        "Failed to install skill: download attempt failed "
                        f"({download_error}); clone attempt failed ({exc})"
                    ) from exc
                raise RuntimeError(f"Failed to clone GitHub repository: {str(exc)}") from exc
            if source.skill_path:
                selected_dir = (repo_root / source.skill_path).resolve()
            else:
                selected_dir = _infer_skill_dir_from_repo_root(repo_root).resolve()
        if selected_dir.is_file() and selected_dir.name.lower() == "skill.md":
            selected_dir = selected_dir.parent.resolve()
        try:
            selected_dir.relative_to(repo_root)
        except Exception as exc:
            raise ValueError("Resolved skill path is outside cloned repository.") from exc
        if not selected_dir.exists() or not selected_dir.is_dir():
            raise ValueError(f"Skill directory does not exist: {source.skill_path or '.'}")

        skill_md = selected_dir / "SKILL.md"
        if not skill_md.is_file():
            raise ValueError("Selected directory does not contain SKILL.md.")

        base_name = selected_dir.name if selected_dir.name else source.repo
        install_name = _sanitize_skill_install_dir_name(base_name) or _sanitize_skill_install_dir_name(
            source.repo
        )
        if not install_name:
            raise ValueError("Unable to derive a valid skill folder name from URL.")
        destination = (managed_dir / install_name).resolve()
        if destination.exists():
            raise FileExistsError(
                f"Skill folder already exists: {destination}. Remove it first or choose a different skill path."
            )
        try:
            destination.relative_to(managed_dir)
        except Exception as exc:
            raise ValueError("Resolved destination path is outside managed skills directory.") from exc

        try:
            shutil.copytree(selected_dir, destination, ignore=shutil.ignore_patterns(".git"))
            entry = _load_skill_entry(destination / "SKILL.md", SOURCE_MANAGED, cfg)
            if not entry:
                raise ValueError(
                    "Installed SKILL.md could not be parsed or exceeds skill file size limits."
                )
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise

        resolved_path = source.skill_path or ("" if selected_dir == repo_root else selected_dir.name)
        return SkillInstallResult(
            skill_name=entry.name,
            destination=str(destination),
            repo=f"{source.owner}/{source.repo}",
            ref=source.ref,
            skill_path=resolved_path,
            source_url=source.url,
        )


def _resolve_skill_entry(entries: list[SkillEntry], raw_name: str) -> SkillEntry | None:
    needle = str(raw_name or "").strip().lower()
    if not needle:
        return None
    for entry in entries:
        if entry.name.strip().lower() == needle:
            return entry
        skill_key = _resolve_skill_key(entry).strip().lower()
        if skill_key and skill_key == needle:
            return entry
    return None


def _resolve_install_id(spec: SkillInstallSpec, index: int) -> str:
    raw = str(spec.id or "").strip()
    if raw:
        return raw
    return f"{spec.kind}-{index}"


def _select_preferred_install_spec(
    specs: list[tuple[int, SkillInstallSpec]],
    cfg: Config,
) -> tuple[int, SkillInstallSpec] | None:
    if not specs:
        return None
    prefer_brew = bool(getattr(cfg.skills.install, "prefer_brew", True))
    brew_available = shutil.which("brew") is not None

    def _pick(kind: str, require_brew: bool = False) -> tuple[int, SkillInstallSpec] | None:
        for item in specs:
            if item[1].kind != kind:
                continue
            if require_brew and not brew_available:
                continue
            return item
        return None

    ordered_picks = [
        _pick("brew", require_brew=prefer_brew),
        _pick("uv"),
        _pick("node"),
        _pick("brew", require_brew=True),
        _pick("go"),
        _pick("download"),
    ]
    for selected in ordered_picks:
        if selected:
            return selected
    return specs[0]


def _normalize_install_specs_for_platform(
    specs: list[SkillInstallSpec],
) -> list[tuple[int, SkillInstallSpec]]:
    platform = _resolve_runtime_platform()
    filtered: list[tuple[int, SkillInstallSpec]] = []
    for index, spec in enumerate(specs):
        if spec.os and platform not in spec.os:
            continue
        filtered.append((index, spec))
    return filtered


def _run_install_command(
    argv: list[str],
    timeout_seconds: int,
    env: dict[str, str] | None = None,
) -> tuple[int | None, str, str]:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1, int(timeout_seconds)),
            env=env,
        )
    except Exception as exc:
        return None, "", str(exc)
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def _resolve_node_install_command(package_name: str, cfg: Config) -> list[str]:
    manager = str(getattr(cfg.skills.install, "node_manager", "npm") or "npm").strip().lower()
    if manager == "pnpm":
        return ["pnpm", "add", "-g", "--ignore-scripts", package_name]
    if manager == "yarn":
        return ["yarn", "global", "add", "--ignore-scripts", package_name]
    if manager == "bun":
        return ["bun", "add", "-g", "--ignore-scripts", package_name]
    return ["npm", "install", "-g", "--ignore-scripts", package_name]


def _resolve_skill_tools_root(entry: SkillEntry) -> Path:
    root = Path(_SKILLS_INSTALL_TOOLS_DIR).expanduser().resolve()
    key = _resolve_skill_key(entry)
    safe_key = re.sub(r"[^A-Za-z0-9._-]+", "-", key).strip("-._") or "skill"
    return (root / safe_key).resolve()


def _resolve_download_target_dir(entry: SkillEntry, spec: SkillInstallSpec) -> Path:
    safe_root = _resolve_skill_tools_root(entry)
    raw_target = str(spec.target_dir or "").strip()
    if not raw_target:
        target = safe_root
    else:
        base = Path(raw_target).expanduser()
        if base.is_absolute() or raw_target.startswith("~"):
            target = base.resolve()
        else:
            target = (safe_root / base).resolve()
    try:
        target.relative_to(safe_root)
    except Exception as exc:
        raise ValueError(
            f'Refusing to install outside the skill tools directory: "{target}" (allowed root: "{safe_root}")'
        ) from exc
    return target


def _detect_archive_type(spec: SkillInstallSpec, filename: str) -> str | None:
    explicit = str(spec.archive or "").strip().lower()
    if explicit:
        return explicit
    lower = filename.lower()
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return "tar.gz"
    if lower.endswith(".tar.bz2") or lower.endswith(".tbz2"):
        return "tar.bz2"
    if lower.endswith(".zip"):
        return "zip"
    return None


def _normalize_archive_member_path(raw: str, strip_components: int = 0) -> str | None:
    cleaned = str(raw or "").replace("\\", "/")
    if not cleaned:
        return None
    parts = [part for part in cleaned.split("/") if part and part != "."]
    if strip_components > 0:
        parts = parts[strip_components:]
    if not parts:
        return None
    normalized = posixpath.normpath("/".join(parts))
    if not normalized or normalized in {".", ".."}:
        return None
    if normalized.startswith("../") or normalized.startswith("/"):
        raise ValueError(f"Archive member escapes target directory: {raw}")
    if any(part in {"..", ""} for part in normalized.split("/")):
        raise ValueError(f"Archive member escapes target directory: {raw}")
    return normalized


def _extract_zip_archive(archive_path: Path, target_dir: Path, strip_components: int) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            rel_path = _normalize_archive_member_path(member.filename, strip_components)
            if not rel_path:
                continue
            destination = (target_dir / rel_path).resolve()
            destination.relative_to(target_dir)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source_file:
                with destination.open("wb") as out_file:
                    shutil.copyfileobj(source_file, out_file)


def _extract_tar_archive(archive_path: Path, target_dir: Path, strip_components: int) -> None:
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError("Archive contains symbolic or hard links; refusing extraction.")
            rel_path = _normalize_archive_member_path(member.name, strip_components)
            if not rel_path:
                continue
            destination = (target_dir / rel_path).resolve()
            destination.relative_to(target_dir)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_file = archive.extractfile(member)
            if source_file is None:
                continue
            with source_file:
                with destination.open("wb") as out_file:
                    shutil.copyfileobj(source_file, out_file)


def _install_download_spec(
    entry: SkillEntry,
    spec: SkillInstallSpec,
    timeout_seconds: int,
) -> SkillDependencyInstallResult:
    url = str(spec.url or "").strip()
    install_id = _resolve_install_id(spec, 0)
    if not url:
        return SkillDependencyInstallResult(
            ok=False,
            skill_name=entry.name,
            install_id=install_id,
            kind=spec.kind,
            command="download",
            message="Download installer is missing URL.",
        )

    target_dir = _resolve_download_target_dir(entry, spec)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        parsed = urlparse(url)
        filename = Path(parsed.path).name or "download"
    except Exception:
        filename = "download"
    archive_path = (target_dir / filename).resolve()
    try:
        archive_path.relative_to(target_dir)
    except Exception as exc:
        raise ValueError("Resolved download path is outside target directory.") from exc

    timeout_value = max(1, int(timeout_seconds))
    with httpx.Client(timeout=timeout_value, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        archive_path.write_bytes(response.content)

    archive_type = _detect_archive_type(spec, filename)
    should_extract = bool(spec.extract) if spec.extract is not None else bool(archive_type)
    if should_extract:
        if not archive_type:
            raise ValueError("extract=true requested but archive type could not be inferred.")
        strip_components = max(0, int(spec.strip_components or 0))
        if archive_type == "zip":
            _extract_zip_archive(archive_path, target_dir, strip_components)
        elif archive_type in {"tar.gz", "tgz", "tar.bz2", "tbz2"}:
            _extract_tar_archive(archive_path, target_dir, strip_components)
        else:
            raise ValueError(f"Unsupported archive type: {archive_type}")

    return SkillDependencyInstallResult(
        ok=True,
        skill_name=entry.name,
        install_id=install_id,
        kind=spec.kind,
        command=f"download {url}",
        message=f"Installed dependencies for {entry.name} via download into {target_dir}",
        stdout=str(target_dir),
        stderr="",
        code=0,
    )


def install_skill_dependencies(
    skill_name: str,
    workspace_dir: str | Path,
    cfg: Config,
    install_id: str | None = None,
    timeout_seconds: int = _DEFAULT_INSTALL_TIMEOUT_SECONDS,
) -> SkillDependencyInstallResult:
    """Install runtime dependencies declared in a skill metadata.install block."""
    entries = load_workspace_skill_entries(workspace_dir, cfg)
    entry = _resolve_skill_entry(entries, skill_name)
    if not entry:
        return SkillDependencyInstallResult(
            ok=False,
            skill_name=str(skill_name or "").strip(),
            install_id=str(install_id or "").strip(),
            kind="",
            command="",
            message=f"Skill not found: {skill_name}",
        )

    specs = list(entry.metadata.install if entry.metadata else [])
    supported_specs = _normalize_install_specs_for_platform(specs)
    if not supported_specs:
        return SkillDependencyInstallResult(
            ok=False,
            skill_name=entry.name,
            install_id=str(install_id or "").strip(),
            kind="",
            command="",
            message=f'No install options declared for skill "{entry.name}".',
        )

    selected: tuple[int, SkillInstallSpec] | None = None
    requested_id = str(install_id or "").strip()
    if requested_id:
        for index, spec in supported_specs:
            if _resolve_install_id(spec, index) == requested_id:
                selected = (index, spec)
                break
        if selected is None:
            return SkillDependencyInstallResult(
                ok=False,
                skill_name=entry.name,
                install_id=requested_id,
                kind="",
                command="",
                message=f'Installer not found for "{entry.name}": {requested_id}',
            )
    else:
        selected = _select_preferred_install_spec(supported_specs, cfg)

    if selected is None:
        return SkillDependencyInstallResult(
            ok=False,
            skill_name=entry.name,
            install_id=requested_id,
            kind="",
            command="",
            message=f'No eligible installers found for "{entry.name}".',
        )

    index, spec = selected
    resolved_install_id = _resolve_install_id(spec, index)
    timeout_value = max(1, int(timeout_seconds))

    try:
        if spec.kind == "download":
            result = _install_download_spec(entry, spec, timeout_value)
            result.install_id = resolved_install_id
            return result
    except Exception as exc:
        return SkillDependencyInstallResult(
            ok=False,
            skill_name=entry.name,
            install_id=resolved_install_id,
            kind=spec.kind,
            command=f"download {spec.url or ''}".strip(),
            message=f"Install failed: {str(exc)}",
        )

    argv: list[str]
    env: dict[str, str] | None = None
    if spec.kind == "brew":
        formula = str(spec.formula or "").strip()
        if not formula:
            return SkillDependencyInstallResult(
                ok=False,
                skill_name=entry.name,
                install_id=resolved_install_id,
                kind=spec.kind,
                command="brew install",
                message="Install failed: missing brew formula.",
            )
        if shutil.which("brew") is None:
            return SkillDependencyInstallResult(
                ok=False,
                skill_name=entry.name,
                install_id=resolved_install_id,
                kind=spec.kind,
                command=f"brew install {formula}",
                message="Install failed: brew is not installed.",
            )
        argv = ["brew", "install", formula]
    elif spec.kind == "node":
        package = str(spec.package or "").strip()
        if not package:
            return SkillDependencyInstallResult(
                ok=False,
                skill_name=entry.name,
                install_id=resolved_install_id,
                kind=spec.kind,
                command="node install",
                message="Install failed: missing node package name.",
            )
        argv = _resolve_node_install_command(package, cfg)
    elif spec.kind == "go":
        module = str(spec.module or "").strip()
        if not module:
            return SkillDependencyInstallResult(
                ok=False,
                skill_name=entry.name,
                install_id=resolved_install_id,
                kind=spec.kind,
                command="go install",
                message="Install failed: missing go module.",
            )
        argv = ["go", "install", module]
    elif spec.kind == "uv":
        package = str(spec.package or "").strip()
        if not package:
            return SkillDependencyInstallResult(
                ok=False,
                skill_name=entry.name,
                install_id=resolved_install_id,
                kind=spec.kind,
                command="uv tool install",
                message="Install failed: missing uv package name.",
            )
        argv = ["uv", "tool", "install", package]
    else:
        return SkillDependencyInstallResult(
            ok=False,
            skill_name=entry.name,
            install_id=resolved_install_id,
            kind=spec.kind,
            command="",
            message=f"Install failed: unsupported installer kind '{spec.kind}'.",
        )

    code, stdout, stderr = _run_install_command(argv, timeout_value, env=env)
    command_text = " ".join(shlex.quote(part) for part in argv)
    if code == 0:
        return SkillDependencyInstallResult(
            ok=True,
            skill_name=entry.name,
            install_id=resolved_install_id,
            kind=spec.kind,
            command=command_text,
            message=f'Installed dependencies for "{entry.name}".',
            stdout=stdout.strip(),
            stderr=stderr.strip(),
            code=code,
        )

    details = stderr.strip() or stdout.strip() or "Unknown installer failure."
    return SkillDependencyInstallResult(
        ok=False,
        skill_name=entry.name,
        install_id=resolved_install_id,
        kind=spec.kind,
        command=command_text,
        message=f"Install failed: {details}",
        stdout=stdout.strip(),
        stderr=stderr.strip(),
        code=code,
    )


def _candidate_catalog_markdown_urls(source_url: str) -> tuple[str, list[str]]:
    """Return display source URL plus candidate markdown URLs to fetch."""
    raw = str(source_url or "").strip()
    if not raw:
        raise ValueError("Skill search source URL is not configured.")
    parsed = urlparse(raw)
    scheme = parsed.scheme.lower().strip()
    if scheme not in {"https", "http"}:
        raise ValueError("Skill search source URL must start with http:// or https://.")
    host = parsed.netloc.lower().strip()
    path_parts = [part for part in parsed.path.strip("/").split("/") if part]

    if host in {"github.com", "www.github.com"} and len(path_parts) >= 2:
        owner = path_parts[0]
        repo = path_parts[1]
        if repo.endswith(".git"):
            repo = repo[:-4]
        display_url = f"https://github.com/{owner}/{repo}"
        return display_url, [
            f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
            f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md",
        ]

    if raw.lower().endswith(".md"):
        return raw, [raw]

    base = raw.rstrip("/")
    return base, [f"{base}/README.md", base]


def fetch_skill_catalog_markdown(source_url: str, timeout_seconds: int = 20) -> tuple[str, str, str]:
    """Fetch catalog markdown content from configured source."""
    display_url, candidates = _candidate_catalog_markdown_urls(source_url)
    timeout_value = max(1, int(timeout_seconds))
    headers = {"User-Agent": "Captain Claw/0.1.0 (Skill Catalog Search)"}
    last_error = ""
    with httpx.Client(timeout=timeout_value, follow_redirects=True, headers=headers) as client:
        for url in candidates:
            try:
                response = client.get(url)
                response.raise_for_status()
            except Exception as exc:
                last_error = str(exc)
                continue
            body = str(response.text or "").strip()
            if body:
                return body, display_url, url
            last_error = f"Empty response from {url}"
    if not last_error:
        last_error = "No catalog URL candidates could be fetched."
    raise RuntimeError(last_error)


def parse_skill_catalog_entries(markdown: str, source_base_url: str, max_candidates: int = 5000) -> list[SkillCatalogEntry]:
    """Parse markdown bullet list into skill catalog entries."""
    text = str(markdown or "")
    if not text.strip():
        return []

    limit = max(1, int(max_candidates))
    entries: list[SkillCatalogEntry] = []
    seen: set[tuple[str, str]] = set()
    line_re = re.compile(
        r"^\s*[-*+]\s+\[(?P<name>[^\]]+)\]\((?P<url>[^)]+)\)\s*(?:[-:–]\s*(?P<desc>.+))?$"
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = line_re.match(line)
        if not match:
            continue
        name = re.sub(r"\s+", " ", str(match.group("name") or "").strip())
        raw_url = str(match.group("url") or "").strip()
        description = re.sub(r"\s+", " ", str(match.group("desc") or "").strip())
        if not name or not raw_url:
            continue
        if raw_url.startswith("#"):
            continue
        full_url = urljoin(source_base_url.rstrip("/") + "/", raw_url)
        key = (name.lower(), full_url.lower())
        if key in seen:
            continue
        seen.add(key)
        entries.append(SkillCatalogEntry(name=name, url=full_url, description=description))
        if len(entries) >= limit:
            break
    return entries


def load_skill_catalog_entries(
    source_url: str,
    timeout_seconds: int = 20,
    max_candidates: int = 5000,
) -> tuple[str, list[SkillCatalogEntry]]:
    """Load parsed skill catalog entries from configured source."""
    markdown, display_url, _ = fetch_skill_catalog_markdown(source_url, timeout_seconds=timeout_seconds)
    entries = parse_skill_catalog_entries(
        markdown,
        source_base_url=display_url,
        max_candidates=max_candidates,
    )
    return display_url, entries


def rank_skill_catalog_entries(
    query: str,
    entries: list[SkillCatalogEntry],
    limit: int = 10,
) -> list[SkillCatalogEntry]:
    """Fallback lexical ranking for skill catalog entries."""
    requested = max(1, int(limit))
    words = [token for token in re.findall(r"[a-z0-9]+", str(query or "").lower()) if token]
    if not words:
        return list(entries[:requested])
    scored: list[tuple[float, SkillCatalogEntry]] = []
    for entry in entries:
        haystack_name = str(entry.name or "").lower()
        haystack_desc = str(entry.description or "").lower()
        score = 0.0
        for token in words:
            if token in haystack_name:
                score += 4.0
            if token in haystack_desc:
                score += 1.5
        phrase = " ".join(words)
        if phrase and phrase in haystack_name:
            score += 8.0
        if phrase and phrase in haystack_desc:
            score += 3.0
        if score <= 0:
            continue
        scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], item[1].name.lower()))
    if not scored:
        return []
    return [entry for _, entry in scored[:requested]]


def _resolve_path_from_workspace(workspace_dir: Path, raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (workspace_dir / path).resolve()
    return path.resolve()


def _load_plugin_manifest_skill_dirs(plugin_root: Path) -> list[Path]:
    manifest_path = plugin_root / "openclaw.plugin.json"
    if not manifest_path.is_file():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    raw_skills = payload.get("skills")
    if not isinstance(raw_skills, list):
        return []
    resolved: list[Path] = []
    for item in raw_skills:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        candidate = (plugin_root / text).resolve()
        resolved.append(candidate)
    return resolved


def _resolve_plugin_skill_roots(workspace_dir: Path, cfg: Config) -> list[Path]:
    roots: list[Path] = []
    candidates: list[Path] = []

    for raw in getattr(cfg.skills.load, "plugin_dirs", []) or []:
        text = str(raw).strip()
        if not text:
            continue
        candidates.append(_resolve_path_from_workspace(workspace_dir, text))

    for raw in getattr(cfg.tools, "plugin_dirs", []) or []:
        text = str(raw).strip()
        if not text:
            continue
        tool_dir = _resolve_path_from_workspace(workspace_dir, text)
        if tool_dir.name.lower() == "tools":
            candidates.append(tool_dir.parent.resolve())
        else:
            candidates.append(tool_dir.resolve())

    seen: set[Path] = set()
    for plugin_root in candidates:
        if plugin_root in seen:
            continue
        seen.add(plugin_root)

        manifest_skill_dirs = _load_plugin_manifest_skill_dirs(plugin_root)
        if manifest_skill_dirs:
            roots.extend(manifest_skill_dirs)
            continue

        inferred_skills_dir = (plugin_root / "skills").resolve()
        if inferred_skills_dir.exists() and inferred_skills_dir.is_dir():
            roots.append(inferred_skills_dir)

    deduped: list[Path] = []
    seen_roots: set[Path] = set()
    for root in roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)
        deduped.append(root)
    return deduped


def _resolve_skill_roots(workspace_dir: Path, cfg: Config) -> list[tuple[str, Path]]:
    managed_dir = Path(cfg.skills.managed_dir).expanduser().resolve()
    extra_roots = [
        Path(entry).expanduser().resolve()
        for entry in cfg.skills.load.extra_dirs
        if str(entry).strip()
    ]
    plugin_roots = _resolve_plugin_skill_roots(workspace_dir, cfg)
    bundled_dir = _resolve_bundled_skills_dir()
    personal_agents = (Path.home() / ".agents" / "skills").resolve()
    project_agents = (workspace_dir / ".agents" / "skills").resolve()
    workspace_skills = (workspace_dir / "skills").resolve()

    ordered_roots: list[tuple[str, Path]] = []
    ordered_roots.extend((SOURCE_EXTRA, root) for root in extra_roots)
    ordered_roots.extend((SOURCE_PLUGIN, root) for root in plugin_roots)
    ordered_roots.append((SOURCE_BUNDLED, bundled_dir))
    ordered_roots.append((SOURCE_MANAGED, managed_dir))
    ordered_roots.append((SOURCE_AGENTS_PERSONAL, personal_agents))
    ordered_roots.append((SOURCE_AGENTS_PROJECT, project_agents))
    ordered_roots.append((SOURCE_WORKSPACE, workspace_skills))

    roots: list[tuple[str, Path]] = []
    seen_paths: set[Path] = set()
    for source, root in ordered_roots:
        if root in seen_paths:
            continue
        seen_paths.add(root)
        roots.append((source, root))
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
        raw_script_path = str(
            entry.frontmatter.get("command-script", entry.frontmatter.get("command_script", ""))
        ).strip()
        if dispatch_kind == "script" or (not dispatch_kind and raw_script_path):
            script_interpreter = str(
                entry.frontmatter.get(
                    "command-script-interpreter",
                    entry.frontmatter.get("command_script_interpreter", ""),
                )
            ).strip()
            if not raw_script_path:
                log.warning(
                    "Skill command requested script dispatch but command-script is missing",
                    skill_name=entry.name,
                )
            else:
                try:
                    base_dir = Path(entry.base_dir).resolve()
                    script_path = (base_dir / raw_script_path).resolve()
                    script_path.relative_to(base_dir)
                except Exception:
                    log.warning(
                        "Skill command script path resolves outside skill directory",
                        skill_name=entry.name,
                        script_path=raw_script_path,
                    )
                else:
                    dispatch = SkillCommandDispatch(
                        kind="script",
                        arg_mode="raw",
                        script_path=str(script_path),
                        script_interpreter=script_interpreter,
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
                        "script_path": command.dispatch.script_path,
                        "script_interpreter": command.dispatch.script_interpreter,
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
            kind = str(dispatch_raw.get("kind", "")).strip().lower() or "tool"
            tool_name = str(dispatch_raw.get("tool_name", "")).strip()
            script_path = str(dispatch_raw.get("script_path", "")).strip()
            if kind == "tool" and tool_name:
                dispatch = SkillCommandDispatch(
                    kind="tool",
                    tool_name=tool_name,
                    arg_mode=str(dispatch_raw.get("arg_mode", "raw")).strip() or "raw",
                )
            elif kind == "script" and script_path:
                dispatch = SkillCommandDispatch(
                    kind="script",
                    tool_name=tool_name,
                    arg_mode=str(dispatch_raw.get("arg_mode", "raw")).strip() or "raw",
                    script_path=script_path,
                    script_interpreter=str(dispatch_raw.get("script_interpreter", "")).strip(),
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
