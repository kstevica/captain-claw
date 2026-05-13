"""Persistent storage for agent-app manifests.

An *app manifest* is the contract between an agent (a Captain Claw
"Claw") and a renderer (today: the Flight-Deck app runtime in the
browser; tomorrow: voice, smart glasses, etc.). It declares the
entities, feeds, actions, and surfaces the agent exposes — and which
MCP server backs each tool call.

Storage is intentionally simple for Phase 1: one YAML or JSON file per
app under ``~/.captain-claw-fd/app_manifests/``. The filename
(``<agent_id>.yaml``) is authoritative for the id; the file body is
validated against :class:`AgentManifest` on load.

This lives on the Captain Claw side (not in the renderer) because the
manifest is part of the framework — renderers consume it but should
never own it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


# ── storage location ──────────────────────────────────────────────────


def _manifests_dir() -> Path:
    base = os.environ.get("CAPTAIN_CLAW_FD_HOME") or os.path.expanduser("~/.captain-claw-fd")
    p = Path(base) / "app_manifests"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── pydantic schema (mirrors flight-deck/src/app-runtime/types.ts) ────


class FieldRef(BaseModel):
    """Cross-entity reference type: ``{ "ref": "<entity_id>" }``."""
    ref: str


# Field types support either a string ("string", "text", ...) or a {ref: ...}.
# Pydantic v2 handles the union via a discriminator-less union.
FieldType = str | FieldRef


class EntityField(BaseModel):
    type: FieldType
    label: str | None = None
    values: list[str] | None = None
    primary: bool = False
    title: bool = False
    required: bool = False


class EntityDef(BaseModel):
    id: str
    label: str
    plural: str | None = None
    fields: dict[str, EntityField]
    default_view: str | None = None  # "card" | "row" | "summary"


class FeedDef(BaseModel):
    id: str
    label: str
    mcp_tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    returns: str
    surfaces: list[str] = Field(default_factory=list)
    refresh_seconds: int | None = None
    proactive: bool = False
    description: str | None = None


class ActionInputDef(BaseModel):
    type: FieldType
    label: str | None = None
    required: bool = False
    values: list[str] | None = None


class ActionDef(BaseModel):
    id: str
    label: str
    mcp_tool: str
    inputs: dict[str, ActionInputDef] = Field(default_factory=dict)
    nl_aliases: list[str] = Field(default_factory=list)
    surfaces: list[str] = Field(default_factory=list)
    prefill: dict[str, str] = Field(default_factory=dict)
    returns: str | None = None  # "markdown" | "entity" | "none"
    confirm: bool = False
    prominent: bool = False
    description: str | None = None


class SurfaceSection(BaseModel):
    type: str  # "feed" | "action" | "chat"
    id: str
    filter: dict[str, str] = Field(default_factory=dict)
    prefill: dict[str, str] = Field(default_factory=dict)
    prominent: bool = False


class SurfaceDef(BaseModel):
    id: str
    label: str | None = None
    layout: str  # "dashboard" | "list" | "entity" | "inbox" | "upload"
    entity: str | None = None
    sources: list[str] = Field(default_factory=list)
    sections: list[SurfaceSection] = Field(default_factory=list)
    accept: str | None = None        # upload: input accept filter (e.g. "image/*")
    multiple: bool = False           # upload: allow multi-file picker


class ChatDef(BaseModel):
    enabled: bool = False
    context_aware: bool = False
    default_actions: list[str] = Field(default_factory=list)


class AgentInfo(BaseModel):
    id: str
    name: str
    tagline: str | None = None
    mcp_server: str


class AgentManifest(BaseModel):
    manifest_version: int = 1
    agent: AgentInfo
    entities: dict[str, EntityDef] = Field(default_factory=dict)
    feeds: dict[str, FeedDef] = Field(default_factory=dict)
    actions: dict[str, ActionDef] = Field(default_factory=dict)
    surfaces: dict[str, SurfaceDef] = Field(default_factory=dict)
    chat: ChatDef | None = None
    home_surface: str | None = None

    @field_validator("manifest_version")
    @classmethod
    def _check_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported manifest_version: {v}")
        return v


# ── loader ────────────────────────────────────────────────────────────


def _load_one(path: Path) -> AgentManifest | None:
    """Parse and validate a single manifest file. Returns ``None`` on error."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
    except (yaml.YAMLError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return AgentManifest.model_validate(raw)
    except Exception:
        return None


def load_all() -> list[AgentManifest]:
    """Load every valid manifest in the manifests dir."""
    out: list[AgentManifest] = []
    seen: set[str] = set()
    for path in sorted(_manifests_dir().iterdir()):
        if path.suffix.lower() not in (".yaml", ".yml", ".json"):
            continue
        m = _load_one(path)
        if m is None or m.agent.id in seen:
            continue
        seen.add(m.agent.id)
        out.append(m)
    return out


def get(agent_id: str) -> AgentManifest | None:
    """Load a single manifest by agent id."""
    d = _manifests_dir()
    for ext in (".yaml", ".yml", ".json"):
        p = d / f"{agent_id}{ext}"
        if p.exists():
            return _load_one(p)
    # Fallback: scan and match by parsed agent.id
    for m in load_all():
        if m.agent.id == agent_id:
            return m
    return None


def list_summaries() -> list[dict[str, Any]]:
    """Return ``[{id, name, tagline}]`` for every loadable manifest."""
    return [
        {"id": m.agent.id, "name": m.agent.name, "tagline": m.agent.tagline}
        for m in load_all()
    ]


# ── write side ────────────────────────────────────────────────────────


def _safe_id(agent_id: str) -> str:
    """Allow only filesystem-safe characters in agent ids."""
    out = "".join(c for c in agent_id if c.isalnum() or c in ("-", "_"))
    if not out:
        raise ValueError(f"agent_id has no safe characters: {agent_id!r}")
    return out


def save(manifest: AgentManifest) -> Path:
    """Write a manifest to disk as ``<agent_id>.yaml``. Overwrites if present."""
    agent_id = _safe_id(manifest.agent.id)
    path = _manifests_dir() / f"{agent_id}.yaml"
    tmp = path.with_suffix(".yaml.tmp")
    payload = manifest.model_dump(exclude_none=False)
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(tmp, path)
    # Best-effort: remove any rival format file for the same id so the loader
    # doesn't end up with two definitions.
    for rival_ext in (".yml", ".json"):
        rival = _manifests_dir() / f"{agent_id}{rival_ext}"
        if rival.exists():
            try:
                rival.unlink()
            except OSError:
                pass
    return path


def delete(agent_id: str) -> bool:
    """Remove every manifest file for ``agent_id``. Returns True if any deleted."""
    agent_id = _safe_id(agent_id)
    removed = False
    for ext in (".yaml", ".yml", ".json"):
        p = _manifests_dir() / f"{agent_id}{ext}"
        if p.exists():
            try:
                p.unlink()
                removed = True
            except OSError:
                pass
    return removed
