"""Make a bundle of one process agent.

What goes in is chosen by category (catalog.CATEGORIES). The agent's own
files are copied from its folder, its SQLite stores snapshotted
(stores.snapshot) and trimmed to the categories asked for; Flight Deck's side
— registry fields, model settings without keys, instructions, chat log,
scheduled tasks — goes in ``fd/``. Works whether the agent runs, sleeps or is
off: nothing is asked of the agent itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import shutil
import sqlite3
import tempfile
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from captain_claw.flight_deck.agent_bundle import (
    archive,
    catalog,
    provenance,
    review,
    secrets_box,
    stores,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

# Template: who it is and what it was taught. Trained: and what it learned
# since. Full backup: everything, conversations and files too.
PRESETS: dict[str, tuple[str, ...]] = {
    "template": ("identity", "onboarding"),
    "trained": ("identity", "onboarding", "memory", "data", "schedules"),
    "full": catalog.CATEGORIES,
}


@dataclass
class Options:
    """What the export review decided, beyond the categories."""
    # Insight categories that go (None: all of them); single insights left out.
    insight_categories: set[str] | None = None
    exclude_insights: set[str] = field(default_factory=set)
    # Its contacts (people it knows) — off for anyone else, on in a backup.
    contacts: bool = True
    # A full backup with this passphrase carries the agent's keys, sealed.
    passphrase: str = ""
    # A named package: this export is its next version (provenance) — an agent
    # that holds an earlier one can be updated from it.
    package: str = ""

# config.yaml: only these travel (never a key, token, port, path or address of this deck).
_MODEL_FIELDS = ("provider", "model", "temperature", "max_tokens", "base_url", "vision", "allowed")


def bundles_dir() -> Path:
    from captain_claw.flight_deck import server as srv
    return srv.DATA_DIR / "agent-bundles"


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("captain-claw")
    except Exception:
        return ""


def agent_settings(agent_dir: Path) -> dict[str, Any]:
    """What a new agent made from the bundle is spawned with, read from the
    agent's config.yaml — the model without its key, tools, context window,
    runtime. An address on this machine (localhost) is kept: the importer's
    deck decides what it means."""
    try:
        cfg = yaml.safe_load((agent_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        cfg = {}
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    out: dict[str, Any] = {k: model[k] for k in _MODEL_FIELDS if k in model}
    allowed = out.get("allowed")
    if isinstance(allowed, list):     # a per-session model list: names only, never a key in it
        out["allowed"] = [{k: v for k, v in m.items() if "key" not in str(k).lower()}
                          for m in allowed if isinstance(m, dict)]
    context = cfg.get("context") if isinstance(cfg.get("context"), dict) else {}
    if context.get("max_tokens"):
        out["max_context"] = context["max_tokens"]
    tools = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
    if isinstance(tools.get("enabled"), list):
        out["tools"] = [str(t) for t in tools["enabled"]]
    mrav = cfg.get("mrav") if isinstance(cfg.get("mrav"), dict) else {}
    if mrav.get("enabled"):
        out["runtime"] = "mrav"
    return out


def _read_text(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8")[:limit].strip()
    except (OSError, UnicodeDecodeError):
        return ""


def _copy_dir(src: Path, dst: Path, part: catalog.Part, notes: list[str]) -> int:
    """Copy a folder part (files only, links never followed); the count copied."""
    copied = 0
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if any(seg in catalog.SKIP_NAMES for seg in rel.parts):
            continue
        if any(rel.as_posix() == s or rel.as_posix().startswith(s + "/") for s in part.skip):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        name = f"agent/{part.path}/{rel.as_posix()}"
        if not archive.safe_name(name):
            notes.append(f"Left out a file whose name can't travel: {rel.as_posix()[:120]}")
            continue
        target = dst / part.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def stage(agent_dir: Path, staging: Path, categories: set[str], *, workspace_moved: bool,
          options: Options | None = None) -> dict[str, Any]:
    """Copy the agent's parts for *categories* into ``staging/agent`` and
    ``staging/stores``; what was taken, per part."""
    options = options or Options()
    taken: dict[str, Any] = {}
    notes: list[str] = []
    for part in catalog.PARTS:
        src = agent_dir / part.path
        if part.kind == "sqlite":
            if not src.is_file():
                continue
            snap = staging / "stores" / f"{part.key}.db"
            stores.snapshot(src, snap)
            all_tables = stores.tables(snap)
            keep = catalog.wanted_tables(part, all_tables, categories)
            if part.key == "sessions" and not options.contacts:
                keep = [t for t in keep if t not in ("contacts", "contact_mentions")]
            if part.key == "insights" and "insights" in keep:
                review.filter_insights(snap, categories=options.insight_categories, exclude=options.exclude_insights)
            if not keep:
                snap.unlink()
                continue
            stores.keep_only(snap, set(keep), blank=part.secret_columns)
            counts = stores.counts(snap)
            if not any(counts.get(t) for t in keep):
                snap.unlink()
                continue
            taken[part.key] = {"kind": "sqlite", "category": part.category, "path": part.path,
                               "tables": {t: counts.get(t, 0) for t in keep},
                               "categories": sorted({catalog.table_category(part, t) for t in keep})}
            continue
        if part.category not in categories:
            continue
        if workspace_moved and part.path.startswith(catalog.WORKSPACE):
            continue
        if part.kind == "file":
            if src.is_file() and not src.is_symlink():
                (staging / "agent" / part.path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, staging / "agent" / part.path)
                taken[part.key] = {"kind": "file", "category": part.category, "path": part.path}
        elif src.is_dir() and not src.is_symlink():
            n = _copy_dir(src, staging / "agent", part, notes)
            if n:
                taken[part.key] = {"kind": "dir", "category": part.category, "path": part.path, "files": n}
    if workspace_moved and ({"files", "onboarding"} & categories):
        notes.append("Its workspace is a folder outside its own (Code mode), so its files and documents stayed out.")
    return {"parts": taken, "notes": notes}


async def _chat_log(owner_id: str, slug: str) -> dict[str, Any]:
    from captain_claw.flight_deck.auth import _fd_auth_enabled, get_db
    if not owner_id or not _fd_auth_enabled():
        return {}
    try:
        return await get_db().export_chat_log(owner_id, f"proc-{slug}")
    except Exception as exc:
        log.warning("agent bundle: its chat log couldn't be read", agent=slug, error=type(exc).__name__)
        return {}


_JOB_FIELDS = ("name", "schedule", "tz", "prompt", "prompt_author", "delivery_kind", "delivery_target",
               "delivery", "enabled", "ignore_quiet_hours")


def _schedules(owner_id: str, slug: str) -> list[dict[str, Any]]:
    from captain_claw.flight_deck import fd_scheduler
    out: list[dict[str, Any]] = []
    try:
        for job in fd_scheduler.get_store().list(owner_id or None):
            if str(job.get("agent_slug") or "") != slug or job.get("flow_id"):
                continue
            out.append({k: job.get(k) for k in _JOB_FIELDS})
    except Exception as exc:
        log.warning("agent bundle: its scheduled tasks couldn't be read", agent=slug, error=type(exc).__name__)
    return out


def person_ref(owner_id: str) -> str:
    """Who made a bundle, on which deck — the same value only for the same
    person on the same deck (a hash; neither is readable from it)."""
    from captain_claw.flight_deck.agent_secret import get_or_create_agent_secret
    return hashlib.sha256(f"{get_or_create_agent_secret()}:{owner_id}".encode()).hexdigest()[:24]


def _read_config(agent_dir: Path) -> dict[str, Any]:
    try:
        cfg = yaml.safe_load((agent_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


# .env names that are Flight Deck's wiring, not the agent's own keys.
_ENV_WIRING = ("FD_", "CLAW_", "OLLAMA_BASE_URL", "HOME", "PATH")


def gather_secrets(agent_dir: Path) -> dict[str, Any]:
    """The agent's own keys: its .env, its model key, and the credentials of
    the APIs it learned (sessions.db) — for a sealed full backup only."""
    env: dict[str, str] = {}
    try:
        for line in (agent_dir / ".env").read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            key = key.strip()
            if sep and key and value.strip() and not key.startswith(_ENV_WIRING):
                env[key] = value.strip()
    except (OSError, UnicodeDecodeError):
        pass
    model = _read_config(agent_dir).get("model")
    model_key = str((model or {}).get("api_key") or "") if isinstance(model, dict) else ""
    columns: dict[str, dict[str, str]] = {}
    src = agent_dir / catalog.BY_KEY["sessions"].path
    if src.is_file():
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "s.db"
            stores.snapshot(src, snap)
            with closing(sqlite3.connect(snap)) as c:
                for table, cols in catalog.BY_KEY["sessions"].secret_columns.items():
                    have = {r[1] for r in c.execute(f'PRAGMA table_info("{table}")')}
                    if "id" not in have:
                        continue
                    for col in cols:
                        if col in have:
                            got = {str(i): str(v) for i, v in c.execute(
                                f'SELECT id, "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL AND "{col}" != \'\'')}
                            if got:
                                columns[f"{table}.{col}"] = got
    return {"env": env, "model_api_key": model_key, "columns": columns}


async def build(slug: str, owner_id: str, entry: dict[str, Any], *, categories: set[str],
                preset: str = "", made_by: str = "", options: Options | None = None,
                keep_in: str = "exports") -> dict[str, Any]:
    """Make the bundle; its record (id, file name, manifest) — the file is
    ``bundles_dir()/exports/<id>.captain`` until it is fetched or swept."""
    from captain_claw.flight_deck import server as srv

    categories = catalog.normalize_categories(categories)
    options = options or Options()
    if options.passphrase and preset != "full":
        raise archive.BundleError("Only a full backup carries its keys.")
    agent_dir = srv.DATA_DIR / slug
    if not agent_dir.is_dir():
        raise archive.BundleError("The agent has no folder on this deck.")
    bundle_id = secrets.token_hex(12)
    staging = bundles_dir() / "tmp" / bundle_id
    out = bundles_dir() / keep_in / f"{bundle_id}{archive.EXTENSION}"
    hc = agent_dir / catalog.HC
    workspace_moved = bool(entry.get("workspace"))
    try:
        staged = await asyncio.to_thread(stage, agent_dir, staging, categories, workspace_moved=workspace_moved,
                                         options=options)
        instructions = (await srv._owner_agent_instructions(owner_id, "process")).get(slug, "") if owner_id else ""
        agent = {
            "name": str(entry.get("name") or slug),
            "slug": slug,
            "description": str(entry.get("description") or ""),
            "archetype": str(entry.get("archetype") or ""),
            "tier": str(entry.get("tier") or ""),
            "model_tier": str(entry.get("model_tier") or ""),
            "grid_tags": list(entry.get("grid_tags") or []),
            "grid_recall": str(entry.get("grid_recall") or ""),
            "settings": agent_settings(agent_dir),
            "cognitive_mode": _read_text(hc / "cognitive_mode.txt", 40) or "neutra",
            "eco_mode": _read_text(hc / "eco_mode.txt", 10),
            "fleet_instructions": instructions,
        }
        fd_dir = staging / "fd"
        fd_dir.mkdir(parents=True, exist_ok=True)
        (fd_dir / "agent.json").write_text(json.dumps(agent, ensure_ascii=False, indent=1), encoding="utf-8")
        links = review.connections(slug, owner_id, agent["settings"], _read_config(agent_dir))
        (fd_dir / "connections.json").write_text(json.dumps(links, ensure_ascii=False, indent=1), encoding="utf-8")
        sealed = False
        if options.passphrase:
            payload = await asyncio.to_thread(gather_secrets, agent_dir)
            (staging / "secrets.enc").write_bytes(secrets_box.seal(payload, options.passphrase))
            sealed = True
        counts: dict[str, int] = {}
        if "conversations" in categories:
            chats = await _chat_log(owner_id, slug)
            if chats.get("sessions"):
                (fd_dir / "chats.json").write_text(json.dumps(chats, ensure_ascii=False), encoding="utf-8")
                counts["chat_messages"] = len(chats.get("messages") or [])
        if "schedules" in categories:
            jobs = _schedules(owner_id, slug)
            if jobs:
                (fd_dir / "schedules.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
                counts["schedules"] = len(jobs)
        manifest = {
            "format": archive.FORMAT,
            "format_version": archive.FORMAT_VERSION,
            "made": time.time(),
            "captain_version": _version(),
            "made_by": made_by,
            "preset": preset,
            "categories": sorted(categories),
            "agent": {"name": agent["name"], "slug": slug, "description": agent["description"]},
            "parts": staged["parts"],
            "counts": counts,
            "notes": staged["notes"],
            "secrets": sealed,
            "connections": links,
            "owner_ref": person_ref(owner_id),
            "insight_categories": sorted(options.insight_categories) if options.insight_categories is not None else None,
            "excluded_insights": len(options.exclude_insights),
            "contacts": options.contacts,
            "package": provenance.next_version(agent_dir, options.package) if options.package.strip() else None,
        }
        manifest = await asyncio.to_thread(archive.write, staging, manifest, out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    record = {"id": bundle_id, "owner": owner_id, "slug": slug,
              "file_name": file_name(agent["name"], preset), "size": out.stat().st_size, "made": manifest["made"]}
    (out.with_suffix(".json")).write_text(json.dumps(record), encoding="utf-8")
    return {**record, "manifest": summary(manifest)}


def file_name(agent_name: str, preset: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in " -_" else "-" for ch in agent_name).strip() or "agent"
    day = time.strftime("%Y-%m-%d")
    kind = {"full": "backup", "template": "template", "trained": "trained"}.get(preset, preset or "export")
    return f"{stem} {kind} {day}{archive.EXTENSION}"


def summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """What the dialogs show of a bundle: who, when, which categories, how much."""
    parts = manifest.get("parts") or {}
    per_category: dict[str, dict[str, int]] = {}
    for key, p in parts.items():
        if not isinstance(p, dict):
            continue
        part = catalog.BY_KEY.get(key)
        if p.get("kind") == "sqlite":
            for table, n in (p.get("tables") or {}).items():
                cat = catalog.table_category(part, table) if part else str(p.get("category") or "")
                per_category.setdefault(cat, {})[f"{key}.{table}"] = int(n or 0)
        else:
            per_category.setdefault(str(p.get("category") or ""), {})[key] = int(p.get("files") or 1)
    agent = manifest.get("agent") if isinstance(manifest.get("agent"), dict) else {}
    return {
        "agent": {"name": str(agent.get("name") or ""), "description": str(agent.get("description") or "")},
        "made": manifest.get("made"),
        "captain_version": str(manifest.get("captain_version") or ""),
        "made_by": str(manifest.get("made_by") or ""),
        "preset": str(manifest.get("preset") or ""),
        "categories": [c for c in catalog.CATEGORIES if c in (manifest.get("categories") or [])],
        "size": int(manifest.get("size") or 0),
        "counts": {**_headline(per_category), **(manifest.get("counts") or {})},
        "notes": [str(n) for n in manifest.get("notes") or []][:20],
        "secrets": bool(manifest.get("secrets")),
        "connections": [{"kind": str(c.get("kind") or ""), "label": str(c.get("label") or "")}
                        for c in manifest.get("connections") or [] if isinstance(c, dict)][:30],
        "package": provenance.bundle_package(manifest),
    }


_HEADLINES = {
    "insights": "insights.insights", "intentions": "intentions.intentions", "sessions": "sessions.sessions",
    "intuitions": "intuitions.intuitions", "playbooks": "sessions.playbooks", "contacts": "sessions.contacts",
    "topics": "topics.topics",
}


def _headline(per_category: dict[str, dict[str, int]]) -> dict[str, int]:
    flat = {k: v for d in per_category.values() for k, v in d.items()}
    out = {name: flat[key] for name, key in _HEADLINES.items() if flat.get(key)}
    for key in ("knowledge", "skills", "reflections", "workspace"):
        if flat.get(key):
            out[key] = flat[key]
    tables = sum(1 for k in flat if k.startswith("datastore.ds_"))
    if tables:
        out["data_tables"] = tables
    return out
