"""Make a new agent from a bundle.

The agent is spawned the normal way (``server._spawn_process_locked``): the
agent-count limit, its port and web token, the owner marker, the owner's
profile, instructions and provider key are all as for any new agent. The
bundle's files and stores are unpacked inside the spawn, before the process
starts (``unpack_on_spawn``, like an onboarding package); Flight Deck's side —
chat log, scheduled tasks, registry fields — once it is registered
(``after_spawn``).

Only paths the catalogue lists are written, below the agent's own folder.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from captain_claw.flight_deck.agent_bundle import archive, catalog, export, stores
from captain_claw.logging import get_logger

log = get_logger(__name__)

_ECO_TARGETS = (f"{catalog.HC}/eco_mode.txt", "data/home-config/.captain-claw/eco_mode.txt",
                "data/home-config/eco_mode.txt")


def inspect(path: Path) -> dict[str, Any]:
    """What the import dialog shows of an uploaded bundle (raises BundleError)."""
    bundle = archive.open_bundle(path)
    agent = bundle.read_json("fd/agent.json")
    if not isinstance(agent, dict) or not str(agent.get("name") or "").strip():
        raise archive.BundleError("The bundle doesn't say which agent it holds.")
    return {**export.summary(bundle.manifest),
            "model": _model_line(agent.get("settings") if isinstance(agent.get("settings"), dict) else {})}


def _model_line(settings: dict[str, Any]) -> dict[str, str]:
    return {"provider": str(settings.get("provider") or ""), "model": str(settings.get("model") or "")}


def free_name(wanted: str) -> str:
    """*wanted*, or ``wanted 2``… — a name whose folder and registry entry are free."""
    from captain_claw.flight_deck import server as srv

    wanted = " ".join(str(wanted or "").split())[:80] or "Agent"
    registry = srv._load_process_registry()
    for n in range(1, 200):
        name = wanted if n == 1 else f"{wanted} {n}"
        slug = srv._slug(name)
        if slug not in registry and not (srv.DATA_DIR / slug).exists():
            return name
    raise archive.BundleError("No free name for the agent — give it another one.")


def _vision(value: Any) -> str:
    if value is True:
        return "on"
    if value is False:
        return "off"
    return "auto"


def spawn_config(agent: dict[str, Any], name: str):
    """The spawn request for a new agent from the bundle's ``fd/agent.json``."""
    from captain_claw.flight_deck.server import AgentConfig

    s = agent.get("settings") if isinstance(agent.get("settings"), dict) else {}
    fields: dict[str, Any] = {
        "name": name,
        "description": str(agent.get("description") or "")[:2000],
        "cognitive_mode": str(agent.get("cognitive_mode") or "neutra")[:40],
        "fleet_instructions": str(agent.get("fleet_instructions") or "")[:64_000],
        "grid_memory_tags": [str(t) for t in agent.get("grid_tags") or []][:20],
        "grid_recall_mode": str(agent.get("grid_recall") or ""),
        "vision": _vision(s.get("vision")),
        "runtime": "mrav" if s.get("runtime") == "mrav" else "",
    }
    for key, cast in (("provider", str), ("model", str), ("base_url", str),
                      ("temperature", float), ("max_tokens", int), ("max_context", int)):
        if s.get(key) not in (None, ""):
            try:
                fields[key] = cast(s[key])
            except (TypeError, ValueError):
                pass
    if isinstance(s.get("tools"), list) and s["tools"]:
        fields["tools"] = [str(t) for t in s["tools"]][:200]
    if isinstance(s.get("allowed"), list):
        fields["allowed_models"] = [m for m in s["allowed"] if isinstance(m, dict)][:50]
    try:
        return AgentConfig(**fields)
    except Exception as exc:  # pydantic: a value this deck can't take
        raise archive.BundleError(f"The bundle's agent settings can't be used here ({exc}).") from exc


def _inside(root: Path, target: Path) -> bool:
    """Is *target* — through its nearest folder that exists, links resolved — below *root*?"""
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        probe.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def unpack(path: Path, agent_dir: Path) -> list[str]:
    """Write the bundle's files and stores into *agent_dir*; notes for the person."""
    bundle = archive.open_bundle(path)
    agent_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    skipped = 0
    for name in bundle.names("agent/"):
        rel = name[len("agent/"):]
        if catalog.part_for_path(rel) is None:
            skipped += 1
            continue
        dest = agent_dir / rel
        if not _inside(agent_dir, dest.parent):
            skipped += 1
            continue
        bundle.extract(name, dest)
    for name in bundle.names("stores/"):
        key = name[len("stores/"):].removesuffix(".db")
        part = catalog.BY_KEY.get(key)
        if part is None or part.kind != "sqlite" or name != f"stores/{key}.db":
            skipped += 1
            continue
        dest = agent_dir / part.path
        if not _inside(agent_dir, dest.parent):
            skipped += 1
            continue
        tmp = dest.with_name(f".{dest.name}.import")
        bundle.extract(name, tmp)
        try:
            stores.check(tmp)
        except stores.StoreError as exc:
            tmp.unlink(missing_ok=True)
            notes.append(f"{part.label} left out: {exc}")
            continue
        for leftover in (dest, dest.with_name(dest.name + "-wal"), dest.with_name(dest.name + "-shm")):
            leftover.unlink(missing_ok=True)
        tmp.replace(dest)
    if skipped:
        notes.append(f"{skipped} file(s) in the bundle weren't the agent's own and were left out.")
    return notes


def write_eco(agent_dir: Path, value: str) -> None:
    if value not in ("on", "off"):
        return
    for rel in _ECO_TARGETS:
        target = agent_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")


async def unpack_on_spawn(ref: dict[str, Any], slug: str, agent_dir: Path, owner_id: str) -> None:
    """The spawn's hook: the bundle in *ref* into the new agent's folder, before it starts."""
    notes = await asyncio.to_thread(unpack, Path(ref["path"]), agent_dir)
    write_eco(agent_dir, str(ref.get("eco_mode") or ""))
    ref.setdefault("notes", []).extend(notes)
    log.info("agent bundle unpacked", agent=slug, owner=owner_id or "-", notes=len(notes))


async def after_spawn(path: Path, slug: str, name: str, owner_id: str, agent: dict[str, Any]) -> dict[str, Any]:
    """Flight Deck's side of the new agent: registry fields, chat log, scheduled tasks."""
    from captain_claw.flight_deck import server as srv

    registry = srv._load_process_registry()
    if slug in registry:
        for key in ("archetype", "tier", "model_tier"):
            if agent.get(key):
                registry[slug][key] = str(agent[key])
        srv._save_process_registry(registry)

    bundle = await asyncio.to_thread(archive.open_bundle, path)
    out: dict[str, Any] = {"chat_messages": 0, "schedules": 0, "notes": []}
    chats = bundle.read_json("fd/chats.json") if bundle.has("fd/chats.json") else None
    if isinstance(chats, dict) and owner_id:
        from captain_claw.flight_deck.auth import _fd_auth_enabled, get_db
        if _fd_auth_enabled():
            try:
                out["chat_messages"] = await get_db().import_chat_log(owner_id, f"proc-{slug}", name, chats)
            except Exception as exc:
                log.warning("agent bundle: chat log not imported", agent=slug, error=type(exc).__name__)
                out["notes"].append("Its chat log couldn't be brought back.")
    jobs = bundle.read_json("fd/schedules.json") if bundle.has("fd/schedules.json") else None
    if isinstance(jobs, list):
        out["schedules"] = _schedules(jobs, slug, owner_id, out["notes"])
    return out


def _schedules(jobs: list, slug: str, owner_id: str, notes: list[str]) -> int:
    """Its scheduled tasks, switched off — its owner turns them back on."""
    from captain_claw.flight_deck import fd_scheduler

    made = 0
    for job in jobs[:200]:
        if not isinstance(job, dict):
            continue
        fields = {k: job.get(k) for k in export._JOB_FIELDS if k not in ("enabled", "prompt_author")}
        try:
            fd_scheduler.get_store().create(
                owner_id=owner_id, prompt_author=str(job.get("prompt_author") or "human"),
                **{**fields, "agent_slug": slug, "agent_auth": "", "enabled": False})
            made += 1
        except Exception as exc:
            notes.append(f"Scheduled task “{str(job.get('name') or '')[:60]}” left out: {exc}")
    return made


def needs_key(config) -> bool:
    """Does the new agent's model want a key it didn't get (none in its spawn,
    none in Flight Deck's environment)?"""
    from captain_claw.flight_deck import server as srv

    if (config.provider_api_key or "").strip():
        return False
    try:
        names = srv._provider_key_env_names(config.provider)
    except Exception:
        return False
    if not names or (config.base_url or "").strip():
        return False
    return not any(os.environ.get(n) for n in names)
