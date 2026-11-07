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
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from captain_claw.flight_deck.agent_bundle import (
    archive,
    catalog,
    export,
    provenance,
    secrets_box,
    stores,
)
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
            "model": _model_line(agent.get("settings") if isinstance(agent.get("settings"), dict) else {}),
            "model_tier": str(agent.get("model_tier") or ""),
            "has_secrets": bundle.has("secrets.enc")}


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


def spawn_config(agent: dict[str, Any], name: str, *, keep_model: bool = True):
    """The spawn request for a new agent from the bundle's ``fd/agent.json``.
    Not *keep_model*: one of the importer's tiers gives the model, its context
    sizes and vision (``apply_tier``)."""
    from captain_claw.flight_deck.server import AgentConfig

    s = agent.get("settings") if isinstance(agent.get("settings"), dict) else {}
    if not keep_model:
        s = {k: v for k, v in s.items() if k in ("tools", "runtime")}
    fields: dict[str, Any] = {
        "name": name,
        "description": str(agent.get("description") or "")[:2000],
        "cognitive_mode": str(agent.get("cognitive_mode") or "neutra")[:40],
        "fleet_instructions": str(agent.get("fleet_instructions") or "")[:64_000],
        "grid_memory_tags": [str(t) for t in agent.get("grid_tags") or []][:20],
        "grid_recall_mode": str(agent.get("grid_recall") or ""),
        "runtime": "mrav" if s.get("runtime") == "mrav" else "",
    }
    if keep_model:
        fields["vision"] = _vision(s.get("vision"))
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


async def tiers(owner_id: str) -> list[dict[str, str]]:
    """The importer's model tiers (no keys): what "use one of your models" offers."""
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers
    try:
        tiers_map, _ = await _load_owner_tiers(get_db(), owner_id)
    except Exception:
        return []
    return [{"tier": str(t), "provider": str(c.get("provider") or ""), "model": str(c.get("model") or "")}
            for t, c in (tiers_map or {}).items() if isinstance(c, dict) and c.get("model")]


async def apply_tier(config, owner_id: str, tier: str) -> None:
    """Run the new agent on the importer's *tier* — as an archetype spawn on that tier would."""
    from captain_claw.flight_deck import server as srv
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers
    tiers_map, owner_env = await _load_owner_tiers(get_db(), owner_id)
    tcfg = (tiers_map or {}).get(tier) or {}
    if not tcfg.get("model"):
        raise archive.BundleError(f"You have no “{tier}” model set up — pick another, or keep its own.")
    srv._apply_tier_config(config, tcfg, tier)
    srv._apply_owner_env(config, owner_env)


def open_secrets(bundle: archive.Bundle, passphrase: str) -> dict[str, Any]:
    """The bundle's sealed keys ({} when it has none or no passphrase is given)."""
    if not passphrase or not bundle.has("secrets.enc"):
        return {}
    with bundle_tmp(bundle, "secrets.enc") as path:
        return secrets_box.open_sealed(path.read_bytes(), passphrase)


@contextmanager
def bundle_tmp(bundle: archive.Bundle, name: str) -> Iterator[Path]:
    """An entry of the bundle, unpacked (checked) to a temporary file for a moment."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "entry"
        bundle.extract(name, path)
        yield path


def apply_secrets(config, keys: dict[str, Any]) -> None:
    """The model key and .env keys on the spawn; the APIs' credentials wait for the unpack."""
    from captain_claw.flight_deck import server as srv
    model_key = str(keys.get("model_api_key") or "")
    env = keys.get("env") if isinstance(keys.get("env"), dict) else {}
    own = set(srv._provider_key_env_names(config.provider) or ())
    if not (config.provider_api_key or "").strip():
        config.provider_api_key = model_key or next((str(env[n]) for n in own if env.get(n)), "")
    have = {str(ev.get("key") or "") for ev in config.env_vars}
    for name, value in env.items():
        if name not in have and name not in own and str(value):
            config.env_vars.append({"key": str(name), "value": str(value)})


def restore_columns(agent_dir: Path, columns: dict[str, Any]) -> int:
    """Put the APIs' credentials back into the restored sessions.db."""
    db = agent_dir / catalog.BY_KEY["sessions"].path
    if not columns or not db.is_file():
        return 0
    allowed = {f"{t}.{c}" for t, cols in catalog.BY_KEY["sessions"].secret_columns.items() for c in cols}
    n = 0
    with closing(sqlite3.connect(db)) as c:
        for key, values in columns.items():
            if key not in allowed or not isinstance(values, dict):
                continue
            table, col = key.split(".", 1)
            for row_id, value in values.items():
                n += c.execute(f'UPDATE "{table}" SET "{col}" = ? WHERE id = ?', (str(value), str(row_id))).rowcount
        c.commit()
    return n


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
    if ref.get("columns"):
        await asyncio.to_thread(restore_columns, agent_dir, ref["columns"])
    await asyncio.to_thread(note_package, Path(ref["path"]), agent_dir, "reset", {}, base=not ref.get("same_person"))
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
    package = provenance.bundle_package(bundle.manifest)
    if package and agent.get("fleet_instructions"):       # the spawn set them: an update may replace them
        provenance.note_instructions(srv.DATA_DIR / slug, package["id"], str(agent["fleet_instructions"])[:64_000])
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
        same = str(bundle.manifest.get("owner_ref") or "") == export.person_ref(owner_id)
        made = _schedules(jobs, slug, owner_id, out["notes"], same_person=same)
        out["schedules"] = len(made)
        if package and made:
            provenance.add_schedules(srv.DATA_DIR / slug, package["id"], made)
    links = bundle.read_json("fd/connections.json") if bundle.has("fd/connections.json") else None
    out["connections"] = [{"kind": str(c.get("kind") or ""), "label": str(c.get("label") or "")}
                          for c in links or [] if isinstance(c, dict)][:30] if isinstance(links, list) else []
    return out


def _schedules(jobs: list, slug: str, owner_id: str, notes: list[str], *, same_person: bool = True) -> list[str]:
    """Its scheduled tasks, switched off — its owner turns them back on. For
    someone else they deliver to the agent's chat only: the phone numbers,
    chats and addresses they went to were the other person's. The ids made."""
    from captain_claw.flight_deck import fd_scheduler

    made: list[str] = []
    for job in jobs[:200]:
        if not isinstance(job, dict):
            continue
        fields = {k: job.get(k) for k in export._JOB_FIELDS if k not in ("enabled", "prompt_author")}
        if not same_person:
            fields.update({"delivery_kind": "agent", "delivery_target": "",
                           "delivery": {"chat": True, "email": False, "telegram": False, "email_to": ""}})
        try:
            row = fd_scheduler.get_store().create(
                owner_id=owner_id, prompt_author=str(job.get("prompt_author") or "human"),
                **{**fields, "agent_slug": slug, "agent_auth": "", "enabled": False})
            made.append(str(row["id"]))
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


# ── into an agent that already exists (merge.py) ────────────────────────────

def apply_to_existing(path: Path, agent_dir: Path, mode: str, columns: dict[str, Any] | None = None, *,
                      same_person: bool = True, owner_id: str = "") -> dict[str, Any]:
    """The bundle's stores and files into *agent_dir* (stopped): *mode* is
    "reset", "append" or "update" (merge.MODES) — update first takes the
    package's earlier version out (provenance). What was added and kept."""
    from captain_claw.flight_deck.agent_bundle import merge

    bundle = archive.open_bundle(path)
    package = provenance.bundle_package(bundle.manifest)
    taken_out: dict[str, int] = {}
    if mode == "update":
        earlier = provenance.imported(agent_dir).get(package["id"]) if package else None
        if not earlier:
            raise archive.BundleError("This agent holds no earlier version of that package — add it or replace instead.")
        if int(earlier.get("version") or 0) >= package["version"]:
            raise archive.BundleError(f"It already has version {earlier.get('version')} of this package.")
        taken_out = provenance.take_out(agent_dir, earlier.get("items") or {}, owner_id=owner_id)
    categories = set(bundle.manifest.get("categories") or []) & set(catalog.CATEGORIES)
    parts = bundle.manifest.get("parts") if isinstance(bundle.manifest.get("parts"), dict) else {}
    stored: dict[str, dict[str, dict[str, int]]] = {}
    notes: list[str] = []
    for name in bundle.names("stores/"):
        key = name[len("stores/"):].removesuffix(".db")
        part = catalog.BY_KEY.get(key)
        if part is None or part.kind != "sqlite" or name != f"stores/{key}.db":
            continue
        dest = agent_dir / part.path
        if not _inside(agent_dir, dest.parent):
            continue
        listed = (parts.get(key) or {}).get("tables") if isinstance(parts.get(key), dict) else None
        tables = [str(t) for t in listed] if isinstance(listed, dict) else []
        with bundle_tmp(bundle, name) as tmp:
            try:
                if dest.is_file():
                    stored[key] = merge.merge_store(tmp, dest, tables, mode)
                else:                               # a store it hasn't got yet: the bundle's, as a new agent's
                    stores.check(tmp)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(tmp, dest)
                    stored[key] = {t: {"added": n, "skipped": 0} for t, n in stores.counts(dest).items()}
            except (stores.StoreError, sqlite3.Error) as exc:
                notes.append(f"{part.label} left as it was: {exc}")
    files = merge.merge_files(bundle, agent_dir, categories, mode, _inside, base=bool(package) and not same_person)
    if columns:
        restore_columns(agent_dir, columns)
    done = {"stores": stored, "files": files, "notes": notes, "categories": sorted(categories),
            "package": package, "taken_out": taken_out}
    note_package(path, agent_dir, mode, done, base=False)
    return done


def note_package(path: Path, agent_dir: Path, mode: str, done: dict[str, Any], *, base: bool) -> None:
    """When the bundle is a version of a package: what it put into the agent
    (provenance), so a later version can update it; for a new agent of
    someone else's package (*base*), its onboarding becomes the base layer."""
    bundle = archive.open_bundle(path)
    package = provenance.bundle_package(bundle.manifest)
    if not package:
        return
    if base:
        provenance.base_layer_on_new(agent_dir)
    # a new agent holds all of it; an existing one what this import added (or, reset, replaced)
    items = provenance.items_of_merge(done, agent_dir) if done else provenance.items_of_new_agent(bundle, agent_dir)
    provenance.record(agent_dir, package, items, made_by=str(bundle.manifest.get("made_by") or ""))


def headline(stored: dict[str, dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    """The counts the dialog leads with: insights, intentions, playbooks, sessions… added and already there."""
    picks = {"insights": ("insights", "insights"), "intentions": ("intentions", "intentions"),
             "intuitions": ("intuitions", "intuitions"), "playbooks": ("sessions", "playbooks"),
             "contacts": ("sessions", "contacts"), "sessions": ("sessions", "sessions"), "topics": ("topics", "topics")}
    out = {}
    for label, (store, table) in picks.items():
        got = (stored.get(store) or {}).get(table)
        if got and (got.get("added") or got.get("skipped")):
            out[label] = {"added": int(got.get("added") or 0), "skipped": int(got.get("skipped") or 0)}
    return out


async def fd_side_into_existing(path: Path, slug: str, owner_id: str, agent: dict[str, Any], mode: str,
                                tag: str, earlier: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flight Deck's side into an existing agent: instructions (reset: the
    bundle's; append: only when it has none), chat log (reset: replaced;
    append: beside its own, in the lane ``imported-<tag>``), scheduled tasks
    (reset: replaced; append: added unless the same task is there) — all off."""
    from captain_claw.flight_deck import fd_scheduler
    from captain_claw.flight_deck import server as srv
    from captain_claw.flight_deck.auth import _fd_auth_enabled, get_db

    bundle = await asyncio.to_thread(archive.open_bundle, path)
    out: dict[str, Any] = {"chat_messages": 0, "schedules": 0, "notes": [], "connections": []}
    package = provenance.bundle_package(bundle.manifest)
    text = str(agent.get("fleet_instructions") or "")
    if owner_id and text:
        mine = (await srv._owner_agent_instructions(owner_id, "process")).get(slug, "")
        # update: the package's own instructions, while nobody changed them here
        set_by_earlier = bool(earlier) and provenance.text_sha(mine) == str(
            ((earlier or {}).get("items") or {}).get("instructions") or "")
        if mode == "reset" or not mine.strip() or (mode == "update" and set_by_earlier):
            await srv._set_owner_agent_instructions(owner_id, "process", slug, text[:64_000])
            if package:
                provenance.note_instructions(srv.DATA_DIR / slug, package["id"], text[:64_000])
        elif mode == "update":
            out["notes"].append("Its instructions were changed here, so the new version's were left out.")
    chats = bundle.read_json("fd/chats.json") if bundle.has("fd/chats.json") else None
    if isinstance(chats, dict) and owner_id and _fd_auth_enabled():
        db = get_db()
        if mode == "reset":
            await db.drop_chat_log(owner_id, f"proc-{slug}")
        out["chat_messages"] = await db.import_chat_log(
            owner_id, f"proc-{slug}", str(agent.get("name") or slug), chats,
            lane="" if mode == "reset" else f"imported-{tag}")         # append, update: beside its own
    jobs = bundle.read_json("fd/schedules.json") if bundle.has("fd/schedules.json") else None
    if isinstance(jobs, list):
        store = fd_scheduler.get_store()
        if mode == "reset":
            for job in store.list(owner_id or None):
                if str(job.get("agent_slug") or "") == slug:
                    store.delete(str(job["id"]))
        else:
            have = {(_n(j.get("name")), _n(j.get("prompt"))) for j in store.list(owner_id or None)
                    if str(j.get("agent_slug") or "") == slug}
            jobs = [j for j in jobs if isinstance(j, dict) and (_n(j.get("name")), _n(j.get("prompt"))) not in have]
        same = str(bundle.manifest.get("owner_ref") or "") == export.person_ref(owner_id)
        made = _schedules(jobs, slug, owner_id, out["notes"], same_person=same)
        out["schedules"] = len(made)
        if package and made:
            provenance.add_schedules(srv.DATA_DIR / slug, package["id"], made)
    links = bundle.read_json("fd/connections.json") if bundle.has("fd/connections.json") else None
    if isinstance(links, list):
        out["connections"] = [{"kind": str(c.get("kind") or ""), "label": str(c.get("label") or "")}
                              for c in links if isinstance(c, dict)][:30]
    return out


def _n(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()
