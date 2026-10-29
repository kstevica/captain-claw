"""Pass an onboarded agent on: its onboarding, packaged with a new archetype.

"Pass it on" on the Onboarding page packages what the agent was taught:
- its documents: the knowledge files, read or not, with their sheets as tables;
- its lessons: rules, when-to-ask, templates, glossary, facts, and what it was taught by hand;
- its skills;
- the questions of its last exam.

The package is kept by Flight Deck (``onboarding-packages/<id>/``), and a new
archetype of its owner's names it. The archetype also carries the agent's
role, instructions, tools and mode.

An agent made from that archetype by anyone it is shared with starts with all
of it as its base: what its new user teaches it adds to that, and can't change
it (agent_lessons). Its owner's own agents made from it start with it as their
own, as it was (the base of an admin's stays the base). The exam's questions
wait in its Exams tab, to be asked when its user wants (the answers and scores
stay with the agent they were asked of).

    POST /fd/agent-package/{slug}     package it, with a new archetype (role, description, include)
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import shutil
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck import agent_lessons
from captain_claw.flight_deck import agent_preset as ap
from captain_claw.flight_deck import agent_preset_tables as ptables
from captain_claw.flight_deck.auth import get_db, get_managed_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/agent-package", tags=["agent-package"])

_FILE = "package.json"
_SAFE_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]*\.(?:md|txt)")     # what knowledge/ keeps as a document
_SHARED_FILES = ("INDEX.md", "glossary.md")              # written from the record with the documents and glossary
_QUESTION = ("id", "kind", "question", "expected", "points", "doc", "doc_title", "doc_name", "title", "path", "key",
             "read")


def packages_dir() -> Path:
    from captain_claw.flight_deck import server as srv
    return srv.DATA_DIR / "onboarding-packages"


def _package_dir(package_id: str) -> Path | None:
    if not re.fullmatch(r"[a-f0-9]{16}", package_id or ""):
        return None
    return packages_dir() / package_id


def _skills_dir(agent_dir: Path) -> Path:
    return agent_dir / "data" / "home-config-parent" / ".captain-claw" / "skills"


def _base(item: Any) -> Any:
    return {**item, "layer": "base"} if isinstance(item, dict) else item


def _as_base(record: dict) -> dict:
    """The packaged record, everything in it the base (for someone else's agent)."""
    return {**record, **{g: [_base(i) for i in record.get(g) or []]
                         for g in (*ap._KEPT_GROUPS, "lessons", "knowledge", "tables")},
            "base_skills": list(record.get("skills") or [])}


def _read_tables(agent_dir: Path, names: list[str]) -> list[dict]:
    """The agent's tables named, read whole from its datastore (read-only: it may be running)."""
    from captain_claw.flight_deck.data_library import (
        TABLE_PREFIX,
        _columns,
        _connect,
        datastore_file,
    )
    path = datastore_file(agent_dir)
    if not names or not path.is_file():
        return []
    out: list[dict] = []
    try:
        with closing(_connect(path)) as c:
            for name in names:
                internal = TABLE_PREFIX + name
                if not c.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                                 (internal,)).fetchone():
                    continue
                cols = _columns(c, name)
                quoted = ", ".join('"' + col["name"].replace('"', '""') + '"' for col in cols)
                rows = [list(r) for r in c.execute(
                    f'SELECT {quoted} FROM "{internal}" LIMIT {ptables.MAX_ROWS}')]
                out.append({"name": name, "columns": cols, "rows": rows})
    except sqlite3.Error:
        log.warning("agent package: its tables couldn't be read", agent=agent_dir.name, exc_info=True)
    return out


def _last_exam(agent_dir: Path) -> dict | None:
    """The questions of the agent's newest exam that finished."""
    best: dict | None = None
    for f in (agent_dir / "exams").glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("state") == "done" and data.get("questions") and (
                best is None or float(data.get("finished") or 0) > float(best.get("finished") or 0)):
            best = data
    if best is None:
        return None
    return {"size": best.get("size") or "", "focus": best.get("focus") or "",
            "questions": [{k: q[k] for k in _QUESTION if k in q} for q in best["questions"] if isinstance(q, dict)]}


def build(slug: str, owner_id: str, agent_name: str, include: dict[str, bool]) -> dict:
    """Package the agent's onboarding; the package as kept (``package.json``),
    with its ``id`` — as it is now (unpack makes it the base of someone else's agent)."""
    from captain_claw.flight_deck import server as srv
    agent_dir = srv.DATA_DIR / slug
    entry = srv._load_process_registry().get(slug) or {}
    workspace = Path(entry["workspace"]) if entry.get("workspace") else agent_dir / "data" / "workspace"
    record = ap.earlier_setup(agent_dir, workspace)
    folder = workspace / "knowledge"
    package_id = secrets.token_hex(8)
    target = packages_dir() / package_id
    (target / "knowledge").mkdir(parents=True, exist_ok=True)

    knowledge: list[dict] = []
    tables: list[dict] = []
    if include.get("docs"):
        # Every document its knowledge holds, as its knowledge tool lists them — with what the record says of
        # each (one an earlier setup, or the agent itself, left there has no entry yet: it gets one).
        from captain_claw.tools.knowledge import KnowledgeTool
        by_file = {e["file"]: e for e in record.get("knowledge") or [] if isinstance(e, dict) and e.get("file")}
        for d in KnowledgeTool().documents(folder) if folder.is_dir() else []:
            e = by_file.get(d.file) or {}
            stem = Path(d.file).stem
            files = [n for n in dict.fromkeys([d.file, *d.files, *(e.get("files") or []), f"{stem}.map.md"])
                     if isinstance(n, str) and _SAFE_FILE.fullmatch(n) and (folder / n).is_file()]
            if d.file not in files:
                continue
            for n in files:
                shutil.copy2(folder / n, target / "knowledge" / n)
            knowledge.append(e or {
                "doc": d.file, "file": d.file, "title": d.title, "summary": d.summary, "use": "",
                "parts": max(1, sum(1 for n in files if re.search(r"-part-\d+\.md$", n))),
                "map": f"{stem}.map.md" if f"{stem}.map.md" in files else "", "files": files})
        named = [t["name"] for t in record.get("tables") or [] if isinstance(t, dict) and t.get("name")]
        got = {t["name"]: t for t in _read_tables(agent_dir, named)}
        tables = [{**t, "columns": got[t["name"]]["columns"], "rows": got[t["name"]]["rows"]}
                  for t in record.get("tables") or [] if isinstance(t, dict) and t.get("name") in got]
    skills: list[str] = []
    if include.get("skills"):
        # Every skill in its folder: a setup's, and any it has besides.
        src = _skills_dir(agent_dir)
        for name in sorted(p.name for p in src.iterdir() if p.is_dir()) if src.is_dir() else []:
            if ap.skill_name(name) == name and (src / name / "SKILL.md").is_file():
                shutil.copytree(src / name, target / "skills" / name, dirs_exist_ok=True)
                skills.append(name)
    lessons = bool(include.get("lessons"))
    out_record: dict[str, Any] = {
        **{g: list(record.get(g) or []) if lessons else [] for g in ap._KEPT_GROUPS},
        "lessons": list(record.get("lessons") or []) if lessons else [],
        "facts": int(record.get("facts") or 0) if lessons else 0,
        "knowledge": knowledge, "skills": skills,
        "base_skills": [n for n in record.get("base_skills") or [] if n in skills],
        "tables": [{k: v for k, v in t.items() if k not in ("columns", "rows")}
                   | {"columns": [c["name"] for c in t["columns"]], "rows": len(t["rows"])} for t in tables],
    }
    for n in _SHARED_FILES:
        wanted = include.get("docs") if n == "INDEX.md" else lessons
        if wanted and (folder / n).is_file():
            shutil.copy2(folder / n, target / "knowledge" / n)
    exam = _last_exam(agent_dir) if include.get("exam") and knowledge else None
    package = {
        "id": package_id, "owner": owner_id, "made": time.time(),
        "from": {"slug": slug, "name": agent_name},
        "record": out_record,
        "tables": [{"name": t["name"], "doc": t.get("doc") or "", "sheet": t.get("sheet"),
                    "columns": t["columns"], "rows": t["rows"]} for t in tables],
        "exam": exam,
        "counts": {"docs": len(knowledge), "tables": len(tables), "skills": len(skills),
                   "lessons": len(agent_lessons.from_record(out_record)),
                   "questions": len((exam or {}).get("questions") or [])},
    }
    (target / _FILE).write_text(json.dumps(package, ensure_ascii=False, default=str), encoding="utf-8")
    return package


def load(package_id: str) -> dict | None:
    where = _package_dir(package_id)
    try:
        data = json.loads((where / _FILE).read_text(encoding="utf-8")) if where else None
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def forget(package_id: str) -> None:
    where = _package_dir(package_id)
    if where is not None:
        shutil.rmtree(where, ignore_errors=True)


def unpack(package_id: str, owner: str, agent_dir: Path, workspace: Path, agent_name: str,
           agent_owner: str = "") -> str | None:
    """A new agent's onboarding from a package — its documents, skills,
    lessons and tables (left to make at its first start), and its exam's
    questions waiting to be asked: the base of someone else's agent, its
    own of its owner's (*agent_owner*). Only for a package of the
    archetype's owner, and only into an agent with no onboarding of its own
    yet. The block its instructions end with; None when nothing was
    unpacked."""
    package = load(package_id)
    where = _package_dir(package_id)
    if not package or where is None or str(package.get("owner") or "") != owner:
        log.warning("agent package: not unpacked (gone, or not the archetype owner's)", package=package_id)
        return None
    if ap.read_record(agent_dir):
        return None
    record = dict(package.get("record") or {})
    if agent_owner != owner:
        record = _as_base(record)
    folder = workspace / "knowledge"
    if (where / "knowledge").is_dir() and any((where / "knowledge").iterdir()):
        folder.mkdir(parents=True, exist_ok=True)
        for f in (where / "knowledge").iterdir():
            if _SAFE_FILE.fullmatch(f.name):
                shutil.copy2(f, folder / f.name)
    skills_dir = _skills_dir(agent_dir)
    for name in record.get("skills") or []:
        if isinstance(name, str) and name and ap.skill_name(name) == name and (where / "skills" / name).is_dir():
            shutil.copytree(where / "skills" / name, skills_dir / name, dirs_exist_ok=True)
    if package.get("tables"):
        ptables.leave_tables(agent_dir, [], package["tables"])
    blocks = ap.rewrite_lessons(agent_dir, workspace, record)
    exam = package.get("exam") or {}
    if exam.get("questions"):
        from captain_claw.flight_deck import agent_exam as ex
        job = ex.ExamJob(id=secrets.token_hex(8), agent=agent_dir.name, agent_name=agent_name,
                         size=str(exam.get("size") or ex.DEFAULT_SIZE), focus=str(exam.get("focus") or ""),
                         state="ready", phase="asking", steps=len(exam["questions"]),
                         questions=[dict(q) for q in exam["questions"]],
                         notes=[f"The questions {(package.get('from') or {}).get('name') or 'the agent it was made from'}"
                                " was examined on, passed on with its archetype."])
        ex.save(job)
    log.info("agent package: unpacked", package=package_id, agent=agent_dir.name, **(package.get("counts") or {}))
    return blocks["block"]


async def unpack_on_spawn(onboarding: dict, slug: str, agent_dir: Path, workspace: Path, agent_name: str,
                          instructions: str, agent_owner: str = "") -> str:
    """Spawn's hook (server._spawn_process_locked): the agent's instructions,
    ending with its onboarding's block when its archetype passed one on.
    Never fails the spawn."""
    try:
        block = await asyncio.to_thread(unpack, str(onboarding.get("package") or ""), str(onboarding.get("owner") or ""),
                                        agent_dir, workspace, agent_name, agent_owner)
    except Exception:
        log.warning("agent package: unpacking failed", agent=slug, exc_info=True)
        return instructions
    return ap.with_block(ap.strip_block(instructions or ""), block) if block else instructions


# ── Route ─────────────────────────────────────────────────────────────


class PackageBody(BaseModel):
    role: str = Field(..., max_length=120)
    description: str = Field("", max_length=1000)
    docs: bool = True
    lessons: bool = True
    skills: bool = True
    exam: bool = True


def _agent_tools(agent_dir: Path) -> list[str]:
    import yaml
    try:
        cfg = yaml.safe_load((agent_dir / "data" / "home-config-parent" / ".captain-claw" / "config.yaml")
                             .read_text(encoding="utf-8")) or {}
        tools = ((cfg.get("tools") or {}).get("enabled")) if isinstance(cfg, dict) else None
    except Exception:
        return []
    return [str(t) for t in tools] if isinstance(tools, list) else []


@router.post("/{slug}")
async def package_agent(slug: str, body: PackageBody, user: dict = Depends(get_managed_user)) -> dict:
    """Package the agent's onboarding with a new archetype of its owner's —
    the agent's role, instructions (without its onboarding's block, which
    the package gives an agent made from it), tools, mode and tier — and
    return the archetype."""
    from captain_claw.config import without_retired_tools
    from captain_claw.flight_deck import server as srv
    from captain_claw.flight_deck.archetype_routes import _slugify
    from captain_claw.flight_deck.archetypes import merged_archetypes

    owner_id = ap._owner(user)
    entry = ap._own_process(slug, owner_id)
    role = " ".join(body.role.split())
    if not role:
        raise HTTPException(400, "Give the archetype a role.")
    if not (body.docs or body.lessons or body.skills):
        raise HTTPException(400, "Pass on its documents, lessons or skills — at least one.")
    archetype_owner = owner_id or str(user.get("id") or "")
    db = get_db()
    taken = {str(a.get("id") or "") for a in await merged_archetypes(db, archetype_owner or None)}
    stem = _slugify(role) or "onboarded"
    aid = next(f"{stem}-{n}" if n > 1 else stem for n in range(1, 1000) if (f"{stem}-{n}" if n > 1 else stem) not in taken)
    agent_dir = srv.DATA_DIR / slug
    name = str(entry.get("name") or slug)
    package = await asyncio.to_thread(build, slug, archetype_owner, name, body.model_dump())
    # Where the agent came from fills what it doesn't say itself.
    came_from = next((a for a in await merged_archetypes(db, archetype_owner or None)
                      if a.get("id") == str(entry.get("archetype") or "")), {})
    instructions = (await srv._owner_agent_instructions(owner_id, "process")).get(slug, "") if owner_id else ""
    mode_file = agent_dir / "data" / "home-config-parent" / ".captain-claw" / "cognitive_mode.txt"
    mode = mode_file.read_text(encoding="utf-8").strip() if mode_file.is_file() else ""
    tier = str(entry.get("model_tier") or came_from.get("tier") or "balanced")
    data = {
        "id": aid, "role": role, "family": str(came_from.get("family") or "Onboarded"),
        "description": " ".join(body.description.split()) or f"Made from {name}, with its onboarding.",
        "cognitive_mode": mode or str(came_from.get("cognitive_mode") or "neutra"),
        "tier": tier if tier in ("reason", "balanced", "fast", "longctx", "micro") else "balanced",
        "tools": without_retired_tools(_agent_tools(agent_dir) or list(came_from.get("tools") or [])),
        "fleet_instructions": ap.strip_block(instructions or str(came_from.get("fleet_instructions") or "")),
        "keywords": list(came_from.get("keywords") or []),
        "lead": False, "reliability_seed": 0.7,
        "runtime": str(came_from.get("runtime") or ""),
        "onboarding": {"package": package["id"], "from": name, "made": package["made"], "counts": package["counts"]},
    }
    try:
        await db.create_user_archetype(archetype_owner, aid, json.dumps(data))
    except Exception:
        await asyncio.to_thread(forget, package["id"])
        raise
    log.info("agent package: passed on", agent=slug, archetype=aid, **package["counts"])
    return {"archetype": data, "counts": package["counts"]}
