"""Anonymize files where they are: a VFS file, a whole VFS folder, a Library file.

The anonymized copy goes next to the original — ``Contract.anonymized.docx``
beside the file, ``Clients_anonymized/`` beside the folder. When the person
can't write there (a Drive or OneDrive mount, a read-only link, a folder
shared view-only, a shared ``@alias`` folder) it goes to an ``Anonymized``
folder of their own instead, under the source folder's name. Names in the
copied paths are tokenized too, into the same vault.

A folder runs as a background job (one per person) the page polls; a file
that can't be anonymized is skipped and listed, never copied as it is.

Routes (prefix ``/fd/anonymize``):

* ``POST /vfs/file``    — {project, path, owner?} + the run options
* ``POST /vfs/folder``  — {project, path ("" = the whole folder), owner?} + options → job
* ``GET  /vfs/job``     — the person's folder job (or ``{"running": false}``)
* ``POST /vfs/job/cancel``
* ``POST /library/{file_id}`` — a Library file; the copy lands beside it in the agent's workspace
* ``POST /agent/vfs/folder``, ``GET /agent/vfs/job`` — the same folder job for an agent's tool,
  in its person's VFS and vault
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from captain_claw.flight_deck import anonymize_routes as ar
from captain_claw.flight_deck import vfs_routes as vr
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/anonymize", tags=["anonymize"])

OWN_FOLDER = "Anonymized"  # where copies from read-only places go
MAX_FOLDER_FILES = 2000
SUFFIX = "_anonymized"


# ── where a copy goes ───────────────────────────────────────────────


@dataclasses.dataclass
class Place:
    """A VFS location: whose root, which folder (project), which path in it."""
    uid: str
    project: str
    path: str  # project-relative, posix, "" = the folder itself
    owner: str = ""  # as the UI names it ("" = the caller's own)

    def view(self) -> dict[str, str]:
        return {"project": self.project, "path": self.path, "owner": self.owner}


async def _writable_root(caller: str, project: str, owner: str) -> str | None:
    """Whose root a copy beside (project, …) would be written to, or None when
    the caller can't write there."""
    from captain_claw import pack_access

    if pack_access.is_pack_project(project):
        return None
    try:
        oid = await vr._eff_owner(caller, project, owner, write=True)
        vr._assert_writable(oid, project)
    except HTTPException:
        return None
    if vr._is_mount(vr._link_entry(oid, project)):
        return None
    return oid


def _unique(target: Path) -> Path:
    """*target*, or ``name (2).ext`` … when it already exists (never overwrite)."""
    if not target.exists():
        return target
    stem, suffix = (target.stem, target.suffix) if target.is_file() or target.suffix else (target.name, "")
    for n in range(2, 1000):
        cand = target.with_name(f"{stem} ({n}){suffix}")
        if not cand.exists():
            return cand
    raise HTTPException(409, f"too many copies of {target.name} already")


def _new_project(uid: str, name: str) -> str:
    """A folder (project) name not yet taken in *uid*'s root, as the VFS spells it."""
    from captain_claw.vfs import safe_name

    base = safe_name(name, fallback="anonymized")
    for n in range(1, 1000):
        cand = base if n == 1 else f"{base}-{n}"
        if not vr._project_root(uid, cand).exists():
            return cand
    raise HTTPException(409, f"too many copies of {base} already")


def _rel(place_root: Path, target: Path) -> str:
    return target.relative_to(place_root).as_posix()


def _write(place: Place, rel: str, blob: bytes) -> str:
    """Write *blob* at *rel* under *place* (no overwrite); the path written."""
    vr._refuse_reserved(rel)
    root = vr._project_root(place.uid, place.project)
    target = _unique(vr._resolve(place.uid, place.project, rel))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    written = _rel(root, target)
    vr._dm().on_write(place.uid, place.project, written)
    return written


def _join(*parts: str) -> str:
    return "/".join(p.strip("/") for p in parts if p and p.strip("/"))


async def _read(caller: str, project: str, path: str, owner: str) -> tuple[bytes, str]:
    serve, name = await vr._served_file(caller, project, path, owner)
    size = serve.stat().st_size
    if size > ar.MAX_FILE_BYTES:
        raise HTTPException(413, f"{name} is over {ar.MAX_FILE_BYTES // (1024 * 1024)} MB")
    return await asyncio.to_thread(serve.read_bytes), name


def _options(body: dict[str, Any]) -> dict[str, Any]:
    keys = ("vault", "mode", "keep", "extra", "lane", "generalize", "kinds", "default_cc")
    return {k: body[k] for k in keys if k in body}


# ── a single VFS file ───────────────────────────────────────────────


def _one_way_state(data: dict[str, Any]):
    """One in-memory state for a one_way job (None for a vault run)."""
    from captain_claw.anonymize import engine

    return engine.VaultState() if str(data.get("mode") or "reversible") == "one_way" else None


@router.post("/vfs/file")
async def vfs_file(body: dict[str, Any] = Body(...), user: dict = Depends(get_current_user)) -> dict[str, Any]:
    caller = str(user["id"])
    actor = f"person:{caller}"
    project, path, owner = str(body.get("project") or ""), str(body.get("path") or ""), str(body.get("owner") or "")
    if not project or not path:
        raise HTTPException(422, "project and path are required")
    blob, name = await _read(caller, project, path, owner)
    data = _options(body)
    data.setdefault("vault", project)
    state = _one_way_state(data)
    out_blob, out = await ar.anonymize_blob(caller, name, blob, data, actor=actor, source="vfs",
                                            one_way_state=state)
    parent = str(PurePosixPath(path).parent)
    parent = "" if parent == "." else parent
    oid = await _writable_root(caller, project, owner)
    if oid is not None:
        place, beside = Place(oid, project, "", "" if oid == caller else oid), True
        rel = _join(parent, out["filename"])
    else:
        place, beside = Place(caller, OWN_FOLDER, ""), False
        folders = await ar.anonymize_names(caller, [project, *[p for p in parent.split("/") if p]], data,
                                           actor=actor, one_way_state=state)
        rel = _join(*folders, out["filename"])
    written = await asyncio.to_thread(_write, place, rel, out_blob)
    out.pop("preview", None)
    out.update({"written": {**place.view(), "path": written}, "beside": beside, "size": len(out_blob)})
    return out


# ── a whole VFS folder (a job) ──────────────────────────────────────


@dataclasses.dataclass
class FolderJob:
    id: str
    source: dict[str, str]
    dest: dict[str, str] = dataclasses.field(default_factory=dict)  # set once the copies are placed
    vault: str = ""
    beside: bool = True
    phase: str = "anonymizing"  # → placing → done
    running: bool = True
    cancelled: bool = False
    total: int = 0
    done: int = 0
    current: str = ""
    skipped: list[dict[str, str]] = dataclasses.field(default_factory=list)
    failed: list[dict[str, str]] = dataclasses.field(default_factory=list)
    entities: dict[str, int] = dataclasses.field(default_factory=dict)
    message: str = ""
    started_at: float = dataclasses.field(default_factory=time.time)
    finished_at: float = 0.0
    task: asyncio.Task | None = None

    def view(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "task"}
        out["skipped"] = self.skipped[:200]
        out["failed"] = self.failed[:200]
        out["skipped_count"], out["failed_count"] = len(self.skipped), len(self.failed)
        return out


_JOBS: dict[str, FolderJob] = {}


def _supported(name: str) -> bool:
    """Files a folder run copies. Pictures stay out (a folder of photos would
    come back as files of OCR text); a data room reads them on purpose."""
    suffix = PurePosixPath(name).suffix.lower()
    return suffix in ar.FILE_KINDS and suffix not in ar._IMAGE_EXT


def _a_copy(rel: str) -> bool:
    """Already an anonymized copy (``x.anonymized.docx``, or inside an earlier
    ``…_anonymized/`` folder): left out of a folder run."""
    p = PurePosixPath(rel)
    return p.stem.endswith(".anonymized") or any(part.endswith(SUFFIX) for part in p.parts[:-1])


async def _folder_dest(caller: str, src: dict[str, str], data: dict[str, Any], state,
                       actor: str) -> tuple[Place, str, bool]:
    """(place, path of the new folder in it, beside?) for a copy of the source
    folder — named after the files ran, with what they taught the vault."""
    project, owner = src["project"], src["owner"]
    parts = [p for p in src["path"].split("/") if p]
    oid = await _writable_root(caller, project, owner)
    if oid is not None and parts:
        (name,) = await ar.anonymize_names(caller, [parts[-1]], data, actor=actor, one_way_state=state)
        place = Place(oid, project, "", "" if oid == caller else oid)
        base = _join(*parts[:-1], name + SUFFIX)
        # never into an earlier copy: "PERSON_1_anonymized (2)"
        base = _rel(vr._project_root(place.uid, project), _unique(vr._resolve(place.uid, project, base)))
        return place, base, True
    if oid is not None:  # the whole folder: a new folder of the caller's own beside it
        (name,) = await ar.anonymize_names(caller, [project], data, actor=actor, one_way_state=state)
        return Place(caller, _new_project(caller, name + SUFFIX), ""), "", True
    names = await ar.anonymize_names(caller, [project, *parts], data, actor=actor, one_way_state=state)
    place = Place(caller, OWN_FOLDER, "")
    base = _rel(vr._project_root(caller, OWN_FOLDER), _unique(vr._resolve(caller, OWN_FOLDER, _join(*names))))
    return place, base, False


async def _run_folder(job: FolderJob, caller: str, files: list[str], src: dict[str, str],
                      data: dict[str, Any], actor: str) -> None:
    """Two phases: anonymize every file into a scratch folder outside the VFS,
    then name the copy's folders (the vault knows the names in them by then)
    and place the files. A stopped job places what it finished."""
    import tempfile

    state = _one_way_state(data)
    made: list[tuple[str, str, Path]] = []  # (source sub-folder, new file name, scratch file)
    try:
        with tempfile.TemporaryDirectory(prefix="fd-anonymize-") as scratch:
            for i, rel in enumerate(files):
                if job.cancelled:
                    job.message = "Stopped."
                    break
                job.current = rel
                try:
                    blob, name = await _read(caller, src["project"], _join(src["path"], rel), src["owner"])
                    out_blob, out = await ar.anonymize_blob(caller, name, blob, data, actor=actor,
                                                            source="vfs-folder", one_way_state=state)
                    tmp = Path(scratch) / str(i)
                    await asyncio.to_thread(tmp.write_bytes, out_blob)
                    sub = str(PurePosixPath(rel).parent)
                    made.append(("" if sub == "." else sub, out["filename"], tmp))
                    for kind, n in (out.get("summary") or {}).items():
                        job.entities[kind] = job.entities.get(kind, 0) + int(n)
                    job.done += 1
                except HTTPException as exc:
                    job.failed.append({"path": rel, "error": str(exc.detail)[:300]})
                except Exception as exc:  # noqa: BLE001 — one file never stops the folder
                    log.warning("Folder anonymization: a file failed", error=type(exc).__name__)
                    job.failed.append({"path": rel, "error": type(exc).__name__})
            if not made:
                return
            job.phase, job.current = "placing", ""
            place, base, job.beside = await _folder_dest(caller, src, data, state, actor)
            job.dest = {**place.view(), "path": base}
            subs = sorted({sub for sub, _n, _t in made if sub})
            named = {sub: _join(*await ar.anonymize_names(caller, sub.split("/"), data, actor=actor,
                                                          one_way_state=state)) for sub in subs}
            for sub, new_name, tmp in made:
                blob = await asyncio.to_thread(tmp.read_bytes)
                await asyncio.to_thread(_write, place, _join(base, named.get(sub, ""), new_name), blob)
    except Exception as exc:  # noqa: BLE001 — shown on the job
        log.warning("Folder anonymization failed", error=type(exc).__name__)
        job.message = f"Stopped: {getattr(exc, 'detail', None) or type(exc).__name__}"
    finally:
        job.phase, job.running, job.current = "done", False, ""
        job.finished_at, job.task = time.time(), None


async def start_folder(caller: str, body: dict[str, Any], *, actor: str) -> dict[str, Any]:
    """Start anonymizing a folder of *caller*'s VFS (theirs, or shared with them)."""
    current = _JOBS.get(caller)
    if current is not None and current.running:
        raise HTTPException(409, "A folder is already being anonymized — wait for it or stop it.")
    project, owner = str(body.get("project") or ""), str(body.get("owner") or "")
    path = str(body.get("path") or "").strip("/")
    if not project:
        raise HTTPException(422, "project is required")
    oid = await vr._eff_owner(caller, project, owner, write=False)
    root = vr._resolve(oid, project, path)
    if not root.is_dir():
        raise HTTPException(404, "folder not found")
    data = _options(body)
    data.setdefault("vault", PurePosixPath(path).name if path else project)
    if str(data.get("mode") or "reversible") not in ("reversible", "one_way"):
        raise HTTPException(422, "mode is 'reversible' or 'one_way'")
    listed, seen = await asyncio.to_thread(vr._walk_files, root, MAX_FOLDER_FILES + 1)
    files = sorted(rel for rel, size, _m in listed
                   if _supported(rel) and size <= ar.MAX_FILE_BYTES and not _a_copy(rel))
    src = {"project": project, "path": path, "owner": owner}
    job = FolderJob(id=uuid.uuid4().hex, source=src,
                    vault="" if data.get("mode") == "one_way" else str(data["vault"]), total=len(files))
    for rel, size, _m in listed:
        if _a_copy(rel):
            job.skipped.append({"path": rel, "reason": "already an anonymized copy"})
        elif not _supported(rel):
            job.skipped.append({"path": rel, "reason": "can't be anonymized (pictures, archives, other kinds)"})
        elif size > ar.MAX_FILE_BYTES:
            job.skipped.append({"path": rel, "reason": f"over {ar.MAX_FILE_BYTES // (1024 * 1024)} MB"})
    if seen > MAX_FOLDER_FILES:
        files = files[:MAX_FOLDER_FILES]
        job.total = len(files)
        job.message = f"Only the first {MAX_FOLDER_FILES:,} files are done — split the folder for the rest."
    _JOBS[caller] = job
    job.task = asyncio.get_running_loop().create_task(_run_folder(job, caller, files, src, data, actor))
    return job.view()


@router.post("/vfs/folder")
async def vfs_folder(body: dict[str, Any] = Body(...), user: dict = Depends(get_current_user)) -> dict[str, Any]:
    caller = str(user["id"])
    return await start_folder(caller, body, actor=f"person:{caller}")


# An agent's folder run: the vault and the VFS are its person's — the member
# on a shared agent's member turn, else its owner (anonymize_routes._agent_person).

@router.post("/agent/vfs/folder")
async def agent_vfs_folder(request: Request) -> dict[str, Any]:
    person, actor = await ar._agent_person(request)
    return await start_folder(person, await ar._body(request), actor=actor)


@router.get("/agent/vfs/job")
async def agent_vfs_job(request: Request) -> dict[str, Any]:
    person, _actor = await ar._agent_person(request)
    job = _JOBS.get(person)
    return job.view() if job else {"running": False}


@router.get("/vfs/job")
async def vfs_job(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    job = _JOBS.get(str(user["id"]))
    return job.view() if job else {"running": False}


@router.post("/vfs/job/cancel")
async def vfs_job_cancel(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    job = _JOBS.get(str(user["id"]))
    if job is None:
        return {"running": False}
    job.cancelled = True
    return job.view()


# ── a Library file ──────────────────────────────────────────────────


@router.post("/library/{file_id}")
async def library_file(file_id: str, body: dict[str, Any] = Body(default={}),
                       user: dict = Depends(get_current_user)) -> dict[str, Any]:
    from captain_claw.flight_deck import file_library as fl

    caller = str(user["id"])
    row, agents = fl._owned_row(file_id, user)
    src = fl._resolve_path(row)
    if src is None:
        raise HTTPException(404, "file not found")
    if src.stat().st_size > ar.MAX_FILE_BYTES:
        raise HTTPException(413, f"{src.name} is over {ar.MAX_FILE_BYTES // (1024 * 1024)} MB")
    data = _options(body)
    data.setdefault("vault", agents.get(row["agent_slug"], {}).get("name") or row["agent_slug"])
    blob = await asyncio.to_thread(src.read_bytes)
    out_blob, out = await ar.anonymize_blob(caller, src.name, blob, data, actor=f"person:{caller}",
                                            source="library")
    target = _unique(src.with_name(out["filename"]))
    await asyncio.to_thread(target.write_bytes, out_blob)
    workspace = fl._workspace(row["agent_slug"])
    rel = target.relative_to(workspace).as_posix()
    try:  # list it in the Library now (its summary comes with the next scan)
        fl.get_store().sync_agent(row["agent_slug"], await asyncio.to_thread(fl.walk_workspace, row["agent_slug"]))
    except Exception as exc:  # noqa: BLE001
        log.debug("Library not refreshed after anonymizing", error=type(exc).__name__)
    out.pop("preview", None)
    out.update({"written": {"agent_slug": row["agent_slug"], "rel_path": rel, "filename": target.name},
                "size": len(out_blob)})
    return out
