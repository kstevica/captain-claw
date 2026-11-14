"""Running a data room (``docs/data-room-plan.md`` P1): every new file is
anonymized into the room's vault, the folders get their names, a final
sweep catches what the vault learned late, the text Claude will read is made
and indexed, and a leak check blocks any file that still holds a real value.

* **Anonymize** — the anonymizer's own file path (``anonymize_blob``): both
  lanes when a trusted model is set up, pictures replaced (their text read
  with OCR into alt text), scans read with OCR, the seed terms registered
  first. One file failing never stops the room.
* **Names** — folder names go through the vault after the files ran.
* **Sweep** — files are done one by one, so a name first found in file 300
  is still raw in file 12. Code alone (no model, no guessing at one-word
  names) runs again over every clean copy with the final vault, up to three
  rounds, until nothing changes.
* **Text** — markdown of every clean copy (alt text of pictures included),
  split into pages and indexed for search.
* **Leak check** — every real value in the vault (its spellings too, diacritic
  and case tolerant) is looked for in every text. A hit blocks the file; the
  report names tokens, never values.

One room runs at a time on a deck (the trusted model is the bottleneck); the
others wait. Progress is kept in ``rooms.db``, so a deck restart resumes.
The job record holds counts and file ids only — no names.
"""

from __future__ import annotations

import asyncio
import io
import posixpath
import re
import tempfile
import time
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import HTTPException
from lxml import etree

from captain_claw.anonymize import detectors, engine
from captain_claw.flight_deck import anonymize_routes as ar
from captain_claw.flight_deck import anonymize_vault as av
from captain_claw.flight_deck import rooms_intake as ri
from captain_claw.flight_deck import rooms_store as rs
from captain_claw.logging import get_logger

log = get_logger(__name__)

SWEEP_ROUNDS = 3
PAGE_CHARS = 6000          # text without its own pages is indexed in pieces this long
MIN_NEEDLE = 4             # shorter real values ("Ana") aren't looked for: too many false hits
SEED_KINDS = {"orgs": "ORG", "people": "PERSON", "products": "ORG", "domains": "ORG", "ids": "ID_NUMBER",
              "other": "DETAIL"}

_RUN_SLOT = asyncio.Semaphore(1)
_TASKS: dict[str, asyncio.Task] = {}
_STOP: set[str] = set()


# ── helpers ─────────────────────────────────────────────────────────


def seed_terms(seed: dict[str, Any]) -> list[dict[str, str]]:
    """The seed sheet as the anonymizer's extra terms."""
    out: list[dict[str, str]] = []
    for key, kind in SEED_KINDS.items():
        for value in seed.get(key) or []:
            text = re.sub(r"\s+", " ", str(value or "")).strip()
            if text and len(text) <= 300:
                out.append({"text": text, "kind": kind})
    return out[: ar.MAX_TERMS]


def _clean_name(anonymized: str) -> str:
    """``⟦…⟧ ugovor.anonymized.docx`` → ``⟦…⟧ ugovor.docx`` (a room holds only copies)."""
    return re.sub(r"\.anonymized(?=\.[^.]+$)", "", anonymized)


def _fold_loose(text: str) -> str:
    """Case, accents and spacing folded away, for the leak check."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", text).casefold()


def needles(state: Any) -> list[tuple[str, str | re.Pattern[str]]]:
    """(token, folded real value) for every value worth looking for — and, for a
    company of two words or more, (token, pattern) for its name with a short
    word put in ("plava rijeka prc capital"), as the anonymizer matches it too."""
    out: list[tuple[str, str | re.Pattern[str]]] = []
    seen: set[tuple[str, str]] = set()
    for e in state.entities:
        for value in [e.original, *e.forms, *getattr(e, "spellings", [])]:
            folded = _fold_loose(detectors.fold(value))
            if len(folded.replace(" ", "")) < MIN_NEEDLE or (e.token, folded) in seen:
                continue
            seen.add((e.token, folded))
            out.append((e.token, folded))
        if e.kind == "ORG":
            words = engine.loose_name_words(e.original)
            if words:
                between = rf"\W{{1,3}}(?:\w{{1,{engine.LOOSE_INSERTED_LEN}}}\W{{1,3}})?"
                out.append((e.token, re.compile(r"(?<!\w)" + between.join(map(re.escape, words)) + r"(?!\w)")))
    return out


def leaks(text: str, needle_list: list[tuple[str, str | re.Pattern[str]]]) -> list[str]:
    """Tokens whose real value is still in *text* (as a whole word)."""
    hay = _fold_loose(text)
    hits: list[str] = []
    for token, needle in needle_list:
        if token in hits:
            continue
        if isinstance(needle, re.Pattern):
            if needle.search(hay):
                hits.append(token)
            continue
        if needle not in hay:
            continue
        if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", hay):
            hits.append(token)
    return hits


def _alt_texts(blob: bytes) -> list[str]:
    """Picture text kept as alt text in a clean Office copy."""
    out: list[str] = []
    try:
        z = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        return out
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    for name in z.namelist():
        if not re.fullmatch(r"(word|ppt|xl)/.*\.xml", name):
            continue
        try:
            root = etree.fromstring(z.read(name), parser)
        except etree.XMLSyntaxError:
            continue
        for el in root.iter():
            if isinstance(el.tag, str) and etree.QName(el.tag).localname in ("docPr", "cNvPr"):
                descr = (el.get("descr") or "").strip()
                if "Text in the picture:" in descr:
                    out.append(descr)
    return out


def text_view(clean_name: str, blob: bytes) -> str:
    """What Claude reads of a clean copy: markdown."""
    from captain_claw.tools import document_extract as dx

    ext = PurePosixPath(clean_name).suffix.lower()
    if ext in (".docx", ".pptx", ".xlsx"):
        with tempfile.TemporaryDirectory(prefix="room-text-") as tmp:
            # The extractor heads the text with the file's name: the room's own.
            stem = re.sub(r"[^\w.\- ⟦⟧:]", "_", PurePosixPath(clean_name).stem)[:120] or "file"
            path = Path(tmp) / f"{stem}{ext}"
            path.write_bytes(blob)
            if ext == ".docx":
                text = dx._extract_docx_markdown(path)
            elif ext == ".pptx":
                text = dx._extract_pptx_markdown(path, 2000)
            else:
                text = dx._extract_xlsx_markdown(path, 20000)
        pictures = _alt_texts(blob)
        if pictures:
            text += "\n\n## Pictures (removed; their text kept)\n\n" + "\n".join(f"- {p}" for p in pictures)
        return text
    return blob.decode("utf-8", errors="replace")


def pages_of(text: str) -> list[tuple[str, str]]:
    """(page label, text) for the index: a PDF's pages, slides or sheets when
    the text has them, else pieces of about PAGE_CHARS."""
    marks = list(re.finditer(r"(?m)^## (Page|Slide|Sheet) ?([^\n]*)$", text))
    if len(marks) >= 2:
        out = []
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            out.append((f"{m.group(1)} {m.group(2).strip()}".strip(), text[m.start():end]))
        return out
    return [(str(i // PAGE_CHARS + 1), text[i:i + PAGE_CHARS]) for i in range(0, max(len(text), 1), PAGE_CHARS)]


async def source_root(room: dict[str, Any]) -> Path:
    """Where a room's folder files are read from (checked again on every run)."""
    src = room.get("source") or {}
    if src.get("kind") == "local":
        return Path(src["path"])
    if src.get("kind") == "vfs":
        from captain_claw.flight_deck import vfs_routes as vr

        oid = await vr._eff_owner(room["owner_id"], src["project"], src.get("owner", ""), write=False)
        return vr._resolve(oid, src["project"], src.get("path", ""))
    raise ri.IntakeError("this room has no folder")


async def read_original(room: dict[str, Any], f: dict[str, Any], cipher: Any) -> bytes:
    store = rs.store()
    if f["origin"] == "zip":
        return await asyncio.to_thread(store.read_raw, room["id"], f["id"], cipher)
    root = await source_root(room)
    return await asyncio.to_thread(ri.inside(root, f["path"]).read_bytes)


# ── the job ─────────────────────────────────────────────────────────


def running(room_id: str) -> bool:
    task = _TASKS.get(room_id)
    return task is not None and not task.done()


async def start(room: dict[str, Any], *, retry_failed: bool = False) -> None:
    if running(room["id"]):
        raise rs.RoomError("This room is already running.")
    store = rs.store()
    files = await store.files(room["id"])
    if not files:
        raise rs.RoomError("Add files first (a ZIP or a folder).")
    if retry_failed:
        for f in files:
            if f["state"] == "failed":
                await store.set_file(room["id"], f["id"], state="new", error="")
    _STOP.discard(room["id"])
    await store.update(room["id"], state="running", job={"phase": "queued", "queued_at": time.time()})
    _TASKS[room["id"]] = asyncio.get_running_loop().create_task(_run(room["id"]))


STOP_WAIT = 10.0  # seconds a stop waits for the run to wind down before answering


async def stop(room_id: str) -> bool:
    """Stop a room's run now — mid-file too (the trusted model's calls are
    cancelled). The file it was on goes back to "not run yet"; files already
    done stay done. Waits (briefly) until the room says it stopped."""
    task = _TASKS.get(room_id)
    if task is None or task.done():
        return False
    _STOP.add(room_id)
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), STOP_WAIT)
    except (asyncio.CancelledError, TimeoutError, Exception):  # noqa: BLE001 — the run tidies itself up
        pass
    return True


async def resume_all() -> int:
    """At deck start: rooms that were running carry on."""
    n = 0
    for room in await rs.store().in_state("running"):
        if not running(room["id"]):
            _TASKS[room["id"]] = asyncio.get_running_loop().create_task(_run(room["id"]))
            n += 1
    return n


async def _set_job(room_id: str, job: dict[str, Any]) -> None:
    job["updated_at"] = time.time()
    await rs.store().update(room_id, job=dict(job))


def _data(room: dict[str, Any], vault_id: str) -> dict[str, Any]:
    settings = room.get("settings") or {}
    data: dict[str, Any] = {"vault": vault_id, "mode": "reversible",
                            "pictures": "strip" if settings.get("pictures", "strip") == "strip" else "keep",
                            "extra": seed_terms(room.get("seed") or {}), "source": "data-room"}
    if settings.get("lane") in ar.LANES:
        data["lane"] = settings["lane"]
    return data


async def _run(room_id: str) -> None:
    store = rs.store()
    job: dict[str, Any] = {"phase": "queued"}
    try:
        async with _RUN_SLOT:
            room = await store.get_any(room_id)
            if room is None:
                return
            job = {"phase": "anonymizing", "started_at": time.time(), "done": 0, "failed": 0}
            await _set_job(room_id, job)
            await _work(room, job)
            final = "review"
    except asyncio.CancelledError:
        if room_id not in _STOP:
            raise  # the deck is shutting down: the room stays "running" and carries on at the next start
        # The person stopped it: wind down like a finished run.
        task = asyncio.current_task()
        if task is not None and hasattr(task, "uncancel"):
            task.uncancel()
        final = "review"
    except Exception as exc:  # noqa: BLE001 — shown on the room, never the file contents
        log.warning("Data room run failed", room=room_id, error=type(exc).__name__)
        job["message"] = f"Stopped: {getattr(exc, 'detail', None) or type(exc).__name__}"
        final = "review"
    finally:
        _TASKS.pop(room_id, None)
    stopped = room_id in _STOP
    _STOP.discard(room_id)
    if stopped:
        left = sum(1 for f in await store.files(room_id) if f["state"] == "new")
        job["message"] = (f"Stopped by you — {left} file{'s' if left != 1 else ''} not run yet. Run the room to "
                          "finish them." if left else "Stopped by you.")
        log.info("Data room run stopped", room=room_id, files_left=left)
    for key in ("current", "step", "chunk", "chunks", "file_started_at", "chunks_started_at", "chunk_at",
                "current_size", "file_index", "file_count", "replace"):
        job.pop(key, None)
    job["phase"] = "done"
    job["finished_at"] = time.time()
    files = await store.files(room_id)
    if not any(f["state"] in ("done", "blocked") for f in files):
        final = "setup"
    await _set_job(room_id, job)
    await store.update(room_id, state=final)


async def _file_start(room_id: str, job: dict[str, Any], f: dict[str, Any]) -> None:
    """A file begins: which one, how big, nothing read yet."""
    now = time.time()
    job.update({"current": f["id"], "current_size": int(f["size"] or 0), "step": "reading", "chunk": 0,
                "chunks": 0, "file_started_at": now, "chunks_started_at": None, "chunk_at": None,
                "file_index": job.get("done", 0) + job.get("failed", 0) + 1})
    job.pop("replace", None)
    log.info("Data room file started", room=room_id, file=f["id"], size=int(f["size"] or 0),
             index=job["file_index"], of=job.get("total", 0))
    await _set_job(room_id, job)


def _file_progress(room_id: str, job: dict[str, Any], f: dict[str, Any]):
    """The hook anonymize_blob reports through: the step, and the trusted
    model's chunks (n of N), kept in the job for the page — and logged."""
    async def hook(event: dict[str, Any]) -> None:
        if job.get("phase") != "anonymizing" or job.get("current") != f["id"]:
            return  # a late tick from a file (or a run) that's over — a stopped run's thread, say
        step = str(event.get("step") or "")
        now = time.time()
        if step == "model":
            done, total = int(event.get("chunk") or 0), int(event.get("chunks") or 0)
            if job.get("step") != "model":
                job["chunks_started_at"] = now
            job.update({"step": "model", "chunk": done, "chunks": total, "chunk_at": now})
            log.info("Data room file chunk", room=room_id, file=f["id"], chunk=done, chunks=total,
                     seconds=round(now - (job.get("chunks_started_at") or now), 1))
        elif step == "replacing" and "pass" in event:
            # The engine's own progress: round r, pass (looking / matching /
            # replacing / checking), n of N pieces of text.
            prev = job.get("replace") or {}
            r, name = int(event.get("round") or 1), str(event.get("pass") or "")
            done, of = int(event.get("done") or 0), int(event.get("of") or 0)
            fresh = prev.get("round") != r or prev.get("pass") != name
            started = now if fresh else prev.get("started_at", now)
            if fresh and prev.get("pass"):
                log.info("Data room file pass done", room=room_id, file=f["id"], round=prev.get("round"),
                         stage=prev.get("pass"), of=prev.get("of"), seconds=round(now - prev.get("started_at", now), 1))
            if fresh:
                log.info("Data room file pass", room=room_id, file=f["id"], round=r, stage=name, of=of,
                         unit=event.get("unit"))
            job["step"] = "replacing"
            job["replace"] = {"round": r, "pass": name, "done": done, "of": of, "unit": event.get("unit") or "parts",
                              "started_at": started}
        elif step:
            if step != job.get("step"):
                log.info("Data room file step", room=room_id, file=f["id"], step=step,
                         **({"pieces": event["pieces"]} if "pieces" in event else {}))
            job["step"] = step
            if step == "replacing":
                job["replace"] = {"round": 1, "pass": "", "done": 0, "of": int(event.get("pieces") or 0),
                                  "unit": event.get("unit") or "parts", "started_at": now}
        await _set_job(room_id, job)
    return hook


async def _work(room: dict[str, Any], job: dict[str, Any]) -> None:
    store = rs.store()
    vaults = av.store()
    owner, room_id = room["owner_id"], room["id"]
    vault_row = await vaults.get(owner, room["vault_id"])
    if vault_row is None:
        raise rs.RoomError("this room's vault is gone")
    cipher = vaults.cipher(vault_row)
    data = _data(room, room["vault_id"])
    actor = f"room:{room_id}"

    # 1. anonymize the new files
    todo = [f for f in await store.files(room_id) if f["state"] == "new"]
    job.update({"total": len(todo), "done": 0, "failed": 0, "bytes_total": sum(int(f["size"] or 0) for f in todo),
                "bytes_done": 0})
    for f in todo:
        if room_id in _STOP:
            return
        await _file_start(room_id, job, f)
        hook = ar.PROGRESS.set(_file_progress(room_id, job, f))
        try:
            try:
                raw = await read_original(room, f, cipher)
                if not f.get("info") and PurePosixPath(f["path"]).suffix.lower() in ri.TABLE_KINDS:
                    await store.set_file(room_id, f["id"], info=await asyncio.to_thread(ri.table_info, f["path"], raw))
                blob, info = await ar.anonymize_blob(owner, posixpath.basename(f["path"]), raw, data, actor=actor,
                                                     source="data-room")
            except HTTPException as exc:
                await store.set_file(room_id, f["id"], state="failed", error=str(exc.detail)[:300])
                job["failed"] += 1
                continue
            except (ri.IntakeError, OSError, rs.RoomError) as exc:
                await store.set_file(room_id, f["id"], state="failed", error=str(exc)[:300])
                job["failed"] += 1
                continue
            except Exception as exc:  # noqa: BLE001 — one file never stops the room
                log.warning("Data room file failed", room=room_id, file=f["id"], error=type(exc).__name__)
                await store.set_file(room_id, f["id"], state="failed", error=type(exc).__name__)
                job["failed"] += 1
                continue
            name = _clean_name(info["filename"])
            ext = PurePosixPath(name).suffix
            path = store.clean_file(room_id, f["id"], ext)
            path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(path.write_bytes, blob)
            await store.set_file(room_id, f["id"], state="done", error="", clean_path=name,
                                 kind=ext.lstrip(".").lower(), warnings=list(info.get("warnings") or []),
                                 entities=dict(info.get("summary") or {}))
            job["done"] += 1
        finally:
            ar.PROGRESS.reset(hook)
            if room_id not in _STOP:  # a file stopped half way stays "not run yet"
                job["bytes_done"] = job.get("bytes_done", 0) + int(f["size"] or 0)
                log.info("Data room file finished", room=room_id, file=f["id"], size=int(f["size"] or 0),
                         seconds=round(time.time() - job.get("file_started_at", time.time()), 1),
                         chunks=job.get("chunks", 0), files_left=len(todo) - job["done"] - job["failed"])
    for key in ("current", "step", "chunk", "chunks", "file_started_at", "chunks_started_at", "chunk_at",
                "current_size", "file_index", "replace"):
        job.pop(key, None)
    if room_id in _STOP:
        return

    files = [f for f in await store.files(room_id) if f["state"] in ("done", "blocked")]

    # 2. folder names, through the vault (it knows the names in them by now)
    job["phase"] = "naming"
    await _set_job(room_id, job)
    folders = sorted({str(PurePosixPath(f["path"]).parent) for f in files} - {"."})
    named: dict[str, str] = {}
    for folder in folders:
        parts = folder.split("/")
        named[folder] = "/".join(await ar.anonymize_names(owner, parts, data, actor=actor))
    for f in files:
        base = posixpath.basename(f["clean_path"]) or f["clean_path"]
        folder = str(PurePosixPath(f["path"]).parent)
        clean_path = f"{named[folder]}/{base}" if folder in named else base
        if clean_path != f["clean_path"]:
            await store.set_file(room_id, f["id"], clean_path=clean_path)
            f["clean_path"] = clean_path

    # 3. final sweep: code only, the final vault, until nothing changes
    sweep = {**data, "lane": "deterministic", "one_word_names": False, "pictures": "keep"}
    job["phase"] = "sweeping"
    for round_no in range(1, SWEEP_ROUNDS + 1):
        if room_id in _STOP:
            return
        job["sweep_round"] = round_no
        await _set_job(room_id, job)
        changed = 0
        for i, f in enumerate(files, 1):
            job.update({"current": f["id"], "file_index": i, "file_count": len(files)})
            await _set_job(room_id, job)
            path = store.clean_file(room_id, f["id"], PurePosixPath(f["clean_path"]).suffix)
            try:
                blob = await asyncio.to_thread(path.read_bytes)
                if not f.get("info") and PurePosixPath(f["path"]).suffix.lower() in ri.TABLE_KINDS:
                    # Rows and sheets for a file done before they were counted (same in the copy).
                    await store.set_file(room_id, f["id"], info=await asyncio.to_thread(ri.table_info, f["path"], blob))
                out, info = await ar.anonymize_blob(owner, posixpath.basename(f["clean_path"]), blob, sweep,
                                                    actor=actor, source="data-room-sweep")
            except Exception as exc:  # noqa: BLE001 — the leak check below still looks at this file
                log.warning("Data room sweep skipped a file", room=room_id, file=f["id"], error=type(exc).__name__)
                continue
            if sum((info.get("summary") or {}).values()):
                await asyncio.to_thread(path.write_bytes, out)
                changed += 1
                if f["released"]:
                    await store.set_file(room_id, f["id"], released=False, warnings=[
                        *f["warnings"], "A name found later was still in this file after it was released — "
                        "Claude may have read the old copy. Release it again."])
        job["sweep_changed"] = job.get("sweep_changed", 0) + changed
        if not changed:
            break

    # 4. the text Claude reads, and 5. the leak check
    job["phase"] = "checking"
    await _set_job(room_id, job)
    async with vaults.lock(vault_row["id"]):
        state = await vaults.load_state(vault_row)
    needle_list = needles(state)
    blocked = 0
    for i, f in enumerate(files, 1):
        if room_id in _STOP:
            return
        job.update({"current": f["id"], "file_index": i, "file_count": len(files)})
        await _set_job(room_id, job)
        path = store.clean_file(room_id, f["id"], PurePosixPath(f["clean_path"]).suffix)
        blob = await asyncio.to_thread(path.read_bytes)
        text = await asyncio.to_thread(text_view, f["clean_path"], blob)
        hits = await asyncio.to_thread(leaks, text + "\n" + f["clean_path"], needle_list)
        text_path = store.text_file(room_id, f["id"])
        text_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(text_path.write_text, text, "utf-8")
        if hits:
            blocked += 1
            await store.set_file(room_id, f["id"], state="blocked", released=False, text_chars=len(text),
                                 error=f"Still holds the real value of {', '.join(hits[:12])}"
                                       + (f" and {len(hits) - 12} more" if len(hits) > 12 else ""))
            await store.index_text(room_id, f["id"], [])
            continue
        await store.set_file(room_id, f["id"], state="done", error="", text_chars=len(text))
        await store.index_text(room_id, f["id"], pages_of(text))
    job["blocked"] = blocked
    for key in ("current", "file_index", "file_count"):
        job.pop(key, None)
