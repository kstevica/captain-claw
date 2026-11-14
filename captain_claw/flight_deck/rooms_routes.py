"""Data room routes for the person (``docs/data-room-plan.md`` P1): create a
room, put files in (a ZIP or a folder), run it, check what came out, set
roles, release it. Claude's side (the MCP connector) comes in P2.

A room is its owner's alone: every route answers only the owner, and an
admin's act-as is refused — the originals and the vault behind a room aren't
an admin's to open.
"""

from __future__ import annotations

import asyncio
import posixpath
import re
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from captain_claw.anonymize import engine, scans
from captain_claw.flight_deck import anonymize_routes as ar
from captain_claw.flight_deck import anonymize_trusted as trusted
from captain_claw.flight_deck import anonymize_vault as av
from captain_claw.flight_deck import fd_crypto
from captain_claw.flight_deck import rooms_intake as ri
from captain_claw.flight_deck import rooms_job as rj
from captain_claw.flight_deck import rooms_store as rs
from captain_claw.flight_deck.auth import ACT_AS_HEADER, get_current_user, get_db

router = APIRouter(prefix="/fd/rooms", tags=["data-rooms"])

MAX_SEED_PER_LIST = 200
MAX_ROLES = 300
MAX_ROLE = 120
_TEXT_PREVIEW = 200_000


def _owner(request: Request, user: dict) -> str:
    if request.headers.get(ACT_AS_HEADER):
        raise HTTPException(403, "A data room is its owner's alone — admin act-as can't open it.")
    return str(user["id"])


async def _room(owner: str, room_id: str) -> dict[str, Any]:
    try:
        room = await rs.store().get(owner, room_id)
    except rs.RoomError:
        room = None
    except fd_crypto.CryptoUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    if room is None:
        raise HTTPException(404, "No such data room")
    return room


def _counts(files: list[dict[str, Any]]) -> dict[str, int]:
    out = {"files": len(files), "released": sum(1 for f in files if f["released"])}
    for state in rs.FILE_STATES:
        out[state] = sum(1 for f in files if f["state"] == state)
    out["warnings"] = sum(1 for f in files if f["state"] == "done" and f["warnings"])
    return out


async def _summary(room: dict[str, Any]) -> dict[str, Any]:
    files = await rs.store().files(room["id"])
    job = dict(room.get("job") or {})
    job["running"] = rj.running(room["id"])
    deliveries = await rs.store().deliveries(room["id"])
    tasks = await rs.store().tasks(room["id"])
    return {"id": room["id"], "codename": room["codename"], "name": room["name"], "state": room["state"],
            "counts": _counts(files), "job": job, "created_at": room["created_at"],
            "updated_at": room["updated_at"], "released_at": room["released_at"],
            "inbox": {"total": len(deliveries), "unseen": sum(1 for d in deliveries if not d["seen"])},
            "tasks": {"total": len(tasks), "open": sum(1 for t in tasks if t["state"] in ("waiting", "taken"))}}


def _vault_name(codename: str, room_id: str) -> str:
    return av.clean_name(f"Data room · {codename} · {room_id[-4:]}")


# ── rooms ───────────────────────────────────────────────────────────


@router.get("")
async def list_rooms(request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    return {"rooms": [await _summary(r) for r in await rs.store().list_rooms(owner)]}


@router.post("")
async def create_room(request: Request, body: dict[str, Any] = Body(...),
                      user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    try:
        codename = rs.clean_codename(str(body.get("codename") or ""))
        room_id = rs.new_room_id()
        vault = await av.store().find_or_create(owner, _vault_name(codename, room_id), created_by=f"person:{owner}")
        room = await rs.store().create(owner, codename, name=str(body.get("name") or ""), vault_id=vault["id"],
                                       vault_name=vault["name"], room_id=room_id)
    except (rs.RoomError, av.VaultError) as exc:
        raise HTTPException(422, str(exc)) from None
    except fd_crypto.CryptoUnavailable as exc:
        raise HTTPException(503, f"Data rooms need this deck's data key: {exc}") from None
    return await _detail(room)


async def _detail(room: dict[str, Any]) -> dict[str, Any]:
    out = await _summary(room)
    model = await trusted.usable(get_db())
    out.update({"vault_id": room["vault_id"], "vault_name": room["vault_name"], "source": room["source"],
                "seed": room["seed"], "roles": room["roles"], "settings": room["settings"],
                "allowed": room.get("allowed") or [],
                "trusted_model": model.label if model else "", "ocr": await asyncio.to_thread(scans.available),
                "max_zip_mb": ri.MAX_ZIP_BYTES // (1024 * 1024), "max_file_mb": ar.MAX_FILE_BYTES // (1024 * 1024)})
    return out


@router.get("/{room_id}")
async def get_room(room_id: str, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    return await _detail(await _room(_owner(request, user), room_id))


def _seed(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise HTTPException(422, "seed is an object of lists (orgs, people, products, domains, ids, other)")
    out: dict[str, list[str]] = {}
    for key in rj.SEED_KINDS:
        items = value.get(key) or []
        if not isinstance(items, list):
            raise HTTPException(422, f"seed.{key} is a list")
        clean = []
        for item in items[:MAX_SEED_PER_LIST]:
            text = re.sub(r"\s+", " ", str(item or "")).strip()[:300]
            if text and text not in clean:
                clean.append(text)
        if clean:
            out[key] = clean
    return out


async def _roles(room: dict[str, Any], value: Any) -> dict[str, str]:
    """Roles are shown to Claude, so a role that names a real value from the
    vault is refused (nothing is saved), never stored with a warning."""
    if not isinstance(value, dict) or len(value) > MAX_ROLES:
        raise HTTPException(422, f"roles is an object token → role, at most {MAX_ROLES}")
    roles: dict[str, str] = {}
    for token, role in value.items():
        token, role = str(token).strip(), re.sub(r"\s+", " ", str(role or "")).strip()[:MAX_ROLE]
        if not engine.TOKEN_RE.fullmatch(token):
            raise HTTPException(422, f"not a token: {token[:40]}")
        if role:
            roles[token] = role
    vault = await av.store().get(room["owner_id"], room["vault_id"])
    if vault is not None and roles:
        state = await av.store().load_state(vault)
        needle_list = rj.room_needles(state, room)
        named = [(token, rj.leaks(role, needle_list)) for token, role in roles.items()]
        named = [(token, hits) for token, hits in named if hits]
        if named:
            token, hits = named[0]
            more = f" (and {len(named) - 1} more)" if len(named) > 1 else ""
            raise HTTPException(422, f"The role for {token} names a real value ({', '.join(hits[:3])}){more} — "
                                     "Claude sees roles. Say what it is, not who. Nothing was saved.")
    return roles


@router.patch("/{room_id}")
async def update_room(room_id: str, request: Request, body: dict[str, Any] = Body(...),
                      user: dict = Depends(get_current_user)) -> dict[str, Any]:
    room = await _room(_owner(request, user), room_id)
    fields: dict[str, Any] = {}
    if "codename" in body:
        fields["codename"] = str(body["codename"] or "")
    if "name" in body:
        fields["name"] = str(body["name"] or "")
    if "seed" in body:
        fields["seed"] = _seed(body["seed"])
    if "roles" in body:
        fields["roles"] = await _roles(room, body["roles"])
    if "settings" in body:
        s = body["settings"] if isinstance(body["settings"], dict) else {}
        settings = dict(room["settings"])
        if "pictures" in s:
            if s["pictures"] not in ar.PICTURE_MODES:
                raise HTTPException(422, f"settings.pictures is one of {', '.join(ar.PICTURE_MODES)}")
            settings["pictures"] = s["pictures"]
        if "lane" in s:
            if s["lane"] not in ("", *ar.LANES):
                raise HTTPException(422, f"settings.lane is '' or one of {', '.join(ar.LANES)}")
            settings["lane"] = s["lane"]
        fields["settings"] = settings
    try:
        await rs.store().update(room_id, **fields)
    except rs.RoomError as exc:
        raise HTTPException(422, str(exc)) from None
    return await _detail(await _room(room["owner_id"], room_id))


# ── "not a name here" ───────────────────────────────────────────────


@router.post("/{room_id}/allowed")
async def allow_spelling(room_id: str, request: Request, body: dict[str, Any] = Body(...),
                         user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """A spelling the check found that isn't a name in this room ("Ured" the
    office, not a person): it is left as it is in every file, and the check
    stops looking for it. With ``rerun``, the files where it was hidden after
    the check go back to "not run yet", to come back with the word in place."""
    owner = _owner(request, user)
    room = await _room(owner, room_id)
    _not_running(room)
    text = re.sub(r"\s+", " ", str(body.get("text") or "")).strip()
    if not text or len(text) > 300:
        raise HTTPException(422, "text is the spelling as the check found it (up to 300 characters)")
    tokens = [str(t) for t in body.get("tokens") or [] if engine.TOKEN_RE.fullmatch(str(t))][:20]
    key = rj.allow_key(text)
    if len(key.replace(" ", "")) < rj.MIN_NEEDLE:
        raise HTTPException(422, "That's too short to be looked for anyway.")
    allowed = [a for a in room.get("allowed") or [] if rj.allow_key(str(a.get("text") or "")) != key]
    if len(allowed) >= rj.MAX_ALLOWED:
        raise HTTPException(422, f"A room can leave at most {rj.MAX_ALLOWED} words as they are.")
    import time

    allowed.append({"text": text, "tokens": tokens, "at": time.time()})
    store = rs.store()
    await store.update(room_id, allowed=allowed)
    requeued = 0
    if body.get("rerun"):
        for f in await store.files(room_id):
            hidden = (f.get("leftovers") or {}).get("hidden") or []
            if any(rj.allow_key(str(x.get("text") or "")) == key for x in hidden):
                await store.set_file(room_id, f["id"], state="new", released=False, error="")
                requeued += 1
    return {"requeued": requeued, "room": await _detail(await _room(owner, room_id))}


@router.post("/{room_id}/allowed/remove")
async def unallow_spelling(room_id: str, request: Request, body: dict[str, Any] = Body(...),
                           user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Look for a spelling again (it is hidden again at the next run)."""
    owner = _owner(request, user)
    room = await _room(owner, room_id)
    _not_running(room)
    key = rj.allow_key(str(body.get("text") or ""))
    allowed = [a for a in room.get("allowed") or [] if rj.allow_key(str(a.get("text") or "")) != key]
    await rs.store().update(room_id, allowed=allowed)
    return {"room": await _detail(await _room(owner, room_id))}


# ── "maybe the same" ────────────────────────────────────────────────

MAX_NOT_SAME = 5000
MAX_BATCH = 300  # pairs answered in one go: as many as are listed


async def _room_vault(owner: str, room: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    vaults = av.store()
    row = await vaults.get(owner, room["vault_id"])
    if row is None:
        raise HTTPException(409, "This room's vault is gone.")
    return vaults, row


async def _texts(room_id: str, files: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(file id, the text Claude reads) for every file that has it."""
    store = rs.store()

    def read() -> list[tuple[str, str]]:
        out = []
        for f in files:
            try:
                out.append((f["id"], store.text_file(room_id, f["id"]).read_text("utf-8")))
            except OSError:
                continue
        return out

    return await asyncio.to_thread(read)


def _bare_token(value: Any) -> str:
    token = str(value or "").strip()
    m = engine.TOKEN_RE.fullmatch(token)
    if not m or ":" in token:
        raise HTTPException(422, "Give a token as it is written, without a :2 (⟦PERSON_3⟧).")
    return engine.make_token(m.group(1) or m.group(3), int(m.group(2) or m.group(4)))


@router.get("/{room_id}/alike")
async def alike_tokens(room_id: str, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Tokens that may be one party — a typo, a first name alone, a changed
    surname, a company's short name — with their real values, for the owner
    to say. Pairs already answered aren't listed again."""
    from captain_claw.flight_deck import rooms_alike as ra

    owner = _owner(request, user)
    room = await _room(owner, room_id)
    vaults, row = await _room_vault(owner, room)
    async with vaults.lock(row["id"]):
        state = await vaults.load_state(row)
    by_file, counts = ra.token_stats(await _texts(room_id, await rs.store().files(room_id)))
    not_same = list((room.get("settings") or {}).get("not_same") or [])
    suggestions = await asyncio.to_thread(ra.suggest, state, by_file, not_same, counts)
    merged = [{"token": e.token, "text": e.original, "into": e.same_as} for e in state.entities if e.same_as]
    return {"suggestions": suggestions, "answered": len(not_same), "merged": merged}


def _pairs_in(body: dict[str, Any], first: str, second: str) -> tuple[list[tuple[str, str]], bool]:
    """The pairs a body answers — {first, second} for one, {"pairs": [...]}
    for many — and whether it was many."""
    many = body.get("pairs")
    if many is None:
        return [(_bare_token(body.get(first)), _bare_token(body.get(second)))], False
    if not isinstance(many, list) or not many or not all(isinstance(p, dict) for p in many):
        raise HTTPException(422, "Give the pairs as a list of objects.")
    if len(many) > MAX_BATCH:
        raise HTTPException(422, f"At most {MAX_BATCH} pairs at once.")
    return [(_bare_token(p.get(first)), _bare_token(p.get(second))) for p in many], True


@router.post("/{room_id}/same")
async def make_same(room_id: str, request: Request, body: dict[str, Any] = Body(...),
                    user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """*other* is the same party as *keep*: in the vault, *other*'s values
    become *keep*'s spellings, and every copy in the room — Office files, the
    text Claude reads, the search index — gets *keep*'s number in its place.
    A file whose format can't be rewritten in place runs again.

    {"pairs": [{keep, other}, …]} makes many one at once, in that order, and
    rewrites each file once; a pair that can't be (already one, a token the
    vault hasn't) is skipped and said, not an error. A token already made
    one with another is never made one with a third: that would join two
    parties no one said are one."""

    owner = _owner(request, user)
    room = await _room(owner, room_id)
    _not_running(room)
    pairs, many = _pairs_in(body, "keep", "other")
    vaults, row = await _room_vault(owner, room)
    mapping: dict[str, str] = {}
    made: list[str] = []
    skipped: list[dict[str, str]] = []

    def skip(keep_token: str, other_token: str, status: int, why: str) -> None:
        if not many:
            raise HTTPException(status, why)
        skipped.append({"keep": keep_token, "other": other_token, "why": why})

    async with vaults.lock(row["id"]):
        state = await vaults.load_state(row)
        for keep_token, other_token in pairs:
            if keep_token == other_token:
                skip(keep_token, other_token, 422, "Those are the same token.")
                continue
            keep, other = state.by_token.get(keep_token), state.by_token.get(other_token)
            if keep is None or other is None:
                skip(keep_token, other_token, 404, "This room's vault doesn't have that token.")
                continue
            keep = state.target(keep)
            if keep is other:
                skip(keep_token, other_token, 409, f"{keep_token} and {other_token} are one already.")
                continue
            if other.same_as:
                where = state.target(other).token
                skip(keep_token, other_token, 409, f"{other_token} is already one with {where}.")
                continue
            try:
                step = state.merge(keep, other)
            except ValueError as exc:
                skip(keep_token, other_token, 422, f"These can't be one: {exc}.")
                continue
            # A token an earlier pair rewrote to *other*'s spelling now goes to *keep*'s.
            mapping = {k: step.get(v, v) for k, v in mapping.items()}
            mapping.update(step)
            made.append(other_token)
        if made:
            await vaults.save_run(row, state)
    if not made:
        return {"tokens": 0, "merged": 0, "skipped": skipped, "rewritten": 0, "requeued": 0,
                "room": await _detail(room)}
    rewritten, requeued = await _rewrite_room(room_id, mapping)
    fields: dict[str, Any] = {}
    roles = dict(room.get("roles") or {})
    for token in made:
        if token in roles:
            roles.setdefault(state.target(state.by_token[token]).token, roles.pop(token))
            fields["roles"] = roles
    settings = dict(room.get("settings") or {})
    answered = settings.get("not_same") or []

    def one(key: str) -> bool:
        a, _, b = key.partition("|")
        ea, eb = state.by_token.get(a), state.by_token.get(b)
        return ea is not None and eb is not None and state.target(ea) is state.target(eb)

    if any(one(k) for k in answered):
        settings["not_same"] = [k for k in answered if not one(k)]
        fields["settings"] = settings
    if fields:
        await rs.store().update(room_id, **fields)
    return {"tokens": len(mapping), "merged": len(made), "skipped": skipped, "rewritten": rewritten,
            "requeued": requeued, "room": await _detail(await _room(owner, room_id))}


async def _rewrite_room(room_id: str, mapping: dict[str, str]) -> tuple[int, int]:
    """Every copy in the room with its tokens written as *mapping* says: (files
    rewritten, files to run again because their format can't be)."""
    from captain_claw.flight_deck import rooms_alike as ra

    store = rs.store()
    rewritten = requeued = 0
    for f in await store.files(room_id):
        if not f.get("clean_path"):
            continue
        clean = store.clean_file(room_id, f["id"], PurePosixPath(f["clean_path"]).suffix)
        text_path = store.text_file(room_id, f["id"])
        changed = False
        if clean.exists():
            blob = await asyncio.to_thread(clean.read_bytes)
            new = await asyncio.to_thread(ra.rewrite_blob, f["clean_path"], blob, mapping)
            if new is None:
                await store.set_file(room_id, f["id"], state="new", released=False, error="")
                requeued += 1
                continue
            if new != blob:
                await asyncio.to_thread(clean.write_bytes, new)
                changed = True
        if text_path.exists():
            text = await asyncio.to_thread(text_path.read_text, "utf-8")
            new_text = ra.rewrite_text(text, mapping)
            if new_text != text:
                await asyncio.to_thread(text_path.write_text, new_text, "utf-8")
                changed = True
                if f["state"] == "done":
                    await store.index_text(room_id, f["id"], rj.pages_of(new_text))
        fields: dict[str, Any] = {}
        clean_path = ra.rewrite_text(f["clean_path"], mapping)
        if clean_path != f["clean_path"]:
            fields["clean_path"] = clean_path  # its name in a folder; on disk it's kept by its id
        leftovers = ra.rewrite_leftovers(f.get("leftovers") or {}, mapping)
        if leftovers != (f.get("leftovers") or {}):
            fields["leftovers"] = leftovers
        if fields:
            await store.set_file(room_id, f["id"], **fields)
        rewritten += 1 if changed or fields else 0
    return rewritten, requeued


@router.post("/{room_id}/not-same")
async def not_same(room_id: str, request: Request, body: dict[str, Any] = Body(...),
                   user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Two tokens that are two parties: the pair isn't suggested again.
    {"pairs": [{a, b}, …]} answers many at once."""
    from captain_claw.flight_deck import rooms_alike as ra

    owner = _owner(request, user)
    room = await _room(owner, room_id)
    pairs, _many = _pairs_in(body, "a", "b")
    keys = list(dict.fromkeys(ra.pair_key(a, b) for a, b in pairs if a != b))
    new = set(keys)
    settings = dict(room.get("settings") or {})
    answered = [k for k in settings.get("not_same") or [] if k not in new]
    if len(answered) + len(keys) > MAX_NOT_SAME:
        raise HTTPException(422, f"A room remembers at most {MAX_NOT_SAME} answers.")
    settings["not_same"] = [*answered, *keys]
    await rs.store().update(room_id, settings=settings)
    return {"answered": len(settings["not_same"])}


@router.delete("/{room_id}")
async def delete_room(room_id: str, request: Request, vault: bool = Query(False),
                      user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    room = await _room(owner, room_id)
    if rj.running(room_id):
        raise HTTPException(409, "Stop the run first.")
    await rs.store().delete(owner, room_id)
    vault_deleted = await av.store().delete(owner, room["vault_id"], actor=f"person:{owner}") if vault else False
    return {"deleted": True, "vault_deleted": bool(vault_deleted)}


# ── files in ────────────────────────────────────────────────────────


def _not_running(room: dict[str, Any]) -> None:
    if rj.running(room["id"]):
        raise HTTPException(409, "The room is running — wait for it, or stop it.")


@router.put("/{room_id}/zip")
async def upload_zip(room_id: str, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    room = await _room(owner, room_id)
    _not_running(room)
    declared = int(request.headers.get("content-length") or 0)
    if declared > ri.MAX_ZIP_BYTES:
        raise HTTPException(413, f"The ZIP is over {ri.MAX_ZIP_BYTES // (1024 * 1024)} MB — link a folder on the "
                                 "deck's machine instead.")
    store = rs.store()
    part = store.room_dir(room_id) / "upload.zip.part"
    part.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with open(part, "wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > ri.MAX_ZIP_BYTES:
                    raise HTTPException(413, f"The ZIP is over {ri.MAX_ZIP_BYTES // (1024 * 1024)} MB — link a "
                                             "folder on the deck's machine instead.")
                out.write(chunk)
        if not size:
            raise HTTPException(422, "The upload is empty.")
        vault = await av.store().get(owner, room["vault_id"])
        if vault is None:
            raise HTTPException(409, "This room's vault is gone.")
        report = await ri.intake_zip(room, part, av.store().cipher(vault))
    except ri.IntakeError as exc:
        raise HTTPException(422, str(exc)) from None
    finally:
        part.unlink(missing_ok=True)
    return {"report": report.view(), "room": await _detail(await _room(owner, room_id))}


@router.post("/{room_id}/folder")
async def link_folder(room_id: str, request: Request, body: dict[str, Any] = Body(...),
                      user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    room = await _room(owner, room_id)
    _not_running(room)
    try:
        if body.get("project"):
            from captain_claw.flight_deck import vfs_routes as vr

            source = {"kind": "vfs", "project": str(body["project"]), "path": str(body.get("path") or "").strip("/"),
                      "owner": str(body.get("owner") or "")}
            oid = await vr._eff_owner(owner, source["project"], source["owner"], write=False)
            root = vr._resolve(oid, source["project"], source["path"])
        else:
            if user.get("role") != "admin":
                raise HTTPException(403, "Only an admin can use a folder of the deck's machine — pick a folder "
                                         "from Files instead.")
            # rooms.db sits in the deck's data folder: that folder is never a room.
            root = ri.guard_local_folder(str(body.get("path") or ""), rs.store().path.parent)
            source = {"kind": "local", "path": str(root)}
        old = room.get("source") or {}
        if old and old != source and any(f["origin"] == "folder" for f in await rs.store().files(room_id)):
            raise HTTPException(409, "This room already reads another folder — make a new room for this one.")
        await rs.store().update(room_id, source=source)
        report = await ri.intake_folder({**room, "source": source}, Path(root))
    except ri.IntakeError as exc:
        raise HTTPException(422, str(exc)) from None
    return {"report": report.view(), "room": await _detail(await _room(owner, room_id))}


# ── run, check, release ─────────────────────────────────────────────


@router.post("/{room_id}/run")
async def run_room(room_id: str, request: Request, body: dict[str, Any] = Body(default={}),
                   user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    room = await _room(owner, room_id)
    try:
        await rj.start(room, retry_failed=bool((body or {}).get("retry_failed")))
    except rs.RoomError as exc:
        raise HTTPException(409, str(exc)) from None
    return await _detail(await _room(owner, room_id))


@router.post("/{room_id}/stop")
async def stop_room(room_id: str, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    await _room(owner, room_id)
    return {"stopping": await rj.stop(room_id)}


def _file_view(f: dict[str, Any]) -> dict[str, Any]:
    out = {k: f.get(k) for k in ("id", "origin", "path", "clean_path", "kind", "size", "state", "warnings", "error",
                                 "entities", "text_chars", "released", "updated_at", "info")}
    # Counts only: the spellings come with the file's own text (its page).
    lo = f.get("leftovers") or {}
    out["leftovers"] = {"hidden": sum(int(x.get("count") or 0) for x in lo.get("hidden") or []),
                        "left": sum(int(x.get("count") or 0) for x in lo.get("left") or [])}
    return out


@router.get("/{room_id}/files")
async def list_files(room_id: str, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    await _room(owner, room_id)
    return {"files": [_file_view(f) for f in await rs.store().files(room_id)]}


async def _file(room_id: str, file_id: str) -> dict[str, Any]:
    f = await rs.store().file(room_id, file_id)
    if f is None:
        raise HTTPException(404, "No such file in this room")
    return f


def _original_text(name: str, blob: bytes) -> str:
    """The original's text, for the side-by-side check (owner only, never stored)."""
    from captain_claw.tools.web_fetch import _extract_readable_text

    ext = PurePosixPath(name).suffix.lower()
    if ext in (".docx", ".pptx", ".xlsx"):
        return rj.text_view(name, blob)
    if ext == ".pdf":
        try:
            return ar._pdf_text(blob)[0]
        except HTTPException as exc:
            return f"_({exc.detail})_"
    if ext in (".html", ".htm"):
        return _extract_readable_text(blob.decode("utf-8", errors="replace"))
    if ext in ar._IMAGE_EXT:
        return "_(a picture: its text is read with OCR when the room runs)_"
    return blob.decode("utf-8", errors="replace")


@router.get("/{room_id}/files/{file_id}/text")
async def file_text(room_id: str, file_id: str, request: Request, original: bool = Query(False),
                    user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    room = await _room(owner, room_id)
    f = await _file(room_id, file_id)
    store = rs.store()
    out: dict[str, Any] = {"id": f["id"], "path": f["path"], "clean_path": f["clean_path"], "state": f["state"]}
    # Real values the check found in this file (owner only): hidden after the
    # check, and still there (what blocks it), each as it is written.
    lo = f.get("leftovers") or {}
    out["leftovers"] = {"hidden": lo.get("hidden") or [], "left": lo.get("left") or []}
    text_path = store.text_file(room_id, file_id)
    clean = await asyncio.to_thread(text_path.read_text, "utf-8") if text_path.exists() else ""
    out["clean"] = clean[:_TEXT_PREVIEW]
    out["clean_cut"] = len(clean) > _TEXT_PREVIEW  # the page shows the start of a long file
    if original:
        vault = await av.store().get(owner, room["vault_id"])
        if vault is None:
            raise HTTPException(409, "This room's vault is gone.")
        try:
            blob = await rj.read_original(room, f, av.store().cipher(vault))
        except (ri.IntakeError, OSError) as exc:
            raise HTTPException(409, f"The original can't be read: {exc}") from None
        original_text = await asyncio.to_thread(_original_text, f["path"], blob)
        out["original"] = original_text[:_TEXT_PREVIEW]
        out["original_cut"] = len(original_text) > _TEXT_PREVIEW
    return out


@router.get("/{room_id}/files/{file_id}/clean")
async def file_clean(room_id: str, file_id: str, request: Request,
                     user: dict = Depends(get_current_user)) -> FileResponse:
    owner = _owner(request, user)
    await _room(owner, room_id)
    f = await _file(room_id, file_id)
    if f["state"] not in ("done", "blocked") or not f["clean_path"]:
        raise HTTPException(409, "This file has no anonymized copy yet.")
    path = rs.store().clean_file(room_id, file_id, PurePosixPath(f["clean_path"]).suffix)
    if not path.exists():
        raise HTTPException(404, "The anonymized copy is missing — run the room again.")
    return FileResponse(path, filename=posixpath.basename(f["clean_path"]))


@router.delete("/{room_id}/files/{file_id}")
async def delete_file(room_id: str, file_id: str, request: Request,
                      user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    room = await _room(owner, room_id)
    _not_running(room)
    f = await _file(room_id, file_id)
    store = rs.store()
    await store.remove_file(room_id, file_id)
    for path in (store.raw_path(room_id, file_id), store.text_file(room_id, file_id),
                 store.clean_file(room_id, file_id, PurePosixPath(f["clean_path"]).suffix) if f["clean_path"] else None):
        if path is not None:
            path.unlink(missing_ok=True)
    return {"deleted": True}


@router.post("/{room_id}/release")
async def release_room(room_id: str, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    room = await _room(owner, room_id)
    _not_running(room)
    store = rs.store()
    released = await store.release(room_id)
    files = await store.files(room_id)
    if not any(f["released"] for f in files):
        raise HTTPException(409, "Nothing can be released yet — run the room, and fix or remove blocked files.")
    import time

    await store.update(room_id, state="released", released_at=time.time())
    return {"released_now": released, "room": await _detail(await _room(owner, room_id))}


@router.get("/{room_id}/search")
async def search_room(room_id: str, request: Request, q: str = Query(..., min_length=1, max_length=300),
                      user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    await _room(owner, room_id)
    return {"results": await rs.store().search(room_id, q, released_only=False)}


@router.get("/{room_id}/reads")
async def room_reads(room_id: str, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """What Claude read of this room, and any alarm (a real value in what Claude sent)."""
    owner = _owner(request, user)
    await _room(owner, room_id)
    store = rs.store()
    files = {f["id"]: f["clean_path"] for f in await store.files(room_id)}
    names: dict[str, str] = {}
    db = get_db()

    async def app_name(client_id: str) -> str:
        if client_id not in names:
            if not client_id:
                names[client_id] = "You"  # an export, or a check of your own
            else:
                client = await db.get_oauth_client(client_id)
                names[client_id] = (client or {}).get("client_name") or "A connected app"
        return names[client_id]

    reads = [{"at": r["at"], "tool": r["tool"], "path": files.get(r["file_id"], ""), "page": r["page"],
              "chars": r["chars"], "alarm": bool(r["alarm"]), "app": await app_name(r["client_id"])}
             for r in await store.reads(room_id)]
    alarms = [{"id": a["id"], "at": a["at"], "tool": a["tool"], "tokens": a["tokens"], "seen": a["seen"],
               "app": await app_name(a["client_id"])} for a in await store.alarms(room_id)]
    return {"reads": reads, "alarms": alarms}


@router.post("/{room_id}/export")
async def export_room(room_id: str, request: Request, body: dict[str, Any] = Body(default={}),
                      user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """“Download for Claude”: build the ZIP of the released room (P6); the
    download is one GET away, for an hour."""
    from captain_claw.flight_deck import rooms_export

    owner = _owner(request, user)
    room = await _room(owner, room_id)
    _not_running(room)
    try:
        return await rooms_export.build(room, text_only=bool((body or {}).get("text_only")))
    except rooms_export.ExportError as exc:
        raise HTTPException(409, str(exc)) from None


@router.get("/{room_id}/export/{token}")
async def download_export(room_id: str, token: str, request: Request, user: dict = Depends(get_current_user)):
    from starlette.background import BackgroundTask

    from captain_claw.flight_deck import rooms_export

    owner = _owner(request, user)
    room = await _room(owner, room_id)
    path = rooms_export.export_path(room_id, token)
    if path is None:
        raise HTTPException(404, "That download is gone — make a new one.")
    name = re.sub(r"[^\w\- ]+", "_", room["codename"]).strip() or "data-room"
    return FileResponse(path, media_type="application/zip", filename=f"{name} - for Claude.zip",
                        background=BackgroundTask(path.unlink, missing_ok=True))


@router.post("/{room_id}/alarms/seen")
async def room_alarms_seen(room_id: str, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    await _room(owner, room_id)
    await rs.store().mark_alarms_seen(room_id)
    return {"ok": True}


# ── the Inbox: what Claude sent back, restored (P3) ─────────────────


async def _cipher(owner: str, room: dict[str, Any]) -> Any:
    vault = await av.store().get(owner, room["vault_id"])
    if vault is None:
        raise HTTPException(409, "This room's vault is gone — its deliveries can't be opened.")
    try:
        return av.store().cipher(vault)
    except fd_crypto.CryptoUnavailable as exc:
        raise HTTPException(503, str(exc)) from None


async def _app_names(client_ids: set[str]) -> dict[str, str]:
    db = get_db()
    out = {}
    for cid in client_ids:
        if not cid:
            out[cid] = "Added by you"  # a file Claude made that the person dropped in
            continue
        client = await db.get_oauth_client(cid)
        out[cid] = (client or {}).get("client_name") or "A connected app"
    return out


@router.get("/{room_id}/deliveries")
async def list_deliveries(room_id: str, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    from captain_claw.flight_deck import rooms_inbox

    owner = _owner(request, user)
    room = await _room(owner, room_id)
    items = await rs.store().deliveries(room_id)
    cipher = await _cipher(owner, room) if items else None
    names = await _app_names({d["client_id"] for d in items})
    out = []
    for d in items:
        try:
            p = rooms_inbox.open_payload(cipher, d)
            title = p.get("title_fixed") or p["title"]
        except Exception:  # noqa: BLE001 — a payload this key can't open is listed, never guessed
            title = "(can't be opened)"
        out.append({"seq": d["seq"], "at": d["at"], "title": title, "chars": d["chars"], "report": d["report"],
                    "seen": d["seen"], "app": names[d["client_id"]]})
    return {"deliveries": out}


@router.put("/{room_id}/deliveries/file")
async def drop_claude_file(room_id: str, request: Request, name: str = Query(..., min_length=1, max_length=200),
                           user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """A file Claude made (in tokens) that the person downloaded from Claude and
    drops here: the names go back inside it and it joins the Inbox."""
    from captain_claw.flight_deck import rooms_inbox

    owner = _owner(request, user)
    room = await _room(owner, room_id)
    if int(request.headers.get("content-length") or 0) > rooms_inbox.MAX_OWN_FILE:
        raise HTTPException(413, f"The file is over {rooms_inbox.MAX_OWN_FILE // (1024 * 1024)} MB.")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > rooms_inbox.MAX_OWN_FILE:
            raise HTTPException(413, f"The file is over {rooms_inbox.MAX_OWN_FILE // (1024 * 1024)} MB.")
        chunks.append(chunk)
    try:
        seq, report = await rooms_inbox.take_file(room, "", name=name, blob=b"".join(chunks), source="owner")
    except rooms_inbox.DeliveryError as exc:
        raise HTTPException(422, str(exc)) from None
    return {"seq": seq, "report": report}


async def _delivery(owner: str, room: dict[str, Any], seq: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from captain_claw.flight_deck import rooms_inbox

    d = await rs.store().delivery(room["id"], seq)
    if d is None:
        raise HTTPException(404, "No such delivery")
    try:
        return d, rooms_inbox.open_payload(await _cipher(owner, room), d)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        raise HTTPException(409, "This delivery can't be opened with the room's key.") from None


@router.get("/{room_id}/deliveries/{seq}")
async def get_delivery(room_id: str, seq: int, request: Request,
                       user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    room = await _room(owner, room_id)
    d, p = await _delivery(owner, room, seq)
    await rs.store().mark_delivery_seen(room_id, seq)
    names = await _app_names({d["client_id"]})
    return {"seq": d["seq"], "at": d["at"], "app": names[d["client_id"]], "report": d["report"],
            "title": p["title"], "title_tokens": p["title_tokens"], "body": p["body"],
            "body_tokens": p["body_tokens"], "format": p["format"],
            # P7: the names in the case their sentence needs, when the trusted model changed any.
            "title_fixed": p.get("title_fixed"), "body_fixed": p.get("body_fixed")}


# ── tasks for Claude, asked from the Inbox ──────────────────────────


@router.get("/{room_id}/tasks")
async def list_tasks(room_id: str, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    from captain_claw.flight_deck import rooms_tasks

    owner = _owner(request, user)
    room = await _room(owner, room_id)
    try:
        return {"tasks": await rooms_tasks.owner_list(room)}
    except Exception:  # noqa: BLE001 — the vault is gone: nothing to show
        raise HTTPException(409, "This room's vault is gone — its tasks can't be opened.") from None


@router.post("/{room_id}/tasks/preview")
async def preview_task(room_id: str, request: Request, body: dict[str, Any] = Body(...),
                       user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """What Claude will read of a task: the person's words, anonymized with the room's vault."""
    from captain_claw.flight_deck import rooms_tasks

    owner = _owner(request, user)
    room = await _room(owner, room_id)
    try:
        return await rooms_tasks.anonymize_prompt(room, body.get("prompt"))
    except rooms_tasks.TaskError as exc:
        raise HTTPException(422, str(exc)) from None


@router.post("/{room_id}/tasks")
async def create_task(room_id: str, request: Request, body: dict[str, Any] = Body(...),
                      user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Leave Claude a task in this room; Claude picks it up when asked to check (room_tasks)."""
    from captain_claw.flight_deck import rooms_tasks

    owner = _owner(request, user)
    room = await _room(owner, room_id)
    try:
        return await rooms_tasks.queue(room, body.get("prompt"), body.get("about"))
    except rooms_tasks.TaskError as exc:
        raise HTTPException(422, str(exc)) from None


@router.delete("/{room_id}/tasks/{seq}")
async def cancel_task(room_id: str, seq: int, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    from captain_claw.flight_deck import rooms_tasks

    owner = _owner(request, user)
    room = await _room(owner, room_id)
    try:
        return {"cancelled": await rooms_tasks.cancel(room, seq)}
    except rooms_tasks.TaskError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/{room_id}/deliveries/{seq}/grammar")
async def fix_delivery_grammar(room_id: str, seq: int, request: Request,
                               user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Fix the case endings of a delivery's names now (P7) — one that wasn't
    taken for Croatian, or came in before a trusted model was set up."""
    from captain_claw.flight_deck import anonymize_trusted as trusted
    from captain_claw.flight_deck import rooms_grammar

    owner = _owner(request, user)
    room = await _room(owner, room_id)
    d, _p = await _delivery(owner, room, seq)
    if rooms_grammar.running(room_id, seq):
        return {"grammar": d["report"].get("grammar") or {"state": "waiting"}}
    if await trusted.usable(get_db()) is None:
        raise HTTPException(409, "No trusted model is set up — an admin sets one under Anonymization, then "
                                 "case endings can be fixed.")
    report = dict(d["report"])
    report["grammar"] = {"state": "waiting", "language": (report.get("grammar") or {}).get("language") or "hr",
                         "asked": True}
    await rs.store().update_delivery(room_id, seq, report=report)
    rooms_grammar.queue(room_id, seq)
    return {"grammar": report["grammar"]}


@router.get("/{room_id}/deliveries/{seq}/download")
async def download_delivery(room_id: str, seq: int, request: Request, as_: str = Query("md", alias="as"),
                            plain: bool = Query(False), user: dict = Depends(get_current_user)):
    from fastapi.responses import Response as RawResponse

    from captain_claw.flight_deck import rooms_inbox

    owner = _owner(request, user)
    room = await _room(owner, room_id)
    _d, p = await _delivery(owner, room, seq)
    if not plain:
        # P7: the names in their case, when the trusted model changed any; ?plain=1 as restored.
        p = {**p, "title": p.get("title_fixed") or p["title"], "body": p.get("body_fixed") or p["body"]}
    kind, ext = p.get("kind"), p.get("file")
    if kind and (as_ == "file" or (as_ == "docx" and ext == "docx")):
        # A file built on the deck (P4): the copy with the names back, decrypted for the owner.
        path = rooms_inbox.file_path(room_id, seq, ext, fixed=True)
        if plain or not path.exists():
            path = rooms_inbox.file_path(room_id, seq, ext)
        if not path.exists():
            raise HTTPException(404, "The file is missing.")
        cipher = await _cipher(owner, room)
        blob = cipher.decrypt(await asyncio.to_thread(path.read_bytes))
        stem = p["title"].rsplit(".", 1)[0] if p["title"].lower().endswith(f".{ext}") else p["title"]
        name = rooms_inbox.safe_filename(stem, ext)
        media = {"xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                 "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                 "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                 "json": "application/json", "csv": "text/csv; charset=utf-8",
                 "tsv": "text/tab-separated-values; charset=utf-8"}.get(ext, "text/plain; charset=utf-8")
    elif as_ == "file":
        raise HTTPException(422, "This delivery is text — download it as md or docx.")
    elif as_ == "docx":
        blob = await asyncio.to_thread(rooms_inbox.docx_bytes, p["title"], p["body"])
        name = rooms_inbox.safe_filename(p["title"], "docx")
        media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif as_ == "md":
        blob = f"# {p['title']}\n\n{p['body']}\n".encode()
        name, media = rooms_inbox.safe_filename(p["title"], "md"), "text/markdown; charset=utf-8"
    else:
        raise HTTPException(422, "as is 'md', 'docx' or 'file'")
    from urllib.parse import quote

    return RawResponse(blob, media_type=media, headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}", "Cache-Control": "no-store"})


@router.delete("/{room_id}/deliveries/{seq}")
async def delete_delivery(room_id: str, seq: int, request: Request,
                          user: dict = Depends(get_current_user)) -> dict[str, Any]:
    from captain_claw.flight_deck import rooms_inbox

    owner = _owner(request, user)
    await _room(owner, room_id)
    if not await rs.store().delete_delivery(room_id, seq):
        raise HTTPException(404, "No such delivery")
    rooms_inbox.remove_files(room_id, seq)
    return {"deleted": True}


# ── Claude's access: which connected app reads which rooms ──────────

access_router = APIRouter(prefix="/fd/rooms-access", tags=["data-rooms"])


@access_router.get("")
async def list_access(request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    from captain_claw.flight_deck.mcp_oauth_routes import ROOMS_PATH, _public_base

    owner = _owner(request, user)
    db = get_db()
    rooms = {r["id"]: r["codename"] for r in await rs.store().list_rooms(owner)}
    out = []
    for g in await rs.store().grants(owner):
        client = await db.get_oauth_client(g["client_id"])
        out.append({"client_id": g["client_id"], "app": (client or {}).get("client_name") or "A connected app",
                    "all_rooms": g["all_rooms"],
                    "rooms": [{"id": rid, "codename": rooms[rid]} for rid in g["rooms"] if rid in rooms],
                    "created_at": g["created_at"], "updated_at": g["updated_at"]})
    return {"grants": out, "connector_url": f"{_public_base(request)}{ROOMS_PATH}"}


@access_router.put("/{client_id}")
async def set_access(client_id: str, request: Request, body: dict[str, Any] = Body(...),
                     user: dict = Depends(get_current_user)) -> dict[str, Any]:
    owner = _owner(request, user)
    if await rs.store().grant(owner, client_id) is None:
        raise HTTPException(404, "No such connected app")
    own = {r["id"] for r in await rs.store().list_rooms(owner)}
    rooms = [str(r) for r in body.get("rooms") or [] if str(r) in own]
    all_rooms = bool(body.get("all_rooms"))
    await rs.store().set_grant(owner, client_id, rooms, all_rooms)
    return {"ok": True, "rooms": rooms, "all_rooms": all_rooms}


@access_router.delete("/{client_id}")
async def revoke_access(client_id: str, request: Request, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Disconnect the app from the rooms at once (its sign-in stops working too)."""
    owner = _owner(request, user)
    deleted = await rs.store().delete_grant(owner, client_id)
    revoked = await get_db().revoke_oauth_grant(owner, client_id)
    return {"ok": True, "deleted": deleted, "revoked_tokens": revoked}
