"""Getting a data room's files in: a ZIP upload, or a folder on the deck's
machine (``docs/data-room-plan.md`` P1).

A ZIP is checked before anything is kept: entry count, unpacked size, the
unpacked-to-packed ratio (ZIP bombs), absolute paths and ``..`` (zip slip),
symlinks, password protection. A ZIP inside the ZIP is unpacked one level.
Names in old Windows ZIPs come in a DOS code page, not UTF-8 — Croatian
ones usually cp852 — so a name is decoded by the code page that gives real
letters. Each original is stored encrypted with the room vault's key.

A folder is read where it is, never written: its files are listed and hashed
now (so a change shows later) and read again when the room runs. An admin may
name any folder of the deck's machine (as for VFS links); everyone else picks
a VFS folder they can read.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import posixpath
import re
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from captain_claw.flight_deck import anonymize_routes as ar
from captain_claw.flight_deck import rooms_store as rs

MAX_ZIP_BYTES = int(float(os.environ.get("FD_ROOMS_MAX_ZIP_MB", "25") or 25) * 1024 * 1024)
MAX_ENTRIES = 3000
MAX_UNPACKED = 500 * 1024 * 1024
MAX_RATIO = 200           # unpacked / packed, for entries over 1 MB
MAX_FOLDER_FILES = 3000
_JUNK_NAMES = {".ds_store", "thumbs.db", "desktop.ini", "icon\r"}
_JUNK_DIRS = {"__macosx", ".git", ".svn", "node_modules", "$recycle.bin"}
_CROATIAN = set("čćžšđČĆŽŠĐ")
_BOX = set("░▒▓│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬╧╨╤╥╙╘╒╓╫╪┘┌█▄▌▐▀")


class IntakeError(ValueError):
    """The whole upload or folder is refused (shown to the person)."""


@dataclass
class Report:
    added: int = 0
    replaced: int = 0
    unchanged: int = 0
    unread: list[dict[str, str]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    nested: int = 0

    def view(self) -> dict[str, Any]:
        return {"added": self.added, "replaced": self.replaced, "unchanged": self.unchanged,
                "unread": self.unread[:200], "unread_count": len(self.unread),
                "skipped": self.skipped[:200], "skipped_count": len(self.skipped), "nested_zips": self.nested}


# ── names ───────────────────────────────────────────────────────────


def decode_name(info: zipfile.ZipInfo) -> str:
    """An entry's name as the person wrote it. Python decodes names without the
    UTF-8 flag as cp437; the bytes are recovered and tried as UTF-8, then as the
    DOS / Windows code page that gives the most real letters."""
    name = info.filename
    if info.flag_bits & 0x800:
        return name
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass

    def score(text: str) -> int:
        return sum(3 for ch in text if ch in _CROATIAN) - sum(5 for ch in text if ch in _BOX)

    candidates = [raw.decode(cp, errors="replace") for cp in ("cp852", "cp1250", "cp437")]
    return max(candidates, key=score)


def safe_rel(name: str) -> str | None:
    """A ZIP entry's name as a safe relative path, or None (absolute, ``..``,
    a drive letter, control characters, junk)."""
    name = name.replace("\\", "/")
    if not name or name.startswith("/") or re.match(r"^[A-Za-z]:", name) or "\x00" in name:
        return None
    if any(ord(ch) < 32 for ch in name):
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    if any(p.casefold() in _JUNK_DIRS for p in parts[:-1]):
        return None
    last = parts[-1]
    if last.casefold() in _JUNK_NAMES or last.startswith(("~$", "._")):
        return None
    return "/".join(parts)


def kind_of(rel: str) -> str:
    return PurePosixPath(rel).suffix.lower().lstrip(".")


def readable(rel: str) -> bool:
    return PurePosixPath(rel).suffix.lower() in ar.FILE_KINDS


# ── ZIP ─────────────────────────────────────────────────────────────


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def _check(zf: zipfile.ZipFile, *, label: str) -> list[zipfile.ZipInfo]:
    infos = zf.infolist()
    if len(infos) > MAX_ENTRIES:
        raise IntakeError(f"{label} has {len(infos):,} entries — at most {MAX_ENTRIES:,}.")
    total = 0
    for info in infos:
        if info.flag_bits & 0x1:
            raise IntakeError(f"{label} is password-protected — unpack it without a password first.")
        total += info.file_size
        if info.file_size > 1024 * 1024 and info.file_size > MAX_RATIO * max(info.compress_size, 1):
            raise IntakeError(f"{label} unpacks far bigger than it is (a ZIP bomb?) — refused.")
    if total > MAX_UNPACKED:
        raise IntakeError(f"{label} unpacks to {total // (1024 * 1024):,} MB — at most "
                          f"{MAX_UNPACKED // (1024 * 1024):,} MB.")
    return infos


def _read_bounded(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """An entry's bytes, never more than it says it has (a lying header can't
    unpack more than that)."""
    out = io.BytesIO()
    with zf.open(info) as src:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            if out.tell() > info.file_size:
                raise IntakeError("a ZIP entry unpacks bigger than its header says — refused.")
    return out.getvalue()


async def _keep(room: dict[str, Any], rel: str, blob: bytes, cipher: Any, report: Report) -> None:
    store = rs.store()
    sha = hashlib.sha256(blob).hexdigest()
    if not readable(rel):
        await store.add_file(room["id"], origin="zip", path=rel, kind=kind_of(rel), size=len(blob), sha256=sha,
                             state="unread", error="this kind of file can't be read")
        report.unread.append({"path": rel, "reason": "this kind of file can't be read"})
        return
    if len(blob) > ar.MAX_FILE_BYTES:
        reason = f"over {ar.MAX_FILE_BYTES // (1024 * 1024)} MB"
        await store.add_file(room["id"], origin="zip", path=rel, kind=kind_of(rel), size=len(blob), sha256=sha,
                             state="unread", error=reason)
        report.unread.append({"path": rel, "reason": reason})
        return
    fid, replaced = await store.add_file(room["id"], origin="zip", path=rel, kind=kind_of(rel),
                                         size=len(blob), sha256=sha)
    raw = store.raw_path(room["id"], fid)
    if replaced or not raw.exists():
        await asyncio.to_thread(store.write_raw, room["id"], fid, blob, cipher)
        if replaced:
            report.replaced += 1
        else:
            report.added += 1
    else:
        report.unchanged += 1


async def _entries(zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo], prefix: str, room: dict[str, Any],
                   cipher: Any, report: Report, depth: int) -> None:
    for info in infos:
        if info.is_dir():
            continue
        rel = safe_rel(decode_name(info))
        if rel is None or _is_symlink(info):
            report.skipped.append({"path": decode_name(info)[:300], "reason": "unsafe or junk entry"})
            continue
        rel = posixpath.join(prefix, rel) if prefix else rel
        blob = await asyncio.to_thread(_read_bounded, zf, info)
        if rel.lower().endswith(".zip") and depth == 0:
            try:
                inner = zipfile.ZipFile(io.BytesIO(blob))
                inner_infos = _check(inner, label=f"The ZIP inside ({posixpath.basename(rel)})")
            except zipfile.BadZipFile:
                report.unread.append({"path": rel, "reason": "a damaged ZIP inside"})
                continue
            report.nested += 1
            await _entries(inner, inner_infos, rel[:-4], room, cipher, report, depth + 1)
            continue
        await _keep(room, rel, blob, cipher, report)


async def intake_zip(room: dict[str, Any], zip_path: Path, cipher: Any) -> Report:
    """Take a ZIP's files into *room* (originals encrypted with *cipher*)."""
    if zip_path.stat().st_size > MAX_ZIP_BYTES:
        raise IntakeError(f"The ZIP is over {MAX_ZIP_BYTES // (1024 * 1024)} MB — link a folder on the deck's "
                          "machine instead.")
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise IntakeError("That isn't a ZIP file (or it's damaged).") from None
    with zf:
        infos = _check(zf, label="The ZIP")
        report = Report()
        await _entries(zf, infos, "", room, cipher, report, 0)
    return report


# ── folders ─────────────────────────────────────────────────────────


def _walk(root: Path) -> list[tuple[str, int]]:
    """(relative path, size) of the files under *root*: no hidden files or
    folders, no symlinks, no junk, at most MAX_FOLDER_FILES + 1."""
    out: list[tuple[str, int]] = []
    stack: list[tuple[Path, str]] = [(root, "")]
    while stack and len(out) <= MAX_FOLDER_FILES:
        here, rel = stack.pop()
        try:
            with os.scandir(here) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            if e.name.startswith("."):
                continue
            try:
                if e.is_symlink():
                    continue
                if e.is_dir(follow_symlinks=False):
                    if e.name.casefold() not in _JUNK_DIRS:
                        stack.append((Path(e.path), f"{rel}{e.name}/"))
                    continue
                if not e.is_file(follow_symlinks=False):
                    continue
                size = e.stat(follow_symlinks=False).st_size
            except OSError:
                continue
            path = f"{rel}{e.name}"
            if safe_rel(path) is None:
                continue
            out.append((path, size))
    return out


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(root: Path, rel: str) -> Path:
    """``root/rel``, refusing anything that resolves outside *root* (symlinks)."""
    target = (root / rel).resolve()
    if not target.is_relative_to(root.resolve()):
        raise IntakeError("a file points outside the folder")
    return target


async def intake_folder(room: dict[str, Any], root: Path) -> Report:
    """List and hash the files of a folder (read where they are)."""
    if not root.is_dir():
        raise IntakeError("That folder doesn't exist on the deck's machine.")
    listed = await asyncio.to_thread(_walk, root)
    if len(listed) > MAX_FOLDER_FILES:
        raise IntakeError(f"The folder has more than {MAX_FOLDER_FILES:,} files — pick a smaller one.")
    store = rs.store()
    report = Report()
    for rel, size in listed:
        if not readable(rel) or size > ar.MAX_FILE_BYTES:
            reason = "this kind of file can't be read" if not readable(rel) else \
                f"over {ar.MAX_FILE_BYTES // (1024 * 1024)} MB"
            await store.add_file(room["id"], origin="folder", path=rel, kind=kind_of(rel), size=size,
                                 sha256="", state="unread", error=reason)
            report.unread.append({"path": rel, "reason": reason})
            continue
        sha = await asyncio.to_thread(_sha_file, inside(root, rel))
        _fid, replaced = await store.add_file(room["id"], origin="folder", path=rel, kind=kind_of(rel),
                                              size=size, sha256=sha)
        if replaced:
            report.replaced += 1
        else:
            report.added += 1
    return report


def guard_local_folder(path: str, data_dir: Path) -> Path:
    """An admin's folder of the deck's machine: absolute, existing, and not
    Flight Deck's own data (or a folder holding it)."""
    p = Path(str(path or "")).expanduser()
    if not p.is_absolute():
        raise IntakeError("Give the folder's full path (it starts with /).")
    p = p.resolve()
    if not p.is_dir():
        raise IntakeError("That folder doesn't exist on the deck's machine.")
    data = data_dir.resolve()
    if p == data or data.is_relative_to(p) or p.is_relative_to(data):
        raise IntakeError("Flight Deck's own data folder can't be a data room.")
    return p
