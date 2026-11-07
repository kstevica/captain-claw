"""The ``.captain`` file: a zip with ``manifest.json`` and the agent's parts.

Writing: everything under a staging folder goes in, each file's size and
sha256 recorded in the manifest. Reading never trusts the file: every entry
name must be a plain relative path (no ``..``, no absolute path, no backslash,
no symlink), the sizes have caps, and an entry is only used when its bytes
match the manifest's sha256.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

FORMAT = "captain-agent"
FORMAT_VERSION = 1
EXTENSION = ".captain"
MANIFEST = "manifest.json"

_CHUNK = 1 << 20
_MAX_MANIFEST = 8 << 20


class BundleError(ValueError):
    """A file that isn't a bundle this deck can use; the message is for the person."""


def max_bytes() -> int:
    """The largest bundle (and the most it may unpack to): ``FD_AGENT_BUNDLE_MAX_MB``, default 2 GB."""
    try:
        mb = int(os.environ.get("FD_AGENT_BUNDLE_MAX_MB", "2048"))
    except ValueError:
        mb = 2048
    return max(1, mb) << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name: str) -> bool:
    """A zip entry name this deck will read: relative, forward slashes, no
    ``.``/``..`` segments, every segment plain."""
    if not name or name.startswith("/") or "\\" in name or "\x00" in name or len(name) > 512:
        return False
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return False
    return not any(ord(ch) < 32 or ch == "\x7f" for ch in name)


def write(staging: Path, manifest: dict[str, Any], out: Path) -> dict[str, Any]:
    """Zip *staging* into *out* with *manifest* (its ``files`` filled in here)."""
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in staging.rglob("*") if p.is_file() and not p.is_symlink()):
        rel = path.relative_to(staging).as_posix()
        files.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {**manifest, "files": files, "size": sum(f["size"] for f in files)}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        zf.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=1, default=str))
        for f in files:
            zf.write(staging / f["path"], f["path"])
    tmp.replace(out)
    return manifest


@dataclass
class Bundle:
    path: Path
    manifest: dict[str, Any]
    _files: dict[str, dict[str, Any]]

    def has(self, name: str) -> bool:
        return name in self._files

    def names(self, prefix: str = "") -> list[str]:
        return [n for n in self._files if n.startswith(prefix)]

    def read_json(self, name: str, default: Any = None) -> Any:
        if name not in self._files:
            return default
        with zipfile.ZipFile(self.path) as zf:
            data = self._read_checked(zf, name)
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise BundleError(f"{name} in the bundle isn't readable") from exc

    def _read_checked(self, zf: zipfile.ZipFile, name: str) -> bytes:
        meta = self._files[name]
        with zf.open(name) as src:
            data = src.read(int(meta["size"]) + 1)
        if len(data) != int(meta["size"]) or hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise BundleError(f"{name} in the bundle doesn't match its manifest — the file was changed or damaged")
        return data

    def extract(self, name: str, dest: Path) -> None:
        """Write the entry *name* to *dest*, checking its size and sha256 as it streams."""
        meta = self._files[name]
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.unpack")
        h = hashlib.sha256()
        size = 0
        try:
            with zipfile.ZipFile(self.path) as zf, zf.open(name) as src, tmp.open("wb") as out:
                for chunk in iter(lambda: src.read(_CHUNK), b""):
                    size += len(chunk)
                    if size > int(meta["size"]):
                        raise BundleError(f"{name} in the bundle is bigger than its manifest says")
                    h.update(chunk)
                    out.write(chunk)
            if size != int(meta["size"]) or h.hexdigest() != meta["sha256"]:
                raise BundleError(f"{name} in the bundle doesn't match its manifest — the file was changed or damaged")
            tmp.replace(dest)
        finally:
            if tmp.exists():
                tmp.unlink()


def open_bundle(path: Path) -> Bundle:
    """Read and check a bundle's manifest and entries (not their contents yet)."""
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise BundleError("This isn't an agent bundle (a .captain file).") from exc
    with zf:
        infos = zf.infolist()
        by_name: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            name = info.filename
            if name.endswith("/"):
                continue
            if not safe_name(name):
                raise BundleError(f"The bundle holds an unsafe path ({name[:120]!r}) — not opened.")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise BundleError(f"The bundle holds a link ({name[:120]}) — not opened.")
            if name in by_name:
                raise BundleError(f"The bundle holds {name[:120]} twice — not opened.")
            by_name[name] = info
        if MANIFEST not in by_name:
            raise BundleError("This isn't an agent bundle: it has no manifest.")
        if by_name[MANIFEST].file_size > _MAX_MANIFEST:
            raise BundleError("The bundle's manifest is too big.")
        try:
            manifest = json.loads(zf.read(MANIFEST).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise BundleError("The bundle's manifest isn't readable.") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise BundleError("This isn't an agent bundle.")
    version = manifest.get("format_version")
    if not isinstance(version, int) or version < 1:
        raise BundleError("The bundle's format version isn't readable.")
    if version > FORMAT_VERSION:
        raise BundleError("This bundle was made by a newer Captain than this deck runs — update the deck first.")
    files: dict[str, dict[str, Any]] = {}
    total = 0
    for f in manifest.get("files") or []:
        if not isinstance(f, dict):
            raise BundleError("The bundle's manifest isn't readable.")
        name, size, digest = f.get("path"), f.get("size"), f.get("sha256")
        if not isinstance(name, str) or not safe_name(name) or name == MANIFEST or \
                not isinstance(size, int) or size < 0 or not isinstance(digest, str) or \
                not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise BundleError("The bundle's manifest isn't readable.")
        info = by_name.get(name)
        if info is None or info.file_size != size:
            raise BundleError(f"{name[:120]} is missing from the bundle or not the size its manifest says.")
        total += size
        files[name] = {"size": size, "sha256": digest}
    if total > max_bytes():
        raise BundleError(f"The bundle unpacks to {total >> 20} MB, more than this deck takes "
                          f"({max_bytes() >> 20} MB).")
    extra = set(by_name) - set(files) - {MANIFEST}
    if extra:
        raise BundleError(f"The bundle holds files its manifest doesn't list ({sorted(extra)[0][:120]}).")
    return Bundle(path=path, manifest=manifest, _files=files)


def save_stream(src: BinaryIO, dest: Path) -> int:
    """Copy an upload to *dest*, refusing one over :func:`max_bytes`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    limit = max_bytes()
    size = 0
    tmp = dest.with_name(dest.name + ".part")
    try:
        with tmp.open("wb") as out:
            for chunk in iter(lambda: src.read(_CHUNK), b""):
                size += len(chunk)
                if size > limit:
                    raise BundleError(f"The file is bigger than this deck takes ({limit >> 20} MB).")
                out.write(chunk)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return size


def sweep(folder: Path, max_age_s: float) -> None:
    """Remove what *folder* has kept longer than *max_age_s*."""
    if not folder.is_dir():
        return
    cutoff = time.time() - max_age_s
    for p in folder.iterdir():
        try:
            if p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
        except OSError:
            continue
