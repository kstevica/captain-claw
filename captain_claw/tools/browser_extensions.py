"""Chrome extensions in the agent's own headless browser.

The headless browser (``browser_session.BrowserSession``) is Playwright's
Chromium on the machine the agent runs on. Extensions load only into a
browser started with a profile of its own (``launch_persistent_context``) in
Chromium's new headless mode (``channel="chromium"``); the stripped headless
shell Playwright uses by default can't run them. So once an agent has an
extension installed, its browser starts that way, with ``--load-extension``
for every installed one.

Here:

* **the store** — per agent, in its workspace: ``.browser-extensions/<id>/``
  (the unpacked extension) and ``index.json`` (what is installed, from where);
* **unpacking** — a folder, a ``.zip``, a ``.crx`` (CRX2 or CRX3), or a Chrome
  Web Store id or address (the CRX comes from Google's update server); paths
  in an archive can't leave the folder, links aren't followed, size is capped;
* **the manifest** — name (``__MSG_…__`` resolved), version, permissions, the
  sites it reads, content scripts, its popup and options pages;
* **the id** — from the manifest's ``key``: a Web Store extension gets its
  CRX's own key, so it keeps its store id; any other gets a key made here, so
  its id is the same on every machine and in every session;
* **the policy** — ``tools.browser.extension_policy``: ``ask`` (the person
  approves each install in Flight Deck, with what it can do), ``allow`` or
  ``off``; ids in ``extensions_allowed`` skip the question; a ``user``-tier
  agent on a multi-user deck never installs one (an extension's code would
  run with the server's network).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shutil
import stat
import struct
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

STORE_DIRNAME = ".browser-extensions"
INDEX_NAME = "index.json"
MAX_FILES = 5000
DEFAULT_PRODVERSION = "140.0.0.0"
WEBSTORE_CRX_URL = ("https://clients2.google.com/service/update2/crx?response=redirect&prodversion={version}"
                    "&acceptformat=crx2,crx3&x=id%3D{id}%26installsource%3Dondemand%26uc")
_ID_RE = re.compile(r"(?<![a-z])([a-p]{32})(?![a-z])")


class ExtensionError(Exception):
    """An extension that can't be installed (said plainly to the agent)."""


# ── ids ───────────────────────────────────────────────────────────────


def chrome_id(public_key_der: bytes) -> str:
    """Chrome's id for an extension with this public key: the first 16 bytes
    of its SHA-256, each hex digit written as a letter a–p."""
    digest = hashlib.sha256(public_key_der).hexdigest()[:32]
    return "".join(chr(ord("a") + int(c, 16)) for c in digest)


def webstore_id(text: str) -> str:
    """The extension id in a Chrome Web Store address or a bare id, else ""."""
    s = str(text or "").strip()
    if re.fullmatch(r"[a-p]{32}", s):
        return s
    if re.match(r"https?://(chromewebstore\.google\.com|chrome\.google\.com/webstore)/", s, re.I):
        m = _ID_RE.search(s.split("?", 1)[0])
        return m.group(1) if m else ""
    return ""


def _new_public_key() -> bytes | None:
    """A fresh RSA public key (DER, SubjectPublicKeyInfo) for an extension without one."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        return None
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.public_key().public_bytes(serialization.Encoding.DER,
                                         serialization.PublicFormat.SubjectPublicKeyInfo)


# ── CRX ───────────────────────────────────────────────────────────────


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        if i >= len(buf):
            raise ExtensionError("The .crx header is cut short.")
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, i
        shift += 7
        if shift > 63:
            raise ExtensionError("The .crx header is damaged.")


def _proto_fields(buf: bytes) -> list[tuple[int, Any]]:
    """The (field number, value) pairs of a protobuf message — enough for a CRX header."""
    out: list[tuple[int, Any]] = []
    i = 0
    while i < len(buf):
        tag, i = _varint(buf, i)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i:i + 8], i + 8
        elif wire == 2:
            n, i = _varint(buf, i)
            value, i = buf[i:i + n], i + n
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        else:
            raise ExtensionError("The .crx header is damaged.")
        out.append((field, value))
    return out


def parse_crx(blob: bytes) -> tuple[bytes, bytes | None]:
    """``(zip bytes, the extension's public key or None)`` from a .crx (v2 or
    v3) — or from a plain .zip, which has no key."""
    if blob[:4] == b"PK\x03\x04":
        return blob, None
    if blob[:4] != b"Cr24" or len(blob) < 16:
        raise ExtensionError("That isn't a Chrome extension (.crx) or a .zip.")
    version = struct.unpack("<I", blob[4:8])[0]
    if version == 2:
        key_len, sig_len = struct.unpack("<II", blob[8:16])
        key = blob[16:16 + key_len]
        return blob[16 + key_len + sig_len:], key or None
    if version != 3:
        raise ExtensionError(f"Unknown .crx version {version}.")
    header_len = struct.unpack("<I", blob[8:12])[0]
    header = blob[12:12 + header_len]
    body = blob[12 + header_len:]
    keys: list[bytes] = []
    crx_id = b""
    for field, value in _proto_fields(header):
        if field in (2, 3):   # sha256_with_rsa, sha256_with_ecdsa: AsymmetricKeyProof
            for f, v in _proto_fields(value):
                if f == 1:
                    keys.append(v)
        elif field == 10000:  # signed_header_data: SignedData
            for f, v in _proto_fields(value):
                if f == 1:
                    crx_id = v
    for key in keys:
        if hashlib.sha256(key).digest()[:16] == crx_id:
            return body, key
    return body, (keys[0] if keys else None)


def extract_zip(data: bytes, dest: Path, max_bytes: int) -> None:
    """Unpack *data* into *dest*: no path out of it, no links, at most
    *max_bytes* unpacked and MAX_FILES files. A zip holding one top folder
    (no manifest at its root) is unpacked from inside that folder."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ExtensionError("The extension's archive is damaged.") from exc
    entries = [i for i in zf.infolist() if not i.is_dir()]
    if len(entries) > MAX_FILES:
        raise ExtensionError(f"The extension has over {MAX_FILES} files.")
    names = []
    for info in entries:
        raw = info.filename.replace("\\", "/")
        p = PurePosixPath(raw)
        if p.is_absolute() or ".." in p.parts or raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
            raise ExtensionError(f"The archive has a file outside its folder ({raw[:80]}).")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ExtensionError(f"The archive has a link ({raw[:80]}); extensions don't need links.")
        names.append(p)
    if sum(i.file_size for i in entries) > max_bytes:
        raise ExtensionError(f"The extension is over {max_bytes // (1024 * 1024)} MB unpacked.")
    strip = 0
    if names and not any(str(n) == "manifest.json" for n in names):
        tops = {n.parts[0] for n in names}
        if len(tops) == 1 and all(len(n.parts) > 1 for n in names):
            strip = 1
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for info, p in zip(entries, names):
        rel = PurePosixPath(*p.parts[strip:])
        if not rel.parts:
            continue
        target = dest.joinpath(*rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise ExtensionError(f"The extension is over {max_bytes // (1024 * 1024)} MB unpacked.")
                out.write(chunk)


# ── the manifest ──────────────────────────────────────────────────────


def read_manifest(folder: Path) -> dict[str, Any]:
    path = folder / "manifest.json"
    if not path.is_file():
        raise ExtensionError("There's no manifest.json in it — not a Chrome extension.")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        data = json.loads(text)
    except ValueError:
        # Chrome reads manifests with // comments; keep strings intact while dropping them.
        cleaned = re.sub(r'("(?:\\.|[^"\\])*")|//[^\n]*|/\*.*?\*/', lambda m: m.group(1) or "", text, flags=re.S)
        try:
            data = json.loads(cleaned)
        except ValueError as exc:
            raise ExtensionError(f"Its manifest.json isn't valid JSON ({exc}).") from exc
    if not isinstance(data, dict):
        raise ExtensionError("Its manifest.json isn't an object.")
    if data.get("manifest_version") not in (2, 3):
        raise ExtensionError("Its manifest says no manifest_version 2 or 3.")
    return data


def localized(folder: Path, manifest: dict[str, Any], value: Any) -> str:
    """*value*, with ``__MSG_name__`` read from the default locale's messages."""
    s = str(value or "")
    m = re.fullmatch(r"__MSG_(\w+)__", s.strip())
    if not m:
        return s
    locale = str(manifest.get("default_locale") or "en")
    for loc in (locale, locale.replace("-", "_"), "en", "en_US"):
        f = folder / "_locales" / loc / "messages.json"
        if f.is_file():
            try:
                msgs = json.loads(f.read_text(encoding="utf-8-sig"))
            except ValueError:
                continue
            wanted = m.group(1).lower()
            for k, v in msgs.items():
                if k.lower() == wanted and isinstance(v, dict):
                    return str(v.get("message") or s)
    return s


def summary(folder: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """What the extension is and what it can do."""
    mv = manifest.get("manifest_version")
    perms = [p for p in manifest.get("permissions") or [] if isinstance(p, str)]
    hosts = [p for p in manifest.get("host_permissions") or [] if isinstance(p, str)]
    if mv == 2:
        hosts += [p for p in perms if "://" in p or p == "<all_urls>"]
        perms = [p for p in perms if not ("://" in p or p == "<all_urls>")]
    scripts = sorted({m for cs in manifest.get("content_scripts") or [] if isinstance(cs, dict)
                      for m in cs.get("matches") or [] if isinstance(m, str)})
    action = manifest.get("action") or manifest.get("browser_action") or manifest.get("page_action") or {}
    options = (manifest.get("options_ui") or {}).get("page") or manifest.get("options_page") or ""
    return {
        "name": localized(folder, manifest, manifest.get("name")) or "(no name)",
        "version": str(manifest.get("version") or ""),
        "manifest_version": mv,
        "description": localized(folder, manifest, manifest.get("description"))[:300],
        "permissions": perms[:40],
        "sites": hosts[:40],
        "content_scripts": scripts[:40],
        "popup": str(action.get("default_popup") or "") if isinstance(action, dict) else "",
        "options": str(options or ""),
    }


def describe(s: dict[str, Any]) -> str:
    """The summary as a few lines (for the agent and the person's approval)."""
    lines = [f"{s['name']} {s['version']}".strip()]
    if s.get("description"):
        lines.append(s["description"])
    lines.append("Permissions: " + (", ".join(s["permissions"]) or "none"))
    reach = sorted(set(s.get("sites", [])) | set(s.get("content_scripts", [])))
    lines.append("Reads and changes: " + (", ".join(reach[:12]) + (" …" if len(reach) > 12 else "") if reach else "no sites"))
    return "\n".join(lines)


def ensure_key(folder: Path, manifest: dict[str, Any], public_key: bytes | None) -> str:
    """Give the extension a ``key`` (its CRX's own, else a new one) so its id is
    fixed, and return that id."""
    key_b64 = manifest.get("key")
    if isinstance(key_b64, str) and key_b64.strip():
        try:
            return chrome_id(base64.b64decode(re.sub(r"\s+", "", key_b64)))
        except ValueError:
            pass
    der = public_key or _new_public_key()
    if der is None:
        raise ExtensionError("Can't give the extension an id here (the cryptography package is missing).")
    manifest["key"] = base64.b64encode(der).decode("ascii")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return chrome_id(der)


# ── the store ─────────────────────────────────────────────────────────


@dataclass
class Installed:
    id: str
    name: str
    version: str
    source: str
    installed_at: float
    summary: dict[str, Any]

    def view(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "version": self.version, "source": self.source,
                "installed_at": self.installed_at, "summary": self.summary}


class ExtensionStore:
    """An agent's installed extensions: ``<root>/<id>/`` and ``<root>/index.json``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @classmethod
    def for_tool(cls, kwargs: dict[str, Any]) -> ExtensionStore:
        base = kwargs.get("_runtime_base_path")
        return cls((Path(base) if base else Path.cwd()) / STORE_DIRNAME)

    def _index_path(self) -> Path:
        return self.root / INDEX_NAME

    def all(self) -> list[Installed]:
        try:
            raw = json.loads(self._index_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out = []
        for e in raw if isinstance(raw, list) else []:
            if isinstance(e, dict) and re.fullmatch(r"[a-p]{32}", str(e.get("id") or "")) and (self.root / e["id"]).is_dir():
                out.append(Installed(e["id"], str(e.get("name") or ""), str(e.get("version") or ""),
                                     str(e.get("source") or ""), float(e.get("installed_at") or 0),
                                     e.get("summary") if isinstance(e.get("summary"), dict) else {}))
        return out

    def dirs(self) -> list[str]:
        """The folders to load, in install order."""
        return [str((self.root / e.id).resolve()) for e in self.all()]

    def key_of_source(self, source: str) -> bytes | None:
        """The key of the extension installed from *source*, if one was."""
        for e in self.all():
            if e.source == source:
                try:
                    key = read_manifest(self.root / e.id).get("key")
                    return base64.b64decode(key) if isinstance(key, str) and key else None
                except (ExtensionError, ValueError):
                    return None
        return None

    def get(self, ext_id: str) -> Installed | None:
        wanted = str(ext_id or "").strip().lower()
        for e in self.all():
            if e.id == wanted or e.name.lower() == wanted:
                return e
        return None

    def _save(self, entries: list[Installed]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._index_path().with_suffix(".tmp")
        tmp.write_text(json.dumps([e.view() for e in entries], indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._index_path())

    def stage(self) -> Path:
        """An empty folder to unpack into."""
        d = self.root / f".staging-{int(time.time() * 1000)}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        return d

    def commit(self, staged: Path, ext_id: str, s: dict[str, Any], source: str) -> Installed:
        """Move a staged extension in as *ext_id* (replacing an older copy)."""
        final = self.root / ext_id
        if final.exists():
            shutil.rmtree(final)
        staged.rename(final)
        entry = Installed(ext_id, s["name"], s["version"], source, time.time(), s)
        self._save([e for e in self.all() if e.id != ext_id] + [entry])
        return entry

    def remove(self, ext_id: str) -> Installed | None:
        e = self.get(ext_id)
        if e is None:
            return None
        shutil.rmtree(self.root / e.id, ignore_errors=True)
        self._save([x for x in self.all() if x.id != e.id])
        return e


# ── getting one ───────────────────────────────────────────────────────


async def _download(url: str, max_bytes: int) -> bytes:
    import httpx

    data = bytearray()
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            async with client.stream("GET", url, headers={"User-Agent": "Mozilla/5.0 Chrome"}) as resp:
                if resp.status_code == 204:
                    raise ExtensionError("The Chrome Web Store has no such extension (or none for this Chrome).")
                if resp.status_code >= 400:
                    raise ExtensionError(f"The download answered {resp.status_code}.")
                async for chunk in resp.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        raise ExtensionError(f"The download is over {max_bytes // (1024 * 1024)} MB.")
    except httpx.HTTPError as exc:
        raise ExtensionError(f"Couldn't download it ({type(exc).__name__}).") from exc
    if not data:
        raise ExtensionError("The download was empty (the Chrome Web Store may not offer it for this Chrome).")
    return bytes(data)


async def fetch(*, store: ExtensionStore, path: Path | None = None, url: str = "", store_id: str = "",
                max_bytes: int, prodversion: str = DEFAULT_PRODVERSION) -> tuple[Path, str, dict[str, Any], str]:
    """Get an extension into a staging folder: ``(folder, id, summary, source)``.
    Nothing is installed yet — :meth:`ExtensionStore.commit` does that."""
    staged = store.stage()
    try:
        key: bytes | None = None
        if store_id:
            blob = await _download(WEBSTORE_CRX_URL.format(version=prodversion, id=store_id), max_bytes)
            data, key = parse_crx(blob)
            extract_zip(data, staged, max_bytes)
            source = f"Chrome Web Store {store_id}"
        elif url:
            blob = await _download(url, max_bytes)
            data, key = parse_crx(blob)
            extract_zip(data, staged, max_bytes)
            source = url
        elif path is not None:
            if path.is_dir():
                total = 0
                for f in path.rglob("*"):
                    if f.is_symlink():
                        raise ExtensionError(f"The folder has a link ({f.name}); extensions don't need links.")
                    if f.is_file():
                        total += f.stat().st_size
                        if total > max_bytes:
                            raise ExtensionError(f"The extension is over {max_bytes // (1024 * 1024)} MB.")
                shutil.rmtree(staged)
                shutil.copytree(path, staged, ignore=shutil.ignore_patterns(".git", "node_modules", ".DS_Store"))
            elif path.is_file():
                if path.stat().st_size > max_bytes:
                    raise ExtensionError(f"The file is over {max_bytes // (1024 * 1024)} MB.")
                data, key = parse_crx(path.read_bytes())
                extract_zip(data, staged, max_bytes)
            else:
                raise ExtensionError(f"Nothing at {path}.")
            source = str(path)
        else:
            raise ExtensionError("Say what to install: path (a folder, .zip or .crx), url, or extension_id "
                                 "(a Chrome Web Store id or address).")
        manifest = read_manifest(staged)
        if key is None and not manifest.get("key"):
            key = store.key_of_source(source)   # an update of one installed from here keeps its id
        ext_id = ensure_key(staged, manifest, key)
        if store_id and ext_id != store_id:
            log.warning("Web Store extension id differs from its key", store_id=store_id, got=ext_id)
        return staged, ext_id, summary(staged, manifest), source
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def discard(staged: Path) -> None:
    shutil.rmtree(staged, ignore_errors=True)


# ── policy ────────────────────────────────────────────────────────────


def policy(cfg: Any) -> str:
    value = str(getattr(cfg, "extension_policy", "ask") or "ask").strip().lower()
    return value if value in ("ask", "allow", "off") else "ask"


def allowed_without_asking(cfg: Any, ext_id: str) -> bool:
    listed = {str(x).strip().lower() for x in getattr(cfg, "extensions_allowed", None) or []}
    return bool(ext_id) and ext_id in listed


def deck_refusal() -> str | None:
    """Why this agent may never install one (a user-tier agent on a multi-user deck), else None."""
    try:
        from captain_claw import deck_confine
    except ImportError:
        return None
    if deck_confine.active() and deck_confine.tier() == "user":
        return ("Extensions can't be installed on this deck: their code would run in the server's browser with "
                "the server's network. Ask the deck's admin if you need one.")
    return None


async def ask_person(s: dict[str, Any], ext_id: str, source: str, kwargs: dict[str, Any]) -> tuple[bool, str]:
    """``(approved, why not)`` — the person's answer in Flight Deck."""
    from captain_claw.tools.ask_user import AskUserTool

    help_text = (describe(s) + f"\nFrom: {source}\nId: {ext_id}\n"
                 "It runs in this agent's own browser on the server (not in your Chrome), with the pages and "
                 "logins that browser has.")
    passing = {k: kwargs[k] for k in ("_agent", "_stream_callback") if k in kwargs}
    result = await AskUserTool().execute(
        question=f"Install the extension “{s['name']}” in the agent's browser?",
        confirm=True, help=help_text[:1500], wait_minutes=10, **passing)
    if not result.success:
        return False, (f"{result.error} Installing needs the person's OK: ask them in your reply; they can also "
                       "allow it without asking (tools.browser.extensions_allowed, or extension_policy: allow).")
    answer = (result.content or "").strip()
    if answer == "They answered: yes":
        return True, ""
    if answer == "They answered: no":
        return False, "The person said no. Don't install it."
    return False, f"No go-ahead from the person ({answer[:200]}). Don't install it unless they say so."
