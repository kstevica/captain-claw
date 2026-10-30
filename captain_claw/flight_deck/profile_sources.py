"""The material behind a person's owner profile — where "About me", "My
company" and "My standing preferences" can be drawn from.

A person gives Flight Deck their links (LinkedIn, GitHub, a personal site),
their company's name and links, and documents (a CV, a bio, a company
one-pager). `profile_research` reads all of it, looks the person and the
company up on the web, and drafts the three profile texts for them to review.
The material stays here, so the drafts can be made again when it changes.

Storage:

* the index — links, company name and links, and the list of documents — is
  user_settings ``fd:tenant-profile:sources`` (the auth-off deck's ``local``
  user has no users row, so system_settings ``fd:tenant-profile:sources:local``;
  `settings_routes` refuses the whole ``fd:tenant-profile`` prefix, so only
  these routes write it);
* a document's text (read in full when it is added, like a setup document)
  and the file it came as live under ``DATA_DIR/.profile-sources/<owner>/<id>/``.

Links are http(s) URLs only. Whether a host may be fetched is decided when
the research fetches it (`deck_confine.url_public`): a link to a private
address is kept but never read.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from captain_claw.flight_deck import agent_preset_docs as pdocs
from captain_claw.logging import get_logger

log = get_logger(__name__)

SOURCES_SETTING = "fd:tenant-profile:sources"
LOCAL_SOURCES_SETTING = "fd:tenant-profile:sources:local"

MAX_LINKS = 8
MAX_COMPANY_LINKS = 8
MAX_DOCUMENTS = 10
URL_MAX = 500
LABEL_MAX = 40
COMPANY_NAME_MAX = 120
DOC_TEXT_MAX = 400_000      # characters kept of one document (a CV is ~10k)

CAPS = {
    "links": MAX_LINKS, "company_links": MAX_COMPANY_LINKS, "documents": MAX_DOCUMENTS,
    "url": URL_MAX, "label": LABEL_MAX, "company_name": COMPANY_NAME_MAX,
    "document_bytes": pdocs.MAX_FILE_BYTES,
}
DOCUMENT_KINDS = tuple(pdocs.SUPPORTED)

# A link's label when the person gives none, by host.
_KNOWN_HOSTS = (
    ("linkedin.com", "LinkedIn"), ("github.com", "GitHub"), ("x.com", "X"), ("twitter.com", "X"),
    ("instagram.com", "Instagram"), ("facebook.com", "Facebook"), ("youtube.com", "YouTube"),
    ("medium.com", "Medium"), ("substack.com", "Substack"), ("threads.net", "Threads"),
    ("bsky.app", "Bluesky"), ("mastodon.social", "Mastodon"), ("tiktok.com", "TikTok"),
    ("crunchbase.com", "Crunchbase"), ("scholar.google.com", "Google Scholar"),
    ("researchgate.net", "ResearchGate"), ("orcid.org", "ORCID"), ("dribbble.com", "Dribbble"),
    ("behance.net", "Behance"), ("stackoverflow.com", "Stack Overflow"), ("gitlab.com", "GitLab"),
)


class SourceError(ValueError):
    """What is wrong with a link or a field, in words for the person."""


# ── Storage ─────────────────────────────────────────────────────────────


def _auth_enabled() -> bool:
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    return _fd_auth_enabled()


def _local_id() -> str:
    from captain_claw.flight_deck.auth import _LOCAL_USER

    return str(_LOCAL_USER["id"])


def blank() -> dict:
    return {"links": [], "company_name": "", "company_links": [], "documents": []}


def _link_rows(raw: Any, limit: int) -> list[dict]:
    out: list[dict] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("url"), str):
            continue
        url = item["url"].strip()[:URL_MAX]
        if not url:
            continue
        out.append({"url": url, "label": str(item.get("label") or "").strip()[:LABEL_MAX]})
        if len(out) >= limit:
            break
    return out


def _doc_rows(raw: Any) -> list[dict]:
    out: list[dict] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not re.fullmatch(r"[0-9a-f]{24}", str(item.get("id") or "")):
            continue
        out.append({
            "id": item["id"], "name": str(item.get("name") or "document")[:120],
            "pages": int(item.get("pages") or 0), "chars": int(item.get("chars") or 0),
            "added": float(item.get("added") or 0),
        })
        if len(out) >= MAX_DOCUMENTS:
            break
    return out


def parse(raw: str | None) -> dict:
    try:
        data = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {
        "links": _link_rows(data.get("links"), MAX_LINKS),
        "company_name": str(data.get("company_name") or "").strip()[:COMPANY_NAME_MAX]
        if isinstance(data.get("company_name"), str) else "",
        "company_links": _link_rows(data.get("company_links"), MAX_COMPANY_LINKS),
        "documents": _doc_rows(data.get("documents")),
    }


async def load(db, user_id: str) -> dict:
    """``user_id``'s material (empty when none was saved)."""
    if user_id == _local_id():
        raw = await db.get_system_setting(LOCAL_SOURCES_SETTING)
    else:
        raw = await db.get_setting(user_id, SOURCES_SETTING)
    return parse(raw)


async def save(db, user_id: str, sources: dict) -> None:
    blob = json.dumps(parse(json.dumps(sources)))
    if user_id == _local_id():
        await db.set_system_setting(LOCAL_SOURCES_SETTING, blob)
    else:
        await db.set_settings(user_id, {SOURCES_SETTING: blob})


def is_empty(sources: dict) -> bool:
    """Nothing to research from."""
    return not (sources.get("links") or sources.get("company_name") or sources.get("company_links")
                or sources.get("documents"))


# ── Links ───────────────────────────────────────────────────────────────


def normalize_url(value: Any) -> str:
    """A link as the person typed it, as an http(s) URL — ``linkedin.com/in/x``
    becomes ``https://linkedin.com/in/x``; a bare word, another scheme, or an
    overlong one raises SourceError."""
    text = " ".join(str(value or "").split())
    if not text:
        raise SourceError("A link is empty.")
    if len(text) > URL_MAX:
        raise SourceError(f"A link is too long (at most {URL_MAX} characters).")
    if "://" not in text:
        if text.startswith("//"):
            text = text[2:]
        text = "https://" + text
    try:
        parts = urlsplit(text)
    except ValueError:
        raise SourceError(f"{text[:60]} is not a web address.")
    if parts.scheme not in ("http", "https"):
        raise SourceError(f"{text[:60]}: only http and https links can be read.")
    host = (parts.hostname or "").strip().lower()
    if not host or "." not in host.strip(".") and host != "localhost":
        raise SourceError(f"{text[:60]} is not a web address.")
    if any(ch.isspace() for ch in text):
        raise SourceError(f"{text[:60]} is not a web address.")
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))


def host_of(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def label_for(url: str) -> str:
    """A name for a link: the network it is on, else its host."""
    host = host_of(url)
    for suffix, name in _KNOWN_HOSTS:
        if host == suffix or host.endswith("." + suffix):
            return name
    return host[:LABEL_MAX]


def clean_links(raw: Any, limit: int, what: str = "links") -> list[dict]:
    """The links a request sets — each normalised, labelled, no repeats, at
    most ``limit`` — or SourceError."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise SourceError(f"{what} must be a list.")
    out: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            item = {"url": item}
        if not isinstance(item, dict):
            raise SourceError(f"Each of the {what} must be a link.")
        url = normalize_url(item.get("url"))
        if url in seen:
            continue
        seen.add(url)
        label = " ".join(str(item.get("label") or "").split())[:LABEL_MAX] or label_for(url)
        out.append({"url": url, "label": label})
    if len(out) > limit:
        raise SourceError(f"At most {limit} {what}.")
    return out


def clean_company_name(raw: Any) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise SourceError("company_name must be text.")
    name = " ".join(raw.split())
    if len(name) > COMPANY_NAME_MAX:
        raise SourceError(f"The company name is too long (at most {COMPANY_NAME_MAX} characters).")
    return name


# ── Documents ───────────────────────────────────────────────────────────


def _data_dir() -> Path:
    from captain_claw.flight_deck.server import DATA_DIR

    return DATA_DIR


def owner_key(owner_id: str) -> str:
    """The folder a person's documents live in; a deck without sign-in has one."""
    key = re.sub(r"[^A-Za-z0-9_-]", "", str(owner_id or ""))[:64]
    return key if key and _auth_enabled() else "_deck"


def owner_root(owner_id: str) -> Path:
    return _data_dir() / ".profile-sources" / owner_key(owner_id)


def _doc_dir(owner_id: str, doc_id: str) -> Path | None:
    if not re.fullmatch(r"[0-9a-f]{24}", str(doc_id or "")):
        return None
    return owner_root(owner_id) / doc_id


def _no_text(name: str, err: str) -> str:
    log.info("profile sources: no text in a document", file=name, error=err)
    return f"{name}: no text could be read from it (a scanned PDF or a picture?), so it was left out."


def _write_document(owner_id: str, name: str, text: str, raw: bytes) -> dict:
    doc_id = secrets.token_hex(12)
    folder = _doc_dir(owner_id, doc_id)
    assert folder is not None
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "text.txt").write_text(text, encoding="utf-8")
    suffix = Path(name).suffix.lower()
    if re.fullmatch(r"\.[a-z0-9]{1,5}", suffix):
        (folder / f"original{suffix}").write_bytes(raw)
    pages, _exact = pdocs.pages_of(text)
    meta = {"id": doc_id, "name": name, "pages": pages, "chars": len(text), "added": time.time()}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


async def add_document(owner_id: str, name: str, raw: bytes) -> tuple[dict | None, str]:
    """Read *raw* (the file as uploaded) in full and keep it: its row for the
    index, or None and what to tell the person."""
    name = Path(name or "document").name[:120] or "document"
    suffix = Path(name).suffix.lower()
    if suffix not in DOCUMENT_KINDS:
        return None, (f"{name}: not a kind of document that can be read "
                      f"({', '.join(k.lstrip('.') for k in DOCUMENT_KINDS)}).")
    if not raw:
        return None, f"{name}: the file is empty."
    if len(raw) > pdocs.MAX_FILE_BYTES:
        return None, f"{name}: larger than 25 MB, left out."
    text, err = await asyncio.to_thread(pdocs.extract_bytes, raw, suffix)
    text = (text or "").strip()
    if not text:
        return None, _no_text(name, err)
    text = text[:DOC_TEXT_MAX]
    return await asyncio.to_thread(_write_document, owner_id, name, text, raw), ""


def remove_document(owner_id: str, doc_id: str) -> None:
    folder = _doc_dir(owner_id, doc_id)
    if folder is not None and folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)


def document_text(owner_id: str, doc_id: str) -> str:
    """A document's text, "" when it is gone."""
    folder = _doc_dir(owner_id, doc_id)
    if folder is None:
        return ""
    try:
        return (folder / "text.txt").read_text(encoding="utf-8")
    except OSError:
        return ""


def remove_all(owner_id: str) -> None:
    """Every document of an owner (their account was deleted)."""
    try:
        shutil.rmtree(owner_root(owner_id), ignore_errors=True)
    except Exception:
        pass
