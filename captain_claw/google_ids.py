"""Google Drive file / folder ids from the URLs people paste.

Agents are handed Drive, Docs, Sheets and Slides links far more often than bare
ids. The ``google_drive`` tool takes an id, and the web tools (browser,
web_fetch, shell curl/wget) redirect Drive links to it — both need the same
answer to "which file does this URL name?". Kept dependency-free so any tool
can import it cheaply.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

# Hosts that serve Drive content. sheets./slides. redirect to docs.google.com;
# drive.usercontent.google.com is Drive's download host. NOT
# storage.googleapis.com — that is Cloud Storage, which has no Drive ids.
_DRIVE_HOSTS = (
    "drive.google.com",
    "docs.google.com",
    "sheets.google.com",
    "slides.google.com",
    "drive.usercontent.google.com",
)

# A Drive id in a URL: base64url-ish, never shorter than this.
_ID_CHARS_RE = re.compile(r"^[A-Za-z0-9_-]{10,}$")
_RESOURCE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{4,}$")

# A Drive URL inside free text (a pasted sentence, a curl command), with or
# without its scheme.
_URL_IN_TEXT_RE = re.compile(
    r"(?:https?://|(?<![\w./-])(?=(?:www\.)?(?:drive\.usercontent|drive|docs|sheets|slides)"
    r"\.google\.com/))[^\s<>\"'`)\]]+",
    re.IGNORECASE,
)


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _is_drive_host(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in _DRIVE_HOSTS)


def _with_scheme(text: str) -> str:
    """``docs.google.com/...`` → ``https://docs.google.com/...`` (urlparse
    finds no host without a scheme)."""
    lowered = text.lower()
    if "://" not in text and any(
        lowered.startswith(h + "/") or lowered.startswith("www." + h + "/")
        for h in _DRIVE_HOSTS
    ):
        return "https://" + text
    return text


def is_google_drive_url(url: str) -> bool:
    """True if *url* points at Google Drive / Docs / Sheets / Slides."""
    return _is_drive_host(_host(_with_scheme(str(url or "").strip())))


def is_drive_folder_url(url: str) -> bool:
    """True if *url* is a Drive folder link (``/folders/<id>``)."""
    candidate = _with_scheme(str(url or "").strip())
    if _host(candidate) not in ("drive.google.com", "www.drive.google.com"):
        return False
    path = urlparse(candidate).path
    return "/folders/" in path or path.rstrip("/").endswith("/embeddedfolderview")


def _looks_like_bare_id(text: str) -> bool:
    """A token that is plausibly a Drive id on its own.

    Drive ids are random base64url strings: they always mix in digits or
    upper-case letters. Requiring that keeps words and slugs
    (``documentation``, ``my-report-final``) and pure numbers out; paths,
    file names and prose already fail on ``/``, ``.`` or whitespace.
    """
    if not _ID_CHARS_RE.match(text) or text.isdigit():
        return False
    return any(c.isdigit() or c.isupper() for c in text)


def _id_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if not _is_drive_host((parsed.hostname or "").lower()):
        return None
    parts = [p for p in parsed.path.split("/") if p]

    # /document/d/<id>/edit, /spreadsheets/d/<id>, /file/d/<id>/view,
    # /document/u/1/d/<id>/... — the segment after "d". "/d/e/<key>" is a
    # *published* copy whose key is not a Drive file id.
    for i, part in enumerate(parts[:-1]):
        if part == "d":
            nxt = parts[i + 1]
            if nxt == "e":
                return None
            return nxt if _ID_CHARS_RE.match(nxt) else None

    # /drive/folders/<id>, /drive/u/0/folders/<id>
    for i, part in enumerate(parts[:-1]):
        if part == "folders":
            nxt = parts[i + 1]
            return nxt if _ID_CHARS_RE.match(nxt) else None

    # open?id=, uc?id=&export=download, thumbnail?id=, download?id=, ...
    ids = parse_qs(parsed.query).get("id") or []
    if ids and _ID_CHARS_RE.match(ids[0]):
        return ids[0]
    return None


def _is_public_page(url: str) -> bool:
    """A Google Form or a published-to-web copy (``/d/e/<key>/pub...``).

    Both are made to be opened without signing in, and neither is a Drive
    file google_drive can read: a published key is not a file id, and a Form
    has no exportable content.
    """
    parts = [p for p in urlparse(url).path.split("/") if p]
    if "forms" in parts[:3]:  # /forms/..., /a/<domain>/forms/...
        return True
    return any(a == "d" and b == "e" for a, b in zip(parts, parts[1:]))


def is_public_google_page(url: str) -> bool:
    """True if *url* is a Google Form or a published (``/d/e/``) Doc/Sheet/Slides page."""
    candidate = _with_scheme(str(url or "").strip())
    return _is_drive_host(_host(candidate)) and _is_public_page(candidate)


def google_drive_can_open(url: str) -> bool:
    """True if *url* names a Drive file or folder the google_drive tool takes.

    A Drive/Docs/Sheets/Slides URL with an id in it. Forms and published
    pages are not (see :func:`is_public_google_page`), nor are Drive URLs
    without an id (``/drive/my-drive``) — redirecting those would leave the
    agent nothing to pass.
    """
    candidate = _with_scheme(str(url or "").strip())
    if not _is_drive_host(_host(candidate)) or _is_public_page(candidate):
        return False
    return _id_from_url(candidate) is not None


def drive_resource_key(text: str) -> str | None:
    """The ``resourcekey`` of the first Drive URL in *text*, or None.

    Older link-shared files only open with it (Drive's 2021 security update);
    google_drive sends it with the file's requests.
    """
    url = first_drive_url(text)
    if not url:
        return None
    try:
        keys = parse_qs(urlparse(url).query).get("resourcekey") or []
    except ValueError:
        return None
    key = keys[0].strip() if keys else ""
    return key if _RESOURCE_KEY_RE.match(key) else None


def drive_id_from_url(text: str) -> str | None:
    """The Drive file / folder id named by *text*, or None.

    *text* may be a Drive/Docs/Sheets/Slides URL (with or without scheme),
    free text containing one (the first Drive URL wins), or a bare id — which
    passes through unchanged when it looks like one (see
    :func:`_looks_like_bare_id`).
    """
    raw = str(text or "").strip().strip("<>\"'")
    if not raw:
        return None
    if _looks_like_bare_id(raw):
        return raw
    url = first_drive_url(raw)
    return _id_from_url(url) if url else None


def first_drive_url(text: str) -> str | None:
    """The first Drive URL in *text* (scheme added when missing), or None."""
    raw = str(text or "").strip().strip("<>\"'")
    candidate = _with_scheme(raw)
    if _is_drive_host(_host(candidate)) and not any(c.isspace() for c in candidate):
        return candidate
    for match in _URL_IN_TEXT_RE.finditer(raw):
        url = _with_scheme(match.group(0).rstrip(".,;:!?"))
        if _is_drive_host(_host(url)):
            return url
    return None


def google_drive_redirect(url_or_text: str, intro: str, *, connected: bool = False) -> str:
    """Steer a blocked Drive/Docs URL to the ``google_drive`` tool.

    Prints the exact call with the id pulled from the URL when there is one
    (``list`` for a folder link, ``read`` / ``info`` / ``download`` for a
    file), the generic instruction otherwise. *intro* says what was refused.
    *connected* says Google is connected, so a missing google_drive tool
    means it is not enabled for this agent — not "connect Google".
    """
    url = first_drive_url(url_or_text)
    file_id = _id_from_url(url) if url else None
    folder = bool(file_id) and is_drive_folder_url(url or "")
    # A link-shared file that needs its resource key gets the URL itself,
    # which carries the key; google_drive reads both from it.
    key = drive_resource_key(url) if url and file_id and not folder else None
    ref = url if key and "'" not in url else file_id

    lines = [f"{intro} Use the google_drive tool instead:"]
    if file_id and folder:
        lines += [
            f"  - google_drive(action='list', folder_id='{file_id}') lists the folder",
            f"  - google_drive(action='info', file_id='{file_id}') for folder metadata",
        ]
    elif file_id:
        lines += [
            f"  - google_drive(action='read', file_id='{ref}') returns the content "
            "inline (Docs as markdown, Sheets/Slides exported, PDF/DOCX/XLSX/PPTX extracted)",
            f"  - google_drive(action='info', file_id='{ref}') for file metadata",
            f"  - google_drive(action='download', file_id='{ref}') saves a local copy "
            "and returns its path",
        ]
    else:
        lines += [
            "  - google_drive(action='read', file_id='<id>') returns a file's content "
            "inline; the id is the part after /d/ or ?id= in a Drive link",
            "  - google_drive(action='list', folder_id='<id>') for a folder link "
            "(the part after /folders/)",
            "  - google_drive(action='info' | 'download', file_id='<id>') for "
            "metadata or a local copy",
        ]
    if connected:
        lines.append(
            "Google is connected. If google_drive is not among your tools, it is "
            "not enabled for this agent: tell the user so (connecting Google again "
            "will not help)."
        )
    else:
        lines.append(
            "If google_drive is not among your tools, ask the user to connect Google "
            "(Flight Deck → Connections → Google)."
        )
    return "\n".join(lines)
