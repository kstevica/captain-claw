"""Pictures found while researching the web, kept for the reports built from it.

``web_fetch`` keeps the pictures a page shows that could matter to the research
— ones with a real description (alt text, a caption, a title) inside the page's
content, not its logos, icons, avatars, tracking pixels or "related" teasers —
downloads them, and lists them in its result. A report then uses them by format:

* ``.md``   — the picture's ORIGINAL URL, an external link. The write/edit tools
  swap a saved copy's local path for that URL.
* ``.html`` — inline base64. The write/edit tools embed every ``<img>`` that
  points at a saved copy, another local image file or a remote image, so the
  file stands alone. Reading the file back shows each picture as a short
  ``[inline image …]`` placeholder, which the edit/write tools put back.
* ``.docx`` — the saved file, added with python-docx's ``add_picture``.

Saved copies live in ``saved/media/<session>/web-images/`` — a Flight Deck run's
workers share ``vfs:<project>/sources/images/`` instead, so the reporter sees
every teammate's — named ``<slug>-<sha8>.<ext>`` after their content, each with
a ``.json`` sidecar (original URL, page, alt, caption…) and an ``index.md``.
Every download goes through the public-only client: a page decides these URLs,
so it never gets to point the agent at a private address.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

from captain_claw.logging import get_logger

log = get_logger(__name__)

IMAGES_DIRNAME = "web-images"
INDEX_NAME = "index.md"

# What a kept picture may be: a download cap, then the size the copy is cut to.
_MAX_DOWNLOAD_BYTES = 12 * 1024 * 1024
_MAX_EDGE = 1600
# A JPEG/PNG/GIF this small (and within _MAX_EDGE) is kept byte for byte.
_KEEP_AS_IS_BYTES = 1_500_000
# Smaller than this, or stretched further, it is an icon, a divider or a banner.
_MIN_LONG_EDGE = 300
_MIN_SHORT_EDGE = 120
_MAX_ASPECT = 5.0
_DOWNLOAD_TIMEOUT_S = 10.0
_DOWNLOAD_CONCURRENCY = 4
# Pictures embedded in one written HTML file, and how long remote ones may take.
_MAX_EMBEDS = 40
_REMOTE_EMBED_BUDGET_S = 20.0
# Image CDNs (Wikimedia's among them) want a client that says who it is.
_USER_AGENT = "Mozilla/5.0 (compatible; CaptainClaw/0.1; +https://captain-claw.com)"

# A candidate needs this score to be kept (a caption, or real alt text inside
# the page's content, each clears it on their own).
_MIN_SCORE = 3

_MIME_EXT = {"image/jpeg": "jpg", "image/png": "png", "image/gif": "gif",
             "image/webp": "webp", "image/avif": "avif", "image/bmp": "bmp"}
_IMAGE_EXT_RE = re.compile(r"\.(?:jpe?g|png|gif|webp|avif|bmp|tiff?|heic)$", re.I)
_SKIP_EXT_RE = re.compile(r"\.(?:svg|ico|cur)$", re.I)

# Words in a picture's URL path that mark it as chrome, not content.
_JUNK_PATH_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:logos?|favicons?|icons?|avatars?|gravatar|sprites?|badges?|"
    r"spacer|blank|placeholder|emoji|spinner|1x1|pixel\.gif|transparent\.gif|"
    r"ads|adserver|doubleclick)(?:[^a-z0-9]|$)",
    re.I,
)
# The same for the <img>'s own class/id, plus share buttons and ad slots.
_JUNK_CLASS_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:logos?|favicons?|icons?|avatars?|gravatar|sprites?|badges?|"
    r"spacer|placeholder|emoji|spinner|ads?|advert\w*|share|sharing|social)(?:[^a-z0-9]|$)",
    re.I,
)
# Ancestors (by tag), and the class/id token prefixes of ancestors inside the
# page's content, that a content picture never sits in.
_SKIP_ANCESTOR_TAGS = frozenset({"nav", "footer", "aside", "form", "button"})
_SKIP_ANCESTOR_TOKEN_RE = re.compile(
    r"^(?:nav|navbar|menu|footer|sidebar|related|recommend\w*|promo\w*|advert\w*|ads?|"
    r"sponsor\w*|share|sharing|social|comments?|newsletter|subscribe|cookie\w*|"
    r"breadcrumbs?|author|byline|masthead)(?:$|[-_])",
    re.I,
)
# Alt text that names a file or a placeholder rather than what the picture shows.
_WEAK_ALT_RE = re.compile(
    r"^(?:image|img|photo|picture|pic|graphic|thumbnail|thumb|banner|logo|icon|"
    r"untitled|screenshot|figure|placeholder|null|undefined|none)\s*\d*$",
    re.I,
)
_FILENAME_ALT_RE = re.compile(r"^[\w-]+\.(?:jpe?g|png|gif|webp|avif|svg)$", re.I)
_LOGO_ALT_RE = re.compile(r"\b(?:logo|icon|avatar|badge)\b", re.I)
_ERROR_PAGE_RE = re.compile(
    r"^\W*(?:404|410|error 40\d)\b|\b(?:page|article|content) (?:was )?not found\b"
    r"|^\W*not found\W*$|\bpage (?:doesn't|does not|no longer) exist", re.I)
_LAZY_ATTRS = ("data-src", "data-lazy-src", "data-original", "data-url",
               "data-hi-res-src", "data-full-src", "data-orig-file", "src")


def code_mode() -> bool:
    """A code agent writing into a repo: its HTML is source, never a report to
    inline pictures into (``<img src="logo.png">`` must stay a file reference)."""
    return bool(os.environ.get("CLAW_WRITE_DIRECT") or os.environ.get("CLAW_CODE_AGENT"))


def normalize_session_id(raw: Any) -> str:
    """The write tool's session folder name for *raw* (``default`` when empty)."""
    value = str(raw or "").strip()
    if not value:
        return "default"
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in value).strip("-")
    return safe or "default"


def _clean(text: Any, limit: int) -> str:
    out = re.sub(r"\s+", " ", str(text or "")).strip()
    return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"


def _meaningful(text: str) -> bool:
    """Whether *text* says what a picture shows (not a file name or a stub)."""
    t = (text or "").strip()
    if len(t) < 8 or len(t.split()) < 2 and len(t) < 15:
        return False
    return not (_WEAK_ALT_RE.match(t) or _FILENAME_ALT_RE.match(t))


# ── Finding the pictures on a page ───────────────────────────────────


@dataclass
class PageImage:
    """A picture on a fetched page that could matter to the research."""

    url: str
    alt: str = ""
    caption: str = ""
    title: str = ""
    section: str = ""   # the nearest heading above it
    context: str = ""   # the text around it
    width: int = 0      # as declared in the markup (0 = unknown)
    height: int = 0
    score: int = 0

    def description(self) -> str:
        return self.caption or self.alt or self.title


def _int_attr(value: Any) -> int:
    match = re.match(r"\s*(\d+)", str(value or ""))
    return int(match.group(1)) if match else 0


def _best_from_srcset(value: str) -> str:
    """The largest candidate (≤ 2000w) of a ``srcset`` value."""
    best_url, best_w = "", -1.0
    for item in re.split(r",\s+", (value or "").strip()):
        parts = item.strip().split()
        if not parts:
            continue
        url = parts[0].rstrip(",")
        weight = 1.0
        if len(parts) > 1:
            desc = parts[1].rstrip(",").lower()
            try:
                if desc.endswith("w"):
                    weight = float(desc[:-1])
                    if weight > 2000:
                        weight = -weight  # any within the cap beats it; else the smallest above
                elif desc.endswith("x"):
                    weight = float(desc[:-1]) * 1000
            except ValueError:
                pass
        if url and not url.startswith("data:") and weight > best_w:
            best_url, best_w = url, weight
    return best_url


def _pick_src(img: Any) -> str:
    """The best real URL an ``<img>`` names (lazy-load attributes included)."""
    for attr in ("srcset", "data-srcset"):
        value = img.get(attr)
        if value:
            found = _best_from_srcset(value)
            if found:
                return found
    picture = img.find_parent("picture")
    if picture is not None:
        for source in picture.find_all("source"):
            found = _best_from_srcset(source.get("srcset") or source.get("data-srcset") or "")
            if found:
                return found
    for attr in _LAZY_ATTRS:
        value = (img.get(attr) or "").strip()
        if value and not value.startswith("data:"):
            return value
    return ""


def _tokens_of(tag: Any) -> list[str]:
    classes = tag.get("class") or []
    if isinstance(classes, str):
        classes = classes.split()
    return [str(c) for c in (*classes, tag.get("id") or "") if c]


def _caption_of(img: Any) -> str:
    figure = img.find_parent("figure")
    if figure is not None:
        cap = figure.find("figcaption")
        if cap is not None:
            return _clean(cap.get_text(" ", strip=True), 300)
    parent = img.parent
    for _ in range(3):
        if parent is None or getattr(parent, "name", None) in (None, "body", "html", "article", "main"):
            break
        for el in parent.find_all(class_=re.compile(r"caption", re.I), limit=3):
            if el is not img and el.find("img") is None:
                text = _clean(el.get_text(" ", strip=True), 300)
                if text:
                    return text
        parent = parent.parent
    return ""


def _context_of(img: Any, caption: str) -> str:
    block = img.find_parent(["p", "li", "td", "blockquote"])
    if block is not None:
        text = _clean(block.get_text(" ", strip=True), 220)
        if len(text) > 30 and text != caption:
            return text
    bits: list[str] = []
    for found in (img.find_previous("p"), img.find_next("p")):
        if found is not None:
            text = _clean(found.get_text(" ", strip=True), 110)
            if len(text) > 20 and text != caption:
                bits.append(text)
    return " … ".join(bits)


def _strip_query_key(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.netloc.lower()}{parts.path}"


def find_page_images(soup: Any, base_url: str | None) -> list[PageImage]:
    """The pictures on a parsed page that could matter, best first.

    *soup* is a BeautifulSoup tree, read BEFORE scripts and the like are
    stripped (a ``<noscript>`` fallback ``<img>`` is often the real one). A
    picture is kept when it has a description — a caption, alt text that says
    what it shows, or a title — and sits in the page's content rather than its
    navigation, sidebar, ads or share bar.
    """
    # An error page's pictures (a 404 mascot) are never research material; a
    # headless fetch can't see the status, so go by what the page calls itself.
    heading = soup.find("h1")
    for label in (soup.title.string if soup.title and soup.title.string else "",
                  heading.get_text(" ", strip=True) if heading is not None else ""):
        if _ERROR_PAGE_RE.search(label or ""):
            return []
    content_root = (soup.find("article") or soup.find("main")
                    or soup.find(attrs={"role": "main"}))
    og = ""
    meta = soup.find("meta", attrs={"property": "og:image"}) or soup.find(
        "meta", attrs={"name": "twitter:image"})
    if meta is not None and meta.get("content"):
        og = urljoin(base_url or "", str(meta["content"]).strip())

    found: dict[str, PageImage] = {}
    for img in soup.find_all("img"):
        raw = _pick_src(img)
        if not raw:
            continue
        url = urljoin(base_url, raw) if base_url else raw
        if not url.lower().startswith(("http://", "https://")):
            continue
        path = unquote(urlsplit(url).path)
        if _SKIP_EXT_RE.search(path) or _JUNK_PATH_RE.search(path):
            continue
        if any(_JUNK_CLASS_RE.search(t) for t in _tokens_of(img)):
            continue
        width, height = _int_attr(img.get("width")), _int_attr(img.get("height"))
        if (width and width < _MIN_SHORT_EDGE) or (height and height < _MIN_SHORT_EDGE // 2):
            continue

        in_content = content_root is not None and any(p is content_root for p in img.parents)
        skip = False
        teaser = False
        for depth, parent in enumerate(img.parents):
            name = getattr(parent, "name", None)
            if name in (None, "body", "html", "[document]") or parent is content_root:
                break  # wrappers around the article don't describe the picture
            if name in _SKIP_ANCESTOR_TAGS or (name == "header" and not in_content):
                skip = True
                break
            if depth < 6 and any(_SKIP_ANCESTOR_TOKEN_RE.match(t) for t in _tokens_of(parent)):
                skip = True
                break
            if name == "a" and parent.get("href") and not looks_like_image_url(str(parent["href"])):
                teaser = True  # links away (a teaser), not to its own full size
        if skip:
            continue

        alt = _clean(img.get("alt"), 240)
        title = _clean(img.get("title") or img.get("aria-label"), 200)
        caption = _caption_of(img)
        if alt and _LOGO_ALT_RE.search(alt) and len(alt.split()) <= 4:
            continue
        good_alt = _meaningful(alt)
        good_title = _meaningful(title)
        if not (caption or good_alt or good_title):
            continue  # nothing says what it shows — can't judge its relevance

        score = 0
        score += 3 if caption else 0
        score += (2 + (1 if len(alt.split()) >= 4 else 0)) if good_alt else 0
        score += 1 if good_title else 0
        score += 1 if img.find_parent("figure") is not None else 0
        if content_root is not None:
            score += 1 if in_content else -2
        if max(width, height) >= 400:
            score += 1
        if teaser:
            score -= 1
        if og and _strip_query_key(og) == _strip_query_key(url):
            score += 1
        if score < _MIN_SCORE:
            continue

        heading = img.find_previous(["h1", "h2", "h3", "h4"])
        cand = PageImage(
            url=url, alt=alt if good_alt else "", caption=caption,
            title=title if good_title else "",
            section=_clean(heading.get_text(" ", strip=True), 120) if heading is not None else "",
            context=_context_of(img, caption), width=width, height=height, score=score,
        )
        # One picture often appears twice (a hero crop and an in-body one, a
        # lazy <img> and its <noscript> twin): same file, or the same words.
        key = _strip_query_key(url)
        said = f"{cand.caption}|{cand.alt}|{cand.title}".lower()
        twin = next((k for k, c in found.items()
                     if f"{c.caption}|{c.alt}|{c.title}".lower() == said), None)
        key = twin or key
        prev = found.get(key)
        if prev is None or cand.score > prev.score:
            found[key] = cand
    # Best first; among equals, page order (dicts keep insertion order).
    return sorted(found.values(), key=lambda c: -c.score)


# ── Checking and normalising a downloaded picture ────────────────────


def sniff_mime(head: bytes) -> str:
    """The image type *head* starts with ('' when it isn't a raster image)."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[4:12] in (b"ftypavif", b"ftypavis"):
        return "image/avif"
    if head.startswith(b"BM"):
        return "image/bmp"
    return ""


def normalize_image(data: bytes, *, check_size: bool = True) -> tuple[bytes, str, int, int] | None:
    """``(bytes, mime, width, height)`` of a copy fit for a report, or None.

    A JPEG, PNG or GIF within the size limits is kept byte for byte; anything
    else (WebP, AVIF, TIFF, BMP, an oversized photo) is re-encoded — PNG when it
    has transparency, else JPEG — and cut to ``_MAX_EDGE`` on its long side, so
    python-docx (which can't read WebP/AVIF) and a browser both take it. With
    *check_size*, an icon-sized, a banner-shaped or a broken picture is refused.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover — Pillow is a core dependency
        mime = sniff_mime(data[:16])
        return (data, mime, 0, 0) if mime in ("image/jpeg", "image/png", "image/gif") else None
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = (im.format or "").upper()
            width, height = im.size
            if width <= 0 or height <= 0:
                return None
            if check_size:
                long_edge, short_edge = max(width, height), min(width, height)
                if (long_edge < _MIN_LONG_EDGE or short_edge < _MIN_SHORT_EDGE
                        or long_edge / short_edge > _MAX_ASPECT):
                    return None
            fits = max(width, height) <= _MAX_EDGE
            if fmt in ("JPEG", "PNG") and fits and len(data) <= _KEEP_AS_IS_BYTES:
                return data, f"image/{fmt.lower()}", width, height
            if fmt == "GIF" and fits and len(data) <= _KEEP_AS_IS_BYTES * 2:
                return data, "image/gif", width, height
            if fmt == "JPEG":
                im.draft("RGB", (_MAX_EDGE, _MAX_EDGE))
            im.seek(0)
            frame = im.copy()
        frame.thumbnail((_MAX_EDGE, _MAX_EDGE))
        alpha = frame.mode in ("RGBA", "LA", "PA") or (
            frame.mode == "P" and "transparency" in frame.info)
        out = io.BytesIO()
        if alpha:
            frame.convert("RGBA").save(out, "PNG", optimize=True)
            mime = "image/png"
        else:
            frame.convert("RGB").save(out, "JPEG", quality=85, optimize=True, progressive=True)
            mime = "image/jpeg"
        return out.getvalue(), mime, frame.size[0], frame.size[1]
    except Exception as e:  # noqa: BLE001 — a broken picture is simply not kept
        log.debug("image normalise failed", error=str(e))
        return None


# ── The saved copies ─────────────────────────────────────────────────


@dataclass
class SavedImage:
    """A picture kept on disk, with where it came from."""

    name: str
    url: str = ""          # the original image URL ('' for one made locally)
    page_url: str = ""     # the page it was found on
    alt: str = ""
    caption: str = ""
    title: str = ""
    section: str = ""
    context: str = ""
    width: int = 0
    height: int = 0
    mime: str = ""
    bytes: int = 0
    sha8: str = ""
    saved_at: int = 0
    path: str = field(default="", compare=False)    # on disk (not stored)
    shown: str = field(default="", compare=False)   # as the agent writes it (not stored)

    def description(self) -> str:
        return self.caption or self.alt or self.title

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("path", None)
        data.pop("shown", None)
        return data


def _slug(text: str, limit: int = 48) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:limit].strip("-") or "image"


def _sha8(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:8]


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        tmp = ""
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


class ImageStore:
    """One folder of saved pictures (``web-images/`` or a run's ``sources/images/``)."""

    def __init__(self, directory: Path, shown_prefix: str):
        self.directory = Path(directory)
        self.shown_prefix = shown_prefix.rstrip("/")

    # -- where ---------------------------------------------------------

    @staticmethod
    def run_store() -> ImageStore | None:
        """The run's shared store, for a Flight Deck worker bound to a run folder."""
        from captain_claw import context_spill

        found = context_spill.run_sources_dir()
        if not found:
            return None
        directory, prefix = found
        return ImageStore(directory / "images", f"{prefix}/images")

    @staticmethod
    def session_store(saved_root: Path, runtime_base: Path | None, session_id: Any) -> ImageStore:
        directory = Path(saved_root) / "media" / normalize_session_id(session_id) / IMAGES_DIRNAME
        shown = str(directory)
        if runtime_base is not None:
            try:
                shown = directory.resolve().relative_to(Path(runtime_base).resolve()).as_posix()
            except (ValueError, OSError):
                pass
        return ImageStore(directory, shown)

    @classmethod
    def for_tool(cls, kwargs: dict[str, Any]) -> ImageStore:
        """Where this tool call keeps pictures: the run's folder, else the session's."""
        run = cls.run_store()
        if run is not None:
            return run
        saved_root, runtime_base = tool_roots(kwargs)
        return cls.session_store(saved_root, runtime_base, kwargs.get("_session_id"))

    def shown_for(self, name: str) -> str:
        return f"{self.shown_prefix}/{name}"

    # -- what ----------------------------------------------------------

    def entries(self) -> list[SavedImage]:
        if not self.directory.is_dir():
            return []
        out: list[SavedImage] = []
        for sidecar in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict) or not data.get("name"):
                continue
            known = {k: v for k, v in data.items() if k in SavedImage.__dataclass_fields__}
            entry = SavedImage(**known)
            image = self.directory / entry.name
            if not image.is_file():
                continue
            entry.path, entry.shown = str(image), self.shown_for(entry.name)
            out.append(entry)
        out.sort(key=lambda e: e.saved_at)
        return out

    def find(self, *, url: str = "", name: str = "", sha8: str = "") -> SavedImage | None:
        for entry in self.entries():
            if (url and entry.url == url) or (name and entry.name == name) or (
                    sha8 and entry.sha8 == sha8):
                return entry
        return None

    def for_page(self, page_url: str) -> list[SavedImage]:
        return [e for e in self.entries() if e.page_url == page_url]

    # -- keeping -------------------------------------------------------

    def save(self, data: bytes, mime: str, width: int, height: int, *,
             url: str = "", page_url: str = "", meta: PageImage | None = None,
             alt: str = "") -> SavedImage:
        """Keep normalised picture bytes (re-using an identical copy) and index it."""
        self.directory.mkdir(parents=True, exist_ok=True)
        sha8 = _sha8(data)
        same = self.find(sha8=sha8)
        if same is not None:
            return same
        base = (meta.description() if meta else "") or alt or Path(urlsplit(url).path).stem
        name = f"{_slug(base)}-{sha8}.{_MIME_EXT.get(mime, 'img')}"
        entry = SavedImage(
            name=name, url=url, page_url=page_url,
            alt=(meta.alt if meta else alt), caption=meta.caption if meta else "",
            title=meta.title if meta else "", section=meta.section if meta else "",
            context=meta.context if meta else "", width=width, height=height,
            mime=mime, bytes=len(data), sha8=sha8, saved_at=int(time.time()),
        )
        _atomic_write(self.directory / name, data)
        _atomic_write(self.directory / f"{name}.json",
                      json.dumps(entry.to_json(), ensure_ascii=False, indent=1).encode("utf-8"))
        entry.path, entry.shown = str(self.directory / name), self.shown_for(name)
        self.write_index()
        return entry

    def write_index(self) -> None:
        """``index.md``: every saved picture and how a report uses it."""
        entries = self.entries()
        lines = [
            "# Pictures saved while researching", "",
            "Use the ones that support a finding, next to it, with their caption and source: "
            "**.md** → `![caption](original URL)`; **.html** → `<img src=\"<saved file>\">` "
            "(the write tool embeds it as base64); **.docx** → "
            "`doc.add_picture(\"<saved file>\", width=Inches(6))`; **.pptx** → the pptx tool's "
            "`image` with the saved file.", "",
        ]
        for e in entries:
            desc = e.description() or "(no description)"
            lines.append(f"- `{e.shown}` — {e.width}×{e.height} — {desc}")
            if e.section:
                lines.append(f"  - section: {e.section}")
            if e.url:
                lines.append(f"  - original: {e.url}")
            if e.page_url:
                lines.append(f"  - page: {e.page_url}")
        try:
            _atomic_write(self.directory / INDEX_NAME, ("\n".join(lines) + "\n").encode("utf-8"))
        except OSError as e:
            log.debug("image index write failed", error=str(e))


def tool_roots(kwargs: dict[str, Any]) -> tuple[Path, Path | None]:
    """``(saved_root, runtime_base)`` as the registry handed them to a tool."""
    runtime_raw = kwargs.get("_runtime_base_path")
    runtime_base = Path(runtime_raw).expanduser().resolve() if runtime_raw is not None else None
    saved_raw = kwargs.get("_saved_base_path")
    if saved_raw is not None:
        saved_root = Path(saved_raw).expanduser().resolve()
    elif runtime_base is not None:
        saved_root = runtime_base / "saved"
    else:
        saved_root = Path.cwd().resolve() / "saved"
    return saved_root, runtime_base


def lookup_stores(kwargs: dict[str, Any]) -> list[ImageStore]:
    """Every store a written report may point into: the run's, this session's,
    then the agent's other sessions' (a report written later in a new session)."""
    saved_root, runtime_base = tool_roots(kwargs)
    stores: list[ImageStore] = []
    run = ImageStore.run_store()
    if run is not None:
        stores.append(run)
    own = ImageStore.session_store(saved_root, runtime_base, kwargs.get("_session_id"))
    stores.append(own)
    media = saved_root / "media"
    try:
        others = sorted(media.glob(f"*/{IMAGES_DIRNAME}"))
    except OSError:
        others = []
    for directory in others:
        if directory.resolve() != own.directory.resolve():
            stores.append(ImageStore.session_store(saved_root, runtime_base, directory.parent.name))
    return stores


# ── Downloading ──────────────────────────────────────────────────────


def looks_like_image_url(url: str) -> bool:
    """Whether *url*'s path names a raster image file."""
    try:
        return bool(_IMAGE_EXT_RE.search(unquote(urlsplit(url).path)))
    except ValueError:
        return False


def make_client() -> Any:
    """The public-only HTTP client every picture download goes through."""
    from captain_claw import speaker

    client = speaker.make_public_http_client()
    client.timeout = _DOWNLOAD_TIMEOUT_S
    return client


async def download_image(client: Any, url: str, referer: str = "", *, allow_svg: bool = False) -> bytes | None:
    """The bytes of the picture at *url* (None if it isn't one or can't be had).
    An SVG only for a caller who draws it itself (the pptx tool): a report can't."""
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8",
    }
    if referer:
        headers["Referer"] = referer
    try:
        async with client.stream("GET", url, headers=headers) as resp:
            if resp.status_code != 200:
                return None
            ctype = str(resp.headers.get("content-type") or "").lower()
            kinds = ("image/", "application/octet-stream", "binary/")
            if allow_svg:  # Often served as plain text or XML.
                kinds += ("text/", "application/xml")
            if ctype and not ctype.startswith(kinds):
                return None
            if "svg" in ctype and not allow_svg:
                return None
            chunks: list[bytes] = []
            size = 0
            async for chunk in resp.aiter_bytes():
                size += len(chunk)
                if size > _MAX_DOWNLOAD_BYTES:
                    return None
                chunks.append(chunk)
    except Exception as e:  # noqa: BLE001 — a picture that won't come is skipped
        log.debug("image download failed", url=url, error=str(e))
        return None
    data = b"".join(chunks)
    return data if sniff_mime(data[:16]) or (allow_svg and b"<svg" in data[:4096]) else None


async def save_remote(store: ImageStore, client: Any, url: str, *, page_url: str = "",
                      meta: PageImage | None = None, alt: str = "",
                      check_size: bool = True) -> SavedImage | None:
    """Download, check and keep one picture (an already-kept URL is re-used)."""
    known = store.find(url=url)
    if known is not None:
        return known
    data = await download_image(client, url, referer=page_url)
    if data is None:
        return None
    fitted = await asyncio.to_thread(normalize_image, data, check_size=check_size)
    if fitted is None:
        return None
    body, mime, width, height = fitted
    try:
        return await asyncio.to_thread(store.save, body, mime, width, height, url=url,
                                       page_url=page_url, meta=meta, alt=alt)
    except OSError as e:
        log.warning("image save failed", url=url, error=str(e))
        return None


async def _settle(tasks: list[asyncio.Task], budget_s: float) -> None:
    """Wait up to *budget_s* for *tasks*; cancel (and reap) the rest."""
    _, pending = await asyncio.wait(tasks, timeout=max(1.0, budget_s))
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _result_of(task: asyncio.Task) -> Any:
    """A finished task's result, or None (cancelled, or it raised)."""
    if not task.done() or task.cancelled() or task.exception() is not None:
        return None
    return task.result()


async def save_page_images(candidates: list[PageImage], page_url: str, store: ImageStore,
                           *, limit: int, budget_s: float, client: Any = None) -> list[SavedImage]:
    """Keep up to *limit* of a page's candidates within *budget_s* seconds.

    Candidates are tried best first; one that fails (a 403, a too-small or
    broken file) lets the next in. Whatever hasn't finished when the budget
    runs out is dropped — a slow CDN never holds up the fetch.
    """
    if limit <= 0 or not candidates:
        return []
    own_client = client is None
    client = client or make_client()
    sem = asyncio.Semaphore(_DOWNLOAD_CONCURRENCY)
    pool = candidates[: limit * 3]

    async def _one(cand: PageImage) -> SavedImage | None:
        async with sem:
            return await save_remote(store, client, cand.url, page_url=page_url, meta=cand)

    tasks = [asyncio.create_task(_one(c)) for c in pool]
    try:
        await _settle(tasks, budget_s)
        kept: list[SavedImage] = []
        seen: set[str] = set()
        for task in tasks:  # candidate order, not finish order
            entry = _result_of(task)
            if entry is not None and entry.name not in seen:
                seen.add(entry.name)
                kept.append(entry)
        return kept[:limit]
    finally:
        if own_client:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass


# ── What web_fetch says about them ───────────────────────────────────

_REPORT_RULE = (
    "Put the ones that support a finding next to it, with caption + source page: "
    ".md → ![caption](original URL); .html → <img src=\"saved path\"> (write embeds it "
    "as base64); .docx → doc.add_picture(\"saved path\"); .pptx → the pptx tool's image "
    "with the saved path. Skip any that add nothing."
)


def format_saved(entries: list[SavedImage], *, earlier: bool = False) -> str:
    """The ``[Images …]`` block a fetch result ends with ('' when none)."""
    if not entries:
        return ""
    when = "saved from this page earlier in this run" if earlier else "saved from this page"
    lines = [f"[Images — {len(entries)} relevant-looking picture(s) {when}. {_REPORT_RULE}]"]
    for i, e in enumerate(entries, 1):
        bits = [f"{i}. {e.shown}", f"{e.width}×{e.height}"]
        if e.caption:
            bits.append(f"caption \"{e.caption}\"")
        if e.alt and e.alt != e.caption:
            bits.append(f"alt \"{e.alt}\"")
        if e.title and not (e.caption or e.alt):
            bits.append(f"title \"{e.title}\"")
        if e.section:
            bits.append(f"section \"{e.section}\"")
        if e.context:
            bits.append(f"near \"{_clean(e.context, 140)}\"")
        if e.url:
            bits.append(f"original {e.url}")
        lines.append(" — ".join(bits))
    return "\n".join(lines)


def format_unsaved(candidates: list[PageImage], limit: int) -> str:
    """The block for pictures listed but not kept (a shared agent's member)."""
    picks = candidates[:limit]
    if not picks:
        return ""
    lines = [f"[Images — {len(picks)} relevant-looking picture(s) on this page. "
             "To use one in a report, link its original URL: ![caption](URL).]"]
    for i, c in enumerate(picks, 1):
        bits = [f"{i}. {c.url}"]
        desc = c.description()
        if desc:
            bits.append(f"\"{desc}\"")
        if c.section:
            bits.append(f"section \"{c.section}\"")
        lines.append(" — ".join(bits))
    return "\n".join(lines)


def format_direct(entry: SavedImage) -> str:
    """What web_fetch says after saving an image URL fetched directly."""
    return (f"[URL: {entry.url}]\n[Image saved: {entry.shown} — {entry.width}×{entry.height}, "
            f"{entry.mime}, {max(1, entry.bytes // 1024)} KB]\n\n"
            "This URL is a picture, not a page — it is saved for your report. "
            + _REPORT_RULE)


# ── Inline pictures in HTML: placeholders for reading, base64 for writing ──

_DATA_URI_RE = re.compile(r"data:(image/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/]{200,}={0,2})")
_PLACEHOLDER_RE = re.compile(
    r"(?:data:(image/[A-Za-z0-9.+-]+);base64,)?\[inline image ([0-9a-f]{8})[^\]\n]{0,40}\]")
PLACEHOLDER_NOTE = (
    "inline base64 picture(s) shown as \"[inline image …]\" placeholders — leave them "
    "as they are; edit/write put the pictures back"
)


def collapse_inline_images(text: str) -> tuple[str, dict[str, str]]:
    """*text* with each inline base64 picture as a short placeholder, and the
    ``{sha8: base64}`` map that puts them back (sha8 of the picture's bytes,
    the same id a saved copy's name ends with)."""
    if "base64," not in text:
        return text, {}
    mapping: dict[str, str] = {}

    def _sub(m: re.Match) -> str:
        b64 = m.group(2)
        try:
            raw = base64.b64decode(b64 + "=" * (-len(b64) % 4))
        except (ValueError, TypeError):
            return m.group(0)
        sha8 = _sha8(raw)
        mapping[sha8] = b64
        return f"data:{m.group(1)};base64,[inline image {sha8} · {max(1, len(raw) // 1024)} KB]"

    return _DATA_URI_RE.sub(_sub, text), mapping


def collapsed_size(path: Path) -> int:
    """Bytes of an HTML file with its inline pictures as placeholders (any
    other file: its size) — what the read/edit size limits measure."""
    path = Path(path)
    if path.suffix.lower() not in HTML_SUFFIXES:
        return path.stat().st_size
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return path.stat().st_size
    return len(collapse_inline_images(text)[0].encode("utf-8"))


def _store_b64(stores: list[ImageStore], sha8: str) -> tuple[str, str] | None:
    entry = _find_saved(stores, sha8=sha8)
    if entry is None:
        return None
    try:
        data = Path(entry.path).read_bytes()
    except OSError:
        return None
    return entry.mime or sniff_mime(data[:16]), base64.b64encode(data).decode("ascii")


def restore_inline_images(text: str, mapping: dict[str, str],
                          stores: list[ImageStore] | None = None) -> tuple[str, int, list[str]]:
    """Put placeholders back as base64: ``(text, restored, missing ids)``."""
    if "[inline image " not in text:
        return text, 0, []
    restored = 0
    missing: list[str] = []

    def _sub(m: re.Match) -> str:
        nonlocal restored
        mime, sha8 = m.group(1), m.group(2)
        b64 = mapping.get(sha8)
        if b64 is None and stores:
            found = _store_b64(stores, sha8)
            if found is not None:
                mime, b64 = mime or found[0], found[1]
        if b64 is None:
            missing.append(sha8)
            return m.group(0)
        if not mime:
            try:
                mime = sniff_mime(base64.b64decode(b64[:24] + "=" * (-len(b64[:24]) % 4)))
            except (ValueError, TypeError):
                mime = ""
        restored += 1
        return f"data:{mime or 'image/png'};base64,{b64}"

    return _PLACEHOLDER_RE.sub(_sub, text), restored, missing


# ── Pictures in a report the agent writes ────────────────────────────

_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.I)
_SRC_ATTR_RE = re.compile(r"""(\ssrc\s*=\s*)(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""", re.I)
_SRCSET_ATTR_RE = re.compile(r"""\s(?:srcset|sizes)\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'>]+)""", re.I)
_ALT_ATTR_RE = re.compile(r"""\salt\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_MD_IMAGE_RE = re.compile(r"(!\[[^\]\n]*\]\(\s*)(<?)([^)\s>]+)(>?(?:\s+\"[^\"\n]*\")?\s*\))")
HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})
MD_SUFFIXES = frozenset({".md", ".markdown"})


@dataclass
class ReportImages:
    """What the write/edit tools did with a report's pictures."""

    embedded: int = 0
    embedded_bytes: int = 0
    linked: int = 0
    restored: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)

    def note(self) -> str:
        parts: list[str] = []
        if self.embedded:
            parts.append(f"Embedded {self.embedded} picture(s) inline as base64 "
                         f"({max(1, self.embedded_bytes // 1024)} KB) — the file stands alone.")
        if self.restored:
            parts.append(f"Kept {self.restored} inline picture(s) from the placeholders.")
        if self.linked:
            parts.append(f"Linked {self.linked} saved picture(s) to their original URLs "
                         "(Markdown reports use external links).")
        if self.failed:
            shown = "; ".join(f"{src[:90]} ({why})" for src, why in self.failed[:6])
            parts.append(
                f"{len(self.failed)} picture(s) could not be embedded: {shown}. Point those "
                "<img> tags at a saved picture (web_fetch an image URL to save it) or drop them.")
        return " ".join(parts)


def _allowed_roots(kwargs: dict[str, Any]) -> list[Path]:
    saved_root, runtime_base = tool_roots(kwargs)
    roots = [saved_root]
    if runtime_base is not None:
        roots.append(runtime_base)
    try:
        from captain_claw import vfs

        roots.append(vfs.user_root())
    except Exception:  # noqa: BLE001 — no VFS here
        pass
    out: list[Path] = []
    for root in roots:
        try:
            out.append(Path(root).resolve())
        except OSError:
            continue
    return out


def _within(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _resolve_local(src: str, file_path: Path | None, kwargs: dict[str, Any],
                   stores: list[ImageStore]) -> Path | None:
    """The local image file an ``<img src>`` / ``![](…)`` target names, if any.

    Only a file inside the agent's own folders (saved/, its workspace, its VFS)
    is ever read — a report can't be used to carry anything else out.
    """
    value = unquote(src.strip())
    if value.lower().startswith("file://"):
        value = value[7:]
    value = value.split("?", 1)[0].split("#", 1)[0]
    if not value:
        return None
    # A saved copy's name is unique (it ends with its sha8), wherever it's pointed at.
    entry = _find_saved(stores, name=Path(value).name)
    if entry is not None:
        return Path(entry.path)
    from captain_claw.vfs import is_vfs_path, resolve_vfs_path

    if is_vfs_path(value):
        target = resolve_vfs_path(value)
        return target if target is not None and target.is_file() else None
    roots = _allowed_roots(kwargs)
    saved_root, runtime_base = tool_roots(kwargs)
    raw = Path(value).expanduser()
    tries = [raw] if raw.is_absolute() else [
        *([file_path.parent / raw] if file_path is not None else []),
        *([runtime_base / raw] if runtime_base is not None else []),
        saved_root / raw, saved_root.parent / raw,
    ]
    for cand in tries:
        try:
            real = cand.resolve()
        except OSError:
            continue
        if real.is_file() and _within(real, roots):
            return real
    return None


def _local_b64(path: Path) -> tuple[str, str, int] | None:
    """``(mime, base64, bytes)`` of a local picture — re-encoded when too big."""
    try:
        if path.stat().st_size > _MAX_DOWNLOAD_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    mime = sniff_mime(data[:16])
    if not mime:
        return None
    if len(data) > _KEEP_AS_IS_BYTES * 2 or mime in ("image/bmp",):
        fitted = normalize_image(data, check_size=False)
        if fitted is None:
            return None
        data, mime = fitted[0], fitted[1]
    return mime, base64.b64encode(data).decode("ascii"), len(data)


def _absolute(src: str) -> str:
    return f"https:{src}" if src.startswith("//") else src


def _find_saved(stores: list[ImageStore], **query: str) -> SavedImage | None:
    for store in stores:
        entry = store.find(**query)
        if entry is not None:
            return entry
    return None


def _src_of(tag: str) -> tuple[re.Match | None, str]:
    m = _SRC_ATTR_RE.search(tag)
    if m is None:
        return None, ""
    return m, (m.group(2) if m.group(2) is not None else
               m.group(3) if m.group(3) is not None else m.group(4) or "")


def _alt_of(tag: str) -> str:
    m = _ALT_ATTR_RE.search(tag)
    return _clean((m.group(1) if m and m.group(1) is not None else m.group(2) if m else ""), 200)


async def embed_html_images(text: str, kwargs: dict[str, Any], *, file_path: Path | None = None,
                            previous: str = "", allow_remote: bool = True) -> tuple[str, ReportImages]:
    """*text* with its pictures inline as base64, and what was done.

    Placeholders a read showed are put back first (from *previous*, the file's
    content before this write, else from the saved copies). Then every
    ``<img src>`` naming a saved copy, a local image file or (with
    *allow_remote*) a remote image is embedded; a remote one is saved too, so
    the next report re-uses it. ``srcset``/``sizes`` go — a browser would
    otherwise prefer them over the embedded picture.
    """
    report = ReportImages()
    stores = lookup_stores(kwargs)
    _, mapping = collapse_inline_images(previous) if previous else ("", {})
    text, report.restored, missing = restore_inline_images(text, mapping, stores)
    report.failed.extend((f"[inline image {sha}]", "picture not found") for sha in missing)

    tags = list(dict.fromkeys(_IMG_TAG_RE.findall(text)))
    srcs: dict[str, str] = {}  # src → alt (first seen)
    for tag in tags:
        _, src = _src_of(tag)
        src = src.strip()
        if src and not src.lower().startswith("data:"):
            srcs.setdefault(src, _alt_of(tag))
    if not srcs:
        return text, report

    resolved: dict[str, tuple[str, str, int]] = {}
    remote: list[str] = []
    for src in list(srcs)[:_MAX_EMBEDS]:
        if src.lower().startswith(("http://", "https://", "//")):
            known = _find_saved(stores, url=_absolute(src))
            found = await asyncio.to_thread(_local_b64, Path(known.path)) if known else None
            if found is not None:
                resolved[src] = found
            else:
                remote.append(src)
            continue
        local = _resolve_local(src, file_path, kwargs, stores)
        found = await asyncio.to_thread(_local_b64, local) if local is not None else None
        if found is None:
            report.failed.append((src, "not a picture in your folders" if local is None
                                  else "not an image file"))
        else:
            resolved[src] = found

    if remote and allow_remote:
        store = ImageStore.for_tool(kwargs)
        client = make_client()
        sem = asyncio.Semaphore(_DOWNLOAD_CONCURRENCY)

        async def _fetch(src: str) -> SavedImage | None:
            async with sem:
                return await save_remote(store, client, _absolute(src), alt=srcs[src],
                                         check_size=False)

        tasks = [asyncio.create_task(_fetch(s)) for s in remote]
        try:
            await _settle(tasks, _REMOTE_EMBED_BUDGET_S)
            for src, task in zip(remote, tasks):
                entry = _result_of(task)
                found = await asyncio.to_thread(_local_b64, Path(entry.path)) if entry else None
                if found is None:
                    report.failed.append((src, "too slow" if task.cancelled() else "download failed"))
                else:
                    resolved[src] = found
        finally:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass
    elif remote:
        report.failed.extend((src, "remote pictures aren't fetched here") for src in remote)

    if not resolved:
        return text, report

    def _sub(m: re.Match) -> str:
        tag = m.group(0)
        src_m, src = _src_of(tag)
        found = resolved.get(src.strip()) if src_m is not None else None
        if found is None:
            return tag
        mime, b64, size = found
        report.embedded += 1
        report.embedded_bytes += size
        new = tag[: src_m.start()] + f'{src_m.group(1)}"data:{mime};base64,{b64}"' + tag[src_m.end():]
        return _SRCSET_ATTR_RE.sub("", new)

    return _IMG_TAG_RE.sub(_sub, text), report


def link_markdown_images(text: str, kwargs: dict[str, Any],
                         file_path: Path | None = None) -> tuple[str, ReportImages]:
    """*text* with saved pictures linked by their ORIGINAL URL (``![](…)`` and
    ``<img src>``): a Markdown report points at the web, not at a local file."""
    report = ReportImages()
    if "](" not in text and "<img" not in text.lower():
        return text, report
    stores = lookup_stores(kwargs)

    def _original(src: str) -> str:
        value = src.strip()
        if not value or value.lower().startswith(("http://", "https://", "//", "data:", "#")):
            return ""
        entry = _find_saved(stores, name=Path(unquote(value.split("?", 1)[0].split("#", 1)[0])).name)
        return entry.url if entry is not None else ""

    def _md(m: re.Match) -> str:
        url = _original(m.group(3))
        if not url:
            return m.group(0)
        report.linked += 1
        return f"{m.group(1)}{m.group(2)}{url}{m.group(4)}"

    def _tag(m: re.Match) -> str:
        tag = m.group(0)
        src_m, src = _src_of(tag)
        url = _original(src) if src_m is not None else ""
        if not url:
            return tag
        report.linked += 1
        return tag[: src_m.start()] + f'{src_m.group(1)}"{url}"' + tag[src_m.end():]

    text = _MD_IMAGE_RE.sub(_md, text)
    text = _IMG_TAG_RE.sub(_tag, text)
    return text, report
