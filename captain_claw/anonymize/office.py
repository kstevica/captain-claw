"""Office files anonymized inside their own zip.

A ``.docx``, ``.xlsx`` or ``.pptx`` comes back as the same kind of file:
the text is replaced where it stands — in Word's runs, a slide's text runs,
a sheet's cells — so formatting, pictures, charts and everything else stay
byte for byte. Loading and re-saving with python-docx / openpyxl would drop
charts and images, so this works on the OOXML parts directly (lxml, entities
off, no network).

What is covered, beyond the visible text:

* Word: headers, footers, footnotes, endnotes, comments, text boxes, the text
  of **tracked deletions** (``w:delText`` — still in the file), field codes
  (``HYPERLINK "mailto:…"``), external link targets; every ``author`` /
  ``initials`` attribute (comments, tracked changes, people.xml) is blanked.
* Excel: shared strings, inline strings, cached formula results, formulas,
  long numbers (an OIB or a phone stored as a number), comments and threaded
  comments, sheet names (formulas that point at a renamed sheet follow it),
  text boxes, chart labels; comment authors and people are blanked. Column
  headers work as hints, as in a CSV.
* PowerPoint: slides, notes, layouts and masters, comments, chart labels,
  external link targets; comment authors are blanked.
* Every kind: the document properties — author and "last modified by" are
  emptied, title / subject / keywords / company … are anonymized or emptied;
  the alt text and titles of pictures and shapes; text in ``customXml`` parts;
  an embedded Office file (a chart's data workbook) is anonymized the same way,
  inside its own zip; the thumbnail (a picture of the first page) is removed.

Pictures stay as they are unless the caller asks for ``pictures="strip"``:
then every picture becomes a grey "picture removed" box of the same shape
(logos, signatures, faces, scanned tables all go), and the text a
``picture_reader`` (OCR) finds in it becomes the picture's alt text, which is
anonymized with the rest. Embedded objects that aren't Office files are
emptied when stripping, else the caller is told.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from lxml import etree

from captain_claw.anonymize.detectors import hint_for
from captain_claw.anonymize.engine import Segment

OFFICE_KINDS = ("docx", "xlsx", "pptx")
MAX_UNCOMPRESSED = 300 * 1024 * 1024
MAX_MEMBERS = 5000

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC = "http://purl.org/dc/elements/1.1/"
EP = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
VT = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"
CUSTOM = "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
TC = "http://schemas.microsoft.com/office/spreadsheetml/2018/threadedcomments"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
VML = "urn:schemas-microsoft-com:vml"
VML_O = "urn:schemas-microsoft-com:office:office"
THUMBNAIL_REL = "http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

PICTURE_MODES = ("keep", "strip")
# (picture bytes, extension without the dot) → the text in it, "" for none.
PictureReader = Callable[[bytes, str], str]
MAX_DEPTH = 2  # an Office file inside an Office file inside an Office file, no deeper
MAX_PICTURE_TEXT = 1500
_EMBEDDED_KINDS = {".docx": "docx", ".docm": "docx", ".xlsx": "xlsx", ".xlsm": "xlsx",
                   ".pptx": "pptx", ".pptm": "pptx"}
# Picture formats Pillow writes, so a placeholder keeps the part's name.
_RASTER = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "jpe": "JPEG", "gif": "GIF", "bmp": "BMP",
           "tif": "TIFF", "tiff": "TIFF", "webp": "WEBP"}
_MEDIA_RE = re.compile(r"(word|ppt|xl)/media/[^/]+")
_PICTURE_PARTS = {
    "docx": r"word/(document|header\d*|footer\d*|footnotes|endnotes|comments\w*)\.xml|word/glossary/document\.xml",
    "pptx": r"ppt/(slides|slideLayouts|slideMasters|notesSlides|notesMasters|handoutMasters)/[^/]+\.xml",
    "xlsx": r"xl/drawings/[^/]+\.xml",
}

_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False, remove_blank_text=False)


class OfficeError(ValueError):
    """Not an office file this can open (or one too big to open safely)."""


def kind_of(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext if ext in OFFICE_KINDS else ""


@dataclass
class _Unit:
    """One segment: the elements whose text it is made of (in order), or one
    attribute of one element."""
    part: str
    elements: list[Any] = field(default_factory=list)
    attr: tuple[Any, str] | None = None
    hint: str = ""
    number_cell: Any = None   # an xlsx numeric <c>: becomes an inline string if changed
    sheet_name: bool = False
    para: Any = None          # the paragraph element (Word / slides): for grouping

    def text(self) -> str:
        if self.attr is not None:
            el, name = self.attr
            return el.get(name) or ""
        return "".join(el.text or "" for el in self.elements)


@dataclass
class OfficePrepared:
    kind: str
    segments: list[Segment]
    rebuild: Callable[[list[list[tuple[int, int, str]]]], bytes]
    warnings: list[str]


def _local(tag: Any) -> str:
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def _apply_to_pieces(elements: list[Any], edits: list[tuple[int, int, str]]) -> bool:
    """Put *edits* (positions in the joined text) into the elements' texts,
    each token in the piece where its span starts; pieces the span runs into
    lose that part. Returns whether anything changed."""
    if not edits:
        return False
    texts = [el.text or "" for el in elements]
    lengths = [len(t) for t in texts]
    starts = []
    pos = 0
    for n in lengths:
        starts.append(pos)
        pos += n
    for s, e, token in sorted(edits, reverse=True):
        i = next((k for k in range(len(texts)) if starts[k] <= s < starts[k] + lengths[k]), None)
        if i is None:
            continue
        ps, pe = starts[i], starts[i] + lengths[i]
        if e <= pe:
            texts[i] = texts[i][:s - ps] + token + texts[i][e - ps:]
        else:
            texts[i] = texts[i][:s - ps] + token
            j = i + 1
            while j < len(texts) and starts[j] < e:
                # Positions before e are untouched by later edits, so original offsets hold.
                texts[j] = texts[j][e - starts[j]:] if e < starts[j] + lengths[j] else ""
                j += 1
    changed = False
    for el, new in zip(elements, texts):
        if (el.text or "") != new:
            el.text = new
            if _local(el.tag) in ("t", "delText", "instrText") and etree.QName(el.tag).namespace == W:
                el.set(XML_SPACE, "preserve")
            changed = True
    return changed


def _pieces_of(paragraph: Any, piece_tags: set[str], para_tag: str) -> list[Any]:
    """Text elements whose nearest paragraph ancestor is *paragraph* (a text
    box's paragraphs inside it are their own)."""
    out = []
    for el in paragraph.iter(*piece_tags):
        anc = el.getparent()
        while anc is not None and anc.tag != para_tag:
            anc = anc.getparent()
        if anc is paragraph:
            out.append(el)
    return out


def _rph_free(root: Any, tag: str) -> list[Any]:
    """Elements of *tag* under *root* that aren't phonetic hints (``rPh``)."""
    out = []
    for el in root.iter(tag):
        anc = el.getparent()
        skip = False
        while anc is not None and anc is not root:
            if anc.tag == f"{{{S}}}rPh":
                skip = True
                break
            anc = anc.getparent()
        if not skip:
            out.append(el)
    return out


class _Package:
    def __init__(self, data: bytes) -> None:
        try:
            self.zin = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise OfficeError("not a valid office file (it isn't a zip)") from None
        infos = self.zin.infolist()
        if len(infos) > MAX_MEMBERS or sum(i.file_size for i in infos) > MAX_UNCOMPRESSED:
            raise OfficeError("this file unpacks to too much to open safely")
        self.names = [i.filename for i in infos]
        self.trees: dict[str, Any] = {}
        self.dirty: set[str] = set()
        self.replaced: dict[str, bytes] = {}   # binary parts written anew (placeholders, nested files)
        self.renamed: dict[str, str] = {}      # part → new name (a placeholder in another format)
        self.removed: set[str] = set()

    def tree(self, name: str) -> Any | None:
        if name not in self.trees:
            if name not in self.names:
                return None
            try:
                self.trees[name] = etree.fromstring(self.zin.read(name), _PARSER)
            except etree.XMLSyntaxError:
                raise OfficeError(f"a part of this file is damaged ({name})") from None
        return self.trees[name]

    def matching(self, pattern: str) -> list[str]:
        rx = re.compile(pattern)
        return sorted(n for n in self.names if rx.fullmatch(n))

    def write(self) -> bytes:
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as zout:
            for info in self.zin.infolist():
                name = info.filename
                if name in self.removed:
                    continue
                if name in self.replaced:
                    data = self.replaced[name]
                elif name in self.dirty:
                    data = etree.tostring(self.trees[name], xml_declaration=True,
                                          encoding="UTF-8", standalone=True)
                else:
                    data = self.zin.read(name)
                if name in self.renamed:
                    moved = zipfile.ZipInfo(self.renamed[name], date_time=info.date_time)
                    moved.compress_type = info.compress_type
                    moved.external_attr = info.external_attr
                    zout.writestr(moved, data)
                else:
                    zout.writestr(info, data)
        return out.getvalue()


# ── Relationships (which part points at which) ──────────────────────


def _rels_of(part: str) -> str:
    """``word/document.xml`` → ``word/_rels/document.xml.rels``."""
    folder, base = posixpath.split(part)
    return posixpath.join(folder, "_rels", base + ".rels")


def _rels_folder(rels: str) -> str:
    """The folder a ``.rels`` file's relative targets start from."""
    return posixpath.dirname(posixpath.dirname(rels))


def _resolve(folder: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(folder, target)) if folder else posixpath.normpath(target)


def _internal_targets(pkg: _Package, part: str) -> dict[str, str]:
    """The part's relationship ids → the parts they point at (inside the zip)."""
    root = pkg.tree(_rels_of(part))
    if root is None:
        return {}
    folder = posixpath.dirname(part)
    out = {}
    for rel in root.iter(f"{{{REL}}}Relationship"):
        if rel.get("TargetMode") != "External" and rel.get("Id") and rel.get("Target"):
            out[rel.get("Id")] = _resolve(folder, rel.get("Target"))
    return out


def _retarget(pkg: _Package, old: str, new: str) -> None:
    """Every relationship pointing at part *old* points at *new* (same folder)."""
    for rels in pkg.matching(r"(.*/)?_rels/[^/]*\.rels"):
        root = pkg.tree(rels)
        folder = _rels_folder(rels)
        for rel in root.iter(f"{{{REL}}}Relationship"):
            target = rel.get("Target") or ""
            if rel.get("TargetMode") != "External" and target and _resolve(folder, target) == old:
                rel.set("Target", target[: len(target) - len(posixpath.basename(target))] + posixpath.basename(new))
                pkg.dirty.add(rels)


def _drop_override(pkg: _Package, part: str) -> None:
    types = pkg.tree("[Content_Types].xml")
    if types is None:
        return
    for el in list(types.iter(f"{{{CT}}}Override")):
        if (el.get("PartName") or "").lstrip("/") == part:
            types.remove(el)
            pkg.dirty.add("[Content_Types].xml")


def _ensure_default(pkg: _Package, ext: str, content_type: str) -> None:
    types = pkg.tree("[Content_Types].xml")
    if types is None:
        return
    if any((el.get("Extension") or "").lower() == ext for el in types.iter(f"{{{CT}}}Default")):
        return
    el = etree.SubElement(types, f"{{{CT}}}Default")
    el.set("Extension", ext)
    el.set("ContentType", content_type)
    pkg.dirty.add("[Content_Types].xml")


def _remove_thumbnails(pkg: _Package) -> None:
    """The thumbnail is a picture of the first page: it goes, with its links."""
    thumbs = pkg.matching(r"docProps/thumbnail\.\w+")
    if not thumbs:
        return
    root = pkg.tree("_rels/.rels")
    if root is not None:
        for rel in list(root.iter(f"{{{REL}}}Relationship")):
            if rel.get("Type") == THUMBNAIL_REL or _resolve("", rel.get("Target") or "") in thumbs:
                root.remove(rel)
                pkg.dirty.add("_rels/.rels")
    for name in thumbs:
        pkg.removed.add(name)
        _drop_override(pkg, name)


# ── Pictures ────────────────────────────────────────────────────────


def _holder_of(el: Any, prefer_docpr: bool) -> Any | None:
    """The element carrying a picture's alt text: Word's ``wp:docPr`` (what Word
    shows), else the nearest ``cNvPr`` (slides, sheets, shapes filled with it)."""
    fallback = None
    anc = el.getparent()
    while anc is not None:
        for ch in anc:
            if not isinstance(ch.tag, str):
                continue
            local = _local(ch.tag)
            if local == "docPr":
                return ch
            if fallback is None and local.startswith("nv") and local.endswith("Pr"):
                c_nv = next((g for g in ch if isinstance(g.tag, str) and _local(g.tag) == "cNvPr"), None)
                if c_nv is not None:
                    fallback = c_nv
                    if not prefer_docpr:
                        return c_nv
        anc = anc.getparent()
    return fallback


def _picture_holders(pkg: _Package, kind: str) -> dict[str, list[tuple[str, Any, str]]]:
    """Each picture part → where it is drawn: (xml part, element, alt attribute)."""
    out: dict[str, list[tuple[str, Any, str]]] = {}
    for part in pkg.matching(_PICTURE_PARTS[kind]):
        targets = _internal_targets(pkg, part)
        if not targets:
            continue
        for el in pkg.tree(part).iter():
            if not isinstance(el.tag, str):
                continue
            local, ns = _local(el.tag), etree.QName(el.tag).namespace
            if local == "blip":
                rid = el.get(f"{{{R}}}embed")
            elif local == "imagedata" and ns == VML:
                rid = el.get(f"{{{R}}}id")
            else:
                continue
            if not rid or rid not in targets:
                continue
            if local == "imagedata":
                shape = el.getparent()
                if shape is not None:
                    out.setdefault(targets[rid], []).append((part, shape, "alt"))
                continue
            holder = _holder_of(el, prefer_docpr=kind == "docx")
            if holder is not None:
                out.setdefault(targets[rid], []).append((part, holder, "descr"))
    return out


def placeholder(blob: bytes, ext: str) -> tuple[bytes, str]:
    """A grey "picture removed" box with the picture's proportions, in the
    picture's own format where Pillow writes it (else PNG): (bytes, extension)."""
    from PIL import Image, ImageDraw, ImageFont

    ext = ext.lower()
    width, height = 400, 300
    try:
        with Image.open(io.BytesIO(blob)) as im:  # reads the header only
            width, height = im.size
    except Exception:  # noqa: BLE001 — unknown or damaged picture: a 4:3 box
        pass
    if ext == "svg":
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
               f'preserveAspectRatio="none"><rect width="100%" height="100%" fill="#d9d9d9"/>'
               f'<text x="50%" y="50%" fill="#5a5a5a" font-family="sans-serif" font-size="{max(10, height // 10)}" '
               f'text-anchor="middle" dominant-baseline="middle">picture removed</text></svg>')
        return svg.encode("utf-8"), "svg"
    scale = min(1.0, 600 / max(width, height, 1))
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    img = Image.new("RGB", (w, h), (217, 217, 217))
    draw = ImageDraw.Draw(img)
    label = "picture removed"
    size = max(8, min(h // 6, w // 9))
    font = ImageFont.load_default(size=size)
    box = draw.textbbox((0, 0), label, font=font)
    if box[2] - box[0] <= w and box[3] - box[1] <= h:
        draw.text(((w - (box[2] - box[0])) / 2, (h - (box[3] - box[1])) / 2 - box[1]), label,
                  fill=(90, 90, 90), font=font)
    fmt = _RASTER.get(ext)
    if fmt is None:
        fmt, ext = "PNG", "png"
    if fmt == "GIF":
        img = img.convert("P")
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue(), ext


@dataclass
class _Stats:
    pictures: int = 0          # replaced with a placeholder
    pictures_read: int = 0     # of those, text found and kept as alt text
    pictures_left: int = 0     # kept as they are
    embedded_done: int = 0     # Office files inside, anonymized
    embedded_emptied: int = 0
    embedded_left: int = 0


def _picture_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:MAX_PICTURE_TEXT]


def _strip_pictures(pkg: _Package, kind: str, reader: PictureReader | None, stats: _Stats) -> None:
    holders = _picture_holders(pkg, kind)
    for name in [n for n in pkg.names if _MEDIA_RE.fullmatch(n)]:
        blob = pkg.zin.read(name)
        ext = posixpath.splitext(name)[1].lstrip(".").lower()
        text = ""
        if reader is not None:
            try:
                text = _picture_text(reader(blob, ext))
            except Exception:  # noqa: BLE001 — a picture OCR can't read just has no text
                text = ""
        if text:
            stats.pictures_read += 1
            for part, el, attr in holders.get(name, []):
                before = (el.get(attr) or "").strip()
                el.set(attr, f"{before} — Text in the picture: {text}" if before else f"Text in the picture: {text}")
                pkg.dirty.add(part)
        new_blob, new_ext = placeholder(blob, ext)
        pkg.replaced[name] = new_blob
        if new_ext != ext:
            stem = posixpath.splitext(name)[0]
            new_name = f"{stem}.{new_ext}"
            if new_name in pkg.names:
                new_name = f"{stem}-removed.{new_ext}"
            pkg.renamed[name] = new_name
            _retarget(pkg, name, new_name)
            _drop_override(pkg, name)
            _ensure_default(pkg, new_ext, f"image/{new_ext}")
        stats.pictures += 1


def _alt_units(pkg: _Package, kind: str, units: list[_Unit]) -> None:
    """Alt text and titles of pictures and shapes."""
    for part in pkg.matching(_PICTURE_PARTS[kind]):
        for el in pkg.tree(part).iter():
            if not isinstance(el.tag, str):
                continue
            local, ns = _local(el.tag), etree.QName(el.tag).namespace
            if local in ("docPr", "cNvPr"):
                attrs = ("descr", "title")
            elif ns == VML and local == "shape":
                attrs = ("alt",)
            elif ns == VML and local == "imagedata":
                attrs = (f"{{{VML_O}}}title",)
            else:
                continue
            for attr in attrs:
                if (el.get(attr) or "").strip():
                    units.append(_Unit(part, attr=(el, attr)))


def _custom_xml_units(pkg: _Package, units: list[_Unit]) -> None:
    """``customXml`` parts (SharePoint fields, data bound to content controls):
    every piece of text in them."""
    for name in pkg.matching(r"customXml/item\d+\.xml"):
        try:
            root = pkg.tree(name)
        except OfficeError:
            continue
        for el in root.iter():
            if isinstance(el.tag, str) and (el.text or "").strip():
                units.append(_Unit(name, [el]))


def _paragraph_units(pkg: _Package, name: str, para: str, pieces: set[str], units: list[_Unit]) -> None:
    root = pkg.tree(name)
    if root is None:
        return
    for p in root.iter(para):
        els = _pieces_of(p, pieces, para)
        if els and any((el.text or "").strip() for el in els):
            units.append(_Unit(name, els, para=p))


def _single_units(pkg: _Package, name: str, tag: str, units: list[_Unit], hint: str = "") -> None:
    root = pkg.tree(name)
    if root is None:
        return
    for el in root.iter(tag):
        if (el.text or "").strip():
            units.append(_Unit(name, [el], hint=hint))


def _blank_attrs(pkg: _Package, name: str, values: dict[str, str]) -> None:
    """Every attribute with one of these local names gets the given value."""
    root = pkg.tree(name)
    if root is None:
        return
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        for attr in list(el.attrib):
            local = etree.QName(attr).localname
            if local in values and el.get(attr) != values[local]:
                el.set(attr, values[local])
                pkg.dirty.add(name)


def _link_units(pkg: _Package, rels_pattern: str, units: list[_Unit]) -> None:
    for name in pkg.matching(rels_pattern):
        root = pkg.tree(name)
        for rel in root.iter(f"{{{REL}}}Relationship"):
            if rel.get("TargetMode") == "External" and rel.get("Target"):
                units.append(_Unit(name, attr=(rel, "Target")))


def _properties(pkg: _Package, units: list[_Unit], scrub: bool) -> None:
    core = pkg.tree("docProps/core.xml")
    if core is not None:
        for tag in (f"{{{DC}}}creator", f"{{{CP}}}lastModifiedBy", f"{{{DC}}}description"):
            for el in core.iter(tag):
                if scrub and (el.text or ""):
                    el.text = ""
                    pkg.dirty.add("docProps/core.xml")
        for tag in (f"{{{DC}}}title", f"{{{DC}}}subject", f"{{{CP}}}keywords", f"{{{CP}}}category"):
            _single_units(pkg, "docProps/core.xml", tag, units)
    app = pkg.tree("docProps/app.xml")
    if app is not None:
        for tag in (f"{{{EP}}}Company", f"{{{EP}}}Manager", f"{{{EP}}}HyperlinkBase"):
            for el in app.iter(tag):
                if scrub and (el.text or ""):
                    el.text = ""
                    pkg.dirty.add("docProps/app.xml")
        _single_units(pkg, "docProps/app.xml", f"{{{VT}}}lpstr", units)
    if pkg.tree("docProps/custom.xml") is not None:
        for tag in (f"{{{VT}}}lpwstr", f"{{{VT}}}lpstr", f"{{{VT}}}bstr"):
            _single_units(pkg, "docProps/custom.xml", tag, units)


def _warnings(stats: _Stats) -> list[str]:
    out = []
    if stats.pictures:
        read = (f"; the text read from {stats.pictures_read} of them is kept as their alt text, anonymized"
                if stats.pictures_read else "")
        out.append(f"{stats.pictures} picture(s) replaced with a grey “picture removed” box{read}.")
    if stats.pictures_left:
        out.append(f"{stats.pictures_left} picture(s) inside were left as they are — check them for faces, "
                   "logos, signatures or text.")
    if stats.embedded_emptied:
        out.append(f"{stats.embedded_emptied} embedded object(s) that aren't Office files were emptied.")
    if stats.embedded_left:
        out.append(f"{stats.embedded_left} embedded object(s) that aren't Office files were left as they are.")
    return out


def _embedded(pkg: _Package, *, scrub: bool, pictures: str, reader: PictureReader | None, depth: int,
              stats: _Stats) -> list[tuple[str, OfficePrepared]]:
    """Office files embedded in this one (a chart's data workbook, an inserted
    document) are prepared the same way, inside their own zip; other embedded
    objects are emptied when stripping, else counted."""
    nested: list[tuple[str, OfficePrepared]] = []
    for name in pkg.matching(r"(word|ppt|xl)/embeddings/[^/]+"):
        kind = _EMBEDDED_KINDS.get(posixpath.splitext(name)[1].lower(), "")
        if kind and depth < MAX_DEPTH:
            try:
                prep = prepare(pkg.zin.read(name), kind, scrub=scrub, pictures=pictures,
                               picture_reader=reader, _depth=depth + 1)
            except OfficeError:
                prep = None
            if prep is not None:
                nested.append((name, prep))
                stats.embedded_done += 1
                continue
        if scrub and pictures == "strip":
            pkg.replaced[name] = b""
            stats.embedded_emptied += 1
        elif scrub:
            stats.embedded_left += 1
    return nested


# ── Word ────────────────────────────────────────────────────────────


def _docx(pkg: _Package, scrub: bool) -> list[_Unit]:
    units: list[_Unit] = []
    parts = pkg.matching(r"word/(document|header\d*|footer\d*|footnotes|endnotes|comments\w*)\.xml"
                         r"|word/glossary/document\.xml|word/charts/chart\d*\.xml")
    for name in parts:
        if name.startswith("word/charts/"):
            _paragraph_units(pkg, name, f"{{{A}}}p", {f"{{{A}}}t"}, units)
            _single_units(pkg, name, f"{{{C}}}v", units)
            continue
        _paragraph_units(pkg, name, f"{{{W}}}p", {f"{{{W}}}t", f"{{{W}}}delText"}, units)
        _single_units(pkg, name, f"{{{W}}}instrText", units)
        root = pkg.tree(name)
        for el in root.iter(f"{{{W}}}fldSimple"):
            if el.get(f"{{{W}}}instr"):
                units.append(_Unit(name, attr=(el, f"{{{W}}}instr")))
        # text boxes drawn with DrawingML inside Word
        _paragraph_units(pkg, name, f"{{{A}}}p", {f"{{{A}}}t"}, units)
    if scrub:
        for name in parts + pkg.matching(r"word/people\.xml|word/comments\w*\.xml"):
            _blank_attrs(pkg, name, {"author": "Author", "initials": "A", "userId": "", "providerId": ""})
    _link_units(pkg, r"word/_rels/[^/]+\.rels", units)
    return units


# ── PowerPoint ──────────────────────────────────────────────────────


def _pptx(pkg: _Package, scrub: bool) -> list[_Unit]:
    units: list[_Unit] = []
    parts = pkg.matching(r"ppt/(slides|notesSlides|slideLayouts|slideMasters|notesMasters|handoutMasters"
                         r"|comments|charts|diagrams)/[^/]+\.xml")
    for name in parts:
        _paragraph_units(pkg, name, f"{{{A}}}p", {f"{{{A}}}t"}, units)
        if name.startswith("ppt/comments/"):
            _single_units(pkg, name, f"{{{P}}}text", units)
        if name.startswith("ppt/charts/"):
            _single_units(pkg, name, f"{{{C}}}v", units)
    if scrub:
        for name in pkg.matching(r"ppt/(commentAuthors|authors)\.xml") + pkg.matching(r"ppt/comments/[^/]+\.xml"):
            _blank_attrs(pkg, name, {"name": "Author", "initials": "A", "userId": "", "providerId": ""})
    _link_units(pkg, r"ppt/slides/_rels/[^/]+\.rels", units)
    return units


# ── Excel ───────────────────────────────────────────────────────────

_CELL_REF = re.compile(r"^([A-Z]+)(\d+)$")
HEADER_ROWS = 30  # a sheet's header row is looked for in its first rows
# A Croatian phone number kept as a number: the "+" (or 00) is gone, 385 stays.
_HR_PHONE_NUMBER = re.compile(r"385[1-9]\d{7,8}")


def _xlsx(pkg: _Package, scrub: bool) -> tuple[list[_Unit], Callable[[], None]]:
    units: list[_Unit] = []
    shared: list[tuple[Any, list[Any]]] = []
    sst = pkg.tree("xl/sharedStrings.xml")
    if sst is not None:
        for si in sst.iter(f"{{{S}}}si"):
            shared.append((si, _rph_free(si, f"{{{S}}}t")))
    shared_hint: dict[int, str] = {}

    def shared_text(i: int) -> str:
        return "".join(el.text or "" for el in shared[i][1]) if 0 <= i < len(shared) else ""

    sheets = pkg.matching(r"xl/worksheets/sheet\d+\.xml")
    for name in sheets:
        root = pkg.tree(name)
        # Column headers say what a column holds ("OIB", "Telefon") — in the
        # header row, which is often below a title or two, not only row 1. A
        # header applies to the rows under it, until the next header row.
        headers: dict[str, list[tuple[int, str]]] = {}
        for row in root.iter(f"{{{S}}}row"):
            try:
                r = int(row.get("r") or 0)
            except ValueError:
                continue
            if not 1 <= r <= HEADER_ROWS:
                continue
            found: dict[str, str] = {}
            numbers = False
            for c in row.iter(f"{{{S}}}c"):
                m = _CELL_REF.match(c.get("r") or "")
                if not m:
                    continue
                t = c.get("t")
                v = c.find(f"{{{S}}}v")
                text = ""
                if t == "s":
                    text = shared_text(int(v.text)) if v is not None and (v.text or "").isdigit() else ""
                elif t == "inlineStr":
                    text = "".join(el.text or "" for el in _rph_free(c, f"{{{S}}}t"))
                elif t in (None, "n") and v is not None:
                    numbers = True
                hint = hint_for(text)
                if hint:
                    found[m.group(1)] = hint
            if found and (r == 1 or not numbers):  # a row of data has numbers in it
                for col, hint in found.items():
                    headers.setdefault(col, []).append((r, hint))

        def hint_at(col: str, rownum: int) -> str:
            return next((h for r, h in reversed(headers.get(col, [])) if r < rownum), "")

        for c in root.iter(f"{{{S}}}c"):
            m = _CELL_REF.match(c.get("r") or "")
            col, rownum = (m.group(1), int(m.group(2))) if m else ("", 0)
            hint = hint_at(col, rownum) if col else ""
            t = c.get("t")
            v = c.find(f"{{{S}}}v")
            if t == "s":
                if v is not None and (v.text or "").isdigit() and hint:
                    shared_hint.setdefault(int(v.text), hint)
            elif t == "inlineStr":
                els = _rph_free(c, f"{{{S}}}t")
                if els:
                    units.append(_Unit(name, els, hint=hint))
            elif t == "str":
                if v is not None and (v.text or "").strip():
                    units.append(_Unit(name, [v], hint=hint))
            elif t in (None, "n") and v is not None:
                # A whole number of 6+ digits may be an identifier kept as a
                # number (an OIB, a JMBG, a phone number that lost its "+").
                # Amounts and counts stay numbers: only what the detectors (or
                # the column's header) take for an identifier is replaced.
                whole = re.fullmatch(r"(-?\d{6,})(?:\.0+)?", (v.text or "").strip())
                if whole:
                    if v.text != whole.group(1):
                        v.text = whole.group(1)  # "12345678903.0" is 12345678903
                    if not hint and _HR_PHONE_NUMBER.fullmatch(whole.group(1)):
                        hint = "PHONE"
                    units.append(_Unit(name, [v], hint=hint, number_cell=c))
            f = c.find(f"{{{S}}}f")
            if f is not None and '"' in (f.text or ""):
                units.append(_Unit(name, [f]))
    for i, (_si, els) in enumerate(shared):
        if els:
            units.append(_Unit("xl/sharedStrings.xml", els, hint=shared_hint.get(i, "")))

    for name in pkg.matching(r"xl/comments\d*\.xml"):
        root = pkg.tree(name)
        for cm in root.iter(f"{{{S}}}comment"):
            els = _rph_free(cm, f"{{{S}}}t")
            if els:
                units.append(_Unit(name, els))
        if scrub:
            for author in root.iter(f"{{{S}}}author"):
                if author.text != "Author":
                    author.text = "Author"
                    pkg.dirty.add(name)
    for name in pkg.matching(r"xl/threadedComments/[^/]+\.xml"):
        _single_units(pkg, name, f"{{{TC}}}text", units)
    if scrub:
        for name in pkg.matching(r"xl/persons/[^/]+\.xml"):
            _blank_attrs(pkg, name, {"displayName": "Person", "userId": "", "providerId": ""})
    for name in pkg.matching(r"xl/(drawings|charts)/[^/]+\.xml"):
        _paragraph_units(pkg, name, f"{{{A}}}p", {f"{{{A}}}t"}, units)
        if name.startswith("xl/charts/"):
            _single_units(pkg, name, f"{{{C}}}v", units)
    _link_units(pkg, r"xl/worksheets/_rels/[^/]+\.rels", units)

    book = pkg.tree("xl/workbook.xml")
    old_names: dict[Any, str] = {}
    if book is not None:
        for sheet in book.iter(f"{{{S}}}sheet"):
            if sheet.get("name"):
                old_names[sheet] = sheet.get("name")
                units.append(_Unit("xl/workbook.xml", attr=(sheet, "name"), sheet_name=True))

    def follow_renames() -> None:
        """Formulas and defined names pointing at a renamed sheet follow it."""
        renames = {old: el.get("name") for el, old in old_names.items() if el.get("name") != old}
        if not renames:
            return
        targets = [(n, f) for n in sheets for f in pkg.tree(n).iter(f"{{{S}}}f")]
        if book is not None:
            targets += [("xl/workbook.xml", d) for d in book.iter(f"{{{S}}}definedName")]
        for part, el in targets:
            text = el.text or ""
            new = text
            for old, now in renames.items():
                quoted = "'" + now.replace("'", "''") + "'!"
                new = new.replace("'" + old.replace("'", "''") + "'!", quoted)
                new = re.sub(rf"(?<![\w'.]){re.escape(old)}!", lambda _m: quoted, new)
            if new != text:
                el.text = new
                pkg.dirty.add(part)

    return units, follow_renames


_SHEET_FORBIDDEN = re.compile(r"[\[\]:*?/\\]")


def _set_unit(unit: _Unit, new: str, pkg: _Package) -> None:
    if unit.attr is not None:
        el, name = unit.attr
        if unit.sheet_name:
            new = _SHEET_FORBIDDEN.sub("_", new)[:31] or "Sheet"
        if el.get(name) != new:
            el.set(name, new)
            pkg.dirty.add(unit.part)
        return
    if unit.number_cell is not None:
        c = unit.number_cell
        if new != (unit.elements[0].text or ""):
            for child in list(c):
                if child.tag in (f"{{{S}}}v", f"{{{S}}}is"):
                    c.remove(child)
            c.set("t", "inlineStr")
            is_el = etree.SubElement(c, f"{{{S}}}is")
            t_el = etree.SubElement(is_el, f"{{{S}}}t")
            t_el.text = new
            pkg.dirty.add(unit.part)


@dataclass
class _Group:
    """Units read as one text (a document part's paragraphs), so clues on
    one line reach the next ("Lijep pozdrav," above a name; "Ime:" in the
    cell before it). Edits are split back to each unit."""
    units: list[_Unit]
    seps: list[str]

    def text(self) -> str:
        return "".join(sep + u.text() for sep, u in zip(self.seps, self.units))

    def spans(self) -> list[tuple[int, int]]:
        out, pos = [], 0
        for sep, u in zip(self.seps, self.units):
            pos += len(sep)
            n = len(u.text())
            out.append((pos, pos + n))
            pos += n
        return out

    def split(self, edits: list[tuple[int, int, str]]) -> list[list[tuple[int, int, str]]]:
        per: list[list[tuple[int, int, str]]] = [[] for _ in self.units]
        spans = self.spans()
        for s, e, token in edits:
            first = True
            for k, (us, ue) in enumerate(spans):
                if ue <= s or us >= e or us == ue:
                    continue
                a, b = max(s, us) - us, min(e, ue) - us
                per[k].append((a, b, token if first else ""))
                first = False
        return per


def _row_cell(p: Any) -> tuple[Any, Any]:
    """The table row and cell a paragraph stands in (None, None outside tables)."""
    cell = row = None
    anc = p.getparent() if p is not None else None
    while anc is not None:
        local = _local(anc.tag)
        if local == "tc" and cell is None:
            cell = anc
        elif local == "tr" and cell is not None:
            row = anc
            break
        anc = anc.getparent()
    return row, cell


def _groups(units: list[_Unit], kind: str) -> list[_Group]:
    groups: list[_Group] = []
    prev: _Unit | None = None
    prev_rc: tuple[Any, Any] = (None, None)
    for u in units:
        alone = kind == "xlsx" or u.attr is not None or u.number_cell is not None or bool(u.hint) or u.sheet_name
        rc = _row_cell(u.para)
        if alone or prev is None or prev.part != u.part or not groups or len(groups[-1].units) == 0 \
                or groups[-1].units[-1] is not prev:
            groups.append(_Group([u], [""]))
        else:
            same_row = rc[0] is not None and rc[0] is prev_rc[0] and rc[1] is not prev_rc[1]
            groups[-1].units.append(u)
            groups[-1].seps.append("\t" if same_row else "\n")
        prev = None if alone else u
        prev_rc = rc
    return groups


def prepare(data: bytes, kind: str, *, scrub: bool = True, pictures: str = "keep",
            picture_reader: PictureReader | None = None, _depth: int = 0) -> OfficePrepared:
    """The segments of an office file, and how to write it back with each
    segment's edits applied. *scrub* (anonymizing, not restoring) also blanks
    authors and personal document properties and removes the thumbnail;
    with ``pictures="strip"`` it replaces every picture (see the module doc)."""
    kind = kind.lower()
    if kind not in OFFICE_KINDS:
        raise OfficeError(f"not an office kind: {kind}")
    if pictures not in PICTURE_MODES:
        raise OfficeError(f"pictures must be one of {', '.join(PICTURE_MODES)}")
    pkg = _Package(data)
    if not any(n.startswith({"docx": "word/", "pptx": "ppt/", "xlsx": "xl/"}[kind]) for n in pkg.names):
        raise OfficeError(f"this isn't a .{kind} file inside")
    stats = _Stats()
    if scrub:
        _remove_thumbnails(pkg)
        if pictures == "strip":
            # Before the units: the text read from a picture becomes alt text,
            # which is then anonymized like every other alt text.
            _strip_pictures(pkg, kind, picture_reader, stats)
        else:
            stats.pictures_left = sum(1 for n in pkg.names if _MEDIA_RE.fullmatch(n))
    after: Callable[[], None] = lambda: None  # noqa: E731
    if kind == "docx":
        units = _docx(pkg, scrub)
    elif kind == "pptx":
        units = _pptx(pkg, scrub)
    else:
        units, after = _xlsx(pkg, scrub)
    _alt_units(pkg, kind, units)
    _custom_xml_units(pkg, units)
    _properties(pkg, units, scrub)
    nested = _embedded(pkg, scrub=scrub, pictures=pictures, reader=picture_reader, depth=_depth, stats=stats)
    groups = _groups(units, kind)
    segments = [Segment(g.text(), g.units[0].hint if len(g.units) == 1 else "") for g in groups]
    warnings = _warnings(stats)
    for _name, prep in nested:
        segments.extend(prep.segments)
        warnings.extend(f"Inside an embedded file: {w}" for w in prep.warnings)

    def rebuild(edits: list[list[tuple[int, int, str]]]) -> bytes:
        from captain_claw.anonymize.engine import apply_edits

        for group, group_edits in zip(groups, edits):
            if not group_edits:
                continue
            for unit, unit_edits in zip(group.units, group.split(group_edits)):
                if not unit_edits:
                    continue
                if unit.attr is not None or unit.number_cell is not None:
                    _set_unit(unit, apply_edits(unit.text(), unit_edits), pkg)
                elif _apply_to_pieces(unit.elements, unit_edits):
                    pkg.dirty.add(unit.part)
        after()
        pos = len(groups)
        for name, prep in nested:
            n = len(prep.segments)
            sub = list(edits[pos:pos + n])
            sub += [[] for _ in range(n - len(sub))]
            pkg.replaced[name] = prep.rebuild(sub)  # always: scrubbing may have changed it
            pos += n
        return pkg.write()

    return OfficePrepared(kind, segments, rebuild, warnings)
