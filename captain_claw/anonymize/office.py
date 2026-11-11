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
  emptied, title / subject / keywords / company … are anonymized or emptied.

Not changed (the caller is told): pictures and embedded files.
"""

from __future__ import annotations

import io
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
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

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
                if info.filename in self.dirty:
                    data = etree.tostring(self.trees[info.filename], xml_declaration=True,
                                          encoding="UTF-8", standalone=True)
                else:
                    data = self.zin.read(info.filename)
                zout.writestr(info, data)
        return out.getvalue()


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


def _warnings(pkg: _Package) -> list[str]:
    out = []
    pictures = [n for n in pkg.names if re.search(r"/media/", n)]
    embedded = [n for n in pkg.names if re.search(r"/embeddings/", n)]
    if pictures:
        out.append(f"{len(pictures)} picture(s) inside were left as they are — check them for faces, "
                   "signatures or text.")
    if embedded:
        out.append(f"{len(embedded)} embedded file(s) (e.g. a chart's data) were left as they are.")
    return out


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
        hints: dict[str, str] = {}
        for row in root.iter(f"{{{S}}}row"):
            if row.get("r") != "1":
                continue
            for c in row.iter(f"{{{S}}}c"):
                m = _CELL_REF.match(c.get("r") or "")
                if not m:
                    continue
                t = c.get("t")
                text = ""
                if t == "s":
                    v = c.find(f"{{{S}}}v")
                    text = shared_text(int(v.text)) if v is not None and (v.text or "").isdigit() else ""
                elif t == "inlineStr":
                    text = "".join(el.text or "" for el in _rph_free(c, f"{{{S}}}t"))
                hint = hint_for(text)
                if hint:
                    hints[m.group(1)] = hint
        for c in root.iter(f"{{{S}}}c"):
            m = _CELL_REF.match(c.get("r") or "")
            col, rownum = (m.group(1), int(m.group(2))) if m else ("", 0)
            hint = hints.get(col, "") if rownum > 1 else ""
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
            elif t in (None, "n") and v is not None and re.fullmatch(r"-?\d{6,}", (v.text or "").strip()):
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


def prepare(data: bytes, kind: str, *, scrub: bool = True) -> OfficePrepared:
    """The segments of an office file, and how to write it back with each
    segment's edits applied. *scrub* (anonymizing, not restoring) also blanks
    authors and personal document properties."""
    kind = kind.lower()
    if kind not in OFFICE_KINDS:
        raise OfficeError(f"not an office kind: {kind}")
    pkg = _Package(data)
    after: Callable[[], None] = lambda: None  # noqa: E731
    if kind == "docx":
        units = _docx(pkg, scrub)
    elif kind == "pptx":
        units = _pptx(pkg, scrub)
    else:
        units, after = _xlsx(pkg, scrub)
    _properties(pkg, units, scrub)
    if not any(n.startswith({"docx": "word/", "pptx": "ppt/", "xlsx": "xl/"}[kind]) for n in pkg.names):
        raise OfficeError(f"this isn't a .{kind} file inside")
    groups = _groups(units, kind)
    segments = [Segment(g.text(), g.units[0].hint if len(g.units) == 1 else "") for g in groups]

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
        return pkg.write()

    return OfficePrepared(kind, segments, rebuild, _warnings(pkg))
