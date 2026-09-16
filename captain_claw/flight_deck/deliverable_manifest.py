"""Deliverable manifest — the declared shape of a Vatra run's long-form output.

Pure logic (no DB, no LLM, no Flight Deck imports beyond ``write_guard``). A
manifest names ONE assembled deliverable file, its ordered parts, one owner per
part, optional chapter/section ranges, and the seam owner — so the scheduler,
the synthesis step, and the done-gate can reason about the deliverable as a
checkable object instead of trusting a weak model's prose.

Two ways to get one:

* :func:`parse` — from the caller's explicit ``deliverable`` request field.
* :func:`derive` — from the Group-0 plan + subtasks when the caller opted into
  ``quality.derive_manifest`` but gave no explicit manifest.

Increment 4 adds :func:`part_status`, :func:`assemble` and :func:`gate` for
deterministic synthesis and the completion gate.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from captain_claw import write_guard

# ── number words (chapter "Twelve" etc.) ──────────────────────────────
_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}


def word_to_int(word: str) -> int | None:
    """Parse a small English number word ('one'…'forty-two') to an int, or None."""
    w = (word or "").strip().lower().replace("_", "-")
    if not w:
        return None
    if w.isdigit():
        return int(w)
    if w in _ONES:
        return _ONES[w]
    if w in _TENS:
        return _TENS[w]
    if "-" in w:
        a, _, b = w.partition("-")
        if a in _TENS and b in _ONES:
            return _TENS[a] + _ONES[b]
    return None


def slugify(text: str, maxlen: int = 40) -> str:
    """Lowercase, non-alphanumerics → single hyphens, trimmed to ``maxlen``."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:maxlen].strip("-")) or "part"


def parse_range(raw: Any) -> tuple[int, int] | None:
    """Parse a chapter/section range like 'ch6-7', 'chapters six to seven', '6' → (6,7)/(6,6)."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            a, b = int(raw[0]), int(raw[1])
            return (min(a, b), max(a, b))
        except (TypeError, ValueError):
            return None
    s = str(raw).strip().lower()
    if not s:
        return None
    # pull number tokens (digits or number-words), in order
    toks = re.findall(r"[a-z]+-[a-z]+|\d+|[a-z]+", s)
    nums = [n for n in (word_to_int(t) for t in toks) if n is not None]
    if not nums:
        return None
    if len(nums) == 1:
        return (nums[0], nums[0])
    return (min(nums[0], nums[-1]), max(nums[0], nums[-1]))


_SECTION_RE_DEFAULT = r"(?im)^\s{0,3}#{1,3}\s*(?:chapter|part|section)\b"
_CHAPTER_NUM_RE = re.compile(
    r"(?im)^\s{0,3}#{1,3}\s*(?:chapter|part)\s+([0-9]+|[a-z][a-z\-]*)\b"
)


def chapters_in(text: str) -> list[int]:
    """Chapter/part numbers named by markdown headings, in document order."""
    out: list[int] = []
    for m in _CHAPTER_NUM_RE.finditer(text or ""):
        n = word_to_int(m.group(1))
        if n is not None:
            out.append(n)
    return out


def count_sections(text: str, regex: str = "") -> int:
    """Count section/chapter headings in *text* (default: markdown chapter/part/section)."""
    pat = regex.strip() or _SECTION_RE_DEFAULT
    try:
        rx = re.compile(pat) if regex.strip() else re.compile(_SECTION_RE_DEFAULT)
    except re.error:
        rx = re.compile(_SECTION_RE_DEFAULT)
    return len(rx.findall(text or ""))


# ── manifest model ────────────────────────────────────────────────────

@dataclass
class Part:
    path: str                       # vfs:<project>/<name.ext>
    order: int
    owner: str = ""                 # subtask id
    range: tuple[int, int] | None = None
    min_bytes: int = 0
    sequential: bool = False        # start only after the previous part is frozen

    def basename(self) -> str:
        return write_guard.basename_of(self.path)


@dataclass
class Manifest:
    path: str                       # the assembled deliverable, vfs:<project>/<name.ext>
    kind: str = ""                  # "" | "document" | "fiction"
    min_bytes: int = 0
    min_sections: int = 0
    section_regex: str = ""
    parts: list[Part] = field(default_factory=list)
    seam_owner: str = ""            # subtask id of the seam owner (or "")

    def basename(self) -> str:
        return write_guard.basename_of(self.path)

    def declared_basenames(self) -> list[str]:
        names = [self.basename()] + [p.basename() for p in self.parts]
        seen: set[str] = set()
        return [n for n in names if n and not (n in seen or seen.add(n))]

    def part_for(self, owner_subtask_id: str) -> Part | None:
        for p in self.parts:
            if p.owner and p.owner == owner_subtask_id:
                return p
        return None


def _vfs_norm(name: str, vfs_project: str) -> str:
    """Normalise a bare filename or vfs path to ``vfs:<project>/<basename>``."""
    base = write_guard.basename_of(str(name or "").strip())
    return f"vfs:{vfs_project}/{base}" if base else ""


def _default_ext(kind: str) -> str:
    return ".md"


# ── parse (explicit request manifest) ─────────────────────────────────

def parse(raw: dict | None, subtasks: list[dict], vfs_project: str) -> Manifest | None:
    """Build a Manifest from the caller's ``deliverable`` field, or None if absent.

    Normalises paths to ``vfs:<project>/<name>``, requires each to carry an
    extension, drops parts whose declared owner is not a real subtask, drops
    duplicate paths, defaults ``order`` from position, and marks parts
    ``sequential`` for a fiction deliverable unless the part says otherwise.
    """
    if not raw or not isinstance(raw, dict):
        return None
    valid_ids = {str(s.get("id")) for s in (subtasks or [])}
    dpath_raw = str(raw.get("path") or "").strip()
    if not dpath_raw:
        return None
    kind = str(raw.get("kind") or "").strip().lower()
    if kind not in ("", "document", "fiction"):
        kind = ""
    dpath = _vfs_norm(dpath_raw, vfs_project)
    if write_guard.requires_extension(dpath):
        dpath = dpath + _default_ext(kind)

    parts: list[Part] = []
    seen_paths: set[str] = set()
    for i, rp in enumerate(raw.get("parts") or []):
        if not isinstance(rp, dict):
            continue
        pname = str(rp.get("path") or "").strip()
        if not pname:
            continue
        owner = str(rp.get("owner") or "").strip()
        if owner and valid_ids and owner not in valid_ids:
            continue  # drop unknown owner
        ppath = _vfs_norm(pname, vfs_project)
        if write_guard.requires_extension(ppath):
            ppath = ppath + _default_ext(kind)
        if ppath in seen_paths:
            continue  # drop duplicate path
        seen_paths.add(ppath)
        try:
            order = int(rp.get("order", i + 1))
        except (TypeError, ValueError):
            order = i + 1
        seq = bool(rp.get("sequential", kind == "fiction"))
        try:
            mb = max(0, int(rp.get("min_bytes", 0)))
        except (TypeError, ValueError):
            mb = 0
        parts.append(Part(path=ppath, order=order, owner=owner,
                          range=parse_range(rp.get("range")), min_bytes=mb, sequential=seq))
    parts.sort(key=lambda p: p.order)

    _seam_raw = raw.get("seam_owner")
    seam = str(_seam_raw).strip() if _seam_raw is not None else ""
    # seam_owner may be an index into parts (the incident uses `1`)
    if seam and seam not in valid_ids:
        try:
            idx = int(seam)
            if 0 <= idx < len(parts):
                seam = parts[idx].owner
            else:
                seam = ""
        except (TypeError, ValueError):
            seam = ""

    def _int(key: str) -> int:
        try:
            return max(0, int(raw.get(key, 0)))
        except (TypeError, ValueError):
            return 0

    return Manifest(
        path=dpath, kind=kind, min_bytes=_int("min_bytes"),
        min_sections=_int("min_sections"),
        section_regex=str(raw.get("section_regex") or "").strip(),
        parts=parts, seam_owner=seam,
    )


# ── topological helpers ───────────────────────────────────────────────

def topo_layers(subtasks: list[dict]) -> list[list[str]]:
    """Dependency layers over subtasks (Kahn); members of a cycle share one layer."""
    ids = [str(s.get("id")) for s in (subtasks or [])]
    idset = set(ids)
    deps = {
        str(s.get("id")): [d for d in (s.get("depends_on") or []) if d in idset and d != str(s.get("id"))]
        for s in (subtasks or [])
    }
    placed: set[str] = set()
    layers: list[list[str]] = []
    while len(placed) < len(ids):
        ready = [i for i in ids if i not in placed and all(d in placed for d in deps[i])]
        if not ready:
            # cycle: place everything still unplaced in one layer
            ready = [i for i in ids if i not in placed]
        layers.append(ready)
        placed.update(ready)
    return layers


def _topo_order(subtasks: list[dict]) -> list[str]:
    order: list[str] = []
    for layer in topo_layers(subtasks):
        order.extend(layer)
    return order


# ── derive (from the Group-0 plan + subtasks) ─────────────────────────

def derive(group0_by_subtask: dict | None, subtasks: list[dict], vfs_project: str,
           kind_hint: str = "", deliverable_name: str = "") -> Manifest | None:
    """Derive a manifest from the plan when the caller gave none.

    Each subtask becomes one part, named by the planner's ``produces_file`` when
    present, else ``<id>-<slug(title)>.md``. Parts are ordered
    depends_on-topologically then by subtask order. When consecutive parts carry
    ranges (a split manuscript) and the Lead omitted the edge, a ``depends_on``
    edge is added between them (mutating ``subtasks``) so the sequencer can honour
    it. The assembled deliverable is a single concatenation file.
    """
    if not subtasks:
        return None
    kind = (kind_hint or "").strip().lower()
    if kind not in ("", "document", "fiction"):
        kind = ""
    g0 = group0_by_subtask or {}
    by_id = {str(s.get("id")): s for s in subtasks}
    order_ids = _topo_order(subtasks)

    parts: list[Part] = []
    seen_paths: set[str] = set()
    for i, sid in enumerate(order_ids):
        s = by_id.get(sid) or {}
        entry = g0.get(sid) or {}
        pfile = str(entry.get("produces_file") or "").strip()
        if pfile:
            ppath = _vfs_norm(pfile, vfs_project)
        else:
            ppath = _vfs_norm(f"{sid}-{slugify(str(s.get('title') or sid))}.md", vfs_project)
        if write_guard.requires_extension(ppath):
            ppath = ppath + _default_ext(kind)
        if ppath in seen_paths:
            ppath = _vfs_norm(f"{sid}-{slugify(str(s.get('title') or sid))}-{i}.md", vfs_project)
        seen_paths.add(ppath)
        rng = parse_range(entry.get("range") or s.get("range"))
        parts.append(Part(path=ppath, order=i + 1, owner=sid, range=rng,
                          sequential=bool(rng) and kind == "fiction"))

    # For a ranged/sequential split, chain consecutive parts if the Lead omitted
    # the dependency, so producer-before-consumer sequencing has an edge to honour.
    ranged = [p for p in parts if p.range]
    if len(ranged) >= 2:
        ranged.sort(key=lambda p: (p.range[0], p.order))
        for prev, cur in zip(ranged, ranged[1:]):
            cs = by_id.get(cur.owner)
            if cs is not None and prev.owner and prev.owner not in (cs.get("depends_on") or []):
                cs.setdefault("depends_on", [])
                if prev.owner not in cs["depends_on"]:
                    cs["depends_on"].append(prev.owner)

    dname = write_guard.basename_of(deliverable_name) or "deliverable.md"
    dpath = _vfs_norm(dname, vfs_project)
    if write_guard.requires_extension(dpath):
        dpath = dpath + _default_ext(kind)
    seam = ranged[-1].owner if ranged else (parts[-1].owner if parts else "")
    return Manifest(path=dpath, kind=kind, parts=parts, seam_owner=seam)


# ── artifact assignment + inputs ──────────────────────────────────────

def assign_artifacts(subtasks: list[dict], manifest: Manifest | None) -> dict[str, str]:
    """Stamp ``st['artifact']`` = the part path each subtask owns. Returns the map."""
    out: dict[str, str] = {}
    if not manifest:
        return out
    for s in subtasks or []:
        sid = str(s.get("id"))
        part = manifest.part_for(sid)
        if part:
            s["artifact"] = part.path
            out[sid] = part.path
    return out


def inputs_for(manifest: Manifest | None, subtask_id: str, subtasks: list[dict]) -> list[Part]:
    """Parts this subtask's declared dependencies produce (the inputs it must wait on)."""
    if not manifest:
        return []
    by_id = {str(s.get("id")): s for s in (subtasks or [])}
    s = by_id.get(str(subtask_id)) or {}
    deps = [d for d in (s.get("depends_on") or []) if d != str(subtask_id)]
    out: list[Part] = []
    for d in deps:
        p = manifest.part_for(d)
        if p:
            out.append(p)
    return out


# ── Increment 4: part status, deterministic assembly, done gate ───────

def _sha8(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:8]


def part_status(vfs_dir, part: Part, producer_done: bool = True) -> dict:
    """On-disk status of a part: exists / bytes / placeholder / sections / chapters / landed.

    ``landed`` (ready for a consumer) = exists ∧ non-empty ∧ ≥ its byte floor ∧ not a
    placeholder marker ∧ (its producer has finished OR the part is not sequential).
    """
    fp = Path(vfs_dir) / part.basename()
    exists = fp.is_file()
    raw = b""
    if exists:
        try:
            raw = fp.read_bytes()
        except Exception:
            exists = False
    data = raw.decode("utf-8", errors="replace") if raw else ""
    b = len(raw)
    placeholder = bool(data) and write_guard.is_placeholder_content(data)
    landed = (
        exists and b > 0 and (b >= (part.min_bytes or 0))
        and not placeholder and (producer_done or not part.sequential)
    )
    return {
        "exists": exists, "bytes": b, "placeholder": placeholder,
        "sections": count_sections(data), "chapters": chapters_in(data),
        "landed": landed, "sha8": _sha8(raw) if raw else "",
    }


def assemble(vfs_dir, manifest: Manifest | None,
             producer_done: dict[str, bool] | None = None) -> dict:
    """Concatenate the manifest's parts in order and report deterministic seam findings.

    Returns ``{text, parts, seams, findings}``. Findings use the ``quality_findings``
    shape (kind/source/severity/detail): ``part_missing``, ``placeholder_part``,
    ``missing_chapter``, ``duplicate_chapter``.
    """
    if manifest is None:
        return {"text": "", "parts": [], "seams": [], "findings": []}
    done = producer_done or {}
    texts: list[str] = []
    part_infos: list[dict] = []
    findings: list[dict] = []
    seen_chapter: dict[int, str] = {}
    for p in sorted(manifest.parts, key=lambda x: x.order):
        st = part_status(vfs_dir, p, done.get(p.owner, True))
        part_infos.append({"path": p.path, "owner": p.owner, **st})
        if not st["exists"]:
            findings.append({"kind": "part_missing", "source": "assembly",
                             "severity": "critical",
                             "detail": f"{p.basename()} was never written"})
            continue
        if st["placeholder"]:
            findings.append({"kind": "placeholder_part", "source": "assembly",
                             "severity": "critical",
                             "detail": f"{p.basename()} is a placeholder marker, not content"})
            continue
        data = (Path(vfs_dir) / p.basename()).read_text(encoding="utf-8", errors="replace")
        if p.range and st["chapters"]:
            lo, hi = p.range
            present = set(st["chapters"])
            missing = [n for n in range(lo, hi + 1) if n not in present]
            if missing:
                findings.append({"kind": "missing_chapter", "source": "assembly",
                                 "severity": "critical",
                                 "detail": f"{p.basename()} declares chapters {lo}-{hi} "
                                           f"but is missing {missing}"})
        for n in st["chapters"]:
            if n in seen_chapter and seen_chapter[n] != p.path:
                findings.append({"kind": "duplicate_chapter", "source": "assembly",
                                 "severity": "critical",
                                 "detail": f"chapter {n} appears in both "
                                           f"{write_guard.basename_of(seen_chapter[n])} "
                                           f"and {p.basename()}"})
            else:
                seen_chapter[n] = p.path
        texts.append(data.strip())
    return {"text": "\n\n".join(texts), "parts": part_infos, "seams": [], "findings": findings}


def gate(manifest: Manifest | None, text: str,
         min_chars_fallback: int = 0, min_sections_fallback: int = 0) -> dict:
    """Is *text* an acceptable deliverable? Returns ``{ok, reasons, bytes, sections, chapters}``."""
    body = text or ""
    b = len(body.encode("utf-8", errors="replace"))
    sec_regex = manifest.section_regex if manifest else ""
    secs = count_sections(body, sec_regex)
    chaps = chapters_in(body)
    min_b = (manifest.min_bytes if manifest else 0) or min_chars_fallback or 0
    min_s = (manifest.min_sections if manifest else 0) or min_sections_fallback or 0
    reasons: list[str] = []
    if not body.strip():
        reasons.append("deliverable_missing")
    elif write_guard.is_placeholder_content(body):
        reasons.append("deliverable_placeholder")
    else:
        if min_b and b < min_b:
            reasons.append(f"below_min_bytes ({b} < {min_b})")
        if min_s and secs < min_s:
            reasons.append(f"too_few_sections ({secs} < {min_s})")
    return {"ok": not reasons, "reasons": reasons, "bytes": b, "sections": secs, "chapters": chaps}


def to_analysis(manifest: Manifest | None, statuses: dict | None = None) -> dict:
    """A JSON-safe summary of the manifest for the run's ``analysis`` block."""
    if not manifest:
        return {}
    return {
        "path": manifest.path,
        "kind": manifest.kind,
        "min_bytes": manifest.min_bytes,
        "min_sections": manifest.min_sections,
        "seam_owner": manifest.seam_owner,
        "parts": [
            {
                "path": p.path, "order": p.order, "owner": p.owner,
                "range": list(p.range) if p.range else None,
                "sequential": p.sequential,
                **({"status": statuses.get(p.path)} if statuses and p.path in statuses else {}),
            }
            for p in manifest.parts
        ],
    }
