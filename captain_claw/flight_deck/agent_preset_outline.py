"""Section maps of an agent's knowledge (see agent_preset) — token-free.

A long document reaches the agent whole, in parts its semantic memory
indexes; this gives it a way around one too. ``outline`` finds the sections
from the headings (a PDF's pages or a deck's slides grouped instead when they
are all it has; fenced code never counts), keeping as many heading levels
as fit in ``MAX_SECTIONS``. ``add_gists`` gives each a one-line gist — its
most central sentence by the local model2vec embeddings semantic memory uses,
else its first — and ``map_md`` writes ``<name>.map.md``: every section with
the file and lines it is on, so the agent reads that section, not a 200 KB
part. No model is called and nothing leaves the machine.
"""

from __future__ import annotations

import re
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

MAX_SECTIONS = 1200       # sections in one document's map, at most
MAX_MAP_BYTES = 120_000   # ~30k tokens: read in slices, and one file semantic memory indexes (256 KB)
GIST_CHARS = 180
_SENTENCES_PER_SECTION = 40
_PAGES_PER_GROUP = 10

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEAD = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_MARK = re.compile(r"^(Page|Slide) (\d+)$")
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[\"'(\[“‘]?[A-Z0-9ČĆŠŽĐ])")
# Lines that aren't prose: a link definition ("[docs]: https://…"), a badge or image row, a bare URL.
_NOT_PROSE = re.compile(r"^\s*(\[[^\]]*\]:\s*\S|\[?!\[|<?https?://\S+>?\s*$)")
_NOISE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<[^>]+>|`{1,3}|\*\*|__|^#+\s*|^\s*[-*+]\s+|^\s*\d+[.)]\s+", re.M)


def _headings(text: str) -> list[tuple[int, int, str]]:
    """(offset, level, title) of every Markdown heading outside fenced code."""
    out: list[tuple[int, int, str]] = []
    pos, fenced = 0, False
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if _FENCE.match(bare):
            fenced = not fenced
        elif not fenced:
            m = _HEAD.match(bare)
            if m:
                title = re.sub(r"\s+", " ", m[2]).strip()
                if title:
                    out.append((pos, len(m[1]), title[:160]))
        pos += len(line)
    return out


def outline(text: str, max_sections: int = MAX_SECTIONS) -> list[dict[str, Any]]:
    """The document's sections: ``level`` (1 at the top), ``title``, ``start``
    and ``end`` (character offsets; a section runs to the next heading of its
    level or above, its subsections included) and ``own`` (where its own text
    ends — at its first subsection). Empty for a text with no headings."""
    heads = _headings(text)
    marks = [h for h in heads if _MARK.match(h[2])]
    real = [h for h in heads if not _MARK.match(h[2])]
    if marks and len(real) < 3:
        # A PDF's pages or a deck's slides, and little else: groups of them.
        noun = "Pages" if marks[0][2].startswith("Page") else "Slides"
        groups = [marks[i:i + _PAGES_PER_GROUP] for i in range(0, len(marks), _PAGES_PER_GROUP)]
        heads = []
        for g in groups:
            first, last = _MARK.match(g[0][2])[2], _MARK.match(g[-1][2])[2]
            heads.append((g[0][0], 1, f"{noun} {first}–{last}" if first != last else f"{noun[:-1]} {first}"))
    else:
        heads = real
    if not heads:
        return []
    # The heading levels that fit: the top one always, deeper ones while the count allows.
    levels = sorted({lvl for _, lvl, _ in heads})
    keep = [levels[0]]
    for lvl in levels[1:]:
        if sum(1 for h in heads if h[1] in (*keep, lvl)) > max_sections:
            break
        keep.append(lvl)
    heads = [h for h in heads if h[1] in keep][:max_sections]
    top = min(h[1] for h in heads)
    sections: list[dict[str, Any]] = []
    for i, (start, lvl, title) in enumerate(heads):
        end = next((h[0] for h in heads[i + 1:] if h[1] <= lvl), len(text))
        own = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        sections.append({"level": lvl - top + 1, "title": title, "start": start, "end": end, "own": own})
    return sections


def _sentences(body: str) -> list[str]:
    """A section's own prose as sentences, Markdown and tables left out."""
    lines, fenced = [], False
    for line in body.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if fenced or line.lstrip().startswith(("|", "#", ">")) or not line.strip() or _NOT_PROSE.match(line):
            lines.append("\n")
            continue
        lines.append(line)
    prose = _NOISE.sub("", "\n".join(lines))
    prose = re.sub(r"\[([^\]]+)\](?:\([^)]*\)|\[[^\]]*\])", r"\1", prose)  # [text](url), [text][ref]
    out: list[str] = []
    for para in re.split(r"\n\s*\n", prose):
        para = re.sub(r"\s+", " ", para).strip()
        for s in _SENT_SPLIT.split(para):
            # Prose: long enough, and not mostly a link.
            if 30 <= len(s) <= 600 and len(" ".join(re.findall(r"https?://\S+", s))) < len(s) / 2:
                out.append(s)
                if len(out) >= _SENTENCES_PER_SECTION:
                    return out
    return out


def _clip(s: str) -> str:
    if len(s) <= GIST_CHARS:
        return s
    cut = s[:GIST_CHARS].rsplit(" ", 1)[0].rstrip(",;:—-")
    return cut + "…"


_encoder: Any = None


def _encode(texts: list[str]) -> Any:
    """Unit vectors for *texts* by the model2vec model semantic memory uses."""
    global _encoder
    import numpy as np

    if _encoder is None:
        from captain_claw.semantic_memory import _Model2VecEmbeddingProvider
        _encoder = _Model2VecEmbeddingProvider()._ensure_encoder()
    vecs = np.asarray(_encoder.encode(texts), dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


def add_gists(text: str, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each section with a ``gist``: the sentence of its own text closest to
    what the whole of it says (model2vec), else its first; "" when it has no
    prose of its own."""
    found = [_sentences(text[s["start"]:s["own"]].partition("\n")[2]) for s in sections]  # past the heading line
    flat = [x for xs in found for x in xs]
    vecs = None
    if flat:
        try:
            vecs = _encode(flat)
        except Exception:
            log.warning("agent preset: no local embeddings for the gists — first sentences instead", exc_info=True)
    at = 0
    for s, sents in zip(sections, found):
        gist = ""
        if sents and vecs is not None and len(sents) > 2:
            block = vecs[at:at + len(sents)]
            centre = block.mean(axis=0)
            scores = block @ centre
            scores[0] += 0.02  # a section's opening line, when it is about as central
            gist = sents[int(scores.argmax())]
        elif sents:
            gist = sents[0]
        at += len(sents)
        s["gist"] = _clip(gist)
    return sections


def _line_of(prefix: str, body: str, offset: int) -> int:
    """The 1-based line of *offset* in *body* once written after *prefix*."""
    return prefix.count("\n") + body[:max(0, offset)].count("\n") + 1


def map_md(title: str, doc: str, files: list[dict[str, Any]], sections: list[dict[str, Any]]) -> str:
    """The section map: *files* are what the document was written as — each
    ``name``, ``label``, ``start`` (where its ``raw`` text begins in the
    document), ``body`` (that text as written: stripped) and ``prefix`` (what
    the file holds before it) — and every section is listed under the file it
    starts in, with its lines there. Too big for one indexed file, the
    deepest level's gists go, then the deepest level."""
    rows: list[tuple[int, int, str, str]] = []  # (file index, level, entry, gist)
    for s in sections:
        k = max((i for i, f in enumerate(files) if f["start"] <= s["start"]), default=0)
        f = files[k]
        lead = len(f["raw"]) - len(f["raw"].lstrip())
        first = _line_of(f["prefix"], f["body"], s["start"] - f["start"] - lead)
        stop = min(s["end"], f["start"] + len(f["raw"]))
        last = _line_of(f["prefix"], f["body"], stop - f["start"] - lead - 1)
        more = f" (continues in {files[k + 1]['name']})" if s["end"] > f["start"] + len(f["raw"]) and k + 1 < len(files) else ""
        rows.append((k, s["level"], f"**{s['title']}** — lines {first}–{max(first, last)}{more}",
                     s.get("gist") or ""))

    def render(depth: int, gists_to: int) -> str:
        out = [f"# {title} — section map", "",
               f"Source: {doc} — given to you by your owner. {len(sections)} sections"
               + (f" in {len(files)} files" if len(files) > 1 else "") + ".",
               "Find what you need here, then read only those lines — `read` with offset (the first line) and "
               "limit (how many) — instead of the whole file. Search your memory for a phrase when you don't "
               "know where it is.", ""]
        shown = -1
        for k, lvl, line, gist in rows:
            if lvl > depth:
                continue
            if k != shown:
                f = files[k]
                out += ["", f"## {f['name']}" + (f" · {f['label']}" if f["label"] else ""), ""]
                shown = k
            out.append("  " * (lvl - 1) + f"- {line}" + (f" · {gist}" if gist and lvl <= gists_to else ""))
        return "\n".join(out) + "\n"

    depth = max((lvl for _, lvl, _, _ in rows), default=1)
    gists_to = depth
    text = render(depth, gists_to)
    while len(text.encode("utf-8")) > MAX_MAP_BYTES and depth > 1:
        if gists_to >= depth:
            gists_to = depth - 1
        else:
            depth -= 1
        text = render(depth, gists_to)
    return text
