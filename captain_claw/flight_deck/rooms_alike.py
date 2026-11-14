"""Data rooms: tokens that may be one party, and making them one.

Code makes one token of what it can prove is one person, company or address
(word order, case, diacritics, a middle name more when only one person fits,
"Kovačka c. 12" / "KOVAČKA CESTA 12"). What it can't prove it leaves apart
— and lists here for the person, who knows: a one-letter typo, a first name
alone, a surname that changed between two files, a company's short name, one
street and number in two towns.

When the person says two are one, the second token becomes the first one's
spelling (⟦PERSON_2:4⟧) in the vault and in every copy in the room — Office
files, text, the search index — without a new run. Its old tokens still
restore (deliveries made before keep working). "Not the same" is remembered,
so a pair is asked about once.
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from captain_claw.anonymize import detectors as d
from captain_claw.anonymize import engine

MAX_SUGGESTIONS = 300
# A word companies share is worth asking about when this few of them have it
# ("Sokol Dom d.o.o." / "Sokol d.o.o."), not when many do ("Zagreb").
MAX_SHARING = 4
# Two files list "mostly the same people" when the smaller has this many or
# more and half of them are in the other too.
MIN_PARALLEL = 5
_GENERIC_ORG_WORDS = frozenset({
    "doo", "dd", "jdoo", "obrt", "ltd", "gmbh", "inc", "llc", "grupa", "group", "holding", "hrvatska", "zagreb",
    "trgovina", "usluge", "servis", "centar", "hotel", "dom", "banka", "bank", "osiguranje", "consulting",
    "company", "international", "solutions", "systems", "media", "studio", "agencija", "projekt", "plus",
})
_TOKEN_ANY = re.compile(r"⟦[A-Z][A-Z0-9_]*?_\d{1,6}(?::[a-z0-9]{1,3})?⟧")


def _fold(text: str) -> str:
    s = unicodedata.normalize("NFKD", str(text or "").replace("đ", "dj").replace("Đ", "Dj"))
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _fold(text))


def _one_edit(a: str, b: str) -> bool:
    """*a* and *b* differ by one letter: changed, added, dropped or two swapped."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return short[i:] == long_[i + 1:]


def _letters_off(w1: list[str], w2: list[str]) -> bool:
    """Two names of as many words, at least one the same, every other one a
    letter off ("Lorena Mae Dizon" / "LORENA MAY DIZEN")."""
    if len(w1) != len(w2) or not set(w1) & set(w2):
        return False
    rest1, rest2 = list(w1), list(w2)
    for x in set(w1) & set(w2):
        rest1.remove(x)
        rest2.remove(x)
    if not rest1 or len(rest1) > 2:
        return False
    for x in rest1:
        y = next((y for y in rest2 if min(len(x), len(y)) >= 3 and _one_edit(x, y)), None)
        if y is None:
            return False
        rest2.remove(y)
    return True


def tokens_by_file(texts: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """{token: the files it is in} from (file id, text Claude reads)."""
    out: dict[str, set[str]] = defaultdict(set)
    for fid, text in texts:
        for m in engine.TOKEN_RE.finditer(text or ""):
            kind, n = m.group(1) or m.group(3), m.group(2) or m.group(4)
            out[engine.make_token(kind, int(n))].add(fid)
    return out


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def suggest(state: engine.VaultState, files: dict[str, set[str]],
            not_same: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Pairs of tokens that may be one party, for the person to say: each
    {kind, a, b, why}, a and b {token, text, files}. Only tokens in the
    room's files; never a pair already answered."""
    skip = set(not_same)
    live = [e for e in state.entities if not e.same_as and not e.key.startswith(("~", "/")) and files.get(e.token)]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(kind: str, a: engine.Entity, b: engine.Entity, why: str, strength: int) -> None:
        key = pair_key(a.token, b.token)
        if key in skip or key in seen:
            return
        seen.add(key)
        if len(files.get(a.token, ())) < len(files.get(b.token, ())):
            a, b = b, a  # the one in more files first: the likely one to keep
        out.append({"kind": kind, "why": why, "strength": strength, "key": key,
                    "a": {"token": a.token, "text": a.original, "files": len(files.get(a.token, ()))},
                    "b": {"token": b.token, "text": b.original, "files": len(files.get(b.token, ()))}})

    people = [e for e in live if e.kind == "PERSON"]
    full = [(e, _words(e.original)) for e in people]
    full = [(e, w) for e, w in full if w]
    by_word: dict[str, list[tuple[engine.Entity, list[str]]]] = defaultdict(list)
    for e, w in full:
        for x in set(w):
            by_word[x].append((e, w))

    # A one-letter typo: the same words but one, that one a letter off.
    by_len: dict[int, list[tuple[engine.Entity, list[str]]]] = defaultdict(list)
    for e, w in full:
        if len(w) >= 2:
            by_len[len(w)].append((e, w))
    for group in by_len.values():
        # Each name with one of its words left out: two names that share all
        # the rest, the left-out words a letter apart, are a typo's two sides.
        index: dict[tuple[str, ...], list[tuple[engine.Entity, str]]] = defaultdict(list)
        for e, w in group:
            ws = sorted(w)
            for i in range(len(ws)):
                index[tuple(ws[:i] + ws[i + 1:])].append((e, ws[i]))
        for cands in index.values():
            if not 2 <= len(cands) <= 6:
                continue
            for i, (e1, x1) in enumerate(cands):
                for e2, x2 in cands[i + 1:]:
                    if e1.token != e2.token and min(len(x1), len(x2)) >= 4 and _one_edit(x1, x2):
                        add("PERSON", e1, e2, "One letter apart", 3)

    # A name alone: one word that is a word of a full name (one to three people).
    for e, w in full:
        if len(w) != 1 or len(w[0]) < 3 or w[0] in d._STOP:
            continue
        cands = [(f, fw) for f, fw in by_word.get(w[0], ()) if len(fw) >= 2 and f.token != e.token]
        if 1 <= len(cands) <= 3:
            for f, _fw in cands:
                add("PERSON", f, e, "A name alone", 2)

    # The same first name (or the same surname), never in one file — but in
    # two files that list mostly the same people (two registers, a register
    # and its payroll): a name that changed between them (a married name).
    people_in: dict[str, set[str]] = defaultdict(set)
    for e in people:
        for fid in files.get(e.token, ()):
            people_in[fid].add(e.token)
    parallel_cache: dict[tuple[str, str], bool] = {}

    def parallel(x: str, y: str) -> bool:
        key = (x, y) if x < y else (y, x)
        if key not in parallel_cache:
            px, py = people_in.get(x, set()), people_in.get(y, set())
            small = min(len(px), len(py))
            parallel_cache[key] = small >= MIN_PARALLEL and len(px & py) >= small / 2
        return parallel_cache[key]

    for e1, w1 in full:
        if len(w1) < 2:
            continue
        for x in set(w1):
            group = by_word.get(x, ())
            if len(x) < 3 or len(group) > 12:
                continue
            for e2, w2 in group:
                if e2.token <= e1.token or len(w2) < 2:
                    continue
                if _letters_off(w1, w2):
                    add("PERSON", e1, e2, "A letter or two apart", 3)
                    continue
                f1, f2 = files.get(e1.token, set()), files.get(e2.token, set())
                if f1 & f2 or not any(parallel(x, y) for x in f1 for y in f2):
                    continue
                if set(w1) & set(w2) and not (set(w1) <= set(w2) or set(w2) <= set(w1)):
                    add("PERSON", e1, e2, "Share a name, never on the same list", 1)

    # Companies: one's words inside the other's, a typo, or a rare word shared.
    orgs = [(e, [w for w in _words(d.LEGAL_SUFFIX_RE.sub("", e.original)) if w not in ("d", "o", "doo", "dd")])
            for e in live if e.kind == "ORG"]
    orgs = [(e, w) for e, w in orgs if w]
    org_by_word: dict[str, list[tuple[engine.Entity, list[str]]]] = defaultdict(list)
    for e, w in orgs:
        for x in set(w):
            org_by_word[x].append((e, w))
    for x, group in org_by_word.items():
        if len(x) < 4 or x in _GENERIC_ORG_WORDS or x in d._STOP or not 2 <= len(group) <= MAX_SHARING:
            continue
        for i, (e1, w1) in enumerate(group):
            for e2, w2 in group[i + 1:]:
                if e1.token == e2.token:
                    continue
                inside = set(w1) <= set(w2) or set(w2) <= set(w1)
                add("ORG", e1, e2, "One name inside the other" if inside else f"Both say “{x}”", 3 if inside else 1)
    cores = defaultdict(list)
    for e, w in orgs:
        cores[len(" ".join(w))].append((e, " ".join(w)))
    for n, group in cores.items():
        near = group + cores.get(n + 1, [])
        for i, (e1, c1) in enumerate(group):
            if len(c1) < 5:
                continue
            for e2, c2 in near[i + 1:]:
                if e1.token != e2.token and _one_edit(c1, c2):
                    add("ORG", e1, e2, "One letter apart", 3)

    # Addresses: one street and number in two towns, or a letter off.
    places = [(e, engine._address_parts(e.original)) for e in live if e.kind == "ADDRESS"]
    places = [(e, p) for e, p in places if p is not None]
    by_number: dict[str, list[tuple[engine.Entity, Any]]] = defaultdict(list)
    for e, p in places:
        by_number[p.number].append((e, p))
    for group in by_number.values():
        if len(group) > 40:
            continue
        for i, (e1, p1) in enumerate(group):
            for e2, p2 in group[i + 1:]:
                if e1.token == e2.token:
                    continue
                s1 = [w for w in p1.street if w not in engine._STREET_KINDS] or list(p1.street)
                s2 = [w for w in p2.street if w not in engine._STREET_KINDS] or list(p2.street)
                same_street = len(s1) == len(s2) and all(map(engine._word_fits, s1, s2))
                if same_street and p1.town and p2.town:
                    add("ADDRESS", e1, e2, "Same street and number, another town", 2)
                elif not same_street and len(s1) == len(s2) and sum(a != b for a, b in zip(s1, s2)) == 1 \
                        and all(a == b or (min(len(a), len(b)) >= 4 and _one_edit(a, b)) for a, b in zip(s1, s2)):
                    add("ADDRESS", e1, e2, "One letter apart", 3)

    out.sort(key=lambda s: (-s["strength"], s["kind"], s["a"]["text"].casefold()))
    return out[:MAX_SUGGESTIONS]


# ── making two one ─────────────────────────────────────────────────


def rewrite_text(text: str, mapping: dict[str, str]) -> str:
    if not mapping or "⟦" not in text:
        return text
    return _TOKEN_ANY.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)


def rewrite_blob(name: str, blob: bytes, mapping: dict[str, str]) -> bytes | None:
    """*blob* with its tokens written as *mapping* says: an Office file's XML
    parts, or text. None when it can't be (another format) — that file runs
    again instead. The same bytes when nothing in it changes."""
    if blob[:2] == b"PK":
        try:
            src = zipfile.ZipFile(io.BytesIO(blob))
        except zipfile.BadZipFile:
            return None
        changed = False
        parts: list[tuple[zipfile.ZipInfo, bytes]] = []
        for info in src.infolist():
            data = src.read(info)
            if re.search(r"\.(?:xml|rels|txt|csv|md|html?)$", info.filename, re.I) and "⟦".encode() in data:
                try:
                    new = rewrite_text(data.decode("utf-8"), mapping).encode("utf-8")
                except UnicodeDecodeError:
                    new = data
                if new != data:
                    data, changed = new, True
            parts.append((info, data))
        if not changed:
            return blob
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as dst:
            for info, data in parts:
                dst.writestr(info, data, compress_type=info.compress_type)
        return out.getvalue()
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return None if "⟦".encode() in blob else blob
    return rewrite_text(text, mapping).encode("utf-8")


def rewrite_leftovers(leftovers: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """A file's leftover lists with the merged tokens given their new number."""
    base = {k: re.sub(r":\d+⟧$", "⟧", v) for k, v in mapping.items()}

    def fix(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**x, "token": base.get(str(x.get("token") or ""), x.get("token"))} for x in items]

    return {k: fix(v) if isinstance(v, list) else v for k, v in (leftovers or {}).items()}
