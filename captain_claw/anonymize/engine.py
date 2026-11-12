"""The anonymization engine — the part both lanes share.

A lane only *detects* (spans). Everything after that happens here, the same
way for every lane:

1. pick non-overlapping spans (higher priority, then longer, wins; the
   caller's ``keep`` values and tokens already in the text are never touched);
2. resolve each span to an entity — the same person, email, IBAN … gets the
   same entity however it is written (``Ivan Horvat`` / ``I. Horvat`` /
   ``Ivana Horvata``);
3. find every known entity again everywhere (the vault's values and this run's
   finds — so a name found once by a label is caught in plain prose too);
4. replace with tokens (``⟦PERSON_3⟧``), assigned per vault and never derived
   from the value;
5. check: run everything again on the output; whatever turns up is replaced
   in another round, and if something still turns up after the last round the
   run fails (:class:`VerifyFailed`) — nothing is returned.

Pure: no I/O, no clock. :class:`VaultState` holds what a vault knows; the
caller loads it, runs, and stores what changed (``Entity.new`` /
``Entity.changed``, ``VaultState.counters``).
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from captain_claw.anonymize import detectors as d
from captain_claw.anonymize.detectors import (
    CAPS_WORD,
    DIGIT_KINDS,
    INITIAL,

    NAME_WORD,
    Span,
    fold,
)

TOKEN_RE = re.compile(
    r"⟦([A-Z][A-Z0-9_]*?)_(\d{1,6})(?::[a-z]{1,3})?⟧|\[\[([A-Z][A-Z0-9_]*?)_(\d{1,6})\]\]")
MAX_ROUNDS = 3
MAX_FORMS = 24
_MASK = "\x00"


def make_token(kind: str, n: int) -> str:
    return f"⟦{kind}_{n}⟧"


class VerifyFailed(RuntimeError):
    """Private data was still found in the output after the last round."""

    def __init__(self, kinds: dict[str, int]) -> None:
        self.kinds = dict(kinds)
        what = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
        super().__init__(f"private data still found after {MAX_ROUNDS} rounds ({what}) — nothing was returned")


@dataclass
class Entity:
    token: str
    kind: str
    key: str
    original: str
    forms: list[str] = field(default_factory=list)
    new: bool = False
    changed: bool = False
    count: int = 0
    # Where a new entity came from (a run's bookkeeping, not stored with the
    # value): the finder that first matched it ("email", "oib-label", "ner",
    # "extra" — the trusted model or the caller's list — "known-slug"), and
    # for a slug, the token of the name it spells.
    found_by: str = ""
    derived_from: str = ""

    def add_form(self, form: str) -> None:
        form = re.sub(r"\s+", " ", form).strip()
        if not form or form == self.original or form in self.forms:
            return
        if self.kind == "PERSON" and _better_name(form, self.original):
            # Restore puts back the plainest full form: "Ivan Horvat", not the
            # "Ivana Horvata" or "I. Horvat" that happened to come first.
            form, self.original = self.original, form
            self.changed = True
            if form in self.forms:
                return
        if len(self.forms) >= MAX_FORMS:
            return
        self.forms.append(form)
        self.changed = True


def _better_name(candidate: str, current: str) -> bool:
    cw, ow = _name_words(candidate), _name_words(current)
    if len(cw) != len(ow) or len(cw) < 2:
        return False
    if [stem_key(w) for w in cw if not _is_initial(w)] != [stem_key(w) for w in ow if not _is_initial(w)] \
            and sum(map(_is_initial, cw)) >= sum(map(_is_initial, ow)):
        return False
    score = lambda words, text: (sum(map(_is_initial, words)), len(text))  # noqa: E731
    return score(cw, candidate) < score(ow, current)


@dataclass
class Options:
    kinds: frozenset[str] | None = None
    keep: tuple[str, ...] = ()
    extra: tuple[tuple[str, str], ...] = ()  # (text, kind)
    default_cc: str = "385"
    # False: the deterministic lane doesn't run (lane "llm") — only the
    # caller's / the model's terms and what the vault already knows are found.
    deterministic: bool = True
    # One-way runs only: kinds whose values are made coarser instead of
    # tokenized (date of birth → year, address → city, age and amounts → a
    # range). AGE and AMOUNT are then also looked for.
    generalize: frozenset[str] = frozenset()


@dataclass
class Segment:
    text: str
    hint: str = ""


@dataclass
class RunResult:
    outputs: list[str]
    used: list[Entity]
    fixed: int
    rounds: int
    # Per segment, what was replaced, in the ORIGINAL text's positions:
    # (start, end, token), sorted. A writer that keeps a document's own
    # structure (Word runs, slide text) puts each token where it belongs.
    edits: list[list[tuple[int, int, str]]] = field(default_factory=list)


def _to_original(edits: list[tuple[int, int, str]], pos: int) -> int:
    """A position in the current text (original + *edits*) as one in the
    original. Never inside a token: spans that would touch one are blocked."""
    shift = 0
    for start, end, token in edits:
        if pos <= start + shift:
            break
        shift += len(token) - (end - start)
    return pos - shift


class VaultState:
    """What one vault knows: its entities and the last number used per kind."""

    def __init__(self, entities: Iterable[Entity] = (), counters: dict[str, int] | None = None) -> None:
        self.entities: list[Entity] = []
        self.counters: dict[str, int] = dict(counters or {})
        self.by_key: dict[tuple[str, str], Entity] = {}
        self.by_token: dict[str, Entity] = {}
        for e in entities:
            self._index(e)
            m = TOKEN_RE.fullmatch(e.token)
            if m:
                self.counters[e.kind] = max(self.counters.get(e.kind, 0), int(m.group(2)))

    def _index(self, e: Entity) -> None:
        self.entities.append(e)
        self.by_key.setdefault((e.kind, e.key), e)
        self.by_token[e.token] = e

    def snapshot(self) -> "VaultState":
        """A copy to run on without changing this one (a preview pass)."""
        copies = [Entity(e.token, e.kind, e.key, e.original, list(e.forms)) for e in self.entities]
        return VaultState(copies, dict(self.counters))

    def add(self, kind: str, key: str, original: str) -> Entity:
        n = self.counters.get(kind, 0) + 1
        self.counters[kind] = n
        e = Entity(make_token(kind, n), kind, key, re.sub(r"\s+", " ", original).strip(), new=True)
        self._index(e)
        return e


# ── normalization ───────────────────────────────────────────────────


def normalize(kind: str, text: str, default_cc: str = "385") -> str:
    t = str(text or "").strip()
    if kind == "EMAIL":
        return t.casefold()
    if kind == "PHONE":
        t = t.replace("(0)", "")
        digits = re.sub(r"\D", "", t)
        if t.lstrip("( ").startswith("+"):
            return digits
        if digits.startswith("00"):
            return digits[2:]
        if digits.startswith("0"):
            return default_cc + digits[1:]
        return digits
    if kind in ("IBAN", "CARD", "OIB", "JMBG", "SSN", "CVV", "PLATE", "ID_NUMBER"):
        return re.sub(r"[^0-9A-Za-zČĆŠŽĐčćšžđ]", "", t).upper()
    if kind == "DOB":
        return d.parse_date(t) or fold(t)
    if kind == "PROFILE_URL":
        return re.sub(r"^(?:https?://)?(?:www\.|m\.|mobile\.)?", "", t.casefold()).rstrip("/")
    if kind == "ORG":
        return fold(d.LEGAL_SUFFIX_RE.sub("", t)) or fold(t)
    if kind == "PERSON":
        return " ".join(fold(w) for w in _name_words(t))
    return fold(t)


# ── names ───────────────────────────────────────────────────────────

_NAME_TOKEN_RE = re.compile(rf"{NAME_WORD}|{CAPS_WORD}|{INITIAL}")
name_stem = d.name_stem

# A name is often written without its diacritics (file names, emails, old
# systems): Kovač / Kovac, Đurić / Djuric, Müller / Mueller.
_TRANSLIT = (("ü", ("u", "ue")), ("ö", ("o", "oe")), ("ä", ("a", "ae")), ("ß", ("ss",)), ("đ", ("d", "dj")))


def _name_words(text: str) -> list[str]:
    return _NAME_TOKEN_RE.findall(text)


def _is_initial(word: str) -> bool:
    return bool(re.fullmatch(INITIAL, word))


def stem_key(word: str) -> str:
    """A name word's stem with diacritics folded — what makes two spellings one name."""
    return name_stem(d.strip_accents(word))


def _key_variants(word: str) -> frozenset[str]:
    forms = {word.casefold()}
    for src, alts in _TRANSLIT:
        forms |= {f.replace(src, a) for f in forms if src in f for a in alts}
    return frozenset(name_stem(d.strip_accents(f)) for f in forms)


@dataclass(frozen=True)
class _WordInfo:
    """How a known name word is recognised in a text: by its stem (any
    spelling), with a case ending unless the bare stem is itself a name
    ("Horvata" → Horvat, "Ivana" → Ivan; "Ana" never matches "An")."""
    keys: frozenset[str]
    bare_ok: bool
    initial: str

    @staticmethod
    def of(word: str) -> "_WordInfo":
        stem = name_stem(word)
        bare = stem == word.casefold() or (len(stem) >= 3 and stem[-1] not in "aeiou")
        return _WordInfo(_key_variants(word), bare, d.strip_accents(word[:1]).casefold())


def _text_word_key(word: str) -> tuple[str, bool]:
    """(stem key, whether the word is its bare stem) for a word in a text."""
    folded = d.strip_accents(word).casefold()
    key = name_stem(folded)
    return key, key == folded


def _word_matches(tk: tuple[str, bool], info: _WordInfo) -> bool:
    key, bare = tk
    return key in info.keys and (info.bare_ok or not bare)


@dataclass
class _Person:
    entity: Entity
    words: list[str]  # name words of the canonical form, no initials
    stems: tuple[str, ...]


def _slug_fold(word: str) -> str:
    return d.strip_accents(word).casefold()


def _slug_words(text: str) -> tuple[str, ...]:
    return tuple(_slug_fold(w) for w in _SLUG_WORD_RE.findall(text) if not w.startswith("%"))


def _person_view(e: Entity) -> _Person | None:
    if e.kind != "PERSON" or e.key.startswith(("~", "/")):  # "/…": a spelling in a link, not a name
        return None
    words = [w for w in _name_words(e.original) if not _is_initial(w)]
    if not words:
        return None
    return _Person(e, words, tuple(stem_key(w) for w in words))


class _Resolver:
    """Spans → entities for one run."""

    def __init__(self, state: VaultState, options: Options) -> None:
        self.state = state
        self.options = options
        self._seen = 0
        self.by_last: dict[str, list[_Person]] = {}
        self.by_first: dict[str, list[_Person]] = {}
        self.by_stems: dict[tuple[str, ...], list[_Person]] = {}

    def _refresh(self) -> None:
        # Entities are only ever appended (or their canonical form improved,
        # which keeps its stems): index the new ones.
        for e in self.state.entities[self._seen:]:
            p = _person_view(e)
            if p is None:
                continue
            self.by_stems.setdefault(p.stems, []).append(p)
            if len(p.words) > 1:
                self.by_last.setdefault(p.stems[-1], []).append(p)
                self.by_first.setdefault(p.stems[0], []).append(p)
        self._seen = len(self.state.entities)

    def resolve(self, kind: str, text: str) -> Entity:
        key = normalize(kind, text, self.options.default_cc)
        if kind == "PERSON":
            e = self._person(text, key)
            if e is not None:
                return e
        found = self.state.by_key.get((kind, key))
        if found is not None:
            return found
        return self.state.add(kind, key, text)

    def _person(self, text: str, key: str) -> Entity | None:
        words = _name_words(text)
        if not words:
            return None
        self._refresh()
        exact = self.state.by_key.get(("PERSON", key))
        full = [w for w in words if not _is_initial(w)]
        if len(words) == 1:
            stem = stem_key(words[0])
            by_last = self.by_last.get(stem, [])
            by_first = self.by_first.get(stem, [])
            if len(by_last) == 1:
                return by_last[0].entity
            if len(by_last) > 1:
                # "Horvat" with two Horvats known: a person of its own ("a Horvat").
                skey = "~" + stem
                return self.state.by_key.get(("PERSON", skey)) or self.state.add("PERSON", skey, words[0])
            if exact is not None:
                return exact
            if len(by_first) == 1:
                return by_first[0].entity
            return None
        if exact is not None:
            return exact
        if len(full) == len(words):
            same = self.by_stems.get(tuple(stem_key(w) for w in full))
            return same[0].entity if same else None
        # Initials: "I. Horvat" is Ivan Horvat when only one person fits.
        if not full:
            return None
        fits = []
        for p in self.by_last.get(stem_key(words[-1]), []) if not _is_initial(words[-1]) else []:
            if len(p.words) != len(words):
                continue
            if all((pw[:1].casefold() == w[:1].casefold()) if _is_initial(w) else stem_key(w) == ps
                   for w, pw, ps in zip(words, p.words, p.stems)):
                fits.append(p)
        return fits[0].entity if len(fits) == 1 else None


# ── finding known entities again ────────────────────────────────────


# Words stop at underscores: a saved file's name joins words with them
# ("Acme_-_Supply_Agreement.docx", "Ivan_Horvat_CV.pdf"), and a
# known name in it must be found like anywhere else.
_TEXT_TOKEN_RE = re.compile(r"[^\W_]+|[^\w\s]|_")
_GAP_RE = re.compile(r"[ \t]+|_")
# Slugs: a known name as links and file names write it — "acme-supply",
# "Acme+Supply", "acme%20supply", "acmesupply", "Acme_Supply_offer.pdf". Each
# spelling is an entity of its own, so its token puts back exactly what was
# there (a link still opens, a file is still found).
_SLUG_WORD_RE = re.compile(r"%[0-9A-Fa-f]{2}|[^\W_]+")  # an escape ("%20") is a gap, not a word
_SLUG_GAP_RE = re.compile(r"[_+\-]|%20|[ \t]+")
_LINK_RE = re.compile(r"(?:https?://|www\.|mailto:|tel:)[^\s<>\"'\[\]{}|\\^`]+")
SLUG_MIN_JOINED = 6
# Name words in a row taken as one person's name, at most ("Ana Marija Horvat").
MAX_NAME_RUN = 4
_LEGAL_AFTER_RE = re.compile(rf",?[ \t]+{d._LEGAL}(?![\w])")
_DIGITS_RE = re.compile(r"[+(]?\d[\d \t()./\-]{4,40}\d")


def _known_spans_finder(state: VaultState, options: Options) -> Callable[[str], list[tuple[Span, Entity]]]:
    """A function that finds every known entity in a text: people by their
    name words (case endings, initials, spellings without diacritics), other
    values as written (case-insensitive), numbers whatever their spacing.

    Word indexes, not one big pattern: the time grows with the text, not with
    the vault (a vault may know thousands of people)."""
    Target = Entity | Callable[[], Entity]
    multi: dict[str, list[tuple[Entity, list[_WordInfo]]]] = {}
    surnames: dict[str, list[tuple[_WordInfo, Target, _WordInfo | None]]] = {}
    singles: dict[str, list[tuple[_WordInfo, Entity]]] = {}

    def index(table: dict, info: _WordInfo, item: Any) -> None:
        for k in info.keys:
            table.setdefault(k, []).append(item)

    persons = [p for p in (_person_view(e) for e in state.entities) if p]
    if options.kinds is None or "PERSON" in options.kinds:
        last_groups: dict[str, list[_Person]] = {}
        first_groups: dict[str, list[_Person]] = {}
        for p in persons:
            if len(p.words) > 1:
                infos = [_WordInfo.of(w) for w in p.words]
                index(multi, infos[0], (p.entity, infos))
                last_groups.setdefault(p.stems[-1], []).append(p)
                first_groups.setdefault(p.stems[0], []).append(p)
            elif len(p.stems[0]) >= 3 and fold(p.words[0]) not in d._STOP:
                info = _WordInfo.of(p.words[0])
                index(singles, info, (info, p.entity))
        for stem, group in last_groups.items():
            surname = group[0].words[-1]
            if len(stem) < 3 or fold(surname) in d._STOP:
                continue
            info = _WordInfo.of(surname)
            if len(group) == 1:
                index(surnames, info, (info, group[0].entity, _WordInfo.of(group[0].words[0])))
            else:
                def make(stem: str = stem, word: str = surname) -> Entity:
                    skey = "~" + stem
                    return state.by_key.get(("PERSON", skey)) or state.add("PERSON", skey, word)

                index(surnames, info, (info, make, None))
        for stem, group in first_groups.items():
            first = group[0].words[0]
            if len(group) == 1 and len(stem) >= 3 and fold(first) not in d._STOP:
                info = _WordInfo.of(first)
                index(singles, info, (info, group[0].entity))
        for e in state.entities:
            if e.kind == "PERSON" and e.key.startswith("~") and len(e.key) >= 4:
                info = _WordInfo.of(e.original)
                index(singles, info, (info, e))

    literal: dict[str, list[tuple[str, int, Entity, bool]]] = {}
    digit_keys: dict[str, Entity] = {}

    def add_literal(form: str, e: Entity, upper_first: bool) -> None:
        tokens = _TEXT_TOKEN_RE.findall(form.casefold())
        if tokens:
            literal.setdefault(tokens[0], []).append((form.casefold(), len(form), e, upper_first))

    for e in state.entities:
        if e.kind == "PERSON" or (options.kinds is not None and e.kind not in options.kinds):
            continue
        if e.kind in DIGIT_KINDS:
            digit_keys.setdefault(e.key, e)
        if e.kind == "ORG":
            core = d.LEGAL_SUFFIX_RE.sub("", e.original).strip()
            if len(core) >= 3:
                add_literal(core, e, core[:1].isupper() or core[:1].isdigit())
            continue
        if e.kind in ("CVV", "CARD_EXPIRY"):
            continue  # "123" / "03/27" mean something only next to a card
        for form in dict.fromkeys([e.original, *e.forms]):
            digits = re.sub(r"\D", "", form)
            if len(form) >= 4 and not (digits == re.sub(r"[\s.\-/]", "", form) and len(digits) < 6):
                add_literal(form, e, False)

    # Slug spellings of companies (and products) and of people's full names.
    slug_multi: dict[str, list[tuple[tuple[str, ...], Entity]]] = {}
    slug_joined: dict[str, list[Entity]] = {}
    slug_single: dict[str, list[Entity]] = {}  # a one-word company, inside links only

    def add_slug(words: tuple[str, ...], e: Entity, single: bool) -> None:
        if len(words) >= 2:
            slug_multi.setdefault(words[0], []).append((words, e))
            joined = "".join(words)
            if len(joined) >= SLUG_MIN_JOINED:
                slug_joined.setdefault(joined, []).append(e)
        elif single and words and len(words[0]) >= 3 and words[0] not in d._STOP:
            slug_single.setdefault(words[0], []).append(e)

    for e in state.entities:
        if e.key.startswith("/") or (options.kinds is not None and e.kind not in options.kinds):
            continue
        if e.kind == "ORG":
            add_slug(_slug_words(d.LEGAL_SUFFIX_RE.sub("", e.original)), e, True)
        elif e.kind == "PERSON" and not e.key.startswith("~"):
            add_slug(tuple(_slug_fold(w) for w in _name_words(e.original) if not _is_initial(w)), e, False)

    def find(text: str) -> list[tuple[Span, Entity]]:
        out: list[tuple[Span, Entity]] = []

        def emit(start: int, end: int, target: Target, source: str) -> None:
            e = target if isinstance(target, Entity) else target()
            out.append((Span(start, end, e.kind, 2, source), e))

        def slug(start: int, end: int, base: Entity) -> None:
            surface = text[start:end]
            if surface == base.original:
                emit(start, end, base, "known-slug")
                return
            emit(start, end, lambda: _spelling(state, base, surface), "known-slug")

        if multi or surnames or singles:
            words = [(m.start(), m.end(), m.group(0)) for m in _NAME_TOKEN_RE.finditer(text)
                     if not (m.start() > 0 and (text[m.start() - 1].isalnum() or text[m.start() - 1] in "@."))]
            keys = [_text_word_key(w) for _s, _e, w in words]

            def adjacent(i: int) -> bool:  # word i and i+1 stand next to each other
                return bool(_GAP_RE.fullmatch(text, words[i][1], words[i + 1][0]))

            # Known full names first: their words are theirs.
            covered: set[int] = set()
            for i, (start, _end, w) in enumerate(words):
                if _is_initial(w):
                    continue
                for e, infos in multi.get(keys[i][0], ()):
                    n = len(infos)
                    if i + n <= len(words) and all(
                            not _is_initial(words[i + k][2]) and _word_matches(keys[i + k], infos[k])
                            for k in range(n)) and all(adjacent(i + k) for k in range(n - 1)):
                        emit(start, words[i + n - 1][1], e, "known")
                        covered.update(range(i, i + n))

            def namey(j: int) -> bool:  # a word that can be part of a name
                w = words[j][2]
                return j not in covered and not _is_initial(w) and len(w) >= 2 and fold(w) not in d._STOP

            def lone(i: int, target: Target, own: list[_WordInfo]) -> None:
                """A known first name or surname at word *i*. Alone, it is that
                person. With name words right next to it, they are one name:
                the same person written the other way round ("HORVAT ANA"), or
                somebody else — "Ivan" of one known person (Ivan Babić) and
                "Horvat" of another (Ana Horvat) make "Ivan Horvat", a new
                person, not those two."""
                a = b = i
                while b + 1 < len(words) and b - a < MAX_NAME_RUN - 1 and adjacent(b) and namey(b + 1):
                    b += 1
                while a > 0 and b - a < MAX_NAME_RUN - 1 and adjacent(a - 1) and namey(a - 1):
                    a -= 1
                if a == b:
                    emit(words[i][0], words[i][1], target, "known")
                    return
                e = known_by_words(a, b)
                if e is not None:
                    emit(words[a][0], words[b][1], e, "known")
                elif all(any(_word_matches(keys[j], info) for info in own) for j in range(a, b + 1)):
                    emit(words[a][0], words[b][1], target, "known")
                else:
                    new_person(words[a][0], words[b][1])

            def known_by_words(a: int, b: int) -> Entity | None:
                """The known person whose name is exactly words a..b in another
                order ("HORVAT ANA"), if there is one."""
                n = b - a + 1
                for j in range(a, b + 1):
                    for e, infos in multi.get(keys[j][0], ()):
                        if len(infos) == n and all(any(_word_matches(keys[k], inf) for inf in infos)
                                                   for k in range(a, b + 1)) \
                                and all(any(_word_matches(keys[k], inf) for k in range(a, b + 1)) for inf in infos):
                            return e
                return None

            def new_person(s: int, t: int) -> None:
                surface = text[s:t]
                key = normalize("PERSON", surface, options.default_cc)
                emit(s, t, lambda: state.by_key.get(("PERSON", key)) or state.add("PERSON", key, surface),
                     "name-run")

            for i, (start, end, w) in enumerate(words):
                if _is_initial(w) or i in covered:
                    continue
                tk = keys[i]
                for info, target, first in surnames.get(tk[0], ()):
                    if not _word_matches(tk, info):
                        continue
                    if i > 0 and adjacent(i - 1):
                        pw = words[i - 1][2]
                        if first is not None and (
                                (_is_initial(pw) and d.strip_accents(pw[:1]).casefold() == first.initial)
                                or (not _is_initial(pw) and _word_matches(keys[i - 1], first))):
                            emit(words[i - 1][0], end, target, "known")
                            continue
                        if _is_initial(pw):  # "M. Horvat" is not Ana Horvat
                            new_person(words[i - 1][0], end)
                            continue
                    lone(i, target, [info] if first is None else [info, first])
                for info, e in singles.get(tk[0], ()):
                    if _word_matches(tk, info):
                        lone(i, e, [_WordInfo.of(x) for x in _name_words(e.original) if not _is_initial(x)])

        if literal:
            for m in _TEXT_TOKEN_RE.finditer(text):
                cands = literal.get(m.group(0).casefold())
                if not cands:
                    continue
                pos = m.start()
                if pos > 0 and (text[pos - 1].isalnum() or text[pos - 1] in "@."):
                    continue
                for form_cf, length, e, upper_first in cands:
                    end = pos + length
                    if text[pos:end].casefold() != form_cf:
                        continue
                    if end < len(text) and (text[end].isalnum() or text[end] == "@"):
                        continue
                    if upper_first and not (text[pos].isupper() or text[pos].isdigit()):
                        continue
                    if e.kind == "ORG":
                        suffix = _LEGAL_AFTER_RE.match(text, end)
                        end = suffix.end() if suffix else end
                    emit(pos, end, e, "known")

        if slug_multi or slug_joined or slug_single:
            links = [(m.start(), m.end()) for m in _LINK_RE.finditer(text)] if slug_single else []
            words = [(m.start(), m.end(), _slug_fold(m.group(0))) for m in _SLUG_WORD_RE.finditer(text)
                     if not m.group(0).startswith("%")]
            for i, (start, end, w) in enumerate(words):
                if start > 0 and text[start - 1] == "@":
                    continue  # an email's domain: the email is found whole
                for ws, e in slug_multi.get(w, ()):
                    n = len(ws)
                    if i + n > len(words) or any(words[i + k][2] != ws[k] for k in range(1, n)):
                        continue
                    gaps = [text[words[i + k][1]:words[i + k + 1][0]] for k in range(n - 1)]
                    if not all(_SLUG_GAP_RE.fullmatch(g) for g in gaps) or not any(g.strip() for g in gaps):
                        continue  # written with spaces: found as the name itself
                    slug(start, words[i + n - 1][1], e)
                for e in slug_joined.get(w, ()):
                    slug(start, end, e)
                if any(a <= start and end <= b for a, b in links):
                    for e in slug_single.get(w, ()):
                        slug(start, end, e)

        if digit_keys:
            for m in _DIGITS_RE.finditer(text):
                for kind in ("PHONE", "IBAN"):
                    e = digit_keys.get(normalize(kind, m.group(0), options.default_cc))
                    if e is not None:
                        emit(m.start(), m.end(), e, "known-digits")
                        break
                else:
                    e = digit_keys.get(re.sub(r"\D", "", m.group(0)))
                    if e is not None:
                        emit(m.start(), m.end(), e, "known-digits")
        return out

    return find


# ── selection ───────────────────────────────────────────────────────


def _keep_ranges(text: str, keep: tuple[str, ...]) -> list[tuple[int, int]]:
    ranges = []
    for term in keep:
        term = term.strip()
        if len(term) < 2:
            continue
        for m in re.finditer(rf"(?<![\w])(?i:{re.escape(term)})(?![\w])", text):
            ranges.append((m.start(), m.end()))
    return ranges


def _select(text: str, spans: list[Span], blocked: list[tuple[int, int]], keep_folded: set[str],
            keep_ranges: list[tuple[int, int]]) -> list[Span]:
    """Non-overlapping winners: higher priority, then longer, then earlier."""
    starts: list[int] = []
    ends: list[int] = []
    for s, e in sorted(blocked):
        i = bisect.bisect_left(starts, s)
        starts.insert(i, s)
        ends.insert(i, e)
    chosen: list[Span] = []
    for sp in sorted(spans, key=lambda s: (-s.priority, -(s.end - s.start), s.start)):
        value = text[sp.start:sp.end]
        if fold(value) in keep_folded or (sp.kind == "ORG" and fold(d.LEGAL_SUFFIX_RE.sub("", value)) in keep_folded):
            continue
        if any(ks <= sp.start and sp.end <= ke for ks, ke in keep_ranges):
            continue
        i = bisect.bisect_left(starts, sp.start)
        if i > 0 and ends[i - 1] > sp.start:
            continue
        if i < len(starts) and starts[i] < sp.end:
            continue
        starts.insert(i, sp.start)
        ends.insert(i, sp.end)
        chosen.append(sp)
    return sorted(chosen, key=lambda s: s.start)


def _spelling(state: VaultState, base: Entity, surface: str) -> Entity:
    """*surface* as an entity of its own (key ``/surface``), so its token
    restores exactly that spelling; noted as a spelling of *base*."""
    key = "/" + surface
    got = state.by_key.get((base.kind, key))
    if got is None:
        got = state.add(base.kind, key, surface)
        got.found_by, got.derived_from = "known-slug", base.token
    return got


def _mask_tokens(text: str) -> tuple[str, list[tuple[int, int]]]:
    ranges = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
    if not ranges:
        return text, []
    chars = list(text)
    for s, e in ranges:
        chars[s:e] = _MASK * (e - s)
    return "".join(chars), ranges


def _extra_spans(text: str, extra: tuple[tuple[str, str], ...]) -> list[Span]:
    out = []
    for term, kind in extra:
        term = term.strip()
        if len(term) < 2:
            continue
        # Never a piece of an email, a domain or a dotted word ("marko" in
        # marko.k@x.hr); priority 2 so a longer find (the whole email) wins.
        for m in re.finditer(rf"(?<![\w@.\-])(?i:{re.escape(term)})(?![\w@\-]|\.\w)", text):
            out.append(Span(m.start(), m.end(), kind, 2, "extra"))
    return out


# ── the run ─────────────────────────────────────────────────────────


def run(segments: list[Segment], state: VaultState, options: Options | None = None) -> RunResult:
    """Anonymize *segments* (one text, or the cells of a table) against
    *state*, which gains this run's new entities. Raises :class:`VerifyFailed`
    when private data is still found after the last round."""
    options = options or Options()
    resolver = _Resolver(state, options)
    keep_folded = {fold(k) for k in options.keep if k.strip()}
    texts = [s.text for s in segments]
    hints = [s.hint for s in segments]
    edits: list[list[tuple[int, int, str]]] = [[] for _ in segments]
    # Names and companies the caller (or the trusted model) named are known
    # from the start, so their other forms are found too ("Ana Kovač" also
    # catches "Ane Kovač"). Stored only if they turn up.
    for term, kind in options.extra:
        if kind in ("PERSON", "ORG") and len(term.strip()) >= 2:
            e = resolver.resolve(kind, term.strip())
            if e.new and not e.found_by:
                e.found_by = "extra"
    used: dict[str, Entity] = {}
    order: list[str] = []
    fixed = 0
    rounds = 0

    for round_no in range(1, MAX_ROUNDS + 1):
        # Detect in every segment (tokens already in the text are masked).
        found: list[list[Span]] = []
        masks: list[list[tuple[int, int]]] = []
        for i, text in enumerate(texts):
            masked, token_ranges = _mask_tokens(text)
            # The name model reads the original text only: on the engine's own
            # masked output it mistakes the words around a token for names,
            # and what it found in round 1 is known by then anyway.
            spans = d.detect(masked, options.kinds, opt_in=options.generalize & set(d.OPT_IN_KINDS),
                             ner=round_no == 1) \
                if options.deterministic else []
            if round_no == 1 and options.deterministic:
                h = d.hint_span(masked, hints[i])
                if h is not None and (options.kinds is None or h.kind in options.kinds):
                    spans.append(h)
            spans.extend(_extra_spans(masked, options.extra))
            masks.append(token_ranges)
            found.append(_select(masked, spans, token_ranges, keep_folded, _keep_ranges(masked, options.keep)))

        # Resolve: full names first, so a lone surname can attach to one.
        pending: dict[tuple[int, int, int], Entity] = {}

        def order_key(item: tuple[int, Span]) -> tuple[int, int]:
            _i, sp = item
            single = sp.kind == "PERSON" and len(_name_words(texts[_i][sp.start:sp.end])) < 2
            return (1 if single else 0, 0)

        items = [(i, sp) for i, spans in enumerate(found) for sp in spans]
        for i, sp in sorted(items, key=order_key):
            pending[(i, sp.start, sp.end)] = resolver.resolve(sp.kind, texts[i][sp.start:sp.end])

        # Find every known entity again, then pick the final spans.
        finder = _known_spans_finder(state, options)
        changed = False
        new_texts: list[str] = []
        for i, text in enumerate(texts):
            masked, token_ranges = _mask_tokens(text)
            known = finder(masked)
            by_span: dict[tuple[int, int], Entity] = {}
            spans = []
            for sp in found[i]:
                spans.append(sp)
                by_span[(sp.start, sp.end)] = pending[(i, sp.start, sp.end)]
            for sp, e in known:
                if options.kinds is not None and e.kind not in options.kinds:
                    continue
                spans.append(sp)
                if sp.source == "known-slug":
                    # The same words as a name found otherwise, but spelled as a link or a
                    # file name writes them: the token must put back that spelling.
                    by_span[(sp.start, sp.end)] = e
                else:
                    by_span.setdefault((sp.start, sp.end), e)
            final = _select(masked, spans, token_ranges, keep_folded, _keep_ranges(masked, options.keep))
            if not final:
                new_texts.append(text)
                continue
            links = [(m.start(), m.end()) for m in _LINK_RE.finditer(masked)]
            changed = True
            if round_no > 1:
                fixed += len(final)
            pieces = []
            pos = 0
            new_edits = []
            for sp in final:
                e = by_span[(sp.start, sp.end)]
                if links and text[sp.start:sp.end] != e.original \
                        and any(a <= sp.start and sp.end <= b for a, b in links):
                    # Inside a link a token puts back exactly what was there
                    # ("?first=Ivan" stays "Ivan", not the full name it stood for).
                    e = _spelling(state, e, text[sp.start:sp.end])
                if e.new and not e.found_by:
                    e.found_by = sp.source
                new_edits.append((_to_original(edits[i], sp.start), _to_original(edits[i], sp.end), e.token))
                surface = text[sp.start:sp.end]
                if e.kind != "PERSON" or len(_name_words(surface)) > 1:
                    e.add_form(surface)
                pieces.append(text[pos:sp.start])
                pieces.append(e.token)
                pos = sp.end
                e.count += 1
                if e.token not in used:
                    used[e.token] = e
                    order.append(e.token)
            pieces.append(text[pos:])
            new_texts.append("".join(pieces))
            edits[i] = sorted(edits[i] + new_edits)
        texts = new_texts
        rounds = round_no
        if not changed:
            break
    else:
        # The last round still replaced something: one more look, and if
        # anything is still there, fail closed.
        leftover: dict[str, int] = {}
        finder = _known_spans_finder(state, options)
        for text in texts:
            masked, token_ranges = _mask_tokens(text)
            spans = (d.detect(masked, options.kinds, opt_in=options.generalize & set(d.OPT_IN_KINDS), ner=False)
                     if options.deterministic else []) \
                + _extra_spans(masked, options.extra) + [sp for sp, _e in finder(masked)]
            for sp in _select(masked, spans, token_ranges, keep_folded, _keep_ranges(masked, options.keep)):
                leftover[sp.kind] = leftover.get(sp.kind, 0) + 1
        if leftover:
            raise VerifyFailed(leftover)

    return RunResult(outputs=texts, used=[used[t] for t in order], fixed=fixed, rounds=rounds, edits=edits)


def anonymize_text(text: str, state: VaultState, options: Options | None = None) -> tuple[str, RunResult]:
    result = run([Segment(text)], state, options)
    return result.outputs[0], result


# ── restore ─────────────────────────────────────────────────────────


def restore(text: str, state: VaultState) -> tuple[str, int, list[str]]:
    """*text* with every token of *state* put back: (text, how many were put
    back, tokens this vault doesn't know)."""
    count = 0
    unknown: list[str] = []

    def sub(m: re.Match[str]) -> str:
        nonlocal count
        kind = m.group(1) or m.group(3)
        n = int(m.group(2) or m.group(4))
        e = state.by_token.get(make_token(kind, n))
        if e is None:
            if m.group(0) not in unknown:
                unknown.append(m.group(0))
            return m.group(0)
        count += 1
        return e.original

    return TOKEN_RE.sub(sub, str(text or "")), count, unknown


# ── file names ──────────────────────────────────────────────────────


def anonymize_filename(stem: str, state: VaultState, options: Options | None = None) -> str:
    """A file name's stem with what it says about people taken out
    (``Ivan_Horvat_CV`` → ``PERSON_1_CV``), spelled filesystem-safe."""
    spaced = re.sub(r"[_\-.]+", " ", stem)
    out, _r = anonymize_text(spaced, state, options)
    out = TOKEN_RE.sub(lambda m: f"{m.group(1) or m.group(3)}_{m.group(2) or m.group(4)}", out)
    out = re.sub(r"[^\w\-]+", "_", out, flags=re.UNICODE).strip("_")
    return out or "document"


def summary(used: list[Entity]) -> dict[str, int]:
    """How many distinct entities of each kind (never their values)."""
    out: dict[str, int] = {}
    for e in used:
        out[e.kind] = out.get(e.kind, 0) + 1
    return out


__all__ = [
    "Entity", "Options", "RunResult", "Segment", "TOKEN_RE", "VaultState", "VerifyFailed",
    "anonymize_filename", "anonymize_text", "make_token", "name_stem", "normalize", "restore", "run",
    "summary",
]


def apply_edits(text: str, edits: list[tuple[int, int, str]]) -> str:
    """*text* with *edits* (original positions) applied — what :func:`run`
    returned as the output."""
    out = []
    pos = 0
    for start, end, token in sorted(edits):
        out.append(text[pos:start])
        out.append(token)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def restore_edits(text: str, state: VaultState) -> tuple[list[tuple[int, int, str]], int, list[str]]:
    """The edits that put a vault's values back into *text*: (edits, how many,
    tokens the vault doesn't know)."""
    out: list[tuple[int, int, str]] = []
    unknown: list[str] = []
    for m in TOKEN_RE.finditer(str(text or "")):
        kind = m.group(1) or m.group(3)
        e = state.by_token.get(make_token(kind, int(m.group(2) or m.group(4))))
        if e is None:
            if m.group(0) not in unknown:
                unknown.append(m.group(0))
            continue
        out.append((m.start(), m.end(), e.original))
    return out, len(out), unknown


# ── one way: coarser instead of tokens ──────────────────────────────

GENERALIZE_KINDS = ("DOB", "ADDRESS", "AGE", "AMOUNT")
_STEPS = (1, 2, 5)


def _band(value: float) -> tuple[float, float] | None:
    """The 1–2–5 band *value* falls in (1 250 → 1 000–2 000)."""
    if value < 1:
        return None
    k = 1.0
    while k * 10 <= value:
        k *= 10
    for i, step in enumerate(_STEPS):
        lo = step * k
        hi = (_STEPS[i + 1] * k) if i + 1 < len(_STEPS) else 10 * k
        if lo <= value < hi:
            return lo, hi
    return 10 * k, 20 * k


def _parse_amount(num: str) -> tuple[float, str] | None:
    """(value, thousands separator to write back) for "1.250,00", "4,237.10",
    "12 500", "250"."""
    raw = num.replace("\u00a0", " ").strip()
    if "." in raw and "," in raw:
        dec = "," if raw.rfind(",") > raw.rfind(".") else "."
    elif "," in raw:
        dec = "," if re.search(r",\d{1,2}$", raw) else ""
    elif "." in raw:
        dec = "" if re.search(r"\.\d{3}(?:\.\d{3})*$", raw) else "."
    else:
        dec = ""
    thousands = {",": ".", ".": ","}.get(dec, "." if "." in raw else ("," if "," in raw else " " if " " in raw else ""))
    cleaned = raw
    for sep in {".", ",", " "} - {dec}:
        cleaned = cleaned.replace(sep, "")
    if dec:
        cleaned = cleaned.replace(dec, ".")
    try:
        return float(cleaned), thousands or ("." if dec == "," else ",")
    except ValueError:
        return None


def _fmt(value: float, thousands: str) -> str:
    return f"{int(value):,}".replace(",", thousands)


def coarse(kind: str, text: str) -> str | None:
    """A value made coarser, or None to keep the token."""
    if kind == "DOB":
        iso = d.parse_date(text)
        if iso:
            return iso[:4]
        m = re.search(r"(?:19|20)\d{2}", text)
        return m.group(0) if m else None
    if kind == "ADDRESS":
        m = re.search(rf"\b(?:[A-Z]{{1,2}}-)?\d{{4,5}}[ \t]+({NAME_WORD}(?:[ \t]+{NAME_WORD})?)", text)
        return m.group(1) if m else None
    if kind == "AGE":
        m = re.search(r"\d{1,3}", text)
        if not m:
            return None
        lo = int(m.group(0)) // 10 * 10
        return text[:m.start()] + f"{lo}–{lo + 9}" + text[m.end():]
    if kind == "AMOUNT":
        m = re.search(r"\d[\d.,\u00a0 ]*\d|\d", text)
        parsed = _parse_amount(m.group(0)) if m else None
        band = _band(parsed[0]) if parsed else None
        if not band:
            return None
        return text[:m.start()] + f"{_fmt(band[0], parsed[1])}–{_fmt(band[1], parsed[1])}" + text[m.end():]
    return None


def generalize(segments: list[Segment], result: RunResult, state: VaultState,
               kinds: frozenset[str] | set[str]) -> RunResult:
    """*result* with the tokens of *kinds* replaced by coarser values made from
    each occurrence's own text (one-way runs: nothing is put back later)."""
    import dataclasses

    outputs, all_edits = [], []
    for seg, edits in zip(segments, result.edits):
        new_edits = []
        for s, e, token in edits:
            ent = state.by_token.get(token)
            value = coarse(ent.kind, seg.text[s:e]) if ent is not None and ent.kind in kinds else None
            new_edits.append((s, e, value if value is not None else token))
        all_edits.append(new_edits)
        outputs.append(apply_edits(seg.text, new_edits))
    return dataclasses.replace(result, outputs=outputs, edits=all_edits)
