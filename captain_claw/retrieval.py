"""Shared retrieval scoring (context engine P5).

One way to turn text into search terms and to score what comes back, used by
every store the context engine recalls from (conversation topics, insights,
semantic memory): content words with EN+HR stopwords removed and long words
matched by stem, FTS5 OR-queries ranked by bm25, bm25 normalised to 0..1,
reciprocal-rank fusion of several rankings, and age decay applied after a
store's relevance floor (so an old but exact match is not dropped by its age).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable

RRF_K = 60

# Function words dropped from search terms (English + Croatian).
STOPWORDS = frozenset("""
the and for are but not you all any can had her was one our out has have him his how its may new now
old see two who did get got let put say she too use with that this from they will would there their
what about which when were your been into than then them these those some such only also just like
more most other over very much many each here where while why after before again because could
should does doing being both same own off once under until above below between through during
please thanks thank okay yes hello want need make know think tell give show find look help
back still really maybe something anything everything going able
sam smo ste jesam nije nisu bio bila bilo biti ima imam imamo nema samo ali ako ili kao koji koja
koje kojeg kojem kojim kad kada gdje kako zato zbog što sto tko neki neka neko nešto nesto ovo ovaj
ova taj ta to tamo ovdje jer još jos već vec sve svi sva svoj svoja moj moja tvoj tvoja naš nas
vaš vas njih njega njoj nego pa te ni niti treba trebam možeš mozes može moze molim hvala daj
""".split())


def fold(text: str) -> str:
    """Lower-case, combining accents removed — what FTS5 unicode61 folds
    (č→c, š→s; not đ, which has no decomposition)."""
    decomposed = unicodedata.normalize("NFKD", str(text or "").lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# Attachment markers, links and paths: machine text, not what was asked.
_NOISE_RE = re.compile(r"\[(?:Attached|Earlier) [^\]]*\]|https?://\S+|\S*[/\\]\S*")


def query_terms(text: str, max_terms: int = 8, *, keep_identifiers: bool = False) -> list[str]:
    """Content words of *text* for search: 3+ letters, no stopwords, no
    numbers or hex ids (a year stays), nothing from attachment markers,
    links or paths; the longest *max_terms*, in their original order.

    *keep_identifiers* (searching stored text, where an exact path, commit
    or id is the best match there is): paths and links are kept as their
    words, and ids and numbers of 4+ characters stay."""
    seen: dict[str, int] = {}
    raw = str(text or "") if keep_identifiers else _NOISE_RE.sub(" ", str(text or ""))
    for pos, word in enumerate(re.findall(r"\w+", raw.lower())):
        word = word.strip("_")
        if len(word) < 3 or fold(word) in STOPWORDS or word in STOPWORDS:
            continue
        digits = sum(ch.isdigit() for ch in word)
        if keep_identifiers:
            if digits and len(word) < 4 and word.isdigit():
                continue
        elif digits * 2 >= len(word) and not (word.isdigit() and len(word) == 4):
            continue
        elif digits and re.fullmatch(r"[0-9a-f]+", word):
            continue
        seen.setdefault(word, pos)
    keep = sorted(seen, key=lambda w: (-len(w), seen[w]))[: max(1, max_terms)]
    return sorted(keep, key=lambda w: seen[w])


def stem(term: str) -> str:
    """The prefix a term matches on: long words drop their last two letters
    (inflection: putovanje/putovanja, invoices/invoice)."""
    if len(term) < 6:
        return term
    return term[: max(5, len(term) - 2)]


def fts_term(term: str, prefix_all: bool = False) -> str:
    """One FTS5 term: a word of 4+ letters matches as a prefix (a long one by
    its stem); a 3-letter word only as itself, unless *prefix_all*. An id
    (any digit in it) matches only as itself."""
    if any(ch.isdigit() for ch in term) and not prefix_all:
        return f'"{term.replace(chr(34), "")}"'
    base = stem(term).replace('"', "")
    return f'"{base}"*' if (prefix_all or len(term) >= 4) else f'"{base}"'


def fts_match(terms: Iterable[str], prefix_all: bool = False) -> str:
    """An FTS5 MATCH expression that ORs *terms* (bm25 ranks by how many and
    how rare); empty when there are none."""
    return " OR ".join(fts_term(t, prefix_all) for t in terms)


def matched_terms(terms: Iterable[str], *texts: str) -> list[str]:
    """The *terms* that occur in *texts* the way FTS matches them: a 3-letter
    word as a whole word, a longer one by its stem as a word prefix."""
    tokens = set(re.findall(r"\w+", fold(" ".join(str(t or "") for t in texts))))
    hits = []
    for term in terms:
        folded = fold(term)
        if any(ch.isdigit() for ch in term):
            if folded in tokens:
                hits.append(term)
        elif len(term) >= 4:
            prefix = stem(folded)
            if any(tok.startswith(prefix) for tok in tokens):
                hits.append(term)
        elif folded in tokens:
            hits.append(term)
    return hits


def bm25_relevance(rank: float, scale: float = 4.0) -> float:
    """FTS5 bm25 (negative, lower is better) as a 0..1 relevance: a hit
    scoring ``scale`` lands at 0.5."""
    rel = max(0.0, -float(rank or 0.0))
    return rel / (rel + scale) if rel > 0 else 0.0


def rrf(rankings: Iterable[Iterable[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal-rank fusion: id -> sum of 1/(k + rank) over the rankings."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def decayed(score: float, age_days: float, half_life_days: float) -> float:
    """*score* halved every *half_life_days* of age (applied after a store's
    relevance floor, never before)."""
    if half_life_days <= 0 or age_days <= 0:
        return score
    return score * math.pow(0.5, age_days / half_life_days)
