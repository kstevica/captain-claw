"""The deterministic lane: find private data with code alone.

No model, no network, no optional dependency — the same text gives the same
spans on every deck. Each detector returns :class:`Span` objects (which may
overlap); :mod:`captain_claw.anonymize.engine` picks the winners, resolves them
to entities and replaces them.

Priorities (higher wins an overlap, then the longer span):

* 3 — validated by a checksum (IBAN, card, OIB, JMBG) or named by the caller
* 2 — structural: a label ("Name:", "Adresa:"), a column / key hint, an email
  address, an international phone number, a value already in the vault
* 1 — context: honorifics, greetings, sign-offs, legal suffixes, street shapes
* 0 — weak shapes (a national phone number without a label)
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

# Upper / lower case letters, Latin with the diacritics of the languages
# Captain is used in (Python's ``re`` has no \p{Lu}).
UP = "A-ZÀ-ÖØ-ÞĀĂĄĆĈĊČĎĐĒĔĖĘĚĜĞĠĢĤĦĨĪĬĮİĴĶĹĻĽĿŁŃŅŇŌŎŐŒŔŖŘŚŜŞŠŢŤŦŨŪŬŮŰŲŴŶŸŹŻŽ"
LO = "a-zß-öø-ÿāăąćĉċčďđēĕėęěĝğġģĥħĩīĭįıĵķĺļľŀłńņňōŏőœŕŗřśŝşšţťŧũūŭůűųŵŷźżž"

KINDS = (
    "PERSON", "ORG", "EMAIL", "PHONE", "ADDRESS", "DOB", "IBAN", "CARD", "CVV", "CARD_EXPIRY",
    "OIB", "JMBG", "ID_NUMBER", "SSN", "IP", "PROFILE_URL", "PLATE",
)

# Found only when asked for (``opt_in``): ages and money amounts aren't private
# by themselves, but in a document shared for good they narrow a person down,
# so one-way runs can turn them into ranges.
OPT_IN_KINDS = ("AGE", "AMOUNT")

# Kinds whose value is a number: compared digit by digit when checking output.
DIGIT_KINDS = frozenset({"PHONE", "IBAN", "CARD", "OIB", "JMBG", "SSN"})


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str
    priority: int
    source: str


# ── shared pieces ───────────────────────────────────────────────────

# Ivan, Kovačić-Horvat, O'Brien, McDonald.
NAME_WORD = rf"(?:[{UP}]['’])?[{UP}][{LO}]+(?:[{UP}][{LO}]+)?(?:[-'’][{UP}]?[{LO}]+)*"
CAPS_WORD = rf"[{UP}]{{2,}}(?:-[{UP}]{{2,}})?"
INITIAL = rf"[{UP}]\."
PARTICLE = r"(?:van|von|de|da|di|del|della|der|den|du|la|le|bin|ibn|al|ten|ter)"


def name_seq(min_words: int, max_words: int) -> str:
    """1+ capitalized words on one line (initials and particles allowed;
    an all-caps surname after the first word: ``Ivan HORVAT``)."""
    first = rf"(?:{NAME_WORD}|{INITIAL})"
    tail = rf"(?:[ \t]+(?:{PARTICLE}[ \t]+)?(?:{NAME_WORD}|{CAPS_WORD}|{INITIAL}))"
    return rf"{first}{tail}{{{max(0, min_words - 1)},{max(0, max_words - 1)}}}"


# Capitalized words that are not names (case-folded). A name span loses them
# from both ends; a span left with none of its own words is dropped.
_STOP = {
    # EN function words, greetings, mail and document words
    "the", "this", "that", "these", "those", "a", "an", "and", "or", "of", "for", "with", "from",
    "to", "in", "on", "at", "by", "as", "is", "it", "we", "i", "you", "he", "she", "they", "my",
    "our", "your", "dear", "hi", "hello", "hey", "thanks", "thank", "regards", "best", "kind",
    "warm", "cheers", "sincerely", "yours", "team", "sir", "madam", "all", "everyone", "subject",
    "re", "fwd", "fw", "date", "sent", "cc", "bcc", "attn", "note", "please", "yes", "no", "ok",
    "mr", "mrs", "ms", "miss", "dr", "prof", "today", "tomorrow", "yesterday", "page", "total",
    "name", "address", "phone", "email", "company", "client", "customer", "street", "city",
    "country", "inc", "ltd", "llc", "gmbh", "invoice", "contract", "agreement", "section",
    "board", "management", "staff", "support", "office", "department", "family", "customer", "service",
    "sales", "admin", "hr", "it", "legal", "finance", "accounts", "marketing", "director", "manager",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december",
    # HR
    "poštovani", "poštovana", "poštovanje", "dragi", "draga", "bok", "pozdrav", "pozdravi",
    "lijep", "lijepi", "srdačan", "srdačni", "hvala", "s", "sa", "i", "u", "na", "za", "od", "do",
    "ime", "prezime", "adresa", "tvrtka", "društvo", "firma", "klijent", "kupac", "ugovor",
    "račun", "ulica", "grad", "gospodin", "gospođa", "gospođica", "g", "gđa", "gđica",
    "ponedjeljak", "utorak", "srijeda", "četvrtak", "petak", "subota", "nedjelja",
    "siječanj", "veljača", "ožujak", "travanj", "svibanj", "lipanj", "srpanj", "kolovoz", "rujan",
    "listopad", "studeni", "prosinac", "danas", "sutra", "jučer", "stranica", "ukupno",
    "republika", "hrvatska", "zagreb", "tim", "uprava", "služba", "podrška", "ured", "obitelj", "odjel",
    "prodaja", "računovodstvo", "direkcija",
}

_WORD_RE = re.compile(rf"{NAME_WORD}|{CAPS_WORD}|{INITIAL}|[^\s]+")


# Croatian case endings a name takes (Ivan → Ivana, Ivanu, Ivanom, Ivane;
# Ana → Ane, Ani, Anu, Anom). Possessive -in/-ov forms are left out on
# purpose: they would make Marin and Marina one person.
NAME_ENDINGS = ("ama", "ima", "om", "em", "oj", "a", "e", "i", "o", "u")


def name_stem(word: str) -> str:
    w = word.casefold()
    for end in NAME_ENDINGS:
        if w.endswith(end) and len(w) - len(end) >= 2:
            return w[: -len(end)]
    return w


def fold(text: str) -> str:
    """Case-folded, whitespace collapsed, ends stripped of punctuation."""
    return re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n,;:.!?\"'()[]<>").casefold()


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).replace(
        "đ", "d").replace("Đ", "D")


def trim_name(text: str, start: int, end: int) -> tuple[int, int] | None:
    """The span with non-name words taken off both ends; None when nothing
    name-like is left (only initials, or only stop words)."""
    words = [(m.start() + start, m.end() + start, m.group(0)) for m in _WORD_RE.finditer(text[start:end])]
    while words and (fold(words[0][2]) in _STOP or not re.fullmatch(rf"{NAME_WORD}|{INITIAL}", words[0][2])):
        words.pop(0)
    while words and (fold(words[-1][2]) in _STOP or re.fullmatch(INITIAL, words[-1][2])):
        words.pop()
    if not words or not any(re.fullmatch(rf"{NAME_WORD}|{CAPS_WORD}", w[2]) for w in words):
        return None
    return words[0][0], words[-1][1]


def _trim(text: str, start: int, end: int, chars: str = " \t\r\n,;:!?\"'()[]<>") -> tuple[int, int]:
    while start < end and text[start] in chars:
        start += 1
    while end > start and text[end - 1] in chars:
        end -= 1
    return start, end


# ── checksums ───────────────────────────────────────────────────────

_IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22, "BH": 22,
    "BR": 29, "CH": 21, "CY": 28, "CZ": 24, "DE": 22, "DK": 18, "EE": 20, "ES": 24, "FI": 18,
    "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23, "GL": 18, "GR": 27, "HR": 21, "HU": 28,
    "IE": 22, "IL": 23, "IS": 26, "IT": 27, "KW": 30, "KZ": 20, "LB": 28, "LI": 21, "LT": 20,
    "LU": 20, "LV": 21, "MC": 27, "MD": 24, "ME": 22, "MK": 19, "MT": 31, "NL": 18, "NO": 15,
    "PL": 28, "PT": 25, "RO": 24, "RS": 22, "SA": 24, "SE": 24, "SI": 19, "SK": 24, "SM": 27,
    "TR": 26, "UA": 29, "XK": 20,
}


def iban_ok(value: str) -> bool:
    v = re.sub(r"[\s\-]", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", v):
        return False
    want = _IBAN_LENGTHS.get(v[:2])
    if want is not None and len(v) != want:
        return False
    moved = v[4:] + v[:4]
    digits = "".join(str(int(c, 36)) for c in moved)
    return int(digits) % 97 == 1


def oib_ok(digits: str) -> bool:
    """Croatian OIB: ISO 7064, MOD 11,10."""
    if not re.fullmatch(r"\d{11}", digits):
        return False
    a = 10
    for d in digits[:10]:
        a = (a + int(d)) % 10 or 10
        a = (a * 2) % 11
    check = 11 - a
    return (0 if check == 10 else check) == int(digits[10])


def jmbg_ok(digits: str) -> bool:
    """Former-Yugoslav JMBG: DDMMYYY RR BBB K, mod 11."""
    if not re.fullmatch(r"\d{13}", digits):
        return False
    day, month = int(digits[0:2]), int(digits[2:4])
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return False
    a = [int(c) for c in digits]
    s = sum(w * (a[i] + a[i + 6]) for i, w in enumerate((7, 6, 5, 4, 3, 2)))
    m = 11 - s % 11
    return (0 if m > 9 else m) == a[12]


# ── dates ───────────────────────────────────────────────────────────

_MONTHS_EN = ("january|february|march|april|may|june|july|august|september|october|november|december"
              "|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec")
_MONTHS_HR = ("siječnja|veljače|ožujka|travnja|svibnja|lipnja|srpnja|kolovoza|rujna|listopada|studenoga"
              "|studenog|prosinca|siječanj|veljača|ožujak|travanj|svibanj|lipanj|srpanj|kolovoz|rujan"
              "|listopad|studeni|prosinac")
_MONTH_NUM = {}
for _i, _names in enumerate((
    ("january", "jan", "siječnja", "siječanj"), ("february", "feb", "veljače", "veljača"),
    ("march", "mar", "ožujka", "ožujak"), ("april", "apr", "travnja", "travanj"),
    ("may", "svibnja", "svibanj"), ("june", "jun", "lipnja", "lipanj"),
    ("july", "jul", "srpnja", "srpanj"), ("august", "aug", "kolovoza", "kolovoz"),
    ("september", "sep", "sept", "rujna", "rujan"), ("october", "oct", "listopada", "listopad"),
    ("november", "nov", "studenoga", "studenog", "studeni"), ("december", "dec", "prosinca", "prosinac"),
), start=1):
    for _n in _names:
        _MONTH_NUM[_n] = _i

DATE = (
    r"(?:\d{1,2}\.[ \t]?\d{1,2}\.[ \t]?(?:19|20)\d{2}\.?"
    r"|\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}"
    r"|(?:19|20)\d{2}-\d{1,2}-\d{1,2}"
    rf"|\d{{1,2}}\.?[ \t]+(?i:{_MONTHS_EN}|{_MONTHS_HR})[ \t]+(?:19|20)\d{{2}}\.?"
    rf"|(?i:{_MONTHS_EN})[ \t]+\d{{1,2}},?[ \t]+(?:19|20)\d{{2}})"
)
_DATE_FULL_RE = re.compile(rf"^{DATE}$")


def parse_date(text: str) -> str | None:
    """``YYYY-MM-DD`` for a date in any of :data:`DATE`'s shapes (day first
    for numeric dates, as in Europe), else None."""
    s = text.strip().rstrip(".")
    m = re.fullmatch(r"(\d{1,2})[./-][ \t]?(\d{1,2})[./-][ \t]?(\d{4})", s)
    if m:
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
    elif m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s):
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    elif m := re.fullmatch(r"(\d{1,2})\.?\s+(\w+)\s+(\d{4})", s):
        d, mo, y = int(m[1]), _MONTH_NUM.get(m[2].casefold(), 0), int(m[3])
    elif m := re.fullmatch(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", s):
        mo, d, y = _MONTH_NUM.get(m[1].casefold(), 0), int(m[2]), int(m[3])
    else:
        return None
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


# ── the patterns ────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"(?<![\w.+\-])[A-Za-z0-9][A-Za-z0-9._%+\-]*@[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?"
                       r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)*\.[A-Za-z]{2,}(?![\w\-])")

_PHONE_INTL_RE = re.compile(r"(?<![\w+])(?:\+|00)[1-9]\d{0,2}(?:[ \t.\-/]?\(?\d{1,4}\)?){2,6}(?![\w])")
_PHONE_NAT_RE = re.compile(r"(?<![\w/.\-+])\(?0[1-9]\d{0,2}\)?(?:[ \t.\-/]?\d{2,4}){2,3}(?![\w/\-])")
_PHONE_LABEL_RE = re.compile(
    r"(?:(?i:\b(?:tel|telephone|phone|mob|mobile|cell|fax|gsm|telefon|mobitel|kontakt[ \t]+telefon))\.?[ \t]*[:\-–]?"
    r"|\b[TtMm][ \t]*:)[ \t]*([+(]?\d[\d \t()./\-]{5,20}\d)")

_IBAN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2}\d{2}(?:[ \-]?[A-Z0-9]{1,4}){3,8})(?![A-Za-z0-9])")
_OIB_RE = re.compile(r"(?<!\d)\d{11}(?!\d)")
_OIB_LABEL_RE = re.compile(r"(?i:\bOIB)[ \t]*[:\-–]?[ \t]*(\d{11})(?!\d)")
_JMBG_RE = re.compile(r"(?<!\d)\d{13}(?!\d)")
_JMBG_LABEL_RE = re.compile(r"(?i:\bJMBG)[ \t]*[:\-–]?[ \t]*(\d{13})(?!\d)")
# A Croatian company's registry number, "MB: 01234567" (matični broj, 8
# digits; older ones 7). Capitals only: "512 MB" is a size.
_MB_LABEL_RE = re.compile(r"(?<![\w])MB[ \t]*[:\-–]?[ \t]*(\d{7,8})(?!\d)")
_SSN_RE = re.compile(r"(?<![\d\-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d\-])")

_IPV4_RE = re.compile(r"(?<![\d.\w])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.]?\d)")
_IPV6_RE = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}(?![\w:])"
                      r"|(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){1,7}:(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){0,6})?(?![\w:])")
_IP_SKIP = {"127.0.0.1", "0.0.0.0", "255.255.255.255"}

_PROFILE_RE = re.compile(
    r"(?<![\w/])(?:https?://)?(?:www\.|m\.|mobile\.)?"
    r"(?:linkedin\.com/in|facebook\.com|fb\.com|instagram\.com|twitter\.com|x\.com|github\.com|t\.me|wa\.me"
    r"|tiktok\.com)/@?([A-Za-z0-9_.\-%+]{2,80})/?", re.I)
_PROFILE_SKIP = {"share", "sharer", "sharer.php", "help", "about", "login", "home", "intent", "search",
                 "explore", "settings", "privacy", "policies", "pages", "groups", "watch", "events",
                 "hashtag", "i", "features", "pricing", "orgs", "topics", "marketplace", "legal"}

_PLATE_RE = re.compile(
    r"(?<![\w])(?:ZG|ST|RI|OS|ZD|PU|SB|KA|VŽ|VZ|BJ|ČK|CK|DA|DE|DJ|DU|GS|IM|KC|KR|KT|KŽ|KZ|MA|NA|NG|OG|PS|PŽ|PZ"
    r"|SK|SL|ŠI|SI|VK|VT|VU|ŽU|ZU|BM|PG|MO)[ \-]?\d{3,4}[ \-]?[A-ZČĆŠŽĐ]{1,2}(?![\w])")

_HONORIFIC = (r"(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Sir|Dame|Lady|Lord|Madam|Mme|Mlle|Herr|Frau|dr|prof|mr|mag|doc|ing"
              r"|g|gđa|gđica|gosp|gospodin(?:e|a|u|om)?|gospođ(?:a|e|i|u|o|om)|gospođic(?:a|e|i|u|o|om))")
_HONORIFIC_RE = re.compile(
    rf"(?<![\w]){_HONORIFIC}\.?(?:[ \t]+(?:sc|med|iur|oec|ing|dipl)\.)*[ \t]+({name_seq(1, 4)})")

_PERSON_LABELS = (
    r"name|full[ \t]+name|first[ \t]+name|last[ \t]+name|surname|given[ \t]+name|family[ \t]+name"
    r"|contact(?:[ \t]+person)?|client|patient|employee|owner|director|ceo|signed(?:[ \t]+by)?|prepared[ \t]+by"
    r"|author|attn|attention|applicant|tenant|landlord|witness|insured|beneficiary|holder|account[ \t]+holder"
    r"|ime(?:[ \t]+i[ \t]+prezime)?|prezime|kontakt(?:[ \t]+osoba)?|klijent|pacijent|zaposlenik|zaposlenica"
    r"|vlasnik|vlasnica|direktor|direktorica|potpis|potpisao|potpisala|pripremio|pripremila|autor|autorica"
    r"|odgovorna[ \t]+osoba|primatelj|pošiljatelj|podnositelj(?:[ \t]+zahtjeva)?|korisnik|korisnica"
    r"|ugovaratelj|osiguranik|svjedok|zakupnik|zakupodavac|najmoprimac|najmodavac|nositelj|imatelj"
)
_PERSON_LABEL_RE = re.compile(
    rf"(?<![\w])(?i:{_PERSON_LABELS})[ \t]*[:\-–][ \t]*({name_seq(1, 4)})(?=[ \t]*(?:$|[\n,;(<|]))", re.M)

_GREETING_RE = re.compile(
    rf"(?<![\w])(?i:dear|hi|hello|hey|poštovani|poštovana|dragi|draga|bok|zdravo|pozdrav)[ \t]+"
    rf"(?:{_HONORIFIC}\.?[ \t]+)?({name_seq(1, 3)})[ \t]*(?=[,!:\n]|$)", re.M)

_SIGNOFF_RE = re.compile(
    r"(?i:best[ \t]+regards|kind[ \t]+regards|warm[ \t]+regards|regards|best|cheers|thanks|thank[ \t]+you"
    r"|sincerely|yours[ \t]+sincerely|yours[ \t]+truly|lijep[ \t]+pozdrav|lijepi[ \t]+pozdrav"
    r"|srdačan[ \t]+pozdrav|srdačni[ \t]+pozdravi|s[ \t]+poštovanjem|pozdrav|pozdravi|hvala)"
    rf"[,.!]?[ \t]*\r?\n[ \t]*\r?\n?[ \t]*({name_seq(1, 4)})[ \t]*(?=\r?\n|$)")

_DISPLAY_NAME_RE = re.compile(
    rf"[\"']?({name_seq(1, 4)})[\"']?[ \t]*<[^<>@\s]+@[^<>\s]+>")

_ORG_WORD = rf"(?:{NAME_WORD}|[{UP}0-9][{UP}{LO}0-9&\-]*)"
# Croatian legal forms written out, as registries name a company ("… društvo s
# ograničenom odgovornošću za računalne i srodne aktivnosti").
_LEGAL_LONG = (r"(?i:jednostavno[ \t]+društvo[ \t]+s[ \t]+ograničenom[ \t]+odgovornošću"
               r"|društvo[ \t]+s[ \t]+ograničenom[ \t]+odgovornošću|dioničko[ \t]+društvo|komanditno[ \t]+društvo"
               r"|javno[ \t]+trgovačko[ \t]+društvo)")
_LEGAL = (r"(?:(?i:j\.d\.o\.o\.|d\.o\.o\.?|d\.d\.|k\.d\.|j\.t\.d\.)|" + _LEGAL_LONG + r"|GmbH(?:[ \t]*&[ \t]*Co\.?[ \t]*KG)?|AG|KG"
          r"|Ltd\.?|Limited|LLC|L\.L\.C\.|LLP|Inc\.?|Corp\.?|Corporation|S\.A\.|S\.p\.A\.|S\.r\.l\.|SRL|S\.L\."
          r"|B\.V\.|BV|N\.V\.|Oy|AB|ApS|A/S|plc|PLC|Kft\.?|s\.r\.o\.|Sp\.[ \t]*z[ \t]*o\.o\.|SAS|SARL"
          r"|Pty[ \t]+Ltd\.?|(?i:obrt))")
# A name's legal part, to its end — with what a registry adds after a legal
# form written out ("… odgovornošću za računalne i srodne aktivnosti").
LEGAL_SUFFIX_RE = re.compile(rf",?[ \t]+(?:{_LEGAL_LONG}(?:[ \t]+za[ \t]+[^\n,;()]{{1,100}})?|{_LEGAL})$")
_ORG_RE = re.compile(
    rf"(?<![\w])((?:{_ORG_WORD}[ \t]+(?:&[ \t]+)?){{0,4}}{_ORG_WORD}),?[ \t]+{_LEGAL}(?![\w])")
_ORG_LABEL_RE = re.compile(
    r"(?<![\w])(?i:company|employer|organi[sz]ation|firm|tvrtka|poslodavac|naziv[ \t]+tvrtke"
    rf"|naziv[ \t]+društva|društvo|firma)[ \t]*[:\-–][ \t]*"
    rf"((?:{_ORG_WORD}[ \t]+(?:&[ \t]+)?){{0,5}}{_ORG_WORD}(?:,?[ \t]+{_LEGAL})?)(?![\w])")

_STREET_HEAD = r"(?:Ulica|Ul\.|Trg|Cesta|Put|Avenija|Prilaz|Obala|Šetalište|Bulevar|Vijenac|Odvojak|Naselje)"
_STREET_TAIL = r"(?:ulica|cesta|trg|put|avenija|obala|prilaz|vijenac)"
_HOUSE = r"\d{1,4}[a-zA-Z]?(?:/\d{1,3})?"
_POSTAL_CITY = rf"(?:,?[ \t]*(?:[A-Z]{{1,2}}-)?\d{{4,5}}[ \t]+{NAME_WORD}(?:[ \t]+{NAME_WORD})?)"
_ADDRESS_RES = [
    re.compile(rf"(?<![\w]){_STREET_HEAD}(?:[ \t]+[{UP}{LO}.]+){{1,5}}?[ \t]+{_HOUSE}(?![\w]){_POSTAL_CITY}?"),
    re.compile(rf"(?<![\w]){NAME_WORD}(?:[ \t]+{NAME_WORD}){{0,3}}[ \t]+{_STREET_TAIL}[ \t]+{_HOUSE}(?![\w]){_POSTAL_CITY}?"),
    re.compile(rf"(?<![\w]){NAME_WORD}(?:[ \t]+[{UP}{LO}.]+){{0,3}}[ \t]+{_HOUSE},[ \t]*\d{{5}}[ \t]+{NAME_WORD}(?:[ \t]+{NAME_WORD})?"),
    re.compile(rf"(?<![\w])\d{{1,5}}[ \t]+(?:{NAME_WORD}[ \t]+){{1,3}}(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard"
               rf"|Blvd\.?|Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?|Way|Place|Pl\.?|Square|Sq\.?|Terrace|Close|Crescent"
               rf"|Highway|Hwy)(?![\w])(?:,?[ \t]+(?:Apt|Suite|Unit|#)\.?[ \t]*\w+)?{_POSTAL_CITY}?"),
    re.compile(rf"(?<![\w])(?:{NAME_WORD}[ \t]+)?[{UP}][{LO}]*(?:straße|strasse|weg|gasse|platz|allee|Straße|Strasse"
               rf"|Weg|Gasse|Platz|Allee)[ \t]+{_HOUSE}(?![\w]){_POSTAL_CITY}?"),
]
_ADDRESS_LABEL_RE = re.compile(
    r"(?<![\w])(?i:address|home[ \t]+address|street[ \t]+address|adresa|adresa[ \t]+stanovanja|prebivalište"
    r"|boravište|sjedište|ulica[ \t]+i[ \t]+kućni[ \t]+broj)[ \t]*[:\-–][ \t]*([^\n]{3,120})", re.M)

_DOB_RE = re.compile(
    r"(?<![\w])(?i:date[ \t]+of[ \t]+birth|birth[ \t]?date|d\.?o\.?b\.?|born(?:[ \t]+on)?|datum[ \t]+rođenja"
    r"|rođen[ao]?(?:[ \t]+je)?(?:[ \t]+dana)?|rođ\.|god(?:ina|\.)?[ \t]+rođenja)[ \t]*[:\-–]?[ \t]*"
    rf"({DATE})")

_ID_LABELS = (
    r"passport(?:[ \t]+(?:no\.?|number|#))?|br(?:oj|\.)?[ \t]+putovnice|putovnica|osobna[ \t]+iskaznica"
    r"|br(?:oj|\.)?[ \t]+osobne(?:[ \t]+iskaznice)?|id[ \t]+card(?:[ \t]+(?:no\.?|number))?"
    r"|identity[ \t]+card(?:[ \t]+(?:no\.?|number))?|personal[ \t]+id(?:[ \t]+number)?"
    r"|national[ \t]+id(?:[ \t]+number)?|driver'?s[ \t]+licen[cs]e(?:[ \t]+(?:no\.?|number))?"
    r"|vozačka[ \t]+dozvola|br(?:oj|\.)?[ \t]+vozačke(?:[ \t]+dozvole)?|tax[ \t]+id|tin|vat(?:[ \t]+(?:no\.?|number|id))?"
    r"|pdv(?:[ \t]+(?:id|broj))?|mbs|matični[ \t]+broj|social[ \t]+security(?:[ \t]+(?:no\.?|number))?"
    r"|nino?|policy[ \t]+(?:no\.?|number)|broj[ \t]+police|member(?:ship)?[ \t]+(?:no\.?|number)"
    r"|patient[ \t]+id|broj[ \t]+kartona|mbo|customer[ \t]+(?:no\.?|number|id)|broj[ \t]+kupca"
)
_ID_LABEL_RE = re.compile(
    rf"(?<![\w])(?i:{_ID_LABELS})[ \t]*[:#\-–]?[ \t]*([A-Z0-9][A-Z0-9\-/]{{2,24}}(?:[ ][A-Z0-9\-/]{{1,12}}){{0,3}})(?![\w])")


# ── detection ───────────────────────────────────────────────────────


def _phone_digits_ok(text: str, low: int, high: int) -> bool:
    digits = re.sub(r"\D", "", text)
    return low <= len(digits) <= high


def _looks_like_date(text: str) -> bool:
    return bool(re.fullmatch(r"\(?\d{1,4}[./\-][ \t]?\d{1,2}[./\-][ \t]?\d{2,4}\.?\)?", text.strip()))


_CURRENCY = r"(?:€|EUR|eura?|USD|\$|£|GBP|CHF|HRK|kn|kuna)"
_NUMBER = r"\d{1,3}(?:[.,\u00a0 ]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"
_AMOUNT_RE = re.compile(
    rf"(?<![\w.,])(?:{_CURRENCY}[ \u00a0]?(?:{_NUMBER})|(?:{_NUMBER})[ \u00a0]?{_CURRENCY})(?![\w])", re.I)
_AGE_RE = re.compile(
    r"(?<![\w.,])(\d{1,3})[ \t]*(?:godin[aeu]?|god\.|years?[ \t]+old|yrs?\b|y/o)(?![\w])"
    r"|(?i:\bage[ \t]*[:=]?|\bdob[ \t]*[:=]|\bstar[aoi][ \t]+)[ \t]*(\d{1,3})(?![\d.,])", re.I)


def detect(text: str, kinds: frozenset[str] | set[str] | None = None,
           opt_in: frozenset[str] | set[str] = frozenset(), *, ner: bool = True) -> list[Span]:
    """Every span the deterministic lane finds in *text* (they may overlap).
    *opt_in*: kinds of :data:`OPT_IN_KINDS` to find too. *ner*: ask the
    name-recognition pack too, if one is on (the engine asks it only on the
    original text, not on its own masked output)."""
    text = str(text or "")
    out: list[Span] = []

    def want(kind: str) -> bool:
        return kinds is None or kind in kinds

    def add(start: int, end: int, kind: str, priority: int, source: str) -> None:
        if end > start and want(kind):
            out.append(Span(start, end, kind, priority, source))

    def add_name(start: int, end: int, priority: int, source: str) -> None:
        trimmed = trim_name(text, start, end)
        if trimmed:
            add(trimmed[0], trimmed[1], "PERSON", priority, source)

    # validated
    if want("IBAN"):
        for m in _IBAN_RE.finditer(text):
            raw = m.group(1)
            # The run may carry a word after the IBAN: try the longest prefix that validates.
            ends = [g.end() for g in re.finditer(r"[A-Z0-9]+", raw)]
            for end in reversed(ends):
                if iban_ok(raw[:end]):
                    add(m.start(1), m.start(1) + end, "IBAN", 3, "iban")
                    break
    if want("CARD") or want("CVV") or want("CARD_EXPIRY"):
        from captain_claw import card_scrub

        # Card numbers and expiry dates as card_scrub finds them; CVVs only
        # where labelled — card_scrub's "a lone 3–4 digit number on the card's
        # line" is right for a chat message, too eager for a document.
        for f in card_scrub.find(text):
            if f.kind == "pan":
                add(f.start, f.end, "CARD", 3, "card")
            elif f.kind == "expiry":
                add(f.start, f.end, "CARD_EXPIRY", 2, "card")
        for m in card_scrub._CVV_LABEL_RE.finditer(text):
            add(m.start(1), m.end(1), "CVV", 2, "card")
    for m in _OIB_RE.finditer(text):
        if oib_ok(m.group(0)):
            add(m.start(), m.end(), "OIB", 3, "oib")
    for m in _OIB_LABEL_RE.finditer(text):
        add(m.start(1), m.end(1), "OIB", 3, "oib-label")
    for m in _JMBG_RE.finditer(text):
        if jmbg_ok(m.group(0)):
            add(m.start(), m.end(), "JMBG", 3, "jmbg")
    for m in _JMBG_LABEL_RE.finditer(text):
        add(m.start(1), m.end(1), "JMBG", 3, "jmbg-label")
    for m in _MB_LABEL_RE.finditer(text):
        add(m.start(1), m.end(1), "ID_NUMBER", 3, "mb-label")

    # structural
    for m in _EMAIL_RE.finditer(text):
        add(m.start(), m.end(), "EMAIL", 2, "email")
    for m in _PHONE_INTL_RE.finditer(text):
        s, e = _trim(text, m.start(), m.end(), " \t.-/(")
        if _phone_digits_ok(text[s:e], 8, 15):
            add(s, e, "PHONE", 2, "phone-intl")
    for m in _PHONE_LABEL_RE.finditer(text):
        s, e = _trim(text, m.start(1), m.end(1), " \t.-/")
        if _phone_digits_ok(text[s:e], 6, 15) and not _looks_like_date(text[s:e]):
            add(s, e, "PHONE", 2, "phone-label")
    for m in _PHONE_NAT_RE.finditer(text):
        s, e = m.start(), m.end()
        cand = text[s:e]
        if _looks_like_date(cand):
            continue
        digits = re.sub(r"\D", "", cand)
        separated = bool(re.search(r"[ \t.\-/()]", cand))
        if 8 <= len(digits) <= 11 and (separated or (9 <= len(digits) <= 10 and digits[1] != "0")):
            add(s, e, "PHONE", 0, "phone-national")
    for m in _SSN_RE.finditer(text):
        add(m.start(), m.end(), "SSN", 2, "ssn")
    for m in _IPV4_RE.finditer(text):
        before = text[max(0, m.start() - 10):m.start()].casefold()
        if m.group(0) in _IP_SKIP or re.search(r"(?:v|version|ver\.?)\s*$", before):
            continue
        add(m.start(), m.end(), "IP", 2, "ipv4")
    for m in _IPV6_RE.finditer(text):
        val = m.group(0)
        if "::" in val or val.count(":") >= 4 or re.search(r"[A-Fa-f]", val):
            if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", val):
                continue
            add(m.start(), m.end(), "IP", 2, "ipv6")
    for m in _PROFILE_RE.finditer(text):
        if m.group(1).casefold().rstrip("/") in _PROFILE_SKIP:
            continue
        s, e = _trim(text, m.start(), m.end(), " \t.,;:)/")
        add(s, e, "PROFILE_URL", 2, "profile-url")
    for m in _PLATE_RE.finditer(text):
        add(m.start(), m.end(), "PLATE", 1, "plate")
    for m in _DOB_RE.finditer(text):
        s, e = _trim(text, m.start(1), m.end(1), " \t,;")
        add(s, e, "DOB", 2, "dob-label")
    for m in _ID_LABEL_RE.finditer(text):
        s, e = _trim(text, m.start(1), m.end(1), " \t.,;:-/")
        if re.search(r"\d", text[s:e]) and e - s >= 4:
            add(s, e, "ID_NUMBER", 2, "id-label")
    for m in _ADDRESS_LABEL_RE.finditer(text):
        s, e = m.start(1), m.end(1)
        # The rest of the line, up to the first sentence end after the house
        # number ("Ilica 12, 10000 Zagreb. Plaća …" stops at "Zagreb";
        # "Trg b. J. Jelačića 1" keeps its initials), or a ";" / " | ".
        value = text[s:e]
        cut = len(value)
        for c in re.finditer(r"(?<=[\w)])\.\s+(?=[A-ZČĆŠŽĐ])|;|\s\|\s", value):
            sentence_end = c.group(0).startswith(".")
            if not sentence_end or re.search(r"\d", value[:c.start()]):
                cut = c.start()
                break
        s, e = _trim(text, s, s + cut, " \t.,;:")
        if re.search(rf"[{UP}{LO}]", text[s:e]):
            add(s, e, "ADDRESS", 2, "address-label")
    for m in _PERSON_LABEL_RE.finditer(text):
        add_name(m.start(1), m.end(1), 2, "person-label")
    for m in _DISPLAY_NAME_RE.finditer(text):
        add_name(m.start(1), m.end(1), 2, "display-name")
    for m in _ORG_LABEL_RE.finditer(text):
        s, e = _trim(text, m.start(1), m.end(1), " \t,;:")
        if re.match(rf"[{UP}0-9]", text[s:e] or "x"):
            add(s, e, "ORG", 2, "org-label")

    # context
    for m in _HONORIFIC_RE.finditer(text):
        add_name(m.start(1), m.end(1), 1, "honorific")
    for m in _GREETING_RE.finditer(text):
        add_name(m.start(1), m.end(1), 1, "greeting")
    for m in _SIGNOFF_RE.finditer(text):
        add_name(m.start(1), m.end(1), 1, "signoff")
    for m in _ORG_RE.finditer(text):
        start = m.start(1)
        # Leading words that aren't part of a company's name ("Yesterday Acme d.o.o.").
        for w in re.finditer(r"\S+", text[m.start(1):m.end(1)]):
            if fold(w.group(0)) in _STOP:
                start = m.start(1) + w.end()
                continue
            break
        s, e = _trim(text, start, m.end(), " \t,")
        if s < e and not LEGAL_SUFFIX_RE.fullmatch(" " + text[s:e]):
            add(s, e, "ORG", 1, "legal-suffix")
    for rx in _ADDRESS_RES:
        for m in rx.finditer(text):
            s, e = _trim(text, m.start(), m.end(), " \t.,;:")
            add(s, e, "ADDRESS", 1, "street")

    if want("PERSON"):
        out.extend(_names_from_emails(text, out))
    if "AMOUNT" in opt_in:
        for m in _AMOUNT_RE.finditer(text):
            s, e = _trim(text, m.start(), m.end(), " \t\u00a0")
            out.append(Span(s, e, "AMOUNT", 1, "amount"))
    if "AGE" in opt_in:
        for m in _AGE_RE.finditer(text):
            g = 1 if m.group(1) else 2
            if 0 < int(m.group(g)) < 125:
                out.append(Span(m.start(), m.end(), "AGE", 1, "age"))
    if ner and _ner_backend is not None and (want("PERSON") or want("ORG")):
        out.extend(sp for sp in ner_spans(text) if want(sp.kind))
    return out


# ── an optional local name-recognition model (the NER pack) ────────

# Set by whoever runs the engine (Flight Deck) when an admin has turned the
# pack on; None everywhere else, so the lane stays pure code by default.
_ner_backend: "Callable[[str], list[tuple[int, int, str, float]]] | None" = None
# Measured (scripts/measure_name_models.py): below 0.80 BERTić finds more
# names and breaks nothing more; 0.70 keeps a margin.
NER_MIN_SCORE = 0.70
NER_MAX_CHARS = 200_000
_NER_KINDS = {"PER": "PERSON", "PERSON": "PERSON", "B-PER": "PERSON", "I-PER": "PERSON", "ORG": "ORG"}


def set_ner_backend(backend: "Callable[[str], list[tuple[int, int, str, float]]] | None") -> None:
    """*backend(text)* → ``[(start, end, label, score)]`` (labels like PER /
    ORG / LOC); None turns the pack off."""
    global _ner_backend
    _ner_backend = backend


def ner_spans(text: str) -> list[Span]:
    """People and companies the model finds (places and the rest are left: a
    city on its own isn't private). Never raises: a broken model finds nothing."""
    backend = _ner_backend
    if backend is None or not text.strip() or len(text) > NER_MAX_CHARS:
        return []
    try:
        # Tokens already in the text are masked with NULs; a model reads
        # spaces better (same length, so the offsets hold).
        found = backend(text.replace("\x00", " "))
    except Exception:  # noqa: BLE001 — the lane keeps working without it
        return []
    out = []
    for start, end, label, score in found:
        kind = _NER_KINDS.get(str(label).upper())
        if kind is None or float(score) < NER_MIN_SCORE or not (0 <= start < end <= len(text)):
            continue
        s, e = _trim(text, start, end, " \t\r\n,;:.!?\"'()[]")
        if kind == "PERSON":
            trimmed = trim_name(text, s, e)
            if not trimmed:
                continue
            s, e = trimmed
        if e - s >= 2:
            out.append(Span(s, e, kind, 1, "ner"))
    return out


def _names_from_emails(text: str, spans: list[Span]) -> list[Span]:
    """``ivan.horvat@…`` spells a name: where those words stand capitalized in
    the text ("Ivan Horvat"), that's a person."""
    wanted: set[tuple[str, ...]] = set()
    for s in spans:
        if s.kind != "EMAIL":
            continue
        local = text[s.start:s.end].split("@", 1)[0]
        parts = [p for p in re.split(r"[._\-]+", local) if p]
        if 2 <= len(parts) <= 3 and all(p.isalpha() and len(p) >= 2 for p in parts):
            wanted.add(tuple(name_stem(strip_accents(p)) for p in parts))
    if not wanted:
        return []
    found: list[Span] = []
    for m in re.finditer(name_seq(2, 3), text):
        words = tuple(name_stem(re.sub(r"[^a-z]", "", strip_accents(w).casefold()))
                      for w in re.findall(rf"{NAME_WORD}|{CAPS_WORD}", m.group(0)))
        if words in wanted or tuple(reversed(words)) in wanted:
            found.append(Span(m.start(), m.end(), "PERSON", 2, "email-name"))
    return found


# ── column / key hints (tables, JSON) ───────────────────────────────

_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"^(?:e_?mail|mail|email_address|e_?posta|e_?pošta)(?:_\d+)?$")),
    ("PHONE", re.compile(r"^(?:phone|tel|telephone|mobile|mob|cell|fax|gsm|telefon|mobitel|phone_number)(?:_\d+)?$")),
    ("ADDRESS", re.compile(r"^(?:address|street|addr|adresa|ulica|prebivaliste|prebivalište|boraviste|boravište"
                           r"|address_line_?\d?|street_address|home_address)$")),
    ("DOB", re.compile(r"^(?:dob|date_of_birth|birth_?date|birthday|born|datum_rodenja|datum_rođenja|datum_rodjenja"
                       r"|rodendan|rođendan)$")),
    ("IBAN", re.compile(r"^(?:iban|bank_account|account_number|racun|račun|broj_racuna|broj_računa)$")),
    ("OIB", re.compile(r"^oib$")),
    ("JMBG", re.compile(r"^jmbg$")),
    ("ORG", re.compile(r"^(?:company|company_name|organization|organisation|employer|firm|tvrtka|poslodavac|firma"
                       r"|drustvo|društvo|naziv_tvrtke)$")),
    ("ID_NUMBER", re.compile(r"^(?:passport|passport_number|id_number|national_id|ssn|tax_id|vat|vat_number|pdv"
                             r"|mbs|licen[cs]e|licen[cs]e_number|broj_osobne|broj_putovnice)$")),
    ("IP", re.compile(r"^(?:ip|ip_address|ipv4|ipv6|client_ip|remote_addr)$")),
    ("PERSON", re.compile(r"^(?:full_?name|first_?name|last_?name|surname|given_?name|family_?name|middle_?name"
                          r"|contact|contact_name|contact_person|client_name|customer_name|patient|patient_name"
                          r"|employee|employee_name|owner|owner_name|author|ime|prezime|ime_i_prezime|kontakt"
                          r"|kontakt_osoba|klijent|kupac|pacijent|zaposlenik|vlasnik|autor|korisnik|person"
                          r"|signatory|holder|account_holder)$")),
    ("PERSON?", re.compile(r"^(?:name|naziv)$")),  # only when the value looks like a name
]


def hint_for(key: str) -> str:
    """The kind a column header / JSON key names ("" when none)."""
    k = re.sub(r"[^0-9a-zčćđšž]+", "_", str(key or "").casefold()).strip("_")
    for kind, rx in _HINTS:
        if rx.match(k):
            return kind
    return ""


_NAME_VALUE_RE = re.compile(rf"^{name_seq(1, 4)}$")


def hint_span(text: str, hint: str) -> Span | None:
    """The whole value as the kind its column / key names, when it fits."""
    if not hint:
        return None
    s, e = _trim(text, 0, len(text), " \t\r\n")
    value = text[s:e]
    if not value or len(value) > 200:
        return None
    if hint in ("PERSON", "PERSON?"):
        if not _NAME_VALUE_RE.match(value) or (hint == "PERSON?" and " " not in value):
            return None
        trimmed = trim_name(text, s, e)
        return Span(trimmed[0], trimmed[1], "PERSON", 2, "hint") if trimmed else None
    if hint == "DOB":
        return Span(s, e, "DOB", 2, "hint") if parse_date(value) else None
    if hint in ("PHONE", "IBAN", "OIB", "JMBG", "ID_NUMBER"):
        return Span(s, e, hint, 2, "hint") if len(re.sub(r"\D", "", value)) >= 4 else None
    if hint in ("EMAIL", "IP"):
        return None  # their own patterns are exact
    return Span(s, e, hint, 2, "hint")
