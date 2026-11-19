"""The list funnel's code-only parts: reading what a list comes from, and picking
facts out of text (docs/funnel-runner-plan.md).

Nothing here calls a model or the network. The funnel gathers text (a search's
results, a page, a row of a file, a part of a document) and these pull out what
code can find for free: emails (on a given site), phones, dates, a company-size
band, a US state, links. A fact found here goes to the model with the page it was
on, and a value the model writes is checked against the text it came from
(:func:`in_text`).
"""

from __future__ import annotations

import csv
import json
import re
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# ── reading a list's sources ───────────────────────────────────────────

TABLE_EXTS = (".csv", ".tsv", ".xlsx", ".jsonl")
DOC_EXTS = (".pdf", ".docx", ".pptx", ".md", ".markdown", ".txt", ".html", ".htm", ".rst")
TABLE_ROWS_MAX = 50_000
DOC_CHARS_MAX = 2_000_000


def read_table(path: Path, limit: int = TABLE_ROWS_MAX) -> tuple[list[str], list[dict[str, str]]]:
    """A table file's columns and rows (each a dict of strings). CSV/TSV, the
    first sheet of an XLSX, or JSON lines. Raises ValueError when it can't be read."""
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t" if ext == ".tsv" else ",")
            rows = []
            for i, r in enumerate(reader):
                if i >= limit:
                    break
                rows.append({str(k).strip(): str(v or "").strip() for k, v in r.items() if k})
            return [str(c).strip() for c in (reader.fieldnames or []) if c], rows
    if ext == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb.worksheets[0]
            it = ws.iter_rows(values_only=True)
            head = next(it, None)
            if not head:
                return [], []
            cols = [str(c).strip() if c is not None else f"column {i + 1}" for i, c in enumerate(head)]
            rows = []
            for i, values in enumerate(it):
                if i >= limit:
                    break
                if not any(v not in (None, "") for v in values):
                    continue
                rows.append({cols[j]: _cell(v) for j, v in enumerate(values[:len(cols)])})
            return cols, rows
        finally:
            wb.close()
    if ext == ".jsonl":
        rows, cols = [], []
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
            if i >= limit:
                break
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                row = {str(k): _cell(v) for k, v in obj.items()}
                cols += [k for k in row if k not in cols]
                rows.append(row)
        return cols, rows
    raise ValueError(f"{path.name} isn't a table (csv, tsv, xlsx or jsonl)")


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (date,)):
        return v.isoformat()
    return str(v).strip()


def read_document(path: Path) -> str:
    """A document's text (PDF, Word, PowerPoint, Markdown, plain text, HTML).
    Raises ValueError when there's none to read."""
    ext = path.suffix.lower()
    if ext in (".md", ".markdown", ".txt", ".rst", ".html", ".htm"):
        text = path.read_text(encoding="utf-8", errors="replace")[:DOC_CHARS_MAX]
        if ext in (".html", ".htm"):
            text = re.sub(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", " ", text, flags=re.S | re.I)
        return text
    from captain_claw.tools import document_extract as dx
    if ext == ".pdf":
        text, err = dx._extract_pdf_markdown(path, 400)
        if err:
            raise ValueError(err)
        text = text or ""
    elif ext == ".docx":
        text = dx._extract_docx_markdown(path)
    elif ext == ".pptx":
        text = dx._extract_pptx_markdown(path, 300)
    else:
        raise ValueError(f"{path.name} isn't a document this reads")
    if not text.strip() or ("No extractable text found" in text and len(text) < 300):
        raise ValueError(f"no text could be read from {path.name} (a scan or an image-only file)")
    return text[:DOC_CHARS_MAX]


def chunks(text: str, size: int = 6000, overlap: int = 300) -> list[str]:
    """Text cut into parts of about ``size`` characters, at a paragraph or line
    break when there is one near the end, each overlapping the last a little."""
    text = text or ""
    if len(text) <= size:
        return [text] if text.strip() else []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind("\n\n", start + size // 2, end), text.rfind("\n", start + size // 2, end))
            end = cut if cut > start else end
        part = text[start:end]
        if part.strip():
            out.append(part)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


LINK_LINE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)(?::\s*(.*))?")


def read_links(text: str) -> list[dict[str, str]]:
    """The links in a links file the owner added as extra data ("1. [title](url): note"),
    or any bare URLs in a text."""
    out, seen = [], set()
    for m in LINK_LINE.finditer(text or ""):
        url = m.group(2).strip()
        if url not in seen:
            seen.add(url)
            out.append({"title": m.group(1).strip(), "url": url, "note": (m.group(3) or "").strip()})
    for url in links(text or ""):
        if url not in seen:
            seen.add(url)
            out.append({"title": "", "url": url, "note": ""})
    return out


# ── search results ─────────────────────────────────────────────────────

_RESULT = re.compile(r"^\s*\d+\.\s+(?P<title>.*)\n\s+URL:\s*(?P<url>\S+)(?P<rest>(?:\n\s{3,}.*)*)", re.M)


def search_results(text: str) -> list[dict[str, str]]:
    """web_search's results as ``{title, url, snippet, text}``."""
    out = []
    for m in _RESULT.finditer(text or ""):
        rest = m.group("rest") or ""
        snip = re.search(r"Snippet:\s*(.*)", rest)
        out.append({"title": m.group("title").strip(), "url": m.group("url").strip(),
                    "snippet": (snip.group(1).strip() if snip else "").removesuffix("-").strip(),
                    "text": f"{m.group('title').strip()}\n{rest.strip()}"})
    return out


# ── facts in text ──────────────────────────────────────────────────────

def domain_of(url: str) -> str:
    """``example.com`` from ``https://www.example.com/about``; "" when it isn't a link."""
    s = str(url or "").strip()
    if not s:
        return ""
    if "://" not in s:
        s = "http://" + s
    try:
        host = (urlparse(s).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def same_site(host: str, domain: str) -> bool:
    host, domain = host.lower().strip("."), domain.lower().strip(".")
    return bool(domain) and (host == domain or host.endswith("." + domain))


_EMAIL = re.compile(r"(?<![\w.+-])([A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]+\.[A-Za-z]{2,24})(?![\w-])")
_NOT_EMAIL_TLD = ("png", "jpg", "jpeg", "gif", "svg", "webp", "css", "js")


def emails(text: str, domain: str = "") -> list[str]:
    """The email addresses in a text, in order, each once; only those on ``domain``
    (or its subdomains) when one is given."""
    out = []
    for m in _EMAIL.finditer(text or ""):
        e = m.group(1).strip(".").lower()
        if e.rsplit(".", 1)[-1] in _NOT_EMAIL_TLD or e in out:
            continue
        if domain and not same_site(e.split("@", 1)[1], domain):
            continue
        out.append(e)
    return out


SHARED_LOCAL = re.compile(r"^(info|hello|hi|contact|contacts|office|admin|team|sales|support|enquiries|"
                          r"inquiries|mail|general|help|careers|jobs|press|media|marketing|partners)\b")


def email_kind(email: str) -> str:
    """``shared`` for a role inbox (info@, hello@, …), else ``personal``."""
    return "shared" if SHARED_LOCAL.match(email.split("@", 1)[0].lower()) else "personal"


_PHONE = re.compile(r"(?<!\d)(\+?\d[\d ()./-]{7,18}\d)(?!\d)")


def phones(text: str) -> list[str]:
    out = []
    for m in _PHONE.finditer(text or ""):
        raw = m.group(1)
        digits = re.sub(r"\D", "", raw)
        if ". " in raw or re.search(r"\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2}-\d{2}-\d{2}", raw):
            continue  # dates, or a sentence's end running into a number
        if 9 <= len(digits) <= 15 and raw.strip() not in out:
            out.append(raw.strip())
    return out


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_FORMS = [
    (re.compile(r"\b((?:19|20)\d{2})-(0[1-9]|1[0-2])(?:-(0[1-9]|[12]\d|3[01]))?\b"), "ymd"),
    (re.compile(r"\b" + _MON + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{2})\b", re.I), "mdy"),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MON + r"\.?,?\s+((?:19|20)\d{2})\b", re.I), "dmy"),
    (re.compile(r"\b" + _MON + r"\.?,?\s+((?:19|20)\d{2})\b", re.I), "my"),
    (re.compile(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/((?:19|20)\d{2})\b"), "us"),
]


def dates(text: str) -> list[tuple[str, str]]:
    """The dates in a text as ``(iso, the words around it)``: ``2025-03-14``, or
    ``2025-03`` when there's no day. Each date once, in order."""
    found: list[tuple[int, str, str]] = []
    taken: list[tuple[int, int]] = []
    t = text or ""
    for rx, form in _DATE_FORMS:
        for m in rx.finditer(t):
            if any(a <= m.start() < b for a, b in taken):
                continue
            g = m.groups()
            try:
                if form == "ymd":
                    iso = f"{g[0]}-{g[1]}" + (f"-{g[2]}" if g[2] else "")
                elif form == "mdy":
                    iso = f"{g[2]}-{_MONTHS[g[0][:3].lower()]:02d}-{int(g[1]):02d}"
                elif form == "dmy":
                    iso = f"{g[2]}-{_MONTHS[g[1][:3].lower()]:02d}-{int(g[0]):02d}"
                elif form == "my":
                    iso = f"{g[1]}-{_MONTHS[g[0][:3].lower()]:02d}"
                else:
                    iso = f"{g[2]}-{int(g[0]):02d}-{int(g[1]):02d}"
                if len(iso) == 10:
                    date.fromisoformat(iso)
            except (KeyError, ValueError):
                continue
            taken.append((m.start(), m.end()))
            around = " ".join(t[max(0, m.start() - 80):m.end() + 80].split())
            found.append((m.start(), iso, around))
    out, seen = [], set()
    for _, iso, around in sorted(found):
        if iso not in seen:
            seen.add(iso)
            out.append((iso, around))
    return out


def months_ago(iso: str, today: date | None = None) -> float | None:
    """How many months before ``today`` an ISO date (or year-month) is; None when it
    isn't one."""
    m = re.fullmatch(r"((?:19|20)\d{2})-(0[1-9]|1[0-2])(?:-(\d{2}))?", str(iso or "").strip())
    if not m:
        return None
    today = today or date.today()
    y, mo = int(m.group(1)), int(m.group(2))
    d = int(m.group(3) or 15)
    return (today.year - y) * 12 + (today.month - mo) + (today.day - d) / 30.0


_BAND = re.compile(r"\b(\d{1,3}(?:,\d{3})*)\s*(?:-|–|to)\s*(\d{1,3}(?:,\d{3})*)\s+(?:employees|people|staff)\b"
                   r"|\b(\d{1,3}(?:,\d{3})*)\+\s*(?:employees|people|staff)\b", re.I)


def size_band(text: str) -> tuple[int, int] | None:
    """A company-size band ("11-50 employees", "Company size 2-10 employees",
    "10,001+ employees") as ``(low, high)``; high is 10**9 for "N+"."""
    m = _BAND.search(text or "")
    if not m:
        return None
    if m.group(3):
        return int(m.group(3).replace(",", "")), 10**9
    return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))


US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}
_STATE_ABBR = re.compile(r",\s*(" + "|".join(US_STATES) + r")\.?\s+\d{5}(?:-\d{4})?\b")
_STATE_NAME = re.compile(r"\b(" + "|".join(sorted(US_STATES.values(), key=len, reverse=True)) + r")\b")


def us_state(text: str) -> str:
    """The two-letter US state of the first address in a text (", TX 78701"), else
    of the first state named; "" when none."""
    m = _STATE_ABBR.search(text or "")
    if m:
        return m.group(1)
    m = _STATE_NAME.search(text or "")
    if m:
        return next(k for k, v in US_STATES.items() if v == m.group(1))
    return ""


_LINK = re.compile(r"https?://[^\s<>()\"'\]]+")


def links(text: str) -> list[str]:
    out = []
    for m in _LINK.finditer(text or ""):
        url = m.group(0).rstrip(".,;:!?)")
        if url not in out:
            out.append(url)
    return out


EXTRACTORS = ("emails", "phones", "dates", "size_band", "us_state", "links")


def extract(name: str, text: str, *, domain: str = "") -> list[str]:
    """One extractor's finds in a text, as strings (a size band as "11-50")."""
    if name == "emails":
        return emails(text, domain)
    if name == "phones":
        return phones(text)
    if name == "dates":
        return [f"{iso} ({around})" for iso, around in dates(text)[:12]]
    if name == "size_band":
        band = size_band(text)
        return [f"{band[0]}+" if band and band[1] >= 10**9 else f"{band[0]}-{band[1]}"] if band else []
    if name == "us_state":
        s = us_state(text)
        return [s] if s else []
    if name == "links":
        return links(text)[:30]
    raise ValueError(f"no extractor {name!r}")


# ── checking a value against its source ───────────────────────────────

def _norm(s: str) -> str:
    return " ".join(re.sub(r"[‘’“”]", "'", str(s or "")).lower().split())


def in_text(value: str, text: str) -> bool:
    """Does the text hold the value: case and spacing aside, and for a long value
    (over 60 characters, a sentence written about the source) its first 40
    characters, since a model may reword the end."""
    v, t = _norm(value), _norm(text)
    if not v:
        return False
    if v in t:
        return True
    if "@" in v:  # an email may sit in a mailto: link, percent-encoded or not
        return v.replace("@", "%40") in t
    return len(v) > 60 and v[:40] in t


def parse_json(text: str) -> Any:
    """A model's JSON answer: fenced or not, with thinking or prose around it.
    Returns None when there's none."""
    t = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    pairs = sorted((("[", "]"), ("{", "}")), key=lambda p: t.find(p[0]) if p[0] in t else len(t))
    for opener, closer in pairs:  # whichever comes first is the answer's outside
        start = t.find(opener)
        end = t.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(t[start:end + 1])
            except ValueError:
                continue
    return None
