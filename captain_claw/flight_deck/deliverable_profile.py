"""What a run actually delivered, read by Captain itself.

A long run's result can be a list, a document, a report, code, data, or several of
these. The reviewers used to see a file listing, a few excerpts and the workers'
own account of what they made; a list of 500 "leads" that was 40% university
staff and 43% info@ inboxes passed that way. This module reads the delivered
files and says what's in them, so the judge (and the owner) get facts, not claims.

* :func:`sort_files` puts a run's files in three groups: **main** (what its checks
  name, else a file called a result, else the newest at the top of its folder),
  **other** (the rest at the top) and **work** (subfolders, scripts, temp files).
* :func:`profile_file` reads one file by its kind: a **table** (rows, columns,
  broken rows, duplicates, empty cells; for an email column the share of role
  inboxes, free mail and the domains it leans on; for a link column the share
  that are links), a **document** (words, headings, links, placeholders left,
  repeated paragraphs), **code** (files, lines, tests, syntax errors, TODOs),
  **JSON** (valid, how many records).
* :func:`flags` turns a profile into the few things worth a look, in words;
  :func:`render` is the block the reviewers read.

Nothing here runs anything it reads, or reaches the network.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

MAX_MAIN_PROFILED = 6
TABLE_MAX_BYTES = 30_000_000
DOC_MAX_BYTES = 5_000_000
CODE_MAX_FILES = 400
SAMPLE_ROWS = 6

_MAIN_NAME = re.compile(r"(^|[\W_])(final|result|results|report|output|deliverable|summary|merged|master|complete)"
                        r"($|[\W_])", re.IGNORECASE)
_WORK_NAME = re.compile(r"^(tmp|temp|scratch|test|debug)([\W_]|$)|[\W_](tmp|temp|scratch)$", re.IGNORECASE)
_WORK_EXTS = {".pyc", ".log", ".bak", ".swp"}
_CODE_EXTS = {".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
              ".jsx": "JavaScript", ".go": "Go", ".rs": "Rust", ".java": "Java", ".kt": "Kotlin", ".rb": "Ruby",
              ".php": "PHP", ".cs": "C#", ".swift": "Swift", ".c": "C", ".cpp": "C++", ".h": "C", ".sh": "Shell",
              ".sql": "SQL", ".html": "HTML", ".css": "CSS", ".vue": "Vue", ".svelte": "Svelte"}
_SCRIPT_EXTS = {".py", ".sh", ".js", ".mjs", ".ts", ".ipynb"}
_TABLE_EXTS = {".csv": ",", ".tsv": "\t"}
_DOC_EXTS = {".md", ".markdown", ".txt", ".html", ".htm", ".rst"}
_PROJECT_MARKERS = ("package.json", "pyproject.toml", "setup.py", "requirements.txt", "go.mod", "Cargo.toml",
                    "pom.xml", "build.gradle", "Gemfile", "composer.json", "index.html", "Makefile")

ROLE_LOCAL = re.compile(
    r"^(info|hello|hi|hey|contact|contactus|support|help|press|media|pr|team|partnerships?|partner|sales|admin|"
    r"enquir(y|ies)|inquir(y|ies)|office|business|biz|bookings?|speaking|events?|marketing|newsletter|editor|"
    r"editorial|tips|news|general|mail|sponsor(ship)?s?|ads|advertising|collab|work|careers|jobs|feedback|"
    r"community|management|mgmt|noreply|no-reply)([._+-].*)?$", re.IGNORECASE)
FREE_MAIL = {"gmail.com", "googlemail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "proton.me",
             "protonmail.com", "aol.com", "live.com", "me.com", "hotmail.co.uk", "yahoo.co.uk", "btinternet.com",
             "gmx.com", "gmx.de", "mail.com", "yandex.com"}
_EMAIL = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_LINK = re.compile(r"^https?://\S+$", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME|XXX|lorem ipsum)\b|\[(insert|add|your|placeholder)[^\]]*\]|\?\?\?",
                          re.IGNORECASE)
_TODO_CODE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
_MD_LINK = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)|(?<![\w(])https?://[^\s)>\]]+", re.IGNORECASE)


# ── which files are the result ─────────────────────────────────────────

def _kind_of(rel: str, named: set[str]) -> str:
    if rel in named:
        return "main"
    name = Path(rel).name
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    if "/" in rel or ext in _WORK_EXTS or ext in _SCRIPT_EXTS or _WORK_NAME.search(stem):
        return "work"
    return "main" if _MAIN_NAME.search(stem) else "other"


def is_code_project(found: list[tuple[float, str, Path, int]]) -> bool:
    """A folder whose result is code: a project marker at the top, or mostly source files."""
    rels = [rel for _, rel, _, _ in found]
    if any(rel in _PROJECT_MARKERS for rel in rels):
        return True
    code = sum(1 for rel in rels if Path(rel).suffix.lower() in _CODE_EXTS)
    return code >= 5 and code >= len(rels) * 0.5


def sort_files(found: list[tuple[float, str, Path, int]], named: set[str]) -> dict[str, list[tuple[float, str, Path, int]]]:
    """``found`` (newest first, from ``bat_routes._run_files``) into main / other / work.
    A code project is one result: its files are all "main"."""
    groups: dict[str, list[tuple[float, str, Path, int]]] = {"main": [], "other": [], "work": []}
    if is_code_project(found):
        groups["main"] = list(found)
        return groups
    for item in found:
        groups[_kind_of(item[1], named)].append(item)
    groups["main"].sort(key=lambda f: f[1] not in named)  # what the checks name leads
    if not groups["main"] and groups["other"]:
        groups["main"].append(groups["other"].pop(0))  # the newest at the top
    return groups


# ── reading one file ───────────────────────────────────────────────────

def _domain(email: str) -> str:
    return email.rsplit("@", 1)[-1].strip().lower()


def _base(host: str) -> str:
    parts = [p for p in host.lower().split(".") if p]
    if len(parts) > 2 and parts[-2] in ("co", "org", "ac", "com", "gov", "net") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _column(header: list[str], *names: str) -> int:
    low = [h.strip().lower() for h in header]
    for n in names:
        for i, h in enumerate(low):
            if h == n or h.replace("_", " ") == n:
                return i
    for n in names:
        for i, h in enumerate(low):
            if n in h:
                return i
    return -1


def profile_table(path: Path) -> dict[str, Any]:
    sep = _TABLE_EXTS.get(path.suffix.lower(), ",")
    if path.stat().st_size > TABLE_MAX_BYTES:
        return {"kind": "table", "too_big": True}
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        rows = [r for r in csv.reader(fh, delimiter=sep)]
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        return {"kind": "table", "rows": 0, "columns": []}
    header, body = [h.strip() for h in rows[0]], rows[1:]
    width = len(header)
    broken = sum(1 for r in body if len(r) != width)
    norm = [tuple(c.strip() for c in r) for r in body]
    dup_rows = len(norm) - len(set(norm))
    empties = {header[i]: sum(1 for r in body if i >= len(r) or not r[i].strip()) for i in range(width)}
    out: dict[str, Any] = {"kind": "table", "rows": len(body), "columns": header, "broken_rows": broken,
                           "duplicate_rows": dup_rows, "empty": {k: v for k, v in empties.items() if v}}
    dups: dict[str, int] = {}
    for col in ("email", "url", "link", "website", "name", "id", "title", "company"):
        i = _column(header, col)
        if i < 0 or header[i] in dups:
            continue
        vals = [r[i].strip().lower() for r in body if i < len(r) and r[i].strip()]
        if vals and len(vals) - len(set(vals)):
            dups[header[i]] = len(vals) - len(set(vals))
    out["duplicates"] = dups
    ei = _column(header, "email", "e-mail", "mail")
    if ei >= 0:
        emails = [r[ei].strip() for r in body if ei < len(r) and r[ei].strip()]
        doms = Counter(_domain(e) for e in emails if "@" in e)
        n = max(1, len(emails))
        out["email"] = {
            "column": header[ei], "count": len(emails),
            "invalid": sum(1 for e in emails if not _EMAIL.match(e)),
            "role": sum(1 for e in emails if ROLE_LOCAL.match(e.split("@")[0])),
            "free": sum(1 for e in emails if _domain(e) in FREE_MAIL),
            "academic": sum(1 for e in emails if re.search(r"\.(ac\.uk|edu|ac\.[a-z]{2})$", _domain(e))),
            "top_domains": doms.most_common(5),
            "top3_share": round(sum(c for _, c in doms.most_common(3)) / n, 3),
        }
    for col in ("source", "url", "link", "website", "reference"):
        si = _column(header, col)
        if si >= 0:
            vals = [r[si].strip() for r in body if si < len(r) and r[si].strip()]
            out["links"] = {"column": header[si], "count": len(vals), "links": sum(1 for v in vals if _LINK.match(v))}
            if ei >= 0:  # does the email come from the page it cites?
                same = 0
                for r in body:
                    if si < len(r) and ei < len(r) and "@" in r[ei]:
                        m = re.match(r"^https?://([^/:?#]+)", r[si].strip(), re.IGNORECASE)
                        if m and _base(m.group(1).removeprefix("www.")) == _base(_domain(r[ei])):
                            same += 1
                out["links"]["email_on_source_site"] = same
            break
    for col in ("country", "location", "region", "state", "city"):
        ci = _column(header, col)
        if ci >= 0:
            out["spread"] = {"column": header[ci], "values": Counter(r[ci].strip() for r in body if ci < len(r)).most_common(6)}
            break
    out["sample"] = [header] + [r[:width] for r in body[:SAMPLE_ROWS]]
    return out


def _headings(text: str, ext: str) -> list[str]:
    if ext in (".html", ".htm"):
        return [re.sub(r"<[^>]+>", "", h).strip() for h in re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", text, re.I | re.S)]
    return [m.group(2).strip() for m in re.finditer(r"^(#{1,3})\s+(.+)$", text, re.M)]


def profile_document(path: Path) -> dict[str, Any]:
    if path.stat().st_size > DOC_MAX_BYTES:
        return {"kind": "document", "too_big": True}
    text = path.read_text(encoding="utf-8", errors="replace")
    ext = path.suffix.lower()
    plain = re.sub(r"<[^>]+>", " ", text) if ext in (".html", ".htm") else text
    words = len(re.findall(r"\w+", plain))
    paras = [p.strip() for p in re.split(r"\n\s*\n", plain) if len(p.strip()) > 80]
    repeated = sum(c - 1 for c in Counter(paras).values() if c > 1)
    hold = [m.group(0) for m in _PLACEHOLDER.finditer(plain)]
    heads = _headings(text, ext)
    return {"kind": "document", "words": words, "headings": heads[:30], "heading_count": len(heads),
            "links": len(_MD_LINK.findall(text)), "placeholders": Counter(h.upper() for h in hold).most_common(5),
            "repeated_paragraphs": repeated, "opening": plain.strip()[:600],
            "sources_section": bool(re.search(r"^(#{1,3}\s*)?(sources|references|bibliography|citations)\b",
                                              text, re.I | re.M))}


def profile_docx(path: Path) -> dict[str, Any]:
    try:
        import docx  # python-docx, when the deck has it
    except ImportError:
        return {"kind": "document", "unread": "no reader for .docx here"}
    d = docx.Document(str(path))
    paras = [p for p in d.paragraphs if p.text.strip()]
    heads = [p.text.strip() for p in paras if (p.style.name or "").lower().startswith("heading")]
    text = "\n\n".join(p.text for p in paras)
    hold = [m.group(0) for m in _PLACEHOLDER.finditer(text)]
    return {"kind": "document", "words": len(re.findall(r"\w+", text)), "headings": heads[:30],
            "heading_count": len(heads), "links": len(_MD_LINK.findall(text)),
            "placeholders": Counter(h.upper() for h in hold).most_common(5), "repeated_paragraphs": 0,
            "opening": text[:600], "sources_section": bool(re.search(r"\b(sources|references)\b", text, re.I))}


def profile_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError as e:
        return {"kind": "json", "valid": False, "error": str(e)[:160]}
    count = len(data) if isinstance(data, (list, dict)) else 1
    return {"kind": "json", "valid": True, "top": type(data).__name__, "count": count}


def profile_code(found: list[tuple[float, str, Path, int]]) -> dict[str, Any]:
    langs: Counter[str] = Counter()
    lines = todos = 0
    syntax: list[str] = []
    tests = 0
    for _, rel, path, size in found[:CODE_MAX_FILES]:
        ext = path.suffix.lower()
        lang = _CODE_EXTS.get(ext)
        if re.search(r"(^|/)(tests?|__tests__|spec)(/|$)|(^|/)test_[^/]+\.py$|_test\.(py|go)$|\.(test|spec)\.[jt]sx?$", rel):
            tests += 1
        if not lang or size > 2_000_000:
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        langs[lang] += 1
        lines += body.count("\n") + 1
        todos += len(_TODO_CODE.findall(body))
        if ext == ".py":
            try:
                ast.parse(body)
            except SyntaxError as e:
                syntax.append(f"{rel}:{e.lineno}")
    rels = {rel for _, rel, _, _ in found}
    return {"kind": "code", "files": len(found), "languages": langs.most_common(5), "lines": lines,
            "test_files": tests, "syntax_errors": syntax[:10], "todos": todos,
            "readme": any(Path(r).name.lower().startswith("readme") for r in rels),
            "manifest": [m for m in _PROJECT_MARKERS if m in rels]}


def profile_file(path: Path) -> dict[str, Any]:
    ext = path.suffix.lower()
    try:
        if ext in _TABLE_EXTS:
            return profile_table(path)
        if ext in _DOC_EXTS:
            return profile_document(path)
        if ext == ".docx":
            return profile_docx(path)
        if ext == ".json":
            return profile_json(path)
    except Exception as e:  # noqa: BLE001 — a file that can't be read is said so, not fatal
        return {"kind": "unread", "error": str(e)[:200]}
    return {"kind": "file"}


def profile(found: list[tuple[float, str, Path, int]], named: set[str] | None = None) -> dict[str, Any]:
    """The run's delivered files, profiled: ``{kind, main: [{path, size, profile, flags}], other, work, flags}``."""
    named = named or set()
    groups = sort_files(found, named)
    out: dict[str, Any] = {"files": len(found), "other": len(groups["other"]), "work": len(groups["work"]), "main": []}
    if is_code_project(found):
        prof = profile_code(found)
        out.update(kind="code", main=[{"path": ".", "size": sum(f[3] for f in found), "profile": prof,
                                        "flags": flags(prof)}])
    else:
        for _, rel, path, size in groups["main"][:MAX_MAIN_PROFILED]:
            prof = profile_file(path)
            out["main"].append({"path": rel, "size": size, "profile": prof, "flags": flags(prof)})
        kinds = {m["profile"].get("kind") for m in out["main"]}
        out["kind"] = "none" if not kinds else kinds.pop() if len(kinds) == 1 else "mixed"
    missing = sorted(n for n in named if n not in {rel for _, rel, _, _ in found})
    out["missing"] = missing
    out["flags"] = ([f"a file its checks name is missing: {', '.join(missing)}"] if missing else []) + \
        [f"{m['path']}: {f}" for m in out["main"] for f in m["flags"]]
    return out


# ── what's worth a look ────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    return f"{round(100 * n / max(1, total))}%"


def flags(p: dict[str, Any]) -> list[str]:
    """The few things in a file's profile worth a look, in words. Thresholds are
    for raising a question, not for failing a run: the checks decide that."""
    out: list[str] = []
    kind = p.get("kind")
    if p.get("too_big"):
        return ["too big to read here"]
    if kind == "unread":
        return [f"couldn't be read ({p.get('error', '')})"]
    if kind == "table":
        n = p.get("rows", 0)
        if not n:
            return ["the table has no rows"]
        if p.get("broken_rows"):
            out.append(f"{p['broken_rows']} rows have a different number of fields than the header")
        if p.get("duplicate_rows"):
            out.append(f"{p['duplicate_rows']} duplicate rows")
        for col, d in (p.get("duplicates") or {}).items():
            out.append(f"{d} repeated values in {col}")
        for col, e in (p.get("empty") or {}).items():
            if e >= max(1, n * 0.05):
                out.append(f"{col} is empty in {e} rows")
        em = p.get("email")
        if em and em["count"]:
            c = em["count"]
            if em["invalid"]:
                out.append(f"{em['invalid']} values in {em['column']} aren't email addresses")
            if em["role"] >= c * 0.2:
                out.append(f"{_pct(em['role'], c)} of {em['column']} are role inboxes (info@, hello@, support@…), not people")
            if em["free"] >= c * 0.1:
                out.append(f"{_pct(em['free'], c)} of {em['column']} are free-mail addresses")
            if em["academic"] >= c * 0.15:
                out.append(f"{_pct(em['academic'], c)} of {em['column']} are university addresses")
            if c >= 20 and em["top3_share"] >= 0.25:
                tops = ", ".join(f"{d} {k}" for d, k in em["top_domains"][:3])
                out.append(f"{round(em['top3_share'] * 100)}% of {em['column']} sit on three domains ({tops})")
        ln = p.get("links")
        if ln and ln["count"]:
            if ln["links"] < ln["count"] * 0.8:
                out.append(f"only {_pct(ln['links'], ln['count'])} of {ln['column']} are links ({ln['links']} of {ln['count']})")
            if "email_on_source_site" in ln and em and em["count"] >= 20 and ln["email_on_source_site"] < em["count"] * 0.5:
                out.append(f"only {_pct(ln['email_on_source_site'], em['count'])} of emails are on the site their source points to")
    elif kind == "document":
        if p.get("unread"):
            return [p["unread"]]
        words = p.get("words", 0)
        if words < 150:
            out.append(f"only {words} words")
        if p.get("placeholders"):
            out.append("placeholders left: " + ", ".join(f"{h} ×{n}" for h, n in p["placeholders"]))
        if words >= 1500 and not p.get("heading_count"):
            out.append(f"{words:,} words with no headings")
        if p.get("repeated_paragraphs"):
            out.append(f"{p['repeated_paragraphs']} repeated paragraphs")
        if words >= 800 and not p.get("links") and not p.get("sources_section"):
            out.append("no links or sources")
    elif kind == "code":
        if p.get("syntax_errors"):
            out.append(f"syntax errors: {', '.join(p['syntax_errors'][:5])}")
        if not p.get("test_files"):
            out.append("no tests")
        if p.get("todos", 0) >= 5:
            out.append(f"{p['todos']} TODO/FIXME left")
        if not p.get("readme"):
            out.append("no README")
    elif kind == "json":
        if not p.get("valid"):
            out.append(f"not valid JSON ({p.get('error', '')})")
    return out


def render(prof: dict[str, Any]) -> str:
    """The reviewers' block: what the delivered files hold, read by Captain."""
    if not prof.get("main"):
        return ""
    lines = ["## VERIFIED — what the delivered files hold (Captain read them itself; believe this over any report)"]
    for m in prof["main"]:
        p = m["profile"]
        kind = p.get("kind")
        if kind == "table":
            head = (f"- {m['path']}: a table, {p.get('rows', 0):,} rows × {len(p.get('columns') or [])} columns "
                    f"({', '.join((p.get('columns') or [])[:12])})")
            lines.append(head)
            if p.get("email"):
                em = p["email"]
                lines.append(f"  - {em['column']}: {em['count']} values, {em['role']} role inboxes, {em['free']} free mail, "
                             f"{em['academic']} university; top domains " + ", ".join(f"{d} ({k})" for d, k in em["top_domains"]))
            if p.get("links"):
                ln = p["links"]
                lines.append(f"  - {ln['column']}: {ln['links']} of {ln['count']} are links")
            if p.get("spread"):
                lines.append(f"  - {p['spread']['column']}: " + ", ".join(f"{v or '(empty)'} {k}" for v, k in p["spread"]["values"]))
            sample = p.get("sample") or []
            if len(sample) > 1:
                buf = io.StringIO()
                csv.writer(buf).writerows(sample)
                lines.append("  - first rows:\n" + "\n".join("    " + ln for ln in buf.getvalue().splitlines()))
        elif kind == "document":
            lines.append(f"- {m['path']}: a document, {p.get('words', 0):,} words, {p.get('heading_count', 0)} headings, "
                         f"{p.get('links', 0)} links" + (", a sources section" if p.get("sources_section") else ""))
            if p.get("headings"):
                lines.append("  - outline: " + " · ".join(p["headings"][:15]))
        elif kind == "code":
            lines.append(f"- code: {p.get('files', 0)} files, {p.get('lines', 0):,} lines ("
                         + ", ".join(f"{lang} {n}" for lang, n in p.get("languages") or []) + f"), {p.get('test_files', 0)} test files"
                         + (f", manifest {', '.join(p['manifest'])}" if p.get("manifest") else ""))
        elif kind == "json":
            lines.append(f"- {m['path']}: JSON, " + (f"{p.get('count')} {p.get('top')} items" if p.get("valid") else "NOT valid"))
        else:
            lines.append(f"- {m['path']}: {m['size']:,} bytes")
        for f in m["flags"]:
            lines.append(f"  - WORTH A LOOK: {f}")
    if prof.get("missing"):
        lines.append(f"- MISSING: {', '.join(prof['missing'])} (named by its checks)")
    if prof.get("other") or prof.get("work"):
        lines.append(f"- beside these: {prof.get('other', 0)} other files and {prof.get('work', 0)} working files")
    return "\n".join(lines)


def summary(prof: dict[str, Any]) -> dict[str, Any]:
    """What's kept with a finished run: no sample rows or openings."""
    keep = []
    for m in prof.get("main") or []:
        p = {k: v for k, v in m["profile"].items() if k not in ("sample", "opening")}
        keep.append({"path": m["path"], "size": m["size"], "profile": p, "flags": m["flags"]})
    return {"kind": prof.get("kind"), "files": prof.get("files"), "main": keep, "flags": prof.get("flags") or [],
            "missing": prof.get("missing") or []}
