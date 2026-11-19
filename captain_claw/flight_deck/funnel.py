"""The list funnel: a Bat engine that builds a list the cheap way
(docs/funnel-runner-plan.md).

Bat's agents find and check a list's items in long turns, each model call
resending everything the worker has gathered (about 120k tokens a call on the
night's run). The funnel works through items with a fixed recipe instead:

* **sources** give the items: web searches, list pages, the owner's links, table
  files (CSV, TSV, XLSX, JSON lines) and documents (PDF, Word, text) in the run's
  folder. Offline lists need no web at all;
* **filters** drop what code can rule out for free (already seen, a field's value,
  a company-size band from a search snippet);
* **gather** reads what the checks need (a search, a few pages of the item's site)
  and code picks facts out of it (emails, dates, size, state);
* **decide**: one small model call per item (or per few, for small rows), in a
  fresh context: keep, maybe or drop, and the row;
* **checks** by code: every value the rules say was copied from a source must be in
  the text Captain read there, dates in their window, values from the allowed set.

A funnel run is a Bat run (``config.engine == "funnel"``): the run record and
folder, progress, caps, the judge, fix rounds and results are Bat's. Its planning
writes the recipe (``shared/recipe.json``) and the conventions, then one step per
group the sources name (or one for all), then the merge. Steps the fix rounds add
are Bat's own agent steps, with the conventions and the counts.

State lives in ``shared/funnel/`` as it goes, so a run stopped at its cap goes on
from where it was.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
import types
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

import structlog

from captain_claw.flight_deck import funnel_extract as fx
from captain_claw.llm import reasoning

log = structlog.get_logger(__name__)

ENGINE = "funnel"
RECIPE = "shared/recipe.json"
STATE = "shared/funnel"
ITEMS = f"{STATE}/items.csv"      # every item looked at: what became of it, and why
ROWS = f"{STATE}/rows.csv"        # the kept rows, the merge's input
MAYBE = f"{STATE}/maybe.csv"      # nearly there: what the finishing step works on
QUERIES = f"{STATE}/queries.json"  # searches made, queries used, new-query rounds
FINISH = f"{STATE}/finish.csv"            # rows the finishing agent adds, checked at the merge
FINISH_ITEMS = f"{STATE}/finish-items.csv"  # the nearly-there items it's given
COUNTS = f"{STATE}/counts.json"    # per group: how items went, stage by stage (for the run panel)
SUMMARY = "summary.md"
EXCLUSIONS = "exclusions.md"
LEDGER_FILE = "already-checked.csv"  # in extra/: what the factory's earlier runs checked (first_mate_ledger)

SEARCH_BUDGET = 1000
FETCH_CHARS = 6000
CONCURRENCY = 4
MAX_QUERY_ROUNDS = 2
ITEMS_MAX = 5000     # items looked at per step (config funnel_items_max)
READ_CHARS = 12_000  # what one item's judging call reads of its pages, in all
MORE_PAGES = 10      # list pages found in search results, followed per step
DECIDE_TOKENS = 8192
# What each funnel call may write. Its calls run with thinking off (a JSON answer and
# nothing more): with it on, Qwen thought 160k characters on the setup call, looped,
# and never answered, twice. A small limit also keeps a loop from running for minutes
# on the tier's whole output.
OUT_TOKENS = {"setup": 6000, "extract": 6000, "queries": 2000, "decide": 2500}
DECIDE_PER_ITEM = 600       # more room for a judging call with several short rows
THINKING_TOKENS = 16_000    # a judging call's room when the run turns thinking on
LINKS_AS_ITEMS = 5          # a links file with fewer links is reference, not a list to go through
NOTE_EVERY = 10
FINISH_MAX = 30          # nearly-there items one finishing step is given
AGENT_CONTEXT = 64_000   # a funnel run's agent steps work in this much context (config funnel_agent_context)

SOURCE_KINDS = ("search", "page", "links", "file", "document")
FILTER_OPS = ("regex", "not_regex", "in", "not_in", "min", "max", "size_max", "size_min")
CHECK_OPS = ("required", "in", "regex", "in_evidence", "within_months", "min", "max")
_ITEM_COLS = ["key", "name", "group", "status", "stage", "reason", "source", "step", "at"]
_MAYBE_COLS = ["key", "name", "group", "fields", "why", "sources"]
# Where an item left the funnel: code's filters, after reading, the judging call, a
# code check, its group already full, a duplicate; or kept, maybe, or the finishing
# agent's (kept, or refused when its cited page didn't bear it out).
STAGES = ("filter", "read", "judge", "check", "full", "duplicate", "kept", "maybe", "agent", "agent-check")
_GROUP_COL = "_group"
_SKIP_SITES = ("linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com", "youtube.com",
               "wikipedia.org", "crunchbase.com", "zoominfo.com", "rocketreach.co", "bbb.org", "yelp.com",
               "clutch.co", "glassdoor.com", "indeed.com", "apollo.io", "dnb.com", "bloomberg.com",
               "upcity.com", "designrush.com", "goodfirms.co", "expertise.com", "manta.com", "yellowpages.com",
               "mapquest.com", "signalhire.com", "craft.co", "owler.com", "pitchbook.com", "tiktok.com",
               "pinterest.com", "medium.com", "reddit.com", "google.com", "bing.com", "amazon.com",
               "sortlist.com", "themanifest.com", "agencyspotter.com", "chamberofcommerce.com", "buzzfile.com",
               "opencorporates.com", "bizapedia.com", "cbinsights.com", "tracxn.com", "builtin.com")
REFERENCE_CHARS = 3000   # what a judging call reads of the run's reference files, in all
CHARTER_CHARS = 3500     # and of the owner's charter (the columns' own rules)
# Words a group shares with others: a group's searches name it by the rest.
_GROUP_STOP = {"firms", "firm", "shops", "shop", "agencies", "agency", "boutiques", "boutique", "companies",
               "company", "services", "service", "groups", "group", "small", "and", "the", "for", "with"}


# ── the recipe ─────────────────────────────────────────────────────────

def _rel(path: Any) -> str:
    s = str(path or "").strip().replace("\\", "/")
    if not s or s.startswith(("/", "~")):
        return ""
    parts = [p for p in s.split("/") if p not in ("", ".")]
    return "" if not parts or ".." in parts else "/".join(parts)[:300]


def _int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _names(v: Any, limit: int = 60) -> list[str]:
    out: list[str] = []
    for x in v if isinstance(v, list) else []:
        name = str((x.get("name") if isinstance(x, dict) else x) or "").strip()[:80]
        if name and name not in out:
            out.append(name)
    return out[:limit]


def from_contract(contract: list[dict]) -> dict[str, Any]:
    """What the run's shaped checks already say about the list: its file, columns,
    groups and their amounts, the values a column allows, what must be unique."""
    out: dict[str, Any] = {"output": "", "columns": [], "target": 0, "group_by": "", "groups": {},
                           "allowed": {}, "key": "", "exclude": ""}
    for r in contract or []:
        check = r.get("check") or {}
        if check.get("type") != "table":
            continue
        if check.get("path") and not out["output"]:
            out["output"] = _rel(check["path"])
        out["columns"] += [c for c in check.get("columns") or [] if c not in out["columns"]]
        out["target"] = max(out["target"], int(check.get("min_rows") or 0))
        for col, vals in (check.get("allowed") or {}).items():
            out["allowed"].setdefault(col, list(vals))
        for col, n in (check.get("min_per") or {}).items():
            if not out["group_by"]:
                out["group_by"] = col
                vals = next((v for c, v in (check.get("allowed") or {}).items() if c.lower() == col.lower()), [])
                out["groups"] = {v: int(n) for v in vals}
        if check.get("unique") and not out["key"]:
            out["key"] = check["unique"][0]
        if check.get("exclude_pattern") and not out["exclude"]:
            out["exclude"] = check["exclude_pattern"]
    return out


def normalize_recipe(raw: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """A recipe as the funnel runs it: the model's draft, checked field by field,
    with what the run's checks say taking its place where they say it."""
    raw = raw if isinstance(raw, dict) else {}
    base = base or from_contract([])
    about = {}
    for c in raw.get("columns") or []:
        if isinstance(c, dict) and str(c.get("name") or "").strip():
            about[str(c["name"]).strip()[:80]] = str(c.get("what") or c.get("about") or "").strip()[:300]
    columns = list(base["columns"]) or _names(raw.get("columns"))  # the checks' columns win
    group_by = base["group_by"] or (str(raw.get("group_by") or "").strip() if raw.get("group_by") in columns else "")
    groups = dict(base["groups"])
    if not groups and isinstance(raw.get("groups"), dict):
        groups = {str(k)[:80]: _int(v, 0, 100_000, 0) for k, v in list(raw["groups"].items())[:20] if str(k).strip()}
    key = base["key"] if base["key"] in columns else str(raw.get("key") or "").strip()
    if key not in columns and key != "domain":
        key = next((c for c in columns if re.search(r"website|domain|url|e-?mail|link", c, re.I)), "") \
            or (columns[0] if columns else "")
    site = str(raw.get("site_column") or "").strip()
    if site not in columns:
        site = next((c for c in columns if re.search(r"website|domain|homepage|site\b|url", c, re.I)), "")
    output = base["output"] or _rel(raw.get("output")) or "results.csv"
    if not output.lower().endswith(".csv"):
        output = output.rsplit(".", 1)[0] + ".csv"
    recipe: dict[str, Any] = {
        "item": _short_item(raw.get("item")),
        "output": output, "columns": columns, "about": {c: about.get(c, "") for c in columns},
        "key": key, "site_column": site, "group_by": group_by, "groups": groups,
        "target": base["target"] or _int(raw.get("target"), 0, 1_000_000, 0),
        "sources": [], "filters": [], "gather": [], "extract": [], "checks": [],
        "decide": str(raw.get("decide") or "").strip()[:4000],
    }
    for s in raw.get("sources") or []:
        src = _source(s, groups)
        if src:
            recipe["sources"].append(src)
    for f in raw.get("filters") or []:
        if isinstance(f, dict) and f.get("op") in FILTER_OPS and str(f.get("column") or "").strip():
            recipe["filters"].append({"column": str(f["column"]).strip()[:80], "op": f["op"], "value": f.get("value")})
    if base["exclude"]:
        recipe["filters"].append({"column": "*", "op": "not_regex", "value": base["exclude"]})
    for g in raw.get("gather") or []:
        if not isinstance(g, dict):
            continue
        if g.get("kind") == "search" and str(g.get("query") or "").strip():
            recipe["gather"].append({"kind": "search", "query": str(g["query"]).strip()[:300],
                                     "count": _int(g.get("count"), 1, 10, 5)})
        elif g.get("kind") == "fetch" and str(g.get("url") or "").strip():
            paths = [str(p).strip()[:80] for p in (g.get("paths") or [""]) if isinstance(p, str)][:6] or [""]
            recipe["gather"].append({"kind": "fetch", "url": str(g["url"]).strip()[:300], "paths": paths})
    recipe["extract"] = [e for e in raw.get("extract") or [] if e in fx.EXTRACTORS]
    for col, vals in base["allowed"].items():
        if col in columns:
            recipe["checks"].append({"column": col, "op": "in", "value": list(vals)})
    for c in raw.get("checks") or []:
        if not isinstance(c, dict) or c.get("op") not in CHECK_OPS or c.get("column") not in columns:
            continue
        if any(x["column"] == c["column"] and x["op"] == c["op"] for x in recipe["checks"]):
            continue
        item = {"column": c["column"], "op": c["op"]}
        if c["op"] in ("in", "regex", "within_months", "min", "max"):
            item["value"] = c.get("value")
        if c["op"] == "in_evidence" and c.get("source_column") in columns:
            item["source_column"] = c["source_column"]
        recipe["checks"].append(item)
    for col in (key, group_by):
        if col and col in columns and not any(x["column"] == col and x["op"] == "required" for x in recipe["checks"]):
            recipe["checks"].append({"column": col, "op": "required"})
    recipe["decide_batch"] = _int(raw.get("decide_batch"), 1, 25, 1 if recipe["gather"] else 10)
    recipe["reference"] = [r for r in (_rel(x) for x in raw.get("reference") or []) if r][:6]
    return recipe


def _short_item(v: Any) -> str:
    """What one row is, as a noun ("small US knowledge firm"), never a sentence with lists."""
    text = re.split(r"\s*[(\[,;:—–]\s*|\s+-\s+", str(v or "").strip())[0]
    return " ".join(text.split())[:40] or "item"


def _source(s: Any, groups: dict[str, int]) -> dict[str, Any] | None:
    if not isinstance(s, dict) or s.get("kind") not in SOURCE_KINDS:
        return None
    if str(s.get("path") or "").endswith(LEDGER_FILE):
        return None  # what earlier runs checked: items to skip, not to take
    group = str(s.get("group") or "").strip()
    group = group if group in groups else ""
    kind = s["kind"]
    if kind == "search":
        queries = [str(q).strip()[:300] for q in s.get("queries") or [] if str(q).strip()][:40]
        if not queries:
            return None
        return {"kind": kind, "queries": queries, "group": group, "country": _country(s.get("country")),
                "count": _int(s.get("count"), 5, 50, 20), "about": str(s.get("about") or "").strip()[:300]}
    if kind == "page":
        url = str(s.get("url") or "").strip()
        return {"kind": kind, "url": url, "group": group} if url.startswith(("http://", "https://")) else None
    path = _rel(s.get("path"))
    if not path:
        return None
    out = {"kind": kind, "path": path, "group": group}
    if kind == "links":
        out["as"] = "pages" if s.get("as") == "pages" else "items"
    return out


_COUNTRY_NAMES = {"usa": "US", "united states": "US", "america": "US", "uk": "GB", "united kingdom": "GB",
                  "great britain": "GB", "england": "GB"}


def _country(v: Any) -> str:
    """A search's country as a two-letter code ("US"), or "" for any."""
    s = " ".join(str(v or "").strip().lower().replace(".", "").split())
    if re.fullmatch(r"[a-z]{2}", s):
        return s.upper()
    return _COUNTRY_NAMES.get(s, "")


def reference_text(folder: Path, recipe: dict[str, Any]) -> str:
    """What every judging call reads besides the item: the recipe's reference files,
    and the run's other given documents (not its tables, not what earlier runs checked,
    not a links file), up to REFERENCE_CHARS in all."""
    paths = [p for p in recipe.get("reference") or []]
    extra = folder / "extra"
    if extra.is_dir():
        for p in sorted(extra.iterdir()):
            rel = f"extra/{p.name}"
            if (p.is_file() and not p.name.startswith(".") and p.suffix.lower() in fx.DOC_EXTS
                    and p.name != LEDGER_FILE and not _links_count(p) and rel not in paths):
                paths.append(rel)
    out, room = [], REFERENCE_CHARS
    for rel in paths:
        if room <= 200:
            break
        try:
            text = " ".join(fx.read_document(folder / rel).split())
        except (OSError, ValueError):
            continue
        out.append(f"[{rel}] {text[:room]}")
        room -= len(out[-1])
    return "\n".join(out)


def charter_checks(task: str, columns: list[str]) -> list[dict[str, Any]]:
    """A column's fixed values, as the charter lists them ("CaptainFeature is exactly
    one of: A, B, C."), anywhere in its text, as checks: they hold even when drafting
    dropped the goal that said so."""
    out = []
    text = str(task or "")
    for col in columns:
        m = re.search(r"(?<![\w-])" + re.escape(col) + r"\s+is\s+(?:exactly\s+)?one\s+of\s*:?\s*([^.\n]+)",
                      text, re.I)
        if not m:
            continue
        vals = [v.strip(" '\"") for v in re.split(r",|;|\s+or\s+", m.group(1)) if v.strip(" '\"")]
        vals = [re.sub(r"^and\s+", "", v, flags=re.I) for v in vals]
        if len(vals) >= 2 and all(len(v) <= 60 for v in vals):
            out.append({"column": col, "op": "in", "value": vals})
    return out


def _search_country(code: str) -> str:
    """The country a search engine can be told (``web_search`` drops what its engine
    doesn't take, e.g. Brave has no Croatia): the code, or "" to search everywhere and
    let the words name the place."""
    try:
        from captain_claw.tools.web_search import _BRAVE_COUNTRIES
    except Exception:  # noqa: BLE001
        return code
    return code if code in _BRAVE_COUNTRIES else ""


def fix_output(recipe: dict[str, Any], task: str) -> bool:
    """The list's file as the charter names it ("list them in leads.csv"), when nothing
    else named it (the checks weren't shaped, the setup left it out). True when it
    changed."""
    if recipe.get("output") not in ("", "results.csv"):
        return False
    m = re.search(r"(?<![\w/.-])([\w-]+\.csv)\b", str(task or ""))
    if not m or m.group(1) == recipe.get("output"):
        return False
    recipe["output"] = m.group(1)
    return True


def _group_words(group: str) -> list[str]:
    return [w[:6] for w in re.findall(r"[a-z]+", group.lower()) if len(w) >= 4 and w not in _GROUP_STOP]


def _group_queries(group: str, place: str, size: re.Match | None) -> list[str]:
    """Searches for pages that list many of a group's items, any kind of item: lists,
    directories, databases, rankings. A company-size band adds a search for company
    pages with that band."""
    qs = [f"list of {group} {place}", f"{group} {place}", f"best {group} {place}", f"{group} directory {place}",
          f"{group} database {place}", f"top {group} {place}"]
    if size:
        qs.append(f'site:linkedin.com/company {group} {place} "{size.group(1)}-{size.group(2)} employees"')
    return [" ".join(q.split()) for q in qs]


def tidy_recipe(recipe: dict[str, Any], task: str, folder: Path | None = None) -> dict[str, Any]:
    """What code puts right in a model's recipe: a group's searches that never name it
    (copied from another group) are written for it; a file the charter points to for
    information is reference, not a source; the charter's fixed value lists become
    checks."""
    text = " ".join(str(task or "").split())
    places = fx.target_places(text)
    recipe["places"] = places
    place = fx.place_name(places[0]) if len(places) == 1 else ""
    size = _SIZE.search(text)
    fix_output(recipe, text)
    seen_lists: list[list[str]] = []
    for s in recipe["sources"]:
        if s["kind"] != "search" or not s.get("group"):
            continue
        words = _group_words(s["group"])
        joined = " ".join(s["queries"]).lower()
        if (words and not any(w in joined for w in words)) or s["queries"] in seen_lists:
            s["queries"] = _group_queries(s["group"], place, size)
            s["about"] = s["about"] or s["group"]
        seen_lists.append(s["queries"])
    if place:  # one place: a search that names none finds items anywhere
        code = places[0]
        for s in recipe["sources"]:
            if s["kind"] == "search":
                s["queries"] = [q if any(c == code for c, _ in fx.places_named(q)) else f"{q} {place}"
                                for q in s["queries"]]
                s["country"] = _search_country(code)
    refs = list(recipe.get("reference") or [])
    for s in list(recipe["sources"]):
        path = s.get("path") or ""
        name = Path(path).name if path else ""
        pointed = name and re.search(r"\b(?:is|are|described|explained|listed|see)\b[^.]{0,40}?"
                                     + re.escape(name), text, re.I)
        if s["kind"] == "document" and (path in refs or pointed):
            recipe["sources"].remove(s)
            if path not in refs:
                refs.append(path)
    recipe["reference"] = refs
    for c in charter_checks(text, recipe["columns"]):
        if not any(x["column"] == c["column"] and x["op"] == "in" for x in recipe["checks"]):
            recipe["checks"].append(c)
    return recipe


def render(template: str, item: dict[str, str], site_column: str = "") -> str | None:
    """A gather template filled from an item's fields: ``{Firm}`` (any field, case
    aside), ``{domain}`` (the item's site). None when a field it needs is empty."""
    fields = {k.lower(): v for k, v in item.items()}
    site = fields.get(site_column.lower(), "") if site_column else ""
    fields.setdefault("domain", fx.domain_of(site or fields.get("url", "") or fields.get("website", "")))
    missing = False

    def sub(m: re.Match) -> str:
        nonlocal missing
        v = str(fields.get(m.group(1).lower(), "") or "").strip()
        if not v:
            missing = True
        return v

    out = re.sub(r"\{([^{}]{1,80})\}", sub, template)
    return None if missing else out


def conventions(recipe: dict[str, Any]) -> str:
    """The recipe as the run's shared conventions, for any agent step that joins it."""
    lines = ["# Conventions for this list (written by the list funnel)", "",
             f"Each row is one {recipe['item']}. The list is `{recipe['output']}`.", "", "## Columns, in order", ""]
    for c in recipe["columns"]:
        lines.append(f"- **{c}**" + (f": {recipe['about'][c]}" if recipe["about"].get(c) else ""))
    if recipe["key"]:
        lines += ["", f"No two rows share a {recipe['key']}."]
    if recipe["groups"]:
        lines += ["", f"## {recipe['group_by'] or 'Groups'}", ""]
        lines += [f"- {g}: {n}" if n else f"- {g}" for g, n in recipe["groups"].items()]
    elif recipe["target"]:
        lines += ["", f"The list needs {recipe['target']} rows."]
    if recipe["decide"]:
        lines += ["", "## What belongs on the list", "", recipe["decide"]]
    checks = [c for c in recipe["checks"] if c["op"] != "required"]
    if checks:
        lines += ["", "## What Captain checks in every row", ""]
        lines += [f"- {c['column']} {_check_words(c)}" for c in checks]
    lines += ["", "Work in progress is in `shared/funnel/`: `items.csv` has every item looked at and what became "
              "of it; `maybe.csv` the ones nearly there."]
    return "\n".join(lines) + "\n"


def _check_words(c: dict) -> str:
    op, v = c["op"], c.get("value")
    if op == "in":
        return "is one of: " + ", ".join(map(str, v if isinstance(v, list) else [v]))
    if op == "regex":
        return f"matches /{v}/"
    if op == "in_evidence":
        return "appears in the page or file it came from" + (
            f" (the one in {c['source_column']})" if c.get("source_column") else "")
    if op == "within_months":
        return f"is a date (YYYY-MM-DD or YYYY-MM) within the last {v} months"
    return f"is at {'least' if op == 'min' else 'most'} {v}"


def load_recipe(folder: Path) -> dict[str, Any] | None:
    try:
        return json.loads((folder / RECIPE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ── setup: the recipe from the charter ─────────────────────────────────

SETUP_SYSTEM = """You set up a list-building run. Code will do the searching, fetching and checking; a model
will judge each item once, in a short call. Write the recipe it follows, as ONLY a JSON object:

{"item": "<what one row is, e.g. firm, person, product, invoice>",
 "columns": [{"name": "<column>", "what": "<what it holds, where it comes from>"}],
 "key": "<the column no two rows share (a website, an email, an id)>",
 "site_column": "<the column holding the item's own website, if it has one>",
 "group_by": "<column the list is split by, or empty>", "groups": {"<group>": <how many>},
 "target": <rows wanted, 0 = every item that passes>,
 "sources": [
   {"kind": "file", "path": "<table in the folder>", "group": ""},
   {"kind": "document", "path": "<pdf/docx/text to pull items out of>", "group": ""},
   {"kind": "links", "path": "<links file>", "as": "items|pages"},
   {"kind": "page", "url": "<a page that lists many items>", "group": ""},
   {"kind": "search", "queries": ["<query>", ...], "group": "<group>", "country": "US", "count": 20,
    "about": "<what these searches look for>"}],
 "filters": [{"column": "<column, or size_band>", "op": "regex|not_regex|in|not_in|min|max|size_max|size_min",
              "value": ...}],
 "gather": [{"kind": "search", "query": "\\"{Firm}\\" site:linkedin.com/company", "count": 5},
            {"kind": "fetch", "url": "{Website}", "paths": ["", "about", "team", "contact", "news"]}],
 "extract": ["emails", "phones", "dates", "size_band", "us_state", "links"],
 "decide": "<the owner's rules for an item, short, in their words>",
 "checks": [{"column": "<column>", "op": "required|in|regex|in_evidence|within_months|min|max",
             "value": ..., "source_column": "<column with the link it came from>"}],
 "decide_batch": <items per judging call: 1 when each item has pages to read, up to 20 for short rows>,
 "reference": ["<files that describe something the rows need (your product, rules, a price list), not items>"]}

Rules:
- Use the owner's own files and links first; they cost nothing to read. Search only when the list has to be
  found on the web, and then write 10-25 searches per group that find pages listing many of the right items
  (directories, catalogues, databases, rankings, "best of" and category pages), not one item each.
- Gather only what the checks and the rules need, and put the cheapest test first: a size band from a search
  snippet ("size_max" filter) before reading pages.
- Templates name columns in braces ({Firm}, {Website}) or {domain}.
- checks: "in_evidence" for every value copied from a source (an email, a date, a quote) with the column that
  holds its link; "within_months" for a date window; "in" for fixed values.
- An offline list (a file or documents only) has no searches and no gather unless the rules need the web.
- A file the goal points to for information (what the owner's product is, house rules) is "reference", never a
  source: no items are taken from it, and every judging call reads it.
- Each group's searches name that group; never copy one group's searches to another. When the list is about one
  place, every search names it (the country, or a state or city in it).
- Never invent a file or a link that isn't in what you're shown."""


def _folder_listing(folder: Path) -> str:
    """What the run was given (extra/) and anything else at the top of its folder, for
    the setup call: tables with their columns and first rows, documents' first lines,
    links files' first links."""
    lines: list[str] = []
    places = [folder / "extra", folder]
    seen: set[Path] = set()
    for place in places:
        if not place.is_dir():
            continue
        for p in sorted(place.iterdir()):
            if not p.is_file() or p.name.startswith(".") or p in seen or len(lines) > 60 or p.name == LEDGER_FILE:
                continue
            seen.add(p)
            rel = p.relative_to(folder).as_posix()
            ext = p.suffix.lower()
            try:
                if ext in fx.TABLE_EXTS:
                    cols, rows = fx.read_table(p, limit=2000)
                    lines.append(f"- {rel}: table, {len(rows)} rows, columns {', '.join(cols[:30])}")
                    lines += [f"    {json.dumps(r, ensure_ascii=False)[:300]}" for r in rows[:3]]
                elif ext in fx.DOC_EXTS:
                    text = fx.read_document(p)
                    found = fx.read_links(text)
                    if len(found) >= 3 and len(found) * 120 > len(text) * 0.3:
                        lines.append(f"- {rel}: links file, {len(found)} links, first: "
                                     + "; ".join(f"{x['title'] or x['url']} ({x['url']})" for x in found[:5]))
                    else:
                        lines.append(f"- {rel}: document, {len(text):,} characters: "
                                     + " ".join(text[:400].split()))
            except Exception as e:  # noqa: BLE001 — listed as unreadable
                lines.append(f"- {rel}: can't be read ({str(e)[:120]})")
    return "\n".join(lines) or "(no files)"


def _links_count(p: Path) -> int:
    """How many links a links file the owner added holds (0 when it isn't one)."""
    try:
        text = p.read_text(encoding="utf-8", errors="replace")[:200_000]
    except OSError:
        return 0
    found = fx.read_links(text)
    return len(found) if found and (text.lstrip().startswith("# Links") or len(found) * 120 > len(text) * 0.3) else 0


def default_recipe(folder: Path, base: dict[str, Any]) -> dict[str, Any] | None:
    """A recipe without a model: the folder's own tables and documents as sources, and
    a links file as a list to go through only when it holds a list (a few links are
    reference: the product's site, say). None when there's nothing to read."""
    sources = []
    for place in (folder / "extra", folder):
        if not place.is_dir():
            continue
        for p in sorted(place.iterdir()):
            if not p.is_file() or p.name.startswith(".") or p.name == LEDGER_FILE:
                continue
            rel = p.relative_to(folder).as_posix()
            if rel == base.get("output"):
                continue
            ext = p.suffix.lower()
            links = _links_count(p) if ext in (".md", ".markdown", ".txt") else 0
            if links:
                if links >= LINKS_AS_ITEMS:
                    sources.append({"kind": "links", "path": rel, "as": "items"})
            elif ext in fx.TABLE_EXTS:
                sources.append({"kind": "file", "path": rel})
            elif ext in fx.DOC_EXTS:
                sources.append({"kind": "document", "path": rel})
    if not sources:
        return None
    return normalize_recipe({"sources": sources}, base)


_NOUNS = re.compile(r"\b(firms?|compan(?:y|ies)|agenc(?:y|ies)|startups?|businesses|people|persons?|products?|"
                    r"schools?|clinics?|vendors?|suppliers?|shops?|stores?|restaurants?|investors?|funds?)\b", re.I)
_SIZE = re.compile(r"(\d{1,4})\s*(?:-|–|—|to)\s*(\d{1,6})\s+(?:employees|people|staff)", re.I)
_MONTHS = re.compile(r"(?:last|within|past)\s+(\d{1,3})\s+months", re.I)


_ASK = re.compile(r"\b(?:find|list|collect|gather|compile|research|identify|source|build a list of|make a list of)"
                  r"\s+(?:(?:about|at least|up to|some|all|the|\d+)\s+)*([a-z][a-z/&'-]*(?:\s+[a-z][a-z/&'-]*){0,4}?)"
                  r"(?=\s+(?:in|from|that|which|who|with|for|across|on|near|around|and list|,)\b|[.,;:]|$)", re.I)


def item_from(text: str) -> str:
    """What one row of the list is, from the charter's own ask ("Find 40 vegan
    restaurants in Zagreb" → "vegan restaurant"), as a short singular noun; "item"
    when it doesn't say."""
    m = _ASK.search(text or "")
    if m:
        words = [w for w in m.group(1).split() if not fx.places_named(w) and w.lower() not in ("small", "new")]
        if words:
            words[-1] = _singular(words[-1])
            return " ".join(words)[:40]
    noun = _NOUNS.search(text or "")
    return _singular(noun.group(1)) if noun else "item"


def _singular(word: str) -> str:
    w = word.lower()
    for plural, one in (("ies", "y"), ("people", "person"), ("businesses", "business")):
        if w.endswith(plural):
            return w[: -len(plural)] + one if plural == "ies" else one
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


def web_recipe(base: dict[str, Any], task: str) -> dict[str, Any] | None:
    """A recipe without a model for a list found on the web, from the run's checks and
    charter: searches for each group (directories, lists, member lists, company pages
    with a size band), the size band first when the charter names one, then the item's
    own pages; emails checked against the page they're cited from. None without the
    list's columns."""
    cols = base["columns"]
    if not cols:
        return None
    text = " ".join(str(task or "").split())
    item = item_from(text)
    places = fx.target_places(text)
    place = fx.place_name(places[0]) if len(places) == 1 else ""
    country = _search_country(places[0]) if len(places) == 1 else ""
    size = _SIZE.search(text)
    groups = list(base["groups"]) or [""]
    sources = []
    for g in groups:
        what = g or f"{item}s"
        sources.append({"kind": "search", "group": g, "country": country, "count": 20,
                        "queries": _group_queries(what, place, size), "about": f"{what} {place}".strip()})
    site = next((c for c in cols if re.search(r"website|domain|homepage|site\b|url", c, re.I)
                 and not re.search(r"source", c, re.I)), "")
    name = next((c for c in cols if c != site and not re.search(r"mail|url|source|date|segment|state", c, re.I)), "")
    gather: list[dict[str, Any]] = []
    filters: list[dict[str, Any]] = []
    if size and name:
        gather.append({"kind": "search", "query": f'"{{{name}}}" site:linkedin.com/company', "count": 5})
        filters.append({"column": "size_band", "op": "size_max", "value": int(size.group(2))})
    if site:
        gather.append({"kind": "fetch", "url": f"{{{site}}}", "paths": ["", "about", "team", "contact", "news"]})
    checks: list[dict[str, Any]] = []
    for c in cols:
        src = next((x for x in cols if x.lower().replace(" ", "") in (f"{c}sourceurl".lower(), f"{c}source".lower())), "")
        if src and re.search(r"mail", c, re.I):
            checks.append({"column": c, "op": "in_evidence", "source_column": src})
    months = _MONTHS.search(text)
    date_col = next((c for c in cols if re.search(r"date", c, re.I)), "")
    if months and date_col:
        checks.append({"column": date_col, "op": "within_months", "value": int(months.group(1))})
    extract = ["emails", "dates"] + (["size_band"] if size else []) + (["us_state"] if "US" in places else [])
    return normalize_recipe({"item": item, "site_column": site, "sources": sources, "gather": gather,
                             "filters": filters, "extract": extract, "checks": checks, "decide": text[:3000]}, base)


def fallback_recipe(folder: Path, base: dict[str, Any], task: str) -> dict[str, Any] | None:
    """When no model's recipe can be had: the folder's own lists and documents, else
    web searches built from the run's checks and charter."""
    return default_recipe(folder, base) or web_recipe(base, task)


async def setup(run: dict, folder: Path) -> tuple[dict[str, Any] | None, str]:
    """The run's recipe, written to ``shared/recipe.json`` with the conventions
    beside it. ``(recipe, why)``: None and why when there's no list it can build."""
    from captain_claw.flight_deck import bat_routes
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers
    from captain_claw.llm import Message
    cfg = run.get("config") or {}
    base = from_contract(list(cfg.get("contract") or []))
    have = load_recipe(folder)
    if have and have.get("sources"):
        return have, ""  # planned before a restart: the same recipe
    recipe: dict[str, Any] | None = None
    why = ""
    try:
        tiers, _ = await _load_owner_tiers(get_db(), run["owner_id"])
        creds = bat_routes._creds_for(tiers)
    except Exception as e:  # noqa: BLE001
        creds, why = None, f"no model to set it up with ({e})"
    if creds:
        rules = "\n".join(f"- {r.get('text')}" for r in cfg.get("contract") or [] if r.get("text"))
        known = {k: v for k, v in base.items() if v}
        prompt = (f"# The goal\n{str(run.get('task', ''))[:6000]}\n\n"
                  + (f"# Its checks\n{rules}\n\n" if rules else "")
                  + (f"# Already fixed by its checks (keep these)\n{json.dumps(known, ensure_ascii=False)}\n\n"
                     if known else "")
                  + f"# The run's folder\n{await asyncio.to_thread(_folder_listing, folder)}")
        for attempt in range(2):
            try:
                async with bat_routes._metered(run["id"], "funnel-setup"):
                    prov, mt = bat_routes.think_call(creds, temperature=0.2)
                    with reasoning.call_effort("off"):  # a JSON answer: no thinking to loop in
                        resp = await prov.complete(messages=[
                            Message(role="system", content=SETUP_SYSTEM
                                    + ("\nThe answer is JSON only, nothing before or after it." if attempt else "")),
                            Message(role="user", content=prompt),
                        ], temperature=0.2, max_tokens=min(mt, OUT_TOKENS["setup"]))
                raw = fx.parse_json(resp.content or "")
                if isinstance(raw, dict):
                    recipe = normalize_recipe(raw, base)
                    if recipe["sources"]:
                        break
                    why = "its recipe named no source to take items from"
                else:
                    why = "its recipe wasn't JSON"
            except Exception as e:  # noqa: BLE001 — tried again, then the folder's own files
                why = f"{type(e).__name__}: {str(e)[:160]}"
                log.warning("funnel setup call failed", run_id=run.get("id"), error=str(e))
            recipe = None
    if recipe is None:
        recipe = await asyncio.to_thread(fallback_recipe, folder, base, str(run.get("task", "")))
        if recipe is not None:
            how = "web searches built from its checks" if all(
                s["kind"] == "search" for s in recipe["sources"]) else "the folder's own files"
            why = f"set up from {how} ({why})" if why else ""
    if recipe is not None:
        recipe = tidy_recipe(recipe, str(run.get("task", "")), folder)
        if not recipe["sources"]:  # its only source was a file to read for information
            web = web_recipe(from_contract(list(cfg.get("contract") or [])), str(run.get("task", "")))
            if web is not None:
                recipe = tidy_recipe({**web, "reference": recipe.get("reference") or []}, str(run.get("task", "")), folder)
                why = (why + "; " if why else "") + "its only file is reference, so it searches the web"
    if recipe is None or not recipe["columns"] or not recipe["sources"]:
        return None, why or ("no columns for the list" if recipe is not None else "nothing to build a list from")
    (folder / STATE).mkdir(parents=True, exist_ok=True)
    (folder / RECIPE).write_text(json.dumps(recipe, indent=1, ensure_ascii=False), encoding="utf-8")
    (folder / "shared" / "conventions.md").write_text(conventions(recipe), encoding="utf-8")
    return recipe, why


async def plan(run: dict) -> list[dict] | None:
    """A funnel run's steps: one per group its sources search for (the owner's files
    first, in a step of their own), then the merge. None, with a note saying why,
    when there's no list to build this way: the run is planned as any Bat run."""
    from captain_claw.flight_deck import bat_loop, bat_routes
    store = bat_loop.get_store()
    folder = bat_routes._run_folder(run)
    recipe, why = await setup(run, folder)
    if recipe is None:
        if store is not None:
            await store.append_event(run["id"], "note", f"The list funnel can't run this ({why}): "
                                     "it runs as a Bat run instead.")
        return None
    steps: list[dict] = []
    kinds: dict[str, dict] = {}

    def add(title: str, kind: str, group: str = "") -> None:
        key = f"step-{len(steps) + 1}"
        steps.append({"step_key": key, "title": title[:400], "seq": len(steps),
                      "depends_on": [steps[-1]["step_key"]] if steps else []})
        kinds[key] = {"kind": kind, "group": group}

    item = recipe["item"]
    plural = item if item.endswith("s") else f"{item}s"
    untagged = [s for s in recipe["sources"] if not s.get("group")]
    tagged = []
    for s in recipe["sources"]:
        if s.get("group") and s["group"] not in tagged:
            tagged.append(s["group"])
    if untagged or not tagged:
        what = ", ".join(sorted({_source_words(s) for s in untagged})) or "its sources"
        add(f"Go through {what} for {plural} and check each", "work")
    for g in tagged:
        n = recipe["groups"].get(g) or 0
        add(f"Find and check {n} {g}" if n else f"Find and check {g}", "work", g)
    if leftovers_on(run) and (recipe["groups"] or recipe.get("target")):
        add("Finish what's still short from the nearly-there items", "finish")
    add(f"Put the list together in {recipe['output']}", "merge")
    await bat_routes.merge_config(run["id"], {"funnel_steps": kinds})
    if store is not None:
        checks = sum(1 for c in recipe["checks"] if c["op"] != "required")
        searches = sum(len(s.get("queries") or []) for s in recipe["sources"])
        await store.append_event(run["id"], "note", (
            f"List funnel: {len(recipe['sources'])} source{'' if len(recipe['sources']) == 1 else 's'}"
            + (f" ({searches} searches planned)" if searches else "")
            + f", {checks} check{'' if checks == 1 else 's'} by code, each {item} judged in one short call. "
            "The recipe is in shared/recipe.json." + (f" {why}" if why else "")))
    return steps


def _source_words(s: dict) -> str:
    return {"search": "web searches", "page": "list pages", "links": "the owner's links",
            "file": f"`{s.get('path')}`", "document": f"`{s.get('path')}`"}[s["kind"]]


def step_kind(run: dict, step: dict) -> dict | None:
    """A funnel step's kind (``work`` for a group, ``merge``), or None: a step the fix
    rounds added, which Bat's agents do."""
    if (run.get("config") or {}).get("engine") != ENGINE:
        return None
    return ((run.get("config") or {}).get("funnel_steps") or {}).get(step.get("step_key", ""))


# ── the state on disk ──────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def _append_csv(path: Path, cols: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({c: ("" if row.get(c) is None else str(row.get(c))) for c in cols})


def skip_sites(recipe: dict[str, Any]) -> tuple[str, ...]:
    """The directory and social sites that aren't an item's own: all of them, but one
    the list's own site column is about (a "LinkedIn" column, a "GitHub repo" link, an
    "Amazon URL"): that site is where its items live."""
    site = (recipe.get("site_column") or "").lower()
    about = (recipe.get("about") or {}).get(recipe.get("site_column") or "", "").lower()
    words = f"{site} {about}"
    return tuple(s for s in _SKIP_SITES if not re.search(r"(?<![a-z])" + re.escape(s.split(".")[0]) + r"(?![a-z])", words))


def _own_result(results: list[dict[str, str]], name: str, url: str) -> str:
    """An item's own search result (its title, link and snippet), so code reads what
    was said of it alone, not of the other results beside it."""
    d = fx.domain_of(url)
    low = name.lower()
    for r in results:
        if url and (r["url"].rstrip("/") == url.rstrip("/") or (d and fx.domain_of(r["url"]) == d
                                                                  and not any(fx.same_site(d, s) for s in _SKIP_SITES))):
            return f"{r['text']}\n{r['url']}"
    for r in results:
        if low and low in r["title"].lower():
            return f"{r['text']}\n{r['url']}"
    return ""


def known_keys(folder: Path) -> set[str]:
    """The items the factory's earlier runs checked, as the run was given them
    (``extra/already-checked.csv``): the funnel doesn't look at them again."""
    return {norm_key(r.get("key", "")) for r in _read_csv(folder / "extra" / LEDGER_FILE) if r.get("key")}


def norm_key(v: str) -> str:
    """An item's key as compared: a link or a site by its domain, the rest by its
    lowercased words."""
    s = str(v or "").strip()
    if re.match(r"^(https?://|www\.)", s, re.I) or re.fullmatch(r"[\w-]+(\.[\w-]+)+/?", s):
        return fx.domain_of(s)
    return " ".join(s.lower().split())


# ── the run's tools (search and fetch) ─────────────────────────────────

class Tools:
    """Search and fetch through one agent of the run's own (spawned like a Bat
    worker, never given a turn), so the owner's search provider and keys, the
    worker floor and rehearsal rules apply. Spawned on the first call."""

    def __init__(self, run: dict, step: dict):
        self.run, self.step = run, step
        self.port, self.auth, self.slug = 0, "", ""

    async def _ensure(self) -> None:
        if self.port:
            return
        from captain_claw.flight_deck import bat_routes
        from captain_claw.flight_deck.auth import get_db
        from captain_claw.flight_deck.basna_routes import _load_owner_tiers
        run, key = self.run, self.step["step_key"]
        owner = run["owner_id"]
        db = get_db()
        user = await db.get_user_by_id(owner) or {"id": owner}
        tiers, env_vars = await _load_owner_tiers(db, owner)
        env = bat_routes._bat_env(run["id"], f"{key}-tools", owner)
        if (run.get("config") or {}).get("rehearsal"):
            env.append({"key": "CLAW_REHEARSAL", "value": str(run["config"]["rehearsal"])})
        stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id=owner))
        got = await bat_routes._spawn_bat_worker(
            stub, user, name=f"bat-{run['id'][:8]}-{key}-tools"[:60], tier=bat_routes.step_tier(run, self.step, tiers),
            tiers=tiers, env_vars=env_vars, extra_env=env)
        if not got["ok"]:
            raise RuntimeError(f"couldn't start the run's search and fetch agent: {got['message']}")
        self.port, self.auth, self.slug = got["port"], got["auth"], got["slug"]
        try:
            from captain_claw.flight_deck import first_mate_agent
            if (run.get("config") or {}).get("factory_id"):
                await first_mate_agent.label_worker(self.slug, run, key)
            await bat_routes._label_worker(self.slug, run, key)
        except Exception:  # noqa: BLE001 — a label is for the sidebar
            log.debug("funnel: couldn't label the tools agent", exc_info=True)

    async def call(self, tool: str, args: dict) -> tuple[bool, str]:
        await self._ensure()
        import httpx
        url = f"http://localhost:{self.port}/api/tool" + (f"?token={quote(self.auth, safe='')}" if self.auth else "")
        body = {"tool": tool, "args": args, "token": self.auth}
        deadline = time.monotonic() + 90
        while True:
            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    resp = await client.post(url, json=body)
                data = resp.json() if resp.status_code == 200 else {}
                if resp.status_code != 200:
                    return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
                return bool(data.get("success")), str(data.get("content") or data.get("error") or "")
            except (httpx.ConnectError, httpx.RemoteProtocolError):
                if time.monotonic() > deadline:  # the agent is still starting: wait for it a while
                    return False, "the run's search and fetch agent didn't answer"
                await asyncio.sleep(2)

    async def search(self, query: str, count: int, country: str = "") -> tuple[bool, str]:
        args: dict[str, Any] = {"query": query, "count": count}
        if country:
            args["country"] = country
        return await self.call("web_search", args)

    async def fetch(self, url: str) -> tuple[bool, str]:
        return await self.call("web_fetch", {"url": url, "deep_fetch": False})

    async def close(self) -> None:
        if self.slug:
            from captain_claw.flight_deck import bat_routes
            try:
                await asyncio.to_thread(bat_routes._teardown, [self.slug])
            except Exception:  # noqa: BLE001 — the run's end removes it too
                log.debug("funnel: tools agent teardown failed", exc_info=True)
            self.slug, self.port = "", 0


def make_tools(run: dict, step: dict) -> Tools:
    return Tools(run, step)


# ── a step's work ──────────────────────────────────────────────────────

EXTRACT_SYSTEM = """Pull the items out of this text: each {item} it names that could belong on the list.
Reply ONLY with JSON: {{"items": [{{"name": "...", "url": "<its own website or page, if the text gives one>",
"fields": {{"<column>": "<value the text gives>"}}}}], "pages": ["<links in the text to pages that list more such
items>"]}}. Only what the text says; no item that is a directory, an article or a list itself. Keep your
thinking short."""

DECIDE_SYSTEM = """You judge items for a list, one by one, from what Captain read about each. Use only the facts
shown; a value you can't support from them stays empty, and a source is a link shown with that fact.
Reply ONLY with a JSON array, one object per item, in order:
[{"id": <its id>, "verdict": "keep|maybe|drop", "reason": "<one short line>",
  "row": {"<column>": "<value>"}}]
keep: it meets every rule and the row is filled from the facts. maybe: it seems to fit but a fact the rules need
wasn't in what was read; also when nothing read came from the item's own website, unless a fact shown breaks a
rule. drop: it breaks a rule. A column with a fixed set of values takes one of them exactly. Dates as YYYY-MM-DD,
or YYYY-MM when there's no day.
Keep your thinking short."""


class Work:
    """One work step: items from its sources, until its group is full or the sources
    run out."""

    def __init__(self, run: dict, step: dict, recipe: dict[str, Any], folder: Path, group: str = ""):
        self.run, self.step, self.recipe, self.folder, self.group = run, step, recipe, folder, group
        self.key = step["step_key"]
        self.cfg = run.get("config") or {}
        self.budget = _int(self.cfg.get("funnel_search_budget"), 0, 100_000, SEARCH_BUDGET)
        self.items_max = _int(self.cfg.get("funnel_items_max"), 1, 1_000_000, ITEMS_MAX)
        self.thinking = str(self.cfg.get("funnel_thinking") or "off").lower() == "on"
        self.charter = " ".join(str(run.get("task") or "").split())[:CHARTER_CHARS]
        self.places = list(recipe.get("places") or sorted({s["country"] for s in recipe["sources"] if s.get("country")}))
        self.skip_sites = skip_sites(recipe)
        self.reference = reference_text(folder, recipe)
        self.more_pages: list[str] = []
        self.tools: Tools | None = None
        self.creds: dict | None = None
        self.seen: set[str] = set()
        self.kept: dict[str, int] = {}
        self.counts = {"looked": 0, "kept": 0, "maybe": 0, "dropped": 0, "known": 0}
        self.known = known_keys(folder)
        self.reasons: dict[str, int] = {}
        self.pending: list[dict] = []
        self.recent: list[str] = []
        self.queries = self._load_queries()
        self.stopped = ""
        for r in _read_csv(folder / ITEMS):
            self.seen.add(r.get("key", ""))
            if r.get("status") == "kept":
                self.kept[r.get("group", "")] = self.kept.get(r.get("group", ""), 0) + 1

    # ── state ──
    def _load_queries(self) -> dict:
        try:
            return json.loads((self.folder / QUERIES).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"searches": 0, "used": [], "rounds": {}}

    def _save_queries(self) -> None:
        (self.folder / QUERIES).parent.mkdir(parents=True, exist_ok=True)
        (self.folder / QUERIES).write_text(json.dumps(self.queries, indent=1), encoding="utf-8")

    def quota(self, group: str) -> int:
        return int(self.recipe["groups"].get(group) or 0)

    def full(self) -> bool:
        groups = self.recipe["groups"]
        if self.group:
            q = self.quota(self.group)
            return bool(q) and self.kept.get(self.group, 0) >= q
        if groups and all(groups.values()):
            return all(self.kept.get(g, 0) >= n for g, n in groups.items())
        target = int(self.recipe.get("target") or 0)
        return bool(target) and sum(self.kept.values()) >= target

    async def _event(self, kind: str, detail: str) -> None:
        from captain_claw.flight_deck import bat_loop
        store = bat_loop.get_store()
        if store is not None:
            await store.append_event(self.run["id"], "action", kind, agent=self.key, detail=detail[:300])

    async def _tools(self) -> Tools:
        if self.tools is None:
            self.tools = make_tools(self.run, self.step)
        return self.tools

    async def close(self) -> None:
        if self.tools is not None:
            await self.tools.close()

    # ── the model ──
    async def _creds(self) -> dict:
        if self.creds is None:
            from captain_claw.flight_deck import bat_routes
            from captain_claw.flight_deck.auth import get_db
            from captain_claw.flight_deck.basna_routes import _load_owner_tiers
            tiers, _ = await _load_owner_tiers(get_db(), self.run["owner_id"])
            tier = bat_routes.step_tier(self.run, self.step, tiers)
            self.creds = bat_routes._creds_for(tiers, prefer=(tier, "fast", "balanced", "reason"))
            if not self.creds:
                raise RuntimeError("no model is set up to judge items with")
        return self.creds

    async def _ask(self, label: str, system: str, user: str, items: int = 1) -> Any:
        """One of the funnel's short calls: thinking off and a small limit, except a
        judging call in a run that turns thinking on (``config.funnel_thinking``)."""
        from captain_claw.flight_deck import bat_routes
        from captain_claw.llm import Message
        creds = await self._creds()
        think = label == "decide" and self.thinking
        limit = THINKING_TOKENS if think else OUT_TOKENS.get(label, 4000) + (
            DECIDE_PER_ITEM * max(0, items - 1) if label == "decide" else 0)
        async with bat_routes._metered(self.run["id"], f"funnel-{label}"):
            prov, mt = bat_routes.think_call(creds, temperature=0.0, default_max=DECIDE_TOKENS)
            with reasoning.call_effort(None if think else "off"):
                resp = await prov.complete(messages=[Message(role="system", content=system),
                                                     Message(role="user", content=user)],
                                           temperature=0.0, max_tokens=min(mt, limit))
        return fx.parse_json(resp.content or "")

    # ── the run ──
    async def run_step(self) -> dict:
        sources = [s for s in self.recipe["sources"] if (s.get("group") or "") == self.group]
        try:
            for src in sources:
                if self.full() or self.stopped:
                    break
                async for batch in self.candidates(src):
                    await self.handle(batch)
                    if self.full() or self.stopped:
                        break
            followed = 0
            while self.more_pages and followed < len(self.more_pages) and not self.full() and not self.stopped:
                url = self.more_pages[followed]
                followed += 1
                async for batch in self.candidates({"kind": "page", "url": url, "group": self.group}):
                    await self.handle(batch)
                    if self.full() or self.stopped:
                        break
            searches = [s for s in sources if s["kind"] == "search"]
            rounds = int(self.queries["rounds"].get(self.group, 0))
            while searches and not self.full() and not self.stopped and rounds < MAX_QUERY_ROUNDS:
                rounds += 1
                self.queries["rounds"][self.group] = rounds
                self._save_queries()
                fresh = await self.new_queries(searches[0])
                if not fresh:
                    break
                async for batch in self.candidates({**searches[0], "queries": fresh}):
                    await self.handle(batch)
                    if self.full() or self.stopped:
                        break
            await self.flush()
        finally:
            await asyncio.to_thread(self.note, True)
        return self.result()

    async def candidates(self, src: dict):
        """A source's items, a batch at a time (one search's, one part of a document's,
        one slice of a table's)."""
        kind = src["kind"]
        if kind == "file":
            try:
                _cols, rows = await asyncio.to_thread(fx.read_table, self.folder / src["path"])
            except (OSError, ValueError) as e:
                await self._event("funnel", f"couldn't read {src['path']}: {e}")
                return
            for i in range(0, len(rows), 25):
                yield [{"fields": r, "origin": f"{src['path']} row {i + j + 2}",
                        "text": "\n".join(f"{k}: {v}" for k, v in r.items() if v)}
                       for j, r in enumerate(rows[i:i + 25])]
        elif kind == "document":
            try:
                text = await asyncio.to_thread(fx.read_document, self.folder / src["path"])
            except (OSError, ValueError) as e:
                await self._event("funnel", f"couldn't read {src['path']}: {e}")
                return
            for n, part in enumerate(fx.chunks(text), start=1):
                yield await self.extract(part, f"{src['path']} part {n}")
        elif kind == "links":
            try:
                text = (self.folder / src["path"]).read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                await self._event("funnel", f"couldn't read {src['path']}: {e}")
                return
            found = fx.read_links(text)
            if src.get("as") == "pages":
                for link in found:
                    async for batch in self.candidates({"kind": "page", "url": link["url"], "group": src.get("group", "")}):
                        yield batch
            else:
                yield [{"fields": {"name": x["title"], "url": x["url"], "note": x["note"]}, "origin": x["url"],
                        "text": f"{x['title']}\n{x['url']}\n{x['note']}".strip()} for x in found]
        elif kind == "page":
            ok, text = await (await self._tools()).fetch(src["url"])
            await self._event("web_fetch", f"url={src['url']}")
            if not ok:
                return
            for n, part in enumerate(fx.chunks(text, 8000)[:6], start=1):
                yield await self.extract(part, src["url"])
        elif kind == "search":
            for q in src["queries"]:
                if self.full() or self.stopped:
                    return
                if q in self.queries["used"]:
                    continue
                if self.queries["searches"] >= self.budget:
                    self.stopped = f"the run's {self.budget} searches are used"
                    return
                ok, text = await (await self._tools()).search(q, src.get("count", 20), src.get("country", ""))
                self.queries["searches"] += 1
                self.queries["used"].append(q)
                self._save_queries()
                await self._event("web_search", f"query={q}")
                if not ok:
                    continue
                yield await self.extract(text, f"search: {q}", about=src.get("about", ""))

    async def extract(self, text: str, origin: str, about: str = "") -> list[dict]:
        """The items in a text (a search's results, a page, a part of a document), by a
        short model call. Links to more list pages are kept for later."""
        cols = ", ".join(self.recipe["columns"])
        got = await self._ask("extract", EXTRACT_SYSTEM.format(item=self.recipe["item"]),
                              f"# Looking for\n{self.recipe['item']}s"
                              + (f" ({self.group})" if self.group else "") + (f": {about}" if about else "")
                              + f"\n\n# Columns of the list\n{cols}\n\n# Text ({origin})\n{text[:12000]}")
        items = got.get("items") if isinstance(got, dict) else got if isinstance(got, list) else []
        if isinstance(got, dict):
            for url in got.get("pages") or []:
                url = str(url).strip()
                if url.startswith(("http://", "https://")) and url not in self.more_pages \
                        and len(self.more_pages) < MORE_PAGES:
                    self.more_pages.append(url)
        results = fx.search_results(text) if origin.startswith("search: ") else []
        out = []
        for it in items or []:
            if not isinstance(it, dict) or not str(it.get("name") or "").strip():
                continue
            fields = {str(k): str(v) for k, v in (it.get("fields") or {}).items() if v not in (None, "")}
            fields.setdefault("name", str(it["name"]).strip())
            if it.get("url"):
                fields.setdefault("url", str(it["url"]).strip())
            own = _own_result(results, fields["name"], str(it.get("url") or ""))
            out.append({"fields": fields, "origin": origin,
                        "text": own or (text if len(text) <= 3000 else "")})
        return out

    def item_key(self, item: dict) -> str:
        """What names an item: its key column, else its own site, else its name. A link
        on a directory or a social site (its LinkedIn page) is never its own site."""
        f = {k.lower(): str(v or "") for k, v in item["fields"].items()}
        key = self.recipe["key"]
        site = self.recipe.get("site_column") or ""

        def own(v: str) -> str:
            d = fx.domain_of(v) if re.match(r"^(https?://|www\.)", v.strip(), re.I) else ""
            return "" if d and any(fx.same_site(d, s) for s in self.skip_sites) else v

        v = own(f.get(key.lower(), "")) if key and key != "domain" else ""
        v = v or own(f.get(site.lower(), "")) or own(f.get("url", "")) or own(f.get("website", "")) or f.get("name", "")
        return norm_key(v)

    def passes(self, item: dict, found: dict[str, list[tuple[str, str]]] | None = None) -> str:
        """"" when an item passes the recipe's filters, else why not. Before gathering
        only the item's fields and its listing are known: one that places it outside
        the US (for a US list) goes before any page is read; a size band is tested once
        it's found."""
        f = {k.lower(): v for k, v in item["fields"].items()}
        if found is None and self.places:
            where = fx.foreign_place(f"{item.get('text') or ''}\n" + " ".join(map(str, f.values())), self.places)
            if where:
                return f"not in {' or '.join(fx.place_phrase(c) for c in self.places)} (its listing says {where})"
        for flt in self.recipe["filters"]:
            col, op, val = flt["column"], flt["op"], flt.get("value")
            if op in ("size_max", "size_min"):
                band = (found or {}).get("size_band") or []
                if not band:
                    continue
                low, _, high = band[0][0].partition("-")
                lo, hi = int(re.sub(r"\D", "", low) or 0), int(re.sub(r"\D", "", high) or 10**9)
                try:
                    limit = int(val)
                except (TypeError, ValueError):
                    continue
                if op == "size_max" and lo > limit:
                    return f"{band[0][0]} people ({band[0][1]})"
                if op == "size_min" and hi < limit:
                    return f"{band[0][0]} people ({band[0][1]})"
                continue
            if found is not None:
                continue  # field filters ran before gathering
            text = " ".join(f.values()) if col == "*" else f.get(col.lower(), "")
            try:
                if op == "regex" and not re.search(str(val), text, re.I):
                    return f"{col} doesn't match"
                if op == "not_regex" and re.search(str(val), text, re.I):
                    return f"{'it' if col == '*' else col} matches what's excluded"
                if op == "in" and text and text.lower() not in {str(x).lower() for x in val or []}:
                    return f"{col} is {text}"
                if op == "not_in" and text.lower() in {str(x).lower() for x in val or []}:
                    return f"{col} is {text}"
                if op in ("min", "max") and text:
                    n = float(re.sub(r"[^\d.]", "", text) or "nan")
                    if (op == "min" and n < float(val)) or (op == "max" and n > float(val)):
                        return f"{col} is {text}"
            except (re.error, TypeError, ValueError):
                continue
        return ""

    async def handle(self, items: list[dict]) -> None:
        wave: list[dict] = []
        for item in items:
            if self.full() or self.stopped:
                break
            if self.counts["looked"] >= self.items_max:
                self.stopped = f"{self.items_max} items looked at"
                break
            key = self.item_key(item)
            if not key or key in self.seen:
                continue
            if key in self.known:  # checked in one of the factory's earlier runs
                self.seen.add(key)
                self.counts["known"] += 1
                continue
            if self.recipe.get("site_column") and any(fx.same_site(key, s) for s in self.skip_sites):
                continue  # items with sites of their own: a directory or a social page isn't one
            item["key"] = key
            self.seen.add(key)
            self.counts["looked"] += 1
            why = self.passes(item)
            if why:
                await self.record(item, "dropped", why, stage="filter")
                continue
            wave.append(item)
            if len(wave) >= CONCURRENCY:
                await self._gather_wave(wave)
                wave = []
        if wave:
            await self._gather_wave(wave)

    async def _gather_wave(self, wave: list[dict]) -> None:
        await asyncio.gather(*(self.gather(it) for it in wave))
        for it in wave:
            why = self.passes(it, it.get("found") or {})
            if why:
                await self.record(it, "dropped", why, stage="read")
                continue
            self.pending.append(it)
            if len(self.pending) >= self.recipe.get("decide_batch", 1):
                await self.flush()

    async def gather(self, item: dict) -> None:
        """What the recipe reads about an item, step by step, and the facts code finds
        in it. After each step the cheap tests run again (a size band from a search
        snippet), so an item that fails one isn't read any further."""
        evidence = [{"source": item["origin"], "text": item.get("text") or
                     "\n".join(f"{k}: {v}" for k, v in item["fields"].items() if v)}]
        site = self.recipe.get("site_column") or ""
        if site and not self._site_of(item) and any(g["kind"] == "fetch" for g in self.recipe["gather"]):
            await self.find_site(item, evidence)
        item["evidence"], item["found"] = evidence, self.facts(item, evidence)
        for g in self.recipe["gather"]:
            if self.passes(item, item["found"]):
                return  # it already fails a test: nothing more to read
            if g["kind"] == "search":
                q = render(g["query"], item["fields"], site)
                if not q or self.queries["searches"] >= self.budget:
                    continue
                ok, text = await (await self._tools()).search(q, g.get("count", 5))
                self.queries["searches"] += 1
                self._save_queries()
                await self._event("web_search", f"query={q}")
                if ok:
                    for r in fx.search_results(text):
                        evidence.append({"source": r["url"], "text": r["text"]})
            elif g["kind"] == "fetch":
                base = render(g["url"], item["fields"], site)
                if not base:
                    continue
                if not base.startswith(("http://", "https://")):
                    base = "https://" + base.lstrip("/")
                parts = urlsplit(base)
                root = f"{parts.scheme}://{parts.netloc}/"
                urls = list(dict.fromkeys(base if not p else urljoin(root, p.lstrip("/")) for p in g["paths"]))
                tools = await self._tools()
                got = await asyncio.gather(*(tools.fetch(u) for u in urls))
                for u, (ok, text) in zip(urls, got):
                    await self._event("web_fetch", f"url={u}")
                    if ok and text.strip():
                        evidence.append({"source": u, "text": text[:FETCH_CHARS]})
            item["found"] = self.facts(item, evidence)

    def _site_of(self, item: dict) -> str:
        site = (self.recipe.get("site_column") or "").lower()
        f = {k.lower(): str(v or "").strip() for k, v in item["fields"].items()}
        for k in (site, "website", "url", "site", "homepage", "domain"):
            v = f.get(k, "")
            d = fx.domain_of(v) if v else ""
            if d and "." in d and not any(fx.same_site(d, s) for s in self.skip_sites):
                return v
        return ""

    async def find_site(self, item: dict, evidence: list[dict]) -> None:
        """An item found on a directory or a LinkedIn page has no site of its own to
        read: one search for its name, and the first result on a site of its own whose
        title or address carries the name's words."""
        f = item["fields"]
        name = str(f.get("name") or f.get(self.recipe["key"]) or next(iter(f.values()), "")).strip()
        if not name or self.queries["searches"] >= self.budget:
            return
        words = [w for w in re.findall(r"[a-z0-9]+", name.lower())
                 if len(w) >= 3 and w not in {"the", "and", "inc", "llc", "ltd", "group", "partners", "company",
                                              "consulting", "co", "corp", "firm", "associates", "services"}]
        place = fx.place_name(self.places[0]) if len(self.places) == 1 else ""
        kind = self.recipe["item"] if self.recipe["item"] != "item" else ""
        q = " ".join(f'"{name}" {kind} {place}'.split())
        ok, text = await (await self._tools()).search(q, 8)
        self.queries["searches"] += 1
        self._save_queries()
        await self._event("web_search", f"query={q}")
        if not ok:
            return
        for r in fx.search_results(text):
            d = fx.domain_of(r["url"])
            if not d or any(fx.same_site(d, s) for s in self.skip_sites):
                continue
            where = f"{d} {r['title']}".lower()
            if words and not any(w in where for w in words):
                continue
            f[self.recipe["site_column"]] = f"https://{d}"
            evidence.append({"source": r["url"], "text": r["text"]})
            return

    def facts(self, item: dict, evidence: list[dict]) -> dict[str, list[tuple[str, str]]]:
        """What code finds in what was read: each extractor's values, with the page."""
        site = self.recipe.get("site_column") or ""
        domain = fx.domain_of(next((v for k, v in item["fields"].items()
                                    if k.lower() in (site.lower(), "url", "website") and v), ""))
        names = list(self.recipe["extract"])
        if any(f["op"] in ("size_max", "size_min") for f in self.recipe["filters"]) and "size_band" not in names:
            names.append("size_band")
        found: dict[str, list[tuple[str, str]]] = {}
        for e in evidence:
            for name in names:
                for v in fx.extract(name, e["text"], domain=domain if name == "emails" else ""):
                    found.setdefault(name, [])
                    if all(v != x for x, _ in found[name]):
                        found[name].append((v, e["source"]))
        return found

    async def flush(self) -> None:
        """Judge what's waiting, in one call."""
        if not self.pending:
            return
        batch, self.pending = self.pending, []
        lines = []
        for i, it in enumerate(batch, start=1):
            facts = [f"  {name}: " + "; ".join(f"{v} [{src}]" for v, src in vals[:8])
                     for name, vals in (it.get("found") or {}).items()]
            share = READ_CHARS // max(1, len(it.get("evidence") or []))
            excerpt = "\n".join(f"  [{e['source']}]\n  " + " ".join(e["text"].split())[:share]
                                for e in it.get("evidence") or [])
            lines.append(f"## id {i}\nfields: {json.dumps(it['fields'], ensure_ascii=False)[:1500]}\n"
                         + ("facts Captain found:\n" + "\n".join(facts) + "\n" if facts else "")
                         + f"what was read:\n{excerpt}")
        cols = "\n".join(f"- {c}" + (f": {self.recipe['about'][c]}" if self.recipe["about"].get(c) else "")
                         for c in self.recipe["columns"])
        groups = (f"\n# {self.recipe['group_by']}\n" + ", ".join(self.recipe["groups"])
                  + (f"\nThis step looks for: {self.group}" if self.group else "")) if self.recipe["groups"] else ""
        try:
            got = await self._ask("decide", DECIDE_SYSTEM,
                                  f"# Rules\n{self.recipe['decide'] or '(take every item that fits the list)'}\n\n"
                                  + (f"# The goal, in the owner's words (what each column takes)\n{self.charter}\n\n"
                                     if self.charter else "")
                                  + (f"# Reference\n{self.reference}\n\n" if self.reference else "")
                                  + f"# Columns\n{cols}{groups}\n\n# Items\n" + "\n\n".join(lines), items=len(batch))
        except Exception as e:  # noqa: BLE001 — these go back for the next call
            log.warning("funnel decide call failed", run_id=self.run["id"], error=str(e))
            got = None
        verdicts = {int(v.get("id")): v for v in got if isinstance(v, dict) and str(v.get("id", "")).isdigit()} \
            if isinstance(got, list) else {}
        for i, it in enumerate(batch, start=1):
            v = verdicts.get(i)
            if v is None:
                await self.record(it, "dropped", "the judging call gave no answer for it")
                continue
            verdict = str(v.get("verdict") or "").lower()
            reason = " ".join(str(v.get("reason") or "").split())[:200]
            if verdict == "keep":
                row = {c: " ".join(str((v.get("row") or {}).get(c) or "").split()) for c in self.recipe["columns"]}
                problem = self.check(row, it)
                if problem:
                    await self.record(it, "dropped", f"check: {problem}", stage="check")
                    continue
                group = row.get(self.recipe["group_by"], "") if self.recipe["group_by"] else ""
                group = next((g for g in self.recipe["groups"] if g.lower() == group.lower()), group) or self.group
                if self.quota(group) and self.kept.get(group, 0) >= self.quota(group):
                    await self.record(it, "dropped", f"{group} is already full", group=group, stage="full")
                    continue
                key = norm_key(row.get(self.recipe["key"], "")) if self.recipe["key"] in row else it["key"]
                if key and key != it["key"] and (key in self.seen or key in self.known):
                    await self.record(it, "dropped", "already looked at under another name"
                                      + (" in an earlier run" if key in self.known else ""), stage="duplicate")
                    continue
                self.seen.add(key)
                self.kept[group] = self.kept.get(group, 0) + 1
                _append_csv(self.folder / ROWS, self.recipe["columns"] + [_GROUP_COL], {**row, _GROUP_COL: group})
                await self.record(it, "kept", reason or "meets the rules", group=group)
            elif verdict == "maybe":
                hint = str((v.get("row") or {}).get(self.recipe["group_by"]) or "") if self.recipe["group_by"] else ""
                group = self.group or next((g for g in self.recipe["groups"] if g.lower() == hint.lower()), "")
                await self.record(it, "maybe", reason or "a fact it needs wasn't found", group=group)
                name = it["fields"].get("name") or next(iter(it["fields"].values()), "")
                _append_csv(self.folder / MAYBE, _MAYBE_COLS,
                            {"key": it["key"], "name": name, "group": group,
                             "fields": json.dumps(it["fields"], ensure_ascii=False),
                             "why": reason, "sources": " ".join(e["source"] for e in it.get("evidence") or [])})
            else:
                await self.record(it, "dropped", reason or "breaks a rule")

    def check(self, row: dict[str, str], item: dict) -> str:
        """Why a row the model kept fails the recipe's checks by code, or ""."""
        return check_row(self.recipe, row, {e["source"]: e["text"] for e in item.get("evidence") or []})

    async def record(self, item: dict, status: str, reason: str, group: str = "", stage: str = "") -> None:
        self.counts["kept" if status == "kept" else status if status == "maybe" else "dropped"] += 1
        if status == "dropped":
            short = re.sub(r"\(.*?\)|'[^']*'|\d[\d,.-]*", "", reason).strip()[:60]
            self.reasons[short] = self.reasons.get(short, 0) + 1
        name = item["fields"].get("name") or next(iter(item["fields"].values()), "") or item.get("key", "")
        _append_csv(self.folder / ITEMS, _ITEM_COLS,
                    {"key": item.get("key", ""), "name": str(name)[:120], "group": group or self.group,
                     "status": status, "stage": stage or (status if status in ("kept", "maybe") else "judge"),
                     "reason": reason, "source": item.get("origin", ""), "step": self.key,
                     "at": time.strftime("%Y-%m-%d %H:%M")})
        line = f"{status} {str(name)[:60]}" + (f" ({group})" if group and not self.group else "") + f": {reason}"
        self.recent = (self.recent + [line])[-8:]
        await self._event("decide", line)
        if (self.counts["kept"] + self.counts["maybe"] + self.counts["dropped"]) % NOTE_EVERY == 0:
            await asyncio.to_thread(self.note)

    async def new_queries(self, src: dict) -> list[str]:
        """More searches for a group that's still short, from what worked so far."""
        if self.queries["searches"] >= self.budget:
            return []
        items = _read_csv(self.folder / ITEMS)
        worked: dict[str, int] = {}
        for r in items:
            if r.get("status") == "kept" and r.get("source", "").startswith("search: "):
                q = r["source"][8:]
                worked[q] = worked.get(q, 0) + 1
        used = [q for q in self.queries["used"] if q in src.get("queries", []) or q in worked]
        got = await self._ask("queries", "Write new web searches for a list that is still short. Reply ONLY with "
                                         'JSON: {"queries": ["...", ...]} (10-15 new ones, none already used). '
                                         "Keep your thinking short.",
                              f"# Looking for\n{self.recipe['item']}s" + (f" ({self.group})" if self.group else "")
                              + (f": {src.get('about')}" if src.get("about") else "")
                              + f"\n\n# Rules\n{self.recipe['decide'][:1500]}\n\n# Searches that found kept items\n"
                              + ("\n".join(f"- {q} ({n})" for q, n in sorted(worked.items(), key=lambda x: -x[1])[:15])
                                 or "(none yet)")
                              + "\n\n# Already used\n" + "\n".join(f"- {q}" for q in used[-40:]))
        fresh = got.get("queries") if isinstance(got, dict) else got if isinstance(got, list) else []
        out = [str(q).strip()[:300] for q in fresh or [] if str(q).strip() and str(q).strip() not in self.queries["used"]]
        if out:
            await self._event("funnel", f"{len(out)} new searches for {self.group or 'the list'}")
        return out[:15]

    def note(self, final: bool = False) -> None:
        """The step's progress note, as Bat's workers keep one."""
        from captain_claw.flight_deck import bat_routes
        lines = [f"# {self.step.get('title') or self.key}", "",
                 f"Status: {'finished' if final else 'working'}" + (f" (stopped: {self.stopped})" if self.stopped else "")]
        if self.recipe["groups"]:
            lines += ["", "Kept so far:"] + [f"- {g}: {self.kept.get(g, 0)}" + (f" of {n}" if n else "")
                                             for g, n in self.recipe["groups"].items()]
        else:
            total = sum(self.kept.values())
            target = int(self.recipe.get("target") or 0)
            lines += ["", f"Kept so far: {total}" + (f" of {target}" if target else "")]
        c = self.counts
        lines += ["", f"This step looked at {c['looked']} items: {c['kept']} kept, {c['maybe']} maybe, "
                      f"{c['dropped']} dropped. Searches used in the run: {self.queries['searches']} of {self.budget}."
                  + (f" Skipped {c['known']} that earlier runs already checked." if c["known"] else "")]
        if self.reasons:
            top = sorted(self.reasons.items(), key=lambda x: -x[1])[:6]
            lines += ["", "Dropped most often for:"] + [f"- {r or 'other'} ({n})" for r, n in top]
        if self.recent:
            lines += ["", "Latest:"] + [f"- {r}" for r in self.recent]
        p = self.folder / "progress" / f"{bat_routes._step_file(self.key)}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            write_counts(self.folder, self.recipe)
        except OSError:
            log.debug("funnel: couldn't write its counts", exc_info=True)

    def result(self) -> dict:
        c = self.counts
        if self.group:
            q = self.quota(self.group)
            have = self.kept.get(self.group, 0)
            head = f"{self.group}: {have}" + (f" of {q}" if q else "") + " kept."
            short = bool(q) and have < q
        else:
            have = sum(self.kept.values())
            target = int(self.recipe.get("target") or 0)
            head = f"{have} kept" + (f" of {target}" if target else "") + "."
            short = not self.full() and bool(self.recipe["groups"] or target)
        why_short = ""
        if short:
            why_short = (f" Short: {self.stopped}." if self.stopped
                         else " Short: its sources ran out." if not self.full() else "")
        top = ", ".join(f"{r or 'other'} ({n})" for r, n in sorted(self.reasons.items(), key=lambda x: -x[1])[:4])
        out = (f"{head}{why_short} Looked at {c['looked']} items: {c['kept']} kept, {c['maybe']} maybe "
               f"(in {MAYBE}), {c['dropped']} dropped" + (f", most often for: {top}" if top else "") + ". "
               + (f"Skipped {c['known']} that earlier runs already checked. " if c["known"] else "")
               + f"Searches used in the run: {self.queries['searches']} of {self.budget}.")
        return {"ok": True, "output": out, "usd": 0.0, "tokens": 0, "error": ""}


# ── checks ─────────────────────────────────────────────────────────────

def check_row(recipe: dict[str, Any], row: dict[str, str], texts: dict[str, str], *,
              unsourced_ok: bool = False) -> str:
    """Why a row fails the recipe's checks by code, or "". ``texts`` is what was read,
    by link. A value that must be in its source is looked for on the page its source
    column names (any page read, when the check names no source column; skipped then
    with ``unsourced_ok``, for rows whose pages Captain didn't read itself)."""
    every = "\n".join(texts.values())
    for c in recipe["checks"]:
        col, op = c["column"], c["op"]
        v = row.get(col, "")
        if op == "required":
            if not v:
                return f"{col} is empty"
            continue
        if not v:
            continue
        if op == "in" and v.lower() not in {str(x).lower() for x in c.get("value") or []}:
            return f"{col} '{v}' isn't one of the allowed values"
        if op == "regex":
            try:
                if not re.search(str(c.get("value")), v, re.I):
                    return f"{col} '{v}' doesn't match"
            except re.error:
                continue
        if op == "in_evidence":
            src = row.get(c.get("source_column", ""), "") if c.get("source_column") else ""
            if not src and unsourced_ok:
                continue
            where = texts.get(src) if src and src in texts else None
            if where is None and src:
                where = next((t for s, t in texts.items() if s.rstrip("/") == src.rstrip("/")), None)
            if src and where is None:
                return f"{col}'s source {src} isn't a page Captain read"
            if not fx.in_text(v, where if where is not None else every):
                return f"{col} '{v[:60]}' isn't in " + (src if src else "what was read")
        if op == "within_months":
            ago = fx.months_ago(v[:10])
            try:
                limit = float(c.get("value"))
            except (TypeError, ValueError):
                continue
            if ago is None:
                return f"{col} '{v}' isn't a date"
            if ago > limit or ago < -1:
                return f"{col} {v} is outside the last {int(limit)} months"
        if op in ("min", "max"):
            try:
                n, lim = float(re.sub(r"[^\d.-]", "", v)), float(c.get("value"))
            except (TypeError, ValueError):
                continue
            if (op == "min" and n < lim) or (op == "max" and n > lim):
                return f"{col} {v} is {'under' if op == 'min' else 'over'} {c.get('value')}"
    return ""


def kept_per_group(folder: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in _read_csv(folder / ROWS):
        g = r.get(_GROUP_COL, "")
        out[g] = out.get(g, 0) + 1
    return out


def shortfall(folder: Path, recipe: dict[str, Any]) -> dict[str, int]:
    """How many more each group needs (``""``: the whole list, when it has no groups)."""
    kept = kept_per_group(folder)
    if recipe["groups"]:
        return {g: n - kept.get(g, 0) for g, n in recipe["groups"].items() if n and kept.get(g, 0) < n}
    target = int(recipe.get("target") or 0)
    total = sum(kept.values())
    return {"": target - total} if target and total < target else {}


def counts(folder: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    """How items went, per group and stage, for the run panel and the summary."""
    groups: dict[str, dict[str, int]] = {}
    for r in _read_csv(folder / ITEMS):
        g = groups.setdefault(r.get("group", ""), {"looked": 0, **{st: 0 for st in STAGES}})
        g["looked"] += 1
        stage = r.get("stage") or ("kept" if r.get("status") == "kept" else "maybe" if r.get("status") == "maybe"
                                   else "judge")
        if stage in g:
            g[stage] += 1
    kept = kept_per_group(folder)
    for name, quota in (recipe["groups"] or {}).items():
        groups.setdefault(name, {"looked": 0, **{st: 0 for st in STAGES}})
    for name, g in groups.items():
        g["quota"] = int((recipe["groups"] or {}).get(name) or 0)
        g["on_list"] = kept.get(name, 0)
    try:
        searches = int(json.loads((folder / QUERIES).read_text(encoding="utf-8")).get("searches") or 0)
    except (OSError, ValueError):
        searches = 0
    return {"groups": groups, "target": int(recipe.get("target") or 0), "searches": searches,
            "updated": time.strftime("%Y-%m-%d %H:%M")}


def live_counts(folder: Path) -> dict[str, Any] | None:
    """A going run's counts for its panel: as its last progress note left them, or
    worked out now; None before it has a recipe."""
    try:
        got = json.loads((folder / COUNTS).read_text(encoding="utf-8"))
        if isinstance(got, dict) and got.get("groups") is not None:
            return got
    except (OSError, ValueError):
        pass
    recipe = load_recipe(folder)
    return counts(folder, recipe) if recipe else None


def write_counts(folder: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    got = counts(folder, recipe)
    (folder / COUNTS).parent.mkdir(parents=True, exist_ok=True)
    (folder / COUNTS).write_text(json.dumps(got, indent=1), encoding="utf-8")
    return got


# ── finishing what's short (an agent step) ─────────────────────────────

def leftovers_on(run: dict) -> bool:
    """Whether a group still short gets a finishing agent step (``config.funnel_leftovers``
    ``agent``, the default, or ``none``)."""
    return str((run.get("config") or {}).get("funnel_leftovers") or "agent").lower() != "none"


def agent_context(run: dict) -> int:
    """The context a funnel run's agent steps get (the finishing step, the fix rounds'),
    so their calls stay small; 0 for a run that isn't the funnel's."""
    cfg = run.get("config") or {}
    if cfg.get("engine") != ENGINE:
        return 0
    return _int(cfg.get("funnel_agent_context"), 16_000, 1_000_000, AGENT_CONTEXT)


def prepare_finish(run: dict, step: dict) -> dict:
    """The finishing step's work: ``{"work": text}`` for its agent, or ``{"result": ...}``
    when there's nothing for it (nothing short, nothing nearly there, or the run's
    settings leave a short group as it is)."""
    from captain_claw.flight_deck import bat_routes

    def done(text: str) -> dict:
        return {"result": {"ok": True, "output": text, "usd": 0.0, "tokens": 0, "error": ""}}

    folder = bat_routes._run_folder(run)
    recipe = load_recipe(folder)
    if recipe is None:
        return done("Nothing to finish: the funnel's recipe is missing.")
    if not leftovers_on(run):
        return done("Nothing to finish: the run's settings leave a short group as it is.")
    short = shortfall(folder, recipe)
    if not short:
        return done("Nothing to finish: no group is short.")
    on_list = {norm_key(r.get(recipe["key"], "")) for r in _read_csv(folder / ROWS)} if recipe["key"] else set()
    items = [m for m in _read_csv(folder / MAYBE)
             if (m.get("group", "") in short or not m.get("group") or "" in short) and m.get("key") not in on_list]
    items.sort(key=lambda m: -short.get(m.get("group", ""), 0))
    items = items[:FINISH_MAX]
    words = "; ".join(f"{g or 'the list'} {n} more" for g, n in short.items())
    if not items:
        return done(f"Nothing nearly there to finish ({words}). What's still short goes to the judge's "
                    "fix rounds.")
    target = folder / FINISH_ITEMS
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_MAYBE_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)
    finish = folder / FINISH
    if not finish.is_file():
        with finish.open("w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(recipe["columns"])
    project = bat_routes._project_of(run)
    sourced = [c for c in recipe["checks"] if c["op"] == "in_evidence" and c.get("source_column")]
    lines = [
        "# What to work on",
        f"The list funnel went through its sources. Still short: {words}.",
        f"{len(items)} items came close: each seems to fit, but a fact the rules need wasn't found. They are in "
        f"vfs:{project}/{FINISH_ITEMS} (with why, and the pages already read). For each, the shortest group first:",
        "1. find what was missing (search; read the item's own pages);",
        f"2. if it meets every rule in shared/conventions.md, add its row to vfs:{project}/{FINISH}, with the "
        "columns exactly as in its first line, one row per item;",
    ]
    if sourced:
        lines.append("3. in " + ", ".join(sorted({c["source_column"] for c in sourced}))
                     + " put the link of the exact page the value is on: Captain reads that page again and drops "
                     "a row whose value isn't there.")
    lines += ["Stop once the short groups are full. Work only on these items: don't look for new ones, don't add "
              f"one already on the list, and don't change any other file in {STATE}/.", "", "The items:"]
    for m in items:
        group = f" ({m['group']})" if m.get("group") else " (no group yet)" if recipe["groups"] else ""
        lines.append(f"- {m.get('name') or m.get('key')}{group}: {m.get('why') or ''}")
    return {"work": "\n".join(lines)}


# ── the merge ──────────────────────────────────────────────────────────

async def check_finish(run: dict, step: dict, folder: Path, recipe: dict[str, Any]) -> tuple[int, int, int]:
    """The rows the finishing agent added, checked as the funnel's own are: by the
    recipe's checks, with the pages their source columns cite read again. Rows that
    pass join the list; the rest go to the items dropped, with why. Returns
    ``(added, refused, unsourced)``: unsourced rows passed checks that couldn't look
    at a page (no source column to cite)."""
    rows = _read_csv(folder / FINISH)
    if not rows:
        return 0, 0, 0
    done = {r.get("key") for r in _read_csv(folder / ITEMS) if r.get("stage") in ("agent", "agent-check")}
    kept = kept_per_group(folder)
    on_list = {norm_key(r.get(recipe["key"], "")) for r in _read_csv(folder / ROWS)} if recipe["key"] else set()
    sources = sorted({c["source_column"] for c in recipe["checks"] if c["op"] == "in_evidence" and c.get("source_column")})
    unsourced_check = any(c["op"] == "in_evidence" and not c.get("source_column") for c in recipe["checks"])
    tools: Tools | None = None
    pages: dict[str, str] = {}
    added = refused = unsourced = 0
    try:
        for raw in rows:
            row = {c: " ".join(str(raw.get(c) or "").split()) for c in recipe["columns"]}
            key = norm_key(row.get(recipe["key"], "")) if recipe["key"] in row else ""
            key = key or norm_key(next((v for v in row.values() if v), ""))
            if not key or key in done or key in on_list:
                continue
            done.add(key)
            group = row.get(recipe["group_by"], "") if recipe["group_by"] else ""
            group = next((g for g in recipe["groups"] if g.lower() == group.lower()), group)
            name = next((v for v in row.values() if v), key)
            texts: dict[str, str] = {}
            for col in sources:
                url = row.get(col, "")
                if not url.startswith(("http://", "https://")):
                    continue
                if url not in pages:
                    if tools is None:
                        tools = make_tools(run, step)
                    ok, text = await tools.fetch(url)
                    pages[url] = text[:FETCH_CHARS * 4] if ok else ""
                if pages[url]:
                    texts[url] = pages[url]
            problem = check_row(recipe, row, texts, unsourced_ok=True)
            if not problem and recipe["groups"] and recipe["groups"].get(group) and kept.get(group, 0) >= recipe["groups"][group]:
                problem = f"{group} is already full"
            item = {"key": key, "name": str(name)[:120], "group": group, "source": "finishing step",
                    "step": step.get("step_key", ""), "at": time.strftime("%Y-%m-%d %H:%M")}
            if problem:
                refused += 1
                _append_csv(folder / ITEMS, _ITEM_COLS, {**item, "status": "dropped", "stage": "agent-check",
                                                         "reason": f"added by the finishing step, but {problem}"})
                continue
            added += 1
            if unsourced_check and not sources:
                unsourced += 1
            kept[group] = kept.get(group, 0) + 1
            on_list.add(key)
            _append_csv(folder / ROWS, recipe["columns"] + [_GROUP_COL], {**row, _GROUP_COL: group})
            _append_csv(folder / ITEMS, _ITEM_COLS, {**item, "status": "kept", "stage": "agent",
                                                     "reason": "added by the finishing step; its pages read again"})
    finally:
        if tools is not None:
            await tools.close()
    return added, refused, unsourced


def _md(s: Any) -> str:
    return " ".join(str(s or "").split()).replace("|", "\\|")


def write_list(folder: Path, recipe: dict[str, Any]) -> list[dict[str, str]]:
    """The kept rows as the list: its columns in order, no two with the same key."""
    cols = recipe["columns"]
    out, seen = [], set()
    for r in _read_csv(folder / ROWS):
        k = norm_key(r.get(recipe["key"], "")) if recipe["key"] in cols else ""
        if k and k in seen:
            continue
        if k:
            seen.add(k)
        out.append({c: r.get(c, "") for c in cols})
    target = folder / recipe["output"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    return out


def write_exclusions(folder: Path, recipe: dict[str, Any]) -> int:
    dropped = [r for r in _read_csv(folder / ITEMS) if r.get("status") == "dropped"]
    lines = [f"# Dropped {recipe['item']}s", "", f"{len(dropped)} looked at and left off the list, with why.", "",
             "| Item | Group | Where | Why | Found at |", "|---|---|---|---|---|"]
    lines += [f"| {_md(r.get('name') or r.get('key'))} | {_md(r.get('group'))} | {_md(r.get('stage'))} | "
              f"{_md(r.get('reason'))} | {_md(r.get('source'))} |" for r in dropped]
    (folder / EXCLUSIONS).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(dropped)


_STAGE_WORDS = {"filter": "ruled out by code", "read": "ruled out after reading", "judge": "judged out",
                "check": "failed a check", "full": "group already full", "duplicate": "duplicates"}


def write_summary(folder: Path, recipe: dict[str, Any], rows: int, dropped: int,
                  finished: tuple[int, int, int], budget: int) -> str:
    """``summary.md``: how many, how short and why, how items went, how rows were
    checked, and where everything is."""
    got = write_counts(folder, recipe)
    groups, target = got["groups"], got["target"]
    item = recipe["item"]
    lines = [f"# {recipe['output']}: summary", "",
             f"{rows} {item}{'' if rows == 1 else 's'} on the list" + (f" of {target} wanted" if target else "")
             + ". Built by the list funnel: code searched, read and checked; a model judged each item once.", ""]
    short = shortfall(folder, recipe)
    if recipe["groups"]:
        lines += ["## Per group", "", "| Group | On the list | Wanted | Short |", "|---|---|---|---|"]
        for g, n in recipe["groups"].items():
            have = groups.get(g, {}).get("on_list", 0)
            lines.append(f"| {_md(g)} | {have} | {n or '-'} | {short.get(g, 0) or '-'} |")
        lines.append("")
    if short:
        maybe = sum(1 for m in _read_csv(folder / MAYBE))
        why = (f"the run's {budget} searches were used" if got["searches"] >= budget
               else "the sources ran out of items that pass")
        lines += ["## Short", "", "; ".join(f"{g or 'the list'}: {n} more" for g, n in short.items())
                  + f". Why: {why}." + (f" {maybe} came close (`{MAYBE}`)." if maybe else ""), ""]
    lines += ["## How items went", "", "| Group | Looked at | " + " | ".join(_STAGE_WORDS.values())
              + " | Maybe | Kept | Added by the finishing step |", "|---" * (len(_STAGE_WORDS) + 5) + "|"]
    for g, c in groups.items():
        lines.append(f"| {_md(g) or '-'} | {c['looked']} | " + " | ".join(str(c.get(st, 0)) for st in _STAGE_WORDS)
                     + f" | {c.get('maybe', 0)} | {c.get('kept', 0)} | {c.get('agent', 0)} |")
    checks = [c for c in recipe["checks"] if c["op"] != "required"]
    lines += ["", "## Checked in every row, by code", ""]
    lines += [f"- {c['column']} {_check_words(c)}" for c in checks] or ["- (no checks beyond the required columns)"]
    added, refused, unsourced = finished
    if added or refused:
        lines += ["", f"The finishing step added {added} row{'' if added == 1 else 's'} that passed the same checks, "
                  "their cited pages read again" + (f" ({unsourced} with no page to cite)" if unsourced else "")
                  + f"; {refused} it added didn't pass and were dropped."]
    known = len(known_keys(folder))
    if known:
        lines += ["", f"{known} items the factory's earlier runs already checked weren't looked at again "
                  f"(`extra/{LEDGER_FILE}`)."]
    src = [f"`{s['path']}`" if s.get("path") else s.get("url") or f"{len(s.get('queries') or [])} planned searches"
           for s in recipe["sources"]]
    lines += ["", "## Sources", "", ", ".join(src) + f". Searches made: {got['searches']} of {budget}.", "",
              "## Files", "", f"- `{recipe['output']}`: the list",
              f"- `{EXCLUSIONS}`: every {item} dropped, where and why ({dropped})",
              f"- `{ITEMS}`: everything looked at", f"- `{MAYBE}`: the ones that came close",
              f"- `{RECIPE}`: the recipe the run followed"]
    text = "\n".join(lines) + "\n"
    (folder / SUMMARY).write_text(text, encoding="utf-8")
    return text


async def merge(run: dict, step: dict, folder: Path, recipe: dict[str, Any]) -> dict:
    """The list, from the funnel's rows and the finishing step's that pass its checks;
    ``exclusions.md``; ``summary.md``."""
    finished = await check_finish(run, step, folder, recipe)
    out = await asyncio.to_thread(write_list, folder, recipe)
    dropped = await asyncio.to_thread(write_exclusions, folder, recipe)
    budget = _int((run.get("config") or {}).get("funnel_search_budget"), 0, 100_000, SEARCH_BUDGET)
    await asyncio.to_thread(write_summary, folder, recipe, len(out), dropped, finished, budget)
    per = ""
    if recipe["groups"]:
        kept = kept_per_group(folder)
        per = " " + "; ".join(f"{g} {kept.get(g, 0)}" + (f" of {n}" if n else "") for g, n in recipe["groups"].items()) + "."
    added, refused, _ = finished
    text = (f"`{recipe['output']}`: {len(out)} rows" + (f" of {recipe['target']}" if recipe.get("target") else "")
            + f".{per}" + (f" The finishing step added {added} (and {refused} that failed a check)." if added or refused else "")
            + f" `{EXCLUSIONS}`: {dropped} dropped, with why. `{SUMMARY}`: how it went.")
    return {"ok": True, "output": text, "usd": 0.0, "tokens": 0, "error": ""}


def judge_note(run: dict) -> str:
    """For the judge's reviewers: what code already checked in every row of a funnel
    run's list, so they weigh the rules no check covers. "" for other runs."""
    if (run.get("config") or {}).get("engine") != ENGINE:
        return ""
    from captain_claw.flight_deck import bat_routes
    folder = bat_routes._run_folder(run)
    recipe = load_recipe(folder)
    if recipe is None:
        return ""
    checks = [c for c in recipe["checks"] if c["op"] != "required"]
    got = counts(folder, recipe)
    looked = sum(g["looked"] for g in got["groups"].values())
    kept = sum(g.get("kept", 0) + g.get("agent", 0) for g in got["groups"].values())
    lines = ["## The list funnel's own checks",
             f"Every row on `{recipe['output']}` passed these, by code, before it was kept"
             + (" (rows the finishing step added had their cited pages read again):" if any(
                 g.get("agent", 0) for g in got["groups"].values()) else ":")]
    lines += [f"- {c['column']} {_check_words(c)}" for c in checks] or ["- the required columns are filled"]
    lines.append(f"It looked at {looked} items and kept {kept}; `{EXCLUSIONS}` says why each other one went. "
                 "Rules these checks don't cover are yours to judge.")
    return "\n".join(lines)


async def attempt(run: dict, step: dict) -> dict:
    """A funnel step, in place of a Bat worker's turn."""
    from captain_claw.flight_deck import bat_routes
    kind = step_kind(run, step) or {}
    folder = bat_routes._run_folder(run)
    recipe = load_recipe(folder)
    if recipe is None:
        return {"ok": False, "error": f"the list funnel's recipe ({RECIPE}) is missing", "usd": 0.0, "tokens": 0}
    if fix_output(recipe, str(run.get("task", ""))):  # a recipe written before the charter's name was read
        (folder / RECIPE).write_text(json.dumps(recipe, indent=1, ensure_ascii=False), encoding="utf-8")
    if kind.get("kind") == "merge":
        return await merge(run, step, folder, recipe)
    work = Work(run, step, recipe, folder, kind.get("group", ""))
    try:
        return await work.run_step()
    except RuntimeError as e:
        return {"ok": False, "error": str(e), "usd": 0.0, "tokens": 0}
    finally:
        await work.close()
