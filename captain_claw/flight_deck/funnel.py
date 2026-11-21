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
import unicodedata
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
HANDOVER = f"{STATE}/handover.csv"  # items one group's step found that fit another group
LOOSE = f"{STATE}/loose.csv"      # rows that meet the soft rules only by their looser reading
QUERIES = f"{STATE}/queries.json"  # searches made, queries used, new-query rounds
FINISH = f"{STATE}/finish.csv"            # rows the finishing agent adds, checked at the merge
FINISH_ITEMS = f"{STATE}/finish-items.csv"  # the nearly-there items it's given
COUNTS = f"{STATE}/counts.json"    # per group: how items went, stage by stage (for the run panel)
TAPE_IN = "extra/tape"             # a recording to replay instead of the web (and its recipe, its answers)
TAPE_OUT = f"{STATE}/tape"         # what a recorded run searched and read
TAPE_MISSES = f"{STATE}/tape-misses.json"
BRIEF = f"{STATE}/brief.md"        # what the owner's documents and links say, read before the run
BRIEF_LINKS = 8          # pages read for it: the charter's links, a short links file's, the documents' own
BRIEF_DOCS = 6           # documents read for it; a larger pile is the list's material, not reading matter
BRIEF_PAGE_CHARS = 6000  # what the brief's call reads of each page or document
BRIEF_CHARS = 3000       # what a brief may be
SUMMARY = "summary.md"
EXCLUSIONS = "exclusions.md"
CLOSE_SUFFIX = "-came-close"  # <the list>-came-close.csv: the items that nearly made it, a result of its own
LEDGER_FILE = "already-checked.csv"  # in extra/: what the factory's earlier runs checked (first_mate_ledger)

SEARCH_BUDGET = 1000        # a run's searches, at least (config funnel_search_budget sets it outright)
SEARCHES_PER_ROW = 12       # and this many for each row the list asks for, when that is more
LOOKS_PER_ROW = 2           # items looked into per step, for each row it wants, when more than LOOK_MAX
FETCH_CHARS = 40_000  # what code keeps of one page: the whole of it, its footer (contact, hours) too
CONCURRENCY = 4
MAX_QUERY_ROUNDS = 40   # rounds of new searches a step may write (config funnel_query_rounds)
QUERY_ROUNDS_DRY = 2    # such rounds in a row that put no row on the list, before the step gives up
ITEMS_MAX = 5000     # items looked at per step (config funnel_items_max)
READ_CHARS = 12_000  # what one item's judging call reads of its pages, in all
MORE_PAGES = 10      # list pages found in search results, followed per step
DECIDE_TOKENS = 8192
# What each funnel call may write. Its calls run with thinking off (a JSON answer and
# nothing more): with it on, Qwen thought 160k characters on the setup call, looped,
# and never answered, twice. A small limit also keeps a loop from running for minutes
# on the tier's whole output.
OUT_TOKENS = {"setup": 6000, "extract": 6000, "queries": 2000, "decide": 2500, "owner": 3000, "look": 300, "group": 200, "columns": 600, "brief": 1200}
LOOK_STEPS = 4              # what one look into a nearly-there item may do (config funnel_look_steps)
LOOK_MAX = 60               # items looked into per step (config funnel_look_max)
LOOK_FILES = 400            # the owner's files a look is told of, by name
LISTED_WHOLE = 60           # lines of the folder shown to the setup with first rows and first lines
LISTED_NAMES = 400          # files past those, shown by name only
SOURCES_MAX = 400           # files one pattern in a recipe's source may stand for
JOIN_ROWS = 10              # rows of the owner's other tables one item may be given
JOIN_TABLE_ROWS = 5000      # a table larger than this isn't looked items up in
DECIDE_PER_ITEM = 600       # more room for a judging call with several short rows
THINKING_TOKENS = 16_000    # a judging call's room when the run turns thinking on
LINKS_AS_ITEMS = 5          # a links file with fewer links is reference, not a list to go through
NOTE_EVERY = 10
FINISH_MAX = 30          # nearly-there items one finishing step is given
AGENT_CONTEXT = 64_000   # a funnel run's agent steps work in this much context (config funnel_agent_context)

SOURCE_KINDS = ("search", "page", "links", "file", "document")
FILTER_OPS = ("regex", "not_regex", "in", "not_in", "min", "max", "size_max", "size_min")
CHECK_OPS = ("required", "in", "regex", "in_evidence", "within_months", "min", "max")
# "looked": how many more pages, results or file parts a look into the item read ("0": it looked and found none)
_ITEM_COLS = ["key", "name", "group", "status", "stage", "reason", "source", "step", "at", "looked"]
_MAYBE_COLS = ["key", "name", "group", "fields", "why", "sources", "row", "missing"]  # row: the cells the judging
# call filled; missing: the rules it named as not shown (to trace why an item came close)
_HANDOVER_COLS = ["key", "name", "group", "fields", "origin", "why", "from"]
_LOOSE_COL = "_loosened"
RULES_MAX = 12
# A group counts its loose rows only once it has looked at this many items (3 per row
# wanted, at least 30): before that a strict row is still likely to turn up.
LOOSE_AFTER_PER, LOOSE_AFTER_MIN = 3, 30
# Where an item left the funnel: code's filters, after reading, the judging call, a
# code check, its group already full, a duplicate; or kept, maybe, loose (it meets the
# soft rules only by their looser reading), or the finishing agent's (kept, or refused
# when its cited page didn't bear it out).
STAGES = ("filter", "read", "judge", "check", "full", "duplicate", "kept", "maybe", "loose", "agent", "agent-check")
_GROUP_COL = "_group"
_SKIP_SITES = ("linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com", "youtube.com",
               "wikipedia.org", "crunchbase.com", "zoominfo.com", "rocketreach.co", "bbb.org", "yelp.com",
               "clutch.co", "glassdoor.com", "indeed.com", "apollo.io", "dnb.com", "bloomberg.com",
               "upcity.com", "designrush.com", "goodfirms.co", "expertise.com", "manta.com", "yellowpages.com",
               "mapquest.com", "signalhire.com", "craft.co", "owler.com", "pitchbook.com", "tiktok.com",
               "pinterest.com", "medium.com", "reddit.com", "google.com", "bing.com", "amazon.com",
               "sortlist.com", "themanifest.com", "agencyspotter.com", "chamberofcommerce.com", "buzzfile.com",
               "opencorporates.com", "bizapedia.com", "cbinsights.com", "tracxn.com", "builtin.com",
               "leadiq.com", "theorg.com", "thumbtack.com", "lusha.com", "contactout.com", "cognism.com",
               "seamless.ai", "kompass.com", "angi.com", "houzz.com", "bark.com", "fiverr.com", "upwork.com")
REFERENCE_CHARS = 3000   # what a judging call reads of the run's reference files, in all
CHARTER_CHARS = 3500     # and of the owner's charter (the columns' own rules)
JEV_TIMEOUT_S = 20       # one typed question to Jev about an item's group
JEV_SURE = 0.6           # Jev's confidence a group pick needs to stand over the judging answer
JEV_NO = 0.4             # below this p(yes), Jev says an item doesn't meet a charter's alternative
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
                           "allowed": {}, "key": "", "exclude": "", "required": []}
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
        out["required"] += [c for c in check.get("non_empty") or [] if c not in out["required"]]
    return out


def normalize_recipe(raw: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """A recipe as the funnel runs it: the model's draft, checked field by field,
    with what the run's checks say taking its place where they say it."""
    raw = raw if isinstance(raw, dict) else {}
    base = base or from_contract([])
    about = {}
    owner_cols: list[str] = []
    for c in raw.get("columns") or []:
        if isinstance(c, dict) and str(c.get("name") or "").strip():
            about[str(c["name"]).strip()[:80]] = str(c.get("what") or c.get("about") or "").strip()[:300]
            if str(c.get("from") or "").strip().lower() == "owner":
                owner_cols.append(str(c["name"]).strip()[:80])
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
    from captain_claw.flight_deck.code_contract import sane_exclude
    keep = list(groups) + [v for vals in base["allowed"].values() for v in vals]
    for f in raw.get("filters") or []:
        if isinstance(f, dict) and f.get("op") in FILTER_OPS and str(f.get("column") or "").strip():
            if f["op"] == "not_regex" and not sane_exclude(str(f.get("value") or ""), keep):
                continue  # it would rule out every item, or a group's own (an "agency" filter on a list of agencies)
            if f["op"] in ("in", "not_in") and str(f["column"]).strip().lower() in {c.lower() for c in columns}:
                # a list column's fixed values are the judging call's to set: what a listing says ("Boutique
                # consulting firm") is free text, never one of them exactly. Its check holds them on the row.
                continue
            flt = {"column": str(f["column"]).strip()[:80], "op": f["op"], "value": f.get("value")}
            only = [g for g in _names(f.get("groups")) if g in groups]
            if only and len(only) < len(groups):  # it holds for some of the list's groups, not all
                flt["groups"] = only
            recipe["filters"].append(flt)
    if base["exclude"] and sane_exclude(base["exclude"], keep):
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
    recipe["owner_columns"] = [c for c in owner_cols if c in columns and c not in (key, group_by)]
    words = raw.get("group_words") if isinstance(raw.get("group_words"), dict) else {}
    recipe["group_words"] = {g: [w for w in (" ".join(str(x).split()).lower()[:40] for x in (words.get(g) or [])
                                             if isinstance(x, str)) if len(w) >= 2][:12] for g in groups if words.get(g)}
    recipe["skip_sites"] = [d for d in (fx.domain_of(str(x)) for x in raw.get("skip_sites") or [] if isinstance(x, str))
                            if d and "." in d][:30]
    recipe["rules"] = _rules(raw.get("rules"), recipe)
    # a column the run's done-checks want filled in every row is required here too, so no
    # row on the list fails them later; but never one a rule's own alternative lets be
    # empty ("with no rating, judged good"): that alternative is the owner's word
    optional = {c for r in recipe["rules"] if r.get("if_not_shown") for c in r.get("columns") or []}
    for col in base.get("required") or []:
        if col in columns and col not in optional and not any(
                x["column"] == col and x["op"] == "required" for x in recipe["checks"]):
            recipe["checks"].append({"column": col, "op": "required"})
    recipe["checks"] = [x for x in recipe["checks"] if not (x["op"] == "required" and x["column"] in optional)]
    recipe["shortage_ok"] = raw.get("shortage_ok") is True
    files = raw.get("files") if isinstance(raw.get("files"), dict) else {}
    for key in ("summary", "exclusions"):  # else summary.md and exclusions.md
        name = _rel(files.get(key))
        if name and name.lower().endswith((".md", ".markdown", ".txt")) and name not in (output, recipe.get("summary_file")):
            recipe[f"{key}_file"] = name
    return recipe


UNSHOWN = "not shown in what was read, and nothing read says it breaks the rule"
# A rule about something measured (a count, an amount, a date, a rating): many items
# never publish it, so not finding it is no proof it's broken.
_MEASURED = re.compile(r"\d|\b(?:employees?|people|staff|headcount|team size|size|revenue|turnover|sales|profit|"
                       r"funding|raised|valuation|budget|price|cost|fees?|rates?|founded|since|years?|months?|"
                       r"age|old|recent|dated|rating|rated|stars?|reviews?|score|members?|followers?|users?|"
                       r"customers?|clients?|downloads?|seats?|capacity|citations?|cited)\b", re.I)
# What a list can't be used without: a way to reach the item.
_CONTACT = re.compile(r"e-?mail|inbox|phone|contact|address|reach", re.I)


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text or "")).lower())
    return {w[:5] for w in words if len(w) >= 4 and w not in _GROUP_STOP | {"shown", "page", "pages", "with", "from"}}


def _rule_columns(text: str, recipe: dict[str, Any]) -> list[str]:
    """The list's columns a rule is about, by their words ("2-50 employees shown on
    LinkedIn" → Employees, EmployeesSourceURL), when the setup gave none."""
    words = _stems(text)
    return [c for c in recipe["columns"] if _stems(c) & words]


# How the item may be reached on a looser reading: its own site's contact page or form,
# never an address guessed or taken from a contact-list site. Many small firms print no
# address at all, only a form; their row says how to reach them, the address left empty.
CONTACT_LOOSER = ("a way to reach it on its own site: a contact page or form (its link kept, the address "
                  "column left empty); never an address guessed or taken from a contact-list site")


# What holds for a way to reach an item when it publishes no address, in every run (a
# short list accepted or not): its own site's contact page or form. Code finds that
# page among what was read and writes its link where the address would be; with no
# such page the item comes close, never dropped for it.
CONTACT_ALT = ("no address published: a contact page or form on its own site, its link given in the "
               "address's place")
_CONTACT_PAGE = re.compile(r"contact|kontakt|contatt|get[-_ ]?in[-_ ]?touch|reach[-_ ]?(?:us|me|out)|enquir|"
                           r"inquir|write[-_ ]to[-_ ](?:us|me)|impressum", re.I)
_CONTACT_COL = re.compile(r"e-?mail|phone|contact|kontakt|telef|\btel\b", re.I)


def _is_contact(rule: dict[str, Any]) -> bool:
    return _fact_of(rule) == "contact"


FACTS = ("identity", "place", "measured", "contact", "safety", "other")


def _fact_of(rule: dict[str, Any]) -> str:
    """What a rule is about: the setup's own word for it (any language), else read
    from its words (English): a place named, a way to reach it, something measured."""
    if rule.get("fact") in FACTS:
        return rule["fact"]
    text = rule["rule"]
    if fx.places_named(text):
        return "place"
    if _CONTACT.search(text) or any(_CONTACT.search(c) for c in rule.get("columns") or []):
        return "contact"
    return "measured" if _MEASURED.search(text) else "other"


def _protected(rule: dict[str, Any], recipe: dict[str, Any]) -> bool:
    """A must that is never loosened: every must, but one about how to reach the item
    (met loosely by a contact page on its own site, ``CONTACT_LOOSER``) or about
    something measured (a count, an amount, a date, a rating) that many items never
    publish (:func:`_fact_of`). A measured one stays when it's about the list's key,
    group or site column. Unshown, a protected must leaves the item nearly there,
    never loose."""
    fact = _fact_of(rule)
    if fact == "contact":
        return False
    if fact != "measured":
        return True
    fixed = {recipe["key"], recipe["group_by"], recipe.get("site_column") or ""} - {""}
    return bool(set(rule.get("columns") or []) & fixed)


def _rules(raw: Any, recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """The owner's rules for an item, as the setup model sorted them: a must (an item
    that breaks it is dropped) or a prefer rule with a looser reading (used only when
    rows run short). A prefer rule without a looser reading is a must. Every rule
    names the columns it's about (inferred from its words when the setup gave none);
    the key and the group never loosen. A must is ``protected`` when it's about what
    the item is, where it is, or how to reach it; any other must, unshown, may be met
    by "not shown" when rows run short, as a prefer rule by its looser reading."""
    out: list[dict[str, Any]] = []
    fixed = {recipe["key"], recipe["group_by"], recipe.get("site_column") or ""}
    for r in raw if isinstance(raw, list) else []:
        if not isinstance(r, dict) or not str(r.get("rule") or "").strip():
            continue
        text = " ".join(str(r["rule"]).split())[:300]
        looser = " ".join(str(r.get("looser") or "").split())[:300]
        kind = "prefer" if str(r.get("kind") or "").lower() == "prefer" and looser else "must"
        given = [c for c in _names(r.get("columns")) if c in recipe["columns"]]
        # what the setup named, and what the rule's own words name ("a founder's or partner's
        # email" → FounderOrPartner too): a column it left out stays required on a loose row
        cols = list(dict.fromkeys(given + _rule_columns(text, recipe)))
        rule: dict[str, Any] = {"rule": text, "kind": kind, "columns": cols}
        fact = str(r.get("fact") or "").strip().lower()
        if fact in FACTS:
            rule["fact"] = fact
        if kind == "must":
            rule["protected"] = _protected(rule, recipe)
            if not rule["protected"] and _is_contact(rule) and not looser:
                looser = CONTACT_LOOSER
        if recipe["group_by"] and recipe["group_by"] in rule["columns"]:
            rule["of_group"] = True  # which group an item is in: the rule about the list's groups
        alt = " ".join(str(r.get("if_not_shown") or "").split())[:300]
        if kind == "must" and _is_contact(rule) and alt in ("", CONTACT_ALT):
            # the cells that say how to reach it (the list's key too, when that is its address)
            apart = {recipe["group_by"], recipe.get("site_column") or ""}
            reach = [c for c in rule["columns"] if c not in apart and _CONTACT_COL.search(c)] \
                or [c for c in rule["columns"] if c not in apart] \
                or [c for c in recipe["columns"] if c not in apart and _CONTACT_COL.search(c)]
            if reach:
                rule["reach"], alt = reach, CONTACT_ALT
        rule["columns"] = [c for c in rule["columns"] if c not in fixed]
        if looser:
            rule["looser"] = looser
        if alt:  # the charter's own words for when its fact isn't found: it holds always
            rule["if_not_shown"] = alt
        out.append(rule)
        if len(out) >= RULES_MAX:
            break
    return out


# What a judging call is told about a column that joins the item to what the owner offers.
OWNER_COLUMN_NOTE = (" — written from the owner's material under Reference together with what was read about this "
                     "item: it names what of the owner's fits (by its name there) and the fact about this item "
                     "that shows it. Never a line about the item alone, never one that would fit any item.")


def judgement_columns(recipe: dict[str, Any], reference: str = "") -> list[str]:
    """Columns a judging call fills from the owner's own material, not from what an
    item publishes: the ones the setup marked ``"from": "owner"``, and one named after
    a reference file (CaptainReason beside Captain.md). Always filled, a row on a looser
    reading too. A column a rule is about never is. (``reference`` is kept for callers;
    a value's words in the reference are no sign: "personal" is in any text.)"""
    ruled = {c for r in recipe.get("rules") or [] for c in r.get("columns") or []}
    fixed = {recipe.get("group_by") or "", recipe.get("key") or ""}
    stems = {re.sub(r"[^a-z]", "", Path(str(ref)).stem.lower()) for ref in recipe.get("reference") or []}
    stems = {st for st in stems if len(st) >= 4}
    owner = set(recipe.get("owner_columns") or [])
    # one the setup itself marked stays the owner's when a rule also names it (a rule on its
    # form: "one specific sentence")
    return [c for c in recipe["columns"] if c not in fixed
            and (c in owner or (c not in ruled and any(st in c.lower() for st in stems)))]


def unshown_musts(recipe: dict[str, Any], row: dict[str, Any]) -> list[str]:
    """The never-loosened place and identity rules a row shows nothing for: the rule
    names columns and every one of them is empty ("headquartered in the US" with no
    State). Such a row isn't kept or loose, only nearly there. Columns a loosened rule
    also covers don't count (a safety rule's email columns stay empty on a row reached
    by its contact page)."""
    soft = loosenable(recipe)
    soft_cols = {c for r in soft for c in r.get("columns") or []}
    out = []
    for r in recipe.get("rules") or []:
        if r in soft or r.get("kind") != "must" or _fact_of(r) not in ("place", "identity"):
            continue
        cols = [c for c in r.get("columns") or [] if c not in soft_cols]
        if cols and not any(str(row.get(c) or "").strip() for c in cols):
            out.append(f"{r['rule']} ({', '.join(cols)} empty)")
    return out


def soft_rules(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in recipe.get("rules") or [] if r.get("kind") == "prefer"]


def loosenable(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """The rules a row on a looser reading may miss: the prefer rules, and the musts
    that aren't protected (met by "not shown" when nothing read shows them broken)."""
    return [r for r in recipe.get("rules") or [] if r.get("kind") == "prefer" or not r.get("protected", True)]


def looser_of(rule: dict[str, Any]) -> str:
    return rule.get("looser") or UNSHOWN


def loosen_on(run: dict, recipe: dict[str, Any] | None = None) -> bool:
    """Whether rows on a looser reading may fill a group that runs short
    (``config.funnel_loosen``: ``short``, the default, or ``never``), and the recipe
    has a rule to loosen. Never when the charter itself accepts a short list with the
    reason given (``shortage_ok``): its rules are kept as written, and what meets them
    only loosely comes close."""
    on = str((run.get("config") or {}).get("funnel_loosen") or "short").lower() != "never"
    if recipe is not None and recipe.get("shortage_ok"):
        return False
    return on and (recipe is None or bool(loosenable(recipe)))


_ITEM_TAIL = re.compile(r"\s+(?:in|for|that|which|who|with|without|suitable|near|at|from|of the|serving|"
                        r"offering|located|based|where|whose)\b.*$", re.I)


def _short_item(v: Any) -> str:
    """What one row is, as a noun ("small US knowledge firm"), never a sentence: no list,
    no "a" or "the" before it, nothing after "in", "for", "suitable"… ("a restaurant in
    Zagreb suitable for a business lunch" → "restaurant"), and whole words only."""
    text = re.split(r"\s*[(\[,;:—–]\s*|\s+-\s+", str(v or "").strip())[0]
    text = re.sub(r"^(?:an?|the|one|each)\s+", "", " ".join(text.split()), flags=re.I)
    text = _ITEM_TAIL.sub("", text).strip()
    if len(text) > 40:
        text = text[:41].rsplit(" ", 1)[0]
    return text or "item"


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


_BARE_LINK = re.compile(r"(?<![\w@/.-])((?:[a-z0-9-]+\.)+[a-z]{2,10})(/[^\s)\]>\"']*)?", re.I)
_NOT_SITES = {e.lstrip(".") for e in fx.TABLE_EXTS + fx.DOC_EXTS} | {
    "json", "yaml", "yml", "xml", "py", "js", "ts", "zip", "png", "jpg", "jpeg", "gif", "svg", "doc", "xls", "ppt",
    "log", "sql", "sh", "exe", "eg", "ie", "etc"}

BRIEF_SYSTEM = """Before a list is built, you read what its owner gave: their documents and the pages their goal
links to. Write the brief the run works from: plain notes, at most 350 words, only what the texts say.
- What each thing is, what it does and who it is for.
- The facts, names, terms and numbers that help find the right items and tell one that fits from one that doesn't.
- Anything the texts say about how the list is to be made.
No advice of your own, nothing the texts don't say, no headings. A text that says nothing useful gets no line.
Reply with the notes only."""


def text_links(text: str) -> list[str]:
    """The links a text names: full ones, and a bare site ("acme-tools.com/pricing") that
    isn't a file's name ("leads.csv") or an email's host."""
    out = list(fx.links(text or ""))
    for m in _BARE_LINK.finditer(text or ""):
        host = m.group(1).lower()
        if host.rsplit(".", 1)[-1] in _NOT_SITES or re.fullmatch(r"[\d.]+", host):
            continue
        if text[max(0, m.start() - 3):m.start()].endswith("://"):
            continue  # already taken, whole
        url = "https://" + host + (m.group(2) or "").rstrip(".,;:")
        if not any(fx.domain_of(u) == fx.domain_of(url) and urlsplit(u).path.rstrip("/") == urlsplit(url).path.rstrip("/")
                   for u in out):
            out.append(url)
    return out


def brief_inputs(task: str, folder: Path) -> tuple[list[str], list[str]]:
    """What a run reads before it starts: ``(links, documents)``. The links its charter
    names, those of a short links file the owner added (a long one is a list to go
    through), and those in the documents; the documents the owner gave, when they are
    a few (a pile of them is what the list is built from, read item by item)."""
    links: list[str] = text_links(task)
    docs: list[str] = []
    extra = folder / "extra"
    files = sorted(p for p in extra.iterdir() if p.is_file() and not p.name.startswith(".")
                   and p.name != LEDGER_FILE and p.suffix.lower() in fx.DOC_EXTS) if extra.is_dir() else []
    plain = []
    for p in files:
        n = _links_count(p)
        if 0 < n < LINKS_AS_ITEMS:
            try:
                links += [x["url"] for x in fx.read_links(p.read_text(encoding="utf-8", errors="replace"))]
            except OSError:
                pass
        elif not n:
            plain.append(p)
    if len(plain) <= BRIEF_DOCS:
        docs = [f"extra/{p.name}" for p in plain]
    else:  # a pile: only the ones the charter names
        docs = [f"extra/{p.name}" for p in plain if p.name.lower() in task.lower()][:BRIEF_DOCS]
    for rel in docs:
        try:
            links += fx.links(fx.read_document(folder / rel)[:50_000])
        except (OSError, ValueError):
            continue
    return list(dict.fromkeys(u.rstrip(".,;") for u in links))[:BRIEF_LINKS], docs


def _brief_without_sources(folder: Path, recipe: dict[str, Any]) -> None:
    """Once the recipe says what the list is built from: a brief that read only those
    files is dropped. They are gone through item by item; summed up beside every item
    they would say other items' facts."""
    text = brief_text(folder)
    head = text.split("\n", 1)[0]
    if not head.startswith("Read before the run: "):
        return
    read = [x.strip() for x in head[len("Read before the run: "):].rstrip(".").split(", ") if x.strip()]
    paths = [str(s.get("path") or "") for s in recipe.get("sources") or [] if s.get("kind") in ("file", "document")]
    from fnmatch import fnmatch
    if read and all(any(fnmatch(r, pat) for pat in paths) for r in read):
        (folder / BRIEF).write_text("", encoding="utf-8")


def brief_text(folder: Path) -> str:
    try:
        return (folder / BRIEF).read_text(encoding="utf-8").strip()[:BRIEF_CHARS + 400]
    except OSError:
        return ""


async def build_brief(run: dict, folder: Path, creds: dict | None) -> str:
    """Read what the owner gave before anything is searched or judged: their documents
    and the pages their links lead to, summed up once (``shared/funnel/brief.md``).
    The setup writes its searches with it, new searches are written with it, and every
    judging call reads it. "" when the owner gave nothing to read, or it couldn't be
    read; a run that already has one keeps it."""
    have = brief_text(folder)
    if have or (folder / BRIEF).exists() or (folder / TAPE_IN).is_dir():
        return have
    links, docs = await asyncio.to_thread(brief_inputs, str(run.get("task") or ""), folder)
    if not creds or not (links or docs):
        return ""
    from captain_claw.flight_deck import bat_loop, bat_routes
    from captain_claw.llm import Message
    parts: list[tuple[str, str]] = []
    for rel in docs:
        try:
            text = " ".join((await asyncio.to_thread(fx.read_document, folder / rel)).split())
        except (OSError, ValueError):
            continue
        if text:
            parts.append((rel, text[:BRIEF_PAGE_CHARS]))
    unread: list[str] = []
    if links:
        tools = make_tools(run, {"step_key": "brief"})
        try:
            got = await asyncio.gather(*(tools.fetch(u) for u in links), return_exceptions=True)
        finally:
            await tools.close()
        for u, g in zip(links, got):
            ok, text = g if isinstance(g, tuple) else (False, "")
            if ok and str(text).strip():
                parts.append((u, " ".join(str(text).split())[:BRIEF_PAGE_CHARS]))
            else:
                unread.append(u)
    text = ""
    if parts:
        try:
            async with bat_routes._metered(run["id"], "funnel-brief"):
                prov, mt = bat_routes.think_call(creds, temperature=0.2)
                with reasoning.call_effort("off"):
                    resp = await prov.complete(messages=[
                        Message(role="system", content=BRIEF_SYSTEM),
                        Message(role="user", content=f"# The goal\n{str(run.get('task') or '')[:3000]}\n\n"
                                + "\n\n".join(f"# {name}\n{body}" for name, body in parts)),
                    ], temperature=0.2, max_tokens=min(mt, OUT_TOKENS["brief"]))
            text = re.sub(r"<think>.*?</think>", "", resp.content or "", flags=re.S).strip()[:BRIEF_CHARS]
        except Exception as e:  # noqa: BLE001 — the run goes on with the documents as they are
            log.warning("funnel brief call failed", run_id=run.get("id"), error=str(e))
    (folder / STATE).mkdir(parents=True, exist_ok=True)
    body = (f"Read before the run: {', '.join(name for name, _ in parts)}.\n\n{text}\n" if text else "")
    (folder / BRIEF).write_text(body, encoding="utf-8")  # empty: tried, nothing to say; not tried again
    store = bat_loop.get_store()
    if store is not None and (text or unread):
        await store.append_event(run["id"], "note", (
            f"Read before starting: {', '.join(name for name, _ in parts)}. The searches and every judging call "
            "work from what they say." if text else "Nothing could be read before starting.")
            + (f" Couldn't read: {', '.join(unread)}." if unread else ""))
    return body.strip()


def groups_not_the_charters(recipe: dict[str, Any], task: str) -> str:
    """"" when a recipe's groups can be the charter's, else what is wrong, for the setup
    to put right: a group's amount is a number the charter never states (two of its
    groups were merged into one: "200 firms and 75 people in the US" as "US: 275")."""
    stated = {int(n.replace(",", "")) for n in re.findall(r"(?<![\w.])\d{1,3}(?:,\d{3})+|(?<![\w.])\d+", task or "")}
    odd = [f"{g} {n}" for g, n in (recipe.get("groups") or {}).items() if n and n not in stated]
    if not odd or len(stated) < 2:
        return ""
    return ("These groups' amounts are not in the goal: " + ", ".join(odd) + ". The groups are the goal's own, "
            "one for each amount it gives, each with that amount. A group that is two things at once (a kind "
            "and a place) stays one group, named by both; its searches look for that group only.")


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


# Group nouns too vague to search by alone: "research shops" is also looked for as
# "research firms" when the run's item is a firm.
_VAGUE_NOUNS = {"shops", "shop", "boutiques", "boutique", "outfits", "outfit", "houses", "players", "providers"}


def _group_words(group: str, said: list[str] | None = None) -> list[str]:
    """The stems a search for a group should carry, any one of them: the group's own
    words (shops, firms and such aside) and the words the setup gave for it
    (``group_words``: "law firm" for "legal", in any language)."""
    words = re.findall(r"[a-z]+", _fold(group))
    stems = [w[:6] for w in words if len(w) >= 4 and w not in _GROUP_STOP]
    stems += [w[:6] for phrase in said or [] for w in re.findall(r"[a-z0-9]+", _fold(phrase))
              if len(w) >= 3 and w not in _GROUP_STOP]
    return list(dict.fromkeys(stems))


def _plural(noun: str) -> str:
    n = noun.strip()
    if not n or n.endswith("s"):
        return n
    return n[:-1] + "ies" if n.endswith("y") and n[-2:-1] not in "aeiou" else n + "s"


def _group_phrases(group: str, item: str = "") -> list[str]:
    """How to say a group in a search: a slashed word as each of its parts ("research/
    analysis shops" → "research shops", "analysis shops"), and a vague group noun also
    as the run's item ("research firms")."""
    group = " ".join(str(group or "").split())
    m = re.search(r"\S+/\S+", group)
    phrases = [group.replace(m.group(0), part) for part in m.group(0).split("/") if part] if m else [group]
    item_words = str(item or "").split()
    noun = _plural(item_words[-1].lower()) if 0 < len(item_words) <= 2 else ""  # not "US knowledge firm lead"
    if noun in ("items", "rows", "entries", "leads", "results"):
        noun = ""  # no noun of its own to search by
    out: list[str] = []
    for ph in phrases:
        out.append(ph)
        words = ph.split()
        if noun and len(words) > 1 and words[-1].lower() in _VAGUE_NOUNS and noun not in ph.lower():
            out.append(" ".join(words[:-1] + [noun]))
    return list(dict.fromkeys(out))


def _short_names(group: str) -> list[str]:
    """A group's own short words ("US", "UK", "B2B"): too short to look for inside other
    words, so a search names the group by one only as a whole word."""
    return [w for w in re.findall(r"[a-z0-9]+", _fold(group)) if 2 <= len(w) <= 3 and w not in _GROUP_STOP]


def names_group(queries: list[str], group: str, said: list[str] | None = None) -> bool | None:
    """Whether a group's searches name it: by a stem of its words or of the words the
    setup gave for it, or by one of its short words whole. None when the group's name
    gives nothing to look for."""
    words, short = _group_words(group, said), _short_names(group)
    if not words and not short:
        return None
    joined = _fold(" ".join(queries))
    return any(w in joined for w in words) or any(re.search(rf"\b{re.escape(w)}\b", joined) for w in short)


def _group_queries(group: str, place: str, size: re.Match | None, item: str = "", about: str = "") -> list[str]:
    """Searches for pages that list many of a group's items, any kind of item: lists,
    directories, databases, rankings, for each way to say the group. A company-size
    band adds a search for company pages with that band. A group whose name says no
    kind of thing (a place, a code) is searched by what its source looks for."""
    phrases = _group_phrases(group, item) or [group]
    head = " ".join(str(about or "").split(":")[0].split())[:80]
    if head and not _group_words(group):
        phrases = [head if re.search(rf"\b{re.escape(group)}\b", head, re.I) else f"{group} {head}"]
    qs = [q for t in ("list of {} {}", "best {} {}", "{} directory {}", "{} {}", "top {} {}", "{} database {}")
          for q in (t.format(ph, place) for ph in phrases)]
    if size:
        qs += [f'site:linkedin.com/company {ph} {place} "{size.group(1)}-{size.group(2)} employees"' for ph in phrases]
    return list(dict.fromkeys(" ".join(q.split()) for q in qs))[:24]


def _spread_patterns(sources: list[dict[str, Any]], folder: Path, output: str = "") -> list[dict[str, Any]]:
    """A source whose path is a pattern ("extra/report_*") as one source per file it
    matches, in name order: a table as a table, a document as a document. A folder of
    many files alike is then one line of the recipe, not one a file."""
    out: list[dict[str, Any]] = []
    have = {s.get("path") for s in sources if s.get("path")}
    for s in sources:
        path = str(s.get("path") or "")
        if s["kind"] not in ("file", "document") or not any(c in path for c in "*?["):
            out.append(s)
            continue
        root = folder.resolve()
        for p in sorted(folder.glob(path))[:SOURCES_MAX]:
            rel = p.relative_to(folder).as_posix()
            ext = p.suffix.lower()
            if (not p.is_file() or p.name.startswith(".") or p.name == LEDGER_FILE or rel == output or rel in have
                    or root not in p.resolve().parents):
                continue
            kind = "file" if ext in fx.TABLE_EXTS else "document" if ext in fx.DOC_EXTS else ""
            if kind:
                have.add(rel)
                out.append({**s, "kind": kind, "path": rel})
    return out


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
        named = names_group(s["queries"], s["group"], (recipe.get("group_words") or {}).get(s["group"]))
        if named is False or s["queries"] in seen_lists:
            s["queries"] = _group_queries(s["group"], place, size, recipe.get("item", ""), s.get("about") or "")
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
    if folder is not None:
        recipe["sources"] = _spread_patterns(recipe["sources"], folder, recipe.get("output") or "")
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
    # "in the last 24 months" holds the list's one date column to that window, whatever
    # check the setup wrote for it (a format check let a two-year-old date through)
    window = re.search(r"\b(?:last|past|within|previous)\s+(\d{1,3})\s+(months?|years?)\b", text, re.I)
    dates = [c for c in recipe["columns"] if re.search(r"date", c, re.I)]
    if window and len(dates) == 1 and not any(x["column"] == dates[0] and x["op"] == "within_months"
                                              for x in recipe["checks"]):
        n = int(window.group(1)) * (12 if window.group(2).lower().startswith("year") else 1)
        recipe["checks"].append({"column": dates[0], "op": "within_months", "value": n})
    return recipe


def group_cell(recipe: dict[str, Any], group: str, current: str = "") -> str:
    """What the column a list is grouped by holds for an item of ``group``: the group's
    name, unless that column may only take values the name isn't one of. A list grouped
    by two things at once ("US_company" under a Type that is company or individual)
    had every good row dropped by its own check, the group's name written over the
    cell. Then the cell keeps what was written when that is allowed, else takes the
    allowed value the group's name carries ("company"), else stays as it was."""
    col = recipe.get("group_by") or ""
    allowed = [str(x) for c in recipe.get("checks") or [] if c.get("column") == col and c.get("op") == "in"
               for x in c.get("value") or []]
    if not allowed or group.lower() in {a.lower() for a in allowed}:
        return group
    kept = next((a for a in allowed if a.lower() == " ".join(str(current or "").split()).lower()), "")
    if kept:
        return kept
    # by its words, one or many of a thing alike ("US companies" carries "company")
    forms = lambda text: {_singular(w) for w in re.findall(r"[a-z0-9]+", _fold(text))}  # noqa: E731
    words = forms(group)
    carried = [a for a in allowed if forms(a) and forms(a) <= words]
    return carried[0] if len(carried) == 1 else str(current or "")


def rows_wanted(recipe: dict[str, Any]) -> int:
    """How many rows the list asks for: its groups' amounts, else its target; 0 when it
    takes every item that passes."""
    groups = recipe.get("groups") or {}
    if groups and all(groups.values()):
        return sum(int(n) for n in groups.values())
    return int(recipe.get("target") or 0)


def search_budget(run: dict, recipe: dict[str, Any]) -> int:
    """The searches a run may make: what its owner set, else 1,000, or more for a long
    list (a list of 500 can't be found in the searches a list of 40 needs)."""
    said = (run.get("config") or {}).get("funnel_search_budget")
    if said not in (None, ""):
        return _int(said, 0, 100_000, SEARCH_BUDGET)
    return min(100_000, max(SEARCH_BUDGET, SEARCHES_PER_ROW * rows_wanted(recipe)))


def limit_rules(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """The limits code checks on a column ("ZoneMm at least 15"), as rules for the judging
    call: a limit the setup made a check was no rule, so an item whose value wasn't found
    at all could only be dropped ("the threshold can't be shown"), never come close. As
    a rule it is named as missing when its value isn't found, and as broken, with the
    words read, when a value fails it. None for a column a rule is already about."""
    ruled = {c for r in recipe.get("rules") or [] for c in r.get("columns") or []}
    out: list[dict[str, Any]] = []
    for c in recipe.get("checks") or []:
        col = c.get("column")
        if c.get("op") not in ("min", "max") or not col or col in ruled or col not in (recipe.get("columns") or []):
            continue
        ruled.add(col)
        out.append({"rule": f"{col} is {'at least' if c['op'] == 'min' else 'at most'} {c.get('value')}",
                    "kind": "must", "fact": "measured", "columns": [col], "protected": False, "limit": True})
    return out


def _column_words(name: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str(name))]


_NAMING_WORDS = {"id", "no", "nr", "number", "num", "name", "code", "key", "ref"}


def _same_column(theirs: str, mine: str) -> bool:
    """A table's column that is the list's by its name: the same letters ("Strain ID"),
    the same but for a word that only says it names something ("Strain" and "StrainID",
    never "Researcher" and "ResearcherEmail"), or every word of the list's name among
    the table's ("zone", "mm" in "mean_zone_mm")."""
    a, b = _column_words(theirs), _column_words(mine)
    if not a or not b:
        return False
    plain = lambda ws: "".join(ws)  # noqa: E731
    bare = lambda ws: [w for w in ws if w not in _NAMING_WORDS] or ws  # noqa: E731
    if plain(a) == plain(b) or plain(bare(a)) == plain(bare(b)):
        return True
    return set(b) <= set(a)


def _join_value(v: Any) -> str:
    """A cell as a key to look up by: its words and numbers, case, punctuation and
    leading zeros aside ("NG-0101" and "ng 101"). "" for an empty cell, a bare number or
    one too short to name anything."""
    parts = [str(int(x)) if x.isdigit() else x for x in re.findall(r"[^\W\d_]+|\d+", _fold(str(v or "")))]
    out = " ".join(parts)
    return out if len(out) >= 3 and not all(x.isdigit() for x in parts) else ""


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
    soft = [] if recipe.get("shortage_ok") else loosenable(recipe)
    if recipe.get("rules"):
        fixed = [r["rule"] for r in recipe["rules"] if r not in soft]
        lines += ["", "## Rules", ""]
        if recipe.get("shortage_ok"):
            lines += ["The charter accepts a short list when the summary says why: every rule is kept as written, "
                      "never read loosely to fill a group.", ""]
        alt = {r["rule"]: r["if_not_shown"] for r in recipe["rules"] if r.get("if_not_shown")}
        if fixed:
            lines += ["Never loosened:"] + [f"- {r}" + (f" — if its fact isn't found: {alt[r]}" if r in alt else "")
                                            for r in fixed]
        if soft:
            lines += ["", "May be met by a looser reading, only when a group runs short of rows that meet every "
                          "rule as written (unshown is not broken; an item shown to break a rule is still out):"]
            lines += [f"- {r['rule']} — looser: {looser_of(r)}" + (f"; if its fact isn't found: {r['if_not_shown']}"
                                                                   if r.get("if_not_shown") else "") for r in soft]
    checks = [c for c in recipe["checks"] if c["op"] != "required"]
    if checks:
        lines += ["", "## What Captain checks in every row", ""]
        lines += [f"- {c['column']} {_check_words(c)}" for c in checks]
    lines += ["", "Work in progress is in `shared/funnel/`: `items.csv` has every item looked at and what became "
              "of it; `maybe.csv` the ones nearly there."]
    return "\n".join(lines) + "\n"


# A link's label stops at a table cell's edge: "| Zagreb | Web (…)" is labelled "Web".
_LABELLED_LINK = re.compile(r"([^\n()|]{0,80}?)\s*\((https?://[^)\s]+)\)")


def site_links(text: str, base: str, paths: list[str], most: int = 5) -> list[str] | None:
    """The pages of an item's own site its page links to that the recipe's paths ask for:
    links on the same site (never another's, never the same page or an anchor on it)
    whose address or label carries a word of those paths ("kontakt", "menu", "o-nama"
    → "Kontakt (…/hr/kontakt-nas)", "Jelovnik (…/nasa-ponuda/jelovnik)"). None when the
    page shows no links at all (read as plain text): then the paths are all there is."""
    found = _LABELLED_LINK.findall(text or "")
    bare = fx.links(text or "")
    if not found and not bare:
        return None
    words = {w for p in paths for w in re.findall(r"[a-z]+", _fold(p)) if len(w) >= 3}
    if not words:
        return []
    host = fx.domain_of(base)
    labelled = {u.rstrip(".,;:!?"): label for label, u in found}
    out: list[str] = []
    for url in list(labelled) + [u for u in bare if u not in labelled]:
        page = url.split("#", 1)[0]
        if not page or not fx.same_site(fx.domain_of(page), host) or page.rstrip("/") == base.rstrip("/"):
            continue
        said = _fold(f"{urlsplit(page).path} {labelled.get(url, '')}")
        if any(w in said for w in words) and page not in out:
            out.append(page)
        if len(out) >= most:
            break
    return out


def page_excerpt(text: str, share: int, marks: list[str] | None = None) -> str:
    """What a judging call reads of one page, in ``share`` characters: all of it when it
    fits; else its start, its end (a site's footer: its address, phone, opening hours,
    rating) and short windows around the facts code found on it, in page order."""
    t = " ".join(str(text or "").split())
    if len(t) <= share:
        return t
    head, tail = int(share * 0.45), int(share * 0.3)
    spans = [(0, head), (len(t) - tail, len(t))]
    room = share - head - tail
    for m in marks or []:
        at = t.find(str(m).split(" (")[0][:60])
        if at < 0 or any(a <= at < b for a, b in spans) or room <= 0:
            continue
        a, b = max(0, at - 100), min(len(t), at + 140)
        spans.append((a, b))
        room -= b - a
    spans.sort()
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return " … ".join(t[a:b] for a, b in merged)


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


def summary_name(recipe: dict[str, Any]) -> str:
    """The run's summary: under the name the charter gives it, else ``summary.md``."""
    return recipe.get("summary_file") or SUMMARY


def exclusions_name(recipe: dict[str, Any]) -> str:
    """The items left out, with why: under the charter's name, else ``exclusions.md``."""
    return recipe.get("exclusions_file") or EXCLUSIONS


def load_recipe(folder: Path) -> dict[str, Any] | None:
    try:
        return json.loads((folder / RECIPE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ── setup: the recipe from the charter ─────────────────────────────────

SETUP_SYSTEM = """You set up a list-building run. Code will do the searching, fetching and checking; a model
will judge each item once, in a short call. Write the recipe it follows, as ONLY a JSON object:

{"item": "<what one row is, in a word or two: firm, person, product, restaurant>",
 "columns": [{"name": "<column>", "what": "<what it holds, where it comes from>",
              "from": "item|owner (owner: filled from the owner's own material, e.g. which of their products fits)"}],
 "key": "<the column no two rows share (a website, an email, an id)>",
 "site_column": "<the column holding the item's own website, if it has one>",
 "group_by": "<column the list is split by, or empty>", "groups": {"<group>": <how many>},
 "group_words": {"<group>": ["<words a page or search may use for it, any language>"]},
 "target": <rows wanted, 0 = every item that passes>,
 "sources": [
   {"kind": "file", "path": "<table in the folder>", "group": ""},
   {"kind": "document", "path": "<pdf/docx/text to pull items out of; many files alike: one pattern, extra/report_*>", "group": ""},
   {"kind": "links", "path": "<links file>", "as": "items|pages"},
   {"kind": "page", "url": "<a page that lists many items>", "group": ""},
   {"kind": "search", "queries": ["<query>", ...], "group": "<group>", "country": "US", "count": 20,
    "about": "<what these searches look for>"}],
 "filters": [{"column": "<column, or size_band>", "op": "regex|not_regex|in|not_in|min|max|size_max|size_min",
              "value": ...}],
 "gather": [{"kind": "search", "query": "\\"{Name}\\" <where the facts the rules need are listed>", "count": 5},
            {"kind": "fetch", "url": "{Website}", "paths": ["", "<the item's pages that hold those facts>"]}],
 "extract": ["emails", "phones", "dates", "size_band", "us_state", "links"],
 "decide": "<the owner's rules for an item, short, in their words, every clause kept>",
 "rules": [{"rule": "<one of the owner's rules for an item>", "kind": "must|prefer",
            "fact": "identity|place|measured|contact|safety|other",
            "looser": "<prefer only: the loosest reading that still serves the goal>",
            "if_not_shown": "<only the charter's own words for when this rule's fact isn't found, if it says>",
            "columns": ["<the columns this rule is about>"]}],
 "skip_sites": ["<sites that list or mention items of this kind but are never one's own site>"],
 "shortage_ok": <true when the charter says a short list is fine if the summary says why, else false>,
 "files": {"summary": "<the file the charter names for the run's summary, if it names one>",
           "exclusions": "<the file it names for the items left out with why, if it names one>"},
 "checks": [{"column": "<column>", "op": "required|in|regex|in_evidence|within_months|min|max",
             "value": ..., "source_column": "<column with the link it came from>"}],
 "decide_batch": <items per judging call: 1 when each item has pages to read, up to 20 for short rows>,
 "reference": ["<files that describe something the rows need (your product, rules, a price list), not items>"]}

Rules:
- A folder of many files alike (one item or a few in each) is one source with a pattern in its path
  ("extra/report_*", "extra/*.pdf"), never a source per file. A table to look a fact up in (people, places,
  codes, corrections) is no source: the items are looked up in it as they are judged.
- Use the owner's own files and links first; they cost nothing to read. Search only when the list has to be
  found on the web, and then write 10-25 searches per group that find pages listing many of the right items
  (directories, catalogues, databases, rankings, "best of" and category pages), not one item each.
- Gather only what the checks and the rules need, from the pages this kind of item keeps them on (a firm's about,
  team and contact pages; a restaurant's menu and contact pages; a product's spec page; a person's profile), and
  put the cheapest test first: a company's size band from a search snippet ("size_max" filter) before reading.
  size_max and size_min count people (staff, employees) only, never anything else: a limit on locations,
  branches, seats, products or years is a rule, not a size filter.
- A filter that holds for only some of the list's groups (a size limit on its companies, when it lists people
  too) names them: "groups": ["<group>", ...]. Without it a filter rules items out in every group.
- fact, for each rule: identity (what the item is: its kind, category, segment, and a state that rules it out:
  withdrawn, closed, discontinued, contaminated, banned), place (where it is), measured (a
  count, amount, date or rating), contact (how to reach it), safety (what must never be done), other. Code reads
  these in any language: a measured or contact rule may be met loosely when rows run short; identity, place,
  safety and other never are.
- shortage_ok: true only when the charter itself accepts fewer rows than it asks for, with the reason given ("8
  per group, or the summary explains the shortfall"); then no rule is read loosely to fill the list.
- files: when the charter names the file for the run's summary or for the items left out ("sazetak.md",
  "excluded.md"), give those names: code writes its summary and its list of dropped items under them.
- group_words: for each group, the words pages and searches use for it ("legal" → law firm, attorney, advokat).
- columns "from": "owner" for a column filled from the owner's own material (a reference file, what their
  documents and links say), never from the item's pages alone; every row fills it. A column that says why or how
  an item fits what the owner offers (a reason to use it, a feature to pitch, an angle) is one of these: it joins
  what the owner's material says to a fact read about the item. Never write a rule or a column meaning that takes
  such a column from the item's pages only.
- Templates name columns in braces ({Firm}, {Website}) or {domain}.
- checks: "in_evidence" for every value copied from a source (an email, a date, a quote) with the column that
  holds its link; "within_months" for a date window; "in" for fixed values.
- An offline list (a file or documents only) has no searches and no gather unless the rules need the web.
- A file the goal points to for information (what the owner's product is, house rules) is "reference", never a
  source: no items are taken from it, and every judging call reads it.
- The groups are the goal's own: one for each amount it gives, with that amount, never two of them added up. A
  group that is two things at once (a kind and a place: "200 firms in the US, 75 people in the US") stays one
  group, named by both ("US firms", "US people"), with its own searches.
- Each group's searches name that group; never copy one group's searches to another. When the list is about one
  place, every search names it (the country, or a state or city in it).
- Never invent a file or a link that isn't in what you're shown.
- rules: split the owner's rules for an item, one per line, into two kinds. "must" is what makes an item the
  right kind of thing at all: its type or category, its place, the owner's explicit exclusions, and what the list
  is useless without (a way to reach it on a contact list, the price on a price list). "prefer" is evidence many
  good items never publish or a few pages rarely show: recent activity, dated news or posts, a signal of
  interest, a rating, a certification, a count or an amount (people, revenue, members, a founding year). Give
  each prefer rule its looser reading: a wider time window, any mention undated, another kind of page (any page
  that states it: a directory, a profile, a register), or not shown at all. Unshown is not broken: an item is
  dropped for a rule only when what was read shows it breaks it.
- When the charter itself says what to do when a fact isn't found or doesn't apply ("if there is no rating, judge
  how good it is", "or a contact form if there's no email"), that alternative is part of the rule: write it into
  the rule's own words ("a rating of 4+ on Google or TripAdvisor; with no rating, judged good enough on what was
  read") and into "decide", and put the alternative alone in that rule's "if_not_shown". A fact the charter asks
  for only "if shown" or "if visible" ("its price per person, if visible") has the alternative "left empty". It holds as written, even
  when no rule may be read loosely; never drop such a clause, and never invent one the charter doesn't give.
- A rule that asks for a fact AND the page it must be shown on ("2-50 employees shown on LinkedIn or its team
  page", "revenue under $10M per its filings") is two rules: the exclusion is a must ("not shown to have more
  than 50 people"), the shown proof is a prefer, looser "stated on any page, or not shown". Rows on a looser
  reading are used only when a group runs short, after every strict row. Never make a prefer rule of what the
  owner wrote as "only", "never", "exactly" or an exclusion, nor of a way to reach the item, and never loosen a
  safety rule (don't guess an email, don't fill in forms). Examples: "a dated AI or efficiency signal in the last
  24 months" → prefer, looser "any mention of AI or automation on its site or its social page, dated or not; the
  date column left empty when undated"; "a founder's or a shared email on its own site" on a contact list →
  must; "vegan restaurants in Zagreb rated 4.5 or more" → vegan and Zagreb must, the rating prefer, looser
  "rated 4.0 or more, or no rating found"; "papers on X since 2022 with code" → topic and year must, code
  prefer, looser "no code link, column empty"; "founded before 2015" → prefer, looser "founding year not shown"."""


def _folder_listing(folder: Path) -> str:
    """What the run was given (extra/) and anything else at the top of its folder, for
    the setup call: tables with their columns and first rows, documents' first lines,
    links files' first links."""
    lines: list[str] = []
    more = 0
    places = [folder / "extra", folder]
    seen: set[Path] = set()
    for place in places:
        if not place.is_dir():
            continue
        for p in sorted(place.iterdir()):
            if not p.is_file() or p.name.startswith(".") or p in seen or p.name == LEDGER_FILE:
                continue
            seen.add(p)
            rel = p.relative_to(folder).as_posix()
            if len(lines) > LISTED_WHOLE:  # a large folder: the rest by name, so none goes unseen
                if len(seen) <= LISTED_NAMES:
                    lines.append(f"- {rel}")
                else:
                    more += 1
                continue
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
    if more:
        lines.append(f"- … and {more} more files")
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
                        "queries": _group_queries(what, place, size, item if g else ""),
                        "about": f"{what} {place}".strip()})
    site = next((c for c in cols if re.search(r"website|domain|homepage|site\b|url", c, re.I)
                 and not re.search(r"source", c, re.I)), "")
    name = next((c for c in cols if c != site and not re.search(r"mail|url|source|date|segment|state", c, re.I)), "")
    gather: list[dict[str, Any]] = []
    filters: list[dict[str, Any]] = []
    if size and name:
        gather.append({"kind": "search", "query": f'"{{{name}}}" site:linkedin.com/company', "count": 5})
        filters.append({"column": "size_band", "op": "size_max", "value": int(size.group(2))})
    if site:  # an item's own pages; a company's team page only when its size is asked for
        gather.append({"kind": "fetch", "url": f"{{{site}}}",
                       "paths": ["", "about", "contact", "news"] + (["team"] if size else [])})
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
    taped = folder / TAPE_IN / "recipe.json"
    if taped.is_file():  # a replay: the recording's own recipe, so its searches are the ones on the tape
        try:
            recipe = json.loads(taped.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recipe = None
        if isinstance(recipe, dict) and recipe.get("sources") and recipe.get("columns"):
            (folder / STATE).mkdir(parents=True, exist_ok=True)
            (folder / RECIPE).write_text(json.dumps(recipe, indent=1, ensure_ascii=False), encoding="utf-8")
            (folder / "shared" / "conventions.md").write_text(conventions(recipe), encoding="utf-8")
            return recipe, f"replayed from a recording (`{TAPE_IN}`): its recipe, searches and pages, not the web"
    try:
        tiers, _ = await _load_owner_tiers(get_db(), run["owner_id"])
        brief = await build_brief(run, folder, bat_routes._creds_for(tiers))
    except Exception as e:  # noqa: BLE001 — a run without a brief is the run as it was
        log.warning("funnel brief failed", run_id=run.get("id"), error=str(e))
        brief = ""
    from captain_claw.flight_deck import first_mate_recipe
    kept, how = await asyncio.to_thread(first_mate_recipe.for_run, run, folder)
    if kept and kept.get("sources") and kept.get("columns"):  # one recipe per charter, not per run
        (folder / STATE).mkdir(parents=True, exist_ok=True)
        (folder / RECIPE).write_text(json.dumps(kept, indent=1, ensure_ascii=False), encoding="utf-8")
        (folder / "shared" / "conventions.md").write_text(conventions(kept), encoding="utf-8")
        await asyncio.to_thread(_brief_without_sources, folder, kept)
        return kept, how
    recipe: dict[str, Any] | None = None
    why = ""
    built = False  # a model wrote it (not a fallback): worth pinning for the factory's later runs
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
                  + (f"# What the owner's documents and links say (read already; write the searches, the "
                     f"rules and each column's meaning with it)\n{brief}\n\n" if brief else "")
                  + f"# The run's folder\n{await asyncio.to_thread(_folder_listing, folder)}")
        wrong, spare = "", None
        for attempt in range(3):
            try:
                async with bat_routes._metered(run["id"], "funnel-setup"):
                    prov, mt = bat_routes.think_call(creds, temperature=0.2)
                    with reasoning.call_effort("off"):  # a JSON answer: no thinking to loop in
                        resp = await prov.complete(messages=[
                            Message(role="system", content=SETUP_SYSTEM
                                    + ("\nThe answer is JSON only, nothing before or after it." if attempt else "")),
                            Message(role="user", content=prompt + (f"\n\n# Put right from your last answer\n{wrong}"
                                                                   if wrong else "")),
                        ], temperature=0.2, max_tokens=min(mt, OUT_TOKENS["setup"]))
                raw = fx.parse_json(resp.content or "")
                if isinstance(raw, dict):
                    recipe = normalize_recipe(raw, base)
                    if recipe["sources"]:
                        why = ""  # an earlier try's trouble is past
                        built = True
                        wrong = "" if base["groups"] or wrong else groups_not_the_charters(recipe, str(run.get("task", "")))
                        if wrong:  # asked once to put its groups right; this one stands if it can't
                            spare, recipe = recipe, None
                            continue
                        break
                    why = "its recipe named no source to take items from"
                else:
                    why = "its recipe wasn't JSON"
            except Exception as e:  # noqa: BLE001 — tried again, then the folder's own files
                why = f"{type(e).__name__}: {str(e)[:160]}"
                log.warning("funnel setup call failed", run_id=run.get("id"), error=str(e))
            recipe = None
            if attempt >= 1 and spare is None:
                break  # two tries that gave no recipe: the folder's own files, or web searches from its checks
        if recipe is None and spare is not None:
            recipe, built, why = spare, True, ""
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
    await asyncio.to_thread(_brief_without_sources, folder, recipe)
    if built and recipe["sources"] and await asyncio.to_thread(first_mate_recipe.pin_for_run, run, recipe):
        why = (why + "; " if why else "") + "pinned as the factory's recipe: its later runs follow it"
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
        named = sorted({_source_words(s) for s in untagged})
        files = [s for s in untagged if s["kind"] in ("file", "document")]
        if len(files) > 4:  # a folder of files: counted, not named one by one
            rest = sorted({_source_words(s) for s in untagged if s not in files})
            named = [f"{len(files)} of the owner's files"] + rest
        what = ", ".join(named) or "its sources"
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
        tiers = rules_note(recipe, run)
        if tiers:
            await store.append_event(run["id"], "note", tiers)
    return steps


def rules_note(recipe: dict[str, Any], run: dict) -> str:
    """The owner's rules as the run reads them: which an item must meet, and which may
    be met by a looser reading when a group runs short. "" when it sorted none."""
    rules = recipe.get("rules") or []
    if not rules:
        return ""
    soft = loosenable(recipe)
    must = [r["rule"] for r in rules if r not in soft]
    text = "Rules: must — " + ("; ".join(must) if must else "(none)") + "."
    if soft and loosen_on(run, recipe):
        text += " Loosened only when a group runs short — " + "; ".join(
            f"{r['rule']} (looser: {looser_of(r)})" for r in soft) + "."
    elif soft and recipe.get("shortage_ok"):
        text += (" Also kept as written, since the charter accepts a short list with the reason given (what meets "
                 "them only loosely comes close) — " + "; ".join(r["rule"] for r in soft) + ".")
    elif soft:
        text += " Also kept strictly (the factory never loosens) — " + "; ".join(r["rule"] for r in soft) + "."
    return text[:1500]


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


def rows_in(path: Path) -> int:
    return len(_read_csv(path))


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
    base = tuple(s for s in _SKIP_SITES if not re.search(r"(?<![a-z])" + re.escape(s.split(".")[0]) + r"(?![a-z])", words))
    return base + tuple(d for d in recipe.get("skip_sites") or [] if d not in base)


_NAME_STOP = frozenset({"the", "and", "inc", "llc", "ltd", "group", "partners", "company", "consulting", "co",
                        "corp", "firm", "associates", "services"})


_FOLD = str.maketrans({"đ": "d", "Đ": "d", "ß": "ss", "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "ł": "l",
                       "Ł": "l", "ı": "i", "œ": "oe", "þ": "th", "ð": "d"})


# How else addresses spell some letters: Đurđević → djurdjevic, Müller → mueller.
_FOLD_ALT = str.maketrans({"đ": "dj", "Đ": "dj", "ü": "ue", "Ü": "ue", "ö": "oe", "Ö": "oe", "ä": "ae", "Ä": "ae",
                           "å": "aa", "Å": "aa", "ø": "oe", "Ø": "oe"})


def _fold(text: str, alt: bool = False) -> str:
    """Lower case without accents ("Čarobnjak Đuro" → "carobnjak duro"), as addresses
    are written; ``alt`` spells some letters the other way addresses do ("djuro").
    Letters of other scripts are kept as they are."""
    t = str(text or "")
    t = unicodedata.normalize("NFKD", (t.translate(_FOLD_ALT) if alt else t).translate(_FOLD))
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def _words(text: str, alt: bool = False) -> list[str]:
    return re.findall(r"[a-z0-9]+", _fold(text, alt))


def name_words(name: str) -> list[str]:
    """The words that pick an item's name out on a page: its distinctive ones ("P&S
    International Strategies" → international, strategies), else all of them ("U.S.
    Consulting Services")."""
    words = _words(name)
    strong = [w for w in words if len(w) >= 3 and w not in _NAME_STOP]
    return list(dict.fromkeys(strong or [w for w in words if len(w) >= 2]))


_INITIALS_SKIP = frozenset({"the", "and", "of", "llc", "inc", "ltd", "co", "corp", "plc", "gmbh"})
_SECOND_LEVEL = frozenset({"co", "com", "org", "net", "ac", "gov", "edu", "or", "ne"})


def site_names_item(name: str, domain: str, title: str = "") -> bool:
    """Whether a site's address is the item's own: a distinctive word of its name is in
    the domain (darbyconsulting.com, nera.com, 2visions.com, accents aside:
    "Čarobnjak" → carobnjak.hr), or the domain is its initials (bcg.com). A page that
    only mentions it (an article on forbes.com, a listing on a directory) never is,
    whatever its title says. A name with no Latin letters (Дубрава, 東京堂) can't be
    read in an address: then the result's title must carry the name's words."""
    labels = [x for x in _fold(domain).removeprefix("www.").split(".") if x]
    if len(labels) < 2:
        return False
    labels = labels[:-1]
    if len(labels) >= 2 and labels[-1] in _SECOND_LEVEL:
        labels = labels[:-1]  # example.co.uk
    core = "".join(labels).replace("-", "")
    if not _words(name):
        own = [w for w in re.findall(r"\w+", str(name or "").lower()) if len(w) >= 2]
        return bool(own) and all(w in str(title or "").lower() for w in own)
    return any(_core_names(core, _words(name, alt)) for alt in (False, True))


def _core_names(core: str, words: list[str]) -> bool:
    strong = [w for w in words if len(w) >= 3 and w not in _NAME_STOP]
    if strong:
        if any(w in core for w in strong):
            return True
    else:
        weak = [w for w in words if len(w) >= 2]
        if weak and sum(1 for w in weak if w in core) >= min(2, len(weak)):
            return True
    initials = "".join(w[0] for w in words if w not in _INITIALS_SKIP)
    return len(initials) >= 2 and (core == initials or (len(initials) >= 3 and core.startswith(initials)))


def names_item(words: list[str], url: str, title: str) -> bool:
    """Whether a search result is about the item: its address and title carry the
    item's name (every word of a one- or two-word name; at least half, and two, of a
    longer one). A result about something else ("LinkedIn News") never is."""
    if not words:
        return False
    where = f"{url} {title}".lower()
    hits = sum(1 for w in words if w in where)
    return hits >= (len(words) if len(words) <= 2 else max(2, (len(words) + 1) // 2))


def _own_result(results: list[dict[str, str]], name: str, url: str) -> str:
    """An item's own search result (its title, link and snippet), so code reads what
    was said of it alone, not of the other results beside it."""
    d = fx.domain_of(url)
    low = name.lower()
    front = d and not _site_page(url)  # its link is a site's front page: any page of that site is its own
    for r in results:
        same = url and r["url"].rstrip("/") == url.rstrip("/")
        own_site = front and fx.domain_of(r["url"]) == d and not any(fx.same_site(d, s) for s in _SKIP_SITES)
        if same or own_site:  # a list page on the host of its directory link ("restaurantguru.com/…") isn't
            return f"{r['text']}\n{r['url']}"
    for r in results:
        if low and low in r["title"].lower():
            return f"{r['text']}\n{r['url']}"
    words = name_words(name)  # its name in a result's words: a list result's snippet names it
    for r in results:
        if words and names_item(words, _fold(r["url"]), _fold(r["text"])):
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
    words = s.lower().replace(",", " ").replace(".", "").split()  # "d.o.o." and "Inc." stay one word
    while len(words) > 1 and words[-1] in _LEGAL_FORMS:  # "Lux Insights Inc." is "Lux Insights"
        words.pop()
    return " ".join(words)


# Company legal forms, which never make two items of one name (any country's).
_LEGAL_FORMS = frozenset({"inc", "incorporated", "llc", "l.l.c", "ltd", "limited", "plc", "gmbh", "ag", "sa", "srl",
                          "sarl", "sas", "bv", "nv", "oy", "oyj", "ab", "as", "asa", "aps", "kft", "doo", "d.o.o",
                          "dd", "zrt", "spa", "sl", "pty", "co", "corp", "corporation", "company", "lp", "llp",
                          "pllc", "pc", "pa", "kg", "ohg", "ug", "sro", "s.r.o", "sp", "zoo", "kk", "bhd", "sdn"})


# ── the run's tools (search and fetch) ─────────────────────────────────

class Tools:
    """Search and fetch through one agent of the run's own (spawned like a Bat
    worker, never given a turn), so the owner's search provider and keys, the
    worker floor and rehearsal rules apply. Spawned on the first call."""

    def __init__(self, run: dict, step: dict):
        self.run, self.step = run, step
        self.port, self.auth, self.slug = 0, "", ""
        self.tape: Path | None = None  # where a recorded run keeps its searches and pages
        self._starting = asyncio.Lock()  # several first calls at once start one agent, not one each

    def _record(self, name: str, row: dict) -> None:
        if self.tape is None:
            return
        try:
            self.tape.mkdir(parents=True, exist_ok=True)
            with (self.tape / name).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            log.debug("funnel: couldn't record to the tape", exc_info=True)

    async def _ensure(self) -> None:
        if self.port:
            return
        async with self._starting:
            if not self.port:
                await self._start()

    async def _start(self) -> None:
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
            except httpx.TimeoutException:  # a slow page or search: this call fails, the step goes on
                return False, f"{tool} got no answer in time"
            except httpx.HTTPError as e:
                return False, f"{tool} failed: {type(e).__name__}"

    async def search(self, query: str, count: int, country: str = "") -> tuple[bool, str]:
        args: dict[str, Any] = {"query": query, "count": count}
        if country:
            args["country"] = country
        ok, text = await self.call("web_search", args)
        self._record("searches.jsonl", {"query": query, "count": count, "ok": ok, "text": text})
        return ok, text

    async def fetch(self, url: str) -> tuple[bool, str]:
        ok, text = await self.call("web_fetch", {"url": url, "deep_fetch": False})
        self._record("pages.jsonl", {"url": url, "ok": ok, "text": text[:FETCH_CHARS]})
        return ok, text

    async def close(self) -> None:
        if self.slug:
            from captain_claw.flight_deck import bat_routes
            try:
                await asyncio.to_thread(bat_routes._teardown, [self.slug])
            except Exception:  # noqa: BLE001 — the run's end removes it too
                log.debug("funnel: tools agent teardown failed", exc_info=True)
            self.slug, self.port = "", 0


def _tape_key(text: str) -> str:
    return " ".join(str(text or "").split()).rstrip("/").lower()


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    except OSError:
        pass
    return out


class TapeTools:
    """Search and fetch answered from a recording (``extra/tape/``) instead of the web:
    the same searches and pages every time, so a change to the funnel is measured on
    what it does, not on what the web returned that day. A search or page that isn't
    on the tape finds nothing, and is counted (``shared/funnel/tape-misses.json``)."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.searches = {_tape_key(r.get("query")): (bool(r.get("ok", True)), str(r.get("text") or ""))
                         for r in _read_jsonl(folder / TAPE_IN / "searches.jsonl")}
        self.pages = {_tape_key(r.get("url")): (bool(r.get("ok", True)), str(r.get("text") or ""))
                      for r in _read_jsonl(folder / TAPE_IN / "pages.jsonl")}
        self.asked = {"searches": 0, "pages": 0}
        self.missed: dict[str, list[str]] = {"searches": [], "pages": []}

    def _answer(self, kind: str, tape: dict, key: str, nothing: tuple[bool, str]) -> tuple[bool, str]:
        self.asked[kind] += 1
        got = tape.get(_tape_key(key))
        if got is None:
            if key not in self.missed[kind]:
                self.missed[kind].append(key)
            return nothing
        return got

    async def search(self, query: str, count: int, country: str = "") -> tuple[bool, str]:
        return self._answer("searches", self.searches, query, (True, "[RESULTS: 0]\n\nNo results found."))

    async def fetch(self, url: str) -> tuple[bool, str]:
        return self._answer("pages", self.pages, url, (False, "not on the tape"))

    async def close(self) -> None:
        path = self.folder / TAPE_MISSES
        try:
            before = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError):
            before = {}
        out = {"asked": {k: int((before.get("asked") or {}).get(k, 0)) + v for k, v in self.asked.items()},
               "missed": {k: list(dict.fromkeys(list((before.get("missed") or {}).get(k, [])) + v))
                          for k, v in self.missed.items()}}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        except OSError:
            log.debug("funnel: couldn't write the tape's misses", exc_info=True)
        self.asked = {"searches": 0, "pages": 0}
        self.missed = {"searches": [], "pages": []}


def recording(run: dict) -> bool:
    """Whether this run keeps what it searched and read (``shared/funnel/tape/``), to
    replay later: ``config.funnel_record``, or CLAW_FUNNEL_RECORD on the deck."""
    import os
    return bool((run.get("config") or {}).get("funnel_record")) or \
        str(os.environ.get("CLAW_FUNNEL_RECORD", "")).strip().lower() in ("1", "true", "yes")


def make_tools(run: dict, step: dict) -> Tools | TapeTools:
    return tools_for(run, step)


def tools_for(run: dict, step: dict) -> Tools | TapeTools:
    """A run's search and fetch: from its recording when it has one (``extra/tape/``),
    else the web, kept on a tape of its own when the run records."""
    from captain_claw.flight_deck import bat_routes
    folder = bat_routes._run_folder(run)
    if (folder / TAPE_IN).is_dir():
        return TapeTools(folder)
    tools = Tools(run, step)
    if recording(run):
        tools.tape = folder / TAPE_OUT
    return tools


# ── a step's work ──────────────────────────────────────────────────────

COLUMNS_SYSTEM = """A table is read as items for a list, but its columns are named differently from the list's.
Say which of the table's columns holds each of the list's columns, from the names and the sample rows. Map a column
only when it holds that very thing (a measurement of something else is not it); leave out what the table doesn't
have. Reply ONLY with JSON: {"columns": {"<the table's column>": "<the list's column>"}}"""

EXTRACT_SYSTEM = """Pull the items out of this text: each {item} it names that could belong on the list.
Reply ONLY with JSON: {{"items": [{{"name": "...", "url": "<its own website or page, if the text gives one>",
"fields": {{"<column>": "<value the text gives>"}}}}], "pages": ["<links in the text to pages that list more such
items>"]}}. Only what the text says; no item that is a directory, an article or a list itself. A number the text
doesn't state is left out, never counted up or guessed from other words. Keep your thinking short."""

DECIDE_SYSTEM = """You judge items for a list, one by one, from what Captain read about each. Use only the facts
shown; a value you can't support from them stays empty, and a source is a link shown with that fact. A value that
follows directly from a fact shown may be filled from it (the district a street address is in, the country of a
city, the year of a full date); nothing else is filled unless it is shown.
Reply ONLY with a JSON array, one object per item, in order:
[{"id": <its id>, "verdict": "keep|maybe|drop", "reason": "<one short line>",
  "row": {"<column>": "<value>"}}]
keep: it meets every rule and the row is filled from the facts. maybe: it seems to fit but a fact the rules need
wasn't in what was read; also when nothing read came from the item's own website, unless a fact shown breaks a
rule. drop: it breaks a rule. A rule that says itself what to do when its fact isn't found ("if there is no rating,
judge how good it is") is decided that way: judge the item on what was read and keep or drop it on that
judgement, saying so in the reason; that fact being missing is never the reason for maybe. A column with a fixed
set of values takes one of them exactly. Dates as YYYY-MM-DD,
or YYYY-MM when there's no day. A column about the owner's own offer (a feature to pitch, why it fits, from
the reference) is always filled for a row you keep: when the fact it would rest on isn't shown, tie it to what
was read.
Keep your thinking short."""

# The same, when a group that runs short may take rows on a looser reading: a fact
# that simply isn't shown is no reason for maybe, unless a rule that never loosens
# needs it; a maybe names the rules whose facts weren't shown, so code can tell.
DECIDE_LOOSE = DECIDE_SYSTEM.replace(
    '''[{"id": <its id>, "verdict": "keep|maybe|drop", "reason": "<one short line>",
  "row": {"<column>": "<value>"}}]
keep: it meets every rule and the row is filled from the facts. maybe: it seems to fit but a fact the rules need
wasn't in what was read; also when nothing read came from the item's own website, unless a fact shown breaks a
rule. drop: it breaks a rule.''',
    '''[{"id": <its id>, "verdict": "keep|loose|maybe|drop", "reason": "<one short line>",
  "row": {"<column>": "<value>"}, "loosened": [<rule number>], "missing": [<rule number>], "broken": [<rule number>],
  "shown": "<the words read that break it>"}]
keep: it meets every rule as written and the row is filled from the facts. loose: it meets every rule that never
loosens, and each other rule as written or by its looser reading (a fact that isn't shown, with nothing saying
the item breaks the rule, is met by "not shown"); name those rules in "loosened" and fill the row. maybe: a fact a
rule that never loosens needs wasn't in what was read, or nothing read came from the item's own website; name in
"missing" every rule whose fact wasn't shown, and fill the row with what was. drop: what was read shows it breaks
a rule; name it in "broken" and copy into "shown" the words read that break it (a number, a date, a place), and
fill the row with what was shown. Not finding a fact is never "shown". The verdict and the reason agree: when you
find no rule it breaks, the verdict is not drop.''')

GROUP_SYSTEM = """A list sorts its items into groups. Say which group one item is, by what it is.
A narrower or regional kind of a group is that group (a region's own style of a country's kind, a specialty within a
field, one member of a family the group names). Answer "none" only when it is a kind none of the groups covers.
Reply ONLY with JSON: {"group": "<one of the groups, written exactly as given, or none>"}"""

LOOK_SYSTEM = """An item nearly fits a list, but something the list needs wasn't in what was read about it.
Find it. Choose ONE next step and reply ONLY with JSON:
{"do": "fetch", "url": "<a page likely to show it: the item's own contact, about or detail page, a link seen>"}
{"do": "search", "query": "<a short search that would find it>"}
{"do": "read", "path": "<one of the owner's files>", "find": "<words to look for in it, or empty for its start>"}
{"do": "stop"}
Use only the kinds of step you are told you may. Prefer a link already seen over a guessed address. Stop when what
is lacking was found in what you did so far, or when it isn't likely to be found."""

# The same when nothing is read loosely (every rule kept as written): no loose verdict,
# but a drop still names the rule and the words read that break it, so code can tell a
# rule broken from a fact not found (that one is nearly there, never dropped).
DECIDE_STRICT = DECIDE_SYSTEM.replace(
    '''[{"id": <its id>, "verdict": "keep|maybe|drop", "reason": "<one short line>",
  "row": {"<column>": "<value>"}}]
keep: it meets every rule and the row is filled from the facts. maybe: it seems to fit but a fact the rules need
wasn't in what was read; also when nothing read came from the item's own website, unless a fact shown breaks a
rule. drop: it breaks a rule.''',
    '''[{"id": <its id>, "verdict": "keep|maybe|drop", "reason": "<one short line>",
  "row": {"<column>": "<value>"}, "missing": [<rule number>], "broken": [<rule number>],
  "shown": "<the words read that break it>"}]
keep: it meets every rule as written and the row is filled from the facts. maybe: it seems to fit but a fact a rule
needs wasn't in what was read, or nothing read came from the item's own website; name in "missing" every rule
whose fact wasn't shown, and fill the row with what was. drop: what was read shows it breaks a rule; name it in
"broken" and copy into "shown" the words read that break it (a number, a date, a place, what it is), and fill the
row with what was shown. Not finding a fact is never "shown": an item a fact wasn't found for is maybe. The
verdict and the reason agree: when you find no rule it breaks, the verdict is not drop.''')
assert DECIDE_STRICT != DECIDE_SYSTEM


class Work:
    """One work step: items from its sources, until its group is full or the sources
    run out."""

    def __init__(self, run: dict, step: dict, recipe: dict[str, Any], folder: Path, group: str = ""):
        recipe = {**recipe, "rules": list(recipe.get("rules") or []) + limit_rules(recipe)}
        self.run, self.step, self.recipe, self.folder, self.group = run, step, recipe, folder, group
        self.key = step["step_key"]
        self.cfg = run.get("config") or {}
        self.budget = search_budget(run, recipe)
        self.items_max = _int(self.cfg.get("funnel_items_max"), 1, 1_000_000, ITEMS_MAX)
        self.thinking = str(self.cfg.get("funnel_thinking") or "off").lower() == "on"
        self.charter = " ".join(str(run.get("task") or "").split())[:CHARTER_CHARS]
        self.places = list(recipe.get("places") or sorted({s["country"] for s in recipe["sources"] if s.get("country")}))
        self.skip_sites = skip_sites(recipe)
        self.brief = brief_text(folder)
        self.reference = "\n\n".join(x for x in (
            f"[the owner's documents and links, in brief]\n{self.brief}" if self.brief else "",
            reference_text(folder, recipe)) if x)
        self.reach = {c for r in recipe["rules"] for c in r.get("reach") or []}  # cells a contact page may fill
        self.query_rounds = _int(self.cfg.get("funnel_query_rounds"), 0, 1000, MAX_QUERY_ROUNDS)
        self.more_pages: list[str] = []
        self.tools: Tools | None = None
        self.creds: dict | None = None
        self.seen: set[str] = set()
        self.same = SameItems(recipe, self.skip_sites)
        self.jev_off = False  # the owner has no Jev connected: don't ask again this step
        self._tables: dict[str, list[tuple[str, int, dict]]] | None = None  # the owner's tables, by key value
        self.again: list[dict] = []  # items to judge once more: left unanswered, or just looked into
        self.look = look_on(run)
        self.look_steps = _int(self.cfg.get("funnel_look_steps"), 1, 12, LOOK_STEPS)
        wanted = self.quota(group) if group else rows_wanted(recipe)  # this step's rows
        self.look_max = _int(self.cfg.get("funnel_look_max"), 0, 100_000, max(LOOK_MAX, LOOKS_PER_ROW * wanted))
        self.looks = 0
        self.site_hosts: dict[str, set[str]] = {}  # host → the pages items' sites are on it (several: a directory)  # rows on the list or waiting, by their own site
        for r in _read_csv(folder / ROWS) + _read_csv(folder / LOOSE):
            self.same.add(r)
        self.kept: dict[str, int] = {}
        self.loose: dict[str, int] = {}       # rows on a looser reading, per group
        self.looked_in: dict[str, int] = {}   # items judged or ruled out, per group
        self.loosen = loosen_on(run, recipe)
        self.counts = {"looked": 0, "kept": 0, "maybe": 0, "loose": 0, "dropped": 0, "known": 0}
        self.known = known_keys(folder)
        self.reasons: dict[str, int] = {}
        self.pending: list[dict] = []
        self.recent: list[str] = []
        self.queries = self._load_queries()
        self.stopped = ""
        # the items looked at, as their listings show them: one found again under another
        # name ("Konoba X", then "konoba-x.hr") isn't judged twice
        self.looked = SameItems(recipe, self.skip_sites)
        for r in _read_csv(folder / ROWS) + _read_csv(folder / LOOSE):
            self.looked.add(r)
        for m in _read_csv(folder / MAYBE):
            try:
                got = json.loads(m.get("row") or "{}")
            except ValueError:
                got = {}
            if isinstance(got, dict):
                self.looked.add(got)
        for r in _read_csv(folder / ITEMS):
            self.seen.add(r.get("key", ""))
            g = r.get("group", "")
            self.looked_in[g] = self.looked_in.get(g, 0) + 1
            if r.get("status") == "kept":
                self.kept[g] = self.kept.get(g, 0) + 1
            elif r.get("status") == "loose":
                self.loose[g] = self.loose.get(g, 0) + 1

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
            return bool(q) and self._enough([self.group], q)
        if groups and all(groups.values()):
            return all(self._enough([g], n) for g, n in groups.items())
        target = int(self.recipe.get("target") or 0)
        return bool(target) and self._enough(list(set(self.kept) | set(self.loose)), target)

    def _enough(self, groups: list[str], wanted: int) -> bool:
        """Enough rows: strict ones, or, once the step has looked at enough items that
        more strict ones are unlikely (3 per row wanted, at least 30), strict and loose
        together. Strict rows still go first on the list."""
        strict = sum(self.kept.get(g, 0) for g in groups)
        if strict >= wanted:
            return True
        if not self.loosen:
            return False
        looked = self.looked_in.get(self.group, 0) if self.group else sum(self.looked_in.values())
        return looked >= max(LOOSE_AFTER_MIN, LOOSE_AFTER_PER * wanted) \
            and strict + sum(self.loose.get(g, 0) for g in groups) >= wanted

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
            if self.group:
                await self._take_handover()
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
            # While its group is short and the run has searches left, new searches are written
            # from what worked. It gives up after rounds in a row that put nothing on the list.
            rounds, dry = int(self.queries["rounds"].get(self.group, 0)), 0
            while (searches and not self.full() and not self.stopped and rounds < self.query_rounds
                   and dry < QUERY_ROUNDS_DRY and self.queries["searches"] < self.budget):
                rounds += 1
                self.queries["rounds"][self.group] = rounds
                self._save_queries()
                src = searches[(rounds - 1) % len(searches)]  # each of its sources in turn, not only the first
                fresh = await self.new_queries(src)
                if not fresh:
                    break
                before = self._rows_now()
                async for batch in self.candidates({**src, "queries": fresh}):
                    await self.handle(batch)
                    if self.full() or self.stopped:
                        break
                await self.flush()
                dry = 0 if self._rows_now() > before else dry + 1
            await self.flush()
        finally:
            await asyncio.to_thread(self.note, True)
        return self.result()

    def _rows_now(self) -> int:
        groups = [self.group] if self.group else list(set(self.kept) | set(self.loose))
        return sum(self.kept.get(g, 0) + self.loose.get(g, 0) for g in groups)

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
            named = await self._table_columns(src["path"], _cols, rows)
            if named is None:
                await self._event("funnel", f"couldn't read {src['path']} as {self.recipe['item']}s: none of its "
                                            f"columns ({', '.join(_cols[:12])}) is the list's {self.recipe['key']}")
                return
            if named:  # the list's own names beside the table's, so its key, filters and checks find them
                rows = [{**r, **{mine: r[theirs] for theirs, mine in named.items()
                                 if str(r.get(theirs) or "").strip() and not str(r.get(mine) or "").strip()}}
                        for r in rows]
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

    async def _table_columns(self, path: str, cols: list[str], rows: list[dict]) -> dict[str, str] | None:
        """Which of a table's columns hold the list's, when the table names them its own
        way ("Strain" for the list's "StrainID"): a row with nothing under the list's key
        got no key and was passed over without a word. {} when the table needs none (it
        has the key column, a name or a link); None when its key can't be found in it.
        By their names where only one fits, else by one short call with sample rows."""
        key = self.recipe["key"]
        low = {c.lower() for c in cols}
        site = str(self.recipe.get("site_column") or "").lower()
        if not key or key == "domain" or key.lower() in low or low & {"name", "url", "website"} or (site and site in low):
            return {}
        mine = [c for c in self.recipe["columns"] if c.lower() not in low]
        mine_low = {c.lower() for c in self.recipe["columns"]}  # a column the list has by that name stays that
        named: dict[str, str] = {}
        for c in mine:
            fits = [t for t in cols if t.lower() not in mine_low and _same_column(t, c)]
            if len(fits) == 1 and fits[0] not in named:
                named[fits[0]] = c
        if len(named) < len(mine):
            about = "\n".join(f"- {c}" + (f": {self.recipe['about'][c]}" if self.recipe["about"].get(c) else "")
                              for c in self.recipe["columns"])
            sample = "\n".join(json.dumps(r, ensure_ascii=False)[:400] for r in rows[:4])
            try:
                got = await self._ask("columns", COLUMNS_SYSTEM,
                                      f"# The list\n{(self.recipe.get('decide') or self.recipe['item'])[:600]}\n\n"
                                      f"# The list's columns\n{about}\n\n# The table ({path})\n"
                                      f"Columns: {', '.join(cols)}\nFirst rows:\n{sample}")
            except RuntimeError:
                raise  # no model to call at all
            except Exception as e:  # noqa: BLE001 — what the names alone gave
                log.warning("funnel columns call failed", run_id=self.run["id"], error=str(e))
                got = None
            said = got.get("columns") if isinstance(got, dict) and isinstance(got.get("columns"), dict) else {}
            taken = set(named.values())
            for theirs, ours in said.items():
                ours = next((c for c in mine if c.lower() == str(ours).strip().lower()), "")
                if theirs in cols and theirs.lower() not in mine_low and ours and ours not in taken and theirs not in named:
                    named[theirs] = ours
                    taken.add(ours)
        if key not in named.values():
            return None
        await self._event("funnel", f"{path}: read as " + ", ".join(f"{t} = {c}" for t, c in named.items()))
        return named

    def _owner_tables(self) -> dict[str, list[tuple[str, int, dict]]]:
        """The owner's tables as places to look an item up: each table's rows by the value
        of its key column, the first column whose values are all different (an ID, a
        name). Read once a step."""
        if self._tables is not None:
            return self._tables
        index: dict[str, list[tuple[str, int, dict]]] = {}
        extra = self.folder / "extra"
        for p in sorted(extra.iterdir()) if extra.is_dir() else []:
            if not p.is_file() or p.name.startswith(".") or p.name == LEDGER_FILE or p.suffix.lower() not in fx.TABLE_EXTS:
                continue
            try:
                cols, rows = fx.read_table(p, limit=JOIN_TABLE_ROWS + 1)
            except (OSError, ValueError):
                continue
            if len(rows) < 2 or len(rows) > JOIN_TABLE_ROWS:
                continue
            for c in cols:
                vals = [_join_value(r.get(c)) for r in rows]
                if all(vals) and len(set(vals)) == len(vals):
                    for i, (r, v) in enumerate(zip(rows, vals)):
                        index.setdefault(v, []).append((f"extra/{p.name}", i + 2, r))
                    break
        self._tables = index
        return index

    def _joined(self, item: dict) -> list[dict]:
        """Rows of the owner's other tables that are about this item: one whose key is the
        item's own key or name, or a value the item carries (the person it names, a code),
        and then one such value on from those rows. Found by code, before the item is
        judged: a fact kept in another file was otherwise left empty unless a rule
        happened to need it. Never the row the item itself came from."""
        index = self._owner_tables()
        if not index:
            return []
        out: list[dict] = []
        took = {str(item.get("origin") or "")}
        values = [self._name_of(item)] + [str(v) for v in item["fields"].values()]
        for _hop in range(2):
            more: list[str] = []
            for v in values:
                for rel, n, row in index.get(_join_value(v), []):
                    where = f"{rel} row {n}"
                    if where in took or len(out) >= JOIN_ROWS:
                        continue
                    took.add(where)
                    out.append({"source": where, "text": "\n".join(f"{k}: {x}" for k, x in row.items() if x)})
                    more += [str(x) for x in row.values()]
            values = more
        return out

    async def extract(self, text: str, origin: str, about: str = "") -> list[dict]:
        """The items in a text (a search's results, a page, a part of a document), by a
        short model call. Links to more list pages are kept for later."""
        cols = ", ".join(self.recipe["columns"])
        try:
            got = await self._ask("extract", EXTRACT_SYSTEM.format(item=self.recipe["item"]),
                                  f"# Looking for\n{self.recipe['item']}s"
                                  + (f" ({self.group})" if self.group else "") + (f": {about}" if about else "")
                                  + f"\n\n# Columns of the list\n{cols}\n\n# Text ({origin})\n{text[:12000]}")
        except RuntimeError:
            raise  # no model to call at all
        except Exception as e:  # noqa: BLE001 — a slow or broken call: these items are skipped, the step goes on
            log.warning("funnel extract call failed", run_id=self.run["id"], error=str(e))
            await self._event("funnel", f"couldn't pull items out of {origin[:120]} ({_plain_error(e)})")
            return []
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
            # never the whole page for an item it isn't alone on: what it says of the others
            # ("Konoba Lav: 4.6, open daily") would be read as this item's facts
            alone = len([x for x in items or [] if isinstance(x, dict) and str(x.get("name") or "").strip()]) == 1
            out.append({"fields": fields, "origin": origin, "extracted": True,
                        "text": own or (text if alone and len(text) <= 3000 else "")})
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
            where = fx.foreign_place(f"{item.get('text') or ''}\n" + " ".join(map(str, f.values())), self.places,
                                     name=self._name_of(item))
            if where:
                return f"not in {' or '.join(fx.place_phrase(c) for c in self.places)} (its listing says {where})"
        for flt in self.recipe["filters"]:
            col, op, val = flt["column"], flt["op"], flt.get("value")
            if flt.get("groups") and self.group and self.group not in flt["groups"]:
                continue  # a limit on another group's kind of item (a firm's size, on a list with people too)
            if op == "regex" and found is None and col in self.reach:
                continue  # how to reach it is settled once it's read: a listing rarely shows the address
            if op == "size_max" and (found or {}).get("headcount"):
                try:
                    over = [(int(n), src) for n, src in found["headcount"] if int(n) > int(val)]
                except (TypeError, ValueError):
                    over = []
                if over:  # its own pages say how many people it has
                    return f"{over[0][0]} people, said on {over[0][1]}"
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
                # a column the listing doesn't give ("Adresa" on a search hit) isn't known yet,
                # not a mismatch: the rules and checks see it once the item is read
                if op == "regex" and text and not re.search(str(val), text, re.I):
                    return f"{col} doesn't match"
                if op == "not_regex" and re.search(str(val), text, re.I):
                    return f"{'it' if col == '*' else col} matches what's excluded"
                if op == "in" and text and not any(_stems(text) & _stems(str(x)) or text.lower() == str(x).lower()
                                                   for x in val or []):
                    return f"{col} is {text}"  # it shares no word with any value it may take
                if op == "not_in" and text.lower() in {str(x).lower() for x in val or []}:
                    return f"{col} is {text}"
                if op in ("min", "max") and text:
                    if item.get("extracted") and item.get("text") and not fx.in_text(text, item["text"]):
                        # a value a model pulled out of a text that doesn't state it ("1" for "read by
                        # one person"): no ground to rule the item out on; its judging call reads the text
                        continue
                    span = number_span(text)
                    if span and ((op == "min" and span[1] < float(val)) or (op == "max" and span[0] > float(val))):
                        return f"{col} is {text}"
            except (re.error, TypeError, ValueError):
                continue
        return ""

    async def handle(self, items: list[dict]) -> None:
        for item in items:  # the hosts these items' sites are on, and which page, before any is read
            site = self._site_of(item)
            host = fx.domain_of(site)
            if host:
                self.site_hosts.setdefault(host, set()).add(_site_page(site))
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
            listing = self._listing_row(item)
            twin = self.looked.of(listing) if listing.get(self.recipe.get("site_column") or "") else ""
            if twin:  # looked at under another name: the same site, no address or phone that differs
                item["key"] = key
                self.seen.add(key)
                await self.record(item, "dropped", f"already looked at as {twin} (the same site)", stage="duplicate")
                continue
            self.looked.add(listing)
            item["site_seen"] = bool(listing.get(self.recipe.get("site_column") or ""))
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
        got = await asyncio.gather(*(self.gather(it) for it in wave), return_exceptions=True)
        for it, err in zip(wave, got):
            if isinstance(err, Exception):
                log.warning("funnel: reading an item failed", run_id=self.run["id"], error=str(err))
                await self.record(it, "dropped", f"couldn't be read ({_plain_error(err)})", stage="read")
                continue
            if not it.get("site_seen") and self._site_of(it):  # a site found for it by a search
                listing = self._listing_row(it)
                twin = self.looked.of(listing)
                if twin:
                    await self.record(it, "dropped", f"already looked at as {twin} (the same site)", stage="duplicate")
                    continue
                self.looked.add(listing)
            why = self.passes(it, it.get("found") or {}) or self._foreign_site(it)
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
        evidence += await asyncio.to_thread(self._joined, item)
        site = self.recipe.get("site_column") or ""
        if site and not self._site_of(item) and any(g["kind"] == "fetch" for g in self.recipe["gather"]):
            await self.find_site(item, evidence)
        own_site = self._site_of(item) if site else ""
        slot = next((k for k in item["fields"] if k.lower() == site.lower()), site) if site else ""
        if own_site and not str(item["fields"].get(slot) or "").strip():
            item["fields"][slot] = own_site  # its site, listed as its "url": what {Web} reads
        item["evidence"], item["found"] = evidence, self.facts(item, evidence)
        for g in self.recipe["gather"]:
            if self.passes(item, item["found"]):
                return  # it already fails a test: nothing more to read
            if g["kind"] == "search":
                q = render(g["query"], self._template_fields(item), site)
                if not q or self.queries["searches"] >= self.budget:
                    continue
                ok, text = await (await self._tools()).search(q, g.get("count", 5))
                self.queries["searches"] += 1
                self._save_queries()
                await self._event("web_search", f"query={q}")
                if ok:
                    words, own_site = name_words(self._name_of(item)), fx.domain_of(self._site_of(item))
                    for r in fx.search_results(text):
                        # a result about something else ("LinkedIn News") is read, but
                        # code takes no size from it
                        own = names_item(words, r["url"], r["title"]) or bool(
                            own_site and fx.same_site(fx.domain_of(r["url"]), own_site))
                        evidence.append({"source": r["url"], "text": r["text"], "own": own})
            elif g["kind"] == "fetch":
                base = render(g["url"], self._template_fields(item), site)
                if not base:
                    continue
                if not base.startswith(("http://", "https://")):
                    base = "https://" + base.lstrip("/")
                parts = urlsplit(base)
                root = f"{parts.scheme}://{parts.netloc}/"
                tools = await self._tools()
                if self._shared_site(base):
                    # the item's page on a directory or social site (its LinkedIn page, no
                    # site of its own found): that page only. The site's own about, news
                    # or contact pages aren't the item's (linkedin.com/news gave two firms
                    # another company's size).
                    urls = [base] if parts.path.strip("/") else []
                else:
                    # its own site: the page first, then the pages its own links lead to that
                    # the recipe asks for ("Kontakt (…/hr/kontakt-nas)"), never guessed
                    # addresses; guessed only when the page couldn't be read or shows no links
                    ok, text = await tools.fetch(base)
                    await self._event("web_fetch", f"url={base}")
                    if ok and text.strip():
                        evidence.append({"source": base, "text": text[:FETCH_CHARS]})
                    followed = site_links(text if ok else "", base, g["paths"])
                    if followed is None:
                        followed = [u for u in dict.fromkeys(urljoin(root, p.lstrip("/")) for p in g["paths"] if p)
                                    if u.rstrip("/") != base.rstrip("/")]
                    urls = followed
                if not urls:
                    continue
                got = await asyncio.gather(*(tools.fetch(u) for u in urls))
                for u, (ok, text) in zip(urls, got):
                    await self._event("web_fetch", f"url={u}")
                    if ok and text.strip():
                        e = {"source": u, "text": text[:FETCH_CHARS]}
                        if self._shared_site(base):
                            # a page on a site many share is the item's own only when it names it: an
                            # article or a video page that mentions it says other things' sizes
                            head = next((ln for ln in text.splitlines() if ln.strip()), "")[:200]  # its title
                            e["own"] = names_item(name_words(self._name_of(item)), u, head)
                        evidence.append(e)
            item["found"] = self.facts(item, evidence)

    def _contact_link(self, item: dict, row: dict | None = None) -> str:
        """The item's contact page (:meth:`_contact_page`), on the site its listing gave or,
        when the listing gave none (found on a directory, its site learnt while reading),
        the site its row names: only pages read there count, so a site merely named
        proves nothing."""
        col = self.recipe.get("site_column") or ""
        named = str((row or {}).get(col) or "").strip() if col else ""
        for site in dict.fromkeys(x for x in (self._site_of(item), named) if x):
            link = self._contact_page(item, site)
            if link:
                return link
        return ""

    def _contact_page(self, item: dict, site: str) -> str:
        """The contact page of an item's own site, from what was read: a page read there
        whose address says so, else a link on one of its pages that does (by its words or
        its address). "" when it has no site of its own or none was found: a directory's
        or a social site's contact page is never the item's."""
        own = fx.domain_of(site)
        if not own or self._shared_site(site) or any(fx.same_site(own, s) for s in self.skip_sites):
            return ""
        pages = [e for e in item.get("evidence") or [] if str(e.get("source") or "").startswith(("http://", "https://"))
                 and fx.same_site(fx.domain_of(e["source"]), own)]
        for e in pages:
            if _CONTACT_PAGE.search(urlsplit(e["source"]).path):
                return e["source"]
        for e in pages:
            for label, url in _LABELLED_LINK.findall(e.get("text") or ""):
                if (fx.same_site(fx.domain_of(url), own)
                        and (_CONTACT_PAGE.search(label) or _CONTACT_PAGE.search(urlsplit(url).path))):
                    return url
        return ""

    def _shared_site(self, url: str) -> bool:
        """Whether a site link is an item's page on a site many share, not its own site:
        a directory or social site; a host the sites of several of this run's items are
        on (a booking site, a city guide); or a page deep inside a host
        (/restoran/grad-zagreb/konoba-mid/), where an own site's front page never is.
        Only that page is read there, and nothing on it counts as the item's own."""
        d = fx.domain_of(url)
        if not d:
            return False
        if any(fx.same_site(d, s) for s in _SKIP_SITES + tuple(self.skip_sites)):
            return True
        if len(self.site_hosts.get(d, ())) >= 2:  # different pages of one host: items listed on it
            return True
        try:
            path = urlsplit(url if "://" in url else "http://" + url).path
        except ValueError:
            return False
        parts = [x for x in path.split("/") if x]
        if parts and _LANG_PART.match(parts[0]):
            parts = parts[1:]
        return len(parts) >= 2

    def _own_text(self, item: dict) -> tuple[str, str]:
        """The item's own site and what was read on it (its pages, not results about it).
        Nothing for an item whose site is a page on a site many share (a directory's
        phone or email is never the item's)."""
        site = self._site_of(item)
        domain = fx.domain_of(site)
        if not domain or self._shared_site(site):
            return "", ""
        return domain, "\n".join(e.get("text", "") for e in item.get("evidence") or []
                                 if fx.same_site(fx.domain_of(e.get("source", "")), domain))

    def _foreign_site(self, item: dict) -> str:
        """Why an item whose own pages point only to another country isn't on a list for
        this place (a +27 phone, an address on .co.za, a country named), or ""."""
        if not self.places:
            return ""
        domain, text = self._own_text(item)
        if not text and not domain:
            return ""
        where = fx.foreign_on_own_pages(text, domain, self.places, name=self._name_of(item))
        return f"its own site points to {where}, not {' or '.join(fx.place_phrase(c) for c in self.places)}" if where else ""

    def _site_of(self, item: dict) -> str:
        site = (self.recipe.get("site_column") or "").lower()
        f = {k.lower(): str(v or "").strip() for k, v in item["fields"].items()}
        for k in (site, "website", "url", "site", "homepage", "domain"):
            v = f.get(k, "")
            d = fx.domain_of(v) if v else ""
            if d and "." in d and not any(fx.same_site(d, s) for s in self.skip_sites):
                return v
        return ""

    def _listing_row(self, item: dict) -> dict[str, str]:
        """An item as its listing shows it, in the list's columns (its site and name too)."""
        low = {str(k).lower(): str(v or "") for k, v in item["fields"].items()}
        row = {c: low.get(c.lower(), "") for c in self.recipe["columns"]}
        site = self.recipe.get("site_column") or ""
        if site and not row.get(site):
            row[site] = self._site_of(item)
        if self.recipe["key"] and not row.get(self.recipe["key"]):
            row[self.recipe["key"]] = self._name_of(item)
        return row

    def _template_fields(self, item: dict) -> dict[str, str]:
        """An item's fields as a gather template reads them. An item found by a search
        carries its name as ``name``; a template names the list's own key column, and
        found nothing to fill it with, so that item's search was never made."""
        f, key = item["fields"], self.recipe["key"]
        if not key or key.lower() == str(self.recipe.get("site_column") or "").lower():
            return f  # keyed by its site: a name is no address to read
        if any(k.lower() == key.lower() and str(v or "").strip() for k, v in f.items()):
            return f
        name = self._name_of(item)
        return {**{k: v for k, v in f.items() if k.lower() != key.lower()}, key: name} if name else f

    def _name_of(self, item: dict) -> str:
        f = item["fields"]
        return str(f.get("name") or f.get(self.recipe["key"]) or next(iter(f.values()), "")).strip()

    async def find_site(self, item: dict, evidence: list[dict]) -> None:
        """An item found on a directory or a LinkedIn page has no site of its own to
        read: one search for its name, and the first result whose address is its own
        (``site_names_item``). None found: it has no site, and its judging call says
        so, rather than read someone else's page about it."""
        f = item["fields"]
        name = self._name_of(item)
        if not name or self.queries["searches"] >= self.budget:
            return
        place = fx.place_name(self.places[0]) if len(self.places) == 1 else ""
        kind = self.recipe["item"] if self.recipe["item"] != "item" and len(self.recipe["item"].split()) <= 2 else ""
        q = " ".join(f'"{name}" {kind} {place}'.split())  # a long item phrase ("US knowledge firm lead") only blurs it
        ok, text = await (await self._tools()).search(q, 8)
        self.queries["searches"] += 1
        self._save_queries()
        await self._event("web_search", f"query={q}")
        if not ok:
            return
        for r in fx.search_results(text):
            d = fx.domain_of(r["url"])
            if not d or any(fx.same_site(d, s) for s in self.skip_sites) or not site_names_item(name, d, r["title"]):
                continue
            f[self.recipe["site_column"]] = f"https://{d}"
            evidence.append({"source": r["url"], "text": r["text"]})
            return

    def facts(self, item: dict, evidence: list[dict]) -> dict[str, list[tuple[str, str]]]:
        """What code finds in what was read: each extractor's values, with the page."""
        site = self.recipe.get("site_column") or ""
        link = next((v for k, v in item["fields"].items() if k.lower() in (site.lower(), "url", "website") and v), "")
        domain = "" if self._shared_site(str(link)) else fx.domain_of(link)  # a directory's emails aren't "its own"
        names = list(self.recipe["extract"])
        if any(f["op"] in ("size_max", "size_min") for f in self.recipe["filters"]):
            names += [n for n in ("size_band", "headcount") if n not in names]
        found: dict[str, list[tuple[str, str]]] = {}
        for e in evidence:
            for name in names:
                if name in ("size_band", "headcount") and not e.get("own", True):
                    continue  # another company's size, read off a page that isn't about this one
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
        alone = [it for it in batch if it.get("asked_again")]
        if alone and len(batch) > 1:
            # left out of an answer once: each is asked about in a call of its own, where a
            # small model that lost it among the others has only it to answer for
            for it in alone:
                self.pending = [it]
                await self.flush()
            batch = [it for it in batch if not it.get("asked_again")]
            if not batch:
                return
        lines = []
        for i, it in enumerate(batch, start=1):
            facts = [f"  {name}: " + "; ".join(f"{v} [{src}]" for v, src in vals[:8])
                     for name, vals in (it.get("found") or {}).items()]
            share = READ_CHARS // max(1, len(it.get("evidence") or []))
            marks = {e["source"]: [v for vals in (it.get("found") or {}).values() for v, src in vals
                                   if src == e["source"]] for e in it.get("evidence") or []}
            excerpt = "\n".join(f"  [{e['source']}]\n  " + page_excerpt(e["text"], share, marks.get(e["source"]))
                                for e in it.get("evidence") or [])
            lines.append(f"## id {i}\nfields: {json.dumps(it['fields'], ensure_ascii=False)[:1500]}\n"
                         + ("facts Captain found:\n" + "\n".join(facts) + "\n" if facts else "")
                         + f"what was read:\n{excerpt}")
        ours = set(judgement_columns(self.recipe, self.reference)) if self.reference else set()
        cols = "\n".join(f"- {c}" + (f": {self.recipe['about'][c]}" if self.recipe["about"].get(c) else "")
                         + (OWNER_COLUMN_NOTE if c in ours else "") for c in self.recipe["columns"])
        groups = (f"\n# {self.recipe['group_by']}\n" + ", ".join(self.recipe["groups"])
                  + "\nEvery answer, a drop too, also carries \"group\": the one of these the item is (by what it "
                  "is, whatever this step looks for), written exactly as here, or \"none\"."
                  + (f"\nThis step looks for: {self.group}. An item that fits another of these groups instead is "
                     f"that group's find: keep it (or answer loose or maybe) with that group in "
                     f"{self.recipe['group_by']}, never drop it for not being {self.group}. Drop an item for its "
                     "kind only when it fits none of them; one that fits several takes the closest. A narrower or "
                     "regional kind of a group is that group (a region's own style of a country's kind, a "
                     "specialty within a field): it fits it."
                     if self.group else "")) if self.recipe["groups"] else ""
        try:
            got = await self._ask("decide", DECIDE_LOOSE if self.loosen else DECIDE_STRICT,
                                  f"# Today\n{time.strftime('%Y-%m-%d')} (a date up to today is in the past, never "
                                  "\"in the future\")\n\n"
                                  f"# Rules\n{self.recipe['decide'] or '(take every item that fits the list)'}\n\n"
                                  + self._numbered_rules()
                                  + self._tiers_text()
                                  + (f"# The goal, in the owner's words\n{self.charter}\n(These words decide. What "
                                     "each column takes, and any rule above that leaves out something they say, "
                                     "such as what to do when a fact isn't found, follows them.)\n\n"
                                     if self.charter else "")
                                  + (f"# Reference\n{self.reference}\n\n" if self.reference else "")
                                  + f"# Columns\n{cols}{groups}\n\n# Items\n" + "\n\n".join(lines)
                                  + ("\n\n(Asked a second time: the first answer couldn't be read. A JSON array "
                                     "with one object for each id above, nothing else.)"
                                     if any(it.get("asked_again") for it in batch) else ""), items=len(batch))
        except Exception as e:  # noqa: BLE001 — these go back for the next call
            log.warning("funnel decide call failed", run_id=self.run["id"], error=str(e))
            got = None
        verdicts = answers_by_id(got, len(batch))
        for i, it in enumerate(batch, start=1):
            if i in verdicts and isinstance(verdicts[i].get("row"), dict):
                self._fill_from_facts(verdicts[i]["row"], it)
        await self._fill_owner_columns(verdicts, batch)
        again = self.again
        for i, it in enumerate(batch, start=1):
            v = verdicts.get(i)
            if v is None:
                if not it.get("asked_again"):  # left out of the answer: asked once more, never dropped for it
                    it["asked_again"] = True
                    again.append(it)
                    continue
                await self.record(it, "dropped", "the judging call gave no answer for it, twice")
                continue
            verdict = str(v.get("verdict") or "").lower()
            reason = " ".join(str(v.get("reason") or "").split())[:400]  # whole: its last words say why
            if verdict == "loose" and not self.loosen:
                verdict = "maybe"  # no looser reading in this run: it's nearly there
            if verdict == "drop" and self._unbacked_drop(v, it):
                # it named only rules that may go unshown, and quoted nothing read that breaks them:
                # not finding a fact isn't breaking the rule
                v = {**v, "missing": list(v.get("missing") or []) + list(v.get("broken") or [])}
                verdict = "maybe"
            loosened = []
            if verdict in ("maybe", "loose") and not it.get("looked"):
                # what it lacks is looked for first: a fact found beats a looser reading or an
                # alternative for a fact not found
                lacking = self._named(v.get("missing")) + self._named(v.get("loosened"))
                if await self._go_look(it, reason, lacking):
                    self.again.append(it)
                    continue
            if verdict == "maybe":
                met, rest = self._alternatives(v)
                far = [r for r in met if r.get("reach")
                       and not self._contact_link(it, v.get("row") if isinstance(v.get("row"), dict) else None)]
                if far:  # no contact page read on its own site: that alternative isn't met
                    met, rest = [r for r in met if r not in far], rest + far
                has_row = isinstance(v.get("row"), dict) and any(str(x).strip() for x in v["row"].values())
                if met and not rest and has_row:
                    doubt = await self._alternative_doubt(v, it, met, reason)
                    if not doubt:
                        # what it lacked, the charter itself says how to take ("with no rating, judged
                        # good enough"): met, not nearly there, whether or not rules may be read loosely
                        verdict = "keep"
                        note = "met by the charter's own words for a fact not found: " + "; ".join(
                            f"{r['rule']} ({r['if_not_shown']})" for r in met)
                        reason = f"{reason}; {note}" if reason else note
                    else:  # its own alternative isn't met after all: nearly there, saying why
                        reason = f"{reason}; {doubt}" if reason else doubt
                        v = {**v, "_lacked": [r["rule"] for r in met]}
                elif met:  # what's left to find; the near miss still records all it lacked
                    v = {**v, "missing": [r["rule"] for r in rest], "_lacked": [r["rule"] for r in met + rest]}
            if verdict == "maybe" and self.loosen:
                verdict, loosened = self._maybe_or_loose(v)
            elif verdict == "loose":
                loosened = self._loosened(v.get("loosened"))
            if verdict == "loose" and not loosened:
                verdict = "maybe"  # it named only rules that never loosen: what it lacks must still be found
            if verdict in ("keep", "loose"):
                filled = v.get("row") if isinstance(v.get("row"), dict) else {}
                empty = [c for c in judgement_columns(self.recipe, self.reference) if not str(filled.get(c) or "").strip()]
                unshown = unshown_musts(self.recipe, filled)
                if empty:  # it fits, but the row isn't finished: nearly there, never on the list half-filled
                    verdict, reason = "maybe", f"left {', '.join(empty)} empty" + (f" ({reason})" if reason else "")
                elif unshown:  # a rule that never loosens, with nothing shown for it
                    verdict, reason = "maybe", "shows nothing for: " + "; ".join(unshown)
            if verdict in ("keep", "loose"):
                row = {c: " ".join(str((v.get("row") or {}).get(c) or "").split()) for c in self.recipe["columns"]}
                named = self._group_named(row.get(self.recipe["group_by"], "")) if self.recipe["group_by"] else ""
                named = named or self._answer_group(v)  # its typed "group", when the column says none of them
                if named and self.recipe["group_by"]:
                    # a group written loosely ("Italian") is the list's group, written as the list does
                    row[self.recipe["group_by"]] = group_cell(self.recipe, named, row.get(self.recipe["group_by"], ""))
                elif self.recipe["group_by"] and self.group and not row.get(self.recipe["group_by"]):
                    row[self.recipe["group_by"]] = group_cell(self.recipe, self.group)  # no group said: this step's
                relax = {c for r in loosened for c in r.get("columns") or []}
                read = {e["source"]: e["text"] for e in it.get("evidence") or []}
                reach = self.reach
                unproven = {c["column"]: row[c["column"]] for c in self.recipe["checks"]
                            if c["op"] == "in_evidence" and row.get(c["column"])
                            and check_row({**self.recipe, "checks": [c]}, row, read)}
                if unproven:  # a value what was read doesn't bear out is left out, never the item with it
                    for col in unproven:
                        row[col] = ""
                    self._fill_from_facts(row, it)  # what its own pages do show (their email, not a guessed one)
                    unproven = {c: v for c, v in unproven.items() if not row.get(c)}
                if unproven:
                    note = "couldn't confirm " + ", ".join(f"{c} '{v[:60]}'" for c, v in unproven.items()) \
                        + " in what was read"
                    reason = f"{reason}; {note}" if reason else note
                    soft = loosenable(self.recipe) if self.loosen else []
                    covered = {c for r in soft for c in r.get("columns") or []}
                    if any(c not in covered and c not in reach for c in unproven):  # a cell the list needs: nearly there
                        await self._nearly(it, row, named or self.group, reason)
                        continue
                    eased = [r for r in soft if set(r.get("columns") or []) & set(unproven) and r not in loosened
                             and not any(str(row.get(x) or "").strip() for x in r.get("columns") or [])]
                    if eased:  # a soft rule now shown by nothing: met by its looser reading
                        loosened, verdict = loosened + eased, "loose"
                        relax |= {c for r in eased for c in r.get("columns") or []}
                unreached = ""
                for r in self.recipe["rules"]:
                    cols = r.get("reach") or []
                    if not cols or any(row.get(c) for c in cols):
                        continue
                    link = self._contact_link(it, row)
                    if not link:
                        lacked = self._named(v.get("missing")) + self._named(v.get("_lacked"))
                        if r in loosened or (r not in lacked and not any(
                                c["op"] == "required" and c["column"] in cols for c in self.recipe["checks"])):
                            continue  # met loosely, or an address the list doesn't require and the
                            # judging call didn't miss: empty, as before
                        unreached, lacks = "no address published, and no contact page found on its own site", r
                        break
                    # no address published, but its own site has a contact page: that is how it's
                    # reached, in every run. Code found the page; its link stands where the address would.
                    row[cols[0]] = link
                    read.setdefault(link, "")
                    relax |= set(cols)
                    loosened = [x for x in loosened if x is not r]
                    note = f"no address published; reached by its contact page ({cols[0]})"
                    reason = f"{reason}; {note}" if reason else note
                if unreached:
                    await self._nearly(it, row, named or self.group, f"{reason}; {unreached}" if reason else unreached,
                                       [lacks])
                    continue
                if verdict == "loose" and not loosened:
                    verdict = "keep"  # what it met loosely, it now meets by the contact page code found
                gaps = [c["column"] for c in self.recipe["checks"]
                        if c["op"] == "required" and not row.get(c["column"]) and c["column"] not in relax]
                if gaps:  # a cell it left empty isn't a rule broken: a soft rule's looser reading, or nearly there
                    soft = [r for r in loosenable(self.recipe) if set(r.get("columns") or []) & set(gaps)] \
                        if self.loosen else []
                    still = [c for c in gaps if not any(c in (r.get("columns") or []) for r in soft)]
                    if still:
                        await self._nearly(it, row, named or self.group,
                                           f"left {', '.join(still)} empty" + (f" ({reason})" if reason else ""))
                        continue
                    loosened = loosened + [r for r in soft if r not in loosened]
                    verdict = "loose"
                    relax |= {c for r in soft for c in r.get("columns") or []}
                problem = check_row(self.recipe, row, read, relax=relax)
                if problem:
                    await self.record(it, "dropped", f"check: {problem}", stage="check")
                    continue
                group = named or self._group_named(row.get(self.recipe["group_by"], "")
                                                   if self.recipe["group_by"] else "") or self.group
                if self.recipe["group_by"] and group in self.recipe["groups"]:  # as the list names it
                    row[self.recipe["group_by"]] = group_cell(self.recipe, group, row.get(self.recipe["group_by"], ""))
                twin = self.same.of(row)
                if twin:  # found again under another name: its own site (and phone or email) say so
                    await self.record(it, "dropped", f"the same {self.recipe['item']} as {twin}, found under "
                                      "another name", group=group, stage="duplicate")
                    continue
                if self.quota(group) and self.kept.get(group, 0) >= self.quota(group):
                    await self.record(it, "dropped", f"{group} is already full", group=group, stage="full")
                    continue
                key = norm_key(row.get(self.recipe["key"], "")) if self.recipe["key"] in row else it["key"]
                if key and key != it["key"] and (key in self.seen or key in self.known):
                    await self.record(it, "dropped", "already looked at under another name"
                                      + (" in an earlier run" if key in self.known else ""), stage="duplicate")
                    continue
                self.seen.add(key)
                self.same.add(row)
                if verdict == "loose":
                    names = "; ".join(r["rule"] for r in loosened)
                    self.loose[group] = self.loose.get(group, 0) + 1
                    _append_csv(self.folder / LOOSE, self.recipe["columns"] + [_GROUP_COL, _LOOSE_COL],
                                {**row, _GROUP_COL: group, _LOOSE_COL: names})
                    await self.record(it, "loose", f"meets only the looser reading of: {names}"
                                      + (f" ({reason})" if reason else ""), group=group, stage="loose")
                    continue
                self.kept[group] = self.kept.get(group, 0) + 1
                _append_csv(self.folder / ROWS, self.recipe["columns"] + [_GROUP_COL], {**row, _GROUP_COL: group})
                await self.record(it, "kept", reason or "meets the rules", group=group)
            elif verdict == "maybe":
                hint = str((v.get("row") or {}).get(self.recipe["group_by"]) or "") if self.recipe["group_by"] else ""
                got = v.get("row") if isinstance(v.get("row"), dict) else {}
                await self._nearly(it, got, self._group_named(hint) or self._answer_group(v) or self.group,
                                   reason or "a fact it needs wasn't found",
                                   self._named(v.get("_lacked") or v.get("missing")))
            else:
                other = await self._drop_goes_to(v, it, reason)
                if other and other != self.group:  # "Japanese, not Croatian": the Asian step judges it
                    await self._hand_over(it, other, reason)
                    continue
                if other:  # dropped for its kind, yet its kind is this group's ("Dalmatian" is Croatian)
                    got = v.get("row") if isinstance(v.get("row"), dict) else {}
                    note = f"its kind is {other} after all, so not dropped for it"
                    await self._nearly(it, got, other, f"{reason}; {note}" if reason else note,
                                       self._named(v.get("missing")))
                    continue
                await self.record(it, "dropped", reason or "breaks a rule")
        if self.again:  # judged again now, in a call of their own (left unanswered, or looked into)
            self.again, self.pending = [], self.again + self.pending
            await self.flush()

    async def _nearly(self, it: dict, got: dict, group: str, reason: str,
                      missing: list[dict[str, Any]] | None = None) -> None:
        """An item that seems to fit but lacks something. First it is looked into (once:
        :meth:`_go_look`) and judged again on what that found; still lacking, it is a
        maybe, with the cells found so far (a value that isn't in what was read isn't
        passed on) and the rules the judging call named as not shown."""
        if await self._go_look(it, reason, missing or []):
            self.again.append(it)
            return
        await self.record(it, "maybe", reason, group=group)
        name = it["fields"].get("name") or next(iter(it["fields"].values()), "")
        cells = {c: " ".join(str((got or {}).get(c) or "").split()) for c in self.recipe["columns"]}
        if self.recipe["group_by"] and group in self.recipe["groups"]:
            cells[self.recipe["group_by"]] = group_cell(self.recipe, group, cells.get(self.recipe["group_by"], ""))
        read = {e["source"]: e["text"] for e in it.get("evidence") or []}
        for c in self.recipe["checks"]:
            if c["op"] == "in_evidence" and cells.get(c["column"]) and check_row(
                    {**self.recipe, "checks": [c]}, cells, read):
                cells[c["column"]] = ""
        _append_csv(self.folder / MAYBE, _MAYBE_COLS,
                    {"key": it["key"], "name": name, "group": group,
                     "fields": json.dumps(it["fields"], ensure_ascii=False),
                     "why": reason, "sources": " ".join(e["source"] for e in it.get("evidence") or []),
                     "row": json.dumps({c: x for c, x in cells.items() if x}, ensure_ascii=False),
                     "missing": "; ".join(dict.fromkeys(r["rule"] for r in missing or []))})

    def _look_files(self) -> list[str]:
        """The owner's files a look may read: what the run was given, but the record of
        earlier runs and a recording."""
        extra = self.folder / "extra"
        if not extra.is_dir():
            return []
        return [f"extra/{p.name}" for p in sorted(extra.iterdir())
                if p.is_file() and not p.name.startswith(".") and p.name != LEDGER_FILE
                and p.suffix.lower() in fx.TABLE_EXTS + fx.DOC_EXTS][:LOOK_FILES]

    def _file_text(self, rel: str, find: str = "") -> str:
        """A file of the run's as text; with ``find``, only its lines that carry one of
        those words (and the one after each), so a long file gives up the part asked for."""
        path = self.folder / rel
        if path.suffix.lower() in fx.TABLE_EXTS:
            cols, rows = fx.read_table(path, limit=2000)
            lines = [", ".join(cols)] + [", ".join(f"{k}: {v}" for k, v in r.items() if v) for r in rows]
        else:
            # by line, and by sentence where a reader gives the text as one run
            lines = [x for x in re.split(r"\n+|(?<=[.!?])\s+", fx.read_document(path)) if x.strip()]
        words = [w for w in re.findall(r"[^\W_]{3,}", _fold(find))]
        if words:
            keep: list[str] = []
            for i, line in enumerate(lines):
                if any(w in _fold(line) for w in words):
                    keep += lines[i:i + 2]
            lines = list(dict.fromkeys(keep))
        return "\n".join(lines)[:FETCH_CHARS]

    async def _go_look(self, it: dict, reason: str, missing: list[dict[str, Any]]) -> bool:
        """Look for what a nearly-there item lacks, as a person would: a few steps, each
        one page to read, one search, or one of the owner's files, chosen by the model
        from what it has seen so far. What it reads joins the item's evidence, so every
        value is still checked against what was read. True when it read something new
        (the item is judged again). Once per item, while its group is short, within the
        step's number of looks; never for a run with nothing to look in."""
        if not self.look or it.get("looked") or self.full() or self.stopped or self.looks >= self.look_max:
            return False
        web = bool(self.recipe.get("gather")) or any(s["kind"] in ("search", "page", "links") for s in self.recipe["sources"])
        files = self._look_files()
        if not web and not files:
            return False
        it["looked"] = True
        self.looks += 1
        evidence = it.setdefault("evidence", [])
        read = {e["source"] for e in evidence}
        name = self._name_of(it)
        log_lines: list[str] = []
        new = 0
        for _ in range(self.look_steps):
            seen = [u for e in evidence for u in fx.links(e.get("text", ""))][:40]
            user = (f"# The list\n{self.recipe['decide'] or self.recipe['item']}\n\n"
                    f"# The item\n{name}\nfields: {json.dumps(it['fields'], ensure_ascii=False)[:800]}\n\n"
                    "# What it still lacks\n" + ("\n".join(f"- {r['rule']}" for r in missing) or "- (see the reason)")
                    + f"\nreason: {reason}\n\n"
                    "# Already read\n" + ("\n".join(f"- {x}" for x in sorted(read)) or "- nothing") + "\n\n"
                    + ("# Links seen on what was read\n" + "\n".join(f"- {u}" for u in dict.fromkeys(seen)) + "\n\n"
                       if seen and web else "")
                    + ("# The owner's files\n" + "\n".join(f"- {f}" for f in files) + "\n\n" if files else "")
                    + ("# What you did so far\n" + "\n".join(log_lines) + "\n\n" if log_lines else "")
                    + "You may: " + ", ".join((["fetch a page", "search the web"] if web else [])
                                              + (["read one of the owner's files"] if files else [])) + ", or stop.")
            try:
                got = await self._ask("look", LOOK_SYSTEM, user)
            except Exception as e:  # noqa: BLE001 — the item stays as it was
                log.warning("funnel look call failed", run_id=self.run["id"], error=str(e))
                break
            do = str((got or {}).get("do") or "").lower() if isinstance(got, dict) else ""
            if do == "fetch" and web:
                url = str(got.get("url") or "").strip()
                if not url.startswith(("http://", "https://")) or url in read:
                    log_lines.append(f"- fetch {url}: not a new page")
                    continue
                ok, text = await (await self._tools()).fetch(url)
                await self._event("web_fetch", f"url={url} (looking into {name})")
                read.add(url)
                if ok and text.strip():
                    evidence.append({"source": url, "text": text[:FETCH_CHARS]})
                    new += 1
                log_lines.append(f"- fetched {url}: " + (page_excerpt(text, 600) if ok and text.strip() else "nothing there"))
            elif do == "search" and web:
                q = " ".join(str(got.get("query") or "").split())[:200]
                if not q or self.queries["searches"] >= self.budget:
                    break
                ok, text = await (await self._tools()).search(q, 5)
                self.queries["searches"] += 1
                self._save_queries()
                await self._event("web_search", f"query={q} (looking into {name})")
                words = name_words(name)
                hits = [r for r in fx.search_results(text) if r["url"] not in read] if ok else []
                for r in hits:
                    evidence.append({"source": r["url"], "text": r["text"], "own": names_item(words, r["url"], r["title"])})
                    read.add(r["url"])
                new += len(hits)
                log_lines.append(f"- searched \"{q}\": " + ("; ".join(f"{r['title'][:60]} ({r['url']})" for r in hits[:5])
                                                             or "nothing new"))
            elif do == "read" and files:
                rel = str(got.get("path") or "").strip()
                find = " ".join(str(got.get("find") or "").split())[:120]
                key = f"{rel}#{find}" if find else rel
                if rel not in files or key in read:
                    log_lines.append(f"- read {rel}: not a file to read, or read already")
                    continue
                try:
                    text = await asyncio.to_thread(self._file_text, rel, find)
                except Exception as e:  # noqa: BLE001 — an unreadable file is nothing found
                    text = ""
                    log.debug("funnel look: couldn't read a file", path=rel, error=str(e))
                read.add(key)
                await self._event("read", f"path={rel}" + (f" find={find}" if find else "") + f" (looking into {name})")
                if text.strip():
                    evidence.append({"source": key, "text": text})
                    new += 1
                log_lines.append(f"- read {key}: " + (page_excerpt(text, 600) if text.strip() else "nothing there"))
            else:
                break
        it["look_read"] = new
        if new:
            it["found"] = self.facts(it, evidence)
        return bool(new)

    def _fill_from_facts(self, row: dict, item: dict) -> None:
        """Cells a judging call left empty that code found on the item's own pages: its
        email (on its own domain) with the page it's on and whether it's a shared inbox,
        its phone, its US state from an address. A fact read is never left out."""
        cols = self.recipe["columns"]
        domain, _text = self._own_text(item)
        if not domain:
            return  # no site of its own: nothing of its own to take a fact from
        own = [e for e in item.get("evidence") or [] if fx.same_site(fx.domain_of(e.get("source", "")), domain)]

        def empty(c: str) -> bool:
            return bool(c) and c in cols and not str(row.get(c) or "").strip()

        mails = [(m, e["source"]) for e in own for m in fx.emails(e.get("text", ""), domain)]
        mail_col = next((c for c in cols if re.search(r"e-?mail", c, re.I) and not re.search(r"source|url|type", c, re.I)), "")
        if mails and empty(mail_col):
            value, src = mails[0]
            row[mail_col] = value
            source = next((x.get("source_column") for x in self.recipe["checks"]
                           if x["column"] == mail_col and x["op"] == "in_evidence" and x.get("source_column")), "") \
                or next((c for c in cols if c != mail_col and re.search(r"e-?mail", c, re.I) and re.search(r"source|url", c, re.I)), "")
            if empty(source) and re.match(r"https?://", src):
                row[source] = src
            kind = next((c for c in cols if re.search(r"e-?mail", c, re.I) and re.search(r"type|kind", c, re.I)), "")
            values = [str(v) for x in self.recipe["checks"] if x["column"] == kind and x["op"] == "in" for v in x.get("value") or []]
            shared = next((v for v in values if re.search(r"shared|role|generic|general", v, re.I)), "")
            named = next((v for v in values if re.search(r"personal|direct|named|individual", v, re.I)), "")
            if empty(kind) and shared and named:
                row[kind] = shared if fx.SHARED_LOCAL.match(value.split("@", 1)[0]) else named
        phones = [(n, e["source"]) for e in own for n in fx.phones(e.get("text", ""))]
        phone_col = next((c for c in cols if _PHONE_COL.search(c) and not re.search(r"source|url", c, re.I)), "")
        if phones and empty(phone_col):
            row[phone_col] = phones[0][0]
        for rule in self.recipe.get("rules") or []:
            if rule.get("kind") != "must" or _fact_of(rule) != "place" or not rule.get("columns"):
                continue
            rcols = [c for c in rule["columns"] if c in cols]
            addr = [c for c in rcols if _ADDRESS.search(c)]
            if addr:  # its address, though another of the rule's cells (a district) was filled
                to = next((c for c in addr if not str(row.get(c) or "").strip()), "")
            else:
                to = rcols[0] if rcols and not any(str(row.get(c) or "").strip() for c in rcols) else ""
            if not to:
                continue
            line = place_line("\n".join(e.get("text", "") for e in own), rule["rule"], self.places)
            if line:
                row[to] = line
        state_col = next((c for c in cols if re.fullmatch(r"(us\s*)?state", c.strip(), re.I)), "")
        if "US" in self.places and empty(state_col):
            state = fx.us_state("\n".join(e.get("text", "") for e in own), address_only=True)
            if state:
                row[state_col] = state

    async def _fill_owner_columns(self, verdicts: dict[int, dict], batch: list[dict] | None = None) -> None:
        """The columns filled from the owner's own material (:func:`judgement_columns`),
        where a judging call left them empty: one short call for the batch, from that
        material and the row so far. A row they're still empty in is nearly there."""
        cols = judgement_columns(self.recipe, self.reference)
        if not cols:
            return
        todo = {i: v for i, v in verdicts.items()
                if str(v.get("verdict") or "").lower() in ("keep", "loose", "maybe") and isinstance(v.get("row"), dict)
                and any(not str(v["row"].get(c) or "").strip() for c in cols)}
        if not todo:
            return
        values = {x["column"]: x.get("value") or [] for x in self.recipe["checks"] if x["op"] == "in"}
        spec = "\n".join(f"- {c}" + (f": {self.recipe['about'][c]}" if self.recipe["about"].get(c) else "")
                         + (f" (exactly one of: {', '.join(map(str, values[c]))})" if values.get(c) else "") for c in cols)
        def read(i: int) -> str:  # a little of what was read about it, so a reason can cite a fact
            it = (batch or [])[i - 1] if batch and 0 < i <= len(batch) else {}
            return " ".join(" ".join(e.get("text", "").split())[:600] for e in (it.get("evidence") or [])[:3])[:1500]

        items = "\n\n".join(f"## id {i}\n{json.dumps(v['row'], ensure_ascii=False)[:1500]}\nread: {read(i)}"
                             for i, v in todo.items())
        try:
            got = await self._ask("owner", "Fill the owner's columns for each item, from the owner's material and what is "
                                           "known about the item. Reply ONLY with JSON: {\"<id>\": {\"<column>\": "
                                           "\"<value>\"}}. A column with a fixed set of values takes one of them exactly. "
                                           "Each value joins the two: it names what of the owner's fits (by its "
                                           "name in their material) and ties it to something read about this item "
                                           "(what it does, who it serves), never a sentence that would fit any item. "
                                           "Keep your thinking short.",
                                  f"# The owner's material\n{self.reference or '(none)'}\n\n# Columns to fill\n{spec}"
                                  f"\n\n# Items\n{items}")
        except Exception as e:  # noqa: BLE001 — the rows stay as they are: nearly there
            log.warning("funnel owner-columns call failed", run_id=self.run["id"], error=str(e))
            return
        if not isinstance(got, dict):
            return
        for i, v in todo.items():
            fill = got.get(str(i)) if isinstance(got.get(str(i)), dict) else {}
            for c in cols:
                if not str(v["row"].get(c) or "").strip() and str(fill.get(c) or "").strip():
                    v["row"][c] = " ".join(str(fill[c]).split())[:500]

    def _group_named(self, value: str) -> str:
        """The list's group a value names: exactly, else by its words and the words the
        setup gave for each group ("Italian" → "talijanska kuhinja"); "" for none."""
        v = " ".join(str(value or "").split())
        if not v or not self.recipe["groups"]:
            return ""
        exact = next((g for g in self.recipe["groups"] if g.lower() == v.lower()), "")
        if exact:
            return exact
        said = set(_group_words(v))
        words = {g: set(_group_words(g, (self.recipe.get("group_words") or {}).get(g))) for g in self.recipe["groups"]}
        shared = set.intersection(*words.values()) if words else set()
        scores = {g: len(said & (words[g] - shared)) for g in self.recipe["groups"]}
        best = max(scores.values(), default=0)
        tops = [g for g, n in scores.items() if n == best]
        return tops[0] if best and len(tops) == 1 else ""

    def _naming(self) -> dict[str, tuple[set[str], list[str]]]:
        """How a text names each group: the stems of its own name (but those every group's
        name shares, "kuhinja"), and the setup's group words as whole phrases ("traditional
        croatian food", never "food" alone)."""
        names = {g: {w[:6] for w in re.findall(r"[a-z]+", _fold(g)) if len(w) >= 4 and w not in _GROUP_STOP}
                 for g in self.recipe["groups"]}
        common = set.intersection(*names.values()) if names else set()
        phrases = {g: [" ".join(re.findall(r"[a-z0-9]+", _fold(x))) for x in
                       (self.recipe.get("group_words") or {}).get(g) or []] for g in self.recipe["groups"]}
        everyone = set.intersection(*(set(v) for v in phrases.values())) if phrases else set()
        return {g: (names[g] - common, [x for x in phrases[g] if x and x not in everyone]) for g in self.recipe["groups"]}

    def _names_group(self, group: str, text: str, naming: dict | None = None) -> bool:
        stems, phrases = (naming or self._naming())[group]
        folded = _fold(text)
        words = re.findall(r"[a-z0-9]+", folded)
        if any(w.startswith(st) for st in stems for w in words):
            return True
        return any(re.search(r"(?<![a-z0-9])" + r"[a-z0-9]*\W+".join(re.escape(w[:5]) for w in x.split())
                             + r"[a-z0-9]*", folded) for x in phrases)

    def _answer_group(self, v: dict) -> str:
        """The list's group a judging answer's typed "group" names, or ""."""
        if not self.recipe["groups"]:
            return ""
        g = " ".join(str(v.get("group") or "").split())
        if not g or g.lower() == "none":
            return ""
        return next((x for x in self.recipe["groups"] if x.lower() == g.lower()), "") or self._group_named(g)

    async def _drop_goes_to(self, v: dict, item: dict, reason: str) -> str:
        """The list's group a dropped item is, when it was dropped for its kind: another
        group's (that group's find), or this step's own (a narrower kind it fits after
        all: nearly there, not dropped); "" when it fits none or the drop stands. Only for
        a drop about its kind: one also shown to break another rule (a hotel's
        restaurant, a 4.3 rating read) fails there too, while a fact not found (no phone
        read) is no reason to keep it from the group it fits. Which group: Jev's choice
        when the owner has Jev connected and it's sure, else the answer's typed "group",
        else a group its reason names alone."""
        if not self.group or len(self.recipe["groups"]) < 2:
            return ""
        read = "\n".join(e.get("text", "") for e in item.get("evidence") or [])
        shown = " ".join(str(v.get("shown") or "").split())
        shown_read = bool(shown) and fx.in_text(shown, read)
        broke = [r for r in self._named(v.get("broken")) if not self._is_group_rule(r)]
        if any(_fact_of(r) in ("identity", "safety") or shown_read for r in broke):
            return ""
        typed = self._answer_group(v)
        by_words = self._other_group(reason)
        if not (typed and typed != self.group) and not by_words and not any(
                self._is_group_rule(r) for r in self._named(v.get("broken"))):
            return ""  # nothing says it's about its kind
        picked = await self._jev_group(item)
        if picked is None and not typed and not by_words:
            # "its kind is none of the groups" in a long judging answer: asked again, alone
            picked = await self._asked_group(item)
        target = picked if picked is not None else (typed or by_words)
        return target if target in self.recipe["groups"] else ""

    async def _jev_group(self, item: dict) -> str | None:
        """Which of the list's groups an item is, by Jev (a typed choice, calibrated), when
        the owner has Jev connected and it's sure; "" when Jev says none fits; None when
        Jev can't tell (not connected, unsure, failed): the judging answer decides."""
        if self.jev_off:
            return None
        from captain_claw import jev_client
        from captain_claw.flight_deck import jev_routes
        words = self.recipe.get("group_words") or {}
        options = {g: ("; ".join(words.get(g) or []) or g)[:300] for g in self.recipe["groups"]}
        options["none"] = "none of these"
        state = {"list": (self.recipe.get("decide") or "")[:800], "item": self._name_of(item),
                 "what_was_read": "\n\n".join(e.get("text", "")[:1200] for e in (item.get("evidence") or [])[:4])}
        questions = {"group": {"type": "choice", "options": options,
                               "instructions": "Which of these groups is this item, by what it is (its kind, "
                                               "its cuisine, its category), from what was read about it?"}}
        try:
            out = await asyncio.wait_for(jev_routes.ask_for_owner(
                str(self.run.get("owner_id") or ""), state, questions, source="list-funnel"), JEV_TIMEOUT_S)
        except jev_routes.JevNotConnected:
            self.jev_off = True
            return None
        except (TimeoutError, jev_client.JevError) as e:
            log.info("funnel: Jev didn't answer", run_id=self.run["id"], error=type(e).__name__)
            return None
        except Exception as e:  # noqa: BLE001 — the judging answer decides
            log.warning("funnel: Jev call failed", run_id=self.run["id"], error=str(e))
            return None
        ans = ((out or {}).get("answers") or {}).get("group") or {}
        choice, conf = ans.get("choice"), ans.get("confidence")
        if not isinstance(conf, (int, float)) or conf < JEV_SURE:
            return None
        if choice == "none":
            return ""
        return choice if choice in self.recipe["groups"] else None

    async def _asked_group(self, item: dict) -> str | None:
        """Which of the list's groups an item is, asked of the run's model as one short
        question of its own: a judging call that weighs every rule at once reads "one of
        these groups" literally and drops a narrower kind of one. "" when it says none;
        None when it can't be asked or names nothing usable (the judging answer decides)."""
        words = self.recipe.get("group_words") or {}
        groups = "\n".join(f"- {g}" + (f" (such as: {', '.join((words.get(g) or [])[:8])})" if words.get(g) else "")
                           for g in self.recipe["groups"])
        read = "\n\n".join(e.get("text", "")[:1200] for e in (item.get("evidence") or [])[:4])
        try:
            got = await self._ask("group", GROUP_SYSTEM,
                                  f"# The list\n{(self.recipe.get('decide') or '')[:600]}\n\n# Its groups\n{groups}\n\n"
                                  f"# The item\n{self._name_of(item)}\n\n# What was read about it\n{read}")
        except Exception as e:  # noqa: BLE001 — the judging answer decides
            log.warning("funnel group call failed", run_id=self.run["id"], error=str(e))
            return None
        g = " ".join(str((got or {}).get("group") or "").split()) if isinstance(got, dict) else ""
        if not g:
            return None
        if g.lower() == "none":
            return ""
        return next((x for x in self.recipe["groups"] if x.lower() == g.lower()), None)

    def _is_group_rule(self, rule: dict[str, Any]) -> bool:
        """The rule that says which groups an item may be in ("cuisine is one of: ...")."""
        if rule.get("of_group") or (self.recipe["group_by"] and self.recipe["group_by"] in (rule.get("columns") or [])):
            return True
        naming = self._naming()
        return sum(self._names_group(g, rule.get("rule", ""), naming) for g in self.recipe["groups"]) >= 2

    def _other_group(self, reason: str) -> str:
        """Another group of the list a drop's reason says the item fits ("serves Italian
        food, not Croatian"); "" when it names none, or more than one."""
        if not self.group or len(self.recipe["groups"]) < 2 or not reason:
            return ""
        naming = self._naming()
        hits = [g for g in self.recipe["groups"] if g != self.group and self._names_group(g, reason, naming)]
        return hits[0] if len(hits) == 1 else ""

    async def _hand_over(self, item: dict, group: str, reason: str) -> None:
        """An item this step found that fits another group: that group's step judges it
        (or, if that step is done, the finishing step sees it as nearly there)."""
        name = item["fields"].get("name") or next(iter(item["fields"].values()), "") or item.get("key", "")
        _append_csv(self.folder / HANDOVER, _HANDOVER_COLS,
                    {"key": item.get("key", ""), "name": name, "group": group,
                     "fields": json.dumps(item["fields"], ensure_ascii=False), "origin": item.get("origin", ""),
                     "why": reason, "from": self.group})
        _append_csv(self.folder / MAYBE, _MAYBE_COLS,
                    {"key": item.get("key", ""), "name": name, "group": group,
                     "fields": json.dumps(item["fields"], ensure_ascii=False),
                     "why": f"fits {group}: {reason}", "sources": " ".join(e["source"] for e in item.get("evidence") or [])})
        await self.record(item, "handed", f"fits {group} instead, handed to its step ({reason})", group=group,
                          stage="handover")

    async def _take_handover(self) -> None:
        """Items other groups' steps found for this group, judged here first."""
        taken = {r.get("key") for r in _read_csv(self.folder / ITEMS)
                 if r.get("group") == self.group and r.get("stage") not in ("handover", "")}
        todo = []
        for h in _read_csv(self.folder / HANDOVER):
            if h.get("group") != self.group or h.get("key") in taken:
                continue
            try:
                fields = json.loads(h.get("fields") or "{}")
            except ValueError:
                continue
            taken.add(h.get("key"))
            todo.append({"fields": fields, "origin": h.get("origin") or "handed over", "text": "", "key": h.get("key", "")})
        for i in range(0, len(todo), CONCURRENCY):
            if self.full() or self.stopped:
                break
            self.counts["looked"] += len(todo[i:i + CONCURRENCY])
            await self._gather_wave(todo[i:i + CONCURRENCY])

    def _numbered_rules(self) -> str:
        """The rules, numbered, for a judging call to name in "missing", "broken" and
        "loosened" by number: code then knows which rule it meant in any language or
        wording ("rating 4+ on Google" for "ocjena 4 ili viša na Googleu…")."""
        rules = self.recipe.get("rules") or []
        if not rules:
            return ""
        lines = ["# Rules by number (name them by these numbers in missing, broken and loosened)"]
        lines += [f"{n}. {r['rule']}" + (f" — if its fact isn't found: {r['if_not_shown']}" if r.get("if_not_shown") else "")
                  for n, r in enumerate(rules, start=1)]
        return "\n".join(lines) + "\n\n"

    def _tiers_text(self) -> str:
        """The rule tiers for a judging call, when rows may be on a looser reading."""
        if not self.loosen:
            return ""
        rules = self.recipe.get("rules") or []
        soft = loosenable(self.recipe)
        num = {id(r): n for n, r in enumerate(rules, start=1)}
        fixed = [f"{num[id(r)]}. {r['rule']}" for r in rules if r not in soft]
        lines = ["# Rule tiers"]
        if fixed:
            lines += ["Never loosened (what the item is, where it is, how to reach it): an item that breaks one is "
                      "dropped; one that doesn't show it is maybe:"] + [f"- {r}" for r in fixed]
        lines += ["May be met by a looser reading:"]
        lines += [f"- {num[id(r)]}. {r['rule']} — looser: {looser_of(r)}" for r in soft]
        lines += ['Verdict "loose": the item meets every never-loosened rule, and each other rule it doesn\'t meet as '
                  'written it meets by its looser reading. Unshown is not broken: a fact what was read simply doesn\'t '
                  'show, with nothing read saying the item breaks the rule, is met by "not shown". An item shown to '
                  'break a rule is still dropped. Name the rules met loosely in "loosened", by their numbers; a column '
                  'the looser reading gives no value for stays empty. "keep" only when every rule is met as written.']
        return "\n".join(lines) + "\n\n"

    def _named(self, named: Any) -> list[dict[str, Any]]:
        """The rules a judging answer names: by number (1, "3", "R2", "rule 4"), else by
        words that are part of the rule or hold it."""
        rules = self.recipe.get("rules") or []
        if not isinstance(named, list):
            named = [named] if isinstance(named, (int, str)) and str(named).strip() else []
        out: list[dict[str, Any]] = []
        for x in named:
            m = re.fullmatch(r"\s*(?:r|rule|pravilo|#)?\s*\.?\s*(\d{1,3})\s*\.?\s*", str(x), re.I) \
                if not isinstance(x, bool) else None
            if m and 1 <= int(m.group(1)) <= len(rules):
                hit = [rules[int(m.group(1)) - 1]]
            else:
                said = " ".join(str(x).split()).lower()
                hit = [r for r in rules if said and (said in r["rule"].lower() or r["rule"].lower() in said)]
            out += [r for r in hit if r not in out]
        return out

    def _unbacked_drop(self, v: dict, item: dict) -> bool:
        """A drop that shows no fact: every rule it says is broken may go unshown, and the
        words it quotes as proof (if any) aren't in what was read. An item shown to break
        a rule (a stated 120 people, a dated post, a foreign address) stays dropped, as
        does a drop that names no rule or one that never loosens. When nothing is read
        loosely, any rule but what the item is (its kind, an exclusion) and what must
        never be done may go unshown: "no rating, no contact found" is nearly there."""
        broken = self._named(v.get("broken"))
        if self.loosen:
            soft = loosenable(self.recipe)
        else:
            soft = [r for r in self.recipe.get("rules") or [] if _fact_of(r) not in ("identity", "safety")]
        if not broken or any(r not in soft for r in broken):
            return False
        shown = " ".join(str(v.get("shown") or "").split())
        if not shown:
            return True
        read = "\n".join(e.get("text", "") for e in item.get("evidence") or [])
        return not fx.in_text(shown, read)

    async def _alternative_doubt(self, v: dict, item: dict, met: list[dict[str, Any]], reason: str) -> str | None:
        """Why an item that lacks only facts the charter gives an alternative for still
        isn't kept on that alternative, or "" when it is. Its typed group must be one of
        the list's (one the judging call says is "none" fits no group, whatever it lacks);
        and when the owner has Jev connected, Jev is asked, from what was read, whether
        the item meets each alternative ("judged good from what was read"): a confident
        no keeps it nearly there. No Jev, or unsure: kept, as the judging call's maybe
        said it seemed to fit."""
        if self.recipe["groups"] and " ".join(str(v.get("group") or "").split()).lower() == "none":
            return "its kind fits none of the list's groups, so the charter's alternative doesn't make it one"
        met = [r for r in met if not r.get("reach")]  # a contact page is found by code, not judged
        if self.jev_off or not met:
            return ""
        from captain_claw import jev_client
        from captain_claw.flight_deck import jev_routes
        state = {"list": (self.recipe.get("decide") or "")[:800], "item": self._name_of(item),
                 "judged": reason[:400],
                 "what_was_read": "\n\n".join(e.get("text", "")[:1200] for e in (item.get("evidence") or [])[:4])}
        questions = {f"r{n}": {"type": "noul", "instructions": f"From what was read about this item, does it meet "
                               f"this, as the list's owner asks: {r['if_not_shown']} (for the rule: {r['rule']})?",
                               "yes_means": "it meets it, on what was read",
                               "no_means": "what was read shows it doesn't, or gives nothing to judge it by"}
                     for n, r in enumerate(met, start=1)}
        try:
            out = await asyncio.wait_for(jev_routes.ask_for_owner(
                str(self.run.get("owner_id") or ""), state, questions, source="list-funnel"), JEV_TIMEOUT_S)
        except jev_routes.JevNotConnected:
            self.jev_off = True
            return ""
        except (TimeoutError, jev_client.JevError) as e:
            log.info("funnel: Jev didn't answer", run_id=self.run["id"], error=type(e).__name__)
            return ""
        except Exception as e:  # noqa: BLE001 — the judging answer decides
            log.warning("funnel: Jev call failed", run_id=self.run["id"], error=str(e))
            return ""
        answers = (out or {}).get("answers") or {}
        no = [r for n, r in enumerate(met, start=1)
              if isinstance((answers.get(f"r{n}") or {}).get("noul"), (int, float))
              and answers[f"r{n}"]["noul"] < JEV_NO]
        return ("Jev: not met by the charter's own alternative: " + "; ".join(
            f"{r['if_not_shown']} ({r['rule']})" for r in no)) if no else ""

    def _alternatives(self, v: dict) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """The rules a maybe names as missing, split: those the charter itself gives an
        alternative for when their fact isn't found (``if_not_shown``), and the rest."""
        missing = self._named(v.get("missing"))
        met = [r for r in missing if r.get("if_not_shown")]
        rest = [r for r in missing if not r.get("if_not_shown")]
        return list({id(r): r for r in met}.values()), list({id(r): r for r in rest}.values())

    def _maybe_or_loose(self, v: dict) -> tuple[str, list[dict[str, Any]]]:
        """A maybe whose missing facts are all ones a looser reading lets go unshown is
        loose: unshown is not broken. Code decides from the rules it names, so a judging
        call that keeps to "maybe" for every gap still lets such an item through. It
        stays maybe when it names a rule that never loosens, names none, or gave no row."""
        missing = self._named(v.get("missing")) + self._named(v.get("loosened"))
        soft = loosenable(self.recipe)
        row = v.get("row") if isinstance(v.get("row"), dict) else {}
        if missing and all(r in soft for r in missing) and any(str(x).strip() for x in row.values()):
            once: list[dict[str, Any]] = []
            for r in missing:
                if r not in once:
                    once.append(r)
            return "loose", once
        return "maybe", []

    def _loosened(self, named: Any) -> list[dict[str, Any]]:
        """The rules a loose verdict names that may be met loosely (the prefer rules
        when it names none of the rules at all; none when it names only rules that
        never loosen)."""
        soft = loosenable(self.recipe)
        said = [" ".join(str(x).split()).lower() for x in named or [] if str(x).strip()] \
            if isinstance(named, list) else []
        named_rules = [r for r in self.recipe.get("rules") or []
                       if any(x and (x in r["rule"].lower() or r["rule"].lower() in x) for x in said)]
        if not named_rules:
            return soft_rules(self.recipe) or soft
        return [r for r in named_rules if r in soft]

    async def record(self, item: dict, status: str, reason: str, group: str = "", stage: str = "") -> None:
        if status != "handed":
            self.counts[status if status in ("kept", "maybe", "loose") else "dropped"] += 1
        g = group or self.group
        self.looked_in[g] = self.looked_in.get(g, 0) + 1
        if status == "dropped":
            short = re.sub(r"\(.*?\)|'[^']*'|\d[\d,.-]*", "", reason).strip()[:60]
            self.reasons[short] = self.reasons.get(short, 0) + 1
        name = item["fields"].get("name") or next(iter(item["fields"].values()), "") or item.get("key", "")
        _append_csv(self.folder / ITEMS, _ITEM_COLS,
                    {"key": item.get("key", ""), "name": str(name)[:120], "group": group or self.group,
                     "status": status, "stage": stage or (status if status in ("kept", "maybe") else "judge"),
                     "reason": reason, "source": item.get("origin", ""), "step": self.key,
                     "at": time.strftime("%Y-%m-%d %H:%M"),
                     "looked": str(item["look_read"]) if item.get("looked") and "look_read" in item else ""})
        line = f"{status} {str(name)[:60]}" + (f" ({group})" if group and not self.group else "") + f": {reason}"
        self.recent = (self.recent + [line])[-8:]
        await self._event("decide", line)
        if (self.counts["kept"] + self.counts["maybe"] + self.counts["loose"] + self.counts["dropped"]) % NOTE_EVERY == 0:
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
        mine = set(src.get("queries", [])) | set(worked) | set(self.queries.get("written", {}).get(self.group, []))
        used = [q for q in self.queries["used"] if q in mine]
        out: list[str] = []
        for _try in range(2):  # an empty or unreadable answer is asked for once more, not taken as "no more"
            try:
                got = await self._new_queries_call(src, worked, used)
            except Exception as e:  # noqa: BLE001 — no new searches this round
                log.warning("funnel queries call failed", run_id=self.run["id"], error=str(e))
                continue
            fresh = got.get("queries") if isinstance(got, dict) else got if isinstance(got, list) else []
            out = list(dict.fromkeys(str(q).strip()[:300] for q in fresh or []
                                     if str(q).strip() and str(q).strip() not in self.queries["used"]))
            if out:
                break
        self.queries.setdefault("written", {}).setdefault(self.group, []).extend(out[:15])
        if out:
            await self._event("funnel", f"{len(out)} new searches for {self.group or 'the list'}")
        return out[:15]

    async def _new_queries_call(self, src: dict, worked: dict[str, int], used: list[str]) -> Any:
        return await self._ask("queries", "Write new web searches for a list that is still short. Reply ONLY with "
                                         'JSON: {"queries": ["...", ...]} (10-15 new ones, none already used). '
                                         "Keep your thinking short.",
                              f"# Looking for\n{self.recipe['item']}s" + (f" ({self.group})" if self.group else "")
                              + (f": {src.get('about')}" if src.get("about") else "")
                              + f"\n\n# Rules\n{self.recipe['decide'][:1500]}\n\n"
                              + (f"# What the owner's documents and links say\n{self.brief[:1500]}\n\n"
                                 if self.brief else "")
                              + "# Searches that found kept items\n"
                              + ("\n".join(f"- {q} ({n})" for q, n in sorted(worked.items(), key=lambda x: -x[1])[:15])
                                 or "(none yet)")
                              + "\n\n# Already used (write different ones: other words, other kinds of page, "
                              "other places within the list's own)\n" + "\n".join(f"- {q}" for q in used[-60:]))

    def note(self, final: bool = False) -> None:
        """The step's progress note, as Bat's workers keep one."""
        from captain_claw.flight_deck import bat_routes
        lines = [f"# {self.step.get('title') or self.key}", "",
                 f"Status: {'finished' if final else 'working'}" + (f" (stopped: {self.stopped})" if self.stopped else "")]
        if self.recipe["groups"]:
            lines += ["", "Kept so far:"] + [f"- {g}: {self.kept.get(g, 0)}" + (f" of {n}" if n else "")
                                             + (f" (and {self.loose[g]} on a looser reading)" if self.loose.get(g) else "")
                                             for g, n in self.recipe["groups"].items()]
        else:
            total = sum(self.kept.values())
            target = int(self.recipe.get("target") or 0)
            loose = sum(self.loose.values())
            lines += ["", f"Kept so far: {total}" + (f" of {target}" if target else "")
                      + (f" (and {loose} on a looser reading)" if loose else "")]
        c = self.counts
        lines += ["", f"This step looked at {c['looked']} items: {c['kept']} kept, "
                      + (f"{c['loose']} on a looser reading, " if c["loose"] else "") + f"{c['maybe']} maybe, "
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
            loose = self.loose.get(self.group, 0)
            head = f"{self.group}: {have}" + (f" of {q}" if q else "") + " kept" \
                + (f", {loose} more on a looser reading." if loose else ".")
            short = bool(q) and have + loose < q
        else:
            have = sum(self.kept.values())
            loose = sum(self.loose.values())
            target = int(self.recipe.get("target") or 0)
            head = f"{have} kept" + (f" of {target}" if target else "") \
                + (f", {loose} more on a looser reading." if loose else ".")
            if self.recipe["groups"]:
                short = any(self.kept.get(g, 0) + self.loose.get(g, 0) < n for g, n in self.recipe["groups"].items() if n)
            else:
                short = bool(target) and have + loose < target
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

def answers_by_id(got: Any, asked: int) -> dict[int, dict]:
    """A judging call's answers, by item id, however the model wrapped them: the array
    asked for, an object holding it ({"items": [...]}), or, for one item, the bare
    object itself. One item's answer is that item's whatever id it carries: a small
    model asked about one thing often answers with one object, and asking again in the
    same words got the same answer and the item dropped as unanswered."""
    if isinstance(got, dict):
        inner = next((v for v in got.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
                      and any("verdict" in x for x in v)), None)
        got = inner if inner is not None else [got] if "verdict" in got else []
    if not isinstance(got, list):
        return {}
    said = [v for v in got if isinstance(v, dict)]
    out = {int(v["id"]): v for v in said if str(v.get("id", "")).strip().isdigit() and 1 <= int(v["id"]) <= asked}
    if asked == 1 and not out:
        one = next((v for v in said if "verdict" in v), None)
        if one is not None:
            out = {1: one}
    return out


def check_row(recipe: dict[str, Any], row: dict[str, str], texts: dict[str, str], *,
              unsourced_ok: bool = False, relax: set[str] | frozenset[str] = frozenset()) -> str:
    """Why a row fails the recipe's checks by code, or "". ``texts`` is what was read,
    by link. A value that must be in its source is looked for on the page its source
    column names (any page read, when the check names no source column; skipped then
    with ``unsourced_ok``, for rows whose pages Captain didn't read itself). ``relax``:
    columns of rules met only by their looser reading, which may stay empty and keep
    no date window; a value given there must still be in its source."""
    every = "\n".join(texts.values())
    for c in recipe["checks"]:
        col, op = c["column"], c["op"]
        v = row.get(col, "")
        if col in relax and op in ("required", "within_months"):
            continue
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
        if op == "in_evidence" and re.match(r"https?://", v) and any(
                s.rstrip("/") == v.rstrip("/") for s in texts):
            continue  # a link to a page that was read: it's where the value came from
        if op == "in_evidence":
            src = row.get(c.get("source_column", ""), "") if c.get("source_column") else ""
            if src and not _LINK_LIKE.match(src.strip()):
                src = ""  # a source named, not linked ("Google"): it names no page, so anything read counts
            if not src and unsourced_ok:
                continue
            where = texts.get(src) if src and src in texts else None
            if where is None and src:
                where = next((t for s, t in texts.items() if s.rstrip("/") == src.rstrip("/")), None)
            site = _site_pages(src, texts)
            if site is not None and (where is None or not fx.in_text(v, where)):
                # the item's own site: the value may be on any page read there (the phone on
                # its contact page while the row cites its homepage, or "www." not read as such)
                where = site
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
            span = number_span(v)
            try:
                lim = float(c.get("value"))
            except (TypeError, ValueError):
                continue
            if span and ((op == "min" and span[1] < lim) or (op == "max" and span[0] > lim)):
                return f"{col} {v} is {'under' if op == 'min' else 'over'} {c.get('value')}"
    return ""


def number_span(text: str) -> tuple[float, float] | None:
    """A number or a range in free text, as (low, high): "12" → (12, 12), "11-50" and
    "11 to 50 employees" → (11, 50), "1,001+" → (1001, inf), "$2.5M" → (2.5, 2.5).
    None when it has no number."""
    t = str(text or "").replace(",", "")
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
    if not nums:
        return None
    if re.search(r"\d\s*\+", t) and len(nums) == 1:
        return nums[0], float("inf")
    if len(nums) >= 2 and re.search(r"\d\s*(?:-|–|to)\s*\d", t):
        return min(nums[:2]), max(nums[:2])
    return nums[0], nums[0]


# A column for an address, in a few languages; else a place rule's first column.
_ADDRESS = re.compile(r"addr|adres|anschrift|indirizzo|direcci|endere|street|ulica|straße|strasse", re.I)


def place_line(text: str, rule: str, targets: list[str]) -> str:
    """An address on an item's own pages for a place rule: the shortest line that
    names the rule's place (or the list's) and holds a number, a house number or a
    postcode ("Ilica 12, 10000 Zagreb"); "" when none does."""
    named = fx.places_named(rule)
    stems = {fx.place_stem(w) for _code, w in named}  # the rule's own place ("u gradu Zagrebu"), in any case
    codes = {code for code, _w in named} or set(targets)
    if not codes:
        return ""
    best = ""
    for raw in re.split(r"[\n\r|•·]+|\s{3,}", text or ""):
        line = " ".join(raw.split()).strip(" ,;-#")
        # "Adresa: Ilica 12, …" (a list item or table cell keeps its label): the
        # address without it is shorter, so it wins when it names the place too
        for said in (line, _LABEL_LEAD.sub("", line, count=1)):
            if not (8 <= len(said) <= 120) or not re.search(r"\d", said):
                continue
            here = fx.places_named(said)
            if any(code in codes and (not stems or fx.place_stem(w) in stems or w.lower() in stems) for code, w in here):
                if not best or len(said) < len(best):
                    best = said
    return best


_LABEL_LEAD = re.compile(r"^[^\d:()]{1,25}?\s*:\s*")  # "Adresa : ", "Address: ", "Find us: "


def _site_pages(src: str, texts: dict[str, str]) -> str | None:
    """Everything read on the site a source link is on, when that is one site's own
    pages (not a directory or social site, where pages are other people's); None when
    nothing was read there."""
    d = fx.domain_of(src) if re.match(r"https?://|www\.", str(src or ""), re.I) else ""
    if not d or any(fx.same_site(d, s) for s in _SKIP_SITES):
        return None
    pages = [t for s, t in texts.items() if fx.same_site(fx.domain_of(s), d)]
    return "\n".join(pages) if pages else None


_LINK_LIKE = re.compile(r"(https?://|www\.)\S+$|[\w-]+(\.[\w-]+)+(/\S*)?$", re.I)
_PHONE_COL = re.compile(r"phone|telef|^tel\b|mobil|handy|cell|m[oó]vil|celular|gsm", re.I)
_LANG_PART = re.compile(r"^[a-z]{2}(?:[-_][a-z]{2})?$", re.I)
_HOME_PARTS = {"index.html", "index.htm", "index.php", "home", "default.aspx"}
_PROFILE_PARTS = {"in", "company", "companies", "pub", "pages", "people", "profile", "user", "users", "channel",
                  "c", "u", "r", "groups", "biz", "school", "org"}


def _site_page(url: str) -> str:
    """The page a site link points to on its host, a language part and a home page
    aside ("" for its front page): two links to one front page are one site."""
    try:
        path = urlsplit(url if "://" in url else "http://" + url).path
    except ValueError:
        return ""
    parts = [x.lower() for x in path.split("/") if x and x.lower() not in _HOME_PARTS]
    while parts and _LANG_PART.match(parts[0]):
        parts.pop(0)
    return "/".join(parts)


def site_id(url: str, shared: tuple[str, ...] = _SKIP_SITES) -> str:
    """Who an item's own site says it is: ``example.com`` for its home page (a language
    part such as ``/en`` aside), host and path for a page deeper in; on a site many
    share (a social or directory site) the item's page there (``instagram.com/restac``),
    and "" for such a site's own pages or anything that isn't a link."""
    u = str(url or "").strip()
    if not u or " " in u or "." not in u:
        return ""
    d = fx.domain_of(u)
    if not d or "." not in d:
        return ""
    try:
        path = urlsplit(u if "://" in u else "http://" + u).path
    except ValueError:
        return ""
    parts = [x.lower() for x in path.split("/") if x and x.lower() not in _HOME_PARTS]
    if any(fx.same_site(d, x) for x in shared):
        parts = parts[:2] if parts and parts[0] in _PROFILE_PARTS else parts[:1]
        if not parts or parts[-1] in _PROFILE_PARTS:
            return ""
        return f"{d}/{'/'.join(parts)}"
    while parts and _LANG_PART.match(parts[0]):
        parts.pop(0)
    return d + ("/" + "/".join(parts) if parts else "")


def _plain_words(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", str(text or "").lower())
    return re.findall(r"[a-z0-9]+", "".join(ch for ch in t if not unicodedata.combining(ch)))


def _address_parts(text: str) -> tuple[frozenset[str], frozenset[str]]:
    """An address's house numbers and street words (each word's first four letters, so
    "Lipova" and "Lipove" agree), without postal codes and the names of places (the
    city, the country)."""
    places = {w for _code, name in fx.places_named(text) for w in _plain_words(name)}
    words = [w for w in _plain_words(text) if w not in places]
    numbers = frozenset(w for w in words if w.isdigit() and len(w) <= 4)
    stems = frozenset(w[:4] for w in words if w.isalpha() and len(w) >= 4)
    return numbers, stems


class SameItems:
    """The rows on a list, to tell an item found again under another name ("Restac" and
    "Restac 900"). The same item: the same own site (its home page, or the same page on
    it); the same host with the same phone or email; or a word of its name shared with
    the same phone or email, or with the same address. Never when the two are apart: a
    phone or an address that differs (another house number, another street), so two
    branches of one chain stay two. An address written two ways ("Ilička ulica" and
    "Ilička", "Lipova 10" and "Ul. Lipove 10") isn't apart."""

    def __init__(self, recipe: dict[str, Any], skip: tuple[str, ...] = ()) -> None:
        self.site = recipe.get("site_column") or ""
        self.key = recipe["key"]
        cols = recipe["columns"]
        self.phones = [c for c in cols if _PHONE_COL.search(c)]
        self.addresses = [c for c in cols if _ADDRESS.search(c) and c not in self.phones]
        self.shared = tuple(dict.fromkeys(_SKIP_SITES + tuple(skip)))
        self.rows: list[dict[str, Any]] = []

    def _seen(self, row: dict[str, str]) -> dict[str, Any]:
        url = str(row.get(self.site) or "") if self.site else ""
        sid = site_id(url, self.shared)
        host = "" if not sid or any(fx.same_site(fx.domain_of(url), x) for x in self.shared) else fx.domain_of(url)
        phones = {re.sub(r"\D", "", str(row.get(c) or ""))[-8:] for c in self.phones
                  if len(re.sub(r"\D", "", str(row.get(c) or ""))) >= 7}
        mails = {str(v).strip().lower() for v in row.values()
                 if re.fullmatch(r"\s*[^@\s]+@[^@\s]+\.\w+\s*", str(v or ""))}
        address = " ".join(str(row.get(c) or "") for c in self.addresses).strip()
        numbers, stems = _address_parts(address) if address else (frozenset(), frozenset())
        name = str(row.get(self.key) or "") or next((str(v) for v in row.values() if v), sid)
        return {"sid": sid, "host": host, "phones": phones, "contacts": phones | mails, "name": name,
                "words": {w for w in _plain_words(name) if len(w) >= 3}, "numbers": numbers, "stems": stems}

    @staticmethod
    def _apart(a: dict[str, Any], b: dict[str, Any]) -> bool:
        if a["phones"] and b["phones"] and a["phones"].isdisjoint(b["phones"]):
            return True
        if a["numbers"] and b["numbers"] and a["numbers"].isdisjoint(b["numbers"]):
            return True
        return bool(a["stems"] and b["stems"] and a["stems"].isdisjoint(b["stems"]))

    def of(self, row: dict[str, str]) -> str:
        """The name of the row on the list this one repeats, or ""."""
        me = self._seen(row)
        for other in self.rows:
            if self._apart(me, other):
                continue  # another place of the same name, site or chain
            same_contact = bool(me["contacts"] & other["contacts"])
            if me["sid"] and me["sid"] == other["sid"]:
                return other["name"]
            if me["host"] and me["host"] == other["host"] and same_contact:
                return other["name"]
            if me["words"] & other["words"] and (same_contact or (
                    me["numbers"] and me["numbers"] == other["numbers"] and me["stems"] == other["stems"])):
                return other["name"]
        return ""

    def add(self, row: dict[str, str]) -> None:
        self.rows.append(self._seen(row))


def kept_per_group(folder: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in _read_csv(folder / ROWS):
        g = r.get(_GROUP_COL, "")
        out[g] = out.get(g, 0) + 1
    return out


def loose_per_group(folder: Path) -> dict[str, int]:
    """Rows on a looser reading not yet on the list, per group."""
    on_list = {r.get("key", "") for r in _read_csv(folder / ITEMS) if r.get("status") == "kept"}
    out: dict[str, int] = {}
    for r in _read_csv(folder / ITEMS):
        if r.get("status") == "loose" and r.get("key") not in on_list:
            out[r.get("group", "")] = out.get(r.get("group", ""), 0) + 1
    return out


def fill_from_loose(folder: Path, recipe: dict[str, Any]) -> list[dict[str, str]]:
    """Rows on a looser reading where a group is still short after every strict row
    (the funnel's and the finishing step's): fewest rules loosened first, then in the
    order found. They join the list's rows, and their items count as kept. Returns
    what was added: name, group and the rules loosened."""
    short = shortfall(folder, recipe)
    loose = _read_csv(folder / LOOSE)
    if not short or not loose:
        return []
    key = recipe["key"]
    listed = _read_csv(folder / ROWS)
    on_list = {norm_key(r.get(key, "")) for r in listed} if key else set()
    same = SameItems(recipe, tuple(recipe.get("skip_sites") or ()))
    for r in listed:
        same.add(r)
    ranked = sorted(enumerate(loose), key=lambda x: (len([n for n in x[1].get(_LOOSE_COL, "").split(";") if n.strip()]),
                                                     x[0]))
    added: list[dict[str, str]] = []
    for _i, r in ranked:
        group = r.get(_GROUP_COL, "")
        slot = group if recipe["groups"] else ""
        if short.get(slot, 0) <= 0:
            continue
        k = norm_key(r.get(key, "")) if key else ""
        if (k and k in on_list) or same.of(r):
            continue
        on_list.add(k)
        same.add(r)
        short[slot] -= 1
        _append_csv(folder / ROWS, recipe["columns"] + [_GROUP_COL], {**r, _GROUP_COL: group})
        name = next((r.get(c, "") for c in recipe["columns"] if r.get(c)), k)
        added.append({"key": k, "name": name, "group": group, "loosened": r.get(_LOOSE_COL, "")})
    if added:
        used = {a["key"] for a in added if a["key"]}
        names = {a["name"].lower() for a in added}
        items = _read_csv(folder / ITEMS)
        for it in items:
            if it.get("status") == "loose" and (norm_key(it.get("key", "")) in used or it.get("name", "").lower() in names):
                it["status"] = "kept"
                it["reason"] = "on the list on a looser reading: " + it.get("reason", "").removeprefix(
                    "meets only the looser reading of: ")
        with (folder / ITEMS).open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_ITEM_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(items)
    return added


def loose_on_list(folder: Path) -> list[dict[str, str]]:
    """The list's rows that meet the soft rules only by their looser reading."""
    return [r for r in _read_csv(folder / ITEMS) if r.get("status") == "kept" and r.get("stage") == "loose"]


def shortfall(folder: Path, recipe: dict[str, Any]) -> dict[str, int]:
    """How many more each group needs (``""``: the whole list, when it has no groups)."""
    kept = kept_per_group(folder)
    if recipe["groups"]:
        return {g: n - kept.get(g, 0) for g, n in recipe["groups"].items() if n and kept.get(g, 0) < n}
    target = int(recipe.get("target") or 0)
    total = sum(kept.values())
    return {"": target - total} if target and total < target else {}


def counts(folder: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    """How items went, per group and stage, for the run panel and the summary.
    ``on_list``: rows on the list now; ``loose_waiting``: rows on a looser reading
    that will fill what's short at the merge (a group full of them is full)."""
    groups: dict[str, dict[str, int]] = {}
    for r in _read_csv(folder / ITEMS):
        g = groups.setdefault(r.get("group", ""), {"looked": 0, "loose_waiting": 0, **{st: 0 for st in STAGES}})
        g["looked"] += 1
        if r.get("status") == "loose":
            g["loose_waiting"] += 1
        stage = r.get("stage") or ("kept" if r.get("status") == "kept" else "maybe" if r.get("status") == "maybe"
                                   else "judge")
        if stage in g:
            g[stage] += 1
    kept = kept_per_group(folder)
    for name, quota in (recipe["groups"] or {}).items():
        groups.setdefault(name, {"looked": 0, "loose_waiting": 0, **{st: 0 for st in STAGES}})
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

def look_on(run: dict) -> bool:
    """Whether a nearly-there item is looked into as it comes up (``config.funnel_look``:
    on by default, ``off`` for the old way: a finishing agent step at the end)."""
    return str((run.get("config") or {}).get("funnel_look") or "on").lower() != "off"


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
    if look_on(run):
        return done("Nothing to finish: each item that came close was looked into as it came up.")
    if not leftovers_on(run):
        return done("Nothing to finish: the run's settings leave a short group as it is.")
    short = shortfall(folder, recipe)
    if not short:
        return done("Nothing to finish: no group is short.")
    if loosen_on(run, recipe):
        loose = loose_per_group(folder)
        short = {g: n - loose.get(g, 0) if g else n - sum(loose.values()) for g, n in short.items()}
        short = {g: n for g, n in short.items() if n > 0}
        if not short:
            return done("Nothing to finish: rows on a looser reading of the soft rules fill what's short.")
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
    loose_ok = loosen_on(run, recipe)
    if not finish.is_file():
        with finish.open("w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(recipe["columns"] + ([_LOOSE_COL] if loose_ok else []))
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
    if loose_ok:
        lines.append(f"4. a row that meets a rule only by its looser reading (the Rules in shared/conventions.md) "
                     f"is fine: name those rules in its {_LOOSE_COL} column, separated by \";\", and leave a column "
                     "empty when the looser reading gives no value. Don't spend long proving a fact the item "
                     "simply doesn't publish.")
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
    listed = _read_csv(folder / ROWS)
    on_list = {norm_key(r.get(recipe["key"], "")) for r in listed} if recipe["key"] else set()
    same = SameItems(recipe, tuple(recipe.get("skip_sites") or ()))
    for r in listed + _read_csv(folder / LOOSE):
        same.add(r)
    sources = sorted({c["source_column"] for c in recipe["checks"] if c["op"] == "in_evidence" and c.get("source_column")})
    unsourced_check = any(c["op"] == "in_evidence" and not c.get("source_column") for c in recipe["checks"])
    soft = loosenable(recipe) if loosen_on(run, recipe) else []
    ref_text = reference_text(folder, recipe)
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
                    pages[url] = text[:FETCH_CHARS] if ok else ""
                if pages[url]:
                    texts[url] = pages[url]
            said = [x.strip().lower() for x in str(raw.get(_LOOSE_COL) or "").split(";") if x.strip()]
            loosened = [r for r in soft if any(x in r["rule"].lower() or r["rule"].lower() in x for x in said)]
            relax = {c for r in loosened for c in r.get("columns") or []}
            problem = check_row(recipe, row, texts, unsourced_ok=True, relax=relax)
            empty = [c for c in judgement_columns(recipe, ref_text) if not row.get(c)]
            if not problem and empty:
                problem = f"{', '.join(empty)} left empty"
            unshown = unshown_musts(recipe, row)
            if not problem and unshown:
                problem = "it shows nothing for: " + "; ".join(unshown)
            twin = same.of(row)
            if not problem and twin:
                problem = f"it is {twin} under another name"
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
            on_list.add(key)
            same.add(row)
            if loosened:  # waits with the funnel's loose rows: on the list only where a group is still short
                names = "; ".join(r["rule"] for r in loosened)
                _append_csv(folder / LOOSE, recipe["columns"] + [_GROUP_COL, _LOOSE_COL],
                            {**row, _GROUP_COL: group, _LOOSE_COL: names})
                _append_csv(folder / ITEMS, _ITEM_COLS, {**item, "status": "loose", "stage": "agent",
                                                         "reason": f"added by the finishing step on a looser reading "
                                                                   f"of: {names}"})
                continue
            kept[group] = kept.get(group, 0) + 1
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
    same = SameItems(recipe, tuple(recipe.get("skip_sites") or ()))
    for r in _read_csv(folder / ROWS):
        k = norm_key(r.get(recipe["key"], "")) if recipe["key"] in cols else ""
        if (k and k in seen) or same.of(r):
            continue
        if k:
            seen.add(k)
        same.add(r)
        out.append({c: r.get(c, "") for c in cols})
    target = folder / recipe["output"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    return out


def came_close_name(recipe: dict[str, Any]) -> str:
    """Where the items that came close go: next to the list, named after it."""
    out = Path(recipe["output"])
    return str(out.with_name(f"{out.stem}{CLOSE_SUFFIX}.csv"))


_CLOSE_WHY, _CLOSE_SRC = "Why it came close", "Sources"


def _cells_from_fields(fields: dict[str, Any], cols: list[str]) -> dict[str, str]:
    """An item's own fields (a source file's columns, a search hit's name and link)
    put under the list's columns of the same name."""
    low = {str(k).strip().lower(): v for k, v in (fields or {}).items()}
    return {c: " ".join(str(low[c.lower()]).split()) for c in cols if str(low.get(c.lower()) or "").strip()}


def write_came_close(folder: Path, recipe: dict[str, Any]) -> list[dict[str, str]]:
    """``<the list>-came-close.csv``: the items that nearly made the list, in the list's
    columns as far as they were found, with why and where they were read. Rows that
    meet the soft rules only by their looser reading and weren't needed come first
    (they were filled in full), then the items a fact the rules need wasn't found
    for. Nothing already on the list, and nothing later dropped. With nothing that
    came close there is no file (one an earlier merge wrote goes)."""
    cols = list(recipe["columns"])
    key, group_by = recipe["key"], recipe["group_by"]
    on_list = {norm_key(r.get(key, "")) for r in _read_csv(folder / ROWS)} if key else set()
    last = {}
    for r in _read_csv(folder / ITEMS):
        if r.get("stage") != "duplicate":  # found again under another name: what it was stands
            last[r.get("key", "")] = r.get("status", "")
    gone = {k for k, st in last.items() if k and st in ("dropped", "kept")}
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    same = SameItems(recipe, tuple(recipe.get("skip_sites") or ()))
    for r in _read_csv(folder / ROWS):
        same.add(r)

    def add(k: str, row: dict[str, str], group: str, why: str, sources: str) -> None:
        nk = norm_key(row.get(key, "")) if key and row.get(key) else ""
        if any(x and (x in gone or x in seen) for x in (k, nk)) or (nk and nk in on_list) or same.of(row):
            return
        seen.update(x for x in (k, nk) if x)
        same.add(row)
        if group_by and (group in recipe["groups"] or (group and not row.get(group_by))):
            row[group_by] = group_cell(recipe, group, row.get(group_by, ""))  # as the list names it, never a word a page used ("Sushi")
        out.append({**{c: row.get(c, "") for c in cols}, _GROUP_COL: group, _CLOSE_WHY: why, _CLOSE_SRC: sources})

    for r in _read_csv(folder / LOOSE):
        names = "; ".join(n.strip() for n in (r.get(_LOOSE_COL) or "").split(";") if n.strip())
        add("", {c: r.get(c, "") for c in cols}, r.get(_GROUP_COL, ""),
            f"meets these rules only by their looser reading: {names}" if names else "meets a rule only by its looser reading",
            "")
    maybe = {}
    for m in _read_csv(folder / MAYBE):
        maybe[m.get("key") or m.get("name") or str(len(maybe))] = m  # the latest word on each item
    for k, m in maybe.items():
        try:
            fields = json.loads(m.get("fields") or "{}")
        except ValueError:
            fields = {}
        try:
            filled = json.loads(m.get("row") or "{}")
        except ValueError:
            filled = {}
        row = {**_cells_from_fields(fields if isinstance(fields, dict) else {}, cols),
               **{c: str(v) for c, v in (filled if isinstance(filled, dict) else {}).items() if c in cols and v}}
        if key and key in cols and not row.get(key):
            row[key] = str(m.get("name") or "")
        add(m.get("key", ""), row, m.get("group", ""), m.get("why") or "a fact the rules need wasn't found",
            " ".join((m.get("sources") or "").split()[:5]))
    order = {g: i for i, g in enumerate(recipe["groups"])}
    out.sort(key=lambda r: order.get(r[_GROUP_COL], len(order)))  # stable: looser-reading rows first in each group
    header = cols + ([] if not recipe["groups"] or group_by in cols else ["Group"]) + [_CLOSE_WHY, _CLOSE_SRC]
    target = folder / came_close_name(recipe)
    if not out:
        target.unlink(missing_ok=True)
        return out
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows([{**r, "Group": r[_GROUP_COL]} for r in out])
    return out


def write_exclusions(folder: Path, recipe: dict[str, Any]) -> int:
    dropped = [r for r in _read_csv(folder / ITEMS) if r.get("status") == "dropped"]
    lines = [f"# Dropped {recipe['item']}s", "", f"{len(dropped)} looked at and left off the list, with why.", "",
             "| Item | Group | Where | Why | Found at |", "|---|---|---|---|---|"]
    lines += [f"| {_md(r.get('name') or r.get('key'))} | {_md(r.get('group'))} | {_md(r.get('stage'))} | "
              f"{_md(r.get('reason'))} | {_md(r.get('source'))} |" for r in dropped]
    target = folder / exclusions_name(recipe)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(dropped)


_STAGE_WORDS = {"filter": "ruled out by code", "read": "ruled out after reading", "judge": "judged out",
                "check": "failed a check", "full": "group already full", "duplicate": "duplicates"}


def write_summary(folder: Path, recipe: dict[str, Any], rows: int, dropped: int,
                  finished: tuple[int, int, int], budget: int, stopped: str = "") -> str:
    """``summary.md``: how many, how short and why, how items went, how rows were
    checked, and where everything is. ``stopped``: why the run ended before its merge."""
    got = write_counts(folder, recipe)
    groups, target = got["groups"], got["target"]
    item = recipe["item"]
    lines = [f"# {recipe['output']}: summary", ""] + (
        [f"The run was stopped before it finished ({stopped}): this is what it had found by then.", ""]
        if stopped else []) + [
             f"{rows} {item}{'' if rows == 1 else 's'} on the list" + (f" of {target} wanted" if target else "")
             + ". Built by the list funnel: code searched, read and checked; a model judged each item once.", ""]
    short = shortfall(folder, recipe)
    if recipe["groups"]:
        lines += ["## Per group", "", "| Group | On the list | Wanted | Short |", "|---|---|---|---|"]
        for g, n in recipe["groups"].items():
            have = groups.get(g, {}).get("on_list", 0)
            lines.append(f"| {_md(g)} | {have} | {n or '-'} | {short.get(g, 0) or '-'} |")
        lines.append("")
    close = rows_in(folder / came_close_name(recipe))
    if short:
        why = (f"the run's {budget} searches were used" if got["searches"] >= budget
               else "the sources ran out of items that pass")
        lines += ["## Short", "", "; ".join(f"{g or 'the list'}: {n} more" for g, n in short.items())
                  + f". Why: {why}." + (f" {close} came close (`{came_close_name(recipe)}`)." if close else "")
                  + (" The charter accepts a short list with the reason given, so no rule was read loosely to fill "
                     "it: what meets a rule only loosely is among those that came close." if recipe.get("shortage_ok")
                     else ""), ""]
    loose = loose_on_list(folder)
    if loose:
        soft = loosenable(recipe)
        lines += ["## On a looser reading", "",
                  f"{len(loose)} row{'' if len(loose) == 1 else 's'} meet every rule that never loosens, but some "
                  "rules only by their looser reading (often: a fact the item doesn't publish), used because a group "
                  "ran short of rows that meet every rule as written:", ""]
        lines += [f"- {r['rule']} — looser: {looser_of(r)}" for r in soft] + ["", "| Item | Group | Why |", "|---|---|---|"]
        lines += [f"| {_md(r.get('name') or r.get('key'))} | {_md(r.get('group'))} | {_md(r.get('reason'))} |"
                  for r in loose]
        lines.append("")
    lines += ["## How items went", "", "| Group | Looked at | " + " | ".join(_STAGE_WORDS.values())
              + " | Maybe | Looser reading | Kept | Added by the finishing step |", "|---" * (len(_STAGE_WORDS) + 6) + "|"]
    for g, c in groups.items():
        lines.append(f"| {_md(g) or '-'} | {c['looked']} | " + " | ".join(str(c.get(st, 0)) for st in _STAGE_WORDS)
                     + f" | {c.get('maybe', 0)} | {c.get('loose', 0)} | {c.get('kept', 0)} | {c.get('agent', 0)} |")
    looked = [r for r in _read_csv(folder / ITEMS) if str(r.get("looked") or "").strip()]
    if looked:
        ended = {st: sum(1 for r in looked if r.get("status") == st) for st in ("kept", "loose", "maybe", "dropped")}
        lines += ["", f"Looked into, for a fact that wasn't in what was first read: {len(looked)} "
                  f"{item}{'' if len(looked) == 1 else 's'}. After it: {ended['kept'] + ended['loose']} on the list, "
                  f"{ended['maybe']} still close, {ended['dropped']} dropped. "
                  f"{sum(1 for r in looked if str(r.get('looked')).strip() == '0')} looks found nothing new to read."]
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
              "## Files", "", f"- `{recipe['output']}`: the list"]
    if close:
        lines.append(f"- `{came_close_name(recipe)}`: the {close} that came close, with why (not on the list)")
    lines += [f"- `{exclusions_name(recipe)}`: every {item} dropped, where and why ({dropped})",
              f"- `{ITEMS}`: everything looked at", f"- `{RECIPE}`: the recipe the run followed"]
    text = "\n".join(lines) + "\n"
    target = folder / summary_name(recipe)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return text


async def merge(run: dict, step: dict, folder: Path, recipe: dict[str, Any]) -> dict:
    """The list, from the funnel's rows and the finishing step's that pass its checks;
    ``exclusions.md``; ``summary.md``."""
    finished = await check_finish(run, step, folder, recipe)
    loosened = await asyncio.to_thread(fill_from_loose, folder, recipe) if loosen_on(run, recipe) else []
    out = await asyncio.to_thread(write_list, folder, recipe)
    close = await asyncio.to_thread(write_came_close, folder, recipe)
    dropped = await asyncio.to_thread(write_exclusions, folder, recipe)
    budget = search_budget(run, recipe)
    await asyncio.to_thread(write_summary, folder, recipe, len(out), dropped, finished, budget)
    await asyncio.to_thread(_test_report, folder)
    per = ""
    if recipe["groups"]:
        kept = kept_per_group(folder)
        per = " " + "; ".join(f"{g} {kept.get(g, 0)}" + (f" of {n}" if n else "") for g, n in recipe["groups"].items()) + "."
    added, refused, _ = finished
    text = (f"`{recipe['output']}`: {len(out)} rows" + (f" of {recipe['target']}" if recipe.get("target") else "")
            + f".{per}" + (f" The finishing step added {added} (and {refused} that failed a check)." if added or refused else "")
            + (f" {len(loosened)} meet the soft rules only by their looser reading (listed in `{summary_name(recipe)}`)."
               if loosened else "")
            + (f" `{came_close_name(recipe)}`: {len(close)} that came close, with why." if close else "")
            + f" `{exclusions_name(recipe)}`: {dropped} dropped, with why. `{summary_name(recipe)}`: how it went.")
    return {"ok": True, "output": text, "usd": 0.0, "tokens": 0, "error": ""}


def contract_for_checks(run: dict, rules: list[dict]) -> list[dict]:
    """The run's done-checks as Captain runs them on a funnel run's list: a table check
    that wants a column filled in every row doesn't ask it of a column a rule's own
    alternative lets be empty ("with no rating, judged good from what was read"), so a
    row the charter allows isn't failed for it. Any other run: the rules as they are."""
    if (run.get("config") or {}).get("engine") != ENGINE:
        return rules
    from captain_claw.flight_deck import bat_routes
    recipe = load_recipe(bat_routes._run_folder(run))
    optional = {c for r in (recipe or {}).get("rules") or [] if r.get("if_not_shown") for c in r.get("columns") or []}
    if not optional:
        return rules
    out = []
    for r in rules:
        check = r.get("check") or {}
        if check.get("type") == "table" and check.get("non_empty"):
            check = {**check, "non_empty": [c for c in check["non_empty"] if c not in optional]}
        out.append({**r, "check": check})
    return out


def _test_report(folder: Path) -> None:
    """A replayed run with expected answers scores itself (``test-report.md``)."""
    try:
        from captain_claw.flight_deck import funnel_eval
        funnel_eval.write_report(folder)
    except Exception:  # noqa: BLE001 — a report is extra, never the run's failure
        log.debug("funnel: couldn't write the test report", exc_info=True)


async def write_on_stop(run: dict, why: str) -> str:
    """A funnel run that ends before its merge (its owner stopped it, a cap ran out)
    still leaves its result: the list so far, what came close, the items dropped and
    the summary, written by code alone (rows the finishing step added aren't taken,
    their pages unread). A note saying so, or "" for a run with nothing to write."""
    if (run.get("config") or {}).get("engine") != ENGINE:
        return ""
    from captain_claw.flight_deck import bat_routes
    folder = bat_routes._run_folder(run)
    recipe = load_recipe(folder)
    if recipe is None or not (folder / ITEMS).is_file():
        return ""
    if loosen_on(run, recipe):
        await asyncio.to_thread(fill_from_loose, folder, recipe)
    out = await asyncio.to_thread(write_list, folder, recipe)
    close = await asyncio.to_thread(write_came_close, folder, recipe)
    dropped = await asyncio.to_thread(write_exclusions, folder, recipe)
    budget = search_budget(run, recipe)
    await asyncio.to_thread(write_summary, folder, recipe, len(out), dropped, (0, 0, 0), budget, why)
    await asyncio.to_thread(_test_report, folder)
    return (f"Stopped ({why}): what it found so far is written: `{recipe['output']}` {len(out)} rows"
            + (f", `{came_close_name(recipe)}` {len(close)}" if close else "")
            + f", `{exclusions_name(recipe)}` {dropped} dropped, `{summary_name(recipe)}`.")


def review_settled(run: dict, findings: str) -> str:
    """Whether the reviewers' "not done" settles a funnel run's list anyway: when the
    list is complete (no group short), or short where the charter accepts a short list
    with the reason given, and every row passed its checks by code, the run is done and
    their points go into the summary as open points. A note saying so, or "" when the
    list is still short and should be full (fix rounds may still add rows)."""
    if (run.get("config") or {}).get("engine") != ENGINE:
        return ""
    from captain_claw.flight_deck import bat_routes
    folder = bat_routes._run_folder(run)
    recipe = load_recipe(folder)
    short = shortfall(folder, recipe) if recipe is not None else {}
    if recipe is None or not _read_csv(folder / ROWS) or (short and not recipe.get("shortage_ok")):
        return ""  # still short, and the charter wants it full: fix rounds may still add rows
    loose = loose_on_list(folder)
    summary = folder / summary_name(recipe)
    state = ("complete" if not short else "short (" + "; ".join(f"{g or 'the list'} {n} more" for g, n in short.items())
             + f"), which the charter accepts with the reason given in {summary_name(recipe)},")
    try:
        text = summary.read_text(encoding="utf-8") if summary.is_file() else ""
        if "## Open points from the review" not in text:
            summary.write_text(text.rstrip("\n") + "\n\n## Open points from the review\n\n"
                               + f"The reviewers didn't pass the list as it stands. It is {state} and every row "
                               "passed the checks by code"
                               + (f"; {len(loose)} rows meet some rules only by their looser reading, which the "
                                  "factory allows" if loose else "")
                               + ", so the run ends here rather than sending agents to redo it. What they said:\n\n"
                               + (findings or "- (no reason given)") + "\n", encoding="utf-8")
    except OSError:
        log.debug("funnel: couldn't add the review's points to the summary", exc_info=True)
    return (f"The list is {state} and passes every check by code"
            + (f" ({len(loose)} rows on a looser reading the factory allows)" if loose else "")
            + f": the run is done; the reviewers' points are in {summary_name(recipe)} as open points.")


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
    loose = loose_on_list(folder)
    kept += len(loose)
    lines.append(f"It looked at {looked} items and kept {kept}; `{exclusions_name(recipe)}` says why each other one went. "
                 "Rules these checks don't cover are yours to judge.")
    if rows_in(folder / came_close_name(recipe)):
        lines.append(f"`{came_close_name(recipe)}` holds the items that came close, kept apart from the list on "
                     "purpose with why each didn't make it: it isn't part of the list, so don't judge the list by it.")
    if loose:
        lines += ["", f"{len(loose)} rows meet some rules only by their looser reading. The factory lets the run use "
                      "them when a group runs short of rows that meet every rule as written, so they aren't faults "
                      "to fix; their empty columns are the ones those rules fill:"]
        lines += [f"- {r['rule']} — looser: {looser_of(r)}" for r in loosenable(recipe)]
        lines.append("Those rows: " + ", ".join(r.get("name") or r.get("key", "") for r in loose[:40]) + ".")
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
    except Exception as e:  # noqa: BLE001 — what it did is kept: the next try picks up from there
        log.warning("funnel step stopped", run_id=run.get("id"), step=step.get("step_key"), error=str(e))
        return {"ok": False, "usd": 0.0, "tokens": 0,
                "error": f"the list funnel stopped on a call that broke ({_plain_error(e)}); its items so far "
                         "are kept and the next try picks up where it left off"}
    finally:
        await work.close()


def _plain_error(e: BaseException) -> str:
    """An error in a few words. A slow call reads as "no answer in time", never as
    "timed out": Bat takes that for a step too big for one turn and splits it into
    agent steps, which a funnel step isn't."""
    import httpx
    if isinstance(e, (httpx.TimeoutException, asyncio.TimeoutError)) or "timeout" in type(e).__name__.lower():
        return "a call got no answer in time"
    text = f"{type(e).__name__}: {e}"[:160]
    return re.sub(r"(?i)timed? ?out", "no answer in time", text)
