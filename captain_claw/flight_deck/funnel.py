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
SUMMARY = "summary.md"
EXCLUSIONS = "exclusions.md"
CLOSE_SUFFIX = "-came-close"  # <the list>-came-close.csv: the items that nearly made it, a result of its own
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
OUT_TOKENS = {"setup": 6000, "extract": 6000, "queries": 2000, "decide": 2500, "owner": 3000}
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
_MAYBE_COLS = ["key", "name", "group", "fields", "why", "sources", "row"]  # row: the cells the judging call filled
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
            recipe["filters"].append({"column": str(f["column"]).strip()[:80], "op": f["op"], "value": f.get("value")})
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
        rule["columns"] = [c for c in rule["columns"] if c not in fixed]
        if looser:
            rule["looser"] = looser
        out.append(rule)
        if len(out) >= RULES_MAX:
            break
    return out


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
    return [c for c in recipe["columns"] if c not in fixed and c not in ruled
            and (c in owner or any(st in c.lower() for st in stems))]


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
    has a rule to loosen."""
    on = str((run.get("config") or {}).get("funnel_loosen") or "short").lower() != "never"
    return on and (recipe is None or bool(loosenable(recipe)))


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


def _group_queries(group: str, place: str, size: re.Match | None, item: str = "") -> list[str]:
    """Searches for pages that list many of a group's items, any kind of item: lists,
    directories, databases, rankings, for each way to say the group. A company-size
    band adds a search for company pages with that band."""
    phrases = _group_phrases(group, item) or [group]
    qs = [q for t in ("list of {} {}", "best {} {}", "{} directory {}", "{} {}", "top {} {}", "{} database {}")
          for q in (t.format(ph, place) for ph in phrases)]
    if size:
        qs += [f'site:linkedin.com/company {ph} {place} "{size.group(1)}-{size.group(2)} employees"' for ph in phrases]
    return list(dict.fromkeys(" ".join(q.split()) for q in qs))[:24]


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
        words = _group_words(s["group"], (recipe.get("group_words") or {}).get(s["group"]))
        joined = " ".join(s["queries"]).lower()
        if (words and not any(w in joined for w in words)) or s["queries"] in seen_lists:
            s["queries"] = _group_queries(s["group"], place, size, recipe.get("item", ""))
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
    # "in the last 24 months" holds the list's one date column to that window, whatever
    # check the setup wrote for it (a format check let a two-year-old date through)
    window = re.search(r"\b(?:last|past|within|previous)\s+(\d{1,3})\s+(months?|years?)\b", text, re.I)
    dates = [c for c in recipe["columns"] if re.search(r"date", c, re.I)]
    if window and len(dates) == 1 and not any(x["column"] == dates[0] and x["op"] == "within_months"
                                              for x in recipe["checks"]):
        n = int(window.group(1)) * (12 if window.group(2).lower().startswith("year") else 1)
        recipe["checks"].append({"column": dates[0], "op": "within_months", "value": n})
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
    soft = loosenable(recipe)
    if recipe.get("rules"):
        fixed = [r["rule"] for r in recipe["rules"] if r not in soft]
        lines += ["", "## Rules", ""]
        if fixed:
            lines += ["Never loosened:"] + [f"- {r}" for r in fixed]
        if soft:
            lines += ["", "May be met by a looser reading, only when a group runs short of rows that meet every "
                          "rule as written (unshown is not broken; an item shown to break a rule is still out):"]
            lines += [f"- {r['rule']} — looser: {looser_of(r)}" for r in soft]
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
 "columns": [{"name": "<column>", "what": "<what it holds, where it comes from>",
              "from": "item|owner (owner: filled from the owner's own material, e.g. which of their products fits)"}],
 "key": "<the column no two rows share (a website, an email, an id)>",
 "site_column": "<the column holding the item's own website, if it has one>",
 "group_by": "<column the list is split by, or empty>", "groups": {"<group>": <how many>},
 "group_words": {"<group>": ["<words a page or search may use for it, any language>"]},
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
 "gather": [{"kind": "search", "query": "\\"{Name}\\" <where the facts the rules need are listed>", "count": 5},
            {"kind": "fetch", "url": "{Website}", "paths": ["", "<the item's pages that hold those facts>"]}],
 "extract": ["emails", "phones", "dates", "size_band", "us_state", "links"],
 "decide": "<the owner's rules for an item, short, in their words>",
 "rules": [{"rule": "<one of the owner's rules for an item>", "kind": "must|prefer",
            "fact": "identity|place|measured|contact|safety|other",
            "looser": "<prefer only: the loosest reading that still serves the goal>",
            "columns": ["<the columns this rule is about>"]}],
 "skip_sites": ["<sites that list or mention items of this kind but are never one's own site>"],
 "checks": [{"column": "<column>", "op": "required|in|regex|in_evidence|within_months|min|max",
             "value": ..., "source_column": "<column with the link it came from>"}],
 "decide_batch": <items per judging call: 1 when each item has pages to read, up to 20 for short rows>,
 "reference": ["<files that describe something the rows need (your product, rules, a price list), not items>"]}

Rules:
- Use the owner's own files and links first; they cost nothing to read. Search only when the list has to be
  found on the web, and then write 10-25 searches per group that find pages listing many of the right items
  (directories, catalogues, databases, rankings, "best of" and category pages), not one item each.
- Gather only what the checks and the rules need, from the pages this kind of item keeps them on (a firm's about,
  team and contact pages; a restaurant's menu and contact pages; a product's spec page; a person's profile), and
  put the cheapest test first: a company's size band from a search snippet ("size_max" filter) before reading.
  size_max and size_min count people (staff, employees) only, never anything else: a limit on locations,
  branches, seats, products or years is a rule, not a size filter.
- fact, for each rule: identity (what the item is: its kind, category, segment), place (where it is), measured (a
  count, amount, date or rating), contact (how to reach it), safety (what must never be done), other. Code reads
  these in any language: a measured or contact rule may be met loosely when rows run short; identity, place,
  safety and other never are.
- group_words: for each group, the words pages and searches use for it ("legal" → law firm, attorney, advokat).
- columns "from": "owner" for a column filled from the owner's own material (a reference file), never from the
  item's pages; every row fills it.
- Templates name columns in braces ({Firm}, {Website}) or {domain}.
- checks: "in_evidence" for every value copied from a source (an email, a date, a quote) with the column that
  holds its link; "within_months" for a date window; "in" for fixed values.
- An offline list (a file or documents only) has no searches and no gather unless the rules need the web.
- A file the goal points to for information (what the owner's product is, house rules) is "reference", never a
  source: no items are taken from it, and every judging call reads it.
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
                        why = ""  # an earlier try's trouble is past
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
            except httpx.TimeoutException:  # a slow page or search: this call fails, the step goes on
                return False, f"{tool} got no answer in time"
            except httpx.HTTPError as e:
                return False, f"{tool} failed: {type(e).__name__}"

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
  "row": {"<column>": "<value>"}, "loosened": ["<rule>"], "missing": ["<rule>"], "broken": ["<rule>"],
  "shown": "<the words read that break it>"}]
keep: it meets every rule as written and the row is filled from the facts. loose: it meets every rule that never
loosens, and each other rule as written or by its looser reading (a fact that isn't shown, with nothing saying
the item breaks the rule, is met by "not shown"); name those rules in "loosened" and fill the row. maybe: a fact a
rule that never loosens needs wasn't in what was read, or nothing read came from the item's own website; name in
"missing" every rule whose fact wasn't shown, and fill the row with what was. drop: what was read shows it breaks
a rule; name it in "broken" and copy into "shown" the words read that break it (a number, a date, a place), and
fill the row with what was shown. Not finding a fact is never "shown".''')


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
        self.same = SameItems(recipe, self.skip_sites)  # rows on the list or waiting, by their own site
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
                if op == "regex" and not re.search(str(val), text, re.I):
                    return f"{col} doesn't match"
                if op == "not_regex" and re.search(str(val), text, re.I):
                    return f"{'it' if col == '*' else col} matches what's excluded"
                if op == "in" and text and not any(_stems(text) & _stems(str(x)) or text.lower() == str(x).lower()
                                                   for x in val or []):
                    return f"{col} is {text}"  # it shares no word with any value it may take
                if op == "not_in" and text.lower() in {str(x).lower() for x in val or []}:
                    return f"{col} is {text}"
                if op in ("min", "max") and text:
                    span = number_span(text)
                    if span and ((op == "min" and span[1] < float(val)) or (op == "max" and span[0] > float(val))):
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
        got = await asyncio.gather(*(self.gather(it) for it in wave), return_exceptions=True)
        for it, err in zip(wave, got):
            if isinstance(err, Exception):
                log.warning("funnel: reading an item failed", run_id=self.run["id"], error=str(err))
                await self.record(it, "dropped", f"couldn't be read ({_plain_error(err)})", stage="read")
                continue
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
                    words, own_site = name_words(self._name_of(item)), fx.domain_of(self._site_of(item))
                    for r in fx.search_results(text):
                        # a result about something else ("LinkedIn News") is read, but
                        # code takes no size from it
                        own = names_item(words, r["url"], r["title"]) or bool(
                            own_site and fx.same_site(fx.domain_of(r["url"]), own_site))
                        evidence.append({"source": r["url"], "text": r["text"], "own": own})
            elif g["kind"] == "fetch":
                base = render(g["url"], item["fields"], site)
                if not base:
                    continue
                if not base.startswith(("http://", "https://")):
                    base = "https://" + base.lstrip("/")
                parts = urlsplit(base)
                root = f"{parts.scheme}://{parts.netloc}/"
                if any(fx.same_site(fx.domain_of(base), s) for s in self.skip_sites):
                    # the item's page on a directory or social site (its LinkedIn page, no
                    # site of its own found): that page only. The site's own about, news
                    # or contact pages aren't the item's (linkedin.com/news gave two firms
                    # another company's size).
                    urls = [base] if parts.path.strip("/") else []
                else:
                    urls = list(dict.fromkeys(base if not p else urljoin(root, p.lstrip("/")) for p in g["paths"]))
                if not urls:
                    continue
                tools = await self._tools()
                got = await asyncio.gather(*(tools.fetch(u) for u in urls))
                for u, (ok, text) in zip(urls, got):
                    await self._event("web_fetch", f"url={u}")
                    if ok and text.strip():
                        evidence.append({"source": u, "text": text[:FETCH_CHARS]})
            item["found"] = self.facts(item, evidence)

    def _own_text(self, item: dict) -> tuple[str, str]:
        """The item's own site and what was read on it (its pages, not results about it)."""
        domain = fx.domain_of(self._site_of(item))
        if not domain:
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
        where = fx.foreign_on_own_pages(text, domain, self.places)
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
        domain = fx.domain_of(next((v for k, v in item["fields"].items()
                                    if k.lower() in (site.lower(), "url", "website") and v), ""))
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
                  + (f"\nThis step looks for: {self.group}. An item that fits another of these groups instead is "
                     f"that group's find: keep it (or answer loose or maybe) with that group in "
                     f"{self.recipe['group_by']}, never drop it for not being {self.group}. Drop an item for its "
                     "kind only when it fits none of them; one that fits several takes the closest."
                     if self.group else "")) if self.recipe["groups"] else ""
        try:
            got = await self._ask("decide", DECIDE_LOOSE if self.loosen else DECIDE_SYSTEM,
                                  f"# Today\n{time.strftime('%Y-%m-%d')} (a date up to today is in the past, never "
                                  "\"in the future\")\n\n"
                                  f"# Rules\n{self.recipe['decide'] or '(take every item that fits the list)'}\n\n"
                                  + self._tiers_text()
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
            if i in verdicts and isinstance(verdicts[i].get("row"), dict):
                self._fill_from_facts(verdicts[i]["row"], it)
        await self._fill_owner_columns(verdicts, batch)
        for i, it in enumerate(batch, start=1):
            v = verdicts.get(i)
            if v is None:
                await self.record(it, "dropped", "the judging call gave no answer for it")
                continue
            verdict = str(v.get("verdict") or "").lower()
            reason = " ".join(str(v.get("reason") or "").split())[:200]
            if verdict == "loose" and not self.loosen:
                verdict = "maybe"  # no looser reading in this run: it's nearly there
            if verdict == "drop" and self.loosen and self._unbacked_drop(v, it):
                # it named only rules that may go unshown, and quoted nothing read that breaks them:
                # not finding a fact isn't breaking the rule
                v = {**v, "missing": list(v.get("missing") or []) + list(v.get("broken") or [])}
                verdict = "maybe"
            if verdict == "maybe" and self.loosen:
                verdict, loosened = self._maybe_or_loose(v)
            else:
                loosened = self._loosened(v.get("loosened")) if verdict == "loose" else []
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
                if named:  # a group written loosely ("Italian") is the list's group, written as the list does
                    row[self.recipe["group_by"]] = named
                elif self.recipe["group_by"] and self.group and not row.get(self.recipe["group_by"]):
                    row[self.recipe["group_by"]] = self.group  # no group said: this step's
                relax = {c for r in loosened for c in r.get("columns") or []}
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
                problem = check_row(self.recipe, row, {e["source"]: e["text"] for e in it.get("evidence") or []},
                                    relax=relax)
                if problem:
                    await self.record(it, "dropped", f"check: {problem}", stage="check")
                    continue
                group = self._group_named(row.get(self.recipe["group_by"], "") if self.recipe["group_by"] else "") \
                    or self.group
                if self.recipe["group_by"] and group in self.recipe["groups"]:
                    row[self.recipe["group_by"]] = group  # as the list names it
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
                await self._nearly(it, got, self._group_named(hint) or self.group,
                                   reason or "a fact it needs wasn't found")
            else:
                broke = [r for r in self._named(v.get("broken")) if not self._is_group_rule(r)]
                other = "" if broke else self._other_group(reason)  # a hotel's restaurant, a low rating: fail there too
                if other:  # "Italian, not Croatian": the Italian step judges it
                    await self._hand_over(it, other, reason)
                    continue
                await self.record(it, "dropped", reason or "breaks a rule")

    async def _nearly(self, it: dict, got: dict, group: str, reason: str) -> None:
        """An item that seems to fit but lacks something: a maybe, with the cells found
        so far (a value that isn't in what was read isn't passed on)."""
        await self.record(it, "maybe", reason, group=group)
        name = it["fields"].get("name") or next(iter(it["fields"].values()), "")
        cells = {c: " ".join(str((got or {}).get(c) or "").split()) for c in self.recipe["columns"]}
        if self.recipe["group_by"] and group in self.recipe["groups"]:
            cells[self.recipe["group_by"]] = group
        read = {e["source"]: e["text"] for e in it.get("evidence") or []}
        for c in self.recipe["checks"]:
            if c["op"] == "in_evidence" and cells.get(c["column"]) and check_row(
                    {**self.recipe, "checks": [c]}, cells, read):
                cells[c["column"]] = ""
        _append_csv(self.folder / MAYBE, _MAYBE_COLS,
                    {"key": it["key"], "name": name, "group": group,
                     "fields": json.dumps(it["fields"], ensure_ascii=False),
                     "why": reason, "sources": " ".join(e["source"] for e in it.get("evidence") or []),
                     "row": json.dumps({c: x for c, x in cells.items() if x}, ensure_ascii=False)})

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
                                           "Tie each value to something read about this item (what it does, who it "
                                           "serves), never a sentence that would fit any item. "
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

    def _is_group_rule(self, rule: dict[str, Any]) -> bool:
        """The rule that says which groups an item may be in ("cuisine is one of: ...")."""
        if self.recipe["group_by"] and self.recipe["group_by"] in (rule.get("columns") or []):
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

    def _tiers_text(self) -> str:
        """The rule tiers for a judging call, when rows may be on a looser reading."""
        if not self.loosen:
            return ""
        rules = self.recipe.get("rules") or []
        soft = loosenable(self.recipe)
        fixed = [r["rule"] for r in rules if r not in soft]
        lines = ["# Rule tiers"]
        if fixed:
            lines += ["Never loosened (what the item is, where it is, how to reach it): an item that breaks one is "
                      "dropped; one that doesn't show it is maybe:"] + [f"- {r}" for r in fixed]
        lines += ["May be met by a looser reading:"]
        lines += [f"- {r['rule']} — looser: {looser_of(r)}" for r in soft]
        lines += ['Verdict "loose": the item meets every never-loosened rule, and each other rule it doesn\'t meet as '
                  'written it meets by its looser reading. Unshown is not broken: a fact what was read simply doesn\'t '
                  'show, with nothing read saying the item breaks the rule, is met by "not shown". An item shown to '
                  'break a rule is still dropped. Name the rules met loosely in "loosened": ["<rule>", ...]; a column '
                  'the looser reading gives no value for stays empty. "keep" only when every rule is met as written.']
        return "\n".join(lines) + "\n\n"

    def _named(self, named: Any) -> list[dict[str, Any]]:
        said = [" ".join(str(x).split()).lower() for x in named or [] if str(x).strip()] \
            if isinstance(named, list) else []
        return [r for r in self.recipe.get("rules") or []
                if any(x and (x in r["rule"].lower() or r["rule"].lower() in x) for x in said)]

    def _unbacked_drop(self, v: dict, item: dict) -> bool:
        """A drop that shows no fact: every rule it says is broken may go unshown, and the
        words it quotes as proof (if any) aren't in what was read. An item shown to break
        a rule (a stated 120 people, a dated post, a foreign address) stays dropped, as
        does a drop that names no rule or one that never loosens."""
        broken = self._named(v.get("broken"))
        soft = loosenable(self.recipe)
        if not broken or any(r not in soft for r in broken):
            return False
        shown = " ".join(str(v.get("shown") or "").split())
        if not shown:
            return True
        read = "\n".join(e.get("text", "") for e in item.get("evidence") or [])
        return not fx.in_text(shown, read)

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
                     "at": time.strftime("%Y-%m-%d %H:%M")})
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
        used = [q for q in self.queries["used"] if q in src.get("queries", []) or q in worked]
        try:
            got = await self._new_queries_call(src, worked, used)
        except Exception as e:  # noqa: BLE001 — no new searches this round
            log.warning("funnel queries call failed", run_id=self.run["id"], error=str(e))
            return []
        fresh = got.get("queries") if isinstance(got, dict) else got if isinstance(got, list) else []
        out = [str(q).strip()[:300] for q in fresh or [] if str(q).strip() and str(q).strip() not in self.queries["used"]]
        if out:
            await self._event("funnel", f"{len(out)} new searches for {self.group or 'the list'}")
        return out[:15]

    async def _new_queries_call(self, src: dict, worked: dict[str, int], used: list[str]) -> Any:
        return await self._ask("queries", "Write new web searches for a list that is still short. Reply ONLY with "
                                         'JSON: {"queries": ["...", ...]} (10-15 new ones, none already used). '
                                         "Keep your thinking short.",
                              f"# Looking for\n{self.recipe['item']}s" + (f" ({self.group})" if self.group else "")
                              + (f": {src.get('about')}" if src.get("about") else "")
                              + f"\n\n# Rules\n{self.recipe['decide'][:1500]}\n\n# Searches that found kept items\n"
                              + ("\n".join(f"- {q} ({n})" for q, n in sorted(worked.items(), key=lambda x: -x[1])[:15])
                                 or "(none yet)")
                              + "\n\n# Already used\n" + "\n".join(f"- {q}" for q in used[-40:]))

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
        line = " ".join(raw.split()).strip(" ,;-")
        if not (8 <= len(line) <= 120) or not re.search(r"\d", line):
            continue
        here = fx.places_named(line)
        if any(code in codes and (not stems or fx.place_stem(w) in stems or w.lower() in stems) for code, w in here):
            if not best or len(line) < len(best):
                best = line
    return best


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
                    pages[url] = text[:FETCH_CHARS * 4] if ok else ""
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
        if group_by and group and not row.get(group_by):
            row[group_by] = group
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
    close = rows_in(folder / came_close_name(recipe))
    if short:
        why = (f"the run's {budget} searches were used" if got["searches"] >= budget
               else "the sources ran out of items that pass")
        lines += ["## Short", "", "; ".join(f"{g or 'the list'}: {n} more" for g, n in short.items())
                  + f". Why: {why}." + (f" {close} came close (`{came_close_name(recipe)}`)." if close else ""), ""]
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
    lines += [f"- `{EXCLUSIONS}`: every {item} dropped, where and why ({dropped})",
              f"- `{ITEMS}`: everything looked at", f"- `{RECIPE}`: the recipe the run followed"]
    text = "\n".join(lines) + "\n"
    (folder / SUMMARY).write_text(text, encoding="utf-8")
    return text


async def merge(run: dict, step: dict, folder: Path, recipe: dict[str, Any]) -> dict:
    """The list, from the funnel's rows and the finishing step's that pass its checks;
    ``exclusions.md``; ``summary.md``."""
    finished = await check_finish(run, step, folder, recipe)
    loosened = await asyncio.to_thread(fill_from_loose, folder, recipe) if loosen_on(run, recipe) else []
    out = await asyncio.to_thread(write_list, folder, recipe)
    close = await asyncio.to_thread(write_came_close, folder, recipe)
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
            + (f" {len(loosened)} meet the soft rules only by their looser reading (listed in `{SUMMARY}`)."
               if loosened else "")
            + (f" `{came_close_name(recipe)}`: {len(close)} that came close, with why." if close else "")
            + f" `{EXCLUSIONS}`: {dropped} dropped, with why. `{SUMMARY}`: how it went.")
    return {"ok": True, "output": text, "usd": 0.0, "tokens": 0, "error": ""}


def review_settled(run: dict, findings: str) -> str:
    """Whether the reviewers' "not done" settles a funnel run's list anyway: when the
    list is complete (no group short) and every row passed its checks by code, the run
    is done and their points go into ``summary.md`` as open points. A note saying so,
    or "" when the list is still short (fix rounds may still add rows)."""
    if (run.get("config") or {}).get("engine") != ENGINE:
        return ""
    from captain_claw.flight_deck import bat_routes
    folder = bat_routes._run_folder(run)
    recipe = load_recipe(folder)
    if recipe is None or not _read_csv(folder / ROWS) or shortfall(folder, recipe):
        return ""
    loose = loose_on_list(folder)
    summary = folder / SUMMARY
    try:
        text = summary.read_text(encoding="utf-8") if summary.is_file() else ""
        if "## Open points from the review" not in text:
            summary.write_text(text.rstrip("\n") + "\n\n## Open points from the review\n\n"
                               + "The reviewers didn't pass the list as it stands. It is complete and every row "
                               "passed the checks by code"
                               + (f"; {len(loose)} rows meet some rules only by their looser reading, which the "
                                  "factory allows" if loose else "")
                               + ", so the run ends here rather than sending agents to redo it. What they said:\n\n"
                               + (findings or "- (no reason given)") + "\n", encoding="utf-8")
    except OSError:
        log.debug("funnel: couldn't add the review's points to the summary", exc_info=True)
    return ("The list is complete and passes every check by code"
            + (f" ({len(loose)} rows on a looser reading the factory allows)" if loose else "")
            + ": the run is done; the reviewers' points are in summary.md as open points.")


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
    lines.append(f"It looked at {looked} items and kept {kept}; `{EXCLUSIONS}` says why each other one went. "
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
